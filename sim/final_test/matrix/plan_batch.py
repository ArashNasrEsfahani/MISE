"""Phase B planning through the Anthropic Message Batches API (stage BATCH-A29, 2026-10-13; Arash: "Batch API for all kitchens").

    python sim/final_test/matrix/plan.py --namespace NS --pass core --r 0 --batch [--batch-hold-usd 10] [--batch-max-cells 200] [--batch-poll-s 30]
    python sim/final_test/matrix/planfirst.py --namespace NS --batch ...          (the Anthropic stages run in batch mode, Gemma / Codex unchanged)

`BatchPlan` is `plan.Plan` with ONE change: the cells of a CHUNK (whole presentation blocks, in the registered seeded order) are planned
together in rounds of batches instead of one synchronous call at a time. Everything else is the Plan code path itself: the request-identity
and masking gates, the treatment check, `_call_cell` (prompt re-render + sha check), `_record` (the same plan journal row), the cost guard at
block boundaries, `_finish`. Only Anthropic cells are batchable; a selection with a Gemma / Codex cell is refused (plan those synchronously).

Rounds: round 0 = every cell's first call; round k = the k-th repair of the cells still invalid (identical repair frames: `PlannerRunner` builds
them, `planners.anthropic_batch` explains how); resubmissions of unbilled results (expired / canceled / retryable errored) ride in later
rounds, at most 3 times, then the cell is `infra` (re-queued in a later session, s13).

Cost guard:
  * at SUBMISSION: the chunk's worst case (every cell: first call + two repairs, at the batch price) is held as one block per scope BEFORE the
    first batch (BlockRefused -> exit 4, nothing sent); every request is then reserved at its own pre-call reserve (batch price) before its
    batch is created;
  * at RESULT: each succeeded request settles at its billed batch cost; after every chunk `_boundary` runs the registered guard (A15 / A19 /
    A21 B: mean per billed Opus cell, projected completion vs the core / total ceilings, ledger-vs-journal reconciliation) on ACTUAL spend.
Chunks are bounded by `batch_hold_usd` (the hold) and `batch_max_cells`, so a pause still falls between presentations (arms balanced).

Resume: every batch is recorded (`batch_submitted`, `batch_collected` events). A start finds submitted-not-collected batches and re-attaches
to them (polls, collects, settles) before anything new is sent; answers collected for cells that have no plan row yet are served as billed
answers (not replays), so the journal and the ledger agree. Open reservations that belong to no batch quarantine their cells (exit 7) exactly
as in the parallel path.
"""
from __future__ import annotations

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402
import plan as PL                                                        # noqa: E402

from planners import ARMS as PLANNER_ARMS                                 # noqa: E402
from planners.base import CostCeilingExceeded                            # noqa: E402
from planners.anthropic_batch import (BATCH_DISCOUNT, UNBILLED_TYPES, BatchClient, BatchDriver, BatchPlanner,   # noqa: E402
                                      BatchResultMissing, BatchSubmitAmbiguous, NeedCall)

BATCHABLE = ("opus", "sonnet", "haiku")
DEFAULT_HOLD_USD = 10.0
DEFAULT_MAX_CELLS = 200


def batch_factory(spec: str, **kw):
    if spec not in BATCHABLE:
        raise ValueError(f"batch mode plans Anthropic cells only; {spec!r} must be planned without --batch")
    fam, model = PLANNER_ARMS[spec]
    return BatchPlanner(model, **kw)


class BatchPlan(PL.Plan):
    def __init__(self, ns: str, *, batch_client=None, batch_poll_s: float = 30.0, batch_hold_usd: float = DEFAULT_HOLD_USD,
                 batch_max_cells: int = DEFAULT_MAX_CELLS, batch_sleep=None, batch_now=None, provider_factory=None, **kw):
        kw.pop("parallel", None)
        kw.pop("codex_parallel", None)
        super().__init__(ns, parallel=1, provider_factory=provider_factory or batch_factory, **kw)
        self.batch_client = batch_client
        self.batch_poll_s, self.batch_hold_usd, self.batch_max_cells = float(batch_poll_s), float(batch_hold_usd), int(batch_max_cells)
        self._bsleep, self._bnow = batch_sleep, batch_now
        self.driver = None

    # ---------------------------------------------------------------------------------------- setup
    def _prepare(self):
        super()._prepare()
        bad = sorted({c["planner"] for c in self.sel if c["planner"] not in ("O", "S", "H")})
        if bad:
            raise ValueError(f"--batch plans Anthropic cells only; the selection holds planner(s) {bad} (plan those passes without --batch)")

    def _open(self):
        super()._open()
        import time as _t
        client = self.batch_client or BatchClient(self.key_getter)
        self.driver = BatchDriver(client, self.ledger, journal=lambda event, **kw: self.ev(event, **kw), poll_s=self.batch_poll_s,
                                  sleep=self._bsleep or _t.sleep, now=self._bnow or _t.time, out=self.say)
        self.ev("batch_mode", discount=BATCH_DISCOUNT, hold_usd=self.batch_hold_usd, max_cells=self.batch_max_cells,
                poll_s=self.batch_poll_s, note="same request body and decoding as the synchronous path; billing at the batch price")

    def _cells_by_key(self) -> dict:
        return {c["key"]: c for c in self.all_cells}

    def _prov_for_cell(self, cell_key: str):
        c = self._cells_by_key().get(cell_key)
        return self._runner(c["planner"])[0] if c else None

    # ---------------------------------------------------------------------------------------- resume
    def _inflight(self) -> list:
        evs = self.E.rows()
        got = {e.get("batch_id") for e in evs if e.get("event") == "batch_collected"}
        return [e for e in evs if e.get("event") == "batch_submitted" and e.get("batch_id") not in got
                and (e.get("pass_id") or "") == self.pass_id]

    def _resume_inflight(self):
        """Re-attach to batches a killed run submitted; then load answers collected earlier for cells without a plan row. Those cells are
        finished FIRST (`_loop`), before the first guard boundary, so the ledger (already settled) and the journal agree again."""
        self._resume_cells = set()
        for rec in self._inflight():
            provs = {}
            for it in rec["items"]:
                p = self._prov_for_cell(it["cell"])
                if p is not None:
                    provs[it["cache_key"]] = p
            self.ev("batch_reattach", batch_id=rec["batch_id"], n=len(rec["items"]))
            try:
                self.driver.collect(rec, provs)
            except BatchResultMissing as e:
                raise PL.PlanAbort("abort_batch_result_missing", f"{e}; reservations stay open (reconcile before any resubmission)") from None
            self._resume_cells.update(it["cell"] for it in rec["items"])
        evs = self.E.rows()
        collected = {e.get("batch_id"): e for e in evs if e.get("event") == "batch_collected"}
        n = 0
        for e in evs:
            if e.get("event") != "batch_submitted" or e.get("batch_id") not in collected or (e.get("pass_id") or "") != self.pass_id:
                continue
            types = collected[e["batch_id"]].get("types") or {}
            for it in e["items"]:
                row = self.latest.get(it["cell"])
                if row is not None and row.get("status") != "ceiling":
                    continue                                       # the cell has its row: its answers are ordinary replays now
                if types.get(it["custom_id"]) != "succeeded":
                    continue
                p = self._prov_for_cell(it["cell"])
                if p is None or it["cache_key"] in p.fresh:
                    continue
                hit = p.cache.get(it["cache_key"])
                if hit is not None:
                    p.fresh[it["cache_key"]] = hit
                    self._resume_cells.add(it["cell"])
                    n += 1
        if n:
            self.ev("batch_answers_restored", n=n)
        # Astra A29 6: the unbilled-retry allowance survives a restart: every earlier unbilled result of a cell without a terminal row counts
        for prov, _ in self._provs.values():
            prov.failures.clear()                                   # recomputed from the journal (the re-attach above already counted in memory)
        r = 0
        for e in evs:
            if e.get("event") != "batch_submitted" or e.get("batch_id") not in collected or (e.get("pass_id") or "") != self.pass_id:
                continue
            types = collected[e["batch_id"]].get("types") or {}
            for it in e["items"]:
                row = self.latest.get(it["cell"])
                if (row is not None and row.get("status") != "ceiling") or types.get(it["custom_id"]) not in UNBILLED_TYPES:
                    continue
                p = self._prov_for_cell(it["cell"])
                if p is not None:
                    p.failures[it["cache_key"]] = p.failures.get(it["cache_key"], 0) + 1
                    r += 1
        if r:
            self.ev("batch_retry_counts_restored", n=r)

    def _quarantine_check(self):
        """Batch mode: first re-attach to in-flight batches, then quarantine cells whose open reservations belong to no batch."""
        self._resume_inflight()
        if not any(c["scope"] != "free" for c in self.sel):
            return
        rows = [r for r in self.ledger.open_reservations()["rows"] if r["reserved"] > 0]
        if not rows:
            return
        cells = sorted({r.get("cell") or "?" for r in rows})
        usd = round(sum(r["reserved"] for r in rows), 6)
        if self.allow_replacement:
            self.ev("replacement_attempts_allowed", n=len(rows), reserved_usd=usd, cells=cells[:50], by="--allow-replacement")
            return
        if "?" in cells:
            raise PL.PlanAbort("abort_open_reservations", f"{len(rows)} open paid reservation(s) ({usd:.4f} USD) carry no cell: reconcile "
                                                          "them (--reconcile-open) or allow replacement attempts (--allow-replacement)")
        self._quarantined = set(cells)
        self.ev("quarantine", n=len(rows), reserved_usd=usd, cells=cells[:50])
        self.say(f"QUARANTINE: {len(rows)} open reservation(s) of {len(cells)} cell(s)")

    # ---------------------------------------------------------------------------------------- the loop
    def _chunks(self) -> list:
        """Whole presentation blocks in the registered order, grouped until the hold or the cell cap would be passed."""
        out, cur, cur_usd, cur_n = [], [], 0.0, 0
        for pres, cells in self._order():
            todo = [c for c in cells if PL.classify(c, self.latest, self.requeue) == "pending"]
            if self._quarantined:
                held = [c for c in todo if ((f"{self.pass_id}:" if self.pass_id else "") + c["key"]) in self._quarantined]
                self._quarantine_skipped += len(held)
                todo = [c for c in todo if c not in held]
            if not todo:
                continue
            w = self._worst(todo)
            if cur and (cur_usd + w > self.batch_hold_usd + 1e-12 or cur_n + len(todo) > self.batch_max_cells):
                out.append(cur)
                cur, cur_usd, cur_n = [], 0.0, 0
            cur.append((pres, todo))
            cur_usd, cur_n = cur_usd + w, cur_n + len(todo)
        if cur:
            out.append(cur)
        return out

    def _run_chunk(self, chunk: list, idx: int):
        cells = [c for _, todo in chunk for c in todo]
        for scope in ("core", "other"):
            cs = [c for c in cells if c["scope"] == scope]
            if not cs:
                continue
            worst = self._worst(cs)
            with contextlib.ExitStack() as st:
                try:
                    st.enter_context(self.ledger.block(scope, worst))
                except C.BlockRefused as e:
                    self.ev("block_refused", pres=chunk[0][0], scope=scope, cells=len(cs), worst_usd=worst, detail=str(e)[:300],
                            committed=self.ledger.committed(scope), total=self.ledger.committed(), batch_chunk=idx)
                    raise PL._Stop(PL.EXIT["ceiling"], info={"pres": chunk[0][0]}) from None
                units = []
                for c in cs:
                    units.append({"id": c["key"], "provider": self._runner(c["planner"])[0], "scope": scope,
                                  "cell_tag": (f"{self.pass_id}:" if self.pass_id else "") + c["key"],
                                  "run": (lambda c=c: self._call_cell(c, self.cap[c["pres"]]))})
                try:
                    self.driver.run(units, tag=f"chunk{idx}:{scope}")
                except CostCeilingExceeded as e:
                    self.ev("ceiling", scope=scope, detail=str(e)[:300], batch_chunk=idx)
                    raise PL._Stop(PL.EXIT["ceiling"], info={"pres": chunk[0][0]}) from None
                except BatchResultMissing as e:
                    raise PL.PlanAbort("abort_batch_result_missing", f"{e}; the reservations stay open (reconcile with the provider's batch "
                                                                     "results before any resubmission)") from None
                except BatchSubmitAmbiguous as e:
                    raise PL.PlanAbort("abort_batch_submit_ambiguous", f"batch creation may have happened without an answer: {e}; the "
                                                                       "reservations stay open (reconcile with the provider's batch list)") from None
                finally:
                    self.ledger.cell_tag = None

    def _loop(self) -> int:
        resume = sorted(getattr(self, "_resume_cells", set()))
        if resume:                                               # answers already billed: record those cells before the guard reconciles
            want = set(resume)
            chunk = [(p, [c for c in cs if c["key"] in want]) for p, cs in self._order()]
            chunk = [(p, [c for c in cs if PL.classify(c, self.latest, self.requeue) == "pending"]) for p, cs in chunk]
            chunk = [(p, cs) for p, cs in chunk if cs]
            if chunk:
                self.ev("batch_resume_chunk", cells=sum(len(cs) for _, cs in chunk))
                self._run_chunk(chunk, -1)
            self._resume_cells = set()
        self._boundary(None)
        for i, chunk in enumerate(self._chunks()):
            self._run_chunk(chunk, i)
            self._boundary(chunk[-1][0])                         # the registered guard on ACTUAL spend, at a presentation boundary
            if self.interrupt_after is not None and self.recorded >= self.interrupt_after:
                self.ev("injected_interrupt", after=self.recorded)
                raise PL._Stop(PL.EXIT["interrupt"])
        return self._finish()

    def _finish(self) -> int:
        if self.driver is not None:
            s = dict(self.driver.stats)
            w = s.pop("batch_wall_s", [])
            self.ev("batch_report", **s, batch_wall_s_max=max(w) if w else None, batch_wall_s_sum=round(sum(w), 1) if w else 0.0)
        return super()._finish()


__all__ = ["BatchPlan", "batch_factory", "NeedCall"]
