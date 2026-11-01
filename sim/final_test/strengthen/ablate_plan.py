"""A3: plan-only screen of the Scene Memory ablation arms (Windows; no simulator).

    python sim/final_test/strengthen/gpu_lock.py run A -- python sim/final_test/strengthen/ablate_plan.py --planner gemma --ns <ns>
    python sim/final_test/strengthen/ablate_plan.py --planner opus --ns <ns>            (no GPU; hard 25 USD cap)

Reads docs/final-test/strengthen/A/blocks.jsonl (A2) and the namespace's capture / main plan journals; appends one fsync'd row per
(planner, arm, presentation) cell to docs/final-test/strengthen/A/plans.jsonl.

Rules (design.md section 6):
  * a request byte-identical to the captured full-memory request reuses the recorded G+M / O+M r0 plan, one identical to the no-memory
    request the recorded G0 / O0 r0 plan; only the `rep` control re-plans full-memory requests;
  * every other DISTINCT request is planned once per planner (one call per request sha), with plan.py's contract, provider, decoding,
    validator and repair turns (PlannerRunner, ft-v2, observed types), and shared by every arm that produced it;
  * response cache and ledger are private (sim/final_test/strengthen/work/A/); a resumed run replays cached answers (0 USD) and skips
    cells already journaled, so no paid call is ever repeated;
  * Opus: THESIS_ANTHROPIC_API_KEY is required before any call (never ANTHROPIC_API_KEY, as plan.py); the private ledger has a hard
    25 USD global ceiling, and before every call the committed spend plus that call's worst case (first call + 2 repairs at the
    provider's reserve, plan.cell_worst) must stay under it, else the run stops (exit 4) with nothing sent;
  * Gemma: only under the strengthen GPU lock held by task A; Ollama must answer at --ollama; the model is unloaded at the end;
  * the namespace's treatment identity must equal the one its capture ran with (the frozen-hash gate); a mismatch refuses the run
    (exit 3) unless --identity-decision "<who / what was decided>" is given, which is recorded in every row.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate_common as A          # noqa: E402

import common as C                 # noqa: E402
import plan as PL                  # noqa: E402
import treatment as T              # noqa: E402
from planners import PlannerRunner, ResponseCache, make_planner, prompt_sha256  # noqa: E402
from planners.base import CostCeilingExceeded, PlannerSkip  # noqa: E402
from planners.ledger import CostLedger  # noqa: E402

BLOCKS = os.path.join(A.DOCS, "blocks.jsonl")
PLANS = os.path.join(A.DOCS, "plans.jsonl")
FULL = {"G": "G+M", "O": "O+M"}
NONE = {"G": "G0", "O": "O0"}
SPEC = {"G": "gemma", "O": "opus"}
EXIT = {"ok": 0, "abort": 1, "skip": 2, "identity": 3, "cap": 4}


def cell_key(ns, planner, arm, pres) -> str:
    return f"A|{ns}|{planner}|{arm}|{pres}"


def opus_spent() -> float:
    p = os.path.join(A.WORK, "opus_ledger.db")
    return CostLedger(p, global_ceiling_usd=A.OPUS_CAP_USD).spent() if os.path.exists(p) else 0.0


class Screen:
    def __init__(self, ns: str, planner: str, arms=None, ollama: str = "http://127.0.0.1:11434", identity_decision: str | None = None,
                 out=print):
        self.ns, self.pl, self.out = ns, planner, out
        self.cap = A.load_capture(ns)
        self.main = A.main_plans(ns)
        self.types, _ = A.load_types(ns)
        self.contract = PL.make_contract(self.types, ns)
        allowed = A.arms_for(ns) if planner == "G" else A.opus_arms_for(ns)
        self.arms = [a for a in (arms or allowed) if a in allowed]
        self.blocks = {(r["arm"], r["pres"]): r for r in A.read_jsonl(BLOCKS) if r.get("ns") == ns}
        miss = [(a, p) for a in self.arms for p in self.cap if (a, p) not in self.blocks]
        if miss:
            raise SystemExit(f"blocks.jsonl lacks {len(miss)} rows for {ns} (first {miss[0]}): run A2 first")
        bad = [k for k in self.blocks if k[0] in self.arms and not self.blocks[k]["checks"]["ok"]]
        if bad:
            raise SystemExit(f"{len(bad)} blocks failed their checks (first {bad[0]})")
        self.ident = self._identity(identity_decision)
        self.ollama = ollama
        self.secrets = ()
        self.prov = self.runner = None
        self.rows = A.read_jsonl(PLANS)
        self.done = {r["key"] for r in self.rows}
        self.calls = {}                                             # request sha -> the cell row that made the call (this planner)
        for r in self.rows:
            if r.get("ns") == ns and r.get("planner") == planner and r.get("source") == "call":
                self.calls[r["request_sha"]] = r

    def _identity(self, decision):
        now = T.identity(self.ns, None)["sha"]
        cap_ids = {r.get("treatment_id") for r in self.cap.values()}
        rec = {"identity_now": now, "identity_capture": sorted(cap_ids), "match": cap_ids == {now}, "decision": decision}
        if not rec["match"] and not decision:
            raise SystemExit(f"IDENTITY: {self.ns} treatment identity is {now[:12]} but its capture ran {sorted(x[:12] for x in cap_ids)} "
                             "(code changed after the test). Refused; pass --identity-decision \"<Arash's decision>\" to proceed.")
        return rec

    # ---------------------------------------------------------------- provider
    def open(self):
        cache = ResponseCache(os.path.join(A.WORK, f"cache_{SPEC[self.pl]}"), namespace=f"strengthen-A|{self.ns}")
        if self.pl == "O":
            self.secrets = (C.thesis_key().reveal(),)                # PlannerSkip -> clean exit 2, nothing sent
            self.ledger = CostLedger(os.path.join(A.WORK, "opus_ledger.db"), global_ceiling_usd=A.OPUS_CAP_USD)
            self.prov = make_planner("opus", cache=cache, ledger=self.ledger)
        else:
            self.ledger = CostLedger(os.path.join(A.WORK, "gemma_ledger.db"))
            self.prov = make_planner("gemma", cache=cache, ledger=self.ledger, endpoint=self.ollama)
            ctl = PL.OllamaCtl(self.ollama, allow_start=False)
            self.prov.check_available = ctl.ensure
            self.ctl = ctl
            ctl.ensure()
            self.env = self.prov.environment()
        self.runner = PlannerRunner(self.prov, self.contract)
        self.dec = self.prov.decoding()

    def close(self):
        if self.pl == "G" and getattr(self, "ctl", None) is not None:
            left = self.ctl.unload(self.prov.model)
            self.out(f"gemma unloaded; still loaded: {left}")

    # ---------------------------------------------------------------- one cell
    def _row(self, arm, p, blk, *, source, res=None, base=None, call_of=None, wall=0.0, note=""):
        rec = self.cap[p]
        full = self.main.get((FULL[self.pl], p))
        none = self.main.get((NONE[self.pl], p))
        if res is not None:
            status = "ok" if res["ok"] else ("infra" if res["infra_failure"] else "rejected")
            plan, first = (C.norm_plan(res["plan"]) if res["ok"] else []), C.norm_plan(res["first_plan"])
            atts = [{"k": a["k"], "stage": a["result"].get("stage"), "tokens_in": a["result"]["tokens_in"], "tokens_out": a["result"]["tokens_out"],
                     "cost_usd": a["result"]["cost_usd"], "latency_s": a["result"]["latency_s"], "cached": a["result"]["cached"],
                     "original_cost_usd": a["result"].get("original_cost_usd"), "stop_reason": a["result"].get("stop_reason"),
                     "error": a["result"].get("error"), "error_kind": a["result"].get("error_kind"), "raw": a["result"].get("raw"),
                     "valid": a["validation"]["ok"], "errors": a["validation"].get("errors"), "prompt_sha256": a["result"].get("prompt_sha256"),
                     "served_model": (a["result"].get("extra") or {}).get("served_model"),
                     "request_id": (a["result"].get("extra") or {}).get("request_id")} for a in res["attempts"]]
            ex = {"first_ok": res["first_ok"], "repaired": res["repaired"], "calls": res["calls"], "cached_calls": res["cached_calls"],
                  "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"], "cost_usd": res["cost_usd"], "attempts": atts,
                  "say": res.get("say", ""), "codes_by_attempt": res.get("codes_by_attempt"), "model": self.prov.model, "decoding": self.dec}
        else:
            status, plan, first = base["status"], C.norm_plan(base.get("plan") or []), C.norm_plan(base.get("first_plan") or [])
            ex = {"first_ok": base.get("first_ok"), "repaired": base.get("repaired"), "calls": 0, "cached_calls": 0, "tokens_in": 0,
                  "tokens_out": 0, "cost_usd": 0.0, "attempts": [], "model": base.get("model"), "reused_key": base["key"],
                  "reused_cost_usd": base.get("cost_usd", 0.0)}
        fplan = C.norm_plan(full["plan"]) if full and full["status"] == "ok" else None
        nplan = C.norm_plan(none["plan"]) if none and none["status"] == "ok" else None
        mine = plan if status == "ok" else None
        row = {"schema": "strengthen.A.plan/1", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": cell_key(self.ns, self.pl, arm, p),
               "ns": self.ns, "scene": A.KITCHENS[self.ns], "planner": self.pl, "arm": arm, "pres": p, "item": rec["item"], "lang": rec["lang"],
               "level": rec["level"], "request_sha": blk["request_sha"], "block_sha": blk["text_sha"], "same_request_as": blk["same_request_as"],
               "source": source, "call_of": call_of, "status": status, "plan": plan, "first_plan": first,
               "full_key": full["key"] if full else None, "none_key": none["key"] if none else None,
               "differs_from_full": not (status == "ok" and full is not None and full["status"] == "ok" and mine == fplan),
               "equals_none_plan": bool(mine is not None and nplan is not None and mine == nplan),
               "identity": self.ident, "wall_s": round(wall, 3), "note": note, **ex}
        hits = C.leak_paths(row, self.secrets)
        if hits:
            raise SystemExit(f"key-like content in the row of {row['key']} ({hits}); not written")
        return row

    def cell(self, arm: str, p: str):
        k = cell_key(self.ns, self.pl, arm, p)
        if k in self.done:
            return None
        blk = self.blocks[(arm, p)]
        rec = self.cap[p]
        if arm != "rep" and blk["same_request_as"] in ("S*", "none"):
            src = FULL[self.pl] if blk["same_request_as"] == "S*" else NONE[self.pl]
            base = self.main[(src, p)]
            if base["request_sha"] != blk["request_sha"]:
                raise SystemExit(f"{k}: the reused {src} row was planned from another request")
            row = self._row(arm, p, blk, source="reuse_" + ("full" if src == FULL[self.pl] else "none"), base=base)
        elif blk["request_sha"] in self.calls:
            first = self.calls[blk["request_sha"]]
            row = self._row(arm, p, blk, source="shared_call", base={**first, "key": first["key"]}, call_of=first["key"])
        else:
            msgs = self.contract.messages(rec["instruction"], PL.summary_of(rec), rec.get("holding"), memory_block=blk["text"] or None)
            if prompt_sha256(msgs) != blk["request_sha"]:
                raise SystemExit(f"{k}: the re-rendered request differs from blocks.jsonl")
            if self.pl == "O":
                worst = PL.cell_worst(self.prov, self.contract, msgs)
                spent = self.ledger.spent()
                if spent + worst > A.OPUS_CAP_USD:
                    raise CostCeilingExceeded(f"cap: committed {spent:.4f} + worst case {worst:.4f} > {A.OPUS_CAP_USD} USD")
            t0 = time.time()
            res = self.runner.plan(rec["instruction"], PL.summary_of(rec), rec.get("holding"), memory_block=blk["text"] or None,
                                   fewshot_block=None, replicate=0, observed_types=set(rec.get("observed_types") or []))
            if res["attempts"][0]["result"].get("prompt_sha256") != blk["request_sha"]:
                raise SystemExit(f"{k}: the provider sent a different prompt than the rendered one")
            row = self._row(arm, p, blk, source="call", res=res, wall=time.time() - t0)
            self.calls[blk["request_sha"]] = row
        A.append_jsonl(PLANS, row)
        self.done.add(k)
        self.rows.append(row)
        return row

    def run(self) -> int:
        order = sorted(self.cap, key=lambda p: A.sha256_text(f"{A.SEED}|A3|{self.ns}|{p}"))
        n = 0
        t0 = time.time()
        for i, p in enumerate(order):
            for arm in self.arms:
                r = self.cell(arm, p)
                if r is not None:
                    n += 1
                    if r["source"] == "call":
                        self.out(f"  {arm:<22s} {p:<44s} {r['status']:<8s} calls={r['calls']}({r['cached_calls']} cached) usd={r['cost_usd']:.4f} "
                                 f"differs={r['differs_from_full']}")
            if (i + 1) % 10 == 0 or i + 1 == len(order):
                self.checkpoint(i + 1, len(order), n, time.time() - t0)
        return EXIT["ok"]

    def checkpoint(self, i, total, n, secs):
        mine = [r for r in self.rows if r.get("ns") == self.ns and r.get("planner") == self.pl]
        spent = opus_spent()
        cmd = (("python sim/final_test/strengthen/gpu_lock.py run A -- " if self.pl == "G" else "") +
               f"python sim/final_test/strengthen/ablate_plan.py --planner {SPEC[self.pl]} --ns {self.ns}" +
               (f" --identity-decision \"{self.ident['decision']}\"" if self.ident.get("decision") else ""))
        A.update_resume(f"A3 planning {SPEC[self.pl]} {self.ns}: {i}/{total} presentations visited",
                        [f"A2 blocks.jsonl written", f"A3 {SPEC[self.pl]} {self.ns}: {len(mine)} cells journaled "
                         f"({sum(1 for r in mine if r['source'] == 'call')} calls) in A/plans.jsonl"],
                        spent, cmd, ["plans.jsonl is append-only and fsync'd per cell; a resume skips journaled cells and replays the "
                                     "private response cache, so no Opus call is paid twice."])
        self.out(f"[checkpoint] {i}/{total} presentations, {n} new cells, {secs:.0f} s, opus committed {spent:.4f} USD")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--planner", required=True, choices=("gemma", "opus"))
    ap.add_argument("--ns", required=True, choices=sorted(A.KITCHENS))
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--identity-decision", default=None)
    ap.add_argument("--identity-decision-file", default=None, help="a JSON file with {'decision': '...'} (A/decision.json)")
    ap.add_argument("--dry-run", action="store_true", help="count what would be planned; no provider is opened")
    ap.add_argument("--steal-lock", action="store_true", help="remove a stale ablate_plan.lock left by a killed run")
    a = ap.parse_args(argv)
    if a.identity_decision_file and not a.identity_decision:
        with open(a.identity_decision_file, encoding="utf-8") as f:
            a.identity_decision = json.load(f)["decision"]
    pl = "G" if a.planner == "gemma" else "O"
    if pl == "G" and not a.dry_run:
        import ablate_blocks as AB
        if not AB.lock_held_by_a():
            print("refused: Gemma planning runs only under `gpu_lock.py run A -- ...`", file=sys.stderr)
            return EXIT["skip"]
    try:
        s = Screen(a.ns, pl, a.arms, a.ollama, a.identity_decision)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return EXIT["identity"] if "IDENTITY" in str(e) else EXIT["abort"]
    if a.dry_run:
        todo = {}
        seen = set(s.calls)
        for p in s.cap:
            for arm in s.arms:
                b = s.blocks[(arm, p)]
                if cell_key(s.ns, pl, arm, p) in s.done:
                    continue
                kind = "reuse" if (arm != "rep" and b["same_request_as"]) else ("shared" if b["request_sha"] in seen else "call")
                if kind == "call":
                    seen.add(b["request_sha"])
                todo.setdefault(arm, {}).setdefault(kind, 0)
                todo[arm][kind] += 1
        print(json.dumps({"ns": s.ns, "planner": a.planner, "identity": s.ident, "todo": todo}, indent=1))
        return 0
    # one planning process at a time: plans.jsonl has a single writer (a stale lock of a killed run is removed with --steal-lock)
    lk = os.path.join(A.WORK, "ablate_plan.lock")
    os.makedirs(A.WORK, exist_ok=True)
    if a.steal_lock and os.path.exists(lk):
        os.remove(lk)
    try:
        fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"LOCKED: another ablate_plan run holds {lk}: {open(lk, encoding='utf-8').read()}", file=sys.stderr)
        return EXIT["skip"]
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "planner": a.planner, "ns": a.ns, "start": time.strftime("%Y-%m-%d %H:%M:%S")}))
    try:
        try:
            s.open()
        except PlannerSkip as e:
            print(f"SKIP: {e}", file=sys.stderr)
            return EXIT["skip"]
        try:
            return s.run()
        except CostCeilingExceeded as e:
            print(f"STOP at the Opus cap: {e}", file=sys.stderr)
            return EXIT["cap"]
        finally:
            s.close()
    finally:
        os.remove(lk)


if __name__ == "__main__":
    sys.exit(main())
