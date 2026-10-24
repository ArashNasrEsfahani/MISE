"""K23 CODEX ADDENDUM: the Codex planner models on the Kitchen 2 / 3 test namespaces, AFTER their main four-arm tests.

Arash's decision (2026-10-21 ~01:45, docs/final-test/progress.md; record docs/final-test/k23/decisions/K23-CODEX-ADDENDUM.json): "unfreeze and run
the codex models after the main testing was done; do not repeat the executions with exactly the same planning; keep the Gemma and Opus plannings".
This is an ADDITION outside the registered K2 / K3 design (A29 item 1: the four core arms). It is decided before any K2 / K3 outcome was seen and is
reported as an addendum, never mixed into the main analysis.

Design (mirrors Kitchen 1, where gpt-6-sol was the main executed comparator and the other Codex models were plan-only):
  arms  sol      X0 / X+M (gpt-6-sol) at r = 0 on every canonical presentation      -> executed later (execute phase, dedupe)
        explore  X-<m>0 / X-<m>+M for gpt-6-astra, gpt-6-luna, gpt-5.6-terra at r = 0 -> plan-only (as in Kitchen 1)
  The requests are the main test's captured requests (capture.jsonl, read-only; sources 'none' and 'S*', the same bytes G0 / O0 and G+M / O+M got);
  the Codex CLI configuration is Kitchen 1's (codex-cli 0.162.1 pinned, read-only sandbox, restricted network, effort high, identity-checked).

Isolation (nothing of the main test is touched):
  writes only  docs/final-test/k23/codex/<ns>/{plan.<pass>.jsonl, events.jsonl}  and  sim/final_test/matrix/work/codex-<ns>/<pass>/{cache, ledger.db}
  reads        the main capture.jsonl and the main events.jsonl (capture phase_complete check) of <ns>, read-only
  The X arms are built here (C.arm_info without the namespace: the K23 arm guard of common.py is not edited); no frozen / pinned file changes.

    python sim/final_test/matrix/k23_codex.py plan --namespace test-k2-v1 --arms sol [--codex-parallel 2] [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import k23_entry as E       # noqa: E402
import k23_run as R         # noqa: E402
import plan as PL           # noqa: E402

CODEX_PIN = "codex-cli 0.162.1"                       # Kitchen 1's frozen CLI (docs/final-test/freeze.json codex_cli_version)
PASS_IDS = {"sol": "codex-sol", "explore": "codex-explore"}
NAMESPACES = ("test-k2-v1", "test-k3-v1", "rehearsal-k2-v1", "rehearsal-k3-v1")   # rehearsals: smoke / selftest only


def arms_of(mode: str) -> tuple:
    if mode == "sol":
        return tuple(C.X_MAIN_ARMS)                                                   # ("X0", "X+M")
    if mode == "explore":
        return tuple(a for m in C.CODEX_EXPLORATORY for a in (f"X-{m}0", f"X-{m}+M"))
    raise ValueError(f"unknown arms mode {mode!r}")


def docs_dir(ns: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "codex", ns)


def work_dir(ns: str, pass_id: str, root: str | None = None) -> str:
    return os.path.join(os.path.dirname(C.ns_work(ns, root)), f"codex-{ns}", pass_id)


def x_cell(ns: str, arm: str, pres: str) -> dict:
    """A plan cell of an X arm (r = 0). Same key scheme as plan.make_cell; the arm registry is read WITHOUT the namespace."""
    info = C.arm_info(arm)
    if not info["planner"].startswith("X:"):
        raise ValueError(f"{arm} is not a Codex arm")
    return {"key": C.cell_id(ns, "plan", arm, pres, "r0"), "arm": arm, "pres": pres, "r": 0,
            "block": "main" if arm in C.X_MAIN_ARMS else "codex", "scope": "free", "planner": info["planner"],
            "pass": "xmain" if arm in C.X_MAIN_ARMS else "codex", "source": info["prompt"]}


class CodexPlan(PL.Plan):
    """plan.Plan on a K23 namespace with ONLY the requested Codex arms, every output redirected to the addendum folders."""

    def __init__(self, ns: str, mode: str, *, root: str | None = None, limit: int | None = None, **kw):
        if ns not in NAMESPACES:
            raise ValueError(f"{ns} is not a K23 test / rehearsal namespace")
        self.mode, self.arms, self.limit = mode, arms_of(mode), limit
        pid = PASS_IDS[mode]
        kw.setdefault("codex_pin", CODEX_PIN)
        kw.setdefault("gpu_sample", False)
        super().__init__(ns, root=root, pass_id=pid, passes=("xmain", "codex"), allow_test=E.kind(ns) == "test", **kw)
        d, w = docs_dir(ns, root), work_dir(ns, pid, root)
        os.makedirs(d, exist_ok=True)
        os.makedirs(w, exist_ok=True)
        self.E_main = C.Journal(C.journal_path(ns, "events", root))                  # read-only: the capture phase_complete
        self.J = C.Journal(os.path.join(d, f"plan.{pid}.jsonl"))
        self.E = C.Journal(os.path.join(d, "events.jsonl"))
        self._cache_dir = os.path.join(w, "cache")
        self._ledger_path = os.path.join(w, "ledger.db")

    def _manifest(self):
        return E.k23_manifest(self.ns, self.root)

    def run(self) -> int:
        """plan.Plan.run with the phase lock in the addendum work folder: the main namespace's plan.lock is never taken (the main test's own
        planning of the same namespace must never meet a held lock)."""
        if self.dry_run:
            self._prepare()
            return self._dry_report()
        lk = C.PhaseLock(self.ns, "plan", self.root, steal=self.steal_lock)
        lk.path = os.path.join(os.path.dirname(self._ledger_path), "plan.lock")
        code = PL.EXIT["ok"]
        with lk:
            try:
                self._prepare()
                self._open()
                self._pause_gate()
                self._quarantine_check()
                code = self._loop()
            except PL._Stop as s:
                code = s.code
            except PL.PlanAbort as e:
                self.ev(e.event, msg=str(e)[:400])
                self.say(f"ABORT {e.event}: {e}")
                code = PL.EXIT["abort"]
            except C.TestItemRefused as e:
                self.ev("test_items_refused", msg=str(e)[:300])
                self.say(f"REFUSED: {e}")
                code = PL.EXIT["abort"]
            except PL.PlannerSkip as e:
                self.ev("planner_skipped", reason=str(e)[:300])
                self.say(f"SKIPPED (nothing recorded): {e}")
                code = PL.EXIT["skip"]
            finally:
                self._stop_gpu()
                self._finish_parallel_report()
                self.ctl.close()
        return code

    def _capture_rows(self) -> dict:
        done = [e for e in self.E_main.rows() if e.get("event") == "phase_complete" and e.get("phase") == "capture"]
        if not done:
            raise PL.PlanAbort("abort_phase_order", f"{self.ns}: the main capture has no phase_complete event")
        want = done[-1].get("journal_sha")
        if want and want != C.sha256_file(self.cap_path):
            raise PL.PlanAbort("abort_capture_changed", "capture.jsonl changed after its phase_complete event")
        rows = {}
        for r in C.Journal(self.cap_path).rows():
            if r.get("namespace") != self.ns:
                raise PL.PlanAbort("abort_design", f"capture record of namespace {r.get('namespace')!r} inside {self.ns}")
            if int(r.get("capture_run", 1)) == 1:
                rows[r["pres"]] = r
        if not rows:
            raise PL.PlanAbort("abort_design", "capture run 1 holds no presentation")
        return rows

    def _prepare(self):
        self.cap = self._capture_rows()
        items = sorted({C.parse_pres(p)["item"] for p in self.cap})
        C.guard_items(items, self.ns, allow_test=self.allow_test, root=self.root)
        canon_pres, _rep, _rel = self._design(self.cap)
        self.cuts = set()
        cells = [x_cell(self.ns, a, p) for p in sorted(canon_pres) for a in self.arms]
        if self.limit:
            keep = sorted(canon_pres)[: int(self.limit)]                              # smoke runs only (rehearsal namespaces)
            cells = [c for c in cells if c["pres"] in keep]
        self.all_cells_full = self.all_cells = self.sel = cells
        self.n_cut = 0
        self.types, self.types_source = PL.load_types(self.types_path, self.root)
        self.contract = PL.make_contract(self.types, self.ns)
        self.latest = self.J.latest()
        self._treatment_check()
        self._verify_requests()


def build(ns: str, mode: str, *, codex_parallel: int = 2, dry_run: bool = False, limit: int | None = None, root=None, **kw) -> CodexPlan:
    E.install_perception_reader()                                                     # as k23_run: the perception freeze's WSL paths via the share
    PL.load_types = R.k23_load_types(ns)                                              # as k23_run.run_plan (this process only)
    tp = R.types_path(E.scene_of(ns), root)
    if limit and E.kind(ns) == "test":
        raise ValueError("--limit is for rehearsal smoke runs only")
    return CodexPlan(ns, mode, root=root, limit=limit, types_path=tp, codex_parallel=codex_parallel, parallel=2, dry_run=dry_run, **kw)   # parallel > 1 enables plan.Plan's block-parallel path + the Codex throttle


# ================================================================================================ execute (WSL): new Sol units only
EXEC_PASS = "codex-exec"


def exec_paths(ns: str, root: str | None = None) -> dict:
    d = docs_dir(ns, root)
    return {"exec": os.path.join(d, "exec.codex.jsonl"), "outcome": os.path.join(d, "outcome.codex.jsonl"),
            "events": os.path.join(d, "events.jsonl"), "units": os.path.join(d, "units.codex.json"),
            "xplan": os.path.join(d, f"plan.{PASS_IDS['sol']}.jsonl"), "lock": os.path.join(work_dir(ns, EXEC_PASS, root), "execute.lock")}


def sol_plan_complete(ns: str, root: str | None = None) -> str | None:
    """None when the Sol planning pass is complete and its journal unchanged since, else the reason."""
    p = exec_paths(ns, root)
    if not os.path.exists(p["xplan"]):
        return "no Sol plan journal"
    sha = C.sha256_file(p["xplan"])
    ok = [e for e in C.Journal(p["events"]).rows() if e.get("event") == "plan_pass_complete" and e.get("pass_id") == PASS_IDS["sol"]
          and e.get("journal_sha") == sha]
    return None if ok else "the Sol planning pass has no plan_pass_complete bound to the current journal"


def _execute_cls():
    import execute as XE

    class CodexExecute(XE.ExecutePhase):
        """execute.ExecutePhase over the main test's units PLUS the Sol cells: a unit the main test already ran (same arm-free unit key: same
        plan, presentation, restored world, summary, caps, seed policy, executor policy) is REUSED (its main exec row is the result); only units
        that exist for Sol alone are executed, into the addendum journals. The main journals are read, never written."""

        def __init__(self, ns: str, *, root: str | None = None, **kw):
            super().__init__(ns, root=root, allow_test=E.kind(ns) == "test", worlds=("canon",), p_arms=XE.P_ARMS, replay=False, **kw)
            self.xp = exec_paths(ns, root)
            os.makedirs(os.path.dirname(self.xp["lock"]), exist_ok=True)
            os.makedirs(os.path.dirname(self.xp["exec"]), exist_ok=True)
            self.main = {k: self.J[k] for k in ("plan", "exec", "outcome", "events")}
            self.J = {"capture": self.J["capture"], "plan": self.J["plan"], "exec": C.Journal(self.xp["exec"]),
                      "outcome": C.Journal(self.xp["outcome"]), "events": C.Journal(self.xp["events"])}
            self.units_path = self.xp["units"]
            self._main_ix = None
            self.reused = 0

        def _epochs(self):
            self.epoch = {}
            rows = self.main["events"].rows() + self.J["events"].rows()
            for r in rows:
                if r.get("event") == "invalid_resolved":
                    for p in r.get("pres") or []:
                        self.epoch[p] = self.epoch.get(p, 0) + 1
            self.unrecoverable = C.invalid_pres(rows)

        def preflight(self) -> dict:
            mev = self.main["events"].rows()
            pf = XE.plan_first_problem(mev, self.main["plan"].path)
            if pf:
                raise XE.ExecAborted("main test: " + pf)
            for ph in ("capture", "plan", "execute"):
                if not [e for e in mev if e.get("event") == "phase_complete" and e.get("phase") == ph]:
                    raise XE.ExecAborted(f"the main test of {self.ns} has no {ph} phase_complete: the Codex execution waits for it")
            why = sol_plan_complete(self.ns, self.root)
            if why:
                raise XE.ExecAborted(why)
            self.captures = {}
            for r in self.J["capture"].rows():
                if r.get("capture_run") == 1:
                    self.captures[r["pres"]] = r
            main_plans = {k: r for k, r in self.main["plan"].latest().items() if k}
            xplans = {k: r for k, r in C.Journal(self.xp["xplan"]).latest().items() if k}
            bad = [k for k, r in xplans.items() if r.get("arm") not in C.X_MAIN_ARMS]
            if bad or set(xplans) & set(main_plans):
                raise XE.ExecAborted(f"the Sol journal holds a non-Sol or a main-test key ({(bad or sorted(set(xplans) & set(main_plans)))[:2]})")
            self.plans = dict(main_plans, **xplans)
            pres_all = sorted(set(self.captures) | {r["pres"] for r in self.plans.values()})
            guard = C.guard_items(sorted({C.parse_pres(p)["item"] for p in pres_all}), self.ns, allow_test=self.allow_test, root=self.root)
            for p in pres_all:
                if p not in self.captures:
                    raise RuntimeError(f"plan rows name {p} but there is no capture row (run 1)")
            self._epochs()
            cs = [e for e in mev if e.get("event") == "capture_start"]
            dp = C.dataset_problems(self.ns, self.root, (cs[-1].get("dataset_sha256") if cs else None))
            if dp:
                raise XE.ExecAborted("; ".join(dp))
            if self._invalid_open():
                raise XE.ExecAborted("an earlier invalid Codex execution is unresolved")
            self._treatment_check()
            return guard

        def _main_rows(self, key: str) -> list:
            if self._main_ix is None:
                ix = {}
                for r in self.main["exec"].rows():
                    ix.setdefault(r.get("key"), []).append(r)
                self._main_ix = ix
            return self._main_ix.get(key, [])

        def cell_state(self, unit: dict, run: int) -> tuple:
            mrows = [r for r in self._main_rows(self.cell_key(unit["unit"], run)) if int(r.get("epoch", 0)) >= self.epoch.get(unit["pres"], 0)]
            if mrows:                                               # executed by the main test: reused, never run again
                last = mrows[-1]
                if last["status"] in ("done", "not_executed") or last["attempt"] >= XE.MAX_ATTEMPT or unit["pres"] in self.unrecoverable:
                    return "done", last["attempt"]
            return super().cell_state(unit, run)

        def _finish(self, info: dict) -> None:
            """execute._finish plus the binding of this completion to the Sol plan journal and the units file it ran (score checks both)."""
            for kind in ("exec", "outcome", "events"):
                p = self.J[kind].path
                if os.path.exists(p) and C.scan_jsonl_leaks(p, self._secrets()):
                    raise XE.ExecInvalid(f"key-like text found in {p}: phase_complete withheld")
            sha = C.sha256_file(self.J["exec"].path) if os.path.exists(self.J["exec"].path) else ""
            self.event("phase_complete", n=info["cells"], journal_sha=sha, units=info["units"], sol_plan_sha=C.sha256_file(self.xp["xplan"]),
                       units_sha=C.sha256_file(self.units_path), sol_units=info["sol_units"], reused=info["sol_units_reused"])
            info["complete"] = True

        def run(self, stop_after: int | None = None, dry_run: bool = False) -> dict:
            guard = self.preflight()
            b = self.build()
            xkeys = {k for k, r in self.plans.items() if r.get("arm") in C.X_MAIN_ARMS}
            xunits = [u for u in b["units"].values() if any(c["key"] in xkeys for c in u["cells"])]
            shared = [u for u in xunits if any(self._main_rows(self.cell_key(u["unit"], run)) for run in range(u["runs"]))]
            info = {"guard": guard, "units": len(b["units"]), "cells": sum(u["runs"] for u in b["units"].values()), "sol_units": len(xunits),
                    "sol_units_reused": len(shared),
                    "sol_units_new": len(xunits) - len(shared), "unexecuted_plans": len(b["unexecuted"]), "executed": 0,
                    "interrupted": False, "complete": False}
            if dry_run:
                info["todo"] = {r: sum(len(us) for _, us in blocks) for r, blocks in self.todo().items()}
                return info
            lk = C.PhaseLock(self.ns, XE.PHASE, self.root, steal=getattr(self, "steal", False))
            lk.path = self.xp["lock"]
            with lk:
                self.event("codex_execute_start", sol_units=len(xunits), reused=len(shared), new=len(xunits) - len(shared))
                try:
                    for run in range(XE.REPEAT_RUNS):
                        for pres, unit_keys in self.todo()[run]:
                            for uk in unit_keys:
                                if stop_after is not None and info["executed"] >= stop_after:
                                    info["interrupted"] = True
                                    self.event("interrupt", n=info["executed"])
                                    return info
                                rec = self.do_cell(self.b["units"][uk], run)
                                info["executed"] += 1
                                self.out(json.dumps({"t": __import__("time").strftime("%H:%M:%S"), "pres": pres, "unit": uk[:10], "run": run,
                                                     "status": rec["status"], "attempt": rec["attempt"], "s": rec["wall_s"]}, ensure_ascii=False))
                finally:
                    self._drop_executor()
                remaining = sum(len(us) for blocks in self.todo().values() for _, us in blocks)
                if remaining == 0:
                    self._finish(info)
            return info

    return CodexExecute, XE


def run_execute(ns: str, *, dry_run: bool = False, steal_lock: bool = False, root=None) -> int:
    CodexExecute, XE = _execute_cls()
    PL.load_types = R.k23_load_types(ns)
    ph = CodexExecute(ns, root=root, executor_factory=None if dry_run else XE.live_executor_factory,
                      memory_opener=None if dry_run else XE.live_memory_opener)
    ph.steal = bool(steal_lock)               # the wrapper passes it only under the sim lock with no live Codex execute process (a hang exit 75 leaves the lock)
    if steal_lock and os.path.exists(ph.xp["lock"]):
        ph.event("lock_stolen", lock=os.path.basename(ph.xp["lock"]), held=open(ph.xp["lock"], encoding="utf-8").read()[:200])
    try:
        info = ph.run(dry_run=dry_run)
    except C.TestItemRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return XE.EXIT["refused"]
    except C.PhaseLocked as e:
        print(f"LOCKED: {e}", file=sys.stderr)
        return XE.EXIT["refused"]
    except (XE.ExecAborted, XE.ExecInvalid) as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return XE.EXIT["abort_invalid"]
    print(json.dumps({k: v for k, v in info.items() if k != "guard"}))
    return 0


# ================================================================================================ score: the addendum's own artifact
def run_score(ns: str, *, root=None) -> int:
    import score as S
    xp = exec_paths(ns, root)
    d = docs_dir(ns, root)
    mev = C.Journal(C.journal_path(ns, "events", root)).rows()
    if not [e for e in mev if e.get("event") == "phase_complete" and e.get("phase") == "score"]:
        print("REFUSED: the main test is not scored yet", file=sys.stderr)
        return 2
    aev = C.Journal(xp["events"]).rows()
    sol_done = [e for e in aev if e.get("event") == "phase_complete" and e.get("phase") == "execute"
                and e.get("journal_sha") == (C.sha256_file(xp["exec"]) if os.path.exists(xp["exec"]) else "")]
    if not sol_done:
        print("REFUSED: the Codex execution is not complete", file=sys.stderr)
        return 2
    if sol_done[-1].get("sol_plan_sha") != C.sha256_file(xp["xplan"]) or sol_done[-1].get("units_sha") != C.sha256_file(xp["units"]):
        print("REFUSED: the Sol plan journal or units.codex.json changed after the execution completed (re-run the execution)", file=sys.stderr)
        return 2
    gold, gold_sha = S._gold_load(root, None, verify=True)
    fam = S._family_map(root)
    main = S.Inputs(ns, root)                                          # main exec / outcome (read-only), for reused units
    ax = C.Journal(xp["exec"]).latest()
    ao = C.Journal(xp["outcome"]).latest()
    ahist, astarts = {}, {}
    for r in C.Journal(xp["exec"]).rows():
        ahist.setdefault(r.get("key"), []).append(r)
    for ev in C.Journal(xp["events"]).rows():
        if ev.get("event") == S.A36.START_EVENT:
            astarts.setdefault(ev.get("key"), []).append(ev)
    by_unit = {}
    for k, e in main.exec_rows.items():
        by_unit[(e.get("unit"), int(e.get("run", 0)))] = (e, main.outcome_rows.get(k), main.exec_hist.get(k), main.exec_starts.get(k, []), "main")
    for k, e in ax.items():
        by_unit[(e.get("unit"), int(e.get("run", 0)))] = (e, ao.get(k), ahist.get(k), astarts.get(k, []), "codex")
    units = json.load(open(xp["units"], encoding="utf-8"))["units"]
    cell_unit = {c["key"]: uk for uk, u in units.items() for c in u["cells"]}
    inval = S._invalid_pres(mev + aev)                                 # main AND addendum invalidation events
    epoch = {}
    for ev in mev + aev:
        if ev.get("event") == "invalid_resolved":
            for p_ in ev.get("pres") or []:
                epoch[p_] = epoch.get(p_, 0) + 1
    rows, diag, missing = [], [], []
    for mode in ("sol", "explore"):
        path = os.path.join(d, f"plan.{PASS_IDS[mode]}.jsonl")
        if not os.path.exists(path):
            continue
        for cell, prow in sorted(C.Journal(path).latest().items()):
            cp = S._cell_parts(cell)
            pp = C.parse_pres(cp["pres"])
            g, f = gold[pp["item"]], fam.get(pp["item"])
            if C.plan_only(cp["arm"]):
                diag.append(S.score_cell(cell, 0, plan_row=prow, exec_row=None, outcome=None, gold_item=g, family=f, ns=ns,
                                         invalid_episode=cp["pres"] in inval, plan_only=True))
                continue
            if prow["status"] != "ok":
                rows.append(dict(S.score_cell(cell, 0, plan_row=prow, exec_row=None, outcome=None, gold_item=g, family=f, ns=ns,
                                              invalid_episode=cp["pres"] in inval), exec_source=None))
                continue
            uk = cell_unit.get(cell)
            hit = by_unit.get((uk, 0)) if uk else None
            if hit is None:
                missing.append(cell)
                continue
            e, o, hist, starts, src = hit
            rows.append(dict(S.score_cell(cell, 0, plan_row=prow, exec_row=e, outcome=o, gold_item=g, family=f, ns=ns,
                                          invalid_episode=cp["pres"] in inval, exec_history=hist, exec_starts=starts, block=prow.get("block"),
                                          exec_epoch=epoch.get(cp["pres"], 0)), exec_source=src))
    if missing:
        print(f"ERROR: {len(missing)} accepted Sol cell(s) have no execution (first {missing[0]})", file=sys.stderr)
        return 3
    out = os.path.join(d, "score.codex.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    dg = S.plan_diagnostics(diag, ns)
    with open(os.path.join(d, "plan_diagnostics.codex.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dg, fh, ensure_ascii=False, indent=1)
    summ = summarize(ns, rows, root)
    summ.update(gold_sha256=gold_sha, score_sha256=C.sha256_file(out), plan_only_rows=len(diag))
    with open(os.path.join(d, "summary.codex.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    C.Journal(xp["events"]).append(C.make_record("events", event="phase_complete", phase="score", n=len(rows), journal_sha=C.sha256_file(out)))
    print(json.dumps({"namespace": ns, "rows": len(rows), "plan_only": len(diag), "summary": summ["by_arm"]}, ensure_ascii=False)[:2000])
    return 0


def summarize(ns: str, xrows: list, root=None) -> dict:
    """Per arm and level: the level metric (L1 SR, L2 PC, L3 TDS) plus SR everywhere; the main arms from the main sealed score for context."""
    main_rows = [r for r in C.Journal(C.journal_path(ns, "score", root)).rows() if r.get("scored") and r.get("run") == 0]

    def agg(rows):
        out = {}
        for lv in ("L1", "L2", "L3"):
            rr = [r for r in rows if r.get("level") == lv]
            if not rr:
                continue
            sr = [r["sr"] for r in rr if r.get("sr") is not None]
            pc = [r["pc_final"] for r in rr if r.get("pc_final") is not None]
            td = [r["tds_final"] for r in rr if r.get("tds_final") is not None]
            out[lv] = {"n": len(rr), "sr": round(sum(sr) / len(sr), 4) if sr else None, "pc": round(sum(pc) / len(pc), 4) if pc else None,
                       "tds": round(sum(td) / len(td), 4) if td else None}
        sr_all = [r["sr"] for r in rows if r.get("sr") is not None]
        out["all"] = {"n": len(rows), "sr": round(sum(sr_all) / len(sr_all), 4) if sr_all else None}
        return out
    by = {}
    for arm in sorted({r["arm"] for r in xrows}):
        by[arm] = agg([r for r in xrows if r["arm"] == arm and r.get("scored", True)])
    ctx = {arm: agg([r for r in main_rows if r.get("arm") == arm]) for arm in ("G0", "G+M", "O0", "O+M")}
    reused = sum(1 for r in xrows if r.get("exec_source") == "main")
    return {"namespace": ns, "by_arm": by, "main_arms_context": ctx, "sol_rows": len(xrows), "executions_reused_from_main": reused,
            "status": "ADDENDUM (K23-CODEX-ADDENDUM): post-registration, reported separately from the main K2 / K3 analysis"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("execute")
    x.add_argument("--namespace", required=True, choices=NAMESPACES)
    x.add_argument("--dry-run", action="store_true")
    x.add_argument("--steal-lock", action="store_true", help="wrapper only: the addendum lock of a dead (hang-killed) execute, under the sim lock")
    sc = sub.add_parser("score")
    sc.add_argument("--namespace", required=True, choices=NAMESPACES)
    p = sub.add_parser("plan")
    p.add_argument("--namespace", required=True, choices=NAMESPACES)
    p.add_argument("--arms", required=True, choices=sorted(PASS_IDS))
    p.add_argument("--codex-parallel", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="rehearsal smoke: only the first N presentations")
    p.add_argument("--requeue-infra", action="store_true", help="plan.py's registered re-queue of infra cells (up to 2 later sessions)")
    s = sub.add_parser("status")
    s.add_argument("--namespace", required=True, choices=NAMESPACES)
    a = ap.parse_args(argv)
    if a.cmd == "execute":
        return run_execute(a.namespace, dry_run=a.dry_run, steal_lock=a.steal_lock)
    if a.cmd == "score":
        E.install_perception_reader()
        return run_score(a.namespace)
    if a.cmd == "status":
        out = {}
        for mode, pid in PASS_IDS.items():
            path = os.path.join(docs_dir(a.namespace), f"plan.{pid}.jsonl")
            rows = C.Journal(path).latest() if os.path.exists(path) else {}
            by = {}
            for r in rows.values():
                by.setdefault(r["arm"], {}).setdefault(r["status"], 0)
                by[r["arm"]][r["status"]] += 1
            out[mode] = {"rows": len(rows), "by_arm": by}
        print(json.dumps(out, indent=1))
        return 0
    pl = build(a.namespace, a.arms, codex_parallel=a.codex_parallel, dry_run=a.dry_run, limit=a.limit, requeue_infra=a.requeue_infra)
    try:
        code = pl.run()
    except (PL.PlanAbort, C.TestItemRefused, C.PhaseLocked, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return PL.EXIT["abort"]
    print(json.dumps({"namespace": a.namespace, "arms": a.arms, "exit": code, "journal": os.path.relpath(pl.J.path, C.paths()["root"])}))
    return code


if __name__ == "__main__":
    sys.exit(main())
