"""K2 RENDER-WINDOW RE-EXECUTION: a separate pass that re-executes the 54 test-k2-v1 units of the suspected software-rendering window.

Arash's decision (2026-10-21 morning, record docs/final-test/k23/decisions/K23-WINDOW-RERUN.json), asked "the 54 flagged Kitchen 2 units:
keep the registered handling plus the sensitivity check, or re-run them in a separate pass? The primary results stay untouched either way":
"you can do all of these three now". Decided AFTER the K2 / K3 results were seen: a disclosed, post-hoc SENSITIVITY analysis. The primary
(registered) results and their sealed score stay exactly as they are; this pass never replaces them.

What is re-executed: every (unit, run) of docs/final-test/k23/infra/K2-RENDER-WINDOW.json (the units with an attempt in the 02:02:49-03:21:36
window: 31 ended blocked_state, 23 done), with the main test's own plans, captures, restored worlds, snapshots, treatment and executor
(execute.ExecutePhase, attempts 0-2 as registered). Every other unit is skipped (its main row stands).

Isolation (nothing of the main test is touched):
  writes only  docs/final-test/k23/rerun/<ns>/{exec.rerun.jsonl, outcome.rerun.jsonl, events.jsonl, units.rerun.json, score.rerun.jsonl,
               summary.rerun.json, analysis_rerun.{json,md}}  and  sim/final_test/matrix/work/rerun-<ns>/{execute.lock, testlogs}
  reads        the main capture / plan / exec / outcome / events / score journals of <ns> (and test-k3-v1's sealed store for the analysis)

    python sim/final_test/matrix/k23_rerun.py execute --namespace test-k2-v1 [--dry-run]     (WSL, via k23_run/rerun/run_rerun_exec.sh)
    python sim/final_test/matrix/k23_rerun.py score   --namespace test-k2-v1                 (Windows)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import k23_entry as E       # noqa: E402
import k23_run as R         # noqa: E402
import plan as PL           # noqa: E402

PASS_ID = "k2-window-rerun"
NAMESPACES = ("test-k2-v1",)
WINDOW_RECORD = {"test-k2-v1": "K2-RENDER-WINDOW.json"}
WINDOW_UNITS_N = {"test-k2-v1": 54}


def docs_dir(ns: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "rerun", ns)


def work_dir(ns: str, root: str | None = None) -> str:
    return os.path.join(os.path.dirname(C.ns_work(ns, root)), f"rerun-{ns}")


def paths(ns: str, root: str | None = None) -> dict:
    d = docs_dir(ns, root)
    return {"exec": os.path.join(d, "exec.rerun.jsonl"), "outcome": os.path.join(d, "outcome.rerun.jsonl"),
            "events": os.path.join(d, "events.jsonl"), "units": os.path.join(d, "units.rerun.json"),
            "score": os.path.join(d, "score.rerun.jsonl"), "summary": os.path.join(d, "summary.rerun.json"),
            "lock": os.path.join(work_dir(ns, root), "execute.lock")}


def window_record_path(ns: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "infra", WINDOW_RECORD[ns])


def window_keys(ns: str, root: str | None = None) -> set:
    """The exec keys (ns|exec|unit|runN) of the window record; refuses a record of another namespace or another size."""
    with open(window_record_path(ns, root), encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("namespace") != ns:
        raise ValueError(f"window record is for {rec.get('namespace')!r}, not {ns}")
    keys = set()
    for unit, rows in (rec.get("units") or {}).items():
        for r in rows:
            if not str(r.get("key", "")).startswith(f"{ns}|exec|{unit}|run"):
                raise ValueError(f"window record row {r.get('key')!r} does not belong to unit {unit}")
            keys.add(r["key"])
    n_units = len({k.split("|")[2] for k in keys})
    if n_units != WINDOW_UNITS_N[ns] or n_units != rec.get("affected_units"):
        raise ValueError(f"window record names {n_units} units; expected {WINDOW_UNITS_N[ns]} (affected_units {rec.get('affected_units')})")
    return keys


# ================================================================================================ execute (WSL)
def _execute_cls():
    import execute as XE

    class RerunExecute(XE.ExecutePhase):
        """execute.ExecutePhase over the main test's units, restricted to the window (unit, run) keys; every other unit is 'done' (its main row
        stands). Attempts 0-2 start afresh in this pass's own journal. The main journals are read, never written."""

        def __init__(self, ns: str, *, root: str | None = None, window: set | None = None, **kw):
            super().__init__(ns, root=root, allow_test=E.kind(ns) == "test", worlds=("canon",), p_arms=XE.P_ARMS, replay=False, **kw)
            self.xp = paths(ns, root)
            os.makedirs(os.path.dirname(self.xp["lock"]), exist_ok=True)
            os.makedirs(os.path.dirname(self.xp["exec"]), exist_ok=True)
            self.work = work_dir(ns, root)                        # live_memory_opener's testlogs go here, not into the main work dir
            self.main = {k: self.J[k] for k in ("plan", "exec", "outcome", "events")}
            self.J = {"capture": self.J["capture"], "plan": self.J["plan"], "exec": C.Journal(self.xp["exec"]),
                      "outcome": C.Journal(self.xp["outcome"]), "events": C.Journal(self.xp["events"])}
            self.units_path = self.xp["units"]
            self.window = set(window) if window is not None else window_keys(ns, root)

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
            for ph in ("capture", "plan", "execute", "score"):
                if not [e for e in mev if e.get("event") == "phase_complete" and e.get("phase") == ph]:
                    raise XE.ExecAborted(f"the main test of {self.ns} has no {ph} phase_complete: the re-execution waits for it")
            self.captures = {}
            for r in self.J["capture"].rows():
                if r.get("capture_run") == 1:
                    self.captures[r["pres"]] = r
            self.plans = {k: r for k, r in self.main["plan"].latest().items() if k}
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
                raise XE.ExecAborted("an earlier invalid re-execution is unresolved")
            self._treatment_check()
            return guard

        def window_units(self) -> list:
            """The window (unit, run) pairs, each checked against the rebuilt units AND the main execution: same unit key, same cells."""
            mx = {}
            for r in self.main["exec"].rows():
                mx.setdefault(r.get("key"), []).append(r)
            out = []
            for k in sorted(self.window):
                _, _, uk, rr = k.split("|")
                u = self.b["units"].get(uk)
                if u is None:
                    raise XE.ExecAborted(f"window unit {uk[:12]} is not among the rebuilt units (the inputs differ from the main test)")
                if not mx.get(k):
                    raise XE.ExecAborted(f"window key {k} has no main exec row")
                if sorted(c["key"] for c in u["cells"]) != sorted(mx[k][-1].get("cells") or []):
                    raise XE.ExecAborted(f"window unit {uk[:12]}: rebuilt cells differ from the main execution's cells")
                out.append((u, int(rr[3:])))
            return out

        def cell_state(self, unit: dict, run: int) -> tuple:
            if self.cell_key(unit["unit"], run) not in self.window:
                return "done", 0                                    # outside the window: the main row stands, never run here
            return super().cell_state(unit, run)

        def _append(self, kind: str, rec: dict) -> None:
            if kind in ("exec", "outcome"):
                rec = dict(rec, pass_id=PASS_ID)
            super()._append(kind, rec)

        def _finish(self, info: dict) -> None:
            for kind in ("exec", "outcome", "events"):
                p = self.J[kind].path
                if os.path.exists(p) and C.scan_jsonl_leaks(p, self._secrets()):
                    raise XE.ExecInvalid(f"key-like text found in {p}: phase_complete withheld")
            sha = C.sha256_file(self.J["exec"].path) if os.path.exists(self.J["exec"].path) else ""
            self.event("phase_complete", n=info["window_units"], journal_sha=sha, units_sha=C.sha256_file(self.units_path),
                       window_record_sha=C.sha256_file(window_record_path(self.ns, self.root)), pass_id=PASS_ID)
            info["complete"] = True

        def run(self, stop_after: int | None = None, dry_run: bool = False) -> dict:
            guard = self.preflight()
            self.build()
            wu = self.window_units()
            info = {"guard": guard, "units": len(self.b["units"]), "window_units": len(wu), "executed": 0, "interrupted": False, "complete": False}
            if dry_run:
                info["todo"] = {r: sum(len(us) for _, us in blocks) for r, blocks in self.todo().items()}
                return info
            lk = C.PhaseLock(self.ns, XE.PHASE, self.root, steal=getattr(self, "steal", False))
            lk.path = self.xp["lock"]
            with lk:
                self.event("rerun_execute_start", window_units=len(wu), pass_id=PASS_ID,
                           window_record_sha=C.sha256_file(window_record_path(self.ns, self.root)))
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
                                self.out(json.dumps({"t": time.strftime("%H:%M:%S"), "pres": pres, "unit": uk[:10], "run": run,
                                                     "status": rec["status"], "attempt": rec["attempt"], "s": rec["wall_s"]}, ensure_ascii=False))
                finally:
                    self._drop_executor()
                remaining = sum(len(us) for blocks in self.todo().values() for _, us in blocks)
                if remaining == 0:
                    self._finish(info)
            return info

    return RerunExecute, XE


def run_execute(ns: str, *, dry_run: bool = False, steal_lock: bool = False, root=None) -> int:
    RerunExecute, XE = _execute_cls()
    PL.load_types = R.k23_load_types(ns)
    ph = RerunExecute(ns, root=root, executor_factory=None if dry_run else XE.live_executor_factory,
                      memory_opener=None if dry_run else XE.live_memory_opener)
    ph.steal = bool(steal_lock)               # the wrapper passes it only under the sim lock with no live re-execution process
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


# ================================================================================================ score + analysis (Windows)
_STRIP = ("ts",)


def _same(a: dict, b: dict) -> bool:
    return {k: v for k, v in a.items() if k not in _STRIP} == {k: v for k, v in b.items() if k not in _STRIP}


def rescore(ns: str, *, root=None) -> dict:
    """The main sealed score rows with every row of a window unit re-scored from this pass's execution. Self-check: the same re-scoring
    applied to the MAIN execution reproduces each main row exactly (ts aside). Returns {rows, changed, check}."""
    import score as S
    xp = paths(ns, root)
    mev = C.Journal(C.journal_path(ns, "events", root)).rows()
    if not [e for e in mev if e.get("event") == "phase_complete" and e.get("phase") == "score"]:
        raise RuntimeError("the main test is not scored")
    aev = C.Journal(xp["events"]).rows()
    sha = C.sha256_file(xp["exec"]) if os.path.exists(xp["exec"]) else ""
    done = [e for e in aev if e.get("event") == "phase_complete" and e.get("phase") == "execute" and e.get("journal_sha") == sha]
    if not done:
        raise RuntimeError("the re-execution is not complete (no phase_complete bound to the current exec.rerun.jsonl)")
    if done[-1].get("window_record_sha") != C.sha256_file(window_record_path(ns, root)) or done[-1].get("units_sha") != C.sha256_file(xp["units"]):
        raise RuntimeError("the window record or units.rerun.json changed after the re-execution completed")
    seal = S.verify_seal(ns, root)
    if not seal.get("complete"):
        raise RuntimeError("the main score seal is not complete")
    win = window_keys(ns, root)
    gold, _ = S._gold_load(root, None, verify=True)
    fam = S._family_map(root)
    main = S.Inputs(ns, root)
    inval = S._invalid_pres(mev + aev)
    ax, ao = C.Journal(xp["exec"]).latest(), C.Journal(xp["outcome"]).latest()
    ahist, astarts = {}, {}
    for r in C.Journal(xp["exec"]).rows():
        ahist.setdefault(r.get("key"), []).append(r)
    for ev in aev:
        if ev.get("event") == S.A36.START_EVENT:
            astarts.setdefault(ev.get("key"), []).append(ev)
    missing = sorted(k for k in win if k not in ax)
    if missing:
        raise RuntimeError(f"{len(missing)} window key(s) have no re-execution row (first {missing[0]})")
    main_rows = C.Journal(main.paths["score"]).rows()
    out, changed, check_bad, n_win = [], [], [], 0
    for r in main_rows:
        ek = f"{ns}|exec|{r.get('unit')}|run{int(r.get('run', 0))}"
        if r.get("unit") is None or ek not in win:
            out.append(r)
            continue
        n_win += 1
        cp = S._cell_parts(r["cell"])
        prow = main.plan.get(r["cell"])
        g, f = gold[C.parse_pres(cp["pres"])["item"]], fam.get(C.parse_pres(cp["pres"])["item"])
        kw = dict(plan_row=prow, gold_item=g, family=f, ns=ns, invalid_episode=cp["pres"] in inval,
                  block=S._block_of(main, cp, prow), exec_epoch=main.exec_epoch.get(cp["pres"], 0))
        me, mo = main.exec_rows[ek], main.outcome_rows.get(ek)
        again = S.score_cell(r["cell"], int(r["run"]), exec_row=me, outcome=mo, exec_history=main.exec_hist.get(ek),
                             exec_starts=main.exec_starts.get(ek, []), plan_steps_for_metrics=(None if prow else S._plan_of_exec(me, mo)), **kw)
        if not _same(again, r):
            check_bad.append(r["key"])
        xe, xo = ax[ek], ao.get(ek)
        new = S.score_cell(r["cell"], int(r["run"]), exec_row=xe, outcome=xo, exec_history=ahist.get(ek), exec_starts=astarts.get(ek, []),
                           plan_steps_for_metrics=(None if prow else S._plan_of_exec(xe, xo)), **kw)
        new.update(rerun=True, rerun_pass=PASS_ID, main_sr=r["sr"], main_exec_status=r.get("exec_status"))
        out.append(new)
        changed.append({"key": r["key"], "arm": r["arm"], "pres": r["pres"], "level": r["level"], "main": [r.get("exec_status"), r["sr"]],
                        "rerun": [new.get("exec_status"), new["sr"]]})
    return {"rows": out, "main_rows": main_rows, "changed": changed, "n_window_rows": n_win, "check_bad": check_bad, "seal": seal}


def _sr_table(rows) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm and r.get("scored") and r.get("sr") is not None]
        d = {"all": [round(100 * sum(r["sr"] for r in rr) / len(rr), 2), len(rr)] if rr else None}
        for lv in ("L1", "L2", "L3"):
            x = [r for r in rr if r.get("level") == lv]
            d[lv] = [round(100 * sum(r["sr"] for r in x) / len(x), 2), len(x)] if x else None
        out[arm] = d
    return out


def run_score(ns: str, *, root=None, analyse: bool = True) -> int:
    xp = paths(ns, root)
    try:
        res = rescore(ns, root=root)
    except RuntimeError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if res["check_bad"]:
        print(f"ERROR: re-scoring the MAIN execution does not reproduce {len(res['check_bad'])} main row(s) (first {res['check_bad'][0]})",
              file=sys.stderr)
        return 3
    rows = res["rows"]
    with open(xp["score"], "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    st = {}
    for c in res["changed"]:
        t = (c["main"][0], c["rerun"][0])
        st[f"{t[0]} -> {t[1]}"] = st.get(f"{t[0]} -> {t[1]}", 0) + 1
    flips = {"0->1": sum(1 for c in res["changed"] if c["main"][1] == 0 and c["rerun"][1] == 1),
             "1->0": sum(1 for c in res["changed"] if c["main"][1] == 1 and c["rerun"][1] == 0)}
    summ = {"namespace": ns, "pass_id": PASS_ID,
            "status": "POST-HOC SENSITIVITY (K23-WINDOW-RERUN): decided after the results were seen; the registered primary results stand",
            "window_record_sha256": C.sha256_file(window_record_path(ns, root)), "main_score_seal_sha256": res["seal"].get("sha256"),
            "rerun_score_sha256": C.sha256_file(xp["score"]), "n_rows": len(rows), "n_window_rows": res["n_window_rows"],
            "exec_status_transitions": st, "sr_flips": flips,
            "sr_pct": {"main": _sr_table(res["main_rows"]), "window_reexecuted": _sr_table(rows)}, "rows_changed": res["changed"]}
    if analyse:
        summ["analysis"] = run_analysis(ns, rows, root=root)
    with open(xp["summary"], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    C.Journal(xp["events"]).append(C.make_record("events", event="phase_complete", phase="score", n=len(rows), journal_sha=summ["rerun_score_sha256"],
                                                 pass_id=PASS_ID))
    print(json.dumps({k: summ[k] for k in ("n_rows", "n_window_rows", "exec_status_transitions", "sr_flips")}, ensure_ascii=False))
    return 0


def analysis_exec_rows(main_rows, rerun_rows, win: set) -> list:
    """The execution history the analysis sees: the main rows of every NON-window key, then the re-execution's rows. Every main row of a
    window key is removed, so neither the execution links nor execution_block (which keeps only terminal successful rows) can fall back to
    a main attempt of a substituted unit (Astra K23-RERUN-r1 item 3: a main done -> re-run blocked transition kept 28.07 s)."""
    return [r for r in (main_rows or []) if r.get("key") not in win] + list(rerun_rows or [])


def run_analysis(ns: str, rows: list, *, root=None) -> dict:
    """The registered K2 / K3 analysis (k23_analyze, pinned, by import) on the K2 store with the window rows re-executed and the K3 store as
    sealed; written beside this pass, never over analysis_k23.*. Returns the headline contrasts."""
    import k23_analyze as KA
    E.install_perception_reader()
    stores = {s: KA.load_store(C.ns_docs(KA.NS_OF[s], root), s, root=root) for s in KA.SCENES}
    k2 = stores["FloorPlan15"]
    if k2["namespace"] != ns:
        raise RuntimeError("the re-executed store is not FloorPlan15")
    k2["rows"] = rows
    k2["exec_rows"] = analysis_exec_rows(k2["exec_rows"], C.Journal(paths(ns, root)["exec"]).rows(), window_keys(ns, root))
    man = C.load_manifest(root)
    fam_of = {i: f["family"] for f in man["families"] for i in f["items"]}
    for s, st in stores.items():
        C.guard_items(sorted({C.parse_pres(r["pres"])["item"] for r in st["rows"]}), st["namespace"], allow_test=True, root=root)
    meta = {"namespaces": {s: KA.NS_OF[s] for s in KA.SCENES}, "seal_sha256": {s: stores[s]["seal"].get("sha256") for s in stores},
            "n_score_rows": {s: len(stores[s]["rows"]) for s in stores}, "complete": True, "partial_flag": False, "kitchens_not_run": [],
            "synthetic": False, "B_cost": 20000, "rerun_pass": PASS_ID,
            "note": "FloorPlan15 rows of the 54 render-window units re-scored from the separate re-execution (POST-HOC SENSITIVITY); "
                    "FloorPlan17 as sealed; the registered primary analysis is docs/final-test/k23/analysis/analysis_k23.*"}
    freeze = {s: KA.freeze_cost_record(s, root) for s in stores}
    res = KA.analyse_k23(stores, fam_of_manifest=fam_of, meta=meta, freeze_records=freeze)
    d = docs_dir(ns, root)
    with open(os.path.join(d, "analysis_rerun.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(d, "analysis_rerun.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("> POST-HOC SENSITIVITY (K23-WINDOW-RERUN): FloorPlan15's 54 render-window units re-executed in a separate pass; not the registered "
                "primary analysis (docs/final-test/k23/analysis/analysis_k23.md).\n\n" + KA.render_md(res))
    pick = lambda p: {kk: (p or {}).get(kk) for kk in ("estimate", "interval", "se", "nu", "interval_label")}   # noqa: E731
    crit = ((res.get("combined") or {}).get("g_m_vs_o0_criteria") or {}).get("primary") or {}
    return {"pooled_primary": {k: pick(vs.get("primary")) for k, vs in (res.get("combined") or {}).get("contrasts", {}).items()},
            "g_m_vs_o0_criteria_primary": {k: (v or {}).get("status") if isinstance(v, dict) else v for k, v in crit.items()},
            "per_kitchen": {s: {k: pick(v.get("primary")) for k, v in res["kitchens"][s]["contrasts"].items()} for s in res["kitchens"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("execute")
    x.add_argument("--namespace", required=True, choices=NAMESPACES)
    x.add_argument("--dry-run", action="store_true")
    x.add_argument("--steal-lock", action="store_true", help="wrapper only: the pass's lock of a dead (hang-killed) execute, under the sim lock")
    sc = sub.add_parser("score")
    sc.add_argument("--namespace", required=True, choices=NAMESPACES)
    sc.add_argument("--no-analysis", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "execute":
        return run_execute(a.namespace, dry_run=a.dry_run, steal_lock=a.steal_lock)
    E.install_perception_reader()
    return run_score(a.namespace, analyse=not a.no_analysis)


if __name__ == "__main__":
    sys.exit(main())
