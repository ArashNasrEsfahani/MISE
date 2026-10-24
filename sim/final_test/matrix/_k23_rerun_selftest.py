"""Offline selftest of the K2 render-window re-execution pass (k23_rerun.py), no simulator.

Uses the real rehearsal-k2-v1 main journals (read-only) with a synthetic 3-unit 'window' written to a temp folder that replaces the pass's
folders; the executor is replaced by synthetic exec / outcome records. Checks: only window units run, resume, steal-lock after a dead lock,
refusal of a window unit the rebuilt units do not know, re-scoring (non-window rows byte-equal to the main rows, the main execution
re-scored reproduces the main rows, window rows from the pass), refusal after a changed window record, and that no main journal changes."""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import k23_rerun as KR      # noqa: E402
import treatment as T       # noqa: E402

NS = "rehearsal-k2-v1"
ok_n = fail_n = 0


def check(name, cond):
    global ok_n, fail_n
    if cond:
        ok_n += 1
    else:
        fail_n += 1
        print("FAIL", name)


real_rerun = KR.docs_dir(NS)
real_before = os.path.exists(real_rerun)
main_before = {k: C.sha256_file(C.journal_path(NS, k)) for k in ("plan", "exec", "outcome", "events", "score", "capture")
               if os.path.exists(C.journal_path(NS, k))}
units_before = C.sha256_file(os.path.join(C.ns_docs(NS), "units.json"))
tmp = tempfile.mkdtemp(prefix="k23rerun_")
KR.docs_dir = lambda ns, root=None: os.path.join(tmp, ns)
KR.work_dir = lambda ns, root=None: os.path.join(tmp, "work", ns)
_paths = KR.paths

# a synthetic window: the first 3 main units (run 0) whose latest main row is terminal, one of them with a goal reached
mx = C.Journal(C.journal_path(NS, "exec")).latest()
main_score = C.Journal(C.journal_path(NS, "score")).rows()
scored_units = {r.get("unit") for r in main_score if r.get("unit")}
cand = sorted(k for k, r in mx.items() if r.get("status") in ("done", "blocked_state") and r.get("unit") in scored_units and k.endswith("|run0"))
WIN = set(cand[:3])
rec_path = os.path.join(tmp, "window.json")
with open(rec_path, "w", encoding="utf-8") as f:
    json.dump({"namespace": NS, "units": {k.split("|")[2]: [{"key": k}] for k in WIN}, "affected_units": 3}, f)
KR.window_record_path = lambda ns, root=None: rec_path
KR.WINDOW_UNITS_N = {NS: 3}
check("window_keys reads the record (3 keys)", KR.window_keys(NS) == WIN)

RerunExecute, XE = KR._execute_cls()


def synthetic(self, u, run):
    """do_cell stand-in: a done exec row + an outcome whose goal is NOT reached, through the phase's own _append (the commit order of do_cell)."""
    key = self.cell_key(u["unit"], run)
    self.event("exec_attempt_start", key=key, unit=u["unit"], run=run, attempt=0, epoch=0)
    outc = C.make_record("outcome", key=key, namespace=self.ns, unit=u["unit"], run=run, goal_ok=False, exec_ok=True, steps=[], pres=u["pres"],
                         attempt=0, sr=0.0, prims=3)
    rec = C.make_record("exec", key=key, namespace=self.ns, unit=u["unit"], run=run, attempt=0, pres=u["pres"], status="done",
                        state_check={"ok": True, "breaches": []}, audit={}, prims=3, wall_s=0.1, epoch=0, error=None, chooser=u["chooser"],
                        cells=[c["key"] for c in u["cells"]], snapshot_sha_before=None, snapshot_sha_after=None)
    rec.update(treatment=T.FT2_VERSION, treatment_id=(self.ident or {}).get("sha"), exec_policy=u.get("exec_policy"), briefing_log=[])
    self._append("outcome", outc)
    self._append("exec", rec)
    return rec


RerunExecute.do_cell = synthetic
info = RerunExecute(NS, out=lambda *_: None).run(dry_run=True)
check("dry run: 3 window units, exactly 3 to do", info["window_units"] == 3 and info["todo"].get(0) == 3 and sum(info["todo"].values()) == 3)
info = RerunExecute(NS, out=lambda *_: None).run()
check("run: 3 executed, complete", info["executed"] == 3 and info["complete"])
xp = KR.paths(NS)
xrows = C.Journal(xp["exec"]).rows()
check("exec rows: the window keys only, tagged with the pass id", {r["key"] for r in xrows} == WIN and all(r.get("pass_id") == KR.PASS_ID for r in xrows))
check("units.rerun.json written in the pass folder; main units.json unchanged",
      os.path.exists(xp["units"]) and C.sha256_file(os.path.join(C.ns_docs(NS), "units.json")) == units_before)
info2 = RerunExecute(NS, out=lambda *_: None).run()
check("resume: nothing left, still complete", info2["executed"] == 0 and info2["complete"])

os.makedirs(os.path.dirname(xp["lock"]), exist_ok=True)
open(xp["lock"], "w").write('{"pid": 999999}')
try:
    RerunExecute(NS, out=lambda *_: None).run()
    check("a left lock refuses a plain run", False)
except C.PhaseLocked:
    check("a left lock refuses a plain run", True)
p3 = RerunExecute(NS, out=lambda *_: None)
p3.steal = True
check("steal-lock run completes", p3.run()["complete"] and not os.path.exists(xp["lock"]))

bogus = set(WIN) | {f"{NS}|exec|{'0' * 32}|run0"}
try:
    RerunExecute(NS, window=bogus, out=lambda *_: None).run(dry_run=True)
    check("a window unit unknown to the rebuilt units is refused", False)
except XE.ExecAborted:
    check("a window unit unknown to the rebuilt units is refused", True)

rc = KR.run_score(NS, analyse=False)
check("score rc 0 (the main execution re-scored reproduces every main window row)", rc == 0)
rows = [json.loads(l) for l in open(xp["score"], encoding="utf-8")]
check("score: as many rows as the main sealed score", len(rows) == len(main_score))
win_units = {k.split("|")[2] for k in WIN}
outside = [(a, b) for a, b in zip(main_score, rows) if a.get("unit") not in win_units]
check("non-window rows are the main rows, unchanged", outside and all(a == b for a, b in outside))
inside = [b for b in rows if b.get("unit") in win_units]
check("window rows come from the pass (rerun flag, SR 0 as executed here)", inside and all(r.get("rerun") and r["sr"] == 0.0 for r in inside))
sm = json.load(open(xp["summary"], encoding="utf-8"))
check("summary: window rows counted, main and re-executed SR tables", sm["n_window_rows"] == len(inside) and set(sm["sr_pct"]) == {"main", "window_reexecuted"})
# Astra K23-RERUN-r1 item 3: a main done -> re-run blocked transition must not keep the main execution time in the analysis
import analyze as A                                                                                  # noqa: E402
k0 = sorted(WIN)[0]
u0 = k0.split("|")[2]
cell0 = next(r["cell"] for r in main_score if r.get("unit") == u0)
m_done = {"key": k0, "unit": u0, "run": 0, "status": "done", "attempt": 0, "wall_s": 28.07, "cells": [cell0]}
m_other = {"key": f"{NS}|exec|{'f' * 32}|run0", "unit": "f" * 32, "run": 0, "status": "done", "attempt": 0, "wall_s": 5.0, "cells": ["x"]}
r_blk = {"key": k0, "unit": u0, "run": 0, "status": "blocked_state", "attempt": 2, "wall_s": 13.0, "cells": [cell0], "pass_id": KR.PASS_ID}
mix = KR.analysis_exec_rows([m_done, m_other], [r_blk], WIN)
check("analysis exec rows: the substituted main row is gone, other main rows and the re-run rows kept",
      m_done not in mix and m_other in mix and r_blk in mix)
srow = {"key": "s", "cell": cell0, "unit": u0, "run": 0, "r": 0, "arm": "G0", "world": "canon", "scored": True,
        "failure_class": "exec_blocked_state", "exec_status": "blocked_state"}
eb = A.execution_block([srow], mix)
eb_bad = A.execution_block([srow], [m_done, m_other, r_blk])
check("execution_block: the unfiltered concatenation keeps the main 28.07 s (the defect this guards)", eb_bad.get("total_wall_s", 0) > 0)
check("execution_block: no time left from the substituted main 'done' row", eb.get("total_wall_s") == 0 and eb.get("unique_runs") == 0)
check("execution links: the re-run row links the cell", A.execution_link_problems([srow], mix) == [])
with open(rec_path, "a", encoding="utf-8") as f:
    f.write("\n")
check("a changed window record is refused at scoring", KR.run_score(NS, analyse=False) == 2)
main_after = {k: C.sha256_file(C.journal_path(NS, k)) for k in main_before}
check("no main journal changed", main_after == main_before)
check("the real pass folder was not created by the selftest", os.path.exists(real_rerun) == real_before)
shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok_n}/{ok_n + fail_n}")
sys.exit(1 if fail_n else 0)
