"""Offline selftest of the K23 Codex addendum execution + scoring path (k23_codex.py), no simulator, no Codex call.

Uses the real rehearsal-k2-v1 main journals (read-only) and the real rehearsal Sol smoke journal, COPIED into a temp folder that replaces the
addendum folders; the executor is replaced by synthetic exec / outcome records. Checks: dedupe (only Sol-only units run), completion binding,
resume, steal-lock after a dead lock, scoring sources, refusal after a changed Sol journal, and that no main journal changes."""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402
import k23_codex as KC      # noqa: E402
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


real_docs = KC.docs_dir(NS)
tmp = tempfile.mkdtemp(prefix="k23codex_")
real_before = {f: C.sha256_file(os.path.join(real_docs, f)) for f in sorted(os.listdir(real_docs)) if os.path.isfile(os.path.join(real_docs, f))}
main_before = {k: C.sha256_file(C.journal_path(NS, k)) for k in ("plan", "exec", "outcome", "events", "score") if os.path.exists(C.journal_path(NS, k))}
os.makedirs(os.path.join(tmp, NS))
for f in (f"plan.{KC.PASS_IDS['sol']}.jsonl", "events.jsonl"):
    shutil.copy(os.path.join(real_docs, f), os.path.join(tmp, NS, f))
KC.docs_dir = lambda ns, root=None: os.path.join(tmp, ns)
KC.work_dir = lambda ns, pid, root=None: os.path.join(tmp, "work", ns, pid)

CodexExecute, XE = KC._execute_cls()


def synthetic(self, u, run):
    """do_cell stand-in: a done exec row + an outcome (goal reached), through the phase's own _append (the commit order of do_cell)."""
    key = self.cell_key(u["unit"], run)
    self.event("exec_attempt_start", key=key, unit=u["unit"], run=run, attempt=0, epoch=0)
    outc = C.make_record("outcome", key=key, namespace=self.ns, unit=u["unit"], run=run, goal_ok=True, exec_ok=True, steps=[], pres=u["pres"],
                         attempt=0, sr=1.0, prims=3)
    rec = C.make_record("exec", key=key, namespace=self.ns, unit=u["unit"], run=run, attempt=0, pres=u["pres"], status="done",
                        state_check={"ok": True, "breaches": []}, audit={}, prims=3, wall_s=0.1, epoch=0, error=None, chooser=u["chooser"],
                        cells=[c["key"] for c in u["cells"]], snapshot_sha_before=None, snapshot_sha_after=None)
    rec.update(treatment=T.FT2_VERSION, treatment_id=(self.ident or {}).get("sha"), exec_policy=u.get("exec_policy"), briefing_log=[])   # as run_one (ft-v2)
    self._append("outcome", outc)
    self._append("exec", rec)
    return rec


CodexExecute.do_cell = synthetic
ph = CodexExecute(NS, out=lambda *_: None)
info = ph.run(dry_run=True)
check("dry run: 8 Sol units, 6 reused from the main rehearsal, 2 new", (info["sol_units"], info["sol_units_reused"], info["sol_units_new"]) == (8, 6, 2))
check("dry run: exactly the 2 new units are to do", info["todo"].get(0) == 2)
info = CodexExecute(NS, out=lambda *_: None).run()
check("run: 2 executed, complete", info["executed"] == 2 and info["complete"])
xp = KC.exec_paths(NS)
pc = [e for e in C.Journal(xp["events"]).rows() if e.get("event") == "phase_complete" and e.get("phase") == "execute"]
check("phase_complete bound to the Sol plan and units shas", pc and pc[-1].get("sol_plan_sha") == C.sha256_file(xp["xplan"])
      and pc[-1].get("units_sha") == C.sha256_file(xp["units"]) and pc[-1].get("reused") == 6)
info2 = CodexExecute(NS, out=lambda *_: None).run()
check("resume: nothing left to execute, still complete", info2["executed"] == 0 and info2["complete"])

# a dead lock (hang exit 75) blocks a plain run; --steal-lock (wrapper, under the sim lock) takes it
os.makedirs(os.path.dirname(xp["lock"]), exist_ok=True)
open(xp["lock"], "w").write('{"pid": 999999}')
try:
    CodexExecute(NS, out=lambda *_: None).run()
    check("a left lock refuses a plain run", False)
except C.PhaseLocked:
    check("a left lock refuses a plain run", True)
p3 = CodexExecute(NS, out=lambda *_: None)
p3.steal = True
check("steal-lock run completes", p3.run()["complete"] and not os.path.exists(xp["lock"]))

# scoring
rc = KC.run_score(NS)
check("score rc 0", rc == 0)
rows = [json.loads(l) for l in open(os.path.join(tmp, NS, "score.codex.jsonl"), encoding="utf-8")]
src = {}
for r in rows:
    src[r["exec_source"]] = src.get(r["exec_source"], 0) + 1
check("score: 8 Sol rows, 6 from reused main executions, 2 from the addendum", len(rows) == 8 and src == {"main": 6, "codex": 2})
check("score: arms X0 / X+M only, SR defined", {r["arm"] for r in rows} == {"X0", "X+M"} and all(r["sr"] in (0.0, 1.0) for r in rows))
sm = json.load(open(os.path.join(tmp, NS, "summary.codex.json"), encoding="utf-8"))
check("summary: by_arm + main-arm context + reuse count", set(sm["by_arm"]) == {"X0", "X+M"} and sm["executions_reused_from_main"] == 6)
with open(xp["xplan"], "a", encoding="utf-8") as f:
    f.write("\n")
check("a changed Sol journal is refused at scoring", KC.run_score(NS) == 2)
main_after = {k: C.sha256_file(C.journal_path(NS, k)) for k in main_before}
check("no main journal changed", main_after == main_before)
real_after = {f: C.sha256_file(os.path.join(real_docs, f)) for f in sorted(os.listdir(real_docs)) if os.path.isfile(os.path.join(real_docs, f))}
check("the real addendum folder is byte-identical before / after the selftest", real_after == real_before)
shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok_n}/{ok_n + fail_n}")
sys.exit(1 if fail_n else 0)
