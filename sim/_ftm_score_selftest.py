"""Selftest for sim/final_test/matrix/score.py (stage M7). Offline, synthetic journals in a temp root.

    python sim/_ftm_score_selftest.py
"""
from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                   # noqa: E402
import score as S                                    # noqa: E402

N = {"pass": 0, "fail": 0}
REAL_ROOT = C.ROOT


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception as e:                                        # noqa: BLE001
        print("   (other exception)", type(e).__name__, e)
        return False
    return False


MAN = C.load_manifest(REAL_ROOT)                             # A26 (K-B2b): the EFFECTIVE split (archived ids removed)
DATA = json.load(open(C.paths(REAL_ROOT)["dataset"], encoding="utf-8"))
ITEMS = {i["id"]: i for i in DATA["items"]}
DEV = [i for i in MAN["dev_pilot"] if ITEMS[i]["plan"]]
TEST_ITEM = MAN["test"][0]
NS = "rehearsal-v1"
LOG = lambda *a, **k: None                                    # noqa: E731


def make_root():
    r = tempfile.mkdtemp(prefix="ftm_score_")
    os.makedirs(os.path.join(r, "docs", "final-test"))
    os.makedirs(os.path.join(r, "sim", "dataset"))
    os.makedirs(os.path.join(r, "sim", "step4"))
    shutil.copy(C.paths(REAL_ROOT)["split"], C.paths(r)["split"])
    shutil.copy(C.paths(REAL_ROOT)["dataset"], C.paths(r)["dataset"])
    return r


def step(a, o=None, t=None, g=None):
    return {"action": a, "object": o, "to": t, "arg": g}


def ref_steps(item):
    return [step(s["action"], s.get("object"), s.get("to"), s.get("arg")) for s in ITEMS[item]["plan"]]


def pcell(arm, pres, r=0, ns=NS):
    return C.cell_id(ns, "plan", arm, pres, f"r{r}")


def plan_row(arm, pres, status="ok", plan=None, first=None, r=0, ns=NS):
    return C.make_record("plan", key=pcell(arm, pres, r, ns), namespace=ns, arm=arm, block="main", pres=pres, r=r, session=0,
                         request_sha="x", status=status, plan=plan, first_plan=first if first is not None else plan,
                         first_ok=status == "ok", repaired=False, calls=1, cached_calls=0, tokens_in=10, tokens_out=5,
                         cost_usd=0.001, model="m", scope="free")


def exec_pair(root, unit, run, cells, pres, status="done", goal_ok=True, exec_ok=True, steps=None, attempt=0, ns=NS, extra=None):
    key = f"{ns}|exec|{unit}|run{run}"
    e = C.make_record("exec", key=key, namespace=ns, unit=unit, run=run, attempt=attempt, pres=pres, status=status,
                      state_check={"ok": True, "breaches": []}, audit={"violations": 0}, prims=10, wall_s=1.0,
                      snapshot_sha_before=None, snapshot_sha_after=None, cells=cells, **(extra or {}))
    o = C.make_record("outcome", key=key, namespace=ns, unit=unit, run=run, goal_ok=goal_ok, exec_ok=exec_ok,
                      steps=steps if steps is not None else [{"i": 0, "action": "pickup", "ok": exec_ok, "prims": 3, "holding": None}])
    return e, o


def write(root, kind, rows, ns=NS):
    j = C.Journal(C.journal_path(ns, kind, root))
    for r in rows:
        j.append(r)
    return j


def _sha(p):
    return C.sha256_file(p) if os.path.exists(p) else None


def complete_events(root, ns=NS, plan=True, ex=True):
    j = C.Journal(C.journal_path(ns, "events", root))
    if plan:
        j.append(C.make_record("events", event="phase_complete", phase="plan", n=0, journal_sha=_sha(C.journal_path(ns, "plan", root))))
    if ex:
        j.append(C.make_record("events", event="phase_complete", phase="exec", n=0, journal_sha=_sha(C.journal_path(ns, "exec", root))))


def scenario(root):
    """4 dev items, G0 arm, EN. d0 good (shared with the P0 reference cell, REPEAT-style 3 runs), d1 rejected, d2 infra, d3 wrong plan,
    plus an empty accepted plan on d4 (goal false)."""
    d = DEV
    p = lambda i, lang="en": C.pres_id(i, lang)                   # noqa: E731
    plans = [plan_row("G0", p(d[0]), plan=ref_steps(d[0])),
             plan_row("G0", p(d[1]), status="rejected", plan=None, first=[step("fly", "Apple")]),
             plan_row("G0", p(d[2]), status="infra", plan=None, first=None),
             plan_row("G0", p(d[3]), plan=[step("search_for", "Apple"), step("open", "Fridge")]),
             plan_row("G0", p(d[4]), plan=[])]
    write(root, "plan", plans)
    ex, out = [], []
    cells0 = [pcell("G0", p(d[0])), pcell("P0", p(d[0]))]
    for run in (0, 1, 2):
        e, o = exec_pair(root, "u0", run, cells0, p(d[0]), goal_ok=True)
        ex.append(e); out.append(o)
    e, o = exec_pair(root, "u3", 0, [pcell("G0", p(d[3]))], p(d[3]), goal_ok=False, exec_ok=False,
                     steps=[{"i": 0, "action": "open", "ok": False, "prims": 4, "holding": None}])
    ex.append(e); out.append(o)
    e, o = exec_pair(root, "u4", 0, [pcell("G0", p(d[4]))], p(d[4]), goal_ok=False, exec_ok=True, steps=[])
    ex.append(e); out.append(o)
    write(root, "exec", ex)
    write(root, "outcome", out)
    complete_events(root)
    return d


# ================================================================================================ static
src = open(S.__file__, encoding="utf-8").read()
tree = ast.parse(src)
check("score.py parses", True)
opens = []
for fn in ast.walk(tree):
    if isinstance(fn, ast.FunctionDef):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open":
                opens.append((fn.name, ast.unparse(n.args[0]) if n.args else ""))
gold_open = [f for f, a in opens if a == "p"]
check("the gold file is opened only inside _gold_load", gold_open == ["_gold_load"], gold_open)
check("score.py never hard-codes the dataset path (uses common.paths)", "general.json" not in src)
check("score.py imports no simulator / planner module", not any(f"import {m}" in src for m in ("ai2thor", "thor_actions", "honest", "webapp", "anthropic", "ollama", "step5_live")))
check("score.py never reads an API key", "get_anthropic_key" not in src and "THESIS_ANTHROPIC" not in src)

# ================================================================================================ labels / metrics (pure)
g = [step("search_for", "Apple"), step("look_around"), step("pickup", "Apple"), step("scan_for", "Pot"), step("open", "Fridge")]
r = [step("pickup", "Apple"), step("open", "Fridge")]
check("search_for / look_around / scan_for stripped from the generated sequence", S.labels_of(g) == S.labels_of(r), S.labels_of(g))
check("... and from the reference sequence", S.labels_of([step("search_for", "Apple")] + r) == S.labels_of(r))
m = S.plan_metrics(g, r)
check("PC / TDS = 1.0 for an equivalent plan", m["pc"] == 1.0 and m["tds"] == 1.0, m)
m2 = S.plan_metrics([step("pickup", "Apple")], r)
check("PC = 0.5 for one missing step", abs(m2["pc"] - 0.5) < 1e-9, m2)
check("TDS is order-free: reversed plan has PC < 1 but TDS = 1",
      S.plan_metrics(list(reversed(r)), r)["pc"] < 1.0 and S.plan_metrics(list(reversed(r)), r)["tds"] == 1.0)
check("labels keep the destination and argument (action:object:to:arg)",
      S.step_label(step("pick_and_place", "Apple", "Fridge", None)) == "pick_and_place:Apple:Fridge:" and
      S.labels_of([step("pick_and_place", "Apple", "Fridge")]) != S.labels_of([step("pick_and_place", "Apple", "Sink")]))
check("no plan -> metrics None (a rejected plan has no generated sequence)", S.plan_metrics(None, r)["pc"] is None)
check("an empty generated plan against a non-empty reference scores 0", S.plan_metrics([], r)["pc"] == 0.0 and S.plan_metrics([], r)["tds"] == 0.0)

# slice realism
sl = lambda hold_before, ok=True: [{"action": "pickup", "object": "Knife", "ok": True, "prims": 2, "holding": hold_before},   # noqa: E731
                                   {"action": "slice", "object": "Apple", "ok": ok, "prims": 2, "holding": hold_before}]
# A26 item 6 (K-B2b 2026-10-13): the default rule is "a26" (only the ButterKnife is a cutting tool; any executed slice without it is
# flagged, whatever the goal). The old checks are kept below under rule="legacy" (history: rounds 1-5 / step 4).
check("A26: slice with a ButterKnife held is realistic", S.slice_realism(sl("ButterKnife|+1.0|2"), True)["unrealistic_slice"] is False)
check("A26: slice with the Knife held is NOT realistic (Knife is no cutting tool)", S.slice_realism(sl("Knife"), True)["unrealistic_slice"] is True)
check("A26: an executed empty-handed slice is flagged even when the goal is NOT met (executor fault, independent of goal)",
      S.slice_realism(sl(None), False)["unrealistic_slice"] is True and S.slice_realism(sl(None), True)["unrealistic_slice"] is True)
check("A26: a refused slice (no_cutting_tool) never ran, even if the step's tidy spent primitives",
      S.slice_realism([{"action": "slice", "object": "Apple", "ok": False, "prims": 3, "cause": "no_cutting_tool", "holding": None}], True)["n_slice"] == 0)
check("legacy: slice with a Knife held is realistic", S.slice_realism(sl("Knife"), True, rule="legacy")["unrealistic_slice"] is False)
check("legacy: slice empty-handed with the goal met is an unrealistic success", S.slice_realism(sl(None), True, rule="legacy")["unrealistic_slice"] is True)
check("legacy: slice empty-handed with the goal NOT met is not counted", S.slice_realism(sl(None), False, rule="legacy")["unrealistic_slice"] is False)
check("goal met without any slice step: not unrealistic", S.slice_realism([{"action": "pickup", "ok": True, "prims": 1, "holding": "Apple"}], True)["unrealistic_slice"] is False)
check("a slice step that never ran does not count",
      S.slice_realism([{"action": "slice", "ok": False, "prims": 0, "holding": None}], True)["n_slice"] == 0)
check("holding recorded before: the ButterKnife dropped by the slice step itself still counts as held",
      S.slice_realism([{"action": "pickup", "ok": True, "prims": 1, "holding": "ButterKnife"},
                       {"action": "slice", "ok": True, "prims": 1, "holding": None}], True)["unrealistic_slice"] is False)
check("unknown holding -> None (not guessed)", S.slice_realism([{"action": "slice", "ok": True, "prims": 1}], True)["unrealistic_slice"] is None)
check("start holding (setup) counts for step 0", S.slice_realism([{"action": "slice", "ok": True, "prims": 1}], True, "ButterKnife")["unrealistic_slice"] is False
      and S.slice_realism([{"action": "slice", "ok": True, "prims": 1}], True, "Knife")["unrealistic_slice"] is True)

# ================================================================================================ refusal: phase completion
root = make_root()
try:
    d = scenario(root)
    # exec not complete
    shutil.rmtree(os.path.dirname(C.journal_path(NS, "events", root)), ignore_errors=False) if False else None
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root, ex=False)
    check("refused without phase_complete exec", raises(lambda: S.run_score(NS, root=root, log=LOG), S.ScoreRefused))
    check("--partial without a stopping-rule event is refused", raises(lambda: S.run_score(NS, root=root, partial=True, log=LOG), S.ScoreRefused))
    check("nothing written by a refused run", not os.path.exists(C.journal_path(NS, "score", root)))
    for ev_ in ("ceiling", "pause_cost_guard"):                  # M-fix2: neither a refused block nor a cost pause is a stopping rule
        C.Journal(C.journal_path(NS, "events", root)).append(C.make_record("events", event=ev_, phase="plan"))
    check("--partial with only a ceiling / pause_cost_guard event is refused (a pause is not a stopping rule)",
          raises(lambda: S.run_score(NS, root=root, partial=True, log=LOG), S.ScoreRefused))
    C.Journal(C.journal_path(NS, "events", root)).append(C.make_record("events", event="stopping_rule", phase="plan", rule="incomplete_study",
                                                                       reason="budget", by="Arash"))
    r = S.run_score(NS, root=root, partial=True, log=LOG)
    seal = json.load(open(os.path.join(C.ns_docs(NS, root), "score.seal.json"), encoding="utf-8"))
    check("--partial with a recorded stopping rule scores and seals complete:false", r["status"] == "partial" and seal["complete"] is False and seal["partial"] is True, (r, seal))
finally:
    shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ full run
root = make_root()
try:
    d = scenario(root)
    r = S.run_score(NS, root=root, log=LOG)
    rows = C.Journal(C.journal_path(NS, "score", root)).rows()
    by = {(x["arm"], x["pres"], x["run"]): x for x in rows}
    p = lambda i, lang="en": C.pres_id(i, lang)               # noqa: E731
    check("run complete, sealed", r["status"] == "scored" and r["seal"]["complete"] is True and r["seal"]["n"] == len(rows), r)
    seal = S.verify_seal(NS, root)
    check("seal sha equals score.jsonl", seal["sha256"] == C.sha256_file(C.journal_path(NS, "score", root)))
    check("phase_complete score event appended with the journal sha",
          any(e.get("event") == "phase_complete" and e.get("phase") == "score" and e.get("journal_sha") == seal["sha256"]
              for e in C.Journal(C.journal_path(NS, "events", root)).rows()))
    check("27 -> expected row count: 3 runs x (G0,P0) of d0 + d1 + d2 + d3 + d4 = 10", len(rows) == 10, len(rows))
    a = by[("G0", p(d[0]), 0)]
    check("good plan: sr 1, strict 1, valid, not infra, scored", a["sr"] == 1.0 and a["strict_sr"] == 1.0 and a["valid"] and not a["infra"] and a["scored"])
    check("good plan = reference: PC / TDS 1.0 (first and final)", a["pc_final"] == 1.0 and a["pc_first"] == 1.0 and a["tds_final"] == 1.0)
    check("level / lang / family attached from gold and manifest", a["level"] == ITEMS[d[0]]["level"] and a["lang"] == "en" and bool(a["family"]), a)
    pz = by[("P0", p(d[0]), 0)]
    check("the reference-plan cell (listed by the exec record only) is scored and shares the unit", pz["unit"] == a["unit"] == "u0" and pz["sr"] == 1.0, pz)
    check("REPEAT runs 1, 2 kept but not the scored ones", by[("G0", p(d[0]), 1)]["scored"] is False and by[("G0", p(d[0]), 2)]["scored"] is False and a["scored"])
    rj = by[("G0", p(d[1]), 0)]
    check("rejected plan: sr 0, valid (not an INVALID episode), not infra, class plan_rejected, PC final None", rj["sr"] == 0 and rj["valid"] and not rj["invalid_episode"] and not rj["infra"] and
          rj["failure_class"] == "plan_rejected" and rj["pc_final"] is None, rj)
    inf = by[("G0", p(d[2]), 0)]
    check("infra plan: sr 0, infra flag set, class plan_infra", inf["sr"] == 0 and inf["infra"] and inf["failure_class"] == "plan_infra")
    w = by[("G0", p(d[3]), 0)]
    check("executed failing plan: sr 0, strict 0, valid (a real task failure, not infra)", w["sr"] == 0 and w["strict_sr"] == 0 and w["valid"] and not w["infra"] and w["goal_ok"] is False)
    check("search_for stripped from the generated plan before PC (so PC reflects the remaining step)", w["pc_final"] is not None and 0.0 <= w["pc_final"] <= 1.0)
    e0 = by[("G0", p(d[4]), 0)]
    check("empty plan scores 0 and is flagged empty-plan-success False", e0["sr"] == 0 and e0["plan_len"] == 0 and e0["empty_plan_success"] is False)
    check("no gold / planner secret fields leak into score rows (no goal / setup / reference plan)",
          not any(k in C.GOLD_KEYS for x in rows for k in x), [k for x in rows for k in x if k in C.GOLD_KEYS][:3])
    check("no score row is key-like", not any(C.key_leak(json.dumps(x, ensure_ascii=False)) for x in rows))
    # idempotent
    r2 = S.run_score(NS, root=root, log=LOG)
    check("second run is a no-op (already-done)", r2["status"] == "already-done" and len(C.Journal(C.journal_path(NS, "score", root)).rows()) == len(rows))
    # tamper
    with open(C.journal_path(NS, "score", root), "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": "x"}) + "\n")
    check("an edit after the seal is detected by verify_seal", raises(lambda: S.verify_seal(NS, root), S.ScoreRefused))
    check("... and by a re-run", raises(lambda: S.run_score(NS, root=root, log=LOG), S.ScoreIntegrity))
finally:
    shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ resume after an interrupt
root = make_root()
try:
    d = scenario(root)
    S.run_score(NS, root=root, log=LOG)
    full = C.Journal(C.journal_path(NS, "score", root)).rows()
    sp = C.journal_path(NS, "score", root)
    lines = open(sp, "rb").read().split(b"\n")
    keep = b"\n".join(lines[:4]) + b"\n" + lines[4][: len(lines[4]) // 2]          # 4 rows + a torn 5th line (crash mid-write)
    open(sp, "wb").write(keep)
    os.remove(os.path.join(C.ns_docs(NS, root), "score.seal.json"))
    r = S.run_score(NS, root=root, log=LOG)
    again = C.Journal(sp).rows()
    check("resume after a crash mid-write completes with no duplicate keys", len({x["key"] for x in again}) == len(again) == len(full), (len(again), len(full)))
    check("resume wrote exactly the missing rows", r["new"] == len(full) - 4, r)
    check("the torn line was moved aside", os.path.exists(sp + ".torn"))
finally:
    shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ integrity failures
root = make_root()
try:
    d = scenario(root)
    # an accepted plan whose outcome is missing
    write(root, "plan", [plan_row("G0", C.pres_id(d[5], "en"), plan=ref_steps(d[5]))])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    check("an accepted plan with no executed outcome is an integrity failure (never a silent 0)",
          raises(lambda: S.run_score(NS, root=root, log=LOG), S.ScoreIntegrity))
    r = S.run_score(NS, root=root, partial=True, log=LOG) if False else None
finally:
    shutil.rmtree(root, ignore_errors=True)

root = make_root()
try:
    d = scenario(root)
    e, o = exec_pair(root, "ub", 0, [], C.pres_id(d[1], "en"), status="invalid", goal_ok=False)
    # an exec row INVALID must be built without the outcome firewall complaining: status invalid is legal
    write(root, "exec", [e])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    check("an INVALID exec row aborts scoring", raises(lambda: S.run_score(NS, root=root, log=LOG), S.ScoreIntegrity))
finally:
    shutil.rmtree(root, ignore_errors=True)

root = make_root()
try:
    d = scenario(root)
    write(root, "plan", [plan_row("G0", C.pres_id(d[5], "en"), status="ceiling", plan=None)])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    check("a non-terminal 'ceiling' plan under a complete plan phase is refused", raises(lambda: S.run_score(NS, root=root, log=LOG), S.ScoreIntegrity))
finally:
    shutil.rmtree(root, ignore_errors=True)

root = make_root()
try:
    d = scenario(root)
    open(C.journal_path(NS, "plan", root), "ab").write(b"{broken json}\n" + json.dumps({"a": 1}).encode() + b"\n")
    os.remove(C.journal_path(NS, "events", root))
    check("a corrupt plan journal fails fast", raises(lambda: S.run_score(NS, root=root, log=LOG), C.JournalCorrupt))
finally:
    shutil.rmtree(root, ignore_errors=True)

# crash / blocked executions
root = make_root()
try:
    d = scenario(root)
    p = lambda i: C.pres_id(i, "en")                            # noqa: E731
    write(root, "plan", [plan_row("G0", p(d[5]), plan=ref_steps(d[5])), plan_row("G0", p(d[6]), plan=ref_steps(d[6]))])
    e1, o1 = exec_pair(root, "uc", 0, [pcell("G0", p(d[5]))], p(d[5]), status="crash", attempt=2, goal_ok=False)
    e2, o2 = exec_pair(root, "ub", 0, [pcell("G0", p(d[6]))], p(d[6]), status="blocked_state", attempt=2, goal_ok=False)
    write(root, "exec", [e1, e2])
    write(root, "outcome", [o1, o2])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    S.run_score(NS, root=root, log=LOG)
    rows = {(x["pres"]): x for x in C.Journal(C.journal_path(NS, "score", root)).rows() if x["arm"] == "G0"}
    check("terminal crash: sr 0, exec_failure (not planner infra), class exec_crash", rows[p(d[5])]["sr"] == 0 and rows[p(d[5])]["exec_failure"] and not rows[p(d[5])]["infra"] and rows[p(d[5])]["failure_class"] == "exec_crash")
    check("terminal blocked_state: sr 0, exec_failure (not planner infra), class exec_blocked_state", rows[p(d[6])]["exec_failure"] and not rows[p(d[6])]["infra"] and rows[p(d[6])]["failure_class"] == "exec_blocked_state")
    check("a crash row cannot score a success even if its outcome claimed goal_ok",
          S.score_cell(pcell("G0", p(d[5])), 0, plan_row=plan_row("G0", p(d[5]), plan=ref_steps(d[5])),
                       exec_row={"unit": "q", "status": "crash"}, outcome={"goal_ok": True, "exec_ok": True, "steps": []},
                       gold_item=ITEMS[d[5]], family="f", ns=NS)["sr"] == 0)
    # ---- A21: run classes and the X flags on a score row
    pr_x = plan_row("G0", p(d[5]), plan=ref_steps(d[5]))
    pr_x["attempts"] = [{"k": 0, "extra": {"isolation_violation": True}}, {"k": 1, "extra": {"identity_violation": True}}]
    rx = S.score_cell(pcell("G0", p(d[5])), 0, plan_row=pr_x, exec_row={"unit": "q", "status": "blocked_state", "attempt": 2}, outcome=None, gold_item=ITEMS[d[5]], family="f", ns=NS)
    check("A21 A/E: a blocked execution is run_class infra_exec; tool events and identity problems of the plan call are carried onto the score row",
          rx["run_class"] == "infra_exec" and rx["exec_attempt"] == 2 and rx["x_tool_events"] is True and rx["x_identity_problem"] is True and rx["sr"] == 0)
    rj = S.score_cell(pcell("G0", p(d[5])), 0, plan_row=plan_row("G0", p(d[5]), status="infra", plan=[]), exec_row=None, outcome=None, gold_item=ITEMS[d[5]], family="f", ns=NS)
    rr = S.score_cell(pcell("G0", p(d[5])), 0, plan_row=plan_row("G0", p(d[5]), status="rejected", plan=[]), exec_row=None, outcome=None, gold_item=ITEMS[d[5]], family="f", ns=NS)
    check("A21 A: a planner transport failure is infra_planner; a rejected plan is the agent's, not infrastructure",
          rj["run_class"] == "infra_planner" and rj["infra"] and rr["run_class"] == "agent" and not rr["infra"] and not rr["exec_failure"])
    # ---- A27 / A25 item 3: plan-only cells (exploratory Codex) get plan metrics and NO SR (None, never 0)
    _xa = "X-gpt-6-luna0"
    rpo = S.score_cell(pcell(_xa, p(d[5])), 0, plan_row=plan_row(_xa, p(d[5]), plan=ref_steps(d[5])), exec_row=None, outcome=None,
                       gold_item=ITEMS[d[5]], family="f", ns=NS, plan_only=True)
    rpj = S.score_cell(pcell(_xa, p(d[5])), 0, plan_row=plan_row(_xa, p(d[5]), status="rejected", plan=[]), exec_row=None, outcome=None,
                       gold_item=ITEMS[d[5]], family="f", ns=NS, plan_only=True)
    check("A27: a plan-only cell has sr / strict_sr None (not 0), executed False, PC / TDS of the plan, failure_class plan_only; a rejected one keeps plan_rejected",
          rpo["plan_only"] and rpo["sr"] is None and rpo["strict_sr"] is None and not rpo["executed"] and rpo["failure_class"] == "plan_only"
          and rpo["pc_final"] == 1.0 and rpo["plan_valid"] and rpj["failure_class"] == "plan_rejected" and not rpj["plan_valid"], (rpo, rpj))
    _pd = S.plan_diagnostics([rpo, rpj], NS)
    check("A27: plan_diagnostics summarises per arm (valid rate 0.5 over the two rows), carries the rows, never an SR",
          _pd["per_arm"][_xa]["n"] == 2 and _pd["per_arm"][_xa]["plan_valid"] == 0.5 and len(_pd["rows"]) == 2 and "sr" not in _pd["per_arm"][_xa], _pd["per_arm"])
finally:
    shutil.rmtree(root, ignore_errors=True)

# retried exec: latest record per key wins
root = make_root()
try:
    d = scenario(root)
    p = lambda i: C.pres_id(i, "en")                            # noqa: E731
    write(root, "plan", [plan_row("G0", p(d[5]), plan=ref_steps(d[5]))])
    e1, o1 = exec_pair(root, "ur", 0, [pcell("G0", p(d[5]))], p(d[5]), status="crash", attempt=0, goal_ok=False)
    e2, o2 = exec_pair(root, "ur", 0, [pcell("G0", p(d[5]))], p(d[5]), status="done", attempt=1, goal_ok=True)
    write(root, "exec", [e1, e2])
    write(root, "outcome", [o1, o2])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    S.run_score(NS, root=root, log=LOG)
    x = [r for r in C.Journal(C.journal_path(NS, "score", root)).rows() if r["pres"] == p(d[5])][0]
    check("re-run after a crash: the later successful attempt is the one scored", x["sr"] == 1.0 and x["exec_status"] == "done", x)
finally:
    shutil.rmtree(root, ignore_errors=True)

# unit recomputation fallback (exec records without `cells`)
root = make_root()
try:
    d = scenario(root)
    p = lambda i: C.pres_id(i, "en")                            # noqa: E731
    g = S._gold_load(root)[0]
    plan = ref_steps(d[6])
    unit = C.execution_unit_key(plan=plan, pres=p(d[6]), chooser="prior", snapshot_sha=None,
                                restored_state=S._restored_state("canon", g[d[6]]["setup"], None), summary_sha="sumsha")
    write(root, "capture", [C.make_record("capture", key="k", namespace=NS, pres=p(d[6]), item=d[6], lang="en", level="L1", world="canon",
                                          capture_run=1, instruction="t", holding=None, summary=[], summary_sha="sumsha", blocks={}, requests={},
                                          state_ref="s", world_hash="h", audit={})])
    write(root, "plan", [plan_row("G0", p(d[6]), plan=plan)])
    e, o = exec_pair(root, unit, 0, [], p(d[6]), goal_ok=True)
    e.pop("cells")
    write(root, "exec", [e])
    write(root, "outcome", [o])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    S.run_score(NS, root=root, log=LOG)
    x = [r for r in C.Journal(C.journal_path(NS, "score", root)).rows() if r["pres"] == p(d[6])][0]
    check("unit key recomputed from capture + gold setup matches the executor's and finds the outcome", x["sr"] == 1.0 and x["unit"] == unit, x)
finally:
    shutil.rmtree(root, ignore_errors=True)

# unrealistic slice through a full cell
root = make_root()
try:
    d = scenario(root)
    p = lambda i: C.pres_id(i, "en")                            # noqa: E731
    write(root, "plan", [plan_row("G0", p(d[7]), plan=ref_steps(d[7]))])
    e, o = exec_pair(root, "us", 0, [pcell("G0", p(d[7]))], p(d[7]), goal_ok=True,
                     steps=[{"i": 0, "action": "slice", "object": "Apple", "ok": False, "prims": 3, "holding": None}], exec_ok=False)
    write(root, "exec", [e])
    write(root, "outcome", [o])
    os.remove(C.journal_path(NS, "events", root))
    complete_events(root)
    S.run_score(NS, root=root, log=LOG)
    x = [r for r in C.Journal(C.journal_path(NS, "score", root)).rows() if r["pres"] == p(d[7])][0]
    check("goal met by an empty-handed slice: sr 1 (simulator-goal success), strict 0, unrealistic_slice True",
          x["sr"] == 1.0 and x["strict_sr"] == 0.0 and x["unrealistic_slice"] is True, x)
finally:
    shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ test-item guard
root = make_root()
try:
    write(root, "plan", [plan_row("G0", C.pres_id(TEST_ITEM, "en"), plan=[], ns="test-v1")], ns="test-v1")
    write(root, "exec", [], ns="test-v1")
    check("a test item is refused in test-v1 without --allow-test / freeze / families",
          raises(lambda: S.run_score("test-v1", root=root, log=LOG), C.TestItemRefused))
    check("... even with --allow-test (no valid freeze.json in the root)",
          raises(lambda: S.run_score("test-v1", root=root, allow_test=True, log=LOG), C.TestItemRefused))
    check("a test item in rehearsal-v1 is refused",
          raises(lambda: (write(root, "plan", [plan_row("G0", C.pres_id(TEST_ITEM, "en"), plan=[])]), S.run_score(NS, root=root, log=LOG)), C.TestItemRefused))
    check("nothing was scored for the refused runs", not os.path.exists(C.journal_path("test-v1", "score", root)))
    check("CLI returns 2 on a refusal", S.main(["--namespace", "test-v1"], root=root) == 2)
finally:
    shutil.rmtree(root, ignore_errors=True)

# gold sha must match the manifest
root = make_root()
try:
    d = scenario(root)
    with open(C.paths(root)["dataset"], "ab") as f:
        f.write(b" ")
    check("a changed gold dataset is refused (sha vs manifest)", raises(lambda: S.run_score(NS, root=root, log=LOG), S.ScoreRefused))
finally:
    shutil.rmtree(root, ignore_errors=True)

# lock
root = make_root()
try:
    d = scenario(root)
    with C.PhaseLock(NS, "score", root):
        check("a second scorer is locked out", raises(lambda: S.run_score(NS, root=root, log=LOG), C.PhaseLocked))
finally:
    shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ gate 12.5
root = make_root()
try:
    d = scenario(root)
    p = lambda i: C.pres_id(i, "en")                            # noqa: E731
    # reference-plan cells for dev items, plus an empty-plan control that succeeds (a defect) / fails (correct)
    ref_items = DEV[:4]
    base = []
    for k, it in enumerate(ref_items):
        base.append({"id": it, "status": "ok" if k else "FAIL", "goal_ok": bool(k % 2 == 0)})
    open(os.path.join(root, S.STEP4_BASELINE), "w", encoding="utf-8").write("\n".join(json.dumps(b) for b in base) + "\n")

    def build(now_ok, empty_ok):
        write(root, "plan", [])
        ex, out = [], []
        for k, it in enumerate(ref_items):
            e, o = exec_pair(root, f"p{k}", 0, [pcell("P0", p(it))], p(it), goal_ok=now_ok[k])
            ex.append(e); out.append(o)
        e, o = exec_pair(root, "pe", 0, [pcell("P0", p(DEV[8]))], p(DEV[8]), goal_ok=empty_ok, steps=[])
        e["plan"] = []
        ex.append(e); out.append(o)
        return ex, out

    # drop the scenario rows: this namespace has only P0 cells
    for k in ("plan", "exec", "outcome", "events"):
        os.remove(C.journal_path(NS, k, root))
    for k, ok in (("good", [False, False, True, False]),):
        ex, out = build([bool(b["goal_ok"]) for b in base], False)
        write(root, "exec", ex)
        write(root, "outcome", out)
        complete_events(root)
        S.run_score(NS, root=root, log=LOG)
    res = S.evaluator_check(NS, root=root)
    check("12.5 passes: P0 reproduces step 4 and the empty plan scores 0", res["pass"] and res["agreement"] == 1.0 and res["empty_plan_rows"] >= 1, res)
    check("12.5 writes evaluator_check.json", os.path.exists(os.path.join(C.ns_docs(NS, root), "evaluator_check.json")))
    # flip two results: agreement 0.5 < 0.8 -> fail
    sp = C.journal_path(NS, "score", root)
    rows = C.Journal(sp).rows()
    for x in rows:
        if x["arm"] == "P0" and x["item"] in ref_items[:2]:
            x["goal_ok"] = not x["goal_ok"]
    open(sp, "w", encoding="utf-8").write("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    res2 = S.evaluator_check(NS, root=root, write=False)
    check("12.5 fails when the reference plans no longer reproduce step 4", (not res2["pass"]) and len(res2["disagreements"]) == 2, res2)
    for x in rows:
        if x["arm"] == "P0" and x["item"] in ref_items[:2]:
            x["goal_ok"] = not x["goal_ok"]
        if x.get("plan_len") == 0:
            x["goal_ok"] = True
    open(sp, "w", encoding="utf-8").write("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    res3 = S.evaluator_check(NS, root=root, write=False)
    check("12.5 fails when the empty plan 'succeeds'", (not res3["pass"]) and res3["empty_plan_successes"], res3)
    for x in rows:
        if x.get("plan_len") == 0:
            x["goal_ok"] = False
    rows = [x for x in rows if x.get("plan_len") != 0]
    open(sp, "w", encoding="utf-8").write("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    res4 = S.evaluator_check(NS, root=root, write=False)
    check("12.5 requires an empty-plan control row", (not res4["pass"]) and any("empty-plan" in r for r in res4["reasons"]), res4)
    check("... unless explicitly waived", S.evaluator_check(NS, root=root, write=False, require_empty_plan=False)["pass"])
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
