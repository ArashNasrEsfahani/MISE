"""Selftest for FREEZE-GATES-FT2 (2026-10-15): the empty-plan control (E0) and the arm subset (--only-arms) of execute.py, the A26 12.5
registration guard, and the two new gate readers of gate_evidence.py (G-iso and 12.5 under the FINAL identity, namespace freeze-gates-ft2).
Offline: no simulator, no WSL, no Ollama, no API; a fake executor stands in for step5_live.LiveExecutor (temporary roots only).

    python sim/_ftm_freeze_gates_selftest.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                   # noqa: E402
import execute as X                                  # noqa: E402
import fg_runner as FG                               # noqa: E402
import gate_evidence as GE                           # noqa: E402
import score as S                                    # noqa: E402
from final_test import reset_gate as rg              # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


FIN = GE.FT2_FINAL_IDENTITY
OTHER = "9b26a99f" + "0" * 56

# ================================================================================================ 0 the namespace and the arm
print("== 0 namespace + arm")
ns = C.NAMESPACES.get(GE.FG_NS) or {}
check("0.1 freeze-gates-ft2 is a registered ft-v2 dev namespace, Gemma only (0 USD ceilings), no archived arm",
      ns.get("items") == "dev_pilot" and ns.get("treatment") == "ft-v2" and ns.get("ceilings") == {"core": 0.0, "total": 0.0} and not ns.get("archived_arms"))
import treatment as T                                # noqa: E402
check("0.2 it is NOT a draft-briefing / STRICT namespace (so it gets the final identity)",
      GE.FG_NS not in T.DEV_DRAFT_NAMESPACES and GE.FG_NS not in T.STRICT_NAMESPACES)
ident = T.identity(GE.FG_NS)["sha"]
check("0.3 its live identity == test-v1 == the registered final identity 7a2ee211", ident == T.identity("test-v1")["sha"] == FIN, ident[:12])
check("0.4 E0 = an empty-plan control arm: no call, prompt none, prior chooser, no hook B, free scope, not in any matrix roster",
      C.arm_info("E0") == {"planner": "empty", "prompt": "none", "chooser": "prior"} and not T.gets_hook("E0") and C.scope_of("E0", "main") == "free"
      and "E0" not in C.MAIN_ARMS and "E0" not in X.P_ARMS and not C.repeat_exec_arm("E0") and FG.EMPTY_ARM == "E0")
check("0.5 capture.py stays at its signed pin (4378ee70); the runner only subclasses execute.ExecutePhase (no edit of execute.py by this stage)",
      C.sha256_file(os.path.join(HERE, "final_test", "matrix", "capture.py")).startswith("4378ee70")
      and all(hasattr(X.ExecutePhase, m) for m in ("build", "run_one", "run", "_finish", "g_iso")) and hasattr(X, "build_units"))
_src = open(FG.__file__, encoding="utf-8").read()
import ast                                           # noqa: E402
check("0.6 fg_runner.py: python 3.10 syntax, LF, no top-level simulator import", chr(13) not in _src and bool(ast.parse(_src, feature_version=(3, 10)))
      and not any(m in _src.split('def ')[0] for m in ("import step5_live", "import ai2thor", "import capture")))
check("0.7 the capture entry refuses any other namespace (and never touches capture.py on disk)",
      FG.main(["capture", "--namespace", "rehearsal-v3", "--items", "dev"]) == 2 and FG.main(["capture", "--items", "dev"]) == 2)

# ================================================================================================ 1 execute.py: E0, --only-arms, the registration guard
print("== 1 execute.py")
DEV = ["g-l1-off-stove-knob", "g-l3-tool-flip"]
OPEN = {"action": "open", "object": "Fridge", "to": None, "arg": None}
TOL = {"tau_pos_m": 0.05, "tau_rot_deg": 5.0}
ITEMS = {d: {"id": d, "level": f"L{i + 1}", "en": f"do {d}", "fa": f"do-fa {d}", "setup": [], "plan": [dict(OPEN)]} for i, d in enumerate(DEV)}
AGENT = {"position": [1, 0.9, 1], "rotation": [0, 90, 0], "horizon": 0, "standing": True}


def state_for(pres):
    objs = [{"name": "Fridge_1", "objectId": "Fridge|+1|0|1", "objectType": "Fridge", "position": {"x": 2.0, "y": 0, "z": 1.0},
             "rotation": {"x": 0, "y": 0, "z": 0}, "openable": True, "receptacle": True, "isOpen": False}]
    summ = [f"I see a fridge ({pres})"]
    s0 = rg.capture_state(objs, AGENT, summary=summ, seen_types=["Fridge"], labels={"goal_clauses": {"x": False}}, feasibility={})
    return rg.capture_state(objs, AGENT, summary=summ, seen_types=["Fridge"], labels={"goal_clauses": {"x": False}}, feasibility=rg.feasibility_of(s0, []))


class FakeEx:
    supports_reloc = False
    frame, map_rev = "f0", "m0"

    def __init__(self):
        self.state_check = self.plan_hook = None
        self.calls, self.last = [], None

    def close(self):
        pass

    def execute(self, req):
        pres = f"canon:{req.item['id']}:{req.lang}"
        summ = [f"I see a fridge ({pres})"]
        post = {"summary": summ, "summary_sha": C.sha256_text("\n".join(summ)), "holding": None, "events": [], "sweep_prims": 5}
        chk = self.state_check(state_for(pres), post)
        assert chk.get("ok"), chk
        steps = self.plan_hook(req, req.item, None, req.memory, post=post)["steps"]
        self.calls.append((pres, len(steps)))
        self.last = {"adapter": SimpleNamespace(stream=[]), "chooser": None, "skills": None}
        goal = any(s["action"] == "open" for s in steps)                  # the goal is false at reset: only a real step reaches it
        rows = [{"i": i, "action": s["action"], "object": s.get("object"), "to": s.get("to"), "ok": True, "cause": None, "retries": 0, "prims": 3}
                for i, s in enumerate(steps)]
        return SimpleNamespace(inapplicable=None, sr=1.0 if goal else 0.0, goal_ok=goal, steps=5.0, initial_state_hash="wh", found=None, retries=0,
                               walking_m=0.0, opens=0, stale_visits=0, gate_pass=None, applied=None,
                               extra={"core": {"goal_ok": goal, "exec_ok": True, "prims": 5 + 3 * len(steps),
                                               "audit": {"violations": 0, "counts": {"oracle": 0, "map-build": 0}}, "n_events": 1},
                                      "steps": rows, "summary_sha": post["summary_sha"], "sweep_prims": 5})


def mkroot(tmp):
    for sub in ("docs/final-test", "sim/final_test/snapshots/o0/k10", "sim/step4"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    for f in ("split_manifest.json", "preregistration.md", "families_approved.json"):
        shutil.copy(os.path.join(C.ROOT, "docs", "final-test", f), os.path.join(tmp, "docs", "final-test", f))
    open(os.path.join(tmp, "sim", "final_test", "snapshots", "o0", "k10", "db"), "wb").write(b"snap")
    return tmp


def seed(root, ns="rehearsal-v1"):
    cj, pj, ej = (C.Journal(C.journal_path(ns, k, root)) for k in ("capture", "plan", "events"))
    os.makedirs(os.path.join(C.ns_docs(ns, root), "states"), exist_ok=True)
    pres_all = []
    for d in DEV:
        for lang in ("en", "fa"):
            pres = C.pres_id(d, lang)
            pres_all.append(pres)
            ref = f"states/{pres.replace(':', '_')}.c1.json"
            json.dump(state_for(pres), open(os.path.join(C.ns_docs(ns, root), ref), "w", encoding="utf-8"))
            summ = [f"I see a fridge ({pres})"]
            cj.append(C.make_record("capture", key=C.cell_id(ns, "capture", pres, "c1"), namespace=ns, pres=pres, item=d, lang=lang, level=ITEMS[d]["level"],
                                    world="canon", capture_run=1, instruction=ITEMS[d][lang], holding=None, summary=summ,
                                    summary_sha=C.sha256_text("\n".join(summ)), blocks={}, requests={}, state_ref=ref, world_hash="wh", audit={"violations": 0}))
            pj.append(C.make_record("plan", key=C.cell_id(ns, "plan", "G0", pres, "r0"), namespace=ns, arm="G0", block="main", pres=pres, r=0, session=0,
                                    request_sha="x", status="ok", plan=[OPEN, OPEN], first_plan=[OPEN, OPEN], first_ok=True, repaired=False, calls=1,
                                    cached_calls=0, tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", scope="free"))
    ej.append(C.make_record("events", event="phase_complete", phase="capture", n=1, journal_sha="s"))
    ej.append(C.make_record("events", event="stopping_rule", phase="plan", rule="incomplete_study", reason="selftest", by="selftest"))
    return pres_all


def write_reg(root, pres, **over):
    reg = {"schema": "ftm.a26-12.5-registration/1", "namespace": GE.FG_NS, "treatment_identity": FIN, "presentations": pres,
           "items": [{"item": d, "expected_outcome": True} for d in DEV], "expected_changes": [], "variance_allowance": {"min_agreement": 0.80},
           "written_utc": "2026-10-15T00:00:00Z"}
    reg.update(over)
    reg["sha256"] = C.self_sha(reg)
    p = os.path.join(root, *FG.A26_125_REGISTRATION)
    open(p, "w", encoding="utf-8").write(json.dumps(reg))
    return reg, p


tmp = tempfile.mkdtemp(prefix="ftm-fg-")
try:
    root = mkroot(os.path.join(tmp, "r1"))
    pres = seed(root)
    holder = []

    def mk(**kw):
        def factory(ph):
            e = FakeEx()
            holder.append(e)
            return e
        return FG.FGExecutePhase("rehearsal-v1", root=root, executor_factory=factory, memory_opener=None, items=ITEMS, tolerances=TOL,
                              partial=True, out=lambda s: None, **kw)
    ph = mk(p_arms=("P0", "E0"), only_arms=("P0", "E0"))
    ph.preflight()
    b = ph.build()
    U = b["units"]
    e0 = [u for u in U.values() if any(c["arm"] == "E0" for c in u["cells"])]
    p0 = [u for u in U.values() if any(c["arm"] == "P0" for c in u["cells"])]
    check("1.1 E0 builds one unit per presentation with the EMPTY plan; P0 keeps the reference plan", len(e0) == len(pres) and all(u["plan"] == [] for u in e0)
          and len(p0) == len(pres) and all(u["plan"] == C.norm_plan([OPEN]) for u in p0))
    check("1.2 an E0 unit never shares with a P0 / G0 unit (different plan -> different unit key)", not ({u["unit"] for u in e0} & {u["unit"] for u in p0}))
    check("1.3 --only-arms P0,E0: no G0 cell is executed (units hold only P0 / E0 cells), each runs once",
          all(c["arm"] in ("P0", "E0") for u in U.values() for c in u["cells"]) and all(u["runs"] == 1 for u in U.values())
          and json.load(open(ph.units_path, encoding="utf-8")).get("only_arms") == ["P0", "E0"])
    ph_full = mk(p_arms=("P0", "E0"))
    ph_full.preflight()
    check("1.4 without --only-arms the G0 units are still built (the filter changes nothing else)",
          any(c["arm"] == "G0" for u in ph_full.build()["units"].values() for c in u["cells"]))
    try:
        ph.run()
        refused = False
    except X.ExecAborted as e:
        refused = "registration" in str(e)
    ex_rows = C.Journal(C.journal_path("rehearsal-v1", "exec", root)).rows()
    check("1.5 an E0 run without the A26 12.5 registration is refused before any execution", refused and not ex_rows and not holder)
    reg, rp = write_reg(root, pres)
    bad = dict(reg, presentations=pres[:1])
    open(rp, "w", encoding="utf-8").write(json.dumps(bad))
    try:
        mk(p_arms=("P0", "E0"), only_arms=("P0", "E0")).run()
        refused2 = False
    except X.ExecAborted:
        refused2 = True
    check("1.6 an altered registration (self sha) is refused too", refused2 and not C.Journal(C.journal_path("rehearsal-v1", "exec", root)).rows())
    reg, rp = write_reg(root, pres)
    info = mk(p_arms=("P0", "E0"), only_arms=("P0", "E0")).run()
    ev = C.Journal(C.journal_path("rehearsal-v1", "events", root)).rows()
    ex_rows = C.Journal(C.journal_path("rehearsal-v1", "exec", root)).rows()
    oc = {r["key"]: r for r in C.Journal(C.journal_path("rehearsal-v1", "outcome", root)).rows()}
    regev = [e for e in ev if e.get("event") == "a26_12_5_registration"]
    check("1.7 the run records the registration sha as an event BEFORE its first execution row",
          len(regev) == 1 and regev[0]["sha256"] == C.sha256_file(rp) and regev[0]["ts"] <= min(r["ts"] for r in ex_rows))
    check("1.8 all 2 x 4 cells executed; an arm subset never writes phase_complete (arm_subset_complete instead)",
          info["executed"] == 2 * len(pres) and info["complete"] is False and not any(e.get("event") == "phase_complete" and e.get("phase") == "execute" for e in ev)
          and any(e.get("event") == "arm_subset_complete" for e in ev))
    e0_rows = [r for r in ex_rows if any("|E0|" in c for c in r["cells"])]
    p0_rows = [r for r in ex_rows if any("|P0|" in c for c in r["cells"])]
    check("1.9 E0 exec rows carry plan [] (and only they); the executor got 0 steps for E0, 1 for P0",
          len(e0_rows) == len(pres) and all(r.get("plan") == [] for r in e0_rows) and all("plan" not in r for r in p0_rows)
          and sorted(n for _, n in holder[-1].calls) == [0] * len(pres) + [1] * len(pres))
    gold = {d: {"plan": [OPEN], "level": ITEMS[d]["level"], "setup": []} for d in DEV}
    sc = []
    for r in e0_rows + p0_rows:
        cell = r["cells"][0]
        o = oc.get(r["key"])
        sc.append(S.score_cell(cell, 0, plan_row=None, exec_row=r, outcome=o, gold_item=gold[C.parse_pres(r["pres"])["item"]], family=None,
                               ns="rehearsal-v1", plan_steps_for_metrics=S._plan_of_exec(r, o)))
    se = [x for x in sc if x["arm"] == "E0"]
    sp = [x for x in sc if x["arm"] == "P0"]
    check("1.10 score.py (unchanged) scores E0 as executed empty-plan rows (plan_len 0, goal_ok false); P0 rows keep plan_len None",
          all(x["executed"] and x["plan_len"] == 0 and x["goal_ok"] is False and not x["empty_plan_success"] for x in se)
          and all(x["executed"] and x["goal_ok"] for x in sp))
    j = C.Journal(C.journal_path("rehearsal-v1", "score", root))
    for x in sc:
        j.append(x)
    ec = S.evaluator_check("rehearsal-v1", root=root, write=False, baseline_path=os.path.join(root, "nobaseline.jsonl"))
    check("1.11 score.evaluator_check finds the E0 rows as its empty-plan rows (0 successes)", ec["empty_plan_rows"] == len(pres) and ec["empty_plan_successes"] == [])
    check("1.12 a26_registration() on the REAL repo file is intact", FG.a26_registration(None)["self_sha256"]
          == json.load(open(os.path.join(C.ROOT, *FG.A26_125_REGISTRATION), encoding="utf-8"))["sha256"])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- (d) g_iso: the report is re-sealed with the identity it ran under (the frozen g_iso itself is stubbed: no simulator)
tmp2 = tempfile.mkdtemp(prefix="ftm-fg2-")
_orig = X.ExecutePhase.g_iso
try:
    X.ExecutePhase.g_iso = lambda self, n=12, conds=None: dict(schema="ftm.g_iso/2", status="pass", namespace=self.ns, n=n, sha256="stale")
    ph = FG.FGExecutePhase("rehearsal-v1", root=mkroot(os.path.join(tmp2, "r")), items=ITEMS, tolerances=TOL)
    ph.ft2, ph.ident = True, {"sha": FIN}
    rep = ph.g_iso(n=12)
    disk = json.load(open(os.path.join(ph.docs, "g_iso.json"), encoding="utf-8"))
    check("1.13 g_iso: treatment_id written and re-sealed (self sha) on disk + a g_iso_identity event",
          rep["treatment_id"] == FIN and disk == rep and disk["sha256"] == C.self_sha(disk)
          and any(e.get("event") == "g_iso_identity" for e in C.Journal(C.journal_path("rehearsal-v1", "events", ph.root)).rows()))
finally:
    X.ExecutePhase.g_iso = _orig
    shutil.rmtree(tmp2, ignore_errors=True)

# ================================================================================================ 2 G-iso reader
print("== 2 G-iso (final identity)")


def giso(**over):
    g = {"schema": "ftm.g_iso/2", "rule": "A20.1", "namespace": GE.FG_NS, "n": 12, "n_registered": 12, "status": "pass", "diff": [], "blocked": [],
         "short": False, "instrumentation_missing": {}, "executions": {"all_real": True}, "audit_sensitivity": {"ok": True},
         "hook_b_on_zero_arm": [], "treatment_id": FIN}
    g.update(over)
    g["sha256"] = C.self_sha(g)
    return g


CAP = [{"treatment_id": FIN}] * 3
PLN = [{"treatment_id": FIN}] * 2
st, why = GE.g_iso_final_verdict(None, final_sha=FIN, cap_rows=CAP, plan_rows=PLN)
check("2.1 missing report -> PENDING", st == "PENDING", why)
st, why = GE.g_iso_final_verdict(giso(), final_sha=FIN, cap_rows=CAP, plan_rows=PLN)
check("2.2 a pass report bound to 7a2ee211 -> PASS", st == "PASS" and not why, why)
check("2.3 report under another identity -> FAIL", GE.g_iso_final_verdict(giso(treatment_id=OTHER), final_sha=FIN, cap_rows=CAP, plan_rows=PLN)[0] == "FAIL")
check("2.4 capture rows under another identity -> FAIL",
      GE.g_iso_final_verdict(giso(), final_sha=FIN, cap_rows=CAP + [{"treatment_id": OTHER}], plan_rows=PLN)[0] == "FAIL")
check("2.5 the live final identity drifted from 7a2ee211 -> FAIL", GE.g_iso_final_verdict(giso(), final_sha=OTHER, cap_rows=CAP, plan_rows=PLN)[0] == "FAIL")
alt = giso()
alt["diff"] = ["canon:x:en"]
check("2.6 an altered report (self sha) -> FAIL", GE.g_iso_final_verdict(alt, final_sha=FIN, cap_rows=CAP, plan_rows=PLN)[0] == "FAIL")
check("2.7 an inconclusive report is not passed -> FAIL",
      GE.g_iso_final_verdict(giso(status="inconclusive", inconclusive_reasons=["blocked run"]), final_sha=FIN, cap_rows=CAP, plan_rows=PLN)[0] == "FAIL")
check("2.8 another namespace's report -> FAIL", GE.g_iso_final_verdict(giso(namespace="rehearsal-v3"), final_sha=FIN, cap_rows=CAP, plan_rows=PLN)[0] == "FAIL")
check("2.9 no capture rows -> FAIL", GE.g_iso_final_verdict(giso(), final_sha=FIN, cap_rows=[], plan_rows=PLN)[0] == "FAIL")

# ================================================================================================ 3 12.5 reader
print("== 3 12.5 (A26, final identity)")
ITM = ["g-a", "g-b", "g-c", "g-d", "g-e"]
PRES = [f"canon:{i}:{l}" for i in ITM for l in ("en", "fa")]          # the journals' presentation ids (C.pres_id)
PRES_REG = [f"{i}:{l}:canon" for i in ITM for l in ("en", "fa")]      # the registration file's order (item:lang:world)
BASE = {i: (i != "g-e") for i in ITM}                       # step 4: g-e failed
REG = {"schema": "ftm.a26-12.5-registration/1", "namespace": GE.FG_NS, "treatment_identity": FIN, "presentations": PRES_REG,
       "items": [{"item": i, "expected_outcome": BASE[i]} for i in ITM], "expected_changes": [], "variance_allowance": {"min_agreement": 0.80}}
REG["sha256"] = C.self_sha(REG)
RSHA = "f" * 64


def srow(arm, p, goal, **kw):
    r = {"scored": True, "arm": arm, "pres": p, "cell": f"{GE.FG_NS}|plan|{arm}|{p}|r0", "unit": f"u-{arm}-{p}", "executed": True, "goal_ok": goal, "plan_len": 0 if arm == "E0" else None, "unrealistic_slice": False,
         "n_slice_no_knife": 0}
    r.update(kw)
    return r


def good_rows(p0=None):
    p0 = p0 or {}
    return [srow("P0", p, p0.get(p, BASE[C.parse_pres(p)["item"]])) for p in PRES] + [srow("E0", p, False) for p in PRES]


def exec_of(rows, ident=FIN):
    return [{"treatment_id": ident, "ts": "2026-10-16T01:00:00", "status": "done", "unit": r["unit"], "cells": [r["cell"]]} for r in rows]


EXEC = exec_of(good_rows())
EV = [{"event": "a26_12_5_registration", "sha256": RSHA, "ts": "2026-10-16T00:59:59"}]


def v(rows=None, exec_rows=EXEC, events=EV, reg=REG, rsha=RSHA, base=BASE, final=FIN):
    return GE.eval_125_verdict(good_rows() if rows is None else rows, exec_rows, events, reg, rsha, base, final_sha=final, min_agreement=S.EVAL_MIN_AGREEMENT)


check("3.1 nothing scored -> PENDING", v(rows=[])["status"] == "PENDING")
r = v()
check("3.2 complete P0 + E0 under 7a2ee211 -> PASS (agreement 1.0, 10 empty-plan rows)", r["status"] == "PASS" and r["p0"]["agreement"] == 1.0
      and r["e0"]["empty_plan_rows"] == 10, r["reasons"])
check("3.3 exec rows under another identity -> FAIL", v(exec_rows=EXEC + [{"treatment_id": OTHER, "ts": "2026-10-16T02:00:00"}])["status"] == "FAIL")
check("3.4 the live final identity is not 7a2ee211 -> FAIL", v(final=OTHER)["status"] == "FAIL")
check("3.5 registration missing -> FAIL (scored rows exist)", v(reg=None, rsha=None)["status"] == "FAIL")
check("3.6 registration event missing / for another file / after the first execution -> FAIL",
      v(events=[])["status"] == "FAIL" and v(rsha="e" * 64)["status"] == "FAIL"
      and v(events=[dict(EV[0], ts="2026-10-16T01:00:01")])["status"] == "FAIL")
r = v(rows=good_rows({PRES[0]: False, PRES[1]: False}))
check("3.7 two unregistered differences of 10 (agreement .8) are inside the registered variance allowance -> PASS, both listed",
      r["status"] == "PASS" and len(r["p0"]["unregistered_differences"]) == 2, r["reasons"])
r = v(rows=good_rows({PRES[0]: False, PRES[1]: False, PRES[2]: False}))
check("3.8 three unregistered differences (agreement .7) -> FAIL", r["status"] == "FAIL")
reg2 = copy.deepcopy(REG)
reg2["expected_changes"] = [{"item": "g-a", "expected_outcome": False, "evidence": "trace"}]
reg2["items"][0]["expected_outcome"] = False
reg2["sha256"] = C.self_sha(reg2)
r = v(rows=good_rows({PRES[0]: False, PRES[1]: False, PRES[2]: False}), reg=reg2)
check("3.9 a REGISTERED expected change (g-a True -> False) is applied before comparing: 1 unregistered difference -> PASS",
      r["status"] == "PASS" and r["p0"]["registered_changes"] == ["g-a"] and len(r["p0"]["unregistered_differences"]) == 1, r["reasons"])
reg3 = copy.deepcopy(REG)
reg3["items"][0]["expected_outcome"] = False                # changed expectation without a registered change
reg3["sha256"] = C.self_sha(reg3)
check("3.10 an expectation that does not follow step 4 + the registered changes -> FAIL", v(reg=reg3)["status"] == "FAIL")
reg4 = dict(REG, variance_allowance={"min_agreement": 0.5})
reg4["sha256"] = C.self_sha(reg4)
check("3.11 a variance allowance other than the previously specified 0.80 -> FAIL", v(reg=reg4)["status"] == "FAIL")
rows = good_rows()
rows[0]["unrealistic_slice"] = True
check("3.12 an unrealistic slice -> FAIL", v(rows=rows)["status"] == "FAIL")
rows = good_rows()
rows[-1]["goal_ok"] = True
check("3.13 the empty plan succeeds on one presentation -> FAIL", v(rows=rows)["status"] == "FAIL")
check("3.14 an E0 presentation missing -> FAIL", v(rows=good_rows()[:-1])["status"] == "FAIL")
rows = good_rows()
rows[0]["executed"] = False
r = v(rows=rows)
check("3.15 an unexecuted P0 presentation is listed and FAILS the gate", r["p0"]["not_executed"] == [PRES[0]] and r["p0"]["agree"] == 9 and r["status"] == "FAIL")
r = v(rows=[x for x in good_rows() if not (x["arm"] == "P0" and x["pres"] in PRES[:2])])
check("3.17 NEGATIVE (Astra r1 1): 2 of 10 P0 presentations missing (agreement still .8) -> FAIL, never covered by the allowance",
      r["status"] == "FAIL" and r["p0"]["agreement"] == 0.8 and any("not executed" in x for x in r["reasons"]), r["reasons"])
r = v(rows=[x for x in good_rows() if not (x["arm"] == "P0" and x["pres"] == PRES[0])])
check("3.18 NEGATIVE: one P0 presentation missing (agreement .9) -> FAIL", r["status"] == "FAIL")
check("3.19 NEGATIVE (Astra r1 2): complete score rows with an EMPTY / absent exec journal -> FAIL", v(exec_rows=[])["status"] == "FAIL"
      and v(exec_rows=None)["status"] == "FAIL")
r = v(exec_rows=[e for e in EXEC if not e["cells"][0].endswith(PRES[0] + "|r0") or "|E0|" in e["cells"][0]])
check("3.20 NEGATIVE: one scored P0 row without its exec row -> FAIL (listed)", r["status"] == "FAIL"
      and r["exec_evidence"]["scored_rows_without_exec_evidence"] == [PRES[0] + "#P0"], r.get("exec_evidence"))
bad_ex = [dict(e, status="blocked_state") if e["cells"][0].endswith(PRES[1] + "|r0") and "|E0|" in e["cells"][0] else e for e in EXEC]
check("3.21 NEGATIVE: the matching exec row is not done -> FAIL", v(exec_rows=bad_ex)["status"] == "FAIL")
mixed = [dict(e, treatment_id=OTHER) if e["cells"][0].endswith(PRES[2] + "|r0") else e for e in EXEC]
check("3.22 NEGATIVE: the matching exec rows exist only under another identity -> FAIL", v(exec_rows=mixed)["status"] == "FAIL")
check("3.23 NEGATIVE: exec rows of other units (cells match, units differ) -> FAIL",
      v(exec_rows=[dict(e, unit="other") for e in EXEC])["status"] == "FAIL")
check("3.24 reg_pres: item:lang:world -> C.pres_id; an already-canonical id is kept; the real registration maps onto real dev pres ids",
      [GE.reg_pres(p) for p in PRES_REG] == PRES and GE.reg_pres(PRES[0]) == PRES[0]
      and all(GE.reg_pres(p) == C.pres_id(p.split(":")[0], p.split(":")[1]) for p in
              json.load(open(os.path.join(C.ROOT, *GE.A26_125_REG), encoding="utf-8"))["presentations"]))
reg5 = dict(REG, presentations=[p.replace(":canon", ":reloc") for p in PRES_REG])
reg5["sha256"] = C.self_sha(reg5)
check("3.25 NEGATIVE: registered presentations that match no scored row (other world) -> FAIL (all not executed)",
      v(reg=reg5)["status"] == "FAIL" and len(v(reg=reg5)["p0"]["not_executed"]) == 10)
check("3.16 a registration under another identity -> FAIL", v(reg=dict(REG, treatment_identity=OTHER, sha256=C.self_sha(dict(REG, treatment_identity=OTHER))))["status"] == "FAIL")

# ================================================================================================ 4 the real registration file
print("== 4 the real A26 12.5 registration")
rp = os.path.join(C.ROOT, *GE.A26_125_REG)
real = json.load(open(rp, encoding="utf-8"))
man = C.load_manifest()
base = {r["id"]: bool(r.get("goal_ok")) for r in (GE._jl(os.path.join(C.ROOT, S.STEP4_BASELINE)) or []) if r.get("status") in ("ok", "FAIL")}
check("4.1 intact, final identity, namespace freeze-gates-ft2, the variance allowance = score.EVAL_MIN_AGREEMENT",
      real["sha256"] == C.self_sha(real) and real["treatment_identity"] == FIN and real["namespace"] == GE.FG_NS
      and real["variance_allowance"]["min_agreement"] == S.EVAL_MIN_AGREEMENT)
check("4.2 it covers exactly the effective dev set (EN + FA, canon) and omits the A26-archived dev items",
      sorted(x["item"] for x in real["items"]) == sorted(man["dev_pilot"]) and len(real["presentations"]) == 2 * len(man["dev_pilot"])
      and sorted(o["item"] for o in real["omitted"]) == sorted(set(C.load_manifest(effective=False)["dev_pilot"]) - set(man["dev_pilot"])))
check("4.3 every expectation follows the step-4 baseline (no expected change: no slice in any registered plan / setup / step-4 trace)",
      real["expected_changes"] == [] and all(x["expected_outcome"] == base[x["item"]] and not x["slice_in_reference_plan_or_setup"]
                                             and not x["step4_trace"]["slice_executed"] for x in real["items"]))
check("4.4 baseline sha recorded = the step-4 file now", real["baseline"]["sha256"] == C.sha256_file(os.path.join(C.ROOT, S.STEP4_BASELINE)))

# ================================================================================================ 5 tokens / context / leakage: final identity
print("== 5 tokens / context / leakage under the final identity (acceptance-v2-a2)")
NS2 = GE.ACC_FINAL_NS
CAP, PLN = "c" * 64, "p" * 64


def selfed(d):
    d = dict(d)
    d.pop("sha256", None)
    d["sha256"] = C.self_sha(d)
    return d


GEL = selfed({"schema": "ft.acceptance-g/1", "namespace": NS2, "ok": True, "problems": [], "treatment_id": FIN, "capture_sha": CAP})
CFT = {"namespace": NS2, "gemma_verdict": "PASS_MEASURED", "freeze_grade": True, "not_freeze_grade_because": [], "capture_sha": CAP, "n_checks": 3,
       "by_model": {"gemma": {"verdicts": {"PASS_MEASURED": 3, "FAIL_MEASURED": 0}}}}
C11A = {"namespace": NS2, "ok": True, "problems": [], "system_text_hits": [], "capture_sha": CAP, "plan_sha": PLN, "requests": 4, "bound": 4,
        "first_transmissions": 6, "first_bound": 6, "repair_turns": 1, "repair_bound": 1}
DOCS = {"g_eligible.json": GEL, "context_fit.json": CFT, "c11_acceptance.json": C11A}
SHAS = {"g_eligible.json": "1" * 64, "context_fit.json": "2" * 64, "c11_acceptance.json": "3" * 64, "capture.jsonl": CAP, "plan.jsonl": PLN}
crit = {k: {"pass": True, "n_reasons": 0, "reasons": []} for k in ("a", "b", "c", "c11", "d", "e", "f", "g", "h", "i")}
crit["e"] = {"pass": False, "n_reasons": 1, "reasons": ["P0 does not reach the goal"]}          # acceptance NOT PASS overall (A36)
V2 = selfed({"schema": "ft.acceptance-verdict/4", "attempt": 2, "namespace": NS2, "verdict": "NOT PASS", "treatment_id": FIN,
             "identity": {"sha": FIN}, "decision": {"criteria": crit, "verdict": "NOT PASS"}, "evidence": dict(SHAS),
             "bundle": {"manifest_sha": "b" * 64}})


def av(gid, v=V2, auth=(), fs=FIN, shas=None, docs=None):
    return GE.acc_final_verdict(gid, v, list(auth), final_sha=fs, shas_now=dict(SHAS if shas is None else shas), docs=DOCS if docs is None else docs)


for g in ("tokens", "context", "leakage"):
    check(f"5.1 {g}: authentic attempt-2 verdict, final identity, its criteria pass, evidence unchanged -> PASS (overall NOT PASS ignored)",
          av(g)[0] == "PASS", av(g)[1])
    check(f"5.2 {g}: no verdict -> PENDING", av(g, v=None)[0] == "PENDING")
    check(f"5.3 {g}: authentication problem (bundle replay) -> FAIL", av(g, auth=["attempt 2: bundle file changed"])[0] == "FAIL")
    check(f"5.4 {g}: tampered verdict (edited, self sha stale) -> FAIL", av(g, v=dict(V2, verdict="PASS"))[0] == "FAIL")
    check(f"5.5 {g}: verdict under another identity -> FAIL", av(g, v=selfed(dict(V2, treatment_id=OTHER, identity={"sha": OTHER})))[0] == "FAIL")
    check(f"5.6 {g}: live final identity is not 7a2ee211 -> FAIL", av(g, fs=OTHER)[0] == "FAIL")
    for k in GE.ACC_FINAL_CRITERIA[g]:
        c2 = dict(crit, **{k: {"pass": False, "n_reasons": 1, "reasons": ["x"]}})
        vv = selfed(dict(V2, decision={"criteria": c2, "verdict": "NOT PASS"}))
        check(f"5.7 {g}: criterion {k} failing -> FAIL", av(g, v=vv)[0] == "FAIL")
        c3 = {kk: vv2 for kk, vv2 in crit.items() if kk != k}
        check(f"5.8 {g}: criterion {k} absent -> FAIL", av(g, v=selfed(dict(V2, decision={"criteria": c3})))[0] == "FAIL")
    for f in GE.ACC_FINAL_FILES[g]:
        check(f"5.9 {g}: {f} changed since the verdict -> FAIL", av(g, shas=dict(SHAS, **{f: "9" * 64}))[0] == "FAIL")
        check(f"5.10 {g}: {f} missing -> FAIL", av(g, shas=dict(SHAS, **{f: None}))[0] == "FAIL")
    check(f"5.11 {g}: no bundle recorded -> FAIL", av(g, v=selfed(dict(V2, bundle={})))[0] == "FAIL")
check("5.12 a criterion of ANOTHER gate failing does not fail this one (tokens ignores i; context ignores g / h)",
      av("tokens", v=selfed(dict(V2, decision={"criteria": dict(crit, i={"pass": False, "reasons": ["x"]})})))[0] == "PASS"
      and av("context", v=selfed(dict(V2, decision={"criteria": dict(crit, g={"pass": False, "reasons": ["x"]})})))[0] == "PASS")
check("5.13 evidence-file content: g_eligible not ok / context_fit not freeze-grade / c11_acceptance unbound -> FAIL",
      av("tokens", docs=dict(DOCS, **{"g_eligible.json": selfed(dict(GEL, ok=False))}))[0] == "FAIL"
      and av("tokens", docs=dict(DOCS, **{"g_eligible.json": dict(GEL, ok=False)}))[0] == "FAIL"
      and av("context", docs=dict(DOCS, **{"context_fit.json": dict(CFT, freeze_grade=False)}))[0] == "FAIL"
      and av("context", docs=dict(DOCS, **{"context_fit.json": dict(CFT, by_model={"opus": {"verdicts": {"UNRESOLVED_EST": 1}}})}))[0] == "FAIL"
      and av("leakage", docs=dict(DOCS, **{"c11_acceptance.json": dict(C11A, bound=3)}))[0] == "FAIL"
      and av("leakage", docs=dict(DOCS, **{"c11_acceptance.json": dict(C11A, plan_sha="0" * 64)}))[0] == "FAIL")
# the c11 audit half
WB = {"dataset": "d" * 64, "treatment_test_v1": FIN, "capture": CAP, "plan": PLN}
AUD = {"ok": True, "problems": [], "controls": {"n": 24, "passed": 24}, "binding": dict(WB, audit_script="s" * 64, prereg="r" * 64)}


def cf(c11=AUD, fs=FIN, wb=WB, ss="s" * 64, pr="r" * 64):
    return GE.c11_final_problems(c11, final_sha=fs, want_bind=wb, script_sha=ss, prereg_sha=pr)


check("5.14 c11 audit clean + bound to the final identity + current script / prereg -> no reasons", cf() == [], cf())
check("5.15 c11 audit NEGATIVES: missing / not ok / a control failing / other identity / binding mismatch / stale script / stale prereg",
      cf(c11={}) and cf(c11=dict(AUD, ok=False)) and cf(c11=dict(AUD, controls={"n": 24, "passed": 23}))
      and cf(c11=dict(AUD, binding=dict(AUD["binding"], treatment_test_v1=OTHER)), wb=dict(WB, treatment_test_v1=OTHER))
      and cf(wb=dict(WB, capture="0" * 64)) and cf(ss="0" * 64) and cf(pr="0" * 64) and cf(fs=OTHER))
# the real files: the recorded attempt-2 verdict authenticates (bundle replay) and each final part passes on it; the gate docs carry it
A = GE.acc_final_inputs(C.ROOT)
check("5.16 REAL: the acceptance-v2-a2 verdict authenticates (acc_attempt2.verdict_problems, replay) and is under 7a2ee211",
      A["verdict"] is not None and A["auth"] == [] and A["verdict"]["treatment_id"] == FIN, A["auth"][:2])
for g in ("tokens", "context", "leakage"):
    st, why = GE.acc_final_verdict(g, A["verdict"], A["auth"], final_sha=FIN, shas_now=A["shas"], docs=A["docs"])
    check(f"5.17 REAL {g}: the acceptance part passes on the real attempt-2 evidence", st == "PASS", why[:2])
check("5.18 REAL: a byte flip of the real capture.jsonl sha is caught",
      GE.acc_final_verdict("tokens", A["verdict"], A["auth"], final_sha=FIN, shas_now=dict(A["shas"], **{"capture.jsonl": "0" * 64}), docs=A["docs"])[0] == "FAIL")

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
