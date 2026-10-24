"""Gate for the round-6b CHECK harness (sim/final_test/signal/r6b/{r6b_common,plan_r6b,exec_r6b,score_r6b}.py; stage r6b-harness).
DRY RUN on synthetic rows: no simulator, no LLM, no network, no Anthropic call. Windows `python`, run from the repo root:

    python sim/_sig_r6b_check_selftest.py

Covers: the gate (authorization, freeze shas, pre-K-B2b pins, harness self-pin, settings, settings-selected inputs), the sets
(= r6/pkg_r6.sets()), completeness, the Opus cost estimate and the A24 guard, the decision rule (85 % rule, 90 % aim + Arash's
reading, matched zero-loss, spoon rule (a), slicing rule, improvement, STRICT, integrity, need checks outside the gates), and that the
three scripts REFUSE on the current tree before importing any planner or executor."""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R6B = os.path.join(ROOT, "sim", "final_test", "signal", "r6b")
sys.path.insert(0, R6B)
sys.path.insert(0, os.path.join(ROOT, "sim", "final_test", "signal"))
sys.path.insert(0, os.path.join(ROOT, "sim"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import r6b_common as C  # noqa: E402
import score_r6b as S   # noqa: E402

N = [0, 0]


def ok(cond, name):
    N[1] += 1
    if cond:
        N[0] += 1
    else:
        print("FAIL:", name)


# =========================================================================================== 1 the gate (pure check_gate)
FILES = {r: "a" * 64 for r in C.MUST_PIN + C.HARNESS_FILES}
FILES.update({"sim/honest_skills.py": "b" * 64, "sim/planners/exec_contract.json": "c" * 64})
SET_OK = {"status": "FROZEN", "dataset": "sim/dataset/general.v2.json", "round5_dataset": "sim/dataset/archive/general.v1.json",
          "package_rows": {"universal": "r6b/u.jsonl", "strict": "r6b/s.jsonl"}, "package_builder": "r6b/pkg.py",
          "executor_switches": {"objmem.DEDUP_FRAME": True}, "seeds": {"world_seed": 0}, "opus_cap_usd": 2.5, "cache_dir": "x",
          "runtime_pins": ["docs/briefing.json"]}
for r in C.settings_selected(SET_OK):
    FILES[r] = "d" * 64
AM_OK = {"authorization": {"status": "CONFIRMED", "confirmed": True, "confirmed_date": "2026-10-14"}, "freeze": dict(FILES)}
sha_of = lambda r: FILES.get(r)
ok(C.check_gate(AM_OK, SET_OK, sha_of) == [], "gate: a complete re-freeze passes")
am = json.loads(json.dumps(AM_OK)); am["authorization"]["status"] = "PENDING"
ok(any("PENDING" in b for b in C.check_gate(am, SET_OK, sha_of)), "gate: PENDING authorization refused")
am = json.loads(json.dumps(AM_OK)); am["authorization"]["confirmed"] = False
ok(any("confirmed" in b for b in C.check_gate(am, SET_OK, sha_of)), "gate: CONFIRMED without confirmed=true refused")
am = json.loads(json.dumps(AM_OK)); am["authorization"]["confirmed_date"] = None
ok(C.check_gate(am, SET_OK, sha_of) != [], "gate: CONFIRMED without a date refused")
ok(any("changed after the freeze" in b for b in C.check_gate(AM_OK, SET_OK, lambda r: "e" * 64 if r == "sim/ltm_package.py" else FILES.get(r))),
   "gate: a changed frozen file refused")
ok(any("missing" in b for b in C.check_gate(AM_OK, SET_OK, lambda r: None if r == "sim/goal_schema.py" else FILES.get(r))),
   "gate: a missing frozen file refused")
am = json.loads(json.dumps(AM_OK)); am["freeze"]["sim/honest_skills.py"] = C.PRE_KB2B["sim/honest_skills.py"]
ok(any("pre-K-B2b" in b for b in C.check_gate(am, SET_OK, lambda r: am["freeze"].get(r))), "gate: a freeze pinning the pre-K-B2b executor refused")
am = json.loads(json.dumps(AM_OK)); am["freeze"]["sim/planners/exec_contract.json"] = C.PRE_KB2B["sim/planners/exec_contract.json"]
ok(any("pre-K-B2b" in b for b in C.check_gate(am, SET_OK, lambda r: am["freeze"].get(r))), "gate: a freeze pinning the pre-K-B2b contract refused")
am = json.loads(json.dumps(AM_OK)); del am["freeze"]["sim/final_test/signal/r6b/score_r6b.py"]
ok(any("check harness file" in b for b in C.check_gate(am, SET_OK, sha_of)), "gate: harness not self-pinned refused")
am = json.loads(json.dumps(AM_OK)); del am["freeze"]["sim/objmem.py"]
ok(any("sim/objmem.py" in b for b in C.check_gate(am, SET_OK, sha_of)), "gate: a MUST_PIN runtime file not pinned refused")
st = dict(SET_OK, status="DRAFT")
ok(any("DRAFT" in b for b in C.check_gate(AM_OK, st, sha_of)), "gate: DRAFT settings refused")
st = dict(SET_OK, dataset=None)
ok(any("dataset not set" in b for b in C.check_gate(AM_OK, st, sha_of)), "gate: unset dataset refused")
st = dict(SET_OK, package_rows={"universal": "r6b/u.jsonl", "strict": None})
ok(any("package_rows.strict" in b for b in C.check_gate(AM_OK, st, sha_of)), "gate: an unset package row path refused")
am = json.loads(json.dumps(AM_OK)); del am["freeze"]["docs/briefing.json"]
ok(any("settings-selected input docs/briefing.json" in b for b in C.check_gate(am, SET_OK, lambda r: am["freeze"].get(r))),
   "gate: a settings-selected runtime pin not frozen refused")
ok(any("settings missing" in b or "missing" in b for b in C.check_gate(AM_OK, None, sha_of)), "gate: no settings file refused")
ok(any("no freeze map" in b for b in C.check_gate({"authorization": AM_OK["authorization"]}, SET_OK, sha_of)), "gate: empty freeze map refused")

# =========================================================================================== 2 the sets (current tree) = r6/pkg_r6.sets()
man, by, pres, forbid = C.sets()
ok([sum(p["set"] == s for p in pres) for s in C.SETS] == [20, 22, 8], "sets: 20 / 22 / 8 presentations")
ok(not ({p["item"] for p in pres} & forbid), "sets: no test / acceptance item")
ok(not any(p["item"] in ("g-l3-tool-cut", "g-l3-cutlery-sink") for p in pres), "sets: the D2-archived Knife items are out")
ok(any(p["item"] == C.SPOON_ITEM for p in pres if p["set"] == "fresh"), "sets: the spoon item is in fresh-22")
try:
    sys.path.insert(0, os.path.join(ROOT, "sim", "final_test", "signal", "r6"))
    import pkg_r6
    _m, _b, pres6 = pkg_r6.sets()
    ok([(p["item"], p["lang"], p["set"]) for p in pres6] == [(p["item"], p["lang"], p["set"]) for p in pres],
       "sets: identical (order included) to the frozen r6/pkg_r6.sets()")
except Exception as e:                                                          # noqa: BLE001
    ok(False, "sets: pkg_r6.sets() comparison raised %s" % e)

# =========================================================================================== 3 completeness
mini = [{"item": "a", "lang": "en", "set": "orig"}]
rows3 = [{"cell": "G0|a|en"}, {"cell": "GP|a|en"}, {"cell": "GP|a|en"}, {"cell": "O0|zz|en"}]
c3 = C.completeness(rows3, mini)
ok(c3["missing"] == ["GPs|a|en", "O0|a|en"] and c3["duplicates"] == ["GP|a|en"] and c3["unexpected"] == ["O0|zz|en"], "completeness: missing / dup / unexpected")

# =========================================================================================== 4 cost estimate + A24 guard
prompts = [("O0|a|en", "x" * 6000), ("O0|b|en", "x" * 6000), ("O0|c|en", "x" * 6000)]
states = {"O0|a|en": "replay", "O0|b|en": "new", "O0|c|en": "repair"}
hist = {"tokens_out_mean": 100.0, "tokens_out_max": 200}
est = C.estimate_opus(prompts, lambda c: states[c], hist=hist, chars_per_token=2.0)
ok(est["cached"] == 1 and est["new"] == 1 and est["repair_only"] == 1, "estimate: replay / new / repair-only split")
ok(abs(est["usd_mean"] - ((3000 * 4 + 100 * 20) + ((3000 + 100 + 200) * 4 + 100 * 20)) / 1e6) < 1e-4, "estimate: mean = new + priced repair, replay free")
est0 = C.estimate_opus(prompts[:1], lambda c: "replay", hist=hist)
ok(est0["usd_mean"] == 0 and est0["usd_high"] == 0 and est0["usd_worst"] == 0, "estimate: a fully replayed chain costs 0")
ok(est["usd_mean"] < est["usd_high"] < est["usd_worst"], "estimate: mean < high < worst")
sp = {"committed": 2.0}
ok(C.cost_guard(est, sp, 2.5) == [], "guard: small round inside the cap and the A24 gate")
ok(any("exceeds this round" in b for b in C.cost_guard(dict(est, usd_high=3.0), sp, 2.5)), "guard: high estimate over the cap refused")
ok(any("A24 gate" in b for b in C.cost_guard(est, {"committed": 78.0}, 2.5)), "guard: spend + cap over the 80 USD gate refused")
ok(any("projection" in b for b in C.cost_guard(est, sp, 2.5, projection_usd=76.0)), "guard: spend + cap + remaining projection over 80 refused")
ok(C.cost_guard(est, sp, 2.5, projection_usd=59.01) == [], "guard: with the current S2 projection (59.01) still inside 80")
proj, src = C.remaining_projection()
ok(proj is None or proj > 0, "projection: read from final_projection.json (S2) or absent")

# =========================================================================================== 5 the decision rule on synthetic rows
SPOON = C.SPOON_ITEM
ORIG = [("o%d" % k, "en") for k in range(20)]
FRESH = [(SPOON, "en"), (SPOON, "fa")] + [("f%d" % k, "fa") for k in range(20)]
NEED = [("n%d" % k, "en") for k in range(8)]
PRES = [{"item": i, "lang": l, "set": "orig"} for i, l in ORIG] + [{"item": i, "lang": l, "set": "fresh"} for i, l in FRESH] + \
       [{"item": i, "lang": l, "set": "need"} for i, l in NEED]
REF = [{"action": "pickup", "object": "Spoon", "to": None, "arg": None}]
WRONG = [{"action": "pickup", "object": "Fork", "to": None, "arg": None}]
EXP = {"plans_sha": "P" * 8, "freeze_id": "F" * 8, "executor_sha": {"x": "1"}, "settings_sha": "S" * 8}


def mk_plans(plan=REF):
    out = {}
    for p in PRES:
        for a in C.ARMS:
            out[C.cell(a, p["item"], p["lang"])] = {"cell": C.cell(a, p["item"], p["lang"]), "cond": a, "item": p["item"], "lang": p["lang"],
                                                    "ok": True, "plan": plan, "ref": REF, "holding": None, "infra": False, "prompt_excluded": []}
    return out


def ex_row(plans, cl, goal=True, cause=None, **kw):
    r = {"cell": cl, "goal_ok": goal, "audit_clean": True, "violations": 0, "oracle_reads": 0, "error": None, "inapplicable": None,
         "plan_key": C.plan_key(plans[cl]["plan"]), **EXP,
         "exec_steps": [{"action": "pickup", "object": "Spoon", "ok": goal or cause is None, "cause": None if goal else cause}]}
    r.update(kw)
    return r


def mk_runs(plans, goal=lambda rn, cl: True, cause="not_detected"):
    return {rn: {cl: ex_row(plans, cl, goal(rn, cl), cause) for cl in plans} for rn in ("run1", "rep2")}


def mk_r5(goal=lambda rn, cl: True):
    return {rn: {C.cell(a, i, l): {"cell": C.cell(a, i, l), "goal_ok": goal(rn, C.cell(a, i, l))} for i, l in ORIG + FRESH for a in C.ARMS}
            for rn in ("run1", "rep2")}


def dec(plans, runs, r5, **kw):
    kw.setdefault("expected", EXP)
    return S.decide(plans, runs, r5, PRES, **kw)


P0 = mk_plans()
R0 = mk_runs(P0)
d = dec(P0, R0, mk_r5())
ok(d["verdict"]["universal_pass"] and d["verdict"]["strict_pass"] and d["integrity"]["pass"], "decide: all goals -> PASS, integrity clean")
ok(d["verdict"]["claim"] == "retained performance", "decide: no +1 over round 5 -> 'retained performance'")
ok(d["runs"]["run1"]["sets"]["orig"]["universal"]["target_90"] == "MET" and d["verdict"]["target_90_met_all"], "decide: 20/20 -> 90 % MET")
r5i = mk_r5(lambda rn, cl: not cl.endswith("|o0|en"))
d = dec(P0, R0, r5i)
ok(d["verdict"]["claim"] == "improvement", "decide: +1 orig-20 over round 5 in BOTH runs -> improvement")
r5h = mk_r5(lambda rn, cl: not (cl.endswith("|o0|en") and rn == "run1"))
ok(dec(P0, R0, r5h)["verdict"]["claim"] == "retained performance", "decide: +1 in only one run -> no improvement claim")

# zero-loss gate, matched pairing
R = mk_runs(P0, lambda rn, cl: not (cl == "GP|f3|fa" and rn == "run1"), cause="wrong_target")
d = dec(P0, R, mk_r5())
ok(not d["verdict"]["universal_pass"] and d["verdict"]["wording"] == "r6b FAILED its predeclared rule", "decide: a fresh GP loss fails the zero-loss gate")
ok(d["runs"]["run1"]["sets"]["fresh"]["regression"]["losses"][0]["cell"] == "GP|f3|fa", "decide: the loss is listed with its cell")
ok("keep round 5 + D2" in d["verdict"]["next"], "decide: failure -> keep round 5 + D2, repair selection ends")
r5m = mk_r5(lambda rn, cl: not (cl == "GP|f3|fa" and rn == "run1"))
d = dec(P0, R, r5m)
ok(d["runs"]["run1"]["sets"]["fresh"]["regression"]["pass"], "decide: matched pairing: no loss when round-5 run 1 also failed that cell")
ok(any(x["cell"] == "GP|f3|fa" for x in d["runs"]["run1"]["sets"]["fresh"]["regression"]["info_losses_vs_either_round5_run"]),
   "decide: ... but it is listed as a loss vs EITHER round-5 run")
gain_r5 = mk_r5(lambda rn, cl: cl != "GP|f4|fa")
d = dec(P0, mk_runs(P0, lambda rn, cl: cl != "GP|f3|fa", cause="wrong_target"), gain_r5)
ok(not d["verdict"]["universal_pass"], "decide: a gain elsewhere does not offset a loss")

# spoon rule (a)
R = mk_runs(P0, lambda rn, cl: cl != "GP|%s|en" % SPOON, cause="not_detected")
d = dec(P0, R, mk_r5())
fr = d["runs"]["run1"]["sets"]["fresh"]
ok(d["verdict"]["universal_pass"] and not fr["regression"]["losses"] and fr["regression"]["exempt_losses_listed"][0]["exemption"] == "spoon_a",
   "spoon (a): correct plan + not_detected -> exempt, still listed")
ok(fr["universal"]["S_GP"] == 21, "spoon (a): the failed goal still counts in the performance number")
P1 = mk_plans()
P1["GP|%s|en" % SPOON]["plan"] = WRONG
R = mk_runs(P1, lambda rn, cl: cl != "GP|%s|en" % SPOON, cause="not_detected")
ok(dec(P1, R, mk_r5())["runs"]["run1"]["sets"]["fresh"]["regression"]["losses"], "spoon (a): a wrong plan -> the loss counts")
R = mk_runs(P0, lambda rn, cl: cl != "GP|%s|fa" % SPOON, cause="wrong_target")
ok(dec(P0, R, mk_r5())["runs"]["run1"]["sets"]["fresh"]["regression"]["losses"], "spoon (a): cause wrong_target -> the loss counts")
R = mk_runs(P0, lambda rn, cl: cl != "GP|%s|en" % SPOON, cause="not_detected")
R["run1"]["GP|%s|en" % SPOON]["plan_key"] = C.plan_key(WRONG)
d = dec(P0, R, mk_r5())
ok(d["runs"]["run1"]["sets"]["fresh"]["regression"]["losses"] and not d["integrity"]["pass"],
   "spoon (a): the executed plan is not the compared plan -> no exemption, integrity fails (provenance)")
ok(S.exemption({"item": "f3", "lang": "fa", "ok": True, "plan": REF, "ref": REF}, ex_row(P0, "GP|f3|fa", False, "not_detected")) is None,
   "spoon (a): only the spoon item")

# slicing rule (predeclared list, empty)
R = mk_runs(P0, lambda rn, cl: cl != "GP|f5|fa", cause="no_cutting_tool")
ok(dec(P0, R, mk_r5())["runs"]["run1"]["sets"]["fresh"]["regression"]["losses"], "slicing: no_cutting_tool on a non-eligible item counts")
S.SLICE_RULE_ELIGIBLE = frozenset({("f5", "fa")})
d = dec(P0, R, mk_r5())
ok(not d["runs"]["run1"]["sets"]["fresh"]["regression"]["losses"] and
   d["runs"]["run1"]["sets"]["fresh"]["regression"]["exempt_losses_listed"][0]["exemption"] == "new_slice_rule",
   "slicing: an eligible item with no_cutting_tool -> exempt, listed")
S.SLICE_RULE_ELIGIBLE = frozenset()
ok(S.SLICE_RULE_ELIGIBLE == frozenset(), "slicing: the shipped list is empty")

# 85 % rule, 90 % aim, Arash's reading
ok(S.perf(16, 19)["pass_85"] is False and S.perf(17, 19)["pass_85"] is True, "85 %: 16/19 fails, 17/19 passes (ceil 16.15 = 17)")
ok(S.perf(17, 19)["target_90"] == "AROUND_90_OK", "90 %: 17/19 = .8947 -> AROUND_90_OK (Arash: 'like 89')")
ok(S.perf(18, 20)["target_90"] == "MET" and S.perf(18, 19)["target_90"] == "MET", "90 %: 18/20 and 18/19 MET")
ok(S.perf(16, 19)["target_90"] == "UNMET" and S.perf(8, 9)["target_90"] == "UNMET", "90 %: 16/19 and 8/9 (.889 < .89) UNMET")
ok(S.perf(89, 100)["target_90"] == "AROUND_90_OK" and S.perf(0, 0)["target_90"] == "N/A", "90 %: 89/100 AROUND (exact), S_O0 = 0 -> N/A")
ok(S.perf(19, 22)["pass_85"] and S.perf(18, 22)["pass_85"] is False, "85 %: fresh 19/22 passes, 18/22 fails")
low = mk_runs(P0, lambda rn, cl: not (cl.startswith("GP|o") and int(cl.split("|")[1][1:]) < 2))
d = dec(P0, low, mk_r5(lambda rn, cl: not (cl.startswith("GP|o") and int(cl.split("|")[1][1:]) < 2)))
ok(d["verdict"]["universal_pass"] and d["runs"]["run1"]["sets"]["orig"]["universal"]["target_90"] == "MET" and
   d["verdict"]["standing_goal_90"].startswith("MET"), "decide: 18/20 vs 20 -> PASS and 90 % MET")
low3 = mk_runs(P0, lambda rn, cl: not (cl.startswith("GP|o") and int(cl.split("|")[1][1:]) < 3))
d = dec(P0, low3, mk_r5(lambda rn, cl: not (cl.startswith("GP|o") and int(cl.split("|")[1][1:]) < 3)))
ok(d["verdict"]["universal_pass"] and d["runs"]["run1"]["sets"]["orig"]["universal"]["target_90"] == "UNMET" and
   d["verdict"]["standing_goal_90"].startswith("NOT met"), "decide: 17/20 = .85 -> 85 % PASS but the 90 % aim is NOT met (reported apart)")
low4 = mk_runs(P0, lambda rn, cl: not (cl.startswith("GP|o") and int(cl.split("|")[1][1:]) < 4))
d = dec(P0, low4, mk_r5(lambda rn, cl: not (cl.startswith("GP|o") and int(cl.split("|")[1][1:]) < 4)))
ok(not d["verdict"]["universal_pass"], "decide: 16/20 < ceil(0.85 x 20) = 17 -> FAIL")

# STRICT
R = mk_runs(P0, lambda rn, cl: not (cl.startswith("GPs|f") and int(cl.split("|")[1][1:] or 0) < 5 if cl.split("|")[1][1:].isdigit() else False))
d = dec(P0, R, mk_r5())
ok(d["verdict"]["universal_pass"] and not d["verdict"]["strict_pass"] and d["verdict"]["wording"] == "UNIVERSAL passed; STRICT failed (no robustness claim)",
   "STRICT: separate; 17/22 fails -> 'UNIVERSAL passed; STRICT failed'")
ok(d["runs"]["run1"]["sets"]["fresh"]["strict_losses_vs_round5_GPs_info"], "STRICT: losses vs round-5 GPs listed (not a gate)")

# integrity
def integ_fail(mut, name):
    R = mk_runs(P0)
    P = mk_plans()
    mut(P, R)
    d = dec(P, R, mk_r5())
    ok(not d["integrity"]["pass"] and not d["verdict"]["universal_pass"] and not d["verdict"]["strict_pass"], "integrity: " + name)


integ_fail(lambda P, R: R["run1"]["O0|o1|en"].update(violations=1, audit_clean=False), "a violation in an O0 cell fails")
integ_fail(lambda P, R: R["rep2"]["G0|f2|fa"].update(oracle_reads=2, audit_clean=False), "an oracle read fails")
integ_fail(lambda P, R: R["run1"].pop("GPs|o3|en"), "a missing execution fails")
integ_fail(lambda P, R: P.pop("O0|o3|en"), "a missing plan fails")
integ_fail(lambda P, R: P["G0|o4|en"].update(infra=True), "a planner infra failure fails")
integ_fail(lambda P, R: P["O0|o4|en"].update(prompt_excluded=["Knife"]), "a prompt naming the Knife fails")
integ_fail(lambda P, R: R["run1"]["GP|o5|en"].update(error="RuntimeError: x"), "an execution error fails")
integ_fail(lambda P, R: R["run1"]["GP|o5|en"].update(inapplicable="goal true at reset"), "an inapplicable cell fails")
integ_fail(lambda P, R: R["run1"]["GP|o6|en"].pop("oracle_reads"), "missing audit evidence fails")
integ_fail(lambda P, R: R["rep2"]["GP|o7|en"].update(plans_sha="OTHER"), "an execution under another plans file fails")
integ_fail(lambda P, R: R["rep2"]["GP|o7|en"].update(freeze_id="OTHER"), "an execution under another freeze identity fails")
integ_fail(lambda P, R: R["rep2"]["GP|o7|en"].update(executor_sha={"x": "2"}), "an execution on another executor fails")
integ_fail(lambda P, R: R["rep2"]["GP|o7|en"].update(settings_sha="OTHER"), "an execution under other settings fails")
R = mk_runs(P0)
r5c = mk_r5()
r5c["rep2"].pop("GP|o8|en")
ok(not dec(P0, R, r5c)["integrity"]["pass"], "integrity: incomplete round-5 comparator coverage fails")
raw = {"plans": list(P0.values()), "run1": list(R["run1"].values()) + [R["run1"]["GP|o1|en"]], "rep2": list(R["rep2"].values())}
ok(not dec(P0, R, mk_r5(), raw=raw)["integrity"]["pass"], "integrity: a duplicate execution row fails")
raw = {"plans": list(P0.values()) + [{"cell": "O0|g-l1-x|en"}], "run1": list(R["run1"].values()), "rep2": list(R["rep2"].values())}
ok(not dec(P0, R, mk_r5(), raw=raw)["integrity"]["pass"], "integrity: an unexpected cell fails")
raw = {"plans": list(P0.values()) + [dict(P0["O0|o1|en"], infra=True)], "run1": list(R["run1"].values()), "rep2": list(R["rep2"].values())}
ok(dec(P0, R, mk_r5(), raw=raw)["integrity"]["pass"], "integrity: an infra plan row superseded by a good row is not a duplicate")
ok(not dec(P0, R, mk_r5(), freeze_ok=False)["integrity"]["pass"], "integrity: freeze_ok False fails")

# need checks: outside every gate, but listed
R = mk_runs(P0, lambda rn, cl: not cl.split("|")[1].startswith("n"))
R["run1"]["GP|n1|en"].update(violations=3, audit_clean=False)
d = dec(P0, R, mk_r5())
ok(d["verdict"]["universal_pass"] and d["integrity"]["pass"], "need: failures and a violation in a need cell do not touch any gate")
ok(d["need_checks"]["n1|en"]["GP"]["run1"]["violations"] == 3 and d["need_checks"]["n0|en"]["GP"]["run1"]["goal"] is False,
   "need: their goals and violations are listed")
ok(d["sets"]["need"] == 8 and d["runs"]["run1"]["sets"]["orig"]["n"] == 20 and d["runs"]["run1"]["sets"]["fresh"]["n"] == 22,
   "need: counts use 20 + 22 only")

# text-changed presentations count like any other, flagged
R = mk_runs(P0, lambda rn, cl: cl != "GP|o9|en", cause="wrong_target")
d = dec(P0, R, mk_r5(), text_changed=[("o9", "en")])
ok(not d["verdict"]["universal_pass"] and d["runs"]["run1"]["sets"]["orig"]["regression"]["losses"][0]["text_changed"],
   "text-changed: a loss on a changed request still counts, flagged")
ok(any("successor-task-version" in x for x in d["disclosure"]) and any("RESULT-INFORMED" in x for x in d["disclosure"]),
   "disclosure: result-informed revision + successor-task version stated")

# =========================================================================================== 6 the scripts refuse on the current tree
def run(args):
    p = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode, p.stdout + p.stderr


FROZEN_NOW = (C.load_settings() or {}).get("status") == "FROZEN"
CLOSED_NOW = False
if not FROZEN_NOW:
    for script in ("plan_r6b.py", "exec_r6b.py", "score_r6b.py"):
        rc, txt = run([os.path.join(R6B, script)])
        ok(rc != 0 and "REFUSED" in txt and "PENDING" in txt, "refusal: %s refuses on the current tree (authorization PENDING)" % script)
else:
    # K-B2b re-freeze (2026-10-13): the tree is now FROZEN and CONFIRMED, so a bare run of the scripts would really plan / execute;
    # the selftest never starts them. It checks the pure gate on the real files instead: no refusal reason is left.
    _am, _st = C.load_json(C.AMEND_PATH), C.load_settings()
    _bad = C.check_gate(_am, _st, C._sha_of)
    import r6b_closed as RC   # BK-LINE 2026-10-13 (Astra BK-LINE-closed): round 6b was RUN and is CLOSED; approved later stages changed pins
    CLOSED_NOW = RC.closed_record() is not None
    if not CLOSED_NOW:
        ok(_bad == [], "re-frozen tree: the gate passes on the real amendment + settings (authorization CONFIRMED, every pin matches) %s" % _bad[:3])
    else:
        # replaces (pre-run state): ok(_bad == [], "re-frozen tree: the gate passes on the real amendment + settings ...")
        _drift, _other = RC.gate_drift(_bad)
        _reg = set((RC.load_register().get("entries") or {}))
        ok(_bad != [] and not _other and _drift == _reg and RC.check_closed() == [],
           "CLOSED tree: the gate REFUSES on the live tree, only for the registered post-run drift (%d paths, each archived + approved bytes), "
           "so round 6b cannot be re-run on the new prompt / executor %s" % (len(_reg), (_other or RC.check_closed())[:3]))
    ok(_am["authorization"]["status"] == "CONFIRMED" and "ARASH DECISION: R6b AUTHORIZED" in json.dumps(_am["authorization"]),
       "re-frozen tree: the authorization quotes the ledger entry 'ARASH DECISION: R6b AUTHORIZED'")
rc, txt = run([os.path.join(R6B, "plan_r6b.py"), "--provisional"])
ok(rc != 0 and "only with --estimate" in txt, "refusal: --provisional without --estimate refused")
rc, txt = run([os.path.join(R6B, "plan_r6b.py"), "--no-verify-packages"])
ok(rc != 0 and "REFUSED" in txt, "refusal: a real run without the package rebuild refused")
def _shas_r6b_now():
    return {f: C.sha_file(os.path.join(R6B, f)) for f in sorted(os.listdir(R6B)) if os.path.isfile(os.path.join(R6B, f))}


_shas_r6b = _shas_r6b_now()
rc, txt = run([os.path.join(R6B, "exec_r6b.py"), "--check-complete"])
if not FROZEN_NOW:
    ok(rc != 0 and "REFUSED" in txt, "refusal: exec --check-complete also passes the gate first")
elif not CLOSED_NOW:   # K-B2b re-freeze: the gate passes, and with no execution rows yet the completeness check fails (non-zero, all 200 missing)
    ok(rc != 0 and '"missing": 200' in txt, "frozen tree: exec --check-complete passes the gate and reports the study incomplete (200 missing)")
else:   # BK-LINE: CLOSED -> the gate refuses first (exec_r6b.py calls C.gate() before anything), naming a registered path; nothing written
    ok(rc != 0 and "REFUSED" in txt and "frozen artifact changed after the freeze" in txt and _shas_r6b == _shas_r6b_now(),
       "CLOSED tree: exec --check-complete is refused by the gate (registered drift) and no r6b file changed")

# =========================================================================================== 7 static checks of the harness sources
SRC = {n: open(os.path.join(R6B, n), encoding="utf-8").read() for n in ("r6b_common.py", "plan_r6b.py", "exec_r6b.py", "score_r6b.py")}
for n in ("plan_r6b.py", "exec_r6b.py"):
    s = SRC[n]
    g = s.index("C.gate()")
    heavy = [m.start() for m in re.finditer(r"^import (planners|honest_skills|objmem|step5_live|kitchen_briefing)|^from planners", s, re.M)]
    ok(heavy and all(g < h for h in heavy), "static: %s calls the gate before importing any planner / executor" % n)
sm = SRC["score_r6b.py"]
ok(sm.index("C.gate()", sm.index("def main")) < sm.index("C.sets(", sm.index("def main")), "static: score main() gates before reading results")
for n, s in SRC.items():
    ok(not re.search(r"^\s*assert\b", s, re.M), "static: no assert in %s (explicit refusals only)" % n)
ok("allowed = frozenset(p[\"item\"] for p in pres)" in SRC["exec_r6b.py"], "static: exec allow-list = the validated presentation ids only")
ok("frozenset(man[\"nontest\"])" not in SRC["exec_r6b.py"], "static: exec does not allow the whole nontest pool")
ok("CUR[\"cond\"] in C.PLUS_M" in SRC["exec_r6b.py"] and C.PLUS_M == frozenset({"GP", "GPs"}), "static: briefing-guided search for the +M arms only")
ok("TIDY_SWITCH_CLOSED = not strict_off" in SRC["exec_r6b.py"] and C.strict_e1_off() ==
   frozenset({"g-l2-microwave-run-egg", "g-l2-microwave-run-potato"}), "static: E1 off only for GPs on the tidy_switch_closed items")
ok("C.cost_guard(" in SRC["plan_r6b.py"] and "CostLedger(os.path.join(HERE, \"ledger_r6b.db\"), CAP" in SRC["plan_r6b.py"],
   "static: Opus calls behind the cost guard and the round's ledger cap")
for r in C.HARNESS_FILES:
    ok(os.path.exists(os.path.join(ROOT, r)), "static: harness file exists: %s" % r)
st_now = C.load_settings()
ok(st_now and st_now["status"] in ("DRAFT", "FROZEN") and (st_now["status"] == "DRAFT") == (not FROZEN_NOW),
   "settings: shipped as DRAFT, FROZEN only by the re-freeze (now %s)" % st_now["status"])
ok(st_now["executor_requires"] == {"honest_skills.CUTTING_TOOLS": ["ButterKnife"]}, "settings: the post-K-B2b executor is required")
ok(st_now["briefing"]["allow_unapproved"] is False, "settings: only the APPROVED briefing is used")
sh = open(os.path.join(R6B, "run_r6b.sh"), encoding="utf-8").read()
ok("exit 3" in sh and "--check-complete" in sh and sh.index("--check-complete") < sh.index("echo done"), "static: run_r6b.sh exits non-zero when incomplete, done only after completeness")
ok("\r" not in sh, "static: run_r6b.sh has LF line ends")

print("r6b check harness selftest: %d/%d" % tuple(N))
sys.exit(0 if N[0] == N[1] else 1)
