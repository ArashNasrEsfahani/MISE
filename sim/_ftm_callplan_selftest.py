"""Selftest for sim/final_test_callplan.py (stage M10). Offline: synthetic manifest, no simulator, no network, no API.

    python sim/_ftm_callplan_selftest.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import final_test_callplan as P   # noqa: E402
import common as C                # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


# synthetic sizes: independent of the real manifest
SZ = {"test": 20, "repeat": 6, "relocation": 4, "dev": 4, "manifest_sha256": "x"}
cs = P.cells(SZ)
src = open(os.path.join(HERE, "final_test_callplan.py"), encoding="utf-8").read()

check("no OpenRouter row (A6)", not any(c["arm"].startswith("R") or "openrouter" in json.dumps(c).lower() for c in cs))
check("no OpenRouter in model table", "R" not in P.MODEL_OF)
check("every arm resolves via C.arm_info", all(C.arm_info(c["arm"]) for c in cs))
check("blocks are C.BLOCKS names", {c["block"] for c in cs} <= set(C.BLOCKS))
check("scope equals C.scope_of", all(c["scope"] == C.scope_of(c["arm"], c["block"]) for c in cs))
check("calls = presentations x replicates", all(c["calls"] == c["presentations"] * c["replicates"] for c in cs))
main = [c for c in cs if c["block"] == "main"]
check("main: 8 arms (A23: 4 core + 2 FS + A18 X0, X+M) on all test presentations", len(main) == 8 and {"X0", "X+M"} <= {c["arm"] for c in main} and all(c["calls"] == 20 for c in main))
rep = [c for c in cs if c["block"] == "repeat"]
check("repeat: r=1,2 only (r=0 shared); A27: core four only (no FS, no X0 / X+M)", all(c["replicates"] == 2 and c["calls"] == 12 for c in rep) and len(rep) == 4
      and {c["arm"] for c in rep} == set(C.REPEAT_ARMS) == {"G0", "G+M", "O0", "O+M"})
curve = [c for c in cs if c["block"] == "curve"]
check("curve: A27 6 cells (G+M / O+M x order 0 K = 1, 3, 5), K=0 and o0k10 absent (shared), no order 1 / 2", len(curve) == 6
      and not any(c["arm"].endswith("k0") or c["arm"].endswith("o0k10") for c in curve) and not any("@o1" in c["arm"] or "@o2" in c["arm"] for c in curve))
check("curve arms use snapshot names", all(C.arm_info(c["arm"])["chooser"] == c["arm"].split("@")[1] for c in curve))
reloc = [c for c in cs if c["block"] == "reloc"]
check("reloc: G0,G+M,O0,O+M in reloc world", [c["arm"] for c in reloc] == ["G0", "G+M", "O0", "O+M"] and all(c["world"] == "reloc" for c in reloc))
codex = [c for c in cs if c["block"] == "codex"]
check("codex: 3 exploratory models x {0,+M} x 3 calls, free (gpt-6-sol is a main arm, A18)", len(codex) == 6 and not any("gpt-6-sol" in c["arm"] for c in codex) and all(c["replicates"] == 3 and c["scope"] == "free" for c in codex))
xm = [c for c in cs if c["arm"] in ("X0", "X+M")]
check("A18: X0 / X+M (gpt-6-sol) are free main arms, same snapshot S* for +M; A27: main r = 0 only, no REPEAT block",
      len(xm) == 2 and all(c["scope"] == "free" and c["model_key"] == "X" for c in xm)
      and {(c["block"], c["replicates"]) for c in xm} == {("main", 1)}
      and C.arm_info("X+M") == {"planner": "X:gpt-6-sol", "prompt": "S*", "chooser": "S*"} and C.arm_info("X0")["prompt"] == "none")
check("A18: the exploratory Codex block lists three models (not gpt-6-sol)", {c["arm"] for c in codex} == {f"X-{m}{s_}" for m in C.CODEX_EXPLORATORY for s_ in ("0", "+M")})
pj = P.projection(SZ, cs)
check("projection: paid USD from the priors, within the ceilings; free X arms add episodes but no USD",
      pj["paid_usd_total_prior"] > 0 and pj["x_main_arms"]["cells"] == 2 * 20 and pj["x_main_arms"]["episodes_upper_bound"] == pj["x_main_arms"]["cells"]
      and pj["ceilings"] == {"core": 60.0, "total": 100.0} and pj["hours_upper_bound"] > 0, pj)   # A37: test-v1 core 60 / total 100 (A24.1-C 30 / 50)
_ex_calls = sum(c["calls"] for c in cs if not C.plan_only(c["arm"]))
check("projection: A27 REPEAT units of the core four run 2 times, exploratory Codex plan-only (0 episodes)",
      P.REPEAT_RUNS == 2 and pj["episodes_by_block"]["codex"] == 0 and pj["episodes_by_block"]["repeat"] == 2 * sum(c["calls"] for c in rep)
      and pj["episodes_upper_bound"] == _ex_calls + 4 * SZ["repeat"] * (2 - 1) + sum(c["calls"] for c in rep) * (2 - 1), pj)
core = [c for c in cs if c["scope"] == "core"]
check("core scope = the paid core arms (A23: Opus O0 / O+M) in main + repeat only",
      {c["arm"] for c in core} == {"O0", "O+M"} and {c["block"] for c in core} == {"main", "repeat"})
check("Gemma never paid", all(c["scope"] == "free" for c in cs if c["model_key"] == "G"))
check("FS is 'other'", all(c["scope"] == "other" for c in cs if c["arm"] == "O+FS"))
check("G+FS free (Gemma)", all(c["scope"] == "free" for c in cs if c["arm"] == "G+FS"))
check("no reference / replay arm makes a call", not any(C.arm_info(c["arm"])["planner"] in ("ref",) or C.arm_info(c["arm"])["planner"].startswith("replay") for c in cs))
bs = P.by_scope(cs)
check("by_scope sums to total", sum(bs.values()) == sum(c["calls"] for c in cs))
check("by_scope core = O0 / O+M main+repeat", bs["core"] == 2 * (20 + 12), bs)
tot = P.totals(cs)
check("totals core_calls from scope", tot["O"]["core_calls"] == 2 * (20 + 12) and "S" not in tot and "H" not in tot and tot["G"]["core_calls"] == 0, tot)
res = P.block_reservations(cs, {"S": 0.05, "O": 0.10, "H": 0.02})
check("reservations: no free block, repairs doubled",
      all(r["scope"] != "free" for r in res) and
      [r for r in res if r["block"] == "main" and r["model_key"] == "O" and r["scope"] == "core"][0]["worst_usd"] == C.block_worst_case(40, 0.10))
check("pilot cells use scopes", all("scope" in c for c in P.pilot_cells(SZ)))
check("usd monotone", P.usd(10, 10, {"input": 5, "output": 25}, 500) > P.usd(10, 0, {"input": 5, "output": 25}, 500))

# real manifest: sizes reconcile with the frozen split
sz = P.sizes()
check("real sizes match manifest", sz["test"] == 250 and sz["repeat"] == 60 and sz["relocation"] == 20 and sz["dev"] == 24, sz)
check("A23: the real plan has no Sonnet / Haiku (archived) and no R rows", {c["model_key"] for c in P.cells(sz)} == {"G", "O", "X"})
check("cells carry no item ids (counts only)", "canon:" not in json.dumps(P.cells(sz)))

# the SPEC guard: test items refused unless it holds
def refused(fn):
    try:
        fn()
    except C.TestItemRefused:
        return True
    except Exception as e:
        print("   other:", type(e).__name__, e)
        return False
    return False

check("test items refused in rehearsal-v1", refused(lambda: P.item_ids("main", "rehearsal-v1", allow_test=True)))
check("test items refused in practice-v1", refused(lambda: P.item_ids("main", "practice-v1", allow_test=True)))
check("test items refused without --allow-test", refused(lambda: P.item_ids("main", "test-v1")))
check("test items refused without a freeze", refused(lambda: P.item_ids("main", "test-v1", allow_test=True)) or C.freeze_status()["ok"])
check("repeat/curve/codex/reloc sets are test-guarded too",
      all(refused(lambda b=b: P.item_ids(b, "rehearsal-v1", allow_test=True)) for b in ("repeat", "curve", "reloc", "codex")))
try:
    P.item_ids("nope", "rehearsal-v1")
    bad = False
except ValueError:
    bad = True
check("unknown block rejected", bad)
check("module never imports the item ids of the dataset", "general.json" not in src)

# synthetic manifest path for sizes()
with tempfile.TemporaryDirectory() as d:
    mp = os.path.join(d, "m.json")
    json.dump({"test": list("abc"), "repeat_subset": ["a"], "relocation_probe": ["a", "b"], "dev_pilot": ["z"], "sha256": "s"}, open(mp, "w"))
    s2 = P.sizes(mp)
    check("sizes() on a synthetic manifest", s2 == {"test": 6, "repeat": 2, "relocation": 4, "dev": 2, "manifest_sha256": "s"}, s2)

# ---- A21 G: the final cost projection ---------------------------------------------------------------------------------------
BLK = {"memory": 150, "fs": 759, "max": 800, "source": "test"}
fp = P.final_projection(SZ, cs, BLK, treatment="ft-v1")   # TREAT-FREEZE: the legacy model, named explicitly after the flip
sc = list(fp["scenarios"].values())
check("A21 G: three scenarios (pilot-like, +500 tail reference, stress), each with core / other / total, per model and per cuttable group",
      len(sc) == 3 and all({"usd_core", "usd_other", "usd_total", "by_model", "usd_by_cuttable_group", "cut_needed_in_order", "fits_after_cut"} <= set(x) for x in sc))
check("A21 G: cost rises with the thinking tail, the repair rate and the failed-call rate (S1 < S2 < S3)", sc[0]["usd_total"] < sc[1]["usd_total"] < sc[2]["usd_total"])
check("A21 G: the free arms (Gemma, Codex X0 / X+M and the three exploratory models, reference / replay) cost 0 USD: only Opus is priced in test-v1 (A23: S / H archived)",
      set(sc[0]["by_model"]) == {"O"} and fp["scenarios"][list(fp["scenarios"])[0]]["usd_by_cuttable_group"]["codex"] == 0.0)
paid_core = [c for c in cs if c["scope"] == "core"]
check("A21 G: core USD = the core cells only (six core arms in main and REPEAT)", abs(sc[0]["usd_core"] - sum(c["calls"] * P.cell_usd(c, P.SCENARIOS[list(P.SCENARIOS)[0]], BLK) for c in paid_core)) < 0.01)
sc0 = dict(P.SCENARIOS[list(P.SCENARIOS)[0]], tail=0, repair=0.0, failed=0.0)
c_o = next(c for c in cs if c["arm"] == "O0")
check("A21 G: with no tail, no repairs and no failed calls an Opus no-memory cell costs exactly its post-A11 prior", abs(P.cell_usd(c_o, sc0, BLK) - P.PRIOR_USD_PER_CELL["O"]) < 1e-12)
c_om = next(c for c in cs if c["arm"] == "O+M")
check("A21 G: a +M cell adds its block tokens at the input price (150 tokens x 4 USD/M)", abs(P.cell_usd(c_om, sc0, BLK) - P.cell_usd(c_o, sc0, BLK) - 150 * 4.0 / 1e6) < 1e-12)
c_fs = next(c for c in cs if c["arm"] == "O+FS")
check("A21 G: an FS cell adds the 759-token experience block", abs(P.cell_usd(c_fs, sc0, BLK) - P.cell_usd(c_o, sc0, BLK) - 759 * 4.0 / 1e6) < 1e-12)
tail = dict(sc0, tail=500)
check("A21 G: a thinking tail is billed at the OUTPUT price (500 tokens x 20 USD/M for Opus)", abs(P.cell_usd(c_o, tail, BLK) - P.cell_usd(c_o, sc0, BLK) - 500 * 20.0 / 1e6) < 1e-12)
check("A21 G: the cut order lists a PREFIX of the registered order and stops as soon as the total fits",
      all(x["cut_needed_in_order"] == list(C.CUT_ORDER[:len(x["cut_needed_in_order"])]) for x in sc))
big = P.final_projection({"test": 250, "repeat": 60, "relocation": 20, "dev": 24, "manifest_sha256": "x"}, None, BLK, treatment="ft-v1")
s_ref = list(big["scenarios"].values())
check("A37 on the real sizes (legacy list-price model): ceilings core 60 / total 100 under the 180 USD cap and the 80 USD gate; S1 / S2 / S3 "
      "all fit both (under A24.1-C's 30 / 50 the S3 core did not); each within-flag agrees with the LITERAL 60 / 100 bound",
      big["ceilings"] == {"core": 60.0, "total": 100.0} and big["cap_usd"] == 180.0 and big["gate_usd"] == 80.0
      and all(x["total_within_ceiling"] and x["core_within_ceiling"] for x in s_ref)
      and all((x["usd_core"] <= 60.0) == x["core_within_ceiling"] and (x["usd_total"] <= 100.0) == x["total_within_ceiling"] for x in s_ref))
# Astra K23-A37-ALIGN-r1 2: negative coverage. (a) at 2.5 x the real sizes the larger scenarios exceed the literal 60 / 100 and their flags
# say so (computed flags, not constants); (b) boundary semantics with the test-v1 ceilings monkeypatched for one call each (restored): the
# projection reports 2-decimal values (internal value within +-0.005), so a ceiling 0.006 above the reported value is within and 0.006 below
# is not (exact-equality semantics, pause only ABOVE, are covered by the cost-guard cases: mean exactly 0.080, projection exactly 100.00)
_big25 = P.final_projection({"test": 625, "repeat": 150, "relocation": 50, "dev": 60, "manifest_sha256": "x"}, None, BLK, treatment="ft-v1")
s_25 = list(_big25["scenarios"].values())
check("A37 negative projection: at 2.5 x the real sizes some scenario exceeds the literal core 60 and total 100, some does not, and every flag "
      "agrees with the literal bound",
      any(x["usd_core"] > 60.0 for x in s_25) and any(x["usd_core"] <= 60.0 for x in s_25) and any(x["usd_total"] > 100.0 for x in s_25)
      and all((x["usd_core"] <= 60.0) == x["core_within_ceiling"] and (x["usd_total"] <= 100.0) == x["total_within_ceiling"] for x in s_25)
      and not all(x["core_within_ceiling"] for x in s_25) and not all(x["total_within_ceiling"] for x in s_25), [(x["usd_core"], x["usd_total"]) for x in s_25])
_ce0 = dict(C.NAMESPACES["test-v1"]["ceilings"])
_s3 = s_ref[2]
_REAL = {"test": 250, "repeat": 60, "relocation": 20, "dev": 24, "manifest_sha256": "x"}
try:
    C.NAMESPACES["test-v1"]["ceilings"] = {"core": _s3["usd_core"] + 0.006, "total": _s3["usd_total"] + 0.006}
    _eq = list(P.final_projection(_REAL, None, BLK, treatment="ft-v1")["scenarios"].values())[2]
    C.NAMESPACES["test-v1"]["ceilings"] = {"core": _s3["usd_core"] - 0.006, "total": _s3["usd_total"] - 0.006}
    _lo = list(P.final_projection(_REAL, None, BLK, treatment="ft-v1")["scenarios"].values())[2]
finally:
    C.NAMESPACES["test-v1"]["ceilings"] = _ce0
# Astra K23-A37-ALIGN completion review 2: EQUALITY at the LITERAL ceilings in force (60 / 100, not monkeypatched). Controlled projections: one
# core and one other template cell of the real roster, 1 call each, priced by a stubbed cell_usd at exact binary values (restored), read through
# final_projection's OWN within-flags in every scenario: below, exactly at (within: pause only ABOVE), core above with the total exactly at,
# total above with the core exactly at.
_core_c = next(c for c in cs if c["scope"] == "core")
_oth_c = next(c for c in cs if c["scope"] == "other")
_cu0 = P.cell_usd


def _lit_flags(core, other):
    try:
        P.cell_usd = lambda c, sc, b: c["_usd"]
        s = P.final_projection(_REAL, [dict(_core_c, calls=1, _usd=core), dict(_oth_c, calls=1, _usd=other)], BLK, treatment="ft-v1")
    finally:
        P.cell_usd = _cu0
    return {(x["usd_core"], x["usd_total"], x["core_within_ceiling"], x["total_within_ceiling"]) for x in s["scenarios"].values()}


_lit = {"below": _lit_flags(59.5, 40.0), "at": _lit_flags(60.0, 40.0), "core_above": _lit_flags(60.5, 39.5), "total_above": _lit_flags(60.0, 40.5)}
check("A37 projection boundary: (a) a ceiling just above the reported core / total (+0.006) is within, just below (-0.006) is not; (b) at the LITERAL "
      "ceilings 60 / 100 (controlled projections, callplan's own flags): 59.50 / 99.50 within, EXACTLY 60.00 / 100.00 within, core 60.50 refused "
      "while its total 100.00 is within, total 100.50 refused while its core 60.00 is within (ceilings and cell_usd restored)",
      _eq["core_within_ceiling"] and _eq["total_within_ceiling"] and not _lo["core_within_ceiling"] and not _lo["total_within_ceiling"]
      and _lit == {"below": {(59.5, 99.5, True, True)}, "at": {(60.0, 100.0, True, True)}, "core_above": {(60.5, 100.0, False, True)},
                   "total_above": {(60.0, 100.5, True, False)}}
      and C.NAMESPACES["test-v1"]["ceilings"] == {"core": 60.0, "total": 100.0} and P.cell_usd is _cu0, (_eq.get("usd_core"), _lo.get("usd_core"), _lit))
check("A21 G: the projection carries the simulator upper bound and the X-main cell count", big["simulator"]["episodes_upper_bound"] > 0 and big["x_main_arms"]["cells"] == 500)
check("A21 G: the measured block means come from the real-token gate file", P.measured_blocks()["fs"] == 759 and P.measured_blocks()["source"] == "freeze_tokens.json")

# ---- A24.10: the 100 USD cap and the 80 USD cost gate; A23: archived arms add nothing in test-v1 -------------------------------------
check("A37 constants: cap 180 (A24.10: 100), gate 80 (unchanged), contingency 100; test-v1 core 60 / total 100 (A24.1-C: 30 / 50)",
      C.ANTHROPIC_CAP_USD == 180.0 and C.COST_GATE_USD == 80.0 and C.CONTINGENCY_USD == 100.0 and C.NAMESPACES["test-v1"]["ceilings"] == {"core": 60.0, "total": 100.0})
g0 = P.cost_gate(0.0, SZ, cs, treatment="ft-v1")   # TREAT-FREEZE: legacy model named explicitly (the default is ft-v2 after the flip)
check("A24.10 gate: S2 projection at the package maximum, acceptance incl. a possible full re-run, rehearsal and the retained blocks",
      g0["scenario"] == P.GATE_SCENARIO and g0["blocks_tokens"]["memory"] == 2000 and g0["remaining"]["acceptance_rerun_included"]
      and abs(g0["remaining"]["total"] - (g0["remaining"]["retained_blocks"] + g0["remaining"]["acceptance"] + g0["remaining"]["rehearsal"] + g0["remaining"]["practice"])) < 0.02)
g1 = P.cost_gate(10.0, SZ, cs, treatment="ft-v1")
check("A24.10 gate: every dollar already spent counts (spend adds one-for-one)", abs(g1["projected_total_usd"] - g0["projected_total_usd"] - 10.0) < 0.02)
check("A24.10 gate: without the re-run the acceptance part halves",
      abs(P.cost_gate(0.0, SZ, cs, acceptance_rerun=False, treatment="ft-v1")["remaining"]["acceptance"] * 2 - g0["remaining"]["acceptance"]) < 0.02)
gbig = P.cost_gate(100.0, SZ, cs, treatment="ft-v1")
check("A24.10 gate: fails when spend + projection > 80; the cut is a prefix of the registered order; core / Gemma / FS / X are never cut",
      (not gbig["gate_pass"]) and gbig["cut_needed_in_order"] == list(C.CUT_ORDER[:len(gbig["cut_needed_in_order"])]) and gbig["cut_needed_in_order"] != []
      and not gbig["fits_after_cut"] and "never cut" in gbig["if_cannot_fit"])
rem0 = g0["remaining"]["total"]
check("A24.10 gate: just over 80 fails, just under passes; margin sign matches",
      (not P.cost_gate(80.0 - rem0 + 0.5, SZ, cs, treatment="ft-v1")["gate_pass"]) and P.cost_gate(80.0 - rem0 - 0.5, SZ, cs, treatment="ft-v1")["gate_pass"]
      and P.cost_gate(80.0 - rem0 + 0.5, SZ, cs, treatment="ft-v1")["margin_to_gate_usd"] < 0)
check("A24.10 gate: the 2-repair line is a sensitivity (not gated) and exceeds the S2 retained blocks", g0["worst_two_repairs_usd"] > g0["remaining"]["retained_blocks"])
real = P.cost_gate(5.0, treatment="ft-v1")
check("A24.10 real sizes: with 5 USD spent the gate passes with margin; only Opus is priced", real["gate_pass"] and real["margin_to_gate_usd"] > 0)
try:
    P.cells(SZ, "archived-nope")
    ok_ref = False
except ValueError:
    ok_ref = True
C.NAMESPACES["archived-sh-v1"] = {"items": "dev_pilot", "ceilings": {"core": 0.0, "total": 1.0}, "archived_arms": ("S0", "S+M", "H0", "H+M")}
try:
    ca = P.cells(SZ, "archived-sh-v1")
    ok_opt = ({c["arm"] for c in ca} & set(C.ARCHIVED_ARMS)) == set(C.ARCHIVED_ARMS) and {c["model_key"] for c in ca} >= {"S", "H"} \
        and all(c["scope"] == "other" for c in ca if c["arm"] in C.ARCHIVED_ARMS)
    ok_cost = P.cost_gate(0.0, SZ, ca, treatment="ft-v1")["remaining"]["retained_blocks"] > g0["remaining"]["retained_blocks"]
finally:
    C.NAMESPACES.pop("archived-sh-v1", None)
check("A23 an unknown namespace is refused", ok_ref)
check("A23 an opted-in namespace carries Sonnet / Haiku rows (priced as 'other') and its projection is higher", ok_opt and ok_cost)
check("A23 test-v1 cells carry no archived arm", not ({c["arm"] for c in cs} & set(C.ARCHIVED_ARMS)))

# ---- A27 (draft): the real execution units incl. the P0 / P+M and /p controls, before and after (Astra runtime-cuts.txt)
REAL = {"test": 250, "repeat": 60, "relocation": 20, "dev": 24, "manifest_sha256": "x"}
en = P.execution_enumeration(REAL)
ea, eb = en["after"]["summary"], en["before"]["summary"]
check("A27 enumeration: listed matrix 2,000 + 1,200 + 360 + 80 = 3,640 executions (19.2 h at 19 s), Astra's figure",
      ea["listed_matrix"] == {"episodes": 3640, "hours": 19.2}, ea["listed_matrix"])
check("A27 enumeration: with P0 / P+M (540) and /p replay (1,140) controls 5,320 executions (28.1 h); 0.778 dedup -> 21.8 h",
      ea["executions_by_group"] == {"plan_arm": 3640, "control_P": 540, "control_replay": 1140} and ea["with_controls"] == {"episodes": 5320, "hours": 28.1}
      and ea["dedup"]["uniform_0.778"]["hours"] == 21.8, ea)
check("A27 enumeration: before A27 8,320 listed / 10,360 with controls; plan cells 5,440 -> 4,000 (1,080 plan-only Codex cells, 0 executions)",
      eb["listed_matrix"]["episodes"] == 8320 and eb["with_controls"]["episodes"] == 10360 and eb["plan_cells"] == 5440 and ea["plan_cells"] == 4000
      and ea["plan_only_cells"] == 1080 and ea["executions_by_block"]["codex"] == 0, (eb, ea))
check("A27 enumeration: every row carries cells >= 0 and executions >= 0; plan-only rows have 0 executions",
      all(r["executions"] >= 0 and r["cells"] >= 0 for r in en["after"]["rows"]) and all(r["executions"] == 0 for r in en["after"]["rows"] if r.get("plan_only")))
check("A27 constants: one source in common.py (callplan / plan / capture agree)",
      P.CURVE_SNAPSHOTS == [tuple(x) for x in C.CURVE_SNAPSHOTS] == [(0, 1), (0, 3), (0, 5)] and C.EXEC_RUNS_REPEAT == 2 and C.plan_only("X-gpt-6-luna0")
      and not C.plan_only("X0") and C.repeat_exec_arm("G+M/p") and C.repeat_exec_arm("O0") and not C.repeat_exec_arm("G+FS") and not C.repeat_exec_arm("X+M")
      and not C.repeat_exec_arm("P0"))

# ---- C4-open (b): ft-v2 = up to 3 calls per cell (A24 items 2 and 10; Astra docs/final-test/_astra/C4-open.txt Q-b)
import treatment as T   # noqa: E402
# TREAT-FREEZE 2026-10-13: replaces "C4-open: test-v1 is still ft-v1, so the default gate keeps the legacy numbers byte for byte".
check("C4-open: test-v1 is flipped (TREAT-FREEZE), so the default gate is the ft-v2 model; the legacy model stays byte for byte when named",
      P.test_treatment() == T.TEST_V1_TREATMENT == "ft-v2" and P.cost_gate(0.0, SZ, cs) == P.cost_gate(0.0, SZ, cs, treatment="ft-v2")
      and "treatment" not in g0 and P.cost_gate(0.0, SZ, cs, treatment="ft-v1") == g0)
TOK0 = {"I0": 1000, "A": 100, "R": 100, "F": 10, "P": 50}
sc0 = {"tail": 0, "repair": 0.0, "failed": 0.0, "block": "measured", "_name": "x"}
c_o = next(c for c in cs if c["arm"] == "O0" and c["block"] == "main")
c_om = next(c for c in cs if c["arm"] == "O+M" and c["block"] == "main")
c_fs = next(c for c in cs if c["arm"] == "O+FS" and c["block"] == "main")
c_g = next(c for c in cs if c["arm"] == "G0" and c["block"] == "main")
BLK2 = {"memory": 300, "fs": 500, "max": 2000}
k1 = (1000 * 4.0 + 100 * 20.0) / 1e6
check("C4-open: call 1 = I0 x input price + A x output price; no repair -> the cell is exactly call 1",
      abs(P.ft2_call_usd(1, "O", 0, sc0, P.ft2_tokens(TOK0)) - k1) < 1e-12 and abs(P.ft2_cell_usd(c_o, sc0, BLK2, TOK0) - k1) < 1e-12)
check("C4-open: the conversation grows: call k adds (k-1) x (R + F + P) input tokens",
      abs(P.ft2_call_usd(3, "O", 0, sc0, P.ft2_tokens(TOK0)) - P.ft2_call_usd(1, "O", 0, sc0, P.ft2_tokens(TOK0)) - 2 * 160 * 4.0 / 1e6) < 1e-12)
check("C4-open: the +M block (memory) and the FS block (fs) are added to every call's input",
      abs(P.ft2_cell_usd(c_om, sc0, BLK2, TOK0) - k1 - 300 * 4.0 / 1e6) < 1e-12 and abs(P.ft2_cell_usd(c_fs, sc0, BLK2, TOK0) - k1 - 500 * 4.0 / 1e6) < 1e-12)
sc_r = dict(sc0, repair=0.2)
c1_, c2_, c3_ = (P.ft2_call_usd(k, "O", 0, sc_r, P.ft2_tokens(TOK0)) for k in (1, 2, 3))
check("C4-open: expected cell = c1 + p1 c2 + p1 p2 c3 (p2 given explicitly)", abs(P.ft2_cell_usd(c_o, sc_r, BLK2, TOK0, p2=0.5) - (c1_ + 0.2 * c2_ + 0.1 * c3_)) < 1e-12)
check("C4-open: the tail is billed at the output price on EVERY call (repairs included)",
      abs(P.ft2_cell_usd(c_o, dict(sc_r, tail=500), BLK2, TOK0, p2=0.5) - P.ft2_cell_usd(c_o, sc_r, BLK2, TOK0, p2=0.5) - 500 * 20.0 / 1e6 * (1 + 0.2 + 0.1)) < 1e-12)
check("C4-open: failed multiplies the whole cell once (extra billed workload)",
      abs(P.ft2_cell_usd(c_o, dict(sc_r, failed=0.05), BLK2, TOK0, p2=0.5) - 1.05 * P.ft2_cell_usd(c_o, sc_r, BLK2, TOK0, p2=0.5)) < 1e-12)
check("C4-open: worst = all 3 calls; > the expected value; Gemma and Codex cost 0",
      abs(P.ft2_cell_usd(c_o, sc_r, BLK2, TOK0, worst=True) - (c1_ + c2_ + c3_)) < 1e-12 and P.ft2_cell_usd(c_g, sc_r, BLK2, TOK0, worst=True) == 0.0)
check("C4-open: S1 / S2 / S3 each carry a p2 (S2 = 0.5); expected calls per cell S2 = 1 + .15 + .075",
      set(P.FT2_P2) == set(P.SCENARIOS) and P.FT2_P2[P.GATE_SCENARIO] == 0.5 and abs(P.ft2_expected_calls(P.GATE_SCENARIO) - 1.225) < 1e-12)
gf = P.cost_gate(0.0, SZ, cs, treatment="ft-v2")
check("C4-open: the ft-v2 gate has the same workload terms (retained + acceptance with re-run + rehearsal + practice) and states its model",
      gf["treatment"] == "ft-v2" and "3 calls" in gf["cost_model"] and gf["remaining"]["acceptance_rerun_included"]
      and abs(gf["remaining"]["total"] - (gf["remaining"]["retained_blocks"] + gf["remaining"]["acceptance"] + gf["remaining"]["rehearsal"])) < 0.02)
check("C4-open: under ft-v2 the gate is dearer than the legacy 1-repair prior model (more calls, larger prompts)",
      gf["remaining"]["total"] > g0["remaining"]["total"], (gf["remaining"], g0["remaining"]))
check("C4-open: sensitivities on the SAME workload: every-cell-3-calls > repair-output-x3 > the gated S2 value",
      gf["sensitivity_every_cell_3_calls_usd"] > gf["sensitivity_repair_output_x3_usd"] > gf["projected_total_usd"])
check("C4-open: spend adds one-for-one under ft-v2 too", abs(P.cost_gate(7.0, SZ, cs, treatment="ft-v2")["projected_total_usd"] - gf["projected_total_usd"] - 7.0) < 0.02)
gfb = P.cost_gate(200.0, SZ, cs, treatment="ft-v2")
check("C4-open: ft-v2 gate fails above 80 with the registered cut order; protected arms never cut",
      not gfb["gate_pass"] and gfb["cut_needed_in_order"] == list(C.CUT_ORDER[:len(gfb["cut_needed_in_order"])]) and "never cut" in gfb["if_cannot_fit"])
check("C4-open: the gate prices +M / FS at the 2,000 maximum by default; token priors are labelled provisional",
      gf["blocks_tokens"]["memory"] == gf["blocks_tokens"]["fs"] == 2000 and "provisional" in gf["tokens"]["status"])
check("C4-open: a recounted I0 (post-sweep prompts) raises the gate monotonically",
      P.cost_gate(0.0, SZ, cs, treatment="ft-v2", tokens={"I0": 4000})["projected_total_usd"] > gf["projected_total_usd"])
fpf = P.final_projection(SZ, cs, None, treatment="ft-v2")
check("C4-open: final_projection under ft-v2: per-scenario totals, S3 >= S2 >= S1, every-cell-3-calls reported",
      fpf["treatment"] == "ft-v2" and fpf["scenarios"][list(P.SCENARIOS)[2]]["usd_total"] >= fpf["scenarios"][P.GATE_SCENARIO]["usd_total"]
      >= fpf["scenarios"][list(P.SCENARIOS)[0]]["usd_total"] and all(v["every_cell_3_calls_usd_total"] >= v["usd_total"] for v in fpf["scenarios"].values()))
check("C4-open: final_projection: the named ft-v1 model is unchanged; the default (after the TREAT-FREEZE flip) is the ft-v2 projection",
      P.final_projection(SZ, cs, BLK, treatment="ft-v1") == fp and P.final_projection(SZ, cs, None) == fpf)
# A31 step 5: was P.ft2_measured_blocks("rehearsal-v3") on the REAL root, which stops holding once a real ft-v2 capture exists there
# (the A31 rehearsal capture); the fallback is now checked on an empty temporary root, and the real capture is checked to be measured.
_root_cp = tempfile.mkdtemp(prefix="ftm-callplan-")
check("C4-open: ft2_measured_blocks falls back to the maximum when a namespace has no ft-v2 capture",
      P.ft2_measured_blocks("rehearsal-v3", root=_root_cp)["memory"] == 2000)
print("   INFO ft-v2 gate on the real sizes at 0 USD spent: %.2f USD (legacy model %.2f); every cell 3 calls %.2f; I0 4,000: %.2f" % (
    P.cost_gate(0.0, treatment="ft-v2")["projected_total_usd"], P.cost_gate(0.0, treatment="ft-v1")["projected_total_usd"],
    P.cost_gate(0.0, treatment="ft-v2")["sensitivity_every_cell_3_calls_usd"],
    P.cost_gate(0.0, treatment="ft-v2", tokens={"I0": 4000})["projected_total_usd"]))

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
