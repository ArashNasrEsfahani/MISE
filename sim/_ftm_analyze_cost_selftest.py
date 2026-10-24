"""Selftest for the A24.2 cost-effectiveness block of sim/final_test/matrix/analyze.py (stage P-cost). Offline, synthetic data only: no simulator, no
network, no API, no test item, no real result.

    python sim/_ftm_analyze_cost_selftest.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "final_test", "matrix")
sys.path.insert(0, HERE)
sys.path.insert(0, MATRIX)

import common as C                      # noqa: E402
import analyze as A                     # noqa: E402
ROOT = C.paths()["root"]

N = {"pass": 0, "fail": 0}
BB = 2000                                # bootstrap replicates for the selftest (production 20000)


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def close(a, b, tol=1e-12):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def mk(arm, item, lang, sr, fam, lvl="L1"):
    return {"key": f"{arm}|{item}|{lang}", "namespace": "rehearsal-v1", "arm": arm, "pres": C.pres_id(item, lang), "r": 0, "run": 0,
            "unit": f"u|{arm}|{item}|{lang}", "sr": float(sr), "strict_sr": float(sr), "valid": True, "infra": False, "level": lvl, "lang": lang,
            "family": fam}


def prow(arm, item, lang, *, usd=0.0, wall=1.0, gpu=None, cpu=None, r=0, block="main", **kw):
    row = {"key": f"p|{arm}|{item}|{lang}|{r}|{block}", "arm": arm, "pres": C.pres_id(item, lang), "r": r, "block": block, "cost_usd": usd,
           "wall_s": wall, "tokens_in": 0, "tokens_out": 0, **kw}
    if gpu is not None:
        row["energy_j_gpu"] = gpu
    if cpu is not None:
        row["energy_j_cpu"] = cpu
    return row


def world(sr_g, sr_o, n_fam=12, per_fam=2, g_cost=(1.0, 360.0, 0.0), o_usd=0.01, extra_arms=()):
    """score rows + plan rows. sr_g / sr_o: function(k, lang) -> 0/1. g_cost = (wall_s, gpu J, cpu J) per Gemma plan row."""
    rows, plan, k = [], [], 0
    for f in range(n_fam):
        for j in range(per_fam):
            item = f"c-{f:02d}-{j}"
            for lang in A.LANGS:
                rows.append(mk("G+M", item, lang, sr_g(k, lang), f"fam{f:02d}"))
                rows.append(mk("O0", item, lang, sr_o(k, lang), f"fam{f:02d}"))
                plan.append(prow("G+M", item, lang, wall=g_cost[0], gpu=g_cost[1], cpu=g_cost[2]))
                plan.append(prow("O0", item, lang, usd=o_usd, wall=8.0))
                for arm in extra_arms:
                    rows.append(mk(arm, item, lang, sr_o(k, lang), f"fam{f:02d}"))
                    plan.append(prow(arm, item, lang, wall=30.0))
            k += 1
    return rows, plan


ZERO_PRAC = [{"cost_usd": 0.0, "plan_secs": 0.0, "energy_j_gpu": 0.0, "energy_j_cpu": 0.0, "cost_kind": "gemma"}]   # measured, zero-cost practice


def block(rows, plan, prep=ZERO_PRAC, complete=True, cfg=None, h1=None, drop=None, **kw):
    idx = A.build_index(rows)
    S = idx["S"]
    if drop:                                              # remove scored cells (arm, item, lang) to break cohort coverage
        for k in drop:
            S.pop(k, None)
    arms = sorted({a for (a, i, l) in S})
    return A.cost_effectiveness_block(S, idx["fam"], arms, A.invalid_presentations(S), plan, prep, cfg=CFG_GEN if cfg is None else cfg, h1=h1,
                                      complete=complete, B=BB, **kw)


cfg0 = A.A24_2_COST
#: C5-SIMPLE: the plan's practice-row rule is ACTIVE (Arash 2026-10-14) and needs the 10 real practice-v2 rows; the generic synthetic worlds below
#: run with the rule switched off (the C5 block tests the rule itself)
CFG_GEN = dict(cfg0, energy_method=dict(cfg0["energy_method"], unmeasured_rule_basis=None, unmeasured_rule_terms=None))

# ---------------------------------------------------------------------------------------------------- 1. the registered constants
check("constants: 0.20 USD/kWh, 399 USD over 6,000 h, Jetson factor 3, claim .80 / .10", cfg0["usd_per_kwh"] == 0.20 and cfg0["hardware_price_usd"] == 399.0
      and cfg0["operating_hours"] == 6000.0 and cfg0["hardware_time_factor"] == 3.0 and cfg0["claim"]["R_lower_min"] == 0.80 and cfg0["claim"]["C_upper_max"] == 0.10)
check("A24.2 accounting is part of the hashed analysis plan", A.ANALYSIS_PLAN.get("a24_2_cost") is cfg0)
sha0 = A.analysis_plan_sha()
A.ANALYSIS_PLAN["a24_2_cost"] = dict(cfg0, usd_per_kwh=0.21)
sha1 = A.analysis_plan_sha()
A.ANALYSIS_PLAN["a24_2_cost"] = cfg0
check("changing a cost constant changes the analysis-plan sha", sha0 != sha1 and A.analysis_plan_sha() == sha0)
check("claim text is the A24.2 sentence", cfg0["claim"]["text"] == "Gemma with the package retained at least 80% of unassisted Opus's success rate at at most "
      "10% of its specified planning cost")

# ---------------------------------------------------------------------------------------------------- 2. pricing one row
pr = A.price_row({"wall_s": 3600.0, "energy_j_gpu": 3_000_000.0, "energy_j_cpu": 600_000.0, "cost_usd": 0.0}, "gemma", cfg0)
check("energy: 3.6 MJ = 1 kWh = 0.20 USD", close(pr["energy"], 0.20) and close(pr["joules"], 3_600_000.0), str(pr))
check("hardware: 1 planning hour = 399 / 6000 = 0.0665 USD", close(pr["hardware"], 0.0665), str(pr))
check("hardware formula is linear in planning time (90 s)", close(A.price_row({"wall_s": 90.0, "energy_j_gpu": 0, "energy_j_cpu": 0}, "gemma", cfg0)["hardware"],
                                                                 399.0 / 6000.0 * 90.0 / 3600.0))
check("planning_s overrides wall_s", close(A.price_row({"planning_s": 36.0, "wall_s": 3600.0, "energy_j_gpu": 0, "energy_j_cpu": 0}, "gemma", cfg0)["hardware"],
                                           0.0665 / 100))
check("no planning_s / wall_s: NO latency-sum fallback (it would omit parsing / retrieval time) -> Gemma row UNKNOWN",
      A.price_row({"attempts": [{"latency_s": 1800.0}], "energy_j_gpu": 0, "energy_j_cpu": 0}, "gemma", cfg0)["known"] is False)
check("C5-SIMPLE: a Gemma row with only the GPU field is known (CPU not measured); without the GPU field it is UNKNOWN",
      A.price_row({"wall_s": 1, "energy_j_gpu": 5}, "gemma", cfg0)["known"] is True
      and close(A.price_row({"wall_s": 1, "energy_j_gpu": 5}, "gemma", cfg0)["joules"], 5.0)
      and A.price_row({"wall_s": 1, "energy_j_cpu": 5}, "gemma", cfg0)["known"] is False)
check("a Gemma row without any time is UNKNOWN", A.price_row({"energy_j_gpu": 5, "energy_j_cpu": 1}, "gemma", cfg0)["known"] is False)
pa = A.price_row({"wall_s": 9.0, "cost_usd": 0.0123}, "api", cfg0)
check("API row: cost_usd only, no energy or hardware", pa["known"] and close(pa["api"], 0.0123) and pa["energy"] == 0 and pa["hardware"] == 0)
pp = A.price_row({"plan_secs": 36.0, "wall_s": 999.0, "energy_j_gpu": 0, "energy_j_cpu": 0}, "gemma", cfg0, prep=True)
check("practice row: plan_secs is priced, never the episode wall_s", close(pp["seconds"], 36.0) and close(pp["hardware"], 0.0665 / 100), str(pp))
check("practice row without plan_secs / planning_s has no time (UNKNOWN for Gemma)",
      A.price_row({"wall_s": 999.0, "energy_j_gpu": 0, "energy_j_cpu": 0}, "gemma", cfg0, prep=True)["known"] is False)
check("arm kinds: G* gemma, O* api, X* codex", [A.arm_kind(a) for a in ("G0", "G+M", "G+FS", "O0", "O+M", "X0", "X+M")]
      == ["gemma"] * 3 + ["api"] * 2 + ["codex"] * 2)

# ---------------------------------------------------------------------------------------------------- 3. power integration helper
check("integrate: 50 W for 10 s = 500 J", close(A.integrate_power_j([(0, 50), (5, 50), (10, 50)], 0, 10), 500.0))
check("integrate: linear ramp 0 -> 100 W over 10 s = 500 J, window [2, 8] = 300 J",
      close(A.integrate_power_j([(0, 0), (10, 100)], 0, 10), 500.0) and close(A.integrate_power_j([(0, 0), (10, 100)], 2, 8), 300.0))
check("integrate: window not covered by samples -> None", A.integrate_power_j([(1, 50), (10, 50)], 0, 5) is None and A.integrate_power_j([(0, 5)], 0, 1) is None)

# ---------------------------------------------------------------------------------------------------- 4. bootstrap ratio states
import numpy as np                                                    # noqa: E402
W = np.random.default_rng(1).multinomial(4, [0.25] * 4, size=500).astype(float)
br = A.boot_ratio([1, 1, 1, 1], [2, 2, 2, 2], W)
check("boot_ratio: constant ratio -> point .5 and CI [.5, .5]", close(br["estimate"], 0.5) and close(br["ci95"][0], 0.5) and close(br["ci95"][1], 0.5), str(br))
bi = A.boot_ratio([1, 1, 1, 1], [0, 0, 0, 1], W)
check("boot_ratio: replicates with denominator 0 and cost > 0 are infinite and KEPT (counted)", bi["n_infinite"] > 0 and bi["n_undefined"] == 0
      and bi["ci95"] is not None and bi["ci95"][1] == math.inf, str({k: bi[k] for k in ("n_infinite", "ci95")}))
bu = A.boot_ratio([0, 0, 0, 1], [0, 0, 0, 1], W)
check("boot_ratio: any 0 / 0 replicate -> bounds NOT obtainable, count reported", bu["n_undefined"] > 0 and bu["ci95"] is None and bu["bounds_obtainable"] is False
      and "undefined" in bu["reason"])
check("point states: cost > 0 with 0 successes = infinite; 0 / 0 = undefined",
      A.boot_ratio([1, 0], [0, 0], W[:, :2])["state"] == "infinite" and A.boot_ratio([0, 0], [0, 0], W[:, :2])["state"] == "undefined"
      and A.boot_ratio([0, 0], [0, 0], W[:, :2])["estimate"] is None)

# ---------------------------------------------------------------------------------------------------- 5. per-arm USD per success, mean per task, views
g_ok = lambda k, l: 1.0 if k % 4 != 0 else 0.0                       # noqa: E731   G+M: 18 of 24 items succeed (both languages)
o_ok = lambda k, l: 1.0 if k % 5 != 0 else 0.0                       # noqa: E731   O0: k in 0..23, 5 multiples -> 19 of 24
rows, plan = world(g_ok, o_ok, g_cost=(2.0, 720.0, 180.0), o_usd=0.01)
ce = block(rows, plan)
g, o = ce["by_arm"]["G+M"], ce["by_arm"]["O0"]
e_one = 900.0 / 3.6e6 * 0.20
h_one = 399.0 / 6000.0 * 2.0 / 3600.0
check("G+M costed; per-presentation energy and hardware exact", g["cost_state"] == "costed" and close(g["components_usd"]["energy"], 48 * e_one)
      and close(g["components_usd"]["hardware"], 48 * h_one), str(g.get("components_usd")))
check("G+M joules over the scored cells = 48 x 900 J", close(g["joules_scored"], 48 * 900.0))
check("USD per success (a) = total / successes (G+M 36 successes, O0 38)", close(g["views"]["headline"]["usd_per_success"]["estimate"], 48 * (e_one + h_one) / 36)
      and close(o["views"]["headline"]["usd_per_success"]["estimate"], 48 * 0.01 / 38))
check("mean USD per attempted task (O0) = 0.01", close(o["views"]["headline"]["mean_usd_per_attempted_task"]["estimate"], 0.01))
check("views: electricity_only = headline - hardware; hardware_only = hardware; hw_time_3x = energy + 3 x hardware",
      close(g["views"]["electricity_only"]["total_usd"], 48 * e_one) and close(g["views"]["hardware_only"]["total_usd"], 48 * h_one)
      and close(g["views"]["hw_time_3x"]["total_usd"], 48 * (e_one + 3 * h_one)))
check("hw_time_3x is labelled as a hardware-time sensitivity with laptop energy held fixed",
      cfg0["views"]["hw_time_3x"] == "3x hardware-time sensitivity; measured laptop energy held fixed")
check("latency per presentation includes the cell wall time (G+M 2 s, O0 8 s)", close(g["latency"]["mean_s_per_presentation"], 2.0)
      and close(o["latency"]["mean_s_per_presentation"], 8.0) and close(g["latency"]["total_planning_hours"], 96.0 / 3600))
check("SR item-weighted (G+M .75, O0 19/24)", close(g["sr_item_weighted"]["estimate"], 0.75) and close(o["sr_item_weighted"]["estimate"], 19 / 24))

# ---------------------------------------------------------------------------------------------------- 6. R and C, matched, and the reading rule
rc = ce["R_C"]
check("R point = SR(G+M) / SR(O0) on the matched set", close(rc["R"]["estimate"], 0.75 / (19 / 24)), str(rc["R"]))
check("C point (headline) = mean G+M cost per task / 0.01", close(rc["C"]["headline"]["estimate"], (e_one + h_one) / 0.01))
check("C views: headline, electricity_only, hw_time_3x (no hardware-only C)", set(rc["C"]) == {"headline", "electricity_only", "hw_time_3x"})
check("cohort: 48 presentations, both arms cover it", ce["cohort_presentations"] == 48 and rc["cohort_complete"] is True
      and g["missing_vs_cohort"] == 0 and o["missing_vs_cohort"] == 0)
check("R CI brackets the point and is finite", rc["R"]["ci95"][0] <= rc["R"]["estimate"] <= rc["R"]["ci95"][1] and all(map(math.isfinite, rc["R"]["ci95"])))
cl = ce["claim"]
check("reading rule computed from lower R and upper C of the headline view", close(cl["R_lower"], rc["R"]["ci95"][0])
      and close(cl["C_upper"], rc["C"]["headline"]["ci95"][1]) and cl["statement_permitted"] == (cl["R_lower"] >= 0.8 and cl["C_upper"] <= 0.1), str(cl))
# a strong G+M: statement permitted
rows_s, plan_s = world(lambda k, l: 1.0, lambda k, l: 1.0, g_cost=(1.0, 90.0, 10.0))
ce_s = block(rows_s, plan_s)
check("equal SR + tiny Gemma cost -> R = 1, C << .10, statement PERMITTED", ce_s["claim"]["statement_permitted"] is True and close(ce_s["R_C"]["R"]["estimate"], 1.0),
      str(ce_s["claim"]))
ce_inc = block(rows_s, plan_s, complete=False)
check("incomplete study: statement not permitted (descriptive only)", ce_inc["claim"]["statement_permitted"] is False and "incomplete" in ce_inc["claim"]["reason"])
# R too low: G+M half the success of O0
rows_l, plan_l = world(lambda k, l: 1.0 if k % 2 else 0.0, lambda k, l: 1.0, g_cost=(1.0, 90.0, 10.0))
ce_l = block(rows_l, plan_l)
check("R .5 -> rule NOT met although C is tiny", ce_l["claim"]["statement_permitted"] is False and "not met" in ce_l["claim"]["reason"]
      and ce_l["R_C"]["R"]["ci95"][1] < 0.8)
# C too high: Gemma planning hour-scale (hardware dominates) vs a cheap API arm
rows_c, plan_c = world(lambda k, l: 1.0, lambda k, l: 1.0, g_cost=(3600.0, 0.0, 0.0), o_usd=0.01)
ce_c = block(rows_c, plan_c)
check("C > .10 (Gemma 1 h per plan = 0.0665 USD vs 0.01) -> rule NOT met", ce_c["claim"]["statement_permitted"] is False
      and ce_c["R_C"]["C"]["headline"]["estimate"] > 0.1)
# O0 zero successes: R infinite point, H1 not involved
rows_z, plan_z = world(lambda k, l: 1.0, lambda k, l: 0.0)
ce_z = block(rows_z, plan_z)
check("O0 SR = 0 -> R infinite (never a silent blank); bounds infinite -> rule not met",
      ce_z["R_C"]["R"]["state"] == "infinite" and ce_z["claim"]["statement_permitted"] is False, str(ce_z["R_C"]["R"]))
rows_zz, plan_zz = world(lambda k, l: 0.0, lambda k, l: 0.0)
ce_zz = block(rows_zz, plan_zz)
check("both SR 0 -> R undefined, bounds not obtainable, rule cannot be satisfied", ce_zz["R_C"]["R"]["state"] == "undefined"
      and ce_zz["R_C"]["R"]["ci95"] is None and ce_zz["claim"]["statement_permitted"] is False and "not obtainable" in ce_zz["claim"]["reason"])
check("an arm with cost and 0 successes: USD per success infinite", ce_zz["by_arm"]["O0"]["views"]["headline"]["usd_per_success"]["state"] == "infinite")
# H1 flag passes through
h1 = {"primary": {"decision": {"claims": {"ni": True}}}}
check("h1_established is read from the H1 block and noted", block(rows_s, plan_s, h1=h1)["claim"]["h1_established"] is True
      and ce_s["claim"]["h1_established"] is False)

# ---------------------------------------------------------------------------------------------------- 7. unknown energy blocks the claim
plan_u = [dict(p) for p in plan_s]
for p in plan_u:
    if p["arm"] == "G+M" and p["pres"].endswith(":fa"):
        p.pop("energy_j_gpu", None)
        break
ce_u = block(rows_s, plan_u)
check("one Gemma row without GPU energy -> G+M cost UNKNOWN, C not obtainable, rule cannot be satisfied",
      ce_u["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_u["R_C"]["C"]["headline"]["state"] == "unknown"
      and ce_u["claim"]["statement_permitted"] is False and "C:" in ce_u["claim"]["reason"], str(ce_u["claim"]))
plan_m = [p for p in plan_s if not (p["arm"] == "O0" and p["pres"] == C.pres_id("c-00-0", "en"))]
ce_m = block(rows_s, plan_m)
check("a scored O0 cell with no plan row -> O0 cost unknown (missing counted), rule cannot be satisfied",
      ce_m["by_arm"]["O0"]["presentations_without_plan_row"] == 1 and ce_m["claim"]["statement_permitted"] is False)
check("no plan journal -> block unavailable, statement not permitted", block(rows_s, None)["claim"]["statement_permitted"] is False)

# ---------------------------------------------------------------------------------------------------- 8. workload scope + allocation
plan_x = plan_s + [prow("O0", "c-00-0", "en", usd=5.0, r=1, block="repeat"), prow("O0", "c-00-1", "en", usd=7.0, block="curve")]
ce_x = block(rows_s, plan_x)
check("REPEAT r > 0 and other blocks are NOT in the per-task cost; totalled separately", close(ce_x["by_arm"]["O0"]["views"]["headline"]["total_usd"],
      ce_s["by_arm"]["O0"]["views"]["headline"]["total_usd"]) and close(ce_x["by_arm"]["O0"]["other_blocks_usd_api"], 12.0)
      and ce_x["by_arm"]["O0"]["other_blocks_rows"] == 2)
plan_r = plan_s + [prow("O0", "c-00-0", "en", usd=0.02, wall=3.0)]
ce_r = block(rows_s, plan_r)
check("every incurred attempt of a scored cell is summed (a second row for the same cell)",
      close(ce_r["by_arm"]["O0"]["views"]["headline"]["total_usd"], ce_s["by_arm"]["O0"]["views"]["headline"]["total_usd"] + 0.02))
rows_m, plan_mm = world(lambda k, l: 1.0, lambda k, l: 1.0, g_cost=(1.0, 90.0, 10.0), extra_arms=("O+M",))
plan_mm = [dict(p, cost_usd=0.01) if p["arm"] == "O+M" else p for p in plan_mm]
prac = [{"cost_usd": 0.0, "plan_secs": 3600.0, "wall_s": 99999.0, "energy_j_gpu": 3.6e6, "energy_j_cpu": 0.0, "cost_kind": "gemma"}]
ce_p = block(rows_m, plan_mm, prep=prac)
tot_p = 0.20 + 0.0665
al_g, al_o = ce_p["by_arm"]["G+M"]["allocated"], ce_p["by_arm"]["O+M"]["allocated"]
check("practice split across the +M arms by attempted presentations (equal here: half each), episode wall_s ignored",
      close(al_g["energy"] + al_g["hardware"], tot_p / 2) and close(al_o["energy"] + al_o["hardware"], tot_p / 2), str((al_g, al_o)))
check("O0 (no memory) carries no practice cost", ce_p["by_arm"]["O0"]["allocated"]["rows"] == 0)
check("allocated practice enters the per-task cost and the hardware part goes to hardware_only",
      close(ce_p["by_arm"]["O+M"]["views"]["headline"]["total_usd"], 48 * 0.01 + tot_p / 2)
      and close(ce_p["by_arm"]["O+M"]["views"]["hardware_only"]["total_usd"], 0.0665 / 2))
cfg_full = dict(CFG_GEN, allocation_rule=dict(cfg0["allocation_rule"], mode="full"))
ce_f = block(rows_m, plan_mm, prep=prac, cfg=cfg_full)
check("mode 'full': each consumer carries the whole practice row (upper bound)", close(ce_f["by_arm"]["G+M"]["allocated"]["energy"], 0.20)
      and close(ce_f["by_arm"]["O+M"]["allocated"]["energy"], 0.20))
ce_k = block(rows_m, plan_mm, prep=[{"cost_usd": 1.0, "plan_secs": 1.0}])
check("a prep row without cost_kind makes its targets' cost unknown", ce_k["by_arm"]["G+M"]["cost_state"].startswith("unknown")
      and ce_k["by_arm"]["O0"]["cost_state"] == "costed")
ce_a = block(rows_m, plan_mm, prep=[{"cost_usd": 0.5, "cost_kind": "api", "arm": "O+M"}])
check("a prep row with `arm` goes to that arm only", close(ce_a["by_arm"]["O+M"]["allocated"]["api"], 0.5) and ce_a["by_arm"]["G+M"]["allocated"]["rows"] == 0)
ce_gp = block(rows_m, plan_mm, prep=[{"cost_usd": 0.0, "plan_secs": 10.0, "cost_kind": "gemma"}])
# C5 (2026-10-14, after Astra C5 r1 #1): a Gemma practice row without energy (practice-v2 ran before C5) keeps an UNKNOWN cost unless an
# Arash-approved rule covers it; its seconds are reported either way
check("C5: a Gemma practice row without energy -> the +M arms UNKNOWN (no approved rule); unmeasured seconds reported",
      ce_gp["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_gp["by_arm"]["O+M"]["cost_state"].startswith("unknown")
      and ce_gp["by_arm"]["G+M"]["energy_unmeasured"]["allocated_rows"] == 1
      and close(ce_gp["by_arm"]["G+M"]["energy_unmeasured"]["allocated_seconds"] + ce_gp["by_arm"]["O+M"]["energy_unmeasured"]["allocated_seconds"], 10.0))
# C5 (Astra C5 r3 #1): the approved terms are applied ROW BY ROW. Practice rows are ESTIMATED at the run's measured Gemma rate (joined rows only),
# flagged, the measured figure kept as the primary one; an ineligible class, sensitivity_only, malformed terms or too few measured rows stay UNKNOWN.
EM0 = cfg0["energy_method"]


def cfg_terms(terms, basis="amendment:A99"):
    return dict(cfg0, energy_method=dict(EM0, unmeasured_rule_basis=basis, unmeasured_rule_terms=terms))


PRAC10 = [{"cost_usd": 0.0, "plan_secs": 10.0, "cost_kind": "gemma", "cost_source": "practice", "key": "prac|1"}]
PID = A.practice_row_id(PRAC10[0])


def tp(claim="headline", eligible=("practice",), ids=(PID,)):
    t = {"claim": claim, "eligible": list(eligible), "rate": "measured_rate"}
    if "practice" in eligible:
        t["rows"] = {"journal": "selftest/practice.jsonl", "journal_sha256": "0" * 64, "row_ids": list(ids)}
    return t


cfg_ap = cfg_terms(tp())
plan_meas = [dict(p, planning_s=p["wall_s"], energy_state="measured") if p["arm"] == "G+M" else p for p in plan_mm]   # 48 joined rows, 100 W
rate_ap = A.measured_rate(plan_meas, cfg_ap)
_UA = A.UNMEASURED_TERMS_APPROVED
_pv2 = os.path.join(ROOT, *_UA["rows"]["journal"].split("/"))
_pv2_rows = A.read_jsonl(_pv2)
check("C5: the approved terms = Arash's ledger decision (the 10 practice-v2 rows only, bound by journal sha + row content ids; headline with "
      "the estimate flagged; measured rate); ACTIVE in the plan under basis arash-decision:2026-10-14 (C5-SIMPLE)",
      {k: _UA[k] for k in ("claim", "eligible", "rate")} == {"claim": "headline", "eligible": ["practice"], "rate": "measured_rate"}
      and _UA["rows"]["journal"] == "docs/final-test/matrix/practice-v2/practice.jsonl"
      and _UA["rows"]["journal_sha256"] == hashlib.sha256(open(_pv2, "rb").read()).hexdigest()
      and sorted(A.practice_row_id(r) for r in _pv2_rows) == _UA["rows"]["row_ids"] and len(_pv2_rows) == 10
      and EM0["unmeasured_rule_terms"] == _UA and EM0["unmeasured_rule_basis"] == "arash-decision:2026-10-14" and A.approved_terms(cfg0) == _UA
      and A.unmeasured_terms_problems(_UA, EM0) == [])
check("C5: measured_rate = sum J / sum planning_s over JOINED (energy_state 'measured') Gemma rows only: 48 rows, 100 W",
      rate_ap["cells"] == 48 and close(rate_ap["rate_w"], 100.0) and A.measured_rate(plan_mm, cfg_ap)["rate_w"] is None
      and A.measured_rate(plan_mm, cfg_ap)["cells"] == 0)
few = [p for p in plan_meas if p["arm"] != "G+M"] + [p for p in plan_meas if p["arm"] == "G+M"][:EM0["unmeasured_estimate"]["min_cells"] - 1]
check("C5: fewer than min_cells measured rows -> no rate; one more -> a rate", A.measured_rate(few, cfg_ap)["rate_w"] is None
      and A.measured_rate(few + [dict(plan_meas[0], energy_state="measured", planning_s=1.0, key="x")], cfg_ap)["rate_w"] is not None)
check("C5: refused / incomplete rows never feed the rate", A.measured_rate(
      [dict(p, energy_state="unmeasured") if p["arm"] == "G+M" else p for p in plan_meas], cfg_ap)["cells"] == 0
      and A.measured_rate([dict(p, energy_j_gpu=None) if p["arm"] == "G+M" else p for p in plan_meas], cfg_ap)["cells"] == 0
      and A.measured_rate([dict(p, planning_s=0.0) if p["arm"] == "G+M" else p for p in plan_meas], cfg_ap)["cells"] == 0)
ce_gpa = block(rows_m, plan_meas, prep=PRAC10, cfg=cfg_ap)
ga, oa = ce_gpa["by_arm"]["G+M"], ce_gpa["by_arm"]["O+M"]
check("C5: under the approved terms the practice row is ESTIMATED at 100 W x 10 s, flagged, the +M arms costed; measured joules reported apart",
      ga["cost_state"] == "costed" and oa["cost_state"] == "costed"
      and close(ga["allocated"]["energy"] + oa["allocated"]["energy"], 100.0 * 10.0 / 3.6e6 * 0.20)
      and ga["energy_estimated"]["allocated_rows"] == 1
      and close(ga["energy_estimated"]["allocated_joules"] + oa["energy_estimated"]["allocated_joules"], 1000.0)
      and ga["energy_estimated"]["scored_rows"] == 0 and close(ga["energy_measured"]["scored_joules"], 48 * 100.0)
      and close(ga["energy_estimated"]["rate_w"], 100.0) and ce_gpa["energy_rate"]["active_terms"] == tp(),
      (ga["cost_state"], ga["energy_estimated"]))
cl_d = ce_gpa["claim"].get("energy_estimate_disclosure") or {}
check("C5: the claim carries the estimate disclosure (rows, joules, measured joules, rate) for G+M",
      cl_d.get("arms", {}).get("G+M", {}).get("rows") == 1 and close(cl_d["arms"]["G+M"]["measured_joules"], 4800.0)
      and close(cl_d.get("rate_w"), 100.0), cl_d)
md_e = "\n".join(A.render_cost_md(ce_gpa))
check("C5: markdown labels the estimate and the measured primary figure", "ESTIMATED (flagged, in the cost)" in md_e
      and "measured Gemma energy (primary)" in md_e and "Measured Gemma rate" in md_e)
ce_gpn = block(rows_m, plan_mm, prep=PRAC10, cfg=cfg_ap)
check("C5: approved terms but NO measured rate (no joined rows) -> the practice row stays UNKNOWN",
      ce_gpn["by_arm"]["G+M"]["cost_state"].startswith("unknown"))
T_SENS = tp(claim="sensitivity_only")
ce_gps = block(rows_m, plan_meas, prep=PRAC10, cfg=cfg_terms(T_SENS))
check("C5: sensitivity_only -> the practice row stays UNKNOWN (statement blocked); its estimate is reported, never priced",
      ce_gps["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_gps["by_arm"]["G+M"]["energy_estimated"]["allocated_rows"] == 0
      and ce_gps["claim"]["statement_permitted"] is False
      and close(ce_gps["by_arm"]["G+M"]["energy_unmeasured"]["usd_at_measured_rate"] + ce_gps["by_arm"]["O+M"]["energy_unmeasured"]["usd_at_measured_rate"],
                1000.0 / 3.6e6 * 0.20)
      and close(A.price_row(PRAC10[0], "gemma", cfg_terms(T_SENS), prep=True, rate_w=100.0, practice_ids={PID})["estimated_joules"], 1000.0)
      and A.price_row(PRAC10[0], "gemma", cfg_terms(T_SENS), prep=True, rate_w=100.0, practice_ids={PID})["known"] is False)
# the r3 probe: a practice-only approval must not price an unmeasured TEST plan row
hole_pres = next(p["pres"] for p in plan_meas if p["arm"] == "G+M")
plan_hole = []
for p in plan_meas:
    if p["arm"] == "G+M" and p["pres"] == hole_pres:
        p = {k: v for k, v in p.items() if k not in ("energy_j_gpu", "energy_j_cpu", "energy_state")}
        p["planning_s"] = 8.0
    plan_hole.append(p)
ce_h = block(rows_m, plan_hole, prep=PRAC10, cfg=cfg_ap)
check("C5 (r3 probe): practice-only terms leave an unmeasured 8 s TEST row UNKNOWN (no estimate), the statement blocked; its 8 s kept",
      ce_h["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_h["by_arm"]["G+M"]["energy_estimated"]["scored_rows"] == 0
      and ce_h["claim"]["statement_permitted"] is False and ce_h["by_arm"]["G+M"]["energy_unmeasured"]["scored_rows"] == 1
      and close(ce_h["by_arm"]["G+M"]["energy_unmeasured"]["scored_seconds"], 8.0)
      and A.price_row({"planning_s": 8.0}, "gemma", cfg_ap, rate_w=100.0)["known"] is False
      and A.price_row({"planning_s": 8.0, "energy_state": "unmeasured"}, "gemma", cfg_ap, rate_w=100.0)["known"] is False)
cfg_all = cfg_terms(tp(eligible=("practice", "no_energy_row")))
check("C5: only a class named in eligible is estimated (no_energy_row eligible -> the 8 s row = 800 J; refused_interval still UNKNOWN)",
      close(A.price_row({"planning_s": 8.0}, "gemma", cfg_all, rate_w=100.0)["joules"], 800.0)
      and A.price_row({"planning_s": 8.0}, "gemma", cfg_all, rate_w=100.0)["estimated"] is True
      and A.price_row({"planning_s": 8.0, "energy_state": "unmeasured"}, "gemma", cfg_all, rate_w=100.0)["known"] is False)
bad_terms = [None, {}, {"claim": "headline", "eligible": ["practice"], "rate": "assumed_w"},
             {"claim": "headline", "eligible": ["practice"], "rate": "1 W"},
             {"claim": "headline", "eligible": ["practice"], "rate": "measured_rate", "assumed_w": 1.0},
             {"claim": "headline", "eligible": [], "rate": "measured_rate"},
             {"claim": "headline", "eligible": ["test"], "rate": "measured_rate"},
             {"claim": "all", "eligible": ["practice"], "rate": "measured_rate"},
             {"claim": "headline", "eligible": ["practice", "practice"], "rate": "measured_rate"},
             {"claim": "headline", "eligible": "practice", "rate": "measured_rate"},
             {"claim": "headline", "eligible": ["practice"], "rate": "measured_rate"},                    # practice without a rows binding
             tp(ids=()), tp(ids=("x" * 64,)), tp(ids=(PID, PID)),
             dict(tp(), rows=dict(tp()["rows"], journal_sha256="short")), dict(tp(), rows=dict(tp()["rows"], extra=1)),
             dict(tp(eligible=("no_energy_row",)), rows=tp()["rows"]),                                     # rows without practice
             {"claim": "headline", "eligible": [{}], "rate": "measured_rate"},                             # Astra C5 r4 #2: no TypeError
             {"claim": "headline", "eligible": [["practice"]], "rate": "measured_rate"},
             {"claim": ["headline"], "eligible": ["no_energy_row"], "rate": "measured_rate"},
             {"claim": "headline", "eligible": ["no_energy_row"], "rate": {"x": 1}}]
def _no_raise_none(t):
    try:
        return A.approved_terms(cfg_terms(t)) is None and bool(A.unmeasured_terms_problems(t, EM0)) and A.price_row(
            PRAC10[0], "gemma", cfg_terms(t), prep=True, rate_w=100.0, practice_ids={PID})["known"] is False
    except Exception as e:                                               # noqa: BLE001
        return f"raised {type(e).__name__}"


_bt = [_no_raise_none(t) for t in bad_terms]
check("C5: malformed / conflicting terms are never approved and never raise (another rate, an extra wattage key, empty / unknown / repeated / "
      "non-string classes, unknown claim, practice without a valid rows binding, rows without practice, none)", all(x is True for x in _bt), _bt)
# Astra C5 r4 #1: the approval covers ONLY the bound practice rows, never other preparation rows or later practice rows
PREP8 = {"cost_usd": 0.0, "plan_secs": 8.0, "cost_kind": "gemma", "key": "prep|test-v1"}
ce_prep = block(rows_m, plan_meas, prep=PRAC10 + [dict(PREP8, cost_source="preparation")], cfg=cfg_ap)
ce_v99 = block(rows_m, plan_meas, prep=PRAC10 + [dict(PRAC10[0], key="practice-v99|1")], cfg=cfg_ap)
ce_dup = block(rows_m, plan_meas, prep=PRAC10 + PRAC10, cfg=cfg_ap)
check("C5 (r4 probe): a prep_cost row (8 s) stays UNKNOWN under practice-only terms; a later practice row not in row_ids stays UNKNOWN; an "
      "approved row duplicated is estimated neither time",
      ce_prep["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_v99["by_arm"]["G+M"]["cost_state"].startswith("unknown")
      and ce_dup["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_dup["by_arm"]["G+M"]["energy_estimated"]["allocated_rows"] == 0
      and A.unmeasured_class(PREP8, prep=True) == "preparation" and A.unmeasured_class(dict(PREP8, cost_source="practice"), prep=True) == "practice"
      and A.approved_practice_ids(PRAC10 + PRAC10, tp()) == set() and A.approved_practice_ids(PRAC10, tp()) == {PID})
# through analyse(): the source is set by the analysis, never read from the row; the 10 real practice-v2 rows under the real approved terms
cfg_real = cfg_terms(dict(_UA))
_sr, _ = A.synthetic_rows(20261010)
_pl = A.synthetic_plan_rows(_sr, energy=True)
_pl = [dict(r, planning_s=r["wall_s"], energy_state="measured") if A.arm_kind(r["arm"]) == "gemma" else r for r in _pl]
_prac_real, _ = A.strip_energy_fields(_pv2_rows)
ce_real = A.analyse(_sr, plan_rows=_pl, practice_rows=_prac_real, B=99, B_cost=200, meta={"namespace": "synthetic", "synthetic": True},
                    cost_cfg=cfg_real)["cost_effectiveness"]
_tot_s = sum(sum(ce_real["by_arm"][x]["energy_estimated"]["allocated_seconds"] for x in ce_real["by_arm"]) for _ in [0])
_rw = ce_real["energy_rate"]["rate_w"]
check("C5: through analyse() the 10 real practice-v2 rows (21.42 s) are estimated at the measured rate under the real approved terms, the +M "
      "arms costed", _rw is not None and close(_tot_s, 21.42)
      and close(sum(ce_real["by_arm"][x]["energy_estimated"]["allocated_joules"] for x in ce_real["by_arm"]), _rw * 21.42)
      and ce_real["by_arm"]["G+M"]["cost_state"] == "costed", (_rw, _tot_s, ce_real["by_arm"]["G+M"]["cost_state"]))
ce_forge = A.analyse(_sr, plan_rows=_pl, practice_rows=_prac_real, prep_rows=[dict(_prac_real[0], cost_source="practice", cost_kind="gemma",
                     key="forged")] + [dict(PREP8, cost_source="practice")], B=99, B_cost=200,
                     meta={"namespace": "synthetic", "synthetic": True}, cost_cfg=cfg_real)["cost_effectiveness"]
ce_late = A.analyse(_sr, plan_rows=_pl, practice_rows=_prac_real + [dict(_prac_real[0], key="practice-v99|x")], B=99, B_cost=200,
                    meta={"namespace": "synthetic", "synthetic": True}, cost_cfg=cfg_real)["cost_effectiveness"]
ce_omit = A.analyse(_sr, plan_rows=_pl, practice_rows=_prac_real[:-1], B=99, B_cost=200, meta={"namespace": "synthetic", "synthetic": True},
                    cost_cfg=cfg_real)["cost_effectiveness"]
check("C5 (r5 probe): an approved practice row missing from the consumed rows -> the +M arms UNKNOWN (never a silently smaller 19.50 s)",
      ce_omit["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_omit["by_arm"]["G+M"]["allocated"]["known"] is False
      and A.practice_binding_missing(_prac_real[:-1] and [dict(r, cost_source="practice") for r in _prac_real[:-1]], dict(_UA))
      == [A.practice_row_id(_prac_real[-1])], ce_omit["by_arm"]["G+M"]["cost_state"])
check("C5: the real bound journal passes the freeze-time binding check; a missing journal / zero sha / other rows fail",
      A.practice_journal_problems(ROOT, dict(_UA)) == []
      and A.practice_journal_problems(ROOT, dict(_UA, rows=dict(_UA["rows"], journal="docs/final-test/none.jsonl"))) != []
      and A.practice_journal_problems(ROOT, dict(_UA, rows=dict(_UA["rows"], journal_sha256="0" * 64))) != []
      and A.practice_journal_problems(ROOT, dict(_UA, rows=dict(_UA["rows"], row_ids=_UA["rows"]["row_ids"][:-1]))) != [])
# Astra C5 r6 #1: a practice row's own cost_kind label cannot move it out of Gemma energy / hardware accounting
_e_real = sum(ce_real["by_arm"][x]["energy_estimated"]["allocated_joules"] for x in ce_real["by_arm"])
_hw_real = sum(ce_real["by_arm"][x]["allocated"]["hardware"] for x in ce_real["by_arm"])
_relab = []
for _kind in ("api", "codex"):
    _ce = A.analyse(_sr, plan_rows=_pl, practice_rows=[dict(r, cost_kind=_kind) for r in _prac_real], B=99, B_cost=200,
                    meta={"namespace": "synthetic", "synthetic": True}, cost_cfg=cfg_real)["cost_effectiveness"]
    _relab.append((close(sum(_ce["by_arm"][x]["energy_estimated"]["allocated_joules"] for x in _ce["by_arm"]), _e_real)
                   and close(sum(_ce["by_arm"][x]["allocated"]["hardware"] for x in _ce["by_arm"]), _hw_real)
                   and _ce["by_arm"]["G+M"]["cost_state"] == "costed"))
check("C5 (r6 probe): practice rows labelled cost_kind 'api' / 'codex' keep the same estimated energy and hardware allocation (practice is "
      "always priced as Gemma)", _hw_real > 0 and _e_real > 0 and all(_relab), (_relab, _e_real, _hw_real))
# Astra C5 r6 #2 / #3: containment with either separator; non-object rows are problems, never exceptions
_simroot = os.path.join(ROOT, "sim")
_pp = []
for _j in ("..\\docs\\final-test\\matrix\\practice-v2\\practice.jsonl", "../docs/final-test/matrix/practice-v2/practice.jsonl",
           "docs\\final-test\\..\\final-test\\matrix\\practice-v2\\practice.jsonl", "C:/x.jsonl", "C:x.jsonl", "/etc/x.jsonl",
           "\\\\server\\share\\x.jsonl", "./docs/final-test/matrix/practice-v2/practice.jsonl"):
    _pp.append(A.practice_journal_problems(_simroot if _j.startswith("..") else ROOT, dict(_UA, rows=dict(_UA["rows"], journal=_j))) != [])
check("C5: the journal path must stay inside the root: '..' with '\\' or '/', drive / UNC / absolute paths and '.' parts are refused",
      all(_pp), _pp)
import tempfile as _tf2                                                  # noqa: E402
_tdir = _tf2.mkdtemp(prefix="c5pj_")
_bad_rows = []
for _ln in (b"null", b"[]", b"42", b"{not json"):
    with open(os.path.join(_tdir, "p.jsonl"), "wb") as _fh:
        _fh.write(open(_pv2, "rb").read() + _ln + b"\n")
    _t = dict(_UA, rows=dict(_UA["rows"], journal="p.jsonl", journal_sha256=hashlib.sha256(open(os.path.join(_tdir, "p.jsonl"), "rb").read()).hexdigest()))
    try:
        _w = A.practice_journal_problems(_tdir, _t)
        _bad_rows.append(any("row 11" in x for x in _w))
    except Exception as e:                                               # noqa: BLE001
        _bad_rows.append(f"raised {type(e).__name__}")
check("C5: journal rows null / [] / 42 / unparsable are reported as line problems, never raised", all(x is True for x in _bad_rows), _bad_rows)
check("C5: through analyse() a prep_cost row claiming cost_source 'practice' is re-labelled preparation and stays UNKNOWN (the +M arms "
      "UNKNOWN); a later practice row outside row_ids stays UNKNOWN", ce_forge["by_arm"]["G+M"]["cost_state"].startswith("unknown")
      and ce_late["by_arm"]["G+M"]["cost_state"].startswith("unknown"),
      (ce_forge["by_arm"]["G+M"]["cost_state"], ce_late["by_arm"]["G+M"]["cost_state"]))
check("C5: well-formed terms without an 'amendment:' basis are not approved",
      A.approved_terms(cfg_terms(dict(A.UNMEASURED_TERMS_APPROVED), basis=None)) is None
      and A.approved_terms(cfg_terms(dict(A.UNMEASURED_TERMS_APPROVED), basis="declined")) is None)
check("C5: a changed registered rate method is caught (terms.rate must equal unmeasured_estimate.rate)",
      A.unmeasured_terms_problems(A.UNMEASURED_TERMS_APPROVED, dict(EM0, unmeasured_estimate=dict(EM0["unmeasured_estimate"], rate="other"))) != [])
ce_gp0 = block(rows_m, plan_mm, prep=PRAC10, cfg={k: v for k, v in cfg0.items() if k != "energy_method"})
check("without an energy method (pre-C5 plan) the same practice row makes the +M arms UNKNOWN",
      ce_gp0["by_arm"]["G+M"]["cost_state"].startswith("unknown") and ce_gp0["by_arm"]["O+M"]["cost_state"].startswith("unknown"))
ce_gp1 = block(rows_m, plan_meas, prep=[{"cost_usd": 0.0, "cost_kind": "gemma"}], cfg=cfg_ap)
check("a Gemma practice row without energy AND without plan_secs stays UNKNOWN even under the approved terms (nothing to estimate)",
      ce_gp1["by_arm"]["G+M"]["cost_state"].startswith("unknown"))
check("C5: a Gemma PLAN row without energy is UNKNOWN; an incoming energy_state is ignored for pricing",
      A.price_row({"wall_s": 2.0}, "gemma", cfg0)["known"] is False
      and A.price_row({"planning_s": 2.0, "energy_state": "estimated"}, "gemma", cfg0, rate_w=100.0)["known"] is False)

# ---------------------------------------------------------------------------------------------------- 9. Codex, Pareto, determinism, freeze
rows_x, plan_x2 = world(lambda k, l: 1.0 if k % 3 else 0.0, lambda k, l: 1.0, g_cost=(1.0, 90.0, 10.0), extra_arms=("X0",))
ce_x2 = block(rows_x, plan_x2)
xr = ce_x2["by_arm"]["X0"]
check("X0 is shown, CLI planning unpriced (incomplete), no USD views; SR reported", xr["cost_state"].startswith("incomplete") and xr["views"] == {}
      and xr["sr_item_weighted"]["estimate"] == 1.0)
par = {p["arm"]: p for p in ce_x2["pareto"]}
check("Pareto rows for every arm; X0 has no USD and no efficiency flag", set(par) == {"G+M", "O0", "X0"} and par["X0"]["mean_usd_per_attempted_task"] is None
      and par["X0"]["pareto_efficient"] is None)
check("Pareto: cheaper G+M and higher-SR O0 are both efficient here", par["G+M"]["pareto_efficient"] is True and par["O0"]["pareto_efficient"] is True)
ce_d = block(rows_x, plan_x2)
check("deterministic: same inputs -> identical block", json.dumps(ce_d, sort_keys=True, default=str) == json.dumps(ce_x2, sort_keys=True, default=str))
svg = A.pareto_svg(ce_x2["pareto"], "t")
check("pareto.svg: well-formed root, placed arms drawn, uncosted arm listed below", svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
      and "G+M" in svg and "O0" in svg and "X0: SR" in svg and "not placed" in svg)
FZ = {"usd_per_kwh": 0.2, "hardware_price_usd": 399, "operating_hours": 6000, "hardware_time_factor": 3, "energy_method": "nvidia-smi board + CPU package",
      "unmeasured_rule": {"terms": A.UNMEASURED_TERMS_APPROVED}}   # C5-SIMPLE: the active practice-row rule must be in the freeze record
check("freeze: no a24_2_cost -> recorded False (flagged)", A.check_cost_freeze({})["recorded"] is False)
fz_nb = A.check_cost_freeze({"a24_2_cost": FZ})
check("freeze: matching numbers but NO hardware price basis -> not recorded (Astra Q4: 399 is a reference-device price)", fz_nb["recorded"] is False
      and any("hardware_price_basis" in m for m in fz_nb["missing"]), str(fz_nb))
check("freeze: basis laptop_purchase_price or amendment:<id> -> recorded True",
      A.check_cost_freeze({"a24_2_cost": dict(FZ, hardware_price_basis="laptop_purchase_price")})["recorded"] is True
      and A.check_cost_freeze({"a24_2_cost": dict(FZ, hardware_price_basis="amendment:A26")})["recorded"] is True)
check("freeze: no energy method -> not recorded", A.check_cost_freeze({"a24_2_cost": {k: v for k, v in dict(FZ, hardware_price_basis="laptop_purchase_price").items()
                                                                                     if k != "energy_method"}})["recorded"] is False)
check("the shipped plan carries the D13 basis amendment:A33 (prereg A33, Arash 2026-10-14) with its disclosure, no longer PENDING",
      cfg0["hardware_price_basis"] == "amendment:A33" and "PENDING" not in cfg0["hardware_price_status"]
      and "laptop" in cfg0["hardware_price_disclosure"] and "hw_time_3x" in cfg0["hardware_price_disclosure"] and cfg0["hardware_price_usd"] == 399.0)
check("freeze: a NULL energy method counts as missing (a key with None is not a recorded method)",
      A.check_cost_freeze({"a24_2_cost": dict(FZ, hardware_price_basis="amendment:A33", energy_method=None)})["recorded"] is False
      and A.check_cost_freeze({"a24_2_cost": dict(FZ, hardware_price_basis="amendment:A33")})["recorded"] is True)
check("the cost markdown prints the A33 disclosure", any("A33" in ln and "laptop" in ln for ln in A.render_cost_md({"assumptions": cfg0, "available": False, "reason": "x"})))
try:
    A.check_cost_freeze({"a24_2_cost": {"usd_per_kwh": 0.2, "hardware_price_usd": 1200, "operating_hours": 6000}})
    bad = False
except A.AnalysisError:
    bad = True
check("freeze: a different hardware price -> refused (no silent override)", bad)

# ---------------------------------------------------------------------------------------------------- 9b. Astra P-cost changes
ce_fr = block(rows_s, plan_s, freeze_required=True, freeze_recorded=False)
check("real data without a complete freeze record: statement NOT permitted (estimates still computed)", ce_fr["claim"]["statement_permitted"] is False
      and "freeze.json" in ce_fr["claim"]["reason"] and ce_fr["R_C"]["R"]["ci95"] is not None)
check("real data with the freeze record: statement permitted as before", block(rows_s, plan_s, freeze_required=True, freeze_recorded=True)["claim"]["statement_permitted"] is True)
ce_cm = block(rows_s, plan_s, drop=[("G+M", "c-03-1", "fa")])
check("G+M missing one cohort presentation: reported, R/C still estimated on the matched set, statement NOT permitted",
      ce_cm["by_arm"]["G+M"]["missing_vs_cohort"] == 1 and ce_cm["R_C"]["cohort_complete"] is False and ce_cm["R_C"]["matched_presentations"] == 47
      and ce_cm["claim"]["statement_permitted"] is False and "cohort" in ce_cm["claim"]["reason"])
ce_np = block(rows_m, plan_mm, prep=None)
check("no practice / preparation rows at all: consumer arms (+M) UNKNOWN, O0 still costed", ce_np["by_arm"]["G+M"]["cost_state"].startswith("unknown")
      and ce_np["by_arm"]["O+M"]["cost_state"].startswith("unknown") and ce_np["by_arm"]["O0"]["cost_state"] == "costed")
rows_5, plan_5 = world(lambda k, l: 1.0, lambda k, l: 1.0, g_cost=(1.0, 90.0, 10.0), extra_arms=("O+M", "X+M", "G+FS", "O+FS", "X0", "G0"))
plan_5 = [dict(p, cost_usd=0.01, wall_s=8.0) if p["arm"] in ("O+M", "O+FS") else
          dict(p, energy_j_gpu=90.0, energy_j_cpu=10.0, wall_s=1.0) if p["arm"] in ("G+FS", "G0") else p for p in plan_5]
ce_5 = block(rows_5, plan_5, prep=prac)
shares = {a: ce_5["by_arm"][a]["allocated"]["energy"] for a in ("G+M", "O+M", "X+M", "G+FS", "O+FS", "G0", "O0", "X0")}
check("practice consumers = +M and +FS arms: one fifth each; G0 / O0 / X0 carry none", all(close(shares[a], 0.20 / 5) for a in ("G+M", "O+M", "X+M", "G+FS", "O+FS"))
      and shares["G0"] == 0 and shares["O0"] == 0 and shares["X0"] == 0, str(shares))
check("allocated once: the shares add up to the practice row (no duplicated expenditure)", close(sum(shares.values()), 0.20))
x5 = ce_5["by_arm"]["X+M"]
check("X+M keeps and reports its known preparation USD; total incomplete (never zero), no cost coordinate", close(x5["known_preparation_usd"], (0.20 + 0.0665) / 5)
      and x5["cost_state"].startswith("incomplete") and x5["views"] == {} and ce_5["by_arm"]["X0"]["known_preparation_usd"] == 0.0)
par5 = {p["arm"]: p for p in ce_5["pareto"]}
check("Pareto table lists all 8 arms; X rows carry the known preparation USD", len(par5) == 8 and close(par5["X+M"]["known_preparation_usd"], (0.20 + 0.0665) / 5))
check("the Codex Pareto exception is documented in the plan", "exception" in cfg0["codex_pareto_exception"])

# ---------------------------------------------------------------------------------------------------- 10. end to end + the old blocks unchanged
rows_e, _ = A.synthetic_rows(seed=7, infra_rate=0.0)
plan_e = A.synthetic_plan_rows(rows_e)
res = A.analyse(rows_e, plan_rows=plan_e, practice_rows=A.synthetic_practice_rows(), B=499, B_cost=BB, meta={"namespace": "synthetic", "synthetic": True})
ce_e = res["cost_effectiveness"]
check("synthetic end to end: every non-X arm costed, X arms incomplete (CLI unpriced)", all(v["cost_state"] == "costed" for a, v in ce_e["by_arm"].items() if not a.startswith("X"))
      and all(v["cost_state"].startswith("incomplete") for a, v in ce_e["by_arm"].items() if a.startswith("X")))
res_real = A.analyse(rows_e, plan_rows=plan_e, practice_rows=A.synthetic_practice_rows(), B=99, B_cost=200, meta={"namespace": "rehearsal-v1"})
check("analyse on non-synthetic meta without a freeze record: the A24.2 statement is blocked", res_real["cost_effectiveness"]["claim"]["statement_permitted"] is False
      and "freeze.json" in res_real["cost_effectiveness"]["claim"]["reason"])
check("synthetic end to end: R and the three C views have finite CIs", all(math.isfinite(x) for x in ce_e["R_C"]["R"]["ci95"])
      and all(all(math.isfinite(x) for x in ce_e["R_C"]["C"][v]["ci95"]) for v in A.C_VIEWS))
md = A.render_md(res)
check("markdown carries the A24.2 section, the rule line and the latency table", "Cost-effectiveness (A24.2" in md and "A24.2 reading rule" in md
      and "planning s / presentation" in md and "API spend per success" in md)
import importlib.machinery                                            # noqa: E402
_pre_path = os.path.join(HERE, "final_test", "_checkpoints", "P-cost", "analyze.py.pre")
spec = importlib.util.spec_from_loader("analyze_pre", importlib.machinery.SourceFileLoader("analyze_pre", _pre_path))
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)
res_pre = pre.analyse(rows_e, plan_rows=plan_e, practice_rows=A.synthetic_practice_rows(), B=499, meta={"namespace": "synthetic", "synthetic": True})
same = {k: json.dumps(res[k], sort_keys=True, default=str) == json.dumps(res_pre[k], sort_keys=True, default=str) for k in res_pre if k not in ("analysis_plan", "analysis_plan_sha")}
check("every pre-change result block is byte-identical (only the new block and the plan hash differ)", all(same.values()),
      str([k for k, v in same.items() if not v]))
check("new keys only: cost_effectiveness", set(res) - set(res_pre) == {"cost_effectiveness"})
pr_old = A.synthetic_plan_rows(rows_e, energy=False)
check("synthetic plan rows keep the old cost_usd stream (old cost_block unchanged)", [r["cost_usd"] for r in pr_old] == [r["cost_usd"] for r in plan_e])
res_ne = A.analyse(rows_e, plan_rows=pr_old, practice_rows=A.synthetic_practice_rows(), B=99, B_cost=200, meta={"namespace": "synthetic", "synthetic": True})
check("without energy fields the Gemma arms are UNKNOWN and the reading rule cannot be satisfied (no silent 0)",
      all(res_ne["cost_effectiveness"]["by_arm"][a]["cost_state"].startswith("unknown") for a in ("G0", "G+M", "G+FS"))
      and res_ne["cost_effectiveness"]["claim"]["statement_permitted"] is False)

# Astra C5 r2 #3: an UNKNOWN energy cost never turns known planning seconds into 0 s
_rw, _pw = world(lambda k, l: 1, lambda k, l: 1, g_cost=(8.0, None, None))
_bu = block(_rw, _pw)["by_arm"]["G+M"]
_pm = [dict(r) for r in _pw]
_gi = [n for n, r in enumerate(_pm) if r["arm"] == "G+M"]
for n in _gi[: len(_gi) // 2]:
    _pm[n].update(energy_j_gpu=100.0, energy_j_cpu=50.0)                 # half measured, half unmeasured
_bm = block(_rw, _pm)["by_arm"]["G+M"]
check("unmeasured Gemma calls: cost UNKNOWN but latency = their 8 s (mean / median / max / total), not 0; mixed measured / unmeasured the same",
      _bu["cost_state"].startswith("unknown") and close(_bu["latency"]["mean_s_per_presentation"], 8.0) and close(_bu["latency"]["median_s"], 8.0)
      and close(_bu["latency"]["max_s"], 8.0) and close(_bu["latency"]["total_planning_hours"], 48 * 8.0 / 3600)
      and _bm["cost_state"].startswith("unknown") and close(_bm["latency"]["mean_s_per_presentation"], 8.0),
      (_bu.get("cost_state"), _bu.get("latency"), _bm.get("latency")))

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
