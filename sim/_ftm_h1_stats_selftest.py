"""Selftest for sim/final_test_h1_stats.py (stage H-H1-stats; h1-rule-1 = C7, exploratory E1). Offline, Windows python, numpy + scipy.
No outcome data.

    python sim/_ftm_h1_stats_selftest.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                          # noqa: E402

import final_test_h1 as H                                                   # noqa: E402
import final_test_h1_stats as T                                             # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("PASS", name)
    else:
        FAIL += 1
        print("FAIL", name, detail)


sizes = H.family_sizes()
rng = np.random.default_rng(7)


def data(cfg, R):
    en, fa = (H.gen_A if cfg["dgp"] == "A" else H.gen_B)(rng, R, sizes, cfg)
    en, fa = H.apply_missing(rng, en, fa, cfg)
    return H.summarise(en, fa, sizes)


CFGS = [{"dgp": "A", "d": .05, "D": -.05, "rho": .6, "twin": .7}, {"dgp": "A", "d": .02, "D": 0.0, "rho": .9, "twin": .7},
        {"dgp": "A", "d": .2, "D": -.05, "rho": .3, "twin": .7}, {"dgp": "A", "d": .4, "D": .05, "rho": 0.0, "twin": .4},
        {"dgp": "B", "p_O": .8, "s_e": .2, "s_g": .1, "D": -.05}, {"dgp": "A", "kind": "allornothing", "a": .05, "b": 0.0, "D": -.05, "twin": .7},
        {"dgp": "A", "d": 0.0, "D": 0.0, "rho": 0.0, "twin": .7}, {"dgp": "A", "d": .6, "D": -.05, "rho": 0.0, "twin": .7}]
MISS = {"dgp": "A", "d": .3, "D": -.05, "rho": .3, "twin": .7, "mcar": .25}

# ---- 1: batch engine == per-dataset deployable functions of final_test_h1
mx_bt = mx_cr = 0.0
undef_seen = 0
same_def = True
for cfg in CFGS:
    S, n, K, _ = data(cfg, 35)
    b, c = T.bt_batch(S, n), T.cr2_batch(S, n)
    for j in range(len(S)):
        r1, r2 = H.bootstrap_t(S[j], n[j]), H.cr2(S[j], n[j])
        same_def &= (r1["defined"] == bool(b["defined"][j])) and (r2["defined"] == bool(c["defined"][j]))
        if r1["defined"]:
            for x, y in ((r1["L"], b["L"][j]), (r1["U"], b["U"][j])):
                if np.isinf(x) or np.isinf(y):
                    same_def &= (x == y)
                else:
                    mx_bt = max(mx_bt, abs(x - y))
        else:
            undef_seen += 1
        if r2["defined"]:
            mx_cr = max(mx_cr, abs(r2["L"] - c["L"][j]), abs(r2["U"] - c["U"][j]))
check("bt_batch == final_test_h1.bootstrap_t on 280 datasets (8 DGPs incl. zero / low / high discordance): bounds within 1e-9, same defined flags",
      same_def and mx_bt < 1e-9, f"max diff {mx_bt}, same_def {same_def}")
check("the comparison covered undefined (se = 0) datasets", undef_seen > 0, f"{undef_seen}")
check("cr2_batch == final_test_h1.cr2 (bounds within 1e-12)", mx_cr < 1e-12, f"{mx_cr}")
S, n, K, _ = data(CFGS[2], 20)
check("cr2_single at .025 == final_test_h1.cr2", all(abs(T.cr2_single(S[j], n[j], .025)["L"] - H.cr2(S[j], n[j])["L"]) < 1e-12 for j in range(20)))

# ---- 2: rule_batch claims == final_test_h1.decide for the screened candidates (guard, dropped families)
mismatch, dropped = {}, 0
for cfg in CFGS + [MISS]:
    S, n, K, _ = data(cfg, 25)
    dropped += int((~(n > 0).all(1)).sum())
    for cand in ("C2", "C3", "C5", "C6"):
        rb = T.rule_batch(cand, S, n, K)
        for j in range(len(S)):
            keep = n[j] > 0
            ref = H.decide(cand, S[j][keep], n[j][keep], int(K[j]))
            if any(bool(rb[k][j]) != bool(ref[k]) for k in ("ni", "sup", "inf", "lower")):
                mismatch[cand] = mismatch.get(cand, 0) + 1
check("rule_batch claims == final_test_h1.decide for C2, C3, C5, C6 on 225 datasets (guard K >= 20; families with n_f = 0 dropped)", not mismatch, str(mismatch))
check("the missing-data case dropped families (per-dataset fallback ran)", dropped > 0, f"{dropped}")

# ---- 3: the deployable C7 rule (what analyze.py calls) == the batch engine V100 uses
mis, mxb, n_claim = 0, 0.0, 0
for cfg in CFGS + [MISS]:
    S, n, K, _ = data(cfg, 60)
    rb = T.rule_batch("C7", S, n, K)
    for j in range(len(S)):
        keep = n[j] > 0
        d = T.h1_rule_c7(S[j][keep], n[j][keep], int(K[j]))
        n_claim += d["claims"]["ni"] or d["claims"]["inf"]
        if any(bool(rb[k][j]) != bool(d["claims"][k]) for k in ("ni", "sup", "inf", "lower")) or bool(rb["defined"][j]) != d["defined"]:
            mis += 1
        if d["defined"]:
            mxb = max(mxb, abs(d["interval"][0] - rb["L"][j]), abs(d["interval"][1] - rb["U"][j]))
check("h1_rule_c7 (deployable) == rule_batch('C7') on 540 datasets: same claims, same defined flag, bounds within 1e-12", mis == 0 and mxb < 1e-12, f"{mis} {mxb}")
check("...and those datasets include real claims (not vacuous)", n_claim > 20, f"{n_claim}")
S, n, K, _ = data(CFGS[7], 30)
c6, c7 = T.rule_batch("C6", S, n, K), T.rule_batch("C7", S, n, K)
check("C7 interval (nominal 96.5 %) contains the C6 95 % interval; every C7 claim is also a C6 claim",
      bool(np.all(c7["L"] <= c6["L"] + 1e-15) and np.all(c7["U"] >= c6["U"] - 1e-15)) and all(np.all(~c7[k] | c6[k]) for k in ("ni", "sup", "inf", "lower")))
d = T.h1_rule_c7(S[0], n[0], 25)
check("C7 critical value is t_nu(.9825), nominal level .965, procedure h1-rule-1", abs(d["t_crit"] - float(T._tq(.0175, d["nu"]))) < 1e-12
      and d["interval_nominal_level"] == 0.965 and d["procedure"] == "h1-rule-1" and T.ALPHA_STAR == 0.0175)

# ---- 4: wording logic, guard, zero variance
S, n, K, _ = data({"dgp": "A", "d": .4, "D": .1, "rho": 0.0, "twin": .7}, 200)
cl = T.rule_batch("C7", S, n, K)
check("fixed sequence: sup => ni; lower => ni; never ni and materially inferior together",
      bool(np.all(~cl["sup"] | cl["ni"]) and np.all(~cl["lower"] | cl["ni"]) and not np.any(cl["ni"] & cl["inf"])))
check("a strong positive effect (Delta +10 pp, d .4) gives superiority claims", cl["sup"].mean() > .3, f"{cl['sup'].mean()}")
z = T.rule_batch("C7", np.zeros((5, 42)), np.tile(np.asarray(sizes, float), (5, 1)), np.zeros(5, int))
check("zero discordance: undefined, no claim", not z["defined"].any() and not any(z[k].any() for k in ("ni", "sup", "inf", "lower")))
zz = T.h1_rule_c7([0.0] * 42, sizes, 0)
check("deployable C7 on zero variance: inconclusive wording, interval None", zz["wording"].startswith("non-inferiority unresolved") and zz["interval"] == [None, None])
g = T.rule_batch("C7", S[:20], n[:20], np.full(20, 19))
check("guard K = 19 < 20: no claim, never inferior", not any(g[k].any() for k in ("ni", "sup", "inf", "lower")))
gg = T.h1_rule_c7(S[0], n[0], 19)
check("deployable guard failure wording names K and the threshold", "guard failed (K = 19 < 20" in gg["wording"] and not any(gg["claims"].values()))
check("wordings come from final_test_h1.wording (four registered phrases)", T.h1_rule_c7(S[0], n[0], 30)["wording"].split(" (")[0].split(";")[0] in
      ("superior", "non-inferior", "materially inferior", "non-inferiority unresolved"))

# ---- 5: exploratory E1
S, n, K, _ = data({"dgp": "A", "d": .4, "D": 0.0, "rho": 0.0, "twin": .7}, 40)
e = T.rule_batch("E1", S, n, K)
agree = all(T.equiv_e1(S[j], n[j], int(K[j]))["equivalent"] == bool(e["defined"][j] and e["guard_ok"][j] and e["L"][j] > -.05 and e["U"][j] < .05)
            for j in range(40))
check("equiv_e1 (deployable) == the batch TOST rule (90 % CR2 + guard)", agree)
check("E1 is the plain 90 % interval (alpha .05 per side), not the C7 calibration", T.RULES["E1"] == ("cr2", 0.05, True) and T.EQ_ALPHA == 0.05)

# ---- 6: grids, seeds, shas
v = T.v100_configs()
base = H.configs()
ch = [c for c in v if c.get("challenge")]
check("V100 = the frozen 146 configurations unchanged (ids, nulls, order) + a challenge set", [c["id"] for c in v[:146]] == [c["id"] for c in base]
      and all(v[i]["nulls"] == base[i]["nulls"] for i in range(146)) and len(ch) >= 60, f"challenge {len(ch)}")
check("V100 ids unique, indexes 0..N-1, R >= 100,000", len({c["id"] for c in v}) == len(v) and [c["index"] for c in v] == list(range(len(v))) and T.R_V100 >= 100_000)
check("challenge set covers guard transitions, high discordance both directions, size allocation, missingness, probit",
      all(any(c["id"].startswith(p) for c in ch) for p in ("X-tr-", "X-hi-D-0.1", "X-hi-D0.05", "X-size-", "X-mis0.06", "X-B-")))
feas = True
for c in v + T.power_configs() + T.equiv_configs():
    if c["dgp"] == "A" and c.get("kind") in (None, "A"):
        feas &= abs(c["D"]) <= c["d"] + 1e-12
    if c.get("kind") == "size":
        feas &= abs(c["D_L"]) <= c["d_L"] + 1e-12 and abs(c["D_S"]) <= c["d_S"] + 1e-12 and abs((c["D_L"] * 90 + c["D_S"] * 35) / 125 - c["D"]) < 1e-9
check("every configuration is feasible (|D| <= d) and size allocations preserve the item-weighted Delta", feas)
seeds = {}
clash = False
for st, (fn, _rk, sk, _r) in T.STAGES.items():
    for c in fn():
        s = getattr(T, sk) + c["index"]
        clash |= s in seeds
        seeds[s] = st
for c in base:
    for s in (H.SEED_SCREEN + c["index"], T.SEED_CALIB + c["index"]):
        clash |= s in seeds
        seeds[s] = "screen/calib"
check("random streams: every (stage, configuration) seed is distinct across screen, calibration, V100, power and equivalence", not clash and max(len(f()) for f, *_ in T.STAGES.values()) < 1000)
check("V100 seeds are the frozen Stage V seeds 20261013 * 1000 + index (never used: no validation journal of final_test_h1 exists)",
      T.SEED_V100 == 20261013 * 1000 and not (H.OUT / "validation.C6.jsonl").exists() and not (H.OUT / "validation.C3.jsonl").exists())
check("screen protocol sha unchanged (414982f540f7e4e4)", H.protocol_sha()[:16] == "414982f540f7e4e4")
check("stage shas stable and distinct", T.stage_sha("v100") == T.stage_sha("v100") and len({T.stage_sha(s) for s in T.STAGES}) == 3)
check("equivalence grid has error cells at and outside both boundaries", all(any(abs(c["D"] - x) < 1e-9 for c in T.equiv_configs()) for x in (.05, -.05, .075, -.10)))

# ---- 7: the runner counts == a slow loop over the deployable rule on the SAME datasets (same seed, R <= 2,000)
saved = T.R_V100
try:
    T.R_V100 = 80
    for cid in ("N-A-d0.6-r0.0", "N-Amcar-d0.1", "X-tr-D-0.05-d0.2-r0.0".replace("d0.2", "d0.225")):
        cfg = next(c for c in T.v100_configs() if c["id"] == cid)
        got = T.run_one((cfg, "v100"))["counts"]
        g2 = np.random.default_rng(T.SEED_V100 + cfg["index"])
        en, fa = (H.gen_A if cfg["dgp"] == "A" else H.gen_B)(g2, 80, sizes, cfg)
        en, fa = H.apply_missing(g2, en, fa, cfg)
        S, n, K, _ = H.summarise(en, fa, sizes)
        ref = {k: 0 for k in ("ni", "sup", "inf", "lower", "neg", "guard_pass", "undefined")}
        for j in range(80):
            keep = n[j] > 0
            dd = T.h1_rule_c7(S[j][keep], n[j][keep], int(K[j]))
            for k in ("ni", "sup", "inf", "lower"):
                ref[k] += dd["claims"][k]
            ref["neg"] += dd["claims"]["inf"] or dd["claims"]["lower"]
            ref["guard_pass"] += dd["guard"]["passed"]
            ref["undefined"] += not dd["defined"]
        check(f"run_one counts == slow loop over h1_rule_c7 ({cid}, 80 datasets)", all(got[k] == ref[k] for k in ref), f"{got} vs {ref}")
finally:
    T.R_V100 = saved

# ---- 8: exploratory calibration at alpha* .025 reproduces C6 (same seed)
saved = (T.SEED_CALIB, T.R_CALIB)
try:
    T.SEED_CALIB, T.R_CALIB = H.SEED_SCREEN, 80
    for cfg in (next(c for c in base if c["id"] == "N-A-d0.6-r0.0"), next(c for c in base if c.get("mcar"))):
        a = T.run_calib((cfg,))["counts"]["0.025"]
        bref = H.run_config((cfg, "screen", ["C6"], 80))["counts"]["C6"]
        check(f"calibration at alpha* .025 == C6 counts ({cfg['id']})", all(a[k] == bref[k] for k in ("ni", "sup", "inf", "lower", "neg")), f"{a} vs {bref}")
finally:
    T.SEED_CALIB, T.R_CALIB = saved

# ---- 9: Clopper-Pearson and the acceptance edge
lo, hi = T.cp_interval(0, 100_000)
check("CP interval at 0 / 100,000: [0, ~3.7e-5]", lo == 0.0 and 3.6e-5 < hi < 3.8e-5, f"{hi}")
check("one-sided CP upper at 2,500 / 100,000 is about .0258 (<= .0275); 2,700 / 100,000 is above", abs(H.cp_upper(2500, 100_000) - .02583) < .0002
      and H.cp_upper(2700, 100_000) > .0275)

print(f"\n{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
