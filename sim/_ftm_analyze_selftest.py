"""Selftest for sim/final_test/matrix/analyze.py (stage M9-analyze). Offline: no simulator, no network, no API, no test item.

    python sim/_ftm_analyze_selftest.py
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "final_test", "matrix")
sys.path.insert(0, HERE)
sys.path.insert(0, MATRIX)

import common as C                      # noqa: E402
import analyze as A                     # noqa: E402
import final_test_power as P            # noqa: E402

N = {"pass": 0, "fail": 0}
B = 999                                  # small B keeps the selftest fast; the production default is 9999


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def mk(arm, item, lang, sr, fam, lvl="L1", infra=False, valid=True, r=0, run=0, unit=None, ns="rehearsal-v1", **kw):
    return {"key": f"{arm}|{item}|{lang}|{r}|{run}", "namespace": ns, "arm": arm, "pres": C.pres_id(item, lang), "r": r, "run": run,
            "unit": unit or f"u|{arm}|{item}|{lang}", "sr": float(sr), "strict_sr": float(sr), "valid": valid, "infra": infra,
            "level": lvl, "lang": lang, "family": fam, **kw}


def grid(arms_sr, n_fam=10, per_fam=3, **kw):
    """arms_sr: arm -> function(item_index) -> sr. Deterministic rows over n_fam families."""
    rows, k = [], 0
    for f in range(n_fam):
        for j in range(per_fam):
            item = f"t-{f:02d}-{j}"
            for arm, fn in arms_sr.items():
                for lang in A.LANGS:
                    rows.append(mk(arm, item, lang, fn(k, lang), f"fam{f:02d}", lvl=("L1", "L2", "L3")[f % 3], **kw))
            k += 1
    return rows


# ---------------------------------------------------------------------------------------------------- 1. holm
h = A.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
check("holm: step-down values", abs(h["a"]["p_holm"] - 0.004) < 1e-12 and abs(h["b"]["p_holm"] - 0.06) < 1e-12
      and abs(h["c"]["p_holm"] - 0.08) < 1e-12 and h["d"]["p_holm"] == 0.5, str(h))
check("holm: only the first rejects at .05", [k for k, v in h.items() if v["reject"]] == ["a"])
h2 = A.holm({"a": 0.30, "b": 0.001})
check("holm: adjusted p is monotone in rank order", h2["b"]["p_holm"] <= h2["a"]["p_holm"] and abs(h2["b"]["p_holm"] - 0.002) < 1e-12)
h3 = A.holm({"a": None, "b": 0.001, "c": 0.002})
check("holm: unavailable test counts as p=1 and keeps m", h3["a"]["p_holm"] == 1.0 and abs(h3["b"]["p_holm"] - 0.003) < 1e-12, str(h3))

# ---------------------------------------------------------------------------------------------------- 2. synthetic end to end
rows, truth = A.synthetic_rows(seed=7, infra_rate=0.0)
res = A.analyse(rows, plan_rows=A.synthetic_plan_rows(rows), practice_rows=[{"cost_usd": 0.0}], B=B, meta={"namespace": "synthetic", "synthetic": True})
check("synthetic: 42 families, 8 main arms (A23: 4 core + 2 FS + the A18 X0 / X+M; no Sonnet / Haiku)", res["n_families"] == 42 and len(res["arms"]) == 8
      and not (set(res["arms"]) & set(C.ARCHIVED_ARMS)))
check("synthetic: recovers known memory gain H2a (+.18 true)", abs(res["holm_family"]["H2a"]["estimate"] - 0.18) < 0.07, str(res["holm_family"]["H2a"]["estimate"]))
check("synthetic: recovers the EXPLORATORY H5a (+.09 true) and H5c (G+M - X0 = +.01 true)", abs(res["exploratory_contrasts"]["H5a"]["estimate"] - 0.09) < 0.08
      and abs(res["exploratory_contrasts"]["H5c"]["estimate"] - 0.01) < 0.08)
check("synthetic: recovers the EXPLORATORY H4a (+.08 true)", abs(res["exploratory_contrasts"]["H4a"]["estimate"] - 0.08) < 0.07)
check("A24.5: the Holm family has exactly H2a, H2c, H3a (m = 3); H4a-b / H5a-c are exploratory: raw p, no p_holm, no reject, flagged not in Holm",
      set(res["holm_family"]) == {"H2a", "H2c", "H3a"} and set(res["exploratory_contrasts"]) == {"H4a", "H4b", "H5a", "H5b", "H5c"}
      and all(v["exploratory"] and not v["in_holm"] and "reject" not in v and "p_holm" not in v and 0 <= v["p"] <= 1 for v in res["exploratory_contrasts"].values())
      and set(res["withdrawn_hypotheses"]) == {"H2b", "H3b", "H3c"})
check("A24.5: the m = 8 transparency adjustment of H2a / H2c / H3a is never smaller than the m = 3 adjustment (the smaller m relaxes Holm)",
      set(res["holm_m8_for_transparency"]) == {"H2a", "H2c", "H3a"}
      and all(res["holm_m8_for_transparency"][h] >= res["holm_family"][h]["p_holm"] - 1e-12 for h in res["holm_family"]))
check("synthetic: recovers H1 estimate (-.02 true)", abs(res["H1"]["primary"]["estimate"] - (-0.02)) < 0.06)
check("synthetic: true big effects reject in Holm", res["holm_family"]["H2a"]["reject"])
check("synthetic: a true null is not rejected (H3b +.07 is small; H1 never says superior)",
      "superior" not in res["H1"]["wording"].replace("non-inferior", ""))
check("synthetic: H1 wording is one of the four registered phrases",
      any(res["H1"]["wording"].startswith(w) for w in ("superior", "non-inferior", "materially inferior", "non-inferiority unresolved")))
check("synthetic: H1 views carry the A24.1 h1-rule-1 decision (interval, guard, claims, wording)",
      all(k in res["H1"]["primary"]["decision"] for k in ("procedure", "interval", "guard", "claims", "wording"))
      and res["H1"]["primary"]["decision"]["procedure"] == "h1-rule-1")
check("synthetic: family-weighted and infra-paired views exist", res["H1"]["family_weighted"] and res["H1"]["infra_paired"])
check("synthetic: per-level and per-language tables", all(t["level"]["L2"] and t["lang"]["en"] and t["lang"]["fa"] for t in res["arm_tables"].values()))
check("synthetic: L2 PC and L3 TDS diagnostics", res["arm_tables"]["G+M"]["diagnostics"]["L2_pc_final"] and res["arm_tables"]["G+M"]["diagnostics"]["L3_tds_final"])
check("synthetic: variance decomposition has both parts", res["variance_decomposition"]["G+M"]["planner"]["n"] > 0
      and res["variance_decomposition"]["G+M"]["execution"]["n"] > 0)
check("synthetic: cost per success for a paid arm > 0 and free arm 0", res["cost_per_success"]["O0"]["usd_per_success"] > 0
      and res["cost_per_success"]["G0"]["usd_per_success"] == 0)
res2 = A.analyse(rows, plan_rows=A.synthetic_plan_rows(rows), practice_rows=[{"cost_usd": 0.0}], B=B, meta={"namespace": "synthetic", "synthetic": True})
check("deterministic: same input -> identical analysis", json.dumps(res, sort_keys=True) == json.dumps(res2, sort_keys=True))

# ---------------------------------------------------------------------------------------------------- 3. estimand against a hand calculation
# 2 families: f0 has 1 item (both langs: a=1,1 b=0,0 -> diff 1.0), f1 has 3 items (diff 0 each, except one 0.5).
R = []
def add(item, fam, sa_en, sa_fa, sb_en, sb_fa):
    for lang, sa, sb in (("en", sa_en, sb_en), ("fa", sa_fa, sb_fa)):
        R.append(mk("G+M", item, lang, sa, fam)); R.append(mk("O0", item, lang, sb, fam))
add("x-0", "f0", 1, 1, 0, 0)
add("x-1", "f1", 1, 1, 1, 1); add("x-2", "f1", 1, 0, 1, 0); add("x-3", "f1", 1, 1, 1, 0)
idx = A.build_index(R)
d, info = A.item_diffs(idx["S"], "G+M", "O0", set())
check("item diff = mean of EN and FA differences", d == {"x-0": 1.0, "x-1": 0.0, "x-2": 0.0, "x-3": 0.5}, str(d))
fams, tot, sizes = A.by_family(d, idx["fam"])
check("family totals and sizes", fams == ["f0", "f1"] and tot == [1.0, 0.5] and sizes == [1, 3], str((tot, sizes)))
check("estimate = sum T / sum n (item-weighted)", abs(P.estimate(tot, sizes) - 1.5 / 4) < 1e-12)
fw, totw, sizw = A.by_family(d, idx["fam"], weighted=True)
check("family-weighted: each family counts once", abs(P.estimate(totw, sizw) - (1.0 + 0.5 / 3) / 2) < 1e-12, str(totw))
check("win/tie/loss", A.win_tie_loss(fams, tot, sizes) == {"families_a_better": 2, "tie": 0, "families_b_better": 0})
check("fewer than 2 families is refused", raises(lambda: A.by_family({"x-0": 1.0}, {"x-0": "f0"}), A.AnalysisError))

# ---------------------------------------------------------------------------------------------------- 4. infra + invalid handling
R2 = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0 if k % 2 else 1}, n_fam=6, per_fam=2)
for r in R2:                                         # the first O0 presentation of item t-00-0 EN is an infra failure scored 0
    if r["arm"] == "O0" and r["pres"] == C.pres_id("t-00-0", "en"):
        r["infra"], r["sr"] = True, 0.0
i2 = A.build_index(R2)
dp, ip = A.item_diffs(i2["S"], "G+M", "O0", set())
dd, idr = A.item_diffs(i2["S"], "G+M", "O0", set(), drop_infra=True)
check("infra-paired drops the presentation where either arm had a failure", idr["dropped_infra"] == 1 and idr["n_pres"] == ip["n_pres"] - 1, str((ip, idr)))
check("infra-paired item diff uses the other language only", abs(dd["t-00-0"] - 0.0) < 1e-12 and abs(dp["t-00-0"] - 0.5) < 1e-12, str((dd["t-00-0"], dp["t-00-0"])))
R3 = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0}, n_fam=4, per_fam=2)
for r in R3:
    if r["pres"] == C.pres_id("t-01-0", "fa") and r["arm"] == "O0":
        r["valid"] = False
i3 = A.build_index(R3)
inv = A.invalid_presentations(i3["S"])
dv, iv = A.item_diffs(i3["S"], "G+M", "O0", inv)
check("INVALID presentation is dropped from every arm (pairing kept)", inv == {("t-01-0", "fa")} and iv["n_pres"] == 15, str((inv, iv)))
r3 = A.analyse(R3, B=B)
check("invalid count reported", r3["invalid_presentations_dropped"] == 1)
check("infra table counts per arm", A.infra_table(i2["S"], ["O0", "G+M"])["O0"]["infra"] == 1)

# ---------------------------------------------------------------------------------------------------- 5. H1 wording on constructed data
# A24.1: h1-rule-1 needs K >= 20 families with a discordant presentation and between-family variation, so these grids use 24 families and a k % 4 pattern
Rs = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0 if k % 4 else 1}, n_fam=24, per_fam=3)
check("H1: large gain reads superior", A.analyse(Rs, B=B)["H1"]["wording"].startswith("superior"))
Rg = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0 if k % 4 else 1}, n_fam=12, per_fam=3)
check("H1 (A24.1): the same large gain with only 12 discordant families is inconclusive (guard K >= 20)",
      "guard failed (K = 12" in A.analyse(Rg, B=B)["H1"]["wording"])
Rn = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 1}, n_fam=12, per_fam=3)
rn = A.analyse(Rn, B=B)["H1"]
check("H1: identical arms with zero variance never read 'superior'", not rn["wording"].startswith("superior"), rn["wording"])
Ri = grid({"G+M": lambda k, l: 0 if k % 4 else 1, "O0": lambda k, l: 1}, n_fam=24, per_fam=3)
check("H1: large loss reads materially inferior", A.analyse(Ri, B=B)["H1"]["wording"].startswith("materially inferior"))

# ---------------------------------------------------------------------------------------------------- 6. missing arms
res_m = A.analyse(grid({"G+M": lambda k, l: 1, "G0": lambda k, l: 0, "O0": lambda k, l: 1}, n_fam=8, per_fam=2), B=B)
check("A24.5 missing FS arms: H4a is exploratory and unavailable, the Holm family still has m = 3 (H2a / H2c / H3a unavailable members count as p = 1)",
      not res_m["exploratory_contrasts"]["H4a"]["available"] and "p_holm" not in res_m["exploratory_contrasts"]["H4a"]
      and len(res_m["holm_family"]) == 3 and res_m["holm_family"]["H2c"]["p_holm"] == 1.0 and not res_m["holm_family"]["H2c"]["available"])
check("missing core arms are listed (A23: the four core arms)", "O+M" in res_m["core_arms_missing"] and not any(a[0] in "SH" for a in res_m["core_arms_missing"]))
check("no scored cells is refused", raises(lambda: A.analyse([mk("G0", "x", "en", 1, "f", r=1)], B=B), A.AnalysisError))

# ---------------------------------------------------------------------------------------------------- 7. Codex kept separate
Rx = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0, "X-gpt-6-astra0": lambda k, l: 0, "X-gpt-6-astra+M": lambda k, l: 1}, n_fam=8, per_fam=2)
rx = A.analyse(Rx, B=B)
check("codex arms never enter the Holm family or the arm tables", all("X-" not in str(v.get("a")) + str(v.get("b")) for v in rx["holm_family"].values())
      and not any(a.startswith("X-") for a in rx["arm_tables"]))
check("codex table has rates and a memory gain", set(rx["codex"]["arms"]) == {"X-gpt-6-astra0", "X-gpt-6-astra+M"}
      and abs(rx["codex"]["memory_gain"]["X-gpt-6-astra+M"]["estimate"] - 1.0) < 1e-12)
check("codex arms are not in H1 views", rx["H1"]["primary"]["a"] == "G+M" and rx["H1"]["primary"]["b"] == "O0")

# ---------------------------------------------------------------------------------------------------- 8. variance decomposition on known data
V = []
for item, ids, srs in (("v-0", ("a", "a", "a"), (1, 1, 1)), ("v-1", ("a", "b", "c"), (1, 0, 0))):
    for rr in range(3):
        for run in range(3):
            sr = srs[rr] if (rr == 0 and item == "v-1") or True else 0
            V.append(mk("G+M", item, "en", srs[rr], "fv", r=rr, run=run, unit=f"{item}|{ids[rr]}"))
# execution disagreement on v-0 r=2 only
for x in V:
    if x["pres"] == C.pres_id("v-0", "en") and x["r"] == 2 and x["run"] == 2:
        x["sr"] = 0.0
vd = A.variance_decomposition(V, ["G+M"])["G+M"]
check("plan identity: one of two presentations repeats its plan", abs(vd["planner"]["plan_identity_rate"] - 0.5) < 1e-12, str(vd))
check("planner SR spread = mean of (0, 1)", abs(vd["planner"]["sr_spread_mean"] - 0.5) < 1e-12)
check("execution agreement: 5 of 6 (pres, r) cells agree", abs(vd["execution"]["outcome_agreement_rate"] - 5 / 6) < 1e-12, str(vd["execution"]))

# ---------------------------------------------------------------------------------------------------- 9. cost
Sx = {("O0", "i", "en"): {"sr": 1.0}, ("O0", "i", "fa"): {"sr": 1.0}, ("O+M", "i", "en"): {"sr": 1.0}, ("O+M", "i", "fa"): {"sr": 0.0}}
pr = [{"arm": "O0", "cost_usd": 0.10}, {"arm": "O0", "cost_usd": 0.20}, {"arm": "O+M", "cost_usd": 0.30}]
cb = A.cost_block(["O0", "O+M"], Sx, pr, [{"cost_usd": 0.05}, {"cost_usd": 0.05}])
check("cost per success: plan spend / successes", abs(cb["O0"]["usd_per_success"] - 0.15) < 1e-12)
check("cost per success: practice cost added to +M arms only", abs(cb["O+M"]["usd_per_success"] - 0.40) < 1e-12 and cb["O0"]["practice_usd_added"] == 0.0, str(cb))
check("cost per success unknown without a plan journal", A.cost_block(["O0"], Sx, None, None)["O0"]["usd_per_success"] is None)
Sz = {k: dict(v, sr=0.0) for k, v in Sx.items()}
cz = A.cost_block(["O0"], Sz, pr, None)["O0"]
check("cost per success: cost with zero successes is INFINITE (A24.2), 0 / 0 undefined, no journal unknown",
      cz["usd_per_success_state"] == "infinite" and A.cost_block(["O0"], Sz, [], None)["O0"]["usd_per_success_state"] == "undefined"
      and A.cost_block(["O0"], Sx, None, None)["O0"]["usd_per_success_state"] == "unknown", str(cz))

# ---------------------------------------------------------------------------------------------------- 10. seal + guard + outputs (temp dir)
tmp = tempfile.mkdtemp(prefix="ftm_analyze_")
man = C.load_manifest()
dev = list(man["dev_pilot"])
fam_of = {i: f["family"] for f in man["families"] for i in f["items"]}


def write_score(d, rows, complete=True, seal=True):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "score.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    if seal:
        with open(os.path.join(d, "score.seal.json"), "w", encoding="utf-8") as f:
            json.dump({"sha256": C.sha256_file(p), "n": len(rows), "complete": complete, "created": "x"}, f)
    return p


def dev_rows(ids, ns="rehearsal-v1"):
    out = []
    for n, item in enumerate(ids):
        for arm in ("G+M", "O0", "G0", "O+M"):
            for lang in A.LANGS:
                sr = 1.0 if ((n + (arm == "G+M") + (lang == "en")) % 3) else 0.0
                out.append({**mk(arm, item, lang, sr, fam_of.get(item, "fam-x"), ns=ns)})
    return out


try:
    d_ok = os.path.join(tmp, "rehearsal-v1")
    p_ok = write_score(d_ok, dev_rows(dev))
    poison = os.path.join(d_ok, "outcome.sealed.jsonl")
    os.makedirs(poison)                                          # opening it as a file raises: proves analyze never touches it
    os.makedirs(os.path.join(d_ok, "states"))
    resd, (jp, mp) = A.run_sealed(p_ok, B=B)
    check("sealed dev run succeeds and writes analysis.json + analysis.md beside the journal", os.path.exists(jp) and os.path.exists(mp)
          and os.path.dirname(jp) == d_ok)
    check("analysis.json is valid and not marked synthetic", json.load(open(jp, encoding="utf-8"))["meta"]["synthetic"] is False)
    check("seal sha is recorded in the analysis", resd["meta"]["seal_sha256"] == C.sha256_file(p_ok))
    check("markdown names the H1 wording", "H1 (primary)" in open(mp, encoding="utf-8").read())

    # tamper
    with open(p_ok, "a", encoding="utf-8") as f:
        f.write("\n")
    check("edited score.jsonl is refused (seal sha)", raises(lambda: A.verify_seal(p_ok), A.AnalysisError))
    p_ns = write_score(os.path.join(tmp, "noseal"), dev_rows(dev), seal=False)
    check("missing seal is refused", raises(lambda: A.verify_seal(p_ns), A.AnalysisError))
    p_inc = write_score(os.path.join(tmp, "inc"), dev_rows(dev), complete=False)
    check("incomplete journal refused without --partial, allowed with it", raises(lambda: A.verify_seal(p_inc), A.AnalysisError)
          and A.verify_seal(p_inc, True)["complete"] is False)
    p_n = write_score(os.path.join(tmp, "nmis"), dev_rows(dev))
    s = json.load(open(os.path.join(tmp, "nmis", "score.seal.json")))
    s["n"] += 1
    json.dump(s, open(os.path.join(tmp, "nmis", "score.seal.json"), "w"))
    check("seal row count mismatch is refused", raises(lambda: A.verify_seal(p_n), A.AnalysisError))

    # guard: a test item
    t_items = list(man["test"][:6])
    fam_of.update({i: f["family"] for f in man["families"] for i in f["items"]})
    p_t1 = write_score(os.path.join(tmp, "rehearsal-t"), dev_rows(t_items, "rehearsal-v1"))
    check("test items in rehearsal-v1 are refused", raises(lambda: A.run_sealed(p_t1, allow_test=True, B=B), C.TestItemRefused))
    p_t2 = write_score(os.path.join(tmp, "test-v1"), dev_rows(t_items, "test-v1"))
    check("test items in test-v1 without --allow-test are refused", raises(lambda: A.run_sealed(p_t2, B=B), C.TestItemRefused))
    fs = C.freeze_status()
    if not fs["ok"]:
        check("test items with --allow-test but no valid freeze are refused", raises(lambda: A.run_sealed(p_t2, allow_test=True, B=B), C.TestItemRefused))
    else:
        check("freeze valid in this checkout: guard path covered by common's selftest", True)
    p_mix = write_score(os.path.join(tmp, "mixns"), dev_rows(dev[:3], "rehearsal-v1") + dev_rows(dev[3:6], "practice-v1"))
    check("two namespaces in one score journal are refused", raises(lambda: A.run_sealed(p_mix, B=B), A.AnalysisError))
    p_out = write_score(os.path.join(tmp, "outside"), dev_rows(["not-an-item-id"]))
    check("an item outside the namespace set is refused", raises(lambda: A.run_sealed(p_out, B=B), C.TestItemRefused))

    # CLI exit codes
    check("CLI returns 2 on refusal", A.main(["--sealed", p_t2, "--B", "99"]) == 2)
    syn_dir = os.path.join(tmp, "syn")
    check("CLI --synthetic returns 0 and writes json + md", A.main(["--synthetic", "--B", "199", "--outdir", syn_dir]) == 0
          and os.path.exists(os.path.join(syn_dir, "analysis.json")) and os.path.exists(os.path.join(syn_dir, "analysis.md")))
    sj = json.load(open(os.path.join(syn_dir, "analysis.json"), encoding="utf-8"))
    check("synthetic output is labelled synthetic with the known truth", sj["meta"]["synthetic"] is True and "true_arm_success_probabilities" in sj["meta"])
    check("default synthetic folder differs from every namespace folder", os.path.basename(os.path.join(C.paths()["matrix_docs"], "synthetic")) not in C.NAMESPACES)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------------------------------- 10b. A21: blocked runs, dependence, Holm, H5c
import random as _random

check("A21 F: the analysis plan is frozen with a stable hash and lists exactly the 3 Holm members (A24.5) and the 5 exploratory contrasts; H1 is NOT in the family",
      A.analysis_plan_sha() == A.analysis_plan_sha() and set(A.ANALYSIS_PLAN["holm"]["members"]) == set(A.FAMILY) and A.ANALYSIS_PLAN["holm"]["m"] == 3 and set(A.ANALYSIS_PLAN["holm"]["exploratory"]) == set(A.EXPLORATORY)
      and "H1" not in A.FAMILY and "NOT a member" in A.ANALYSIS_PLAN["h1"]["family"] and A.ANALYSIS_PLAN["ni_guard"]["observed_discordance_below"] == A.LOW_DISCORDANCE)


def indep_rows(arms, n_fam=42, per_fam=3, p=0.55, seed=3, **kw):
    """Independent Bernoulli(p) outcomes per arm and presentation: true D = 0 between arms, discordance about 2p(1-p)."""
    rg = _random.Random(seed)
    rows = []
    for f in range(n_fam):
        for j in range(per_fam):
            item = f"t-{f:02d}-{j}"
            for arm in arms:
                for lang in A.LANGS:
                    rows.append(mk(arm, item, lang, 1.0 if rg.random() < p else 0.0, f"fam{f:02d}", lvl=("L1", "L2", "L3")[f % 3], **kw))
    return rows


rows_ni = indep_rows(["G+M", "X0"])
idx_ni = A.build_index(rows_ni)
det, _m8 = A.exploratory_contrasts(idx_ni["S"], idx_ni["fam"], set(), 999)          # A24.5: H5c is exploratory now
h5c = det["H5c"]
c_ni = A.contrast(idx_ni["S"], idx_ni["fam"], "G+M", "X0", set(), 0)
p_boundary = P.wcr_p(c_ni["_tot"], c_ni["_sizes"], -P.MARGIN, "greater", 999)
p_zero = P.wcr_p(c_ni["_tot"], c_ni["_sizes"], 0.0, "greater", 999)
check("A21 F / A24.5: H5c (exploratory) reports the one-sided wild test with the null BOUNDARY (-5 pp) imposed, raw; its m = 8 value is doubled and capped at 1 -- never a test of D = 0",
      h5c["null_boundary_imposed"] == -P.MARGIN and abs(h5c["p_one_sided"] - p_boundary) < 1e-12 and abs(h5c["p"] - p_boundary) < 1e-12
      and abs(h5c["p_m8"] - min(1.0, 2 * p_boundary)) < 1e-12
      and p_boundary < p_zero and h5c["discordance"] >= A.LOW_DISCORDANCE and "ni_guard" not in h5c, f"{p_boundary} {p_zero} {h5c.get('p')}")
check("A21 F: with true D = 0 and high discordance H5c rejects its H0 (D <= -5 pp) only if the boundary p is small (here the test has a real boundary p, not 0.5)",
      p_boundary < 0.5 and h5c["p_one_sided"] != p_zero)
sim = [mk(a, f"t-{f:02d}-{j}", l, 1.0 if (f * 3 + j) % 2 else 0.0, f"fam{f:02d}") for f in range(42) for j in range(3) for a in ("G+M", "X0") for l in A.LANGS]
for r_ in sim[:6]:
    if r_["arm"] == "X0":
        r_["sr"] = 1.0 - r_["sr"]
idx_lo = A.build_index(sim)
det_lo, _m8lo = A.exploratory_contrasts(idx_lo["S"], idx_lo["fam"], set(), 999)
check("A21 F / A24.5: below .25 observed discordance the m = 8 comparison does not reject H5c (p_m8 := 1) and the guard is recorded; the exploratory raw p stays descriptive",
      det_lo["H5c"]["ni_guard"]["applied"] and det_lo["H5c"]["p_m8"] == 1.0 and det_lo["H5c"]["ni_guard"]["p_before_guard"] < 1.0 and "reject" not in det_lo["H5c"]
      and det_lo["H5c"]["p"] < 1.0 and "H3c" not in det_lo)
check("A21 F: the guard does not touch the other 'greater' / two-sided members", "ni_guard" not in det_lo.get("H5a", {}) and det_lo["H5a"]["available"] is False)

# ---- holm m = 8 transparency: adding H5a-c can only make the registered eight's adjusted p larger or equal
syn_rows, _truth = A.synthetic_rows(seed=7)
idx_s = A.build_index(syn_rows)
det_s = A.holm_family(idx_s["S"], idx_s["fam"], A.invalid_presentations(idx_s["S"]), 499)
_ex_s, m8 = A.exploratory_contrasts(idx_s["S"], idx_s["fam"], A.invalid_presentations(idx_s["S"]), 499, det_s)
check("A24.5: the m = 8 adjustment (reported for transparency) is never smaller than the registered m = 3 adjustment",
      set(m8) == set(A.FAMILY) and all(m8[h] >= det_s[h]["p_holm"] - 1e-12 for h in A.FAMILY), str(m8)[:300])

# ---- dependence: shared executions are copies
rows_d = []
rg_ = _random.Random(11)
for f in range(12):
    for j in range(3):
        item = f"t-{f:02d}-{j}"
        for lang in A.LANGS:
            sh = rg_.random() < 0.4
            sr = 1.0 if rg_.random() < 0.5 else 0.0
            for arm in ("G0", "O0"):
                rows_d.append(mk(arm, item, lang, sr if (sh or arm == "G0") else 1.0 - sr if rg_.random() < 0.5 else sr, f"fam{f:02d}",
                                 unit=(f"shared|{item}|{lang}" if sh else f"u|{arm}|{item}|{lang}")))
idx_d = A.build_index(rows_d)
c_all = A.contrast(idx_d["S"], idx_d["fam"], "G0", "O0", set(), 0)
c_dist = A.contrast(idx_d["S"], idx_d["fam"], "G0", "O0", set(), 0, drop_shared=True)
n_sh = sum(1 for i, l in {(k[1], k[2]) for k in idx_d["S"]} if idx_d["S"][("G0", i, l)]["unit"] == idx_d["S"][("O0", i, l)]["unit"])
check("A21 D: presentations where both arms point at one execution unit are counted and can be dropped (distinct-units-only sensitivity)",
      c_all["n_shared"] == n_sh > 0 and c_dist["n_pres"] == c_all["n_pres"] - n_sh and c_dist["n_shared"] == 0, f"{c_all['n_shared']} {n_sh} {c_dist['n_pres']}")
check("A21 D: a shared presentation has difference 0 by construction (so shared copies cannot manufacture a non-zero effect)",
      all(idx_d["S"][("G0", i, l)]["sr"] == idx_d["S"][("O0", i, l)]["sr"] for (a, i, l) in idx_d["S"] if a == "G0"
          and idx_d["S"][("G0", i, l)]["unit"] == idx_d["S"][("O0", i, l)]["unit"]))
dep = A.dependence_block(idx_d["S"], ["G0", "O0"])
check("A21 D: the dependence block reports cells, distinct units and the share of cells sharing a unit with another arm",
      dep["G0"]["cells"] == 72 and dep["G0"]["cells_sharing_a_unit_with_another_arm"] == n_sh and dep["O0"]["cells_sharing_a_unit_with_another_arm"] == n_sh
      and abs(dep["G0"]["shared_share"] - n_sh / 72) < 1e-12)
dr = A._dep_row(idx_d["S"], idx_d["fam"], "G0", "O0", set())
check("A21 D: the dependence row separates shared-unit and distinct-unit presentations", dr["shared_unit_presentations"] == n_sh and dr["distinct_unit_presentations"] == 72 - n_sh)

# ---- the cluster is the family: copies live inside a family, so resampling families keeps shared outcomes together
tot_sizes = A.by_family(A.item_diffs(idx_d["S"], "G0", "O0", set())[0], idx_d["fam"])
check("A21 D: the unit of resampling is the family (every execution unit sits inside one family)", len(tot_sizes[0]) == 12)

# ---- blocked / crashed executions: infrastructure, scored 0, dropped as pairs in the sensitivity, flagged above 5 %
rows_b = indep_rows(["G+M", "O0"], n_fam=20, per_fam=3, seed=5)
for r_ in rows_b:
    if r_["arm"] == "O0" and r_["pres"].startswith("canon:t-00-0:"):
        r_.update(sr=0.0, exec_failure=True, run_class="infra_exec", failure_class="exec_blocked_state")
idx_b = A.build_index(rows_b)
c_p = A.contrast(idx_b["S"], idx_b["fam"], "G+M", "O0", set(), 0)
c_i = A.contrast(idx_b["S"], idx_b["fam"], "G+M", "O0", set(), 0, drop_infra=True)
check("A21 A: an execution infrastructure failure (blocked state after the re-runs) is kept at SR 0 in the primary contrast and dropped from BOTH arms in the paired sensitivity",
      c_i["dropped_infra"] == 2 and c_i["n_pres"] == c_p["n_pres"] - 2 and A.is_infra(idx_b["S"][("O0", "t-00-0", "en")]) and not A.is_infra(idx_b["S"][("G+M", "t-00-0", "en")]))
it = A.infra_table(idx_b["S"], ["G+M", "O0"])
check("A21 A: the infrastructure table separates planner and execution failures; a rate above 5 % marks the arm infrastructure-sensitive",
      it["O0"]["exec_infra"] == 2 and it["O0"]["infra"] == 0 and it["G+M"]["infra_total"] == 0 and not it["O0"]["infrastructure_sensitive"])
for r_ in rows_b:
    if r_["arm"] == "O0" and r_["pres"].split(":")[1].startswith("t-01"):
        r_.update(exec_failure=True, run_class="infra_exec")
it2 = A.infra_table(A.build_index(rows_b)["S"], ["G+M", "O0"])
check("A21 A: crossing the 5 % line sets the flag", it2["O0"]["infrastructure_sensitive"] and it2["O0"]["infra_total_rate"] > C.INFRA_FLAG_RATE)
check("A21 A: rows from before A21 (infra only) are still infrastructure", A.is_infra({"infra": True}) and not A.is_infra({"infra": False}) and A.is_infra({"run_class": "infra_planner"}))

# ---- X isolation sensitivity: presentations with a tool event or identity problem are dropped from H5 pairs
rows_x = indep_rows(["X+M", "X0", "G+M"], n_fam=10, per_fam=3, seed=9)
for r_ in rows_x:
    if r_["arm"] == "X0" and r_["pres"].startswith("canon:t-02-1:"):
        r_["x_tool_events"] = True
idx_x = A.build_index(rows_x)
cx0 = A.contrast(idx_x["S"], idx_x["fam"], "X+M", "X0", set(), 0)
cx1 = A.contrast(idx_x["S"], idx_x["fam"], "X+M", "X0", set(), 0, drop_flag=A._x_flagged)
check("A21 E: presentations whose X cell had a tool event are dropped from both arms in the isolation-paired sensitivity",
      cx1["dropped_flagged"] == 2 and cx1["n_pres"] == cx0["n_pres"] - 2 and A._x_flagged({"x_identity_problem": True}) and not A._x_flagged({}))

# ---- simulator time: a shared execution's wall time is split among the cells that share it
sc = [{"scored": True, "unit": "U1", "arm": "G0"}, {"scored": True, "unit": "U1", "arm": "O0"}, {"scored": True, "unit": "U2", "arm": "G0"},
      {"scored": False, "unit": "U2", "arm": "O0"}]
ex = [{"status": "done", "run": 0, "unit": "U1", "wall_s": 20.0}, {"status": "done", "run": 0, "unit": "U2", "wall_s": 10.0}, {"status": "done", "run": 1, "unit": "U1", "wall_s": 99.0}]
eb = A.execution_block(sc, ex)
check("A21 D: a shared execution's wall time is split equally among the scored cells that share it; every real run is counted once",
      eb["per_arm_wall_s"] == {"G0": 20.0, "O0": 10.0} and eb["total_wall_s"] == 30.0 and eb["unique_runs"] == 2 and A.execution_block(sc, None) == {"available": False})

# ---- every arm is linked to its execution
sr_ = [{"key": "k1", "unit": "U1", "run": 0, "cell": "c1", "exec_status": "done", "failure_class": "ok"},
       {"key": "k2", "unit": "U1", "run": 0, "cell": "c2", "exec_status": "done", "failure_class": "ok"},
       {"key": "k3", "unit": None, "run": 0, "cell": "c3", "exec_status": None, "failure_class": "plan_rejected"}]
ex_ = [{"unit": "U1", "run": 0, "cells": ["c1", "c2"], "status": "done"}]
check("A21 D: score rows of executed plans are linked to their execution record and cell (a rejected plan needs none)", A.execution_link_problems(sr_, ex_) == [])
bad_ = A.execution_link_problems(sr_ + [{"key": "k4", "unit": "U9", "run": 0, "cell": "c4", "exec_status": "done", "failure_class": "ok"},
                                         {"key": "k5", "unit": "U1", "run": 0, "cell": "cX", "exec_status": "done", "failure_class": "ok"},
                                         {"key": "k6", "unit": None, "run": 0, "cell": "c6", "exec_status": "done", "failure_class": "ok"}], ex_)
check("A21 D: an unlinked score row (missing unit, missing exec record, cell not listed by its unit) is reported", len(bad_) == 3, str(bad_))

# ---- the report mentions the frozen multiplicity statement and the new tables
res_syn = A.analyse(syn_rows, B=299, meta={"namespace": "synthetic", "synthetic": True, "complete": True})
md = A.render_md(res_syn)
check("A21: analysis.json carries the plan hash; the report states H1's treatment, the m = 3 Holm table with the m = 8 transparency line, the exploratory table and the dependence tables (A24.5)",
      res_syn["analysis_plan_sha"] == A.analysis_plan_sha() and "H1 stands alone" in md and "m=3" in md and "m = 8" in md and "Exploratory contrasts" in md and "Shared executions" in md
      and "distinct units only" in md and "planner OR execution infrastructure" in md)

# ---------------------------------------------------------------------------------------------------- 11. static
src = open(os.path.join(MATRIX, "analyze.py"), encoding="utf-8").read()
ast.parse(src)
check("static: never opens the outcome journal, states or the dataset",
      "open(" not in src.split("def read_jsonl")[0] and "outcome.sealed" not in src.replace("outcome.sealed.jsonl, states/", "").replace("Never opens\noutcome.sealed.jsonl", "")
      .replace("outcome.sealed.jsonl or", "") and "general.json" not in src and 'journal_path(' not in src)
check("static: no key or secret handling", all(w not in src for w in ("API_KEY", "keys.py", "os.environ", "getenv")))
check("static: uses the locked inference functions (H1 through h1-rule-1 since A24.1, never P.h1_decision)",
      all(w in src for w in ("H1S.h1_rule_c7", "P.wcr_p", "P.cluster_ci", "P.estimate")) and "P.h1_decision(" not in src)
check("static: no simulator / planner imports", all(w not in src for w in ("ai2thor", "import planners", "step5_live", "ollama")))
check("static: A24.5 Holm family = H2a, H2c, H3a (m = 3); exploratory = H4a-b + the A18 H5a-c; H2b / H3b / H3c are withdrawn and in neither",
      set(A.FAMILY) == {"H2a", "H2c", "H3a"} and set(A.EXPLORATORY) == {"H4a", "H4b", "H5a", "H5b", "H5c"}
      and not ({"H2b", "H3b", "H3c"} & (set(A.FAMILY) | set(A.EXPLORATORY))) and set(A.WITHDRAWN) == {"H2b", "H3b", "H3c"})
check("static: exploratory H5a memory gain on X (one-sided), H5b X+M vs G+M (two-sided), H5c G+M vs X0 non-inferiority (raw p; min(1, 2p) only in the m = 8 comparison) (A18, A24.5)",
      A.EXPLORATORY["H5a"] == ("X+M", "X0", "greater", 0.0) and A.EXPLORATORY["H5b"] == ("X+M", "G+M", "two", 0.0) and A.EXPLORATORY["H5c"] == ("G+M", "X0", "ni_x2", -0.05))
check("static: the three confirmatory members are superiority / two-sided only (no non-inferiority member since H3c was withdrawn)", all(v[2] in ("greater", "two") for v in A.FAMILY.values()))

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
