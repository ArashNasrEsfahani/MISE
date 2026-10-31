"""Three-kitchen intervals with a JOINT family bootstrap. Read only on every input; writes only joint_bootstrap.json (see run.py).

The three-kitchen mean of compute_final.py (noexcl.stratified_mean) is the equal-weight mean of the three per-kitchen
paired estimates; its 95 % interval resamples families INDEPENDENTLY within each kitchen (B = 20000, kitchen k seeded 20261010 + k).
The instruction set is scene-agnostic, so the same items and families recur in every kitchen. The joint bootstrap below draws family ids
ONCE per replicate from the union of the families of the contrast and applies the same multinomial weight to that family in every kitchen
where it appears. Within a kitchen nothing else changes: a family's total is the sum of its items' paired differences (each item the mean
over its languages, A36 paired common subset, shared executions kept), exactly the (_tot, _sizes) of analyze.contrast; each kitchen's
replicate is the weighted ratio; kitchens get equal weight.

Rows are the final rows of compute_final.py (re-run rows + the rescue pass, A36 drop), built with compute_final's own functions, by import.
compute_final.main() is NOT called here; run.py runs it first, and this script reads its final_results.json.

Gates (the script stops, nothing written, if any fails):
  J1 every point estimate (float, pp, per kitchen) equals final_results.json exactly;
  J2 every family partition rebuilt here gives the same (_tot, _sizes) as noexcl.parts_of;
  J3 this code in 'independent' mode with compute_final's seeds reproduces the ci95_pp of final_results.json exactly, and with five other
     seed sets stays within 0.5 pp of it (Monte Carlo error).

    python docs/final-test/reexecuted/run.py        (no simulator, no language model, no network)
"""
import sys

sys.dont_write_bytecode = True

import json   # noqa: E402
import os     # noqa: E402

import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compute_final as CF   # noqa: E402

CN, AN, A36, PLD = CF.CN, CF.AN, CF.A36, CF.PLD
NS, S_, INPUTS = CN.NS, CN.S_, CN.INPUTS
B, SEED, pc1 = CN.B, CN.P.SEED, CN.pc1
OUT = os.path.join(HERE, "joint_bootstrap.json")
FINAL = None                           # None: compute_final.OUT (the final_results.json run.py has just written)
EXTRA_SEEDS = (1, 2, 3, 4, 5)          # seed sets 20261010 + 1000 j (+ k): Monte Carlo check only
MC_TOL_PP = 0.5
bad = []


# ------------------------------------------------------------------------------------------------ rows (compute_final's own path)
def load():
    src = CN.sources()
    raw_new = {K: sum((AN.read_jsonl(p) for p in new), []) for K, (_orig, new) in src.items()}
    gold, _ = CF.S._gold_load(None, None, verify=True)
    rs = CF.rescue_sources()
    raw_final = {K: CF.apply_rescue(raw_new[K], rs, gold) for K in NS}
    main = {K: CF.idx_of(raw_final[K]) for K in NS}
    a6rows = AN.read_jsonl(os.path.join(INPUTS, "rows", "A6.no-exclusion.jsonl"))
    abl = {}
    for K in NS:
        base = [r for r in raw_final[K] if r.get("arm") in CF.A6_MAIN]
        abl[K] = CF.idx_of(base + CF.apply_rescue([r for r in a6rows if r["namespace"] == NS[K] and r["a_version"] == "no-exclusion"], rs, gold))
    larger = {}
    for K, ns in NS.items():
        rows = [r for r in raw_final[K] if r.get("arm") in ("G0", "G+M")]
        for tag in ("e4b", "q9b"):
            rows += CF.apply_rescue(AN.read_jsonl(os.path.join(S_, "N", ns, f"score_{tag}.noexcl.jsonl")), rs, gold)
        larger[K] = CF.idx_of(rows)
    return main, abl, larger, raw_final


# ------------------------------------------------------------------------------------------------ named family partition
def parts_named(idx, d, a, b, level="all", drop=True):
    """(family ids, family totals, family sizes) of analyze.contrast; checked against noexcl.parts_of (gate J2)."""
    ref = CF.parts_safe(idx, d, a, b, level, drop)
    if ref is None:
        return None
    S = PLD.level_index(idx["S"], idx["level"], level)
    diffs, _ = AN.item_diffs(S, a, b, d, False, drop_flag=A36.flagged if drop else None)
    fams, tot, sizes = AN.by_family(diffs, idx["fam"])
    if list(tot) != list(ref[0]) or list(sizes) != list(ref[1]):
        bad.append(f"J2 partition {a} - {b} {level}")
    return fams, tot, sizes


# ------------------------------------------------------------------------------------------------ the two bootstraps
def replicates(pts, mode, seed=SEED):
    """-> (B x 3 per-kitchen replicate ratios, zero-denominator count). mode 'independent' = noexcl.stratified_mean's draw
    (kitchen k: its own families, rng seed + k); 'joint' = one draw over the union of family ids, shared by every kitchen."""
    cols, zero = [], 0
    if mode == "independent":
        for k, (_f, t, s) in enumerate(pts):
            T_, n_ = np.asarray(t, float), np.asarray(s, float)
            F = len(n_)
            W = np.random.default_rng(seed + k).multinomial(F, [1 / F] * F, size=B).astype(float)
            den = W @ n_
            zero += int((den <= 0).sum())
            cols.append((W @ T_) / np.maximum(den, 1e-12))
    else:
        U = sorted(set().union(*[set(f) for f, _t, _s in pts]))
        pos = {f: j for j, f in enumerate(U)}
        W = np.random.default_rng(seed).multinomial(len(U), [1 / len(U)] * len(U), size=B).astype(float)
        for f, t, s in pts:
            Wk = W[:, [pos[x] for x in f]]
            den = Wk @ np.asarray(s, float)
            zero += int((den <= 0).sum())
            cols.append((Wk @ np.asarray(t, float)) / np.maximum(den, 1e-12))
    return np.vstack(cols).T, zero


def interval(pts, mode, seed=SEED):
    R, zero = replicates(pts, mode, seed)
    m = R.mean(axis=1)
    return {"ci95_pp": [pc1(float(np.percentile(m, 2.5))), pc1(float(np.percentile(m, 97.5)))], "se_pp": round(100 * float(m.std(ddof=1)), 2),
            "zero_denominator_replicates": zero, "_R": R}


def kitchen_corr(R):
    """Correlation of the per-kitchen replicates under the joint draw (None where a kitchen's replicates are constant, e.g. every difference 0)."""
    sd = R.std(axis=0)
    one = lambda i, j: None if sd[i] == 0 or sd[j] == 0 else round(float(np.corrcoef(R[:, i], R[:, j])[0, 1]), 3)   # noqa: E731
    return {"K1-K2": one(0, 1), "K1-K3": one(0, 2), "K2-K3": one(1, 2)}


def analyse(name, pts, pub):
    est = [float(sum(t)) / float(sum(s)) for _f, t, s in pts]
    mean = sum(est) / len(est)
    if not (mean == pub["estimate"] and pc1(mean) == pub["estimate_pp"] and [pc1(e) for e in est] == pub["per_kitchen_pp"]):
        bad.append(f"J1 {name}: estimate {mean} {[pc1(e) for e in est]} vs {pub['estimate']} {pub['per_kitchen_pp']}")
    ind = interval(pts, "independent")
    if ind["ci95_pp"] != pub["ci95_pp"]:
        bad.append(f"J3 {name}: independent {ind['ci95_pp']} vs final_results.json {pub['ci95_pp']}")
    mc_ind = [interval(pts, "independent", SEED + 1000 * j)["ci95_pp"] for j in EXTRA_SEEDS]
    dev_ind = max(abs(x[i] - pub["ci95_pp"][i]) for x in mc_ind for i in (0, 1))
    if dev_ind > MC_TOL_PP:
        bad.append(f"J3 {name}: independent mode with other seeds deviates {dev_ind} pp")
    jt = interval(pts, "joint")
    mc_jt = [interval(pts, "joint", SEED + 1000 * j)["ci95_pp"] for j in EXTRA_SEEDS]
    fam_sets = [set(f) for f, _t, _s in pts]
    U = set().union(*fam_sets)
    excl = lambda iv: iv[0] > 0 or iv[1] < 0   # noqa: E731
    return {"estimate": mean, "estimate_pp": pc1(mean), "per_kitchen_pp": [pc1(e) for e in est],
            "existing_ci95_pp": pub["ci95_pp"], "existing_excludes_zero": excl(pub["ci95_pp"]),
            "independent_reproduced": ind["ci95_pp"], "independent_se_pp": ind["se_pp"], "independent_other_seeds": mc_ind, "independent_max_dev_pp": round(dev_ind, 2),
            "joint_ci95_pp": jt["ci95_pp"], "joint_excludes_zero": excl(jt["ci95_pp"]), "joint_se_pp": jt["se_pp"], "joint_other_seeds": mc_jt,
            "joint_zero_denominator_replicates": jt["zero_denominator_replicates"],
            "width_ratio_joint_over_existing": round((jt["ci95_pp"][1] - jt["ci95_pp"][0]) / (pub["ci95_pp"][1] - pub["ci95_pp"][0]), 3) if pub["ci95_pp"][1] > pub["ci95_pp"][0] else None,
            "joint_kitchen_correlation": kitchen_corr(jt["_R"]),
            "families": {"per_kitchen": [len(s) for s in fam_sets], "union": len(U), "in_all_three": len(set.intersection(*fam_sets)),
                         "in_two_or_more": sum(1 for f in U if sum(f in s for s in fam_sets) >= 2)}}


def family_overlap(main):
    """Families and items of the scored cells in each kitchen (canon, r = 0, run = 0), all arms."""
    fam = {K: set(main[K][0]["fam"].values()) for K in NS}
    items = {K: set(main[K][0]["fam"]) for K in NS}
    clash = sorted(i for i in set.intersection(*items.values()) if len({main[K][0]["fam"][i] for K in NS}) > 1)
    U = set().union(*fam.values())
    by_count = {n: sum(1 for f in U if sum(f in fam[K] for K in NS) == n) for n in (1, 2, 3)}
    return {"families_per_kitchen": {K: len(v) for K, v in fam.items()}, "families_union": len(U), "families_in_all_three": len(set.intersection(*fam.values())),
            "families_by_number_of_kitchens": by_count, "families_only_in": {K: sorted(fam[K] - set().union(*(fam[J] for J in NS if J != K))) for K in NS},
            "items_per_kitchen": {K: len(v) for K, v in items.items()}, "items_in_all_three": len(set.intersection(*items.values())),
            "items_union": len(set().union(*items.values())), "items_with_different_family_across_kitchens": clash}


def shared_units(raw_final):
    u = {K: {r.get("unit") for r in raw_final[K] if r.get("unit")} for K in NS}
    return sum(1 for x in set().union(*u.values()) if sum(x in u[K] for K in NS) > 1)


def main():
    R = json.load(open(FINAL or CF.OUT, encoding="utf-8"))
    print("loading the final rows (compute_final path)", flush=True)
    I, AI, LI, raw_final = load()
    out = {"schema": "reexecuted.joint_bootstrap/1", "definition": __doc__.split("\n\n")[1].replace("\n", " "), "B": B, "seed": SEED, "contrasts": {}}
    C = {}
    # main three-kitchen gain and the language gains
    C["G+M - G0"] = ([parts_named(I[K][0], I[K][1], "G+M", "G0") for K in NS], R["three_kitchen_gain"]["G+M - G0"]["all"])
    for lg, lab in (("en", "English G+M - G0"), ("fa", "Persian G+M - G0")):
        C[lab] = ([parts_named(CN.lang_filter(I[K][0], lg), I[K][1], "G+M", "G0") for K in NS], R["language"]["paired_gain_three_kitchen"][lg])
    # larger planners (all eight three-kitchen contrasts of compute_final.larger_planners)
    for key in R["larger_planners"]["three_kitchen"]:
        a, b = key.split(" - ")
        C[key + " (larger-planner rows)"] = ([parts_named(LI[K][0], LI[K][1], a, b) for K in NS], R["larger_planners"]["three_kitchen"][key])
    # memory components (compute_final.tk_arm labels)
    A6 = json.load(open(os.path.join(INPUTS, "analysis", "A6.json"), encoding="utf-8"))
    TKMAP = {"no_request - G+M": "no_request", "no_procedures - G+M": "no_procedures", "no_req_proc - G+M": "no_req_proc", "only_req_proc - G+M": "only_req_proc",
             "irrelevant - G+M": "irrelevant", "rep - G+M": "rep", "other_kitchen - G+M": None,
             "O no_request - O+M": "O:no_request", "O no_procedures - O+M": "O:no_procedures", "O no_req_proc - O+M": "O:no_req_proc", "O only_req_proc - O+M": "O:only_req_proc"}
    for name, m in TKMAP.items():
        pts = []
        for K, ns in NS.items():
            lab = A6["three_kitchen"]["adjusted"][name]["arms"][ns] if m is None else (("O~" + m[2:]) if m.startswith("O:") else ("G~" + m))
            pts.append(parts_named(AI[K][0], AI[K][1], lab, "G+M" if lab.startswith("G~") else "O+M"))
        C["ablation " + name] = (pts, R["ablation"]["three_kitchen"][name])
    C["ablation O+M - O0"] = ([parts_named(I[K][0], I[K][1], "O+M", "O0") for K in NS], R["ablation"]["three_kitchen"]["O+M - O0"])
    for name, (pts, pub) in C.items():
        if any(p is None for p in pts):
            bad.append(f"{name}: a kitchen has no contrast")
            continue
        print(" ", name, flush=True)
        out["contrasts"][name] = analyse(name, pts, pub)
    out["family_overlap_scored_cells"] = family_overlap(I)
    out["family_overlap_scored_cells"]["execution_units_in_more_than_one_kitchen"] = shared_units(raw_final)
    if bad:
        raise SystemExit("GATE FAILED, nothing written:\n  " + "\n  ".join(bad[:40]))
    out["gates"] = {"J1_point_estimates_equal_final_results": True, "J2_partitions_equal_parts_of": True,
                    "J3_independent_mode_reproduces_published_ci_exactly": True, "J3_mc_tolerance_pp_other_seeds": MC_TOL_PP}
    v = list(out["contrasts"].values())
    out["summary"] = {"n_contrasts": len(v), "zero_exclusion_changes": [n for n, x in out["contrasts"].items() if x["existing_excludes_zero"] != x["joint_excludes_zero"]],
                      "width_ratio_range": [min(x["width_ratio_joint_over_existing"] for x in v if x["width_ratio_joint_over_existing"]),
                                            max(x["width_ratio_joint_over_existing"] for x in v if x["width_ratio_joint_over_existing"])]}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("gates passed; wrote", OUT)
    for n, x in out["contrasts"].items():
        print(f"{n:42s} {x['estimate_pp']:+6.1f}  existing {x['existing_ci95_pp']}  joint {x['joint_ci95_pp']}  ratio {x['width_ratio_joint_over_existing']}")


if __name__ == "__main__":
    main()
