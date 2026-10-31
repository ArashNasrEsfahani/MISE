"""Shared definitions and helpers of the re-executed analysis. Read only.

Estimators, by import:
  rates          adjusted.tables -> analyze.per_arm_tables (item-weighted SR, 95 % family-cluster bootstrap)
  K1 paired      paired.paired -> analyze.contrast -> final_test_power.cluster_ci (95 %, B = 20000, seed 20261010)
  K2, K3         adjusted.cr2_public -> k23_analyze.kitchen_cr2 (nominal 96.5 % CR2)
  held-out mean  k23_analyze.pooled_result (nominal 96.5 % CR2)
  three kitchens equal-weight mean of the per-kitchen estimates, 95 % kitchen-stratified family bootstrap (stratified_mean below)
Rounding: rates "%.1f" % (100 x + 1e-9), differences f"{100 x:.1f}".
"""
import sys

sys.dont_write_bytecode = True

import hashlib   # noqa: E402
import os        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adjusted as CA               # noqa: E402
import paired as PLD                # noqa: E402
import numpy as np                  # noqa: E402

AN, A36, KA = CA.AN, CA.A36, CA.KA
P = PLD.P
ROOT = CA.ROOT
S_ = os.path.join(ROOT, "docs", "final-test", "strengthen")
INPUTS = os.path.join(HERE, "inputs")
NS = {"K1": "test-v1", "K2": "test-k2-v1", "K3": "test-k3-v1"}
SCENE = {"K2": "FloorPlan15", "K3": "FloorPlan17"}
PLANNERS = ["G0", "G+M", "O0", "O+M", "X0", "X+M"]
FRONTIER = ["O0", "O+M", "X0", "X+M"]
B = 20000
EXTRA_PAIRS = (("X+M", "X0"), ("P+M", "P0"), ("G+M", "X0"), ("G+M", "X+M"))
LEVEL_PAIRS = (("G+M", "G0"), ("O+M", "O0"), ("X+M", "X0"), ("P+M", "P0"), ("G+M", "O0"), ("G+M", "O+M"), ("G+M", "X0"), ("G+M", "X+M"))
LEVELS = ("L1", "L2", "L3", "all")
#: the items removed by the contamination sensitivity check: 10 exact matches (EXACT) and 8 further matching items (CONTAINS)
EXACT = {"g-l2-coffee-machine", "g-l3-coffee", "g-l2-stove-pan", "g-l2-stove-pot", "g-l3-pan-heating", "g-l3-pot-heating",
         "g-l2-slice-bread", "g-l3-potato-for-frying", "g-l3-shred-lettuce", "g-l3-tomato-for-salad"}
CONTAINS = {"g-l2-knife-bread-toaster", "g-l2-stove-water-kettle", "g-l2-stove-water-pot", "g-l3-boil-water",
            "g-l3-chilled-water", "g-l3-sandwich-prep", "g-l3-tea-kettle", "g-l3-toast"}
p1, pc1 = PLD.p1, PLD.pc1


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def sources():
    """Per kitchen: (sealed score files, re-run score files). The re-run files are the sealed rows with each restoration-failure row
    replaced by the outcome of its re-execution with the same cached plan (task R)."""
    out = {}
    for K, ns in NS.items():
        new = [os.path.join(S_, "R", ns, "score.noexcl.jsonl")]
        if K != "K1":
            new.append(os.path.join(S_, "R", ns, "score.codex.noexcl.jsonl"))
        orig = {"K1": [CA.K1_SCORE], "K2": [CA.K2_SCORE, CA.K2_CODEX], "K3": [CA.K3_SCORE, CA.K3_CODEX]}[K]
        out[K] = (orig, new)
    return out


def rate_pp2(t):
    return None if t is None else {"sr": round(100 * t["sr"], 2), "ci95": [round(100 * x, 2) for x in t["ci95"]], "n_items": t["n_items"]}


def rate_out(t):
    if t is None:
        return None
    return {"sr": t["sr"], "pp": p1(t["sr"]), "ci95": t["ci95"], "ci95_pp": [p1(t["ci95"][0]), p1(t["ci95"][1])], "n_items": t["n_items"],
            "n_families": t["n_families"]}


def tables_out(tab):
    return {a: {"overall": rate_out(v["overall"]), "n_presentations": v["n_presentations"], "level_presentations": v["level_presentations"],
                "level": {L: rate_out(x) for L, x in v["level"].items()}, "lang": {g: rate_out(x) for g, x in v["lang"].items()},
                "level_x_lang": {k: rate_out(x) for k, x in v["level_x_lang"].items()}} for a, v in tab.items()}


def k23_parts(i, d, drop):
    parts = CA.contrasts(i, d, drop)
    for a, b in EXTRA_PAIRS:
        kw = {"drop_flag": A36.flagged} if drop else {}
        parts[f"{a} - {b}"] = KA.contrast_parts(i["S"], i["fam"], a, b, d, **kw)
    return parts


def stratified_mean(pts, seed=P.SEED):
    """Equal-weight mean of the per-kitchen estimates; 95 % interval from a family bootstrap stratified by kitchen."""
    est = [float(sum(t)) / float(sum(s)) for t, s in pts]
    sims = []
    for k, (t, s) in enumerate(pts):
        T_, n_ = np.asarray(t, float), np.asarray(s, float)
        F = len(n_)
        W = np.random.default_rng(seed + k).multinomial(F, [1 / F] * F, size=B).astype(float)
        sims.append((W @ T_) / np.maximum(W @ n_, 1e-12))
    m = np.mean(np.vstack(sims), axis=0)
    return {"estimate": sum(est) / len(est), "estimate_pp": pc1(sum(est) / len(est)), "per_kitchen_pp": [pc1(e) for e in est],
            "ci95_pp": [pc1(float(np.percentile(m, 2.5))), pc1(float(np.percentile(m, 97.5)))]}


def parts_of(idx, d, a, b, level, drop):
    S = PLD.level_index(idx["S"], idx["level"], level)
    c = AN.contrast(S, idx["fam"], a, b, d, None, drop_flag=A36.flagged if drop else None, amended=drop)
    return None if c is None else (c["_tot"], c["_sizes"])


def lang_filter(idx, lang):
    return dict(idx, S={k: v for k, v in idx["S"].items() if k[2] == lang})


def without_items(idx, items):
    return dict(idx, S={k: v for k, v in idx["S"].items() if k[1] not in items})


def level_means(tab, arms):
    """Equal-weight mean of the per-kitchen rates (each kitchen's SR is item-weighted), unrounded inputs."""
    out = {}
    for a in arms:
        o = {L: sum(tab[K][a]["level"][L]["sr"] for K in NS) / 3 for L in ("L1", "L2", "L3")}
        o["all"] = sum(tab[K][a]["overall"]["sr"] for K in NS) / 3
        out[a] = {k: {"sr": v, "pp": p1(v)} for k, v in o.items()}
    return out


def steps_complete_goal_failed(idx, arm, level="L3"):
    n, hit = 0, 0
    for (a, i, _), r in idx["S"].items():
        if a == arm and idx["level"].get(i) == level:
            n += 1
            if r.get("executed") and not r.get("goal_ok") and r.get("steps_n") and r.get("steps_ok_n") == r.get("steps_n") and not AN.is_infra(r):
                hit += 1
    return hit, n


def agreement(rows, arms=("G0", "G+M", "O0", "O+M")):
    v = AN.variance_decomposition(rows, list(arms))
    return {a: {"outcome_agreement_rate": v[a]["execution"]["outcome_agreement_rate"], "n": v[a]["execution"]["n"]} for a in arms}
