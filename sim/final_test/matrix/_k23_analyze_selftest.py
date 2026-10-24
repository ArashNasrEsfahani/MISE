"""Selftest for sim/final_test/matrix/k23_analyze.py (the A29 Kitchens 2/3 replication analysis). Hermetic: SYNTHETIC stores written to a
temp dir (no real data), no simulator, no network, no API. Kept beside the K23 modules (like _k23_namespaces_selftest.py), outside the
frozen sim/_*selftest.py gate manifest of Kitchen 1.

    python sim/final_test/matrix/_k23_analyze_selftest.py
"""
from __future__ import annotations

import ast
import builtins
import json
import math
import os
import random
import shutil
import sys
import tempfile

MATRIX = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(MATRIX))
sys.path.insert(0, SIM)
sys.path.insert(0, MATRIX)

import numpy as np                      # noqa: E402
from scipy.linalg import sqrtm          # noqa: E402
from scipy.stats import t as tdist      # noqa: E402

import common as C                      # noqa: E402
import k23_ns as K                      # noqa: E402
import final_test_h1_stats as H1S       # noqa: E402
import k23_analyze as KA                # noqa: E402

N = {"pass": 0, "fail": 0}
LANGS = ("en", "fa")
B_COST = 499
NS15, NS17 = "test-k2-v1", "test-k3-v1"


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception, text=None):
    try:
        fn()
    except exc as e:
        return text is None or text in str(e)
    except Exception:                                                  # noqa: BLE001
        return False
    return False


def close(x, y, tol=1e-9):
    return x is not None and y is not None and abs(float(x) - float(y)) <= tol * max(1.0, abs(float(x)), abs(float(y)))


TMP = tempfile.mkdtemp(prefix="k23_analyze_")
_roots = [0]


def new_root():
    _roots[0] += 1
    r = os.path.join(TMP, f"root{_roots[0]}")
    os.makedirs(C.paths(r)["matrix_docs"], exist_ok=True)
    return r


def row(ns, arm, item, lang, sr, family, level="L1", **kw):
    pres = C.pres_id(item, lang)
    r = {"schema": "ftm.score/1", "key": f"{ns}|score|{arm}|{pres}|r0|run0", "namespace": ns, "arm": arm, "pres": pres, "r": 0, "run": 0,
         "unit": f"u|{ns}|{arm}|{item}|{lang}", "cell": f"{ns}|plan|{arm}|{pres}|r0", "scored": True, "sr": float(sr), "strict_sr": float(sr),
         "valid": True, "executed": True, "invalid_episode": False, "infra": False, "exec_failure": False, "failure_class": "ok", "run_class": "ok",
         "item": item, "lang": lang, "world": "canon", "level": level, "family": family, "block": "main", "unrealistic_slice": False, "n_slice": 0}
    r.update(kw)
    return r


def mk_rows(ns, table, fam, level_of=None):
    """table: {arm: {item: sr | {"en": sr, "fa": sr}}}; sr None = presentation missing."""
    out = []
    for arm, per in table.items():
        for item, v in per.items():
            for lg in LANGS:
                sr = v.get(lg) if isinstance(v, dict) else v
                if sr is None:
                    continue
                out.append(row(ns, arm, item, lg, sr, fam[item], (level_of or {}).get(item, "L1")))
    return out


def write_store(root, ns, rows, *, complete=True, seal_ns=None, dirname=None, plan_rows=None):
    d = os.path.join(C.paths(root)["matrix_docs"], dirname or ns)
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "score.jsonl")
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(d, "score.seal.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"schema": "ftm.score.seal/1", "namespace": seal_ns or ns, "sha256": C.sha256_file(sp), "n": len(rows),
                   "complete": complete}, f)
    if plan_rows is not None:
        with open(os.path.join(d, "plan.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in plan_rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return d


def stores_of(root, d15, d17, partial=False):
    return {"FloorPlan15": KA.load_store(d15, "FloorPlan15", allow_partial=partial, root=root),
            "FloorPlan17": KA.load_store(d17, "FloorPlan17", allow_partial=partial, root=root)}


# ================================================================================================ the hand-checkable toy
FAM = {"i1": "f1", "i2": "f1", "i3": "f2", "i4": "f3", "i5": "f2", "i6": "f4", "i7": "f4"}
T15 = {"G+M": {"i1": 1, "i2": 0, "i3": 0, "i4": 1}, "O0": {"i1": 0, "i2": 0, "i3": 1, "i4": 1},
       "G0": {"i1": 0, "i2": 0, "i3": 0, "i4": 0}, "O+M": {"i1": 1, "i2": 1, "i3": 1, "i4": 0},
       "P0": {"i1": 1, "i2": 1, "i3": 1, "i4": 1}, "P+M": {"i1": 1, "i2": 0, "i3": 1, "i4": 1}}
T17 = {"G+M": {"i1": 1, "i3": 0, "i5": 0, "i6": 1, "i7": 0}, "O0": {"i1": 0, "i3": 0, "i5": 0, "i6": 0, "i7": 1},
       "G0": {"i1": 0, "i3": 0, "i5": 0, "i6": 0, "i7": 0}, "O+M": {"i1": 1, "i3": 1, "i5": 0, "i6": 1, "i7": 1},
       "P0": {"i1": 1, "i3": 1, "i5": 1, "i6": 1, "i7": 1}, "P+M": {"i1": 1, "i3": 1, "i5": 1, "i6": 1, "i7": 0}}


def hand_diffs(table, a, b):
    return {i: table[a][i] - table[b][i] for i in table[a] if i in table[b]}


def hand_one(diffs, fam):
    """A24.1-H1 item 2 algebra, written out by hand (independent of the module)."""
    fams = sorted({fam[i] for i in diffs})
    S = [sum(d for i, d in diffs.items() if fam[i] == f) for f in fams]
    n = [sum(1 for i in diffs if fam[i] == f) for f in fams]
    Nn = sum(n)
    est = sum(S) / Nn
    se2 = sum((S[j] - n[j] * est) ** 2 / (1 - n[j] / Nn) for j in range(len(fams))) / Nn ** 2
    c = [(1 - n[j] / Nn) ** -0.5 for j in range(len(fams))]
    G = [[c[f] * c[h] * ((n[f] if f == h else 0) - n[f] * n[h] / Nn) for h in range(len(fams))] for f in range(len(fams))]
    nu = sum(G[f][f] for f in range(len(fams))) ** 2 / sum(G[f][h] ** 2 for f in range(len(fams)) for h in range(len(fams)))
    q = tdist.ppf(0.9825, nu)
    return {"estimate": est, "se2": se2, "nu": nu, "L": est - q * math.sqrt(se2), "U": est + q * math.sqrt(se2)}


def hand_pooled(d15, d17, fam):
    """A29 item 3 closed form, written out by hand (loops; independent of the module)."""
    kit = []
    for d in (d15, d17):
        fams = sorted({fam[i] for i in d})
        S = {f: sum(x for i, x in d.items() if fam[i] == f) for f in fams}
        n = {f: sum(1 for i in d if fam[i] == f) for f in fams}
        Nk = sum(n.values())
        kit.append({"S": S, "n": n, "N": Nk, "est": sum(S.values()) / Nk, "a": 1 / (2 * Nk)})
    allf = sorted(set(kit[0]["S"]) | set(kit[1]["S"]))
    est = (kit[0]["est"] + kit[1]["est"]) / 2

    def c(k, f):
        return (1 - k["n"][f] / k["N"]) ** -0.5 if f in k["n"] else 0.0
    se2 = sum(sum(k["a"] * c(k, f) * (k["S"].get(f, 0) - k["n"].get(f, 0) * k["est"]) for k in kit) ** 2 for f in allf)
    B = {(f, h): sum(k["a"] ** 2 * c(k, f) * c(k, h) * ((k["n"].get(f, 0) if f == h else 0) - k["n"].get(f, 0) * k["n"].get(h, 0) / k["N"])
                     for k in kit) for f in allf for h in allf}
    nu = sum(B[(f, f)] for f in allf) ** 2 / sum(v * v for v in B.values())
    q = tdist.ppf(0.9825, nu)
    return {"estimate": est, "se2": se2, "nu": nu, "L": est - q * math.sqrt(se2), "U": est + q * math.sqrt(se2)}


def ref_cr2(kitchen_items, w):
    """A THIRD, independent CR2 for the cross-check: Bell-McCaffrey sandwich with A_g = sqrtm(inv(I - H_gg)) and the Satterthwaite df of the
    quadratic form e'Ze = eps'(I-H)Z(I-H)eps under the identity working model: nu = tr(W)^2 / tr(W W)."""
    rows = [(k, f, float(y)) for k, its in enumerate(kitchen_items) for f, y in its]
    n, nK = len(rows), len(kitchen_items)
    X = np.zeros((n, nK))
    y = np.array([r[2] for r in rows])
    for i, r in enumerate(rows):
        X[i, r[0]] = 1.0
    M = np.linalg.inv(X.T @ X)
    H = X @ M @ X.T
    IH = np.eye(n) - H
    e = IH @ y
    Z = np.zeros((n, n))
    var = 0.0
    for f in sorted({r[1] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r[1] == f]
        Ag = np.real(sqrtm(np.linalg.inv(np.eye(len(ix)) - H[np.ix_(ix, ix)])))
        z = np.zeros(n)
        z[ix] = Ag @ X[ix] @ M @ np.asarray(w)
        Z += np.outer(z, z)
        var += float(z @ e) ** 2
    W = IH @ Z @ IH
    return {"estimate": float(np.asarray(w) @ (M @ X.T @ y)), "se": math.sqrt(var), "nu": float(np.trace(W) ** 2 / np.trace(W @ W))}


# ------------------------------------------------------------------------------------------------ 1. toy: every contrast, per kitchen + pooled
root = new_root()
d15 = write_store(root, NS15, mk_rows(NS15, T15, FAM))
d17 = write_store(root, NS17, mk_rows(NS17, T17, FAM))
res = KA.analyse_k23(stores_of(root, d15, d17), B_cost=B_COST)
for a, b in KA.CONTRASTS:
    ck = KA.ckey(a, b)
    for s, tab in (("FloorPlan15", T15), ("FloorPlan17", T17)):
        r = res["kitchens"][s]["contrasts"][ck]["primary"]
        h = hand_one(hand_diffs(tab, a, b), FAM)
        check(f"toy {s} {ck}: estimate / se / nu / nominal 96.5 % interval = the hand A24.1-H1 algebra",
              r["available"] and close(r["estimate"], h["estimate"]) and close(r["se"] ** 2, h["se2"]) and close(r["nu"], h["nu"])
              and close(r["interval"][0], h["L"]) and close(r["interval"][1], h["U"]), (r, h))
    p = res["combined"]["contrasts"][ck]["primary"]
    hp = hand_pooled(hand_diffs(T15, a, b), hand_diffs(T17, a, b), FAM)
    check(f"toy pooled {ck}: (est_15 + est_17) / 2, se, nu, interval = the hand A29 closed form",
          p["available"] and close(p["estimate"], hp["estimate"]) and close(p["se"] ** 2, hp["se2"]) and close(p["nu"], hp["nu"])
          and close(p["interval"][0], hp["L"]) and close(p["interval"][1], hp["U"]), (p, hp))
r = res["kitchens"]["FloorPlan15"]["contrasts"]["G+M - O0"]["primary"]
check("toy literal FloorPlan15 G+M - O0: est 0, se^2 = 10/48, nu = 1.8, interval -/+ t_1.8(.9825) se",
      r["estimate"] == 0.0 and close(r["se"] ** 2, 10 / 48) and close(r["nu"], 1.8)
      and close(r["interval"][1], tdist.ppf(0.9825, 1.8) * math.sqrt(10 / 48)), r)
r = res["kitchens"]["FloorPlan17"]["contrasts"]["G+M - O0"]["primary"]
check("toy literal FloorPlan17 G+M - O0: est 1/5, S_f = (1, 0, 0), n_f = (1, 2, 2)", close(r["estimate"], 0.2) and r["S"] == [1.0, 0.0, 0.0]
      and r["n"] == [1, 2, 2], r)
lit = {"G+M - O0": (0.0, 0.2), "G+M - G0": (0.5, 0.4), "O+M - O0": (0.25, 0.6), "G+M - O+M": (-0.25, -0.4)}
check("toy literal kitchen estimates of all four contrasts (15 / 17): 0/.2, .5/.4, .25/.6, -.25/-.4",
      all(close(res["kitchens"]["FloorPlan15"]["contrasts"][k]["primary"]["estimate"], v[0])
          and close(res["kitchens"]["FloorPlan17"]["contrasts"][k]["primary"]["estimate"], v[1]) for k, v in lit.items()))
p = res["combined"]["contrasts"]["G+M - O0"]["primary"]
check("toy literal pooled G+M - O0: est 0.1, se^2 = 0.1119465631 (hand value), 4 clusters, 2 shared families",
      close(p["estimate"], 0.1) and close(p["se"] ** 2, 0.11194656311834908) and p["n_clusters"] == 4 and p["shared_families"] == 2, p)
check("toy: the run-time matrix-CR2 cross-check ran and agreed on every pooled contrast and view",
      all(v["cross_check"]["agree"] for k in res["combined"]["contrasts"].values() for v in k.values() if v.get("available")))

# ------------------------------------------------------------------------------------------------ 2. equal weighting
check("equal weight: pooled est = (0 + 0.2) / 2 = 0.1, NOT the item-pooled mean 1/9", close(p["estimate"], 0.1) and not close(p["estimate"], 1 / 9))
rt2 = new_root()
F2 = dict(FAM, **{f"j{k}": f"g{k % 3}" for k in range(12)})
T17big = {arm: dict(v) for arm, v in T17.items()}
for k in range(12):                                         # 12 extra FloorPlan17 items: G+M wins 3 of them, the rest tie at 0
    for arm in T17big:
        T17big[arm][f"j{k}"] = 1 if (arm == "G+M" and k < 3) else 0
rb = KA.analyse_k23(stores_of(rt2, write_store(rt2, NS15, mk_rows(NS15, T15, FAM)), write_store(rt2, NS17, mk_rows(NS17, T17big, F2))), B_cost=B_COST)
e17b = 4 / 17                                               # (1 + 0 + 0 + 1 - 1) + 3 over 5 + 12 items
pb = rb["combined"]["contrasts"]["G+M - O0"]["primary"]
check("equal weight: a kitchen 4x larger still gets weight 1/2 (est = (est_15 + est_17) / 2, N_k 4 vs 17)",
      close(pb["estimate"], (0.0 + e17b) / 2) and pb["N_k"] == [4.0, 17.0] and pb["weights"] == [0.5, 0.5], pb)
pw = KA.pooled_cr2([{"fams": ["a", "b"], "S": [1.0, 0.0], "n": [1, 1]}, {"fams": ["a", "b"], "S": [10.0, 0.0], "n": [10, 10]}])
check("equal weight: kitchen means 0.5 and 0.5 with N 2 vs 20 -> pooled 0.5 (weights do not follow N)", close(pw["estimate"], 0.5))

# ------------------------------------------------------------------------------------------------ 3. shared-family clustering
pl = [{"fams": ["f1", "f2", "f3"], "S": [1.0, -1.0, 0.0], "n": [2, 1, 1]}, {"fams": ["f1", "f2", "f4"], "S": [1.0, 0.0, 0.0], "n": [1, 2, 2]}]
sh = KA.pooled_cr2(pl)
sep = KA.pooled_cr2([pl[0], {"fams": ["x1", "x2", "f4"], "S": [1.0, 0.0, 0.0], "n": [1, 2, 2]}])     # same data, families NOT shared
manual_shared = (1 / 8 * math.sqrt(2) + 1 / 10 * math.sqrt(5) / 2 * 0.8) ** 2 + (-1 / 8 * 2 / math.sqrt(3) - 1 / 10 * math.sqrt(5 / 3) * 0.4) ** 2 \
    + (1 / 10 * math.sqrt(5 / 3) * 0.4) ** 2
manual_sep = (1 / 8 * math.sqrt(2)) ** 2 + (1 / 10 * math.sqrt(5) / 2 * 0.8) ** 2 + (1 / 8 * 2 / math.sqrt(3)) ** 2 \
    + (1 / 10 * math.sqrt(5 / 3) * 0.4) ** 2 + (1 / 10 * math.sqrt(5 / 3) * 0.4) ** 2
check("shared families: a family in both kitchens is ONE cluster (se^2 = sum_f [sum_k ...]^2, by hand), 4 clusters",
      close(sh["se"] ** 2, manual_shared) and sh["n_clusters"] == 4, (sh["se"] ** 2, manual_shared))
check("shared families: treating them as separate clusters gives the different (wrong) value, 6 clusters",
      close(sep["se"] ** 2, manual_sep) and sep["n_clusters"] == 6 and not close(sh["se"], sep["se"]))
g = res["combined"]["contrasts"]["G+M - O0"]["primary"]["guard"]
check("shared families: pooled guard counts DISTINCT discordant families across kitchens (f1 discordant in both counted once)",
      g["K"] == 3 and g["K_sum_of_kitchens"] == 4 and g["K_per_kitchen"] == {"FloorPlan15": 2, "FloorPlan17": 2} and not g["passed"], g)

# ------------------------------------------------------------------------------------------------ 4. CR2 cross-check vs a standard implementation
def parts_from_items(items):
    out = []
    for its in items:
        fams = sorted({f for f, _ in its})
        out.append({"fams": fams, "S": [sum(y for f2, y in its if f2 == f) for f in fams], "n": [sum(1 for f2, _ in its if f2 == f) for f in fams]})
    return out


cases = {
    "unequal sizes": [[("a", 1.0), ("a", 0.0), ("a", 1.0), ("b", -1.0), ("c", 0.5), ("c", 0.0)],
                      [("a", 0.0), ("b", 1.0), ("b", 0.5), ("c", -0.5), ("d", 1.0), ("d", 1.0), ("d", 0.0), ("d", -1.0)]],
    "shared families": [[("a", 1.0), ("b", 0.0), ("b", 1.0), ("c", -1.0)], [("a", 1.0), ("b", -0.5), ("c", 0.0), ("c", 1.0)]],
    "missing data (families absent from one kitchen)": [[("a", 1.0), ("a", 0.5), ("b", 0.0), ("e", -1.0)],
                                                       [("b", 1.0), ("c", 0.0), ("c", 0.5), ("d", -0.5), ("d", 1.0)]],
}
rng = random.Random(20261018)
for k in range(25):
    allf = [f"F{j}" for j in range(rng.randint(3, 12))]
    kit = []
    for _ in range(2):
        fs = [f for f in allf if rng.random() < 0.75] or allf[:2]
        if len(fs) < 2:
            fs = allf[:2]
        kit.append([(f, rng.choice((-1.0, -0.5, 0.0, 0.5, 1.0))) for f in fs for _ in range(rng.randint(1, 4))])
    cases[f"random {k}"] = kit
bad = []
for name, items in cases.items():
    pc = KA.pooled_cr2(parts_from_items(items))
    mx = KA.cr2_matrix(items)
    rf = ref_cr2(items, [0.5, 0.5])
    if not (pc["defined"] and close(pc["estimate"], rf["estimate"]) and close(pc["se"], rf["se"]) and close(pc["nu"], rf["nu"])
            and close(mx["se"], rf["se"]) and close(mx["nu"], rf["nu"])):
        bad.append((name, pc.get("se"), rf["se"], pc.get("nu"), rf["nu"]))
check(f"CR2 cross-check: closed form = module matrix CR2 = an independent sqrtm/trace CR2 on {len(cases)} datasets "
      "(unequal sizes, shared families, missing data, 25 random)", not bad, bad[:3])
okw = []
for k in range(15):
    fams = [f"F{j}" for j in range(rng.randint(2, 15))]
    items = [(f, rng.choice((-1.0, -0.5, 0.0, 0.5, 1.0))) for f in fams for _ in range(rng.randint(1, 5))]
    pp = parts_from_items([items])[0]
    one = KA.pooled_cr2([pp], [1.0])
    c7 = H1S.h1_rule_c7(pp["S"], pp["n"], 25)
    okw.append(one["defined"] == c7["defined"] and close(one["estimate"], c7["estimate"])
               and (not one["defined"] or (close(one["se"], c7["se"]) and close(one["nu"], c7["nu"]) and close(one["interval"][0], c7["interval"][0])
                                           and close(one["interval"][1], c7["interval"][1]))))
check("one-kitchen weight 1 recovers A24.1-H1 (h1_rule_c7: estimate, se, nu, interval) on 15 random datasets", all(okw))
check("pooled: a kitchen with a single family -> CR2 undefined (leverage 1), matrix singular, cross-check consistent",
      (lambda r, m: not r["defined"] and "single family" in r["reason"] and m["singular"] and KA.cross_check_agrees(r, m))(
          KA.pooled_cr2([{"fams": ["a"], "S": [1.0], "n": [2]}, {"fams": ["a", "b"], "S": [1.0, 0.0], "n": [1, 1]}]),
          KA.cr2_matrix([[("a", 1.0), ("a", 0.0)], [("a", 1.0), ("b", 0.0)]])))
z = KA.pooled_cr2([{"fams": ["a", "b"], "S": [2.0, 1.0], "n": [2, 1]}, {"fams": ["a", "c"], "S": [1.0, 1.0], "n": [1, 1]}])
check("pooled: constant differences -> observed se = 0 -> interval undefined (no threshold statement possible)",
      not z["defined"] and z["reason"] == "observed se = 0" and z["interval"] == [None, None]
      and KA.cross_check_agrees(z, KA.cr2_matrix([[("a", 1.0), ("a", 1.0), ("b", 1.0)], [("a", 1.0), ("c", 1.0)]])))

# ------------------------------------------------------------------------------------------------ 5. wording, guard, criteria
def big_table(n_fam, gm_minus_o0, seed, ns_fam_prefix="F"):
    """n_fam families of 2-4 items; item 0 of every family (and item 1 of every 4th) is discordant with sign gm_minus_o0 (+1: G+M wins,
    -1: O0 wins); the rest tie. Unequal family sizes and discordance counts give between-family variation (se > 0)."""
    fam, tab = {}, {a: {} for a in ("G+M", "O0", "G0", "O+M", "P0", "P+M")}
    rg = random.Random(seed)
    for f in range(n_fam):
        n_disc = 2 if f % 4 == 0 else 1
        for j in range(2 + f % 3):
            it = f"b{f}-{j}"
            fam[it] = f"{ns_fam_prefix}{f}"
            base = rg.choice((0, 1))
            gm, o0 = (base, base) if j >= n_disc else ((1, 0) if gm_minus_o0 > 0 else (0, 1))
            tab["G+M"][it], tab["O0"][it] = gm, o0
            tab["G0"][it], tab["O+M"][it], tab["P0"][it], tab["P+M"][it] = 0, o0, 1, 1
    return tab, fam


def run_big(sign15, sign17, partial=False):
    rt = new_root()
    t15, f15 = big_table(24, sign15, 1)
    t17, f17 = big_table(24, sign17, 2)
    a = write_store(rt, NS15, mk_rows(NS15, t15, f15), complete=not partial)
    b = write_store(rt, NS17, mk_rows(NS17, t17, f17))
    return KA.analyse_k23(stores_of(rt, a, b, partial=partial), B_cost=B_COST)


toy_c = res["combined"]["g_m_vs_o0_criteria"]["primary"]
check("toy (K < 20): both criteria 'not met', no threshold statement, no thesis 'met' sentence; intervals still reported",
      toy_c["average"]["status"] == "not met" and not toy_c["average"]["threshold_statement_permitted"]
      and toy_c["average"]["upper_bound_below_minus_5pp"] is None and "guard failed" in toy_c["average"]["reason"]
      and toy_c["each_kitchen"]["status"] == "not met" and not any("met the exploratory" in x for x in toy_c["thesis_sentences"])
      and res["combined"]["contrasts"]["G+M - O0"]["primary"]["interval"][0] is not None, toy_c)
rp = run_big(+1, +1)
c = rp["combined"]["g_m_vs_o0_criteria"]["primary"]
check("both kitchens G+M ahead in 24 families (K >= 20): average AND each-kitchen criteria met, with the registered sentences",
      c["average"]["met"] is True and c["each_kitchen"]["met"] is True and c["average"]["threshold_statement_permitted"]
      and any(KA.AVERAGE_SENTENCE in x for x in c["thesis_sentences"]) and any(KA.EACH_SENTENCE in x for x in c["thesis_sentences"]), c)
check("never 'was shown non-inferior' and no superiority wording in any criterion or sentence",
      not any(KA.NEVER_SENTENCE in x for x in c["thesis_sentences"])
      and "superior" not in json.dumps({k: v for k, v in c.items() if k not in ("superiority", "never_say")}).replace("superiority", ""))
gp_ = rp["combined"]["contrasts"]["G+M - O0"]["primary"]["guard"]
check("guard is within the kitchen for a kitchen result and across kitchens for the pooled one (24 shared families: K 24 / 24, "
      "pooled 24 distinct, per-kitchen sum 48 disclosed)",
      c["each_kitchen"]["per_kitchen"]["FloorPlan15"]["threshold_statement_permitted"] and gp_["K"] == 24 and gp_["K_sum_of_kitchens"] == 48
      and gp_["passed"] and rp["kitchens"]["FloorPlan15"]["contrasts"]["G+M - O0"]["primary"]["guard"]["K"] == 24, gp_)
rm = run_big(+1, -1)
cm = rm["combined"]["g_m_vs_o0_criteria"]["primary"]
check("FloorPlan15 met, FloorPlan17 inferior: each-kitchen 'not met' (a conjunction) and the average is reported separately",
      cm["each_kitchen"]["per_kitchen"]["FloorPlan15"]["met"] is True and cm["each_kitchen"]["per_kitchen"]["FloorPlan17"]["met"] is False
      and cm["each_kitchen"]["status"] == "not met" and cm["average"]["status"] in ("met", "not met") and cm["average"]["scope"] == "two-kitchen average")
check("inferiority threshold statement 'upper bound < -5 pp' is made under the guard (FloorPlan17) and not for FloorPlan15",
      cm["each_kitchen"]["per_kitchen"]["FloorPlan17"]["upper_bound_below_minus_5pp"] is True
      and cm["each_kitchen"]["per_kitchen"]["FloorPlan15"]["upper_bound_below_minus_5pp"] is False)
others = [rm["kitchens"]["FloorPlan15"]["contrasts"][KA.ckey(a, b)]["primary"] for a, b in KA.CONTRASTS[1:]]
check("the other three contrasts are estimation-only: estimate + interval + guard, no criterion / threshold keys",
      all(o["available"] and "met" not in o and "upper_bound_below_minus_5pp" not in o and "status" not in o for o in others)
      and set(rm["combined"]["g_m_vs_o0_criteria"]["primary"]) >= {"average", "each_kitchen"} and "G+M - G0" not in json.dumps(
          {k: v for k, v in rm["combined"]["g_m_vs_o0_criteria"].items() if k != "infrastructure_sensitive_arms"}))
check("no Holm / multiplicity adjustment anywhere (plan says none registered); intervals labelled not assured",
      "p_holm" not in json.dumps(rm) and "none registered" in rm["analysis_plan_k23"]["multiplicity"]
      and rm["kitchens"]["FloorPlan15"]["contrasts"]["G+M - O0"]["primary"]["interval_label"] == KA.INTERVAL_LABEL)
ri = run_big(+1, +1, partial=True)
ci = ri["combined"]["g_m_vs_o0_criteria"]["primary"]
check("incomplete study (--partial): estimates and intervals reported, NO criterion evaluated (not evaluated, met None)",
      not ri["complete"] and ci["average"]["status"] == "not evaluated" and ci["average"]["met"] is None
      and ci["each_kitchen"]["status"] == "not evaluated" and ri["combined"]["contrasts"]["G+M - O0"]["primary"]["interval"][0] is not None)

# ------------------------------------------------------------------------------------------------ 6. presentations, languages, infra, INVALID
rt = new_root()
T15l = {a: dict(v) for a, v in T15.items()}
T15l["G+M"]["i4"] = {"en": 1, "fa": 0}                      # EN / FA disagree -> item difference 0.5 against O0 (1, 1)
T15l["O0"]["i2"] = {"en": 0, "fa": None}                    # FA missing in O0 -> i2 uses EN only
a_ = write_store(rt, NS15, mk_rows(NS15, T15l, FAM))
b_ = write_store(rt, NS17, mk_rows(NS17, T17, FAM))
rl = KA.analyse_k23(stores_of(rt, a_, b_), B_cost=B_COST)
r = rl["kitchens"]["FloorPlan15"]["contrasts"]["G+M - O0"]["primary"]
check("languages: item = mean of its paired EN / FA differences; a presentation only if both arms have it",
      r["S"] == [1.0, -1.0, -0.5] and r["n_pres"] == 7 and r["missing_pres"] == 1, r)
t = rl["kitchens"]["FloorPlan15"]["arm_tables"]["G+M"]
check("descriptive rates per kitchen: EN / FA and L1 for every arm incl. P0 / P+M",
      close(t["lang"]["en"]["sr"], 0.5) and close(t["lang"]["fa"]["sr"], 0.25) and "P0" in rl["kitchens"]["FloorPlan15"]["arm_tables"]
      and "P+M" in rl["kitchens"]["FloorPlan17"]["arm_tables"] and t["level"]["L1"] is not None)
rt = new_root()
rows15 = mk_rows(NS15, T15, FAM)
for x in rows15:
    if x["arm"] == "O0" and x["item"] == "i3":                # O0 wins i3 in the toy; make it an infra failure (SR 0, stays in the denominator)
        x.update(sr=0.0, strict_sr=0.0, infra=True, run_class="infra_planner", failure_class="plan_infra")
    if x["arm"] == "G+M" and x["item"] == "i1" and x["lang"] == "fa":
        x["invalid_episode"] = True                           # INVALID -> (i1, fa) dropped from EVERY arm
a_ = write_store(rt, NS15, rows15)
b_ = write_store(rt, NS17, mk_rows(NS17, T17, FAM))
rf_ = KA.analyse_k23(stores_of(rt, a_, b_), B_cost=B_COST)
k15 = rf_["kitchens"]["FloorPlan15"]
pr, ip = k15["contrasts"]["G+M - O0"]["primary"], k15["contrasts"]["G+M - O0"]["infra_paired"]
check("infra = SR 0 in the primary analysis (A21 A.4): i3 difference becomes 0; the infra-paired sensitivity drops it from both arms",
      pr["S"] == [1.0, 0.0, 0.0] and ip["dropped_infra"] == 2 and ip["n"] == [2, 1] and "f2" not in ip["fams"], (pr, ip))
check("INVALID presentation dropped from every arm (section 13): (i1, fa) gone in G+M - O0 AND G+M - G0, reported by count",
      k15["invalid_presentations_dropped"] == 1 and pr["n_pres"] == 7 and k15["contrasts"]["G+M - G0"]["primary"]["n_pres"] == 7)
check("infrastructure-sensitive flag (> 5 % infra cells) per kitchen and the infra-paired criteria shown side by side",
      k15["infra_by_arm"]["O0"]["infrastructure_sensitive"] and "O0" in rf_["combined"]["g_m_vs_o0_criteria"]["infrastructure_sensitive_arms"]["FloorPlan15"]
      and "average" in rf_["combined"]["g_m_vs_o0_criteria"]["infra_paired_sensitivity"])
check("distinct-units-only sensitivity and dependence table present per kitchen and pooled",
      k15["contrasts"]["G+M - O0"]["distinct_units_only"]["available"] and rf_["combined"]["contrasts"]["G+M - O0"]["distinct_units_only"]["available"]
      and k15["dependence"]["by_arm"]["G+M"]["distinct_units"] == 8)
_ix = KA.A.build_index(rows15)
_dr = KA.A.invalid_presentations(_ix["S"])
check("dependence rows per contrast equal analyze._dep_row (the Kitchen 1 A21 D fields) wherever that is defined",
      all(k15["dependence"]["contrasts"][KA.ckey(a, b)] == KA.A._dep_row(_ix["S"], _ix["fam"], a, b, _dr) for a, b in KA.CONTRASTS))
rows15b = mk_rows(NS15, T15, FAM)
for x in rows15b:
    if x["arm"] == "G0" and x["item"] == "i2":
        x["excluded_reset"] = True
rt = new_root()
ra36 = KA.analyse_k23(stores_of(rt, write_store(rt, NS15, rows15b), write_store(rt, NS17, mk_rows(NS17, T17, FAM))), B_cost=B_COST)
check("A36 flags are disclosed, not applied (A21 rule stays: the flagged G0 cells still count)",
      ra36["kitchens"]["FloorPlan15"]["a36_disclosure"]["excluded_reset_cells_by_arm"] == {"G0": 2}
      and ra36["kitchens"]["FloorPlan15"]["contrasts"]["G+M - G0"]["primary"]["n_pres"] == 8)

# ------------------------------------------------------------------------------------------------ 7. common items, unavailable kitchen contrast
check("common items: the toy kitchens differ (4 vs 5 items, 2 common) -> intersection reported descriptively, no criterion",
      res["combined"]["common_items"]["n_common"] == 2 and not res["combined"]["common_items"]["identical"]
      and "G+M - O0" in res["combined"]["common_items"]["contrasts"] and '"met":' not in json.dumps(res["combined"]["common_items"])
      and '"upper_bound_below_minus_5pp":' not in json.dumps(res["combined"]["common_items"]))
check("common items: identical item sets -> 'identical', no duplicate analysis", rp["combined"]["common_items"]["identical"] is True
      and "contrasts" not in rp["combined"]["common_items"])
rt = new_root()
T17one = {a: {"i1": v["i1"], "i2x": v["i1"]} for a, v in T17.items()}
d_one = write_store(rt, NS17, mk_rows(NS17, T17one, dict(FAM, i2x="f1")))
ro = KA.analyse_k23(stores_of(rt, write_store(rt, NS15, mk_rows(NS15, T15, FAM)), d_one), B_cost=B_COST)
po = ro["combined"]["contrasts"]["G+M - O0"]["primary"]
check("a kitchen whose contrast has < 2 families: kitchen result unavailable AND the pooled summary unavailable (never one-kitchen)",
      not ro["kitchens"]["FloorPlan17"]["contrasts"]["G+M - O0"]["primary"]["available"] and not po["available"]
      and "never a one-kitchen result" in po["reason"]
      and ro["combined"]["g_m_vs_o0_criteria"]["primary"]["average"]["status"] == "not met")

# ------------------------------------------------------------------------------------------------ 8. refusals
rt = new_root()
g15 = write_store(rt, NS15, mk_rows(NS15, T15, FAM))
check("refuse: Kitchen 3 store missing (load_store and run)", raises(lambda: KA.load_store(os.path.join(C.paths(rt)["matrix_docs"], NS17), "FloorPlan17", root=rt),
                                                                    KA.K23AnalysisError, "store missing")
      and raises(lambda: KA.run(root=rt, allow_test=True), KA.K23AnalysisError, "store missing"))
check("refuse: analyse_k23 with one kitchen only (no silent one-kitchen summary)",
      raises(lambda: KA.analyse_k23({"FloorPlan15": KA.load_store(g15, "FloorPlan15", root=rt)}), KA.K23AnalysisError, "both replication kitchens"))
rt = new_root()
mix = mk_rows(NS15, T15, FAM) + [row(NS17, "G0", "i9", "en", 1, "f9")]
dm = write_store(rt, NS15, mix)
check("refuse: a FloorPlan15 store holding a FloorPlan17 row (mixes kitchens)", raises(lambda: KA.load_store(dm, "FloorPlan15", root=rt),
                                                                                    KA.K23AnalysisError, "mixes kitchens"))
rt = new_root()
k1row = mk_rows(NS15, T15, FAM) + [row("test-" + "v1", "G0", "i9", "en", 1, "f9")]
check("refuse: a store holding a Kitchen 1 (test namespace) row", raises(lambda: KA.load_store(write_store(rt, NS15, k1row), "FloorPlan15", root=rt),
                                                                       KA.K23AnalysisError, "mixes kitchens"))
rt = new_root()
scn = mk_rows(NS15, T15, FAM)
scn[0]["scene"] = "FloorPlan1"
check("refuse: a row whose scene field names another kitchen", raises(lambda: KA.load_store(write_store(rt, NS15, scn), "FloorPlan15", root=rt),
                                                                    KA.K23AnalysisError, "mixes kitchens"))
rt = new_root()
check("refuse: seal written for another namespace", raises(lambda: KA.load_store(write_store(rt, NS15, mk_rows(NS15, T15, FAM), seal_ns=NS17), "FloorPlan15",
                                                                                 root=rt), KA.K23AnalysisError, "seal"))
rt = new_root()
check("refuse: a store directory that is not the namespace's storage dir", raises(lambda: KA.load_store(
    write_store(rt, NS15, mk_rows(NS15, T15, FAM), dirname="somewhere"), "FloorPlan15", root=rt), KA.K23AnalysisError, "storage directory"))
rt = new_root()
dd = write_store(rt, NS15, mk_rows(NS15, T15, FAM))
with open(os.path.join(dd, "score.jsonl"), "a", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(row(NS15, "G0", "i9", "en", 1, "f9")) + "\n")
check("refuse: score.jsonl edited after sealing", raises(lambda: KA.load_store(dd, "FloorPlan15", root=rt), KA.A.AnalysisError, "seal"))
rt = new_root()
di = write_store(rt, NS15, mk_rows(NS15, T15, FAM), complete=False)
check("refuse: incomplete seal without --partial; accepted with it", raises(lambda: KA.load_store(di, "FloorPlan15", root=rt), KA.A.AnalysisError,
                                                                            "not complete")
      and KA.load_store(di, "FloorPlan15", root=rt, allow_partial=True)["seal"]["complete"] is False)
rt = new_root()
plan_bad = [{"key": "x", "namespace": NS17, "arm": "G0", "pres": C.pres_id("i1", "en"), "r": 0, "block": "main", "cost_usd": 0.0}]
check("refuse: a plan journal row of the other kitchen beside the score journal", raises(lambda: KA.load_store(
    write_store(rt, NS15, mk_rows(NS15, T15, FAM), plan_rows=plan_bad), "FloorPlan15", root=rt), KA.K23AnalysisError, "mixes kitchens"))
rt = new_root()
famx = dict(FAM, i1="fX")
check("refuse: the same item in two different families across kitchens",
      raises(lambda: KA.analyse_k23(stores_of(rt, write_store(rt, NS15, mk_rows(NS15, T15, FAM)), write_store(rt, NS17, mk_rows(NS17, T17, famx))),
                                    B_cost=B_COST), KA.K23AnalysisError, "family"))
rt = new_root()
xr = row(NS15, "X0", "i1", "en", 0, "f1")
xr.update(sr=None, strict_sr=None, plan_only=True, run_class="plan_only")
xrow = mk_rows(NS15, T15, FAM) + [xr]
check("refuse: an arm outside the K2/K3 design (e.g. a plan-only Codex row) in a kitchen store",
      raises(lambda: KA.analyse_k23(stores_of(rt, write_store(rt, NS15, xrow), write_store(rt, NS17, mk_rows(NS17, T17, FAM))), B_cost=B_COST),
             KA.K23AnalysisError, "not in the K2/K3 design"))
check("refuse: FloorPlan1 (or any non-replication scene) is never a kitchen here", raises(lambda: KA.check_store_path(TMP, "FloorPlan1"),
                                                                                       KA.K23AnalysisError, "not a replication kitchen"))
check("refuse: the same store given for both kitchens", raises(lambda: KA.run(g15, g15, root=os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(g15)))), allow_test=True), KA.K23AnalysisError))
check("CLI: missing stores -> REFUSED, exit 2", KA.main(["--k2", os.path.join(TMP, "nope", NS15), "--k3", os.path.join(TMP, "nope", NS17)]) == 2)

# ------------------------------------------------------------------------------------------------ 9. Kitchen 1 is never read
rt = new_root()
md = C.paths(rt)["matrix_docs"]
k1dir = write_store(rt, "test-" + "v1", mk_rows("test-" + "v1", T15, FAM), plan_rows=[{"key": "k1", "namespace": "test-" + "v1", "arm": "G0"}])
s15 = write_store(rt, NS15, mk_rows(NS15, T15, FAM))
s17 = write_store(rt, NS17, mk_rows(NS17, T17, FAM))
opened = []
_open, _listdir = builtins.open, os.listdir


def rec_open(file, *a, **kw):
    if isinstance(file, (str, bytes, os.PathLike)):
        opened.append(os.path.abspath(os.fsdecode(file)))
    return _open(file, *a, **kw)


def rec_listdir(p="."):
    opened.append(os.path.abspath(os.fsdecode(p)))
    return _listdir(p)


patches = {(C, "guard_items"): lambda items, ns, **kw: {"n": len(items), "n_test": 0, "namespace": ns},
           (C, "load_manifest"): lambda root=None, **kw: {"families": [{"family": f, "items": sorted(i for i in FAM if FAM[i] == f)}
                                                                        for f in sorted(set(FAM.values()))]},
           (K, "a29_roster"): lambda ns, root=None: sorted({i for i in (T15["G0"] if ns == NS15 else T17["G0"])})}
saved = {k: getattr(*k) for k in patches}
outd = os.path.join(TMP, "out_k1check")
try:
    for (mod, name), fn in patches.items():
        setattr(mod, name, fn)
    builtins.open, os.listdir = rec_open, rec_listdir
    res_run, (jp, mp) = KA.run(root=rt, allow_test=True, B_cost=B_COST, outdir=outd)
    refused_k1 = raises(lambda: KA.run(k1dir, s17, root=rt, allow_test=True, outdir=outd), KA.K23AnalysisError, "Kitchen 1 store")
    refused_out = raises(lambda: KA.write_outputs(res_run, os.path.join(k1dir, "sub"), rt), KA.K23AnalysisError, "Kitchen 1 store")
finally:
    builtins.open, os.listdir = _open, _listdir
    for k, v in saved.items():
        setattr(*k, v)
real_k1 = [d for d in KA.kitchen1_store_dirs(None)]
touched = [p for p in opened if any(KA._inside(p, d) for d in [k1dir] + real_k1)]
check("Kitchen 1 never read: the full run (default store paths) opened no file under a Kitchen 1 store (fake K1 store in the root + the repo's)",
      res_run["kitchens"]["FloorPlan15"]["namespace"] == NS15 and not touched and any(KA._inside(p, s15) for p in opened), touched[:3])
check("Kitchen 1 never read: a Kitchen 1 store given as a kitchen is refused BEFORE any file in it is opened; outputs never go there",
      refused_k1 and refused_out and not touched)
src = open(os.path.join(MATRIX, "k23_analyze.py"), encoding="utf-8").read()
tree = ast.parse(src)
attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
doc_ids = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
           and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids]
check("static: k23_analyze never calls the Kitchen 1 entry points (run_sealed / analyse / journal_path / completeness_problems) and names no "
      "Kitchen 1 namespace, outcome or states file in code", not ({"run_sealed", "analyse", "journal_path", "completeness_problems", "run_synthetic"} & attrs)
      and not any(("test-" + "v1") == c or "outcome.sealed" in c or c == "states" for c in consts))

# ------------------------------------------------------------------------------------------------ 10. outputs, determinism, cost
check("outputs: analysis_k23.json + analysis_k23.md written, per kitchen + combined, schema and plan sha",
      os.path.basename(jp) == "analysis_k23.json" and os.path.basename(mp) == "analysis_k23.md"
      and set(json.load(open(jp, encoding="utf-8"))["kitchens"]) == {"FloorPlan15", "FloorPlan17"}
      and json.load(open(jp, encoding="utf-8"))["analysis_plan_k23_sha"] == KA.analysis_plan_sha())
md_txt = open(mp, encoding="utf-8").read()
check("markdown names the exploratory status, the four contrasts, both kitchens and the two-kitchen average",
      "Exploratory" in md_txt and all(KA.ckey(a, b) in md_txt for a, b in KA.CONTRASTS) and "two-kitchen average" in md_txt
      and "FloorPlan17" in md_txt and "not assured" in md_txt)
rt = new_root()
x15, x17 = write_store(rt, NS15, mk_rows(NS15, T15, FAM)), write_store(rt, NS17, mk_rows(NS17, T17, FAM))
r1 = KA.analyse_k23(stores_of(rt, x15, x17), B_cost=B_COST)
r2 = KA.analyse_k23(stores_of(rt, x15, x17), B_cost=B_COST)
o1, o2 = os.path.join(TMP, "det1"), os.path.join(TMP, "det2")
KA.write_outputs(r1, o1)
KA.write_outputs(r2, o2)
check("determinism (fixed seeds): two runs give byte-identical analysis_k23.json and .md",
      json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
      and all(C.sha256_file(os.path.join(o1, f)) == C.sha256_file(os.path.join(o2, f)) for f in (KA.OUT_JSON, KA.OUT_MD)))
rt = new_root()
plan15 = []
for arm in ("G0", "G+M", "O0", "O+M"):
    for it in T15[arm]:
        for lg in LANGS:
            pr_ = {"key": f"{NS15}|plan|{arm}|{C.pres_id(it, lg)}|r0", "namespace": NS15, "arm": arm, "pres": C.pres_id(it, lg), "r": 0, "block": "main",
                   "cost_usd": 0.01 if arm[0] == "O" else 0.0, "tokens_in": 100, "tokens_out": 10, "calls": 1, "cached_calls": 0, "wall_s": 1.0}
            if arm == "O+M" and it == "i1" and lg == "en":
                pr_["billing_tier"] = "batch"
            plan15.append(pr_)
rc_ = KA.analyse_k23(stores_of(rt, write_store(rt, NS15, mk_rows(NS15, T15, FAM), plan_rows=plan15), write_store(rt, NS17, mk_rows(NS17, T17, FAM))),
                     B_cost=B_COST)
t4 = rc_["kitchens"]["FloorPlan15"]["cost"]["a29_item4"]
check("cost per kitchen (A29 item 4): billed USD, tokens, calls / cell; standard-price doubles only the recorded batch charge",
      t4["available"] and close(t4["by_arm"]["O0"]["billed_usd"], 0.08) and close(t4["by_arm"]["O+M"]["standard_price_usd"], 0.09)
      and t4["by_arm"]["O+M"]["rows_with_batch_tier"] == 1 and t4["by_arm"]["G+M"]["calls_per_cell"] == 1.0
      and not rc_["kitchens"]["FloorPlan17"]["cost"]["a29_item4"]["available"])
a24 = rc_["kitchens"]["FloorPlan15"]["cost"]["a24_2"]
check("A24.2 outcomes per kitchen (R, C with CIs) computed; the A24.2 statement is never permitted for Kitchens 2/3",
      (a24.get("R_C") or {}).get("available") and a24["claim"]["statement_permitted"] is False
      and rc_["kitchens"]["FloorPlan17"]["cost"]["a24_2"]["claim"]["statement_permitted"] is False
      and "not applied" in a24["unmeasured_rule"])

# ------------------------------------------------------------------------------------------------ 11. Astra K23-ANALYZE-r1 change 4: --partial = descriptive only
def patched_with(pp, fn):
    """run() with guard / manifest / roster stubs (synthetic roots have no split manifest or freeze); always restored."""
    saved_ = {k: getattr(*k) for k in pp}
    try:
        for (mod, name), f in pp.items():
            setattr(mod, name, f)
        return fn()
    finally:
        for k, v in saved_.items():
            setattr(*k, v)


def patched(fn):
    return patched_with(patches, fn)


def no_criterion(cr):
    return (cr["average"]["status"] == "not evaluated" and cr["average"]["met"] is None and cr["each_kitchen"]["status"] == "not evaluated"
            and cr["each_kitchen"]["met"] is None and cr["average"]["upper_bound_below_minus_5pp"] is None
            and all(q["status"] == "not evaluated" and q["upper_bound_below_minus_5pp"] is None for q in cr["each_kitchen"]["per_kitchen"].values())
            and not cr["thesis_sentences"])


rt = new_root()
tb15, fb15 = big_table(24, +1, 1)
tb17, fb17 = big_table(24, +1, 2)
q15, q17 = write_store(rt, NS15, mk_rows(NS15, tb15, fb15)), write_store(rt, NS17, mk_rows(NS17, tb17, fb17))
full = KA.analyse_k23(stores_of(rt, q15, q17), B_cost=B_COST)
part = KA.analyse_k23(stores_of(rt, q15, q17), B_cost=B_COST, partial=True)
check("change 4: --partial on COMPLETE seals (a case whose criteria are otherwise met): no criterion, no threshold statement, no sentence",
      full["combined"]["g_m_vs_o0_criteria"]["primary"]["average"]["met"] is True and part["complete"] is True
      and part["criteria_evaluated"] is False and part["partial_flag"] is True
      and no_criterion(part["combined"]["g_m_vs_o0_criteria"]["primary"]) and no_criterion(part["combined"]["g_m_vs_o0_criteria"]["infra_paired_sensitivity"])
      and KA.DESCRIPTIVE_PARTIAL in part["descriptive_only"]["reasons"] and part["descriptive_only"]["active"])
check("change 4: --partial keeps every estimate and interval identical (per kitchen and pooled, all contrasts and views)",
      all(json.dumps(full["kitchens"][s]["contrasts"], sort_keys=True) == json.dumps(part["kitchens"][s]["contrasts"], sort_keys=True) for s in KA.SCENES)
      and json.dumps(full["combined"]["contrasts"], sort_keys=True) == json.dumps(part["combined"]["contrasts"], sort_keys=True))
fam_big = dict(fb15, **fb17)
patches_big = dict(patches)
patches_big[(C, "load_manifest")] = lambda root=None, **kw: {"families": [{"family": f, "items": sorted(i for i in fam_big if fam_big[i] == f)}
                                                                           for f in sorted(set(fam_big.values()))]}
patches_big[(K, "a29_roster")] = lambda ns, root=None: sorted(tb15["G0"] if ns == NS15 else tb17["G0"])
rp4 = patched_with(patches_big, lambda: KA.run(q15, q17, root=rt, allow_test=True, allow_partial=True, B_cost=B_COST,
                                               outdir=os.path.join(TMP, "out_partial")))[0]
md4 = open(os.path.join(TMP, "out_partial", KA.OUT_MD), encoding="utf-8").read()
check("change 4: the run() / CLI path forwards --partial on complete seals: criteria not evaluated; the md says 'Descriptive only' with the reason",
      rp4["meta"]["complete"] is True and rp4["criteria_evaluated"] is False and no_criterion(rp4["combined"]["g_m_vs_o0_criteria"]["primary"])
      and "Descriptive only" in md4 and KA.DESCRIPTIVE_PARTIAL in md4 and KA.AVERAGE_SENTENCE not in md4)
rt_i = new_root()
pi = KA.analyse_k23(stores_of(rt_i, write_store(rt_i, NS15, mk_rows(NS15, tb15, fb15), complete=False), write_store(rt_i, NS17, mk_rows(NS17, tb17, fb17)),
                              partial=True), B_cost=B_COST, partial=True)
check("change 4: --partial with an incomplete seal names both reasons; still no criterion",
      pi["complete"] is False and set(pi["descriptive_only"]["reasons"]) == {KA.DESCRIPTIVE_PARTIAL, KA.DESCRIPTIVE_INCOMPLETE}
      and no_criterion(pi["combined"]["g_m_vs_o0_criteria"]["primary"]))


# ------------------------------------------------------------------------------------------------ 12. change 5: a recorded "kitchen not run" (A29 item 6)
def put_not_run(root_, scene, rec):
    p = KA.not_run_path(scene, root_)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f)
    return p


REASON = "A29 item 6 checkpoint after FloorPlan15: the 100-presentation projection exceeds 80 USD; planning paused (synthetic test record)"
good17 = KA.make_not_run_record("FloorPlan17", reason=REASON, date="2026-10-25", decided_by="Arash (synthetic selftest)")
check("change 5: a well-formed record validates; the helper writes schema, scene, namespace, reason, date, decided_by, basis and a self-sha",
      KA.not_run_problems(good17, "FloorPlan17") == [] and set(good17) == set(KA.NOT_RUN_KEYS) and good17["namespace"] == NS17
      and good17["sha256"] == C.self_sha(good17))
bad_cases = {"edited after writing": dict(good17, reason=REASON + " (edited)"),
             "wrong kitchen": KA.make_not_run_record("FloorPlan15", reason=REASON, date="2026-10-25", decided_by="Arash"),
             "short reason": KA.make_not_run_record("FloorPlan17", reason="budget", date="2026-10-25", decided_by="Arash"),
             "no decider": KA.make_not_run_record("FloorPlan17", reason=REASON, date="2026-10-25", decided_by=" "),
             "bad date": KA.make_not_run_record("FloorPlan17", reason=REASON, date="yesterday", decided_by="Arash"),
             "basis without A29": KA.make_not_run_record("FloorPlan17", reason=REASON, date="2026-10-25", decided_by="Arash", basis="budget"),
             "extra key": (lambda r: dict(r, sha256=C.self_sha(r)))(dict({k: v for k, v in good17.items() if k != "sha256"}, note="x"))}
check("change 5: malformed records are rejected (edited, wrong kitchen, short reason, no decider, bad date, basis without A29, extra key)",
      all(KA.not_run_problems(r, "FloorPlan17") for r in bad_cases.values()), [k for k, r in bad_cases.items() if not KA.not_run_problems(r, "FloorPlan17")])
rt5 = new_root()
n15 = write_store(rt5, NS15, mk_rows(NS15, T15, FAM))
check("change 5: an UNEXPLAINED missing store is still refused (no record)",
      raises(lambda: patched(lambda: KA.run(root=rt5, allow_test=True, B_cost=B_COST, outdir=os.path.join(TMP, "o5a"))), KA.K23AnalysisError,
             "no recorded kitchen-not-run decision"))
put_not_run(rt5, "FloorPlan17", good17)
opened.clear()
builtins.open, os.listdir = rec_open, rec_listdir
try:
    r5, (j5, m5) = patched(lambda: KA.run(root=rt5, allow_test=True, B_cost=B_COST, outdir=os.path.join(TMP, "o5b")))
finally:
    builtins.open, os.listdir = _open, _listdir
md5 = open(m5, encoding="utf-8").read()
check("change 5: with the record, the completed kitchen's estimates only: FloorPlan15 reported, FloorPlan17 recorded as not run (reason, "
      "date, decider, file)", list(r5["kitchens"]) == ["FloorPlan15"] and r5["kitchens_not_run"]["FloorPlan17"]["reason"] == REASON
      and r5["kitchens_not_run"]["FloorPlan17"]["decided_by"].startswith("Arash") and r5["kitchens_not_run"]["FloorPlan17"]["file"].endswith(
          "docs/final-test/k23/kitchen_not_run_FloorPlan17.json"))
check("change 5: no pooled result, no replication criteria, no common-item analysis; the study is incomplete and criteria are not evaluated",
      r5["combined"]["available"] is False and r5["combined"]["pooled"] is None and r5["combined"]["g_m_vs_o0_criteria"] is None
      and r5["combined"]["common_items"] is None and "contrasts" not in r5["combined"] and r5["complete"] is False
      and r5["criteria_evaluated"] is False and "kitchen not run" in r5["combined"]["reason"])
check("change 5: the completed kitchen's estimates equal those of a two-kitchen run on the same store (nothing pooled into them)",
      json.dumps(r5["kitchens"]["FloorPlan15"]["contrasts"], sort_keys=True) == json.dumps(res["kitchens"]["FloorPlan15"]["contrasts"], sort_keys=True))
check("change 5: the md states it plainly (Kitchen not run, the reason, NO two-kitchen summary, NO replication criteria) and shows no pooled row",
      "**Kitchen not run: FloorPlan17**" in md5 and REASON in md5 and "NO two-kitchen summary and NO replication criteria" in md5
      and "two-kitchen average" not in md5 and "exploratory criteria:" not in md5 and "## FloorPlan17:" not in md5)
check("change 5: the not-run path read no Kitchen 1 file (open() recorded)",
      not [p for p in opened if any(KA._inside(p, d) for d in KA.kitchen1_store_dirs(rt5))] and any(KA._inside(p, n15) for p in opened))
put_not_run(rt5, "FloorPlan17", bad_cases["edited after writing"])
check("change 5: a record file that does not validate is refused, never ignored",
      raises(lambda: patched(lambda: KA.run(root=rt5, allow_test=True, B_cost=B_COST, outdir=os.path.join(TMP, "o5c"))), KA.K23AnalysisError,
             "not a valid kitchen-not-run record"))
put_not_run(rt5, "FloorPlan17", good17)
write_store(rt5, NS17, mk_rows(NS17, T17, FAM))
check("change 5: a store AND a not-run record for the same kitchen are contradictory: refused",
      raises(lambda: patched(lambda: KA.run(root=rt5, allow_test=True, B_cost=B_COST, outdir=os.path.join(TMP, "o5d"))), KA.K23AnalysisError,
             "contradictory"))
rt6 = new_root()
put_not_run(rt6, "FloorPlan15", KA.make_not_run_record("FloorPlan15", reason=REASON, date="2026-10-25", decided_by="Arash"))
put_not_run(rt6, "FloorPlan17", good17)
check("change 5: both kitchens recorded as not run: nothing to analyse, refused",
      raises(lambda: patched(lambda: KA.run(root=rt6, allow_test=True, B_cost=B_COST, outdir=os.path.join(TMP, "o5e"))), KA.K23AnalysisError,
             "nothing to analyse"))
st15 = {"FloorPlan15": KA.load_store(n15, "FloorPlan15", root=rt5)}
st17 = KA.load_store(os.path.join(C.paths(rt5)["matrix_docs"], NS17), "FloorPlan17", root=rt5)
check("change 5 (analyse_k23): one kitchen without a record refused; an invalid record refused; a store plus a record refused; valid -> no pooled",
      raises(lambda: KA.analyse_k23(st15, B_cost=B_COST), KA.K23AnalysisError, "kitchen-not-run")
      and raises(lambda: KA.analyse_k23(st15, B_cost=B_COST, not_run={"FloorPlan17": bad_cases["short reason"]}), KA.K23AnalysisError, "invalid")
      and raises(lambda: KA.analyse_k23(dict(st15, FloorPlan17=st17), B_cost=B_COST, not_run={"FloorPlan17": good17}), KA.K23AnalysisError,
                 "cannot also be recorded")
      and KA.analyse_k23(st15, B_cost=B_COST, not_run={"FloorPlan17": good17})["combined"]["available"] is False)
check("analysis plan records both changes and the review (version k23-analysis-2)",
      KA.ANALYSIS_PLAN_K23["version"] == "k23-analysis-2" and "K23-ANALYZE-r1" in KA.ANALYSIS_PLAN_K23["review"]
      and "change 4" in KA.ANALYSIS_PLAN_K23["partial"] and "change 5" in KA.ANALYSIS_PLAN_K23["kitchen_not_run"])

shutil.rmtree(TMP, ignore_errors=True)
tot = N["pass"] + N["fail"]
print(f"\n_k23_analyze_selftest: {N['pass']}/{tot} OK" + ("" if not N["fail"] else f" -- {N['fail']} FAILED"))
sys.exit(0 if not N["fail"] else 1)
