"""Selftest for sim/final_test_h1.py (stage G-H1-validate, A22 item 4 / A22.1). Offline, Windows python, numpy + scipy. No outcome data.

    python sim/_ftm_h1_selftest.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import numpy as np                                                          # noqa: E402

import final_test_h1 as H                                                   # noqa: E402
import final_test_power as P                                                # noqa: E402

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
F, N = len(sizes), sum(sizes)
check("real family sizes: 42 families, 125 items, 17 singletons, 10 of six", F == 42 and N == 125 and sizes.count(1) == 17 and sizes.count(6) == 10)
cs = H.configs()
check("frozen grid: 146 configurations (80 at -5 pp, 40 at 0, 26 interior), unique ids", len(cs) == 146 and len({c["id"] for c in cs}) == 146
      and sum(c["id"].startswith("N-") for c in cs) == 80 and sum(c["id"].startswith("Z-") for c in cs) == 40)
check("protocol sha is stable", H.protocol_sha() == H.protocol_sha() and len(H.protocol_sha()) == 64)
check("null sets (Astra 4): -5 pp protects ni / inf / sup; 0 protects inf / sup / neg; -10 pp ni / sup; +5 pp inf / neg",
      H.null_sets(-0.05) == {"ni": True, "inf": True, "sup": True, "neg": False} and H.null_sets(0.0) == {"ni": False, "inf": True, "sup": True, "neg": True}
      and H.null_sets(-0.10) == {"ni": True, "inf": False, "sup": True, "neg": False} and H.null_sets(0.05) == {"ni": False, "inf": True, "sup": False, "neg": True})
check("power set = 12 DGP-A Delta 0 configurations present in the grid", len(H.POWER_SET) == 12 and set(H.POWER_SET) <= {c["id"] for c in cs})
check("bootstrap-t ranks for B 9,999 are 250 and 9,750 (one-based)", H.bt_ranks(9999) == (250, 9750))

rng = np.random.default_rng(7)


def dataset(cfg, seed):
    r = np.random.default_rng(seed)
    en, fa = (H.gen_A if cfg["dgp"] == "A" else H.gen_B)(r, 1, sizes, cfg)
    en, fa = H.apply_missing(r, en, fa, cfg)
    S, n, K, disc = H.summarise(en, fa, sizes)
    keep = n[0] > 0
    return S[0][keep], n[0][keep], int(K[0]), en[0], fa[0]


# ---- C0 reproduces the locked rule exactly
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import analyze as A                                                         # noqa: E402

pick = [c for c in cs if c["id"] in ("N-A-d0.05-r0.6", "N-A-d0.1-r0.9", "N-A-d0.3-r0.3", "Z-A-d0.2-r0.0", "I-p05-A-d0.4-r0.3", "N-B-p0.95-high", "I-m10-A-d0.4-r0.0")]
same = tot = 0
seen = set()
for c in pick:
    for s in range(6):
        S, n, K, *_ = dataset(c, 1000 + s)
        ref = A.fixed_sequence(P.h1_decision(list(S), list(n)))["wording"]
        mine = H.c0_claims(S, n)["_wording"]
        tot += 1
        same += ref == mine
        seen.add(mine.split(" (")[0].split(";")[0])
check(f"C0 == final_test_power.h1_decision + analyze.fixed_sequence on {tot} datasets (wordings seen: {sorted(seen)})", same == tot, f"{same}/{tot}")

# ---- bootstrap-t algebra vs brute force
S, n, K, *_ = dataset(next(c for c in cs if c["id"] == "N-A-d0.3-r0.3"), 11)
r = H.bootstrap_t(S, n)
W = H.multinomial_counts(len(n))
est = S.sum() / n.sum()
se = math.sqrt(len(n) / (len(n) - 1) * np.sum((S - n * est) ** 2)) / n.sum()
ok = abs(r["estimate"] - est) < 1e-12 and abs(r["se"] - se) < 1e-12
bad = 0
for b in range(40):
    idx = np.repeat(np.arange(len(n)), W[b].astype(int))           # expanded resample: repeated families are separate clusters
    Sb, nb = S[idx], n[idx]
    eb = Sb.sum() / nb.sum()
    seb = math.sqrt(len(n) / (len(n) - 1) * np.sum((Sb - nb * eb) ** 2)) / nb.sum()      # factor of the ORIGINAL F (Astra 2)
    tb = (eb - est) / seb
    Nb = W[b] @ n
    eb2 = (W[b] @ S) / Nb
    ss = W[b] @ (S * S) - 2 * eb2 * (W[b] @ (S * n)) + eb2 ** 2 * (W[b] @ (n * n))
    tb2 = (eb2 - est) / (math.sqrt(len(n) / (len(n) - 1) * ss) / Nb)
    bad += abs(tb - tb2) > 1e-8 * max(1, abs(tb))
check("bootstrap-t: estimate, se and 40 resampled t* equal the expanded-cluster brute force (m_f, not m_f^2)", ok and bad == 0, f"bad {bad}")
check("bootstrap-t interval is ordered and contains the estimate", r["defined"] and r["L"] <= r["estimate"] <= r["U"])
q_lo, q_hi = r["q"]
check("bootstrap-t bounds use the order statistics: L = est - se q_hi, U = est - se q_lo", abs(r["L"] - (est - se * q_hi)) < 1e-12 and abs(r["U"] - (est - se * q_lo)) < 1e-12)

z = H.bootstrap_t(np.zeros(42), np.asarray(sizes, float))
check("observed se = 0 (no between-family variation) -> undefined -> inconclusive for every wording",
      not z["defined"] and H.claims_from_bounds(z["L"], z["U"], z["defined"]) == {"ni": False, "sup": False, "inf": False, "lower": False}
      and "inconclusive" in H.h1_rule("C2", np.zeros(42), sizes)["wording"])
Sone = np.zeros(42); Sone[0] = -1.0
o = H.bootstrap_t(Sone, np.asarray(sizes, float))
check("one discordant family: resamples without it have se* = 0 -> t* = +inf share > 2.5 % -> L = -inf (no NI claim)",
      o["defined"] and o["L"] == -math.inf and not H.claims_from_bounds(o["L"], o["U"])["ni"])

# ---- CR2 vs explicit matrices
S, n, K, *_ = dataset(next(c for c in cs if c["id"] == "N-A-d0.2-r0.3"), 12)
nn = n.astype(int)
y = []
fam = []
rs = np.random.default_rng(3)
for f, (s_, m_) in enumerate(zip(S, nn)):
    v = rs.normal(size=m_); v = v - v.mean() + s_ / m_          # any item values with the family total S_f
    y += list(v); fam += [f] * m_
y, fam = np.asarray(y), np.asarray(fam)
Ntot = len(y)
X = np.ones((Ntot, 1))
Hm = X @ X.T / Ntot
e = y - y.mean()
Vsum = 0.0
g = []
Mres = np.eye(Ntot) - Hm
for f in range(len(nn)):
    ix = np.where(fam == f)[0]
    Hff = Hm[np.ix_(ix, ix)]
    w_, Q = np.linalg.eigh(np.eye(len(ix)) - Hff)
    Af = Q @ np.diag(w_ ** -0.5) @ Q.T
    Vsum += (np.ones(len(ix)) @ Af @ e[ix]) ** 2
    gf = np.zeros(Ntot)
    gf += (Mres[ix, :].T @ (Af @ np.ones(len(ix))))
    g.append(gf / Ntot)
se_exp = math.sqrt(Vsum) / Ntot
G = np.array([[gi @ gj for gj in g] for gi in g])
nu_exp = np.trace(G) ** 2 / np.sum(G * G)
c2 = H.cr2(S, n)
check("CR2 se equals the explicit (I - H_ff)^(-1/2) computation", abs(c2["se"] - se_exp) < 1e-10, f"{c2['se']} vs {se_exp}")
check("CR2 Satterthwaite nu equals the explicit g_f'g_h formula", abs(c2["nu"] - nu_exp) < 1e-8, f"{c2['nu']} vs {nu_exp}")
check("CR2 bounds are est -/+ t_nu(.975) se", abs(c2["U"] - c2["estimate"] - (c2["estimate"] - c2["L"])) < 1e-12 and c2["t_crit"] > 1.96)

# ---- wordings, guard
check("claims: L > -5 pp -> NI; L > 0 -> superior; U < -5 pp -> materially inferior; NI and U < 0 -> lower",
      H.claims_from_bounds(-0.04, 0.02) == {"ni": True, "sup": False, "inf": False, "lower": False}
      and H.claims_from_bounds(0.01, 0.1)["sup"] and H.claims_from_bounds(-0.2, -0.06)["inf"] and H.claims_from_bounds(-0.04, -0.01)["lower"]
      and not any(H.claims_from_bounds(-0.05, 0.1).values()))
check("guard failure -> inconclusive, never inferior", H.claims_from_bounds(-0.2, -0.06, guard_ok=False) == {"ni": False, "sup": False, "inf": False, "lower": False}
      and "guard failed" in H.h1_rule("C3", *dataset(cs[0], 5)[:2], K=3)["wording"])
check("wording strings match analyze/final_test_power", H.wording({"ni": True, "sup": True, "inf": False, "lower": False}) == "superior"
      and H.wording({"ni": False, "sup": False, "inf": False, "lower": False}) == "non-inferiority unresolved (inconclusive)")

# ---- summarise == analyze.item_diffs + by_family on data with missing presentations
cfgm = dict(next(c for c in cs if c["id"] == "N-Amcar-d0.2"))
cfgm["mcar"] = 0.15
ok_all = True
for s in range(5):
    S, n, K, en, fa = dataset(cfgm, 50 + s)
    rows = {}
    fam_of = {}
    item_f = np.repeat(np.arange(F), sizes)
    for i in range(N):
        fam_of[f"i{i}"] = f"f{item_f[i]:02d}"
        for l, x in (("en", en[i]), ("fa", fa[i])):
            if np.isnan(x):
                if (i + s) % 2:
                    rows[("O0", f"i{i}", l)] = {"sr": 1.0}        # one arm present, the other missing -> unused
                continue
            g_, o_ = (1.0, 0.0) if x > 0 else ((0.0, 1.0) if x < 0 else (1.0, 1.0))
            rows[("G+M", f"i{i}", l)] = {"sr": g_}
            rows[("O0", f"i{i}", l)] = {"sr": o_}
    diffs, info = A.item_diffs(rows, "G+M", "O0", set())
    fams, tot, sz = A.by_family(diffs, fam_of)
    Ka = A.discordant_families(rows, "G+M", "O0", set(), fam_of) if hasattr(A, "discordant_families") else K
    ok_all &= np.allclose(np.asarray(tot), S) and np.allclose(np.asarray(sz), n) and Ka == K
check("summarise (S_f, n_f, K) == analyze.item_diffs + by_family (+ discordant_families) under 15 % missing", ok_all)

# ---- DGP means (Monte Carlo, item-weighted estimate) and missing rate
for cid in ("N-A-d0.05-r0.6", "N-Atw-t1.0-d0.1-r0.6", "N-Ssize-d0.2-DL-0.08", "N-Sconc-small-q0", "N-Slang-1", "N-S2pt-pi0.2", "N-Saon-a.15b.10",
            "N-B-p0.95-high", "Z-B-p0.6-hetero", "I-p05-A-d0.3-r0.3", "I-m10-B-p0.8-medium", "Z-Ssize-DL0.03"):
    c = next(x for x in cs if x["id"] == cid)
    rr = np.random.default_rng(99)
    R = 4000
    en, fa = (H.gen_A if c["dgp"] == "A" else H.gen_B)(rr, R, sizes, c)
    Sx, nx, Kx, dx = H.summarise(en, fa, sizes)
    est = Sx.sum(1) / nx.sum(1)
    m, sd = est.mean(), est.std() / math.sqrt(R)
    check(f"DGP mean {cid}: {m:+.4f} vs Delta {c['D']:+.3f} (MC SE {sd:.4f}, mean discordance {dx.mean():.3f}, mean K {Kx.mean():.1f})", abs(m - c["D"]) < 4.5 * sd + 1e-4)
c = next(x for x in cs if x["id"] == "N-Amcar-d0.1")
en, fa = H.gen_A(np.random.default_rng(5), 3000, sizes, c)
en, fa = H.apply_missing(np.random.default_rng(6), en, fa, c)
mr = float(np.isnan(np.concatenate([en, fa])).mean())
check(f"MCAR 3 % per arm -> {mr:.4f} of paired presentations missing (expected {1 - .97 ** 2:.4f})", abs(mr - (1 - .97 ** 2)) < 0.003)

# ---- runner on a tiny R
row = H.run_config((next(x for x in cs if x["id"] == "N-A-d0.2-r0.3"), "screen", list(H.ORDER), 30))
check("run_config: counts for every candidate, guard pass recorded for C3 / C6, protocol sha attached",
      set(row["counts"]) == set(H.ORDER) and row["R"] == 30 and row["protocol_sha"] == H.protocol_sha()
      and all(0 <= row["counts"][k]["ni"] <= 30 for k in H.ORDER) and row["counts"]["C3"]["ni"] <= row["counts"]["C2"]["ni"]
      and row["counts"]["C6"]["ni"] <= row["counts"]["C5"]["ni"])
check("final_test_power.py is untouched by this stage (locked)", "h1_validation" not in open(os.path.join(HERE, "final_test_power.py"), encoding="utf-8").read())

print(f"{PASS}/{PASS + FAIL} passed")
sys.exit(1 if FAIL else 0)
