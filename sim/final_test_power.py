"""Final test -- the LOCKED inference functions (preregistration section 9) and the anticipated-precision simulation.

Offline, no simulator, no LLM. numpy is imported lazily (analysis only).

Inference (frozen):
  * estimate     D = sum_f T_f / sum_f n_f  (T_f = family total of item-level differences, n_f = test items in family f;
                 item difference = mean of its EN and FA presentation differences) -- item-weighted SR difference;
  * interval     paired cluster bootstrap over families, percentile 95 % (`cluster_ci`) -- DISPLAY interval;
  * decisions    one-sided wild cluster bootstrap tests with the null imposed (`wcr_p`): Webb 6-point weights, B = 9999,
                 cluster-robust t statistic with small-sample factor F/(F-1). H0: D <= delta0 ("greater") or
                 H0: D >= delta0 ("less"); reject at p < .025. H1 needs the wild test AND the percentile bound to agree
                 (`h1_decision`): with 42 unequal families both were measured anticonservative at the margin when
                 discordance is low and clustering high (`python sim/final_test_power.py --calib`).
Simulation model (a planning aid, not evidence): per presentation the paired outcome is discordant with probability d_f
and favours Gemma with probability q_f (E[diff] = d(2q-1) = D); family heterogeneity d_f, q_f ~ Beta(mean, ICC rho);
the FA twin copies the EN outcome with probability `twin`.

    python sim/final_test_power.py              # precision / power grid (wild test decisions)
    python sim/final_test_power.py --calib      # type-I error at the margin: percentile CI vs wild test
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MARGIN = 0.05
ALPHA_ONE_SIDED = 0.025
SEED = 20261010
WEBB = (-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5))


# ------------------------------------------------------------------------------------------- locked analysis functions
def estimate(fam_tot, sizes) -> float:
    return float(sum(fam_tot)) / float(sum(sizes))


def _se(fam_tot, sizes, est):
    import numpy as np
    F, N = len(sizes), float(np.sum(sizes))
    e = np.asarray(fam_tot, float) - est * np.asarray(sizes, float)
    return math.sqrt(F / (F - 1) * float(np.sum(e ** 2))) / N


def cluster_ci(fam_tot, sizes, B=20000, seed=SEED, level=0.95):
    """Percentile interval of the paired cluster bootstrap (resample whole families with replacement)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    T, n = np.asarray(fam_tot, float), np.asarray(sizes, float)
    F = len(n)
    W = rng.multinomial(F, [1 / F] * F, size=B).astype(float)
    stat = (W @ T) / (W @ n)
    a = (1 - level) / 2
    lo, hi = np.percentile(stat, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def wcr_p(fam_tot, sizes, delta0, side="greater", B=9999, seed=SEED, V=None):
    """One-sided wild cluster bootstrap p-value, null imposed. side='greater': H0 D <= delta0 vs H1 D > delta0."""
    import numpy as np
    T, n = np.asarray(fam_tot, float), np.asarray(sizes, float)
    F, N = len(n), float(n.sum())
    est = float(T.sum() / N)
    se = _se(T, n, est)
    t = (est - delta0) / max(se, 1e-12)
    if V is None:
        V = np.asarray(WEBB)[np.random.default_rng(seed).integers(0, 6, size=(B, F))]
    u = T - delta0 * n                                   # residual family totals under the null
    Tb = delta0 * n + V * u
    estb = Tb.sum(1) / N
    eb = Tb - estb[:, None] * n
    seb = np.sqrt(F / (F - 1) * np.sum(eb ** 2, 1)) / N
    tb = (estb - delta0) / np.maximum(seb, 1e-12)
    if side == "greater":
        return float((1 + np.sum(tb >= t)) / (1 + len(tb)))
    return float((1 + np.sum(tb <= t)) / (1 + len(tb)))


def h1_decision(fam_tot, sizes, B=9999, seed=SEED) -> dict:
    """Preregistered H1 wording (section 9.3). Four one-sided tests at .025, nested by construction."""
    p_ni = wcr_p(fam_tot, sizes, -MARGIN, "greater", B, seed)
    p_sup = wcr_p(fam_tot, sizes, 0.0, "greater", B, seed)
    p_lower = wcr_p(fam_tot, sizes, 0.0, "less", B, seed)
    p_inf = wcr_p(fam_tot, sizes, -MARGIN, "less", B, seed)
    a = ALPHA_ONE_SIDED
    lo, hi = cluster_ci(fam_tot, sizes, seed=seed)
    # CONJUNCTION (locked): a claim needs BOTH the wild test and the percentile interval -- both were measured
    # anticonservative in some regimes (--calib), so the stricter of the two decides.
    if p_sup < a and lo > 0:
        word = "superior"
    elif p_ni < a and lo > -MARGIN:
        word = "non-inferior (within 5 pp)" + ("; SR also statistically lower" if (p_lower < a and hi < 0) else "")
    elif p_inf < a and hi < -MARGIN:
        word = "materially inferior (worse by more than 5 pp)"
    else:
        word = "non-inferiority unresolved (inconclusive)"
    return {"estimate": estimate(fam_tot, sizes), "ci95": (lo, hi), "p_ni": p_ni,
            "p_sup": p_sup, "p_lower": p_lower, "p_materially_inferior": p_inf, "wording": word}


# ------------------------------------------------------------------------------------------------------- simulation
def _beta(rng, mean, rho, size):
    import numpy as np
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    if rho <= 0:
        return np.full(size, mean)
    k = (1 - rho) / rho
    return rng.beta(mean * k, (1 - mean) * k, size)


def _draw_family_totals(rng, sizes, fam, D, d, rho, twin):
    import numpy as np
    F, N = len(sizes), int(np.sum(sizes))
    q = 0.5 * (1 + D / d)
    d_f, q_f = _beta(rng, d, rho, F), _beta(rng, q, rho, F)
    di, qi = d_f[fam], q_f[fam]

    def draw():
        return (rng.random(N) < di) * np.where(rng.random(N) < qi, 1.0, -1.0)
    en = draw()
    fa = np.where(rng.random(N) < twin, en, draw())
    return np.bincount(fam, weights=0.5 * (en + fa), minlength=F)


def simulate(sizes, D, d, rho, twin=0.7, sims=400, B=999, seed=SEED, calib=False):
    import numpy as np
    if not (0 <= 0.5 * (1 + D / d) <= 1):
        return None
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes)
    F = len(sizes)
    fam = np.repeat(np.arange(F), sizes)
    V = np.asarray(WEBB)[rng.integers(0, 6, size=(B, F))]
    out = {"half_width": [], "ni": 0, "sup": 0, "pct_ni": 0}
    for _ in range(sims):
        T = _draw_family_totals(rng, sizes, fam, D, d, rho, twin)
        lo, hi = cluster_ci(T, sizes, B=B, seed=int(rng.integers(1 << 31)))
        out["half_width"].append((hi - lo) / 2)
        out["pct_ni"] += lo > -MARGIN
        out["ni"] += wcr_p(T, sizes, -MARGIN, "greater", V=V) < ALPHA_ONE_SIDED
        if not calib:
            out["sup"] += wcr_p(T, sizes, 0.0, "greater", V=V) < ALPHA_ONE_SIDED
    return {"half_width": float(np.mean(out["half_width"])), "p_ni": out["ni"] / sims, "p_sup": out["sup"] / sims,
            "p_ni_percentile": out["pct_ni"] / sims}


def _sizes():
    import final_test_split as fts
    m = fts.build()
    test = set(m["test"])
    return [sum(1 for i in f["items"] if i in test) for f in m["families"] if f["role"] == "test"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="final-test precision / power / calibration")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--calib", action="store_true", help="type-I error at the margin (true D = -5 pp)")
    a = ap.parse_args(argv)
    sizes = _sizes()
    print(f"test families {len(sizes)}, items {sum(sizes)}, sizes {sorted(sizes)}")
    if a.calib:
        print(f"true D = -5 pp (H0 boundary); nominal one-sided alpha {ALPHA_ONE_SIDED}")
        print(f"{'disc':>5} {'rho':>4} {'percentile-CI':>13} {'wild test':>9}")
        for d in (0.10, 0.20, 0.30):
            for rho in (0.0, 0.3, 0.6):
                r = simulate(sizes, -MARGIN, d, rho, sims=a.sims, B=a.boot, calib=True)
                print(f"{d:5.2f} {rho:4.1f} {r['p_ni_percentile']:13.3f} {r['p_ni']:9.3f}")
        return 0
    print(f"{'D':>6} {'disc':>5} {'rho':>4} {'CI half':>8} {'P(NI)':>6} {'P(sup)':>6}")
    for D in (0.0, 0.05, 0.10, 0.15):
        for d in (0.10, 0.20, 0.30):
            for rho in (0.0, 0.3, 0.6):
                r = simulate(sizes, D, d, rho, sims=a.sims, B=a.boot)
                if r is None:
                    continue
                print(f"{D:+6.2f} {d:5.2f} {rho:4.1f} {100 * r['half_width']:7.1f}pp {r['p_ni']:6.2f} {r['p_sup']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
