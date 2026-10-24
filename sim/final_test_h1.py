"""Final test -- H1 validity (prereg amendment A22 item 4, stage G-H1-validate). Offline, numpy (+ scipy for CR2), no simulator, no LLM.

Why: the registered H1 rule (final_test_power.h1_decision = wild cluster test AND percentile bound) was never simulated as a whole and its
wild half is far above nominal at low discordance (docs/final-test/h1_validation/audit.json). This module holds
  * the candidate H1 decision rules, written as the DEPLOYABLE functions analyze.py calls (fixed seeds, so each rule is a deterministic
    function of the per-family data);
  * the data-generating processes and the frozen configuration grid of the protocol in docs/final-test/progress.md
    ("G-H1-validate PROTOCOL", frozen before any simulation);
  * a resumable, parallel runner: one jsonl row per finished configuration in docs/final-test/h1_validation/.

final_test_power.py is LOCKED and is not edited; C0 below reproduces its h1_decision + analyze.fixed_sequence exactly (selftest).

    python sim/final_test_h1.py --stage screen   [--workers 18]        # Stage S: every candidate, 4,000 datasets per configuration
    python sim/final_test_h1.py --report screen                         # screen verdicts + selection (writes screen_summary.json)
    python sim/final_test_h1.py --stage validate --candidate C2         # Stage V: ONE candidate, fresh seeds, 16,000 per configuration
    python sim/final_test_h1.py --report validate --candidate C2        # validation verdict (writes validation_summary.json)
Selftest: python sim/_ftm_h1_selftest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import final_test_power as P                                                  # noqa: E402  (locked; only read)

MARGIN = P.MARGIN                    # 0.05
ALPHA = P.ALPHA_ONE_SIDED            # 0.025
SEED = P.SEED                        # 20261010: the fixed bootstrap seed of the deployed rules
B_BT = 9999                          # bootstrap-t resamples (C2 / C3)
B_WILD = 9999                        # C0 wild test (registered)
B_PCT = 20000                        # C0 percentile interval (registered)
K_GUARD = 20                         # C3 / C6 guard: families with >= 1 discordant used presentation
TOL = 0.0275                         # declared calibration allowance (1.1 alpha, Bradley stringent)
R_SCREEN, R_VALID = 4000, 16000
SEED_SCREEN, SEED_VALID = 20261012 * 1000, 20261013 * 1000
ORDER = ("C2", "C5", "C3", "C6", "C0")
CANDIDATES = {
    "C0": "registered: wild cluster test (B 9,999) AND percentile bound (B 20,000), fixed sequence (baseline only)",
    "C2": "studentized cluster bootstrap (bootstrap-t), B 9,999, equal-tailed 95 %, one-based ranks 250 / 9,750, no interpolation",
    "C3": "C2 + guard K >= 20 (families with at least one discordant used presentation)",
    "C5": "CR2 + Satterthwaite t (intercept-only model of item differences, working independence)",
    "C6": "C5 + guard K >= 20",
}
OUT = HERE.parent / "docs" / "final-test" / "h1_validation"
CLAIMS = ("ni", "sup", "inf", "lower")

_CACHE: dict = {}


# ============================================================================================ deployable decision rules
def _np():
    import numpy as np
    return np


def multinomial_counts(F: int, B: int = B_BT, seed: int = SEED):
    """The fixed family-resampling matrix (B x F multiplicities m_f), numpy default_rng(seed).multinomial, as final_test_power.cluster_ci."""
    key = ("mult", F, B, seed)
    if key not in _CACHE:
        np = _np()
        _CACHE[key] = np.random.default_rng(seed).multinomial(F, [1 / F] * F, size=B).astype(float)
    return _CACHE[key]


def bt_ranks(B: int = B_BT, level: float = 0.95):
    """One-based ranks of the order statistics used for the equal-tailed bootstrap-t interval: lo = floor(a (B+1)), hi = ceil((1-a)(B+1))."""
    a = (1 - level) / 2
    return int(math.floor(a * (B + 1) + 1e-9)), int(math.ceil((1 - a) * (B + 1) - 1e-9))


def cr1_se(S, n) -> float:
    np = _np()
    S, n = np.asarray(S, float), np.asarray(n, float)
    F, N = len(n), float(n.sum())
    est = float(S.sum() / N)
    return math.sqrt(F / (F - 1) * float(np.sum((S - n * est) ** 2))) / N


def bootstrap_t(S, n, B: int = B_BT, seed: int = SEED) -> dict:
    """C2. est = sum S_f / N; se^2 = F/(F-1) sum (S_f - n_f est)^2 / N^2; per resample (multiplicities m_f): N* = sum m n, est* = sum m S / N*,
    se*^2 = F/(F-1) sum m (S - n est*)^2 / N*^2 (m, not m^2; factor F/(F-1) of the ORIGINAL F). t* = (est* - est)/se*; se* = 0 -> signed infinity
    (0 when est* = est). L = est - se q_hi, U = est - se q_lo (order statistics 9,750 / 250 for B = 9,999). Observed se = 0 -> undefined."""
    np = _np()
    S, n = np.asarray(S, float), np.asarray(n, float)
    F, N = len(n), float(n.sum())
    est = float(S.sum() / N)
    ss0 = float(np.sum((S - n * est) ** 2))
    scale0 = float(np.sum(S * S) + est * est * np.sum(n * n))
    if not (F >= 2 and math.isfinite(ss0)) or ss0 <= 1e-12 * max(scale0, 1e-300):
        return {"estimate": est, "se": 0.0, "L": None, "U": None, "defined": False, "reason": "observed se = 0"}
    se = math.sqrt(F / (F - 1) * ss0) / N
    W = multinomial_counts(F, B, seed)
    Nb = W @ n
    estb = (W @ S) / Nb
    a2, a1, a0 = W @ (S * S), W @ (S * n), W @ (n * n)
    ss = a2 - 2 * estb * a1 + estb * estb * a0
    zero = ss <= 1e-12 * (a2 + estb * estb * a0 + 1e-300)
    ss = np.where(zero, 0.0, ss)
    seb = np.sqrt(F / (F - 1) * ss) / Nb
    diff = estb - est
    diff = np.where(np.abs(diff) <= 1e-12, 0.0, diff)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(zero, np.sign(diff) * np.inf, diff / np.where(zero, 1.0, seb))
    t = np.where(zero & (diff == 0), 0.0, t)
    lo_r, hi_r = bt_ranks(B)
    q_lo = float(np.partition(t, lo_r - 1)[lo_r - 1])
    q_hi = float(np.partition(t, hi_r - 1)[hi_r - 1])
    L = -math.inf if q_hi == math.inf else est - se * q_hi
    U = math.inf if q_lo == -math.inf else est - se * q_lo
    return {"estimate": est, "se": se, "L": L, "U": U, "defined": True, "q": (q_lo, q_hi)}


def satterthwaite_nu(n) -> float:
    """CR2 Satterthwaite degrees of freedom for the intercept-only model, working independence: g_f'g_h = c_f c_h (delta_fh n_f - n_f n_h / N) / N^2,
    c_f = (1 - n_f/N)^(-1/2); nu = (sum_f g_f'g_f)^2 / sum_{f,h} (g_f'g_h)^2 (the N^2 cancels)."""
    np = _np()
    n = np.asarray(n, float)
    key = ("nu", tuple(n.tolist()))
    if key not in _CACHE:
        N = n.sum()
        c = 1.0 / np.sqrt(1.0 - n / N)
        G = np.outer(c, c) * (np.diag(n) - np.outer(n, n) / N)
        _CACHE[key] = float(np.trace(G) ** 2 / np.sum(G * G))
    return _CACHE[key]


def cr2(S, n) -> dict:
    """C5. se^2 = sum (S_f - n_f est)^2 / (1 - n_f/N) / N^2; bounds est -/+ t_{nu}(.975) se."""
    np = _np()
    from scipy.stats import t as tdist
    S, n = np.asarray(S, float), np.asarray(n, float)
    N = float(n.sum())
    est = float(S.sum() / N)
    r2 = (S - n * est) ** 2
    ss = float(np.sum(r2 / (1.0 - n / N)))
    if ss <= 1e-12 * max(float(np.sum(S * S) + est * est * np.sum(n * n)), 1e-300):
        return {"estimate": est, "se": 0.0, "L": None, "U": None, "defined": False, "reason": "observed se = 0"}
    se = math.sqrt(ss) / N
    nu = satterthwaite_nu(n)
    key = ("tq", round(nu, 12))
    if key not in _CACHE:
        _CACHE[key] = float(tdist.ppf(1 - ALPHA, nu))
    q = _CACHE[key]
    return {"estimate": est, "se": se, "L": est - q * se, "U": est + q * se, "defined": True, "nu": nu, "t_crit": q}


def claims_from_bounds(L, U, defined=True, guard_ok=True) -> dict:
    """Wordings from a two-sided 95 % interval (each side one-sided .025). Undefined or guard failure -> no claim (inconclusive, never inferior)."""
    if not defined or not guard_ok or L is None or U is None:
        return {"ni": False, "sup": False, "inf": False, "lower": False}
    ni = L > -MARGIN
    return {"ni": bool(ni), "sup": bool(ni and L > 0), "inf": bool(U < -MARGIN), "lower": bool(ni and U < 0)}


def wording(cl: dict) -> str:
    if cl["sup"]:
        return "superior"
    if cl["ni"]:
        return "non-inferior (within 5 pp)" + ("; SR also statistically lower" if cl["lower"] else "")
    if cl["inf"]:
        return "materially inferior (worse by more than 5 pp)"
    return "non-inferiority unresolved (inconclusive)"


def _webb_V(F: int, B: int = B_WILD, seed: int = SEED):
    key = ("webb", F, B, seed)
    if key not in _CACHE:
        np = _np()
        _CACHE[key] = np.asarray(P.WEBB)[np.random.default_rng(seed).integers(0, 6, size=(B, F))]
    return _CACHE[key]


def _pct_ci(T, n, B: int = B_PCT, seed: int = SEED, level=0.95):
    """Identical to final_test_power.cluster_ci (same generator call), with the resampling matrix cached."""
    np = _np()
    T, n = np.asarray(T, float), np.asarray(n, float)
    W = multinomial_counts(len(n), B, seed)
    stat = (W @ T) / (W @ n)
    a = (1 - level) / 2
    lo, hi = np.percentile(stat, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def c0_claims(S, n) -> dict:
    """C0 = final_test_power.h1_decision + analyze.fixed_sequence, reproduced with cached (identical) resampling matrices."""
    V = _webb_V(len(n))
    p_ni = P.wcr_p(S, n, -MARGIN, "greater", V=V)
    p_sup = P.wcr_p(S, n, 0.0, "greater", V=V)
    p_lower = P.wcr_p(S, n, 0.0, "less", V=V)
    p_inf = P.wcr_p(S, n, -MARGIN, "less", V=V)
    lo, hi = _pct_ci(S, n)
    a = ALPHA
    ni_holds = p_ni < a and lo > -MARGIN
    if p_sup < a and lo > 0:
        w = "superior"
    elif ni_holds:
        w = "non-inferior (within 5 pp)" + ("; SR also statistically lower" if (p_lower < a and hi < 0) else "")
    elif p_inf < a and hi < -MARGIN:
        w = "materially inferior (worse by more than 5 pp)"
    else:
        w = "non-inferiority unresolved (inconclusive)"
    if w == "superior" and not ni_holds:                       # analyze.fixed_sequence
        w = "non-inferiority unresolved (inconclusive)"
    return {"ni": w == "superior" or w.startswith("non-inferior "), "sup": w == "superior", "inf": w.startswith("materially"),
            "lower": "statistically lower" in w, "_wording": w}


def decide(candidate: str, S, n, K: int | None = None) -> dict:
    """Claims of one candidate on one dataset (S = family totals of item differences, n = items per family, K = families with a discordant
    used presentation; K is needed only by the guarded candidates)."""
    if candidate == "C0":
        return c0_claims(S, n)
    if candidate in ("C2", "C3"):
        r = bootstrap_t(S, n)
    elif candidate in ("C5", "C6"):
        r = cr2(S, n)
    else:
        raise ValueError(candidate)
    guard_ok = True
    if candidate in ("C3", "C6"):
        if K is None:
            raise ValueError("guarded candidate needs K")
        guard_ok = K >= K_GUARD
    return claims_from_bounds(r["L"], r["U"], r["defined"], guard_ok)


def h1_rule(candidate: str, S, n, K: int | None = None) -> dict:
    """The deployable H1 block for analyze.py: estimate, interval, guard, claims and the wording."""
    if candidate == "C0":
        cl = c0_claims(S, n)
        lo, hi = _pct_ci(S, n)
        return {"candidate": "C0", "estimate": P.estimate(S, n), "interval": [lo, hi], "claims": {k: cl[k] for k in CLAIMS}, "wording": cl["_wording"]}
    r = bootstrap_t(S, n) if candidate in ("C2", "C3") else cr2(S, n)
    guard = None
    guard_ok = True
    if candidate in ("C3", "C6"):
        guard_ok = K is not None and K >= K_GUARD
        guard = {"K": K, "threshold": K_GUARD, "passed": bool(guard_ok)}
    cl = claims_from_bounds(r["L"], r["U"], r["defined"], guard_ok)
    w = wording(cl)
    if not r["defined"]:
        w = "non-inferiority unresolved (inconclusive): interval undefined (no between-family variation)"
    elif not guard_ok:
        w = f"non-inferiority unresolved (inconclusive): guard failed (K = {K} < {K_GUARD})"
    return {"candidate": candidate, "method": CANDIDATES[candidate], "estimate": r["estimate"], "se": r["se"],
            "interval": [r["L"], r["U"]], "defined": r["defined"], "guard": guard, "claims": cl, "wording": w}


# ============================================================================================ data-generating processes
def family_sizes():
    """The 42 real test family sizes in the frozen split-manifest order (no outcome data)."""
    if "sizes" not in _CACHE:
        _CACHE["sizes"] = P._sizes()
    return _CACHE["sizes"]


def _beta(rng, mean, rho, size):
    np = _np()
    if mean <= 0 or mean >= 1 or rho <= 0:
        return np.full(size, float(min(max(mean, 0.0), 1.0)))
    k = (1 - rho) / rho
    return rng.beta(mean * k, (1 - mean) * k, size)


def _q(d, D):
    if d <= 0:
        return 0.5
    q = 0.5 * (1 + D / d)
    if q < -1e-12 or q > 1 + 1e-12:
        raise ValueError(f"infeasible: |Delta| {D} > d {d}")
    return min(max(q, 0.0), 1.0)


def _draw_pres(rng, R, di, qi):
    """(R, N) presentation differences: discordant with prob di, +1 with prob qi else -1 (di, qi broadcast to (R, N))."""
    np = _np()
    shape = di.shape
    return (rng.random(shape) < di) * np.where(rng.random(shape) < qi, 1.0, -1.0)


def gen_A(rng, R, sizes, cfg):
    """DGP-A and its structural stresses. Returns en, fa: (R, N) float in {-1, 0, 1}."""
    np = _np()
    sizes = np.asarray(sizes)
    F, N = len(sizes), int(sizes.sum())
    fam = np.repeat(np.arange(F), sizes)
    kind = cfg.get("kind", "A")
    twin = cfg.get("twin", 0.7)
    rho = cfg.get("rho", 0.0)
    if kind == "allornothing":                                   # every presentation of a family identical: -1 w.p. a, +1 w.p. b, else 0
        u = rng.random((R, F))
        v = np.where(u < cfg["a"], -1.0, np.where(u < cfg["a"] + cfg["b"], 1.0, 0.0))
        en = v[:, fam]
        return en, en.copy()
    if kind == "twopoint":                                       # share pi of families discordant at d_hi with direction q_hi, others concordant
        disc = rng.random((R, F)) < cfg["pi"]
        df = np.where(disc, cfg["d_hi"], 0.0)
        qf = np.full((R, F), cfg["q_hi"])
        di, qi = df[:, fam], qf[:, fam]
        en = _draw_pres(rng, R, di, qi)
        fa = np.where(rng.random((R, N)) < twin, en, _draw_pres(rng, R, di, qi))
        return en, fa
    if kind == "lang":                                           # EN / FA asymmetry, twin 0, family heterogeneity per language
        out = []
        for d, D in ((cfg["d_en"], cfg["D_en"]), (cfg["d_fa"], cfg["D_fa"])):
            q = _q(d, D)
            df = np.stack([_beta(rng, d, rho, F) for _ in range(R)])
            qf = np.stack([_beta(rng, q, rho, F) if 0 < q < 1 else np.full(F, q) for _ in range(R)])
            out.append(_draw_pres(rng, R, df[:, fam], qf[:, fam]))
        return out[0], out[1]
    if kind == "size":                                           # effect / discordance by family size group (n_f >= 4 vs <= 3)
        large = sizes >= 4
        df = np.empty((R, F))
        qf = np.empty((R, F))
        for grp, (d, D) in ((large, (cfg["d_L"], cfg["D_L"])), (~large, (cfg["d_S"], cfg["D_S"]))):
            q = _q(d, D)
            m = int(grp.sum())
            df[:, grp] = np.stack([_beta(rng, d, rho, m) for _ in range(R)])
            qf[:, grp] = np.stack([_beta(rng, q, rho, m) if 0 < q < 1 else np.full(m, q) for _ in range(R)])
        di, qi = df[:, fam], qf[:, fam]
        en = _draw_pres(rng, R, di, qi)
        fa = np.where(rng.random((R, N)) < twin, en, _draw_pres(rng, R, di, qi))
        return en, fa
    d, D = cfg["d"], cfg["D"]
    if d <= 0:
        z = np.zeros((R, N))
        return z, z.copy()
    q = _q(d, D)
    df = np.stack([_beta(rng, d, rho, F) for _ in range(R)])
    qf = np.stack([_beta(rng, q, rho, F) if 0 < q < 1 else np.full(F, q) for _ in range(R)])
    di, qi = df[:, fam], qf[:, fam]
    en = _draw_pres(rng, R, di, qi)
    fa = np.where(rng.random((R, N)) < twin, en, _draw_pres(rng, R, di, qi))
    return en, fa


def gen_B(rng, R, sizes, cfg):
    """DGP-B latent probit. Shared variance S = 1 - s_g - s_e split .3 family / .5 item / .2 presentation."""
    np = _np()
    from scipy.stats import norm
    sizes = np.asarray(sizes)
    F, N = len(sizes), int(sizes.sum())
    fam = np.repeat(np.arange(F), sizes)
    se_, sg = cfg["s_e"], cfg["s_g"]
    Ssh = 1.0 - se_ - sg
    mu_O = norm.ppf(cfg["p_O"])
    mu_G = norm.ppf(cfg["p_O"] + cfg["D"])
    a = rng.normal(0, math.sqrt(0.3 * Ssh), (R, F))[:, fam]
    b = rng.normal(0, math.sqrt(0.5 * Ssh), (R, N))
    out = []
    gG = rng.normal(0, math.sqrt(sg), (R, F))[:, fam]
    gO = rng.normal(0, math.sqrt(sg), (R, F))[:, fam]
    for _lang in range(2):
        c = rng.normal(0, math.sqrt(0.2 * Ssh), (R, N))
        shared = a + b + c
        yG = (mu_G + shared + gG + rng.normal(0, math.sqrt(se_), (R, N))) > 0
        yO = (mu_O + shared + gO + rng.normal(0, math.sqrt(se_), (R, N))) > 0
        out.append(yG.astype(float) - yO.astype(float))
    return out[0], out[1]


def apply_missing(rng, en, fa, cfg):
    """Missing presentations as NaN. MCAR: each ARM's presentation is missing w.p. p (paired use needs both) -> 1 - (1-p)^2. MNAR (stress):
    a presentation where G is worse (diff = -1) is missing w.p. m."""
    np = _np()
    p = cfg.get("mcar", 0.0)
    m = cfg.get("mnar", 0.0)
    out = []
    for x in (en, fa):
        x = x.copy()
        if p > 0:
            x[rng.random(x.shape) < 1 - (1 - p) ** 2] = np.nan
        if m > 0:
            x[(x == -1.0) & (rng.random(x.shape) < m)] = np.nan
        out.append(x)
    return out


def summarise(en, fa, sizes):
    """Per dataset (rows): S_f, n_f (items with >= 1 used presentation), K, discordance -- the analyze.item_diffs / by_family rules."""
    np = _np()
    sizes = np.asarray(sizes)
    F, N = len(sizes), int(sizes.sum())
    fam = np.repeat(np.arange(F), sizes)
    M = np.zeros((N, F))
    M[np.arange(N), fam] = 1.0
    ue, uf = ~np.isnan(en), ~np.isnan(fa)
    cnt = ue.astype(float) + uf.astype(float)
    tot = np.nan_to_num(en) + np.nan_to_num(fa)
    valid = cnt > 0
    X = np.where(valid, tot / np.maximum(cnt, 1), 0.0)
    S = X @ M
    n = valid.astype(float) @ M
    dpres = (ue & (np.nan_to_num(en) != 0)) | (uf & (np.nan_to_num(fa) != 0))
    K = ((dpres.astype(float) @ M) > 0).sum(1)
    used = cnt.sum(1)
    nd = (ue & (np.nan_to_num(en) != 0)).sum(1) + (uf & (np.nan_to_num(fa) != 0)).sum(1)
    disc = np.where(used > 0, nd / np.maximum(used, 1), 0.0)
    return S, n, K, disc


# ============================================================================================ the frozen configuration grid
def configs() -> list:
    """The frozen grid of the protocol (ledger entry 'G-H1-validate PROTOCOL'). Order fixes the outer seeds."""
    C = []

    def add(cid, D, **kw):
        C.append({"id": cid, "D": round(D, 6), **kw})
    M_ = -MARGIN
    for d in (.05, .075, .10, .125, .15, .175, .20, .30, .40, .60):
        for rho in (0.0, .3, .6, .9):
            add(f"N-A-d{d}-r{rho}", M_, dgp="A", d=d, rho=rho, twin=.7)
    for twin in (.4, 1.0):
        for d in (.05, .10, .20):
            for rho in (0.0, .6):
                add(f"N-Atw-t{twin}-d{d}-r{rho}", M_, dgp="A", d=d, rho=rho, twin=twin)
    for d in (.05, .10, .20):
        add(f"N-Amcar-d{d}", M_, dgp="A", d=d, rho=.6, twin=.7, mcar=.03)
    for d in (.10, .20):
        add(f"N-Amnar-d{d}", M_, dgp="A", d=d, rho=.6, twin=.7, mnar=.10, report_only=True)
    NL, NS = 90, 35
    for d, DL in ((.20, -.08), (.20, -(0.05 * 125 - 0.15 * 35) / 90), (.10, -.08)):
        DS = (-0.05 * 125 - DL * NL) / NS
        dS = max(d, abs(DS)) if abs(DS) > d else d
        add(f"N-Ssize-d{d}-DL{round(DL, 4)}", M_, dgp="A", kind="size", d_L=d, D_L=DL, d_S=dS, D_S=DS, rho=.3, twin=.7)
    DLc = -0.05 * 125 / NL
    DSc = -0.05 * 125 / NS
    add("N-Sconc-large-q0", M_, dgp="A", kind="size", d_L=-DLc, D_L=DLc, d_S=0.0, D_S=0.0, rho=.3, twin=.7)
    add("N-Sconc-large-d.15", M_, dgp="A", kind="size", d_L=.15, D_L=DLc, d_S=0.0, D_S=0.0, rho=.3, twin=.7)
    add("N-Sconc-small-q0", M_, dgp="A", kind="size", d_L=0.0, D_L=0.0, d_S=-DSc, D_S=DSc, rho=.3, twin=.7)
    add("N-Sconc-small-d.30", M_, dgp="A", kind="size", d_L=0.0, D_L=0.0, d_S=.30, D_S=DSc, rho=.3, twin=.7)
    add("N-Slang-1", M_, dgp="A", kind="lang", d_en=.05, D_en=-.02, d_fa=.10, D_fa=-.08, rho=.6)
    add("N-Slang-2", M_, dgp="A", kind="lang", d_en=.05, D_en=0.0, d_fa=.15, D_fa=-.10, rho=.6)
    for pi, dh in ((.1, .5), (.2, .25), (.5, .1)):
        add(f"N-S2pt-pi{pi}", M_, dgp="A", kind="twopoint", pi=pi, d_hi=dh, q_hi=0.0, twin=.7)
    add("N-Saon-a.05", M_, dgp="A", kind="allornothing", a=.05, b=0.0)
    add("N-Saon-a.15b.10", M_, dgp="A", kind="allornothing", a=.15, b=.10)
    SPLITS = {"high": (.05, .05), "medium": (.20, .10), "hetero": (.10, .30)}
    for pO in (.60, .80, .95):
        for sname, (se_, sg) in SPLITS.items():
            add(f"N-B-p{pO}-{sname}", M_, dgp="B", p_O=pO, s_e=se_, s_g=sg)
    # ---- Delta = 0
    for d in (.02, .05, .10, .20, .30, .40):
        for rho in (0.0, .3, .6, .9):
            add(f"Z-A-d{d}-r{rho}", 0.0, dgp="A", d=d, rho=rho, twin=.7)
    add("Z-A-identical", 0.0, dgp="A", d=0.0, rho=0.0, twin=.7)
    add("Z-Amcar-d0.1", 0.0, dgp="A", d=.10, rho=.6, twin=.7, mcar=.03)
    for DL in (.03, -.03):
        DS = (0.0 - DL * NL) / NS
        add(f"Z-Ssize-DL{DL}", 0.0, dgp="A", kind="size", d_L=.20, D_L=DL, d_S=.20, D_S=DS, rho=.3, twin=.7)
    add("Z-Slang", 0.0, dgp="A", kind="lang", d_en=.05, D_en=.03, d_fa=.05, D_fa=-.03, rho=.6)
    add("Z-Saon", 0.0, dgp="A", kind="allornothing", a=.05, b=.05)
    add("Z-S2pt", 0.0, dgp="A", kind="twopoint", pi=.2, d_hi=.25, q_hi=.5, twin=.7)
    for pO in (.60, .80, .95):
        for sname, (se_, sg) in SPLITS.items():
            add(f"Z-B-p{pO}-{sname}", 0.0, dgp="B", p_O=pO, s_e=se_, s_g=sg)
    # ---- interior: Delta = -.10 and +.05
    for d in (.10, .20, .40):
        for rho in (0.0, .6):
            add(f"I-m10-A-d{d}-r{rho}", -.10, dgp="A", d=d, rho=rho, twin=.7)
    for pO in (.60, .80):
        add(f"I-m10-B-p{pO}-medium", -.10, dgp="B", p_O=pO, s_e=.20, s_g=.10)
    for d in (.10, .20, .30, .40):
        for rho in (0.0, .3, .6):
            add(f"I-p05-A-d{d}-r{rho}", .05, dgp="A", d=d, rho=rho, twin=.7)
    for pO in (.60, .80):
        for sname, (se_, sg) in SPLITS.items():
            add(f"I-p05-B-p{pO}-{sname}", .05, dgp="B", p_O=pO, s_e=se_, s_g=sg)
    for i, c in enumerate(C):
        c["index"] = i
        c["nulls"] = null_sets(c["D"])
    return C


def null_sets(D: float) -> dict:
    """Which claims are ERRORS at this true Delta (Astra 4): NI over D <= -.05, inf over D >= -.05, sup over D <= 0, negative (inf or lower) over D >= 0."""
    e = 1e-9
    return {"ni": D <= -MARGIN + e, "inf": D >= -MARGIN - e, "sup": D <= 0 + e, "neg": D >= 0 - e}


POWER_SET = [f"Z-A-d{d}-r{rho}" for d in (.10, .20, .30, .40) for rho in (0.0, .3, .6)]


def protocol_sha() -> str:
    spec = {"configs": configs(), "candidates": CANDIDATES, "order": ORDER, "tol": TOL, "R": [R_SCREEN, R_VALID], "seeds": [SEED_SCREEN, SEED_VALID],
            "B": [B_BT, B_WILD, B_PCT], "K_GUARD": K_GUARD, "power_set": POWER_SET, "bootstrap_seed": SEED}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# ============================================================================================ runner
def run_config(args):
    cfg, stage, cands, R = args
    np = _np()
    t0 = time.time()
    base = SEED_SCREEN if stage == "screen" else SEED_VALID
    rng = np.random.default_rng(base + cfg["index"])
    sizes = family_sizes()
    counts = {c: {"ni": 0, "sup": 0, "inf": 0, "lower": 0, "neg": 0, "any": 0, "guard_pass": 0, "undefined": 0} for c in cands}
    disc_sum = K_sum = 0.0
    nfam_short = 0
    CH = 2000
    done = 0
    while done < R:
        r = min(CH, R - done)
        if cfg["dgp"] == "A":
            en, fa = gen_A(rng, r, sizes, cfg)
        else:
            en, fa = gen_B(rng, r, sizes, cfg)
        en, fa = apply_missing(rng, en, fa, cfg)
        S, n, K, disc = summarise(en, fa, sizes)
        disc_sum += float(disc.sum())
        K_sum += float(K.sum())
        for j in range(r):
            keep = n[j] > 0
            Sj, nj = S[j][keep], n[j][keep]
            if not keep.all():
                nfam_short += 1
            Kj = int(K[j])
            bt = bootstrap_t(Sj, nj) if ("C2" in cands or "C3" in cands) else None
            cr = cr2(Sj, nj) if ("C5" in cands or "C6" in cands) else None
            for c in cands:
                if c == "C0":
                    cl = c0_claims(Sj, nj)
                else:
                    r_ = bt if c in ("C2", "C3") else cr
                    cl = claims_from_bounds(r_["L"], r_["U"], r_["defined"], (Kj >= K_GUARD) if c in ("C3", "C6") else True)
                    ct0 = counts[c]
                    ct0["undefined"] += not r_["defined"]
                ct = counts[c]
                for k in CLAIMS:
                    ct[k] += bool(cl[k])
                ct["neg"] += bool(cl["inf"] or cl["lower"])
                ct["any"] += bool(cl["ni"] or cl["inf"])
                if c in ("C3", "C6"):
                    ct["guard_pass"] += int(K[j]) >= K_GUARD
        done += r
    return {"schema": "ftm.h1-validation/1", "stage": stage, "protocol_sha": protocol_sha(), "config": cfg, "R": R, "candidates": list(cands),
            "counts": counts, "mean_discordance": disc_sum / R, "mean_K": K_sum / R, "datasets_with_dropped_family": nfam_short,
            "seconds": round(time.time() - t0, 1)}


def journal_path(stage: str, candidate: str | None = None) -> Path:
    return OUT / ("screen.jsonl" if stage == "screen" else f"validation.{candidate}.jsonl")


def done_ids(path: Path, sha: str) -> set:
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("protocol_sha") == sha:
                    out.add(r["config"]["id"])
    return out


def run_stage(stage: str, workers: int, candidate: str | None = None, only=None, limit=None) -> int:
    import multiprocessing as mp
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    sha = protocol_sha()
    path = journal_path(stage, candidate)
    done = done_ids(path, sha)
    cands = list(ORDER) if stage == "screen" else [candidate]
    R = R_SCREEN if stage == "screen" else R_VALID
    todo = [c for c in configs() if c["id"] not in done and (not only or c["id"] in only)]
    if limit:
        todo = todo[:limit]
    print(f"[{stage}] protocol {sha[:16]}; {len(done)} done, {len(todo)} to run, R {R}, candidates {cands}, workers {workers}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_config, [(c, stage, cands, R) for c in todo]), 1):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"[{stage}] {i}/{len(todo)} {row['config']['id']} {row['seconds']}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return 0


# ============================================================================================ reports
def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def _rows(path: Path, sha: str) -> list:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("protocol_sha") == sha:
                rows[r["config"]["id"]] = r
    return [rows[c["id"]] for c in configs() if c["id"] in rows]


def error_cells(rows, cand):
    """Every (config, error type) cell where the claim is an error: rate, MC SE, one-sided 95 % Clopper-Pearson upper bound."""
    cells = []
    for r in rows:
        cfg, R = r["config"], r["R"]
        ct = r["counts"][cand]
        for et, key in (("ni", "ni"), ("inf", "inf"), ("sup", "sup"), ("neg", "neg")):
            if not cfg["nulls"][et]:
                continue
            k = ct[key]
            p = k / R
            cells.append({"config": cfg["id"], "error": et, "D": cfg["D"], "k": k, "R": R, "rate": p, "mc_se": math.sqrt(max(p * (1 - p), 1e-12) / R),
                          "cp_upper95": cp_upper(k, R), "report_only": bool(cfg.get("report_only")), "mean_discordance": r["mean_discordance"], "mean_K": r["mean_K"]})
    return cells


def report(stage: str, candidate: str | None = None) -> dict:
    sha = protocol_sha()
    path = journal_path(stage, candidate)
    rows = _rows(path, sha)
    n_cfg = len(configs())
    cands = list(ORDER) if stage == "screen" else [candidate]
    out = {"schema": f"ftm.h1-{stage}-summary/1", "protocol_sha": sha, "configs_done": len(rows), "configs_total": n_cfg, "complete": len(rows) == n_cfg,
           "tolerance": TOL, "candidates": {}}
    for c in cands:
        cells = error_cells(rows, c)
        acc = [x for x in cells if not x["report_only"]]
        if stage == "screen":
            fails = [x for x in acc if x["rate"] > TOL + 1e-12]
        else:
            fails = [x for x in acc if x["cp_upper95"] > TOL + 1e-12]
        worst = {}
        for et in ("ni", "inf", "sup", "neg"):
            xs = [x for x in acc if x["error"] == et]
            if xs:
                w = max(xs, key=lambda x: (x["rate"], x["cp_upper95"]))
                worst[et] = {k: w[k] for k in ("config", "rate", "mc_se", "cp_upper95", "k", "R", "mean_discordance", "mean_K")}
        stress = [x for x in cells if x["report_only"]]
        pw = [r for r in rows if r["config"]["id"] in POWER_SET]
        power = (sum(r["counts"][c]["ni"] / r["R"] for r in pw) / len(pw)) if pw else None
        union0 = [{"config": r["config"]["id"], "sup": r["counts"][c]["sup"] / r["R"], "neg": r["counts"][c]["neg"] / r["R"],
                   "union": (r["counts"][c]["sup"] + r["counts"][c]["neg"]) / r["R"]} for r in rows if abs(r["config"]["D"]) < 1e-9]
        out["candidates"][c] = {"passes": out["complete"] and not fails, "n_cells": len(acc), "n_fail": len(fails),
                                "fails": sorted(fails, key=lambda x: -x["rate"])[:40], "worst_by_error": worst,
                                "stress_report_only": stress, "power_ni_selection_set": power, "max_union_at_zero": max((u["union"] for u in union0), default=None),
                                "power_ni_by_config": {r["config"]["id"]: r["counts"][c]["ni"] / r["R"] for r in rows if r["config"]["D"] >= -1e-9},
                                "guard_pass_rate": ({r["config"]["id"]: r["counts"][c]["guard_pass"] / r["R"] for r in rows} if c in ("C3", "C6") else None)}
    if stage == "screen" and out["complete"]:
        passers = [c for c in ORDER if out["candidates"][c]["passes"]]
        if passers:
            best = max(out["candidates"][c]["power_ni_selection_set"] for c in passers)
            chosen = next(c for c in ORDER if c in passers and out["candidates"][c]["power_ni_selection_set"] >= best - 0.01)
        else:
            chosen = None
        out["selection"] = {"passers": passers, "chosen": chosen, "rule": "highest mean NI power on POWER_SET; within .01 of the best -> earliest in ORDER",
                            "order": ORDER, "power_set": POWER_SET}
    fn = OUT / ("screen_summary.json" if stage == "screen" else f"validation_summary.{candidate}.json")
    fn.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("screen", "validate"))
    ap.add_argument("--report", choices=("screen", "validate"))
    ap.add_argument("--candidate", choices=tuple(CANDIDATES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        cs = configs()
        print(f"{len(cs)} configurations; protocol {protocol_sha()[:16]}")
        for c in cs:
            print(c["index"], c["id"], c["D"], c["nulls"])
        return 0
    if a.stage:
        if a.stage == "validate" and not a.candidate:
            ap.error("--stage validate needs --candidate")
        return run_stage(a.stage, a.workers, a.candidate, a.only, a.limit)
    if a.report:
        res = report(a.report, a.candidate)
        print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
        for c, v in res["candidates"].items():
            print(c, "passes" if v["passes"] else "FAILS", f"fails {v['n_fail']}/{v['n_cells']}", "power", v["power_ni_selection_set"],
                  json.dumps(v["worst_by_error"]))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
