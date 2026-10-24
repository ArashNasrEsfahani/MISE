"""Final test -- H1 statistics, stage H-H1-stats (FINAL-PLAN item 4; prereg A22.4 / A24 item 5; amendment A24.1 H1 part).

Offline, Windows python, numpy + scipy. No simulator, no LLM, no outcome data (the DGPs use only the 42 real family SIZES).

History. The frozen G-H1-validate screen (sim/final_test_h1.py, protocol sha 414982f540f7e4e4, 146 configurations x 4,000 datasets) found that
NO candidate (C0 registered, C2, C3, C5, C6) keeps every null error <= .0275; its frozen consequence was "H1 estimation-only". Before any
acceptance or test data, A24.1 amends that consequence to allow ONE more, named and versioned rule (Astra, docs/final-test/_astra/
H-H1-stats.procedure.txt, CHANGE applied):

  h1-rule-1 = C7 = CR2 + Satterthwaite (the exact C5 algebra: intercept-only model of item differences, working independence, families as
  clusters) + guard K >= 20 (families with at least one discordant used presentation; guard failure = inconclusive, never inferior) +
  one-sided computational level alpha* = .0175 in each tail: L, U = est -/+ t_nu(.9825) se (a NOMINALLY 96.5 % interval). Wordings:
  NI iff L > -.05; superior iff NI and L > 0 (fixed sequence); materially inferior iff U < -.05; "statistically lower" iff NI and U < 0;
  observed se = 0 or undefined -> inconclusive. alpha* and the guard were fixed after a disclosed EXPLORATORY calibration (own seeds, never
  acceptance evidence: docs/final-test/h1_validation/calib_explore*.json).

This module holds
  * a VECTORISED engine (batch bootstrap-t / CR2 over many datasets), checked against the per-dataset deployable functions (selftest);
  * the deployable per-dataset rule `h1_rule_c7` that analyze.py calls (selftest: batch == deployable);
  * Stage V100: validation of C7 only, at R = 100,000 datasets in EVERY configuration of the frozen 146-configuration grid PLUS a frozen
    challenge set (guard transitions, the neighbourhood of the screen's high-discordance failures, effect allocation to large vs small
    families, missingness x transition x heterogeneity, probit neighbourhood). Seeds: 20261013 * 1000 + index (never used before).
    PASS (all-or-nothing) iff in EVERY acceptance null cell the one-sided 95 % Clopper-Pearson upper bound <= .0275 AND the point
    estimate <= .025. Any failure -> H1 estimation-only, no further candidate, no recalibration;
  * a power report (every wording of the complete hierarchy, guard-pass, conditional rates) on a power-only grid;
  * an EXPLORATORY equivalence procedure E1 for G+M vs O+M: plain CR2 + Satterthwaite 90 % interval + guard K >= 20, equivalent iff
    L > -.05 and U < +.05 (TOST at .05 per side); size at and outside the +/-5 pp boundaries and power at |D| < .05.

    python sim/final_test_h1_stats.py --bench
    python sim/final_test_h1_stats.py --stage v100  --workers 16
    python sim/final_test_h1_stats.py --stage power --workers 16
    python sim/final_test_h1_stats.py --stage equiv --workers 16
    python sim/final_test_h1_stats.py --report all
Selftest: python sim/_ftm_h1_stats_selftest.py
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

import final_test_h1 as H                                                     # noqa: E402  (screen module; read, not changed)

PROCEDURE_VERSION = "h1-rule-1"          # the replacement H1 rule = candidate C7 (A24.1)
H1_CANDIDATE = "C7"
EQUIV_PROCEDURE = "E1"                   # exploratory equivalence rule (A24.1)
MARGIN, ALPHA, TOL = H.MARGIN, H.ALPHA, H.TOL
ALPHA_STAR = 0.0175                      # C7 one-sided computational level per tail (nominal 96.5 % interval)
EQ_ALPHA = 0.05                          # E1: TOST at .05 per side = the 90 % interval
#: (base procedure, one-sided level per tail, guarded) for every rule this module can run
RULES = {"C2": ("bt", 0.025, False), "C3": ("bt", 0.025, True), "C5": ("cr2", 0.025, False), "C6": ("cr2", 0.025, True),
         "C7": ("cr2", ALPHA_STAR, True), "E1": ("cr2", EQ_ALPHA, True)}
R_V100 = 100_000                         # datasets per configuration, Stage V100 (every configuration, challenge set included)
R_POWER = 20_000
R_EQUIV = 20_000
SEED_V100 = H.SEED_VALID                 # 20261013 * 1000 (+ config index): the frozen Stage V seeds, never used before this stage
SEED_POWER = 20261014 * 1000
SEED_EQUIV = 20261015 * 1000
SEED_CALIB = 20261016 * 1000
CHUNK = 250                              # datasets per batch (B x CHUNK doubles per array, bootstrap rules)
OUT = H.OUT


def _np():
    import numpy as np
    return np


def _tq(alpha_side: float, nu):
    from scipy.stats import t as tdist
    return tdist.ppf(1.0 - alpha_side, nu)


# ============================================================================================ vectorised engine
def bt_batch(S, n, alpha_side: float = 0.025, B: int = H.B_BT, seed: int = H.SEED):
    """Batch bootstrap-t (C2) over R datasets with the SAME F families (rows of S, n: (R, F)). Identical arithmetic to
    final_test_h1.bootstrap_t, per column. Ranks via final_test_h1.bt_ranks at level 1 - 2 alpha_side. Returns est, se, defined, L, U."""
    np = _np()
    S, n = np.asarray(S, float), np.asarray(n, float)
    R, F = S.shape
    N = n.sum(1)
    est = S.sum(1) / N
    ss0 = ((S - n * est[:, None]) ** 2).sum(1)
    scale0 = (S * S).sum(1) + est * est * (n * n).sum(1)
    defined = np.isfinite(ss0) & (ss0 > 1e-12 * np.maximum(scale0, 1e-300)) & (F >= 2)
    se = np.where(defined, np.sqrt(F / (F - 1) * np.where(defined, ss0, 0.0)) / N, 0.0)
    W = H.multinomial_counts(F, B, seed)
    Nb = W @ n.T
    estb = (W @ S.T) / Nb
    a2, a1, a0 = W @ (S * S).T, W @ (S * n).T, W @ (n * n).T
    ss = a2 - 2 * estb * a1 + estb * estb * a0
    zero = ss <= 1e-12 * (a2 + estb * estb * a0 + 1e-300)
    ss = np.where(zero, 0.0, ss)
    seb = np.sqrt(F / (F - 1) * ss) / Nb
    diff = estb - est[None, :]
    diff = np.where(np.abs(diff) <= 1e-12, 0.0, diff)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(zero, np.sign(diff) * np.inf, diff / np.where(zero, 1.0, seb))
    t = np.where(zero & (diff == 0), 0.0, t)
    lo_r, hi_r = H.bt_ranks(B, round(1 - 2 * alpha_side, 12))
    tp = np.partition(t, sorted({lo_r - 1, hi_r - 1}), axis=0)
    q_lo, q_hi = tp[lo_r - 1], tp[hi_r - 1]
    with np.errstate(invalid="ignore"):
        L = np.where(q_hi == np.inf, -np.inf, est - se * q_hi)
        U = np.where(q_lo == -np.inf, np.inf, est - se * q_lo)
    return {"estimate": est, "se": se, "defined": defined, "L": np.where(defined, L, np.nan), "U": np.where(defined, U, np.nan)}


def cr2_batch(S, n, alpha_side: float = 0.025):
    """Batch CR2 + Satterthwaite, identical arithmetic to final_test_h1.cr2 (which is alpha_side .025); n may differ per row (nu per row)."""
    np = _np()
    S, n = np.asarray(S, float), np.asarray(n, float)
    N = n.sum(1)
    est = S.sum(1) / N
    r2 = (S - n * est[:, None]) ** 2
    ss = (r2 / (1.0 - n / N[:, None])).sum(1)
    defined = ss > 1e-12 * np.maximum((S * S).sum(1) + est * est * (n * n).sum(1), 1e-300)
    se = np.where(defined, np.sqrt(np.where(defined, ss, 0.0)) / N, 0.0)
    if len({tuple(r) for r in n.tolist()}) > 1:
        nu = np.array([H.satterthwaite_nu(row) for row in n])
    else:
        nu = np.full(len(N), H.satterthwaite_nu(n[0]))
    q = _tq(alpha_side, nu)
    return {"estimate": est, "se": se, "defined": defined, "nu": nu,
            "L": np.where(defined, est - q * se, np.nan), "U": np.where(defined, est + q * se, np.nan)}


def claims_batch(L, U, defined, guard_ok):
    """Vectorised final_test_h1.claims_from_bounds: undefined or guard failure -> no claim (the complete wording hierarchy + fixed sequence)."""
    np = _np()
    ok = defined & guard_ok
    with np.errstate(invalid="ignore"):
        ni = ok & (L > -MARGIN)
        return {"ni": ni, "sup": ni & (L > 0), "inf": ok & (U < -MARGIN), "lower": ni & (U < 0)}


def cr2_single(S, n, alpha_side: float) -> dict:
    """Per-dataset CR2 at any one-sided level: final_test_h1.cr2's arithmetic with the critical value t_nu(1 - alpha_side)."""
    np = _np()
    S, n = np.asarray(S, float), np.asarray(n, float)
    N = float(n.sum())
    est = float(S.sum() / N)
    ss = float(np.sum((S - n * est) ** 2 / (1.0 - n / N)))
    if not math.isfinite(ss) or ss <= 1e-12 * max(float(np.sum(S * S) + est * est * np.sum(n * n)), 1e-300):
        return {"estimate": est, "se": 0.0, "L": None, "U": None, "defined": False, "reason": "observed se = 0"}
    se = math.sqrt(ss) / N
    nu = H.satterthwaite_nu(n)
    q = float(_tq(alpha_side, nu))
    if not (math.isfinite(nu) and nu > 0 and math.isfinite(q)):
        return {"estimate": est, "se": se, "L": None, "U": None, "defined": False, "reason": "degrees of freedom undefined"}
    return {"estimate": est, "se": se, "L": est - q * se, "U": est + q * se, "defined": True, "nu": nu, "t_crit": q}


def single(cand: str, S, n) -> dict:
    """Per-dataset bounds of rule `cand` through the deployable functions (C2/C3/C5/C6 = exactly final_test_h1.bootstrap_t / cr2)."""
    base, a, _g = RULES[cand]
    if base == "bt":
        if a == 0.025:
            return H.bootstrap_t(S, n)
        np = _np()
        b = bt_batch(np.asarray(S, float)[None, :], np.asarray(n, float)[None, :], a)
        d = bool(b["defined"][0])
        return {"defined": d, "L": float(b["L"][0]) if d else None, "U": float(b["U"][0]) if d else None, "estimate": float(b["estimate"][0])}
    return H.cr2(S, n) if a == 0.025 else cr2_single(S, n, a)


def rule_batch(cand: str, S, n, K):
    """Claims of rule `cand` for R datasets. Datasets where any family has n_f = 0 go through the per-dataset deployable path (families
    dropped as analyze.item_diffs does). Returns claims (bool arrays) + defined, guard_ok, L, U."""
    np = _np()
    base, a, guarded = RULES[cand]
    S, n, K = np.asarray(S, float), np.asarray(n, float), np.asarray(K)
    R = len(S)
    full = (n > 0).all(1)
    guard_ok = (K >= H.K_GUARD) if guarded else np.ones(R, bool)
    L, U, defined = np.full(R, np.nan), np.full(R, np.nan), np.zeros(R, bool)
    idx = np.flatnonzero(full)
    if len(idx):
        b = (bt_batch if base == "bt" else cr2_batch)(S[idx], n[idx], a)
        L[idx], U[idx], defined[idx] = b["L"], b["U"], b["defined"]
    for j in np.flatnonzero(~full):                                    # rare: a family lost every presentation -> F < 42
        keep = n[j] > 0
        r = single(cand, S[j][keep], n[j][keep])
        defined[j] = r["defined"]
        if r["defined"]:
            L[j], U[j] = r["L"], r["U"]
    out = claims_batch(L, U, defined, guard_ok)
    out.update(defined=defined, guard_ok=guard_ok, L=L, U=U)
    return out


# ============================================================================================ the deployable H1 rule (analyze.py calls this)
def h1_rule_c7(S, n, K) -> dict:
    """h1-rule-1 (C7) on one dataset: S = family totals of item differences, n = items per family (families with n_f = 0 already dropped),
    K = families with at least one discordant used presentation. Returns estimate, se, nu, the nominal 96.5 % interval, guard, claims, wording."""
    r = cr2_single(S, n, ALPHA_STAR)
    guard_ok = K is not None and int(K) >= H.K_GUARD
    cl = H.claims_from_bounds(r["L"], r["U"], r["defined"], guard_ok)
    if not r["defined"]:
        w = "non-inferiority unresolved (inconclusive): interval undefined (" + r.get("reason", "no between-family variation") + ")"
    elif not guard_ok:
        w = f"non-inferiority unresolved (inconclusive): guard failed (K = {K} < {H.K_GUARD} families with a discordant presentation)"
    else:
        w = H.wording(cl)
    return {"procedure": PROCEDURE_VERSION, "candidate": H1_CANDIDATE, "alpha_one_sided_computational": ALPHA_STAR,
            "interval_nominal_level": round(1 - 2 * ALPHA_STAR, 6), "estimate": r["estimate"], "se": r["se"], "nu": r.get("nu"),
            "t_crit": r.get("t_crit"), "interval": [r["L"], r["U"]], "defined": r["defined"],
            "guard": {"K": None if K is None else int(K), "threshold": H.K_GUARD, "passed": bool(guard_ok)}, "claims": cl, "wording": w}


def equiv_e1(S, n, K) -> dict:
    """EXPLORATORY E1 (A24.1): CR2 + Satterthwaite 90 % interval + guard K >= 20; 'equivalent within +/-5 pp' iff L > -.05 and U < +.05."""
    r = cr2_single(S, n, EQ_ALPHA)
    guard_ok = K is not None and int(K) >= H.K_GUARD
    eq = bool(r["defined"] and guard_ok and r["L"] > -MARGIN and r["U"] < MARGIN)
    return {"procedure": EQUIV_PROCEDURE, "exploratory": True, "estimate": r["estimate"], "interval90": [r["L"], r["U"]], "defined": r["defined"],
            "guard": {"K": None if K is None else int(K), "threshold": H.K_GUARD, "passed": bool(guard_ok)}, "equivalent": eq,
            "wording": ("exploratory evidence of +/-5 pp equivalence" if eq else "equivalence not established")}


# ============================================================================================ grids
def challenge_configs(start: int):
    """Frozen challenge set (Astra H-H1-stats point 3), added to V100 BEFORE it ran. Indexes continue after the 146 frozen ones."""
    C = []

    def add(cid, D, **kw):
        C.append({"id": cid, "D": round(D, 6), **kw})
    # guard transitions: P(K >= 20) appreciably on both sides (screen guard-pass .1-.9), at the NI boundary and at D = 0
    trans = [(0.0, .15), (0.0, .175), (0.0, .225), (0.0, .25), (.3, .25), (.3, .275), (.3, .325), (.3, .35), (.6, .35), (.6, .45)]
    for D in (-.05, 0.0):
        for rho, d in trans:
            add(f"X-tr-D{D}-d{d}-r{rho}", D, dgp="A", d=d, rho=rho, twin=.7)
        for rho, d in ((0.0, .20), (.3, .30)):
            add(f"X-trtw-D{D}-d{d}-r{rho}", D, dgp="A", d=d, rho=rho, twin=.4)
    # neighbourhood of the screen's high-discordance failures, both skew directions
    for d in (.5, .6, .7, .8):
        for rho in (0.0, .3):
            for D in (-.10, -.05, 0.0, .05):
                add(f"X-hi-D{D}-d{d}-r{rho}", D, dgp="A", d=d, rho=rho, twin=.7)
    # effect allocation to large (n_f >= 4, 90 items) vs small (35 items) families at high discordance, item-weighted Delta preserved
    for Dt, DL in ((-.05, -.10), (-.05, 0.0), (0.0, .05), (0.0, -.05)):
        DS = (Dt * 125 - DL * 90) / 35
        add(f"X-size-D{Dt}-DL{DL}", Dt, dgp="A", kind="size", d_L=.6, D_L=DL, d_S=max(.6, abs(DS)), D_S=DS, rho=.3, twin=.7)
    # missingness x guard transition x heterogeneity (MCAR per arm-presentation)
    for m in (.03, .06):
        add(f"X-mis{m}-A-d.2-r0", -.05, dgp="A", d=.20, rho=0.0, twin=.7, mcar=m)
        add(f"X-mis{m}-A-d.3-r.3", -.05, dgp="A", d=.30, rho=.3, twin=.7, mcar=m)
        add(f"X-mis{m}-B-p.6-hetero", -.05, dgp="B", p_O=.60, s_e=.10, s_g=.30, mcar=m)
    # probit neighbourhood of the screen failures (p_O .6 heterogeneous / medium)
    for pO in (.50, .60, .70):
        for sname, (se_, sg) in {"medium": (.20, .10), "hetero": (.10, .30)}.items():
            for D in (-.05, 0.0):
                add(f"X-B-D{D}-p{pO}-{sname}", D, dgp="B", p_O=pO, s_e=se_, s_g=sg)
    for pO in (.50, .60):
        add(f"X-B-D0.05-p{pO}-medium", .05, dgp="B", p_O=pO, s_e=.20, s_g=.10)
    for i, c in enumerate(C):
        c["index"] = start + i
        c["nulls"] = H.null_sets(c["D"])
        c["challenge"] = True
    return C


def v100_configs():
    """Stage V100 = the frozen 146-configuration grid (unchanged; indexes fix the seeds) + the frozen challenge set."""
    base = H.configs()
    return base + challenge_configs(len(base))


def power_configs():
    """Power-only grid (not in acceptance): DGP-A Delta in {+.10, +.05, 0, -.10} x d x rho; DGP-B Delta +.10 / +.05 / 0."""
    C = []

    def add(cid, D, **kw):
        C.append({"id": cid, "D": round(D, 6), **kw})
    for D in (.10, .05, 0.0, -.10):
        for d in (.10, .15, .20, .30, .40):
            if abs(D) > d:
                continue
            for rho in (0.0, .3, .6):
                add(f"P-A-D{D}-d{d}-r{rho}", D, dgp="A", d=d, rho=rho, twin=.7)
    for D in (.10, .05, 0.0):
        for pO in (.60, .80):
            for sname, (se_, sg) in {"high": (.05, .05), "medium": (.20, .10), "hetero": (.10, .30)}.items():
                add(f"P-B-D{D}-p{pO}-{sname}", D, dgp="B", p_O=pO, s_e=se_, s_g=sg)
    for i, c in enumerate(C):
        c["index"] = i
    return C


def equiv_configs():
    """Exploratory equivalence (G+M vs O+M): size at AND outside the +/-.05 boundaries (|D| .05, .075, .10), power at |D| <= .025,
    guard transitions included; DGP-B at D 0 and +/-.05."""
    C = []

    def add(cid, D, **kw):
        C.append({"id": cid, "D": round(D, 6), **kw})
    for D in (0.0, .025, -.025, .05, -.05, .075, -.075, .10, -.10):
        for d in (.05, .10, .15, .20, .25, .30, .40, .60):
            if abs(D) > d:
                continue
            for rho in (0.0, .3, .6):
                add(f"E-A-D{D}-d{d}-r{rho}", D, dgp="A", d=d, rho=rho, twin=.7)
    for D in (0.0, .05, -.05):
        for pO in (.60, .80):
            for sname, (se_, sg) in {"high": (.05, .05), "medium": (.20, .10), "hetero": (.10, .30)}.items():
                add(f"E-B-D{D}-p{pO}-{sname}", D, dgp="B", p_O=pO, s_e=se_, s_g=sg)
    for i, c in enumerate(C):
        c["index"] = i
    return C


STAGES = {"v100": (v100_configs, "R_V100", "SEED_V100", H1_CANDIDATE), "power": (power_configs, "R_POWER", "SEED_POWER", H1_CANDIDATE),
          "equiv": (equiv_configs, "R_EQUIV", "SEED_EQUIV", EQUIV_PROCEDURE)}


def stage_spec(stage: str) -> dict:
    fn, rk, sk, rule = STAGES[stage]
    return {"stage": stage, "rule": rule, "rule_def": RULES[rule], "procedure_version": PROCEDURE_VERSION, "configs": fn(), "R": globals()[rk],
            "seed": globals()[sk], "K_GUARD": H.K_GUARD, "tol": TOL, "acceptance": "cp_upper95 <= .0275 AND rate <= .025 in every acceptance null cell",
            "screen_protocol_sha": H.protocol_sha()}


def stage_sha(stage: str) -> str:
    return hashlib.sha256(json.dumps(stage_spec(stage), sort_keys=True).encode()).hexdigest()


# ============================================================================================ runner
COUNT_KEYS = ("ni", "sup", "inf", "lower", "neg", "any", "inconclusive", "guard_pass", "undefined", "eq", "dropped_family",
              "cover", "lo_miss", "hi_miss", "defined_n", "ni_given_guard", "inf_given_guard")


def run_one(args):
    cfg, stage = args
    np = _np()
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    spec = stage_spec(stage)
    R, rule = spec["R"], spec["rule"]
    t0 = time.time()
    rng = np.random.default_rng(spec["seed"] + cfg["index"])
    sizes = H.family_sizes()
    ct = {k: 0 for k in COUNT_KEYS}
    disc_sum = K_sum = 0.0
    D = cfg["D"]
    done = 0
    while done < R:
        r = min(2000, R - done)
        en, fa = (H.gen_A if cfg["dgp"] == "A" else H.gen_B)(rng, r, sizes, cfg)
        en, fa = H.apply_missing(rng, en, fa, cfg)
        S, n, K, disc = H.summarise(en, fa, sizes)
        disc_sum += float(disc.sum())
        K_sum += float(K.sum())
        step = CHUNK if RULES[rule][0] == "bt" else r
        for s in range(0, r, step):
            sl = slice(s, min(s + step, r))
            res = rule_batch(rule, S[sl], n[sl], K[sl])
            for k in ("ni", "sup", "inf", "lower"):
                ct[k] += int(res[k].sum())
            ct["neg"] += int((res["inf"] | res["lower"]).sum())
            ct["any"] += int((res["ni"] | res["inf"]).sum())
            ct["inconclusive"] += int((~(res["ni"] | res["inf"])).sum())
            g = res["guard_ok"]
            ct["guard_pass"] += int(g.sum())
            ct["ni_given_guard"] += int((res["ni"] & g).sum())
            ct["inf_given_guard"] += int((res["inf"] & g).sum())
            d = res["defined"]
            ct["undefined"] += int((~d).sum())
            ct["defined_n"] += int(d.sum())
            ct["dropped_family"] += int((~(n[sl] > 0).all(1)).sum())
            with np.errstate(invalid="ignore"):                     # interval coverage, all defined datasets, guard ignored (reported apart)
                ct["cover"] += int((d & (res["L"] < D) & (res["U"] > D)).sum())
                ct["lo_miss"] += int((d & (res["L"] >= D)).sum())
                ct["hi_miss"] += int((d & (res["U"] <= D)).sum())
                if stage == "equiv":
                    ct["eq"] += int((d & g & (res["L"] > -MARGIN) & (res["U"] < MARGIN)).sum())
        done += r
    return {"schema": "ftm.h1-stats/2", "stage": stage, "rule": rule, "procedure_version": PROCEDURE_VERSION, "stage_sha": stage_sha(stage),
            "config": cfg, "R": R, "seed": spec["seed"] + cfg["index"], "counts": ct, "mean_discordance": disc_sum / R, "mean_K": K_sum / R,
            "seconds": round(time.time() - t0, 1)}


def journal(stage: str) -> Path:
    return OUT / f"{stage}.{STAGES[stage][3]}.jsonl"


def done_ids(path: Path, sha: str) -> set:
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("stage_sha") == sha:
                    out.add(r["config"]["id"])
    return out


def run_stage(stage: str, workers: int, only=None, limit=None) -> int:
    import multiprocessing as mp
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    sha = stage_sha(stage)
    path = journal(stage)
    done = done_ids(path, sha)
    todo = [c for c in stage_spec(stage)["configs"] if c["id"] not in done and (not only or c["id"] in only)]
    if limit:
        todo = todo[:limit]
    print(f"[{stage}] rule {STAGES[stage][3]} {PROCEDURE_VERSION} sha {sha[:16]}; {len(done)} done, {len(todo)} to run, workers {workers}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, [(c, stage) for c in todo]), 1):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"[{stage}] {i}/{len(todo)} {row['config']['id']} {row['seconds']}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return 0


# ============================================================================================ exploratory calibration (history; informed alpha*)
CALIB_ALPHAS = (0.025, 0.0225, 0.02, 0.0175, 0.015, 0.0125, 0.01)
R_CALIB = 10_000


def cr2_parts(S, n):
    """est, se, nu, defined per dataset for the CR2 rule, dropping empty families per dataset (as analyze.item_diffs)."""
    np = _np()
    R = len(S)
    est, se, nu, d = np.zeros(R), np.zeros(R), np.zeros(R), np.zeros(R, bool)
    full = (n > 0).all(1)
    idx = np.flatnonzero(full)
    if len(idx):
        b = cr2_batch(S[idx], n[idx])
        est[idx], se[idx], nu[idx], d[idx] = b["estimate"], b["se"], b["nu"], b["defined"]
    for j in np.flatnonzero(~full):
        keep = n[j] > 0
        r = H.cr2(S[j][keep], n[j][keep])
        est[j], se[j], d[j] = r["estimate"], r["se"], r["defined"]
        nu[j] = H.satterthwaite_nu(n[j][keep])
    return est, se, nu, d


def run_calib(args):
    """EXPLORATORY (informed the choice of alpha* only; never acceptance): CR2 + Satterthwaite + guard K >= 20 at one-sided level alpha* in
    CALIB_ALPHAS, on the frozen 146 configurations, own seeds (20261016 * 1000 + index), R_CALIB datasets."""
    cfg, = args
    np = _np()
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    t0 = time.time()
    rng = np.random.default_rng(SEED_CALIB + cfg["index"])
    sizes = H.family_sizes()
    ct = {str(a): {k: 0 for k in ("ni", "sup", "inf", "lower", "neg")} for a in CALIB_ALPHAS}
    gp = 0
    done = 0
    while done < R_CALIB:
        r = min(2000, R_CALIB - done)
        en, fa = (H.gen_A if cfg["dgp"] == "A" else H.gen_B)(rng, r, sizes, cfg)
        en, fa = H.apply_missing(rng, en, fa, cfg)
        S, n, K, _ = H.summarise(en, fa, sizes)
        est, se, nu, d = cr2_parts(S, n)
        ok = d & (K >= H.K_GUARD)
        gp += int((K >= H.K_GUARD).sum())
        for a in CALIB_ALPHAS:
            q = _tq(a, np.where(nu > 0, nu, 1.0))
            cl = claims_batch(est - q * se, est + q * se, ok, np.ones(r, bool))
            c = ct[str(a)]
            for k in ("ni", "sup", "inf", "lower"):
                c[k] += int(cl[k].sum())
            c["neg"] += int((cl["inf"] | cl["lower"]).sum())
        done += r
    return {"schema": "ftm.h1-calib/1", "exploratory": True, "config": cfg, "R": R_CALIB, "seed": SEED_CALIB + cfg["index"], "alphas": list(CALIB_ALPHAS),
            "counts": ct, "guard_pass": gp, "seconds": round(time.time() - t0, 1)}


def calib_stage(workers: int) -> int:
    import multiprocessing as mp
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    path = OUT / "calib_explore.C6.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(x)["config"]["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [c for c in H.configs() if c["id"] not in done]
    print(f"[calib] {len(done)} done, {len(todo)} to run", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_calib, [(c,) for c in todo]), 1):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[calib] {i}/{len(todo)} {row['config']['id']} {row['seconds']}s", flush=True)
    return 0


def calib_report() -> dict:
    rows = [json.loads(x) for x in (OUT / "calib_explore.C6.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = {"schema": "ftm.h1-calib-summary/1", "exploratory": True, "n_configs": len(rows), "by_alpha": {}}
    for a in CALIB_ALPHAS:
        worst, nfail, n25, pw = {}, 0, 0, []
        for r in rows:
            cfg, R, c = r["config"], r["R"], r["counts"][str(a)]
            for et in ("ni", "inf", "sup", "neg"):
                if cfg["nulls"][et] and not cfg.get("report_only"):
                    p = c[et] / R
                    nfail += p > TOL
                    n25 += p > ALPHA
                    if p > worst.get(et, ("", -1))[1]:
                        worst[et] = (cfg["id"], p)
            if cfg["id"] in H.POWER_SET:
                pw.append(c["ni"] / R)
        out["by_alpha"][str(a)] = {"worst": worst, "n_cells_above_.0275": nfail, "n_cells_above_.025": n25,
                                   "power_selection_set": sum(pw) / len(pw) if pw else None}
    (OUT / "calib_explore_summary.C6.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return out


# ============================================================================================ reports
def cp_interval(k: int, n: int, conf: float = 0.95):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf((1 - conf) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))
    return lo, hi


def rows_of(stage: str) -> list:
    sha = stage_sha(stage)
    path = journal(stage)
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("stage_sha") == sha:
                    rows[r["config"]["id"]] = r
    return [rows[c["id"]] for c in stage_spec(stage)["configs"] if c["id"] in rows]


def _rate(k, R):
    p = k / R
    lo, hi = cp_interval(k, R)
    return {"k": k, "rate": p, "mc_se": math.sqrt(max(p * (1 - p), 1e-12) / R), "cp95": [lo, hi]}


def report_v100() -> dict:
    rows = rows_of("v100")
    n_cfg = len(v100_configs())
    cells = []
    for r in rows:
        cfg, R, ct = r["config"], r["R"], r["counts"]
        for et in ("ni", "inf", "sup", "neg"):
            if not cfg["nulls"][et]:
                continue
            x = _rate(ct[et], R)
            x.update(config=cfg["id"], error=et, D=cfg["D"], R=R, cp_upper95_one_sided=H.cp_upper(ct[et], R), report_only=bool(cfg.get("report_only")),
                     challenge=bool(cfg.get("challenge")), mean_discordance=r["mean_discordance"], mean_K=r["mean_K"], guard_pass=ct["guard_pass"] / R)
            x["passes"] = x["cp_upper95_one_sided"] <= TOL + 1e-12 and x["rate"] <= ALPHA + 1e-12
            cells.append(x)
    acc = [c for c in cells if not c["report_only"]]
    fails = [c for c in acc if not c["passes"]]
    worst = {}
    for et in ("ni", "inf", "sup", "neg"):
        xs = [x for x in acc if x["error"] == et]
        if xs:
            worst[et] = max(xs, key=lambda x: x["rate"])
    cover = []
    for r in rows:
        ct = r["counts"]
        if ct["defined_n"]:
            cover.append({"config": r["config"]["id"], "D": r["config"]["D"], "coverage": ct["cover"] / ct["defined_n"], "lo_miss": ct["lo_miss"] / ct["defined_n"],
                          "hi_miss": ct["hi_miss"] / ct["defined_n"], "defined_share": ct["defined_n"] / r["R"]})
    complete = len(rows) == n_cfg
    out = {"schema": "ftm.h1-v100-summary/2", "rule": H1_CANDIDATE, "procedure_version": PROCEDURE_VERSION, "alpha_star": ALPHA_STAR, "stage_sha": stage_sha("v100"),
           "configs_done": len(rows), "configs_total": n_cfg, "frozen_grid": len(H.configs()), "challenge": n_cfg - len(H.configs()), "complete": complete,
           "R_per_config": R_V100, "tolerance": TOL,
           "rule_text": "PASS (all-or-nothing) iff every acceptance null cell has one-sided 95 % Clopper-Pearson upper bound <= .0275 AND point rate <= .025; "
                        "R = 100,000 in every configuration; any failure -> H1 estimation-only",
           "passes": complete and not fails, "n_cells": len(acc), "n_fail": len(fails), "fails": sorted(fails, key=lambda x: -x["rate"]),
           "max_rate": max((c["rate"] for c in acc), default=None), "max_cp_upper": max((c["cp_upper95_one_sided"] for c in acc), default=None),
           "n_cells_point_above_nominal_025": sum(c["rate"] > ALPHA for c in acc), "worst_by_error": worst,
           "report_only": [c for c in cells if c["report_only"]],
           "union_at_zero": [{"config": r["config"]["id"], "sup": r["counts"]["sup"] / r["R"], "neg": r["counts"]["neg"] / r["R"],
                              "union": (r["counts"]["sup"] + r["counts"]["neg"]) / r["R"]} for r in rows if abs(r["config"]["D"]) < 1e-9],
           "coverage_nominal_965": {"note": "coverage of the nominal 96.5 % interval over defined datasets, guard ignored; reported apart from claim errors",
                                    "min": min((c["coverage"] for c in cover), default=None), "by_config": cover},
           "cells": cells}
    (OUT / f"v100_summary.{H1_CANDIDATE}.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return out


def report_power(stage: str = "power") -> dict:
    rows = rows_of(stage)
    tab = []
    for r in rows:
        R, ct = r["R"], r["counts"]
        g = ct["guard_pass"]
        row = {"config": r["config"]["id"], "D": r["config"]["D"], "R": R, "mean_discordance": round(r["mean_discordance"], 4), "mean_K": round(r["mean_K"], 2),
               "guard_pass": g / R}
        keys = ("eq",) if stage == "equiv" else ("ni", "sup", "inf", "lower", "inconclusive")
        for k in keys:
            p = ct[k] / R
            row[k] = p
            row[k + "_mc_se"] = math.sqrt(max(p * (1 - p), 1e-12) / R)
        if stage != "equiv":
            row["non_inferior_only"] = (ct["ni"] - ct["sup"]) / R
            row["ni_given_guard"] = (ct["ni_given_guard"] / g) if g else None
        tab.append(row)
    out = {"schema": f"ftm.h1-{stage}-summary/2", "rule": STAGES[stage][3], "procedure_version": PROCEDURE_VERSION, "stage_sha": stage_sha(stage),
           "configs_done": len(rows), "configs_total": len(stage_spec(stage)["configs"]), "table": tab,
           "note": "power over ALL generated datasets (guard failure = no claim); scenario-specific, not a forecast for the real experiment"}
    if stage == "equiv":
        err = [t for t in tab if abs(t["D"]) >= MARGIN - 1e-9]
        w = max(err, key=lambda t: t["eq"]) if err else None
        out["size_max_at_or_outside_boundary"] = None if w is None else {"config": w["config"], "rate": w["eq"], "mc_se": w["eq_mc_se"]}
        out["size_cells_above_.05"] = [t["config"] for t in err if t["eq"] > EQ_ALPHA]
        out["note"] += ("; EXPLORATORY E1: CR2 + Satterthwaite 90 % interval + guard K >= 20, equivalent iff L > -.05 and U < +.05; error = an "
                        "equivalence claim when |D| >= .05 (nominal <= .05)")
    (OUT / f"{stage}_summary.{STAGES[stage][3]}.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return out


def bench(rule="C7", R=4000) -> dict:
    np = _np()
    rng = np.random.default_rng(1)
    cfg = {"id": "bench", "D": -.05, "dgp": "A", "d": .2, "rho": .3, "twin": .7}
    sizes = H.family_sizes()
    rule_batch(rule, *H.summarise(*H.gen_A(rng, 10, sizes, cfg), sizes)[:2], np.zeros(10, int))
    t0 = time.time()
    en, fa = H.gen_A(rng, R, sizes, cfg)
    S, n, K, _ = H.summarise(en, fa, sizes)
    t1 = time.time()
    step = CHUNK if RULES[rule][0] == "bt" else R
    for s in range(0, R, step):
        rule_batch(rule, S[s:s + step], n[s:s + step], K[s:s + step])
    t2 = time.time()
    return {"rule": rule, "R": R, "gen_s": round(t1 - t0, 2), "rule_s": round(t2 - t1, 2), "projected_s_per_100k": round((t2 - t0) / R * 1e5, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=tuple(STAGES))
    ap.add_argument("--report", choices=tuple(STAGES) + ("all",))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bench", choices=tuple(RULES))
    ap.add_argument("--calib", choices=("run", "report"))
    a = ap.parse_args(argv)
    if a.calib:
        if a.calib == "run":
            return calib_stage(a.workers)
        print(json.dumps(calib_report(), indent=1))
        return 0
    if a.bench:
        for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")
        print(json.dumps(bench(a.bench)))
        return 0
    if a.stage:
        return run_stage(a.stage, a.workers, a.only, a.limit)
    if a.report:
        for st in (tuple(STAGES) if a.report == "all" else (a.report,)):
            res = report_v100() if st == "v100" else report_power(st)
            print(st, json.dumps({k: v for k, v in res.items() if k in ("configs_done", "configs_total", "complete", "passes", "n_fail", "n_cells", "max_rate",
                                                                         "max_cp_upper", "n_cells_point_above_nominal_025", "size_max_at_or_outside_boundary")}))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
