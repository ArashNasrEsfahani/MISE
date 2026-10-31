"""The paired contrast on a (level-restricted) score index, and the two rounding conventions.

D = SR(a) - SR(b) on the paired common subset (a presentation flagged `excluded_reset` in either arm leaves both arms; A36 rule), item
value = mean of its retained EN/FA differences, equal item weights. Interval: 95 % percentile family-cluster bootstrap (whole families
resampled, B = 20000, seed final_test_power.SEED), i.e. analyze.contrast(..., drop_flag=a36.flagged, amended=True) ->
final_test_power.cluster_ci. Level subsets: the score index restricted to the items of that level, so only families containing such items
are resampled (as in analyze.arm_rates by level).
"""
import sys

sys.dont_write_bytecode = True

import adjusted as CA     # noqa: E402  (puts sim/final_test/matrix and sim on sys.path; imports analyze, a36, k23_analyze)

AN, A36, KA = CA.AN, CA.A36, CA.KA
import final_test_power as P      # noqa: E402,F401  (used through noexcl.P: the bootstrap seed)


def p1(x):
    """Rates: "%.1f" % (100 x + 1e-9)."""
    return float("%.1f" % (100 * x + 1e-9))


def pc1(x):
    """Contrasts: f"{100 x:.1f}" (exact ties to even), as analyze and a36 print them."""
    return float(f"{100 * x:.1f}")


def tie(x):
    """True when 100 x sits on a rounding tie at one decimal (the two conventions above then differ by 0.1)."""
    return pc1(x) != p1(x)


def level_index(S, lvl_of, level):
    return S if level == "all" else {k: r for k, r in S.items() if lvl_of.get(k[1]) == level}


def paired(idx, drop_invalid, a, b, level, drop=True):
    """analyze.contrast on the (level-restricted) index: A36 paired common subset, item-weighted, 95 % family-cluster bootstrap."""
    S = level_index(idx["S"], idx["level"], level)
    c = AN.contrast(S, idx["fam"], a, b, drop_invalid, None, drop_flag=A36.flagged if drop else None, amended=drop)
    if c is None:
        return {"a": a, "b": b, "level": level, "available": False}
    lo, hi = c["ci95"]
    return {"a": a, "b": b, "level": level, "available": True, "estimate": c["estimate"], "ci95": [lo, hi],
            "estimate_pp": pc1(c["estimate"]), "ci95_pp": [pc1(lo), pc1(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "rounding_tie": [n for n, v in (("estimate", c["estimate"]), ("lo", lo), ("hi", hi)) if tie(v)],
            "n_presentations": c["n_pres"], "n_items": c["n_items"], "n_families": c["n_families"],
            "dropped_flagged_presentations": c["dropped_flagged"], "missing_presentations": c["missing_pres"],
            "shared_execution_presentations": c["n_shared"], "discordant_presentations": c["n_discordant"],
            "discordant_families": c["discordant_families"],
            "families_a_better": c["families_a_better"], "families_tied": c["tie"], "families_b_better": c["families_b_better"]}
