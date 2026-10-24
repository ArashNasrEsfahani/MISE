"""Step 5 / A2 - weights, tiers, retirement (docs/step5-plan.md section 5). Pure functions, no store access.

    w = w0[kind, verification] * 2^(-(E_now - E_support) / H[kind]) * gamma^c * rho^n_resets

E is the episode clock (completed practice episodes), never wall time. `e_now` is clamped so that an explicit clock
older than a support stamp cannot produce a weight above w0 (age >= 0). `n_resets` is relative to the epoch of the
claim's last support, so fresh support clears the reset penalty. Retirement (`w < retire_below`) is computed on read;
nothing is deleted. A frozen clock cannot represent physical change between runs: it only fixes the ranking.
"""
from __future__ import annotations

TIERS = ("T0", "T1", "T2", "T3", "T3b", "T4")
TIER_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T3b": 4, "T4": 5, "EXPLORE": 9}
RESET_KINDS = ("location", "placement")
KIND_H = {"location": "H_location", "placement": "H_placement", "negative": "H_negative", "habit": "H_habit",
          "lesson": "H_lesson", "procedure": "H_procedure"}


def half_life(params: dict, kind: str) -> float:
    return float(params["decay"][KIND_H[kind]]["value"])


def w0(params: dict, kind: str, verification: str | None = None) -> float:
    table = params["decay"]["w0"]["value"]
    key = kind if verification is None else f"{kind}_{verification}"
    return float(table[key])


def weight(params: dict, kind: str, verification: str | None, e_now: int, e_support: int, c: int = 0,
           n_resets: int = 0) -> float:
    d = params["decay"]
    age = max(0, int(e_now) - int(e_support))
    w = w0(params, kind, verification) * 2.0 ** (-age / half_life(params, kind))
    w *= float(d["gamma_contradiction"]["value"]) ** max(0, int(c))
    if kind in RESET_KINDS:
        w *= float(d["reset_factor"]["value"]) ** max(0, int(n_resets))
    return w


def retired(params: dict, w: float) -> bool:
    return w < float(params["decay"]["retire_below"]["value"])


def location_tier(row: dict, epoch_now: int, frame_now=None):
    """(tier, status) of a location claim row, or (None, 'retired') when its status is retired.

    c >= 2 -> T3b 'doubted' (below habits); older epoch / other frame -> T2 'maybe'; c == 1 -> T2 'contradicted-once';
    otherwise T0 'current'. Demotion is by tier, not only by weight.
    """
    if row["status"] == "retired":
        return None, "retired"
    c = int(row.get("contradictions") or 0)
    maybe = row["epoch"] < epoch_now or (frame_now is not None and row.get("frame") not in (None, frame_now))
    if c >= 2:
        return "T3b", "doubted"
    if maybe:
        return "T2", "maybe"
    if c == 1:
        return "T2", "contradicted-once"
    return "T0", "current"
