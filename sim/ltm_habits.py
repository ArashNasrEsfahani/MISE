"""Step 5 / A2 - habits: empirical search preferences (docs/step5-plan.md section 6, Astra A2 review).

A trial is one immutable contribution per configuration unit U = (kitchen, epoch, otype, canonical place). Habit counts
are DERIVED on read from the `search_trial` rows, collapsed to canonical places (so a merge or unmerge changes the
counts without rewriting history), positive over negative inside one unit, each decayed from the episode that wrote it
with `H_habit`:

    pref(rtype) = (P + a * p0) / (P + N + a)           a = habits.prior_strength

KITCHEN habits are ACTIVE.  GENERAL habits are built and tested but INACTIVE (`habits.general_active == false`): they
need >= `min_kitchens` kitchens, each with >= `min_discoveries_per_kitchen` distinct canonical natural discoveries,
every kitchen weighted equally, validated on a genuinely different third kitchen that was excluded from the fit
(leave-one-out top-type agreement) AND the registry switch. With one kitchen nothing is promotable.
"""
from __future__ import annotations

try:
    from ltm_decay import half_life
except ImportError:                     # pragma: no cover
    from sim.ltm_decay import half_life

ALIASES = {"SinkBasin": "Sink"}


def default_prior(otype: str) -> dict:
    """The fixed declared containment prior (sim/search.py); {} when unavailable."""
    try:
        try:
            from search import prior_for
        except ImportError:
            from sim.search import prior_for
        return dict(prior_for(otype))
    except Exception:
        return {}


def p0_for(prior: dict, rtype: str, params: dict) -> float:
    """Prior weight of `rtype` normalised to sum 1 over the listed receptacle types; unlisted -> p0_unlisted."""
    pr = {ALIASES.get(k, k): float(v) for k, v in prior.items()}
    total = sum(pr.values())
    key = ALIASES.get(rtype, rtype)
    if total > 0 and key in pr:
        return pr[key] / total
    return float(params["habits"]["p0_unlisted"]["value"])


def alias_map(store) -> dict:
    return {r["place"]: r["canonical"] for r in store.select("place_alias") if r["active"]}


def canon_place(alias: dict, pid):
    seen = set()
    while pid in alias and pid not in seen:
        seen.add(pid)
        pid = alias[pid]
    return pid


def collapse_trials(rows, alias: dict) -> list:
    """One entry per (kitchen, epoch, otype, canonical place): positive if any row is positive; E = earliest E of the
    winning outcome (the discovery time). Deterministic order."""
    best: dict = {}
    for r in rows:
        key = (r["kitchen"], r["epoch"], r["otype"], canon_place(alias, r["place"]))
        cur = best.get(key)
        cand = {"kitchen": r["kitchen"], "epoch": r["epoch"], "otype": r["otype"], "place": key[3],
                "rtype": r["rtype"], "outcome": r["outcome"], "E": r["E"]}
        if cur is None:
            best[key] = cand
        elif cur["outcome"] == "negative" and cand["outcome"] == "positive":
            best[key] = cand
        elif cur["outcome"] == cand["outcome"] and cand["E"] < cur["E"]:
            cur["E"] = cand["E"]
    return [best[k] for k in sorted(best, key=lambda k: (k[0], k[1], k[2], str(k[3])))]


def kitchen_prefs(collapsed, params: dict, e_now: int, otype: str, kitchen: str, prior: dict | None = None) -> list:
    """Per-rtype preferences for one kitchen and type. Sorted by (-pref, rtype): ties are deterministic."""
    H = half_life(params, "habit")
    a = float(params["habits"]["prior_strength"]["value"])
    prior = default_prior(otype) if prior is None else prior
    acc: dict = {}
    for t in collapsed:
        if t["kitchen"] != kitchen or t["otype"] != otype:
            continue
        d = acc.setdefault(t["rtype"], {"P": 0.0, "N": 0.0, "n_pos": 0, "n_neg": 0})
        w = 2.0 ** (-max(0, e_now - t["E"]) / H)
        if t["outcome"] == "positive":
            d["P"] += w
            d["n_pos"] += 1
        else:
            d["N"] += w
            d["n_neg"] += 1
    out = []
    for rtype, d in acc.items():
        p0 = p0_for(prior, rtype, params)
        out.append({"rtype": rtype, "P": d["P"], "N": d["N"], "n_pos": d["n_pos"], "n_neg": d["n_neg"], "p0": p0,
                    "pref": (d["P"] + a * p0) / (d["P"] + d["N"] + a)})
    out.sort(key=lambda r: (-round(r["pref"], 12), r["rtype"]))
    return out


def discoveries(collapsed, kitchen: str, otype: str) -> int:
    return sum(1 for t in collapsed if t["kitchen"] == kitchen and t["otype"] == otype and t["outcome"] == "positive")


def general_habit(collapsed, params: dict, e_now: int, otype: str, validation_kitchen: str | None = None,
                  prior: dict | None = None) -> dict:
    """The general (cross-kitchen) habit for `otype`.

    Returns {fit_kitchens, validation_kitchen, prefs: [...], validated, switch, active, reasons}. `active` is True only
    if the registry switch is on AND the promotion rule holds. Kitchens are discovered from the data; the validation
    kitchen is never part of the fit.
    """
    h = params["habits"]
    min_k, min_d = int(h["min_kitchens"]["value"]), int(h["min_discoveries_per_kitchen"]["value"])
    switch = bool(h["general_active"]["value"])
    kitchens = sorted({t["kitchen"] for t in collapsed if t["otype"] == otype})
    ok_k = [k for k in kitchens if discoveries(collapsed, k, otype) >= min_d]
    reasons = []
    res = {"fit_kitchens": [], "validation_kitchen": None, "prefs": [], "validated": False, "switch": switch,
           "active": False, "reasons": reasons, "kitchens_seen": kitchens, "eligible_kitchens": ok_k}
    if validation_kitchen is None:
        cands = ok_k[-1:] if len(ok_k) > min_k else []
        validation_kitchen = cands[0] if cands else None
    fit = [k for k in ok_k if k != validation_kitchen]
    res["fit_kitchens"], res["validation_kitchen"] = fit, validation_kitchen
    if len(fit) < min_k:
        reasons.append(f"needs >= {min_k} fit kitchens with >= {min_d} discoveries each (have {len(fit)})")
    # equal-weight mean of the per-kitchen preferences (a kitchen with more data does not dominate)
    prior_d = default_prior(otype) if prior is None else prior
    per_k = {k: {r["rtype"]: r["pref"] for r in kitchen_prefs(collapsed, params, e_now, otype, k, prior_d)} for k in fit}
    rtypes = sorted({rt for d in per_k.values() for rt in d})
    # a kitchen that never tried a receptacle type contributes the prior for it (P = N = 0 gives pref = p0)
    prefs = [{"rtype": rt, "pref": sum(per_k[k].get(rt, p0_for(prior_d, rt, params)) for k in fit) / len(fit),
              "kitchens": sum(1 for k in fit if rt in per_k[k])} for rt in rtypes] if fit else []
    prefs.sort(key=lambda r: (-round(r["pref"], 12), r["rtype"]))
    res["prefs"] = prefs
    if validation_kitchen is None or validation_kitchen in fit:
        reasons.append("needs a genuinely different third kitchen excluded from the fit for validation")
    elif discoveries(collapsed, validation_kitchen, otype) < min_d:
        reasons.append("validation kitchen has too few discoveries")
    elif len(fit) >= min_k and prefs:
        vp = kitchen_prefs(collapsed, params, e_now, otype, validation_kitchen, prior)
        if vp and vp[0]["rtype"] == prefs[0]["rtype"]:
            res["validated"] = True
        else:
            reasons.append("validation kitchen's top receptacle type disagrees with the general fit")
    if not switch:
        reasons.append("registry switch habits.general_active is false")
    res["active"] = bool(switch and res["validated"] and len(fit) >= min_k)
    return res
