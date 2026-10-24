"""Step 5 / A2 - `where_is` (docs/step5-plan.md sections 5.2 and 8.1). A6 extends this module with concepts / RRF / budgets.

`where_is(ltm, otype, wm=None)` is THE one answer shared by search and Gemma: an exact-key lookup (no LLM, no embeddings,
no wall clock) returning at most `retrieval.where_k` ranked candidates plus an always-present EXPLORE option.

Ranking: tier first, then weight.
    T0 current-epoch location claims, verified before trusted, no contradiction
    T1 current-epoch placements (verified, then actuator-only)
    T2 'maybe' locations (earlier epoch / other frame) and claims contradicted once
    T3 kitchen habits by preference (active general habits fill in after them, only if promoted)
    T3b doubted claims (contradicted twice or more)
    T4 the fixed declared prior
Recent verified evidence therefore always outranks a habit. A qualified negative of the CURRENT epoch removes a place from
the list (it goes to EXPLORE, last, so the suppression stays overridable) unless a newer positive defeats it. Retired
claims (weight below `retire_below`) are not offered; they remain in the database. Gemma may reorder only T2-T4 and
only the ids returned here (enforced by `reorder_within`).
"""
from __future__ import annotations

try:
    from ltm_consolidate import capable, inspection_coverage
    from ltm_decay import TIER_RANK, location_tier, retired, weight
    from ltm_habits import alias_map, canon_place, collapse_trials, default_prior, general_habit, kitchen_prefs, p0_for
    from ltm_identity import PlaceBook, id_key
except ImportError:                     # pragma: no cover
    from sim.ltm_consolidate import capable, inspection_coverage
    from sim.ltm_decay import TIER_RANK, location_tier, retired, weight
    from sim.ltm_habits import alias_map, canon_place, collapse_trials, default_prior, general_habit, kitchen_prefs, p0_for
    from sim.ltm_identity import PlaceBook, id_key

REORDERABLE = ("T2", "T3", "T3b", "T4")


def _round(x):
    return round(float(x), 6)


def _support_count(store, table, key):
    return sum(1 for r in store.select("evidence", where={"claim_table": table, "claim": f"{table}/{key}"})
               if r["mutation"] in ("create", "support", "reactivate"))


def wm_masks(ltm, wm, otype: str, book: PlaceBook) -> dict:
    """What this task has already inspected: {canonical place: 'qualified' | 'partial'} for `otype`."""
    masks: dict = {}
    if wm is None:
        return masks
    opens, closed_after = {}, {}
    for e in wm.events:
        if e["kind"] in ("open", "close", "inspection"):
            kind, pid, _ = book.resolve(e["rtype"], e["xyz"], create=False)
            cp = book.canonical(pid) if pid else None
            if e["kind"] == "open":
                opens[e["ev"]] = (cp, bool(e.get("ok")) or e.get("state_seen") == "open")
            elif e["kind"] == "close":
                if cp:
                    closed_after[cp] = e["ev"]
            elif cp:
                if otype not in set(e.get("targets") or []):
                    continue
                op = opens.get(e.get("open_ev"))
                contemporaneous = bool(op and op[1] and op[0] == cp and not (closed_after.get(cp, -1) > e["open_ev"]))
                cov = inspection_coverage(ltm.params, e, contemporaneous)
                if cov == "qualified" and (otype in set(e.get("detected") or []) or not capable(ltm.params, e.get("detector_rev"), otype)):
                    cov = "none"
                if cov in ("qualified", "partial") and masks.get(cp) != "qualified":
                    masks[cp] = cov
    return masks


CHANNEL_NAMES = ("places", "placements", "habits", "negatives", "prior", "doubted")


def where_is(ltm, otype: str, wm=None, e_now: int | None = None, prior: dict | None = None, channels=None) -> dict:
    """`channels=None` (default) is the full A-stage behaviour. A set of CHANNEL_NAMES (B1) is applied BEFORE candidate
    competition, suppression and the top-k cut, so a channel that is off can never displace one that is on:
    places = T0/T2 location claims; doubted = T3b (claims contradicted twice+); placements = T1; habits = T3;
    negatives = cross-task qualified-negative suppression; prior = the fixed T4 guess. Within-task masks (what this
    very task already inspected) are working memory and are always applied."""
    ch = None if channels is None else frozenset(channels)
    if ch is not None and not ch <= set(CHANNEL_NAMES):
        raise ValueError(f"unknown channels {sorted(ch - set(CHANNEL_NAMES))}")

    def on(name):
        return ch is None or name in ch
    p, store, k = ltm.params, ltm.store, ltm.kitchen
    E = ltm.clock() if e_now is None else int(e_now)           # the store clock, or an explicit frozen one - never wall time
    epoch_now = ltm.current_epoch()
    frame_now = ltm.frame_of_epoch(epoch_now)
    where_k = int(p["retrieval"]["where_k"]["value"])

    places_rows = store.select("place_instance", where={"kitchen": k})
    places = {r["place"]: r for r in places_rows}
    alias = {a: c for a, c in alias_map(store).items() if a in places}

    def C(pid):
        return canon_place(alias, pid)

    book = PlaceBook(p, [r for r in places_rows if frame_now is None or r["frame"] == frame_now], alias)
    masks = wm_masks(ltm, wm, otype, book)

    deferred = {cp for cp, st in masks.items() if st == "partial"}

    def _order(c):
        d = 1 if (c["kind"] == "place" and c["id"] in deferred) else 0
        return (TIER_RANK[c["tier"]], d, c["sub"], -c["weight"], c["kind"], id_key(c["id"]) if c["kind"] == "place" else (3, 0),
                str(c["id"]))

    cands: dict = {}                                            # canonical key -> candidate

    def offer(c):
        cur = cands.get(c["key"])
        if cur is None or _order(c) < _order(cur):
            cands[c["key"]] = c

    # -- T0 / T2 / T3b: location claims --------------------------------------------------------------------------
    for r in (store.select("location_claim", where={"kitchen": k, "otype": otype}) if on("places") else []):
        tier, status = location_tier(r, epoch_now, frame_now)
        if tier is None or (tier == "T3b" and not on("doubted")):
            continue
        w = weight(p, "location", r["verified"], E, r["E_support"], r["contradictions"], max(0, epoch_now - r["epoch"]))
        if retired(p, w):
            continue
        cp = C(r["place"])
        rt = places[r["place"]]["rtype"] if r["place"] in places else None
        offer({"key": ("place", cp), "id": cp, "kind": "place", "rtype": rt, "tier": tier, "sub": 0 if r["verified"] == "verified" else 1,
               "source": "location_claim", "age": max(0, E - r["E_support"]), "support": _support_count(store, "location_claim", r["claim"]),
               "contradictions": r["contradictions"], "status": status, "weight": _round(w),
               "provenance": f"{r['provenance']}/{r['verified']}", "E_support": r["E_support"]})
    # -- T1: placements ---------------------------------------------------------------------------------------------
    for r in (store.select("placement", where={"kitchen": k, "otype": otype}) if on("placements") else []):
        if r["status"] != "current" or r["epoch"] < epoch_now:
            continue                                            # earlier-epoch placements are historical
        w = weight(p, "placement", r["provenance"], E, r["E"], 0, 0)
        if retired(p, w):
            continue
        cp = C(r["place"]) if r["place"] else None
        key = ("place", cp) if cp else ("xyz", tuple(round(x, 1) for x in (r["xyz"] or [0, 0, 0])))
        offer({"key": key, "id": cp or "xyz:%.1f,%.1f" % (r["xyz"][0], r["xyz"][2]), "kind": "place" if cp else "xyz",
               "rtype": places[r["place"]]["rtype"] if r["place"] in places else None, "tier": "T1",
               "sub": 0 if r["provenance"] == "verified" else 1, "source": "placement", "age": max(0, E - r["E"]),
               "support": 1, "contradictions": 0, "status": "current", "weight": _round(w),
               "provenance": f"agent_placed/{r['provenance']}", "E_support": r["E"], "xyz": r["xyz"]})

    # -- suppression by current-epoch qualified negatives (a newer positive defeats an older negative) -----------------
    suppressed: dict = {}
    for r in (store.select("negative_evidence", where={"kitchen": k, "otype": otype}) if on("negatives") else []):
        if r["superseded"] or r["epoch"] != epoch_now:
            continue
        w = weight(p, "negative", None, E, r["E"])
        if retired(p, w):
            continue
        cp = C(r["place"])
        cur = cands.get(("place", cp))
        if cur is not None and cur["E_support"] >= r["E"] and cur["source"] in ("location_claim", "placement") \
                and cur["contradictions"] == 0:
            continue                                            # newer (or equal) positive support defeats the negative
        suppressed[cp] = {"place": cp, "reason": "qualified_negative", "age": max(0, E - r["E"]), "weight": _round(w)}
    for cp, st in masks.items():
        if st == "qualified" and cp not in suppressed:
            suppressed[cp] = {"place": cp, "reason": "inspected_this_task", "age": 0, "weight": 1.0}
    for key in [c for c in cands if c[0] == "place" and c[1] in suppressed]:
        del cands[key]

    # -- T3 / T3b-adjacent: habits, T4: prior ---------------------------------------------------------------------------
    collapsed = collapse_trials(store.select("search_trial"), alias)
    prior = default_prior(otype) if prior is None else prior
    prefs = kitchen_prefs(collapsed, p, E, otype, k, prior) if on("habits") else []
    place_rtypes = {}
    for c in cands.values():
        if c["kind"] == "place" and c["rtype"]:
            place_rtypes[c["rtype"]] = place_rtypes.get(c["rtype"], 0) + 1
    known_places = {}
    for pid, r in places.items():
        if C(pid) == pid:
            known_places[r["rtype"]] = known_places.get(r["rtype"], 0) + 1
    suppressed_rt = {}
    for cp in suppressed:
        if cp in places:
            suppressed_rt[places[cp]["rtype"]] = suppressed_rt.get(places[cp]["rtype"], 0) + 1

    def covered(rt):
        """An rtype-level candidate adds nothing when every known place of that type is already listed or suppressed."""
        return known_places.get(rt, 0) > 0 and known_places.get(rt, 0) <= place_rtypes.get(rt, 0) + suppressed_rt.get(rt, 0)

    seen_rt = set()
    for r in prefs:
        if covered(r["rtype"]):
            continue
        seen_rt.add(r["rtype"])
        cands[("rtype", r["rtype"])] = {"key": ("rtype", r["rtype"]), "id": r["rtype"], "kind": "rtype", "rtype": r["rtype"],
                                        "tier": "T3", "sub": 0, "source": "habit", "age": None,
                                        "support": r["n_pos"] + r["n_neg"], "contradictions": 0, "status": "empirical",
                                        "weight": _round(r["pref"]), "provenance": "kitchen_habit",
                                        "E_support": None}
    gen = None
    if on("habits") and bool(p["habits"]["general_active"]["value"]):
        gen = general_habit(collapsed, p, E, otype, prior=prior)
        if gen["active"]:
            for r in gen["prefs"]:
                if r["rtype"] in seen_rt or covered(r["rtype"]):
                    continue
                seen_rt.add(r["rtype"])
                cands[("rtype", r["rtype"])] = {"key": ("rtype", r["rtype"]), "id": r["rtype"], "kind": "rtype",
                                                "rtype": r["rtype"], "tier": "T3", "sub": 1, "source": "general_habit",
                                                "age": None, "support": r["kitchens"], "contradictions": 0,
                                                "status": "empirical", "weight": _round(r["pref"]),
                                                "provenance": "general_habit", "E_support": None}
    for rt in (sorted(prior, key=lambda r: (-prior[r], r)) if on("prior") else []):
        rt = {"SinkBasin": "Sink"}.get(rt, rt)
        if rt in seen_rt or ("rtype", rt) in cands or covered(rt):
            continue
        cands[("rtype", rt)] = {"key": ("rtype", rt), "id": rt, "kind": "rtype", "rtype": rt, "tier": "T4", "sub": 0,
                                "source": "prior", "age": None, "support": 0, "contradictions": 0, "status": "declared",
                                "weight": _round(p0_for(prior, rt, p)), "provenance": "fixed_prior", "E_support": None}

    # -- order, cut, EXPLORE -----------------------------------------------------------------------------------------------
    ranked = sorted(cands.values(), key=_order)
    cut = ranked[:where_k]
    out = []
    for c in cut:
        c = {kk: vv for kk, vv in c.items() if kk not in ("key", "sub", "E_support")}
        if c["kind"] == "place" and c["id"] in deferred:
            c["status"] += "+deferred"
        out.append(c)
    explore = {"id": "EXPLORE", "tier": "EXPLORE", "source": "explore", "unknown": True,
               "suppressed": sorted(suppressed.values(), key=lambda s: (id_key(s["place"]), s["reason"])),
               "deferred": sorted(deferred, key=id_key), "n_beyond_cut": max(0, len(ranked) - where_k)}
    return {"otype": otype, "kitchen": k, "e_now": E, "epoch": epoch_now, "candidates": out, "explore": explore}


def reorder_within(answer: dict, proposed_ids: list) -> list:
    """Gemma ranks, never routes: honour a proposed order only inside the T2-T4 block and only for ids in the answer.
    Anything else in `proposed_ids` is ignored; missing ids keep their original position."""
    cand = answer["candidates"]
    ids = [c["id"] for c in cand]
    movable = [c["id"] for c in cand if c["tier"] in REORDERABLE]
    want = [i for i in proposed_ids if i in movable]
    want += [i for i in movable if i not in want]
    it = iter(want)
    return [next(it) if c["tier"] in REORDERABLE else c["id"] for c in cand] if ids else []
