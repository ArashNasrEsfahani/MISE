"""Systematic scene search — the strategy behind "find the apple".

Before this module, "find the apple" produced a one-step plan (`look_around`) that spun the agent on the spot
and declared success whether or not an apple was ever seen. That is not a search: it never goes anywhere, it
never looks *inside* anything, and it cannot fail. What a search needs is a **policy over where to look next**,
and that policy is what lives here.

The policy is deliberately a PURE function of what the agent knows:

    remembered belief   (memory.query_location — "I saw an Apple on the CounterTop, conf 0.72")
  + learned facts       (knowledge base — "apples are usually in the fridge")
  + a curated prior     (RECEPTACLE_PRIOR, below — used only where the first two are silent)
  + what is in reach    (receptacle types actually present, with their distances)
  + the discovered map  (navmap free cells, to cover the room when the above are exhausted)
  -> an ORDERED list of stops, each with a reason in words.

Keeping it pure is what makes it testable without a GPU (`sim/_search_selftest.py`), and keeping the reason
attached to every stop is what lets the console narrate the search instead of just performing it.

Nothing here reads AI2-THOR's ground-truth containment (`receptacleObjectIds`) for the target. That would be
looking up the answer and then walking to it, which is not a search and would make the perception ablation
meaningless. Receptacle POSITIONS come from the simulator — as they do for every other skill in this system,
since `walk_to` needs a nav target — but *whether the apple is in the fridge* is only ever established by
looking. See `docs/live-view.md §2` for the same observation/execution split applied to perception.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------
# Containment prior — where kitchen things usually live.
#
# This is a FALLBACK, not the system's knowledge: a remembered observation and a learned KB fact both outrank it
# (see `rank_stops`). It exists so a cold agent in an unseen kitchen still searches sensibly instead of walking a
# lawnmower pattern. Weights are rough plausibilities in [0, 1], not probabilities.
#
# Keys are AI2-THOR objectTypes, which are language-neutral identifiers — the same property the FA->EN pivot in
# agentic_rag.py relies on, so a Persian command reaches the same priors as an English one.
# ----------------------------------------------------------------------------------------------------------------
_FOOD = {"Fridge": 0.90, "CounterTop": 0.70, "Bowl": 0.50, "Plate": 0.40, "Sink": 0.30,
         "Microwave": 0.25, "Cabinet": 0.20, "DiningTable": 0.45, "GarbageCan": 0.10, "Shelf": 0.30}
_CUTLERY = {"Drawer": 0.85, "CounterTop": 0.50, "Sink": 0.40, "DiningTable": 0.35, "Cabinet": 0.30}
_VESSEL = {"Cabinet": 0.70, "CounterTop": 0.60, "Sink": 0.50, "DiningTable": 0.40, "Fridge": 0.15,
           "Shelf": 0.55}
_COOKWARE = {"Cabinet": 0.70, "StoveBurner": 0.60, "CounterTop": 0.50, "Sink": 0.40}

RECEPTACLE_PRIOR: dict = {}
for _t in ("Apple", "Tomato", "Lettuce", "Potato", "Bread", "Egg", "AppleSliced", "TomatoSliced",
           "LettuceSliced", "PotatoSliced", "BreadSliced"):
    RECEPTACLE_PRIOR[_t] = dict(_FOOD)
for _t in ("Knife", "ButterKnife", "Fork", "Spoon", "Spatula", "Ladle", "PepperShaker", "SaltShaker"):
    RECEPTACLE_PRIOR[_t] = dict(_CUTLERY)
for _t in ("Mug", "Cup", "Bowl", "Plate", "Bottle", "WineBottle"):
    RECEPTACLE_PRIOR[_t] = dict(_VESSEL)
for _t in ("Pot", "Pan", "Kettle"):
    RECEPTACLE_PRIOR[_t] = dict(_COOKWARE)
RECEPTACLE_PRIOR["Mug"]["CoffeeMachine"] = 0.55
RECEPTACLE_PRIOR["Egg"]["Fridge"] = 0.95
RECEPTACLE_PRIOR["Bottle"]["Fridge"] = 0.70
RECEPTACLE_PRIOR["WineBottle"]["Fridge"] = 0.55

# Used when the target type is unknown to the prior. Broad and shallow on purpose: it should not out-argue a
# remembered observation, only beat walking at random.
GENERIC_PRIOR = {"CounterTop": 0.50, "DiningTable": 0.40, "Cabinet": 0.40, "Fridge": 0.30, "Drawer": 0.30,
                 "Sink": 0.25, "Shelf": 0.30, "Microwave": 0.15, "GarbageCan": 0.10}

#: AI2-THOR ships BOTH `Sink` (the fixture) and `SinkBasin` (the interactable basin things actually rest in),
#: and the scene reports the basin. A prior keyed on "Sink" therefore matched nothing at all -- the sink was
#: silently missing from every search route until this was noticed in a live run.
RECEPTACLE_ALIASES: dict = {"SinkBasin": "Sink"}

# Receptacles you have to OPEN before their contents are visible. Searching one of these without opening it is
# the single most common way a search misses something that was right there.
OPENABLE = frozenset(("Fridge", "Cabinet", "Drawer", "Microwave", "Safe", "Box", "Kettle"))


def normalize_type(t) -> str:
    """AI2-THOR objectType from a raw string: strips an objectId suffix and trims. Case is preserved because the
    type names are proper identifiers ('CounterTop', not 'countertop')."""
    s = str(t or "").strip()
    if "|" in s:
        s = s.split("|", 1)[0]
    return s


def prior_for(object_type: str) -> dict:
    """Containment prior for a target type: the curated table, else the generic fallback."""
    t = normalize_type(object_type)
    if t in RECEPTACLE_PRIOR:
        return dict(RECEPTACLE_PRIOR[t])
    # a sliced/cracked variant searches like its base type ("AppleSliced" -> "Apple")
    for suffix in ("Sliced", "Cracked", "Cooked"):
        if t.endswith(suffix) and t[: -len(suffix)] in RECEPTACLE_PRIOR:
            return dict(RECEPTACLE_PRIOR[t[: -len(suffix)]])
    return dict(GENERIC_PRIOR)


def surface_forms(t) -> list:
    """Ways a CamelCase objectType is actually written in free text: ``["countertop", "counter top"]``.

    KB entries are written by humans and by the self-improving loop, so "garbage can" and "counter top" appear
    spaced far more often than the identifier is. Matching only the identifier made `kb_affinity` silently miss
    every fact of the form "apples are kept in the garbage can" — i.e. exactly the ones it exists to catch.
    """
    import re
    t = normalize_type(t)
    if not t:
        return []
    low = t.lower()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", t).lower()
    return [low] if spaced == low else [low, spaced]


#: Phrases that make a knowledge-base sentence a NEGATIVE result. Without this the loop eats itself: a failed
#: search writes "WineBottle was NOT at: Cabinet, Shelf, Fridge", `kb_affinity` sees the target and three
#: receptacle names in one sentence, and the NEXT search is steered straight back to the three places already
#: ruled out -- with the confidence of learned knowledge. Observed live on 2026-08-22.
_NEGATIVE_MARKERS: tuple = (
    "not at", "not found", "not in", "not on", "was not", "were not", "wasn't", "weren't", "isn't", "aren't",
    "no longer", "could not find", "couldn't find", "failed to find", "did not find", "didn't find",
    "نبود", "پیدا نشد", "نیست", "یافت نشد",
)


def is_negative(text) -> bool:
    """True when a KB sentence records an ABSENCE rather than a location."""
    low = str(text or "").lower()
    return any(m in low for m in _NEGATIVE_MARKERS)


def kb_exclusions(object_type: str, kb_texts, receptacle_types) -> set:
    """Receptacles the knowledge base says the object was NOT at, as a set of types.

    These are demoted rather than removed in :func:`rank_stops`: "it wasn't in the fridge last time" is good
    evidence about where to look *first*, not proof it can never be there — someone may have put it back.
    """
    tforms = surface_forms(object_type)
    out: set = set()
    if not tforms:
        return out
    mentions = lambda low, forms: any(f and (f in low or (f + "s") in low) for f in forms)
    for text in (kb_texts or []):
        low = str(text or "").lower()
        if not is_negative(low) or not mentions(low, tforms):
            continue
        for r in (receptacle_types or []):
            name = normalize_type(r)
            if name and mentions(low, surface_forms(name)):
                out.add(name)
    return out


def kb_affinity(object_type: str, kb_texts, receptacle_types) -> dict:
    """Receptacle weights LEARNED from the knowledge base, as ``{receptacle_type: weight}``.

    A KB entry counts when it mentions both the target type and a receptacle type — e.g. the fact
    "mugs can be inside cabinets" yields ``{'Cabinet': ...}`` for a Mug search. Crude on purpose: the KB is
    free text written by the loop and by the user, so anything cleverer would be guessing at grammar. What
    matters is that a fact the agent actually learned OUTRANKS the hard-coded prior, which is the whole point
    of having a knowledge base.

    Matching is case-insensitive and tolerates a trailing plural 's', so "apples"/"cabinets" both hit.
    """
    tforms = surface_forms(object_type)
    out: dict = {}
    if not tforms:
        return out
    mentions = lambda low, forms: any(f and (f in low or (f + "s") in low) for f in forms)
    for text in (kb_texts or []):
        low = str(text or "").lower()
        # A sentence recording an ABSENCE names the target and a receptacle exactly like a positive one does.
        # Reading it as affinity is how a failed search teaches the next search to repeat it.
        if is_negative(low) or not mentions(low, tforms):
            continue
        for r in (receptacle_types or []):
            name = normalize_type(r)
            if name and mentions(low, surface_forms(name)):
                out[name] = max(out.get(name, 0.0), 0.95)
    return out


def sweep_schedule(turn_deg: float, detect_every_deg: float, total_deg: float = 360.0) -> list:
    """The rotation/detection schedule for one look-around, as ``[(degrees, detect?), ...]``.

    This exists because the first version of the perceived-mode sweep called ``look_around(turn=45)`` so that
    YOLO ran only 8 times — and thereby made the agent spin a full circle in **8 rendered frames**, roughly a
    fifth of a second. The most important moment in the whole feature was over before you could see it.

    Rotation granularity and detection cadence are now independent: turn in small (visible) steps, run the
    detector only every ``detect_every_deg``. A 360 deg sweep at 8 deg/step with detection every 45 deg is
    45 frames and 8 inferences — watchable AND cheap.
    """
    turn = max(1.0, float(turn_deg))
    every = max(turn, float(detect_every_deg))
    steps = max(1, int(round(float(total_deg) / turn)))
    out, since = [], 0.0
    for i in range(steps):
        since += turn
        detect = since + 1e-9 >= every or i == steps - 1
        out.append((turn, detect))
        if detect:
            since = 0.0
    return out


#: How thorough a search is, per motion profile. Coupled to motion on purpose: at `fast` the operator wants the
#: answer, so look in fewer places; at `cinematic` they want to watch it happen, so look in more. `detect_every`
#: does NOT move with them — it is bounded by inference cost, not by how watchable the spin is, and keeping the
#: two independent is the entire point of :func:`sweep_schedule`.
SEARCH_BUDGETS: dict = {
    "fast":      {"max_stops": 4, "frontier_stops": 1, "detect_every": 45.0, "max_seconds": 45.0},
    "normal":    {"max_stops": 6, "frontier_stops": 2, "detect_every": 45.0, "max_seconds": 90.0},
    "cinematic": {"max_stops": 8, "frontier_stops": 3, "detect_every": 45.0, "max_seconds": 180.0},
}


def search_budget(speed: str = "normal") -> dict:
    """Search budget for a motion profile name. Unknown names fall back to `normal` rather than raising —
    a mistyped speed should search normally, not abort the command."""
    return dict(SEARCH_BUDGETS.get(str(speed or "").strip(), SEARCH_BUDGETS["normal"]))


def not_found_lesson(object_type: str, scene: str | None, visited) -> dict:
    """The lesson a FAILED search writes back, as ``{'problem', 'lesson', 'places'}``.

    A search that finds nothing is evidence, not a non-event: it rules places out. Recording *where the agent
    already looked* is what lets the next search start somewhere else — the self-improving loop applied to
    perception rather than to plans. It is also the honest counterpart to the bug this replaced, where a search
    that saw nothing was stored as a proven procedure.
    """
    places = [str(v) for v in (visited or []) if v]
    where = ", ".join(places) if places else "nowhere reachable"
    return {
        "problem": f"searched for {object_type} in {scene or 'this scene'} and did not find it",
        "lesson": (f"{object_type} was NOT at: {where} (scene {scene or '?'}). "
                   f"Try somewhere else first next time."),
        "places": places,
    }


def _dist(a, b) -> float:
    if not a or not b:
        return 0.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def rank_stops(object_type: str, *, belief=None, kb_texts=(), receptacles=(), agent_pos=None,
               nav_cells=(), max_stops: int = 6, frontier_stops: int = 2) -> list:
    """Ordered places to go and look for ``object_type``. Each stop carries WHY it is on the list.

    ``receptacles`` are the candidate containers/surfaces actually present, as
    ``[{'objectId','objectType','pos':(x,z),'isOpen':bool,'openable':bool}, ...]``.
    ``belief`` is ``memory.query_location``'s answer (or None). ``kb_texts`` are retrieved KB entry texts.
    ``nav_cells`` are discovered-free map cells ``[(x, z), ...]`` used only to cover the room once the
    knowledge-driven stops run out.

    Order of authority, highest first — this ordering IS the thesis claim that retrieval changes behaviour:

      1. ``recall``     — somewhere this exact type was actually observed before;
      2. ``receptacle`` — a container the KB associates with it, else the curated prior, scaled by proximity;
      3. ``frontier``   — unexplored-ish free cells, so the room still gets covered when knowledge is silent.

    A stop's ``open`` flag marks a closed openable that must be opened before looking inside.
    """
    t = normalize_type(object_type)
    stops: list = []

    # 0. Always look from where you already stand: it costs one sweep and often ends the search immediately.
    stops.append({"kind": "here", "target": None, "type": None, "pos": None, "open": False,
                  "score": 1e9, "why": "look around from here first"})

    recs = [r for r in (receptacles or []) if r and r.get("objectId")]
    by_type: dict = {}
    for r in recs:
        by_type.setdefault(normalize_type(r.get("objectType")), []).append(r)

    # 1. Remembered observation — the strongest signal there is, decayed by the memory layer already.
    seen_ids = set()
    rec_name = normalize_type((belief or {}).get("receptacle")) if belief else ""
    if belief and rec_name and rec_name in by_type:
        conf = float(belief.get("confidence") or 0.0)
        cand = min(by_type[rec_name], key=lambda r: _dist(agent_pos, r.get("pos")))
        seen_ids.add(cand["objectId"])
        stops.append({
            "kind": "recall", "target": cand["objectId"], "type": rec_name, "pos": cand.get("pos"),
            "open": bool(cand.get("openable") and not cand.get("isOpen")),
            "score": 100.0 + conf,
            "why": f"remembered: {t} was last seen at {rec_name} (confidence {conf:.2f})"})

    # 2. Knowledge-driven receptacles: KB first, curated prior second, proximity as the tie-break.
    learned = kb_affinity(t, kb_texts, list(by_type.keys()))
    ruled_out = kb_exclusions(t, kb_texts, list(by_type.keys()))
    prior = prior_for(t)
    for rtype, group in by_type.items():
        w = learned.get(rtype, prior.get(RECEPTACLE_ALIASES.get(rtype, rtype), 0.0))
        if w <= 0.0:
            continue
        cand = min(group, key=lambda r: _dist(agent_pos, r.get("pos")))
        if cand["objectId"] in seen_ids:
            continue
        seen_ids.add(cand["objectId"])
        d = _dist(agent_pos, cand.get("pos"))
        why = (f"the knowledge base says {t} goes with {rtype}" if rtype in learned
               else f"{t} is usually at/in a {rtype}")
        penalty = 0.0
        if rtype in ruled_out:
            # Demoted, not dropped: someone may have put it back since.
            penalty = 6.0
            why = f"{t} was NOT here last time - checking again only after everywhere else"
        stops.append({
            "kind": "receptacle", "target": cand["objectId"], "type": rtype, "pos": cand.get("pos"),
            "open": bool(cand.get("openable") and not cand.get("isOpen")),
            "score": 10.0 * w - 0.25 * d - penalty,
            "why": why + (" (closed — will open it)" if (cand.get("openable") and not cand.get("isOpen")) else "")})

    stops = [stops[0]] + sorted(stops[1:], key=lambda s: -s["score"])
    stops = stops[:max_stops]

    # 3. Frontier coverage — farthest-first over discovered free cells, so the sweep spreads out instead of
    #    re-looking at the same corner. Only appended when knowledge did not fill the budget.
    room = max(0, max_stops - len(stops))
    if room and nav_cells:
        chosen: list = []
        picked_from = [agent_pos] if agent_pos else []
        pool = [tuple(c) for c in nav_cells]
        for _ in range(min(frontier_stops, room)):
            if not pool:
                break
            ref = picked_from or [(0.0, 0.0)]
            best = max(pool, key=lambda c: min(_dist(c, p) for p in ref))
            pool = [c for c in pool if _dist(c, best) > 1.0]
            picked_from.append(best)
            chosen.append(best)
        for i, c in enumerate(chosen):
            stops.append({"kind": "frontier", "target": None, "type": None, "pos": c, "open": False,
                          "score": -1.0 - i, "why": "sweeping an unsearched part of the room"})
    return stops


def summarize(stops) -> str:
    """One-line plan of the search, for the transcript: 'plan: here -> Fridge -> CounterTop -> +1 sweep'."""
    parts = []
    for s in stops or []:
        if s["kind"] == "here":
            parts.append("here")
        elif s["kind"] == "frontier":
            parts.append("sweep")
        else:
            parts.append(str(s.get("type") or "?") + ("*" if s.get("open") else ""))
    return " -> ".join(parts) if parts else "(nowhere to look)"
