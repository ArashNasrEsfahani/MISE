"""Where to stand, what to open, and where to point the camera when building a scene's detector dataset.

`scan_scene` used to photograph a kitchen the way a tourist photographs a room: teleport to two dozen random
floor points, face north/east/south/west, look level and slightly down, take eight pictures, leave. Every frame
it produced was therefore eye-level, axis-aligned and mid-distance — and every cupboard, drawer, fridge and
microwave in the room was SHUT for all of them. An apple that lives in the fridge was never photographed once,
so the detector was asked to learn a class from zero examples, and a detector asked to do that reports whatever
the loss finds convenient. That is the mechanism behind "the models are dogshit at detecting".

This module is the policy half of the fix — which containers are worth opening, how many frames each one
deserves, where to stand, and which way to point the camera once the door is open. It is a pure function of
AI2-THOR object metadata plus the per-class frame counts collected so far: no controller, no numpy, no GPU, so
every decision is testable headlessly (`sim/_scan_selftest.py`). `ThorAgent.peek_inside` and
`ThorAgent.scan_scene` supply the embodiment — navigation, the door swing, the capture, and putting the room
back the way it was found.

Unlike `sim/search.py`, this module DOES read ground-truth containment (`receptacleObjectIds`). That is not a
double standard. A search must not look up the answer it is pretending to hunt for, or the perception ablation
measures nothing; a dataset scan is an oracle-labelled procedure from end to end — `gen_dataset.capture_frame`
already draws its boxes from AI2-THOR's own instance masks. Knowing that four items sit inside this fridge is
how the scan decides the fridge is worth twelve frames and the empty cupboard beside it is worth one.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------------------------------------------------------
# Camera geometry.
# ----------------------------------------------------------------------------------------------------------------
#: Camera height above the agent's reported position, per stance (metres). These mirror the fallback in
#: `ThorAgent.look_at_object`, which is where they came from: AI2-THOR puts the standing eye at ~1.576 m in a
#: kitchen whose floor-standing agent reports y ~= 0.901.
EYE_STANDING = 0.675
EYE_CROUCHED = 0.40

#: The agent's pitch limits. The planner has to know about them because the simulator clamps silently: a drawer
#: whose interior sits at 0.4 m cannot be looked INTO from 0.5 m away while standing — that needs ~67 degrees of
#: down-pitch and the agent has 60 — which is exactly why :func:`stance_for` crouches instead of shrugging.
HORIZON_MIN = -30.0
HORIZON_MAX = 60.0

#: How close to a pitch limit still counts as "cannot really see into it" when :func:`stance_for` picks a stance.
#: A drawer photographed at 59 degrees of down-pitch is technically in frame and practically a picture of the
#: drawer's front edge; crouching costs nothing and buys back the angle. The hard limits above still govern the
#: clamp — this margin only decides which stance is PREFERRED.
COMFORT_MARGIN = 8.0


def clamp_horizon(h: float) -> float:
    """Clamp a camera pitch to what the agent can physically adopt."""
    return HORIZON_MIN if h < HORIZON_MIN else (HORIZON_MAX if h > HORIZON_MAX else float(h))


def eye_of(agent_pos: dict, standing: bool = True) -> dict:
    """The camera position for an agent standing (or crouching) at `agent_pos`."""
    return {"x": float(agent_pos["x"]), "z": float(agent_pos["z"]),
            "y": float(agent_pos["y"]) + (EYE_STANDING if standing else EYE_CROUCHED)}


def _pitch_raw(target: dict, eye: dict) -> float:
    """Un-clamped camera pitch onto `target` (AI2-THOR convention: + looks DOWN, - looks UP)."""
    d = math.hypot(float(target["x"]) - eye["x"], float(target["z"]) - eye["z"])
    dy = float(target["y"]) - eye["y"]
    if d < 1e-6:                                    # standing on top of it: look as far up/down as we can
        return HORIZON_MAX if dy < 0 else HORIZON_MIN
    return -math.degrees(math.atan2(dy, d))


def pitch_for(target: dict, eye: dict) -> float:
    """Camera pitch that centres world point `target` from `eye`, clamped to the agent's limits."""
    return clamp_horizon(_pitch_raw(target, eye))


def yaw_for(target: dict, eye: dict) -> float:
    """Camera yaw (degrees, AI2-THOR's compass) that faces world point `target` from `eye`."""
    return math.degrees(math.atan2(float(target["x"]) - eye["x"], float(target["z"]) - eye["z"])) % 360.0


def stance_for(target: dict, agent_pos: dict, prefer_standing: bool = True) -> bool:
    """Stand or crouch to photograph `target` — whichever stance can actually point at it.

    Crouching lowers the eye by ~0.28 m, which SHALLOWS the angle down into a low drawer and STEEPENS it up at a
    wall cabinet. So the rule is not "crouch for small things", it is "adopt the stance that can comfortably
    point at this", and only then prefer standing. A drawer photographed from a standing pose with the pitch
    pinned at its limit shows the front of the drawer and none of its contents.
    """
    stand_err = _overshoot(_pitch_raw(target, eye_of(agent_pos, True)), COMFORT_MARGIN)
    crouch_err = _overshoot(_pitch_raw(target, eye_of(agent_pos, False)), COMFORT_MARGIN)
    if abs(stand_err - crouch_err) < 1e-6:
        return bool(prefer_standing)
    return stand_err < crouch_err


def _overshoot(pitch: float, margin: float = 0.0) -> float:
    """How far a requested pitch falls outside the agent's (optionally margin-shrunk) range; 0.0 if comfortable."""
    return max(0.0, pitch - (HORIZON_MAX - margin)) + max(0.0, (HORIZON_MIN + margin) - pitch)


def aim_from(target: dict, agent_pos: dict, prefer_standing: bool = True):
    """``(yaw, horizon, standing)`` that centres world point `target` from an agent at `agent_pos`."""
    standing = stance_for(target, agent_pos, prefer_standing=prefer_standing)
    return (yaw_for(target, eye_of(agent_pos, standing)),
            pitch_for(target, eye_of(agent_pos, standing)),
            standing)


def aim_point(container: dict, contents=()) -> dict:
    """The world point to centre when photographing a container's INTERIOR.

    Aiming at the container is what a naive implementation does, and for a fridge that means aiming at the
    middle of the door — the contents sit behind and below it, half out of frame. When the scan knows what is
    inside (it does; see the module header on why reading containment is legitimate here) the honest target is
    the centroid of the things it is trying to photograph. Falls back to the object's bounding-box centre when
    the container is empty, which is the right answer for a frame whose subject IS the open door.
    """
    pts = [c.get("position") for c in (contents or []) if isinstance(c, dict) and (c.get("position") or {}).get("x") is not None]
    if pts:
        n = float(len(pts))
        return {"x": sum(p["x"] for p in pts) / n,
                "y": sum(p["y"] for p in pts) / n,
                "z": sum(p["z"] for p in pts) / n}
    box = (container.get("axisAlignedBoundingBox") or {}).get("center")
    return dict(box or container.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0})


# ----------------------------------------------------------------------------------------------------------------
# Viewpoint variety.
# ----------------------------------------------------------------------------------------------------------------
def view_offsets(n: int, yaw_spread: float = 16.0, pitch_spread: float = 8.0) -> list:
    """`n` ``(yaw_delta, pitch_delta)`` nudges around a centred aim, dead-on first.

    A detector trained only on frames where the subject sits dead centre, whole and unoccluded, learns to need
    exactly that. The offsets push the subject off-centre and let the frame crop it, which is what every real
    view of a cupboard shelf looks like. Deterministic and symmetric so two scans of one scene plan the same
    shots.
    """
    out = [(0.0, 0.0)]
    for i in range(1, max(1, int(n))):
        ring = (i + 1) // 2                          # 1, 1, 2, 2, 3, 3 ...
        sign = 1.0 if i % 2 else -1.0
        out.append((sign * float(yaw_spread) * ring,
                    sign * float(pitch_spread) * (1.0 if ring % 2 else -1.0)))
    return out


def openness_ladder(n_levels: int = 2, min_open: float = 0.45) -> list:
    """Door positions to photograph, widest first: ``[1.0]``, then progressively less open.

    A half-open door is a genuinely different picture: it occludes part of the interior, changes the lighting
    inside, and gives the container's own ``_open`` class something other than one canonical pose to learn. It
    is also safe to label — AI2-THOR's ``visible`` flag accounts for occlusion, so an item hidden behind a
    part-shut door is simply left out of the labels rather than being labelled and then not there.
    """
    n = max(1, int(n_levels))
    if n == 1:
        return [1.0]
    lo = max(0.05, min(0.95, float(min_open)))
    return [round(1.0 - (1.0 - lo) * i / (n - 1), 3) for i in range(n)]


#: Distance bands (metres from the subject) a scan should cover: close enough to fill the frame, mid, and far
#: enough to include context. Bounded above by AI2-THOR's default visibility gate, since a frame captured beyond
#: it arrives with the subject present and UNLABELLED — see `scan_scene`'s note on `visibility_distance`.
BANDS = (0.55, 0.95, 1.45)


def select_poses(poses, target_xz, max_poses: int = 2, bands=BANDS, max_distance=None) -> list:
    """A DIVERSE handful of interactable poses: different distances, different angles, both stances.

    ``GetInteractablePoses`` hands back dozens of poses in navmesh order, and every caller in this codebase took
    ``poses[0]``. Every frame captured that way is the same photograph from the same spot, which is how a
    dataset ends up with four hundred images and one viewpoint. This buckets the poses by (distance band, 45°
    yaw bucket, stance) and takes one per distance band first, so two shots means near AND far rather than two
    that are twenty centimetres apart.
    """
    want = max(1, int(max_poses))
    bands = tuple(bands) or BANDS
    cand = []
    for p in (poses or []):
        try:
            x, z = float(p["x"]), float(p["z"])
        except (KeyError, TypeError, ValueError):
            continue
        d = math.hypot(x - float(target_xz[0]), z - float(target_xz[1]))
        if max_distance is not None and d > float(max_distance):
            continue
        band = min(range(len(bands)), key=lambda i: abs(d - bands[i]))
        yaw_b = int((float(p.get("rotation") or 0.0) % 360.0) // 45.0)
        cand.append((band, yaw_b, bool(p.get("standing", True)), d, p))
    if not cand:
        return list(poses or [])[:want]

    best = {}
    for band, yaw_b, standing, d, p in cand:
        key = (band, yaw_b, standing)
        if key not in best or d < best[key][0]:
            best[key] = (d, p)
    ordered = sorted(best.items(), key=lambda kv: (kv[0][0], kv[1][0], kv[0][1]))

    picked, used_bands = [], set()
    for key, (_d, p) in ordered:                     # one per distance band first — variety of distance is the point
        if key[0] in used_bands:
            continue
        used_bands.add(key[0])
        picked.append(p)
        if len(picked) >= want:
            return picked
    for _key, (_d, p) in ordered:                    # then fill from whatever is left, nearest first
        if any(p is q for q in picked):
            continue
        picked.append(p)
        if len(picked) >= want:
            break
    return picked


#: Horizons the coverage sweep cycles through. The old scan used ``(0, 30)`` — level and slightly down — so it
#: never once looked UP, and wall cabinets (the top half of every kitchen) were chronically under-photographed.
COVERAGE_HORIZONS = (-15.0, 0.0, 20.0, 45.0)

#: How many yaw phases the coverage sweep precesses through. With 4 compass yaws and 4 phases a scan visits 16
#: distinct headings across the room instead of the same 4 at every single position.
YAW_PHASES = 4


def coverage_views(index: int, n_yaws: int = 4, horizons=COVERAGE_HORIZONS, per_yaw: int = 2) -> list:
    """``(yaw, horizon, standing)`` triples to capture from coverage position `index`.

    The frame COUNT per position is unchanged (``n_yaws * per_yaw`` = 8 by default). What changes is that the
    other dimensions vary ACROSS positions instead of being nailed to the same four compass points, the same two
    horizons and the same eye height at every one of them — so the same budget buys 16 headings, 4 pitches and
    both stances over a scan rather than 4/2/1. Deterministic in `index`, so a rescan reproduces the route.
    """
    i0 = int(index)
    n = max(1, int(n_yaws))
    hs = list(horizons) or [0.0]
    step = 360.0 / n
    phase = (i0 % YAW_PHASES) * step / YAW_PHASES
    out = []
    for s in range(n):
        yaw = (s * step + phase) % 360.0
        for v in range(max(1, int(per_yaw))):
            hz = float(hs[(i0 + s + 2 * v) % len(hs)])
            # ~2 standing frames per crouched one: most of a kitchen is seen from eye height, but low shelves
            # and open drawers are only ever seen properly from a low one.
            standing = ((i0 + s + v) % 3) != 2
            out.append((round(yaw, 2), hz, bool(standing)))
    return out


def spread_positions(positions, k: int) -> list:
    """`k` reachable points SPREAD across the room (farthest-point sampling), not `k` drawn at random.

    ``random.sample`` over reachable positions clusters: reachable points are dense wherever the floor is open,
    so a uniform draw spends most of its budget in the middle of the room and can miss a whole corner. Farthest-
    point sampling gives the same count with guaranteed spatial coverage, and being deterministic it also makes
    a rescan comparable with the scan before it.
    """
    pts = [p for p in (positions or []) if isinstance(p, dict) and p.get("x") is not None and p.get("z") is not None]
    k = max(0, int(k))
    if not pts or not k:
        return []
    if k >= len(pts):
        return list(pts)
    cx = sum(p["x"] for p in pts) / len(pts)
    cz = sum(p["z"] for p in pts) / len(pts)
    first = min(pts, key=lambda p: ((p["x"] - cx) ** 2 + (p["z"] - cz) ** 2, p["x"], p["z"]))
    chosen = [first]
    rest = [p for p in pts if p is not first]
    dmin = [(p["x"] - first["x"]) ** 2 + (p["z"] - first["z"]) ** 2 for p in rest]
    while len(chosen) < k and rest:
        j = max(range(len(rest)), key=lambda i: (dmin[i], rest[i]["x"], rest[i]["z"]))
        pick = rest.pop(j)
        dmin.pop(j)
        chosen.append(pick)
        for i, p in enumerate(rest):
            d = (p["x"] - pick["x"]) ** 2 + (p["z"] - pick["z"]) ** 2
            if d < dmin[i]:
                dmin[i] = d
    return chosen


# ----------------------------------------------------------------------------------------------------------------
# Which containers to open, and how much film to spend on each.
# ----------------------------------------------------------------------------------------------------------------
def _dist(a, b) -> float:
    if not a or not b:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rank_containers(objects, *, agent_pos=None, counts=None, per_class_target: int = 40,
                    max_containers: int = 10, empty_per_type: int = 2, class_of=None) -> list:
    """Openable receptacles worth opening, best first, each carrying WHY it is on the list.

    ``objects`` is AI2-THOR's object metadata; ``counts`` is the scan's ``{class_name: frames_so_far}`` tally,
    which is what turns this from a fixed route into a response to the gap actually left in the dataset.
    ``class_of(obj) -> class_name`` supplies the stateful class naming (``perception.class_name``); it is
    injected rather than imported so this module stays dependency-free and testable on its own.

    The ranking says, in order: a class with ZERO frames is an emergency (it is the case the user hit — an
    object that only ever exists inside a shut cupboard is a class the detector was asked to learn from nothing);
    a class under target is worth topping up; more items behind one door is more value per door; a container
    whose own ``_open`` class has never been photographed is worth the trip on its own; and walking further
    costs something.
    """
    counts = dict(counts or {})
    name_of = class_of or (lambda o: o.get("objectType") or "")
    by_id = {o.get("objectId"): o for o in (objects or []) if o.get("objectId")}

    scored = []
    for o in (objects or []):
        if not o.get("openable"):
            continue
        pos = o.get("position") or {}
        if pos.get("x") is None:
            continue
        otype = o.get("objectType") or "container"
        contents = [by_id[i] for i in (o.get("receptacleObjectIds") or []) if i in by_id]
        classes = [name_of(c) for c in contents]
        new = sorted({c for c in classes if c and counts.get(c, 0) <= 0})
        short = sorted({c for c in classes if c and 0 < counts.get(c, 0) < per_class_target})
        d = _dist((pos["x"], pos["z"]), agent_pos)
        # The class this container will WEAR once its door is open — photographing `Fridge_open` is only
        # possible while it is open, so a scan that never opens one never captures the class at all.
        own_open = name_of(dict(o, isOpen=True))
        own_new = counts.get(own_open, 0) <= 0
        score = (3.0 * len(new) + 1.0 * len(short) + 0.4 * min(len(contents), 6)
                 + (1.5 if own_new else 0.0) - 0.20 * d)

        if new:
            why = (f"the {otype} hides {len(contents)} item(s), {len(new)} of a kind the dataset has never seen "
                   f"({', '.join(new[:3])})")
        elif short:
            why = f"the {otype} holds {len(contents)} item(s) still short of target"
        elif contents:
            why = f"the {otype} holds {len(contents)} item(s), already well covered"
        else:
            why = f"the {otype} is empty — worth one frame for its open door"
        scored.append({
            "objectId": o.get("objectId"), "objectType": otype, "pos": (pos["x"], pos["z"]),
            "n_contents": len(contents), "new_classes": new, "short_classes": short,
            "open_class": own_open, "was_open": bool(o.get("isOpen")),
            "distance": round(d, 3), "score": round(score, 4), "why": why,
        })

    scored.sort(key=lambda s: (-s["score"], s["distance"], s["objectId"] or ""))
    kept, empty_seen = [], {}
    for s in scored:
        if s["n_contents"] == 0:
            # Twelve identical empty cupboards teach the detector nothing the second one did not. Keep a couple
            # per type for door-state variety and spend the rest of the budget where there is something inside.
            t = s["objectType"]
            if empty_seen.get(t, 0) >= max(0, int(empty_per_type)):
                continue
            empty_seen[t] = empty_seen.get(t, 0) + 1
        kept.append(s)
        if len(kept) >= max(0, int(max_containers)):
            break
    return kept


def capture_budget(stop, base_views: int = 3, openness_levels: int = 2, max_poses: int = 2) -> dict:
    """How many frames one container is worth: ``{'views', 'openness', 'poses', 'frames'}``.

    An empty cupboard and a full fridge are not the same photograph, and spending equal film on them is how a
    container pass turns a 3-minute scan into a 20-minute one for no extra classes. An empty container gets a
    single frame of its open door; one holding items the dataset has never seen gets the full treatment.
    """
    n_new = len(stop.get("new_classes") or [])
    n_contents = int(stop.get("n_contents") or 0)
    if n_contents <= 0:
        views, levels, poses = 1, 1, 1
    elif n_new <= 0:
        views, levels, poses = max(1, int(base_views) - 1), min(2, max(1, int(openness_levels))), 1
    else:
        views, levels, poses = max(1, int(base_views)), max(1, int(openness_levels)), max(1, int(max_poses))
    ladder = openness_ladder(levels)
    return {"views": views, "openness": ladder, "poses": poses, "frames": views * len(ladder) * poses}


#: Rough cost model for the up-front estimate. Measured shape, not measured values: a capture is a teleport plus
#: a PNG/JPEG write, and a container costs a walk there plus two animated door swings on top of its frames.
SECONDS_PER_FRAME = 0.45
SECONDS_PER_CONTAINER = 14.0


def estimate_frames(coverage_positions: int, views_per_position: int, container_stops=(), *,
                    base_views: int = 3, openness_levels: int = 2, max_poses: int = 2,
                    balance_frames: int = 0) -> dict:
    """Up-front cost of a scan: ``{'coverage', 'containers', 'balance', 'total', 'minutes', 'n_containers'}``.

    A container pass makes a scan materially longer, and an operator who is told that before it starts can
    choose; one who finds out by watching a progress log for twenty minutes cannot. The minute figure is a rough
    model (see :data:`SECONDS_PER_FRAME`), so report it as an estimate.
    """
    stops = list(container_stops or [])
    cov = max(0, int(coverage_positions)) * max(0, int(views_per_position))
    cont = sum(capture_budget(s, base_views, openness_levels, max_poses)["frames"] for s in stops)
    bal = max(0, int(balance_frames))
    total = cov + cont + bal
    minutes = (total * SECONDS_PER_FRAME + len(stops) * SECONDS_PER_CONTAINER) / 60.0
    return {"coverage": cov, "containers": cont, "balance": bal, "total": total,
            "n_containers": len(stops), "minutes": round(minutes, 1)}


def summarize(stops) -> str:
    """One-line plan of the container pass, for the transcript: ``'Fridge(4) -> Cabinet(2) -> Drawer(0)'``."""
    if not stops:
        return "(no containers worth opening)"
    return " -> ".join(f"{s['objectType']}({s['n_contents']})" for s in stops)


# --------------------------------------------------------------------------------------------------------------------
# STATE PASS — which objects to flip so BOTH halves of every state axis get photographed.
# --------------------------------------------------------------------------------------------------------------------
#: The state property each kind of flip changes, in the order a multi-capable object is flipped by.
STATE_KINDS = (("openable", "isOpen", "open"), ("toggleable", "isToggled", "toggle"),
               ("canFillWithLiquid", "isFilledWithLiquid", "fill"))


def flipped_class(obj, kind_prop, class_of):
    """The class `obj` would wear with `kind_prop` inverted — the half of the axis a flip photographs."""
    return class_of(dict(obj, **{kind_prop: not bool(obj.get(kind_prop))}))


def plan_state_flips(objects, counts=None, *, agent_pos=None, max_objects: int = 16, per_type: int = 2,
                     per_class_target: int = 40, class_of=None, skip_types=("LightSwitch",)) -> list:
    """Objects to flip in the state pass, best first, each saying WHICH missing half it buys.

    Found on the first GPU scan (2026-09-15): the old pass flipped the nearest ten *visible* objects from wherever
    the balance pass happened to leave the agent — in practice a few cabinets and drawers — so Faucet, StoveBurner,
    StoveKnob, Toaster, CoffeeMachine and Microwave were each photographed in ONE state only, and the detector's
    state axis for all of them was untrained. This plans over the WHOLE room's metadata instead:

      * one candidate per ``(object, kind)`` — an openable-and-toggleable Microwave is two flips;
      * an axis whose other half has ZERO frames outranks one that is merely short, which outranks one already
        covered (skipped unless there is budget left);
      * at most ``per_type`` objects of a type, so twelve cabinets do not eat the budget the stove needed;
      * nearer first, as a tie-break only.

    ``skip_types`` defaults to LightSwitch: flipping it darkens the room for every later capture if a restore is
    ever missed, and a dark-room dataset is the opposite of what the scan is for. Objects not ``visible`` from
    anywhere (inside a shut cupboard) are still planned; the executor walks to them.
    """
    counts = dict(counts or {})
    name_of = class_of or (lambda o: o.get("objectType") or "")
    cands = []
    for o in (objects or []):
        otype = o.get("objectType") or ""
        pos = o.get("position") or {}
        if not otype or otype in (skip_types or ()) or pos.get("x") is None:
            continue
        for cap, prop, kind in STATE_KINDS:
            if not o.get(cap):
                continue
            now, other = name_of(o), flipped_class(o, prop, name_of)
            if now == other:                 # the class name does not encode this axis: flipping teaches nothing
                continue
            n_other = counts.get(other, 0)
            if n_other >= per_class_target:
                tier = 2
            elif n_other <= 0:
                tier = 0
            else:
                tier = 1
            d = _dist((pos["x"], pos["z"]), agent_pos)
            why = {0: f"{other} has never been photographed", 1: f"{other} is short ({n_other} frames)",
                   2: f"{other} already covered"}[tier]
            cands.append({"objectId": o.get("objectId"), "objectType": otype, "kind": kind, "prop": prop,
                          "was": bool(o.get(prop)), "current_class": now, "flipped_class": other,
                          "tier": tier, "distance": round(d, 3), "pos": (pos["x"], pos["z"]), "why": why})
    # Covered axes sort last, so they only ever spend budget the missing and short halves left unused.
    cands.sort(key=lambda c: (c["tier"], c["distance"], c["objectId"] or "", c["kind"]))
    kept, per_axis = [], {}
    for c in cands:
        axis = (c["objectType"], c["kind"])
        if per_axis.get(axis, 0) >= max(1, int(per_type)):
            continue
        per_axis[axis] = per_axis.get(axis, 0) + 1
        kept.append(c)
        if len(kept) >= max(0, int(max_objects)):
            break
    return kept


# --------------------------------------------------------------------------------------------------------------------
# BALANCE PASS — top up starved classes, opening the container an object is hidden in.
# --------------------------------------------------------------------------------------------------------------------
def closed_parent(obj, by_id):
    """The objectId of a CLOSED openable receptacle `obj` sits in, or None. A fork in a shut drawer cannot be
    photographed by walking up to the drawer; the balance pass has to open it first."""
    for pid in (obj.get("parentReceptacles") or []):
        p = by_id.get(pid)
        if p and p.get("openable") and not p.get("isOpen"):
            return pid
    return None


def plan_balance(objects, counts=None, *, per_class_target: int = 40, max_frames_per_class: int = 12,
                 max_objects_per_class: int = 3, class_of=None) -> list:
    """Classes still under target, starved first, each with the objects to visit and the frames wanted.

    Returns ``[{cls, deficit, frames_wanted, targets: [{objectId, open_parent}]}]``. The old balance pass stopped
    after FOUR frames per class and never opened a parent, so a Fork, Spoon or Egg that lives in a drawer finished
    the June and September scans with three training frames each. ``max_frames_per_class`` caps one class's share
    of the budget; spreading those frames over up to ``max_objects_per_class`` instances is what keeps them from
    being one photograph taken twelve times.
    """
    counts = dict(counts or {})
    name_of = class_of or (lambda o: o.get("objectType") or "")
    by_id = {o.get("objectId"): o for o in (objects or []) if o.get("objectId")}
    by_cls = {}
    for o in (objects or []):
        if (o.get("position") or {}).get("x") is None:
            continue
        by_cls.setdefault(name_of(o), []).append(o)
    out = []
    for cls, objs in by_cls.items():
        have = counts.get(cls, 0)
        if not cls or have >= per_class_target:
            continue
        deficit = per_class_target - have
        # visible instances first: a hidden one costs an open + close on top of the frames
        objs = sorted(objs, key=lambda o: (closed_parent(o, by_id) is not None, o.get("objectId") or ""))
        out.append({"cls": cls, "have": have, "deficit": deficit,
                    "frames_wanted": min(deficit, max(1, int(max_frames_per_class))),
                    "targets": [{"objectId": o.get("objectId"), "open_parent": closed_parent(o, by_id)}
                                for o in objs[:max(1, int(max_objects_per_class))]]})
    out.sort(key=lambda b: (b["have"] > 0, b["have"], b["cls"]))
    return out


# ===================================================================== PUBLIC API ====================================
# Pure policy for the dataset scan. Stdlib only (math); no ai2thor, no numpy, no torch. Self-test:
# `python sim/_scan_selftest.py` from the REPO ROOT.
#
# GEOMETRY (AI2-THOR convention: +horizon looks DOWN, -horizon looks UP; limits HORIZON_MIN/MAX = -30/+60)
#   eye_of(agent_pos, standing=True) -> {x,y,z}        camera point (EYE_STANDING/EYE_CROUCHED above the agent)
#   pitch_for(target, eye) / yaw_for(target, eye)      angles that centre `target`, pitch clamped to the limits
#   stance_for(target, agent_pos, prefer_standing=True) -> bool   the stance whose required pitch is COMFORTABLE
#                                                                 (inside the limits by COMFORT_MARGIN), else standing
#   aim_from(target, agent_pos, prefer_standing=True) -> (yaw, horizon, standing)
#   aim_point(container, contents=()) -> {x,y,z}       centroid of the contents, else the container's bbox centre
#
# VIEWPOINT VARIETY
#   view_offsets(n, yaw_spread=16, pitch_spread=8) -> [(dyaw, dpitch)]   dead-on first, then oblique/cropped
#   openness_ladder(n_levels=2, min_open=0.45) -> [1.0, ...]             door positions, widest first
#   select_poses(poses, target_xz, max_poses=2, bands=BANDS, max_distance=None) -> [pose]
#   coverage_views(index, n_yaws=4, horizons=COVERAGE_HORIZONS, per_yaw=2) -> [(yaw, horizon, standing)]
#   spread_positions(positions, k) -> [pos]            farthest-point sampling, deterministic
#
# CONTAINER PASS
#   rank_containers(objects, *, agent_pos, counts, per_class_target=40, max_containers=10, empty_per_type=2,
#                   class_of=None) -> [{objectId, objectType, pos, n_contents, new_classes, short_classes,
#                                       open_class, was_open, distance, score, why}]
#   capture_budget(stop, base_views=3, openness_levels=2, max_poses=2) -> {views, openness, poses, frames}
#   estimate_frames(coverage_positions, views_per_position, container_stops, ...) -> {coverage, containers,
#                   balance, total, n_containers, minutes}
#   summarize(stops) -> str
