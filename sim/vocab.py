"""Object TYPES for the detectors and the STATE SCHEMA for the per-object state estimator (option b, 2026-09-28).

Design (approved by the author after two Codex/gpt-6-astra reviews, scratchpad ASTRA_OPINION_step3*.md):

* The two YOLO detectors (primary, close-view specialist) predict **base object types only**. Every sufficiently
  visible instance gets a box whatever its state — an unobservable state never removes a detection label.
* A separate crop classifier predicts each object's **state**, axis by axis, and may answer *unknown*. For every
  (object, axis) the dataset stores three separate facts: is the axis **applicable** to this type, what is its
  physical **value** (simulator truth), and is it **observable** in this crop (decided later by calibration
  against the author's hand labels — never assumed at capture time).

Axes are the visible state variables the human-verified instruction set tests (goal keys counted over the 277
items of ``sim/dataset/general.json`` on 2026-09-28): isOpen 60, isToggled 57, isSliced 30, isDirty 27,
isFilledWithLiquid 25 + fillLiquid 25, isCooked 5, isBroken 5. Temperature is tested but never visible, so it is
not an axis. Stdlib only.
"""
from __future__ import annotations

import math

#: axis -> AI2-THOR metadata field
AXIS_FIELD = {
    "open": "isOpen", "on": "isToggled", "sliced": "isSliced", "cooked": "isCooked",
    "dirty": "isDirty", "broken": "isBroken", "filled": "isFilledWithLiquid",
}
#: The categorical axis: which liquid. Applicable only where "filled" is; value None when empty.
LIQUIDS = ("water", "coffee", "wine")
AXES_ALL = tuple(AXIS_FIELD) + ("liquid",)

#: Reversible axes can be flipped and flipped back in place — the basis of paired-render evidence.
REVERSIBLE = {"open": ("OpenObject", "CloseObject"), "on": ("ToggleObjectOn", "ToggleObjectOff"),
              "dirty": ("DirtyObject", "CleanObject"), "filled": ("FillObjectWithLiquid", "EmptyLiquidFromObject")}

#: AI2-THOR implements slicing / cracking by spawning NEW objects; they fold into the base type for the detector,
#: and the original type + parent id are kept as lineage in every capture record.
DERIVED_TYPES = {
    "AppleSliced": "Apple", "TomatoSliced": "Tomato", "LettuceSliced": "Lettuce",
    "BreadSliced": "Bread", "PotatoSliced": "Potato", "EggCracked": "Egg",
}
#: Derived pieces carry their defining state even when the metadata flag is not set on the piece.
DERIVED_FORCES = {"AppleSliced": ("sliced", True), "TomatoSliced": ("sliced", True),
                  "LettuceSliced": ("sliced", True), "BreadSliced": ("sliced", True),
                  "PotatoSliced": ("sliced", True), "EggCracked": ("broken", True)}

#: Which axes apply to which base type. Only axes the instruction set tests or that change the object's look a
#: lot; the comment says what was left out and why.
TYPE_AXES = {
    "Fridge": ("open",), "Cabinet": ("open",), "Drawer": ("open",), "Box": ("open",), "Safe": ("open",),
    "Book": ("open",), "Laptop": ("open", "on"), "Blinds": ("open",),
    "Microwave": ("open", "on"),
    "Kettle": ("open",),                        # opaque body: fill is never visible, so no fill axis
    "StoveKnob": ("on",), "StoveBurner": ("on",), "Faucet": ("on",), "Toaster": ("on",),
    "CoffeeMachine": ("on",), "LightSwitch": ("on",), "Television": ("on",), "CellPhone": ("on",),
    "DeskLamp": ("on",), "FloorLamp": ("on",),
    "Apple": ("sliced",), "Tomato": ("sliced",), "Lettuce": ("sliced",),
    "Bread": ("sliced", "cooked"), "Potato": ("sliced", "cooked"),
    "Egg": ("broken", "cooked"),                # an egg "breaks" into EggCracked; cooked only once cracked
    "Mug": ("broken", "dirty", "filled", "liquid"), "Cup": ("broken", "dirty", "filled", "liquid"),
    "Bowl": ("broken", "dirty", "filled", "liquid"),
    "Plate": ("broken", "dirty"),
    "Pot": ("dirty", "filled", "liquid"),
    "Pan": ("dirty",),                          # not canFillWithLiquid in AI2-THOR 5.0
    "Bottle": ("filled", "liquid"), "WineBottle": ("filled", "liquid"),
}


def base_type(obj_type: str) -> str:
    """``'AppleSliced'`` -> ``'Apple'``; any other type unchanged. This is the detector class."""
    return DERIVED_TYPES.get(obj_type, obj_type)


def axes_of(obj_type: str) -> tuple:
    return TYPE_AXES.get(base_type(obj_type), ())


def state_record(obj: dict) -> dict:
    """``{axis: {'applicable': bool, 'value': bool|str|None}}`` over ALL axes, from simulator metadata.

    ``observable`` is deliberately absent: it is not a property of the object but of the object IN A CROP, and it
    is assigned later from calibrated evidence (paired renders, geometry, the author's hand labels).
    """
    t = obj.get("objectType") or ""
    app = set(axes_of(t))
    forced = DERIVED_FORCES.get(t)
    out = {}
    for ax in AXES_ALL:
        if ax not in app:
            out[ax] = {"applicable": False, "value": None}
            continue
        if ax == "liquid":
            v = (obj.get("fillLiquid") or "").lower() or None if obj.get("isFilledWithLiquid") else None
        else:
            v = bool(obj.get(AXIS_FIELD[ax]))
            if forced and forced[0] == ax:
                v = forced[1]
        out[ax] = {"applicable": True, "value": v}
    return out


#: States the capture must NOT produce, with the reason. Verified 2026-09-28 in the simulator: BreakObject on a
#: Plate/Mug keeps the object (isBroken=True) but its shards are NOT attributed to the object's id in instance
#: segmentation, so they can never be labelled and would sit in frames as unlabelled "background". The state stays
#: in the SCHEMA (a broken mug is still a mug with isBroken=True); it is unsupported for training, and the state
#: estimator must answer "unknown" for it, never "intact". The Egg is unaffected: it cracks into an EggCracked
#: object that is labelled normally. Affects 2 of the 277 instruction goals (a broken Mug, a broken Plate).
CAPTURE_UNSUPPORTED = {("Mug", "broken"), ("Cup", "broken"), ("Bowl", "broken"), ("Plate", "broken")}


def composite_targets(obj_type: str, capture: bool = False) -> list:
    """Every FULL state combination an object of this type may be put into, as ``{axis: value}`` over all its axes.

    Structural rules only (a broken vessel is not also dirty/filled; bread cooks only as a slice, an egg only once
    cracked; liquid only when filled). Physical feasibility is decided by the probe per INSTANCE, never assumed.
    ``capture=True`` also drops :data:`CAPTURE_UNSUPPORTED` states.
    """
    if capture:
        base = base_type(obj_type)
        return [t for t in composite_targets(obj_type)
                if not any(t.get(ax) for (b, ax) in CAPTURE_UNSUPPORTED if b == base)]
    import itertools
    axes = [a for a in axes_of(obj_type) if a != "liquid"]
    if not axes:
        return []
    has_liquid = "liquid" in axes_of(obj_type)
    base = base_type(obj_type)
    out, seen = [], set()
    for bits in itertools.product((False, True), repeat=len(axes)):
        t = dict(zip(axes, bits))
        if t.get("broken") and base in ("Mug", "Cup", "Bowl", "Plate"):
            t = {a: (a == "broken") for a in axes}          # broken collapses every other vessel axis
        if t.get("cooked") and base == "Bread" and not t.get("sliced"):
            continue
        if t.get("cooked") and base == "Egg" and not t.get("broken"):
            continue
        liquids = LIQUIDS if (has_liquid and t.get("filled")) else (None,)
        for liq in liquids:
            tt = dict(t)
            if has_liquid:
                tt["liquid"] = liq
            k = tuple(sorted(tt.items(), key=lambda kv: kv[0]))
            if k not in seen:
                seen.add(k)
                out.append(tt)
    return out


IRREVERSIBLE_AXES = ("sliced", "cooked", "broken")

#: Types whose state is DRIVEN by another object, never set directly: AI2-THOR refuses to toggle a StoveBurner
#: ("target object is controlled by another sim object") — its StoveKnob lights it. The burner's state is still
#: read and labelled from metadata; the capture only stops trying to set it (smoke v2b: 12 refused re-rolls).
CONTROLLED_TYPES = {"StoveBurner"}

#: Heads the STATE ESTIMATOR does not learn, with the reason (Arash's go 2026-09-29, Astra step3g). StoveKnob: over
#: 192 targeted views per state in FloorPlan1 the knob never reached 900 px — insufficient support in THIS capture
#: setup (not intrinsic unlearnability). Knob DETECTION and knob actions stay; the stove's on-state is read from the
#: burner head (flame), and is "unknown" when no burner is visible — never "off".
STATE_EXCLUDED = {("StoveKnob", "on")}


def compatible(target: dict, record: dict) -> bool:
    """Can ``target`` be reached from an object's CURRENT state without undoing an irreversible change?
    (A sliced apple cannot be whole again; a cooked potato cannot be raw.)"""
    for ax in IRREVERSIBLE_AXES:
        if ax in target and bool((record.get(ax) or {}).get("value")) != bool(target[ax]):
            return False
    return True


def state_key(obj_type: str, record: dict) -> str:
    """Composite key of an object's REALISED state (from :func:`state_record`), e.g. ``'Mug|broken=0,dirty=1,
    filled=1,liquid=coffee'``; ``'Floor|-'`` for a stateless type. The unit the balancer counts."""
    parts = []
    for ax in axes_of(obj_type):
        v = record.get(ax, {}).get("value")
        parts.append(f"{ax}={int(v) if isinstance(v, bool) else (v if v is not None else '-')}")
    return f"{base_type(obj_type)}|{','.join(parts) or '-'}"


def actions_for(target: dict, current: dict = None) -> list:
    """Ordered ``(action, kwargs)`` steps that take an object to ``target`` (objectId supplied by the caller).

    Order encodes AI2-THOR's rules: slice/crack before cook, dirty before fill, break alone. With ``current`` (the
    object's :func:`state_record`), axes already at their target value emit no step — so "empty an empty mug" is
    not attempted and cannot be recorded as a spurious failure.
    """
    if current is not None:
        cur = {ax: (current.get(ax) or {}).get("value") for ax in target}
        target = {ax: v for ax, v in target.items()
                  if (cur.get(ax) != v if ax == "liquid" else bool(cur.get(ax)) != bool(v))
                  or (ax == "filled" and v and cur.get("liquid") != target.get("liquid"))}
        if "liquid" in target and target["liquid"] is not None and "filled" not in target:
            target["filled"] = True
    if target.get("broken") and not target.get("cooked"):
        return [("BreakObject", {})]
    steps = []
    # OFF comes first and ON last: AI2-THOR refuses to open a running microwave ("Target must be OFF to open!")
    # and to switch on an open one — the smoke probe marked 'open microwave' unreachable before this ordering.
    if "on" in target and not target["on"]:
        steps.append(("ToggleObjectOff", {}))
    if "open" in target:
        steps.append(("OpenObject", {"openness": 1.0}) if target["open"] else ("CloseObject", {}))
    if target.get("sliced"):
        steps.append(("SliceObject", {}))
    if target.get("broken"):
        steps.append(("BreakObject", {}))
    if target.get("cooked"):
        steps.append(("CookObject", {}))
    if "dirty" in target:
        steps.append(("DirtyObject", {}) if target["dirty"] else ("CleanObject", {}))
    if target.get("filled"):
        if current is not None and (current.get("filled") or {}).get("value"):
            steps.append(("EmptyLiquidFromObject", {}))       # switching liquid: empty first
        steps.append(("FillObjectWithLiquid", {"fillLiquid": target.get("liquid") or "water"}))
    elif "filled" in target:
        steps.append(("EmptyLiquidFromObject", {}))
    if target.get("on"):
        steps.append(("ToggleObjectOn", {}))
    return steps


def realised(target: dict, record: dict) -> bool:
    """Did the object end up in ``target``? ``record`` is :func:`state_record` of the object (or a piece)."""
    for ax, want in target.items():
        got = record.get(ax, {}).get("value")
        if ax == "liquid":
            if got != want:
                return False
        elif bool(got) != bool(want):
            return False
    return True


def axis_keys(obj_type: str, record: dict) -> list:
    """Per-HEAD state keys of a realised state: one ``'Type|axis=value'`` per applicable axis, e.g. a dirty mug with
    wine -> ``['Mug|broken=0', 'Mug|dirty=1', 'Mug|filled=1', 'Mug|liquid=wine']``.

    This is the unit the multi-head state estimator learns (one head per axis), so it is the unit the capture
    balances on (policy v2c-3, 2026-09-29). Semantics made explicit (Astra step3e): ``liquid`` is INAPPLICABLE
    when the vessel is empty (no key); a broken vessel keeps only its ``broken`` key (the other heads are masked).
    """
    base = base_type(obj_type)
    axes = axes_of(obj_type)
    if not axes:
        return []
    broken_vessel = base in ("Mug", "Cup", "Bowl", "Plate") and (record.get("broken") or {}).get("value")
    out = []
    for ax in axes:
        v = (record.get(ax) or {}).get("value")
        if broken_vessel and ax != "broken":
            continue
        if ax == "liquid":
            if v is None:
                continue
            out.append(f"{base}|liquid={v}")
        else:
            out.append(f"{base}|{ax}={int(bool(v))}")
    return out


def target_key(obj_type: str, target: dict) -> str:
    """The :func:`state_key` an object will have once ``target`` (a composite target) is realised."""
    rec = {ax: {"value": target.get(ax)} for ax in axes_of(obj_type)}
    return state_key(obj_type, rec)


def rim_elevation_deg(camera: dict, obj: dict):
    """Degrees the camera sits above the top of ``obj``'s bounding box (geometric EVIDENCE for fill visibility)."""
    bb = obj.get("axisAlignedBoundingBox") or {}
    c, s = bb.get("center"), bb.get("size")
    if not camera or not c or not s:
        return None
    top = c["y"] + s["y"] / 2.0
    return math.degrees(math.atan2(camera["y"] - top, max(math.hypot(c["x"] - camera["x"], c["z"] - camera["z"]),
                                                          1e-6)))


# ===================================================================== PUBLIC API ====================================
# AXIS_FIELD, AXES_ALL, LIQUIDS, REVERSIBLE, DERIVED_TYPES, TYPE_AXES
# base_type(t) -> detector class          axes_of(t) -> tuple
# state_record(obj) -> {axis: {applicable, value}}      (observability is assigned later, per crop)
# composite_targets(t) -> [{axis: value}]   actions_for(target) -> [(action, kwargs)]
# realised(target, record) -> bool        state_key(t, record) -> str      target_key(t, target) -> str
# axis_keys(t, record) -> ['Type|axis=value', ...]   the per-head balancing unit (liquid only when filled)
# rim_elevation_deg(camera, obj) -> float|None
