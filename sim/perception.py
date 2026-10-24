"""Per-scene YOLO perception with **stateful** object classes — the active-perception layer (Phase 4).

The thesis uses a *per-scene* detector: each `FloorPlan` gets its own dataset + weights (`weights/<scene>.pt`);
the agent rescans & retrains on entering a scene, so no cross-scene generalization is needed. Crucially the YOLO
classes are **state-aware**: a class is ``'<objectType>_<state>'`` (e.g. ``Fridge_open`` vs ``Fridge_closed``,
``Apple_sliced`` vs ``Apple_whole``) so a single detection answers both *what* and *what state* — the agent can see
that the fridge is open without a separate metadata query.

To keep the class list small we use **ONE salient state axis per objectType** (see :data:`STATE_AXES`), not the full
boolean product of every effect field. Types with no salient axis keep their bare ``objectType`` as the class.

Stdlib-/numpy-first per CLAUDE.md: heavy deps (``ultralytics``/``torch``/``cv2``) are imported **lazily inside the
functions that need them**, never at module top. ``import perception`` therefore succeeds with only numpy present
(and even the bare label helpers need nothing but stdlib). This mirrors ``rag.py``/``memory.py``.
"""
from __future__ import annotations

import os

# ----------------------------------------------------------------------------------------------------------------------
# State axes — the curated map objectType -> the SINGLE salient state axis to encode in the class name.
#
# Each value is a spec ``{'field': <metadata key>, True: <true-label>, False: <false-label>}`` read off an AI2-THOR
# object's metadata (the boolean EFFECT_KEYS fields from thor_actions.py: isOpen / isToggled / isSliced / isBroken /
# isCooked / isFilledWithLiquid).
#
# Several kitchen types fit MORE THAN ONE axis (a Microwave both opens and toggles; a Potato both slices and cooks).
# We pick exactly one, by this priority (highest first) — ordered by how *visually dominant / unambiguous* the change
# is in an RGB frame, which is what the detector actually has to learn:
#
#       isOpen  >  isToggled  >  isBroken  >  isSliced  >  isCooked  >  isFilledWithLiquid
#
#   - isOpen / isToggled win for appliances: a swung-open door or a lit/running unit reshapes the whole object, so
#     a Microwave is keyed on its door (open/closed), not its on/off light.
#   - isBroken (Egg only): in AI2-THOR an egg *cracks* rather than slices, so cracked/whole is its real axis.
#   - isSliced beats isCooked for food (Apple/Bread/Tomato/Potato/Lettuce): slicing is a large geometry change,
#     cooking is a subtler texture/colour shift. isCooked stays in the vocabulary for any future type that has no
#     more-salient axis.
#   - isFilledWithLiquid is the subtlest axis and is used only for vessels (Mug/Cup/Bottle/Bowl/Pot/Pan/WineBottle),
#     none of which overlap another axis, so the priority never has to demote it.
#
# The class list a scene actually trains on is derived from the objects PRESENT in that scene (see
# :func:`scene_class_names`), so unused axes cost nothing.
# ----------------------------------------------------------------------------------------------------------------------
def _axis(field: str, true_label: str, false_label: str) -> dict:
    return {"field": field, True: true_label, False: false_label}


STATE_AXES = {
    # openables -> isOpen -> open / closed
    "Fridge":       _axis("isOpen", "open", "closed"),
    "Microwave":    _axis("isOpen", "open", "closed"),   # also toggleable; door wins (more visually dominant)
    "Cabinet":      _axis("isOpen", "open", "closed"),
    "Drawer":       _axis("isOpen", "open", "closed"),
    "Kettle":       _axis("isOpen", "open", "closed"),
    "Safe":         _axis("isOpen", "open", "closed"),
    "Box":          _axis("isOpen", "open", "closed"),

    # toggleables -> isToggled -> on / off
    "StoveBurner":  _axis("isToggled", "on", "off"),
    "StoveKnob":    _axis("isToggled", "on", "off"),     # AI2-THOR's real type for the stove control (cf. "knife")
    "Faucet":       _axis("isToggled", "on", "off"),
    "Toaster":      _axis("isToggled", "on", "off"),
    "CoffeeMachine": _axis("isToggled", "on", "off"),
    "LightSwitch":  _axis("isToggled", "on", "off"),
    "Television":   _axis("isToggled", "on", "off"),

    # breakable (Egg) -> isBroken -> cracked / whole
    "Egg":          _axis("isBroken", "cracked", "whole"),

    # sliceables -> isSliced -> sliced / whole
    "Apple":        _axis("isSliced", "sliced", "whole"),
    "Bread":        _axis("isSliced", "sliced", "whole"),
    "Tomato":       _axis("isSliced", "sliced", "whole"),
    "Potato":       _axis("isSliced", "sliced", "whole"),
    "Lettuce":      _axis("isSliced", "sliced", "whole"),

    # fillables -> isFilledWithLiquid -> filled / empty
    "Mug":          _axis("isFilledWithLiquid", "filled", "empty"),
    "Cup":          _axis("isFilledWithLiquid", "filled", "empty"),
    "Bottle":       _axis("isFilledWithLiquid", "filled", "empty"),
    "Bowl":         _axis("isFilledWithLiquid", "filled", "empty"),
    "Pot":          _axis("isFilledWithLiquid", "filled", "empty"),
    "Pan":          _axis("isFilledWithLiquid", "filled", "empty"),
    "WineBottle":   _axis("isFilledWithLiquid", "filled", "empty"),
}


# ----------------------------------------------------------------------------------------------------------------------
# Labels — pure (stdlib-only) helpers mapping AI2-THOR object metadata -> class names.
# ----------------------------------------------------------------------------------------------------------------------
def state_label(obj: dict) -> str:
    """The salient-state suffix for an AI2-THOR object dict, or ``''`` if its type has no axis.

    Reads :data:`STATE_AXES` for ``obj['objectType']`` and returns the true/false label for the spec's metadata
    field (a missing/falsy field reads as the false label — e.g. an unopened fridge -> ``'closed'``).
    """
    spec = STATE_AXES.get(obj.get("objectType"))
    if not spec:
        return ""
    return spec[bool(obj.get(spec["field"]))]


def class_name(obj: dict) -> str:
    """The YOLO class for an AI2-THOR object: ``objectType`` or ``objectType_<state_label>``."""
    t = obj.get("objectType") or ""
    s = state_label(obj)
    return f"{t}_{s}" if s else t


def scene_class_names(objects) -> list:
    """Sorted, de-duplicated class list for the objects present in a scene (the per-scene YOLO label set)."""
    return sorted({class_name(o) for o in objects if o.get("objectType")})


def split_class_name(name: str):
    """Inverse of :func:`class_name`: ``'Fridge_open'`` -> ``('Fridge', 'open')``; ``'Sink'`` -> ``('Sink', '')``.

    Splits on the first ``'_'`` (AI2-THOR object types and our state labels contain none), so a stateful class is
    cleanly separated back into ``(type, state)``.
    """
    if "_" in name:
        t, s = name.split("_", 1)
        return t, s
    return name, ""


# ----------------------------------------------------------------------------------------------------------------------
# Boxes — mask -> bbox, bbox -> YOLO label line.
# ----------------------------------------------------------------------------------------------------------------------
def box_from_mask(mask):
    """Tight axis-aligned bbox ``(x1, y1, x2, y2)`` (ints, inclusive corners) of a boolean HxW mask; ``None`` if empty.

    Used to turn AI2-THOR instance masks (``ThorAgent.instance_masks()``) into detection boxes for dataset labels.
    """
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    ys, xs = np.where(m)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def yolo_line(cls_id: int, box_xyxy, img_w: int, img_h: int) -> str:
    """One YOLO label line ``'<cls> <cx> <cy> <w> <h>'`` with the box normalized to [0,1] of the frame.

    ``box_xyxy`` is ``(x1, y1, x2, y2)`` in pixels; centre/size are normalized by ``img_w``/``img_h`` and clamped to
    [0,1] (a box flush to the frame edge can otherwise round just outside).
    """
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / float(img_w)
    h = (y2 - y1) / float(img_h)
    clamp = lambda v: 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return f"{int(cls_id)} {clamp(cx):.6f} {clamp(cy):.6f} {clamp(w):.6f} {clamp(h):.6f}"


# ----------------------------------------------------------------------------------------------------------------------
# Detector — thin wrapper over an ultralytics YOLO model trained on a scene's stateful classes.
# ----------------------------------------------------------------------------------------------------------------------
class Detector:
    """Loads ``weights/<scene>.pt`` and runs inference, splitting each ``Type_state`` class back into type + state.

    ``ultralytics``/``torch`` are imported lazily in :meth:`__init__` so ``import perception`` needs neither.
    """

    def __init__(self, weights_path: str):
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path!r}. Train this scene's detector first "
                f"(gen_dataset.py -> ultralytics train) so weights/<scene>.pt exists."
            )
        try:
            from ultralytics import YOLO  # lazy: heavy (pulls torch); keep module import light
        except Exception as exc:
            raise RuntimeError(
                f"ultralytics is required to load a Detector but failed to import ({exc}). "
                f"Install it in the WSL venv (`pip install ultralytics`)."
            ) from exc
        self.weights_path = weights_path
        load_path = weights_path
        if "'" in os.path.abspath(weights_path):
            # ultralytics strips apostrophes from absolute weight paths (".../Master's Project/..." -> "Masters
            # Project") and then can't open them. The saved file is fine — load it from a clean temp copy instead.
            import shutil
            import tempfile
            load_path = os.path.join(tempfile.gettempdir(), "thor_yolo_" + os.path.basename(weights_path))
            shutil.copyfile(weights_path, load_path)
        self.model = YOLO(load_path)
        self.names = dict(self.model.names)  # {cls_id: 'Type_state'}

    def detect(self, frame_rgb, conf: float = 0.25) -> list:
        """Detect on an RGB HxWx3 frame; return ``[{cls, type, state, bbox_xyxy, conf}]`` (one per box).

        ultralytics treats a NumPy image as OpenCV **BGR** and flips it to RGB itself (8.4.83 ``predictor.py``
        preprocess, ``[..., ::-1]``), while training reads JPEGs with cv2 (BGR) and flips them the same way. An
        AI2-THOR frame is RGB, so passing it straight in showed the network red and blue swapped — every live
        detection before 2026-09-28 (and the 0.49 oracle score) was measured on colour-swapped frames. Found by
        the Codex/Astra review; hand it BGR.
        """
        import numpy as np
        frame_bgr = np.ascontiguousarray(np.asarray(frame_rgb)[..., ::-1])
        res = self.model(frame_bgr, conf=conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls[0])
            name = self.names.get(cid, str(cid))
            t, s = split_class_name(name)
            x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy[0].tolist())
            out.append({
                "cls": name, "type": t, "state": s,
                "bbox_xyxy": (x1, y1, x2, y2), "conf": float(b.conf[0]),
            })
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Perceived-mode observation — the VISION-ONLY counterpart to reading AI2-THOR metadata.
#
# `oracle` hands the planner AI2-THOR's ground-truth object list: it is told what is in the room whether or not the
# robot could possibly have seen it. `perceived` hands it ONLY what this scene's trained detector finds in the agent's
# own camera while the agent turns and searches. That gap is the perception ablation axis, so the two paths must be
# ONE implementation (webapp + eval both call in here) and a failure to perceive must be *reported*, never silently
# papered over with the oracle answer.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Step-3 detectors (kitchen_v3, 2026-10): per kitchen a PRIMARY (YOLO26s, every frame, 49 type-only classes) and an
# AMBIGUITY / inspection specialist (YOLO26m, 10 look-alike classes) that runs only while the robot holds an object up
# for inspection — the deployment rule frozen in docs/ambiguity-spec.md ("gated strict replacement").
# ----------------------------------------------------------------------------------------------------------------------
V3_PRIMARY, V3_SPECIALIST = "yolo26s", "amb26m"


def _scene_stem(scene: str) -> str:
    base = str(scene or "").strip()
    return base[:-len("_physics")] if base.endswith("_physics") else base


def v3_paths(scene: str, weights_dir: str = None) -> dict:
    """``{"primary": path | "", "specialist": path | ""}`` for the step-3 models of ``scene`` (weights/v3/)."""
    root = os.path.join(weights_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"), "v3")
    stem = _scene_stem(scene)
    out = {}
    for k, tag in (("primary", V3_PRIMARY), ("specialist", V3_SPECIALIST)):
        p = os.path.join(root, f"{stem}.{tag}.pt") if stem else ""
        out[k] = p if p and os.path.isfile(p) else ""
    return out


class TwoStageDetector:
    """The step-3 perception stack: ``primary`` on every frame; while ``inspecting`` (the robot is holding an object
    up — robot state, never simulator identity) the ``specialist`` REPLACES the primary's predictions for the
    specialist's classes (gated strict replacement). Same ``detect`` contract as :class:`Detector`, plus a
    ``source`` field ("primary" / "specialist") on every detection.

    ``gate`` is an optional zero-argument callable consulted when ``inspecting`` is not passed (the agent's
    ``detect_objects`` path calls ``detect(frame, conf=...)``); ``mode`` is "auto" (gate decides), "off" (primary
    only) or "always" (specialist on every frame — the ungated ablation)."""

    def __init__(self, primary, specialist=None, gate=None, mode: str = "auto", states=None):
        self.primary, self.specialist, self.gate, self.mode = primary, specialist, gate, mode
        self.states = states                        # optional StateEstimator: adds "state" text to each box
        self.targets = set((getattr(specialist, "names", None) or {}).values()) if specialist else set()
        self.weights_path = getattr(primary, "weights_path", "")
        self.last_inspecting = False

    def active(self, inspecting=None) -> bool:
        if self.specialist is None or self.mode == "off":
            return False
        if self.mode == "always":
            return True
        if inspecting is None:
            try:
                inspecting = bool(self.gate()) if self.gate else False
            except Exception:
                inspecting = False
        return bool(inspecting)

    def detect(self, frame_rgb, conf: float = 0.25, inspecting=None) -> list:
        prim = [dict(d, source="primary") for d in self.primary.detect(frame_rgb, conf=conf)]
        self.last_inspecting = self.active(inspecting)
        out = prim
        if self.last_inspecting:
            spec = [dict(d, source="specialist") for d in self.specialist.detect(frame_rgb, conf=conf)]
            out = fuse_gated(prim, spec, self.targets)
        if self.states is not None:
            try:
                out = self.states.annotate(frame_rgb, out)
            except Exception:
                pass                                 # a state failure never hides the detections themselves
        return out


STATE_WORDS = {"open": ("closed", "open"), "on": ("off", "on"), "sliced": ("whole", "sliced"),
               "cooked": ("raw", "cooked"), "broken": ("intact", "broken"), "dirty": ("clean", "dirty"),
               "filled": ("empty", "filled")}


class StateEstimator:
    """Option (b)'s third piece: one small classifier per state axis applied to each DETECTED box (crop padded 15 %,
    letterboxed square, as in sim/state_data.crop_square). A state is reported only when the box is at least the size
    from which Arash could see that state in his blind hand labels AND the confidence clears the VAL-calibrated
    threshold for that axis and value (>= 95 % precision); otherwise it is "unknown" ("sliced?"), never a guess and
    never the negative value (sim/state_eval.py -> kitchen_v3/<scene>/state_calib.json)."""

    def __init__(self, scene: str, weights_dir: str = None):
        import json as _json
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(weights_dir or os.path.join(here, "weights"), "v3")
        stem = _scene_stem(scene)
        calib = os.path.join(here, "kitchen_v3", stem, "state_calib.json")
        self.calib = _json.load(open(calib)) if os.path.isfile(calib) else None
        self.models = {}
        for ax in STATE_WORDS:
            p = os.path.join(root, f"{stem}.state_{ax}.pt")
            if os.path.isfile(p):
                self.models[ax] = Detector(p).model
        import vocab as _vocab
        import state_data as _sd
        self._vocab, self._sd = _vocab, _sd

    @staticmethod
    def available(scene: str, weights_dir: str = None) -> bool:
        here = os.path.dirname(os.path.abspath(__file__))
        stem = _scene_stem(scene)
        root = os.path.join(weights_dir or os.path.join(here, "weights"), "v3")
        return (os.path.isfile(os.path.join(here, "kitchen_v3", stem, "state_calib.json"))
                and any(os.path.isfile(os.path.join(root, f"{stem}.state_{a}.pt")) for a in STATE_WORDS))

    def axes_for(self, t):
        return [a for a in self._vocab.TYPE_AXES.get(t, ()) if a in self.models
                and (a not in self._sd.AXIS_TYPES or t in self._sd.AXIS_TYPES[a])]

    def annotate(self, frame_rgb, dets: list) -> list:
        import numpy as np
        from PIL import Image
        img = None
        floors = (self.calib or {}).get("floors_box_area", {})
        th = (self.calib or {}).get("thresholds", {})
        jobs = {}                                        # axis -> [(det index, crop, area)]
        for i, d in enumerate(dets):
            axes = self.axes_for(d.get("type"))
            if not axes:
                continue
            if img is None:
                img = Image.fromarray(np.ascontiguousarray(np.asarray(frame_rgb)[..., :3]))
            x1, y1, x2, y2 = d["bbox_xyxy"]
            crop = self._sd.crop_square(img, (x1, y1, x2, y2))
            for ax in axes:
                jobs.setdefault(ax, []).append((i, crop, (x2 - x1) * (y2 - y1)))
        out = {}                                         # det index -> [(axis, value or None)]
        for ax, lst in jobs.items():                     # ONE batched call per axis
            res = self.models[ax]([c for _i, c, _a in lst], imgsz=128, verbose=False)
            for (i, _c, area), r in zip(lst, res):
                v, c = r.names[int(r.probs.top1)], float(r.probs.top1conf)
                t = th.get(ax, {}).get(v)
                ok = area >= floors.get(ax, 0) and t is not None and c >= t
                out.setdefault(i, []).append((ax, v if ok else None))
        for i, lst in out.items():
            dets[i]["states"] = dict(lst)
            dets[i]["state"] = ", ".join(STATE_WORDS[ax][int(v)] if v is not None else f"{ax}?" for ax, v in lst)
        return dets


def fuse_gated(primary_dets: list, specialist_dets: list, targets) -> list:
    """Strict replacement: drop every primary detection of a specialist class, keep the rest, add the specialist's."""
    t = set(targets or ())
    return [d for d in primary_dets if d.get("type") not in t] + list(specialist_dets)


def weights_path(scene: str, weights_dir: str = None) -> str:
    """Resolve a scene's trained YOLO weights file, or ``''`` when none exists.

    AI2-THOR reports the *live* scene as ``FloorPlan1_physics`` while the dataset/eval side names it
    ``FloorPlan1``, and the weights file is written under whichever spelling did the scan. Trying both (in either
    direction) is what stops perceived mode from degrading to oracle purely because the caller happened to hold
    the other half of the name.
    """
    root = weights_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    base = str(scene or "").strip()
    if not base:
        return ""
    stem = base[:-len("_physics")] if base.endswith("_physics") else base
    for cand in (base, stem, stem + "_physics"):
        p = os.path.join(root, f"{cand}.pt")
        if os.path.isfile(p):
            return p
    return ""


def observation_from(seen: dict, held_type: str = "none") -> str:
    """Render accumulated ``{objectType: state}`` detections as the OBSERVATION block for the planner prompt.

    ``held`` is proprioception, not vision — a robot knows what is in its own gripper — so it is the one field
    here that does not come out of the detector.
    """
    items = ", ".join(f"{t}" + (f"({s})" if s else "") for t, s in sorted((seen or {}).items()))
    return f"held={held_type or 'none'}\nvisible (perceived): {items or '(nothing detected)'}"


def sweep_detections(ag, detector, out: dict, conf: float = 0.25, sweep: bool = True, turn: float = 45.0,
                     detect_every: float = None):
    """GENERATOR: run ``detector`` over the agent's camera while it spins a full look-around, folding every
    detection into ``out`` as ``{objectType: state}``.

    This is the thesis's active-perception gesture made literal: the agent cannot read the room, so it *turns
    to look at it*. Yields a status label per rotation step, which is what lets a GUI animate the search rather
    than freeze for the duration. Drive it with ``yield from`` (eval, single-threaded) or by handing it to the
    worker queue (webapp — only the worker thread may touch the controller).

    ``turn`` is how far the agent rotates per RENDERED FRAME; ``detect_every`` is how far it rotates between
    INFERENCES. They were once the same number, fixed at 45 deg to keep YOLO to 8 calls -- which also meant the
    agent completed a 360 deg search in 8 frames, about a fifth of a second, and nobody could see it happen.
    Splitting them (8 deg / 45 deg = 45 frames, still 8 inferences) makes the search watchable at no extra cost.
    ``detect_every=None`` keeps them equal, i.e. the original behaviour.
    """
    import search as _search

    def fold():
        for d in ag.detect_objects(detector, conf=conf) or []:
            t = d.get("type")
            if t and not out.get(t):
                out[t] = d.get("state") or ""

    fold()
    if sweep:
        for deg, do_detect in _search.sweep_schedule(turn, detect_every if detect_every else turn):
            ag.rotate_right(deg)
            if do_detect:
                fold()
            yield f"searching · {len(out)} type(s) seen"
    yield f"perceived {len(out)} object type(s)"


def observation_text(ag, scene: str = None, mode: str = "oracle", weights_dir: str = None,
                     detector=None, sweep: bool = True, max_items: int = 30, conf: float = 0.25):
    """OBSERVATION block for the planner prompt. Returns ``(text, degraded)``.

    ``degraded`` is True when perceived mode could not run (no trained weights, no ultralytics, inference
    error) and the text fell back to oracle. Callers MUST surface it: a silent fallback turns the perception
    ablation into a measurement of nothing.

    Blocking — it drives :func:`sweep_detections` to exhaustion itself. A caller that owns a render loop
    (the webapp worker) should drive that generator directly instead so the search animates.
    """
    if str(mode or "oracle") != "perceived":
        return ag.describe(max_items=max_items), False
    try:
        det = detector
        if det is None:
            wp = weights_path(scene or getattr(ag, "scene", ""), weights_dir)
            if not wp:
                raise FileNotFoundError(
                    f"no trained detector for scene {scene or getattr(ag, 'scene', '')!r} "
                    f"(expected sim/weights/<scene>.pt — run 'scan & learn' then 'train detector')")
            det = Detector(wp)
        seen: dict = {}
        for _ in sweep_detections(ag, det, seen, conf=conf, sweep=sweep):
            pass
        held = ag.held_object()
        return observation_from(seen, (held or {}).get("objectType") or "none"), False
    except Exception as exc:
        try:
            return ag.describe(max_items=max_items) + f"\n[perceived-mode degraded: {exc}]", True
        except Exception:
            return "", True


# ----------------------------------------------------------------------------------------------------------------------
# Self-test — numpy only; no torch / ultralytics / ai2thor needed.
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    # box_from_mask: a 4x6 rectangle of True inside a 20x20 frame -> tight inclusive corners.
    m = np.zeros((20, 20), dtype=bool)
    m[5:9, 4:10] = True            # rows 5..8, cols 4..9
    assert box_from_mask(m) == (4, 5, 9, 8), box_from_mask(m)
    assert box_from_mask(np.zeros((20, 20), dtype=bool)) is None

    # yolo_line: centre of (4,5,9,8) in a 20x20 frame; width 5/20=0.25, height 3/20=0.15.
    line = yolo_line(3, (4, 5, 9, 8), 20, 20)
    parts = line.split()
    assert parts[0] == "3", line
    assert abs(float(parts[1]) - 0.325) < 1e-6 and abs(float(parts[2]) - 0.325) < 1e-6, line
    assert abs(float(parts[3]) - 0.25) < 1e-6 and abs(float(parts[4]) - 0.15) < 1e-6, line

    # state_label / class_name on fake metadata (no AI2-THOR needed).
    assert class_name({"objectType": "Fridge", "isOpen": True}) == "Fridge_open"
    assert class_name({"objectType": "Fridge", "isOpen": False}) == "Fridge_closed"
    assert class_name({"objectType": "Fridge"}) == "Fridge_closed"        # missing field reads as false
    assert class_name({"objectType": "Apple", "isSliced": False}) == "Apple_whole"
    assert class_name({"objectType": "Apple", "isSliced": True}) == "Apple_sliced"
    assert class_name({"objectType": "Egg", "isBroken": True}) == "Egg_cracked"
    assert class_name({"objectType": "Mug", "isFilledWithLiquid": True}) == "Mug_filled"
    assert class_name({"objectType": "StoveBurner", "isToggled": True}) == "StoveBurner_on"
    assert state_label({"objectType": "Sink"}) == ""                      # no axis -> bare type
    assert class_name({"objectType": "Sink"}) == "Sink"

    # scene_class_names: sorted + de-duplicated over a scene's objects.
    objs = [
        {"objectType": "Fridge", "isOpen": True},
        {"objectType": "Fridge", "isOpen": False},
        {"objectType": "Apple", "isSliced": False},
        {"objectType": "Sink"},
    ]
    assert scene_class_names(objs) == ["Apple_whole", "Fridge_closed", "Fridge_open", "Sink"], scene_class_names(objs)

    # split_class_name round-trips class_name.
    assert split_class_name("Fridge_open") == ("Fridge", "open")
    assert split_class_name("Sink") == ("Sink", "")

    # ---- perceived-mode helpers (no torch / ai2thor needed) --------------------------------------------------
    import tempfile as _tf

    with _tf.TemporaryDirectory() as _wd:
        assert weights_path("FloorPlan1", _wd) == ""                 # nothing trained yet -> '' (never a guess)
        assert weights_path("", _wd) == ""
        open(os.path.join(_wd, "FloorPlan1_physics.pt"), "wb").close()
        # the live sceneName and the dataset name are two spellings of ONE scene; both must resolve
        assert weights_path("FloorPlan1_physics", _wd).endswith("FloorPlan1_physics.pt")
        assert weights_path("FloorPlan1", _wd).endswith("FloorPlan1_physics.pt")
        open(os.path.join(_wd, "FloorPlan2.pt"), "wb").close()
        assert weights_path("FloorPlan2_physics", _wd).endswith("FloorPlan2.pt")
        assert weights_path("FloorPlan3", _wd) == ""

    # observation_from: sorted, state-annotated, and honest when the detector saw nothing
    _NL = chr(10)
    assert observation_from({"Fridge": "open", "Apple": ""}, "Mug") == (
        "held=Mug" + _NL + "visible (perceived): Apple, Fridge(open)")
    assert observation_from({}, "") == "held=none" + _NL + "visible (perceived): (nothing detected)"

    # sweep_detections: a fake agent proves the fold/yield contract without torch or a simulator
    class _FakeAg:
        def __init__(self):
            self.pos = 0
            self.reads = 0
        def detect_objects(self, det, conf=0.25):
            self.reads += 1
            return [{"type": "Apple", "state": "whole"}] if self.pos == 0 else [{"type": "Fridge", "state": "closed"}]
        def rotate_right(self, deg=45.0):
            self.pos += 1

        def look_around(self, turn=45.0):
            for _ in range(int(round(360.0 / turn))):
                self.pos += 1
                yield "looking around"

    ag = _FakeAg()
    seen = {}
    labels = list(sweep_detections(ag, object(), seen, sweep=True, turn=90.0))
    assert seen == {"Apple": "whole", "Fridge": "closed"}, seen
    assert ag.reads == 5, ag.reads            # 1 before the sweep + one per 90 deg step
    assert len(labels) == 5 and labels[-1] == "perceived 2 object type(s)", labels
    seen2 = {}
    assert list(sweep_detections(_FakeAg(), object(), seen2, sweep=False)) == ["perceived 1 object type(s)"]
    assert seen2 == {"Apple": "whole"}        # sweep=False -> the current frame only, no rotation

    # rotation granularity and detection cadence are INDEPENDENT: a fine sweep must not cost extra inferences
    fine = _FakeAg()
    labels = list(sweep_detections(fine, object(), {}, sweep=True, turn=8.0, detect_every=45.0))
    assert len(labels) == 46, len(labels)              # 45 rotation frames + the closing label
    assert fine.reads == 9, fine.reads                 # 1 before the sweep + 8 at 45 deg intervals
    coarse = _FakeAg()
    list(sweep_detections(coarse, object(), {}, sweep=True, turn=45.0))
    assert coarse.reads == 9, coarse.reads             # detect_every=None -> unchanged from the original

    # step-3 two-stage detector: gated strict replacement
    class _FakeDet:
        def __init__(self, names, dets):
            self.names, self._d = names, dets

        def detect(self, frame, conf=0.25):
            return [dict(x) for x in self._d]

    P = _FakeDet({0: "Mug", 1: "Spoon"}, [{"type": "Mug", "cls": "Mug", "conf": .9, "bbox_xyxy": (0, 0, 9, 9)},
                                          {"type": "Spoon", "cls": "Spoon", "conf": .6, "bbox_xyxy": (20, 20, 30, 30)}])
    S = _FakeDet({0: "Fork", 1: "Spoon"}, [{"type": "Fork", "cls": "Fork", "conf": .8, "bbox_xyxy": (20, 20, 30, 30)}])
    held = {"v": False}
    two = TwoStageDetector(P, S, gate=lambda: held["v"])
    out = two.detect(None)
    assert [d["type"] for d in out] == ["Mug", "Spoon"] and not two.last_inspecting      # not holding -> primary
    held["v"] = True
    out = two.detect(None)
    assert sorted(d["type"] for d in out) == ["Fork", "Mug"], out                        # Spoon replaced by Fork
    assert {d["source"] for d in out} == {"primary", "specialist"} and two.last_inspecting
    assert [d["type"] for d in two.detect(None, inspecting=False)] == ["Mug", "Spoon"]    # explicit gate wins
    two.mode = "off"
    assert [d["type"] for d in two.detect(None)] == ["Mug", "Spoon"]
    two.mode, held["v"] = "always", False
    assert sorted(d["type"] for d in two.detect(None)) == ["Fork", "Mug"]
    assert TwoStageDetector(P, None, gate=lambda: True).detect(None)[1]["type"] == "Spoon"  # no specialist -> primary
    _wd2 = _tf.mkdtemp()
    os.makedirs(os.path.join(_wd2, "v3"))
    assert v3_paths("FloorPlan1_physics", _wd2) == {"primary": "", "specialist": ""}
    open(os.path.join(_wd2, "v3", "FloorPlan1.yolo26s.pt"), "wb").close()
    assert v3_paths("FloorPlan1_physics", _wd2)["primary"].endswith("FloorPlan1.yolo26s.pt")
    assert v3_paths("FloorPlan1", _wd2)["specialist"] == ""

    print("PERCEPTION SELFTEST OK")
