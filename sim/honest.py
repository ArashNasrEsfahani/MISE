"""Step 4 (s1-rules) -- the honest observation layer: the ONE place that decides what the agent may know.

In ``honest`` mode the agent decides only from what a real robot would have:

* its camera (RGB + depth) run through the step-3 perception stack (``perception.TwoStageDetector``: primary
  YOLO26s, the gated inspection specialist, and the StateEstimator's state text),
* its own pose (proprioception), the success bool of its last action, and its own action history
  ("I picked up an Apple" -> holding Apple).

Everything the simulator knows beyond that is privileged. It is reachable only through ``HonestAgent.meta(reason,
fields)`` -- the single door -- and every read is counted by a :class:`MetaAudit` under a reason tag:

    pose            agent position / rotation / horizon / standing / camera position (proprioception)
    actuator        lastActionSuccess (+ errorMessage, which is LOGGED for the failure breakdown, never returned)
    action-resolve  the return of a raw ``GetObjectInFrame(point)``: an opaque id used once for the action call
    map-build       the return of ``GetReachablePositions`` (coverage report only)
    scoring         goal checks / applicability gate, only inside ``with agent.scoring():`` (no acting there)
    oracle          the oracle comparison mode only

In honest mode an untagged read, an ``oracle`` read, a field outside the reason's whitelist, or a scoring read
outside the scoring phase raises :class:`HonestyViolation`. The violation is recorded first, so the episode is
marked invalid for honest-arm scoring even when a caller swallows the exception. Spec: docs/step4-plan.md (s1).

The static guard in ``sim/_step4_rules_selftest.py`` allows privileged tokens only between the ADAPTERS markers
below. Code outside that region must stay free of them -- including comments.
"""
from __future__ import annotations

import contextlib
import copy
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

MODES = ("honest", "oracle")
REASONS = ("pose", "actuator", "action-resolve", "map-build", "scoring", "oracle")

#: Per-reason field whitelist (dotted paths into the simulator's per-step record). "*" = the whole record.
ALLOWED_FIELDS = {
    "pose": {"agent.position", "agent.rotation", "agent.cameraHorizon", "agent.isStanding", "cameraPosition"},
    "actuator": {"lastActionSuccess", "errorMessage"},
    "action-resolve": {"actionReturn"},
    "map-build": {"actionReturn"},
    "scoring": {"*"},
    "oracle": {"*"},
}
#: The primitive a reason's ``actionReturn`` must come from (else the return belongs to some other action).
RETURN_OF = {"action-resolve": "GetObjectInFrame", "map-build": "GetReachablePositions"}

#: Locomotion / camera primitives an honest agent may send (no Teleport: that is a privileged shortcut).
MOVE_PRIMS = frozenset({"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight",
                        "LookUp", "LookDown", "Crouch", "Stand", "Pass"})
#: Interaction primitives that take a target (resolved from a screen point).
TARGET_PRIMS = frozenset({"PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn",
                          "ToggleObjectOff", "SliceObject", "BreakObject", "CookObject", "DirtyObject",
                          "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject"})
#: Interaction primitives that act on whatever is in hand (no target).
#: SPEC-INHAND fix B: RotateHeldObject turns the object in the hand for an inspection look (no target, no release)
HAND_PRIMS = frozenset({"DropHandObject", "ThrowObject", "RotateHeldObject"})
RELEASE_PRIMS = frozenset({"PutObject", "DropHandObject", "ThrowObject"})

#: Smooth display (console only, 2026-10-07). ONE logical primitive may be sent as several smaller physical sends so
#: a person can follow it: a move as short moves, a turn / look as small turns, a door as openness steps. The
#: policy sees one primitive either way (``step`` counts logical primitives; ``sends`` counts physical sends), so the
#: agent decides identically at every display speed. A smooth profile is a dict
#: ``{"step": m, "turn": deg, "look": deg, "doors": n, "pause": s}``; ``None`` = one send per primitive (default).
SMOOTH_MOVES = {"MoveAhead": "MoveBack", "MoveBack": "MoveAhead", "MoveLeft": "MoveRight", "MoveRight": "MoveLeft"}
SMOOTH_TURNS = {"RotateLeft": "RotateRight", "RotateRight": "RotateLeft", "LookUp": "LookDown", "LookDown": "LookUp"}


class HonestyViolation(RuntimeError):
    """An honest-mode code path tried to read privileged simulator state."""


class MetaAudit:
    """Counts every privileged read per reason; records violations. ``strict`` -> a violation also raises."""

    def __init__(self, strict: bool = True):
        self.strict = bool(strict)
        self.new_episode(None)

    def new_episode(self, episode):
        self.episode = episode
        self.counts = {r: 0 for r in REASONS}
        self.log = []            # (step, reason, fields, caller)
        self.violations = []     # (step, reason, fields, caller, why)
        self.errors = []         # (step, action, raw error text) -- failure breakdown only, never decision input
        self.invalid = False     # sticky: one violation invalidates the episode for honest-arm scoring

    def record(self, step, reason, fields, caller):
        self.counts[reason] = self.counts.get(reason, 0) + 1
        self.log.append((step, reason, tuple(fields), caller))

    def violate(self, step, reason, fields, caller, why):
        self.violations.append((step, reason, tuple(fields or ()), caller, why))
        self.invalid = True
        if self.strict:
            raise HonestyViolation(f"step {step}: {why} (reason={reason!r}, fields={list(fields or ())}, "
                                   f"caller={caller})")

    def summary(self) -> dict:
        return {"episode": self.episode, "counts": dict(self.counts), "violations": len(self.violations),
                "invalid": self.invalid, "errors": len(self.errors)}


@dataclass
class Obs:
    """One honest observation. Holds no simulator handle and no object ids."""
    step: int
    rgb: object
    depth: object
    pose: dict
    dets: list = field(default_factory=list)   # [{type, state, states, conf, bbox_xyxy, point, point_ok, source}]
    holding: Optional[str] = None
    #: E3 (2026-10-13): boxes BELOW the operating confidence (``HonestAgent.low_conf`` <= conf < ``conf``), empty unless the
    #: caller asked for them. Never folded into memory; only the look-alike check (DESSTA ambiguity points) reads them.
    low: list = field(default_factory=list)


# ------------------------------------------------------------------------------------------------- pure helpers
def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0.0 else None


def ground_point(depth, bbox_xyxy, tol_abs: float = 0.10, tol_rel: float = 0.10):
    """Pick the screen point to act on for a detection box, from depth only.

    Returns ``((px, py), ok)``. The box centre wins when its depth agrees with the median depth of the box
    (within ``max(tol_abs, tol_rel * median)``); otherwise the nearest interior pixel that agrees. ``ok`` is False
    ("ungroundable") when the box has no valid depth or no pixel agrees -- the point is then the bare centre.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
    cx, cy = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
    if depth is None:
        return (cx, cy), False
    import numpy as np
    d = np.asarray(depth, dtype="float32")
    h, w = d.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    cx, cy = max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
    win = d[y1:y2 + 1, x1:x2 + 1]
    valid = np.isfinite(win) & (win > 0)
    if not valid.any():
        return (cx, cy), False
    med = float(np.median(win[valid]))
    tol = max(tol_abs, tol_rel * med)
    agree = valid & (np.abs(win - med) <= tol)
    c = _num(d[cy, cx])
    if c is not None and abs(c - med) <= tol:
        return (cx, cy), True
    ys, xs = np.nonzero(agree)
    if len(xs) == 0:
        return (cx, cy), False
    k = int(np.argmin((xs + x1 - cx) ** 2 + (ys + y1 - cy) ** 2))
    return (int(xs[k] + x1), int(ys[k] + y1)), True


def _caller(depth: int = 3) -> str:
    try:
        f = sys._getframe(depth)
        return f"{f.f_globals.get('__name__', '?')}:{f.f_code.co_name}"
    except ValueError:
        return "?"


def _sanitize(err: str) -> str:
    """Actuator error text can name hidden objects; decisions only ever see a coarse category."""
    return "refused" if err else ""


# ----------------------------------------------------------------------------------------------- the agent
class HonestAgent:
    """Wraps a ``ThorAgent`` (kept private) and a detector; gives the agent what it may know.

    ``mode="oracle"`` keeps the old privileged path for comparison: ``observe`` reads ground truth and ``act``
    accepts an ``object_id``. Both modes go through the same audit so the comparison is accountable.
    """

    def __init__(self, thor_agent, detector=None, mode: str = "honest", audit: MetaAudit = None,
                 conf: float = 0.25, force: Optional[bool] = None, smooth=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "honest" and detector is None:
            raise RuntimeError("honest mode needs a detector (missing weights is a hard error; never oracle)")
        self._thor = thor_agent
        self.detector = detector
        self.mode = mode
        self.audit = audit or MetaAudit(strict=True)
        self.conf = conf
        self.force = (mode == "oracle") if force is None else bool(force)
        self.step = 0
        self.holding: Optional[str] = None       # own belief from own successful actions; "?" = unknown type
        self.history = []                         # [(step, action, target_type, ok)] -- no ids, ever
        self._last_prim = None
        self._phase = "act"                       # "act" | "scoring"
        self._depth_checked = False
        #: the last pose this agent itself read through pose() (+ its step). A display (the console map) draws the
        #: agent from this so it never adds reads of its own to the audit.
        self.last_pose: Optional[dict] = None
        #: smooth display profile: None, a dict, or a callable returning one (the console's live motion setting)
        self.smooth = smooth
        self.sends = 0                            # physical sends (sub-sends included); ``step`` = logical primitives
        self.last_subs = 0                        # physical sends of the last logical primitive
        self._sub = None                          # (k, n) while a subdivided primitive is being sent (audit caller)
        self._swing = None                        # door openness the caller itself commanded / saw (open/close)
        self._last_err = ""
        #: E3: when set (by the executor, only while it searches for a look-alike-prone type), ``observe`` also returns the
        #: boxes with ``low_conf <= conf < self.conf`` in ``Obs.low``; ``Obs.dets`` is unchanged (the same boxes >= conf)
        self.low_conf: Optional[float] = None

    # -------------------------------------------------------------- episode
    def reset_episode(self, episode=None, holding: Optional[str] = None):
        """New episode: fresh audit, history and holding belief (``holding`` = what setup put in hand, as history)."""
        self.audit.new_episode(episode)
        self.history, self.holding, self.step = [], holding, 0
        if holding:
            self.history.append((0, "setup:holding", holding, True))

    # -------------------------------------------------------------- the single door
    def meta(self, reason: Optional[str], fields=None) -> dict:
        """Read privileged simulator state. ``fields`` = dotted paths; ``None`` = whole record (scoring/oracle)."""
        caller = _caller()
        fields = ["*"] if fields is None else list(fields)
        why = self._check(reason, fields)
        if self._sub is not None:
            caller = f"{caller}#sub{self._sub[0]}/{self._sub[1]}"
        if why:
            self.audit.violate(self.step, reason, fields, caller, why)   # raises when strict
            # non-strict: the read is recorded (episode invalid) but the privileged data is WITHHELD (s7 audit)
            self.audit.record(self.step, reason if reason in REASONS else "untagged", fields, caller)
            return {} if fields == ["*"] else {f: None for f in fields}
        self.audit.record(self.step, reason, fields, caller)
        return self._read(fields)

    def _check(self, reason, fields) -> str:
        if reason not in REASONS:
            return "untagged privileged read"
        if reason == "oracle":
            return "" if self.mode == "oracle" else "oracle read in honest mode"
        if reason == "scoring":
            return "" if self._phase == "scoring" else "scoring read outside the scoring phase"
        bad = [f for f in fields if f not in ALLOWED_FIELDS[reason]]
        if bad:
            return f"field(s) {bad} not allowed under {reason!r}"
        need = RETURN_OF.get(reason)
        if need and self._last_prim != need:
            return f"{reason!r} return read after {self._last_prim!r}, not {need!r}"
        if self._phase == "scoring" and reason != "scoring":
            return "decision-time read inside the scoring phase"
        return ""

    @contextlib.contextmanager
    def scoring(self):
        """Phase for goal checks: ``meta('scoring')`` is legal here, and observe/move/act are not."""
        prev, self._phase = self._phase, "scoring"
        try:
            yield self
        finally:
            self._phase = prev

    def _acting(self, what):
        if self._phase == "scoring":
            self.audit.violate(self.step, "scoring", (), _caller(), f"{what} called inside the scoring phase")
            # non-strict too: nothing is ever sent or observed while scoring (s7 audit: fail closed)
            raise HonestyViolation(f"{what} inside the scoring phase is never executed")

    # -------------------------------------------------------------- proprioception + actuators
    def pose(self) -> dict:
        m = self.meta("pose", ["agent.position", "agent.rotation", "agent.cameraHorizon", "agent.isStanding",
                               "cameraPosition"])
        p = {"position": m["agent.position"], "rotation": m["agent.rotation"], "horizon": m["agent.cameraHorizon"],
             "standing": m["agent.isStanding"], "camera": m["cameraPosition"]}
        self.last_pose = {**copy.deepcopy(p), "step": self.step}
        return p

    def _phys(self, action, **kw) -> bool:
        """One physical send + its audited success bool (error text logged for the breakdown only)."""
        self._thor.do(action, **kw)
        self.sends += 1
        self.last_subs += 1
        m = self.meta("actuator", ["lastActionSuccess", "errorMessage"])
        ok = bool(m["lastActionSuccess"])
        if not ok:
            self.audit.errors.append((self.step, action, m.get("errorMessage") or ""))
        self._last_err = "" if ok else _sanitize(m.get("errorMessage") or "x")
        return ok

    def _profile(self) -> Optional[dict]:
        s = self.smooth
        if callable(s):
            try:
                s = s()
            except Exception:
                s = None
        return s if isinstance(s, dict) and s else None

    def _send(self, action, **kw) -> bool:
        """One LOGICAL primitive: ``step`` +1; with a smooth profile it may be several physical sends (see top)."""
        self.step += 1
        self._last_prim = action
        self.last_subs = 0
        swing, self._swing = self._swing, None
        prof = self._profile() if (action in SMOOTH_MOVES or action in SMOOTH_TURNS
                                   or action in ("OpenObject", "CloseObject")) else None
        if prof is None:
            return self._phys(action, **kw)
        try:
            if action in ("OpenObject", "CloseObject"):
                return self._send_door(action, kw, prof, swing)
            return self._send_split(action, kw, prof)
        finally:
            self._sub = None

    @staticmethod
    def _pause(prof):
        p = float(prof.get("pause") or 0.0)
        if p > 0:
            time.sleep(p)

    def _send_split(self, action, kw, prof) -> bool:
        """A move / turn / look as ``n`` equal sub-sends. A refusal part-way UNDOES the sub-sends already made (in
        reverse) so a refused logical move leaves the pose as it was -- honest_map checks exactly that from pose,
        and a failed undo shows up there as 'a failed move still moved' (reported, never hidden)."""
        if action in SMOOTH_MOVES:
            key, unit = "moveMagnitude", float(prof.get("step") or 0.0)
        else:
            key = "degrees"
            unit = float(prof.get("look" if action in ("LookUp", "LookDown") else "turn") or 0.0)
        total = kw.get(key)
        if total is None or unit <= 0:
            return self._phys(action, **kw)
        total = float(total)
        n = int(math.ceil(abs(total) / unit - 1e-9))
        if n <= 1:
            return self._phys(action, **kw)
        part = {**kw, key: round(total / n, 6)}
        last = part
        if key == "degrees":
            # AI2-THOR refuses turns / looks that are not a multiple of 0.1 deg (seen live 2026-10-07: "LookDown action
            # requires degree value to be a multiple of 0.1f"): equal 0.1-deg parts, the remainder in the last one
            part = {**kw, key: round(total / n, 1)}
            last = {**kw, key: round(total - part[key] * (n - 1), 6)}
        done = 0
        for k in range(n):
            self._sub = (k + 1, n)
            ok = self._phys(action, **(last if k == n - 1 else part))
            self._pause(prof)
            if not ok:
                break
            done += 1
        if done == n:
            return True
        err = self._last_err
        back = SMOOTH_MOVES.get(action) or SMOOTH_TURNS[action]
        for k in range(done):
            self._sub = (n + k + 1, n)             # undo sends are numbered after the forward ones
            self._phys(back, **part)
            self._pause(prof)
        self._last_err = err or "refused"
        return False

    def _send_door(self, action, kw, prof, swing) -> bool:
        """Open / close through openness steps, ONLY from an openness the caller itself commanded or saw (``swing``:
        0.0 = seen closed, 1.0 = opened by this agent). The object handle in ``kw`` was resolved once for this
        logical action and is used only inside it (then dropped by ``act``). A refusal mid-swing falls through to
        the one canonical call, whose success bool is the result."""
        n = int(prof.get("doors") or 1)
        if swing is None or n <= 1:
            return self._phys(action, **kw)
        s0 = max(0.0, min(1.0, float(swing)))
        if action == "OpenObject":
            target = float(kw.get("openness", 1.0))
            vals = [s0 + (target - s0) * k / n for k in range(1, n)]
        else:
            vals = [s0 * (1.0 - k / n) for k in range(1, n)]
        base = {k: v for k, v in kw.items() if k != "openness"}
        for i, v in enumerate(vals):
            self._sub = (i + 1, n)
            ok = self._phys("OpenObject", **base, openness=round(max(0.02, min(1.0, v)), 4))
            self._pause(prof)
            if not ok:
                break
        self._sub = (n, n)
        ok = self._phys(action, **kw)
        self._pause(prof)
        return ok

    def move(self, prim: str, **kw) -> bool:
        self._acting("move")
        if prim not in MOVE_PRIMS:
            raise ValueError(f"{prim!r} is not an honest locomotion primitive")
        ok = self._send(prim, **kw)
        self.history.append((self.step, prim, None, ok))
        return ok

    def reachable_positions(self) -> list:
        """Map-build only (coverage report): the simulator's reachable grid."""
        self._acting("reachable_positions")
        self._send("GetReachablePositions")
        return self.meta("map-build", ["actionReturn"])["actionReturn"] or []

    def act(self, action: str, point=None, target_type: Optional[str] = None, object_id: Optional[str] = None,
            swing: Optional[float] = None, **kw) -> dict:
        """Interact with whatever is at screen ``point`` (pixels in the last frame). Returns ``{ok, err}``.

        The point is fixed by the caller BEFORE resolution; it is resolved once by a raw ``GetObjectInFrame`` and
        the opaque id goes straight into the action call and is dropped -- never returned, stored or parsed.
        ``target_type`` is the caller's own belief (the detection it aimed at), used for the holding belief.
        ``object_id`` is accepted in oracle mode only. ``swing`` (open/close, smooth display only) is the door
        openness the caller itself saw or commanded (0.0 seen closed, 1.0 opened by it); None = no door animation.
        """
        self._acting("act")
        if action not in TARGET_PRIMS and action not in HAND_PRIMS:
            raise ValueError(f"{action!r} is not an interaction primitive")
        if object_id is not None and self.mode != "oracle":
            self.audit.violate(self.step, "oracle", ("objectId",), _caller(2), "object_id passed in honest mode")
            object_id = None
        args = dict(kw)
        if action in TARGET_PRIMS:
            oid = object_id
            if oid is None:
                if point is None:
                    return {"ok": False, "err": "no-point"}
                oid = self._resolve(point)
                if not oid:
                    self.history.append((self.step, action, target_type, False))
                    return {"ok": False, "err": "nothing-at-point"}
            args["objectId"] = oid
            if action not in ("CookObject", "DirtyObject", "UseUpObject"):
                args.setdefault("forceAction", self.force)
        if action in ("OpenObject", "CloseObject") and swing is not None:
            self._swing = float(swing)
        ok = self._send(action, **args)
        args = oid = None                          # the id is used for this ONE logical action, then dropped
        if ok and action == "PickupObject":
            self.holding = target_type or "?"
        elif ok and action in RELEASE_PRIMS:
            self.holding = None
        self.history.append((self.step, action, target_type, ok))
        return {"ok": ok, "err": "" if ok else self._last_err}

    def _resolve(self, point) -> Optional[str]:
        w, h = self._wh
        px, py = point
        if not ok_range(px, w) or not ok_range(py, h):
            return None
        if not self._send("GetObjectInFrame", x=(px + 0.5) / w, y=(py + 0.5) / h):
            return None
        rid = self.meta("action-resolve", ["actionReturn"])["actionReturn"]
        return rid if isinstance(rid, str) and rid else None

    # -------------------------------------------------------------- perception
    @property
    def _wh(self):
        f = self._thor.frame
        return (int(f.shape[1]), int(f.shape[0])) if f is not None else (1, 1)

    def observe(self) -> Obs:
        self._acting("observe")
        import numpy as np
        rgb = np.asarray(self._thor.frame).copy()
        depth = self._thor.depth_frame
        if not self._depth_checked:
            if depth is None:
                raise RuntimeError("HonestAgent needs a controller started with render_depth=True")
            self._depth_checked = True
        depth = np.asarray(depth, dtype="float32").copy()
        pose = self.pose()
        low = []
        if self.mode == "oracle":
            dets = self._oracle_dets(depth)
        else:
            dets = self._detect(rgb)
            if self.low_conf is not None:
                low = [d for d in dets if d["conf"] < self.conf]
                dets = [d for d in dets if d["conf"] >= self.conf]
            for d in dets + low:
                d["point"], d["point_ok"] = ground_point(depth, d["bbox_xyxy"])
        return Obs(step=self.step, rgb=rgb, depth=depth, pose=pose, dets=dets, holding=self.holding, low=low)

    def _detect(self, rgb) -> list:
        inspecting = self.holding is not None           # proprioception: the specialist runs while holding
        conf = self.conf if self.low_conf is None else min(self.conf, float(self.low_conf))
        try:
            raw = self.detector.detect(rgb, conf=conf, inspecting=inspecting)
        except TypeError:
            raw = self.detector.detect(rgb, conf=conf)
        out = []
        for d in raw:
            out.append({"type": d.get("type"), "state": d.get("state") or "?", "states": dict(d.get("states") or {}),
                        "conf": float(d.get("conf", 0.0)), "bbox_xyxy": tuple(d["bbox_xyxy"]),
                        "source": d.get("source", "primary")})
        return out

    # --- ADAPTERS BEGIN: the only code allowed to touch simulator internals (see the static guard) ---------------
    def _read(self, fields) -> dict:
        md = self._thor.event.metadata
        if fields == ["*"]:
            return copy.deepcopy(md)
        out = {}
        for path in fields:
            v = md
            for part in path.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            out[path] = copy.deepcopy(v)
        return out

    def _oracle_dets(self, depth) -> list:
        """Oracle comparison mode only: ground-truth visible objects as pseudo-detections (tag ``oracle``)."""
        md = self.meta("oracle")
        boxes = getattr(self._thor.event, "instance_detections2D", None) or {}
        out = []
        for o in md.get("objects", []):
            if not o.get("visible"):
                continue
            b = boxes.get(o["objectId"])
            bbox = tuple(int(v) for v in b) if b is not None else None
            pt, ok = ground_point(depth, bbox) if bbox else (None, False)
            out.append({"type": o.get("objectType"), "state": self._oracle_state(o), "states": {}, "conf": 1.0,
                        "bbox_xyxy": bbox, "point": pt, "point_ok": ok, "source": "oracle",
                        "object_id": o["objectId"]})
        return out

    #: oracle mode only: ground-truth state words for a pseudo-detection (a transform of an ``oracle`` read)
    _ORACLE_AXES = (("openable", "isOpen", "open", "closed"), ("toggleable", "isToggled", "on", "off"),
                    ("sliceable", "isSliced", "sliced", "whole"), ("cookable", "isCooked", "cooked", "raw"),
                    ("breakable", "isBroken", "broken", "intact"), ("dirtyable", "isDirty", "dirty", "clean"),
                    ("canFillWithLiquid", "isFilledWithLiquid", "filled", "empty"))

    @classmethod
    def _oracle_state(cls, o: dict) -> str:
        words = [(yes if o.get(flag) else no) for cap, flag, yes, no in cls._ORACLE_AXES if o.get(cap)]
        return ", ".join(words) or "?"
    # --- ADAPTERS END --------------------------------------------------------------------------------------------


def ok_range(v, n) -> bool:
    return 0 <= v < n


def load_detector(scene: str = "FloorPlan1", weights_dir: str = None, mode: str = "auto"):
    """The step-3 FloorPlan1 stack via ``perception.v3_paths``. Missing primary weights is a hard error."""
    import perception
    v3 = perception.v3_paths(scene, weights_dir)
    if not v3["primary"]:
        raise FileNotFoundError(f"no step-3 primary weights for {scene} (perception.v3_paths) -- honest mode "
                                "never falls back to oracle")
    spec = perception.Detector(v3["specialist"]) if v3["specialist"] else None
    states = (perception.StateEstimator(scene, weights_dir)
              if perception.StateEstimator.available(scene, weights_dir) else None)
    return perception.TwoStageDetector(perception.Detector(v3["primary"]), spec, mode=mode, states=states)
