"""Function Library for AI2-THOR — a comprehensive wrapper over the default embodied agent.

Each method maps to ONE AI2-THOR action: these are the Level-1 "atomic" skills of the thesis hierarchy. Methods
mutate `self.event` (the latest ai2thor Event) and return it; check `agent.success` / `agent.error` afterwards.

The `OBJECT_ACTION_SPECS` registry + `applicable_actions(obj)` expose which object-interaction actions are valid for a
given object based on its metadata flags — used by the GUI to render context-aware buttons, and later by the LLM
planner to know its tool set.

Run headless to self-test:   MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python thor_actions.py
"""
from __future__ import annotations

import os

# GPU/display safety net so the simulator picks the RTX 4060 even if the shell env wasn't set (see docs/ai2thor-wsl-setup.md).
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from ai2thor.controller import Controller
except Exception as _exc:       # ai2thor is WSL-only (CLAUDE.md); the Windows dev box has none.
    # Guarding this import is what makes the SKILL LAYER unit-testable off-GPU. `Controller` is referenced in
    # exactly one place (`ThorAgent.__init__`), so every generator below -- walk_to, the composites, search_for --
    # can be bound to a fake agent and driven headlessly. Constructing a real ThorAgent still fails, loudly and
    # with the original error, which is the only behaviour that ever depended on this being a hard import.
    _THOR_IMPORT_ERROR = _exc                  # `except ... as` unbinds at block exit; keep our own name
    class Controller:                          # noqa: N801 - stands in for the real class, refuses to be built
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "ai2thor is not installed in this interpreter, so a live ThorAgent cannot be created. "
                "Run simulator code inside the WSL venv (see CLAUDE.md). "
                f"Original import error: {_THOR_IMPORT_ERROR!r}")

# ----------------------------------------------------------------------------------------------------------------------
# Scene catalogue (iTHOR). Kitchens are the thesis's primary environment.
# ----------------------------------------------------------------------------------------------------------------------
KITCHEN_SCENES = [f"FloorPlan{i}" for i in range(1, 31)]
LIVING_ROOM_SCENES = [f"FloorPlan{i}" for i in range(201, 231)]
BEDROOM_SCENES = [f"FloorPlan{i}" for i in range(301, 331)]
BATHROOM_SCENES = [f"FloorPlan{i}" for i in range(401, 431)]
ALL_SCENES = KITCHEN_SCENES + LIVING_ROOM_SCENES + BEDROOM_SCENES + BATHROOM_SCENES
LIQUIDS = ["water", "coffee", "wine"]

# Object state fields that an action can *change* (pose/visibility deliberately excluded) — used by snapshot/diff
# to detect an action's real effect for the self-improving loop.
EFFECT_KEYS = ("isOpen", "isToggled", "isPickedUp", "isSliced", "isCooked", "isBroken", "isDirty",
               "isFilledWithLiquid", "fillLiquid", "temperature", "isUsedUp", "parentReceptacles")


class ThorAgent:
    """High-level, fully-documented control surface over an AI2-THOR Controller (default agent)."""

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 600,
        height: int = 600,
        grid_size: float = 0.25,
        rotate_degrees: float = 30.0,
        look_degrees: float = 30.0,
        visibility_distance: float = 1.5,
        field_of_view: float = 90.0,
        render_depth: bool = False,
        render_instance_seg: bool = False,
        render_semantic_seg: bool = False,
        render_normals: bool = False,
        agent_mode: str = "default",
        snap_to_grid: bool = False,
        **controller_kwargs,
    ):
        self.move_magnitude = grid_size
        self.rotate_degrees = rotate_degrees
        # Motion granularity for the LOCOMOTION GENERATORS (walk_to / _rotate_to / turn_to_object / scan_for and
        # every composite skill built on them). A GUI renders exactly one frame per `yield`, so these two numbers
        # ARE the frame rate of the robot's movement: at 30 deg / 0.25 m per yield a turn-and-walk is a handful of
        # frames and reads as a slideshow. Mutate this dict to re-time every skill at once -- the alternative was
        # threading `step`/`turn` through a dozen composite signatures that never wanted to know about them.
        self.motion = {"step": grid_size, "turn": rotate_degrees}
        #: Which sensor streams are currently rendering. Seeded from the constructor so `set_sensors` can tell a
        #: real change from a no-op -- see its docstring for why a spurious one is expensive.
        self._sensors = {"depth": bool(render_depth), "instance": bool(render_instance_seg),
                         "semantic": bool(render_semantic_seg), "normals": bool(render_normals)}
        #: Outcome of the most recent `search_for`. Pre-seeded so a caller can read it before any search has run.
        self.last_search = {"found": False, "type": None, "receptacle": None, "stops_planned": 0,
                            "stops_visited": 0, "visited": [], "elapsed": 0.0, "reason": "no search yet"}
        self.look_degrees = look_degrees
        #: The range gate on AI2-THOR's `visible` flag, tracked here because `scan_scene` widens it for the
        #: duration of a dataset scan and has to put it back (see `set_sensors`).
        self.visibility_distance = float(visibility_distance)
        self.field_of_view = field_of_view   # vertical FOV (deg) the controller renders with — used by depth_backproject
        self.controller = Controller(
            agentMode=agent_mode,
            scene=scene,
            gridSize=grid_size,
            rotateStepDegrees=rotate_degrees,
            visibilityDistance=visibility_distance,
            width=width,
            height=height,
            fieldOfView=field_of_view,
            renderDepthImage=render_depth,
            renderInstanceSegmentation=render_instance_seg,
            renderSemanticSegmentation=render_semantic_seg,
            renderNormalsImage=render_normals,
            # snapToGrid MUST default to False: ai2thor rejects a non-orthogonal rotateStepDegrees (e.g. 30) while
            # snapToGrid=True. The GUI uses free rotation/step sizes, so we disable grid-snapping (verified via the
            # action audit against the live simulator).
            snapToGrid=snap_to_grid,
            **controller_kwargs,
        )
        self.event = self.controller.last_event

    # ------------------------------------------------------------------ core
    def do(self, action: str, **kwargs):
        """Send any raw AI2-THOR action. Escape hatch for actions not wrapped below."""
        if getattr(self, "_logging_actions", False):
            self._action_log.append((action, dict(kwargs)))
        self.event = self.controller.step(action=action, **kwargs)
        return self.event

    def pass_(self):
        """No-op step returning fresh metadata without moving (handy to refresh visibility/physics)."""
        return self.do("Pass")

    def done(self):
        """Cheap no-op that clears rendered metadata for the next step."""
        return self.do("Done")

    @property
    def success(self) -> bool:
        return bool(self.event.metadata["lastActionSuccess"])

    @property
    def error(self) -> str:
        return self.event.metadata.get("errorMessage", "") or ""

    @property
    def scene(self) -> str:
        """Name of the currently-loaded FloorPlan (from live metadata)."""
        return self.event.metadata.get("sceneName", "") or ""

    @property
    def frame(self):
        """Egocentric RGB frame as an (H, W, 3) numpy array."""
        return self.event.frame

    @property
    def third_party_frames(self):
        return self.event.third_party_camera_frames

    def stop(self):
        self.controller.stop()

    def gpu_status(self, probe_frames: int = 15) -> dict:
        """Best-effort check of whether rendering is GPU-accelerated.

        Measures live render FPS (CPU/llvmpipe ≈ 1-3 fps vs GPU ≈ 20-40 fps at 600px) and reads the GL renderer
        string. Returns {fps, renderer, gpu(bool), gpu_name, adapter_pref}.
        """
        import os
        import re
        import subprocess
        import time

        t0 = time.time()
        for _ in range(max(1, probe_frames)):
            self.controller.step(action="Pass")
        dt = time.time() - t0
        self.event = self.controller.last_event
        fps = (probe_frames / dt) if dt > 0 else 0.0

        renderer = ""
        try:
            out = subprocess.run(["glxinfo", "-B"], capture_output=True, text=True, timeout=6).stdout
            for line in out.splitlines():
                if "OpenGL renderer string" in line:
                    renderer = line.split(":", 1)[1].strip()
                    break
                if not renderer and line.strip().startswith("Device:"):
                    renderer = line.split(":", 1)[1].strip()
        except Exception:
            renderer = ""

        rl = renderer.lower()
        software = any(s in rl for s in ("llvmpipe", "softpipe", "swrast"))
        is_gpu = (any(g in rl for g in ("nvidia", "geforce", "radeon", "intel")) or
                  (not software and fps >= 10))
        name = ""
        if renderer:
            m = re.search(r"\(([^)]*(?:NVIDIA|GeForce|Radeon|Intel)[^)]*)\)", renderer)
            name = (m.group(1) if m else renderer).strip()
        return {"fps": round(fps, 1), "renderer": renderer, "gpu": bool(is_gpu),
                "gpu_name": name, "adapter_pref": os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME", "")}

    # ------------------------------------------------------------- navigation
    def move_ahead(self, magnitude: Optional[float] = None):
        return self.do("MoveAhead", moveMagnitude=magnitude or self.move_magnitude)

    def move_back(self, magnitude: Optional[float] = None):
        return self.do("MoveBack", moveMagnitude=magnitude or self.move_magnitude)

    def move_left(self, magnitude: Optional[float] = None):
        return self.do("MoveLeft", moveMagnitude=magnitude or self.move_magnitude)

    def move_right(self, magnitude: Optional[float] = None):
        return self.do("MoveRight", moveMagnitude=magnitude or self.move_magnitude)

    def rotate_right(self, degrees: Optional[float] = None):
        return self.do("RotateRight", degrees=degrees or self.rotate_degrees)

    def rotate_left(self, degrees: Optional[float] = None):
        return self.do("RotateLeft", degrees=degrees or self.rotate_degrees)

    def look_up(self, degrees: Optional[float] = None):
        return self.do("LookUp", degrees=degrees or self.look_degrees)

    def look_down(self, degrees: Optional[float] = None):
        return self.do("LookDown", degrees=degrees or self.look_degrees)

    def crouch(self):
        return self.do("Crouch")

    def stand(self):
        return self.do("Stand")

    def teleport(self, x, y, z, rotation_y=None, horizon=None, standing=True, force=False):
        kw = {"position": {"x": x, "y": y, "z": z}, "standing": standing, "forceAction": force}
        if rotation_y is not None:
            kw["rotation"] = {"x": 0, "y": rotation_y, "z": 0}
        if horizon is not None:
            kw["horizon"] = horizon
        return self.do("Teleport", **kw)

    def reachable_positions(self) -> list:
        self.do("GetReachablePositions")
        return self.event.metadata.get("actionReturn") or []

    def teleport_to_random_reachable(self):
        positions = self.reachable_positions()
        if not positions:
            return self.event
        p = random.choice(positions)
        return self.teleport(p["x"], p["y"], p["z"], rotation_y=random.choice([0, 90, 180, 270]))

    def teleport_full(self, x, y, z, rotation_y, horizon=0.0, standing=True):
        """Fully-specified deterministic teleport (every pose field pinned) — ideal for reproducible resets."""
        return self.do("TeleportFull", position={"x": x, "y": y, "z": z},
                       rotation={"x": 0, "y": rotation_y, "z": 0}, horizon=horizon, standing=standing)

    def interactable_poses(self, object_id: str, max_distance: Optional[float] = None) -> list:
        """Agent poses from which `object_id` is visible/interactable — foundation for a navigation planner."""
        kw = {"objectId": object_id}
        if max_distance is not None:
            kw["maxDistance"] = max_distance
        self.do("GetInteractablePoses", **kw)
        return self.event.metadata.get("actionReturn") or []

    # ------------------------------------------------------------ manipulation
    def pickup(self, object_id: str, force: bool = False):
        o = self.object_by_id(object_id)
        if o and o.get("isPickedUp"):
            return self.pass_()                         # already in hand → succeed (no-op)
        return self.do("PickupObject", objectId=object_id, forceAction=force)

    def put(self, receptacle_object_id: str, force: bool = False):
        """Put the currently-held object into/onto the given receptacle."""
        return self.do("PutObject", objectId=receptacle_object_id, forceAction=force)

    def drop(self, force: bool = False):
        return self.do("DropHandObject", forceAction=force)

    def throw(self, magnitude: float = 100.0, force: bool = False):
        return self.do("ThrowObject", moveMagnitude=magnitude, forceAction=force)

    def move_held(self, ahead: float = 0.0, right: float = 0.0, up: float = 0.0):
        return self.do("MoveHeldObject", ahead=ahead, right=right, up=up, forceVisible=False)

    def rotate_held(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
        return self.do("RotateHeldObject", pitch=pitch, yaw=yaw, roll=roll)

    # ------------------------------------------------------------- object state
    def open(self, object_id: str, openness: float = 1.0, force: bool = False):
        return self.do("OpenObject", objectId=object_id, openness=openness, forceAction=force)

    def close(self, object_id: str, force: bool = False):
        return self.do("CloseObject", objectId=object_id, forceAction=force)

    def toggle_on(self, object_id: str, force: bool = False):
        return self.do("ToggleObjectOn", objectId=object_id, forceAction=force)

    def toggle_off(self, object_id: str, force: bool = False):
        return self.do("ToggleObjectOff", objectId=object_id, forceAction=force)

    def slice(self, object_id: str, force: bool = False):
        return self.do("SliceObject", objectId=object_id, forceAction=force)

    def cook(self, object_id: str, force: bool = False):
        return self.do("CookObject", objectId=object_id, forceAction=force)

    def break_object(self, object_id: str, force: bool = False):
        return self.do("BreakObject", objectId=object_id, forceAction=force)

    def dirty(self, object_id: str, force: bool = False):
        return self.do("DirtyObject", objectId=object_id, forceAction=force)

    def clean(self, object_id: str, force: bool = False):
        return self.do("CleanObject", objectId=object_id, forceAction=force)

    def fill_with_liquid(self, object_id: str, liquid: str = "water", force: bool = False):
        return self.do("FillObjectWithLiquid", objectId=object_id, fillLiquid=liquid, forceAction=force)

    def empty_liquid(self, object_id: str, force: bool = False):
        return self.do("EmptyLiquidFromObject", objectId=object_id, forceAction=force)

    def use_up(self, object_id: str, force: bool = False):
        return self.do("UseUpObject", objectId=object_id, forceAction=force)

    def push(self, object_id: str, magnitude: float = 100.0, force: bool = False):
        return self.do("PushObject", objectId=object_id, moveMagnitude=magnitude, forceAction=force)

    def pull(self, object_id: str, magnitude: float = 100.0, force: bool = False):
        return self.do("PullObject", objectId=object_id, moveMagnitude=magnitude, forceAction=force)

    def directional_push(self, object_id: str, magnitude: float = 100.0, angle: float = 0.0, force: bool = False):
        """Push a moveable object at an arbitrary angle (0..360°) — generalizes push/pull."""
        return self.do("DirectionalPush", objectId=object_id, moveMagnitude=magnitude, pushAngle=angle, forceAction=force)

    # ----------------------------------------------------------------- scene
    def reset(self, scene: Optional[str] = None):
        self.event = self.controller.reset(scene=scene) if scene else self.controller.reset()
        return self.event

    def set_sensors(self, depth: bool = False, instance: bool = False,
                    semantic: bool = False, normals: bool = False, force: bool = False,
                    visibility_distance: Optional[float] = None):
        """Reconfigure which sensor streams render (fewer = higher FPS).

        **IDEMPOTENT.** Returns immediately when the requested streams are already the ones rendering, and that
        is the important part: changing render flags is only possible through ``controller.reset()``, which
        RELOADS the scene — lighting, object states, open doors and the agent's pose all snap back. A console
        that called this on every camera-view switch was therefore reloading the world to move between two
        views that render the *same* RGB buffer (RGB and Detect), and re-reloading it every time you re-picked
        the view you were already on. From the driving seat that reads as the lighting going haywire.

        When a change genuinely is required the reload is unavoidable — but the agent's POSE is restored
        afterwards, so at least the camera does not teleport back across the room. Object states still reset;
        that is AI2-THOR's behaviour and cannot be worked around here. Third-party (map) cameras are cleared.

        `visibility_distance` (metres, None = leave as it is) rides along because it is an *initialization*
        parameter too, so changing it costs the same reload and would otherwise cost a second one. `scan_scene`
        widens it for the duration of a dataset scan; nothing else should, because the same number decides how
        far away a `PickupObject` is allowed to reach from.

        `force=True` reloads even when nothing changed (the escape hatch for a genuinely stuck renderer).
        """
        want = {"depth": bool(depth), "instance": bool(instance),
                "semantic": bool(semantic), "normals": bool(normals)}
        vis = float(getattr(self, "visibility_distance", 1.5) if visibility_distance is None
                    else visibility_distance)
        if not force and want == getattr(self, "_sensors", None) and abs(vis - getattr(self, "visibility_distance", 1.5)) < 1e-6:
            return self.event                       # nothing to change: a reset here would be a free scene wipe
        try:
            pose = self.agent_pose()
        except Exception:
            pose = None
        scene = self.event.metadata.get("sceneName")
        self.event = self.controller.reset(
            scene=scene,
            visibilityDistance=vis,
            renderDepthImage=want["depth"],
            renderInstanceSegmentation=want["instance"],
            renderSemanticSegmentation=want["semantic"],
            renderNormalsImage=want["normals"],
        )
        self._sensors = want
        self.visibility_distance = vis
        if pose:
            try:
                p = pose["position"]
                self.teleport_full(p["x"], p["y"], p["z"], pose["rotation"]["y"],
                                   horizon=pose.get("horizon", 0.0), standing=pose.get("standing", True))
            except Exception:
                pass
        return self.event

    def randomize_materials(self):
        return self.do("RandomizeMaterials")

    def randomize_lighting(self):
        return self.do("RandomizeLighting")

    def randomize_colors(self):
        return self.do("RandomizeColors")

    def random_spawn(self, seed: int = 0):
        return self.do("InitialRandomSpawn", randomSeed=seed, forceVisible=False, numPlacementAttempts=5)

    def add_top_down_camera(self):
        """Add an orthographic bird's-eye third-party camera; its frame appears in `third_party_frames[0]`."""
        props = self.do("GetMapViewCameraProperties").metadata.get("actionReturn")
        if props:
            self.do("AddThirdPartyCamera", **props, skyboxColor="white")
        return self.event

    def map_view_camera_properties(self) -> Optional[dict]:
        """Orthographic bird's-eye camera parameters for the current scene (for custom overview cameras)."""
        return self.do("GetMapViewCameraProperties").metadata.get("actionReturn")

    def update_third_party_camera(self, camera_id: int, position=None, rotation=None, field_of_view=None):
        """Reposition an existing third-party camera (e.g. to follow the agent)."""
        kw = {"thirdPartyCameraId": camera_id}
        if position is not None:
            kw["position"] = position
        if rotation is not None:
            kw["rotation"] = rotation
        if field_of_view is not None:
            kw["fieldOfView"] = field_of_view
        return self.do("UpdateThirdPartyCamera", **kw)

    def change_resolution(self, width: int, height: int):
        """Resize the rendered camera output at runtime (i.e. the 'simulator window' size)."""
        return self.do("ChangeResolution", x=int(width), y=int(height))

    # --------------------------------------------------------- sensing frames
    def _safe_frame(self, *attrs):
        """First available Event frame attribute (None if its render flag is off)."""
        for a in attrs:
            try:
                f = getattr(self.event, a)
            except Exception:
                f = None
            if f is not None:
                return f
        return None

    @property
    def depth_frame(self):
        """HxW float32 depth in meters — needs render_depth=True."""
        return self._safe_frame("depth_frame")

    @property
    def instance_segmentation_frame(self):
        """HxWx3 uint8 per-instance colour mask — needs render_instance_seg=True."""
        return self._safe_frame("instance_segmentation_frame")

    @property
    def semantic_segmentation_frame(self):
        """HxWx3 uint8 per-class colour mask — needs render_semantic_seg=True."""
        return self._safe_frame("semantic_segmentation_frame", "class_segmentation_frame")

    @property
    def normals_frame(self):
        """HxWx3 uint8 surface-normals image — needs render_normals=True."""
        return self._safe_frame("normals_frame")

    @property
    def color_to_object_id(self) -> dict:
        """Mapping from instance-segmentation colour → objectId (needs render_instance_seg)."""
        return getattr(self.event, "color_to_object_id", {}) or {}

    @property
    def instance_masks(self) -> dict:
        """objectId → boolean HxW mask (needs render_instance_seg)."""
        return getattr(self.event, "instance_masks", {}) or {}

    # ---------------------------------------------------------- capture / photos
    def frame_of_kind(self, kind: str):
        """Raw frame array for a capture kind: rgb|depth|instance|semantic|normals|map (None if unavailable)."""
        k = kind.lower()
        if k == "rgb":
            return self.frame
        if k == "depth":
            return self.depth_frame
        if k in ("instance", "instance_seg", "instanceseg"):
            return self.instance_segmentation_frame
        if k in ("semantic", "class", "semantic_seg"):
            return self.semantic_segmentation_frame
        if k == "normals":
            return self.normals_frame
        if k == "map":
            return self.third_party_frames[0] if self.third_party_frames else None
        raise ValueError(f"unknown frame kind: {kind!r}")

    def save_frame(self, path: str, kind: str = "rgb") -> str:
        """Save one frame (rgb|depth|instance|semantic|normals|map) to `path` as PNG; returns the path."""
        from PIL import Image
        import numpy as np
        frame = self.frame_of_kind(kind)
        if frame is None:
            raise RuntimeError(
                f"No '{kind}' frame available — enable the matching render flag "
                f"(render_depth/render_instance_seg/render_semantic_seg/render_normals) at construction, "
                f"or add a top-down camera first for 'map'.")
        if kind.lower() == "depth":
            Image.fromarray(depth_to_gray(frame), mode="L").save(path)
        else:
            Image.fromarray(np.asarray(frame).astype("uint8")).save(path)
        return path

    def capture(self, directory: str = "captures", prefix: Optional[str] = None, kinds=("rgb",)) -> list:
        """Save the requested frame kinds into `directory`; returns saved paths ('!'-prefixed entries = failures)."""
        import os as _os
        import time as _time
        _os.makedirs(directory, exist_ok=True)
        self._cap_n = getattr(self, "_cap_n", 0) + 1
        scene = self.event.metadata.get("sceneName", "scene")
        base = prefix or f"{scene}_{_time.strftime('%H%M%S')}_{self._cap_n:03d}"
        saved = []
        for kind in kinds:
            path = _os.path.join(directory, f"{base}_{kind}.png")
            try:
                saved.append(self.save_frame(path, kind))
            except Exception as exc:
                saved.append(f"!{kind}: {exc}")
        return saved

    # ----------------------------------------------- frame queries & navigation
    def object_in_frame(self, x: float, y: float) -> Optional[str]:
        """objectId at normalized screen point (x,y in [0,1], origin top-left). Needs instance segmentation."""
        self.do("GetObjectInFrame", x=float(x), y=float(y))
        return self.event.metadata.get("actionReturn") if self.success else None

    # Broad horizontal surfaces that things REST ON — their collider spans the whole top, so GetObjectInFrame returns
    # them even when you click an item sitting on them. Deprioritised in object_in_frame_best so the item wins.
    _SURFACE_TYPES = frozenset({
        "CounterTop", "Floor", "Shelf", "ShelvingUnit", "DiningTable", "SideTable", "CoffeeTable", "Desk",
        "StoveBurner", "Sink", "SinkBasin", "Stool", "Chair", "ArmChair", "Sofa", "Bed", "TVStand", "Dresser",
    })

    def object_in_frame_best(self, x: float, y: float) -> Optional[str]:
        """objectId under a click, snapping to a small item resting on a big surface.

        ``GetObjectInFrame`` raycasts the nearest collider; for an apple on a counter that collider is usually the
        COUNTER (it spans the whole top), so a naive click selects the counter. When the direct hit is a broad surface
        (``_SURFACE_TYPES``) — or nothing — this samples a small ring around the click and prefers a pickupable /
        openable / toggleable object hit nearby, so clicking the apple selects the apple. Falls back to the direct hit
        when no better object is found (never worse than ``object_in_frame``).
        """
        direct = self.object_in_frame(x, y)
        od = self.object_by_id(direct) if direct else None
        if od and od.get("objectType") not in self._SURFACE_TYPES:
            return direct                                   # direct hit is already a real object — done (1 raycast)
        radii = (0.0, 0.018, 0.035, 0.06)                   # multi-scale: small items on broad counters can be missed
        best, best_score = direct, -1e9
        offsets = [(0.0, 0.0)]
        for r in radii[1:]:
            offsets.extend([(r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)])
        for dx, dy in offsets:
            oid = self.object_in_frame(min(0.999, max(0.0, x + dx)), min(0.999, max(0.0, y + dy)))
            if not oid:
                continue
            o = self.object_by_id(oid) or {}
            surface = o.get("objectType") in self._SURFACE_TYPES
            s = 0.0
            if o.get("pickupable"):
                s += 8.0
            if o.get("visible"):
                s += 2.0
            if o.get("openable") or o.get("toggleable"):
                s += 1.0
            if surface:
                s -= 6.0
            s -= (dx * dx + dy * dy) * 35                    # tie-break toward the exact click point
            s -= float(o.get("distance") or 0.0) * 0.02
            if s > best_score:
                best_score, best = s, oid
        return best

    def coordinate_from_raycast(self, x: float, y: float) -> Optional[dict]:
        """World {x,y,z} struck by a ray cast through normalized screen point (x,y in [0,1])."""
        self.do("GetCoordinateFromRaycast", x=float(x), y=float(y))
        return self.event.metadata.get("actionReturn") if self.success else None

    def spawn_coordinates_above_receptacle(self, object_id: str, anywhere: bool = False) -> list:
        """Candidate world points on/above a receptacle where a held object could be placed."""
        self.do("GetSpawnCoordinatesAboveReceptacle", objectId=object_id, anywhere=anywhere)
        return self.event.metadata.get("actionReturn") or []

    def place_object_at_point(self, object_id: str, position: dict, rotation: Optional[dict] = None,
                              force_kinematic: bool = False):
        """Teleport an object to a world position (physics-independent placement).

        NOTE: AI2-THOR's PlaceObjectAtPoint takes `rotation` + `forceKinematic` (NOT forceAction).
        """
        kw = {"objectId": object_id, "position": position, "forceKinematic": force_kinematic}
        if rotation is not None:
            kw["rotation"] = rotation
        return self.do("PlaceObjectAtPoint", **kw)

    def aim_angles(self, object_id: str):
        """``(yaw, horizon)`` that would centre `object_id` in the frame from where the agent stands, or None.

        Uses the true eye point (`metadata['cameraPosition']`) and the object's bounding-box centre, so the
        target lands in the middle of the view both horizontally and vertically. Split out of
        `look_at_object` so the same geometry can drive an ANIMATED aim (`aim_at_object`) as well as the
        instant one.
        """
        import math
        obj = self.object_by_id(object_id)
        if not obj:
            return None
        pose = self.agent_pose()
        cam = self.event.metadata.get("cameraPosition")
        if not cam:                                # fallback eye height if the build doesn't report it
            cam = dict(pose["position"])
            cam["y"] += 0.675 if pose["standing"] else 0.40
        target = (obj.get("axisAlignedBoundingBox") or {}).get("center") or obj["position"]
        dx, dy, dz = target["x"] - cam["x"], target["y"] - cam["y"], target["z"] - cam["z"]
        horiz = math.hypot(dx, dz)
        yaw = math.degrees(math.atan2(dx, dz)) % 360
        horizon = -math.degrees(math.atan2(dy, horiz))     # AI2-THOR: +horizon looks down, -looks up
        return yaw, max(-30.0, min(60.0, horizon))         # clamp to the agent's pitch limits

    def _pitch_to(self, target_horizon: float, step: Optional[float] = None):
        """Generator: tilt the camera toward `target_horizon` in `step`° chunks, one rendered frame each."""
        step = max(1.0, float(self.motion["turn"] if step is None else step))
        for _ in range(60):                                # hard bound: a refused LookUp must not spin forever
            cur = self.agent_pose()["horizon"]
            remaining = target_horizon - cur
            if abs(remaining) <= 0.75:
                return
            # AI2-THOR rejects a LookUp/LookDown whose degrees are not a multiple of 0.1 ("requires degree value
            # to be a multiple of 0.1f"). `remaining` is an arbitrary float, so the last, partial tilt of nearly
            # every aim was refused -- the walk still arrived, but reported failure.
            d = round(min(step, abs(remaining)), 1)
            if d < 0.1:
                return
            (self.look_down if remaining > 0 else self.look_up)(d)
            if not self.success:
                return
            if abs(self.agent_pose()["horizon"] - cur) < 1e-3:
                return                                     # the pitch limit; further requests are no-ops
            yield "looking"

    def aim_at_object(self, object_id: str, turn: Optional[float] = None):
        """Generator: SWING the camera onto `object_id` — yaw, then pitch — over several rendered frames.

        `look_at_object` reaches the identical pose with a single teleport, which lands as a hard cut at the end
        of every walk. The end state is the same; this just takes the journey visibly, which is the difference
        between a robot that turns to look at something and one that blinks and is already looking at it.
        """
        ang = self.aim_angles(object_id)
        if not ang:
            return
        yaw, horizon = ang
        yield from self._rotate_to(yaw, turn)
        yield from self._pitch_to(horizon, turn)

    def _closest_pose(self, object_id: str, poses: list) -> Optional[dict]:
        """The interactable pose NEAREST the object.

        `GetInteractablePoses` returns poses in navmesh order and guarantees only that the object is reachable
        from each, not that any of them is close. `poses[0]` is routinely metres further away than another valid
        one — which for a pickup is the difference between standing at the counter and reaching across it.
        """
        import math
        if not poses:
            return None
        obj = self.object_by_id(object_id)
        if not obj:
            return poses[0]
        t = (obj.get("axisAlignedBoundingBox") or {}).get("center") or obj["position"]
        return min(poses, key=lambda p: math.hypot(p["x"] - t["x"], p["z"] - t["z"]))

    def close_in(self, object_id: str, min_distance: float = 0.55, step: Optional[float] = None,
                 max_steps: int = 10):
        """Generator: edge forward until as close to `object_id` as the geometry allows.

        Run after arriving at an interactable pose, for actions where being NEAR the object matters rather than
        merely being able to reach it. Stops the moment a step fails or stops closing the gap — the aim is to
        stand at the object, not to push through it — so it is safe to call unconditionally.
        """
        step = max(0.02, float(self.motion["step"] if step is None else step))
        for _ in range(max(1, int(max_steps))):
            o = self.object_by_id(object_id)
            if not o or o.get("isPickedUp"):
                return
            d = o.get("distance")
            if d is None or d <= min_distance:
                return
            before = d
            self.move_ahead(min(step, max(0.02, d - min_distance)))
            if not self.success:
                return                                     # blocked: this IS as close as the geometry allows
            yield "closing in"
            after = (self.object_by_id(object_id) or {}).get("distance")
            if after is None or after >= before - 1e-3:
                return                                     # no longer closing the gap

    def look_at_object(self, object_id: str) -> Optional[dict]:
        """Aim the camera (yaw *and* pitch) so `object_id` sits centered in the frame, without moving.

        Instant (one teleport). Prefer :meth:`aim_at_object` anywhere a GUI is rendering the result.
        """
        import math
        obj = self.object_by_id(object_id)
        if not obj:
            return None
        pose = self.agent_pose()
        cam = self.event.metadata.get("cameraPosition")
        if not cam:                                # fallback eye height if the build doesn't report it
            cam = dict(pose["position"])
            cam["y"] += 0.675 if pose["standing"] else 0.40
        target = (obj.get("axisAlignedBoundingBox") or {}).get("center") or obj["position"]
        dx, dy, dz = target["x"] - cam["x"], target["y"] - cam["y"], target["z"] - cam["z"]
        horiz = math.hypot(dx, dz)
        yaw = math.degrees(math.atan2(dx, dz)) % 360
        horizon = -math.degrees(math.atan2(dy, horiz))     # AI2-THOR: +horizon looks down, -looks up
        horizon = max(-30.0, min(60.0, horizon))           # clamp to the agent's pitch limits
        p = pose["position"]
        return self.teleport(p["x"], p["y"], p["z"], rotation_y=yaw,
                             horizon=horizon, standing=pose["standing"], force=True)

    def face_object(self, object_id: str):
        """'Look at' `object_id`: aim the camera so the object is centered in the frame (yaw + pitch)."""
        return self.look_at_object(object_id)

    def go_to(self, object_id: str, max_distance: Optional[float] = None, center: bool = True) -> Optional[dict]:
        """Teleport to a pose from which `object_id` is interactable, then center it in view. Returns the pose or None."""
        o = self.object_by_id(object_id)
        if o and o.get("isPickedUp"):
            return {"held": True}                  # already holding it; nothing to navigate to
        poses = self.interactable_poses(object_id, max_distance=max_distance)
        if not poses:
            return None
        p = self._closest_pose(object_id, poses) or poses[0]
        self.teleport_full(p["x"], p["y"], p["z"], p["rotation"],
                           horizon=p.get("horizon", 0.0), standing=p.get("standing", True))
        if center:                                 # aim the camera so the object lands in the middle of the frame
            self.look_at_object(object_id)
        return p

    # ------------------------------------------------------ visible locomotion
    def _path_corners(self, start: dict, target: dict) -> Optional[list]:
        """World-space waypoints (collision-free straight segments) from start→target, via the navmesh."""
        self.do("GetShortestPathToPoint",
                position={"x": start["x"], "y": start["y"], "z": start["z"]},
                target={"x": target["x"], "y": target["y"], "z": target["z"]})
        if not self.success:
            return None
        ret = self.event.metadata.get("actionReturn") or {}
        return ret.get("corners") or None

    def _rotate_to(self, target_yaw: float, turn: Optional[float] = None):
        """Generator: rotate in place toward `target_yaw` in `turn`° chunks (so the turn is visible).

        `turn=None` takes the agent's current motion granularity (see `self.motion`).
        """
        turn = self.motion["turn"] if turn is None else turn
        cur = self.agent_pose()["rotation"]["y"]
        remaining = abs((target_yaw - cur + 540) % 360 - 180)      # shortest angular distance
        right = ((target_yaw - cur + 540) % 360 - 180) > 0          # >0 ⇒ clockwise (RotateRight)
        while remaining > 0.5:
            d = min(turn, remaining)
            self.do("RotateRight" if right else "RotateLeft", degrees=d)
            remaining -= d
            yield "turning"

    def turn_to_object(self, object_id: str, turn: Optional[float] = None):
        """Generator: spin in place to face and center `object_id` (yaw chunks, then precise pitch)."""
        import math
        obj = self.object_by_id(object_id)
        if not obj:
            yield "not found"
            return
        a = self.agent_pose()["position"]
        t = (obj.get("axisAlignedBoundingBox") or {}).get("center") or obj["position"]
        yaw = math.degrees(math.atan2(t["x"] - a["x"], t["z"] - a["z"])) % 360
        yield from self._rotate_to(yaw, turn)
        ang = self.aim_angles(object_id)
        if ang:
            yield from self._pitch_to(ang[1], turn)             # tilt onto it rather than snapping
        yield "centered"

    def walk_to(self, object_id: str, max_distance: Optional[float] = None,
                step: Optional[float] = None, turn: Optional[float] = None,
                max_steps: Optional[int] = None):
        """Generator: WALK to an interactable pose for `object_id` using Move/Rotate primitives, then center it.

        Yields a short status label after every controller micro-step, so a GUI can repaint between moves and
        show real-time locomotion. Falls back to a teleport if the navmesh has no path. On arrival the object is
        centered in the frame.
        """
        import math
        step = self.motion["step"] if step is None else step
        turn = self.motion["turn"] if turn is None else turn
        # The budget is a PATH LENGTH (~20 m), not a step count: at the finer granularities a fixed 80 steps
        # would strand the agent mid-kitchen with "arrived" never reached.
        max_steps = int(round(20.0 / max(0.02, step))) if max_steps is None else max_steps
        held = self.object_by_id(object_id)
        if held and held.get("isPickedUp"):
            yield "arrived"                                # you're already holding it; nothing to walk to
            return
        poses = self.interactable_poses(object_id, max_distance=max_distance)
        if not poses:
            yield "no path"
            return
        goal = self._closest_pose(object_id, poses)         # nearest valid pose, not whatever came back first
        corners = self._path_corners(self.agent_pose()["position"], goal)
        if not corners:
            self.go_to(object_id)                              # fallback: instant teleport
            yield "teleported (no nav path)"
            return
        budget = max_steps
        for c in corners[1:]:                                  # corners[0] is the current position
            cx, cz = c["x"], c["z"]
            a = self.agent_pose()["position"]
            if math.hypot(cx - a["x"], cz - a["z"]) < 1e-3:
                continue
            yaw = math.degrees(math.atan2(cx - a["x"], cz - a["z"])) % 360
            yield from self._rotate_to(yaw, turn)              # face the next waypoint
            while budget > 0:                                  # step forward to the waypoint
                budget -= 1
                a = self.agent_pose()["position"]
                dx, dz = cx - a["x"], cz - a["z"]
                dist = math.hypot(dx, dz)
                if dist < step * 0.55:
                    break
                yaw = math.degrees(math.atan2(dx, dz)) % 360
                ddiff = (yaw - self.agent_pose()["rotation"]["y"] + 540) % 360 - 180
                if abs(ddiff) > 3:                             # correct small heading drift
                    self.do("RotateRight" if ddiff > 0 else "RotateLeft", degrees=abs(ddiff))
                    yield "turning"
                self.move_ahead(min(step, dist))
                if not self.success:
                    # Blocked. Try to squeeze past with smaller steps before resorting to a hop — a teleport
                    # cannot be animated, so every one of them is a visible cut in the feed.
                    squeezed = False
                    for frac in (0.5, 0.25):
                        self.move_ahead(min(step * frac, dist))
                        if self.success:
                            yield "squeezing past"
                            squeezed = True
                            break
                    if squeezed:
                        continue
                    self.teleport(cx, a["y"], cz, rotation_y=yaw, force=True)
                    yield "stepping (assisted)"
                    break
                yield "stepping"
        yield from self.aim_at_object(object_id, turn)          # arrive LOOKING TOWARD it, over several frames
        yield "arrived"

    def walk_to_point(self, x: float, z: float, step: Optional[float] = None, turn: Optional[float] = None,
                      max_steps: Optional[int] = None):
        """Generator: WALK to a world (x, z) — the point-target twin of :meth:`walk_to`.

        A search that covers unexplored parts of the room has nothing to aim at, only somewhere to be. Without
        this it teleported, which is both a visible cut and a lie about how the robot got there.
        """
        import math
        step = self.motion["step"] if step is None else step
        turn = self.motion["turn"] if turn is None else turn
        max_steps = int(round(20.0 / max(0.02, step))) if max_steps is None else max_steps
        a0 = self.agent_pose()["position"]
        corners = self._path_corners(a0, {"x": x, "y": a0["y"], "z": z})
        if not corners:
            yield "no path"
            return
        budget = max_steps
        for c in corners[1:]:
            cx, cz = c["x"], c["z"]
            a = self.agent_pose()["position"]
            if math.hypot(cx - a["x"], cz - a["z"]) < 1e-3:
                continue
            yaw = math.degrees(math.atan2(cx - a["x"], cz - a["z"])) % 360
            yield from self._rotate_to(yaw, turn)
            while budget > 0:
                budget -= 1
                a = self.agent_pose()["position"]
                dx, dz = cx - a["x"], cz - a["z"]
                dist = math.hypot(dx, dz)
                if dist < step * 0.55:
                    break
                self.move_ahead(min(step, dist))
                if not self.success:
                    yield "blocked"
                    return
                yield "stepping"
        yield "arrived"

    # ----------------------------------------------------------------- queries
    def objects(self) -> list:
        return self.event.metadata["objects"]

    def object_by_id(self, object_id: str) -> Optional[dict]:
        return next((o for o in self.objects() if o["objectId"] == object_id), None)

    def objects_of_type(self, object_type: str) -> list:
        return [o for o in self.objects() if o["objectType"] == object_type]

    def visible_objects(self) -> list:
        return [o for o in self.objects() if o.get("visible")]

    def pickupable_objects(self) -> list:
        return [o for o in self.objects() if o.get("pickupable")]

    def receptacles(self) -> list:
        return [o for o in self.objects() if o.get("receptacle")]

    def held_object(self) -> Optional[dict]:
        return next((o for o in self.objects() if o.get("isPickedUp")), None)

    def agent_pose(self) -> dict:
        a = self.event.metadata["agent"]
        return {"position": a["position"], "rotation": a["rotation"], "horizon": a["cameraHorizon"], "standing": a.get("isStanding", True)}

    def nearest_visible(self, predicate: Callable[[dict], bool]) -> Optional[dict]:
        candidates = [o for o in self.visible_objects() if _safe(predicate, o)]
        return min(candidates, key=lambda o: o.get("distance", 9e9)) if candidates else None

    def resolve(self, name_or_id: Optional[str], prefer_visible: bool = True) -> Optional[dict]:
        """Find an object by exact objectId, exact objectType, or fuzzy type substring.

        Among matches, prefer visible ones, then the nearest. This is the single place composite skills and the
        planner use to turn a loose name ('apple', 'fridge') into a concrete object.
        """
        if not name_or_id:
            return None
        exact = self.object_by_id(name_or_id)
        if exact:
            return exact
        key = str(name_or_id).lower().strip()
        objs = self.objects()
        matches = [o for o in objs if o["objectType"].lower() == key]
        if not matches:
            matches = [o for o in objs if key in o["objectType"].lower()]
        if not matches:
            return None
        if prefer_visible:
            matches = [o for o in matches if o.get("visible")] or matches
        return min(matches, key=lambda o: o.get("distance", 9e9))

    def find(self, object_type: str, visible_only: bool = False) -> list:
        """All objects whose type contains `object_type` (case-insensitive), nearest first."""
        key = object_type.lower().strip()
        pool = self.visible_objects() if visible_only else self.objects()
        hits = [o for o in pool if key in o["objectType"].lower()]
        return sorted(hits, key=lambda o: o.get("distance", 9e9))

    def objects_in_receptacle(self, receptacle_id_or_type: str) -> list:
        """Objects currently contained in/on the given receptacle (resolves a type to the nearest match)."""
        rec = self.resolve(receptacle_id_or_type, prefer_visible=False)
        if not rec:
            return []
        ids = set(rec.get("receptacleObjectIds") or [])
        return [o for o in self.objects() if o["objectId"] in ids]

    def temperature(self, object_id_or_type: str) -> Optional[str]:
        """Temperature of a target: 'Hot' | 'Cold' | 'RoomTemp' (None if unknown)."""
        o = self.resolve(object_id_or_type, prefer_visible=False)
        return (o or {}).get("temperature") if o else None

    def describe(self, max_items: int = 40, visible_only: bool = True) -> str:
        """Compact scene summary for a human or an LLM: agent state + grouped objects with key flags."""
        held = self.held_object()
        pose = self.agent_pose()
        pool = self.visible_objects() if visible_only else self.objects()
        pool = sorted(pool, key=lambda o: o.get("distance", 9e9))[:max_items]
        head = (f"scene={self.event.metadata.get('sceneName')}  "
                f"held={held['objectType'] if held else 'none'}  "
                f"standing={pose['standing']}")
        lines = [f"- {o['objectType']} ({o.get('distance', 0):.1f}m){(' ' + object_state_flags(o)) if object_state_flags(o) else ''}"
                 for o in pool]
        return head + "\n" + "\n".join(lines)

    def is_visible(self, object_type: str) -> bool:
        """True if any object whose type contains `object_type` is currently in view."""
        return len(self.find(object_type, visible_only=True)) > 0

    def count(self, object_type: str) -> int:
        """How many objects of (a type containing) `object_type` exist in the scene."""
        return len(self.find(object_type))

    def type_list(self, visible_only: bool = False) -> list:
        """Sorted unique object types present (optionally restricted to what's currently visible)."""
        pool = self.visible_objects() if visible_only else self.objects()
        return sorted({o["objectType"] for o in pool})

    def whats_in(self, receptacle: str) -> str:
        """Comma-separated types contained in/on a receptacle (resolved by type), or '(empty)'."""
        items = self.objects_in_receptacle(receptacle)
        return ", ".join(sorted(o["objectType"] for o in items)) or "(empty)"

    def find_nearest(self, object_type: str, n: int = 1, visible_only: bool = False) -> list:
        """The `n` nearest objects whose type contains `object_type` (nearest first)."""
        return self.find(object_type, visible_only=visible_only)[:max(1, n)]

    def distance_between(self, a: str, b: str) -> Optional[float]:
        """Straight-line world distance between two resolved objects (metres), or None."""
        oa, ob = self.resolve(a, prefer_visible=False), self.resolve(b, prefer_visible=False)
        if not oa or not ob:
            return None
        pa, pb = oa["position"], ob["position"]
        return ((pa["x"] - pb["x"]) ** 2 + (pa["y"] - pb["y"]) ** 2 + (pa["z"] - pb["z"]) ** 2) ** 0.5

    def object_bounds(self, name: str) -> Optional[dict]:
        """Axis-aligned bounding-box size {x,y,z} (metres) of a resolved object, or None."""
        o = self.resolve(name, prefer_visible=False)
        return ((o or {}).get("axisAlignedBoundingBox") or {}).get("size") if o else None

    def find_where(self, predicate: Callable[[dict], bool], visible_only: bool = False) -> list:
        """All objects satisfying `predicate(obj)`, nearest first — a general property filter."""
        pool = self.visible_objects() if visible_only else self.objects()
        return sorted([o for o in pool if _safe(predicate, o)], key=lambda o: o.get("distance", 9e9))

    def objects_near(self, target, radius: float = 1.5) -> list:
        """Objects within `radius` m of a target (object id/type, or a {x,y,z} point), nearest first."""
        if isinstance(target, dict) and "x" in target:
            p = target
            skip = None
        else:
            t = self.resolve(target, prefer_visible=False)
            if not t:
                return []
            p, skip = t["position"], t["objectId"]
        scored = []
        for o in self.objects():
            if o["objectId"] == skip:
                continue
            q = o["position"]
            d = ((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 + (p["z"] - q["z"]) ** 2) ** 0.5
            if d <= radius:
                scored.append((d, o))
        return [o for _, o in sorted(scored, key=lambda t: t[0])]

    def container_chain(self, name: str) -> list:
        """The receptacle chain a resolved object sits in, innermost-first (e.g. Apple -> [Bowl, CounterTop])."""
        o = self.resolve(name, prefer_visible=False)
        chain, seen = [], set()
        while o:
            parents = o.get("parentReceptacles") or []
            if not parents or parents[0] in seen:
                break
            seen.add(parents[0])
            o = self.object_by_id(parents[0])
            chain.append(o["objectType"] if o else parents[0].split("|")[0])
        return chain

    def contents_tree(self, name: str, _depth: int = 0, max_depth: int = 4) -> dict:
        """Nested dict of what a receptacle (recursively) contains: {type: subtree-or-None}."""
        o = self.resolve(name, prefer_visible=False)
        if not o or _depth > max_depth:
            return {}
        tree = {}
        for cid in (o.get("receptacleObjectIds") or []):
            co = self.object_by_id(cid)
            if not co:
                continue
            tree[co["objectType"]] = self.contents_tree(cid, _depth + 1, max_depth) if co.get("receptacle") else None
        return tree

    def path_to(self, name: str, max_distance: Optional[float] = None) -> Optional[list]:
        """Navmesh waypoints to an interactable pose for `name`, WITHOUT moving (None if no path)."""
        o = self.resolve(name, prefer_visible=False)
        if not o:
            return None
        poses = self.interactable_poses(o["objectId"], max_distance=max_distance)
        if not poses:
            return None
        return self._path_corners(self.agent_pose()["position"], poses[0])

    def steps_to(self, name: str, step: float = 0.25) -> Optional[int]:
        """Rough number of move-steps to reach `name` (path length / step); None if unreachable."""
        import math
        corners = self.path_to(name)
        if not corners:
            return None
        total = sum(math.hypot(b["x"] - a["x"], b["z"] - a["z"]) for a, b in zip(corners, corners[1:]))
        return max(0, round(total / step))

    # ---------------------------------------- state snapshots & verification (self-improving loop)
    def snapshot(self, keys: tuple = EFFECT_KEYS) -> dict:
        """Capture a {objectId: {state-key: value}} map of every object — the 'before' for an action diff."""
        return {o["objectId"]: {k: o.get(k) for k in keys} for o in self.objects()}

    def diff_state(self, before: dict, after: Optional[dict] = None, keys: tuple = EFFECT_KEYS) -> list:
        """What changed between a `before` snapshot and `after` (or the live state). Pose/visibility excluded.

        Returns [{objectId, objectType, changes:{key:(old,new)}}] — lets the agent see an action's real effect,
        which is how it judges success and what to write to the knowledge base.
        """
        after = after if after is not None else self.snapshot(keys)
        out = []
        for oid, aft in after.items():
            bef = before.get(oid)
            if bef is None:
                continue
            delta = {k: (bef.get(k), aft.get(k)) for k in keys if bef.get(k) != aft.get(k)}
            if delta:
                out.append({"objectId": oid, "objectType": oid.split("|")[0], "changes": delta})
        return out

    def changed_since(self, before: dict, keys: tuple = EFFECT_KEYS) -> list:
        """Convenience: diff the current live state against an earlier snapshot."""
        return self.diff_state(before, None, keys)

    def object_state(self, name_or_id: str, key: str):
        """Read one state field of a resolved object (None if absent)."""
        o = self.resolve(name_or_id, prefer_visible=False)
        return o.get(key) if o else None

    def verify(self, name_or_id: str, key: str, value) -> bool:
        """True iff a resolved object's `key` currently equals `value` (post-action goal check)."""
        return self.object_state(name_or_id, key) == value

    def spatial_snapshot(self, track: str = "oracle") -> list:
        """Oracle spatial observations for the Phase-5 feedback funnel — one distilled record per positioned object.

        Reads live AI2-THOR metadata (the ground-truth *oracle* observation source) and emits
        ``[{object_type, object_id, receptacle, pos, state, track}]`` — exactly the shape that
        ``EnvFeedback.observe_scene`` / ``Memory.observe_location`` consume for the change-gated spatial writer, so a
        planner/worker can pass `ag.spatial_snapshot()` straight into `feedback.log_step(..., observations=...)`.
        For each object with a known position:

          - ``receptacle`` — the objectType of the FIRST entry in ``parentReceptacles`` (the object's immediate
            container/surface), resolved id→type via a one-pass lookup over ``objects()`` (falling back to the id's
            type prefix if that receptacle isn't in the lookup); ``None`` when the object rests on nothing.
          - ``pos`` — the object's world ``{x, y, z}`` position (drives the writer's move-threshold change gate).
          - ``state`` — the salient state label (``'open'``/``'on'``/… via :func:`perception.state_label`), or ``None``
            for types with no state axis. Perception is imported LAZILY and degrades to ``None`` if it (or its torch/
            numpy deps) is unavailable, so this method never hard-fails on a machine without the vision stack.

        Objects with no ``position`` are skipped (there is nothing spatial to remember). ``track`` labels each record
        with its observation source (``'oracle'`` here); the writer stamps that track onto the stored belief.
        """
        objs = self.objects()
        id_to_type = {o["objectId"]: o["objectType"] for o in objs}   # one pass: receptacle id -> type
        try:
            import perception                                          # light label helpers; torch stays lazy inside
        except Exception:
            perception = None                                         # no vision stack -> state degrades to None
        out = []
        for o in objs:
            pos = o.get("position")
            if not pos:
                continue                                              # nothing spatial to remember
            parents = o.get("parentReceptacles") or []
            receptacle = None
            if parents:
                pid = parents[0]
                receptacle = id_to_type.get(pid) or str(pid).split("|")[0]
            state = None
            if perception is not None:
                try:
                    state = perception.state_label(o) or None         # '' (no axis) -> None, uniform with the fallback
                except Exception:
                    state = None
            out.append({
                "object_type": o.get("objectType"),
                "object_id": o.get("objectId"),
                "receptacle": receptacle,
                "pos": pos,
                "state": state,
                "track": track,
            })
        return out

    # ----------------------------------------------------- composite skills (L2)
    # These chain atomic skills into the task-level actions a planner reaches for. Each is a GENERATOR that yields a
    # short progress label after every controller step — so the GUI can animate them with _drive_gen, while headless
    # code runs them to completion with `drain()`. They navigate (walk_to) as needed and use forceAction on the final
    # interaction so they're robust to small pose/visibility gaps. Check `agent.success` after draining.
    def drain(self, gen) -> bool:
        """Run a skill generator to completion (no animation) and return the final `success`."""
        for _ in gen:
            pass
        return self.success

    def goto(self, name: str):
        """Walk to an object given by id or (fuzzy) type."""
        obj = self.resolve(name)
        if not obj:
            yield f"no '{name}'"
            return
        yield from self.walk_to(obj["objectId"])

    def _approach(self, obj: dict):
        """Walk to an interactable pose in front of the object (helper for the composites below).

        Always approaches unless the object is in hand — being merely 'visible' can mean an awkward side-on
        angle, so we navigate to a proper interactable pose to stand squarely in front of it.
        """
        if not obj.get("isPickedUp"):
            yield from self.walk_to(obj["objectId"])

    def pick_up(self, name: str, force: bool = True):
        """Navigate to and pick up `name`. Retries once after re-approaching on failure."""
        obj = self.resolve(name)
        if not obj:
            yield f"no '{name}'"
            return
        oid = obj["objectId"]
        yield from self._approach(obj)
        self.pickup(oid, force=force)
        yield "picking up"
        if not self.success:
            yield from self.walk_to(oid)
            self.pickup(oid, force=force)
            yield "retry pick up"

    def put_in(self, receptacle: str, force: bool = True):
        """Put the currently-held object into/onto `receptacle` (navigate there first)."""
        if not self.held_object():
            yield "hand is empty"
            return
        rec = self.resolve(receptacle)
        if not rec:
            yield f"no '{receptacle}'"
            return
        rid = rec["objectId"]
        yield from self._approach(rec)
        # A closed fridge / microwave / cabinet / drawer refuses the put ("Target openable Receptacle is CLOSED"),
        # and PLANNER_SYSTEM tells the model that high-level actions handle "their own walking, opening". Until
        # this opened the door, that promise was false for every closed destination -- and every kitchen starts
        # with all of them closed -- so "put the apple in the fridge" could not succeed as a single action. The
        # door is left open: whether to close it is the instruction's business, not this skill's.
        if rec.get("openable") and not rec.get("isOpen"):
            self.open(rid, force=force)
            yield "opening it first"
        self.put(rid, force=force)
        yield "placing"
        if not self.success:
            yield from self.walk_to(rid)
            self.put(rid, force=force)
            yield "retry place"

    def pick_and_place(self, obj_name: str, receptacle: str, force: bool = True):
        """The canonical rearrangement skill: pick up `obj_name`, carry it to `receptacle`, put it down."""
        yield from self.pick_up(obj_name, force=force)
        if not self.held_object():
            yield "could not pick it up"
            return
        yield from self.put_in(receptacle, force=force)
        yield "placed"

    def open_it(self, name: str, openness: float = 1.0, force: bool = True):
        obj = self.resolve(name)
        if not obj:
            yield f"no '{name}'"
            return
        yield from self._approach(obj)
        self.open(obj["objectId"], openness=openness, force=force)
        yield "opening"

    def close_it(self, name: str, force: bool = True):
        obj = self.resolve(name)
        if not obj:
            yield f"no '{name}'"
            return
        yield from self._approach(obj)
        self.close(obj["objectId"], force=force)
        yield "closing"

    def switch(self, name: str, on: bool = True, force: bool = True):
        """Navigate to a toggleable object and turn it on (on=True) or off."""
        obj = self.resolve(name)
        if not obj:
            yield f"no '{name}'"
            return
        yield from self._approach(obj)
        (self.toggle_on if on else self.toggle_off)(obj["objectId"], force=force)
        yield "turning on" if on else "turning off"

    def slice_it(self, name: str, force: bool = True):
        """Slice `name` (the agent should be holding a knife). Navigates to it first."""
        obj = self.resolve(name)
        if not obj:
            yield f"no '{name}'"
            return
        yield from self._approach(obj)
        self.slice(obj["objectId"], force=force)
        yield "slicing"

    def heat_object(self, food: str, appliance: str = "Microwave", force: bool = True):
        """Heat `food` in the microwave: hold it, open→put-in→close→run→open, then take it back out."""
        f = self.resolve(food)
        if not f:
            yield f"no '{food}'"
            return
        if not f.get("isPickedUp"):
            yield from self.pick_up(food, force=force)
            if not self.held_object():
                yield "could not pick up the food"
                return
        m = self.resolve(appliance)
        if not m:
            yield f"no '{appliance}'"
            return
        mid = m["objectId"]
        food_id = f["objectId"]
        yield from self.walk_to(mid)
        self.open(mid, force=force); yield "opening microwave"
        self.put(mid, force=force); yield "putting food in"
        if not self.success:
            # The put is pose-sensitive and intermittently refused; unchecked, the rest of the sequence ran on
            # an empty microwave and the food came back un-heated with every step reporting success. One retry
            # from a fresh approach -- the pattern put_in already uses.
            yield from self.walk_to(mid)
            self.put(mid, force=force); yield "retry putting food in"
        self.close(mid, force=force); yield "closing microwave"
        self.toggle_on(mid, force=force); yield "microwave running"
        self.toggle_off(mid, force=force); yield "stopping microwave"
        self.open(mid, force=force); yield "opening microwave"
        self.pickup(food_id, force=force); yield "taking food out"   # leave it in hand, hot
        yield "heated"

    def cool_object(self, food: str, appliance: str = "Fridge", force: bool = True):
        """Chill `food` in the fridge: hold it, open→put-in→close, then open and take it back out."""
        f = self.resolve(food)
        if not f:
            yield f"no '{food}'"
            return
        if not f.get("isPickedUp"):
            yield from self.pick_up(food, force=force)
            if not self.held_object():
                yield "could not pick up the food"
                return
        fr = self.resolve(appliance)
        if not fr:
            yield f"no '{appliance}'"
            return
        fid = fr["objectId"]
        food_id = f["objectId"]
        yield from self.walk_to(fid)
        self.open(fid, force=force); yield "opening fridge"
        self.put(fid, force=force); yield "putting food in"
        self.close(fid, force=force); yield "closing fridge"
        for _ in range(8):                      # let it chill a few sim frames inside
            self.pass_()
        yield "chilling"
        self.open(fid, force=force); yield "opening fridge"
        self.pickup(food_id, force=force); yield "taking food out"
        yield "chilled"

    def wash_object(self, item: str, force: bool = True):
        """Wash/rinse `item`: place it in the sink basin, run the faucet briefly, then turn it off."""
        it = self.resolve(item)
        if not it:
            yield f"no '{item}'"
            return
        if not it.get("isPickedUp"):
            yield from self.pick_up(item, force=force)
            if not self.held_object():
                yield "could not pick up the item"
                return
        basin = self.resolve("SinkBasin") or self.resolve("Sink")
        if not basin:
            yield "no sink"
            return
        item_id = it["objectId"]
        yield from self.walk_to(basin["objectId"])
        self.put(basin["objectId"], force=force); yield "placing in sink"
        faucet = self.resolve("Faucet")
        if faucet:
            self.toggle_on(faucet["objectId"], force=force); yield "water on"
        self.clean(item_id, force=force); yield "scrubbing clean"   # faucet water alone doesn't clean in this build
        if faucet:
            self.toggle_off(faucet["objectId"], force=force); yield "water off"
        yield "washed"

    def cook_food(self, food: str, force: bool = True):
        """Cook `food`. Cookable items become cooked (and hot) via the microwave; others are just heated."""
        f = self.resolve(food)
        if not f:
            yield f"no '{food}'"
            return
        yield from self.heat_object(food, force=force)
        yield "cooked"

    # ------------------------------------------------ active perception (L?/VLM)
    def look_around(self, turn: Optional[float] = None):
        """Generator: spin a full 360° in place, accumulating every object type seen. Stores `self.last_seen`.

        This is the locomotion half of the thesis's *active perception* component — sweep the view to build an
        up-to-date picture of the scene before deciding what to do.
        """
        # A survey sweep, not locomotion: it keeps its own 45 deg default rather than following self.motion,
        # because perceived-mode runs a detector at every stop and a 8 deg sweep would be 45 inferences.
        turn = 45.0 if turn is None else turn
        seen: set = set()
        steps = max(1, int(round(360.0 / max(1.0, turn))))
        for _ in range(steps):
            for o in self.visible_objects():
                seen.add(o["objectType"])
            self.rotate_right(turn)
            yield "looking around"
        for o in self.visible_objects():
            seen.add(o["objectType"])
        self.last_seen = sorted(seen)
        yield f"saw {len(self.last_seen)} object types"

    def scan_for(self, name: str, turn: Optional[float] = None):
        """Generator: turn in place to bring an object of type `name` into the centre of view.

        Rotates toward the object's known location (so it works at any distance, unlike AI2-THOR's range-limited
        `visible` flag). Reports 'not found' only if no such type exists in the scene.
        """
        obj = self.resolve(name, prefer_visible=False)
        if not obj:
            yield f"{name} not found"
            return
        yield from self.turn_to_object(obj["objectId"], turn=turn)
        yield f"found {name}"

    # -------------------------------------- systematic search (policy lives in sim/search.py)
    def _search_sees(self, object_type: str, detector=None, conf: float = 0.25) -> bool:
        """Is `object_type` in view RIGHT NOW?

        Two sensors, one question, so the same skill works under both perception arms:
          * `detector` given  -> vision only: the type appears in this frame's YOLO detections;
          * otherwise         -> AI2-THOR's own `visible` flag, which is range-gated by `visibilityDistance`
                                 (1.5 m by default) and is therefore a *sighting*, not a lookup of where the
                                 object is. Reading `position`/`distance`/`parentReceptacles` for the target
                                 instead would turn the search into a homing beacon; see sim/search.py's header.
        """
        t = str(object_type or "").strip().lower()
        if not t:
            return False
        if detector is not None:
            try:
                for d in self.detect_objects(detector, conf=conf) or []:
                    if str(d.get("type") or "").strip().lower() == t:
                        return True
            except Exception:
                return False
            return False
        return bool(self.find(object_type, visible_only=True))

    def _swing(self, object_id: str, target: float = 1.0, steps: int = 8):
        """Generator: drive a door's `openness` to `target` over `steps` rendered frames.

        AI2-THOR opens in one controller step, so without this the container you are searching goes from shut to
        wide open between two frames — the moment the search is *about*. Any refusal mid-swing falls through to a
        single canonical call, so the end state matches the un-animated path either way.
        """
        obj = self.object_by_id(object_id) or {}
        cur = obj.get("openness")
        start = float(cur if cur is not None else (1.0 if obj.get("isOpen") else 0.0))
        if steps > 1 and abs(target - start) > 1e-3:
            for k in range(1, int(steps)):
                v = start + (target - start) * (k / float(steps))
                self.open(object_id, openness=max(0.02, min(1.0, v)), force=True)
                if not self.success:
                    break
                yield "opening" if target > start else "closing"
        if target <= 1e-6:
            self.close(object_id, force=True)
        else:
            self.open(object_id, openness=min(1.0, target), force=True)

    def search_for(self, name: str, *, detector=None, memory=None, kb=None, scene: Optional[str] = None,
                   speed: str = "normal", turn: Optional[float] = None, max_stops: Optional[int] = None,
                   on_event=None):
        """GENERATOR: search the scene for `name` systematically, and narrate every decision.

        This is the skill `look_around` and `scan_for` were being mistaken for. Both of those spin **in place**:
        they cannot find anything outside the agent's current field of view, and `scan_for` resolves its target
        from metadata, so it only ever "finds" what the simulator already told it. This one decides **where to go
        and look next**, walks there, opens closed containers, and reports whether it actually saw anything.

        The ordering policy is `sim/search.py` (pure, GPU-free, self-tested). What this method contributes is the
        embodiment: navigation, door-opening, the sweep, and the sensor.

        Knowledge in, in order of authority: a remembered observation of this type (`memory.query_location`), then
        receptacles the knowledge base associates with it, then a curated containment prior, then coverage of the
        discovered nav map. Every optional dependency degrades to "skip that source", so the skill still runs on
        a cold agent with no memory, no KB and no detector.

        Sets ``self.last_search`` -- ``{found, type, receptacle, stops_planned, stops_visited, visited,
        elapsed, reason}`` -- which is how the caller can score the command honestly instead of assuming a search
        that executed is a search that succeeded.

        `on_event(kind, text, data)` is an optional narration sink (the console logs and traces through it). It is
        fully guarded: a failing sink can never break a search.
        """
        import time as _time
        import search as _search

        t0 = _time.time()
        target = _search.normalize_type(name)
        scene = scene or self.scene
        budget = _search.search_budget(speed)
        if max_stops:
            budget["max_stops"] = int(max_stops)
        turn = float(turn if turn is not None else self.motion.get("turn", 15.0))

        def say(kind, text, **data):
            if on_event is None:
                return
            try:
                on_event(kind, text, data)
            except Exception:
                pass

        self.last_search = {"found": False, "type": target, "receptacle": None, "stops_planned": 0,
                            "stops_visited": 0, "visited": [], "elapsed": 0.0, "reason": "not started"}
        if not target:
            say("error", "search: no object type given")
            yield "search: nothing to look for"
            return

        # ---- gather what the agent knows (each source is optional and independently guarded) ----------------
        belief = None
        if memory is not None:
            try:
                b = memory.query_location(scene, target)
                # query_location does NOT filter by scene in SQL -- scene is only a ranking term, so a belief
                # from a DIFFERENT kitchen comes back looking authoritative. Reject it here.
                if b and (b.get("scene") in (None, scene)):
                    belief = b
            except Exception:
                belief = None
        kb_texts = []
        if kb is not None:
            try:
                for hit in (kb.search(f"{target} where location receptacle", limit=20) or []):
                    e = hit.get("entry") or {}
                    kb_texts += [e.get("text"), e.get("lesson"), e.get("problem"), e.get("command")]
                kb_texts = [x for x in kb_texts if x]
            except Exception:
                kb_texts = []
        nav_cells = []
        if memory is not None:
            try:
                import navmap as _navmap
                # cell_size/origin are NOT persisted with the cells (the caller owns the mapping), so these must
                # match what scan_scene/explore_navigation wrote with or the world coords come out wrong.
                nav_cells = [(c["x"], c["z"]) for c in _navmap.NavMap(memory, scene, 0.25, (0.0, 0.0)).cells()
                             if c.get("state") == "free"]
            except Exception:
                nav_cells = []

        pos = self.agent_pose()["position"]
        agent_pos = (pos["x"], pos["z"])
        receptacles = []
        for o in self.objects():
            if not o.get("receptacle"):
                continue
            p3 = o.get("position") or {}
            if "x" not in p3:
                continue
            receptacles.append({"objectId": o["objectId"], "objectType": o["objectType"],
                                "pos": (p3["x"], p3["z"]), "isOpen": bool(o.get("isOpen")),
                                "openable": bool(o.get("openable"))})

        if belief:
            say("recall", f"search: recall - {target} was last seen at "
                          f"{belief.get('receptacle') or '?'} (confidence {float(belief.get('confidence') or 0):.2f})",
                belief=belief.get("receptacle"), confidence=belief.get("confidence"))
        elif memory is not None:
            say("recall", f"search: no memory of {target} in {scene} - going on knowledge and priors")
        if kb is not None and not kb_texts:
            say("recall", "search: knowledge base has nothing about where this belongs - using the built-in prior")

        stops = _search.rank_stops(target, belief=belief, kb_texts=kb_texts, receptacles=receptacles,
                                   agent_pos=agent_pos, nav_cells=nav_cells,
                                   max_stops=budget["max_stops"], frontier_stops=budget["frontier_stops"])
        self.last_search["stops_planned"] = len(stops)
        say("plan", f"search: plan - {_search.summarize(stops)}", stops=stops)

        # ---- walk the plan ----------------------------------------------------------------------------------
        visited = []
        opened = []            # containers THIS search opened, so it can leave the kitchen as it found it
        hold = 2

        def shut():
            """Generator: close every container this search opened.

            A search that leaves four cupboard doors and the fridge hanging open has not finished, it has just
            stopped. This closes ALL of them, including the one the object was found in: a follow-up "pick it
            up" opens the container again by itself (see `_approach_then`), so there is no reason to leave the
            kitchen standing open on the chance that a follow-up arrives.
            """
            while opened:
                oid = opened.pop()
                o = self.object_by_id(oid) or {}
                if not o.get("isOpen"):
                    continue
                say("close", f"search: closing the {o.get('objectType') or 'container'} again")
                yield from self._swing(oid, 0.0, steps=max(2, int(8 * (15.0 / max(1.0, turn)))))
        for i, stop in enumerate(stops, 1):
            if (_time.time() - t0) > budget["max_seconds"]:
                self.last_search["reason"] = "time budget exhausted"
                say("stop", f"search: giving up after {budget['max_seconds']:.0f}s")
                break
            label = stop.get("type") or ("here" if stop["kind"] == "here" else "an unsearched spot")
            say("stop", f"search [{i}/{len(stops)}]: {stop['why']}", stop=stop)

            if stop.get("target"):
                last = ""
                for lab in self.walk_to(stop["target"]):
                    last = lab
                    yield lab
                if last == "no path":
                    # Unreachable is NOT searched -- counting it would let the agent claim it looked somewhere
                    # it never stood.
                    say("skip", f"search: cannot reach {label} - skipping it")
                    continue
            elif stop.get("pos"):
                p = stop["pos"]
                last = ""
                for lab in self.walk_to_point(p[0], p[1]):
                    last = lab
                    yield lab
                if last != "arrived":
                    say("skip", "search: cannot reach that spot - skipping it")
                    continue

            if stop.get("open") and stop.get("target"):
                say("open", f"search: {label} is closed - opening it to look inside")
                yield from self._swing(stop["target"], 1.0, steps=max(2, int(12 * (15.0 / max(1.0, turn)))))
                opened.append(stop["target"])

            visited.append(label)
            self.last_search["stops_visited"] = len(visited)
            for deg, do_detect in _search.sweep_schedule(turn, budget["detect_every"]):
                self.rotate_right(deg)
                yield "searching"
                if not do_detect:
                    continue
                for _ in range(hold):
                    yield "looking"
                if self._search_sees(target, detector):
                    where = stop.get("type") or ""
                    self.last_search.update(found=True, receptacle=where, visited=visited,
                                            elapsed=_time.time() - t0, reason="seen")
                    say("found", f"search: FOUND {target}" + (f" at {where}" if where else " in view"),
                        receptacle=where, stops_visited=len(visited))
                    yield from shut()                          # tidy up before reporting the find
                    yield f"found {target}"
                    return

        yield from shut()                                   # found nothing: leave the room as it was
        self.last_search.update(found=False, visited=visited, stops_visited=len(visited),
                                elapsed=_time.time() - t0,
                                reason=self.last_search.get("reason") or "exhausted the search plan")
        say("notfound", f"search: NOT FOUND - {target} was not at " + (", ".join(visited) or "anywhere reachable"),
            visited=visited)
        yield f"{target} not found"

    # -------------------------------------- Phase-4: per-scene YOLO scan + perception
    # These build the agent's own *state-aware* detector and *discovered* nav map for the current FloorPlan (the thesis's
    # active-perception contribution). Heavy/sibling deps (gen_dataset/navmap/perception, numpy) are imported lazily
    # inside the methods so `import thor_actions` stays light and free of import cycles.
    def _scan_memory(self, memory=None):
        """Resolve the structured Memory to log perception labels / nav cells into (cache one if we must build it)."""
        if memory is not None:
            return memory
        m = getattr(self, "_memory", None)
        if m is not None:
            return m
        import memory as _mem
        self._memory = _mem.Memory()
        return self._memory

    def contents_of(self, object_id: str) -> list:
        """The objects currently inside/on the receptacle `object_id` (empty list if it holds nothing)."""
        ids = set((self.object_by_id(object_id) or {}).get("receptacleObjectIds") or [])
        return [o for o in self.objects() if o["objectId"] in ids] if ids else []

    def peek_inside(self, object_id: str, on_capture=None, *, views: int = 3, openness=(1.0, 0.55),
                    poses: int = 2, swing_steps: int = 6, turn: Optional[float] = None):
        """GENERATOR: open a container, photograph its INTERIOR properly, and shut it again.

        This is the skill the dataset scan was missing. Coverage walks the floor and photographs whatever the
        room happens to be showing; an apple in a shut fridge is shown to nobody, so its class is trained on
        zero images and the detector never learns it. Opening the door is only half of it — a frame aimed at
        the fridge is aimed at the door, with the shelves below and behind it. So this aims at the CONTENTS
        (`scan_plan.aim_point`), from a couple of genuinely different distances (`select_poses`), at whatever
        pitch and stance actually points into the thing (`aim_from` — a drawer is looked down into from a
        crouch, a wall cabinet is looked up at from further back), at more than one door angle
        (`openness_ladder`), with the subject sometimes off-centre and cropped (`view_offsets`).

        `on_capture()` is called once per composed viewpoint — `scan_scene` passes its labelling capture, and a
        caller that just wants to look inside passes nothing.

        **It leaves the container as it found it**, the same discipline `search_for` follows, and it does so
        from a `finally` so that a caller who abandons the generator mid-swing still gets a closed door rather
        than a kitchen standing open.
        """
        import scan_plan

        obj = self.object_by_id(object_id) or self.resolve(object_id, prefer_visible=False)
        if not obj:
            yield f"no '{object_id}'"
            return
        oid = obj["objectId"]
        otype = obj.get("objectType") or "container"
        if not obj.get("openable"):
            yield f"the {otype} does not open"
            return

        was_open = bool(obj.get("isOpen"))
        was_openness = float(obj.get("openness") if obj.get("openness") is not None else (1.0 if was_open else 0.0))
        turn = float(turn if turn is not None else self.motion.get("turn", 15.0))
        swing = max(2, int(swing_steps * (15.0 / max(1.0, turn))))
        levels = [float(v) for v in (openness or (1.0,))] or [1.0]
        pos = obj.get("position") or {}
        if pos.get("x") is None:
            yield f"the {otype} has no position to stand at"
            return
        target_xz = (pos["x"], pos["z"])
        touched = False                       # did WE move this door? (drives the restore below)

        try:
            plan = scan_plan.select_poses(self.interactable_poses(oid), target_xz, max_poses=max(1, int(poses)))
            if not plan:
                yield from self.walk_to(oid)  # no interactable pose came back: walk there and shoot from where we land
                plan = [None]
            for stand_at in plan:
                if stand_at is not None:
                    self.teleport_full(stand_at["x"], stand_at["y"], stand_at["z"], stand_at["rotation"],
                                       horizon=stand_at.get("horizon", 0.0),
                                       standing=stand_at.get("standing", True))
                    if not self.success:
                        continue
                    yield f"standing at the {otype}"
                for level in levels:
                    # Flagged BEFORE the swing, not after: the swing yields a frame per degree of door travel, so
                    # a caller that stops the scan mid-swing leaves a half-open door that the `finally` below has
                    # to know about. Setting this afterwards is how the door got left hanging open.
                    touched = True
                    yield from self._swing(oid, level, steps=swing)
                    # Re-read AFTER the swing: an open door moves the contents' visibility, and on the second
                    # pose the door is already where we left it.
                    aim = scan_plan.aim_point(self.object_by_id(oid) or obj, self.contents_of(oid))
                    here = self.agent_pose()["position"]
                    yaw, horizon, standing = scan_plan.aim_from(aim, here)
                    for dyaw, dpitch in scan_plan.view_offsets(max(1, int(views))):
                        self.teleport(here["x"], here["y"], here["z"], rotation_y=(yaw + dyaw) % 360.0,
                                      horizon=scan_plan.clamp_horizon(horizon + dpitch),
                                      standing=standing, force=True)
                        if not self.success:
                            continue
                        yield f"looking inside the {otype}"
                        if on_capture is not None:
                            on_capture()
            # Normal exit: shut it visibly, so the console sees the door go back rather than teleporting shut.
            if touched and not was_open:
                yield from self._swing(oid, 0.0, steps=swing)
                touched = False
            elif touched and was_open:
                yield from self._swing(oid, was_openness, steps=swing)
                touched = False
            # A refused restore is the one failure that outlives the scan: the operator is left driving a kitchen
            # with a door hanging open and nothing in the transcript said so. Say it. (The `finally` below cannot —
            # GeneratorExit forbids yielding there — so this is the only place the console can hear about it.)
            end = self.object_by_id(oid) or {}
            want = was_openness if was_open else 0.0
            if abs(float(end.get("openness") if end.get("openness") is not None
                         else (1.0 if end.get("isOpen") else 0.0)) - want) > 0.05:
                yield f"WARNING: the {otype} would not go back to how it was — it is still open"
            yield f"finished with the {otype}"
        finally:
            # Safety net for the abandoned-generator path (GeneratorExit forbids yielding, so this one is a
            # plain call): whatever happened above, the door goes back to how we found it.
            if touched:
                if was_open:
                    self.open(oid, openness=was_openness, force=True)
                else:
                    self.close(oid, force=True)

    def scan_scene(self, dataset_root: Optional[str] = None, cell_size: float = 0.25, per_class_target: int = 40,
                   max_positions: int = 24, do_states: bool = True, do_destructive: bool = False, memory=None,
                   *, do_containers: bool = True, max_containers: int = 10, container_views: int = 3,
                   container_openness: int = 2, container_poses: int = 2, visibility_distance: float = 6.0,
                   min_box_area: int = 150):
        """Generator: (re)scan the current scene into a per-scene YOLO dataset + a discovered nav map.

        Drives the whole Phase-4 capture pipeline, yielding short progress strings (like look_around/walk_to) so a GUI
        can stream it. Steps: (1) COVERAGE — spread over the reachable floor (farthest-point, not random), marking the
        discovered occupancy grid (free where we land, blocked where a teleport fails) and auto-labelling frames over a
        precessing schedule of yaws/pitches/stances via gen_dataset.capture_frame; (2) CONTAINERS — open every
        receptacle worth opening and photograph what is inside (see below); (3) BALANCE — top up under-target classes by
        visiting those object types directly, from several viewpoints; (4) reversible STATES — walk to the object,
        checkpoint, flip open/toggle/fill, capture both states, restore; (5) irreversible STATES (do_destructive, LAST)
        — slice/cook/crack a few items, capture, then reload the scene clean; (6) FINALIZE — write data.yaml and report
        a per-class histogram + discovered-vs-oracle nav comparison. Does NOT train (call gen_dataset.train_scene).

        **The container pass is why this method was rewritten.** Coverage photographs what the room is showing, and a
        kitchen shows none of its cupboards' contents: an apple that lives in the fridge appeared in zero training
        frames, so its class was trained on nothing. `do_containers` walks the ranked plan from `scan_plan`, opens each
        one, aims at the CONTENTS rather than the door, shoots it at a few distances/door-angles, and shuts it again —
        `peek_inside` owns that sequence, including putting the room back. It is bounded by `max_containers` and
        priced up front: the scan yields an estimated frame count and duration before it starts moving.

        `visibility_distance` (metres) widens AI2-THOR's range gate FOR THE SCAN ONLY, and is restored afterwards.
        `capture_frame` labels only objects the simulator reports as `visible`, and that flag is range-gated at 1.5 m
        by default — so at the default every object further than that appeared in the frame WITHOUT a label, teaching
        the detector that a fridge four metres away is background. That is a quieter but equally real reason the
        trained detectors were poor.

        `min_box_area` (px²) is the same argument one level down. `capture_frame` defaults to 400 px² — a 20x20 box —
        and at 600px/90° that is an egg photographed from closer than 0.7 m or an apple from closer than 1.2 m.
        Every frame taken from further away had the egg IN it and no label ON it, which is a false negative the
        training run reads as "eggs look like background". 150 px² (~12x12) labels an apple out to about 2 m; below
        that a box really is noise. Raise it if the labels start looking speckled.
        """
        # Instance segmentation feeds the auto-labeler's ground-truth masks; depth feeds depth_backproject; the wider
        # visibility gate is what makes distant objects LABELLED rather than silently unlabelled. NOTE: this reloads
        # the scene (resets pose/state), so it MUST run before any teleport/capture below.
        prev_visibility = float(getattr(self, "visibility_distance", 1.5))
        try:
            self.set_sensors(instance=True, depth=True, visibility_distance=visibility_distance)
        except Exception as exc:                    # an older build that rejects the parameter must not kill the scan
            yield f"scan: keeping the default visibility gate ({exc})"
            self.set_sensors(instance=True, depth=True)
        try:
            yield from self._scan_body(dataset_root, cell_size, per_class_target, max_positions, do_states,
                                       do_destructive, memory, do_containers, max_containers, container_views,
                                       container_openness, container_poses, min_box_area)
        finally:
            # The gate also decides how far a PickupObject may reach, so leaving it wide would quietly change how
            # the whole console behaves after a scan — including a scan the operator cancelled halfway.
            if abs(prev_visibility - float(getattr(self, "visibility_distance", prev_visibility))) > 1e-6:
                try:
                    self.set_sensors(instance=True, depth=True, visibility_distance=prev_visibility)
                except Exception:
                    pass

    def _scan_body(self, dataset_root, cell_size, per_class_target, max_positions, do_states, do_destructive,
                   memory, do_containers, max_containers, container_views, container_openness, container_poses,
                   min_box_area):
        """The scan itself. Split out of :meth:`scan_scene` only so the visibility gate has somewhere to be restored
        from, including when the caller abandons the generator mid-scan."""
        import gen_dataset
        import navmap
        import perception
        import scan_plan

        mem = self._scan_memory(memory)
        # Anchor the dataset dir at this module's directory (sim/) so scan-writes and webapp's train-reads agree
        # regardless of the process CWD (webapp._train_scene_bg reads HERE/datasets/<scene>).
        if dataset_root is None:
            dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        scene = self.scene
        dataset_dir = os.path.join(dataset_root, scene)
        nav = navmap.NavMap(mem, scene, cell_size)

        # Pre-seed the class->id map from the objects present so ids are deterministic regardless of capture order.
        class_to_id: dict = {}
        for n in perception.scene_class_names(self.objects()):
            class_to_id.setdefault(n, len(class_to_id))
        counts: dict = {}      # class_name -> number of frames it appears in (a per-class abundance proxy)
        train_counts: dict = {}   # ...of which are TRAIN frames — the only ones the detector ever learns from
        cap_i = 0              # monotonic index -> stable per-frame uids
        site_i = -1            # capture SITE: one coverage position / one container / one balance target
        pin_split = None       # when set, this site's frames go to that split regardless of the site rota

        def _xz():
            p = self.agent_pose()["position"]
            return p["x"], p["z"]

        def _site():
            """Start a new capture site. Everything shot from here lands on the same side of the train/val split."""
            nonlocal site_i
            site_i += 1

        def _grab():
            nonlocal cap_i
            # Split by SITE, not by frame. A scan takes eight to twelve near-identical frames from one spot, so a
            # 1-in-5 FRAME split puts near-duplicates of every val image into train: the val mAP that comes back
            # then measures memorisation, and a detector that scores well on it can still be useless in the room.
            # Holding out whole sites is the difference between a number that predicts field performance and one
            # that flatters it.
            split = pin_split or ("val" if site_i % 5 == 0 else "train")
            res = gen_dataset.capture_frame(self, scene, dataset_dir, split=split, min_area=min_box_area,
                                            class_to_id=class_to_id, index=cap_i, memory=mem)
            cap_i += 1
            if res.get("written"):
                for c in res.get("classes", []):
                    counts[c] = counts.get(c, 0) + 1
                    if split == "train":
                        train_counts[c] = train_counts.get(c, 0) + 1
            return res

        # (0) PLAN + PRICE — say what this is going to cost before spending twenty minutes of the operator's GPU. ------
        positions = self.reachable_positions() or []
        # Farthest-point, not random: reachable points are dense wherever the floor is open, so a uniform draw
        # spends the budget in the middle of the room and can miss a corner entirely.
        sample = scan_plan.spread_positions(positions, min(max_positions, len(positions)))
        views_per_pos = len(scan_plan.coverage_views(0))
        plan = []
        if do_containers:
            # Priced with the counts we have (none yet), which makes this an upper bound — the real pass re-plans
            # after coverage, by which time some of these classes will already have frames.
            plan = scan_plan.rank_containers(self.objects(), agent_pos=_xz(), counts=counts,
                                             per_class_target=per_class_target, max_containers=max_containers,
                                             class_of=perception.class_name)
        est = scan_plan.estimate_frames(len(sample), views_per_pos, plan, base_views=container_views,
                                        openness_levels=container_openness, max_poses=container_poses)
        yield (f"scan {scene}: {len(sample)} coverage positions x {views_per_pos} views, "
               f"{est['n_containers']} containers to open, {len(class_to_id)} seed classes")
        yield (f"estimate: ~{est['total']} frames ({est['coverage']} coverage + {est['containers']} container) "
               f"~{est['minutes']:.0f} min")

        # (1) COVERAGE -------------------------------------------------------------------------------------------------
        for i, p in enumerate(sample):
            self.teleport(p["x"], p["y"], p["z"], force=True)
            if not self.success:
                nav.mark_blocked(p["x"], p["z"])            # couldn't get there -> discovered as blocked
                yield f"pos {i + 1}/{len(sample)} blocked"
                continue
            ax, az = _xz()
            nav.mark_free(ax, az)                           # we stood here -> discovered as free
            _site()
            # The view schedule precesses with `i`, so the scan covers 16 headings, 4 pitches and both eye heights
            # over the room for the same 8 frames per position the fixed N/E/S/W + level/down grid used to buy.
            for yaw, hz, standing in scan_plan.coverage_views(i):
                self.teleport(p["x"], p["y"], p["z"], rotation_y=yaw, horizon=hz, standing=standing, force=True)
                if not self.success:
                    continue        # a refused pose (a crouch under a counter, say) would otherwise re-shoot the last one
                _grab()
            yield f"pos {i + 1}/{len(sample)} captured ({cap_i} frames, {len(class_to_id)} classes)"

        # (2) CONTAINERS — open what the room keeps hidden and photograph the inside. ----------------------------------
        # This is the phase the dataset was missing. Everything above photographs what the kitchen chooses to show;
        # a shut cupboard shows nothing, so every class that only exists inside one had zero training frames.
        if do_containers:
            plan = scan_plan.rank_containers(self.objects(), agent_pos=_xz(), counts=counts,
                                             per_class_target=per_class_target, max_containers=max_containers,
                                             class_of=perception.class_name)
            yield f"containers: {scan_plan.summarize(plan)}"
            for i, stop in enumerate(plan, 1):
                budget = scan_plan.capture_budget(stop, container_views, container_openness, container_poses)
                yield f"container {i}/{len(plan)}: {stop['why']}"
                before = cap_i
                _site()
                # A container is ONE site, so the site rota would hand a whole cupboard to val — and a class that
                # exists only inside that cupboard then has every frame it will ever get on the held-out side and
                # NONE to train on. That is precisely the starvation this pass exists to end, so a container that
                # is the sole source of an unseen class is pinned to train; the ones that add nothing new still
                # take their turn in the holdout, which is what keeps val honest.
                pin_split = "train" if any(counts.get(c, 0) <= 0 for c in stop["new_classes"]) else None
                try:
                    yield from self.peek_inside(stop["objectId"], on_capture=_grab, views=budget["views"],
                                                openness=budget["openness"], poses=budget["poses"])
                finally:
                    pin_split = None
                found = sorted({c for c in stop["new_classes"] if counts.get(c, 0) > 0})
                yield (f"container {i}/{len(plan)}: {cap_i - before} frame(s) inside the {stop['objectType']}"
                       + (f" — first sight of {', '.join(found[:4])}" if found else ""))

        # (3) BALANCE — top up classes still under target by visiting their object type directly. ----------------------
        def _under():
            return [c for c in class_to_id if counts.get(c, 0) < per_class_target]

        for cls in _under():
            otype = perception.split_class_name(cls)[0]
            visited = 0
            for o in self.objects_of_type(otype):
                if counts.get(cls, 0) >= per_class_target or visited >= 4:
                    break
                oid = o["objectId"]
                pos = o.get("position") or {}
                if pos.get("x") is None:
                    continue
                # Several poses, not `poses[0]`: a class topped up from one spot is topped up with one photograph
                # taken N times, which is what a balance pass is supposed to be fixing.
                _site()
                for q in scan_plan.select_poses(self.interactable_poses(oid), (pos["x"], pos["z"]), max_poses=2):
                    if counts.get(cls, 0) >= per_class_target:
                        break
                    self.teleport_full(q["x"], q["y"], q["z"], q["rotation"],
                                       horizon=q.get("horizon", 0.0), standing=q.get("standing", True))
                    if not self.success:
                        continue
                    self.look_at_object(oid)
                    _grab()
                    visited += 1
            if visited:
                yield f"balance {cls}: +{visited} (now {counts.get(cls, 0)})"

        # (4) REVERSIBLE STATES — flip a few openable/toggleable/fillable objects, capturing each side, then restore. --
        if do_states:
            flipped = 0
            # Whatever happens to be `visible` from wherever BALANCE left the agent — which, with the scan's wide
            # visibility gate, is most of the room. Nearest first, and we walk to each one: a state pair captured
            # from four metres away teaches the detector to tell `Fridge_open` from `Fridge_closed` in forty pixels.
            pool = sorted(self.visible_objects(), key=lambda o: o.get("distance") or 9e9)
            for o in pool:
                if flipped >= 10:
                    break
                oid = o["objectId"]
                is_fill = False
                if o.get("openable"):
                    flip = self.close if o.get("isOpen") else self.open
                elif o.get("toggleable"):
                    flip = self.toggle_off if o.get("isToggled") else self.toggle_on
                elif o.get("canFillWithLiquid"):
                    flip = self.empty_liquid if o.get("isFilledWithLiquid") else self.fill_with_liquid
                    is_fill = True
                else:
                    continue
                pos = o.get("position") or {}
                if pos.get("x") is not None:
                    for q in scan_plan.select_poses(self.interactable_poses(oid), (pos["x"], pos["z"]), max_poses=1):
                        self.teleport_full(q["x"], q["y"], q["z"], q["rotation"],
                                           horizon=q.get("horizon", 0.0), standing=q.get("standing", True))
                        if self.success:
                            self.look_at_object(oid)
                _site()
                cp = self.checkpoint()
                _grab()                                     # current state
                flip(oid, force=True)
                self.pass_()
                _grab()                                     # flipped state
                # checkpoint()/restore() track pose + open/toggle but NOT liquid fill, so a fill/empty flip would leak
                # into every later capture and mislabel frames — undo it explicitly before restoring pose.
                if is_fill:
                    (self.fill_with_liquid if o.get("isFilledWithLiquid") else self.empty_liquid)(oid, force=True)
                    self.pass_()
                self.restore(cp)
                flipped += 1
                yield f"states {o['objectType']} flipped ({flipped})"

        # (5) IRREVERSIBLE STATES — runs LAST; capture _sliced/_cooked/_cracked, then reload the scene clean. --------
        if do_destructive:
            destroyed = 0
            for o in self.objects():
                if destroyed >= 8:
                    break
                oid = o["objectId"]
                if o.get("sliceable") and not o.get("isSliced"):
                    act = self.slice
                elif o.get("cookable") and not o.get("isCooked"):
                    act = self.cook
                elif o.get("breakable") and not o.get("isBroken"):
                    act = self.break_object
                else:
                    continue
                poses = self.interactable_poses(oid)
                if poses:
                    q = poses[0]
                    self.teleport_full(q["x"], q["y"], q["z"], q["rotation"],
                                       horizon=q.get("horizon", 0.0), standing=q.get("standing", True))
                    self.look_at_object(oid)
                act(oid, force=True)
                self.pass_()
                _site()
                _grab()
                destroyed += 1
                yield f"destructive {o['objectType']} ({destroyed})"
            self.reset(scene)                               # reload the scene to its clean initial state
            yield f"destructive done ({destroyed}); scene reset clean"

        # (6) FINALIZE -------------------------------------------------------------------------------------------------
        gen_dataset.build_data_yaml(dataset_dir, class_to_id)
        hist = ", ".join(f"{c}={counts.get(c, 0)}" for c in sorted(class_to_id))
        cmp = nav.compare(self)
        yield (f"histogram: {hist}")
        # The one number that predicts a useless detector: a class in data.yaml with no frames behind it is a class
        # the training run will happily accept and never learn. Say it out loud instead of burying it in the tally.
        # Counted over TRAIN frames only — a class whose every frame landed in val is just as untrainable as one
        # that was never photographed, and reporting it as covered is the more dangerous of the two lies.
        starved = sorted(c for c in class_to_id if train_counts.get(c, 0) == 0)
        yield (f"coverage: {len(class_to_id) - len(starved)}/{len(class_to_id)} classes have TRAINING frames"
               + (f" — STILL EMPTY: {', '.join(starved[:8])}" + (" …" if len(starved) > 8 else "")
                  if starved else " — every class in the scene was photographed"))
        yield (f"scan complete: {cap_i} frames, {len(class_to_id)} classes -> {dataset_dir} | "
               f"nav free={cmp['discovered_free']} oracle={cmp['oracle_total']} "
               f"agreement={cmp['agreement']:.0%} false_blocked={cmp['false_blocked']} "
               f"undiscovered={cmp['undiscovered']}")

    def explore_navigation(self, cell_size: float = 0.25, memory=None, clear: bool = True):
        """Generator: drive the WHOLE reachable area to build a COMPLETE discovered navigation map.

        Unlike scan_scene (which builds the nav map as a side effect of a sparse dataset scan), this visits EVERY
        reachable position — marking each as `free` where the agent actually stands — then derives the obstacle
        boundary (grid neighbours of free cells that are themselves unreachable -> `blocked`). The result is a full
        occupancy map of the room, persisted into nav_cells via navmap.NavMap and scored against the sim oracle.
        Yields short progress strings so the GUI can stream it. Set clear=False to accumulate onto an existing map.
        """
        import navmap

        mem = self._scan_memory(memory)
        scene = self.scene
        if clear:
            try:
                mem.clear_nav(scene)
            except Exception:
                pass
        nav = navmap.NavMap(mem, scene, cell_size)
        positions = self.reachable_positions() or []
        yield f"explore {scene}: visiting {len(positions)} reachable cells"
        free = set()
        for i, p in enumerate(positions):
            self.teleport(p["x"], p["y"], p["z"], force=True)
            if self.success:
                nav.mark_free(p["x"], p["z"])
                free.add(navmap.world_to_cell(p["x"], p["z"], cell_size))
            else:
                nav.mark_blocked(p["x"], p["z"])
            if i % 15 == 0:
                yield f"explored {i + 1}/{len(positions)}"
        # Obstacle boundary: any grid neighbour of a free cell that is itself not free is a wall/obstacle edge.
        blocked = set()
        for (gx, gz) in free:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (gx + dx, gz + dz)
                    if (dx or dz) and nb not in free:
                        blocked.add(nb)
        for (gx, gz) in blocked:
            wx, wz = navmap.cell_to_world(gx, gz, cell_size)
            nav.mark_blocked(wx, wz)
        cmp = nav.compare(self)
        yield (f"explore complete: {len(free)} free + {len(blocked)} boundary cells | "
               f"oracle agreement {cmp['agreement']:.0%} ({cmp['discovered_free']}/{cmp['oracle_total']})")

    def detect_objects(self, detector, conf: float = 0.25) -> list:
        """Run a trained per-scene `perception.Detector` on the current RGB frame.

        Returns the detector's list of ``{cls, type, state, bbox_xyxy, conf}`` dicts — the *perceived* (vision-only)
        counterpart to reading object metadata.
        """
        return detector.detect(self.frame, conf=conf)

    def whats_on(self, receptacle, detector=None) -> list:
        """Sorted unique object types resting in/on `receptacle` (an objectId or type).

        Perceived mode (a `detector` is given): detect on the current frame, locate the receptacle's own detection box,
        and return the types of every other detection whose box centre falls inside it — no metadata used. Without a
        detector: read AI2-THOR's ground-truth containment (`receptacleObjectIds`).
        """
        if detector is not None:
            dets = self.detect_objects(detector)
            rec = self.resolve(receptacle, prefer_visible=True)
            rtype = rec["objectType"] if rec else str(receptacle).split("|")[0]
            rboxes = [d for d in dets if d["type"] == rtype]
            if not rboxes:
                return []
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
            rx1, ry1, rx2, ry2 = max(rboxes, key=lambda d: area(d["bbox_xyxy"]))["bbox_xyxy"]
            out = set()
            for d in dets:
                if d["type"] == rtype:
                    continue
                x1, y1, x2, y2 = d["bbox_xyxy"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    out.add(d["type"])
            return sorted(out)
        # Metadata path: ground-truth containment.
        rec = self.resolve(receptacle, prefer_visible=False)
        if not rec:
            return []
        ids = set(rec.get("receptacleObjectIds") or [])
        return sorted({o["objectType"] for o in self.objects() if o["objectId"] in ids})

    def depth_backproject(self, box_xyxy) -> Optional[dict]:
        """World ``{x, y, z}`` of the point seen at the centre of `box_xyxy` (pixels), via the depth frame + a pinhole.

        Samples the rendered depth at the box centre and un-projects that pixel through AI2-THOR's pinhole camera into
        the world, accounting for the agent's yaw (`rotation.y`) and camera pitch (`horizon`). Needs the depth sensor on
        (`render_depth=True` / `set_sensors(depth=True)`); returns ``None`` if depth is unavailable or the sample is
        non-finite (e.g. sky / no surface hit).

        Camera model (AI2-THOR renders a vertical FOV with square pixels, so fx == fy):
            f = (H / 2) / tan(vFOV / 2)                  focal length in pixels
            u = px - W/2,  v = py - H/2                  pixel offset from the principal point (image centre)
        With the rendered depth `d` taken as the distance ALONG the optical axis, the camera-frame ray point is
            X_cam = (u / f) * d   (right)   Y_cam = (-v / f) * d   (up, image-y points down)   Z_cam = d   (forward)
        which is then rotated by the camera basis (right Rc, up Uc, forward Fc) built from yaw+pitch and offset by the
        eye position.
        """
        import math
        import numpy as np

        depth = self.depth_frame
        if depth is None:
            return None
        depth = np.asarray(depth, dtype="float32")
        h, w = depth.shape[0], depth.shape[1]
        x1, y1, x2, y2 = box_xyxy
        px = int(round((x1 + x2) / 2.0))
        py = int(round((y1 + y2) / 2.0))
        px = 0 if px < 0 else (w - 1 if px >= w else px)
        py = 0 if py < 0 else (h - 1 if py >= h else py)
        d = float(depth[py, px])
        if not math.isfinite(d) or d <= 0.0:
            return None

        # Pinhole intrinsics from the vertical FOV (square pixels -> one focal length for both axes).
        vfov = math.radians(self.field_of_view or 90.0)
        f = (h / 2.0) / math.tan(vfov / 2.0)
        u = px - w / 2.0
        v = py - h / 2.0
        xc = (u / f) * d            # camera right
        yc = (-v / f) * d           # camera up (pixel-y grows downward, so negate)
        zc = d                      # camera forward (depth along the optical axis)

        # Camera basis in world coords from agent yaw + camera pitch (horizon: +ve looks DOWN). AI2-THOR yaw is measured
        # clockwise from +z, so the level forward heading is (sin yaw, 0, cos yaw).
        pose = self.agent_pose()
        yaw = math.radians(pose["rotation"]["y"])
        pitch = math.radians(pose["horizon"])
        sy, cy = math.sin(yaw), math.cos(yaw)
        sp, cp = math.sin(pitch), math.cos(pitch)
        fwd = (sy * cp, -sp, cy * cp)               # forward, tilted down by pitch
        up = (sy * sp, cp, cy * sp)                 # up, perpendicular in the same vertical plane
        right = (cy, 0.0, -sy)                       # right, unchanged by pitch (pitch is about this axis)

        cam = self.event.metadata.get("cameraPosition")
        if not cam:                                  # fallback eye point if the build doesn't report it (cf. look_at_object)
            cam = dict(pose["position"])
            cam["y"] += 0.675 if pose["standing"] else 0.40
        wx = cam["x"] + xc * right[0] + yc * up[0] + zc * fwd[0]
        wy = cam["y"] + xc * right[1] + yc * up[1] + zc * fwd[1]
        wz = cam["z"] + xc * right[2] + yc * up[2] + zc * fwd[2]
        return {"x": wx, "y": wy, "z": wz}

    # --------------------------------------------------- batch "do-to-all" skills
    def _each(self, object_type: Optional[str], predicate: Callable, action_fn: Callable,
              label: str, limit: int = 12):
        """Generator: walk to every object of `object_type` (or all types) matching `predicate` and run
        `action_fn(objectId)` on it, nearest first. Yields a progress label after each."""
        targets = [o for o in self.objects()
                   if (object_type is None or object_type.lower() in o["objectType"].lower())
                   and _safe(predicate, o)]
        targets = sorted(targets, key=lambda o: o.get("distance", 9e9))[:limit]
        if not targets:
            yield f"nothing to {label}"
            return
        done = 0
        for o in targets:
            oid = o["objectId"]
            if not o.get("isPickedUp"):
                yield from self.walk_to(oid)
            action_fn(oid)
            done += 1 if self.success else 0
            yield f"{label} {o['objectType']} ({done}/{len(targets)})"
        yield f"{label}: {done}/{len(targets)} done"

    def open_all(self, object_type: Optional[str] = None, force: bool = True, limit: int = 12):
        """Open every openable object (optionally only those whose type matches `object_type`)."""
        yield from self._each(object_type, lambda o: o.get("openable") and not o.get("isOpen"),
                              lambda oid: self.open(oid, force=force), "open", limit)

    def close_all(self, object_type: Optional[str] = None, force: bool = True, limit: int = 12):
        """Close every open object (optionally only those whose type matches `object_type`)."""
        yield from self._each(object_type, lambda o: o.get("openable") and o.get("isOpen"),
                              lambda oid: self.close(oid, force=force), "close", limit)

    #: Room lighting. Excluded from an UNFILTERED switch_all -- see below.
    LIGHT_TYPES = ("LightSwitch", "FloorLamp", "DeskLamp")

    def switch_all(self, on: bool = True, object_type: Optional[str] = None, force: bool = True,
                   limit: int = 12, include_lights: bool = False):
        """Turn every toggleable object on (on=True) or off, optionally filtered by `object_type`.

        With NO `object_type` this is the "turn everything off" sweep, and it used to include every
        ``LightSwitch`` — so a planner step ``turn_off_all`` with a null object (which is exactly what the
        prompt tells the model to emit) plunged the kitchen into darkness, and read as the lighting going
        haywire rather than as an action anyone had asked for.

        Room lights are therefore excluded from the unfiltered sweep. They are still reachable deliberately:
        name ``LightSwitch`` as the `object_type`, use the console's Lights ON/OFF batch, or pass
        `include_lights=True`. Switching the lights is a decision, not a side effect of tidying the appliances.
        """
        fn = self.toggle_on if on else self.toggle_off
        skip_lights = (object_type is None) and not include_lights
        lights = tuple(t.lower() for t in self.LIGHT_TYPES)

        def pred(o):
            if skip_lights and str(o.get("objectType", "")).lower() in lights:
                return False
            return o.get("toggleable") and bool(o.get("isToggled")) != on

        yield from self._each(object_type, pred, lambda oid: fn(oid, force=force),
                              "turn on" if on else "turn off", limit)

    def collect_all(self, object_type: str, receptacle: str, force: bool = True, limit: int = 8):
        """Pick up every `object_type` and place each into `receptacle`, one trip at a time."""
        rec = self.resolve(receptacle)
        if not rec:
            yield f"no '{receptacle}'"
            return
        targets = [o["objectId"] for o in self.find(object_type) if o.get("pickupable")][:limit]
        if not targets:
            yield f"no '{object_type}' to collect"
            return
        done = 0
        for oid in targets:
            yield from self.walk_to(oid)
            self.pickup(oid, force=force)
            if not self.success:
                yield f"couldn't pick {oid.split('|')[0]}"
                continue
            yield from self.walk_to(rec["objectId"])
            self.put(rec["objectId"], force=force)
            done += 1 if self.success else 0
            yield f"collected {oid.split('|')[0]} ({done}/{len(targets)})"
        yield f"collected {done}/{len(targets)}"

    # ----------------------------------------------- episode recording (GIF)
    def start_recording(self):
        """Begin accumulating RGB frames for a GIF of the next actions."""
        self._recording = True
        self._rec_frames = []
        self.record_frame()

    def record_frame(self):
        """Append the current RGB frame to the recording (no-op unless recording)."""
        if getattr(self, "_recording", False):
            import numpy as np
            f = self.frame
            if f is not None:
                self._rec_frames.append(np.asarray(f).astype("uint8"))

    def stop_recording(self) -> int:
        """Stop recording; returns the number of captured frames."""
        self._recording = False
        return len(getattr(self, "_rec_frames", []))

    def recording_length(self) -> int:
        return len(getattr(self, "_rec_frames", []))

    def save_gif(self, path: str, fps: int = 10) -> str:
        """Write the recorded frames to an animated GIF at `path`."""
        from PIL import Image
        frames = getattr(self, "_rec_frames", [])
        if not frames:
            raise RuntimeError("no frames recorded — start_recording() first")
        imgs = [Image.fromarray(f) for f in frames]
        duration = max(20, int(1000 / max(1, fps)))
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
        return path

    # ----------------------------------------------- state checkpoint & macros
    def checkpoint(self) -> dict:
        """Snapshot the agent pose + every movable/openable/toggleable object's pose & state — for undo/restore."""
        cp_objs = {}
        for o in self.objects():
            if o.get("pickupable") or o.get("moveable") or o.get("openable") or o.get("toggleable"):
                cp_objs[o["objectId"]] = {
                    "position": dict(o["position"]), "rotation": dict(o["rotation"]),
                    "pickupable": o.get("pickupable"), "moveable": o.get("moveable"),
                    "openable": o.get("openable"), "isOpen": o.get("isOpen"),
                    "toggleable": o.get("toggleable"), "isToggled": o.get("isToggled"),
                    "isPickedUp": o.get("isPickedUp"),
                }
        held = self.held_object()
        self._checkpoint = {"agent": self.agent_pose(), "objects": cp_objs,
                            "held": held["objectId"] if held else None}
        return self._checkpoint

    def restore_steps(self, cp: Optional[dict] = None):
        """Generator form of restore() — yields after the agent and each object, so a GUI can animate it.

        Object positions use PlaceObjectAtPoint (physics-independent); open/toggle states re-applied with forceAction.
        """
        cp = cp or getattr(self, "_checkpoint", None)
        if not cp:
            return
        a = cp["agent"]
        self.teleport(a["position"]["x"], a["position"]["y"], a["position"]["z"],
                      rotation_y=a["rotation"]["y"], horizon=a["horizon"], standing=a["standing"], force=True)
        yield "agent"
        for oid, st in cp["objects"].items():
            if st.get("isPickedUp"):
                continue                                   # don't fling held objects to a stale world pose
            try:
                if st.get("pickupable") or st.get("moveable"):
                    self.place_object_at_point(oid, st["position"], rotation=st["rotation"])
                if st.get("openable") and st.get("isOpen") is not None:
                    (self.open if st["isOpen"] else self.close)(oid, force=True)
                if st.get("toggleable") and st.get("isToggled") is not None:
                    (self.toggle_on if st["isToggled"] else self.toggle_off)(oid, force=True)
            except Exception:
                pass                                       # one stubborn object shouldn't abort the whole restore
            yield oid

    def restore(self, cp: Optional[dict] = None) -> dict:
        """Restore a checkpoint (default: the last): agent pose + object positions + open/toggle states. Blocking."""
        cp = cp or getattr(self, "_checkpoint", None)
        if not cp:
            return {"ok": False, "msg": "no checkpoint taken yet"}
        n = sum(1 for _ in self.restore_steps(cp))
        return {"ok": True, "objects_restored": max(0, n - 1)}     # minus the 'agent' yield

    def start_action_log(self):
        """Begin recording every controller action (for replay / macros)."""
        self._action_log = []
        self._logging_actions = True

    def stop_action_log(self) -> list:
        """Stop recording; return the list of (action, kwargs) captured."""
        self._logging_actions = False
        return list(getattr(self, "_action_log", []))

    def replay(self, log: list):
        """Re-issue a recorded list of (action, kwargs) steps in order."""
        was = getattr(self, "_logging_actions", False)
        self._logging_actions = False                      # don't re-record while replaying
        for action, kwargs in log:
            self.controller.step(action=action, **kwargs)
        self.event = self.controller.last_event
        self._logging_actions = was
        return self.event


# ----------------------------------------------------------------------------------------------------------------------
# Object-interaction registry: which actions a given object supports, derived from its metadata flags.
# `param_ui` tells the GUI to render an extra control: 'openness' (0..1 slider), 'liquid' (dropdown), 'magnitude' (force).
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ObjectActionSpec:
    label: str
    thor_action: str
    enabled: Callable[[dict], bool]
    extra: dict = field(default_factory=dict)   # static kwargs sent with the action
    param_ui: Optional[str] = None              # 'openness' | 'liquid' | 'magnitude' | None


OBJECT_ACTION_SPECS = [
    ObjectActionSpec("Pick Up", "PickupObject", lambda o: o.get("pickupable") and not o.get("isPickedUp")),
    ObjectActionSpec("Put Here (held →)", "PutObject", lambda o: o.get("receptacle")),
    ObjectActionSpec("Open", "OpenObject", lambda o: o.get("openable") and not o.get("isOpen"), param_ui="openness"),
    ObjectActionSpec("Close", "CloseObject", lambda o: o.get("openable") and o.get("isOpen")),
    ObjectActionSpec("Toggle On", "ToggleObjectOn", lambda o: o.get("toggleable") and not o.get("isToggled")),
    ObjectActionSpec("Toggle Off", "ToggleObjectOff", lambda o: o.get("toggleable") and o.get("isToggled")),
    ObjectActionSpec("Slice", "SliceObject", lambda o: o.get("sliceable") and not o.get("isSliced")),
    ObjectActionSpec("Cook", "CookObject", lambda o: o.get("cookable") and not o.get("isCooked")),
    ObjectActionSpec("Break", "BreakObject", lambda o: o.get("breakable") and not o.get("isBroken")),
    ObjectActionSpec("Dirty", "DirtyObject", lambda o: o.get("dirtyable") and not o.get("isDirty")),
    ObjectActionSpec("Clean", "CleanObject", lambda o: o.get("dirtyable") and o.get("isDirty")),
    ObjectActionSpec("Fill", "FillObjectWithLiquid", lambda o: o.get("canFillWithLiquid") and not o.get("isFilledWithLiquid"), param_ui="liquid"),
    ObjectActionSpec("Empty", "EmptyLiquidFromObject", lambda o: o.get("canFillWithLiquid") and o.get("isFilledWithLiquid")),
    ObjectActionSpec("Use Up", "UseUpObject", lambda o: o.get("canBeUsedUp") and not o.get("isUsedUp")),
    ObjectActionSpec("Push", "PushObject", lambda o: o.get("moveable") or o.get("pickupable"), param_ui="magnitude"),
    ObjectActionSpec("Pull", "PullObject", lambda o: o.get("moveable") or o.get("pickupable"), param_ui="magnitude"),
]


def depth_to_gray(depth):
    """Normalize a float depth frame to an 8-bit grayscale array (near = bright) for display/saving."""
    import numpy as np
    arr = np.asarray(depth, dtype="float32")
    finite = arr[np.isfinite(arr)]
    hi = float(finite.max()) if finite.size else 1.0
    hi = hi if hi > 0 else 1.0
    return (255.0 * (1.0 - np.clip(arr, 0.0, hi) / hi)).astype("uint8")


def _safe(predicate, obj) -> bool:
    try:
        return bool(predicate(obj))
    except Exception:
        return False


def applicable_actions(obj: dict) -> list:
    """Return the ObjectActionSpecs whose enabling condition holds for this object."""
    return [s for s in OBJECT_ACTION_SPECS if _safe(s.enabled, obj)]


def object_state_flags(obj: dict) -> str:
    """Compact human-readable state string, e.g. 'open, on, filled(water)'."""
    flags = []
    if obj.get("isPickedUp"): flags.append("HELD")
    if obj.get("openable"): flags.append("open" if obj.get("isOpen") else "closed")
    if obj.get("toggleable"): flags.append("on" if obj.get("isToggled") else "off")
    if obj.get("isSliced"): flags.append("sliced")
    if obj.get("isCooked"): flags.append("cooked")
    if obj.get("isBroken"): flags.append("broken")
    if obj.get("isDirty"): flags.append("dirty")
    if obj.get("isFilledWithLiquid"): flags.append(f"filled({obj.get('fillLiquid')})")
    if obj.get("isUsedUp"): flags.append("usedup")
    return ", ".join(flags)


# Map common AI2-THOR failure-message fragments to a short, actionable hint for the user.
ERROR_HINTS = [
    ("not visible", "the object isn't in view — go to / face it first"),
    ("not interactable", "out of range — walk closer (go to)"),
    ("too far", "out of range — walk closer (go to)"),
    ("beyond", "out of range — walk closer (go to)"),
    ("visibilitydistance", "out of range — walk closer (go to)"),
    ("blocked", "path blocked — reposition and retry"),
    ("collid", "path blocked — reposition and retry"),
    ("already", "it's already in that state"),
    ("not pickupable", "this object can't be picked up"),
    ("can't be opened", "this object can't be opened"),
    ("is not openable", "this object can't be opened"),
    ("is not toggleable", "this object can't be toggled on/off"),
    ("hand", "free your hand first (drop what you're holding)"),
    ("no valid", "no interactable pose nearby — move and retry"),
    ("standing", "try standing / crouching to reach it"),
]


def error_hint(err: str) -> str:
    """Turn a raw AI2-THOR error message into a short actionable suggestion ('' if none matches)."""
    e = (err or "").lower()
    for key, hint in ERROR_HINTS:
        if key in e:
            return hint
    return ""


if __name__ == "__main__":
    # Headless self-test against the cached Unity build.
    agent = ThorAgent(scene="FloorPlan1")
    print("Scene FloorPlan1 |", len(agent.objects()), "objects |", len(agent.visible_objects()), "visible")
    agent.move_ahead(); print("MoveAhead ->", agent.success, agent.error)
    agent.rotate_right(); print("RotateRight ->", agent.success)
    fridge = next((o for o in agent.objects() if o["objectType"] == "Fridge"), None)
    if fridge:
        print("Fridge applicable actions:", [s.label for s in applicable_actions(fridge)])
    print("Pickupable items:", sorted({o["objectType"] for o in agent.pickupable_objects()}))
    print("Capture ->", agent.capture(directory="captures", kinds=["rgb"]))
    agent.stop()
    print("thor_actions self-test OK")


# ===================================================================== PUBLIC API (Phase-4 perception) ==============
# All four hang off ThorAgent; they lazy-import the gen_dataset/navmap/perception siblings (no import cycle, no heavy
# deps at module load). The first uses instance+depth sensors; depth_backproject needs the depth sensor on.
#
# scan_scene(self, dataset_root='datasets', cell_size=0.25, per_class_target=40, max_positions=24, do_states=True,
#            do_destructive=False, memory=None, *, do_containers=True, max_containers=10, container_views=3,
#            container_openness=2, container_poses=2, visibility_distance=6.0) -> generator[str]
#     (Re)scan the current scene into a per-scene YOLO dataset at <dataset_root>/<scene> AND a discovered nav map.
#     Turns instance+depth sensors on and WIDENS the visibility gate for the scan (reloads the scene; the gate is
#     restored afterwards, including on an abandoned generator — `capture_frame` labels only objects AI2-THOR reports
#     `visible`, so at the 1.5 m default everything further away sat in the frame unlabelled). Seeds a deterministic
#     class->id map from perception.scene_class_names, then yields progress strings through: PLAN (rank the containers
#     worth opening + yield an estimated frame count/duration), COVERAGE (scan_plan.spread_positions over
#     reachable_positions -> navmap.mark_free/mark_blocked; capture scan_plan.coverage_views(i) — 8 frames per position
#     over a precessing schedule of 16 yaws x 4 pitches x both stances — via gen_dataset.capture_frame, logging labels
#     to Memory), CONTAINERS (do_containers: peek_inside each ranked openable receptacle, photographing its CONTENTS,
#     then shutting it), BALANCE (top up classes under per_class_target from scan_plan.select_poses viewpoints),
#     reversible STATES (do_states: walk to it -> checkpoint -> flip open/toggle/fill -> capture both -> restore),
#     irreversible STATES (do_destructive, LAST: slice/cook/break a bounded set, capture, then reset the scene clean),
#     FINALIZE (gen_dataset.build_data_yaml + a per-class histogram + a starved-class report + navmap.compare(self)).
#     Does NOT train. `memory` defaults to self._memory or a freshly built memory.Memory.
#     The train/val split is per capture SITE (one coverage position / one container / one balance target), 1 site in
#     5 to val — NOT per frame, which put near-duplicates of every val image into train and made val mAP a
#     memorisation score.
#
# peek_inside(self, object_id, on_capture=None, *, views=3, openness=(1.0, 0.55), poses=2, swing_steps=6,
#             turn=None) -> generator[str]
#     Open an openable receptacle, photograph its INTERIOR from `poses` distinct interactable poses x each openness
#     level x `views` framings (aimed at scan_plan.aim_point — the contents' centroid, not the door — at the pitch and
#     stance scan_plan.aim_from says can actually see into it), calling `on_capture()` per viewpoint, then put the door
#     back exactly as it was found. The restore runs from a `finally`, so abandoning the generator still shuts the door.
#
# contents_of(self, object_id) -> [obj]
#     The objects currently inside/on a receptacle (its receptacleObjectIds, resolved against live metadata).
#
# detect_objects(self, detector, conf=0.25) -> [{'cls','type','state','bbox_xyxy','conf'}, ...]
#     Run a perception.Detector on the current RGB frame (thin pass-through to detector.detect(self.frame, conf)).
#
# whats_on(self, receptacle, detector=None) -> [str]
#     Sorted unique types in/on `receptacle` (objectId or type). With a detector (perceived): the types of detections
#     whose box centre lies inside the receptacle's own detection box. Without: AI2-THOR receptacleObjectIds.
#
# depth_backproject(self, box_xyxy) -> {'x','y','z'} | None
#     Un-project the centre pixel of box_xyxy through the depth frame + a pinhole (vertical FOV self.field_of_view,
#     square pixels) and the agent yaw+horizon to a world point. None if depth is off or the sample is non-finite.
