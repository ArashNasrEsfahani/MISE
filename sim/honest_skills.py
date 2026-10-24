"""Step 4 (s4-act) -- honest skills: act on a SCREEN POINT, retry up to 3 times, search by asking a chooser.

Every skill here decides only from what ``honest.HonestAgent`` gives it -- RGB + depth run through the step-3
detector (boxes + StateEstimator state text), the agent's own pose, the success bool of its last action and its own
action history -- plus the agent's own floor map (``honest_map``) and object memory (``objmem``). The target of an
interaction is always a detection in the current frame; the action is sent at that detection's screen point, and the
point is turned into the simulator's object handle only inside ``HonestAgent.act`` (tag ``action-resolve``), used
once, and dropped. Spec: docs/step4-plan.md (s4-act). Arash decisions 2 (screen point), 4 (search by reasoning over
the kitchen's type list) and 5 (3 retries from a slightly different spot or angle).

One interaction (``_interact``):
  1. locate the target: trusted memory (nearest, preferring the needed state), else a placed belief, else SEARCH;
  2. walk (A* on the agent's own grid) to a cell 0.5-1.0 m from the believed xyz, unless already that close;
  3. face it (yaw, then pitch from the camera height);
  4. observe, and choose the detection of the type nearest the pixel where the belief projects (identity first,
     then the state the action needs, e.g. closed for open). If it already shows the wanted state: satisfied;
  5. act at its point (alternate interior points on retries; for a put, a receptacle-surface point that avoids the
     boxes of things standing on it);
  6. verify: the success bool, then -- where the StateEstimator covers the axis -- re-detect at the acted pixel. A
     confident opposite state after a "successful" action means the point hit something else (``wrong_target``).

Retries: at most ``MAX_RETRIES`` (3) recovery attempts per top-level step, shared by every nested sub-step of a
composite and by the openings a search makes (Astra). Recovery k: 1 = fresh detection + alternate point + pitch
+-15 deg, 2 = sidestep one map cell and re-face, 3 = step back one cell (door clearance) and re-face. A step also
has a primitive cap.

Search (decision 4): ``chooser(target, type_list, memory_summary) -> ordered place types``. The agent sweeps where
it stands, then visits remembered sightings of each ranked place (at most ``MAX_PLACES`` distinct instances, up to
``PER_TYPE`` per place type per pass; a container it opened and found empty is closed again),
opening a closed container it finds there, and otherwise tours observation viewpoints spread over its own map
(farthest-point sampling). Every sighting goes into object memory (and ``places_seen``). Exhausting the budget
means "not found within budget", not "absent". Choosers: ``mock`` (tests), ``prior_chooser`` (the labelled no-LLM
stand-in for s5) and ``LLMChooser(ask)`` (s6 plugs Gemma in as ``ask``).

Search modes (2026-10-07, Arash: "for cases like these it should find everything"). ``search="bounded"`` (default;
the s5 reference-plan runner and every published number) is the policy above. ``search="exhaustive"`` (the console):
a remembered sighting that is not re-seen after ``LOCAL_LOOKS`` free local looks is SET ASIDE for this episode (its
evidence stays in memory; objmem's own miss rule decides contradiction) and the next belief, else the full search,
is tried -- at most ``MAX_RELOCATE`` times per step, never charged to the 3 action retries. The exhaustive search
visits EVERY remembered place instance (chooser-ranked types first, then every other receptacle / openable class),
opens each closed or unsure container it reaches and closes it again, then walks viewpoints covering the whole own
map (every free cell within ``VP_COVER`` m of one) with a sweep at each, revisiting newly seen places, under its own
budget ``SEARCH_PRIM_CAP``. "Not found" then means "not seen in the reachable known map", never "absent".
Containers are inspected from the CURRENT frame (``_candidates``) and each one squarely (``_inspect``): an own-map
cell in front of the front plane fitted to its own depth, crouched for low doors, opened once (``_match``: geometry +
image overlap + own history, never a simulator handle), then an honest inside-visible check (``inside_check``) with
ordered re-tries (crouch / step back / side-step) before "unsure, kept as a candidate". (2026-10-07, Astra-reviewed;
docs/console-v2-progress.md "fix: container search quality".)

Tidiness (``tidy=True``, the console): ``opened_by_me`` lists every container this agent opened in the episode (type,
aim point, why). A container a search looked into is closed again (one retry from another angle); one the target was
found in is closed right after the pickup (or at step end); one opened only for a put is closed at step end -- each
unless a later plan step (``exec_step(upcoming=...)``) still needs it. ``tidy_up()`` closes whatever is left except
explicit ``open`` steps. Tidy closes never spend the retry budget and never fail a step. ``tidy=False`` (default) keeps
the s5 behaviour.

Pose first, then open (2026-10-08, the console policy ``pose_first`` = exhaustive; Astra-reviewed; docs/console-v2-
progress.md "fix: pose first, then open"). A put into an openable (``_put_openable``), an explicit ``open`` step and a
search inspection choose WHERE TO STAND before the door moves: a cell squarely in front of the opening, outside the
floor its door will sweep (``door_model`` / ``swept``: hinge unknown -> both edges; once seen open -> the hinge side
from its own door points), within open + put reach (``put_cells``); walk there with the door closed, re-check the front
from there, open, put without walking (pixel / pitch / crouch), close from the same pose. While a door this agent
opened stands open, its swept floor cells are blocked in the own map (``GridMap.block_cells``), unblocked on close.
What moved when it opened (``opened_front``: deeper = the closed door before; nearer = the leaf now; ``leaf_front``)
corrects a front estimate that took another face (a fridge's side) for the door.

Systematic search route (2026-10-08, console / exhaustive only; Astra-reviewed; docs/console-v2-progress.md "fix:
systematic search route"). No second look-around where it already looked around (own history); an explicit plan before
moving: the agent's own FRONT REGISTRY (openable fronts fitted to its own depth in every look) + remembered places not
yet looked at closely, weighted by the chooser's rank x a declared type-level height prior, grouped into STOPS (greedy
set cover over the cells that serve each front squarely -- a drawer column and its neighbouring doors = one stop),
ordered by a weighted search-cost surrogate (nearest-next / weight seeds, 2-opt, relocation), each stop moved to the
serving cell that shortens the route. Pass 1 = the likely places, pass 2 = everything else. At a stop the CURRENT frames
supply the containers; every one the stop (or one cell beside it) serves is inspected before leaving. New fronts join a
planned stop or get a stop inserted where it costs least (the route keeps its order). Then coverage viewpoints only
where own depth has not yet seen (partial looks). Walks keep one heading and side- / back-step on short legs.

Failure causes: not_detected, wrong_target, unreachable, action_refused, state_unrecognised, retry_exhausted.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Callable, Optional

import honest_map as hm
import objmem
import type_scope

CAUSES = ("not_detected", "wrong_target", "unreachable", "action_refused", "state_unrecognised", "retry_exhausted",
          "no_cutting_tool")          # A26 item 2 (K-B2b 2026-10-13): slice refused without a ButterKnife in hand
MAX_RETRIES = 3
STEP_PRIM_CAP = 700            # primitives one top-level step may send (walks + looks + acts)
RMIN, RMAX = 0.5, 1.0          # stand-off annulus around a target (m, horizontal)
NEAR_OK = (0.35, 1.15)         # already this close: do not walk, just face
SWEEP_N = 8                    # look-around = 8 views, 45 deg apart
SWEEP_HORIZON = 15.0           # camera pitch for a look-around (down; sees counters near and far)
PITCH_MIN, PITCH_MAX = -30.0, 60.0
PITCH_ALT = 15.0
YAW_ALT = 30.0                 # put recovery: turn this far off the receptacle anchor (doors, rims)
MAX_PLACES = 12                # distinct place instances a search visits (s5: was 4 -- policy revision, see ledger)
PER_TYPE = 6                   # instances of one ranked place type visited per pass
SEARCH_RESERVE = 120           # primitives of the step cap a search leaves for acting on what it finds
SAME_PLACE = 0.3               # m (3D): remembered tracks this close count as one place instance
N_VIEWPOINTS = 8
SEARCH_MODES = ("bounded", "exhaustive")
SEARCH_PRIM_CAP = 2500         # exhaustive search: extra logical primitives one step may spend searching
LOCAL_LOOKS = 2                # free local looks (pitch, sidestep) before a remembered sighting is set aside
MAX_RELOCATE = 4               # beliefs set aside per step before the step gives up re-locating
VP_COVER = 1.0                 # m: exhaustive viewpoints cover every free cell of the own map within this distance
MAX_VIEWPOINTS = 40
MAX_EXH_PLACES = 80            # place instances one exhaustive search may visit (finite even if tracks jitter)
REACH = 1.5                    # m: a receptacle point beyond this cannot be acted on
OPENABLE = frozenset({"Fridge", "Cabinet", "Drawer", "Microwave", "Box", "Safe", "Kettle"})
#: mirrors perception.STATE_WORDS (axis -> (negative word, positive word)); the selftest checks they agree.
STATE_WORDS = {"open": ("closed", "open"), "on": ("off", "on"), "sliced": ("whole", "sliced"),
               "cooked": ("raw", "cooked"), "broken": ("intact", "broken"), "dirty": ("clean", "dirty"),
               "filled": ("empty", "filled")}
#: plan verb -> (interaction primitive, postcondition (axis, value) or None)
VERBS = {
    "pickup": ("PickupObject", None),
    "open": ("OpenObject", ("open", True)),
    "close": ("CloseObject", ("open", False)),
    "toggle_on": ("ToggleObjectOn", ("on", True)),
    "toggle_off": ("ToggleObjectOff", ("on", False)),
    "slice": ("SliceObject", ("sliced", True)),
    "break": ("BreakObject", ("broken", True)),
    "dirty": ("DirtyObject", ("dirty", True)),
    "clean": ("CleanObject", ("dirty", False)),
    "fill": ("FillObjectWithLiquid", ("filled", True)),
    "empty": ("EmptyLiquidFromObject", ("filled", False)),
}
NAV = {"move_ahead": "MoveAhead", "move_back": "MoveBack", "move_left": "MoveLeft", "move_right": "MoveRight",
       "rotate_left": "RotateLeft", "rotate_right": "RotateRight", "look_up": "LookUp", "look_down": "LookDown",
       "crouch": "Crouch", "stand": "Stand"}
ALIASES = {"put": "put_in", "fill_with_liquid": "fill", "empty_liquid": "empty", "break_object": "break",
           "goto": "go_to", "find": "search_for", "cook_food": "cook", "heat_object": "heat", "wash_object": "wash"}
LIQUIDS = ("water", "coffee", "wine")
#: the honest action set offered to a planner (s6 builds its tool list from this)
#: verbs that return a thing to rest; with several instances, a visible not-at-rest one is the target
REST_VERBS = frozenset({"close", "toggle_off"})
# P-round5 E1 (2026-10-13): openable appliances that switch on ONLY with the door closed (simulator contract: ToggleObjectOn of an
# open Microwave is refused). A later toggle_on of such a type does not "need it open", so the end-of-step tidy closes a door this
# agent opened implicitly (put route / composite) before that step. Explicit opens are untouched. False = the round-4 behaviour.
SWITCH_ON_CLOSED = frozenset({"Microwave"})
TIDY_SWITCH_CLOSED = True
TOOLS = sorted(set(VERBS) | {"pick_and_place", "put_in", "heat", "cook", "wash", "go_to", "search_for", "drop",
                             "close_all", "crouch", "stand"})


class _Soft(Exception):
    """A failed attempt that a recovery move might fix."""

    def __init__(self, cause, why="", free=False):
        super().__init__(f"{cause}: {why}")
        self.cause, self.why = cause, why
        self.free = free        # B3 r2: an attempt that cost nothing and was bounded elsewhere (see _put_openable)


class _Fail(Exception):
    """A failure that ends the step (no retry can fix it, or the budget is spent)."""

    def __init__(self, cause, why="", under=None):
        super().__init__(f"{cause}: {why}")
        self.cause, self.why = cause, why
        self.under = under      # B1: structured wrapped cause of a retry_exhausted (never parsed from ``why``)


# ----------------------------------------------------------------------------------------------- pure helpers
def state_of(det: dict, axis: str) -> Optional[bool]:
    """The StateEstimator's verdict on ``axis`` for a detection: True / False, or None (not covered / unsure)."""
    neg, pos = STATE_WORDS[axis]
    v = (det.get("states") or {}).get(axis)
    if v is not None:
        s = str(v).strip().lower()
        if s in ("1", "true", pos):
            return True
        if s in ("0", "false", neg):
            return False
    words = [w.strip().lower() for w in str(det.get("state") or "").split(",")]
    if pos in words:
        return True
    if neg in words:
        return False
    return None


def _wh(depth):
    return (int(depth.shape[1]), int(depth.shape[0]))


def _depth(depth, pt):
    try:
        v = float(depth[int(pt[1])][int(pt[0])])
    except (IndexError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


#: SPEC-INHAND fix A (2026-10-13; Astra SPEC-INHAND Q2): the in-hand verdicts decide "at the hand" with the shared, supported
#: near-depth predicate objmem.hand_depth (q-th percentile of the box's own depth, a connected near patch of a minimum size
#: that fits the box), not with the depth at the box's ground point (the background behind a thin held object).
hand_depth = objmem.hand_depth


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inside(pt, b, pad=0):
    return b[0] - pad <= pt[0] <= b[2] + pad and b[1] - pad <= pt[1] <= b[3] + pad


def candidate_points(depth, bbox, n: int = 5, agree: bool = True, avoid=(), max_depth: Optional[float] = None):
    """Interior pixels of ``bbox`` on an n x n grid (inner 80 %), nearest the box centre first.

    ``agree`` keeps only pixels whose depth agrees with the box's median depth (an object point, not background);
    ``avoid`` drops pixels inside other boxes (things standing on a receptacle); ``max_depth`` drops points out of
    reach. Pure, from depth and boxes only.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pts = []
    for i in range(n):
        for j in range(n):
            px = int(round(x1 + (x2 - x1) * (0.1 + 0.8 * (i + 0.5) / n)))
            py = int(round(y1 + (y2 - y1) * (0.1 + 0.8 * (j + 0.5) / n)))
            pts.append((px, py))
    vals = [(p, _depth(depth, p)) for p in pts]
    vals = [(p, d) for p, d in vals if d is not None]
    if agree and vals:
        ds = sorted(d for _, d in vals)
        med = ds[len(ds) // 2]
        tol = max(0.10, 0.10 * med)
        vals = [(p, d) for p, d in vals if abs(d - med) <= tol]
    if max_depth is not None:
        vals = [(p, d) for p, d in vals if d <= max_depth]
    vals = [(p, d) for p, d in vals if not any(_inside(p, b) for b in avoid)]
    vals.sort(key=lambda v: (v[0][0] - cx) ** 2 + (v[0][1] - cy) ** 2)
    out = []
    for p, _ in vals:
        if p not in out:
            out.append(p)
    return out


def farthest_points(cells, k: int, start) -> list:
    """Farthest-point sampling of ``k`` cells over ``cells`` from ``start`` (deterministic)."""
    cells = sorted(set(cells))
    if not cells:
        return []
    start = min(cells, key=lambda c: (abs(c[0] - start[0]) + abs(c[1] - start[1]), c))
    chosen = [start]
    dmin = {c: math.hypot(c[0] - start[0], c[1] - start[1]) for c in cells}
    while len(chosen) < min(k, len(cells)):
        c = max(cells, key=lambda c: (dmin[c], c))
        if dmin[c] <= 0:
            break
        chosen.append(c)
        for o in cells:
            dmin[o] = min(dmin[o], math.hypot(o[0] - c[0], o[1] - c[1]))
    return chosen


def cover_points(cells, radius: float, start, kmax: int = MAX_VIEWPOINTS) -> list:
    """Farthest-point viewpoints until every cell is within ``radius`` (cells) of one, then ordered as a greedy
    nearest-next walk from ``start``. Pure and deterministic."""
    cells = sorted(set(cells))
    if not cells:
        return []
    s0 = min(cells, key=lambda c: (abs(c[0] - start[0]) + abs(c[1] - start[1]), c))
    chosen = [s0]
    dmin = {c: math.hypot(c[0] - s0[0], c[1] - s0[1]) for c in cells}
    while len(chosen) < kmax:
        c = max(cells, key=lambda c: (dmin[c], c))
        if dmin[c] <= radius:
            break
        chosen.append(c)
        for o in cells:
            dmin[o] = min(dmin[o], math.hypot(o[0] - c[0], o[1] - c[1]))
    tour, cur, left = [], s0, chosen[1:]
    while left:
        nxt = min(left, key=lambda c: (math.hypot(c[0] - cur[0], c[1] - cur[1]), c))
        tour.append(nxt)
        left.remove(nxt)
        cur = nxt
    return [s0] + tour


# ----------------------------------------------------------------------------------------------- container inspection
#: (2026-10-07, "fix: container search quality"; Astra-reviewed). Pure geometry from the agent's own depth + pose.
#: Thresholds were set a priori on FloorPlan1 geometry (development data), not tuned on a held-out set.
INSPECT_D = {"low": 0.75, "mid": 0.85, "high": 0.95, "Fridge": 1.05}   # stand-off (m, horizontal) by height band
INSPECT_SPAN = (-0.25, 0.30)   # m around that stand-off (the distance term of the score keeps it near the stand-off)
CROUCH_DROP = 0.675            # m: camera drop when crouched (own-body calibration, FloorPlan1 2026-10-07: camera
                               #    1.576 m standing, 0.901 m crouched; was a 0.30 guess) -- for the reach check
REACH3D = 1.10                 # m: camera -> front centroid at the inspection pose (was 1.4; the open gate is 1.15)
ENUM_RANGE = 2.5               # m: containers in view this close are queued for inspection
CAVITY = {"Cabinet": 0.15, "Fridge": 0.20, "Microwave": 0.12, "Box": 0.10, "Safe": 0.12, "Kettle": 0.05}
VIS_MIN_SUPPORT = 6            # probe points in frame with valid depth needed to judge at all
VIS_OK, VIS_PART = 0.6, 0.25   # see-in fraction: visible / partly
MAX_INSPECT = 40               # container inspections one search may make
MAX_REPOSITION = 3             # ordered re-tries after a "not visible" verdict: crouch / step back / side-step
CROUCH_Y = 0.78                # m: a front whose centre is lower than this (under-counter cabinet, low drawer):
                               #    inspect CROUCHED, pitched down (Arash, 2026-10-07)
HIGH_Y = 1.30                  # m: above this (wall cabinet): stand and pitch up; between: stand, pitch down
FRONT_OK = 25.0                # deg: heading within this of the front normal = "square" (reported per inspection)
SIGHT_NOW = True               # search_for succeeds on a sighting in the CURRENT frame within reach (no re-face)
ID_SAME_D, ID_SAME_DY, ID_OVERLAP = 0.25, 0.15, 0.6     # confident "same container": near AND image overlap
ID_NEAR_D, ID_NEAR_DY = 0.35, 0.20                      # nearer than this but no overlap evidence: ambiguous
ID_TIGHT_D, ID_TIGHT_DY = 0.08, 0.06                    # this close counts as the same even without overlap
DRAWER_SAME_DY, DRAWER_NEAR_DY = 0.05, 0.09             # stacked drawers are only 0.12-0.17 m apart (FloorPlan1)
BIG_FRONTS = {"Fridge": 1.0, "Microwave": 0.6}         # m: declared front-width priors of single-door appliances


#: reach gate (2026-10-07, "fix: no far-away opens"; Astra-reviewed). Checked on the CURRENT frame, at the exact pixel
#: the gateway will resolve, before any interaction is sent. A conservative surface-reach constraint from the agent's
#: own depth (Euclidean camera -> back-projected point), NOT a reproduction of AI2-THOR's object-distance rule.
OPEN_REACH = 1.15              # m: open (a farther front is never opened, even though THOR would accept 1.5)
CLOSE_REACH = 1.50             # m: close -- the point must be on / inside the door THIS agent opened (door gate); an
                               #    open cavity reads deeper than its front, so only THOR's own limit applies here
ACT_REACH = 1.40               # m: every other interaction (pickup, put, toggle, slice, ...); provisional
EDGE_DZ = 0.10                 # m: depth spread in the 7 x 7 px window around a container pixel -> a depth edge
PLANE_TOL = 0.15               # m: the acted point must lie this close to the intended front's plane
FRONT_PAD, DRAWER_PAD = 0.06, 0.03   # m: lateral / vertical margin around the intended front's measured extent
DOOR_SWEEP = 0.75              # m: a swung door stays within this of its closed front (close gate)
HAND_DEPTH = 0.50              # m: while holding, a pixel nearer than this AND inside a detection of the held type is
                               #    the carried object (body geometry: carried ~0.3-0.5 m in front of the camera); a
                               #    door leaf that swung close is still a valid close pixel (after_v7)
GATE_POINTS = 24               # candidate pixels one gated action tries before giving up (pure checks, no sends)


def front_extent(front) -> tuple:
    """(lateral lo, hi, ymin, ymax) of a front patch, lateral measured along the front plane."""
    nx, nz = front["n"]
    lats = [p[0] * -nz + p[2] * nx for p in front["pts"]] or [front["c"][0] * -nz + front["c"][2] * nx]
    return min(lats), max(lats), front["ymin"], front["ymax"]


def front_width(front) -> float:
    """Measured width (m) of a front patch (its points sample the inner ~80 % of the box)."""
    lo, hi, _, _ = front_extent(front)
    return (hi - lo) / 0.8


def reach_check(depth, pose, fov, pt, limit: float, front=None, mode: str = "front", ctype: str = "",
                edges: bool = True) -> dict:
    """Pure gate for ONE pixel: own depth at exactly ``pt`` -> 3D point; ok when it is within ``limit`` m of the
    camera, not on a depth edge (containers), and -- with an intended ``front`` -- on that front (``mode='front'``:
    within PLANE_TOL of its plane and inside its measured extent + margin) or, for a close (``mode='door'``), within
    DOOR_SWEEP of its closed front and at its height. Returns {ok, dist, xyz, why}."""
    d = _depth(depth, pt)
    if d is None:
        return {"ok": False, "dist": None, "xyz": None, "why": "no depth at the point"}
    wh = _wh(depth)
    xyz = objmem.backproject(pt[0], pt[1], d, pose, wh, fov)
    eye = objmem._basis(pose)[0]
    dist = math.dist(eye, xyz)
    out = {"ok": False, "dist": round(dist, 3), "xyz": xyz, "why": ""}
    if dist > limit:
        out["why"] = f"out of reach ({dist:.2f} m > {limit:.2f} m)"
        return out
    if edges:
        win = [_depth(depth, (pt[0] + dx, pt[1] + dy)) for dx in (-3, 0, 3) for dy in (-3, 0, 3)]
        win = [v for v in win if v is not None]
        if len(win) < 5 or max(win) - min(win) > max(EDGE_DZ, 0.08 * d):
            out["why"] = "on a depth edge"
            return out
    if front is not None:
        c, (nx, nz) = front["c"], front["n"]
        lo, hi, ymin, ymax = front_extent(front)
        vpad = DRAWER_PAD if ctype == "Drawer" else FRONT_PAD
        if ctype == "Drawer" and mode != "door":
            # stacked drawers are 0.12-0.17 m apart and their boxes overlap: only the measured front point's own
            # band counts as this drawer (after_v1: the drawer below opened through an overlapping box)
            ymin, ymax = max(ymin, c[1] - 0.05), min(ymax, c[1] + 0.05)
        if mode == "door":
            # the swung door / pulled-out drawer lies IN FRONT of the closed front's plane (agent side); the
            # aperture shows the interior (behind it) and its contents. A door pixel is preferred; an interior
            # pixel is a fallback; a point in front of the plane but below the front (the floor) is neither
            s_ = (xyz[0] - c[0]) * nx + (xyz[2] - c[2]) * nz
            out["s"] = round(s_, 3)
            if math.hypot(xyz[0] - c[0], xyz[2] - c[2]) > DOOR_SWEEP:
                out["why"] = "not on the door this agent opened"
                return out
            if s_ >= 0.04 and ymin - 0.05 <= xyz[1] <= ymax + 0.10:
                out["kind"] = "door"
            elif -0.60 <= s_ < 0.04 and ymin - 0.15 <= xyz[1] <= ymax + 0.15:   # cavity up to 0.6 m deep
                out["kind"] = "interior"
            else:
                out["why"] = "not on the door this agent opened"
                return out
        else:
            plane = abs((xyz[0] - c[0]) * nx + (xyz[2] - c[2]) * nz)
            lat = xyz[0] * -nz + xyz[2] * nx
            if plane > PLANE_TOL:
                out["why"] = f"not on the intended front ({plane:.2f} m off its plane)"
                return out
            if not (lo - FRONT_PAD <= lat <= hi + FRONT_PAD) or not (ymin - vpad <= xyz[1] <= ymax + vpad):
                out["why"] = "outside the intended front"
                return out
    out["ok"] = True
    return out


def _depth_tol(z):
    return min(0.08, 0.04 + 0.02 * float(z))


def front_patch(depth, bbox, pose, fov, n: int = 7) -> Optional[dict]:
    """The closed front of a container from the agent's own depth: 3D points on an n x n grid of depth-agreeing
    pixels inside ``bbox`` (inner 80 %), their centroid, and a horizontal front normal from a least-squares plane
    (oriented towards the camera). ``planar`` False -> the normal is the camera direction (approach, not a surface
    normal; reported as such). None when fewer than 4 points have depth."""
    if depth is None or not bbox:
        return None
    wh = _wh(depth)
    pts = []
    for p in candidate_points(depth, bbox, n=n, agree=True):
        d = _depth(depth, p)
        xyz = objmem.backproject(p[0], p[1], d, pose, wh, fov) if d else None
        if xyz:
            pts.append(tuple(xyz))
    if len(pts) < 4:
        return None
    c = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
    eye = objmem._basis(pose)[0]
    to_cam = (eye[0] - c[0], eye[2] - c[2])
    tl = math.hypot(*to_cam) or 1.0
    approach = (to_cam[0] / tl, to_cam[1] / tl)
    normal, planar, rms = approach, False, None
    if len(pts) >= VIS_MIN_SUPPORT:
        import numpy as np
        a = np.asarray(pts, dtype="float64") - np.asarray(c)
        _, s, vt = np.linalg.svd(a, full_matrices=False)
        nv = vt[-1]
        rms = float(s[-1] / math.sqrt(len(pts)))
        hx, hz = float(nv[0]), float(nv[2])
        hn = math.hypot(hx, hz)
        fit = None
        if hn >= 0.8:
            hx, hz = hx / hn, hz / hn
            if hx * approach[0] + hz * approach[1] < 0:
                hx, hz = -hx, -hz
            fit = (hx, hz)
        if rms <= 0.03 and fit is not None:
            normal, planar = fit, True
    else:
        fit = None
    ys = [p[1] for p in pts]
    return {"pts": pts, "c": c, "n": normal, "planar": planar, "rms": rms, "ymin": min(ys), "ymax": max(ys),
            "bbox": tuple(bbox), "n_fit": fit}


def coincide(pts, ref, tol: float = 0.08) -> float:
    """Share of ``pts`` lying within ``tol`` m of some point of ``ref`` (the same physical surface, from depth)."""
    if not pts or not ref:
        return 0.0
    return sum(1 for p in pts if min(math.dist(p, q) for q in ref) <= tol) / len(pts)


def front_patch_like(rec, c, pts) -> dict:
    """A front patch rebuilt from an inspection record's stored view (for a revisit)."""
    ys = [p[1] for p in pts] or [c[1]]
    n = rec.get("normal") or (0.0, 1.0)
    return {"pts": list(pts), "c": tuple(c), "n": tuple(n), "planar": rec.get("planar"), "rms": None,
            "ymin": min(ys), "ymax": max(ys), "bbox": None}


def height_band(patch, cam_y: float, ctype: str) -> str:
    if ctype == "Fridge":
        return "Fridge"
    dy = cam_y - patch["c"][1]
    return "low" if dy > 0.6 else ("high" if dy < -0.15 else "mid")


def inspection_cells(gm, patch, cam_h: float, ctype: str, exclude=(), model=None) -> list:
    """Free cells of the own map from which the closed front is seen squarely: on the front side of the plane,
    near the height-dependent stand-off, within reach -- and, with a door ``model``, outside the floor its door will
    sweep (2026-10-08). Sorted best first as ``(score, cell)``."""
    c, (nx, nz) = patch["c"], patch["n"]
    band = height_band(patch, cam_h, ctype)
    dstar = INSPECT_D[band]
    tb = getattr(gm, "_tb", ())
    out = []
    for cell in gm.free:
        if cell in exclude or cell in tb:
            continue
        wx, wz = gm.world(cell)
        vx, vz = wx - c[0], wz - c[2]
        r = math.hypot(vx, vz)
        if r < 1e-6 or not (dstar + INSPECT_SPAN[0] <= r <= dstar + INSPECT_SPAN[1]):
            continue
        along = (vx * nx + vz * nz) / r
        if along <= 0.35:                                  # behind or almost edge-on to the front
            continue
        eye = cam_h - (CROUCH_DROP if c[1] < CROUCH_Y else 0.0)       # low fronts (drawers too) are OPENED crouched
        if math.hypot(r, eye - c[1]) > REACH3D:
            continue
        if model is not None and swept(model, wx, wz):
            continue
        off = math.degrees(math.acos(max(-1.0, min(1.0, along))))
        out.append((off / 10.0 + abs(r - dstar) / 0.25, cell))
    out.sort()
    return out


def interior_probes(patch, ctype: str, extent: Optional[float] = None) -> list:
    """3D points that lie INSIDE the container once it is open, from the closed front patch: a cabinet / fridge /
    microwave interior is behind the front plane; a drawer's is in front of it (pulled out by ``extent``, measured
    after opening) and a little below its top. Only the inner part of the patch is used (Astra: keep probes away
    from uncertain boundaries)."""
    c, (nx, nz) = patch["c"], patch["n"]
    pts = patch["pts"]
    xs = sorted(p[0] * -nz + p[2] * nx for p in pts)           # lateral coordinate along the front
    lo, hi = xs[int(0.15 * (len(xs) - 1))], xs[int(0.85 * (len(xs) - 1))]
    out = []
    if ctype == "Drawer":
        e = max(0.10, min(0.45, extent if extent else 0.25))
        top = patch["ymax"]
        for p in pts:
            lat = p[0] * -nz + p[2] * nx
            if not (lo <= lat <= hi):
                continue
            for f in (0.35, 0.65):
                out.append((p[0] + nx * e * f, top - 0.06, p[2] + nz * e * f))
        return out
    dep = CAVITY.get(ctype, 0.12)
    ylo = patch["ymin"] + 0.15 * (patch["ymax"] - patch["ymin"])
    yhi = patch["ymax"] - 0.15 * (patch["ymax"] - patch["ymin"])
    for p in pts:
        lat = p[0] * -nz + p[2] * nx
        if lo <= lat <= hi and ylo <= p[1] <= yhi:
            out.append((p[0] - nx * dep, p[1], p[2] - nz * dep))
    return out


def inside_check(depth, pose, fov, probes, patch, door_boxes=(), face_off: float = 0.0) -> dict:
    """Honest "can I see inside?" from the agent's own depth + pose. A probe is SEE-IN when it projects into the
    frame and the observed depth there is not nearer than the probe (the ray reaches the interior). Verdict:
    visible (>= VIS_OK of the supported probes, spread over >= 2 lateral thirds), partly (>= VIS_PART), else
    not-visible, with an observable reason; too few supported probes -> unsure ("not in view")."""
    wh = _wh(depth)
    w, h = wh
    sup = see = near = flat = out_frame = 0
    thirds = set()
    c, (nx, nz) = patch["c"], patch["n"]
    lat0 = c[0] * -nz + c[2] * nx
    lats = [p[0] * -nz + p[2] * nx - lat0 for p in probes] or [0.0]
    span = (max(lats) - min(lats)) or 1.0
    door_hit = 0
    for p, lat in zip(probes, lats):
        pr = objmem.project(p, pose, wh, fov)
        if pr is None or not (2 <= pr[0] < w - 2 and 2 <= pr[1] < h - 2):
            out_frame += 1
            continue
        d = _depth(depth, (pr[0], pr[1]))
        if d is None:
            out_frame += 1
            continue
        sup += 1
        z = pr[2]
        if d >= z - _depth_tol(z):
            see += 1
            thirds.add(min(2, int(3 * (lat - min(lats)) / span)))
        else:
            fp = objmem.project((p[0] + nx * face_off, p[1], p[2] + nz * face_off), pose, wh, fov)
            if d < (fp[2] if fp else z) - 0.08:            # nearer than the closed front was: something in front
                near += 1
                if any(_inside((pr[0], pr[1]), b) for b in door_boxes):
                    door_hit += 1
            else:
                flat += 1
    res = {"support": sup, "see": see, "probes": len(probes), "frac": round(see / sup, 2) if sup else 0.0,
           "thirds": len(thirds)}
    if sup < VIS_MIN_SUPPORT:
        res.update(verdict="unsure", reason="interior not in view" if out_frame else "too few depth points")
    elif res["frac"] >= VIS_OK and len(thirds) >= 2:
        res.update(verdict="visible", reason="")
    elif res["frac"] >= VIS_PART:
        res.update(verdict="partly", reason="part of the inside hidden")
    else:
        if near >= flat:
            res.update(verdict="not-visible", reason="door in the way" if door_hit * 2 >= max(1, near)
                       else "a nearer surface blocks the view")
        else:
            res.update(verdict="not-visible", reason="still shut / no cavity seen")
    return res


# ----------------------------------------------------------------------------------------------- pose first, then open
#: (2026-10-08, "fix: pose first, then open"; Astra-reviewed). A put into an openable receptacle (and every console open)
#: chooses where to stand BEFORE the door moves: a cell squarely in front of the opening, outside the floor the door
#: will sweep, within open + put reach; walk there with the door closed, open, put without walking, close from there.
#: While a door this agent opened stands open, its swept floor is temporarily blocked in the own map. Door geometry is
#: the agent's own: the closed front (own depth) and, once it saw the door open, the hinge side from the door's own
#: depth points. Radii / margins below are DECLARED PRIORS (FloorPlan1 development data), not measured facts.
AGENT_R = 0.20                 # m: body clearance around the swept floor (cell centres are tested)
DOOR_FRAC = 0.85               # unknown door: leaf radius = this x the measured front width (the body frame is wider)
DOOR_MAX = {"Fridge": 1.0, "Cabinet": 0.6, "Microwave": 0.6, "Box": 0.4, "Safe": 0.5, "Kettle": 0.2}
DRAWER_OUT = 0.45              # m: how far a drawer may come out (unknown)
PATCH_SPAN = 0.686             # the front patch samples the inner 80 % of the box on a 7-grid: its extent / box width
PUT_OPEN3D = 1.10              # m: eye -> nearest front height at the put pose (the open gate is 1.15 at the pixel)
PUT_REACH3D = 1.35             # m: eye -> interior point at the put pose (the put gate is 1.40 at the pixel)
PUT_SQUARE = 0.906             # cos 25 deg: a put pose looks into the opening within FRONT_OK of its normal


def _lat(front, x, z):
    nx, nz = front["n"]
    return x * -nz + z * nx


def split_faces(patch, eye) -> list:
    """A non-planar front patch seen at a corner (e.g. a fridge's door AND its side): two vertical faces from its own
    3D points -- the points ordered by azimuth around the eye, split where two straight lines (top view) fit best.
    Returns [] unless both parts are straight (rms <= 0.03 m), >= 6 points each, and >= 30 deg apart. Pure."""
    import numpy as np
    pts = list(patch.get("pts") or ())
    if len(pts) < 12:
        return []
    pts.sort(key=lambda p: math.atan2(p[0] - eye[0], p[2] - eye[2]))
    best = None
    for k in range(6, len(pts) - 5):
        fits = []
        for part in (pts[:k], pts[k:]):
            xz = np.asarray([(p[0], p[2]) for p in part], dtype="float64")
            m = xz.mean(axis=0)
            _, sv, vt = np.linalg.svd(xz - m, full_matrices=False)
            fits.append((float(sv[-1] / math.sqrt(len(part))), vt[-1], m, part))
        tot = fits[0][0] + fits[1][0]
        if best is None or tot < best[0]:
            best = (tot, fits)
    if best is None:
        return []
    out = []
    for rms, nv, m, part in best[1]:
        if rms > 0.03:
            return []
        hx, hz = float(nv[0]), float(nv[1])
        if hx * (eye[0] - m[0]) + hz * (eye[2] - m[1]) < 0:
            hx, hz = -hx, -hz
        ys = [p[1] for p in part]
        out.append({"pts": part, "c": (float(m[0]), sum(ys) / len(ys), float(m[1])), "n": (hx, hz), "planar": True,
                    "rms": rms, "ymin": min(ys), "ymax": max(ys), "bbox": patch.get("bbox"), "from": "a corner split"})
    a, b = out
    if math.degrees(math.acos(max(-1.0, min(1.0, a["n"][0] * b["n"][0] + a["n"][1] * b["n"][1])))) < 30:
        return []
    return out


def door_model(front, ctype: str, door_pts=None, prev: Optional[dict] = None) -> dict:
    """The floor a container's door / drawer sweeps, from the agent's own closed-front estimate (and, once it saw it
    open, the door's own 3D points). Swing doors: hinge at one front edge (both edges while unknown), leaf radius r;
    drawers: a slab pulled out along the normal. Pure. Returns {kind, c, n, edges, hinges, r, out, observed, why}."""
    lo, hi, _, _ = front_extent(front)
    mid = (lo + hi) / 2.0
    width = max(0.15, (hi - lo) / PATCH_SPAN)
    edges = (mid - width / 2.0, mid + width / 2.0)
    m = {"kind": "slide" if ctype == "Drawer" else "swing", "c": tuple(front["c"]), "n": tuple(front["n"]),
         "edges": edges, "hinges": list(edges), "r": min(DOOR_MAX.get(ctype, 0.6), DOOR_FRAC * width),
         "out": DRAWER_OUT, "observed": False, "why": "prior (hinge unknown: both edges)", "width": round(width, 2)}
    if prev is not None and prev.get("observed"):
        m.update({k: prev[k] for k in ("hinges", "r", "out", "observed", "why")})
    pts = list(door_pts or ())
    if len(pts) >= 3:
        c, (nx, nz) = front["c"], front["n"]
        ls = [(_lat(front, p[0], p[2]), (p[0] - c[0]) * nx + (p[2] - c[2]) * nz) for p in pts]
        if m["kind"] == "slide":
            out = max(s for _, s in ls)
            m.update(out=max(0.10, min(DRAWER_OUT, out)), observed=True, why=f"seen pulled out {out:.2f} m")
        else:
            lm = sum(l for l, _ in ls) / len(ls)
            if abs(lm - mid) < 0.25 * width:           # not clearly at one edge: hinge stays unknown
                return m
            h = min(edges, key=lambda e: abs(e - lm))
            r_seen = max(math.hypot(l - h, s) for l, s in ls)
            # the leaf is often only partly in view (edge-on, out of the frame): never shrink below the prior radius
            r = max(m["r"], min(DOOR_MAX.get(ctype, 0.6) + 0.1, r_seen))
            m.update(hinges=[h], r=r, observed=True,
                     why=f"hinge on the {'right' if h > mid else 'left'} edge (door seen open, {len(pts)} points)")
    return m


def swept(model, x: float, z: float, margin: float = AGENT_R) -> bool:
    """Is the floor point (x, z) inside the door's sweep (+ body clearance)? Front side of the closed plane only."""
    c, (nx, nz) = model["c"], model["n"]
    s = (x - c[0]) * nx + (z - c[2]) * nz
    if s < -0.05:
        return False
    lat = _lat(model, x, z)
    if model["kind"] == "slide":
        return s <= model["out"] + margin and model["edges"][0] - margin <= lat <= model["edges"][1] + margin
    return any(math.hypot(lat - h, s) <= model["r"] + margin for h in model["hinges"])


def swept_cells(gm, model, margin: float = AGENT_R) -> set:
    return {c for c in gm.free if swept(model, *gm.world(c), margin=margin)}


def clearance(model, x: float, z: float) -> float:
    """How far (m) the floor point (x, z) is outside the door's sweep (negative = inside), body margin excluded."""
    c, (nx, nz) = model["c"], model["n"]
    s = (x - c[0]) * nx + (z - c[2]) * nz
    if s < -0.05:
        return 9.0
    lat = _lat(model, x, z)
    if model["kind"] == "slide":
        return max(s - model["out"], model["edges"][0] - lat, lat - model["edges"][1])
    return min(math.hypot(lat - h, s) - model["r"] for h in model["hinges"])


def put_cells(gm, front, cam_h: float, ctype: str, model=None, exclude=(), square: float = PUT_SQUARE,
              is_open: bool = False, least_swept: bool = False, stages: Optional[dict] = None) -> list:
    """Free own-map cells to put into (and open, and close) an openable from: on the front side, within ``square`` of
    the front normal, outside the door's sweep, the front's nearest height within PUT_OPEN3D (unless ``is_open``) and
    an interior point within PUT_REACH3D of the eye (crouched eye for low fronts). Sorted best first as
    ``(score, cell)``. ``least_swept``: no cell is outside the sweep -- the reachable square cells ordered by their
    clearance from it (the cell straight in front first when the hinge is unknown). Pure."""
    c, (nx, nz) = front["c"], front["n"]
    band = height_band(front, cam_h, ctype)
    dstar = INSPECT_D[band]
    low = c[1] < CROUCH_Y
    eye = cam_h - (CROUCH_DROP if low else 0.0)
    y_open = max(front["ymin"], min(front["ymax"], eye))
    cav = CAVITY.get(ctype, 0.12) if ctype != "Drawer" else -0.25
    tb = getattr(gm, "_tb", ())
    out = []
    st = stages if stages is not None else {}
    for k in ("free", "excluded_or_blocked", "distance", "square", "reach", "sweep", "ok"):
        st.setdefault(k, 0)
    for cell in gm.free:
        st["free"] += 1
        if cell in exclude or cell in tb:
            st["excluded_or_blocked"] += 1
            continue
        wx, wz = gm.world(cell)
        vx, vz = wx - c[0], wz - c[2]
        r = math.hypot(vx, vz)
        if r < 0.35 or r > dstar + 0.35:
            st["distance"] += 1
            continue
        along = (vx * nx + vz * nz) / r
        if along < square:
            st["square"] += 1
            continue
        if (not is_open and math.hypot(r, eye - y_open) > PUT_OPEN3D) or math.hypot(r + cav, eye - y_open) > PUT_REACH3D:
            st["reach"] += 1
            continue
        off = math.degrees(math.acos(max(-1.0, min(1.0, along))))
        score = off / 10.0 + abs(r - dstar) / 0.25
        if model is not None and least_swept:
            st["ok"] += 1
            out.append((-round(clearance(model, wx, wz), 2), score, cell))
            continue
        if model is not None and swept(model, wx, wz):
            st["sweep"] += 1
            continue
        st["ok"] += 1
        out.append((score, cell))
    out.sort()
    return [(t[0] + t[1] / 100.0, t[2]) for t in out] if least_swept else out


# ----------------------------------------------------------------------------------------------- unreachable fixes
#: (2026-10-09, "fix: unreachable failures"; Astra-reviewed; docs/step4-progress.md). Console policy only.
#: A. an OPEN door that stands still is an obstacle where its LEAF is; the rest of its sweep matters only while it moves
#:    -> leaf footprint (fitted to its own observed door points, inflated) HARD, the rest of the sweep SOFT.
#: B. an estimate that no own-map cell can act on (3D, either stance) triggers an investigation of its box's depth:
#:    a spatially coherent NEAR component well in front of the box median is taken as a foreground candidate
#:    (episode-local hypothesis with frame / pixel provenance; memory unchanged); the reach gate still decides.
#: D. stand-off and out-of-reach recovery from predicted 3D reach per stance, never a blind step away.
LEAF_PAD = 0.18                # m: cell-centre test slack around the fitted leaf (half a 0.25 m cell's diagonal, 0.177)
REACH_EPS = 0.10               # m: target + eye position error allowed for in reach PREDICTIONS (the gate is unchanged).
                               #    Declared; s3 found 54/58 depth sightings within 0.10 m on FloorPlan1 (tuned on FP1)
FG_SPAN = 0.25                 # m: depth extent along the ray one near fixture may have (declared prior: a faucet,
                               #    handle or utensil is thinner than this)
FG_GAP = 0.30                  # m: the box median must lie at least this far behind the near component (= objmem's
                               #    OCCLUSION_TOL: nearer than this counts as the same surface)
DEPTH_ERR = (0.02, 0.01)       # depth error model for the component tolerance: 0.02 m + 1 % of range (declared; THOR
                               #    depth is exact, a real sensor is not)
FG_MIN_PX = 4                  # px: fewer near pixels than this are noise, not an object
FG_MAX_FRAC = 0.5              # a near component covering this much of the box's valid depth: no ambiguity


def foreground_components(depth, bbox, n: int = 6, k: int = 1) -> list:
    """Up to ``k`` DISTINCT near spatially coherent depth components inside ``bbox`` (nearest first) when the box is
    depth-ambiguous (a thin object in front of a far background: most of the box is what is behind it), each as a pixel
    list ordered from the component's interior centre outwards. [] when there is no ambiguity (the nearest component
    covers most of the box, lies within FG_GAP of the box median, or is smaller than FG_MIN_PX). The nearer ones can be
    something else in front of the object (B1: a running faucet's water stream), hence more than one. Pure, own depth."""
    import numpy as np
    if depth is None or bbox is None:
        return []
    d = np.asarray(depth, dtype="float32")
    h, w = d.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return []
    win = d[y1:y2 + 1, x1:x2 + 1]
    valid = np.isfinite(win) & (win > 0)
    nv = int(valid.sum())
    if nv < FG_MIN_PX:
        return []
    med = float(np.median(win[valid]))
    out, used = [], set()
    for seed in np.unique(win[valid])[:3 + 3 * (k - 1)]:   # the nearest few depth levels as seeds (noise-tolerant)
        span = FG_SPAN + DEPTH_ERR[0] + DEPTH_ERR[1] * float(seed)
        mask = valid & (win <= float(seed) + span)
        if used:
            for (yy, xx) in used:
                mask[yy, xx] = False                   # a further component: not the pixels of one already taken
        comp = _largest_component(mask)
        if comp is None or len(comp) < FG_MIN_PX:
            continue
        if len(comp) >= FG_MAX_FRAC * nv:
            if out:
                continue
            return []
        cd = float(np.median([win[y, x] for y, x in comp]))
        if med - cd < FG_GAP:
            if out:
                continue
            return []
        cs = set(comp)
        inner = [(y, x) for y, x in comp
                 if all((y + a, x + b) in cs for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        pts = inner or comp
        cy = sum(p[0] for p in pts) / len(pts)
        cx = sum(p[1] for p in pts) / len(pts)
        pts = sorted(pts, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
        step = max(1, len(pts) // (4 * n))
        out.append([(int(x + x1), int(y + y1)) for y, x in pts[::step][:n]])
        if len(out) >= k:
            break
        used |= cs
        # components further away than this one only: grow the mask exclusion to its whole depth band
        used |= {(yy, xx) for yy, xx in zip(*np.nonzero(valid & (win <= float(seed) + span)))}
    return out


def foreground_points(depth, bbox, n: int = 6) -> list:
    """Pixels of the NEAREST spatially coherent depth component inside ``bbox`` when the box is depth-ambiguous (see
    ``foreground_components``), else []. Pure, own depth only."""
    c = foreground_components(depth, bbox, n, 1)
    return c[0] if c else []


#: B1 (2026-10-09): re-aiming at an object THIS agent put somewhere (a dish in a sink) when the detector no longer
#: finds it. The aim comes from the agent's own depth and pose (change since the frame it put the object from, then a
#: declared fallback grid over the receptacle's extent), never from the detector; the reach gate, the success bool and
#: a state re-check still decide. All numbers are declared priors or derived from the agent's own observations.
TOGGLE_PRIMS = ("ToggleObjectOn", "ToggleObjectOff")
BLIND_PRIMS = ("CleanObject", "PickupObject")   # identity-safe enough: the object was put by this agent, here
BLIND_MAX = 10                  # pixels tried per interaction (declared; each is one cheap actuator call)
BLIND_COMPS = 3                # change components tried before the fallback grid (declared)
BLIND_STRIDE = 3               # px: sampling stride over the receptacle region (declared; the frame is ~300 px)
BLOCK_MARGIN = 0.10          # m: B1f line-of-sight margin on top of the depth error model (a record is a surface point; the object sits a little behind it)
PUT_HALF_MIN, PUT_HALF_MAX = 0.15, 0.60   # m: clamp of the receptacle half-extent read from its box at the put (declared)
PUT_RING = 0.08               # m: the put point's neighbourhood tried first when the change test is blind there (declared)
PUT_DROP = 0.35                # m: ... and down to this far below it (a basin floor lies below its rim; declared prior)
PUT_RISE = 0.40                # m: the region reaches this far above the put point's surface (declared: dish + a hand)


#: B2 (2026-10-09). Detector classes that can hold or carry things (a published, scene-independent type prior -- the
#: affordance table of the simulator's object types, NOT a read of any scene): a pixel inside one of these boxes is a
#: pixel on that thing, not on the receptacle being put on.
RECEPTACLE_TYPES = frozenset({
    "CounterTop", "SinkBasin", "Sink", "BathtubBasin", "StoveBurner", "Cabinet", "Drawer", "Fridge", "Microwave",
    "Toaster", "CoffeeMachine", "GarbageCan", "Plate", "Bowl", "Pot", "Pan", "Cup", "Mug", "Box", "DiningTable",
    "SideTable", "Shelf", "Stool", "Chair", "Sofa", "ArmChair", "Bed", "Desk", "Dresser", "TVStand", "Ottoman",
    "CoffeeTable", "Safe", "Kettle", "Toilet", "TableTop", "Bathtub"})
PUT_PAD = 0.06                 # box-edge margin kept clear of a competing receptacle, as a fraction of its long side (declared)
PUT_SMALL = 0.25               # a box this small relative to the receptacle's is a thing standing on it (declared)
PUT_PLANE_BIN = 0.04           # m: height histogram bin of the support-plane estimate (declared; 2x the depth error at 1 m)
PUT_PLANE_MIN = 3              # candidate pixels a support-plane hypothesis needs (declared)
PUT_BASE_TOL = 0.06            # m: an object's observed base this far BELOW the support plane = not on it (declared)
PUT_REJECT_R = 0.25            # m: a spot where a put was judged wrong stays excluded within this radius (~ object size)
#: receptacles that are put INTO (a basin's floor, a can's bottom): the support-plane check of where a put landed is
#: about open surfaces, so it is not run for these (a lower place is where they are meant to be)
CONTAINER_LIKE = frozenset({"SinkBasin", "Sink", "BathtubBasin", "GarbageCan", "Box", "Bowl", "Pot", "Pan", "Cup", "Mug",
                            "Kettle", "Toilet", "Bathtub"})
PUT_SURFACE_ON = True           # B2 switch (the offline test shows the legacy behaviour with it off)
PUT_IN_VERIFY_ON = True         # B3 switch (the offline test shows the legacy behaviour with it off)
PUT_IN_SIGMAS = 3.0            # B3: a put into a container is judged wrong only beyond this many combined depth sigmas (declared)
PUT_DEEP = 0.15                # m: a surface this far behind the closed front plane is the cavity itself, not the rim / door /
                               # overhang the counter above shares with it (door + frame thickness are a few cm, declared)
PUT_BAD_R = 0.5                # m: beliefs of a receptacle type within this of an instance judged bad are that instance (a cabinet is ~0.6 m wide, declared)
PUT_IN_SAME = 0.15             # m: a detection this near one seen before the put is the same instance (declared)
PUT_FREE_MAX = 3               # B3 r2: attempts per put that find no pixel of the opening in view / learn a landing was wrong
                               # are not charged to the retry budget (they change the world by nothing or are undone and are
                               # bounded here and by PUT_FIX_MAX); the (PUT_FREE_MAX+1)th is charged
PUT_IN_REJECT_R = 0.20         # m: a landing spot judged wrong stays excluded within this 3D radius (~ a shaker / cup, declared)
PUT_SAME_INST = 0.10           # m: two remembered places this near are one instance (declared: a belief's own position error, ~2 sigma at
                               # 1 m; stacked drawers lie >= 0.15 m apart)
PUT_DRAWER_SAME = 0.15         # m: a remembered open drawer / door record this near (3D) a belief is that drawer (declared: about one drawer
                               # height; the neighbours above and below a drawer lie at least this far from it)
PUT_DRAWER_PIX_ON = True       # B4 switch (ablation): tray-aimed pixels, up-facing / below-rim / off-plane ranking, observed band
PUT_REFUSE_MAX = 2             # B4: refused puts at one receptacle instance before another instance of the type is tried (declared:
                               # a refusal cannot be told from a bad pixel, so one is retried at another point; two in a row
                               # blame the instance for the rest of the episode)
PUT_FIX_MAX = 2                # corrections (pick back up + re-put) one put may spend; shares the step's retry budget


PUT_PLANE_FRAC = 0.05          # a higher level counts as the surface from this share of the up-facing pixels (declared)
PUT_UP_MIN = 0.85              # |n_y| of a surface normal that counts as facing up (declared; ~32 deg)
PUT_NORMAL_STRIDE = 3          # px: neighbour offset of the normal estimate (declared)
PUT_EDGE_JUMP = 0.30           # m: a depth jump between neighbours = an edge, not a surface (declared; = FG_GAP)


def _horizontal(depth, pose, fov, p, wh, s: int = PUT_NORMAL_STRIDE) -> bool:
    """True if the surface at pixel ``p`` faces up: the normal of the 3D points at p +/- s px (own depth + pose) is
    near vertical and the neighbours are on one surface (no depth edge). Pure."""
    ds, pts = [], []
    for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
        q = (p[0] + dx, p[1] + dy)
        d = _depth(depth, q)
        xyz = objmem.backproject(q[0], q[1], d, pose, wh, fov) if d else None
        if xyz is None:
            return False
        ds.append(d)
        pts.append(xyz)
    if max(ds) - min(ds) > PUT_EDGE_JUMP:
        return False
    v1 = [pts[0][i] - pts[1][i] for i in range(3)]
    v2 = [pts[2][i] - pts[3][i] for i in range(3)]
    n = (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])
    nn = math.sqrt(sum(c * c for c in n))
    return nn > 1e-9 and abs(n[1]) / nn >= PUT_UP_MIN


def put_surface(depth, pose, fov, bbox, dets, rtype: str, rejected=(), n: int = 9, max_depth: Optional[float] = None):
    """Pixels of the receptacle box ``bbox`` that are on the receptacle's own surface, best first (B2, live 2026-10-09:
    ``put Pot -> CounterTop`` landed in the sink: a counter box spans the whole counter, including the basin cut into
    it and whatever else is in view, and the nearest-the-centre pixel was a basin pixel). From the agent's own frame
    only: depth, pose, detector boxes. Tiers, each a strict relaxation of the one before:
      1. not inside a competing receptacle's box (padded), not inside a small box (a thing standing on it), then only
         pixels on the dominant horizontal support plane of what is left (histogram of back-projected heights);
      2. the same without the small-box rule;
      3. nothing left: if a competing receptacle box lies over the receptacle's box the answer is ``blocked`` (no safe
         pixel -- the caller must look again or fail), else the legacy centre pixel is allowed.
    ``rejected`` = world (x, z, r) spots where an earlier put was judged wrong: excluded in map coordinates.
    -> {"pts", "plane" (m or None), "tier", "blocked", "note"}. Pure."""
    wh = _wh(depth)
    ra = max(1.0, _area(bbox))
    comp, small = [], []
    for o in dets or ():
        b = o.get("bbox_xyxy")
        if not b or o.get("type") == rtype:
            continue
        if o.get("type") in RECEPTACLE_TYPES and _area(b) < ra:
            pad = PUT_PAD * max(b[2] - b[0], b[3] - b[1])
            comp.append((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))
        elif _area(b) <= PUT_SMALL * ra:
            small.append(b)
    cands = []
    for p in candidate_points(depth, bbox, n=n, agree=False, max_depth=max_depth):
        d = _depth(depth, p)
        xyz = objmem.backproject(p[0], p[1], d, pose, wh, fov) if d else None
        if xyz is None:
            continue
        if any(math.hypot(xyz[0] - x, xyz[2] - z) < r for x, z, r in rejected):
            continue
        cands.append((p, xyz, d))
    free = [c for c in cands if not any(_inside(c[0], b) for b in comp)]
    overlapped = len(free) < len(cands)
    note = f"{len(cands)} pixels, {len(cands) - len(free)} inside competing receptacle boxes"
    for tier, pool in ((1, [c for c in free if not any(_inside(c[0], b) for b in small)]), (2, free)):
        if not pool:
            continue
        plane = None
        flat = [c for c in pool if _horizontal(depth, pose, fov, c[0], wh)]      # a support surface faces up
        if len(flat) >= PUT_PLANE_MIN:
            bins = {}
            for c in flat:
                bins.setdefault(int(round(c[1][1] / PUT_PLANE_BIN)), []).append(c)
            counts = {k: len(bins.get(k - 1, [])) + len(v) + len(bins.get(k + 1, [])) for k, v in bins.items()}
            top = max(counts.values())
            if rtype in CONTAINER_LIKE:               # put INTO it: where most of its up-facing pixels are (the floor)
                k = max(k for k, c in counts.items() if c >= 0.8 * top)
            else:                                     # put ON it: a cut-out (basin, hole) is lower than its surround,
                good = [k for k, c in counts.items() if c >= max(PUT_PLANE_MIN, PUT_PLANE_FRAC * len(flat))]     # never higher
                k = max(good) if good else None
            if k is not None:
                support = bins.get(k - 1, []) + bins.get(k, []) + bins.get(k + 1, [])
                ys = sorted(c[1][1] for c in support)
                if len(support) >= PUT_PLANE_MIN:
                    plane = ys[len(ys) // 2]
        keep = pool
        if plane is not None:
            keep = [c for c in flat if abs(c[1][1] - plane) <= max(PUT_BASE_TOL, 2 * (DEPTH_ERR[0] + DEPTH_ERR[1] * c[2]))]
            note += f"; support plane y={plane:.2f} m keeps {len(keep)} of {len(pool)}"
        if keep:
            return {"pts": [c[0] for c in keep], "plane": plane, "tier": tier, "blocked": False, "note": note}
    if overlapped:
        return {"pts": [], "plane": None, "tier": 3, "blocked": True,
                "note": note + "; no pixel on the receptacle's own surface"}
    return {"pts": [], "plane": None, "tier": 3, "blocked": False, "note": note}


def put_region(p0, half, pose, wh, fov, rise: float = PUT_RISE):
    """Pixel box (x1, y1, x2, y2) of the receptacle volume around the put point ``p0`` seen from ``pose``: the
    projection of its 8 corners (half-extent ``half`` m in x and z, from just below ``p0`` up ``rise`` m), clipped to
    the image. None if it lies wholly behind the camera or off screen. Pure."""
    pts = []
    for dx in (-half, half):
        for dz in (-half, half):
            for dy in (-PUT_DROP, rise):
                pr = objmem.project((p0[0] + dx, p0[1] + dy, p0[2] + dz), pose, wh, fov)
                if pr is not None:
                    pts.append(pr)
    if len(pts) < 4:
        return None
    w, h = wh
    x1, x2 = max(0, int(min(p[0] for p in pts))), min(w - 1, int(max(p[0] for p in pts)))
    y1, y2 = max(0, int(min(p[1] for p in pts))), min(h - 1, int(max(p[1] for p in pts)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def target_blocked(p0, depth, pose, fov):
    """B1f: is the line of sight from the camera to the recorded point ``p0`` blocked by something nearer? The point is
    projected into the current frame and the depth read around that pixel (3x3 median) is compared with the projected
    depth: nearer by more than the margin = blocked (live 2026-10-10: crouched at a sink, the camera below the counter
    top, the basin's contents hidden behind the counter lip; the pixel read the cabinet front 0.3 m away instead of the
    dish 0.65 m away). True / False, or None when it cannot be judged (off screen, no depth). Same depth convention as
    ``objmem.project`` (camera-axis depth). Margin = 2 x the depth error model + MARGIN_REC (the record is a surface
    point, the object may sit a little behind it). Pure."""
    wh = _wh(depth)
    pr = objmem.project(p0, pose, wh, fov)
    if pr is None:
        return None
    w, h = wh
    cx, cy = int(round(pr[0])), int(round(pr[1]))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    vals = sorted(v for v in (_depth(depth, (cx + dx, cy + dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)) if v is not None)
    if len(vals) < 3:
        return None
    med = vals[len(vals) // 2]
    margin = 2.0 * (DEPTH_ERR[0] + DEPTH_ERR[1] * pr[2]) + BLOCK_MARGIN
    return bool(med < pr[2] - margin)


def change_mask(before_depth, before_pose, depth, pose, fov, region, stride: int = BLIND_STRIDE, vol=None):
    """Newly occupied space inside ``region`` (pixel box in the CURRENT frame): current pixel -> world point -> the
    frame it was put from; the point is new when that earlier frame saw a surface FARTHER along the same ray than the
    point (by more than the depth error model) in the whole small neighbourhood of the projected pixel (a depth edge,
    e.g. the far window beside a fixture, must not read as 'new'). Where the point was not visible then (off screen,
    behind the held object, no depth) nothing is claimed. ``vol`` = (p0, half, rise): only points inside the
    receptacle volume (within ``half`` m of the put point horizontally, from PUT_DROP below it up ``rise`` m) count.
    Returns ``{(gx, gy): (px, py)}`` for the flagged stride-grid cells. Pure."""
    out = {}
    if before_depth is None or depth is None or region is None:
        return out
    wh, bwh = _wh(depth), _wh(before_depth)
    x1, y1, x2, y2 = region
    for gy, py in enumerate(range(y1, y2 + 1, stride)):
        for gx, px in enumerate(range(x1, x2 + 1, stride)):
            d = _depth(depth, (px, py))
            if d is None:
                continue
            P = objmem.backproject(px, py, d, pose, wh, fov)
            if P is None:
                continue
            if vol is not None:
                p0, half, rise = vol
                if abs(P[0] - p0[0]) > half or abs(P[2] - p0[2]) > half or not (p0[1] - PUT_DROP <= P[1] <= p0[1] + rise):
                    continue
            q = objmem.project(P, before_pose, bwh, fov)
            if q is None or not (1 <= q[0] < bwh[0] - 1 and 1 <= q[1] < bwh[1] - 1):
                continue
            nb = [_depth(before_depth, (q[0] + a, q[1] + b)) for a in (-1, 0, 1) for b in (-1, 0, 1)]
            if any(v is None for v in nb):
                continue
            if min(nb) - q[2] > 2 * DEPTH_ERR[0] + DEPTH_ERR[1] * q[2]:
                out[(gx, gy)] = (px, py)
    return out


def blind_pixels(rec, depth, pose, fov, n: int = BLIND_MAX):
    """Pixels to try, in order, to act on an object this agent put down (``rec`` = its put record): the interior of the
    largest newly-occupied components (``change_mask``; weakly ordered by nearness to the put point), then a declared
    fallback grid over the receptacle's region nearest the put point first. Returns (pixels, note). Pure."""
    wh = _wh(depth)
    region = put_region(rec["p0"], rec["half"], pose, wh, fov)
    if region is None:
        return [], "the receptacle region is not in view"
    cells = change_mask(rec["depth"], rec["pose"], depth, pose, fov, region,
                        vol=(rec["p0"], rec["half"], PUT_RISE))
    ref = objmem.project(rec["p0"], pose, wh, fov)
    refp = (ref[0], ref[1]) if ref else ((region[0] + region[2]) / 2.0, (region[1] + region[3]) / 2.0)
    comps, seen = [], set()
    for c0 in sorted(cells):
        if c0 in seen:
            continue
        q, i = [c0], 0
        seen.add(c0)
        while i < len(q):
            gx, gy = q[i]
            i += 1
            for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(q)
    out, note = [], ""
    refp_ = refp

    def near(c):
        cx = sum(cells[g][0] for g in c) / len(c)
        cy = sum(cells[g][1] for g in c) / len(c)
        return math.hypot(cx - refp_[0], cy - refp_[1]) / max(1.0, math.sqrt(len(c)))     # big and close first

    comps = sorted([c for c in comps if len(c) >= 3], key=near)
    for c in comps[:BLIND_COMPS]:
        cs = set(c)
        inner = [g for g in c if all((g[0] + a, g[1] + b) in cs for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))] or c
        my = sum(cells[g][1] for g in inner) / len(inner)
        mx = sum(cells[g][0] for g in inner) / len(inner)
        g = min(inner, key=lambda g: (cells[g][0] - mx) ** 2 + (cells[g][1] - my) ** 2)
        out.append(cells[g])
    note = (f"{len(comps)} newly occupied component(s) in the receptacle volume since the put frame, the one nearest "
            f"the put point has {len(comps[0])} sample(s)" if comps else
            "no newly occupied space seen in the receptacle volume since the put frame")
    # the put point itself, then a ring around it (declared 0.08 m = half of a small dish): the put frame had the held
    # object in front of exactly this part of the view, so the change test is blind there (live 2026-10-09, dishes)
    ring = []
    if ref and target_blocked(rec["p0"], depth, pose, fov) is not True:
        r_px = max(4.0, PUT_RING * objmem._focal(wh[1], fov) / max(0.3, ref[2]))
        offs = [(0, 0)] + [(dx * r_px, dy * r_px) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))]
        for dx, dy in offs:
            q_ = (int(round(ref[0] + dx)), int(round(ref[1] + dy)))
            if region[0] <= q_[0] <= region[2] and region[1] <= q_[1] <= region[3] and _depth(depth, q_) is not None:
                ring.append(q_)
    sep0 = 4
    for q_ in ring:
        if len(out) >= n:
            break
        if all(math.hypot(q_[0] - o[0], q_[1] - o[1]) > sep0 for o in out):
            out.append(q_)
    grid = []
    for py in range(region[1], region[3] + 1, max(2 * BLIND_STRIDE, (region[3] - region[1]) // 5)):
        for px in range(region[0], region[2] + 1, max(2 * BLIND_STRIDE, (region[2] - region[0]) // 5)):
            d_ = _depth(depth, (px, py))
            if d_ is None:
                continue
            P_ = objmem.backproject(px, py, d_, pose, wh, fov)
            p0_ = rec["p0"]
            if P_ is not None and abs(P_[0] - p0_[0]) <= rec["half"] and abs(P_[2] - p0_[2]) <= rec["half"]                     and p0_[1] - PUT_DROP <= P_[1] <= p0_[1] + PUT_RISE:
                grid.append((px, py))
    grid.sort(key=lambda p: (p[0] - refp[0]) ** 2 + (p[1] - refp[1]) ** 2)
    sep = max(4, BLIND_STRIDE * 2)
    for p in grid:
        if len(out) >= n:
            break
        if all(math.hypot(p[0] - o[0], p[1] - o[1]) > sep for o in out):
            out.append(p)
    return out[:n], note



def _largest_component(mask):
    """Largest 4-connected component of a boolean array as [(y, x), ...] (BFS), or None."""
    import numpy as np
    seen = np.zeros(mask.shape, dtype=bool)
    best = None
    hh, ww = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        q, i = [(int(y0), int(x0))], 0
        seen[y0, x0] = True
        while i < len(q):
            y, x = q[i]
            i += 1
            for ny, nx_ in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < hh and 0 <= nx_ < ww and mask[ny, nx_] and not seen[ny, nx_]:
                    seen[ny, nx_] = True
                    q.append((ny, nx_))
        if best is None or len(q) > len(best):
            best = q
    return best


def reach_pred(h: float, eye_stand: float, y: float) -> tuple:
    """Predicted 3D eye -> target distance from a floor point ``h`` m (horizontal) from a target at height ``y``:
    ``(distance, stance)``; standing is kept when it already reaches (within ACT_REACH - REACH_EPS), else the nearer of
    the two stances. Pure."""
    ds = math.hypot(h, eye_stand - y)
    dc = math.hypot(h, eye_stand - CROUCH_DROP - y)
    # crouching helps only for a target BELOW the crouched eye (it then looks down at it); at or above that height the
    # eye would sit at counter level behind the counter edge (live 2026-10-09: a crouch before the faucet, refused)
    if ds <= ACT_REACH - REACH_EPS or ds <= dc or y >= eye_stand - CROUCH_DROP:
        return ds, "stand"
    return dc, "crouch"


def ring_max(eye_stand: float, y: float, limit: Optional[float] = None) -> Optional[float]:
    """Largest horizontal stand-off from which some stance's eye is within ``limit - REACH_EPS`` (3D) of a target at
    height ``y``: sqrt((R - eps)^2 - dy^2) for the better stance; None when neither stance can ever reach it. Pure."""
    R = (ACT_REACH if limit is None else limit) - REACH_EPS
    best = None
    for eye in (eye_stand, eye_stand - CROUCH_DROP):
        if eye < eye_stand and y >= eye:
            continue                                   # crouching only for targets below the crouched eye
        q = R * R - (eye - y) ** 2
        if q > 0:
            best = max(best or 0.0, math.sqrt(q))
    return best


def min_view_h(eye: float, y: float) -> float:
    """Closest horizontal distance from which a target ``eye - y`` m below the eye is still in view at the steepest
    allowed pitch (PITCH_MAX down). Pure."""
    dy = eye - y
    return dy / math.tan(math.radians(PITCH_MAX)) if dy > 0 else 0.0


def reach_from_map(gm, xyz, eye_stand: float, cells=None) -> float:
    """Smallest predicted 3D eye distance from ANY own-map cell (either stance) to ``xyz`` -- geometry only, not
    connectivity (Astra: keep the two apart). Pure."""
    best = 99.0
    for c in (cells if cells is not None else (set(gm.free) | set(getattr(gm, "here_free", ())))):
        wx, wz = gm.world(c)
        best = min(best, reach_pred(math.hypot(wx - xyz[0], wz - xyz[2]), eye_stand, xyz[1])[0])
    return best


def leaf_cells(gm, model, pts, pad: Optional[float] = None) -> set:
    """HARD cells of an open swing door: own-map cells whose centre lies within AGENT_R + LEAF_PAD of the leaf fitted to
    the door's own observed points (front side only): the segment from the hinge (observed, else the candidate edge
    nearer the points) on the closed front line through the farthest observed point, plus each point itself. Fewer
    than 3 observed points = no leaf estimate: nothing is hard (the whole sweep is soft). Pure."""
    pad = AGENT_R + LEAF_PAD if pad is None else pad
    c, (nx, nz) = model["c"], model["n"]
    q = [p for p in pts or () if -0.05 <= (p[0] - c[0]) * nx + (p[2] - c[2]) * nz <= model["r"] + 0.3]
    if len(q) < 3:
        return set()
    lm = sum(_lat(model, p[0], p[2]) for p in q) / len(q)
    hl = min(model["hinges"], key=lambda e: abs(e - lm))
    tx, tz = -nz, nx                                   # lateral unit vector (as _lat measures it)
    c_lat = _lat(model, c[0], c[2])
    hx, hz = c[0] + tx * (hl - c_lat), c[2] + tz * (hl - c_lat)
    far = max(q, key=lambda p: math.hypot(p[0] - hx, p[2] - hz))
    fx, fz = far[0], far[2]

    def seg_d(px, pz):
        vx, vz = fx - hx, fz - hz
        L2 = vx * vx + vz * vz
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((px - hx) * vx + (pz - hz) * vz) / L2))
        return math.hypot(px - (hx + t * vx), pz - (hz + t * vz))
    out = set()
    for cell in gm.free:
        wx, wz = gm.world(cell)
        if seg_d(wx, wz) <= pad or any(math.hypot(wx - p[0], wz - p[2]) <= pad for p in q):
            out.add(cell)
    return out


def opened_front(d0, d1, pose, fov, near, ctype: str = "", step: int = 6):
    """What moved when a door opened, from two depth frames of the SAME pose (before / after the open): pixels that got
    DEEPER showed the closed door before (their before-points are the closed front -- the face that actually opened,
    even when the estimate took a side face for it); pixels that got NEARER show the open leaf now (door points). Only
    points within reach of ``near`` (the acted 3D point). Returns (front or None, door points). Pure, own depth only."""
    import numpy as np
    if d0 is None or d1 is None or getattr(d0, "shape", None) != getattr(d1, "shape", None):
        return None, []
    a, b = np.asarray(d0, dtype="float64")[::step, ::step], np.asarray(d1, dtype="float64")[::step, ::step]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    wh = (int(d0.shape[1]), int(d0.shape[0]))
    far, door = [], []
    rad = DOOR_MAX.get(ctype, 0.6) + 0.3
    for (iy, ix) in zip(*np.nonzero(ok & (b > a + 0.08))):
        xyz = objmem.backproject(ix * step, iy * step, float(a[iy, ix]), pose, wh, fov)
        if xyz and math.dist(xyz, near) <= 0.9:
            far.append(tuple(xyz))
    for (iy, ix) in zip(*np.nonzero(ok & (b < a - 0.08))):
        xyz = objmem.backproject(ix * step, iy * step, float(b[iy, ix]), pose, wh, fov)
        if xyz and math.dist(xyz, near) <= rad:
            door.append(tuple(xyz))
    if len(far) < 8:
        return None, door
    P = np.asarray(far)
    c = P.mean(axis=0)
    _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    nv = vt[-1]
    rms = float(sv[-1] / math.sqrt(len(far)))
    hx, hz = float(nv[0]), float(nv[2])
    hn = math.hypot(hx, hz)
    if rms > 0.04 or hn < 0.8:
        return None, door
    hx, hz = hx / hn, hz / hn
    eye = objmem._basis(pose)[0]
    if hx * (eye[0] - c[0]) + hz * (eye[2] - c[2]) < 0:
        hx, hz = -hx, -hz
    pts = far[:: max(1, len(far) // 49)]
    ys = [q[1] for q in far]
    return ({"pts": pts, "c": tuple(float(v) for v in c), "n": (hx, hz), "planar": True, "rms": rms,
             "ymin": min(ys), "ymax": max(ys), "bbox": None, "from": "what moved when it opened"}, door)


def leaf_front(near, eye, body_c, ctype: str):
    """The closed front inferred from an open door LEAF the agent sees close in front of it (own depth points that
    came nearer when it opened): the leaf is a vertical plane; at ~90 deg it sticks out along the front normal, its
    hinge end is the end nearer the container body (``body_c``, the agent's earlier estimate on it), and the opening is
    on the leaf side away from the camera (it was looking at the leaf's outer face). Returns a front patch (``from``:
    "the open door leaf") or None when the points are not one vertical plane. Pure; an approximation (opening angle
    assumed ~90 deg)."""
    import numpy as np
    if len(near) < 40:
        return None
    P = np.asarray(near, dtype="float64")
    xz = P[:, [0, 2]]
    m = xz.mean(axis=0)
    _, sv, vt = np.linalg.svd(xz - m, full_matrices=False)
    u = vt[0]
    if sv[1] / math.sqrt(len(near)) > 0.05:          # not a thin vertical slab
        return None
    t = (xz - m) @ u
    e1, e2 = m + u * t.min(), m + u * t.max()
    if math.hypot(*(e2 - e1)) < 0.25:
        return None
    bc = np.asarray([body_c[0], body_c[2]])
    hinge, free = (e1, e2) if np.linalg.norm(e1 - bc) <= np.linalg.norm(e2 - bc) else (e2, e1)
    n = (free - hinge) / max(1e-6, np.linalg.norm(free - hinge))
    lat = np.asarray([-n[1], n[0]])
    if lat @ (m - np.asarray([eye[0], eye[2]])) < 0:   # the opening is on the far side of the leaf
        lat = -lat
    length = float(np.linalg.norm(free - hinge))
    w = max(length, 0.3)
    c = hinge + lat * (w / 2.0)
    ys = P[:, 1]
    ylo, yhi = float(np.percentile(ys, 5)), float(np.percentile(ys, 95))
    pts = [(float(hinge[0] + lat[0] * w * f), float(y), float(hinge[1] + lat[1] * w * f))
           for f in (0.1, 0.3, 0.5, 0.7, 0.9) for y in np.linspace(ylo, yhi, 5)]
    return {"pts": pts, "c": (float(c[0]), (ylo + yhi) / 2.0, float(c[1])), "n": (float(n[0]), float(n[1])),
            "planar": True, "rms": None, "ymin": ylo, "ymax": yhi, "bbox": None, "from": "the open door leaf",
            "hinge": (float(hinge[0]), float(hinge[1])), "leaf": round(length, 2)}


def door_points(depth, pose, fov, boxes, front, ctype: str) -> list:
    """3D points (own depth) inside this type's detection boxes that lie IN FRONT of the closed front plane, at its
    height, near its edges: the open door leaf / pulled-out drawer. Pure; the floor and far background are dropped."""
    if depth is None or front is None:
        return []
    wh = _wh(depth)
    c, (nx, nz) = front["c"], front["n"]
    lo, hi, ymin, ymax = front_extent(front)
    mid, half = (lo + hi) / 2.0, (hi - lo) / PATCH_SPAN / 2.0
    reach = DOOR_MAX.get(ctype, 0.6) + 0.15 if ctype != "Drawer" else DRAWER_OUT + 0.1
    eye = objmem._basis(pose)[0]
    out = []
    for b in boxes:
        for p in candidate_points(depth, b, n=9, agree=False):
            d = _depth(depth, p)
            xyz = objmem.backproject(p[0], p[1], d, pose, wh, fov) if d else None
            if not xyz or math.hypot(xyz[0] - eye[0], xyz[2] - eye[2]) < 0.55:     # the hand / held object
                continue
            s_ = (xyz[0] - c[0]) * nx + (xyz[2] - c[2]) * nz
            lat = _lat(front, xyz[0], xyz[2])
            if not (0.04 <= s_ <= reach) or not (ymin - 0.05 <= xyz[1] <= ymax + 0.10):
                continue
            if not (mid - half - reach <= lat <= mid + half + reach):
                continue
            out.append(tuple(xyz))
    return out


def detector_types(detector) -> list:
    """The FloorPlan1 detector's class names (trained knowledge, not scene contents) as object types."""
    names = set()
    for part in (getattr(detector, "primary", None), getattr(detector, "specialist", None), detector):
        for v in (getattr(part, "names", None) or {}).values():
            names.add(str(v).split("_", 1)[0])
    return sorted(n for n in names if n)


def remembered_types(memory_summary) -> list:
    """Object types the memory summary lines mention first (``"Fridge [closed] on/in ? at (...)"``)."""
    out = []
    for line in memory_summary or ():
        m = re.match(r"\s*([A-Za-z]+)", str(line))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


# ----------------------------------------------------------------------------------------------- choosers
def prior_chooser(target, type_list, memory_summary):
    """No-LLM STAND-IN for s5 (labelled as such): rank place types by ``search.prior_for(target)``, remembered
    places first. Uses the curated containment prior over detector classes only -- no scene contents."""
    import search
    pri = search.prior_for(target)
    alias = getattr(search, "RECEPTACLE_ALIASES", {})

    def w(t):
        return pri.get(t, pri.get(alias.get(t, ""), 0.0))

    known = set(remembered_types(memory_summary))
    cands = [t for t in type_list if t != target and w(t) > 0]
    return sorted(cands, key=lambda t: (t not in known, -w(t), t))


prior_chooser.label = "prior (no-LLM stand-in: search.prior_for over detector classes)"

CHOOSER_PROMPT = """You are a kitchen robot looking for a {target}. It is not where you remember seeing things.
Object types this kitchen's detector knows: {types}.
What you remember seeing (type [state] place, position):
{memory}
Which places should you look in or on, most likely first? Answer ONLY a JSON list of up to {k} types from the list
above, e.g. ["Fridge", "CounterTop"]."""


def parse_places(text: str, type_list, target: str = "") -> list:
    """Place types from an LLM answer: a JSON list if present, else type names in order of appearance. Only
    names in ``type_list`` survive (the LLM may only rank the fixed list)."""
    allowed = {t.lower(): t for t in type_list}
    out = []
    m = re.search(r"\[[^\[\]]*\]", text or "")
    items = []
    if m:
        try:
            items = [str(x) for x in json.loads(m.group(0))]
        except ValueError:
            items = []
    if not items:
        hits = []
        for t in type_list:
            for mm in re.finditer(r"\b%s\b" % re.escape(t), text or "", flags=re.I):
                hits.append((mm.start(), t))
        items = [t for _, t in sorted(hits)]
    for x in items:
        t = allowed.get(x.strip().lower())
        if t and t != target and t not in out:
            out.append(t)
    return out


class LLMChooser:
    """Decision 4's chooser: ask an LLM (``ask(prompt) -> text``) where to look. Mockable; s6 wires Gemma."""

    label = "llm"

    def __init__(self, ask: Callable[[str], str], k: int = 6):
        self.ask, self.k = ask, k
        self.last = None

    def __call__(self, target, type_list, memory_summary):
        prompt = CHOOSER_PROMPT.format(target=target, types=", ".join(type_list), k=self.k,
                                       memory="\n".join(memory_summary or ["(nothing yet)"]))
        try:
            text = self.ask(prompt)
        except Exception as e:                       # an LLM outage degrades to "no ranking", never to oracle
            text, err = "", str(e)
        else:
            err = ""
        places = parse_places(text, type_list, target)[: self.k]
        self.last = {"prompt": prompt, "answer": text, "places": places, "error": err}
        return places


# ----------------------------------------------------------------------------------------------- systematic search route
#: (2026-10-08, "fix: systematic search route"; Astra-reviewed; console / exhaustive mode only). An explicit search
#: plan before moving: candidate containers from the agent's OWN front registry (front planes fitted to its own depth)
#: and remembered places, grouped into stops (one cell that inspects a drawer column and the neighbouring doors), the
#: stops ordered by a weighted search-cost surrogate (rank weights are preferences, not calibrated probabilities).
#: Thresholds are a priori choices on FloorPlan1 (development data), declared here, not tuned on held-out episodes.
STRAFE_MAX = 1.25              # m: a console walk this short keeps its heading (side / back steps, no turn); longer
                               #    walks face the way most of the path goes (forward-facing travel)
REG_RANGE = 3.0                # m: openable fronts seen this close enter the own front registry (the plan's input)
PLAN_RANGE = 2.5               # m: a front seen only once and from farther than this gets no stop of its own
SERVE_TOL = 1.2                # inspection-cell score slack (score = deg off / 10 + stand-off error / 0.25 m): a stop
                               #    serves a container from a cell at most ~12 deg / 0.3 m worse than its best cell
PLACE_SERVE = (0.45, 1.40)     # m: a stop serves a remembered place (one look) from this far
CLOSE_LOOK = 1.6               # m: a place in frame from this close counts as looked at
T_INSPECT, T_LOOK = 1.0, 0.3   # plan cost (m-equivalents) of inspecting a container / looking at a place
ROUTE_LAMBDA = 0.25            # weight of the total route length in the plan objective (discourages zig-zag)
RANK_DECAY = 0.6               # chooser rank r -> preference weight RANK_DECAY ** r (unranked types: half the last)
SEEN_RANGE = 3.5               # m: own depth points this close mark their map cell as SEEN (coverage phase)
SEEN_STRIDE = 12               # px: depth subsample for the seen-cells grid
VIEW_GAIN = 4                  # unseen cells of interest a coverage viewpoint must add to be visited
VIEW_R = 2.5                   # m: cells this close to a viewpoint count toward its gain
VIEW_FOV = 80.0                # deg: one look covers this much of the bearings (camera 90 deg, minus a margin)
MAX_STOP_PLANS = 12            # plan rebuilds one search may make (new containers seen, unreachable stops)
TIER_FRAC = 0.35               # pass 1 visits items weighted >= this x the best weight (the chooser's likely places,
                               #    e.g. cabinets for a plate, drawers for a spoon); pass 2 every other container
#: Step 5 / B1 (default OFF: only used when ``place_hints`` is given): a memory hint is a NAVIGATION prior, never a
#: sighting. An item within the hint's radius (the place-identity gate of its type, carried by the hint) of a hinted
#: place of the same type has its plan weight multiplied; manipulation still needs a current sighting.
HINT_BOOST = 4.0
HINT_RADIUS = 0.75                 # m: used only when a hint carries no radius of its own
#: DECLARED hand-designed height prior (general kitchen convention -- not derived from FloorPlan1's placements):
#: dishes / vessels are kept in wall cabinets more than in base cabinets; cutlery in drawers.
VESSELS = frozenset({"Plate", "Bowl", "Cup", "Mug", "WineBottle", "Bottle", "Glass", "Pot", "Pan", "Kettle"})
CUTLERY = frozenset({"ButterKnife", "Fork", "Spoon", "Spatula", "Ladle"})   # A26 item 1: Knife out of task scope (was in the set)
HEIGHT_PRIOR = {"vessel": {"high": 1.5, "low": 0.7, "Drawer": 0.6}, "cutlery": {"Drawer": 1.5, "high": 0.7}}


def serve_cells(gm, patch, cells) -> dict:
    """The cells (from ``inspection_cells``: hard checks already applied -- front side, stand-off band, reach, door
    sweep) from which a stop may inspect this front: every SQUARE one (heading within FRONT_OK of the front normal)
    plus any within SERVE_TOL of the best score. cell -> score. Pure."""
    if not cells:
        return {}
    best = cells[0][0]
    c, (nx, nz) = patch["c"], patch["n"]
    out = {}
    for sc, cell in cells:
        wx, wz = gm.world(cell)
        vx, vz = wx - c[0], wz - c[2]
        r = math.hypot(vx, vz) or 1e-6
        off = math.degrees(math.acos(max(-1.0, min(1.0, (vx * nx + vz * nz) / r))))
        if off <= FRONT_OK or sc <= best + SERVE_TOL:
            out[cell] = sc
    return out


def height_prior(target: str, ctype: str, y: Optional[float]) -> float:
    """The declared height prior factor for a container front centred at height ``y`` (own depth)."""
    kind = "vessel" if target in VESSELS else "cutlery" if target in CUTLERY else None
    if kind is None or y is None:
        return 1.0
    pri = HEIGHT_PRIOR[kind]
    if ctype == "Drawer":
        return pri.get("Drawer", 1.0)
    if ctype != "Cabinet":
        return 1.0
    return pri.get("high", 1.0) if y > HIGH_Y else pri.get("low", 1.0) if y < CROUCH_Y else 1.0


def rank_weight(ptype: str, ranked) -> float:
    """Chooser rank -> preference weight (a search-cost surrogate, not a probability)."""
    if ptype in ranked:
        return RANK_DECAY ** ranked.index(ptype)
    return 0.5 * RANK_DECAY ** len(ranked) if ranked else 1.0


def grid_dist(gm, start) -> dict:
    """Walking distance (m) on the own map from ``start`` to every reachable free cell (BFS, 4-connected)."""
    start = tuple(start)
    out, q, i = {start: 0}, [start], 0
    while i < len(q):
        c = q[i]
        i += 1
        for n in gm.neighbours(c):
            if n not in out:
                out[n] = out[c] + 1
                q.append(n)
    return {c: v * gm.cell for c, v in out.items()}


def cover_stops(items) -> list:
    """Group plan items into stops: greedy weighted set cover over the cells that serve them (``item["serve"]``:
    cell -> score, lower = squarer), then every item re-assigned to the chosen stop that serves it best (Astra: each
    item belongs to one stop). Returns ``[{"cell", "items": [item, ...]}]`` (unordered). Pure."""
    left = [it for it in items if it.get("serve")]
    chosen = []
    while left:
        cand = {}
        for it in left:
            for c, sc in it["serve"].items():
                v = cand.setdefault(c, [0.0, 0.0])
                v[0] += 1.0 + 2.0 * it["q"]
                v[1] += sc
        c = max(cand, key=lambda c: (round(cand[c][0] - 0.05 * cand[c][1], 9), -cand[c][1], c))
        chosen.append(c)
        left = [it for it in left if c not in it["serve"]]
    stops = {c: [] for c in chosen}
    for it in items:
        if not it.get("serve"):
            continue
        best = min((c for c in chosen if c in it["serve"]), key=lambda c: (it["serve"][c], c))
        stops[best].append(it)
    return [{"cell": c, "items": sorted(v, key=lambda it: -it["q"])} for c, v in stops.items() if v]


def route_cost(order, start, dist, lam: float = ROUTE_LAMBDA) -> float:
    """Weighted search-cost surrogate of visiting the stops in ``order`` from ``start``: the sum over items of
    q_item x the cumulative cost when that item is done (walking + per-item inspection), + ``lam`` x route length."""
    t = L = J = 0.0
    prev = tuple(start)
    for s in order:
        d = dist(prev, s["cell"])
        t += d
        L += d
        for it in s["items"]:
            t += it["cost"]
            J += it["q"] * t
        prev = s["cell"]
    return J + lam * L


def order_stops(stops, start, dist, lam: float = ROUTE_LAMBDA, iters: int = 60) -> list:
    """Order stops minimising ``route_cost``: a nearest-next seed and a weight-per-distance seed, then 2-opt segment
    reversals and single-stop relocations until no move improves (Astra: every move scored on the full objective)."""
    if len(stops) <= 1:
        return list(stops)

    def greedy(key):
        out, left, cur = [], list(stops), tuple(start)
        while left:
            s = min(left, key=lambda s: key(cur, s))
            out.append(s)
            left.remove(s)
            cur = s["cell"]
        return out

    seeds = [greedy(lambda cur, s: (dist(cur, s["cell"]), s["cell"])),
             greedy(lambda cur, s: ((dist(cur, s["cell"]) + 0.5) / max(1e-6, sum(it["q"] for it in s["items"])),
                                    s["cell"]))]
    best = min(seeds, key=lambda o: route_cost(o, start, dist, lam))
    bc = route_cost(best, start, dist, lam)
    n = len(best)
    for _ in range(iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = route_cost(cand, start, dist, lam)
                if c < bc - 1e-9:
                    best, bc, improved = cand, c, True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rest = best[:i] + best[i + 1:]
                cand = rest[:j] + [best[i]] + rest[j:]
                c = route_cost(cand, start, dist, lam)
                if c < bc - 1e-9:
                    best, bc, improved = cand, c, True
        if not improved:
            break
    return best


def refine_cells(order, start, dist, rounds: int = 2) -> list:
    """Keep the stop order, move each stop to the cell (among those that serve ALL its items) that shortens the walk
    in + out of it (several valid inspection cells per group; ties: the squarer cell). Pure."""
    for _ in range(rounds):
        for i, s in enumerate(order):
            common = None
            for it in s["items"]:
                common = set(it["serve"]) if common is None else common & set(it["serve"])
            if not common:
                continue
            prev = tuple(start) if i == 0 else order[i - 1]["cell"]
            nxt = order[i + 1]["cell"] if i + 1 < len(order) else None

            def cost(c):
                return (dist(prev, c) + (dist(c, nxt) if nxt is not None else 0.0),
                        sum(it["serve"][c] for it in s["items"]), c)
            s["cell"] = min(common, key=cost)
    return order


def _turn(a: float, b: float, d: int) -> float:
    return (b - a) % 360.0 if d > 0 else (a - b) % 360.0


def view_headings(bearings, yaw0: float, fov: float = VIEW_FOV) -> list:
    """Fewest look headings whose ``fov``-wide views cover every bearing (deg), ordered as ONE turn direction from
    ``yaw0`` with the least total rotation (no left/right oscillation). Pure."""
    bs = sorted({round(b % 360.0, 1) for b in bearings})
    if not bs:
        return []
    if len(bs) == 1:
        return [bs[0]]
    gaps = [((bs[(i + 1) % len(bs)] - bs[i]) % 360.0, i) for i in range(len(bs))]
    _, i = max(gaps)                                   # cut the circle at the largest empty gap
    seq = bs[i + 1:] + bs[:i + 1]
    s0 = seq[0]
    arc = [(b - s0) % 360.0 for b in seq]
    heads, k = [], 0
    while k < len(arc):
        lo = arc[k]
        hi = max(a for a in arc if a <= lo + fov)
        heads.append((s0 + (lo + hi) / 2.0) % 360.0)
        while k < len(arc) and arc[k] <= lo + fov:
            k += 1
    if len(heads) == 1:
        return heads

    def cost(seq_, d):
        first = min(_turn(yaw0, seq_[0], 1), _turn(yaw0, seq_[0], -1))
        return first + sum(_turn(a, b, d) for a, b in zip(seq_, seq_[1:]))
    return heads if cost(heads, 1) <= cost(heads[::-1], -1) else heads[::-1]


def seen_cells(depth, pose, fov, gm, stride: int = SEEN_STRIDE, rng: float = SEEN_RANGE) -> set:
    """Map cells where the agent's own depth rays END (a surface it actually saw: floor, counter, wall, shelf) within
    ``rng`` m -- never cells a ray merely passed over (Astra: passing above is not inspecting). Pure numpy."""
    import numpy as np
    if depth is None:
        return set()
    d = np.asarray(depth, dtype="float64")
    if d.ndim != 2:
        return set()
    H, W = d.shape
    ys, xs = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    dd = d[ys, xs]
    ok = np.isfinite(dd) & (dd > 0.05) & (dd <= rng)
    if not ok.any():
        return set()
    xs, ys, dd = xs[ok], ys[ok], dd[ok]
    f = (H / 2.0) / math.tan(math.radians(fov) / 2.0)
    xc, yc = ((xs + 0.5) - W / 2.0) / f * dd, -((ys + 0.5) - H / 2.0) / f * dd
    eye, r, u, fw = objmem._basis(pose)
    X = eye[0] + xc * r[0] + yc * u[0] + dd * fw[0]
    Z = eye[2] + xc * r[2] + yc * u[2] + dd * fw[2]
    gx = np.rint((X - gm.origin[0]) / gm.cell).astype(int)
    gz = np.rint((Z - gm.origin[1]) / gm.cell).astype(int)
    return set(zip(gx.tolist(), gz.tolist()))


def interest_cells(gm) -> set:
    """Cells where a target could be seen lying: the own map's free floor and its one-ring (counters, shelves,
    fixtures next to the walkable floor)."""
    out = set(gm.free)
    for c in gm.free:
        for dx, dz in hm.DIRS:
            out.add((c[0] + dx, c[1] + dz))
    return out


# ----------------------------------------------------------------------------------------------- the skills
# ---------------------------------------------------------------------------------------------------------------
# Round 2 / J-R1-executor (2026-10-13). General executor upgrades for ALL arms (docs/final-test/kitchen_memory/EXEC-DIAG.md):
#   E1 aim at what PROTRUDES from the support surface, not at the box centre, with pixel-diverse retries;
#   E2 a costed identity check of the object in the hand (specialist detector, observed image only) with a bounded swap-back;
#   E3 container-aware leads: an object seen inside an open container is a place to look again first;
#   P  purpose-based tool search through a declared affordance map (default OFF; recorded when used);
#   B  briefing-derived search-order hints, accepted ONLY by an executor built for a +package arm.
# Every number below is a declared prior or derived from the agent's own depth, boxes and pose; nothing is read from the simulator.
AIM_PROTRUDE_ON = True          # E1 switch (an ablation shows the legacy aim with it off)
AIM_THR_REL = 0.008             # a pixel nearer than the support plane by this share of its depth protrudes (~8 mm at 1 m; thinnest tool ~14 mm)
AIM_RING = 0.5                  # the support plane is fitted on a ring this many box sizes wide around the box
AIM_PAD = 6                     # px (on a 600 px frame): margin kept clear of every competing box
AIM_K = 6                       # protruding pixels proposed
AIM_SEP = 7                     # px (600 px frame): minimum separation of two proposed pixels
AIM_FRAME = 600.0               # the frame width the two pixel constants above are declared for
AIM_MIN_RING, AIM_MIN_PROT = 20, 4
AIM_SIGMA_K = 3.0               # the protrusion threshold is at least this many robust sigmas of the ring's scatter about its plane
AIM_SIGMA_MAX = 0.02            # a ring scattering more than this (relative depth) is not a plane: no proposal
INSIDE_R = 0.7                  # E3: m, a detection this near an open container's interior centre was inside it (own depth; declared ~ drawer / cabinet half-size)
AIM_PER_TRY = 3                 # pixels one attempt may send before any camera move (each refusal costs 2 cheap primitives)
#: SPEC-INHAND-apply (Arash 2026-10-13, "project several points if needed (if one point failed) to make sure you can pick up the
#: object"): EVERY pickup path aims at up to AIM_K object pixels and tries the next one before giving up, bounded and counted
#: (``HonestSkills.pick_stats``). (1) No protrusion found -> the legacy point first, then up to AIM_K depth-agreeing interior pixels
#: of the box (``_gate_points``), and a refused pixel goes on to the next one (it used to stop at the one legacy pixel). (2) A
#: pickup that is sent ONCE from one frame (the E3 look-alike candidate) may send up to AIM_K pixels in that one call (the camera
#: does not move between them, so the frame stays valid). The ordinary pickup / pick_and_place pick / identity swap retry keep
#: AIM_PER_TRY pixels per attempt and rotate to the NEXT pixels on each retry (``aim_round``), i.e. AIM_K pixels over 2 attempts.
PICK_MULTI_ON = True
#: surfaces and containers: a box of these types that overlaps an object's box is what it stands on / in, not something in front of it
SURFACE_TYPES = frozenset({"CounterTop", "DiningTable", "SideTable", "TableTop", "Floor", "Wall", "Window", "Drawer", "Cabinet",
                           "Fridge", "Microwave", "StoveBurner", "StoveKnob", "SinkBasin", "Sink", "Shelf", "ShelvingUnit", "Stool",
                           "Chair", "Sofa", "ArmChair", "Bed", "Desk", "Dresser", "TVStand", "Ottoman", "CoffeeTable", "GarbageCan",
                           "CoffeeMachine", "Toaster", "Safe", "Box", "Faucet", "Curtains", "Blinds", "LightSwitch", "Painting"})
#: E2: the specialist detector's classes (docs/ambiguity-spec.md): the only types whose identity in the hand is re-checked
AMBIG_TYPES = frozenset({"Fork", "Spoon", "ButterKnife", "Knife", "Spatula", "SaltShaker", "PepperShaker", "Plate", "Pot", "Pan"})
IDENT_ON = True                 # E2 switch
IDENT_CONF = 0.50               # a different type in the hand counts as a swap only at this confidence ...
IDENT_MARGIN = 0.15             # ... and when the asked type is not within this of it (false rejection costs a retry, so doubt keeps the pickup)
IDENT_DEPTH_BAND = 0.12         # m: only detections this close to the nearest one in the frame can be the object in the hand
IDENT_SWAPS_MAX = 2             # swap-backs one interaction may spend before the step fails honestly
#: SPEC-INHAND fix B (2026-10-13): when the in-hand look is not conclusive, turn the held object a quarter ROLL and look again
#: (then the other way), as every training held view of the specialist was rotated (docs/ambiguity-spec.md amendment 1). The
#: default grip shows a butter knife edge-on (5 px wide, specialist 0.06); a quarter roll shows its flat (0.83 / 0.93, live probe
#: sim/final_test/diag/spec_inhand/probe.jsonl). Only when the first look is unclear: a roll turns a spatula edge-on instead.
INSPECT_RELOOK_ON = True
INSPECT_ROLLS = (90.0, -180.0)  # relative RotateHeldObject rolls, in order (net +90 then -90); undone after the last look
#: P: a DECLARED generic affordance map (purpose -> tool types in preference order). Nothing here is a scene fact.
#: A26 item 2 (K-B2b 2026-10-13): the ONLY cutting tool. ``slice`` is refused (cause ``no_cutting_tool``, before any primitive,
#: 0 primitives, not retried within the step) unless the type this agent honestly holds -- its own pickup / setup record, after
#: the E2 identity check of a specialist-class type -- is in this set. Unknown ("?") or missing holding never authorizes a slice.
#: An experimenter-imposed realism convention, NOT an AI2-THOR constraint (ai2thor 5.0.0 slices with empty hands, K-B1).
CUTTING_TOOLS = frozenset({"ButterKnife"})
CUT_RULE_ON = True              # A26 switch (False = the old executor: slice with anything or nothing in hand; history only)
#: A26 item 1 / A8: detections of an excluded type (type_scope.EXCLUDED_TYPES = Knife) are dropped at the exposure boundary
#: (memory, search, look-alike candidates, observation summary). The raw frame stays available to the in-hand identity checks
#: only (A7: a specialist "Knife" on a held ButterKnife = identity not confirmed). False = unfiltered (history only).
TYPE_SCOPE_ON = True
PURPOSE_TYPES = {"cut": ("ButterKnife",), "slice": ("ButterKnife",), "flip": ("Spatula",),   # A26: was ("Knife", "ButterKnife")
                 "scoop": ("Spoon", "Ladle"), "stab": ("Fork",), "season_salt": ("SaltShaker",), "season_pepper": ("PepperShaker",)}


def protruding_points(depth, bbox, avoid=(), k: int = AIM_K, first=None) -> list:
    """Screen points ON the object inside ``bbox`` -- pixels nearer to the camera than the support surface around it --
    for a thin or partly covered object whose box centre lands on the surface (or the neighbour) behind it.

    The support is a plane in 1/z (``1/z = a x + b y + c``) fitted robustly on a ring around the box that excludes the
    box and every ``avoid`` box (padded); a box pixel protrudes when it is nearer than that plane by ``AIM_THR_REL`` of
    its depth. Returned nearest the centroid of the protruding set first, at least ``AIM_SEP`` px apart, at most ``k``.
    ``first`` (the legacy point) goes first when it itself protrudes, so a large object keeps its old aim. [] when the
    ring is too small, the plane does not fit or nothing protrudes (the caller then keeps its legacy points).
    Pure, own depth and boxes only."""
    import numpy as np
    if depth is None or bbox is None:
        return []
    d = np.asarray(depth, dtype="float32")
    if d.ndim != 2:
        return []
    h, w = d.shape
    sc = max(0.25, w / AIM_FRAME)
    pad, sep = int(round(AIM_PAD * sc)), max(3, int(round(AIM_SEP * sc)))
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2, y1, y2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1)), max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return []
    rx1, ry1 = max(0, int(x1 - AIM_RING * bw)), max(0, int(y1 - AIM_RING * bh))
    rx2, ry2 = min(w - 1, int(x2 + AIM_RING * bw)), min(h - 1, int(y2 + AIM_RING * bh))
    ys, xs = np.mgrid[ry1:ry2 + 1, rx1:rx2 + 1]
    z = d[ry1:ry2 + 1, rx1:rx2 + 1]
    inbox = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    inoth = np.zeros_like(inbox)
    for b in avoid:
        inoth |= (xs >= b[0] - pad) & (xs <= b[2] + pad) & (ys >= b[1] - pad) & (ys <= b[3] + pad)
    valid = np.isfinite(z) & (z > 0)
    ring = (~inbox) & (~inoth) & valid
    if int(ring.sum()) < AIM_MIN_RING:
        return []
    A = np.c_[xs[ring], ys[ring], np.ones(int(ring.sum()))].astype("float64")
    rhs = 1.0 / z[ring].astype("float64")
    keep = np.ones(len(rhs), bool)
    sol = None
    for _ in range(3):
        if int(keep.sum()) < AIM_MIN_RING:
            return []
        sol = np.linalg.lstsq(A[keep], rhs[keep], rcond=None)[0]
        res = np.abs(A @ sol - rhs)
        keep = res <= max(1e-4, 2.5 * float(np.median(res[keep])))
    zp = 1.0 / np.maximum(sol[0] * xs + sol[1] * ys + sol[2], 1e-6)
    prot = (zp - z) / zp
    # depth-uncertainty handling (Astra, round 2): the ring's own robust scatter about the fitted plane is the noise level;
    # a ring that does not fit a plane at all is no support surface, so nothing is proposed from it
    sig = 1.4826 * float(np.median(np.abs(prot[ring])[keep])) if int(keep.sum()) else 1.0
    if sig > AIM_SIGMA_MAX:
        return []
    thr = max(AIM_THR_REL, AIM_SIGMA_K * sig)
    cand = inbox & (~inoth) & valid & (prot > thr) & (z > 0.05 * zp)
    pts = np.argwhere(cand)
    if len(pts) < AIM_MIN_PROT:
        return []
    P = np.array([(int(xs[p[0], p[1]]), int(ys[p[0], p[1]])) for p in pts])
    cen = P.mean(0)
    order = np.argsort(((P - cen) ** 2).sum(1))
    out = []
    if first is not None:
        f = (int(first[0]), int(first[1]))
        if 0 <= f[1] - ry1 < cand.shape[0] and 0 <= f[0] - rx1 < cand.shape[1] and bool(cand[f[1] - ry1, f[0] - rx1]):
            out.append(f)                              # the legacy point is itself on the object: the large-object aim is unchanged
    for i in order:
        q = (int(P[i][0]), int(P[i][1]))
        if all(math.hypot(q[0] - o[0], q[1] - o[1]) >= sep for o in out):
            out.append(q)
        if len(out) >= k:
            break
    return out


def _raw_dets(obs):
    """A26 A7: the unfiltered boxes of a frame (``_observe`` keeps them as ``raw_dets``), else its ``dets``."""
    raw = getattr(obs, "raw_dets", None)
    return obs.dets if raw is None else raw


def identity_verdict(dets, asked: str, depth=None, near: float = objmem.HELD_NEAR, conf_min: float = IDENT_CONF,
                     margin: float = IDENT_MARGIN):
    """Is the object in the hand the ``asked`` type? From ONE observed frame's detections (the specialist runs while the
    agent holds something). Only detections whose point is nearer than ``near`` (the hand) vote; surfaces never do.
    Returns ``(verdict, seen_type, conf)``: ``ok`` (asked type is there, or nothing contradicts it), ``wrong`` (a DIFFERENT
    specialist-class type, confident, and the asked type not within ``margin`` of it) or ``unclear`` (nothing near the
    hand to judge -- the pickup is kept: doubt never rejects). Pure."""
    hand = []
    for d in dets or []:
        t = d.get("type")
        if not t or not d.get("bbox_xyxy") or t in SURFACE_TYPES:
            continue
        dd = hand_depth(depth, d["bbox_xyxy"]) if depth is not None else None    # SPEC-INHAND A: supported near depth
        if dd is None or dd >= near:
            continue
        hand.append((float(d.get("conf", 0.0)), t, dd))
    if not hand:
        return "unclear", None, 0.0
    # association with the HELD object (Astra, round 2): a thing merely near the camera is not the thing in the hand. Only the
    # detections at the nearest depth (within IDENT_DEPTH_BAND of it) can be the held object; the others do not vote.
    dmin = min(h_[2] for h_ in hand)
    hand = [h_ for h_ in hand if h_[2] <= dmin + IDENT_DEPTH_BAND]
    mine = max((c for c, t, _ in hand if t == asked), default=None)
    other = max(((c, t) for c, t, _ in hand if t != asked and t in AMBIG_TYPES), default=None)
    if other is None:
        return ("ok", asked, mine or 0.0) if mine is not None else ("unclear", None, 0.0)
    c_o, t_o = other
    if mine is not None and mine >= c_o - margin:
        return "ok", asked, mine
    if c_o >= conf_min:
        return "wrong", t_o, c_o
    return "unclear", t_o, c_o


# ------------------------------------------------------------------------------------------------------------------------------------
# E3 (2026-10-13): active disambiguation of LOOK-ALIKES, the DESSTA method of A. NasrEsfahani et al., "Embodied AI for Kitchen Scene
# Rearrangement", Intelligent Service Robotics 19:71 (2026), sec. 2.3.4 and Algorithm 4 (RevaluateObject). The paper: an Ambiguous
# Objects Map M built from the PRIMARY model's confusion matrix; an ambiguity point = a low-confidence detection, or a detection of a
# class that M lists as confusable, where a confident sighting of the expected object is missing; RevaluateObject = navigate to the
# point's frame, capture f, pick the object up, capture f', place it back, identity <- AmbiguousObjectDetection(M, f, f') with the
# specialist. Here the "expected object" is the type a pickup asks for. DECLARED ADAPTATIONS (DESSTA-inspired active re-evaluation
# for online pickup; not verbatim Algorithm 4; Astra E3 review): (a) the trigger is "no usable TRACKED sighting of the asked type
# in this frame" during a pickup's search; (b) f supplies the candidate, the pickup pixels and the restoration context, f' (the
# in-hand look) supplies the identity; (c) a candidate is KEPT only on positive specialist evidence (the pickup's postcondition
# is then met); every other candidate is restored -- bounded, landing observed or reported unverified, never dropped; (d) one
# FROZEN map M per kitchen (its own primary VAL confusion; FloorPlan1's for Kitchen 1; A11 / B4, 2026-10-17). The earlier E2 identity check
# covered only the held-object case (a box ALREADY labelled as the asked type); a look-alike box was never picked up for checking
# (Arash, 2026-10-13). All settings below are development settings (not validated optima), identical for every arm.
LOOKALIKE_ON = True             # E3 switch (False = the round-5 executor). Back ON 2026-10-13 after the min-rate / ceiling /
#                                 put-elsewhere fix (butter knife 10/10); NOTE the in-hand specialist look still read 'unclear' live
LOOKALIKE_MAX = 3               # re-evaluation calls that SENT a pickup, per pickup interaction (a cap, not a target)
LOOKALIKE_LOW = 0.10            # a box of the asked type at LOOKALIKE_LOW <= conf < the operating 0.25 is an ambiguity point
LOOKALIKE_KEEP = IDENT_CONF     # keep only when the specialist names the asked type at the hand with at least this confidence ...
LOOKALIKE_MARGIN = IDENT_MARGIN  # ... and ahead of every other specialist class at the hand by this much (positive evidence)
LOOKALIKE_MIN_RATE = 0.05      # a look-alike class needs a MEASURED confusion rate >= 5 % (a 2.5 % pair is not worth a pickup)
LOOKALIKE_CONF_CEIL = 0.75     # ... and its box must be UNCERTAIN (< 0.75; the paper's high-confidence level): a confident other
#                                class is that class (live 2026-10-13: a 0.96 Spatula picked up for a ButterKnife, 0/10)
LOOKALIKE_BORDER = 0.10         # the paper's hard filter: a box centre within 10 % of an image border is partly visible -> ignored
LOOKALIKE_MAX_DEPTH = 3.0       # m: a candidate needs a finite positive own depth at its point, at most this far (poorly resolved beyond)
LOOKALIKE_THETA = 0.02          # M: Y in M[X] iff the primary's VAL confusion (true X -> predicted Y) is >= 2 % of X's instances
LOOKALIKE_SAME = 0.10           # m: a point within this of one already checked in THIS pickup interaction is not checked again
LOOKALIKE_RESTORE_TRIES = 3     # PutObject attempts (each from a fresh frame) to restore a candidate that is not the asked type
LOOKALIKE_RESTORE_NEAR = 0.30   # m: a box of the candidate's type this near the old point after the put = "seen again near where it was"
LOOKALIKE_PUT_SURFACES = frozenset({"CounterTop", "DiningTable", "SideTable", "TableTop", "Shelf", "Desk", "CoffeeTable"})
LOOKALIKE_EXCLUDE = frozenset({"Knife"})   # Knife is being removed from the system (A26): never a target, never a candidate
#: the M source: the primary's VAL error analysis at the operating confidence (the evidence docs/ambiguity-spec.md chose its targets from)
AMBIG_MAP_FILE = "kitchen_v3/{scene}/val_errors_yolo26s.json"
#: the FROZEN map files, one per kitchen (Arash, packet decision 12: "know what objects may be mistaken for each other"): built
#: once from that kitchen's MEASURED primary confusions by ``freeze_ambiguous_map``, hash-pinned. Until a kitchen's own map is
#: registered the executor refuses to start (no silent FloorPlan1 fallback); LOOKALIKE_MAP_SCENE is only the default kitchen.
LOOKALIKE_MAP_FILE = "step5/lookalike_map_{kitchen}.json"
LOOKALIKE_MAP_SCENE = "FloorPlan1"


def ambiguous_objects_map(rows, theta: float = LOOKALIKE_THETA, targets=None, exclude=LOOKALIKE_EXCLUDE) -> dict:
    """DESSTA's Ambiguous Objects Map from the primary's confusion rows ``[{cls, n, confused_with: {pred: count}}]``:
    ``{X: {Y: rate}}`` with rate = count(true X predicted as Y) / n_X >= ``theta``. Direction matters: searching for X, a box
    LABELLED Y may be an X when the primary turns X into Y. Surfaces and excluded types never enter. Pure; from data only."""
    out = {}
    tg = set(AMBIG_TYPES if targets is None else targets) - set(exclude or ())
    for r in rows or []:
        x, n = r.get("cls"), float(r.get("n") or 0)
        if x not in tg or n <= 0:
            continue
        m = {y: round(c / n, 4) for y, c in (r.get("confused_with") or {}).items()
             if y != x and y not in SURFACE_TYPES and y not in (exclude or ()) and c / n >= theta}
        if m:
            out[x] = dict(sorted(m.items(), key=lambda kv: -kv[1]))
    return out


def load_ambiguous_map(scene: str, theta: float = LOOKALIKE_THETA) -> dict:
    """M for ``scene`` (``AMBIG_MAP_FILE``); {} when the kitchen has no confusion analysis (then only low-confidence boxes of the
    asked type are ambiguity points)."""
    import os
    stem = str(scene or "").replace("_physics", "")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), AMBIG_MAP_FILE.format(scene=stem))
    if not stem or not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return ambiguous_objects_map(json.load(fh).get("rows") or [], theta)


def freeze_ambiguous_map(path: Optional[str] = None, scene: str = LOOKALIKE_MAP_SCENE) -> dict:
    """Build the frozen development M from ``scene``'s source (``AMBIG_MAP_FILE``) and write it with its provenance (source
    file, its sha256, theta, exclusions, the raw counts and denominators). Run once; the file is then hash-pinned."""
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, AMBIG_MAP_FILE.format(scene=scene))
    data = open(src, "rb").read()
    doc = json.loads(data.decode("utf-8"))
    rows = doc.get("rows") or []
    m = ambiguous_objects_map(rows, LOOKALIKE_THETA)
    pairs = [{"true": x, "predicted_as": y, "count": int(next(r for r in rows if r["cls"] == x)["confused_with"][y]),
              "n_true": int(next(r for r in rows if r["cls"] == x)["n"]), "rate": rate}
             for x, ys in m.items() for y, rate in ys.items()]
    out = {"schema": "e3.lookalike_map/1", "kitchen": scene,
           "what": ("E3 Ambiguous Objects Map M (DESSTA): for each target type X, the types Y the PRIMARY detector predicts for "
                    "true X at the operating confidence, measured on this kitchen's evaluation (VAL) split; the agent picks up "
                    "a box labelled Y (and a low-confidence box of X) to re-check it with the specialist while searching for X"),
           "pairs": pairs,
           "source": AMBIG_MAP_FILE.format(scene=scene), "source_sha256": hashlib.sha256(data).hexdigest(),
           "source_split": doc.get("split"), "source_weights": doc.get("weights"), "source_op_conf": doc.get("op_conf"),
           "theta": LOOKALIKE_THETA, "exclude": sorted(LOOKALIKE_EXCLUDE),
           "direction": "M[X][Y] = count(true X predicted as Y) / n_X (searching for X, a box labelled Y may be an X)",
           "counts": {r["cls"]: {"n": r.get("n"), "confused_with": r.get("confused_with")} for r in rows if r.get("cls") in m},
           "map": m}
    path = path or os.path.join(here, LOOKALIKE_MAP_FILE.format(kitchen=scene))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


def load_frozen_map(path: Optional[str] = None, kitchen: str = LOOKALIKE_MAP_SCENE) -> dict:
    """The frozen M of ``kitchen`` (``LOOKALIKE_MAP_FILE``). A missing or malformed file is a configuration error (raised), never
    a silent ``{}`` (Astra E3 review: a silently empty map would be a different executor)."""
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), LOOKALIKE_MAP_FILE.format(kitchen=kitchen))
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    m = doc.get("map")
    if not isinstance(m, dict) or not m:
        raise RuntimeError(f"E3: the frozen look-alike map {path} has no map")
    if doc.get("kitchen") != kitchen:       # K23 (Astra K23-AGENT-r1 1): another kitchen's map is refused, never used
        raise RuntimeError(f"E3: the frozen look-alike map {path} is for kitchen {doc.get('kitchen')!r}, not {kitchen!r}")
    return {k: dict(v) for k, v in m.items()}


#: K23 (Astra K23-AGENT-r1 2; FP1-EQUIV-r1 e): a SEPARATELY hashed provenance supplement per kitchen that records ALL nine target rows
#: (AMBIG_TYPES minus the exclusion), including targets with no confusion at or above theta, from the same frozen VAL evidence as the
#: map. It never changes the map: the loader recomputes M from the supplement's rows and requires it to equal the frozen map exactly.
LOOKALIKE_PROVENANCE_FILE = "step5/lookalike_provenance_{kitchen}.json"
LOOKALIKE_PROVENANCE_SCHEMA = "e3.lookalike_provenance/1"


def _canon_sha(doc: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "sha256"}, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def build_lookalike_provenance(kitchen: str, source_path: str, map_path: str) -> dict:
    """The nine-target supplement of ``kitchen`` from its frozen VAL evidence (``source_path``) and its frozen map (``map_path``).
    Per target: ``in_map`` (M has a row), ``below_theta`` (VAL evidence with n > 0 but no non-surface, non-excluded confusion at or
    above theta: zero confusion for the map), ``absent_evidence`` (no VAL row, or n = 0: unknown, NOT zero). Pure apart from reading."""
    import hashlib
    src = open(source_path, "rb").read()
    mp = open(map_path, "rb").read()
    sdoc, mdoc = json.loads(src.decode("utf-8")), json.loads(mp.decode("utf-8"))
    if mdoc.get("kitchen") != kitchen or mdoc.get("source_sha256") != hashlib.sha256(src).hexdigest():
        raise RuntimeError(f"E3 provenance: {map_path} is not {kitchen}'s map of {source_path}")
    rows = {r.get("cls"): r for r in sdoc.get("rows") or []}
    targets = sorted(set(AMBIG_TYPES) - set(LOOKALIKE_EXCLUDE))
    mm = mdoc.get("map") or {}
    tgt = {}
    for t in targets:
        r = rows.get(t)
        n = int((r or {}).get("n") or 0)
        status = "absent_evidence" if (r is None or n <= 0) else ("in_map" if t in mm else "below_theta")
        tgt[t] = {"status": status, "n": n if r is not None else None, "confused_with": dict((r or {}).get("confused_with") or {}),
                  "map_row": dict(mm.get(t) or {})}
    doc = {"schema": LOOKALIKE_PROVENANCE_SCHEMA, "kitchen": kitchen, "targets": tgt, "n_targets": len(targets),
           "map_file": LOOKALIKE_MAP_FILE.format(kitchen=kitchen), "map_sha256": hashlib.sha256(mp).hexdigest(),
           "source": mdoc.get("source"), "source_sha256": mdoc.get("source_sha256"), "source_split": sdoc.get("split"),
           "theta": mdoc.get("theta"), "exclude": sorted(LOOKALIKE_EXCLUDE), "surface_types": sorted(SURFACE_TYPES),
           "what": ("supplement to the frozen look-alike map: every target row of the frozen VAL evidence, so absent evidence is not "
                    "read as zero confusion; the map itself is unchanged and is recomputed from these rows at load")}
    doc["sha256"] = _canon_sha(doc)
    return doc


def load_lookalike_provenance(kitchen: str, amap: dict, path: Optional[str] = None) -> dict:
    """Require the kitchen's supplement: intact (self sha), this kitchen, all nine targets with a valid status, bound to the frozen
    map file's sha, and M recomputed from its rows (``ambiguous_objects_map``) EQUAL to the loaded map. Any problem is raised."""
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(here, LOOKALIKE_PROVENANCE_FILE.format(kitchen=kitchen))
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    want = sorted(set(AMBIG_TYPES) - set(LOOKALIKE_EXCLUDE))
    probs = []
    if doc.get("schema") != LOOKALIKE_PROVENANCE_SCHEMA or doc.get("sha256") != _canon_sha(doc):
        probs.append("schema / self sha")
    if doc.get("kitchen") != kitchen:
        probs.append(f"kitchen {doc.get('kitchen')!r}")
    tg = doc.get("targets") or {}
    if sorted(tg) != want or any((v or {}).get("status") not in ("in_map", "below_theta", "absent_evidence") for v in tg.values()):
        probs.append(f"targets {sorted(tg)} != the nine {want}, or a status is invalid")
    mpath = os.path.join(here, *str(doc.get("map_file") or "").split("/"))
    if not os.path.isfile(mpath) or hashlib.sha256(open(mpath, "rb").read()).hexdigest() != doc.get("map_sha256"):
        probs.append("not bound to the frozen map file on disk")
    rows = [{"cls": t, "n": v.get("n") or 0, "confused_with": v.get("confused_with") or {}} for t, v in tg.items()]
    if ambiguous_objects_map(rows, float(doc.get("theta") or LOOKALIKE_THETA)) != {k: dict(v) for k, v in (amap or {}).items()}:
        probs.append("M recomputed from the supplement rows differs from the frozen map")
    if probs:
        raise RuntimeError(f"E3 provenance {path}: " + "; ".join(probs))
    return doc


def lookalike_candidates(obs, asked: str, amap: dict, low_min: float = LOOKALIKE_LOW, border: float = LOOKALIKE_BORDER,
                         op_conf: float = 0.25, max_depth: float = LOOKALIKE_MAX_DEPTH, min_rate: float = LOOKALIKE_MIN_RATE,
                         conf_ceil: float = LOOKALIKE_CONF_CEIL) -> list:
    """DESSTA ambiguity points in ONE frame for a pickup of ``asked``: boxes of a class M lists as confusable with it (at the
    operating confidence, ``obs.dets``), and boxes of ``asked`` itself with ``low_min <= conf < op_conf`` (``obs.low``). Hard
    filters: the image must have valid dimensions; a box centre within ``border`` of an image edge is dropped (centres near an
    edge, not every clipped box); a box needs a grounded point with a finite positive own depth <= ``max_depth``. Order: the
    asked type's own low-confidence boxes (highest first), then look-alikes by confusion rate, then confidence. Overlapping
    boxes (IoU >= 0.5) count once (candidate prioritisation, not identity). Each candidate keeps a reference to the detection
    it came from (``_source_det``: aiming excludes exactly that box). Pure: own detections only."""
    m = (amap or {}).get(asked) or {}
    dep = getattr(obs, "depth", None)
    if dep is None or getattr(dep, "ndim", 0) != 2 or dep.shape[0] < 2 or dep.shape[1] < 2:
        return []
    wh = (dep.shape[1], dep.shape[0])
    pool = [(0, -float(d.get("conf", 0.0)), d) for d in getattr(obs, "low", None) or []
            if d.get("type") == asked and low_min <= float(d.get("conf", 0.0)) < op_conf]
    pool += [(1, -m[d["type"]], -float(d.get("conf", 0.0)), d) for d in obs.dets or []
             if d.get("type") in m and m[d["type"]] >= min_rate and op_conf <= float(d.get("conf", 0.0)) < conf_ceil]
    out = []
    for e in sorted(pool, key=lambda e: e[:-1]):
        d = e[-1]
        b = d.get("bbox_xyxy")
        if not b or not d.get("point") or not d.get("point_ok"):
            continue
        cx, cy = (b[0] + b[2]) / 2.0 / wh[0], (b[1] + b[3]) / 2.0 / wh[1]
        if not (border <= cx <= 1 - border and border <= cy <= 1 - border):
            continue
        z = _depth(dep, d["point"])
        if z is None or not math.isfinite(z) or z <= 0 or z > max_depth:
            continue
        if any(_iou(b, o["bbox_xyxy"]) >= 0.5 for o in out):
            continue
        out.append({**d, "_source_det": d, "trigger": "low_conf" if e[0] == 0 else "lookalike", "m_rate": m.get(d.get("type"))})
    return out


def lookalike_confirm(dets, asked: str, depth=None, keep: float = LOOKALIKE_KEEP, margin: float = LOOKALIKE_MARGIN,
                      near: float = objmem.HELD_NEAR):
    """E3 POSITIVE confirmation that the object in the hand is ``asked`` (stricter than E2's ``identity_verdict``, where doubt
    keeps a pickup the primary already called ``asked``; here a DIFFERENTLY labelled candidate is promoted, so evidence must be
    positive). Only SPECIALIST-sourced detections at the hand vote (point nearer than ``near``, within IDENT_DEPTH_BAND of the
    nearest one); surfaces never. Returns ``(keep, seen, conf, rival)``: keep iff ``asked`` is there at >= ``keep`` and ahead
    of every other specialist-class type at the hand by >= ``margin``; ``seen`` = the asked type when kept, else the strongest
    type at the hand (None if nothing); ``rival`` = (conf, type) of the strongest other class or None. Pure."""
    hand = []
    for d in dets or []:
        t = d.get("type")
        if d.get("source") != "specialist" or not t or not d.get("bbox_xyxy") or t in SURFACE_TYPES:
            continue
        dd = hand_depth(depth, d["bbox_xyxy"]) if depth is not None else None    # SPEC-INHAND A: supported near depth
        if dd is None or dd >= near:
            continue
        hand.append((float(d.get("conf", 0.0)), t, dd))
    if not hand:
        return False, None, 0.0, None
    dmin = min(h_[2] for h_ in hand)
    hand = [h_ for h_ in hand if h_[2] <= dmin + IDENT_DEPTH_BAND]
    mine = max((c for c, t, _ in hand if t == asked), default=None)
    other = max(((c, t) for c, t, _ in hand if t != asked and t in AMBIG_TYPES), default=None)
    ok = mine is not None and mine >= keep and (other is None or mine >= other[0] + margin)
    if ok:
        return True, asked, mine, other
    top = max(hand)
    return False, top[1], top[0], other


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


class HonestSkills:
    """``exec_step(step) -> {ok, cause, retries, detail}`` over an honest agent, its map and its object memory."""

    def __init__(self, hagent, gridmap, memory, chooser=None, type_list=None, max_retries: int = MAX_RETRIES,
                 prim_cap: int = STEP_PRIM_CAP, log: Optional[Callable[[str], None]] = None,
                 search: str = "bounded", tidy: bool = False, search_cap: int = SEARCH_PRIM_CAP,
                 pose_first: Optional[bool] = None, on_event: Optional[Callable] = None,
                 place_hints: Optional[Callable] = None, hint_boost: Optional[float] = None,
                 hint_goals: bool = False, aim_protrude: Optional[bool] = None, identity_check: Optional[bool] = None,
                 purpose_search: bool = False, briefing_hints: Optional[Callable] = None,
                 allow_briefing_hints: bool = False, lookalike: Optional[bool] = None,
                 ambiguous_map: Optional[dict] = None, kitchen: str = LOOKALIKE_MAP_SCENE):
        if search not in SEARCH_MODES:
            raise ValueError(f"search must be one of {SEARCH_MODES}, got {search!r}")
        self.search_mode, self.tidy, self.search_cap = search, bool(tidy), int(search_cap)
        self.exhaustive = search == "exhaustive"
        #: pose first, then open (2026-10-08): the console policy (exhaustive); the bounded s5 runner keeps its flow
        self.pose_first = self.exhaustive if pose_first is None else bool(pose_first)
        #: own door records (geometry only): [{type, c, model}] -- the closed front and the sweep seen when it opened
        self.doors = []
        self.h, self.gm, self.om = hagent, gridmap, memory
        self.chooser = chooser or prior_chooser
        #: Step 5 / B1 (both default None = no call, code path unchanged): ``on_event(kind, fields)`` gets one call per
        #: open / close / inspection / search start + end with enumerated own-measurement fields only;
        #: ``place_hints(otype) -> [{rtype, xyz, radius?}]`` biases the search ROUTE toward remembered places.
        self.on_event, self.place_hints = on_event, place_hints
        #: B2: weight multiplier of a plan item at a hinted place; None = HINT_BOOST (sensitivity analysis only; no effect without hints)
        self.hint_boost = None if hint_boost is None else float(hint_boost)
        #: B2: with hints on, a hinted place that no known container / place item covers becomes a LOOK goal of its own (the
        #: agent walks there, faces the remembered point and looks; containers it then sees are inspected). Navigation only.
        self.hint_goals = bool(hint_goals)
        #: Round 2 / J-R1 (general executor upgrades, ALL arms, module defaults ON): E1 aim at what protrudes from the support
        #: surface; E2 a costed identity check of the object in the hand after a pickup of a specialist-class type.
        self.aim_protrude = AIM_PROTRUDE_ON if aim_protrude is None else bool(aim_protrude)
        self.identity_check = IDENT_ON if identity_check is None else bool(identity_check)
        #: P (default OFF): ``exec_step`` of a step carrying ``purpose`` may take another type of that purpose's tool family
        #: when the asked type is not found; the substitution is recorded in the step detail. Whether a goal accepts it is the
        #: caller's business -- this flag only makes the search purpose-aware.
        self.purpose_search = bool(purpose_search)
        #: B: briefing-derived search-order hints, ``briefing_hints(otype) -> [container type, ...]``. ONLY an executor built for a
        #: +package arm may receive them (``allow_briefing_hints=True``); a no-memory arm that is handed any is refused loudly.
        if briefing_hints is not None and not allow_briefing_hints:
            raise ValueError("briefing_hints are only for +package arms (allow_briefing_hints=True); a no-memory arm must not get them")
        self.briefing_hints = briefing_hints if allow_briefing_hints else None
        self.allow_briefing_hints = bool(allow_briefing_hints)
        self.briefing_log = []                         # per search: {otype, hint, used} (audit of where the briefing entered)
        self.identity_stats = {"checks": 0, "detector_calls": 0, "wrong": 0, "unclear": 0, "swaps": 0, "extra_prims": 0}
        #: E3 (DESSTA RevaluateObject, see the module constants): ALL arms, module default LOOKALIKE_ON; the ONE frozen
        #: development map M in every kitchen (``ambiguous_map`` overrides it only in the offline gates)
        self.lookalike = LOOKALIKE_ON if lookalike is None else bool(lookalike)
        #: K23 B4 (A11; Astra K23-AGENT-r1): the frozen look-alike map of THIS kitchen (FloorPlan1 by default = unchanged K1)
        self.kitchen = kitchen
        self.amb_map = dict(ambiguous_map) if ambiguous_map is not None else (load_frozen_map(kitchen=kitchen) if self.lookalike else {})
        if ambiguous_map is None and self.lookalike:                  # K23: the kitchen's nine-target provenance supplement is required
            self.amb_provenance = load_lookalike_provenance(kitchen, self.amb_map)       # and must reproduce the frozen map exactly
        if self.lookalike:
            self.identity_stats.update(la_checks=0, la_kept=0, la_put_back=0, la_put_seen=0, la_put_unseen=0,
                                       la_restore_failed=0, la_put_elsewhere=0, la_dropped=0, la_refused=0, la_unreached=0, la_prims=0, la_detector_calls=0,
                                       la_revisits=0, la_revisit_prims=0, la_abandoned=0)
        self._la = None                                # the active look-alike context {otype, n, pending, checked} (one pickup)
        self._la_obs = None                            # the look taken after a restoration (reused as the fresh frame)
        self.identity_log = []
        self._step_purpose = None
        self._aim_next = False
        self._aim_cap = None                           # SPEC-INHAND-apply: per-call pickup pixel cap (None = the default)
        #: SPEC-INHAND-apply: pickup aim counters (every PickupObject sent through _act_at): calls, pixels sent, NEXT-pixel tries
        #: after a refused / missed pixel in the same call, successes, the largest pixel list offered, fallback aims (no protrusion)
        self.pick_stats = {"calls": 0, "sends": 0, "next_point": 0, "ok": 0, "offered_max": 0, "fallback_aims": 0}
        self.event_errors = []
        self._srch = None                              # B1: the active search {otype, first} (flag on only)
        self.type_list = sorted(type_list or detector_types(getattr(hagent, "detector", None)))
        if TYPE_SCOPE_ON:
            self.type_list = type_scope.filter_types(self.type_list)   # A26 item 1: the agent never searches for / ranks a Knife
        self.cut_refusals = 0                          # A26 item 2: slices refused for want of a cutting tool (episode-scoped)
        self.max_retries, self.prim_cap = int(max_retries), int(prim_cap)
        self._log = log
        self._vps = None
        #: own action history: container type -> the xyz this agent aimed at when IT opened it (episode-scoped).
        #: A swung door is detected as the container too and gets its own, nearer track; aiming a later put/close
        #: at that door track leaves the interior out of view (s4 live smoke). The anchor keeps the body.
        self.opened_at = {}
        #: B1: own record of each object THIS agent put down {type: {p0, half, depth, pose, rec, step}} (own depth + pose)
        self.put_rec = {}
        self.toggled_on_at = {}                  # B1b: type -> world point this agent switched it ON at (own depth + pose)
        self.put_reject = []                     # B2: world (x, z, r) spots where a put was judged wrong
        self.put_in_reject = []                  # B3: world (x, y, z, r) landing spots of a put INTO a container judged wrong
        self.put_bad_recs = []                   # B3: (receptacle type, held type, x, z) instances that put it outside
        self.put_in_caps = []                    # B3 r2: (x, z, y) a pixel at this height of this front's opening landed outside
        self.put_refused_recs = []               # B4: (receptacle type, x, y, z, radius) instances that refused PUT_REFUSE_MAX puts
        self.put_in_landed = []                  # B3 r2: world points where the held object was seen after a wrong landing (then picked up)
        #: own action history, episode-scoped: every container THIS agent opened and has not closed again
        #: [{type, xyz, why: search|search-found|look|put|composite|explicit, step}] (tidiness)
        self.opened_by_me = []
        #: beliefs set aside this episode: track -> step (not re-seen at the remembered place; usable again once the
        #: track is seen after that step). Memory evidence is kept -- this is a search decision, not a contradiction.
        self.aside = {}
        #: own action history, episode-scoped: every container this agent inspected in a search -- its closed front
        #: (3D points from its own depth, per view), the cells it looked from, opens, and the honest inside-visible
        #: verdict (visible | partly | found | unsure). Geometry only; never a simulator handle (Astra, 2026-10-07).
        self.inspected = []
        self._crouched = False                 # own stance (from its own Crouch / Stand commands)
        #: reach-gate decisions (own depth only): [{prim, type, dist, sent, ok|why}] -- for events / measurement
        self.gate_log = []
        self._last_look = None
        self._pre_open = None
        self.tidy_log = []
        self.upcoming = []
        self._step_action = None
        #: systematic search route (2026-10-08; exhaustive / console only): side / back steps on short walks, the own
        #: front registry, where it looked around, the cells its own depth has seen, the poses it looked from
        self.strafe = STRAFE_MAX if self.exhaustive else 0.0
        self._route_reset()
        self._reset_step()

    # -------------------------------------------------------------- bookkeeping
    def _reset_step(self):
        self.left = self.max_retries
        self.retries = 0
        self.events = []
        self.sub = []
        self.verify = []
        self.search_info = None
        self._step0 = self.h.step
        self._limit = self.prim_cap + (self.search_cap if self.exhaustive else 0)
        self.tidy_log = []

    def _ev(self, msg):
        self.events.append(f"[{self.h.step}] {msg}")
        if self._log:
            self._log(msg)

    def _emit(self, kind, **fields):
        """B1 hook: a failing consumer never breaks the agent; the failure is recorded for the caller to check."""
        if self.on_event is None:
            return
        try:
            self.on_event(kind, fields)
        except Exception as e:                         # noqa: BLE001 - recorded, checked by the executor
            self.event_errors.append({"kind": kind, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    def _hint_list(self, otype) -> list:
        if self.place_hints is None:
            return []
        try:
            return list(self.place_hints(otype) or [])
        except Exception as e:                         # noqa: BLE001
            self.event_errors.append({"kind": "place_hints", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            return []

    def _hint_boost(self, otype, ctype, c) -> float:
        """HINT_BOOST for a plan item near a hinted place of its type, else 1.0 (always 1.0 without hints)."""
        for h in self._hint_list(otype):
            if h.get("rtype") == ctype and h.get("xyz") and c is not None \
                    and math.dist(tuple(h["xyz"]), tuple(c)) <= float(h.get("radius") or HINT_RADIUS):
                return HINT_BOOST if self.hint_boost is None else self.hint_boost
        return 1.0

    def _hint_sort(self, otype, place, beliefs) -> list:
        """Bounded policy: beliefs of ``place`` near a hint first (stable, so the old order is kept otherwise)."""
        hs_ = [h for h in self._hint_list(otype) if h.get("rtype") == place and h.get("xyz")]
        if not hs_:
            return beliefs
        near = lambda b: any(math.dist(tuple(h["xyz"]), tuple(b["xyz"])) <= float(h.get("radius") or HINT_RADIUS)  # noqa: E731
                             for h in hs_)
        return sorted(beliefs, key=lambda b: 0 if near(b) else 1)

    def _cap(self):
        if self.h.step - self._step0 > self._limit:
            raise _Fail("retry_exhausted", f"primitive cap {self._limit} reached", under="step_prim_cap")

    def _spend(self, e: _Soft):
        self._cap()
        if self.left <= 0:
            raise _Fail("retry_exhausted", f"{e.cause}: {e.why}", under=e.cause)
        self.left -= 1
        self.retries += 1
        self._ev(f"retry {self.retries}/{self.max_retries} after {e.cause} ({e.why})")

    # -------------------------------------------------------------- episode
    def start_episode(self, episode, holding: Optional[str] = None, sweep: bool = True) -> dict:
        """New episode: fresh agent history/audit and memory; a 360-degree look-around seeds the memory."""
        self.h.reset_episode(episode, holding=holding)
        self.om.new_episode(episode)
        self.opened_at = {}
        self.put_rec = {}
        self.toggled_on_at = {}                  # B1b: type -> world point this agent switched it ON at (own depth + pose)
        self.put_reject = []                     # B2: world (x, z, r) spots where a put was judged wrong
        self.put_in_reject = []                  # B3: world (x, y, z, r) landing spots of a put INTO a container judged wrong
        self.put_bad_recs = []                   # B3: (receptacle type, held type, x, z) instances that put it outside
        self.put_in_caps = []                    # B3 r2: (x, z, y) a pixel at this height of this front's opening landed outside
        self.put_refused_recs = []               # B4: (receptacle type, x, y, z, radius) instances that refused PUT_REFUSE_MAX puts
        self.put_in_landed = []                  # B3 r2: world points where the held object was seen after a wrong landing (then picked up)
        self.opened_by_me = []
        self.aside = {}
        self.inspected = []
        self.gate_log = []
        self.identity_stats = {k: 0 for k in self.identity_stats}
        self.pick_stats = {k: 0 for k in self.pick_stats}
        self.identity_log = []
        self.cut_refusals = 0
        self._la, self._la_obs = None, None
        self.briefing_log = []
        self._route_reset()
        for k in list(getattr(self.gm, "temp_blocked", {}) or {}):
            self.gm.unblock_cells(k)                   # a new episode: every door is back where the world put it
        if hasattr(self.gm, "drop_here"):
            self.gm.drop_here()
        self._reset_step()
        if self.exhaustive:
            self._adopt_here()
            self._start_ev = list(self.events)         # shown again at the start of the first step's events
        if sweep:
            self._sweep()
        return {"tracks": len(self.om.tracks), "summary": self.om.summary()}

    def _adopt_here(self):
        """C (2026-10-09): the agent stands on a cell its own map does not have as free (it was placed there): its own
        pose proves that cell for this episode; recorded-blocked edges from it may be re-tried once each (logged).
        Off the lattice: not adopted (never silently snapped), said so."""
        if not hasattr(self.gm, "adopt_here"):
            return
        p = self.h.pose()
        x, z = float(p["position"]["x"]), float(p["position"]["z"])
        cur = self.gm.cell_of(x, z)
        res = self.gm.residual(x, z)
        if res > hm.POS_TOL:
            # off the lattice (live 2026-10-09: the oracle setup left the agent 0.08-0.13 m between cells): never
            # snapped silently -- an explicit correction move onto the nearest FREE cell centre within one cell
            # diagonal (own pose + own map), axis by axis after aligning the heading; said so; refused -> said so
            near = sorted((math.hypot(self.gm.world(c)[0] - x, self.gm.world(c)[1] - z), c)
                          for c in [(cur[0] + i, cur[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
                          if c in self.gm.free)
            near = [t for t in near if t[0] <= self.gm.cell * math.sqrt(2) + 1e-6]
            if not near:
                self._ev(f"own pose ({x:.2f},{z:.2f}) is {res:.2f} m off the lattice with no free own-map cell "
                         "within one cell diagonal - not moved")
                return
            d0, tgt = near[0]
            ok = True
            try:
                self._stand()
                k = hm.align_heading(self.h)
                tx, tz = self.gm.world(tgt)
                for ax, dv in (("x", tx - x), ("z", tz - z)):
                    p2 = self.h.pose()
                    dv = (tx - float(p2["position"]["x"])) if ax == "x" else (tz - float(p2["position"]["z"]))
                    if abs(dv) <= hm.POS_TOL:
                        continue
                    d = (1 if dv > 0 else 3) if ax == "x" else (0 if dv > 0 else 2)     # DIRS index
                    ok = self.h.move(hm.REL_MOVES[(d - k) % 4], moveMagnitude=round(abs(dv), 4)) and ok
            except hm.MapAbort as e:
                ok = False
                self._ev(f"lattice correction: heading alignment failed ({e})")
            p = self.h.pose()
            x, z = float(p["position"]["x"]), float(p["position"]["z"])
            cur, res2 = self.gm.cell_of(x, z), self.gm.residual(x, z)
            self._ev(f"own pose was {res:.2f} m off the lattice: moved {d0:.2f} m onto the nearest free own-map cell "
                     f"{tuple(tgt)} ({'ok' if ok and res2 <= hm.POS_TOL else f'not fully: {res2:.2f} m off'})")
            if res2 > hm.POS_TOL:
                return
        if cur in self.gm.free:
            return
        es = self.gm.adopt_here(cur)
        self._ev(f"own cell {tuple(cur)} is not on the own map: standing here proves the pose (episode only); "
                 f"{len(es)} recorded-blocked edge(s) from it may be re-tried once"
                 + (f" ({', '.join(str(tuple(b if a == cur else a)) for a, b in es)})" if es else ""))
        self._here_adopted = cur

    def look_around(self) -> dict:
        """Public 360-degree look-around in place (the console's "Look around" button). Memory is kept."""
        self._reset_step()
        self._sweep()
        return {"tracks": len(self.om.tracks), "summary": self.om.summary(), "prims": self.h.step - self._step0}

    # -------------------------------------------------------------- perception / motion helpers
    def _observe(self):
        """The ONE observation path of the skills (A26 item 1 / A8): the gateway's frame with excluded-type boxes dropped from
        ``dets`` and ``low`` (memory, search, look-alike candidates and summaries never see a Knife). The unfiltered boxes stay
        on the frame as ``raw_dets`` for the in-hand identity checks only (A7)."""
        obs = self.h.observe()
        if TYPE_SCOPE_ON:
            obs.raw_dets = list(obs.dets)
            obs.dets = [d for d in obs.dets if type_scope.in_scope(d.get("type"))]
            obs.low = [d for d in (obs.low or []) if type_scope.in_scope(d.get("type"))]
        return obs

    def _cutting_tool_check(self, ot):
        """A26 item 2: refuse ``slice`` unless the honestly held type is a cutting tool. Called before ANY primitive."""
        if not CUT_RULE_ON:
            return
        held = self.h.holding
        if held in CUTTING_TOOLS:
            return
        self.cut_refusals += 1
        self._ev(f"slice {ot}: refused - no cutting tool in hand (holding {held or 'nothing'}; the ButterKnife is needed)")
        raise _Fail("no_cutting_tool", f"slice needs a ButterKnife in hand; holding {held or 'nothing'}")

    def _look(self):
        obs = self._observe()
        self.om.update(obs)
        self._last_look = obs.step
        if obs.depth is not None:
            self._last_wh = _wh(obs.depth)
        if self.exhaustive:
            self._note_view(obs)
            self._note_foreground(obs)
        return obs

    def _eye_stand(self) -> float:
        """Own standing eye height (the camera height now, plus the crouch drop if crouched)."""
        return self._cam_y() + (CROUCH_DROP if self._crouched else 0.0)

    def _actable(self, xyz) -> bool:
        """Can ANY own-map cell bring the eye within the act reach of ``xyz`` (either stance)? Geometry only."""
        return xyz is not None and reach_from_map(self.gm, xyz, self._eye_stand()) <= ACT_REACH

    def _note_foreground(self, obs):
        """B (2026-10-09): a detection whose grounded estimate no own-map cell can act on -> investigate its box: a
        coherent near depth component well in front of the box median becomes an episode-local foreground hypothesis
        (frame + pixel provenance). The estimate stays in memory; the hypothesis is dropped once the track's own
        estimate is actable again. Not a diagnosis: the reach gate, success bool and state re-check still decide."""
        if obs.depth is None:
            return
        wh = _wh(obs.depth)
        eye = self._eye_stand()
        cache = self.__dict__.setdefault("_act_cache", {})
        for d in obs.dets:
            tid = d.get("track")
            t = self.om.tracks.get(tid) if tid else None
            if t is None or not t.xyz or not d.get("bbox_xyxy"):
                continue
            key = (tid, tuple(round(v, 2) for v in t.xyz))
            if key not in cache:
                if len(cache) > 5000:
                    cache.clear()
                cache[key] = reach_from_map(self.gm, t.xyz, eye) <= ACT_REACH
            if cache[key]:
                self.fg_hyp.pop(tid, None)
                continue
            pts = foreground_points(obs.depth, d["bbox_xyxy"])
            xyz = None
            for q in pts:
                dq = _depth(obs.depth, q)
                xyz = objmem.backproject(q[0], q[1], dq, obs.pose, wh, self.om.fov) if dq else None
                if xyz:
                    break
            if not xyz or reach_from_map(self.gm, xyz, eye) > ACT_REACH:
                continue
            first = tid not in self.fg_hyp
            self.fg_hyp[tid] = {"xyz": tuple(xyz), "step": obs.step, "px": tuple(pts[0]),
                                "est_d": round(math.dist(objmem._basis(obs.pose)[0], t.xyz), 2)}
            if first:
                self._ev(f"{d['type']}: its estimate ({self.fg_hyp[tid]['est_d']:.1f} m from the eye) is not "
                         f"actable from any own-map cell - foreground candidate selected at {tuple(pts[0])} "
                         f"({_depth(obs.depth, pts[0]):.2f} m; the box median is suspected background)")

    def _hyp(self, b):
        """A belief with its episode-local foreground hypothesis applied (B), else unchanged."""
        if not b or not self.exhaustive:
            return b
        hy = self.fg_hyp.get(b.get("track"))
        if hy is None:
            return b
        return {**b, "xyz": hy["xyz"], "regrounded": hy["step"]}

    def _in_view_now(self, xyz, margin: float = 0.15) -> bool:
        """Own pose + the last frame size: does ``xyz`` project well inside the image from where it looks now?"""
        wh = getattr(self, "_last_wh", None)
        if wh is None:
            return False
        pr = objmem.project(xyz, self.h.pose(), wh, self.om.fov)
        return (pr is not None and margin * wh[0] <= pr[0] <= (1 - margin) * wh[0]
                and margin * wh[1] <= pr[1] <= (1 - margin) * wh[1])

    def _pose_xz(self):
        p = self.h.pose()
        return float(p["position"]["x"]), float(p["position"]["z"])

    def _hdist(self, xyz):
        x, z = self._pose_xz()
        return math.hypot(xyz[0] - x, xyz[2] - z)

    def _set_horizon(self, target):
        target = max(PITCH_MIN, min(PITCH_MAX, float(target)))
        cur = float(self.h.pose()["horizon"] or 0.0)
        d = round(target - cur, 1)
        if abs(d) >= 0.5:
            self.h.move("LookDown" if d > 0 else "LookUp", degrees=abs(d))

    def _face(self, xyz, dpitch: float = 0.0, dyaw: float = 0.0):
        hm.face_point(self.h, xyz)
        if abs(dyaw) >= 0.5:
            self.h.move("RotateRight" if dyaw > 0 else "RotateLeft", degrees=abs(dyaw))
        p = self.h.pose()
        cam = p.get("camera") or {}
        cy = float(cam.get("y", float(p["position"]["y"]) + 0.675))
        hd = max(0.05, self._hdist(xyz))
        self._set_horizon(math.degrees(math.atan2(cy - xyz[1], hd)) + dpitch)

    def _sweep(self, otype: Optional[str] = None):
        """Look around in place (8 views). Returns the belief for ``otype`` as soon as one is sighted."""
        self._set_horizon(SWEEP_HORIZON)
        for i in range(SWEEP_N):
            obs = self._look()
            if otype:
                b = self._seen(obs, otype)
                if b:
                    return b
            if i < SWEEP_N - 1:
                self.h.move("RotateRight", degrees=360.0 / SWEEP_N)
            self._cap()
        cur, _ = hm.current_cell(self.h, self.gm)
        self.looked_around.append({"cell": cur, "step": self.h.step, "ver": self._world_ver,
                                   "crouched": self._crouched})
        return None

    def _seen(self, obs, otype):
        tids = [d.get("track") for d in obs.dets if d.get("type") == otype and d.get("track")]
        for b in self.om.lookup(otype):
            if b["track"] in tids and b["xyz"]:
                if self._srch is not None and self._srch["otype"] == otype and self._srch["first"] is None:
                    self._srch["first"] = self.h.step      # B1: the primitive of the first detection of the searched type
                return b
        if self._la is not None and self._la["otype"] == otype:
            return self._la_here(obs, otype)              # E3: no confident sighting here -> re-evaluate the look-alikes in view
        return None

    def _goto(self, xyz, known_height: bool = True):
        # D needs the target's own height; a placed belief's xyz is the receptacle anchor (live 2026-10-09: a sink
        # anchor at y = 0 made the agent crouch to the floor and lose the cup in the sink) -> the fixed stand-off
        if self.exhaustive and known_height:
            return self._goto_reach(xyz)
        if NEAR_OK[0] <= self._hdist(xyz) <= NEAR_OK[1]:
            return {"ok": True, "steps": 0}
        self._stand()
        r = hm.goto_near(self.h, self.gm, xyz, RMIN, RMAX, strafe=self.strafe)
        if not r["ok"]:
            r = hm.goto_near(self.h, self.gm, xyz, 0.25, 1.4, strafe=self.strafe)
        if not r["ok"]:
            raise _Fail("unreachable", f"no walk to within {RMAX} m on the own map ({r['why']})")
        return r

    def _close_blocking_door(self) -> bool:
        """A walk failed while a door THIS agent opened stands open with its swept floor (hard or soft) at or next to
        the agent (live 2026-10-09, groceries: the open fridge leaf on one side, its body on the other -- every move
        refused): close that door from where it stands (its opening pose) and say so; a later step that needs it opens
        it again. Returns True if one was closed."""
        if not self.tidy or getattr(self, "_closing_block", False):
            return False
        cur, _ = hm.current_cell(self.h, self.gm)
        near = {cur} | {(cur[0] + dx, cur[1] + dz) for dx, dz in hm.DIRS}
        for e in sorted(self.opened_by_me, key=lambda e: -e["step"]):
            key = e.get("door_key")
            cells = set(self.gm.temp_blocked.get(key, ())) | set(getattr(self.gm, "temp_soft", {}).get(key, ()))
            if not key or not (cells & near) or e.get("why") == "explicit":
                continue                               # an explicit open is the plan's state: never closed here
            self._ev(f"walk blocked while the {e['type']} this agent opened stands open beside it - closing it "
                     "first (a later step opens it again)")
            self._closing_block = True
            try:
                res = self._tidy_close(e, "(it blocks the way)")
            finally:
                self._closing_block = False
            if res in ("closed", "already"):
                return True
        return False

    def _goto_reach(self, xyz):
        """D (2026-10-09, console): the stand-off is derived from the act reach and the target's height -- r_max =
        sqrt((ACT_REACH - REACH_EPS)^2 - dy^2) for the better stance -- instead of a fixed 1.0 m: the comfortable ring
        [RMIN, min(RMAX, r_max)] first, then [0.25, r_max], then the nearest own-map cell it can walk to (reported;
        the gate decides). Never nearer than the steepest downward pitch allows to see it."""
        eye = self._eye_stand()
        rmax = ring_max(eye, xyz[1])
        hd = self._hdist(xyz)
        lo_v = min_view_h(eye - CROUCH_DROP, xyz[1])
        if NEAR_OK[0] <= hd <= min(NEAR_OK[1], rmax if rmax is not None else NEAR_OK[1]):
            return {"ok": True, "steps": 0}
        self._stand()
        rings = []
        if rmax is not None:
            if rmax > RMIN:
                rings.append((max(RMIN, lo_v), min(RMAX, rmax)))
            rings.append((max(0.25, lo_v), rmax))
        r = {"ok": False, "why": "no goal"}
        for attempt in range(2):
            for lo, hi in rings:
                if hi <= lo:
                    continue
                r = hm.goto_near(self.h, self.gm, xyz, lo, hi, strafe=self.strafe)
                if r["ok"]:
                    return r
            # goal cells exist but no walk reaches them: a door this agent opened may be what boxes it in
            if attempt or r.get("why") == "no goal" or not self._close_blocking_door():
                break
        why0 = r.get("why")
        r = self._walk_nearest(xyz, lo_v)
        if r["ok"]:
            self._ev(f"no own-map cell {'within reach of' if rmax is None or why0 == 'no goal' else 'walkable within reach of'} "
                     f"the target ({why0}) - walked to the nearest one, {self._hdist(xyz):.2f} m (the gate decides)")
            return r
        raise _Fail("unreachable", f"no walk to within {RMAX} m on the own map ({why0}; nearest: {r['why']})")

    def _walk_nearest(self, xyz, lo: float = 0.0):
        """Walk to the own-map cell nearest (horizontally) to ``xyz`` that the walk can reach (hard blocks excluded,
        soft ones cost more); cells nearer than ``lo`` are skipped."""
        cur, _ = hm.current_cell(self.h, self.gm)
        dist = grid_dist(self.gm, cur)
        cand = []
        for c in dist:
            wx, wz = self.gm.world(c)
            h_ = math.hypot(wx - xyz[0], wz - xyz[2])
            if h_ >= lo:
                cand.append((round(h_ / 0.05), dist[c], c))
        if not cand:
            return {"ok": False, "why": "no reachable cell"}
        cand.sort()
        best = cand[0][0]
        goals = [c for k, _, c in cand if k <= best + 1][:6]
        if cur in goals:
            return {"ok": True, "steps": 0, "why": ""}
        return hm.walk_path(self.h, self.gm, self.gm.astar(cur, goals), goals=goals, retries=2, strafe=self.strafe,
                            face=xyz if self.strafe else None)

    def _reach_move(self, xyz, why: str = "reach") -> bool:
        """D (2026-10-09, console): after the reach gate refused every pixel as out of reach, go where the PREDICTED 3D
        reach is better: the own-map cell (reachable on the own map, any detour) whose best stance brings the eye
        nearest the target, preferring a short walk; crouch there when standing cannot reach and crouching is nearer.
        Never a blind step away. Returns True when it moved or changed stance (then the caller re-looks + re-gates)."""
        eye = self._eye_stand()
        cur, _ = hm.current_cell(self.h, self.gm)
        hx = self._hdist(xyz)
        now = math.hypot(hx, self._cam_y() - xyz[1])
        lo_v = min_view_h(eye - CROUCH_DROP, xyz[1])
        dist = grid_dist(self.gm, cur)
        R = ACT_REACH - REACH_EPS
        cand = []
        for c, wd in dist.items():
            wx, wz = self.gm.world(c)
            h_ = math.hypot(wx - xyz[0], wz - xyz[2])
            if h_ < lo_v:
                continue
            d3, stance = reach_pred(h_, eye, xyz[1])
            ds = math.hypot(h_, eye - xyz[1])
            st_ok = ds <= R
            if st_ok:
                d3, stance = ds, "stand"
            # standing that reaches beats any crouch (live 2026-10-09: crouched, the eye could not see into the sink);
            # a cell an open door may sweep is a last resort
            soft = self.gm.is_soft(c) if hasattr(self.gm, "is_soft") else False
            # the nearest predicted reach first (margin against estimate error), the walk only breaks ties
            cand.append((soft, 0 if st_ok else 1, round(d3 / 0.05), wd, c, d3, stance, st_ok))
        better = [x for x in cand if (x[7] and now > R) or x[5] < now - REACH_EPS]
        if not better:
            best = min(cand, key=lambda x: x[5]) if cand else None
            self._ev(f"move nearer the target for {why}: no own-map pose predicts a nearer reach "
                     f"(now {now:.2f} m" + (f", best {best[5]:.2f} m {best[6]}" if best else "") + ")")
            return False
        better.sort()
        _, _, _, _, cell, d3, stance, _ = better[0]
        if cell != cur:
            self._stand()
            r = hm.walk_path(self.h, self.gm, self.gm.astar(cur, [cell]), goals=[cell], retries=2,
                             strafe=self.strafe, face=xyz if self.strafe else None)
            if not r["ok"]:
                self._ev(f"move nearer the target for {why}: walk to {tuple(cell)} failed ({r['why']})")
                return False
        if stance == "crouch":
            self._crouch()
        else:
            self._stand()
        self._face(xyz)
        self._ev(f"move nearer the target for {why}: cell {tuple(cell)}, {stance}, predicted {d3:.2f} m "
                 f"(was {now:.2f} m)")
        return True

    def _stance_for_reach(self, xyz):
        """D: standing cannot reach ``xyz`` from here (predicted) -> the reach move: a reachable cell where standing
        reaches first, a crouch (here or nearer) only when no such cell exists and the target is below the crouched
        eye. Nothing when standing already reaches."""
        if xyz is None or not self.exhaustive:
            return
        _, stance = reach_pred(self._hdist(xyz), self._eye_stand(), xyz[1])
        if stance == "crouch" and not self._crouched:
            self._reach_move(xyz, "reach (before acting)")

    def _walk_cell(self, cell, face=None) -> bool:
        self._stand()
        cur, _ = hm.current_cell(self.h, self.gm)
        r = hm.walk_path(self.h, self.gm, self.gm.astar(cur, [cell]), goals=[cell], retries=1, strafe=self.strafe,
                         face=face if self.strafe else None)
        return bool(r["ok"])

    # -------------------------------------------------------------- locating
    def _rank(self, beliefs, need=None):
        srank = {"trusted": 0, "placed": 1}

        def key(b):
            pref = 1
            if need is not None:
                v = state_of({"state": b.get("state")}, need[0])
                pref = 0 if v == need[1] else (1 if v is None else 2)
            return (srank.get(b["status"], 3), pref, self._hdist(b["xyz"]))

        beliefs = [self._hyp(b) for b in beliefs]
        return sorted([b for b in beliefs if b["status"] in srank and b.get("xyz") and not self._is_aside(b)],
                      key=key)

    def _is_aside(self, b) -> bool:
        st = self.aside.get(b.get("track"))
        return st is not None and int(b.get("step") or 0) <= st

    def _set_aside(self, b, why):
        if b and b.get("track"):
            self.aside[b["track"]] = self.h.step
        self._ev(f"belief {b.get('type', '?') if b else '?'} at "
                 f"{tuple(round(v, 2) for v in (b or {}).get('xyz') or ())} set aside: {why} (evidence kept)")

    def _locate(self, otype, need=None, search=True):
        bel = self._rank(self.om.lookup(otype), need)
        pb = self._put_belief(otype)
        if pb is not None and (not bel or bel[0]["status"] != "trusted" or int(bel[0].get("step") or 0) < pb["step"]):
            self._ev(f"{otype}: I put it in {pb['place']} myself (step {pb['step']}) and have not seen it elsewhere since"
                     " - aiming there" + ("" if not bel else " (the sighting I have is older than the put)"))
            return pb
        if bel:
            return bel[0]
        if not search:
            return None
        b = self._hyp(self._search(otype))
        if not b:
            rep = (self.search_info or {}).get("report")
            raise _Fail("not_detected", f"no {otype} found: {rep}" if rep else
                        f"no {otype} found within the search budget")
        return b

    # -------------------------------------------------------------- choosing a detection and a point
    def _pick(self, obs, otype, expect=None, need=None, inhand=False):
        wh = _wh(obs.depth)
        cands = []
        for d in obs.dets:
            if d.get("type") != otype or not d.get("point") or not d.get("bbox_xyxy"):
                continue
            dd = _depth(obs.depth, d["point"])
            near_hand = dd is not None and dd < objmem.HELD_NEAR
            if not near_hand and self.h.holding == otype and d.get("type") not in SURFACE_TYPES:
                near_hand = objmem.at_hand(obs.depth, d)  # SPEC-INHAND-apply: several near pixels, not the one ground point
            if inhand != (near_hand and self.h.holding == otype):
                continue
            cands.append(d)
        if not cands:
            return None
        if inhand:
            return max(cands, key=lambda d: _area(d["bbox_xyxy"]))
        ex = objmem.project(expect, obs.pose, wh, self.om.fov) if expect else None

        def key(d):
            b = d["bbox_xyxy"]
            if ex is None:
                pix, bucket = 0.0, 0
            else:
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                pix = 0.0 if _inside(ex, b, 10) else math.hypot(ex[0] - cx, ex[1] - cy)
                bucket = int(pix // (0.1 * wh[1]))
            pref = 1
            if need is not None:
                v = state_of(d, need[0])
                pref = 0 if v == need[1] else (1 if v is None else 2)
            return (bucket, pref, pix, -float(d.get("conf", 0.0)))

        return min(cands, key=key)

    def _point(self, obs, det, alt: int, receptacle: bool = False):
        b = det["bbox_xyxy"]
        self._put_plane = None
        if receptacle and PUT_SURFACE_ON and self.exhaustive and det.get("type") not in OPENABLE:
            res = put_surface(obs.depth, obs.pose, self.om.fov, b, obs.dets, det.get("type"),
                              rejected=self.put_reject, max_depth=REACH)
            self._put_plane = res["plane"]
            if res["pts"]:
                if res["tier"] > 1 or res["plane"] is not None:
                    self._ev(f"put surface (tier {res['tier']}): {res['note']}")
                return res["pts"][min(alt, len(res["pts"]) - 1)]
            if res["blocked"]:
                self._ev(f"put surface: {res['note']}")
                return None
        if receptacle:
            avoid = [o["bbox_xyxy"] for o in obs.dets if o is not det and o.get("bbox_xyxy")
                     and o.get("type") != det.get("type") and _area(o["bbox_xyxy"]) < _area(b)]
            pts = candidate_points(obs.depth, b, n=7, agree=False, avoid=avoid, max_depth=REACH)
            if not pts:
                return det.get("point")
            return pts[min(alt, len(pts) - 1)]
        if alt <= 0 or not det.get("point_ok"):
            return det.get("point")
        first = tuple(det["point"])
        pts = [p for p in candidate_points(obs.depth, b) if math.hypot(p[0] - first[0], p[1] - first[1]) > 3]
        return pts[min(alt - 1, len(pts) - 1)] if pts else det.get("point")

    def _xyz_of(self, det, fallback=None):
        hy = getattr(self, "fg_hyp", {}).get(det.get("track")) if det.get("track") and self.exhaustive else None
        if hy is not None:
            return hy["xyz"]
        t = self.om.tracks.get(det.get("track")) if det.get("track") else None
        return (t.xyz if t is not None and t.xyz else None) or fallback

    # -------------------------------------------------------------- recovery moves
    def _recover(self, k, xyz, st, inhand=False, yaw=False):
        if inhand or xyz is None:
            st["dpitch"] = (PITCH_ALT, 2 * PITCH_ALT, -PITCH_ALT)[min(k, 3) - 1]
            st["alt"] = k
            if xyz is None:
                self._set_horizon(float(self.h.pose()["horizon"] or 0.0) + PITCH_ALT)
            else:
                self._face(xyz, st["dpitch"])
            self._ev(f"recover {k}: camera pitch {st['dpitch']:+.0f}, alternate point")
            return
        if k == 1:
            st["dpitch"] = PITCH_ALT if st.get("below", True) else -PITCH_ALT
            st["alt"] = 1
            st["dyaw"] = YAW_ALT if yaw else 0.0
            self._face(xyz, st["dpitch"], st["dyaw"])
            self._ev(f"recover 1: fresh detection, alternate point, pitch {st['dpitch']:+.0f}"
                     + (f", yaw {st['dyaw']:+.0f}" if yaw else ""))
            return
        cur, _ = hm.current_cell(self.h, self.gm)
        dcur = math.hypot(*(a - b for a, b in zip(self.gm.world(cur), (xyz[0], xyz[2]))))
        opts = []
        for n in self.gm.neighbours(cur):
            dn = math.hypot(*(a - b for a, b in zip(self.gm.world(n), (xyz[0], xyz[2]))))
            if k == 2 and 0.3 <= dn <= 1.3:
                opts.append((abs(dn - dcur), n))           # sideways: about the same distance
            elif k >= 3 and dn > dcur:
                opts.append((-dn, n))                      # back: farther from the target
        moved = False
        for _, n in sorted(opts):
            if self._walk_cell(n):
                moved = True
                break
        st["dpitch"], st["alt"] = 0.0, 0
        st["dyaw"] = -YAW_ALT if yaw else 0.0
        self._face(xyz, 0.0, st["dyaw"])
        self._ev(f"recover {k}: {'sidestep' if k == 2 else 'step back'} one cell"
                 f"{'' if moved else ' (no free neighbour; re-faced only)'}"
                 + (f", yaw {st['dyaw']:+.0f}" if yaw else ""))

    def _local_look(self, k, b, st):
        """Exhaustive mode: one free local look at a remembered spot. The first also opens the container the
        object was last seen in/on, if that is an openable (it may have been shut since)."""
        place = (b or {}).get("place")
        if k == 1 and place in OPENABLE and b.get("xyz"):
            if self._open_container(place, b["xyz"], why="look"):
                self._face(b["xyz"], st.get("dpitch", 0.0))
                return
        self._recover(k, b["xyz"] if b else None, st)

    # -------------------------------------------------------------- one interaction with retries
    def _interact(self, verb, otype, belief=None, inhand=None, **kw):
        if verb == "slice":
            self._cutting_tool_check(otype)              # A26 item 2 / A3: no path may send SliceObject without the tool
        prim, post = VERBS[verb]
        need = (post[0], not post[1]) if post else None
        if inhand is None:
            inhand = (self.h.holding == otype and verb != "pickup")
        if belief is None and verb == "close" and otype in self.opened_at:
            belief = {"status": "trusted", "xyz": self.opened_at[otype], "anchor": "opened-at"}
        st = {"alt": 0, "dpitch": 0.0, "approached": False, "local": 0, "relocs": 0}
        k = 0
        while True:
            try:
                out = self._attempt(verb, prim, post, need, otype, belief, inhand, st, kw)
                self.sub.append({"verb": verb, "object": otype, "ok": True, **out})
                return out
            except _Soft as e:
                self.sub.append({"verb": verb, "object": otype, "ok": False, "cause": e.cause})
                if (self.exhaustive and e.cause == "not_detected" and not inhand and belief is None
                        and st.get("b") is not None and st["relocs"] < MAX_RELOCATE):
                    self._cap()
                    if st["local"] < LOCAL_LOOKS:            # a free local look first (searching, not retrying)
                        st["local"] += 1
                        self._local_look(st["local"], st["b"], st)
                        continue
                    self._set_aside(st["b"], f"{otype} not re-seen after {LOCAL_LOOKS} local looks")
                    st.update(approached=False, local=0, alt=0, dpitch=0.0, b=None)
                    st["relocs"] += 1
                    continue
                self._spend(e)
                k += 1
                if (self.exhaustive and e.cause == "unreachable" and "out of reach" in (e.why or "")
                        and st.get("xyz") is not None and not inhand and not st.get("anchor")):
                    # D (2026-10-09): out of reach -> go where the predicted reach is better; never step away
                    if not self._reach_move(st["xyz"], "reach (retry)"):
                        self._recover(1, st["xyz"], st, inhand)   # alternate point + pitch only, no step away
                    continue
                self._recover(k, st.get("xyz"), st, inhand)

    def _attempt(self, verb, prim, post, need, otype, belief, inhand, st, kw):
        self._cap()
        xyz = None
        if not inhand:
            if not st["approached"]:
                st["b"] = belief or self._locate_la(verb, otype, need, st)
            b = st["b"]
            if b.get("la_kept"):
                return self._la_finish(otype, b)          # E3: the search's look-alike check already has it in the hand
            xyz = b["xyz"]
            st["xyz"] = xyz
            if not st["approached"]:
                st["anchor"] = b.get("status") == "placed"      # xyz = the receptacle anchor, not its own height
                self._goto(xyz, known_height=not st["anchor"])
                if b.get("status") == "placed" and b.get("place") in OPENABLE:
                    self._open_container(b["place"], xyz)
                self._face(xyz, st["dpitch"])
                if prim not in ("OpenObject", "CloseObject") and not st["anchor"]:
                    self._stance_for_reach(xyz)
                st["approached"] = True
        obs = self._look()
        det = self._pick(obs, otype, xyz, need, inhand)
        if det is None:
            if self.exhaustive and not inhand and st.get("anchor") and not st.get("blind")                     and prim in BLIND_PRIMS and otype in self.put_rec:
                st["blind"] = True
                out = self._blind_act(verb, prim, post, otype, st, obs, kw)
                if out is not None:
                    return out
            raise _Soft("not_detected", f"no {otype} in view")
        st["below"] = det["bbox_xyxy"][1] + det["bbox_xyxy"][3] >= obs.depth.shape[0]
        if post and state_of(det, post[0]) == post[1] and verb in REST_VERBS:
            # several instances (drawers, knobs): the one nearest the belief is already at rest, but another one
            # in this same frame is confidently NOT -- that one is what the step is about (s5 live: close Drawer)
            alt = [d for d in obs.dets if d is not det and d.get("type") == otype and d.get("point")
                   and d.get("bbox_xyxy") and state_of(d, post[0]) == need[1]]
            if alt:
                det = max(alt, key=lambda d: float(d.get("conf", 0.0)))
                xyz = st["xyz"] = self._xyz_of(det, xyz)
                if st.get("b") is not None and xyz is not None:
                    st["b"] = {**st["b"], "xyz": xyz}          # a retry re-aims at the one being acted on
                self._ev(f"{verb} {otype}: nearest one already {STATE_WORDS[post[0]][int(post[1])]}, another in "
                         f"view is {STATE_WORDS[post[0]][int(need[1])]} -> act on that one")
        if post and state_of(det, post[0]) == post[1]:
            self._ev(f"{verb} {otype}: already {STATE_WORDS[post[0]][int(post[1])]} (seen)")
            self.verify.append("already")
            return {"verify": "already"}
        pt = self._point(obs, det, st["alt"])
        if pt is None:
            raise _Soft("not_detected", f"{otype} box has no point")
        sw = None
        if verb == "open" and state_of(det, "open") is False:
            sw = 0.0
        elif verb == "close" and any(e["type"] == otype for e in self.opened_by_me):
            sw = 1.0
        container = otype in OPENABLE and prim in ("OpenObject", "CloseObject")
        front, mode = None, "front"
        if container and prim == "CloseObject":
            front, mode = self._open_front(otype, xyz or self._xyz_of(det)), "door"
        elif container:
            front = front_patch(obs.depth, det["bbox_xyxy"], obs.pose, self.om.fov)
        pts = self._gate_points(obs, det, first=[pt], openable=True) if container else [pt]
        self._toggle_now = prim in TOGGLE_PRIMS
        if self.exhaustive and prim in TOGGLE_PRIMS and not container and not inhand:
            pts = self._gate_points(obs, det, first=pts)      # a few more pixels of the box for a refused toggle
        if self.exhaustive and not container and not inhand:
            pts = self._fg_first(obs, det, pts)
        self._aim_next = False
        if prim == "PickupObject" and not inhand and not container:
            pts = self._aim_pickup(obs, det, pts, st)
        if self.exhaustive and prim == "ToggleObjectOff" and otype in self.toggled_on_at:
            # B1b: a thin fixture (faucet) whose pixels read the far window, with a stream in front once it runs: the
            # near component is the stream, not the fixture. I switched it ON at a point I measured myself; aim there.
            q = objmem.project(self.toggled_on_at[otype], obs.pose, _wh(obs.depth), self.om.fov)
            h_, w_ = obs.depth.shape[:2]
            if q is not None and 0 <= q[0] < w_ and 0 <= q[1] < h_:
                qp = (int(round(q[0])), int(round(q[1])))
                self._ev(f"{otype}: aiming at the point I switched it on at (own record), pixel {qp}")
                pts = [qp] + [t for t in pts if t != qp]
        r = self._act_at(prim, obs, pts, otype, front=front, mode=mode, swing=sw, **kw)
        self._aim_next = False
        if not r["sent"]:
            tgt = xyz or self._xyz_of(det)
            if st.get("anchor") and det.get("track") and self.om.tracks.get(det["track"]) is not None \
                    and self.om.tracks[det["track"]].xyz:
                tgt = self._xyz_of(det)                    # seen now: its own estimate, not the receptacle anchor
            if "out of reach" in r["why"] and not st.get("closer") and not inhand and tgt is not None:
                st["closer"] = True                       # once per interaction: walk nearer, look again, re-gate
                if self._closer(tgt, known_height=not st.get("anchor") or tgt is not xyz):
                    return self._attempt(verb, prim, post, need, otype, belief, inhand, st, kw)
            raise _Soft("unreachable", f"not sent: {r['why']}")
        pt = r["pt"]
        self._ev(f"{prim} {otype} at {tuple(pt)}: {'ok' if r['ok'] else r['err']}")
        if r["ok"] and prim in TOGGLE_PRIMS:
            if prim == "ToggleObjectOn":
                dd_ = _depth(obs.depth, tuple(pt))
                xyz_ = objmem.backproject(pt[0], pt[1], dd_, obs.pose, _wh(obs.depth), self.om.fov) if dd_ else None
                if xyz_ is not None:
                    self.toggled_on_at[otype] = tuple(xyz_)
            else:
                self.toggled_on_at.pop(otype, None)
        if not r["ok"]:
            raise _Soft("action_refused", r["err"])
        if verb == "open" and container and front is not None:
            if self._after_open(obs, pt, otype, front, otype) == "wrong":
                raise _Soft("wrong_target", "the point opened a different front (closed again)")
        if verb == "pickup":
            self._identity_check(otype, st)               # may put a wrong object back and raise a retryable wrong_target
            self.om.on_pickup(True, det.get("track"), otype, step=self.h.step)
            self._world_ver += 1
            self.verify.append("success-bool")
            if self.tidy:
                for e in [e for e in self.opened_by_me if e.get("target") == otype and e["why"] == "search-found"]:
                    self._tidy_close(e, f"(tidy, {otype} taken out)")
            return {"verify": "success-bool"}
        if verb == "open" and otype in OPENABLE and xyz:
            self.opened_at[otype] = tuple(xyz)            # own history: where I aimed when I opened it
            self._note_open(otype, xyz, "explicit" if self._step_action == "open" else
                            ("put" if self._in_put else "composite"), front=front)
        elif verb == "close":
            self.opened_at.pop(otype, None)
            self._note_closed(otype, xyz)
        if not post:
            self.verify.append("success-bool")
            return {"verify": "success-bool"}
        v = self._recheck(otype, pt, post[0])             # verify the postcondition on the acted target
        if v is not None and v != post[1]:
            # the first look reads the frame rendered by the action itself (e.g. no water stream yet): let one
            # simulation step pass and look again before calling it a wrong target
            self.h.move("Pass")
            v = self._recheck(otype, pt, post[0])
        if v is None:
            self.verify.append("state_unrecognised")
            self._ev(f"{verb} {otype}: success, state not readable (verified by success bool only)")
            return {"verify": "state_unrecognised"}
        if v != post[1]:
            raise _Soft("wrong_target", f"{otype} still {STATE_WORDS[post[0]][int(v)]} after a successful {prim}")
        self.verify.append("state")
        return {"verify": "state"}

    # -------------------------------------------------------------- E1 aim and E2 identity (Round 2 / J-R1)
    def _aim_pickup(self, obs, det, pts, st) -> list:
        """E1: pickup pixels ON the object (protruding from the support surface, clear of neighbouring boxes), the legacy
        points kept after them. Each attempt starts ``AIM_PER_TRY`` further along the list, so a retry never re-sends the
        pixels the last one used. Unchanged (legacy points only) when the geometry gives nothing to go on."""
        if not (AIM_PROTRUDE_ON and self.aim_protrude) or det.get("bbox_xyxy") is None:
            return pts
        b = det["bbox_xyxy"]
        avoid = []
        for o in obs.dets:
            ob = o.get("bbox_xyxy")
            if o is det or not ob or o.get("type") in SURFACE_TYPES:
                continue
            avoid.append(ob)
        aim = protruding_points(obs.depth, b, avoid, first=pts[0] if pts else det.get("point"))
        if not aim:
            if not PICK_MULTI_ON:
                return pts
            # SPEC-INHAND-apply: no pixel stands out -> the legacy point(s) first (unchanged first send), then the NEXT object
            # pixels of the box, so a refused pixel is not the end of the attempt; each retry starts further along
            legacy = [tuple(p) for p in pts if p is not None]
            more = [p for p in self._gate_points(obs, det, n=AIM_K) if p not in legacy][:AIM_K]
            if not more:
                return pts
            rnd = st["aim_round"] = st.get("aim_round", -1) + 1
            k = (AIM_PER_TRY * rnd) % len(more)
            more = more[k:] + more[:k]
            self._aim_next = True
            self.pick_stats["fallback_aims"] += 1
            self._ev(f"{det.get('type')}: no pixel stands out from the surface behind (try {rnd + 1}) - the usual point first, "
                     f"then {len(more)} more pixel(s) of the box if it is refused")
            return (legacy[:1] + more + legacy[1:]) if legacy else more
        rnd = st["aim_round"] = st.get("aim_round", -1) + 1
        k = (AIM_PER_TRY * rnd) % len(aim)
        order = aim[k:] + aim[:k]
        rest = [tuple(p) for p in pts if p is not None and tuple(p) not in order]
        # J-regress (live regression 2026-10-13): a hollow or open object (a cup seen from the side) has no pixel that
        # "stands out" which the ray accepts, and an attempt sends at most AIM_PER_TRY pixels, so three aimed pixels
        # starved the legacy centre point and the pickup was refused every time. The legacy first point is therefore the
        # LAST of the pixels one attempt sends: aimed pixels get the first AIM_PER_TRY-1 slots, the old aim the next one.
        if rest and AIM_PER_TRY > 1:
            order = order[:AIM_PER_TRY - 1] + rest[:1] + order[AIM_PER_TRY - 1:]
            rest = rest[1:]
        self._aim_next = True
        self._ev(f"{det.get('type')}: aiming at {len(aim)} pixel(s) that stand out from the surface behind (try {rnd + 1}), "
                 f"first {order[0]}; the box centre is {tuple(int(v) for v in ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))}")
        return order + rest

    def _inhand_relook(self, judge, summarize):
        '''SPEC-INHAND fix B (Astra SPEC-INHAND Q3): up to len(INSPECT_ROLLS) more in-hand looks, each after a relative quarter
        roll of the held object (``RotateHeldObject``: a hand primitive, no target, no privileged read). No locomotion happens
        inside the sequence (AI2-THOR resets the grip on a move), so the accumulated inverse is valid. ``judge(obs) -> (done,
        result)``; ``summarize(result) -> dict`` for the per-view log. Bounded: at most len(INSPECT_ROLLS) rolls + 1 undo
        (3 rotation actions), len(INSPECT_ROLLS) looks. The undo is CHECKED: a refused undo is logged and counted and the world
        geometry invalidated (``_world_ver``); the caller refreshes before any further targeting (``info["stale"]``). Every
        action, refusal, look and exception is counted in ``identity_stats``. One look = one observation = one primary + one
        specialist inference while holding. Returns ``(result of the last look or None, info)``.'''
        stt = self.identity_stats
        info = {"looks": 0, "rotations": 0, "rot_refused": 0, "undo": None, "undo_ok": None, "views": [], "error": None}
        p0, total, res = self.h.step, 0.0, None
        try:
            for roll in INSPECT_ROLLS:
                r = self.h.act("RotateHeldObject", roll=float(roll))
                info["rotations"] += 1
                if not r["ok"]:
                    info["rot_refused"] += 1
                    info["views"].append({"roll": total + float(roll), "sent": False, "err": r.get("err")})
                    break
                total += float(roll)
                obs = self._observe()
                info["looks"] += 1
                done, res = judge(obs)
                info["views"].append(dict({"roll": total}, **summarize(res)))
                if done:
                    break
        except Exception as e:                            # logged, the undo below still runs, then re-raised
            info["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            if total:
                info["undo"] = -total
                try:
                    u = self.h.act("RotateHeldObject", roll=-total)
                    info["undo_ok"] = bool(u["ok"])
                    if not u["ok"]:
                        info["undo_err"] = u.get("err")
                except Exception as e:                    # noqa: BLE001 - recorded; never masks the original outcome
                    info["undo_ok"], info["undo_err"] = False, f"{type(e).__name__}: {e}"
                info["rotations"] += 1
                if not info["undo_ok"]:
                    self._world_ver += 1                  # the held object's pose is not what the agent believes: re-observe
            info["prims"] = self.h.step - p0
            info["stale"] = self.h.step != p0
            for k, v in (("inspect_relooks", info["looks"]), ("inspect_rotations", info["rotations"]),
                         ("inspect_rot_refused", info["rot_refused"]), ("inspect_undo_failed", int(info["undo_ok"] is False)),
                         ("inspect_errors", int(bool(info["error"]))), ("inspect_prims", info["prims"])):
                stt[k] = stt.get(k, 0) + v
        return res, info

    def _identity_check(self, otype, st):
        """E2 (costed perception, Astra round 2): after a successful pickup of a specialist-class type, look at the hand ONCE
        (the specialist runs while the agent holds something; observed image only, memory is not updated from this look). A
        different confident type in the hand is put back (``DropHandObject``) and the attempt retried with the next pixels;
        doubt keeps the pickup. At most ``IDENT_SWAPS_MAX`` swap-backs per interaction, then the step fails honestly.
        Costs are counted: detector calls (one look = primary + specialist), extra primitives, swaps."""
        if not (IDENT_ON and self.identity_check) or otype not in AMBIG_TYPES or self.h.holding != otype:
            return
        p0 = self.h.step
        obs = self._observe()
        verdict, seen, conf = identity_verdict(_raw_dets(obs), otype, obs.depth)   # A7: the unfiltered frame (Knife class kept)
        relooks, insp = 0, None
        first = (verdict, seen, round(float(conf), 3))
        if verdict == "unclear" and INSPECT_RELOOK_ON:                                # SPEC-INHAND fix B
            res, insp = self._inhand_relook(
                lambda o: (lambda v: (v[0] != "unclear", v))(identity_verdict(_raw_dets(o), otype, o.depth)),
                lambda v: {"verdict": v[0], "seen": v[1], "conf": round(float(v[2]), 3)} if v else {})
            relooks = insp["looks"]
            if res is not None:                           # the LAST look's verdict is the result (unclear if none was conclusive)
                verdict, seen, conf = res
        rec = {"asked": otype, "step": p0, "verdict": verdict, "seen": seen, "conf": round(float(conf), 3), "swap": False,
               "relooks": relooks}
        if insp is not None:
            rec.update(first_look=first, inspect=insp)
        stt = self.identity_stats
        stt["checks"] += 1
        stt["detector_calls"] += 1 + relooks
        self.identity_log.append(rec)
        self._ev(f"identity check: asked {otype}, the hand shows {seen or '-'} ({conf:.2f}) -> {verdict}")
        if verdict == "unclear":
            stt["unclear"] += 1
        if verdict != "wrong":
            stt["extra_prims"] += self.h.step - p0
            return
        stt["wrong"] += 1
        n = st["ident_swaps"] = st.get("ident_swaps", 0) + 1
        r = self.h.act("DropHandObject")
        self._world_ver += 1
        rec["swap"] = True
        stt["swaps"] += 1
        stt["extra_prims"] += self.h.step - p0
        self._ev(f"identity check: the object in hand is a {seen}, not the {otype} - put it down ({'ok' if r['ok'] else r['err']})")
        if n > IDENT_SWAPS_MAX or not r["ok"]:
            raise _Fail("wrong_target", f"the hand holds a {seen}, not the {otype} (after {n} swap-back(s)"
                        + ("" if r["ok"] else "; could not put it down") + ")", under="identity")
        raise _Soft("wrong_target", f"picked up a {seen} instead of the {otype} (identity check; put back)")

    # -------------------------------------------------------------- E3 look-alike re-evaluation (DESSTA, Algorithm 4)
    def _locate_la(self, verb, otype, need, st):
        """``_locate`` with the E3 context on for a PICKUP of a look-alike-prone type with an empty hand (exhaustive search only):
        while it runs, every search look without a usable TRACKED sighting of the asked type re-evaluates the ambiguity points
        in view (``_seen``) and the camera also reports the asked type's low-confidence boxes. Otherwise exactly ``_locate``.
        The context (checks spent, points checked, points pending) lives in the interaction state ``st``: it persists across
        retries of THIS pickup and never reaches a later pickup."""
        if not (LOOKALIKE_ON and self.lookalike and self.exhaustive and verb == "pickup" and self.h.holding is None
                and otype in AMBIG_TYPES and otype not in LOOKALIKE_EXCLUDE):
            return self._locate(otype, need)
        ctx = st.setdefault("la", {"otype": otype, "n": 0, "pending": [], "checked": []})
        prev, prev_low = self._la, getattr(self.h, "low_conf", None)
        self._la = ctx
        self.h.low_conf = LOOKALIKE_LOW
        try:
            return self._locate(otype, need)
        finally:
            self._la = prev
            self.h.low_conf = prev_low

    def _la_here(self, obs, otype):
        """The ambiguity points in THIS frame (``lookalike_candidates``), each re-evaluated in turn (``_revaluate``) until one
        is the asked type or the interaction's cap is spent. After any check that acted, the agent looks again and the
        caller's ``obs`` is REFRESHED IN PLACE with that new frame (Astra E3 Q6.6: a caller goes on interpreting ``obs``
        after ``_seen``; it must not act on a frame from before the pickup / put back); a tracked sighting of the asked type
        in the fresh frame ends it as usual. Returns a belief or None."""
        ctx = self._la
        if ctx is None or obs is None or obs.step != self.h.step or self.h.holding is not None:
            return None
        tried = []                                         # points of this frame that could not be sent from here
        while ctx["n"] < LOOKALIKE_MAX and not self._search_spent():
            wh = _wh(obs.depth) if obs.depth is not None else None
            todo = []
            for c in lookalike_candidates(obs, otype, self.amb_map):
                dd = _depth(obs.depth, c["point"]) if wh else None
                xyz = objmem.backproject(c["point"][0], c["point"][1], dd, obs.pose, wh, self.om.fov) if dd else None
                if xyz is None or any(math.dist(xyz, q) <= LOOKALIKE_SAME for q in ctx["checked"] + tried):
                    continue
                todo.append((c, tuple(xyz)))
            if not todo:
                return None
            c, xyz = todo[0]
            s0 = self.h.step
            got = self._revaluate(obs, otype, c, xyz)
            if got is not None:
                return got
            if self.h.holding is not None:
                return None                                # could not free the hand (failure already raised or logged)
            if self.h.step == s0:
                tried.append(xyz)                          # nothing was sent: the frame is still current
                continue
            fresh = self._la_obs if (self._la_obs is not None and self._la_obs.step == self.h.step) else self._look()
            self._la_obs = None
            for f_ in dataclasses.fields(obs):
                setattr(obs, f_.name, getattr(fresh, f_.name))
            tried = []
            tids = [d.get("track") for d in obs.dets if d.get("type") == otype and d.get("track")]
            for b in self.om.lookup(otype):
                if b["track"] in tids and b["xyz"]:
                    return b
        return None

    def _revaluate(self, obs, otype, cand, xyz):
        """DESSTA RevaluateObject for one ambiguity point (own frames only): f = ``obs`` (the object in place: it supplies the
        candidate, the pickup pixels and the restoration context), pick it up at its own pixels, f' = one look with the object
        in the hand (the gated specialist runs: the robot is holding something), identity <- ``lookalike_confirm`` on f'
        (declared adaptation: the specialist sees f' only). KEPT only on POSITIVE specialist evidence (the asked type at the
        hand >= LOOKALIKE_KEEP and ahead of every other specialist class by IDENT_MARGIN); otherwise restored (``_la_restore``:
        bounded, from fresh frames, landing observed or reported unverified, never dropped). Memory follows every step:
        on_pickup binds the candidate's track, relabel_held corrects a kept object's type, on_put records a restoration."""
        ctx, stt = self._la, self.identity_stats
        p0 = self.h.step
        ctype = cand.get("type")
        src = cand.get("_source_det", cand)
        self._aim_next = False
        try:
            pts = self._aim_pickup(obs, src, [tuple(cand["point"])], {})
            self._aim_cap = AIM_K if PICK_MULTI_ON else None   # SPEC-INHAND-apply: one call from one frame -> up to AIM_K pixels
            r = self._act_at("PickupObject", obs, pts, ctype)
        finally:
            self._aim_next = False
            self._aim_cap = None
        place = objmem.place_of(src, obs.dets)
        rec = {"kind": "lookalike", "asked": otype, "candidate": ctype, "trigger": cand.get("trigger"),
               "cand_conf": round(float(cand.get("conf", 0.0)), 3), "m_rate": cand.get("m_rate"), "step": p0,
               "xyz": [round(v, 3) for v in xyz], "place": place,
               "f_types": sorted({d.get("type") for d in obs.dets if d.get("type") and d.get("type") not in SURFACE_TYPES})[:12]}
        if not r["sent"]:
            stt["la_unreached"] += 1
            oor = "out of reach" in (r.get("why") or "")
            if oor and not any(math.dist(xyz, p["xyz"]) <= LOOKALIKE_SAME for p in ctx["pending"]):
                ctx["pending"].append({"xyz": xyz, "type": ctype, "step": p0, "place": place})
            elif not oor:
                ctx["checked"].append(xyz)                 # no pixel of it can be acted on: not tried again in THIS pickup
            rec.update(result="not sent", why=r.get("why"))
            self.identity_log.append(rec)
            self._ev(f"look-alike check: {ctype} ({rec['cand_conf']:.2f}, {rec['trigger']}) for the {otype} not sent - {r.get('why')}")
            stt["la_prims"] += self.h.step - p0
            return None
        ctx["n"] += 1
        stt["la_checks"] += 1
        ctx["checked"].append(xyz)
        ctx["pending"] = [p for p in ctx["pending"] if math.dist(p["xyz"], xyz) > LOOKALIKE_SAME]
        if not r["ok"]:
            stt["la_refused"] += 1
            stt["la_prims"] += self.h.step - p0
            rec.update(result="pickup refused")
            self.identity_log.append(rec)
            self._ev(f"look-alike check: pickup of the {ctype} for the {otype} refused ({r['err']})")
            return None
        self._world_ver += 1
        self.om.on_pickup(True, cand.get("track"), ctype, step=self.h.step)   # the candidate's own track is now in the hand
        try:
            f2 = self._observe()                          # f': the object in the hand; memory is not updated from this look
            stt["la_detector_calls"] += 1
            keep, seen, conf, rival = lookalike_confirm(_raw_dets(f2), otype, f2.depth)   # A7: unfiltered in-hand frame
            if not keep and INSPECT_RELOOK_ON:                                          # SPEC-INHAND fix B
                rec["first_look"] = {"kept": keep, "seen": seen, "conf": round(float(conf), 3),
                                     "rival": list(rival) if rival else None}
                # conclusive = kept, OR the specialist names a DIFFERENT type at the KEEP level (Astra Q3: stop on a rival too)
                res, insp = self._inhand_relook(
                    lambda o: (lambda v: (bool(v[0]) or (v[1] not in (None, otype) and v[2] >= LOOKALIKE_KEEP), v))(
                        lookalike_confirm(_raw_dets(o), otype, o.depth)),
                    lambda v: {"kept": bool(v[0]), "seen": v[1], "conf": round(float(v[2]), 3),
                               "rival": list(v[3]) if v[3] else None} if v else {})
                stt["la_detector_calls"] += insp["looks"]
                rec["relooks"], rec["inspect"] = insp["looks"], insp
                if res is not None:                                                     # the LAST look's evidence is the result
                    keep, seen, conf, rival = res
                if insp["stale"]:                                                       # a hand action was sent: f2 is stale
                    f2 = self._observe()                                                # the restore aims from a current frame
                    stt["la_detector_calls"] += 1
                    rec["relook_refresh"] = 1
            rec.update(seen=seen, conf=round(float(conf), 3), rival=list(rival) if rival else None, kept=keep)
            self._ev(f"look-alike check {ctx['n']}/{LOOKALIKE_MAX}: picked up the {ctype} ({rec['cand_conf']:.2f}, "
                     f"{rec['trigger']}) seen while looking for the {otype}; in the hand the specialist says {seen or '-'} "
                     f"({conf:.2f}) -> " + ("it IS the " + otype + ", kept" if keep else "not confirmed as the " + otype
                                            + ", putting it back"))
            if keep:
                self.h.holding = otype                     # own belief, from the specialist's look at the hand
                self.om.relabel_held(otype, conf, step=self.h.step,
                                     why=f"identity corrected in the hand by the specialist: {ctype} -> {otype} ({conf:.2f})")
                stt["la_kept"] += 1
                stt["la_prims"] += self.h.step - p0
                rec.update(result="kept")
                self.identity_log.append(rec)
                return {"track": self.om.held, "type": otype, "status": "held", "xyz": xyz, "state": None, "place": None,
                        "step": self.h.step, "conf": float(conf), "la_kept": True, "la_from": ctype}
            self._la_restore(f2, ctype, cand, xyz, place, rec)
        except _Fail:
            raise
        except Exception as e:                             # an inspection cut short: the record says so, memory is already right
            rec.setdefault("result", f"inspection incomplete ({type(e).__name__}: {e})")
            self.identity_log.append(rec)
            raise
        stt["la_prims"] += self.h.step - p0
        self.identity_log.append(rec)
        return None

    def _la_restore(self, f2, ctype, cand, xyz, place, rec):
        """Algorithm 4 line 6, bounded (Astra E3 Q6.7): at most LOOKALIKE_RESTORE_TRIES PutObject attempts, each from its own
        frame (the first is f'), at pixels where the stored point ``xyz`` (where the object was taken from) projects in the
        CURRENT view, then nearby support pixels; pixels on the held object are never used. After a successful put: memory
        on_put (anchor = the old point, exact position unknown), one look, and the landing is recorded as SEEN (a box of the
        type within LOOKALIKE_RESTORE_NEAR of the old point) or NOT VERIFIED. No drop fallback: when every attempt fails the
        step fails honestly with the object still held (memory says so)."""
        stt = self.identity_stats
        b0 = cand["bbox_xyxy"]
        mw, mh = 0.25 * (b0[2] - b0[0]) + 2, 0.25 * (b0[3] - b0[1]) + 2
        f, put, tries = f2, {"ok": False}, 0
        for k in range(LOOKALIKE_RESTORE_TRIES):
            if k:
                f = self._observe()
                stt["la_detector_calls"] += 1
            if f.depth is None:
                continue
            wh = _wh(f.depth)
            held = [d["bbox_xyxy"] for d in f.dets if d.get("bbox_xyxy") and d.get("point")             # SPEC-INHAND Q4: + the
                    and ((_depth(f.depth, d["point"]) or 9.0) < objmem.HELD_NEAR                          # shared predicate, for
                         or (d.get("type") not in SURFACE_TYPES and objmem.at_hand(f.depth, d)))]         # non-surface boxes only
            pr = objmem.project(xyz, f.pose, wh, self.om.fov)
            pts = []
            if pr is not None and 0 <= pr[0] < wh[0] and 0 <= pr[1] < wh[1]:
                c0 = (int(pr[0]), int(pr[1]))
                pts.append(c0)
                box = (c0[0] - mw, c0[1] - mh, c0[0] + mw, c0[1] + mh)
                pts += [tuple(q) for q in candidate_points(f.depth, box, n=5, agree=False, avoid=held, max_depth=REACH)]
            pts = [q for q in dict.fromkeys(pts) if not any(_inside(q, hb, 2) for hb in held)]
            if not pts:
                continue
            tries += 1
            put = self._act_at("PutObject", f, pts, place if place != "?" else None)
            if put.get("ok"):
                break
        rec["put_tries"] = tries
        if not put.get("ok"):
            stt["la_restore_failed"] += 1
            stt["la_prims"] += self.h.step - rec["step"]
            rec.update(result="could not restore (object still held)", put_why=put.get("why") or put.get("err"))
            self.identity_log.append(rec)
            self._ev(f"look-alike check: the {ctype} could NOT be put back after {tries} attempt(s)")
            how = self._la_put_elsewhere(ctype, rec)       # coordinator / Arash: a failed restore never fails the step
            if how is None:
                raise _Fail("wrong_target", f"could not put the inspected {ctype} down anywhere; it remains held",
                            under="lookalike_restore")
            return
        self._world_ver += 1
        self.om.on_put(True, place if place != "?" else None, xyz, ctype, step=self.h.step,
                       why="put back after a look-alike check (where it landed is unknown until it is seen again)")
        stt["la_put_back"] += 1
        fo = self._look()                                  # where did it land? (own camera)
        stt["la_detector_calls"] += 1
        self._la_obs = fo
        land = []
        if fo.depth is not None:
            whf = _wh(fo.depth)
            for d in fo.dets:
                if d.get("type") != ctype or not d.get("point"):
                    continue
                dq = _depth(fo.depth, d["point"])
                q3 = objmem.backproject(d["point"][0], d["point"][1], dq, fo.pose, whf, self.om.fov) if dq else None
                if q3 is not None and math.dist(q3, xyz) <= LOOKALIKE_RESTORE_NEAR:
                    land.append(tuple(q3))
        if land:
            self._la["checked"] += land                    # the restored object, seen again: not a new ambiguity point
            stt["la_put_seen"] += 1
            rec.update(result="put back (seen again near where it was)")
        else:
            stt["la_put_unseen"] += 1
            rec.update(result="put back (landing not verified)")
        self._ev(f"look-alike check: the {ctype} {rec['result']}")

    def _la_put_elsewhere(self, ctype, rec):
        """After a failed restoration: put the candidate down on any open surface in view and in reach (own detections,
        nearest first), else drop it (DropHandObject) as the last resort; memory records where (on_put / on_drop) and the
        search goes on. Counted (la_put_elsewhere / la_dropped) and logged. Returns "put" / "drop" / None."""
        stt = self.identity_stats
        f = self._observe()
        stt["la_detector_calls"] += 1
        how = None
        if f.depth is not None:
            held = [d["bbox_xyxy"] for d in f.dets if d.get("bbox_xyxy") and d.get("point")             # SPEC-INHAND Q4: + the
                    and ((_depth(f.depth, d["point"]) or 9.0) < objmem.HELD_NEAR                          # shared predicate, for
                         or (d.get("type") not in SURFACE_TYPES and objmem.at_hand(f.depth, d)))]         # non-surface boxes only
            surf = [d for d in f.dets if d.get("type") in LOOKALIKE_PUT_SURFACES and d.get("bbox_xyxy") and d.get("point")]
            surf.sort(key=lambda d: _depth(f.depth, d["point"]) or 9.0)
            for d in surf[:3]:
                pts = [tuple(q) for q in candidate_points(f.depth, d["bbox_xyxy"], n=5, agree=False, avoid=held, max_depth=REACH)]
                pts = [q for q in pts if not any(_inside(q, hb, 2) for hb in held)]
                if not pts:
                    continue
                put = self._act_at("PutObject", f, pts, d["type"])
                if put.get("ok"):
                    dq = _depth(f.depth, put["pt"]) if put.get("pt") else None
                    q3 = objmem.backproject(put["pt"][0], put["pt"][1], dq, f.pose, _wh(f.depth), self.om.fov) if dq else None
                    self.om.on_put(True, d["type"], q3, ctype, step=self.h.step,
                                   why=f"put down on a {d['type']} after a failed look-alike restoration (not where it was)")
                    if q3 is not None:
                        self._la["checked"].append(tuple(q3))
                    how = "put"
                    rec.update(result=f"could not restore; put down on a {d['type']} instead")
                    stt["la_put_elsewhere"] += 1
                    break
                if put.get("sent"):
                    f = self._observe()                   # a refused put: re-gate on a fresh frame
                    stt["la_detector_calls"] += 1
        if how is None:
            dr = self.h.act("DropHandObject")
            if dr["ok"]:
                self.om.on_drop(True, f.pose, step=self.h.step)
                how = "drop"
                rec.update(result="could not restore; dropped")
                stt["la_dropped"] += 1
        self._world_ver += 1
        self._ev(f"look-alike check: the {ctype} " + {"put": "put down elsewhere", "drop": "dropped"}.get(how, "could not be put down")
                 + " - the search goes on" * (how is not None))
        return how

    def _la_revisit(self, otype, info):
        """Algorithm 4 line 2 for the ambiguity points seen OUT OF REACH during the search: walk there (own map), reopen the
        container it was seen in (own observation of its place, if openable), face the point, look; ``_seen`` then
        re-evaluates what is in view. One pass over a SNAPSHOT of the queue (a point that is out of reach again is deferred,
        never revisited in the same pass), nearest first, within the cap and the search budget; what is left is counted as
        abandoned."""
        ctx, stt = self._la, self.identity_stats
        todo = list(ctx["pending"])
        ctx["pending"].clear()
        b = None
        while todo and ctx["n"] < LOOKALIKE_MAX and not self._search_spent() and b is None:
            x, z = self._pose_xz()
            p = min(todo, key=lambda q: math.hypot(q["xyz"][0] - x, q["xyz"][2] - z))
            todo.remove(p)
            if any(math.dist(p["xyz"], q) <= LOOKALIKE_SAME for q in ctx["checked"]):
                continue
            stt["la_revisits"] += 1
            p0 = self.h.step
            self._ev(f"look-alike check: going back to the {p['type']} seen out of reach at "
                     f"{tuple(round(v, 2) for v in p['xyz'])} while looking for the {otype}")
            try:                                           # navigation only: a failure of the check itself must propagate
                self._goto(p["xyz"])
                if p.get("place") in OPENABLE:
                    self._open_container(p["place"], p["xyz"])
                self._face(p["xyz"])
                self._stance_for_reach(p["xyz"])
            except _Fail as e:
                self._ev(f"look-alike check: cannot get there ({e.why})")
                stt["la_revisit_prims"] += self.h.step - p0
                continue
            stt["la_revisit_prims"] += self.h.step - p0
            b = self._seen(self._look(), otype)
        stt["la_abandoned"] += len(todo) + len(ctx["pending"])
        if b:
            info["found"] = "lookalike" if b.get("la_kept") else info.get("found")
        return b

    def _la_finish(self, otype, b):
        """E3: the pickup interaction ends with the re-evaluated object in the hand (no second pickup; memory already holds
        it: on_pickup + relabel_held in ``_revaluate``). Tidy as an ordinary pickup: the containers opened for THIS target
        first; only when none is linked, a search-opened container within 1 m of where it was taken."""
        self.verify.append("lookalike-specialist")
        if self.tidy:
            linked = [e for e in self.opened_by_me if e.get("target") == otype and e["why"] == "search-found"]
            near = [] if linked else [e for e in self.opened_by_me if e["why"] in ("search", "search-found")
                                      and math.dist(e["xyz"], b["xyz"]) < 1.0]
            for e in linked or near:
                self._tidy_close(e, f"(tidy, {otype} taken out)")
        return {"verify": "lookalike-specialist", "lookalike_from": b.get("la_from")}

    def _note_put(self, held, rec, obs, det, pt):
        """B1: remember where and from what view this agent put ``held`` down (own depth + pose only): the put
        pixel's surface point, the receptacle's half-extent from its box, and the frame itself. An aim reference,
        not the object's position -- the engine may settle it anywhere on the receptacle."""
        try:
            if obs is None or obs.depth is None or pt is None:
                return
            d = _depth(obs.depth, pt)
            wh = _wh(obs.depth)
            p0 = objmem.backproject(pt[0], pt[1], d, obs.pose, wh, self.om.fov) if d else None
            if p0 is None:
                return
            b = det.get("bbox_xyxy") if det else None
            half = 0.5 * max(b[2] - b[0], b[3] - b[1]) * d / objmem._focal(wh[1], self.om.fov) if b else PUT_HALF_MIN
            half = min(PUT_HALF_MAX, max(PUT_HALF_MIN, half))
            self.put_rec[held] = {"p0": tuple(p0), "half": float(half), "depth": obs.depth, "pose": obs.pose,
                                  "rec": rec, "step": self.h.step}
        except Exception as e:                              # a record is a convenience, never a reason to fail a put
            self._ev(f"put record not kept ({e})")

    def _seen_clean(self, otype) -> bool:
        """B1: with the water off, look at where this agent left ``otype`` and read its own state estimate: True only if
        it is SEEN there and reads clean. Sight only -- no success claim from a refused action."""
        rec = self.put_rec.get(otype)
        if rec is None:
            return False
        try:
            self._face(rec["p0"])
            obs = self._look()
            det = self._pick(obs, otype, rec["p0"])
        except _Soft:
            det = None
        if det is None:
            self._ev(f"{otype}: not seen in the sink with the water off - cannot say it is clean")
            return False
        v = state_of(det, "dirty")
        self._ev(f"{otype}: seen in the sink with the water off, it reads "
                 f"{'unreadable' if v is None else STATE_WORDS['dirty'][int(v)]}")
        return v is False

    def _blind_act(self, verb, prim, post, otype, st, obs, kw):
        """B1: the detector does not find ``otype`` where this agent put it (a wet or half-hidden dish in a sink).
        Aim from the agent's own record instead: newly occupied space since the put frame, then a declared grid over
        the receptacle region; each pixel passes the reach gate, and the action's success bool decides. A hypothesis,
        said so in the events -- it is not a re-sighting, so memory gets no sighting from it. None if nothing worked."""
        rec = self.put_rec.get(otype)
        if self.exhaustive and rec and self._crouched and target_blocked(rec["p0"], obs.depth, obs.pose, self.om.fov):
            # B1f: crouched, the counter lip hides the basin: the recorded point reads a much nearer surface. One stand
            # per call (my own body action), then everything is recomputed from a fresh look; reach is re-gated per pixel.
            self._ev(f"{otype}: the place I put it is hidden behind a nearer surface from this crouch - standing up to look")
            self._stand()
            obs = self._look()
        pts, note = blind_pixels(rec, obs.depth, obs.pose, self.om.fov)
        self._ev(f"{otype}: not detected where it was put ({rec['rec']}) - aiming from my own depth instead "
                 f"(hypothesis): {note}; {len(pts)} pixel(s) to try")
        tried = 0
        for pt in pts:
            o = obs if tried == 0 else self._look()
            tried += 1
            r = self._act_at(prim, o, [pt], otype, **kw)
            if not r["sent"]:
                continue
            self._ev(f"{prim} {otype} at {tuple(r['pt'])} (own-depth aim): {'ok' if r['ok'] else r['err']}")
            if not r["ok"]:
                continue
            if verb == "pickup":
                self.om.on_pickup(True, (st.get("b") or {}).get("track"), otype, step=self.h.step)
                self.put_rec.pop(otype, None)
                self._world_ver += 1
            self.verify.append("success-bool-own-aim")
            if post and verb != "pickup":
                v = self._recheck(otype, r["pt"], post[0])
                if v is not None and v != post[1]:
                    self._ev(f"{verb} {otype}: still {STATE_WORDS[post[0]][int(v)]} after a successful {prim}")
                    continue
            return {"verify": "success-bool-own-aim"}
        self._ev(f"{otype}: own-depth aim found nothing to act on ({tried} pixel(s) tried)")
        if verb == "pickup":
            rec["dead"] = True                              # give up on this record: the search takes over
        else:
            # the aim came from my own record, not from a view that a recovery move would improve: no retry cycle
            # (live 2026-10-09: three recoveries walked away from the sink and the faucet could not be turned off)
            raise _Fail("not_detected", f"no {otype} in view; aiming from my own depth found nothing to act on")
        return None

    def _fg_first(self, obs, det, pts) -> list:
        """B: when the chosen pixel's own-depth point is not actable from any own-map cell, the box's near coherent
        component (if any) goes first -- the pixel it would have sent is on what is behind the object."""
        p0 = pts[0] if pts else None
        dd = _depth(obs.depth, p0) if p0 else None
        xyz = objmem.backproject(p0[0], p0[1], dd, obs.pose, _wh(obs.depth), self.om.fov) if dd else None
        if xyz is not None and self._actable(xyz):
            return pts
        comps = foreground_components(obs.depth, det["bbox_xyxy"], k=2 if self._toggle_now else 1)
        fg = [q for tup in zip(*[c[:3] for c in comps]) for q in tup] if len(comps) > 1 else (comps[0] if comps else [])
        if not fg:
            return pts
        if len(comps) > 1:
            self._ev(f"{det.get('type')}: {len(comps)} near components in the box, at "
                     + ", ".join(f"{_depth(obs.depth, c[0]):.2f} m" for c in comps) + " (a toggle may need the next one)")
        self._ev(f"{det.get('type')}: the pixel {tuple(p0) if p0 else None} reads "
                 f"{'no depth' if dd is None else f'{dd:.2f} m'} (not actable from the own map) - the box's near "
                 f"component first ({_depth(obs.depth, fg[0]):.2f} m)")
        return [tuple(q) for q in fg] + [tuple(q) for q in pts if q is not None]

    def _recheck(self, otype, pt, axis):
        """Fresh look; the StateEstimator's ``axis`` verdict on the ``otype`` detection at/near the acted pixel."""
        obs2 = self._look()
        best, bd = None, None
        for d in obs2.dets:
            if d.get("type") == otype and d.get("bbox_xyxy"):
                b = d["bbox_xyxy"]
                dist = 0.0 if _inside(pt, b, 5) else math.hypot(pt[0] - (b[0] + b[2]) / 2, pt[1] - (b[1] + b[3]) / 2)
                if bd is None or dist < bd:
                    best, bd = d, dist
        return state_of(best, axis) if best is not None and bd <= 0.15 * obs2.depth.shape[0] else None

    def _note_open(self, place, xyz, why, front=None):
        """Own history: what I opened, its closed front (own measurement), and the cell + stance I opened it from
        (a close is sent from there: the door swung clear of the agent there)."""
        if xyz is None:
            return
        for e in [e for e in self.opened_by_me if e["type"] == place and math.dist(e["xyz"], xyz) < SAME_PLACE]:
            self.opened_by_me.remove(e)
            if e.get("door_key"):
                self.gm.unblock_cells(e["door_key"])
        cur, _ = hm.current_cell(self.h, self.gm)
        ent = {"type": place, "xyz": tuple(xyz), "why": why, "step": self.h.step,
               "front": front, "cell": cur, "crouched": bool(self._crouched),
               "edges0": set(self.gm.blocked_edges)}
        self.opened_by_me.append(ent)
        self._emit("open", rtype=place, xyz=tuple(xyz), ok=True, by_me=True, state_seen="open")
        if self.pose_first and front is not None and place in OPENABLE and hasattr(self.gm, "block_cells"):
            self._block_door(ent)

    def _note_closed(self, place, xyz=None):
        es = [e for e in self.opened_by_me if e["type"] == place]
        if not es:
            return
        e = min(es, key=lambda e: math.dist(e["xyz"], xyz) if xyz is not None else -e["step"])
        self.opened_by_me.remove(e)
        self._emit("close", rtype=place, xyz=tuple(e["xyz"]), ok=True, by_me=True, state_seen="closed")
        if e.get("door_key") and e.get("edges0") is not None and self.exhaustive:
            # moves refused next to this open door say nothing about the floor once it is shut: forget the edges
            # learned while it stood open that touch ITS swept floor (other evidence is kept -- Astra 6)
            k = e["door_key"]
            mine = set(self.gm.temp_blocked.get(k, ())) | set(getattr(self.gm, "temp_soft", {}).get(k, ()))
            new = {ed for ed in self.gm.blocked_edges - e["edges0"] if ed[0] in mine or ed[1] in mine}
            if new:
                self.gm.blocked_edges -= new
                self._ev(f"{place} closed: {len(new)} blocked edge(s) learned beside it while it stood open forgotten")
        if e.get("door_key") and hasattr(self.gm, "unblock_cells"):
            n = len(self.gm.unblock_cells(e["door_key"]))
            if n:
                self._ev(f"{place} closed: its {n} swept floor cell(s) unblocked")

    # -------------------------------------------------------------- the reach gate (every interaction send)
    def _gate_points(self, obs, det, first=(), openable: bool = False, n: int = 7, near_first: bool = False) -> list:
        """Candidate pixels for acting on ``det``: the caller's own choices first, then depth-agreeing interior
        pixels of the box away from other openables' boxes (containers) / smaller boxes on it, nearest the centre
        first, then any interior pixel. Pure."""
        b = det["bbox_xyxy"]
        avoid = []
        if openable:
            avoid = [o["bbox_xyxy"] for o in obs.dets if o is not det and o.get("type") in OPENABLE
                     and o.get("bbox_xyxy") and not (o["bbox_xyxy"][0] <= b[0] and o["bbox_xyxy"][1] <= b[1]
                                                     and o["bbox_xyxy"][2] >= b[2] and o["bbox_xyxy"][3] >= b[3])]
        out = []
        for p in list(first) + candidate_points(obs.depth, b, n=n, agree=True, avoid=avoid) \
                + candidate_points(obs.depth, b, n=n, agree=False, avoid=avoid):
            if p is not None and tuple(p) not in out:
                out.append(tuple(int(v) for v in p))
        if near_first:                                 # pose-first opens: the nearest surface points first (own depth)
            out.sort(key=lambda p: self._eye_dist(obs, p))
        return out[:GATE_POINTS]

    def _eye_dist(self, obs, p) -> float:
        d = _depth(obs.depth, p)
        xyz = objmem.backproject(p[0], p[1], d, obs.pose, _wh(obs.depth), self.om.fov) if d else None
        return math.dist(objmem._basis(obs.pose)[0], xyz) if xyz else 99.0

    def _act_at(self, prim, obs, pts, otype, front=None, mode: str = "front", limit: Optional[float] = None,
                swing=None, **kw) -> dict:
        """THE way decision code sends an interaction at a screen point: the first pixel of ``pts`` that passes
        ``reach_check`` on ``obs`` -- which must be the frame of the current pose (no primitive sent since) -- is
        sent; if none passes, nothing is sent. Returns the gateway result + {sent, why, gate, pt}."""
        if limit is None:
            limit = {"OpenObject": OPEN_REACH, "CloseObject": CLOSE_REACH}.get(prim, ACT_REACH)
        if obs is None or obs.step != self.h.step:
            return {"ok": False, "err": "gate", "sent": False, "why": "stale frame (moved since the look)", "pt": None}
        edges = front is not None and mode != "door"       # a swung door is often seen edge-on: thin by nature
        why = None
        passed = []
        cap = (self._aim_cap if (self._aim_cap and prim == "PickupObject" and self._aim_next)
               else (AIM_PER_TRY if self._aim_next else 3))
        for pt in pts:
            if pt is None:
                continue
            if self.exhaustive and self.h.holding and prim != "PutObject":
                dpt = _depth(obs.depth, pt)
                if dpt is not None and dpt < HAND_DEPTH and any(
                        d.get("type") == self.h.holding and d.get("bbox_xyxy") and _inside(pt, d["bbox_xyxy"], 2)
                        for d in obs.dets):
                    # while holding, a pixel this near the camera is the carried object itself (after_v6: the close
                    # after taking the bottle out hit the bottle in the hand: "WineBottle is not an Openable")
                    why = why or "pixel on the held object"
                    continue
            g = reach_check(obs.depth, obs.pose, self.om.fov, pt, limit, front, mode, otype or "", edges)
            if g["ok"]:
                passed.append((pt, g))
                if mode != "door" and len(passed) >= cap:
                    break                                  # open / other: the caller's order (central first)
            else:
                why = why or g["why"]
        if passed:
            # a close: a pixel on the swung door first (caller's order, central first), an interior pixel only as a
            # fallback; never a floor pixel that merely lies in the open box (baseline: "Floor is not an Openable"),
            # never the aperture's contents first (after_v1: "WineBottle is not an Openable")
            if mode == "door":
                passed = [t for t in passed if t[1].get("kind") == "door"] +                          [t for t in passed if t[1].get("kind") != "door"]
            r = None
            pk = self.pick_stats if prim == "PickupObject" else None
            if pk is not None:
                pk["calls"] += 1
                pk["offered_max"] = max(pk["offered_max"], len([p for p in pts if p is not None]))
            for i_, (pt, g) in enumerate(passed[:cap]):
                r = self.h.act(prim, tuple(pt), target_type=otype, swing=swing, **kw)
                if pk is not None:
                    pk["sends"] += 1
                    pk["next_point"] += int(i_ > 0)
                    pk["ok"] += int(bool(r["ok"]))
                if prim == "OpenObject" and r["ok"]:
                    # own frame from just before a successful open (same pose): what moved tells the door apart
                    self._pre_open = {"obs": obs, "xyz": g["xyz"], "step": self.h.step, "type": otype}
                self.gate_log.append({"prim": prim, "type": otype, "dist": g["dist"], "sent": True, "ok": r["ok"],
                                      "kind": g.get("kind")})
                if r["ok"] or not (r["err"] == "nothing-at-point" or (prim in TOGGLE_PRIMS and self.exhaustive)
                                   or (prim == "PickupObject" and self._aim_next and r["err"] == "refused")):
                    # the ray met no object there: the next passing pixel. A refused TOGGLE also goes on to the next
                    # one (B1, live 2026-10-09: the faucet's near-component pixel can land on the stream / basin;
                    # the toggle is idempotent in effect, so another pixel is safe)
                    break
            return {**r, "sent": True, "why": "", "gate": g, "pt": tuple(pt)}
        why = why or "no point to act at"
        self.gate_log.append({"prim": prim, "type": otype, "dist": None, "sent": False, "why": why})
        self._ev(f"{prim} {otype}: not sent - {why} (reach gate)")
        return {"ok": False, "err": "gate", "sent": False, "why": why, "pt": None}

    def _closer(self, xyz, known_height: bool = True) -> bool:
        """Walk nearer ``xyz`` (own map) after the reach gate refused every pixel as out of reach."""
        if self.exhaustive and xyz and known_height:
            return self._reach_move(xyz)
        self._stand()
        # any free cell NEARER than where the agent stands now (2026-10-09 re-run: a faucet at the back of a counter
        # has no free floor within 0.75 m, so the old 0.35-0.75 m annulus gave "no goal" and the faucet stayed on)
        r = hm.goto_near(self.h, self.gm, xyz, 0.35, 0.75, strafe=self.strafe)
        if not r["ok"] and xyz and r.get("why") == "no goal":
            here = self._hdist(xyz)
            if here - 0.2 > 0.75:
                r = hm.goto_near(self.h, self.gm, xyz, 0.25, here - 0.2, strafe=self.strafe)
        if r["ok"]:
            self._face(xyz)
        self._ev(f"move nearer the {'target' if xyz else '?'} for reach: {'ok' if r['ok'] else r['why']}")
        return bool(r["ok"])

    def _after_open(self, obs0, pt, ctype, front, label) -> str:
        """After a successful open, from the agent's own views: ``ok`` (the intended front reads open, or its cavity is
        seen), ``wrong`` (a DIFFERENT front, not open before, now reads open while the intended one still reads
        closed -- that one is closed again at once) or ``unverified`` (no evidence either way; kept in the own open
        list so tidy tries it). Astra: "intended not seen open" alone is unverified, never "wrong"."""
        ov = self._look()
        fc = front["c"] if front else None
        pfc = objmem.project(fc, ov.pose, _wh(ov.depth), self.om.fov) if fc else None
        mine = [d for d in ov.dets if d.get("type") == ctype and d.get("bbox_xyxy") and state_of(d, "open") is True
                and ((ctype == "Drawer" and pt is not None and _inside(pt, d["bbox_xyxy"], 10))   # slid out
                     or (pfc is not None and _inside(pfc, d["bbox_xyxy"], 5)))]
        if mine:
            return "ok"
        dv = self._det_at(ov, ctype, fc) if fc else None
        def cavity_seen():
            if front is None or ctype == "Drawer":
                return False
            chk = inside_check(ov.depth, ov.pose, self.om.fov, interior_probes(front, ctype), front)
            return chk["verdict"] in ("visible", "partly")    # its cavity is seen: it did open (state text lags)

        if dv is None or state_of(dv, "open") is not False:
            return "ok" if cavity_seen() else "unverified"
        # the intended front still reads CLOSED after a successful open: what opened instead?
        before = []
        for d in obs0.dets if obs0 is not None else ():
            if d.get("type") in OPENABLE and d.get("bbox_xyxy") and state_of(d, "open") is True:
                pa = front_patch(obs0.depth, d["bbox_xyxy"], obs0.pose, self.om.fov)
                if pa:
                    before.append(pa["c"])
        others = []
        for d in ov.dets:
            if d.get("type") not in OPENABLE or not d.get("bbox_xyxy") or state_of(d, "open") is not True:
                continue
            pa = front_patch(ov.depth, d["bbox_xyxy"], ov.pose, self.om.fov)
            if pa is None or math.hypot(pa["c"][0] - fc[0], pa["c"][2] - fc[2]) < 0.25:
                continue
            if any(math.dist(pa["c"], b) < 0.25 for b in before):
                continue                                   # it was open already before this open
            if d.get("type") == ctype and front is not None:
                nx, nz = front["n"]
                lo, hi, _, _ = front_extent(front)
                lat = pa["c"][0] * -nz + pa["c"][2] * nx
                s_ = (pa["c"][0] - fc[0]) * nx + (pa["c"][2] - fc[2]) * nz
                if s_ > 0.04 and lo - 0.15 <= lat <= hi + 0.15:
                    continue                               # may be this front's own swung door: no evidence
            others.append((math.hypot(pa["c"][0] - fc[0], pa["c"][2] - fc[2]), d, pa))
        if not others and cavity_seen():
            return "ok"
        if not others:
            # reads closed, but nothing else is seen open either: unverified (Astra), never called wrong on that alone
            # (after_v1: the fridge's state text read "closed" after it had opened; closing it there wasted it)
            self._ev(f"{label}: reads closed after a successful open, nothing else seen open -> unverified")
            return "unverified"
        _, d, pa = min(others, key=lambda t: t[0])
        self._ev(f"{label}: still reads closed after a successful open -- the point opened something else "
                 f"(a {d['type']} {math.dist(pa['c'], fc):.2f} m from it now reads open); closing that")
        cur, _ = hm.current_cell(self.h, self.gm)
        e = {"type": d["type"], "xyz": tuple(pa["c"]), "why": "mistake", "step": self.h.step, "front": pa,
             "cell": cur, "crouched": bool(self._crouched)}
        self.opened_by_me.append(e)                    # own history: tidy retries it if this close fails
        self._tidy_close(e, "opened by mistake")
        return "wrong"

    def _open_container(self, place, xyz, why: str = "search") -> bool:
        """Open a container seen closed (or unsure) near ``xyz``: one attempt, no budget (search / placed lookup).
        Exhaustive mode: inspect-then-open -- the front under the remembered point is measured, the agent walks to its
        inspection pose, re-identifies it there and only then opens it (gated); every mode: reach gate + verify."""
        obs = self._look()
        det = self._pick(obs, place, xyz, ("open", False))
        own = self._own_record(place, xyz) if self.exhaustive else None
        if own is None and (det is None or state_of(det, "open") is True):
            return False
        if own is not None:
            # this agent inspected that container before: its own front from the inspection pose (precise), and the
            # cell it looked from -- not a front re-estimated from wherever it stands now (after_v2 repeat runs)
            _, rec, vc, vpts = own
            front = front_patch_like(rec, vc, vpts)
        else:
            front = front_patch(obs.depth, det["bbox_xyxy"], obs.pose, self.om.fov)
        if self.exhaustive and front is not None:
            cur, _ = hm.current_cell(self.h, self.gm)
            cells = [c for _, c in self._insp_cells(front, place)]
            known = [tuple(c) for c in (own[1]["cells"] if own else []) if tuple(c) in self.gm.free]
            if cur not in cells[:6] and cur not in known:
                walked = bool(known) and self._walk_cell(known[0])
                if not walked and not self._walk_inspect(front, place, set()):
                    self._ev(f"open {place}: no inspection pose reachable for it - not opened")
                    return False
            if self._stance_for(front, place) == "crouch" or (place == "Drawer" and front["c"][1] < CROUCH_Y):
                self._crouch()
            self._face(front["c"])
            obs = self._look()
            det = self._det_at(obs, place, front["c"])
            p2 = front_patch(obs.depth, det["bbox_xyxy"], obs.pose, self.om.fov) if det else None
            on_front = False
            if det is not None:                        # the measured front point is on a surface where it should be
                pr = objmem.project(front["c"], obs.pose, _wh(obs.depth), self.om.fov)
                dz = _depth(obs.depth, (pr[0], pr[1])) if pr and _inside(pr, det["bbox_xyxy"], 2) else None
                on_front = dz is not None and abs(dz - pr[2]) <= 0.08
            near = p2 is not None and math.dist(p2["c"], front["c"]) <= 0.15
            if det is None or not (near or on_front) or state_of(det, "open") is True:
                self._ev(f"open {place}: not re-identified at its inspection pose - not opened")
                self._stand()
                return False
            if near and place != "Drawer":
                front = p2                             # drawers: keep the measured front (their boxes overlap)
        sw = 0.0 if state_of(det, "open") is False else None
        fpt = None
        if front is not None:
            pr = objmem.project(front["c"], obs.pose, _wh(obs.depth), self.om.fov)
            fpt = (int(round(pr[0])), int(round(pr[1]))) if pr else None
        pts = self._gate_points(obs, det, first=[p for p in (fpt, det.get("point")) if p], openable=True)
        r = self._act_at("OpenObject", obs, pts, place, front=front, swing=sw)
        if not r["sent"] and not self.exhaustive and front is not None and self._closer(front["c"]):
            obs = self._look()
            det = self._pick(obs, place, front["c"], ("open", False))
            if det is not None and state_of(det, "open") is not True:
                r = self._act_at("OpenObject", obs, self._gate_points(obs, det, first=[det.get("point")],
                                                                      openable=True), place, front=front, swing=sw)
        self._ev(f"open {place} to look inside: {'ok' if r['ok'] else (r['why'] or r['err'])}")
        if r["ok"]:
            self.opened_at[place] = tuple(xyz)
            where = xyz
            if self.exhaustive:                        # tidy needs the container's own front, not the target's spot
                where = front["c"] if front else self._xyz_of(det, xyz)
            self._note_open(place, where, why, front=front)
            if self._after_open(obs, r["pt"], place, front, place) == "wrong":
                self._note_closed(place, where)
                self.opened_at.pop(place, None)
                self._stand()
                return False
            if place == "Drawer":
                self._stand()                          # a drawer is looked into from above
            self._look()
        elif place == "Drawer" or not r["ok"]:
            self._stand()
        return r["ok"]

    def _own_record(self, place, xyz):
        """This agent's own inspection record of a ``place`` container at a remembered point ``xyz`` (e.g. where it
        saw the target inside it), best view first: ``(key, rec, view_c, view_pts)`` or None. Own history only."""
        if xyz is None:
            return None
        best = None
        dy_max = 0.12 if place == "Drawer" else 0.35
        for r in self.inspected:
            if r["type"] != place:
                continue
            for vc, vpts in r["views"]:
                dh, dy = math.hypot(xyz[0] - vc[0], xyz[2] - vc[2]), abs(xyz[1] - vc[1])
                if dh <= 0.6 and dy <= dy_max and len(vpts) >= 4:
                    key = (0 if r.get("verdict") == "found" else 1, round(dh + dy, 3))
                    if best is None or key < best[0]:
                        best = (key, r, vc, vpts)
        return best

    def _close_container(self, place, xyz, what: str = "after looking inside") -> bool:
        """Close a container this agent opened: one attempt, no budget. True only when a close was sent and ok."""
        obs = self._look()
        det = self._pick(obs, place, xyz, ("open", True))
        if det is None or state_of(det, "open") is False:
            return False
        front = self._open_front(place, xyz)
        r = self._act_at("CloseObject", obs, self._gate_points(obs, det, first=[det.get("point")], openable=True),
                         place, front=front, mode="door", swing=1.0)
        self._ev(f"close {place} {what}: {'ok' if r['ok'] else (r['why'] or r['err'])}")
        if r["ok"]:
            self.opened_at.pop(place, None)
            self._note_closed(place, xyz)
        return r["ok"]

    def _open_front(self, place, xyz):
        """The closed front (own measurement) of a container this agent opened near ``xyz``, else a point front."""
        es = [e for e in self.opened_by_me if e["type"] == place and e.get("front")]
        if xyz is not None and es:
            e = min(es, key=lambda e: math.dist(e["xyz"], xyz))
            if math.dist(e["xyz"], xyz) < 0.5:
                return e["front"]
        if xyz is None:
            return None
        return {"c": tuple(xyz), "n": (0.0, 1.0), "pts": [tuple(xyz)], "ymin": xyz[1] - 0.5, "ymax": xyz[1] + 0.5}

    # -------------------------------------------------------------- container inspection (exhaustive search)
    def _cam_y(self):
        p = self.h.pose()
        return float((p.get("camera") or {}).get("y", float(p["position"]["y"]) + 0.675))

    def _det_at(self, obs, ctype, xyz, pad=15):
        """The ``ctype`` detection at the pixel where ``xyz`` projects (identity first: box containing it, else the
        nearest box centre within 15 % of the frame). Never a silent nearest-of-type substitute."""
        wh = _wh(obs.depth)
        pr = objmem.project(xyz, obs.pose, wh, self.om.fov)
        if pr is None:
            return None
        ds = [d for d in obs.dets if d.get("type") == ctype and d.get("bbox_xyxy") and d.get("point")]
        inside = [d for d in ds if _inside(pr, d["bbox_xyxy"], pad)]
        if inside:
            if ctype == "Drawer":                      # a drawer column's boxes overlap: the one centred on it
                return min(inside, key=lambda d: (math.hypot(pr[0] - (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2,
                                                             pr[1] - (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2),
                                                  _area(d["bbox_xyxy"])))
            return min(inside, key=lambda d: _area(d["bbox_xyxy"]))
        best = min(ds, key=lambda d: math.hypot(pr[0] - (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2,
                                                 pr[1] - (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2), default=None)
        if best is None:
            return None
        b = best["bbox_xyxy"]
        return best if math.hypot(pr[0] - (b[0] + b[2]) / 2, pr[1] - (b[1] + b[3]) / 2) <= 0.15 * wh[1] else None

    def _container_point(self, obs, det, alt=0, front=None):
        """Where to act on a container front. First choice: the pixel where the front this agent measured (``front``,
        a 3D point from its own depth) projects, if the depth there agrees -- boxes of a drawer column overlap, the
        measured point does not. Else a depth-agreeing pixel of the box in no OTHER openable's box, nearest the box
        centre; ``alt`` picks the next one."""
        b = det["bbox_xyxy"]
        if front is not None and alt == 0:
            pr = objmem.project(front, obs.pose, _wh(obs.depth), self.om.fov)
            if pr is not None and _inside(pr, b, 2):
                d = _depth(obs.depth, (pr[0], pr[1]))
                if d is not None and abs(d - pr[2]) <= 0.08:
                    return (int(round(pr[0])), int(round(pr[1])))
        avoid = [o["bbox_xyxy"] for o in obs.dets if o is not det and o.get("type") in OPENABLE and o.get("bbox_xyxy")
                 and not (o["bbox_xyxy"][0] <= b[0] and o["bbox_xyxy"][1] <= b[1] and o["bbox_xyxy"][2] >= b[2]
                          and o["bbox_xyxy"][3] >= b[3])]
        pts = candidate_points(obs.depth, b, n=5, agree=True, avoid=avoid, max_depth=REACH)
        if not pts:
            return self._point(obs, det, alt)
        return pts[min(alt, len(pts) - 1)]

    def _overlap(self, pts, obs, det) -> float:
        """Share of a remembered front's 3D points that project into ``det``'s box at the depth they should have."""
        wh = _wh(obs.depth)
        n = hit = 0
        for p in pts:
            pr = objmem.project(p, obs.pose, wh, self.om.fov)
            if pr is None or not (0 <= pr[0] < wh[0] and 0 <= pr[1] < wh[1]):
                continue
            n += 1
            d = _depth(obs.depth, (pr[0], pr[1]))
            if d is not None and abs(d - pr[2]) <= 0.10 and _inside(pr, det["bbox_xyxy"], 4):
                hit += 1
        return hit / n if n else 0.0

    def _match(self, ctype, patch, obs=None, det=None):
        """Is this closed front one this agent already inspected (own action history)? ``("same", rec)`` needs
        near 3D fronts AND image overlap (or near-coincident fronts); ``("ambiguous", rec)`` = near but unproven;
        ``(None, None)`` = new. Geometry + view evidence only (Astra: never resolved identity)."""
        amb = None
        c = patch["c"]
        same_dy, near_dy = (DRAWER_SAME_DY, DRAWER_NEAR_DY) if ctype == "Drawer" else (ID_SAME_DY, ID_NEAR_DY)
        for r in self.inspected:
            if r["type"] != ctype:
                # another type's inspected front at the same place: this box is that front, mislabelled or seen
                # through (a cabinet box lying on the fridge door) -- same physical front, never opened again
                if r["type"] in OPENABLE and any(coincide(patch["pts"], vpts) >= ID_OVERLAP for _, vpts in r["views"]):
                    self._match_why = f"its surface is the {r['type']} {r['n']} front"
                    return "same", r
                continue
            for vc, vpts in r["views"]:
                dh, dy = math.hypot(c[0] - vc[0], c[2] - vc[2]), abs(c[1] - vc[1])
                if ctype in BIG_FRONTS and dh <= max(ID_NEAR_D, 0.6 * BIG_FRONTS[ctype]) and dy <= 0.45:
                    # one large appliance door seen from another angle (after_v2: the fridge opened twice, its two
                    # front estimates 0.3-0.5 m apart) -- a declared size prior, never an id
                    self._match_why = f"fronts {dh:.2f} m apart on one {ctype} front (size prior)"
                    return "same", r
                if dh <= ID_TIGHT_D and dy <= ID_TIGHT_DY:
                    self._match_why = f"fronts {dh:.2f} m apart, dy {dy:.2f}"
                    return "same", r
                if dh <= ID_SAME_D and dy <= same_dy:
                    co = coincide(patch["pts"], vpts)
                    ov = self._overlap(vpts, obs, det) if obs is not None and det is not None else 0.0
                    if co >= 0.5 or ov >= ID_OVERLAP:
                        self._match_why = f"fronts {dh:.2f} m apart, dy {dy:.2f}, surface shared {co:.0%}, image {ov:.0%}"
                        return "same", r
                if dh <= ID_NEAR_D and dy <= near_dy:
                    amb = r
        return ("ambiguous", amb) if amb else (None, None)

    def _candidates(self, obs, rank) -> list:
        """Closed / unsure openable containers in THIS frame within ENUM_RANGE that this agent has not inspected."""
        eye = objmem._basis(obs.pose)[0]
        out = []
        for d in obs.dets:
            t = d.get("type")
            if t not in OPENABLE or t not in rank or state_of(d, "open") is True or not d.get("bbox_xyxy"):
                continue
            if t == "Drawer" and any(o is not d and o.get("type") == "Drawer" and o.get("bbox_xyxy")
                                     and _area(d["bbox_xyxy"]) > 1.6 * _area(o["bbox_xyxy"])
                                     and _inside(((o["bbox_xyxy"][0] + o["bbox_xyxy"][2]) / 2,
                                                  (o["bbox_xyxy"][1] + o["bbox_xyxy"][3]) / 2), d["bbox_xyxy"])
                                     for o in obs.dets):
                continue                                   # a box spanning a stacked drawer's own box: the union
            pa = front_patch(obs.depth, d["bbox_xyxy"], obs.pose, self.om.fov)
            if pa is None or math.dist(eye, pa["c"]) > ENUM_RANGE:
                continue
            m, _ = self._match(t, pa, obs, d)
            if m == "same":
                continue
            if any(q["type"] == t and math.hypot(q["patch"]["c"][0] - pa["c"][0], q["patch"]["c"][2] - pa["c"][2])
                   <= 0.15 and abs(q["patch"]["c"][1] - pa["c"][1]) <= 0.10 for q in out):
                continue
            out.append({"type": t, "patch": pa, "ambiguous": m == "ambiguous", "rank": rank.index(t)})
        return out

    def _note_inside(self, rec, ob, pc, ctype):
        """E3: remember every movable type this agent's own detector found INSIDE the container while it stood open (a detection
        whose own-depth point lies within ``INSIDE_R`` m of the interior centre ``pc``). The record keeps the lead after the
        container is closed and the track drifts or is contradicted by views of the closed front."""
        try:
            wh = _wh(ob.depth)
            for d in ob.dets:
                t = d.get("type")
                if not t or t in SURFACE_TYPES or t in OPENABLE or not d.get("point") or t == ctype:
                    continue
                dd = _depth(ob.depth, d["point"])
                xyz = objmem.backproject(d["point"][0], d["point"][1], dd, ob.pose, wh, self.om.fov) if dd else None
                if xyz is not None and math.dist(xyz, pc) <= INSIDE_R:
                    st_ = rec.setdefault("seen_types", [])
                    if t not in st_:
                        st_.append(t)
        except Exception as e:                         # noqa: BLE001 - a lead is never worth a crash
            self.event_errors.append({"kind": "note_inside", "error": f"{type(e).__name__}: {str(e)[:120]}"})

    def _emit_inspection(self, rec, verdict, otype):
        """B1: one inspection event, enumerated fields from the own record + the best own inside check (own depth/pose)."""
        chk = None
        for c in rec.get("checks") or []:
            if c.get("support") is not None and (chk is None or c.get("frac", 0) >= chk.get("frac", 0)):
                chk = c
        x, z = self._pose_xz()
        c = rec["c"]
        ev_ = {"rtype": rec["type"], "xyz": tuple(c), "verdict": {"unsure": "unsure"}.get(verdict, verdict),
               "views": len(rec.get("views") or []), "crouched": bool(rec.get("stance") == "crouch"),
               "targets": [otype] if otype else [], "detected": [otype] if verdict == "found" and otype else [],
               "range_m": round(math.hypot(c[0] - x, c[2] - z), 3)}
        if chk is not None:
            ev_.update(frac=float(chk["frac"]), thirds=int(chk["thirds"]), support=int(chk["support"]))
        self._emit("inspection", **ev_)

    def _insp_summary(self, r) -> dict:
        p0 = (r.get("poses") or [None])[0]
        return {"n": r["n"], "type": r["type"], "verdict": r["verdict"], "reason": r["reason"], "opens": r["opens"],
                "step": r["step"], "planar": r["planar"], "pose": p0, "poses": len(r.get("poses") or []),
                "crouched": any(p["stance"] == "crouched" for p in r.get("poses") or [])}

    def _crouch(self) -> bool:
        if self._crouched:
            return True
        ok = self.h.move("Crouch")
        self._crouched = bool(ok)
        return bool(ok)

    def _stand(self):
        if getattr(self, "_crouched", False):
            self.h.move("Stand")
            self._crouched = False

    def _stance_for(self, patch, ctype=None) -> str:
        """crouch (a low DOOR: the shelf behind it is seen level), stand-down (counter height, and every drawer --
        a drawer is looked into from above; crouched, its rim hides the inside: measured 2026-10-07, 7 of 8 crouched
        low-drawer inspections had the inside hidden), stand-up (wall cabinets)."""
        y = patch["c"][1]
        if y > HIGH_Y:
            return "stand-up"
        return "crouch" if (y < CROUCH_Y and ctype != "Drawer") else "stand-down"

    def _log_pose(self, label, patch, rec, why="inspection pose"):
        """Own pose at an inspection, relative to the front this agent estimated (cell, yaw, stance, pitch, angle
        between its heading and the front normal) -- logged per container for the events / ticket / measurement."""
        p = self.h.pose()
        cur, _ = hm.current_cell(self.h, self.gm)
        yaw = float(p["rotation"]["y"])
        nx, nz = patch["n"]
        want = math.degrees(math.atan2(-nx, -nz))          # heading that looks straight into the front
        off = abs((yaw - want + 180.0) % 360.0 - 180.0)
        pose = {"cell": list(cur), "yaw": round(yaw), "stance": "crouched" if self._crouched else "standing",
                "pitch": round(float(p["horizon"] or 0.0)), "off": round(off), "planar": bool(patch.get("planar")),
                "dist": round(self._hdist(patch["c"]), 2), "why": why}
        rec.setdefault("poses", []).append(pose)
        self._ev(f"{label}: {why}: cell {tuple(cur)}, yaw {pose['yaw']}, {pose['stance']}, pitch {pose['pitch']:+d}, "
                 f"{pose['dist']:.2f} m, {pose['off']} deg off its front"
                 + ("" if pose["planar"] else " (front normal approximate)"))
        return pose

    def _reposition(self, kind, patch, door_side) -> str:
        """One ordered re-try for a better view inside: crouch, step back (one cell farther along the front normal),
        or side-step to the side the swung door is NOT on. Returns a description, or '' if it could not."""
        if kind == "crouch":
            return "crouch" if (not self._crouched and self._crouch()) else ""
        was = self._crouched
        cur, _ = hm.current_cell(self.h, self.gm)
        nx, nz = patch["n"]
        cx, cz = self.gm.world(cur)
        best, bscore = None, None
        for nb in self.gm.neighbours(cur):
            wx, wz = self.gm.world(nb)
            dx, dz = wx - cx, wz - cz
            along = dx * nx + dz * nz                        # + = away from the front
            lat = dx * -nz + dz * nx                         # lateral, along the front
            if kind == "back":
                sc = along
            elif door_side == 0:
                sc = abs(lat) - abs(along)
            else:
                sc = (-lat if door_side > 0 else lat) - abs(along)
            if sc > 0.05 and (bscore is None or sc > bscore):
                best, bscore = nb, sc
        if best is None:
            return ""
        self._stand()
        if not self._walk_cell(best, face=patch["c"]):
            return ""
        if was:
            self._crouch()
        return "step back" if kind == "back" else "side-step away from the door"

    def _walk_inspect(self, patch, ctype, used) -> bool:
        self._stand()
        cells = self._insp_cells(patch, ctype, exclude=used)
        if not cells:
            return False
        cur, _ = hm.current_cell(self.h, self.gm)
        best = cells[0][0]
        stop = self._stop
        if stop is not None and not used:
            # systematic route (2026-10-08): inspect from the stop -- or one cell beside it -- when that pose serves
            # this front within SERVE_TOL of its best cell (hard checks first: reach, front side, door sweep are
            # already in the cell list), instead of walking to a separate pose per container
            serve = serve_cells(self.gm, patch, cells)
            if cur in serve:
                stop["served"] += 1
                return True
            near = sorted((serve[n], n) for n in self.gm.neighbours(cur) if n in serve)
            if tuple(stop["cell"]) in serve and tuple(stop["cell"]) != cur:
                near = [(serve[tuple(stop["cell"])], tuple(stop["cell"]))] + near
            for _, n in near[:2]:
                if self._walk_cell(n, face=patch["c"]):
                    stop["shifted"] += 1
                    return True
            stop["split"] += 1
            self._ev(f"{ctype}: this stop does not serve it squarely - walking to its own inspection pose (split stop)")
        goals = [c for s, c in cells if s <= best + 0.25][:4]
        if cur in goals:
            return True
        self._stand()
        r = hm.walk_path(self.h, self.gm, self.gm.astar(cur, goals), goals=goals, retries=2, strafe=self.strafe,
                         face=patch["c"] if self.strafe else None)
        return bool(r["ok"])

    def _inspect(self, ctype, patch, otype, info, rec=None):
        """Inspect one closed container squarely: walk to an inspection pose on the own map (front side, height-
        dependent stand-off, within reach), face it, open it, aim at the interior and CHECK honestly that the inside
        is visible (``inside_check``); if not, re-position (bounded) and re-check; close it again unless the target
        is in it. Returns the target belief if seen, else None. Never spends the action-retry budget."""
        new = rec is None
        if new:
            n = 1 + sum(1 for r in self.inspected if r["type"] == ctype)
            rec = {"type": ctype, "n": n, "views": [(patch["c"], patch["pts"])], "c": patch["c"], "normal": patch["n"],
                   "planar": patch["planar"], "cells": [], "opens": 0, "verdict": None, "reason": "", "checks": [],
                   "step": self.h.step, "revisits": 0}
            self.inspected.append(rec)
            info.setdefault("inspections", []).append(None)
            rec["slot"] = len(info["inspections"]) - 1
        else:
            rec["revisits"] += 1
            if "slot" not in rec:
                info.setdefault("inspections", []).append(None)
                rec["slot"] = len(info["inspections"]) - 1
        label = f"{ctype} {rec['n']}"

        def done(verdict, reason, got=None):
            rec["verdict"], rec["reason"], rec["step"] = verdict, reason, self.h.step
            if verdict == "found":
                rec["target"] = otype                      # own history: what this agent saw in / at it
            info["inspections"][rec["slot"]] = self._insp_summary(rec)
            if self.on_event is not None:
                self._emit_inspection(rec, verdict, otype)
            mark = {"visible": "inside visible ✓", "partly": "inside partly visible ~",
                    "found": "target seen inside ✓"}.get(verdict, f"inside NOT visible ✗")
            self._ev(f"{label}: {mark}" + (f" ({reason})" if reason else "")
                     + (" -> unsure, kept as a candidate" if verdict == "unsure" else ""))
            return got

        if not self._walk_inspect(patch, ctype, set(rec["cells"]) if not new else set()):
            # no open from wherever a failed walk ended (2026-10-07, "fix: no far-away opens")
            cells = [c for _, c in self._insp_cells(patch, ctype)]
            cur, _ = hm.current_cell(self.h, self.gm)
            if cur not in cells:
                info["unreachable"].append(ctype)
                return done("unsure", "no inspection pose reached - not opened")
        cur, _ = hm.current_cell(self.h, self.gm)
        rec["cells"].append(cur)
        stance = self._stance_for(patch, ctype)
        rec["stance"] = stance
        if stance == "crouch":
            self._crouch()
        self._face(patch["c"])
        obs = self._look()
        got = self._seen(obs, otype)
        if got:
            self._log_pose(label, patch, rec)
            return done("found", "seen from the inspection pose", got)
        det = self._det_at(obs, ctype, patch["c"])
        if det is None:
            return done("unsure", "container not re-detected at the inspection pose")
        st0 = state_of(det, "open")
        p2 = front_patch(obs.depth, det["bbox_xyxy"], obs.pose, self.om.fov)
        if p2 is not None and len(p2["pts"]) >= VIS_MIN_SUPPORT and st0 is not True:
            here = math.dist(p2["c"], patch["c"]) <= 0.10   # else that box is not (only) this front (drawer columns)
            m, r2 = self._match(ctype, p2, obs, det)
            if m == "same" and r2 is not rec and not here and r2["type"] == ctype:
                # the box at this pose is an already-inspected front, not clearly this one: do not open blind
                return done("unsure", f"not re-identified at the inspection pose (box there is {ctype} {r2['n']})")
            if (m == "same" and r2 is not rec
                    and r2.get("verdict") in ("visible", "partly", "found", "unsure")):
                # seen squarely, this front is one already inspected (seen before from another side): merge, no open
                r2["views"].extend(rec["views"])
                if new:
                    self.inspected.remove(rec)
                    info["inspections"][rec["slot"]] = None
                self._ev(f"{label} is {r2['type']} {r2['n']} seen from another side (same front: "
                         f"{getattr(self, '_match_why', '')}): not opened again")
                return None
            # the box at this pose is the front about to be opened (here, or a not-yet-inspected front): it is the
            # better reference for the inside check and for recognising this container later
            patch = p2
            rec["views"].append((p2["c"], p2["pts"]))
            rec["planar"], rec["c"], rec["normal"] = p2["planar"], p2["c"], p2["n"]
        if st0 is not True and ctype == "Drawer" and patch["c"][1] < CROUCH_Y and not self._crouched:
            # a low drawer is OPENED crouched (standing, its front is beyond the open reach: baseline 1.24-1.55 m) and
            # looked into standing, from above
            self._crouch()
            self._face(patch["c"])
            obs = self._look()
            det = self._det_at(obs, ctype, patch["c"])
            if det is None:
                self._stand()
                return done("unsure", "container not re-detected crouched")
        opened = False
        edges0 = set(self.gm.blocked_edges)                # edges learned while this door stands open are its own
        if st0 is not True:
            for att in range(2):
                pt = self._container_point(obs, det, att, front=patch["c"] if ctype == "Drawer" else None)
                r = self._act_at("OpenObject", obs, self._gate_points(obs, det, first=[pt], openable=True), ctype,
                                 front=patch, swing=0.0 if st0 is False else None)
                self._ev(f"open {ctype} to look inside: {'ok' if r['ok'] else (r['why'] or r['err'])}")
                if r["ok"]:
                    opened = True
                    pt = r["pt"]
                    break
                if not r["sent"]:
                    return done("unsure", f"not opened: {r['why']}")
                if att == 0:
                    d0 = self._hdist(patch["c"])
                    if d0 < 0.7:                           # own geometry says it is close: step back for clearance
                        cur, _ = hm.current_cell(self.h, self.gm)
                        back = [nb for nb in self.gm.neighbours(cur)
                                if math.hypot(*(a - b for a, b in zip(self.gm.world(nb), (patch["c"][0],
                                                                                          patch["c"][2])))) > d0]
                        if back and self._walk_cell(max(back, key=lambda nb: math.hypot(
                                *(a - b for a, b in zip(self.gm.world(nb), (patch["c"][0], patch["c"][2])))))):
                            self._face(patch["c"])
                    obs = self._look()
                    det = self._det_at(obs, ctype, patch["c"])
                    if det is None:
                        break
            if not opened:
                return done("unsure", "it did not open")
            open_cell, _ = hm.current_cell(self.h, self.gm)    # where the door swung clear of the agent
            open_crouched = self._crouched
            self._note_open(ctype, patch["c"], "search", front=patch)
            # verify (own views): does THIS front now read open? a different front that opened is closed at once
            if self._after_open(obs, pt, ctype, patch, label) == "wrong":
                self._note_closed(ctype, patch["c"])
                self._stand()
                return done("unsure", "wrong thing opened (closed again)")
            rec["opens"] += 1
            info["opened"].append(ctype)
            self.opened_at[ctype] = tuple(patch["c"])
            if ctype == "Drawer" and self._crouched:
                self._stand()                              # opened crouched, looked into from above
        extent = None
        if ctype == "Drawer":                              # how far the drawer came out, from its own depth
            o1 = self._look()
            d1 = self._det_at(o1, ctype, patch["c"], pad=40)
            p1 = front_patch(o1.depth, d1["bbox_xyxy"], o1.pose, self.om.fov) if d1 else None
            if p1:
                nx, nz = patch["n"]
                extent = max((p[0] - patch["c"][0]) * nx + (p[2] - patch["c"][2]) * nz for p in p1["pts"])
        probes = interior_probes(patch, ctype, extent)
        if not probes:
            got = None
            verdict, reason = "unsure", "no interior estimate"
        else:
            pc = tuple(sum(p[i] for p in probes) / len(probes) for i in range(3))
            face_off = 0.0 if ctype == "Drawer" else CAVITY.get(ctype, 0.12)
            best = None
            plan = (["back", "side"] if self._crouched or stance == "stand-up" or ctype == "Drawer"
                    else ["crouch", "back", "side"])
            plan = plan[:MAX_REPOSITION]
            door_side = 0
            for k in range(len(plan) + 1):
                if k >= 1:
                    desc = self._reposition(plan[k - 1], patch, door_side)
                    if not desc:
                        continue
                    cur, _ = hm.current_cell(self.h, self.gm)
                    if cur not in rec["cells"]:
                        rec["cells"].append(cur)
                    self._ev(f"{label}: re-position for a view inside ({best['reason']}): {desc}")
                self._face(pc)
                self._log_pose(label, patch, rec, "inspection pose" if k == 0 else "re-check pose")
                ob = self._look()
                self._note_inside(rec, ob, pc, ctype)
                got = self._seen(ob, otype)
                if got:
                    for e in self.opened_by_me:
                        if e["type"] == ctype and math.dist(e["xyz"], patch["c"]) < SAME_PLACE:
                            e["why"], e["target"] = "search-found", otype
                    rec["checks"].append({"verdict": "found"})
                    return done("found", "", got)
                doors = [d["bbox_xyxy"] for d in ob.dets if d.get("type") == ctype and d.get("bbox_xyxy")
                         and not _inside(objmem.project(pc, ob.pose, _wh(ob.depth), self.om.fov) or (-99, -99),
                                         d["bbox_xyxy"])]
                chk = inside_check(ob.depth, ob.pose, self.om.fov, probes, patch, doors, face_off)
                ppc = objmem.project(pc, ob.pose, _wh(ob.depth), self.om.fov)
                if doors and ppc is not None:              # which side the swung door is on, in the image
                    dxs = [((b[0] + b[2]) / 2.0) - ppc[0] for b in doors]
                    door_side = 1 if sum(dxs) > 0 else -1
                chk["step"] = self.h.step
                rec["checks"].append(chk)
                order = {"visible": 3, "partly": 2, "not-visible": 1, "unsure": 0}
                if best is None or order[chk["verdict"]] > order[best["verdict"]]:
                    best = chk
                if chk["verdict"] in ("visible", "partly"):
                    break
            verdict, reason = best["verdict"], best["reason"]
            if verdict in ("not-visible", "unsure"):
                verdict = "unsure"
        if opened:
            cur, _ = hm.current_cell(self.h, self.gm)
            if cur != open_cell or self._crouched != open_crouched:   # close from where (and how) it opened
                self._walk_cell(open_cell, face=patch["c"])
                if open_crouched:
                    self._crouch()
                self._face(patch["c"])
            ent = next((e for e in self.opened_by_me if e["type"] == ctype and math.dist(e["xyz"], patch["c"]) < 0.05),
                       None) or {"type": ctype, "xyz": tuple(patch["c"]), "why": "search", "front": patch,
                                 "cell": open_cell, "crouched": open_crouched}
            res = self._tidy_close(ent, "after looking inside")
            self._stand()
            (info["closed"] if res in ("closed", "already") else info["close_failed"]).append(ctype)
            if res in ("closed", "already") and self.gm.blocked_edges != edges0:
                # moves refused while the door stood open say nothing about the floor once it is shut again
                self._ev(f"{label}: {len(self.gm.blocked_edges - edges0)} blocked edge(s) learned while it was open "
                         "forgotten (door closed again)")
                self.gm.blocked_edges = set(edges0)
        return done(verdict, reason)

    def _step_away(self, xyz) -> bool:
        """Walk one free cell farther from ``xyz`` (door clearance), on the own map. True if it moved."""
        cur, _ = hm.current_cell(self.h, self.gm)
        d0 = math.hypot(*(a - b for a, b in zip(self.gm.world(cur), (xyz[0], xyz[2]))))
        opts = sorted(((-math.hypot(*(a - b for a, b in zip(self.gm.world(n), (xyz[0], xyz[2])))), n)
                       for n in self.gm.neighbours(cur)), key=lambda t: t[0])
        for negd, n in opts:
            if -negd >= d0 - 0.01 and self._walk_cell(n):
                return True
        return False

    def _tidy_close(self, entry, what) -> str:
        """Close one container this agent opened: walk near if needed, face, close; if refused or not seen, ONE
        retry from a slightly different angle (exhaustive mode: two retries, the first one cell farther away for
        door clearance, and a close is verified by re-detecting the front). Never spends the retry budget, never
        fails the step. Returns ``closed`` | ``already`` (seen closed) | ``failed``."""
        place, xyz = entry["type"], entry["xyz"]
        front = entry.get("front") or self._open_front(place, xyz)
        res = "failed"
        back = False
        far = False
        tried = False
        for att in range(3 if self.exhaustive else 2):
            try:
                cell = entry.get("cell")
                cur, _ = hm.current_cell(self.h, self.gm)
                if att == 0 and self.exhaustive and cell is not None and tuple(cell) in self.gm.free:
                    # close from the cell (and stance) it was opened from: the door swung clear of the agent there
                    if cur != tuple(cell) or self._crouched != bool(entry.get("crouched")):
                        if cur != tuple(cell) and not self._walk_cell(tuple(cell), face=xyz):
                            self._goto(xyz)
                        if entry.get("crouched"):
                            self._crouch()
                elif att == 0 and not (NEAR_OK[0] <= self._hdist(xyz) <= NEAR_OK[1]):
                    self._goto(xyz)
                if att >= 1 and far:
                    back = False
                    self._closer(xyz)                    # the gate found every door pixel out of reach
                elif att == 1 and self.exhaustive and self._hdist(xyz) < 1.0:
                    back = self._step_away(xyz)          # a refused close is often the door meeting the agent
                here = entry.get("cell") is not None and tuple(hm.current_cell(self.h, self.gm)[0]) == tuple(entry["cell"])
                if not (att == 0 and self.exhaustive and here and self._last_look == self.h.step
                        and self._in_view_now(xyz)):
                    # (console: no re-aim when the front is already well in view -- the turn back after looking
                    # into the interior was a left/right reversal at every container)
                    self._face(xyz, PITCH_ALT * min(att, 1), YAW_ALT * (1 if att == 1 else -1 if att == 2 else 0))
            except _Fail as e:
                self._ev(f"tidy: cannot get to the {place} to close it ({e.why})")
                break
            obs = self._look()
            det = self._pick(obs, place, xyz, ("open", True))
            if det is None and self.exhaustive and not tried:
                # not detected, but own history says it stands open where I opened it: aim around where its front
                # projects (the door gate still demands a point on that door / opening)
                pr = objmem.project(xyz, obs.pose, _wh(obs.depth), self.om.fov)
                w_, h_ = _wh(obs.depth)
                if pr is not None and 0 <= pr[0] < w_ and 0 <= pr[1] < h_:
                    rr = 0.12 * h_
                    det = {"type": place, "state": "?", "bbox_xyxy": (max(0, pr[0] - rr), max(0, pr[1] - rr),
                                                                       min(w_ - 1, pr[0] + rr), min(h_ - 1, pr[1] + rr)),
                           "point": (int(pr[0]), int(pr[1])), "own": True}
            if det is not None and state_of(det, "open") is False and att >= 1 and (tried or not self.exhaustive):
                res = "already"
                self._note_closed(place, xyz)
                self.opened_at.pop(place, None)
                break
            if det is not None and (state_of(det, "open") is not False or (self.exhaustive and not tried)):
                # own history says I opened it and have not closed it since: a "closed" reading is not enough to skip
                # the close (after_v2: the fridge read "closed" while open and was left open); a shut one refuses it
                tried = True
                r = self._act_at("CloseObject", obs, self._gate_points(obs, det, first=[self._point(obs, det, att)],
                                                                       openable=True),
                                 place, front=front, mode="door", swing=1.0)
                far = (not r["sent"]) and "out of reach" in r["why"]
                self._ev(f"close {place} {what}{' (retry, other angle' + (', stepped back)' if back else ')') if att else ''}: "
                         f"{'ok' if r['ok'] else (r['why'] or r['err'])}")
                if r["ok"] and self.exhaustive:            # verify: the front I aimed at no longer reads open
                    ob2 = self._look()
                    d2 = self._pick(ob2, place, xyz, ("open", True))
                    for _settle in range(2):
                        # a reading right after a successful close can lag (the frame / state text of the moving
                        # door): wait a frame and read again before calling it still open
                        if not (d2 is not None and state_of(d2, "open") is True):
                            break
                        self.h.move("Pass")
                        ob2 = self._look()
                        d2 = self._pick(ob2, place, xyz, ("open", True))
                    if d2 is not None and state_of(d2, "open") is True and _inside(
                            objmem.project(xyz, ob2.pose, _wh(ob2.depth), self.om.fov) or (-99, -99),
                            d2["bbox_xyxy"], 20):
                        self._ev(f"close {place}: still reads open after the close -- trying again")
                        continue
                if r["ok"]:
                    res = "closed"
                    self.opened_at.pop(place, None)
                    self._note_closed(place, xyz)
                    break
            elif det is None:
                self._ev(f"tidy: {place} not in view to close{' (retry)' if att else ''}")
        if entry.get("crouched") or self._crouched:
            self._stand()
        self.tidy_log.append({"type": place, "why": entry.get("why"), "what": what, "result": res})
        return res

    def _needed_later(self, entry) -> bool:
        """A later plan step still uses this container (or the object found in it): leave it open for that step."""
        for st in self.upcoming or ():
            a = ALIASES.get(str(st.get("action") or "").lower(), str(st.get("action") or "").lower())
            if (TIDY_SWITCH_CLOSED and a == "toggle_on" and entry["type"] in SWITCH_ON_CLOSED
                    and st.get("object") == entry["type"] and not st.get("to")):
                continue                                  # E1: that step needs the door CLOSED, not open
            if entry["type"] in (st.get("object"), st.get("to")) and a != "close_all":
                return True
            if entry.get("target") and st.get("object") == entry["target"]:
                return True
        return False

    def _tidy_step_end(self):
        for e in [e for e in self.opened_by_me if e["why"] != "explicit"]:
            if e not in self.opened_by_me:
                continue
            if self._needed_later(e):
                self._ev(f"tidy: leaving the {e['type']} open - a later step needs it")
                continue
            self._tidy_close(e, "(tidy, end of step)")

    def tidy_up(self, keep_explicit: bool = True) -> dict:
        """End of a run: close every container this agent opened and left open (explicit ``open`` steps excepted)."""
        self._reset_step()
        todo = [e for e in self.opened_by_me if not (keep_explicit and e["why"] == "explicit")]
        kept = [e["type"] for e in self.opened_by_me if keep_explicit and e["why"] == "explicit"]
        for e in todo:
            if e in self.opened_by_me:
                self._tidy_close(e, "(tidy, end of run)")
        return {"closed": [t["type"] for t in self.tidy_log if t["result"] in ("closed", "already")],
                "failed": [t["type"] for t in self.tidy_log if t["result"] == "failed"], "kept_open": kept,
                "log": list(self.tidy_log), "events": list(self.events), "prims": self.h.step - self._step0}

    # -------------------------------------------------------------- pose first, then open (2026-10-08)
    def _door_rec(self, ctype, c):
        tol = 0.12 if ctype == "Drawer" else 0.35
        best = None
        for d in self.doors:
            if d["type"] == ctype:
                dd = math.dist(d["c"], c)
                if dd <= tol and (best is None or dd < best[0]):
                    best = (dd, d)
        return best[1] if best else None

    def _door_for(self, front, ctype):
        """The door model for this front: own record (seen open before) else the prior (hinge unknown)."""
        d = self._door_rec(ctype, front["c"])
        return door_model(front, ctype, prev=d["model"] if d else None)

    def _insp_cells(self, patch, ctype, exclude=()):
        """Inspection cells; with pose-first, outside the door's (estimated) sweep. If none is left, the unfiltered
        list (said so in the events: the door may meet the agent)."""
        if not self.pose_first:
            return inspection_cells(self.gm, patch, self._cam_y(), ctype, exclude=exclude)
        cells = inspection_cells(self.gm, patch, self._cam_y(), ctype, exclude=exclude,
                                 model=self._door_for(patch, ctype))
        if cells:
            return cells
        cells = inspection_cells(self.gm, patch, self._cam_y(), ctype, exclude=exclude)
        if cells:
            self._ev(f"{ctype}: no inspection pose outside its door's estimated sweep - using the best one inside it")
        return cells

    def _block_door(self, ent):
        """A door this agent just opened: observe it (own depth), update its own door record, and temporarily block
        the floor cells it sweeps (the agent's own cell kept free so it can always back off)."""
        place, front = ent["type"], ent["front"]
        ent["corrected"] = ent["toward"] = False
        pts, leaf = [], None
        try:
            ob = self._look()
            pre = self._pre_open
            if pre is not None and pre["step"] == self.h.step and pre["type"] == place and place != "Drawer":
                f2, dpts = opened_front(pre["obs"].depth, ob.depth, ob.pose, self.om.fov, pre["xyz"], place)
                eye = objmem._basis(ob.pose)[0]
                near = [q for q in dpts if math.hypot(q[0] - eye[0], q[2] - eye[2]) <= 1.2]
                if f2 is None and len(near) >= 40:
                    # nothing revealed behind the estimated front, but a large surface came NEARER: the leaf swung
                    # towards the agent -- the estimate took another face (e.g. the fridge's side) for the door
                    leaf = leaf_front(near, eye, front["c"], place)
                    if leaf is not None:
                        f2 = leaf
                        ent["toward"] = True
                if f2 is not None:
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, f2["n"][0] * front["n"][0]
                                                                + f2["n"][1] * front["n"][1]))))
                    dc = math.hypot(f2["c"][0] - front["c"][0], f2["c"][2] - front["c"][2])
                    if ang > FRONT_OK or dc > 0.30:
                        self._ev(f"{place}: the face that opened is not the one estimated ({ang:.0f} deg, {dc:.2f} m "
                                 f"off) - front re-measured from {f2['from']}")
                        ent["corrected"] = True
                    front = ent["front"] = f2
                    ent["xyz"] = tuple(f2["c"])
                if leaf is not None:
                    pts = [q for q in near]
            if not pts:
                boxes = [d["bbox_xyxy"] for d in ob.dets if d.get("type") == place and d.get("bbox_xyxy")]
                pts = door_points(ob.depth, ob.pose, self.om.fov, boxes, front, place)
        except Exception as e:                         # no frame / depth: the prior model
            self._ev(f"{place}: door not measured ({type(e).__name__}) - the prior sweep")
            pts = []
        rec = self._door_rec(place, front["c"])
        m = door_model(front, place, pts, prev=rec["model"] if rec else None)
        if m["kind"] == "slide":
            # B4: the height band and lateral span of what actually came out (the points that moved in front of the closed
            # plane). A stack's closed-front estimate can merge neighbours (its band spans two drawers); the pulled-out
            # one is where these points are
            cx_, (nx_, nz_) = front["c"], front["n"]
            mv = [q for q in pts if (q[0] - cx_[0]) * nx_ + (q[2] - cx_[2]) * nz_ >= 0.04]
            if len(mv) >= 3:
                m["moved"] = {"y": (min(q[1] for q in mv), max(q[1] for q in mv)), "n": len(mv)}
            elif rec and rec["model"].get("moved"):
                m["moved"] = rec["model"]["moved"]
        if rec is None:
            self.doors.append({"type": place, "c": tuple(front["c"]), "front": front, "model": m})
        else:
            rec.update(model=m, front=front, c=tuple(front["c"]))
        cur, _ = hm.current_cell(self.h, self.gm)
        key = f"{place}@{front['c'][0]:.2f},{front['c'][2]:.2f}"
        sweep = swept_cells(self.gm, m)
        inside = swept(m, *self._pose_xz())
        size = f"drawer out {m['out']:.2f} m" if m["kind"] == "slide" else f"radius {m['r']:.2f} m"
        if m["kind"] == "slide" or not self.exhaustive:
            cells = self.gm.block_cells(key, sweep, keep=[cur])
            ent["door_key"] = key
            self._ev(f"{place} door open: {len(cells)} floor cell(s) in its sweep blocked on the own map "
                     f"({m['why']}, {size})" + (" - the agent stands inside it (own cell kept free)" if inside else ""))
            return m
        # A (2026-10-09): the still, open leaf is the obstacle (hard, fitted to its own door points + LEAF_PAD); the
        # rest of the sweep only matters while the door moves (soft: routes avoid it when a detour exists; a close is
        # still sent from the pose it was opened from); no observed leaf -> no hard cells, the whole sweep soft
        hard = leaf_cells(self.gm, m, pts)
        before_n = len(grid_dist(self.gm, cur))
        cells = self.gm.block_cells(key, hard, keep=[cur], soft=sweep - hard)
        if hard and len(grid_dist(self.gm, cur)) < 0.5 * before_n:
            # the fitted leaf would cut the agent off from most of the floor it could reach a moment ago -- but the
            # door just opened WITHOUT meeting the agent, so the estimate next to its body is the least reliable
            # (live 2026-10-09, groceries): the cells beside it become soft, then all of them if still enclosed
            nb = {(cur[0] + dx, cur[1] + dz) for dx, dz in hm.DIRS}
            for hard2, what in ((set(hard) - nb, "the cells beside the agent"), (set(), "all of them")):
                cells = self.gm.block_cells(key, hard2, keep=[cur], soft=sweep | set(hard) - hard2)
                if len(grid_dist(self.gm, cur)) >= 0.5 * before_n:
                    break
            self._ev(f"{place}: the fitted leaf would enclose the agent (it opened without meeting it) - "
                     f"{what} kept soft instead of hard")
            hard = set(cells)
        ent["door_key"] = key
        ent["soft_n"] = len(sweep - set(hard))
        self._ev(f"{place} door open: {len(cells)} floor cell(s) at its leaf blocked"
                 + (" (fitted to its own door points)" if hard else " (leaf not observed)")
                 + f", {len(sweep - set(cells))} more of its sweep avoided where a detour exists (soft) "
                 f"({m['why']}, {size})" + (" - the agent stands inside the sweep" if inside else ""))
        return m

    def _own_front(self, rec, near):
        """This agent's own closed-front estimate of a ``rec`` container near ``near`` (xyz): an open entry of its
        own, a door record, or an inspection record -- else (None, None). Own history only."""
        def d(c):
            if near is None:
                return 0.0
            # B4: drawers stack -- the neighbours above / below share the place (x, z), so for them the height counts
            return math.dist(c, near) if rec == "Drawer" else math.hypot(c[0] - near[0], c[2] - near[2])
        lim = PUT_DRAWER_SAME if rec == "Drawer" else 0.8     # B4: a drawer's neighbour in the stack is another instance
        es = [e for e in self.opened_by_me if e["type"] == rec and e.get("front") and d(e["front"]["c"]) <= lim]
        if es:
            e = min(es, key=lambda e: d(e["front"]["c"]))
            return e["front"], e
        ds = [x for x in self.doors if x["type"] == rec and x.get("front") and d(x["c"]) <= lim]
        if ds:
            return min(ds, key=lambda x: d(x["c"]))["front"], None
        own = self._own_record(rec, near) if near is not None else None
        if own is not None:
            _, r_, vc, vpts = own
            return front_patch_like(r_, vc, vpts), None
        return None, None

    def _face_of(self, front, obs, rec):
        """A non-planar front seen at a corner: split it into its two faces and take as the door the one with free
        floor squarely in front of it on the own map (an appliance's side usually stands against a counter or a
        wall) -- a declared prior, checked again from the pose and when it opens (what moved / the leaf)."""
        if front is None or front.get("planar") or rec == "Drawer":
            return front
        faces = split_faces(front, objmem._basis(obs.pose)[0])
        if not faces:
            return front
        cam = self._cam_y() + (CROUCH_DROP if self._crouched else 0.0)
        n_cells = [len(put_cells(self.gm, f, cam, rec, None, square=0.8)) for f in faces]
        i = max(range(2), key=lambda j: (n_cells[j], front_extent(faces[j])[1] - front_extent(faces[j])[0]))
        self._ev(f"{rec}: two faces seen (a corner); the one with free floor in front of it is taken as its door "
                 f"({n_cells[i]} vs {n_cells[1 - i]} square cells)")
        return faces[i]

    def _put_pose(self, rec, used=(), front_hint=None):
        """Choose the put pose of an openable ``rec`` BEFORE it is opened and walk there (door closed). Returns
        {front, model, entry, cell} or None (no closed front this agent can measure: the caller's plain flow)."""
        if front_hint is not None:
            front, ent = front_hint, None
        else:
            b = self._locate_put(rec)
            front, ent = self._own_front(rec, b["xyz"])
            if front is not None:
                self._ev(f"put into the {rec}: front taken from {'an open entry' if ent is not None else 'a door / inspection record'} "
                         f"at ({front['c'][0]:.2f}, {front['c'][1]:.2f}, {front['c'][2]:.2f}) for the candidate at "
                         f"({b['xyz'][0]:.2f}, {b['xyz'][1]:.2f}, {b['xyz'][2]:.2f})")
        if front is None:
            walked = False
            if self._hdist(b["xyz"]) > 2.2:
                self._goto(b["xyz"])
                walked = True
            self._stand()
            self._face(b["xyz"])
            obs = self._look()
            det = self._pick(obs, rec, b["xyz"], ("open", False))
            if det is None and not walked and self.exhaustive:
                # not seen from where it stands (occluded / an odd angle): walk near it once and look again
                # (after_v5: the remembered fridge was 1-2 m away behind a counter corner; three retries were spent
                # on the same unseen view without a single step)
                self._goto(b["xyz"])
                self._face(b["xyz"])
                obs = self._look()
                det = self._pick(obs, rec, b["xyz"], ("open", False))
            if det is None:
                if self.exhaustive and b.get("track"):
                    self._set_aside(b, f"{rec} not seen at its remembered place")
                raise _Soft("not_detected", f"no {rec} in view")
            if state_of(det, "open") is True:
                self._ev(f"put into the {rec}: it is open and this agent has no closed-front estimate of it "
                         "- the plain put flow")
                return None
            front = front_patch(obs.depth, det["bbox_xyxy"], obs.pose, self.om.fov)
            if front is None:
                raise _Soft("not_detected", f"no depth on the {rec} front")
            front = self._face_of(front, obs, rec)
        model = self._door_for(front, rec)
        cur, _ = hm.current_cell(self.h, self.gm)
        if ent is not None and ent.get("cell") is not None and tuple(ent["cell"]) not in used:
            cells = [(0.0, tuple(ent["cell"]))]           # it is open already: the pose it was opened from
        else:
            self._stand()
            cells = put_cells(self.gm, front, self._cam_y(), rec, model, exclude=used)
            if not cells:
                cells = put_cells(self.gm, front, self._cam_y(), rec, model, exclude=used, square=0.7)
            if not cells:
                cells = put_cells(self.gm, front, self._cam_y(), rec, model, exclude=used, least_swept=True)
                if cells:
                    self._ev(f"put into the {rec}: no pose outside its door's estimated sweep ({model['why']}) - "
                             "the square one farthest from it")
        if not cells:
            stg = {}
            put_cells(self.gm, front, self._cam_y(), rec, model, exclude=used, square=0.7, least_swept=True,
                      stages=stg)
            dist = grid_dist(self.gm, cur)
            stg["connected_free"] = sum(1 for c in self.gm.free if c in dist)
            self._ev(f"no put pose for the {rec}: candidate cells by filter {json.dumps(stg)}; excluded poses "
                     f"{len(used)}; holding {self.h.holding}; front from {front.get('from') or 'own depth'}")
            raise _Soft("unreachable", f"no put pose for the {rec} on the own map (reach + sweep; filters {stg})")
        best = cells[0][0]
        goals = [c for sc, c in cells if sc <= best + 0.25][:4]
        if cur not in goals:
            self._stand()
            r = hm.walk_path(self.h, self.gm, self.gm.astar(cur, goals), goals=goals, retries=2, strafe=self.strafe)
            if not r["ok"] and len(cells) > len(goals):
                more = [c for _, c in cells if c not in goals][:6]
                r = hm.walk_path(self.h, self.gm, self.gm.astar(hm.current_cell(self.h, self.gm)[0], more),
                                 goals=more, retries=2, strafe=self.strafe)
            if not r["ok"]:
                raise _Soft("unreachable", f"no walk to a put pose for the {rec} ({r['why']})")
        cell, _ = hm.current_cell(self.h, self.gm)
        sw = swept(model, *self._pose_xz())
        self._ev(f"put pose for the {rec}: cell {tuple(cell)}, {self._hdist(front['c']):.2f} m from its front, "
                 + ("outside" if not sw else "INSIDE") + f" its door's sweep ({model['why']}) - chosen before opening")
        return {"front": front, "model": model, "entry": ent, "cell": cell}

    def _locate_put(self, rec):
        """B3: the receptacle instance to put into -- like ``_locate``, but never an instance of this type that already
        put the held object OUTSIDE it (the engine says ok and drops it on the surface its volume overlaps; a pixel
        cannot fix an instance). Beliefs within PUT_BAD_R of such an instance are skipped; if no other instance is
        known the same one is tried again (the corrections cap bounds it, and a landing outside never reports ok)."""
        held = self.h.holding
        bad = [(x, z) for r_, h_, x, z in self.put_bad_recs if r_ == rec and h_ == held]
        # B4: instances that refused the puts (any held object). Identity = the remembered instance's place in 3D, so a
        # drawer in a stack does not shadow the ones above and below it
        ref = [(x, y, z, h_) for r_, x, y, z, h_ in self.put_refused_recs if r_ == rec]
        if not bad and not ref:
            return self._locate(rec)
        bel = [b for b in self._rank(self.om.lookup(rec))
               if all(math.hypot(b["xyz"][0] - x, b["xyz"][2] - z) > PUT_BAD_R for x, z in bad)
               and not any(math.dist(b["xyz"], (x, y, z)) <= r_ for x, y, z, r_ in ref)]
        if bel:
            self._ev(f"put into the {rec}: skipping {len(bad)} instance(s) that put the {held} outside it and {len(ref)} that "
                     f"refused the puts; next candidate at ({bel[0]['xyz'][0]:.2f}, {bel[0]['xyz'][1]:.2f}, {bel[0]['xyz'][2]:.2f})")
            return bel[0]
        self._ev(f"put into the {rec}: no other {rec} is known - trying the same one again (bounded by the corrections cap)")
        return self._locate(rec)

    def _pose_and_open(self, rec, why, used=()):
        """Pose first, then open: walk to the put pose, face, re-identify the closed front there, open it (reach gate),
        verify. If what moved shows the estimate took another face for the door and this pose does not suit the face
        that opened, it is closed again from here and the pose is planned once more for that face (own geometry).
        Returns the pose dict (front updated) or None (plain flow). Raises _Soft on a failed attempt."""
        hint = None
        for it in range(4):
            pose = self._pose_and_open_once(rec, why, used, hint, recheck=it < 2)
            if pose is None or not pose.get("replan"):
                return pose
            hint = pose["front"]
        raise _Soft("unreachable", f"no put pose suits the {rec} face that opened")

    def _pose_and_open_once(self, rec, why, used, hint, recheck=True):
        pose = self._put_pose(rec, used, front_hint=hint)
        if pose is None:
            return None
        front = pose["front"]
        if self._stance_for(front, rec) == "crouch" or (rec == "Drawer" and front["c"][1] < CROUCH_Y):
            self._crouch()
        self._face(front["c"])
        obs = self._look()
        det = self._det_at(obs, rec, front["c"], pad=25)
        if det is None:
            raise _Soft("not_detected", f"no {rec} in view at the put pose")
        if pose["entry"] is not None or state_of(det, "open") is True:
            if pose["entry"] is None:
                self._ev(f"{rec}: already reads open at the put pose")
            return pose
        p2 = self._face_of(front_patch(obs.depth, det["bbox_xyxy"], obs.pose, self.om.fov), obs, rec)
        if p2 is not None and p2.get("planar") and rec != "Drawer":
            ang = math.degrees(math.acos(max(-1.0, min(1.0, p2["n"][0] * front["n"][0] + p2["n"][1] * front["n"][1]))))
            dc = math.hypot(p2["c"][0] - front["c"][0], p2["c"][2] - front["c"][2])
            cur, _ = hm.current_cell(self.h, self.gm)
            cam = self._cam_y() + (CROUCH_DROP if self._crouched else 0.0)
            ok_here = cur in [c for _, c in put_cells(self.gm, p2, cam, rec, self._door_for(p2, rec))]
            if (recheck and not ok_here and front.get("from") in (None, "a corner split", "the put-pose view")
                    and (ang > 10.0 or dc > 0.15)):
                # the closer view from the pose shows the front elsewhere than the estimate it was planned from (an
                # oblique first look sees only part of it) and this cell is not square to it: plan again from what is
                # seen here -- the door is still closed, nothing has moved
                self._ev(f"{rec}: the front seen from the put pose differs from the estimate ({ang:.0f} deg, "
                         f"{dc:.2f} m) and this cell is not square to it - planning the pose again before opening")
                self._stand()
                return {"front": dict(p2, **{"from": "the put-pose view"}), "replan": True}
            if math.dist(p2["c"], front["c"]) <= 0.15 and ang <= FRONT_OK:
                front = pose["front"] = p2                 # the square view's own front is the better reference
        pr = objmem.project(front["c"], obs.pose, _wh(obs.depth), self.om.fov)
        fpt = (int(round(pr[0])), int(round(pr[1]))) if pr else None
        pts = self._gate_points(obs, det, first=[p for p in (fpt, det.get("point")) if p], openable=True,
                                near_first=True)
        r = self._act_at("OpenObject", obs, pts, rec, front=front, swing=0.0)
        if not r["sent"]:
            raise _Soft("unreachable", f"open not sent from the put pose: {r['why']}")
        if not r["ok"]:
            # B3 r2: a refused open has one cause the agent can check by looking: it was already open (AI2-THOR refuses
            # to open an open door, and a door seen from the side or far away may read 'closed' / unknown). Look
            # again from the same spot; if it now reads open, the put goes on from here -- nothing was refused for a
            # reason that needs a retry. Anything else stays a refusal.
            if self.exhaustive and rec != "Drawer":
                try:
                    obs2 = self._look()
                    det2 = self._det_at(obs2, rec, front["c"], pad=25)
                except _Soft:
                    det2 = None
                if det2 is not None and state_of(det2, "open") is True:
                    self._ev(f"open {rec}: refused, and it reads open when looked at again - it was already open")
                    return pose
            if self.h.holding:
                # AI2-THOR usually allows it; a refusal says only that it failed (no cause is observed -- Astra)
                self._ev(f"open {rec}: refused while holding the {self.h.holding} (cause not observed)")
                raise _Soft("action_refused", f"open refused while holding the {self.h.holding}")
            raise _Soft("action_refused", r["err"] or "open refused")
        self._ev(f"OpenObject {rec} at {tuple(r['pt'])}: ok (from the put pose)")
        if self._after_open(obs, r["pt"], rec, front, rec) == "wrong":
            raise _Soft("wrong_target", "the point opened a different front (closed again)")
        self.opened_at[rec] = tuple(front["c"])
        self._note_open(rec, front["c"], why, front=front)
        ent = next((e for e in self.opened_by_me if e["type"] == rec and e.get("step") == self.h.step), None)
        pose["entry"] = ent
        if ent is not None and ent.get("corrected"):
            # the estimate took another face (e.g. the fridge's side) for the door: does this pose suit the face that
            # opened? If not: close it from here (the door swung clear of the agent here) and plan again for it
            front = pose["front"] = ent["front"]
            pose["model"] = self._door_for(front, rec)
            cur, _ = hm.current_cell(self.h, self.gm)
            cam = self._cam_y() + (CROUCH_DROP if self._crouched else 0.0)
            cells = put_cells(self.gm, front, cam, rec, None, is_open=True, square=0.7)
            if cur not in [c for _, c in cells] or ent.get("toward"):
                self._ev(f"{rec}: this pose does not suit the face that opened"
                         + (" (its door swung towards the agent)" if ent.get("toward") else "")
                         + " - closing it from here, planning the pose again")
                res = self._tidy_close(ent, "(to stand where it opens)")
                self._stand()
                if res not in ("closed", "already"):
                    raise _Soft("action_refused", "could not close it to re-plan the pose")
                return {"front": front, "replan": True}
        return pose

    def _put_points(self, obs, det, front, rec, alt):
        """Receptacle pixels for a put: inside the opening (behind the closed front plane, within its extent) first,
        away from smaller boxes of other things; ``alt`` rotates the order. Pure on the frame."""
        dpix = rec == "Drawer" and PUT_DRAWER_PIX_ON       # B4 switch: the drawer-specific pixel choice (ablation)
        b = det["bbox_xyxy"]
        avoid = [o["bbox_xyxy"] for o in obs.dets if o is not det and o.get("bbox_xyxy")
                 and o.get("type") != det.get("type") and _area(o["bbox_xyxy"]) < _area(b)]
        wh = _wh(obs.depth)
        c, (nx, nz) = front["c"], front["n"]
        lo, hi, ymin, ymax = front_extent(front)
        mid, half = (lo + hi) / 2.0, (hi - lo) / PATCH_SPAN / 2.0
        inner, other, weak = [], [], []
        # B3 r2: a pixel of THIS opening at height y0 was owned by something that is not the cavity (the put landed
        # outside: the surface above the opening shares its volume, and the engine acts on the object that owns the
        # pixel). Whatever sits at or above that height in the same opening is probably owned by it too -> demoted
        # behind the pixels below (a preference, not an exclusion; a pixel below is tried first)
        caps = [q for q in self.put_in_caps if math.hypot(q[0] - c[0], q[1] - c[2]) < PUT_BAD_R]
        self._put_geo = {}
        # B4: how far this drawer is seen pulled out (its tray lies between the closed plane and that); the prior if unseen
        d_rec = self._door_rec("Drawer", c) if dpix else None
        out_d = (d_rec["model"].get("out") if d_rec else None) or DRAWER_OUT
        mv_ = d_rec["model"].get("moved") if d_rec else None
        if mv_:
            ymin, ymax = mv_["y"]                    # the opening is where the drawer was seen to come out (see _note_open)
        cand = candidate_points(obs.depth, b, n=9, agree=False, avoid=avoid, max_depth=REACH)
        if dpix:
            # B4: the detection box of a stacked drawer can cover only closed fronts (the neighbours', on the closed plane)
            # while the pulled-out tray lies in front of that plane, lower in the image. Aim a few pixels at the tray
            # itself from the agent's own geometry (closed plane + seen pull-out, extent of the front); each one is judged
            # below by the depth it actually returns, like every other candidate
            lat_c, ux, uz = _lat(front, c[0], c[2]), -nz, nx
            have = set(cand)
            self._n_tray_aim = 0
            for sf in (0.5, 0.8):
                for lf in (-0.6, -0.3, 0.0, 0.3, 0.6):
                    for yf in (0.1, 0.4):
                        l_, s2 = mid + lf * half - lat_c, sf * out_d
                        pr = objmem.project((c[0] + nx * s2 + ux * l_, ymin + yf * (ymax - ymin), c[2] + nz * s2 + uz * l_),
                                            obs.pose, wh, self.om.fov)
                        if pr and 0 <= pr[0] < wh[0] and 0 <= pr[1] < wh[1]:
                            q = (int(round(pr[0])), int(round(pr[1])))
                            d_ = _depth(obs.depth, q)
                            if (q not in have and d_ is not None and d_ <= REACH
                                    and not any(_inside(q, a_) for a_ in avoid)):
                                have.add(q)
                                cand.append(q)
                                self._n_tray_aim += 1
        for p in cand:
            d = _depth(obs.depth, p)
            xyz = objmem.backproject(p[0], p[1], d, obs.pose, wh, self.om.fov) if d else None
            if not xyz:
                continue
            if any(math.dist(xyz, (q[0], q[1], q[2])) < q[3] for q in self.put_in_reject):
                continue                               # B3: a landing spot already judged wrong (3D, never returns)
            s_ = (xyz[0] - c[0]) * nx + (xyz[2] - c[2]) * nz
            lat = _lat(front, xyz[0], xyz[2])
            eye = objmem._basis(obs.pose)[0]
            dist = math.dist(eye, xyz)
            # B3: the opening's own height range, plus only the depth error of THIS pixel's range: a pixel on the
            # surface above the door (the counter the cabinet stands under) is outside the cavity, not 15 cm of slack
            tol_y = PUT_IN_SIGMAS * (DEPTH_ERR[0] + DEPTH_ERR[1] * dist)
            ok = ((s_ <= -0.02 if rec != "Drawer" else 0.0 <= s_ <= out_d + tol_y) and mid - half - 0.05 <= lat <= mid + half + 0.05
                  and ymin - 0.15 <= xyz[1] <= ymax + 0.15)
            strict = ok and ymin - tol_y <= xyz[1] <= ymax + tol_y
            capped = strict and any(xyz[1] >= q[2] - (DEPTH_ERR[0] + DEPTH_ERR[1] * dist) for q in caps)
            deep = 0 if (rec != "Drawer" and s_ <= -PUT_DEEP) else 1        # well inside the cavity first (see PUT_DEEP)
            # B4: a put needs a surface that faces UP. The front panel of a drawer is vertical and its top rim borders the
            # next drawer / the slab above: the engine acts on the object that owns the pixel and refuses ("openable
            # receptacle is closed") when that is the neighbour. So the tray's up-facing pixels come first (drawers only:
            # a cabinet's pixels were settled in B3). Below the front's top edge (by the depth error of the pixel) first:
            # the rim itself is the neighbour's border
            up = 0
            if strict and not capped and dpix:
                up = 0 if _horizontal(obs.depth, obs.pose, self.om.fov, p, wh) else 1
                if xyz[1] > ymax - tol_y:
                    up += 2          # at or above the front's top edge (within the depth error): the rim, not the tray
                if s_ < 2.0 * tol_y:
                    up += 4          # on the closed front plane within the combined depth + front-fit error (2 x one pixel's
                                     # tolerance): that is where the neighbours' faces lie, not the pulled-out tray
            if capped:
                weak.append(((-1, 0, dist), p))                             # inside, but at/above a height that failed
            else:
                (inner if strict else (weak if ok else other)).append(((up, deep if strict else 0, dist), p))
            self._put_geo[tuple(p)] = (round(s_, 3), round(lat, 3), round(xyz[1], 3), round(ymin, 3), round(ymax, 3),
                                       round(dist, 2), "capped" if capped else "strict" if strict else ("loose" if ok else "outside"),
                                       "up" if (strict and up == 0) else "plane" if (strict and up >= 4) else "rim" if (strict and up >= 2) else "")
        pts = [p for _, p in sorted(inner)] + [p for _, p in sorted(weak)] + [p for _, p in sorted(other)]  # nearest first
        if dpix and inner:
            # B4: the alternate point stays inside the best group (same quality); rotating the whole list would promote
            # closed-plane / rim pixels ahead of every good one
            best = sorted(inner)[0][0][0]
            n_best = sum(1 for key, _ in inner if key[0] == best)
            if n_best > 1:
                k = (max(0, alt) * 3) % n_best
                return pts[k:n_best] + pts[:k] + pts[n_best:]
        k = min(max(0, alt) * 3, max(0, len(pts) - 1))
        return pts[k:] + pts[:k]

    def _put_openable(self, rec):
        """Put the held object into an openable ``rec``: pose first (outside the door's sweep, square, in reach),
        open from there, put WITHOUT walking (pixel / pitch / crouch only); tidy closes the door from there.
        Retries: 1 = alternate pixel + pitch, 2 = crouch / stand + another pixel, 3 = close it, another pose, reopen."""
        if not self.h.holding:
            raise _Fail("action_refused", "hand is empty")
        held = self.h.holding
        st = {"alt": 0, "dpitch": 0.0, "pose": None, "used": set()}
        k = 0
        while True:
            try:
                self._cap()
                if st["pose"] is None:
                    pose = self._pose_and_open(rec, "explicit" if self._step_action == "open" else "put",
                                               used=st["used"])
                    if pose is None:
                        return self._put_inner(rec)
                    st["pose"] = pose
                front = st["pose"]["front"]
                cav = CAVITY.get(rec, 0.12)
                ic = (front["c"][0] - front["n"][0] * cav, front["c"][1], front["c"][2] - front["n"][1] * cav)
                self._face(ic, st["dpitch"])
                obs = self._look()
                det = (self._det_at(obs, rec, ic, pad=30) or self._det_at(obs, rec, front["c"], pad=30)
                       or self._pick(obs, rec, ic))
                if det is None:
                    raise _Soft("not_detected", f"no {rec} in view at the put pose")
                pts_ = self._put_points(obs, det, front, rec, st["alt"])
                self._ev_put_geo(pts_, front, rec)
                if (self.exhaustive and PUT_IN_VERIFY_ON and st.get("free", 0) < PUT_FREE_MAX and pts_
                        and not any(self._put_geo.get(tuple(p), ("",) * 8)[6] in ("strict", "capped", "loose")
                                    and self._put_geo.get(tuple(p), ("",) * 8)[7:] != ("plane",)
                                    for p in pts_)):
                    # no pixel of the opening in this view (the front seen from here lies elsewhere / too close): a
                    # put at a pixel outside it would be a put on whatever owns that pixel -> look again first
                    raise _Soft("no_inside", f"no pixel of the {rec} opening in view from here", free=True)
                r = self._act_at("PutObject", obs, pts_, rec)
                if not r["sent"]:
                    raise _Soft("unreachable", f"not sent: {r['why']}")
                self._ev(f"PutObject {held} -> {rec} at {tuple(r['pt'])}: {'ok' if r['ok'] else r['err']} "
                         "(from the put pose, no walk after opening)")
                if not r["ok"]:
                    if self.exhaustive and PUT_IN_VERIFY_ON:
                        # B4: the engine said no. Count it against THIS instance; the second refusal in a row blames
                        # the instance (not the pixel): remember it for the episode and go to another one of the type
                        key = (round(front["c"][0], 1), round(front["c"][2], 1))
                        cnt = st.setdefault("refused", {})
                        cnt[key] = cnt.get(key, 0) + 1
                        if cnt[key] >= PUT_REFUSE_MAX:
                            # identity = the remembered instance (belief) nearest this front, in 3D: drawers stack
                            near_ = sorted(self.om.lookup(rec), key=lambda b_: math.dist(b_["xyz"], front["c"]))
                            ref_c = tuple(near_[0]["xyz"]) if near_ and math.dist(near_[0]["xyz"], front["c"]) <= PUT_BAD_R                                 else tuple(front["c"])
                            self.put_refused_recs.append((rec, ref_c[0], ref_c[1], ref_c[2], PUT_SAME_INST))
                            self._ev(f"{rec} at ({front['c'][0]:.2f}, {front['c'][2]:.2f}) refused {cnt[key]} puts - "
                                     f"remembered as refusing; another {rec} next")
                            raise _Soft("refusing_instance", r["err"], free=True)
                    raise _Soft("action_refused", r["err"])
                if self.exhaustive and PUT_IN_VERIFY_ON:
                    self._verify_put_in(held, rec, front, r, obs, st)       # may pick it back up and raise _Soft
                self.om.on_put(True, rec, tuple(front["c"]), otype=held, step=self.h.step)
                self._world_ver += 1
                self.sub.append({"verb": "put_in", "object": rec, "ok": True, "verify": "success-bool"})
                self.verify.append("success-bool")
                return {"verify": "success-bool"}
            except _Soft as e:
                self.sub.append({"verb": "put_in", "object": rec, "ok": False, "cause": e.cause})
                if e.free and st.get("free", 0) < PUT_FREE_MAX:
                    st["free"] = st.get("free", 0) + 1
                    self._cap()
                    self._ev(f"not charged to the retry budget ({e.cause}; {st['free']}/{PUT_FREE_MAX} such attempts)")
                else:
                    self._spend(e)
                k += 1
                if st["pose"] is None or k >= 3 or e.cause in ("wrong_instance", "refusing_instance"):
                    # no pose reached / the third retry: close what it opened from here, another pose, reopen
                    if st["pose"] is not None:
                        ent = st["pose"].get("entry")
                        if ent is not None and ent in self.opened_by_me:
                            self._tidy_close(ent, "(another put pose next)")
                    cur, _ = hm.current_cell(self.h, self.gm)
                    same = st["pose"] is None and e.cause == "action_refused" and k == 1
                    if not same:                           # a refused open: once more from the same pose first
                        st["used"].add(cur)
                    st.update(pose=None, alt=0, dpitch=0.0)
                    self._stand()
                    self._ev(f"recover {k}: " + ("the same pose again (the open was refused)" if same
                                                 else "another put pose (this one excluded)"))
                elif k == 1:
                    st.update(alt=1, dpitch=PITCH_ALT)
                    self._ev(f"recover 1: alternate point, pitch {PITCH_ALT:+.0f} (no walk)")
                else:
                    if self._crouched:
                        self._stand()
                    else:
                        self._crouch()
                    st.update(alt=2, dpitch=0.0)
                    self._ev(f"recover 2: {'crouched' if self._crouched else 'stood up'}, another point (no walk)")

    def _ev_put_geo(self, pts, front, rec):
        """B3 diagnostics (own depth only): where the first put pixels lie against the opening, by tier."""
        try:
            tiers = {}
            for p in pts:
                g = self._put_geo.get(tuple(p))
                if g:
                    tiers[g[6]] = tiers.get(g[6], 0) + 1
            first = [f"{tuple(p)}: {self._put_geo[tuple(p)][6]}{(' ' + self._put_geo[tuple(p)][7]) if self._put_geo[tuple(p)][7:] and self._put_geo[tuple(p)][7] else ''} y={self._put_geo[tuple(p)][2]:.2f} s={self._put_geo[tuple(p)][0]:+.2f}"
                     for p in pts[:3] if tuple(p) in self._put_geo]
            geo0 = next(iter(self._put_geo.values()), (0, 0, 0, front["ymin"], front["ymax"]))
            ss = sorted(g[0] for g in self._put_geo.values())
            if ss:
                self._ev(f"{rec} put candidates: {len(ss)}, depth-derived s (m in front of the closed plane) min {ss[0]:+.2f} "
                         f"median {ss[len(ss) // 2]:+.2f} max {ss[-1]:+.2f}; tray-aimed {getattr(self, '_n_tray_aim', 0)}")
            self._ev(f"{rec} put pixels by tier {tiers or '-'}; opening y {geo0[3]:.2f}..{geo0[4]:.2f} m; first: "
                     + "; ".join(first))
        except Exception:
            pass

    def _seen_of_type(self, obs, held):
        """B3: every detection of ``held`` in this frame (not the one in hand) with its centre and base from the agent's
        OWN depth -> [(centre xyz, base xyz, range)]. A box bottom pixel may sample the background, so the base is only
        ever evidence AGAINST a put, never for one."""
        wh = _wh(obs.depth)
        out = []
        for det in obs.dets:
            b = det.get("bbox_xyxy")
            if det.get("type") != held or not b:
                continue
            cen = det.get("point") or ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
            dc = _depth(obs.depth, cen)
            if dc is None or dc < objmem.HELD_NEAR:
                continue                                     # in the hand / no depth
            mid = objmem.backproject(cen[0], cen[1], dc, obs.pose, wh, self.om.fov)
            base_pt = ((b[0] + b[2]) // 2, max(b[1], b[3] - 3))
            db = _depth(obs.depth, base_pt)
            base = objmem.backproject(base_pt[0], base_pt[1], db, obs.pose, wh, self.om.fov) if db else mid
            if mid:
                out.append((mid, base or mid, dc))
        return out

    def _verify_put_in(self, held, rec, front, r, obs_put, st):
        """B3: a PutObject the engine called ok on an OPENABLE receptacle is not yet a put INTO it. The engine puts the
        object on whatever object owns the pixel (the CounterTop whose volume overlaps the cabinet's, in the live
        failure) and says ok, then places it anywhere on THAT receptacle -- not at the pixel. So: look for the object
        again (this frame, then toward the put point), ignoring detections that were already there before the put
        (another instance), and judge each NEW one from its own depth: inside the opening's volume (not in front of its
        plane, within its width and height, plus the combined depth error of the reading) = in; clearly outside =
        contrary evidence -> pick it back up, exclude that landing spot in 3D, raise _Soft so the retry loop tries
        another pixel / pose. Nothing new seen = 'unverified' (success bool kept and said so). Nothing here turns a
        result into a claim."""
        pt = r.get("pt")
        d0 = _depth(obs_put.depth, pt) if pt else None
        p0 = objmem.backproject(pt[0], pt[1], d0, obs_put.pose, _wh(obs_put.depth), self.om.fov) if d0 else None
        pre = [m for m, _, _ in self._seen_of_type(obs_put, held)]      # instances already there before the put
        pre += [tuple(b_["xyz"]) for b_ in self.om.lookup(held) if b_.get("status") == "trusted" and b_.get("xyz")
                and not any(math.dist(b_["xyz"], m) < PUT_IN_SAME for m in self.put_in_landed)]   # our own, picked up again
        found = []
        # this frame, toward the put point, and toward the surface just above the opening (the other owner of the
        # volume: where an object that did not go in lies -- the live landing was 0.6 m behind the front, above it)
        views = [None] + ([p0, (p0[0], front["ymax"] + 0.30, p0[2])] if p0 else [])
        for v in views:
            try:
                if v is not None:
                    self._face(v)
                obs = self._look()
            except _Soft:
                continue
            for mid, base, dd in self._seen_of_type(obs, held):
                if any(math.dist(mid, q) < PUT_IN_SAME for q in pre):
                    continue
                if not any(math.dist(mid, f[0]) < PUT_IN_SAME for f in found):
                    found.append((mid, base, dd))
            if found:
                break
        if not found:
            self._ev(f"{held}: not seen after the put (nothing new in view) - unverified, success bool only")
            return
        nx, nz = front["n"]
        lo, hi, ymin, ymax = front_extent(front)
        mid_l, half = (lo + hi) / 2.0, (hi - lo) / PATCH_SPAN / 2.0
        ev_ = []
        inside = None
        low = None
        for mid, base, dd in found:
            tol = PUT_IN_SIGMAS * math.sqrt(2.0) * (DEPTH_ERR[0] + DEPTH_ERR[1] * dd)
            s_ = (mid[0] - front["c"][0]) * nx + (mid[2] - front["c"][2]) * nz
            lat = _lat(front, mid[0], mid[2])
            ok_ = (s_ <= tol and mid_l - half - tol <= lat <= mid_l + half + tol
                   and ymin - tol <= mid[1] <= ymax + tol)
            # final regression: the opening's height range is only what the moving door leaf showed (a fridge door covers
            # one compartment), so a landing BELOW it that is behind the front plane and laterally inside is not evidence
            # of a wrong put (the failure B3 targets lies ABOVE the opening, on the surface that shares its volume). It
            # is not proof of containment either (a counter below a wall cabinet): "low", unverified, no correction.
            low_ = (s_ <= tol and mid_l - half - tol <= lat <= mid_l + half + tol
                    and mid[1] < ymin - tol)
            ev_.append(f"({mid[0]:.2f},{mid[1]:.2f},{mid[2]:.2f}) {s_:+.2f} m from the front plane, lateral "
                       f"{lat - mid_l:+.2f} m, y {mid[1]:.2f} vs opening {ymin:.2f}..{ymax:.2f}, tol {tol:.2f}: "
                       + ("inside" if ok_ else "below the estimated opening, behind its front: containment unverified"
                          if low_ else "OUTSIDE"))
            if ok_ and inside is None:
                inside = mid
            elif low_:
                low = mid
        self._ev(f"{held} seen after the put, new: " + " | ".join(ev_))
        if inside is not None:
            self.verify.append("put-seen-in-opening")
            return
        if low is not None:
            return                          # unverified: the engine's ok stands as a bool, nothing is claimed
        mid = found[0][0]
        self.put_in_landed.append(tuple(mid))
        self._ev(f"{held} is outside the {rec} opening - it is not in {rec}; picking it back up "
                 "(hypothesis: the put pixel belonged to the surface that overlaps it)")
        if st.get("fixes", 0) >= PUT_FIX_MAX:
            raise _Fail("wrong_target", f"{held} is still outside {rec} after {PUT_FIX_MAX} corrections - "
                                        f"not reporting a put into {rec}")
        st["fixes"] = st.get("fixes", 0) + 1
        if p0:
            self.put_in_reject.append((p0[0], p0[1], p0[2], PUT_IN_REJECT_R))
            self.put_in_caps.append((front["c"][0], front["c"][2], p0[1]))
            self._ev(f"{held}: pixels of this {rec} opening at or above {p0[1]:.2f} m come last from now on "
                     "(the pixel at that height was owned by something else)")
        try:
            self._interact("pickup", held, belief={"status": "trusted", "xyz": mid, "type": held, "anchor": "put-check"})
        except _Fail as e:
            raise _Fail("wrong_target", f"{held} was put outside {rec} and could not be picked back up ({e})")
        geo = self._put_geo.get(tuple(pt))
        key = (round(front["c"][0], 1), round(front["c"][2], 1))
        n_here = st.setdefault("wrong_here", {}).get(key, 0) + 1
        st["wrong_here"][key] = n_here
        # a pixel BELOW the height cap that also landed outside (n_here >= 2) points at the instance, not the pixel
        deep = False
        if deep or n_here >= 2:
            # a pixel well inside the cavity landed outside (a surface >= PUT_DEEP behind the front plane cannot be the
            # counter above), or a second pixel did: the INSTANCE is the likely cause, not the pixel -- try another
            # one of the type (the same one again only if no other is known)
            self.put_bad_recs.append((rec, held, front["c"][0], front["c"][2]))
            raise _Soft("wrong_instance", f"{held} landed outside this {rec} "
                        + ("from a pixel deep inside it" if deep else "twice") + "; picked back up, another "
                        + rec + " next", free=True)
        raise _Soft("wrong_target", f"{held} landed outside the {rec} opening; picked back up", free=True)

    def _open_posed(self, rec):
        """An explicit ``open`` plan step on a container (pose-first mode): the same pose a put would use."""
        used = set()
        while True:
            try:
                self._cap()
                pose = self._pose_and_open(rec, "explicit", used=used)
                if pose is None:
                    return self._interact("open", rec)
                self.verify.append("state" if pose.get("entry") else "already")
                return {"verify": "opened from the put pose"}
            except _Soft as e:
                self.sub.append({"verb": "open", "object": rec, "ok": False, "cause": e.cause})
                self._spend(e)
                used.add(hm.current_cell(self.h, self.gm)[0])  # another pose next time
                self._stand()

    # -------------------------------------------------------------- put
    def _put(self, rec):
        self._in_put = True
        try:
            if self.pose_first and rec in OPENABLE:
                return self._put_openable(rec)
            return self._put_inner(rec)
        finally:
            self._in_put = False

    _in_put = False
    _put_plane = None
    _put_geo = {}
    _toggle_now = False

    def _ev_put_point(self, obs, det, pt, rec):
        """B2 diagnostics (own depth + detections only): where the put pixel is, how high that surface is, and which
        other boxes lie near it -- the evidence for 'it went onto the wrong receptacle'."""
        try:
            d = _depth(obs.depth, pt)
            xyz = objmem.backproject(pt[0], pt[1], d, obs.pose, _wh(obs.depth), self.om.fov) if d else None
            near = [f"{o.get('type')}{tuple(int(v) for v in o['bbox_xyxy'])}" for o in obs.dets
                    if o is not det and o.get("bbox_xyxy") and _inside(pt, o["bbox_xyxy"], 25)][:5]
            msg = (f"put point {tuple(pt)} on {rec} box {tuple(int(v) for v in det['bbox_xyxy'])}: "
                   + (f"surface y={xyz[1]:.2f} m" if xyz else "no depth"))
            self._ev(msg + (f"; other boxes near it: {', '.join(near)}" if near else ""))
        except Exception:
            pass

    def _verify_put(self, held, rec, st, plane):
        """B2: a put the engine called ok is not yet a put on ``rec``. Look at where the object now rests (own depth):
        its observed base lower than the receptacle's support plane (the plane fitted from this agent's own frame
        when it chose the pixel) means it sits in something lower -- a basin, a gap -- not on the surface. Then pick
        it back up, exclude that spot (map coordinates) and let the retry loop put it elsewhere. Unreadable (not
        seen, no plane, another instance) stays 'unverified' and keeps the success bool; only a contrary observation
        overrides the action's result, and nothing here turns a result into a claim."""
        pr = self.put_rec.get(held)
        if plane is None or pr is None:
            self._ev(f"put verified by success bool only ({'no support plane' if plane is None else 'no put record'})")
            return
        try:
            self._face(pr["p0"])
            obs = self._look()
            det = self._pick(obs, held, pr["p0"])
        except _Soft:
            det = None
        if det is None or not det.get("bbox_xyxy"):
            self._ev(f"{held}: not seen after the put - unverified (success bool only)")
            return
        wh = _wh(obs.depth)
        b = det["bbox_xyxy"]
        base_pt = ((b[0] + b[2]) // 2, max(b[1], b[3] - 3))
        d = _depth(obs.depth, base_pt)
        base = objmem.backproject(base_pt[0], base_pt[1], d, obs.pose, wh, self.om.fov) if d else None
        cen = det.get("point") or ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        dc = _depth(obs.depth, cen)
        mid = objmem.backproject(cen[0], cen[1], dc, obs.pose, wh, self.om.fov) if dc else None
        if base is None or mid is None:
            self._ev(f"{held}: seen after the put but no depth at its base - unverified")
            return
        away = math.hypot(mid[0] - pr["p0"][0], mid[2] - pr["p0"][2])
        if away > PUT_HALF_MAX:
            self._ev(f"{held}: the nearest one seen is {away:.2f} m from the put point (another instance?) - unverified")
            return
        tol = max(PUT_BASE_TOL, 2 * (DEPTH_ERR[0] + DEPTH_ERR[1] * d))
        if base[1] >= plane - tol:
            self._ev(f"{held}: base at y={base[1]:.2f} m, {rec} surface y={plane:.2f} m - rests on the surface (seen)")
            self.verify.append("put-seen-on-surface")
            return
        self._ev(f"{held}: base at y={base[1]:.2f} m is {plane - base[1]:.2f} m BELOW the {rec} surface "
                 f"(y={plane:.2f} m) - it is not on {rec}; picking it back up (hypothesis: it sits in a lower receptacle)")
        if st.get("fixes", 0) >= PUT_FIX_MAX:
            raise _Fail("wrong_target", f"{held} is still below the {rec} surface after {PUT_FIX_MAX} corrections - "
                                        f"not reporting a put on {rec}")
        self.put_reject.append((mid[0], mid[2], PUT_REJECT_R))
        self.put_reject.append((pr["p0"][0], pr["p0"][2], PUT_REJECT_R))
        st["fixes"] = st.get("fixes", 0) + 1
        pr["dead"] = True
        try:
            self._interact("pickup", held, belief={"status": "trusted", "xyz": mid, "type": held, "anchor": "put-check"})
        except _Fail as e:
            raise _Fail("wrong_target", f"{held} was put below the {rec} surface and could not be picked back up ({e})")
        raise _Soft("wrong_target", f"{held} landed below the {rec} surface; picked back up")

    def _put_stale_destination(self, rec, st, e) -> bool:
        """L3 audit (stage H-B9, general executor defect): a PUT whose destination is not re-seen where it was believed to be had no
        way to let go of that belief -- three recovery moves, then retry_exhausted -- while a PICKUP (`_interact`) treats the same
        situation as a stale sighting: free local looks, then the belief is SET ASIDE and the search runs again. Live (dev, non-test):
        the search stopped on a single low-confidence "Plate" on a counter, the real plate lay in a closed cabinet, and every
        `pick_and_place X -> Plate` ended in not_detected. Same policy, same caps (LOCAL_LOOKS, MAX_RELOCATE), exhaustive policy only,
        sightings only (never the agent's own put record, never a container it opened itself). True = handled, loop again."""
        b = st.get("b")
        if (not self.exhaustive or e.cause != "not_detected" or b is None or b.get("status") == "placed" or b.get("anchor")
                or rec in self.opened_at or st.get("relocs", 0) >= MAX_RELOCATE):
            return False
        self._cap()
        if st.get("local", 0) < LOCAL_LOOKS:                 # a free local look first (searching, not retrying)
            st["local"] = st.get("local", 0) + 1
            self._local_look(st["local"], b, st)
            return True
        self._set_aside(b, f"{rec} (put destination) not re-seen after {LOCAL_LOOKS} local looks")
        st.update(approached=False, local=0, alt=0, dpitch=0.0, b=None)
        st["relocs"] = st.get("relocs", 0) + 1
        return True

    def _put_inner(self, rec):
        if not self.h.holding:
            raise _Fail("action_refused", "hand is empty")
        held = self.h.holding
        st = {"alt": 0, "dpitch": 0.0, "approached": False}
        k = 0
        while True:
            try:
                self._cap()
                if st["approached"]:
                    b = st["b"]
                elif rec in self.opened_at:               # I opened it: aim at its body, not at its swung door
                    b = {"status": "trusted", "xyz": self.opened_at[rec], "anchor": "opened-at"}
                else:
                    b = self._locate(rec)
                st["b"], st["xyz"] = b, b["xyz"]
                if not st["approached"]:
                    self._goto(b["xyz"])
                    self._face(b["xyz"], st["dpitch"])
                    st["approached"] = True
                obs = self._look()
                det = self._pick(obs, rec, b["xyz"], None)
                if det is None:
                    raise _Soft("not_detected", f"no {rec} in view")
                if rec in OPENABLE:
                    v = state_of(det, "open")
                    if v is False:
                        self._interact("open", rec, belief={"status": "trusted", "xyz": self._xyz_of(det, b["xyz"])})
                        st["b"] = b = {**b, "xyz": self.opened_at.get(rec, b["xyz"])}
                        st["xyz"] = b["xyz"]
                        self._face(b["xyz"], st["dpitch"])
                        obs = self._look()
                        det = self._pick(obs, rec, b["xyz"], None) or det
                    elif v is None and k == 0:
                        self._open_container(rec, b["xyz"], why="put")
                        obs = self._look()
                        det = self._pick(obs, rec, b["xyz"], None) or det
                pt = self._point(obs, det, st["alt"], receptacle=True)
                if pt is None:
                    raise _Soft("wrong_target", f"no put point on {rec} clear of the other receptacles in view")
                self._ev_put_point(obs, det, pt, rec)
                r = self._act_at("PutObject", obs, [pt], rec)
                if not r["sent"]:
                    nc = int(st.get("closer") or 0)
                    if "out of reach" in r["why"] and (nc == 0 or (self.exhaustive and nc <= k)):
                        # walk nearer once, then look again from there (console: once more after each recovery
                        # move, which can step it back out of reach -- 2026-10-08, the low garbage can)
                        st["closer"] = nc + 1
                        if self._closer(b["xyz"]):
                            continue
                    raise _Soft("unreachable", f"not sent: {r['why']}")
                self._ev(f"PutObject {held} -> {rec} at {tuple(pt)}: {'ok' if r['ok'] else r['err']}")
                if not r["ok"]:
                    raise _Soft("action_refused", r["err"])
                self.om.on_put(True, rec, self._xyz_of(det, b["xyz"]), otype=held, step=self.h.step)
                self._note_put(held, rec, obs, det, r["pt"] if r.get("pt") else pt)
                if self.exhaustive and PUT_SURFACE_ON and rec not in OPENABLE and rec not in CONTAINER_LIKE:
                    self._verify_put(held, rec, st, self._put_plane)       # may pick it back up and raise _Soft
                self._world_ver += 1
                self.sub.append({"verb": "put_in", "object": rec, "ok": True, "verify": "success-bool"})
                self.verify.append("success-bool")
                return {"verify": "success-bool"}
            except _Soft as e:
                self.sub.append({"verb": "put_in", "object": rec, "ok": False, "cause": e.cause})
                if self._put_stale_destination(rec, st, e):
                    continue
                self._spend(e)
                k += 1
                if rec in OPENABLE and k == 1:
                    self._open_container(rec, st.get("xyz") or (0, 0, 0), why="put")
                self._recover(k, st.get("xyz"), st, yaw=True)

    # -------------------------------------------------------------- search (decision 4)
    def viewpoints(self) -> list:
        if self._vps is None:
            cur, _ = hm.current_cell(self.h, self.gm)
            self._vps = farthest_points(self.gm.free, N_VIEWPOINTS, cur)
        return list(self._vps)

    def _search_spent(self) -> bool:
        """Leave room under the step's primitive cap for acting on what the search finds (and cleanup)."""
        return self.h.step - self._step0 > self._limit - SEARCH_RESERVE

    def _search(self, otype):
        if self.on_event is None:
            return self._search_exhaustive(otype) if self.exhaustive else self._search_bounded(otype)
        self._emit("search_start", otype=otype, prims=self.h.step)
        self._srch = {"otype": otype, "first": None}
        b = None
        try:
            b = self._search_exhaustive(otype) if self.exhaustive else self._search_bounded(otype)
            return b
        finally:
            info = self.search_info or {}
            first, self._srch = (self._srch or {}).get("first"), None
            self._emit("search_end", otype=otype, found=bool(b), prims=self.h.step, first_seen=first,
                       cap=self._step0 + self._limit - SEARCH_RESERVE,
                       visited=[str(t) for t in info.get("visited") or []], opened=[str(t) for t in info.get("opened") or []],
                       unreachable=sorted({str(t) for t in info.get("unreachable") or []}),
                       viewpoints=int(info.get("viewpoints") or 0), budget=bool(info.get("budget")),
                       hinted=bool(info.get("hinted")))

    def _briefed(self, otype, ranked, info) -> list:
        """B: put the container types a briefing names for ``otype`` FIRST in the search order (a +package arm only; an
        executor built without ``allow_briefing_hints`` has no hook and returns ``ranked`` untouched). Only an ORDER: every
        other place is still searched, so a wrong hint costs a visit, never the object. Recorded in the search info."""
        if self.briefing_hints is None:
            return ranked
        try:
            raw = list(self.briefing_hints(otype) or [])
        except Exception as e:                         # noqa: BLE001
            self.event_errors.append({"kind": "briefing_hints", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            return ranked
        hint = []
        for h_ in raw:
            t = h_.get("rtype") if isinstance(h_, dict) else h_
            if isinstance(t, str) and t in self.type_list and t != otype and t not in hint:
                hint.append(t)
        self.briefing_log.append({"otype": otype, "hint": list(hint), "asked": len(raw)})
        if not hint:
            return ranked
        info["briefing"] = list(hint)
        self._ev(f"search {otype}: the briefing names {hint} - looking there first (every other place is still searched)")
        return hint + [p for p in ranked if p not in hint]

    def _search_bounded(self, otype):
        info = {"target": otype, "chooser": getattr(self.chooser, "label", getattr(self.chooser, "__name__", "?")),
                "ranked": [], "visited": [], "opened": [], "viewpoints": 0, "found": False}
        self.search_info = info
        self._ev(f"search {otype}: sweep here first")
        b = self._sweep(otype)
        if b:
            info["found"] = "here"
            return b
        try:
            ranked = list(self.chooser(otype, self.type_list, self.om.summary()) or [])
        except Exception as e:
            self._ev(f"chooser failed ({e}); viewpoints only")
            ranked = []
        ranked = [p for p in ranked if p in self.type_list and p != otype]
        ranked = self._briefed(otype, ranked, info)
        info["ranked"] = ranked
        if self.place_hints is not None:
            info["hinted"] = [{"rtype": h.get("rtype"), "xyz": [round(float(x), 2) for x in h["xyz"]]}
                              for h in self._hint_list(otype) if h.get("xyz")]
        self._ev(f"search {otype}: places ranked {ranked[:6]}")
        visited, seen_xyz = set(), []

        def visit_places():
            for place in ranked:
                n_type = 0
                for pb in self._hint_sort(otype, place, self._rank(self.om.lookup(place))) \
                        if self.place_hints is not None else self._rank(self.om.lookup(place)):
                    if len(visited) >= MAX_PLACES or self._search_spent():
                        return None
                    if n_type >= PER_TYPE:
                        break
                    if pb["track"] in visited or any(math.dist(pb["xyz"], v) < SAME_PLACE for v in seen_xyz):
                        continue
                    visited.add(pb["track"])
                    seen_xyz.append(tuple(pb["xyz"]))
                    n_type += 1
                    info["visited"].append(place)
                    try:
                        self._goto(pb["xyz"])
                    except _Fail:
                        continue
                    self._face(pb["xyz"])
                    obs = self._look()
                    got = self._seen(obs, otype)
                    if got:
                        return got
                    if place in OPENABLE and self._open_container(place, pb["xyz"]):
                        info["opened"].append(place)
                        self._face(pb["xyz"], PITCH_ALT)
                        got = self._seen(self._look(), otype)
                        if got:
                            return got
                        # not inside: close it again -- its swung door otherwise blocks the next walks (s5 live)
                        self._face(pb["xyz"])
                        if self._close_container(place, pb["xyz"]):
                            info.setdefault("closed", []).append(place)
                    self._cap()
            return None

        b = visit_places()
        if not b:
            for vp in self.viewpoints():
                if (len(visited) >= MAX_PLACES and info["viewpoints"] >= N_VIEWPOINTS) or self._search_spent():
                    break
                if not self._walk_cell(vp):
                    continue
                info["viewpoints"] += 1
                b = self._sweep(otype)
                if b:
                    break
                if len(visited) < MAX_PLACES:
                    b = visit_places()
                    if b:
                        break
        if b:
            info["found"] = True
            self._ev(f"search {otype}: found ({b['status']}) after {len(info['visited'])} place(s), "
                     f"{info['viewpoints']} viewpoint(s)")
        else:
            self._ev(f"search {otype}: not found within budget")
        return b or None

    def _search_exhaustive(self, otype):
        """Search everything reachable (see the module docstring) along an explicit, numbered route (2026-10-08,
        "fix: systematic search route"): look around once (not again where it already did), plan stops from its own
        front registry + remembered places + the chooser's ranking, visit them in order inspecting every container a
        stop serves, re-plan only on new information, then coverage viewpoints that add unseen area. Returns the
        belief seen now, or None."""
        info = {"target": otype, "mode": "exhaustive",
                "chooser": getattr(self.chooser, "label", getattr(self.chooser, "__name__", "?")),
                "ranked": [], "visited": [], "opened": [], "closed": [], "close_failed": [], "unreachable": [],
                "viewpoints": 0, "viewpoints_planned": 0, "found": False, "complete": False, "budget": False,
                "plan": [], "plans": 0}
        self.search_info = info
        la = self._swept_here()
        if la is not None:
            self._ev(f"search {otype} (exhaustive): already looked around from this spot (step {la['step']}) - "
                     "not again; planning from what it has seen")
            b = None
        else:
            self._ev(f"search {otype} (exhaustive): look around here first")
            b = self._sweep(otype)
        if b:
            info["found"] = "here"
            return b
        try:
            ranked = list(self.chooser(otype, self.type_list, self.om.summary()) or [])
        except Exception as e:
            self._ev(f"chooser failed ({e}); every place type, viewpoints")
            ranked = []
        ranked = [p for p in ranked if p in self.type_list and p != otype]
        ranked = self._briefed(otype, ranked, info)
        rest = sorted(t for t in self.type_list if t != otype and t not in ranked
                      and (t in OPENABLE or t in objmem.RECEPTACLE_TYPES))
        order = ranked + rest
        info["ranked"] = ranked
        if self.place_hints is not None:
            info["hinted"] = [{"rtype": h.get("rtype"), "xyz": [round(float(x), 2) for x in h["xyz"]]}
                              for h in self._hint_list(otype) if h.get("xyz")]
        info["order"] = order
        self._ev(f"search {otype}: places ranked {ranked[:6]}, then every other place type ({len(rest)})")
        rank_types = [t for t in order if t in OPENABLE]

        b = None
        for r in [r for r in self.inspected if (r.get("target") == otype and r["verdict"] == "found")
                  or (otype in r.get("seen_types", ()) and r["revisits"] < 2 and r["verdict"] in ("found", "visible", "partly", "unsure"))]:
            # where THIS agent saw the target inside a container in an earlier step (or seen it there while looking for
            # something else, E3): look there again first
            if b or self._search_spent():
                break
            vc, vpts = max(r["views"], key=lambda v: len(v[1]))
            self._ev(f"{r['type']} {r['n']}: the {otype} was seen inside it before - looking there again")
            b = self._inspect(r["type"], front_patch_like(r, vc, vpts), otype, info, rec=r)
        if not b:
            items = self._plan_items(otype, ranked, order)
            wbest = max((it["w"] for it in items), default=0.0)
            info["wmin"] = TIER_FRAC * wbest
            likely = sorted({it["type"] for it in items if it["w"] >= info["wmin"]})
            self._ev(f"search {otype}: pass 1 - the likely places first ({', '.join(likely) or '-'}; weight >= "
                     f"{info['wmin']:.2f}, chooser rank x declared height prior)")
            b = self._run_plan(otype, info, ranked, order, rank_types, "memory + the chooser's ranking")
            if not b and not info["budget"] and info["wmin"] > 0:
                info["wmin"] = 0.0
                self._ev(f"search {otype}: pass 2 - every other container and place, one more route")
                b = self._run_plan(otype, info, ranked, order, rank_types, "pass 2: everything else")
        if not b and not info["budget"]:
            # containers whose inside was not visible: one revisit each from another pose, nearest first
            todo = [r for r in self.inspected if r["verdict"] == "unsure" and r["revisits"] == 0
                    and r["reason"] not in ("unreachable", "wrong thing opened")
                    and not str(r["reason"]).startswith("not re-identified")]
            while todo and not b:
                if self._search_spent() or len(self.inspected) > MAX_INSPECT:
                    break
                x, z = self._pose_xz()
                r = min(todo, key=lambda r: math.hypot(r["c"][0] - x, r["c"][2] - z))
                todo.remove(r)
                vc, vpts = r["views"][-1]
                pa = {**front_patch_like(r, vc, vpts)}
                if not self._insp_cells(pa, r["type"], exclude=set(r["cells"])):
                    self._ev(f"{r['type']} {r['n']}: no other view to try; stays unsure")
                    continue
                self._ev(f"{r['type']} {r['n']}: revisit from another pose (inside was not visible)")
                b = self._inspect(r["type"], pa, otype, info, rec=r)
        if not b and not info["budget"]:
            b = self._coverage(otype, info, ranked, order, rank_types)
            if not b and not info["budget"]:
                info["complete"] = not any(not f["done"] for f in self.fronts if f["type"] in order)
        if not b and self._la is not None and self._la["otype"] == otype and self._la.get("pending"):
            b = self._la_revisit(otype, info)            # E3: ambiguity points seen out of reach -> go there (RevaluateObject l.2)
        info["inspections"] = [i for i in info.get("inspections") or [] if i]
        info.pop("order", None)
        for r in self.inspected:                       # slots were positions in the uncompacted list
            r.pop("slot", None)
        if b:
            info["found"] = True
            self._ev(f"search {otype}: found ({b['status']}) after {len(info['visited'])} place(s), "
                     f"{len(info['opened'])} container(s) opened, {info['viewpoints']} viewpoint(s)")
        else:
            info["report"] = (f"not seen after checking {len(info['visited'])} place(s), opening "
                              f"{len(info['opened'])} container(s), {info['viewpoints']} coverage viewpoint(s)"
                              + (" - search budget reached" if info["budget"] else
                                 f" - searched the reachable known map: every planned stop visited, no viewpoint "
                                 f"adds >= {VIEW_GAIN} unseen cells ({info.get('unseen_left', '?')} cell(s) of "
                                 "interest never seen)")
                              + (f"; unreachable: {sorted(set(info['unreachable']))}" if info["unreachable"] else ""))
            self._ev(f"search {otype}: {info['report']}")
        return b or None

    # -------------------------------------------------------------- systematic search route (2026-10-08)
    def _route_reset(self):
        """Episode-scoped own history for the search route (geometry from own depth / pose only)."""
        self.fronts = []               # own front registry: [{id, type, patch, d, step, seen, done}]
        self.looked_around = []        # full look-arounds: [{cell, step, ver, crouched}]
        self.seen = set()              # map cells its own depth rays ended in (coverage phase)
        self.view_poses = []           # [(pose, wh)] it looked from (a place "looked at" check)
        self.places_done = []          # place xyz it walked to and looked at in a search
        self._world_ver = 0            # own successful pickups / puts / drops (a look-around before one is stale)
        self._stop = None
        self._new_fronts = 0
        self._front_n = 0
        #: B (2026-10-09): episode-local foreground hypotheses, track -> {xyz, step, px, depth, median}; persistent
        #: memory is unchanged (its xyz is the box's grounded point); used only while that xyz is not actable
        self.fg_hyp = {}
        self._act_cache = {}

    def _note_view(self, obs):
        """Every look (exhaustive mode): seen cells, the pose, and the openable fronts in view -> own registry."""
        try:
            self.seen |= seen_cells(obs.depth, obs.pose, self.om.fov, self.gm)
        except Exception:                              # no / odd depth: nothing marked seen (conservative)
            pass
        if obs.depth is not None:
            self.view_poses.append((dict(obs.pose), _wh(obs.depth)))
            if len(self.view_poses) > 800:
                del self.view_poses[:200]
        self._register_fronts(obs)

    def _register_fronts(self, obs):
        if obs.depth is None:
            return
        eye = objmem._basis(obs.pose)[0]
        for d in obs.dets:
            t = d.get("type")
            if t not in OPENABLE or not d.get("bbox_xyxy") or state_of(d, "open") is True:
                continue
            if t == "Drawer" and any(o is not d and o.get("type") == "Drawer" and o.get("bbox_xyxy")
                                     and _area(d["bbox_xyxy"]) > 1.6 * _area(o["bbox_xyxy"])
                                     and _inside(((o["bbox_xyxy"][0] + o["bbox_xyxy"][2]) / 2,
                                                  (o["bbox_xyxy"][1] + o["bbox_xyxy"][3]) / 2), d["bbox_xyxy"])
                                     for o in obs.dets):
                continue                               # a box spanning a stacked drawer's own box: the union
            pa = front_patch(obs.depth, d["bbox_xyxy"], obs.pose, self.om.fov)
            if pa is None:
                continue
            dist = math.dist(eye, pa["c"])
            if dist > REG_RANGE:
                continue
            dy_tol = DRAWER_SAME_DY + 0.01 if t == "Drawer" else ID_SAME_DY
            hit = None
            for f in self.fronts:
                fc = f["patch"]["c"]
                if f["type"] != t or abs(fc[1] - pa["c"][1]) > dy_tol:
                    continue
                # same front: near centroids or shared surface points -- tight on purpose: the two doors of a double
                # wall cabinet are ~0.3-0.5 m apart and must stay two (after_v1: merged -> one door never opened);
                # one big front seen as several entries only costs a look (a stop enumerates its CURRENT frame)
                tol = 0.20 if t != "Fridge" else max(0.20, 0.5 * front_width(f["patch"]))
                if (math.hypot(fc[0] - pa["c"][0], fc[2] - pa["c"][2]) <= tol
                        or coincide(pa["pts"], f["patch"]["pts"]) >= 0.5):
                    hit = f
                    break
            if hit is None:
                self._front_n += 1
                self.fronts.append({"id": self._front_n, "type": t, "patch": pa, "d": dist, "step": obs.step,
                                    "seen": 1, "done": None, "misses": 0})
                self._new_fronts += 1
            else:
                hit["seen"] += 1
                if dist < hit["d"] - 0.10 or (pa["planar"] and not hit["patch"]["planar"] and dist <= hit["d"] + 0.3):
                    hit["patch"], hit["d"], hit["step"] = pa, dist, obs.step   # a nearer / planar view is better

    def _covered(self, f) -> bool:
        """For PLANNING only: a registry front this agent already inspected (``_match`` "same": near fronts AND
        shared surface / image evidence) gets no stop. Merely NEAR an inspected one is not enough -- the other door
        of a double cabinet is 0.3-0.5 m away (after_v1); at a stop the current frame decides."""
        m, _ = self._match(f["type"], f["patch"])
        return m == "same"

    def _swept_here(self):
        """Own history: a full look-around from THIS cell, standing, with no own pickup / put / drop since (Astra:
        proximity alone is not the same view)."""
        cur, _ = hm.current_cell(self.h, self.gm)
        return next((la for la in self.looked_around if tuple(la["cell"]) == tuple(cur)
                     and la["ver"] == self._world_ver and not la["crouched"]), None)

    def _looked_at(self, xyz) -> bool:
        """A place counts as looked at when it was in frame from <= CLOSE_LOOK m in one of its own looks."""
        for pose, wh in self.view_poses:
            eye = objmem._basis(pose)[0]
            if math.hypot(eye[0] - xyz[0], eye[2] - xyz[2]) > CLOSE_LOOK:
                continue
            pr = objmem.project(xyz, pose, wh, self.om.fov)
            if pr is not None and 0.05 * wh[0] <= pr[0] <= 0.95 * wh[0] and 0.05 * wh[1] <= pr[1] <= 0.95 * wh[1]:
                return True
        return any(math.dist(xyz, p) < SAME_PLACE for p in self.places_done)

    @staticmethod
    def _band(ctype, y):
        if ctype == "Fridge":
            return "fridge"
        if ctype == "Drawer":
            return "drawer"
        return "wall" if y > HIGH_Y else "base" if y < CROUCH_Y else "counter"

    def _weight(self, otype, ctype, y, ranked) -> float:
        return rank_weight(ctype, ranked) * (height_prior(otype, ctype, y) if ctype in OPENABLE else 1.0)

    def _plan_items(self, otype, ranked, order, wmin: float = 0.0):
        """Plan input: uninspected fronts of the own registry + remembered places not looked at closely, each with
        a preference weight and the cells that serve it (hard checks first: front side, reach, door sweep)."""
        cam = self._cam_y()
        items = []
        for f in self.fronts:
            if f["done"] or f["type"] not in order:
                continue
            if f["d"] > PLAN_RANGE and f["seen"] < 2:
                continue                               # one far glimpse: no stop of its own (registered again near)
            if self._covered(f):
                f["done"] = "inspected"
                continue
            pa = f["patch"]
            cells = inspection_cells(self.gm, pa, cam, f["type"],
                                     model=self._door_for(pa, f["type"]) if self.pose_first else None)
            inside = False
            if not cells:
                cells, inside = inspection_cells(self.gm, pa, cam, f["type"]), True
            if not cells:
                continue
            best = cells[0][0]
            w = self._weight(otype, f["type"], pa["c"][1], ranked)
            if self.place_hints is not None:
                w *= self._hint_boost(otype, f["type"], pa["c"])
            if w < wmin:
                continue
            items.append({"kind": "front", "front": f, "type": f["type"], "c": pa["c"], "w": w, "cost": T_INSPECT,
                          "serve": serve_cells(self.gm, pa, cells), "in_sweep": inside,
                          "band": self._band(f["type"], pa["c"][1])})
        seen_xyz = []
        for ptype in order:
            if ptype in OPENABLE or rank_weight(ptype, ranked) < wmin:
                continue
            for b in self._rank(self.om.lookup(ptype)):
                xyz = tuple(b["xyz"])
                if any(math.dist(xyz, v) < SAME_PLACE for v in seen_xyz) or self._looked_at(xyz):
                    continue
                seen_xyz.append(xyz)
                serve = {}
                for c in self.gm.free:
                    wx, wz = self.gm.world(c)
                    r = math.hypot(wx - xyz[0], wz - xyz[2])
                    if PLACE_SERVE[0] <= r <= PLACE_SERVE[1] and not self.gm.is_temp_blocked(c):
                        serve[c] = abs(r - 0.9)
                if serve and len(seen_xyz) <= MAX_EXH_PLACES:
                    wp = rank_weight(ptype, ranked)
                    if self.place_hints is not None:
                        wp *= self._hint_boost(otype, ptype, xyz)
                    items.append({"kind": "place", "type": ptype, "c": xyz, "w": wp,
                                  "cost": T_LOOK, "serve": serve, "band": "place"})
        if self.place_hints is not None and self.hint_goals:
            for h in self._hint_list(otype):
                if not h.get("xyz"):
                    continue
                hxyz, rad = tuple(float(v) for v in h["xyz"]), float(h.get("radius") or HINT_RADIUS)
                if any(math.dist(tuple(it["c"]), hxyz) <= rad for it in items) or self._looked_at(hxyz):
                    continue                                   # a known container / place already covers it (boosted), or it was looked at
                serve = {}
                for c in self.gm.free:
                    wx, wz = self.gm.world(c)
                    r = math.hypot(wx - hxyz[0], wz - hxyz[2])
                    if PLACE_SERVE[0] <= r <= PLACE_SERVE[1] and not self.gm.is_temp_blocked(c):
                        serve[c] = abs(r - 0.9)
                if serve:
                    items.append({"kind": "place", "type": h.get("rtype") or "hint", "c": hxyz,
                                  "w": rank_weight(h.get("rtype") or "", ranked) * (HINT_BOOST if self.hint_boost is None else self.hint_boost),
                                  "cost": T_LOOK, "serve": serve, "band": "place", "hint": True})
        tot = sum(it["w"] for it in items) or 1.0
        for it in items:
            it["q"] = it["w"] / tot
        return items

    def _dist_fn(self):
        dmaps = {}

        def dist(a, b):
            a, b = tuple(a), tuple(b)
            if a not in dmaps:
                dmaps[a] = grid_dist(self.gm, a)
            return dmaps[a].get(b, 99.0)
        return dist

    def _label(self, s):
        cnt = {}
        for it in s["items"]:
            k = (it["type"], it["band"])
            cnt[k] = cnt.get(k, 0) + 1
        return ", ".join(f"{t} x{n}" + ("" if b in ("place", "fridge") else f" ({b})") for (t, b), n in cnt.items())

    def _publish(self, stops, info):
        """Number the pending stops in visiting order after the ones already visited (events / ticket / map)."""
        info["plan"] = [p for p in info.get("plan") or [] if p["status"] != "planned"]
        n0 = len(info["plan"])
        for i, s in enumerate(stops):
            wx, wz = self.gm.world(s["cell"])
            s["summary"] = {"n": n0 + i + 1, "cell": list(s["cell"]), "x": round(wx, 2), "z": round(wz, 2),
                            "label": self._label(s), "items": len(s["items"]), "status": "planned",
                            "weight": round(sum(it["q"] for it in s["items"]), 3)}
            info["plan"].append(s["summary"])

    def _drop_far(self, stops, cur, dist, info):
        far = [s for s in stops if dist(cur, s["cell"]) >= 99.0]
        for s in far:
            for it in s["items"]:
                if it["kind"] == "front":
                    it["front"]["done"] = "unreachable"
            info["unreachable"].extend(it["type"] for it in s["items"])
        return [s for s in stops if s not in far]

    def _make_plan(self, otype, ranked, order, info, why):
        """Build the explicit search plan from where the agent stands: stops = set cover of the items' serve cells,
        ordered by the weighted search-cost surrogate. Shown as numbered stops (events, ticket, map)."""
        items = self._plan_items(otype, ranked, order, info.get("wmin", 0.0))
        self._new_fronts = 0
        info.setdefault("plan", [])
        if not items:
            self._publish([], info)
            return []
        cur, _ = hm.current_cell(self.h, self.gm)
        dist = self._dist_fn()
        stops = order_stops(self._drop_far(cover_stops(items), cur, dist, info), cur, dist)
        stops = refine_cells(stops, cur, dist)
        self._publish(stops, info)
        info["plans"] = info.get("plans", 0) + 1
        route = sum(dist(a, b) for a, b in zip([cur] + [s["cell"] for s in stops], [s["cell"] for s in stops]))
        self._ev(f"search plan {info['plans']} ({why}): {len(stops)} stop(s), {sum(len(s['items']) for s in stops)} "
                 f"place(s)/container(s), route {route:.1f} m - " + " | ".join(
                     f"{s['summary']['n']} ({s['summary']['x']:.2f},{s['summary']['z']:.2f}): {s['summary']['label']}"
                     for s in stops))
        return stops

    def _merge_new(self, stops, otype, ranked, order, info):
        """New fronts / places seen (Astra F: re-plan only on new information, batched after a stop): an item a
        remaining stop already serves joins that stop; items no remaining stop serves get new stops INSERTED where
        they cost least (cheapest insertion) -- the rest of the route keeps its order, so it does not swing back and
        forth as things come into view."""
        planned = {id(it["front"]) for s in stops for it in s["items"] if it["kind"] == "front"}
        planned_pl = [it["c"] for s in stops for it in s["items"] if it["kind"] == "place"]
        new = [it for it in self._plan_items(otype, ranked, order, info.get("wmin", 0.0))
               if (it["kind"] == "front" and id(it["front"]) not in planned)
               or (it["kind"] == "place" and not any(math.dist(it["c"], c) < SAME_PLACE for c in planned_pl))]
        self._new_fronts = 0
        if not new:
            return stops
        loose, joined = [], 0

        def near_score(s, it):
            # the stop's cell or one cell beside it (the stop's inspections may shift one cell) serves the item
            cs = [tuple(s["cell"])] + list(self.gm.neighbours(tuple(s["cell"])))
            sc = [it["serve"][c] + (0.0 if i == 0 else 0.5) for i, c in enumerate(cs) if c in it["serve"]]
            return min(sc) if sc else None
        for it in new:
            s = min((s for s in stops if near_score(s, it) is not None),
                    key=lambda s: near_score(s, it), default=None)
            if s is None:
                loose.append(it)
            else:
                s["items"].append(it)
                joined += 1
        cur, _ = hm.current_cell(self.h, self.gm)
        dist = self._dist_fn()
        added = self._drop_far(cover_stops(loose), cur, dist, info) if loose else []
        for ns in added:
            best = min(range(len(stops) + 1),
                       key=lambda i: (route_cost(stops[:i] + [ns] + stops[i:], cur, dist), i))
            stops.insert(best, ns)
        if added:
            stops = refine_cells(stops, cur, dist)
        if joined or added:
            self._publish(stops, info)
            info["updates"] = info.get("updates", 0) + 1
            self._ev(f"search plan updated ({len(new)} new place(s)/container(s) seen): {joined} joined a planned stop"
                     + (f", {len(added)} new stop(s) inserted where they cost least" if added else "")
                     + " - " + " | ".join(f"{s['summary']['n']} ({s['summary']['x']:.2f},{s['summary']['z']:.2f}): "
                                          f"{s['summary']['label']}" for s in stops))
        return stops

    def _stop_order(self, items):
        """Order a stop's items as one sweep of the head: columns by bearing (one direction, starting at the end
        nearer the current heading), within a column top to bottom; identity-ambiguous fronts last."""
        x, z = self._pose_xz()
        yaw = float(self.h.pose()["rotation"]["y"])

        def rel(it):
            b = math.degrees(math.atan2(it["c"][0] - x, it["c"][2] - z))
            return (b - yaw + 180.0) % 360.0 - 180.0
        rs = sorted(items, key=rel)
        cols = []
        for it in rs:
            if cols and abs(rel(it) - rel(cols[-1][-1])) <= 12.0:
                cols[-1].append(it)
            else:
                cols.append([it])
        cols = [sorted(c, key=lambda it: -it["c"][1]) for c in cols]
        if cols and abs(rel(cols[-1][0])) < abs(rel(cols[0][0])):
            cols = cols[::-1]
        return [it for c in cols for it in c]

    def _serves_here(self, patch, ctype) -> bool:
        """This stop (its cell, the cell the agent stands on, or one cell beside it) inspects this front within
        SERVE_TOL of its best inspection cell (hard checks are in the cell list)."""
        cells = self._insp_cells_quiet(patch, ctype)
        if not cells:
            return False
        serve = set(serve_cells(self.gm, patch, cells))
        cur, _ = hm.current_cell(self.h, self.gm)
        near = {tuple(cur)} | set(self.gm.neighbours(cur))
        if self._stop is not None:
            near.add(tuple(self._stop["cell"]))
        return bool(serve & near)

    def _insp_cells_quiet(self, patch, ctype):
        cam = self._cam_y()
        cells = inspection_cells(self.gm, patch, cam, ctype,
                                 model=self._door_for(patch, ctype) if self.pose_first else None)
        return cells or inspection_cells(self.gm, patch, cam, ctype)

    def _in_frame(self, xyz, obs, margin: float = 0.1) -> bool:
        pr = objmem.project(xyz, obs.pose, _wh(obs.depth), self.om.fov)
        w, h = _wh(obs.depth)
        return pr is not None and margin * w <= pr[0] <= (1 - margin) * w and margin * h <= pr[1] <= (1 - margin) * h

    def _at_stop(self, stop, otype, info, rank_types):
        """At one stop: look toward its planned items (one sweep of the head, no left/right oscillation), take the
        closed containers of the CURRENT frames that this stop serves squarely, and open / inspect / close them one
        after another from here (pitch / crouch / small yaw; one cell beside it at most, else a split stop). Planned
        items in view from here with no uninspected front left there are done (the frame decides, not the plan)."""
        x0, z0 = self._pose_xz()
        yaw0 = float(self.h.pose()["rotation"]["y"])

        def rel(c):                                    # bearing from the stop, against the arrival heading (fixed)
            b = math.degrees(math.atan2(c[0] - x0, c[2] - z0))
            return (b - yaw0 + 180.0) % 360.0 - 180.0

        targets = self._stop_order(stop["items"])
        sweep = 1 if (len(targets) < 2 or rel(targets[-1]["c"]) >= rel(targets[0]["c"])) else -1
        cursor = [rel(targets[0]["c"]) if targets else 0.0]   # where the head is in this stop's one sweep
        queue = []

        wmin, ranked = info.get("wmin", 0.0), info.get("ranked") or []
        hint_first = self.place_hints is not None and self.hint_goals     # B2: a container at a remembered place is inspected first

        def enqueue(obs):
            for c in self._candidates(obs, rank_types):
                if any(q["type"] == c["type"] and math.dist(q["patch"]["c"], c["patch"]["c"]) <= 0.15 for q in queue):
                    continue
                if self._weight(otype, c["type"], c["patch"]["c"][1], ranked)                         * (self._hint_boost(otype, c["type"], c["patch"]["c"]) if (self.place_hints is not None and self.hint_goals) else 1.0) < wmin:
                    continue                           # a less likely container: the second pass
                if self._serves_here(c["patch"], c["type"]):
                    queue.append(c)

        def next_q():
            nonlocal sweep
            ahead = [q for q in queue if (rel(q["patch"]["c"]) - cursor[0]) * sweep >= -12.0]
            if not ahead:                              # only behind the head: turn the sweep round once
                sweep = -sweep
                ahead = list(queue)
            q = min(ahead, key=lambda q: (0 if (hint_first and self._hint_boost(otype, q["type"], q["patch"]["c"]) > 1.0) else 1,
                                          q.get("late", False),
                                          round((rel(q["patch"]["c"]) - cursor[0]) * sweep / 12.0),
                                          -q["patch"]["c"][1]))
            queue.remove(q)
            cursor[0] = rel(q["patch"]["c"])
            return q

        last = None
        for it in targets:
            if self._search_spent():
                info["budget"] = True
                return None
            if it["kind"] == "front" and it["front"]["done"]:
                continue
            if it["kind"] == "place" and (any(math.dist(it["c"], p) < SAME_PLACE for p in self.places_done)
                                          or self._looked_at(it["c"])):
                continue                               # already in one of its own close looks: no turn for it
            if last is None or not self._in_frame(it["c"], last):
                self._stand()
                self._face(it["c"])
                last = self._look()
                got = self._seen(last, otype)
                if got:
                    return got
            if it["kind"] == "place":
                self.places_done.append(tuple(it["c"]))
                info["visited"].append(it["type"])
            enqueue(last)
            while queue:
                if self._search_spent():
                    info["budget"] = True
                    return None
                if len(self.inspected) >= MAX_INSPECT:
                    queue.clear()
                    break
                q = next_q()
                m, r0 = self._match(q["type"], q["patch"])
                if m == "same":
                    continue
                if m == "ambiguous" and not q.get("late"):
                    queue.append({**q, "late": True})
                    if all(x.get("late") for x in queue):
                        q = queue.pop()
                    else:
                        continue
                info["visited"].append(q["type"])
                got = self._inspect(q["type"], q["patch"], otype, info)
                if got:
                    return got
                last = self._look()
                got = self._seen(last, otype)
                if got:
                    return got
                enqueue(last)
                self._cap()
            # planned fronts now in view from here with no uninspected front left near them: done (one that is
            # still there but not served from this stop stays planned for another stop)
            left = self._candidates(last, rank_types)
            for f in [t["front"] for t in stop["items"] if t["kind"] == "front" and not t["front"]["done"]]:
                fc = f["patch"]["c"]
                if (self._in_frame(fc, last) and math.hypot(fc[0] - self._pose_xz()[0], fc[2] - self._pose_xz()[1]) <= 2.0
                        and not any(c["type"] == f["type"] and math.hypot(c["patch"]["c"][0] - fc[0],
                                                                          c["patch"]["c"][2] - fc[2]) <= 0.35
                                    for c in left)):
                    f["done"] = "looked at from a stop"
        for f in [t["front"] for t in stop["items"] if t["kind"] == "front" and not t["front"]["done"]]:
            f["misses"] += 1
            if f["misses"] >= 2:
                f["done"] = "not re-detected"
        return None

    def _run_plan(self, otype, info, ranked, order, rank_types, why):
        """Plan, then visit the stops in order; re-plan only on new information (new fronts seen, a stop that
        could not be reached). Returns the target belief, or None."""
        stops = self._make_plan(otype, ranked, order, info, why)
        while stops:
            if self._search_spent():
                info["budget"] = True
                return None
            stop = stops.pop(0)
            ent = stop["summary"]
            cur, _ = hm.current_cell(self.h, self.gm)
            self._ev(f"search stop {ent['n']} ({ent['x']:.2f},{ent['z']:.2f}): {ent['label']}")
            first = max(stop["items"], key=lambda it: it["q"])["c"] if stop["items"] else None
            if tuple(cur) != tuple(stop["cell"]) and not self._walk_cell(stop["cell"], face=first):
                ent["status"] = "unreachable"
                self._ev(f"search stop {ent['n']}: not reached on the own map - re-planning without it")
                for it in stop["items"]:
                    it["serve"].pop(tuple(stop["cell"]), None)
                if info.get("plans", 0) < MAX_STOP_PLANS:
                    stops = self._make_plan(otype, ranked, order, info, "a stop was not reachable")
                continue
            ent["status"] = "running"
            self._stop = {"cell": tuple(stop["cell"]), "served": 0, "shifted": 0, "split": 0}
            try:
                got = self._at_stop(stop, otype, info, rank_types)
            finally:
                st, self._stop = self._stop, None
            ent.update(status="found" if got else "done", served=st["served"], shifted=st["shifted"],
                       split=st["split"])
            if got:
                return got
            if info["budget"]:
                return None
            if self._new_fronts:
                stops = self._merge_new(stops, otype, ranked, order, info)
        return None

    def _scan_unseen(self, otype, unseen):
        """At a coverage viewpoint: look only where own depth has not yet seen (fewest headings, one turn
        direction) instead of a full look-around. Returns the target belief, or None."""
        x, z = self._pose_xz()
        bear = []
        for c in unseen:
            wx, wz = self.gm.world(c)
            if 0.3 <= math.hypot(wx - x, wz - z) <= VIEW_R:
                bear.append(math.degrees(math.atan2(wx - x, wz - z)) % 360.0)
        yaw = float(self.h.pose()["rotation"]["y"])
        heads = view_headings(bear, yaw)
        self._set_horizon(SWEEP_HORIZON)
        for hd in heads:
            yaw = float(self.h.pose()["rotation"]["y"])
            d = round((hd - yaw + 180.0) % 360.0 - 180.0, 1)
            if abs(d) >= 0.5:
                self.h.move("RotateRight" if d > 0 else "RotateLeft", degrees=abs(d))
            b = self._seen(self._look(), otype)
            if b:
                return b
            self._cap()
        return None

    def _coverage(self, otype, info, ranked, order, rank_types):
        """After the plan: viewpoints that add unseen area (own seen-cells grid), partial looks there, and a new plan
        whenever new containers / places come into view. Ends when no viewpoint adds VIEW_GAIN unseen cells
        (reported as a coverage threshold, never as "absent")."""
        interest = interest_cells(self.gm)
        used = []
        tried = set()                                  # unseen cells a scan aimed at and still did not see (inside
        while True:                                    # furniture / behind walls): never a reason for another stop
            if self._search_spent():
                info["budget"] = True
                return None
            if len(used) >= MAX_VIEWPOINTS:
                break
            unseen = interest - self.seen - tried
            cur, _ = hm.current_cell(self.h, self.gm)
            dmap = grid_dist(self.gm, cur)
            best, bs = None, 0.0
            for v, dv in dmap.items():
                if self.gm.is_temp_blocked(v) or any(abs(v[0] - u[0]) + abs(v[1] - u[1]) <= 2 for u in used):
                    continue
                vx, vz = self.gm.world(v)
                gain = sum(1 for c in unseen if 0.3 <= math.hypot(self.gm.world(c)[0] - vx,
                                                                  self.gm.world(c)[1] - vz) <= VIEW_R)
                if gain < VIEW_GAIN:
                    continue
                sc = gain / (1.0 + dv / 2.0)
                if sc > bs:
                    best, bs = v, sc
            if best is None:
                break
            used.append(best)
            info["viewpoints_planned"] = info.get("viewpoints_planned", 0) + 1
            if not self._walk_cell(best):
                continue
            info["viewpoints"] += 1
            n0 = len(unseen)
            got = self._scan_unseen(otype, unseen)
            bx, bz = self.gm.world(best)
            tried |= {c for c in unseen - self.seen
                      if 0.3 <= math.hypot(self.gm.world(c)[0] - bx, self.gm.world(c)[1] - bz) <= VIEW_R}
            self._ev(f"coverage viewpoint {info['viewpoints']} ({bx:.2f},{bz:.2f}): "
                     f"{n0 - len(unseen - self.seen)} unseen cell(s) now seen")
            if got:
                return got
            got = self._run_plan(otype, info, ranked, order, rank_types, "new view")
            if got or info["budget"]:
                return got
        info["unseen_left"] = len(interest - self.seen)
        return None

    # -------------------------------------------------------------- composites
    def _clear_basin(self, keep):
        """B1b (live 2026-10-10, g-l3-cookware): a basin that still holds the dish washed before has no room for a pot
        ("No valid positions to place object found": every put refused). The agent KNOWS what it left in the basin (its
        own put record), so with empty hands it moves those dishes to a counter first. Console policy only; a failure
        to do so is logged and the wash goes on (it may still fit)."""
        if not self.exhaustive or self.h.holding:
            return
        for t in [t for t, r in list(self.put_rec.items())
                  if t != keep and r.get("rec") == "SinkBasin" and not r.get("dead")]:
            try:
                self._ev(f"making room in the sink: moving the {t} I left there to a counter")
                self._interact("pickup", t, belief=self._placed(t))
                self._put("CounterTop")
            except _Fail as e:
                self._ev(f"could not clear the {t} out of the sink ({e.cause}); washing on")
                if self.h.holding:
                    break                              # still carrying it: the wash step's own pickup handles that

    def _ensure_holding(self, otype):
        if self.h.holding == otype:
            return
        self._interact("pickup", otype)

    def _placed(self, otype):
        """The belief this agent made by putting ``otype`` somewhere (it knows where it left it), else None."""
        b = next((b for b in self.om.lookup(otype) if b["status"] == "placed" and b.get("xyz")), None)
        return b or self._put_belief(otype)

    def _put_belief(self, otype):
        """B1 (live 2026-10-09: the dish was re-seen in the sink, then 'not seen in 2 clear views' once the water ran,
        so memory contradicted it): what the agent KNOWS from its own put -- it left ``otype`` there. An aim reference
        (the put pixel's surface point), not a sighting: nothing is written to memory from it. None unless the console
        policy kept a record that is still alive (not picked up since, not given up on)."""
        rec = self.put_rec.get(otype) if self.exhaustive else None
        if not rec or rec.get("dead"):
            return None
        return {"track": None, "type": otype, "status": "placed", "xyz": rec["p0"], "state": None,
                "place": rec["rec"], "step": rec["step"], "conf": 0.0, "own_put": True}

    def _go(self, otype):
        """go_to / search_for: get near a sighting of ``otype`` and see it again."""
        if self.exhaustive:
            return self._go_exhaustive(otype)
        st, k = {"alt": 0, "dpitch": 0.0}, 0
        b = self._locate(otype)
        st["xyz"] = b["xyz"]
        self._goto(b["xyz"])
        self._face(b["xyz"])
        while True:
            obs = self._look()
            if self._pick(obs, otype, b["xyz"]) is not None:
                self.verify.append("seen")
                return {"verify": "seen", "distance": round(self._hdist(b["xyz"]), 2)}
            self._spend(_Soft("not_detected", f"{otype} not in view at its remembered place"))
            k += 1
            self._recover(k, b["xyz"], st)

    def _go_exhaustive(self, otype):
        """Exhaustive mode: a remembered sighting not re-seen after the local looks is set aside and the next one,
        else the full search, is tried. No action retries are spent on searching."""
        st = {"alt": 0, "dpitch": 0.0}
        for _ in range(MAX_RELOCATE + 1):
            from_memory = bool(self._rank(self.om.lookup(otype)))
            b = self._locate(otype)
            st["xyz"] = b["xyz"]
            if (SIGHT_NOW and b.get("status") == "trusted" and self._last_look == self.h.step
                    and int(b.get("step") or -1) == self.h.step and self._hdist(b["xyz"]) <= REACH):
                # seen in the CURRENT frame, within reach: that is the sighting search_for asks for (Spoon, baseline:
                # re-facing it at a steeper pitch lost the detection five times)
                self.verify.append("seen")
                for e in self.opened_by_me:
                    if e["why"] in ("look", "search") and math.dist(e["xyz"], b["xyz"]) < 1.0:
                        e["why"], e["target"] = "search-found", otype
                self._ev(f"{otype} in view now ({self._hdist(b['xyz']):.2f} m): sighting confirmed")
                return {"verify": "seen", "distance": round(self._hdist(b["xyz"]), 2)}
            try:
                self._goto(b["xyz"])
            except _Fail as e:
                if not from_memory:
                    raise
                self._set_aside(b, f"unreachable ({e.why})")
                continue
            if b.get("status") == "placed" and b.get("place") in OPENABLE:
                self._open_container(b["place"], b["xyz"], why="look")
            self._face(b["xyz"])
            for k in range(LOCAL_LOOKS + 1):
                obs = self._look()
                if self._pick(obs, otype, b["xyz"]) is not None:
                    self.verify.append("seen")
                    for e in self.opened_by_me:          # it is in a container I opened: close it after the pickup
                        if e["why"] in ("look", "search") and math.dist(e["xyz"], b["xyz"]) < 1.0:
                            e["why"], e["target"] = "search-found", otype
                    return {"verify": "seen", "distance": round(self._hdist(b["xyz"]), 2)}
                self._cap()
                if k < LOCAL_LOOKS:
                    self._local_look(k + 1, b, st)
            self._set_aside(b, f"{otype} not re-seen after {LOCAL_LOOKS} local looks")
        raise _Fail("not_detected", f"{otype} not re-seen after {MAX_RELOCATE + 1} tries at remembered places")

    def _drop(self):
        if not self.h.holding:
            raise _Fail("action_refused", "hand is empty")
        k = 0
        while True:
            r = self.h.act("DropHandObject")
            self._ev(f"DropHandObject: {'ok' if r['ok'] else r['err']}")
            if r["ok"]:
                self.om.on_drop(True, self.h.pose(), step=self.h.step)
                self._world_ver += 1
                self.verify.append("success-bool")
                return {"verify": "success-bool"}
            self._spend(_Soft("action_refused", r["err"]))
            k += 1
            (self.h.move("RotateRight", degrees=90) if k == 1 else self.h.move("MoveBack") if k == 2
             else self.h.move("LookDown", degrees=PITCH_ALT))

    def _close_all(self):
        self._sweep()
        todo = []
        for t in sorted(OPENABLE & set(self.type_list)):
            for b in self._rank(self.om.lookup(t)):
                if state_of({"state": b["state"]}, "open") is True:
                    todo.append((t, b))
        self._ev(f"close_all: {len(todo)} seen open: {[t for t, _ in todo]}")
        for t, b in todo:
            self._interact("close", t, belief=b)
        self.verify.append("seen-open-closed" if todo else "nothing-seen-open")
        return {"verify": "seen-open-closed" if todo else "nothing-seen-open", "closed": [t for t, _ in todo]}

    # -------------------------------------------------------------- dispatch
    def exec_step(self, step: dict, upcoming=None) -> dict:
        """Run one plan step honestly. Returns ``{ok, cause, retries, detail}``. ``upcoming`` = the plan's later
        steps (tidiness: a container a later step still needs is left open)."""
        self._reset_step()
        if getattr(self, "_start_ev", None):
            self.events.extend(f"{e} (episode start)" for e in self._start_ev)
            self._start_ev = None
        action = str(step.get("action") or "").strip().lower()
        action = ALIASES.get(action, action)
        self._step_action = action
        self._step_purpose = str(step.get("purpose") or "").strip().lower() or None
        n_ident = len(self.identity_log)
        self.upcoming = list(upcoming or [])
        ot = step.get("object")
        to = step.get("to")
        arg = step.get("arg")
        out, cause, why, under = {}, None, "", None
        try:
            out = self._dispatch(action, ot, to, arg) or {}
        except _Fail as e:
            cause, why, under = e.cause, e.why, (e.under or e.cause)
        if self.tidy:
            try:
                self._tidy_step_end()
            except _Fail as e:
                self._ev(f"tidy stopped: {e.why}")
        ok = cause is None
        detail = {**out, "action": action, "object": ot, "to": to, "prims": self.h.step - self._step0,
                  "verify": list(self.verify), "sub": list(self.sub), "events": list(self.events), "why": why,
                  "holding": self.h.holding}
        if self.search_info:
            detail["search"] = dict(self.search_info)
        if len(self.identity_log) > n_ident:
            detail["identity"] = [dict(r) for r in self.identity_log[n_ident:]]
        if self.on_event is not None:                  # B1 (flag on only): structured cause for the memory events
            detail["under"] = under
        if self.tidy or self.tidy_log:
            detail["tidy"] = list(self.tidy_log)
            detail["left_open"] = [e["type"] for e in self.opened_by_me]
        return {"ok": ok, "cause": cause, "retries": self.retries, "detail": detail}

    def _pickup_for_purpose(self, ot, purpose):
        """P: pick up ``ot`` for a stated purpose. The asked type goes first (remembered sighting, else a full search); if it
        cannot be found, a type of the same declared tool family that this agent has SEEN (memory, including what the search
        sweeps added) is taken instead and the substitution is recorded. Nothing is taken that the agent has not seen."""
        fam = [ot] + [t for t in PURPOSE_TYPES[purpose] if t != ot]
        for t in fam:
            if t != ot and not self.om.lookup(t):
                continue
            try:
                out = self._interact("pickup", t)
            except _Fail as e:
                if e.cause != "not_detected" or t == fam[-1]:
                    raise
                self._ev(f"purpose '{purpose}': no {t} found - trying the next tool of that kind that I have seen")
                continue
            if t != ot:
                out = {**out, "substituted": {"asked": ot, "got": t, "purpose": purpose}}
                self._ev(f"purpose '{purpose}': took a {t} in place of the {ot}")
            return out
        raise _Fail("not_detected", f"no {ot} found for '{purpose}'")

    def _dispatch(self, action, ot, to, arg):
        if action in ("done", "pass", "wait", "noop"):
            return {}
        if action in NAV:
            ok = self.h.move(NAV[action])
            if not ok:
                raise _Fail("action_refused", f"{action} refused")
            return {}
        if action == "drop":
            return self._drop()
        if action == "close_all":
            return self._close_all()
        if not ot:
            raise _Fail("not_detected", f"{action} needs an object")
        if action in ("go_to", "search_for"):
            return self._go(ot)
        if action == "pickup":
            if self.h.holding == ot:
                return {"verify": "already"}
            if self.purpose_search and self._step_purpose in PURPOSE_TYPES:
                return self._pickup_for_purpose(ot, self._step_purpose)
            return self._interact("pickup", ot)
        if action == "put_in":
            return self._put(ot)
        if action == "open" and self.pose_first and ot in OPENABLE:
            return self._open_posed(ot)
        if action == "close" and self.pose_first and ot in OPENABLE and any(e["type"] == ot for e in self.opened_by_me):
            e = max([e for e in self.opened_by_me if e["type"] == ot], key=lambda e: e["step"])
            res = self._tidy_close(e, "(plan step)")
            if res not in ("closed", "already"):
                raise _Fail("action_refused", f"close {ot} failed (from the pose it was opened from)")
            self.verify.append("closed-from-open-pose")
            return {"verify": res}
        if action == "pick_and_place":
            self._ensure_holding(ot)
            return self._put(to)
        if action in ("heat", "cook"):
            self._ensure_holding(ot)
            self._put("Microwave")
            self._interact("close", "Microwave")
            self._interact("toggle_on", "Microwave")
            self._interact("toggle_off", "Microwave")
            self._interact("open", "Microwave")
            self._interact("pickup", ot, belief=self._placed(ot))
            return {"procedure": "heated in the microwave (temperature not visible; goal check decides)"}
        if action == "wash":
            self._clear_basin(ot)
            self._ensure_holding(ot)
            self._put("SinkBasin")
            self._interact("toggle_on", "Faucet")
            failed = None
            try:
                self._interact("clean", ot, belief=self._placed(ot))
            except _Fail as e:
                if not self.exhaustive or e.cause not in ("retry_exhausted", "not_detected", "unreachable",
                                                          "action_refused"):
                    raise
                failed = e
            self._interact("toggle_off", "Faucet")
            if failed is not None:
                # B1 (live 2026-10-09: wash Pan): running water can clean a dish by itself, after which an explicit
                # clean is refused everywhere; with the water off the dish is often visible again -> read it
                if not self._seen_clean(ot):
                    raise failed
                self.verify.append("state-after-water")
                return {"procedure": "left in the sink under the running water; read clean by sight afterwards"}
            return {"procedure": "rinsed and scrubbed in the sink (left in the sink)"}
        if action == "fill":
            liquid = str(arg or to or "water").strip().lower()
            return self._interact("fill", ot, fillLiquid=liquid if liquid in LIQUIDS else "water")
        if action == "slice":
            self._cutting_tool_check(ot)                 # A26 item 2: before any primitive, 0 primitives, not retried
        if action in VERBS:
            return self._interact(action, ot)
        raise _Fail("action_refused", f"unknown action {action!r}")
