"""Step 4 (s2-map) -- the agent's own floor map, built by walking. Kept deliberately simple.

The agent constructs a start-relative 0.25 m navigation grid by depth-first physical exploration, using only its
known self-pose (``HonestAgent.pose``, tag ``pose``) and the success bool of each move (tag ``actuator``). No
teleport, no simulator path queries. The simulator's reachable-position list is read once, AFTER the build, and only
for the coverage report (the agent's map-build door, tag ``map-build``) -- never to explore or to stop.

* Build: from the reset pose, align the heading to a multiple of 90 deg, then an iterative depth-first walk with
  ``MoveAhead`` (one cell) and 90-degree turns. A successful move marks the neighbour cell free; a failed move records
  a blocked EDGE (undirected), not a blocked cell. Backtracking faces child->parent and steps ahead. Every move is
  checked: the displacement must match the heading, a successful move must land on the lattice, and a failed move
  must leave the pose unchanged; a violation aborts the build (the partial map is kept, status says why).
* Temporary blocks (2026-10-08): while a door the agent opened stands open, the cells its own estimate says the door
  sweeps are blocked under a key (``block_cells`` / ``unblock_cells``); planning never enters them. Not persisted.
  2026-10-09 ("fix: unreachable failures", Astra-reviewed): a key carries HARD cells (the open leaf's estimated
  footprint -- never entered) and SOFT cells (the rest of the uncertain sweep -- A* pays ``SOFT_COST`` extra steps to
  enter one, so routes go round it when a short detour exists, but the agent is never walled in by an estimate).
* Episode start node (2026-10-09): ``adopt_here`` -- the agent stands on a cell its map does not have as free (it was
  placed there, e.g. by a harness's setup). Its own pose proves that cell; for this episode it is a free node and each
  incident edge to a free neighbour that the map recorded BLOCKED may be re-tried once (logged; a refusal re-blocks it
  for good, a success proves the passage). The map file never changes.
* Use: ``GridMap.astar`` plans on the agent's own grid (4-connected, blocked edges removed; unprobed edges between
  two free cells are assumed passable -- optimistic, corrected by ``walk_path``'s replanning), ``cells_near`` gives
  stand-off cells around a target point, and ``walk_path`` turns and steps, adding a blocked edge and replanning from
  its verified pose on a failed step, up to 3 times per call. With ``strafe`` (2026-10-08, the console's exhaustive
  policy only) a walk keeps one lattice heading and side- / back-steps instead of turning at every path corner.

Pure stdlib except :func:`render_png` (PIL, lazy). Spec: docs/step4-plan.md (s2). Grid helpers reused from navmap.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from typing import Iterable, Optional

from navmap import cell_to_world, world_to_cell

SCHEMA = 1
CELL = 0.25
#: heading index k <-> yaw 90*k deg; AI2-THOR yaw 0 faces +z, 90 faces +x.
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
POS_TOL = 0.05          # metres: displacement / lattice residual tolerance
YAW_TOL = 1.0           # degrees
#: extra A* steps for entering a SOFT cell (2026-10-09): a declared preference, not a measurement -- a detour of up to
#: 4 cells (1 m) is preferred to crossing floor an open door MAY occupy; a refused move there still re-plans as usual.
SOFT_COST = 4
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "step4")


class MapAbort(RuntimeError):
    """A move broke an invariant (wrong displacement, off-lattice landing, a 'failed' move that moved)."""


def edge(a, b) -> tuple:
    """Undirected edge key between two cells."""
    a, b = tuple(a), tuple(b)
    return (a, b) if a <= b else (b, a)


def _wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------------------------- the map
class GridMap:
    """Free cells + blocked edges on a start-relative square grid. Cells are integer ``(gx, gz)`` tuples."""

    def __init__(self, scene: str = "FloorPlan1", cell: float = CELL, origin=(0.0, 0.0), free: Iterable = (),
                 blocked_edges: Iterable = (), info: Optional[dict] = None):
        self.scene, self.cell, self.origin = scene, float(cell), (float(origin[0]), float(origin[1]))
        self.free = {tuple(c) for c in free}
        self.blocked_edges = {edge(a, b) for a, b in blocked_edges}
        self.info = dict(info or {})
        #: cells TEMPORARILY blocked by the agent's own estimate of an open door's swept floor (key -> cells),
        #: never persisted; A*, goal choice and neighbour moves skip them until the key is unblocked (2026-10-08,
        #: "fix: pose first, then open")
        self.temp_blocked = {}
        self._tb = frozenset()
        #: soft cells per key (2026-10-09): passable at SOFT_COST extra steps; never persisted
        self.temp_soft = {}
        self._ts = frozenset()
        #: episode start node (2026-10-09): cells proven by the agent's own pose though not on the map, and the
        #: recorded-blocked edges into them that may be re-tried once; never persisted
        self.here_free = set()
        self.retry_edges = set()

    # -------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "scene": self.scene, "cell": self.cell, "origin": list(self.origin),
                "free": sorted([list(c) for c in self.free]),
                "blocked_edges": sorted([[list(a), list(b)] for a, b in self.blocked_edges]),
                "info": self.info}

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(DEFAULT_DIR, f"{self.scene}.map.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path_or_scene: str = "FloorPlan1", cell: float = CELL) -> "GridMap":
        path = path_or_scene if path_or_scene.endswith(".json") else os.path.join(DEFAULT_DIR, f"{path_or_scene}.map.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"map schema {d.get('schema')!r} != {SCHEMA}")
        if abs(float(d["cell"]) - cell) > 1e-9:
            raise ValueError(f"map cell {d['cell']} != requested {cell}")
        return cls(d["scene"], d["cell"], d["origin"], (tuple(c) for c in d["free"]),
                   ((tuple(a), tuple(b)) for a, b in d["blocked_edges"]), d.get("info"))

    # -------------------------------------------------------------- geometry
    def cell_of(self, x: float, z: float) -> tuple:
        return world_to_cell(x, z, self.cell, self.origin)

    def world(self, c) -> tuple:
        return cell_to_world(c[0], c[1], self.cell, self.origin)

    def residual(self, x: float, z: float) -> float:
        """Distance (m) from ``(x, z)`` to the nearest lattice point."""
        cx, cz = self.world(self.cell_of(x, z))
        return math.hypot(x - cx, z - cz)

    def add_blocked(self, a, b) -> None:
        e = edge(a, b)
        self.blocked_edges.add(e)
        getattr(self, "retry_edges", set()).discard(e)    # a refusal ends a re-try (never re-tried again)

    def block_cells(self, key, cells, keep=(), soft=()) -> frozenset:
        """Temporarily block ``cells`` (HARD: never entered) and mark ``soft`` (passable at ``SOFT_COST``) under
        ``key`` (an open door); ``keep`` (the agent's own cell) is never blocked, so it can always back off. Replaces
        an earlier block under the same key. Returns the hard cells."""
        keep = {tuple(k) for k in keep}
        ok = lambda c: tuple(c) in self.free and tuple(c) not in keep            # noqa: E731
        hard = frozenset(tuple(c) for c in cells if ok(c))
        self.temp_blocked[key] = hard
        self.temp_soft[key] = frozenset(tuple(c) for c in soft if ok(c) and tuple(c) not in hard)
        self._refresh_temp()
        return hard

    def _refresh_temp(self):
        self._tb = frozenset().union(*self.temp_blocked.values()) if self.temp_blocked else frozenset()
        ts = getattr(self, "temp_soft", {})
        self._ts = (frozenset().union(*ts.values()) - self._tb) if ts else frozenset()

    def unblock_cells(self, key) -> frozenset:
        got = self.temp_blocked.pop(key, frozenset())
        got = got | getattr(self, "temp_soft", {}).pop(key, frozenset())
        self._refresh_temp()
        return got

    def is_temp_blocked(self, c) -> bool:
        return tuple(c) in getattr(self, "_tb", ())

    def is_soft(self, c) -> bool:
        return tuple(c) in getattr(self, "_ts", ())

    def soft_cells(self) -> frozenset:
        return getattr(self, "_ts", frozenset())

    def is_free(self, c) -> bool:
        """On the map, or proven this episode by the agent standing there (``adopt_here``)."""
        return tuple(c) in self.free or tuple(c) in getattr(self, "here_free", ())

    def adopt_here(self, c) -> list:
        """The agent stands on ``c`` (own pose, on the lattice) and ``c`` is not free on the map: make it a free node
        for this episode; every recorded-blocked edge from ``c`` to a free neighbour becomes re-triable ONCE (a
        refused move re-blocks it). Returns the re-triable edges. Never persisted (``to_dict`` saves ``free`` only)."""
        c = tuple(c)
        if c in self.free:
            return []
        self.here_free.add(c)
        out = []
        for dx, dz in DIRS:
            n = (c[0] + dx, c[1] + dz)
            e = edge(c, n)
            if n in self.free and e in self.blocked_edges:
                self.retry_edges.add(e)
                out.append(e)
        return out

    def drop_here(self):
        self.here_free = set()
        self.retry_edges = set()

    def _passable(self, c, n) -> bool:
        e = edge(c, n)
        return e not in self.blocked_edges or e in getattr(self, "retry_edges", ())

    def neighbours(self, c) -> list:
        out = []
        tb = getattr(self, "_tb", ())
        for dx, dz in DIRS:
            n = (c[0] + dx, c[1] + dz)
            if self.is_free(n) and self._passable(c, n) and n not in tb:
                out.append(n)
        return out

    # -------------------------------------------------------------- planning
    def astar(self, start, goals) -> Optional[list]:
        """Shortest 4-connected path ``[start, ..., goal]`` on the agent's own grid to any of ``goals``; None if none."""
        start = tuple(start)
        tb = getattr(self, "_tb", ())
        ts = getattr(self, "_ts", ())
        goals = {tuple(g) for g in goals if self.is_free(g) and tuple(g) not in tb}
        if not goals:
            return None
        if start in goals:
            return [start]

        def h(c):
            return min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)

        openq, came, g = [(h(start), 0, start)], {start: None}, {start: 0}
        while openq:
            _, gc, c = heapq.heappop(openq)
            if c in goals:
                path = [c]
                while came[path[-1]] is not None:
                    path.append(came[path[-1]])
                return path[::-1]
            if gc > g.get(c, 1 << 30):
                continue
            for n in self.neighbours(c):
                ng = gc + 1 + (SOFT_COST if n in ts else 0)   # soft: an open door MAY stand there (2026-10-09)
                if ng < g.get(n, 1 << 30):
                    g[n], came[n] = ng, c
                    heapq.heappush(openq, (ng + h(n), ng, n))
        return None

    def cells_near(self, xyz, rmin: float = 0.5, rmax: float = 1.0) -> list:
        """Free cells whose centre lies ``rmin..rmax`` m (horizontal) from ``xyz`` (x,y,z or x,z), nearest first.

        Proximity only: a returned cell is a candidate stand-off spot, not a guarantee of view or reach.
        """
        x, z = (xyz[0], xyz[2]) if len(xyz) >= 3 else (xyz[0], xyz[1])
        out = []
        tb = getattr(self, "_tb", ())
        for c in self.free:
            if c in tb:
                continue
            cx, cz = self.world(c)
            d = math.hypot(cx - x, cz - z)
            if rmin <= d <= rmax:
                out.append((d, c))
        return [c for _, c in sorted(out)]


# ----------------------------------------------------------------------------------------------- motion on the grid
def _pose_xz_yaw(h):
    p = h.pose()
    return float(p["position"]["x"]), float(p["position"]["z"]), float(p["rotation"]["y"])


def align_heading(h) -> int:
    """Turn to the nearest multiple of 90 deg; return the heading index. Raises MapAbort if the turn does not hold."""
    _, _, yaw = _pose_xz_yaw(h)
    target = (round(yaw / 90.0) * 90) % 360
    delta = _wrap(target - yaw)
    if abs(delta) > YAW_TOL / 10:
        h.move("RotateRight" if delta > 0 else "RotateLeft", degrees=abs(delta))
    _, _, yaw = _pose_xz_yaw(h)
    if abs(_wrap(yaw - target)) > YAW_TOL:
        raise MapAbort(f"heading alignment failed: yaw {yaw:.1f} != {target}")
    return int(target // 90) % 4


def turn_to(h, k_cur: int, k_new: int) -> int:
    diff = (k_new - k_cur) % 4
    if diff == 1:
        h.move("RotateRight", degrees=90)
    elif diff == 2:
        h.move("RotateRight", degrees=180)
    elif diff == 3:
        h.move("RotateLeft", degrees=90)
    if diff:
        _, _, yaw = _pose_xz_yaw(h)
        if abs(_wrap(yaw - 90 * k_new)) > YAW_TOL:
            raise MapAbort(f"turn failed: yaw {yaw:.1f} != {90 * k_new}")
    return k_new


REL_MOVES = ("MoveAhead", "MoveRight", "MoveBack", "MoveLeft")     # (direction - heading) % 4


def step(h, gm: GridMap, k: int, strict_lattice: bool = True, d: Optional[int] = None):
    """One move of one cell along lattice direction ``d`` (default: the heading ``k``, i.e. ``MoveAhead``; else the
    side / back move that goes there without turning). Returns ``(ok, cell_after)``; checks the invariants."""
    d = k if d is None else d
    x0, z0, _ = _pose_xz_yaw(h)
    ok = h.move(REL_MOVES[(d - k) % 4], moveMagnitude=gm.cell)
    x1, z1, _ = _pose_xz_yaw(h)
    dx, dz = x1 - x0, z1 - z0
    k = d
    if ok:
        ex, ez = DIRS[k][0] * gm.cell, DIRS[k][1] * gm.cell
        if math.hypot(dx - ex, dz - ez) > POS_TOL:
            raise MapAbort(f"move displacement ({dx:.3f},{dz:.3f}) != expected ({ex},{ez})")
        if strict_lattice and gm.residual(x1, z1) > POS_TOL:
            raise MapAbort(f"landed off the lattice at ({x1:.3f},{z1:.3f})")
    elif math.hypot(dx, dz) > POS_TOL:
        raise MapAbort(f"a failed move still moved ({dx:.3f},{dz:.3f})")
    return ok, gm.cell_of(x1, z1)


# ----------------------------------------------------------------------------------------------- build
def explore(h, scene: str = "FloorPlan1", cell: float = CELL, budget: int = 6000, log=None) -> GridMap:
    """Depth-first physical exploration from the current pose. Stops on stack exhaustion, budget or invariant break.

    ``budget`` caps the number of primitive sends. ``log(msg)`` (optional) gets progress lines. The map's ``info``
    records status (``complete`` | ``budget`` | ``abort: ...``), sends used and the start pose.
    """
    x, z, yaw = _pose_xz_yaw(h)
    gm = GridMap(scene, cell, origin=(x, z))
    start = gm.cell_of(x, z)
    gm.free.add(start)
    s0 = h.step
    status = "complete"
    try:
        k = align_heading(h)
        # frame: [cell, headings still to try, heading that led here from the parent (None at the root)]
        stack = [[start, [(k + j) % 4 for j in (0, 1, 3, 2)], None]]
        while stack:
            if h.step - s0 >= budget:
                status = "budget"
                break
            c, todo, came = stack[-1]
            if todo:
                kk = todo.pop(0)
                n = (c[0] + DIRS[kk][0], c[1] + DIRS[kk][1])
                if n in gm.free or edge(c, n) in gm.blocked_edges:
                    continue
                k = turn_to(h, k, kk)
                ok, got = step(h, gm, k)
                if ok:
                    if got != n:
                        raise MapAbort(f"expected cell {n}, landed in {got}")
                    gm.free.add(n)
                    stack.append([n, [(kk + j) % 4 for j in (0, 1, 3)], kk])   # never probe straight back
                    if log and len(gm.free) % 25 == 0:
                        log(f"free={len(gm.free)} blocked_edges={len(gm.blocked_edges)} sends={h.step - s0}")
                else:
                    if got != c:
                        raise MapAbort(f"failed move left cell {c} for {got}")
                    gm.add_blocked(c, n)
            else:
                stack.pop()
                if came is None:
                    break
                k = turn_to(h, k, (came + 2) % 4)
                ok, got = step(h, gm, k)
                parent = stack[-1][0]
                if not ok or got != parent:
                    raise MapAbort(f"backtrack {c}->{parent} failed (ok={ok}, at {got})")
    except MapAbort as e:
        status = f"abort: {e}"
    gm.info.update({"status": status, "sends": h.step - s0, "start": {"x": x, "z": z, "yaw": yaw},
                    "method": "depth-first walk, MoveAhead + 90-degree turns, own pose + move success only",
                    "free_cells": len(gm.free), "blocked_edges": len(gm.blocked_edges)})
    return gm


# ----------------------------------------------------------------------------------------------- walking
def current_cell(h, gm: GridMap):
    x, z, _ = _pose_xz_yaw(h)
    return gm.cell_of(x, z), gm.residual(x, z)


def walk_heading(path, k_cur: int, strafe: float, cell: float = CELL, face_yaw: Optional[float] = None,
                 yaw: Optional[float] = None) -> int:
    """The lattice heading to walk ``path`` with when side / back moves are allowed (``strafe`` > 0, 2026-10-08
    "systematic search route"): a short walk (<= ``strafe`` m) keeps the current heading (no turn at all); a longer
    one faces the direction most of its moves go (forward-facing travel; the rest are side-steps)."""
    moves = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    moves = [DIRS.index(m) for m in moves if m in DIRS]
    if not moves or len(moves) * cell <= strafe + 1e-9:
        # short walk: the lattice heading with the least turn now + turn on arrival to where it will look (Astra:
        # minimise the sum; continuing the turn is only a tie-break)
        if face_yaw is None:
            return k_cur
        y0 = 90.0 * k_cur if yaw is None else yaw
        return min(range(4), key=lambda i: (abs(_wrap(90.0 * i - y0)) + abs(_wrap(face_yaw - 90.0 * i)),
                                            abs(_wrap(face_yaw - 90.0 * i))))
    n = [moves.count(i) for i in range(4)]
    return max(range(4), key=lambda i: (n[i], -min((i - k_cur) % 4, (k_cur - i) % 4)))


def walk_path(h, gm: GridMap, path: Optional[list], goals=None, retries: int = 3, strafe: float = 0.0,
              face=None) -> dict:
    """Follow ``path`` on the grid; on a failed step add the blocked edge and replan (from the verified pose) to
    ``goals`` (default ``{path[-1]}``), at most ``retries`` times per call. Returns
    ``{ok, cell, replans, steps, why, side, back}``. Off-lattice start poses are tolerated (cells by rounding).
    ``strafe`` > 0 (the console's exhaustive policy): no turn per path corner -- one heading per walk
    (``walk_heading``) and side / back moves for the rest; 0 (default, the s5 runner) = turn and MoveAhead.
    """
    goals = {tuple(g) for g in (goals or ([path[-1]] if path else []))}
    replans, steps = 0, 0
    side = back = 0
    cur, _ = current_cell(h, gm)
    if not goals:
        return {"ok": False, "cell": cur, "replans": 0, "steps": 0, "why": "no goal"}
    if strafe > 0 and cur in goals:
        return {"ok": True, "cell": cur, "replans": 0, "steps": 0, "why": "", "side": 0, "back": 0}   # no snap turn
    if not path or tuple(path[0]) != cur:
        path = gm.astar(cur, goals)
    try:
        if strafe > 0 and path:
            # ONE turn, straight from the current yaw to the walk heading (no snap to the nearest lattice heading
            # first and a turn back after): the heading is chosen from the path and where it will look on arrival
            fy = None
            if face is not None:
                gx, gz = gm.world(path[-1])
                fy = math.degrees(math.atan2(face[0] - gx, (face[2] if len(face) >= 3 else face[1]) - gz)) % 360.0
            _, _, yaw = _pose_xz_yaw(h)
            k = walk_heading(path, int(round(yaw / 90.0)) % 4, strafe, gm.cell, fy, yaw)
            d = _wrap(90.0 * k - yaw)
            if abs(d) > YAW_TOL / 10:
                h.move("RotateRight" if d > 0 else "RotateLeft", degrees=round(abs(d), 1))
            _, _, yaw = _pose_xz_yaw(h)
            if abs(_wrap(yaw - 90 * k)) > YAW_TOL:
                # the turn was refused (e.g. a held object would hit something): side / back steps need no turn
                # if it is on the lattice already, else the smallest alignment
                k0 = int(round(yaw / 90.0)) % 4
                if abs(_wrap(yaw - 90 * k0)) > YAW_TOL:
                    k0 = align_heading(h)
                k = k0
        else:
            k = align_heading(h)
    except MapAbort as e:
        return {"ok": False, "cell": cur, "replans": 0, "steps": 0, "why": str(e)}
    while True:
        if cur in goals:
            return {"ok": True, "cell": cur, "replans": replans, "steps": steps, "why": "", "side": side,
                    "back": back}
        if not path:
            return {"ok": False, "cell": cur, "replans": replans, "steps": steps, "why": "no path on own map",
                    "side": side, "back": back}
        failed = False
        if strafe > 0 and replans:
            try:
                k = turn_to(h, k, walk_heading(path, k, strafe, gm.cell))
            except MapAbort as e:
                return {"ok": False, "cell": cur, "replans": replans, "steps": steps, "why": str(e), "side": side,
                        "back": back}
        for n in path[1:]:
            n = tuple(n)
            d = (n[0] - cur[0], n[1] - cur[1])
            if d not in DIRS:
                failed = True
                break
            try:
                if strafe > 0:
                    rel = (DIRS.index(d) - k) % 4
                    side += rel in (1, 3)
                    back += rel == 2
                    ok, got = step(h, gm, k, strict_lattice=False, d=DIRS.index(d))
                else:
                    k = turn_to(h, k, DIRS.index(d))
                    ok, got = step(h, gm, k, strict_lattice=False)
            except MapAbort as e:
                return {"ok": False, "cell": cur, "replans": replans, "steps": steps, "why": str(e), "side": side,
                        "back": back}
            steps += 1
            if ok and got == n:
                cur = n
                continue
            if not ok:
                gm.add_blocked(cur, n)
            cur = got
            failed = True
            break
        if not failed:
            continue                                   # reached path end; loop re-checks goals
        replans += 1
        if replans > retries:
            return {"ok": False, "cell": cur, "replans": replans - 1, "steps": steps, "why": "retries exhausted",
                    "side": side, "back": back}
        path = gm.astar(cur, goals)


def goto_near(h, gm: GridMap, xyz, rmin: float = 0.5, rmax: float = 1.0, retries: int = 3,
              strafe: float = 0.0) -> dict:
    """Walk to the cheapest free cell ``rmin..rmax`` m from ``xyz`` (the agent's own estimate of a target)."""
    goals = gm.cells_near(xyz, rmin, rmax)
    cur, _ = current_cell(h, gm)
    return walk_path(h, gm, gm.astar(cur, goals), goals=goals, retries=retries, strafe=strafe,
                     face=xyz if strafe > 0 else None)


def face_point(h, xyz) -> bool:
    """Turn (yaw only) to face ``xyz`` from the current pose. Leaves the heading off the 90-degree lattice;
    the next ``walk_path`` re-aligns."""
    x, z, yaw = _pose_xz_yaw(h)
    target = math.degrees(math.atan2(xyz[0] - x, (xyz[2] if len(xyz) >= 3 else xyz[1]) - z)) % 360
    delta = _wrap(target - yaw)
    if abs(delta) < 0.5:
        return True
    return h.move("RotateRight" if delta > 0 else "RotateLeft", degrees=abs(delta))


# ----------------------------------------------------------------------------------------------- report
def coverage(gm: GridMap, ref_points: list) -> dict:
    """Set-based coverage of the discovered map against reference points (report only, never a build input)."""
    res = [gm.residual(p["x"], p["z"]) for p in ref_points]
    ref = {gm.cell_of(p["x"], p["z"]) for p in ref_points}
    hit = ref & gm.free
    return {"discovered_free": len(gm.free), "reference_cells": len(ref), "reference_points": len(ref_points),
            "matched": len(hit), "coverage": (len(hit) / len(ref)) if ref else 0.0,
            "unmatched_discovered": len(gm.free - ref), "missed_reference": sorted(ref - gm.free),
            "ref_lattice_residual_max": max(res) if res else 0.0}


def render_png(gm: GridMap, path: str, ref_points: Optional[list] = None, px: int = 22, title: str = "") -> str:
    """Small top-down picture: reference cells (grey outline), discovered free cells (teal), blocked edges (red),
    start (dark dot). +x right, +z up."""
    from PIL import Image, ImageDraw
    ref = {gm.cell_of(p["x"], p["z"]) for p in (ref_points or [])}
    allc = gm.free | ref
    xs, zs = [c[0] for c in allc], [c[1] for c in allc]
    x0, x1, z0, z1 = min(xs) - 1, max(xs) + 1, min(zs) - 1, max(zs) + 1
    top = 26 if title else 6
    W, H = (x1 - x0 + 1) * px, (z1 - z0 + 1) * px + top
    im = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(im)

    def box(c):
        u, v = (c[0] - x0) * px, (z1 - c[1]) * px + top
        return (u + 1, v + 1, u + px - 2, v + px - 2)

    for c in ref - gm.free:
        d.rectangle(box(c), outline=(170, 170, 170), width=2)
    for c in gm.free:
        d.rectangle(box(c), fill=(46, 139, 138) if c in ref or not ref else (230, 160, 40))
    for a, b in gm.blocked_edges:
        (ua, va, ub, vb), (uc, vc, ud, vd) = box(a), box(b)
        mx, my = (ua + ub + uc + ud) / 4, (va + vb + vc + vd) / 4
        if a[0] != b[0]:
            d.line((mx, my - px * 0.4, mx, my + px * 0.4), fill=(200, 40, 40), width=2)
        else:
            d.line((mx - px * 0.4, my, mx + px * 0.4, my), fill=(200, 40, 40), width=2)
    sx, sz = gm.info.get("start", {}).get("x"), gm.info.get("start", {}).get("z")
    if sx is not None:
        u0, v0, u1, v1 = box(gm.cell_of(sx, sz))
        d.ellipse((u0 + px * .25, v0 + px * .25, u1 - px * .25, v1 - px * .25), fill=(20, 20, 20))
    if title:
        d.text((6, 6), title, fill=(30, 30, 30))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path
