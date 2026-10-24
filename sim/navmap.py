"""Discovered occupancy grid — the agent's navigation map, built from REAL movement (sibling to memory.py).

Per the Phase-4 author decision, the nav map is **discovered**, not handed to us by the simulator: a cell the agent
actually reaches is ``free``; a move/teleport that fails or collides is ``blocked``; everything unvisited stays
``unknown``. The sim's :meth:`reachable_positions` is kept ONLY as an **oracle baseline** to score the discovered map
against (see :meth:`NavMap.compare`) — never as the source of truth.

World coordinates (AI2-THOR metres) are quantised onto a square grid of side ``cell_size`` anchored at ``origin``;
cells are stored as integer ``(gx, gz)`` indices in the ``nav_cells`` table via :mod:`memory` (every write auditable).
This module persists nothing itself — it owns the world↔cell mapping and the discovered-vs-oracle comparison, and
delegates all storage to the injected :class:`memory.Memory`.

Pure stdlib (no numpy/ai2thor at import). The live ``agent`` passed to :meth:`oracle`/:meth:`compare`/:meth:`to_json`
only needs a ``.reachable_positions()`` returning ``[{'x','y','z'}, ...]`` — the thor_actions wrapper supplies it.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- world <-> cell mapping
def world_to_cell(x, z, cell_size: float = 0.25, origin=(0.0, 0.0)):
    """Quantise world coords ``(x, z)`` (metres) to integer grid indices ``(gx, gz)`` (nearest cell)."""
    ox, oz = origin
    return (int(round((x - ox) / cell_size)), int(round((z - oz) / cell_size)))


def cell_to_world(gx, gz, cell_size: float = 0.25, origin=(0.0, 0.0)):
    """Inverse of :func:`world_to_cell`: the world-space ``(x, z)`` CENTRE of cell ``(gx, gz)``."""
    ox, oz = origin
    return (ox + gx * cell_size, oz + gz * cell_size)


def map_project(x, z, cam, w, h):
    """Project a world point ``(x, z)`` (metres) to a pixel ``(px, py)`` in AI2-THOR's top-down orthographic frame.

    ``cam`` is the dict from ``GetMapViewCameraProperties`` (``{'position':{x,y,z}, 'rotation':{x,y,z},
    'orthographicSize': S}``): a camera hovering at ``(cx, cz)`` looking straight down, framing ``±S`` world-metres
    vertically over ``h`` pixels (square pixels -> the same scale horizontally). The camera's ``rotation.y`` yaw is
    folded in so the projection stays correct if the map view is rotated. Image convention: world +x -> right,
    world +z (north) -> up.
    """
    import math
    pos = cam.get("position", {}) or {}
    rot = cam.get("rotation", {}) or {}
    S = float(cam.get("orthographicSize") or 1.0) or 1.0
    cx, cz = float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    ry = math.radians(float(rot.get("y", 0.0)))
    dx, dz = x - cx, z - cz
    cos_r, sin_r = math.cos(ry), math.sin(ry)
    ix = dx * cos_r + dz * sin_r            # component along the image's right axis
    iz = -dx * sin_r + dz * cos_r           # component along the image's up axis (north)
    k = h / (2.0 * S)                       # pixels per world-metre
    return (w * 0.5 + ix * k, h * 0.5 - iz * k)


def _as_xz(p):
    """Pull an ``(x, z)`` pair out of a pose/point in any of the shapes the caller might hand us.

    Accepts a pose dict (``{'position': {'x','y','z'}, ...}``), a flat ``{'x',...,'z'}`` dict, a 2-seq ``(x, z)``,
    or a 3-seq ``(x, y, z)``. Returns ``(x, z)`` or ``None`` if it can't be read.
    """
    if p is None:
        return None
    if isinstance(p, dict):
        if isinstance(p.get("position"), dict):
            p = p["position"]
        x, z = p.get("x"), p.get("z")
        return (x, z) if x is not None and z is not None else None
    try:
        if len(p) == 2:
            return (p[0], p[1])
        if len(p) >= 3:
            return (p[0], p[2])
    except TypeError:
        pass
    return None


# --------------------------------------------------------------------------- discovered occupancy map
class NavMap:
    """Discovered occupancy grid for one scene, persisted through :class:`memory.Memory`'s ``nav_cells`` API.

    Owns the world↔cell mapping (``cell_size`` + ``origin``); cell occupancy is accumulated from real navigation via
    :meth:`mark_free`/:meth:`mark_blocked`/:meth:`record_move`, and scored against the sim oracle via :meth:`compare`.
    """

    def __init__(self, memory, scene: str, cell_size: float = 0.25, origin=(0.0, 0.0)):
        self.memory = memory
        self.scene = scene
        self.cell_size = float(cell_size)
        self.origin = (float(origin[0]), float(origin[1]))

    # ----------------------------------------------------------------- writes (occupancy from real movement)
    def mark_free(self, x, z, why: str = "reached by navigation") -> None:
        """Mark the cell containing world point ``(x, z)`` as ``free`` (the agent stood there)."""
        gx, gz = world_to_cell(x, z, self.cell_size, self.origin)
        self.memory.mark_cell(self.scene, gx, gz, "free", why=why)

    def mark_blocked(self, x, z, why: str = "move/teleport failed or collided") -> None:
        """Mark the cell containing world point ``(x, z)`` as ``blocked`` (a move there failed/collided)."""
        gx, gz = world_to_cell(x, z, self.cell_size, self.origin)
        self.memory.mark_cell(self.scene, gx, gz, "blocked", why=why)

    def record_move(self, new_pose, success, attempted_xz=None) -> None:
        """Fold one navigation attempt into the map.

        On ``success`` the agent is now at ``new_pose`` (a pose dict / point) → that cell is ``free``. On failure, if
        the intended destination ``attempted_xz`` is given, that cell is ``blocked`` (the agent couldn't get there).
        """
        if success:
            xz = _as_xz(new_pose)
            if xz is not None:
                self.mark_free(xz[0], xz[1])
        elif attempted_xz is not None:
            xz = _as_xz(attempted_xz)
            if xz is not None:
                self.mark_blocked(xz[0], xz[1])

    # ----------------------------------------------------------------- reads
    def cells(self) -> list:
        """Discovered cells as ``[{'x','z','state'}, ...]`` with world-space cell CENTRES (for drawing/analysis)."""
        out = []
        for r in self.memory.get_nav_cells(self.scene):
            x, z = cell_to_world(r["gx"], r["gz"], self.cell_size, self.origin)
            out.append({"x": x, "z": z, "state": r["state"]})
        return out

    def oracle(self, agent) -> list:
        """The sim's ground-truth reachable points as ``[{'x','z'}, ...]`` — the baseline, NOT the discovered map."""
        return [{"x": p["x"], "z": p["z"]} for p in (agent.reachable_positions() or [])]

    def compare(self, agent) -> dict:
        """Score the discovered map against the sim oracle (oracle points snapped onto the SAME grid first).

        Returns ``{discovered_free, oracle_total, agreement, false_blocked, undiscovered}`` where, over the set of
        distinct oracle grid cells: ``agreement`` is the fraction we independently discovered as ``free``,
        ``false_blocked`` is how many we wrongly marked ``blocked``, and ``undiscovered`` is how many we never reached
        (no ``free``/``blocked`` mark). ``discovered_free`` is the total ``free`` cells in the map.
        """
        rows = self.memory.get_nav_cells(self.scene)
        free_set = {(r["gx"], r["gz"]) for r in rows if r["state"] == "free"}
        blocked_set = {(r["gx"], r["gz"]) for r in rows if r["state"] == "blocked"}
        oracle_cells = {world_to_cell(p["x"], p["z"], self.cell_size, self.origin) for p in self.oracle(agent)}
        oracle_total = len(oracle_cells)
        agree = sum(1 for c in oracle_cells if c in free_set)
        false_blocked = sum(1 for c in oracle_cells if c in blocked_set)
        return {
            "discovered_free": len(free_set),
            "oracle_total": oracle_total,
            "agreement": (agree / oracle_total) if oracle_total else 0.0,
            "false_blocked": false_blocked,
            "undiscovered": oracle_total - agree - false_blocked,
        }

    def to_json(self, agent=None) -> dict:
        """Serialisable snapshot for the GUI: grid params, world bounds, discovered cells, and (if ``agent``) oracle.

        ``cells`` use the compact key ``s`` for state (``'free'``/``'blocked'``/``'unknown'``). ``oracle`` is included
        only when a live ``agent`` is supplied. ``bounds`` spans every plotted point (cells, plus oracle if present).
        """
        cells = self.cells()
        pts = [(c["x"], c["z"]) for c in cells]
        oracle = None
        if agent is not None:
            oracle = self.oracle(agent)
            pts += [(o["x"], o["z"]) for o in oracle]
        if pts:
            xs = [p[0] for p in pts]
            zs = [p[1] for p in pts]
            bounds = {"minx": min(xs), "maxx": max(xs), "minz": min(zs), "maxz": max(zs)}
        else:
            bounds = {"minx": 0.0, "maxx": 0.0, "minz": 0.0, "maxz": 0.0}
        out = {
            "scene": self.scene,
            "cell_size": self.cell_size,
            "origin": [self.origin[0], self.origin[1]],
            "bounds": bounds,
            "cells": [{"x": c["x"], "z": c["z"], "s": c["state"]} for c in cells],
        }
        if oracle is not None:
            out["oracle"] = oracle
        return out


if __name__ == "__main__":
    # Self-test: pure stdlib + a TEMP Memory (no live agent / ai2thor). Exercises the world<->cell mapping, the three
    # write paths (mark_free, mark_blocked, record_move success+failure), and the cells()/to_json() shapes.
    import os
    import tempfile

    from memory import Memory

    # mapping round-trips
    assert world_to_cell(0.0, 0.0) == (0, 0)
    assert world_to_cell(0.25, -0.5) == (1, -2)
    assert cell_to_world(1, -2) == (0.25, -0.5)
    assert world_to_cell(*cell_to_world(3, -4)) == (3, -4)
    assert _as_xz({"position": {"x": 0.5, "y": 0.9, "z": 1.0}}) == (0.5, 1.0)
    assert _as_xz((0.75, 0.0)) == (0.75, 0.0)
    assert _as_xz((0.75, 0.9, 0.0)) == (0.75, 0.0)
    assert _as_xz(None) is None

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "navmap_selftest.db")
    mem = Memory(path=db)
    nm = NavMap(mem, "FloorPlan1", cell_size=0.25, origin=(0.0, 0.0))

    nm.mark_free(0.0, 0.0)
    nm.mark_free(0.25, 0.0)
    nm.record_move({"position": {"x": 0.5, "y": 0.9, "z": 0.0}}, success=True)        # -> free (2,0)
    nm.record_move(None, success=False, attempted_xz=(0.75, 0.0))                     # -> blocked (3,0)
    nm.mark_blocked(1.0, 0.0)                                                         # -> blocked (4,0)
    nm.record_move({"x": 99.0, "z": 99.0}, success=False)                            # no attempted_xz -> no-op

    cells = nm.cells()
    assert len(cells) == 5, f"expected 5 distinct cells, got {len(cells)}: {cells!r}"
    assert all(set(c) == {"x", "z", "state"} for c in cells), cells
    states = [c["state"] for c in cells]
    assert states.count("free") == 3 and states.count("blocked") == 2, states

    j = nm.to_json()
    assert j["scene"] == "FloorPlan1" and j["cell_size"] == 0.25 and j["origin"] == [0.0, 0.0]
    assert set(j["bounds"]) == {"minx", "maxx", "minz", "maxz"}
    assert j["bounds"]["minx"] == 0.0 and j["bounds"]["maxx"] == 1.0
    assert all(set(c) == {"x", "z", "s"} for c in j["cells"]) and len(j["cells"]) == 5
    assert "oracle" not in j, "oracle must be absent when no agent is given"
    print("cells:", len(cells), "| free/blocked:", states.count("free"), "/", states.count("blocked"),
          "| bounds:", j["bounds"])
    print("NAVMAP SELFTEST OK")

    mem.close()
    for p in (db, os.path.splitext(db)[0] + "_audit.log"):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass


# ===================================================================== PUBLIC API (discovered nav map) ==============
# Free functions (world <-> integer grid-cell mapping; cell_size in metres, origin = world anchor of cell (0,0)):
#   world_to_cell(x, z, cell_size=0.25, origin=(0.0, 0.0)) -> (gx, gz)    nearest cell: round((x-ox)/cell_size)
#   cell_to_world(gx, gz, cell_size=0.25, origin=(0.0, 0.0)) -> (x, z)    world CENTRE of that cell
#
# class NavMap(memory, scene, cell_size=0.25, origin=(0.0, 0.0)):
#   mark_free(x, z, why=...) -> None        world point (x,z) -> mark its cell 'free'   (agent stood there)
#   mark_blocked(x, z, why=...) -> None     world point (x,z) -> mark its cell 'blocked' (move there failed)
#   record_move(new_pose, success, attempted_xz=None) -> None
#       success -> mark_free(new_pose); failure -> mark_blocked(attempted_xz) if given. Accepts pose dicts/points.
#   cells() -> [{'x','z','state'}, ...]     discovered cells, world-space CENTRES (state in free|blocked|unknown).
#   oracle(agent) -> [{'x','z'}, ...]       sim ground-truth reachable points (BASELINE only) from .reachable_positions().
#   compare(agent) -> {discovered_free, oracle_total, agreement, false_blocked, undiscovered}
#       discovered-vs-oracle score; oracle points snapped to this grid first. agreement = free-fraction of oracle cells.
#   to_json(agent=None) -> {scene, cell_size, origin, bounds:{minx,maxx,minz,maxz}, cells:[{x,z,s}], oracle?:[{x,z}]}
#       GUI snapshot; 's' is the cell state; 'oracle' present only when a live agent is supplied.
