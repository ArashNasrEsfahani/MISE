"""Offline selftest for step 4 s2-map (sim/honest_map.py). Windows `python`, no WSL/GPU/Ollama.

Run from the repo root:  python sim/_step4_map_selftest.py

Uses the REAL ``honest.HonestAgent`` over a fake grid-world simulator, so every pose / move read goes through the
audited door. Covers:
  1. A* (detour around a blocked edge, no path, start already a goal), cells_near.
  2. save / load round trip, schema + cell-size checks.
  3. exploration: discovers exactly the reachable free cells, records thin walls as blocked edges (both cells free),
     moves only with MoveAhead + rotations, never teleports, audit = pose/actuator only, 0 violations.
  4. heading alignment from an off-axis start yaw; an off-lattice start origin.
  5. invariants: a simulator that over-moves aborts the build (partial map kept); the budget stops the build.
  6. walk_path: a wall the map did not know -> blocked edge + replan -> arrives; unreachable goal -> explicit failure
     after the retry cap; goto_near + face_point.
  7. coverage report (set-based, residual, unmatched) and map-build read only after the build.
  8. PNG render (if PIL is installed).
  9. 2026-10-09 unreachable fixes: soft cells (passable at SOFT_COST, never a wall), hard cells, unblock; the episode
     start node (adopt_here): off-map start reachable only after adoption, a re-tried edge walked for real, a refused
     re-try re-blocked for good, nothing persisted.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import honest                                                         # noqa: E402
import honest_map as hm                                               # noqa: E402
from honest_map import GridMap, edge                                  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ----------------------------------------------------------------------------------------------- fake grid world
class FakeEvent:
    def __init__(self, md):
        self.metadata = md


class GridThor:
    """A fake simulator on a 0.25 m lattice anchored at (ox, oz). ``cells`` = walkable cells; ``walls`` = thin
    walls between two walkable cells. ``over`` = metres to over-move on MoveAhead (invariant test)."""

    def __init__(self, cells, walls=(), start=(0, 0), yaw=90.0, ox=0.0, oz=0.0, over=0.0, cell=0.25):
        self.cells, self.walls = set(cells), {edge(a, b) for a, b in walls}
        self.ox, self.oz, self.over, self.cell = ox, oz, over, cell
        self.calls = []
        self.frame = None
        self.depth_frame = None
        x, z = ox + start[0] * cell, oz + start[1] * cell
        self.event = FakeEvent({"agent": {"position": {"x": x, "y": 0.9, "z": z}, "rotation": {"x": 0, "y": yaw, "z": 0},
                                          "cameraHorizon": 0.0, "isStanding": True},
                                "cameraPosition": {"x": x, "y": 1.5, "z": z}, "objects": [],
                                "lastActionSuccess": True, "errorMessage": "", "actionReturn": None})

    def here(self):
        p = self.event.metadata["agent"]["position"]
        return (round((p["x"] - self.ox) / self.cell), round((p["z"] - self.oz) / self.cell))

    def do(self, action, **kw):
        self.calls.append((action, dict(kw)))
        md = self.event.metadata
        md.update(lastActionSuccess=True, errorMessage="", actionReturn=None)
        a = md["agent"]
        if action in ("RotateRight", "RotateLeft"):
            d = float(kw.get("degrees", 30))
            a["rotation"]["y"] = (a["rotation"]["y"] + (d if action == "RotateRight" else -d)) % 360
        elif action == "MoveAhead":
            yaw = math.radians(a["rotation"]["y"])
            mag = float(kw.get("moveMagnitude", self.cell))
            nx = a["position"]["x"] + math.sin(yaw) * (mag + self.over)
            nz = a["position"]["z"] + math.cos(yaw) * (mag + self.over)
            c0 = self.here()
            c1 = (round((nx - self.ox) / self.cell), round((nz - self.oz) / self.cell))
            if c1 in self.cells and edge(c0, c1) not in self.walls:
                a["position"]["x"], a["position"]["z"] = round(nx, 4), round(nz, 4)
            else:
                md.update(lastActionSuccess=False, errorMessage="blocked by Wall|1|2|3")
        elif action == "GetReachablePositions":
            md["actionReturn"] = [{"x": self.ox + c[0] * self.cell, "y": 0.9, "z": self.oz + c[1] * self.cell}
                                  for c in sorted(self.cells)]
        elif action == "Teleport":
            raise AssertionError("teleport is not allowed")
        md["cameraPosition"] = {"x": a["position"]["x"], "y": 1.5, "z": a["position"]["z"]}
        return self.event


def agent(thor):
    return honest.HonestAgent(thor, detector=object(), mode="honest")


# a 6x5 room with an obstacle block and a thin wall; plus an island not reachable from the start
ROOM = {(x, z) for x in range(6) for z in range(5)} - {(2, 1), (2, 2), (3, 2)}
ISLAND = {(10, 10), (10, 11)}
WALL = [((4, 0), (4, 1)), ((0, 3), (1, 3))]      # thin walls between free cells

# ======================================================================================================== 1
print("1. A* and cells_near")
g = GridMap("T", free={(x, 0) for x in range(4)} | {(x, 1) for x in range(4)})
p = g.astar((0, 0), [(3, 0)])
check("straight path", p == [(0, 0), (1, 0), (2, 0), (3, 0)], str(p))
g.add_blocked((1, 0), (2, 0))
p = g.astar((0, 0), [(3, 0)])
check("detours around a blocked edge", p is not None and len(p) == 6 and ((1, 0), (2, 0)) not in zip(p, p[1:]), str(p))
check("blocked edge is undirected", g.astar((2, 0), [(1, 0)]) is not None and len(g.astar((2, 0), [(1, 0)])) == 4)
check("start already a goal", g.astar((0, 1), [(0, 1), (3, 1)]) == [(0, 1)])
check("no path to a non-free goal", g.astar((0, 0), [(9, 9)]) is None)
g2 = GridMap("T", free={(0, 0), (5, 5)})
check("no path between disconnected cells", g2.astar((0, 0), [(5, 5)]) is None)
near = g.cells_near((0.0, 0.9, 0.0), 0.5, 0.8)
check("cells_near uses horizontal distance, nearest first",
      near and all(0.5 <= math.hypot(*g.world(c)) <= 0.8 for c in near)
      and [round(math.hypot(*g.world(c)), 3) for c in near] == sorted(round(math.hypot(*g.world(c)), 3) for c in near),
      str(near))
check("cells_near accepts (x, z)", g.cells_near((0.0, 0.0), 0.5, 0.8) == near)

# ======================================================================================================== 2
print("\n2. save / load")
tmp = tempfile.mkdtemp()
g.info["status"] = "complete"
path = g.save(os.path.join(tmp, "T.map.json"))
g3 = GridMap.load(path)
check("round trip keeps free, blocked edges, origin, info",
      g3.free == g.free and g3.blocked_edges == g.blocked_edges and g3.origin == g.origin and g3.info == g.info)
import json                                                           # noqa: E402
d = json.load(open(path))
d["schema"] = 99
json.dump(d, open(path, "w"))
try:
    GridMap.load(path)
    check("wrong schema is rejected", False)
except ValueError:
    check("wrong schema is rejected", True)
g.save(path)
try:
    GridMap.load(path, cell=0.5)
    check("wrong cell size is rejected", False)
except ValueError:
    check("wrong cell size is rejected", True)

# ======================================================================================================== 3
print("\n3. exploration on a fake room")
thor = GridThor(ROOM | ISLAND, walls=WALL, start=(0, 0), yaw=90.0, ox=-1.25, oz=0.5)
h = agent(thor)
h.reset_episode("map-build")
gm = hm.explore(h, "T")
want = ROOM                                 # in fake-sim cells; start-relative origin == fake origin here
got = {(c[0], c[1]) for c in gm.free}
check("status complete", gm.info["status"] == "complete", gm.info["status"])
check("discovers exactly the reachable room (not the island)", got == want, f"missing {want - got} extra {got - want}")
inner = [e for e in gm.blocked_edges if e[0] in ROOM and e[1] in ROOM]
check("blocked edges between two room cells are only the real thin walls (no false walls)",
      all(e in {edge(*w) for w in WALL} for e in inner), str(inner))
check("no blocked edge joins two cells the sim lets you cross",
      all(not (a in ROOM and b in ROOM and edge(a, b) not in {edge(*w) for w in WALL}) for a, b in gm.blocked_edges))
acts = {a for a, _ in thor.calls}
check("only MoveAhead + rotations sent (no teleport, no path query)", acts <= {"MoveAhead", "RotateRight", "RotateLeft"},
      str(acts))
check("agent is back at the start cell", thor.here() == (0, 0), str(thor.here()))
s = h.audit.summary()
check("audit: pose + actuator reads only, no violations",
      not h.audit.invalid and set(k for k, v in h.audit.counts.items() if v) <= {"pose", "actuator"}, str(s))
thor2 = GridThor(ROOM, walls=WALL, start=(4, 0), yaw=0.0, ox=-1.25, oz=0.5)
r = hm.walk_path(agent(thor2), gm, gm.astar((4, 0), [(4, 1)]))
check("a thin wall the build never probed is learned while walking, then routed around",
      r["ok"] and thor2.here() == (4, 1) and edge((4, 0), (4, 1)) in gm.blocked_edges, str(r))
check("info records sends and method", gm.info["sends"] == len(thor.calls) and "MoveAhead" in gm.info["method"])

# ======================================================================================================== 4
print("\n4. off-axis start yaw, off-lattice origin")
thor = GridThor(ROOM, walls=WALL, start=(1, 1), yaw=75.0, ox=0.13, oz=-0.07)
h = agent(thor)
gm = hm.explore(h, "T")
shift = {(c[0] + 1, c[1] + 1) for c in gm.free}     # map is start-relative: fake (1,1) is map (0,0)
check("heading aligned to 90 and room discovered", gm.info["status"] == "complete" and shift == ROOM,
      gm.info["status"])
check("origin = start pose", abs(gm.origin[0] - 0.38) < 1e-6 and abs(gm.origin[1] - 0.18) < 1e-6, str(gm.origin))

# ======================================================================================================== 5
print("\n5. invariants and budget")
thor = GridThor(ROOM, start=(0, 0), over=0.1)
gm = hm.explore(agent(thor), "T")
check("over-moving simulator aborts the build", gm.info["status"].startswith("abort:"), gm.info["status"])
check("partial map kept on abort", (0, 0) in gm.free)
thor = GridThor(ROOM, start=(0, 0))
gm = hm.explore(agent(thor), "T", budget=10)
check("budget stops the build", gm.info["status"] == "budget" and gm.info["sends"] >= 10, str(gm.info))

# ======================================================================================================== 6
print("\n6. walking on the agent's own map")
full = {(x, z) for x in range(5) for z in range(3)}
belief = GridMap("T", 0.25, (0.0, 0.0), full)            # the map thinks the row-0 corridor is open
thor = GridThor(full, walls=[((2, 0), (3, 0))], start=(0, 0), yaw=0.0)
h = agent(thor)
r = hm.walk_path(h, belief, belief.astar((0, 0), [(4, 0)]))
check("unknown wall -> blocked edge + replan -> arrives", r["ok"] and r["replans"] == 1 and thor.here() == (4, 0), str(r))
check("the wall is now in the map", edge((2, 0), (3, 0)) in belief.blocked_edges)
check("walking sends only MoveAhead + rotations", {a for a, _ in thor.calls} <= {"MoveAhead", "RotateRight", "RotateLeft"})
cut = [((2, z), (3, z)) for z in range(3)]
belief = GridMap("T", 0.25, (0.0, 0.0), full)
thor = GridThor(full, walls=cut, start=(0, 0), yaw=0.0)
r = hm.walk_path(agent(thor), belief, belief.astar((0, 0), [(4, 0)]))
check("unreachable goal -> explicit failure (no path left / retry cap), never a false success",
      not r["ok"] and r["why"] in ("retries exhausted", "no path on own map") and r["replans"] <= 3
      and thor.here()[0] <= 2, str(r))
belief = GridMap("T", 0.25, (0.0, 0.0), full)
thor = GridThor(full, walls=cut, start=(0, 0), yaw=0.0)
r = hm.walk_path(agent(thor), belief, belief.astar((0, 0), [(4, 0)]), retries=1)
check("retry cap honoured (retries=1)", not r["ok"] and r["why"] == "retries exhausted" and r["replans"] == 1, str(r))
belief = GridMap("T", 0.25, (0.0, 0.0), full)
thor = GridThor(full, start=(0, 0), yaw=0.0)
h = agent(thor)
target = (1.0, 0.9, 0.5)                                   # world xyz
r = hm.goto_near(h, belief, target, 0.4, 0.6)
x, z = belief.world(r["cell"])
check("goto_near stops 0.4..0.6 m from the target", r["ok"] and 0.4 <= math.hypot(x - 1.0, z - 0.5) <= 0.6, str(r))
hm.face_point(h, target)
yaw = thor.event.metadata["agent"]["rotation"]["y"]
want_yaw = math.degrees(math.atan2(1.0 - x, 0.5 - z)) % 360
check("face_point turns to the target", abs(((yaw - want_yaw) + 180) % 360 - 180) < 0.5, f"{yaw} vs {want_yaw}")
r = hm.walk_path(h, belief, None, goals=[(0, 0)])
check("walk_path re-aligns after face_point and plans when no path is given", r["ok"] and thor.here() == (0, 0), str(r))
check("no goal -> failure", not hm.walk_path(h, belief, None)["ok"])

# ======================================================================================================== 7
print("\n7. coverage report")
thor = GridThor(ROOM | ISLAND, walls=WALL, start=(0, 0), ox=-1.25, oz=0.5)
h = agent(thor)
gm = hm.explore(h, "T")
before = h.audit.counts.get("map-build", 0)
ref = h.reachable_positions()
cov = hm.coverage(gm, ref)
check("map-build read happens once, after the build", before == 0 and h.audit.counts["map-build"] == 1
      and not h.audit.invalid)
check("coverage is set-based: room / (room + island)", cov["matched"] == len(ROOM)
      and cov["reference_cells"] == len(ROOM | ISLAND)
      and abs(cov["coverage"] - len(ROOM) / len(ROOM | ISLAND)) < 1e-9, str(cov)[:200])
check("missed reference cells listed, no unmatched discovered", sorted(cov["missed_reference"]) == sorted(ISLAND)
      and cov["unmatched_discovered"] == 0)
check("reference lattice residual ~0", cov["ref_lattice_residual_max"] < 1e-6)
off = [{"x": p["x"] + 0.1, "z": p["z"]} for p in ref]
check("an offset reference grid shows up in the residual", hm.coverage(gm, off)["ref_lattice_residual_max"] > 0.09)

# ======================================================================================================== 8
print("\n8. PNG")
try:
    import PIL                                                    # noqa: F401
    out = hm.render_png(gm, os.path.join(tmp, "T.png"), ref, title="T")
    check("png written", os.path.getsize(out) > 100)
except ImportError:
    print("  (PIL not installed: skipped)")

# ======================================================================================================== 9
print("\n9. soft / hard temporary cells and the episode start node (2026-10-09)")
g9 = GridMap("T", 0.25, (0.0, 0.0), {(x, z) for x in range(7) for z in range(3)})
g9.block_cells("d", [(3, 1)], keep=[(0, 1)], soft=[(2, 1), (4, 1)])
p9 = g9.astar((0, 1), [(6, 1)])
check("soft cells are avoided when a detour of <= SOFT_COST steps exists; hard never entered",
      p9 is not None and not any(c in p9 for c in [(2, 1), (3, 1), (4, 1)]), str(p9))
g9b = GridMap("T", 0.25, (0.0, 0.0), {(x, 0) for x in range(5)})
g9b.block_cells("d", [], keep=[(0, 0)], soft=[(1, 0), (2, 0)])
check("soft cells on the only way stay passable (an estimate never walls the agent in)",
      g9b.astar((0, 0), [(4, 0)]) == [(x, 0) for x in range(5)])
check("soft / hard queries and unblock", g9b.is_soft((1, 0)) and not g9b.is_temp_blocked((1, 0))
      and g9b.unblock_cells("d") == {(1, 0), (2, 0)} and not g9b.soft_cells())
check("to_dict never carries temporary cells", "temp_soft" not in g9.to_dict() and "temp_blocked" not in g9.to_dict())
# episode start node: the agent stands on (2,0); the map has only z >= 1 and recorded (2,0)-(2,1) blocked
cells9 = {(x, z) for x in range(4) for z in range(1, 4)}
thor9 = GridThor(cells9 | {(2, 0)}, start=(2, 0), yaw=0.0)
h9 = agent(thor9)
g9c = GridMap("T", 0.25, (0.0, 0.0), cells9, blocked_edges=[((2, 0), (2, 1))])
check("off-map start with its only edge recorded blocked: no path", g9c.astar((2, 0), [(0, 3)]) is None)
check("adopt_here: the edge becomes re-triable once; a path exists", g9c.adopt_here((2, 0)) == [edge((2, 0), (2, 1))]
      and g9c.astar((2, 0), [(0, 3)]) is not None and g9c.is_free((2, 0)) and (2, 0) not in g9c.free)
r9 = hm.walk_path(h9, g9c, g9c.astar((2, 0), [(0, 3)]), goals=[(0, 3)])
check("the re-tried edge is walked for real (the world allows it) -> arrives", r9["ok"] and thor9.here() == (0, 3),
      str(r9))
check("adoption is episode-only: free cells / file unchanged", (2, 0) not in g9c.free
      and [2, 0] not in g9c.to_dict()["free"] and (g9c.drop_here() or True) and not g9c.here_free)
thor9b = GridThor(cells9 | {(2, 0)}, walls=[((2, 0), (2, 1))], start=(2, 0), yaw=0.0)
h9b = agent(thor9b)
g9d = GridMap("T", 0.25, (0.0, 0.0), cells9, blocked_edges=[((2, 0), (2, 1))])
g9d.adopt_here((2, 0))
r9b = hm.walk_path(h9b, g9d, g9d.astar((2, 0), [(0, 3)]), goals=[(0, 3)])
check("a refused re-try re-blocks the edge for good: explicit failure, not repeated",
      not r9b["ok"] and edge((2, 0), (2, 1)) in g9d.blocked_edges and not g9d.retry_edges
      and sum(1 for a, _ in thor9b.calls if a == "MoveAhead") == 1, str(r9b))

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("STEP4 MAP SELFTEST OK")
