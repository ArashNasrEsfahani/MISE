"""Offline selftest for step 4 / s3-memory (sim/objmem.py + the sightings/places_seen tables in sim/memory.py).

No WSL, GPU or Ollama: synthetic ``honest.Obs`` views rendered from point objects with a pinhole depth image.

  1. geometry: project <-> backproject round trip from the agent's own pose
  2. first sighting -> trusted track, evidence row, places_seen with the containing receptacle
  3. unchanged re-sighting refreshes (no row); a state change supersedes; a duplicate observation is ignored
  4. contradiction needs 2 clear views from distinct viewpoints; a match resets the count
  5. occlusion / out of range / out of frame never count as a miss
  6. one-to-one association keeps nearby same-type objects apart; ambiguous matches are not guessed
  7. own actions: failed pickup/put change nothing; held / placed / unbound held; placed is not contradicted
  8. lookup order trusted > placed > held; contradicted left out; summary lines
  9. persistence: rows + statuses + reasons in sqlite, audit lines, resume, new_episode
 10. purity: objmem imports nothing from the simulator

Run from the repo root:  python sim/_step4_memory_selftest.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import objmem  # noqa: E402
from honest import Obs  # noqa: E402
from memory import Memory  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


W = H = 300
WALL = 6.0


def pose(x=0.0, z=0.0, yaw=0.0, horizon=30.0):
    return {"position": {"x": x, "y": 0.9, "z": z}, "rotation": {"x": 0, "y": yaw, "z": 0}, "horizon": horizon,
            "standing": True, "camera": {"x": x, "y": 1.575, "z": z}}


def render(p, objs, step, occlude=(), holding=None, hide=(), extra=()):
    """objs: [(type, xyz, state)]. Visible ones become dets (unless in ``hide``); ``occlude`` = types to occlude
    (depth in front of them, no det). ``extra`` = raw det dicts appended (receptacles etc.)."""
    depth = np.full((H, W), WALL, dtype="float32")
    dets = []
    for typ, xyz, state in objs:
        pr = objmem.project(xyz, p, (W, H))
        if pr is None:
            continue
        px, py, zc = pr
        if not (0 <= px < W and 0 <= py < H):
            continue
        ix, iy = int(round(px)), int(round(py))
        y0, y1, x0, x1 = max(0, iy - 6), min(H, iy + 7), max(0, ix - 6), min(W, ix + 7)
        if typ in occlude:
            depth[y0:y1, x0:x1] = zc - 0.6
            continue
        depth[y0:y1, x0:x1] = zc
        if typ in hide:
            continue
        dets.append({"type": typ, "state": state, "states": {}, "conf": 0.9,
                     "bbox_xyxy": (ix - 6, iy - 6, ix + 6, iy + 6), "point": (ix, iy), "point_ok": True,
                     "source": "primary"})
    dets.extend(dict(e) for e in extra)
    return Obs(step=step, rgb=None, depth=depth, pose=p, dets=dets, holding=holding)


def near(a, b, tol=0.03):
    return a is not None and b is not None and all(abs(a[i] - b[i]) <= tol for i in range(3))


tmp = tempfile.mkdtemp()
db = os.path.join(tmp, "s3.db")
store = Memory(path=db)

# ================================================================================================== 1
print("1. geometry")
p0 = pose(0, 0, 0, 30)
for xyz in [(0.3, 0.95, 1.2), (-0.5, 0.2, 2.0), (0.0, 1.5, 0.8)]:
    pr = objmem.project(xyz, p0, (W, H))
    back = objmem.backproject(pr[0], pr[1], pr[2], p0, (W, H))
    check(f"round trip {xyz}", near(back, xyz, 1e-6), f"{back}")
check("behind the camera -> None", objmem.project((0, 1.5, -1.0), p0, (W, H)) is None)
check("no depth -> None", objmem.backproject(10, 10, float("nan"), p0, (W, H)) is None)
p90 = pose(0, 0, 90, 0)
pr = objmem.project((2.0, 1.575, 0.0), p90, (W, H))
check("yaw 90 looks along +x (centre pixel)", abs(pr[0] - (W / 2 - .5)) < 1e-6 and abs(pr[1] - (H / 2 - .5)) < 1e-6
      and abs(pr[2] - 2.0) < 1e-9, f"{pr}")

# ================================================================================================== 2
print("\n2. first sighting")
om = objmem.ObjectMemory("FloorPlan1", store=store, episode="e1")
APPLE = (0.2, 0.95, 1.3)
counter = {"type": "CounterTop", "state": "?", "states": {}, "conf": 0.95, "bbox_xyxy": (0, 100, 299, 299),
           "point": (150, 250), "point_ok": False, "source": "primary"}
o = render(p0, [("Apple", APPLE, "whole")], step=1, extra=[counter])
r = om.update(o)
check("one new track", len(r["new"]) == 1, f"{r}")
tid = r["new"][0]
t = om.tracks[tid]
check("trusted at the true xyz", t.status == "trusted" and near(t.xyz, APPLE, 0.02), f"{t}")
check("det carries its track id", o.dets[0]["track"] == tid)
check("CounterTop without valid point -> unplaced evidence row", r["unplaced"] == 1)
rows = store.load_sightings("FloorPlan1", "e1")
ar = [x for x in rows if x["track"] == tid]
check("evidence row has type/state/conf/box/point/pose/xyz", len(ar) == 1 and ar[0]["state"] == "whole"
      and ar[0]["box"] and ar[0]["point"] and ar[0]["pose"]["yaw"] == 0.0 and ar[0]["xyz_ok"] == 1
      and ar[0]["obs_id"] == "e1:1", f"{ar}")
pl = store.places_seen("FloorPlan1", "Apple")
check("places_seen: Apple on CounterTop", len(pl) == 1 and pl[0]["place_type"] == "CounterTop", f"{pl}")

# ================================================================================================== 3
print("\n3. refresh / supersede / duplicate")
n_rows = len(store.load_sightings("FloorPlan1"))
r = om.update(render(pose(0.05, 0, 5, 30), [("Apple", APPLE, "whole")], step=2, extra=[counter]))
check("unchanged re-sighting refreshes", r["refreshed"] == [tid] and not r["new"], f"{r}")
check("... and writes no sighting row (only the unplaced counter)", len(store.load_sightings("FloorPlan1")) == n_rows + 1)
check("places_seen not re-logged for the same (track, place)", len(store.places_seen("FloorPlan1", "Apple")) == 1)
r = om.update(render(p0, [("Apple", APPLE, "sliced")], step=3))
check("state change supersedes", r["superseded"] == [tid] and om.tracks[tid].state == "sliced", f"{r}")
rows = [x for x in store.load_sightings("FloorPlan1", "e1") if x["track"] == tid]
check("old row superseded with reason, new row trusted", [x["status"] for x in rows] == ["superseded", "trusted"]
      and "state whole -> sliced" in rows[0]["why"], f"{[(x['status'], x['why']) for x in rows]}")
r = om.update(render(p0, [("Apple", APPLE, "sliced")], step=3))
check("same observation id twice is ignored", r["duplicate"] and not r["refreshed"])

# ================================================================================================== 4
print("\n4. contradiction needs 2 distinct clear views")
r = om.update(render(p0, [("Apple", APPLE, "sliced")], step=4, hide=("Apple",)))
check("miss 1 counted, still trusted", r["missed"] == [tid] and om.tracks[tid].status == "trusted", f"{r}")
r = om.update(render(p0, [("Apple", APPLE, "sliced")], step=5, hide=("Apple",)))
check("same viewpoint again does not count", not r["missed"] and om.tracks[tid].status == "trusted", f"{r}")
r = om.update(render(p0, [("Apple", APPLE, "sliced")], step=6))
check("a match resets the miss count", r["refreshed"] == [tid] and om.tracks[tid].misses == [])
om.update(render(p0, [("Apple", APPLE, "sliced")], step=7, hide=("Apple",)))
r = om.update(render(pose(0.3, -0.1, 10, 30), [("Apple", APPLE, "sliced")], step=8, hide=("Apple",)))
check("2nd miss from a distinct viewpoint contradicts", r["contradicted"] == [tid]
      and om.tracks[tid].status == "contradicted", f"{r}")
cur = [x for x in store.load_sightings("FloorPlan1", "e1") if x["track"] == tid][-1]
check("row re-statused contradicted with the two obs ids", cur["status"] == "contradicted"
      and "e1:7" in cur["why"] and "e1:8" in cur["why"], cur["why"])
check("contradicted not offered by lookup", om.lookup("Apple") == [])
r = om.update(render(p0, [("Apple", APPLE, "sliced")], step=9))
check("seen again at the spot -> trusted again (location belief restored)", r["superseded"] == [tid]
      and om.tracks[tid].status == "trusted", f"{r}")

# ================================================================================================== 5
print("\n5. views that say nothing")
EGG = (-0.3, 0.95, 1.5)
r = om.update(render(p0, [("Egg", EGG, "?")], step=10))
egg = r["new"][0]
for k, pz in enumerate([(0, 0, 0), (0.3, 0, 12), (-0.3, 0.1, 350)]):
    r = om.update(render(pose(pz[0], pz[1], pz[2], 30), [("Egg", EGG, "?")], step=11 + k, occlude=("Egg",)))
    check(f"occluded view {k + 1}: no miss", egg not in r["missed"], f"{r}")
check("occlusion keeps trust", om.tracks[egg].status == "trusted" and om.tracks[egg].misses == [])
r = om.update(render(pose(0, -2.5, 0, 10), [("Egg", EGG, "?")], step=20, hide=("Egg",)))
check("beyond 2.5 m: no miss", egg not in r["missed"], f"{r}")
r = om.update(render(pose(0, 0, 180, 30), [], step=21))
check("behind the camera: no miss", egg not in r["missed"])
r = om.update(render(pose(0, 0, 60, 30), [], step=22))
check("outside the frame: no miss", egg not in r["missed"])
far_det = {"type": "Egg", "state": "?", "states": {}, "conf": 0.3, "bbox_xyxy": (0, 0, 5, 5), "point": None,
           "point_ok": False, "source": "primary"}
pr = objmem.project(EGG, pose(0.2, 0, 3, 30), (W, H))
near_det = dict(far_det, bbox_xyxy=(int(pr[0]) + 4, int(pr[1]) - 3, int(pr[0]) + 20, int(pr[1]) + 3))
r = om.update(render(pose(0.2, 0, 3, 30), [("Egg", EGG, "?")], step=23, hide=("Egg",), extra=[near_det]))
check("an ungrounded same-type box near the spot blocks the miss", egg not in r["missed"], f"{r}")

# ================================================================================================== 6
print("\n6. association")
om2 = objmem.ObjectMemory("FloorPlan1", store=None, episode="e2")
M1, M2 = (0.0, 0.95, 1.2), (0.3, 0.95, 1.2)
r = om2.update(render(p0, [("Mug", M1, "clean"), ("Mug", M2, "clean")], step=1))
check("two mugs 0.3 m apart -> two tracks", len(r["new"]) == 2, f"{r}")
r = om2.update(render(pose(0.05, 0, 2, 30), [("Mug", M1, "dirty"), ("Mug", M2, "clean")], step=2))
check("re-seen one-to-one (no merge, no new track)", len(om2.tracks) == 2 and not r["new"], f"{r}")
check("state attached to the right mug", sorted(t.state for t in om2.tracks.values()) == ["clean", "dirty"])
mid = (0.15, 0.95, 1.2)
r = om2.update(render(p0, [("Mug", mid, "clean")], step=3))
check("detection equidistant from two tracks is not guessed", not r["new"] and r["unplaced"] == 1
      and not r["refreshed"] and not r["superseded"], f"{r}")
r = om2.update(render(p0, [("Mug", (0.9, 0.95, 1.5), "clean")], step=4))
check("same type > 0.4 m away -> new track", len(r["new"]) == 1)
o = render(p0, [("Bread", (0, 0.95, 1.0), "?")], step=5)
o.dets[0]["point_ok"] = False
r = om2.update(o)
check("no valid depth point -> evidence only, no track", r["unplaced"] == 1 and not r["new"])

# ================================================================================================== 7
print("\n7. the agent's own actions")
om3 = objmem.ObjectMemory("FloorPlan1", store=store, episode="e3")
POT = (0.0, 0.95, 1.4)
o = render(p0, [("Potato", POT, "raw")], step=1)
om3.update(o)
pt = o.dets[0]["track"]
check("failed pickup changes nothing", om3.on_pickup(False, pt, "Potato", step=2) is None
      and om3.tracks[pt].status == "trusted")
check("successful pickup -> held", om3.on_pickup(True, pt, "Potato", step=3) == pt and om3.tracks[pt].status == "held"
      and om3.held == pt)
lk = om3.lookup("Potato")
check("held gives no navigation xyz", lk and lk[0]["status"] == "held" and lk[0]["xyz"] is None, f"{lk}")
inhand = (0.0, 1.4, 0.5)
r = om3.update(render(p0, [("Potato", inhand, "raw")], step=4, holding="Potato"))
check("the object in hand is not a new sighting", not r["new"] and r["skipped"], f"{r}")
r = om3.update(render(p0, [], step=5, holding="Potato"))
check("held is never contradicted", om3.tracks[pt].status == "held")
check("failed put changes nothing", om3.on_put(False, "Pan", (0.5, 0.95, 1.5), step=6) is None
      and om3.tracks[pt].status == "held")
check("successful put -> placed at the receptacle anchor, state ?",
      om3.on_put(True, "Pan", (0.5, 0.95, 1.5), step=7) == pt and om3.tracks[pt].status == "placed"
      and om3.tracks[pt].anchor == (0.5, 0.95, 1.5) and om3.tracks[pt].state == "?" and om3.held is None)
for k, pz in enumerate([(0, 0, 0), (0.4, 0, 15)]):
    om3.update(render(pose(*pz[:2], pz[2], 30), [], step=8 + k))
check("placed is not contradicted by misses (may be inside)", om3.tracks[pt].status == "placed")
r = om3.update(render(p0, [("Potato", (0.65, 0.97, 1.45), "raw")], step=11))
check("seen again within 1 m of the anchor -> trusted", r["superseded"] == [pt] and om3.tracks[pt].status == "trusted"
      and om3.tracks[pt].state == "raw", f"{r}")
o = render(p0, [("Lettuce", (-0.4, 0.95, 1.3), "?")], step=12)
om3.update(o)
lt = o.dets[0]["track"]
nt = om3.on_pickup(True, lt, "Tomato", step=13)
check("pickup bound to a different type -> unbound held track, the other track untouched",
      nt != lt and om3.tracks[nt].type == "Tomato" and om3.tracks[lt].status == "trusted")
om3.on_put(True, "Plate", (0.1, 0.95, 1.1), step=14)
nb = om3.on_pickup(True, None, "Cup", step=15)
check("pickup with no track -> new held track of the aimed type", om3.tracks[nb].type == "Cup"
      and om3.tracks[nb].status == "held")
check("drop -> placed on the floor near the agent", om3.on_drop(True, p0, step=16) == nb
      and om3.tracks[nb].place == "Floor" and om3.tracks[nb].anchor == (0.0, 0.9, 0.0))
check("pick the lettuce up (stays held for the resume check)", om3.on_pickup(True, lt, "Lettuce", step=17) == lt)

# ================================================================================================== 8
print("\n8. lookup order + summary")
om4 = objmem.ObjectMemory("FloorPlan1", store=None, episode="e4")
o = render(p0, [("Mug", (0, 0.95, 1.2), "clean"), ("Mug", (0.6, 0.95, 1.2), "clean")], step=1)
om4.update(o)
a, b = o.dets[0]["track"], o.dets[1]["track"]
om4.on_pickup(True, b, "Mug", step=2)
om4.on_put(True, "Sink", (1.0, 0.9, 1.0), step=3)
c = om4.on_pickup(True, None, "Mug", step=4)
check("trusted > placed > held", [x["status"] for x in om4.lookup("Mug")] == ["trusted", "placed", "held"]
      and [x["track"] for x in om4.lookup("Mug")] == [a, b, c])
check("placed lookup returns the anchor", om4.lookup("Mug")[1]["xyz"] == (1.0, 0.9, 1.0))
s = om4.summary()
check("summary lines for each usable belief", len(s) == 3 and any("put on/in Sink" in x for x in s)
      and any("in hand" in x for x in s), f"{s}")

# ================================================================================================== 9
print("\n9. persistence + audit")
statuses = {x["status"] for x in store.load_sightings("FloorPlan1")}
check("all five statuses appear in the store", statuses >= {"trusted", "contradicted", "superseded", "held",
                                                             "placed"}, f"{statuses}")
check("every sighting row carries a reason", all(x["why"] for x in store.load_sightings("FloorPlan1")))
res = objmem.ObjectMemory.resume(store, "FloorPlan1", "e3")
check("resume rebuilds the episode's beliefs", {k: v.status for k, v in res.tracks.items()}
      == {k: v.status for k, v in om3.tracks.items()}, f"{ {k: v.status for k, v in res.tracks.items()} }")
check("resume keeps xyz / anchor", near(res.tracks[pt].xyz, om3.tracks[pt].xyz)
      and res.tracks[nb].anchor == om3.tracks[nb].anchor)
check("resume continues track numbering", res._new_tid() not in om3.tracks)
res1 = objmem.ObjectMemory.resume(store, "FloorPlan1", "e1")
check("resume of e1 matches RAM (the apple was later missed in the egg views -> contradicted)",
      {k: v.status for k, v in res1.tracks.items()} == {k: v.status for k, v in om.tracks.items()}
      and res1.tracks[tid].status == "contradicted" and res1.tracks[egg].status == "trusted",
      f"{ {k: v.status for k, v in res1.tracks.items()} }")
audit = open(store.audit_path, encoding="utf-8").read()
check("audit has SIGHT add / SIGHT status / PLACE seen lines", "[SIGHT add]" in audit and "[SIGHT status]" in audit
      and "[PLACE seen]" in audit)
check("audit carries the contradiction reason", "not seen in 2 clear views" in audit)
check("held -> superseded audit line", "picked up by the agent" in audit)
om3.new_episode("e5")
check("new_episode clears RAM beliefs", om3.tracks == {} and om3.held is None and om3.lookup("Potato") == [])
check("... but the store keeps the evidence", len(store.load_sightings("FloorPlan1", "e3")) > 0)
store.close()
m2 = Memory(path=db)
check("tables survive reopen (CREATE IF NOT EXISTS is idempotent)", len(m2.load_sightings("FloorPlan1")) > 0
      and len(m2.places_seen("FloorPlan1")) > 0)
m2.close()

# ================================================================================================== 10
print("\n10. purity")
src = open(os.path.join(HERE, "objmem.py"), encoding="utf-8").read()
check("objmem imports nothing from the simulator", all(t not in src for t in
      ("import thor_actions", "ai2thor", "import honest", "import webapp", "Controller")))
check("objmem never stores ids", "objectId" not in src and "object_id" not in src)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
try:
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
except OSError:
    pass
sys.exit(0 if FAIL == 0 else 1)
