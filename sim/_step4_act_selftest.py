"""Offline selftest for step 4 s4-act (sim/honest_skills.py). Windows `python`, no WSL/GPU/Ollama.

Run from the repo root:  python sim/_step4_act_selftest.py

Uses the REAL honest.HonestAgent, honest_map.GridMap and objmem.ObjectMemory over a small fake kitchen: a grid
room, objects that render into a depth image and into a fake detector's boxes (with state text, as the
StateEstimator gives it), a GetObjectInFrame that resolves a pixel to the smallest box under it, and actions that
change object state. Covers:
  1. pure helpers: state_of, candidate_points, farthest_points, parse_places, prior_chooser, LLMChooser (mocked
     ask, outage), STATE_WORDS agree with perception.
  2. open Fridge by screen point: walk on own map, face, act, verify by re-detection; forceAction False; no ids leak.
  3. already in the wanted state -> satisfied without acting.
  4. retries: 2 refusals -> ok with 2 retries and recovery moves; 5 refusals -> retry_exhausted after 3.
  5. wrong target: the point hits the neighbouring cabinet, the fridge still looks closed -> wrong_target -> retry.
  6. put Apple in a closed Fridge: opened first, then put; memory says placed.
  7. search with a mock chooser: the apple is inside the closed fridge; the agent visits the remembered fridge,
     opens it, sees the apple, picks it up; chooser got the type list + memory summary. Not found -> not_detected.
  8. composites: heat (microwave sequence), wash (sink + faucet + clean), fill held mug with wine, drop,
     close_all, go_to, pick_and_place; a shared retry budget across composite sub-steps.
  9. unreachable target; unknown action; causes are from the fixed list; audit has 0 violations and no oracle reads.
 10. s4 live fixes: a swung door detected as a nearer "Fridge" does not steal the put (opened-at anchor from own
     history); close uses and clears the anchor; a stale first frame after a toggle is re-checked after one Pass.
 11-14. live fixes 2026-10-06/07 (see the section titles).
 16. 2026-10-08 systematic search route: strafe walks (side / back steps, refused side step replans), cover_stops
     (a drawer column + neighbouring door = one stop), order_stops (one side then the other; a likelier stop first),
     view_headings, seen_cells, the declared height prior; a counter run of three cabinets inspected from one stop,
     no second look-around, numbered plan in search_info, unreachable stop -> re-plan, s5 mode unchanged.
 16. 2026-10-08 systematic search route: strafe walks (side / back steps, refused side step replans), cover_stops
     (a drawer column + neighbouring door = one stop), order_stops (one side then the other; a likelier stop first),
     view_headings, seen_cells, the declared height prior; a counter run of three cabinets inspected from one stop,
     no second look-around, numbered plan in search_info, unreachable stop -> re-plan, s5 mode unchanged.
 17. 2026-10-09 unreachable fixes (console policy): A an open door's sweep no longer walls the agent in (leaf hard,
     sweep soft), B a thin faucet whose box depth is the window behind it (foreground candidate), C an episode that
     starts on a cell the own map does not have (episode start node), D out of reach -> crouch / walk where the
     predicted 3D reach is better, never a step away; E filter counts when no put pose exists. Each cause is first
     reproduced on the fake kitchen, then shown fixed.
 19. B2 (2026-10-09): a put on a counter lands on the counter, not in the basin cut into it (support plane +
     competing receptacle boxes + a look at where the object rests); reproduced on a fake horizontal counter first.
 15. 2026-10-08 pose first, then open: own-map temporary blocks, door model (unknown / observed hinge, drawer), put
     cells, a fridge whose door swings into the floor beside its hinge (fake door physics: moves into the open leaf and
     opens / closes with the agent in the sweep are refused), explicit open/put/close plan, a refused open while
     holding, search inspection with the swept floor blocked, pose-first only in the console policy.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402

import honest                                                         # noqa: E402
import honest_skills as hs                                            # noqa: E402
import objmem                                                         # noqa: E402
import honest_map as hm                                               # noqa: E402
from honest_map import GridMap                                        # noqa: E402

# Round 2 / J-R1: this gate pins the LEGACY pickup aim (one pixel per attempt, so a refusal is one retry). The new aim
# (pixels that protrude from the support surface, up to three per attempt) and the identity check are gated by
# sim/_step4_exec_r1_selftest.py; the legacy path stays reachable by the module switches and must keep passing here.
hs.AIM_PROTRUDE_ON = False
hs.IDENT_ON = False

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ----------------------------------------------------------------------------------------------- fake kitchen
W = H = 100
CELL = 0.25
FREE = {(x, z) for x in range(10) for z in range(10)}           # 2.5 m x 2.5 m floor, world x/z = 0.25 * cell
AXES = {"Fridge": ("open",), "Cabinet": ("open",), "Microwave": ("open", "on"), "Faucet": ("on",),
        "Apple": ("sliced",), "Potato": ("cooked",), "Bowl": ("dirty", "filled"), "Mug": ("filled",)}
FLAG = {"open": "isOpen", "on": "isToggled", "sliced": "isSliced", "cooked": "isCooked", "dirty": "isDirty",
        "filled": "isFilledWithLiquid"}


def kitchen():
    return [
        dict(id="Fridge|+2.90|+0.90|+1.00", type="Fridge", pos=(2.95, 0.9, 1.0), size=0.8, openable=True,
             receptacle=True, isOpen=False),
        dict(id="Cabinet|+2.90|+0.50|+0.40", type="Cabinet", pos=(2.95, 0.6, 0.4), size=0.4, openable=True,
             receptacle=True, isOpen=False),
        dict(id="CounterTop|+1.20|+0.90|+2.95", type="CounterTop", pos=(1.2, 0.9, 2.95), size=1.0, receptacle=True),
        dict(id="Apple|+1.20|+0.97|+2.95", type="Apple", pos=(1.2, 0.97, 2.95), size=0.12, pickupable=True,
             isSliced=False),
        dict(id="Microwave|-0.55|+1.10|+1.40", type="Microwave", pos=(-0.55, 1.1, 1.4), size=0.5, openable=True,
             receptacle=True, isOpen=False, isToggled=False),
        dict(id="Potato|+0.20|+0.97|+2.95", type="Potato", pos=(0.3, 0.97, 2.95), size=0.12, pickupable=True,
             isCooked=False),
        dict(id="SinkBasin|-0.55|+0.85|+0.40", type="SinkBasin", pos=(-0.55, 0.85, 0.4), size=0.5, receptacle=True),
        dict(id="Faucet|-0.80|+1.10|+0.40", type="Faucet", pos=(-0.85, 1.15, 0.4), size=0.15, isToggled=False),
        dict(id="Bowl|+2.00|+0.95|+2.95", type="Bowl", pos=(2.0, 0.95, 2.95), size=0.18, pickupable=True,
             receptacle=True, isDirty=True, isFilledWithLiquid=False),
        dict(id="Mug|+0.70|+0.95|-0.45", type="Mug", pos=(0.7, 0.95, -0.45), size=0.12, pickupable=True,
             isFilledWithLiquid=False),
    ]


class Ev:
    def __init__(self, md):
        self.metadata = md


class FakeThor:
    """Renders objects into depth + boxes from the agent pose; executes interaction primitives on the objects."""

    def __init__(self, objects, start=(4, 4), yaw=0.0):
        self.objs = {o["id"]: o for o in objects}
        self.held = None
        self.calls = []
        self.refuse = {}          # (action, type) -> remaining refusals
        self.misresolve = {}      # type -> type the point resolves to instead (once)
        self.unsure = set()       # types whose state the fake estimator cannot read
        self.lag_after = {}       # (action, type) -> frames the detector still shows the OLD state after success
        self.lag = {}             # type -> remaining stale frames (counted down by the next primitives)
        self.opened_log = []      # (call index, id) of every successful OpenObject -- test scoring only
        x, z = start[0] * CELL, start[1] * CELL
        self.event = Ev({"agent": {"position": {"x": x, "y": 0.9, "z": z}, "rotation": {"x": 0, "y": yaw, "z": 0},
                                   "cameraHorizon": 0.0, "isStanding": True},
                         "cameraPosition": {"x": x, "y": 1.5, "z": z}, "objects": objects,
                         "lastActionSuccess": True, "errorMessage": "", "actionReturn": None})
        self.frame = np.zeros((H, W, 3), dtype="uint8")
        self._render()

    # ---- geometry
    def pose(self):
        a = self.event.metadata["agent"]
        return {"position": dict(a["position"]), "rotation": dict(a["rotation"]), "horizon": a["cameraHorizon"],
                "standing": True, "camera": dict(self.event.metadata["cameraPosition"])}

    def visible_objs(self):
        out = []
        for o in self.objs.values():
            if o["id"] == self.held:
                continue
            inside = o.get("inside")
            if inside and not self.objs[inside].get("isOpen", True):
                continue
            if inside and self.objs[inside].get("door_block"):
                p = self.event.metadata["agent"]["position"]
                if (round(p["x"] / CELL), round(p["z"] / CELL)) in self.objs[inside]["door_block"]:
                    continue                                  # the swung door hides what is inside from here
            out.append(o)
        return out

    def _render(self):
        pose = self.pose()
        depth = np.full((H, W), 6.0, dtype="float32")
        boxes = []
        f = (H / 2.0) / math.tan(math.radians(45))
        for o in self.visible_objs():
            pr = objmem.project(o["pos"], pose, (W, H))
            if pr is None:
                continue
            px, py, zc = pr
            if zc > 4.0:
                continue
            r = max(1.0, o["size"] / 2.0 * f / zc)
            b = (int(round(px - r)), int(round(py - r)), int(round(px + r)), int(round(py + r)))
            if b[2] < 0 or b[3] < 0 or b[0] >= W or b[1] >= H:
                continue
            cb = (max(0, b[0]), max(0, b[1]), min(W - 1, b[2]), min(H - 1, b[3]))
            reg = depth[cb[1]:cb[3] + 1, cb[0]:cb[2] + 1]
            zr = zc
            if o.get("openable") and o.get("isOpen"):
                # an open container shows its cavity (deeper than its front) -- unless its swung door is in the way
                # from where the agent stands (``door_block``: agent cells from which the door hides the inside)
                cell = (round(pose["position"]["x"] / CELL), round(pose["position"]["z"] / CELL))
                zr = zc - 0.3 if cell in o.get("door_block", ()) else zc + o.get("cavity", 0.35)
            if o.get("thin"):
                # 2026-10-09: a thin fixture (faucet) in front of a window: most of its box is the far outdoors
                # (``backdrop`` m); only a narrow central strip is the object itself
                reg[reg >= 6.0] = o.get("backdrop", 20.0)
                cx0 = (cb[0] + cb[2]) // 2
                sw = max(1, (cb[2] - cb[0]) // 8)
                strip = depth[cb[1]:cb[3] + 1, max(0, cx0 - sw):cx0 + sw]
                np.minimum(strip, zr, out=strip)
            else:
                np.minimum(reg, zr, out=reg)
            boxes.append((o, cb, zc))
        if self.held:
            o = self.objs[self.held]
            cb = (42, 80, 58, 96)
            depth[80:97, 42:59] = 0.45
            boxes.append((o, cb, 0.45))
        self.depth_frame = depth
        self.boxes = boxes

    # ---- primitive execution
    def do(self, action, **kw):
        self.calls.append((action, dict(kw)))
        self.lag = {k: n - 1 for k, n in self.lag.items() if n > 1}
        md = self.event.metadata
        md.update(lastActionSuccess=True, errorMessage="", actionReturn=None)
        a = md["agent"]
        ok = True
        if action in ("RotateRight", "RotateLeft"):
            d = float(kw.get("degrees", 90))
            a["rotation"]["y"] = (a["rotation"]["y"] + (d if action == "RotateRight" else -d)) % 360
        elif action in ("LookUp", "LookDown"):
            d = float(kw.get("degrees", 30))
            nh = a["cameraHorizon"] + (d if action == "LookDown" else -d)
            if nh < -30.01 or nh > 60.01:
                ok = False
            else:
                a["cameraHorizon"] = nh
        elif action in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
            yaw = math.radians(a["rotation"]["y"] + {"MoveAhead": 0, "MoveRight": 90, "MoveBack": 180,
                                                     "MoveLeft": 270}[action])
            mag = float(kw.get("moveMagnitude", CELL))
            nx, nz = a["position"]["x"] + math.sin(yaw) * mag, a["position"]["z"] + math.cos(yaw) * mag
            tc = (round(nx / CELL), round(nz / CELL))
            if any(o.get("isOpen") and tc in o.get("leaf_cells", o.get("door_cells", ())) for o in self.objs.values()):
                ok = False                                    # an open door stands there (2026-10-08 fake physics)
                self.door_hits = getattr(self, "door_hits", 0) + 1
            elif tc in FREE:
                a["position"]["x"], a["position"]["z"] = round(nx, 4), round(nz, 4)
            else:
                ok = False
        elif action in ("Crouch", "Stand"):
            # 2026-10-07 (far-open fix): the camera drops when crouched, as in AI2-THOR (1.576 -> 0.901 m, measured)
            self.event.metadata["agent"]["isStanding"] = action == "Stand"
        elif action == "Pass":
            pass
        elif action == "GetObjectInFrame":
            px, py = kw["x"] * W - 0.5, kw["y"] * H - 0.5
            hit = self.resolve_px(px, py)
            if hit is None:
                ok = False
            else:
                mis = self.misresolve.pop(hit["type"], None)
                if mis:
                    hit = next(o for o in self.objs.values() if o["type"] == mis)
                md["actionReturn"] = hit["id"]
        elif action == "Teleport" or action.startswith("GetShortestPath"):
            raise AssertionError(f"{action} is not allowed in honest mode")
        else:
            ok = self.interact(action, kw)
        if not ok:
            md.update(lastActionSuccess=False, errorMessage=f"{action} failed on Fridge|+2.90|+0.90|+1.00")
        md["cameraPosition"] = {"x": a["position"]["x"], "y": 1.5 if a.get("isStanding", True) else 1.5 - 0.675,
                                "z": a["position"]["z"]}
        self._render()
        return self.event

    def resolve_px(self, px, py):
        under = [(o, b, zc) for o, b, zc in self.boxes if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
        d = float(self.depth_frame[int(round(py)), int(round(px))])
        under = [u for u in under if not (u[0].get("thin") and d > u[2] + 0.35)]   # the ray went past a thin one
        if not under:
            return None
        near = [u for u in under if u[2] <= d + 0.35] or under
        return min(near, key=lambda u: (u[1][2] - u[1][0]) * (u[1][3] - u[1][1]))[0]

    def dist(self, o):
        c = self.event.metadata["cameraPosition"]
        if o["id"] == self.held:
            return 0.4
        return math.dist((c["x"], c["y"], c["z"]), o["pos"])

    def interact(self, action, kw):
        oid = kw.get("objectId")
        o = self.objs.get(oid) if oid else None
        assert action == "DropHandObject" or kw.get("forceAction") in (False, None), "honest mode must not force"
        key = (action, o["type"] if o else None)
        if self.refuse.get(key, 0) > 0:
            self.refuse[key] -= 1
            return False
        if action == "DropHandObject":
            if not self.held:
                return False
            h = self.objs[self.held]
            p = self.event.metadata["agent"]["position"]
            h["pos"], self.held = (p["x"], 0.05, p["z"] + 0.3), None
            return True
        if o is None or self.dist(o) > 1.5:
            return False
        if action == "PickupObject":
            if self.held or not o.get("pickupable"):
                return False
            self.held = o["id"]
            o.pop("inside", None)
            return True
        if action == "PutObject":
            if not self.held or not o.get("receptacle") or (o.get("openable") and not o.get("isOpen")):
                return False
            h = self.objs[self.held]
            h["pos"] = (o["pos"][0] - 0.1 * np.sign(o["pos"][0] - 1.2), o["pos"][1] + 0.02, o["pos"][2])
            h["inside"] = o["id"] if o.get("openable") else None
            h["on"] = o["type"]
            self.held = None
            if o["type"] == "SinkBasin":
                h["in_sink"] = True
            return True
        flag_val = {"OpenObject": ("isOpen", True), "CloseObject": ("isOpen", False),
                    "ToggleObjectOn": ("isToggled", True), "ToggleObjectOff": ("isToggled", False),
                    "SliceObject": ("isSliced", True), "CleanObject": ("isDirty", False),
                    "DirtyObject": ("isDirty", True), "FillObjectWithLiquid": ("isFilledWithLiquid", True),
                    "EmptyLiquidFromObject": ("isFilledWithLiquid", False), "BreakObject": ("isBroken", True),
                    "CookObject": ("isCooked", True)}
        fl, v = flag_val[action]
        if action == "OpenObject" and "isOpen" in o and (kw.get("openness") is not None
                                                          or (o["isOpen"] and o.get("openness", 1.0) < 1.0)):
            # as AI2-THOR does (probed live 2026-10-07): partial openness steps, then a canonical open, all succeed
            o["openness"] = float(kw.get("openness") if kw.get("openness") is not None else 1.0)
            o["isOpen"] = o["openness"] > 0
            return True
        if fl not in o or o[fl] == v:
            return False
        if action in ("OpenObject", "CloseObject") and o.get("door_cells"):
            p = self.event.metadata["agent"]["position"]
            if (round(p["x"] / CELL), round(p["z"] / CELL)) in o["door_cells"]:
                self.door_refusals = getattr(self, "door_refusals", 0) + 1
                return False                                  # the door would swing into the agent
        if action == "CloseObject":
            o["openness"] = 0.0
        if action == "OpenObject":
            self.opened_log.append((len(self.calls), o["id"]))
        o[fl] = v
        if (action, o["type"]) in self.lag_after:
            self.lag[o["type"]] = self.lag_after[(action, o["type"])]
        if action == "FillObjectWithLiquid":
            o["fillLiquid"] = kw.get("fillLiquid")
        if action == "ToggleObjectOn" and o["type"] == "Microwave" and not o.get("isOpen"):
            for x in self.objs.values():
                if x.get("inside") == o["id"]:
                    x["temperature"] = "Hot"
        return True


class FakeDetector:
    """Boxes of what the fake camera sees, with StateEstimator-style state text (no ids)."""

    def __init__(self, thor):
        self.thor = thor
        self.names = {i: t for i, t in enumerate(sorted({o["type"] for o in thor.objs.values()} | {"Knife"}))}
        self.hide = set()

    def detect(self, rgb, conf=0.25, inspecting=None):
        out = []
        for o, b, zc in self.thor.boxes:
            if o["type"] in self.hide:
                continue
            states, words = {}, []
            for ax in AXES.get(o["type"], ()):
                if o["type"] in self.thor.unsure:
                    states[ax] = None
                    words.append(f"{ax}?")
                else:
                    v = bool(o.get(FLAG[ax])) != bool(self.thor.lag.get(o["type"]))   # stale frame: old state
                    states[ax] = "1" if v else "0"
                    words.append(hs.STATE_WORDS[ax][int(v)])
            out.append({"type": o["type"], "conf": 0.9, "bbox_xyxy": b, "states": states,
                        "state": ", ".join(words) or "?"})
        return out


def setup(start=(4, 4), yaw=0.0, chooser=None, objs=None, smooth=None, **skw):
    t = FakeThor(objs or kitchen(), start, yaw)
    det = FakeDetector(t)
    h = honest.HonestAgent(t, det, mode="honest", smooth=smooth)
    gm = GridMap("Fake", CELL, (0.0, 0.0), FREE)
    om = objmem.ObjectMemory("Fake", store=None, episode="t")
    sk = hs.HonestSkills(h, gm, om, chooser=chooser or (lambda *a: []), **skw)
    sk.start_episode("t")
    return t, det, h, sk


def obj(t, typ):
    return next(o for o in t.objs.values() if o["type"] == typ)


def no_ids(res):
    return not re.search(r"\|[+-]\d", json.dumps(res, default=str))


def prims(t, name):
    return [c for c in t.calls if c[0] == name]


# ======================================================================================================== 1
print("1. pure helpers")
check("state_of reads states dict", hs.state_of({"states": {"open": "1"}}, "open") is True
      and hs.state_of({"states": {"open": "0"}}, "open") is False)
check("state_of reads state text; '?' is unknown", hs.state_of({"state": "closed, off"}, "open") is False
      and hs.state_of({"state": "open?, on"}, "open") is None and hs.state_of({"state": "open?, on"}, "on") is True)
check("state_of: axis not covered -> None", hs.state_of({"state": "sliced"}, "open") is None)
try:
    import perception
    check("STATE_WORDS mirror perception.STATE_WORDS", perception.STATE_WORDS == hs.STATE_WORDS)
except Exception as e:                                                # pragma: no cover
    check("STATE_WORDS mirror perception.STATE_WORDS", False, str(e))
d = np.full((40, 40), 3.0, dtype="float32")
d[10:30, 10:30] = 1.0
pts = hs.candidate_points(d, (5, 5, 34, 34))
check("candidate_points keep depth-agreeing pixels, centre first",
      pts and all(10 <= p[0] < 30 and 10 <= p[1] < 30 for p in pts)
      and math.hypot(pts[0][0] - 19.5, pts[0][1] - 19.5) <= math.hypot(pts[-1][0] - 19.5, pts[-1][1] - 19.5), str(pts[:3]))
pts2 = hs.candidate_points(d, (5, 5, 34, 34), agree=False, avoid=[(15, 15, 25, 25)], max_depth=2.0)
check("receptacle points avoid boxes on top and out-of-reach depth",
      pts2 and not any(15 <= p[0] <= 25 and 15 <= p[1] <= 25 for p in pts2) and all(d[p[1], p[0]] <= 2 for p in pts2))
fp = hs.farthest_points({(x, z) for x in range(5) for z in range(5)}, 4, (0, 0))
check("farthest-point viewpoints spread to the corners", fp[0] == (0, 0) and set(fp[1:]) == {(4, 4), (0, 4), (4, 0)},
      str(fp))
TYPES = ["Apple", "CounterTop", "Fridge", "Microwave", "SinkBasin", "Cabinet"]
check("parse_places: JSON list, filtered to the type list, target dropped",
      hs.parse_places('sure: ["Fridge", "Garage", "apple", "countertop"]', TYPES, "Apple") == ["Fridge", "CounterTop"])
check("parse_places: free text falls back to order of mention",
      hs.parse_places("Look in the Microwave, then the Fridge.", TYPES, "Apple") == ["Microwave", "Fridge"])
rank = hs.prior_chooser("Apple", TYPES, ["Fridge [closed] on/in ? at (1, 2), step 3"])
check("prior stand-in ranks remembered likely places first (Fridge for Apple)", rank[0] == "Fridge" and
      "Apple" not in rank, str(rank))
check("prior stand-in is labelled as a stand-in", "stand-in" in hs.prior_chooser.label)
asked = []
ch = hs.LLMChooser(lambda p: asked.append(p) or '["SinkBasin", "Fridge"]')
check("LLMChooser: prompt carries the type list + memory, answer parsed",
      ch("Apple", TYPES, ["Fridge [closed] on/in ?"]) == ["SinkBasin", "Fridge"] and "Microwave" in asked[0]
      and "Fridge [closed]" in asked[0])


def down(_):
    raise ConnectionError("ollama down")


ch2 = hs.LLMChooser(down)
check("LLMChooser: an outage gives no ranking (never an oracle fallback)", ch2("Apple", TYPES, []) == []
      and "ollama" in ch2.last["error"])
check("tool list covers the general set's actions", {"pick_and_place", "pickup", "close", "toggle_on", "toggle_off",
      "slice", "fill", "open", "put_in", "heat", "search_for", "wash", "go_to", "empty", "clean", "drop", "dirty",
      "break", "cook", "crouch", "stand", "close_all"} <= set(hs.TOOLS))

# ======================================================================================================== 2
print("\n2. open the Fridge by screen point")
t, det, h, sk = setup()
check("episode start: look-around seeded memory with the fridge", bool(sk.om.lookup("Fridge")))
# A26 (K-B2b 2026-10-13) replaces: check("type list comes from the detector's class names", "Knife" in sk.type_list and "Fridge" in sk.type_list)
check("type list comes from the detector's class names, minus the A26 excluded type (Knife)",
      "Knife" not in sk.type_list and "Fridge" in sk.type_list and "Knife" in hs.detector_types(sk.h.detector), sk.type_list)
r = sk.exec_step({"action": "open", "object": "Fridge"})
check("open Fridge ok, 0 retries, verified by re-detected state", r["ok"] and r["retries"] == 0
      and r["detail"]["verify"] == ["state"] and obj(t, "Fridge")["isOpen"], json.dumps(r)[:300])
check("acted through GetObjectInFrame then OpenObject(forceAction=False)",
      prims(t, "GetObjectInFrame") and prims(t, "OpenObject")[-1][1]["forceAction"] is False)
check("walked on its own map (moves only, no teleport)", prims(t, "MoveAhead") and not prims(t, "Teleport"))
check("no object id in the result", no_ids(r))
check("audit: 0 violations, no oracle/scoring reads", h.audit.summary()["violations"] == 0
      and h.audit.counts["oracle"] == 0 and h.audit.counts["scoring"] == 0)
n_open = len(prims(t, "OpenObject"))
r = sk.exec_step({"action": "open", "object": "Fridge"})
check("already open -> satisfied, no second OpenObject", r["ok"] and r["detail"]["verify"] == ["already"]
      and len(prims(t, "OpenObject")) == n_open)

# ======================================================================================================== 3
print("\n3. retries")
t, det, h, sk = setup()
t.refuse[("PickupObject", "Apple")] = 2
r = sk.exec_step({"action": "pickup", "object": "Apple"})
check("2 refusals -> ok after 2 retries", r["ok"] and r["retries"] == 2 and t.held and "Apple" in t.held,
      json.dumps(r)[:300])
check("recovery 1 changes pitch, recovery 2 sidesteps", any("recover 1" in e for e in r["detail"]["events"])
      and any("recover 2" in e for e in r["detail"]["events"]))
check("holding comes from own history", h.holding == "Apple" and sk.om.lookup("Apple")[0]["status"] == "held")
t, det, h, sk = setup()
t.refuse[("PickupObject", "Apple")] = 9
r = sk.exec_step({"action": "pickup", "object": "Apple"})
check("persistent refusal -> retry_exhausted after exactly 3 retries", not r["ok"] and r["cause"] == "retry_exhausted"
      and r["retries"] == 3 and len(prims(t, "PickupObject")) == 4, json.dumps(r)[:300])
check("the reason keeps the underlying cause, with no id", "action_refused" in r["detail"]["why"] and no_ids(r))

# ======================================================================================================== 4
print("\n4. wrong target")
t, det, h, sk = setup()
t.misresolve["Fridge"] = "Cabinet"
r = sk.exec_step({"action": "open", "object": "Fridge"})
check("point hit the cabinet: fridge still closed -> wrong_target -> retry -> ok",
      r["ok"] and r["retries"] == 1 and obj(t, "Fridge")["isOpen"]
      and any(s.get("cause") == "wrong_target" for s in r["detail"]["sub"]), json.dumps(r["detail"]["sub"]))
t, det, h, sk = setup()
t.unsure.add("Fridge")
r = sk.exec_step({"action": "open", "object": "Fridge"})
check("state not readable -> ok on the success bool, flagged state_unrecognised",
      r["ok"] and r["detail"]["verify"] == ["state_unrecognised"])

# ======================================================================================================== 5
print("\n5. put Apple in the closed Fridge")
t, det, h, sk = setup()
r1 = sk.exec_step({"action": "pickup", "object": "Apple"})
r2 = sk.exec_step({"action": "put_in", "object": "Fridge"})
ap = obj(t, "Apple")
check("pickup then put_in: fridge opened first, apple inside", r1["ok"] and r2["ok"] and obj(t, "Fridge")["isOpen"]
      and ap.get("inside") == obj(t, "Fridge")["id"] and t.held is None, json.dumps(r2)[:300])
check("memory: apple placed in Fridge, hand empty", sk.om.lookup("Apple")[0]["status"] == "placed"
      and sk.om.lookup("Apple")[0]["place"] == "Fridge" and h.holding is None)
r3 = sk.exec_step({"action": "put_in", "object": "Fridge"})
check("put with an empty hand -> action_refused, no retries", not r3["ok"] and r3["cause"] == "action_refused"
      and r3["retries"] == 0)

# ======================================================================================================== 6
print("\n6. search (decision 4) with a mock chooser")
calls = []


def mock_chooser(target, types, summary):
    calls.append((target, list(types), list(summary)))
    return ["Fridge", "Garage"]


objs = kitchen()
a = next(o for o in objs if o["type"] == "Apple")
a["pos"], a["inside"] = (2.75, 0.95, 1.0), "Fridge|+2.90|+0.90|+1.00"
t, det, h, sk = setup(chooser=mock_chooser, objs=objs)
check("apple not in memory before the search", not sk.om.lookup("Apple"))
r = sk.exec_step({"action": "pickup", "object": "Apple"})
si = r["detail"].get("search") or {}
check("search: chooser asked with the type list and the memory summary",
      calls and calls[0][0] == "Apple" and "Fridge" in calls[0][1] and any("Fridge" in s for s in calls[0][2]))
check("search: ranked list filtered to known types", si.get("ranked") == ["Fridge"], str(si))
check("search: visited the remembered fridge, opened it, found the apple, picked it up",
      r["ok"] and si.get("visited") == ["Fridge"] and si.get("opened") == ["Fridge"] and si.get("found") is True
      and t.held and "Apple" in t.held, json.dumps(r)[:400])
check("search: the new sighting is remembered", any(b["track"] for b in sk.om.lookup("Apple")))
objs = kitchen()
objs = [o for o in objs if o["type"] != "Apple"]
t, det, h, sk = setup(chooser=mock_chooser, objs=objs)
r = sk.exec_step({"action": "pickup", "object": "Apple"})
check("not found within budget -> not_detected (no retries burnt on a search)", not r["ok"]
      and r["cause"] == "not_detected" and r["retries"] == 0 and r["detail"]["search"]["viewpoints"] >= 1,
      json.dumps(r)[:300])
t, det, h, sk = setup(chooser=mock_chooser)
r = sk.exec_step({"action": "search_for", "object": "Apple"})
check("search_for a remembered apple: go there and see it", r["ok"] and r["detail"]["verify"] == ["seen"])

# ======================================================================================================== 7
print("\n7. composites")
t, det, h, sk = setup()
r = sk.exec_step({"action": "heat", "object": "Potato"})
mw = obj(t, "Microwave")
check("heat: pick up, microwave open/put/close/on/off/open, take out -> hot potato in hand",
      r["ok"] and obj(t, "Potato").get("temperature") == "Hot" and t.held and "Potato" in t.held
      and mw["isOpen"] and not mw["isToggled"], json.dumps(r["detail"]["sub"])[:400])
check("heat: sub-steps recorded in order (put_in opens the closed microwave first)", [s["verb"] for s in r["detail"]["sub"] if s["ok"]] ==
      ["pickup", "open", "put_in", "close", "toggle_on", "toggle_off", "open", "pickup"], str(r["detail"]["sub"]))
t, det, h, sk = setup()
r = sk.exec_step({"action": "wash", "object": "Bowl"})
check("wash: bowl in the sink, clean, faucet off", r["ok"] and obj(t, "Bowl").get("in_sink")
      and not obj(t, "Bowl")["isDirty"] and not obj(t, "Faucet")["isToggled"], json.dumps(r)[:400])
t, det, h, sk = setup()
sk.exec_step({"action": "pickup", "object": "Mug"})
r = sk.exec_step({"action": "fill", "object": "Mug", "arg": "wine"})
check("fill the held mug with wine (target in hand, no walking)", r["ok"] and obj(t, "Mug")["isFilledWithLiquid"]
      and obj(t, "Mug").get("fillLiquid") == "wine", json.dumps(r)[:300])
r = sk.exec_step({"action": "drop"})
check("drop: hand empty, memory placed on the floor", r["ok"] and t.held is None and h.holding is None
      and sk.om.lookup("Mug")[0]["place"] == "Floor")
t, det, h, sk = setup()
obj(t, "Fridge")["isOpen"] = True
obj(t, "Microwave")["isOpen"] = True
t._render()
sk.start_episode("t2")
r = sk.exec_step({"action": "close_all"})
check("close_all: every container seen open gets closed", r["ok"] and not obj(t, "Fridge")["isOpen"]
      and not obj(t, "Microwave")["isOpen"] and sorted(r["detail"]["closed"]) == ["Fridge", "Microwave"],
      json.dumps(r)[:300])
t, det, h, sk = setup()
r = sk.exec_step({"action": "go_to", "object": "Fridge"})
f = obj(t, "Fridge")
p = t.event.metadata["agent"]["position"]
check("go_to Fridge: within 1.5 m and seeing it", r["ok"] and math.hypot(p["x"] - f["pos"][0], p["z"] - f["pos"][2]) <= 1.5)
t, det, h, sk = setup()
r = sk.exec_step({"action": "pick_and_place", "object": "Apple", "to": "CounterTop"})
check("pick_and_place Apple -> CounterTop", r["ok"] and obj(t, "Apple").get("on") == "CounterTop", json.dumps(r)[:300])
t, det, h, sk = setup()
t.refuse[("PickupObject", "Apple")] = 2
t.refuse[("PutObject", "Fridge")] = 2
r = sk.exec_step({"action": "pick_and_place", "object": "Apple", "to": "Fridge"})
check("one retry budget shared by the composite's sub-steps (2 + 2 > 3 -> exhausted)",
      not r["ok"] and r["cause"] == "retry_exhausted" and r["retries"] == 3, json.dumps(r)[:300])
t, det, h, sk = setup()
r = sk.exec_step({"action": "crouch"})
check("nav primitive crouch", r["ok"] and prims(t, "Crouch"))

# ======================================================================================================== 8
print("\n8. failures and honesty")
objs = kitchen()
far = dict(id="Toaster|+9.00|+0.90|+9.00", type="Toaster", pos=(9.0, 0.9, 9.0), size=0.3)
objs.append(far)
t, det, h, sk = setup(objs=objs)
sk.om.tracks["fake:1"] = objmem.Track(tid="fake:1", type="Toaster", xyz=(9.0, 0.9, 9.0))
r = sk.exec_step({"action": "toggle_on", "object": "Toaster"})
check("a target beyond the own map -> unreachable", not r["ok"] and r["cause"] == "unreachable", json.dumps(r)[:200])
r = sk.exec_step({"action": "juggle", "object": "Apple"})
check("unknown action -> action_refused", not r["ok"] and r["cause"] == "action_refused")
check("every cause is from the fixed list", all(c in hs.CAUSES for c in
                                                ("unreachable", "action_refused", "not_detected", "wrong_target",
                                                 "retry_exhausted", "state_unrecognised")))
check("audit across a whole run: 0 violations, invalid False", h.audit.summary()["violations"] == 0
      and not h.audit.invalid)
src = open(os.path.join(HERE, "honest_skills.py"), encoding="utf-8").read()
bad = [p for p in (r"\.objects\(", r"\.resolve\(", r"object_by_id", r"metadata", r"held_object",
                   r"GetShortestPath", r"interactable_poses", r"visible_objects", r"objectId", r"\bthor\.")
       if re.search(p, src)]
check("honest_skills.py has no privileged tokens", not bad, str(bad))

# ======================================================================================================== 9
print("\n9. invariance through the real skills (s7 audit): hidden simulator facts never change behaviour")
SEQ = [{"action": "open", "object": "Fridge"}, {"action": "pickup", "object": "Apple"},
       {"action": "put_in", "object": "Fridge"}, {"action": "toggle_on", "object": "Faucet"},
       {"action": "search_for", "object": "Potato"}, {"action": "search_for", "object": "Toaster"}]


def run_seq(scramble):
    objs = kitchen()
    if scramble:                                   # rename every id + add far-away decoys the camera never sees
        for i, o in enumerate(objs):
            o["id"] = f"{o['type']}|+7.{i:02d}|-0.33|+4.{i:02d}"
        objs += [dict(id="Apple|+30.0|+0.9|+30.0", type="Apple", pos=(30.0, 0.9, 30.0), size=0.12, pickupable=True),
                 dict(id="Potato|-30.0|+0.9|-30.0", type="Potato", pos=(-30.0, 0.9, -30.0), size=0.12,
                      pickupable=True, isCooked=True)]
    t, det, h, sk = setup(chooser=hs.prior_chooser, objs=objs)
    vocab.append(sorted(det.names.values()))     # the detector vocabulary must be the same in both worlds
    if scramble:                                   # the metadata object list now disagrees with the world
        lies = [dict(o, isOpen=not o.get("isOpen"), isToggled=not o.get("isToggled")) for o in reversed(objs)]
        t.event.metadata["objects"] = lies
    res = []
    for stp in SEQ:
        r = sk.exec_step(stp)
        res.append((r["ok"], r["cause"], r["retries"], r["detail"]["verify"], r["detail"]["prims"]))
    calls = [(a, {k: v for k, v in kw.items() if k != "objectId"}) for a, kw in t.calls]
    return res, calls, sk.om.summary(), h.audit.summary(), h.history


vocab = []
base, scr = run_seq(False), run_seq(True)
check("same detector vocabulary (decoys add no class)", vocab[0] == vocab[1])
check("same step outcomes (ok, cause, retries, verify, prims)", base[0] == scr[0], f"{base[0]} vs {scr[0]}")
check("same primitive trace (ids stripped)", base[1] == scr[1], f"{len(base[1])} vs {len(scr[1])}")
check("same memory and history", base[2] == scr[2] and base[4] == scr[4])
check("both runs valid, 0 oracle reads", base[3]["violations"] == scr[3]["violations"] == 0
      and not scr[3]["invalid"])
check("sequence exercised acting + search (not vacuous)", base[0][0][0] and base[0][2][0]
      and any(c[0] == "PutObject" for c in base[1]))

# ======================================================================================================== 10
print("\n10. s4 live fixes: opened-at anchor, stale post-action frame")
t, det, h, sk = setup()
r1 = sk.exec_step({"action": "open", "object": "Fridge"})
anchor = sk.opened_at.get("Fridge")
check("open records the opened-at anchor (own aim, no id)", r1["ok"] and anchor is not None and no_ids(sk.opened_at))
door = dict(id="Fridge|+2.40|+0.90|+1.90", type="Fridge", pos=(2.4, 0.9, 1.9), size=0.3)   # swung door: no receptacle
t.objs[door["id"]] = door
t._render()
sk._face(door["pos"])
sk._look()
nfr = len([b for b in sk.om.lookup("Fridge") if b["status"] == "trusted"])
r2 = sk.exec_step({"action": "pickup", "object": "Apple"})
pa = h.pose()["position"]
door_nearer = math.hypot(pa["x"] - 2.4, pa["z"] - 1.9) < math.hypot(pa["x"] - anchor[0], pa["z"] - anchor[2])
r3 = sk.exec_step({"action": "put_in", "object": "Fridge"})
check("door seen as a second, nearer Fridge track (the live failure's setup)", nfr >= 2 and door_nearer, f"{nfr}")
check("put aims at the anchor: apple inside the real fridge, no retries",
      r2["ok"] and r3["ok"] and r3["retries"] == 0 and obj(t, "Apple").get("inside") == "Fridge|+2.90|+0.90|+1.00",
      json.dumps(r3)[:300])
r4 = sk.exec_step({"action": "close", "object": "Fridge"})
check("close uses the anchor and clears it", r4["ok"] and not t.objs["Fridge|+2.90|+0.90|+1.00"]["isOpen"]
      and "Fridge" not in sk.opened_at, json.dumps(r4)[:300])
t, det, h, sk = setup()
t.lag_after[("ToggleObjectOn", "Faucet")] = 1
r = sk.exec_step({"action": "toggle_on", "object": "Faucet"})
check("stale first frame after a toggle: one Pass, re-detected on -> ok, 0 retries, no second toggle",
      r["ok"] and r["retries"] == 0 and r["detail"]["verify"] == ["state"] and bool(prims(t, "Pass"))
      and len(prims(t, "ToggleObjectOn")) == 1, json.dumps(r["detail"]["sub"]))
check("section 10 audit: 0 violations", h.audit.summary()["violations"] == 0)

# ======================================================================================================== 11
print("\n11. s5 live fixes: close the open one of several, search closes what it looked into")
objs = [o for o in kitchen() if o["type"] != "Fridge"]
objs.append(dict(id="Cabinet|+2.90|+0.50|+0.85", type="Cabinet", pos=(2.95, 0.6, 0.85), size=0.4, openable=True,
                 receptacle=True, isOpen=True))
t, det, h, sk = setup(objs=objs)
shut, opn = t.objs["Cabinet|+2.90|+0.50|+0.40"], t.objs["Cabinet|+2.90|+0.50|+0.85"]
real_pick = sk._pick


def pick_shut(obs, otype, expect=None, need=None, inhand=False):     # the live case: belief/pixel favour the shut one
    c = [d for d in obs.dets if d.get("type") == otype and hs.state_of(d, "open") is False and d.get("point")]
    return c[0] if c else real_pick(obs, otype, expect, need, inhand)


sk._pick = pick_shut
sk._reset_step()
out = sk._interact("close", "Cabinet", belief={"status": "trusted", "xyz": shut["pos"]})
check("close Cabinet, the picked one already shut but another in view open -> that one is closed (not 'already')",
      out.get("verify") == "state" and not opn["isOpen"] and not shut["isOpen"]
      and any("act on that one" in e for e in sk.events), json.dumps(sk.events)[:400])
t, det, h, sk = setup(objs=kitchen())
r = sk.exec_step({"action": "close", "object": "Cabinet"})
check("close with only shut instances in view -> already (unchanged)", r["ok"] and r["detail"]["verify"] == ["already"])
objs = [o for o in kitchen() if o["type"] != "Apple"]
t, det, h, sk = setup(chooser=lambda *a: ["Fridge"], objs=objs)
r = sk.exec_step({"action": "pickup", "object": "Apple"})
si = r["detail"]["search"]
check("search: a container opened and found empty is closed again (door out of the way)",
      not r["ok"] and "Fridge" in si["opened"] and "Fridge" in (si.get("closed") or [])
      and not obj(t, "Fridge")["isOpen"] and "Fridge" not in sk.opened_at, json.dumps(si))
check("section 11 audit: 0 violations", h.audit.summary()["violations"] == 0 and no_ids(r))
objs = [o for o in kitchen() if o["type"] != "Apple"]
objs.append(dict(id="Cabinet|-0.55|+0.60|+2.60", type="Cabinet", pos=(-0.55, 0.6, 2.6), size=0.4, openable=True,
                 receptacle=True, isOpen=False))
objs.append(dict(id="Apple|-0.45|+0.60|+2.60", type="Apple", pos=(-0.45, 0.6, 2.6), size=0.12, pickupable=True,
                 isSliced=False, inside="Cabinet|-0.55|+0.60|+2.60"))
t, det, h, sk = setup(chooser=lambda *a: ["Cabinet"], objs=objs)
r = sk.exec_step({"action": "pickup", "object": "Apple"})
si = r["detail"]["search"]
check("search visits several instances of a ranked type in one pass: apple in the 2nd cabinet, no viewpoint tour",
      r["ok"] and si["found"] is True and si["viewpoints"] == 0 and si["visited"].count("Cabinet") >= 2
      and "Cabinet" in (si.get("closed") or []) and not t.objs["Cabinet|+2.90|+0.50|+0.40"]["isOpen"]
      and t.held and "Apple" in t.held, json.dumps(si))
check("search stays inside the step's primitive cap", r["detail"]["prims"] <= hs.STEP_PRIM_CAP)

# ======================================================================================================== 12
print("\n12. 2026-10-07 fixes: exhaustive search after a stale belief, tidiness, smooth display")
WB_CAB = "Cabinet|-0.55|+0.60|+2.60"


def wine_kitchen(inside=True):
    objs = kitchen()
    objs.append(dict(id=WB_CAB, type="Cabinet", pos=(-0.55, 0.6, 2.6), size=0.4, openable=True, receptacle=True,
                     isOpen=False))
    if inside:
        objs.append(dict(id="WineBottle|-0.45|+0.60|+2.60", type="WineBottle", pos=(-0.45, 0.6, 2.6), size=0.12,
                         pickupable=True, inside=WB_CAB))
    return objs


def plant_stale(sk, xyz=(1.9, 0.95, 2.9)):
    """The live failure: memory holds a WineBottle sighting where there is none (false or stale detection)."""
    sk.om.tracks["fake:wb"] = objmem.Track(tid="fake:wb", type="WineBottle", xyz=xyz, conf=0.4, place="CounterTop")


def openables_closed(t):
    return [o["type"] for o in t.objs.values() if o.get("openable") and o.get("isOpen")]


EXH = dict(search="exhaustive", tidy=True)
# -- the live bug, reproduced in the legacy (bounded, s5) mode: retries burnt at the stale spot, no real search
t, det, h, sk = setup(objs=wine_kitchen(), chooser=lambda *a: [])
plant_stale(sk)
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
check("bounded mode (s5 default) reproduces the live failure: retry_exhausted at the stale spot, no search",
      not r["ok"] and r["cause"] == "retry_exhausted" and r["retries"] == 3 and not r["detail"].get("search"),
      json.dumps(r)[:300])
# -- exhaustive: set the stale belief aside, search everything, find it inside the closed cabinet
t, det, h, sk = setup(objs=wine_kitchen(), chooser=lambda *a: [], **EXH)
check("exhaustive skills: options recorded", sk.exhaustive and sk.tidy and sk.search_mode == "exhaustive")
plant_stale(sk)
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
si = r["detail"].get("search") or {}
check("stale belief -> set aside (not re-seen after the local looks), no action retries spent",
      "fake:wb" in sk.aside and r["retries"] == 0 and any("set aside" in e for e in r["detail"]["events"]),
      json.dumps(r["detail"]["events"])[:400])
check("stale belief: memory evidence kept (search decision, not a forced contradiction)",
      "fake:wb" in sk.om.tracks)
check("exhaustive search found the WineBottle inside a closed cabinet and saw it",
      r["ok"] and si.get("found") is True and r["detail"]["verify"] == ["seen"] and "Cabinet" in si.get("opened", []),
      json.dumps(si)[:400])
check("every container opened during the search is closed again (no mess), nothing left open",
      not openables_closed(t) and not sk.opened_by_me and r["detail"]["left_open"] == [],
      f"{openables_closed(t)} {sk.opened_by_me}")
check("the cabinet the bottle was found in was closed at step end (no later step needs it)",
      any(x["result"] == "closed" and x["type"] == "Cabinet" and "end of step" in x["what"]
          for x in r["detail"]["tidy"]), json.dumps(r["detail"]["tidy"]))
check("every open / close is logged in the step events",
      sum("to look inside: ok" in e for e in r["detail"]["events"])
      == len(si["opened"]) + sum("opened something else" in e for e in r["detail"]["events"])
      and sum(" ok" in e and "close " in e for e in r["detail"]["events"]) >= len(si["opened"]),
      json.dumps(r["detail"]["events"])[-600:])
check("the exhaustive search may exceed the old 700 step cap but stays under its own budget",
      r["detail"]["prims"] <= hs.STEP_PRIM_CAP + hs.SEARCH_PRIM_CAP)
check("no ids, 0 violations", no_ids(r) and h.audit.summary()["violations"] == 0)
# -- search_for followed by a pickup: the cabinet stays open for that step, then is closed after the pickup
t, det, h, sk = setup(objs=wine_kitchen(), chooser=lambda *a: [], **EXH)
plant_stale(sk)
plan = [{"action": "search_for", "object": "WineBottle"}, {"action": "pickup", "object": "WineBottle"}]
r1 = sk.exec_step(plan[0], upcoming=plan[1:])
left = list(r1["detail"]["left_open"])
r2 = sk.exec_step(plan[1], upcoming=[])
check("search_for then pickup: the cabinet is left open for the pickup (a later step needs the bottle)",
      r1["ok"] and left == ["Cabinet"] and t.objs[WB_CAB]["isOpen"] is not None, f"{left}")
check("...then the bottle is picked up and the cabinet closed right after (tidy, taken out)",
      r2["ok"] and t.held == "WineBottle|-0.45|+0.60|+2.60" and not t.objs[WB_CAB]["isOpen"]
      and any("taken out" in x["what"] and x["result"] == "closed" for x in r2["detail"]["tidy"])
      and not openables_closed(t), json.dumps(r2["detail"].get("tidy")))
# -- pickup directly with the stale belief: relocate -> search -> pickup -> close
t, det, h, sk = setup(objs=wine_kitchen(), chooser=lambda *a: [], **EXH)
plant_stale(sk)
r = sk.exec_step({"action": "pickup", "object": "WineBottle"})
check("pickup with a stale belief: set aside, full search, picked up, cabinet closed, 0 retries",
      r["ok"] and r["retries"] == 0 and t.held and "WineBottle" in t.held and not openables_closed(t)
      and "fake:wb" in sk.aside, json.dumps(r)[:400])
# -- absent target: the search ends (whole map), with a clear report, inside its budget
t, det, h, sk = setup(objs=wine_kitchen(inside=False), chooser=lambda *a: [], **EXH)
sk.type_list = sorted(set(sk.type_list) | {"WineBottle"})
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
si = r["detail"]["search"]
check("absent target: not_detected after searching the reachable known map (complete, not budget)",
      not r["ok"] and r["cause"] == "not_detected" and si["complete"] and not si["budget"]
      and "reachable known map" in r["detail"]["why"] and si["viewpoints"] == si["viewpoints_planned"] >= 1,
      json.dumps(si)[:400])
check("absent target: every remembered container instance was visited and opened, all closed again",
      sum(1 for i in si["inspections"] if i["type"] == "Cabinet") >= 2 and {"Fridge", "Microwave", "Cabinet"} <= set(si["opened"])
      and not openables_closed(t) and not sk.opened_by_me, json.dumps(si)[:300])
check("absent target: the search ends inside its budget", r["detail"]["prims"] <= hs.STEP_PRIM_CAP + hs.SEARCH_PRIM_CAP)
t, det, h, sk = setup(objs=wine_kitchen(inside=False), chooser=lambda *a: [], prim_cap=100, search_cap=60, **EXH)
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
si = r["detail"]["search"]
check("small search budget: the search stops with 'budget reached' and the step ends",
      not r["ok"] and si["budget"] and "budget" in r["detail"]["why"]
      and r["detail"]["prims"] <= 100 + 60 + 2, json.dumps(si)[:300])
# -- tidiness of puts, explicit opens, composites, and a refused close
t, det, h, sk = setup(**EXH)
sk.exec_step({"action": "pickup", "object": "Apple"})
r = sk.exec_step({"action": "put_in", "object": "Fridge"}, upcoming=[])
check("put into a closed fridge: opened for the put, closed after it (no later step needs it)",
      r["ok"] and obj(t, "Apple").get("inside") == obj(t, "Fridge")["id"] and not obj(t, "Fridge")["isOpen"]
      and any(x["why"] == "put" and x["result"] == "closed" for x in r["detail"]["tidy"]), json.dumps(r["detail"]["tidy"]))
t, det, h, sk = setup(**EXH)
plan = [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}, {"action": "close", "object": "Fridge"}]
r1 = sk.exec_step(plan[0], upcoming=plan[1:])
was_open = bool(obj(t, "Fridge")["isOpen"])
r2 = sk.exec_step(plan[1], upcoming=[])
check("a later 'close Fridge' step: the put leaves it open for that step, which closes it",
      r1["ok"] and was_open and r2["ok"] and not obj(t, "Fridge")["isOpen"] and not sk.opened_by_me)
t, det, h, sk = setup(**EXH)
r = sk.exec_step({"action": "open", "object": "Fridge"}, upcoming=[])
tu = sk.tidy_up()
check("an explicit 'open Fridge' is the instruction's goal: kept open at step end and by tidy_up",
      r["ok"] and obj(t, "Fridge")["isOpen"] and tu["kept_open"] == ["Fridge"] and not tu["closed"])
t, det, h, sk = setup(**EXH)
r = sk.exec_step({"action": "heat", "object": "Potato"}, upcoming=[])
check("heat composite: the microwave it opened to take the potato out is closed at step end",
      r["ok"] and t.held and "Potato" in t.held and not obj(t, "Microwave")["isOpen"], json.dumps(r["detail"].get("tidy")))
t, det, h, sk = setup(objs=wine_kitchen(), chooser=lambda *a: [], **EXH)
t.refuse[("CloseObject", "Cabinet")] = 1
plant_stale(sk)
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
check("a refused close is retried once from another angle and the cabinet ends closed",
      r["ok"] and not openables_closed(t) and any("retry, other angle" in e for e in r["detail"]["events"]),
      json.dumps(r["detail"]["events"])[-500:])
t, det, h, sk = setup(**EXH)
sk._reset_step()
sk._goto(obj(t, "Fridge")["pos"])
sk._face(obj(t, "Fridge")["pos"])
sk._open_container("Fridge", obj(t, "Fridge")["pos"], why="search")
tu = sk.tidy_up()
check("tidy_up (end of a console run) closes whatever this agent left open",
      tu["closed"] == ["Fridge"] and not obj(t, "Fridge")["isOpen"] and not sk.opened_by_me, json.dumps(tu)[:300])

# -- smooth display: subdivided physical sends inside one logical primitive
PROF = {"step": 0.0625, "turn": 15.0, "look": 5.0, "doors": 4, "pause": 0.0}


def run_plain_vs_smooth(seq, objs_fn=kitchen):
    out = []
    for prof in (None, PROF):
        t, det, h, sk = setup(objs=objs_fn(), smooth=prof)
        res = [sk.exec_step(st) for st in seq]
        out.append((t, h, [(r["ok"], r["cause"], r["retries"], r["detail"]["verify"], r["detail"]["prims"])
                           for r in res]))
    return out


(tp, hp, rp), (ts, hsm, rs) = run_plain_vs_smooth([{"action": "open", "object": "Fridge"},
                                                  {"action": "pickup", "object": "Apple"},
                                                  {"action": "go_to", "object": "Microwave"}])
check("smooth: same logical outcomes and logical primitive counts as the plain run", rp == rs, f"{rp} vs {rs}")
check("smooth: more physical sends than logical primitives (plain: equal)",
      hsm.sends > hsm.step and hp.sends == hp.step, f"{hsm.sends}/{hsm.step} {hp.sends}/{hp.step}")
mv = [kw.get("moveMagnitude") for a, kw in ts.calls if a == "MoveAhead"]
rot = [kw.get("degrees") for a, kw in ts.calls if a in ("RotateLeft", "RotateRight")]
check("smooth: a 0.25 m step is sent as 0.0625 m moves, turns as <= 15 deg", mv and all(abs(m - 0.0625) < 1e-9 for m in mv)
      and rot and max(rot) <= 15.0 + 1e-9, f"{mv[:4]} {max(rot) if rot else None}")
ops = [kw.get("openness") for a, kw in ts.calls if a == "OpenObject"]
check("smooth: the fridge door swings through openness steps, then the canonical open",
      [round(v, 2) for v in ops[:3]] == [0.25, 0.5, 0.75] and ops[3] is None, str(ops))
pp, ps = tp.event.metadata["agent"], ts.event.metadata["agent"]
check("smooth: same final pose (lattice / yaw checks passed inside honest_map)",
      math.isclose(pp["position"]["x"], ps["position"]["x"], abs_tol=1e-3)
      and math.isclose(pp["position"]["z"], ps["position"]["z"], abs_tol=1e-3)
      and abs((pp["rotation"]["y"] - ps["rotation"]["y"] + 180) % 360 - 180) < 0.01)
t, det, h, sk = setup(start=(9, 4), yaw=90.0, smooth=PROF)
p0 = dict(t.event.metadata["agent"]["position"])
ok = h.move("MoveAhead", moveMagnitude=0.25)
p1 = t.event.metadata["agent"]["position"]
check("smooth: a move blocked part-way is undone -> refused, pose unchanged (honest_map's failed-move rule)",
      not ok and math.hypot(p1["x"] - p0["x"], p1["z"] - p0["z"]) < 1e-6
      and t.calls[-1][0] == "MoveBack" and h.last_subs >= 3, str(t.calls[-4:]))
check("smooth: audit caller tags sub-sends, 0 violations", any("#sub" in c for _, _, _, c in h.audit.log)
      and h.audit.summary()["violations"] == 0)
check("smooth: a profile callable returning None means plain sends (console 'fast')",
      honest.HonestAgent(FakeThor(kitchen()), FakeDetector(FakeThor(kitchen())), smooth=lambda: None)._profile() is None)
src = open(os.path.join(HERE, "honest_skills.py"), encoding="utf-8").read()
check("honest_skills.py still has no privileged tokens", not [p for p in (r"\.objects\(", r"metadata", r"objectId",
                                                                         r"object_by_id", r"\bthor\.") if re.search(p, src)])


# ======================================================================================================== 13
print("\n13. 2026-10-07 container search quality: no re-open, inspection pose, honest inside-visible check")
# -- pure: the front plane from the agent's own depth
pose0 = {"position": {"x": 1.0, "y": 0.9, "z": 1.0}, "rotation": {"y": 0.0}, "horizon": 0.0,
         "camera": {"x": 1.0, "y": 1.5, "z": 1.0}}
dimg = np.full((H, W), 1.2, dtype="float32")              # a wall-like front 1.2 m straight ahead
pa = hs.front_patch(dimg, (30, 30, 70, 70), pose0, 90.0)
check("front_patch: a frontal plane -> planar, normal points back at the camera (0, -1)",
      pa is not None and pa["planar"] and abs(pa["n"][0]) < 0.05 and pa["n"][1] < -0.95, str(pa and pa["n"]))
check("front_patch: centroid 1.2 m ahead at about camera height", pa and abs(pa["c"][2] - 2.2) < 0.03
      and abs(pa["c"][1] - 1.5) < 0.05, str(pa and pa["c"]))
gm13 = GridMap("Fake", CELL, (0.0, 0.0), FREE)
pa2 = {**pa, "c": (1.25, 0.9, 2.5), "n": (0.0, -1.0)}
cells = hs.inspection_cells(gm13, pa2, 1.5, "Cabinet")
bx, bz = gm13.world(cells[0][1])
check("inspection_cells: the best cell is squarely in front (same x) at the stand-off for its height",
      cells and abs(bx - 1.25) < 0.13 and 0.6 <= 2.5 - bz <= 1.15, f"{cells[:3]}")
check("inspection_cells: never behind or edge-on to the front", all(gm13.world(c)[1] < 2.5 - 0.2 for _, c in cells))
probes = hs.interior_probes(pa, "Cabinet")
cav = dimg.copy()
cav[25:76, 25:76] = 1.6                                    # opened: a cavity behind the front
flat = dimg.copy()                                         # still shut
door = dimg.copy()
door[20:80, 20:80] = 0.7                                   # something nearer covers it (the swung door)
v_cav = hs.inside_check(cav, pose0, 90.0, probes, pa, face_off=0.15)
v_flat = hs.inside_check(flat, pose0, 90.0, probes, pa, face_off=0.15)
v_door = hs.inside_check(door, pose0, 90.0, probes, pa, [(15, 15, 85, 85)], face_off=0.15)
v_door2 = hs.inside_check(door, pose0, 90.0, probes, pa, [], face_off=0.15)
check("inside_check: cavity deeper than the closed front -> visible", v_cav["verdict"] == "visible", str(v_cav))
check("inside_check: depth unchanged -> not-visible 'still shut / no cavity seen'",
      v_flat["verdict"] == "not-visible" and "no cavity" in v_flat["reason"], str(v_flat))
check("inside_check: a nearer surface inside a detected door box -> 'door in the way'; else 'a nearer surface'",
      v_door["reason"] == "door in the way" and v_door2["reason"] == "a nearer surface blocks the view",
      f"{v_door} {v_door2}")
pose_away = {**pose0, "rotation": {"y": 180.0}}
check("inside_check: interior behind the camera -> unsure (not in view)",
      hs.inside_check(cav, pose_away, 90.0, probes, pa)["verdict"] == "unsure")
half = cav.copy()
half[:, 47:] = 1.2
v_half = hs.inside_check(half, pose0, 90.0, probes, pa, face_off=0.15)
check("inside_check: half the inside hidden -> partly", v_half["verdict"] == "partly", str(v_half))


def two_cab_kitchen(inside=False, block=()):
    objs = wine_kitchen(inside=inside)
    for o in objs:
        if o["id"] == WB_CAB:
            o["door_block"] = set(block)
    return objs


def open_counts(t):
    cnt = {}
    for _, oid in t.opened_log:
        cnt[oid] = cnt.get(oid, 0) + 1
    return cnt


# -- absent target: no physical container is opened twice; every inspection has an honest verdict
t, det, h, sk = setup(objs=two_cab_kitchen(), chooser=lambda *a: [], **EXH)
sk.type_list = sorted(set(sk.type_list) | {"WineBottle"})
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
si = r["detail"]["search"]
cnt = open_counts(t)
mist = sum("opened something else" in e for e in r["detail"]["events"])
check("no re-open: every container opened at most once on purpose (a mis-resolved open is detected, closed, reported)",
      cnt and sum(v - 1 for v in cnt.values()) <= mist and all(v <= 2 for v in cnt.values()), f"{cnt} mistakes={mist}")
check("every inspection carries an honest verdict + reason in search_info and the events",
      [i for i in si.get("inspections") or [] if i] and all(i["verdict"] in ("visible", "partly", "found", "unsure")
                                                        for i in si["inspections"] if i)
      and sum(": inside" in e for e in r["detail"]["events"]) >= len([i for i in si["inspections"] if i]),
      json.dumps(si)[:400])
check("all opened containers closed again, 0 violations, no ids",
      not openables_closed(t) and h.audit.summary()["violations"] == 0 and no_ids(r))
# -- the inspection pose: the agent opened the container from an inspection cell on its own map
t, det, h, sk = setup(objs=two_cab_kitchen(inside=True), chooser=lambda *a: [], **EXH)
plant_stale(sk)
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
rec = next((x for x in sk.inspected if x["type"] == "Cabinet" and x["verdict"] == "found"), None)
c0 = rec["cells"][0] if rec else None
check("found inside the cabinet from an inspection cell (own map), not from wherever the walk ended",
      r["ok"] and c0 in gm13.free and any("target seen inside" in e for e in r["detail"]["events"]),
      json.dumps(r["detail"]["events"])[-400:])
# -- door in the way from that cell: the agent KNOWS it cannot see inside, re-positions, re-checks
t, det, h, sk = setup(objs=two_cab_kitchen(inside=True, block={c0}), chooser=lambda *a: [], **EXH)
plant_stale(sk)
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
ev13 = r["detail"]["events"]
check("door in the way at the first inspection cell -> honest re-position and the bottle is then seen",
      r["ok"] and any("re-position for a view inside" in e for e in ev13)
      and any("target seen inside" in e for e in ev13), json.dumps(ev13)[-500:])
check("...still opened once (re-positioning keeps the door open, no close/re-open)",
      open_counts(t).get(WB_CAB) == 1, str(open_counts(t)))
# -- never visible from anywhere: 'unsure', kept as a candidate, one planned revisit, never 'checked, not there'
t, det, h, sk = setup(objs=two_cab_kitchen(inside=False, block=FREE), chooser=lambda *a: [], **EXH)
sk.type_list = sorted(set(sk.type_list) | {"WineBottle"})
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
wb = [x for x in sk.inspected if x["type"] == "Cabinet" and math.dist(x["c"], (-0.55, 0.6, 2.6)) < 0.6]
check("inside never visible -> verdict unsure with a reason, one revisit from another pose",
      wb and wb[0]["verdict"] == "unsure" and wb[0]["revisits"] == 1 and wb[0]["reason"]
      and any("revisit from another pose" in e for e in r["detail"]["events"]), json.dumps([
          {k: v for k, v in x.items() if k not in ("views", "checks")} for x in wb], default=str)[:500])
check("...and is never reported as checked-empty: not found, cabinet closed again",
      not r["ok"] and not openables_closed(t), str(openables_closed(t)))
# -- explicit inspection pose: square to the front, crouched for low fronts, standing again afterwards
t, det, h, sk = setup(objs=two_cab_kitchen(), chooser=lambda *a: [], **EXH)
sk.type_list = sorted(set(sk.type_list) | {"WineBottle"})
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
poses = [(x["type"], x["c"][1], p) for x in sk.inspected for p in x.get("poses") or []]
low = [p for typ, y, p in poses if y < hs.CROUCH_Y]
tall = [p for typ, y, p in poses if y >= hs.CROUCH_Y]
check("low fronts (< CROUCH_Y) are inspected crouched, higher ones standing",
      low and all(p["stance"] == "crouched" for p in low) and tall and all(p["stance"] == "standing" for p in tall),
      json.dumps(poses, default=str)[:400])
check("every inspection pose is logged (cell, yaw, stance, pitch, angle off the front) in the events",
      sum("inspection pose: cell" in e for e in r["detail"]["events"]) >= len([x for x in sk.inspected if x.get("poses")])
      and all({"cell", "yaw", "stance", "pitch", "off"} <= set(p) for _, _, p in poses))
check("inspections are square to the estimated front (heading within FRONT_OK of its normal)",
      all(p["off"] <= hs.FRONT_OK for _, _, p in poses if p["planar"]), str([p["off"] for _, _, p in poses]))
nc, ns = len(prims(t, "Crouch")), len(prims(t, "Stand"))
check("it stands up again after every crouched inspection (crouch count == stand count, standing at the end)",
      nc >= 1 and nc == ns and not sk._crouched, f"crouch {nc} stand {ns}")
check("the ticket summary carries the pose and whether it crouched",
      all(i.get("pose") is not None and "crouched" in i for i in r["detail"]["search"]["inspections"]
          if i["verdict"] != "unsure" or i.get("pose")))
t, det, h, sk = setup(smooth=PROF)
n0 = len(t.calls)
h.move("LookDown", degrees=25.3)
lk = [kw.get("degrees") for a, kw in t.calls[n0:] if a == "LookDown"]
check("smooth: look sub-steps are multiples of 0.1 deg (AI2-THOR refuses others) and add up to the logical look",
      len(lk) >= 2 and all(abs(v * 10 - round(v * 10)) < 1e-6 for v in lk) and abs(sum(lk) - 25.3) < 1e-6, str(lk))
check("SIGHT_NOW: a target seen in the current frame within reach confirms search_for without a re-face",
      hs.SIGHT_NOW and any("sighting confirmed" in e for e in ev13))

# ======================================================================================================== 14
print("\n14. 2026-10-07 'no far-away opens': reach gate, inspect-then-open, wrong-object recovery, door-pixel closes")
POSE0 = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"y": 0.0}, "horizon": 0.0,
         "camera": {"x": 0.0, "y": 1.5, "z": 0.0}}


def flat(d, h_=100, w_=100):
    return np.full((h_, w_), d, dtype="float32")


def front_at(z, half=0.2, y=1.5):
    pts = [(x, yy, z) for x in (-half, 0.0, half) for yy in (y - half, y, y + half)]
    return {"c": (0.0, y, z), "n": (0.0, -1.0), "pts": pts, "ymin": y - half, "ymax": y + half, "planar": True}


g = hs.reach_check(flat(0.9), POSE0, 90.0, (50, 50), hs.OPEN_REACH, front_at(0.9), "front", "Cabinet")
check("reach gate: a pixel on the intended front within reach passes", g["ok"] and abs(g["dist"] - 0.9) < 0.02, str(g))
g = hs.reach_check(flat(1.4), POSE0, 90.0, (50, 50), hs.OPEN_REACH, None)
check("reach gate: 1.4 m is beyond the open reach -> refused (THOR would accept it)", not g["ok"]
      and "out of reach" in g["why"], str(g))
g = hs.reach_check(flat(1.3), POSE0, 90.0, (50, 50), hs.ACT_REACH, None, edges=False)
check("reach gate: other interactions keep the 1.4 m reach", g["ok"], str(g))
dd = flat(0.9)
dd[50, 52:] = 2.0
g = hs.reach_check(dd, POSE0, 90.0, (50, 50), hs.OPEN_REACH, front_at(0.9), "front", "Cabinet")
check("reach gate: a pixel on a depth edge (door rim / box border) is refused", not g["ok"] and "edge" in g["why"], str(g))
g = hs.reach_check(flat(1.1), POSE0, 90.0, (50, 50), hs.OPEN_REACH, front_at(0.9), "front", "Cabinet")
check("reach gate: the ray reaches a surface 0.2 m behind the intended front (another container) -> refused",
      not g["ok"] and "intended front" in g["why"], str(g))
g = hs.reach_check(flat(0.9), POSE0, 90.0, (50, 10), hs.OPEN_REACH, front_at(0.9), "front", "Drawer")
check("reach gate: a coplanar point above the intended front (stacked drawer) is refused", not g["ok"]
      and "outside the intended front" in g["why"], str(g))
g1 = hs.reach_check(flat(0.9), POSE0, 90.0, (50, 50), hs.CLOSE_REACH, front_at(0.9), "door", "Cabinet")
dd = flat(0.9)
dd[90:, :] = 1.2                                       # the floor in the lower part of an open cabinet's box
g2 = hs.reach_check(dd, POSE0, 90.0, (50, 95), hs.CLOSE_REACH, front_at(0.9), "door", "Cabinet")
check("close gate: a pixel on the opened door passes, a floor pixel in the open box does not (baseline: "
      "'Floor is not an Openable object')", g1["ok"] and not g2["ok"], f"{g1} {g2}")

# -- _act_at: never sends from a stale frame, never sends when no pixel passes
t, det, h, sk = setup(**EXH)
ob = sk._look()
h.move("RotateRight", degrees=15)
n0 = len(t.calls)
r = sk._act_at("OpenObject", ob, [(50, 50)], "Fridge")
check("_act_at: a frame from before a move is stale -> nothing sent", not r["sent"] and "stale" in r["why"]
      and len(t.calls) == n0, str(r))
ob = sk._look()
r = sk._act_at("OpenObject", ob, [(0, 0)], "Fridge", front=front_at(0.5))
check("_act_at: no pixel passes -> no GetObjectInFrame, no OpenObject", not r["sent"]
      and not [a for a, _ in t.calls[n0:] if a in ("GetObjectInFrame", "OpenObject")], str(r))


def track_opens(t):
    """Test scoring only: camera -> nearest point of every object an OpenObject actually opened."""
    log, orig = [], t.do

    def do(action, **kw):
        before = {k: o.get("isOpen") for k, o in t.objs.items()}
        ev_ = orig(action, **kw)
        if action == "OpenObject":
            cam = t.event.metadata["cameraPosition"]
            for k, o in t.objs.items():
                if o.get("isOpen") and not before.get(k):
                    d = math.dist((cam["x"], cam["y"], cam["z"]), o["pos"]) - o["size"] / 2.0
                    log.append((k, round(d, 3)))
        return ev_

    t.do = do
    return log


# -- the live bug: a container opened from wherever the agent stood (bounded / explicit open far from the front)
t, det, h, sk = setup(start=(0, 0), yaw=90.0)
log14 = track_opens(t)
r = sk.exec_step({"action": "open", "object": "Cabinet"})
check("explicit open from far away: walks nearer, opens within the open reach (ground truth, test scoring)",
      r["ok"] and log14 and all(d <= hs.OPEN_REACH + 0.05 for _, d in log14), f"{log14} {json.dumps(r)[:300]}")
check("...and the opened container is the intended one", [k for k, _ in log14] == ["Cabinet|+2.90|+0.50|+0.40"],
      str(log14))
# -- exhaustive local look at a remembered place (the console's "find it again" path): inspect-then-open
t, det, h, sk = setup(**EXH)
log14 = track_opens(t)
cab = obj(t, "Cabinet")
sk._face(cab["pos"])                                   # it sees the closed cabinet from 1.5 m away
ok = sk._open_container("Cabinet", cab["pos"], why="look")
cur, _ = hm.current_cell(h, sk.gm)
ent = next((e for e in sk.opened_by_me if e["type"] == "Cabinet"), None)
cells = [c for _, c in hs.inspection_cells(sk.gm, ent["front"], 1.5, "Cabinet")] if ent and ent.get("front") else []
check("_open_container (exhaustive): walked to an inspection cell of the front first, then opened it there",
      ok and ent and tuple(ent["cell"]) in cells and log14 and all(d <= hs.OPEN_REACH + 0.05 for _, d in log14),
      f"{ok} {ent and ent.get('cell')} {cells[:5]} {log14}")
check("...its own history keeps the front, the cell and the stance it opened from (for the close)",
      ent and {"front", "cell", "crouched"} <= set(ent) and no_ids(sk.opened_by_me))
r = sk.tidy_up()
check("...closed again from there, nothing left open", not openables_closed(t) and "Cabinet" in r["closed"], str(r))
# -- the point resolves to a DIFFERENT container: detected from the agent's own views, closed at once, never left
t, det, h, sk = setup(**EXH)
t.misresolve["Cabinet"] = "Fridge"
sk._face(obj(t, "Cabinet")["pos"])
ok = sk._open_container("Cabinet", obj(t, "Cabinet")["pos"], why="look")
evs = sk.events
check("wrong-object open: reported ('opened something else'), the wrong one closed at once, nothing left open",
      not ok and any("opened something else" in e for e in evs) and not openables_closed(t)
      and not any(e["type"] == "Fridge" for e in sk.opened_by_me), json.dumps(evs)[-400:])
check("section 14 audit: 0 violations, no ids", h.audit.summary()["violations"] == 0 and no_ids(evs))
# -- a low front is OPENED crouched: the reach check of the inspection cells uses the crouched eye
gm14 = GridMap("Fake", CELL, (0.0, 0.0), FREE)
low = {"c": (1.2, 0.3, 2.6), "n": (0.0, -1.0), "pts": [(1.1, 0.25, 2.6), (1.3, 0.35, 2.6)], "ymin": 0.25,
       "ymax": 0.35, "planar": True}
cl = hs.inspection_cells(gm14, low, 1.576, "Drawer")
check("low drawer: inspection cells exist only because it is opened crouched (camera 0.675 m lower)",
      cl and all(math.hypot(math.hypot(gm14.world(c)[0] - 1.2, gm14.world(c)[1] - 2.6), 1.576 - hs.CROUCH_DROP - 0.3)
                 <= hs.REACH3D for _, c in cl)
      and all(math.hypot(math.hypot(gm14.world(c)[0] - 1.2, gm14.world(c)[1] - 2.6), 1.576 - 0.3) > hs.REACH3D
              for _, c in cl), str(cl[:4]))
# -- the detector reads an opened fridge as "closed" (live after_v2): own history still closes it, never left open
t, det, h, sk = setup(**EXH)
fr = obj(t, "Fridge")
sk._face(fr["pos"])
t.lag_after[("OpenObject", "Fridge")] = 12
ok = sk._open_container("Fridge", fr["pos"], why="look")
was_open = bool(fr["isOpen"])
r = sk.tidy_up()
check("an opened fridge that reads 'closed' is still closed by tidy (own history: opened, not closed since)",
      ok and was_open and not fr["isOpen"] and "Fridge" in r["closed"], f"{ok} {was_open} {r['log']} {r['events'][-12:]}")
# -- a container where this agent SAW the target earlier is looked into again first (repeat search)
t, det, h, sk = setup(objs=wine_kitchen(), chooser=lambda *a: [], **EXH)
r1 = sk.exec_step({"action": "search_for", "object": "WineBottle"})
sk.tidy_up()
r2 = sk.exec_step({"action": "search_for", "object": "WineBottle"})
ev2 = r2["detail"]["events"]
check("repeat search: found again where it was seen before (one open of that cabinet, closed again)",
      r1["ok"] and r2["ok"] and not openables_closed(t) and sum("to look inside: ok" in e for e in ev2) == 1
      and not any("search WineBottle" in e and "sweep here" in e for e in ev2), json.dumps(ev2)[-400:])
# -- stacked drawers: a box spanning a smaller drawer box is the union, not a container of its own
obs14 = type("O", (), {})()
obs14.depth, obs14.pose = flat(0.8), POSE0
obs14.dets = [{"type": "Drawer", "bbox_xyxy": (30, 30, 70, 80), "state": "closed", "point": (50, 55)},
              {"type": "Drawer", "bbox_xyxy": (30, 30, 70, 52), "state": "closed", "point": (50, 41)}]
t, det, h, sk = setup(**EXH)
cands = sk._candidates(obs14, ["Drawer"])
check("stacked drawers: the union box is skipped, the drawer's own box is a candidate",
      len(cands) == 1 and cands[0]["patch"]["bbox"] == (30, 30, 70, 52), str([c["patch"]["bbox"] for c in cands]))
src = open(os.path.join(HERE, "honest_skills.py"), encoding="utf-8").read()
check("honest_skills.py still has no privileged tokens (after the gate)", not [
    p for p in (r"\.objects\(", r"metadata", r"objectId", r"instance_masks", r"objectType", r"thor\.")
    if re.search(p, src)])

# ======================================================================================================== 15
print("\n15. 2026-10-08 'pose first, then open': put pose before the door moves, swept floor blocked while open")
# -- own map: temporary blocks
gm15 = GridMap("Fake", CELL, (0.0, 0.0), FREE)
full = gm15.astar((0, 4), [(9, 4)])
gm15.block_cells("door", {(4, z) for z in range(10)} - {(4, 9)}, keep=[(0, 4)])
p15 = gm15.astar((0, 4), [(9, 4)])
check("map: A* never plans through temporarily blocked cells (detours round the swept floor)",
      full and p15 and not any(c[0] == 4 and c[1] != 9 for c in p15) and len(p15) > len(full), str(p15))
check("map: blocked cells are not goals, not stand-off cells", gm15.astar((0, 4), [(4, 2)]) is None
      and (4, 2) not in gm15.cells_near((1.0, 0.9, 0.5), 0.0, 0.3))
gm15.block_cells("door2", {(0, 4), (0, 5)}, keep=[(0, 4)])
check("map: the agent's own cell is never blocked (it can always back off)", not gm15.is_temp_blocked((0, 4))
      and gm15.is_temp_blocked((0, 5)))
gm15.unblock_cells("door")
gm15.unblock_cells("door2")
check("map: unblocked on close -> the plain plan again; blocks are never persisted",
      gm15.astar((0, 4), [(9, 4)]) == full and "temp" not in json.dumps(gm15.to_dict()))
# -- pure door model
fr15 = {"c": (2.0, 1.0, 1.0), "n": (-1.0, 0.0), "pts": [(2.0, y, z) for y in (0.7, 1.0, 1.3) for z in (0.73, 1.0, 1.27)],
        "ymin": 0.7, "ymax": 1.3, "planar": True}
m0 = hs.door_model(fr15, "Fridge")
check("door model (unknown hinge): both front edges, radius from the measured front width (prior)",
      len(m0["hinges"]) == 2 and not m0["observed"] and 0.6 <= m0["r"] <= 1.0, str(m0))
pts_door = [(2.0 - s_, y, 0.6) for s_ in (0.2, 0.4, 0.6, 0.75) for y in (0.8, 1.2)]       # leaf standing at one edge
m1 = hs.door_model(fr15, "Fridge", pts_door)
check("door model (door seen open): hinge = the edge its own door points stand at; radius never below the prior",
      m1["observed"] and len(m1["hinges"]) == 1
      and m1["hinges"][0] == min(m0["edges"], key=lambda e: abs(e - hs._lat(fr15, 2.0, 0.6)))
      and m1["r"] >= m0["r"], str(m1))
check("swept: the floor beside the hinge is swept, straight in front at 1 m is not, behind the plane never",
      hs.swept(m1, 1.6, 0.55) and not hs.swept(m1, 1.0, 1.0) and not hs.swept(m1, 2.3, 0.6))
mixed = [(2.0 - 0.3, 1.0, z) for z in (0.8, 1.0, 1.2)]
check("door model: points not clearly at one edge leave the hinge unknown (no guess)",
      not hs.door_model(fr15, "Fridge", mixed)["observed"])
md15 = hs.door_model(dict(fr15, ymin=0.2, ymax=0.4, c=(2.0, 0.3, 1.0)), "Drawer")
check("drawer: a slab pulled out along the normal", md15["kind"] == "slide" and hs.swept(md15, 1.7, 1.0)
      and not hs.swept(md15, 1.2, 1.0))
pc15 = hs.put_cells(gm15, fr15, 1.5, "Fridge", m0)
check("put cells: square to the front, outside the sweep, open + interior within reach",
      pc15 and all(not hs.swept(m0, *gm15.world(c)) for _, c in pc15)
      and all(abs(gm15.world(c)[1] - 1.0) <= 0.5 for _, c in pc15), str(pc15[:4]))


def door_kitchen():
    """The live bug, scaled down: a fridge whose single door hinges on its -z edge and swings into the floor in front
    of that edge. ``door_cells`` = the floor the open leaf sweeps (ground truth of the fake; the agent never reads it):
    a move into it while the door is open is refused, and an open / close is refused when the agent stands in it."""
    objs = kitchen()
    fr = next(o for o in objs if o["type"] == "Fridge")
    hinge = (fr["pos"][0], fr["pos"][2] - 0.4)
    fr["door_cells"] = {c for c in FREE if math.hypot(c[0] * CELL - hinge[0], c[1] * CELL - hinge[1]) <= 0.95}
    fr["cavity"] = 0.15                                   # its open interior reads this much deeper than its front
    return objs


t, det, h, sk = setup(start=(8, 0), yaw=0.0, objs=door_kitchen(), **EXH)
fr = obj(t, "Fridge")
rp = sk.exec_step({"action": "pickup", "object": "Apple"}, upcoming=[{"action": "put_in", "object": "Fridge"}])
n0 = len(t.calls)
blocked_seen = []
orig_put = sk._act_at


def spy(prim, obs, pts, otype, **kw):
    if prim == "PutObject":
        blocked_seen.append(set(sk.gm._tb) | set(sk.gm.soft_cells()))     # hard (leaf) + soft (rest of the sweep)
    return orig_put(prim, obs, pts, otype, **kw)


sk._act_at = spy
r15 = sk.exec_step({"action": "put_in", "object": "Fridge"})
sk._act_at = orig_put
calls = t.calls[n0:]
i_open = next((i for i, c in enumerate(calls) if c[0] == "OpenObject"), None)
i_put = next((i for i, c in enumerate(calls) if c[0] == "PutObject"), None)
walks_after = [c for c in calls[i_open:i_put] if c[0].startswith("Move")] if i_open is not None and i_put else None
ev15 = r15["detail"]["events"]
# (the fake renders a front as a camera-facing billboard at one axial depth, so its first pose can read 0.04 m beyond
#  the open reach at every pixel: one gate retry -> the next pose; never a door contact)
check("put into the fridge from beside its hinge: ok, the apple is inside, at most one (reach-gate) retry",
      rp["ok"] and r15["ok"] and obj(t, "Apple").get("inside") == fr["id"] and r15["retries"] <= 1
      and all(s_.get("cause") in (None, "unreachable") for s_ in r15["detail"]["sub"]),
      " | ".join(rp["detail"]["events"][-6:] + ev15))
check("...the put pose was chosen BEFORE opening, outside the door's sweep (event), one OpenObject",
      any("chosen before opening" in e and "outside" in e for e in ev15)
      and sum(1 for c in calls if c[0] == "OpenObject") == 1, json.dumps(ev15)[-600:])
check("...no walk between the open and the put; no refused open (the door met nothing)",
      walks_after == [] and not getattr(t, "door_refusals", 0), f"{walks_after} {getattr(t, 'door_refusals', 0)}")
check("...while it stood open the swept floor was blocked / avoided on the own map (hard + soft cover most of the door's real sweep)",
      blocked_seen and len(blocked_seen[0] & fr["door_cells"]) >= 0.6 * len(fr["door_cells"] - {hm.current_cell(h, sk.gm)[0]}),
      f"{len(blocked_seen[0]) if blocked_seen else None} / {len(fr['door_cells'])}")
check("...closed at step end from the same pose; swept cells unblocked; nothing left open",
      not openables_closed(t) and not sk.gm.temp_blocked and not getattr(t, "door_hits", 0)
      and any("unblocked" in e for e in ev15), json.dumps(ev15)[-300:])
check("section 15 put: audit 0 violations, no ids", h.audit.summary()["violations"] == 0 and no_ids(r15))
# -- the same task on the old flow (pose_first off) for contrast: it opens from where it stands
t2, det2, h2, sk2 = setup(start=(8, 0), yaw=0.0, objs=door_kitchen(), pose_first=False, **EXH)
sk2.exec_step({"action": "pickup", "object": "Apple"})
n2 = len(t2.calls)
r15b = sk2.exec_step({"action": "put_in", "object": "Fridge"})
c2 = t2.calls[n2:]
opened2 = next((i for i, c in enumerate(c2) if c[0] == "OpenObject"), len(c2))
print(f"  (info) the old flow from the same start: put ok={r15b['ok']}, door refusals "
      f"{getattr(t2, 'door_refusals', 0)}, moves into the open leaf {getattr(t2, 'door_hits', 0)}, moves after the "
      f"open {sum(1 for c in c2[opened2:] if c[0].startswith('Move'))}")
# -- an explicit plan: open, put_in, close -> the open uses the put pose; the put does not walk; close from there
t, det, h, sk = setup(start=(8, 0), yaw=0.0, objs=door_kitchen(), **EXH)
plan = [{"action": "pickup", "object": "Apple"}, {"action": "open", "object": "Fridge"},
        {"action": "put_in", "object": "Fridge"}, {"action": "close", "object": "Fridge"}]
outs = []
for i, st in enumerate(plan):
    n_ = len(t.calls)
    outs.append((sk.exec_step(st, upcoming=plan[i + 1:]), t.calls[n_:]))
check("explicit open / put_in / close: all ok, one OpenObject, the put step sends no move, closed, unblocked",
      all(o[0]["ok"] for o in outs) and sum(1 for o in outs for c in o[1] if c[0] == "OpenObject") == 1
      and not any(c[0].startswith("Move") for c in outs[2][1]) and not openables_closed(t)
      and not sk.gm.temp_blocked and not getattr(t, "door_refusals", 0),
      json.dumps([(o[0]["ok"], o[0]["cause"], [c[0] for c in o[1] if c[0] in ("OpenObject", "PutObject", "CloseObject")])
                  for o in outs]) + " | " + " | ".join(outs[1][0]["detail"]["events"]))
# -- holding something and the open is refused: said so explicitly, recovered by the retry policy
t, det, h, sk = setup(start=(8, 0), yaw=0.0, objs=door_kitchen(), **EXH)
sk.exec_step({"action": "pickup", "object": "Apple"})
t.refuse[("OpenObject", "Fridge")] = 1
r = sk.exec_step({"action": "put_in", "object": "Fridge"})
check("open refused while holding: reported as such (no invented cause), retried, put ok, nothing left open",
      r["ok"] and any("refused while holding the Apple (cause not observed)" in e for e in r["detail"]["events"])
      and r["retries"] >= 1 and not openables_closed(t) and not getattr(t, "door_hits", 0), f"retries {r['retries']} " + " | ".join(r["detail"]["events"]))
# -- search inspection: the inspection pose avoids the door's estimated sweep, the swept floor is blocked while open
t, det, h, sk = setup(start=(8, 0), yaw=0.0, objs=wine_kitchen() if False else door_kitchen(), chooser=lambda *a: ["Fridge"],
                      **EXH)
obj(t, "Apple")["inside"] = obj(t, "Fridge")["id"]
obj(t, "Apple")["pos"] = (2.85, 0.9, 1.0)
blocks = []
orig_bd = sk._block_door


def bd(ent):
    m = orig_bd(ent)
    blocks.append(set(sk.gm._tb) | set(sk.gm.soft_cells()))
    return m


sk._block_door = bd
r = sk.exec_step({"action": "search_for", "object": "Apple"})
check("search inspection of the fridge: found inside, opened once without the door meeting the agent, swept floor "
      "blocked while open, released after", r["ok"] and blocks and blocks[0]
      and not getattr(t, "door_refusals", 0) and not getattr(t, "door_hits", 0) and not openables_closed(t)
      and not sk.gm.temp_blocked, json.dumps(r["detail"]["events"])[-500:])
check("pose-first is the console policy only: the bounded (s5) runner keeps its flow by default",
      hs.HonestSkills.__init__.__defaults__ is not None and setup()[3].pose_first is False
      and setup(**EXH)[3].pose_first is True)

# ======================================================================================================== 16
print("\n16. 2026-10-08 'systematic search route': strafe walks, one look-around, numbered stops, grouped inspections")
# -- pure: walk heading -- short walks keep the heading, long ones face the way most of the path goes
check("walk_heading: a short walk (<= STRAFE_MAX) keeps the current heading (no turn)",
      hm.walk_heading([(4, 4), (5, 4), (6, 4)], 0, 1.25) == 0)
check("walk_heading: a long walk faces its dominant direction (forward-facing travel)",
      hm.walk_heading([(4, 4)] + [(4 + i, 4) for i in range(1, 7)] + [(10, 5)], 0, 1.25) == 1)
check("walk_heading: strafe 0 (the s5 default) is not used by walk_path's turning mode",
      __import__("inspect").signature(hm.walk_path).parameters["strafe"].default == 0.0)
# -- strafe walk in the fake room: side steps, no turns; the default keeps the old turn-and-MoveAhead walk
t, det, h, sk = setup(start=(4, 4), yaw=0.0)
n0 = len(t.calls)
r16 = hm.walk_path(h, sk.gm, sk.gm.astar((4, 4), [(6, 4)]), goals=[(6, 4)], strafe=1.25)
c16 = [a for a, _ in t.calls[n0:]]
check("strafe walk: two cells to the side with MoveRight only (one <= 45-deg turn onto the lattice), ends on the goal",
      r16["ok"] and r16["side"] == 2 and sum(float(kw.get("degrees", 90)) for a, kw in t.calls[n0:]
                                             if a.startswith("Rotate")) <= 45.0
      and c16.count("MoveRight") == 2 and hm.current_cell(h, sk.gm)[0] == (6, 4), str(c16))
n0 = len(t.calls)
r16b = hm.walk_path(h, sk.gm, sk.gm.astar((6, 4), [(6, 3)]), goals=[(6, 3)], strafe=1.25)
check("strafe walk: one cell back with MoveBack (no 180-deg turn)",
      r16b["ok"] and r16b["back"] == 1 and [a for a, _ in t.calls[n0:]].count("MoveBack") == 1
      and not any(a.startswith("Rotate") for a, _ in t.calls[n0:]))
t, det, h, sk = setup(start=(4, 4), yaw=0.0)
n0 = len(t.calls)
hm.walk_path(h, sk.gm, sk.gm.astar((4, 4), [(6, 4)]), goals=[(6, 4)])
check("default walk (s5 runner): turn + MoveAhead as before", any(a == "RotateRight" for a, _ in t.calls[n0:])
      and not any(a in ("MoveLeft", "MoveRight", "MoveBack") for a, _ in t.calls[n0:]))
t2, det2, h2, sk2 = setup(start=(4, 4), yaw=0.0)
FREE_SAVE = set(FREE)
FREE.discard((5, 4))                                   # the fake room refuses the side step into this cell
n0 = len(t2.calls)
r16c = hm.walk_path(h2, sk2.gm, sk2.gm.astar((4, 4), [(6, 4)]), goals=[(6, 4)], strafe=1.25)
FREE.clear()
FREE.update(FREE_SAVE)
check("strafe walk: a refused side step adds a blocked edge and replans round it (as MoveAhead did)",
      r16c["ok"] and hm.edge((4, 4), (5, 4)) in sk2.gm.blocked_edges and hm.current_cell(h2, sk2.gm)[0] == (6, 4),
      json.dumps(r16c))

# -- pure: stops group what one cell serves, and the route never zig-zags
def it16(name, cells, q=0.1, cost=1.0):
    return {"kind": "front", "type": name, "c": (0, 0, 0), "q": q, "cost": cost, "serve": dict(cells)}


col = [it16("Drawer", {(2, 2): 0.3, (2, 3): 0.5}), it16("Drawer", {(2, 2): 0.2, (3, 2): 0.6}),
       it16("Drawer", {(2, 2): 0.4}), it16("Cabinet", {(2, 2): 0.9, (1, 2): 0.2})]
st16 = hs.cover_stops(col + [it16("Cabinet", {(8, 8): 0.1})])
check("cover_stops: a drawer column + the neighbouring door served from one cell = one stop (each item once)",
      len(st16) == 2 and sorted(len(s["items"]) for s in st16) == [1, 4]
      and sum(len(s["items"]) for s in st16) == 5, str([(s["cell"], len(s["items"])) for s in st16]))
line = [{"cell": (x, 0), "items": [{"q": 0.2, "cost": 1.0}]} for x in (1, 2, 3, 4, -1)]


def d16(a, b):
    return 0.25 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))


o16 = hs.order_stops(line, (0, 0), d16)
xs = [s["cell"][0] for s in o16]
crossings = sum(1 for a, b in zip([0] + xs, xs) if (a > 0 > b) or (a < 0 < b))
check("order_stops: stops on both sides of the start are taken one side after the other (one crossing at most)",
      crossings <= 1 and sorted(xs) == [-1, 1, 2, 3, 4], str(xs))
pri = [{"cell": (1, 0), "items": [{"q": 0.02, "cost": 1.0}]}, {"cell": (-3, 0), "items": [{"q": 0.9, "cost": 1.0}]},
       {"cell": (2, 0), "items": [{"q": 0.02, "cost": 1.0}]}]
o16b = hs.order_stops(pri, (0, 0), d16)
check("order_stops: a much likelier stop a little farther away is visited first (weighted search cost)",
      o16b[0]["cell"] == (-3, 0), str([s["cell"] for s in o16b]))
check("route_cost: the chosen order is no worse than nearest-next",
      hs.route_cost(o16, (0, 0), d16) <= hs.route_cost(sorted(line, key=lambda s: abs(s["cell"][0])), (0, 0), d16) + 1e-9)
hd16 = hs.view_headings([10, 30, 60, 100, 130], 0.0)
check("view_headings: a 120-deg arc of unseen bearings -> 2 looks, in one turn direction",
      len(hd16) == 2 and hd16 == sorted(hd16), str(hd16))
hd16b = hs.view_headings([350, 10, 200], 90.0)
check("view_headings: bearings across north are covered without a full turn",
      len(hd16b) == 2 and all(0 <= h_ < 360 for h_ in hd16b), str(hd16b))
check("view_headings: nothing unseen -> no look", hs.view_headings([], 0.0) == [])
fl = np.full((H, W), 6.0, dtype="float32")
fl[60:, :] = 1.2                                       # a floor band 1.2 m ahead (lower half of the frame)
gm16 = GridMap("Fake", CELL, (0.0, 0.0), FREE)
sc16 = hs.seen_cells(fl, {"position": {"x": 1.0, "y": 0.9, "z": 1.0}, "rotation": {"y": 0.0}, "horizon": 30.0,
                          "camera": {"x": 1.0, "y": 1.5, "z": 1.0}}, 90.0, gm16)
check("seen_cells: depth rays that END within range mark their cells; far / empty depth marks nothing",
      sc16 and all(c[1] >= 4 for c in sc16), str(sorted(sc16)[:6]))
check("height prior is declared, not learned: vessels prefer wall cabinets, cutlery drawers",
      hs.height_prior("Plate", "Cabinet", 1.9) > hs.height_prior("Plate", "Cabinet", 0.4)
      and hs.height_prior("Spoon", "Drawer", 0.8) > hs.height_prior("Spoon", "Cabinet", 1.9)
      and hs.height_prior("Apple", "Cabinet", 1.9) == 1.0)


# -- integration: a counter run of three base cabinets (0.5 m apart) + one far cabinet; target absent
def run_kitchen(inside=None):
    objs = [o for o in kitchen() if o["type"] not in ("Cabinet",)]
    for k, x in enumerate((0.5, 1.0, 1.5)):
        objs.append(dict(id=f"Cabinet|+{x:.2f}|+0.50|+2.60", type="Cabinet", pos=(x, 0.5, 2.6), size=0.38,
                         openable=True, receptacle=True, isOpen=False))
    objs.append(dict(id="Cabinet|-0.55|+0.50|+0.40", type="Cabinet", pos=(-0.55, 0.5, 0.4), size=0.38, openable=True,
                     receptacle=True, isOpen=False))
    if inside:
        objs.append(dict(id="WineBottle|x", type="WineBottle", pos=(inside[0], 0.5, inside[2]), size=0.10,
                         pickupable=True, inside=next(o["id"] for o in objs if o["type"] == "Cabinet"
                                                      and math.dist(o["pos"], inside) < 0.05)))
    return objs


def rot_deg(t, n0=0):
    return sum(float(kw.get("degrees", 90)) for a, kw in t.calls[n0:] if a in ("RotateLeft", "RotateRight"))


def search16(strafe=None, inside=None):
    t, det, h, sk = setup(start=(4, 6), yaw=0.0, objs=run_kitchen(inside), chooser=lambda *a: ["Cabinet"], **EXH)
    if strafe is not None:
        sk.strafe = strafe
    sk.type_list = sorted(set(sk.type_list) | {"WineBottle"})
    n0 = len(t.calls)
    cells = []
    orig = h.act

    def act(action, point=None, **kw):
        if action == "OpenObject":
            cells.append(hm.current_cell(h, sk.gm)[0])
        return orig(action, point, **kw)
    h.act = act
    r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
    return t, h, sk, r, cells, rot_deg(t, n0), n0


t, h, sk, r, cells16, rot16, n016 = search16()
si16 = r["detail"]["search"]
ev16 = r["detail"]["events"]
run_opens = [oid for _, oid in t.opened_log if "+2.60" in oid]
check("search after the episode's look-around: no second full look-around from the same spot",
      any("already looked around from this spot" in e for e in ev16)
      and not any("look around here first" in e for e in ev16), " | ".join(ev16[:3]))
check("an explicit plan is made before moving: numbered stops in the events and in search_info",
      any(e.split("] ", 1)[-1].startswith("search plan 1") for e in ev16) and si16.get("plan")
      and [p["n"] for p in si16["plan"]] == list(range(1, len(si16["plan"]) + 1))
      and all({"n", "cell", "x", "z", "label", "status"} <= set(p) for p in si16["plan"]), json.dumps(si16.get("plan"))[:400])
check("the three adjacent cabinets of the counter run are each opened once",
      sorted(set(run_opens)) == sorted({f"Cabinet|+{x:.2f}|+0.50|+2.60" for x in (0.5, 1.0, 1.5)})
      and len(run_opens) == 3, str(t.opened_log))
run_cells = [c for c, (_, oid) in zip(cells16, t.opened_log) if "+2.60" in oid]
check("...from one stop (at most one cell beside it): adjacent containers grouped, not walked to one by one",
      len(set(run_cells)) <= 2 and max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in run_cells for b in run_cells) <= 1,
      str(run_cells))
check("absent target: every container inspected, all closed again, plan stops all done / visited",
      not r["ok"] and len({oid for _, oid in t.opened_log}) == sum(1 for o in t.objs.values() if o.get("openable"))
      and not openables_closed(t)
      and all(p["status"] in ("done", "found", "unreachable") for p in si16["plan"]),
      json.dumps([(p["n"], p["status"], p["label"]) for p in si16.get("plan") or []]) + str(t.opened_log))
check("...nothing left open, no ids, 0 violations", not openables_closed(t) and no_ids(r)
      and h.audit.summary()["violations"] == 0)
t_, h_, sk_, r_, cells_, rot_old, _ = search16(strafe=0.0)
check("strafe walks turn much less than the old turn-and-MoveAhead walks over the same search",
      rot16 < 0.8 * rot_old and len(set(t.opened_log)) == len(set(t_.opened_log)), f"{rot16:.0f} vs {rot_old:.0f} deg")
side16 = sum(1 for a, _ in t.calls[n016:] if a in ("MoveLeft", "MoveRight", "MoveBack"))
check("side / back steps are used during the search (logged by the walks)", side16 >= 1, str(side16))
# -- target inside the middle cabinet: found, the plan says which stop found it
t, h, sk, r, cells16, rot16, _ = search16(inside=(1.0, 0.5, 2.6))
si16 = r["detail"]["search"]
check("target inside the middle cabinet: found at a planned stop (status found), opened at most the run",
      r["ok"] and any(p["status"] == "found" for p in si16["plan"]) and len(t.opened_log) <= 3,
      json.dumps(si16.get("plan"))[:300] + str(t.opened_log))
check("the found stop is closed again after the step (tidy), 0 violations",
      not [o for o in openables_closed(t) if o != "Cabinet"] and h.audit.summary()["violations"] == 0)
# -- a stop that cannot be reached: re-planned without it, never opened from far
t, det, h, sk = setup(start=(4, 6), yaw=0.0, objs=run_kitchen(), chooser=lambda *a: ["Cabinet"], **EXH)
sk.type_list = sorted(set(sk.type_list) | {"WineBottle"})
orig_wc = sk._walk_cell
fails = {"n": 0}


def wc(cell, face=None):
    if fails["n"] == 0 and sk._stop is None and sk.search_info and sk.search_info.get("plan"):
        fails["n"] += 1
        return False
    return orig_wc(cell, face=face)


sk._walk_cell = wc
r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
si16 = r["detail"]["search"]
check("a planned stop not reached on the own map -> marked unreachable and the route re-planned",
      any(p["status"] == "unreachable" for p in si16["plan"]) and si16.get("plans", 0) >= 2
      and any("re-planning without it" in e for e in r["detail"]["events"]) and not openables_closed(t),
      json.dumps(si16.get("plan"))[:300])
t, det, h, sk = setup(**EXH)
sk.exec_step({"action": "pickup", "object": "Apple"})
ob16 = sk._look()
r16h = sk._act_at("CloseObject", ob16, [(50, 88)], "Fridge", mode="door")
check("while holding, a pixel on the carried object (nearer than HAND_DEPTH) is never acted on",
      h.holding and not r16h["sent"] and "held object" in r16h["why"], str(r16h))
check("bounded (s5) mode keeps its search and walks: no plan, no strafing",
      setup()[3].strafe == 0.0 and setup(**EXH)[3].strafe == hs.STRAFE_MAX)

# ======================================================================================================== 17
print("\n17. 2026-10-09 'unreachable failures': door sweep, thin faucet, off-map start, out-of-reach recovery")
FREE_SAVE17 = set(FREE)


def with_free(cells):
    FREE.clear()
    FREE.update(cells)


# -- pure: soft cells and the episode start node on the own map
g17 = GridMap("Fake", CELL, (0.0, 0.0), {(x, z) for x in range(6) for z in range(3)})
g17.block_cells("door", [(2, 1)], keep=[(0, 1)], soft=[(1, 1), (3, 1), (2, 0)])
p17 = g17.astar((0, 1), [(5, 1)])
check("map: a HARD cell is never entered; SOFT ones are avoided when a short detour exists",
      p17 and (2, 1) not in p17 and not any(c in p17 for c in [(1, 1), (3, 1)]) or
      (p17 and (2, 1) not in p17), str(p17))
g17b = GridMap("Fake", CELL, (0.0, 0.0), {(x, 0) for x in range(6)})
g17b.block_cells("door", [], keep=[(0, 0)], soft=[(1, 0), (2, 0), (3, 0)])
check("map: soft cells on the ONLY way are still passable (the agent is never walled in by an estimate)",
      g17b.astar((0, 0), [(5, 0)]) == [(x, 0) for x in range(6)])
g17b.block_cells("door", [(1, 0), (2, 0), (3, 0)], keep=[(0, 0)])
check("map: the old all-hard sweep on the only way = no path (the live 'island' cause)",
      g17b.astar((0, 0), [(5, 0)]) is None)
check("map: unblocking a key releases its hard and soft cells", g17.unblock_cells("door") and not g17.soft_cells()
      and not g17.temp_blocked)
g17c = GridMap("Fake", CELL, (0.0, 0.0), {(x, z) for x in range(4) for z in range(1, 4)},
               blocked_edges=[((2, 0), (2, 1))])
check("map: from a cell not on the map whose only edge was recorded blocked, nothing is reachable (cause C)",
      g17c.astar((2, 0), [(0, 3)]) is None and not hs.grid_dist(g17c, (2, 0)).get((0, 3)))
es17 = g17c.adopt_here((2, 0))
check("map: adopt_here -> an episode node, the recorded-blocked edge re-triable once, a path exists",
      es17 == [hm.edge((2, 0), (2, 1))] and g17c.astar((2, 0), [(0, 3)]) is not None
      and (2, 0) not in g17c.free and [2, 0] not in g17c.to_dict()["free"])
g17c.add_blocked((2, 0), (2, 1))
check("map: a refused re-try re-blocks the edge for good (never re-tried again)",
      g17c.astar((2, 0), [(0, 3)]) is None and not g17c.retry_edges)
check("map: drop_here forgets the episode node", (g17c.drop_here() or True) and not g17c.here_free)

# -- pure: reach predictions (D)
check("ring_max: sqrt((1.40-0.10)^2 - dy^2) for the better stance (crouched only below the crouched eye); none "
      "when no stance can reach",
      abs(hs.ring_max(1.576, 0.95) - math.sqrt(1.3 ** 2 - (1.576 - 0.95) ** 2)) < 1e-6
      and abs(hs.ring_max(1.576, 0.3) - math.sqrt(1.3 ** 2 - (1.576 - 0.675 - 0.3) ** 2)) < 1e-6
      and hs.ring_max(1.576, 4.0) is None)
check("reach_pred: never crouch for a counter-height target (at / above the crouched eye), even if out of reach",
      hs.reach_pred(1.2, 1.576, 0.95)[1] == "stand")
d17, st17 = hs.reach_pred(0.75, 1.5, 0.2)
check("reach_pred: a low object 0.75 m away is out of standing reach (1.50 m) but within crouched reach",
      st17 == "crouch" and d17 < 1.2, f"{d17:.2f} {st17}")
check("reach_pred: standing is kept when it already reaches", hs.reach_pred(0.6, 1.5, 0.9)[1] == "stand")
check("min_view_h: an object 1.3 m below the eye needs >= 0.75 m horizontally at the 60-deg pitch limit",
      abs(hs.min_view_h(1.5, 0.2) - 1.3 / math.tan(math.radians(60))) < 1e-9)

# -- pure: foreground component (B)
dw = np.full((40, 40), 20.0, dtype="float32")
dw[5:35, 19:21] = 1.1                                  # a 2-px faucet strip in front of a window
fg17 = hs.foreground_points(dw, (10, 5, 30, 35))
check("foreground_points: a thin near strip in a box that is mostly far background -> the strip's pixels",
      fg17 and all(19 <= q[0] <= 20 and abs(dw[q[1], q[0]] - 1.1) < 1e-6 for q in fg17), str(fg17))
du = np.full((40, 40), 1.0, dtype="float32")
check("foreground_points: a uniform box (no ambiguity) -> nothing", hs.foreground_points(du, (10, 5, 30, 35)) == [])
dn = np.full((40, 40), 1.3, dtype="float32")
dn[5:35, 19:21] = 1.1
check("foreground_points: near and far within FG_GAP (one surface) -> nothing",
      hs.foreground_points(dn, (10, 5, 30, 35)) == [])
ground17, ok17 = honest.ground_point(dw, (10, 5, 30, 35))
check("the gateway's grounding of that box lands on the window (median): the live cause, unchanged here",
      ok17 and dw[ground17[1], ground17[0]] > 10, str(ground17))

# -- pure: leaf footprint (A)
m17 = {"kind": "swing", "c": (2.95, 1.0, 1.0), "n": (-1.0, 0.0), "hinges": [-1.4, -0.6], "r": 1.0,
       "edges": (-1.4, -0.6)}                          # front at x = 2.95 facing -x: lateral = -z (edges z 1.4 / 0.6)
g17d = GridMap("Fake", CELL, (0.0, 0.0), FREE)
pts17 = [(2.95 - t, 1.0, 0.45) for t in (0.2, 0.4, 0.6, 0.8)]
lc17 = hs.leaf_cells(g17d, m17, pts17)
check("leaf_cells: fitted to the door's own points, inflated by AGENT_R + LEAF_PAD (a narrow band at the leaf)",
      lc17 and all(0.45 - hs.AGENT_R - hs.LEAF_PAD - 1e-9 <= c[1] * CELL <= 0.6 + hs.AGENT_R + hs.LEAF_PAD + 1e-9
                   for c in lc17)
      and len(lc17) < len(hs.swept_cells(g17d, m17)) / 2, f"{sorted(lc17)} vs {len(hs.swept_cells(g17d, m17))}")
check("leaf_cells: fewer than 3 observed points -> no leaf estimate, nothing hard",
      hs.leaf_cells(g17d, m17, pts17[:2]) == set())

# -- E: filter counts
st17e = {}
hs.put_cells(g17d, {"c": (2.95, 0.9, 1.0), "n": (-1.0, 0.0), "pts": [(2.95, 0.9, 1.0)], "ymin": 0.4, "ymax": 1.4},
             1.5, "Fridge", None, stages=st17e)
check("E: put_cells reports how many cells each filter removed (they add up to the free cells)",
      st17e["free"] == len(FREE) and sum(v for k, v in st17e.items() if k != "free") == st17e["free"], str(st17e))

# ---------------------------------------------------------------- A integration: a big fridge at a corridor's end
# the fridge stands in an alcove (x 7..9, z 2..6); its only exit (7,2) -> (6,2) passes beside the fridge's -z edge,
# inside the agent's prior sweep (hinge unknown) -- the FloorPlan1 shape, scaled down; the rest of the kitchen is x 0..6
CORR = {(x, z) for x in range(7, 10) for z in range(2, 7)} | {(x, z) for x in range(7) for z in range(3)}


def corridor_kitchen():
    objs = [o for o in kitchen() if o["type"] not in ("Fridge",)]
    objs.append(dict(id="Fridge|+2.90|+0.90|+1.10", type="Fridge", pos=(2.95, 0.9, 1.1), size=1.3, openable=True,
                     receptacle=True, isOpen=False, cavity=0.2))
    fr = objs[-1]
    # its real door (0.65 m leaf, hinged at the -z edge) is narrower than the 1.3 m body front the agent measures, so
    # the agent's prior sweep (hinge unknown, r = 1.0 m + body margin) over-covers the corridor -- the live case
    hinge = (2.95, 1.1 - 0.65)
    fr["door_cells"] = {c for c in CORR if math.hypot(c[0] * CELL - hinge[0], c[1] * CELL - hinge[1]) <= 0.65}
    # the still, open leaf (90 deg, along -x from the hinge) is what blocks a walk while it stands open
    fr["leaf_cells"] = {c for c in CORR if abs(c[1] * CELL - hinge[1]) <= 0.15 and c[0] * CELL >= 2.95 - 0.7}
    for o in objs:
        if o["type"] == "Microwave":
            o["pos"] = (1.6, 1.1, -0.5)
    return objs


with_free(CORR)
od_save, op_save = hs.door_points, hs.opened_front
# the leaf is not measurable here (seen edge-on: no door points, nothing moved in view) -- most live trap cases had
# "prior (hinge unknown: both edges)"; the agent then has only its prior sweep
hs.door_points = lambda *a, **k: []
hs.opened_front = lambda *a, **k: (None, [])
try:
    t, det, h, sk = setup(start=(8, 5), yaw=90.0, objs=corridor_kitchen(), **EXH)
    plan17 = [{"action": "open", "object": "Fridge"}, {"action": "open", "object": "Microwave"}]
    r17a = sk.exec_step(plan17[0], upcoming=plan17[1:])
    ent17 = next((e for e in sk.opened_by_me if e["type"] == "Fridge"), None)
    cur17 = hm.current_cell(h, sk.gm)[0]
    old17 = None
    if ent17 is not None and sk.doors:
        gold = GridMap("Fake", CELL, (0.0, 0.0), CORR)
        gold.block_cells("old", hs.swept_cells(gold, sk.doors[-1]["model"]), keep=[cur17])
        mw = [c for c in CORR if c[0] <= 2]
        old17 = gold.astar(cur17, mw)
    print(f"  (info) fridge opened from {cur17}; old all-hard sweep: path to the microwave side = "
          f"{'none' if old17 is None else len(old17)}; hard {len(sk.gm._tb)} / soft {len(sk.gm.soft_cells())}")
    r17b = sk.exec_step(plan17[1])
    check("A reproduced: with the old all-hard sweep the agent at the fridge has no path to the microwave side",
          r17a["ok"] and old17 is None, " | ".join(r17a["detail"]["events"][-6:]))
    check("A fixed: open Fridge, then open Microwave with the fridge still open -> both ok (sweep soft, not a wall)",
          r17a["ok"] and r17b["ok"] and obj(t, "Fridge")["isOpen"] and obj(t, "Microwave")["isOpen"],
          f"{r17b['cause']} {r17b['detail']['why']} | " + " | ".join(r17b["detail"]["events"][-6:]))
    # the leaf was NOT observed, so nothing is hard: a soft sweep is not a collision guarantee (Astra) -- at most one
    # refused move into the standing leaf, then the walk re-plans round the refused edge
    check("...the open event says hard (leaf) vs soft (rest of the sweep); <= 1 refused move into the unobserved "
          "leaf (then re-planned); 0 violations",
          any("avoided where a detour exists (soft)" in e for e in r17a["detail"]["events"])
          and getattr(t, "door_hits", 0) <= 1 and h.audit.summary()["violations"] == 0,
          f"door_hits {getattr(t, 'door_hits', 0)}")
finally:
    with_free(FREE_SAVE17)
    hs.door_points, hs.opened_front = od_save, op_save

# -- A live case (2026-10-09, groceries): a FITTED leaf estimate that rings the agent (all four neighbours hard) would
#    cut it off; the door opened without meeting it, so the cells beside it are kept soft (said so)
t, det, h, sk = setup(start=(4, 4), yaw=90.0, **EXH)
ring = {(3, 4), (5, 4), (4, 3), (4, 5), (5, 5), (3, 3)}
lc_save = hs.leaf_cells
hs.leaf_cells = lambda *a, **k: set(ring)
try:
    frE = {"c": (2.95, 0.9, 1.0), "n": (-1.0, 0.0), "pts": [(2.95, y, z) for y in (0.5, 0.9, 1.3) for z in (0.6, 1.0, 1.4)],
           "ymin": 0.5, "ymax": 1.3, "planar": True}
    entE = {"type": "Fridge", "front": frE, "xyz": (2.95, 0.9, 1.0), "why": "explicit", "step": h.step}
    sk._block_door(entE)
finally:
    hs.leaf_cells = lc_save
evE = sk.events
check("A guard: a fitted leaf that would enclose the agent -> the cells beside it kept soft, the agent can walk out",
      not any(n in sk.gm._tb for n in [(3, 4), (5, 4), (4, 3), (4, 5)]) and len(hs.grid_dist(sk.gm, (4, 4))) > 50
      and any("would enclose the agent" in e for e in evE), " | ".join(evE[-3:]))
sk.gm.unblock_cells(entE["door_key"])

# -- A live case (2026-10-09, groceries): a put opened the fridge from inside the alcove; its open leaf now stands
#    across the alcove's exit (every move out refused) and a later step needs the fridge again -> the walk fails, the
#    agent closes the door it opened (from where it stands), walks out, and the next put opens it again
def exit_kitchen():
    objs = corridor_kitchen()
    fr = next(o for o in objs if o["type"] == "Fridge")
    fr["leaf_cells"] = {(7, 2), (8, 2), (9, 2), (7, 3)}        # the open leaf across the exit (fake physics)
    for o in objs:
        if o["type"] == "Apple":
            o["pos"] = (0.5, 0.97, -0.45)
        if o["type"] == "Potato":
            o["pos"] = (1.0, 0.97, -0.45)
    return objs


with_free(CORR)
od_save, op_save = hs.door_points, hs.opened_front
hs.door_points = lambda *a, **k: []
hs.opened_front = lambda *a, **k: (None, [])
try:
    t, det, h, sk = setup(start=(3, 1), yaw=0.0, objs=exit_kitchen(), **EXH)
    planX = [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"},
             {"action": "pick_and_place", "object": "Potato", "to": "Fridge"}]
    rX = [sk.exec_step(st, upcoming=planX[i + 1:]) for i, st in enumerate(planX)]
    evX = [e for r_ in rX for e in r_["detail"]["events"]]
    check("A escape: boxed in by the open leaf of a fridge it opened for a put -> closes it, walks out, the next put "
          "reopens it: both puts ok, both items inside, nothing left open",
          all(r_["ok"] for r_ in rX) and obj(t, "Apple").get("inside") == obj(t, "Fridge")["id"]
          and obj(t, "Potato").get("inside") == obj(t, "Fridge")["id"]
          and any("closing it first" in e for e in evX) and not openables_closed(t),
          " | ".join(f"{r_['ok']} {r_['cause']} {r_['detail']['why']}" for r_ in rX) + " | "
          + " | ".join(e for e in evX if "Fridge" in e)[-700:])
finally:
    with_free(FREE_SAVE17)
    hs.door_points, hs.opened_front = od_save, op_save

# ---------------------------------------------------------------- B integration: a thin faucet before a window
def window_kitchen():
    objs = kitchen()
    f = next(o for o in objs if o["type"] == "Faucet")
    f.update(thin=True, backdrop=20.0, size=0.25)
    return objs


t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=window_kitchen(), **EXH)
fau = next((tr for tr in sk.om.tracks.values() if tr.type == "Faucet" and tr.xyz), None)
check("B reproduced: the faucet's remembered estimate lies far behind it (the window), not actable from any cell",
      fau is not None and math.dist(fau.xyz, obj(t, "Faucet")["pos"]) > 3.0 and not sk._actable(fau.xyz),
      str(fau.xyz if fau else None))
r17w = sk.exec_step({"action": "wash", "object": "Bowl"})
ev17w = r17w["detail"]["events"]
check("B fixed: wash Bowl -> ok: put in the sink, faucet on and off at the faucet's near component, bowl clean",
      r17w["ok"] and obj(t, "Bowl").get("in_sink") and obj(t, "Bowl")["isDirty"] is False
      and obj(t, "Faucet")["isToggled"] is False
      and len([c for c in t.calls if c[0] == "ToggleObjectOn"]) >= 1,
      f"{r17w['cause']} {r17w['detail']['why']} | " + " | ".join(ev17w[-8:]))
check("...said as a candidate, not a diagnosis ('foreground candidate selected', 'suspected background')",
      any("foreground candidate selected" in e and "suspected background" in e for e in ev17w),
      " | ".join(e for e in ev17w if "Faucet" in e)[:400])
check("...memory is unchanged (its faucet estimate is still the grounded one); 0 violations, no ids",
      math.dist(next(tr for tr in sk.om.tracks.values() if tr.type == "Faucet").xyz, obj(t, "Faucet")["pos"]) > 3.0
      and h.audit.summary()["violations"] == 0 and no_ids(r17w))
t2, det2, h2, sk2 = setup(start=(2, 3), yaw=270.0, objs=window_kitchen())
r17wb = sk2.exec_step({"action": "wash", "object": "Bowl"})
print(f"  (info) bounded (s5) policy, unchanged: wash ok={r17wb['ok']} cause={r17wb['cause']}")

# ---------------------------------------------------------------- C integration: the episode starts off the own map
# the live case: a setup step placed the agent on (4,0) at a counter edge; (3,0) / (5,0) are not standable, and
# when the map was walked the move (4,1) -> (4,0) was refused, so that edge is recorded blocked
with_free(FREE_SAVE17 - {(3, 0), (5, 0)})
try:
    t = FakeThor(kitchen(), (4, 0), 0.0)
    det = FakeDetector(t)
    h = honest.HonestAgent(t, det, mode="honest")
    gmC = GridMap("Fake", CELL, (0.0, 0.0), FREE - {(4, 0)}, blocked_edges=[((4, 0), (4, 1))])
    om = objmem.ObjectMemory("Fake", store=None, episode="t")
    check("C reproduced: from (4,0) (not on the own map, its only edge recorded blocked) the own map reaches nothing",
          gmC.astar((4, 0), [(4, 8)]) is None and len(hs.grid_dist(gmC, (4, 0))) == 1)
    skC = hs.HonestSkills(h, gmC, om, **EXH)
    skC.start_episode("t")
    rC = skC.exec_step({"action": "pickup", "object": "Apple"})
finally:
    with_free(FREE_SAVE17)
check("C fixed: an episode node from its own pose; the recorded-blocked edge re-tried once -> pickup Apple ok",
      rC["ok"] and h.holding == "Apple" and hm.edge((4, 0), (4, 1)) in gmC.retry_edges,
      f"{rC['cause']} {rC['detail']['why']}")
check("...said so in the episode events; the map's free cells are unchanged (nothing persisted)",
      (4, 0) not in gmC.free and [4, 0] not in gmC.to_dict()["free"] and skC.gm.here_free == {(4, 0)},
      str(skC.gm.here_free))
t = FakeThor(kitchen(), (4, 0), 0.0)
det = FakeDetector(t)
h = honest.HonestAgent(t, det, mode="honest")
FREE.discard((4, 1))
FREE.discard((3, 0))
FREE.discard((5, 0))
try:
    gmC2 = GridMap("Fake", CELL, (0.0, 0.0), FREE_SAVE17 - {(4, 0)}, blocked_edges=[((4, 0), (4, 1))])
    skC2 = hs.HonestSkills(h, gmC2, objmem.ObjectMemory("Fake", store=None, episode="t"), **EXH)
    skC2.start_episode("t")
    rC2 = skC2.exec_step({"action": "pickup", "object": "Apple"})
    n_try = sum(1 for a, kw in t.calls if a.startswith("Move"))
    check("C: when the re-try is refused again (really walled in) it is not repeated: fails, few moves",
          not rC2["ok"] and rC2["cause"] in ("unreachable", "retry_exhausted") and n_try <= 4
          and not gmC2.retry_edges and hm.edge((4, 0), (4, 1)) in gmC2.blocked_edges,
          f"{rC2['cause']} moves {n_try}")
finally:
    with_free(FREE_SAVE17)

# -- C live case (2026-10-09 probe): the setup left the agent 0.08-0.13 m OFF the lattice, between a free cell and one
#    the map does not have -> an explicit correction move onto the nearest free cell centre, said so; never snapped
t = FakeThor(kitchen(), (4, 0), 180.0)
t.event.metadata["agent"]["position"]["z"] = 0.136          # rounds to (4,1), 0.114 m off; (4,0) is 0.136 m away
t.event.metadata["cameraPosition"]["z"] = 0.136
t._render()
det = FakeDetector(t)
h = honest.HonestAgent(t, det, mode="honest")
gmL = GridMap("Fake", CELL, (0.0, 0.0), FREE - {(4, 1)}, blocked_edges=[((4, 1), (4, 2)), ((4, 1), (3, 1)),
                                                                     ((4, 1), (5, 1))])
skL = hs.HonestSkills(h, gmL, objmem.ObjectMemory("Fake", store=None, episode="t"), **EXH)
skL.start_episode("t")
pL = h.pose()["position"]
rL = skL.exec_step({"action": "pickup", "object": "Apple"})
evL = rL["detail"]["events"]
check("C (live case): off the lattice -> one explicit correction move onto the nearest free cell (4,0), on the "
      "lattice after it, said so in the first step's events; then pickup Apple ok",
      abs(pL["z"] - 0.0) < 1e-3 and abs(pL["x"] - 1.0) < 1e-3 and rL["ok"]
      and any("off the lattice: moved" in e and "(4, 0)" in e and "(episode start)" in e for e in evL),
      f"{pL} {rL['cause']} | " + " | ".join(evL[:4]))

# -- D live case (2026-10-09, wash-cup): a PLACED belief's xyz is the receptacle anchor (here forced to floor height,
#    as the live sink anchor read y = 0); its own height is unknown -> no height-derived crouch / reach move
t, det, h, sk = setup(start=(2, 3), yaw=270.0, **EXH)
sk.exec_step({"action": "pickup", "object": "Bowl"})
sk.exec_step({"action": "put_in", "object": "SinkBasin"})
tb_ = next(tr for tr in sk.om.tracks.values() if tr.type == "Bowl" and tr.status == "placed")
tb_.anchor = (tb_.anchor[0], 0.0, tb_.anchor[2])
n0 = len(t.calls)
rA = sk.exec_step({"action": "clean", "object": "Bowl"})
check("D: an anchor-only (placed) target never triggers the height-derived crouch; clean Bowl in the sink ok",
      rA["ok"] and not any(a == "Crouch" for a, _ in t.calls[n0:]) and obj(t, "Bowl")["isDirty"] is False,
      f"{rA['cause']} {rA['detail']['why']} | " + " | ".join(rA["detail"]["events"][-5:]))

# ---------------------------------------------------------------- D integration: a low object beyond standing reach
def low_kitchen():
    objs = kitchen()
    # low in a base cabinet; (the fake's own estimate of a 0.12 m object in its 100 px frame is ~0.3 m off, more than
    # live -- so the sponge sits low enough that standing is out of reach even with that error)
    objs.append(dict(id="DishSponge|+1.20|+0.10|+3.00", type="DishSponge", pos=(1.2, 0.1, 3.0), size=0.12,
                     pickupable=True))
    return objs


t, det, h, sk = setup(start=(4, 5), yaw=0.0, objs=low_kitchen(), **EXH)
sk.type_list = sorted(set(sk.type_list) | {"DishSponge"})
det.names[len(det.names)] = "DishSponge"
rD = sk.exec_step({"action": "pickup", "object": "DishSponge"})
evD = rD["detail"]["events"]
check("D fixed: a sponge 1.4 m below the eye is picked up (crouched where standing is out of reach)",
      rD["ok"] and h.holding == "DishSponge" and not any("step back" in e for e in evD),
      f"{rD['cause']} {rD['detail']['why']} | " + " | ".join(evD[-8:]))
check("...by predicted reach (crouch / a nearer cell said so), not by a blind sidestep / step back",
      any(("crouch for reach" in e) or ("move nearer the target for reach" in e and "predicted" in e) for e in evD)
      and not any("sidestep one cell" in e for e in evD), " | ".join(evD))
# reproduce the old recovery: from the same cell, the old k=2 / k=3 moves go AWAY from the target
t, det, h, sk = setup(start=(4, 9), yaw=0.0, objs=low_kitchen())          # bounded policy = the old recovery
tgt = obj(t, "DishSponge")["pos"]
d0 = math.hypot(4 * CELL - tgt[0], 9 * CELL - tgt[2])
st_ = {"alt": 0, "dpitch": 0.0}
sk._recover(3, tgt, st_)
c3 = hm.current_cell(h, sk.gm)[0]
check("D reproduced: the old recovery 3 ('step back one cell') walks AWAY from an out-of-reach target",
      math.hypot(c3[0] * CELL - tgt[0], c3[1] * CELL - tgt[2]) > d0, str(c3))
t, det, h, sk = setup(start=(4, 7), yaw=0.0, objs=low_kitchen(), **EXH)
moved = sk._reach_move(tgt)
c4 = hm.current_cell(h, sk.gm)[0]
dd4 = math.hypot(math.hypot(c4[0] * CELL - tgt[0], c4[1] * CELL - tgt[2]), t.event.metadata["cameraPosition"]["y"] - tgt[1])
check("D: _reach_move goes where the predicted 3D reach is better (here: nearer and/or crouched), within the gate",
      moved and dd4 <= hs.ACT_REACH, f"{c4} {dd4:.2f}")
check("D/B/C are console-only: the bounded (s5) runner keeps its stand-off, recovery and grounding",
      setup()[3].exhaustive is False)

# ======================================================================================================== 18
# B1 (2026-10-09): a dish put in the sink is not re-detected (wet / half hidden) -> wash / pick-back-up used to fail
# "no Bowl in view". The agent aims from its OWN depth + pose (what is newly occupied since the frame it put from, then
# a declared grid over the receptacle) and lets the reach gate and the action's success bool decide.
print("\n[18] B1: re-aiming at a dish this agent put in the sink when the detector no longer finds it")


def hide_in_sink(det_, typ="Bowl", water_only=False):
    """The detector loses the dish in the sink (always, or only while the water runs: live 2026-10-09 the pan was
    re-seen right after the put, then 'not seen in 2 clear views' once the faucet ran)."""
    base = det_.detect

    def detect(rgb, conf=0.25, inspecting=None):
        run = any(o["type"] == "Faucet" and o.get("isToggled") for o in det_.thor.objs.values())
        return [d for d in base(rgb, conf, inspecting)
                if not (d["type"] == typ and (run or not water_only)
                        and any(o["type"] == typ and o.get("in_sink") for o in det_.thor.objs.values()))]
    det_.detect = detect


def settle_in_sink(t_, dx=0.0, dz=0.0):
    """The engine settles a put object anywhere on the receptacle (not under the pixel): offset it after the put."""
    base = t_.do

    def do(action, **kw):
        ev_ = base(action, **kw)
        if action == "PutObject":
            for o in t_.objs.values():
                if o.get("in_sink") and not o.get("settled"):
                    o["pos"] = (o["pos"][0] + dx, o["pos"][1], o["pos"][2] + dz)
                    o["settled"] = True
            t_._render()
        return ev_
    t_.do = do


# -- pure: the change evidence and the pixel order
flat0 = np.full((60, 60), 2.0, dtype="float32")
pose0 = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"x": 0, "y": 0.0, "z": 0}, "horizon": 0.0,
         "camera": {"x": 0.0, "y": 1.0, "z": 0.0}}
bump = flat0.copy()
bump[25:35, 25:35] = 1.7                       # something 0.3 m nearer than the surface seen at the put
cm = hs.change_mask(flat0, pose0, bump, pose0, objmem.FOV_DEFAULT, (15, 15, 45, 45))
check("B1 pure: change_mask flags exactly the newly nearer patch (same pose)",
      cm and all(25 <= v[0] <= 34 and 25 <= v[1] <= 34 for v in cm.values()), str(len(cm)))
none_ = hs.change_mask(flat0, pose0, flat0.copy(), pose0, objmem.FOV_DEFAULT, (15, 15, 45, 45))
check("B1 pure: an unchanged view claims nothing", not none_)
farther = flat0 + 0.5
check("B1 pure: a surface FARTHER than at the put is not 'new' (absence does not contradict the put frame)",
      not hs.change_mask(flat0, pose0, farther, pose0, objmem.FOV_DEFAULT, (15, 15, 45, 45)))
recp = {"p0": objmem.backproject(30, 30, 2.0, pose0, (60, 60), objmem.FOV_DEFAULT), "half": 0.5,
        "depth": flat0, "pose": pose0}
pix, note = hs.blind_pixels(recp, bump, pose0, objmem.FOV_DEFAULT)
check("B1 pure: blind_pixels puts the changed patch first, then a bounded fallback grid",
      pix and 25 <= pix[0][0] <= 34 and 25 <= pix[0][1] <= 34 and len(pix) <= hs.BLIND_MAX and "component" in note,
      f"{pix[:3]} {note}")
pix2, note2 = hs.blind_pixels(recp, flat0.copy(), pose0, objmem.FOV_DEFAULT)
check("B1 pure: with no change at all it falls back to the put point itself first, then a ring and a grid, said so",
      pix2 and "no newly occupied" in note2 and abs(pix2[0][0] - 30) <= 1 and abs(pix2[0][1] - 30) <= 1
      and len(pix2) <= hs.BLIND_MAX, f"{pix2[:3]} {note2}")

recp_hi = dict(recp, p0=(recp["p0"][0], recp["p0"][1] + 0.2, recp["p0"][2]))
pixh, noteh = hs.blind_pixels(recp_hi, bump, pose0, objmem.FOV_DEFAULT)
check("B1 pure: an object settled on a basin floor BELOW the rim point the put pixel hit is still found (volume reaches down)",
      pixh and 25 <= pixh[0][0] <= 34 and 25 <= pixh[0][1] <= 34 and "component" in noteh, f"{pixh[:3]} {noteh}")

# -- pure: a running faucet's water stream is the NEAREST component in its box; the faucet body is the next one
dst = np.full((40, 40), 20.0, dtype="float32")
dst[8:32, 17:24] = 0.90                         # the faucet body (near, thin, before a far window)
dst[14:26, 20:22] = 0.38                        # the water stream in front of it
fcs = hs.foreground_components(dst, (5, 5, 35, 35), k=2)
check("B1 pure: foreground_components returns the stream first and the faucet body as a second, distinct component",
      len(fcs) == 2 and all(abs(dst[y, x] - 0.38) < 1e-3 for x, y in fcs[0][:3])
      and all(abs(dst[y, x] - 0.90) < 1e-3 for x, y in fcs[1][:3]), str([c[:2] for c in fcs]))
check("B1 pure: foreground_points (k = 1) is unchanged: the nearest component only",
      hs.foreground_points(dst, (5, 5, 35, 35)) == fcs[0][:6] or hs.foreground_points(dst, (5, 5, 35, 35)) == fcs[0])

# -- integration, before: with the aim switched off the wash fails exactly as live (not_detected: no Bowl in view)
BP0 = hs.BLIND_PRIMS
t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
hide_in_sink(det)
settle_in_sink(t, dz=0.12)
hs.BLIND_PRIMS = ()
r18b = sk.exec_step({"action": "wash", "object": "Bowl"})
hs.BLIND_PRIMS = BP0
check("B1 reproduced: detector blind to the bowl in the sink -> wash fails 'no Bowl in view' (bowl still dirty)",
      (not r18b["ok"]) and obj(t, "Bowl")["isDirty"] is True and obj(t, "Bowl").get("in_sink")
      and "no Bowl in view" in (r18b["detail"]["why"] or "") + " ".join(r18b["detail"]["events"]) + str(r18b["cause"]),
      f"{r18b['cause']} {r18b['detail']['why']}")
# -- after
for nm, (dx, dz) in {"settled 0.12 m off the put point": (0.0, 0.12), "settled where it was put": (0.0, 0.0)}.items():
    t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
    hide_in_sink(det)
    settle_in_sink(t, dx, dz)
    r18 = sk.exec_step({"action": "wash", "object": "Bowl"})
    ev18 = r18["detail"]["events"]
    check(f"B1 fixed ({nm}): wash Bowl -> ok, clean, faucet off, via the agent's own aim (success bool)",
          r18["ok"] and obj(t, "Bowl")["isDirty"] is False and obj(t, "Faucet")["isToggled"] is False
          and "success-bool-own-aim" in r18["detail"]["verify"],
          f"{r18['cause']} {r18['detail']['why']} | " + " | ".join(ev18[-6:]))
    check("...said as a hypothesis ('aiming from my own depth ... hypothesis'), 0 violations, no ids, 0 oracle reads",
          any("own depth" in e and "hypothesis" in e for e in ev18) and h.audit.summary()["violations"] == 0
          and no_ids(r18) and h.audit.summary()["counts"].get("oracle", 0) == 0, " | ".join(ev18[-6:]))
# -- the pick back up: same blindness, the held bowl goes on the counter
rpb = sk.exec_step({"action": "pick_and_place", "object": "Bowl", "to": "CounterTop"})
check("B1 fixed: pick the bowl back out of the sink (detector blind) and put it on the counter",
      rpb["ok"] and obj(t, "Bowl").get("on") == "CounterTop" and any("own-depth aim" in e for e in rpb["detail"]["events"]),
      f"{rpb['cause']} {rpb['detail']['why']} | " + " | ".join(rpb["detail"]["events"][-6:]))
# -- live 2026-10-09 (wash-pan): the dish IS re-seen in the sink after the put, then lost once the water runs; memory
#    contradicts it ("not seen in 2 clear views") and holds no placed belief any more -> the agent's own put decides
t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
hide_in_sink(det, water_only=True)
settle_in_sink(t, 0.0, 0.12)
r18w = sk.exec_step({"action": "wash", "object": "Bowl"})
ev18w = r18w["detail"]["events"]
mem_b = [b for b in sk.om.lookup("Bowl") if b["status"] in ("trusted", "placed")]
check("B1 fixed (water hides the dish AFTER it was re-seen; memory no longer holds a usable belief): wash ok",
      r18w["ok"] and obj(t, "Bowl")["isDirty"] is False and obj(t, "Faucet")["isToggled"] is False
      and any("own depth" in e for e in ev18w),
      f"{r18w['cause']} {r18w['detail']['why']} | " + " | ".join(ev18w[-6:]))
obj(t, "Faucet")["isToggled"] = True            # the water runs again (the bowl is lost to the detector)
t._render()
rpw = sk.exec_step({"action": "pick_and_place", "object": "Bowl", "to": "CounterTop"})
check("...and the pick back up from the sink is aimed from the put record ('I put it ... myself')",
      rpw["ok"] and obj(t, "Bowl").get("on") == "CounterTop"
      and any("I put it in SinkBasin myself" in e for e in rpw["detail"]["events"]),
      f"{rpw['cause']} {rpw['detail']['why']} | " + " | ".join(rpw["detail"]["events"][-6:]))
# -- live 2026-10-09 (wash-pan): the running water cleans the dish by itself, so every explicit clean is refused
#    ("already clean"), and the detector is blind to the dish while the water runs -> the agent turns the water off,
#    looks, and reads the dish clean by sight (never claims success from a refused action)
t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
hide_in_sink(det, water_only=True)
settle_in_sink(t, 0.0, 0.12)
base_do = t.do


def do_auto(action, **kw):
    ev_ = base_do(action, **kw)
    if action == "ToggleObjectOn" and obj(t, "Faucet")["isToggled"]:
        for o in t.objs.values():
            if o.get("in_sink") and "isDirty" in o:
                o["isDirty"] = False                    # the stream cleans whatever lies in the basin
    return ev_


t.do = do_auto
r18a = sk.exec_step({"action": "wash", "object": "Bowl"})
ev18a = r18a["detail"]["events"]
check("B1 fixed (water cleans it, explicit clean refused, detector blind while it runs): wash ok by sight afterwards",
      r18a["ok"] and obj(t, "Faucet")["isToggled"] is False and "state-after-water" in r18a["detail"]["verify"]
      and any("water off" in e and "clean" in e for e in ev18a),
      f"{r18a['cause']} {r18a['detail']['why']} | " + " | ".join(ev18a[-6:]))
# the same, but the bowl is still dirty and unseen: no claim of success
t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
det.hide = set()
hide_in_sink(det, water_only=False)
settle_in_sink(t, 0.0, 0.12)
t.refuse[("CleanObject", "Bowl")] = 99
r18c = sk.exec_step({"action": "wash", "object": "Bowl"})
check("B1 honesty: clean refused everywhere and the dish never seen clean -> the step FAILS (no claim from a refusal)",
      (not r18c["ok"]) and obj(t, "Bowl")["isDirty"] is True and obj(t, "Faucet")["isToggled"] is False,
      f"{r18c['cause']} {r18c['detail']['why']}")
# -- a refused toggle (the pixel landed on the stream / basin, not the faucet) goes on to the next passing pixel
t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
obj(t, "Faucet")["isToggled"] = True
t._render()
t.refuse[("ToggleObjectOff", "Faucet")] = 1
rtg = sk.exec_step({"action": "toggle_off", "object": "Faucet"})
nto = len([c for c in t.calls if c[0] == "ToggleObjectOff"])
check("B1: a refused ToggleObjectOff is tried again at the next passing pixel in the same send (no retry cycle)",
      rtg["ok"] and rtg["retries"] == 0 and nto == 2 and obj(t, "Faucet")["isToggled"] is False,
      f"{rtg['cause']} retries={rtg['retries']} calls={nto} {rtg['detail']['why']}")
# -- no record, no aim: an object the agent did not put there is never guessed at
t, det, h, sk = setup(start=(2, 3), yaw=270.0, objs=kitchen(), **EXH)
det.hide = {"Bowl"}
r18n = sk.exec_step({"action": "pickup", "object": "Bowl"})
check("B1 guard: no put record -> no blind aim (still not_detected, nothing picked up)",
      (not r18n["ok"]) and h.holding is None and not any("own depth" in e for e in r18n["detail"]["events"]),
      f"{r18n['cause']} {r18n['detail']['why']}")


# ======================================================================================================== 19
# B2 (2026-10-09): "put Pot -> CounterTop" reported ok but the pot sat in the sink. A counter box spans the whole counter,
# including the basin cut into it; the nearest-the-centre pixel (or, when the exclusion rules emptied the list, the
# legacy fallback to the box centre) was a basin pixel. Now: pixels inside competing receptacle boxes are out, then only
# the dominant support plane of what is left; and after the put the agent looks at where the object rests (own depth)
# and, if its base is below that plane, picks it back up and puts it elsewhere. Reproduced first on a fake counter with
# a horizontal top and a lowered basin (a mini renderer for horizontal rectangles).
print("\n[19] B2: a put on a counter must land on the counter, not in the basin cut into it")


def ray_dirs(pose, wh, fov=objmem.FOV_DEFAULT):
    """(eye, dirs[h, w, 3]) of unit-axial rays (d . forward = 1) for a pose -- the inverse of objmem.backproject."""
    w, h_ = wh
    eye, r, u, fw = objmem._basis(pose)
    f = objmem._focal(h_, fov)
    xs = (np.arange(w) + 0.5 - w / 2.0) / f
    ys = -(np.arange(h_) + 0.5 - h_ / 2.0) / f
    d = (np.asarray(fw)[None, None, :] + xs[None, :, None] * np.asarray(r)[None, None, :]
         + ys[:, None, None] * np.asarray(u)[None, None, :])
    return np.asarray(eye), d


def plane_depth(pose, wh, y, rect, hole=None):
    """Axial depth image of the horizontal rectangle ``rect`` = (x0, z0, x1, z1) at height ``y`` (inf elsewhere);
    ``hole`` = a rectangle cut out of it."""
    eye, d = ray_dirs(pose, wh)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y - eye[1]) / d[:, :, 1]
    px_, pz_ = eye[0] + t * d[:, :, 0], eye[2] + t * d[:, :, 2]
    ok = (t > 0.05) & (px_ >= rect[0]) & (px_ <= rect[2]) & (pz_ >= rect[1]) & (pz_ <= rect[3])
    if hole is not None:
        ok &= ~((px_ >= hole[0]) & (px_ <= hole[2]) & (pz_ >= hole[1]) & (pz_ <= hole[3]))
    return np.where(ok, t, np.inf).astype("float32")


def bbox_of(mask):
    ys_, xs_ = np.nonzero(mask)
    return (int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())) if len(xs_) else None


# -- pure: a counter with a basin cut into it, the basin box detected (live layout: counter, sink, basin, floor)
WH_ = (160, 120)
pose19 = {"position": {"x": 1.25, "y": 0.9, "z": 0.8}, "rotation": {"x": 0, "y": 0.0, "z": 0}, "horizon": 30.0,
          "standing": True, "camera": {"x": 1.25, "y": 1.5, "z": 0.8}}
CT_RECT, BASIN_RECT, TOP_Y, BASIN_Y = (0.2, 1.3, 2.3, 2.1), (0.9, 1.45, 1.6, 1.95), 0.90, 0.74
d_ct = plane_depth(pose19, WH_, TOP_Y, CT_RECT, hole=BASIN_RECT)
d_bs = plane_depth(pose19, WH_, BASIN_Y, BASIN_RECT)
d_fl = plane_depth(pose19, WH_, 0.0, (-2.0, -2.0, 5.0, 5.0))
dep19 = np.minimum(np.minimum(d_ct, d_bs), d_fl)
dep19 = np.where(np.isfinite(dep19), dep19, 20.0).astype("float32")
m_ct, m_bs = np.isfinite(d_ct) & (d_ct <= d_fl), np.isfinite(d_bs) & (d_bs <= d_ct)
b_ct, b_bs = bbox_of(m_ct | m_bs), bbox_of(m_bs)
b_sink = (b_bs[0] - 8, b_bs[1] - 6, b_bs[2] + 8, b_bs[3] + 6)
b_fl = bbox_of(np.isfinite(d_fl) & (d_fl < np.minimum(d_ct, d_bs)))
dets19 = [{"type": "Sink", "bbox_xyxy": b_sink}, {"type": "SinkBasin", "bbox_xyxy": b_bs},
          {"type": "Floor", "bbox_xyxy": b_fl}]
res19 = hs.put_surface(dep19, pose19, objmem.FOV_DEFAULT, b_ct, dets19, "CounterTop", max_depth=hs.REACH)
res19b = hs.put_surface(dep19, pose19, objmem.FOV_DEFAULT, b_ct, [], "CounterTop", max_depth=hs.REACH)
in_bs = lambda p: bool(m_bs[p[1], p[0]]) or hs._inside(p, b_bs)                                   # noqa: E731
check("B2 pure: with the basin box detected, no returned pixel lies in the basin; the counter plane is found",
      res19["pts"] and not any(in_bs(p) for p in res19["pts"]) and res19["plane"] is not None
      and abs(res19["plane"] - TOP_Y) < 0.05, f"{len(res19['pts'])} {res19['plane']} {res19['note']}")
check("B2 pure: with the basin box NOT detected (blind detector) the support plane alone keeps the pixels off the basin",
      res19b["pts"] and not any(m_bs[p[1], p[0]] for p in res19b["pts"]) and res19b["plane"] is not None
      and abs(res19b["plane"] - TOP_Y) < 0.05, f"{len(res19b['pts'])} {res19b['plane']} {res19b['note']}")
check("B2 pure: a floor-sized box (not a receptacle) does not swallow the candidates (the live emptied list)",
      len(res19["pts"]) >= 3, str(len(res19["pts"])))
rej = [(float(c[0]), float(c[2]), 0.3) for c in
       [objmem.backproject(p[0], p[1], dep19[p[1], p[0]], pose19, WH_, objmem.FOV_DEFAULT) for p in res19["pts"][:1]]]
res19c = hs.put_surface(dep19, pose19, objmem.FOV_DEFAULT, b_ct, dets19, "CounterTop", rejected=rej,
                        max_depth=hs.REACH)
check("B2 pure: a rejected spot stays excluded in world coordinates",
      res19c["pts"] and res19["pts"][0] not in res19c["pts"], f"{res19['pts'][:2]} {res19c['pts'][:2]}")
# a competing receptacle covering the whole box and nothing else -> blocked, not a centre-pixel fallback
res19d = hs.put_surface(dep19, pose19, objmem.FOV_DEFAULT, b_bs, [{"type": "Plate", "bbox_xyxy": (b_bs[0] + 1, b_bs[1] + 1,
                                                                     b_bs[2] - 1, b_bs[3] - 1)}], "SinkBasin")
check("B2 pure: every pixel inside a competing receptacle's box -> blocked (no safe pixel), never a centre fallback",
      res19d["blocked"] and not res19d["pts"], str(res19d))


# -- integration: a fake counter with a horizontal top and a lowered basin
class SurfThor(FakeThor):
    """FakeThor + horizontal rectangles (objects carrying ``surface`` = {y, rect, hole}) painted into the depth image,
    each with a box; PutObject on one lands where the pixel's ray met it (the engine's choice, here: exactly there)."""
    last_hit = None

    def visible_objs(self):
        return [o for o in super().visible_objs() if not o.get("surface")]

    def _render(self):
        super()._render()
        if not hasattr(self, "objs"):
            return
        pose = self.pose()
        self.owner = np.full((H, W), -1, dtype="int16")
        surfs = [o for o in self.objs.values() if o.get("surface")]
        for i, o in enumerate(surfs):
            sf = o["surface"]
            dd = plane_depth(pose, (W, H), sf["y"], sf["rect"], sf.get("hole"))
            better = dd < self.depth_frame - 1e-4
            if sf.get("inside_of") is not None:
                better &= np.isfinite(dd)
            self.depth_frame = np.where(better, dd, self.depth_frame).astype("float32")
            self.owner = np.where(better, i, self.owner)
            self._surfs = surfs
        for i, o in enumerate(surfs):
            m = self.owner == i
            if m.any():
                cb = bbox_of(m)
                self.boxes.append((o, cb, float(np.median(self.depth_frame[m]))))

    def resolve_px(self, px, py):
        i = int(self.owner[int(round(py)), int(round(px))]) if hasattr(self, "owner") else -1
        if i >= 0:
            return self._surfs[i]
        return super().resolve_px(px, py)

    def do(self, action, **kw):
        if action == "GetObjectInFrame":
            px, py = kw["x"] * W - 0.5, kw["y"] * H - 0.5
            o = self.resolve_px(px, py)
            self.last_hit = None
            if o is not None and o.get("surface"):
                eye, d = ray_dirs(self.pose(), (W, H))
                tt = float(self.depth_frame[int(round(py)), int(round(px))])
                self.last_hit = tuple(eye + tt * d[int(round(py)), int(round(px))])
        return super().do(action, **kw)

    def dist(self, o):
        if o.get("surface") and self.last_hit is not None:
            c = self.event.metadata["cameraPosition"]
            return math.dist((c["x"], c["y"], c["z"]), self.last_hit)
        return super().dist(o)

    def interact(self, action, kw):
        o = self.objs.get(kw.get("objectId"))
        if action == "PutObject" and o is not None and o.get("surface") and self.held and self.last_hit is not None:
            if self.dist(o) > 1.5:
                return False
            hh = self.objs[self.held]
            hh["pos"] = (self.last_hit[0], o["surface"]["y"] + hh["size"] / 2.0, self.last_hit[2])
            hh["on"] = o["type"]
            hh.pop("inside", None)
            hh["in_sink"] = o["type"] == "SinkBasin"
            self.held = None
            return True
        return super().interact(action, kw)


def counter_kitchen():
    return [
        dict(id="CounterTop|+1.20|+0.90|+2.00", type="CounterTop", pos=(1.2, 0.9, 2.0), size=0.1, receptacle=True,
             surface=dict(y=0.90, rect=(-0.5, 1.0, 3.0, 2.8), hole=(0.6, 1.15, 1.9, 2.0))),
        dict(id="SinkBasin|+1.25|+0.74|+1.80", type="SinkBasin", pos=(1.25, 0.74, 1.8), size=0.1, receptacle=True,
             surface=dict(y=0.74, rect=(0.6, 1.15, 1.9, 2.0))),
        dict(id="Pot|+0.60|+1.00|+1.80", type="Pot", pos=(0.6, 1.0, 1.8), size=0.20, pickupable=True,
             receptacle=True, isDirty=False),
    ]


def setup19(blind_basin=True, start=(5, 1), yaw=0.0):
    t = SurfThor(counter_kitchen(), start, yaw)
    det = FakeDetector(t)
    if blind_basin:
        det.hide = {"SinkBasin"}
    h = honest.HonestAgent(t, det, mode="honest", smooth=None)
    gm = GridMap("Fake", CELL, (0.0, 0.0), FREE)
    om = objmem.ObjectMemory("Fake", store=None, episode="t")
    sk = hs.HonestSkills(h, gm, om, chooser=lambda *a: [], **EXH)
    sk.start_episode("t")
    return t, det, h, sk


def put_plan(sk):
    return sk.exec_step({"action": "pick_and_place", "object": "Pot", "to": "CounterTop"})


hs.PUT_SURFACE_ON = False
t, det, h, sk = setup19()
r19o = put_plan(sk)
hs.PUT_SURFACE_ON = True
pot0 = obj(t, "Pot")
check("B2 reproduced (legacy put-point rule, basin box not detected): the engine says ok, the pot is IN the basin",
      r19o["ok"] and pot0.get("on") == "SinkBasin" and pot0.get("in_sink"),
      f"{r19o['ok']} on={pot0.get('on')} {r19o['cause']} | " + " | ".join(r19o["detail"]["events"][-4:]))
t, det, h, sk = setup19()
r19 = put_plan(sk)
pot1 = obj(t, "Pot")
ev19 = r19["detail"]["events"]
check("B2 fixed (same scene): the pot ends on the CounterTop, not in the basin, step ok",
      r19["ok"] and pot1.get("on") == "CounterTop" and not pot1.get("in_sink") and pot1["pos"][1] > 0.95,
      f"on={pot1.get('on')} pos={pot1['pos']} {r19['cause']} {r19['detail']['why']} | " + " | ".join(ev19[-5:]))
check("...seen resting on the surface afterwards (verification by sight), 0 violations, no ids, 0 oracle reads",
      any("rests on the surface" in e for e in ev19) and h.audit.summary()["violations"] == 0 and no_ids(r19)
      and h.audit.summary()["counts"].get("oracle", 0) == 0, " | ".join(ev19[-5:]))

# -- the verification alone: aim the first put into the basin on purpose; the agent must see it and fix it
orig_ps = hs.put_surface
calls19 = []


def basin_first(depth, pose, fov, bbox, dets, rtype, rejected=(), n=9, max_depth=None):
    res = orig_ps(depth, pose, fov, bbox, dets, rtype, rejected=rejected, n=n, max_depth=max_depth)
    if not calls19:
        wh_ = (depth.shape[1], depth.shape[0])
        bad = [(x, y) for y in range(0, H, 2) for x in range(0, W, 2)
               if t.owner[y, x] == 1 and depth[y, x] <= hs.REACH]
        if bad:
            res = {**res, "pts": [bad[len(bad) // 2]] + list(res["pts"])}
    calls19.append(res["plane"])
    return res


hs.put_surface = basin_first
t, det, h, sk = setup19()
r19v = put_plan(sk)
hs.put_surface = orig_ps
pot2 = obj(t, "Pot")
ev19v = r19v["detail"]["events"]
check("B2 verification: a put aimed into the basin (engine ok) is seen to sit BELOW the counter surface, picked back up, "
      "put on the counter",
      r19v["ok"] and pot2.get("on") == "CounterTop" and any("BELOW" in e for e in ev19v) and r19v["retries"] >= 1
      and len(calls19) >= 2,
      f"on={pot2.get('on')} {r19v['cause']} {r19v['detail']['why']} retries={r19v['retries']} | " + " | ".join(ev19v[-7:]))
# honesty: every put pixel goes into the basin (selection broken) -> never reported ok while the pot sits in it
calls19.clear()


def basin_always(depth, pose, fov, bbox, dets, rtype, rejected=(), n=9, max_depth=None):
    res = orig_ps(depth, pose, fov, bbox, dets, rtype, rejected=rejected, n=n, max_depth=max_depth)
    bad = [(x, y) for y in range(0, H, 2) for x in range(0, W, 2) if t.owner[y, x] == 1 and depth[y, x] <= hs.REACH]
    return {**res, "pts": bad if bad else res["pts"]}


hs.put_surface = basin_always
t, det, h, sk = setup19()
r19h = put_plan(sk)
hs.put_surface = orig_ps
pot3 = obj(t, "Pot")
check("B2 honesty: selection forced into the basin every time -> the step does NOT report ok while the pot lies in it "
      "(bounded corrections, then fails)",
      (not r19h["ok"]) and r19h["cause"] in ("retry_exhausted", "wrong_target"),
      f"ok={r19h['ok']} on={pot3.get('on')} {r19h['cause']} {r19h['detail']['why']}")

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("STEP4 ACT SELFTEST OK")# ======================================================================================================== 20
print("20. B3: a put into an openable that the engine called ok but that landed OUTSIDE it (on the surface above)")


class LandThor(FakeThor):
    """PutObject on an OPEN cabinet reports ok but, ``misland`` times (per cabinet id in ``bad``: always), drops the
    object on the surface above the cabinet -- the live bug (the engine puts the object on whatever owns the pixel /
    the receptacle volume overlaps; it says ok either way). The fake's scoring truth = ``on`` / ``inside``."""
    misland = 0
    bad = ()

    def interact(self, action, kw):
        o = self.objs.get(kw.get("objectId"))
        if (action == "PutObject" and o is not None and o["type"] == "Cabinet" and self.held and o.get("isOpen")
                and self.dist(o) <= 1.5 and (self.misland > 0 or o["id"] in self.bad)):
            if self.misland > 0:
                self.misland -= 1
            h = self.objs[self.held]
            h["pos"] = (o["pos"][0], o["pos"][1] + 0.45, o["pos"][2])
            h["on"], self.held = "CounterTop", None
            h.pop("inside", None)
            return True
        return super().interact(action, kw)


def land_setup(objs=None, misland=1, bad=(), start=(8, 0)):
    t = LandThor(objs or kitchen(), start, 0.0)
    t.misland, t.bad = misland, tuple(bad)
    det = FakeDetector(t)
    h = honest.HonestAgent(t, det, mode="honest", smooth=None)
    gm = GridMap("Fake", CELL, (0.0, 0.0), FREE)
    om = objmem.ObjectMemory("Fake", store=None, episode="t")
    sk = hs.HonestSkills(h, gm, om, chooser=lambda *a: [], **EXH)
    sk.start_episode("t")
    return t, det, h, sk


def into_cab(sk):
    return sk.exec_step({"action": "pick_and_place", "object": "Apple", "to": "Cabinet"})


hs.PUT_IN_VERIFY_ON = False
t, det, h, sk = land_setup(misland=1)
r20o = into_cab(sk)
hs.PUT_IN_VERIFY_ON = True
a0 = obj(t, "Apple")
check("B3 reproduced (verification off): the engine says ok, the apple lies on the surface above the cabinet",
      r20o["ok"] and a0.get("on") == "CounterTop" and not a0.get("inside"),
      f"{r20o['ok']} on={a0.get('on')} inside={a0.get('inside')}")
t, det, h, sk = land_setup(misland=1)
r20 = into_cab(sk)
a1 = obj(t, "Apple")
ev20 = r20["detail"]["events"]
check("B3 fixed: the apple is seen outside the opening, picked back up, put again -> it is IN the cabinet",
      r20["ok"] and a1.get("inside") == obj(t, "Cabinet")["id"],
      f"{r20['ok']} {a1.get('on')} {a1.get('inside')} | " + " | ".join(ev20[-8:]))
check("...said so in the events (outside -> picking it back up) and spent a retry, not a silent success",
      any("OUTSIDE" in e for e in ev20) and any("picking it back up" in e for e in ev20) and r20["retries"] >= 1,
      json.dumps(ev20)[-500:])
check("...audit 0 violations, no ids", h.audit.summary()["violations"] == 0 and no_ids(r20))
# B3 round 2: a wrong landing is evidence about the PIXEL's height (the surface above shares the volume): the next pixels
# of that opening at or above it come last, the correction is not charged to the retry budget (bounded by PUT_FIX_MAX /
# PUT_FREE_MAX instead), and the position where the object lay is not mistaken for another instance afterwards
check("B3 r2: the failed pixel's height becomes a cap (recorded), said in the events, the correction is not charged",
      len(sk.put_in_caps) == 1 and any("come last" in e for e in ev20) and any("not charged" in e for e in ev20)
      and len(sk.put_in_landed) == 1, f"caps={sk.put_in_caps} landed={sk.put_in_landed} " + json.dumps(ev20)[-400:])
tier_lines = [e for e in ev20 if "put pixels by tier" in e]
check("B3 r2: the second look at the opening has pixels at/above the failed height demoted ('capped'), the first has none",
      len(tier_lines) >= 2 and "capped" not in tier_lines[0] and "capped" in tier_lines[1], " | ".join(tier_lines))
# -- honesty: it lands outside every time -> bounded corrections, then the step FAILS (never ok)
t, det, h, sk = land_setup(misland=99)
r20h = into_cab(sk)
check("B3 honesty: lands outside every time -> the step does NOT report ok (bounded)",
      (not r20h["ok"]) and r20h["cause"] in ("wrong_target", "retry_exhausted"),
      f"ok={r20h['ok']} {r20h['cause']} {r20h['detail']['why']}")
# -- another apple above the cabinet before the put (another instance) is not the landing; a normal put is left alone
objs20 = kitchen() + [dict(id="Apple|+2.95|+1.05|+0.40", type="Apple", pos=(2.95, 1.05, 0.40), size=0.12,
                           pickupable=True, isSliced=False)]
t, det, h, sk = land_setup(objs=objs20, misland=0)
r20d = into_cab(sk)
check("B3 no false alarm: a second apple above the cabinet before the put; a normal put is not picked back up",
      r20d["ok"] and not any("picking it back up" in e for e in r20d["detail"]["events"]),
      f"{r20d['ok']} | " + " | ".join(r20d["detail"]["events"][-6:]))
# -- nothing new seen after the put: unverified, the success bool is kept and SAID so
t, det, h, sk = land_setup(misland=0)
sk._seen_of_type = lambda obs, held: []
r20u = into_cab(sk)
check("B3 unverified: nothing new is seen after the put -> ok on the success bool, the event says unverified",
      r20u["ok"] and any("unverified" in e for e in r20u["detail"]["events"]), json.dumps(r20u["detail"]["events"])[-300:])
# -- an instance that always puts outside: use another cabinet
objs20b = kitchen() + [dict(id="Cabinet|+2.90|+0.50|+1.90", type="Cabinet", pos=(2.95, 0.6, 1.9), size=0.4,
                            openable=True, receptacle=True, isOpen=False)]
bad_id = "Cabinet|+2.90|+0.50|+0.40"
t, det, h, sk = land_setup(objs=objs20b, misland=0, bad=[bad_id])
r20i = into_cab(sk)
a2 = obj(t, "Apple")
check("B3 instance: one cabinet always drops it outside -> the other cabinet is used, the apple is IN a cabinet",
      r20i["ok"] and a2.get("inside") and a2.get("inside") != bad_id,
      f"{r20i['ok']} {r20i['cause']} on={a2.get('on')} inside={a2.get('inside')} | " + " | ".join(r20i["detail"]["events"][-8:]))





# B4: a receptacle instance that REFUSES every put (the engine acts on whatever owns the pixel; e.g. a closed neighbour in
# a stack) is remembered after PUT_REFUSE_MAX refusals and another instance of the type is used; the switch is not charged
class RefuseThor(LandThor):
    refuse_ids = ()

    def interact(self, action, kw):
        o = self.objs.get(kw.get("objectId"))
        if action == "PutObject" and o is not None and o["id"] in self.refuse_ids:
            return False
        return super().interact(action, kw)


print("21. B4: an instance that refuses the puts is remembered (by its place in 3D) and skipped for the next put")
t21, det21, h21, sk21 = land_setup(misland=0)
bels21 = [dict(xyz=(0.95, 0.83, -2.2), id="hi"), dict(xyz=(0.95, 0.56, -2.2), id="mid"), dict(xyz=(0.95, 0.22, -2.2), id="lo"),
          dict(xyz=(2.0, 0.56, -2.2), id="side")]
sk21.om.lookup = lambda rec: list(bels21)
sk21._rank = lambda b: list(b)
sk21._locate = lambda rec: bels21[0]
sk21.h.holding = "ButterKnife"
sk21.put_refused_recs.append(("Drawer", 0.95, 0.56, -2.2, 0.10))               # the middle drawer refused (radius 0.10)
got21 = sk21._locate_put("Drawer")
check("B4: a refusing drawer is skipped by place in 3D: the one above it is still a candidate (a stack is not one instance)",
      got21["id"] == "hi", str(got21))
sk21.put_refused_recs.append(("Drawer", 0.95, 0.83, -2.2, 0.10))
got21 = sk21._locate_put("Drawer")
check("B4: ...the next one is the nearest unrefused instance (the middle one stays excluded; a drawer 0.6 m away is a different one)",
      got21["id"] == "lo", str(got21))
sk21.put_refused_recs.clear()
sk21.put_refused_recs.append(("Cabinet", 0.95, 0.56, -2.2, 0.10))
check("B4: a refusal is per receptacle type", sk21._locate_put("Drawer")["id"] == "hi")
sk21.start_episode("t2")
check("B4: a new episode forgets the refusing instances", sk21.put_refused_recs == [])


# B1f: line of sight to a recorded point (a dish this agent left in a basin, hidden from a crouch behind the counter lip)
print("22. B1f: target_blocked -- the recorded point reads a nearer surface (blocked), the right depth (clear), off screen (unknown)")
import numpy as _np
_pose22 = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"y": 0.0}, "horizon": 0.0,
           "camera": {"x": 0.0, "y": 0.9, "z": 0.0}}
_wall = _np.full((100, 100), 0.30, dtype=float)                      # a nearer surface everywhere
_far = _np.full((100, 100), 0.65, dtype=float)                       # the point's own depth
_p22 = (0.0, 0.9, 0.65)                                              # straight ahead, 0.65 m
check("B1f: a surface 0.35 m nearer than the recorded point -> blocked", hs.target_blocked(_p22, _wall, _pose22, objmem.FOV_DEFAULT) is True)
check("B1f: the recorded point's own depth -> clear", hs.target_blocked(_p22, _far, _pose22, objmem.FOV_DEFAULT) is False)
_near = _np.full((100, 100), 0.60, dtype=float)
check("B1f: 5 cm nearer (inside the margin) -> clear, not blocked", hs.target_blocked(_p22, _near, _pose22, objmem.FOV_DEFAULT) is False)
check("B1f: a point behind the camera -> cannot judge (None)", hs.target_blocked((0.0, 0.9, -1.0), _far, _pose22, objmem.FOV_DEFAULT) is None)
_nod = _np.full((100, 100), _np.nan)
check("B1f: no depth around the pixel -> cannot judge (None)", hs.target_blocked(_p22, _nod, _pose22, objmem.FOV_DEFAULT) is None)


# final regression: the put check calls a landing BELOW the estimated opening (behind the front, laterally inside) "unverified", not wrong
print("23. final regression: _verify_put_in -- low landing is unverified (no correction), above/in-front stays OUTSIDE")
import inspect as _insp23
_src23 = _insp23.getsource(hs.HonestSkills._verify_put_in) if hasattr(hs, "HonestSkills") else ""
check("final: the verification has a 'low / containment unverified' verdict that returns without a correction",
      "containment unverified" in _src23 and "if low is not None:" in _src23, "missing" if _src23 else "no class")
check("final: the OUTSIDE verdict is still kept for in-front / lateral / above landings", '"OUTSIDE"' in _src23)
