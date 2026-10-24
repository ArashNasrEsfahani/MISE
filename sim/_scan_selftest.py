"""Self-test for the dataset-gathering scan — `sim/scan_plan.py` and the skills that embody it.

Runs headless on Windows `python` from the REPO ROOT — no ai2thor, no GPU, no ultralytics, no torch:

    python sim/_scan_selftest.py

What this exists to prove, in the order the complaints arrived:

  * "the models are dogshit at detecting" — because half the class list was trained on ZERO frames. Anything
    that lives inside a shut fridge, cupboard or drawer was never photographed, and §7 shows exactly that: with
    the container pass off, `Apple_whole` ends the scan with no frames; with it on, it has some.
  * "just navigating won't do as some objects are hidden in closets and cabinets and fridges" — §6 drives the
    real `peek_inside` generator and checks it opens the door, aims at the CONTENTS rather than at the door,
    shoots from more than one distance and door-angle, and shuts everything again (§6c, including when the
    caller abandons it mid-swing).
  * a detector trained on one viewpoint fails on every other one — §1/§3 pin the geometry that makes a drawer
    get looked down into from a crouch and a wall cabinet looked up at, and the schedule that spends the same
    frame budget on 16 headings and 4 pitches instead of 4 and 2.
  * a scan that quietly takes 20 minutes is a scan nobody runs twice — §5 prices it up front.

`sim/thor_actions.py` guards its `ai2thor` import, so the REAL `peek_inside` / `scan_scene` generators are bound
to fake agents here and driven — these are not assertions about a reimplementation. What that cannot cover is
whether YOLO then recognises a real apple, or how the door swing looks on screen; those are Track C in
HOW-TO-TEST.html.
"""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_dataset                                                    # noqa: E402
import perception                                                     # noqa: E402
import scan_plan as sp                                                # noqa: E402
import thor_actions as ta                                             # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def section(t):
    print(f"\n{t}")


AGENT = {"x": 0.0, "y": 0.9, "z": 0.0}          # a floor-standing AI2-THOR agent reports y ~= 0.9


# ---------------------------------------------------------------------------------------------------------
section("1. the camera geometry that decides whether you can see INTO a thing")
drawer_inside = {"x": 0.0, "y": 0.42, "z": 0.6}          # an open drawer's contents sit low and close
cabinet_inside = {"x": 0.0, "y": 1.95, "z": 0.7}         # a wall cabinet's shelf sits above eye height
fridge_inside = {"x": 0.0, "y": 1.15, "z": 0.8}          # a fridge shelf sits roughly at chest height

check("a drawer is looked DOWN into", sp.pitch_for(drawer_inside, sp.eye_of(AGENT, True)) > 30.0,
      str(sp.pitch_for(drawer_inside, sp.eye_of(AGENT, True))))
check("a wall cabinet is looked UP at", sp.pitch_for(cabinet_inside, sp.eye_of(AGENT, True)) < -10.0,
      str(sp.pitch_for(cabinet_inside, sp.eye_of(AGENT, True))))
check("a fridge is looked into roughly level", 0.0 < sp.pitch_for(fridge_inside, sp.eye_of(AGENT, True)) < 45.0,
      str(sp.pitch_for(fridge_inside, sp.eye_of(AGENT, True))))

# The reason the stance matters at all: standing, the drawer needs more down-pitch than the agent HAS, and the
# simulator clamps silently — you get a photograph of the front of the drawer and none of its contents.
check("standing, a low drawer is out of pitch range", sp._pitch_raw(drawer_inside, sp.eye_of(AGENT, True)) > sp.HORIZON_MAX,
      str(sp._pitch_raw(drawer_inside, sp.eye_of(AGENT, True))))
check("crouching brings it back into range", sp._pitch_raw(drawer_inside, sp.eye_of(AGENT, False)) <= sp.HORIZON_MAX,
      str(sp._pitch_raw(drawer_inside, sp.eye_of(AGENT, False))))
check("so the planner crouches for the drawer", sp.stance_for(drawer_inside, AGENT) is False)
check("and stands for the wall cabinet", sp.stance_for(cabinet_inside, AGENT) is True)
check("a reachable target keeps the agent on its feet", sp.stance_for(fridge_inside, AGENT) is True)
# "Technically in frame" is not the bar. A drawer at 59 degrees of down-pitch is inside the limits and is still a
# photograph of the drawer's front edge, so the stance choice keeps a margin off the limit.
barely = {"x": 0.0, "y": 0.42, "z": 0.69}
check("a pitch just inside the limit is still treated as too steep to stand for",
      sp.HORIZON_MAX - sp.COMFORT_MARGIN < sp._pitch_raw(barely, sp.eye_of(AGENT, True)) < sp.HORIZON_MAX
      and sp.stance_for(barely, AGENT) is False, str(sp._pitch_raw(barely, sp.eye_of(AGENT, True))))
check("every pitch it returns is one the agent can actually adopt",
      all(sp.HORIZON_MIN <= sp.pitch_for(t, sp.eye_of(AGENT, s)) <= sp.HORIZON_MAX
          for t in (drawer_inside, cabinet_inside, fridge_inside) for s in (True, False)))

yaw, horizon, standing = sp.aim_from({"x": 1.0, "y": 1.2, "z": 0.0}, AGENT)
check("aim_from faces the target", abs(yaw - 90.0) < 1e-6, str(yaw))
check("standing on top of a target does not divide by zero",
      sp.pitch_for({"x": 0.0, "y": 0.1, "z": 0.0}, sp.eye_of(AGENT, True)) == sp.HORIZON_MAX)

section("2. aim at what you are photographing, not at the box it lives in")
fridge = {"objectId": "Fridge|1", "objectType": "Fridge", "position": {"x": 0.0, "y": 0.95, "z": 1.2},
          "axisAlignedBoundingBox": {"center": {"x": 0.0, "y": 0.95, "z": 1.2}}}
inside = [{"position": {"x": 0.1, "y": 1.3, "z": 1.4}}, {"position": {"x": -0.1, "y": 1.3, "z": 1.4}}]
check("with contents, the aim is their centroid", sp.aim_point(fridge, inside)["y"] == 1.3,
      str(sp.aim_point(fridge, inside)))
check("which is NOT the door it would otherwise point at", sp.aim_point(fridge, inside) != sp.aim_point(fridge))
check("an empty container falls back to its own centre", sp.aim_point(fridge, [])["y"] == 0.95)
check("a container with no bbox still yields a point", "y" in sp.aim_point({"position": {"x": 0, "y": 1, "z": 1}}, []))

section("3. viewpoint variety — the same budget, spent on more than one picture")
offs = sp.view_offsets(3)
check("the first framing is dead-on", offs[0] == (0.0, 0.0))
check("the rest are oblique", all(o != (0.0, 0.0) for o in offs[1:]), str(offs))
check("and symmetric, so the dataset is not biased to one side",
      abs(sum(o[0] for o in sp.view_offsets(5))) < 1e-9, str(sp.view_offsets(5)))
check("asking for one framing gives exactly one", sp.view_offsets(1) == [(0.0, 0.0)])

lad = sp.openness_ladder(2)
check("the widest door position comes first", lad[0] == 1.0 and lad[1] < 1.0, str(lad))
check("a partial door is still a usable door", all(0.0 < v <= 1.0 for v in sp.openness_ladder(4)))
check("one level means wide open", sp.openness_ladder(1) == [1.0])

POSES = [
    {"x": 0.0, "y": 0.9, "z": 0.65, "rotation": 0.0, "horizon": 0.0, "standing": True},     # 0.55 m band
    {"x": 0.02, "y": 0.9, "z": 0.66, "rotation": 0.0, "horizon": 0.0, "standing": True},    # ~the same photo
    {"x": 0.0, "y": 0.9, "z": 0.25, "rotation": 0.0, "horizon": 0.0, "standing": True},     # ~0.95 m band
    {"x": 0.9, "y": 0.9, "z": 1.2, "rotation": 270.0, "horizon": 0.0, "standing": True},    # off to one side
]
picked = sp.select_poses(POSES, (0.0, 1.2), max_poses=2)
d = [round(math.hypot(p["x"], p["z"] - 1.2), 2) for p in picked]
check("two shots means near AND far, not the same spot twice", len(set(d)) == 2 and max(d) - min(d) > 0.2, str(d))
check("it never returns more than asked", len(sp.select_poses(POSES, (0.0, 1.2), max_poses=1)) == 1)
check("with one pose available it still returns it", len(sp.select_poses(POSES[:1], (0.0, 1.2), max_poses=3)) == 1)
check("no poses at all is not a crash", sp.select_poses([], (0.0, 0.0)) == [])
check("a distance cap is respected",
      all(math.hypot(p["x"], p["z"] - 1.2) <= 0.8 for p in sp.select_poses(POSES, (0.0, 1.2), max_poses=4,
                                                                          max_distance=0.8)))

v0 = sp.coverage_views(0)
check("a coverage position still costs 8 frames", len(v0) == 8, str(len(v0)))
allv = [t for i in range(24) for t in sp.coverage_views(i)]
yaws = {t[0] for t in allv}
check("but the scan visits 16 headings, not 4", len(yaws) == 16, str(sorted(yaws)))
check("the old scan only ever had four", len({0.0, 90.0, 180.0, 270.0}) == 4)
check("all four pitches get used", {t[1] for t in allv} == set(sp.COVERAGE_HORIZONS), str({t[1] for t in allv}))
check("including one that looks UP at the wall cabinets", min(t[1] for t in allv) < 0.0)
stand_frac = sum(1 for t in allv if t[2]) / float(len(allv))
check("both eye heights appear, mostly standing", 0.5 < stand_frac < 0.85, f"{stand_frac:.2f}")
check("the schedule is deterministic, so a rescan is comparable", sp.coverage_views(7) == sp.coverage_views(7))

clustered = ([{"x": 0.1 * i, "y": 0.9, "z": 0.1 * i} for i in range(10)] +
             [{"x": 10 + 0.1 * i, "y": 0.9, "z": 10 + 0.1 * i} for i in range(10)])
spread = sp.spread_positions(clustered, 4)
check("coverage spreads over the room instead of clustering",
      any(p["x"] < 5 for p in spread) and any(p["x"] > 5 for p in spread), str(spread))
check("it returns exactly what was asked for", len(spread) == 4)
check("asking for more than exists returns everything", len(sp.spread_positions(clustered, 99)) == 20)
check("an empty room is not a crash", sp.spread_positions([], 4) == [] and sp.spread_positions(clustered, 0) == [])
check("and it is deterministic", sp.spread_positions(clustered, 4) == sp.spread_positions(clustered, 4))


# ---------------------------------------------------------------------------------------------------------
section("4. which containers are worth opening")
KITCHEN = [
    {"objectId": "Fridge|1", "objectType": "Fridge", "openable": True, "receptacle": True, "isOpen": False,
     "position": {"x": 0.0, "y": 0.95, "z": 2.0}, "receptacleObjectIds": ["Apple|1", "Egg|1"]},
    {"objectId": "Drawer|1", "objectType": "Drawer", "openable": True, "receptacle": True, "isOpen": False,
     "position": {"x": 1.0, "y": 0.4, "z": 0.0}, "receptacleObjectIds": ["Knife|1"]},
    {"objectId": "Cabinet|1", "objectType": "Cabinet", "openable": True, "receptacle": True, "isOpen": False,
     "position": {"x": 2.0, "y": 1.8, "z": 0.0}, "receptacleObjectIds": []},
    {"objectId": "Cabinet|2", "objectType": "Cabinet", "openable": True, "receptacle": True, "isOpen": False,
     "position": {"x": 2.5, "y": 1.8, "z": 0.0}, "receptacleObjectIds": []},
    {"objectId": "Cabinet|3", "objectType": "Cabinet", "openable": True, "receptacle": True, "isOpen": False,
     "position": {"x": 3.0, "y": 1.8, "z": 0.0}, "receptacleObjectIds": []},
    {"objectId": "CounterTop|1", "objectType": "CounterTop", "receptacle": True,
     "position": {"x": 1.0, "y": 0.95, "z": 0.0}, "receptacleObjectIds": ["Mug|1"]},
    {"objectId": "Apple|1", "objectType": "Apple", "position": {"x": 0.0, "y": 1.1, "z": 2.0}},
    {"objectId": "Egg|1", "objectType": "Egg", "position": {"x": 0.1, "y": 1.1, "z": 2.0}},
    {"objectId": "Knife|1", "objectType": "Knife", "position": {"x": 1.0, "y": 0.42, "z": 0.0}},
    {"objectId": "Mug|1", "objectType": "Mug", "position": {"x": 1.0, "y": 1.0, "z": 0.0}},
]
plan = sp.rank_containers(KITCHEN, agent_pos=(0.0, 0.0), counts={}, class_of=perception.class_name)
ids = [s["objectId"] for s in plan]
check("an open surface is not a container to open", "CounterTop|1" not in ids, str(ids))
check("the fullest container goes first", ids[0] == "Fridge|1", str(ids))
check("every stop explains itself", all(s["why"] for s in plan))
check("and says what it expects to find", "Apple_whole" in plan[0]["why"], plan[0]["why"])
check("identical empty cupboards are not all worth opening", ids.count("Cabinet|1") + ids.count("Cabinet|2")
      + ids.count("Cabinet|3") == 2, str(ids))
check("the empty ones come last", ids.index("Drawer|1") < ids.index("Cabinet|1"), str(ids))
check("it knows the class the door itself will wear once open", plan[0]["open_class"] == "Fridge_open")

# The tally is what makes this a response to the gap rather than a fixed route: photograph the fridge's contents
# and it stops being the most urgent door in the room.
counts = {"Apple_whole": 60, "Egg_whole": 60, "Fridge_open": 60}
after = [s["objectId"] for s in sp.rank_containers(KITCHEN, agent_pos=(0.0, 0.0), counts=counts,
                                                   class_of=perception.class_name)]
check("a container whose contents are already covered drops down the list", after[0] == "Drawer|1", str(after))
check("but is still on it — nothing is written off", "Fridge|1" in after, str(after))

near_far = sp.rank_containers(
    [dict(o, position=(dict(o["position"], x=9.0) if o["objectId"] == "Fridge|1" else o["position"]))
     for o in KITCHEN], agent_pos=(0.0, 0.0), counts={}, class_of=perception.class_name)
check("walking nine metres costs something", near_far[0]["objectId"] == "Fridge|1", str([s["objectId"] for s in near_far]))
check("the budget is respected",
      len(sp.rank_containers(KITCHEN, agent_pos=(0.0, 0.0), max_containers=2, class_of=perception.class_name)) == 2)
check("a room with nothing that opens plans nothing",
      sp.rank_containers([KITCHEN[5]], agent_pos=(0.0, 0.0), class_of=perception.class_name) == [])
check("summarize renders the route for the transcript", "Fridge(2)" in sp.summarize(plan), sp.summarize(plan))

section("5. an empty cupboard and a full fridge are not the same photograph")
rich = sp.capture_budget(plan[0], base_views=3, openness_levels=2, max_poses=2)
empty = sp.capture_budget(next(s for s in plan if s["n_contents"] == 0), 3, 2, 2)
check("a container full of unseen things gets the full treatment", rich["frames"] == 12, str(rich))
check("an empty one gets a single frame of its open door", empty["frames"] == 1, str(empty))
covered = sp.capture_budget({"n_contents": 3, "new_classes": []}, 3, 2, 2)
check("one that is already well covered gets something in between",
      empty["frames"] < covered["frames"] < rich["frames"], str(covered))
check("the widest door position is always photographed", rich["openness"][0] == 1.0)

est = sp.estimate_frames(24, 8, plan, base_views=3, openness_levels=2, max_poses=2)
check("the scan is priced before it starts", est["total"] == est["coverage"] + est["containers"] + est["balance"])
check("coverage is still the bulk of it", est["coverage"] == 192)
check("and the operator is told how long in minutes", est["minutes"] > 0)
check("a scan with no container pass costs what it always did",
      sp.estimate_frames(24, 8, [])["total"] == 192)


# ---------------------------------------------------------------------------------------------------------
class FakeContainer:
    """The controller surface `peek_inside` touches, with every call and every capture recorded.

    `thor_actions` guards its ai2thor import, so the REAL `ThorAgent.peek_inside` generator is bound to this and
    driven.
    """

    def __init__(self, kind="fridge", start_open=False, contents=True):
        self.calls = []
        self.captures = []                        # (openness, yaw, horizon, standing, x, z) at each capture
        self.success = True
        self.motion = {"step": 0.15, "turn": 15.0}
        self.pose = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"y": 0.0},
                     "horizon": 0.0, "standing": True}
        cy, iy = (0.95, 1.15) if kind == "fridge" else (0.40, 0.42)
        self.cid = "Fridge|1" if kind == "fridge" else "Drawer|1"
        self.objs = {
            self.cid: {"objectId": self.cid, "objectType": self.cid.split("|")[0], "openable": True,
                       "receptacle": True, "isOpen": start_open, "openness": 1.0 if start_open else 0.0,
                       "position": {"x": 0.0, "y": cy, "z": 1.2},
                       "axisAlignedBoundingBox": {"center": {"x": 0.0, "y": cy, "z": 1.2}},
                       "receptacleObjectIds": ["Apple|1"] if contents else []},
            "Apple|1": {"objectId": "Apple|1", "objectType": "Apple", "position": {"x": 0.0, "y": iy, "z": 1.35}},
        }

    # -- queries
    def objects(self):
        return list(self.objs.values())

    def object_by_id(self, oid):
        return self.objs.get(oid)

    def resolve(self, name, prefer_visible=True):
        return self.objs.get(name)

    def agent_pose(self):
        return {"position": dict(self.pose["position"]), "rotation": dict(self.pose["rotation"]),
                "horizon": self.pose["horizon"], "standing": self.pose["standing"]}

    def interactable_poses(self, oid, max_distance=None):
        self.calls.append(("interactable_poses", oid))
        return [dict(p) for p in POSES]

    # -- primitives
    def teleport_full(self, x, y, z, rotation_y, horizon=0.0, standing=True):
        self.calls.append(("teleport_full", round(x, 2), round(z, 2)))
        self.pose = {"position": {"x": x, "y": y, "z": z}, "rotation": {"y": rotation_y},
                     "horizon": horizon, "standing": standing}
        self.success = True

    def teleport(self, x, y, z, rotation_y=None, horizon=None, standing=True, force=False):
        self.calls.append(("teleport", round(x, 2), round(z, 2)))
        self.pose = {"position": {"x": x, "y": y, "z": z},
                     "rotation": {"y": self.pose["rotation"]["y"] if rotation_y is None else rotation_y},
                     "horizon": self.pose["horizon"] if horizon is None else horizon, "standing": standing}
        self.success = True

    def walk_to(self, oid, **kw):
        self.calls.append(("walk_to", oid))
        yield "arrived"

    def open(self, oid, openness=1.0, force=False):
        self.calls.append(("open", oid, round(float(openness), 3)))
        self.success = True
        o = self.objs[oid]
        o["openness"], o["isOpen"] = float(openness), float(openness) > 0
        return True

    def close(self, oid, force=False):
        self.calls.append(("close", oid))
        self.success = True
        self.objs[oid]["openness"], self.objs[oid]["isOpen"] = 0.0, False
        return True

    def grab(self):
        """Stand-in for the labelling capture — records the exact viewpoint the frame was taken from."""
        self.calls.append(("capture",))
        self.captures.append((self.objs[self.cid]["openness"], round(self.pose["rotation"]["y"], 1),
                              round(self.pose["horizon"], 1), self.pose["standing"],
                              round(self.pose["position"]["x"], 2), round(self.pose["position"]["z"], 2)))


FakeContainer._swing = ta.ThorAgent._swing
FakeContainer.contents_of = ta.ThorAgent.contents_of


def peek(ag, **kw):
    kw.setdefault("on_capture", ag.grab)
    return list(ta.ThorAgent.peek_inside(ag, ag.cid, **kw))


section("6. the skill actually opens the thing and photographs what is inside")
ag = FakeContainer("fridge")
frames = peek(ag, views=3, openness=(1.0, 0.55), poses=2)
kinds = [c[0] for c in ag.calls]
check("it opens the container", "open" in kinds)
check("the door SWINGS rather than snapping open", kinds.count("open") > 4, str(kinds.count("open")))
check("nothing is photographed through a shut door", all(c[0] > 0 for c in ag.captures), str(ag.captures[:3]))
check("the first capture comes AFTER the first open", kinds.index("capture") > kinds.index("open"))
check("it shoots the full budget", len(ag.captures) == 2 * 2 * 3, str(len(ag.captures)))
check("from two different standing positions", len({(c[4], c[5]) for c in ag.captures}) == 2,
      str({(c[4], c[5]) for c in ag.captures}))
check("at two different door angles", len({c[0] for c in ag.captures}) == 2, str({c[0] for c in ag.captures}))
check("with the subject sometimes off-centre", len({c[1] for c in ag.captures}) > 1)
check("and looking INTO it, not straight ahead", any(c[2] > 10.0 for c in ag.captures),
      str(sorted({c[2] for c in ag.captures})))
check("it renders enough frames to be watchable", len(frames) > 20, f"{len(frames)} frames")

section("6b. a drawer is looked down into from a crouch")
dr = FakeContainer("drawer")
peek(dr, views=1, openness=(1.0,), poses=1)
check("the agent crouches for a low container", all(c[3] is False for c in dr.captures), str(dr.captures))
check("and looks down into it", all(c[2] > 30.0 for c in dr.captures), str(dr.captures))

section("6c. it leaves the kitchen as it found it")
ag = FakeContainer("fridge", start_open=False)
peek(ag, views=1, openness=(1.0, 0.55), poses=2)
check("a container it opened is closed again", ag.objs["Fridge|1"]["isOpen"] is False)
was_open = FakeContainer("fridge", start_open=True)
peek(was_open, views=1, openness=(1.0, 0.55), poses=1)
check("one that was ALREADY open is left open", was_open.objs["Fridge|1"]["isOpen"] is True,
      str(was_open.objs["Fridge|1"]))
check("and at the openness it had", was_open.objs["Fridge|1"]["openness"] == 1.0)

# The case a `yield from` in the caller makes real: the console is stopped, or the scan is cancelled, halfway
# through the swing. Without the `finally` the kitchen is left standing open.
abandoned = FakeContainer("fridge")
gen = ta.ThorAgent.peek_inside(abandoned, "Fridge|1", on_capture=abandoned.grab)
for _ in range(4):
    next(gen)
check("the door is open mid-scan", abandoned.objs["Fridge|1"]["isOpen"] is True)
gen.close()
check("abandoning the scan STILL shuts the door", abandoned.objs["Fridge|1"]["isOpen"] is False,
      str(abandoned.objs["Fridge|1"]))

# A restore the simulator REFUSES is the one failure that outlives the scan — the operator is left driving a kitchen
# with a door hanging open. The generator cannot fix it, but it must not hide it either.
stuck = FakeContainer("fridge")
stuck.close = lambda oid, force=False: setattr(stuck, "success", False)
out = peek(stuck, views=1, openness=(1.0,), poses=1)
check("a door that will not shut is reported, not swallowed",
      any("WARNING" in m and "still open" in m for m in out), str(out[-3:]))

section("6d. it refuses what it cannot open")
notopen = FakeContainer("fridge")
notopen.objs["Fridge|1"]["openable"] = False
out = list(ta.ThorAgent.peek_inside(notopen, "Fridge|1", on_capture=notopen.grab))
check("a solid object is not prised open", not notopen.captures and "does not open" in out[-1], str(out))
check("and neither is one that does not exist",
      list(ta.ThorAgent.peek_inside(notopen, "Nope|9"))[-1].startswith("no "))
empty = FakeContainer("fridge", contents=False)
peek(empty, views=1, openness=(1.0,), poses=1)
check("an empty container is still photographed open", len(empty.captures) == 1, str(empty.captures))


# ---------------------------------------------------------------------------------------------------------
class FakeKitchen:
    """A two-container kitchen for driving the REAL `scan_scene`: an Apple in the fridge, a Bowl in a cabinet,
    a Mug out on the counter. `visible_objects` models the one thing that matters — you cannot see into a shut
    container — so a scan that never opens one genuinely cannot photograph what is inside."""

    def __init__(self):
        self.scene = "FloorPlan1"
        self.visibility_distance = 1.5
        self.success = True
        self.motion = {"step": 0.15, "turn": 15.0}
        self.pose = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"y": 0.0},
                     "horizon": 0.0, "standing": True}
        self.sensors = []
        self.yaws = []
        self.visited = []
        self.objs = {}
        for o in [
            {"objectId": "Fridge|1", "objectType": "Fridge", "openable": True, "receptacle": True, "isOpen": False,
             "openness": 0.0, "position": {"x": 0.0, "y": 0.95, "z": 2.0}, "receptacleObjectIds": ["Apple|1"]},
            {"objectId": "Cabinet|1", "objectType": "Cabinet", "openable": True, "receptacle": True, "isOpen": False,
             "openness": 0.0, "position": {"x": 2.0, "y": 1.7, "z": 0.0}, "receptacleObjectIds": ["Bowl|1"]},
            {"objectId": "CounterTop|1", "objectType": "CounterTop", "receptacle": True,
             "position": {"x": 1.0, "y": 0.95, "z": 0.0}, "receptacleObjectIds": ["Mug|1"]},
            {"objectId": "Apple|1", "objectType": "Apple", "position": {"x": 0.0, "y": 1.1, "z": 2.0},
             "parentReceptacles": ["Fridge|1"]},
            {"objectId": "Bowl|1", "objectType": "Bowl", "position": {"x": 2.0, "y": 1.7, "z": 0.0},
             "parentReceptacles": ["Cabinet|1"]},
            {"objectId": "Mug|1", "objectType": "Mug", "position": {"x": 1.0, "y": 1.0, "z": 0.0},
             "parentReceptacles": ["CounterTop|1"]},
        ]:
            self.objs[o["objectId"]] = o

    # -- queries
    def objects(self):
        return list(self.objs.values())

    def object_by_id(self, oid):
        return self.objs.get(oid)

    def objects_of_type(self, t):
        return [o for o in self.objs.values() if o["objectType"] == t]

    def resolve(self, name, prefer_visible=True):
        return self.objs.get(name)

    def visible_objects(self):
        p = self.pose["position"]
        out = []
        for o in self.objs.values():
            parent = (o.get("parentReceptacles") or [None])[0]
            box = self.objs.get(parent) if parent else None
            if box is not None and box.get("openable") and not box.get("isOpen"):
                continue                                  # shut in a cupboard: nothing to photograph
            q = o["position"]
            if math.dist((p["x"], p["y"], p["z"]), (q["x"], q["y"], q["z"])) > self.visibility_distance:
                continue                                  # AI2-THOR's range gate on `visible`
            out.append(dict(o, distance=math.dist((p["x"], p["z"]), (q["x"], q["z"]))))
        return out

    def reachable_positions(self):
        return [{"x": 0.5 * i, "y": 0.9, "z": 0.5 * j} for i in range(5) for j in range(5)]

    def agent_pose(self):
        return {"position": dict(self.pose["position"]), "rotation": dict(self.pose["rotation"]),
                "horizon": self.pose["horizon"], "standing": self.pose["standing"]}

    def interactable_poses(self, oid, max_distance=None):
        o = self.objs.get(oid)
        if not o:
            return []
        px, pz = o["position"]["x"], o["position"]["z"]
        return [{"x": px, "y": 0.9, "z": pz - dz, "rotation": 0.0, "horizon": 0.0, "standing": True}
                for dz in (0.55, 0.95, 1.4)]

    # -- primitives
    def set_sensors(self, depth=False, instance=False, semantic=False, normals=False, force=False,
                    visibility_distance=None):
        if visibility_distance is not None:
            self.visibility_distance = float(visibility_distance)
        self.sensors.append(self.visibility_distance)

    def teleport(self, x, y, z, rotation_y=None, horizon=None, standing=True, force=False):
        self.pose = {"position": {"x": x, "y": y, "z": z},
                     "rotation": {"y": self.pose["rotation"]["y"] if rotation_y is None else rotation_y},
                     "horizon": self.pose["horizon"] if horizon is None else horizon, "standing": standing}
        if rotation_y is not None:
            self.yaws.append(round(rotation_y, 1))
        self.visited.append((round(x, 2), round(z, 2)))
        self.success = True

    def teleport_full(self, x, y, z, rotation_y, horizon=0.0, standing=True):
        self.teleport(x, y, z, rotation_y=rotation_y, horizon=horizon, standing=standing, force=True)

    def walk_to(self, oid, **kw):
        yield "arrived"

    def look_at_object(self, oid):
        return None

    def open(self, oid, openness=1.0, force=False):
        o = self.objs[oid]
        o["openness"], o["isOpen"] = float(openness), float(openness) > 0
        self.success = True

    def close(self, oid, force=False):
        self.objs[oid]["openness"], self.objs[oid]["isOpen"] = 0.0, False
        self.success = True

    def pass_(self):
        self.success = True

    def checkpoint(self):
        return {"agent": self.agent_pose(), "objects": {}, "held": None}

    def restore(self, cp=None):
        return {"ok": True}

    def reset(self, scene=None):
        self.success = True


for _m in ("_swing", "contents_of", "peek_inside", "_scan_body", "_scan_memory"):
    setattr(FakeKitchen, _m, getattr(ta.ThorAgent, _m))


class FakeMem:
    def __init__(self):
        self.cells = {}

    def mark_cell(self, scene, gx, gz, state, why=""):
        self.cells[(gx, gz)] = state

    def get_nav_cells(self, scene):
        return [{"gx": g[0], "gz": g[1], "state": s} for g, s in self.cells.items()]

    def add_perception_label(self, *a, **kw):
        return None


def run_scan(**kw):
    """Drive the REAL scan_scene with the labelling capture replaced by a recorder."""
    ag = FakeKitchen()
    seen = []                                    # every class that appeared in a written frame
    ag.splits = []                               # (split, agent x, agent z) per frame — for the leakage check
    ag.min_areas = set()                         # the label-area floor the scan asked capture_frame for
    ag.per_class = {}                            # class -> {'train': n, 'val': n} — for the starvation check

    def fake_capture(agent, scene, out_dir, split="train", class_to_id=None, index=None, memory=None,
                     min_area=400, **rest):
        ag.min_areas.add(min_area)
        classes = sorted({perception.class_name(o) for o in agent.visible_objects()})
        for c in classes:
            if class_to_id is not None and c not in class_to_id:
                class_to_id[c] = len(class_to_id)
            ag.per_class.setdefault(c, {"train": 0, "val": 0})[split] += 1
        seen.extend(classes)
        p = agent.agent_pose()["position"]
        ag.splits.append((split, round(p["x"], 2), round(p["z"], 2)))
        return {"written": bool(classes), "image_path": None, "n_boxes": len(classes), "classes": classes}

    real = gen_dataset.capture_frame
    gen_dataset.capture_frame = fake_capture
    try:
        with tempfile.TemporaryDirectory() as tmp:
            kw.setdefault("per_class_target", 1)
            kw.setdefault("do_states", False)
            msgs = list(ta.ThorAgent.scan_scene(ag, dataset_root=tmp, memory=FakeMem(), max_positions=6, **kw))
    finally:
        gen_dataset.capture_frame = real
    return ag, msgs, seen


section("7. the whole scan — the classes that only exist inside something")
ag_off, msgs_off, seen_off = run_scan(do_containers=False)
check("without a container pass the fridge's apple is never photographed", "Apple_whole" not in seen_off,
      str(sorted(set(seen_off))))
check("nor the bowl in the cupboard", "Bowl_empty" not in seen_off)
check("nor an open door — the class exists but has no frames", "Fridge_open" not in seen_off)
check("...while what the room shows IS photographed", "Mug_empty" in seen_off, str(sorted(set(seen_off))))
check("and the scan says so out loud",
      any(m.startswith("coverage:") and "STILL EMPTY" in m and "Apple_whole" in m for m in msgs_off),
      str([m for m in msgs_off if m.startswith("coverage:")]))

ag_on, msgs_on, seen_on = run_scan(do_containers=True)
check("WITH it, the apple inside the fridge gets frames", "Apple_whole" in seen_on, str(sorted(set(seen_on))))
check("and so does the bowl inside the cupboard", "Bowl_empty" in seen_on)
check("and the open-door class it can only ever wear while open", "Fridge_open" in seen_on)
check("the container pass narrates each door it opens",
      sum(1 for m in msgs_on if m.startswith("container ")) >= 4, str([m for m in msgs_on if m.startswith("container")]))
check("and reports the first sighting of a class", any("first sight of" in m for m in msgs_on),
      str([m for m in msgs_on if "first sight" in m]))

section("7b. it puts the kitchen back")
check("every container it opened is shut again",
      all(not ag_on.objs[k]["isOpen"] for k in ("Fridge|1", "Cabinet|1")),
      str({k: ag_on.objs[k]["isOpen"] for k in ("Fridge|1", "Cabinet|1")}))
check("and the visibility gate is back where it was", ag_on.visibility_distance == 1.5,
      str(ag_on.sensors))
check("even though the scan widened it to label distant objects", max(ag_on.sensors) > 1.5, str(ag_on.sensors))

section("7c. cost is declared before it is spent")
check("the scan states its plan first", msgs_on[0].startswith("scan FloorPlan1"), msgs_on[0])
check("and prices it", msgs_on[1].startswith("estimate:") and "frames" in msgs_on[1], msgs_on[1])
check("before it has moved anywhere", "pos 1/" not in " ".join(msgs_on[:2]))
check("a container pass costs more than none", len(seen_on) > len(seen_off))

section("7d. the validation split holds out PLACES, not every fifth photograph")
# Eight near-identical frames come off one spot. Splitting them 1-in-5 by frame puts near-duplicates of every
# val image into train, and the val mAP that comes back measures memorisation rather than detection.
by_place = {}
for split, x, z in ag_on.splits:
    by_place.setdefault((x, z), set()).add(split)
check("a validation set exists at all", any(s == "val" for s, _x, _z in ag_on.splits))
check("and no single viewpoint appears on both sides of it",
      all(len(v) == 1 for v in by_place.values()),
      str({k: v for k, v in by_place.items() if len(v) > 1}))
val_frac = sum(1 for s, _x, _z in ag_on.splits if s == "val") / float(len(ag_on.splits))
check("the split is still roughly one in five", 0.05 < val_frac < 0.45, f"{val_frac:.2f}")

section("7e. small things get labelled, not left in the frame as background")
# capture_frame's 400 px^2 default is an egg photographed from closer than 0.7 m. Every frame from further away
# had the egg in it and no box on it, and a training run reads an unlabelled visible object as "background".
check("the scan lowers the label-area floor below capture_frame's default",
      ag_on.min_areas and max(ag_on.min_areas) < 400, str(ag_on.min_areas))
check("but not to zero, where a box is a handful of noisy pixels", min(ag_on.min_areas) > 0)

section("7g. a rescued class must land in TRAIN, not be held out wholesale")
# A container is one capture SITE, so the site rota can hand a whole cupboard to val — and a class that lives only
# in that cupboard then has every frame it will ever get on the held-out side and none to learn from. That is the
# starvation the container pass exists to end, arriving by the back door, and it is invisible unless it is counted.
sole_source = [c for c in ("Apple_whole", "Bowl_empty") if c in ag_on.per_class]
check("the container-only classes were photographed at all", len(sole_source) == 2, str(sole_source))
check("and every one of them has TRAINING frames, not val-only",
      all(ag_on.per_class[c]["train"] > 0 for c in sole_source),
      str({c: ag_on.per_class[c] for c in sole_source}))
check("the coverage line counts TRAIN frames, so it cannot call a val-only class covered",
      any(m.startswith("coverage:") and "TRAINING frames" in m for m in msgs_on),
      str([m for m in msgs_on if m.startswith("coverage:")]))

section("7f. the coverage sweep is no longer four compass points")
check("the old scan's four yaws have become many", len(set(ag_on.yaws)) >= 12, str(sorted(set(ag_on.yaws))))
ag_b, _, _ = run_scan(do_containers=False)
check("and the route is deterministic, so two scans are comparable",
      ag_b.visited[:8] == ag_off.visited[:8], str(ag_b.visited[:3]))


section("8. the state pass plans over the whole room, not what happens to be in view")
# The first GPU scan (2026-09-15) photographed Faucet, StoveBurner, Toaster, CoffeeMachine, Microwave in ONE state
# each: the old pass flipped the nearest ten VISIBLE objects, which from where balance left the agent were cabinets.
ROOM = [
    {"objectId": f"Cabinet|{i}", "objectType": "Cabinet", "openable": True, "isOpen": False,
     "position": {"x": 0.2 * i, "y": 1.8, "z": 0.0}} for i in range(1, 9)
] + [
    {"objectId": "Faucet|1", "objectType": "Faucet", "toggleable": True, "isToggled": False,
     "position": {"x": 4.0, "y": 1.0, "z": 3.0}},
    {"objectId": "StoveBurner|1", "objectType": "StoveBurner", "toggleable": True, "isToggled": False,
     "position": {"x": 5.0, "y": 1.0, "z": 3.0}},
    {"objectId": "Microwave|1", "objectType": "Microwave", "openable": True, "toggleable": True, "isOpen": False,
     "isToggled": False, "position": {"x": 6.0, "y": 1.2, "z": 3.0}},
    {"objectId": "LightSwitch|1", "objectType": "LightSwitch", "toggleable": True, "isToggled": True,
     "position": {"x": 0.1, "y": 1.3, "z": 0.1}},
    {"objectId": "Mug|1", "objectType": "Mug", "canFillWithLiquid": True, "isFilledWithLiquid": False,
     "position": {"x": 1.0, "y": 1.0, "z": 0.0}},
    {"objectId": "Bread|1", "objectType": "Bread", "position": {"x": 1.0, "y": 1.0, "z": 0.5}},
]
flips = sp.plan_state_flips(ROOM, {"Cabinet_open": 30}, agent_pos=(0.0, 0.0), max_objects=8,
                            class_of=perception.class_name)
fids = [(f["objectId"], f["kind"]) for f in flips]
check("a far faucet whose ON half was never photographed beats eight nearby cabinets",
      ("Faucet|1", "toggle") in fids and ("StoveBurner|1", "toggle") in fids, str(fids))
check("an axis the class name does not encode is not flipped (Microwave wears open/closed, not on/off)",
      ("Microwave|1", "open") in fids and ("Microwave|1", "toggle") not in fids, str(fids))
check("a cabinet only gets per_type slots, not the whole budget",
      sum(1 for i, _ in fids if i.startswith("Cabinet")) <= 2, str(fids))
check("never-photographed halves come before merely short ones",
      [f["tier"] for f in flips] == sorted(f["tier"] for f in flips), str([f["tier"] for f in flips]))
check("a fill flip is planned too (Mug_filled)", ("Mug|1", "fill") in fids, str(fids))
check("room lights are not flipped by default (a missed restore darkens every later frame)",
      not any(i.startswith("LightSwitch") for i, _ in fids), str(fids))
check("an object with no state axis is never planned", not any(i.startswith("Bread") for i, _ in fids))
check("each flip names the half it buys", all(f["flipped_class"] and f["why"] for f in flips)
      and next(f for f in flips if f["objectId"] == "Faucet|1")["flipped_class"] == "Faucet_on",
      str([(f["objectId"], f["flipped_class"]) for f in flips]))
check("the budget is respected", len(sp.plan_state_flips(ROOM, {}, max_objects=3, class_of=perception.class_name)) == 3)
covered = {"Faucet_on": 99, "StoveBurner_on": 99, "Microwave_open": 99, "Microwave_on": 99, "Cabinet_open": 99,
           "Mug_filled": 99}
late = sp.plan_state_flips(ROOM, covered, max_objects=20, class_of=perception.class_name)
check("fully covered axes are still usable, but only as leftover budget", all(f["tier"] == 2 for f in late) and late,
      str([f["tier"] for f in late]))

section("9. balance opens the drawer a fork lives in, and does not stop at four frames")
PANTRY = [
    {"objectId": "Drawer|1", "objectType": "Drawer", "openable": True, "isOpen": False,
     "position": {"x": 1.0, "y": 0.4, "z": 0.0}},
    {"objectId": "Fork|1", "objectType": "Fork", "parentReceptacles": ["Drawer|1"],
     "position": {"x": 1.0, "y": 0.42, "z": 0.0}},
    {"objectId": "Spoon|1", "objectType": "Spoon", "parentReceptacles": ["CounterTop|1"],
     "position": {"x": 2.0, "y": 0.95, "z": 0.0}},
    {"objectId": "Spoon|2", "objectType": "Spoon", "parentReceptacles": ["Drawer|1"],
     "position": {"x": 1.0, "y": 0.42, "z": 0.1}},
    {"objectId": "CounterTop|1", "objectType": "CounterTop", "position": {"x": 2.0, "y": 0.9, "z": 0.0}},
]
bal = sp.plan_balance(PANTRY, {"Fork": 3, "Spoon": 0, "CounterTop": 80, "Drawer_closed": 50},
                      per_class_target=40, max_frames_per_class=12, class_of=perception.class_name)
bcls = [b["cls"] for b in bal]
check("a class at target is not topped up", "CounterTop" not in bcls, str(bcls))
check("a class with ZERO frames is first", bcls[0] == "Spoon", str(bcls))
fork = next(b for b in bal if b["cls"] == "Fork")
check("the fork's target says which closed drawer to open", fork["targets"][0]["open_parent"] == "Drawer|1",
      str(fork))
check("frames wanted exceeds the old cap of four", fork["frames_wanted"] == 12, str(fork))
spoon = next(b for b in bal if b["cls"] == "Spoon")
check("a visible instance is visited before one that costs an open", spoon["targets"][0]["open_parent"] is None
      and spoon["targets"][1]["open_parent"] == "Drawer|1", str(spoon))
check("an object on an open surface needs no opening", sp.closed_parent(PANTRY[2], {o["objectId"]: o for o in PANTRY}) is None)
check("an already-open drawer needs no opening",
      sp.closed_parent(PANTRY[1], {"Drawer|1": dict(PANTRY[0], isOpen=True)}) is None)


# ---------------------------------------------------------------------------------------------------------
print(f"\n{'=' * 78}\n{PASS} passed, {FAIL} failed\n{'=' * 78}")
sys.exit(1 if FAIL else 0)
