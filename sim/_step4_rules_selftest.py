"""Self-test for step 4 / s1-rules -- the honest observation layer (`sim/honest.py`).

Runs headless on Windows `python` from the REPO ROOT -- no ai2thor, no GPU, no Ollama, no ultralytics:

    python sim/_step4_rules_selftest.py

What it proves (spec: docs/step4-plan.md, s1-rules):

  1. depth grounding of the action point, with an explicit "ungroundable" path;
  2. every privileged read is counted per reason;
  3. every forbidden door raises AND sticks (a swallowed violation still invalidates the episode);
  4. object ids are opaque: never in observations, history, results or the audit log;
  5. holding comes from the agent's own successful actions, and gates the inspection specialist;
  6. INVARIANCE: scrambling the simulator's object records (ids, positions, open/closed, order, decoys) while
     RGB, depth, detections and pose stay fixed changes no honest decision -- and does change the oracle arm;
  7. a static guard over every step-4 module: privileged tokens only inside honest.py's adapter region.
"""
import glob
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402

import honest                                                         # noqa: E402
from honest import HonestAgent, HonestyViolation, MetaAudit, ground_point  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def raises(fn, exc=HonestyViolation):
    try:
        fn()
    except exc:
        return True
    except Exception as e:                                            # wrong exception type
        print("      (raised", type(e).__name__, e, ")")
        return False
    return False


# ----------------------------------------------------------------------------------------------- fakes
H, W = 60, 80
APPLE_BOX = (10, 10, 20, 20)
FRIDGE_BOX = (40, 5, 70, 50)


def make_depth():
    d = np.full((H, W), 3.0, dtype="float32")
    d[10:21, 10:21] = 1.0
    d[15, 15] = 3.0                      # a hole at the apple's centre: the point must move off it
    d[5:51, 40:71] = 2.0
    return d


def base_objects():
    return [
        {"objectId": "Apple|+0.10|+0.90|-0.20", "objectType": "Apple", "visible": True, "pickupable": True,
         "position": {"x": 0.1, "y": 0.9, "z": -0.2}, "screen_box": APPLE_BOX},
        {"objectId": "Fridge|-2.10|+0.00|+1.07", "objectType": "Fridge", "visible": True, "openable": True,
         "isOpen": False, "receptacle": True, "position": {"x": -2.1, "y": 0.0, "z": 1.07},
         "screen_box": FRIDGE_BOX},
    ]


def scrambled_objects(seed):
    rng = random.Random(seed)
    objs = base_objects()
    for o in objs:
        o["objectId"] = f"{o['objectType']}|{rng.uniform(-3, 3):+.2f}|{rng.uniform(0, 2):+.2f}|{rng.random():.3f}"
        o["position"] = {k: rng.uniform(-3, 3) for k in "xyz"}
        o["isOpen"] = rng.random() < 0.5
        o["distance"] = rng.uniform(0, 5)
    objs += [{"objectId": f"Decoy{i}|0|0|0", "objectType": rng.choice(["Apple", "Egg", "Mug"]),
              "visible": rng.random() < 0.5, "pickupable": True, "position": {"x": 0, "y": 0, "z": 0},
              "screen_box": None} for i in range(5)]
    rng.shuffle(objs)
    return objs


class FakeEvent:
    def __init__(self, md):
        self.metadata = md
        self.instance_detections2D = {}


class FakeThor:
    """Just enough ThorAgent: do / frame / depth_frame / event. Physics = the fixed screen boxes."""

    def __init__(self, objects, depth=True, fail=None, pose_x=0.5):
        self.frame = np.zeros((H, W, 3), dtype="uint8")
        self.depth_frame = make_depth() if depth else None
        self.fail = dict(fail or {})                 # action -> error text
        self.calls = []
        md = {"agent": {"position": {"x": pose_x, "y": 0.9, "z": 0.0}, "rotation": {"x": 0, "y": 90, "z": 0},
                        "cameraHorizon": 30.0, "isStanding": True},
              "cameraPosition": {"x": pose_x, "y": 1.57, "z": 0.0}, "objects": objects,
              "lastActionSuccess": True, "errorMessage": "", "actionReturn": None, "lastAction": "Pass",
              "inventoryObjects": [], "sceneName": "FloorPlan1_physics"}
        self.event = FakeEvent(md)
        self._refresh_boxes()

    def _refresh_boxes(self):
        self.event.instance_detections2D = {o["objectId"]: o["screen_box"]
                                            for o in self.event.metadata["objects"] if o.get("screen_box")}

    def do(self, action, **kw):
        self.calls.append((action, dict(kw)))
        md = self.event.metadata
        md.update(lastAction=action, actionReturn=None, lastActionSuccess=True, errorMessage="")
        if action in self.fail:
            md.update(lastActionSuccess=False, errorMessage=self.fail[action])
        elif action == "GetObjectInFrame":
            px, py = kw["x"] * W, kw["y"] * H
            hit = [o["objectId"] for o in md["objects"] if o.get("screen_box")
                   and o["screen_box"][0] <= px <= o["screen_box"][2] and o["screen_box"][1] <= py <= o["screen_box"][3]]
            md["actionReturn"] = hit[0] if hit else None
            md["lastActionSuccess"] = bool(hit)
        elif action == "GetReachablePositions":
            md["actionReturn"] = [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.25, "y": 0.9, "z": 0.0}]
        return self.event


class FakeDetector:
    def __init__(self):
        self.inspecting_seen = []

    def detect(self, frame_rgb, conf=0.25, inspecting=None):
        self.inspecting_seen.append(inspecting)
        return [{"type": "Apple", "state": "", "conf": 0.91, "bbox_xyxy": APPLE_BOX, "source": "primary"},
                {"type": "Fridge", "state": "closed", "states": {"open": 0}, "conf": 0.88,
                 "bbox_xyxy": FRIDGE_BOX, "source": "primary"},
                {"type": "Apple", "state": "", "conf": 0.30, "bbox_xyxy": (0, 50, 6, 58), "source": "primary"}]


def policy(h):
    """A tiny honest decision routine: find the best apple, pick it up, put it in the fridge."""
    trace = []
    obs = h.observe()
    trace.append(("pose", obs.pose["position"]["x"] > 0))
    if obs.pose["position"]["x"] > 0:
        trace.append(("move", h.move("RotateRight")))
    obs = h.observe()
    apples = sorted([d for d in obs.dets if d["type"] == "Apple"], key=lambda d: -d["conf"])
    trace.append(("apple_point", apples[0]["point"], apples[0]["point_ok"]))
    trace.append(("pickup", h.act("PickupObject", apples[0]["point"], target_type="Apple")))
    obs = h.observe()
    fridge = [d for d in obs.dets if d["type"] == "Fridge"][0]
    trace.append(("open", h.act("OpenObject", fridge["point"], target_type="Fridge")))
    trace.append(("put", h.act("PutObject", fridge["point"], target_type="Fridge")))
    trace.append(("holding", h.holding))
    return trace, obs


def masked_calls(calls):
    return [(a, {k: ("<id>" if k == "objectId" else v) for k, v in kw.items()}) for a, kw in calls]


# ======================================================================================================== 1
print("\n1. depth grounding")
d = make_depth()
pt, ok = ground_point(d, APPLE_BOX)
check("centre on a depth hole moves to the nearest agreeing pixel", ok and pt != (15, 15) and d[pt[1], pt[0]] == 1.0,
      str(pt))
check("...and stays adjacent to the centre", abs(pt[0] - 15) + abs(pt[1] - 15) == 1, str(pt))
pt, ok = ground_point(d, FRIDGE_BOX)
check("centre that agrees with the box median is kept", ok and pt == (55, 28), str(pt))
check("no depth -> ungroundable, centre returned", ground_point(None, APPLE_BOX) == ((15, 15), False))
bad = np.full((H, W), np.nan, dtype="float32")
check("all-invalid depth -> ungroundable", ground_point(bad, APPLE_BOX)[1] is False)
check("box outside the frame is clamped", ground_point(d, (-5, -5, 3, 3))[0] == (0, 0))

# ======================================================================================================== 2
print("\n2. audit counts per reason")
thor, det = FakeThor(base_objects()), FakeDetector()
h = HonestAgent(thor, det)
h.reset_episode("ep-1")
trace, last = policy(h)
c = h.audit.counts
n_sends = len(thor.calls)
check("3 observes -> 3 pose reads", c["pose"] == 3, str(c))
check("one actuator read per primitive sent", c["actuator"] == n_sends, f"{c} sends={n_sends}")
check("one action-resolve per targeted act (3)", c["action-resolve"] == 3, str(c))
check("no oracle / scoring / map-build reads in the honest policy",
      c["oracle"] == 0 and c["scoring"] == 0 and c["map-build"] == 0, str(c))
check("episode valid, no violations", not h.audit.invalid and not h.audit.violations)
check("every audit log line names its caller", all(":" in line[3] for line in h.audit.log))
check("policy succeeded end to end", [t[1]["ok"] for t in trace if t[0] in ("pickup", "open", "put")]
      == [True, True, True], str(trace))
check("honest acts send forceAction=False",
      all(kw.get("forceAction") is False for a, kw in thor.calls if a in ("PickupObject", "OpenObject", "PutObject")))
check("GetObjectInFrame gets the grounded point, normalised",
      thor.calls[[a for a, _ in thor.calls].index("GetObjectInFrame")][1]["x"] < 1.0)
check("summary reports counts + validity", h.audit.summary()["invalid"] is False
      and h.audit.summary()["episode"] == "ep-1")

# ======================================================================================================== 3
print("\n3. forbidden doors raise and stick")
h = HonestAgent(FakeThor(base_objects()), FakeDetector())
check("untagged read raises", raises(lambda: h.meta(None, ["objects"])))
check("unknown reason raises", raises(lambda: h.meta("debug", ["agent.position"])))
check("oracle read raises in honest mode", raises(lambda: h.meta("oracle")))
check("off-whitelist field under 'pose' raises", raises(lambda: h.meta("pose", ["objects"])))
check("isOpen-style field under 'actuator' raises", raises(lambda: h.meta("actuator", ["objects.0.isOpen"])))
h.move("MoveAhead")
check("action-resolve after a non-GetObjectInFrame primitive raises",
      raises(lambda: h.meta("action-resolve", ["actionReturn"])))
check("map-build after the wrong primitive raises", raises(lambda: h.meta("map-build", ["actionReturn"])))
check("scoring read outside the scoring phase raises", raises(lambda: h.meta("scoring")))
with h.scoring():
    full = h.meta("scoring")
    check("scoring read inside the scoring phase is allowed + counted",
          "objects" in full and h.audit.counts["scoring"] == 1)
    check("acting inside the scoring phase raises", raises(lambda: h.move("MoveAhead")))
    check("observing inside the scoring phase raises", raises(lambda: h.observe()))
    check("decision-time pose read inside the scoring phase raises", raises(lambda: h.pose()))
check("object_id passed in honest mode raises", raises(lambda: h.act("PickupObject", object_id="Apple|1|2|3")))
check("Teleport is not an honest primitive", raises(lambda: h.move("Teleport", x=0, y=0, z=0), ValueError))
check("unknown interaction primitive rejected", raises(lambda: h.act("SetObjectStates", (1, 1)), ValueError))
check("episode is invalid after violations", h.audit.invalid and len(h.audit.violations) >= 10,
      str(len(h.audit.violations)))
# a swallowed violation still sticks
h2 = HonestAgent(FakeThor(base_objects()), FakeDetector())
try:
    h2.meta("oracle")
except HonestyViolation:
    pass
check("swallowed violation still invalidates the episode", h2.audit.invalid and len(h2.audit.violations) == 1)
h2.reset_episode("ep-2")
check("reset_episode starts a fresh, valid episode", not h2.audit.invalid and h2.audit.counts["pose"] == 0)
# non-strict audit: records without raising
h3 = HonestAgent(FakeThor(base_objects()), FakeDetector(), audit=MetaAudit(strict=False))
h3.meta(None, ["objects"])
check("non-strict audit records an untagged read without raising",
      h3.audit.invalid and h3.audit.counts.get("untagged") == 1)
# s7 audit: a non-strict violation is recorded but the data is withheld, and nothing is sent while scoring
check("non-strict violation WITHHOLDS the privileged data", h3.meta(None, ["objects"]) == {"objects": None}
      and h3.meta("oracle") == {} and h3.meta("pose", ["objects"]) == {"objects": None})
n_calls = len(h3._thor.calls)
with h3.scoring():
    blocked = raises(lambda: h3.move("MoveAhead")) and raises(lambda: h3.act("OpenObject", (5, 5)))
check("non-strict: acting inside the scoring phase is blocked, nothing sent",
      blocked and len(h3._thor.calls) == n_calls, str(h3._thor.calls[n_calls:]))
check("honest mode without a detector is a hard error", raises(lambda: HonestAgent(FakeThor([]), None), RuntimeError))
check("honest observe without depth is a hard error",
      raises(lambda: HonestAgent(FakeThor([], depth=False), FakeDetector()).observe(), RuntimeError))
check("bad mode rejected", raises(lambda: HonestAgent(FakeThor([]), FakeDetector(), mode="x"), ValueError))

# ======================================================================================================== 4
print("\n4. object ids are opaque")
ids = [o["objectId"] for o in base_objects()]
thor, det = FakeThor(base_objects()), FakeDetector()
h = HonestAgent(thor, det)
trace, last = policy(h)
blob = repr((trace, last.dets, last.pose, last.holding, h.history, h.audit.log, vars(last).keys()))
check("no objectId in observations / history / results / audit log", not any(i in blob for i in ids))
check("no '|' id fragments leak either", "|" not in blob)
check("act() returns only ok/err", all(set(t[1]) == {"ok", "err"} for t in trace if t[0] in ("pickup", "open", "put")))
check("the id did reach the simulator call (used once)", any(kw.get("objectId") in ids for _a, kw in thor.calls))
check("the agent holds no simulator event handle", not any(
    hasattr(v, "metadata") for k, v in vars(h).items() if k != "_thor"))
thor = FakeThor(base_objects(), fail={"PickupObject": "Apple|+0.10|+0.90|-0.20 is not reachable (too far)"})
h = HonestAgent(thor, FakeDetector())
o = h.observe()
r = h.act("PickupObject", o.dets[0]["point"], target_type="Apple")
check("actuator error text is sanitised for decisions", r == {"ok": False, "err": "refused"}, str(r))
check("...but logged for the failure breakdown", h.audit.errors and "too far" in h.audit.errors[-1][2])
r = h.act("PickupObject", (79, 59), target_type="Apple")
check("a point on nothing resolves to nothing", r == {"ok": False, "err": "nothing-at-point"}, str(r))

# ======================================================================================================== 5
print("\n5. holding comes from own history")
thor, det = FakeThor(base_objects()), FakeDetector()
h = HonestAgent(thor, det)
o = h.observe()
check("starts empty-handed, specialist gate off", h.holding is None and det.inspecting_seen[-1] is False)
h.act("PickupObject", o.dets[0]["point"], target_type="Apple")
check("successful pickup -> holding Apple", h.holding == "Apple")
h.observe()
check("holding gates the inspection specialist on", det.inspecting_seen[-1] is True)
h.act("PutObject", o.dets[1]["point"], target_type="Fridge")
check("successful put -> empty-handed", h.holding is None)
h.act("PickupObject", o.dets[0]["point"])
check("pickup with no believed type -> holding '?'", h.holding == "?")
h.act("DropHandObject")
check("drop -> empty-handed (hand primitive, no point)", h.holding is None)
thor = FakeThor(base_objects(), fail={"PickupObject": "no"})
h = HonestAgent(thor, FakeDetector())
o = h.observe()
h.act("PickupObject", o.dets[0]["point"], target_type="Apple")
check("failed pickup leaves holding unchanged", h.holding is None)
h.reset_episode("ep", holding="Mug")
check("setup-held object enters as own history", h.holding == "Mug" and h.history[0][1] == "setup:holding")

# ======================================================================================================== 6
print("\n6. invariance: scrambled simulator records do not change honest decisions")
ref_thor = FakeThor(base_objects())
ref_trace, _ = policy(HonestAgent(ref_thor, FakeDetector()))
same = True
for seed in range(8):
    t = FakeThor(scrambled_objects(seed))
    t.event.metadata["inventoryObjects"] = [{"objectId": "Junk|1", "objectType": "Junk"}]
    tr, _ = policy(HonestAgent(t, FakeDetector()))
    same &= (tr == ref_trace and masked_calls(t.calls) == masked_calls(ref_thor.calls))
check("8 scrambles: identical decision traces and identical simulator calls (ids masked)", same)
h_diff = HonestAgent(FakeThor(base_objects(), pose_x=-0.5), FakeDetector())
check("...while the policy IS sensitive to what it may see (own pose)", policy(h_diff)[0] != ref_trace)
ho = HonestAgent(FakeThor(base_objects()), mode="oracle")
o1 = ho.observe()
ho2 = HonestAgent(FakeThor(scrambled_objects(3)), mode="oracle")
o2 = ho2.observe()
check("the oracle arm DOES change with the scramble (test is sensitive)",
      sorted(d["type"] for d in o1.dets) != sorted(d["type"] for d in o2.dets)
      or [d.get("object_id") for d in o1.dets] != [d.get("object_id") for d in o2.dets])
check("oracle reads are counted under 'oracle'", ho.audit.counts["oracle"] == 1 and not ho.audit.invalid)
r = ho.act("PickupObject", object_id=o1.dets[0]["object_id"], target_type=o1.dets[0]["type"])
check("oracle mode may act by id, forcing", r["ok"] and ho._thor.calls[-1][1].get("forceAction") is True)
hr = HonestAgent(FakeThor(base_objects()), FakeDetector())
cells = hr.reachable_positions()
check("map-build read is legal right after GetReachablePositions", len(cells) == 2
      and hr.audit.counts["map-build"] == 1 and not hr.audit.invalid)

# ======================================================================================================== 7
print("\n7. static guard over step-4 modules")
FORBIDDEN = [r"\.objects\(", r"\.resolve\(", r"object_by_id", r"metadata", r"held_object", r"GetShortestPath",
             r"interactable_poses", r"visible_objects"]
THOR_OK = {"do", "frame", "depth_frame"}
THOR_OK_ADAPTER = THOR_OK | {"event"}
mods = sorted(set(glob.glob(os.path.join(HERE, "honest*.py")) + glob.glob(os.path.join(HERE, "objmem.py"))
                  + glob.glob(os.path.join(HERE, "step4_*.py"))))
check("guard found honest.py", any(os.path.basename(m) == "honest.py" for m in mods), str(mods))
problems = []
for path in mods:
    in_adapter = False
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if "--- ADAPTERS BEGIN" in line and os.path.basename(path) == "honest.py":
            in_adapter = True
            continue
        if "--- ADAPTERS END" in line:
            in_adapter = False
            continue
        if not in_adapter:
            for pat in FORBIDDEN:
                if re.search(pat, line):
                    problems.append(f"{os.path.basename(path)}:{n}: {pat}")
        for m in re.finditer(r"\b(?:_thor|thor_agent|thor)\.(\w+)", line):
            if m.group(1) not in (THOR_OK_ADAPTER if in_adapter else THOR_OK):
                problems.append(f"{os.path.basename(path)}:{n}: ThorAgent.{m.group(1)}")
check(f"{len(mods)} step-4 module(s): privileged tokens only inside honest.py's adapter region",
      not problems, "; ".join(problems[:8]))
# s7 audit: DECISION modules get a stricter token list (no scoring / map-build / harness doors at all) ...
DECISION = ("honest_map.py", "honest_skills.py", "objmem.py", "step4_gemma.py")
FORBIDDEN_DECISION = [r"\[[\"']objects[\"']\]", r"objectId", r"scoring\(", r"\.meta\(", r"reachable_positions",
                      r"audit\.errors", r"instance_detections", r"state_readers", r"run_steps", r"_exec_one",
                      r"type_list\(", r"^\s*(?:import|from)\s+(?:thor_actions|webapp|eval|validate_general)\b"]
dprob = []
for name in DECISION:
    for n, line in enumerate(open(os.path.join(HERE, name), encoding="utf-8"), 1):
        dprob += [f"{name}:{n}: {pat}" for pat in FORBIDDEN_DECISION if re.search(pat, line)]
check("decision modules: no scoring / map-build / harness / raw-agent doors", not dprob, "; ".join(dprob[:8]))
# ... and the HARNESS scripts may touch the raw simulator agent only through named, reasoned doors.
HARNESS_AGENT_OK = {"type_list": "scoring (applicability gate)", "field_of_view": "camera calibration",
                    "stop": "lifecycle"}
HARNESS_EV_OK = {"_reset_world": "lifecycle", "run_steps": "oracle setup / oracle arm",
                 "state_readers": "scoring (goal check)"}
hprob = []
for name in ("step4_refplans.py", "step4_smoke.py", "step4_gemma_trial.py"):
    lines = open(os.path.join(HERE, name), encoding="utf-8").read().splitlines()
    for n, line in enumerate(lines, 1):
        code = line.split("#", 1)[0]
        hprob += [f"{name}:{n}: agent.{m}" for m in re.findall(r"\b(?:ag|sim_agent)\.(\w+)", code)
                  if m not in HARNESS_AGENT_OK]
        hprob += [f"{name}:{n}: ev.{m}(agent)" for m in re.findall(r"\bev\.(\w+)\(ag\b", code)
                  if m not in HARNESS_EV_OK]
        if re.search(r"\.meta\(", code):
            back = " ".join(lines[max(0, n - 4):n])
            if not (re.search(r"\.meta\([\"']scoring[\"']\)", code) and "with h.scoring()" in back):
                hprob.append(f"{name}:{n}: meta() outside a scoring block")
check("harness scripts: raw-agent reads only through named doors; meta only as scoring inside h.scoring()",
      not hprob, "; ".join(hprob[:8]))
rsrc = open(os.path.join(HERE, "step4_refplans.py"), encoding="utf-8").read()
plan = rsrc[rsrc.index("sk.start_episode("):rsrc.index("with h.scoring():")]
check("refplans: between start_episode and the scoring block no oracle/scoring call is made",
      not re.search(r"\bev\.|_goal_ok|_unmet|\bag\b|run_steps", plan), plan[:200])
tsrc = open(os.path.join(HERE, "step4_gemma_trial.py"), encoding="utf-8").read()
tplan = tsrc[tsrc.index("sk.start_episode("):tsrc.index("with h.scoring():")]
check("s6 trial: between start_episode and the scoring block no oracle/scoring call is made",
      not re.search(r"\bev\.(?!labels|levenshtein|pc_score|tds_score)|_goal_ok|_unmet|\bag\b|run_steps", tplan),
      tplan[:200])
gsrc = open(os.path.join(HERE, "step4_gemma.py"), encoding="utf-8").read()
check("s6 planner module imports no simulator / harness module",
      not re.search(r"^\s*(?:import|from)\s+(?:thor_actions|webapp|eval|validate_general|honest\b|step4_refplans|general_set)",
                    gsrc, re.M))
# console v2 (backend): honest_console.py is HARNESS code with its own named doors -- the raw simulator agent
# (``ag`` / ``self._ag``) only for the ThorPort's four names + camera calibration; meta() only as scoring inside
# h.scoring(); no decision-module door at all; and the port really exposes exactly four public names.
CONSOLE_AGENT_OK = {"do": "ThorPort", "frame": "ThorPort", "depth_frame": "ThorPort",
                    "event": "ThorPort (read only by HonestAgent's adapter region)", "field_of_view": "camera calibration"}
check("guard covers honest_console.py", any(os.path.basename(m) == "honest_console.py" for m in mods), str(mods))
cprob = []
clines = open(os.path.join(HERE, "honest_console.py"), encoding="utf-8").read().splitlines()
for n, line in enumerate(clines, 1):
    code = line.split("#", 1)[0]
    cprob += [f"honest_console.py:{n}: agent.{m}" for m in re.findall(r"\b_?ag\.(\w+)", code) if m not in CONSOLE_AGENT_OK]
    cprob += [f"honest_console.py:{n}: {pat}" for pat in (r"objectId", r"instance_detections", r"state_readers",
                                                           r"run_steps", r"_exec_one", r"type_list\(", r"audit\.log")
              if re.search(pat, code)]
    if re.search(r"\.meta\(", code):
        back = " ".join(clines[max(0, n - 4):n])
        if not (re.search(r"\.meta\([\"']scoring[\"']\)", code) and "with h.scoring()" in back):
            cprob.append(f"honest_console.py:{n}: meta() outside a scoring block")
    if re.search(r"^\s*(?:import|from)\s+(?:thor_actions|webapp|eval|validate_general|step4_refplans)\b", code):
        cprob.append(f"honest_console.py:{n}: imports a simulator / harness module")
check("honest_console: raw agent only through the port's names + calibration; meta only as scoring; no oracle doors",
      not cprob, "; ".join(cprob[:8]))
import honest_console                                                 # noqa: E402
port_names = sorted(n for n in dir(honest_console.ThorPort) if not n.startswith("_"))
check("ThorPort exposes exactly do / frame / depth_frame / event", port_names == ["depth_frame", "do", "event", "frame"],
      str(port_names))
check("ThorPort has no __getattr__ passthrough and fixed slots",
      "__getattr__" not in vars(honest_console.ThorPort) and hasattr(honest_console.ThorPort, "__slots__"))
check("cancel / abort cannot be swallowed by `except Exception`",
      not issubclass(honest_console.HonestCancelled, Exception) and not issubclass(honest_console.HonestAborted, Exception))
csrc = "\n".join(clines)
gpos = [m.start() for m in re.finditer(r"self\.goal_check\(self\._ag", csrc)]
check("every goal_check call in honest_console sits inside h.scoring() (after h.meta('scoring'))",
      len(gpos) == 2 and all("with h.scoring():" in csrc[max(0, p - 160):p] and 'h.meta("scoring")' in csrc[max(0, p - 80):p]
                             for p in gpos), str(gpos))
src = open(os.path.join(HERE, "honest.py"), encoding="utf-8").read()
check("adapter region exists exactly once", src.count("--- ADAPTERS BEGIN") == 1 and src.count("--- ADAPTERS END") == 1)
check("the guard itself fires on a leak", bool(re.search(FORBIDDEN[0], "x = thor.objects()")))

# ======================================================================================================== 8
print("\n8. detector loading")
check("missing step-3 weights is a hard error (no oracle fallback)",
      raises(lambda: honest.load_detector("FloorPlan999"), FileNotFoundError))

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("STEP4 RULES SELFTEST OK")
