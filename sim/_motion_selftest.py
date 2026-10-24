"""Self-test for the live-view work: MOVE-BY-MOVE animation + the perceived-only senses switch.

Runs headless on Windows `python` from the REPO ROOT — no ai2thor, no GPU, no Ollama, no ultralytics:

    python sim/_motion_selftest.py

`webapp.py` guards its `import thor_actions`, so every helper under test here is importable without the
simulator. What the fakes stand in for is deliberate:

  * `FakeAgent`  — the AI2-THOR controller surface these helpers actually touch (object_by_id / open / close /
                   success / motion / detect_objects / look_around / held_object). Every call it receives is
                   RECORDED, so the tests assert on the exact primitive sequence the simulator would have seen.
  * `fake_worker` — the real worker's contract in three lines: pull a job, call it with the agent, and if it
                   returns a generator, advance it one step per rendered frame. `_perceived_observation` is
                   only correct if it survives that contract, so the test reproduces it rather than mocking
                   the queue away.

What is NOT covered here (and cannot be, off-GPU): that YOLO finds the right objects, that a swung fridge door
looks right, and the end-to-end frame timing. Those are Track C in HOW-TO-TEST.html.
"""
import os
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import perception                                                    # noqa: E402
import thor_actions as ta                                            # noqa: E402
import webapp                                                        # noqa: E402

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


# ---------------------------------------------------------------------------------------------------------
class FakeAgent:
    """The slice of ThorAgent these helpers touch, with every controller call recorded."""

    def __init__(self, objects=None, fail_after=None, dets=None):
        self.objects_ = {o["objectId"]: dict(o) for o in (objects or [])}
        self.calls = []                     # [(primitive, args...)] in the order the simulator would see them
        self.success = True
        self.error = ""
        self.motion = {"step": 0.25, "turn": 30.0}
        self.scene = "FloorPlan1_physics"
        self.fail_after = fail_after        # nth open() call that reports failure (1-based), or None
        self._opens = 0
        self.dets = dets if dets is not None else [{"type": "Apple", "state": "whole"}]
        self.held = None
        self.can_move = True

    # -- queries
    def object_by_id(self, oid):
        return self.objects_.get(oid)

    def held_object(self):
        return self.held

    # -- primitives
    def open(self, oid, openness=1.0, force=False):
        self._opens += 1
        self.calls.append(("open", round(float(openness), 4)))
        self.success = not (self.fail_after and self._opens >= self.fail_after)
        self.error = "" if self.success else "openness refused"
        if self.success and oid in self.objects_:
            self.objects_[oid]["openness"] = float(openness)
            self.objects_[oid]["isOpen"] = float(openness) > 0
        return self.success

    def close(self, oid, force=False):
        self.calls.append(("close",))
        self.success = True
        if oid in self.objects_:
            self.objects_[oid]["openness"] = 0.0
            self.objects_[oid]["isOpen"] = False
        return True

    def pickup(self, oid, force=False):
        self.calls.append(("pickup",))
        self.success = True
        return True

    def move_ahead(self, magnitude=0.25):
        """Recorded, and it really closes the gap — `close_in` stops as soon as a step stops helping, so a
        no-op move_ahead would make every closing-in test vacuously pass."""
        self.calls.append(("move_ahead", round(float(magnitude), 3)))
        self.success = self.can_move
        if self.success:
            for o in self.objects_.values():
                if o.get("distance") is not None:
                    o["distance"] = max(0.0, o["distance"] - float(magnitude))
        return self.success

    def walk_to(self, oid, **kw):
        self.calls.append(("walk_to", kw.get("step"), kw.get("turn")))
        for _ in range(2):
            yield "stepping"
        yield "arrived"

    def resolve(self, t, **kw):
        for o in self.objects_.values():
            if o.get("objectType") == t:
                return o
        return None

    # `_exec_one` builds its nav/objmap dispatch tables by attribute access BEFORE it knows which action it
    # needs, so every name in them has to exist. They are recording no-ops; only the ones a test actually
    # exercises are asserted on.
    _NAV = ("move_back", "move_left", "move_right", "rotate_left", "rotate_right",
            "look_up", "look_down", "crouch", "stand", "drop")
    _OBJ = ("toggle_on", "toggle_off", "slice", "cook", "break_object", "dirty", "clean", "put",
            "turn_to_object", "fill_with_liquid", "empty_liquid", "use_up", "push", "pull",
            "pick_and_place", "put_in", "heat_object", "cool_object", "cook_food", "wash_object",
            "look_around_", "open_all", "close_all", "switch_all", "collect_all", "scan_for")

    def __getattr__(self, name):
        if name in FakeAgent._NAV or name in FakeAgent._OBJ:
            def rec(*a, **kw):
                self.calls.append((name,) + a)
                self.success = True
                return True
            return rec
        raise AttributeError(name)

    # -- perceived mode
    def detect_objects(self, det, conf=0.25):
        self.calls.append(("detect",))
        return list(self.dets)

    def look_around(self, turn=45.0):
        for _ in range(int(round(360.0 / turn))):
            self.calls.append(("rotate", turn))
            yield "looking around"


def fake_worker(ag, stop, tick=None):
    """The real worker's job contract: pull a job, call it, advance any generator one step per frame."""
    active = None
    while not stop.is_set():
        if active is not None:
            try:
                next(active)
                if tick:
                    tick()
            except StopIteration:
                active = None
            except Exception:
                active = None
            continue
        try:
            r = webapp._jobs.get(timeout=0.05)(ag)
        except Exception:
            continue
        if hasattr(r, "__next__"):
            active = r


FRIDGE = {"objectId": "Fridge|1", "objectType": "Fridge", "openness": 0.0, "isOpen": False, "visible": True}

# `close_in` is a real ThorAgent method (thor_actions guards its ai2thor import), so bind the shipped one.
FakeAgent.close_in = ta.ThorAgent.close_in


# ---------------------------------------------------------------------------------------------------------
section("1. motion profiles")
check("three named profiles", set(webapp.MOTION_PROFILES) == {"fast", "normal", "cinematic"})
check("every profile is complete",
      all({"turn", "step", "delay", "hold", "openness_steps", "note"} <= set(v)
          for v in webapp.MOTION_PROFILES.values()))
check("granularity strictly increases fast -> cinematic",
      webapp.MOTION_PROFILES["fast"]["turn"] > webapp.MOTION_PROFILES["normal"]["turn"]
      > webapp.MOTION_PROFILES["cinematic"]["turn"]
      and webapp.MOTION_PROFILES["fast"]["step"] > webapp.MOTION_PROFILES["normal"]["step"]
      > webapp.MOTION_PROFILES["cinematic"]["step"])
check("only cinematic paces the worker",
      webapp.MOTION_PROFILES["fast"]["delay"] == 0 and webapp.MOTION_PROFILES["normal"]["delay"] == 0
      and webapp.MOTION_PROFILES["cinematic"]["delay"] > 0)
check("default is a real profile", webapp.MOTION_DEFAULT in webapp.MOTION_PROFILES)
_saved = webapp._MOTION["speed"]
webapp._MOTION["speed"] = "not-a-profile"
check("_mp() falls back instead of raising", webapp._mp() is webapp.MOTION_PROFILES[webapp.MOTION_DEFAULT])
webapp._MOTION["speed"] = _saved

section("2. _apply_motion pushes the profile onto the agent")
ag = FakeAgent()
webapp._MOTION["speed"] = "cinematic"
webapp._apply_motion(ag)
check("agent.motion adopts the profile",
      ag.motion == {"step": webapp.MOTION_PROFILES["cinematic"]["step"],
                    "turn": webapp.MOTION_PROFILES["cinematic"]["turn"]}, str(ag.motion))


class _NoMotion:
    pass


try:
    webapp._apply_motion(_NoMotion())
    check("an agent without .motion is tolerated, not fatal", True)
except Exception as exc:                                             # pragma: no cover
    check("an agent without .motion is tolerated, not fatal", False, repr(exc))

section("3. _hold renders extra frames around an instantaneous primitive")
check("yields exactly n labels", list(webapp._hold(4, "slice Apple")) == ["slice Apple"] * 4)
check("n=0 yields nothing", list(webapp._hold(0, "x")) == [])
check("negative n is clamped, not an error", list(webapp._hold(-3, "x")) == [])

section("4. _animate_openness — the one primitive AI2-THOR can actually animate")
webapp._MOTION["speed"] = "normal"                       # openness_steps = 6
STEPS = webapp.MOTION_PROFILES["normal"]["openness_steps"]
ag = FakeAgent([FRIDGE])
frames = list(webapp._animate_openness(ag, "Fridge|1", 1.0))
check("swings over openness_steps-1 intermediate frames", len(frames) == STEPS - 1, str(len(frames)))
check("every intermediate frame is labelled 'opening'", set(frames) == {"opening"}, str(set(frames)))
opens = [c for c in ag.calls if c[0] == "open"]
check("openness increases monotonically", all(a[1] < b[1] for a, b in zip(opens, opens[1:])), str(opens))
check("lands exactly open", opens[-1][1] == 1.0 and ag.objects_["Fridge|1"]["isOpen"] is True, str(opens[-1]))

ag = FakeAgent([dict(FRIDGE, openness=1.0, isOpen=True)])
frames = list(webapp._animate_openness(ag, "Fridge|1", 0.0))
check("closing yields 'closing' frames", set(frames) == {"closing"}, str(set(frames)))
check("never asks for openness 0 mid-swing",
      all(c[1] > 0 for c in ag.calls if c[0] == "open"), str(ag.calls))
check("lands with the canonical CloseObject", ag.calls[-1] == ("close",), str(ag.calls[-1]))
check("object really is shut", ag.objects_["Fridge|1"]["isOpen"] is False)

ag = FakeAgent([FRIDGE], fail_after=2)
list(webapp._animate_openness(ag, "Fridge|1", 1.0))
check("a refused intermediate stops stepping and throws once",
      len([c for c in ag.calls if c[0] == "open"]) == 3, str(ag.calls))
check("the fallback still requests the full target",
      ag.calls[-1] == ("open", 1.0), str(ag.calls[-1]))

webapp._MOTION["speed"] = "fast"                         # openness_steps = 1 -> no animation at all
ag = FakeAgent([FRIDGE])
check("fast profile does not animate", list(webapp._animate_openness(ag, "Fridge|1", 1.0)) == [])
check("fast profile still opens the door", ag.calls == [("open", 1.0)], str(ag.calls))

ag = FakeAgent([dict(FRIDGE, openness=1.0, isOpen=True)])
webapp._MOTION["speed"] = "normal"
check("no-op swing (already open) skips the intermediates",
      list(webapp._animate_openness(ag, "Fridge|1", 1.0)) == [])

section("5. _approach_then walks, closes in, and holds the camera either side of the action")
webapp._MOTION["speed"] = "normal"
HOLD = webapp.MOTION_PROFILES["normal"]["hold"]
WALK = 3                                    # FakeAgent.walk_to yields stepping, stepping, arrived
ag = FakeAgent([dict(FRIDGE, visible=True)])
out = list(webapp._approach_then(ag, "Fridge|1", lambda: ag.pickup("Fridge|1"), "pickup Fridge"))
# AI2-THOR's `visible` means "within visibilityDistance and in frame", NOT "near" — a pickup used to happen
# from 1.4 m away with the agent never having moved.
check("a VISIBLE object is approached too", ("walk_to", None, None) in ag.calls, str(ag.calls))
check("walk + 2*hold + done frames for a one-shot primitive", len(out) == WALK + 2 * HOLD + 1, str(len(out)))
check("the action fires exactly once", [c for c in ag.calls if c[0] == "pickup"] == [("pickup",)])
check("last frame is 'done'", out[-1] == "done")

ag = FakeAgent([dict(FRIDGE, visible=True)])
out = list(webapp._approach_then(ag, "Fridge|1", lambda: webapp._animate_openness(ag, "Fridge|1", 1.0),
                                 "open Fridge"))
check("an ANIMATED primitive is advanced in place of the hold",
      len(out) == WALK + 2 * HOLD + (STEPS - 1) + 1, str(len(out)))
check("the swing frames are inside the sequence", "opening" in out)

ag = FakeAgent([dict(FRIDGE, visible=False)])
out = list(webapp._approach_then(ag, "Fridge|1", lambda: ag.pickup("Fridge|1"), "pickup Fridge"))
check("an unseen object is walked to first", ("walk_to", None, None) in ag.calls, str(ag.calls))
check("walking adds its own frames", len(out) > 2 * HOLD + 1)

held = FakeAgent([dict(FRIDGE, isPickedUp=True)])
list(webapp._approach_then(held, "Fridge|1", lambda: held.pickup("Fridge|1"), "pickup Fridge"))
check("something already in hand is not walked to", not any(c[0] == "walk_to" for c in held.calls))

section("5b. closing in — 'reachable' is not 'near'")
far = FakeAgent([dict(FRIDGE, visible=True, distance=1.60)])
frames = list(far.close_in("Fridge|1", min_distance=0.55, step=0.15))
check("it edges forward until it is at the object",
      far.objects_["Fridge|1"]["distance"] <= 0.55 + 1e-6, str(far.objects_["Fridge|1"]["distance"]))
check("each step is one rendered frame", len(frames) == len([c for c in far.calls if c[0] == "move_ahead"]))
near = FakeAgent([dict(FRIDGE, visible=True, distance=0.4)])
check("already close -> costs nothing", list(near.close_in("Fridge|1", min_distance=0.55)) == [])
blocked = FakeAgent([dict(FRIDGE, visible=True, distance=1.6)])
blocked.can_move = False
check("blocked IS as close as the geometry allows, not a failure",
      list(blocked.close_in("Fridge|1", min_distance=0.55)) == [])
nod = FakeAgent([dict(FRIDGE, visible=True)])          # no distance reported at all
check("an object with no distance is left alone", list(nod.close_in("Fridge|1")) == [])
holdd = FakeAgent([dict(FRIDGE, distance=2.0, isPickedUp=True)])
check("something in hand is not closed in on", list(holdd.close_in("Fridge|1")) == [])

section("6. _exec_one routes open/close through the animator")
ag = FakeAgent([dict(FRIDGE, visible=True)])
webapp._MOTION["speed"] = "cinematic"
gen = webapp._exec_one(ag, {"action": "open", "object": "Fridge"})
check("returns a generator (so the worker animates it)", hasattr(gen, "__next__"))
check("_exec_one adopted the active profile",
      ag.motion["turn"] == webapp.MOTION_PROFILES["cinematic"]["turn"], str(ag.motion))
out = list(gen)
check("cinematic open produces more frames than fast",
      len(out) > 2 * webapp.MOTION_PROFILES["fast"]["hold"] + 1, str(len(out)))
check("it really opened the fridge", ag.objects_["Fridge|1"]["isOpen"] is True)
webapp._MOTION["speed"] = webapp.MOTION_DEFAULT

section("7. frame fan-out — every rendered frame is offered exactly once")
webapp._frames.clear()
webapp._state["seq"] = 0
webapp._state["dropped"] = 0
webapp._state["jpeg"] = b""
check("a fresh client with no frames yet does not crash", webapp._stream_pending(None) == [(0, b"")])
for i in range(1, 6):
    webapp._publish_frame(bytes([i]))
check("publish bumps seq", webapp._state["seq"] == 5)
check("a fresh client starts at the NEWEST frame, not the backlog",
      webapp._stream_pending(None) == [(5, bytes([5]))], str(webapp._stream_pending(None)))
seen, last = [], 5                                              # the fresh client above jumped to frame 5
for i in range(6, 11):
    webapp._publish_frame(bytes([i]))
    for sq, j in webapp._stream_pending(last):
        seen.append(sq)
        last = sq
check("a keeping-up client sees a gapless sequence", seen == [6, 7, 8, 9, 10], str(seen))
check("nothing was counted as dropped", webapp._state["dropped"] == 0)
pending = webapp._stream_pending(last)
check("caught up -> a keepalive resend of the current frame", pending == [(10, bytes([10]))], str(pending))
check("the keepalive is not counted as a drop", webapp._state["dropped"] == 0)

lagged = webapp._state["seq"]                                   # a client that then stalls
for i in range(webapp.STREAM_BACKLOG + 9):
    webapp._publish_frame(bytes([i % 251]))
pend = webapp._stream_pending(lagged)
check("a stalled client is fast-forwarded, bounded by the backlog",
      len(pend) == webapp.STREAM_BACKLOG, str(len(pend)))
check("and the frames it will never see are COUNTED, not hidden",
      webapp._state["dropped"] == 9, str(webapp._state["dropped"]))
webapp._frames.clear()
webapp._state["seq"] = webapp._state["dropped"] = 0

section("8. perceived mode — the planner is fed the detector, or told it could not be")
check("_perception_ready sees the trained scene (either spelling)",
      webapp._perception_ready("FloorPlan1") and webapp._perception_ready("FloorPlan1_physics"),
      "sim/weights/FloorPlan1_physics.pt should exist in this repo")
check("_perception_ready is False for an untrained scene", not webapp._perception_ready("FloorPlan999"))

_real_detector_for = webapp._detector_for
ag = FakeAgent(dets=[{"type": "Apple", "state": "whole"}, {"type": "Fridge", "state": "closed"}])
ag.held = {"objectType": "Mug"}
stop = threading.Event()
th = threading.Thread(target=fake_worker, args=(ag, stop), daemon=True)
th.start()
try:
    webapp._detector_for = lambda scene: object()                 # a detector exists; FakeAgent produces the boxes
    # pin the profile: the default became "fast" on 2026-10-05 (docs/live-view.md), and this check is about
    # rotating at the ACTIVE profile's granularity -- 15 deg for "normal" -- while YOLO stays at one call per 45 deg
    webapp._MOTION["speed"] = "normal"
    res = webapp._perceived_observation(timeout=10)
    webapp._MOTION["speed"] = webapp.MOTION_DEFAULT
    check("perceived run is not degraded", res["degraded"] is False, str(res))
    check("it reports how many types it saw", res["n"] == 2, str(res))
    check("the observation is built from DETECTIONS", "visible (perceived)" in res["text"], res["text"])
    check("Apple and Fridge (with state) reached the prompt",
          "Apple" in res["text"] and "Fridge(closed)" in res["text"], res["text"])
    check("held is proprioception, so it survives", "held=Mug" in res["text"], res["text"])
    # The sweep must be WATCHABLE and CHEAP at the same time: one rendered frame per rotation step at the motion
    # profile's granularity (15° for `normal` = 24 frames), but only one YOLO call per 45°. It used to be 45°
    # for both, i.e. 8 frames — a 360° search that was over in a fifth of a second.
    rots = [c for c in ag.calls if c[0] == "rotate_right"]
    dets = [c for c in ag.calls if c[0] == "detect"]
    check("the agent physically TURNED to look", len(rots) == 24, f"{len(rots)} rotations")
    check("it rotated at the motion profile's granularity",
          all(abs(c[1] - webapp.MOTION_PROFILES["normal"]["turn"]) < 1e-6 for c in rots), str(rots[:2]))
    check("YOLO ran once per 45°, not once per frame", len(dets) == 9, f"{len(dets)} inferences")
    check("so the sweep is ~3x more watchable at no extra inference cost", len(rots) > 2 * len(dets))

    webapp._detector_for = lambda scene: None                     # no trained weights for this scene
    res = webapp._perceived_observation(timeout=10)
    check("a missing detector DEGRADES rather than silently using oracle", res["degraded"] is True, str(res))
    check("and says why, actionably", "train detector" in (res["error"] or ""), str(res.get("error")))
finally:
    webapp._detector_for = _real_detector_for
    stop.set()
    th.join(timeout=2)

section("9. shared weights resolver (one definition for webapp + eval)")
with tempfile.TemporaryDirectory() as wd:
    check("no weights -> '' (never a guess)", perception.weights_path("FloorPlan1", wd) == "")
    open(os.path.join(wd, "FloorPlan1_physics.pt"), "wb").close()
    check("the live sceneName resolves", perception.weights_path("FloorPlan1_physics", wd) != "")
    check("the dataset spelling resolves to the same file",
          perception.weights_path("FloorPlan1", wd) == perception.weights_path("FloorPlan1_physics", wd))
check("observation_from is honest about an empty detection set",
      "(nothing detected)" in perception.observation_from({}, "none"))

section("10. oracle text is withheld from the planner in perceived mode")
# The worker keeps _OBS['oracle'] fresh every frame (it is the fallback), but _OBS['text'] -- the string that
# reaches the prompt -- must only be written by the detector sweep while perceived mode is on.
check("_OBS carries both channels", set(webapp._OBS) == {"text", "oracle"}, str(set(webapp._OBS)))
src = open(os.path.join(HERE, "webapp.py"), encoding="utf-8").read()
check("the worker gates the oracle write on the mode",
      'if _PERCEPTION["mode"] != "perceived":' in src and '_OBS["oracle"] = agent.describe' in src)
check("_PERCEPTION carries a degraded reason", "degraded" in webapp._PERCEPTION)

section("11. the UI/server version handshake")
# sim/web/* is re-read per request but webapp.py is loaded once at start, so editing the server and reloading
# the browser pairs a NEW console with an OLD backend -- every new control 404s and every new panel stays empty
# with nothing anywhere saying why. It happened, on the first real run of this feature.
js = open(os.path.join(HERE, "web", "app.js"), encoding="utf-8").read()
m = __import__("re").search(r"const UI_API\s*=\s*(\d+)", js)
check("app.js declares UI_API", bool(m), "no `const UI_API = <n>` in app.js")
check("UI_API matches WEBAPP_API", bool(m) and int(m.group(1)) == webapp.WEBAPP_API,
      f"app.js={m.group(1) if m else '?'} webapp.WEBAPP_API={webapp.WEBAPP_API}")
check("/status carries the version", '''st["api"] = WEBAPP_API''' in src)
check("app.js checks it before reading any versioned field",
      js.index("checkApi(s.api)") < js.index("renderMotion(s.motion)"))
check("an OLD server (no api field) is treated as stale, not as OK",
      "!(serverApi >= UI_API)" in js)

section("13. reset")
# `everything` is irreversible, so the endpoint is the last line of defence, not the UI.
check("both scopes exist and nothing else does", webapp.RESET_SCOPES == ("session", "all"))
check("the session reset does NOT revert operator settings",
      all(k not in src.split("def _reset_session")[1].split("def _reset_learned")[0]
          for k in ('_MOTION["speed"] =', '_PERCEPTION["mode"] =')),
      "motion/perception are choices, not run state")
sess = src.split("def _reset_session")[1].split("def _reset_learned")[0]
for field in ('_state["log"] = []', '_state["plan"] = None', "ag.reset()", "last_search"):
    check(f"session reset clears {field!r}", field in sess)
learned = src.split("def _reset_learned")[1].split("def improve_reports_path")[0]
for store in ("knowledge base", "memory + bandit", "loop reports", "trace store"):
    check(f"'{store}' is wiped by the full reset", f'"{store}"' in learned)
check("each store is closed before its file is removed",
      "m.close()" in learned and "_trace.close()" in learned)
check("a partial failure is reported, not raised",
      '"failed"' in learned and "except Exception as exc" in learned)
check("the endpoint refuses while a command is running",
      "a command is still running" in src and "code=409" in src)
check("an unknown scope is rejected", "scope must be one of" in src and "code=400" in src)
check("the loop-report path is asked of `improve`, so the two cannot disagree",
      "improve.LOOP_REPORTS_PATH" in src)
check("the UI clears its render caches, or an emptied log repaints as 'unchanged'",
      "_logSig = null" in js and "renderLog([])" in js)
check("`everything` is never one click away", "armReset(true)" in js and 'hidden = !on' in js)

section("12. no plan step finishes in silence")
# "find the apple" produced a one-step plan whose transcript went straight from the plan to the reward: the
# composites and nav primitives never logged, only the _approach_then actions did.
check("the self-logging set is exactly the _approach_then actions",
      webapp._SELF_LOGGING_ACTIONS == frozenset((
          "pickup", "open", "close", "toggle_on", "toggle_off", "slice", "cook", "break",
          "dirty", "clean", "put",
          # fill / empty were offered by PLANNER_SYSTEM but never dispatched; they now run through
          # _approach_then like the rest, so they log their own line too (both spellings are accepted)
          "fill", "fill_with_liquid", "empty", "empty_liquid")), str(sorted(webapp._SELF_LOGGING_ACTIONS)))
for a in ("scan_for", "look_around", "pick_and_place", "heat", "go_to", "move_ahead", "wash"):
    check(f"'{a}' is NOT self-logging (so _tracked_step must log it)",
          a not in webapp._SELF_LOGGING_ACTIONS)
check("_tracked_step logs every step the action did not log itself",
      "if str(step.get(\"action\") or \"\").lower().strip() not in _SELF_LOGGING_ACTIONS:" in src)
check("the added line keeps the documented '<label>: ok' format",
      """_log(f"{_step_label(step)}: {'ok' if ok else str(err)[:48]}")""" in src)
check("_step_label renders a composite readably",
      webapp._step_label({"action": "scan_for", "object": "Apple"}) == "scan for → Apple",
      webapp._step_label({"action": "scan_for", "object": "Apple"}))

section("14. lighting: a camera-view switch must not reload the world")
# Reported as "the lighting goes crazy sometimes". Two independent causes, both pinned here:
#   (a) set_sensors() is the only way AI2-THOR changes render flags and it works by controller.reset(), which
#       reloads lighting, object states, doors and the agent pose -- and the console called it on EVERY view
#       switch, including RGB <-> Detect, which render the same buffer;
#   (b) a planner step turn_off_all with a null object swept every toggleable object, LightSwitch included.


class _Ev:
    def __init__(self):
        self.metadata = {"sceneName": "FloorPlan1_physics"}


class _Ctl:
    def __init__(self):
        self.resets = []

    def reset(self, **kw):
        self.resets.append(kw)
        return _Ev()


class _SensAgent:
    set_sensors = ta.ThorAgent.set_sensors

    def __init__(self):
        self.event = _Ev()
        self.controller = _Ctl()
        self._sensors = {"depth": False, "instance": False, "semantic": False, "normals": False}
        self.teleports = []

    def agent_pose(self):
        return {"position": {"x": 1.25, "y": 0.9, "z": -0.5}, "rotation": {"y": 90.0},
                "horizon": 15.0, "standing": False}

    def teleport_full(self, x, y, z, rot, horizon=0.0, standing=True):
        self.teleports.append((x, y, z, rot, horizon, standing))


sa = _SensAgent()
sa.set_sensors()                                     # RGB -> RGB, and RGB -> Detect, which asks for the same
check("re-selecting the streams already rendering does NOT reload the scene", sa.controller.resets == [],
      str(sa.controller.resets))
sa.set_sensors(depth=True)
check("a real change reloads exactly once", len(sa.controller.resets) == 1, str(sa.controller.resets))
check("with the requested render flags",
      sa.controller.resets[0].get("renderDepthImage") is True
      and sa.controller.resets[0].get("renderInstanceSegmentation") is False)
check("and the pose is restored afterwards, so the camera does not jump across the room",
      sa.teleports == [(1.25, 0.9, -0.5, 90.0, 15.0, False)], str(sa.teleports))
sa.set_sensors(depth=True)
check("asking for the same change again is a no-op", len(sa.controller.resets) == 1)
sa.set_sensors(depth=True, force=True)
check("force=True reloads anyway (the escape hatch for a stuck renderer)", len(sa.controller.resets) == 2)
_ta_src = open(os.path.join(HERE, "thor_actions.py"), encoding="utf-8").read()
check("the constructor seeds the sensor state, so the FIRST call is judged correctly",
      'self._sensors = {"depth": bool(render_depth)' in _ta_src)


class _LightAgent:
    LIGHT_TYPES = ta.ThorAgent.LIGHT_TYPES
    _each = ta.ThorAgent._each
    switch_all = ta.ThorAgent.switch_all

    def __init__(self):
        self.calls = []
        self.success = True

    def objects(self):
        return [{"objectId": t + "|1", "objectType": t, "toggleable": True, "isToggled": True,
                 "distance": float(i + 1)}
                for i, t in enumerate(("LightSwitch", "Toaster", "StoveBurner", "FloorLamp"))]

    def object_by_id(self, oid):
        return next((o for o in self.objects() if o["objectId"] == oid), None)

    def toggle_off(self, oid, force=False):
        self.calls.append(oid.split("|")[0])
        self.success = True
        return True

    toggle_on = toggle_off

    def walk_to(self, oid, **kw):
        yield "arrived"


la = _LightAgent()
list(la.switch_all(False))
check("turn_off_all with a null object leaves the room lights alone",
      "LightSwitch" not in la.calls and "FloorLamp" not in la.calls, str(la.calls))
check("...but still switches the appliances off", {"Toaster", "StoveBurner"} <= set(la.calls), str(la.calls))
la = _LightAgent()
list(la.switch_all(False, "LightSwitch"))
check("naming LightSwitch still switches it", la.calls == ["LightSwitch"], str(la.calls))
la = _LightAgent()
list(la.switch_all(False, include_lights=True))
check("include_lights=True is the deliberate way to sweep everything", "LightSwitch" in la.calls, str(la.calls))
import agent_llm as _al
check("the planner is told the lights are excluded, so it is not surprised",
      "room lights are NOT included" in _al.PLANNER_SYSTEM)

section("15. retraining never erases the detector a published number came from")
# The Train button wrote weights/<scene>.pt, and for FloorPlan1 that file is the detector whose per-class AP the
# thesis cites. A retrain with a new base/recipe must ADD a model, never overwrite the only evidence.
import shutil as _sh
_wd = tempfile.mkdtemp()
try:
    _wp = os.path.join(_wd, "FloorPlan1_physics.pt")
    with open(_wp, "wb") as _fh:
        _fh.write(b"the-thesis-detector")
    _arc = webapp._archive_weights(_wp)
    check("an existing detector is archived before a retrain", bool(_arc) and os.path.isfile(_arc), str(_arc))
    check("into weights/archive/", os.path.basename(os.path.dirname(_arc)) == "archive", str(_arc))
    with open(_arc, "rb") as _fh:
        check("the archive is byte-identical", _fh.read() == b"the-thesis-detector")
    with open(_wp, "rb") as _fh:
        check("and the original is left in place", _fh.read() == b"the-thesis-detector")
    check("nothing to archive -> None, not an error",
          webapp._archive_weights(os.path.join(_wd, "absent.pt")) is None)
finally:
    _sh.rmtree(_wd, ignore_errors=True)
check("_train_scene_bg archives BEFORE it trains",
      src.index("archived = _archive_weights(weights_out)") < src.index("gen_dataset.train_scene(scene, dataset_dir"))

section("16. the training status says WHICH classes fail, not just how badly")
check("per-class recall is carried into /status", '"per_class_recall": per_recall' in src)
check("the three worst-recall classes are precomputed for the one-line status", '"worst_recall": worst' in src)
check("the console renders them", "tr.worst_recall" in js and "worst recall" in js)

section("17. the container pass has a control")
_html = open(os.path.join(HERE, "web", "index.html"), encoding="utf-8").read()
check("a 'look inside containers' toggle exists", 'id="scan-containers"' in _html)
check("it is on by default (hidden objects are the point)", 'id="scan-containers" checked' in _html)
check("scanScene sends it to the endpoint that already understood it", "do_containers=" in js)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("MOTION + PERCEPTION SELFTEST OK")
