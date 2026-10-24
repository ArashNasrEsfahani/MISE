"""Self-test for the systematic search skill (`search_for`) and its policy (`sim/search.py`).

Runs headless on Windows `python` from the REPO ROOT — no ai2thor, no GPU, no Ollama, no ultralytics:

    python sim/_search_selftest.py

What this exists to prove, in the order the failures happened on 2026-08-22:

  * a search DECIDES where to look, and knowledge outranks guesswork (§1–3);
  * a 360° sweep is watchable without costing extra inference (§4) — the old code spun a full circle in 8
    rendered frames because rotation granularity and detection cadence were the same number;
  * the skill actually LEAVES where it is standing, opens closed containers, and skips what it cannot reach (§6);
  * and above all it reports **honestly** whether the object was seen (§7) — the bug that had the loop store
    "find the apple" as a proven procedure for a run in which no apple was ever seen.

`sim/thor_actions.py` guards its `ai2thor` import, so `ThorAgent.search_for` is bound to a fake agent here and
driven for real. What that cannot cover is whether YOLO recognises a real apple, or how the walk looks on screen;
those are Track C in HOW-TO-TEST.html.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agentic_rag as ar                                              # noqa: E402
import eval as ev                                                     # noqa: E402
import search                                                         # noqa: E402
import skills as skills_mod                                           # noqa: E402
import thor_actions as ta                                             # noqa: E402
import webapp                                                         # noqa: E402

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


def kinds(stops):
    return [s["kind"] for s in stops]


def types(stops):
    return [s.get("type") for s in stops]


# A small kitchen: a counter close by, a closed fridge further off, a closed cabinet furthest.
RECS = [
    {"objectId": "CounterTop|1", "objectType": "CounterTop", "pos": (1.0, 0.0), "isOpen": False, "openable": False},
    {"objectId": "Fridge|1", "objectType": "Fridge", "pos": (4.0, 0.0), "isOpen": False, "openable": True},
    {"objectId": "Cabinet|1", "objectType": "Cabinet", "pos": (8.0, 0.0), "isOpen": False, "openable": True},
    {"objectId": "GarbageCan|1", "objectType": "GarbageCan", "pos": (2.0, 2.0), "isOpen": False, "openable": False},
]
AT_ORIGIN = (0.0, 0.0)


# ---------------------------------------------------------------------------------------------------------
section("1. the search always starts where it stands")
stops = search.rank_stops("Apple", receptacles=RECS, agent_pos=AT_ORIGIN)
check("first stop is 'here' (one free sweep before walking anywhere)", stops[0]["kind"] == "here")
check("and it is the only 'here'", kinds(stops).count("here") == 1)
check("a search with nothing known still produces stops", len(stops) > 1, str(kinds(stops)))
check("every stop explains itself", all(s.get("why") for s in stops))

section("2. knowledge outranks the built-in prior")
# A remembered sighting must beat everything, even a receptacle the prior likes more and that is nearer.
belief = {"object_type": "Apple", "receptacle": "Cabinet", "confidence": 0.72, "scene": "FloorPlan1"}
stops = search.rank_stops("Apple", belief=belief, receptacles=RECS, agent_pos=AT_ORIGIN)
check("a remembered sighting is visited first", stops[1]["kind"] == "recall" and stops[1]["type"] == "Cabinet",
      str(stops[1]))
check("and it says so in words", "remembered" in stops[1]["why"] and "0.72" in stops[1]["why"], stops[1]["why"])
check("the remembered place is not also queued as a guess", types(stops).count("Cabinet") == 1, str(types(stops)))

# A learned KB fact must beat the curated prior. The prior puts Fridge (0.90) well above GarbageCan (0.10).
kb_texts = ["apples are usually kept in the garbage can in this kitchen"]
stops = search.rank_stops("Apple", kb_texts=kb_texts, receptacles=RECS, agent_pos=AT_ORIGIN)
check("a learned fact outranks the hard-coded prior",
      stops[1]["type"] == "GarbageCan", str(types(stops)))
check("and is attributed to the knowledge base, not the prior",
      "knowledge base" in stops[1]["why"], stops[1]["why"])
stops_noKB = search.rank_stops("Apple", receptacles=RECS, agent_pos=AT_ORIGIN)
check("without that fact the prior wins instead", stops_noKB[1]["type"] == "Fridge", str(types(stops_noKB)))
check("the prior's reason does NOT claim to be knowledge",
      "usually" in stops_noKB[1]["why"] and "knowledge base" not in stops_noKB[1]["why"], stops_noKB[1]["why"])

section("2b. a NOT-FOUND lesson must not teach the search to go back")
# Observed live 2026-08-22: a failed search wrote "WineBottle was NOT at: Cabinet, Floor, Shelf, CounterTop,
# Fridge", and the next search read that as five learned affinities and walked straight back round the same
# five places — the self-improving loop teaching itself to repeat a failure, with the confidence of knowledge.
neg = ["WineBottle was NOT at: here, Cabinet, Shelf (scene FloorPlan1). Try somewhere else first next time."]
recs_n = [{"objectId": "Cabinet|1", "objectType": "Cabinet", "pos": (1.0, 0.0), "isOpen": False, "openable": True},
          {"objectId": "CounterTop|1", "objectType": "CounterTop", "pos": (1.0, 0.0), "isOpen": False,
           "openable": False}]
check("a negative sentence yields NO affinity",
      search.kb_affinity("WineBottle", neg, ["Cabinet", "Shelf"]) == {}, str(search.kb_affinity("WineBottle", neg, ["Cabinet", "Shelf"])))
check("it is read as an exclusion instead",
      search.kb_exclusions("WineBottle", neg, ["Cabinet", "Shelf"]) == {"Cabinet", "Shelf"})
ns = search.rank_stops("WineBottle", kb_texts=neg, receptacles=recs_n, agent_pos=AT_ORIGIN)
check("a ruled-out place is demoted below one that was never checked",
      types(ns)[1] == "CounterTop", str(types(ns)))
check("but it is still on the route — things get moved back",
      "Cabinet" in types(ns), str(types(ns)))
check("and the reason says why it is late in the list",
      "NOT here last time" in next(x["why"] for x in ns if x["type"] == "Cabinet"))
check("a POSITIVE sentence is still read as affinity",
      search.kb_affinity("Mug", ["mugs live in the cabinet"], ["Cabinet"]) == {"Cabinet": 0.95})
check("negation is detected in Persian too", search.is_negative("سیب اینجا نبود"))

section("2c. AI2-THOR type names the prior has to actually match")
# The scene reports `SinkBasin`; a prior keyed on "Sink" matched nothing, so the sink was silently missing from
# every search route until a live run showed it absent.
sink_only = [{"objectId": "SinkBasin|1", "objectType": "SinkBasin", "pos": (1.0, 0.0), "isOpen": False,
              "openable": False}]
check("SinkBasin resolves to the Sink prior",
      types(search.rank_stops("Apple", receptacles=sink_only, agent_pos=AT_ORIGIN))[1] == "SinkBasin",
      str(types(search.rank_stops("Apple", receptacles=sink_only, agent_pos=AT_ORIGIN))))

section("3. ranking details")
check("a closed openable is flagged to be opened",
      any(s["type"] == "Fridge" and s["open"] for s in stops_noKB), str(stops_noKB))
check("an un-openable surface is not", not any(s["type"] == "CounterTop" and s["open"] for s in stops_noKB))
open_rec = [dict(r, isOpen=True) for r in RECS]
st = search.rank_stops("Apple", receptacles=open_rec, agent_pos=AT_ORIGIN)
check("an ALREADY-open container is not opened again", not any(s.get("open") for s in st))
# Distance is a real term, not a tie-break: for a Mug the prior likes Cabinet (0.70) over CounterTop (0.60),
# but the cabinet is 8 m away and the counter is 1 m, so the counter is checked first. Walking past the likely
# place to reach the slightly-more-likely one is worse search, not better.
mug_stops = search.rank_stops("Mug", receptacles=RECS, agent_pos=AT_ORIGIN)
check("a near, slightly-less-likely place is checked before a far, slightly-more-likely one",
      types(mug_stops)[1] == "CounterTop", str(types(mug_stops)))
check("but the far one is still on the route", "Cabinet" in types(mug_stops), str(types(mug_stops)))
far = search.rank_stops("Mug", receptacles=[dict(r, pos=(1.0, 0.0)) for r in RECS], agent_pos=AT_ORIGIN)
check("with distance equalised the prior decides", types(far)[1] == "Cabinet", str(types(far)))
check("an unknown object type still gets a sensible route",
      search.rank_stops("Zorkmid", receptacles=RECS, agent_pos=AT_ORIGIN)[1]["type"] in
      ("CounterTop", "Cabinet", "Fridge"))
check("a sliced variant searches like its base type",
      search.prior_for("AppleSliced") == search.prior_for("Apple"))
st = search.rank_stops("Apple", receptacles=[], agent_pos=AT_ORIGIN,
                       nav_cells=[(5.0, 5.0), (5.2, 5.1), (-6.0, 3.0)])
check("with no receptacles at all it still covers the room", "frontier" in kinds(st), str(kinds(st)))
check("frontier stops spread out rather than clustering",
      len({s["pos"] for s in st if s["kind"] == "frontier"}) == len([s for s in st if s["kind"] == "frontier"]))
check("the stop budget is respected",
      len(search.rank_stops("Apple", receptacles=RECS, agent_pos=AT_ORIGIN, max_stops=3)) <= 3)
check("summarize renders the route for the transcript",
      "->" in search.summarize(stops_noKB) and search.summarize([]) == "(nowhere to look)")

section("4. the sweep is watchable AND cheap — the two are independent")
sc = search.sweep_schedule(8.0, 45.0)
check("a fine sweep is 45 rendered frames", len(sc) == 45, str(len(sc)))
check("but still only 8 inferences", sum(1 for _, d in sc if d) == 8, str(sum(1 for _, d in sc if d)))
old = search.sweep_schedule(45.0, 45.0)
check("the OLD behaviour was 8 frames for the same 8 inferences",
      len(old) == 8 and sum(1 for _, d in old if d) == 8, str(len(old)))
check("a full circle is always covered", abs(sum(d for d, _ in sc) - 360.0) < 1e-6)
check("the last step always detects, so a sweep never ends blind", sc[-1][1] is True)
check("detect_every below turn cannot detect less often than every frame",
      all(d for _, d in search.sweep_schedule(30.0, 5.0)))
check("a nonsense turn does not hang or divide by zero", len(search.sweep_schedule(0.0, 45.0)) > 0)

section("5. budgets and the not-found lesson")
check("thoroughness scales with the motion profile",
      search.search_budget("fast")["max_stops"] < search.search_budget("normal")["max_stops"]
      < search.search_budget("cinematic")["max_stops"])
check("every budget is time-capped", all(search.search_budget(k)["max_seconds"] > 0
                                         for k in ("fast", "normal", "cinematic")))
check("an unknown speed falls back to normal, it does not raise",
      search.search_budget("zzz") == search.search_budget("normal"))
les = search.not_found_lesson("Apple", "FloorPlan1", ["CounterTop", "Fridge"])
check("the lesson names the places ruled out",
      "CounterTop" in les["lesson"] and "Fridge" in les["lesson"], les["lesson"])
check("and the scene, so it is not applied to the wrong kitchen", "FloorPlan1" in les["lesson"])
check("a search that reached nowhere still produces a usable lesson",
      "nowhere reachable" in search.not_found_lesson("Apple", "FloorPlan1", [])["lesson"])


# ---------------------------------------------------------------------------------------------------------
class FakeAg:
    """The controller surface `search_for` touches, with every call recorded.

    `thor_actions` guards its ai2thor import, so the REAL `ThorAgent.search_for` generator is bound to this and
    driven — these are not assertions about a reimplementation.
    """

    def __init__(self, apple_at=None, unreachable=(), objects=None):
        self.calls = []
        self.success = True
        self.error = ""
        self.motion = {"step": 0.15, "turn": 15.0}
        self.scene = "FloorPlan1"
        self.last_search = {}
        self.at = None                      # objectId of the stop we are standing at
        self.apple_at = apple_at            # objectId whose sweep reveals the Apple; None = nowhere
        self.unreachable = set(unreachable)
        self._objects = objects if objects is not None else [
            dict(r, position={"x": r["pos"][0], "y": 0.9, "z": r["pos"][1]}, receptacle=True) for r in RECS]

    # -- queries
    def agent_pose(self):
        return {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"y": 0.0},
                "horizon": 0.0, "standing": True}

    def objects(self):
        return self._objects

    def object_by_id(self, oid):
        return next((o for o in self._objects if o["objectId"] == oid), None)

    def find(self, object_type, visible_only=False):
        self.calls.append(("find", object_type))
        return [{"objectType": object_type}] if (self.apple_at and self.at == self.apple_at) else []

    def detect_objects(self, det, conf=0.25):
        self.calls.append(("detect",))
        return [{"type": "Apple", "state": "whole"}] if (self.apple_at and self.at == self.apple_at) else []

    # -- primitives
    def walk_to(self, oid, **kw):
        self.calls.append(("walk_to", oid))
        if oid in self.unreachable:
            yield "no path"
            return
        for _ in range(3):
            yield "stepping"
        self.at = oid
        yield "arrived"

    def teleport(self, x, y, z, **kw):
        self.calls.append(("teleport", round(x, 2), round(z, 2)))
        self.at = None
        self.success = True

    def walk_to_point(self, x, z, **kw):
        self.calls.append(("walk_to_point", round(x, 2), round(z, 2)))
        for _ in range(3):
            yield "stepping"
        self.at = None
        yield "arrived"

    def open(self, oid, openness=1.0, force=False):
        self.calls.append(("open", oid, round(float(openness), 3)))
        self.success = True
        o = self.object_by_id(oid)
        if o:
            o["openness"], o["isOpen"] = float(openness), float(openness) > 0
        return True

    def close(self, oid, force=False):
        self.calls.append(("close", oid))
        self.success = True
        o = self.object_by_id(oid)
        if o:
            o["openness"], o["isOpen"] = 0.0, False    # the fake must model the state change it is asked about
        return True

    def rotate_right(self, deg=None):
        self.calls.append(("rotate_right", deg))
        self.success = True


class FakeMem:
    def __init__(self, belief=None):
        self.belief = belief
        self.writes = []

    def query_location(self, scene, object_type, track=None):
        return self.belief

    def get_nav_cells(self, scene):
        return []

    def observe_location(self, *a, **kw):
        self.writes.append((a, kw))
        return {"action": "insert", "id": 1}


# `search_for` calls two private helpers on `self`. Bind the REAL implementations onto the fake rather than
# stubbing them, so what runs below is the shipped generator, not a lookalike.
FakeAg._search_sees = ta.ThorAgent._search_sees
FakeAg._swing = ta.ThorAgent._swing


class FakeMemNav(FakeMem):
    """Supplies discovered nav cells, so `rank_stops` produces frontier coverage to walk to."""

    def get_nav_cells(self, scene):
        return [{"gx": 8, "gz": 8, "state": "free"}, {"gx": -8, "gz": 4, "state": "free"}]


def run_search(ag, **kw):
    """Drive the real generator to exhaustion and return the frames it rendered."""
    return list(ta.ThorAgent.search_for(ag, "Apple", **kw))


section("6. the skill actually leaves where it is standing")
# The apple is in the Cabinet, which the prior ranks LAST — so the search has to work its way there.
ag = FakeAg(apple_at="Cabinet|1")
frames = run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
walked = [c[1] for c in ag.calls if c[0] == "walk_to"]
check("it walks to several stops, in ranked order", len(walked) >= 3, str(walked))
check("the likeliest place is tried first", walked[0] == "Fridge|1", str(walked))
check("it opened each closed container before looking inside",
      {"Fridge|1", "Cabinet|1"} <= {c[1] for c in ag.calls if c[0] == "open"},
      str({c[1] for c in ag.calls if c[0] == "open"}))
check("the door SWINGS rather than snapping open",
      len([c for c in ag.calls if c[0] == "open" and c[1] == "Fridge|1"]) > 3,
      str(len([c for c in ag.calls if c[0] == "open" and c[1] == "Fridge|1"])))
check("it rendered enough frames to be watchable", len(frames) > 100, f"{len(frames)} frames")
check("it found the apple where it actually was", ag.last_search["receptacle"] == "Cabinet",
      str(ag.last_search))
check("the last frame announces the find", frames[-1] == "found Apple", frames[-1])

# ...and it must stop the moment it sees the thing, rather than finishing the route out of habit.
ag = FakeAg(apple_at="Fridge|1")
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
check("a find at the first stop ends the search immediately",
      len([c for c in ag.calls if c[0] == "walk_to"]) == 1, str(ag.calls[:3]))
check("and that is reflected in the outcome",
      ag.last_search["found"] and ag.last_search["stops_visited"] < ag.last_search["stops_planned"],
      str(ag.last_search))

ag = FakeAg(apple_at="Cabinet|1", unreachable={"Fridge|1"})
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
check("an unreachable stop is skipped, not counted as searched",
      "Fridge" not in ag.last_search["visited"], str(ag.last_search["visited"]))
check("and the search carries on to the next one", ag.last_search["found"] is True)

section("6b. it leaves the kitchen as it found it")
# Reported from live use: the search opened the fridge to look inside and left it hanging open.
ag = FakeAg(apple_at=None)
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
opened = {c[1] for c in ag.calls if c[0] == "open"}
check("a failed search opens containers", len(opened) >= 2, str(opened))
check("and closes every one of them again",
      all(not (ag.object_by_id(o) or {}).get("isOpen") for o in opened),
      str({o: (ag.object_by_id(o) or {}).get("isOpen") for o in opened}))

# ...including the one the object was found in. A follow-up "pick it up" re-opens it by itself
# (webapp._open_parent), so there is no reason to leave the kitchen standing open on the chance of one.
ag = FakeAg(apple_at="Fridge|1")
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
check("even the container the object was found in is closed again",
      (ag.object_by_id("Fridge|1") or {}).get("isOpen") is False)

ag = FakeAg(apple_at="Cabinet|1")
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
check("and so is everything opened on the way there",
      not any((ag.object_by_id(k) or {}).get("isOpen") for k in ("Fridge|1", "Cabinet|1")),
      str({k: (ag.object_by_id(k) or {}).get("isOpen") for k in ("Fridge|1", "Cabinet|1")}))
check("a container it never opened is not touched",
      not any(c[0] == "close" and c[1] == "CounterTop|1" for c in ag.calls))

section("6b2. reaching for something inside a closed container opens it first")


class _Inside:
    """Two objects: an Apple inside a closed Fridge."""

    def __init__(self, fridge_open=False):
        self.calls = []
        self.success = True
        self.error = ""
        self.motion = {"step": 0.15, "turn": 15.0}
        self.objs = {
            "Fridge|1": {"objectId": "Fridge|1", "objectType": "Fridge", "openable": True,
                         "isOpen": fridge_open, "openness": 1.0 if fridge_open else 0.0},
            "Apple|1": {"objectId": "Apple|1", "objectType": "Apple", "isPickedUp": False,
                        "parentReceptacles": ["Fridge|1"], "distance": 0.4},
        }

    def object_by_id(self, oid):
        return self.objs.get(oid)

    def open(self, oid, openness=1.0, force=False):
        self.calls.append(("open", oid))
        self.success = True
        self.objs[oid]["openness"], self.objs[oid]["isOpen"] = float(openness), float(openness) > 0
        return True

    def close(self, oid, force=False):
        self.calls.append(("close", oid))
        self.success = True
        return True

    def walk_to(self, oid, **kw):
        self.calls.append(("walk_to", oid))
        yield "arrived"

    def close_in(self, oid, **kw):
        return iter(())

    def pickup(self, oid, force=False):
        self.calls.append(("pickup", oid))
        self.success = True
        return True


shut_fridge = _Inside(fridge_open=False)
list(webapp._approach_then(shut_fridge, "Apple|1", lambda: shut_fridge.pickup("Apple|1"), "pickup Apple"))
check("the closed container is opened before reaching in",
      ("open", "Fridge|1") in shut_fridge.calls, str(shut_fridge.calls))
check("and it walks to the container, not just the object",
      ("walk_to", "Fridge|1") in shut_fridge.calls and ("walk_to", "Apple|1") in shut_fridge.calls,
      str(shut_fridge.calls))
check("the action still happens", ("pickup", "Apple|1") in shut_fridge.calls)
check("and it is CLOSED again once the action is done",
      ("close", "Fridge|1") in shut_fridge.calls, str(shut_fridge.calls))
check("closing happens after the action, not before",
      shut_fridge.calls.index(("pickup", "Apple|1")) < shut_fridge.calls.index(("close", "Fridge|1")),
      str(shut_fridge.calls))

# A door the operator (or an earlier explicit `open` step) left open is not this step's to close. Getting this
# wrong is how a tidy-up starts undoing things the user asked for.
open_fridge = _Inside(fridge_open=True)
list(webapp._approach_then(open_fridge, "Apple|1", lambda: open_fridge.pickup("Apple|1"), "pickup Apple"))
check("an already-open container is not re-opened",
      not any(c[0] == "open" for c in open_fridge.calls), str(open_fridge.calls))
check("and an already-open container is NOT closed by us",
      not any(c[0] == "close" for c in open_fridge.calls), str(open_fridge.calls))

section("6c. an unsearched corner is WALKED to, not teleported into")
ag = FakeAg(apple_at=None, objects=[])          # no receptacles at all -> only frontier stops remain
run_search(ag, memory=FakeMemNav(), kb=None, scene="FloorPlan1", speed="normal")
check("frontier stops use the walker", any(c[0] == "walk_to_point" for c in ag.calls), str(ag.calls[:4]))
check("and never teleport", not any(c[0] == "teleport" for c in ag.calls), str(ag.calls[:4]))

section("7. honesty — a search that finds nothing must SAY so")
ag = FakeAg(apple_at=None)
frames = run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal")
check("found is False when nothing was seen", ag.last_search["found"] is False)
check("it records every place it looked", len(ag.last_search["visited"]) >= 2, str(ag.last_search["visited"]))
check("it says why it stopped", bool(ag.last_search.get("reason")), str(ag.last_search))
check("the last frame announces the failure", frames[-1] == "Apple not found", frames[-1])
check("it searched the whole plan before giving up",
      ag.last_search["stops_visited"] == ag.last_search["stops_planned"], str(ag.last_search))

section("8. memory is consulted, and a foreign-scene belief is rejected")
mem = FakeMem({"object_type": "Apple", "receptacle": "Cabinet", "confidence": 0.9, "scene": "FloorPlan1"})
ag = FakeAg(apple_at="Cabinet|1")
run_search(ag, memory=mem, kb=None, scene="FloorPlan1", speed="normal")
first_walk = next(c[1] for c in ag.calls if c[0] == "walk_to")
check("a remembered location is visited FIRST", first_walk == "Cabinet|1", first_walk)

# query_location does not filter by scene in SQL — scene is only a ranking term — so a belief from another
# kitchen comes back looking authoritative. The skill must reject it.
mem2 = FakeMem({"object_type": "Apple", "receptacle": "Cabinet", "confidence": 0.9, "scene": "FloorPlan7"})
ag2 = FakeAg(apple_at="Cabinet|1")
run_search(ag2, memory=mem2, kb=None, scene="FloorPlan1", speed="normal")
first_walk2 = next(c[1] for c in ag2.calls if c[0] == "walk_to")
check("a belief from a DIFFERENT scene is ignored", first_walk2 != "Cabinet|1", first_walk2)

section("9. the sensor respects the perception arm")
ag = FakeAg(apple_at="Fridge|1")
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal", detector=object())
check("perceived mode asks the DETECTOR", any(c[0] == "detect" for c in ag.calls))
check("and never falls back to the metadata lookup", not any(c[0] == "find" for c in ag.calls))
ag = FakeAg(apple_at="Fridge|1")
run_search(ag, memory=None, kb=None, scene="FloorPlan1", speed="normal", detector=None)
check("oracle mode uses the visible-flag sighting", any(c[0] == "find" for c in ag.calls))
check("and never runs a detector", not any(c[0] == "detect" for c in ag.calls))

section("10. bilingual intent, and no false positives")
for cmd, want in (("find the apple", "Apple"), ("can you find the apple?", "Apple"),
                  ("where is the mug", "Mug"), ("look for the knife", "Knife"),
                  ("سیب را پیدا کن", "Apple"),
                  ("دنبال لیوان بگرد", "Mug")):
    lang = "fa" if any(ch > chr(0x600) for ch in cmd) else "en"
    check(f"[{lang}] a search command resolves to {want}", webapp._search_target(cmd) == want,
          repr(webapp._search_target(cmd)))
for cmd in ("pick up the apple", "open the fridge", "slice the bread",
            "یخچال را باز کن"):
    check("a non-search command is NOT treated as a search", webapp._search_target(cmd) is None, repr(cmd))

section("11. plan repair upgrades a spin-in-place plan")
# The knowledge base already holds two `find the apple -> [look_around]` procedures. The bandit replays them, so
# the repair — not a hand-edit of the user's KB — is what turns each replay into a real search.
steps, changes = webapp._repair_plan_for_command("find the apple", [{"action": "look_around", "object": None}])
check("look_around becomes search_for", [s["action"] for s in steps] == ["search_for"], str(steps))
check("with the right target", steps[0]["object"] == "Apple", str(steps[0]))
check("and the repair is reported, not silent", any("search_for" in c for c in changes), str(changes))
steps, _ = webapp._repair_plan_for_command("find the apple", [])
check("an empty plan gets a search injected", [s["action"] for s in steps] == ["search_for"], str(steps))
steps, changes = webapp._repair_plan_for_command("find the apple",
                                                 [{"action": "search_for", "object": "Apple"}])
check("an already-correct plan is left alone", changes == [], str(changes))
steps, _ = webapp._repair_plan_for_command("open the fridge", [{"action": "look_around", "object": None}])
check("a non-search command is not upgraded", steps[0]["action"] == "look_around", str(steps))
steps, _ = webapp._repair_plan_for_command("pick up the apple",
                                           [{"action": "search_for", "object": "Apple"}])
check("search_for counts as an approach, so no redundant go_to is injected",
      [s["action"] for s in steps] == ["search_for", "pickup"], str([s["action"] for s in steps]))

# One command, one target. "find the wine bottle" pivoted to WineBottle while the LLM planned `search_for
# Bottle`; the search then succeeded on its own terms and the goal check reported the object was never seen.
steps, changes = webapp._repair_plan_for_command("find the wine bottle",
                                                 [{"action": "search_for", "object": "Bottle"}])
check("a search step naming a type the command never asked for is retargeted",
      steps[0]["object"] == "WineBottle", str(steps))
check("and the retarget is reported", any("retargeted" in c for c in changes), str(changes))
steps, changes = webapp._repair_plan_for_command("find the apple and the knife",
                                                 [{"action": "search_for", "object": "Knife"}])
check("but a step naming one of SEVERAL requested types is left alone",
      steps[0]["object"] == "Knife" and changes == [], str((steps, changes)))
check("_search_types returns every type the command named",
      webapp._search_types("find the apple and the knife") == ["Apple", "Knife"]
      or set(webapp._search_types("find the apple and the knife")) == {"Apple", "Knife"},
      str(webapp._search_types("find the apple and the knife")))

section("12. registered everywhere it has to be")
check("in the skills manifest at L2",
      any(s["name"] == "search_for" and s["level"] == "L2" for s in skills_mod.skills_manifest()))
import agent_llm as al_mod
check("in the planner prompt", "search_for" in al_mod.PLANNER_SYSTEM)
check("in the console's composite dispatch",
      "\"search_for\": lambda: _search_gen(ag, ot)" in open(os.path.join(HERE, "webapp.py"),
                                                            encoding="utf-8").read())
check("in BOTH verb lexicons", ar.EN_VERB_LEXICON.get("find") == "search_for"
      and "search_for" in ar.FA_VERB_LEXICON.values())
check("NOT in eval's dropped set — it is a scored action", "search_for" not in ev._NAV_HELPERS)
check("--score-search off restores the old action space", "search_for" in
      (lambda: (ev._SCORE_SEARCH.__setitem__("on", False), ev._dropped_actions(),
                ev._SCORE_SEARCH.__setitem__("on", True))[1])())
# the two plan-repair copies must agree, or the harness and the live console repair the same plan differently
ws = open(os.path.join(HERE, "webapp.py"), encoding="utf-8").read()
es = open(os.path.join(HERE, "eval.py"), encoding="utf-8").read()
approach = '{"go_to", "face_object", "scan_for", "search_for"}'
check("eval and the console share one approach-step set",
      approach in ws and approach in es)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("SEARCH SELFTEST OK")
