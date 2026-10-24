"""Offline selftest for console v2's honest mode (sim/honest_console.py + its webapp.py wiring).

Windows `python` from the REPO ROOT, no WSL / GPU / Ollama:   python sim/_console_honest_selftest.py

The REAL step-4 modules run (honest.HonestAgent, honest_skills.HonestSkills, objmem.ObjectMemory, honest_map
GridMap, step4_gemma.GemmaPlanner) over the fake kitchen of sim/_step4_act_selftest.py (loaded from that file, so
there is one fake world). Only the outside is faked: the simulator thread (a real worker thread with the webapp's
job contract), Ollama (a scripted provider that can block), the display hook, the goal check and the store.

  1. the lease: claim / refuse / release-by-holder-only / generation expiry / atomic admit
  2. mode switch: refused when unavailable; on = one sensor reload, world 1, fresh session; payload shape
  3. an EN instruction end to end: plan -> execute -> done; memory, map, audit (0 violations, 0 oracle reads), archive
  4. lifetimes: same memory episode across instructions (no second sweep); Forget -> new episode, hand kept;
     world change -> new world, hand empty / kept / unknown; tainted run -> holding '?', forced new episode
  5. FA instruction (lang detected), dataset item (level from the dataset, goal check only inside h.scoring()),
     setup items refused, rejected plans not executed, LLM off refused
  6. Stop: during planning (late reply discarded), at a primitive boundary (a successful pickup is recorded first),
     mid-step, during the search chooser's LLM wait (sim.waiting shown; cancel escapes `except Exception`)
  7. failures: stale work after a generation change is dropped; a simulator restart mid-instruction ends it
     'failed: simulator restarted' and the next run starts a new world with an empty hand; a non-simulator
     exception never escapes the job; a simulator-call exception does; a display-hook failure never erases an
     action; a render trip aborts before the next primitive
  8. display: last-look JPEG, the detection overlay is valid only while the action revision is unchanged
  9. invariance: renamed simulator ids + decoy records produce the same payload
 10. webapp wiring over HTTP: /honest*, /status.honest + lease, the honest gate (truth routes, idle routes),
     /cmd lease arbitration, API version 7 on both sides
"""
from __future__ import annotations

import copy
import json
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# sim/select.py (the bandit) shadows the stdlib `select` that the HTTP server needs: import the stdlib one first
_saved_path = list(sys.path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import select                                                         # noqa: E402,F401
import selectors                                                      # noqa: E402,F401
from http.server import ThreadingHTTPServer                           # noqa: E402
sys.path[:] = _saved_path
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402

import honest                                                         # noqa: E402
import honest_console as hc                                           # noqa: E402
import honest_skills as hs                                            # noqa: E402
import objmem                                                         # noqa: E402
import step4_gemma as sg                                              # noqa: E402
from honest_map import GridMap                                        # noqa: E402

PASS = FAIL = 0
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def section(t):
    print(f"\n{t}")


# ----------------------------------------------------------------------------------------- the shared fake world
_src = open(os.path.join(HERE, "_step4_act_selftest.py"), encoding="utf-8").read()
FW = {"np": np, "math": math, "objmem": objmem, "hs": hs}
exec(compile(_src[_src.index("W = H = 100"):_src.index("def setup(")], "_step4_act_selftest(fake world)", "exec"), FW)
FREE, CELL, kitchen = FW["FREE"], FW["CELL"], FW["kitchen"]


class FakeAgent(FW["FakeThor"]):
    """The fake kitchen as the console's raw simulator agent (+ calibration, sensors, a failure switch)."""

    def __init__(self, objects=None, **kw):
        super().__init__(objects or kitchen(), **kw)
        self.field_of_view = 90.0
        self.sensor_calls = []
        self.raise_on = {}                 # action -> exception instance raised by the simulator call itself

    def do(self, action, **kw):
        exc = self.raise_on.get(action)
        if exc is not None:
            raise exc
        return super().do(action, **kw)

    def set_sensors(self, **kw):
        self.sensor_calls.append(kw)


class FakeDetector(FW["FakeDetector"]):
    def __init__(self, thor):
        super().__init__(thor)
        self.fail = None
        self.last_inspecting = False

    def detect(self, rgb, conf=0.25, inspecting=None):
        if self.fail:
            raise self.fail
        self.last_inspecting = bool(inspecting)
        return super().detect(rgb, conf, inspecting)


class Sim:
    """The console's simulator thread: the webapp's job contract (an escaping exception = simulator died)."""

    def __init__(self, ag):
        self.ag, self.q, self.gen = ag, queue.Queue(), 1
        self.died, self.ran = [], 0
        self.on_restart = None
        self.gate = threading.Event()
        self.gate.set()
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def loop(self):
        while not self.stop.is_set():
            try:
                job = self.q.get(timeout=0.02)
            except queue.Empty:
                continue
            self.gate.wait()
            try:
                job(self.ag)
                self.ran += 1
            except BaseException as e:      # what the real supervisor does: restart + drain
                self.died.append(e)
                self.restart()

    def restart(self, ag=None):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.ag = ag or FakeAgent()
        self.gen += 1
        if self.on_restart:
            self.on_restart(self.ag)


class Provider:
    """Scripted Ollama: plan replies by command keyword; chooser replies; optional blocking."""

    def __init__(self):
        self.plans = {}
        self.default = {"say": "ok", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]}
        self.messages = []
        self.block_plan = None             # threading.Event the plan call waits on
        self.block_chooser = None
        self.chooser_calls = 0
        self.last_meta = {}

    def chat(self, model, messages, timeout=None, fmt=None, options=None, think=None):
        self.last_meta = {"done_reason": "stop"}
        if fmt is sg.CHOOSER_SCHEMA:
            self.chooser_calls += 1
            if self.block_chooser is not None:
                self.block_chooser.wait(10)
            return json.dumps({"places": ["CounterTop", "Fridge"]})
        self.messages.append(messages)
        if self.block_plan is not None:
            self.block_plan.wait(10)
        cmd = messages[1]["content"].rsplit("Command:", 1)[-1].strip()
        for k, v in self.plans.items():
            if k in cmd:
                return json.dumps(v)
        return json.dumps(self.default)


def make(objects=None, avail=(True, ""), hook=None, goal=None, out=None):
    ag = FakeAgent(objects)
    sim = Sim(ag)
    det = FakeDetector(ag)
    prov = Provider()
    state = {"llm": True, "rev": 0, "applied": [], "hook": hook, "prims": []}

    def on_prim(a, action):
        state["prims"].append(action)
        if action not in ("Pass", "GetObjectInFrame"):
            state["rev"] += 1
        if state["hook"]:
            state["hook"](a, action)

    def apply(a, on):
        state["applied"].append(on)
        a.set_sensors(depth=on)

    items = [{"id": "g-l1-pickup-apple", "level": "L1", "en": "pick up the apple", "fa": "سیب را بردار",
              "plan": [{"action": "pickup", "object": "Apple"}], "goal": "Apple isPickedUp"},
             {"id": "g-l1-needs-setup", "level": "L1", "en": "close the cabinet", "fa": "کابینت را ببند",
              "setup": [{"action": "open", "object": "Cabinet"}]}]
    c = hc.Console(submit=sim.q.put, sim_generation=lambda: sim.gen, on_primitive=on_prim,
                   goal_check=goal, load_detector=lambda: FakeDetector(sim.ag), make_provider=lambda ep: prov,
                   apply_sensors=apply, act_rev=lambda: state["rev"], items=lambda: items,
                   availability=lambda: avail, load_map=lambda: GridMap("Fake", CELL, (0.0, 0.0), FREE),
                   make_store=None, llm_on=lambda: state["llm"], out_dir=out or tempfile.mkdtemp(),
                   wait_tick=0.01, work_timeout=30)
    c._test = state
    c.load_detector = lambda: det
    sim.on_restart = lambda a: setattr(det, "thor", a)
    return c, sim, det, prov, state


def fresh(c, sim, reason="scene reset"):
    """A real scene reload: the same agent object, back to the start world (like ThorAgent.reset)."""
    done = threading.Event()

    def job(ag):
        FakeAgent.__init__(ag)
        done.set()
    sim.q.put(job)
    done.wait(5)
    c.world_changed(reason, "empty")


def wait_final(c, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with c.lock:
            j = c.job
            if j and j["phase"] in hc.FINAL and c.lease.owner() is None:
                return j["phase"]
        time.sleep(0.01)
    return "timeout"


def wait_for(fn, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.005)
    return False


# ======================================================================================================== 1
section("1. the lease")
g = {"gen": 1}
L = hc.Lease(lambda: g["gen"])
ok1, _ = L.claim("honest", "j1")
ok2, holder = L.claim("oracle-cmd", "j2")
check("first claim wins, second is refused with the holder", ok1 and not ok2 and holder["owner"] == "honest")
check("only the holder can release", not L.release("j2") and L.owner() == "honest" and L.release("j1"))
L.claim("scan", "j3")
g["gen"] = 2
check("a simulator-bound lease expires when the generation changes", L.owner() is None)
L.claim("eval", "j4", sim_bound=False)
g["gen"] = 3
check("an eval lease is not bound to the simulator", L.owner() == "eval")
L.release("j4")
ran = []
okA, _ = L.admit(("honest",), lambda: ran.append(1))
L.claim("honest", "j5")
okB, h5 = L.admit(("honest",), lambda: ran.append(2))
check("admit runs the enqueue under the lease lock, refused while a blocking owner holds it",
      okA and not okB and ran == [1] and h5["owner"] == "honest")
L.release("j5")
try:
    L.claim("nobody", "x")
    bad_owner = False
except ValueError:
    bad_owner = True
check("unknown owners are rejected", bad_owner)

# ======================================================================================================== 2
section("2. mode switch")
c, sim, det, prov, st = make(avail=(False, "honest mode is FloorPlan1 only (the scene is FloorPlan5)"))
r = c.switch(True)
check("refused when unavailable, with the reason", not r["ok"] and "FloorPlan1" in r["error"] and not c.on)
check("no sensor reload on a refusal", st["applied"] == [])
c.availability_fn = lambda: (True, "")
r = c.switch(True)
check("on: ok, one sensor apply, world 1, fresh session", r["ok"] and c.on and st["applied"] == [True]
      and c.world_gen == 1 and c.session["world_gen"] == 1, str(r))
check("the lease is free again after the switch", c.lease.owner() is None)
snap = c.snapshot(full=True)
KEYS = {"ok", "on", "available", "model", "endpoint", "assumptions", "session", "job", "outcome", "planner", "steps",
        "holding", "memory", "places_seen", "map", "last_look", "audit", "history", "lease", "waiting", "rev"}
check("payload has the spec 2.7 keys", KEYS <= set(snap), str(sorted(KEYS - set(snap))))
check("payload declares the two standing assumptions", len(snap["assumptions"]) == 2)
check("payload is JSON-serialisable", bool(json.dumps(snap, ensure_ascii=False, default=None)))
check("map carries the static grid only with full=1", "free" in (snap["map"] or {}) and "free" not in
      (c.snapshot()["map"] or {}))
check("switching on again is a no-op", c.switch(True)["ok"] and st["applied"] == [True])

# ======================================================================================================== 3
section("3. an EN instruction end to end")
r = c.run("pick up the apple")
check("run starts with a job id", r["ok"] and r["job_id"].startswith("i"), str(r))
ph = wait_final(c)
snap = c.snapshot()
check("instruction done", ph == "done", f"{ph} {snap['job']}")
check("lang detected en, level not classified (free text)", snap["job"]["lang"] == "en"
      and snap["job"]["level_source"] == "none" and snap["job"]["level"] is None)
check("planner: valid first pass, plan [pickup Apple]", snap["planner"]["ok"] and snap["planner"]["first_ok"]
      and [(s["action"], s["object"]) for s in snap["planner"]["plan"]] == [("pickup", "Apple")], str(snap["planner"]))
s0 = snap["steps"][0]
check("step 0 ok, with prims, verify and events", s0["status"] == "ok" and s0["prims"] > 0 and s0["verify"]
      and isinstance(s0["events"], list), str(s0))
check("the agent's own holding belief is Apple", snap["holding"] == "Apple")
check("memory has tracks incl. a held Apple", any(t["type"] == "Apple" and t["status"] == "held"
                                                  for t in snap["memory"]["tracks"]), str(snap["memory"])[:300])
check("memory summary lines present", bool(snap["memory"]["lines"]))
check("map: agent drawn from the agent's own last pose (cell + step)", snap["map"]["agent"] is not None
      and snap["map"]["agent"]["step"] is not None and (snap["map"]["agent"]["gx"], snap["map"]["agent"]["gz"]) in FREE)
a = snap["audit"]
check("audit: valid, 0 violations, 0 oracle / scoring / map-build reads, pose reads counted",
      a["invalid"] is False and a["violations"] == 0 and a["counts"]["oracle"] == 0 and a["counts"]["scoring"] == 0
      and a["counts"]["map-build"] == 0 and a["counts"]["pose"] > 0, str(a))
check("outcome: executed 1/1, goal not checked (free text), audit valid",
      snap["outcome"]["exec"] == {"ok_steps": 1, "n_steps": 1} and snap["outcome"]["goal"] is None
      and snap["outcome"]["audit"]["valid"])
check("history has the instruction with its ticket hidden from the list", len(snap["history"]) == 1
      and "ticket" not in snap["history"][0] and c.history_ticket(snap["history"][0]["id"])["steps"])
rows = [json.loads(x) for x in open(os.path.join(c.out_dir, hc.RUNS_FILE), encoding="utf-8")]
check("archived: one row with world, memory episode, code hash, audit", len(rows) == 1 and rows[0]["world_gen"] == 1
      and rows[0]["memory_episode"].startswith("console:") and len(rows[0]["code_hash"]) == 16
      and rows[0]["audit"]["violations"] == 0, str(rows[0])[:300])
check("planner got memory summary + holding, never simulator truth",
      "The robot is holding: nothing" in prov.messages[0][1]["content"]
      and not re.search(r"\|[+-]\d", json.dumps(prov.messages[0])))
check("no simulator id anywhere in the payload", not re.search(r"\|[+-]\d", json.dumps(snap, default=str)))
check("worker never died", sim.died == [])
ep1 = c.session["episode"]
sweeps_before = sum(1 for p in st["prims"] if p == "RotateRight")

# ======================================================================================================== 4
section("4. lifetimes")
prov.plans["put it"] = {"say": "ok", "plan": [{"action": "put_in", "object": "CounterTop", "to": None, "arg": None}]}
c.run("put it on the counter")
ph = wait_final(c)
check("2nd instruction: same memory episode (memory carried)", c.session["episode"] == ep1 and ph == "done",
      f"{ph} {c.session['episode']} {c.snapshot()['job']}")
check("2nd instruction: planner told the agent holds Apple", "The robot is holding: Apple" in
      prov.messages[-1][1]["content"])
check("2nd instruction: its own audit episode", c.h.audit.episode == c.job["id"])
check("Forget is refused while acting and allowed idle", c.forget()["ok"])
prov.plans["take the apple"] = {"say": "ok", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]}
c.run("take the apple")
wait_final(c)
check("after Forget: a new memory episode m2", c.session["episode"].endswith(":m2"), c.session["episode"])
check("after Forget: the hand is kept (Apple held again after this run)", c.h.holding == "Apple")
c.forget()
c.run("put it on the counter")
wait_final(c)
check("Forget kept the hand: planner told 'holding Apple' in the new episode",
      "The robot is holding: Apple" in prov.messages[-1][1]["content"] and c.session["episode"].endswith(":m3"))
# world changes
wg = c.world_gen
c.world_changed("shuffled (materials)", "keep")
check("world change: world_gen + 1, payload reflects it", c.world_gen == wg + 1 and
      c.snapshot()["session"]["world_gen"] == wg + 1)
c.run("take the apple")
wait_final(c)
check("new world (keep): new memory episode, rebuilt agent", c.session["episode"].endswith(f"w{wg + 1}:m4"),
      c.session["episode"])
fresh(c, sim)
prov.plans["what"] = {"say": "ok", "plan": [{"action": "crouch", "object": None, "to": None, "arg": None},
                                           {"action": "stand", "object": None, "to": None, "arg": None}]}
c.run("what do you see")
wait_final(c)
check("new world (reload): hand empty", "The robot is holding: nothing" in prov.messages[-1][1]["content"])
c.world_changed("shuffled (spawn)", "unknown")
c.run("what do you see")
wait_final(c)
check("new world (spawn): hand unknown '?'", "The robot is holding: ?" in prov.messages[-1][1]["content"])
fresh(c, sim)
c.intervention("moved by hand", 3)
snap = c.snapshot()
check("intervention counted on the session, pose marked stale",
      {"kind": "moved by hand", "n": 3} in snap["session"]["interventions"] and c.session["pose_stale"])
c.run("what do you see")
wait_final(c)
check("a fresh pose read clears the stale flag", c.snapshot()["map"]["agent"]["stale"] is False)
# taint: a prohibited read during the run (swallowed by the display hook) still invalidates it
st["hook"] = lambda a, action: c.h.meta("oracle") if action == "Crouch" else None
c.run("what do you see")
ph = wait_final(c)
snap = c.snapshot()
check("a violation (even swallowed) ends the run INVALID", snap["outcome"]["audit"]["valid"] is False
      and snap["history"][-1]["valid"] is False and snap["audit"]["violations"] >= 1, str(snap["outcome"]))
st["hook"] = None
check("tainted session flagged", snap["session"]["tainted"] is True)
c.run("what do you see")
wait_final(c)
check("after a tainted run: forced new episode, holding '?'", "The robot is holding: ?" in
      prov.messages[-1][1]["content"] and c.snapshot()["outcome"]["audit"]["valid"])

# ======================================================================================================== 5
section("5. FA, dataset items, rejections")
fresh(c, sim)
prov.plans["سیب"] = {"say": "باشه", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]}
c.run("سیب را بردار")
wait_final(c)
snap = c.snapshot()
check("FA instruction: lang fa, planned and done", snap["job"]["lang"] == "fa" and snap["job"]["phase"] == "done"
      and snap["planner"]["say"] == "باشه", str(snap["job"]))
calls = []


def goal(ag, item):
    calls.append(c.h._phase)
    return {"ok": True, "unmet": []}


c.goal_check = goal
fresh(c, sim)
r = c.run(item="g-l1-pickup-apple", lang="en")
wait_final(c)
snap = c.snapshot()
check("dataset item: text from the item, level from the dataset", r["ok"] and snap["job"]["command"] ==
      "pick up the apple" and snap["job"]["level"] == "L1" and snap["job"]["level_source"] == "dataset")
check("goal check ran only inside h.scoring() (pre-check + after the plan)", calls == ["scoring", "scoring"],
      str(calls))
check("goal result is on the ticket, scoring reads counted only in the audit",
      snap["outcome"]["goal"]["ok"] is True and snap["outcome"]["goal"]["checked"]
      and snap["audit"]["counts"]["scoring"] == 2 and snap["audit"]["invalid"] is False, str(snap["audit"]))
check("the goal never reached the planner prompt", "isPickedUp" not in json.dumps(prov.messages[-1]))
c.goal_check = None
check("an item that needs a setup is refused", c.run(item="g-l1-needs-setup")["code"] == 400)
check("an unknown item is refused", c.run(item="nope")["code"] == 400)
check("an empty instruction is refused", c.run("  ")["code"] == 400)
prov.plans["fly"] = {"say": "?", "plan": [{"action": "fly", "object": "Apple", "to": None, "arg": None}]}
n_before = len(st["prims"])
c.run("fly to the apple")
ph = wait_final(c)
snap = c.snapshot()
check("an invalid plan is rejected after one repair, not executed", ph == "rejected" and snap["planner"]["rejected"]
      and snap["planner"]["errors"] and all(s["status"] == "skipped" for s in snap["steps"]), str(snap["planner"]))
check("...and no primitive was sent after planning", all(p in ("RotateRight", "LookUp", "LookDown")
                                                        for p in st["prims"][n_before:]), str(st["prims"][n_before:]))
st["llm"] = False
check("LLM off -> refused", c.run("pick up the apple")["code"] == 409)
st["llm"] = True

# ======================================================================================================== 6
section("6. Stop")
prov.block_plan = threading.Event()
c.run("pick up the apple")
check("phase reaches planning", wait_for(lambda: c.job["phase"] == "planning"))
t0 = time.time()
c.stop()
ph = wait_final(c)
check("Stop during planning: cancelled at once, late reply discarded", ph == "cancelled" and time.time() - t0 < 2
      and c.snapshot()["steps"] == [], f"{ph} {time.time() - t0:.2f}s")
prov.block_plan.set()
time.sleep(0.05)
prov.block_plan = None
check("the late planner reply changed nothing", c.snapshot()["steps"] == [] and c.job["phase"] == "cancelled")
# primitive boundary: stop right after a successful pickup -> the pickup is recorded, the next step never runs
fresh(c, sim)
prov.plans["apple to fridge"] = {"say": "ok", "plan": [
    {"action": "pickup", "object": "Apple", "to": None, "arg": None},
    {"action": "put_in", "object": "Fridge", "to": None, "arg": None}]}
st["hook"] = lambda a, action: c.stop() if action == "PickupObject" else None
c.run("apple to fridge")
ph = wait_final(c)
st["hook"] = None
snap = c.snapshot()
check("Stop at the pickup: cancelled, the pickup is recorded (holding Apple, history ok)",
      ph == "cancelled" and c.h.holding == "Apple" and any(x[1] == "PickupObject" and x[3] for x in c.h.history),
      f"{ph} {c.h.holding}")
check("...step 0 ok, step 1 not run", snap["steps"][0]["status"] == "ok"
      and snap["steps"][1]["status"] in ("skipped", "cancelled"), str([s["status"] for s in snap["steps"]]))
# mid-step: stop on the 3rd walk primitive
fresh(c, sim)
cnt = {"n": 0}


def mid(a, action):
    if action == "MoveAhead":
        cnt["n"] += 1
        if cnt["n"] == 2:
            c.stop()


st["hook"] = mid
c.run("pick up the apple")
ph = wait_final(c)
st["hook"] = None
snap = c.snapshot()
check("Stop mid-step: cancelled before the next primitive; step marked cancelled; worker alive",
      ph == "cancelled" and snap["steps"][0]["status"] == "cancelled" and sim.died == []
      and not c.h.holding, f"{ph} {[s['status'] for s in snap['steps']]} died={sim.died}")
check("...no primitive after the stop", st["prims"][-1] == "MoveAhead", str(st["prims"][-3:]))
# chooser wait: a hidden target forces a search, Gemma is asked where to look and does not answer
fresh(c, sim)
det.hide = {"Mug"}
prov.block_chooser = threading.Event()
prov.plans["mug"] = {"say": "ok", "plan": [{"action": "pickup", "object": "Mug", "to": None, "arg": None}]}
c.run("find the mug")
check("the worker waits for Gemma's choice, sim.waiting says so", wait_for(lambda: "choosing where to look" in
                                                                           c.waiting_text(), 20), c.waiting_text())
check("status_brief carries the waiting text", "choosing" in c.status_brief()["waiting"])
c.stop()
ph = wait_final(c)
snap = c.snapshot()
check("Stop during the chooser wait: cancelled (escaped LLMChooser's except Exception)", ph == "cancelled"
      and snap["steps"][0]["status"] == "cancelled" and sim.died == [], f"{ph} {snap['steps'][0]['status']}")
check("waiting text cleared", c.waiting_text() == "")
prov.block_chooser.set()
prov.block_chooser = None
det.hide = set()
# bounded LLM calls
ev_block = threading.Event()
th = threading.Thread(target=lambda: c._bounded("chooser", lambda: ev_block.wait(5), 10), daemon=True)
th.start()
wait_for(lambda: c._llm_busy.get("chooser"))
try:
    c._bounded("chooser", lambda: "x", 5)
    second = "ran"
except RuntimeError as e:
    second = str(e)
check("only one outstanding chooser call", "still running" in second, second)
ev_block.set()
th.join(2)
check("the slot frees when the abandoned call returns", wait_for(lambda: not c._llm_busy.get("chooser"))
      and c._bounded("chooser", lambda: "y", 5) == "y")

# ======================================================================================================== 7
section("7. failures and restarts")
sim.gate.clear()
ev, box = c._work(lambda ag: "applied")
time.sleep(0.03)
sim.gen += 1                                        # a restart happened before the job ran
sim.gate.set()
c._wait(ev, box, sim.gen - 1)
check("stale work after a generation change is dropped, not applied", box.get("stale") and "result" not in box,
      str(box))
sim.gate.clear()
time.sleep(0.03)
ev, box = c._work(lambda ag: "applied")
wg = c.world_gen
fresh(c, sim)
sim.gate.set()
ev.wait(2)
check("work queued before a world change is dropped", box.get("stale") == "the world changed", str(box))
# restart mid-instruction
cnt = {"n": 0}


def kill(a, action):
    if action == "PickupObject" and not cnt["n"]:
        cnt["n"] = 1
        sim.gate.clear()                            # the next step's job waits ...

        def later():
            time.sleep(0.1)
            sim.restart()                           # ... and a restart drains / invalidates it
            sim.gate.set()
        threading.Thread(target=later, daemon=True).start()


st["hook"] = kill
fresh(c, sim)
prov.plans["apple to fridge"] = prov.plans.get("apple to fridge")
c.run("apple to fridge")
ph = wait_final(c)
st["hook"] = None
snap = c.snapshot()
check("a simulator restart mid-instruction ends it 'failed: simulator restarted'", ph == "failed"
      and "restarted" in (snap["job"]["error"] or ""), f"{ph} {snap['job']['error']}")
wg = c.world_gen
c.run("what do you see")
ph = wait_final(c)
check("the next run rebuilds on the fresh simulator: new world, hand empty",
      ph == "done" and c.world_gen == wg + 1 and c._ag is sim.ag and "holding: nothing" in prov.messages[-1][1]["content"],
      f"{ph} {c.world_gen} {wg}")
# non-simulator exception stays inside the job
fresh(c, sim)
det.fail = ValueError("detector exploded")
c.run("pick up the apple")
ph = wait_final(c)
det.fail = None
snap = c.snapshot()
check("a non-simulator exception fails the run but never escapes the job", ph == "failed" and sim.died == []
      and "detector exploded" in (snap["job"]["error"] or "") + json.dumps(snap["steps"]), f"{ph} {snap['job']}")
# a simulator-call exception escapes (the supervisor restarts as before)
fresh(c, sim)
for _a in ("MoveAhead", "RotateRight", "RotateLeft", "LookDown", "LookUp", "PickupObject"):
    sim.ag.raise_on[_a] = TimeoutError("unity did not answer")
n_died = len(sim.died)
c.run("pick up the apple")
ph = wait_final(c)
check("a simulator-call failure escapes to the supervisor and the run fails", ph == "failed"
      and len(sim.died) == n_died + 1 and isinstance(sim.died[-1], TimeoutError), f"{ph} {sim.died}")
c.run("what do you see")
wait_final(c)
# display hook failure never erases an action
fresh(c, sim)
st["hook"] = lambda a, action: (_ for _ in ()).throw(RuntimeError("jpeg encoder broke"))
c.run("pick up the apple")
ph = wait_final(c)
st["hook"] = None
check("a display-hook failure never erases the pickup", ph == "done" and c.h.holding == "Apple", ph)
# render trip
fresh(c, sim)
st["hook"] = lambda a, action: c.render_trip("red_cast") if action == "MoveAhead" else None
c.run("pick up the apple")
ph = wait_final(c)
st["hook"] = None
check("a render trip aborts before the next primitive: failed with the reason", ph == "failed"
      and "render glitch" in (c.snapshot()["job"]["error"] or ""), c.snapshot()["job"]["error"])
check("render trip state is cleared for the next job", c.run("what do you see")["ok"] and wait_final(c) == "done")
# mode switch refusals while acting
prov.block_plan = threading.Event()
c.run("pick up the apple")
wait_for(lambda: c.job["phase"] == "planning")
check("switching off while the agent acts is refused", c.switch(False)["code"] == 409 and c.on)
check("a second run while acting is refused", c.run("x")["code"] == 409)
check("Forget while acting is refused", c.forget()["code"] == 409)
check("Look while acting is refused", c.look()["code"] == 409)
c.stop()
prov.block_plan.set()
wait_final(c)
prov.block_plan = None
time.sleep(0.05)

# ======================================================================================================== 8
section("8. display")
r = c.look()
ph = wait_final(c)
check("Look around runs as its own job", r["ok"] and ph == "done" and c.job["kind"] == "look")
jpg = c.last_look_jpeg()
check("last look is a JPEG", jpg[:2] == b"\xff\xd8")
dets = c.overlay(sim.gen, st["rev"])
check("overlay valid while nothing moved since the look", isinstance(dets, list) and len(dets) > 0, str(dets)[:80])
st["rev"] += 1
check("overlay invalid after any action", c.overlay(sim.gen, st["rev"]) is None)
check("overlay invalid for another simulator generation", c.overlay(sim.gen + 1, st["rev"] - 1) is None)
ll = c.snapshot()["last_look"]
check("last_look summary: step, n_dets, sources", ll["n_dets"] == len(ll["dets"]) and "source" in ll["dets"][0])
rows = [json.loads(x) for x in open(os.path.join(c.out_dir, hc.RUNS_FILE), encoding="utf-8")]
check("every finished run is archived, incl. cancelled / failed / looks",
      {"cancelled", "failed", "done", "rejected"} <= {x["phase"] for x in rows} and any(x["kind"] == "look" for x in rows))
check("history is bounded", len(c.snapshot()["history"]) <= hc.MAX_HISTORY)
r = c.switch(False)
check("off: ok, sensors applied back, session cleared", r["ok"] and not c.on and st["applied"][-1] is False
      and c.snapshot()["session"] is None)
check("runs refused while off", c.run("pick up the apple")["code"] == 409)
sim.stop.set()

# ======================================================================================================== 9
section("9. invariance: renamed ids + decoy records -> the same payload")


def renamed():
    objs = kitchen()
    for i, o in enumerate(objs):
        o["id"] = f"{o['type']}|{-9.99 + i:+.2f}|+7.77|{3.33 - i:+.2f}"
    return objs


def scrub(s):
    s = copy.deepcopy(s)
    for k in ("rev", "lease", "waiting"):
        s.pop(k, None)
    s["session"]["episode"] = re.sub(r"console:\w+:", "console:<s>:", s["session"]["episode"] or "")
    s["session"]["id"] = s["session"]["started"] = None
    s["session"]["worlds"] = [w["why"] for w in s["session"]["worlds"]]
    j = s["job"]
    j["t_start"] = j["elapsed_s"] = None
    for st_ in s["steps"]:
        st_["secs"] = None
    s["planner"]["secs"] = None
    s["planner"]["calls"] = [x["kind"] for x in s["planner"]["calls"]]
    for h_ in s["history"]:
        h_["id"] = None
    s["memory"]["tracks"] = sorted((t["type"], t["status"], t["state"], t["x"], t["z"]) for t in s["memory"]["tracks"])
    return s


outs = []
for objs, decoys in ((None, False), (renamed(), True)):
    c2, sim2, det2, prov2, st2 = make(objects=objs)
    if decoys:
        sim2.ag.event.metadata["objects"] = list(sim2.ag.event.metadata["objects"]) + [
            {"objectId": f"Decoy{i}|+0.00|+0.00|+0.00", "objectType": "Apple", "visible": True} for i in range(5)]
    c2.switch(True)
    c2.run("pick up the apple")
    wait_final(c2)
    outs.append(scrub(c2.snapshot()))
    sim2.stop.set()
check("same payload (ids we never show aside) for renamed ids + decoys", outs[0] == outs[1],
      json.dumps([k for k in outs[0] if outs[0][k] != outs[1].get(k)]))

# ======================================================================================================== 10
section("10. webapp wiring over HTTP")
import webapp                                                         # noqa: E402
from http.server import ThreadingHTTPServer                           # noqa: E402

js = open(os.path.join(HERE, "web", "app.js"), encoding="utf-8").read()
m = re.search(r"const UI_API\s*=\s*(\d+)", js)
check("WEBAPP_API 7 and app.js UI_API 7", webapp.WEBAPP_API == 7 and m and int(m.group(1)) == 7)
c3, sim3, det3, prov3, st3 = make()
webapp._HONEST, webapp._LEASE = c3, c3.lease
webapp._state["scene"] = "FloorPlan1_physics"
srv = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"


def get(path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            body = r.read()
            return r.status, (json.loads(body) if r.headers.get("Content-Type", "").startswith("application/json")
                              else body)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, body


code, s = get("/status")
check("/status carries honest + lease + sim.waiting + api 7", code == 200 and s["honest"]["on"] is False
      and "lease" in s and "waiting" in s["sim"] and s["api"] == 7, str({k: s.get(k) for k in ("honest", "lease")}))
code, h0 = get("/honest")
check("/honest snapshot while off", code == 200 and h0["on"] is False and h0["available"]["ok"])
code, r = get("/honest_run?text=pick%20up%20the%20apple")
check("/honest_run while off -> 409", code == 409 and "off" in r["error"])
code, r = get("/honest?on=1")
check("/honest?on=1 -> on", code == 200 and r["on"] is True, str(r)[:200])
for route in ("/batch?name=open_all", "/act_obj?id=Apple|1&action=pickup", "/held?name=drop", "/scan", "/explore",
              "/click?x=0.5&y=0.5&shift=1", "/capture?all=1", "/view?mode=instance", "/action?name=drop",
              "/cmd?text=hi&model=x", "/eval_run"):
    code, r = get(route)
    check(f"honest mode refuses {route.split('?')[0]}{' (' + route.split('?')[1][:14] + ')' if '?' in route else ''}",
          code == 409 and r.get("honest") is True, f"{code} {r}")
prov3.block_plan = threading.Event()
code, r = get("/honest_run?text=pick%20up%20the%20apple")
check("/honest_run -> job id", code == 200 and r["ok"] and r["job_id"], str(r))
wait_for(lambda: c3.job["phase"] == "planning")
wsrc = open(os.path.join(HERE, "webapp.py"), encoding="utf-8").read()
check("/status: a computing honest step reads busy (not stalled) until the step timeout; LLM wait reads waiting-llm",
      '"busy" if _honest_acting() and age < SIM_STEP_TIMEOUT_S else "stalled"' in wsrc and '"waiting-llm" if waiting' in wsrc)
code, s = get("/status")
check("/status while acting: honest phase + lease honest + busy", s["honest"]["phase"] == "planning"
      and s["lease"]["owner"] == "honest" and s["busy"] is True, str(s["honest"]))
for route in ("/action?name=move_ahead", "/drive?action=move_ahead&on=1", "/scene?name=FloorPlan1", "/reset",
              "/resolution?w=448&h=448", "/setparam?move=0.1", "/randomize?what=lighting"):
    code, r = get(route)
    check(f"while acting, {route.split('?')[0]} -> 409 'Stop first'", code == 409 and "Stop first" in r["error"],
          f"{code} {r}")
code, r = get("/drive?action=move_ahead&on=0")
check("a key-up (drive on=0) is always accepted", code == 200)
code, r = get("/honest?on=0")
check("/honest?on=0 while acting -> 409", code == 409)
code, r = get("/honest_stop")
prov3.block_plan.set()
check("/honest_stop -> stopping", code == 200 and r["ok"])
wait_final(c3)
code, r = get("/honest")
check("after Stop: cancelled, lease free", r["job"]["phase"] == "cancelled" and r["lease"] is None)
prov3.block_plan = None
code, r = get("/honest_run?item=g-l1-pickup-apple&lang=fa")
wait_final(c3)
code, r = get("/honest")
check("/honest_run with a dataset item in FA", r["job"]["lang"] == "fa" and r["job"]["level"] == "L1"
      and r["job"]["command"] == "سیب را بردار", str(r["job"]))
code, jpg = get("/honest_last_look.jpg")
check("/honest_last_look.jpg serves the JPEG", code == 200 and jpg[:2] == b"\xff\xd8")
code, r = get("/honest_forget")
check("/honest_forget idle -> ok", code == 200 and r["ok"])
code, r = get("/honest?ticket=" + c3.history[-1]["id"])
check("/honest?ticket= reopens an earlier ticket", code == 200 and r["ticket"]["job"]["id"] == c3.history[-1]["id"])
code, r = get("/honest?on=0")
check("/honest?on=0 idle -> off", code == 200 and r["on"] is False)
code, r = get("/batch?name=lights_on")
check("oracle mode again: /batch accepted", code == 200)
ok, _ = webapp._LEASE.claim("eval", "test-eval", sim_bound=False)
code, r = get("/cmd?text=hi&model=x")
check("oracle /cmd is refused while another owner holds the lease", code == 200 and r["ok"] is False
      and r.get("owner") == "eval", str(r))
webapp._LEASE.release("test-eval")
srv.shutdown()
sim3.stop.set()
while not webapp._jobs.empty():
    webapp._jobs.get_nowait()

# ======================================================================================================== 11
section("11. the browser side of honest mode (sim/web/honest.js) is display only")
# honest.js may read /honest* and the general set, and send only the six honest routes. Anything else it
# fetched would be a channel from the page to the simulator that bypasses the gate and the audit.
_hjs = open(os.path.join(HERE, "web", "honest.js"), encoding="utf-8").read()
_html = open(os.path.join(HERE, "web", "index.html"), encoding="utf-8").read()
check("ticket shows the honest inside-visible verdict per inspected container (2026-10-07)",
      "s.inspections" in _hjs and "inside not visible" in _hjs and "kept as a candidate" in _hjs)
check("index.html loads honest.js after app.js",
      '<script src="/static/app.js"></script>' in _html and _html.index("/static/app.js") < _html.index("/static/honest.js"))
_routes = set(re.findall(r"""(?:fetch|call)\(\s*["'`](/[A-Za-z_./]*)""", _hjs)) | set(re.findall(r"""["'`](/honest[a-z_.]*)""", _hjs))
_allowed = {"/honest", "/honest_run", "/honest_stop", "/honest_look", "/honest_forget", "/honest_last_look.jpg", "/general"}
check("honest.js touches only the honest routes and the general set", bool(_routes) and _routes <= _allowed, str(sorted(_routes)))
check("app.js routes Run/Enter through honest mode when it is on", "if (honestOn() && window.HONEST) { HONEST.run(); return; }" in js)
check("app.js refuses shift-click pickup and drop in honest mode inside the handlers",
      "e.shiftKey && honestOn()" in js and 'k === "g" && honestOn()' in js)
check("app.js refuses manual drive while the agent acts", "(DRIVE_KEYS[k] || TAP_KEYS[k]) && honestActing()" in js)
check("the oracle ticket never guesses a level from the plan's shape",
      "steps.length === 1" not in js and "plan.goal) lvl" not in js)

print("\n12. 2026-10-07 search / tidy / smooth wiring")
_csrc = open(os.path.join(HERE, "honest_console.py"), encoding="utf-8").read()
_wsrc = open(os.path.join(HERE, "webapp.py"), encoding="utf-8").read()
check("console policy: exhaustive search + tidiness; the s5 runner keeps the bounded default",
      hc.SEARCH_MODE == "exhaustive" and hc.TIDY is True and "search=SEARCH_MODE, tidy=TIDY" in _csrc
      # 2026-10-09: the runner gained --policy console (an explicit arm); its DEFAULT stays bounded
      and 'choices=("bounded", "console"), default="bounded"' in open(os.path.join(HERE, "step4_refplans.py"),
                                                                       encoding="utf-8").read())
check("console budgets fit an exhaustive search (episode cap, worker timeout)",
      hc.EP_PRIM_CAP >= hs.STEP_PRIM_CAP + hs.SEARCH_PRIM_CAP and hc.WORK_TIMEOUT >= 1800)
check("console passes the remaining plan to each step and tidies at the end, never after a stop",
      "self._exec(st, rest)" in _csrc and "self._tidy()" in _csrc and "not self._cancel.is_set()" in _csrc)
check("webapp: HonestAgent gets the live smooth profile; fast = plain sends; the Display menu sets it",
      "smooth=_honest_smooth" in _wsrc and '"fast": None' in _wsrc and '_HONEST_MOTION["speed"] = sp' in _wsrc)
check("webapp: an honest primitive refreshes the idle clock (no idle throttling during a run)",
      '_IDLE["last"] = now' in _wsrc[_wsrc.index("def _honest_prim_hook"):_wsrc.index("def _honest_overlay_frame")])

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("CONSOLE HONEST SELFTEST OK")
