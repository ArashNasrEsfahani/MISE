"""Step 5 / B1-wire - offline selftest of the adapter wiring (Windows `python`, repo root; no simulator, no WSL, no Ollama).

    python sim/_step5_adapter_selftest.py

Sections
  1  flag-off identity: HonestSkills without the new kwargs behaves exactly like the pre-B1 copy (same simulator calls, same results)
  2  hooks observe, never steer: with on_event the calls are identical; a failing consumer is recorded, not raised into the agent
  3  the event stream over a fake honest agent (REAL objmem): contract-valid, own-measurement positions, structural places only
  4  forbidden sources (Astra A7): sim ids, metadata positions, oracle state, error text, `group` cannot become events; static guards
  5  working memory ordering: the privileged verdict (practice only, after terminal)
  6  where_is channel filter runs BEFORE competition and top-k; LtmChooser == prior chooser on empty memory
  7  memory_block: bilingual identity (also after truncation), one aggregate budget, empty -> byte-identical no-memory prompt (one renderer)
  8  epoch overlay on frozen tests (independent tests, nothing written)
  9  step5_eval: allowlist (gate 12.10), canonical world, P-location, flags default OFF (config hash unchanged), practice reset by the runner
 10  the live chain on the fake kitchen: observation -> claim -> where_is -> chooser -> hint -> first visit -> sighting, with a route diff
 11  step5_live core: fail-fast invariants, admission only in practice after terminal
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                           # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ltm as L                                             # noqa: E402
import ltm_adapter as A                                     # noqa: E402
import ltm_events as lev                                    # noqa: E402
import objmem                                               # noqa: E402
import step5_eval as se                                     # noqa: E402
import step5_live as live                                   # noqa: E402
import honest                                               # noqa: E402
import honest_skills as hs                                  # noqa: E402
from honest_map import GridMap                              # noqa: E402

# W2 (A12 written into sim/step5/params.json): these checks are about the C2-e path and the event plumbing, so they read the registry WITHOUT the A12 block.
# The A12 block itself is tested in _step5_retention_a12_selftest.py.
_load_params_real = L.load_params


def _load_params_c2e(path=None):
    import copy as _copy
    P = _copy.deepcopy(_load_params_real(path))
    P["retrieval"].pop("retention", None)
    P["retrieval"].pop("a12", None)
    return P


L.load_params = _load_params_c2e

# The adapter takes step-4 pieces by INJECTION (it imports none): the tests inject the real objmem module.
_RawAdapter, _RawFake = A.HonestEventAdapter, A.FakeHonestAgent


class _AdapterG(_RawAdapter):
    def __init__(self, *a, **k):
        k.setdefault("geometry", objmem)
        super().__init__(*a, **k)


class _FakeG(_RawFake):
    def __init__(self, adapter, *a, **k):
        super().__init__(adapter, objmem, *a, **k)


A.HonestEventAdapter, A.FakeHonestAgent = _AdapterG, _FakeG

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ------------------------------------------------------------------------------------------------ the act fake kitchen
def fake_ns(skills_module=None):
    """exec the fake-kitchen head of the step-4 act selftest (FakeThor / FakeDetector / kitchen / setup), optionally with
    another honest_skills module bound to the name `hs` (the pre-B1 copy)."""
    src = open(os.path.join(HERE, "_step4_act_selftest.py"), encoding="utf-8").read().split("\n")
    marker = next(i for i, ln in enumerate(src) if re.match(r"^# =+ 1$", ln))
    ns = {"__name__": "actfake", "__file__": os.path.join(HERE, "_step4_act_selftest.py")}
    exec(compile("\n".join(src[:marker]), "actfake_head", "exec"), ns)
    if skills_module is not None:
        ns["hs"] = skills_module
    return ns


NS = fake_ns()
WB_CAB = "Cabinet|-0.55|+0.60|+2.60"


def world(target="cab", extra=True):
    objs = NS["kitchen"]()
    if extra:
        objs.append(dict(id=WB_CAB, type="Cabinet", pos=(-0.55, 0.6, 2.6), size=0.4, openable=True, receptacle=True, isOpen=False))
    if target == "cab":
        objs.append(dict(id="WineBottle|x", type="WineBottle", pos=(-0.45, 0.6, 2.6), size=0.12, pickupable=True, inside=WB_CAB))
    elif target == "fridge":
        objs.append(dict(id="WineBottle|x", type="WineBottle", pos=(2.85, 0.9, 1.0), size=0.12, pickupable=True,
                         inside="Fridge|+2.90|+0.90|+1.00"))
    return objs


def build(skills_module=None, objs=None, chooser=None, adapter=None, hints=None, on_event="auto", **skw):
    """-> (thor, honest agent, skills). `on_event='auto'` wires the adapter when one is given."""
    ns = NS if skills_module is None else fake_ns(skills_module)
    mod = skills_module or hs
    t = ns["FakeThor"](objs or world(), (4, 4), 0.0)
    det = ns["FakeDetector"](t)
    h = honest.HonestAgent(t, det, mode="honest")
    gm = GridMap("Fake", ns["CELL"], (0.0, 0.0), ns["FREE"])
    om = objmem.ObjectMemory("Fake", store=None, episode="t")
    mem = A.ObjectMemoryProxy(om, adapter) if adapter is not None else om
    kw = dict(skw)
    if on_event == "auto" and adapter is not None:
        kw["on_event"] = adapter.on_event
    elif on_event not in ("auto", None):
        kw["on_event"] = on_event
    if hints is not None:
        kw["place_hints"] = hints
    sk = mod.HonestSkills(h, gm, mem, chooser=chooser or (lambda *a: []), **kw)
    if adapter is not None:
        adapter.clock = lambda: h.step
    return t, h, sk


EXH = dict(search="exhaustive", tidy=True)


def strip(r):
    """A result without anything that is allowed to differ (nothing here is): the full dict."""
    return json.loads(json.dumps(r, default=repr))


# ================================================================================================ 1
print("1. flag-off identity vs the pre-B1 honest_skills")
pre_path = os.path.join(HERE, "step5", "_checkpoints", "B1", "honest_skills.py.pre-B1")
def load_copy(name, path):
    """A checkpoint copy (extension .pre-B1) as a module."""
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old_flag, sys.dont_write_bytecode = sys.dont_write_bytecode, True      # no __pycache__ next to the checkpoints
    try:
        loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old_flag
    return mod


hs_pre = load_copy("hs_pre_b1", pre_path)
check("the pre-B1 checkpoint exists and lacks the new kwargs",
      "on_event" not in inspect.signature(hs_pre.HonestSkills.__init__).parameters
      and "on_event" in inspect.signature(hs.HonestSkills.__init__).parameters
      and "place_hints" in inspect.signature(hs.HonestSkills.__init__).parameters)
d = inspect.signature(hs.HonestSkills.__init__).parameters
check("both new kwargs default to None", d["on_event"].default is None and d["place_hints"].default is None)


def scenario(mod, kind):
    if kind == "bounded":
        objs = world(None, extra=False)
        a = next(o for o in objs if o["type"] == "Apple")
        a["pos"], a["inside"] = (2.75, 0.95, 1.0), "Fridge|+2.90|+0.90|+1.00"
        t, h, sk = build(mod, objs=objs, chooser=lambda tg, ty, su: ["Fridge", "Garage"])
        sk.start_episode("t")
        r = sk.exec_step({"action": "pickup", "object": "Apple"})
    elif kind == "exhaustive":
        t, h, sk = build(mod, objs=world("cab"), **EXH)
        sk.start_episode("t")
        r = sk.exec_step({"action": "search_for", "object": "WineBottle"})
    elif kind == "composite":
        t, h, sk = build(mod, objs=world(None, extra=False))
        sk.start_episode("t")
        r = sk.exec_step({"action": "pick_and_place", "object": "Apple", "to": "CounterTop"})
    else:                                                   # a failure path (retry exhausted)
        t, h, sk = build(mod, objs=world(None, extra=False))
        t.refuse[("PickupObject", "Apple")] = 5
        sk.start_episode("t")
        r = sk.exec_step({"action": "pickup", "object": "Apple"})
    return t, h, sk, r


for kind in ("bounded", "exhaustive", "composite", "failure"):
    t0, h0, sk0, r0 = scenario(hs_pre, kind)
    t1, h1, sk1, r1 = scenario(hs, kind)
    check(f"{kind}: identical simulator calls ({len(t1.calls)})", t0.calls == t1.calls)
    check(f"{kind}: identical step result (ok / cause / retries / detail)", strip(r0) == strip(r1))
    check(f"{kind}: flag-off detail has no `under` key and no hook state", "under" not in r1["detail"]
          and sk1.on_event is None and sk1.place_hints is None and sk1.event_errors == [])

# ================================================================================================ 2
print("\n2. hooks observe, never steer")
tmp = tempfile.mkdtemp()
m0 = L.LongTermMemory(os.path.join(tmp, "h.db"), kitchen="Fake")
m0.reset_world("r0")
ad = A.HonestEventAdapter(m0, episode="h1", mode="practice", detector_rev="fake")
t1, h1, sk1 = build(objs=world("cab"), adapter=ad, **EXH)
ad.on_task("Find the wine bottle", "L1", "en")
sk1.start_episode("t")
r1 = sk1.exec_step({"action": "search_for", "object": "WineBottle"})
t0, h0, sk0, r0 = scenario(hs, "exhaustive")
check("on_event wired: the simulator calls are identical to the hook-free run", t0.calls == t1.calls)
check("on_event wired: same ok / cause / retries / verify", (r0["ok"], r0["cause"], r0["retries"], r0["detail"]["verify"])
      == (r1["ok"], r1["cause"], r1["retries"], r1["detail"]["verify"]))
kinds = [e["kind"] for e in ad.stream]
check("open / close / inspection events reached the working memory", {"open", "close", "inspection"} <= set(kinds))
check("search start + end reached the adapter's metrics (found, steps_to_find)", len(ad.searches) == 1 and ad.searches[0]["found"]
      and ad.searches[0]["steps_to_find"] > 0 and ad.searches[0]["otype"] == "WineBottle")
check("`under` is exposed only when the flag is on", r1["detail"]["under"] is None and "under" not in r0["detail"])
check("no event was rejected, no hook error", ad.rejected == [] and sk1.event_errors == [])


def boom(kind, fields):
    raise RuntimeError("consumer failure Fridge|+2.90|+0.90|+1.00")


t2, h2, sk2 = build(objs=world("cab"), on_event=boom, **EXH)
sk2.start_episode("t")
r2 = sk2.exec_step({"action": "search_for", "object": "WineBottle"})
check("a failing consumer never reaches the agent (calls identical)", t2.calls == t0.calls and r2["ok"] == r0["ok"])
check("... it is recorded for the executor to fail fast on", len(sk2.event_errors) >= 3 and all(
    set(e) == {"kind", "error"} for e in sk2.event_errors))
# structured `under` of a failed step (never parsed from text)
t3, h3, sk3 = build(objs=world(None, extra=False), on_event=lambda k, f: None)
t3.refuse[("PickupObject", "Apple")] = 5
sk3.start_episode("t")
r3 = sk3.exec_step({"action": "pickup", "object": "Apple"})
check("retry_exhausted carries the wrapped cause as a structured `under`", r3["cause"] == "retry_exhausted"
      and r3["detail"]["under"] in hs.CAUSES and r3["detail"]["under"] != "retry_exhausted", str(r3["detail"].get("under")))

# ================================================================================================ 3
print("\n3. the event stream over a fake honest agent (real objmem)")
m1 = L.LongTermMemory(os.path.join(tmp, "s.db"), kitchen="Fake")
m1.reset_world("r1")
ad = A.HonestEventAdapter(m1, episode="s1", mode="practice", detector_rev="fake-rev")
ag = A.FakeHonestAgent(ad)
ad.on_task("Put the apple in the fridge", "L2", "en")
rep = ag.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0, "state": "closed, off"},
               {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9},
               {"type": "CounterTop", "box": [0, 80, 95, 95], "depth": 2.5},
               {"type": "Mug", "box": [70, 82, 78, 90], "depth": 2.4}])
stream = ad.stream
check("episode_start + task_context first; concepts parsed from the text", stream[0]["kind"] == "episode_start"
      and stream[1]["kind"] == "task_context" and stream[1]["concepts"]["roles"][0]["obj"] == "Apple"
      and stream[1]["concepts"]["roles"][0]["dest"] == "Fridge")
rs = [e for e in stream if e["kind"] == "receptacle_seen"]
check("structural receptacles only (Fridge, CounterTop); movable Mug / Apple are not places",
      sorted(e["rtype"] for e in rs) == ["CounterTop", "Fridge"])
sg = {e["otype"]: e for e in stream if e["kind"] == "sighting"}
check("sightings for the movable types, minted tr: ids", set(sg) == {"Apple", "Mug"} and all(re.match(r"^tr:\d+$", e["track"]) for e in sg.values()))
fr_xyz = next(e["xyz"] for e in rs if e["rtype"] == "Fridge")
check("apple sits in the Fridge box (in_basis box_only) at the fridge's own-depth position",
      sg["Apple"]["in_rtype"] == "Fridge" and sg["Apple"]["in_basis"] == "box_only" and sg["Apple"]["in_xyz"] == fr_xyz)
check("the mug on the counter sits in CounterTop", sg["Mug"]["in_rtype"] == "CounterTop")
tr = ag.om._real.tracks
check("sighting xyz == the track's own measurement (own depth + own pose)",
      all(tuple(e["xyz"]) == tuple(tr[k].xyz) for e in sg.values() for k in tr if tr[k].type == e["otype"]))
check("state word is a token ('closed, off' -> closed/off) and the detector revision is carried",
      A.tokify("closed, off") == "closed/off" and sg["Apple"]["detector"] == "fake-rev")
# consistency with objmem.place_of on this view (no movable receptacles present)
from_om = objmem.place_of({"bbox_xyxy": (30, 40, 40, 50), "type": "Apple"},
                          [{"type": "Fridge", "bbox_xyxy": (10, 10, 60, 80)}, {"type": "CounterTop", "bbox_xyxy": (0, 80, 95, 95)}])
check("container choice agrees with objmem.place_of", from_om == "Fridge")
# more views: cap on re-emitted sightings, dedupe of receptacle boxes
n0 = len([e for e in ad.stream if e["kind"] == "sighting"])
for k in range(6):
    ag.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9}])
n1 = len([e for e in ad.stream if e["kind"] == "sighting" and e["otype"] == "Apple"])
check("re-sightings of one track are capped (REFRESH_CAP)", n1 <= 1 + A.REFRESH_CAP, str(n1))
rs2 = [e for e in ad.stream if e["kind"] == "receptacle_seen" and e["rtype"] == "Fridge"]
check("a receptacle box seen in many views is emitted at most twice", len(rs2) <= 2, str(len(rs2)))
# own actions through the proxy
tid = next(k for k in tr if tr[k].type == "Apple")
ag.om.on_pickup(True, tid, "Apple", step=7)
ag.om.on_put(True, "CounterTop", (0.1, 0.9, 2.0), "Apple", step=9)
ag.om.on_drop(True, ag.pose, step=11)
ks = [e["kind"] for e in ad.stream]
check("pickup / put / drop events", {"pickup", "put"} <= set(ks) and ks.count("put") == 1)
put = next(e for e in ad.stream if e["kind"] == "put")
check("put: actuator provenance only (resighted False), receptacle token from the agent's own belief",
      put["resighted"] is False and put["rtype"] == "CounterTop" and put["otype"] == "Apple")
ad.on_step_start(0, {"action": "pickup", "object": "Apple"}, {"holding": None})
ad.on_step_end(0, {"ok": True, "cause": None, "retries": 0, "detail": {"prims": 5}})
ad.on_terminal(True, 40, ad.n_opens)
ad.on_admission(True)
ad.end()
check("the whole stream passes ltm_events.validate and re-validates to itself",
      all(lev.validate(e) == e for e in ad.stream), "")
check("`ev` is strictly increasing from 1; one episode, one kitchen, one epoch, one frame",
      [e["ev"] for e in ad.stream] == list(range(1, len(ad.stream) + 1))
      and len({(e["episode"], e["kitchen"], e["epoch"], e["frame"]) for e in ad.stream}) == 1)
res = m1.end_episode(ad.wm, admission=True)
check("a practice episode over this stream consolidates (places + a claim)", res["status"] == "committed"
      and res["stats"]["places_created"] >= 2 and res["stats"]["claims_created"] >= 1, str(res.get("stats")))

# ================================================================================================ 4
print("\n4. forbidden sources")
m2 = L.LongTermMemory(os.path.join(tmp, "f.db"), kitchen="Fake")
m2.reset_world("r2")
ad = A.HonestEventAdapter(m2, episode="f1", mode="practice")
ag = A.FakeHonestAgent(ad)
ad.on_task("Find the apple", "L1", "en")
OID = "Apple|+01.20|+00.97|+02.95"
JUNK_POS = (91.0, 92.0, 93.0)
junk_meta = {"agent": {"position": {"x": 77.0, "y": 0.9, "z": 78.0}}, "objects": [{"objectId": OID, "position": {"x": 55.0, "y": 1, "z": 56.0}}]}
ag.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0, "objectId": "Fridge|+02.90|+00.90|+01.00",
          "metadata": junk_meta, "parentReceptacles": [OID], "name": "Fridge_1", "isOpen": True, "states": {"open": "1"}},
         {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9, "objectId": OID, "xyz": JUNK_POS, "xyz_gt": JUNK_POS,
          "state": "open|+1.0|+2.0|", "isPickedUp": False, "receptacleObjectIds": [OID], "group": "g-fam-1"}],
        metadata=junk_meta, event=object(), last_event=junk_meta, objectId=OID, controller=object())
dump = json.dumps(ad.stream, default=repr)
check("no object id, metadata key or junk position in the stream", "objectId" not in dump and "|+" not in dump
      and "metadata" not in dump and "91.0" not in dump and "77.0" not in dump and "55.0" not in dump and "Fridge_1" not in dump)
apple = next(e for e in ad.stream if e["kind"] == "sighting")
check("the sighting position is the depth-derived one, not any attached xyz", tuple(apple["xyz"]) != JUNK_POS
      and abs(apple["xyz"][2] - 1.9) < 1e-6)
check("an id-like detector state word is not repaired into a token (it becomes '?')", apple["state"] == "?")
check("`group` and oracle flags never appear", "group" not in dump and "g-fam-1" not in dump and "isOpen" not in dump and "isPickedUp" not in dump)
# step results: free text and ids in `detail` are never read
ad.on_step_start(0, {"action": "pickup", "object": "Apple"}, {"holding": None})
ad.on_step_end(0, {"ok": False, "cause": "Fridge|+1|+2|+3 refused: ERROR", "retries": 2,
                   "detail": {"prims": 9, "why": "OpenObject failed on " + OID, "events": ["[3] " + OID], "under": "boom " + OID,
                              "search": {"x": OID}, "holding": OID}})
so = next(e for e in ad.stream if e["kind"] == "step_outcome")
dump = json.dumps(so)
check("step_outcome: error text, ids and `why` are dropped; cause and under are enumerated ('unknown')",
      OID not in dump and so["cause"] == "unknown" and so["under"] == {"code": "unknown"} and so["retries"] == 2 and so["prims"] == 9)
sig = inspect.signature(A.HonestEventAdapter.on_action)
check("on_action has no error-text parameter", not any(re.search(r"err|text|msg|message", p) for p in sig.parameters))
ad.on_action("open", False, otype="Fridge", rtype="Fridge", i=0, k=1, cause="ERROR: Fridge|+1|+2|+3 is closed")
rf = next(e for e in ad.stream if e["kind"] == "refusal")
check("a refusal carries a bool-derived event and an enumerated cause only", rf["cause"] == "unknown" and "ERROR" not in json.dumps(rf))
# the contract refuses privileged material even if the adapter were asked to emit it
n_rej = len(ad.rejected)
for bad in ({"objectId": OID}, {"metadata": {"a": 1}}, {"errors_logged": ["x"]}, {"group": "g"}):
    try:
        ad._emit("sighting", otype="Apple", track="tr:99", **bad)
        ok_ = False
    except A.AdapterError:
        ok_ = True
    check(f"adapter._emit refuses {list(bad)[0]}", ok_)
check("each refusal is recorded in adapter.rejected", len(ad.rejected) == n_rej + 4)
try:
    ad._emit("sighting", otype="Apple", track="tr:98", state=OID)
    ok_ = False
except A.AdapterError:
    ok_ = True
check("a simulator id in a token field is refused by the contract", ok_)
# the verdict: practice only, after terminal
m3 = L.LongTermMemory(os.path.join(tmp, "g.db"), kitchen="Fake")
m3.reset_world("r3")
adt = A.HonestEventAdapter(m3, episode="v1", mode="test")
adt.on_task("Find the apple", "L1", "en")
try:
    adt.on_admission(True)
    ok_ = False
except A.AdapterError:
    ok_ = True
check("a test episode cannot carry an admission_verdict", ok_)
adp = A.HonestEventAdapter(m3, episode="v2", mode="practice")
adp.on_task("Find the apple", "L1", "en")
try:
    adp.on_admission(True)
    ok_ = False
except A.AdapterError:
    ok_ = True
check("the verdict before `terminal` is refused", ok_)
adp.on_terminal(True, 3, 0)
adp.on_admission(False)
try:
    adp.on_step_start(1, {"action": "pickup", "object": "Apple"})
    ok_ = False
except A.AdapterError:
    ok_ = True
check("nothing but the verdict / episode_end follows `terminal`", ok_)

# --- static guards (AST over the adapter and the live executor)
FORBID_NAMES = {"controller", "metadata", "last_event", "state_readers", "reachable_positions"}


def names_of(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add(a.name.split(".")[0])
    return out, tree


for fn in ("ltm_adapter.py", "step5_live.py", "ltm.py", "ltm_wm.py"):
    nm, tree = names_of(os.path.join(HERE, fn))
    check(f"{fn}: never names the controller / metadata / last_event / state_readers", not (nm & FORBID_NAMES), str(nm & FORBID_NAMES))
nm, tree = names_of(os.path.join(HERE, "ltm_adapter.py"))
top = set()
for n in tree.body:
    if isinstance(n, ast.Import):
        top |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom):
        top.add((n.module or "").split(".")[0])
    elif isinstance(n, ast.Try):
        for b in n.body:
            if isinstance(b, ast.Import):
                top |= {a.name.split(".")[0] for a in b.names}
            elif isinstance(b, ast.ImportFrom):
                top.add((b.module or "").split(".")[0])
every_import = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
    {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
check("ltm_adapter.py: NO step-4 / simulator import anywhere (not even lazily); the pieces are injected",
      not (every_import & {"honest", "honest_skills", "honest_map", "thor_actions", "ai2thor", "ultralytics", "objmem", "webapp", "perception",
                           "step4_gemma", "step4_refplans", "eval", "memory", "trace"}), str(every_import))
try:
    _RawAdapter(None, episode="x", kitchen="Fake").on_observation(type("O", (), {"step": 1, "depth": [[1.0]], "pose": {}, "dets": []})(),
                                                                {"new": []}, {})
    ok_ = False
except A.AdapterError:
    ok_ = True
check("an adapter without the injected geometry refuses observations", ok_)
try:
    A.LtmChooser(None, None)
    ok_ = False
except A.AdapterError:
    ok_ = True
check("LtmChooser requires the injected fallback chooser", ok_)
_, ltree = names_of(os.path.join(HERE, "step5_live.py"))
bad_scoring = []
for fnode in ast.walk(ltree):
    if isinstance(fnode, (ast.FunctionDef, ast.AsyncFunctionDef)) and not fnode.name.startswith("_score"):
        for n in ast.walk(fnode):
            if isinstance(n, ast.Attribute) and n.attr == "scoring":
                bad_scoring.append(fnode.name)
check("step5_live: `.scoring()` appears only inside _score* functions", not bad_scoring, str(bad_scoring))
scoring_fns = [f.name for f in ast.walk(ltree) if isinstance(f, ast.FunctionDef) and f.name.startswith("_score")]
check("... and that scoring block exists", scoring_fns == ["_score"] or "_score" in scoring_fns)
for fn in ("ltm_adapter.py", "step5_live.py", "ltm.py"):
    nm, _ = names_of(os.path.join(HERE, fn))
    check(f"{fn}: does not use step4_gemma's renderer, memory_summary-as-memory or GemmaPlanner.chooser",
          not (nm & {"step4_gemma", "GemmaPlanner", "build_messages"}), str(nm & {"step4_gemma", "GemmaPlanner", "build_messages"}))
src_live = open(os.path.join(HERE, "step5_live.py"), encoding="utf-8").read()
check("step5_live reaches a planner only through PlannerContract.messages(memory_block=...)",
      "memory_block=blk[\"text\"] or None" in src_live and "reorder_within" not in src_live)

# ================================================================================================ 6
print("\n6. where_is channel filter + LtmChooser")
# a store with: a location claim from epoch 1 (Apple seen in the Fridge) and a placement from epoch 2 (Apple put on the counter)
P = L.load_params()
P1 = json.loads(json.dumps(P))
P1["retrieval"]["where_k"]["value"] = 1
mk = L.LongTermMemory(os.path.join(tmp, "c.db"), kitchen="Fake", params=P1)
mk.reset_world("c1")
ad = A.HonestEventAdapter(mk, episode="c-ep1", mode="practice")
ag = A.FakeHonestAgent(ad, episode="c1")
ad.on_task("Find the apple", "L1", "en")
ag.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9},
         {"type": "CounterTop", "box": [0, 80, 95, 95], "depth": 2.5}])
ad.on_terminal(True, 20, 0)
ad.on_admission(True)
ad.end()
mk.end_episode(ad.wm, admission=True)
mk.reset_world("c2")
ad2 = A.HonestEventAdapter(mk, episode="c-ep2", mode="practice")
ag2 = A.FakeHonestAgent(ad2, episode="c2")
ad2.on_task("Put the apple on the counter", "L1", "en")
ag2.look([{"type": "CounterTop", "box": [0, 80, 95, 95], "depth": 2.5}])
ag2.om.on_pickup(True, None, "Apple", step=5)
ag2.om.on_put(True, "CounterTop", (0.1, 0.9, 2.0), "Apple", step=8)
ad2.on_terminal(True, 20, 0)
ad2.on_admission(True)
ad2.end()
r2 = mk.end_episode(ad2.wm, admission=True)
ad_prior = {"Cabinet": 0.5, "Fridge": 0.3, "CounterTop": 0.2}
full = mk.where_is("Apple", None, prior=ad_prior)
check("setup: a placement of epoch 2 (T1) and a claim of epoch 1 (T2)", r2["stats"]["placements"] == 1 and full["epoch"] == 2
      and full["candidates"][0]["tier"] == "T1", str(full["candidates"]))
check("full channels, where_k=1: the placement takes the only slot and the learned claim is cut AFTER competition",
      len(full["candidates"]) == 1 and full["explore"]["n_beyond_cut"] >= 1)
pl = mk.where_is("Apple", None, channels={"places"}, prior=ad_prior)
check("channels={'places'}, same where_k=1: the T2 location claim survives (the filter ran BEFORE competition and the cut)",
      len(pl["candidates"]) == 1 and pl["candidates"][0]["tier"] == "T2" and pl["candidates"][0]["rtype"] == "Fridge", str(pl["candidates"]))
P2 = json.loads(json.dumps(P))
mk2 = L.LongTermMemory(os.path.join(tmp, "c.db"), kitchen="Fake", params=P2)
pl_all = mk2.where_is("Apple", None, channels={"places"}, prior=ad_prior)
check("channels={'places'}: only T0 / T2 location claims, never placements / habits / the prior (even with room)",
      all(c["tier"] in ("T0", "T2") and c["source"] == "location_claim" for c in pl_all["candidates"]) and len(pl_all["candidates"]) == 1)
check("channels={'places','placements'} adds the T1 candidate", {c["tier"] for c in mk2.where_is("Apple", None, channels={"places", "placements"}, prior=ad_prior)["candidates"]}
      == {"T1", "T2"})
check("channels=() returns no candidates, only EXPLORE", mk.where_is("Apple", None, channels=set())["candidates"] == []
      and mk.where_is("Apple", None, channels=set())["explore"]["id"] == "EXPLORE")
try:
    mk.where_is("Apple", None, channels={"bogus"})
    ok_ = False
except ValueError:
    ok_ = True
check("unknown channel names are refused", ok_)
check("channels=None is the A-stage behaviour (the declared prior is present)", any(c["tier"] == "T4" for c in mk2.where_is("Apple", None, prior=ad_prior)["candidates"]))
mk2.close()
# LtmChooser: empty memory == the prior chooser, exactly
types = ["Apple", "CounterTop", "Fridge", "Microwave", "SinkBasin", "Cabinet", "Bowl", "Drawer"]
empty = L.LongTermMemory(os.path.join(tmp, "e.db"), kitchen="Fake")
empty.reset_world("e1")
wm_e = empty.begin_episode("e-test", "practice")
ch = A.LtmChooser(empty, wm_e, fallback=hs.prior_chooser)
for tgt in ("Apple", "Mug", "Fork", "Egg"):
    check(f"empty memory: LtmChooser == prior_chooser for {tgt}", ch(tgt, types, ["Fridge [closed] on/in ? at (1.00, 2.00), step 3"])
          == hs.prior_chooser(tgt, types, ["Fridge [closed] on/in ? at (1.00, 2.00), step 3"]))
check("... and publishes no hints", ch.hints_for("Apple") == [] and not ch.decisions[-1]["changed"])
wm_c = mk.begin_episode("c-test", "test")
ch = A.LtmChooser(mk, wm_c, fallback=lambda *a: ["CounterTop", "Cabinet", "Fridge"])
o = ch("Apple", types, [])
check("learned type first, then the fallback's order, de-duplicated, target excluded",
      o[0] == "Fridge" and o[1:] == ["CounterTop", "Cabinet"] and "Apple" not in o, str(o))
hh = ch.hints_for("Apple")
check("a hint = place xyz + radius (a navigation goal, not a sighting)", len(hh) == 1 and len(hh[0]["xyz"]) == 3 and hh[0]["radius"] > 0
      and hh[0]["rtype"] == "Fridge")
ch_f = A.LtmChooser(mk, wm_c, fallback=lambda t, tl, ms: [x for x in ["CounterTop", "Cabinet", "Fridge"] if x in tl])
check("a learned type outside the detector's list is dropped", ch_f("Apple", ["Apple", "CounterTop", "Cabinet"], []) == ["CounterTop", "Cabinet"])
check("channels=() chooser == fallback", A.LtmChooser(mk, wm_c, channels=(), fallback=lambda *a: ["Cabinet", "Fridge"])("Apple", types, []) == ["Cabinet", "Fridge"])
mk.close()
empty.close()

# ================================================================================================ 7
print("\n7. memory_block: bilingual identity, aggregate budget, one renderer")
import planners.contract as pc                                # noqa: E402

mb = L.LongTermMemory(os.path.join(tmp, "b.db"), kitchen="Fake")
mb.reset_world("b1")
ad = A.HonestEventAdapter(mb, episode="b-ep1", mode="practice")
ag = A.FakeHonestAgent(ad, episode="b1")
ad.on_task("Put the apple in the fridge", "L2", "en")
ag.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9},
         {"type": "CounterTop", "box": [0, 80, 95, 95], "depth": 2.5}, {"type": "Mug", "box": [70, 82, 78, 90], "depth": 2.4}])
ad.on_terminal(True, 20, 0)
ad.on_admission(True)
ad.end()
mb.end_episode(ad.wm, admission=True)
mb.reset_world("b2")


def test_wm(text, lang, level="L2", ep="bt"):
    w = mb.begin_episode(ep, "test")
    adt_ = A.HonestEventAdapter(mb, wm=w, episode=ep, mode="test")
    adt_.on_task(text, level, lang)
    return w


w_en = test_wm("Put the apple in the fridge and put the mug on the counter", "en", ep="bt1")
w_fa = test_wm("سیب را در یخچال بگذار و ماگ را روی کانتر بگذار", "fa", ep="bt2")
b_en = mb.memory_block(w_en, "L2", "en")
b_fa = mb.memory_block(w_fa, "L2", "fa")
check("types come from the parsed task, in order of mention (apple, mug)", mb.block_types(w_en) == ["Apple", "Mug"], str(mb.block_types(w_en)))
check("both learned types get a section under ONE header", b_en["types"] == ["Apple", "Mug"] and b_en["text"].count("Memory (may be outdated") == 1, b_en["text"])
check("EN and FA render the same record ids in the same order", [r["id"] for r in b_en["records"]] == [r["id"] for r in b_fa["records"]]
      and len(b_en["records"]) >= 2)
check("the block is within the level budget and counts the final text", b_en["tokens"] <= b_en["budget"] and b_fa["tokens"] <= b_fa["budget"])
full_ids = [r["id"] for r in b_en["records"]]
sweep = {}
for tiny in range(30, 330, 5):
    e_, f_ = mb.memory_block(w_en, "L2", "en", budget_tokens=tiny), mb.memory_block(w_fa, "L2", "fa", budget_tokens=tiny)
    sweep[tiny] = ([r["id"] for r in e_["records"]], [r["id"] for r in f_["records"]], e_["tokens"], f_["tokens"], e_["text"], f_["text"])
check("every budget 30..325: EN and FA retain the SAME record ids in the same order (joint retention, also after truncation)",
      all(v[0] == v[1] for v in sweep.values()), str([k for k, v in sweep.items() if v[0] != v[1]][:5]))
check("every budget: both languages stay within the budget (final serialised text counted); whole records only",
      all((v[2] <= k and v[3] <= k) for k, v in sweep.items()), str([k for k, v in sweep.items() if v[2] > k or v[3] > k][:5]))
check("the sweep covers all three regimes: empty, partial (first type only), full", any(not v[0] for v in sweep.values())
      and any(0 < len(v[0]) < len(full_ids) for v in sweep.values()) and any(v[0] == full_ids for v in sweep.values()))
check("a smaller budget never reorders: retained ids are always a prefix of the full block's",
      all(v[0] == full_ids[:len(v[0])] for v in sweep.values()))
check("retention is monotone in the budget", all(len(sweep[k][0]) <= len(sweep[k2][0]) for k, k2 in zip(sorted(sweep), sorted(sweep)[1:])))
tr_ = next(k for k, v in sweep.items() if 0 < len(v[0]) < len(full_ids))
check("a truncated block reports what it dropped (budget reasons), the rest is untouched",
      {d["reason"] for d in mb.memory_block(w_en, "L2", "en", budget_tokens=tr_)["dropped"]} <= {"no_record_fits", "after_budget_stop", "budget"}
      and mb.memory_block(w_en, "L2", "en", budget_tokens=tr_)["dropped"])
check("every retained record keeps its qualifier bracket in its line (both languages)",
      all(re.search(r"\[.+\]", r["text"]) for r in b_en["records"] + b_fa["records"]))
check("channels=() -> empty block", mb.memory_block(w_en, "L2", "en", channels=())["empty"])
w_none = test_wm("Open the fridge", "en", ep="bt3")
check("nothing learned applies -> empty text", mb.memory_block(w_none, "L1", "en")["text"] == "")
contract = pc.PlannerContract(sorted(set(types) | {"Mug", "WineBottle"}), tools=sorted(pc.TOOL_SPECS))
msgs0 = contract.messages("Open the fridge", [], None)
msgs1, blk = live.render_planner_messages(contract, "Open the fridge", mb, w_none, level="L1", lang="en", channels=("places",))
check("empty block -> the prompt is byte-identical to the no-memory prompt (the arms differ only when memory applies)", msgs0 == msgs1 and blk["empty"])
msgs2, blk2 = live.render_planner_messages(contract, "Put the apple in the fridge", mb, w_en, level="L2", lang="en", channels=("places",))
check("a non-empty block appears once, under the contract's fixed header, after the command",
      msgs2[1]["content"].count(pc.MEMORY_HEADER) == 1 and msgs2[1]["content"].endswith(blk2["text"])
      and msgs2[1]["content"].index("Command:") < msgs2[1]["content"].index(pc.MEMORY_HEADER))
bc = live.bilingual_check(mb, w_en, "L2", ("places",))
check("bilingual_check reports identical identities", bc["same_ids"] and bc["same_dropped"] and not bc["empty"])
mb.close()

# ================================================================================================ 8
print("\n8. epoch overlay on frozen tests")
mo = L.LongTermMemory(os.path.join(tmp, "o.db"), kitchen="Fake")
mo.reset_world("o1")
ad = A.HonestEventAdapter(mo, episode="o-ep1", mode="practice")
ag = A.FakeHonestAgent(ad, episode="o1")
ad.on_task("Find the apple", "L1", "en")
ag.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9}])
ad.on_terminal(True, 5, 0)
ad.on_admission(True)
ad.end()
check("practice consolidates at epoch 1", mo.end_episode(ad.wm, admission=True)["status"] == "committed" and mo.current_epoch() == 1)
try:
    mo.begin_test_epoch()
    ok_ = False
except Exception:
    ok_ = True
check("begin_test_epoch is refused on a live (non-frozen) store", ok_)
snap = os.path.join(tmp, "osnap")
mo.snapshot(snap)
eps = []
for k in range(3):
    fz = L.LongTermMemory.open_frozen(snap, os.path.join(tmp, f"otl{k}.db"), kitchen="Fake")
    h0 = fz.state_hash()
    ov = fz.begin_test_epoch(frame="f0", map_rev="m0")
    eps.append((ov["epoch"], fz.current_epoch(), fz.frame_of_epoch(fz.current_epoch())))
    ans = fz.where_is("Apple", None, channels={"places"})
    tier = ans["candidates"][0]["tier"] if ans["candidates"] else None
    check(f"test {k}: epoch = snapshot epoch + 1, claim reads as 'maybe' (T2), nothing written", eps[-1][0] == 2 and tier == "T2" and fz.state_hash() == h0)
    fz.end_test_epoch()
    check(f"test {k}: the overlay is RAM-only (a fresh copy of the snapshot starts at the snapshot epoch)", fz.current_epoch() == 1)
    fz.close()
check("independent tests never accumulate epochs", {e[0] for e in eps} == {2})
mo.close()

# ================================================================================================ 9
print("\n9. step5_eval flags: allowlist, canonical world, P-location, defaults OFF")
se_pre = load_copy("se_pre_b1", os.path.join(HERE, "step5", "_checkpoints", "B1", "step5_eval.py.pre-B1"))
check("EvalConfig() defaults hash exactly as before B1 (config hash + to_dict unchanged)",
      se.EvalConfig().sha() == se_pre.EvalConfig().sha() and se.EvalConfig().to_dict() == se_pre.EvalConfig().to_dict())
core = se.core_config({"eval": {"blocks": {"value": 5}, "K": {"value": [0, 1]}, "protocols": {"value": ["P-familiar"]}}})
core0 = se_pre.core_config({"eval": {"blocks": {"value": 5}, "K": {"value": [0, 1]}, "protocols": {"value": ["P-familiar"]}}})
check("core_config is unchanged", core.sha() == core0.sha())
FT = os.path.join(ROOT, "docs", "final-test", "split_manifest.json")
allowed, forbidden, ash = se.final_test_allowlist(FT)
check("allowlist = nontest minus the 12 dev-pilot items (104); forbidden = 125 test + 26 capped-out", len(allowed) == 104 and len(forbidden) == 151
      and not (allowed & forbidden) and len(ash) == 64)
fm = json.load(open(FT, encoding="utf-8"))
check("no dev-pilot item is allowed", not (allowed & set(fm["dev_pilot"])))
try:
    se.assert_items_allowed([sorted(forbidden)[0]], allowed, forbidden, "t")
    ok_ = False
except se.AllowlistError:
    ok_ = True
check("a test-family item is refused", ok_)
try:
    se.assert_items_allowed([sorted(fm["dev_pilot"])[0]], allowed, forbidden, "t")
    ok_ = False
except se.AllowlistError:
    ok_ = True
check("a dev-pilot item is refused (outside the allowlist)", ok_)
items, dsha = se.load_items(os.path.join(HERE, "dataset", "general.json"))
b_items = [it for it in items if it["id"] in allowed]
# A26 (K-B2b): g-l1-pickup-knife and g-l2-find-pickup-knife (allowlisted) are archived and left the dataset's items (104 -> 102)
check("the dataset filtered to the allowlist has 102 items (104 minus the 2 A26-archived allowlisted ids)", len(b_items) == 102, len(b_items))
man = se.build_split(b_items, dsha, 20261010)
check("the step-5 split over them uses only allowed items", {i for g in man["groups"] for i in g["items"]} <= allowed
      and set(man["test_familiar"]) <= allowed and set(man["test_composition"]) <= allowed)
cfgL = se.location_config({})
check("location_config: none vs places, K 0/1, canonical world, P-location, overlay + seeded language + practice reset",
      cfgL.arms == ("none", "places") and cfgL.Ks == (0, 1) and cfgL.world_kinds == ("canonical",) and cfgL.protocols == ("P-location",)
      and cfgL.epoch_overlay and cfgL.test_langs == "seeded" and cfgL.practice_reset and cfgL.blocks == 5 and cfgL.tests_per_block == 8)
plans = [se.make_block_plan(man, b_items, cfgL, b, "canonical") for b in range(5)]
tl = [p["tests"]["P-location"] for p in plans]
check("5 blocks x 8 test items, one seeded language each (not both)", all(len(t) == 8 for t in tl) and all(len({x["item"] for x in t}) == 8 for t in tl))
prac_items = {p["item"] for p in man["practice"]}
check("test items are disjoint from the practice items; all allowed", all(x["item"] not in prac_items and x["item"] in allowed for t in tl for x in t))
check("canonical world: every seed is 0; the plan is a pure function (identical rebuild)", all(x["world_seed"] == 0 for t in tl for x in t)
      and se.canon(plans[0]) == se.canon(se.make_block_plan(man, b_items, cfgL, 0, "canonical")))
draw = lambda p: [(x["item"], x["lang"], x["level"], x["world_seed"], x["policy_seed"]) for x in p["tests"]["P-location"]]    # noqa: E731
check("test draws are identical for every arm and K (they depend on neither)",
      draw(plans[0]) == draw(se.make_block_plan(man, b_items, se.location_config({}, arms=("none",), Ks=(0, 1, 3)), 0, "canonical")))
check("levels are stratified in a block", len({x["level"] for x in tl[0]}) >= 2)
check("both languages occur across the blocks (seeded per block and item, not per arm)", {x["lang"] for t in tl for x in t} == {"en", "fa"})


# a fake executor to drive the real runner
class ProbeExec(se.Executor):
    kind = "fake"

    def __init__(self):
        self.seen = []

    def execute(self, req):
        mem = req.memory
        rec = {"mode": req.mode, "arm": req.arm, "epoch": mem.current_epoch() if mem is not None else None, "item": req.item["id"],
               "world_kind": req.world_kind, "frozen": bool(getattr(mem, "frozen", False)),
               "h": mem.state_hash() if mem is not None else None}
        self.seen.append(rec)
        r = se.EpisodeResult(sr=1.0, goal_ok=True, steps_to_find=3.0, walking_m=1.0, opens=1.0, stale_visits=0.0, retries=0.0, steps=10.0)
        if req.mode == "practice" and mem is not None:
            w = mem.begin_episode(req.episode_id, "practice")
            a_ = A.HonestEventAdapter(mem, wm=w, episode=req.episode_id, mode="practice")
            a_.on_task("Find the apple", "L1", "en")
            a_.on_terminal(True, 3, 0)
            a_.on_admission(True)
            a_.end()
            r.wm, r.admission = w, True
        return r


def small_loc_cfg(**kw):
    return se.location_config({}, run_id="loc", blocks=2, tests_per_block=3, **kw)


ex = ProbeExec()
wd = os.path.join(tmp, "run1")
cfg_s = small_loc_cfg()
runner = se.EvalRunner(cfg_s, b_items, man, ex, wd, factory=se.LtmFactory("FloorPlan1"), allow_items=allowed, forbid_items=forbidden, allow_sha=ash)
runner.run_all()
pr = [s for s in ex.seen if s["mode"] == "practice"]
te = [s for s in ex.seen if s["mode"] == "test" and s["arm"] == "places"]
check("the runner declared a reset before each practice attempt (epoch 1 then, per fresh block memory, always 1 at the first attempt)",
      pr and all(s["epoch"] == 1 for s in pr))
check("every test ran on a frozen copy at snapshot epoch + 1", te and all(s["frozen"] and s["epoch"] in (1, 2) for s in te)
      and any(s["epoch"] == 2 for s in te))
check("canonical world kind reaches the executor", all(s["world_kind"] == "canonical" for s in ex.seen))
jr = se.Journal(os.path.join(wd, "journal.jsonl"))
hdr = jr.header()["payload"]
check("the journal header records the allowlist hash", hdr.get("allowlist_sha") == ash)
check("every journaled item is allowed (gate 12.10)", all(r["item"] in allowed for r in jr.records() if "item" in r))
# resume checks run BEFORE any episode
bad_items = list(b_items) + [it for it in items if it["id"] in forbidden][:1]
try:
    se.EvalRunner(cfg_s, bad_items, man, ProbeExec(), os.path.join(tmp, "run2"), allow_items=allowed, forbid_items=forbidden, allow_sha=ash)
    ok_ = False
except se.AllowlistError:
    ok_ = True
check("items outside the allowlist are refused at construction (no episode runs)", ok_)
wd3 = os.path.join(tmp, "run3")
os.makedirs(wd3)
j3 = se.Journal(os.path.join(wd3, "journal.jsonl"))
j3.append("__header__", "header", {"config_sha": cfg_s.sha(), "manifest_sha": man["sha256"], "params_sha": None, "config": cfg_s.to_dict(),
                                   "executor_kind": "fake", "dataset_sha256": man["dataset_sha256"], "allowlist_sha": ash})
j3.append("x|test", "test", {"kind": "test", "item": sorted(forbidden)[0], "world_kind": "canonical", "block": 0, "arm": "none"})
ex3 = ProbeExec()
try:
    se.EvalRunner(cfg_s, b_items, man, ex3, wd3, allow_items=allowed, forbid_items=forbidden, allow_sha=ash)
    ok_ = False
except se.AllowlistError:
    ok_ = True
check("a resume whose journal holds a test-family item is refused before any episode", ok_ and ex3.seen == [])
try:
    se.EvalRunner(cfg_s, b_items, man, ProbeExec(), wd, allow_items=allowed, forbid_items=forbidden, allow_sha="0" * 64)
    ok_ = False
except se.JournalMismatch:
    ok_ = True
check("a resume with a different allowlist hash is refused", ok_)
rep = se.build_report(os.path.join(wd, "journal.jsonl"))
check("the report builds over P-location / canonical", any(r["protocol"] == "P-location" and r["world_kind"] == "canonical" for r in rep["means"]))

# ================================================================================================ 10
print("\n10. the live chain on the fake kitchen: observation -> claim -> where_is -> chooser -> hint -> first visit -> sighting")
mc = L.LongTermMemory(os.path.join(tmp, "chain.db"), kitchen="Fake")
mc.reset_world("k1")
ad = A.HonestEventAdapter(mc, episode="k-prac", mode="practice", detector_rev="fake")
t_p, h_p, sk_p = build(objs=world("cab"), adapter=ad, **EXH)
ad.on_task("Find the wine bottle", "L1", "en")
sk_p.start_episode("p")
r_p = sk_p.exec_step({"action": "search_for", "object": "WineBottle"})
ad.on_terminal(True, h_p.step, ad.n_opens, ad.walk_m)
ad.on_admission(True)
ad.end()
cons = mc.end_episode(ad.wm, admission=True)
claims = [r for r in mc.store.select("location_claim") if r["otype"] == "WineBottle"]
check("practice: the bottle was found and a claim was written (observation -> claim)", r_p["ok"] and cons["status"] == "committed" and len(claims) == 1
      and mc.store.select("place_instance", where={"place": claims[0]["place"]})[0]["rtype"] == "Cabinet")
ev_rows = mc.store.select("evidence", where={"claim_table": "location_claim", "claim": f"location_claim/{claims[0]['claim']}"})
check("the claim has an evidence row and a write_log entry", len(ev_rows) >= 1 and mc.store.select("write_log", where={"event_id": ev_rows[0]["event_id"]}))
sight = [e for e in ad.stream if e["kind"] == "sighting" and e["otype"] == "WineBottle"]
check("the evidence src_ev hash points at a real sighting event of this practice episode",
      any(lev.ev_hash("k-prac", e["ev"]) in json.dumps(ev_rows[0]["src_ev"]) for e in sight), str(ev_rows[0]["src_ev"]))
snapc = os.path.join(tmp, "chain-snap")
mc.snapshot(snapc)


def test_run(hints_on):
    fz = L.LongTermMemory.open_frozen(snapc, os.path.join(tmp, f"chain-tl{int(hints_on)}.db"), kitchen="Fake")
    fz.begin_test_epoch()
    a_ = A.HonestEventAdapter(fz, episode=f"k-test{int(hints_on)}", mode="test", detector_rev="fake")
    ch_ = A.LtmChooser(fz, a_.wm, fallback=lambda *x: [])
    a_.hint_source = ch_.hints_for
    t_, h_, sk_ = build(objs=world("cab"), adapter=a_, chooser=ch_, hints=(ch_.hints_for if hints_on else None), **EXH)
    a_.on_task("Find the wine bottle", "L1", "en")
    sk_.start_episode("t")
    r_ = sk_.exec_step({"action": "search_for", "object": "WineBottle"})
    return t_, h_, sk_, a_, ch_, r_, fz


tN, hN, skN, adN, chN, rN, fzN = test_run(False)
tH, hH, skH, adH, chH, rH, fzH = test_run(True)
check("test: where_is offered a learned T2 candidate; the chooser put its type first (where_is -> chooser)",
      chH.decisions[-1]["learned"] and chH.decisions[-1]["learned"][0][1] == "T2" and chH.decisions[-1]["order"][0] == "Cabinet" and chH.decisions[-1]["changed"])
check("a hint was published with the claim's place xyz", len(chH.hints_for("WineBottle")) == 1)
check("the hint changed the ACTUAL route: fewer containers inspected and fewer primitives",
      len(rH["detail"]["search"]["inspections"]) < len(rN["detail"]["search"]["inspections"]) and len(tH.calls) < len(tN.calls),
      f"{len(rH['detail']['search']['inspections'])} vs {len(rN['detail']['search']['inspections'])}; {len(tH.calls)} vs {len(tN.calls)}")
rd = live.route_diff(adN.searches, adH.searches)
check("route_diff (ordered inspected containers per search) reports the change", rd and rd[0]["changed"]
      and len(rd[0]["inspected_a"]) > len(rd[0]["inspected_b"]) and rd[0]["inspected_b"][0][0] == "Cabinet")
check("search metrics: first detection primitive, actual cap, censoring flag are recorded",
      adH.searches[0]["first_seen"] is not None and adH.searches[0]["steps_to_find"] == adH.searches[0]["first_seen"] - adH.searches[0]["t0"]
      and adH.searches[0]["cap"] and adH.searches[0]["cap"] > adH.searches[0]["steps_to_find"] and adH.searches[0]["censored"] is False)
check("metrics: the hinted first visit is memory-ranked and not stale; steps_to_find is smaller with the hint",
      adH.inspections[0]["memory_ranked"] and not adH.inspections[0]["stale"] and adH.searches[0]["steps_to_find"] < adN.searches[0]["steps_to_find"]
      and adH.searches[0]["memory_ranked_visits"] == 1)
cr = live.chain_report(mc, adH, chH, "WineBottle")
check("chain_report: claim place, tier, evidence + write_log, order change, first visit, search and the first sighting event",
      cr["found_chain"] and cr["tier"] == "T2" and cr["evidence"] and cr["evidence"][0]["write_log"] and cr["changed"]
      and cr["first_visit"]["memory_ranked"] and cr["search"]["found"] and cr["first_sighting_ev"] is not None, str(cr)[:300])
check("frozen state unchanged by both tests", fzN.state_hash() == L.LongTermMemory.open_frozen(snapc, os.path.join(tmp, "chk.db"), kitchen="Fake").state_hash()
      and fzH.state_hash() == fzN.state_hash())
check("no event rejected, no hook error in either test", adN.rejected == adH.rejected == [] and skN.event_errors == skH.event_errors == [])
# none arm (ltm=None): the same validated stream and metrics, no memory
a_none = A.HonestEventAdapter(None, episode="k-none", mode="test", kitchen="Fake", detector_rev="fake")
t_n, h_n, sk_n = build(objs=world("cab"), adapter=a_none, **EXH)
a_none.on_task("Find the wine bottle", "L1", "en")
sk_n.start_episode("t")
r_n = sk_n.exec_step({"action": "search_for", "object": "WineBottle"})
check("arm none: the same instrumentation without a long-term memory; identical calls to the hook-free baseline",
      a_none.rejected == [] and a_none.searches and t_n.calls == tN.calls and a_none.metrics()["opens"] == adN.metrics()["opens"])

# ================================================================================================ 11
print("\n11. step5_live core: fail-fast and the verdict")


def core_run(practice, mutate=None, score_value=True, mode=None, wall=None):
    m_ = L.LongTermMemory(os.path.join(tmp, f"core{practice}{mutate}.db"), kitchen="Fake")
    m_.reset_world("z1")
    a_ = A.HonestEventAdapter(m_, episode=f"core-{practice}-{mutate}", mode="practice" if practice else "test")
    if not practice:
        pass
    t_, h_, sk_ = build(objs=world("cab"), adapter=a_, **EXH)
    if mutate == "reject":
        a_.rejected.append({"ev": 0, "kind": "x", "error": "forced"})
    if mutate == "hook":
        sk_.event_errors.append({"kind": "open", "error": "forced"})

    def _score():
        with h_.scoring():
            h_.meta("scoring")
            if mutate == "audit":
                h_.audit.invalid = True                      # a violation recorded during the episode
            return score_value
    out = live.run_episode_core(h=h_, sk=sk_, adapter=a_, steps=[{"action": "search_for", "object": "WineBottle"}], holding=None,
                                text="Find the wine bottle", level="L1", lang="en", score=_score, practice=practice, policy="console",
                                **({} if wall is None else {"wall_cap_s": wall[0], "t_start": wall[1]}))
    return out, a_, m_


out, a_, m_ = core_run(True)
kinds = [e["kind"] for e in a_.stream]
check("practice: terminal, then the verdict, then episode_end (last three events)", kinds[-3:] == ["terminal", "admission_verdict", "episode_end"])
check("... the verdict is the scoring block's bool, and the row has the cost fields", a_.wm.admission["goal_ok"] is True and out["goal_ok"]
      and live.aggregate_result(out)["steps_to_find"] > 0 and live.aggregate_result(out)["sr"] == 1.0)
check("practice consolidates", m_.end_episode(a_.wm, admission=True)["status"] == "committed")
# K23: the 900 s episode wall cap (between steps; a cap already passed skips every step as episode_budget / episode_wall)
import time as _time
out_w, _a, _m = core_run(False, wall=(900, _time.time() - 901))
check("K23 wall cap passed -> step not run, cause episode_budget under episode_wall",
      [(r["cause"], r.get("under"), r["prims"]) for r in out_w["steps"]] == [("episode_budget", "episode_wall", 0)])
out_w2, _a, _m = core_run(False, wall=(900, _time.time()))
check("K23 wall cap not passed -> step runs as before", out_w2["steps"][0]["cause"] != "episode_budget")
out, a_, m_ = core_run(False)
kinds = [e["kind"] for e in a_.stream]
check("test: terminal, episode_end, and NO verdict anywhere in the stream", kinds[-2:] == ["terminal", "episode_end"] and "admission_verdict" not in kinds)
check("test: consolidation writes nothing", m_.end_episode(a_.wm)["status"] == "skipped_mode")
for mut, label in (("reject", "a rejected event"), ("hook", "a failing hook"), ("audit", "an invalid honesty audit")):
    try:
        core_run(True, mutate=mut)
        ok_ = False
    except live.LiveInvariant:
        ok_ = True
    check(f"fail fast: {label} aborts the run (nothing consolidated)", ok_)
ex = live.make_executor("places", policy="console", hints=True)
check("make_executor: memory=places, console policy, hints on; the simulator is not touched at construction", ex._ag is None and ex.memory == "places"
      and ex.policy == "console" and ex.hints and ex.kind == "live")
try:
    live.make_executor("habits")
    ok_ = False
except ValueError:
    ok_ = True
check("dormant channels (habits / lessons / plans) are refused", ok_)


# ================================================================================================ 12
print("\n12. Astra B1-wire review: callbacks, frames, parse failures, stale / overlapping hints, provenance, defaults")
# (1) any adapter callback failure is recorded and fails the episode fast; never raised into the agent
m12 = L.LongTermMemory(os.path.join(tmp, "a12.db"), kitchen="Fake")
m12.reset_world("a12")
ad12 = A.HonestEventAdapter(m12, episode="a12-ep", mode="practice")
ad12.on_observation = lambda *a_: (_ for _ in ()).throw(ValueError("boom"))
t12, h12, sk12 = build(objs=world("cab"), adapter=ad12, **EXH)
sk12.start_episode("t")                                         # the sweep calls om.update -> the broken callback
check("a non-contract callback failure never reaches the agent (the sweep completed)", h12.step > 0)
check("... it is recorded in adapter.errors", ad12.errors and ad12.errors[0]["error"].startswith("ValueError"))
try:
    live._check(h12, sk12, ad12)
    ok_ = False
except live.LiveInvariant:
    ok_ = True
check("... and _check raises LiveInvariant", ok_)


# (2) a chooser failure is recorded and fails the episode (the search policies swallow it and would silently use the prior)
class BrokenLtm:
    kitchen = "Fake"

    def where_is(self, *a_, **k_):
        raise RuntimeError("db gone")

    def frame_of_epoch(self, e):
        return "f0"


bad = A.LtmChooser(BrokenLtm(), None, fallback=lambda *a_: [])
t13, h13, sk13 = build(objs=world("cab"), chooser=bad, on_event=lambda k, f: None, **EXH)
sk13.start_episode("t")
r13 = sk13.exec_step({"action": "search_for", "object": "WineBottle"})
check("the search swallowed the chooser exception and carried on (as step 4 does)", any("chooser failed" in e for e in r13["detail"]["events"]))
check("... but the chooser recorded it", bad.errors and bad.errors[0]["error"].startswith("RuntimeError"))
ad13 = A.HonestEventAdapter(None, episode="a13", mode="test", kitchen="Fake")
try:
    live._check(h13, sk13, ad13, bad)
    ok_ = False
except live.LiveInvariant:
    ok_ = True
check("... and the executor check fails the episode (LiveInvariant)", ok_)
# (3) a parse the lexicon cannot fully resolve: types are the surface mentions, in order
mp = L.LongTermMemory(os.path.join(tmp, "p12.db"), kitchen="Fake")
wp = mp.begin_episode("p12", "test")
ap_ = A.HonestEventAdapter(mp, wm=wp, episode="p12", mode="test")
ap_.on_task("Put the apple in the fridge and the mug on the counter", "L2", "en")
check("parse not ok -> block types are the surface mentions of movable types, in order of mention", mp.block_types(wp) == ["Apple", "Mug"],
      str(mp.block_types(wp)))
wq = mp.begin_episode("p12b", "test")
A.HonestEventAdapter(mp, wm=wq, episode="p12b", mode="test").on_task("Put the apple in the fridge and put the mug on the counter", "L2", "en")
check("parse ok -> the roles decide (same types here)", mp.block_types(wq) == ["Apple", "Mug"])
mp.close()
# (4) a re-sighted track that exhausted its allowance and then MOVES is re-emitted with its new container
ad14 = A.HonestEventAdapter(None, episode="a14", mode="practice", kitchen="Fake")
ag14 = A.FakeHonestAgent(ad14, episode="a14")
for _ in range(A.REFRESH_CAP + 3):
    ag14.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9},
               {"type": "CounterTop", "box": [0, 82, 95, 95], "depth": 2.5}])
n_before = len([e for e in ad14.stream if e["kind"] == "sighting"])
check("the allowance is exhausted (capped)", n_before == 1 + A.REFRESH_CAP, str(n_before))
ag14.look([{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "CounterTop", "box": [0, 82, 95, 95], "depth": 2.5},
           {"type": "Apple", "box": [20, 84, 28, 92], "depth": 2.4}])      # the apple is now on the counter, far from the old spot
last = [e for e in ad14.stream if e["kind"] == "sighting"][-1]
check("after the cap, a moved track is emitted again (superseded) with the NEW container: the cap cannot keep an obsolete positive",
      len([e for e in ad14.stream if e["kind"] == "sighting"]) == n_before + 1 and last["in_rtype"] == "CounterTop", str(last))
# (5) cross-frame candidates: type advice only, never untransformed coordinates as hints
fz = L.LongTermMemory.open_frozen(snapc, os.path.join(tmp, "xf.db"), kitchen="Fake")
fz.begin_test_epoch(frame="f1", map_rev="m9")
chx = A.LtmChooser(fz, fz.begin_episode("xf", "test"), fallback=lambda *a_: [])
ox = chx("WineBottle", ["WineBottle", "Cabinet", "Fridge"], [])
check("another frame: the learned TYPE still ranks first (T2, advice)...", ox[:1] == ["Cabinet"])
check("...but no coordinate hint is published for a place of another frame", chx.hints_for("WineBottle") == [])
fz.end_test_epoch()
fz.begin_test_epoch(frame="f0", map_rev="m0")
chx2 = A.LtmChooser(fz, fz.begin_episode("xf2", "test"), fallback=lambda *a_: [])
chx2("WineBottle", ["WineBottle", "Cabinet", "Fridge"], [])
check("the same frame: the hint is published", len(chx2.hints_for("WineBottle")) == 1)
fz.close()


# (6) stale / overlapping hints keep the exhaustive fallback: the target is in the far cabinet, hints point at the FRIDGE and at the
# other cabinet (both wrong, the second one overlapping a neighbour)
def stale_hints(otype):
    return [{"rtype": "Fridge", "xyz": [2.95, 0.9, 1.0], "radius": 1.0}, {"rtype": "Cabinet", "xyz": [2.95, 0.6, 0.4], "radius": 2.0}]


mo6 = L.LongTermMemory(os.path.join(tmp, "o6.db"), kitchen="Fake")
mo6.reset_world("o6")
ad6 = A.HonestEventAdapter(mo6, episode="o6", mode="practice")
ad6.hint_source = stale_hints
t6, h6, sk6 = build(objs=world("cab"), adapter=ad6, hints=stale_hints, **EXH)
ad6.on_task("Find the wine bottle", "L1", "en")
sk6.start_episode("t")
r6 = sk6.exec_step({"action": "search_for", "object": "WineBottle"})
check("stale / overlapping hints: the target (in a container no hint names) is still found - the exhaustive fallback is intact",
      r6["ok"] and r6["detail"]["search"]["found"] is True and r6["detail"]["search"]["hinted"], json.dumps(r6["detail"]["search"].get("visited")))
check("... the hinted container was inspected first and counted STALE (an unsuccessful hinted inspection, not evidence of absence)",
      ad6.searches[0]["stale_visits"] >= 1 and ad6.searches[0]["memory_ranked_visits"] >= ad6.searches[0]["stale_visits"]
      and ad6.searches[0]["inspected"][0][0] == "Fridge", str(ad6.searches[0]["inspected"]))
# (7) provenance: a B memory holds only B episodes
mq = L.LongTermMemory(os.path.join(tmp, "q.db"), kitchen="Fake")
for ep in ("ep:" + "0" * 24, "smoke-practice-1-x", "rp:abcdef012345"):
    mq.reset_world("r-" + ep.replace(":", "-"))
    a_ = A.HonestEventAdapter(mq, episode=ep, mode="practice")
    a_.on_task("Find the apple", "L1", "en")
    a_.on_terminal(True, 3, 0)
    a_.on_admission(True)
    a_.end()
    assert mq.end_episode(a_.wm, admission=True)["status"] == "committed"
    if ep.startswith("rp:"):
        try:
            live.assert_b_memory(mq)
            ok_ = False
        except live.LiveInvariant:
            ok_ = True
        check("a replay seed receipt (rp:...) makes the memory unusable for B", ok_)
    else:
        live.assert_b_memory(mq)
        check(f"episode {ep[:8]}... is a B episode (accepted)", True)
mq.close()
# (8) everything defaults OFF
check("LiveExecutor / make_executor default to memory none and hints off",
      inspect.signature(live.make_executor).parameters["memory"].default == "none"
      and inspect.signature(live.make_executor).parameters["hints"].default is False
      and inspect.signature(live.LiveExecutor.__init__).parameters["memory"].default == "none"
      and inspect.signature(live.LiveExecutor.__init__).parameters["hints"].default is False)
src_live = open(os.path.join(HERE, "step5_live.py"), encoding="utf-8").read()
check("the smoke CLI defaults: --memory off, --hints off, dev-pilot items refused unless --allow-dev-pilot",
      'ap.add_argument("--memory", choices=("off", "places"), default="off")' in src_live
      and 'ap.add_argument("--hints", choices=("on", "off"), default="off")' in src_live
      and 'ap.add_argument("--allow-dev-pilot", action="store_true"' in src_live)
# A26 (K-B2b 2026-10-13) changed step4_refplans.py on purpose (STEP_CAUSES += no_cutting_tool, SOURCES += type_scope.py); B1 still left
# it untouched: with exactly those two K-B2b edits reverted the file is byte-identical to its pre-B1 copy
_rp = open(os.path.join(HERE, "step4_refplans.py"), "rb").read().decode("utf-8").replace(chr(13), "")
_A26_NEW = ["#: the 7 honest step causes (honest_skills.CAUSES; A26 / K-B2b 2026-10-13 added no_cutting_tool) + outcome categories of an item",
            '               "retry_exhausted", "no_cutting_tool")', '"step4_refplans.py",', '                      "type_scope.py"),']
_A26_OLD = ["#: the 6 honest step causes (honest_skills.CAUSES) + outcome categories of a whole item",
            '               "retry_exhausted")', '"step4_refplans.py"),', None]
_lines = []
for _ln in _rp.split(chr(10)):
    _s = _ln.strip()
    _hit = [n for n in _A26_NEW if n.strip() and (_s == n.strip() or _s.endswith(n.strip()))]
    if not _hit:
        _lines.append(_ln)
        continue
    _o = _A26_OLD[_A26_NEW.index(_hit[0])]
    if _o is not None:
        _lines.append(_ln.replace(_hit[0].strip(), _o.strip()))
_pre = open(os.path.join(HERE, "step5", "_checkpoints", "B1", "step4_refplans.py.pre-B1"), "rb").read().decode("utf-8").replace(chr(13), "")
check("step4_refplans.py is untouched by B1 (identical to its pre-B1 copy once the two A26 K-B2b edits are reverted)",
      chr(10).join(_lines) == _pre)


# (9) B2 flags: default OFF; behaviour only with hints AND hint_goals; the executor never lets a live run write the thesis trace store
_, _, sk_def = build(hints=lambda o: [], **EXH)
check("B2: HonestSkills(hint_goals) defaults to False and hint_boost to None",
      sk_def.hint_goals is False and sk_def.hint_boost is None
      and inspect.signature(hs.HonestSkills.__init__).parameters["hint_goals"].default is False
      and inspect.signature(hs.HonestSkills.__init__).parameters["hint_boost"].default is None)
_, _, sk_b = build(hints=lambda o: [dict(rtype="Cabinet", xyz=(0.0, 0.5, 0.0), radius=1.0)], hint_boost=9.0, **EXH)
check("B2: hint_boost overrides the multiplier only at a hinted place of its type",
      sk_b._hint_boost("Cup", "Cabinet", (0.2, 0.5, 0.0)) == 9.0 and sk_b._hint_boost("Cup", "Cabinet", (5.0, 0.5, 5.0)) == 1.0
      and sk_def._hint_boost("Cup", "Cabinet", (0.2, 0.5, 0.0)) == 1.0)
_, _, sk_g = build(hints=lambda o: [dict(rtype="Cabinet", xyz=(2.0, 0.5, 2.0), radius=0.3)], **EXH)
_, _, sk_g1 = build(hints=lambda o: [dict(rtype="Cabinet", xyz=(2.0, 0.5, 2.0), radius=0.3)], hint_goals=True, **EXH)
sk_g.fronts, sk_g1.fronts = [], []
it0 = sk_g._plan_items("WineBottle", ["Cabinet"], ["Cabinet"])
it1 = sk_g1._plan_items("WineBottle", ["Cabinet"], ["Cabinet"])
check("B2: hint_goals off adds no item; on adds ONE navigation look item at the hinted point (boosted)",
      not any(i.get("hint") for i in it0) and sum(1 for i in it1 if i.get("hint")) == 1
      and [i for i in it1 if i.get("hint")][0]["kind"] == "place")
check("B2: the executor sets TRACE_ENABLED=0 before importing the simulator stack; --hint-goals / --hint-boost / --repeat default off",
      'os.environ.setdefault("TRACE_ENABLED", "0")' in src_live
      and 'ap.add_argument("--hint-goals", choices=("on", "off"), default="off"' in src_live
      and 'ap.add_argument("--hint-boost", type=float, default=None' in src_live
      and 'ap.add_argument("--repeat", type=int, default=1' in src_live
      and inspect.signature(live.LiveExecutor.__init__).parameters["hint_goals"].default is False)


# ================================================================================================ 12
print("\n12. P2-FIX (2026-10-14): non-ASCII parser reasons -> event-safe ASCII tokens (adapter -> working memory -> event contract)")
import hashlib as _hl                                                # noqa: E402


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:                                                # noqa: BLE001
        return False
    return False


import ltm_concepts as _lc                                           # noqa: E402
from ltm_wm import WorkingMemory as _WM                              # noqa: E402
_FA = ["complex:تا", "complex:به_جای", "complex:بعد از",
       "complex:بعد_از", "complex:قبل", "complex:قبل_از"]
_ASCII = ["multi_verb", "no_verb", "dangling_adjective", "unknown_noun_after_src", "complex:after", "fa_elided_object_inherited", ""]
check("P2: every ASCII reason is returned unchanged (byte-identical, same object)", all(A.reason_token(r) is r for r in _ASCII))
_enc = [A.reason_token(r) for r in _FA]
check("P2: a Persian complex marker keeps its ASCII head and becomes u8x + the UTF-8 hex of the rest",
      _enc[0] == "complex:u8x" + "تا".encode("utf-8").hex() and all(e.startswith("complex:u8x") for e in _enc), str(_enc))
check("P2: deterministic and injective (distinct reasons stay distinct; 'بعد از' != 'بعد_از')",
      [A.reason_token(r) for r in _FA] == _enc and len(set(_enc)) == len(_FA))
check("P2: every encoded reason passes the UNCHANGED contract token check (ltm_events._tok)",
      all(lev._tok(e, "t") == e for e in _enc))
_long = "complex:" + "ا" * 60
_el = A.reason_token(_long)
check("P2: past the 96-char limit the rest is u8h + 32 hex of sha256 over the whole reason; it passes _tok",
      _el == "complex:u8h" + _hl.sha256(_long.encode("utf-8")).hexdigest()[:32] and len(_el) <= 96 and lev._tok(_el, "t") == _el, _el)
_lh = "h" * 90 + ":" + "تا" * 5
_elh = A.reason_token(_lh)
check("P2: an over-long ASCII head -> the whole token is u8h + the sha prefix (<= 96, passes _tok)",
      _elh == "u8h" + _hl.sha256(_lh.encode("utf-8")).hexdigest()[:32] and lev._tok(_elh, "t") == _elh, _elh)
_nohead = "تا:x"
check("P2: a non-ASCII reason with no ASCII head is encoded whole", A.reason_token(_nohead) == "u8x" + _nohead.encode("utf-8").hex())
check("P2: the contract itself still refuses a non-ASCII token (ltm_events._tok unchanged)", raises(lambda: lev._tok(_FA[0], "t"), lev.EventError))
check("P2: a simulator-id-like ASCII reason is NOT repaired: passed through unchanged, so the contract still refuses it",
      A.reason_token("Apple|+1.0|0.5|") == "Apple|+1.0|0.5|" and raises(lambda: lev._tok("Apple|+1.0|0.5|", "t"), lev.EventError)
      and raises(lambda: lev._tok("Bread_physics", "t"), lev.EventError))
check("P2: a non-string reason is refused by the adapter (never coerced)", raises(lambda: A.reason_token(3), A.AdapterError))
_t_mud = "سیب‌زمینی را تا تمیز نشده بشوی"
_gen = json.load(open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8"))["items"]
_hit = [it for it in _gen if any(not r.isascii() for r in _lc.parse(it["fa"])["reasons"])]
_p2_rej = []
for it in _hit:
    for _arm in ("none", "ltm"):
        _w = _WM("p2:" + it["id"] + _arm)
        _ad = A.HonestEventAdapter(None, wm=_w, episode="p2:" + it["id"] + _arm, mode="test", kitchen="Fake", detector_rev="fake")
        try:
            _ad.on_task(it["fa"], it["level"], "fa")
            _tc = [e for e in _ad.stream if e["kind"] == "task_context"][0]
            _p = _lc.parse(it["fa"])
            _ok = (lev.validate(dict(_tc)) == _tc and _tc["concepts"]["roles"] == _p["roles"] and _tc["concepts"]["ok"] == bool(_p["ok"])
                   and len(_tc["concepts"]["reasons"]) == len(_p["reasons"]) and all(r.isascii() for r in _tc["concepts"]["reasons"])
                   and [r for r in _tc["concepts"]["reasons"] if "u8" not in r] == [r for r in _p["reasons"] if r.isascii()])
            if not _ok:
                _p2_rej.append(it["id"])
        except Exception as e:                                       # noqa: BLE001
            _p2_rej.append(f"{it['id']}: {type(e).__name__}")
check("P2: the dataset's FA records with a Persian parser reason (8 on 2026-10-14) all pass adapter -> working memory -> lev.validate; "
      "roles / ok / reason count unchanged, ASCII reasons kept in place", len(_hit) == 8 and not _p2_rej, f"{len(_hit)} {_p2_rej}")
_mp = L.LongTermMemory(os.path.join(tmp, "p2.db"), kitchen="Fake")
_mp.reset_world("p2")
_wp = _mp.begin_episode("p2-ltm", "test")
_adp = A.HonestEventAdapter(_mp, wm=_wp, episode="p2-ltm", mode="test")
_adp.on_task(_hit[0]["fa"] if _hit else _t_mud, "L3", "fa")
check("P2: the same through a real long-term memory's test working memory: accepted, nothing rejected, memory_block renders",
      _adp.rejected == [] and isinstance(_mp.memory_block(_wp, "L3", "fa", retention="c2e")["text"], str))
_bad_ascii = []
for it in _gen:
    for _lg in ("en", "fa"):
        _p = _lc.parse(it[_lg])
        if all(r.isascii() for r in _p["reasons"]):
            _ad = A.HonestEventAdapter(None, episode="p2a", mode="test", kitchen="Fake")
            _ad.on_task(it[_lg], it["level"], _lg)
            if [e for e in _ad.stream if e["kind"] == "task_context"][0]["concepts"]["reasons"] != _p["reasons"]:
                _bad_ascii.append(f"{it['id']}:{_lg}")
check("P2: every ASCII-only record of the dataset (EN and FA) emits exactly the parser's reasons (no ASCII behaviour change)",
      not _bad_ascii, str(_bad_ascii[:5]))
_src_a = open(os.path.join(HERE, "ltm_adapter.py"), encoding="utf-8").read()
check("P2: on_task maps reasons through reason_token and nothing else in the concepts changed",
      '"reasons": [reason_token(r) for r in parsed["reasons"]]' in _src_a
      and 'concepts = {"roles": parsed["roles"], "ok": bool(parsed["ok"]), "reasons": [reason_token(r)' in _src_a)


print(f"\nstep5 adapter selftest: {PASS}/{PASS + FAIL}")
sys.exit(0 if FAIL == 0 else 1)
