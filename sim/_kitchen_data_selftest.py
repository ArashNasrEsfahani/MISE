"""Headless gate for the per-kitchen perception dataset policy (vocab.py + kitchen_data.py). Windows python, no sim.

    python sim/_kitchen_data_selftest.py
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kitchen_data as kd  # noqa: E402
import vocab  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f"  FAIL {name} {detail}")


# ---------------------------------------------------------------- vocab
check("base_type folds pieces", vocab.base_type("AppleSliced") == "Apple" and vocab.base_type("EggCracked") == "Egg")
check("base_type keeps plain", vocab.base_type("Fridge") == "Fridge")
check("kettle has no fill axis", "filled" not in vocab.axes_of("Kettle"))
check("pan has no fill axis", "filled" not in vocab.axes_of("Pan"))

mug = vocab.composite_targets("Mug")
check("mug composite count 9", len(mug) == 9, len(mug))            # broken + {clean,dirty} x {empty,3 liquids}
check("mug broken collapses", sum(1 for t in mug if t.get("broken")) == 1)
check("mug liquid only when filled", all((t["liquid"] is None) == (not t["filled"]) for t in mug if not t["broken"]))
bread = vocab.composite_targets("Bread")
check("bread: no whole-cooked", not any(t.get("cooked") and not t.get("sliced") for t in bread), bread)
check("bread count 3", len(bread) == 3, bread)
potato = vocab.composite_targets("Potato")
check("potato count 4", len(potato) == 4, potato)
egg = vocab.composite_targets("Egg")
check("egg count 3", len(egg) == 3, egg)
check("microwave count 4 (probe prunes open+on)", len(vocab.composite_targets("Microwave")) == 4)
check("stateless type has no targets", vocab.composite_targets("Floor") == [])

check("actions: toast", [a for a, _ in vocab.actions_for({"sliced": True, "cooked": True})] == ["SliceObject", "CookObject"])
check("actions: fried egg", [a for a, _ in vocab.actions_for({"broken": True, "cooked": True})] == ["BreakObject", "CookObject"])
check("actions: broken mug alone", vocab.actions_for({"broken": True, "dirty": False, "filled": False, "liquid": None}) == [("BreakObject", {})])
a = vocab.actions_for({"broken": False, "dirty": True, "filled": True, "liquid": "coffee"})
check("actions: dirty before fill", [x for x, _ in a] == ["DirtyObject", "FillObjectWithLiquid"] and a[1][1]["fillLiquid"] == "coffee", a)

mw_on = vocab.state_record({"objectType": "Microwave", "isOpen": False, "isToggled": True})
seq = [x for x, _ in vocab.actions_for({"open": True, "on": False}, mw_on)]
check("microwave: off before open", seq == ["ToggleObjectOff", "OpenObject"], seq)
seq = [x for x, _ in vocab.actions_for({"open": False, "on": True})]
check("microwave: close before on", seq == ["CloseObject", "ToggleObjectOn"], seq)

empty_mug = {"objectType": "Mug", "isDirty": False, "isFilledWithLiquid": False, "isBroken": False}
rec = vocab.state_record(empty_mug)
check("state_record liquid None when empty", rec["liquid"] == {"applicable": True, "value": None})
check("state_record inapplicable axis", rec["open"] == {"applicable": False, "value": None})
check("no-op step skipped", vocab.actions_for({"broken": False, "dirty": False, "filled": False, "liquid": None}, rec) == [])
water = vocab.state_record({"objectType": "Mug", "isFilledWithLiquid": True, "fillLiquid": "water"})
sw = [x for x, _ in vocab.actions_for({"broken": False, "dirty": False, "filled": True, "liquid": "wine"}, water)]
check("switch liquid empties first", sw == ["EmptyLiquidFromObject", "FillObjectWithLiquid"], sw)
check("same liquid is a no-op", vocab.actions_for({"broken": False, "dirty": False, "filled": True, "liquid": "water"}, water) == [])

piece = vocab.state_record({"objectType": "AppleSliced", "isSliced": False})
check("piece forced sliced", piece["sliced"]["value"] is True)
check("realised ok", vocab.realised({"sliced": True}, piece))
check("realised liquid mismatch", not vocab.realised({"filled": True, "liquid": "wine"}, water))
k = vocab.state_key("Mug", water)
check("state_key format", k == "Mug|broken=0,dirty=0,filled=1,liquid=water", k)
check("target_key == state_key of realised", vocab.target_key("Mug", {"broken": False, "dirty": False, "filled": True, "liquid": "water"}) == k)
check("state_key stateless", vocab.state_key("Floor", vocab.state_record({"objectType": "Floor"})) == "Floor|-")
cam = {"x": 0.0, "y": 1.5, "z": 0.0}
o = {"axisAlignedBoundingBox": {"center": {"x": 0.0, "y": 1.0, "z": 0.5}, "size": {"x": 0.1, "y": 0.1, "z": 0.1}}}
e = vocab.rim_elevation_deg(cam, o)
check("rim elevation ~41.99 deg", e is not None and abs(e - 41.99) < 0.1, e)

# ---------------------------------------------------------------- kitchen_data: scene selection on the real data
inv = json.load(open(os.path.join(HERE, "dataset", "scene_inventory.json"), encoding="utf-8"))
items = json.load(open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8"))["items"]
feas = kd.feasibility(items, inv)
sel = kd.select_scenes(feas, inv["scenes"])
check("selection = DEFAULT_SCENES", tuple(sel["scenes"]) == kd.DEFAULT_SCENES, sel["scenes"])
check("no instruction infeasible everywhere", sel["uncovered"] == [], sel["uncovered"])
check("FloorPlan5 excluded", "FloorPlan5" not in sel["scenes"])
check("feasible-everywhere reported", sel["feasible_everywhere"] <= sel["feasible_somewhere"] == sel["n_items"])

# ---------------------------------------------------------------- episodes
plan = kd.episode_plan("FloorPlan1", {"train": 64, "val": 8, "test": 8})
from collections import Counter
c = Counter(p["split"] for p in plan)
check("plan split counts", c == Counter({"train": 64, "val": 8, "test": 8}), c)
check("plan stable", plan == kd.episode_plan("FloorPlan1", {"train": 64, "val": 8, "test": 8}))
check("plan differs by scene", [p["seed"] for p in plan] != [p["seed"] for p in kd.episode_plan("FloorPlan6", {"train": 64, "val": 8, "test": 8})])
first = [p["split"] for p in plan[:20]]
check("early episodes include val and test", "val" in first and "test" in first, first)
small = kd.episode_plan("X", {"train": 3, "val": 1, "test": 1})
check("smoke plan has all splits", Counter(p["split"] for p in small) == Counter({"train": 3, "val": 1, "test": 1}))
check("stable_seed deterministic", kd.stable_seed("a", 1) == kd.stable_seed("a", 1) != kd.stable_seed("a", 2))

# ---------------------------------------------------------------- Quota (v2c-3: per-head state keys + combo floor)
MUG_DIRTY = {"broken": {"applicable": True, "value": False}, "dirty": {"applicable": True, "value": True},
             "filled": {"applicable": True, "value": False}, "liquid": {"applicable": True, "value": None}}
mug_heads = vocab.axis_keys("Mug", MUG_DIRTY)
check("axis_keys: empty vessel has no liquid head", mug_heads == ["Mug|broken=0", "Mug|dirty=1", "Mug|filled=0"], mug_heads)
check("axis_keys: liquid head when filled", "Mug|liquid=wine" in vocab.axis_keys("Mug", dict(MUG_DIRTY,
      filled={"applicable": True, "value": True}, liquid={"applicable": True, "value": "wine"})))
check("axis_keys: broken vessel keeps only broken", vocab.axis_keys("Mug", dict(MUG_DIRTY, broken={"applicable": True,
      "value": True})) == ["Mug|broken=1"])
COMBO = "Mug|broken=0,dirty=1,filled=0,liquid=-"
tg = kd.build_targets(["Mug", "Floor"], ["Mug|1"], mug_heads,
                      dict(kd.CAPTURE, type_target={"train": 2, "val": 1, "test": 1},
                           instance_target={"train": 1, "val": 0, "test": 0},
                           state_target={"train": 1, "val": 0, "test": 0}, combo_floor=1), [COMBO])
q = kd.Quota(tg)
lab_mug = {"type": "Mug", "orig_type": "Mug", "oid": "Mug|1", "state": MUG_DIRTY, "state_key": COMBO, "px": 2000,
           "state_min_px": 900}
lab_small = dict(lab_mug, px=100)
lab_floor = {"type": "Floor", "orig_type": "Floor", "oid": "Floor|1", "state": {}, "state_key": "Floor|-",
             "px": 50000, "state_min_px": 900}
check("gain = type + inst + mean head deficit + combo", abs(q.gain([lab_mug]) - 4.0) < 1e-9, q.gain([lab_mug]))
check("small crop gives no state credit", abs(q.gain([lab_small]) - 2.0) < 1e-9)
check("duplicate labels not double counted", abs(q.gain([lab_mug, lab_mug]) - 4.0) < 1e-9)
q.add([lab_mug, lab_floor], "train", episode=0)
check("add counts every head once", q.counts["state"]["train"] == {k: 1 for k in mug_heads}
      and q.counts["type"]["train"] == {"Mug": 1, "Floor": 1} and q.counts["combo"]["train"] == {COMBO: 1})
check("deficit halves", abs(q.deficit("type", "train", "Mug") - 0.5) < 1e-9)
check("val gain uses val targets", abs(q.gain([lab_mug], "val") - 1.0) < 1e-9)
check("accept any positive gain", kd.accept_train(0.01, random.Random(0)))
rng = random.Random(1)
check("accept zero gain rarely", sum(kd.accept_train(0.0, rng) for _ in range(2000)) < 300)
q.add([lab_mug, lab_floor], "train", episode=1)
check("train split done", q.split_done("train"))
check("val not done", not q.split_done("val"))
q2 = kd.Quota.from_dict(json.loads(json.dumps(q.to_dict())))
check("quota roundtrip", q2.counts == q.counts and q2.targets == q.targets and q2.combo_eps == q.combo_eps)
check("quota version saved", q.to_dict()["version"] == kd.QUOTA_VERSION)
a = kd.audit(q, ["Mug", "Floor", "Kettle"])
check("audit fails on missing type", not a["ok"] and a["missing_types"] == ["Kettle"], a["missing_types"])
check("audit names val shortfall", a["short"]["type"]["val"].get("Mug") == [0, 1])
cand = [({"dirty": True}, COMBO, mug_heads), ({"dirty": False}, "Mug|other", ["Mug|dirty=0"])]
picks = Counter(q.pick_target(cand, random.Random(i))[1] for i in range(200))
check("pick_target keeps options alive", picks["Mug|other"] > 0 and picks[COMBO] > 0, picks)
# combo floor needs distinct EPISODES too
qc = kd.Quota(kd.build_targets(["Mug"], [], mug_heads, dict(kd.CAPTURE, combo_floor=2), [COMBO]), combo_min_episodes=3)
qc.add([lab_mug, lab_mug, lab_mug], "train", episode=5)
check("combo floor: count met but one episode is still short", qc.counts["combo"]["train"][COMBO] == 3
      and qc.deficit("combo", "train", COMBO) > 0 and "episodes" in str(kd.audit(qc, ["Mug"])["short"]["combo"]["train"]))
for e in (6, 7):
    qc.add([lab_mug], "train", episode=e)
check("combo floor: met across 3 episodes", qc.deficit("combo", "train", COMBO) == 0)
# normalisation: a 4-head object does not outweigh a 1-head one when every head is equally short
qn = kd.Quota(kd.build_targets(["Mug", "Apple"], [], mug_heads + ["Apple|sliced=1"], kd.CAPTURE))
check("normalised head weight", abs(qn.state_value(mug_heads, None) - qn.state_value(["Apple|sliced=1"], None)) < 1e-9)
# replay recounts from records (the no-recapture switch of counting unit)
qr = kd.Quota(tg)
qr.replay([{"img": "a", "split": "train", "episode": 0, "labels": [lab_mug, lab_floor]},
           {"episode_record": {}}, {"img": "b", "split": "train", "episode": 1, "labels": [lab_mug]}])
check("replay recounts per head", qr.counts["state"]["train"]["Mug|dirty=1"] == 2 and qr.frames["train"] == 2
      and sorted(qr.combo_eps[COMBO]) == [0, 1])

# ---------------------------------------------------------------- v2b fixes
cap = vocab.composite_targets("Mug", capture=True)
check("capture never breaks a mug", len(cap) == 8 and not any(t.get("broken") for t in cap), cap)
check("capture never breaks a plate", not any(t.get("broken") for t in vocab.composite_targets("Plate", capture=True)))
check("capture still cracks an egg", any(t.get("broken") for t in vocab.composite_targets("Egg", capture=True)))
sliced = vocab.state_record({"objectType": "Apple", "isSliced": True})
check("compatible: sliced stays sliced", not vocab.compatible({"sliced": False}, sliced)
      and vocab.compatible({"sliced": True}, sliced))
check("compatible: reversible axes free", vocab.compatible({"dirty": True, "filled": True, "liquid": "wine"}, rec))
qs = kd.Quota(kd.build_targets(["Mug"], [], [], dict(kd.CAPTURE, combo_floor=5), ["K1"]))
qs.suppress_after = 3
for _ in range(3):
    qs.note_target("K1", False)
check("suppressed after 3 blind tries", qs.suppressed("K1") and qs.deficit("combo", "train", "K1") == 0.0)
check("type incentive survives suppression", qs.deficit("type", "train", "Mug") > 0)
check("audit lists tried-never-observed", kd.audit(qs, ["Mug"])["tried_never_observed"] == {"K1": 3})
qs.note_target("K1", True)
check("one hit lifts suppression", not qs.suppressed("K1"))
check("suppression survives roundtrip", kd.Quota.from_dict(json.loads(json.dumps(qs.to_dict()))).tried == {"K1": 4})
check("held-out spawn always", kd.CAPTURE["p_spawn_heldout"] == 1.0)
check("v3 episodes 128+20+20", kd.CAPTURE["episodes"] == {"train": 128, "val": 20, "test": 20})

# ---------------------------------------------------------------- RFS
f = kd.repeat_factors({"a": ["Floor", "Knife"], "b": ["Floor"], "c": ["Floor"], "d": ["Floor"]}, t=0.5)
check("rfs rare image repeats", f["a"] > 1.0 and f["b"] == 1.0, f)
check("rfs expand deterministic", kd.expand_repeats(f, 0) == kd.expand_repeats(f, 0))

# ---------------------------------------------------------------- ambiguity from VALIDATION
s = kd.select_ambiguity_types(["Fork", "Spoon", "Fridge", "Toaster", "Apple"],
                              val_ap50={"Fridge": 0.95, "Toaster": 0.5}, val_support={"Toaster": 50, "Fridge": 50},
                              val_confusion={("Fridge", "Toaster"): 0.08})
check("ambiguity a-priori", {"Fork", "Spoon", "Apple"} <= set(s["types"]))
check("ambiguity low AP joins", "Toaster" in s["types"])
check("ambiguity both confusion sides join", "Fridge" in s["types"])
s2 = kd.select_ambiguity_types(["Toaster"], val_ap50={"Toaster": 0.5}, val_support={"Toaster": 3})
check("ambiguity needs support", "Toaster" not in s2["types"])
check("ambiguity source is validation", s["source"] == "validation split")

check("training optimizer pinned", all(kd.TRAIN[m]["optimizer"] != "auto" for m in ("primary", "specialist", "baseline")))

# ---------------------------------------------------------------- hand-label review (state_review.py)
import state_review as sr  # noqa: E402
st_mug = {"filled": {"applicable": True, "value": True}, "liquid": {"applicable": True, "value": "coffee"},
          "dirty": {"applicable": True, "value": False}, "broken": {"applicable": True, "value": False}}
check("review truth: liquid", sr.truth_of("filled", st_mug) == "coffee")
check("review truth: binary", sr.truth_of("dirty", st_mug) == "0")
fr = [{"img": f"i{k}", "split": "train" if k < 30 else "test", "episode": k % 6, "labels": [
      {"oid": f"Mug|{k}", "type": "Mug", "box": [0, 0, 9, 9], "px": 100 * k, "dist": 0.1 * k, "state": st_mug}]}
      for k in range(40)]
cands = sr.candidates(fr, "FP")
check("review uses TRAIN only", all(c["img"] in {f"i{k}" for k in range(30)} for c in cands) and cands)
check("review questions only applicable axes", {c["axis"] for c in cands} == {"filled", "dirty", "broken"})
pick = sr.stratified(cands, 12, seed=0)
check("review sample size", len(pick) == 12)
check("review sample spans size bins", len({sr._bin(c["px"], sr.PX_BINS) for c in pick}) >= 2)
check("review sample deterministic", [c["img"] for c in pick] == [c["img"] for c in sr.stratified(cands, 12, seed=0)])
truth = {"q0": {"axis": "dirty", "truth": "0", "px": 100, "dist": 1.0},
         "q1": {"axis": "dirty", "truth": "1", "px": 5000, "dist": 1.0},
         "q2": {"axis": "filled", "truth": "coffee", "px": 5000, "dist": 1.0}}
res = sr.score(truth, {"q0": "0", "q1": "cant", "q2": "wine"})
check("review score outcomes", [r["outcome"] for r in sorted(res["rows"], key=lambda r: r["id"])] == ["right", "cant", "wrong"])
check("review page template exists", os.path.exists(sr.TEMPLATE))

# ---------------------------------------------------------------- driver helpers (import only; no simulator)
import build_kitchen_data as bk  # noqa: E402
o_before = {"objectId": "Apple|-00.47|+01.15|+00.48", "objectType": "Apple", "name": "Apple_7c1"}
o_after = {"objectId": "Apple|-01.91|+00.09|+01.98", "objectType": "Apple", "name": "Apple_7c1"}  # after a shuffle
check("identity survives a layout shuffle", bk.instance_key(o_before) == bk.instance_key(o_after) == "Apple_7c1")
kids = {o_after["objectId"]: o_after}
check("slice piece credits its parent", bk.instance_key(
    {"objectId": o_after["objectId"] + "|AppleSliced_0", "objectType": "AppleSliced", "name": "AS_0"}, kids) == "Apple_7c1")
qi = kd.Quota(kd.build_targets(["Apple"], ["Apple_7c1"], [], kd.CAPTURE))
qi.add([{"type": "Apple", "oid": o_after["objectId"], "inst": "Apple_7c1", "state_key": "", "px": 10}], "train")
check("instance quota counts by name, not id", qi.counts["inst"]["train"] == {"Apple_7c1": 1})
check("coverage gate configured", 0 < kd.CAPTURE["min_state_coverage"] <= 1)
check("anti-grind configured", kd.CAPTURE["saturation_window"] > kd.CAPTURE["saturation_keep"] >= 0)
check("dataset version bumped", bk.DATASET_VERSION == "kitchen_v3")
check("parent_of piece",bk.parent_of("Apple|+1|0|AppleSliced_2", "AppleSliced") == "Apple|+1|0")
check("parent_of original", bk.parent_of("Apple|+1|0", "Apple") is None)

ff_cfg = dict(kd.CAPTURE, check_every=0)
def _ff(setup):
    try:
        bk._fail_fast("FP", "/nonexistent", {"episode": 7}, setup, ff_cfg, 7)
        return "passed"
    except bk.CaptureAborted as exc:
        return str(exc)
ok_obj = {"realised": True}
check("fail-fast: v2b-style coverage aborts", "28/48" in _ff({"state_coverage": {"settable": 48, "set": 28, "missed": []},
                                                             "objects": {"a": ok_obj}}))
check("fail-fast: failed state settings abort", "did not take" in _ff({"state_coverage": {"settable": 10, "set": 10},
      "objects": {str(i): {"realised": i > 2} for i in range(10)}}))
check("fail-fast: healthy episode passes", _ff({"state_coverage": {"settable": 44, "set": 44},
                                                "objects": {"a": ok_obj}, "rounds": []}) == "passed")
check("checkpoint configured", kd.CAPTURE["check_every"] > 0)

qw = kd.Quota(kd.build_targets(["Mug"], [], [], dict(kd.CAPTURE, combo_floor=4), ["K"]))
qw.suppress_after = 2
check("state_weight: full combo deficit", abs(qw.state_weight([], "K") - 1.0) < 1e-9)
qw.note_target("K", False); qw.note_target("K", False)
check("state_weight: suppressed combo keeps a quarter", abs(qw.state_weight([], "K") - 0.25) < 1e-9
      and qw.deficit("combo", "train", "K") == 0)
check("state_weight: unknown key gets floor", qw.state_weight(["nope"], None) == 0.02)
check("policy v3-2 recorded", bk.POLICY_VERSION == "v3-2" and kd.CAPTURE["combo_floor"] == 10)

# ---------------------------------------------------------------- render guard (2026-09-29 red/white corruption)
import numpy as np
clean = np.full((160, 160, 3), 120, np.uint8); clean[:, :80] = (90, 91, 88)
red = clean.astype(np.int16); red[..., 0] += 80; red[..., 1] -= 35; red[..., 2] -= 35
red = np.clip(red, 0, 255).astype(np.uint8)
white = np.full((160, 160, 3), 255, np.uint8)
check("anomaly: clean frame passes", bk.frame_anomaly(clean) is None)
check("anomaly: pure white", bk.frame_anomaly(white) == "pure_white")
check("anomaly: red cast", bk.frame_anomaly(red) == "red_cast")
blue = clean.copy(); blue[..., 2] = 250; blue[..., 0] = 60; blue[..., 1] = 60
check("anomaly: blue cast", bk.frame_anomaly(blue) == "blue_cast")
class _Ev:
    metadata = {"lastActionSuccess": True}
class FakeAgent:
    def __init__(self, seq): self.seq, self.i = list(seq), 0
    @property
    def frame(self): return self.seq[min(self.i, len(self.seq) - 1)]
    def do(self, action, **kw): self.i += 1; return _Ev()
gcfg = dict(kd.CAPTURE, guard_tile=40)
ok, n = bk.render_guard(FakeAgent([clean, clean]), gcfg)
check("guard: stable clean view accepted after one extra render", ok and n == 1, (ok, n))
ok, n = bk.render_guard(FakeAgent([red, clean, clean, clean]), gcfg)
check("guard: after a corrupted render, 3 identical clean renders are required", ok and n == 3, (ok, n))
ok, n = bk.render_guard(FakeAgent([clean, red, clean, clean, clean]), gcfg)
check("guard: a corrupted render between clean ones is not accepted", ok and n == 4, (ok, n))
ok, n = bk.render_guard(FakeAgent([clean, red, clean, red, clean, clean, clean]), gcfg)
check("guard: an agreeing pair broken by a new glitch restarts the confirmation", ok and n == 6, (ok, n))
ok, n = bk.render_guard(FakeAgent([red] * 10), gcfg)
check("guard: stable but corrupted is REJECTED (agreement is not correctness)", not ok, (ok, n))
check("guard: that failure is reported as STABLE + colour signature", bk.GUARD_LAST["stable"] and bk.GUARD_LAST["anomaly"] == "red_cast",
      bk.GUARD_LAST)
ok, n = bk.render_guard(FakeAgent([red] * 10), gcfg, allow_anomaly=True)
check("guard: a view proven to be content (red wall) is accepted when allowed", ok and n == 1, (ok, n))
ok, n = bk.render_guard(FakeAgent([clean, red, clean, red, clean, red, clean, red, clean]), gcfg, allow_anomaly=True)
check("guard: the allowance never excuses DISAGREEING renders", not ok, (ok, n))
ok, n = bk.render_guard(FakeAgent([clean, red, clean, red, clean, red, clean, red, clean]), gcfg)
check("guard: an unstable failure is reported as NOT stable", not ok and not bk.GUARD_LAST["stable"], bk.GUARD_LAST)
local = clean.copy(); local[:10, :10] = (255, 0, 0)        # too small to move the whole-image MAE past 1.0 ...
check("guard test premise: whole-image MAE stays below the threshold",
      float(np.abs(local.astype(np.float32) - clean).mean()) < kd.CAPTURE["guard_mae"])
ok, n = bk.render_guard(FakeAgent([clean, local, clean, clean, clean]), gcfg)
check("guard: ... but the tile check still catches a local corruption", ok and n == 4, (ok, n))
ff2 = dict(kd.CAPTURE, check_every=0)
try:
    bk._fail_fast("FP", "/x", {"episode": 3}, {"state_coverage": {"settable": 1, "set": 1}, "objects": {}},
                  ff2, 3, {"views_guarded": 100, "guard_retries": 40})
    check("fail-fast: exploding guard retries abort", False)
except bk.CaptureAborted as exc:
    check("fail-fast: exploding guard retries abort", "render guard" in str(exc))
try:
    bk._fail_fast("FP", "/x", {"episode": 3}, {"state_coverage": {"settable": 1, "set": 1}, "objects": {}},
                  ff2, 3, {"views_guarded": 100, "guard_retries": 14, "guard_skipped": 0})
    check("fail-fast: a normal 14% retry episode (guard working) does NOT abort", True)
except bk.CaptureAborted:
    check("fail-fast: a normal 14% retry episode (guard working) does NOT abort", False)
try:
    bk._fail_fast("FP", "/x", {"episode": 3}, {"state_coverage": {"settable": 1, "set": 1}, "objects": {}},
                  ff2, 3, {"views_guarded": 100, "guard_retries": 10, "guard_skipped": 6})
    check("fail-fast: skipping > 5% of views aborts", False)
except bk.CaptureAborted:
    check("fail-fast: skipping > 5% of views aborts", True)

# ---- v3 (2026-09-30): original colours, per-instance held diversity, viewpoint floor
check("v3: no material randomisation", kd.CAPTURE["p_materials"] == 0.0)
check("v3: mild lighting on <= 20% of episodes", kd.CAPTURE["p_lighting"] <= 0.2
      and tuple(kd.CAPTURE["lighting_brightness"]) == (0.85, 1.15))
check("v3: held cap 12%", kd.CAPTURE["held_cap"] == 0.12)
check("v3: type target 1500 train", kd.CAPTURE["type_target"]["train"] == 1500)
q3 = kd.Quota(kd.build_targets(["Apple", "Mug"], ["Apple_1", "Mug_1"], [], kd.CAPTURE))
q3.vp_target, q3.held_target, q3.held_bg_target = 2, 2, 2
L3 = [{"type": "Apple", "oid": "a", "inst": "Apple_1", "state_key": "", "px": 50}]
check("v3: new viewpoint has gain", q3.vp_gain(L3, [0, 0, 0]) > 0)
q3.add_vp(L3, [0, 0, 0])
check("v3: same viewpoint no gain", q3.vp_gain(L3, [0, 0, 0]) == 0)
q3.add_vp(L3, [1, 0, 0])
check("v3: viewpoint target met -> no gain", q3.vp_gain(L3, [2, 0, 0]) == 0)
check("v3: unheld instance full deficit", q3.held_deficit("Apple_1") == 1.0)
q3.add_held("Apple_1", [0, 0, 0], (1.0, 2.0))
q3.add_held("Apple_1", [0, 0, 0], (1.5, 2.0))
check("v3: frames met but one background -> still short", q3.held_deficit("Apple_1") == 0.5)
q3.add_held("Apple_1", [3, 0, 1], (3.0, 2.0))
check("v3: frames + backgrounds met", q3.held_deficit("Apple_1") == 0.0)
r3 = kd.Quota.from_dict(json.loads(json.dumps(q3.to_dict())))
check("v3: held/vp state survives roundtrip", r3.held_n == q3.held_n and r3.held_pos == q3.held_pos
      and r3.vp == q3.vp and r3.held_target == 2)
r3.replay([{"img": "x", "split": "train", "episode": 1, "labels": L3, "cell": [5, 5, 5], "kind": "held",
            "pose": {"inst": "Apple_1", "cell": [5, 5, 5], "x": 0.0, "z": 0.0}}])
check("v3: replay rebuilds held + viewpoints", r3.held_n == {"Apple_1": 1} and r3.vp == {"Apple": [[5, 5, 5]]})
check("v3: quota version bumped", kd.QUOTA_VERSION == "v3-1")

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed" + (" — ALL PASS" if not FAIL else ""))
sys.exit(1 if FAIL else 0)
