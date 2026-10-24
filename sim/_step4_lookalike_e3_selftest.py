"""Offline selftest for stage E3 (2026-10-13): look-alike re-evaluation (DESSTA RevaluateObject) + the same-view duplicate-box
merge in the object memory (F-dup). Windows `python`, no WSL/GPU/Ollama/LLM.

Run from the repo root:  python sim/_step4_lookalike_e3_selftest.py

Method: A. NasrEsfahani et al., "Embodied AI for Kitchen Scene Rearrangement", Intelligent Service Robotics 19:71 (2026),
sec. 2.3.4 + Algorithm 4. Sections:
  1  M (Ambiguous Objects Map) from the primary's VAL confusion: direction, threshold, exclusions; the FloorPlan1 map pinned
  2  ambiguity points in one frame: look-alike class, low-confidence asked type, border filter, order, one object counted once
  3  F-dup: two boxes of one object in one view make ONE track; separate objects stay two; a later sighting is not "ambiguous"
  4  in the loop (fake kitchen): a Spoon the primary calls a Fork is picked up, the specialist says Spoon -> kept, step ok
  5  a real Fork look-alike is picked up, the specialist says Fork -> put back where it was; the real Spoon is found after
  6  the low-confidence trigger: a Spoon seen only at 0.15 -> picked up and kept; E3 OFF -> not found (the old hole)
  7  the cap: only Fork look-alikes -> at most LOOKALIKE_MAX checks, each put back, honest not_detected, hand empty
  8  scope: not for a non-look-alike type, not for search_for, not for Knife; LOOKALIKE_ON False -> no checks at all
  10 Astra E3 review: frozen development M; positive specialist-sourced confirmation; aiming at the candidate's own box;
     memory lifecycle (pickup / relabel / put); interaction-local checked points; bounded restoration without a drop;
     bounded revisit; the low threshold leaves Obs.dets unchanged; F-dup limits, depth, order, priority, ambiguity
  9  honesty: 0 violations / 0 oracle reads; the E3 code reads no simulator handle
"""
from __future__ import annotations

import contextlib
import inspect as _ins
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):                                # reuse the act gate's fake kitchen (its own checks stay quiet)
    import _step4_act_selftest as A
import honest                                                         # noqa: E402
import honest_skills as hs                                            # noqa: E402
import objmem                                                         # noqa: E402

hs.AIM_PROTRUDE_ON = True
hs.IDENT_ON = True
hs.LOOKALIKE_ON = True
objmem.DEDUP_FRAME = True
check("the act gate's fake kitchen imported clean", A.FAIL == 0, f"{A.FAIL} failing")

# ======================================================================================================== 1
print("1. M from the primary's confusion rows")
rows = [{"cls": "Spoon", "n": 100, "confused_with": {"Fork": 8, "ButterKnife": 5, "Statue": 1, "Knife": 9, "CounterTop": 7}},
        {"cls": "Fork", "n": 50, "confused_with": {"Spoon": 5}},
        {"cls": "Knife", "n": 40, "confused_with": {"ButterKnife": 10}},
        {"cls": "Apple", "n": 40, "confused_with": {"Tomato": 10}}]
m = hs.ambiguous_objects_map(rows, 0.02)
check("Spoon -> Fork, ButterKnife (>= 2 %), ordered by rate", list(m.get("Spoon", {})) == ["Fork", "ButterKnife"], str(m))
check("below the threshold (Statue 1 %) is out", "Statue" not in m["Spoon"])
check("Knife never enters (excluded type), neither as candidate nor as target", "Knife" not in m["Spoon"] and "Knife" not in m)
check("a surface never enters (CounterTop)", "CounterTop" not in m["Spoon"])
check("only specialist-class targets get an entry (Apple has none)", "Apple" not in m)
check("direction: Fork's entry comes from Fork's own row", m["Fork"] == {"Spoon": 0.1})
m1 = hs.load_ambiguous_map("FloorPlan1")
want = {"Spoon": ["Fork", "ButterKnife"], "Fork": ["Spoon", "ButterKnife"], "ButterKnife": ["Fork", "Spatula"],
        "Spatula": ["ButterKnife"], "SaltShaker": ["PepperShaker"], "PepperShaker": ["SaltShaker"], "Plate": ["Pot"],
        "Pan": ["Pot"], "Pot": ["Pan"]}
check("FloorPlan1 M (from kitchen_v3/FloorPlan1/val_errors_yolo26s.json) is the pinned map", {k: list(v) for k, v in m1.items()} == want, str(m1))
check("the _physics spelling resolves to the same map", hs.load_ambiguous_map("FloorPlan1_physics") == m1)
check("a kitchen without a confusion analysis -> {} (only low-confidence boxes then)", hs.load_ambiguous_map("FloorPlanX") == {})


# ======================================================================================================== 2
class O:
    def __init__(self, dets, low=(), wh=(100, 100)):
        self.dets, self.low = list(dets), list(low)
        self.depth = np.ones((wh[1], wh[0]), "float32")


def dd(t, c, b, ok=True):
    return {"type": t, "conf": c, "bbox_xyxy": b, "point": ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2), "point_ok": ok}


print("2. ambiguity points in one frame")
mm = {"Spoon": {"Fork": 0.08, "ButterKnife": 0.05}}
o = O([dd("Fork", 0.6, (40, 40, 50, 60)), dd("ButterKnife", 0.7, (60, 40, 70, 60)), dd("Apple", 0.9, (20, 20, 30, 30)),
       dd("Fork", 0.6, (1, 40, 8, 60)), dd("Fork", 0.9, (25, 40, 33, 60))], low=[dd("Spoon", 0.15, (20, 60, 30, 80)), dd("Spoon", 0.05, (80, 60, 88, 80))])
c = hs.lookalike_candidates(o, "Spoon", mm)
check("order: the asked type's low-confidence box, then look-alikes by confusion rate", [x["type"] for x in c] == ["Spoon", "Fork", "ButterKnife"],
      str([(x["type"], x["conf"]) for x in c]))
check("triggers recorded", [x["trigger"] for x in c] == ["low_conf", "lookalike", "lookalike"])
check("below LOOKALIKE_LOW (0.05) is not a point", all(x["conf"] >= hs.LOOKALIKE_LOW for x in c))
check("a box centre within 10 % of the border is dropped (the paper's hard filter)", all(x["bbox_xyxy"][0] > 5 for x in c))
check("a non-look-alike type is never a point", all(x["type"] != "Apple" for x in c))
o2 = O([dd("Fork", 0.6, (40, 40, 50, 60)), dd("ButterKnife", 0.5, (40, 41, 50, 61)), dd("Fork", 0.8, (60, 40, 70, 60), ok=False)])
c2 = hs.lookalike_candidates(o2, "Spoon", mm)
check("two boxes of one object count once; a box without a grounded point is skipped", len(c2) == 1 and c2[0]["type"] == "Fork", str(c2))
check("no M entry and no low box -> no points", hs.lookalike_candidates(o, "Apple", mm) == [])
check("a CONFIDENT look-alike box (Fork 0.90 >= the 0.75 ceiling) is that class, not a point", all(x["conf"] < hs.LOOKALIKE_CONF_CEIL for x in c))
c3 = hs.lookalike_candidates(O([dd("Spatula", 0.6, (40, 40, 50, 60))]), "ButterKnife", {"ButterKnife": {"Spatula": 0.0246}})
check("a pair with a tiny measured rate (ButterKnife -> Spatula 2.5 % < 5 %) never triggers a pickup", c3 == [])

# ======================================================================================================== 3
print("3. F-dup: same-view duplicate boxes of one object")


def obs_at(dets, step):
    depth = np.full((100, 100), 1.0, "float32")
    pose = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0, "y": 0, "z": 0}, "horizon": 0.0,
            "camera": {"x": 0.0, "y": 1.5, "z": 0.0}}
    for d in dets:
        d.setdefault("state", "?")
    return honest.Obs(step=step, rgb=None, depth=depth, pose=pose, dets=dets)


om = objmem.ObjectMemory("Fake", store=None, episode="d")
o3 = obs_at([dd("Spoon", 0.48, (40, 40, 52, 60)), dd("Spoon", 0.27, (41, 42, 53, 61)), dd("Spoon", 0.27, (39, 41, 51, 59))], 1)
rep = om.update(o3)
check("three boxes of one Spoon in one view -> ONE track", len([t for t in om.tracks.values() if t.type == "Spoon"]) == 1, str(rep))
check("every box carries that track; two are reported as second boxes", len({d["track"] for d in o3.dets}) == 1 and None not in {d["track"] for d in o3.dets}
      and sum(1 for s_ in rep["skipped"] if "second box" in s_[1]) == 2)
o4 = obs_at([dd("Spoon", 0.90, (40, 40, 52, 60))], 2)
om.update(o4)
check("a later 0.90 sighting associates (not 'ambiguous' between two co-located tracks)", o4.dets[0]["track"] is not None)
om2 = objmem.ObjectMemory("Fake", store=None, episode="d")
o5 = obs_at([dd("Fork", 0.9, (10, 40, 22, 60)), dd("Fork", 0.8, (70, 40, 82, 60))], 1)
om2.update(o5)
check("two separate forks (boxes apart) stay two tracks", len([t for t in om2.tracks.values() if t.type == "Fork"]) == 2)
objmem.DEDUP_FRAME = False
om3 = objmem.ObjectMemory("Fake", store=None, episode="d")
o6 = obs_at([dd("Spoon", 0.48, (40, 40, 52, 60)), dd("Spoon", 0.27, (41, 42, 53, 61))], 1)
om3.update(o6)
o7 = obs_at([dd("Spoon", 0.90, (40, 40, 52, 60))], 2)
om3.update(o7)
check("DEDUP_FRAME False reproduces the live failure: two tracks, and the 0.90 sighting is left unassociated",
      len([t for t in om3.tracks.values() if t.type == "Spoon"]) == 2 and o7.dets[0]["track"] is None)
objmem.DEDUP_FRAME = True


# ======================================================================================================== 4-8
class LabelDet:
    """The fake kitchen's detector with a primary that mislabels / under-scores some objects, and a specialist that reads the
    object in the hand right (``spec``). Boxes below ``conf`` are dropped, as the real detector does."""

    def __init__(self, thor):
        self.thor = thor
        self.names = {i: t for i, t in enumerate(sorted({o["type"] for o in thor.objs.values()} | {"Knife", "Fork", "Spoon"}))}
        self.calls = []

    def detect(self, rgb, conf=0.25, inspecting=None):
        self.calls.append((conf, inspecting))
        out = []
        for o, b, zc in self.thor.boxes:
            held = o["id"] == self.thor.held
            if held and inspecting:
                t, c, src = o.get("spec", o["type"]), o.get("sconf", 0.9), "specialist"
            else:
                t, c, src = o.get("prim", o["type"]), o.get("pconf", 0.9), "primary"
            if c < conf:
                continue
            out.append({"type": t, "conf": c, "bbox_xyxy": b, "states": {}, "state": "?", "source": src})
            if held and inspecting and o.get("spec2"):
                t2, c2 = o["spec2"]
                out.append({"type": t2, "conf": c2, "bbox_xyxy": b, "states": {}, "state": "?", "source": "specialist"})
        return out


def kit(spoon=True, fork=False, spoon_kw=None, n_forks=1):
    objs = [dict(id="CounterTop|+1.20|+0.90|+2.95", type="CounterTop", pos=(1.2, 0.9, 2.95), size=1.0, receptacle=True)]
    if spoon:
        sp = dict(id="Spoon|+1.15|+0.97|+2.95", type="Spoon", pos=(1.15, 0.97, 2.95), size=0.2, pickupable=True)
        sp.update(spoon_kw or {})
        if "prim" in sp:
            sp.setdefault("pconf", 0.6)                         # a mislabel is an UNCERTAIN box (below the 0.75 ceiling)
        objs.append(sp)
    for k in range(n_forks if fork else 0):
        x = 1.20 + 0.2 * k                                     # the first fork at the counter centre: the fake's put lands there
        objs.append(dict(id=f"Fork|+{x:.2f}|+0.97|+2.95", type="Fork", pos=(x, 0.97, 2.95), size=0.2, pickupable=True, pconf=0.6))
    objs.append(dict(id="Apple|+0.50|+0.97|+2.60", type="Apple", pos=(0.5, 0.97, 2.6), size=0.2, pickupable=True))
    return objs


def setup(objs, **kw):
    t = A.FakeThor(objs, (4, 4), 0.0)
    det = LabelDet(t)
    h = honest.HonestAgent(t, det, mode="honest")
    gm = A.GridMap("Fake", A.CELL, (0.0, 0.0), A.FREE)
    om_ = objmem.ObjectMemory("Fake", store=None, episode="t")
    sk = hs.HonestSkills(h, gm, om_, chooser=lambda *a: [], search="exhaustive", tidy=True, **kw)
    sk.start_episode("t")
    return t, det, h, sk


def held_type(t):
    return t.objs[t.held]["type"] if t.held else None


print("4. a Spoon the primary calls a Fork: picked up, the specialist says Spoon -> kept")
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
st_ = sk.identity_stats
check("step ok, the SPOON is in the hand (fake world, read by the test only)", r["ok"] and held_type(t) == "Spoon", f"{r['cause']} {held_type(t)} {r['detail'].get('why')}")
check("own holding belief is Spoon", h.holding == "Spoon")
check("the Fork was first seen out of reach, then revisited (Algorithm 4 line 2)", sk.identity_stats["la_unreached"] >= 1 and sk.identity_stats["la_revisits"] >= 1, str(sk.identity_stats))
check("one look-alike check, kept, counted with its detector call and primitives", st_["la_checks"] == 1 and st_["la_kept"] == 1
      and st_["la_detector_calls"] == 1 and st_["la_prims"] >= 1, str(st_))
recs = [x for x in sk.identity_log if x.get("kind") == "lookalike" and "seen" in x]
check("the record says: candidate Fork (look-alike trigger), specialist Spoon, kept", recs and recs[0]["candidate"] == "Fork"
      and recs[0]["trigger"] == "lookalike" and recs[0]["seen"] == "Spoon" and recs[0]["kept"], str(recs))
check("verify says how it was confirmed", "lookalike-specialist" in r["detail"]["verify"])
check("no second pickup was sent after the kept one", sum(1 for c_ in t.calls if c_[0] == "PickupObject") == 1)
check("no object id in the result", A.no_ids(r))
check("low-confidence boxes asked only inside the pickup's search", any(c_[0] == hs.LOOKALIKE_LOW for c_ in det_.calls)
      and h.low_conf is None)
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}), lookalike=False)
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("E3 OFF: the same kitchen fails not_detected (the hole Arash flagged)", (not r["ok"]) and r["cause"] == "not_detected"
      and "la_checks" not in sk.identity_stats, f"{r['ok']} {r['cause']}")
check("E3 OFF: the detector is never asked below the operating confidence", all(c_[0] == 0.25 for c_ in det_.calls))

print("5. a real Fork look-alike: picked up, the specialist says Fork -> put back where it was; then the real Spoon")
t, det_, h, sk = setup(kit(spoon=True, fork=True, spoon_kw={"prim": "Fork", "spec": "Spoon", "pos": (0.45, 0.97, 2.95)}))
fk = next(o for o in t.objs.values() if o["type"] == "Fork")
fk_pos0 = tuple(fk["pos"])
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
st_ = sk.identity_stats
recs = [x for x in sk.identity_log if x.get("kind") == "lookalike" and "seen" in x]
check("step ok with the Spoon in the hand", r["ok"] and held_type(t) == "Spoon", f"{r['cause']} {held_type(t)} {recs}")
check("the nearer real Fork was checked first: specialist says Fork -> put back (not dropped), landing seen",
      recs and recs[0]["seen"] == "Fork" and recs[0]["result"] == "put back (seen again near where it was)" and not recs[0]["kept"], str(recs))
check("the Fork is back on the counter, not on the floor, and not in the hand", fk.get("on") == "CounterTop" and fk["pos"][1] > 0.5
      and t.held != fk["id"], str(fk))
check("put_back counted and its landing seen, no restore failure; then the Spoon kept", st_["la_put_back"] == 1
      and st_["la_put_seen"] == 1 and st_["la_restore_failed"] == 0 and st_["la_kept"] == 1, str(st_))
check("the put-back object is not re-evaluated again", len(recs) == 2, str(recs))
check("checks within the cap", st_["la_checks"] <= hs.LOOKALIKE_MAX, str(st_))

print("6. the low-confidence trigger: a Spoon seen only at 0.15")
t, det_, h, sk = setup(kit(spoon_kw={"pconf": 0.15}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
recs = [x for x in sk.identity_log if x.get("kind") == "lookalike" and "seen" in x]
check("picked up as a low-confidence Spoon, the specialist confirms, kept", r["ok"] and held_type(t) == "Spoon"
      and recs and recs[0]["trigger"] == "low_conf", f"{r['cause']} {recs}")
check("memory never holds a sub-threshold box (no Spoon track before the pickup)", not any(
    tr.type == "Spoon" and tr.status == "trusted" for tr in sk.om.tracks.values()))
t, det_, h, sk = setup(kit(spoon_kw={"pconf": 0.15}), lookalike=False)
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("E3 OFF: not found", (not r["ok"]) and r["cause"] == "not_detected")

print("7. the cap: only Forks, no Spoon")
t, det_, h, sk = setup(kit(spoon=False, fork=True, n_forks=4))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
st_ = sk.identity_stats
check("fails honestly (not_detected), hand empty", (not r["ok"]) and r["cause"] == "not_detected" and t.held is None and h.holding is None,
      f"{r['cause']} {t.held} {h.holding}")
check(f"at most LOOKALIKE_MAX={hs.LOOKALIKE_MAX} checks, every one put back", 1 <= st_["la_checks"] <= hs.LOOKALIKE_MAX
      and st_["la_put_back"] == st_["la_checks"] - st_["la_refused"] and st_["la_restore_failed"] == 0, str(st_))
sent = [x["xyz"] for x in sk.identity_log if x.get("kind") == "lookalike" and "seen" in x]
check("no object re-evaluated twice (own-depth points > LOOKALIKE_SAME apart)",
      all(math.dist(a, b) > hs.LOOKALIKE_SAME for i, a in enumerate(sent) for b in sent[:i]), str(sent))
relooks_ = sum(x.get("relooks", 0) + x.get("relook_refresh", 0) for x in sk.identity_log if x.get("kind") == "lookalike")   # SPEC-INHAND fix B
check("every check costs the in-hand look plus the landing look, plus each counted fix-B re-look + its frame refresh (<= %d per check)"
      % len(hs.INSPECT_ROLLS),
      st_["la_detector_calls"] == 2 * (st_["la_checks"] - st_["la_refused"]) + relooks_
      and relooks_ <= (len(hs.INSPECT_ROLLS) + 1) * (st_["la_checks"] - st_["la_refused"]), str(st_))

print("8. scope")
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}))
r = sk.exec_step({"action": "search_for", "object": "Spoon"})
check("search_for does not pick anything up (no look-alike check)", sk.identity_stats["la_checks"] == 0 and t.held is None)
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}) + [dict(id="Bread|+0.60|+0.97|+2.60", type="Bread", pos=(0.7, 0.97, 2.6),
                                                                             size=0.2, pickupable=True, prim="Apple")])
r = sk.exec_step({"action": "pickup", "object": "Bread"})
check("a type without look-alikes (Bread) never triggers a check", sk.identity_stats["la_checks"] == 0)
t, det_, h, sk = setup(kit(spoon=False, fork=True))
r = sk.exec_step({"action": "pickup", "object": "Knife"})
check("Knife is excluded (no check)", sk.identity_stats["la_checks"] == 0 and t.held is None)
hs.LOOKALIKE_ON = False
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("module switch OFF: no check, identity_stats keys as before E3", "la_checks" not in sk.identity_stats or sk.identity_stats["la_checks"] == 0)
hs.LOOKALIKE_ON = True
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}), lookalike=False)
check("lookalike=False: identity_stats has exactly the E2 keys", set(sk.identity_stats) == {"checks", "detector_calls", "wrong", "unclear", "swaps", "extra_prims"})

print("10. Astra E3 review: the reproduced defects and the added rules")
# 10a the frozen development map
import hashlib as _hl                                                 # noqa: E402
import json as _js                                                    # noqa: E402
fz = _js.load(open(os.path.join(HERE, hs.LOOKALIKE_MAP_FILE.format(kitchen="FloorPlan1")), encoding="utf-8"))
check("the map file is per kitchen (schema, kitchen FloorPlan1) and lists every pair with its count and denominator",
      fz.get("schema") == "e3.lookalike_map/1" and fz.get("kitchen") == "FloorPlan1" and len(fz["pairs"]) == sum(len(v) for v in fz["map"].values())
      and all(p_["count"] >= 1 and p_["n_true"] >= p_["count"] and abs(p_["rate"] - round(p_["count"] / p_["n_true"], 4)) < 1e-9
              for p_ in fz["pairs"]), str(fz.get("pairs"))[:200])
srcp = os.path.join(HERE, hs.AMBIG_MAP_FILE.format(scene=hs.LOOKALIKE_MAP_SCENE))
check("the frozen M is FloorPlan1's map, built from the pinned source (sha256 matches)",
      fz["map"] == {k: dict(v) for k, v in m1.items()} and fz["source_sha256"] == _hl.sha256(open(srcp, "rb").read()).hexdigest()
      and fz["theta"] == hs.LOOKALIKE_THETA and fz["exclude"] == ["Knife"], str(fz.get("source_sha256")))
check("the frozen map records counts and denominators next to the rates", all("n" in v and "confused_with" in v for v in fz["counts"].values()))
t, det_, h, sk = setup(kit())
check("an executor built WITHOUT an explicit map uses the frozen map (not a per-kitchen file)", sk.amb_map == hs.load_frozen_map())
try:
    hs.load_frozen_map(os.path.join(HERE, "no_such_lookalike_map.json"))
    check("a missing frozen map is a configuration error (raised), never a silent {}", False)
except (OSError, RuntimeError):
    check("a missing frozen map is a configuration error (raised), never a silent {}", True)

# 10b positive confirmation (Astra's reproduced case: Spoon 0.50, Fork 0.64 -> E2 would say ok)
dep = np.full((100, 100), 2.0, "float32")
dep[80:97, 42:59] = 0.45


def hd(t, c, src="specialist"):
    return {"type": t, "conf": c, "bbox_xyxy": (42, 80, 58, 96), "point": (50, 88), "source": src}


check("E2 identity_verdict alone keeps Spoon 0.50 vs Fork 0.64 (why E3 needs its own rule)",
      hs.identity_verdict([hd("Spoon", 0.50), hd("Fork", 0.64)], "Spoon", dep)[0] == "ok")
check("E3 confirm: Spoon 0.50 vs Fork 0.64 -> NOT kept", not hs.lookalike_confirm([hd("Spoon", 0.50), hd("Fork", 0.64)], "Spoon", dep)[0])
check("E3 confirm: Spoon 0.70 vs Fork 0.60 (inside the margin) -> NOT kept",
      not hs.lookalike_confirm([hd("Spoon", 0.70), hd("Fork", 0.60)], "Spoon", dep)[0])
check("E3 confirm: Spoon 0.80 vs Fork 0.60 -> kept", hs.lookalike_confirm([hd("Spoon", 0.80), hd("Fork", 0.60)], "Spoon", dep)[0])
check("E3 confirm: a PRIMARY-sourced Spoon 0.95 at the hand -> NOT kept (specialist provenance required)",
      not hs.lookalike_confirm([hd("Spoon", 0.95, "primary")], "Spoon", dep)[0])
check("E3 confirm: Spoon 0.45 alone -> NOT kept (below LOOKALIKE_KEEP)", not hs.lookalike_confirm([hd("Spoon", 0.45)], "Spoon", dep)[0])
far = dict(hd("Spoon", 0.95), point=(10, 10), bbox_xyxy=(5, 5, 15, 15))
check("E3 confirm: a Spoon in the background (not at the hand) -> NOT kept", not hs.lookalike_confirm([far], "Spoon", dep)[0])
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon", "spec2": ("Fork", 0.85)}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
recs = [x for x in sk.identity_log if x.get("kind") == "lookalike" and "seen" in x]
check("live loop: specialist Spoon 0.90 with a Fork 0.85 rival at the hand -> not kept, restored",
      recs and not recs[0]["kept"] and recs[0]["rival"] == [0.85, "Fork"] and sk.identity_stats["la_kept"] == 0, str(recs))

# 10c aiming uses the candidate's own detection (copy vs identity)
o = O([dd("Fork", 0.6, (40, 40, 50, 60))])
c = hs.lookalike_candidates(o, "Spoon", mm)
check("each candidate keeps a reference to the very detection it came from", c and c[0]["_source_det"] is o.dets[0])
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("no candidate reference leaks into the logs", all("_source_det" not in x for x in sk.identity_log)
      and _js.dumps(sk.identity_log, default=str) is not None)
check("_aim_next is reset after a check", sk._aim_next is False)

# 10d memory lifecycle (Astra: a kept Spoon left a trusted Fork behind)
tr = list(sk.om.tracks.values())
check("after a kept check: the held track is a Spoon, no trusted Fork left at the old point",
      sk.om.held and sk.om.tracks[sk.om.held].type == "Spoon" and sk.om.tracks[sk.om.held].status == "held"
      and not any(x.type == "Fork" and x.status == "trusted" for x in tr), str([(x.type, x.status) for x in tr]))
check("exactly one held track (no second on_pickup)", sum(1 for x in tr if x.status == "held") == 1)
t, det_, h, sk = setup(kit(spoon=True, fork=True, spoon_kw={"prim": "Fork", "spec": "Spoon", "pos": (0.45, 0.97, 2.95)}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
tr = list(sk.om.tracks.values())
check("after a restored Fork: the Fork is not held in memory (placed or seen again), the Spoon is held",
      any(x.type == "Fork" and x.status in ("placed", "trusted") for x in tr) and not any(x.type == "Fork" and x.status == "held" for x in tr)
      and sk.om.tracks[sk.om.held].type == "Spoon", str([(x.type, x.status) for x in tr]))

# 10e interaction scope: a later pickup is not poisoned by an earlier one's checked points
t, det_, h, sk = setup(kit(spoon=False, fork=True, n_forks=2))
r1 = sk.exec_step({"action": "pickup", "object": "Spoon"})
n1 = sk.identity_stats["la_checks"]
r2 = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("a second pickup in the same episode checks again (points are interaction-local)", n1 >= 1 and sk.identity_stats["la_checks"] > n1,
      f"{n1} {sk.identity_stats['la_checks']}")

# 10f restoration failure: no drop, the object stays held, memory says so
t, det_, h, sk = setup(kit(spoon=False, fork=True))
t.refuse[("PutObject", "CounterTop")] = 99
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
st_ = sk.identity_stats
check("a failed restoration does NOT fail the step: the Fork is put down (here: every put refused -> dropped), the search "
      "goes on and ends honestly not_detected with the hand empty",
      (not r["ok"]) and r["cause"] == "not_detected" and t.held is None and h.holding is None
      and st_["la_restore_failed"] >= 1 and st_["la_put_elsewhere"] + st_["la_dropped"] == st_["la_restore_failed"], f"{r['cause']} {held_type(t)} {st_}")
check("... bounded: put attempts per check stay within LOOKALIKE_RESTORE_TRIES + 3 surfaces",
      sum(1 for c_ in t.calls if c_[0] == "PutObject") <= st_["la_checks"] * (hs.LOOKALIKE_RESTORE_TRIES + 3) * hs.AIM_PER_TRY)
check("... memory: nothing held, the Fork recorded as put down by the agent",
      sk.om.held is None and any(x.type == "Fork" and x.status in ("placed", "trusted") for x in sk.om.tracks.values()))

# 10g the revisit pass is bounded; pending points that become reachable are not suppressed
t, det_, h, sk = setup(kit(spoon_kw={"prim": "Fork", "spec": "Spoon"}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
st_ = sk.identity_stats
check("out-of-reach point queued, revisited once, then checked and kept (pending did not block it)",
      st_["la_unreached"] >= 1 and st_["la_revisits"] == 1 and st_["la_kept"] == 1 and st_["la_abandoned"] == 0, str(st_))
check("revisit navigation counted separately", st_["la_revisit_prims"] >= 1)

# 10h the low threshold does not change the ordinary detections
t = A.FakeThor(kit(spoon_kw={"pconf": 0.15}) + [dict(id="Bowl|+1.30|+0.97|+2.95", type="Bowl", pos=(1.3, 0.97, 2.95), size=0.2,
                                                     pickupable=True)], (4, 4), 0.0)
det2 = LabelDet(t)
h2 = honest.HonestAgent(t, det2, mode="honest")
a_ = h2.observe()
h2.low_conf = hs.LOOKALIKE_LOW
b_ = h2.observe()
key = lambda ds: sorted((d["type"], round(d["conf"], 3), tuple(d["bbox_xyxy"])) for d in ds)
check("Obs.dets with the low threshold == Obs.dets without it; Obs.low only below the operating confidence",
      key(a_.dets) == key(b_.dets) and b_.low and all(d["conf"] < h2.conf for d in b_.low) and not a_.low, f"{key(a_.dets)} {key(b_.dets)}")

# 10i F-dup: Astra's cases
print("   F-dup cases")


def obs_d(dets, step, depths=None):
    depth = np.full((100, 100), 1.0, "float32")
    for b, z in (depths or []):
        depth[b[1]:b[3] + 1, b[0]:b[2] + 1] = z
    pose = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0, "y": 0, "z": 0}, "horizon": 0.0,
            "camera": {"x": 0.0, "y": 1.5, "z": 0.0}}
    for d in dets:
        d.setdefault("state", "?")
    return honest.Obs(step=step, rgb=None, depth=depth, pose=pose, dets=dets)


omA = objmem.ObjectMemory("Fake", store=None, episode="f")
omA.update(obs_d([dd("Fork", 0.9, (40, 40, 52, 60)), dd("Fork", 0.8, (43, 40, 55, 60))], 1))
check("documented limit: two touching forks (IoU ~0.6, ~5 cm apart) merge into ONE track (a hypothesis, not proof)",
      len([x for x in omA.tracks.values() if x.type == "Fork"]) == 1)
omB = objmem.ObjectMemory("Fake", store=None, episode="f")
omB.update(obs_d([dd("Spoon", 0.6, (40, 40, 52, 60)), dd("Spoon", 0.5, (44, 40, 56, 60))], 1,
                 depths=[((48, 40, 56, 60), 1.4)]))
check("overlapping boxes at different depths (points > DUP_DIST apart) stay two tracks",
      len([x for x in omB.tracks.values() if x.type == "Spoon"]) == 2)
import itertools as _it                                               # noqa: E402
confs = []
for perm in _it.permutations([("Spoon", 0.48, (40, 40, 52, 60)), ("Spoon", 0.27, (41, 42, 53, 61)), ("Spoon", 0.27, (39, 41, 51, 59))]):
    omP = objmem.ObjectMemory("Fake", store=None, episode="p")
    omP.update(obs_d([dd(*x) for x in perm], 1))
    ts = [x for x in omP.tracks.values() if x.type == "Spoon"]
    confs.append((len(ts), round(ts[0].conf, 2) if ts else None))
check("input order does not matter: every permutation -> one track led by the 0.48 box (ties broken by index)",
      set(confs) == {(1, 0.48)}, str(confs))
omQ = objmem.ObjectMemory("Fake", store=None, episode="q")
omQ.update(obs_d([dd("Spoon", 0.48, (40, 40, 52, 60))], 1))
seenQ = omQ.update(obs_d([dd("Spoon", 0.30, (40, 40, 52, 60)), dd("Spoon", 0.90, (41, 41, 53, 61))], 2))
tq = [x for x in omQ.tracks.values() if x.type == "Spoon"]
check("existing association wins: the matched box keeps the track, the 0.90 duplicate joins it (one track, seen once more)",
      len(tq) == 1 and tq[0].seen == 2, str([(x.tid, x.conf, x.seen) for x in tq]))
dets_ = [dd("Spoon", 0.9, (40, 40, 52, 60)), dd("Spoon", 0.8, (44, 40, 56, 60)), dd("Spoon", 0.5, (42, 40, 54, 60))]
cand_ = [(0, dets_[0], (0.0, 1.0, 1.0)), (1, dets_[1], (0.06, 1.0, 1.0)), (2, dets_[2], (0.03, 1.0, 1.0))]
dup_ = objmem.ObjectMemory._frame_duplicates(dets_, cand_, {0: "tA", 1: "tB"}, set())
check("a box overlapping matched boxes of TWO different tracks is left ambiguous, not assigned to one",
      dup_.get(2) == ("ambig", None), str(dup_))
check("DUP_DIST is its own constant (0.10), separate from AMBIG_EPS", objmem.DUP_DIST == 0.10 and hasattr(objmem, "AMBIG_EPS"))

print("9. honesty")
t, det_, h, sk = setup(kit(spoon=True, fork=True, spoon_kw={"prim": "Fork", "spec": "Spoon"}))
r = sk.exec_step({"action": "pickup", "object": "Spoon"})
sm = h.audit.summary() if hasattr(h.audit, "summary") else {"violations": len(getattr(h.audit, "violations", [])), "counts": {}}
check("0 violations, 0 oracle reads", sm["violations"] == 0 and sm["counts"].get("oracle", 0) == 0, str(sm))
src = "".join(_ins.getsource(f) for f in (hs.ambiguous_objects_map, hs.lookalike_candidates, hs.HonestSkills._locate_la,
                                           hs.HonestSkills._la_here, hs.HonestSkills._revaluate, hs.HonestSkills._la_revisit,
                                           hs.HonestSkills._la_finish, hs.HonestSkills._la_restore, hs.HonestSkills._la_put_elsewhere, hs.lookalike_confirm,
                                           hs.freeze_ambiguous_map, hs.load_frozen_map, objmem.ObjectMemory._frame_duplicates,
                                           objmem.ObjectMemory.relabel_held))
check("E3 code names no simulator handle / metadata / object id", not any(w in src for w in ("_thor", "metadata", "objectId", "meta(", "event.")))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
