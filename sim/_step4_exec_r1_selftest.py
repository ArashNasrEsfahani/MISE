"""Offline selftest for Round 2 / J-R1-executor (sim/honest_skills.py). Windows `python`, no WSL/GPU/Ollama/LLM.

Run from the repo root:  python sim/_step4_exec_r1_selftest.py

Gates the general executor upgrades that apply to ALL arms, plus the one flag that must not:
  E1  aim at what PROTRUDES from the support surface (thin object, adjacent pair, large object unchanged, no-ring fallback),
      pixel-diverse retries, a refused pickup goes on to the next pixel at no camera move;
  E2  the costed identity check of the object in the hand (right / wrong / unclear / not a specialist class), the bounded
      swap-back, the counters (detector calls, extra primitives, swaps);
  E3  an object seen inside a container while it stood open is a lead for the next search;
  P   purpose-based pickup through the declared affordance map (default OFF, substitution recorded);
  B   briefing search-order hints: refused loudly by an executor not built for a +package arm; an ORDER only (wrong hint still finds);
  M   the executor regression matrix of Astra's round-2 review, offline (see section 9 for which row is covered by which test).
The legacy behaviour is pinned by sim/_step4_act_selftest.py (it switches the new aim and identity check off).
"""
from __future__ import annotations

import contextlib
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
with contextlib.redirect_stdout(_buf):                                # reuse the fake kitchen of the act gate (its own checks stay quiet)
    import _step4_act_selftest as A
import honest_skills as hs                                            # noqa: E402
import objmem                                                         # noqa: E402

hs.AIM_PROTRUDE_ON = True
hs.IDENT_ON = True
check("the act gate's fake kitchen imported clean (all its checks passed)", A.FAIL == 0, f"{A.FAIL} failing")


# ----------------------------------------------------------------------------------------------- synthetic depth
def plane(h=300, w=300, a=0.00012, b=0.00045, c=0.55):
    """A tilted support surface: 1/z = a x + b y + c (z in metres, about 1.0 to 1.7 m)."""
    ys, xs = np.mgrid[0:h, 0:w]
    return (1.0 / (a * xs + b * ys + c)).astype("float32")


def paint(d, mask, rel=0.025):
    d = d.copy()
    d[mask] = d[mask] * (1.0 - rel)
    return d


def line_mask(shape, p0, p1, width=3):
    m = np.zeros(shape, bool)
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) * 2)
    for i in range(n + 1):
        x = p0[0] + (p1[0] - p0[0]) * i / max(n, 1)
        y = p0[1] + (p1[1] - p0[1]) * i / max(n, 1)
        m[int(round(y)) - width // 2:int(round(y)) + width // 2 + 1, int(round(x)) - width // 2:int(round(x)) + width // 2 + 1] = True
    return m


def rect_mask(shape, b):
    m = np.zeros(shape, bool)
    m[b[1]:b[3] + 1, b[0]:b[2] + 1] = True
    return m


# ======================================================================================================== 1
print("1. E1 protruding_points: a thin diagonal object whose box centre is on the surface behind it")
for tag, (p0, p1), tilt in (("pose A", ((120, 150), (170, 118)), dict()), ("pose B", ((100, 100), (150, 160)), dict(a=0.0002, b=0.0003)),
                            ("pose C", ((200, 90), (250, 95)), dict(a=-0.0001, b=0.0006, c=0.5))):
    d0 = plane(**tilt)
    bar = line_mask(d0.shape, p0, p1, 3)
    d = paint(d0, bar)
    # the detector's box is a little loose on one side (as measured live: the box held the blade AND the counter beside it),
    # so its centre falls beside the bar
    x0_, y0_, x1_, y1_ = min(p0[0], p1[0]) - 2, min(p0[1], p1[1]) - 2, max(p0[0], p1[0]) + 2, max(p0[1], p1[1]) + 2
    box = next(b for b in ((x0_, y0_, x1_ + 30, y1_ + 30), (x0_ - 30, y0_, x1_, y1_ + 30), (x0_, y0_ - 30, x1_ + 30, y1_),
                           (x0_ - 30, y0_ - 30, x1_, y1_))
               if not bar[int(round((b[1] + b[3]) / 2)), int(round((b[0] + b[2]) / 2))])
    cen = (int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2)))
    pts = hs.protruding_points(d, box, first=cen)
    on = [bool(bar[q[1], q[0]]) for q in pts]
    check(f"{tag}: the box centre is NOT on the object (the legacy aim would miss)", not bar[cen[1], cen[0]])
    check(f"{tag}: every proposed pixel is on the object ({sum(on)}/{len(on)})", len(pts) >= 3 and all(on), f"{pts}")
    check(f"{tag}: proposed pixels are separated", all(math.hypot(a[0] - b[0], a[1] - b[1]) >= 3 for i, a in enumerate(pts) for b in pts[:i]))
    check(f"{tag}: deterministic", pts == hs.protruding_points(d, box, first=cen))

print("2. E1 protruding_points: an adjacent pair, the neighbour stands in front of the lower part of the target")
d0 = plane()
salt_b, pep_b = (140, 100, 158, 142), (139, 112, 160, 170)             # the pepper's box overlaps the salt's lower two thirds
salt = rect_mask(d0.shape, salt_b)
pep = rect_mask(d0.shape, pep_b)
d = paint(paint(d0, salt, 0.03), pep, 0.06)                            # pepper nearer than the salt where they overlap
cen = (149, 121)
check("setup: the salt's box centre lies under the pepper (it resolves to the neighbour)", bool(pep[cen[1], cen[0]]))
pts = hs.protruding_points(d, salt_b, avoid=[pep_b], first=cen)
check("avoiding the neighbour's box: proposed pixels are on the salt only", len(pts) >= 3 and all(salt[q[1], q[0]] and not pep[q[1], q[0]] for q in pts), f"{pts}")
pts_no = hs.protruding_points(d, salt_b, avoid=[], first=cen)
check("without the avoid list the neighbour's pixels would be proposed (the avoid list is what protects the aim)",
      any(pep[q[1], q[0]] for q in pts_no))
check("both orders: the pepper's own aim avoids the salt's box",
      all(pep[q[1], q[0]] for q in hs.protruding_points(d, pep_b, avoid=[salt_b], first=(149, 152))))

print("3. E1 protruding_points: a large object keeps its legacy aim; flat / no ring -> [] (legacy points kept)")
d0 = plane()
apple_b = (120, 120, 170, 170)
yy, xx = np.mgrid[0:300, 0:300]
disc = (xx - 145) ** 2 + (yy - 145) ** 2 <= 25 ** 2
d = paint(d0, disc, 0.12)
pts = hs.protruding_points(d, apple_b, first=(145, 145))
check("large object: the legacy centre point goes first, unchanged", pts[:1] == [(145, 145)], f"{pts[:2]}")
check("flat box with nothing standing out -> []", hs.protruding_points(d0, apple_b, first=(145, 145)) == [])
check("no ring (box fills the frame) -> []", hs.protruding_points(np.full((40, 40), 1.0, "float32"), (0, 0, 39, 39)) == [])
check("None depth / None box -> []", hs.protruding_points(None, apple_b) == [] and hs.protruding_points(d, None) == [])
check("NaN depth does not raise", hs.protruding_points(np.where(disc, np.nan, d).astype("float32"), apple_b) is not None)

print("4. E2 identity_verdict")
D = np.full((100, 100), 2.0, "float32")
D[80:97, 42:59] = 0.45
hand = (42, 80, 58, 96)
far = (10, 10, 30, 30)


def det(t, conf, bbox=hand, pt=(50, 88)):
    return {"type": t, "conf": conf, "bbox_xyxy": bbox, "point": pt}


check("the asked type in the hand -> ok", hs.identity_verdict([det("SaltShaker", .9)], "SaltShaker", D)[0] == "ok")
v = hs.identity_verdict([det("PepperShaker", .9)], "SaltShaker", D)
check("another specialist-class type, confident -> wrong (and which)", v[0] == "wrong" and v[1] == "PepperShaker")
check("another type, below the confidence floor -> unclear (doubt keeps the pickup)",
      hs.identity_verdict([det("PepperShaker", .40)], "SaltShaker", D)[0] == "unclear")
check("both seen, the asked one within the margin -> ok",
      hs.identity_verdict([det("PepperShaker", .80), det("SaltShaker", .70)], "SaltShaker", D)[0] == "ok")
check("both seen, the asked one far below -> wrong",
      hs.identity_verdict([det("PepperShaker", .90), det("SaltShaker", .55)], "SaltShaker", D)[0] == "wrong")
D2 = D.copy()
D2[60:79, 10:31] = 0.70                                                # a pepper shaker standing 0.70 m away, left of the hand
check("a nearer thing is the held one: a farther (0.70 m vs 0.45 m) specialist-class detection does not condemn the pickup",
      hs.identity_verdict([det("SaltShaker", .6), det("PepperShaker", .95, (10, 60, 30, 78), (20, 70))], "SaltShaker", D2)[0] == "ok")
D3 = D.copy()
D3[80:97, 60:77] = 0.46
check("...but a pepper at the SAME depth as the held thing is a real conflict (wrong)",
      hs.identity_verdict([det("SaltShaker", .6), det("PepperShaker", .95, (60, 80, 76, 96), (68, 88))], "SaltShaker", D3)[0] == "wrong")
check("nothing near the hand -> unclear", hs.identity_verdict([det("PepperShaker", .9, far, (20, 20))], "SaltShaker", D)[0] == "unclear")
check("a surface near the camera never votes", hs.identity_verdict([det("CounterTop", .99)], "SaltShaker", D)[0] == "unclear")
check("a non-specialist type in the hand does not condemn the pickup", hs.identity_verdict([det("Apple", .99)], "SaltShaker", D)[0] == "unclear")
check("no detections / no depth -> unclear, no exception", hs.identity_verdict([], "Fork", D)[0] == "unclear"
      and hs.identity_verdict([det("Fork", .9)], "Fork", None)[0] == "unclear")


# ======================================================================================================== 5
def shakers():
    return [dict(id="CounterTop|+1.20|+0.90|+2.95", type="CounterTop", pos=(1.2, 0.9, 2.95), size=1.0, receptacle=True),
            dict(id="SaltShaker|+1.20|+0.97|+2.95", type="SaltShaker", pos=(1.15, 0.97, 2.95), size=0.2, pickupable=True),
            dict(id="PepperShaker|+1.30|+0.97|+2.95", type="PepperShaker", pos=(1.35, 0.97, 2.95), size=0.2, pickupable=True),
            dict(id="Apple|+0.50|+0.97|+2.60", type="Apple", pos=(0.5, 0.97, 2.6), size=0.2, pickupable=True)]


print("5. E2 in the loop: salt asked, pepper resolved -> put back, tried again at other pixels, salt in the hand")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
t.misresolve["SaltShaker"] = "PepperShaker"
r = sk.exec_step({"action": "pickup", "object": "SaltShaker"})
check("the step ends ok with the SALT in the hand (the simulator's own state, read by the test only)",
      r["ok"] and t.held and t.objs[t.held]["type"] == "SaltShaker", f"{r['cause']} {t.held}")
check("one swap-back, one detector look per check, two checks", sk.identity_stats["swaps"] == 1 and sk.identity_stats["checks"] == 2
      and sk.identity_stats["detector_calls"] == 2, str(sk.identity_stats))
check("the extra actions are counted (the put-down)", sk.identity_stats["extra_prims"] == 1, str(sk.identity_stats))
check("the step detail carries the identity records (wrong then ok)", [x["verdict"] for x in r["detail"].get("identity", [])] == ["wrong", "ok"])
check("the swap cost one of the 3 retries", r["retries"] == 1)
gp = [c[1] for c in t.calls if c[0] == "GetObjectInFrame"]
check("the second try aimed at DIFFERENT pixels than the first", len(gp) == 2 and (gp[0]["x"], gp[0]["y"]) != (gp[1]["x"], gp[1]["y"]), str(gp))
check("no object id in the result", A.no_ids(r))

print("5b. E2: a right pickup costs one look and no swap; a non-specialist type costs nothing")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
n0 = len(t.calls)
r = sk.exec_step({"action": "pickup", "object": "SaltShaker"})
check("salt: ok, 1 check, 0 swaps, 0 extra primitives", r["ok"] and sk.identity_stats["checks"] == 1 and sk.identity_stats["swaps"] == 0
      and sk.identity_stats["extra_prims"] == 0, str(sk.identity_stats))
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
r = sk.exec_step({"action": "pickup", "object": "Apple"})
check("apple (not a specialist class): no identity check at all", r["ok"] and sk.identity_stats["checks"] == 0 and "identity" not in r["detail"])

print("5c. E2: bounded -- a persistent mix-up swaps at most twice, then the step fails honestly with an empty hand")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True, max_retries=10)
orig = t.do


def always_pepper(action, **kw):
    if action == "GetObjectInFrame":
        t.misresolve["SaltShaker"] = "PepperShaker"
    return orig(action, **kw)


t.do = always_pepper
r = sk.exec_step({"action": "pickup", "object": "SaltShaker"})
check("fails with wrong_target, not ok", (not r["ok"]) and r["cause"] == "wrong_target", f"{r['ok']} {r['cause']}")
check("exactly IDENT_SWAPS_MAX + 1 look-and-put-down cycles, then the hand is empty", sk.identity_stats["swaps"] == hs.IDENT_SWAPS_MAX + 1
      and t.held is None and h.holding is None, f"{sk.identity_stats} {t.held} {h.holding}")

print("5d. E2: the doubt rule -- nothing readable near the hand keeps the pickup")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
real_detect = det_.detect
det_.detect = lambda rgb, conf=0.25, inspecting=None: [d for d in real_detect(rgb, conf) if not (d["type"] in ("SaltShaker", "PepperShaker") and d["bbox_xyxy"][1] >= 78)]
r = sk.exec_step({"action": "pickup", "object": "SaltShaker"})
check("held object unreadable -> unclear -> kept (no false rejection)", r["ok"] and sk.identity_stats["unclear"] == 1 and sk.identity_stats["swaps"] == 0, str(sk.identity_stats))

print("5e. switches: identity check OFF -> the mix-up passes silently (the legacy hole), aim OFF -> legacy single pixel")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True, identity_check=False)
t.misresolve["SaltShaker"] = "PepperShaker"
r = sk.exec_step({"action": "pickup", "object": "SaltShaker"})
check("without the check the PEPPER ends up in the hand while the step says ok", r["ok"] and t.objs[t.held]["type"] == "PepperShaker")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True, aim_protrude=False, identity_check=False)
t.refuse[("PickupObject", "Apple")] = 2
r = sk.exec_step({"action": "pickup", "object": "Apple"})
check("aim OFF: each refusal is one retry (legacy)", r["ok"] and r["retries"] == 2, f"{r['retries']}")

# ======================================================================================================== 6
print("6. E1 in the loop: a refused pickup tries the next pixel with no camera move; the next attempt starts further along")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
t.refuse[("PickupObject", "Apple")] = 2
r = sk.exec_step({"action": "pickup", "object": "Apple"})
pk = [c for c in t.calls if c[0] == "PickupObject"]
check("2 refusals then ok inside ONE attempt (no retry spent, no camera move between)", r["ok"] and r["retries"] == 0 and len(pk) == 3, f"{r['retries']} {len(pk)}")
gp = [(c[1]["x"], c[1]["y"]) for c in t.calls if c[0] == "GetObjectInFrame"]
check("the three pixels sent were three different pixels", len(set(gp)) == 3, str(gp))
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
t.refuse[("PickupObject", "Apple")] = 4
r = sk.exec_step({"action": "pickup", "object": "Apple"})
gp = [(c[1]["x"], c[1]["y"]) for c in t.calls if c[0] == "GetObjectInFrame"]
check("a persistent refusal: the second attempt did NOT repeat the first attempt's three pixels",
      r["ok"] and len(gp) == 5 and not (set(gp[:3]) & set(gp[3:])), str(gp))
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
t.refuse[("PickupObject", "Apple")] = 99
r = sk.exec_step({"action": "pickup", "object": "Apple"})
check("visible and detected but never interactable: fails as retry_exhausted / action_refused, NOT as not_detected",
      (not r["ok"]) and r["cause"] in ("retry_exhausted", "action_refused"), f"{r['cause']}")
check("the primitive cost of that failure is bounded", r["detail"]["prims"] <= hs.STEP_PRIM_CAP, f"{r['detail']['prims']}")

# ======================================================================================================== 7
print("7. E3: what was seen inside a container while it stood open is a lead for the next search")


CAB2 = "Cabinet|-0.55|+0.60|+2.60"


def cutlery():
    objs = [o for o in A.kitchen() if o["type"] != "Apple"]
    objs.append(dict(id=CAB2, type="Cabinet", pos=(-0.55, 0.6, 2.6), size=0.5, openable=True, receptacle=True, isOpen=False))
    objs.append(dict(id="Fork|-0.40|+0.60|+2.60", type="Fork", pos=(-0.40, 0.6, 2.6), size=0.10, pickupable=True, inside=CAB2))
    objs.append(dict(id="Spoon|-0.70|+0.60|+2.60", type="Spoon", pos=(-0.70, 0.6, 2.6), size=0.10, pickupable=True, inside=CAB2))
    return objs


EXH = dict(search="exhaustive", tidy=True)


t, det_, h, sk = A.setup(objs=cutlery(), chooser=lambda *a: [], **EXH)
r = sk.exec_step({"action": "pickup", "object": "Fork"})
cab = [x for x in sk.inspected if x["type"] == "Cabinet"]
check("the Fork was found inside the cabinet", r["ok"] and h.holding == "Fork", f"{r['ok']} {r['cause']}")
seen = cab[0].get("seen_types", []) if cab else []
check("the cabinet's record lists BOTH movable types seen inside while it stood open", "Fork" in seen and "Spoon" in seen, str(seen))
check("surfaces and containers are not listed as seen inside", not ({"Cabinet", "Fridge", "CounterTop"} & set(seen)))
# the spoon's sighting is lost (as the diagnosis showed: contradicted by views of the closed front)
for tid in [k for k, v in sk.om.tracks.items() if v.type == "Spoon"]:
    del sk.om.tracks[tid]
sk.exec_step({"action": "drop"})
r2 = sk.exec_step({"action": "pickup", "object": "Spoon"})
ev = " ".join(r2["detail"]["events"])
check("search for the Spoon goes to that cabinet FIRST ('seen inside it before')", "seen inside it before" in ev, ev[:300])
check("and the Spoon is picked up", r2["ok"] and h.holding == "Spoon", f"{r2['cause']}")
check("the cabinet is the only container opened for it (no walk through the others)", sum(1 for o in t.opened_log) <= 3, str(t.opened_log))

print("7b. E3 not fooled: the object really left -> the lead costs one inspection and the search goes on")
t, det_, h, sk = A.setup(objs=cutlery(), chooser=lambda *a: [], **EXH)
sk.exec_step({"action": "pickup", "object": "Fork"})
for tid in [k for k, v in sk.om.tracks.items() if v.type == "Spoon"]:
    del sk.om.tracks[tid]
t.objs["Spoon|-0.70|+0.60|+2.60"]["pos"] = (1.0, 0.97, 2.95)           # moved out onto the counter while nobody looked
t.objs["Spoon|-0.70|+0.60|+2.60"].pop("inside")
sk.exec_step({"action": "drop"})
r3 = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("the Spoon on the counter is still found after the stale lead", r3["ok"] and h.holding == "Spoon", f"{r3['cause']} {r3['detail'].get('why')}")

# ======================================================================================================== 8
print("8. B: briefing hints are an ORDER for +package arms only")
try:
    A.setup(objs=cutlery(), search="exhaustive", briefing_hints=lambda o: ["Cabinet"])
    refused = False
except ValueError as e:
    refused = "+package" in str(e)
check("an executor built WITHOUT allow_briefing_hints refuses a hint hook (a no-memory arm cannot receive one)", refused)
t, det_, h, sk = A.setup(objs=cutlery(), chooser=lambda *a: [], **EXH)
check("no hook: the briefing log stays empty and nothing is reordered", sk.briefing_hints is None and sk.briefing_log == []
      and sk._briefed("Fork", ["Fridge"], {}) == ["Fridge"])
t, det_, h, sk = A.setup(objs=cutlery(), chooser=lambda *a: [], search="exhaustive", tidy=True,
                         briefing_hints=lambda o: ["Cabinet", "Cabinet", "NoSuchType", "Fork"], allow_briefing_hints=True)
info = {}
check("with the hook: container types first, de-duplicated, unknown / self types dropped, the rest keep their order",
      sk._briefed("Fork", ["Fridge", "Drawer"], info) == ["Cabinet", "Fridge", "Drawer"] and info.get("briefing") == ["Cabinet"],
      str(sk._briefed("Fork", ["Fridge", "Drawer"], {})))
check("the audit log says where the briefing entered", sk.briefing_log and sk.briefing_log[0]["otype"] == "Fork")
t, det_, h, sk = A.setup(objs=cutlery(), search="exhaustive", tidy=True, chooser=lambda *a: ["Fridge", "Microwave", "Cabinet"],
                         briefing_hints=lambda o: ["Cabinet"], allow_briefing_hints=True)
r = sk.exec_step({"action": "pickup", "object": "Fork"})
first_open = t.opened_log[0][1] if t.opened_log else None
check("a CORRECT hint: a cabinet is opened first although the chooser put the fridge and microwave first", r["ok"] and first_open and first_open.startswith("Cabinet"), str(t.opened_log))
t, det_, h, sk = A.setup(objs=cutlery(), search="exhaustive", tidy=True, chooser=lambda *a: ["Fridge", "Microwave", "Cabinet"])
sk.exec_step({"action": "pickup", "object": "Fork"})
check("the same search WITHOUT the hint opens the fridge or microwave first (so the hint is what changed the order)",
      t.opened_log and not t.opened_log[0][1].startswith("Cabinet"), str(t.opened_log))
t, det_, h, sk = A.setup(objs=cutlery(), search="exhaustive", tidy=True, chooser=lambda *a: ["Cabinet"],
                         briefing_hints=lambda o: ["Microwave"], allow_briefing_hints=True)
r = sk.exec_step({"action": "pickup", "object": "Fork"})
check("a WRONG hint (microwave): the fork is still found (the hint only reorders)", r["ok"] and h.holding == "Fork", f"{r['cause']}")
t, det_, h, sk = A.setup(objs=cutlery(), search="exhaustive", tidy=True, chooser=lambda *a: ["Cabinet"],
                         briefing_hints=lambda o: 1 / 0, allow_briefing_hints=True)
r = sk.exec_step({"action": "pickup", "object": "Fork"})
check("a failing hint hook never breaks the search (error recorded)", r["ok"] and sk.event_errors and sk.event_errors[0]["kind"] == "briefing_hints")

# ======================================================================================================== 9
print("9. P: purpose-based pickup (default OFF)")


def knives():
    return [dict(id="CounterTop|+1.20|+0.90|+2.95", type="CounterTop", pos=(1.2, 0.9, 2.95), size=1.0, receptacle=True),
            dict(id="ButterKnife|+1.20|+0.97|+2.95", type="ButterKnife", pos=(1.2, 0.97, 2.95), size=0.2, pickupable=True)]


t, det_, h, sk = A.setup(objs=knives(), search="exhaustive", tidy=True)
r = sk.exec_step({"action": "pickup", "object": "Knife", "purpose": "cut"})
check("flag OFF: the purpose key is ignored; no Knife -> not found, nothing substituted", (not r["ok"]) and r["cause"] == "not_detected" and h.holding is None, f"{r['cause']}")
t, det_, h, sk = A.setup(objs=knives(), search="exhaustive", tidy=True, purpose_search=True)
r = sk.exec_step({"action": "pickup", "object": "Knife", "purpose": "cut"})
check("flag ON + purpose cut: the ButterKnife that was SEEN is taken and the substitution is recorded",
      r["ok"] and h.holding == "ButterKnife" and r["detail"].get("substituted") == {"asked": "Knife", "got": "ButterKnife", "purpose": "cut"}, str(r["detail"].get("substituted")))
t, det_, h, sk = A.setup(objs=knives(), search="exhaustive", tidy=True, purpose_search=True)
r = sk.exec_step({"action": "pickup", "object": "Spatula", "purpose": "flip"})
check("flag ON + purpose flip: nothing of that family exists -> honest not_detected", (not r["ok"]) and r["cause"] == "not_detected")
t, det_, h, sk = A.setup(objs=knives(), search="exhaustive", tidy=True, purpose_search=True)
r = sk.exec_step({"action": "pickup", "object": "Knife"})
check("flag ON but the step states no purpose: no substitution", (not r["ok"]) and h.holding is None)
# A26 (K-B2b 2026-10-13): cut / slice -> ("ButterKnife",) (was ("Knife", "ButterKnife")); purpose substitution stays OFF in every arm
check("the affordance map is declared, English, and has no scene fact", hs.PURPOSE_TYPES["cut"] == ("ButterKnife",) and hs.PURPOSE_TYPES["flip"] == ("Spatula",))

print("10. M: the executor regression matrix (Astra round 2) -- where each row is gated")
rows = {
    "closed drawer / door opens, repositions, detects inside": "act gate sections 7, 11-17 (search inspections) + sections 5 and 7 here",
    "open container (already open)": "act gate section 3 (already in the wanted state) + section 7 here (Fork found after the open)",
    "low / high cabinet (crouch, stand-up)": "act gate low_kitchen / far-open sections",
    "occlusion (a neighbour covers the object)": "section 2 here (avoid list) + 5 (salt/pepper)",
    "empty container": "section 7b here (stale lead) + act gate section 7 (not found -> not_detected)",
    "wrong hints": "section 8 here",
    "unreachable": "act gate section 9 / 17",
    "exhausted budget": "act gate section 4 / 7 + section 6 here (cost bounded)",
    "visibility vs interactability": "section 6 here (visible, detected, never interactable -> action_refused / retry_exhausted)",
}
check("every Astra matrix row names the test that gates it", len(rows) == 9 and all(rows.values()))
print("   matrix rows:")
for k, v in rows.items():
    print(f"     - {k}: {v}")

print("10b. J-regress: an attempt always sends the legacy centre point among its first AIM_PER_TRY pixels")
import types as _types
_dd = plane()
_bx = (120, 120, 180, 180)
_cup = np.zeros(_dd.shape, bool); _cup[135:175, 130:170] = True
_dd = paint(_dd, _cup, rel=0.05)
_self = _types.SimpleNamespace(aim_protrude=True, _aim_next=False, _ev=lambda *a, **k: None)
_obs = _types.SimpleNamespace(depth=_dd, dets=[])
_legacy = (150, 150)
_out = hs.HonestSkills._aim_pickup(_self, _obs, {"type": "Cup", "bbox_xyxy": _bx, "point": _legacy}, [_legacy], {})
check("aim found pixels on the object", _self._aim_next is True and len(_out) >= 2, f"{_out}")
check("the legacy point is inside the first AIM_PER_TRY sent", _legacy in _out[:hs.AIM_PER_TRY], f"{_out[:hs.AIM_PER_TRY]}")
check("no point is dropped or duplicated", len(set(_out)) == len(_out), f"{_out}")

print("11. honesty: no simulator reads in the new code, and the audit is clean")
t, det_, h, sk = A.setup(objs=shakers(), search="exhaustive", tidy=True)
t.misresolve["SaltShaker"] = "PepperShaker"
sk.exec_step({"action": "pickup", "object": "SaltShaker"})
sm = h.audit.summary()
check("0 violations, 0 oracle reads after the identity swap run", sm["violations"] == 0 and sm["counts"].get("oracle", 0) == 0, str(sm))
import inspect as _ins
src = "".join(_ins.getsource(f) for f in (hs.protruding_points, hs.identity_verdict, hs.HonestSkills._aim_pickup,
                                           hs.HonestSkills._identity_check, hs.HonestSkills._briefed, hs.HonestSkills._note_inside,
                                           hs.HonestSkills._pickup_for_purpose))
check("the new code never touches the metadata gateway", "meta(" not in src and "metadata" not in src and "objectId" not in src and "_thor" not in src)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("STEP4 EXEC R1 SELFTEST OK")
