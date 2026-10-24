"""SPEC-INHAND-apply gate (Arash 2026-10-13: "project several points if needed (if one point failed) to make sure you can pick
up the object"). Offline, pure, Windows python. Two rules:
  A. the in-hand association never decides from ONE pixel: objmem.hand_depth needs a connected near patch of
     >= max(HAND_SUPPORT_MIN_PX, ceil(HAND_SUPPORT_SHARE x box area)) pixels and reads the near PERCENTILE of the whole box;
     the box's ground point alone decides nothing (either way).
  B. every pickup path aims at up to AIM_K object pixels and tries the next one before giving up, bounded and counted.
Usage (repo root): python sim/_step4_inhand_points_selftest.py"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honest  # noqa: E402
import honest_skills as hs  # noqa: E402
import objmem  # noqa: E402

P = F = 0


def check(name, cond, detail=""):
    global P, F
    P, F = P + bool(cond), F + (not cond)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ("" if cond else "  " + str(detail)))


def bg(d=1.6):
    return np.full((600, 600), d, dtype="float32")


def det(t, conf, box, depth, src="specialist", point=None):
    x = {"type": t, "conf": conf, "bbox_xyxy": box, "source": src}
    if point is None:
        x["point"], x["point_ok"] = honest.ground_point(depth, box)
    else:
        x["point"], x["point_ok"] = point, True
    return x


print("A. in-hand association: several points, never one")
BOX = (280, 350, 320, 550)                       # 41 x 201 px
AREA = 41 * 201
NEED = max(objmem.HAND_SUPPORT_MIN_PX, math.ceil(objmem.HAND_SUPPORT_SHARE * AREA))


def block(n_px, d0=0.4):
    """a compact connected near block of exactly n_px pixels inside BOX (rows of 20)"""
    d = bg()
    rows, cols = divmod(n_px, 20)
    d[400:400 + rows, 290:310] = d0
    if cols:
        d[400 + rows, 290:290 + cols] = d0
    return d


for n_px, label in ((1, "ONE near pixel"), (15, "15 near pixels"), (NEED - 1, f"{NEED - 1} near pixels (one short of {NEED})")):
    check(f"{label} in a 41x201 box -> NOT at the hand", objmem.hand_depth(block(n_px), BOX) is None,
          objmem.hand_depth(block(n_px), BOX))
check(f"a connected near patch of exactly {NEED} px -> supported (not None)", objmem.hand_depth(block(NEED), BOX) is not None)
check(f"a connected near patch of {2 * NEED} px -> at the hand, depth = the box's near percentile (0.40), not one pixel's",
      abs((objmem.hand_depth(block(2 * NEED), BOX) or 0) - 0.4) < 1e-6, objmem.hand_depth(block(2 * NEED), BOX))
d = bg()
ys, xs = np.mgrid[352:548:4, 282:319:4]
d[ys.ravel(), xs.ravel()] = 0.4                  # many near pixels, but scattered single dots (noise), none connected
check(f"{ys.size} scattered single near pixels (no connected patch) -> NOT at the hand", objmem.hand_depth(d, BOX) is None)
d = bg()
d[540:546, 297:303] = 0.4                        # a tiny near speck exactly where a ground point would land
x_ = det("Spoon", 0.93, BOX, d, point=(300, 543))
check("ground point ON a near speck but no supported patch -> identity 'unclear' (the point alone is not enough)",
      hs.identity_verdict([x_], "Spoon", d)[0] == "unclear", hs.identity_verdict([x_], "Spoon", d))
d = bg()
d[360:540, 297:303] = 0.41                       # a thin held spoon, 6 px wide
x_ = det("Spoon", 0.93, BOX, d, point=(285, 545))  # ground point on the background (1.6 m), as live
check("ground point on the background but a supported near patch -> identity 'ok' (the patch decides)",
      hs.identity_verdict([x_], "Spoon", d)[:2] == ("ok", "Spoon"), hs.identity_verdict([x_], "Spoon", d))
check("lookalike_confirm uses the same supported predicate (kept on the thin patch)",
      hs.lookalike_confirm([x_], "Spoon", d)[0] is True, hs.lookalike_confirm([x_], "Spoon", d))
check("objmem.at_hand == (hand_depth is not None) on the same box",
      objmem.at_hand(d, x_) is True and objmem.at_hand(bg(), x_) is False)
SRC = open(hs.__file__, encoding="utf-8").read()


def fn(name):
    i = SRC.index(f"def {name}(")
    return SRC[i:SRC.index("\ndef ", i + 5)]


check("identity_verdict / lookalike_confirm read hand_depth over the box, never _depth at the ground point",
      all('hand_depth(depth, d["bbox_xyxy"])' in fn(n) and '_depth(depth, d["point"])' not in fn(n)
          for n in ("identity_verdict", "lookalike_confirm")))
pick_src = SRC[SRC.index("    def _pick(self, obs, otype"):SRC.index("    def _point(self, obs, det")]
check("_pick (which box is the one in my hand) also accepts the supported several-pixel predicate",
      "objmem.at_hand(obs.depth, d)" in pick_src)
OM_SRC = open(objmem.__file__, encoding="utf-8").read()
check("objmem.update's held-object skip uses at_hand as well", "or at_hand(depth, d)" in OM_SRC)

print("B. pickup aim: up to AIM_K pixels, the next one before giving up, bounded and counted")


class H:
    def __init__(self, refuse_first=0, err="refused"):
        self.step, self.holding, self.sent, self.refuse_first, self.err = 0, None, [], refuse_first, err

    def act(self, prim, pt=None, **kw):
        self.step += 1
        self.sent.append((prim, tuple(pt) if pt is not None else None))
        if len(self.sent) <= self.refuse_first:
            return {"ok": False, "err": self.err}
        return {"ok": True, "err": ""}


class OM:
    fov = 90.0


def mk(**kw):
    sk = hs.HonestSkills.__new__(hs.HonestSkills)
    sk.h, sk.om, sk.exhaustive, sk.aim_protrude = H(**kw), OM(), False, True
    sk._aim_next, sk._aim_cap, sk.gate_log, sk._ev = False, None, [], (lambda m: None)
    sk.pick_stats = {"calls": 0, "sends": 0, "next_point": 0, "ok": 0, "offered_max": 0, "fallback_aims": 0}
    return sk


_rc0 = hs.reach_check
hs.reach_check = lambda depth, pose, fov, pt, limit, *a, **k: {"ok": True, "xyz": (0, 0, 0), "dist": 0.5, "why": ""}
try:
    flat = bg(0.9)                               # a flat surface: nothing protrudes -> the fallback path
    D = det("Spoon", 0.9, (250, 250, 330, 330), flat, src="primary")
    obs = honest.Obs(step=0, rgb=None, depth=flat, pose={}, dets=[D])
    sk = mk(refuse_first=1)
    st = {}
    pts = sk._aim_pickup(obs, D, [tuple(D["point"])], st)
    check("no protruding pixel -> the legacy point stays FIRST, then more box pixels (<= AIM_K extra), counted as a fallback aim",
          pts[0] == tuple(D["point"]) and 2 <= len(pts) <= hs.AIM_K + 1 and sk._aim_next and sk.pick_stats["fallback_aims"] == 1,
          (pts, sk.pick_stats))
    r = sk._act_at("PickupObject", obs, pts, "Spoon")
    check("the legacy pixel refused -> the NEXT pixel is tried in the same attempt and succeeds (was: stop after one pixel)",
          r["ok"] and len(sk.h.sent) == 2 and sk.h.sent[1][1] == pts[1] and sk.pick_stats["next_point"] == 1
          and sk.pick_stats["sends"] == 2 and sk.pick_stats["ok"] == 1, (r, sk.h.sent, sk.pick_stats))
    sk = mk(refuse_first=99)
    st = {}
    pts = sk._aim_pickup(obs, D, [tuple(D["point"])], st)
    r = sk._act_at("PickupObject", obs, pts, "Spoon")
    check(f"every pixel refused -> an ordinary attempt stops after AIM_PER_TRY={hs.AIM_PER_TRY} pixels (bounded)",
          not r["ok"] and len(sk.h.sent) == hs.AIM_PER_TRY, sk.h.sent)
    first = list(pts[1:hs.AIM_PER_TRY])
    sk._aim_next = False
    pts2 = sk._aim_pickup(obs, D, [tuple(D["point"])], st)
    check("the retry (next aim_round) rotates to the NEXT box pixels, not the ones the last attempt sent",
          st["aim_round"] == 1 and pts2[1] not in first, (first, pts2[:4]))
    sk = mk(refuse_first=99)
    sk._aim_next, sk._aim_cap = True, hs.AIM_K
    many = [(260 + 9 * i, 300) for i in range(10)]
    r = sk._act_at("PickupObject", obs, many, "Spoon")
    check(f"a one-call pickup (E3 candidate, _aim_cap=AIM_K) sends at most AIM_K={hs.AIM_K} pixels, each counted",
          len(sk.h.sent) == hs.AIM_K and sk.pick_stats["sends"] == hs.AIM_K and sk.pick_stats["next_point"] == hs.AIM_K - 1,
          (len(sk.h.sent), sk.pick_stats))
    sk = mk(refuse_first=99)
    sk._aim_next, sk._aim_cap = True, hs.AIM_K
    r = sk._act_at("OpenObject", obs, many, "Cabinet")
    check("the AIM_K cap is for PickupObject only (an open keeps its own cap) and opens are not counted as pickups",
          len(sk.h.sent) <= 3 and sk.pick_stats["calls"] == 0, (len(sk.h.sent), sk.pick_stats))
    sk = mk(refuse_first=0, err="")
    sk._aim_next = True
    r = sk._act_at("PickupObject", obs, many, "Spoon")
    check("a first pixel that works sends exactly one pickup (no extra cost when nothing fails)",
          r["ok"] and len(sk.h.sent) == 1 and sk.pick_stats["next_point"] == 0, sk.h.sent)
    d = bg(0.9)
    d[280:300, 285:295] = 0.85                   # a small object standing out of the surface
    D2 = det("ButterKnife", 0.9, (280, 270, 296, 302), d, src="primary")
    obs2 = honest.Obs(step=0, rgb=None, depth=d, pose={}, dets=[D2])
    sk = mk()
    pts = sk._aim_pickup(obs2, D2, [tuple(D2["point"])], {})
    check("an object that protrudes -> E1 aimed pixels (>1 object pixel), no fallback counted",
          len(pts) > 1 and sk.pick_stats["fallback_aims"] == 0 and sk._aim_next, (pts, sk.pick_stats))
    hs.PICK_MULTI_ON = False
    sk = mk()
    pts = sk._aim_pickup(obs, D, [tuple(D["point"])], {})
    check("PICK_MULTI_ON=False restores the old single-pixel fallback exactly", pts == [tuple(D["point"])] and not sk._aim_next, pts)
    hs.PICK_MULTI_ON = True
finally:
    hs.reach_check = _rc0

sk = mk()
seen = {}


def _fake_aim(obs, d, pts, st):
    seen["aim"] = True
    return [tuple(pts[0]), (1, 1)]


def _fake_act_at(prim, obs, pts, ot, **kw):
    seen["cap"], seen["pts"] = sk._aim_cap, list(pts)
    return {"ok": False, "err": "gate", "sent": False, "why": "no point to act at", "pt": None}


sk._aim_pickup, sk._act_at = _fake_aim, _fake_act_at
sk._la = {"otype": "Spoon", "n": 0, "pending": [], "checked": []}
sk.identity_stats = {"la_unreached": 0, "la_prims": 0}
sk.identity_log = []
cand = {"type": "Fork", "point": (300, 300), "bbox_xyxy": (290, 290, 310, 310), "conf": 0.4, "trigger": "low"}
sk._revaluate(honest.Obs(step=0, rgb=None, depth=bg(), pose={}, dets=[cand]), "Spoon", cand, (0.0, 0.9, 1.0))
check("E3 candidate pickup: E1/fallback aim used and the call may send up to AIM_K pixels; cap cleared afterwards",
      seen.get("aim") and seen.get("cap") == hs.AIM_K and sk._aim_cap is None, seen)
check("no pickup bypasses the aim/count path (no literal h.act('PickupObject') in honest_skills)",
      not re.search(r"h\.act\(\s*[\"']PickupObject", SRC))
calls = re.findall(r"_interact\(\"pickup\"", SRC)
check(f"pick_and_place / purpose / put-check pickups go through _interact('pickup') -> _attempt -> _aim_pickup ({len(calls)} sites)",
      len(calls) >= 5 and "pts = self._aim_pickup(obs, det, pts, st)" in SRC)
check("the identity swap retry is a retryable wrong_target (the next attempt aims at the next aim_round pixels)",
      'raise _Soft("wrong_target", f"picked up a {seen} instead of the {otype}' in SRC)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
