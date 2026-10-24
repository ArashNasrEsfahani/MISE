"""Self-test for the oracle-verified detection benchmark (`sim/oracle_eval.py`).

    python sim/_oracle_eval_selftest.py        # from the REPO ROOT

Headless: no ai2thor, no torch, no ultralytics, no GPU. The scoring core is pure, so the arithmetic is checked
against hand-computed values rather than against itself; `ground_truth_boxes` and `evaluate_live` are driven
with a fake agent that reproduces the surface AI2-THOR actually exposes (instance masks + visible_objects +
teleport_full + detect_objects).

The properties that matter, and why:

  * A greedy one-to-one assignment. Two predictions on one object must be a hit and a DUPLICATE, not two hits;
    getting that wrong inflates precision on exactly the crowded frames this detector struggles with.
  * Inclusive-corner IoU. `perception.box_from_mask` returns inclusive corners, so a 1-pixel box has area 1.
    An off-by-one here biases small objects downward — the population the detector is already worst at.
  * `None` rather than `0.0` for a class that never appeared. Zero looks like a detector failure; absent is a
    coverage gap. They call for opposite fixes, so they must never share a value.
  * State errors are counted BOTH as an error and separately. "Found the fridge, misread the door" and "never
    saw the fridge" are different bugs and a single mAP cannot tell them apart.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import oracle_eval as OE                                             # noqa: E402

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


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------------------------------------------------
section("1. IoU on inclusive-corner boxes")
check("identical boxes -> 1.0", close(OE.iou((0, 0, 9, 9), (0, 0, 9, 9)), 1.0))
check("disjoint -> 0.0", OE.iou((0, 0, 4, 4), (10, 10, 14, 14)) == 0.0)
check("touching but not overlapping -> 0.0", OE.iou((0, 0, 4, 4), (5, 0, 9, 4)) == 0.0)
# 10x10 and 10x10 offset by 5: intersection 5x10=50, union 100+100-50=150
check("half-overlap -> 1/3", close(OE.iou((0, 0, 9, 9), (5, 0, 14, 9)), 50 / 150.0))
check("a 1-pixel box has area 1, not 0", close(OE.iou((3, 3, 3, 3), (3, 3, 3, 3)), 1.0))
check("contained box -> ratio of areas", close(OE.iou((0, 0, 9, 9), (0, 0, 4, 4)), 25 / 100.0))
check("iou is symmetric", close(OE.iou((0, 0, 9, 9), (5, 2, 14, 11)), OE.iou((5, 2, 14, 11), (0, 0, 9, 9))))

section("2. split_state")
check("stateful class splits", OE.split_state("Fridge_open") == ("Fridge", "open"))
check("stateless class keeps its name", OE.split_state("Sink") == ("Sink", ""))
check("empty is tolerated", OE.split_state("") == ("", ""))

section("3. one frame's matching")
GT = [("Apple_whole", (0, 0, 19, 19)), ("Fridge_closed", (100, 100, 199, 199))]

m = OE.match_frame(GT, [("Apple_whole", (0, 0, 19, 19), 0.9)])
check("an exact hit is a TP", len(m["tp"]) == 1 and m["tp"][0][0] == "Apple_whole")
check("and the object it did not predict is a miss", m["fn"] == ["Fridge_closed"], str(m["fn"]))

m = OE.match_frame(GT, [("Apple_whole", (0, 0, 19, 19), 0.9), ("Apple_whole", (1, 1, 20, 20), 0.8)])
check("a second prediction on the same object is a DUPLICATE, not a second hit",
      len(m["tp"]) == 1 and len(m["fp"]) == 1, str(m))
check("the more confident prediction is the one that scores", close(m["tp"][0][1], 0.9))

m = OE.match_frame(GT, [("Apple_whole", (60, 60, 79, 79), 0.9)])
check("a prediction below the IoU threshold matches nothing",
      not m["tp"] and len(m["fp"]) == 1 and len(m["fn"]) == 2, str(m))

m = OE.match_frame(GT, [("Fridge_open", (100, 100, 199, 199), 0.8)])
check("right object, wrong state -> recorded as a STATE error",
      m["state"] == [("Fridge", "closed", "open", 0.8)], str(m["state"]))
check("...and still counted as a false positive", any(w == "wrong state" for _, _, w in m["fp"]))
check("...and still counted as a miss", "Fridge_closed" in m["fn"])
check("...and is NOT a true positive", not m["tp"])
check("the state-confused box is claimed, so it is not double-reported",
      m["fn"].count("Fridge_closed") == 1, str(m["fn"]))

m = OE.match_frame([], [("Apple_whole", (0, 0, 9, 9), 0.5)])
check("a detection with no truth at all is a false positive", len(m["fp"]) == 1 and not m["fn"])
m = OE.match_frame(GT, [])
check("no detections -> every truth is a miss", len(m["fn"]) == 2 and not m["fp"])

section("4. average precision")
check("no ground truth -> None, never 0.0", OE.average_precision([(0.9, True)], 0) is None)
check("ground truth but no predictions -> 0.0", OE.average_precision([], 3) == 0.0)
check("one perfect hit -> 1.0", close(OE.average_precision([(0.9, True)], 1), 1.0))
check("half the objects found, no false alarms -> 0.5",
      close(OE.average_precision([(0.9, True)], 2), 0.5))
# ranked TP, FP, TP against 2 positives: recall .5 prec 1.0 ; then recall 1.0 prec 2/3
# all-point interpolation -> 0.5*1.0 + 0.5*(2/3)
check("a false alarm between two hits costs precision, not recall",
      close(OE.average_precision([(0.9, True), (0.8, False), (0.7, True)], 2), 0.5 * 1.0 + 0.5 * (2 / 3.0)))
check("confidence order decides the curve, not list order",
      close(OE.average_precision([(0.7, True), (0.9, True)], 2),
            OE.average_precision([(0.9, True), (0.7, True)], 2)))
check("all false alarms -> 0.0", close(OE.average_precision([(0.9, False), (0.8, False)], 2), 0.0))

section("5. the tally over many frames")
t = OE.Tally()
t.add([("Apple_whole", (0, 0, 19, 19))], [("Apple_whole", (0, 0, 19, 19), 0.9)])
t.add([("Apple_whole", (0, 0, 19, 19))], [])                       # a miss
t.add([("Fridge_closed", (0, 0, 99, 99))], [("Fridge_open", (0, 0, 99, 99), 0.7)])   # state error
s = t.summary()
rows = {r["cls"]: r for r in s["rows"]}
check("frames are counted", s["frames"] == 3)
check("a frame with truth and no detections is flagged", s["empty_frames"] == 1)
check("Apple_whole recall is 1 of 2", close(rows["Apple_whole"]["recall"], 0.5), str(rows["Apple_whole"]))
check("Fridge_closed was never found", rows["Fridge_closed"]["tp"] == 0 and rows["Fridge_closed"]["gt"] == 1)
check("Fridge_open was predicted but never true",
      rows["Fridge_open"]["gt"] == 0 and rows["Fridge_open"]["fp"] == 1, str(rows["Fridge_open"]))
check("a class that never appeared has AP None, not 0.0", rows["Fridge_open"]["ap"] is None)
check("it is reported as untested rather than failed", "Fridge_open" in s["classes_untested"])
check("Fridge_closed is reported as never detected", "Fridge_closed" in s["never_detected"])
check("the state swap is attributed to the object family",
      s["state_swaps"].get("Fridge", {}).get("swapped") == 1, str(s["state_swaps"]))
check("with its direction", s["state_swaps"]["Fridge"]["detail"].get("closed->open") == 1,
      str(s["state_swaps"]["Fridge"]["detail"]))
check("mAP50 averages only the classes actually tested",
      s["classes_scored"] == 2, str(s["classes_scored"]))
check("worst recall is listed first", s["rows"][0]["cls"] in ("Fridge_closed", "Fridge_open"),
      s["rows"][0]["cls"])

empty = OE.Tally().summary()
check("an empty run reports mAP None, not 0.0", empty["mAP50"] is None)

section("6. the miss profile")
t = OE.Tally()
t.miss_distance = [("A", 0.5), ("A", 1.5), ("B", 2.5), ("B", 4.0), ("C", 7.0), ("C", 9.0)]
prof = dict(OE.miss_profile(t))
check("near misses bucket correctly", prof["<= 1 m"] == 1, str(prof))
check("mid-range misses bucket correctly", prof["1-2 m"] == 1 and prof["2-3 m"] == 1, str(prof))
check("far misses are their own bucket", prof["> 5 m"] == 2, str(prof))
check("every miss lands in exactly one bucket", sum(prof.values()) == 6, str(prof))


# ---------------------------------------------------------------------------------------------------------
section("7. ground truth straight from the simulator's masks")


class FakeAgent:
    """The AI2-THOR surface these two functions touch. Masks are real numpy arrays so the box arithmetic and
    the area/sliver filters are exercised for real, not stubbed."""

    def __init__(self, objs, size=200):
        import numpy as np
        self.size = size
        self.frame = np.zeros((size, size, 3), dtype=np.uint8)
        self._objs = objs
        self.success = True
        self.calls = []
        self.dets = []
        self._masks = {}
        for o in objs:
            m = np.zeros((size, size), dtype=bool)
            x1, y1, x2, y2 = o["box"]
            m[y1:y2 + 1, x1:x2 + 1] = True
            self._masks[o["objectId"]] = m

    def visible_objects(self):
        return [{k: v for k, v in o.items() if k != "box"} for o in self._objs]

    @property
    def instance_masks(self):
        return self._masks

    def teleport_full(self, x, y, z, rot, horizon=0.0, standing=True):
        self.calls.append(("teleport", round(x, 2), round(z, 2)))
        self.success = True

    def detect_objects(self, det, conf=0.25):
        self.calls.append(("detect",))
        return list(self.dets)

    def reachable_positions(self):
        return [{"x": float(i), "y": 0.9, "z": float(i)} for i in range(12)]


OBJS = [
    {"objectId": "Apple|1", "objectType": "Apple", "isSliced": False, "distance": 1.2, "box": (40, 40, 79, 79)},
    {"objectId": "Fridge|1", "objectType": "Fridge", "isOpen": False, "distance": 3.4, "box": (100, 20, 179, 179)},
    {"objectId": "Speck|1", "objectType": "Apple", "isSliced": False, "distance": 6.0, "box": (5, 5, 9, 9)},
]
ag = FakeAgent(OBJS)
gt = OE.ground_truth_boxes(ag, min_area=400)
names = sorted(c for c, _ in gt)
check("stateful class names come from perception.class_name",
      names == ["Apple_whole", "Fridge_closed"], str(names))
check("boxes are the tight mask bounds",
      dict(gt)["Apple_whole"] == (40, 40, 79, 79), str(dict(gt).get("Apple_whole")))
check("a speck below min_area is dropped, exactly as the training labeller drops it",
      all(c != "Apple_whole" or b != (5, 5, 9, 9) for c, b in gt))
check("min_area is honoured", len(OE.ground_truth_boxes(ag, min_area=100000)) == 0)

section("8. sampling poses independent of any scan trajectory")
poses = OE.sample_poses(ag, 20, seed=7)
check("asks for n poses and gets n", len(poses) == 20)
check("every pose is a reachable position",
      all(any(abs(p["x"] - q["x"]) < 1e-9 for q in ag.reachable_positions()) for p in poses))
check("the same seed reproduces the same poses",
      [p["x"] for p in OE.sample_poses(ag, 20, seed=7)] == [p["x"] for p in poses])
check("a different seed does not", [p["x"] for p in OE.sample_poses(ag, 20, seed=8)] != [p["x"] for p in poses])
check("yaw is not axis-aligned every time",
      len({round(p["rotation"], 3) for p in poses}) > 4, str(len({round(p['rotation'], 3) for p in poses})))
check("both stances are sampled", len({p["standing"] for p in poses}) == 2)
check("no reachable positions -> no poses, not a crash",
      OE.sample_poses(type("E", (), {"reachable_positions": lambda s: []})(), 5) == [])

section("9. the live loop end to end")
ag = FakeAgent(OBJS)
ag.dets = [{"cls": "Apple_whole", "bbox_xyxy": (40, 40, 79, 79), "conf": 0.88},
           {"cls": "Fridge_open", "bbox_xyxy": (100, 20, 179, 179), "conf": 0.61}]
tal = {}
labels = list(OE.evaluate_live(ag, object(), OE.sample_poses(ag, 4, seed=1),
                               on_pose=lambda i, t: tal.setdefault("t", t)))
check("one progress label per pose", len(labels) == 4, str(len(labels)))
check("it teleported to each pose", len([c for c in ag.calls if c[0] == "teleport"]) == 4)
check("it ran the detector at each pose", len([c for c in ag.calls if c[0] == "detect"]) == 4)
s = tal["t"].summary()
check("the apple was found every time", {r["cls"]: r for r in s["rows"]}["Apple_whole"]["tp"] == 4)
check("the fridge state error was caught every time",
      s["state_swaps"].get("Fridge", {}).get("swapped") == 4, str(s["state_swaps"]))
check("misses carry a distance, so the profile can be built", len(tal["t"].miss_distance) > 0)


class DeadDetector(FakeAgent):
    def detect_objects(self, det, conf=0.25):
        raise RuntimeError("cuda out of memory")


dead = DeadDetector(OBJS)
lab = list(OE.evaluate_live(dead, object(), OE.sample_poses(dead, 2, seed=1)))
check("a detector that throws is reported per pose, not fatal",
      len(lab) == 2 and all("detector failed" in x for x in lab), str(lab))


class Unreachable(FakeAgent):
    def teleport_full(self, *a, **kw):
        self.success = False


unr = Unreachable(OBJS)
lab = list(OE.evaluate_live(unr, object(), OE.sample_poses(unr, 3, seed=1)))
check("an unreachable pose is skipped and said so", all("unreachable" in x for x in lab), str(lab))

section("10. the report renders")
txt = OE.render_report(s, "FloorPlan1")
check("names the scene", "FloorPlan1" in txt)
check("shows the mAP", "mAP50" in txt)
check("has a state-error section when there are state errors", "STATE ERRORS" in txt)
check("is pure ASCII, so it survives a cp1252 console", all(ord(c) < 128 for c in txt))
check("an empty run still renders", "ORACLE-VERIFIED" in OE.render_report(OE.Tally().summary(), "X"))

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("ORACLE-EVAL SELFTEST OK")
