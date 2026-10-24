"""Score the detector against AI2-THOR's ground truth, at poses it has never seen.

Every other way of measuring this detector has the same flaw: it scores against frames that came out of the
same scan that trained it. Even a site-level split only guarantees the *sites* differ — the lighting, the
object placement and the camera model are identical, and a held-out frame of the same fridge from two metres
left is not an independent test. That is why the June run reported mAP50 0.732 while live detection was, in
the user's words, dogshit. The number was not lying about the frames it scored; those frames were just not a
test.

The simulator removes the excuse. AI2-THOR hands over a per-object instance MASK for whatever is on screen, so
a perfect label can be produced for **any** frame at **any** pose, for free, with no annotation and no
held-out set. So: teleport somewhere the detector has never stood, render, ask the detector what it sees, ask
the simulator what is actually there, and compare. Repeat at freshly sampled poses. There is no leakage to
argue about because there is no split — the evaluation frames are generated after training, from poses drawn
independently of the scan trajectory.

Two things this measures that a val-set mAP cannot:

  * **State errors.** The classes are state-aware (`Fridge_open` vs `Fridge_closed`). A detector that finds
    every fridge but cannot tell an open one from a shut one is useless to this thesis, and in a single mAP
    number that failure is indistinguishable from a detector that simply misses fridges. Here it is its own
    column: the object was found, the state was wrong, and in which direction.
  * **Where it fails.** Every miss carries the pose it happened at and the object's distance, so "it cannot
    see anything past 3 m" and "it cannot see inside cabinets" become measurements rather than impressions.

The scoring core is pure and is unit-tested without a simulator; only :func:`ground_truth_boxes` and
:func:`evaluate_live` need AI2-THOR, and only the latter needs a GPU.

CLI (inside WSL, with the console stopped so the GPU is free):

    python sim/oracle_eval.py --scene FloorPlan1 --poses 60
"""
from __future__ import annotations

import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

#: IoU at which a prediction is allowed to claim a ground-truth box. 0.5 is the PASCAL VOC convention and is
#: what "mAP50" means everywhere else in this repo, so the numbers stay comparable to the ultralytics ones.
DEFAULT_IOU = 0.5


# ----------------------------------------------------------------------------------------------------------
# Pure geometry + matching. No simulator, no torch.
# ----------------------------------------------------------------------------------------------------------
def iou(a, b) -> float:
    """Intersection-over-union of two inclusive-corner boxes ``(x1, y1, x2, y2)``.

    Inclusive corners (the convention `perception.box_from_mask` returns) mean a 1-pixel box has area 1, not
    0 — hence the ``+ 1`` terms. Getting that wrong biases every small object's IoU downward, which is exactly
    the population this detector is worst at.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = (ix2 - ix1 + 1), (iy2 - iy1 + 1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def split_state(cls: str):
    """``'Fridge_open' -> ('Fridge', 'open')``; a class with no state axis -> ``(cls, '')``.

    Delegates to `perception.split_class_name` when it is importable so the two cannot drift, and falls back
    to the same rule inline so this module stays usable (and testable) on its own.
    """
    try:
        import perception
        return perception.split_class_name(cls)
    except Exception:
        s = str(cls or "")
        return (s.split("_", 1)[0], s.split("_", 1)[1]) if "_" in s else (s, "")


def match_frame(gt, preds, iou_thr: float = DEFAULT_IOU) -> dict:
    """Match one frame's predictions to its ground truth.

    ``gt``    -> ``[(class_name, box)]``            — from the simulator, so it is complete and correct.
    ``preds`` -> ``[(class_name, box, confidence)]`` — from the detector.

    Greedy by confidence, one prediction per ground-truth box, which is the standard VOC/COCO assignment: the
    most confident prediction gets first refusal, and a second prediction on the same object is a duplicate
    (false positive), not a second hit. Returns

        {"tp": [(cls, conf, iou)], "fp": [(cls, conf, why)], "fn": [cls],
         "state": [(obj_type, truth_state, pred_state, conf)]}

    A **state error** is counted when a prediction overlaps a ground-truth box well enough and names the right
    object TYPE but the wrong STATE. It is still scored as a miss and a false alarm for the two classes
    involved — nothing is forgiven — but it is reported separately, because "found the fridge, misread the
    door" and "never saw the fridge" call for completely different fixes.
    """
    out = {"tp": [], "fp": [], "fn": [], "state": []}
    taken = set()
    order = sorted(range(len(preds)), key=lambda i: -float(preds[i][2]))
    for i in order:
        pcls, pbox, pconf = preds[i]
        ptype, pstate = split_state(pcls)
        best_j, best_iou, best_type_j, best_type_iou = -1, 0.0, -1, 0.0
        for j, (gcls, gbox) in enumerate(gt):
            if j in taken:
                continue
            ov = iou(pbox, gbox)
            if ov < iou_thr:
                continue
            if gcls == pcls and ov > best_iou:
                best_j, best_iou = j, ov
            if split_state(gcls)[0] == ptype and ov > best_type_iou:
                best_type_j, best_type_iou = j, ov
        if best_j >= 0:
            taken.add(best_j)
            out["tp"].append((pcls, float(pconf), best_iou))
            continue
        if best_type_j >= 0:
            # Right object, wrong state. Claim the box so it is not also reported as an unrelated miss, but
            # still count both halves as errors below.
            taken.add(best_type_j)
            gtype, gstate = split_state(gt[best_type_j][0])
            out["state"].append((gtype, gstate, pstate, float(pconf)))
            out["fp"].append((pcls, float(pconf), "wrong state"))
            out["fn"].append(gt[best_type_j][0])
            continue
        out["fp"].append((pcls, float(pconf), "no match"))
    for j, (gcls, _) in enumerate(gt):
        if j not in taken:
            out["fn"].append(gcls)
    return out


def average_precision(records, n_positives: int) -> float | None:
    """VOC all-point-interpolated AP from ``[(confidence, is_tp)]`` and the number of ground-truth instances.

    ``None`` when there were no ground-truth instances at all — the honest answer for a class the evaluation
    never had a chance to test. Returning 0.0 there would look like a detector failure instead of a coverage
    gap, and those are opposite problems.
    """
    if not n_positives:
        return None
    if not records:
        return 0.0
    ranked = sorted(records, key=lambda r: -float(r[0]))
    tp = fp = 0
    points = []
    for _, is_tp in ranked:
        if is_tp:
            tp += 1
        else:
            fp += 1
        points.append((tp / float(n_positives), tp / float(tp + fp)))   # (recall, precision)
    # all-point interpolation: precision is replaced by the best precision at any recall >= this one
    best = 0.0
    interp = []
    for rec, prec in reversed(points):
        best = max(best, prec)
        interp.append((rec, best))
    interp.reverse()
    ap, prev_r = 0.0, 0.0
    for rec, prec in interp:
        ap += (rec - prev_r) * prec
        prev_r = rec
    return ap


class Tally:
    """Accumulates match results over many frames into a per-class report.

    Deliberately dumb: it stores per-prediction (confidence, hit) records rather than running counts, because
    AP needs the full confidence ranking and a running precision/recall cannot be turned back into one.
    """

    def __init__(self, iou_thr: float = DEFAULT_IOU):
        self.iou_thr = iou_thr
        self.records: dict = {}      # class -> [(conf, is_tp)]
        self.positives: dict = {}    # class -> ground-truth instance count
        self.misses: dict = {}       # class -> miss count
        self.state_pairs: dict = {}  # object type -> {"correct": n, "swapped": n, "detail": {(t,p): n}}
        self.frames = 0
        self.empty_frames = 0        # frames where the simulator saw objects and the detector saw nothing
        self.miss_distance: list = []  # (class, distance_m) for every miss we have a distance for

    def add(self, gt, preds, distances=None):
        self.frames += 1
        for gcls, _ in gt:
            self.positives[gcls] = self.positives.get(gcls, 0) + 1
        m = match_frame(gt, preds, self.iou_thr)
        if gt and not preds:
            self.empty_frames += 1
        for cls, conf, _ in m["tp"]:
            self.records.setdefault(cls, []).append((conf, True))
        for cls, conf, _why in m["fp"]:
            self.records.setdefault(cls, []).append((conf, False))
        for cls in m["fn"]:
            self.misses[cls] = self.misses.get(cls, 0) + 1
            if distances and cls in distances:
                self.miss_distance.append((cls, distances[cls]))
        for gtype, gstate, pstate, _conf in m["state"]:
            e = self.state_pairs.setdefault(gtype, {"correct": 0, "swapped": 0, "detail": {}})
            e["swapped"] += 1
            key = f"{gstate}->{pstate}"
            e["detail"][key] = e["detail"].get(key, 0) + 1
        for cls, _conf, _iou in m["tp"]:
            gtype = split_state(cls)[0]
            e = self.state_pairs.setdefault(gtype, {"correct": 0, "swapped": 0, "detail": {}})
            e["correct"] += 1
        return m

    def per_class(self) -> list:
        """One row per class that either exists in the truth or was ever predicted, worst recall first."""
        rows = []
        for cls in sorted(set(self.positives) | set(self.records)):
            recs = self.records.get(cls, [])
            n_pos = self.positives.get(cls, 0)
            tp = sum(1 for _, hit in recs if hit)
            fp = len(recs) - tp
            rows.append({
                "cls": cls,
                "gt": n_pos,
                "tp": tp,
                "fp": fp,
                "fn": self.misses.get(cls, 0),
                "precision": (tp / float(tp + fp)) if (tp + fp) else None,
                "recall": (tp / float(n_pos)) if n_pos else None,
                "ap": average_precision(recs, n_pos),
            })
        rows.sort(key=lambda r: (r["recall"] is None, r["recall"] if r["recall"] is not None else 0.0, -r["gt"]))
        return rows

    def summary(self) -> dict:
        rows = self.per_class()
        scored = [r["ap"] for r in rows if r["ap"] is not None]
        untested = [r["cls"] for r in rows if r["gt"] == 0]
        never_found = [r["cls"] for r in rows if r["gt"] > 0 and r["tp"] == 0]
        pairs = {t: v for t, v in self.state_pairs.items() if v["swapped"]}
        return {
            "frames": self.frames,
            "empty_frames": self.empty_frames,
            "classes_scored": len(scored),
            "classes_untested": untested,
            "never_detected": never_found,
            # mAP over classes the evaluation ACTUALLY tested. A mean that silently includes 0.0 for classes
            # that never appeared is not a detector score, it is a coverage score wearing a detector's name.
            "mAP50": (sum(scored) / len(scored)) if scored else None,
            "state_swaps": pairs,
            "rows": rows,
        }


def miss_profile(tally, bands=(1.0, 2.0, 3.0, 5.0)) -> list:
    """Misses bucketed by distance — turns "it cannot see anything far away" into a number.

    Returns ``[(label, count)]``. Only misses for which a distance was supplied are counted, so the caller
    must pass ``distances`` to :meth:`Tally.add` for this to say anything.
    """
    out = [(f"<= {bands[0]:.0f} m", 0)]
    for a, b in zip(bands, bands[1:]):
        out.append((f"{a:.0f}-{b:.0f} m", 0))
    out.append((f"> {bands[-1]:.0f} m", 0))
    counts = [0] * len(out)
    for _cls, d in tally.miss_distance:
        idx = len(bands)
        for i, edge in enumerate(bands):
            if d <= edge:
                idx = i
                break
        counts[idx] += 1
    return [(lbl, counts[i]) for i, (lbl, _) in enumerate(out)]


# ----------------------------------------------------------------------------------------------------------
# Simulator side. Needs AI2-THOR; evaluate_live additionally needs the detector (and therefore the GPU).
# ----------------------------------------------------------------------------------------------------------
def ground_truth_boxes(agent, min_area: int = 400, min_visible_frac: float = 0.0) -> list:
    """``[(class_name, box)]`` for the agent's current frame, straight from AI2-THOR's instance masks.

    This is the SAME filter chain `gen_dataset.capture_frame` applies when it writes a training label — same
    edge-sliver rejection, same stateful class name. That is deliberate and it is the whole basis of the
    comparison: if evaluation ground truth were filtered differently from training ground truth, every number
    here would be measuring the difference between two labellers rather than the detector.

    **The filter has two KNOBS, and they must match the labeller too** (found 2026-09-15): `min_area`, and the
    agent's `visibility_distance`, because `visible_objects()` is range-gated by it. The June scan labelled at
    1.5 m / 400 px²; the container-aware scan labels at 6 m / 150 px². Scored against 1.5 m truth, a detector
    trained on 6 m labels "hallucinates" every far cabinet it correctly finds (825 false positives vs 96), and a
    1.5 m detector scored against 6 m truth "misses" objects it was taught were background. Neither gate is
    neutral, so a comparison of two detectors must be reported under BOTH, with the gate stated.
    """
    import gen_dataset
    import numpy as np
    import perception

    frame = agent.frame
    if frame is None:
        return []
    frame = np.asarray(frame)
    h, w = frame.shape[0], frame.shape[1]
    masks = gen_dataset._instance_masks(agent)
    kept = []
    for o in agent.visible_objects():
        mask = masks.get(o.get("objectId"))
        if mask is None:
            continue
        box = perception.box_from_mask(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area < min_area:
            continue
        if gen_dataset._is_edge_sliver(box, w, h):
            continue
        if min_visible_frac > 0.0:
            if int(np.asarray(mask, dtype=bool).sum()) / float(area) < min_visible_frac:
                continue
        kept.append((perception.class_name(o), box))
    return kept


def sample_poses(agent, n: int, seed: int = 0, horizons=(-15.0, 0.0, 15.0, 30.0)) -> list:
    """`n` evaluation poses drawn from the scene's reachable positions, independent of the scan trajectory.

    Independence is the point. The scan visits a coverage pattern and then walks to specific objects; drawing
    uniformly from `reachable_positions()` with a fixed seed gives poses the detector has no reason to have
    seen, while staying reproducible. Yaw is uniform over the four cardinals plus a random offset so the
    camera is not always axis-aligned with the room.
    """
    pts = agent.reachable_positions() or []
    if not pts:
        return []
    rng = random.Random(seed)
    poses = []
    for _ in range(max(1, int(n))):
        p = rng.choice(pts)
        poses.append({
            "x": p["x"], "y": p["y"], "z": p["z"],
            "rotation": rng.choice([0.0, 90.0, 180.0, 270.0]) + rng.uniform(-30.0, 30.0),
            "horizon": rng.choice(list(horizons)),
            "standing": rng.random() > 0.25,
        })
    return poses


def evaluate_live(agent, detector, poses, *, conf: float = 0.25, iou_thr: float = DEFAULT_IOU,
                  min_area: int = 400, on_pose=None):
    """GENERATOR: teleport to each pose, compare the detector against the simulator, yield a progress label.

    Yields once per pose so a console can animate it. The accumulated :class:`Tally` is returned through
    ``on_pose`` (called with ``(index, tally)`` after each frame) — generators cannot be awaited for a value
    here, and the caller usually wants to render progress anyway.
    """
    tally = Tally(iou_thr=iou_thr)
    for i, pose in enumerate(poses):
        agent.teleport_full(pose["x"], pose["y"], pose["z"], pose["rotation"],
                            horizon=pose.get("horizon", 0.0), standing=pose.get("standing", True))
        if not agent.success:
            yield f"pose {i + 1}/{len(poses)}: unreachable, skipped"
            continue
        gt = ground_truth_boxes(agent, min_area=min_area)
        try:
            dets = agent.detect_objects(detector, conf=conf) or []
        except Exception as exc:
            yield f"pose {i + 1}/{len(poses)}: detector failed ({exc})"
            continue
        preds = [(d["cls"], tuple(d["bbox_xyxy"]), float(d.get("conf", 0.0))) for d in dets]
        # distance is oracle metadata, which is legitimate here: this is the MEASUREMENT, not the agent's
        # perception. It is what turns "misses things" into "misses things past 3 m".
        dist = {}
        for o in agent.visible_objects():
            try:
                import perception
                dist[perception.class_name(o)] = float(o.get("distance") or 0.0)
            except Exception:
                pass
        tally.add(gt, preds, distances=dist)
        if on_pose:
            on_pose(i, tally)
        yield f"pose {i + 1}/{len(poses)}: {len(gt)} truth / {len(preds)} detected"
    return


def render_report(summary: dict, scene: str, top: int = 25) -> str:
    """The report as text - worst class first, because that is the list you act on."""
    L = []
    m = summary.get("mAP50")
    L.append(f"ORACLE-VERIFIED DETECTION - {scene}")
    L.append(f"  frames scored        : {summary['frames']}"
             f"   (blank: {summary['empty_frames']} had truth but no detections)")
    L.append(f"  classes scored       : {summary['classes_scored']}")
    L.append(f"  mAP50 (scored only)  : {'n/a' if m is None else f'{m:.3f}'}")
    if summary["never_detected"]:
        L.append(f"  NEVER detected       : {', '.join(summary['never_detected'][:12])}")
    if summary["classes_untested"]:
        L.append(f"  never appeared       : {len(summary['classes_untested'])} class(es) - not a detector fault")
    L.append("")
    L.append(f"  {'class':<26}{'gt':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'recall':>8}{'AP50':>8}")
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    for r in summary["rows"][:top]:
        L.append(f"  {r['cls']:<26}{r['gt']:>5}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
                 f"{fmt(r['precision']):>8}{fmt(r['recall']):>8}{fmt(r['ap']):>8}")
    if summary["state_swaps"]:
        L.append("")
        L.append("  STATE ERRORS - object found, state misread (the failure a single mAP hides):")
        for t, v in sorted(summary["state_swaps"].items(), key=lambda kv: -kv[1]["swapped"]):
            tot = v["correct"] + v["swapped"]
            pct = (100.0 * v["swapped"] / tot) if tot else 0.0
            worst = max(v["detail"].items(), key=lambda kv: kv[1]) if v["detail"] else ("", 0)
            L.append(f"    {t:<22} {v['swapped']:>4} of {tot:>4} ({pct:5.1f}%)   worst: {worst[0]} x{worst[1]}")
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--poses", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--weights", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--gt-visibility", type=float, default=1.5,
                    help="AI2-THOR visibility gate (m) for ground truth; match the labeller (June scan 1.5, "
                         "container-aware scan 6.0)")
    ap.add_argument("--gt-min-area", type=int, default=400,
                    help="minimum ground-truth box area (px^2); match the labeller (June 400, container scan 150)")
    args = ap.parse_args(argv)

    import perception
    import thor_actions as ta

    wp = args.weights or perception.weights_path(args.scene, os.path.join(HERE, "weights"))
    if not wp:
        print(f"no trained detector for {args.scene!r} — run 'scan & learn' then 'train detector' first")
        return 2
    agent = ta.ThorAgent(scene=args.scene, render_instance_seg=True, visibility_distance=args.gt_visibility)
    det = perception.Detector(wp)
    poses = sample_poses(agent, args.poses, seed=args.seed)
    if not poses:
        print("no reachable positions — is the scene loaded?")
        return 2
    print(f"scoring {len(poses)} freshly-sampled poses against ground truth (seed {args.seed})", flush=True)
    tally = Tally(iou_thr=args.iou)

    def keep(_i, t):
        nonlocal tally
        tally = t

    for label in evaluate_live(agent, det, poses, conf=args.conf, iou_thr=args.iou, min_area=args.gt_min_area,
                               on_pose=keep):
        print("  " + label, flush=True)
    summary = tally.summary()
    print()
    print(render_report(summary, args.scene, top=args.top))
    out = args.out or os.path.join(HERE, f"oracle_eval_{args.scene}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"scene": args.scene, "poses": len(poses), "seed": args.seed,
                   "conf": args.conf, "iou": args.iou, "weights": wp,
                   "gt": {"visibility": args.gt_visibility, "min_area": args.gt_min_area}, "summary": summary},
                  fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
