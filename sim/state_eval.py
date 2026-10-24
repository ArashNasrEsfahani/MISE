"""Calibrate and evaluate the per-axis state classifiers ON DETECTOR CROPS (Astra step3s).

1. Size floor per axis from Arash's 300 blind hand labels: the smallest box area from which he could see the state in
   at least half of the crops (rolling window over items sorted by box area). Below it the answer is "unknown".
2. Run the primary detector on VAL and TEST images; match each detection to a GT box of the same type (IoU >= .5);
   crop the DETECTED box exactly as at runtime; classify with each applicable axis model.
3. On VAL only: per axis AND predicted value, the lowest confidence threshold whose accepted answers reach >= 95 %
   precision with >= 20 accepted (else that value is never asserted -> "unknown"). Frozen before TEST.
4. TEST: accuracy among answered, per-value precision, coverage over ALL applicable matched detections (including
   ones below the size floor), and false assertions on unmatched / wrong-type detections.

    ~/thor-env/bin/python sim/state_eval.py           -> sim/kitchen_v3/FloorPlan1/state_calib.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_data as sd  # noqa: E402
import vocab  # noqa: E402

SCENE = "FloorPlan1"
STORE = os.path.expanduser(f"~/thor-data/kitchen_v3/{SCENE}")
WEIGHTS = os.path.join(HERE, "weights", "v3")
REVIEW = os.path.join(HERE, "kitchen_v3", SCENE, "state_review")
OUT = os.path.join(HERE, "kitchen_v3", SCENE, "state_calib.json")
TARGET_PREC, MIN_ACCEPT = 0.95, 20


def axes_for(t):
    return [a for a in vocab.TYPE_AXES.get(t, ()) if a in sd.AXES and (a not in sd.AXIS_TYPES or t in sd.AXIS_TYPES[a])]


def size_floors():
    rows = json.load(open(os.path.join(REVIEW, "score.json")))["rows"]
    out = {}
    for ax in sd.AXES:
        r = sorted((x for x in rows if x["axis"] == ax), key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
        area = [(x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]) for x in r]
        vis = [1 if x["outcome"] == "right" else 0 for x in r]
        floor = None
        w = 8
        for i in range(len(r) - w + 1):
            if sum(vis[i:i + w]) / w >= 0.5:
                floor = area[i]
                break
        out[ax] = int(floor if floor is not None else (area[-1] if area else 0))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def collect(split, det, cls, limit=None):
    """[(axis, gt value | None for unmatched, pred value, conf, box area)] for every applicable detection."""
    from PIL import Image
    recs = []
    rows = [json.loads(l) for l in open(os.path.join(STORE, "frames.jsonl"))]
    rows = [r for r in rows if r.get("split") == split and "img" in r and (r.get("protocol") or "main") == "main"]
    if limit:
        rows = rows[:limit]
    for r in rows:
        img = Image.open(os.path.join(STORE, "images", split, r["img"] + ".jpg")).convert("RGB")
        import numpy as np
        res = det(np.asarray(img)[..., ::-1].copy(), conf=0.25, verbose=False, imgsz=640)[0]
        for b in res.boxes:
            t = det.names[int(b.cls[0])]
            ax_list = axes_for(t)
            if not ax_list:
                continue
            box = tuple(float(v) for v in b.xyxy[0].tolist())
            area = (box[2] - box[0]) * (box[3] - box[1])
            gt = max((l for l in r["labels"] if l["type"] == t), key=lambda l: iou(box, l["box"]), default=None)
            matched = gt is not None and iou(box, gt["box"]) >= 0.5
            crop = sd.crop_square(img, box)
            for ax in ax_list:
                p = cls[ax](crop, imgsz=128, verbose=False)[0]
                pv = p.names[int(p.probs.top1)]
                recs.append({"axis": ax, "type": t, "gt": sd.value_of(gt["state"], ax) if matched else None,
                             "matched": matched, "pred": pv, "conf": float(p.probs.top1conf), "area": area})
    return recs


def thresholds(val):
    th = {}
    for ax in sd.AXES:
        th[ax] = {}
        for v in ("0", "1"):
            cand = sorted((x for x in val if x["axis"] == ax and x["pred"] == v and x["gt"] is not None),
                          key=lambda x: -x["conf"])
            best = None
            ok = 0
            for i, x in enumerate(cand, 1):
                ok += int(x["gt"] == v)
                if i >= MIN_ACCEPT and ok / i >= TARGET_PREC:
                    best = x["conf"]
            th[ax][v] = best                         # None -> this value is never asserted
    return th


def answer(x, th, floors):
    if x["area"] < floors.get(x["axis"], 0):
        return None
    t = th[x["axis"]].get(x["pred"])
    return x["pred"] if (t is not None and x["conf"] >= t) else None


def report(test, th, floors):
    out = {}
    for ax in sd.AXES:
        m = [x for x in test if x["axis"] == ax and x["matched"] and x["gt"] is not None]
        ans = [(x, answer(x, th, floors)) for x in m]
        answered = [(x, a) for x, a in ans if a is not None]
        acc = sum(1 for x, a in answered if a == x["gt"]) / max(1, len(answered))
        per = {}
        for v in ("0", "1"):
            av = [(x, a) for x, a in answered if a == v]
            per[v] = {"asserted": len(av), "precision": round(sum(1 for x, a in av if x["gt"] == v) / max(1, len(av)), 3)}
        un = [x for x in test if x["axis"] == ax and not x["matched"]]
        fa = sum(1 for x in un if answer(x, th, floors) is not None)
        out[ax] = {"applicable_detections": len(m), "answered": len(answered),
                   "coverage": round(len(answered) / max(1, len(m)), 3), "accuracy_answered": round(acc, 3),
                   "per_value": per, "unmatched_detections": len(un), "false_assertions_unmatched": fa}
    return out


def main():
    from ultralytics import YOLO
    import shutil
    import tempfile

    def load(p):
        tmp = os.path.join(tempfile.gettempdir(), "se_" + os.path.basename(p))
        shutil.copyfile(p, tmp)
        return YOLO(tmp)

    det = load(os.path.join(WEIGHTS, f"{SCENE}.yolo26s.pt"))
    cls = {ax: load(os.path.join(WEIGHTS, f"{SCENE}.state_{ax}.pt")) for ax in sd.AXES}
    floors = size_floors()
    print("size floors (box px^2):", floors, flush=True)
    val = collect("val", det, cls)
    th = thresholds(val)
    print("thresholds (VAL):", th, flush=True)
    test = collect("test", det, cls)
    rep = report(test, th, floors)
    for ax, r in rep.items():
        print(f"TEST {ax:7s} coverage {r['coverage']:.2f} acc {r['accuracy_answered']:.3f} per-value {r['per_value']} "
              f"unmatched-FA {r['false_assertions_unmatched']}/{r['unmatched_detections']}", flush=True)
    json.dump({"floors_box_area": floors, "thresholds": th, "target_precision": TARGET_PREC, "min_accept": MIN_ACCEPT,
               "test": rep, "val_report": report(val, th, floors)}, open(OUT, "w"), indent=1)
    print("SAVED", OUT)


if __name__ == "__main__":
    main()
