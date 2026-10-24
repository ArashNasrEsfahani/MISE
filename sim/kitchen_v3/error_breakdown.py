"""Step-3 disambiguation, decision input (Astra step3j): WHERE do the primary's errors come from, on VALIDATION only?

For each ground-truth box of a class (TIDE-style, at the operating threshold): correct, CLASSIFICATION error (a box of
another class overlaps it at IoU >= .5), LOCALISATION error (a box of the right class, .1 <= IoU < .5), or MISSED
(nothing of any class at IoU >= .1). Then the ORACLE RELABELLING CEILING: AP50 if every prediction that overlaps a GT
box at IoU >= .5 took that GT's class — the most any relabelling second stage could ever give. AP is computed at
conf .001 (oracle_eval.Tally), separately from the .25 operating-point breakdown. TEST is never read.

    ~/thor-env/bin/python "sim/kitchen_v3/error_breakdown.py" <weights.pt> <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import oracle_eval  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DS = "/home/arash/thor-data/kitchen_v3/FloorPlan1/yolo_primary"
FRAME = 640
OP_CONF = 0.25


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load_gt(img, names):
    lab = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = []
    if os.path.exists(lab):
        for line in open(lab):
            c, x, y, w, h = line.split()
            x, y, w, h = (float(v) * FRAME for v in (x, y, w, h))
            out.append((names[int(c)], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return out


def main(weights, out_path):
    model = YOLO(weights)
    names = model.names
    imgs = [l.strip() for l in open(os.path.join(DS, "val.txt")) if l.strip()]
    base, oracle = oracle_eval.Tally(0.5), oracle_eval.Tally(0.5)
    kinds = {}                                       # class -> {correct, cls, loc, missed, n, confused_with{}}
    for i in range(0, len(imgs), 32):
        batch = imgs[i:i + 32]
        for img, r in zip(batch, model.predict(batch, conf=0.001, iou=0.7, max_det=300, verbose=False, imgsz=640)):
            gt = load_gt(img, names)
            preds = [(names[int(b.cls[0])], tuple(b.xyxy[0].tolist()), float(b.conf[0])) for b in r.boxes]
            base.add(gt, preds)
            # oracle relabel: a prediction overlapping a GT box at IoU >= .5 takes the best-overlapping GT's class
            relab = []
            for c, box, s in preds:
                best = max(((iou(box, g[1]), g[0]) for g in gt), default=(0.0, None))
                relab.append((best[1] if best[0] >= 0.5 else c, box, s))
            oracle.add(gt, relab)
            op = [p for p in preds if p[2] >= OP_CONF]
            for gc, gb in gt:
                k = kinds.setdefault(gc, {"n": 0, "correct": 0, "cls": 0, "loc": 0, "missed": 0, "confused_with": {}})
                k["n"] += 1
                ov = [(iou(gb, pb), pc) for pc, pb, _ in op]
                if any(v >= 0.5 and pc == gc for v, pc in ov):
                    k["correct"] += 1
                elif any(v >= 0.5 for v, pc in ov):
                    k["cls"] += 1
                    other = max((v, pc) for v, pc in ov if v >= 0.5)[1]
                    k["confused_with"][other] = k["confused_with"].get(other, 0) + 1
                elif any(0.1 <= v < 0.5 and pc == gc for v, pc in ov):
                    k["loc"] += 1
                else:
                    k["missed"] += 1
    ap_b = {r["cls"]: r["ap"] for r in base.per_class() if r["gt"]}
    ap_o = {r["cls"]: r["ap"] for r in oracle.per_class() if r["gt"]}
    rows = []
    for c, k in kinds.items():
        n = k["n"]
        rows.append({"cls": c, "n": n, "ap50": ap_b.get(c), "ap50_oracle_relabel": ap_o.get(c),
                     "correct": round(k["correct"] / n, 3), "cls_err": round(k["cls"] / n, 3),
                     "loc_err": round(k["loc"] / n, 3), "missed": round(k["missed"] / n, 3),
                     "confused_with": dict(sorted(k["confused_with"].items(), key=lambda kv: -kv[1])[:3])})
    rows.sort(key=lambda r: r["ap50"] if r["ap50"] is not None else 1)
    mb = [v for v in ap_b.values() if v is not None]
    mo = [v for v in ap_o.values() if v is not None]
    rep = {"split": "val", "weights": weights, "op_conf": OP_CONF, "images": len(imgs),
           "mAP50": sum(mb) / len(mb), "mAP50_oracle_relabel": sum(mo) / len(mo), "rows": rows}
    json.dump(rep, open(out_path, "w"), indent=1)
    print(f"VAL mAP50 {rep['mAP50']:.3f} -> oracle relabel ceiling {rep['mAP50_oracle_relabel']:.3f}")
    for r in rows[:14]:
        print("ROW", r["cls"], "n", r["n"], "AP50", round(r["ap50"], 3), "-> oracle", round(r["ap50_oracle_relabel"], 3),
              "| correct", r["correct"], "cls", r["cls_err"], "loc", r["loc_err"], "missed", r["missed"],
              r["confused_with"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
