"""State-estimator crops (step 3, option b's third piece; Astra step3s).

The detectors are type-only; states (open, on, sliced, cooked, broken, dirty, filled) come from one small classifier
per axis applied to each detected box. This builds the per-axis crop folders from the kitchen_v3 store's GT boxes:

    <out>/<axis>/{train,val,test}/<value>/<img>_<k>.jpg      (ultralytics classify layout)

* splits = the capture's episode splits (no new split);
* the crop = the GT box padded 15 %, letterboxed onto a square (the whole object kept — no centre crop at train or
  inference; training uses scale=0 so RandomResizedCrop keeps the full crop);
* a minimum visible size per axis (px of the object's mask);
* TRAIN balanced per value (cap); VAL/TEST keep the natural frequencies (Astra: balancing them changes the measured
  precision);
* sliced/broken pieces are already folded into their parent type by the labeller (LettuceSliced -> Lettuce).

    ~/thor-env/bin/python sim/state_data.py --out ~/thor-data/kitchen_v3/state_cls
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

AXES = ("open", "on", "sliced", "cooked", "broken", "dirty", "filled")
MIN_PX = 300
# an axis is judged only for types that show BOTH values in the capture, or the classifier learns identity, not state
# (Astra step3s): vessels were never broken during capture (shards are unlabelled), so "broken" is Egg-only
AXIS_TYPES = {"broken": {"Egg"}}
TRAIN_CAP, EVAL_CAP = 3000, 1500
PAD, SIZE = 0.15, 128


def crop_square(img, box, pad=PAD, size=SIZE):
    from PIL import Image
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(img.width, x2 + pad * w), min(img.height, y2 + pad * h)
    c = img.crop((int(x1), int(y1), int(x2), int(y2)))
    s = max(c.width, c.height, 1)
    sq = Image.new("RGB", (s, s), (114, 114, 114))
    sq.paste(c, ((s - c.width) // 2, (s - c.height) // 2))
    return sq.resize((size, size))


def value_of(state, axis):
    st = (state or {}).get(axis) or {}
    if not st.get("applicable"):
        return None
    v = st.get("value")
    if v is None:
        return None
    return "1" if bool(v) else "0"


def main(argv=None):
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.expanduser("~/thor-data/kitchen_v3/FloorPlan1"))
    ap.add_argument("--out", default=os.path.expanduser("~/thor-data/kitchen_v3/state_cls"))
    a = ap.parse_args(argv)
    rng = random.Random(0)
    items = defaultdict(list)                       # (axis, split, value) -> [(img, box)]
    with open(os.path.join(a.store, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "img" not in r or r.get("quarantined"):
                continue
            for lab in r["labels"]:
                if lab.get("px", 0) < MIN_PX or not lab.get("state"):
                    continue
                for ax in AXES:
                    if ax in AXIS_TYPES and lab["type"] not in AXIS_TYPES[ax]:
                        continue
                    v = value_of(lab["state"], ax)
                    if v is not None:
                        items[(ax, r["split"], v)].append((r["split"], r["img"], tuple(lab["box"]), lab["type"]))
    summary = {}
    cache = {}
    for (ax, sp, v), lst in sorted(items.items()):
        rng.shuffle(lst)
        cap = TRAIN_CAP if sp == "train" else EVAL_CAP
        if sp == "train":
            keep = lst[:cap]
        else:
            keep = lst                               # natural frequency; capped jointly below
        items[(ax, sp, v)] = keep
    for ax in AXES:
        for sp in ("val", "test"):
            allv = [(v, it) for (a_, s_, v), lst in items.items() if a_ == ax and s_ == sp for it in lst]
            rng.shuffle(allv)
            allv = allv[:EVAL_CAP]                   # natural mix kept: one shared cap per axis and split
            byv = defaultdict(list)
            for v, it in allv:
                byv[v].append(it)
            for v in ("0", "1"):
                items[(ax, sp, v)] = byv.get(v, [])
    for (ax, sp, v), lst in sorted(items.items()):
        d = os.path.join(a.out, ax, sp, v)
        os.makedirs(d, exist_ok=True)
        types = Counter()
        for k, (split, img, box, typ) in enumerate(lst):
            p = os.path.join(a.store, "images", split, img + ".jpg")
            if p not in cache:
                if len(cache) > 64:
                    cache.clear()
                cache[p] = Image.open(p).convert("RGB")
            crop_square(cache[p], box).save(os.path.join(d, f"{img}_{k}.jpg"), quality=92)
            types[typ] += 1
        summary[f"{ax}/{sp}/{v}"] = {"n": len(lst), "types": dict(types)}
        print(f"{ax:7s} {sp:5s} {v}: {len(lst):5d}  {dict(types)}", flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
