"""Hand-label a stratified sample of object crops: "can you SEE this object's state?" — the ground truth that
calibrates the state estimator's *unknown* answer (option b, docs/kitchen-datasets.md §6 item 2).

Why: the simulator always knows an object's state, but not whether the state is VISIBLE in a given photo. A mug
seen side-on cannot show its coffee; a far cabinet door may be too small to show it is ajar. Training the state
estimator to answer "filled" on a crop where no liquid is visible teaches it to guess. Paired renders (flip one
state, re-render) are evidence; these hand labels are the check (Codex/gpt-6-astra review, 2026-09-28).

Design:

* **Blind.** The page never contains the simulator's answer. The reviewer says what they SEE ("open / closed /
  can't tell"; for liquids "empty / water / coffee / wine / can't tell"). An item counts as *observable* when the
  reviewer answered AND was right; a confident wrong answer is recorded separately. Truth lives in ``truth.json``,
  which the page never loads.
* **Stratified.** Items are spread evenly over (axis, true value, object size in pixels, distance), including
  deliberately small crops, so the size / angle thresholds can be calibrated rather than guessed.
* **Development data only.** Crops come from TRAIN episodes; val/test stay untouched (no threshold is ever tuned on
  them).
* **Self-contained.** ``make`` writes a folder: ``index.html`` (double-click it), ``items.js``, ``crops/``,
  ``truth.json``. Answers autosave in the browser and export to ``answers.json``; ``score`` joins them with the truth.

    python sim/state_review.py make --scenes FloorPlan1,FloorPlan6 --n 300 \
        --root //wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v2b --out <folder>
    python sim/state_review.py score --dir <folder> --answers <folder>/answers.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TEMPLATE = os.path.join(HERE, "review", "state_review.html")

#: axis -> (question, [(answer_value, button label)]). Answer values are compared with the simulator's truth.
QUESTIONS = {
    "open":   ("Is it open or closed?", [("1", "Open"), ("0", "Closed")]),
    "on":     ("Is it on or off?", [("1", "On"), ("0", "Off")]),
    "sliced": ("Is it sliced or whole?", [("1", "Sliced"), ("0", "Whole")]),
    "broken": ("Is it cracked or whole?", [("1", "Cracked"), ("0", "Whole")]),
    "cooked": ("Is it cooked or raw?", [("1", "Cooked"), ("0", "Raw")]),
    "dirty":  ("Is it dirty or clean?", [("1", "Dirty"), ("0", "Clean")]),
    "filled": ("Is there liquid in it?", [("empty", "Empty"), ("water", "Water"), ("coffee", "Coffee"),
                                          ("wine", "Wine")]),
}
CANT = ("cant", "Can't tell")

PX_BINS = (0, 900, 3000, 10000, 10 ** 9)          # mask pixels at 640x640; 900 is the capture's state_min_px
DIST_BINS = (0.0, 1.0, 2.0, 3.5, 99.0)            # metres


def _bin(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def truth_of(axis: str, state: dict):
    """The answer value the simulator says is correct (``'1'``/``'0'``, or a liquid / ``'empty'``)."""
    if axis == "filled":
        if not (state.get("filled") or {}).get("value"):
            return "empty"
        return (state.get("liquid") or {}).get("value") or "water"
    return "1" if (state.get(axis) or {}).get("value") else "0"


def candidates(frames, scene: str):
    """Every (frame, object, axis) a reviewer could be asked about, from TRAIN frames only."""
    out = []
    for r in frames:
        if r.get("split") != "train" or r.get("quarantined"):
            continue
        for lab in r["labels"]:
            st = lab.get("state") or {}
            for axis in QUESTIONS:
                if not (st.get(axis) or {}).get("applicable"):
                    continue
                if axis != "broken" and (st.get("broken") or {}).get("value"):
                    continue                          # a cracked egg is asked only about cracked/cooked
                out.append({"scene": scene, "img": r["img"], "split": r["split"], "episode": r["episode"],
                            "kind": r.get("kind"), "oid": lab["oid"], "type": lab["type"], "axis": axis,
                            "truth": truth_of(axis, st), "box": lab["box"], "px": lab["px"],
                            "dist": lab.get("dist", 0.0), "rim_elev": lab.get("rim_elev"),
                            "held": lab.get("held", False)})
    return out


def stratified(cands, n: int, seed: int = 0) -> list:
    """Round-robin over strata (axis, truth, size bin, distance bin): every stratum gets an item before any gets a
    second, so rare combinations (a small far open drawer) are represented. At most one item per (frame, object)
    and a spread over episodes."""
    rng = random.Random(seed)
    strata = {}
    for c in cands:
        k = (c["axis"], c["truth"], _bin(c["px"], PX_BINS), _bin(c["dist"], DIST_BINS))
        strata.setdefault(k, []).append(c)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    rng.shuffle(keys)
    picked, used, ep_count = [], set(), {}
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if len(picked) >= n:
                break
            pool = strata[k]
            # prefer an episode that has contributed least so far
            pool.sort(key=lambda c: ep_count.get((c["scene"], c["episode"]), 0))
            while pool:
                c = pool.pop(0)
                uid = (c["scene"], c["img"], c["oid"])
                if uid in used:
                    continue
                used.add(uid)
                ep_count[(c["scene"], c["episode"])] = ep_count.get((c["scene"], c["episode"]), 0) + 1
                picked.append(c)
                break
    rng.shuffle(picked)
    return picked


def make_crop(src_img: str, box, out_path: str, ctx_path: str) -> None:
    """A padded, aspect-preserving crop with the object outlined, plus a small full-frame context image."""
    from PIL import Image, ImageDraw
    im = Image.open(src_img).convert("RGB")
    W, H = im.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    side = max(bw, bh) * 1.5
    side = max(side, 96)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    L, T = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    R, B = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = im.crop((L, T, R, B))
    scale = 360.0 / max(crop.size)
    crop = crop.resize((max(1, int(crop.size[0] * scale)), max(1, int(crop.size[1] * scale))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    d.rectangle([(x1 - L) * scale - 2, (y1 - T) * scale - 2, (x2 - L) * scale + 2, (y2 - T) * scale + 2],
                outline=(255, 214, 0), width=2)
    crop.save(out_path, quality=92)
    ctx = im.copy()
    dc = ImageDraw.Draw(ctx)
    dc.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(255, 214, 0), width=4)
    ctx.thumbnail((240, 240))
    ctx.save(ctx_path, quality=85)


def cmd_make(a) -> int:
    scenes = [s for s in a.scenes.split(",") if s]
    cands = []
    for s in scenes:
        path = os.path.join(a.root, s, "frames.jsonl")
        if not os.path.exists(path):
            print(f"  skip {s}: no {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            frames = [json.loads(l) for l in fh if '"img"' in l]
        cands += candidates(frames, s)
        print(f"  {s}: {len(frames)} frames")
    if not cands:
        print("no candidates — has the capture run?")
        return 2
    picked = stratified(cands, a.n, a.seed)
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    items, truth = [], {}
    for i, c in enumerate(picked):
        iid = f"q{i:04d}"
        src = os.path.join(a.root, c["scene"], "images", c["split"], c["img"] + ".jpg")
        make_crop(src, c["box"], os.path.join(a.out, "crops", iid + ".jpg"),
                  os.path.join(a.out, "crops", iid + "_ctx.jpg"))
        q, opts = QUESTIONS[c["axis"]]
        items.append({"id": iid, "type": c["type"], "axis": c["axis"], "question": q,
                      "options": [list(o) for o in opts] + [list(CANT)]})
        truth[iid] = c
    with open(os.path.join(a.out, "items.js"), "w", encoding="utf-8") as fh:
        fh.write("window.REVIEW = " + json.dumps({"items": items, "created": a.stamp, "scenes": scenes,
                                                  "n": len(items)}, ensure_ascii=False) + ";\n")
    with open(os.path.join(a.out, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": "NOT loaded by the page — keeps the review blind", "items": truth,
                   "px_bins": PX_BINS, "dist_bins": DIST_BINS, "seed": a.seed}, fh, indent=1)
    shutil.copyfile(TEMPLATE, os.path.join(a.out, "index.html"))
    per = {}
    for c in picked:
        per[c["axis"]] = per.get(c["axis"], 0) + 1
    print(f"wrote {len(items)} items to {a.out}  (per axis: {per}) — open index.html")
    return 0


def score(truth: dict, answers: dict) -> dict:
    """Per axis and size/distance bin: answered-and-right (observable), answered-but-wrong (confident error),
    can't-tell (unobservable), unanswered."""
    rows, table = [], {}
    for iid, c in truth.items():
        a = answers.get(iid)
        if a is None:
            outcome = "unanswered"
        elif a == "cant":
            outcome = "cant"
        elif a == c["truth"]:
            outcome = "right"
        else:
            outcome = "wrong"
        rows.append({**c, "id": iid, "answer": a, "outcome": outcome})
        for key in (("axis", c["axis"]), ("axis_px", f"{c['axis']}|px{_bin(c['px'], PX_BINS)}"),
                    ("axis_dist", f"{c['axis']}|d{_bin(c['dist'], DIST_BINS)}")):
            t = table.setdefault(key[0], {}).setdefault(key[1], {"right": 0, "wrong": 0, "cant": 0,
                                                                  "unanswered": 0})
            t[outcome] += 1
    return {"rows": rows, "table": table}


def cmd_score(a) -> int:
    with open(os.path.join(a.dir, "truth.json"), encoding="utf-8") as fh:
        truth = json.load(fh)["items"]
    with open(a.answers, encoding="utf-8") as fh:
        ans = json.load(fh)
    answers = ans.get("answers", ans)
    res = score(truth, answers)
    for group in ("axis", "axis_px", "axis_dist"):
        print(f"\n{group}:")
        for k, t in sorted(res["table"][group].items()):
            tot = sum(t.values()) - t["unanswered"]
            vis = t["right"] / tot if tot else float("nan")
            print(f"  {k:18s} observable {t['right']:3d}  wrong {t['wrong']:3d}  can't-tell {t['cant']:3d}  "
                  f"unanswered {t['unanswered']:3d}  -> visible rate {vis:.2f}")
    out = os.path.join(a.dir, "score.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


def main(argv=None) -> int:
    import time
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--scenes", default="FloorPlan1,FloorPlan6,FloorPlan8,FloorPlan15,FloorPlan17")
    m.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v2b"))
    m.add_argument("--out", required=True)
    m.add_argument("--n", type=int, default=300)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--answers", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        a.stamp = time.strftime("%Y-%m-%d %H:%M")
        return cmd_make(a)
    return cmd_score(a)


if __name__ == "__main__":
    raise SystemExit(main())
