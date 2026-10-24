"""Kitchen 2/3 state calibration and state TEST (A28 item 7), with sim/state_eval.py (Kitchen 1) unchanged.

  calib : BEFORE the freeze. Size floors from Arash's blind hand labels of that kitchen (state_review score.json; the Kitchen 1
          rolling-window rule state_eval.size_floors) with A28's correction: an axis with fewer than 8 labelled items gets NO
          floor and always abstains (encoded for the unchanged runtime perception.StateEstimator as thresholds {0: null,
          1: null} and an unreachable floor; reported as zero coverage). Then the VAL thresholds (state_eval.thresholds:
          >= 95 % precision with >= 20 accepted, else that value abstains). Writes sim/kitchen_v3/<scene>/state_calib.json
          (floors, thresholds, VAL report). TEST is not read.
  test  : AFTER the freeze (refused unless the kitchen's frozen_weights.sha256 lists state_calib.json with its current hash and,
          for FloorPlan15, the COCO sensitivity is frozen too, A38 item 4). Writes sim/kitchen_v3/<scene>/state_test.json.
Frames: the kitchen's main store, main protocol (as Kitchen 1), via a filtered view that also drops quarantined frames and the
held-out exclusions (heldout_exclusions.json) explicitly (state_eval.collect does not check them).

    python sim/kitchen_v3/k23/k23_state_calib.py calib --scene FloorPlan15      (WSL thor-env)
    python sim/kitchen_v3/k23/k23_state_calib.py test  --scene FloorPlan15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SIM)
UNREACHABLE_FLOOR = 10 ** 9            # px^2: no 640x640 box reaches it -> the axis always abstains
MIN_LABELS = 8                         # one rolling window (A28 item 7)


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def point(scene: str):
    import state_eval as SE
    store = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3", scene)
    view = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3", f"_state_eval_{scene}")
    os.makedirs(view, exist_ok=True)
    ex_p = os.path.join(HERE, scene, "heldout_exclusions.json")
    excl = {e["frame"] for e in json.load(open(ex_p, encoding="utf-8"))["entries"]} if os.path.exists(ex_p) else set()
    n = {"kept": 0, "quarantined": 0, "excluded": 0}
    with open(os.path.join(store, "frames.jsonl"), encoding="utf-8") as fi, \
            open(os.path.join(view, "frames.jsonl"), "w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            if "img" not in r:
                continue
            if r.get("quarantined"):
                n["quarantined"] += 1
                continue
            if r["img"] in excl:
                n["excluded"] += 1
                continue
            n["kept"] += 1
            fo.write(json.dumps(r) + "\n")
    link = os.path.join(view, "images")
    if not os.path.lexists(link):
        os.symlink(os.path.join(store, "images"), link)
    SE.SCENE, SE.STORE = scene, view
    SE.REVIEW = os.path.join(HERE, scene, "state_review")
    SE.OUT = os.path.join(SIM, "kitchen_v3", scene, "state_calib.json")
    n["view_manifest_sha256"] = sha_file(os.path.join(view, "frames.jsonl"))
    return SE, n


def load_models(SE, scene):
    import shutil
    import tempfile
    from ultralytics import YOLO

    def load(p):
        tmp = os.path.join(tempfile.gettempdir(), "k23se_" + os.path.basename(p))
        shutil.copyfile(p, tmp)
        return YOLO(tmp)
    det = load(os.path.join(SE.WEIGHTS, f"{scene}.yolo26s.pt"))
    cls = {ax: load(os.path.join(SE.WEIGHTS, f"{scene}.state_{ax}.pt")) for ax in SE.sd.AXES}
    return det, cls


def calib(scene: str) -> dict:
    SE, view = point(scene)
    score = os.path.join(SE.REVIEW, "score.json")
    if not os.path.exists(score):
        raise RuntimeError(f"no hand-label score {score}: Arash's labels come first (A28 item 7)")
    rows = json.load(open(score, encoding="utf-8"))["rows"]
    per_axis = {ax: sum(1 for x in rows if x["axis"] == ax) for ax in SE.sd.AXES}
    floors = SE.size_floors()
    abstain = sorted(ax for ax, k in per_axis.items() if k < MIN_LABELS)
    for ax in abstain:
        floors[ax] = UNREACHABLE_FLOOR
    det, cls = load_models(SE, scene)
    val = SE.collect("val", det, cls)
    th = SE.thresholds(val)
    for ax in abstain:
        th[ax] = {"0": None, "1": None}
    doc = {"scene": scene, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "floors_box_area": floors, "thresholds": th,
           "target_precision": SE.TARGET_PREC, "min_accept": SE.MIN_ACCEPT, "labels_per_axis": per_axis,
           "abstain_axes_too_few_labels": abstain, "val_report": SE.report(val, th, floors),
           "score_sha256": sha_file(score), "frames_view": view, "rule": "A28 item 7 (Kitchen 1 rules; <8 labels -> abstain)",
           "test": "not evaluated here (after the freeze: k23_state_calib.py test)"}
    if os.path.exists(SE.OUT):
        raise RuntimeError(f"{SE.OUT} exists (calibration is fitted once, before the freeze)")
    json.dump(doc, open(SE.OUT, "w", encoding="utf-8"), indent=1)
    return doc


def test(scene: str) -> dict:
    SE, view = point(scene)
    rec = os.path.join(SIM, "kitchen_v3", scene, "frozen_weights.sha256")
    meta = os.path.join(SIM, "kitchen_v3", scene, "frozen_weights.meta.json")
    if not os.path.exists(rec):
        raise RuntimeError("REFUSED: the kitchen is not frozen (A28 item 9)")
    if scene == "FloorPlan15" and json.load(open(meta, encoding="utf-8")).get("test_evaluation_blocked"):
        raise RuntimeError("REFUSED: the COCO sensitivity is not frozen yet (A38 item 4)")
    listed = {l.split(None, 1)[1].strip(): l.split()[0] for l in open(rec, encoding="utf-8") if l.strip()}
    key = f"sim/kitchen_v3/{scene}/state_calib.json"
    if listed.get(key) != sha_file(SE.OUT):
        raise RuntimeError("REFUSED: state_calib.json differs from the frozen record")
    c = json.load(open(SE.OUT, encoding="utf-8"))
    det, cls = load_models(SE, scene)
    t = SE.collect("test", det, cls)
    rep = SE.report(t, c["thresholds"], c["floors_box_area"])
    doc = {"scene": scene, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "test": rep, "frames_view": view,
           "calib_sha256": sha_file(SE.OUT), "frozen_record_sha256": sha_file(rec)}
    json.dump(doc, open(os.path.join(SIM, "kitchen_v3", scene, "state_test.json"), "w", encoding="utf-8"), indent=1)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("calib", "test"))
    ap.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    a = ap.parse_args(argv)
    try:
        d = calib(a.scene) if a.cmd == "calib" else test(a.scene)
    except RuntimeError as e:
        print(e)
        return 2
    print(json.dumps({k: d[k] for k in d if k in ("floors_box_area", "abstain_axes_too_few_labels", "test")}, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
