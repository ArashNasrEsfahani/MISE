"""Train and evaluate the per-kitchen TYPE detectors on the kitchen_v2c store (step 3, first training — Arash's go
2026-09-29, recipe reviewed by Codex/gpt-6-astra step3h).

Models (same data, same 150-epoch ceiling / patience 40, same state-safe augmentation, seed 0, selection on VAL):
    yolo26s   primary — YOLO26s, MuSGD pinned, NMS-free (end2end) head for every val / test / live prediction
    yolo11s   baseline — YOLO11s, SGD, conventional NMS
This is a MODEL + RECIPE comparison (MuSGD vs SGD), not an isolated architecture comparison.

Evaluation per model (reported separately, never pooled):
    test        main-protocol held-out TEST split (per-class AP50 / AP50-95 / P / R, confusion)
    test_close  the pre-declared close-view protocol TEST split (heldout-close-v1)
    live        150 FROZEN poses in the unrandomised kitchen (shared by both models, >= 0.3 m / 20 deg from every
                recorded training pose), render-guarded, ground truth from the SAME type-only labeller as training,
                detector fed BGR; AP at IoU .50:.05:.95

    python sim/train_kitchen.py train --scene FloorPlan1 --model yolo26s     # WSL, simulator stopped
    python sim/train_kitchen.py eval  --scene FloorPlan1 --model yolo26s
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3")
WEIGHTS = os.path.join(HERE, "weights", "v3")
REPORTS = os.path.join(HERE, "kitchen_v3")

MODELS = {
    "yolo26s": {"base": "yolo26s.pt", "optimizer": "MuSGD", "lr0": 0.01, "end2end": True},
    "yolo11s": {"base": "yolo11s.pt", "optimizer": "SGD", "lr0": 0.01, "end2end": None},
}
# 60 epochs, not 150 (2026-09-30): the 150-epoch ceiling was sized for the ~7,000-frame v2c train split (~66k
# iterations at batch 16). The v3 split is 16,299 frames = 1,019 iterations per epoch, so 60 epochs is the SAME number
# of gradient steps (~61k), at ~8 min per epoch on the RTX 4060 Laptop instead of 22 h per model. Measured: pure
# forward+backward 6.3 it/s, dataloader 48 it/s, real training ~2 it/s — the gap is the two-head assignment loss and
# per-step Python overhead, not idle hardware; deterministic off, torch.compile and other batch sizes did not help
# (batch 32 overflows the 8 GB card and drops to 0.24 it/s).
RECIPE = {"epochs": 60, "patience": 20, "imgsz": 640, "batch": 16, "seed": 0}
EVAL = {"conf": 0.001, "iou": 0.7, "max_det": 300}
LIVE_POSES = 150
IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _weights(scene, model):
    return os.path.join(WEIGHTS, f"{scene}.{model}.pt")


def _yolo(path):
    """ultralytics strips apostrophes from absolute paths — load from a clean temp copy."""
    from ultralytics import YOLO
    tmp = os.path.join(tempfile.gettempdir(), "tk_" + os.path.basename(path))
    shutil.copyfile(path, tmp)
    return YOLO(tmp)


def _keep_cvd(fn):
    """ultralytics sets CUDA_VISIBLE_DEVICES and never restores it (breaks AI2-THOR afterwards) — restore it."""
    def wrap(*a, **k):
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            return fn(*a, **k)
        finally:
            if before is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = before
    return wrap


# ======================================================================================================================
# Train.
# ======================================================================================================================
def train(scene: str, model: str) -> dict:
    import gen_dataset
    import torch
    m = MODELS[model]
    ds = os.path.join(ROOT, scene, "yolo_primary")
    if not os.path.exists(os.path.join(ds, "data.yaml")):
        raise RuntimeError(f"{scene}: no yolo_primary/data.yaml — run the `yolo` stage (detection gate) first")
    out = _weights(scene, model)
    os.makedirs(WEIGHTS, exist_ok=True)
    if os.path.exists(out):
        arch = os.path.join(WEIGHTS, "archive")
        os.makedirs(arch, exist_ok=True)
        shutil.copyfile(out, os.path.join(arch, f"{scene}.{model}.{time.strftime('%Y%m%d_%H%M%S')}.pt"))
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    _log(f"{scene} {model}: training {m['base']} ({m['optimizer']}, lr0 {m['lr0']}) {RECIPE}")
    t0 = time.time()
    ckpt = os.path.join(os.path.dirname(os.path.dirname(ds)), "_ckpt", f"{scene}.{model}")
    if os.path.exists(os.path.join(ckpt, "last.pt")) and \
            torch.load(os.path.join(ckpt, "last.pt"), map_location="cpu", weights_only=False).get("epoch", -1) < 0:
        shutil.rmtree(ckpt, ignore_errors=True)   # a FINISHED run (ultralytics strips it: epoch -1) — nothing to resume
    if os.path.exists(os.path.join(ckpt, "last.pt")):
        res = _keep_cvd(_resume)(scene, model, ds, out, ckpt)
    else:
        res = _keep_cvd(gen_dataset.train_scene)(
            scene, ds, out, base=m["base"], epochs=RECIPE["epochs"], imgsz=RECIPE["imgsz"], batch=RECIPE["batch"],
            patience=RECIPE["patience"], seed=RECIPE["seed"], run_name=f"{scene}.{model}",
            overrides={"optimizer": m["optimizer"], "lr0": m["lr0"]})
    shutil.rmtree(ckpt, ignore_errors=True)      # finished: a later run of this model starts fresh
    res["hours"] = round((time.time() - t0) / 3600.0, 2)
    # what ultralytics ACTUALLY used (Astra: verify the resolved optimizer / lr, report real epochs)
    run_dir = os.path.join(ds, "runs", f"{scene}.{model}")
    args_yaml = os.path.join(run_dir, "args.yaml")
    res["resolved_args"] = open(args_yaml, encoding="utf-8").read() if os.path.exists(args_yaml) else None
    csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(csv):
        lines = open(csv, encoding="utf-8").read().strip().splitlines()
        res["epochs_run"] = len(lines) - 1
    res["model"], res["recipe_fixed"] = model, dict(RECIPE, **m)
    _dump(os.path.join(REPORTS, scene, f"train_{model}.json"), res)
    _log(f"{scene} {model}: done in {res['hours']} h, {res.get('epochs_run')} epochs -> {res.get('weights_path')}")
    return res


def _resume(scene: str, model: str, ds: str, out: str, ckpt: str) -> dict:
    """Continue a stopped run from the checkpoint ``kitchen_v3/ckpt_backup.sh`` copied out of /tmp.

    ultralytics resumes from the args stored in ``last.pt`` — including its save dir, the /tmp run folder that a WSL
    restart emptied — so the checkpoint (and the best-so-far weights, which a resume only overwrites on improvement)
    is put back there first. Epoch, optimizer state, EMA and best fitness all come from the checkpoint."""
    from ultralytics import YOLO
    project = open(os.path.join(ckpt, "project.txt"), encoding="utf-8").read().strip()
    wdir = os.path.join(project, f"{scene}.{model}", "weights")
    os.makedirs(wdir, exist_ok=True)
    for f in ("last.pt", "best.pt"):
        if os.path.exists(os.path.join(ckpt, f)):
            shutil.copyfile(os.path.join(ckpt, f), os.path.join(wdir, f))
    if os.path.exists(os.path.join(ckpt, "results.csv")):     # per-epoch history before the stop
        shutil.copyfile(os.path.join(ckpt, "results.csv"), os.path.join(os.path.dirname(wdir), "results.csv"))
    _log(f"{scene} {model}: RESUMING from {ckpt}/last.pt (run dir {wdir})")
    y = YOLO(os.path.join(wdir, "last.pt"))
    y.train(resume=True)
    best = os.path.join(wdir, "best.pt")
    shutil.copyfile(best, out)
    dest = os.path.join(ds, "runs", f"{scene}.{model}")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.dirname(wdir), dest, ignore=shutil.ignore_patterns("*.pt"))
    return {"weights_path": out, "resumed_from": ckpt}


# ======================================================================================================================
# Evaluate on stored splits.
# ======================================================================================================================
@_keep_cvd
def split_metrics(model_obj, data_yaml: str, split: str, model: str) -> dict:
    kw = dict(data=data_yaml, split=split, imgsz=RECIPE["imgsz"], batch=16, plots=True, verbose=False, **EVAL)
    if MODELS[model]["end2end"] is not None:
        kw["end2end"] = MODELS[model]["end2end"]
    kw["project"] = tempfile.mkdtemp(prefix="tk_val_")
    m = model_obj.val(**kw)
    names = dict(model_obj.names)
    idx = [int(i) for i in m.ap_class_index]
    per = {names[c]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i]),
                      "p": float(m.box.p[i]), "r": float(m.box.r[i])} for i, c in enumerate(idx)}
    conf = []
    try:
        mat = m.confusion_matrix.matrix                   # [predicted, true]; last row/col = background
        nc = len(names)
        for t in range(nc):
            col = float(mat[:, t].sum())
            for p in range(nc):
                if p != t and col > 0 and mat[p, t] / col > 0.02:
                    conf.append({"true": names[t], "pred": names[p], "frac": round(float(mat[p, t] / col), 3)})
    except Exception as exc:
        conf = [{"error": str(exc)}]
    return {"split": split, "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
            "recall": float(m.box.mr), "classes_scored": len(per), "per_class": per,
            "confusion": sorted(conf, key=lambda x: -x.get("frac", 0))[:40]}


# ======================================================================================================================
# Live evaluation against the simulator.
# ======================================================================================================================
def frozen_poses(scene: str, agent) -> list:
    """150 poses, drawn once per kitchen and saved; excludes anything within 0.3 m AND 20 deg of a recorded
    training camera pose, so the live test is not a replay of training views."""
    path = os.path.join(ROOT, scene, "live_eval_poses.json")
    if os.path.exists(path):
        saved = json.load(open(path))
        # a smoke run (--live-poses 10) on the real root once froze only 10 poses and the real evaluation reused them
        # (2026-10-01); a list shorter than asked for is regenerated — same seed, so it is a superset of the old one
        if len(saved) >= LIVE_POSES:
            return saved[:LIVE_POSES]
    train_poses = []
    with open(os.path.join(ROOT, scene, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("split") == "train" and r.get("camera"):
                a = r["camera"]["agent"]
                train_poses.append((a["x"], a["z"], r["camera"]["yaw"]))
    rng = random.Random(20260929)
    pos = agent.reachable_positions()
    out, tries = [], 0
    while len(out) < LIVE_POSES and tries < 20000:
        tries += 1
        p = rng.choice(pos)
        yaw = round(rng.uniform(0, 360), 1)
        near = any(math.hypot(p["x"] - x, p["z"] - z) < 0.3 and abs((yaw - y + 180) % 360 - 180) < 20
                   for x, z, y in train_poses)
        if near:
            continue
        out.append({"x": p["x"], "y": p["y"], "z": p["z"], "yaw": yaw, "horizon": round(rng.uniform(-10, 40), 1),
                    "standing": rng.random() > 0.2})
    _dump(path, out)
    return out


COVER_POSES, COVER_MIN_LABELS, COVER_CLASS_FLOOR = 300, 5, 10


def coverage_poses(scene: str, agent, train_poses=None) -> list:
    """The COVERAGE live set (Astra step3j, 2026-10-03). The first live set (``frozen_poses``) excluded everything near
    a training camera and, in a large room, ended up facing bare walls: 118 GT boxes over 150 poses, 9 classes, 81 of
    them Floor. It is kept as the REALISTIC set (random looking, empty views included). This second set is a coverage /
    stress benchmark, reported separately: a pose is kept only if it shows >= 5 labelled objects, is not a near
    duplicate of a training camera (position < 0.3 m AND yaw < 20 deg AND horizon < 10 deg — joint, not "far from
    every training pose"), and while any class has fewer than 10 sightings, only poses that add one are kept.
    Drawn once, rendered through the render guard, saved; the SAME set scores every model."""
    import build_kitchen_data as bk
    path = os.path.join(ROOT, scene, "live_cover_poses.json")
    if os.path.exists(path):
        saved = json.load(open(path))
        if len(saved) >= COVER_POSES:
            return saved[:COVER_POSES]
    if train_poses is None:
        train_poses = []
        with open(os.path.join(ROOT, scene, "frames.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("split") == "train" and r.get("camera"):
                    a = r["camera"]["agent"]
                    train_poses.append((a["x"], a["z"], r["camera"]["yaw"], r["camera"]["horizon"]))
    rng = random.Random(20261003)
    pos = agent.reachable_positions()
    objs = [o for o in agent.objects() if o.get("position")]
    seen, out, tries = {}, [], 0
    while len(out) < COVER_POSES and tries < 20000:
        tries += 1
        p = rng.choice(pos)
        if rng.random() < 0.7 and objs:                    # mostly look AT something, sometimes anywhere
            o = rng.choice(objs)["position"]
            yaw = (math.degrees(math.atan2(o["x"] - p["x"], o["z"] - p["z"])) + rng.gauss(0, 15)) % 360
        else:
            yaw = rng.uniform(0, 360)
        yaw, hor, standing = round(yaw, 1), round(rng.uniform(0, 40), 1), rng.random() > 0.2
        # NO geometric-novelty filter (Astra step3k): 16k training cameras in one kitchen cover almost every
        # content-facing pose, so the filter left 23 usable poses of 8,000. Instead each pose records its distance
        # to the nearest training camera (position, wrapped yaw, horizon of the SAME reference camera).
        near = min(train_poses, key=lambda t: math.hypot(p["x"] - t[0], p["z"] - t[1])
                   + abs((yaw - t[2] + 180) % 360 - 180) / 90.0 + abs(hor - t[3]) / 45.0)
        agent.teleport_full(p["x"], p["y"], p["z"], yaw, horizon=hor, standing=standing)
        if not agent.success:
            continue
        types = [lab["type"] for lab in bk.label_frame(agent)]
        if len(types) < COVER_MIN_LABELS:
            continue
        short = [t for t in set(types) if seen.get(t, 0) < COVER_CLASS_FLOOR]
        under = [t for t, n in seen.items() if n < COVER_CLASS_FLOOR]
        if not short and under and len(out) > COVER_POSES // 2:
            continue                                        # second half: prefer poses that feed short classes
        for t in types:
            seen[t] = seen.get(t, 0) + 1
        out.append({"x": p["x"], "y": p["y"], "z": p["z"], "yaw": yaw, "horizon": hor, "standing": standing,
                    "n_labels": len(types),
                    "nearest_train": {"pos_m": round(math.hypot(p["x"] - near[0], p["z"] - near[1]), 2),
                                      "yaw_deg": round(abs((yaw - near[2] + 180) % 360 - 180), 1),
                                      "horizon_deg": round(abs(hor - near[3]), 1)}})
    _dump(path, out)
    return out


def live_eval(scene: str, model: str, which: str = "realistic") -> dict:
    import numpy as np
    import build_kitchen_data as bk
    import oracle_eval
    m = MODELS[model]
    det = _yolo(_weights(scene, model))
    agent = bk.make_agent(scene)
    try:
        agent.reset(scene)
        poses = frozen_poses(scene, agent) if which == "realistic" else coverage_poses(scene, agent)
        tallies = {t: oracle_eval.Tally(iou_thr=t) for t in IOUS}
        skipped = 0
        for p in poses:
            agent.teleport_full(p["x"], p["y"], p["z"], p["yaw"], horizon=p["horizon"], standing=p["standing"])
            ok, _n = bk.render_guard(agent)
            if not agent.success or not ok:
                skipped += 1
                continue
            gt = [(lab["type"], tuple(lab["box"])) for lab in bk.label_frame(agent)]
            bgr = np.ascontiguousarray(np.asarray(agent.frame)[..., ::-1])
            kw = dict(conf=EVAL["conf"], iou=EVAL["iou"], max_det=EVAL["max_det"], verbose=False, imgsz=RECIPE["imgsz"])
            if m["end2end"] is not None:
                kw["end2end"] = m["end2end"]
            r = _keep_cvd(det)(bgr, **kw)[0]
            preds = [(det.names[int(b.cls[0])], tuple(int(round(v)) for v in b.xyxy[0].tolist()), float(b.conf[0]))
                     for b in r.boxes]
            for t in tallies.values():
                t.add(gt, preds)
        summ = {t: tallies[t].summary() for t in IOUS}
        per = {}
        for row in summ[0.5]["rows"] if isinstance(summ[0.5].get("rows"), list) else []:
            c = row["cls"]
            aps = [next((x["ap"] for x in summ[t]["rows"] if x["cls"] == c), None) for t in IOUS]
            per[c] = {"ap50": row["ap"], "ap50_95": (sum(a for a in aps if a is not None) / len(aps)
                                                     if all(a is not None for a in aps) else None),
                      "recall50": row["recall"], "gt": row["gt"]}
        ap50s = [v["ap50"] for v in per.values() if v["ap50"] is not None]
        ap5095 = [v["ap50_95"] for v in per.values() if v["ap50_95"] is not None]
        floor_met = sorted(c for c, v in per.items() if v["gt"] >= COVER_CLASS_FLOOR)
        return {"set": which, "poses": len(poses), "skipped": skipped, "classes_at_floor": floor_met,
                "mAP50_at_floor": (sum(per[c]["ap50"] for c in floor_met if per[c]["ap50"] is not None)
                                   / max(1, sum(1 for c in floor_met if per[c]["ap50"] is not None))),
                "gt_boxes": sum(v["gt"] for v in per.values()), "classes_with_gt": sum(1 for v in per.values() if v["gt"]), "mAP50": sum(ap50s) / len(ap50s) if ap50s else None,
                "mAP50_95": sum(ap5095) / len(ap5095) if ap5095 else None, "classes_scored": len(per),
                "per_class": per}
    finally:
        agent.stop()


def live_only(scene: str, model: str) -> dict:
    """Re-score ONLY the two live sets into an existing report (the stored split metrics are left untouched)."""
    path = os.path.join(REPORTS, scene, f"eval_{model}.json")
    rep = json.load(open(path))
    rep["live_old_invalid"] = rep.pop("live", None)
    rep["live"] = live_eval(scene, model, "realistic")
    rep["live_cover"] = live_eval(scene, model, "cover")
    _dump(path, rep)
    for k in ("live", "live_cover"):
        lv = rep[k]
        _log(f"{scene} {model}: {k} mAP50 {lv['mAP50'] or float('nan'):.3f} / 50-95 {lv['mAP50_95'] or float('nan'):.3f} "
             f"({lv['poses'] - lv['skipped']} poses, {lv['gt_boxes']} GT boxes, {lv['classes_with_gt']} classes)")
    return rep


def evaluate(scene: str, model: str, live: bool = True) -> dict:
    ds = os.path.join(ROOT, scene, "yolo_primary")
    model_obj = _yolo(_weights(scene, model))
    rep = {"scene": scene, "model": model, "eval": EVAL, "end2end": MODELS[model]["end2end"]}
    rep["val"] = {k: v for k, v in split_metrics(model_obj, os.path.join(ds, "data.yaml"), "val", model).items()
                  if k != "per_class"}
    rep["test"] = split_metrics(model_obj, os.path.join(ds, "data.yaml"), "test", model)
    try:
        rep["test_close"] = split_metrics(model_obj, os.path.join(ds, "data_close.yaml"), "test", model)
    except Exception as exc:
        rep["test_close"] = {"error": str(exc)}
    if live:
        rep["live"] = live_eval(scene, model, "realistic")
        rep["live_cover"] = live_eval(scene, model, "cover")
    rep["targets"] = {"macro_ap50": 0.90, "macro_ap50_95": 0.70}
    _dump(os.path.join(REPORTS, scene, f"eval_{model}.json"), rep)
    lv = rep.get("live") or {}
    _log(f"{scene} {model}: TEST mAP50 {rep['test']['mAP50']:.3f} / 50-95 {rep['test']['mAP50_95']:.3f} | "
         f"CLOSE {rep['test_close'].get('mAP50', float('nan')):.3f} | LIVE mAP50 {lv.get('mAP50') or float('nan'):.3f}"
         f" / 50-95 {lv.get('mAP50_95') or float('nan'):.3f} ({lv.get('poses', 0) - lv.get('skipped', 0)} poses)")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("train", "eval", "both", "live"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--root", default=None, help="data root (smoke tests); default ~/thor-data/kitchen_v2c")
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch ceiling (smoke tests only)")
    ap.add_argument("--live-poses", type=int, default=None, help="override the live pose count (smoke tests only)")
    a = ap.parse_args(argv)
    global ROOT, LIVE_POSES, WEIGHTS, REPORTS
    if a.root:
        ROOT = a.root
        WEIGHTS = os.path.join(a.root, "_weights")          # smoke weights never land in sim/weights
        REPORTS = os.path.join(a.root, "_reports")
    if a.epochs:
        RECIPE["epochs"], RECIPE["patience"] = a.epochs, a.epochs
    if a.live_poses:
        LIVE_POSES = a.live_poses
    if a.cmd in ("train", "both"):
        train(a.scene, a.model)
    if a.cmd == "live":
        live_only(a.scene, a.model)
    if a.cmd in ("eval", "both"):
        evaluate(a.scene, a.model, live=not a.no_live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
