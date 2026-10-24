"""Step 3, second model: the FloorPlan1 AMBIGUITY (specialist) detector, its control, and the fused evaluation.

Frozen in docs/ambiguity-spec.md (Astra step3j-m). Commands (WSL, simulator stopped):

    python sim/train_amb.py lcurve    # YOLO26m on the nested 10 / 20 / 40 train-episode subsets (VAL only decides)
    python sim/train_amb.py control   # the primary (YOLO26s) fine-tuned on primary-train + amb-train, 49 labels
    python sim/train_amb.py eval      # primary vs primary+specialist (strict replacement) vs control, on amb-test and
                                      # primary-test, low-conf AP + operating P/R + latency + paired episode bootstrap
    python sim/train_amb.py holdout   # ONCE, after everything is frozen: the sealed amb holdout

Every training run is resumable: kitchen_v3/ckpt_backup.sh copies /tmp/yolo_runs_*/<run>/weights out every 2 min and a
rerun of the same command resumes the unfinished run from its last epoch.
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

import kitchen_data as kd  # noqa: E402

SCENE = "FloorPlan1"
AMB = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3_amb", SCENE)
PRIM = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3", SCENE, "yolo_primary")
CKPT = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3", "_ckpt")
WEIGHTS = os.path.join(HERE, "weights", "v3")
REPORTS = os.path.join(HERE, "kitchen_v3", SCENE)
PRIMARY_W = os.path.join(WEIGHTS, f"{SCENE}.yolo26s.pt")
SPEC = {"base": "yolo26m.pt", "optimizer": "MuSGD", "lr0": 0.01, "epochs": 60, "patience": 20, "imgsz": 640,
        "batch": 8, "seed": 0}       # batch 8: YOLO26m at 640 does not fit 16 on the 8 GB card next to nothing else
CONTROL = {"optimizer": "MuSGD", "lr0": 0.002, "epochs": 20, "patience": 20, "imgsz": 640, "batch": 16, "seed": 0}
LCURVE = (10, 20, 39)          # 39: train episode 3024 failed after six render redos (logged)
EVAL = {"conf": 0.001, "iou": 0.7, "max_det": 300}
IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
TARGETS = list(kd.AMB_TARGETS)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _clean_copy(path):
    """ultralytics strips apostrophes from absolute paths: load weights from a clean temp copy."""
    tmp = os.path.join(tempfile.gettempdir(), "amb_" + os.path.basename(path))
    shutil.copyfile(path, tmp)
    return tmp


def _episode(img_path):
    b = os.path.basename(img_path)
    return int(b.split("_e")[1].split("_")[0])


# ----------------------------------------------------------------------------------------------------------------------
# Training (with resume)
# ----------------------------------------------------------------------------------------------------------------------
def _train(run, data_yaml, base, recipe, out):
    import yolo_recipe
    from ultralytics import YOLO
    ck = os.path.join(CKPT, run)
    project = "/tmp/yolo_runs_amb"
    wdir = os.path.join(project, run, "weights")
    if os.path.exists(os.path.join(ck, "last.pt")):
        import torch
        if torch.load(os.path.join(ck, "last.pt"), map_location="cpu", weights_only=False).get("epoch", -1) >= 0:
            os.makedirs(wdir, exist_ok=True)
            for f in ("last.pt", "best.pt"):
                if os.path.exists(os.path.join(ck, f)):
                    shutil.copyfile(os.path.join(ck, f), os.path.join(wdir, f))
            if os.path.exists(os.path.join(ck, "results.csv")):
                shutil.copyfile(os.path.join(ck, "results.csv"), os.path.join(project, run, "results.csv"))
            _log(f"{run}: RESUMING from {ck}/last.pt")
            YOLO(os.path.join(wdir, "last.pt")).train(resume=True)
            return _finish(run, project, out)
    kw = yolo_recipe.train_kwargs(n_images=0, epochs=recipe["epochs"], imgsz=recipe["imgsz"], batch=recipe["batch"],
                                  patience=recipe["patience"], seed=recipe["seed"],
                                  overrides={"optimizer": recipe["optimizer"], "lr0": recipe["lr0"]})
    _log(f"{run}: training {base} {recipe} on {data_yaml}")
    m = YOLO(base if os.path.isabs(base) else yolo_recipe.checkpoint_path(base))
    m.train(data=data_yaml, device=0, project=project, name=run, exist_ok=True, **kw)
    return _finish(run, project, out)


def _finish(run, project, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(os.path.join(project, run, "weights", "best.pt"), out)
    dest = os.path.join(REPORTS, "runs", run)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.join(project, run), dest, ignore=shutil.ignore_patterns("*.pt", "*.jpg"))
    shutil.rmtree(os.path.join(CKPT, run), ignore_errors=True)
    rows = open(os.path.join(project, run, "results.csv")).read().strip().splitlines()
    _log(f"SPECIALIST/CONTROL run {run} done: {len(rows) - 1} epochs -> {out}")
    return out


def lcurve():
    """Nested subsets by EPISODE (first 10 / 20 / all 40 train episodes); every run selects on the full amb VAL."""
    ds = os.path.join(AMB, "yolo_amb")
    train = [l.strip() for l in open(os.path.join(ds, "train.txt")) if l.strip()]
    eps = sorted({_episode(p) for p in train})
    rep = {}
    for n in LCURVE:
        keep = set(eps[:n])
        sub = [p for p in train if _episode(p) in keep]
        lst = os.path.join(ds, f"train_ep{n}.txt")
        with open(lst, "w") as fh:
            fh.write("".join(p + "\n" for p in sub))
        yml = os.path.join(ds, f"data_ep{n}.yaml")
        body = open(os.path.join(ds, "data.yaml")).read().replace("train: train.txt", f"train: train_ep{n}.txt")
        with open(yml, "w") as fh:
            fh.write(body)
        out = os.path.join(WEIGHTS, f"{SCENE}.amb26m.ep{n}.pt")
        if not os.path.exists(out):
            _train(f"{SCENE}.amb26m.ep{n}", yml, SPEC["base"], SPEC, out)
        rep[n] = {"episodes": n, "images": len(sub), "val": _val(out, os.path.join(ds, "data.yaml"), "val")}
        _log(f"LCURVE {n} episodes ({len(sub)} images): VAL mAP50 {rep[n]['val']['mAP50']:.3f} / 50-95 "
             f"{rep[n]['val']['mAP50_95']:.3f}")
        _dump(os.path.join(REPORTS, "amb_lcurve.json"), rep)
    shutil.copyfile(os.path.join(WEIGHTS, f"{SCENE}.amb26m.ep{LCURVE[-1]}.pt"), os.path.join(WEIGHTS, f"{SCENE}.amb26m.pt"))
    return rep


def control():
    """The primary fine-tuned on primary-train + amb-train (all 49 labels, original-data replay), selected on
    primary-val + amb-val — tests 'just add the data to one model' against the specialist."""
    out = os.path.join(WEIGHTS, f"{SCENE}.yolo26s.control.pt")
    if not os.path.exists(out):
        _train(f"{SCENE}.yolo26s.control", os.path.join(AMB, "yolo_amb49", "data_plus.yaml"), _clean_copy(PRIMARY_W),
               CONTROL, out)
    rep = {"val_plus": _val(out, os.path.join(AMB, "yolo_amb49", "data_plus.yaml"), "val")}
    _log(f"CONTROL VAL(primary+amb) mAP50 {rep['val_plus']['mAP50']:.3f} / 50-95 {rep['val_plus']['mAP50_95']:.3f}")
    _dump(os.path.join(REPORTS, "amb_control.json"), rep)
    return rep


def _val(weights, data_yaml, split):
    from ultralytics import YOLO
    m = YOLO(_clean_copy(weights)).val(data=data_yaml, split=split, imgsz=640, batch=16, plots=False, verbose=False,
                                       project=tempfile.mkdtemp(prefix="amb_val_"), end2end=True, **EVAL)
    names = dict(m.names)
    per = {names[int(c)]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
           for i, c in enumerate(m.ap_class_index)}
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "per_class": per}


# ----------------------------------------------------------------------------------------------------------------------
# Fused evaluation on full images
# ----------------------------------------------------------------------------------------------------------------------
def _load_gt(img, names, frame=640):
    lab = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = []
    if os.path.exists(lab):
        for line in open(lab):
            c, x, y, w, h = line.split()
            x, y, w, h = (float(v) * frame for v in (x, y, w, h))
            out.append((names[int(c)], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return out


def _predict(model, imgs):
    out, t = {}, 0.0
    for i in range(0, len(imgs), 16):
        b = imgs[i:i + 16]
        t0 = time.time()
        res = model.predict(b, imgsz=640, verbose=False, end2end=True, **EVAL)
        t += time.time() - t0
        for p, r in zip(b, res):
            out[p] = [(model.names[int(x.cls[0])], tuple(x.xyxy[0].tolist()), float(x.conf[0])) for x in r.boxes]
    return out, t / max(1, len(imgs))


def _score(frames, systems, classes):
    """frames: [(episode, gt, {system: preds})] -> per system: mAP50, mAP50-95 over `classes` present in GT, per-class."""
    import oracle_eval
    out = {}
    for s in systems:
        tallies = {t: oracle_eval.Tally(iou_thr=t) for t in IOUS}
        for _ep, gt, preds in frames:
            g = [x for x in gt if x[0] in classes]
            p = [x for x in preds[s] if x[0] in classes]
            for t in tallies.values():
                t.add(g, p)
        per = {}
        rows = {t: {r["cls"]: r for r in tallies[t].per_class()} for t in IOUS}
        for c, r in rows[0.5].items():
            if not r["gt"] or r["ap"] is None:
                continue
            aps = [rows[t].get(c, {}).get("ap") for t in IOUS]
            per[c] = {"ap50": r["ap"], "ap50_95": sum(a or 0 for a in aps) / len(aps), "gt": r["gt"],
                      "recall50": r["recall"]}
        out[s] = {"mAP50": sum(v["ap50"] for v in per.values()) / max(1, len(per)),
                  "mAP50_95": sum(v["ap50_95"] for v in per.values()) / max(1, len(per)),
                  "classes": len(per), "per_class": per}
    return out


def _bootstrap(frames, a, b, classes, n=200, seed=0):
    """Paired episode bootstrap of mAP50(b) - mAP50(a) over `classes`: whole episodes resampled, AP pooled."""
    import oracle_eval
    by_ep = {}
    for f in frames:
        by_ep.setdefault(f[0], []).append(f)
    eps = sorted(by_ep)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        pick = [rng.choice(eps) for _ in eps]
        vals = []
        for s in (a, b):
            t = oracle_eval.Tally(iou_thr=0.5)
            for e in pick:
                for _ep, gt, preds in by_ep[e]:
                    t.add([x for x in gt if x[0] in classes], [x for x in preds[s] if x[0] in classes])
            aps = [r["ap"] for r in t.per_class() if r["gt"] and r["ap"] is not None]
            vals.append(sum(aps) / max(1, len(aps)))
        diffs.append(vals[1] - vals[0])
    diffs.sort()
    return {"mean": sum(diffs) / len(diffs), "ci95": [diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]],
            "episodes": len(eps), "resamples": n}


def _frame_meta(store_frames):
    """img name -> {inspection, held: (type, box) | None, kind, labels, protocol} from a store's frames.jsonl."""
    out = {}
    with open(store_frames) as fh:
        for line in fh:
            r = json.loads(line)
            if "img" not in r:
                continue
            kind = r.get("kind")
            held = None
            hid = (r.get("pose") or {}).get("held")
            if hid:
                lab = next((l for l in r["labels"] if l["oid"] == hid), None)
                if lab:
                    held = (lab["type"], tuple(lab["box"]))
            out[r["img"]] = {"inspection": bool(kind and kind.startswith("held")), "held": held, "kind": kind,
                             "labels": [(l["type"], tuple(l["box"])) for l in r["labels"]],
                             "protocol": r.get("protocol")}
    return out


def _held_outcome(meta, preds, thr=0.25):
    """Inspection frames: did the system name the HELD object correctly? The top-confidence prediction overlapping
    the held box (IoU >= .5) decides; no such prediction = a miss (a failure). Held non-targets: a target-class
    prediction on the held box = a false positive."""
    if not meta["held"]:
        return None
    htype, hbox = meta["held"]
    cand = sorted((p for p in preds if p[2] >= thr and _iou(p[1], hbox) >= 0.5), key=lambda p: -p[2])
    if htype in TARGETS:
        return {"target": True, "ok": bool(cand) and cand[0][0] == htype, "missed": not cand,
                "said": cand[0][0] if cand else None, "true": htype}
    return {"target": False, "fp": any(p[0] in TARGETS for p in cand)}


def evaluate(final=False):
    """Systems: primary; GATED fusion (deployment rule — the specialist replaces target-class predictions only on
    frames where the robot is holding something up for inspection, a state it knows; never simulator identity);
    ungated strict replacement (ablation); control. Endpoints: held-target naming success (primary endpoint, misses
    count as failures), held-non-target false-positive rate, AP on inspection frames and on all frames; paired
    episode bootstrap."""
    from ultralytics import YOLO
    prim = YOLO(_clean_copy(PRIMARY_W))
    spec = YOLO(_clean_copy(os.path.join(WEIGHTS, f"{SCENE}.amb26m.pt")))
    ctrl = YOLO(_clean_copy(os.path.join(WEIGHTS, f"{SCENE}.yolo26s.control.pt")))
    names49 = [prim.names[i] for i in range(len(prim.names))]
    amb_meta = _frame_meta(os.path.join(AMB, "frames.jsonl"))
    prim_meta = _frame_meta(os.path.join(os.path.dirname(PRIM), "frames.jsonl"))
    sets = {}
    if final:
        sets["amb_holdout"] = ([os.path.join(AMB, "images", "test", n + ".jpg") for n, m in amb_meta.items()
                                if m["protocol"] == kd.AMB_HOLDOUT], amb_meta)
    else:
        sets["amb_test"] = ([l.strip() for l in open(os.path.join(AMB, "yolo_amb49", "test.txt")) if l.strip()],
                            amb_meta)
        sets["primary_test"] = ([l.strip() for l in open(os.path.join(PRIM, "test.txt")) if l.strip()], prim_meta)
    S = ("primary", "gated", "ungated", "control")
    rep = {"spec": SPEC, "control": CONTROL, "eval": EVAL, "targets": TARGETS,
           "fusion": "gated strict replacement (inspection frames only); ungated = ablation", "sets": {}}
    for name, (imgs, meta) in sets.items():
        pp, t_p = _predict(prim, imgs)
        sp, t_s = _predict(spec, imgs)
        cp, t_c = _predict(ctrl, imgs)
        frames, held_rows, conflicts = [], [], 0
        for img in imgs:
            m = meta[os.path.basename(img).rsplit(".", 1)[0]]
            gt = m["labels"]
            repl = [x for x in pp[img] if x[0] not in TARGETS] + sp[img]
            preds = {"primary": pp[img], "gated": repl if m["inspection"] else pp[img], "ungated": repl,
                     "control": cp[img]}
            if m["inspection"]:
                conflicts += _conflicts(sp[img])
                held_rows.append((_episode(img), {s: _held_outcome(m, preds[s]) for s in S}))
            frames.append((_episode(img), gt, preds, m["inspection"]))
        fr3 = [(e, g, p) for e, g, p, _i in frames]
        insp = [(e, g, p) for e, g, p, i in frames if i]
        res = {"images": len(imgs), "inspection_images": len(insp), "episodes": len({f[0] for f in frames}),
               "all_frames": {"all49": _score(fr3, S, set(names49)), "targets": _score(fr3, S, set(TARGETS))},
               "inspection_frames": {"targets": _score(insp, S, set(TARGETS))} if insp else None,
               "specialist_conflicting_pairs_at_.25": conflicts,
               "latency_ms_per_image": {"primary": 1000 * t_p, "specialist": 1000 * t_s, "control": 1000 * t_c}}
        ht = [(e, o) for e, o in held_rows if o["primary"] and o["primary"]["target"]]
        hn = [(e, o) for e, o in held_rows if o["primary"] and not o["primary"]["target"]]
        res["held_target_success"] = {s: sum(o[s]["ok"] for _e, o in ht) / max(1, len(ht)) for s in S}
        res["held_target_missed"] = {s: sum(o[s]["missed"] for _e, o in ht) / max(1, len(ht)) for s in S}
        res["held_nontarget_fp_rate"] = {s: sum(o[s]["fp"] for _e, o in hn) / max(1, len(hn)) for s in S}
        res["held_counts"] = {"targets": len(ht), "non_targets": len(hn)}
        per_cls = {}
        for _e, o in ht:
            c = o["primary"]["true"]
            d = per_cls.setdefault(c, {s: [0, 0] for s in S})
            for s_ in S:
                d[s_][0] += int(o[s_]["ok"])
                d[s_][1] += 1
        res["held_target_success_per_class"] = {c: {s_: round(a / b, 3) for s_, (a, b) in d.items()}
                                                for c, d in per_cls.items()}
        res["boot_held_success_gated_minus_primary"] = _boot_rate(ht, "primary", "gated")
        res["boot_held_success_gated_minus_control"] = _boot_rate(ht, "control", "gated")
        if insp:
            res["boot_inspection_targetAP50_gated_minus_primary"] = _bootstrap(insp, "primary", "gated", set(TARGETS))
        res["boot_all_targetAP50_gated_minus_primary"] = _bootstrap(fr3, "primary", "gated", set(TARGETS))
        rep["sets"][name] = res
        hs = res["held_target_success"]
        _log(f"FUSION {name}: held-target success primary {hs['primary']:.3f} | gated {hs['gated']:.3f} | control "
             f"{hs['control']:.3f} (n={len(ht)}); held non-target FP {res['held_nontarget_fp_rate']}; all-frame "
             f"target AP50 primary {res['all_frames']['targets']['primary']['mAP50']:.3f} | gated "
             f"{res['all_frames']['targets']['gated']['mAP50']:.3f} | control "
             f"{res['all_frames']['targets']['control']['mAP50']:.3f}")
        _dump(os.path.join(REPORTS, "amb_eval_final.json" if final else "amb_eval.json"), rep)
    return rep


def _boot_rate(rows, a, b, n=1000, seed=0):
    """Paired episode bootstrap of the held-target success-rate difference b - a."""
    by = {}
    for e, o in rows:
        by.setdefault(e, []).append(o)
    eps = sorted(by)
    if not eps:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        pick = [x for e in (rng.choice(eps) for _ in eps) for x in by[e]]
        diffs.append((sum(o[b]["ok"] for o in pick) - sum(o[a]["ok"] for o in pick)) / max(1, len(pick)))
    diffs.sort()
    return {"mean": sum(diffs) / n, "ci95": [diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]], "episodes": len(eps)}


def _holdout_gt(img):
    """Holdout frames have no YOLO list (sealed): read their 49-class boxes from the store's frames.jsonl."""
    global _HOLD
    if "_HOLD" not in globals():
        _HOLD = {}
        with open(os.path.join(AMB, "frames.jsonl")) as fh:
            for line in fh:
                r = json.loads(line)
                if "img" in r and r.get("protocol") == kd.AMB_HOLDOUT:
                    _HOLD[r["img"]] = [(l["type"], tuple(l["box"])) for l in r["labels"]]
    return _HOLD[os.path.basename(img).rsplit(".", 1)[0]]


def _conflicts(preds, thr=0.25, iou_thr=0.7):
    """Pairs of confident specialist boxes of DIFFERENT target classes on the same object (e.g. Fork vs Spoon)."""
    p = [x for x in preds if x[2] >= thr]
    n = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i][0] != p[j][0] and _iou(p[i][1], p[j][1]) >= iou_thr:
                n += 1
    return n


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("lcurve", "control", "eval", "holdout"))
    a = ap.parse_args(argv)
    {"lcurve": lcurve, "control": control, "eval": evaluate, "holdout": lambda: evaluate(final=True)}[a.cmd]()


if __name__ == "__main__":
    main()
