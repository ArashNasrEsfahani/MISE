"""Kitchen 2/3 state estimator under A28 item 7 and A38 (7 per-axis YOLO11s-cls, 128 px, 30 epochs, patience 10, batch 128,
seed 0 -- sim/train_state.py's recipe; sim/state_data.py and sim/train_state.py stay byte-identical).

  crops : the frozen state_data.main on a FILTERED view of the kitchen's main store: TRAIN and VAL episodes only (A28 item 7:
          "crops from TRAIN / VAL episodes only"; TEST crops are made at the evaluation, after the freeze), quarantined frames
          excluded (state_data also skips them), written to the per-kitchen path ~/thor-data/kitchen_v3/state_cls_<scene>
          (never the shared Kitchen 1 state_cls).
  train : per axis, initialized from the FROZEN Kitchen 1 state model of that axis (A38 item 1; sha256 from
          sim/final_test/_checkpoints/K2-TOPUP/k1_source_hashes.sha256), output layer retained only because the ordered
          label list ("0", "1") is identical (A38 item 2: asserted); fresh optimizer; train_state.py's exact train() keywords;
          best.pt (axis VAL top-1, A28 item 6 selection rule) -> sim/weights/v3/<scene>.state_<axis>.pt. An axis has no
          resume (A28 item 10): a stopped axis restarts from scratch; finished axes are skipped (done marker = weights +
          report row).
Calibration (size floors from Arash's hand labels, VAL thresholds) is a later step (A28 item 7), not done here.

    python sim/kitchen_v3/k23/k23_state_train.py crops --scene FloorPlan15        (WSL thor-env; CPU)
    python sim/kitchen_v3/k23/k23_state_train.py train --scene FloorPlan15 [--axes open,on]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SIM)
AXES = ("open", "on", "sliced", "cooked", "broken", "dirty", "filled")
SCENES = ("FloorPlan15", "FloorPlan17")
DATA_ROOT = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3")
HASHES = os.path.join(SIM, "final_test", "_checkpoints", "K2-TOPUP", "k1_source_hashes.sha256")
WEIGHTS = os.path.join(SIM, "weights", "v3")
LABELS = ["0", "1"]


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def registered_sha(name: str) -> str:
    for line in open(HASHES, encoding="utf-8"):
        sha, f = line.split()
        if f.lstrip("*") == name:
            return sha
    raise RuntimeError(f"A38: no registered hash for {name}")


def out_dir(scene: str) -> str:
    return os.path.join(DATA_ROOT, f"state_cls_{scene}")


def crops(scene: str) -> dict:
    import state_data
    store = os.path.join(DATA_ROOT, scene)
    view = os.path.join(DATA_ROOT, f"_state_src_{scene}")
    out = out_dir(scene)
    if os.path.exists(out):
        raise RuntimeError(f"{out} exists: crops are built once (move it aside to rebuild)")
    os.makedirs(view, exist_ok=True)
    n = {"train": 0, "val": 0, "skipped_test": 0, "skipped_quarantined": 0}
    with open(os.path.join(store, "frames.jsonl"), encoding="utf-8") as fi, \
            open(os.path.join(view, "frames.jsonl"), "w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            if "img" not in r:
                continue
            if r.get("quarantined"):
                n["skipped_quarantined"] += 1
                continue
            if r.get("split") not in ("train", "val"):
                n["skipped_test"] += 1
                continue
            n[r["split"]] += 1
            fo.write(line if line.endswith("\n") else line + "\n")
    link = os.path.join(view, "images")
    if not os.path.lexists(link):
        os.symlink(os.path.join(store, "images"), link)
    if os.path.realpath(link) != os.path.realpath(os.path.join(store, "images")):
        raise RuntimeError(f"{link} does not point at {store}/images")
    n["filtered_manifest_sha256"] = sha_file(os.path.join(view, "frames.jsonl"))
    state_data.main(["--store", view, "--out", out])
    summ = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    if any(k.split("/")[1] == "test" and v["n"] for k, v in summ.items()):
        raise RuntimeError("TEST crops were written: refused (A28 item 7)")
    rec = {"what": f"K23 state crops {scene}", "when": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": n, "out": out,
           "summary": summ, "code_sha256": sha_file(os.path.abspath(__file__))}
    p = os.path.join(HERE, scene, "state", "state_crops.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p, "w", encoding="utf-8"), indent=1)
    return rec


def k1_source(axis: str) -> str:
    name = f"FloorPlan1.state_{axis}.pt"
    src = os.path.join(WEIGHTS, name)
    want = registered_sha(name)
    if sha_file(src) != want:
        raise RuntimeError(f"A38: {name} does not match the registered sha256")
    d = os.path.join(os.path.expanduser("~"), ".cache", "thor-yolo", "a38")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f"FloorPlan1.state_{axis}.k1-source.pt")
    if not os.path.exists(dst) or sha_file(dst) != want:
        shutil.copyfile(src, dst)
    if sha_file(dst) != want:
        raise RuntimeError(f"A38: the copied {name} does not match the registered sha256")
    return dst


def audit_axis(scene: str, axis: str, src: str) -> dict:
    """A38 item 2 (Astra K23-STATE-TRAIN-r1 item 1): loaded / re-initialized parameters (names AND shapes) of the axis model
    built for the kitchen's label list, from the K1 axis checkpoint."""
    import copy
    import torch
    import ultralytics
    from ultralytics.nn.tasks import ClassificationModel
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sm = (ck.get("ema") or ck["model"]).float()
    tgt = ClassificationModel(copy.deepcopy(sm.yaml), nc=len(LABELS), verbose=False)
    ssd, tsd = sm.state_dict(), tgt.state_dict()
    loaded = [{"name": k, "shape": list(v.shape)} for k, v in tsd.items() if k in ssd and ssd[k].shape == v.shape]
    ln = {d["name"] for d in loaded}
    reset = [{"name": k, "shape": list(v.shape), "source_shape": list(ssd[k].shape) if k in ssd else None}
             for k, v in tsd.items() if k not in ln]
    name = f"FloorPlan1.state_{axis}.pt"
    rep = {"what": f"A38 item 2 transfer audit: {scene} state_{axis}", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": f"sim/weights/v3/{name}", "source_sha256": registered_sha(name), "labels": LABELS, "labels_identical": True,
           "ultralytics": ultralytics.__version__, "n_target_params": len(tsd), "n_loaded": len(loaded), "n_reset": len(reset),
           "loaded": loaded, "reset": reset, "code_sha256": sha_file(os.path.abspath(__file__))}
    out = os.path.join(HERE, scene, "state", f"a38_transfer_audit_state_{axis}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    rep["audit_sha256"] = sha_file(out)
    return rep


def _count(d: str) -> int:
    return sum(len(fs) for _r, _d, fs in os.walk(d)) if os.path.isdir(d) else 0


def selected_epoch(results_csv: str) -> dict:
    """The epoch best.pt holds: ultralytics 8.4.83 overwrites best.pt on EQUAL fitness, so the LAST epoch with the maximal
    fitness (classify fitness = mean(top1, top5) = ranking by top1 for two classes) (Astra K23-STATE-TRAIN-r1 item 4)."""
    import csv
    rows = list(csv.DictReader(open(results_csv, encoding="utf-8")))
    key = next(k for k in rows[0] if "top1" in k)
    k5 = next((k for k in rows[0] if "top5" in k), None)
    fit = [(float(r[key]) + (float(r[k5]) if k5 else float(r[key]))) / 2 for r in rows]
    best = max(fit)
    last = max(i for i, f in enumerate(fit) if f == best)
    return {"epoch": int(float(rows[last].get("epoch", last + 1))), "top1": float(rows[last][key]), "tied_epochs": sum(f == best for f in fit)}


def train(scene: str, axes) -> dict:
    from ultralytics import YOLO
    rep_p = os.path.join(HERE, scene, "state", "state_train.json")
    rep = json.load(open(rep_p, encoding="utf-8")) if os.path.exists(rep_p) else {}
    rep.setdefault("_attempts", {})
    data_root = out_dir(scene)

    def save():
        os.makedirs(os.path.dirname(rep_p), exist_ok=True)
        json.dump(rep, open(rep_p, "w", encoding="utf-8"), indent=1)
    for ax in axes:
        w_out = os.path.join(WEIGHTS, f"{scene}.state_{ax}.pt")
        if ax in rep and os.path.exists(w_out) and sha_file(w_out) == rep[ax].get("weights_sha256"):
            print(f"STATE {ax}: done earlier, skipped", flush=True)
            continue
        d = os.path.join(data_root, ax)
        if _count(os.path.join(d, "test")):
            raise RuntimeError(f"{ax}: TEST crops present: refused (A28 item 7)")
        if not _count(os.path.join(d, "val")) or not _count(os.path.join(d, "train")):
            raise RuntimeError(f"{ax}: empty TRAIN or VAL crops: refused")
        # A28 item 10: a state axis has no resume; an earlier unfinished attempt is recorded and discarded, never continued
        att = rep["_attempts"].setdefault(ax, [])
        for a_ in att:
            if a_.get("status") == "started":
                a_["status"] = "interrupted, discarded (restarted from the Kitchen 1 axis model)"
        n = len(att) + 1
        run = f"{scene}.{ax}.a{n}"
        att.append({"attempt": n, "run": run, "status": "started", "when": time.strftime("%Y-%m-%d %H:%M:%S")})
        save()
        t = time.time()
        src = k1_source(ax)
        m = YOLO(src)
        names = [m.names[i] for i in sorted(m.names)] if isinstance(m.names, dict) else list(m.names)
        dirs = sorted(x for x in os.listdir(os.path.join(d, "train")) if os.path.isdir(os.path.join(d, "train", x)))
        if names != LABELS or dirs != LABELS:
            raise RuntimeError(f"A38 item 2: {ax} label lists differ (K1 {names} vs {scene} {dirs}): refused")
        au = audit_axis(scene, ax, src)
        m.train(data=d, imgsz=128, epochs=30, patience=10, batch=128, seed=0, deterministic=True,
                scale=0.0, erasing=0.0, auto_augment=None, fliplr=0.5, mosaic=0.0, workers=8, device=0,
                project="/tmp/state_runs_k23", name=run, exist_ok=False, plots=False, verbose=False, resume=False)
        rd = os.path.join("/tmp/state_runs_k23", run)
        best = os.path.join(rd, "weights", "best.pt")
        shutil.copyfile(best, w_out)
        r = YOLO(best).val(data=d, split="val", imgsz=128, batch=128, verbose=False, plots=False)
        rows = open(os.path.join(rd, "results.csv")).read().strip().splitlines()
        keep = os.path.join(HERE, scene, "state", "runs", run)
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(rd, keep, ignore=shutil.ignore_patterns("*.pt", "*.jpg"))
        rep[ax] = {"init": f"A38 FloorPlan1.state_{ax}.pt", "source_sha256": au["source_sha256"], "audit_sha256": au["audit_sha256"],
                   "audit_loaded": au["n_loaded"], "audit_reset": au["n_reset"], "labels": LABELS, "run": run,
                   "epochs_run": len(rows) - 1, "selected": selected_epoch(os.path.join(rd, "results.csv")),
                   "val_top1": round(float(r.top1), 4),
                   "weights": os.path.relpath(w_out, os.path.dirname(SIM)).replace(os.sep, "/"), "weights_sha256": sha_file(w_out),
                   "seconds": round(time.time() - t), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        att[-1]["status"] = "done"
        save()
        print(f"STATE {ax}: val top1 {r.top1:.3f} (audit loaded {au['n_loaded']}, reset {au['n_reset']}; {time.time() - t:.0f} s)", flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("crops", "train"))
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--axes", default=",".join(AXES))
    a = ap.parse_args(argv)
    if a.cmd == "crops":
        print(json.dumps(crops(a.scene)["frames"]))
        return 0
    train(a.scene, [x for x in a.axes.split(",") if x])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
