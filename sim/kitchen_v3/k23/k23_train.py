"""Kitchen 2/3 detector training under A38 (Arash 2026-10-17: "I sign the K1-weight transfer amendment for Kitchens 2 and
3."; docs/final-test/preregistration.md A38; Astra K23-TRANSFER-r1 + K23-PARALLEL-r1 item 2).

A thin wrapper: sim/train_kitchen.py stays byte-identical (Kitchen 1's recipe). This module only
  * sets the primary YOLO26s starting weights to the FROZEN Kitchen 1 primary (A38 item 1), each kitchen from the same
    file (sha256 verified), never Kitchen 2 -> Kitchen 3;
  * adds `yolo26s_coco` = the identical recipe from the COCO yolo26s.pt (the ONE registered FloorPlan15 sensitivity, A38
    item 4), written to its own weights file and run folder;
  * writes the A38 item 2 audit BEFORE training: every parameter of the target model (the kitchen's class count) marked
    loaded (same name + shape as the source, the rule ultralytics' intersect_dicts applies when training starts from
    those weights) or re-initialized (both YOLO26 classification branches change shape with the class count).
Fresh optimizer and scheduler: training starts from weights only (no resume of a Kitchen 1 run); epochs, patience,
batch, image size, seed, splits and VAL selection are train_kitchen.RECIPE unchanged (A38 item 3).

    python sim/kitchen_v3/k23/k23_train.py audit --scene FloorPlan15            (WSL thor-env; CPU only, writes the audit)
    python sim/kitchen_v3/k23/k23_train.py train --scene FloorPlan15 --model yolo26s|yolo26s_coco
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
K1_PRIMARY = os.path.join(SIM, "weights", "v3", "FloorPlan1.yolo26s.pt")
K1_PRIMARY_SHA = "4418f63f4b84060c51353bffb574d05e49042e763b36136d870c7e56e8b90f54"      # A38 item 1
SCENES = ("FloorPlan15", "FloorPlan17")


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def k1_source(copy_for_ultralytics: bool = False) -> str:
    got = sha_file(K1_PRIMARY)
    if got != K1_PRIMARY_SHA:
        raise RuntimeError(f"A38: {K1_PRIMARY} sha256 {got} != the registered {K1_PRIMARY_SHA}")
    if not copy_for_ultralytics:
        return K1_PRIMARY
    # ultralytics strips the apostrophe of the repo path ("Master's" -> "Masters") when it loads a checkpoint; train from a
    # byte-identical copy under ~/.cache (sha re-verified on the copy)
    import shutil
    d = os.path.join(os.path.expanduser("~"), ".cache", "thor-yolo", "a38")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "FloorPlan1.yolo26s.k1-source.pt")
    if not os.path.exists(dst) or sha_file(dst) != K1_PRIMARY_SHA:
        shutil.copyfile(K1_PRIMARY, dst)
    if sha_file(dst) != K1_PRIMARY_SHA:
        raise RuntimeError("A38: the copied Kitchen 1 source does not match the registered sha256")
    return dst


def audit(scene: str, nc: int, out_path: str) -> dict:
    """Loaded / re-initialized parameter list for a YOLO26s with `nc` classes started from the Kitchen 1 primary."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    src = torch.load(k1_source(), map_location="cpu", weights_only=False)
    smodel = (src.get("ema") or src["model"]).float()
    import copy
    src_nc = int(smodel.yaml.get("nc", 0))                 # read BEFORE building the target (it rewrites the yaml dict)
    tgt = DetectionModel(copy.deepcopy(smodel.yaml), nc=nc, verbose=False)
    ssd, tsd = smodel.state_dict(), tgt.state_dict()
    loaded = [k for k, v in tsd.items() if k in ssd and ssd[k].shape == v.shape]
    reset = [{"name": k, "shape": list(v.shape), "source_shape": list(ssd[k].shape) if k in ssd else None}
             for k, v in tsd.items() if k not in loaded]
    rep = {"what": f"A38 item 2 transfer audit: {scene} YOLO26s primary (nc={nc}) from the frozen Kitchen 1 primary",
           "when": time.strftime("%Y-%m-%d %H:%M:%S"), "source": os.path.relpath(K1_PRIMARY, os.path.dirname(SIM)),
           "source_sha256": K1_PRIMARY_SHA, "source_nc": src_nc, "target_nc": nc,
           "rule": "loaded = same parameter name AND shape (ultralytics intersect_dicts at train start); else re-initialized",
           "n_target_params": len(tsd), "n_loaded": len(loaded), "n_reset": len(reset),
           "reset": reset, "loaded": loaded, "code_sha256": sha_file(os.path.abspath(__file__))}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("audit", "train"))
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--model", default="yolo26s", choices=("yolo26s", "yolo26s_coco"))
    a = ap.parse_args(argv)
    import kitchen_data as kd
    import k23_classes
    nc = json.load(open(k23_classes.EXPECTED, encoding="utf-8"))["scenes"][a.scene]["n_classes"]
    apath = os.path.join(HERE, a.scene, "train", "a38_transfer_audit_yolo26s.json")
    if a.cmd == "audit":
        rep = audit(a.scene, nc, apath)
        print(f"{a.scene}: nc {rep['source_nc']} -> {nc}; loaded {rep['n_loaded']}/{rep['n_target_params']}, "
              f"re-initialized {rep['n_reset']}: {[r['name'] for r in rep['reset']][:8]} -> {apath}")
        return 0
    if a.model == "yolo26s_coco" and a.scene != "FloorPlan15":
        print("REFUSED: the COCO-init sensitivity is registered for FloorPlan15 only (A38 item 4)")
        return 2
    import train_kitchen as tk
    if a.model == "yolo26s":
        if not os.path.exists(apath):
            print(f"REFUSED: write the A38 audit first ({apath})")
            return 2
        tk.MODELS["yolo26s"] = dict(tk.MODELS["yolo26s"], base=k1_source(copy_for_ultralytics=True))   # A38 item 1
    else:
        tk.MODELS["yolo26s_coco"] = dict(tk.MODELS["yolo26s"], base="yolo26s.pt")    # A38 item 4, identical recipe
    res = tk.train(a.scene, a.model)
    print(json.dumps({k: res.get(k) for k in ("weights_path", "epochs_run", "hours")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
