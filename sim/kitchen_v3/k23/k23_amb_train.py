"""Kitchen 2/3 specialist + control training under A28 item 6 and A38 (Arash 2026-10-17: "I sign the K1-weight transfer
amendment for Kitchens 2 and 3.").

A thin wrapper: sim/train_amb.py and sim/build_kitchen_data.py stay byte-identical (Kitchen 1's code). This module only
  * stages the kitchen's amb lists with the frozen `bkd.stage_yolo_amb` (quarantined frames and the sealed holdout
    protocol amb-v1-holdout go into NO list) and verifies them (k23_verify_lists, both list sets);
  * specialist: YOLO26m, the A28 recipe (train_amb.SPEC: MuSGD, 60 epochs, patience 20, 640 px, batch 8, seed 0), the FULL
    amb train set (no learning curve in these kitchens, A28 item 6), selected on amb VAL; initialized from the FROZEN
    Kitchen 1 specialist FloorPlan1.amb26m.pt (A38 item 1, sha256 verified), every parameter loaded only because the
    ordered label list is kd.AMB_TARGETS in both kitchens (A38 item 2: asserted here, else refused);
  * control: train_amb.control() unchanged, pointed at the kitchen: YOLO26s fine-tuned from THAT kitchen's adapted
    primary on primary-train + amb-train (data_plus.yaml), 20 epochs, lr0 .002, pooled VAL (A28 item 6, A38 item 1);
  * writes the A38 item 2 transfer audit (loaded / re-initialized parameters) before the specialist trains.
Fresh optimizer and scheduler (training starts from weights only). Resume after a stop: train_amb._train resumes from
~/thor-data/kitchen_v3/_ckpt/<run>/last.pt (copied out of /tmp by sim/kitchen_v3/ckpt_backup.sh).

    python sim/kitchen_v3/k23/k23_amb_train.py stage --scene FloorPlan15          (WSL thor-env; CPU)
    python sim/kitchen_v3/k23/k23_amb_train.py audit --scene FloorPlan15          (CPU; writes the A38 audit)
    python sim/kitchen_v3/k23/k23_amb_train.py train --scene FloorPlan15 --model amb26m|control
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
sys.path.insert(0, HERE)
K1_SPEC = os.path.join(SIM, "weights", "v3", "FloorPlan1.amb26m.pt")
K1_SPEC_SHA = "b02c5d05c7dcd6860a1f2658873fcc9ec0d5d6ad589ae3de68e512f4b813b356"      # A38 item 1
SCENES = ("FloorPlan15", "FloorPlan17")
AMB_ROOT = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3_amb")


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def k1_spec_source() -> str:
    """The frozen Kitchen 1 specialist, as a sha-verified copy outside the apostrophe path (ultralytics strips it)."""
    if sha_file(K1_SPEC) != K1_SPEC_SHA:
        raise RuntimeError("A38: the Kitchen 1 specialist does not match the registered sha256")
    d = os.path.join(os.path.expanduser("~"), ".cache", "thor-yolo", "a38")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "FloorPlan1.amb26m.k1-source.pt")
    if not os.path.exists(dst) or sha_file(dst) != K1_SPEC_SHA:
        shutil.copyfile(K1_SPEC, dst)
    if sha_file(dst) != K1_SPEC_SHA:
        raise RuntimeError("A38: the copied Kitchen 1 specialist does not match the registered sha256")
    return dst


def point_train_amb(scene: str):
    """train_amb with its module globals set to the kitchen (process-local; the file itself is not edited)."""
    import train_amb as TA
    TA.SCENE = scene
    TA.AMB = os.path.join(AMB_ROOT, scene)
    TA.PRIM = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3", scene, "yolo_primary")
    TA.REPORTS = os.path.join(HERE, scene, "amb")
    TA.PRIMARY_W = os.path.join(TA.WEIGHTS, f"{scene}.yolo26s.pt")
    os.makedirs(TA.REPORTS, exist_ok=True)
    return TA


def label_list(data_yaml: str) -> list:
    names, on = [], False
    for line in open(data_yaml, encoding="utf-8"):
        if line.startswith("names:"):
            on = True
            continue
        if on and line.strip():
            names.append(line.split(":", 1)[1].strip())
    return names


def stage(scene: str) -> dict:
    import build_kitchen_data as bkd
    import kitchen_data as kd
    rep = bkd.stage_yolo_amb(scene, AMB_ROOT)
    names = label_list(os.path.join(AMB_ROOT, scene, "yolo_amb", "data.yaml"))
    if names != list(kd.AMB_TARGETS):
        raise RuntimeError(f"A38 item 2: the specialist label list {names} is not kd.AMB_TARGETS")
    import k23_verify_lists as V
    v = {ds: V.main(["--scene", scene, "--store", "amb", "--dataset", ds]) for ds in ("yolo_amb", "yolo_amb49")}
    rec = {"what": f"K23 amb lists {scene}", "when": time.strftime("%Y-%m-%d %H:%M:%S"), "build": rep,
           "labels": names, "verify_rc": v}
    out = os.path.join(HERE, scene, "amb", "stage_amb.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
    if any(v.values()):
        raise RuntimeError(f"list verification FAILED {v} (see {scene}/clean/verify_amb_*.json)")
    return rec


def audit_path(scene: str, model: str) -> str:
    return os.path.join(HERE, scene, "amb", f"a38_transfer_audit_{model}.json")


def _model_names(m) -> list:
    return [m.names[i] for i in sorted(m.names)] if isinstance(m.names, dict) else list(m.names)


def _clean_tmp(path: str) -> str:
    import tempfile
    dst = os.path.join(tempfile.gettempdir(), "k23audit_" + os.path.basename(path))
    shutil.copyfile(path, dst)
    return dst


def expected_audit_source(scene: str, model: str) -> tuple:
    """(loadable source path, registered sha256, current ordered target labels, required source labels or None)."""
    import kitchen_data as kd
    if model == "amb26m":
        return (k1_spec_source(), K1_SPEC_SHA, label_list(os.path.join(AMB_ROOT, scene, "yolo_amb", "data.yaml")),
                list(kd.AMB_TARGETS))
    TA = point_train_amb(scene)
    if not os.path.exists(TA.PRIMARY_W):
        raise RuntimeError(f"the kitchen's adapted primary {TA.PRIMARY_W} does not exist yet (the control waits for it)")
    return (_clean_tmp(TA.PRIMARY_W), sha_file(TA.PRIMARY_W),
            label_list(os.path.join(AMB_ROOT, scene, "yolo_amb49", "data_plus.yaml")), None)


def audit(scene: str, model: str = "amb26m") -> dict:
    """A38 item 2: loaded / re-initialized parameters (names AND shapes) for the specialist (from the frozen Kitchen 1
    specialist) or the control (from the kitchen's adapted primary). Refused when the ordered label lists differ."""
    import copy
    import torch
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    path, sha, tgt_names, must = expected_audit_source(scene, model)
    src = torch.load(path, map_location="cpu", weights_only=False)
    smodel = (src.get("ema") or src["model"]).float()
    src_names = _model_names(smodel)
    if (must is not None and src_names != must) or tgt_names != src_names:
        raise RuntimeError(f"A38 item 2: ordered label lists differ (source {src_names} vs {scene} {tgt_names}): refused")
    tgt = DetectionModel(copy.deepcopy(smodel.yaml), nc=len(tgt_names), verbose=False)
    ssd, tsd = smodel.state_dict(), tgt.state_dict()
    loaded = [{"name": k, "shape": list(v.shape)} for k, v in tsd.items() if k in ssd and ssd[k].shape == v.shape]
    ln = {d["name"] for d in loaded}
    reset = [{"name": k, "shape": list(v.shape), "source_shape": list(ssd[k].shape) if k in ssd else None}
             for k, v in tsd.items() if k not in ln]
    rep = {"what": f"A38 item 2 transfer audit: {scene} {model}", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": "sim/weights/v3/FloorPlan1.amb26m.pt" if model == "amb26m" else f"sim/weights/v3/{scene}.yolo26s.pt",
           "source_sha256": sha, "labels": tgt_names, "labels_identical": True, "ultralytics": ultralytics.__version__,
           "n_target_params": len(tsd), "n_loaded": len(loaded), "n_reset": len(reset), "loaded": loaded, "reset": reset,
           "code_sha256": sha_file(os.path.abspath(__file__))}
    out = audit_path(scene, model)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    return rep


def check_audit(scene: str, model: str) -> None:
    """At training entry: the audit exists and matches the registered source sha and the CURRENT label list (no stale audit)."""
    p = audit_path(scene, model)
    if not os.path.exists(p):
        raise RuntimeError(f"REFUSED: write the A38 audit first ({p})")
    a = json.load(open(p, encoding="utf-8"))
    _path, sha, labels, _must = expected_audit_source(scene, model)
    if a.get("source_sha256") != sha or a.get("labels") != labels or not a.get("loaded"):
        raise RuntimeError(f"REFUSED: the A38 audit {p} is stale (source / labels / loaded list differ): re-run audit")


def check_resume_lineage(TA, run: str, data_yaml: str, recipe: dict) -> dict:
    """Astra K23-AMB-TRAIN-r1 item 3: a checkpoint is resumed only if it is THIS kitchen run (name, data, recipe), never a
    Kitchen 1 state. {} = fresh start."""
    import torch
    ck = os.path.join(TA.CKPT, run, "last.pt")
    if not os.path.exists(ck):
        return {}
    a = torch.load(ck, map_location="cpu", weights_only=False).get("train_args") or {}
    want = {"name": run, "data": data_yaml, "epochs": recipe["epochs"], "lr0": recipe["lr0"], "batch": recipe["batch"],
            "imgsz": recipe["imgsz"], "seed": recipe["seed"], "optimizer": recipe["optimizer"]}
    bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
    if bad:
        raise RuntimeError(f"REFUSED: checkpoint {ck} is not this run (differs: {bad}); move it aside, never resume it")
    return {"checkpoint": ck, "matched": sorted(want)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("stage", "audit", "train"))
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--model", default="amb26m", choices=("amb26m", "control"))
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        print(json.dumps(stage(a.scene)["build"]))
        return 0
    if a.cmd == "audit":
        r = audit(a.scene, a.model)
        print(f"{a.scene} {a.model}: loaded {r['n_loaded']}/{r['n_target_params']}, re-initialized {r['n_reset']}")
        return 0
    TA = point_train_amb(a.scene)
    try:
        check_audit(a.scene, a.model)
    except RuntimeError as e:
        print(e)
        return 2
    if a.model == "amb26m":
        out = os.path.join(TA.WEIGHTS, f"{a.scene}.amb26m.pt")
        yml = os.path.join(TA.AMB, "yolo_amb", "data.yaml")
        print(json.dumps({"resume_lineage": check_resume_lineage(TA, f"{a.scene}.amb26m", yml, TA.SPEC)}))
        TA._train(f"{a.scene}.amb26m", yml, k1_spec_source(), TA.SPEC, out)
        val = TA._val(out, os.path.join(TA.AMB, "yolo_amb", "data.yaml"), "val")
        TA._dump(os.path.join(TA.REPORTS, "amb_specialist.json"), {"weights": out, "val": val, "init": "A38 K1 amb26m"})
        print(json.dumps({"weights": out, "val": val}, default=str))
        return 0
    print(json.dumps({"resume_lineage": check_resume_lineage(TA, f"{a.scene}.yolo26s.control",
                                                             os.path.join(TA.AMB, "yolo_amb49", "data_plus.yaml"), TA.CONTROL),
                      "control_source": os.path.basename(TA.PRIMARY_W), "control_source_sha256": sha_file(TA.PRIMARY_W)}))
    print(json.dumps(TA.control(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
