"""Kitchen 2/3 perception freeze record (A28 item 9 "Freeze before evaluation" [Astra A28 8]; A38 item 4 for FloorPlan15).

Writes sim/kitchen_v3/<scene>/frozen_weights.sha256 ('<sha256>  <path>' lines; repo-relative, or absolute for files under
~/thor-data) BEFORE any TEST / CLOSE-TEST / LIVE / fused amb TEST / state TEST / holdout evaluation. Covered: every deployed .pt
(primary, specialist, control, 7 state models), the state calibration, the size-floor hand-label files (answers + score), the
look-alike map and its VAL source, the list files and data YAMLs of every model, the A38 transfer audits, and the configuration /
dependency code (not just entry scripts). Refused (nothing written) while any input is missing -- in particular the state
calibration, which needs Arash's hand labels first (A28 item 7) -- or when a record already exists. For FloorPlan15, A38 item 4:
the COCO-initialized sensitivity checkpoint is frozen too before any TEST evaluation; `--coco-pending` writes the record
without it and marks the TEST evaluation blocked until `--add-coco` appends it.

    python sim/kitchen_v3/k23/k23_freeze.py --scene FloorPlan15 [--coco-pending] [--check]
    python sim/kitchen_v3/k23/k23_freeze.py --scene FloorPlan15 --add-coco
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(SIM)
DATA = os.path.join(os.path.expanduser("~"), "thor-data")
AXES = ("open", "on", "sliced", "cooked", "broken", "dirty", "filled")
CODE = ("sim/perception.py", "sim/honest.py", "sim/state_eval.py", "sim/state_data.py", "sim/train_state.py", "sim/train_kitchen.py",
        "sim/train_amb.py", "sim/gen_dataset.py", "sim/yolo_recipe.py", "sim/build_kitchen_data.py", "sim/kitchen_data.py",
        "sim/oracle_eval.py", "sim/kitchen_v3/error_breakdown.py", "sim/honest_skills.py", "sim/type_scope.py", "sim/state_review.py",
        "sim/kitchen_v3/k23/k23_train.py", "sim/kitchen_v3/k23/k23_amb_train.py", "sim/kitchen_v3/k23/k23_state_train.py",
        "sim/kitchen_v3/k23/k23_lookalike.py", "sim/kitchen_v3/k23/k23_lists.py", "sim/kitchen_v3/k23/k23_verify_lists.py",
        "sim/kitchen_v3/k23/k23_clean.py", "sim/kitchen_v3/k23/k23_freeze.py")


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inputs(scene: str, coco: bool) -> list:
    w = f"sim/weights/v3/{scene}"
    rel = [f"{w}.yolo26s.pt", f"{w}.amb26m.pt", f"{w}.yolo26s.control.pt"] + [f"{w}.state_{a}.pt" for a in AXES]
    if coco:
        rel.append(f"{w}.yolo26s_coco.pt")
    rel += [f"sim/kitchen_v3/{scene}/state_calib.json", f"sim/step5/lookalike_map_{scene}.json",
            f"sim/kitchen_v3/{scene}/val_errors_yolo26s.json",
            f"sim/kitchen_v3/k23/{scene}/train/a38_transfer_audit_yolo26s.json",
            f"sim/kitchen_v3/k23/{scene}/amb/a38_transfer_audit_amb26m.json",
            f"sim/kitchen_v3/k23/{scene}/amb/a38_transfer_audit_control.json"]
    rel += [f"sim/kitchen_v3/k23/{scene}/state/a38_transfer_audit_state_{a}.json" for a in AXES]
    sr = f"sim/kitchen_v3/k23/{scene}/state_review"
    rel += [f"{sr}/answers.json", f"{sr}/score.json"]
    rel += list(CODE)
    ab = []
    for d in (f"{DATA}/kitchen_v3/{scene}/yolo_primary", f"{DATA}/kitchen_v3_amb/{scene}/yolo_amb", f"{DATA}/kitchen_v3_amb/{scene}/yolo_amb49"):
        ab += sorted(glob.glob(f"{d}/*.txt")) + sorted(glob.glob(f"{d}/*.yaml"))
    ab += [f"{DATA}/kitchen_v3_amb/{scene}/holdout_manifest.json", f"{DATA}/kitchen_v3/state_cls_{scene}/summary.json"]
    return [os.path.join(ROOT, r) for r in rel] + ab


def record_path(scene: str) -> str:
    return os.path.join(SIM, "kitchen_v3", scene, "frozen_weights.sha256")


def show(p: str) -> str:
    return os.path.relpath(p, ROOT).replace(os.sep, "/") if p.startswith(ROOT) else p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    ap.add_argument("--coco-pending", action="store_true", help="FloorPlan15: freeze without the COCO sensitivity (TEST stays blocked)")
    ap.add_argument("--add-coco", action="store_true")
    ap.add_argument("--check", action="store_true", help="list missing inputs only")
    a = ap.parse_args(argv)
    out = record_path(a.scene)
    if a.add_coco:
        if a.scene != "FloorPlan15" or not os.path.exists(out):
            print("REFUSED: --add-coco is for an existing FloorPlan15 record")
            return 2
        c = os.path.join(ROOT, f"sim/weights/v3/{a.scene}.yolo26s_coco.pt")
        if not os.path.exists(c):
            print(f"REFUSED: {c} missing")
            return 2
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{sha_file(c)}  {show(c)}\n")
        meta = os.path.join(os.path.dirname(out), "frozen_weights.meta.json")
        m = json.load(open(meta, encoding="utf-8"))
        m.update(coco_pending=False, test_evaluation_blocked=False, coco_added=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(m, open(meta, "w", encoding="utf-8"), indent=1)
        print("COCO sensitivity checkpoint appended; TEST evaluation unblocked")
        return 0
    coco = a.scene == "FloorPlan15" and not a.coco_pending
    files = inputs(a.scene, coco)
    missing = [show(p) for p in files if not os.path.exists(p)]
    if a.check or missing:
        print(json.dumps({"scene": a.scene, "n_inputs": len(files), "missing": missing}, indent=1))
        return 0 if (a.check and not missing) else (2 if missing else 0)
    if os.path.exists(out):
        print(f"REFUSED: {out} exists (a freeze is written once)")
        return 2
    lines = [f"{sha_file(p)}  {show(p)}" for p in files]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    meta = {"scene": a.scene, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "n_files": len(lines), "record_sha256": sha_file(out),
            "coco_pending": a.scene == "FloorPlan15" and a.coco_pending,
            "test_evaluation_blocked": a.scene == "FloorPlan15" and a.coco_pending,
            "rule": "A28 item 9 + A38 item 4; no TEST / CLOSE / LIVE / fused amb TEST / state TEST / holdout before this record"}
    json.dump(meta, open(os.path.join(os.path.dirname(out), "frozen_weights.meta.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
