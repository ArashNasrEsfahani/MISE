"""Kitchen 2/3 look-alike map M (A28 item 8; decision 13 #6 with decision 12): the Kitchen 1 rule applied UNCHANGED.

  1. sim/kitchen_v3/error_breakdown.py (frozen; module global DS pointed at the kitchen's yolo_primary, process-local) runs the
     kitchen's FINAL primary on its VAL split only (TEST never read), operating confidence .25, top-3 / greatest-overlap rule;
     output sim/kitchen_v3/<scene>/val_errors_yolo26s.json (the path honest_skills.AMBIG_MAP_FILE names);
  2. honest_skills.freeze_ambiguous_map(scene=<scene>) (frozen; theta .02, Knife excluded, surfaces excluded) writes
     sim/step5/lookalike_map_<scene>.json with its provenance. Run ONCE per kitchen; refused if the map exists.
CPU inference (CUDA hidden) so it can run next to GPU training.

    python sim/kitchen_v3/k23/k23_lookalike.py --scene FloorPlan15        (WSL thor-env, after the primary is final)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "kitchen_v3"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU: the GPU may be training (this process starts no simulator)
    import honest_skills as hs
    map_out = os.path.join(SIM, hs.LOOKALIKE_MAP_FILE.format(kitchen=a.scene))
    if os.path.exists(map_out):
        print(f"REFUSED: {map_out} exists (the map is built once and frozen)")
        return 2
    w = os.path.join(SIM, "weights", "v3", f"{a.scene}.yolo26s.pt")
    if not os.path.exists(w):
        print(f"REFUSED: the final primary {w} does not exist yet")
        return 2
    wsha = hashlib.sha256(open(w, "rb").read()).hexdigest()
    tmp = os.path.join(tempfile.gettempdir(), f"k23_{a.scene}.yolo26s.pt")   # apostrophe-free path for ultralytics
    shutil.copyfile(w, tmp)
    import error_breakdown as EB
    EB.DS = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3", a.scene, "yolo_primary")
    src = os.path.join(SIM, hs.AMBIG_MAP_FILE.format(scene=a.scene))
    os.makedirs(os.path.dirname(src), exist_ok=True)
    EB.main(tmp, src)
    doc = json.load(open(src, encoding="utf-8"))
    doc["weights_sha256"], doc["weights_file"] = wsha, f"sim/weights/v3/{a.scene}.yolo26s.pt"
    json.dump(doc, open(src, "w", encoding="utf-8"), indent=1)
    out = hs.freeze_ambiguous_map(scene=a.scene)
    print(json.dumps({"scene": a.scene, "map": out["map"], "source_sha256": out["source_sha256"], "weights_sha256": wsha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
