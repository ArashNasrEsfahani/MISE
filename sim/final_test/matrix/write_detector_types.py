"""WSL helper: read the class list of the frozen primary detector weights and write docs/final-test/matrix/detector_types.json (gate 12.12).
Names the exact weights sha; freeze.py accepts the file only on a match. No GPU, no simulator."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402
from ultralytics import YOLO                             # noqa: E402

# K23 B5 (Astra K23-AGENT-r1): --scene selects the kitchen whose FROZEN primary is read; FloorPlan1 (default) writes the same file as before,
# another kitchen writes detector_types_<scene>.json (never overwriting Kitchen 1's).
SCENE = sys.argv[sys.argv.index("--scene") + 1] if "--scene" in sys.argv else "FloorPlan1"
w = os.path.join(C.ROOT, "sim", "weights", "v3", f"{SCENE}.yolo26s.pt")
os.chdir(os.path.dirname(w))        # ultralytics strips the apostrophe of the repo path; load by relative name
names = sorted(set(dict(YOLO(os.path.basename(w)).names).values()))
out = {"schema": "ftm.detector-types/1", "weights": f"sim/weights/v3/{SCENE}.yolo26s.pt", "weights_sha": C.sha256_file(w), "types": names, "n": len(names)}
if SCENE != "FloorPlan1":
    out["scene"] = SCENE
p = os.path.join(C.paths(C.ROOT)["matrix_docs"], "detector_types.json" if SCENE == "FloorPlan1" else f"detector_types_{SCENE}.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("wrote", p, len(names))
