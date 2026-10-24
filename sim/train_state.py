"""Train the per-axis state classifiers (YOLO11s-cls, 128 px) on sim/state_data.py crops; copy best weights to
sim/weights/v3/<scene>.state_<axis>.pt. Whole-crop training (scale=0: no random crop away of the state cue).

    ~/thor-env/bin/python sim/train_state.py [--axes open,on,...]
"""
import argparse, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yolo_recipe
from ultralytics import YOLO

DATA = os.path.expanduser("~/thor-data/kitchen_v3/state_cls")
OUT = os.path.join(HERE, "weights", "v3")
AXES = ("open", "on", "sliced", "cooked", "broken", "dirty", "filled")

ap = argparse.ArgumentParser()
ap.add_argument("--axes", default=",".join(AXES))
ap.add_argument("--scene", default="FloorPlan1")
a = ap.parse_args()
for ax in a.axes.split(","):
    t = time.time()
    m = YOLO(yolo_recipe.checkpoint_path("yolo11s-cls.pt"))
    m.train(data=os.path.join(DATA, ax), imgsz=128, epochs=30, patience=10, batch=128, seed=0, deterministic=True,
            scale=0.0, erasing=0.0, auto_augment=None, fliplr=0.5, mosaic=0.0, workers=8, device=0,
            project="/tmp/state_runs", name=ax, exist_ok=True, plots=False, verbose=False)
    best = os.path.join("/tmp/state_runs", ax, "weights", "best.pt")
    shutil.copyfile(best, os.path.join(OUT, f"{a.scene}.state_{ax}.pt"))
    r = YOLO(best).val(data=os.path.join(DATA, ax), split="val", imgsz=128, batch=128, verbose=False, plots=False)
    print(f"STATE {ax}: val top1 {r.top1:.3f} ({time.time() - t:.0f} s)", flush=True)
