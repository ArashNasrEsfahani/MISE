#!/bin/bash
# After the K2 COCO run: one step of the FloorPlan15 COCO freeze append, by name (called from k23_after_coco.ps1).
#   bash sim/kitchen_v3/k23/k23_coco_freeze_steps.sh add-coco | meta-write | meta-inspect
cd "/mnt/d/Personal/Master's Project" || exit 3
export PYTHONDONTWRITEBYTECODE=1
PY=/home/arash/thor-env/bin/python
case "$1" in
  add-coco)     $PY sim/kitchen_v3/k23/k23_freeze.py --scene FloorPlan15 --add-coco ;;
  meta-write)   $PY sim/kitchen_v3/k23/k23_coco_meta.py --write ;;
  meta-inspect) $PY sim/kitchen_v3/k23/k23_coco_meta.py ;;
  *) echo "usage: k23_coco_freeze_steps.sh add-coco|meta-write|meta-inspect"; exit 12 ;;
esac
