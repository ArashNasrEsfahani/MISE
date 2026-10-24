#!/bin/bash
# Re-score both primary detectors on the two live sets (realistic + coverage). Simulator only — run on a fresh WSL.
cd "/mnt/d/Personal/Master's Project"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
LOG="/mnt/d/Personal/Master's Project/sim/kitchen_v3/balance_train.log"
echo "START live rescore $(date '+%F %T')" >> "$LOG"
for m in yolo26s yolo11s; do
  pkill -f '[t]hor-Linux64'
  /home/arash/thor-env/bin/python -u sim/train_kitchen.py live --scene FloorPlan1 --model $m >> "$LOG" 2>&1
  echo "PHASE live FloorPlan1 $m EXIT $? $(date '+%F %T')" >> "$LOG"
done
pkill -f '[t]hor-Linux64'
