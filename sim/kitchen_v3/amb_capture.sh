#!/bin/bash
# Ambiguity (specialist) capture, protocol amb-v1 (docs/ambiguity-spec.md). Resumable per episode.
# amb_capture.sh [scene]  -- scene defaults to FloorPlan1 (unchanged Kitchen 1 behaviour; K23-PREP 2026-10-14).
# FloorPlan15 / FloorPlan17 use sim/kitchen_v3/k23/capture_k23.ps1 -Pipeline amb (lock, batches, QC), not this script.
SCENE="${1:-FloorPlan1}"
cd "/mnt/d/Personal/Master's Project"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
LOG="/mnt/d/Personal/Master's Project/sim/kitchen_v3/amb.log"
pkill -f '[t]hor-Linux64'
/home/arash/thor-env/bin/python -u sim/build_kitchen_data.py run --scenes "$SCENE" --stages amb,yolo_amb \
    --root /home/arash/thor-data/kitchen_v3_amb >> "$LOG" 2>&1
rc=$?
pkill -f '[t]hor-Linux64'
echo "AMB CAPTURE $SCENE EXIT $rc $(date '+%F %T')" >> "$LOG"
exit $rc
