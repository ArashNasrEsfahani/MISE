#!/bin/bash
# One kitchen: probe (skipped if present), capture (resumes per episode), audit. Called by capture_all.ps1 after a
# fresh `wsl --shutdown`, so every kitchen starts on a clean WSLg display (the red/white/segmentation-colour
# corruption of 2026-09-28 came from a display degraded by earlier forced restarts).
cd "/mnt/d/Personal/Master's Project"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
LOG="/mnt/d/Personal/Master's Project/sim/kitchen_v3/capture_all.log"
pkill -f '[t]hor-Linux64'
/home/arash/thor-env/bin/python -u sim/build_kitchen_data.py run --scenes "$1" --stages probe,capture,audit >> "$LOG" 2>&1
rc=$?
pkill -f '[t]hor-Linux64'
echo "SCENE $1 EXIT $rc $(date '+%F %T')" >> "$LOG"
exit $rc
