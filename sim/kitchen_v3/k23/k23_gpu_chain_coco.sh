#!/bin/bash
# K23 GPU chain stage 3: resume the Kitchen 2 COCO-start sensitivity (A38 item 4, FloorPlan15 only) once the given kitchen's
# state-model chain (k23_gpu_chain_state.sh) is DONE with rc=0, so the GPU goes straight on to it and the run is not forgotten.
# usage: bash sim/kitchen_v3/k23/k23_gpu_chain_coco.sh <scene whose chain it follows>   (WSL; launch DETACHED from PowerShell)
# train_k23.sh resumes from ~/thor-data/kitchen_v3/_ckpt/FloorPlan15.yolo26s_coco/last.pt (paused after epoch 33 of 60).
SCENE=$1
cd "/mnt/d/Personal/Master's Project"
L=sim/kitchen_v3/k23/$SCENE/chain_gpu.log
echo "$(date '+%F %T') CHAIN-COCO START (waiting for the state chain to finish with rc 0)" >> "$L"
# match only the exact line k23_gpu_chain_state.sh writes (2026-10-19: a loose grep matched this script's own START line)
until grep -qE "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} state END rc=0; CHAIN-STATE DONE$" "$L"; do
  sleep 120
done
# never two trainings at once
while pgrep -f "k23_train.py train|k23_state_train.py train|k23_amb_train.py train" >/dev/null; do sleep 60; done
echo "$(date '+%F %T') state chain done -> K2 COCO-start sensitivity (resume)" >> "$L"
bash sim/kitchen_v3/k23/train_k23.sh FloorPlan15 yolo26s_coco; RC=$?
echo "$(date '+%F %T') K2 COCO END rc=$RC; CHAIN-COCO DONE" >> "$L"
exit $RC
