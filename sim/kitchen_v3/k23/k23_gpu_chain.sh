#!/bin/bash
# K23 GPU chain for one kitchen's specialist + control, so the GPU is never idle between jobs.
# usage: bash sim/kitchen_v3/k23/k23_gpu_chain.sh <scene>   (WSL; launch DETACHED from PowerShell)
# Waits until the kitchen's primary training ended with rc=0 AND wrote its weights, then runs the specialist, then the control.
# A primary that ended with rc!=0 (heat pause, crash) keeps the chain waiting: the primary is resumed first.
SCENE=$1
cd "/mnt/d/Personal/Master's Project"
L=sim/kitchen_v3/k23/$SCENE/chain_gpu.log
PLOG=sim/kitchen_v3/k23/$SCENE/train/train_yolo26s.log
W=sim/weights/v3/$SCENE.yolo26s.pt
T0=$(date +%s)
echo "$(date '+%F %T') CHAIN START $SCENE (waiting for the primary)" >> "$L"
while true; do
  last=$(grep -E "START|END rc=" "$PLOG" | tail -1)
  if echo "$last" | grep -q "END rc=0" && [ -f "$W" ] && [ $(stat -c %Y "$W") -gt $T0 ] && ! pgrep -f "k23_train.py train" >/dev/null; then break; fi
  sleep 120
done
echo "$(date '+%F %T') primary done -> specialist" >> "$L"
bash sim/kitchen_v3/k23/train_k23_amb.sh "$SCENE" amb26m; RC=$?
echo "$(date '+%F %T') specialist END rc=$RC" >> "$L"
[ $RC -ne 0 ] && { echo "$(date '+%F %T') CHAIN STOP (specialist rc=$RC; resume it with train_k23_amb.sh)" >> "$L"; exit $RC; }
/home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_amb_train.py audit --scene "$SCENE" --model control >> "$L" 2>&1 || { echo "$(date '+%F %T') CHAIN STOP (control audit failed)" >> "$L"; exit 3; }
bash sim/kitchen_v3/k23/train_k23_amb.sh "$SCENE" control; RC=$?
echo "$(date '+%F %T') control END rc=$RC; CHAIN DONE" >> "$L"
exit $RC
