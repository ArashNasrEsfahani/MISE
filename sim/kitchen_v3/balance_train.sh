#!/bin/bash
# balance_train.sh <phase> <scene> [model] — called by balance_train.ps1 on a fresh WSL.
#   phase balance : top-ups (train episodes until every type / state head / combination is met, cap 24; then 12+12
#                   close-protocol held-out), audit, detection gate + yolo split lists
#   phase smoke   : 1-epoch train + eval with 10 live poses into <root>/_weights — proves the chain before the real run
#   phase train   : train + evaluate one model (test, close test, live)
cd "/mnt/d/Personal/Master's Project"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
LOG="/mnt/d/Personal/Master's Project/sim/kitchen_v3/balance_train.log"
pkill -f '[t]hor-Linux64'
if [ "$1" = "balance" ]; then
  /home/arash/thor-env/bin/python -u sim/build_kitchen_data.py run --scenes "$2" --stages topup,audit,yolo >> "$LOG" 2>&1
elif [ "$1" = "smoke" ]; then
  /home/arash/thor-env/bin/python -u sim/train_kitchen.py both --scene "$2" --model "$3" --epochs 1 --live-poses 10 \
      --root /home/arash/thor-data/kitchen_v3 >> "$LOG" 2>&1
elif [ "$1" = "eval" ]; then
  /home/arash/thor-env/bin/python -u sim/train_kitchen.py eval --scene "$2" --model "$3" >> "$LOG" 2>&1
else
  /home/arash/thor-env/bin/python -u sim/train_kitchen.py both --scene "$2" --model "$3" >> "$LOG" 2>&1
fi
rc=$?
pkill -f '[t]hor-Linux64'
echo "PHASE $1 $2 $3 EXIT $rc $(date '+%F %T')" >> "$LOG"
exit $rc
