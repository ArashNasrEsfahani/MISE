#!/bin/bash
# amb_train.sh <lcurve|control|eval|holdout> — the specialist pipeline step, logged to amb.log. Resumable.
cd "/mnt/d/Personal/Master's Project"
LOG="/mnt/d/Personal/Master's Project/sim/kitchen_v3/amb.log"
/home/arash/thor-env/bin/python -u sim/train_amb.py "$1" >> "$LOG" 2>&1
rc=$?
echo "AMB STEP $1 EXIT $rc $(date '+%F %T')" >> "$LOG"
exit $rc
