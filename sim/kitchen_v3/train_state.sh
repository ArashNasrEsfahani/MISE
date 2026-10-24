#!/bin/bash
# Train the per-axis state classifiers (sim/train_state.py); log to ~/train_state.log.
cd /home/arash
/home/arash/thor-env/bin/python "/mnt/d/Personal/Master's Project/sim/train_state.py" "$@" > /home/arash/train_state.log 2>&1
echo "STATE TRAIN EXIT $? $(date '+%F %T')" >> /home/arash/train_state.log
