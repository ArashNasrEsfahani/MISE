#!/bin/bash
# Calibrate + evaluate the state classifiers on detector crops (sim/state_eval.py); log to ~/train_state.log.
cd /home/arash
/home/arash/thor-env/bin/python "/mnt/d/Personal/Master's Project/sim/state_eval.py" >> /home/arash/train_state.log 2>&1
echo "STATE EVAL EXIT $? $(date '+%F %T')" >> /home/arash/train_state.log
