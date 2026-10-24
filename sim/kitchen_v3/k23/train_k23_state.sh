#!/bin/bash
# K23 state-model launcher (A28 item 7, A38; unattended training with the heat watch). State axes have no resume (A28 item 10):
# a stopped axis restarts, finished axes are skipped (k23_state_train.py).
# usage: bash sim/kitchen_v3/k23/train_k23_state.sh <scene>   (WSL; launched DETACHED)
SCENE=$1
cd "/mnt/d/Personal/Master's Project"
D=sim/kitchen_v3/k23/$SCENE/state; mkdir -p "$D"; LOG="$D/train_state.log"; HEAT="$D/heat_state.csv"
echo "$(date '+%F %T') START $SCENE state" >> "$LOG"
/home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_state_train.py train --scene "$SCENE" >> "$LOG" 2>&1 &
PID=$!
hot=0; miss=0
while kill -0 $PID 2>/dev/null; do
  t=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date '+%F %T'),$t" >> "$HEAT"
  c=${t%%,*}
  # Astra K23-CORUN-r1 item 7: lost telemetry is not "cool" -- 5 consecutive unreadable samples stop the job (like heat)
  if ! [[ "$c" =~ ^[0-9]+$ ]]; then miss=$((miss+1)); else miss=0; fi
  if [ "${miss:-0}" -ge 5 ]; then echo "$(date '+%F %T') TELEMETRY LOSS: no GPU temperature for 5 samples -> job stopped (resume like a heat pause)" >> "$LOG"; kill $PID; break; fi
  if [ "${c:-0}" -ge 87 ] 2>/dev/null; then hot=$((hot+1)); else hot=0; fi
  if [ $hot -ge 5 ]; then echo "$(date '+%F %T') HEAT PAUSE: >=87 C for 5 samples -> state training stopped (axis restarts)" >> "$LOG"; kill $PID; break; fi
  sleep 60
done
wait $PID; RC=$?
echo "$(date '+%F %T') END rc=$RC" >> "$LOG"
exit $RC
