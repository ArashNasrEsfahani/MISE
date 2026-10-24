#!/bin/bash
# K23 GPU chain stage 2: the kitchen's 7 state models after the specialist + control chain (k23_gpu_chain.sh) is DONE.
# usage: bash sim/kitchen_v3/k23/k23_gpu_chain_state.sh <scene>   (WSL; launch DETACHED from PowerShell)
# State axes have no resume (A28 item 10): a stopped axis restarts; finished axes are skipped (k23_state_train.py).
SCENE=$1
cd "/mnt/d/Personal/Master's Project"
L=sim/kitchen_v3/k23/$SCENE/chain_gpu.log
D=sim/kitchen_v3/k23/$SCENE/state; mkdir -p "$D"; LOG="$D/train_state.log"; HEAT="$D/heat_state.csv"
echo "$(date '+%F %T') CHAIN-STATE START $SCENE (waiting for CHAIN DONE with control rc=0)" >> "$L"
until grep -q "control END rc=0; CHAIN DONE" "$L"; do
  sleep 120
done
echo "$(date '+%F %T') control done -> state models" >> "$L"
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
echo "$(date '+%F %T') state END rc=$RC; CHAIN-STATE DONE" >> "$L"
exit $RC
