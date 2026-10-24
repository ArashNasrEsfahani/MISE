#!/bin/bash
# K23 specialist / control launcher (A28 item 6, A38; Arash 'Approve 1-5' item 5: unattended training with a heat watch).
# usage: bash sim/kitchen_v3/k23/train_k23_amb.sh <scene> <amb26m|control>   (WSL; launch DETACHED from PowerShell)
SCENE=$1; MODEL=$2
cd "/mnt/d/Personal/Master's Project"
D=sim/kitchen_v3/k23/$SCENE/amb; mkdir -p "$D"; LOG="$D/train_$MODEL.log"; HEAT="$D/heat_$MODEL.csv"
echo "$(date '+%F %T') START $SCENE $MODEL" >> "$LOG"
/home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_amb_train.py train --scene "$SCENE" --model "$MODEL" >> "$LOG" 2>&1 &
PID=$!
# checkpoint backup out of /tmp (a WSL restart empties /tmp; train_amb._train resumes from ~/thor-data/kitchen_v3/_ckpt)
bash sim/kitchen_v3/ckpt_backup.sh >/dev/null 2>&1 &
CKPID=$!
hot=0; miss=0
while kill -0 $PID 2>/dev/null; do
  t=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date '+%F %T'),$t" >> "$HEAT"
  c=${t%%,*}
  # Astra K23-CORUN-r1 item 7: lost telemetry is not "cool" -- 5 consecutive unreadable samples stop the job (like heat)
  if ! [[ "$c" =~ ^[0-9]+$ ]]; then miss=$((miss+1)); else miss=0; fi
  if [ "${miss:-0}" -ge 5 ]; then echo "$(date '+%F %T') TELEMETRY LOSS: no GPU temperature for 5 samples -> job stopped (resume like a heat pause)" >> "$LOG"; kill $PID; break; fi
  if [ "${c:-0}" -ge 87 ] 2>/dev/null; then hot=$((hot+1)); else hot=0; fi
  if [ $hot -ge 5 ]; then echo "$(date '+%F %T') HEAT PAUSE: >=87 C for 5 samples -> training stopped (resumes from its checkpoint)" >> "$LOG"; kill $PID; break; fi
  sleep 60
done
wait $PID; RC=$?; kill $CKPID 2>/dev/null; echo "$(date '+%F %T') END rc=$RC" >> "$LOG"
exit $RC
