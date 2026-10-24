#!/bin/bash
# K3 GPU chain after the 2026-10-19 stop (the primary finished at 16:24; k23_gpu_chain.sh cannot be relaunched because it requires
# primary weights newer than its own start). Waits until no simulator job runs (the K2 A29 check), then: specialist (+ the K3 look-alike
# map on CPU beside it, then its provenance supplement), control audit, control; on control rc 0 it writes the exact line
# k23_gpu_chain_state.sh waits for. Its own log lines never contain that line's text.
# usage: bash sim/kitchen_v3/k23/k23_k3_resume_chain.sh   (WSL; launch DETACHED from PowerShell)
SCENE=FloorPlan17
cd "/mnt/d/Personal/Master's Project"
L=sim/kitchen_v3/k23/$SCENE/chain_gpu.log
PY=/home/arash/thor-env/bin/python
echo "$(date '+%F %T') K3-RESUME START (waiting until no simulator job / training runs)" >> "$L"
while pgrep -f "[a]29_applicability.py|[a]29_run.sh|[t]hor-Linux64|[k]23_train.py train|[k]23_amb_train.py train|[k]23_state_train.py train" >/dev/null; do sleep 60; done
echo "$(date '+%F %T') K3-RESUME: specialist starts; look-alike map on CPU beside it" >> "$L"
( LOG=sim/kitchen_v3/k23/$SCENE/lookalike.log
  echo "$(date '+%F %T') look-alike map (CPU) started by the resume chain" >> "$LOG"
  PYTHONDONTWRITEBYTECODE=1 $PY sim/kitchen_v3/k23/k23_lookalike.py --scene $SCENE >> "$LOG" 2>&1; RC=$?
  echo "$(date '+%F %T') look-alike map rc=$RC" >> "$LOG"
  if [ $RC -eq 0 ]; then
    PYTHONDONTWRITEBYTECODE=1 $PY -c "import sys,json; sys.path.insert(0,'sim'); import honest_skills as hs; d=hs.build_lookalike_provenance('FloorPlan17','sim/kitchen_v3/FloorPlan17/val_errors_yolo26s.json','sim/step5/lookalike_map_FloorPlan17.json'); json.dump(d, open('sim/step5/lookalike_provenance_FloorPlan17.json','w',encoding='utf-8',newline='\n'), indent=1, sort_keys=True); print('provenance written')" >> "$LOG" 2>&1
    echo "$(date '+%F %T') provenance rc=$? ; LOOKALIKE DONE" >> "$LOG"
  fi ) &
bash sim/kitchen_v3/k23/train_k23_amb.sh "$SCENE" amb26m; RC=$?
echo "$(date '+%F %T') K3-RESUME specialist rc=$RC" >> "$L"
[ $RC -ne 0 ] && { echo "$(date '+%F %T') K3-RESUME STOP (specialist rc=$RC; resume it with train_k23_amb.sh)" >> "$L"; exit $RC; }
$PY sim/kitchen_v3/k23/k23_amb_train.py audit --scene "$SCENE" --model control >> "$L" 2>&1 || { echo "$(date '+%F %T') K3-RESUME STOP (control audit failed)" >> "$L"; exit 3; }
bash sim/kitchen_v3/k23/train_k23_amb.sh "$SCENE" control; RC=$?
if [ $RC -eq 0 ]; then
  echo "$(date '+%F %T') control END rc=0; CHAIN DONE" >> "$L"
else
  echo "$(date '+%F %T') K3-RESUME control failed rc=$RC (state models not released)" >> "$L"
fi
exit $RC
