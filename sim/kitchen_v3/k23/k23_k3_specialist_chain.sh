#!/bin/bash
# K3 GPU chain, attempt 2 of the 2026-10-20 resume (k23_k3_resume_chain.sh stopped: the specialist refused because its A38 transfer
# audit had never been written for FloorPlan17; written 12:12, 768/768 loaded). Same stages without the look-alike map (already running):
# specialist (audit if missing) -> control audit -> control; on control rc 0 the exact line k23_gpu_chain_state.sh waits for.
# usage: bash sim/kitchen_v3/k23/k23_k3_specialist_chain.sh   (WSL; launch DETACHED from PowerShell)
SCENE=FloorPlan17
cd "/mnt/d/Personal/Master's Project"
L=sim/kitchen_v3/k23/$SCENE/chain_gpu.log
PY=/home/arash/thor-env/bin/python
while pgrep -f "[a]29_applicability.py|[a]29_run.sh|[t]hor-Linux64|[k]23_train.py train|[k]23_amb_train.py train|[k]23_state_train.py train" >/dev/null; do sleep 30; done
[ -f sim/kitchen_v3/k23/$SCENE/amb/a38_transfer_audit_amb26m.json ] || $PY sim/kitchen_v3/k23/k23_amb_train.py audit --scene "$SCENE" --model amb26m >> "$L" 2>&1 || { echo "$(date '+%F %T') K3-CHAIN2 STOP (specialist audit failed)" >> "$L"; exit 3; }
echo "$(date '+%F %T') K3-CHAIN2: specialist starts (audit present)" >> "$L"
bash sim/kitchen_v3/k23/train_k23_amb.sh "$SCENE" amb26m; RC=$?
echo "$(date '+%F %T') K3-CHAIN2 specialist rc=$RC" >> "$L"
[ $RC -ne 0 ] && { echo "$(date '+%F %T') K3-CHAIN2 STOP (specialist rc=$RC; resume it with train_k23_amb.sh)" >> "$L"; exit $RC; }
$PY sim/kitchen_v3/k23/k23_amb_train.py audit --scene "$SCENE" --model control >> "$L" 2>&1 || { echo "$(date '+%F %T') K3-CHAIN2 STOP (control audit failed)" >> "$L"; exit 3; }
bash sim/kitchen_v3/k23/train_k23_amb.sh "$SCENE" control; RC=$?
if [ $RC -eq 0 ]; then
  echo "$(date '+%F %T') control END rc=0; CHAIN DONE" >> "$L"
else
  echo "$(date '+%F %T') K3-CHAIN2 control failed rc=$RC (state models not released)" >> "$L"
fi
exit $RC
