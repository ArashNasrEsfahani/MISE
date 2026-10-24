#!/bin/bash
# A29 test-item applicability check for one replication kitchen (a29_applicability.py; Astra K23-A29-APPLIC-r1 / K23-A29-RUNNER-r1),
# under the simulator lock + hang guard. Exit 75 (hang) re-launches the SAME command (the attempt is repeated without counting, per the
# runner); any other non-zero exit stops (INVALID / infrastructure stay for review, never retried here).
# usage: bash sim/kitchen_v3/k23/a29_run.sh <FloorPlan15|FloorPlan17> [round]   (WSL; launch DETACHED from PowerShell)
K=$1; ROUND=${2:-1}
cd "/mnt/d/Personal/Master's Project" || exit 3
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=60
PY=/home/arash/thor-env/bin/python
L=sim/kitchen_v3/k23/$K/a29_driver.log
echo "$(date '+%F %T') A29 START $K round $ROUND" >> "$L"
for try in $(seq 1 10); do
  flock /home/arash/sim.lock $PY sim/final_test/matrix/hang_guard.py sim/final_test/matrix/a29_applicability.py --kitchen "$K" --round "$ROUND" \
      >> sim/kitchen_v3/k23/$K/a29_run.log 2>&1
  rc=$?
  pkill -9 -f '[t]hor-Linux64' 2>/dev/null
  echo "$(date '+%F %T') a29 try $try rc=$rc" >> "$L"
  [ $rc -eq 75 ] && continue
  break
done
$PY sim/final_test/matrix/a29_applicability.py --kitchen "$K" --summarize >> "$L" 2>&1
echo "$(date '+%F %T') A29 END $K rc=$rc" >> "$L"
exit $rc
