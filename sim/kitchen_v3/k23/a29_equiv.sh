#!/bin/bash
# A29.1 item 4: the FA equivalence run of a kitchen (10 registered items, reference plans only, no model calls), under the simulator lock +
# hang guard (exit 75 relaunches the same command). usage: bash sim/kitchen_v3/k23/a29_equiv.sh <FloorPlan15|FloorPlan17>   (WSL)
K=$1
cd "/mnt/d/Personal/Master's Project" || exit 3
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=60 PYTHONDONTWRITEBYTECODE=1
PY=/home/arash/thor-env/bin/python
L=sim/kitchen_v3/k23/$K/a29_driver.log
echo "$(date '+%F %T') A29 FA-EQUIVALENCE START $K" >> "$L"
for try in $(seq 1 5); do
  flock /home/arash/sim.lock $PY sim/final_test/matrix/hang_guard.py sim/final_test/matrix/a29_applicability.py --kitchen "$K" --equivalence >> sim/kitchen_v3/k23/$K/a29_equiv.log 2>&1
  rc=$?; pkill -9 -f '[t]hor-Linux64' 2>/dev/null
  echo "$(date '+%F %T') equivalence try $try rc=$rc" >> "$L"
  [ $rc -eq 75 ] && continue
  break
done
echo "$(date '+%F %T') A29 FA-EQUIVALENCE END $K rc=$rc" >> "$L"
exit $rc
