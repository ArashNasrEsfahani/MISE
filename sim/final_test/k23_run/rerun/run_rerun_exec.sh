#!/bin/bash
# K2 render-window re-execution (k23_rerun.py): the execution pass of one namespace INSIDE WSL (simulator), under the shared sim lock, hang-guarded, resumed after
# a hang exit (75) up to 5 times (execution is resumable per unit). Leftover simulator processes of THIS job's process group are killed while
# the lock is still held (never another job's). A retry after 75 takes the addendum's dead phase lock (--steal-lock) only if, INSIDE the
# acquired sim lock, no `k23_rerun.py execute` process is alive (else exit 76, nothing run).
# Usage (WSL, repo root): bash run_rerun_exec.sh <namespace>     (env for the selftest only: RERUN_PY, RERUN_SIM_LOCK, RERUN_OUT, RERUN_ENTRY)
set -u
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO" || exit 9
NS=$1
OUT="${RERUN_OUT:-$REPO/sim/final_test/k23_run/rerun}"
LOG="$OUT/exec-$NS.log"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=${HANG_SECS:-60}
export PYTHONPATH="$REPO/sim/final_test/matrix:$REPO/sim"
PY="${RERUN_PY:-/home/arash/thor-env/bin/python}"
LOCK="${RERUN_SIM_LOCK:-/home/arash/sim.lock}"
ENTRY="${RERUN_ENTRY:-sim/final_test/matrix/hang_guard.py sim/final_test/matrix/k23_rerun.py}"
STEAL=""
rc=1
for t in 1 2 3 4 5 6; do
  echo "$(date '+%F %T') try $t: execute $NS (waiting for the sim lock) $STEAL" >> "$LOG"
  flock -w 3600 "$LOCK" bash -c '
    echo "$(date "+%F %T") lock acquired" >> "$0"
    if [ "$4" = "--steal-lock" ] && pgrep -f "[k]23_rerun[.]py execute" >/dev/null 2>&1; then
      echo "$(date "+%F %T") a re-execution process is alive inside the lock: refusing to steal" >> "$0"; exit 76
    fi
    setsid --wait bash -c "echo \$\$ > \"\$0\"; exec \"\$@\"" "$1.pgid" timeout 14400 "$2" $5 execute --namespace "$3" $4 >> "$0" 2>&1
    rc=$?
    PG=$(cat "$1.pgid" 2>/dev/null); rm -f "$1.pgid"; [ -n "$PG" ] && pkill -9 -g "$PG" -f "thor-Linux64|hang_guard[.]py|k23_rerun[.]py" 2>/dev/null
    exit $rc' "$LOG" "$OUT/exec-$NS" "$PY" "$NS" "$STEAL" "$ENTRY"
  rc=$?
  echo "$(date '+%F %T') try $t rc=$rc" >> "$LOG"
  [ "$rc" -eq 75 ] || break
  STEAL="--steal-lock"
  sleep ${RERUN_RETRY_SLEEP:-20}
done
echo "$(date '+%F %T') END rc=$rc" >> "$LOG"
exit $rc
