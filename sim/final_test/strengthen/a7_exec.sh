#!/bin/bash
# Task A6: ONE execution chunk of one pass INSIDE WSL (simulator), under the shared sim flock (the strengthen GPU lock is held by the
# Windows caller), hang-guarded; a hang exit (75) is retried in the same chunk up to 3 times with --steal-lock (resumable per unit).
# Leftover simulator processes of THIS chunk's process group are killed while the flock is held. Exit code = the chunk's:
# 0 pass complete, 10 more remains, 77 global STOP seen, 4 invariant / integrity abort, other = error.
# Usage (WSL, repo root): bash sim/final_test/strengthen/a7_exec.sh <namespace> <new|rerun|noise> <max_seconds>
set -u
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 9
NS=$1
KIND=$2
SECS=${3:-540}
OUT="$REPO/sim/final_test/strengthen/work/A7"
mkdir -p "$OUT"
LOG="$OUT/a7_exec-$NS-$KIND.log"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HANG_SECS=${HANG_SECS:-60}
export PYTHONPATH="$REPO/sim/final_test/matrix:$REPO/sim"
PY=/home/arash/thor-env/bin/python
LOCK=/home/arash/sim.lock
for i in $(seq 1 90); do [ -S /tmp/.X11-unix/X0 ] && break; sleep 2; done
[ -S /tmp/.X11-unix/X0 ] || { echo "$(date '+%F %T') no X display: not starting" >> "$LOG"; exit 8; }
STEAL=""
rc=1
for t in 1 2 3; do
  echo "$(date '+%F %T') try $t: a7 $KIND $NS (${SECS}s chunk; waiting for the sim flock) $STEAL" >> "$LOG"
  flock -w 3600 "$LOCK" bash -c '
    if [ "$4" = "--steal-lock" ] && pgrep -f "[a]7_exec[.]py execute" >/dev/null 2>&1; then
      echo "$(date "+%F %T") an A6 execution is alive inside the flock: refusing to steal" >> "$0"; exit 76
    fi
    setsid --wait bash -c "echo \$\$ > \"\$0\"; exec \"\$@\"" "$1.pgid" timeout 1800 "$2" sim/final_test/matrix/hang_guard.py sim/final_test/strengthen/a7_exec.py execute --ns "$3" --kind "$5" --max-seconds "$6" $4 >> "$0" 2>&1
    rc=$?
    PG=$(cat "$1.pgid" 2>/dev/null); rm -f "$1.pgid"; [ -n "$PG" ] && pkill -9 -g "$PG" -f "thor-Linux64|hang_guard[.]py|a7_exec[.]py" 2>/dev/null
    exit $rc' "$LOG" "$OUT/a7_exec-$NS-$KIND" "$PY" "$NS" "$STEAL" "$KIND" "$SECS"
  rc=$?
  echo "$(date '+%F %T') try $t rc=$rc" >> "$LOG"
  [ "$rc" -eq 75 ] || break
  STEAL="--steal-lock"
  sleep 20
done
exit $rc
