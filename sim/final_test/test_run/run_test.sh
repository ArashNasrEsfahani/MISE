#!/bin/bash
# KITCHEN 1 LOCKED TEST (test-v1) live runner (WSL Ubuntu-22.04), modelled on acceptance_v2/run/{run,inner}.sh and fg_wsl.sh:
# matrix_run.py <args> under the ONE sim lock (waits for it; never kills a holder) + hang_guard + a per-stage timeout. One try per launch
# (capture / execute resume from their journals). Inside WSL matrix_run runs the phase itself (preflight: test guard, deps, lock, C5 energy gate).
# usage: run_test.sh <job> <timeout_s> <matrix_run args...>   (capture: --ollama http://<windows host>:11434 is appended to the phase args)
set -u
JOB=$1; TMO=$2; shift 2
REPO=$(ls -d /mnt/d/Personal/Master*/ | head -1)
OUT="$REPO/sim/final_test/test_run/logs/$JOB"
mkdir -p "$OUT"
HOST=$(ip route show default | awk '{print $3}')
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=${HANG_SECS:-60}
export PYTHONPATH="$REPO/sim/final_test/matrix:$REPO/sim"
PY=/home/arash/thor-env/bin/python
cd "$REPO"
OLL=()
case " $* " in *" --phase capture "*) OLL=(--ollama "http://$HOST:11434");; esac
echo "$(date '+%F %T') queued matrix_run $* ${OLL[*]} (timeout ${TMO}s)" >> "$OUT/progress.txt"
flock /home/arash/sim.lock bash -c '
  echo "$(date "+%F %T") lock acquired" >> "$0/progress.txt"
  timeout "$1" "$2" sim/final_test/matrix/hang_guard.py sim/final_test/matrix/matrix_run.py "${@:3}" >> "$0/stdout.txt" 2>> "$0/stderr.txt"
  rc=$?; echo $rc > "$0/last_rc"; echo "$(date "+%F %T") exit $rc" >> "$0/progress.txt"
  pkill -9 -f "[t]hor-Linux64" 2>/dev/null; exit 0' "$OUT" "$TMO" "$PY" "$@" "${OLL[@]}"
rc=$(cat "$OUT/last_rc" 2>/dev/null || echo missing); rm -f "$OUT/last_rc"
echo "$(date '+%F %T') RUN END rc=$rc" >> "$OUT/progress.txt"
exit "$( [ "$rc" = "missing" ] && echo 1 || echo "$rc" )"
