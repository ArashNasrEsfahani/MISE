#!/bin/bash
# K23 rehearsal / test live runner (WSL Ubuntu-22.04), modelled on sim/final_test/test_run/run_test.sh (Kitchen 1): k23_run.py <args>
# under the ONE sim lock (waits for it; never kills a holder) + hang_guard + a per-stage timeout. One try per launch (capture / execute
# resume from their journals). Inside WSL k23_run runs the phase itself (preflight: K23 guard incl. the kitchen's agent freeze for test,
# deps, lock). Capture: --ollama http://<windows host>:11434 is appended (the args must end with the pass-through separator --).
# usage: run_k23.sh <job> <timeout_s> <k23_run args...>
set -u
JOB=$1; TMO=$2; shift 2
REPO=$(pwd)                                    # the checkout launch_k23.py chose (wsl --cd); never another tree
[ -f "$REPO/sim/final_test/matrix/k23_run.py" ] || { echo "run_k23.sh: not run from a repo root ($REPO)"; exit 12; }
OUT="$REPO/sim/final_test/k23_run/logs/$JOB"
mkdir -p "$OUT"
HOST=$(ip route show default | awk '{print $3}')
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=${HANG_SECS:-60}
export PYTHONPATH="$REPO/sim/final_test/matrix:$REPO/sim"
PY=/home/arash/thor-env/bin/python
cd "$REPO"
OLL=()
case " $* " in *" --phase capture "*)
  case " $* " in *" -- "*) OLL=(--ollama "http://$HOST:11434");; *) echo "run_k23.sh: capture args must end with -- (pass-through)"; exit 12;; esac;;
esac
echo "$(date '+%F %T') queued k23_run $* ${OLL[*]} (timeout ${TMO}s)" >> "$OUT/progress.txt"
flock /home/arash/sim.lock bash -c '
  echo "$(date "+%F %T") lock acquired" >> "$0/progress.txt"
  timeout "$1" "$2" sim/final_test/matrix/hang_guard.py sim/final_test/matrix/k23_run.py "${@:3}" >> "$0/stdout.txt" 2>> "$0/stderr.txt"
  rc=$?; echo $rc > "$0/last_rc"; echo "$(date "+%F %T") exit $rc" >> "$0/progress.txt"
  pkill -9 -f "[t]hor-Linux64" 2>/dev/null; exit 0' "$OUT" "$TMO" "$PY" "$@" "${OLL[@]}"
rc=$(cat "$OUT/last_rc" 2>/dev/null || echo missing); rm -f "$OUT/last_rc"
echo "$(date '+%F %T') RUN END rc=$rc" >> "$OUT/progress.txt"
exit "$( [ "$rc" = "missing" ] && echo 1 || echo "$rc" )"
