#!/bin/bash
# K23 PRACTICE strict runner (WSL Ubuntu-22.04) for the overnight runner (k23_overnight.py; Astra K23-OVERNIGHT-r1 item 1, r2 item 1, r3 A3):
# ONE try of practice/inner_strict.sh (k23_practice.py under hang_guard + its 4 h timeout, as inner.sh runs it, but WITHOUT --steal-lock and
# with the cleanup limited to its own process group) under the ONE sim lock. The lock is waited for at most <lock_wait> seconds and never
# stolen. Nothing runs unless this launch still holds its one-time authorization: the token file must exist before queuing, and it is CLAIMED
# (atomic rename to <token>.claimed) only after the lock is held and right before the try -- a token the runner revoked (renamed away after a
# start stall) means exit 74 and nothing runs. There is NO restart loop (run.sh restarts after a non-zero exit; here the first failure is
# returned). Logs: sim/final_test/k23_run/practice/<namespace>/{progress,stdout,stderr}.txt
# Exit: k23_practice's rc (0 done, 2 refused, ...); 73 = the sim lock was not acquired within <lock_wait> s; 74 = the authorization is gone;
# 1 = no rc recorded.
# usage: run_strict.sh <practice-k2-v1|practice-k3-v1> <lock_wait_s> <token (repo-relative)>     (SIM_LOCK: selftest override of the lock)
set -u
NS=${1:-}; LW=${2:-}; TOKEN=${3:-}
case "$NS" in practice-k2-v1|practice-k3-v1) ;; *) echo "run_strict.sh: not a K23 practice namespace: $NS"; exit 2;; esac
case "$LW" in ''|*[!0-9]*) echo "run_strict.sh: the lock wait must be a number of seconds"; exit 2;; esac
case "$TOKEN" in sim/final_test/k23_run/overnight/auth/*.token) ;; *) echo "run_strict.sh: a launch authorization token is required"; exit 2;; esac
REPO=$(pwd)                                    # the checkout launch_strict.py checked (wsl --cd); never another tree
[ -f "$REPO/sim/final_test/matrix/k23_practice.py" ] || { echo "run_strict.sh: not run from a repo root ($REPO)"; exit 2; }
OUT="$REPO/sim/final_test/k23_run/practice/$NS"
mkdir -p "$OUT"
if [ ! -f "$REPO/$TOKEN" ]; then
  echo "$(date '+%F %T') STRICT: launch authorization $TOKEN is gone (revoked or used): nothing run" >> "$OUT/progress.txt"; exit 74
fi
HOST=$(ip route show default | awk '{print $3}')
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=60
PY=${PY:-/home/arash/thor-env/bin/python}
LOCK=${SIM_LOCK:-/home/arash/sim.lock}
cd "$REPO"
rm -f "$OUT/last_rc" "$OUT/pgid"
echo "$(date '+%F %T') strict try queued ($NS, lock wait ${LW}s, ollama $HOST, token $TOKEN)" >> "$OUT/progress.txt"
export OUT HOST PY NS
flock -w "$LW" "$LOCK" bash -c 'mv -- "$0" "$0.claimed" 2>/dev/null || exit 74; exec bash "$1"' "$REPO/$TOKEN" \
    "$REPO/sim/final_test/k23_run/practice/inner_strict.sh"
frc=$?
rc=$(cat "$OUT/last_rc" 2>/dev/null || echo missing); rm -f "$OUT/last_rc"
if [ "$rc" = "missing" ]; then
  if [ "$frc" -eq 1 ]; then
    echo "$(date '+%F %T') STRICT: the sim lock was not acquired within ${LW}s (not stolen, nothing run)" >> "$OUT/progress.txt"; exit 73
  fi
  if [ "$frc" -eq 74 ]; then
    echo "$(date '+%F %T') STRICT: the launch authorization was revoked while waiting for the lock: nothing run" >> "$OUT/progress.txt"; exit 74
  fi
  echo "$(date '+%F %T') STRICT: no rc recorded by inner_strict.sh (flock rc $frc)" >> "$OUT/progress.txt"; exit 1
fi
echo "$(date '+%F %T') STRICT RUN END rc=$rc (no restart)" >> "$OUT/progress.txt"
exit "$rc"
