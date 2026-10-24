#!/bin/bash
# K23 PRACTICE live runner (WSL Ubuntu-22.04), modelled on sim/final_test/diag/practice_v2/run.sh (Kitchen 1): k23_practice.py under the
# ONE sim lock + hang guard, order 0, K = 10. Restarts after a hang-guard abort (75) or another non-zero exit while the journal grows;
# stops after 3 restarts without progress. Logs: sim/final_test/k23_run/practice/<namespace>/{progress,stdout,stderr}.txt
# usage: run.sh <practice-k2-v1|practice-k3-v1> [extra k23_practice args, e.g. --limit 1]
set -u
NS=$1; shift
case "$NS" in practice-k2-v1|practice-k3-v1) ;; *) echo "run.sh: not a K23 practice namespace: $NS"; exit 2;; esac
REPO=$(pwd)                                    # the checkout launch.py checked (wsl --cd); never another tree
[ -f "$REPO/sim/final_test/matrix/k23_practice.py" ] || { echo "run.sh: not run from a repo root ($REPO)"; exit 2; }
OUT="$REPO/sim/final_test/k23_run/practice/$NS"
mkdir -p "$OUT"
J="$REPO/docs/final-test/matrix/$NS/practice.jsonl"
HOST=$(ip route show default | awk '{print $3}')
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 HANG_SECS=60
PY=/home/arash/thor-env/bin/python
cd "$REPO"
jsize() { if [ -f "$J" ]; then stat -c %s "$J"; else echo 0; fi; }
stuck=0; n=0
while true; do
  j0=$(jsize); n=$((n+1))
  echo "$(date '+%F %T') try $n queued ($NS, journal $j0 B, ollama $HOST)" >> "$OUT/progress.txt"
  export OUT HOST PY NS
  flock /home/arash/sim.lock bash "$REPO/sim/final_test/k23_run/practice/inner.sh" "$@"
  rc=$(cat "$OUT/last_rc" 2>/dev/null || echo missing); rm -f "$OUT/last_rc"
  if [ "$rc" = "0" ]; then echo "$(date '+%F %T') RUN DONE" >> "$OUT/progress.txt"; exit 0; fi
  if [ "$rc" = "2" ]; then echo "$(date '+%F %T') REFUSED by k23_practice (rc 2, nothing run): see stderr.txt" >> "$OUT/progress.txt"; exit 2; fi
  if [ "$(jsize)" = "$j0" ]; then stuck=$((stuck+1)); else stuck=0; fi
  echo "$(date '+%F %T') restart (rc $rc, without progress $stuck)" >> "$OUT/progress.txt"
  if [ $stuck -ge 3 ]; then echo "$(date '+%F %T') GIVING UP: 3 tries without journal growth" >> "$OUT/progress.txt"; exit 3; fi
done
