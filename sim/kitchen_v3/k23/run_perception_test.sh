#!/bin/bash
# Kitchen 2/3 perception TEST evaluations (k23_perception_test.py), INSIDE WSL, repo root. A28 item 2: FloorPlan15 then FloorPlan17,
# never at the same time as another GPU job. Each step runs through run_perception_step.sh under the exclusive /home/arash/sim.lock
# with its own ownership token (preflight GPU / process checks, the registered thermal guard proven ready and supervised, the step in
# its own verified session, owned-session cleanup only). Steps already done (ledger) are skipped; the chain stops at the first
# non-zero exit (a refusal, a failed step, a thermal pause or a guard failure is inspected by hand; another attempt of a started step
# needs `k23_perception_test.py recover`).
# Usage: bash sim/kitchen_v3/k23/run_perception_test.sh [FloorPlan15|FloorPlan17|both]
set -u
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 9
export PYTHONDONTWRITEBYTECODE=1
PY=/home/arash/thor-env/bin/python
LOCK=/home/arash/sim.lock
K=sim/kitchen_v3/k23
LOG=$K/perception_test.log
WHICH=${1:-both}
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }
note "START ($WHICH)"
for S in FloorPlan15 FloorPlan17; do
  [ "$WHICH" = both ] || [ "$WHICH" = "$S" ] || continue
  D=$K/$S/perception_test
  mkdir -p "$D/guard"
  if [ ! -f "$D/eval_code.sha256" ]; then
    $PY $K/k23_perception_test.py bind --scene "$S" >> "$D/run.log" 2>&1 || { note "STOP: $S bind failed"; echo "END rc=2" >> "$LOG"; exit 2; }
  fi
  STEPS="poses primary coco live fused state holdout"
  [ "$S" = FloorPlan17 ] && STEPS="poses primary live fused state holdout"
  for ST in $STEPS; do
    if $PY $K/k23_perception_test.py status --scene "$S" | $PY -c "import json,sys; sys.exit(0 if '$ST' in json.load(sys.stdin)['done'] else 1)"; then
      note "$S $ST: already done"; continue
    fi
    case $ST in poses|live) TO=5400 ;; *) TO=7200 ;; esac
    STAMP=$(date +%Y%m%dT%H%M%S)
    note "$S $ST: waiting for the sim lock (stamp $STAMP)"
    flock -w 14400 "$LOCK" env KCAP_OWNER="k23p:$S:$ST:$STAMP" setsid --wait bash $K/run_perception_step.sh "$S" "$ST" "$TO" "$STAMP"
    rc=$?
    note "$S $ST rc=$rc (details: $D/guard/step_${ST}_$STAMP.log)"
    [ "$rc" -eq 0 ] || { note "STOP at $S $ST (rc $rc)"; echo "END rc=$rc" >> "$LOG"; exit "$rc"; }
  done
  note "$S: all perception TEST steps done"
done
note "END rc=0"
