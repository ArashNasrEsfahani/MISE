#!/bin/bash
# K23 PRACTICE: the part that runs INSIDE the sim lock (called by run.sh via flock). Paths come from the environment, never from a quoted
# string (the repo path's apostrophe). k23_practice.py re-checks readiness (both A29 samples, final package, perception freeze, pins).
set -u
echo "$(date '+%F %T') lock acquired ($NS)" >> "$OUT/progress.txt"
timeout 14400 "$PY" sim/final_test/matrix/hang_guard.py sim/final_test/matrix/k23_practice.py --namespace "$NS" \
    --ollama "http://$HOST:11434" --steal-lock "$@" >> "$OUT/stdout.txt" 2>> "$OUT/stderr.txt"
rc=$?
echo $rc > "$OUT/last_rc"
echo "$(date '+%F %T') exit $rc" >> "$OUT/progress.txt"
pkill -9 -f '[t]hor-Linux64' 2>/dev/null
exit 0
