#!/bin/bash
# K23 Codex addendum: one planning pass (Windows Python, repo root). Usage: run_codex_plan.sh <namespace> <sol|explore> [codex_parallel]
cd "$(dirname "$0")/../../../.." || exit 9
NS=$1; ARMS=$2; P=${3:-3}
LOG=sim/final_test/k23_run/codex/plan-$NS-$ARMS.log
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') START plan $NS $ARMS codex_parallel=$P" >> "$LOG"
python sim/final_test/matrix/k23_codex.py plan --namespace "$NS" --arms "$ARMS" --codex-parallel "$P" >> "$LOG" 2>&1
rc=$?
echo "$(date '+%F %T') END rc=$rc" >> "$LOG"
exit $rc
