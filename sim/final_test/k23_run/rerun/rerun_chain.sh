#!/bin/bash
# K2 render-window re-execution chain (Windows Git Bash, repo root; decision docs/final-test/k23/decisions/K23-WINDOW-RERUN.json):
#  1 the 54 window units re-executed in WSL (run_rerun_exec.sh: sim lock, hang guard, resume on 75)
#  2 a renderer check of the live simulator ~3 min in (d3d12 libraries mapped, not llvmpipe / swrast), logged
#  3 re-score + the registered analysis on the substituted rows (Windows)
# Stops at the first failure; the main test is never touched. Log: sim/final_test/k23_run/rerun/chain.log
cd "$(dirname "$0")/../../../.." || exit 9
P=sim/final_test/k23_run/rerun
L=$P/chain.log
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
note() { echo "$(date '+%F %T') $*" >> "$L"; }
note "START"
( sleep 180; MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd "/mnt/d/Personal/Master's Project" -- bash sim/final_test/k23_run/rerun/thor_renderer.sh >> "$P/renderer.log" 2>&1
  note "renderer check written to $P/renderer.log" ) &
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd "/mnt/d/Personal/Master's Project" -- bash sim/final_test/k23_run/rerun/run_rerun_exec.sh test-k2-v1; rc=$?
note "re-execution rc=$rc"; [ "$rc" = 0 ] || { note "STOP: re-execution failed"; exit 2; }
python sim/final_test/matrix/k23_rerun.py score --namespace test-k2-v1 >> "$P/score.log" 2>&1; rc=$?
note "re-score + analysis rc=$rc"; [ "$rc" = 0 ] || { note "STOP: re-score failed"; exit 3; }
note "RERUN CHAIN DONE"
