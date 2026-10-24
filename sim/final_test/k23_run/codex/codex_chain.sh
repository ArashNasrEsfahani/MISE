#!/bin/bash
# K23 CODEX ADDENDUM orchestrator (Windows Git Bash, repo root; decision record docs/final-test/k23/decisions/K23-CODEX-ADDENDUM.json).
#  1 K2 Sol planning (launched separately 02:04)
#  2 exploratory planning (plan-only) in ONE background subshell: K2, then K3 after K3's Sol pass; off the Sol critical path, its exit status kept
#  3 K3 Sol planning once the MAIN test-k3-v1 has its plan_first_complete
#  4 Sol executions (WSL, sim lock) ONLY after the overnight runner's DONE.json (both main tests complete): never queued on the sim lock while a
#    main chain may need it; a live rehearsal smoke (rehearsal-k2-v1) first, then test-k2-v1, test-k3-v1; each scored at once
#  5 after the exploratory subshell: its status checked, then a re-score adds the plan-only diagnostics; CODEX CHAIN DONE only if all succeeded
# Stops at the first failure (the main tests are never touched). Log: $P/chain.log
# Env overrides for the selftest only: CODEX_P, CODEX_PLAN, CODEX_EXEC, CODEX_SCORE, CODEX_MAIN_EVENTS_K3, CODEX_DONE, CODEX_NIGHT, CODEX_POLL
cd "$(dirname "$0")/../../../.." || exit 9
P=${CODEX_P:-sim/final_test/k23_run/codex}
L=$P/chain.log
PLAN=${CODEX_PLAN:-"bash sim/final_test/k23_run/codex/run_codex_plan.sh"}
EXEC=${CODEX_EXEC:-"wsl_exec"}
SCORE=${CODEX_SCORE:-"python sim/final_test/matrix/k23_codex.py score --namespace"}
EV3=${CODEX_MAIN_EVENTS_K3:-docs/final-test/matrix/test-k3-v1/events.jsonl}
DONE=${CODEX_DONE:-sim/final_test/k23_run/overnight/DONE.json}
NIGHT=${CODEX_NIGHT:-$(date -d "2026-10-21 00:00" +%s)}
POLL=${CODEX_POLL:-60}
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
note() { echo "$(date '+%F %T') $*" >> "$L"; }
endrc() { grep -a -o 'END rc=[0-9]*' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }
wsl_exec() { MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd "/mnt/d/Personal/Master's Project" -- bash sim/final_test/k23_run/codex/run_codex_exec.sh "$1"; }
score() { $SCORE "$1" >> "$P/score-$1.log" 2>&1; }
note "START"
until [ -n "$(endrc "$P/plan-test-k2-v1-sol.log")" ]; do sleep "$POLL"; done
rc=$(endrc "$P/plan-test-k2-v1-sol.log"); note "K2 Sol planning END rc=$rc"
[ "$rc" = 0 ] || { note "STOP: K2 Sol planning failed"; exit 2; }
( $PLAN test-k2-v1 explore 2; r2=$?
  until [ -n "$(endrc "$P/plan-test-k3-v1-sol.log")" ]; do sleep "$POLL"; done
  r3=9
  [ "$(endrc "$P/plan-test-k3-v1-sol.log")" = 0 ] && { $PLAN test-k3-v1 explore 2; r3=$?; }
  echo "$(date '+%F %T') exploratory subshell: K2 rc=$r2 K3 rc=$r3" >> "$L"
  [ "$r2" = 0 ] && [ "$r3" = 0 ] ) &
XPL=$!
note "exploratory planning (K2, then K3 after its Sol pass) in the background (pid $XPL)"
until grep -q '"event": "plan_first_complete"' "$EV3" 2>/dev/null; do sleep "$POLL"; done
note "main test-k3-v1 plan_first_complete seen: K3 Sol planning"
$PLAN test-k3-v1 sol 2; rc=$?
note "K3 Sol planning rc=$rc"; [ "$rc" = 0 ] || { note "STOP: K3 Sol planning failed"; exit 3; }
until [ -f "$DONE" ] && [ "$(date -r "$DONE" +%s)" -ge "$NIGHT" ]; do sleep "$POLL"; done
note "overnight runner DONE (both main tests complete): Sol executions, rehearsal smoke first"
for NS in rehearsal-k2-v1 test-k2-v1 test-k3-v1; do
  $EXEC "$NS"; rc=$?
  note "Sol execution $NS rc=$rc"; [ "$rc" = 0 ] || { note "STOP: Sol execution $NS failed"; exit 4; }
  score "$NS"; rc=$?
  note "addendum score $NS rc=$rc"; [ "$rc" = 0 ] || { note "STOP: addendum score $NS failed"; exit 5; }
done
note "Sol results scored for both kitchens (score.codex.jsonl / summary.codex.json kept); waiting for the exploratory planning"
wait "$XPL"; xrc=$?
note "exploratory subshell exit $xrc"
[ "$xrc" = 0 ] || { note "STOP: exploratory planning failed (Sol results stand; no plan-only diagnostics re-score)"; exit 7; }
for NS in test-k2-v1 test-k3-v1; do
  score "$NS"; rc=$?
  note "re-score $NS with the exploratory diagnostics rc=$rc"; [ "$rc" = 0 ] || { note "STOP: re-score $NS failed"; exit 6; }
done
note "CODEX CHAIN DONE"
