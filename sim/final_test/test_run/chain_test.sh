#!/bin/bash
# =====================================================================================================================================
# KITCHEN 1 LOCKED TEST -- launch chain (namespace test-v1, treatment ft-v2, identity 7a2ee211). PREPARED 2026-10-16, NOT LAUNCHED.
# Modelled on acceptance_v2/run/chain_a2.sh: stages in order, stop at the FIRST non-zero rc, RESUMABLE (re-run this script: a stage whose
# phase is already complete in `matrix_run.py --phase status` is skipped; capture / plan / execute resume from their journals).
# Infrastructure only: no prompt, package, threshold, item or rule. Start it ONLY after Arash's `freeze.py --record-arash-check`.
#
# SINGLE LAUNCH COMMAND (Windows Git Bash, repo root; two background jobs):
#   bash sim/final_test/test_run/chain_test.sh &  bash sim/final_test/test_run/thermal_test.sh &
#
# Stages (registered W5 order; SPEC s1/s9; A27 workload; A29 batch; planfirst REGISTERED_ORDER):
#   0 preflight (Windows, read-only, <1 min): matrix_run capture --dry-run = the full test guard (validated freeze, Arash's check,
#     --allow-test, families); A29 study-wide check study_wide_problems(0) == [] + gate 12.7 fake over-cap evidence PASS; Ollama up.
#   1 capture (WSL, sim lock + hang_guard, ~1-1.5 h, 0 USD; free count_tokens): 250 test presentations canon + 20 relocation
#     (--runs 1 --worlds canon,reloc), FS / S* / curve (o0 K=1,3,5) blocks, A12 coverage, snapshot / freeze sha checks.
#   2 plan (Windows, simulator OFF): planfirst.py (A37 2026-10-16: SYNCHRONOUS normal API, no --batch; ceilings 60 / 100, tripwire 0.080, cap 180), stages free (Gemma via Windows Ollama 127.0.0.1:11434 + gpt-6-sol
#     X0/X+M) -> core0 -> core12 -> fs -> curve -> reloc (Opus, Message Batches API) -> codex (exploratory, plan-only). C5 energy meter
#     is mandatory (energy auto = on in test-v1). Opus spend: projection 27.11 core / 45.12 total USD; ceilings 30 / 50 USD; cost
#     guard ftm-guard-3 pauses above 0.040 USD per billed Opus cell (exit 3); A29 study-wide cap 100 USD (spent so far ~6.43).
#   3 execute (WSL, sim lock + hang_guard, ~15-22 h, 0 USD): ~4,053 units after dedup (A27: 5,320 before) incl. P0 / P+M and /p
#     controls, REPEAT core four x2, curve, relocation (--worlds canon,reloc). matrix_run applies the C5 energy gate first.
#   4 score (Windows, minutes): sealed score.jsonl + score.seal.json + plan_diagnostics.json.
#   NOT run here (by hand, W6): matrix_run.py --phase analyze --items test --namespace test-v1 --allow-test
#   Beside it (optional, W5 "every 20 min"): python sim/final_test/matrix/matrix_run.py --phase monitor --items test --namespace test-v1 --allow-test -- --every 1200
#
# Logs: sim/final_test/test_run/test-chain.log (stages), test-thermal.log, logs/<job>/{progress,stdout,stderr}.txt per stage,
#       logs/<job>.launcher.txt (WSL launcher). Data: docs/final-test/matrix/test-v1/, sim/final_test/matrix/work/test-v1/.
# Resume: fix the cause, then run the launch command again. A cost-guard pause (plan rc 3) needs Arash's outcome-blind decision through
#   plan.py (planfirst does not forward --pause-response / --outcome-blind), e.g.
#   python sim/final_test/matrix/plan.py --namespace test-v1 --allow-test --pass core --batch --ack-pause "<>=20 chars: spend/token evidence>" --pause-response continue --outcome-blind
#   A killed plan run: re-run with planfirst --steal-lock and an explicit --allow-replacement or plan.py --reconcile-open (Arash).
#   A hang_guard exit (rc 75) stops the chain like any rc; HANG_RELAUNCH=N (default 0) re-launches a WSL stage up to N times on rc 75 only.
# =====================================================================================================================================
cd "/d/Personal/Master's Project" || exit 3
export PYTHONIOENCODING=utf-8
T=sim/final_test/test_run
M=sim/final_test/matrix
NS=test-v1
LOG=$T/test-chain.log
mkdir -p "$T/logs"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
win() {          # win <job> <command...>  (Windows-side stages)
  local J=$1; shift; mkdir -p "$T/logs/$J"
  echo "$(date '+%F %T') start $*" >> "$T/logs/$J/progress.txt"
  "$@" >> "$T/logs/$J/stdout.txt" 2>> "$T/logs/$J/stderr.txt"; local rc=$?
  echo "$(date '+%F %T') RUN END rc=$rc" >> "$T/logs/$J/progress.txt"; return $rc
}
wsl_job() {      # wsl_job <job> <timeout_s> <matrix_run args...>  (WSL simulator stages via launch_test.py -> run_test.sh -> flock + hang_guard)
  local n=0 rc
  while true; do
    python $T/launch_test.py "$@" >> "$T/logs/$1.launcher.txt" 2>&1; rc=$?
    [ $rc -eq 75 ] && [ $n -lt "${HANG_RELAUNCH:-0}" ] || return $rc
    n=$((n + 1)); say "  $1: hang_guard rc 75 -> relaunch $n of ${HANG_RELAUNCH} (resumes from the journal)"
  done
}
step() { local name=$1; shift; say "STAGE $name: $*"; "$@"; local rc=$?; say "STAGE $name rc=$rc"; [ $rc -eq 0 ] || { say "CHAIN STOPPED at $name (rc $rc)"; exit $rc; }; }
done_phase() {   # read-only: is <phase> complete (and unchanged) per matrix_run status?
  python - "$1" <<'PY'
import json, subprocess, sys
o = subprocess.run([sys.executable, "sim/final_test/matrix/matrix_run.py", "--phase", "status", "--items", "test", "--namespace", "test-v1"],
                   capture_output=True, text=True)
ph = json.loads(o.stdout)["phases"][sys.argv[1]]
sys.exit(0 if ph.get("complete") and ph.get("unchanged") is not False else 1)
PY
}
preflight() {
  python $M/matrix_run.py --phase capture --items test --namespace $NS --allow-test --dry-run -- --runs 1 --worlds canon,reloc --ollama http://127.0.0.1:11434 > /dev/null || return $?
  python - <<'PY' || return 9
import json, sys
sys.path.insert(0, "sim/final_test/matrix")
import acc_common as AC
sw = AC.study_spend(None)
bad = AC.study_wide_problems(0.0, spend=sw)
g = json.load(open("docs/final-test/matrix/gates_ft2/12.7.json", encoding="utf-8"))
fo = (g.get("evidence") or {}).get("fake_overcap") or {}
print("study-wide committed", sw["committed"], "USD; problems", bad, "| 12.7", g.get("status"), "fake over-cap ok", fo.get("ok"))
sys.exit(0 if not bad and g.get("status") == "PASS" and fo.get("ok") is True else 1)
PY
  curl -sf -m 10 http://127.0.0.1:11434/api/version > /dev/null || { echo "Ollama not reachable at 127.0.0.1:11434"; return 8; }
}
capture() { done_phase capture && { say "  capture already complete (phase_complete)"; return 0; }
  wsl_job t-capture 21600 --phase capture --items test --namespace $NS --allow-test -- --runs 1 --worlds canon,reloc; }
plan() { done_phase plan && { say "  plan already complete (phase_complete)"; return 0; }
  win t-plan python $M/planfirst.py --namespace $NS --allow-test --parallel 4 --codex-parallel 2 --ollama http://127.0.0.1:11434; }   # A37: synchronous (normal API), no --batch
execute() { done_phase execute && { say "  execute already complete (phase_complete)"; return 0; }
  python $M/matrix_run.py --phase execute --items test --namespace $NS --allow-test --dry-run -- --worlds canon,reloc > /dev/null || return $?
  # K1-EXEC-RESTART (2026-10-16 ~22:10; Astra K1-EXEC-RESTART-r1 ADOPT WITH CONDITIONS; Arash's delegation): a FRESH simulator process
  # per 300 completed executions (~60 min), stopped at a clean unit boundary by the executor's own --inject-interrupt (exit 5), then
  # relaunched; repeated until execute completes. Outcome-blind fixed count; every process start/stop is in logs/t-execute/progress.txt.
  # The interrupted session exits 0 or 5 depending on the wrapper chain, so the loop runs until the phase is really complete.
  local rc=0
  while true; do
    say "  execute session start (fresh simulator; stop after 300 executions)"
    wsl_job t-execute 172800 --phase execute --items test --namespace $NS --allow-test --inject-interrupt 300 -- --worlds canon,reloc; rc=$?
    say "  execute session end rc=$rc"
    done_phase execute && return 0
    { [ $rc -eq 0 ] || [ $rc -eq 5 ]; } || return $rc
    sleep 20
  done; }
score() { done_phase score && { say "  score already sealed complete"; return 0; }
  win t-score python $M/matrix_run.py --phase score --items test --namespace $NS --allow-test; }
say "CHAIN START test-v1 (Kitchen 1 locked test)"
step preflight preflight
step capture   capture
step plan      plan
step execute   execute
step score     score
say "CHAIN DONE (next, by hand: matrix_run.py --phase analyze --items test --namespace test-v1 --allow-test)"
