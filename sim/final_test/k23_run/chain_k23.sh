#!/bin/bash
# =====================================================================================================================================
# KITCHENS 2 / 3 -- rehearsal or test chain for ONE K23 namespace, modelled on sim/final_test/test_run/chain_test.sh (Kitchen 1).
# Stages in order, stop at the FIRST non-zero rc, RESUMABLE (re-run: a stage whose phase is already complete in `k23_run.py --phase
# status` is skipped; capture / plan / execute resume from their journals). Infrastructure only: no prompt, package, threshold, item or rule.
#
#   bash sim/final_test/k23_run/chain_k23.sh rehearsal-k2-v1                 (Windows Git Bash, repo root; after practice-k2-v1)
#   bash sim/final_test/k23_run/chain_k23.sh test-k2-v1                      (ONLY after the kitchen's agent freeze; K2 also after --add-coco)
#   (the test plan meters Gemma energy; ENERGY=off ENERGY_WAIVER="<Arash's note>" is the recorded waiver path; execute / score refuse an
#    incomplete energy coverage without a waiver, so an ENERGY INCOMPLETE plan stops the chain at execute)
#
# Stages: 0 preflight (Windows, read-only: k23_run capture --dry-run = the full K23 guard, for test incl. the kitchen's agent freeze and
#   the K23 entry checks; the A29 study-wide check study_wide_problems(0) == []; Ollama up) -> 1 capture (WSL, sim lock + hang_guard;
#   canon, --runs 1, no FS / curve) -> 2 plan (Windows, simulator off; Gemma then synchronous Opus; test: --energy $ENERGY) -> 3 execute
#   (WSL; a FRESH simulator per 300 executions, approve-all-6 item 3; --no-replay) -> 4 score -> rehearsal only: 5 G-iso (WSL, n = 12)
#   and 6 context fit (Windows, Gemma server, measured): the freeze evidence of gates g_iso / context. Analysis by hand: k23_analyze.py.
# Logs: sim/final_test/k23_run/<ns>-chain.log; logs/<job>/{progress,stdout,stderr}.txt per stage.
# =====================================================================================================================================
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 3      # this checkout (the launchers run their WSL side here too)
export PYTHONIOENCODING=utf-8
NS=${1:-}
case "$NS" in
  rehearsal-k2-v1|rehearsal-k3-v1) T_FLAG=(); KIND=rehearsal;;
  test-k2-v1|test-k3-v1) T_FLAG=(--allow-test); KIND=test
     ENERGY=${ENERGY:-on}                                   # test-v1 parity: the C5 meter is mandatory (Astra K23-TOOLING-G2 1)
     case "$ENERGY" in on) ;; off) [ -n "${ENERGY_WAIVER:-}" ] || { echo "chain_k23.sh: ENERGY=off needs Arash's ENERGY_WAIVER text"; exit 12; };;
       *) echo "chain_k23.sh: ENERGY must be on (default) or off (with ENERGY_WAIVER)"; exit 12;; esac;;
  *) echo "usage: [ENERGY=on|off] chain_k23.sh <rehearsal-k2-v1|rehearsal-k3-v1|test-k2-v1|test-k3-v1>"; exit 12;;
esac
EW=()                                                       # Arash's energy waiver, forwarded to plan, execute and score (Astra G3 1)
[ "${KIND}" = test ] && [ "${ENERGY:-on}" = off ] && EW=(--energy-waiver "$ENERGY_WAIVER")
D=sim/final_test/k23_run
M=sim/final_test/matrix
LOG=$D/$NS-chain.log
J=${NS//-v1/}; J=${J//rehearsal-/r}; J=${J//test-/t}
mkdir -p "$D/logs"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
win() { local job=$1; shift; mkdir -p "$D/logs/$job"
  echo "$(date '+%F %T') start $*" >> "$D/logs/$job/progress.txt"
  "$@" >> "$D/logs/$job/stdout.txt" 2>> "$D/logs/$job/stderr.txt"; local rc=$?
  echo "$(date '+%F %T') RUN END rc=$rc" >> "$D/logs/$job/progress.txt"; return $rc; }
wsl_job() { local n=0 rc
  while true; do
    python $D/launch_k23.py "$@" >> "$D/logs/$1.launcher.txt" 2>&1; rc=$?
    [ $rc -eq 75 ] && [ $n -lt "${HANG_RELAUNCH:-0}" ] || return $rc
    n=$((n + 1)); say "  $1: hang_guard rc 75 -> relaunch $n of ${HANG_RELAUNCH} (resumes from the journal)"
  done; }
step() { local name=$1; shift; say "STAGE $name: $*"; "$@"; local rc=$?; say "STAGE $name rc=$rc"; [ $rc -eq 0 ] || { say "CHAIN STOPPED at $name (rc $rc)"; exit $rc; }; }
done_phase() { python - "$NS" "$1" <<'PY'
import json, subprocess, sys
o = subprocess.run([sys.executable, "sim/final_test/matrix/k23_run.py", "--phase", "status", "--namespace", sys.argv[1]], capture_output=True, text=True)
ph = json.loads(o.stdout)["phases"][sys.argv[2]]
sys.exit(0 if ph.get("complete") and ph.get("unchanged") is not False else 1)
PY
}
preflight() {
  done_phase capture || { python $M/k23_run.py --phase capture --namespace $NS "${T_FLAG[@]}" --dry-run > /dev/null || return $?; }
  python - <<'PY' || return 9
import sys
sys.path.insert(0, "sim/final_test/matrix")
import acc_common as AC, k23_ns as K
sw = AC.study_spend(None)
bad = AC.study_wide_problems(0.0, spend=sw) + K.a37_hold_problems()
print("study-wide committed", sw["committed"], "USD; problems", bad)
sys.exit(0 if not bad else 1)
PY
  curl -sf -m 10 http://127.0.0.1:11434/api/version > /dev/null || { echo "Ollama not reachable at 127.0.0.1:11434"; return 8; }
}
capture() { done_phase capture && { say "  capture already complete (phase_complete)"; return 0; }
  wsl_job $J-capture 21600 --phase capture --namespace $NS "${T_FLAG[@]}" --; }
plan() { done_phase plan && { say "  plan already complete (phase_complete)"; return 0; }
  if [ "$KIND" = test ] && [ "$ENERGY" = off ]; then win $J-plan python $M/k23_run.py --phase plan --namespace $NS --allow-test --energy off --energy-waiver "$ENERGY_WAIVER" --parallel 4
  elif [ "$KIND" = test ]; then win $J-plan python $M/k23_run.py --phase plan --namespace $NS --allow-test --energy on --parallel 4
  else win $J-plan python $M/k23_run.py --phase plan --namespace $NS --parallel 4; fi; }
execute() { done_phase execute && { say "  execute already complete (phase_complete)"; return 0; }
  local rc=0
  while true; do
    say "  execute session start (fresh simulator; stop after 300 executions)"
    wsl_job $J-execute 172800 --phase execute --namespace $NS "${T_FLAG[@]}" "${EW[@]}" --inject-interrupt 300; rc=$?
    say "  execute session end rc=$rc"
    done_phase execute && return 0
    { [ $rc -eq 0 ] || [ $rc -eq 5 ]; } || return $rc
    sleep 20
  done; }
score() { python - "$NS" <<'PY' && { say "  score already sealed complete"; return 0; }
import json, subprocess, sys
o = subprocess.run([sys.executable, "sim/final_test/matrix/k23_run.py", "--phase", "status", "--namespace", sys.argv[1]], capture_output=True, text=True)
sys.exit(0 if json.loads(o.stdout)["phases"]["score"].get("complete") else 1)
PY
  win $J-score python $M/k23_run.py --phase score --namespace $NS "${T_FLAG[@]}" "${EW[@]}"; }
giso() { python - "$NS" <<'PY' && { say "  G-iso already passed"; return 0; }
import json, os, sys
sys.path.insert(0, "sim/final_test/matrix")
import common as C
p = os.path.join(C.ns_docs(sys.argv[1]), "g_iso.json")
sys.exit(0 if os.path.exists(p) and json.load(open(p, encoding="utf-8")).get("status") == "pass" else 1)
PY
  wsl_job $J-giso 21600 --phase execute --namespace $NS -- --g-iso; }
context() { python - "$NS" <<'PY' && { say "  context fit already measured on this capture"; return 0; }
import json, os, sys
sys.path.insert(0, "sim/final_test/matrix")
import common as C
ns = sys.argv[1]
p, cp = os.path.join(C.ns_docs(ns), "context_fit.json"), C.journal_path(ns, "capture")
ok = os.path.exists(p) and os.path.exists(cp)
d = json.load(open(p, encoding="utf-8")) if ok else {}
sys.exit(0 if ok and d.get("freeze_grade") is True and d.get("capture_sha") == C.sha256_file(cp) else 1)
PY
  win $J-context python $M/k23_run.py --phase context --namespace $NS --ollama http://127.0.0.1:11434; }
say "CHAIN START $NS ($KIND)"
step preflight preflight
step capture   capture
step plan      plan
step execute   execute
step score     score
if [ "$KIND" = rehearsal ]; then
  step giso    giso                                   # 12.6 G-iso, live, in this kitchen (n = 12) -- freeze evidence
  step context context                                # A24 1 / 7(i) context fit, measured on the Gemma server -- freeze evidence
fi
say "CHAIN DONE $NS (next, by hand for test: k23_analyze.py)"
