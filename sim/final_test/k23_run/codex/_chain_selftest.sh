#!/bin/bash
# Offline selftest of codex_chain.sh (Git Bash): all-ok -> CODEX CHAIN DONE; an exploratory failure -> exit 7 after the Sol scores, no DONE line;
# a score failure -> exit 5, no DONE line.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ok=0; bad=0
chk() { if eval "$2"; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1"; fi; }
scenario() {   # $1 = explore K2 rc, $2 = score rc for test-k2-v1
  T=$(mktemp -d)
  echo "2026-10-21 02:04:20 START" > "$T/plan-test-k2-v1-sol.log"; echo "2026-10-21 03:20:00 END rc=0" >> "$T/plan-test-k2-v1-sol.log"
  echo '{"event": "plan_first_complete"}' > "$T/ev3.jsonl"; echo '{}' > "$T/DONE.json"
  cat > "$T/plan.sh" <<S
#!/bin/bash
rc=0; [ "\$1 \$2" = "test-k2-v1 explore" ] && rc=$1
echo "x START" > "$T/plan-\$1-\$2.log"; echo "x END rc=\$rc" >> "$T/plan-\$1-\$2.log"; exit \$rc
S
  cat > "$T/exec.sh" <<S
#!/bin/bash
echo "\$1" >> "$T/exec_calls"; exit 0
S
  cat > "$T/score.sh" <<S
#!/bin/bash
echo "\$1" >> "$T/score_calls"; [ "\$1" = test-k2-v1 ] && exit $2; exit 0
S
  CODEX_P="$T" CODEX_PLAN="bash $T/plan.sh" CODEX_EXEC="bash $T/exec.sh" CODEX_SCORE="bash $T/score.sh" CODEX_MAIN_EVENTS_K3="$T/ev3.jsonl" \
    CODEX_DONE="$T/DONE.json" CODEX_NIGHT=0 CODEX_POLL=1 bash "$HERE/codex_chain.sh"
  RC=$?; TD=$T
}
scenario 0 0
chk "all ok: rc 0" '[ $RC -eq 0 ]'
chk "all ok: DONE line" 'grep -q "CODEX CHAIN DONE" "$TD/chain.log"'
chk "all ok: executions rehearsal, K2, K3 in order" '[ "$(tr "\n" " " < "$TD/exec_calls")" = "rehearsal-k2-v1 test-k2-v1 test-k3-v1 " ]'
chk "all ok: re-scores after the exploratory pass" '[ "$(grep -c . "$TD/score_calls")" -eq 5 ]'
rm -rf "$TD"
scenario 1 0
chk "explore failure: rc 7" '[ $RC -eq 7 ]'
chk "explore failure: no DONE line" '! grep -q "CODEX CHAIN DONE" "$TD/chain.log"'
chk "explore failure: Sol scores still written first" '[ "$(tr "\n" " " < "$TD/score_calls")" = "rehearsal-k2-v1 test-k2-v1 test-k3-v1 " ]'
rm -rf "$TD"
scenario 0 4
chk "score failure: rc 5" '[ $RC -eq 5 ]'
chk "score failure: no DONE line" '! grep -q "CODEX CHAIN DONE" "$TD/chain.log"'
chk "score failure: K3 not executed after the failure" '! grep -q test-k3-v1 "$TD/exec_calls"'
rm -rf "$TD"
echo "$ok/$((ok+bad))"
[ $bad -eq 0 ]
