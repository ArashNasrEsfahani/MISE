#!/bin/bash
# Offline selftest of run_codex_exec.sh (WSL): exit-75 recovery with --steal-lock, plain failure, refusal to steal while a Codex execute lives.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d); ok=0; bad=0
chk() { if eval "$2"; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1"; fi; }
cat > "$T/stub.sh" <<'S'
#!/bin/bash
echo "$*" >> "$STUB_CALLS"
n=$(wc -l < "$STUB_CALLS")
[ "$n" -le "${STUB_75:-0}" ] && exit 75
exit "${STUB_RC:-0}"
S
run() { CODEX_PY=/bin/bash CODEX_ENTRY="$T/stub.sh" CODEX_SIM_LOCK="$T/lock" CODEX_OUT="$T" CODEX_RETRY_SLEEP=0 STUB_CALLS="$T/calls" bash "$HERE/run_codex_exec.sh" ns-x; }
# A: one hang (75), then success: two calls, the second with --steal-lock, wrapper rc 0
rm -f "$T/calls" "$T/exec-ns-x.log"; STUB_75=1 STUB_RC=0 run; rc=$?
chk "A rc 0" '[ $rc -eq 0 ]'
chk "A two calls" '[ "$(wc -l < "$T/calls")" -eq 2 ]'
chk "A first without steal" '! head -1 "$T/calls" | grep -q -- --steal-lock'
chk "A second with steal" 'tail -1 "$T/calls" | grep -q -- --steal-lock'
chk "A log END rc=0" 'grep -q "END rc=0" "$T/exec-ns-x.log"'
# B: a plain failure (rc 3): one call, wrapper rc 3, no retry
rm -f "$T/calls" "$T/exec-ns-x.log"; STUB_75=0 STUB_RC=3 run; rc=$?
chk "B rc 3" '[ $rc -eq 3 ]'
chk "B one call" '[ "$(wc -l < "$T/calls")" -eq 1 ]'
# C: a hang, while another Codex execute process is alive: the retry refuses to steal (76), nothing run
rm -f "$T/calls" "$T/exec-ns-x.log"
bash -c 'exec -a "python sim/final_test/matrix/k23_codex.py execute --namespace other" sleep 30' & FAKE=$!
sleep 1
STUB_75=1 STUB_RC=0 run; rc=$?
kill $FAKE 2>/dev/null
chk "C rc 76" '[ $rc -eq 76 ]'
chk "C only the first call ran" '[ "$(wc -l < "$T/calls")" -eq 1 ]'
chk "C refusal logged" 'grep -q "refusing to steal" "$T/exec-ns-x.log"'
rm -rf "$T"
echo "$ok/$((ok+bad))"
[ $bad -eq 0 ]
