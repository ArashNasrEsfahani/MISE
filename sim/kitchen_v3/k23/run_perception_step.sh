#!/bin/bash
# ONE Kitchen 2/3 perception TEST step INSIDE the sim lock (called only by run_perception_test.sh:
#   flock <lock> env KCAP_OWNER=k23p:<scene>:<step>:<stamp> setsid --wait bash run_perception_step.sh <scene> <step> <timeout_s> <stamp>).
# The capture runner's verified pattern (capture_inner_k23.sh; Astra K23-PERCEPTION-TEST r2 1-5, r3 1-4):
#   1 preflight, refusing the launch on any doubt: no simulator / training / execution / console process in WSL, the pgrep itself
#     must succeed (70); GPU memory query succeeded, numeric and < 1500 MiB (71); the Windows GPU process enumeration succeeded (72),
#     its FULL list is logged, no ML runtime in it (73) (on WDDM every desktop graphics client is listed, so "any process" cannot be
#     the rule; the whole-GPU memory check covers loaded models)
#   2 the registered thermal guard (thermal_guard.py) started and proven READY before the launch (87)
#   3 the step in its OWN session, led by a k23_procs holder; ownership published as "pid boot_id starttime token" (k23_own.sh) --
#     a failed publication stops the step (99); signals deferred until ownership is known
#   4 supervision: a guard that dies while the step runs (marker or not) stops the owned session (88; a marker is preserved)
#   5 the guard's outcome: exit 0 without a marker, or exit 86 WITH a marker (a pause: 86); anything else (killed, crashed, a marker
#     without 86) -> 88
#   6 cleanup stops ONLY the owned session (k23_procs: pidfd + start time); a failed enumeration or a failed stop is NOT success (89)
#     and the session stays owned for the exit-trap retry
#   7 a supervisor outcome for the attempt (stamp) is written to the ledger (k23_perception_test.py supervisor): the step counts as
#     done only with ok=1 (exit 0 + valid guard outcome + cleanup); a failed write -> 90
# Integration-test overrides are honoured ONLY with K23_TEST_MODE=1 (_run_perception_step_selftest.sh).
S="$1"; ST="$2"; TO="$3"; STAMP="$4"
REPO="/mnt/d/Personal/Master's Project"; cd "$REPO" || exit 4
. "$REPO/sim/kitchen_v3/k23/k23_own.sh" || exit 4
TOKEN="${KCAP_OWNER:-}"
[ "$TOKEN" = "k23p:$S:$ST:$STAMP" ] || { echo "REFUSED: no / wrong KCAP_OWNER token ($TOKEN) for k23p:$S:$ST:$STAMP"; exit 4; }
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TRACE_ENABLED=0 K23P_STAMP="$STAMP"
PY=/home/arash/thor-env/bin/python
FOREIGN='[t]hor-Linux64|[t]rain_kitchen[.]py train|[t]rain_amb[.]py|[t]rain_state|[k]23_train[.]py train|[k]23_amb_train[.]py train|[k]23_state_train[.]py train|[k]23_rerun[.]py execute|[k]23_codex[.]py execute|[w]ebapp[.]py|[k]23_capture[.]py'
DENY='ollama|python|llama|torch|conda|jupyter|comfy|stable|vllm|kobold|lm studio|lmstudio|text-generation'
GPU_CMD=(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
WIN_CMD=(/mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
PROCS_CMD=("$PY" "$REPO/sim/kitchen_v3/k23/k23_procs.py")
HOLD_CMD=("$PY" "$REPO/sim/kitchen_v3/k23/k23_procs.py")
GUARD_CMD=("$PY" sim/kitchen_v3/k23/thermal_guard.py)
SUP_CMD=("$PY" sim/kitchen_v3/k23/k23_perception_test.py supervisor)
JOB=("$PY" -u sim/kitchen_v3/k23/k23_perception_test.py "$ST" --scene "$S")
D="$REPO/sim/kitchen_v3/k23/$S/perception_test"
GARGS_EXTRA=()
if [ "${K23_TEST_MODE:-}" = "1" ]; then
  D="${K23P_OUTDIR:?}"; [ -n "${K23P_SIMPAT:-}" ] && FOREIGN="$FOREIGN|$K23P_SIMPAT"
  [ -n "${K23P_GPU_CMD:-}" ] && GPU_CMD=($K23P_GPU_CMD); [ -n "${K23P_WIN_CMD:-}" ] && WIN_CMD=($K23P_WIN_CMD)
  [ -n "${K23P_PROCS_CMD:-}" ] && PROCS_CMD=($K23P_PROCS_CMD); [ -n "${K23P_GUARD_CMD:-}" ] && GUARD_CMD=($K23P_GUARD_CMD)
  [ -n "${K23P_SUP_CMD:-}" ] && SUP_CMD=($K23P_SUP_CMD)
  [ -n "${K23P_JOB_CMD:-}" ] && JOB=($K23P_JOB_CMD "$ST" --scene "$S")
  [ -n "${K23P_GUARD_ARGS:-}" ] && GARGS_EXTRA=($K23P_GUARD_ARGS)
fi
G="$D/guard"; mkdir -p "$G"
LOG="$G/step_${ST}_$STAMP.log"; RUNLOG="$D/run.log"
MARK="$G/pause_${ST}_$STAMP.json"; READY="$G/ready_${ST}_$STAMP"; PGIDF="$G/pgid_${ST}_$STAMP"; STOPF="$G/stop_${ST}_$STAMP"
CRCF="$G/crc_${ST}_$STAMP"
[ "${K23_TEST_MODE:-}" = "1" ] && [ -n "${K23P_PGIDF:-}" ] && PGIDF="$K23P_PGIDF"
END_IN=$(( $(date +%s) + TO + 300 ))
GPID=""; CPID=""; OWNED_G=""; GRC=""; GUARD_OK=1; CLEAN_OK=1; PENDING_SIG=""; LAUNCHED=0; SUP_DONE=0; CRC=""
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }
left() { echo $(( END_IN - $(date +%s) )); }
bound() { local l; l=$(( $(left) - $2 )); [ "$l" -lt 0 ] && l=0; [ "$1" -lt "$l" ] && echo "$1" || echo "$l"; }
members() {   # the live members of owned session $1; exit status of the enumeration itself is returned (failure != empty)
  local out rc; out=$("${PROCS_CMD[@]}" list --token "$TOKEN" --session "$1" 2>/dev/null); rc=$?; echo "$out"; return $rc
}
kill_child() { [ -n "$2" ] && "${PROCS_CMD[@]}" signal --token "$TOKEN" --pid "$2" --sig "$1" >/dev/null 2>&1; return 0; }
stop_pg() {   # 0 = the session is verified empty; 1 = members left after KILL; 2 = the enumeration failed
  local g="$1" w out m; [ -z "$g" ] && return 0
  m=$(members "$g") || { note "enumeration of session $g FAILED"; return 2; }
  [ -z "$m" ] && return 0
  w=$(bound "$2" 0)
  out=$("${PROCS_CMD[@]}" stop --token "$TOKEN" --session "$g" --grace "$w" --deadline "$END_IN" 2>&1) && return 0
  note "session $g still has members after KILL (or the stop failed): $out"; return 1
}
own_group() {
  local p="$1" i pg sd m
  for i in $(seq 1 50); do
    read -r pg sd < <(ps -o pgid=,sid= -p "$p" 2>/dev/null)
    [ "$pg" = "$p" ] && [ "$sd" = "$p" ] && { echo "$p"; return 0; }
    if ! kill -0 "$p" 2>/dev/null; then m=$(members "$p") || { echo "$p"; return 0; }; if [ -n "$m" ]; then echo "$p"; else echo "-"; fi; return 0; fi
    sleep 0.1
  done
  return 1
}
defer_signals() { trap 'PENDING_SIG=143' TERM; trap 'PENDING_SIG=130' INT; trap 'PENDING_SIG=129' HUP; }
normal_signals() {
  trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
  [ -n "$PENDING_SIG" ] && { note "signal (exit $PENDING_SIG) during an ownership step -- handled now"; exit "$PENDING_SIG"; }
}
end_guard() {
  [ -z "$GPID" ] && return 0
  touch "$STOPF"
  local i
  for i in $(seq 1 "$1"); do kill -0 "$GPID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$GPID" 2>/dev/null; then note "thermal guard did not end within $1 s -- KILLED (guard_killed)"; kill_child KILL "$GPID"; GUARD_OK=0; fi
  wait "$GPID" 2>/dev/null; GRC=$?; GPID=""
}
write_supervisor() {   # $1 ok (0/1), $2 rc
  [ "$SUP_DONE" = "1" ] && return 0
  SUP_DONE=1
  "${SUP_CMD[@]}" --scene "$S" --step "$ST" --stamp "$STAMP" --ok "$1" \
      --detail "{\"rc\": $2, \"guard_rc\": \"${GRC:-none}\", \"guard_ok\": $GUARD_OK, \"cleanup_ok\": $CLEAN_OK}" >> "$LOG" 2>&1
}
on_exit() {
  local rc=$?
  trap '' TERM INT HUP; trap - EXIT
  [ -n "$GPID" ] && touch "$STOPF"
  if [ -n "$OWNED_G" ]; then stop_pg "$OWNED_G" "$(bound 15 60)" || { CLEAN_OK=0; note "exit-trap cleanup of session $OWNED_G FAILED"; }
  elif [ -n "$CPID" ]; then kill_child KILL "$CPID"; fi
  end_guard "$(bound 75 0)"
  [ "$CLEAN_OK" = "0" ] && [ "$rc" = "0" ] && rc=89
  if [ "$LAUNCHED" = "1" ] && [ "$SUP_DONE" != "1" ]; then write_supervisor 0 "$rc" || rc=90; fi
  note "END $S $ST rc=$rc guard_rc=${GRC:-none} guard_ok=$GUARD_OK cleanup_ok=$CLEAN_OK"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
note "START $S $ST (timeout ${TO}s, token $TOKEN)"

# 1. preflight (any doubt refuses the launch)
F=$(pgrep -af "$FOREIGN" 2>&1); prc=$?
if [ "$prc" -eq 0 ]; then note "REFUSED 70: another simulator / training / execution / console process: $(echo "$F" | cut -c1-200)"; exit 70; fi
if [ "$prc" -ne 1 ]; then note "REFUSED 70: the process check itself failed (pgrep rc $prc: $F)"; exit 70; fi
U=$("${GPU_CMD[@]}" 2>&1); urc=$?; U=$(echo "$U" | head -1 | tr -d ' \r')
if [ "$urc" -ne 0 ] || ! [[ "$U" =~ ^[0-9]+$ ]]; then note "REFUSED 71: GPU memory query failed (rc $urc, output '$U')"; exit 71; fi
[ "$U" -ge 1500 ] && { note "REFUSED 71: GPU not idle ($U MiB used)"; exit 71; }
W=$("${WIN_CMD[@]}" 2>&1); wrc=$?
if [ "$wrc" -ne 0 ]; then note "REFUSED 72: Windows GPU process enumeration failed (rc $wrc): $(echo "$W" | head -2 | tr -d '\r')"; exit 72; fi
note "Windows GPU processes (full list, rc 0):"; echo "$W" | tr -d '\r' | sed 's/^/    /' >> "$LOG"
BAD=$(echo "$W" | tr -d '\r' | grep -i -E "$DENY")
[ -n "$BAD" ] && { note "REFUSED 73: an ML runtime holds the GPU on Windows: $BAD"; exit 73; }
note "preflight ok: GPU $U MiB, no foreign WSL job, no Windows ML runtime"

# 2. thermal guard, proven ready
rm -f "$MARK" "$MARK.tmp" "$READY" "$PGIDF" "$STOPF" "$CRCF"
"${GUARD_CMD[@]}" --log "$G/heat_${ST}_$STAMP.csv" --marker "$MARK" --ready "$READY" --pgid-file "$PGIDF" \
    --stop-file "$STOPF" --stop-pattern "[k]23_perception_test.py $ST --scene $S" --parent-pid $$ "${GARGS_EXTRA[@]}" >> "$G/guard.log" 2>&1 &
GPID=$!
for i in $(seq 1 90); do [ -f "$READY" ] && break; kill -0 $GPID 2>/dev/null || break; sleep 1; done
if [ ! -f "$READY" ] || ! kill -0 $GPID 2>/dev/null; then note "REFUSED 87: thermal guard not ready / not alive"; exit 87; fi

# 3. the step in its own verified session
defer_signals
LAUNCHED=1
setsid "${HOLD_CMD[@]}" hold --rc-file "$CRCF" --until "$END_IN" -- timeout --signal=TERM --kill-after=30 "$TO" "${JOB[@]}" >> "$RUNLOG" 2>&1 < /dev/null &
CPID=$!
LG=$(own_group "$CPID")
if [ -n "$LG" ] && [ "$LG" != "-" ]; then
  OWNED_G=$LG
  if ! own_publish "$PGIDF" "$LG" || [ ! -s "$PGIDF" ]; then note "ownership record NOT written ($PGIDF) -- stopping the step"; normal_signals; exit 99; fi
elif [ "$LG" = "-" ]; then note "the step ended before its session could be observed (nothing left to own)"
else note "the step's own session was not established -- killing pid $CPID"; kill_child KILL "$CPID"; wait "$CPID" 2>/dev/null; CPID=""; normal_signals; exit 99; fi
normal_signals
note "step launched: session ${OWNED_G:-ended}, guard $GPID"

# 4. supervise the guard while the step runs (a dead guard means an unguarded step, marker or not)
while kill -0 "$CPID" 2>/dev/null && [ ! -s "$CRCF" ]; do
  if ! kill -0 "$GPID" 2>/dev/null; then
    kill -0 "$CPID" 2>/dev/null || break                  # the job ended too (a completed pause stop): judged in step 5
    note "THERMAL GUARD ENDED while the step runs (marker: $([ -f "$MARK" ] && echo yes || echo no)) -- stopping the owned session"
    GUARD_OK=0
    stop_pg "$OWNED_G" 15 || CLEAN_OK=0
    break
  fi
  sleep 2
done
if [ -s "$CRCF" ]; then CRC=$(tr -dc '0-9' < "$CRCF"); CRC=${CRC:-99}; else wait "$CPID"; CRC=$?; fi
CPID=""

# 5. the guard's outcome (valid: 0 without a marker, 86 with one)
end_guard "$(bound 75 0)"
PAUSED=0; { [ -f "$MARK" ] || [ -f "$MARK.tmp" ]; } && PAUSED=1
if [ "$GRC" = "0" ] && [ "$PAUSED" = "0" ]; then :
elif [ "$GRC" = "86" ] && [ "$PAUSED" = "1" ]; then :
else GUARD_OK=0; fi

# 6. cleanup of the owned session (kept owned for the exit-trap retry when it fails)
if stop_pg "$OWNED_G" "$(bound 15 0)"; then OWNED_G=""; else CLEAN_OK=0; fi
if [ "$CLEAN_OK" = "0" ]; then note "cleanup of the owned session FAILED: step result not accepted"; CRC=89
elif [ "$GUARD_OK" = "0" ]; then note "guard outcome not valid (rc ${GRC:-none}, marker $PAUSED): step result not accepted"; CRC=88
elif [ "$PAUSED" = "1" ]; then note "THERMAL PAUSE: $(head -c 300 "$MARK" 2>/dev/null | tr '\n' ' ')"; CRC=86; fi

# 7. the attempt's supervisor outcome
OK=0; [ "$CRC" = "0" ] && OK=1
write_supervisor "$OK" "$CRC" || { note "supervisor outcome NOT written: step result not accepted"; CRC=90; }
exit "$CRC"
