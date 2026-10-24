#!/bin/bash
# K23: the part of one batch that runs INSIDE the sim lock (called only by capture_one_k23.sh under flock).
#   capture_inner_k23.sh <scene> <pipeline> <max_new> <deadline_epoch> <invocation> <stamp>
# Order: snapshot (failure = nothing starts) -> refuse a foreign simulator -> deadline check -> pre-QC of committed
# episodes no QC report covers yet (main) -> thermal guard READY -> deadline re-check + READY + guard alive immediately
# before the launch -> capture in its OWN session / process group (verified and published while signals are DEFERRED,
# so a TERM can never land between the launch and the ownership record) -> the guard is ASKED to end (a pause in
# progress completes first) and its outcome collected -> the owned group is stopped (TERM -> KILL; never a name-wide
# kill) -> QC (its own verified group too) on whatever was committed -> ONE receipt attempt.
# Signals (TERM / INT / HUP) and any early exit run the EXIT trap, which itself ignores further TERM / INT / HUP: it asks
# the guard to end, stops the QC group and the capture group (whether or not the capture's pid is still set), waits for
# the guard within the time left (a pause in progress is never cut short by a fixed 5 s; a guard that has to be killed
# is RECORDED as guard_killed, never as ok), and makes the receipt attempt if none was made (rc 99).
# ONE ABSOLUTE END (Astra K23-PREP r3 4): every wait, timeout and the single receipt attempt is bounded by
# END_IN = deadline - OUTER (60 s); capture_one_k23.sh keeps those last 60 s for its own last-resort cleanup. A receipt
# that cannot be written in time leaves thermal/../pending_<stamp>.json (durably PENDING; k23_decide `pre` refuses).
# Budget (k23_decide QC_BUDGET_S 470 + GRACE_S 300): capture KILL-after 30 s, guard end <= 75 s, group stop <= 16 s,
# thermal post-stop <= 110 s, QC <= 370 + 30 s, receipt <= 60 + 5 s, outer reserve 60 s.
SCENE="$1"; PIPE="$2"; MAXNEW="$3"; DEADLINE="$4"; INV="$5"; STAMP="$6"
QCALL=""; case "$PIPE" in topup|close|parity) QCALL="--all-protocols";; esac   # K2-TOPUP: QC also covers top-up / close episodes
REPO="/mnt/d/Personal/Master's Project"; cd "$REPO" || exit 4
# r4 3 / 4: process ownership = the KCAP_OWNER token capture_one_k23.sh gave this invocation (inherited by everything
# started below); without it nothing may start, because nothing could be cleaned up safely
. "$REPO/sim/kitchen_v3/k23/k23_own.sh" || exit 4
TOKEN="${KCAP_OWNER:-}"
[ "$TOKEN" = "k23:$SCENE:$STAMP" ] || { echo "REFUSED: no / wrong KCAP_OWNER token ($TOKEN) for k23:$SCENE:$STAMP"; exit 4; }
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
PY=/home/arash/thor-env/bin/python
PROCS="$REPO/sim/kitchen_v3/k23/k23_procs.py"
QC_BUDGET=470; GRACE=300; OUTER=60
# integration-test overrides are honoured ONLY with K23_TEST_MODE=1 (capture_one_k23.sh refuses them otherwise)
if [ "${K23_TEST_MODE:-}" = "1" ]; then
  D="${K23_OUTDIR:?}"; CAPTURE="${K23_CAPTURE_CMD:-sim/kitchen_v3/k23/k23_capture.py}"; SIMPAT="${K23_TEST_SIM_PATTERN:-[t]hor-Linux64}"
  CAPMAX="${K23_TEST_CAP_S:-1800}"; QCCMD="${K23_QC_CMD:-sim/kitchen_v3/k23/k23_qc.py}"
  STORE_ROOT="${K23_TEST_STORE_ROOT:-/home/arash/thor-data/kitchen_v3}"; QC_TMO="${K23_TEST_QC_S:-370}"
  RECEIPT_CMD="${K23_TEST_RECEIPT_CMD:-sim/kitchen_v3/k23/k23_decide.py}"
  QC_BUDGET="${K23_TEST_QC_BUDGET:-470}"; GRACE="${K23_TEST_GRACE:-300}"; OUTER="${K23_TEST_OUTER_S:-60}"
else
  D="$REPO/sim/kitchen_v3/k23/$SCENE"; CAPTURE="sim/kitchen_v3/k23/k23_capture.py"; SIMPAT="[t]hor-Linux64"; CAPMAX=1800
  QCCMD="sim/kitchen_v3/k23/k23_qc.py"; STORE_ROOT="/home/arash/thor-data/kitchen_v3"; QC_TMO=370
  RECEIPT_CMD="sim/kitchen_v3/k23/k23_decide.py"
fi
# K2-AMB (Astra K2-AMB-r1): amb writes its own store (<root>_amb) and keeps its own QC evidence ($D/qc_amb)
QCD="$D/qc"; [ "$PIPE" = "amb" ] && { STORE_ROOT="${STORE_ROOT}_amb"; QCD="$D/qc_amb"; QCALL="--amb"; }
mkdir -p "$QCD"
LOG="$D/capture.log"
MARK="$D/thermal/pause_$STAMP.json"; READY="$D/thermal/ready_$STAMP"; PGIDF="$D/thermal/pgid_$STAMP"
QPGIDF="$D/thermal/qcpgid_$STAMP"; STOPF="$D/thermal/stop_$STAMP"; PENDF="$D/pending_$STAMP.json"
CRCF="$D/thermal/crc_$STAMP"; QRCF="$D/thermal/qrc_$STAMP"   # exit codes written by the session holders (r5 3)
SNAP="/tmp/k23_snapshot_$STAMP.json"
END_IN=$(( DEADLINE - OUTER ))
OFFSET=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
DONE_RECEIPT=0; RECEIPT_TRIED=0; GPID=""; CPID=""; QPID=""; GRC=""; OWNED_G=""; QC_G=""; GUARD_KILLED=0; GUARD_HUNG=0
PENDING_SIG=""
echo "  got sim lock $(date '+%F %T') invocation $INV (absolute end $(date -d @"$END_IN" '+%F %T'))" >> "$LOG"
# this script runs in its own session (capture_one: flock ... setsid bash ...); published for the last-resort cleanup
[ "$(ps -o sid= -p $$ | tr -d ' ')" = "$$" ] && own_publish "$D/thermal/inner_$STAMP" "$$"

left() { echo $(( END_IN - $(date +%s) )); }
bound() {   # $1 wanted seconds, $2 seconds kept for later steps -> min(wanted, left - kept), never below 0
  local l=$(( $(left) - $2 )); [ "$l" -lt 0 ] && l=0
  if [ "$l" -lt "$1" ]; then echo "$l"; else echo "$1"; fi
}
pending() {   # durable: no receipt could be written; k23_decide `pre` refuses every later launch while it exists
  printf '{"invocation": "%s", "stamp": "%s", "why": "%s", "t": "%s"}\n' "$INV" "$STAMP" "$1" "$(date '+%F %T')" > "$PENDF"
}
receipt() {   # $1 capture rc, $2 qc rc, $3 guard ok (1/0), $4 post-stop json, $5 phase. ONE attempt per batch.
  [ "$RECEIPT_TRIED" = "1" ] && return 1
  RECEIPT_TRIED=1
  local t; t=$(bound 60 5)
  if [ "$t" -lt 5 ]; then
    echo "K23 RECEIPT NOT ATTEMPTED: ${t}s left before the absolute end -- work left PENDING ($PENDF)" >> "$LOG"
    pending "no time left for a receipt (phase $5)"; return 1
  fi
  timeout --signal=TERM --kill-after=5 "$t" $PY "$RECEIPT_CMD" receipt --scene "$SCENE" \
      --pipeline "$PIPE" --invocation "$INV" --stamp "$STAMP" --capture-rc "$1" --qc-rc "$2" --guard-ok "$3" \
      --snapshot "$SNAP" --log "$LOG" --log-offset "$OFFSET" --marker "$MARK" --post-stop "$4" \
      --guard-rc "${GRC:-}" --guard-killed "$GUARD_KILLED" --phase "$5" >> "$LOG" 2>&1 && DONE_RECEIPT=1
  if [ "$DONE_RECEIPT" != "1" ]; then
    echo "K23 RECEIPT FAILED within its ${t}s bound -- no second attempt; work left PENDING ($PENDF)" >> "$LOG"
    pending "receipt failed within ${t}s (phase $5)"; return 1
  fi
}
members() {   # r4 3 / r5 3-4: EVERY live process of the K23-created session $1 (verified by its tagged leader; all its
  # process groups; tokenless members and descendants too) -- k23_procs.py, never a pid by number alone
  $PY "$PROCS" list --token "$TOKEN" --session "$1" 2>/dev/null
}
kill_child() {   # $1 signal, $2 pid of this script's own child: only when that pid is still OUR (tagged) process,
  # through a pidfd re-verified by start time (bash may already have reaped it, freeing the number: r5 4)
  [ -n "$2" ] && $PY "$PROCS" signal --token "$TOKEN" --pid "$2" --sig "$1" >/dev/null 2>&1
  return 0
}
stop_pg() {   # $1 a VERIFIED owned session, $2 seconds of TERM grace (bounded by the absolute end); TERM -> STOP -> KILL
  local g="$1" w out; [ -z "$g" ] && return 0
  [ -z "$(members "$g")" ] && return 0
  w=$(bound "$2" 0)
  out=$($PY "$PROCS" stop --token "$TOKEN" --session "$g" --grace "$w" --deadline "$END_IN" 2>&1) && return 0
  echo "K23 session $g still has members after KILL: $out" >> "$LOG"; return 1
}
own_group() {  # $1 pid of a `setsid` child: wait (<= 5 s) until it leads its own session + group; echo it, or "-" when
  # it already ENDED (bash reaps fast children) and nothing of its session is left (nothing to own); else nothing
  local p="$1" i pg sd
  for i in $(seq 1 50); do
    read -r pg sd < <(ps -o pgid=,sid= -p "$p" 2>/dev/null)
    [ "$pg" = "$p" ] && [ "$sd" = "$p" ] && { echo "$p"; return 0; }
    if ! kill -0 "$p" 2>/dev/null; then
      if [ -n "$(members "$p")" ]; then echo "$p"; else echo "-"; fi; return 0
    fi
    sleep 0.1
  done
  return 1
}
defer_signals() { trap 'PENDING_SIG=143' TERM; trap 'PENDING_SIG=130' INT; trap 'PENDING_SIG=129' HUP; }
normal_signals() {
  trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
  if [ -n "$PENDING_SIG" ]; then echo "K23 signal (exit $PENDING_SIG) arrived during an ownership step -- handled now" >> "$LOG"; exit "$PENDING_SIG"; fi
}
launch_owned() {  # $1 = pidfile to publish; rest = command. Sets LPID / LG. Signals are deferred until ownership is known.
  local f="$1"; shift
  defer_signals
  setsid "$@" &
  LPID=$!
  [ "${K23_TEST_MODE:-}" = "1" ] && [ -n "${K23_TEST_PRE_PGID_SLEEP:-}" ] && [ "$f" = "$PGIDF" ] && sleep "$K23_TEST_PRE_PGID_SLEEP"
  LG=$(own_group "$LPID")
  if [ -n "$LG" ] && [ "$LG" != "-" ]; then own_publish "$f" "$LG"; fi
  [ "$LG" = "-" ] && { echo "  pid $LPID ended before its session could be observed; no member left (nothing to own)" >> "$LOG"; LG="ENDED"; }
}
run_qc() {    # bounded QC in its OWN verified group, in the background (a signal interrupts the wait; the trap cleans it)
  local t r; t=$(bound "$QC_TMO" 100)
  if [ "$t" -lt "$QC_TMO" ] && [ "$t" -lt 10 ]; then echo "  QC not run: ${t}s left before the absolute end" >> "$LOG"; return 255; fi
  rm -f "$QRCF"
  # the QC session is led by a HOLDER (k23_procs.py hold): a child subreaper that stays alive while any descendant
  # lives, so the session stays verified until stop_pg has stopped every member, tokenless ones too (r5 3)
  launch_owned "$QPGIDF" $PY "$PROCS" hold --rc-file "$QRCF" --until "$END_IN" -- \
      timeout --signal=TERM --kill-after=30 "$t" nice -n 10 $PY "$QCCMD" batch --scene "$SCENE" \
      --root "$STORE_ROOT" --out-dir "$QCD" $QCALL >> "$QCD/qc_stdout.log" 2>&1
  QPID=$LPID; QC_G=$LG; [ "$QC_G" = "ENDED" ] && QC_G=""
  if [ -z "$QC_G" ] && [ "$LG" != "ENDED" ]; then
    echo "K23 QC did not get its own process group -- killed (pid $QPID only)" >> "$LOG"
    kill_child KILL "$QPID"; wait "$QPID" 2>/dev/null; QPID=""; normal_signals; return 254
  fi
  normal_signals
  while kill -0 "$QPID" 2>/dev/null && [ ! -s "$QRCF" ]; do sleep 0.5; done
  if [ -s "$QRCF" ]; then r=$(tr -dc '0-9' < "$QRCF"); r=${r:-255}; stop_pg "$QC_G" 5; wait "$QPID" 2>/dev/null
  else wait "$QPID"; r=$?; stop_pg "$QC_G" 5; fi
  QPID=""; QC_G=""
  return $r
}
end_guard() {    # $1 = seconds to wait (bounded by the caller). Ask the guard to end; a pause in progress completes first.
  [ -z "$GPID" ] && return 0
  touch "$STOPF"
  local i
  for i in $(seq 1 "$1"); do kill -0 "$GPID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$GPID" 2>/dev/null; then
    echo "K23 THERMAL GUARD did not end within $1 s of the stop request -- KILLED (recorded: guard_killed; a marker it" \
         "left, even half-written, still counts as a pause)" >> "$LOG"
    kill_child KILL "$GPID"; GUARD_HUNG=1; GUARD_KILLED=1
  fi
  wait "$GPID" 2>/dev/null; GRC=$?; GPID=""
}
on_exit() {   # every exit path: stop what this batch started, ONE best-effort receipt (the supervisor stops on rc 99)
  local rc=$?
  trap '' TERM INT HUP          # cleanup is bounded by END_IN; it is not cut short by a second signal
  trap - EXIT
  [ -n "$GPID" ] && touch "$STOPF"                 # ask the guard first, so it can finish a pause while we clean up
  if [ -n "$QC_G" ]; then stop_pg "$QC_G" "$(bound 30 80)"; elif [ -n "$QPID" ]; then kill_child KILL "$QPID"; fi
  [ -n "$QPID" ] && wait "$QPID" 2>/dev/null; QPID=""; QC_G=""
  if [ -n "$OWNED_G" ]; then stop_pg "$OWNED_G" "$(bound 15 75)"
  elif [ -n "$CPID" ]; then kill_child KILL "$CPID"; fi
  end_guard "$(bound 75 70)"
  if [ "$DONE_RECEIPT" != "1" ] && [ "$RECEIPT_TRIED" != "1" ] && [ -f "$SNAP" ]; then
    echo "K23 BATCH DIED (exit $rc) -- best-effort receipt rc 99" >> "$LOG"; receipt 99 -1 0 "" exit
  fi
  rm -f "$SNAP"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP

# 0. snapshot (its failure prevents the launch; no receipt -> the supervisor's 'missing receipt = STOP')
if ! timeout --kill-after=5 "$(bound 60 5)" $PY sim/kitchen_v3/k23/k23_decide.py snapshot --scene "$SCENE" --pipeline "$PIPE" --out "$SNAP"; then
  rm -f "$SNAP"; echo "K23 SNAPSHOT FAILED -- nothing started" >> "$LOG"; exit 22
fi
# 0b. Kitchen 1 first (A28 15.9 condition 2): another process waiting for the sim lock means a job needs the simulator
#     -> start nothing, release the lock, yield (rc 24, not a stop for a person). Our own flock is this script's parent.
[ "${K23_TEST_MODE:-}" = "1" ] && [ -n "${K23_TEST_PREYIELD_SLEEP:-}" ] && sleep "$K23_TEST_PREYIELD_SLEEP"
WAITERS=$(pgrep -af "flock.*${KCAP_LOCK:-/home/arash/sim.lock}" 2>/dev/null | awk -v me="$PPID" '$1 != me' | grep -v -e pgrep -e 'flock -n' | cut -c1-120)
if [ -n "$WAITERS" ]; then
  echo "K23 YIELD: another process waits for the sim lock -- nothing started: $WAITERS" >> "$LOG"
  receipt 24 -1 0 "" normal; exit 24
fi
# 1. no foreign simulator (never kill one we did not start)
if pgrep -f "$SIMPAT" >/dev/null; then
  echo "K23 FOREIGN SIMULATOR running -- nothing started: $(pgrep -af "$SIMPAT" | cut -c1-120)" >> "$LOG"
  receipt 21 -1 0 "" normal; exit 21
fi
# 2. absolute deadline: the capture must be able to end, be checked and cleaned up before it
case "$PIPE" in main|topup|close|amb|parity) CAP=$CAPMAX; MIN=$CAPMAX;; probe) CAP=900; MIN=600;; *) CAP=420; MIN=300;; esac
[ "$CAP" -lt "$MIN" ] && MIN=$CAP
LEFT=$(( DEADLINE - $(date +%s) - QC_BUDGET - GRACE ))
if [ "$LEFT" -lt "$MIN" ]; then echo "K23 DEADLINE: ${LEFT}s usable < ${MIN}s -- batch not started" >> "$LOG"; receipt 20 -1 1 "" normal; exit 20; fi
# 2b. committed episodes that no QC report covers yet (e.g. after a batch died) are checked BEFORE new capture
if { [ "$PIPE" = "main" ] || [ "$PIPE" = "topup" ] || [ "$PIPE" = "close" ] || [ "$PIPE" = "amb" ] || [ "$PIPE" = "parity" ]; } && [ -f "$STORE_ROOT/$SCENE/capture_state.json" ]; then
  run_qc; PQ=$?
  echo "  pre-QC of earlier committed episodes: rc $PQ" >> "$LOG"
  if [ "$PQ" != "0" ] && [ "$PQ" != "4" ]; then
    echo "K23 PRE-QC did not pass (rc $PQ) -- nothing started" >> "$LOG"; receipt 23 "$PQ" 1 "" normal; exit 23
  fi
fi
# 3. thermal guard, proven alive (one complete telemetry sample) before the capture starts
GARGS=(--log "$D/thermal/session_$STAMP.csv" --marker "$MARK" --ready "$READY" --pgid-file "$PGIDF" --stop-file "$STOPF" --stop-pattern "[k]23_capture.py --scene $SCENE" --parent-pid $$)
[ "$PIPE" = "thermaltest" ] && GARGS+=(--interval 15 --test-trigger)
[ "${K23_TEST_MODE:-}" = "1" ] && [ -n "${K23_TEST_GUARD_INTERVAL:-}" ] && GARGS+=(--interval "$K23_TEST_GUARD_INTERVAL")
$PY sim/kitchen_v3/k23/thermal_guard.py "${GARGS[@]}" >> "$LOG" 2>&1 &
GPID=$!
for i in $(seq 1 60); do [ -f "$READY" ] && break; kill -0 $GPID 2>/dev/null || break; sleep 1; done
if [ ! -f "$READY" ] || ! kill -0 $GPID 2>/dev/null; then
  echo "K23 THERMAL GUARD not ready / not alive (no complete telemetry sample in 60 s) -- nothing started" >> "$LOG"
  end_guard "$(bound 5 70)"; receipt 87 -1 0 "" normal; exit 87
fi
[ "${K23_TEST_MODE:-}" = "1" ] && [ -n "${K23_TEST_PRELAUNCH_SLEEP:-}" ] && sleep "$K23_TEST_PRELAUNCH_SLEEP"
# 4. re-check the deadline AFTER the guard start; READY + guard liveness immediately before the launch
LEFT=$(( DEADLINE - $(date +%s) - QC_BUDGET - GRACE ))
if [ "$LEFT" -lt "$MIN" ]; then echo "K23 DEADLINE (after guard start): ${LEFT}s < ${MIN}s -- batch not started" >> "$LOG"; end_guard "$(bound 75 70)"; receipt 20 -1 1 "" normal; exit 20; fi
if [ ! -f "$READY" ] || ! kill -0 $GPID 2>/dev/null || [ -f "$MARK" ] || [ -f "$MARK.tmp" ]; then
  echo "K23 THERMAL GUARD gone (or paused) just before the launch -- nothing started" >> "$LOG"
  end_guard "$(bound 75 70)"; receipt 87 -1 0 "" normal; exit 87
fi
TMO=$(( LEFT < CAP ? LEFT : CAP ))
echo "  capture: timeout ${TMO}s, guard pid $GPID" >> "$LOG"
# launch + ownership with signals DEFERRED (r3 3): a TERM here is handled right after the group is published
rm -f "$CRCF"
launch_owned "$PGIDF" $PY "$PROCS" hold --rc-file "$CRCF" --until "$END_IN" -- \
    timeout --signal=TERM --kill-after=30 "$TMO" $PY -u "$CAPTURE" --scene "$SCENE" \
    --pipeline "$PIPE" --max-new "$MAXNEW" >> "$LOG" 2>&1 < /dev/null
CPID=$LPID; OWNED_G=$LG; [ "$OWNED_G" = "ENDED" ] && OWNED_G=""
if [ -z "$OWNED_G" ] && [ "$LG" != "ENDED" ]; then       # never signal a group we do not own (it could be this script's own)
  echo "K23 the capture's own session / group was not established for pid $CPID -- killing the pid" >> "$LOG"
  kill_child KILL "$CPID"; wait "$CPID" 2>/dev/null; CPID=""; normal_signals; exit 99
fi
echo "  capture owned group $OWNED_G published" >> "$LOG"
normal_signals
GUARD_OK=1
while kill -0 $CPID 2>/dev/null && [ ! -s "$CRCF" ]; do
  if ! kill -0 $GPID 2>/dev/null && [ ! -f "$MARK" ] && [ ! -f "$MARK.tmp" ]; then
    echo "K23 THERMAL GUARD DIED during the capture -- stopping the capture" >> "$LOG"
    GUARD_OK=0; stop_pg "$OWNED_G" 15; break
  fi
  sleep 5
done
# the capture's exit code: from its holder's rc file (the holder may still be alive, keeping its session verified,
# while tokenless descendants remain); else (the holder itself was stopped, e.g. by a pause) from the holder's exit
if [ -s "$CRCF" ]; then CRC=$(tr -dc '0-9' < "$CRCF"); CRC=${CRC:-99}; else wait $CPID; CRC=$?; CPID=""; fi
# 5. the guard's outcome is collected BEFORE it ends: it is asked to stop and finishes a pause in progress first
end_guard "$(bound 75 70)"
PAUSED=0; { [ -f "$MARK" ] || [ -f "$MARK.tmp" ]; } && PAUSED=1
case "$GRC" in 0|86) ;; *) [ "$PAUSED" = "0" ] && GUARD_OK=0;; esac
[ "$GUARD_HUNG" = "1" ] && GUARD_OK=0
[ "$GRC" = "86" ] && [ "$PAUSED" = "0" ] && GUARD_OK=0
[ "$GUARD_OK" = "0" ] && [ "$PAUSED" = "0" ] && CRC=88
stop_pg "$OWNED_G" "$(bound 15 70)"
[ -n "$CPID" ] && { wait "$CPID" 2>/dev/null; CPID=""; }
POST=""
if [ "$PAUSED" = "1" ]; then
  CRC=86
  if [ "$(left)" -gt 180 ]; then
    sleep 20
    NJ=$(pgrep -fc "[k]23_capture.py --scene" || true); NT=0; [ -n "$OWNED_G" ] && NT=$(pgrep -s "$OWNED_G" -fc "$SIMPAT" || true)
    U=""
    for i in 1 2 3; do
      v=$(timeout --kill-after=5 15 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
      U="$U${U:+, }${v:-null}"; sleep 10
    done
  else
    echo "  thermal post-stop samples skipped: $(left)s left before the absolute end" >> "$LOG"
    NJ=$(pgrep -fc "[k]23_capture.py --scene" || true); NT=0; [ -n "$OWNED_G" ] && NT=$(pgrep -s "$OWNED_G" -fc "$SIMPAT" || true)
    U="null, null, null"
  fi
  POST="{\"jobs_left\": ${NJ:-0}, \"simulators_left\": ${NT:-0}, \"util_after_stop\": [$U]}"
fi
# 6. QC on everything committed (also after an abort or a timeout: committed data is never left unchecked)
QRC=-1
if [ "$PIPE" = "main" ] || [ "$PIPE" = "topup" ] || [ "$PIPE" = "close" ] || [ "$PIPE" = "amb" ] || [ "$PIPE" = "parity" ]; then run_qc; QRC=$?; [ "$QRC" -ge 254 ] && QRC=-1; fi
receipt "$CRC" "$QRC" "$GUARD_OK" "$POST" normal
echo "K23 BATCH END $SCENE capture_rc=$CRC qc_rc=$QRC guard_ok=$GUARD_OK guard_rc=$GRC guard_killed=$GUARD_KILLED $(date '+%F %T')" >> "$LOG"
exit $CRC
