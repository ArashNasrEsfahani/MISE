#!/bin/bash
# K23 one batch, inside WSL (Ubuntu-22.04). Called only by capture_k23.ps1. Never for FloorPlan1.
#   bash sim/kitchen_v3/k23/capture_one_k23.sh <scene> <probe|thermaltest|main> <max_new> <deadline_epoch> <invocation>
# Takes the ONE sim lock without waiting (busy -> exit 9 + a receipt saying so) and runs capture_inner_k23.sh inside it.
# The receipt sim/kitchen_v3/k23/<scene>/last_batch.json carries the invocation id; the supervisor deleted the old
# one before launching and rejects any receipt that is missing, malformed or from another invocation.
# ONE ABSOLUTE END (Astra K23-PREP r3 4): the inner script ends by END_IN = deadline - OUTER (60 s). This script never
# waits past it: flock runs in the background and is polled; an inner script still alive at END_IN + 10 s is sent TERM
# (its EXIT trap cleans up), and one still alive at deadline - 25 s is KILLed.
# OWNERSHIP (Astra K23-PREP r4 3, 4): everything of this invocation carries KCAP_OWNER="k23:<scene>:<stamp>" (k23_own.sh).
# CLEANUP ON EVERY EXIT (r4 3): a TERM / HUP / INT to THIS script is deferred and handled like the last resort (TERM the
# inner script, bounded wait, KILL), and on every path -- normal end, signal, an inner script that died early (even
# KILLed before the watchdog threshold), an unexpected error -- a final sweep stops every process that still carries
# this invocation's token (the capture's and the QC's sessions with all their process groups, e.g. a foreground
# `timeout`; the guard; the receipt writer). If anything had to be stopped there the receipt is untrusted (deleted),
# the work is left PENDING (pending_<stamp>.json) and the exit code is 99. It never returns while anything of this
# invocation still runs, never signals a pid by number alone, and never kills anything by name outside this invocation.
set -u
SCENE="${1:?scene}"; PIPE="${2:?pipeline}"; MAXNEW="${3:?max_new}"; DEADLINE="${4:?deadline_epoch}"; INV="${5:?invocation}"
case "$SCENE" in FloorPlan15|FloorPlan17) ;; *) echo "REFUSED: K23 captures only FloorPlan15 / FloorPlan17"; exit 4;; esac
case "$PIPE" in probe|thermaltest|main|topup|close|amb|parity) ;; *) echo "REFUSED: pipeline $PIPE not in this release"; exit 4;; esac
case "$DEADLINE" in ''|*[!0-9]*) echo "REFUSED: deadline_epoch must be an integer epoch"; exit 4;; esac
REPO="/mnt/d/Personal/Master's Project"
cd "$REPO" || exit 4
. "$REPO/sim/kitchen_v3/k23/k23_own.sh" || exit 4
PY=/home/arash/thor-env/bin/python
PROCS="$REPO/sim/kitchen_v3/k23/k23_procs.py"      # every signal: pidfd + start-time re-verification (r5 4)
OUTER=60
# integration-test overrides: honoured ONLY with K23_TEST_MODE=1; set without it = refused (never by accident)
if [ "${K23_TEST_MODE:-}" = "1" ]; then
  D="${K23_OUTDIR:?}"; LOCK="${K23_LOCK:?}"; OUTER="${K23_TEST_OUTER_S:-60}"
  INNER_SH="${K23_TEST_INNER_SCRIPT:-$REPO/sim/kitchen_v3/k23/capture_inner_k23.sh}"   # last-resort test: an overrunning stand-in
  PROCS="${K23_TEST_PROCS:-$PROCS}"                    # cleanup-failure test: a helper that reports survivors
else
  if env | grep -q '^K23_'; then echo "REFUSED: K23_* test overrides set outside K23_TEST_MODE: $(env | grep '^K23_' | cut -d= -f1 | tr '\n' ' ')"; exit 4; fi
  D="$REPO/sim/kitchen_v3/k23/$SCENE"; LOCK="/home/arash/sim.lock"; INNER_SH="$REPO/sim/kitchen_v3/k23/capture_inner_k23.sh"
fi
# this script carries its own token "k23o:<scene>:<invocation>" (re-exec, so /proc/<pid>/environ shows it): the
# emergency stop finds it by token, never by its command line alone
if [ -z "${KCAP_OWNER:-}" ]; then exec env KCAP_OWNER="k23o:$SCENE:$INV" bash "$0" "$@"; fi
[ "$KCAP_OWNER" = "k23o:$SCENE:$INV" ] || { echo "REFUSED: KCAP_OWNER already set ($KCAP_OWNER): a batch may not run inside another batch"; exit 4; }
mkdir -p "$D/qc" "$D/thermal"
LOG="$D/capture.log"
STAMP="$(date '+%Y%m%d-%H%M%S')-${INV:0:8}"      # unique per invocation: no file of another batch can be mistaken
TOKEN="k23:$SCENE:$STAMP"
END_IN=$(( DEADLINE - OUTER ))
rm -f "$D/last_batch.json" "$D/last_batch.json.tmp"
echo "K23 BATCH START $SCENE $PIPE max_new=$MAXNEW deadline=$(date -d @"$DEADLINE" '+%F %T') invocation=$INV $(date '+%F %T')" >> "$LOG"
now() { date +%s; }
INNER="capture_inner_k23.sh $SCENE $PIPE $MAXNEW $DEADLINE $INV $STAMP"
inner_alive() { pgrep -f "^bash .*$INNER\$" >/dev/null; }
SIG=""; DONE=0; LAST=0; KILLED=0; RC=0; CLEANUP_FAILED=0
# r4 3: a signal to THIS script is deferred (the loop below handles it within ~1 s); the EXIT trap is the backstop
trap 'SIG=143' TERM; trap 'SIG=129' HUP; trap 'SIG=130' INT
finish() {   # $1 = exit code wanted. The final sweep: nothing of this invocation may outlive this script (r4 3, r5 3).
  local rc="$1" out src
  [ "$DONE" = "1" ] && return 0
  DONE=1
  trap '' TERM HUP INT
  touch "$D/thermal/stop_$STAMP"                       # ask a guard still alive first (a pause in progress completes)
  for i in $(seq 1 5); do [ -z "$($PY "$PROCS" list --token "$TOKEN" 2>/dev/null)" ] && break; [ "$(now)" -ge $(( DEADLINE - 12 )) ] && break; sleep 1; done
  # owned = tagged + every member of the sessions K23 created (tokenless children too) + descendants; TERM -> STOP ->
  # KILL through pidfds, bounded by the absolute deadline
  out=$($PY "$PROCS" stop --token "$TOKEN" --grace 3 --deadline $(( DEADLINE - 5 )) 2>&1); src=$?
  if [ "$src" != "0" ] || ! echo "$out" | grep -q '"found": \[\]'; then
    echo "K23 SWEEP (rc $src): processes of this invocation were still running after the batch script ended: $out" >> "$LOG"
    LAST=1; KILLED=1                                   # something survived the batch script: its receipt is untrusted
    [ "$src" != "0" ] && CLEANUP_FAILED=1
  fi
  if [ "$LAST" = "1" ]; then
    [ "$KILLED" = "1" ] && rm -f "$D/last_batch.json" "$D/last_batch.json.tmp"
    if [ ! -f "$D/last_batch.json" ] || [ "${CLEANUP_FAILED:-0}" = "1" ]; then
      rm -f "$D/last_batch.json" "$D/last_batch.json.tmp"
      printf '{"invocation": "%s", "stamp": "%s", "why": "%s", "cleanup_failed": %s, "t": "%s"}\n' "$INV" "$STAMP" \
          "${WHY:-the batch did not end cleanly}; cleanup by capture_one_k23.sh; no trusted receipt" \
          "$([ "${CLEANUP_FAILED:-0}" = "1" ] && echo true || echo false)" "$(date '+%F %T')" > "$D/pending_$STAMP.json"
    fi
    [ "${CLEANUP_FAILED:-0}" = "1" ] && echo "K23 CLEANUP FAILED: owned processes are still alive (see the SWEEP line); work PENDING; a person must check before any batch" >> "$LOG"
    echo "K23 LAST RESORT done (${WHY:-}; inner killed: $KILLED; receipt $([ -f "$D/last_batch.json" ] && echo kept || echo 'none: work PENDING')), exit 99 $(date '+%F %T')" >> "$LOG"
    rc=99
  fi
  # the published ownership records of this invocation are retired once nothing of it runs (evidence kept)
  if [ -z "$($PY "$PROCS" list --token "$TOKEN" 2>/dev/null)" ]; then
    for f in "$D/thermal/pgid_$STAMP" "$D/thermal/qcpgid_$STAMP" "$D/thermal/inner_$STAMP"; do [ -f "$f" ] && own_retire "$f"; done
  fi
  exit "$rc"
}
trap 'WHY="${WHY:-capture_one exited unexpectedly}"; LAST=1; finish 99' EXIT
# -n: never wait behind another job; -E 9: busy = 9. The inner part is a FILE (the capture must not eat its script).
# flock runs in the BACKGROUND so this script keeps the clock (r3 4). The inner script runs in its OWN session (setsid).
KCAP_OWNER="$TOKEN" KCAP_LOCK="$LOCK" flock -n -E 9 "$LOCK" setsid bash "$INNER_SH" "$SCENE" "$PIPE" "$MAXNEW" "$DEADLINE" "$INV" "$STAMP" < /dev/null &
FPID=$!
# flock can end before its child (e.g. flock itself was signalled): wait for BOTH, but never past END_IN + 10 s; an
# early exit of the inner script (crash, outside KILL) ends this loop at once and the sweep in finish() cleans up
while { kill -0 "$FPID" 2>/dev/null || inner_alive; } && [ "$(now)" -lt $(( END_IN + 10 )) ] && [ -z "$SIG" ]; do sleep 1; done
if [ -n "$SIG" ]; then
  LAST=1; WHY="capture_one received a signal (exit $SIG)"
  echo "K23 SIGNAL to capture_one (exit $SIG): TERM to this invocation's batch script, bounded wait, KILL" >> "$LOG"
elif kill -0 "$FPID" 2>/dev/null || inner_alive; then
  LAST=1; WHY="the batch overran its absolute end"
  echo "K23 LAST RESORT: the batch is still running at its absolute end + 10 s -- TERM to this invocation's batch script" >> "$LOG"
fi
if [ "$LAST" = "1" ]; then
  trap '' TERM HUP INT
  $PY "$PROCS" signal --token "$TOKEN" --cmd "capture_inner_k23\.sh $SCENE $PIPE $MAXNEW $DEADLINE $INV $STAMP\$" --sig TERM >> "$LOG" 2>&1
  LIM=$(( DEADLINE - 25 )); [ -n "$SIG" ] && [ $(( $(now) + 60 )) -lt "$LIM" ] && LIM=$(( $(now) + 60 ))
  while inner_alive && [ "$(now)" -lt "$LIM" ]; do sleep 1; done
  if inner_alive; then
    echo "K23 LAST RESORT: the batch script did not end -- KILL (this invocation only)" >> "$LOG"
    KILLED=1; $PY "$PROCS" signal --token "$TOKEN" --cmd "capture_inner_k23\.sh $SCENE $PIPE $MAXNEW $DEADLINE $INV $STAMP\$" --sig KILL >> "$LOG" 2>&1; sleep 1
  fi
fi
# flock carries this invocation's token: it is signalled only through k23_procs (pidfd + start time), never by number
# (bash may already have reaped it, freeing the number -- Astra r6 4)
for i in 1 2 3 4 5; do kill -0 "$FPID" 2>/dev/null || break; sleep 1; done
kill -0 "$FPID" 2>/dev/null && $PY "$PROCS" signal --token "$TOKEN" --pid "$FPID" --sig KILL >> "$LOG" 2>&1
wait "$FPID" 2>/dev/null; RC=$?
if [ "$LAST" = "0" ] && [ "$RC" = "9" ]; then
  echo "K23 LOCK BUSY: another job holds $LOCK -- nothing started $(date '+%F %T')" >> "$LOG"
  SNAP="/tmp/k23_snapshot_$STAMP.json"
  T=$(( END_IN - $(now) - 10 )); [ "$T" -gt 60 ] && T=60
  if [ "$T" -ge 5 ]; then
    timeout --kill-after=5 "$T" $PY sim/kitchen_v3/k23/k23_decide.py snapshot --scene "$SCENE" --out "$SNAP"
    T=$(( END_IN - $(now) - 5 )); [ "$T" -gt 60 ] && T=60
    [ "$T" -ge 5 ] && timeout --kill-after=5 "$T" $PY sim/kitchen_v3/k23/k23_decide.py receipt --scene "$SCENE" --pipeline "$PIPE" --invocation "$INV" --stamp "$STAMP" \
        --capture-rc 9 --qc-rc -1 --guard-ok 0 --snapshot "$SNAP" --log "$LOG" --log-offset 0 >> "$LOG" 2>&1
  fi
  rm -f "$SNAP"
fi
finish "$RC"
