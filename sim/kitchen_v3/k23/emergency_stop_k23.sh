#!/bin/bash
# K23 EMERGENCY STOP for a person (replaces the run plan's old global `pkill -f thor-Linux64`, Astra K23-PREP r3 5).
#   wsl -d Ubuntu-22.04 --cd "/mnt/d/Personal/Master's Project" -- bash sim/kitchen_v3/k23/emergency_stop_k23.sh FloorPlan15
# Stops ONLY what K23 batches of this scene started, proven per PROCESS and signalled LIFETIME-SAFE (Astra K23-PREP
# r4 4, r5 4; k23_procs.py): capture_one carries KCAP_OWNER="k23o:<scene>:<invocation>", everything it starts carries
# "k23:<scene>:<stamp>"; the sessions K23 creates (verified by their tagged leader) also contain tokenless members;
# every signal goes through a pidfd re-verified by start time. No pkill, no pid read from a record is ever signalled:
# archived records (thermal/{pgid,qcpgid,inner}_*) are checked (boot id, leader start time, token) and retired to
# thermal/retired/ when stale. Order: (1) TERM to this scene's capture_one scripts (they hand TERM to their batch
# script, whose cleanup writes the receipt), wait <= 60 s; (2) this scene's live guards asked to end (stop files),
# <= 10 s; (3) every process still owned by a token of this scene: TERM, 10 s, STOP, KILL; (4) stale records retired.
# A simulator of another job or of a person's session is never owned and is left running (and reported).
# Never `wsl --shutdown`. Exit 0 = nothing of this scene's K23 batches is left; 1 = something is (read the output).
set -u
SCENE="${1:?scene}"
case "$SCENE" in FloorPlan15|FloorPlan17) ;; *) echo "REFUSED: K23 batches exist only for FloorPlan15 / FloorPlan17"; exit 4;; esac
REPO="/mnt/d/Personal/Master's Project"
. "$REPO/sim/kitchen_v3/k23/k23_own.sh" || exit 4
PY=/home/arash/thor-env/bin/python
PROCS="$REPO/sim/kitchen_v3/k23/k23_procs.py"
D="$REPO/sim/kitchen_v3/k23/$SCENE"
[ "${K23_TEST_MODE:-}" = "1" ] && D="${K23_OUTDIR:?}"
SIMPAT="[t]hor-Linux64"; [ "${K23_TEST_MODE:-}" = "1" ] && SIMPAT="${K23_TEST_SIM_PATTERN:-$SIMPAT}"
OUTER="k23o:$SCENE:*"; INNER="k23:$SCENE:*"
lst() { $PY "$PROCS" list --token "$1" ${2:+--cmd "$2"} 2>/dev/null; }
echo "K23 EMERGENCY STOP $SCENE $(date '+%F %T')" | tee -a "$D/capture.log"
# 1. this scene's capture_one scripts (TERM: deferred, handled like their last resort; their cleanup runs)
$PY "$PROCS" signal --token "$OUTER" --cmd "capture_one_k23\.sh $SCENE " --sig TERM | tee -a "$D/capture.log"
for i in $(seq 1 60); do [ -z "$(lst "$OUTER" "capture_one_k23\.sh $SCENE ")" ] && break; sleep 1; done
# 2. this scene's live guards: ask first (a pause in progress completes)
for f in "$D"/thermal/pgid_*; do [ -f "$f" ] && touch "$D/thermal/stop_${f##*/pgid_}"; done
for i in $(seq 1 10); do [ -z "$(lst "$INNER" "thermal_guard\.py")" ] && break; sleep 1; done
# 3. everything still owned by a token of this scene
for t in "$OUTER" "$INNER"; do
  L=$(lst "$t"); [ -n "$L" ] && echo "  owned ($t) still running: $L -- TERM, 10 s, STOP, KILL" | tee -a "$D/capture.log"
  $PY "$PROCS" stop --token "$t" --grace 10 | tee -a "$D/capture.log"
done
# 4. archived ownership records: never permission to signal; stale ones are retired
for f in "$D"/thermal/pgid_* "$D"/thermal/qcpgid_* "$D"/thermal/inner_*; do
  [ -f "$f" ] || continue
  if ! $PY "$PROCS" record --file "$f" >/dev/null 2>&1; then own_retire "$f"; echo "  retired stale record $(basename "$f")"; fi
done
LEFT=0
for t in "$OUTER" "$INNER"; do L=$(lst "$t"); [ -n "$L" ] && { echo "  LEFT ($t): $L"; LEFT=1; }; done
O=$(pgrep -af "$SIMPAT" | cut -c1-120)
[ -n "$O" ] && echo "  NOTE: simulator(s) NOT in any K23 batch of $SCENE left running (not ours): $O"
echo "K23 EMERGENCY STOP done, left=$LEFT $(date '+%F %T')" | tee -a "$D/capture.log"
exit $LEFT
