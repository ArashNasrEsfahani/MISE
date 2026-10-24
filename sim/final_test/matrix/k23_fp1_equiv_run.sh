#!/bin/bash
# K23 FP1 equivalence, WSL side: ONE matrix job of ONE paired run, executed in the tree given EXPLICITLY (no Master* glob, no cd string),
# under the one simulator lock (flock /home/arash/sim.lock) + timeout + hang_guard, exactly as the FREEZE-GATES-FT2 runner did.
#   usage: bash k23_fp1_equiv_run.sh <TREE> <EVIDENCE> <RUN> <JOB> <script.py> [args...]
#   e.g.   bash /mnt/d/Personal/MP-agent-side/sim/final_test/matrix/k23_fp1_equiv_run.sh /mnt/d/Personal/MP-base \
#              /mnt/d/Personal/K23-FP1-EQUIV-RUN B1 capture fg_runner.py capture --namespace freeze-gates-ft2 --items dev --runs 1 \
#              --worlds canon --ollama http://WINHOST:11434
# The literal token WINHOST in any argument is replaced by the Windows host IP (ip route show default). One try per launch: the matrix
# scripts resume from their journals, so re-launching the same command resumes (a hang_guard exit 75 is an infra failure, never data).
set -u
if [ "${1:-}" = "--inner" ]; then                    # the part INSIDE the lock
    shift
    TREE=$1; OUT=$2; SCRIPT=$3; shift 3
    cd "$TREE" || exit 2
    echo "$(date '+%F %T') lock acquired" >> "$OUT/progress.txt"
    timeout 14400 /home/arash/thor-env/bin/python sim/final_test/matrix/hang_guard.py "sim/final_test/matrix/$SCRIPT" "$@" \
        >> "$OUT/stdout.txt" 2>> "$OUT/stderr.txt"
    rc=$?
    echo $rc > "$OUT/last_rc"
    echo "$(date '+%F %T') exit $rc" >> "$OUT/progress.txt"
    pkill -9 -f '[t]hor-Linux64' 2>/dev/null         # still inside the lock: only this job's simulator can be running
    exit 0
fi
TREE=$1; EVID=$2; RUN=$3; JOB=$4; SCRIPT=$5; shift 5
case "$TREE" in /mnt/d/Personal/MP-base|/mnt/d/Personal/MP-agent-side) ;; *) echo "refused: tree $TREE is not a paired worktree"; exit 2;; esac
case "$RUN" in B1|B2|BC) [ "$TREE" = /mnt/d/Personal/MP-base ] || { echo "refused: $RUN runs in MP-base"; exit 2; };;
               S1|S2|SC) [ "$TREE" = /mnt/d/Personal/MP-agent-side ] || { echo "refused: $RUN runs in MP-agent-side"; exit 2; };;
               *) echo "refused: run $RUN"; exit 2;; esac
HOST=$(ip route show default | awk '{print $3}')
ARGS=()
for x in "$@"; do ARGS+=("${x//WINHOST/$HOST}"); done
OUT="$EVID/$RUN/logs/$JOB"
mkdir -p "$OUT"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 TRACE_ENABLED=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HANG_SECS=${HANG_SECS:-60}
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt" 2>/dev/null
echo "$(date '+%F %T') queued $RUN/$JOB in $TREE: $SCRIPT ${ARGS[*]}" >> "$OUT/progress.txt"
flock /home/arash/sim.lock bash "$0" --inner "$TREE" "$OUT" "$SCRIPT" "${ARGS[@]}"
rc=$(cat "$OUT/last_rc" 2>/dev/null || echo missing); rm -f "$OUT/last_rc"
echo "$(date '+%F %T') RUN END rc=$rc" >> "$OUT/progress.txt"
echo "$RUN/$JOB rc=$rc (logs: $OUT)"
[ "$rc" = "0" ] && exit 0 || exit 1
