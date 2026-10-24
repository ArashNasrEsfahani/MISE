#!/bin/bash
# Integration smoke of run_perception_step.sh with stub jobs (WSL; K23_TEST_MODE=1; no simulator, no evaluation, no Kitchen 2/3 data;
# the supervisor outcome goes to a stub, never to a real ledger). Astra K23-PERCEPTION-TEST r2 7 + r3 1-4: argument routing, no
# self-match, the PRODUCTION foreign-process patterns (a simulator stub, a K23 training entry point), failed / malformed / busy GPU
# queries, a failed Windows enumeration, a Windows ML runtime, an induced thermal pause stopping the owned session (real guard), the
# guard dying mid-step, the guard dying after writing its marker but before stopping the step, the guard failing right after the step
# completed, a failed ownership publication, a failed cleanup, a wrong token, exit-code pass-through, and the supervisor outcome
# (ok only for a clean, guarded, successful attempt). The real nvidia-smi feeds the real guard.
REPO="/mnt/d/Personal/Master's Project"; cd "$REPO" || exit 9
STEP="$REPO/sim/kitchen_v3/k23/run_perception_step.sh"
OUT=$(mktemp -d /tmp/k23p_selftest.XXXX)
export K23_TEST_MODE=1 K23P_OUTDIR="$OUT" K23P_SIMPAT='[f]ake-thor-sim' K23_TEST_SIM_PATTERN='[f]ake-thor-sim' K23P_SUP_CMD="bash $OUT/sup.sh"
cat > "$OUT/k23_perception_test.py" <<'EOF'
#!/bin/bash
# stub job (a bash file named like the real step so the guard's stop-pattern matches): <step> --scene <scene>
echo "$@" > "$K23P_OUTDIR/args_$1"
[ -n "${STUB_SIM:-}" ] && { (exec -a fake-thor-sim-of-step sleep 600) & }
sleep "${STUB_SLEEP:-2}"
exit "${STUB_RC:-0}"
EOF
cat > "$OUT/sup.sh" <<'EOF'
#!/bin/bash
# stub supervisor: records "<stamp> ok=<0|1>"
while [ $# -gt 0 ]; do case $1 in --stamp) s=$2; shift;; --ok) o=$2; shift;; esac; shift; done
echo "$s ok=$o" >> "$K23P_OUTDIR/sup.log"
EOF
cat > "$OUT/stub_guard.sh" <<'EOF'
#!/bin/bash
# stub guard: STUB_GUARD_MODE=marker_die (ready, marker after 3 s, exit 1 without stopping anything) | die_after_job (ready, exit 1 at 4 s)
while [ $# -gt 0 ]; do case $1 in --ready) R=$2; shift;; --marker) M=$2; shift;; esac; shift; done
echo ready > "$R"
if [ "$STUB_GUARD_MODE" = marker_die ]; then sleep 3; echo '{"reason": "stub pause intent"}' > "$M"; exit 1; fi
sleep 4; exit 1
EOF
cat > "$OUT/procs_nostop.sh" <<EOF
#!/bin/bash
# k23_procs with an injected stop failure (everything else real)
[ "\$1" = stop ] && { echo '{"error": "injected stop failure"}'; exit 1; }
exec /home/arash/thor-env/bin/python "$REPO/sim/kitchen_v3/k23/k23_procs.py" "\$@"
EOF
DEF_GPU="echo 120"; DEF_WIN="echo 6084, C:\\Program_Files\\VS_Code\\Code.exe, [N/A]"
ok=0; bad=0
chk() { if [ "$2" = "1" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $1"; fi; }
run() {   # $1 step, $2 stamp, then env assignments
  local st=$1 sp=$2; shift 2
  env "$@" KCAP_OWNER="k23p:FloorPlan15:$st:$sp" timeout 300 setsid --wait bash "$STEP" FloorPlan15 "$st" 120 "$sp"
}
sup() { grep "^$1 " "$OUT/sup.log" 2>/dev/null | tail -1 | cut -d' ' -f2; }
B="K23P_JOB_CMD=bash $OUT/k23_perception_test.py"
C=("K23P_GPU_CMD=$DEF_GPU" "K23P_WIN_CMD=$DEF_WIN" "K23P_GUARD_ARGS=--interval 1")
left() { pgrep -f "[f]ake-thor-sim|$OUT/[k]23_perception_test.py" | wc -l; }

run poses T1 "$B" "${C[@]}"; rc=$?
chk "normal step rc 0 (no self-match of the preflight)" "$([ $rc = 0 ] && echo 1)"
chk "argument routing: the job got 'poses --scene FloorPlan15'" "$([ "$(cat "$OUT/args_poses" 2>/dev/null)" = "poses --scene FloorPlan15" ] && echo 1)"
chk "the guard ended and nothing of the step is left" "$([ "$(left)" = 0 ] && ! pgrep -f "[t]hermal_guard.py --log $OUT" >/dev/null && echo 1)"
chk "the full Windows process list is logged" "$(grep -q "Code.exe" "$OUT/guard/step_poses_T1.log" && echo 1)"
chk "supervisor outcome ok=1 for the clean attempt" "$([ "$(sup T1)" = ok=1 ] && echo 1)"

(exec -a fake-thor-sim-foreign sleep 60) & FP=$!; sleep 0.5
run primary T2 "$B" "${C[@]}"; rc=$?
kill $FP 2>/dev/null; wait $FP 2>/dev/null
chk "a foreign simulator refuses the launch (70), nothing launched, no supervisor outcome" "$([ $rc = 70 ] && [ ! -f "$OUT/args_primary" ] && [ -z "$(sup T2)" ] && echo 1)"
(exec -a "k23_amb_train.py train --scene FloorPlan15 --model amb26m" sleep 60) & FP=$!; sleep 0.5
run primary T2b "$B" "${C[@]}"; rc=$?
kill $FP 2>/dev/null; wait $FP 2>/dev/null
chk "a K23 training entry point (production pattern) refuses (70)" "$([ $rc = 70 ] && echo 1)"

run primary T3 "$B" "K23P_GPU_CMD=false" "K23P_WIN_CMD=$DEF_WIN"; rc=$?
chk "a failed GPU memory query refuses (71)" "$([ $rc = 71 ] && echo 1)"
run primary T4 "$B" "K23P_GPU_CMD=echo N/A" "K23P_WIN_CMD=$DEF_WIN"; rc=$?
chk "a malformed GPU memory value refuses (71)" "$([ $rc = 71 ] && echo 1)"
run primary T5 "$B" "K23P_GPU_CMD=echo 1500" "K23P_WIN_CMD=$DEF_WIN"; rc=$?
chk "a GPU at the 1500 MiB limit refuses (71)" "$([ $rc = 71 ] && echo 1)"
run primary T6 "$B" "K23P_GPU_CMD=$DEF_GPU" "K23P_WIN_CMD=false"; rc=$?
chk "a failed Windows enumeration refuses (72)" "$([ $rc = 72 ] && echo 1)"
run primary T7 "$B" "K23P_GPU_CMD=$DEF_GPU" "K23P_WIN_CMD=echo 7, C:\\ollama\\ollama.exe, [N/A]"; rc=$?
chk "a Windows ML runtime refuses (73)" "$([ $rc = 73 ] && [ ! -f "$OUT/args_primary" ] && echo 1)"

run live T8 "$B" "K23P_GPU_CMD=$DEF_GPU" "K23P_WIN_CMD=$DEF_WIN" "K23P_GUARD_ARGS=--interval 1 --test-trigger" STUB_SIM=1 STUB_SLEEP=120; rc=$?
sleep 1
chk "an induced thermal pause (real guard) ends the step with 86 and a marker" "$([ $rc = 86 ] && [ -f "$OUT/guard/pause_live_T8.json" ] && echo 1)"
chk "the pause stopped the owned session (job + simulator stub gone)" "$([ "$(left)" = 0 ] && echo 1)"
chk "the marker records a real stop of the owned session; supervisor ok=0" "$(grep -q '"session"' "$OUT/guard/pause_live_T8.json" && [ "$(sup T8)" = ok=0 ] && echo 1)"

( sleep 6; pkill -f "[t]hermal_guard.py --log $OUT/guard/heat_fused_T9" ) &
run fused T9 "$B" "${C[@]}" STUB_SIM=1 STUB_SLEEP=120; rc=$?
sleep 1
chk "guard death mid-step stops the job and returns 88 (supervisor ok=0)" "$([ $rc = 88 ] && [ "$(left)" = 0 ] && grep -q "THERMAL GUARD ENDED" "$OUT/guard/step_fused_T9.log" && [ "$(sup T9)" = ok=0 ] && echo 1)"

run fused T9b "$B" "K23P_GPU_CMD=$DEF_GPU" "K23P_WIN_CMD=$DEF_WIN" "K23P_GUARD_CMD=bash $OUT/stub_guard.sh" STUB_GUARD_MODE=marker_die STUB_SIM=1 STUB_SLEEP=120; rc=$?
sleep 1
chk "a guard dying after its marker but before stopping the step: job stopped, 88 (not 86), marker kept" "$([ $rc = 88 ] && [ "$(left)" = 0 ] && [ -f "$OUT/guard/pause_fused_T9b.json" ] && echo 1)"

run state T9c "$B" "K23P_GPU_CMD=$DEF_GPU" "K23P_WIN_CMD=$DEF_WIN" "K23P_GUARD_CMD=bash $OUT/stub_guard.sh" STUB_GUARD_MODE=die_after_job STUB_SLEEP=1; rc=$?
chk "a guard failing right after the step completed: 88, supervisor ok=0 (the result is not accepted)" "$([ $rc = 88 ] && [ "$(sup T9c)" = ok=0 ] && echo 1)"

run state T9d "$B" "${C[@]}" "K23P_PGIDF=/proc/k23p_cannot_write/pgid" STUB_SIM=1 STUB_SLEEP=120; rc=$?
sleep 1
chk "a failed ownership publication stops the step (99), nothing left, supervisor ok=0" "$([ $rc = 99 ] && [ "$(left)" = 0 ] && [ "$(sup T9d)" = ok=0 ] && echo 1)"

run state T9e "$B" "${C[@]}" "K23P_PROCS_CMD=bash $OUT/procs_nostop.sh" STUB_SIM=1 STUB_SLEEP=2; rc=$?
chk "a failed cleanup is not success (89), supervisor ok=0" "$([ $rc = 89 ] && [ "$(sup T9e)" = ok=0 ] && grep -q "cleanup of the owned session FAILED" "$OUT/guard/step_state_T9e.log" && echo 1)"
pkill -f "[f]ake-thor-sim" 2>/dev/null; sleep 0.5

env "$B" "${C[@]}" KCAP_OWNER="k23p:FloorPlan15:state:WRONG" timeout 60 setsid --wait bash "$STEP" FloorPlan15 state 60 T10; rc=$?
chk "a wrong ownership token refuses (4)" "$([ $rc = 4 ] && echo 1)"

run state T11 "$B" "${C[@]}" STUB_RC=3; rc=$?
chk "the step's own exit code is passed through (3), supervisor ok=0" "$([ $rc = 3 ] && [ "$(sup T11)" = ok=0 ] && echo 1)"
pkill -f "[f]ake-thor-sim" 2>/dev/null
rm -rf "$OUT"
echo "$ok/$((ok+bad))"
[ "$bad" = 0 ]
