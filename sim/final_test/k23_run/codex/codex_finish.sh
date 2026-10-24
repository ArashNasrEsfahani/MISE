#!/bin/bash
# After the K3 exploratory pass ends: plan.py's registered infra re-queue (up to 2 later sessions, --requeue-infra) when it ended
# "incomplete" (exit 6), then the addendum re-score of both kitchens with the plan-only diagnostics. Log: codex/finish.log
cd "$(dirname "$0")/../../../.." || exit 9
P=sim/final_test/k23_run/codex; L=$P/finish.log
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
note() { echo "$(date '+%F %T') $*" >> "$L"; }
endrc() { grep -a -o 'END rc=[0-9]*' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }
note "waiting for the K3 exploratory pass"
until [ -n "$(endrc $P/plan-test-k3-v1-explore.log)" ]; do sleep 60; done
rc=$(endrc $P/plan-test-k3-v1-explore.log); note "K3 exploratory pass END rc=$rc"
for s in 1 2; do
  [ "$rc" = 6 ] || break
  note "re-queue session $s (--requeue-infra)"
  python sim/final_test/matrix/k23_codex.py plan --namespace test-k3-v1 --arms explore --codex-parallel 2 --requeue-infra >> $P/plan-test-k3-v1-explore-requeue.log 2>&1; rc=$?
  note "re-queue session $s rc=$rc"
done
[ "$rc" = 0 ] || note "the exploratory pass is still not complete (rc $rc): infra cells stay infra after the registered sessions; re-scoring anyway (plan-only diagnostics report them)"
for NS in test-k2-v1 test-k3-v1; do
  python sim/final_test/matrix/k23_codex.py score --namespace "$NS" >> $P/score-$NS.log 2>&1; note "re-score $NS rc=$?"
done
note "FINISH DONE"
