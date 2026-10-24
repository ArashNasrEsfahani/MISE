#!/bin/bash
# After the K3 state models (exact line from k23_gpu_chain_state.sh): K3 calibration -> K3 perception freeze (+ both perception readers)
# -> K3 A29 applicability (sim) -> K3 A29 sample -> then the K2 COCO-start sensitivity (replaces k23_gpu_chain_coco.sh, so G9 runs before
# COCO instead of pausing it). A failure before COCO is logged and COCO still starts (the GPU is not left idle); --add-coco stays manual.
# usage: bash sim/kitchen_v3/k23/k23_k3_post_state.sh   (WSL; launch DETACHED from PowerShell)
cd "/mnt/d/Personal/Master's Project"
L=sim/kitchen_v3/k23/FloorPlan17/chain_gpu.log
PY=/home/arash/thor-env/bin/python
export PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') K3-POST START (waiting for the state chain to finish with rc 0)" >> "$L"
until grep -qE "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} state END rc=0; CHAIN-STATE DONE$" "$L"; do sleep 60; done
while pgrep -f "[k]23_train.py train|[k]23_amb_train.py train|[k]23_state_train.py train" >/dev/null; do sleep 30; done
ok=1
echo "$(date '+%F %T') K3-POST calibration" >> "$L"
$PY sim/kitchen_v3/k23/k23_state_calib.py calib --scene FloorPlan17 >> "$L" 2>&1 || { ok=0; echo "$(date '+%F %T') K3-POST calibration FAILED" >> "$L"; }
if [ $ok -eq 1 ]; then
  echo "$(date '+%F %T') K3-POST perception freeze" >> "$L"
  $PY sim/kitchen_v3/k23/k23_freeze.py --scene FloorPlan17 >> "$L" 2>&1 || { ok=0; echo "$(date '+%F %T') K3-POST freeze FAILED" >> "$L"; }
fi
if [ $ok -eq 1 ]; then
  $PY -c "import sys; sys.path.insert(0,'sim/final_test/matrix'); import k23_ns as K, a29_applicability as A; p=K.perception_problems('FloorPlan17'); v=A.verify_freeze('FloorPlan17'); print('K3 perception_problems', p, 'record', v.get('record_sha256')); assert p==[]" >> "$L" 2>&1 || { ok=0; echo "$(date '+%F %T') K3-POST freeze readers FAILED" >> "$L"; }
fi
if [ $ok -eq 1 ]; then
  echo "$(date '+%F %T') K3-POST A29 applicability" >> "$L"
  bash sim/kitchen_v3/k23/a29_run.sh FloorPlan17 1 || { ok=0; echo "$(date '+%F %T') K3-POST A29 stopped (see FloorPlan17/a29_driver.log)" >> "$L"; }
fi
if [ $ok -eq 1 ]; then
  U=$($PY -c "import json; d=json.load(open('docs/final-test/k23/applicability/FloorPlan17/applicability.json')); print(sum(1 for v in d['items'].values() if (v if isinstance(v,str) else v.get('class'))=='unresolved'))" 2>>"$L")
  if [ "$U" = "0" ]; then
    $PY sim/final_test/matrix/a29_sample.py --kitchen FloorPlan17 --applicable docs/final-test/k23/applicability/FloorPlan17/applicability.json >> "$L" 2>&1 && echo "$(date '+%F %T') K3-POST sample written" >> "$L" || echo "$(date '+%F %T') K3-POST sample FAILED" >> "$L"
  else
    echo "$(date '+%F %T') K3-POST sample NOT written: unresolved items = $U (retry round needed first)" >> "$L"
  fi
fi
echo "$(date '+%F %T') K3-POST -> K2 COCO-start sensitivity (resume)" >> "$L"
bash sim/kitchen_v3/k23/train_k23.sh FloorPlan15 yolo26s_coco; RC=$?
echo "$(date '+%F %T') K2 COCO END rc=$RC; K3-POST DONE" >> "$L"
exit $RC
