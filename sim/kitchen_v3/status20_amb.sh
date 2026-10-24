#!/bin/bash
# 20-minute status for the ambiguity capture + specialist training; immediate lines on errors / stage changes.
# status20_amb.sh [scene]  -- scene defaults to FloorPlan1 (K23-PREP 2026-10-14).
export K23_SCENE="${1:-FloorPlan1}"
LOG="D:/Personal/Master's Project/sim/kitchen_v3/amb.log"
PAT="CHECKPOINT|Traceback|STOPPED|EXIT|ALL DONE|yolo_amb:|FAIL|failed|redo [4-6]|SPECIALIST|CONTROL|FUSION|LCURVE"
touch "$LOG"
events() { tr '\r' '\n' < "$LOG" | grep -a -E "$PAT"; }
last_evt=$(events | wc -l)
status() {
  python - <<'EOF'
import json, os, time, glob, csv
st = f"//wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v3_amb/{os.environ.get('K23_SCENE', 'FloorPlan1')}/capture_state.json"
msg = f"STATUS {time.strftime('%H:%M')}"
try:
    d = json.load(open(st))
    msg += f" | amb capture episodes done {len(d['done_episodes'])}/70, failed {len(d.get('failed_episodes') or {})}"
except Exception:
    msg += " | amb capture not started"
runs = sorted(glob.glob("//wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v3/_ckpt/*/results.csv"),
              key=os.path.getmtime)
if runs:
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(runs[-1]))]
    if rows:
        r = rows[-1]
        msg += (f" | training {os.path.basename(os.path.dirname(runs[-1]))} epoch {r['epoch']}: val mAP50 "
                f"{float(r['metrics/mAP50(B)']):.3f} / 50-95 {float(r['metrics/mAP50-95(B)']):.3f}")
print(msg, flush=True)
EOF
}
last_slot=""
while true; do
  n=$(events | wc -l)
  if [ "$n" -gt "$last_evt" ]; then events | tail -n $((n - last_evt)) | cut -c1-260; last_evt=$n; fi
  m=$((10#$(date +%M))); slot="$(date +%H):$((m / 20))"
  if [ $((m % 20)) -eq 0 ] && [ "$slot" != "$last_slot" ]; then status; last_slot=$slot; fi
  sleep 20
done
