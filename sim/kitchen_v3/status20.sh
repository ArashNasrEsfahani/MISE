#!/bin/bash
# Emit a one-line training status at every :00 / :20 / :40, and immediately on errors / stage changes.
# Epoch and scores come from the run's results.csv (backed up by ckpt_backup.sh), which survives a stop + resume.
LOG="D:/Personal/Master's Project/sim/kitchen_v3/balance_train.log"
CK="//wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v3/_ckpt"
PAT="^\[[0-9:]{8}\] |^PHASE |Traceback|STOPPED|ALL BALANCE|SKIP "
events() { tr '\r' '\n' < "$LOG" | grep -a -E "$PAT"; }
last_evt=$(events | wc -l)
status() {
  python - "$CK" <<'EOF'
import csv, glob, os, sys, time
runs = sorted(glob.glob(os.path.join(sys.argv[1], "*", "results.csv")), key=os.path.getmtime)
if not runs:
    print(f"STATUS {time.strftime('%H:%M')} no run checkpoint yet (training starting or evaluating)")
    raise SystemExit
p = runs[-1]
rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(p))]
k50, k95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
last = rows[-1]
best = max(rows, key=lambda r: float(r[k95]))
print(f"STATUS {time.strftime('%H:%M')} {os.path.basename(os.path.dirname(p)).split('.')[-1]} epoch {last['epoch']}/60 done"
      f" | last val: P {float(last['metrics/precision(B)']):.3f} R {float(last['metrics/recall(B)']):.3f}"
      f" mAP50 {float(last[k50]):.3f} mAP50-95 {float(last[k95]):.3f}"
      f" | best epoch {best['epoch']}: mAP50 {float(best[k50]):.3f} mAP50-95 {float(best[k95]):.3f}")
EOF
}
last_slot=""
while true; do
  n=$(events | wc -l)
  if [ "$n" -gt "$last_evt" ]; then
    events | tail -n $((n - last_evt)) | cut -c1-300
    last_evt=$n
  fi
  m=$((10#$(date +%M)))
  slot="$(date +%H):$((m / 20))"
  if [ $((m % 20)) -eq 0 ] && [ "$slot" != "$last_slot" ]; then status; last_slot=$slot; fi
  sleep 20
done
