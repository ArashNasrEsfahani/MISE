#!/bin/bash
# 20-minute status for the K23 capture (Git Bash on Windows, read-only): a line at every :00 / :20 / :40, and at once on
# a STOP / FAIL / abort / thermal / QC line. Run as a Monitor:  bash sim/kitchen_v3/k23/status20_k23.sh FloorPlan15
SCENE="${1:-FloorPlan15}"
D="D:/Personal/Master's Project/sim/kitchen_v3/k23/$SCENE"
mkdir -p "$D/qc"; touch "$D/supervisor.log" "$D/capture.log" "$D/qc/summary.log"
PAT="STOP|FAIL|ABORTED|Traceback|THERMAL|HARD STOP|COMPLETE|K23 QC|LOCK BUSY|CLASS LIST|BATCH LIMIT|RENDER GLITCH|HUNG"
events() { cat "$D/supervisor.log" "$D/capture.log" "$D/qc/summary.log" | tr '\r' '\n' | grep -a -E "$PAT"; }
last_evt=$(events | wc -l)
status() {
  python - "$SCENE" <<'EOF'
import json, sys, time, re
s = sys.argv[1]
st = f"//wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v3/{s}/capture_state.json"
msg = f"STATUS {time.strftime('%H:%M')} {s}"
try:
    d = json.load(open(st))
    msg += f" | main capture episodes done {len(d['done_episodes'])}/168, failed {len(d.get('failed_episodes') or {})}"
except Exception:
    msg += " | capture not started"
try:
    q = [l for l in open(f"D:/Personal/Master's Project/sim/kitchen_v3/k23/{s}/qc/summary.log", encoding="utf-8")
         if l.startswith("K23 QC")]
    msg += f" | last QC: {q[-1].strip()[-60:]}" if q else " | no QC yet"
except Exception:
    msg += " | no QC yet"
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
