#!/bin/bash
# 20-minute status for the state-estimator build (classifier training, calibration); immediate lines on stage changes.
last_evt=0
status() {
  wsl.exe -d Ubuntu-22.04 -- bash -c "tr '\r' '\n' < /home/arash/train_state.log 2>/dev/null | grep -a -E 'STATE' | tail -n 7 | tr '\n' ';'; tr '\r' '\n' < /home/arash/train_state.log 2>/dev/null | grep -a '/30 ' | tail -1 | cut -c1-40" 2>/dev/null | sed "s/^/STATUS $(date +%H:%M) | /"
}
last_slot=""
while true; do
  n=$(wsl.exe -d Ubuntu-22.04 -- bash -c "tr '\r' '\n' < /home/arash/train_state.log 2>/dev/null | grep -a -c -E 'STATE |Traceback|EXIT'" 2>/dev/null | tr -d '\r')
  n=${n:-0}
  if [ "$n" -gt "$last_evt" ]; then
    wsl.exe -d Ubuntu-22.04 -- bash -c "tr '\r' '\n' < /home/arash/train_state.log | grep -a -E 'STATE |Traceback|EXIT' | tail -n $((n - last_evt))" 2>/dev/null
    last_evt=$n
  fi
  m=$((10#$(date +%M))); slot="$(date +%H):$((m / 20))"
  if [ $((m % 20)) -eq 0 ] && [ "$slot" != "$last_slot" ]; then status; last_slot=$slot; fi
  sleep 30
done
