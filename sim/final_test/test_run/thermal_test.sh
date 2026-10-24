#!/bin/bash
# Thermal guard for the Kitchen 1 locked test (Windows Git Bash), modelled on acceptance_v2/run/thermal_a2.sh: every 30 s read the GPU
# temperature; at >= 85 C SIGSTOP the WSL simulator job (matrix_run under hang_guard + AI2-THOR), resume (SIGCONT) at <= 72 C.
# Logs to test_run/test-thermal.log. Stops when test_run/test-chain.log says CHAIN DONE / CHAIN STOPPED (and nothing is paused).
# Note: it pauses only WSL processes; the Windows plan stage (Gemma on Ollama) is not paused.
cd "/d/Personal/Master's Project" || exit 3
L=sim/final_test/test_run/test-thermal.log; C=sim/final_test/test_run/test-chain.log; paused=0
N0=$(cat $C 2>/dev/null | wc -l)   # only chain-log lines written after this guard started count (a resumed chain's old STOPPED line is ignored)
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '\r ')
  if [ -n "$t" ]; then
    if [ "$paused" = 0 ] && [ "$t" -ge 85 ]; then
      wsl -d Ubuntu-22.04 -- bash -c "pkill -STOP -f '[m]atrix_run.py'; pkill -STOP -f '[t]hor-Linux64'"; paused=1; echo "$(date '+%F %T') PAUSE at $t C" >> $L
    elif [ "$paused" = 1 ] && [ "$t" -le 72 ]; then
      wsl -d Ubuntu-22.04 -- bash -c "pkill -CONT -f '[t]hor-Linux64'; pkill -CONT -f '[m]atrix_run.py'"; paused=0; echo "$(date '+%F %T') RESUME at $t C" >> $L
    fi
    [ $(( $(date +%s) % 600 )) -lt 30 ] && echo "$(date '+%F %T') $t C paused=$paused" >> $L
  fi
  tail -n +$((N0 + 1)) $C 2>/dev/null | grep -q "CHAIN DONE\|CHAIN STOPPED" && [ "$paused" = 0 ] && { echo "$(date '+%F %T') chain ended; thermal guard exits" >> $L; exit 0; }
  sleep 30
done
