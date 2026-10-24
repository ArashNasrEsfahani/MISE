#!/bin/bash
# A/B: old console (git HEAD worktree at D:/Personal/oldconsole) vs current, same machine state, measured by
# console_bench.py. Each run: fresh simulator, 60 s warm-up allowance, bench, stop.
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
BENCH="/mnt/d/Personal/Master's Project/sim/kitchen_v3/console_bench.py"
run() {   # $1 label, $2 sim dir
  pkill -f '[w]ebapp.py'; pkill -f '[t]hor-Linux64'; sleep 3
  cd "$2" && /home/arash/thor-env/bin/python webapp.py > "/home/arash/ab_$1.log" 2>&1 &
  sleep 5
  /home/arash/thor-env/bin/python "$BENCH" "$1"
  pkill -f '[w]ebapp.py'; pkill -f '[t]hor-Linux64'; sleep 3
}
run old "/mnt/d/Personal/oldconsole/sim"
run new "/mnt/d/Personal/Master's Project/sim"
