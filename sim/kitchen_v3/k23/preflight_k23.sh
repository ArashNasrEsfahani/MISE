#!/bin/bash
# K23 preflight, inside WSL, READ-ONLY. Run by capture_k23.ps1 BEFORE every `wsl --shutdown`: a shutdown would kill any
# other WSL job, so it is allowed only when this prints "PREFLIGHT OK" (exit 0).
#   bash sim/kitchen_v3/k23/preflight_k23.sh <min_free_gb>
# Fails (exit 1) when: another job holds /home/arash/sim.lock; any simulator / thor-env python / webapp / training /
# matrix job runs; free space in the Linux home is below <min_free_gb>.
MIN_GB="${1:-40}"
ok=1
if ! flock -n /home/arash/sim.lock true; then echo "FAIL lock: /home/arash/sim.lock is held by another job"; ok=0; fi
JOBS=$(pgrep -af 'thor-Linux64|thor-env/bin/python|webapp\.py|train_kitchen|train_amb|train_state|final_test|step4_|build_kitchen_data' | grep -v -e preflight_k23 -e pgrep)
if [ -n "$JOBS" ]; then echo "FAIL jobs: other simulator / python jobs are running:"; echo "$JOBS" | cut -c1-200; ok=0; fi
FREE=$(df -BG --output=avail /home/arash | tail -1 | tr -dc '0-9')
if [ "${FREE:-0}" -lt "$MIN_GB" ]; then echo "FAIL disk: ${FREE} GB free in /home/arash (< ${MIN_GB} GB)"; ok=0; else echo "ok   disk: ${FREE} GB free in /home/arash"; fi
GPU=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
echo "info gpu: ${GPU:-no telemetry}"
if [ "$ok" = "1" ]; then echo "PREFLIGHT OK $(date '+%F %T')"; exit 0; fi
echo "PREFLIGHT FAIL $(date '+%F %T')"; exit 1
