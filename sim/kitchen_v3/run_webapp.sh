#!/bin/bash
# Start the web console in the background-friendly way (used when launching from Windows without START-TESTING.bat).
cd "/mnt/d/Personal/Master's Project/sim"
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0
exec /home/arash/thor-env/bin/python webapp.py > /home/arash/webapp_user.log 2>&1
