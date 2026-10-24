#!/bin/bash
# Renderer evidence of the live AI2-THOR process(es): GPU path = libd3d12 / libdxcore mapped; software = llvmpipe / swrast (K2-RENDER-WINDOW).
for P in $(ps -eo pid,cmd | awk '/thor-Linux64-f0/ && !/awk/ {print $1}'); do
  echo "pid $P started $(ps -o lstart= -p $P)"
  grep -o -E "[a-z0-9_]*d3d12[a-z0-9_.]*|libdxcore[a-z0-9_.]*|swrast[a-z0-9_.]*|llvmpipe|libLLVM[a-z0-9_.-]*" /proc/$P/maps | sort -u | head -8
  top -b -n 2 -d 3 -p $P | tail -1 | cut -c1-100
done
