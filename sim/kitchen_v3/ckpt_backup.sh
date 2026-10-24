#!/bin/bash
# Copy the running YOLO training's last.pt / best.pt out of /tmp every 2 minutes.
# Why: gen_dataset.train_scene trains into /tmp/yolo_runs_*/ (ultralytics mangles paths with an apostrophe), and
# Ubuntu's tmpfiles.d ("D /tmp") empties /tmp on every WSL boot — every driver phase starts with `wsl --shutdown`,
# so a stopped run would otherwise lose its checkpoints and restart from epoch 1. train_kitchen.py resumes from here.
DEST=/home/arash/thor-data/kitchen_v3/_ckpt
mkdir -p "$DEST"
while true; do
  for d in /tmp/yolo_runs_*/*/weights; do
    [ -f "$d/last.pt" ] || continue
    run=$(basename "$(dirname "$d")")
    mkdir -p "$DEST/$run"
    for f in last.pt best.pt; do
      if [ -f "$d/$f" ] && [ "$d/$f" -nt "$DEST/$run/$f" ]; then
        cp "$d/$f" "$DEST/$run/$f.tmp" && mv "$DEST/$run/$f.tmp" "$DEST/$run/$f"
      fi
    done
    [ -f "$d/../results.csv" ] && cp "$d/../results.csv" "$DEST/$run/results.csv"
    echo "$(dirname "$(dirname "$d")")" > "$DEST/$run/project.txt"
  done
  sleep 120
done
