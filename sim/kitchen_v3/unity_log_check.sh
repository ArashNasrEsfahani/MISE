#!/bin/bash
# Summarise the AI2-THOR (Unity) player log: error lines by frequency + the tail (why did the simulator die?).
cd "/home/arash/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR" || exit 1
for f in Player.log Player-prev.log; do
  echo "==== $f"
  grep -a -i -E "lost|crash|signal|fatal|d3d12|mesa|out of mem|exception|error" "$f" | grep -v -i shader | sed 's/[0-9]\+/N/g' | sort | uniq -c | sort -rn | head -10 | cut -c1-170
  echo "---- tail"
  tail -n 8 "$f" | cut -c1-170
done
