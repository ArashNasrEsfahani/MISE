# Task A (Scene Memory ablations): resume point

Updated 2026-11-01 02:00:35.

- Current subphase: A7 complete
- Opus spend so far: 31.3744 USD (cap 60 USD; ledger sim/final_test/strengthen/work/A/opus_ledger.db)
- Done:
  - steps done: A7 blocks, A7 plan K1 Gemma, A7 plan K2 Gemma, A7 plan K3 Gemma, A7 exec K1 new, A7 exec K2 new, A7 exec K3 new, A7 exec K1 rerun, A7 exec K2 rerun, A7 exec K3 rerun

Resume from the last checkpoint (repo root, Windows):

```
python sim/final_test/strengthen/a7_analyze.py   (re-analyse only)
```

Notes:
- Stop: `python sim/final_test/strengthen/stop_all.py stop` (all tasks) or create sim/final_test/strengthen/work/A7/STOP_A7 (task A7 only).
