"""Focused checks for the K23 co-run exemption in k1_priority_k23.py (Astra K23-CORUN-r1 item 4). Hermetic: a temp record,
no WSL / GPU / ledger query. python sim/kitchen_v3/k23/_k23_corun_selftest.py"""
import datetime as dt
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1_priority_k23 as K  # noqa: E402

N = [0, 0]


def ok(c, msg):
    N[0] += 1
    N[1] += bool(c)
    print(("PASS: " if c else "FAIL: ") + msg)


NOW = dt.datetime(2026, 10, 18, 19, 30)
TRAIN = "wsl: JOB 1234 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_train.py train --scene FloorPlan15 --model yolo26s_coco"
AMB = "wsl: JOB 1235 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_amb_train.py train --scene FloorPlan17 --model amb26m"
STATE = "wsl: JOB 1236 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_state_train.py train --scene FloorPlan17"
WSLGPU = "gpu: 4, [Insufficient Permissions] (unknown GPU app)"
OTHERGPU = "gpu: 9876, [Insufficient Permissions] (unknown GPU app)"
UNRES = "gpu: unidentified compute app pid 4 (fail closed)"
UTIL = "gpu: working (median utilization 97 % > 40 %: e.g. Gemma generating)"
VRAM = "gpu: free VRAM 2600 MiB < 7168 MiB (capture 2560 [default] + margin 1024 + Gemma reserve 3584)"
BASE = [TRAIN, WSLGPU, UTIL, VRAM]
td = tempfile.mkdtemp()
rec = os.path.join(td, "k23_corun.json")
K.CORUN_FILE = rec
GOOD = {"schema": "k23-corun/1", "enabled": True, "until": "2026-10-25 23:59", "capture_mib": 1024, "margin_mib": 512}


def put(d):
    json.dump(d, open(rec, "w", encoding="utf-8"))


def run(why, free=2600, procs=None):
    K.VRAM_SEEN.clear(); K.VRAM_SEEN.update(free_mib=free)
    K.PROCS_SEEN.clear(); K.PROCS_SEEN.update(procs if procs is not None else {4: "System"})
    return K.corun_filter(list(why), NOW)


# no / invalid record -> nothing exempted
if os.path.exists(rec):
    os.remove(rec)
ok(run(BASE)[0] == BASE, "no record: nothing exempted")
for name, bad in [("negative budget", dict(GOOD, capture_mib=-1)), ("bool budget", dict(GOOD, margin_mib=True)),
                  ("other budget", dict(GOOD, capture_mib=1536)), ("margin 0", dict(GOOD, margin_mib=0)),
                  ("expiry after 10-25", dict(GOOD, until="2026-10-26 00:00")), ("expired", dict(GOOD, until="2026-10-18 19:00")),
                  ("disabled", dict(GOOD, enabled=False)), ("enabled 'yes'", dict(GOOD, enabled="yes")),
                  ("wrong schema", dict(GOOD, schema="k23-corun/2")), ("missing until", {k: v for k, v in GOOD.items() if k != "until"})]:
    put(bad)
    ok(run(BASE)[0] == BASE, f"invalid record ({name}): nothing exempted")
open(rec, "w").write("{not json")
ok(run(BASE)[0] == BASE, "unreadable record: nothing exempted")
# valid record
put(GOOD)
k, e = run(BASE)
ok(k == [] and len(e) == 4, "valid record, K23 training alone: all four lines exempted")
for j in (AMB, STATE):
    k, e = run([j, WSLGPU, UTIL, VRAM])
    ok(k == [], f"valid record: {j.split('/')[-1].split()[0]} job exempted")
k, e = run(BASE + ["wsl: JOB 99 /home/arash/thor-env/bin/python sim/eval.py"])
ok(set(k) == {WSLGPU, UTIL, VRAM, "wsl: JOB 99 /home/arash/thor-env/bin/python sim/eval.py"}, "mixed WSL jobs: only the training JOB line exempted")
k, e = run(BASE + ["wsl: JOB 98 /home/arash/thor-Linux64/thor-Linux64 -screen"])
ok("wsl: JOB 98 /home/arash/thor-Linux64/thor-Linux64 -screen" in k and UTIL in k, "a running simulator stays busy")
k, e = run(BASE + ["wsl: LOCK_HELD"])
ok(k == ["wsl: LOCK_HELD"], "sim lock held stays busy")
k, e = run(BASE + ["wsl: WAITER 77 flock /home/arash/sim.lock x"])
ok(k == ["wsl: WAITER 77 flock /home/arash/sim.lock x"], "lock waiter stays busy")
k, e = run([UTIL, VRAM, WSLGPU])
ok(set(k) == {UTIL, VRAM, WSLGPU}, "no training job: GPU lines stay busy")
k, e = run(BASE + [OTHERGPU])
ok(OTHERGPU in k, "an unnamed GPU app that is not pid 4 stays busy")
k, e = run([TRAIN, UNRES, UTIL, VRAM])
ok(UNRES in k, "an unresolved compute app stays busy")
k, e = run(BASE, procs={4: "vmmem"})
ok(WSLGPU in k, "pid 4 not resolved to 'System' stays busy")
k, e = run(BASE, free=1535)
ok(VRAM in k, "free VRAM 1535 MiB (< 1024 + 512) stays busy")
k, e = run(BASE, free=1536)
ok(VRAM not in k, "free VRAM 1536 MiB (= 1024 + 512) exempted")
for line in ["ledger: Kitchen 1 stage TEST-V1 is active: x", "windows: a local (GPU) planner runs: gemma", "ollama: runs but /api/ps did not answer (x)",
             "k1 test: the Kitchen 1 locked test (test-v1) has not finished execute"]:
    k, e = run(BASE + [line])
    ok(k == [line], f"Kitchen 1 rule unchanged: {line[:40]}")
print(f"_k23_corun_selftest: {N[1]}/{N[0]} {'OK' if N[0] == N[1] else 'FAILED'}")
sys.exit(0 if N[0] == N[1] else 1)
