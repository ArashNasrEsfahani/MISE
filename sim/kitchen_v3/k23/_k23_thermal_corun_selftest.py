"""Focused checks for the co-run refinement of thermal_guard.py (Astra K23-THERMAL-CORUN-r1 item 9). Pure / hermetic:
temp records, scripted pgrep output, no nvidia-smi, no WSL. python sim/kitchen_v3/k23/_k23_thermal_corun_selftest.py"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thermal_guard as G  # noqa: E402

N = [0, 0]


def ok(c, msg):
    N[0] += 1
    N[1] += bool(c)
    print(("PASS: " if c else "FAIL: ") + msg)


T0 = 1_000_000.0


def smp(i, temp=60.0, sm=850.0, util=96.0, corun=True, flag=False):
    return {"t": T0 + 60 * i, "ts": "x", "temp": temp, "sm": sm, "power": 22.0, "util": util, "corun": corun,
            "thermal_flag": flag}


def ev(samples):
    return G.evaluate(samples, samples[-1]["t"], samples[-1]["t"], T0)


# power-only suppression: verified co-run, flags readable and inactive, cool -> no pause
ok(ev([smp(i) for i in range(5)]) is None, "co-run + power cap only (flags inactive, 60 C): no pause")
# each thermal flag (the parser) and its effect
ok(G.thermal_flags("Active, Not Active") is True and G.thermal_flags("Not Active, Active") is True,
   "sw / hw thermal slowdown flag each read as thermal")
ok(G.thermal_flags("Not Active, Not Active") is False, "both flags inactive read as not thermal")
for bad in (None, "", "[N/A], [N/A]", "Active", "Active, Not Active, Active", "Not Supported, Not Active"):
    ok(G.thermal_flags(bad) is None, f"missing / unsupported flags {bad!r} -> unknown")
ok(ev([smp(i, flag=True) for i in range(5)]) is not None, "co-run + thermal flag active in all 5: SM rule pauses")
# 79 / 80 C boundary for the co-run temperature evidence
ok(ev([smp(i, temp=79.0) for i in range(5)]) is None, "co-run, 79 C, flags inactive: no pause")
ok(ev([smp(i, temp=80.0) for i in range(5)]) is not None, "co-run, 80 C, flags inactive: SM rule pauses")
# 86 / 87 C: the temperature rule is unchanged in every mode
ok(ev([smp(i, temp=86.0, sm=2000.0, util=50.0) for i in range(5)]) is None, "86 C x5 (no SM condition): no pause")
for mode, kw in (("co-run", dict(corun=True, flag=False)), ("no co-run", dict(corun=False, flag=None)),
                 ("unknown co-run", dict(corun=None, flag=False))):
    r = ev([smp(i, temp=87.0, sm=2000.0, util=50.0, **kw) for i in range(5)])
    ok(r is not None and "temperature" in r, f"87 C x5 pauses ({mode})")
# five-sample behaviour
ok(ev([smp(i, flag=True) for i in range(4)]) is None, "4 qualifying samples: no pause")
mix = [smp(0, flag=True), smp(1, flag=True), smp(2, flag=False), smp(3, flag=True), smp(4, flag=True)]
ok(ev(mix) is None, "one non-qualifying co-run sample in 5 breaks the run")
# missing flags or unverified co-run restore the ORIGINAL rule (pause on low clock at full load)
ok(ev([smp(i, flag=None) for i in range(5)]) is not None, "co-run but flags missing: original SM rule pauses")
ok(ev([smp(i, corun=None) for i in range(5)]) is not None, "co-run status unknown: original SM rule pauses")
ok(ev([smp(i, corun=False) for i in range(5)]) is not None, "no co-run (training exited / record revoked): original rule pauses")
legacy = [{k: v for k, v in smp(i).items() if k not in ("corun", "thermal_flag")} for i in range(5)]
ok(ev(legacy) is not None, "standalone regression: samples without the new fields pause exactly as before")
# mode change mid-run: earlier samples keep their own mode
mid = [smp(0, corun=False), smp(1, corun=False), smp(2), smp(3), smp(4)]
ok(ev(mid) is None, "co-run starts mid-run: the co-run samples need thermal evidence (no pause)")
mid2 = [smp(0), smp(1), smp(2, corun=False), smp(3, corun=False), smp(4, corun=False)]
ok(ev(mid2) is None, "training exits mid-run: earlier co-run samples still need evidence (history kept, no pause yet)")
ok(ev(mid2 + [smp(5, corun=False), smp(6, corun=False)]) is not None, "5 consecutive non-co-run low-clock samples: pause")
# no-telemetry rule unchanged
s5 = [smp(i) for i in range(5)]
ok(G.evaluate(s5, s5[-1]["t"] + 301, s5[-1]["t"], T0) is not None, "no telemetry for 301 s pauses (co-run)")
ok(G.evaluate(s5, s5[-1]["t"] + 299, s5[-1]["t"], T0) is None, "299 s without telemetry: no pause")

# the record: exact budget, expiry, revocation; and positive identification of the training job
td = tempfile.mkdtemp()
rec = os.path.join(td, "k23_corun.json")
GOOD = {"schema": "k23-corun/1", "enabled": True, "until": "2026-10-25 23:59", "capture_mib": 1024, "margin_mib": 512}
now = time.strptime("2026-10-18 20:30", "%Y-%m-%d %H:%M")


def rec_ok(d):
    json.dump(d, open(rec, "w", encoding="utf-8"))
    return G.corun_record_ok(now, rec)


ok(rec_ok(GOOD), "valid record accepted")
for name, bad in [("disabled (revoked)", dict(GOOD, enabled=False)), ("expired", dict(GOOD, until="2026-10-18 20:00")),
                  ("expiry after 10-25", dict(GOOD, until="2026-10-26 00:00")), ("other budget", dict(GOOD, capture_mib=1536)),
                  ("bool margin", dict(GOOD, margin_mib=True)), ("wrong schema", dict(GOOD, schema="x"))]:
    ok(not rec_ok(bad), f"record rejected: {name}")
open(rec, "w").write("{bad json")
ok(not G.corun_record_ok(now, rec), "unreadable record rejected")
TR = "401 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_train.py train --scene FloorPlan15 --model yolo26s_coco\n"
ok(G.corun_status(TR, record_ok=True) is True, "valid record + identified training job: co-run verified")
ok(G.corun_status("", record_ok=True) is False, "valid record, no training job (exited): not co-run")
ok(G.corun_status(TR, record_ok=False) is False, "training job but no valid record: not co-run")
ok(G.corun_status("999 pgrep -af thor-env\n", record_ok=True) is False, "only the pgrep line itself: not co-run")
for name, bad in [("malformed time", dict(GOOD, until="2026-10-24 99:99")), ("malformed date", dict(GOOD, until="2026-13-01 10:00")),
                  ("non-string expiry", dict(GOOD, until=20261025))]:
    ok(not rec_ok(bad), f"record rejected: {name}")
for name, line in [("bash -c wrapper", "500 bash -c /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_train.py train --scene x"),
                   ("timeout wrapper", "501 timeout 10 /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_train.py train"),
                   ("other python", "502 /usr/bin/python3 sim/kitchen_v3/k23/k23_train.py train --scene x"),
                   ("audit, not train", "503 /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_train.py audit --scene x"),
                   ("prefix trainer name", "504 /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_train_fake.py train"),
                   ("train as substring", "505 /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_train.py trainer")]:
    ok(G.corun_status(line + chr(10), record_ok=True) is False, f"identity rejected: {name}")
for line in ("402 /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_amb_train.py train --scene FloorPlan17 --model amb26m",
             "403 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_state_train.py train --scene FloorPlan17"):
    ok(G.corun_status(line + chr(10), record_ok=True) is True, "identity accepted: " + line.split("/")[-1][:30])
# A28-ii-SM (Arash 2026-10-19 ~00:27): a readable HARDWARE flag decides in every mode; the software flag is no evidence
def smh(i, hw, temp=56.0, corun=True, flag=True, sm=850.0, util=96.0):
    s = smp(i, temp=temp, sm=sm, util=util, corun=corun, flag=flag)
    s["hw_flag"] = hw
    return s


ok(G.hw_thermal_flag("Active, Not Active") is False and G.hw_thermal_flag("Not Active, Active") is True,
   "A28-ii-SM: the hardware flag is the second field")
for bad in (None, "", "[N/A], [N/A]", "Active", "Not Supported, Not Active"):
    ok(G.hw_thermal_flag(bad) is None, f"A28-ii-SM: unreadable flag pair {bad!r} -> unknown")
for corun in (True, False, None):
    ok(ev([smh(i, False, corun=corun) for i in range(5)]) is None,
       f"A28-ii-SM: software flag Active, hardware Not Active, 56 C, co-run {corun}: no pause")
    ok(ev([smh(i, True, corun=corun) for i in range(5)]) is not None, f"A28-ii-SM: hardware flag active, co-run {corun}: pauses")
ok(ev([smh(i, False, temp=79.0) for i in range(5)]) is None, "A28-ii-SM: 79 C, hardware flag inactive: no pause")
ok(ev([smh(i, False, temp=80.0) for i in range(5)]) is not None, "A28-ii-SM: 80 C, hardware flag inactive: pauses")
r = ev([smh(i, False, temp=87.0, sm=2000.0, util=50.0) for i in range(5)])
ok(r is not None and "temperature" in r, "A28-ii-SM: 87 C x5 still pauses (temperature rule unchanged)")
h5 = [smh(i, False) for i in range(5)]
ok(G.evaluate(h5, h5[-1]["t"] + 301, h5[-1]["t"], T0) is not None, "A28-ii-SM: no telemetry for 301 s still pauses")
ok(ev([smh(i, None, corun=False, flag=None) for i in range(5)]) is not None,
   "A28-ii-SM: unreadable flags, no co-run: original SM rule pauses (fail closed)")
REPLAY = [(870.0, 99.0), (885.0, 93.0), (795.0, 95.0), (885.0, 100.0), (780.0, 92.0)]     # pause_20261019-001102-abf4d3d1
ok(ev([smh(i, False, sm=a, util=b) for i, (a, b) in enumerate(REPLAY)]) is None,
   "A28-ii-SM: the 2026-10-19 00:16 pause samples (56 C, sw Active, hw Not Active) no longer pause")
print(f"_k23_thermal_corun_selftest: {N[1]}/{N[0]} {'OK' if N[0] == N[1] else 'FAILED'}")
sys.exit(0 if N[0] == N[1] else 1)
