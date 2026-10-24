"""Thermal logger + pause watch for one GPU session (TODO A17; A28 item 10). Infrastructure only: it changes no recipe.

Every ``--interval`` seconds (60) it appends one nvidia-smi sample (timestamp, temperature.gpu, clocks.sm, power.draw,
utilization.gpu) to a per-session CSV log and evaluates the registered pause condition (A28 item 10, decision 13 #5,
values as registered, operational rules only):

  * GPU temperature >= 87 C in 5 consecutive 60-s samples, or
  * SM clock < 1,000 MHz in 5 consecutive samples with utilization.gpu >= 90 %, or
  * no telemetry for 5 minutes.

On a pause it does a JOB-SPECIFIC stop (TERM, then KILL, to the capture's OWN process group only), writes a marker
file with the reason and the last samples, and exits 86. It never restarts the job: in this capture window a pause
ends the window (the resume rule proposed in A28 item 15; A28 item 10's resume condition is otherwise OPEN).
A sample counts only with all three rule fields (temperature, SM clock, utilization); otherwise it is "no telemetry".
The stop is TERM, then KILL after a grace period, for the guarded job and the simulator. ``--ready`` proves startup. A heat pause is an infrastructure event, never an outcome.
Ending the guard (Astra K23-PREP r2 1): the batch script never kills a guard that may be mid-pause. It creates the
``--stop-file``; the guard checks it only BETWEEN iterations (each iteration = sample, evaluate, and on a pause the
marker -> stop -> marker update), so a pause in progress always completes and its marker exists before the guard exits.
Real vs induced (Astra K23-PREP r3 1): the marker carries ``"induced": true`` ONLY for the pause-test trigger itself
(a real condition is evaluated first and wins), so k23_decide.py ends the capture window on every real pause.

    # WSL, started in the background by capture_one_k23.sh next to the job it guards:
    python sim/kitchen_v3/k23/thermal_guard.py --log <csv> --marker <json> --stop-pattern "[k]23_capture.py --scene FloorPlan15"

``evaluate`` is pure (selftested in _k23_selftest.py; co-run refinement in _k23_thermal_corun_selftest.py).

CO-RUN REFINEMENT (Arash 2026-10-18 ~20:05 'Refine the rule, keep both'; Astra K23-THERMAL-CORUN-r1): a sample taken while the
K23 co-run is VERIFIED for that sample (k23_corun.json valid -- schema, enabled, exact 1024/512 MiB, until <= 2026-10-25 23:59 --
AND a positively identified K23 training job runs) AND whose nvidia-smi thermal-slowdown flags were read, counts toward the
SM-clock rule only with thermal evidence: sw or hw thermal slowdown active, or temperature >= 80 C. Power capping alone does
not qualify. Any sample with missing / unsupported flags or unverified co-run uses the ORIGINAL rule. The >= 87 C x5 and the
300-s no-telemetry rules are unchanged in every mode; the pause still ends the capture window.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEMP_MAX_C = 87.0
SM_MIN_MHZ = 1000.0
UTIL_FULL = 90.0
CONSECUTIVE = 5
NO_TELEMETRY_S = 300.0
INTERVAL_S = 60.0
EXIT_PAUSE = 86
TEST_REASON = f"TEST TRIGGER after {CONSECUTIVE} samples with the job and the simulator running (pause test)"
FIELDS = ("timestamp", "temperature.gpu", "clocks.sm", "power.draw", "utilization.gpu")
CORUN_TEMP_C = 80.0                 # co-run: thermal evidence by temperature (prospective operational threshold)
FLAG_FIELDS = ("clocks_throttle_reasons.sw_thermal_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown")
CORUN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k23_corun.json")
CORUN_LAST = "2026-10-25 23:59"
K23_TRAIN_PGREP = "thor-env/bin/python.* sim/kitchen_v3/k23/(k23_train|k23_amb_train|k23_state_train)[.]py train"
import re  # noqa: E402
K23_TRAIN_LINE = re.compile(r"^\d+ /home/arash/thor-env/bin/python(?: -u)? sim/kitchen_v3/k23/"
                            r"(?:k23_train|k23_amb_train|k23_state_train)\.py train(?: |$)")


def corun_record_ok(now_struct=None, path=None) -> bool:
    """The pinned co-run record is valid now (same rule as k1_priority_k23.corun_record). Any error = False."""
    try:
        r = json.load(open(path or CORUN_FILE, encoding="utf-8"))
        nowt = time.strftime("%Y-%m-%d %H:%M", now_struct or time.localtime())
        cm, mm = r.get("capture_mib"), r.get("margin_mib")
        until = time.strptime(r["until"], "%Y-%m-%d %H:%M")            # a malformed expiry raises -> False (Astra r2 4)
        last = time.strptime(CORUN_LAST, "%Y-%m-%d %H:%M")
        cur = time.strptime(nowt, "%Y-%m-%d %H:%M")
        return (r.get("schema") == "k23-corun/1" and r.get("enabled") is True and type(cm) is int and type(mm) is int
                and cm == 1024 and mm == 512 and until <= last and cur <= until)
    except Exception:
        return False


def corun_status(pgrep_out=None, record_ok=None) -> "bool | None":
    """True = co-run verified for this sample (record valid AND a K23 training job positively identified); False = not
    co-run; None = could not tell (the caller then uses the ORIGINAL rule)."""
    ok = corun_record_ok() if record_ok is None else record_ok
    if not ok:
        return False
    if pgrep_out is None:
        try:
            r = subprocess.run(["pgrep", "-af", K23_TRAIN_PGREP], capture_output=True, text=True, timeout=20)
        except Exception:
            return None
        if r.returncode not in (0, 1):
            return None
        pgrep_out = r.stdout
    # positive identity only (Astra r2 5): '<pid> /home/arash/thor-env/bin/python [-u] sim/kitchen_v3/k23/<k23 trainer>.py train ...'
    # with exact executable and argument boundaries; wrappers (bash -c '...', timeout, nohup) never establish identity
    return any(K23_TRAIN_LINE.match(l.strip()) for l in pgrep_out.splitlines())


def thermal_flags(line) -> "bool | None":
    """'Active'/'Not Active' x2 -> True if any thermal slowdown is active, False if both inactive; None if missing."""
    if not line:
        return None
    parts = [p.strip().lower() for p in line.split(",")]
    if len(parts) != 2 or any(p not in ("active", "not active") for p in parts):
        return None
    return any(p == "active" for p in parts)


def hw_thermal_flag(line) -> "bool | None":
    """The HARDWARE thermal slowdown flag alone (second field): True / False; None if the pair is missing or unreadable."""
    if thermal_flags(line) is None:
        return None
    return line.split(",")[1].strip().lower() == "active"


def _num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def parse_line(line: str, t_wall: float) -> dict | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 5:
        return None
    s = {"t": t_wall, "ts": parts[0], "temp": _num(parts[1]), "sm": _num(parts[2]), "power": _num(parts[3]),
         "util": _num(parts[4])}
    if s["temp"] is None or s["sm"] is None or s["util"] is None:
        return None          # a sample without ALL three rule fields is no telemetry (Astra K23-PREP 6)
    return s


def sm_rule_sample_ok(s: dict) -> bool:
    """Co-run refinement: under VERIFIED co-run with readable thermal flags, the low-clock sample needs thermal evidence
    (a thermal slowdown flag, or temperature >= 80 C); otherwise (no co-run, unknown co-run, missing flags) the original
    rule applies unchanged (True).
    A28-ii-SM (Arash 2026-10-19 ~00:27, ledger heading of that name; Astra K23-A28SM-r1): whenever the HARDWARE
    thermal-slowdown flag is readable, in every mode, a low-clock sample counts only with that flag active or temperature
    >= 80 C. The software flag is not thermal evidence: on this laptop it reads Active at 55 C, under any clock cap.
    An unreadable flag pair falls through to the co-run refinement / original rule (fail closed)."""
    if s.get("hw_flag") is not None:
        return bool(s["hw_flag"]) or (s.get("temp") is not None and s["temp"] >= CORUN_TEMP_C)
    if s.get("corun") is True and s.get("thermal_flag") is not None:
        return bool(s["thermal_flag"]) or (s.get("temp") is not None and s["temp"] >= CORUN_TEMP_C)
    return True


def evaluate(samples: list, now: float, last_good_t: float | None, started_t: float,
             test_trigger: bool = False) -> str | None:
    """The registered pause condition over the good samples so far (newest last). Returns the reason or None.

    ``test_trigger``: the pause TEST (Astra night-checklist addition 7): once 5 good samples exist AND the caller saw
    the guarded job and the simulator running (``test_trigger`` is then True), the pause fires whatever the readings,
    so the job-specific stop can be shown to remove the GPU load. Never used in a real run.
    A REAL condition (no telemetry, heat, low clock at full load) is evaluated FIRST and always wins over the test
    trigger, so a real pause during the pause test is never mistaken for the induced one (Astra K23-PREP r3 1)."""
    ref = last_good_t if last_good_t is not None else started_t
    if now - ref >= NO_TELEMETRY_S:
        return f"no telemetry for {now - ref:.0f} s (>= {NO_TELEMETRY_S:.0f} s)"
    last = samples[-CONSECUTIVE:]
    if len(last) == CONSECUTIVE:
        if all(s["temp"] is not None and s["temp"] >= TEMP_MAX_C for s in last):
            return f"GPU temperature >= {TEMP_MAX_C:.0f} C in {CONSECUTIVE} consecutive samples " \
                   f"({[s['temp'] for s in last]})"
        if all(s["sm"] is not None and s["util"] is not None and s["sm"] < SM_MIN_MHZ and s["util"] >= UTIL_FULL
               and sm_rule_sample_ok(s) for s in last):
            return f"SM clock < {SM_MIN_MHZ:.0f} MHz with utilization >= {UTIL_FULL:.0f} % in {CONSECUTIVE} " \
                   f"consecutive samples ({[(s['sm'], s['util']) for s in last]})"
    if test_trigger and len(samples) >= CONSECUTIVE:
        return TEST_REASON
    return None


def is_induced(reason: str | None, test_flag: bool) -> bool:
    """True only for the pause TEST's own trigger (never for a real heat / clock / telemetry condition)."""
    return bool(test_flag and reason == TEST_REASON)


def nvidia_smi() -> str:
    if os.environ.get("K23_TEST_MODE") == "1" and os.environ.get("K23_TEST_SMI_CMD"):
        return os.environ["K23_TEST_SMI_CMD"]        # integration test only: scripted telemetry (a REAL-heat pause)
    return shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"


def sample_once(timeout: float = 20.0) -> str | None:
    try:
        r = subprocess.run([nvidia_smi(), "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def sample_flags(timeout: float = 20.0):
    """The two thermal-slowdown flags (a separate query: an unsupported field never spoils the registered sample)."""
    try:
        r = subprocess.run([nvidia_smi(), "--query-gpu=" + ",".join(FLAG_FIELDS), "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


SIM_PATTERN = "[t]hor-Linux64"
if os.environ.get("K23_TEST_MODE") == "1" and os.environ.get("K23_TEST_SIM_PATTERN"):
    SIM_PATTERN = os.environ["K23_TEST_SIM_PATTERN"]          # integration test only: a stub stands for the simulator


def read_pgid(path: str) -> int:
    """The capture's published session id, ONLY when the record is a live one of THIS batch (Astra K23-PREP r4 4,
    r5 4): k23_procs.record_owner (same boot, the leader alive with the recorded START TIME and carrying the token, or
    tagged members of its session alive) AND the record's token equals this guard's own KCAP_OWNER. Else 0 = nothing
    owned (the guard then stops nothing)."""
    try:
        import k23_procs
        tok = open(path, encoding="utf-8").read().split()[3]
    except Exception:
        return 0
    mine = os.environ.get("KCAP_OWNER", "")
    if not mine or tok != mine:
        return 0
    try:
        return k23_procs.record_owner(path)
    except Exception:
        return 0


def running(pattern: str, pgid: int = 0) -> int:
    """Processes matching ``pattern``; with ``pgid`` only those in the capture's OWN SESSION (all its process groups:
    the session holder leads it, `timeout` and the capture form groups inside it -- r5 3)."""
    cmd = ["pgrep", "-fc", pattern] if not pgid else ["pgrep", "-s", str(pgid), "-fc", pattern]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def stop_job(pgid: int, pattern: str, grace: float = 15.0) -> list:
    """Stop the capture's own session (the capture, the simulator it started, every process group and tokenless child
    of that K23-created session, and their descendants) through k23_procs (pidfd + start-time re-verification; TERM,
    grace, STOP, KILL). Never a name-wide kill, never a group kill by number (Astra K23-PREP r2 5, r5 4)."""
    if pgid <= 1:
        return [{"error": f"no owned process group ({pgid}); nothing killed"}]
    tok = os.environ.get("KCAP_OWNER", "")
    if not tok:
        return [{"error": "no KCAP_OWNER token: nothing can be proven ours; nothing killed"}]
    try:
        import k23_procs
        r = k23_procs.do_stop(tok, session=pgid, grace=grace, deadline=time.time() + grace + 20)
    except Exception as exc:
        return [{"error": f"stop failed: {exc!r}"}]
    return [{"session": pgid, "stop": r}, {"left": {"job": running(pattern, pgid), "simulator": running(SIM_PATTERN, pgid),
                                                    "owned": r.get("left")}}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True, help="per-session CSV log (appended)")
    ap.add_argument("--marker", required=True, help="written only on a pause")
    ap.add_argument("--stop-pattern", required=True, help="pgrep -f pattern of THE guarded job (inside its group)")
    ap.add_argument("--pgid-file", required=True, help="file with the guarded capture's process-group id")
    ap.add_argument("--interval", type=float, default=INTERVAL_S)
    ap.add_argument("--stop-file", default=None, help="the batch asks the guard to end (checked between iterations)")
    ap.add_argument("--parent-pid", type=int, default=0, help="exit quietly when this process is gone")
    ap.add_argument("--test-trigger", action="store_true", help="pause test: fire after 5 samples once job + sim run")
    ap.add_argument("--ready", default=None, help="file written after the first good sample (startup proof)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.log)) or ".", exist_ok=True)
    new = not os.path.exists(a.log)
    started = time.time()
    good, last_good = [], None
    with open(a.log, "a", encoding="utf-8") as fh:
        if new:
            fh.write("wall_time," + ",".join(FIELDS) + "\n")
        fh.write(f"# session start {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} guard of "
                 f"{a.stop_pattern!r}\n")
        fh.flush()
        def stop_requested() -> bool:
            return bool(a.stop_file and os.path.exists(a.stop_file))

        while True:
            if a.parent_pid and not os.path.exists(f"/proc/{a.parent_pid}"):
                fh.write(f"# session end {time.strftime('%Y-%m-%d %H:%M:%S')} (guarded job finished)\n")
                return 0
            if stop_requested():
                fh.write(f"# session end {time.strftime('%Y-%m-%d %H:%M:%S')} (stop requested by the batch)\n")
                return 0
            now = time.time()
            line = sample_once()
            s = parse_line(line, now) if line else None
            if s is not None:
                fl_line = sample_flags()
                s["thermal_flag"] = thermal_flags(fl_line)
                s["hw_flag"] = hw_thermal_flag(fl_line)
                s["corun"] = corun_status()
                good.append(s)
                good = good[-50:]
                last_good = now
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{line}\n")
                fh.write(f"# corun {s['corun']} thermal_flags {fl_line!r} -> {s['thermal_flag']} hw {s['hw_flag']}\n")
                if a.ready and not os.path.exists(a.ready):
                    with open(a.ready, "w", encoding="utf-8") as rh:
                        rh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
            else:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},NO_TELEMETRY\n")
            fh.flush()
            pgid = read_pgid(a.pgid_file)
            before = {"job": running(a.stop_pattern, pgid) if pgid else 0,
                      "simulator": running(SIM_PATTERN, pgid) if pgid else 0, "pgid": pgid}
            trig = bool(a.test_trigger and before["job"] and before["simulator"])
            why = evaluate(good, now, last_good, started, trig)
            if why:
                rec = {"paused_at": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": why,
                       "rule": {"temp_max_c": TEMP_MAX_C, "sm_min_mhz": SM_MIN_MHZ, "util_full": UTIL_FULL,
                                "consecutive": CONSECUTIVE, "no_telemetry_s": NO_TELEMETRY_S,
                                "corun_refinement": "verified co-run + readable flags: low clock needs a thermal flag or temp >= 80 C",
                                "a28_ii_sm": "readable hardware flag (any mode): low clock needs the hardware flag or temp >= 80 C"},
                       "last_samples": good[-CONSECUTIVE:], "running_before": before, "stopped": "pending",
                       "test": bool(a.test_trigger), "induced": is_induced(why, bool(a.test_trigger)),
                       "resume": "NOT automatic: a pause ends this capture window (A28 item 15.6 (ii))"}
                # the marker (pause INTENT) is written BEFORE any signal, so the batch can never see the capture
                # end without seeing the pause (Astra K23-PREP r2 1)
                with open(a.marker + ".tmp", "w", encoding="utf-8") as mh:
                    json.dump(rec, mh, indent=1)
                os.replace(a.marker + ".tmp", a.marker)
                rec["stopped"] = stop_job(pgid, a.stop_pattern)
                with open(a.marker + ".tmp", "w", encoding="utf-8") as mh:
                    json.dump(rec, mh, indent=1)
                os.replace(a.marker + ".tmp", a.marker)
                fh.write(f"# PAUSE {rec['paused_at']}: {why}\n")
                print(f"THERMAL PAUSE: {why}", file=sys.stderr, flush=True)
                return EXIT_PAUSE
            t_end = time.time() + a.interval
            while time.time() < t_end and not stop_requested():
                if a.parent_pid and not os.path.exists(f"/proc/{a.parent_pid}"):
                    break                         # the batch script is gone: end at the loop top (r3 3)
                time.sleep(min(1.0, max(0.0, t_end - time.time())))


if __name__ == "__main__":
    raise SystemExit(main())
