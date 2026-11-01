"""Stop or resume every strengthening job at once (Arash, 2026-10-28: "make everything stoppable").

    python sim/final_test/strengthen/stop_all.py status
    python sim/final_test/strengthen/stop_all.py stop          # graceful: each job ends at its next boundary, then GPU idle check
    python sim/final_test/strengthen/stop_all.py stop --now    # immediate: kill the drivers and their simulator sessions
    python sim/final_test/strengthen/stop_all.py resume        # remove STOP and relaunch the drivers that were running
    python sim/final_test/strengthen/stop_all.py snapshot      # record the running drivers (resume after a power-off)

Graceful stop creates docs/final-test/strengthen/STOP. While it exists gpu_lock refuses to start any job and every driver
exits at its next chunk boundary (task D also before its next episode). Every job writes its records atomically, so either
kind of stop loses at most the cell, unit or episode in progress, and the resume repeats it. The command lines of the
drivers found at stop time are saved in STOPPED.json; `resume` relaunches exactly those from the repo root.
"""
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_lock as GL  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REC = os.path.join(ROOT, "docs", "final-test", "strengthen")
STOPPED = os.path.join(REC, "STOPPED.json")
LAST = os.path.join(REC, "drivers_last.json")            # snapshot of the running drivers (fallback for resume)
DRIVERS = ("l_chain.py", "l_drive.py", "d_driver.py", "a_chain.py", "a6_drive.py", "a7_drive.py", "m_drive.py", "v_drive.py", "g_drive.py", "n_drive.py")      # top-level loops; relaunched by `resume`
WSL_SESSIONS = ("[d]_chunk.sh", "[r]un_ablate_exec.sh", "[l]_exec", "[a]6_exec", "[a]7_exec", "[m]_exec", "[v]_exec", "[g]_exec", "[n]_exec")  # simulator sessions inside WSL
GRACE_S = 20 * 60


def procs():
    """Windows python processes running a strengthening script: [(pid, command line)]."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("|")
        if not pid.strip().isdigit():
            continue
        if "final_test" in cmd and "strengthen" in cmd and "stop_all.py" not in cmd and "heat_guard.py" not in cmd:
            rows.append((int(pid), cmd.strip()))
    return rows


def drivers(rows):
    return [(pid, cmd) for pid, cmd in rows if any(d in cmd for d in DRIVERS) and "exec-chunk" not in cmd]


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        used, util = [x.strip() for x in out.split(",")]
        return {"used_mib": int(used), "util_pct": int(util)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def status():
    rows = procs()
    return {"global_stop": GL.stop_requested(), "lock": GL.status_text(), "waiting": GL.live_wants(),
            "drivers": [{"pid": p, "cmd": c} for p, c in drivers(rows)], "strengthen_processes": len(rows), "gpu": gpu()}


def save_drivers(rows):
    keep = [{"pid": p, "cmd": c} for p, c in drivers(rows)]
    if not keep and os.path.exists(STOPPED):
        return                                    # a second stop must not erase the first stop's list
    tmp = STOPPED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"), "drivers": keep}, f, indent=1)
    os.replace(tmp, STOPPED)


def stop(now=False):
    rows = procs()
    save_drivers(rows)
    open(GL.STOP, "w", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S") + (" now" if now else " graceful") + "\n")
    print(f"STOP created; drivers saved for resume: {[c for _, c in drivers(rows)]}")
    if now:
        for pid, cmd in drivers(rows):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            print(f"killed driver {pid} (and its children)")
        for pat in WSL_SESSIONS:
            subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c", f"pkill -TERM -f '{pat}' || true"], capture_output=True)
        time.sleep(15)
        for pid, cmd in procs():
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            print(f"killed leftover {pid}: {cmd[:120]}")
    t0 = time.time()
    while time.time() - t0 < (120 if now else GRACE_S):
        left = procs()
        if not left and not GL.LOCK.exists():
            break
        print(f"  waiting: {len(left)} job process(es), lock {'held' if GL.LOCK.exists() else 'free'}", flush=True)
        time.sleep(30)
    if GL.LOCK.exists() and not procs():
        print(f"lock left behind by a stopped job ({GL.status_text()}); removing it")
        GL.release()
    GL.unload_ollama()
    st = status()
    print(json.dumps(st, indent=1))
    idle = not st["strengthen_processes"] and not GL.LOCK.exists()
    print("ALL STOPPED" if idle else "NOT YET STOPPED: run `stop --now`")
    return 0 if idle else 1


def snapshot():
    """Record the running drivers (done at every 20-minute update), so `resume` also works after a power-off or a crash
    that left no STOPPED.json behind."""
    keep = [{"pid": p, "cmd": c} for p, c in drivers(procs())]
    tmp = LAST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "drivers": keep}, f, indent=1)
    os.replace(tmp, LAST)
    print(f"snapshot: {[c for _, c in [(d['pid'], d['cmd']) for d in keep]]}")


def clear_dead_lock():
    """After a power-off the lock directory survives with an owner whose process is gone: remove it."""
    if not GL.LOCK.exists():
        return
    try:
        pid = json.loads(GL.status_text()).get("pid")
    except ValueError:
        pid = None
    alive = pid and str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    if not alive and not procs():
        print(f"lock owner {pid} is gone; removing the stale lock")
        GL.release()


def resume():
    if os.path.exists(GL.STOP):
        os.remove(GL.STOP)
    clear_dead_lock()
    src = STOPPED if os.path.exists(STOPPED) else LAST
    doc = json.load(open(src, encoding="utf-8")) if os.path.exists(src) else {"drivers": []}
    print(f"resuming the drivers listed in {os.path.basename(src)}")
    running = [c for _, c in drivers(procs())]
    for d in doc["drivers"]:
        cmd = d["cmd"]
        if cmd in running:
            print(f"already running: {cmd}")
            continue
        script = next(x for x in cmd.split() if any(x.endswith(n) for n in DRIVERS))
        args = cmd.split(script, 1)[1].split()
        rel = os.path.relpath(script.strip('"'), ROOT) if os.path.isabs(script.strip('"')) else script
        p = subprocess.Popen([sys.executable, rel, *args], cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"relaunched {rel} {' '.join(args)} -> pid {p.pid}")
    os.replace(STOPPED, STOPPED.replace(".json", f".resumed-{time.strftime('%Y%m%d-%H%M%S')}.json")) if os.path.exists(STOPPED) else None
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "status":
        print(json.dumps(status(), indent=1))
    elif a[0] == "stop":
        raise SystemExit(stop(now="--now" in a))
    elif a[0] == "resume":
        raise SystemExit(resume())
    elif a[0] == "snapshot":
        snapshot()
    else:
        raise SystemExit(__doc__)
