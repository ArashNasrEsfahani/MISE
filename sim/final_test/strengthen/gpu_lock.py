"""One GPU, one job: a lock shared by Windows and WSL processes (an atomic directory on the D: project drive).

    python sim/final_test/strengthen/gpu_lock.py run <task> -- <command ...>   # wait for the lock, run, release
    python sim/final_test/strengthen/gpu_lock.py run --low <task> -- <command ...>   # low priority: only when no other job waits
    python sim/final_test/strengthen/gpu_lock.py status

Global stop: while docs/final-test/strengthen/STOP exists no new job starts (`run` exits 77, waiters give up) and every
driver exits at its next boundary; see stop_all.py. Drivers of normal tasks call hold(task) for their lifetime so a
low-priority job (task D) never starts in the short gap between two of their chunks. Want files are refreshed every
30 s; one older than STALE_S is left over from a killed process and ignored.

The lock directory holds owner.json (task, command, pid, host, start time). `run` waits (polling every 30 s) while another
job holds it and always releases it, also when the command fails. Exit code = the command's exit code.
"""
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

LOCK = pathlib.Path(__file__).resolve().parents[3] / "docs" / "final-test" / "strengthen" / "gpu.lock"


WANTS = LOCK.parent / "gpu.wants"      # one file per normal-priority job waiting for the GPU (or driver holding its place)
STOP = LOCK.parent / "STOP"            # global stop (stop_all.py)
STALE_S = 180                          # a want file not refreshed for this long belongs to a killed process
STOPPED = 77                           # exit code of `run` when the global STOP is present


def live_wants():
    """Names of want files refreshed within STALE_S (stale ones are ignored, never deleted here)."""
    out = []
    try:
        for f in WANTS.iterdir():
            try:
                if time.time() - f.stat().st_mtime < STALE_S:
                    out.append(f.name)
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    return sorted(out)


def stop_requested():
    return STOP.exists()


def _born(f):
    st = f.stat()
    return getattr(st, "st_birthtime", st.st_ctime)


def my_turn(want):
    """First come, first served among normal jobs (2026-10-28 23:10: a driver that re-asks the instant it releases otherwise wins every
    race against a waiter polling every 30 s). Only live waiting jobs count, not driver holds."""
    queue = []
    for name in live_wants():
        if "-hold-" in name:
            continue
        f = WANTS / name
        try:
            queue.append((_born(f), name))
        except FileNotFoundError:
            pass
    return not queue or min(queue)[1] == want.name


class hold:
    """Context manager for a normal task's driver: keeps gpu.wants/<task>-hold-<pid> fresh for the driver's lifetime,
    so a low-priority job does not start between the driver's chunks. Normal-priority jobs are not affected."""

    def __init__(self, task):
        self.path = WANTS / f"{task}-hold-{os.getpid()}"
        self._stop = None

    def __enter__(self):
        import threading
        WANTS.mkdir(exist_ok=True)
        self.path.write_text("driver hold", encoding="utf-8")
        self._stop = threading.Event()

        def beat():
            while not self._stop.wait(30):
                try:
                    self.path.touch()
                except OSError:
                    pass
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.path.unlink(missing_ok=True)
        return False


def acquire(task, cmd, low=False):
    """Normal jobs register a want file while they wait; a low-priority job (task D) only takes a free GPU when no
    normal job is waiting, so dataset capture never delays tasks R, A or L. Returns False when the global STOP appears."""
    announced = False
    want = None
    if not low:
        WANTS.mkdir(exist_ok=True)
        want = WANTS / f"{task}-{os.getpid()}"
        want.write_text(cmd, encoding="utf-8")
    while True:
        if stop_requested():
            if want is not None:
                want.unlink(missing_ok=True)
            print(f"[gpu_lock] {task}: global STOP present, not starting", flush=True)
            return False
        if want is not None:
            try:
                want.touch()
            except OSError:
                pass
        if low and live_wants():
            time.sleep(30)
            continue
        if want is not None and not my_turn(want):
            time.sleep(5)
            continue
        try:
            LOCK.mkdir()
            if want is not None:
                want.unlink(missing_ok=True)
            (LOCK / "owner.json").write_text(json.dumps({
                "task": task, "cmd": cmd, "pid": os.getpid(), "host": platform.system(),
                "start": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
            return True
        except FileExistsError:
            if not announced:
                print(f"[gpu_lock] {task} waiting; held by {status_text()}", flush=True)
                announced = True
            time.sleep(5 if want is not None else 30)


def unload_ollama():
    """Free GPU memory held by Ollama's keep-alive (a model stays loaded for minutes after its last call), so the next
    job (often the simulator) finds the GPU empty. Best effort: silently skipped when Ollama is not reachable."""
    import urllib.request
    base = "http://127.0.0.1:11434"
    try:
        with urllib.request.urlopen(base + "/api/ps", timeout=3) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
        for m in models:
            req = urllib.request.Request(base + "/api/generate", data=json.dumps({"model": m, "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def release():
    unload_ollama()
    try:
        (LOCK / "owner.json").unlink(missing_ok=True)
        LOCK.rmdir()
    except FileNotFoundError:
        pass


def status_text():
    try:
        return (LOCK / "owner.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "free" if not LOCK.exists() else "held (owner unknown)"


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "status":
        print(status_text())
        w = live_wants()
        if w or stop_requested():
            print(json.dumps({"waiting": w, "global_stop": stop_requested()}))
        sys.exit(0)
    if len(sys.argv) < 5 or sys.argv[1] != "run" or "--" not in sys.argv:
        sys.exit(__doc__)
    low = sys.argv[2] == "--low"
    task = sys.argv[3] if low else sys.argv[2]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    if not acquire(task, " ".join(cmd), low=low):
        sys.exit(STOPPED)
    print(f"[gpu_lock] {task} acquired at {time.strftime('%H:%M:%S')}", flush=True)
    try:
        code = subprocess.call(cmd)
    finally:
        release()
        print(f"[gpu_lock] {task} released at {time.strftime('%H:%M:%S')}", flush=True)
    sys.exit(code)
