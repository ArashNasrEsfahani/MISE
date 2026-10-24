"""Windows-side launcher for the W4 practice run (WSL, Gemma). Restarts the run after a hang-guard abort (exit 75) or a stall
(no stdout and no journal growth for STALL_SECS), kills leftover simulator processes, never prints a key.
    python sim/final_test/matrix/guarded_practice.py [practice_k args...]     (run from the repo root)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

STALL_SECS = 900
MAX_RESTARTS_NO_PROGRESS = 4


def journal_size(ns):
    p = C.journal_path(ns, "practice")
    return os.path.getsize(p) if os.path.exists(p) else 0


def wsl(args, env=None, **kw):
    return subprocess.Popen(["wsl", "-d", "Ubuntu-22.04", "--cd", "/mnt/d/Personal/Master's Project", "--"] + args, env=env, **kw)


def cleanup():
    subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c", "pkill -9 -f thor-Linux64; pkill -9 -f hang_guard.py; true"], timeout=60)


def main():
    argv = sys.argv[1:]
    ns = argv[argv.index("--namespace") + 1] if "--namespace" in argv else "practice-v1"
    env = C.wsl_key_env(os.environ)
    base = ["env", "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA", "DISPLAY=:0", "TRACE_ENABLED=0", "PYTHONUNBUFFERED=1", "HANG_SECS=60",
            "/home/arash/thor-env/bin/python", "sim/final_test/matrix/hang_guard.py", "sim/final_test/matrix/practice_k.py"]
    stuck = 0
    restarts = 0
    while True:
        j0 = journal_size(ns)
        cleanup()
        p = wsl(base + argv + ["--steal-lock"], env=env, stdout=sys.stdout, stderr=sys.stderr)
        last = time.time()
        j_last = j0
        rc = None
        while rc is None:
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                j = journal_size(ns)
                if j != j_last:
                    j_last, last = j, time.time()
                elif time.time() - last > STALL_SECS:
                    print("STALL_GUARD: no journal growth for %d s: killing" % STALL_SECS, flush=True)
                    p.kill()
                    cleanup()
                    rc = 75
        print("run exit code", rc, flush=True)
        if rc == 0:
            return 0
        # 75 = hang guard / stall; any other non-zero exit is restarted too (a planner transport timeout is infrastructure, not practice
        # experience: nothing was journaled for the attempt, the resume repeats it). Repeated failure without progress still stops the run.
        restarts += 1
        stuck = stuck + 1 if journal_size(ns) == j0 else 0
        print("hang restart %d (consecutive without progress: %d)" % (restarts, stuck), flush=True)
        if stuck >= MAX_RESTARTS_NO_PROGRESS:
            print("giving up: repeated hang without progress", flush=True)
            return 3


if __name__ == "__main__":
    sys.exit(main())
