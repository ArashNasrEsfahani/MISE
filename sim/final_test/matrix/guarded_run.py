"""Generic Windows-side launcher: run a matrix module (capture.py / execute.py ...) in WSL under the 60 s detector hang guard, with the study key
forwarded through WSLENV (never printed); restart (resume) after a hang (exit 75) or a transport/other non-zero exit, at most 3 times.
    python sim/final_test/matrix/guarded_run.py capture.py --namespace rehearsal-v1 --items dev ...
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402


def main():
    module, argv = sys.argv[1], sys.argv[1:][1:]
    env = C.wsl_key_env(os.environ)
    cmd = ["wsl", "-d", "Ubuntu-22.04", "--cd", "/mnt/d/Personal/Master's Project", "--", "env", "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA", "DISPLAY=:0",
           "TRACE_ENABLED=0", "PYTHONUNBUFFERED=1", "HANG_SECS=60", "/home/arash/thor-env/bin/python", "sim/final_test/matrix/hang_guard.py",
           "sim/final_test/matrix/" + module] + argv + ["--steal-lock"]
    for attempt in range(4):
        subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c", "pkill -9 -f thor-Linux64; pkill -9 -f hang_guard.py; true"], timeout=60)
        rc = subprocess.run(cmd, env=env).returncode
        print("exit code", rc, "attempt", attempt + 1, flush=True)
        if rc == 0:
            return 0
        if rc not in (75, 1):
            return rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
