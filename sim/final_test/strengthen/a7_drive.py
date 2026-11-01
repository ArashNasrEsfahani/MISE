"""A6 driver (task A, strengthened ablation): runs every A6 step in order as chunks of at most ~10 minutes, stoppable at any boundary.

    python sim/final_test/strengthen/a7_drive.py            (repo root, Windows; relaunched by `stop_all.py resume` by name)

Lifetime: `with gpu_lock.hold("A")` (a low-priority job never starts between two chunks). Before every chunk the global STOP
(docs/final-test/strengthen/STOP) is checked: present -> exit 77 (clean stop). A chunk that exits 77 is a clean stop too. Every chunk is
resumable (journals with one fsync'd line per record, atomic JSON), so a stop at any moment loses at most the cell / unit in flight.
GPU chunks run under `gpu_lock.py run A -- ...`; the Opus screen (no GPU) and the analysis run without the lock.

Steps (A6a -> A6b -> A6c -> A6e -> analysis A6b / A6d / A6e / A6f):
  blocks         a6_blocks.py                                  (GPU: token counts)
  plan K1 G / O  a6_plan.py --planner gemma|opus --ns test-v1  (GPU for Gemma; Opus under the cumulative 25 USD cap)
  exec K1 new    a6_exec.sh test-v1 new                         (WSL simulator)
  exec K1 rerun  a6_exec.sh test-v1 rerun                       (no-exclusion replacements, the R way)
  plan K2 / K3   a6_plan.py --planner gemma --ns test-k2-v1|test-k3-v1
  exec K2/K3     a6_exec.sh <ns> new, then <ns> rerun
  exec K1 noise  a6_exec.sh test-v1 noise                       (execution-noise floor, 60 seeded units)
  analyse        a7_analyze.py
A step is done when its chunk exits 0. Exit 10 = more remains (the same step runs again). Anything else stops the driver with that code
(fail fast) and is recorded in A/RESUME.md and work/A7/drive.log.
"""
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a7_common as A6         # noqa: E402  (A7 paths under the A6 names)
import gpu_lock as GL              # noqa: E402

ROOT = A6.ROOT
LOG = os.path.join(A6.WORK, "drive.log")
STATE = os.path.join(A6.WORK, "drive_state.json")
CHUNK_S = 540
PY = sys.executable
WSL = ["wsl", "-d", "Ubuntu-22.04", "--cd", "/mnt/d/Personal/Master's Project", "--"]
RESUME_CMD = "python sim/final_test/strengthen/a7_drive.py"
S = lambda n: os.path.join("sim", "final_test", "strengthen", n)       # noqa: E731
LOCKED = [PY, S("gpu_lock.py"), "run", "A", "--"]


def steps():
    """A7: blocks (done before launch), then Gemma plans in the three kitchens, new executions, no-exclusion re-runs. Analysis is offline."""
    out = [("A7 blocks", LOCKED + [PY, S("a7_blocks.py")])]
    for ns in A6.NAMESPACES:
        out.append((f"A7 plan {A6.KITCHEN_OF[ns]} Gemma", LOCKED + [PY, S("a7_plan.py"), "--planner", "gemma", "--ns", ns, "--max-seconds", str(CHUNK_S)]))
    for ns in A6.NAMESPACES:
        out.append((f"A7 exec {A6.KITCHEN_OF[ns]} new", LOCKED + WSL + ["bash", "sim/final_test/strengthen/a7_exec.sh", ns, "new", str(CHUNK_S)]))
    for ns in A6.NAMESPACES:
        out.append((f"A7 exec {A6.KITCHEN_OF[ns]} rerun", LOCKED + WSL + ["bash", "sim/final_test/strengthen/a7_exec.sh", ns, "rerun", str(CHUNK_S)]))
    return out


def log(msg):
    os.makedirs(A6.WORK, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"done": [], "chunks": {}}


def save_state(st: dict):
    A6.A.atomic_write_json(STATE, st)


def keep_awake():
    """Task L's keep-awake script (read and run, never edited): display + system required, 1-px nudge; ends with this driver."""
    ps = os.path.join(HERE, "l_keep_awake.ps1")
    try:
        return subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps, "-Minutes", "720"],
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def main() -> int:
    if GL.stop_requested():
        log("global STOP present: not starting")
        return GL.STOPPED
    st = load_state()
    ka = keep_awake()
    code = 0
    try:
        with GL.hold("A"):
            log(f"driver start (pid {os.getpid()}); done steps: {st['done']}")
            for name, cmd in steps():
                if name in st["done"]:
                    continue
                while True:
                    if os.path.exists(A6.A6_STOP):
                        log(f"task-A stop file (work/A7/STOP_A7) before a chunk of '{name}': clean stop")
                        A6.resume(f"A6 stopped (STOP_A7) before a chunk of: {name}", [f"steps done: {', '.join(st['done']) or 'none'}"], RESUME_CMD)
                        return GL.STOPPED
                    if GL.stop_requested():
                        log(f"global STOP before a chunk of '{name}': clean stop")
                        A6.resume(f"A6 stopped (global STOP) before a chunk of: {name}", [f"steps done: {', '.join(st['done']) or 'none'}"],
                                  RESUME_CMD, ["Resume with `python sim/final_test/strengthen/stop_all.py resume` or the command above."])
                        return GL.STOPPED
                    n = st["chunks"].get(name, 0) + 1
                    st["chunks"][name] = n
                    save_state(st)
                    A6.resume(f"A6 running: {name} (chunk {n})", [f"steps done: {', '.join(st['done']) or 'none'}"], RESUME_CMD,
                              ["The driver loops <= 10-minute chunks; a stop loses at most the cell / unit in flight.",
                               "Progress: sim/final_test/strengthen/work/A7/drive.log (+ a6_exec-<ns>-<kind>.log for simulator chunks)."])
                    t0 = time.time()
                    log(f"chunk {n} of '{name}': {' '.join(os.path.basename(x) if i < 2 else x for i, x in enumerate(cmd))[:220]}")
                    rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                                                                  TRACE_ENABLED="0"))
                    log(f"chunk {n} of '{name}' rc={rc} ({time.time() - t0:.0f} s)")
                    if rc == 0:
                        st["done"].append(name)
                        save_state(st)
                        break
                    if rc == 10:
                        continue
                    if rc == GL.STOPPED:
                        log(f"'{name}' stopped cleanly (77)")
                        A6.resume(f"A6 stopped (77) during: {name}", [f"steps done: {', '.join(st['done']) or 'none'}"], RESUME_CMD)
                        return GL.STOPPED
                    code = rc
                    log(f"'{name}' FAILED rc={rc}: driver stops (fail fast)")
                    A6.resume(f"A6 STOPPED ON A FAILURE in: {name} (rc={rc})", [f"steps done: {', '.join(st['done']) or 'none'}"], RESUME_CMD,
                              ["Look at sim/final_test/strengthen/work/A7/drive.log and the step's own log before resuming."])
                    return code
            log("all A7 steps done")
            A6.resume("A7 complete", [f"steps done: {', '.join(st['done'])}"], "python sim/final_test/strengthen/a7_analyze.py   (re-analyse only)")
    finally:
        if ka is not None:
            ka.terminate()
    return code


if __name__ == "__main__":
    sys.exit(main())
