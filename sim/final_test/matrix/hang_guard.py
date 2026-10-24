"""HANG GUARD for live runs (infrastructure only; changes no result).

Wraps the innermost detector call (perception.Detector.detect). If one call is stuck for more than HANG_SECS seconds (the
2026-10-11 failure: a CUDA host-to-device copy that never returned) the process logs one line and exits with code 75. The caller
(guarded_practice.py) kills leftover simulator processes and restarts; the runner is resumable per episode, and a planner or
detector infrastructure failure is never experience, so the interrupted attempt is simply repeated.

    python sim/final_test/matrix/hang_guard.py <module.py> [args...]      (runs the module's main() under the guard)
"""
from __future__ import annotations

import os
import runpy
import sys
import threading
import time

HANG_SECS = float(os.environ.get("HANG_SECS", "60"))
EXIT_HANG = 75
_state = {"t0": None, "n": 0}


def _watch():
    while True:
        time.sleep(3)
        t0 = _state["t0"]
        if t0 is not None and time.time() - t0 > HANG_SECS:
            sys.stderr.write("HANG_GUARD: detector call stuck for %.0f s (call #%d): aborting the episode as an infra failure\n"
                             % (time.time() - t0, _state["n"]))
            sys.stderr.flush()
            os._exit(EXIT_HANG)


def install():
    here = os.path.dirname(os.path.abspath(__file__))
    sim = os.path.abspath(os.path.join(here, "..", ".."))
    if sim not in sys.path:
        sys.path.insert(0, sim)
    import perception
    orig = perception.Detector.detect

    def guarded(self, *a, **k):
        _state["t0"] = time.time()
        _state["n"] += 1
        try:
            return orig(self, *a, **k)
        finally:
            _state["t0"] = None

    perception.Detector.detect = guarded
    threading.Thread(target=_watch, daemon=True).start()


if __name__ == "__main__":
    install()
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")
