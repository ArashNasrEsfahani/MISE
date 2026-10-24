"""CACHED-PLANS-ONLY GUARD for the execute phase (plan first, then simulate; infrastructure only, changes no result).

Phase C must read plans from the journal and never plan. Removing the Anthropic key is not enough (Codex account credentials, Ollama on the
Windows host and other credential sources exist), so this wrapper makes every route a planner could take fail loudly, and counts attempts:

    python sim/final_test/matrix/execute_guard.py execute.py --namespace NS ...        (WSL; or any python for the selftest)

  * `planners.base.PlannerProvider.plan`, `http_post_json`, `http_get_json` raise `GuardViolation` (no provider can transmit);
  * `urllib.request.urlopen` and `socket.socket.connect / connect_ex` refuse every address that is not loopback (the simulator talks to
    127.0.0.1 and a unix X socket, both stay allowed);
  * `subprocess.Popen` refuses a command whose program is `codex` or `ollama` (or a shell line naming them);
  * the planner credentials (THESIS_ANTHROPIC_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY) are removed from `os.environ`.

At exit the wrapper writes {attempts blocked, loopback connects allowed, planner modules imported} to $EXEC_GUARD_REPORT (default: next to the
journals is NOT assumed; the caller names the file). A run with `blocked == 0` is the evidence that execution did not plan.
"""
from __future__ import annotations

import atexit
import json
import os
import runpy
import socket
import subprocess
import sys
import urllib.request

LOOPBACK = ("127.", "::1", "localhost", "0.0.0.0", "::")
KEY_VARS = ("THESIS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_ENV_FILE")
BLOCKED_PROGRAMS = ("codex", "ollama")

state = {"blocked": [], "loopback_connects": 0, "installed": False}


class GuardViolation(RuntimeError):
    """The execute phase tried to plan, or to reach a planner / the outside network."""


def _is_loopback(host) -> bool:
    h = str(host or "")
    return h == "" or any(h == p or h.startswith(p) for p in LOOPBACK)


def _block(kind: str, what: str):
    state["blocked"].append({"kind": kind, "what": str(what)[:120]})
    raise GuardViolation(f"execute phase may not plan or reach a planner ({kind}: {what})")


def install(env=None) -> dict:
    """Idempotent. Returns the shared state dict (tests read `blocked`)."""
    if state["installed"]:
        return state
    state["installed"] = True
    e = os.environ if env is None else env
    for k in KEY_VARS:
        e.pop(k, None)
    sim = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if sim not in sys.path:
        sys.path.insert(0, sim)
    try:
        from planners import base as pb
        pb.PlannerProvider.plan = lambda self, *a, **k: _block("provider.plan", type(self).__name__)
        pb.http_post_json = lambda url, *a, **k: _block("http_post_json", url)
        pb.http_get_json = lambda url, *a, **k: _block("http_get_json", url)
    except ImportError:
        pass                                                         # the planners package is not even importable: nothing to block

    orig_urlopen = urllib.request.urlopen

    def urlopen(req, *a, **k):
        url = getattr(req, "full_url", req)
        host = urllib.request.urlparse(url).hostname if hasattr(urllib.request, "urlparse") else None
        if host is None:
            from urllib.parse import urlparse
            host = urlparse(str(url)).hostname
        if not _is_loopback(host):
            _block("urlopen", host)
        return orig_urlopen(req, *a, **k)
    urllib.request.urlopen = urlopen

    orig_connect, orig_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _check(self, address):
        if self.family == getattr(socket, "AF_UNIX", -1):
            return
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback(host):
            _block("socket.connect", host)
        state["loopback_connects"] += 1

    def connect(self, address):
        _check(self, address)
        return orig_connect(self, address)

    def connect_ex(self, address):
        _check(self, address)
        return orig_connect_ex(self, address)
    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex

    orig_init = subprocess.Popen.__init__

    def popen_init(self, args, *a, **k):
        prog = args if isinstance(args, str) else (args[0] if args else "")
        low = os.path.basename(str(prog)).lower()
        line = (" ".join(map(str, args)) if not isinstance(args, str) else args).lower()
        if any(low.startswith(b) for b in BLOCKED_PROGRAMS) or (k.get("shell") and any(b in line for b in BLOCKED_PROGRAMS)):
            _block("subprocess", low)
        return orig_init(self, args, *a, **k)
    subprocess.Popen.__init__ = popen_init
    return state


def report() -> dict:
    mods = sorted(m for m in sys.modules if m.startswith("planners") or m in ("anthropic",))
    return {"blocked": len(state["blocked"]), "blocked_detail": state["blocked"][:20], "loopback_connects": state["loopback_connects"],
            "planner_modules_imported": mods, "pid": os.getpid()}


def _write_report():
    """To $EXEC_GUARD_REPORT, else to work/exec_guard.<pid>.json next to this file (the launcher forwards no custom variable into WSL)."""
    p = os.environ.get("EXEC_GUARD_REPORT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", f"exec_guard.{os.getpid()}.json")
    if p:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report(), f, indent=1)
        except OSError:
            pass


if __name__ == "__main__":
    install()
    atexit.register(_write_report)
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")
