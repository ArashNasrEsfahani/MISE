"""Reader of the rescue passes. Read only.

Rescue: every unique scored unit (run 0) whose latest state was still a restoration failure or a crash was executed once more with the
same cached plan from the same restored state, on a fresh simulator boot, with the unchanged reset gate, for up to 3 passes or until a
pass recovered nothing. One unit serves every condition that shares it. Each kitchen has one record set
(docs/final-test/strengthen/N/<ns>/rescue/); every row carries `rescue_pass`, and a pass is a complete fresh history (attempts 0-2).
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DOCS = os.path.join(ROOT, "docs", "final-test", "strengthen", "N")
INPUTS = os.path.join(HERE, "inputs")
MAX_RESCUE_PASSES = 3
GOOD = ("done", "not_executed")
BAD = ("blocked_state", "crash")


def read_jsonl(path: str) -> list:
    """Tolerates a torn LAST line; anything else unparsable is an error."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln.decode("utf-8")))
        except ValueError:
            if i == len(lines) - 1:
                continue
            raise RuntimeError(f"{path}: unparsable line {i + 1}")
    return out


def final_history(rows) -> bool:
    if not rows:
        return False
    last = rows[-1]
    return last.get("status") in ("done", "not_executed") or int(last.get("attempt", 0)) >= 2


def pass_paths(ns: str) -> dict:
    d = os.path.join(DOCS, ns, "rescue")
    return {"dir": d, "exec": os.path.join(d, "exec.jsonl"), "outcome": os.path.join(d, "outcome.jsonl"),
            "events": os.path.join(INPUTS, "starts", f"N-rescue.{ns}.jsonl")}


class RescueSource:
    """The rescue records as a re-run source: per exec key, the history of the LATEST pass that is final (<= upto, when given)."""

    def __init__(self, ns: str, upto: int | None = None):
        p = pass_paths(ns)
        self.name, self.exec_path = "N-rescue", p["exec"]
        self.passes, self.hist, self.outc, self.starts, self.epoch, self.pass_of = {}, {}, {}, {}, {}, {}
        for r in (read_jsonl(p["exec"]) if os.path.exists(p["exec"]) else []):
            self.passes.setdefault(r.get("key"), {}).setdefault(int(r["rescue_pass"]), []).append(r)
        outs = {(r.get("key"), int(r["rescue_pass"])): r for r in (read_jsonl(p["outcome"]) if os.path.exists(p["outcome"]) else [])}
        starts = {}
        for e in (read_jsonl(p["events"]) if os.path.exists(p["events"]) else []):
            if e.get("event") == "exec_attempt_start":
                starts.setdefault((e.get("key"), int(e["rescue_pass"])), []).append(e)
        for k, by in self.passes.items():
            ok = [q for q, rows in by.items() if final_history(rows) and (upto is None or q <= upto)]
            if ok:
                q = max(ok)
                self.pass_of[k] = q
                self.hist[k] = by[q]
                if (k, q) in outs:
                    self.outc[k] = outs[(k, q)]
                self.starts[k] = starts.get((k, q), [])
