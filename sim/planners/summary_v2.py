"""ft-v2 observation-summary compaction, COMMON to every arm (final test, stage H-B1; PACKAGE-SPEC section 4.5).

Why: in the dev data the planner's "what the robot has seen" list was cut at 40 lines in 210 / 210 prompts, so the
planner often never saw the object it needed. This is a PURE function of the summary lines the honest agent already
produced (``objmem.ObjectMemory.summary(limit=None)``): no simulator read, no new information, only a denser rendering.

Rule (deterministic):
  1. drop ``Floor`` lines;
  2. several sightings of one FIXTURE type become one line  ``<Type> x<n> [<k> closed, <m> open, <j> ?]; <where>``
     (a single sighting keeps its own line, which carries more detail);
  3. one line per type before any type gets a second (round-robin over types, A-Z inside a round);
  4. the kept lines are printed A-Z by type (stable inside a type: newest step first);
  5. cut at ``limit`` (default 40) -- the cut happens in round-robin order, so a late type is never lost to a type
     that has many instances.
The output is never longer than the input and never longer than ``limit``.

``where`` of a group line: with ``agent_xz`` given, ``nearest at (x, z), (x, z)`` (the two sightings nearest the agent);
without it, ``latest at (x, z), (x, z)`` (the two most recent sightings) -- never a claim of "nearest" without a position.
The executor's own chooser keeps reading the full belief list; this function only shapes the PROMPT.
"""
from __future__ import annotations

import re
from typing import Optional

#: fixture types (PACKAGE-SPEC 5.3 ``fixture_types``): objects that do not move, so a group line is meaningful.
FIXTURE_TYPES = frozenset({"Fridge", "Microwave", "Sink", "SinkBasin", "StoveBurner", "StoveKnob", "CounterTop",
                           "Cabinet", "Drawer", "GarbageCan", "CoffeeMachine", "Toaster", "Faucet", "LightSwitch",
                           "Shelf", "ShelvingUnit", "Window"})
DEFAULT_LIMIT = 40

_LINE = re.compile(r"^(?P<type>[A-Za-z]+) \[(?P<state>[^\]]*)\] (?P<where>.*?)"
                   r"(?: at \((?P<x>-?\d+(?:\.\d+)?), (?P<z>-?\d+(?:\.\d+)?)\))?, step (?P<step>\d+)$")
_STATE_ORDER = ("closed", "open", "off", "on")


def _parse(line: str) -> dict:
    m = _LINE.match(line.strip())
    if not m:
        t = (line.strip().split(" ", 1) or [""])[0]
        return {"type": t, "state": "?", "x": None, "z": None, "step": 0, "raw": line, "ok": False}
    d = m.groupdict()
    return {"type": d["type"], "state": (d["state"] or "?").strip() or "?", "step": int(d["step"]), "raw": line,
            "x": float(d["x"]) if d["x"] is not None else None, "z": float(d["z"]) if d["z"] is not None else None,
            "ok": True}


def _state_text(rows) -> str:
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    order = [s for s in _STATE_ORDER if s in counts] + sorted(s for s in counts if s not in _STATE_ORDER and s != "?")
    if "?" in counts:
        order.append("?")
    return ", ".join(f"{counts[s]} {s}" for s in order)


def _group_line(typ: str, rows, agent_xz) -> str:
    pos = [r for r in rows if r["x"] is not None]
    if agent_xz is not None:
        ax, az = agent_xz
        pos.sort(key=lambda r: ((r["x"] - ax) ** 2 + (r["z"] - az) ** 2, r["x"], r["z"]))
        label = "nearest"
    else:
        pos.sort(key=lambda r: (-r["step"], r["x"], r["z"]))
        label = "latest"
    where = ""
    if pos:
        where = f"; {label} at " + ", ".join(f"({r['x']:.2f}, {r['z']:.2f})" for r in pos[:2])
    return f"{typ} x{len(rows)} [{_state_text(rows)}]{where}"


def compact(lines, limit: int = DEFAULT_LIMIT, agent_xz: Optional[tuple] = None) -> list:
    """-> a list of at most ``limit`` summary lines (see the module docstring)."""
    rows = [_parse(l) for l in (lines or [])]
    rows = [r for r in rows if r["type"] != "Floor"]
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    cand = []                                   # (rank, type, text)
    for typ in sorted(by_type):
        rs = sorted(by_type[typ], key=lambda r: (-r["step"], r["x"] if r["x"] is not None else 0.0,
                                                 r["z"] if r["z"] is not None else 0.0, r["raw"]))
        if typ in FIXTURE_TYPES and len(rs) >= 2:
            cand.append((0, typ, _group_line(typ, rs, agent_xz)))
        else:
            for k, r in enumerate(rs):
                cand.append((k, typ, r["raw"].strip()))
    cand.sort(key=lambda c: (c[0], c[1]))       # stable: inside a (rank, type) the newest step stays first
    keep = cand[:max(0, int(limit))]
    keep.sort(key=lambda c: (c[1], c[0]))
    return [c[2] for c in keep]
