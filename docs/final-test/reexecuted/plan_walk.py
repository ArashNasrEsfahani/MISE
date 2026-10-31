"""A symbolic walk of a plan under the executor's action semantics, and the two executor preconditions it checks. No simulator, no
model, no I/O except the frozen read-only inputs.

Copied unchanged from the task-M module (m_variants) of the follow-up analyses: the World class, the precondition check (check_plan)
and the helpers they use. Its rules are built from the executor's action semantics (sim/planners/exec_contract.json, honest_skills).
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (os.path.join(ROOT, "sim", "final_test", "matrix"), os.path.join(ROOT, "sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: executor states a step can make true (exec_contract tools -> the state they set); anything else (cold, fresh ...) is not checkable
EXEC_STATES = ("open", "closed", "on", "off", "sliced", "broken", "clean", "dirty", "filled", "empty", "hot", "cooked")
OPENABLE = ("Fridge", "Cabinet", "Drawer", "Microwave")
SINK_ALIAS = {"Sink": "SinkBasin"}
KNIFE = "ButterKnife"


@functools.lru_cache(maxsize=None)
def _fixture_names() -> dict:
    import ltm_package as LP
    return dict(LP._RENDER_FIXTURE)


def _place(t: str) -> str:
    return SINK_ALIAS.get(t, t)


_SUM_RX = re.compile(r"^([A-Za-z]+) \[([^\]]*)\]")


def observed_states(summary_lines) -> dict:
    """type -> [known executor states] from the post-sweep observation summary ('Microwave [open] on/in ...'); '?' = unknown, skipped."""
    out = {}
    for ln in summary_lines or ():
        m = _SUM_RX.match(str(ln))
        if not m:
            continue
        for s in (x.strip() for x in m.group(2).split(",")):
            if s and not s.endswith("?") and s in EXEC_STATES and s not in out.setdefault(m.group(1), []):
                out[m.group(1)].append(s)
    return out


def _name(t: str) -> str:
    return _fixture_names().get(t, t)


def pred_text(p) -> str:
    kind, t, a = p[0], p[1], p[2]
    if kind == "in":
        return f"{_name(t)} on/in {_name(a)}"
    if kind == "held":
        return f"{_name(t)} in your hand"
    return f"{_name(t)} {a}"


#: executor state -> (property, value)
PROP = {"open": ("open", True), "closed": ("open", False), "on": ("on", True), "off": ("on", False), "filled": ("filled", True),
        "empty": ("filled", False), "clean": ("clean", True), "dirty": ("clean", False), "sliced": ("sliced", True),
        "broken": ("broken", True), "hot": ("hot", True), "cooked": ("cooked", True)}
VALUE_STATE = {v: k for k, v in PROP.items()}


class World:
    """A symbolic walk of a plan under the executor's semantics (exec_contract.json + honest_skills): the hand, object locations and
    object properties. A container a tool opens by itself is closed again by the executor's tidy step (honest_skills._tidy_step_end; a
    Microwave before a later toggle_on too, TIDY_SWITCH_CLOSED): a container a step uses implicitly ends closed unless it was already
    open (seen / stated / an explicit `open`); `close` / `close_all` close it. heat / cook run the whole Microwave cycle and leave it
    closed and off; wash switches the Faucet on and off itself (it ends off)."""

    def __init__(self, holding=None, seen=None, told=()):
        self.hand = holding or None
        self.loc, self.prop, self.why = {}, {}, {}
        for t, ss in (seen or {}).items():
            for st in ss:
                if st in PROP:
                    self.prop[(t, PROP[st][0])] = PROP[st][1]
                    self.why[(t, PROP[st][0])] = "it is seen " + st
        for t, st, _w in told:
            if st in PROP and t != "*" and (t, PROP[st][0]) not in self.prop:
                self.prop[(t, PROP[st][0])] = PROP[st][1]
                self.why[(t, PROP[st][0])] = "the user says it is " + st

    def set(self, t, state, why=None):
        k = (t, PROP[state][0])
        self.prop[k] = PROP[state][1]
        if why:
            self.why[k] = why

    def _tidy(self, r):
        if r in OPENABLE and self.prop.get((r, "open")) is not True:
            self.prop[(r, "open")] = False

    def step(self, i: int, st):
        st = as_step(st)                                                # list steps (planner output) or dict steps (KB routes)
        a, o, to = st.get("action"), st.get("object"), st.get("to")
        if a in ("pick_and_place",) and to:
            self._tidy(to)
        elif a == "put_in" and o:
            self._tidy(o)
        if a == "pickup" and o:
            self.hand = o
            self.loc.pop(o, None)
        elif a == "pick_and_place" and o and to:
            self.loc[o] = _place(to)
            if self.hand == o:
                self.hand = None
        elif a == "put_in" and o:
            if self.hand:
                self.loc[self.hand] = _place(o)
            self.hand = None
        elif a == "drop":
            if self.hand:
                self.loc[self.hand] = "(dropped)"
            self.hand = None
        elif a == "wash" and o:
            self.set(o, "clean")
            self.loc[o] = "SinkBasin"
            self.hand = None
            self.set("Faucet", "off")                    # wash switches the Faucet on and off itself (the KB faucet-run / wash-dish lines)
        elif a in ("heat", "cook") and o:
            self.set(o, "hot")
            if a == "cook":
                self.set(o, "cooked")
            self.hand = o
            self.loc.pop(o, None)
            self.set("Microwave", "off")                 # the whole cycle: in, door closed, on, out again; the door the executor opened is
            self.set("Microwave", "closed")              # closed again by its tidy step
        elif a == "open" and o:
            self.set(o, "open", f"step {i} opens it")
        elif a == "close" and o:
            self.set(o, "closed")
        elif a == "close_all":
            for k in [k for k, v in self.prop.items() if k[1] == "open" and v is True]:
                self.prop[k] = False
        elif a in ("toggle_on", "toggle_off") and o:
            st_ = "on" if a == "toggle_on" else "off"
            self.set(o, st_)
            if o == "StoveKnob":
                self.set("StoveBurner", st_)
        elif a in ("slice", "break", "dirty", "clean", "fill", "empty") and o:
            self.set(o, {"slice": "sliced", "break": "broken", "dirty": "dirty", "clean": "clean", "fill": "filled", "empty": "empty"}[a])

    def holds(self, p):
        """True / False, or None when the predicate is not an executor state (not checked)."""
        kind, t, a = p[0], p[1], p[2]
        if kind == "in":
            return self.loc.get(t) == _place(a)
        if kind == "held":
            return self.hand == t
        if a not in PROP or t == "*":
            return None
        return self.prop.get((t, PROP[a][0])) == PROP[a][1]

    def facts(self) -> list:
        out = [("in", o, r) for o, r in self.loc.items() if r != "(dropped)"]
        if self.hand:
            out.append(("held", self.hand, None))
        out += [("state", t, VALUE_STATE[(pr, v)]) for (t, pr), v in self.prop.items() if (pr, v) in VALUE_STATE]
        return out


def as_step(st) -> dict:
    """A plan step as a dict. The planner's accepted plans are lists [action, object, to, arg]."""
    if isinstance(st, dict):
        return st
    st = list(st) + [None] * (4 - len(st))
    return {"action": st[0], "object": st[1], "to": st[2], "arg": st[3]}


def check_plan(plan, rd: dict, holding=None, summary_lines=()) -> list:
    """-> [(code, fields)] problems (empty = the plan passes), at most 4. Walks the plan with `World` (the executor's effects):
    preconditions the executor does NOT handle itself (toggle_on Microwave with its door open: seen / stated open or opened by an explicit
    step; slice without the ButterKnife in hand), then, for a request the parser read as 'ok', every requested end-state predicate must
    hold at the end (a predicate that is not an executor state is not checked)."""
    seen = observed_states(summary_lines)
    w = World(holding, seen, rd.get("now") or [])
    probs = []
    for i, st in enumerate(plan or [], 1):
        st = as_step(st)
        a, o = st.get("action"), st.get("object")
        if a == "toggle_on" and o == "Microwave" and w.prop.get(("Microwave", "open")) is True:
            probs.append(("microwave_open", {"i": i, "why": w.why.get(("Microwave", "open"), "it is open")}))
        if a == "slice" and w.hand != KNIFE:
            probs.append(("no_knife", {"i": i, "obj": o}))
        w.step(i, st)
    if rd.get("status") == "ok":
        for p in rd["desired"]:
            if w.holds(p) is False:
                probs.append(("uncovered", {"pred": pred_text(p)}))
    out, keys = [], set()
    for c, f in probs:
        k = (c, json.dumps(f, sort_keys=True))
        if k not in keys:
            keys.add(k)
            out.append((c, f))
    return out[:4]
