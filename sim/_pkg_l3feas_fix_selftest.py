"""Offline selftest for the B9 executor fix (honest_skills._put_stale_destination). Stub skills, no simulator.
Run from the repo root:  python sim/_pkg_l3feas_fix_selftest.py

General contract (not an item replay): a put whose destination is not re-seen where it was believed to be treats the belief like a
pickup does -- LOCAL_LOOKS free local looks, then the belief is set aside and the search runs again, at most MAX_RELOCATE times.
It never applies to the agent's own put record, to a container it opened itself, to other causes, or outside the exhaustive policy.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import honest_skills as hs                                          # noqa: E402

N = [0, 0]


def check(name, cond, extra=""):
    N[1] += 1
    N[0] += bool(cond)
    if not cond:
        print("FAIL:", name, extra)


class Stub(hs.HonestSkills):
    """HonestSkills with the simulator-facing methods replaced by recorders (the class under test is the real one)."""
    def __init__(self, exhaustive=True):
        self.exhaustive = exhaustive
        self.calls, self.events, self.opened_at = [], [], {}
        self.aside = {}

        class _H:
            step = 0
        self.h = _H()
        self._step0, self._limit = 0, 10 ** 6

    def _local_look(self, k, b, st):
        self.calls.append(("look", k))

    def _set_aside(self, b, why):
        self.calls.append(("aside", b.get("track")))

    def _ev(self, msg):
        self.events.append(msg)


def soft(cause="not_detected"):
    return hs._Soft(cause, "no Plate in view")


def main():
    b = {"xyz": (0.4, 0.9, -2.4), "track": "t1", "status": "trusted", "type": "Plate", "place": "?"}
    s = Stub()
    st = {"b": b}
    r = [s._put_stale_destination("Plate", st, soft()) for _ in range(hs.LOCAL_LOOKS)]
    check("first LOCAL_LOOKS failures -> free local looks (no retry spent)", r == [True] * hs.LOCAL_LOOKS and
          [c for c in s.calls if c[0] == "look"] == [("look", i + 1) for i in range(hs.LOCAL_LOOKS)])
    check("no belief set aside during the local looks", not [c for c in s.calls if c[0] == "aside"])
    check("next failure -> belief set aside and the put re-locates", s._put_stale_destination("Plate", st, soft()) is True and ("aside", "t1") in s.calls)
    check("state reset for a fresh locate", st["b"] is None and st["approached"] is False and st["local"] == 0 and st["relocs"] == 1)
    st["b"] = dict(b, track="t2")
    n = 1
    while s._put_stale_destination("Plate", st, soft()):
        st["b"] = dict(b, track=f"t{n + 2}")
        n += 1
        if n > 50:
            break
    check("bounded by MAX_RELOCATE set-asides, then the normal retry path takes over", st["relocs"] == hs.MAX_RELOCATE and n <= 50)
    for why, kw, bb, cause in (("another cause", {}, b, "action_refused"), ("unreachable", {}, b, "unreachable"),
                               ("the agent's own put record", {}, dict(b, status="placed"), "not_detected"),
                               ("an anchored belief (container I opened)", {}, dict(b, anchor="opened-at"), "not_detected"),
                               ("no belief yet", {}, None, "not_detected"),
                               ("not the exhaustive policy", {"exhaustive": False}, b, "not_detected")):
        s2 = Stub(**kw)
        check(f"not applied: {why}", s2._put_stale_destination("Plate", {"b": bb}, soft(cause)) is False and s2.calls == [])
    s3 = Stub()
    s3.opened_at["Fridge"] = (0, 0, 0)
    check("not applied to a container the agent opened itself", s3._put_stale_destination("Fridge", {"b": b}, soft()) is False)
    s4 = Stub()
    check("works for any receptacle name (synthetic kitchen)", s4._put_stale_destination("SynthDish", {"b": dict(b, type="SynthDish")}, soft()) is True)
    print(f"{N[0]}/{N[1]}")
    return 0 if N[0] == N[1] else 1


if __name__ == "__main__":
    sys.exit(main())
