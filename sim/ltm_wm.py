"""Step 5 / A2 - WorkingMemory: the per-task event buffer (docs/step5-plan.md sections 1, 2).

A fresh WorkingMemory is created for every task and discarded afterwards; only `consolidate` (ltm_consolidate.py) can
turn its content into long-term claims. It validates every event against the closed contract, enforces a strictly
increasing `ev`, ignores identical duplicates, fails on conflicting duplicates, and refuses an episode that spans a
world reset, a frame change or a different kitchen. Missing observations stay unknown; nothing is inferred here.
"""
from __future__ import annotations

try:
    from ltm_events import EventError, validate
    from ltm_store import canon
except ImportError:                     # pragma: no cover
    from sim.ltm_events import EventError, validate
    from sim.ltm_store import canon


class WorkingMemory:
    def __init__(self, episode=None):
        self.episode = episode
        self.events: list[dict] = []
        self._by_ev: dict[int, dict] = {}
        self.kitchen = self.frame = self.map_rev = None
        self.epoch = None
        self._kinds: dict = {}          # B1: kind -> count (has() was O(n) per call; a live episode ingests thousands)

    # -- ingest --------------------------------------------------------------------------------------------
    def ingest(self, event: dict) -> dict:
        ev = validate(event)
        if self.episode is None:
            self.episode = ev["episode"]
        if ev["episode"] != self.episode:
            raise EventError(f"event of episode {ev['episode']} sent to working memory of {self.episode}")
        prior = self._by_ev.get(ev["ev"])
        if prior is not None:
            if canon(prior) != canon(ev):
                raise EventError(f"conflicting duplicate of ev {ev['ev']}")
            return prior                                        # identical duplicate: ignored
        if self.events and ev["ev"] < self.events[-1]["ev"]:
            raise EventError(f"ev {ev['ev']} arrives after ev {self.events[-1]['ev']} (strictly increasing per episode)")
        if self.kitchen is None:
            self.kitchen, self.frame, self.map_rev, self.epoch = ev["kitchen"], ev["frame"], ev["map_rev"], ev["epoch"]
        else:
            if ev["kitchen"] != self.kitchen:
                raise EventError("an episode cannot span two kitchens")
            if ev["frame"] != self.frame:
                raise EventError("an episode cannot span two coordinate frames")
            if ev["epoch"] != self.epoch:
                raise EventError("an episode cannot span a world reset (epoch changed)")
        if ev["kind"] == "episode_start" and any(e["kind"] == "episode_start" for e in self.events):
            raise EventError("second episode_start")
        if self.has("episode_end"):
            raise EventError("event after episode_end")
        # A7-review: the privileged verdict can only arrive once the agent has stopped acting (plan 1.3: "practice only,
        # after terminal"); after `terminal` nothing but the verdict and episode_end is admitted, so no agent-facing read
        # of this WM during the task can ever see it.
        if self.has("terminal") and ev["kind"] not in ("admission_verdict", "episode_end"):
            raise EventError(f"{ev['kind']} after terminal (only admission_verdict / episode_end may follow it)")
        if ev["kind"] == "admission_verdict":
            if not self.has("terminal"):
                raise EventError("admission_verdict before terminal")
            if self.mode != "practice":
                raise EventError("admission_verdict outside a practice episode")
            if self.has("admission_verdict"):
                raise EventError("second admission_verdict")
        self.events.append(ev)
        self._by_ev[ev["ev"]] = ev
        self._kinds[ev["kind"]] = self._kinds.get(ev["kind"], 0) + 1
        return ev

    @classmethod
    def replay(cls, events) -> "WorkingMemory":
        """Offline batch: sort by (episode, ev), validate, ignore identical duplicates, fail on conflicts."""
        norm = [validate(e) for e in events]
        eps = {e["episode"] for e in norm}
        if len(eps) > 1:
            raise EventError(f"replay of several episodes ({sorted(eps)}); one WorkingMemory per episode")
        wm = cls()
        for e in sorted(norm, key=lambda e: (e["episode"], e["ev"])):
            wm.ingest(e)
        return wm

    # -- views ---------------------------------------------------------------------------------------------
    def has(self, kind: str) -> bool:
        return self._kinds.get(kind, 0) > 0

    def first(self, kind: str):
        for e in self.events:
            if e["kind"] == kind:
                return e
        return None

    @property
    def mode(self):
        e = self.first("episode_start")
        return e["mode"] if e else None

    @property
    def world(self):
        e = self.first("episode_start")
        return e["world"] if e else None

    @property
    def level(self):
        e = self.first("task_context")
        return e.get("level") if e else None

    @property
    def lang(self):
        e = self.first("task_context")
        return e.get("lang") if e else None

    @property
    def terminal(self):
        return self.first("terminal")

    @property
    def admission(self):
        return self.first("admission_verdict")

    @property
    def complete(self) -> bool:
        return self.has("episode_start") and self.has("terminal") and self.has("episode_end")

    @property
    def invalid(self) -> bool:
        t = self.terminal
        return bool(t and t.get("invalid"))

    def targets(self) -> set:
        """Types this task is about: step targets and the types an inspection searched for (R1 'plus targets')."""
        out = set()
        for e in self.events:
            if e["kind"] == "step_start" and e.get("otype"):
                out.add(e["otype"])
            elif e["kind"] == "inspection":
                out.update(e.get("targets") or [])
        out.discard("?")
        return out
