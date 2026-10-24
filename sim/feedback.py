"""Environment feedback — the policy layer that drives the Phase-5 distillation funnel (passive + active).

:mod:`memory` is the *storage* (the funnel's plumbing: change-gated spatial upserts, failure-signature counters,
episode transcript). This module is the *policy* that decides WHAT to feed it and WHEN, wrapping each decision so a
memory hiccup can never break planning or the worker loop.

Two feedback modes, matching the thesis's self-improving loop:

  - **PASSIVE** (:meth:`EnvFeedback.log_step`): every executed step is recorded. The RAW episode is always appended
    (audit/eval). Failures dedup into a signature counter. On success, the resulting scene observations are pushed
    through the change-gated spatial writer — so an unchanged kitchen reinforces confidence instead of piling up rows.
  - **ACTIVE** (:meth:`EnvFeedback.needs_exploration` + :meth:`parse_needed_types`): before planning, the object types
    a command references but memory can't confidently locate become *misses* → the caller goes and looks for them,
    enriching spatial memory before the planner ever sees the prompt.

Pure stdlib + ``import memory``. Deliberately **no** ``ai2thor`` import at module top (this file is unit-testable
without the simulator). Every public method is exception-safe: it degrades to a small status dict and NEVER raises.
"""
from __future__ import annotations

import contextlib
import re

import memory as _memory  # sibling module (sim/); pure-stdlib storage layer — no simulator dependency

# ---------------------------------------------------------------------- observability sidecar (optional)
# See knowledge.py for the TRACE_API_VERSION guard rationale (`trace` is also a stdlib module name).
try:                                    # pragma: no cover - environment dependent
    import trace as _trace              # sim/trace.py — the observability plane
    if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
        _trace = None                   # we resolved the STDLIB trace module: no instrumentation
except Exception:
    _trace = None


def _t(name, *a, **kw):
    """Call ``trace.<name>(*a, **kw)``, swallowing EVERYTHING (the worker thread must never die from a hook).

    Named by string and resolved lazily (see knowledge._t): the attribute must not be dereferenced at the call
    site, or a missing sidecar would raise before the guard below.
    """
    if _trace is None:
        return None
    try:
        fn = getattr(_trace, name, None)
        return None if fn is None else fn(*a, **kw)
    except Exception:
        return None


def _art(store, table_or_kind, pointer, store_path=None) -> str:
    """The PROV node id of one stored entry — always through ``trace.artifact_id`` so it matches the id the
    corresponding ``record_write`` minted, namespace and escaping included. ``''`` without the sidecar."""
    out = _t("artifact_id", store=store, table_or_kind=table_or_kind, pointer=pointer, store_path=store_path)
    return ("art:" + out) if out else ""


def _span(name, **kw):
    """``trace.span`` if the sidecar is live, else a null context. Never raises on entry.

    The span matters more than the events it carries: with it open, every memory write underneath
    (``log_episode`` / ``record_failure`` / ``observe_location``) resolves its causal parent from the ambient
    context automatically — no signature of :mod:`memory` has to change.
    """
    if _trace is None:
        return contextlib.nullcontext(None)
    try:
        return _trace.span(name, **kw)
    except Exception:
        return contextlib.nullcontext(None)


class EnvFeedback:
    """Policy over a :class:`memory.Memory`: turns executed steps and scene observations into distilled memory writes,
    and surfaces the object types that still need active exploration. ``enabled`` is the Phase-5 ablation toggle."""

    def __init__(self, memory, enabled: bool = True):
        self.memory = memory
        self.enabled = bool(enabled)
        # Session loop-guard: (scene, object_type) pairs we have already actively explored this run, so a persistent
        # miss (object genuinely absent / undetectable) doesn't retrigger exploration on every command.
        self._explored: set = set()

    # ----------------------------------------------------------------- PASSIVE: record one executed step
    def log_step(self, scene, command, level, action, args, success, error, before, after,
                 source: str = "environment", observations=None, *,
                 run_id=None, step_ix=None, span_id=None) -> dict:
        """Record one executed step through the funnel. Exception-safe (never raises).

        - RAW: always append the episode (:meth:`memory.Memory.log_episode`) — the audit/eval transcript.
        - FAILURE: on ``not success`` dedup the error into a signature counter (:meth:`memory.Memory.record_failure`).
        - DISTILLED spatial: on success WITH ``observations`` (list of ``{object_type, object_id, receptacle, pos,
          state}`` from ``ag.spatial_snapshot()``), push them through the change-gated writer via
          :meth:`observe_scene` (track ``'oracle'``) so unchanged objects only reinforce confidence.

        ``before``/``after`` are opaque state snapshots stored on the episode (the caller may diff them). Returns
        ``{'episode_id', 'failure', 'spatial_writes'}`` (or ``{'skipped': True}`` when disabled / ``{'error': ...}``
        if anything went wrong).

        ``run_id`` / ``step_ix`` / ``span_id`` are OPTIONAL observability context (keyword-only, defaulted, so
        every existing caller is unaffected). They let the webapp's worker thread re-attach the run that was
        opened on the plan thread; without them the writes below are still recorded, just as orphans.
        """
        if not self.enabled:
            return {"skipped": True}
        try:
            with _span(f"log_step {action}", kind="FEEDBACK", op="feedback", level="DEBUG",
                       scene=scene, actor="env:ai2thor", run_id=run_id, parent_span_id=span_id,
                       step_ix=step_ix,
                       input={"command": command, "level": level, "action": action, "args": args,
                              "success": bool(success), "error": error, "step_ix": step_ix},
                       attrs={"agent.level": level, "thor.action": action, "thor.scene": scene,
                              "thor.success": bool(success), "thor.error": error}):
                episode_id = self.memory.log_episode(
                    scene=scene, command=command, level=level, action=action, args=args,
                    success=success, error=error, state_before=before, state_after=after, source=source)
                failure = None
                if not success:
                    failure = self.memory.record_failure(
                        action, error, object_type=self._obj_of(args), scene=scene, command=command)
                    # The episode -> failure-signature edge is free HERE (both ids are in scope) and
                    # unrecoverable anywhere else: `failures` rows carry no pointer to the step that caused them.
                    if episode_id and isinstance(failure, dict) and failure.get("signature"):
                        # Both ids go through trace.artifact_id: an id spelled by hand here would miss the
                        # store namespace that keeps two different memory.db files' row #1 apart, and the edge
                        # would then point at a node no write ever created.
                        mem_path = getattr(self.memory, "path", None)
                        _t("link", _art("mem", "failures", failure["signature"], mem_path),
                           _art("mem", "episodes", episode_id, mem_path), "wasDerivedFrom",
                           note=f"failing step: {action}")
                writes = {"insert": 0, "reinforce": 0, "skip": 0}
                if success and observations:
                    writes = self.observe_scene(scene, observations, track="oracle", source=source)
                return {"episode_id": episode_id, "failure": failure, "spatial_writes": writes}
        except Exception as e:  # a memory failure must NEVER break the worker/planner
            return {"error": str(e)}

    # ----------------------------------------------------------------- PASSIVE: bulk change-gated spatial upsert
    def observe_scene(self, scene, observations, track: str = "perceived", source: str = "environment", *,
                      run_id=None, span_id=None, step_ix=None) -> dict:
        """Push a batch of scene observations through the change-gated spatial writer; tally the funnel outcome.

        Each ``obs`` is ``{object_type, object_id, receptacle, pos, state}`` (extra keys ignored). Every one is fed to
        :meth:`memory.Memory.observe_location`, which decides insert vs. reinforce. Returns
        ``{'insert': n, 'reinforce': n, 'skip': n}`` (``skip`` = malformed obs or a write that raised). Never raises.

        ``run_id`` / ``span_id`` / ``step_ix`` are OPTIONAL observability context (keyword-only, defaulted).
        ONE summary event is emitted for the whole batch — a 40-object kitchen would otherwise produce 40
        near-identical events per step; the per-observation insert/reinforce writes are already recorded
        individually by :meth:`memory.Memory.observe_location`.
        """
        tally = {"insert": 0, "reinforce": 0, "skip": 0}
        if not self.enabled:
            return tally
        with _span(f"observe_scene {scene}", kind="PERCEPTION", op="observe", level="DEBUG",
                   scene=scene, actor=("env:ai2thor" if track == "oracle" else "system:perception"),
                   run_id=run_id, parent_span_id=span_id, step_ix=step_ix,
                   input={"scene": scene, "track": track, "source": source,
                          "n_observations": len(observations or [])}):
            for obs in (observations or []):
                try:
                    if not isinstance(obs, dict):
                        tally["skip"] += 1
                        continue
                    ot = obs.get("object_type") or obs.get("objectType")
                    if not ot:
                        tally["skip"] += 1
                        continue
                    res = self.memory.observe_location(
                        scene, ot,
                        receptacle=obs.get("receptacle"),
                        pos=obs.get("pos"),
                        state=obs.get("state"),
                        track=track, source=source,
                        object_id=obs.get("object_id") or obs.get("objectId"),
                        why="passive scene observation")
                    act = res.get("action")
                    tally[act if act in tally else "skip"] += 1
                except Exception:
                    tally["skip"] += 1
            _t("emit", "memory.write", level="DEBUG", scene=scene,
               actor=("env:ai2thor" if track == "oracle" else "system:perception"),
               summary=f"observe_scene {scene}: {tally['insert']} insert / {tally['reinforce']} reinforce "
                       f"/ {tally['skip']} skip",
               payload={"op": "observe_scene", "scene": scene, "track": track, "source": source,
                        "tally": dict(tally), "n_observations": len(observations or [])})
        return tally

    # ----------------------------------------------------------------- ACTIVE: what still needs exploring
    def needs_exploration(self, scene, needed_types) -> list:
        """Object types the command needs but memory can't confidently locate, minus those already explored this run.

        Delegates miss detection to :meth:`memory.Memory.query` (a type is a miss if it has no belief or a decayed
        confidence < 0.1) and subtracts this session's ``_explored`` loop-guard set. Never raises (→ ``[]`` on error).
        """
        try:
            if not needed_types:
                return []
            q = self.memory.query(scene, " ".join(str(t) for t in needed_types), needed_types=list(needed_types),
                                  with_text=False)  # miss-check only — skip the rag/Ollama round-trip (runs per command)
            return [t for t in q.get("misses", []) if (scene, t) not in self._explored]
        except Exception:
            return []

    def mark_explored(self, scene, object_type) -> None:
        """Record that ``(scene, object_type)`` was actively explored this session (loop guard). Never raises."""
        try:
            self._explored.add((scene, object_type))
        except Exception:
            pass

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def parse_needed_types(command, known_types) -> list:
        """Object types from ``known_types`` (the live scene's types) referenced by ``command``.

        Case-insensitive matching with light singularization and camelCase splitting: a type like ``CounterTop`` is
        matched against "counter top" as well as "countertop", and a plural in the command ("apples") matches the
        singular type ("Apple"). Returns the matched ``known_types`` (original casing), de-duplicated, in input order.
        """
        cmd = str(command or "").lower()
        if not cmd:
            return []
        tokens = set(re.findall(r"[a-z]+", cmd))
        out = []
        for t in (known_types or []):
            if not t:
                continue
            tl = str(t).lower()
            spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(t)).lower()  # "CounterTop" -> "counter top"
            forms = {tl, spaced, tl.rstrip("s"), tl + "s"}
            hit = any(f and f in cmd for f in forms)
            if not hit:  # token-level singular/plural reconciliation
                for tok in tokens:
                    if tok == tl or tok.rstrip("s") == tl or tok == tl + "s":
                        hit = True
                        break
            if hit and t not in out:
                out.append(t)
        return out

    @staticmethod
    def _obj_of(args):
        """Best-effort object type/name from a step-args dict (``object``/``objectType``/``objectId``/``name``) or None.

        An ``objectId`` like ``"Mug|+01.2|-00.3"`` is reduced to its type prefix (``"Mug"``) so failure signatures
        bucket by type, not by physical id. Non-dict / empty args yield ``None``.
        """
        if not isinstance(args, dict):
            return None
        for key in ("objectType", "object_type", "object", "name"):
            v = args.get(key)
            if v:
                return str(v).split("|")[0]
        oid = args.get("objectId") or args.get("object_id")
        if oid:
            return str(oid).split("|")[0]
        return None


if __name__ == "__main__":
    # Self-test: runs WITHOUT the simulator (temp Memory, pure sqlite). Exercises the funnel through the policy layer:
    # (a) unchanged obs reinforces (no new row), (b) changed obs inserts, (c) failure counter, (d) exploration misses.
    import os
    import tempfile

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "fb_selftest.db")
    mem = _memory.Memory(path=db)
    fb = EnvFeedback(mem, enabled=True)

    scene = "FloorPlan1"

    # (a) unchanged observation -> reinforce, no new spatial row.
    obs0 = [{"object_type": "Mug", "object_id": "Mug|1", "receptacle": "Sink",
             "pos": {"x": 1.0, "y": 0.9, "z": 2.0}, "state": None}]
    t1 = fb.observe_scene(scene, obs0, track="oracle")
    assert t1 == {"insert": 1, "reinforce": 0, "skip": 0}, f"first observe should insert once, got {t1!r}"
    rows_after_insert = mem.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Mug'").fetchone()[0]
    t2 = fb.observe_scene(scene, obs0, track="oracle")
    assert t2 == {"insert": 0, "reinforce": 1, "skip": 0}, f"unchanged observe should reinforce, got {t2!r}"
    rows_after_reinforce = mem.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Mug'").fetchone()[0]
    assert rows_after_reinforce == rows_after_insert, "reinforce must not add a spatial row"
    print("(a) unchanged obs -> reinforce, rows stayed", rows_after_reinforce)

    # (b) changed observation (moved to Cabinet) -> insert a new belief row.
    obs1 = [{"object_type": "Mug", "object_id": "Mug|1", "receptacle": "Cabinet",
             "pos": {"x": 4.0, "y": 0.9, "z": 5.0}, "state": None}]
    t3 = fb.observe_scene(scene, obs1, track="oracle")
    assert t3 == {"insert": 1, "reinforce": 0, "skip": 0}, f"changed observe should insert, got {t3!r}"
    rows_after_change = mem.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Mug'").fetchone()[0]
    assert rows_after_change == rows_after_insert + 1, "changed observe should add exactly one row"
    print("(b) changed obs -> insert, rows now", rows_after_change)

    # (c) 5 identical failures via log_step -> a single failures row with occurrences=5.
    for _ in range(5):
        r = fb.log_step(scene, command="pick up the apple", level="L1", action="PickupObject",
                        args={"objectId": "Apple|+01.2|-00.3"}, success=False,
                        error="PickupObject failed: Apple|+01.2 not visible", before=None, after=None)
        assert "error" not in r, f"log_step should not error, got {r!r}"
    fstats = mem.failure_stats()
    assert fstats == {"rows": 1, "occurrences": 5}, f"5 identical failures -> 1 row occ=5, got {fstats!r}"
    assert r["failure"]["occurrences"] == 5, f"log_step should surface the counter, got {r!r}"
    print("(c) 5 identical failures ->", fstats)

    # (d) needs_exploration reports an unseen type; a known one is not a miss.
    known = ["Mug", "Spatula", "Apple"]
    needed = EnvFeedback.parse_needed_types("bring me the spatula and the mug", known)
    assert set(needed) == {"Spatula", "Mug"}, f"parse_needed_types mismatch: {needed!r}"
    misses = fb.needs_exploration(scene, needed)
    assert "Spatula" in misses, f"Spatula was never observed -> should be a miss, got {misses!r}"
    assert "Mug" not in misses, f"Mug is known -> not a miss, got {misses!r}"
    fb.mark_explored(scene, "Spatula")
    assert "Spatula" not in fb.needs_exploration(scene, needed), "mark_explored should suppress re-exploration"
    print("(d) needs_exploration ->", misses, "| after mark_explored, suppressed")

    # _obj_of sanity.
    assert EnvFeedback._obj_of({"objectId": "Mug|1|2"}) == "Mug"
    assert EnvFeedback._obj_of({"objectType": "Apple"}) == "Apple"
    assert EnvFeedback._obj_of(None) is None

    mem.close()
    for p in (db, os.path.splitext(db)[0] + "_audit.log"):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("FEEDBACK SELFTEST OK")
