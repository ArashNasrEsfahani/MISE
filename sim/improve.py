"""Loop closure -- Phase 7 of the self-improving agent: turn one executed command into knowledge.

This module OWNS the write-back decision of the thesis's self-improving loop. After a command has been
planned and executed (by the webapp worker, or by a headless Phase-8 eval harness), the caller hands the
outcome to :func:`close_loop`, which:

  1. decides SUCCESS: an applicable command-level goal check wins (confidence 0.9); otherwise all-steps-ok
     (confidence 0.6) -- any failed step or failed goal check means failure;
  2. on success, upserts the EXECUTED (post-repair) step sequence as a procedure (``scope='global'``) -- the
     plan that actually worked, not the raw LLM output;
  3. on failure, upserts a TEMPLATE lesson (failed step label + agent error + ``thor_actions.error_hint``),
     scoped to the scene; an optional injected ``llm`` callable (``reflect=True``) may write a better lesson
     but is NEVER required;
  4. best-effort re-embeds new KB entries via ``rag.HybridRAG.reindex()`` (degrades silently without Ollama);
  5. returns a LOOP REPORT dict and appends it as one JSON line to ``sim/loop_reports.jsonl`` (path
     injectable for tests) -- the auditable trace of what the loop learned and why.

Dedup lives in :meth:`knowledge.KnowledgeBase.upsert_procedure` / ``upsert_lesson`` (Phase-5 funnel style:
re-learning reinforces a counter instead of appending duplicates). All auto-writes carry
``source='environment'`` so provenance ranking keeps them below human/demo memories. When
``loop_enabled=False`` (the Phase-10 ablation axis) NOTHING is written and no reindex runs -- the report is
still returned/logged so ablation runs stay traceable. ``rag_enabled=False`` independently skips the
embed/reindex step.

UI-agnostic and exception-safe by contract: ``kb``/``memory`` are dependency-injected instances (the webapp
passes its singletons; tests pass temp-path instances), nothing here touches the simulator, module import is
pure stdlib (``thor_actions``/``rag`` are lazy-imported inside functions), and :func:`close_loop` never
raises -- a loop failure must never break planning or execution.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time

# Sibling module (sim/); pure stdlib. _error_class is knowledge's copy of memory.Memory._error_class, used
# read-only here to match a failure against memory's Phase-5 failure counters.
from knowledge import _error_class, normalize_command, plan_signature  # noqa: F401  (re-exported helpers)

# ---------------------------------------------------------------------- observability sidecar (optional)
# See knowledge.py for the TRACE_API_VERSION guard rationale (`trace` is also a stdlib module name).
try:                                    # pragma: no cover - environment dependent
    import trace as _trace              # sim/trace.py — the observability plane
    if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
        _trace = None                   # we resolved the STDLIB trace module: no instrumentation
except Exception:
    _trace = None

# The kinds `retrieved_ids` reports (KnowledgeBase collection names) -> the singular artifact kind.
_KIND_SINGULAR = {"procedures": "procedure", "lessons": "lesson", "facts": "fact",
                  "procedure": "procedure", "lesson": "lesson", "fact": "fact"}


def _t(name, *a, **kw):
    """Call ``trace.<name>(*a, **kw)``, swallowing EVERYTHING — close_loop's never-raise contract extends to
    its hooks. Named by string and resolved lazily (see knowledge._t)."""
    if _trace is None:
        return None
    try:
        fn = getattr(_trace, name, None)
        return None if fn is None else fn(*a, **kw)
    except Exception:
        return None


def _span(name, **kw):
    """``trace.span`` if the sidecar is live, else a null context. Never raises on entry."""
    if _trace is None:
        return contextlib.nullcontext(None)
    try:
        return _trace.span(name, **kw)
    except Exception:
        return contextlib.nullcontext(None)


def _current_run_id():
    """The ambient run id (contextvar -> thread binding -> process default), or None."""
    if _trace is None:
        return None
    try:
        return (_trace.current() or {}).get("run_id")
    except Exception:
        return None


def _artref(ref):
    """Artifact-id-safe form of a natural key (must match trace's own escaping)."""
    if _trace is None:
        return str(ref)
    try:
        return getattr(_trace, "_safe_ref", str)(ref)
    except Exception:
        return str(ref)


def _art(store, table_or_kind, pointer, store_path=None) -> str:
    """PROV node id of one stored entry, minted by ``trace.artifact_id`` so it is byte-identical to the id the
    matching ``record_write`` produced (store namespace + ref escaping included)."""
    if _trace is None:
        return ""
    try:
        out = _trace.artifact_id(store=store, table_or_kind=table_or_kind, pointer=pointer,
                                 store_path=store_path)
        return ("art:" + out) if out else ""
    except Exception:
        return ""


def _link_episodes(kind, entry_id, episode_ids, run_id, why, kb_path=None, mem_path=None) -> None:
    """``prov:wasDerivedFrom``: the KB entry the loop just wrote <- the episode rows it was distilled from.

    This is the edge that lets the lineage view answer the author's question end-to-end — *procedure #12 was
    derived from episode #301, which was the execution of ``PickupObject(Apple)``*. It exists nowhere on disk
    otherwise: `loop_reports.jsonl` names the KB ids but never the episodes, and `episodes` has no back-pointer.
    """
    if _trace is None or not entry_id or not episode_ids:
        return
    src = _art("kb", kind, entry_id, kb_path)
    if not src:
        return
    for eid in list(episode_ids)[:200]:
        if eid is None:
            continue
        dst = _art("mem", "episodes", eid, mem_path)
        if dst:
            _t("link", src, dst, "wasDerivedFrom", note=why, run_id=run_id)


_LINE_CACHE: dict = {}          # abs path -> (offset at a line boundary, lines before that offset)
_LINE_LOCK = threading.Lock()   # the (offset, count) pair must be read and replaced atomically


def _count_lines(path, upto: int | None = None) -> int:
    """Lines in a jsonl journal up to byte ``upto`` (whole file when ``None``).

    Counts EVERY line, blank ones included, because that is what the viewer's own enumerator does
    (``trace._read_lines`` numbers with ``enumerate(fh, 1)``): one blank line would otherwise offset every
    recorded pointer from the row the browser shows, permanently.

    The journal is append-only and grows for the life of the project, so a full re-scan per closed loop is a
    cost that rises with the project's age — a Phase-10 sweep would pay it once per record. Only the bytes
    added since the last call are scanned; the cache resets whenever the file shrank (truncation/rotation).
    """
    try:
        key = os.path.abspath(path)
        end = int(upto) if upto is not None else os.path.getsize(key)
    except Exception:
        return 0
    with _LINE_LOCK:                       # two threads appending at once must not interleave cache updates
        off, n = _LINE_CACHE.get(key, (0, 0))
        if off > end:                      # truncated, rotated, or another writer is already further ahead
            off, n = 0, 0
        if off == end:
            return n
        try:
            with open(key, "rb") as fh:
                fh.seek(off)
                chunk = fh.read(end - off)
            n += chunk.count(b"\n")
            cut = chunk.rfind(b"\n")
            if cut >= 0:                   # only ever cache a true line boundary, never a partial line
                _LINE_CACHE[key] = (off + cut + 1, n)
            if cut + 1 < len(chunk):
                n += 1                     # a trailing partial line still occupies a line number
            return n
        except Exception:
            return n


_APPEND_LOCK = threading.Lock()          # in-process serialisation of journal appends


@contextlib.contextmanager
def _file_lock(path):
    """Best-effort EXCLUSIVE lock across processes, via whichever stdlib primitive this platform has.

    Windows' CRT implements ``O_APPEND`` as "seek to end, then write", which is not atomic: two writers lose
    each other's lines outright (measured: 8 concurrent appends produced 5 lines). ``fcntl.flock`` /
    ``msvcrt.locking`` close that, and if neither is available the in-process lock still holds — degrade,
    never raise.
    """
    fh = None
    kind = None
    try:
        fh = open(str(path) + ".lock", "a+b")
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            kind = "fcntl"
        except Exception:
            try:
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                kind = "msvcrt"
            except Exception:
                kind = None
    except Exception:
        fh = None
    try:
        yield kind
    finally:
        try:
            if fh is not None:
                if kind == "fcntl":
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                elif kind == "msvcrt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                fh.close()
        except Exception:
            pass


def _append_report_line(path, text: str):
    """Append one journal line and return ITS 1-based line number — the pointer the trace store records.

    Counting the file and then appending is a race: the webapp and a concurrently running eval subprocess both
    read N and both register their (different) report as ``loop_reports.jsonl#N+1``. Here the number is
    derived from the byte offset this very append landed at, under a lock, so two writers get two different
    numbers and neither line is lost. Returns ``None`` if the file could not be written.
    """
    data = text.encode("utf-8", "replace")
    try:
        with _APPEND_LOCK, _file_lock(path):
            with open(path, "ab") as fh:
                fh.seek(0, os.SEEK_END)      # explicit: Windows' O_APPEND is not a guarantee
                fh.write(data)
                fh.flush()
                end = fh.tell()
            return _count_lines(path, upto=end)
    except Exception:
        return None

# Default loop-report trail, next to the module like knowledge_base.json/memory.db. Injectable per call.
LOOP_REPORTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_reports.jsonl")


# ---------------------------------------------------------------------- pure helpers
def decide_success(step_results, goal=None) -> dict:
    """The D3 success signal: ``{'success': bool, 'source': 'goal'|'steps', 'confidence': float}``.

    If a command-level goal check ran (``goal={'applicable': True, 'ok': ...}``, webapp's ``_live_goal_check``
    shape) its verdict WINS at confidence 0.9 -- a held-object check is stronger evidence than per-step action
    success. Otherwise success means every step result is ok (confidence 0.6); an empty ``step_results`` is a
    failure (nothing was executed, nothing was proven).
    """
    if isinstance(goal, dict) and goal.get("applicable"):
        return {"success": bool(goal.get("ok")), "source": "goal", "confidence": 0.9}
    ok = bool(step_results)
    for r in (step_results or []):
        if not (isinstance(r, dict) and bool(r.get("ok", r.get("success")))):
            ok = False
            break
    return {"success": ok, "source": "steps", "confidence": 0.6}


def _step_label(step) -> str:
    """Human-readable label for a plan step (ASCII cousin of webapp._step_label)."""
    if not isinstance(step, dict):
        return str(step)
    a = str(step.get("action") or "?").replace("_", " ")
    o = step.get("object") or ""
    to = step.get("to") or ""
    return (a + (f" {o}" if o else "") + (f" -> {to}" if to else "")).strip()


def _error_hint(error) -> str:
    """Best-effort actionable hint via thor_actions.error_hint. Lazy import -- thor_actions pulls in ai2thor,
    which only exists in the WSL/webapp environment; on a pure-stdlib box this degrades to ''."""
    try:
        import thor_actions
        return thor_actions.error_hint(error) or ""
    except Exception:
        return ""


def _failure_of(executed_steps, step_results, goal) -> tuple:
    """(label, error_text) of the decisive failure: the FIRST failed step, else the failed goal check."""
    steps = list(executed_steps or [])
    for i, r in enumerate(step_results or []):
        if isinstance(r, dict) and not bool(r.get("ok", r.get("success"))):
            step = steps[i] if i < len(steps) else (r.get("step") or {})
            label = r.get("label") or _step_label(step)
            return label, str(r.get("error") or "action failed")
    if isinstance(goal, dict) and goal.get("applicable") and not goal.get("ok"):
        err = goal.get("error") or f"expected held={goal.get('target')}, got {goal.get('held')}"
        return f"goal check (holding {goal.get('target')})", str(err)
    return "plan", ("no steps executed" if not step_results else "unknown failure")


def _related_ids(kb, command, exclude_id=None) -> list:
    """Ids of OTHER KB entries about the same normalized command (D7: a success does not delete old lessons --
    recency/provenance ranking demotes them -- but the loop report records them as related/superseded)."""
    ids = []
    try:
        ncmd = normalize_command(command)
        for kind in ("lessons", "procedures"):
            for e in kb.data.get(kind, []):
                if normalize_command(e.get("command")) == ncmd and e.get("id") != exclude_id:
                    ids.append(e.get("id"))
    except Exception:
        pass
    return ids


def _occurrence_note(memory, scene, error) -> str:
    """Read-only 'seen xN' annotation from memory's Phase-5 failure counters ('' if unknown; never raises)."""
    try:
        if memory is None:
            return ""
        ecls = _error_class(error)
        for f in memory.top_failures(scene=scene, limit=25):
            if f.get("error_class") == ecls:
                return f"seen x{f.get('occurrences')} in failure memory"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------- loop closure
def close_loop(kb, memory, *, scene, command, level, arm, executed_steps, step_results, goal=None,
               retrieved_ids=None, rag_enabled=True, loop_enabled=True, reflect=False, llm=None,
               report_path=None, embed_endpoint=None,
               run_id=None, decision_id=None, span_id=None, episode_ids=None) -> dict:
    """Close the self-improving loop for ONE executed command; return (and journal) the loop report.

    Args:
        kb: injected :class:`knowledge.KnowledgeBase` (webapp passes its ``_get_kb()`` singleton).
        memory: injected :class:`memory.Memory` or None -- used READ-ONLY to annotate lessons with known
            failure occurrences (per-step failure counting already happened in the Phase-5 funnel).
        scene / command / level / arm: run context, recorded verbatim in the report.
        executed_steps: the POST-REPAIR step dicts actually executed (what a procedure stores on success).
        step_results: parallel list of ``{'ok': bool, 'error': str|None}`` (``'success'`` accepted for
            ``'ok'``; optional ``'label'``/``'step'`` override the derived failed-step label).
        goal: webapp ``_live_goal_check`` dict (``{'applicable', 'ok', 'target', 'held', 'error'}``) or None.
        retrieved_ids: entry ids/kinds that were actually injected into the planner prompt (report only).
        rag_enabled: retrieval ablation flag -- when False the embed/reindex step is skipped.
        loop_enabled: loop ablation flag -- when False NO KB write-back and NO reindex happen at all.
        reflect / llm: optional injected callable ``llm(prompt) -> str`` writing a better lesson; template
            lesson is always the fallback and reflection is never required.
        report_path: override for the jsonl trail (tests/bench); defaults to ``LOOP_REPORTS_PATH``.
        run_id / decision_id / span_id / episode_ids: OPTIONAL observability context (keyword-only, defaulted,
            so every existing caller is unaffected). ``run_id`` binds this closure to the run that planned and
            executed the command (falling back to the ambient trace context); ``decision_id`` is the bandit
            ``bandit_decisions.id`` for the same command; ``episode_ids`` are the ``memory.episodes`` rows this
            command produced, which become ``prov:wasDerivedFrom`` edges on whatever the loop writes — the
            literal answer to "which executed actions produced this knowledge-base entry?".

    Returns the loop report::

        {ts, scene, command, level, arm, retrieved, executed_steps,
         outcome: {success, source: 'goal'|'steps', confidence},
         writes: [{kind: 'procedure'|'lesson', id, action: 'insert'|'reinforce', why}],
         related_ids, reindexed: int|None, loop_enabled}

    Exception-safe: any internal failure is swallowed into the report (optional ``error`` key) -- this
    function NEVER raises and never blocks planning/execution.
    """
    outcome = decide_success(step_results, goal)
    rid = run_id or _current_run_id()
    report = {"ts": time.time(), "scene": scene, "command": command, "level": level, "arm": arm,
              "retrieved": list(retrieved_ids or []), "executed_steps": list(executed_steps or []),
              "outcome": outcome, "writes": [], "related_ids": [], "reindexed": None,
              "loop_enabled": bool(loop_enabled),
              # Observability stamps: the report becomes joinable to the trace store instead of being an
              # anonymous line. Additive keys only — every existing consumer does a subset check.
              "run_id": rid, "trace_span_id": None, "decision_id": decision_id}

    # One LOOP span owns the whole closure: the KB upserts underneath resolve THIS span as their causal parent
    # from the ambient context, so no knowledge.py signature has to change to attribute them.
    with _span("close_loop", kind="LOOP", op="close_loop", actor="system:improve", scene=scene,
               run_id=rid, parent_span_id=span_id,
               input={"command": command, "level": level, "arm": arm, "scene": scene,
                      "executed_steps": list(executed_steps or []), "decision_id": decision_id,
                      "loop_enabled": bool(loop_enabled), "rag_enabled": bool(rag_enabled)},
               attrs={"agent.level": level, "agent.arm": arm, "agent.decision_id": decision_id,
                      "agent.loop_enabled": bool(loop_enabled), "thor.scene": scene}) as _sid:
        report["trace_span_id"] = _sid
        try:
            _close_loop_body(kb, memory, report=report, outcome=outcome, scene=scene, command=command,
                             executed_steps=executed_steps, step_results=step_results, goal=goal,
                             retrieved_ids=retrieved_ids, rag_enabled=rag_enabled,
                             loop_enabled=loop_enabled, reflect=reflect, llm=llm, report_path=report_path,
                             embed_endpoint=embed_endpoint, run_id=rid, episode_ids=episode_ids)
        except Exception as exc:
            # Belt and braces: every stage inside already swallows, and the span re-raises the CALLER's
            # exception by contract — so this keeps close_loop's "never raises" promise absolute.
            report["error"] = str(exc)
    return report


def _close_loop_body(kb, memory, *, report, outcome, scene, command, executed_steps, step_results, goal,
                     retrieved_ids, rag_enabled, loop_enabled, reflect, llm, report_path, embed_endpoint,
                     run_id, episode_ids) -> None:
    """The original :func:`close_loop` body, extracted verbatim so the LOOP span can wrap it.

    Mutates ``report`` in place and never raises (each stage keeps its own swallow-everything guard).
    """
    # ---- prov:used — the entries that ACTUALLY reached the planner prompt are what grounded this plan.
    # Deliberately not every KB read: `search()` runs on paths that never influence a plan, and recording
    # those would make "used" meaningless as a grounding signal.
    for hit in (retrieved_ids or []):
        try:
            if isinstance(hit, dict):
                hid, hkind = hit.get("id"), _KIND_SINGULAR.get(str(hit.get("kind") or ""), "procedure")
            else:
                hid, hkind = hit, "procedure"
            if hid is None:
                continue
            _t("record_use", store="kb", table_or_kind=hkind, pointer=hid,
               why=f"retrieved into the planner prompt for {command!r}", run_id=run_id,
               store_path=getattr(kb, "path", None))
        except Exception:
            pass

    # ---- write-back (gated by the loop toggle; all auto-writes are source='environment', D12)
    try:
        if loop_enabled and kb is not None:
            if outcome["success"] and executed_steps:
                # D4: store the EXECUTED (post-repair) sequence -- the plan that actually worked.
                why = (f"loop closure: command succeeded ({outcome['source']} signal, "
                       f"conf {outcome['confidence']})")
                res = kb.upsert_procedure(command, list(executed_steps), why=why, scene=scene,
                                          tags=["auto-loop"], source="environment",
                                          confidence=outcome["confidence"], scope="global")   # D8
                report["writes"].append({"kind": "procedure", "id": res.get("id"),
                                         "action": res.get("action"), "why": why})
                report["related_ids"] = _related_ids(kb, command, exclude_id=res.get("id"))
                _link_episodes("procedure", res.get("id"), episode_ids, run_id, why,
                                kb_path=getattr(kb, "path", None), mem_path=getattr(memory, "path", None))
            elif not outcome["success"]:
                # D5: template lesson -- failed step label + agent error + actionable hint.
                label, err = _failure_of(executed_steps, step_results, goal)
                hint = _error_hint(err)
                problem = f"{label} failed: {err}"
                lesson_text = hint or f"'{label}' failed with: {err} - fix that precondition before retrying"
                if reflect and llm is not None:                # optional reflection, never required
                    try:
                        prompt = (f"Command: {command}\nFailed step: {label}\nError: {err}\n"
                                  "Write ONE short actionable lesson (a single sentence) that would avoid "
                                  "this failure next time.")
                        t = str(llm(prompt)).strip()
                        if t:
                            lesson_text = t[:300]
                    except Exception:
                        pass
                note = _occurrence_note(memory, scene, err)
                why = "loop closure: command failed" + (f" ({note})" if note else "")
                # Key the dedup on the RAW error, not the label-prefixed `problem` (whose leading step label
                # would otherwise consume _error_class's 48-char window and collapse distinct failure modes).
                res = kb.upsert_lesson(command, problem, lesson_text, why=why, scene=scene,
                                       source="environment", confidence=outcome["confidence"],
                                       scope=scene or "global", error=err)                    # D8
                report["writes"].append({"kind": "lesson", "id": res.get("id"),
                                         "action": res.get("action"), "why": why})
                report["related_ids"] = _related_ids(kb, command, exclude_id=res.get("id"))
                _link_episodes("lesson", res.get("id"), episode_ids, run_id, why,
                                kb_path=getattr(kb, "path", None), mem_path=getattr(memory, "path", None))
    except Exception as exc:                                   # a loop failure must never break execution
        report["error"] = str(exc)

    # ---- best-effort embedding of what was just written (D9; only entries missing vectors get embedded)
    try:
        if loop_enabled and rag_enabled and report["writes"]:
            import rag                                         # lazy sibling import (network embed inside)
            # The embed endpoint must match retrieval's: under WSL, Ollama lives on the Windows host, so a
            # bare HybridRAG() (default localhost) silently fails to embed and the re-embed no-ops. Callers
            # (eval.py, webapp) pass their resolved gateway endpoint; absent one we keep the localhost default.
            hr = rag.HybridRAG(kb=kb, endpoint=embed_endpoint) if embed_endpoint else rag.HybridRAG(kb=kb)
            report["reindexed"] = int(hr.reindex())
    except Exception:
        report["reindexed"] = None                             # Ollama down / rag unavailable: degrade silently

    # ---- journal the report (D13) -- one JSON line, best-effort
    path = report_path or LOOP_REPORTS_PATH
    line_no = None
    try:
        # The journal line has no id of its own, so it cannot be referenced from anywhere. Appending through
        # _append_report_line gives it one — the addressable artifact `file:report:<basename>#<n>`.
        line_no = _append_report_line(path, json.dumps(report, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
    _t("record_write", store="file", table_or_kind="loop_reports.jsonl",
       pointer=f"{os.path.basename(path)}#{line_no}", verdict="insert",
       why="loop report journaled", source="environment", scope=scene, actor="system:improve",
       summary=f"{command} -> {len(report['writes'])} write(s), "
               f"{'success' if outcome['success'] else 'failure'}",
       content=report, after={"writes": report["writes"], "outcome": outcome,
                              "reindexed": report.get("reindexed")},
       run_id=run_id)
    _t("emit", "loop.closed", actor="system:improve", scene=scene, run_id=run_id,
       summary=(f"{'success' if outcome['success'] else 'failure'} ({outcome['source']}) -> "
                + (", ".join(f"{w['action']} {w['kind']} #{w['id']}" for w in report["writes"])
                   or ("loop disabled" if not loop_enabled else "no write"))),
       payload=report)
    return None


if __name__ == "__main__":
    # Self-test: pure stdlib, no simulator, no Ollama. Temp-dir KB + fake step results exercise every loop
    # branch: procedure insert -> reinforce dedup, lesson on failure (insert -> reinforce), related-lesson
    # linking after a later success, loop-off writes nothing, reflect hook, stubbed reindex, report shape,
    # and the jsonl trail.
    import shutil
    import sys
    import tempfile
    import types

    import knowledge as _knowledge

    tmp = tempfile.mkdtemp(prefix="improve_selftest_")
    try:
        kb = _knowledge.KnowledgeBase(path=os.path.join(tmp, "kb.json"))
        reports = os.path.join(tmp, "reports.jsonl")
        scene = "FloorPlan1"
        steps = [{"action": "go_to", "object": "Apple"}, {"action": "pickup", "object": "Apple"}]
        ok2 = [{"ok": True, "error": None}, {"ok": True, "error": None}]

        # Pure helpers.
        assert normalize_command("  Put the APPLE, in the Fridge!! ") == "put the apple in the fridge"
        assert plan_signature(steps) == (("go_to", "apple"), ("pickup", "apple"))
        assert decide_success(ok2, {"applicable": True, "ok": False}) == \
            {"success": False, "source": "goal", "confidence": 0.9}, "an applicable goal verdict must win"
        assert decide_success(ok2, {"applicable": False}) == \
            {"success": True, "source": "steps", "confidence": 0.6}
        assert decide_success([], None)["success"] is False, "no executed steps proves nothing"

        # (1) success with an applicable goal -> procedure INSERT (source=environment, scope=global).
        r1 = close_loop(kb, None, scene=scene, command="pick up the apple", level="L1", arm="reuse_memory",
                        executed_steps=steps, step_results=ok2,
                        goal={"applicable": True, "ok": True, "target": "Apple", "held": "Apple"},
                        retrieved_ids=[], rag_enabled=False, loop_enabled=True, report_path=reports)
        assert r1["outcome"] == {"success": True, "source": "goal", "confidence": 0.9}
        assert [(w["kind"], w["action"]) for w in r1["writes"]] == [("procedure", "insert")], f"got {r1['writes']!r}"
        p = kb.data["procedures"][0]
        assert p["source"] == "environment" and p["scope"] == "global" and p["plan"] == steps
        print("(1) success -> procedure insert id", r1["writes"][0]["id"])

        # (2) same command+plan again -> REINFORCE, no duplicate row.
        r2 = close_loop(kb, None, scene=scene, command="Pick up the Apple!", level="L1", arm="reuse_memory",
                        executed_steps=steps, step_results=ok2,
                        goal={"applicable": True, "ok": True, "target": "Apple", "held": "Apple"},
                        rag_enabled=False, loop_enabled=True, report_path=reports)
        assert r2["writes"][0]["action"] == "reinforce" and r2["writes"][0]["id"] == r1["writes"][0]["id"]
        assert len(kb.data["procedures"]) == 1 and kb.data["procedures"][0]["occurrences"] == 2
        print("(2) repeat success -> reinforce, occurrences", kb.data["procedures"][0]["occurrences"])

        # (3) failure -> template lesson INSERT scoped to the scene; fake memory supplies the 'seen xN' note.
        mug_steps = [{"action": "go_to", "object": "Mug"}, {"action": "pickup", "object": "Mug"}]
        bad = [{"ok": True, "error": None},
               {"ok": False, "error": "PickupObject failed: Mug|+01.2|-00.3 not visible"}]

        class _FakeMem:                                        # read-only stand-in for memory.top_failures
            def top_failures(self, scene=None, limit=25):
                return [{"error_class": _error_class("PickupObject failed: Mug|+9.9 not visible"),
                         "occurrences": 4}]

        r3 = close_loop(kb, _FakeMem(), scene=scene, command="pick up the mug", level="L1", arm="plan_fresh",
                        executed_steps=mug_steps, step_results=bad, goal={"applicable": False},
                        retrieved_ids=[{"id": 1, "kind": "procedures"}],
                        rag_enabled=False, loop_enabled=True, report_path=reports)
        assert r3["outcome"]["success"] is False and r3["writes"][0]["kind"] == "lesson"
        les = kb.data["lessons"][0]
        assert les["source"] == "environment" and les["scope"] == scene
        assert "pickup Mug" in les["problem"] and "not visible" in les["problem"]
        assert "seen x4" in r3["writes"][0]["why"], f"failure-counter note missing: {r3['writes'][0]['why']!r}"
        print("(3) failure -> lesson insert:", les["problem"])

        # (4) same failure class again (different object id/coords) -> lesson REINFORCE, still one row.
        bad2 = [{"ok": True, "error": None},
                {"ok": False, "error": "PickupObject failed: Mug|+03.9|-01.7 not visible"}]
        r4 = close_loop(kb, None, scene=scene, command="pick up the mug", level="L1", arm="plan_fresh",
                        executed_steps=mug_steps, step_results=bad2, goal={"applicable": False},
                        rag_enabled=False, loop_enabled=True, report_path=reports)
        assert r4["writes"][0]["action"] == "reinforce" and len(kb.data["lessons"]) == 1
        assert kb.data["lessons"][0]["occurrences"] == 2
        print("(4) repeat failure -> lesson reinforce, occurrences", kb.data["lessons"][0]["occurrences"])

        # (5) later success on the same command -> procedure written, superseded lesson id in related_ids (D7).
        r5 = close_loop(kb, None, scene=scene, command="pick up the mug", level="L1", arm="reuse_memory",
                        executed_steps=mug_steps, step_results=ok2,
                        goal={"applicable": True, "ok": True, "target": "Mug", "held": "Mug"},
                        rag_enabled=False, loop_enabled=True, report_path=reports)
        assert r5["writes"][0]["kind"] == "procedure" and les["id"] in r5["related_ids"]
        assert any(e["id"] == les["id"] for e in kb.data["lessons"]), "old lesson must persist (D7)"
        print("(5) success after failure -> related lesson id", les["id"], "recorded, lesson kept")

        # (6) loop OFF -> no writes, no reindex, KB untouched; report still journaled (ablation trace).
        before = kb.stats()
        r6 = close_loop(kb, None, scene=scene, command="pick up the apple", level="L1", arm="reuse_memory",
                        executed_steps=steps, step_results=ok2,
                        goal={"applicable": True, "ok": True, "target": "Apple", "held": "Apple"},
                        rag_enabled=True, loop_enabled=False, report_path=reports)
        assert r6["writes"] == [] and r6["reindexed"] is None and kb.stats() == before
        assert r6["loop_enabled"] is False
        print("(6) loop off -> zero writes, stats", kb.stats())

        # (7) reflect hook: an injected llm callable writes the lesson text (template is the fallback).
        r7 = close_loop(kb, None, scene=scene, command="slice the bread", level="L1", arm="plan_fresh",
                        executed_steps=[{"action": "slice", "object": "Bread"}],
                        step_results=[{"ok": False, "error": "not holding a knife"}],
                        goal={"applicable": False}, rag_enabled=False, loop_enabled=True,
                        reflect=True, llm=lambda prompt: "hold a knife before slicing",
                        report_path=reports)
        knife = [e for e in kb.data["lessons"] if e["command"] == "slice the bread"][0]
        assert knife["lesson"] == "hold a knife before slicing", f"reflected lesson missing: {knife!r}"
        print("(7) reflect -> injected lesson:", knife["lesson"])

        # (8) reindex path via a stubbed rag module (loop on + rag on + a write happened).
        stub = types.ModuleType("rag")

        class _StubRAG:
            def __init__(self, kb=None, **kw):
                pass

            def reindex(self, force=False):
                return 3

        stub.HybridRAG = _StubRAG
        prev = sys.modules.get("rag")
        sys.modules["rag"] = stub
        try:
            r8 = close_loop(kb, None, scene=scene, command="wash the pan", level="L1", arm="reuse_memory",
                            executed_steps=[{"action": "wash", "object": "Pan"}],
                            step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                            rag_enabled=True, loop_enabled=True, report_path=reports)
        finally:
            if prev is None:
                del sys.modules["rag"]
            else:
                sys.modules["rag"] = prev
        assert r8["reindexed"] == 3, f"stubbed reindex count expected, got {r8['reindexed']!r}"
        print("(8) reindex hook -> reindexed", r8["reindexed"])

        # (9) report shape + jsonl trail: one line per close_loop call, loop-off run included.
        keys = {"ts", "scene", "command", "level", "arm", "retrieved", "executed_steps", "outcome",
                "writes", "related_ids", "reindexed", "loop_enabled"}
        assert keys <= set(r1), f"report keys missing: {keys - set(r1)!r}"
        with open(reports, "r", encoding="utf-8") as fh:
            lines = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
        assert len(lines) == 8 and lines[0]["command"] == "pick up the apple"
        assert lines[2]["writes"][0]["kind"] == "lesson" and lines[5]["writes"] == []
        with open(kb.audit_path, "r", encoding="utf-8") as fh:
            audit = fh.read()
        assert "UPSERT procedure" in audit and "UPSERT lesson" in audit and "<environment>" in audit
        print("(9) jsonl trail:", len(lines), "reports | audit has UPSERT lines")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("IMPROVE SELFTEST OK")
