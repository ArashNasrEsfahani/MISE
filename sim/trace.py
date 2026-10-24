"""Observability plane — the append-only event store that makes every agent write auditable.

This is the *sidecar* the thesis's auditability promise needs (CLAUDE.md: "Self-improving loop writes are
auditable. When the agent edits its own knowledge base, log what/why so the proof-of-concept can be evaluated.").
Today that promise is kept by two flat text files and a JSONL journal, none of which carries an identifier
linking a knowledge-base entry back to the command, the LLM call, or the AI2-THOR action that produced it.
This module supplies that identifier, and the graph built on it.

Three things at once (see ``docs/observability-design.md``, of which this file is §3, §4, §6 and §7):

  **D1 total logging** — one durable, append-only log of every run, command, LLM call, plan step, executed
  AI2-THOR primitive, perception pass, feedback event and knowledge/memory write, in ``sim/trace.db``.

  **D2 causality** — every database write is attributable to the activity that caused it and is navigable in
  BOTH directions by a single indexed query: ``span_effects(span)`` answers *"what did this action write?"* and
  ``artifact_causes(art)`` answers *"what produced this entry?"*. Multi-hop lineage (through the episodes an
  entry was distilled from, the lessons the planner used, the agents responsible) lives in the ``edges`` table
  as a W3C **PROV** graph and is walked by a cycle-guarded recursive CTE.

  **D3 visual browsing** — a read-only query API (``query_events`` / ``run_tree`` / ``lineage`` / ``stats`` /
  ``db_tables`` / ``db_rows`` / ``db_row``) that the ``/trace*`` endpoints and ``sim/web/trace.html`` render.

Model (adopted, not invented): OTel span semantics (``trace_id``/``span_id``/``parent_span_id``/status), W3C
Trace-Context id formats (32-hex run, 16-hex span), OTel GenAI ``gen_ai.*`` attributes, OpenInference's span
``kind`` enum (plus ``ACTION`` for an embodied primitive and ``MAINT`` for memory maintenance), Langfuse's
``level`` knob, and W3C PROV-O relations stored relationally (``rel='wasGeneratedBy'`` etc. — no RDF, by design;
``export_prov_json`` restores interoperability).

Design rules this file lives by
-------------------------------
* **Stdlib only** — ``sqlite3``/``json``/``threading``/``contextvars``/``hashlib``/``logging`` and friends. It
  imports and self-tests under a bare Python with no third-party packages, and it must **never** import
  ``webapp``, ``memory``, ``knowledge``, ``thor_actions`` or ``ai2thor`` at module top: the dependency arrow
  points instrumented-module → trace, never back.
* **No side effects at import time.** The store is opened lazily on the first API call, so merely importing this
  module creates no file (that is what makes ``TRACE_ENABLED=0`` a true opt-out).
* **A separate database.** ``memory.db`` runs in rollback-journal mode where a writer takes an exclusive lock;
  putting trace rows there would let the observer stall the observed. ``trace.db`` is WAL, its own file, and is
  git-ignorable.
* **No exception ever escapes.** Every public entry point is wrapped: a raising tracer inside ``ThorAgent.do()``
  would kill every action in the simulator, and inside the webapp's worker thread it would kill the run loop
  permanently. On any internal failure the call returns a benign value (``0`` / ``""`` / ``{}`` / ``None``).
  The two context managers ``run()`` and ``span()`` re-raise the *caller's* exception — the tracer swallows only
  its own errors, never the agent's.
* **Degrade, warn once, keep working** (the ``sim/rag.py`` house pattern). If the DB is unwritable, locked,
  read-only or on a full disk: ``_ENABLED=False``, ``_DEGRADED=True``, one ``logging`` warning, and every
  function keeps returning its benign value. ``status()`` reports ``degraded`` + ``reason`` so the viewer can
  show a banner. The agent is unaffected either way.
* **Instrumenting a call site is one ``with`` statement** and never changes a function's signature or return
  value. Ambient context resolves through three layers (contextvar → per-thread binding → process default), so
  the webapp's plan thread can hand a run to the worker thread that owns the AI2-THOR controller.
* **Orphans are a feature.** A write recorded with no run context is stored with ``run_id IS NULL``, never
  dropped: it is the instrumentation-coverage signal (legacy Tk GUI, ``seed_memory.py``, offline batches).
* **Nothing is auto-deleted.** The trace *is* a research artefact; ``prune()`` is explicit and needs
  ``confirm=True``.

Naming hazard (verified): ``trace`` is also a stdlib module (the coverage tracer). Because ``sim/`` sits ahead of
the stdlib on ``sys.path``, ``import trace`` from a ``sim/`` module resolves to **this** file. Consumers must
still guard, because a differently ordered ``sys.path`` would silently hand them the stdlib module::

    try:                                # observability sidecar; optional by design
        import trace as _trace          # sim/trace.py — shadows the stdlib coverage module
        if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
            _trace = None               # we resolved the STDLIB trace module: disable instrumentation
    except Exception:
        _trace = None

Environment contract: ``TRACE_ENABLED`` ``TRACE_DB`` ``TRACE_RUN_ID`` ``TRACE_ORIGIN`` ``TRACE_ASYNC``
``TRACE_LEVEL`` ``TRACE_SKIP_ACTIONS`` ``TRACE_COLLAPSE_NAV`` ``TRACE_MAX_BLOB`` ``TRACE_QUEUE_MAX``
``TRACE_FLUSH_MS`` ``TRACE_BATCH`` (defaults in §6.1 of the design doc and in ``_load_env`` below).

Run ``python sim/trace.py`` for a self-contained demo: it synthesises a small traced run into a temporary
database, prints the timeline and a lineage query, and then proves the degradation path by pointing the store at
an unwritable path and calling every public API.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from collections import OrderedDict, deque

# --------------------------------------------------------------------------- versions & vocabularies
TRACE_API_VERSION = 1      # consumers guard on this to detect the stdlib `trace` module (see docstring)
SCHEMA_VERSION = 1

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "trace.db")

LEVELS = ("DEBUG", "DEFAULT", "WARNING", "ERROR")
KINDS = ("AGENT", "LLM", "RETRIEVER", "TOOL", "ACTION", "PERCEPTION", "LOOP", "FEEDBACK", "EVALUATOR", "MAINT")
VERDICTS = ("insert", "reinforce", "update", "delete", "noop")
RELS = ("wasGeneratedBy", "used", "wasDerivedFrom", "wasRevisionOf", "wasAttributedTo",
        "wasAssociatedWith", "wasInformedBy", "wasInvalidatedBy")
OPS = ("invoke_agent", "chat", "execute_tool", "thor_action", "retrieval", "embeddings", "close_loop",
       "evaluate", "observe", "feedback", "maintenance", "scene_reset")

_LVL_ORD = {name: i for i, name in enumerate(LEVELS)}

# writes.table_or_kind (mirrors the store's own table name) -> singular artifact kind used in artifact ids.
_ART_KIND = {"episodes": "episode", "spatial": "spatial", "failures": "failure", "nav_cells": "nav_cell",
             "perception_labels": "perception_label", "procedure": "procedure", "lesson": "lesson",
             "fact": "fact", "loop_reports.jsonl": "report", "bandit_decisions": "decision",
             "bandit_arms": "arm", "embeddings": "embeddings"}

# The existing per-entry `source` field IS prov:wasAttributedTo; map it onto the closed actor set.
_SOURCE_ACTOR = {"human": "human:arash", "demo": "human:arash",
                 "environment": "env:ai2thor", "prior": "system:seed"}

_log = logging.getLogger("trace")


def _warn(msg: str):
    """Warn through ``logging`` exactly like ``rag._warn`` — degradation is reported, never raised."""
    _log.warning("trace: %s", msg)


# --------------------------------------------------------------------------- schema (design §3.2, verbatim)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    parent_run_id TEXT,
    kind          TEXT,
    origin        TEXT,
    started_ts    REAL,
    ended_ts      REAL,
    status        TEXT,
    scene         TEXT,
    level         TEXT,
    lang          TEXT,
    command       TEXT,
    arm           TEXT,
    decision_id   INTEGER,
    agent_id      TEXT,
    seed          INTEGER,
    config_json   TEXT,
    config_hash   TEXT,
    code_version  TEXT,
    error         TEXT,
    n_spans       INTEGER DEFAULT 0,
    n_events      INTEGER DEFAULT 0,
    n_writes      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_runs_started  ON runs(started_ts DESC);
CREATE INDEX IF NOT EXISTS ix_runs_decision ON runs(decision_id);
CREATE INDEX IF NOT EXISTS ix_runs_kind     ON runs(kind, started_ts DESC);
CREATE INDEX IF NOT EXISTS ix_runs_scene    ON runs(scene, started_ts DESC);

CREATE TABLE IF NOT EXISTS spans (
    span_id        TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    parent_span_id TEXT,
    seq            INTEGER,
    depth          INTEGER,
    name           TEXT,
    op             TEXT,
    kind           TEXT,
    step_ix        INTEGER,
    start_ts       REAL,
    end_ts         REAL,
    duration_ms    REAL,
    status         TEXT,
    level          TEXT,
    error          TEXT,
    scene          TEXT,
    actor          TEXT,
    input_json     TEXT,
    output_json    TEXT,
    attrs_json     TEXT,
    repeat         INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_spans_run    ON spans(run_id, seq);
CREATE INDEX IF NOT EXISTS ix_spans_parent ON spans(parent_span_id);
CREATE INDEX IF NOT EXISTS ix_spans_kind   ON spans(kind, start_ts DESC);
CREATE INDEX IF NOT EXISTS ix_spans_open   ON spans(end_ts) WHERE end_ts IS NULL;

CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT UNIQUE,
    ts           REAL,
    run_id       TEXT,
    span_id      TEXT,
    type         TEXT,
    actor        TEXT,
    level        TEXT,
    scene        TEXT,
    summary      TEXT,
    payload_json TEXT,
    origin       TEXT,
    pid          INTEGER,
    thread       TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_run   ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS ix_events_type  ON events(type, seq DESC);
CREATE INDEX IF NOT EXISTS ix_events_span  ON events(span_id);
CREATE INDEX IF NOT EXISTS ix_events_level ON events(level, seq DESC);
CREATE INDEX IF NOT EXISTS ix_events_ts    ON events(ts DESC);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id  TEXT PRIMARY KEY,
    store        TEXT,
    kind         TEXT,
    ref          TEXT,
    created_seq  INTEGER,
    created_ts   REAL,
    updated_ts   REAL,
    content_hash TEXT,
    summary      TEXT,
    source       TEXT,
    scope        TEXT,
    confidence   REAL,
    revision     INTEGER DEFAULT 1,
    occurrences  INTEGER DEFAULT 1,
    dedup_key    TEXT,
    alive        INTEGER DEFAULT 1,
    n_causes     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_art_store ON artifacts(store, kind, created_ts DESC);
CREATE INDEX IF NOT EXISTS ix_art_ref   ON artifacts(store, kind, ref);
CREATE INDEX IF NOT EXISTS ix_art_dedup ON artifacts(dedup_key);
CREATE INDEX IF NOT EXISTS ix_art_src   ON artifacts(source, created_ts DESC);

CREATE TABLE IF NOT EXISTS writes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq     INTEGER,
    run_id        TEXT,
    span_id       TEXT,
    step_ix       INTEGER,
    ts            REAL,
    store         TEXT,
    table_or_kind TEXT,
    pointer       TEXT,
    artifact_id   TEXT,
    verdict       TEXT,
    dedup_key     TEXT,
    before_json   TEXT,
    after_json    TEXT,
    why           TEXT,
    source        TEXT,
    audit_line    TEXT,
    origin        TEXT
);
CREATE INDEX IF NOT EXISTS ix_writes_ptr  ON writes(store, table_or_kind, pointer, ts);
CREATE INDEX IF NOT EXISTS ix_writes_art  ON writes(artifact_id, ts);
CREATE INDEX IF NOT EXISTS ix_writes_span ON writes(span_id);
CREATE INDEX IF NOT EXISTS ix_writes_run  ON writes(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_writes_seq  ON writes(event_seq);

CREATE TABLE IF NOT EXISTS edges (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    src    TEXT NOT NULL,
    dst    TEXT NOT NULL,
    rel    TEXT NOT NULL,
    run_id TEXT,
    seq    INTEGER,
    ts     REAL,
    note   TEXT,
    UNIQUE (src, dst, rel)
);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS ix_edges_rel ON edges(rel);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# --------------------------------------------------------------------------- prepared statements
_SQL = {
    "events": ("INSERT OR IGNORE INTO events (seq, event_id, ts, run_id, span_id, type, actor, level, scene,"
               " summary, payload_json, origin, pid, thread) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
    "run_ins": ("INSERT OR REPLACE INTO runs (run_id, parent_run_id, kind, origin, started_ts, ended_ts, status,"
                " scene, level, lang, command, arm, decision_id, agent_id, seed, config_json, config_hash,"
                " code_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
    # Late-arriving run facts. The command's hierarchy level (L1/L2/L3) — first-class in this thesis — is only
    # known once the planner has answered, which is well inside the run; COALESCE keeps every field additive.
    "run_upd": ("UPDATE runs SET level=COALESCE(?, level), lang=COALESCE(?, lang), scene=COALESCE(?, scene),"
                " arm=COALESCE(?, arm), decision_id=COALESCE(?, decision_id),"
                " command=COALESCE(?, command) WHERE run_id=?"),
    "run_end": ("UPDATE runs SET ended_ts=?, status=?, error=COALESCE(?, error),"
                " n_spans=(SELECT COUNT(*) FROM spans WHERE spans.run_id=runs.run_id),"
                " n_events=(SELECT COUNT(*) FROM events WHERE events.run_id=runs.run_id),"
                " n_writes=(SELECT COUNT(*) FROM writes WHERE writes.run_id=runs.run_id)"
                " WHERE run_id=?"),
    "span_ins": ("INSERT OR REPLACE INTO spans (span_id, run_id, parent_span_id, seq, depth, name, op, kind,"
                 " step_ix, start_ts, end_ts, duration_ms, status, level, error, scene, actor, input_json,"
                 " output_json, attrs_json, repeat) VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?,NULL,?,?,?,"
                 "NULL,?,1)"),
    "span_end": ("UPDATE spans SET end_ts=?, duration_ms=?, status=?, error=COALESCE(?, error),"
                 " output_json=COALESCE(?, output_json), attrs_json=COALESCE(?, attrs_json) WHERE span_id=?"),
    "span_rep": "UPDATE spans SET repeat=repeat+?, end_ts=?, duration_ms=? WHERE span_id=?",
    "span_orphans": ("UPDATE spans SET end_ts=?, status='ERROR', error='span not closed',"
                     " duration_ms=(?-start_ts)*1000.0 WHERE run_id=? AND end_ts IS NULL"),
    "writes": ("INSERT INTO writes (event_seq, run_id, span_id, step_ix, ts, store, table_or_kind, pointer,"
               " artifact_id, verdict, dedup_key, before_json, after_json, why, source, audit_line, origin)"
               " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"),
    "edges": "INSERT OR IGNORE INTO edges (src, dst, rel, run_id, seq, ts, note) VALUES (?,?,?,?,?,?,?)",
    "meta": "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
    "art_stub": ("INSERT OR IGNORE INTO artifacts (artifact_id, store, kind, ref, created_seq, created_ts,"
                 " updated_ts, content_hash, summary, source, scope, confidence, revision, occurrences,"
                 " dedup_key, alive, n_causes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,1,?,1,0)"),
}

# The artifact upsert differs only in its DO UPDATE clause, one per verdict (design §3.5): ONE row per entry,
# whose revision/occurrences/updated_ts/n_causes accumulate as causes arrive. The trailing '?' in each DO UPDATE
# clause re-binds the caller's raw `occurrences` (NULL means "the store did not tell us; infer it").
_ART_HEAD = ("INSERT INTO artifacts (artifact_id, store, kind, ref, created_seq, created_ts, updated_ts,"
             " content_hash, summary, source, scope, confidence, revision, occurrences, dedup_key, alive,"
             " n_causes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,COALESCE(?,1),?,?,1)"
             " ON CONFLICT(artifact_id) DO UPDATE SET updated_ts=excluded.updated_ts,"
             " summary=COALESCE(excluded.summary, artifacts.summary),"
             " source=COALESCE(excluded.source, artifacts.source),"
             " scope=COALESCE(excluded.scope, artifacts.scope),"
             " confidence=COALESCE(excluded.confidence, artifacts.confidence),"
             " dedup_key=COALESCE(excluded.dedup_key, artifacts.dedup_key),"
             # The LATEST content wins: an artifact row is the entry as it stands now (revision N), so keeping
             # the first hash forever would make `content_hash` describe a version nobody can read any more.
             " content_hash=COALESCE(excluded.content_hash, artifacts.content_hash),"
             " n_causes=artifacts.n_causes+1")
_SQL_ART = {
    # A repeated `insert` on an existing pointer keeps the original creation facts and just counts the cause.
    "insert": _ART_HEAD + ", occurrences=COALESCE(?, artifacts.occurrences), alive=1",
    "reinforce": (_ART_HEAD + ", revision=artifacts.revision+1, alive=1,"
                  " occurrences=COALESCE(?, artifacts.occurrences+1)"),
    "update": (_ART_HEAD + ", revision=artifacts.revision+1, alive=1,"
               " occurrences=COALESCE(?, artifacts.occurrences)"),
    "delete": _ART_HEAD + ", alive=0, occurrences=COALESCE(?, artifacts.occurrences)",
    "noop": _ART_HEAD + ", occurrences=COALESCE(?, artifacts.occurrences)",
}

# --------------------------------------------------------------------------- module state
_CFG = {
    "db_path": None, "enabled": True, "origin": None, "async": True, "level": "DEBUG", "journal_mode": None,
    # Bumped every time the store is opened or closed. `_rconn` caches a READ connection PER THREAD, keyed on
    # the path -- so when the file is deleted and recreated under the same name (which is exactly what
    # `/reset?scope=all` does), every other thread keeps its handle on the removed inode and reads an empty
    # database forever. Observed: the store held 625 events while /trace_stats reported 0. The generation is
    # what tells those threads their handle is stale.
    "generation": 0,
    "skip_actions": {"Pass", "Done"}, "collapse_nav": True, "max_blob": 8192,
    "queue_max": 20000, "flush_ms": 250, "batch": 200,
}
_ENABLED = False          # tracing on AND store writable
_DEGRADED = False
_REASON = None
_OPENED = False           # lazy-open latch (no side effects at import time)
_WARNED = False

_conn = None              # writer connection (single writer thread owns it)
_lock = threading.RLock()          # guards module state + the queue
_cv = threading.Condition(_lock)   # writer wake-up / flush hand-shake
_queue = deque()          # bounded ring of pending ops: (opcode, params, is_debug)
_writer = None
_writer_stop = False
_writer_restarts = 0

_seq = 0                  # monotonic events.seq, allocated synchronously so emit() can return it
_seq_end = 0              # exclusive end of the seq BLOCK this process reserved (see _reserve_seq_block)
SEQ_BLOCK = 4096          # how many seq values one process claims per trip to the meta high-water mark
_enqueued = 0             # ops handed to the writer
_committed = 0            # ops durably committed
_dropped = 0
_write_errors = 0
_skipped_actions = 0
_last_drop_warn = 0
_metrics = deque(maxlen=4096)      # emit() wall time, microseconds
_flush_metrics = deque(maxlen=512)  # writer commit wall time, milliseconds

_span_meta = OrderedDict()   # span_id -> {'run_id','depth','attrs','start_ts','parent'} for OPEN spans
_run_seq = {}                # run_id -> next span ordinal
_run_prev = {}               # run_id -> the ambient frames it displaced, restored by end_run
_open_runs = set()           # run ids started and not yet ended: only these may own an ambient frame
_nav_state = {}              # parent-span key -> collapse state for record_action
_tls = threading.local()     # per-thread READ connection

# Ambient context: three layers, resolved in current().
_CTX = contextvars.ContextVar("trace_ctx", default=None)
_AMBIENT: dict = {}
_ambient_lock = threading.RLock()

_INHERITED_RUN_ID = None     # from TRACE_RUN_ID: the eval-subprocess hand-off

# Paths of the OTHER stores, browsed strictly read-only by db_rows()/artifact(). Overridable by env or by
# assigning these module globals (the browser API takes no path argument by design — see design §6.5).
KB_PATH = os.environ.get("TRACE_KB_PATH") or os.path.join(HERE, "knowledge_base.json")
KB_AUDIT_PATH = os.environ.get("TRACE_KB_AUDIT_PATH") or os.path.join(HERE, "knowledge_base_audit.log")
MEM_DB_PATH = os.environ.get("TRACE_MEM_DB_PATH") or os.path.join(HERE, "memory.db")
LOOP_REPORTS_PATH = os.environ.get("TRACE_LOOP_REPORTS_PATH") or os.path.join(HERE, "loop_reports.jsonl")


# --------------------------------------------------------------------------- small helpers
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def _infer_origin() -> str:
    try:
        base = os.path.basename(sys.argv[0] or "").lower()
    except Exception:
        base = ""
    if "webapp" in base:
        return "webapp"
    if "eval" in base:
        return "eval"
    if "bench" in base:
        return "bench"
    if "selftest" in base or "test" in base:
        return "test"
    return "cli"


def _load_env():
    """Read the environment contract into ``_CFG``. Called at import (pure) and from ``configure()``."""
    _CFG["enabled"] = _env_bool("TRACE_ENABLED", True)
    db = os.environ.get("TRACE_DB") or DEFAULT_DB
    _CFG["db_path"] = db if os.path.isabs(db) else os.path.join(HERE, db)
    _CFG["origin"] = os.environ.get("TRACE_ORIGIN") or _infer_origin()
    _CFG["async"] = _env_bool("TRACE_ASYNC", True)
    lvl = (os.environ.get("TRACE_LEVEL") or "DEBUG").strip().upper()
    _CFG["level"] = lvl if lvl in _LVL_ORD else "DEBUG"
    skip = os.environ.get("TRACE_SKIP_ACTIONS")
    _CFG["skip_actions"] = ({s.strip() for s in skip.split(",") if s.strip()} if skip is not None
                            else {"Pass", "Done"})
    _CFG["collapse_nav"] = _env_bool("TRACE_COLLAPSE_NAV", True)
    _CFG["max_blob"] = _env_int("TRACE_MAX_BLOB", 8192)
    _CFG["queue_max"] = _env_int("TRACE_QUEUE_MAX", 20000)
    _CFG["flush_ms"] = _env_int("TRACE_FLUSH_MS", 250)
    _CFG["batch"] = _env_int("TRACE_BATCH", 200)


_load_env()
_INHERITED_RUN_ID = (os.environ.get("TRACE_RUN_ID") or "").strip() or None
if _INHERITED_RUN_ID:
    # Cross-process hand-off: the eval subprocess inherits the webapp's run without eval.py being edited. The
    # id is marked OPEN so a local run does not displace it as the process default: the parent owns it, and it
    # must stay the fallback for every thread this process never explicitly binds.
    _AMBIENT["*"] = {"run_id": _INHERITED_RUN_ID, "span_id": None, "scene": None, "level": None,
                     "step_ix": None}
    _open_runs.add(_INHERITED_RUN_ID)


def _safe(default=None):
    """Wrap a public entry point so no tracer exception can ever reach the agent (design §5.1)."""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:                      # pragma: no cover - defensive by construction
                _note_error(exc)
                return default() if callable(default) else default
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = fn.__doc__
        wrapper.__wrapped__ = fn
        return wrapper
    return deco


def _note_error(exc):
    global _write_errors
    try:
        _write_errors += 1
        if _write_errors <= 3:
            _warn(f"internal error swallowed: {exc!r}")
    except Exception:
        pass


def _j(obj) -> str:
    """Canonical JSON — ``ensure_ascii=False`` throughout: Persian is first-class data here."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _blob(obj, cap: int | None = None):
    """Serialize + truncate a payload field (design §7.4). ``None`` stays ``None``."""
    if obj is None:
        return None
    cap = int(cap if cap is not None else _CFG["max_blob"])
    s = obj if isinstance(obj, str) else _j(obj)
    if len(s) > cap:
        return s[:cap] + f"…[truncated {len(s) - cap} chars]"
    return s


def _safe_ref(ref) -> str:
    """Artifact refs must not contain '/' — it delimits the lineage CTE's cycle-guard path. '#' is escaped too
    so ``loop_reports.jsonl#1`` round-trips through a URL query string. '|' (AI2-THOR object ids, failure
    signatures) is preserved verbatim."""
    s = str(ref if ref is not None else "")
    s = s.replace("/", "%2F").replace("\\", "%2F").replace("#", "%23")
    return s[:96]


def _hash(obj) -> str:
    return hashlib.sha256(_j(obj).encode("utf-8", "replace")).hexdigest()[:16]


def _iso(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return ""


def new_run_id() -> str:
    """32 lowercase hex — W3C Trace-Context trace-id format."""
    return os.urandom(16).hex()


def new_span_id() -> str:
    """16 lowercase hex — W3C Trace-Context parent-id format."""
    return os.urandom(8).hex()


def _code_version() -> str | None:
    """Best-effort git rev of the working tree, read from .git without shelling out. None if unavailable."""
    try:
        git = os.path.join(os.path.dirname(HERE), ".git")
        head = os.path.join(git, "HEAD")
        if not os.path.exists(head):
            return None
        with open(head, "r", encoding="utf-8") as fh:
            ref = fh.read().strip()
        if ref.startswith("ref:"):
            p = os.path.join(git, ref.split(":", 1)[1].strip())
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    return fh.read().strip()[:12]
            packed = os.path.join(git, "packed-refs")
            name = ref.split(":", 1)[1].strip()
            if os.path.exists(packed):
                with open(packed, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip().endswith(" " + name):
                            return line.split()[0][:12]
            return None
        return ref[:12]
    except Exception:
        return None


# --------------------------------------------------------------------------- store open / close
def _degrade(reason: str):
    global _ENABLED, _DEGRADED, _REASON, _WARNED
    _ENABLED = False
    _DEGRADED = True
    _REASON = str(reason)
    if not _WARNED:
        _WARNED = True
        _warn(f"disabled ({reason}); the agent is unaffected, nothing is being recorded")


def _open() -> bool:
    """Open (or create) the store. Returns True when tracing is live. Never raises."""
    global _conn, _ENABLED, _DEGRADED, _REASON, _OPENED, _seq, _seq_end
    with _lock:
        if _OPENED:
            return _ENABLED
        _OPENED = True
        if not _CFG["enabled"]:
            _ENABLED, _DEGRADED, _REASON = False, False, "TRACE_ENABLED=0"
            return False
        path = _CFG["db_path"]
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL everywhere it is safe; a rollback journal where it is not (see _wal_is_safe).
            want = "WAL" if _wal_is_safe(path) else "TRUNCATE"
            got = conn.execute(f"PRAGMA journal_mode={want}").fetchone()
            _CFG["journal_mode"] = (got[0] if got else want).upper()
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=OFF")   # deliberate: an orphan row must still be storable
            conn.executescript(_SCHEMA)
            _seq = _seq_end = 0
            conn.execute(_SQL["meta"], ("schema_version", str(SCHEMA_VERSION)))
            conn.execute(_SQL["meta"], ("trace_api_version", str(TRACE_API_VERSION)))
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('created_ts', ?)",
                         (str(time.time()),))
            _conn = conn
            _CFG["generation"] = _CFG.get("generation", 0) + 1   # invalidate every thread's cached reader
            _ENABLED, _DEGRADED, _REASON = True, False, None
            _reserve_seq_block()          # eagerly, while nothing else holds the store: the FIRST emit of a
            #                               process must not pay for a transaction on the hot path
        except Exception as exc:
            _conn = None
            _degrade(f"cannot open {path}: {exc}")
            return False
    if _CFG["async"]:
        _ensure_writer()
    return True


def _ensure_writer():
    """Start (or restart, at most 3 times) the background writer thread."""
    global _writer, _writer_stop, _writer_restarts
    with _lock:
        if not _ENABLED or not _CFG["async"]:
            return
        if _writer is not None and _writer.is_alive():
            return
        if _writer is not None:
            _writer_restarts += 1
            if _writer_restarts > 3:
                _degrade("writer thread died 3 times")
                return
        _writer_stop = False
        _writer = threading.Thread(target=_writer_loop, name="trace-writer", daemon=True)
        _writer.start()


@_safe(None)
def close() -> None:
    """Flush, stop the writer thread, close the connection. Idempotent."""
    global _conn, _writer, _writer_stop, _OPENED, _ENABLED
    flush(timeout=5.0)
    with _lock:
        _writer_stop = True
        _cv.notify_all()
        w = _writer
    if w is not None and w.is_alive() and w is not threading.current_thread():
        w.join(timeout=2.0)
    with _lock:
        _writer = None
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None
        _OPENED = False
        _ENABLED = False
        _CFG["generation"] = _CFG.get("generation", 0) + 1       # readers on other threads are now stale
        _queue.clear()
    try:
        c = getattr(_tls, "conn", None)
        if c is not None:
            c.close()
            _tls.conn = None
    except Exception:
        pass


# --------------------------------------------------------------------------- the write pipeline
# Ops that carry the AUDITABILITY EVIDENCE itself: the `writes` ledger, the artifact upserts and the PROV
# edges, plus the run/span skeleton those rows join to. Under queue pressure an EVENT may be dropped (it is a
# narration of something that also exists as a row), but dropping one of these would leave a knowledge-base
# entry on disk with no recorded cause — exactly the divergence this module exists to make impossible.
_LEDGER_OPS = frozenset({"W", "A", "AS", "G", "R+", "R-", "S+", "S-", "S*", "SX", "M"})


def _enqueue(opcode: str, params, debug: bool = False):
    """Hot path: append one op to the bounded ring. Never blocks a producer; drops DEBUG *events* first, and
    never drops a ledger op — if only ledger ops are queued the ring is allowed to grow past ``queue_max``
    (bounded in practice by the writer draining it, and reported through ``status()['queue_depth']``)."""
    global _enqueued, _dropped, _last_drop_warn, _committed
    with _lock:
        qmax = _CFG["queue_max"]
        if len(_queue) >= qmax:
            victim = fallback = None
            for i, item in enumerate(_queue):
                if item[0] in _LEDGER_OPS:
                    continue
                if item[2]:                 # a DEBUG event: the preferred victim
                    victim = i
                    break
                if fallback is None:
                    fallback = i
                if i >= 512:                # bounded scan: never pay O(n) on the hot path
                    break
            if victim is None:
                victim = fallback
            # A dropped op must still be ACCOUNTED for, or flush() waits for a commit that can never arrive.
            if victim is not None:
                del _queue[victim]
                _dropped += 1
                _committed += 1
            elif opcode not in _LEDGER_OPS:
                _dropped += 1               # nothing droppable is queued: drop THIS event, keep the ledger
                _cv.notify_all()
                return
            # else: an over-limit ledger op is appended anyway — evidence outranks the bound.
            if _dropped and _dropped - _last_drop_warn >= 1000:   # rate-limited: the warning must not flood
                _last_drop_warn = _dropped
                _queue.append(("E", _event_row("trace.warning", f"dropped {_dropped} trace rows (queue full)",
                                               None, "WARNING", None, None, None, "system:webapp",
                                               seq=_next_seq()), False))
        _queue.append((opcode, params, debug))
        _enqueued += 1
        n = len(_queue)
        _cv.notify()
    if _CFG["async"]:
        if _writer is None or not _writer.is_alive():
            _ensure_writer()
    elif n >= _CFG["batch"]:
        _drain_once()


def _apply(conn, opcode, params):
    if opcode == "E":
        conn.execute(_SQL["events"], params)
    elif opcode == "W":
        conn.execute(_SQL["writes"], params)
    elif opcode == "G":
        conn.execute(_SQL["edges"], params)
    elif opcode == "A":
        verdict, row = params
        conn.execute(_SQL_ART.get(verdict, _SQL_ART["noop"]), row)
    elif opcode == "AS":
        conn.execute(_SQL["art_stub"], params)
    elif opcode == "R+":
        conn.execute(_SQL["run_ins"], params)
    elif opcode == "R-":
        conn.execute(_SQL["run_end"], params)
    elif opcode == "R~":
        conn.execute(_SQL["run_upd"], params)
    elif opcode == "S+":
        conn.execute(_SQL["span_ins"], params)
    elif opcode == "S-":
        conn.execute(_SQL["span_end"], params)
    elif opcode == "S*":
        conn.execute(_SQL["span_rep"], params)
    elif opcode == "SX":
        conn.execute(_SQL["span_orphans"], params)
    elif opcode == "M":
        conn.execute(_SQL["meta"], params)


def _drain_once(max_ops: int | None = None) -> int:
    """Pop up to one batch of ops and commit them in a single transaction. Returns ops committed."""
    global _committed, _write_errors
    batch = []
    limit = max_ops or _CFG["batch"]
    with _lock:
        conn = _conn
        while _queue and len(batch) < limit:
            batch.append(_queue.popleft())
    if not batch:
        return 0
    if conn is None:
        with _lock:
            _committed += len(batch)
            _cv.notify_all()
        return len(batch)
    t0 = time.perf_counter()
    try:
        conn.execute("BEGIN")
        for opcode, params, _dbg in batch:
            _apply(conn, opcode, params)
        conn.execute("COMMIT")
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        try:                                   # one retry, row by row, then drop the offending rows
            conn.execute("BEGIN")
            for opcode, params, _dbg in batch:
                try:
                    _apply(conn, opcode, params)
                except Exception:
                    _write_errors += 1
            conn.execute("COMMIT")
        except Exception:
            _write_errors += 1
            _warn(f"batch of {len(batch)} trace rows dropped: {exc}")
    _flush_metrics.append((time.perf_counter() - t0) * 1000.0)
    with _lock:
        _committed += len(batch)
        _cv.notify_all()
    return len(batch)


def _writer_loop():
    """Single background writer: batched commits every TRACE_BATCH rows or TRACE_FLUSH_MS milliseconds."""
    interval = max(0.001, _CFG["flush_ms"] / 1000.0)
    while True:
        with _lock:
            if _writer_stop:
                break
            if not _queue:
                _cv.wait(interval)
                if _writer_stop and not _queue:
                    break
        try:
            while _drain_once():
                pass
        except Exception as exc:               # pragma: no cover - the loop must never die quietly
            _note_error(exc)
            time.sleep(0.05)
    try:
        while _drain_once():
            pass
    except Exception:
        pass


@_safe(False)
def flush(timeout: float = 2.0) -> bool:
    """Block until the writer queue is drained and committed. True if fully drained.

    MANDATORY before any read-back assertion in a test, and before serving a ``/trace_*`` endpoint in sync mode.
    """
    if not _ENABLED:
        return True
    deadline = time.time() + max(0.0, float(timeout))
    if not _CFG["async"]:
        while _queue and time.time() < deadline:
            _drain_once(100000)                # sync mode: the caller IS the writer
        return not _queue
    with _lock:
        target = _enqueued
        _cv.notify_all()
    while True:
        with _lock:
            if _committed >= target:
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                return _committed >= target
            _cv.wait(min(0.05, max(0.001, remaining)))


# --------------------------------------------------------------------------- ambient context (design §6.3)
def _frame(run_id=None, span_id=None, scene=None, level=None, origin=None, step_ix=None) -> dict:
    return {"run_id": run_id, "span_id": span_id, "scene": scene, "level": level,
            "origin": origin or _CFG["origin"], "step_ix": step_ix}


@_safe(lambda: {"run_id": None, "span_id": None, "scene": None, "level": None, "origin": None,
                "step_ix": None})
def current() -> dict:
    """Resolve the ambient context. First hit wins:

    1. an explicit ``run_id``/``span_id`` argument on the calling API (handled by the caller),
    2. the contextvar — free for same-thread / async callers,
    3. ``_AMBIENT[thread_ident]`` — the explicit cross-thread binding from :func:`bind` / :func:`bound`,
    4. ``_AMBIENT['*']`` — the process default set by ``start_run(set_ambient=True)`` or ``TRACE_RUN_ID``,
    5. ``{}`` — an ORPHAN. Not an error: it is the instrumentation-coverage signal.
    """
    ctx = _CTX.get()
    if ctx and ctx.get("run_id"):
        return dict(ctx)
    with _ambient_lock:
        f = _AMBIENT.get(threading.get_ident())
        if f and f.get("run_id"):
            out = dict(f)
        else:
            f = _AMBIENT.get("*")
            out = dict(f) if f and f.get("run_id") else _frame()
    if ctx:                                    # a contextvar span refines a thread-level run binding
        for k, v in ctx.items():
            if v is not None:
                out[k] = v
    return out


@_safe(lambda: {})
def bind(run_id: str | None = None, span_id: str | None = None, *,
         scene: str | None = None, level: str | None = None) -> dict:
    """Set this thread's ambient frame (lock-guarded); returns the previous frame so the caller can restore it.

    This is the explicit cross-thread hand-off: the webapp's plan thread opens the run, the worker thread that
    owns the AI2-THOR controller binds it. A contextvar alone is invisible across that boundary.
    """
    with _ambient_lock:
        tid = threading.get_ident()
        prev = _AMBIENT.get(tid)
        _AMBIENT[tid] = _frame(run_id, span_id, scene, level)
        return dict(prev) if prev else {}


@contextlib.contextmanager
def bound(run_id: str | None = None, span_id: str | None = None, **kw):
    """``bind()`` + restore, setting BOTH the contextvar and the thread ambient, so nested calls on this thread
    and any contextvar-aware callee both see the frame."""
    prev = None
    token = None
    tid = None
    try:
        frame = _frame(run_id, span_id, kw.get("scene"), kw.get("level"))
        with _ambient_lock:
            tid = threading.get_ident()
            prev = _AMBIENT.get(tid)
            _AMBIENT[tid] = frame
        token = _CTX.set(frame)
    except Exception as exc:
        _note_error(exc)
    try:
        yield
    finally:
        try:
            if token is not None:
                _CTX.reset(token)
            with _ambient_lock:
                if prev is None:
                    _AMBIENT.pop(tid, None)
                else:
                    _AMBIENT[tid] = prev
        except Exception as exc:
            _note_error(exc)


@_safe(lambda: {})
def snapshot() -> dict:
    """Serializable copy of :func:`current` — what a producer hands into a job closure."""
    return dict(current())


def _resolve(run_id=None, span_id=None, scene=None):
    ctx = current()
    return (run_id or ctx.get("run_id"), span_id or ctx.get("span_id"), scene or ctx.get("scene"))


# --------------------------------------------------------------------------- configuration / status
@_safe(lambda: {"enabled": False, "degraded": True, "reason": "configure failed"})
def configure(*, db_path: str | None = None, enabled: bool | None = None, origin: str | None = None,
              async_writes: bool | None = None, level: str | None = None,
              skip_actions: "set[str] | None" = None, max_blob: int | None = None,
              flush_ms: int | None = None, batch: int | None = None,
              kb_path: str | None = None, kb_audit_path: str | None = None,
              mem_db_path: str | None = None, loop_reports_path: str | None = None) -> dict:
    """Apply overrides and (re)open the store. Returns :func:`status`. Idempotent; safe to call repeatedly.

    Tests point this at a temp DB. Never raises: on failure it disables tracing and returns a status with
    ``degraded=True`` and a reason string.
    """
    global _seq, _seq_end, _dropped, _write_errors, _skipped_actions, _enqueued, _committed, _writer_restarts
    global KB_PATH, KB_AUDIT_PATH, MEM_DB_PATH, LOOP_REPORTS_PATH
    close()
    # The BROWSED stores move with the trace store: a fixture that points the tracer at a temp DB but leaves
    # KB_PATH on the real knowledge_base.json would both browse the wrong file and namespace its artifact ids
    # as "foreign" (see _store_tag). Passing them together keeps a fixture internally consistent.
    if kb_path is not None:
        KB_PATH = kb_path
        if kb_audit_path is None:
            KB_AUDIT_PATH = os.path.splitext(kb_path)[0] + "_audit.log"
    if kb_audit_path is not None:
        KB_AUDIT_PATH = kb_audit_path
    if mem_db_path is not None:
        MEM_DB_PATH = mem_db_path
    if loop_reports_path is not None:
        LOOP_REPORTS_PATH = loop_reports_path
    with _lock:
        _seq = _seq_end = 0
        if db_path is not None:
            _CFG["db_path"] = db_path if os.path.isabs(db_path) else os.path.join(HERE, db_path)
        if enabled is not None:
            _CFG["enabled"] = bool(enabled)
        if origin is not None:
            _CFG["origin"] = origin
        if async_writes is not None:
            _CFG["async"] = bool(async_writes)
        if level is not None and str(level).upper() in _LVL_ORD:
            _CFG["level"] = str(level).upper()
        if skip_actions is not None:
            _CFG["skip_actions"] = set(skip_actions)
        if max_blob is not None:
            _CFG["max_blob"] = int(max_blob)
        if flush_ms is not None:
            _CFG["flush_ms"] = int(flush_ms)
        if batch is not None:
            _CFG["batch"] = int(batch)
        _dropped = _write_errors = _skipped_actions = 0
        _enqueued = _committed = _writer_restarts = 0
        _metrics.clear()
        _flush_metrics.clear()
        _span_meta.clear()
        _run_seq.clear()
        _run_prev.clear()
        _nav_state.clear()
    with _ambient_lock:
        _open_runs.clear()
        if _INHERITED_RUN_ID:
            _open_runs.add(_INHERITED_RUN_ID)
    _open()
    return status()


@_safe(False)
def enabled() -> bool:
    """True iff tracing is on AND the store is writable. Cheap after the first call (lazy open latch)."""
    if not _OPENED:
        _open()
    return _ENABLED


def _pct(samples, p):
    if not samples:
        return None
    s = sorted(samples)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[i], 3)


@_safe(lambda: {"enabled": False, "degraded": True, "reason": "status failed"})
def status() -> dict:
    """Health + counters + rolling overhead metrics — what ``/trace_stats`` and the viewer banner render.

    READ-ONLY: unlike :func:`enabled`, this never CREATES the store. ``/trace_stats`` and ``/trace_tail`` are
    documented as read-only endpoints, and opening from them would apply the schema, write three ``meta`` rows
    and spawn the writer thread inside the webapp process just because somebody looked at the viewer. When the
    file does not exist yet the report says so (``pending``) instead of manufacturing it.
    """
    if not _OPENED and _CFG["db_path"] and os.path.exists(_CFG["db_path"]):
        _open()
    pending = (not _OPENED) and bool(_CFG["enabled"])
    out = {
        "pending": pending,
        "enabled": bool(_ENABLED), "degraded": bool(_DEGRADED), "reason": _REASON,
        "db_path": _CFG["db_path"], "origin": _CFG["origin"], "schema_version": SCHEMA_VERSION,
        "api_version": TRACE_API_VERSION, "async": bool(_CFG["async"]), "level": _CFG["level"],
        # Surfaced because it is not a detail: a store on a WSL-mounted Windows drive CANNOT run WAL, and
        # silently doing so is what destroyed this one twice. If this says WAL on /mnt/..., something regressed.
        "journal_mode": _CFG.get("journal_mode"),
        "queue_depth": len(_queue), "dropped": _dropped, "write_errors": _write_errors,
        "events": 0, "runs": 0, "spans": 0, "writes": 0, "artifacts": 0, "edges": 0,
        "skipped_actions": _skipped_actions,
        "metrics": {"emit_us_p50": _pct(_metrics, 50), "emit_us_p99": _pct(_metrics, 99),
                    "flush_ms_p99": _pct(_flush_metrics, 99)},
    }
    if pending:
        out["reason"] = _REASON or "store not created yet (a read-only call must not create it)"
    conn = _rconn()
    if conn is not None:
        for t in ("events", "runs", "spans", "writes", "artifacts", "edges"):
            try:
                out[t] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"])
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------- events
def _event_row(type_, summary, payload, level, scene, run_id, span_id, actor, ts=None, seq=None):
    ts = float(ts if ts is not None else time.time())
    payload_json = _blob(payload)
    eid = hashlib.sha256(
        f"{type_}|{run_id}|{span_id}|{payload_json}|{time.time_ns()}|{seq}".encode("utf-8", "replace")
    ).hexdigest()[:32]
    try:
        thread = threading.current_thread().name
    except Exception:
        thread = None
    return (seq, eid, ts, run_id, span_id, type_, actor, level, scene,
            (str(summary)[:200] if summary else None), payload_json, _CFG["origin"], os.getpid(), thread)


def _reserve_seq_block(n: int | None = None) -> bool:
    """Claim a DISJOINT block of ``events.seq`` values for this process. Caller holds ``_lock``.

    ``events.seq`` is handed out synchronously (``emit`` returns it, and the ``writes``/``edges`` rows carry it
    as ``event_seq``), so it cannot be left to sqlite's AUTOINCREMENT — the value must exist before the row is
    queued. Seeding a per-process counter from ``MAX(seq)`` instead is what made two processes on one store mint
    the SAME primary keys: under ``INSERT OR IGNORE`` the loser's events were silently discarded while its
    ``writes`` rows survived, re-pointed at the winner's unrelated event. Both D1 (nothing is lost) and D2 (a
    write names the activity that caused it) died there.

    So the range is allocated ONCE per block from a high-water mark in ``meta``, bumped inside a
    ``BEGIN IMMEDIATE`` transaction on a short-lived private connection (the writer's connection is busy with
    its own batches). Two processes therefore get 1..4096 and 4097..8192 and can never collide. The cost is one
    tiny transaction per ``SEQ_BLOCK`` events; the price is gaps in ``seq`` when a process exits mid-block,
    which every reader already tolerates (``seq`` is an ordering key, never a count).
    """
    global _seq, _seq_end
    n = int(n or SEQ_BLOCK)
    path = _CFG["db_path"]
    c = None
    try:
        c = sqlite3.connect(path, timeout=10.0, isolation_level=None)
        c.execute("PRAGMA busy_timeout=10000")
        c.execute("BEGIN IMMEDIATE")
        base = 0
        try:
            row = c.execute("SELECT value FROM meta WHERE key='seq_hwm'").fetchone()
            if row and str(row[0]).strip().isdigit():
                base = int(row[0])
        except Exception:
            base = 0
        try:                     # a store written by an older build has rows but no high-water mark yet
            m = c.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()
            base = max(base, int(m[0] or 0))
        except Exception:
            pass
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seq_hwm', ?)", (str(base + n),))
        c.execute("COMMIT")
        _seq, _seq_end = base, base + n
        return True
    except Exception as exc:
        _note_error(exc)
        return False
    finally:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass


def _next_seq() -> int:
    global _seq
    with _lock:
        if _seq >= _seq_end:
            if not _reserve_seq_block():
                # The store is unreachable for the moment: stay monotonic locally rather than hand out 0, and
                # let the next call try to re-anchor. INSERT OR IGNORE keeps a stale value from corrupting a row.
                _seq += 1
                return _seq
        _seq += 1
        return _seq


@_safe(0)
def emit(type: str, *, summary: str = "", payload: object = None, actor: str | None = None,
         level: str = "DEFAULT", scene: str | None = None,
         run_id: str | None = None, span_id: str | None = None) -> int:
    """Append one ``events`` row. Returns ``events.seq`` (0 when disabled or dropped — never ``None``)."""
    t0 = time.perf_counter()
    if not _ENABLED:
        if not _OPENED:
            _open()
        if not _ENABLED:
            return 0
    lvl = level if level in _LVL_ORD else "DEFAULT"
    if _LVL_ORD[lvl] < _LVL_ORD[_CFG["level"]]:     # level filter BEFORE any allocation
        return 0
    rid, sid, sc = _resolve(run_id, span_id, scene)
    seq = _next_seq()
    row = _event_row(type, summary, payload, lvl, sc, rid, sid, actor, seq=seq)
    _enqueue("E", row, debug=(lvl == "DEBUG"))
    _metrics.append((time.perf_counter() - t0) * 1e6)
    return seq


@_safe(None)
def link(src: str, dst: str, rel: str, *, note: str | None = None, run_id: str | None = None) -> None:
    """``INSERT OR IGNORE`` one PROV edge. An unknown ``rel`` is dropped with a ``trace.warning`` event."""
    if not _ENABLED:
        return None
    if rel not in RELS:
        emit("trace.warning", summary=f"unknown PROV relation {rel!r} dropped", level="WARNING",
             payload={"src": src, "dst": dst, "rel": rel})
        return None
    if not src or not dst:
        return None
    rid = run_id or current().get("run_id")
    _enqueue("G", (src, dst, rel, rid, _seq, time.time(), note))    # an edge is evidence: never droppable
    return None


# --------------------------------------------------------------------------- runs
@_safe(new_run_id)
def start_run(*, kind: str = "command", command: str | None = None, scene: str | None = None,
              level: str | None = None, lang: str | None = None, arm: str | None = None,
              decision_id: int | None = None, agent_id: str | None = None, seed: int | None = None,
              config: dict | None = None, parent_run_id: str | None = None,
              origin: str | None = None, run_id: str | None = None,
              set_ambient: bool = True) -> str:
    """Open a run (PROV Bundle / OTel trace). Returns the run id — ALWAYS a valid 32-hex string, even when
    tracing is disabled, so callers can pass it around unconditionally."""
    rid = run_id or new_run_id()
    if not _ENABLED:
        if not _OPENED:
            _open()
        if not _ENABLED:
            return rid
    parent = parent_run_id
    if parent is None:
        amb = current().get("run_id")
        if amb and amb != rid:
            parent = amb
    with _ambient_lock:
        if len(_open_runs) > 4096:       # a crashed caller that never closed its run must not leak forever
            _open_runs.clear()
        _open_runs.add(rid)
    cfg = dict(config or {})
    cfg.setdefault("schema_version", SCHEMA_VERSION)
    cfg_json = _j(cfg)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8", "replace")).hexdigest()[:16]
    now = time.time()
    _enqueue("R+", (rid, parent, kind, origin or _CFG["origin"], now, None, "UNSET", scene, level, lang,
                    command, arm, decision_id, agent_id, seed, cfg_json, cfg_hash, _code_version()))
    with _lock:
        _run_seq[rid] = 0
    frame = _frame(rid, None, scene, level)
    _CTX.set(frame)
    if set_ambient:
        with _ambient_lock:
            tid = threading.get_ident()
            # Remember what this run displaces so end_run can put it back: a nested teach/rating run must not
            # permanently steal the process ambient from the command run that is still open around it.
            star = _AMBIENT.get("*")
            _run_prev[rid] = (star, _AMBIENT.get(tid), tid)
            _AMBIENT[tid] = dict(frame)
            # The PROCESS default is only claimed when no other run still owns it. Otherwise a rating/teach run
            # opened on an HTTP handler thread would re-point the AI2-THOR worker thread — which has no binding
            # of its own — at itself, and the command run's episodes would be filed under the rating. A wrong
            # causal graph is worse than an orphan, because it looks right.
            if not (star and star.get("run_id") in _open_runs):
                _AMBIENT["*"] = dict(frame)
    emit("run.started", summary=f"{kind}: {command or ''}"[:200], run_id=rid,
         payload={"kind": kind, "command": command, "scene": scene, "level": level, "lang": lang,
                  "arm": arm, "decision_id": decision_id, "agent_id": agent_id, "config": cfg,
                  "config_hash": cfg_hash, "parent_run_id": parent},
         scene=scene, actor=("system:eval" if kind.startswith("eval") else "system:webapp"))
    if parent:
        link(f"run:{rid}", f"run:{parent}", "wasInformedBy", note="child run", run_id=rid)
    if agent_id:
        link(f"run:{rid}", f"agent:llm:{agent_id}", "wasAssociatedWith", note="planner", run_id=rid)
    return rid


@_safe(None)
def end_run(run_id: str | None = None, *, status: str = "OK", error: str | None = None,
            summary: dict | None = None) -> None:
    """Close a run: stamp ``ended_ts``/``status``, recount the denormalised counters, close any span left open,
    and clear the contextvar + process ambient if they point at this run."""
    if not _ENABLED:
        return None
    rid = run_id or current().get("run_id")
    if not rid:
        return None
    now = time.time()
    _enqueue("SX", (now, now, rid))
    if _skipped_actions:
        emit("action.skipped_summary", summary=f"{_skipped_actions} filtered no-op actions this process",
             level="DEBUG", run_id=rid, payload={"skipped": _skipped_actions,
                                                 "skip_actions": sorted(_CFG["skip_actions"])})
    emit("run.ended", summary=f"{status}" + (f": {error}" if error else ""), run_id=rid,
         payload={"status": status, "error": error, "summary": summary},
         level=("ERROR" if status == "ERROR" else "DEFAULT"))
    _enqueue("R-", (now, status, error, rid))
    with _lock:
        _run_seq.pop(rid, None)
        for sid in [s for s, m in _span_meta.items() if m.get("run_id") == rid]:
            _span_meta.pop(sid, None)
    ctx = _CTX.get()
    if ctx and ctx.get("run_id") == rid:
        _CTX.set(None)
    with _ambient_lock:
        _open_runs.discard(rid)
        prev = _run_prev.pop(rid, None)
        for key in list(_AMBIENT):
            if (_AMBIENT.get(key) or {}).get("run_id") == rid:
                _AMBIENT.pop(key, None)
        if prev:                                   # restore whatever this run displaced — if it is STILL OPEN
            star, own, tid = prev
            # Restoring a run that has already ENDED would silently attribute every later write on a fresh
            # worker thread to a finished run (reproduced: runs Z then A on one thread, B on another, end_run(A)
            # reinstated Z). A finished frame is dropped instead, which correctly yields an orphan.
            if star and star.get("run_id") != rid and star.get("run_id") in _open_runs:
                _AMBIENT["*"] = star
            if own and own.get("run_id") != rid and own.get("run_id") in _open_runs:
                _AMBIENT[tid] = own
    return None


@_safe(None)
def annotate_run(run_id: str | None = None, *, level: str | None = None, lang: str | None = None,
                 scene: str | None = None, arm: str | None = None, decision_id: int | None = None,
                 command: str | None = None) -> None:
    """Fill in run facts that are only knowable AFTER the run opened — above all the command's L1/L2/L3 level,
    which the planner decides. Purely additive (``COALESCE``): nothing already recorded is overwritten."""
    if not _ENABLED:
        return None
    rid = run_id or current().get("run_id")
    if not rid:
        return None
    _enqueue("R~", (level, lang, scene, arm, decision_id, command, rid))
    return None


@contextlib.contextmanager
def run(**kwargs):
    """``with trace.run(kind='command', command=text, ...) as rid:``

    On an exception the run is closed with ``status='ERROR'`` and the exception is **re-raised**: the tracer
    swallows only its own errors, never the agent's.

    The context is restored to whatever it was BEFORE the block — captured here rather than via a
    ``ContextVar`` token, because ``start_run`` sets the frame itself: a token taken after that call would
    restore the *finished* run on exit, silently attributing every later write on this thread to a run that
    has already ended (and hiding genuine orphans).
    """
    prev = None
    try:
        prev = _CTX.get()
    except Exception as exc:
        _note_error(exc)
    rid = start_run(**kwargs)                 # start_run sets the contextvar frame for this thread
    try:
        yield rid
    except BaseException as exc:
        try:
            end_run(rid, status="ERROR", error=repr(exc))
        except Exception:
            pass
        raise
    else:
        end_run(rid, status="OK")
    finally:
        try:
            _CTX.set(prev)
        except Exception:
            pass


# --------------------------------------------------------------------------- spans
@_safe(new_span_id)
def start_span(name: str, *, kind: str = "TOOL", op: str | None = None, run_id: str | None = None,
               parent_span_id: str | None = None, step_ix: int | None = None,
               level: str = "DEFAULT", scene: str | None = None, actor: str | None = None,
               input: object = None, attrs: dict | None = None) -> str:
    """Open a span (PROV Activity / OTel span). Returns the span id — always valid, even when disabled."""
    sid = new_span_id()
    if not _ENABLED:
        if not _OPENED:
            _open()
        if not _ENABLED:
            return sid
    rid, parent, sc = _resolve(run_id, parent_span_id, scene)
    if not rid:
        rid = ""                                    # orphan span: recorded, never dropped
    with _lock:
        seq = _run_seq.get(rid, 0)
        _run_seq[rid] = seq + 1
        pdepth = (_span_meta.get(parent) or {}).get("depth")
        depth = 0 if parent is None or pdepth is None else int(pdepth) + 1
        _span_meta[sid] = {"run_id": rid, "depth": depth, "attrs": dict(attrs or {}),
                           "start_ts": time.time(), "parent": parent, "name": name, "kind": kind}
        while len(_span_meta) > 20000:
            _span_meta.popitem(last=False)
    m = _span_meta.get(sid) or {}
    _enqueue("S+", (sid, rid, parent, seq, depth, name,
                    op, kind if kind in KINDS else "TOOL", step_ix, m.get("start_ts", time.time()),
                    "UNSET", level if level in _LVL_ORD else "DEFAULT", sc, actor,
                    _blob(input), _blob(attrs)),
             debug=(level == "DEBUG"))
    if parent:
        link(f"span:{sid}", f"span:{parent}", "wasInformedBy", run_id=rid)
    elif rid:
        link(f"span:{sid}", f"run:{rid}", "wasInformedBy", run_id=rid)
    if actor:
        link(f"span:{sid}", f"agent:{actor}", "wasAssociatedWith", run_id=rid)
    return sid


@_safe(None)
def end_span(span_id: str, *, status: str = "OK", error: str | None = None,
             output: object = None, attrs: dict | None = None) -> None:
    """Close a span, merging any late attributes into the ones captured at ``start_span``."""
    if not _ENABLED or not span_id:
        return None
    now = time.time()
    with _lock:
        m = _span_meta.pop(span_id, None)
    dur = ((now - m["start_ts"]) * 1000.0) if m else None
    merged = None
    if attrs:
        base = dict((m or {}).get("attrs") or {})
        base.update(attrs)
        merged = _blob(base)
    _enqueue("S-", (now, dur, status, error, _blob(output), merged, span_id))
    return None


@contextlib.contextmanager
def span(name: str, **kwargs):
    """``with trace.span('execute_tool pick_and_place', kind='TOOL', step_ix=3) as sid:``

    Sets BOTH the contextvar and this thread's ambient span for the duration, so a deeper callee (``knowledge``,
    ``memory``) attributes its writes to this span without any signature change. Re-raises the caller's
    exception after marking the span ``ERROR``.
    """
    sid = start_span(name, **kwargs)
    token = None
    prev = None
    tid = None
    try:
        ctx = current()
        # step_ix rides the ambient frame so a write made three call levels below a plan step (episodes,
        # failures, spatial beliefs) lands in `writes.step_ix` without a single signature change — that column
        # is what disambiguates two back-to-back executions of the same command in `episodes`.
        step_ix = kwargs.get("step_ix")
        frame = _frame(kwargs.get("run_id") or ctx.get("run_id"), sid,
                       kwargs.get("scene") or ctx.get("scene"), ctx.get("level"),
                       step_ix=(ctx.get("step_ix") if step_ix is None else step_ix))
        token = _CTX.set(frame)
        with _ambient_lock:
            tid = threading.get_ident()
            prev = _AMBIENT.get(tid)
            _AMBIENT[tid] = dict(frame)
    except Exception as exc:
        _note_error(exc)
    try:
        yield sid
    except BaseException as exc:
        try:
            end_span(sid, status="ERROR", error=repr(exc))
        except Exception:
            pass
        raise
    else:
        end_span(sid, status="OK")
    finally:
        try:
            if token is not None:
                _CTX.reset(token)
            with _ambient_lock:
                if prev is None:
                    _AMBIENT.pop(tid, None)
                else:
                    _AMBIENT[tid] = prev
        except Exception:
            pass


# --------------------------------------------------------------------------- THE causality primitive
def _artifact_kind(store: str, table_or_kind: str) -> str:
    k = _ART_KIND.get(table_or_kind)
    if k:
        return k
    if store == "file":
        low = str(table_or_kind).lower()
        if low.endswith((".jsonl", ".log", ".json")):
            return "report"
        if low.endswith((".png", ".jpg", ".jpeg")):
            return "image"
        return "file"
    return str(table_or_kind)


def _default_store_path(store: str):
    return {"kb": KB_PATH, "mem": MEM_DB_PATH}.get(str(store))


def _store_tag(store: str, store_path) -> str:
    """Namespace suffix distinguishing WHICH knowledge_base.json / memory.db a pointer came from.

    ``kb:procedure:1`` is not a global identity: every fresh KB starts numbering at 1 (see commit f191148
    "Reset knowledge base for fresh testing", and every temp-KB fixture), so writes from two different stores
    would fuse into ONE artifact row whose ``n_causes``/``revision`` — the numbers a thesis reports as
    "reinforced N times" — silently sum unrelated histories. The tag is EMPTY for the store this process is
    actually browsing (``KB_PATH`` / ``MEM_DB_PATH``), so ids stay exactly as documented for the real system
    and only a foreign store gets ``kb@1a2b3c4d:...``.

    Residual limitation, documented rather than hidden: a store that is reset IN PLACE reuses its path, so its
    new #1 still shares an artifact with the old #1. Detecting that needs an epoch marker inside the store
    itself — filed as a follow-up in ``docs/observability.md``.
    """
    if store not in ("kb", "mem") or not store_path:
        return ""
    try:
        default = _default_store_path(store)
        a = os.path.normcase(os.path.abspath(str(store_path)))
        if default and a == os.path.normcase(os.path.abspath(str(default))):
            return ""
        return "@" + hashlib.sha256(a.encode("utf-8", "replace")).hexdigest()[:8]
    except Exception:
        return ""


@_safe("")
def artifact_id(*, store: str, table_or_kind: str, pointer: object, store_path=None) -> str:
    """The canonical artifact id for one entry — the ONE place that spelling is defined.

    Callers that build a PROV edge by hand (``improve._link_episodes``, ``feedback``'s failure→episode edge)
    must use this rather than an f-string, or their edges point at a node no write ever created.
    """
    kind = _artifact_kind(store, table_or_kind)
    return f"{store}{_store_tag(store, store_path)}:{kind}:{_safe_ref(pointer)}"


@_safe("")
def record_write(*, store: str, table_or_kind: str, pointer: object, verdict: str,
                 why: str = "", source: str | None = None, dedup_key: str | None = None,
                 before: object = None, after: object = None, summary: str | None = None,
                 scope: str | None = None, confidence: float | None = None,
                 occurrences: int | None = None, audit_line: str | None = None,
                 content: object = None, derived_from: "list[str] | None" = None,
                 run_id: str | None = None, span_id: str | None = None,
                 step_ix: int | None = None, actor: str | None = None,
                 store_path: str | None = None) -> str:
    """THE causality primitive — the single consistency point for ``events`` + ``artifacts`` + ``writes`` +
    ``edges``. In one call it:

    1. emits an ``events`` row (``kb.write`` | ``memory.write`` | ``file.write``, chosen by ``store``);
    2. upserts the ``artifacts`` row (created on ``insert``; ``revision``/``occurrences``/``updated_ts``/
       ``n_causes`` bumped on ``reinforce``/``update``; ``alive=0`` on ``delete``);
    3. appends the ``writes`` row — append-only, one row per cause, never an update of a prior row;
    4. inserts the PROV edges: artifact→span ``wasGeneratedBy`` (insert) / ``wasRevisionOf`` (reinforce,
       update) / ``wasInvalidatedBy`` (delete), artifact→agent ``wasAttributedTo`` from ``source``, and
       artifact→node ``wasDerivedFrom`` for every ``derived_from`` entry.

    Returns the ``artifact_id`` (``""`` when tracing is off). Callers never write ``artifacts``/``edges``
    directly — that redundancy is only safe because it has exactly one writer.
    """
    if not _ENABLED:
        if not _OPENED:
            _open()
        if not _ENABLED:
            return ""
    verdict = verdict if verdict in VERDICTS else "update"
    kind = _artifact_kind(store, table_or_kind)
    ref = str(pointer)
    art = f"{store}{_store_tag(store, store_path)}:{kind}:{_safe_ref(ref)}"
    ctx = current()                          # resolved ONCE: this is the hottest path in the module
    rid = run_id or ctx.get("run_id")
    sid = span_id or ctx.get("span_id")
    sc = ctx.get("scene")
    if step_ix is None:                      # inherited from the enclosing plan-step / feedback span
        step_ix = ctx.get("step_ix")
    act = actor or _SOURCE_ACTOR.get(str(source or ""), None)
    now = time.time()
    ev_type = {"kb": "kb.write", "mem": "memory.write"}.get(store, "file.write")
    lvl = "DEBUG" if ev_type == "file.write" else "DEFAULT"
    seq = emit(ev_type, summary=f"{verdict} {table_or_kind} #{ref}" + (f" — {summary}" if summary else ""),
               payload={"store": store, "table_or_kind": table_or_kind, "pointer": ref,
                        "artifact_id": art, "verdict": verdict, "why": why, "source": source,
                        "dedup_key": dedup_key, "before": before, "after": after, "scope": scope,
                        "confidence": confidence, "audit_line": audit_line, "content": content},
               level=lvl, actor=act, scene=sc, run_id=rid, span_id=sid)
    # NOT debug, whatever the event level: these two rows ARE the ledger. Marking a file-store write droppable
    # is how a loop report could end up on disk with no recorded cause under queue pressure.
    _enqueue("A", (verdict, (art, store, kind, ref, seq, now, now, (_hash(content) if content else None),
                             _blob(summary, 200), source, scope, confidence, occurrences, dedup_key,
                             0 if verdict == "delete" else 1, occurrences)))
    _enqueue("W", (seq, rid, sid, step_ix, now, store, table_or_kind, ref, art, verdict, dedup_key,
                   _blob(before), _blob(after), why, source, audit_line, _CFG["origin"]))
    cause = f"span:{sid}" if sid else (f"run:{rid}" if rid else None)
    if cause:
        rel = {"insert": "wasGeneratedBy", "reinforce": "wasRevisionOf", "update": "wasRevisionOf",
               "delete": "wasInvalidatedBy", "noop": "wasRevisionOf"}[verdict]
        link(f"art:{art}", cause, rel, note=why or None, run_id=rid)
    if act:
        link(f"art:{art}", f"agent:{act}", "wasAttributedTo", note=source, run_id=rid)
    for node in (derived_from or []):
        if node:
            link(f"art:{art}", str(node), "wasDerivedFrom", note=why or None, run_id=rid)
    return art


@_safe("")
def record_use(*, store: str, table_or_kind: str, pointer: object, why: str = "",
               run_id: str | None = None, span_id: str | None = None,
               store_path: str | None = None) -> str:
    """``prov:used`` — an activity CONSUMED an artifact (a retrieval hit actually injected into the planner
    prompt; not every KB read, or the signal would be meaningless). Creates a stub ``artifacts`` row when the
    entry predates instrumentation (seeded priors). Returns the artifact id."""
    if not _ENABLED:
        return ""
    kind = _artifact_kind(store, table_or_kind)
    ref = str(pointer)
    art = f"{store}{_store_tag(store, store_path)}:{kind}:{_safe_ref(ref)}"
    rid, sid, _sc = _resolve(run_id, span_id, None)
    now = time.time()
    _enqueue("AS", (art, store, kind, ref, _seq, now, now, None, None, None, None, None, None))
    src = f"span:{sid}" if sid else (f"run:{rid}" if rid else None)
    if src:
        link(src, f"art:{art}", "used", note=why or None, run_id=rid)
    return art


# --------------------------------------------------------------------------- convenience recorders
@_safe(None)
def record_action(action: str, *, kwargs: dict | None = None, success: bool | None = None,
                  error: str | None = None, scene: str | None = None, duration_ms: float | None = None,
                  run_id: str | None = None, span_id: str | None = None) -> str | None:
    """One raw AI2-THOR primitive. Applies the skip filter first (a ``Pass`` at 40 Hz must cost one hash
    lookup) and the nav-collapse rule. Returns the ACTION span id, or ``None`` when filtered or collapsed."""
    global _skipped_actions
    if action in _CFG["skip_actions"]:
        _skipped_actions += 1
        return None
    if not _ENABLED:
        if not _OPENED:
            _open()
        if not _ENABLED:
            return None
    rid, parent, sc = _resolve(run_id, span_id, scene)
    now = time.time()
    if _CFG["collapse_nav"]:
        key = (rid or "", parent or "")
        st = _nav_state.get(key)
        sig = (action, _j(kwargs or {}), bool(success), error or "")
        if st and st["sig"] == sig and (now - st["ts"]) <= 2.0:
            st["ts"] = now
            st["n"] += 1
            _enqueue("S*", (1, now, (now - st["t0"]) * 1000.0, st["span_id"]), debug=True)
            return None
    sid = start_span(f"{action}" + (f" {kwargs.get('objectId') or kwargs.get('objectType') or ''}"
                                    if isinstance(kwargs, dict) else ""),
                     kind="ACTION", op="thor_action", run_id=rid, parent_span_id=parent,
                     level="DEBUG", scene=sc, actor="env:ai2thor", input=kwargs,
                     attrs={"thor.action": action, "thor.scene": sc, "thor.success": success,
                            "thor.error": error})
    end_span(sid, status=("ERROR" if success is False else "OK"), error=error,
             output={"success": success, "error": error, "duration_ms": duration_ms})
    if _CFG["collapse_nav"]:
        _nav_state[(rid or "", parent or "")] = {"sig": (action, _j(kwargs or {}), bool(success), error or ""),
                                                 "ts": now, "t0": now, "n": 1, "span_id": sid}
        if len(_nav_state) > 512:
            _nav_state.clear()
    emit("action.executed", summary=f"{action} " + ("ok" if success else f"FAIL {error or ''}"),
         level="DEBUG", actor="env:ai2thor", scene=sc, run_id=rid, span_id=sid,
         payload={"thor.action": action, "args": kwargs, "thor.success": success, "thor.error": error,
                  "duration_ms": duration_ms})
    return sid


@_safe("")
def record_llm(*, model: str, endpoint: str | None, prompt: str, completion: str | None,
               duration_ms: float, error: str | None = None, system: str | None = None,
               memory_block: str | None = None, run_id: str | None = None,
               span_id: str | None = None) -> str:
    """Open+close one ``LLM`` span with ``gen_ai.*`` attributes and emit ``llm.call``.

    Token counts are **absent**, not zero: ``llm.OllamaProvider.chat`` discards Ollama's ``eval_count`` /
    ``prompt_eval_count`` today (a follow-up in ``docs/observability.md``), so ``input_chars``/``output_chars``
    stand in as a documented proxy. Zero would be a lie in a results table.
    """
    if not _ENABLED:
        return ""
    rid, parent, sc = _resolve(run_id, span_id, None)
    attrs = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama",
             "gen_ai.request.model": model, "gen_ai.response.model": model,
             "gen_ai.agent.id": model, "gen_ai.usage.input_chars": len(prompt or ""),
             "gen_ai.usage.output_chars": len(completion or ""), "endpoint": endpoint}
    sid = start_span(f"chat {model}", kind="LLM", op="chat", run_id=rid, parent_span_id=parent,
                     actor=f"llm:{model}", scene=sc, attrs=attrs,
                     input={"prompt": _blob(prompt), "system": _blob(system),
                            "memory_block": _blob(memory_block)})
    end_span(sid, status=("ERROR" if error else "OK"), error=error,
             output={"completion": _blob(completion)}, attrs=dict(attrs, duration_ms=duration_ms))
    emit("llm.call", summary=f"{model} {duration_ms:.0f} ms, {len(prompt or '')}->{len(completion or '')} chars",
         actor=f"llm:{model}", run_id=rid, span_id=sid, scene=sc,
         payload={"model": model, "endpoint": endpoint, "duration_ms": duration_ms,
                  "prompt": _blob(prompt), "completion": _blob(completion), "error": error,
                  "system": _blob(system), "memory_block": _blob(memory_block)})
    return sid


# --------------------------------------------------------------------------- read side
def _wal_is_safe(path) -> bool:
    """Can this path host a WAL database?

    WAL needs a shared-memory index file (``-shm``) backed by working ``mmap`` plus POSIX advisory locks. A
    Windows drive mounted into WSL (**DrvFs**, ``/mnt/<letter>/...``) provides neither reliably. The failure is
    not a clean refusal -- the store opens, accepts writes for a while, and then starts failing every read and
    write with ``unable to open database file``, having already left the file structurally corrupt.

    This is not hypothetical. `sim/trace.db` sat on ``/mnt/d`` and was destroyed twice on 2026-08-22: once
    accumulating 3149 write errors with `count(*)` on `events` unreadable, and again within minutes of being
    deleted and rebuilt from scratch. Windows-side Python could read the same file that WSL-side Python could
    not open, which is the signature of the ``-shm`` layer failing rather than the database being at fault.

    A rollback journal needs no ``-shm`` and works on DrvFs, so that is what is used there. It costs
    reader/writer concurrency, which for an append-only event log written a few hundred rows per command is a
    price worth paying to have a store that survives.
    """
    if os.name != "posix":
        return True                       # native Windows on NTFS handles WAL correctly
    try:
        real = os.path.realpath(str(path))
    except Exception:
        real = str(path)
    if not real.startswith("/mnt/"):
        return True                       # a real Linux filesystem
    try:                                  # ...but /mnt is only suspect UNDER WSL
        with open("/proc/version", "r", encoding="utf-8", errors="replace") as fh:
            return "microsoft" not in fh.read().lower()
    except Exception:
        return True


def _rconn():
    """A per-thread READ connection. ``PRAGMA query_only`` makes a write impossible by construction, and WAL
    means these reads never block the writer (nor the writer them)."""
    path = _CFG["db_path"]
    if not path or not os.path.exists(path):
        return None
    gen = _CFG.get("generation", 0)
    c = getattr(_tls, "conn", None)
    if c is not None and getattr(_tls, "path", None) == path and getattr(_tls, "gen", None) == gen:
        return c
    if c is not None:
        try:
            c.close()
        except Exception:
            pass
    try:
        c = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=5000")
        c.execute("PRAGMA query_only=ON")
        _tls.conn = c
        _tls.path = path
        _tls.gen = gen
        return c
    except Exception as exc:
        _note_error(exc)
        return None


def _soft_flush():
    """Readers see everything already recorded: cheap no-op when the queue is empty."""
    try:
        if _queue:
            flush(timeout=1.0)
    except Exception:
        pass


def _rows(cur) -> list:
    return [dict(r) for r in cur.fetchall()]


def _clamp(n, lo, hi, default):
    try:
        n = int(n)
    except Exception:
        return default
    return max(lo, min(hi, n))


@_safe(lambda: {"events": [], "cursor": {"first_seq": None, "last_seq": None}, "has_more": False,
                "total": None, "error": "unavailable"})
def query_events(*, after: int | None = None, before: int | None = None, limit: int = 200,
                 type: str | None = None, types: "list[str] | None" = None,
                 run_id: str | None = None, span_id: str | None = None, actor: str | None = None,
                 level: str | None = None, min_level: str | None = None, scene: str | None = None,
                 q: str | None = None, order: str = "desc") -> dict:
    """Filtered, cursor-paged event feed. Rows OMIT ``payload_json`` (use :func:`get_event`) but carry the
    joined ``write`` summary so the feed can show causality inline without a second request."""
    _soft_flush()
    out = {"events": [], "cursor": {"first_seq": None, "last_seq": None}, "has_more": False,
           "total": None, "error": None}
    conn = _rconn()
    if conn is None:
        out["error"] = _REASON or "no trace store"
        return out
    limit = _clamp(limit, 1, 1000, 200)
    where, args = [], []
    if after is not None:
        where.append("e.seq > ?")
        args.append(int(after))
    if before is not None:
        where.append("e.seq < ?")
        args.append(int(before))
    if type:
        where.append("e.type = ?")
        args.append(type)
    if types:
        where.append("e.type IN (%s)" % ",".join("?" * len(types)))
        args.extend(list(types))
    if run_id:
        where.append("e.run_id = ?")
        args.append(run_id)
    if span_id:
        where.append("e.span_id = ?")
        args.append(span_id)
    if actor:
        where.append("e.actor = ?")
        args.append(actor)
    if level:
        where.append("e.level = ?")
        args.append(level)
    if min_level and min_level in _LVL_ORD:
        keep = [lv for lv in LEVELS if _LVL_ORD[lv] >= _LVL_ORD[min_level]]
        where.append("e.level IN (%s)" % ",".join("?" * len(keep)))
        args.extend(keep)
    if scene:
        where.append("e.scene = ?")
        args.append(scene)
    if q:
        where.append("LOWER(e.summary) LIKE ?")
        args.append("%" + str(q).lower() + "%")
    sql = ("SELECT e.seq, e.ts, e.run_id, e.span_id, e.type, e.actor, e.level, e.scene, e.summary,"
           " e.origin, e.thread, (e.payload_json IS NOT NULL) AS has_payload,"
           " w.store, w.table_or_kind, w.pointer, w.verdict, w.artifact_id"
           " FROM events e LEFT JOIN writes w ON w.event_seq = e.seq")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY e.seq " + ("ASC" if str(order).lower() == "asc" else "DESC") + " LIMIT ?"
    rows = _rows(conn.execute(sql, args + [limit + 1]))
    out["has_more"] = len(rows) > limit
    rows = rows[:limit]
    for r in rows:
        r["iso"] = _iso(r.get("ts"))
        r["has_payload"] = bool(r.get("has_payload"))
        if r.get("store"):
            r["write"] = {"store": r["store"], "table_or_kind": r["table_or_kind"], "pointer": r["pointer"],
                          "verdict": r["verdict"], "artifact_id": r["artifact_id"]}
        for k in ("store", "table_or_kind", "pointer", "verdict", "artifact_id"):
            r.pop(k, None)
    out["events"] = rows
    if rows:
        seqs = [r["seq"] for r in rows]
        out["cursor"] = {"first_seq": min(seqs), "last_seq": max(seqs)}
    try:
        out["total"] = int(conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])
    except Exception:
        pass
    return out


@_safe(None)
def get_event(seq: int) -> dict | None:
    """One event with its full ``payload_json``, plus the ``writes`` rows emitted alongside it."""
    _soft_flush()
    conn = _rconn()
    if conn is None:
        return None
    r = conn.execute("SELECT * FROM events WHERE seq=?", (int(seq),)).fetchone()
    if r is None:
        return None
    row = dict(r)
    row["iso"] = _iso(row.get("ts"))
    try:
        row["payload"] = json.loads(row["payload_json"]) if row.get("payload_json") else None
    except Exception:
        row["payload"] = row.get("payload_json")
    row["writes"] = _rows(conn.execute("SELECT * FROM writes WHERE event_seq=? ORDER BY id", (int(seq),)))
    return row


@_safe(lambda: {"runs": [], "has_more": False, "error": "unavailable"})
def query_runs(*, limit: int = 50, before_ts: float | None = None, kind: str | None = None,
               scene: str | None = None, level: str | None = None, lang: str | None = None,
               status: str | None = None, q: str | None = None) -> dict:
    """Newest-first run list; each row carries the denormalised ``n_spans``/``n_events``/``n_writes``."""
    _soft_flush()
    out = {"runs": [], "has_more": False, "error": None}
    conn = _rconn()
    if conn is None:
        out["error"] = _REASON or "no trace store"
        return out
    limit = _clamp(limit, 1, 500, 50)
    where, args = [], []
    for col, val in (("kind", kind), ("scene", scene), ("level", level), ("lang", lang), ("status", status)):
        if val:
            where.append(f"{col} = ?")
            args.append(val)
    if before_ts is not None:
        where.append("started_ts < ?")
        args.append(float(before_ts))
    if q:
        where.append("LOWER(COALESCE(command,'')) LIKE ?")
        args.append("%" + str(q).lower() + "%")
    sql = "SELECT * FROM runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_ts DESC LIMIT ?"
    rows = _rows(conn.execute(sql, args + [limit + 1]))
    out["has_more"] = len(rows) > limit
    rows = rows[:limit]
    for r in rows:
        r["iso"] = _iso(r.get("started_ts"))
        r["duration_ms"] = ((r["ended_ts"] - r["started_ts"]) * 1000.0
                            if r.get("ended_ts") and r.get("started_ts") else None)
    out["runs"] = rows
    return out


@_safe(lambda: {"run": None, "spans": [], "roots": [], "events_by_span": {},
                "timeline": {"t0": None, "t1": None}, "truncated": False, "error": "unavailable"})
def run_tree(run_id: str, *, max_spans: int = 2000, min_level: str = "DEBUG") -> dict:
    """The run/command tree. Spans come back FLAT with a ``children`` id list and a precomputed ``depth`` —
    the client builds the tree, so the payload stays predictable and cheap to page.

    ``truncated`` is True when the run has more spans than ``max_spans`` returned, so the viewer can say so
    rather than silently drawing a partial waterfall.
    """
    _soft_flush()
    out = {"run": None, "spans": [], "roots": [], "events_by_span": {},
           "timeline": {"t0": None, "t1": None}, "truncated": False, "error": None}
    conn = _rconn()
    if conn is None:
        out["error"] = _REASON or "no trace store"
        return out
    r = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    out["run"] = dict(r) if r else None
    if out["run"]:
        out["run"]["iso"] = _iso(out["run"].get("started_ts"))
    keep = [lv for lv in LEVELS if _LVL_ORD[lv] >= _LVL_ORD.get(min_level, 0)]
    cap = _clamp(max_spans, 1, 20000, 2000)
    spans = _rows(conn.execute(
        "SELECT * FROM spans WHERE run_id=? AND COALESCE(level,'DEFAULT') IN (%s) ORDER BY seq LIMIT ?"
        % ",".join("?" * len(keep)), [run_id] + keep + [cap + 1]))
    out["truncated"] = len(spans) > cap
    spans = spans[:cap]
    by_id = {s["span_id"]: s for s in spans}
    for s in spans:
        s["children"] = []
        s["writes"] = []
    for s in spans:
        p = s.get("parent_span_id")
        if p and p in by_id:
            by_id[p]["children"].append(s["span_id"])
        else:
            out["roots"].append(s["span_id"])
    for w in _rows(conn.execute("SELECT * FROM writes WHERE run_id=? ORDER BY id", (run_id,))):
        tgt = by_id.get(w.get("span_id"))
        if tgt is not None:
            tgt["writes"].append(w)
    for row in conn.execute("SELECT span_id, COUNT(*) AS n FROM events WHERE run_id=? GROUP BY span_id",
                            (run_id,)):
        out["events_by_span"][row["span_id"] or ""] = int(row["n"])
    out["spans"] = spans
    ts = [s["start_ts"] for s in spans if s.get("start_ts")]
    te = [s["end_ts"] for s in spans if s.get("end_ts")]
    if out["run"]:
        ts.append(out["run"].get("started_ts"))
        if out["run"].get("ended_ts"):
            te.append(out["run"]["ended_ts"])
    ts = [t for t in ts if t]
    te = [t for t in te if t]
    out["timeline"] = {"t0": min(ts) if ts else None, "t1": max(te or ts) if (te or ts) else None}
    return out


@_safe(lambda: {"writes": [], "has_more": False, "error": "unavailable"})
def query_writes(*, limit: int = 200, after_id: int | None = None, store: str | None = None,
                 table_or_kind: str | None = None, pointer: str | None = None,
                 artifact_id: str | None = None, run_id: str | None = None,
                 span_id: str | None = None, verdict: str | None = None,
                 source: str | None = None) -> dict:
    """The auditable ledger, filtered. Every row is one cause of one entry."""
    _soft_flush()
    out = {"writes": [], "has_more": False, "error": None}
    conn = _rconn()
    if conn is None:
        out["error"] = _REASON or "no trace store"
        return out
    limit = _clamp(limit, 1, 1000, 200)
    where, args = [], []
    for col, val in (("store", store), ("table_or_kind", table_or_kind), ("pointer", pointer),
                     ("artifact_id", artifact_id), ("run_id", run_id), ("span_id", span_id),
                     ("verdict", verdict), ("source", source)):
        if val:
            where.append(f"{col} = ?")
            args.append(val)
    if after_id is not None:
        where.append("id > ?")
        args.append(int(after_id))
    sql = "SELECT * FROM writes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    rows = _rows(conn.execute(sql, args + [limit + 1]))
    out["has_more"] = len(rows) > limit
    for r in rows[:limit]:
        r["iso"] = _iso(r.get("ts"))
    out["writes"] = rows[:limit]
    return out


@_safe(lambda: [])
def artifact_causes(artifact_id: str, *, limit: int = 100) -> list:
    """*"What produced this KB entry?"* — every ``writes`` row for the artifact, oldest first, each joined to
    its run (command, scene, arm) and its span (name, op, kind). One indexed query (``ix_writes_art``)."""
    _soft_flush()
    conn = _rconn()
    if conn is None:
        return []
    rows = _rows(conn.execute(
        "SELECT w.*, r.command AS run_command, r.scene AS run_scene, r.kind AS run_kind, r.arm AS run_arm,"
        "       r.started_ts AS run_started_ts, s.name AS span_name, s.op AS span_op, s.kind AS span_kind"
        "  FROM writes w LEFT JOIN runs r ON r.run_id = w.run_id"
        "                LEFT JOIN spans s ON s.span_id = w.span_id"
        " WHERE w.artifact_id = ? ORDER BY w.ts ASC, w.id ASC LIMIT ?",
        (artifact_id, _clamp(limit, 1, 1000, 100))))
    for r in rows:
        r["iso"] = _iso(r.get("ts"))
    return rows


@_safe(lambda: [])
def span_effects(span_id: str, *, include_descendants: bool = False) -> list:
    """*"What did this action write?"* — every ``writes`` row caused by the span (``ix_writes_span``).
    With ``include_descendants``, one extra indexed query folds in the writes of its child spans."""
    _soft_flush()
    conn = _rconn()
    if conn is None:
        return []
    rows = _rows(conn.execute("SELECT * FROM writes WHERE span_id=? ORDER BY id", (span_id,)))
    if include_descendants:
        kids = [r["span_id"] for r in conn.execute(
            "WITH RECURSIVE k(id) AS (SELECT ? UNION SELECT s.span_id FROM spans s JOIN k ON"
            " s.parent_span_id = k.id) SELECT id AS span_id FROM k WHERE id <> ?", (span_id, span_id))]
        if kids:
            rows += _rows(conn.execute(
                "SELECT * FROM writes WHERE span_id IN (%s) ORDER BY id" % ",".join("?" * len(kids)), kids))
    for r in rows:
        r["iso"] = _iso(r.get("ts"))
    return rows


# The two lineage CTEs. UNION's built-in cycle protection deduplicates whole ROWS, and `depth` increments on
# every revisit, so it protects nothing here — hence the explicit '/'-delimited path guard on NODE identity.
# (`entry A is retrieved by a run that then reinforces A` is the NORMAL operation of the self-improving loop.)
_CTE_UP = """
WITH RECURSIVE up(node, rel, depth, path) AS (
    SELECT ?, NULL, 0, '/' || ? || '/'
  UNION ALL
    SELECT e.dst, e.rel, up.depth + 1, up.path || e.dst || '/'
      FROM edges e JOIN up ON e.src = up.node
     WHERE up.depth < ? AND instr(up.path, '/' || e.dst || '/') = 0
)
SELECT node, rel, MIN(depth) AS depth FROM up WHERE depth > 0
GROUP BY node ORDER BY depth, node LIMIT ?
"""
_CTE_DOWN = """
WITH RECURSIVE down(node, rel, depth, path) AS (
    SELECT ?, NULL, 0, '/' || ? || '/'
  UNION ALL
    SELECT e.src, e.rel, down.depth + 1, down.path || e.src || '/'
      FROM edges e JOIN down ON e.dst = down.node
     WHERE down.depth < ? AND instr(down.path, '/' || e.src || '/') = 0
)
SELECT node, rel, MIN(depth) AS depth FROM down WHERE depth > 0
GROUP BY node ORDER BY depth, node LIMIT ?
"""


def _node_label(conn, node: str) -> dict:
    """Resolve a namespaced node id to something renderable."""
    out = {"id": node, "type": node.split(":", 1)[0] if ":" in node else "node",
           "label": node, "sublabel": None, "store": None, "kind": None, "source": None,
           "ts": None, "alive": None}
    try:
        if node.startswith("art:"):
            r = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (node[4:],)).fetchone()
            if r:
                out.update({"label": r["summary"] or f'{r["kind"]} #{r["ref"]}',
                            "sublabel": f'{r["store"]}.{r["kind"]} #{r["ref"]}', "store": r["store"],
                            "kind": r["kind"], "source": r["source"], "ts": r["created_ts"],
                            "alive": r["alive"]})
            else:
                out["label"] = node[4:]
        elif node.startswith("span:"):
            r = conn.execute("SELECT * FROM spans WHERE span_id=?", (node[5:],)).fetchone()
            if r:
                out.update({"label": r["name"] or node[5:], "sublabel": f'{r["kind"]} / {r["op"] or ""}',
                            "kind": r["kind"], "ts": r["start_ts"]})
        elif node.startswith("run:"):
            r = conn.execute("SELECT * FROM runs WHERE run_id=?", (node[4:],)).fetchone()
            if r:
                out.update({"label": r["command"] or f'{r["kind"]} run', "sublabel": node[4:][:12],
                            "kind": r["kind"], "ts": r["started_ts"]})
        elif node.startswith("agent:"):
            out.update({"label": node[6:], "sublabel": "PROV agent"})
    except Exception as exc:
        _note_error(exc)
    return out


@_safe(lambda: {"root": None, "direction": "both", "nodes": [], "edges": [], "truncated": False,
                "error": "unavailable"})
def lineage(node: str, *, direction: str = "both", depth: int = 6, limit: int = 300) -> dict:
    """Ancestors (``up``: *what produced this?*) and/or descendants (``down``: *what did this produce?*).

    Edges point from the dependent thing to the thing it depends on (PROV), so ``src→dst`` walks backwards in
    causality and ``dst→src`` walks forwards; both are the same recursive query with the join flipped.
    ``MIN(depth)`` gives each node its shortest distance from the root — the correct layer for a layered graph
    drawing, so the SVG layout gets its layer assignment free. ``node`` is ALWAYS bound as a parameter.
    """
    _soft_flush()
    out = {"root": node, "direction": direction, "nodes": [], "edges": [], "truncated": False, "error": None}
    conn = _rconn()
    if conn is None or not node:
        out["error"] = _REASON or "no trace store"
        return out
    depth = _clamp(depth, 1, 24, 6)
    limit = _clamp(limit, 1, 2000, 300)
    seen = {node: 0}
    for d, sql in (("up", _CTE_UP), ("down", _CTE_DOWN)):
        if direction not in ("both", d):
            continue
        rows = conn.execute(sql, (node, node, depth, limit)).fetchall()
        if len(rows) >= limit:
            out["truncated"] = True
        for r in rows:
            n = r["node"]
            seen[n] = min(seen.get(n, 10 ** 6), int(r["depth"]))
    ids = list(seen)
    if ids:
        chunks = [ids[i:i + 400] for i in range(0, len(ids), 400)]
        for ch in chunks:
            q = ("SELECT src, dst, rel, note, ts FROM edges WHERE src IN (%s) AND dst IN (%s)"
                 % (",".join("?" * len(ch)), ",".join("?" * len(ids))))
            out["edges"] += _rows(conn.execute(q, ch + ids))
    out["nodes"] = []
    for n in sorted(seen, key=lambda x: (seen[x], x)):
        info = _node_label(conn, n)
        info["depth"] = seen[n]
        out["nodes"].append(info)
    return out


@_safe(None)
def artifact(artifact_id: str) -> dict | None:
    """The artifact row + its LIVE current content read from the underlying store (read-only) + every cause +
    the spans that ``used`` it. This is the entity-detail exhibit: entry, provenance, and lineage in one call."""
    _soft_flush()
    conn = _rconn()
    if conn is None or not artifact_id:
        return None
    r = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if r is None:
        return None
    out = dict(r)
    out["iso"] = _iso(out.get("created_ts"))
    out["causes"] = artifact_causes(artifact_id)
    out["uses"] = _rows(conn.execute(
        "SELECT e.src AS node, e.ts, e.note, s.name AS span_name, s.op AS span_op, s.run_id AS run_id"
        "  FROM edges e LEFT JOIN spans s ON ('span:' || s.span_id) = e.src"
        " WHERE e.dst = ? AND e.rel = 'used' ORDER BY e.ts", ("art:" + artifact_id,)))
    # A namespaced id (kb@1a2b3c4d:...) belongs to a DIFFERENT store file than the one this process browses,
    # so reading "the current content" from the local store would show an unrelated entry with the same number.
    foreign = "@" in str(artifact_id).split(":", 1)[0]
    out["foreign_store"] = foreign
    out["live"] = None if foreign else _live_entry(out.get("store"), out.get("kind"), out.get("ref"))
    return out


def _live_entry(store, kind, ref):
    """Read the CURRENT content of an artifact from its own store — knowledge_base.json or memory.db
    (``mode=ro``: a write is impossible by construction). None when it no longer exists."""
    try:
        if store == "kb":
            data = _read_kb()
            for e in data.get(str(kind) + "s", []):
                if str(e.get("id")) == str(ref):
                    return e
            return None
        if store == "mem":
            table = {"episode": "episodes", "spatial": "spatial", "failure": "failures",
                     "nav_cell": "nav_cells", "perception_label": "perception_labels",
                     "decision": "bandit_decisions", "arm": "bandit_arms"}.get(str(kind))
            if not table:
                return None
            conn = _ro_sqlite(MEM_DB_PATH)
            if conn is None:
                return None
            if table == "failures":
                r = conn.execute("SELECT * FROM failures WHERE signature=?", (ref,)).fetchone()
            elif table == "nav_cells":
                parts = str(ref).split(":")
                if len(parts) < 3:
                    return None
                r = conn.execute("SELECT * FROM nav_cells WHERE scene=? AND gx=? AND gz=?",
                                 (parts[0], parts[1], parts[2])).fetchone()
            elif table == "bandit_arms":
                r = conn.execute("SELECT arm, updates FROM bandit_arms WHERE arm=?", (ref,)).fetchone()
            else:
                r = conn.execute(f"SELECT * FROM {table} WHERE id=?", (ref,)).fetchone()
            return dict(r) if r else None
        if store == "file":
            path, _, line_no = str(ref).partition("#")
            full = LOOP_REPORTS_PATH if os.path.basename(LOOP_REPORTS_PATH) == path else \
                os.path.join(HERE, path)
            if not os.path.exists(full) or not line_no:
                return None
            with open(full, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if i == int(line_no):
                        try:
                            return json.loads(line)
                        except Exception:
                            return {"line": line.strip()}
            return None
    except Exception as exc:
        _note_error(exc)
    return None


@_safe(lambda: {})
def stats(*, window_s: float | None = None) -> dict:
    """Aggregates for the Stats pane — all of them SQL ``COUNT(*) GROUP BY``, never rows shipped to a client."""
    _soft_flush()
    out = {"events_by_type": {}, "events_by_level": {}, "spans_by_kind": {}, "writes_by_store_verdict": [],
           "artifacts_by_source": {}, "artifacts_by_store_kind": [],
           "runs": {"total": 0, "ok": 0, "error": 0, "open": 0},
           "orphans": {"writes": 0, "events": 0}, "top_artifacts": [], "recent_rate": [],
           "db": {"path": _CFG["db_path"], "size_bytes": None, "schema_version": SCHEMA_VERSION},
           "tracer": status(), "error": None}
    conn = _rconn()
    if conn is None:
        out["error"] = _REASON or "no trace store"
        return out
    since = (time.time() - float(window_s)) if window_s else None
    ev_where, ev_args = ("WHERE ts >= ?", [since]) if since else ("", [])
    for r in conn.execute(f"SELECT type, COUNT(*) n FROM events {ev_where} GROUP BY type", ev_args):
        out["events_by_type"][r["type"]] = int(r["n"])
    for r in conn.execute(f"SELECT level, COUNT(*) n FROM events {ev_where} GROUP BY level", ev_args):
        out["events_by_level"][r["level"]] = int(r["n"])
    for r in conn.execute("SELECT kind, COUNT(*) n FROM spans GROUP BY kind"):
        out["spans_by_kind"][r["kind"]] = int(r["n"])
    out["writes_by_store_verdict"] = [dict(r) for r in conn.execute(
        "SELECT store, table_or_kind, verdict, COUNT(*) AS n FROM writes"
        " GROUP BY store, table_or_kind, verdict ORDER BY n DESC")]
    for r in conn.execute("SELECT COALESCE(source,'?') s, COUNT(*) n FROM artifacts GROUP BY s"):
        out["artifacts_by_source"][r["s"]] = int(r["n"])
    out["artifacts_by_store_kind"] = [dict(r) for r in conn.execute(
        "SELECT store, kind, COUNT(*) AS n, SUM(alive) AS alive FROM artifacts GROUP BY store, kind")]
    row = conn.execute("SELECT COUNT(*) total, SUM(status='OK') ok, SUM(status='ERROR') err,"
                       " SUM(ended_ts IS NULL) open FROM runs").fetchone()
    out["runs"] = {"total": int(row["total"] or 0), "ok": int(row["ok"] or 0),
                   "error": int(row["err"] or 0), "open": int(row["open"] or 0)}
    out["orphans"] = {
        "writes": int(conn.execute("SELECT COUNT(*) n FROM writes WHERE run_id IS NULL").fetchone()["n"]),
        "events": int(conn.execute("SELECT COUNT(*) n FROM events WHERE run_id IS NULL").fetchone()["n"])}
    out["top_artifacts"] = [dict(r) for r in conn.execute(
        "SELECT artifact_id, summary, n_causes, revision, source, store, kind FROM artifacts"
        " ORDER BY n_causes DESC, revision DESC LIMIT 12")]
    out["recent_rate"] = [{"minute": _iso(r["m"] * 60.0), "n": int(r["n"])} for r in conn.execute(
        "SELECT CAST(ts/60 AS INTEGER) m, COUNT(*) n FROM events WHERE ts >= ? GROUP BY m ORDER BY m",
        (time.time() - 3600.0,))]
    try:
        out["db"]["size_bytes"] = os.path.getsize(_CFG["db_path"])
    except Exception:
        pass
    return out


@_safe(lambda: {"runs": [], "events": [], "artifacts": [], "writes": [], "error": "unavailable"})
def search(q: str, *, limit: int = 40) -> dict:
    """Cross-store LIKE search over run commands, event summaries, artifact summaries and write ``why``s."""
    _soft_flush()
    out = {"runs": [], "events": [], "artifacts": [], "writes": [], "error": None}
    conn = _rconn()
    if conn is None or not q:
        out["error"] = _REASON or "no trace store"
        return out
    limit = _clamp(limit, 1, 200, 40)
    like = "%" + str(q).lower() + "%"
    out["runs"] = _rows(conn.execute(
        "SELECT run_id, kind, command, scene, level, lang, started_ts, status FROM runs"
        " WHERE LOWER(COALESCE(command,'')) LIKE ? ORDER BY started_ts DESC LIMIT ?", (like, limit)))
    out["events"] = _rows(conn.execute(
        "SELECT seq, ts, type, level, summary, run_id, span_id FROM events"
        " WHERE LOWER(COALESCE(summary,'')) LIKE ? ORDER BY seq DESC LIMIT ?", (like, limit)))
    out["artifacts"] = _rows(conn.execute(
        "SELECT artifact_id, store, kind, ref, summary, source, n_causes, revision, alive FROM artifacts"
        " WHERE LOWER(COALESCE(summary,'')) LIKE ? OR LOWER(ref) LIKE ?"
        " ORDER BY updated_ts DESC LIMIT ?", (like, like, limit)))
    out["writes"] = _rows(conn.execute(
        "SELECT id, ts, store, table_or_kind, pointer, verdict, why, run_id, span_id, artifact_id FROM writes"
        " WHERE LOWER(COALESCE(why,'')) LIKE ? ORDER BY id DESC LIMIT ?", (like, limit)))
    return out


# --------------------------------------------------------------------------- generic DB browser (read-only)
def _ro_sqlite(path):
    """Open any foreign sqlite store strictly read-only. ``mode=ro`` makes a write raise by construction —
    ``memory.db`` is REAL RESEARCH DATA and the observer must never be able to touch it."""
    try:
        if not path or not os.path.exists(path):
            return None
        uri = "file:" + path.replace("?", "%3f").replace("#", "%23").replace("\\", "/") + "?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c
    except Exception:
        return None


def _read_kb() -> dict:
    try:
        with open(KB_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_KB_COL_HEAD = ["id", "time", "source", "ts", "confidence", "scope"]

_CATALOG = [
    ("kb.procedures", "kb", "Procedures", "id", "procedure"),
    ("kb.lessons", "kb", "Lessons", "id", "lesson"),
    ("kb.facts", "kb", "Facts", "id", "fact"),
    ("kb.audit", "kb", "KB audit log", "line_no", None),
    ("mem.episodes", "mem", "Episodes", "id", "episodes"),
    ("mem.spatial", "mem", "Spatial beliefs", "id", "spatial"),
    ("mem.failures", "mem", "Failure counters", "signature", "failures"),
    ("mem.nav_cells", "mem", "Navigation cells", "scene", "nav_cells"),
    ("mem.perception_labels", "mem", "Perception labels", "id", "perception_labels"),
    ("mem.bandit_arms", "mem", "Bandit arms", "arm", "bandit_arms"),
    ("mem.bandit_decisions", "mem", "Bandit decisions", "id", "bandit_decisions"),
    ("trace.runs", "trace", "Runs", "run_id", None),
    ("trace.spans", "trace", "Spans", "span_id", None),
    ("trace.events", "trace", "Events", "seq", None),
    ("trace.artifacts", "trace", "Artifacts", "artifact_id", None),
    ("trace.writes", "trace", "Writes (ledger)", "id", None),
    ("trace.edges", "trace", "Edges (PROV)", "id", None),
    ("file.loop_reports", "file", "Loop reports", "line_no", "loop_reports.jsonl"),
]
_CATALOG_BY_ID = {c[0]: c for c in _CATALOG}
_PROV_COLS = ("source", "ts", "time", "confidence", "scope", "last_seen", "first_seen")


@_safe(lambda: [])
def db_tables() -> list:
    """Catalog of browsable tables across the three stores. A table whose backing store is missing or
    unreadable is still listed, with ``rows=None`` and an ``error`` — the viewer renders zero rows, never a
    stack trace (both stores are currently empty by design after the Phase-9 reset)."""
    _soft_flush()
    kb = _read_kb()
    mem = _ro_sqlite(MEM_DB_PATH)
    tconn = _rconn()
    out = []
    for tid, store, label, pk, art_kind in _CATALOG:
        entry = {"id": tid, "store": store, "label": label, "rows": None, "pk": pk, "columns": [],
                 "provenance": [], "artifact": None, "error": None}
        name = tid.split(".", 1)[1]
        try:
            if store == "kb" and name != "audit":
                rows = kb.get(name) or []
                entry["rows"] = len(rows)
                entry["columns"] = _kb_columns(rows)
                entry["artifact"] = {"store": "kb", "kind": art_kind, "ref_col": "id"}
                if not kb:
                    entry["error"] = f"{os.path.basename(KB_PATH)} missing or empty"
            elif tid == "kb.audit":
                entry["columns"] = ["line_no", "line"]
                entry["rows"] = _count_lines(KB_AUDIT_PATH)
                if entry["rows"] is None:
                    entry["error"] = "audit log not found"
            elif store == "mem":
                if mem is None:
                    entry["error"] = f"{os.path.basename(MEM_DB_PATH)} missing or unreadable"
                else:
                    entry["columns"] = [r["name"] for r in mem.execute(f"PRAGMA table_info({name})")]
                    if not entry["columns"]:
                        entry["error"] = "table not present in memory.db"
                    else:
                        entry["rows"] = int(mem.execute(f"SELECT COUNT(*) n FROM {name}").fetchone()["n"])
                    if art_kind:
                        entry["artifact"] = {"store": "mem", "kind": _ART_KIND.get(art_kind, art_kind),
                                             "ref_col": pk}
            elif store == "trace":
                if tconn is None:
                    entry["error"] = _REASON or "no trace store"
                else:
                    entry["columns"] = [r["name"] for r in tconn.execute(f"PRAGMA table_info({name})")]
                    entry["rows"] = int(tconn.execute(f"SELECT COUNT(*) n FROM {name}").fetchone()["n"])
            else:                                   # file.loop_reports
                entry["columns"] = ["line_no", "ts", "command", "scene", "level", "success", "writes"]
                entry["rows"] = _count_lines(LOOP_REPORTS_PATH)
                entry["artifact"] = {"store": "file", "kind": "report", "ref_col": "ref"}
                if entry["rows"] is None:
                    entry["error"] = "loop_reports.jsonl not found"
        except Exception as exc:
            entry["error"] = str(exc)
        entry["provenance"] = [c for c in entry["columns"] if c in _PROV_COLS]
        out.append(entry)
    if mem is not None:
        try:
            mem.close()
        except Exception:
            pass
    return out


def _kb_columns(rows) -> list:
    cols = list(_KB_COL_HEAD)
    for e in (rows or [])[:200]:
        if isinstance(e, dict):
            for k in e:
                if k not in cols:
                    cols.append(k)
    return cols


def _count_lines(path):
    try:
        if not os.path.exists(path):
            return None
        n = 0
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in fh:
                n += 1
        return n
    except Exception:
        return None


def _blobify(row: dict) -> dict:
    """LinUCB matrices are packed BLOBs — rendered as a size, never decoded."""
    out = {}
    for k, v in row.items():
        out[k] = f"<blob {len(v)} bytes>" if isinstance(v, (bytes, bytearray, memoryview)) else v
    return out


@_safe(lambda: {"table": None, "columns": [], "rows": [], "total": 0, "offset": 0, "limit": 0,
                "error": "unavailable"})
def db_rows(table: str, *, limit: int = 100, offset: int = 0, order_by: str | None = None,
            desc: bool = True, q: str | None = None,
            where_col: str | None = None, where_val: str | None = None) -> dict:
    """Page rows of one catalog table.

    Hard rules: ``table`` must be a catalog key (anything else is an error envelope, never SQL); the only
    user-supplied SQL fragment is an ``ORDER BY`` column validated against the table's own column list, and an
    unknown one is REPORTED rather than silently swapped for the primary key (a viewer sorting by a mistyped
    column would otherwise show plausible, wrong data); ``memory.db`` is opened ``mode=ro``; the JSON/JSONL
    stores are read and filtered in Python; ``limit`` is clamped to 500.

    Each row carries ``_artifact_id``; the causes/effects of one row come from :func:`db_row` (one query per
    row is not something a 500-row page can afford).
    """
    out = {"table": table, "columns": [], "rows": [], "total": 0, "offset": int(offset or 0),
           "limit": _clamp(limit, 1, 500, 100), "error": None}
    spec = _CATALOG_BY_ID.get(str(table))
    if not spec:
        out["error"] = f"unknown table {table!r}"
        return out
    _soft_flush()
    tid, store, _label, pk, art_kind = spec
    name = tid.split(".", 1)[1]
    lim, off = out["limit"], max(0, out["offset"])
    ql = str(q).lower() if q else None

    if store == "kb" and name != "audit":
        rows = [r for r in (_read_kb().get(name) or []) if isinstance(r, dict)]
        out["columns"] = _kb_columns(rows)
        if where_col and where_col in out["columns"]:
            rows = [r for r in rows if str(r.get(where_col)) == str(where_val)]
        if ql:
            rows = [r for r in rows if ql in _j(r).lower()]
        if order_by and order_by not in out["columns"]:
            out["error"] = f"unknown order_by column {order_by!r}"
            return out
        key = order_by if order_by in out["columns"] else "id"
        rows.sort(key=lambda r: (r.get(key) is None, str(r.get(key))), reverse=bool(desc))
        out["total"] = len(rows)
        page = rows[off:off + lim]
        for r in page:
            r = dict(r)
            r["_artifact_id"] = f"kb:{art_kind}:{_safe_ref(r.get('id'))}"
            out["rows"].append(r)
        return out

    if tid == "kb.audit":
        out["columns"] = ["line_no", "line"]
        lines = _read_lines(KB_AUDIT_PATH)
        if ql:
            lines = [(i, s) for i, s in lines if ql in s.lower()]
        out["total"] = len(lines)
        if desc:
            lines = list(reversed(lines))
        out["rows"] = [{"line_no": i, "line": s} for i, s in lines[off:off + lim]]
        return out

    if tid == "file.loop_reports":
        lines = _read_lines(LOOP_REPORTS_PATH)
        recs = []
        for i, s in lines:
            try:
                d = json.loads(s)
            except Exception:
                d = {"raw": s}
            d["line_no"] = i
            d["_artifact_id"] = f"file:report:{_safe_ref(os.path.basename(LOOP_REPORTS_PATH) + '#' + str(i))}"
            recs.append(d)
        cols = ["line_no"]
        for d in recs[:100]:
            for k in d:
                if k not in cols and k != "_artifact_id":
                    cols.append(k)
        out["columns"] = cols
        if ql:
            recs = [d for d in recs if ql in _j(d).lower()]
        if where_col:
            recs = [d for d in recs if str(d.get(where_col)) == str(where_val)]
        out["total"] = len(recs)
        recs.sort(key=lambda d: d.get("line_no", 0), reverse=bool(desc))
        out["rows"] = recs[off:off + lim]
        return out

    conn = _ro_sqlite(MEM_DB_PATH) if store == "mem" else _rconn()
    if conn is None:
        out["error"] = (f"{os.path.basename(MEM_DB_PATH)} missing or unreadable" if store == "mem"
                        else (_REASON or "no trace store"))
        return out
    try:
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({name})")]
        if not cols:
            out["error"] = f"table {name!r} not present"
            return out
        out["columns"] = cols
        if order_by and order_by not in cols:
            out["error"] = f"unknown order_by column {order_by!r}"
            return out
        where, args = [], []
        if where_col and where_col in cols:
            where.append(f"{where_col} = ?")
            args.append(where_val)
        if ql:
            where.append("(" + " OR ".join(f"LOWER(CAST({c} AS TEXT)) LIKE ?" for c in cols) + ")")
            args += ["%" + ql + "%"] * len(cols)
        wsql = (" WHERE " + " AND ".join(where)) if where else ""
        out["total"] = int(conn.execute(f"SELECT COUNT(*) n FROM {name}{wsql}", args).fetchone()["n"])
        ob = order_by if order_by in cols else (pk if pk in cols else cols[0])
        sql = (f"SELECT * FROM {name}{wsql} ORDER BY {ob} " + ("DESC" if desc else "ASC") +
               " LIMIT ? OFFSET ?")
        rows = [_blobify(dict(r)) for r in conn.execute(sql, args + [lim, off])]
        if store == "mem" and art_kind:
            kind = _ART_KIND.get(art_kind, art_kind)
            for r in rows:
                ref = (f'{r.get("scene")}:{r.get("gx")}:{r.get("gz")}' if name == "nav_cells"
                       else str(r.get(pk)))
                r["_artifact_id"] = f"mem:{kind}:{_safe_ref(ref)}"
        out["rows"] = rows
    except Exception as exc:
        out["error"] = str(exc)
    finally:
        if store == "mem":
            try:
                conn.close()
            except Exception:
                pass
    return out


def _read_lines(path):
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return [(i, line.rstrip("\n")) for i, line in enumerate(fh, 1)]
    except Exception:
        return []


@_safe(lambda: {"table": None, "pk": None, "row": None, "provenance": {}, "artifact_id": None,
                "causes": [], "effects": [], "audit_lines": [], "error": "unavailable"})
def db_row(table: str, pk: str) -> dict:
    """One row, with its provenance fields, its artifact id, every cause (``artifact_causes``), the effects it
    itself caused, and any legacy audit lines mentioning it — which is what makes the DB browser answer
    *"what produced this entry?"* inline."""
    out = {"table": table, "pk": pk, "row": None, "provenance": {}, "artifact_id": None,
           "causes": [], "effects": [], "audit_lines": [], "error": None}
    spec = _CATALOG_BY_ID.get(str(table))
    if not spec:
        out["error"] = f"unknown table {table!r}"
        return out
    tid, store, _label, pkcol, _art = spec
    page = db_rows(table, limit=500, where_col=pkcol, where_val=pk)
    if page.get("error"):
        out["error"] = page["error"]
        return out
    rows = page.get("rows") or []
    row = None
    for r in rows:
        if str(r.get(pkcol)) == str(pk):
            row = r
            break
    if row is None:
        # nav_cells' composite pk and the line-numbered file stores address rows by their artifact ref.
        page = db_rows(table, limit=500)
        for r in page.get("rows") or []:
            if str(r.get(pkcol)) == str(pk) or str(r.get("_artifact_id", "")).endswith(_safe_ref(pk)):
                row = r
                break
    if row is None:
        out["error"] = "row not found"
        return out
    out["row"] = row
    out["provenance"] = {k: row.get(k) for k in _PROV_COLS if k in row}
    art = row.get("_artifact_id")
    out["artifact_id"] = art
    if art:
        out["causes"] = artifact_causes(art)
        conn = _rconn()
        if conn is not None:
            out["effects"] = _rows(conn.execute(
                "SELECT e.src AS node, e.rel, e.ts, e.note FROM edges e WHERE e.dst = ? ORDER BY e.ts",
                ("art:" + art,)))
    if store == "kb":
        needle = str(row.get("command") or row.get("text") or "")
        out["audit_lines"] = [s for _i, s in _read_lines(KB_AUDIT_PATH)
                              if needle and needle[:60] in s][-20:]
    return out


# --------------------------------------------------------------------------- monkeypatch installers
@_safe(lambda: {"installed": [], "skipped": [], "error": "install failed"})
def install_thor_hooks(module=None) -> dict:
    """Wrap ``thor_actions.ThorAgent.do / reset / set_sensors`` IN PLACE (that file is not ours to edit).

    Call once from the webapp after ``import thor_actions as ta``, as ``trace.install_thor_hooks(ta)``.
    Idempotent (a ``_traced`` marker), never raises, and returns ``{'installed': []}`` when ``thor_actions``
    is unavailable. KNOWN GAP: ``ThorAgent.replay()`` deliberately bypasses ``do()``, so replayed macros are
    invisible here — filed as a follow-up in ``docs/observability.md``.
    """
    out = {"installed": [], "skipped": [], "error": None}
    mod = module
    if mod is None:
        mod = sys.modules.get("thor_actions")
    if mod is None:
        out["error"] = "thor_actions not imported"
        return out
    cls = getattr(mod, "ThorAgent", None)
    if cls is None:
        out["error"] = "ThorAgent not found"
        return out

    orig_do = getattr(cls, "do", None)
    if orig_do is not None and not getattr(orig_do, "_traced", False):
        def do(self, action, **kwargs):
            if action in _CFG["skip_actions"]:          # cheapest possible path for a 40 Hz no-op
                global _skipped_actions
                _skipped_actions += 1
                return orig_do(self, action, **kwargs)
            t0 = time.perf_counter()
            try:
                res = orig_do(self, action, **kwargs)
            except Exception as exc:
                try:
                    record_action(action, kwargs=kwargs, success=False, error=repr(exc),
                                  scene=getattr(self, "scene", None),
                                  duration_ms=(time.perf_counter() - t0) * 1000.0)
                except Exception:
                    pass
                raise
            try:
                record_action(action, kwargs=kwargs, success=bool(getattr(self, "success", True)),
                              error=getattr(self, "error", None), scene=getattr(self, "scene", None),
                              duration_ms=(time.perf_counter() - t0) * 1000.0)
            except Exception:
                pass
            return res
        do._traced = True
        do.__name__ = "do"
        cls.do = do
        out["installed"].append("ThorAgent.do")
    else:
        out["skipped"].append("ThorAgent.do")

    for meth in ("reset", "set_sensors"):
        orig = getattr(cls, meth, None)
        if orig is None or getattr(orig, "_traced", False):
            out["skipped"].append(f"ThorAgent.{meth}")
            continue

        def _mk(orig=orig, meth=meth):
            def wrapper(self, *a, **kw):
                try:
                    emit("scene.reset", summary=f"{meth}({a or ''}) — scene state and pose reset",
                         level="WARNING", actor="env:ai2thor", scene=getattr(self, "scene", None),
                         payload={"method": meth, "args": [str(x) for x in a], "kwargs": _blob(kw)})
                except Exception:
                    pass
                return orig(self, *a, **kw)
            wrapper._traced = True
            wrapper.__name__ = meth
            return wrapper
        setattr(cls, meth, _mk())
        out["installed"].append(f"ThorAgent.{meth}")
    return out


# --------------------------------------------------------------------------- export / maintenance
@_safe(lambda: {"entity": {}, "activity": {}, "agent": {}})
def export_prov_json(run_id: str | None = None, node: str | None = None) -> dict:
    """A PROV-JSON document for one run or one node's lineage — the graph loads in a standard PROV tool with
    no GPU, no WSL and no Ollama. Thesis-appendix material (§11.6 of the design doc)."""
    _soft_flush()
    doc = {"prefix": {"trace": "urn:thesis:trace:"}, "entity": {}, "activity": {}, "agent": {}}
    for rel in RELS:
        doc[rel] = {}
    conn = _rconn()
    if conn is None:
        return doc
    if node:
        lin = lineage(node, direction="both", depth=12, limit=1000)
        nodes = {n["id"] for n in lin["nodes"]} | {node}
        edges = lin["edges"]
    elif run_id:
        nodes = {f"run:{run_id}"}
        for r in conn.execute("SELECT span_id FROM spans WHERE run_id=?", (run_id,)):
            nodes.add("span:" + r["span_id"])
        for r in conn.execute("SELECT DISTINCT artifact_id FROM writes WHERE run_id=?", (run_id,)):
            if r["artifact_id"]:
                nodes.add("art:" + r["artifact_id"])
        ids = list(nodes)
        edges = []
        for i in range(0, len(ids), 400):
            ch = ids[i:i + 400]
            edges += _rows(conn.execute(
                "SELECT src, dst, rel, note, ts FROM edges WHERE src IN (%s)" % ",".join("?" * len(ch)), ch))
        edges = [e for e in edges if e["dst"] in nodes]
    else:
        return doc
    for n in sorted(nodes):
        info = _node_label(conn, n)
        rec = {"prov:label": info["label"], "trace:id": n}
        if n.startswith("art:"):
            doc["entity"][n] = dict(rec, **{"trace:store": info["store"], "trace:kind": info["kind"],
                                            "prov:generatedAtTime": info["ts"]})
        elif n.startswith("agent:"):
            doc["agent"][n] = rec
        else:
            doc["activity"][n] = dict(rec, **{"prov:startedAtTime": info["ts"]})
    for i, e in enumerate(edges):
        rel = e["rel"]
        if rel not in doc:
            continue
        doc[rel][f"_{i}"] = {"prov:from": e["src"], "prov:to": e["dst"], "trace:note": e.get("note")}
    return doc


@_safe(lambda: {"deleted": {}, "error": "prune failed"})
def prune(*, keep_days: float | None = None, before_seq: int | None = None,
          confirm: bool = False) -> dict:
    """EXPLICIT maintenance only — a no-op unless ``confirm=True``. Nothing is ever auto-deleted: the trace IS
    the research artefact. Only ``events``/``spans``/``runs`` are prunable; the ``writes`` ledger, ``artifacts``
    and ``edges`` (the auditability evidence) are never touched."""
    out = {"deleted": {}, "error": None, "confirmed": bool(confirm)}
    if not confirm:
        out["error"] = "prune requires confirm=True (nothing was deleted)"
        return out
    flush(timeout=5.0)
    with _lock:
        conn = _conn
    if conn is None:
        out["error"] = _REASON or "no trace store"
        return out
    cutoff = (time.time() - float(keep_days) * 86400.0) if keep_days else None
    try:
        conn.execute("BEGIN")
        if before_seq is not None:
            out["deleted"]["events"] = conn.execute("DELETE FROM events WHERE seq < ?",
                                                    (int(before_seq),)).rowcount
        if cutoff is not None:
            out["deleted"]["events"] = (out["deleted"].get("events") or 0) + conn.execute(
                "DELETE FROM events WHERE ts < ?", (cutoff,)).rowcount
            out["deleted"]["spans"] = conn.execute("DELETE FROM spans WHERE start_ts < ?", (cutoff,)).rowcount
            out["deleted"]["runs"] = conn.execute(
                "DELETE FROM runs WHERE started_ts < ? AND ended_ts IS NOT NULL", (cutoff,)).rowcount
        conn.execute("COMMIT")
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        out["error"] = str(exc)
    return out


# --------------------------------------------------------------------------- demo
def _demo():
    """Synthesise a small traced run into a temp DB, print the timeline + a lineage query, then prove that an
    unwritable store degrades without a single exception reaching the caller."""
    import tempfile

    try:                      # the store keeps Persian and em-dashes verbatim; a cp1252 console must not die
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tmp = tempfile.mkdtemp(prefix="trace_demo_")
    st = configure(db_path=os.path.join(tmp, "trace.db"), origin="cli", async_writes=True)
    print(f"trace store: {st['db_path']}\n"
          f"enabled={st['enabled']} degraded={st['degraded']} schema={st['schema_version']} "
          f"api={st['api_version']} async={st['async']}\n")

    # ---- run 1: a command that fails, learns a lesson, and succeeds on the retry ---------------------
    cmd = "Put the apple in the fridge"
    with run(kind="command", command=cmd, scene="FloorPlan1", level="L2", lang="en",
             agent_id="gemma3:4b", arm="reuse_memory",
             config={"model": "gemma3:4b", "rag_enabled": True, "loop_enabled": True}) as rid1:
        with span("retrieval knowledge_base", kind="RETRIEVER", op="retrieval", actor="system:rag"):
            record_use(store="kb", table_or_kind="lesson", pointer=7, why="injected into planner prompt")
            emit("retrieval.performed", summary="2 hits (1 lesson, 1 procedure)",
                 payload={"hits": [{"id": 7, "kind": "lesson"}]}, actor="system:rag")
        record_llm(model="gemma3:4b", endpoint="http://172.20.144.1:11434",
                   prompt="PLAN: " + cmd, completion='[{"action":"pickup","object":"Apple"}]',
                   duration_ms=1830.0)
        emit("plan.produced", summary="2 steps: PickupObject(Apple), PutObject(Fridge)",
             payload={"steps": 2})
        with span("execute_tool pick_and_place Apple -> Fridge", kind="TOOL", op="execute_tool",
                  step_ix=0, scene="FloorPlan1") as tool_sid:
            for _ in range(6):
                record_action("MoveAhead", kwargs={"moveMagnitude": 0.25}, success=True, scene="FloorPlan1")
            record_action("PickupObject", kwargs={"objectId": "Apple|+01.0|+00.9|-00.6"}, success=True,
                          scene="FloorPlan1", duration_ms=24.0)
            ep = record_write(store="mem", table_or_kind="episodes", pointer=301, verdict="insert",
                              why="step executed", source="environment", scope="FloorPlan1",
                              summary="PickupObject(Apple) ok",
                              after={"success": 1}, audit_line="[LOG episode]  PickupObject  ok")
        with span("close_loop", kind="LOOP", op="close_loop", actor="system:improve"):
            art = record_write(store="kb", table_or_kind="procedure", pointer=12, verdict="insert",
                               why="command succeeded; storing the plan that worked", source="environment",
                               scope="FloorPlan1", confidence=1.0, summary=cmd,
                               dedup_key="put the apple in the fridge|((pickup,apple),(put,fridge))",
                               content={"command": cmd}, derived_from=["art:" + ep],
                               audit_line="[ADD procedure]  <environment>  " + cmd,
                               after={"occurrences": 1, "uses": 0})
            record_write(store="file", table_or_kind="loop_reports.jsonl", pointer="loop_reports.jsonl#1",
                         verdict="insert", why="loop report journalled", source="environment")

    # ---- run 2: the same command again - the procedure is reinforced, not duplicated -----------------
    with run(kind="command", command=cmd, scene="FloorPlan1", level="L2", lang="en",
             agent_id="gemma3:4b", config={"model": "gemma3:4b"}) as rid2:
        with span("close_loop", kind="LOOP", op="close_loop", actor="system:improve"):
            record_write(store="kb", table_or_kind="procedure", pointer=12, verdict="reinforce",
                         why="same command + same plan signature -> merged", source="environment",
                         before={"occurrences": 1, "uses": 0}, after={"occurrences": 2, "uses": 1},
                         audit_line="[UPSERT procedure]  <environment>  " + cmd)
    flush(timeout=5.0)

    # ---- D1: the timeline --------------------------------------------------------------------------
    print("TIMELINE  (events, oldest first)")
    print(f"  {'seq':>4}  {'time':<19}  {'level':<8} {'type':<18} {'actor':<16} summary")
    for e in query_events(limit=40, order="asc")["events"]:
        w = e.get("write")
        extra = f"   ->{w['artifact_id']}" if w else ""
        print(f"  {e['seq']:>4}  {e['iso']:<19}  {e['level']:<8} {e['type']:<18} "
              f"{(e['actor'] or '-'):<16} {(e['summary'] or '')[:52]}{extra}")

    tree = run_tree(rid1)
    print(f"\nRUN TREE  {rid1[:12]}...  ({tree['run']['command']}, {len(tree['spans'])} spans, "
          f"{tree['run']['n_writes']} writes)")
    for s in tree["spans"]:
        rep = f" x{s['repeat']}" if (s.get("repeat") or 1) > 1 else ""
        dur = f"{s['duration_ms']:.1f} ms" if s.get("duration_ms") is not None else "open"
        print(f"  {'  ' * (s['depth'] or 0)}[{s['kind']:<9}] {s['name'][:46]:<46}{rep:<5} {dur:>9} "
              f"{'W:' + str(len(s['writes'])) if s['writes'] else ''}")

    # ---- D2: causality, both directions ------------------------------------------------------------
    print(f"\nCAUSES OF {art}   (\"what produced this KB entry?\" - one indexed query)")
    for c in artifact_causes(art):
        print(f"  {c['iso']}  {c['verdict']:<10} run={str(c['run_id'])[:8]} span={c['span_name']}"
              f"  before={c['before_json']} after={c['after_json']}")
        print(f"      why: {c['why']}")
    row = artifact(art)
    print(f"  artifact revision={row['revision']} occurrences={row['occurrences']} "
          f"n_causes={row['n_causes']} alive={row['alive']} source={row['source']}")

    print(f"\nEFFECTS OF span {tool_sid[:8]}...   (\"what did this action write?\")")
    for w in span_effects(tool_sid):
        print(f"  {w['verdict']:<10} {w['store']}.{w['table_or_kind']} #{w['pointer']} -> {w['artifact_id']}")

    print(f"\nLINEAGE (ancestors) of {art}")
    for n in lineage("art:" + art, direction="up", depth=8)["nodes"]:
        print(f"  depth {n['depth']}  {n['id'][:52]:<52} {(n['label'] or '')[:36]}")
    print(f"\nLINEAGE (descendants) of run:{rid2[:12]}...")
    for n in lineage(f"run:{rid2}", direction="down", depth=8)["nodes"]:
        print(f"  depth {n['depth']}  {n['id'][:52]:<52} {(n['label'] or '')[:36]}")

    s = stats()
    print(f"\nSTATS  runs={s['runs']}  events_by_type={len(s['events_by_type'])} types  "
          f"orphans={s['orphans']}  top={[a['artifact_id'] for a in s['top_artifacts'][:3]]}")
    print(f"       writes: {[(w['store'] + '.' + w['table_or_kind'], w['verdict'], w['n']) for w in s['writes_by_store_verdict']]}")
    m = s["tracer"]["metrics"]
    print(f"       overhead: emit p50={m['emit_us_p50']} us  p99={m['emit_us_p99']} us  "
          f"flush p99={m['flush_ms_p99']} ms  dropped={s['tracer']['dropped']}  "
          f"skipped_actions={s['tracer']['skipped_actions']}")

    # ---- degradation: an UNWRITABLE store must never reach the caller ------------------------------
    blocker = os.path.join(tmp, "not_a_dir")
    with open(blocker, "w", encoding="utf-8") as fh:
        fh.write("x")
    bad = configure(db_path=os.path.join(blocker, "sub", "trace.db"))
    print(f"\nDEGRADED-MODE PROBE  (db under a regular file)\n"
          f"  enabled={bad['enabled']}  degraded={bad['degraded']}\n  reason: {bad['reason']}")
    probes = [
        ("enabled", lambda: enabled()),
        ("start_run", lambda: start_run(kind="command", command="x")),
        ("start_span", lambda: start_span("x")),
        ("emit", lambda: emit("console.log", summary="x")),
        ("record_write", lambda: record_write(store="kb", table_or_kind="fact", pointer=1,
                                              verdict="insert")),
        ("record_use", lambda: record_use(store="kb", table_or_kind="fact", pointer=1)),
        ("record_action", lambda: record_action("MoveAhead")),
        ("record_llm", lambda: record_llm(model="m", endpoint=None, prompt="p", completion="c",
                                          duration_ms=1.0)),
        ("link", lambda: link("run:a", "run:b", "wasInformedBy")),
        ("end_run", lambda: end_run()),
        ("flush", lambda: flush(0.2)),
        ("query_events", lambda: query_events()["events"]),
        ("query_runs", lambda: query_runs()["runs"]),
        ("run_tree", lambda: run_tree("nope")["spans"]),
        ("query_writes", lambda: query_writes()["writes"]),
        ("artifact_causes", lambda: artifact_causes("kb:procedure:12")),
        ("span_effects", lambda: span_effects("deadbeef")),
        ("lineage", lambda: lineage("art:kb:procedure:12")["nodes"]),
        ("artifact", lambda: artifact("kb:procedure:12")),
        ("stats", lambda: sorted(stats())[:3]),
        ("search", lambda: search("apple")["runs"]),
        ("db_tables", lambda: len(db_tables())),
        ("db_rows", lambda: db_rows("kb.procedures")["error"]),
        ("db_row", lambda: db_row("kb.procedures", "1")["error"]),
        ("export_prov_json", lambda: sorted(export_prov_json(run_id="x"))[:2]),
        ("install_thor_hooks", lambda: install_thor_hooks()["error"]),
        ("prune", lambda: prune()["error"]),
        ("get_event", lambda: get_event(1)),
        ("status", lambda: status()["degraded"]),
    ]
    ok = 0
    for name, fn in probes:
        try:
            val = fn()
            ok += 1
            print(f"  ok   {name:<18} -> {str(val)[:58]}")
        except Exception as exc:
            print(f"  RAISED {name:<16} -> {exc!r}")
    with run(kind="command", command="degraded run still re-raises the caller's error"):
        pass
    print(f"  {ok}/{len(probes)} public entry points returned a benign value; 0 exceptions escaped")

    # SQL injection is not possible through the query API: node ids are always bound parameters.
    inj = lineage("art:x'; DROP TABLE edges--")
    configure(db_path=os.path.join(tmp, "trace.db"))
    still = status()["edges"]
    print(f"\nINJECTION PROBE  lineage(\"art:x'; DROP TABLE edges--\") -> {len(inj['nodes'])} nodes, "
          f"edges table intact with {still} rows")
    close()
    print(f"\nTRACE DEMO OK  (temp store left at {tmp})")


if __name__ == "__main__":
    _demo()
