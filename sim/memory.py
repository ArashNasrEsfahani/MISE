"""Structured episodic + spatial memory — the agent's relational long-term store (sibling to the JSON KnowledgeBase).

`knowledge.py`/`rag.py` hold *unstructured* memories (procedures, lessons, facts) retrieved by keyword/vector. This
module is the **structured** half: a stdlib ``sqlite3`` database of things the agent needs to *query relationally*:

  - **spatial** — where objects were last seen ("the Mug is in the Cabinet"), each row tagged with how it was
    obtained (``track`` = ``perceived`` vs ``oracle``) and its provenance (``source``). The agent accumulates many
    observations; :meth:`Memory.query_location` resolves them into a single *best current belief*.
  - **episodes** — a transcript of attempted actions (command, level L1/L2/L3, action+args, success/error, and the
    state before/after) — the raw material the self-improving loop later distils into procedures and lessons.
  - **perception_labels** — VLM/active-perception outputs (bounding boxes per image) for the perception contribution.

**Provenance ranking** is the key idea reused from `knowledge.py`: a belief sourced from a ``human`` outranks one
the agent inferred from the ``environment``; ties break on recency, and confidence decays with age (a location seen
an hour ago is trusted less than one seen a moment ago).

Stdlib-first, per CLAUDE.md: module-top imports are pure stdlib (``json, os, sqlite3, time``). It does **not** import
``ai2thor``; it does **not** import ``numpy`` or ``rag`` at module load — ``rag`` is imported LAZILY inside
:meth:`Memory.recall` / :meth:`Memory.recall_text` so unstructured retrieval is fused in only when actually needed.

**Every write is auditable** (per CLAUDE.md): it is appended to a human-readable audit log with a timestamp, the
entry's ``source``, a one-line summary, and a ``why`` — mirroring ``knowledge.KnowledgeBase._audit``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

# ---------------------------------------------------------------------- observability sidecar (optional)
# See the note in knowledge.py: `trace` is also a stdlib module name, so consumers guard on TRACE_API_VERSION.
# Absent the sidecar every hook below is a no-op and this module behaves exactly as it did before.
try:                                    # pragma: no cover - environment dependent
    import trace as _trace              # sim/trace.py — the observability plane
    if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
        _trace = None                   # we resolved the STDLIB trace module: no instrumentation
except Exception:
    _trace = None


def _t(name, *a, **kw):
    """Call ``trace.<name>(*a, **kw)``, swallowing EVERYTHING — mirrors Memory._audit's IO guard.

    Named by string and resolved lazily: ``_t(_trace.emit, ...)`` would raise at the call site when the
    sidecar is absent, before the guard below could run. See knowledge._t for the full note.
    """
    if _trace is None:
        return None
    try:
        fn = getattr(_trace, name, None)
        return None if fn is None else fn(*a, **kw)
    except Exception:
        return None


def _tracing() -> bool:
    """True when the trace store is live. Guards the few hooks that need an EXTRA read (mark_cell's prior
    state, clear_nav's doomed cells) so tracing-off costs exactly zero additional sqlite work."""
    if _trace is None:
        return False
    try:
        return bool(_trace.enabled())
    except Exception:
        return False


# Twin of knowledge.AUDIT_MIRROR — see the note there (TRACE_AUDIT_MIRROR=0 drops the duplicate narration).
AUDIT_MIRROR = str(os.environ.get("TRACE_AUDIT_MIRROR", "1")).strip().lower() not in ("0", "false", "no", "off")


def _actor(source):
    """PROV agent id for a provenance `source`, or None."""
    if _trace is None:
        return None
    try:
        return getattr(_trace, "_SOURCE_ACTOR", {}).get(str(source or ""))
    except Exception:
        return None


# Provenance trust ranking (who/what produced a memory): human-asserted beats demonstrated beats agent-observed
# beats seeded-prior. Reused from knowledge.py's provenance vocabulary so the two stores rank memories alike.
PROVENANCE = {"human": 3, "demo": 2, "environment": 1, "prior": 0}

# Confidence half-life (seconds): a belief's trust halves every HALF_LIFE seconds since it was last seen. Overridable
# per-instance. ~1h is a sensible default for a kitchen where objects get moved between episodes.
HALF_LIFE_DEFAULT = 3600.0

# DDL for the three tables. ``track`` is constrained to the two measurement modes; ``source`` carries provenance.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS spatial (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene       TEXT,
    object_type TEXT,
    object_id   TEXT,
    receptacle  TEXT,
    x           REAL,
    y           REAL,
    z           REAL,
    track       TEXT CHECK (track IN ('perceived','oracle')),
    source      TEXT,
    confidence  REAL,
    last_seen   REAL,
    state       TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL,
    scene        TEXT,
    command      TEXT,
    level        TEXT,
    action       TEXT,
    args         TEXT,
    success      INTEGER,
    error        TEXT,
    state_before TEXT,
    state_after  TEXT,
    rating       REAL
);
CREATE TABLE IF NOT EXISTS perception_labels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    scene      TEXT,
    boxes      TEXT,
    source     TEXT
);
CREATE TABLE IF NOT EXISTS nav_cells (
    scene     TEXT,
    gx        INTEGER,
    gz        INTEGER,
    state     TEXT CHECK (state IN ('free','blocked','unknown')),
    last_seen REAL,
    source    TEXT DEFAULT 'environment',
    PRIMARY KEY (scene, gx, gz)
);
CREATE TABLE IF NOT EXISTS failures (
    signature     TEXT PRIMARY KEY,             -- f"{action}|{error_class}|{object_type}"
    scene         TEXT,
    action        TEXT,
    error_class   TEXT,
    object_type   TEXT,
    occurrences   INTEGER,
    first_seen    REAL,
    last_seen     REAL,
    last_error    TEXT,
    example_command TEXT
);
CREATE INDEX IF NOT EXISTS ix_spatial_lookup ON spatial (object_type, scene);
-- Step 4 (s3-memory): the honest agent's own object memory. One row per sighting (retained evidence); a track's
-- current belief is its newest non-superseded row. Rules live in sim/objmem.py. Never written from simulator truth.
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene       TEXT,
    episode     TEXT,
    step        INTEGER,
    ts          REAL,
    obs_id      TEXT,
    object_type TEXT,
    state       TEXT,
    conf        REAL,
    box         TEXT,
    point       TEXT,
    pose        TEXT,
    x           REAL,
    y           REAL,
    z           REAL,
    xyz_ok      INTEGER,
    track       TEXT,
    status      TEXT CHECK (status IN ('trusted','contradicted','superseded','held','placed')),
    place       TEXT,
    source      TEXT,
    why         TEXT
);
CREATE INDEX IF NOT EXISTS ix_sightings_track ON sightings (scene, track);
CREATE TABLE IF NOT EXISTS places_seen (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene       TEXT,
    episode     TEXT,
    step        INTEGER,
    ts          REAL,
    object_type TEXT,
    place_type  TEXT,
    x           REAL,
    y           REAL,
    z           REAL,
    pose        TEXT,
    track       TEXT
);
"""


def _j(v):
    """Serialize a structured value (dict/list) to JSON text for storage; pass strings/None through untouched."""
    if v is None or isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


class Memory:
    """SQLite-backed structured memory: spatial beliefs, action episodes, and perception labels — auditable writes."""

    def __init__(self, path: str | None = None, audit_path: str | None = None,
                 half_life: float = HALF_LIFE_DEFAULT, kb_path: str | None = None):
        here = os.path.dirname(os.path.abspath(__file__))
        self.path = path or os.path.join(here, "memory.db")
        self.audit_path = audit_path or (os.path.splitext(self.path)[0] + "_audit.log")
        # KnowledgeBase JSON that the lazy rag layer retrieves from in recall(); resolved next to this module so it
        # works regardless of CWD. Only touched when recall()/recall_text() is called.
        self.kb_path = kb_path or os.path.join(here, "knowledge_base.json")
        self.half_life = float(half_life)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        # Data-safe migration for EXISTING databases created before the Phase-5 funnel: the ``state`` column and the
        # ``failures`` table are new. ``CREATE TABLE IF NOT EXISTS`` above already covers fresh DBs and the new table;
        # only the added *column* on the pre-existing ``spatial`` table needs an ALTER, which raises if it is already
        # present — so we swallow that. No rows are ever dropped or rewritten.
        try:
            self.conn.execute("ALTER TABLE spatial ADD COLUMN state TEXT")
        except Exception:
            pass  # column already exists (or fresh DB already has it) — nothing to migrate
        self.conn.commit()

    # ----------------------------------------------------------------- audit (mirrors knowledge.KnowledgeBase._audit)
    def _audit(self, op: str, summary: str, why: str, source: str | None = None):
        src = f"  <{source}>" if source else ""
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [{op}]{src}  {summary}"
                + (f"   WHY: {why}" if why else ""))
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        # M11: mirror the human-readable line into the structured trail (twin of knowledge._audit's K5 hook),
        # gated by the same TRACE_AUDIT_MIRROR knob — the identical line is already on `writes.audit_line`.
        if AUDIT_MIRROR:
            _t("emit", "audit.line", summary=line[:200], level="DEBUG", actor=_actor(source),
               payload={"store": "mem", "op": op, "source": source, "summary": summary, "why": why,
                        "path": self.audit_path, "line": line})
        return line

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _xyz(pos):
        """Normalize a position (``{'x','y','z'}`` dict, a 3-sequence, or None) into an ``(x, y, z)`` tuple."""
        if pos is None:
            return (None, None, None)
        if isinstance(pos, dict):
            return (pos.get("x"), pos.get("y"), pos.get("z"))
        try:
            return (pos[0], pos[1], pos[2])
        except Exception:
            return (None, None, None)

    def _decay(self, conf, last_seen, now: float | None = None) -> float:
        """Recency-decayed confidence: ``conf * 0.5 ** (age / half_life)`` (no decay if ``last_seen`` is unknown)."""
        c = float(conf or 0.0)
        if last_seen is None:
            return c
        now = now if now is not None else time.time()
        age = max(0.0, now - float(last_seen))
        return c * (0.5 ** (age / self.half_life))

    # ----------------------------------------------------------------- write API
    def remember_location(self, scene: str, object_type: str, receptacle: str | None = None, pos=None,
                          track: str = "perceived", source: str = "environment", confidence: float = 1.0,
                          object_id: str | None = None, ts: float | None = None, why: str = "") -> int:
        """Record an observation of *where* ``object_type`` is (on/in ``receptacle`` and/or at ``pos``).

        Append-only: each call adds a row, so beliefs accumulate a history; :meth:`query_location` resolves them.
        ``track`` is ``'perceived'`` (from the VLM/active perception) or ``'oracle'`` (from the sim metadata).
        ``source`` is provenance (``human``/``demo``/``environment``/``prior``). Returns the new row id.
        """
        x, y, z = self._xyz(pos)
        seen = float(ts) if ts is not None else time.time()
        cur = self.conn.execute(
            "INSERT INTO spatial (scene, object_type, object_id, receptacle, x, y, z, track, source, "
            "confidence, last_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (scene, object_type, object_id, receptacle, x, y, z, track, source, float(confidence), seen))
        self.conn.commit()
        where = receptacle or (f"({x},{y},{z})" if x is not None else "?")
        line = self._audit("REMEMBER loc", f"{object_type} @ {where} in {scene} (track={track})",
                           why, source=source)
        # M1: append-only raw observation (the human-teaching path writes through here).
        _t("record_write", store="mem", table_or_kind="spatial", pointer=cur.lastrowid,
           verdict="insert", why=why, source=source, scope=scene, confidence=float(confidence),
           summary=f"{object_type} @ {where} in {scene}", audit_line=line,
           after={"scene": scene, "object_type": object_type, "object_id": object_id,
                  "receptacle": receptacle, "pos": (x, y, z), "track": track,
                  "confidence": float(confidence), "last_seen": seen}, store_path=self.path)
        return cur.lastrowid

    # ----------------------------------------------------------------- distillation funnel: change-gated spatial write
    def _best_spatial_row(self, scene: str | None, object_type: str, track: str | None,
                          object_id: str | None = None):
        """Current best belief ROW (raw ``sqlite3.Row``) for ``(scene, object_type, track)`` or ``None``.

        Mirrors :meth:`query_location`'s ranking (scene match > provenance > recency) but returns the raw row so the
        change-gate can compare receptacle/state/pos and reinforce it in place. If ``object_id`` is given, an exact
        object-id match is preferred over a type-only match (the same physical object, positively identified).
        """
        sql = "SELECT * FROM spatial WHERE object_type=?"
        params: list = [object_type]
        if scene is not None:                              # scope the change-gate to THIS scene: a same-typed object
            sql += " AND scene=?"                           # in another kitchen must NOT reinforce this scene's belief
            params.append(scene)                            # (the funnel key is (scene, object_type, track))
        if track:
            sql += " AND track=?"
            params.append(track)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return None

        def rank(r):
            id_match = 1 if (object_id is not None and r["object_id"] == object_id) else 0
            scene_match = 1 if (scene is not None and r["scene"] == scene) else 0
            return (id_match, scene_match, PROVENANCE.get(r["source"], 0), r["last_seen"] or 0.0)

        return max(rows, key=rank)

    def observe_location(self, scene: str | None, object_type: str, receptacle: str | None = None, pos=None,
                         state: str | None = None, track: str = "perceived", source: str = "environment",
                         confidence: float = 1.0, object_id: str | None = None, ts: float | None = None,
                         why: str = "", move_eps: float = 0.25) -> dict:
        """CHANGE-GATED spatial upsert — the heart of the distillation funnel (the PASSIVE spatial writer).

        Unlike :meth:`remember_location` (append-only raw history), this collapses repeated identical observations into
        *confidence* rather than *rows*. Given the current best belief for ``(scene, object_type, track)``:

          - **no prior belief** → INSERT a fresh row (``{'action': 'insert', ...}``).
          - **unchanged** (same ``receptacle`` AND same ``state`` AND — if ``pos`` given — within ``move_eps`` metres
            of the stored pos) → REINFORCE: bump ``last_seen`` to now and pull ``confidence`` halfway toward 1.0
            (``conf + (1-conf)*0.5``); NO new row (``{'action': 'reinforce', ...}``).
          - **changed** (receptacle differs, state differs, or the object moved > ``move_eps``) → INSERT a new row so
            the belief history is preserved (``{'action': 'insert', ...}``).

        Duplication therefore becomes signal (confidence), never rows: distilled spatial size stays ≈ one row per
        changed object, not one per observation. Every insert/reinforce is audited. Returns
        ``{'action': 'insert'|'reinforce', 'id': rowid}``.
        """
        now = float(ts) if ts is not None else time.time()
        best = self._best_spatial_row(scene, object_type, track, object_id=object_id)
        x, y, z = self._xyz(pos)

        def _do_insert():
            cur = self.conn.execute(
                "INSERT INTO spatial (scene, object_type, object_id, receptacle, x, y, z, track, source, "
                "confidence, last_seen, state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (scene, object_type, object_id, receptacle, x, y, z, track, source, float(confidence), now, state))
            self.conn.commit()
            where = receptacle or (f"({x},{y},{z})" if x is not None else "?")
            line = self._audit("OBSERVE insert", f"{object_type} @ {where}"
                               + (f" [{state}]" if state else "") + f" in {scene} (track={track})",
                               why or "new/changed belief", source=source)
            # M2: change-gated INSERT — a NEW belief row (first sighting, or the object/state changed).
            _t("record_write", store="mem", table_or_kind="spatial", pointer=cur.lastrowid,
               verdict="insert", why=why or "new/changed belief", source=source, scope=scene,
               confidence=float(confidence), summary=f"{object_type} @ {where} in {scene}",
               audit_line=line, dedup_key=f"{scene}|{object_type}|{track}",
               before=({"receptacle": best["receptacle"], "state": best["state"],
                        "confidence": best["confidence"], "id": best["id"]} if best is not None else None),
               after={"scene": scene, "object_type": object_type, "object_id": object_id,
                      "receptacle": receptacle, "pos": (x, y, z), "state": state, "track": track,
                      "confidence": float(confidence), "last_seen": now}, store_path=self.path)
            return {"action": "insert", "id": cur.lastrowid}

        if best is None:
            return _do_insert()

        # Compare against the current belief to decide reinforce vs. insert.
        same_receptacle = (best["receptacle"] or None) == (receptacle or None)
        same_state = (best["state"] or None) == (state or None)
        moved = False
        if pos is not None and best["x"] is not None:
            try:
                dx = float(x) - float(best["x"]); dy = float(y) - float(best["y"]); dz = float(z) - float(best["z"])
                moved = (dx * dx + dy * dy + dz * dz) ** 0.5 > move_eps
            except Exception:
                moved = False

        if same_receptacle and same_state and not moved:
            conf = float(best["confidence"] or 0.0)
            reinforced = min(1.0, conf + (1.0 - conf) * 0.5)
            self.conn.execute("UPDATE spatial SET last_seen=?, confidence=? WHERE id=?",
                              (now, reinforced, best["id"]))
            self.conn.commit()
            line = self._audit("OBSERVE reinforce", f"{object_type} unchanged in {scene} "
                               f"(track={track}, conf {conf:.2f}->{reinforced:.2f})",
                               why or "duplicate obs -> confidence, not a row", source=source)
            # M3: MERGE — duplication became confidence, not a row. No new id; the existing belief is revised.
            _t("record_write", store="mem", table_or_kind="spatial", pointer=best["id"],
               verdict="reinforce", why=why or "duplicate obs -> confidence, not a row", source=source,
               scope=scene, confidence=reinforced,
               summary=f"{object_type} unchanged in {scene} (track={track})", audit_line=line,
               dedup_key=f"{scene}|{object_type}|{track}",
               before={"confidence": conf, "last_seen": best["last_seen"],
                       "receptacle": best["receptacle"], "state": best["state"]},
               after={"confidence": reinforced, "last_seen": now,
                      "receptacle": best["receptacle"], "state": best["state"]}, store_path=self.path)
            return {"action": "reinforce", "id": best["id"]}

        return _do_insert()

    # ----------------------------------------------------------------- distillation funnel: failure signatures
    @staticmethod
    def _error_class(error) -> str:
        """Normalize a raw error string into a coarse, deterministic bucket used for failure signatures.

        Lowercases, strips object ids / coordinates / any digits, collapses whitespace, and truncates to ~48 chars so
        that "PickupObject failed: Mug|+01.2|-00.3 not visible" and "... Apple|... not visible" map to the same class
        ("pickupobject failed not visible"). This is what makes 50 near-identical failures dedup to a single counter.
        """
        import re
        s = str(error or "").lower()
        s = re.sub(r"[|]", " ", s)                     # object-id pipe separators
        s = re.sub(r"[-+]?\d*\.?\d+", " ", s)          # digits / coordinates / ids
        s = re.sub(r"[^a-z ]+", " ", s)                # punctuation → space
        s = re.sub(r"\s+", " ", s).strip()             # collapse whitespace
        return s[:48].strip()

    def record_failure(self, action: str | None, error, object_type: str | None = None, scene: str | None = None,
                       command: str | None = None, ts: float | None = None) -> dict:
        """Dedup a failure by *signature* into a COUNTER (not a row per occurrence) — the failure arm of the funnel.

        ``signature = f"{action}|{error_class}|{object_type}"``. If the signature already exists, its ``occurrences``
        is incremented and ``last_seen``/``last_error`` refreshed; otherwise a new row is inserted with
        ``occurrences=1`` (recording ``first_seen`` and the ``example_command``). 50 identical failures collapse to
        one row with ``occurrences=50``. Auditable. Returns ``{'signature': ..., 'occurrences': n}``.
        """
        now = float(ts) if ts is not None else time.time()
        ecls = self._error_class(error)
        sig = f"{action}|{ecls}|{object_type}"
        row = self.conn.execute("SELECT occurrences FROM failures WHERE signature=?", (sig,)).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO failures (signature, scene, action, error_class, object_type, occurrences, "
                "first_seen, last_seen, last_error, example_command) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sig, scene, action, ecls, object_type, 1, now, now, str(error or ""), command))
            n = 1
        else:
            n = int(row["occurrences"] or 0) + 1
            self.conn.execute(
                "UPDATE failures SET occurrences=?, last_seen=?, last_error=? WHERE signature=?",
                (n, now, str(error or ""), sig))
        self.conn.commit()
        line = self._audit("RECORD failure", f"{action} [{ecls}] on {object_type} in {scene} (x{n})",
                           command or "", source="environment")
        # M4: INSERT-or-MERGE on the signature primary key. Signatures collide BY DESIGN (that is the counter),
        # so every occurrence gets its own `writes` row — "same signature" must never mean "same event".
        _t("record_write", store="mem", table_or_kind="failures", pointer=sig,
           verdict=("insert" if n == 1 else "reinforce"), why=command or "", source="environment",
           scope=scene, occurrences=n, summary=f"{action} [{ecls}] on {object_type} in {scene}",
           audit_line=line, dedup_key=sig,
           before=(None if n == 1 else {"occurrences": n - 1}),
           after={"occurrences": n, "last_seen": now, "last_error": str(error or ""),
                  "error_class": ecls, "action": action, "object_type": object_type}, store_path=self.path)
        return {"signature": sig, "occurrences": n}

    def top_failures(self, scene: str | None = None, limit: int = 10) -> list:
        """Most-frequent failure signatures (occurrences desc), optionally filtered to ``scene`` — as list[dict]."""
        sql = "SELECT * FROM failures"
        params: list = []
        if scene is not None:
            sql += " WHERE scene=?"
            params.append(scene)
        sql += " ORDER BY occurrences DESC, last_seen DESC LIMIT ?"
        params.append(int(limit))
        return [{k: r[k] for k in r.keys()} for r in self.conn.execute(sql, params).fetchall()]

    def failure_stats(self) -> dict:
        """Failure-tier size: ``{'rows': <distinct signatures>, 'occurrences': <total failures counted>}``."""
        row = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(occurrences), 0) FROM failures").fetchone()
        return {"rows": int(row[0] or 0), "occurrences": int(row[1] or 0)}

    # ----------------------------------------------------------------- distillation funnel: decay + prune
    def prune(self, scene: str | None = None, min_confidence: float = 0.05, now: float | None = None) -> dict:
        """Drop superseded, stale spatial beliefs to keep the distilled tier small & fresh.

        For each ``(scene, object_type, track)`` key, the single newest row is always kept (the current belief); any
        OLDER row whose *decayed* confidence has fallen below ``min_confidence`` is deleted. Never removes the newest
        row for a key even if it has decayed. Auditable. Returns ``{'spatial_removed': n}``.
        """
        now = now if now is not None else time.time()
        sql = "SELECT * FROM spatial"
        params: list = []
        if scene is not None:
            sql += " WHERE scene=?"
            params.append(scene)
        rows = self.conn.execute(sql, params).fetchall()
        # Group by key; within each group sort newest-first so index 0 is protected.
        groups: dict = {}
        for r in rows:
            key = (r["scene"], r["object_type"], r["track"])
            groups.setdefault(key, []).append(r)
        removed = []
        doomed = []                       # M7: full row detail, captured BEFORE the DELETE (see below)
        tracing = _tracing()
        for key, grp in groups.items():
            grp.sort(key=lambda r: (r["last_seen"] or 0.0), reverse=True)
            for r in grp[1:]:  # skip the newest (protected)
                decayed = self._decay(r["confidence"], r["last_seen"], now)
                if decayed < min_confidence:
                    removed.append(r["id"])
                    if tracing:
                        doomed.append({"id": r["id"], "scene": r["scene"], "object_type": r["object_type"],
                                       "track": r["track"], "receptacle": r["receptacle"],
                                       "confidence": r["confidence"], "decayed": round(decayed, 6),
                                       "last_seen": r["last_seen"], "source": r["source"]})
        if removed:
            self.conn.executemany("DELETE FROM spatial WHERE id=?", [(i,) for i in removed])
            self.conn.commit()
            line = self._audit("PRUNE spatial", f"removed {len(removed)} stale superseded belief(s)"
                               + (f" in {scene}" if scene else ""), "decayed < min_confidence",
                               source="environment")
            # M7: the audit line records only a COUNT, so today a pruned belief is gone without trace. Record
            # every deleted row with its raw AND decayed confidence — information that exists nowhere on disk.
            # Marks the artifact alive=0 (prov:wasInvalidatedBy) rather than removing it from the graph.
            for d in doomed:
                _t("record_write", store="mem", table_or_kind="spatial", pointer=d["id"],
                   verdict="delete", why=f"decayed {d['decayed']} < min_confidence {min_confidence}",
                   source="environment", scope=d.get("scene"), confidence=d.get("confidence"),
                   summary=f"{d.get('object_type')} @ {d.get('receptacle')} in {d.get('scene')}",
                   audit_line=line, dedup_key=f"{d.get('scene')}|{d.get('object_type')}|{d.get('track')}",
                   before=d, after=None, actor="system:memory", store_path=self.path)
            _t("emit", "memory.write", level="DEBUG", actor="system:memory", scene=scene,
               summary=f"prune removed {len(removed)} stale spatial belief(s)"
                       + (f" in {scene}" if scene else ""),
               payload={"op": "prune", "scene": scene, "min_confidence": min_confidence,
                        "removed_ids": list(removed), "rows": doomed})
        return {"spatial_removed": len(removed)}

    def log_episode(self, scene: str | None = None, command: str | None = None, level: str | None = None,
                    action: str | None = None, args=None, success: bool | None = None, error: str | None = None,
                    state_before=None, state_after=None, rating: float | None = None,
                    source: str = "environment", ts: float | None = None, why: str = "") -> int:
        """Append one attempted action to the transcript (the raw self-improving-loop signal). Returns the row id.

        ``level`` is the command's hierarchy level (L1/L2/L3); ``args``/``state_before``/``state_after`` may be
        dicts/lists (stored as JSON). ``success`` is stored as 0/1 (or NULL if unknown).
        """
        when = float(ts) if ts is not None else time.time()
        succ = None if success is None else int(bool(success))
        cur = self.conn.execute(
            "INSERT INTO episodes (ts, scene, command, level, action, args, success, error, state_before, "
            "state_after, rating) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (when, scene, command, level, action, _j(args), succ, error, _j(state_before), _j(state_after),
             None if rating is None else float(rating)))
        self.conn.commit()
        outcome = "ok" if succ else ("fail" if succ == 0 else "?")
        line = self._audit("LOG episode", f"[{level or '-'}] {action or command or '?'} -> {outcome} in {scene}",
                           why, source=source)
        # M5: the execution transcript. The (run_id, span) attribution this row acquires from the ambient
        # context is what removes the batching ambiguity documented at _latest_batch_ids above — WITHOUT adding
        # a column to `episodes` (Phase-6's rating/reward path depends on that schema staying put). The plan-step
        # index lives on the enclosing FEEDBACK span (feedback.log_step passes step_ix), one join away.
        _t("record_write", store="mem", table_or_kind="episodes", pointer=cur.lastrowid,
           verdict="insert", why=why, source=source, scope=scene,
           summary=f"[{level or '-'}] {action or command or '?'} -> {outcome}", audit_line=line,
           after={"ts": when, "scene": scene, "command": command, "level": level, "action": action,
                  "success": succ, "error": error, "rating": rating}, store_path=self.path)
        return cur.lastrowid

    # ----------------------------------------------------------------- Phase-6: plan-level rating + reward (bandit RL)
    # These support the LinUCB contextual bandit in select.py, which picks ONE planning strategy per command and learns
    # from a reward = simulator success (episodes.success 0/1, already logged per step) PLUS optional human thumbs
    # (up=+1 / down=-1) written into episodes.rating. Both operate on the "latest batch": the set of episode rows from
    # the MOST RECENT execution of a given (scene, command). Because episodes carry no explicit plan/batch id (and we
    # must NOT change log_episode's schema/behaviour), a batch is identified STRUCTURALLY as the last CONTIGUOUS run of
    # rows for this exact (scene, command) in insertion (id) order — i.e. every such row newer than the newest row that
    # belongs to any DIFFERENT (scene, command). The steps of one plan are logged back-to-back, so they form one
    # contiguous run; a later re-execution of the same command is separated from the earlier one by whatever other
    # commands got logged in between. (Documented limitation: two executions of the *identical* (scene, command) with
    # no other command logged between them cannot be told apart without a batch id, and would count as one batch — a
    # deliberate trade-off given the "don't touch log_episode's schema" constraint.)
    def _latest_batch_ids(self, scene: str | None, command: str | None, before_ts: float | None = None) -> list:
        """Row ids of the most recent (scene, command) plan execution, ascending; ``[]`` if none (see the class note
        above for how a "batch" is delimited).

        ``before_ts`` optionally treats the batch "as of" that time (only rows with ``ts <= before_ts`` are considered
        when locating the newest one), so a rating can target the plan that was running at a given moment.
        SQLite ``IS`` is used instead of ``=`` so ``NULL`` scene/command values compare correctly.
        """
        if before_ts is None:
            last = self.conn.execute(
                "SELECT MAX(id) FROM episodes WHERE scene IS ? AND command IS ?", (scene, command)).fetchone()[0]
        else:
            last = self.conn.execute(
                "SELECT MAX(id) FROM episodes WHERE scene IS ? AND command IS ? AND ts<=?",
                (scene, command, float(before_ts))).fetchone()[0]
        if last is None:
            return []
        sep = self.conn.execute(
            "SELECT MAX(id) FROM episodes WHERE id<? AND NOT (scene IS ? AND command IS ?)",
            (last, scene, command)).fetchone()[0]
        sep = sep if sep is not None else 0
        return [r[0] for r in self.conn.execute(
            "SELECT id FROM episodes WHERE scene IS ? AND command IS ? AND id>? AND id<=? ORDER BY id",
            (scene, command, sep, last)).fetchall()]

    def set_rating(self, scene: str | None, command: str | None, rating: float, *, ts: float | None = None) -> int:
        """Attach a HUMAN thumbs rating to the most recent plan for (scene, command); return the rows updated.

        Writes ``rating`` (convention: ``+1`` thumbs-up, ``-1`` thumbs-down) into the ``rating`` column of EVERY
        episode row in that latest batch (see :meth:`_latest_batch_ids`). Returns the number of rows updated (``0`` if
        no episodes exist for the pair, or on any error). Exception-safe: never raises, so it can never break serving
        or planning. Auditable, logged as a ``human``-sourced write to match the provenance ranking (human >
        environment). ``ts`` optionally selects the batch that was running at/before that time. This is INDEPENDENT of
        :meth:`log_episode`'s own ``rating`` parameter (that path is left untouched and still works as before).
        """
        try:
            ids = self._latest_batch_ids(scene, command, before_ts=ts)
            if not ids:
                self._audit("RATE plan", f"rating={rating} for {command!r} in {scene}: no matching plan to rate",
                            "human thumbs but no episodes for the pair", source="human")
                return 0
            marks = ",".join("?" for _ in ids)
            self.conn.execute(f"UPDATE episodes SET rating=? WHERE id IN ({marks})", [float(rating)] + ids)
            self.conn.commit()
            line = self._audit("RATE plan", f"rating={rating} on {len(ids)} step(s) of {command!r} in {scene}",
                               "human feedback on the latest executed plan", source="human")
            # M6: the audit line records a COUNT; the trace records the IDS, one write per rated episode, so the
            # human thumb becomes a cause attached to each specific step artifact (not an unattributable tally).
            for eid in ids:
                _t("record_write", store="mem", table_or_kind="episodes", pointer=eid,
                   verdict="update", why="human feedback on the latest executed plan", source="human",
                   scope=scene, summary=f"rating={rating} on {command!r}", audit_line=line,
                   before={"rating": None}, after={"rating": float(rating)}, store_path=self.path)
            _t("emit", "feedback.received", actor="human:arash", scene=scene,
               summary=f"rating={rating} on {len(ids)} step(s) of {command!r} in {scene}",
               payload={"kind": "rating", "rating": float(rating), "scene": scene, "command": command,
                        "episode_ids": list(ids), "n": len(ids)})
            return len(ids)
        except Exception:
            return 0

    def plan_reward(self, scene: str | None, command: str | None) -> float | None:
        """Scalar plan-level reward in ``[0, 1]`` for the most recent (scene, command) plan, or ``None``.

        Exact formula (over the latest batch — see :meth:`_latest_batch_ids`):
          * ``base`` = fraction of *scored* steps that succeeded = ``mean(success)`` over rows whose ``success`` is
            NOT NULL. Rows with ``success IS NULL`` are ignored. If there are no scored rows, ``base`` is ``None``.
          * If a HUMAN ``rating`` is present on the batch (the newest non-NULL ``rating`` wins) it OVERRIDES ``base``
            — human outranks environment in the provenance ranking: ``rating > 0 -> 1.0``, ``rating < 0 -> 0.0``. A
            ``rating == 0`` carries no direction and is ignored (falls back to ``base``).
          * Return: the human override when present and directional; else ``base``; else ``None`` (nothing scored —
            no successes/failures recorded and no human rating).
        Exception-safe: returns ``None`` on any error.
        """
        try:
            ids = self._latest_batch_ids(scene, command)
            if not ids:
                return None
            marks = ",".join("?" for _ in ids)
            rows = self.conn.execute(
                f"SELECT success, rating FROM episodes WHERE id IN ({marks}) ORDER BY id", ids).fetchall()
            succ = [int(r["success"]) for r in rows if r["success"] is not None]
            base = (sum(succ) / len(succ)) if succ else None
            human = None
            for r in rows:                       # ascending id -> newest non-NULL rating wins
                if r["rating"] is not None:
                    human = float(r["rating"])
            if human is not None:
                if human > 0:
                    return 1.0
                if human < 0:
                    return 0.0
            return base
        except Exception:
            return None

    def add_perception_label(self, image_path: str, scene: str | None = None, boxes=None,
                             source: str = "environment", why: str = "") -> int:
        """Store a perception result (``boxes`` = list of detections for ``image_path``). Returns the row id."""
        cur = self.conn.execute(
            "INSERT INTO perception_labels (image_path, scene, boxes, source) VALUES (?,?,?,?)",
            (image_path, scene, _j(boxes), source))
        self.conn.commit()
        n = len(boxes) if hasattr(boxes, "__len__") else "?"
        line = self._audit("LABEL perception", f"{n} box(es) on {image_path} in {scene}", why, source=source)
        # M9: the label is DERIVED from an image file — record that edge so the frame is part of the lineage.
        _t("record_write", store="mem", table_or_kind="perception_labels", pointer=cur.lastrowid,
           verdict="insert", why=why, source=source, scope=scene,
           summary=f"{n} box(es) on {image_path} in {scene}", audit_line=line,
           after={"image_path": image_path, "scene": scene, "n_boxes": n, "source": source},
           derived_from=[f"art:file:image:{getattr(_trace, '_safe_ref', str)(image_path)}"], store_path=self.path)
        return cur.lastrowid

    # ----------------------------------------------------------------- discovered nav map (occupancy from real moves)
    def mark_cell(self, scene: str, gx: int, gz: int, state: str, source: str = "environment", why: str = "") -> None:
        """Upsert one grid cell's occupancy ``state`` (``'free'``/``'blocked'``/``'unknown'``) for ``scene``.

        ``gx``/``gz`` are integer grid indices (the caller owns the cell_size → index mapping). REPLACE semantics:
        the newest mark for a cell wins, refreshing ``last_seen``. ``state`` comes from REAL navigation — a cell the
        agent reaches is ``free``; a move/teleport that fails or collides is ``blocked``; unvisited is ``unknown``.
        """
        # M10: INSERT OR REPLACE destroys the prior cell state silently, so read it FIRST — an `unknown -> free`
        # transition is exactly what a navigation trace needs to show. Guarded: costs nothing when tracing is off.
        old = None
        if _tracing():
            try:
                old = self.conn.execute(
                    "SELECT state, last_seen, source FROM nav_cells WHERE scene=? AND gx=? AND gz=?",
                    (scene, int(gx), int(gz))).fetchone()
            except Exception:
                old = None
        now = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO nav_cells (scene, gx, gz, state, last_seen, source) VALUES (?,?,?,?,?,?)",
            (scene, int(gx), int(gz), state, now, source))
        self.conn.commit()
        line = self._audit("MARK cell", f"({gx},{gz}) -> {state} in {scene}", why, source=source)
        _t("record_write", store="mem", table_or_kind="nav_cells",
           pointer=f"{scene}:{int(gx)}:{int(gz)}",
           verdict=("insert" if old is None else "update"), why=why, source=source, scope=scene,
           summary=f"({gx},{gz}) -> {state} in {scene}", audit_line=line,
           dedup_key=f"{scene}|{int(gx)}|{int(gz)}",
           before=(None if old is None else {"state": old["state"], "last_seen": old["last_seen"],
                                             "source": old["source"]}),
           after={"state": state, "last_seen": now, "source": source}, store_path=self.path)

    def get_nav_cells(self, scene: str) -> list:
        """All known occupancy cells for ``scene`` as ``[{gx, gz, state, last_seen}, ...]``."""
        cur = self.conn.execute(
            "SELECT gx, gz, state, last_seen FROM nav_cells WHERE scene=? ORDER BY gx, gz", (scene,))
        return [{"gx": r["gx"], "gz": r["gz"], "state": r["state"], "last_seen": r["last_seen"]}
                for r in cur.fetchall()]

    def nav_stats(self, scene: str) -> dict:
        """Occupancy tallies for ``scene``: ``{'free', 'blocked', 'unknown', 'total'}`` cell counts."""
        out = {"free": 0, "blocked": 0, "unknown": 0}
        cur = self.conn.execute(
            "SELECT state, COUNT(*) FROM nav_cells WHERE scene=? GROUP BY state", (scene,))
        for state, n in cur.fetchall():
            if state in out:
                out[state] = n
        out["total"] = out["free"] + out["blocked"] + out["unknown"]
        return out

    def clear_nav(self, scene: str) -> None:
        """Drop the entire discovered occupancy map for ``scene`` (e.g. before a fresh re-scan)."""
        # M8: bulk DELETE — read the doomed cells BEFORE they vanish (the audit line keeps only a count).
        doomed = []
        if _tracing():
            try:
                doomed = [{"gx": r["gx"], "gz": r["gz"], "state": r["state"], "last_seen": r["last_seen"]}
                          for r in self.conn.execute(
                              "SELECT gx, gz, state, last_seen FROM nav_cells WHERE scene=?", (scene,))]
            except Exception:
                doomed = []
        cur = self.conn.execute("DELETE FROM nav_cells WHERE scene=?", (scene,))
        self.conn.commit()
        line = self._audit("CLEAR nav", f"{cur.rowcount} cell(s) in {scene}", "", source="environment")
        for d in doomed[:500]:            # per-cell provenance, capped; the aggregate event carries the rest
            _t("record_write", store="mem", table_or_kind="nav_cells",
               pointer=f"{scene}:{d['gx']}:{d['gz']}", verdict="delete",
               why="clear_nav: occupancy map dropped for a fresh re-scan", source="environment",
               scope=scene, summary=f"({d['gx']},{d['gz']}) [{d['state']}] in {scene}", audit_line=line,
               before=d, after=None, actor="system:memory", store_path=self.path)
        _t("emit", "memory.write", level="DEBUG", actor="system:memory", scene=scene,
           summary=f"clear_nav removed {cur.rowcount} cell(s) in {scene}",
           payload={"op": "clear_nav", "scene": scene, "rows": cur.rowcount,
                    "detailed": min(len(doomed), 500), "truncated": len(doomed) > 500})

    # ----------------------------------------------------------------- step-4 object memory (sightings / places_seen)
    _SIGHT_COLS = ("scene", "episode", "step", "ts", "obs_id", "object_type", "state", "conf", "box", "point",
                   "pose", "x", "y", "z", "xyz_ok", "track", "status", "place", "source", "why")

    def add_sighting(self, row: dict) -> int:
        """Append one sighting row (step 4, ``sim/objmem.py``). ``box``/``point``/``pose`` are stored as JSON.

        Append-only evidence: a track's belief changes by adding a row and re-statusing the old one
        (:meth:`set_sighting_status`), never by rewriting it. The ``why`` is stored in the row AND the audit log.
        """
        r = dict(row)
        r.setdefault("ts", time.time())
        r.setdefault("source", "perceived")
        x, y, z = self._xyz(r.pop("xyz", None))
        for k, v in (("x", x), ("y", y), ("z", z)):
            r.setdefault(k, v)
        vals = [(_j(r.get(c)) if c in ("box", "point", "pose") else r.get(c)) for c in self._SIGHT_COLS]
        vals[self._SIGHT_COLS.index("xyz_ok")] = int(bool(r.get("xyz_ok")))
        cur = self.conn.execute(
            f"INSERT INTO sightings ({', '.join(self._SIGHT_COLS)}) VALUES ({', '.join('?' * len(self._SIGHT_COLS))})",
            vals)
        self.conn.commit()
        summary = (f"#{cur.lastrowid} {r.get('object_type')} [{r.get('state') or '?'}] track={r.get('track')} "
                   f"-> {r.get('status')} in {r.get('scene')} (ep={r.get('episode')} step={r.get('step')})")
        line = self._audit("SIGHT add", summary, r.get("why") or "", source=r.get("source"))
        _t("record_write", store="mem", table_or_kind="sightings", pointer=cur.lastrowid, verdict="insert",
           why=r.get("why") or "", source=r.get("source"), scope=r.get("scene"), summary=summary, audit_line=line,
           after={c: r.get(c) for c in ("object_type", "state", "track", "status", "place", "x", "y", "z")},
           store_path=self.path)
        return cur.lastrowid

    def set_sighting_status(self, row_id: int, status: str, why: str = "", source: str = "perceived") -> None:
        """Re-status one sighting row (trusted -> superseded / contradicted / held / placed). Audited, reason kept."""
        old = self.conn.execute("SELECT status, why, object_type, track, scene FROM sightings WHERE id=?",
                                (int(row_id),)).fetchone()
        if old is None:
            raise KeyError(f"no sighting #{row_id}")
        note = ((old["why"] or "") + f" | {status}: {why}").strip(" |")
        self.conn.execute("UPDATE sightings SET status=?, why=? WHERE id=?", (status, note, int(row_id)))
        self.conn.commit()
        summary = f"#{row_id} {old['object_type']} track={old['track']}: {old['status']} -> {status}"
        line = self._audit("SIGHT status", summary, why, source=source)
        _t("record_write", store="mem", table_or_kind="sightings", pointer=int(row_id), verdict="update",
           why=why, source=source, scope=old["scene"], summary=summary, audit_line=line,
           before={"status": old["status"]}, after={"status": status}, store_path=self.path)

    def add_place_seen(self, row: dict) -> int:
        """Append one ``places_seen`` row: where an object type was seen (place type may be ``'?'``). For later RAG."""
        x, y, z = self._xyz(row.get("xyz"))
        cur = self.conn.execute(
            "INSERT INTO places_seen (scene, episode, step, ts, object_type, place_type, x, y, z, pose, track) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (row.get("scene"), row.get("episode"), row.get("step"), row.get("ts", time.time()),
             row.get("object_type"), row.get("place_type") or "?", x, y, z, _j(row.get("pose")), row.get("track")))
        self.conn.commit()
        summary = f"{row.get('object_type')} seen at {row.get('place_type') or '?'} in {row.get('scene')}"
        line = self._audit("PLACE seen", summary, row.get("why") or "", source="perceived")
        _t("record_write", store="mem", table_or_kind="places_seen", pointer=cur.lastrowid, verdict="insert",
           why=row.get("why") or "", source="perceived", scope=row.get("scene"), summary=summary, audit_line=line,
           store_path=self.path)
        return cur.lastrowid

    def load_sightings(self, scene: str, episode: str | None = None) -> list:
        """All sighting rows of ``scene`` (optionally one episode), oldest first, JSON columns decoded."""
        q, a = "SELECT * FROM sightings WHERE scene=?", [scene]
        if episode is not None:
            q, a = q + " AND episode=?", a + [episode]
        out = []
        for r in self.conn.execute(q + " ORDER BY id", a).fetchall():
            d = {k: r[k] for k in r.keys()}
            for k in ("box", "point", "pose"):
                try:
                    d[k] = json.loads(d[k]) if d[k] else None
                except Exception:
                    pass
            out.append(d)
        return out

    def places_seen(self, scene: str | None = None, object_type: str | None = None) -> list:
        """The append-only places log (stored now; RAG reads it later)."""
        q, a = "SELECT * FROM places_seen WHERE 1=1", []
        if scene:
            q, a = q + " AND scene=?", a + [scene]
        if object_type:
            q, a = q + " AND object_type=?", a + [object_type]
        return [{k: r[k] for k in r.keys()} for r in self.conn.execute(q + " ORDER BY id", a).fetchall()]

    # ----------------------------------------------------------------- read API
    def query_location(self, scene: str | None, object_type: str, track: str | None = None) -> dict | None:
        """Best current belief about ``object_type``'s location, or ``None`` if never observed.

        Resolution: among all observations of ``object_type`` (optionally filtered to ``track``), prefer rows for
        ``scene``, then rank by provenance (``human`` > ``demo`` > ``environment`` > ``prior``), tie-breaking on the
        most recent ``last_seen``. The returned ``confidence`` is recency-decayed (raw kept as ``raw_confidence``).
        """
        sql = "SELECT * FROM spatial WHERE object_type=?"
        params: list = [object_type]
        if track:
            sql += " AND track=?"
            params.append(track)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return None

        def rank(r):
            scene_match = 1 if (scene is not None and r["scene"] == scene) else 0
            return (scene_match, PROVENANCE.get(r["source"], 0), r["last_seen"] or 0.0)

        best = max(rows, key=rank)
        now = time.time()
        belief = {k: best[k] for k in best.keys()}
        belief["raw_confidence"] = belief.get("confidence")
        belief["confidence"] = self._decay(belief.get("confidence"), belief.get("last_seen"), now)
        belief["pos"] = (best["x"], best["y"], best["z"]) if best["x"] is not None else None
        return belief

    def _known_object_types(self, scene: str | None) -> list:
        """Distinct object types we hold a belief for in ``scene`` (plus rows with no scene)."""
        cur = self.conn.execute(
            "SELECT DISTINCT object_type FROM spatial WHERE object_type IS NOT NULL "
            "AND (? IS NULL OR scene=? OR scene IS NULL)", (scene, scene))
        return [r[0] for r in cur.fetchall()]

    def _belief_line(self, b: dict) -> str:
        where = b.get("receptacle") or (str(b.get("pos")) if b.get("pos") else "?")
        return (f"- **{b.get('object_type')}** @ {where} "
                f"_(scene={b.get('scene')}, track={b.get('track')}, src={b.get('source')}, "
                f"conf={float(b.get('confidence') or 0.0):.2f})_")

    def recall(self, scene: str | None, need_text: str, k: int = 5) -> dict:
        """Combine **structured** spatial beliefs (for any known object type named in ``need_text``) with the
        **unstructured** rag retrieval (procedures/lessons/facts) for the same query.

        Returns ``{'spatial': [belief, ...], 'object_types': [...], 'memory_text': <rag markdown>}``. ``rag`` is
        imported LAZILY here; if it (or its backend) is unavailable, ``memory_text`` is just empty.
        """
        text = (need_text or "").lower()
        matched = [ot for ot in self._known_object_types(scene) if ot and ot.lower() in text]
        spatial = []
        for ot in matched:
            b = self.query_location(scene, ot)
            if b:
                spatial.append(b)

        memory_text = ""
        try:
            import rag  # lazy: keep module-top pure-stdlib; pulls in the (degradable) vector layer only on demand
            hr = rag.HybridRAG(path=self.kb_path)
            memory_text = hr.recall_text(need_text, scene=scene, k=k)
        except Exception:
            memory_text = ""
        return {"spatial": spatial, "object_types": matched, "memory_text": memory_text}

    def recall_text(self, scene: str | None, need_text: str, k: int = 5) -> str:
        """Markdown block fusing known object locations and retrieved memories, for injection into a planner prompt."""
        r = self.recall(scene, need_text, k=k)
        lines = []
        if r["spatial"]:
            lines.append("**Known object locations** (structured memory):")
            lines.extend(self._belief_line(b) for b in r["spatial"])
        if r["memory_text"]:
            if lines:
                lines.append("")
            lines.append(r["memory_text"])
        return "\n".join(lines)

    def query(self, scene: str | None, need_text: str, needed_types=None, k: int = 5,
              with_text: bool = True) -> dict:
        """Superset of :meth:`recall` that also reports **misses** — object types the plan needs but memory lacks.

        Returns ``{'spatial': [...], 'object_types': [...], 'memory_text': ..., 'misses': [...]}``. When
        ``needed_types`` is supplied, ``misses`` lists every type we have no trustworthy belief about *in this scene*:
        :meth:`query_location` returns ``None``, or a *decayed* confidence below 0.1, or a belief that belongs to a
        DIFFERENT scene (a prior from another kitchen doesn't tell us where the thing is *here*). Those misses drive
        ACTIVE exploration (go look for what we can't locate). ``with_text=False`` skips the rag/embedding fusion (no
        Ollama round-trip) for the per-command miss-check path, which only needs ``misses``.
        """
        if with_text:
            base = self.recall(scene, need_text, k=k)
        else:
            # Lightweight miss-check: structured spatial only, no rag/Ollama call (the active-exploration trigger runs
            # this every command and only consumes `misses`).
            text = (need_text or "").lower()
            matched = [ot for ot in self._known_object_types(scene) if ot and ot.lower() in text]
            spatial = []
            for ot in matched:
                b = self.query_location(scene, ot)
                if b:
                    spatial.append(b)
            base = {"spatial": spatial, "object_types": matched, "memory_text": ""}
        misses = []
        for t in (needed_types or []):
            b = self.query_location(scene, t)
            stale = b is None or float(b.get("confidence") or 0.0) < 0.1
            wrong_scene = scene is not None and b is not None and b.get("scene") != scene
            if stale or wrong_scene:
                misses.append(t)
        base["misses"] = misses
        return base

    def recent_episodes(self, limit: int = 20) -> list:
        """The newest ``limit`` episode rows (newest first) as list[dict] — for the webapp ``/episodes`` panel."""
        cur = self.conn.execute(
            "SELECT id, ts, scene, command, level, action, args, success, error, rating "
            "FROM episodes ORDER BY id DESC LIMIT ?", (int(limit),))
        return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]

    def stats(self) -> dict:
        """Row counts per table (cheap health/size check), plus failure-tier size."""
        out = {}
        for t in ("spatial", "episodes", "perception_labels"):
            out[t] = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        fs = self.failure_stats()
        out["failures"] = fs["rows"]
        out["failure_occurrences"] = fs["occurrences"]
        return out

    def dedup_stats(self) -> dict:
        """Funnel-effectiveness snapshot — shows how much duplication the distillation collapsed.

        ``{'spatial_rows', 'spatial_keys', 'failures_rows', 'failures_occurrences'}`` where ``spatial_keys`` is the
        number of DISTINCT ``(scene, object_type, track)`` beliefs. A small ``spatial_keys`` vs a large stream of
        observations (and ``failures_rows`` << ``failures_occurrences``) is the funnel working: duplication became
        counts/confidence, not rows.
        """
        spatial_rows = self.conn.execute("SELECT COUNT(*) FROM spatial").fetchone()[0]
        spatial_keys = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT scene, object_type, track FROM spatial)").fetchone()[0]
        fs = self.failure_stats()
        return {"spatial_rows": spatial_rows, "spatial_keys": spatial_keys,
                "failures_rows": fs["rows"], "failures_occurrences": fs["occurrences"]}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Self-test: pure sqlite only (no numpy / rag / Ollama touched). Verifies that a HUMAN-sourced belief outranks an
    # earlier ENVIRONMENT-sourced one for the same object, via the provenance ranking in query_location.
    import tempfile

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mem_selftest.db")
    m = Memory(path=db)

    m.remember_location("FloorPlan1", "Mug", receptacle="Sink", source="environment", why="seen by VLM")
    m.remember_location("FloorPlan1", "Mug", receptacle="Cabinet", source="human", why="user told us")

    # Exercise the other two tables so their schema is validated too (still pure sqlite).
    m.log_episode(scene="FloorPlan1", command="put the mug away", level="L2", action="PickupObject",
                  args={"objectId": "Mug|1"}, success=True, why="step 1 ok")
    m.add_perception_label("/tmp/frame0.png", scene="FloorPlan1",
                           boxes=[{"label": "Mug", "bbox": [10, 20, 30, 40]}], why="vlm detect")

    belief = m.query_location("FloorPlan1", "Mug")
    expected = ("Cabinet", "human")
    got = (None, None) if belief is None else (belief["receptacle"], belief["source"])
    assert belief is not None and got == expected, f"expected human/Cabinet belief, got {belief!r}"
    print("query_location ->", belief["receptacle"], "| src:", belief["source"],
          "| conf:", round(belief["confidence"], 3), "| stats:", m.stats())

    # Discovered nav map: mark a few cells (some re-marked to test upsert), read back, assert tallies.
    m.mark_cell("FloorPlan1", 0, 0, "free", why="reached by navigation")
    m.mark_cell("FloorPlan1", 1, 0, "free", source="demo", why="reached by navigation")
    m.mark_cell("FloorPlan1", 2, 0, "blocked", why="teleport collided")
    m.mark_cell("FloorPlan1", 0, 1, "unknown", why="never visited")
    m.mark_cell("FloorPlan1", 0, 1, "free", why="now reached -> upsert over unknown")  # REPLACE, not a new row
    m.mark_cell("FloorPlan2", 9, 9, "blocked", why="other scene, must not leak into FloorPlan1 stats")

    cells = m.get_nav_cells("FloorPlan1")
    assert len(cells) == 4, f"expected 4 distinct FloorPlan1 cells (upsert), got {len(cells)}: {cells!r}"
    nstats = m.nav_stats("FloorPlan1")
    assert nstats == {"free": 3, "blocked": 1, "unknown": 0, "total": 4}, f"unexpected nav_stats: {nstats!r}"
    print("nav_stats(FloorPlan1) ->", nstats, "| cells:", len(cells))
    m.clear_nav("FloorPlan1")
    assert m.nav_stats("FloorPlan1")["total"] == 0, "clear_nav left rows behind"
    assert m.nav_stats("FloorPlan2")["blocked"] == 1, "clear_nav wrongly touched another scene"
    print("NAVCELLS SELFTEST OK")

    # ------------------------------------------------------------------ Phase-5 funnel: change-gated spatial upsert.
    # First observation of a NEW object -> INSERT.
    r1 = m.observe_location("FloorPlan1", "Bowl", receptacle="CounterTop", pos={"x": 1.0, "y": 0.9, "z": 2.0},
                            track="oracle", source="environment", why="first sighting")
    assert r1["action"] == "insert", f"first observe should insert, got {r1!r}"
    before_rows = m.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Bowl'").fetchone()[0]
    # Identical repeat observation -> REINFORCE (no new row; confidence climbs).
    r2 = m.observe_location("FloorPlan1", "Bowl", receptacle="CounterTop", pos={"x": 1.0, "y": 0.9, "z": 2.0},
                            track="oracle", source="environment", why="unchanged")
    assert r2["action"] == "reinforce", f"unchanged observe should reinforce, got {r2!r}"
    after_rows = m.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Bowl'").fetchone()[0]
    assert after_rows == before_rows, f"reinforce must not add a row ({before_rows}->{after_rows})"
    # Changed receptacle -> INSERT a new belief row (history preserved).
    r3 = m.observe_location("FloorPlan1", "Bowl", receptacle="Sink", pos={"x": 3.0, "y": 0.9, "z": 4.0},
                            track="oracle", source="environment", why="moved to sink")
    assert r3["action"] == "insert", f"changed observe should insert, got {r3!r}"
    changed_rows = m.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Bowl'").fetchone()[0]
    assert changed_rows == before_rows + 1, f"changed observe should add exactly one row, got {changed_rows}"
    print("OBSERVE funnel: insert/reinforce/insert ->", r1["action"], r2["action"], r3["action"])

    # Cross-scene isolation: the SAME object type observed in ANOTHER kitchen must INSERT its own belief (never
    # reinforce this scene's row), and must resolve back to its own scene. Guards the (scene,object_type,track) key.
    rf2 = m.observe_location("FloorPlan2", "Bowl", receptacle="Sink", pos={"x": 3.0, "y": 0.9, "z": 4.0},
                             track="oracle", source="environment", why="same type, different kitchen")
    assert rf2["action"] == "insert", f"cross-scene observe must insert a fresh belief, not reinforce, got {rf2!r}"
    assert m.query_location("FloorPlan2", "Bowl")["scene"] == "FloorPlan2", "FloorPlan2 Bowl must resolve to FloorPlan2"
    print("CROSS-SCENE isolation ->", rf2["action"], "| FloorPlan2 Bowl scene ok")

    # ------------------------------------------------------------------ Phase-5 funnel: failure signature counter.
    for _ in range(5):
        fr = m.record_failure("PickupObject", "PickupObject failed: Apple|+01.2|-00.3 not visible",
                              object_type="Apple", scene="FloorPlan1", command="pick up the apple")
    assert fr["occurrences"] == 5, f"5 identical failures should count to 5, got {fr!r}"
    fstats = m.failure_stats()
    assert fstats["rows"] == 1 and fstats["occurrences"] == 5, f"expected 1 row / 5 occ, got {fstats!r}"
    top = m.top_failures(scene="FloorPlan1")
    assert top and top[0]["occurrences"] == 5, f"top_failures should surface the x5 signature, got {top!r}"
    print("FAILURE counter: 1 row, occurrences =", fstats["occurrences"])

    # ------------------------------------------------------------------ Phase-5 funnel: query() miss detection.
    q = m.query("FloorPlan1", "bowl and a spatula", needed_types=["Bowl", "Spatula"])
    assert "Bowl" not in q["misses"], "Bowl is known -> not a miss"
    assert "Spatula" in q["misses"], f"Spatula never observed -> should be a miss, got {q['misses']!r}"
    print("QUERY misses ->", q["misses"])
    # Scene-strict: a type known ONLY in another kitchen is still a miss here (a prior elsewhere doesn't locate it
    # in this scene). Also exercises the rag-free with_text=False miss-check path (no Ollama round-trip).
    q3 = m.query("FloorPlan3", "bowl", needed_types=["Bowl"], with_text=False)
    assert "Bowl" in q3["misses"], f"Bowl known only in other scenes -> miss in FloorPlan3 (scene-strict), got {q3['misses']!r}"
    print("QUERY scene-strict miss ->", q3["misses"])

    # ------------------------------------------------------------------ Phase-5 funnel: prune stale superseded rows.
    # Age the superseded (older) Bowl@CounterTop belief far past several half-lives; the newest (Sink) is protected.
    old_ts = time.time() - m.half_life * 40
    m.conn.execute("UPDATE spatial SET last_seen=? WHERE object_type='Bowl' AND receptacle='CounterTop'", (old_ts,))
    m.conn.commit()
    pr = m.prune(scene="FloorPlan1")
    assert pr["spatial_removed"] == 1, f"one stale superseded Bowl belief should prune, got {pr!r}"
    kept = m.conn.execute("SELECT COUNT(*) FROM spatial WHERE object_type='Bowl' AND scene='FloorPlan1'").fetchone()[0]
    assert kept == 1, f"prune must keep the single newest FloorPlan1 Bowl belief, got {kept}"
    assert m.query_location("FloorPlan1", "Bowl")["receptacle"] == "Sink", "newest Bowl belief must survive prune"
    print("PRUNE removed:", pr["spatial_removed"], "| dedup_stats:", m.dedup_stats())

    m.close()
    for p in (db, os.path.splitext(db)[0] + "_audit.log"):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("SELFTEST OK")


# ===================================================================== PUBLIC API (discovered nav map) ==============
# Table:
#   nav_cells(scene TEXT, gx INTEGER, gz INTEGER,
#             state TEXT CHECK(state IN ('free','blocked','unknown')),
#             last_seen REAL, source TEXT DEFAULT 'environment',
#             PRIMARY KEY(scene, gx, gz))
#   gx,gz are integer grid-cell indices; the caller owns the cell_size -> index mapping (raw ints stored here).
#
# Methods (all auditable; occupancy comes from REAL navigation, not the sim oracle):
#   mark_cell(self, scene, gx, gz, state, source='environment', why='') -> None
#       Upsert one cell (INSERT OR REPLACE): newest mark wins, refreshes last_seen. state in free|blocked|unknown.
#   get_nav_cells(self, scene) -> list[dict]
#       [{'gx','gz','state','last_seen'}, ...] for the scene, ordered by (gx, gz).
#   nav_stats(self, scene) -> dict
#       {'free': int, 'blocked': int, 'unknown': int, 'total': int} cell counts for the scene.
#   clear_nav(self, scene) -> None
#       Delete the scene's entire discovered occupancy map (e.g. before a fresh re-scan).
