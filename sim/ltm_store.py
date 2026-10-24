"""Step 5 / A1 - long-term-memory storage engine (docs/step5-plan.md sections 2, 4, 9, 11).

One SQLite file, rollback journal (journal_mode=DELETE, NEVER WAL: the db may live on /mnt/d DrvFs), lazily created,
versioned migrations, and a SINGLE write path, `Store.commit`, that atomically writes in ONE transaction:

    commit_receipt row (idempotency + clock)  +  claim rows  +  one evidence row per claim mutation
    +  one authoritative write_log row per op

Everything else (tracer mirror, snapshots) is derived from, and happens after, that transaction.

Pure standard library, offline, imports no simulator / network module, never touches sim/trace.db or sim/memory.db.
The path is always injectable; the default (sim/step5/ltm.db) is only a convenience for the facade.

Honesty note (docs/step5-plan.md 1.4): the whitelist here is an input-validation boundary, not an information-flow
proof. It constrains shapes (closed enums, id prefixes, recursive key/token allowlists, privileged-key rejection);
that no privileged value is *produced* upstream is the adapter's job and is gated separately.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path

SCHEMA_LATEST = 2
STORE_FORMAT = "s5.ltm/1"
DEFAULT_DB = Path(__file__).resolve().parent / "step5" / "ltm.db"
CONSOLIDATION_VERSION = 1


class StoreError(Exception):
    pass


class WhitelistError(StoreError):
    """A write used a column/key/value the write API does not admit (privileged or malformed field)."""


class ConflictError(StoreError):
    """Same event id / episode / seq / key with different content."""


class OrderError(StoreError):
    """A commit that declared `expect_clock` found a different clock."""


class MissingRowError(StoreError):
    pass


class TruncatedCommitError(StoreError):
    """A commit whose declared op count / digest does not match its ops."""


class FrozenError(StoreError):
    """A write was attempted on a frozen (read-only) store."""


class SchemaError(StoreError):
    pass


# --------------------------------------------------------------------------------------------- canonical encoding
def canon(obj) -> str:
    """Canonical JSON: sorted keys, no spaces, UTF-8 characters kept, NaN/Inf forbidden."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(text: str | bytes) -> str:
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- whitelist machinery
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:.\-]{0,95}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_:.\-?/ ]{0,96}$")      # ASCII tokens allowed inside json blobs
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
HEX_RE = re.compile(r"^[0-9a-f]{8,64}$")
SIM_ID_RE = re.compile(r"[A-Za-z]+\|[-0-9.+]+\||_physics|\bobjectId\b", re.I)
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# json keys / column names that are privileged (compared after lower-casing and stripping '_'); goal_ok is admitted
# ONLY as a declared column (procedure.admit_goal_ok / episodes.goal_ok / application_log.goal_ok), never inside json
PRIVILEGED = {
    "objectid", "objectids", "group", "metadata", "oracle", "errorslogged", "audit", "counts", "detail", "why", "events",
    "err", "error", "errortext", "staterreaders", "statereaders", "reachablepositions", "visible", "goldplan", "goal",
    "goalpredicate", "goalpredicates", "goalok", "receptacleobjectids", "parentreceptacles", "scene", "sceneobjects",
    "position", "rotation", "isopen", "ispickedup", "lastactionmessage", "errormessage", "errormsg",
}
MAX_JSON_BYTES = 16384
MAX_TEXT = 2000
MUTATIONS = ("create", "support", "contradict", "supersede", "retire", "reactivate", "merge", "unmerge")
_ZWNJ = "‌"
_BAD_UNICODE_CATS = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}


def _priv_norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def is_privileged_name(name: str) -> bool:
    return _priv_norm(name) in PRIVILEGED


def _ascii_token(v: str, where: str) -> str:
    if not isinstance(v, str):
        raise WhitelistError(f"{where}: expected string")
    if unicodedata.normalize("NFKC", v) != v or not v.isascii():
        raise WhitelistError(f"{where}: non-ASCII / non-NFKC string not admitted in a token field")
    if SIM_ID_RE.search(v) or "|" in v:
        raise WhitelistError(f"{where}: looks like a simulator id")
    return v


def _unicode_text(v: str, where: str) -> str:
    """Free text columns (Persian allowed): NFC, no control/format/bidi chars except ZWNJ, no sim ids."""
    if not isinstance(v, str):
        raise WhitelistError(f"{where}: expected string")
    if len(v) > MAX_TEXT:
        raise WhitelistError(f"{where}: text too long")
    v = unicodedata.normalize("NFC", v)
    for ch in v:
        if ch == _ZWNJ or ch in ("\n", "\t"):
            continue
        if unicodedata.category(ch) in _BAD_UNICODE_CATS:
            raise WhitelistError(f"{where}: control/format character U+{ord(ch):04X} not admitted")
    if SIM_ID_RE.search(v):
        raise WhitelistError(f"{where}: looks like a simulator id / metadata key")
    return v


def _check_json(v, where: str, depth: int = 0):
    """Recursive allowlist for json blobs: scalars, lists, dicts with allowlisted keys, ASCII tokens only."""
    if depth > 6:
        raise WhitelistError(f"{where}: json nested too deep")
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise WhitelistError(f"{where}: non-finite number")
        return v
    if isinstance(v, str):
        _ascii_token(v, where)
        if not TOKEN_RE.match(v):
            raise WhitelistError(f"{where}: string {v[:24]!r} is not an enumerated token")
        return v
    if isinstance(v, (list, tuple)):
        if len(v) > 256:
            raise WhitelistError(f"{where}: list too long")
        return [_check_json(x, f"{where}[{i}]", depth + 1) for i, x in enumerate(v)]
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str) or not KEY_RE.match(k):
                raise WhitelistError(f"{where}: json key {k!r} is not an admitted key name")
            if is_privileged_name(k):
                raise WhitelistError(f"{where}: privileged json key {k!r}")
            out[k] = _check_json(x, f"{where}.{k}", depth + 1)
        return out
    raise WhitelistError(f"{where}: unsupported json value type {type(v).__name__}")


# --------------------------------------------------------------------------------------------- table specs
# type grammar: id | place (id with pl: prefix) | text (ascii token) | utext (unicode free text) | int | real | bool
#               | json | enum:a|b|c ; the 'since' value is the schema version that introduced the column.
def _c(t, since=1):
    return (t, since)


TABLE_SPEC = {
    "episodes": dict(pk=["episode"], claim=False, cols={
        "episode": _c("id"), "kitchen": _c("id"), "mode": _c("enum:practice|test|probe"),
        "world": _c("enum:reset|persistent"), "epoch": _c("int"), "level": _c("enum:L1|L2|L3"),
        "lang": _c("enum:en|fa"), "E": _c("int"), "exec_ok": _c("bool"),
        "goal_ok": _c("bool"),           # the single privileged verdict (admission/failed-reuse/plan_goal_unmet only)
        "invalid": _c("bool"), "stats": _c("json")}),
    "epochs": dict(pk=["kitchen", "epoch"], claim=False, cols={
        "kitchen": _c("id"), "epoch": _c("int"), "frame": _c("id"), "map_rev": _c("id"),
        "reason": _c("enum:task_reset|scene_reload|manual|eval_block|other"), "E": _c("int")}),
    "place_instance": dict(pk=["place"], claim=True, cols={
        "place": _c("place"), "kitchen": _c("id"), "frame": _c("id"), "epoch": _c("int"), "rtype": _c("id"),
        "cx": _c("real"), "cy": _c("real"), "cz": _c("real"), "extent": _c("json"), "created_E": _c("int")}),
    "place_alias": dict(pk=["place"], claim=True, cols={
        "place": _c("place"), "canonical": _c("place"), "active": _c("bool"), "E": _c("int")}),
    "location_claim": dict(pk=["claim"], claim=True, cols={
        "claim": _c("id"), "kitchen": _c("id"), "epoch": _c("int"), "frame": _c("id"), "otype": _c("id"),
        "place": _c("place"), "xyz": _c("json"), "verified": _c("enum:verified|trusted"),
        "provenance": _c("enum:natural|agent_placed"), "conf": _c("real"), "E_created": _c("int"),
        "E_support": _c("int"), "contradictions": _c("int"), "status": _c("enum:current|maybe|retired")}),
    "negative_evidence": dict(pk=["neg"], claim=True, cols={
        "neg": _c("id"), "kitchen": _c("id"), "epoch": _c("int"), "place": _c("place"), "otype": _c("id"),
        "detector_rev": _c("id"), "support": _c("int"), "E": _c("int"), "superseded": _c("bool"),
        "superseded_by": _c("id")}),
    "placement": dict(pk=["placement"], claim=True, cols={
        "placement": _c("id"), "kitchen": _c("id"), "epoch": _c("int"), "place": _c("place"), "otype": _c("id"),
        "xyz": _c("json"), "provenance": _c("enum:verified|actuator"), "E": _c("int"),
        "status": _c("enum:current|historical|retired")}),
    # one immutable contribution per configuration unit U; habit P/N are DERIVED from these rows by summation
    "search_trial": dict(pk=["u"], claim=True, cols={
        "u": _c("id"), "kitchen": _c("id"), "epoch": _c("int"), "otype": _c("id"), "place": _c("place"),
        "rtype": _c("id"), "outcome": _c("enum:positive|negative"), "E": _c("int")}),
    "habit": dict(pk=["habit"], claim=True, cols={
        "habit": _c("id"), "kitchen": _c("id"), "level": _c("enum:kitchen|general"), "otype": _c("id"),
        "rtype": _c("id"), "place": _c("place"), "active": _c("bool"), "E_last": _c("int")}),
    "procedure": dict(pk=["proc"], claim=True, cols={
        "proc": _c("id"), "kitchen": _c("id"), "intent_key": _c("json"), "intent_text": _c("utext"),
        "body": _c("json"), "status": _c("enum:candidate|trusted|demoted|retired"), "success_uses": _c("int"),
        "failed_reuses": _c("int"), "E_created": _c("int"), "E_support": _c("int"),
        "admit_goal_ok": _c("bool")}),
    "lesson": dict(pk=["lesson"], claim=True, cols={
        "lesson": _c("id"), "kitchen": _c("id"), "conditions": _c("json"), "cond_key": _c("id"),
        "action": _c("enum:try_other_instance|change_pose|approach_closer|crouch|open_first|search_elsewhere|none"),
        "status": _c("enum:tentative|supported|contradicted|retired"), "helpful_units": _c("int"),
        "unhelpful_units": _c("int"), "recurrences": _c("int"), "origin_episode": _c("id"),
        "E_created": _c("int"), "E_support": _c("int"), "revalidate": _c("bool")}),
    # Gemma hypothesis prose: audit only, scanned like all text, NEVER rendered into retrieval or prompts
    "hypothesis_audit": dict(pk=["hyp"], claim=False, cols={
        "hyp": _c("id"), "lesson": _c("id"), "episode": _c("id"), "prose": _c("utext"),
        "evidence": _c("json"), "accepted": _c("bool"), "reason": _c("id")}),
    "application_log": dict(pk=["app"], claim=False, cols={
        "app": _c("id"), "episode": _c("id"), "record": _c("id"), "kind": _c("enum:procedure|lesson"),
        "step_i": _c("int"), "attempt_k": _c("int"), "outcome": _c("enum:neutral|helpful|unhelpful"),
        "goal_ok": _c("bool"), "retrieval": _c("id"), "E": _c("int"), "source": _c("id", 2)}),
    "retrieval_log": dict(pk=["retrieval"], claim=False, cols={
        "retrieval": _c("id"), "episode": _c("id"), "query": _c("json"), "candidates": _c("json"),
        "E": _c("int"), "channel": _c("id", 2)}),
}
LOG_TABLES = ("application_log", "retrieval_log")           # the only tables a frozen store's testlog accepts
CLAIM_TABLES = tuple(t for t, s in TABLE_SPEC.items() if s["claim"])
FK_PLACE = {"place_alias": ["place", "canonical"], "location_claim": ["place"], "negative_evidence": ["place"],
            "placement": ["place"], "search_trial": ["place"], "habit": ["place"]}
KINDS = ("consolidation", "reset_world", "admin", "test_log", "import")
EVIDENCE_COLS = ("evidence", "claim", "claim_table", "rule", "mutation", "episode", "epoch", "E", "summary",
                 "src_ev", "conf", "version", "event_id")
INTERNAL_NOT_CANONICAL = {"export_ack"}       # operational state, excluded from the state hash
SQL_TYPE = {"id": "TEXT", "place": "TEXT", "text": "TEXT", "utext": "TEXT", "int": "INTEGER", "real": "REAL",
            "bool": "INTEGER", "json": "TEXT"}


def _sqltype(t: str) -> str:
    return "TEXT" if t.startswith("enum:") else SQL_TYPE[t]


def _coerce(t: str, v, where: str):
    """Validate + normalise a scalar column value; returns the python value to store (json -> canonical str)."""
    if v is None:
        return None
    if t in ("id", "place"):
        _ascii_token(v, where)
        if not ID_RE.match(v):
            raise WhitelistError(f"{where}: {v!r} is not a well-formed id")
        if t == "place" and not v.startswith("pl:"):
            raise WhitelistError(f"{where}: place ids are agent-minted 'pl:<n>', got {v!r}")
        return v
    if t == "text":
        _ascii_token(v, where)
        if not TOKEN_RE.match(v):
            raise WhitelistError(f"{where}: not a token")
        return v
    if t == "utext":
        return _unicode_text(v, where)
    if t == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            raise WhitelistError(f"{where}: expected int")
        return v
    if t == "real":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise WhitelistError(f"{where}: expected finite real")
        return float(v)
    if t == "bool":
        if v not in (0, 1, True, False) or isinstance(v, float):
            raise WhitelistError(f"{where}: expected bool")
        return 1 if v else 0
    if t == "json":
        c = canon(_check_json(v, where))
        if len(c.encode("utf-8")) > MAX_JSON_BYTES:
            raise WhitelistError(f"{where}: json blob too large")
        return c
    if t.startswith("enum:"):
        if v not in t[5:].split("|"):
            raise WhitelistError(f"{where}: {v!r} is not one of {t[5:]}")
        return v
    raise WhitelistError(f"{where}: unknown type {t}")


# --------------------------------------------------------------------------------------------- schema + migrations
def _table_ddl(name: str, version: int) -> str:
    spec = TABLE_SPEC[name]
    cols = [f"{c} {_sqltype(t)}" for c, (t, since) in spec["cols"].items() if since <= version]
    cols.append("PRIMARY KEY (" + ", ".join(spec["pk"]) + ")")
    for col in FK_PLACE.get(name, []):
        cols.append(f"FOREIGN KEY ({col}) REFERENCES place_instance(place) DEFERRABLE INITIALLY DEFERRED")
    return f"CREATE TABLE {name} ({', '.join(cols)})"


def _ddl_v1() -> list[str]:
    s = [
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "CREATE TABLE commit_receipt (event_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_hash TEXT NOT NULL, "
        "n_ops INTEGER NOT NULL, episode TEXT, seq INTEGER, advance_clock INTEGER NOT NULL, E INTEGER NOT NULL)",
        "CREATE UNIQUE INDEX ux_receipt_episode ON commit_receipt(episode) WHERE advance_clock = 1",
        "CREATE UNIQUE INDEX ux_receipt_seq ON commit_receipt(seq) WHERE advance_clock = 1 AND seq IS NOT NULL",
        "CREATE TABLE evidence (evidence TEXT PRIMARY KEY, claim TEXT NOT NULL, claim_table TEXT NOT NULL, "
        "rule TEXT NOT NULL, mutation TEXT NOT NULL, episode TEXT, epoch INTEGER, E INTEGER NOT NULL, "
        "summary TEXT NOT NULL, src_ev TEXT NOT NULL, conf REAL, version INTEGER NOT NULL, event_id TEXT NOT NULL)",
        "CREATE INDEX ix_evidence_claim ON evidence(claim_table, claim)",
        "CREATE TABLE write_log (wl INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, op_idx INTEGER NOT NULL, "
        "tbl TEXT NOT NULL, row_key TEXT NOT NULL, op TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL, "
        "evidence TEXT, UNIQUE (event_id, op_idx))",
        "CREATE TABLE export_ack (trace_key TEXT NOT NULL, destination TEXT NOT NULL, PRIMARY KEY (trace_key, destination))",
    ]
    order = ["place_instance"] + [t for t in TABLE_SPEC if t != "place_instance"]
    s += [_table_ddl(t, 1) for t in order]
    s += ["CREATE UNIQUE INDEX ux_lesson_cond ON lesson(kitchen, cond_key)",
          "CREATE INDEX ix_loc_type ON location_claim(kitchen, otype, status)",
          "CREATE INDEX ix_trial_type ON search_trial(kitchen, otype)",
          "CREATE INDEX ix_place_type ON place_instance(kitchen, rtype)"]
    return s


def _ddl_v2() -> list[str]:
    """v2: log provenance columns (retrieval channel, application source) + log indices."""
    s = []
    for t, spec in TABLE_SPEC.items():
        for c, (typ, since) in spec["cols"].items():
            if since == 2:
                s.append(f"ALTER TABLE {t} ADD COLUMN {c} {_sqltype(typ)}")
    s += ["CREATE INDEX ix_applog_record ON application_log(record)",
          "CREATE INDEX ix_retlog_episode ON retrieval_log(episode)"]
    return s


MIGRATIONS = {1: ("base schema", _ddl_v1), 2: ("log provenance columns", _ddl_v2)}


def journal_policy(path) -> dict:
    """Journal mode is ALWAYS DELETE (rollback journal). `reason` records whether the path looks like DrvFs."""
    s = str(path).replace("\\", "/")
    drvfs = s.startswith("/mnt/") or s.startswith("//")
    return {"mode": "delete", "reason": "drvfs path: WAL -shm file unsupported" if drvfs else "declared policy: never WAL"}


# --------------------------------------------------------------------------------------------- the store
class Store:
    """SQLite store with the single atomic write API.

    Parameters: path (injectable; file created lazily on first access), kitchen label is NOT held here (rows carry it),
    target_version (tests: open an old schema), readonly (frozen snapshot), tracer (mirror after commit),
    fault_hook(point) (tests: raise to simulate a crash at 'ops_done' | 'pre_receipt' | 'pre_commit' | 'post_commit'
    | 'export_after_emit'), destination (name used to scope export acks).
    """

    def __init__(self, path=None, *, target_version: int = SCHEMA_LATEST, readonly: bool = False, tracer=None,
                 fault_hook=None, destination: str = "trace", consolidation_version: int = CONSOLIDATION_VERSION):
        self.path = Path(path) if path is not None else DEFAULT_DB
        if not (1 <= target_version <= SCHEMA_LATEST):
            raise SchemaError(f"target_version {target_version} outside 1..{SCHEMA_LATEST}")
        self.target_version = target_version
        self.readonly = readonly
        self.tracer = tracer
        self.fault_hook = fault_hook
        self.destination = destination
        self.consolidation_version = consolidation_version
        self.mirror_errors: list[str] = []
        self.version = 0
        self.pragmas: dict = {}
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    # -- connection ------------------------------------------------------------------------------------------
    def _fault(self, point: str):
        if self.fault_hook:
            self.fault_hook(point)

    def exists(self) -> bool:
        return self.path.exists()

    def _open(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if self.readonly:
            if not self.path.exists():
                raise StoreError(f"frozen store {self.path} does not exist")
            conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None,
                                   check_same_thread=False)
            conn.execute("PRAGMA query_only=ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._demote_wal(self.path)
            conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if not self.readonly:
            mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            if str(mode).lower() != "delete":
                conn.close()
                raise StoreError(f"journal_mode is {mode!r}, expected delete (never WAL)")
            conn.execute("PRAGMA synchronous=EXTRA")
        else:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.pragmas = {
            "journal_mode": str(mode).lower(),
            "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
            "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
            "policy": journal_policy(self.path)}
        if self.pragmas["foreign_keys"] != 1:
            conn.close()
            raise StoreError("PRAGMA foreign_keys did not stick")
        if not self.readonly and self.pragmas["synchronous"] != 3:
            conn.close()
            raise StoreError("PRAGMA synchronous=EXTRA did not stick")
        self._conn = conn
        try:
            self._ensure_schema(conn)
        except BaseException:
            self._conn = None
            conn.close()
            raise
        return conn

    @staticmethod
    def _demote_wal(path: Path):
        """If a stray db was left in WAL mode (header bytes 18/19 == 2), checkpoint it back to DELETE first."""
        if not path.exists() or path.stat().st_size < 100:
            return
        with open(path, "rb") as f:
            head = f.read(20)
        if len(head) >= 20 and head[18] == 2:
            c = sqlite3.connect(str(path), isolation_level=None)
            try:
                c.execute("PRAGMA journal_mode=DELETE")
            finally:
                c.close()

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- migrations ------------------------------------------------------------------------------------------
    def _current_version(self, conn) -> int:
        has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone()
        if not has:
            return 0
        r = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(r[0] or 0)

    def _ensure_schema(self, conn):
        cur = self._current_version(conn)
        if cur > SCHEMA_LATEST:
            raise SchemaError(f"database schema v{cur} is newer than this code (v{SCHEMA_LATEST}); refusing to open")
        if self.readonly:
            if cur != self.target_version and cur != SCHEMA_LATEST:
                raise FrozenError(f"frozen store is schema v{cur}; frozen handles never migrate")
            if cur == 0:
                raise FrozenError("frozen store is empty / uninitialised")
            self.version = cur
            return
        if cur > self.target_version:
            raise SchemaError(f"database is v{cur}, target_version {self.target_version} would downgrade")
        if cur < self.target_version and cur > 0 and self.path.exists():
            self._backup_before_migration(conn, cur)
        for v in range(cur + 1, self.target_version + 1):
            name, ddl = MIGRATIONS[v]
            conn.execute("BEGIN IMMEDIATE")
            try:
                for stmt in ddl():
                    conn.execute(stmt)
                conn.execute("INSERT INTO schema_version(version, name) VALUES (?, ?)", (v, name))
                if v == 1:
                    conn.execute("INSERT INTO meta(key, value) VALUES ('E', '0')")
                    conn.execute("INSERT INTO meta(key, value) VALUES ('format', ?)", (STORE_FORMAT,))
                    conn.execute("INSERT INTO meta(key, value) VALUES ('store_id', ?)",
                                 (sha256_hex(os.urandom(16) + str(self.path).encode())[:20],))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        self.version = self._current_version(conn)

    def _backup_before_migration(self, conn, cur: int):
        dest = self.path.with_name(self.path.name + f".premigrate-v{cur}")
        if dest.exists():
            dest.unlink()
        bk = sqlite3.connect(str(dest))
        try:
            conn.backup(bk)
        finally:
            bk.close()

    # -- helpers ---------------------------------------------------------------------------------------------
    def ensure(self) -> "Store":
        """Create the database file + schema now (otherwise it appears on first access)."""
        self._open()
        return self

    def clock(self) -> int:
        return int(self._open().execute("SELECT value FROM meta WHERE key='E'").fetchone()[0])

    def store_id(self) -> str:
        return self._open().execute("SELECT value FROM meta WHERE key='store_id'").fetchone()[0]

    def _cols(self, table: str) -> dict:
        if table not in TABLE_SPEC:
            raise WhitelistError(f"table {table!r} is not writable through the write API")
        return {c: t for c, (t, since) in TABLE_SPEC[table]["cols"].items() if since <= self.version}

    # -- op validation (no database access) ------------------------------------------------------------------
    def _normalise_op(self, op: dict, idx: int) -> dict:
        where = f"op[{idx}]"
        if not isinstance(op, dict):
            raise WhitelistError(f"{where}: op must be a dict")
        extra = set(op) - {"table", "op", "values", "key", "evidence"}
        if extra:
            raise WhitelistError(f"{where}: unknown op fields {sorted(extra)}")
        table = op.get("table")
        cols = self._cols(table)
        kind = op.get("op")
        if kind not in ("insert", "upsert", "update"):
            raise WhitelistError(f"{where}: op must be insert|upsert|update")
        spec = TABLE_SPEC[table]

        def norm(d, label, allow_E_placeholder):
            out = {}
            if not isinstance(d, dict):
                raise WhitelistError(f"{where}.{label}: expected dict")
            for c, v in d.items():
                if not isinstance(c, str) or not IDENT_RE.match(c):
                    raise WhitelistError(f"{where}.{label}: bad column name {c!r}")
                if is_privileged_name(c) and not (c in cols and cols[c] == "bool" and c in ("goal_ok", "admit_goal_ok")):
                    raise WhitelistError(f"{where}.{label}: privileged field {c!r}")
                if c not in cols:
                    raise WhitelistError(f"{where}.{label}: column {c!r} is not in the {table} whitelist (v{self.version})")
                if v == "$E" and cols[c] == "int" and allow_E_placeholder:
                    out[c] = "$E"
                else:
                    out[c] = _coerce(cols[c], v, f"{where}.{table}.{c}")
            return out

        values = norm(op.get("values", {}), "values", True)
        if kind == "update":
            key = norm(op.get("key"), "key", False)
            if sorted(key) != sorted(spec["pk"]) or any(v is None for v in key.values()):
                raise WhitelistError(f"{where}: update key must be exactly {spec['pk']}")
            if not values:
                raise WhitelistError(f"{where}: update without values")
            if any(c in values for c in spec["pk"]):
                raise WhitelistError(f"{where}: primary key columns cannot be updated")
        else:
            if "key" in op:
                raise WhitelistError(f"{where}: key is only for update")
            key = {c: values.get(c) for c in spec["pk"]}
            if any(v is None for v in key.values()):
                raise WhitelistError(f"{where}: {kind} needs primary key {spec['pk']}")
            if "$E" in key.values():
                raise WhitelistError(f"{where}: primary key cannot use $E")
        ev = op.get("evidence")
        if spec["claim"]:
            if not isinstance(ev, dict):
                raise WhitelistError(f"{where}: claim table {table} requires an evidence record")
        elif ev is not None and not isinstance(ev, dict):
            raise WhitelistError(f"{where}: evidence must be a dict")
        if ev is not None:
            extra = set(ev) - {"rule", "mutation", "summary", "src_ev", "conf"}
            if extra:
                raise WhitelistError(f"{where}.evidence: unknown fields {sorted(extra)}")
            rule = _coerce("id", ev.get("rule"), f"{where}.evidence.rule")
            if rule is None:
                raise WhitelistError(f"{where}.evidence: rule required")
            if ev.get("mutation") not in MUTATIONS:
                raise WhitelistError(f"{where}.evidence: mutation must be one of {MUTATIONS}")
            src = ev.get("src_ev", [])
            if not isinstance(src, list) or len(src) > 64 or not all(isinstance(x, str) and HEX_RE.match(x) for x in src):
                raise WhitelistError(f"{where}.evidence.src_ev must be a list of <=64 hex hashes")
            conf = ev.get("conf")
            ev = {"rule": rule, "mutation": ev["mutation"],
                  "summary": json.loads(_coerce("json", ev.get("summary", {}), f"{where}.evidence.summary")),
                  "src_ev": list(src), "conf": None if conf is None else _coerce("real", conf, f"{where}.evidence.conf")}
        pkvals = [key[c] if kind == "update" else values[c] for c in spec["pk"]]
        return {"table": table, "op": kind, "values": values, "key": key if kind == "update" else None,
                "evidence": ev, "row_key": "/".join(str(x) for x in pkvals)}

    # -- the single write path -------------------------------------------------------------------------------
    @staticmethod
    def build_commit(event_id: str, kind: str, ops: list, *, episode=None, seq=None, advance_clock=False,
                     expect_clock=None, stamp=None) -> dict:
        """Package a commit with its declared op count and digest so truncation is detectable."""
        return {"event_id": event_id, "kind": kind, "ops": ops, "episode": episode, "seq": seq,
                "advance_clock": advance_clock, "expect_clock": expect_clock, "stamp": stamp,
                "n_ops": len(ops), "ops_digest": sha256_hex(canon(ops))}

    def apply(self, c: dict) -> dict:
        return self.commit(c["event_id"], c["kind"], c["ops"], episode=c.get("episode"), seq=c.get("seq"),
                           advance_clock=c.get("advance_clock", False), expect_clock=c.get("expect_clock"),
                           stamp=c.get("stamp"), n_ops=c.get("n_ops"), ops_digest=c.get("ops_digest"))

    def commit(self, event_id: str, kind: str, ops: list, *, episode=None, seq=None, advance_clock: bool = False,
               expect_clock=None, stamp=None, n_ops=None, ops_digest=None) -> dict:
        """Atomically apply `ops`. Returns {status: committed|duplicate, E, event_id, write_log, evidence}.

        Idempotent per event_id: identical replay -> status 'duplicate' (nothing written); same id with different
        content -> ConflictError. `seq` (ordinal of an eligible practice episode) makes E order-independent;
        without it E = clock + 1 (order-dependent, declare it with `expect_clock`). `stamp` fixes "$E" for commits
        that do not advance the clock.
        """
        if self.readonly:
            raise FrozenError("this store is frozen (read-only)")
        if kind not in KINDS:
            raise WhitelistError(f"commit kind {kind!r} not in {KINDS}")
        _coerce("id", event_id, "event_id")
        if episode is not None:
            _coerce("id", episode, "episode")
        for nm, v in (("seq", seq), ("expect_clock", expect_clock), ("stamp", stamp)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise WhitelistError(f"{nm} must be a non-negative int")
        if advance_clock and episode is None:
            raise WhitelistError("a clock-advancing commit needs an episode id")
        if seq is not None and not advance_clock:
            raise WhitelistError("seq only applies to a clock-advancing commit")
        if not isinstance(ops, list) or not ops:
            raise WhitelistError("commit needs a non-empty op list")
        if n_ops is not None and n_ops != len(ops):
            raise TruncatedCommitError(f"commit declares {n_ops} ops but carries {len(ops)}")
        if ops_digest is not None and ops_digest != sha256_hex(canon(ops)):
            raise TruncatedCommitError("ops digest mismatch (truncated or altered commit)")
        with self._lock:
            conn = self._open()          # also fixes self.version, which op validation reads
            nops = [self._normalise_op(op, i) for i, op in enumerate(ops)]
            # the request hash is taken BEFORE "$E" substitution and covers everything that determines content
            payload_hash = sha256_hex(canon({
                "event_id": event_id, "kind": kind, "episode": episode, "seq": seq, "advance_clock": bool(advance_clock),
                "expect_clock": expect_clock, "stamp": stamp,
                "ops": [{"t": o["table"], "o": o["op"], "v": o["values"], "k": o["key"], "e": o["evidence"]} for o in nops]}))
            conn.execute("BEGIN IMMEDIATE")
            try:
                prior = conn.execute("SELECT payload_hash, E FROM commit_receipt WHERE event_id=?", (event_id,)).fetchone()
                if prior is not None:
                    if prior["payload_hash"] != payload_hash:
                        raise ConflictError(f"event {event_id}: replay with different content")
                    conn.execute("ROLLBACK")
                    return {"status": "duplicate", "event_id": event_id, "E": int(prior["E"]), "write_log": [], "evidence": []}
                clock = int(conn.execute("SELECT value FROM meta WHERE key='E'").fetchone()[0])
                if expect_clock is not None and clock != expect_clock:
                    raise OrderError(f"event {event_id}: expected clock {expect_clock}, store is at {clock}")
                if advance_clock:
                    if seq is not None:
                        clash = conn.execute("SELECT event_id FROM commit_receipt WHERE advance_clock=1 AND seq=?", (seq,)).fetchone()
                        if clash:
                            raise ConflictError(f"seq {seq} already used by commit {clash[0]}")
                        E = seq
                    else:
                        E = clock + 1
                    new_clock = max(clock, E)
                else:
                    E = stamp if stamp is not None else clock
                    new_clock = clock
                clash = conn.execute("SELECT event_id FROM commit_receipt WHERE advance_clock=1 AND episode=?", (episode,)).fetchone() \
                    if advance_clock else None
                if clash:
                    raise ConflictError(f"episode {episode} already consolidated by commit {clash[0]}")
                wl_ids, ev_ids = [], []
                for idx, o in enumerate(nops):
                    wl, evid = self._apply_op(conn, event_id, idx, o, E, episode)
                    wl_ids.append(wl)
                    if evid:
                        ev_ids.append(evid)
                self._fault("ops_done")
                self._fault("pre_receipt")
                conn.execute("INSERT INTO commit_receipt(event_id, kind, payload_hash, n_ops, episode, seq, advance_clock, E) "
                             "VALUES (?,?,?,?,?,?,?,?)",
                             (event_id, kind, payload_hash, len(nops), episode, seq if advance_clock else None,
                              1 if advance_clock else 0, E))
                if new_clock != clock:
                    conn.execute("UPDATE meta SET value=? WHERE key='E'", (str(new_clock),))
                if conn.execute("PRAGMA foreign_key_check").fetchone():
                    raise ConflictError("foreign key violation (place reference does not exist)")
                self._fault("pre_commit")
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as e:
                self._rollback(conn)
                raise ConflictError(f"integrity: {e}") from e
            except BaseException:
                self._rollback(conn)
                raise
        res = {"status": "committed", "event_id": event_id, "E": E, "write_log": wl_ids, "evidence": ev_ids}
        self._fault("post_commit")
        self._mirror()
        return res

    @staticmethod
    def _rollback(conn):
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _apply_op(self, conn, event_id: str, idx: int, o: dict, E: int, episode):
        table, kind, values = o["table"], o["op"], {c: (E if v == "$E" else v) for c, v in o["values"].items()}
        spec = TABLE_SPEC[table]
        pk = spec["pk"]
        if kind == "update":
            where, wargs = " AND ".join(f"{c}=?" for c in pk), [o["key"][c] for c in pk]
            if conn.execute(f"SELECT 1 FROM {table} WHERE {where}", wargs).fetchone() is None:
                raise MissingRowError(f"update of missing {table} row {o['row_key']}")
            sets = ", ".join(f"{c}=?" for c in values)
            conn.execute(f"UPDATE {table} SET {sets} WHERE {where}", list(values.values()) + wargs)
        else:
            where, wargs = " AND ".join(f"{c}=?" for c in pk), [values[c] for c in pk]
            old = conn.execute(f"SELECT * FROM {table} WHERE {where}", wargs).fetchone()
            if old is not None and kind == "insert":
                diff = [c for c, v in values.items() if old[c] != v]
                if diff:
                    raise ConflictError(f"insert of existing {table} row {o['row_key']} with different {diff}")
            elif old is None:
                cols = list(values)
                conn.execute(f"INSERT INTO {table}({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             [values[c] for c in cols])
            else:                                                         # upsert over an existing row
                rest = {c: v for c, v in values.items() if c not in pk}
                if rest:
                    conn.execute(f"UPDATE {table} SET {', '.join(f'{c}=?' for c in rest)} WHERE {where}",
                                 list(rest.values()) + wargs)
        evid = None
        ev = o["evidence"]
        if ev is not None:
            evid = sha256_hex(f"{event_id}#{idx}")[:24]
            epoch = values.get("epoch")
            conn.execute("INSERT INTO evidence(evidence, claim, claim_table, rule, mutation, episode, epoch, E, summary, src_ev, "
                         "conf, version, event_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (evid, f"{table}/{o['row_key']}", table, ev["rule"], ev["mutation"], episode, epoch, E,
                          canon(ev["summary"]), canon(ev["src_ev"]), ev["conf"], self.consolidation_version, event_id))
        payload = canon({"values": values, "key": o["key"]})
        cur = conn.execute("INSERT INTO write_log(event_id, op_idx, tbl, row_key, op, payload, payload_hash, evidence) "
                           "VALUES (?,?,?,?,?,?,?,?)",
                           (event_id, idx, table, o["row_key"], kind, payload,
                            sha256_hex(f"{event_id}|{idx}|{table}|{o['row_key']}|{kind}|{payload}|{evid}"), evid))
        return int(cur.lastrowid), evid

    # -- reads -----------------------------------------------------------------------------------------------
    def _decode(self, table: str, row) -> dict:
        d = dict(row)
        for c, (t, since) in TABLE_SPEC[table]["cols"].items():
            if t == "json" and d.get(c) is not None:
                d[c] = json.loads(d[c])
        return d

    def get(self, table: str, **key):
        rows = self.select(table, where=key)
        return rows[0] if rows else None

    def select(self, table: str, where: dict | None = None, order: str | None = None, limit: int | None = None) -> list:
        with self._lock:
            conn = self._open()
            cols = self._cols(table) if table in TABLE_SPEC else None
            if table in ("evidence", "write_log", "commit_receipt", "meta", "schema_version"):
                cols = {r["name"]: "x" for r in conn.execute(f"PRAGMA table_info({table})")}
            elif cols is None:
                raise WhitelistError(f"unknown table {table!r}")
            sql, args = f"SELECT * FROM {table}", []
            if where:
                parts = []
                for c, v in where.items():
                    if c not in cols:
                        raise WhitelistError(f"{table}: no column {c!r}")
                    if isinstance(v, bool):
                        v = int(v)
                    parts.append(f"{c}=?")
                    args.append(canon(v) if isinstance(v, (dict, list)) else v)
                sql += " WHERE " + " AND ".join(parts)
            if order:
                if order.lstrip("-") not in cols:
                    raise WhitelistError(f"{table}: cannot order by {order!r}")
                sql += f" ORDER BY {order.lstrip('-')}{' DESC' if order.startswith('-') else ''}"
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql, args).fetchall()
            return [self._decode(table, r) if table in TABLE_SPEC else dict(r) for r in rows]

    def count(self, table: str) -> int:
        return int(self._open().execute(f"SELECT COUNT(*) FROM {self._known_table(table)}").fetchone()[0])

    def _known_table(self, table: str) -> str:
        if table in TABLE_SPEC or table in ("evidence", "write_log", "commit_receipt", "meta", "schema_version", "export_ack"):
            return table
        raise WhitelistError(f"unknown table {table!r}")

    # -- canonical state hash --------------------------------------------------------------------------------
    def canonical_dump(self) -> dict:
        """Deterministic content of the store: independent of autoincrement ids, export acks and store_id."""
        conn = self._open()
        out = {}
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for t in tables:
            if t in INTERNAL_NOT_CANONICAL:
                continue
            cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({t})")]
            if t == "write_log":
                cols = [c for c in cols if c != "wl"]
            if t == "meta":
                rows = conn.execute("SELECT key, value FROM meta WHERE key <> 'store_id'").fetchall()
            else:
                rows = conn.execute(f"SELECT {', '.join(cols)} FROM {t}").fetchall()
            enc = [canon({c: r[c] for c in r.keys()}) for r in rows]
            out[t] = sorted(enc)
        return out

    def state_hash(self) -> str:
        return sha256_hex(canon(self.canonical_dump()))

    # -- integrity -------------------------------------------------------------------------------------------
    def check_integrity(self) -> list[str]:
        conn = self._open()
        bad = []
        r = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if r != "ok":
            bad.append(f"integrity_check: {r}")
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            bad.append("foreign_key_check failed")
        for r in conn.execute("SELECT * FROM write_log"):
            want = sha256_hex(f"{r['event_id']}|{r['op_idx']}|{r['tbl']}|{r['row_key']}|{r['op']}|{r['payload']}|{r['evidence']}")
            if want != r["payload_hash"]:
                bad.append(f"write_log {r['wl']}: payload hash mismatch")
            if r["tbl"] in CLAIM_TABLES and not r["evidence"]:
                bad.append(f"write_log {r['wl']}: claim write without evidence")
        for r in conn.execute("SELECT event_id, n_ops FROM commit_receipt"):
            n = conn.execute("SELECT COUNT(*) FROM write_log WHERE event_id=?", (r["event_id"],)).fetchone()[0]
            if n != r["n_ops"]:
                bad.append(f"commit {r['event_id']}: receipt n_ops {r['n_ops']} != write_log {n}")
        for t in CLAIM_TABLES:
            pk = TABLE_SPEC[t]["pk"]
            for row in conn.execute(f"SELECT {', '.join(pk)} FROM {t}"):
                key = "/".join(str(row[c]) for c in pk)
                if not conn.execute("SELECT 1 FROM evidence WHERE claim_table=? AND claim=? LIMIT 1", (t, f"{t}/{key}")).fetchone():
                    bad.append(f"{t}/{key}: claim row without any evidence")
        return bad

    # -- trace mirror ----------------------------------------------------------------------------------------
    def trace_key(self, event_id: str, idx: int) -> str:
        return f"{self.store_id()}:{event_id}#{idx}"

    def pending_exports(self, limit: int | None = None) -> list:
        conn = self._open()
        sql = ("SELECT w.* FROM write_log w WHERE NOT EXISTS (SELECT 1 FROM export_ack a WHERE a.destination=? AND "
               "a.trace_key = ? || ':' || w.event_id || '#' || w.op_idx) ORDER BY w.wl")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in conn.execute(sql, (self.destination, self.store_id()))]

    def export_pending(self, tracer=None, limit: int | None = None) -> dict:
        """Mirror un-acked write_log rows to the tracer. Contract (docs/step5-plan.md 4): `tracer.emit(key, record)`
        must atomically de-duplicate and persist under `key`; `tracer.has(key)` (optional) only avoids re-sends.
        Delivery is at-least-once; it is exactly-once only against a destination honouring that contract."""
        tracer = tracer or self.tracer
        if tracer is None or self.readonly:
            return {"emitted": 0, "already": 0, "pending": 0}
        emitted = already = 0
        for r in self.pending_exports(limit):
            key = self.trace_key(r["event_id"], r["op_idx"])
            probe = getattr(tracer, "has", None)
            if callable(probe) and probe(key):
                already += 1
            else:
                rec = {"key": key, "store": self.store_id(), "event_id": r["event_id"], "op_idx": r["op_idx"],
                       "table": r["tbl"], "row_key": r["row_key"], "op": r["op"], "payload": json.loads(r["payload"]),
                       "evidence": r["evidence"], "payload_hash": r["payload_hash"]}
                tracer.emit(key, rec)
                emitted += 1
            self._fault("export_after_emit")
            conn = self._open()
            conn.execute("INSERT OR IGNORE INTO export_ack(trace_key, destination) VALUES (?,?)", (key, self.destination))
        return {"emitted": emitted, "already": already, "pending": len(self.pending_exports())}

    def _mirror(self):
        if self.tracer is None:
            return
        try:
            self.export_pending()
        except Exception as e:          # the commit already succeeded; a failed mirror is retried by export_pending()
            self.mirror_errors.append(f"{type(e).__name__}: {e}")

    # -- snapshot / restore ----------------------------------------------------------------------------------
    def snapshot(self, directory, extra: dict | None = None, overwrite: bool = False) -> dict:
        """Copy via the SQLite backup API + manifest derived from the FINISHED copy (not the live source)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / "ltm.db"
        if dest.exists() and not overwrite:
            raise StoreError(f"snapshot target {dest} exists")
        if dest.exists():
            dest.unlink()
        with self._lock:
            conn = self._open()
            if conn.in_transaction:
                raise StoreError("snapshot during an open transaction")
            bk = sqlite3.connect(str(dest))
            try:
                conn.backup(bk)
                bk.execute("PRAGMA journal_mode=DELETE")
            finally:
                bk.close()
        copy = Store(dest, readonly=True)
        try:
            counts = {t: copy.count(t) for t in sorted(TABLE_SPEC)}
            man = {"format": STORE_FORMAT, "state_hash": copy.state_hash(), "file_sha256": file_sha256(dest),
                   "E_frozen": copy.clock(), "schema_version": copy.version, "tables": counts,
                   "integrity": copy.check_integrity(), "extra": extra or {}}
        finally:
            copy.close()
        (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return man

    @staticmethod
    def read_manifest(directory) -> dict:
        return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))

    @staticmethod
    def verify_snapshot(directory) -> list[str]:
        """Empty list = the snapshot's file bytes and canonical state still match its manifest."""
        d = Path(directory)
        try:
            man = Store.read_manifest(d)
        except (OSError, ValueError) as e:
            return [f"manifest unreadable: {e}"]
        bad = []
        if file_sha256(d / "ltm.db") != man["file_sha256"]:
            bad.append("file sha256 differs from manifest")
        s = Store(d / "ltm.db", readonly=True)
        try:
            if s.state_hash() != man["state_hash"]:
                bad.append("canonical state hash differs from manifest")
            if s.clock() != man["E_frozen"]:
                bad.append("clock differs from manifest")
        except StoreError as e:
            bad.append(str(e))
        finally:
            s.close()
        return bad

    @staticmethod
    def restore_snapshot(directory, dest_path, **kw) -> "Store":
        """Copy a verified snapshot to `dest_path` (a fresh writable store) and open it."""
        bad = Store.verify_snapshot(directory)
        if bad:
            raise StoreError("snapshot failed verification: " + "; ".join(bad))
        dest = Path(dest_path)
        if dest.exists():
            raise StoreError(f"restore target {dest} exists")
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(Path(directory).joinpath("ltm.db").resolve().as_uri() + "?mode=ro", uri=True)
        try:
            out = sqlite3.connect(str(dest))
            try:
                src.backup(out)
            finally:
                out.close()
        finally:
            src.close()
        s = Store(dest, **kw)
        s.ensure()
        return s

    @staticmethod
    def open_frozen(directory, testlog_path=None) -> "Store":
        """Open a verified snapshot read-only. Per-test logs go to a separate writable testlog store (never the snapshot)."""
        bad = Store.verify_snapshot(directory)
        if bad:
            raise StoreError("snapshot failed verification: " + "; ".join(bad))
        s = Store(Path(directory) / "ltm.db", readonly=True)
        s.ensure()
        s.testlog = Store(testlog_path) if testlog_path is not None else None
        s._testlog_n = 0
        return s

    def log_test(self, table: str, values: dict) -> dict:
        """Append a retrieval/application log row to the frozen store's separate testlog.db."""
        tl = getattr(self, "testlog", None)
        if not self.readonly or tl is None:
            raise StoreError("log_test is for frozen stores opened with a testlog path")
        if table not in LOG_TABLES:
            raise WhitelistError(f"testlog accepts only {LOG_TABLES}")
        self._testlog_n += 1
        return tl.commit(f"tl:{sha256_hex(canon([table, values]))[:16]}", "test_log",
                         [{"table": table, "op": "insert", "values": values}])

    def stats(self) -> dict:
        conn = self._open()
        pc = conn.execute("PRAGMA page_count").fetchone()[0]
        ps = conn.execute("PRAGMA page_size").fetchone()[0]
        return {"path": str(self.path), "schema_version": self.version, "E": self.clock(), "pragmas": self.pragmas,
                "rows": {t: self.count(t) for t in sorted(TABLE_SPEC)}, "bytes": pc * ps,
                "mirror_errors": list(self.mirror_errors)}
