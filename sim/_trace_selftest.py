"""Observability-plane self-test — the gate for `sim/trace.py` and the five instrumented modules.

Implements `docs/observability-design.md` §12 one-to-one (groups A..H, checks T-1..T-53). Pure stdlib, fully
offline: no AI2-THOR (ai2thor is stubbed at import time like `_phase6_selftest.py` / `_phase7_selftest.py`),
no Ollama, no network, no LLM. Every fixture — trace store, knowledge base, memory sqlite, loop-report jsonl —
is built inside one `tempfile.mkdtemp()` that is removed afterwards, and `trace`'s browse paths (`KB_PATH`,
`KB_AUDIT_PATH`, `MEM_DB_PATH`, `LOOP_REPORTS_PATH`) are redirected at those fixtures **before the first API
call**, so the real research data (`sim/knowledge_base.json`, `sim/memory.db`, `sim/loop_reports.jsonl`,
`sim/trace.db`) is never read, never written, and never even opened.

What it proves, in the author's own words ("I can check which actions resulted in which additions to the
database"):

  * **end-to-end causality** — a simulated run (command → plan → two steps → one failure) against a temp KB;
    querying the created KB entry returns the exact action that caused it, and querying that action returns
    exactly the entries it wrote (groups C, H);
  * **the merge path** — the same procedure written twice from two *different* runs is ONE artifact with TWO
    recorded causes, distinguishable as `insert` then `reinforce` (T-16, T-17, T-48);
  * **failure lineage** — a lesson written after a failure links back to the failing action and its error
    class (T-51);
  * **opt-out safety** — tracing off, and the trace DB unwritable: the agent still runs, nothing escapes,
    nothing is written (T-27, T-28, T-29);
  * **thread and process safety**, **query correctness**, the **read-only** guarantee over `memory.db`, and the
    §7.1 **overhead budget**.

Each group gets its OWN trace store, so the design's exact-count assertions ("orphans == 1") are exact rather
than cumulative. Prints a per-check PASS/FAIL table and exits non-zero on any failure.

Usage:
    python sim/_trace_selftest.py             # groups A..H
    python sim/_trace_selftest.py --regress   # ... plus T-53 (shells out to the four existing self-tests,
                                              #     twice: TRACE_ENABLED=1 and TRACE_ENABLED=0)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types

_results: list = []          # (name, ok, detail)


def _check(name: str, cond, detail: str = "") -> bool:
    ok = bool(cond)
    _results.append((name, ok, detail))
    return ok


def _stub_ai2thor():
    """Import-time placeholder: improve._error_hint lazily imports thor_actions, which imports ai2thor."""
    ai2thor = types.ModuleType("ai2thor")
    controller = types.ModuleType("ai2thor.controller")

    class Controller:  # never instantiated here
        pass

    controller.Controller = Controller
    ai2thor.controller = controller
    sys.modules.setdefault("ai2thor", ai2thor)
    sys.modules.setdefault("ai2thor.controller", controller)


# --------------------------------------------------------------------------- tiny helpers
def _sql(path, q, args=()):
    """Read rows straight out of a sqlite file (the tests assert on the STORE, not only on the API)."""
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in c.execute(q, args)]
    finally:
        c.close()


def _digest(path):
    """(sha256, size, mtime) — the read-only proof for a fixture file."""
    with open(path, "rb") as fh:
        return (hashlib.sha256(fh.read()).hexdigest(), os.path.getsize(path), os.path.getmtime(path))


_VOLATILE = ("ts", "time", "last_seen", "first_seen", "run_id", "trace_span_id", "decision_id")


def _norm(obj):
    """Normalise ids/timestamps out of a store dump so two runs can be compared byte-for-byte."""
    if isinstance(obj, dict):
        return {k: ("<v>" if k in _VOLATILE else _norm(v)) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_norm(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _mem_counts(db_path):
    out = {}
    if not os.path.exists(db_path):
        return out
    c = sqlite3.connect(db_path)
    try:
        for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            out[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    finally:
        c.close()
    return out


# --------------------------------------------------------------------------- the suite
class Suite:
    def __init__(self, tmp: str, here: str, tr):
        self.tmp = tmp
        self.here = here
        self.T = tr

    # -- store handling ----------------------------------------------------
    def fresh(self, name: str, **kw) -> str:
        """Point the tracer at a brand-new temp store and clear all ambient context."""
        T = self.T
        with T._ambient_lock:
            T._AMBIENT.clear()
        T._CTX.set(None)
        kw.setdefault("enabled", True)
        kw.setdefault("async_writes", True)
        kw.setdefault("origin", "test")
        path = os.path.join(self.tmp, name)
        T.configure(db_path=path, **kw)
        return path

    def sub(self, *parts) -> str:
        p = os.path.join(self.tmp, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    # ================================================================= GROUP A — store & schema
    def group_a(self):
        T = self.T
        p = self.fresh("A.db")
        T.emit("console.log", summary="open")
        T.flush(5)

        tables = {r["name"] for r in _sql(p, "SELECT name FROM sqlite_master WHERE type='table'")}
        want_t = {"runs", "spans", "events", "artifacts", "writes", "edges", "meta"}
        idx = {r["name"] for r in _sql(p, "SELECT name FROM sqlite_master WHERE type='index'")}
        want_i = {"ix_runs_started", "ix_runs_decision", "ix_runs_kind", "ix_runs_scene",
                  "ix_spans_run", "ix_spans_parent", "ix_spans_kind", "ix_spans_open",
                  "ix_events_run", "ix_events_type", "ix_events_span", "ix_events_level", "ix_events_ts",
                  "ix_art_store", "ix_art_ref", "ix_art_dedup", "ix_art_src",
                  "ix_writes_ptr", "ix_writes_art", "ix_writes_span", "ix_writes_run", "ix_writes_seq",
                  "ix_edges_src", "ix_edges_dst", "ix_edges_rel"}
        _check("T-1 fresh store creates all 7 tables and all 25 named indexes",
               want_t <= tables and want_i <= idx,
               f"missing tables={sorted(want_t - tables)} indexes={sorted(want_i - idx)}")

        jm = T._conn.execute("PRAGMA journal_mode").fetchone()[0]
        sy = T._conn.execute("PRAGMA synchronous").fetchone()[0]
        _check("T-2 journal_mode=wal, synchronous=1 (NORMAL) on the writer connection",
               str(jm).lower() == "wal" and int(sy) == 1, f"journal_mode={jm} synchronous={sy}")

        n_ev = _sql(p, "SELECT COUNT(*) n FROM events")[0]["n"]
        self.fresh("A.db")                                    # re-open the SAME store
        T.flush(5)
        tables2 = {r["name"] for r in _sql(p, "SELECT name FROM sqlite_master WHERE type='table'")}
        n_ev2 = _sql(p, "SELECT COUNT(*) n FROM events")[0]["n"]
        meta = {r["key"]: r["value"] for r in _sql(p, "SELECT key, value FROM meta")}
        _check("T-3 re-opening an existing store is a no-op: tables and data intact",
               tables2 == tables and n_ev2 == n_ev and meta.get("schema_version") == "1",
               f"tables={len(tables2)} events {n_ev}->{n_ev2} meta={meta}")
        _check("T-4 meta carries schema_version + created_ts + trace_api_version",
               {"schema_version", "created_ts", "trace_api_version"} <= set(meta), str(sorted(meta)))

        # seq comes from a per-process BLOCK reserved off the meta high-water mark (trace._reserve_seq_block),
        # which is what stops two processes minting the same primary key (T-37). Within a process it is still
        # gapless and monotonic; it just does not restart at MAX(seq)+1 after a re-open, and the unused tail of
        # a block is skipped. Both properties are asserted here.
        base = _sql(p, "SELECT COALESCE(MAX(seq),0) m FROM events")[0]["m"]
        hwm = int({r["key"]: r["value"] for r in _sql(p, "SELECT key, value FROM meta")}.get("seq_hwm", 0))
        seqs = [T.emit("console.log", summary=f"seq {i}", level="DEBUG") for i in range(500)]
        _check("T-5 events.seq is strictly increasing with no gaps across 500 emits, above every seq already "
               "stored and inside this process's reserved block",
               seqs == list(range(seqs[0], seqs[0] + 500)) and seqs[0] > base and seqs[-1] <= hwm
               and T.flush(5)
               and [r["seq"] for r in _sql(p, "SELECT seq FROM events WHERE seq>? ORDER BY seq", (base,))]
               == seqs, f"first={seqs[:2]} last={seqs[-2:]} base={base} hwm={hwm}")

        row = T._event_row("trace.warning", "dupe", None, "DEFAULT", None, None, None, "system:webapp",
                           seq=T._next_seq())
        T._enqueue("E", row)
        T._enqueue("E", (T._next_seq(),) + tuple(row[1:]))    # same event_id, different seq
        T.flush(5)
        n_dupe = _sql(p, "SELECT COUNT(*) n FROM events WHERE event_id=?", (row[1],))[0]["n"]
        _check("T-6 event_id is idempotent: the same event twice inserts ONE row", n_dupe == 1, f"rows={n_dupe}")

    # ================================================================= GROUP B — identity & context
    def group_b(self):
        T = self.T
        p = self.fresh("B.db")

        rids = [T.new_run_id() for _ in range(1000)]
        sids = [T.new_span_id() for _ in range(1000)]
        _check("T-7 run ids are 32-hex, span ids 16-hex, 1000 of each unique",
               all(re.fullmatch(r"[0-9a-f]{32}", x) for x in rids) and len(set(rids)) == 1000
               and all(re.fullmatch(r"[0-9a-f]{16}", x) for x in sids) and len(set(sids)) == 1000)

        with T.run(kind="command", command="nested", scene="FloorPlan1", level="L2") as rid:
            with T.span("invoke_agent", kind="AGENT", op="invoke_agent") as s1:
                with T.span("execute_tool", kind="TOOL", op="execute_tool", step_ix=0) as s2:
                    pass
        T.flush(5)
        sp = {r["span_id"]: r for r in _sql(p, "SELECT * FROM spans WHERE run_id=? ORDER BY seq", (rid,))}
        _check("T-8 run()/span() nest: parent_span_id, depth and seq are correct",
               sp.get(s2, {}).get("parent_span_id") == s1 and sp.get(s1, {}).get("parent_span_id") is None
               and sp[s1]["depth"] == 0 and sp[s2]["depth"] == 1
               and sp[s1]["seq"] == 0 and sp[s2]["seq"] == 1,
               f"outer={sp.get(s1)} inner={sp.get(s2)}")

        with T._ambient_lock:
            T._AMBIENT.clear()
        T._CTX.set(None)
        art9 = T.record_write(store="kb", table_or_kind="fact", pointer=999, verdict="insert",
                              why="no run context at all", source="human")
        T.flush(5)
        w9 = _sql(p, "SELECT * FROM writes WHERE pointer='999'")
        orph = T.stats()["orphans"]
        _check("T-9 a write with no run context is RECORDED as an orphan, never dropped",
               len(w9) == 1 and w9[0]["run_id"] is None and w9[0]["artifact_id"] == "kb:fact:999"
               and art9 == "kb:fact:999" and orph["writes"] == 1,
               f"rows={len(w9)} run_id={w9[0]['run_id'] if w9 else '?'} orphans={orph}")

        rid10 = T.start_run(kind="command", command="cross-thread", set_ambient=False)
        got = {}

        def _worker():
            got["orphan"] = T.record_write(store="mem", table_or_kind="episodes", pointer=7001,
                                           verdict="insert", why="unbound worker")
            with T.bound(run_id=rid10):
                got["bound"] = T.record_write(store="mem", table_or_kind="episodes", pointer=7002,
                                              verdict="insert", why="bound worker")

        th = threading.Thread(target=_worker, name="worker")
        th.start()
        th.join(10)
        T.end_run(rid10)
        T.flush(5)
        w_un = _sql(p, "SELECT * FROM writes WHERE pointer='7001'")
        w_bd = _sql(p, "SELECT * FROM writes WHERE pointer='7002'")
        _check("T-10 cross-thread hand-off: bound() attributes the run, unbound is an orphan (both branches)",
               len(w_un) == 1 and w_un[0]["run_id"] is None
               and len(w_bd) == 1 and w_bd[0]["run_id"] == rid10,
               f"unbound={w_un[0]['run_id'] if w_un else '?'} bound={w_bd[0]['run_id'] if w_bd else '?'}")

        r1 = T.start_run(kind="command", command="R1 outer", set_ambient=True)
        T.record_write(store="kb", table_or_kind="fact", pointer=8001, verdict="insert", why="under R1")
        with T.run(kind="teach", command="R2 inner") as r2:
            T.record_write(store="kb", table_or_kind="fact", pointer=8002, verdict="insert", why="under R2")
        T.record_write(store="kb", table_or_kind="fact", pointer=8003, verdict="insert", why="back under R1")
        T.end_run(r1)
        T.flush(5)
        att = {r["pointer"]: r["run_id"] for r in
               _sql(p, "SELECT pointer, run_id FROM writes WHERE pointer IN ('8001','8002','8003')")}
        _check("T-11 contextvar outranks the process ambient, and the ambient is restored after the block",
               att.get("8001") == r1 and att.get("8002") == r2 and att.get("8003") == r1,
               f"{att} r1={r1[:8]} r2={r2[:8]}")

        r12 = T.start_run(kind="command", command="leaky span")
        s12 = T.start_span("never closed", kind="TOOL")
        T.end_run(r12)
        T.flush(5)
        row12 = _sql(p, "SELECT * FROM spans WHERE span_id=?", (s12,))
        _check("T-12 end_run closes a span left open, as ERROR 'span not closed'",
               row12 and row12[0]["status"] == "ERROR" and row12[0]["error"] == "span not closed"
               and row12[0]["end_ts"] is not None, str(row12[:1]))

        raised = False
        r13 = None
        try:
            with T.run(kind="command", command="boom") as r13:
                raise ValueError("kaboom")
        except ValueError:
            raised = True
        T.flush(5)
        row13 = _sql(p, "SELECT * FROM runs WHERE run_id=?", (r13,))
        _check("T-13 run() re-raises the caller's exception AND records status=ERROR",
               raised and row13 and row13[0]["status"] == "ERROR" and "kaboom" in (row13[0]["error"] or ""),
               f"raised={raised} row={row13[:1]}")

    # ================================================================= GROUP C — causality (the core)
    def group_c(self):
        T = self.T
        p = self.fresh("C.db")

        rid = T.start_run(kind="command", command="put the apple in the fridge", scene="FloorPlan1",
                          level="L2", lang="en")
        sid = T.start_span("execute_tool pick_and_place Apple -> Fridge", kind="TOOL", op="execute_tool",
                           step_ix=0, run_id=rid)
        for store, kind, ptr in (("kb", "procedure", 12), ("mem", "episodes", 301), ("mem", "spatial", 44)):
            T.record_write(store=store, table_or_kind=kind, pointer=ptr, verdict="insert",
                           why="first cause", source="environment", occurrences=1,
                           after={"occurrences": 1}, run_id=rid, span_id=sid)
        T.end_span(sid)
        T.end_run(rid)
        T.flush(5)
        eff = T.span_effects(sid)
        eqp = " ".join(str(r["detail"]) for r in
                       _sql(p, "EXPLAIN QUERY PLAN SELECT * FROM writes WHERE span_id=? ORDER BY id", (sid,)))
        _check("T-14 'what did this action write?': span_effects returns exactly 3 rows via ix_writes_span",
               len(eff) == 3 and {r["artifact_id"] for r in eff} ==
               {"kb:procedure:12", "mem:episode:301", "mem:spatial:44"} and "ix_writes_span" in eqp,
               f"n={len(eff)} plan={eqp}")

        for i in range(4):
            with T.run(kind="command", command=f"repeat {i}", scene="FloorPlan1") as r:
                with T.span("close_loop", kind="LOOP", op="close_loop", actor="system:improve") as s:
                    T.record_write(store="kb", table_or_kind="procedure", pointer=12, verdict="reinforce",
                                   why=f"repeat learning {i}", source="environment", occurrences=i + 2,
                                   before={"occurrences": i + 1}, after={"occurrences": i + 2})
        T.flush(5)
        art = _sql(p, "SELECT * FROM artifacts WHERE artifact_id='kb:procedure:12'")
        w12 = _sql(p, "SELECT * FROM writes WHERE artifact_id='kb:procedure:12' ORDER BY ts, id")
        _check("T-16 many causes: ONE artifact row, 5 writes, revision=5, n_causes=5, no cause overwritten",
               len(art) == 1 and len(w12) == 5 and art[0]["revision"] == 5 and art[0]["n_causes"] == 5
               and len({w["run_id"] for w in w12}) == 5,
               f"arts={len(art)} writes={len(w12)} rev={art[0]['revision'] if art else '?'} "
               f"n_causes={art[0]['n_causes'] if art else '?'}")

        causes = T.artifact_causes("kb:procedure:12")
        eqp2 = " ".join(str(r["detail"]) for r in
                        _sql(p, "EXPLAIN QUERY PLAN SELECT * FROM writes WHERE artifact_id=? ORDER BY ts",
                             ("kb:procedure:12",)))
        eqp3 = " ".join(str(r["detail"]) for r in _sql(
            p, "EXPLAIN QUERY PLAN SELECT * FROM writes WHERE store=? AND table_or_kind=? AND pointer=?",
            ("kb", "procedure", "12")))
        _check("T-15 'what produced this KB entry?': causes oldest-first, joined to run command + span name, "
               "one indexed query",
               len(causes) == 5 and causes[0]["run_command"] == "put the apple in the fridge"
               and causes[0]["span_name"].startswith("execute_tool pick_and_place")
               and causes[-1]["run_command"] == "repeat 3"
               and all(c["ts"] <= d["ts"] for c, d in zip(causes, causes[1:]))
               and "ix_writes_art" in eqp2 and "ix_writes_ptr" in eqp3,
               f"n={len(causes)} art_plan={eqp2} ptr_plan={eqp3}")

        verdicts = [c["verdict"] for c in causes]
        occ = [json.loads(c["after_json"] or "{}").get("occurrences") for c in causes]
        before = [json.loads(c["before_json"] or "null") for c in causes]
        _check("T-17 insert vs reinforce stay distinguishable and the counters advance 1->5",
               verdicts == ["insert", "reinforce", "reinforce", "reinforce", "reinforce"]
               and occ == [1, 2, 3, 4, 5] and before[0] is None
               and [b.get("occurrences") for b in before[1:]] == [1, 2, 3, 4],
               f"verdicts={verdicts} after={occ} before={before}")

        # ---- the §3.4 lineage fixture, complete with the self-improving CYCLE it must survive
        p2 = self.fresh("C2.db")
        rid = T.start_run(kind="command", command="put the apple in the fridge", scene="FloorPlan1",
                          agent_id="gemma3:4b")
        root = T.start_span("invoke_agent", kind="AGENT", op="invoke_agent", run_id=rid)
        tool = T.start_span("execute_tool pick_and_place", kind="TOOL", op="execute_tool", run_id=rid,
                            parent_span_id=root, actor="env:ai2thor")
        T.record_write(store="mem", table_or_kind="episodes", pointer=301, verdict="insert",
                       why="executed step", source="environment", run_id=rid, span_id=tool)
        T.end_span(tool)
        loop = T.start_span("close_loop", kind="LOOP", op="close_loop", run_id=rid, parent_span_id=root,
                            actor="system:improve")
        T.record_use(store="kb", table_or_kind="lesson", pointer=7, why="retrieved into the planner prompt",
                     run_id=rid, span_id=loop)
        T.record_write(store="kb", table_or_kind="procedure", pointer=12, verdict="insert",
                       why="loop closure", source="environment", derived_from=["art:mem:episode:301"],
                       run_id=rid, span_id=loop)
        T.end_span(loop)
        T.end_span(root)
        T.end_run(rid)
        # the cycle: a later run RETRIEVES procedure 12 and then REINFORCES it
        rid2 = T.start_run(kind="command", command="put the apple in the fridge (again)", scene="FloorPlan1")
        cyc = T.start_span("close_loop", kind="LOOP", op="close_loop", run_id=rid2)
        T.record_use(store="kb", table_or_kind="procedure", pointer=12, why="reused", run_id=rid2, span_id=cyc)
        T.record_write(store="kb", table_or_kind="procedure", pointer=12, verdict="reinforce",
                       why="reused successfully", source="environment", run_id=rid2, span_id=cyc)
        T.end_span(cyc)
        T.end_run(rid2)
        T.flush(5)

        up = T.lineage("art:kb:procedure:12", direction="up", depth=16, limit=300)
        naive = _sql(p2, """
            WITH RECURSIVE up(node, rel, depth) AS (
                SELECT ?, NULL, 0
              UNION
                SELECT e.dst, e.rel, up.depth + 1 FROM edges e JOIN up ON e.src = up.node WHERE up.depth < ?
            ) SELECT node FROM up WHERE depth > 0""", ("art:kb:procedure:12", 16))
        _check("T-18 cycle safety: the guarded CTE terminates and is strictly smaller than the naive UNION",
               len(up["nodes"]) <= 12 and len(naive) > len(up["nodes"]),
               f"guarded={len(up['nodes'])} naive_rows={len(naive)}")

        d = {n["id"]: n["depth"] for n in up["nodes"]}
        _check("T-19 lineage(up) reaches the generating span (1), the derived-from episode (1), "
               "the used lesson (2) and the responsible agent (1)",
               d.get(f"span:{loop}") == 1 and d.get("art:mem:episode:301") == 1
               and d.get("art:kb:lesson:7") == 2 and d.get("agent:env:ai2thor") == 1
               and d.get(f"span:{tool}") == 2 and d.get(f"run:{rid}") is not None,
               json.dumps(d, ensure_ascii=False)[:200])

        down = T.lineage(f"span:{loop}", direction="down", depth=6)
        _check("T-20 lineage(down) from the loop span reaches the artifacts it generated",
               "art:kb:procedure:12" in {n["id"] for n in down["nodes"]},
               str(sorted(n["id"] for n in down["nodes"]))[:200])

        T.link("art:kb:fact:1", "span:deadbeefdeadbeef", "wasGeneratedBy", note="once")
        T.link("art:kb:fact:1", "span:deadbeefdeadbeef", "wasGeneratedBy", note="twice")
        T.link("art:kb:fact:1", "span:deadbeefdeadbeef", "notARealRelation")
        T.flush(5)
        n_edge = _sql(p2, "SELECT COUNT(*) n FROM edges WHERE src='art:kb:fact:1'")[0]["n"]
        n_bad = _sql(p2, "SELECT COUNT(*) n FROM edges WHERE rel='notARealRelation'")[0]["n"]
        _check("T-21 edges are idempotent (UNIQUE src,dst,rel) and an unknown PROV relation is dropped",
               n_edge == 1 and n_bad == 0, f"edges={n_edge} bad={n_bad}")

        with T.run(kind="command", command="capture") as rc:
            with T.span("save capture", kind="TOOL") as sc:
                art22 = T.record_write(store="file", table_or_kind="captures/x.jpg",
                                       pointer="captures/x.jpg", verdict="insert", why="frame saved",
                                       source="environment")
        T.flush(5)
        lin22 = T.lineage("art:" + art22, direction="up", depth=4)
        _check("T-22 node ids never contain '/': _safe_ref escapes it and the id survives a lineage round-trip",
               T._safe_ref("captures/x.jpg") == "captures%2Fx.jpg" and "/" not in art22
               and art22 == "file:image:captures%2Fx.jpg"
               and f"span:{sc}" in {n["id"] for n in lin22["nodes"]},
               f"art={art22} nodes={[n['id'] for n in lin22['nodes']]}")

        with T.run(kind="maint", command="prune") as rm:
            with T.span("prune spatial", kind="MAINT", op="maintenance", actor="system:memory") as sm:
                T.record_write(store="mem", table_or_kind="spatial", pointer=77, verdict="insert",
                               why="observed", source="environment")
                T.record_write(store="mem", table_or_kind="spatial", pointer=77, verdict="delete",
                               why="decayed 0.004 < min_confidence 0.05", source="environment",
                               before={"confidence": 1.0, "decayed": 0.004})
        T.flush(5)
        a23 = _sql(p2, "SELECT * FROM artifacts WHERE artifact_id='mem:spatial:77'")
        e23 = _sql(p2, "SELECT * FROM edges WHERE src='art:mem:spatial:77' AND rel='wasInvalidatedBy'")
        c23 = T.artifact_causes("mem:spatial:77")
        _check("T-23 delete sets alive=0 + wasInvalidatedBy, and KEEPS the artifact row and its prior causes",
               len(a23) == 1 and a23[0]["alive"] == 0 and len(e23) == 1 and len(c23) == 2
               and [c["verdict"] for c in c23] == ["insert", "delete"],
               f"alive={a23[0]['alive'] if a23 else '?'} invalidated={len(e23)} causes={len(c23)}")

        p3 = self.fresh("C3.db")
        rnd = random.Random(20260816)
        stores = [("kb", "procedure"), ("kb", "lesson"), ("mem", "episodes"), ("mem", "spatial"),
                  ("mem", "failures")]
        with T.run(kind="command", command="randomised consistency sweep", scene="FloorPlan2") as rr:
            with T.span("sweep", kind="TOOL") as ss:
                for _ in range(200):
                    st, tk = rnd.choice(stores)
                    T.record_write(store=st, table_or_kind=tk, pointer=rnd.randint(1, 25),
                                   verdict=rnd.choice(["insert", "reinforce", "update", "noop"]),
                                   why="sweep", source=rnd.choice(["human", "environment", "prior", "demo"]))
        T.flush(5)
        bad_fk = _sql(p3, "SELECT COUNT(*) n FROM writes w LEFT JOIN artifacts a"
                          " ON a.artifact_id=w.artifact_id WHERE a.artifact_id IS NULL")[0]["n"]
        bad_n = _sql(p3, "SELECT COUNT(*) n FROM artifacts a WHERE a.n_causes <>"
                         " (SELECT COUNT(*) FROM writes w WHERE w.artifact_id=a.artifact_id)")[0]["n"]
        bad_edge = _sql(p3, "SELECT COUNT(*) n FROM (SELECT DISTINCT artifact_id, span_id FROM writes"
                            " WHERE verdict='insert') x LEFT JOIN edges e"
                            " ON e.src='art:'||x.artifact_id AND e.dst='span:'||x.span_id"
                            " AND e.rel='wasGeneratedBy' WHERE e.id IS NULL")[0]["n"]
        _check("T-24 consistency after 200 randomised writes: every artifact_id resolves, n_causes matches "
               "COUNT(*), every insert has a wasGeneratedBy edge",
               bad_fk == 0 and bad_n == 0 and bad_edge == 0,
               f"dangling={bad_fk} n_causes_mismatch={bad_n} missing_generated_by={bad_edge}")

    # ================================================================= GROUP D — non-regression / opt-out
    def _exercise(self, kb, mem, reports, improve, scene="FloorPlan1"):
        """One deterministic pass over every instrumented write path; returns the raw return VALUES."""
        out = {}
        plan = [{"action": "go_to", "object": "Kettle"}, {"action": "toggle_on", "object": "Kettle"}]
        out["add"] = kb.add_procedure("boil water", plan, why="seeded", scene=scene, source="human")
        out["upsert_p"] = kb.upsert_procedure("boil water", plan, why="relearned", scene=scene)
        out["upsert_l"] = kb.upsert_lesson("open the fridge", "open Fridge failed: Fridge is closed",
                                           "open the door before reaching in", why="failed",
                                           scene=scene, error="Fridge is closed")
        # ... and again, so the MERGE branch runs too (it is the one that writes the 'UPSERT lesson' line).
        out["upsert_l2"] = kb.upsert_lesson("open the fridge", "open Fridge failed: Fridge is closed",
                                            "open the door before reaching in", why="failed again",
                                            scene=scene, error="Fridge is closed")
        out["reinforce"] = kb.reinforce(out["add"]["id"])
        out["fact"] = kb.add_fact("kettles live on the counter", why="taught", source="human")
        out["episode"] = mem.log_episode(scene=scene, command="boil water", level="L1", action="toggle_on",
                                         args={"objectType": "Kettle"}, success=True, ts=1.0)
        out["observe"] = mem.observe_location(scene, "Kettle", receptacle="CounterTop", track="oracle", ts=2.0)
        out["failure"] = mem.record_failure("pickup", "PickupObject failed: Mug|+01.2 not visible",
                                            object_type="Mug", scene=scene, command="pick up the mug", ts=3.0)
        out["rating"] = mem.set_rating(scene, "boil water", 1.0, ts=None)
        out["loop"] = improve.close_loop(kb, mem, scene=scene, command="boil water", level="L1",
                                         arm="plan_fresh", executed_steps=plan,
                                         step_results=[{"ok": True, "error": None},
                                                       {"ok": True, "error": None}],
                                         goal={"applicable": True, "ok": True, "target": "Kettle",
                                               "held": None},
                                         rag_enabled=False, loop_enabled=True, report_path=reports)
        return out

    def _stores(self, tag, improve, knowledge, memory):
        d = self.sub("stores", tag)
        kb = knowledge.KnowledgeBase(path=os.path.join(d, "knowledge_base.json"))
        mem = memory.Memory(path=os.path.join(d, "memory.db"))
        reports = os.path.join(d, "loop_reports.jsonl")
        return d, kb, mem, reports

    def group_d(self, improve, knowledge, memory, feedback):
        T = self.T
        p = self.fresh("D.db")
        d_on, kb, mem, rep = self._stores("traced", improve, knowledge, memory)
        with T.run(kind="command", command="boil water", scene="FloorPlan1", level="L1", lang="en"):
            res = self._exercise(kb, mem, rep, improve)
        T.flush(5)

        loop = res["loop"]
        orig12 = {"ts", "scene", "command", "level", "arm", "retrieved", "executed_steps", "outcome",
                  "writes", "related_ids", "reindexed", "loop_enabled"}
        _check("T-25 with tracing ON every instrumented function returns its ORIGINAL value/shape",
               isinstance(res["add"], dict) and isinstance(res["add"].get("id"), int)
               and set(res["upsert_p"]) == {"action", "id", "entry"}
               and set(res["upsert_l"]) == {"action", "id", "entry"}
               and res["upsert_p"]["action"] == "reinforce"
               and isinstance(res["episode"], int)
               and set(res["observe"]) == {"action", "id"}
               and set(res["failure"]) == {"signature", "occurrences"}
               and isinstance(res["rating"], int) and res["rating"] >= 1
               and isinstance(loop, dict) and orig12 <= set(loop),
               f"upsert_p={res['upsert_p'].get('action')} episode={type(res['episode']).__name__} "
               f"rating={res['rating']} loop_missing={sorted(orig12 - set(loop))}")

        with open(kb.audit_path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        pat = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d  \[[^\]]+\](  <[^>]+>)?  .*")
        blob = "\n".join(lines)
        bad = [ln for ln in lines if not pat.match(ln)]
        _check("T-26 the legacy audit-line format is untouched (every line matches, all op strings present)",
               not bad and "UPSERT procedure" in blob and "UPSERT lesson" in blob
               and "<environment>" in blob and "[ADD procedure]" in blob,
               f"unmatched={bad[:1]}")

        # ---- T-27 full opt-out
        mem.close()
        off_db = os.path.join(self.tmp, "MUST_NOT_EXIST.db")
        with T._ambient_lock:
            T._AMBIENT.clear()
        T._CTX.set(None)
        st_off = T.configure(db_path=off_db, enabled=False)
        d_off, kb2, mem2, rep2 = self._stores("untraced", improve, knowledge, memory)
        self._exercise(kb2, mem2, rep2, improve)
        mem2.close()
        kb_on = _norm(json.load(open(os.path.join(d_on, "knowledge_base.json"), encoding="utf-8")))
        kb_off = _norm(json.load(open(os.path.join(d_off, "knowledge_base.json"), encoding="utf-8")))
        benign = (T.emit("console.log", summary="x") == 0
                  and T.record_write(store="kb", table_or_kind="fact", pointer=1, verdict="insert") == ""
                  and re.fullmatch(r"[0-9a-f]{32}", T.start_run(kind="command")) is not None
                  and re.fullmatch(r"[0-9a-f]{16}", T.start_span("s")) is not None
                  and T.flush(1) is True and T.record_action("MoveAhead") is None
                  and T.query_events()["events"] == [] and T.lineage("art:kb:fact:1")["nodes"] == []
                  and T.artifact_causes("kb:fact:1") == [] and T.span_effects("x") == []
                  and T.enabled() is False)
        _check("T-27 TRACE_ENABLED=0: no store file, byte-identical KB + memory, every API returns its "
               "benign value",
               not os.path.exists(off_db) and kb_on == kb_off
               and _mem_counts(os.path.join(d_on, "memory.db")) ==
                   _mem_counts(os.path.join(d_off, "memory.db"))
               and st_off["enabled"] is False and benign,
               f"db_created={os.path.exists(off_db)} kb_equal={kb_on == kb_off} benign={benign}")

        # ---- T-28 unwritable store
        blocker = os.path.join(self.tmp, "blocker.file")
        with open(blocker, "w", encoding="utf-8") as fh:
            fh.write("not a directory")
        st_bad = T.configure(db_path=os.path.join(blocker, "trace.db"), enabled=True)
        d_bad, kb3, mem3, rep3 = self._stores("unwritable", improve, knowledge, memory)
        crashed = None
        try:
            self._exercise(kb3, mem3, rep3, improve)
        except Exception as exc:
            crashed = repr(exc)
        mem3.close()
        kb_bad = _norm(json.load(open(os.path.join(d_bad, "knowledge_base.json"), encoding="utf-8")))
        _check("T-28 unwritable trace DB: degraded=True + a reason, no exception escapes, identical results",
               st_bad.get("degraded") is True and st_bad.get("reason") and crashed is None
               and kb_bad == kb_off
               and _mem_counts(os.path.join(d_bad, "memory.db")) ==
                   _mem_counts(os.path.join(d_off, "memory.db")),
               f"degraded={st_bad.get('degraded')} reason={str(st_bad.get('reason'))[:60]} crash={crashed}")

        # ---- T-29 a raising tracer cannot break the agent
        self.fresh("D2.db")
        _d_r, kb4, mem4, rep4 = self._stores("hostile", improve, knowledge, memory)
        saved = {n: getattr(T, n) for n in ("record_write", "record_use", "emit", "link", "span")}

        def _boom(*a, **kw):
            raise RuntimeError("hostile tracer")

        for n in saved:
            setattr(T, n, _boom)
        try:
            fb = feedback.EnvFeedback(mem4, enabled=True)
            r_up = kb4.upsert_procedure("stow the cup", [{"action": "pickup", "object": "Cup"}], why="x")
            r_ep = mem4.log_episode(scene="FloorPlan1", command="stow the cup", action="pickup", success=True)
            r_st = fb.log_step("FloorPlan1", "stow the cup", "L1", "pickup", {"objectType": "Cup"},
                               True, None, None, None)
            r_cl = improve.close_loop(kb4, mem4, scene="FloorPlan1", command="stow the cup", level="L1",
                                      arm="plan_fresh", executed_steps=[{"action": "pickup", "object": "Cup"}],
                                      step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                                      rag_enabled=False, loop_enabled=True, report_path=rep4)
            hostile_crash = None
        except Exception as exc:
            r_up = r_ep = r_st = r_cl = None
            hostile_crash = repr(exc)
        finally:
            for n, fn in saved.items():
                setattr(T, n, fn)
            mem4.close()
        _check("T-29 a tracer that raises on EVERY call cannot break knowledge / memory / feedback / improve",
               hostile_crash is None and isinstance(r_up, dict) and r_up.get("action") == "insert"
               and isinstance(r_ep, int) and isinstance(r_st, dict) and isinstance(r_st.get("episode_id"), int)
               and isinstance(r_cl, dict) and orig12 <= set(r_cl) and r_cl["writes"],
               f"crash={hostile_crash} up={r_up if not isinstance(r_up, dict) else r_up.get('action')} "
               f"step={r_st} loop_writes={(r_cl or {}).get('writes')}")

    # ================================================================= GROUP E — performance
    def group_e(self):
        T = self.T
        p = self.fresh("E.db")
        n = 5000
        t0 = time.perf_counter()
        for i in range(n):
            T.emit("console.log", summary=f"async {i}", level="DEBUG")
        avg = (time.perf_counter() - t0) / n * 1000.0
        drained = T.flush(timeout=5)
        m = T.status()["metrics"]
        rows = _sql(p, "SELECT COUNT(*) n FROM events WHERE type='console.log'")[0]["n"]
        _check("T-30 async budget: 5000 emits average < 0.10 ms (p99 <= 0.10 ms), flush drains, all rows land",
               avg < 0.10 and drained and rows == n and (m["emit_us_p99"] or 0) <= 100.0,
               f"avg={avg:.5f} ms p50={m['emit_us_p50']} us p99={m['emit_us_p99']} us rows={rows}")

        p = self.fresh("E2.db", async_writes=False)
        n2 = 2000
        t0 = time.perf_counter()
        for i in range(n2):
            T.emit("console.log", summary=f"sync {i}", level="DEBUG")
        avg2 = (time.perf_counter() - t0) / n2 * 1000.0
        T.flush(timeout=5)
        m2 = T.status()["metrics"]
        rows2 = _sql(p, "SELECT COUNT(*) n FROM events WHERE type='console.log'")[0]["n"]
        _check("T-31 sync budget (TRACE_ASYNC=0): 2000 emits average < 1.0 ms (p99 <= 1.0 ms)",
               avg2 < 1.0 and rows2 == n2 and (m2["emit_us_p99"] or 0) <= 1000.0,
               f"avg={avg2:.5f} ms p99={m2['emit_us_p99']} us rows={rows2}")

        p = self.fresh("E3.db")
        t0 = time.perf_counter()
        for _ in range(10000):
            T.record_action("Pass")
        el = (time.perf_counter() - t0) * 1000.0
        T.flush(5)
        st = T.status()
        _check("T-32 the Pass/Done skip filter emits ZERO events, costs < 50 ms for 10 000, and counts them",
               st["skipped_actions"] == 10000 and st["events"] == 0 and el < 50.0,
               f"skipped={st['skipped_actions']} events={st['events']} elapsed={el:.2f} ms")

        p = self.fresh("E4.db")
        with T.run(kind="command", command="navigate", scene="FloorPlan1") as rid:
            with T.span("execute_tool go_to", kind="TOOL", op="execute_tool") as sid:
                for _ in range(20):
                    T.record_action("MoveAhead", kwargs={"moveMagnitude": 0.25}, success=True)
                T.record_action("RotateRight", kwargs={"degrees": 90}, success=True)
                for _ in range(3):
                    T.record_action("MoveAhead", kwargs={"moveMagnitude": 0.25}, success=True)
        T.flush(5)
        acts = _sql(p, "SELECT name, repeat FROM spans WHERE kind='ACTION' ORDER BY seq")
        _check("T-33 nav collapse: 20 identical primitives = ONE span with repeat=20; a different action "
               "breaks the run",
               len(acts) == 3 and acts[0]["repeat"] == 20 and acts[1]["repeat"] == 1
               and acts[2]["repeat"] == 3 and acts[0]["name"].startswith("MoveAhead")
               and acts[1]["name"].startswith("RotateRight"), str(acts))

        cap = T._CFG["max_blob"]
        big = "p" * 100_000
        blob = T._blob(big)
        with T.run(kind="command", command="plan"):
            sid = T.record_llm(model="gemma3:4b", endpoint="http://x:11434", prompt=big,
                               completion="ok", duration_ms=12.0)
        T.flush(5)
        stored = _sql(p, "SELECT input_json FROM spans WHERE span_id=?", (sid,))
        inp = (stored[0]["input_json"] if stored else "") or ""
        _check("T-34 blob truncation: a 100 KB prompt is capped at TRACE_MAX_BLOB + 64 with the marker",
               len(blob) <= cap + 64 and blob.endswith("chars]") and "[truncated " in blob
               and 0 < len(inp) <= cap + 64 and inp.endswith("chars]"),
               f"blob={len(blob)} stored={len(inp)} cap={cap}")

        p = self.fresh("E5.db", async_writes=False, batch=10 ** 7)
        T._CFG["queue_max"] = 100
        try:
            for i in range(50):
                T.emit("action.executed", summary=f"debug {i}", level="DEBUG")
            for i in range(50):
                T.emit("command.received", summary=f"default {i}", level="DEFAULT")
            depth_full = len(T._queue)
            for i in range(100):
                T.emit("command.received", summary=f"overflow {i}", level="DEFAULT")
            still_debug = sum(1 for item in T._queue if item[2])
            dropped = T.status()["dropped"]
            depth = len(T._queue)
        finally:
            T._CFG["queue_max"] = 20000
        _check("T-35 queue overflow drops instead of blocking, and drops DEBUG rows first",
               depth_full == 100 and depth <= 100 and dropped == 100 and still_debug == 0,
               f"depth={depth} dropped={dropped} debug_left={still_debug}")

    # ================================================================= GROUP F — concurrency & processes
    def group_f(self):
        T = self.T
        p = self.fresh("F.db")
        got: dict = {}
        errs: list = []

        def _burst(k):
            try:
                got[k] = [T.emit("console.log", summary=f"t{k} {i}", level="DEBUG") for i in range(250)]
            except Exception as exc:
                errs.append(repr(exc))

        ths = [threading.Thread(target=_burst, args=(k,), name=f"burst{k}") for k in range(4)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(30)
        T.flush(10)
        allseq = [s for v in got.values() for s in v]
        rows = _sql(p, "SELECT COUNT(*) n, COUNT(DISTINCT seq) d FROM events WHERE type='console.log'")[0]
        _check("T-36 four threads x 250 emits: 1000 rows, unique monotonic seqs, no lock errors",
               not errs and len(allseq) == 1000 and len(set(allseq)) == 1000
               and rows["n"] == 1000 and rows["d"] == 1000 and T.status()["write_errors"] == 0,
               f"errs={errs[:1]} seqs={len(set(allseq))} rows={rows} write_errors={T.status()['write_errors']}")

        # ---- T-37: the eval-subprocess contract, exercised with a workload that actually OVERLAPS.
        p = self.fresh("F2.db")
        child = os.path.join(self.tmp, "trace_child.py")
        with open(child, "w", encoding="utf-8") as fh:
            fh.write(
                "import os, sys, time\n"
                f"sys.path.insert(0, {self.here!r})\n"
                "import trace as T\n"
                "rid = T.start_run(kind='eval_record', command='child eval record')\n"
                "for i in range(600):\n"
                "    T.emit('eval.scored', summary='child %d' % i)\n"
                "    if i %% 50 == 0: time.sleep(0.01)\n".replace("%%", "%") +
                "T.end_run(rid)\nT.flush(30)\nT.close()\n")
        rid = T.start_run(kind="command", command="eval sweep", origin="webapp")
        T.flush(5)
        env = dict(os.environ, TRACE_DB=p, TRACE_RUN_ID=rid, TRACE_ORIGIN="eval", TRACE_ENABLED="1")
        proc = subprocess.Popen([sys.executable, child], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        for i in range(600):
            T.emit("console.log", summary=f"parent {i}")
            if i % 50 == 0:
                time.sleep(0.01)
        _out, cerr = proc.communicate(timeout=180)
        T.end_run(rid)
        T.flush(20)
        n_par = _sql(p, "SELECT COUNT(*) n FROM events WHERE summary LIKE 'parent %'")[0]["n"]
        n_chi = _sql(p, "SELECT COUNT(*) n FROM events WHERE summary LIKE 'child %'")[0]["n"]
        kid = _sql(p, "SELECT * FROM runs WHERE kind='eval_record'")
        _check("T-37 a second process writes into the same store WHILE the parent writes: both sets present, "
               "child rows carry origin='eval' and the inherited parent_run_id",
               proc.returncode == 0 and n_par == 600 and n_chi == 600
               and len(kid) == 1 and kid[0]["origin"] == "eval" and kid[0]["parent_run_id"] == rid,
               f"rc={proc.returncode} parent={n_par}/600 child={n_chi}/600 "
               f"child_run={{'origin': {kid[0]['origin']!r}, 'parent': {(kid[0]['parent_run_id'] or '')[:8]!r}}}"
               if kid else f"rc={proc.returncode} parent={n_par} child={n_chi} err={cerr[:120]}")

        p = self.fresh("F3.db")
        stop = threading.Event()
        rerr: list = []
        seen: list = []

        def _writer():
            i = 0
            while not stop.is_set():
                T.emit("console.log", summary=f"live {i}", level="DEBUG")
                i += 1
                if i % 25 == 0:          # keep a batch permanently in flight without starving the writer
                    time.sleep(0.001)

        def _reader():
            try:
                for _ in range(40):
                    r = T.query_events(limit=50)
                    if r.get("error"):
                        rerr.append(r["error"])
                    seen.append(len(r["events"]))
            except Exception as exc:
                rerr.append(repr(exc))

        wt = threading.Thread(target=_writer, name="livewriter")
        wt.start()
        rt = threading.Thread(target=_reader, name="livereader")
        rt.start()
        rt.join(60)
        stop.set()
        wt.join(10)
        T.flush(10)
        _check("T-38 read-during-write: query_events never raises and keeps returning rows mid-batch",
               not rerr and len(seen) == 40 and max(seen or [0]) > 0,
               f"errors={rerr[:1]} calls={len(seen)} max_rows={max(seen or [0])}")

    # ================================================================= GROUP G — query & browser payloads
    def group_g(self, improve, knowledge, memory):
        T = self.T
        p = self.fresh("G.db")
        d, kb, mem, rep = self._stores("browse", improve, knowledge, memory)
        # Redirect the read-only browser at the FIXTURE stores (never the real research data).
        T.KB_PATH = kb.path
        T.KB_AUDIT_PATH = kb.audit_path
        T.MEM_DB_PATH = mem.path
        T.LOOP_REPORTS_PATH = rep
        fa = "سیب را از روی میز بردار"                     # Persian is first-class data here
        with T.run(kind="command", command=fa, scene="FloorPlan1", level="L1", lang="fa") as rid_fa:
            with T.span("execute_tool pickup", kind="TOOL", op="execute_tool", step_ix=0) as sid_fa:
                T.emit("plan.produced", summary=fa, payload={"command": fa, "plan": ["pickup سیب"]})
                self._exercise(kb, mem, rep, improve)
        for i in range(1200):
            T.emit("console.log", summary=f"page {i}", level="DEBUG")
        T.flush(10)
        mem.close()

        big = T.query_events(limit=5000)
        _check("T-39a query_events clamps limit to 1000", len(big["events"]) == 1000, str(len(big["events"])))
        pages, seqs = [], []
        cursor = None
        for _ in range(3):
            r = T.query_events(limit=50, before=cursor)
            pages.append(r)
            seqs += [e["seq"] for e in r["events"]]
            cursor = r["cursor"]["first_seq"]
        _check("T-39b three cursor pages return each event exactly once, in order, with no gaps",
               len(seqs) == 150 and len(set(seqs)) == 150 and seqs == sorted(seqs, reverse=True)
               and seqs[0] - seqs[-1] == 149, f"n={len(seqs)} unique={len(set(seqs))} span={seqs[0]-seqs[-1]}")

        filt = T.query_events(limit=200, type="kb.write")
        lvl = T.query_events(limit=1000, min_level="WARNING")
        run_only = T.query_events(limit=1000, run_id=rid_fa)
        _check("T-39c filters actually filter (type, min_level, run_id)",
               filt["events"] and all(e["type"] == "kb.write" for e in filt["events"])
               and all(e["level"] in ("WARNING", "ERROR") for e in lvl["events"])
               and run_only["events"] and all(e["run_id"] == rid_fa for e in run_only["events"]),
               f"type={len(filt['events'])} warn={len(lvl['events'])} run={len(run_only['events'])}")

        one = filt["events"][0]
        full = T.get_event(one["seq"])
        _check("T-40 list rows omit payload_json; get_event returns it plus the writes it emitted",
               "payload_json" not in one and one.get("has_payload") is True
               and full and full.get("payload_json") and isinstance(full.get("payload"), dict)
               and full.get("writes"),
               f"list_keys={sorted(one)[:6]} writes={len(full.get('writes') or []) if full else 0}")

        tree = T.run_tree(rid_fa)
        by_id = {s["span_id"]: s for s in tree["spans"]}
        visited: list = []

        def _walk(sid, depth):
            visited.append(sid)
            if by_id[sid]["depth"] != depth:
                visited.append("BAD-DEPTH")
            for kid in by_id[sid]["children"]:
                _walk(kid, depth + 1)

        for root in tree["roots"]:
            _walk(root, 0)
        _check("T-41 run_tree returns spans flat with correct children/depth/roots; the tree reaches each once",
               tree["run"]["command"] == fa and visited and "BAD-DEPTH" not in visited
               and len(visited) == len(tree["spans"]) == len(set(visited))
               and tree["timeline"]["t0"] is not None,
               f"spans={len(tree['spans'])} visited={len(visited)} roots={len(tree['roots'])}")

        stt = T.stats()
        direct_type = {r["type"]: r["n"] for r in _sql(p, "SELECT type, COUNT(*) n FROM events GROUP BY type")}
        direct_kind = {r["kind"]: r["n"] for r in _sql(p, "SELECT kind, COUNT(*) n FROM spans GROUP BY kind")}
        direct_orph = _sql(p, "SELECT COUNT(*) n FROM writes WHERE run_id IS NULL")[0]["n"]
        direct_runs = _sql(p, "SELECT COUNT(*) n FROM runs")[0]["n"]
        _check("T-42 stats() aggregates match direct COUNT(*) for every bucket it reports",
               stt["events_by_type"] == direct_type and stt["spans_by_kind"] == direct_kind
               and stt["orphans"]["writes"] == direct_orph and stt["runs"]["total"] == direct_runs,
               f"types_ok={stt['events_by_type'] == direct_type} kinds_ok={stt['spans_by_kind'] == direct_kind}")

        cat = T.db_tables()
        ids = [c["id"] for c in cat]
        real_mem = T.MEM_DB_PATH
        T.MEM_DB_PATH = os.path.join(self.tmp, "no_such_memory.db")
        cat_missing = {c["id"]: c for c in T.db_tables()}
        T.MEM_DB_PATH = real_mem
        ep = cat_missing.get("mem.episodes", {})
        _check("T-43 db_tables lists all 18 catalog ids; a missing memory.db degrades to rows=None + error",
               len(ids) == 18 and len(set(ids)) == 18 and "kb.procedures" in ids and "trace.writes" in ids
               and ep.get("rows") is None and ep.get("error"),
               f"n={len(ids)} missing_mem={{'rows': {ep.get('rows')}, 'error': {str(ep.get('error'))[:40]!r}}}")

        rows = T.db_rows("kb.procedures")
        clamp = T.db_rows("kb.procedures", limit=9999)
        allp = T.db_rows("kb.procedures", limit=500)["rows"]
        pg1 = T.db_rows("kb.procedures", limit=1, offset=0)["rows"]
        pg2 = T.db_rows("kb.procedures", limit=1, offset=1)["rows"]
        qhit = T.db_rows("kb.procedures", q="boil water")
        qmiss = T.db_rows("kb.procedures", q="zzz-no-such-command")
        _check("T-44 db_rows reads the JSON KB, injects _artifact_id, honours limit/offset/q, clamps to 500",
               rows["rows"] and all(r["_artifact_id"].startswith("kb:procedure:") for r in rows["rows"])
               and clamp["limit"] == 500 and rows["total"] == len(allp)
               and (len(allp) < 2 or (pg1 and pg2 and pg1[0]["id"] != pg2[0]["id"]
                                      and [pg1[0]["id"], pg2[0]["id"]] == [r["id"] for r in allp[:2]]))
               and qhit["total"] >= 1 and qmiss["total"] == 0,
               f"total={rows['total']} clamp={clamp['limit']} q={qhit['total']}/{qmiss['total']}")

        before = {f: _digest(f) for f in (kb.path, mem.path, rep) if os.path.exists(f)}
        for _ in range(20):
            T.db_tables()
            for t in ("kb.procedures", "kb.lessons", "kb.facts", "kb.audit", "mem.episodes", "mem.spatial",
                      "mem.failures", "mem.nav_cells", "trace.events", "trace.writes", "file.loop_reports"):
                T.db_rows(t, limit=20)
            T.db_row("kb.procedures", str((rows["rows"] or [{}])[0].get("id")))
            T.db_row("mem.episodes", "1")
            T.stats()
            T.search("boil")
            T.artifact("kb:procedure:1")
            T.query_writes(limit=20)
            T.lineage("art:kb:procedure:1")
        after = {f: _digest(f) for f in before}
        ro = T._ro_sqlite(mem.path)
        ro_err = None
        try:
            ro.execute("INSERT INTO episodes (ts, scene) VALUES (1.0, 'HACK')")
            ro.commit()
        except sqlite3.OperationalError as exc:
            ro_err = str(exc)
        except Exception as exc:
            ro_err = f"UNEXPECTED {exc!r}"
        finally:
            ro.close()
        rc = T._rconn()
        rc_err = None
        try:
            rc.execute("INSERT INTO meta (key, value) VALUES ('hack','1')")
        except sqlite3.OperationalError as exc:
            rc_err = str(exc)
        except Exception as exc:
            rc_err = f"UNEXPECTED {exc!r}"
        _check("T-45 read-only proof: 20 passes over every read API leave the fixtures byte-identical, and a "
               "write through the browser path is REJECTED",
               before == after and ro_err and "readonly" in ro_err.replace(" ", "").lower()
               and rc_err and "readonly" in rc_err.replace(" ", "").lower(),
               f"unchanged={before == after} mem_write={ro_err!r} trace_write={rc_err!r}")

        bad_tbl = T.db_rows("kb.procedures; DROP TABLE events")
        bad_ord = T.db_rows("trace.events", order_by="1; DROP TABLE events", limit=5)
        bad_lin = T.lineage("art:x'; DROP TABLE edges--")
        alive = {r["name"] for r in _sql(p, "SELECT name FROM sqlite_master WHERE type='table'")}
        _check("T-46 injection safety: a bogus table id is an error envelope, a bogus order_by cannot reach "
               "SQL, a quoted node id resolves to nothing - events and edges survive",
               bad_tbl.get("error") and bad_tbl["rows"] == [] and bad_ord["rows"] is not None
               and [n["id"] for n in bad_lin["nodes"]] == ["art:x'; DROP TABLE edges--"]
               and bad_lin["edges"] == [] and {"events", "edges"} <= alive,
               f"tbl_err={str(bad_tbl.get('error'))[:40]!r} ord_rows={len(bad_ord['rows'])} "
               f"lineage_nodes={[n['id'] for n in bad_lin['nodes']]} tables={sorted(alive)}")

        runs_fa = T.query_runs(limit=50, lang="fa")["runs"]
        trace_runs = T.db_rows("trace.runs", limit=50)["rows"]
        ev_fa = [e for e in T.query_events(limit=1000, type="plan.produced")["events"] if e["summary"] == fa]
        raw = _sql(p, "SELECT payload_json FROM events WHERE type='plan.produced'")
        _check("T-47 Persian round-trips byte-identically through runs, events and db_rows (no \\uXXXX)",
               any(r["command"] == fa for r in runs_fa)
               and any(r.get("command") == fa for r in trace_runs)
               and len(ev_fa) == 1 and raw and fa in (raw[0]["payload_json"] or "")
               and "\\u" not in (raw[0]["payload_json"] or ""),
               f"runs={len(runs_fa)} events={len(ev_fa)}")

    # ================================================================= GROUP H — end-to-end integration
    def group_h(self, improve, knowledge, memory, feedback):
        T = self.T
        p = self.fresh("H.db")
        d, kb, mem, rep = self._stores("e2e", improve, knowledge, memory)
        T.KB_PATH, T.KB_AUDIT_PATH, T.MEM_DB_PATH, T.LOOP_REPORTS_PATH = kb.path, kb.audit_path, mem.path, rep
        fb = feedback.EnvFeedback(mem, enabled=True)
        scene, cmd = "FloorPlan1", "put the apple in the fridge"
        steps = [{"action": "go_to", "object": "Apple"}, {"action": "pick_and_place", "object": "Apple",
                                                          "to": "Fridge"}]
        results = [{"ok": True, "error": None}, {"ok": True, "error": None}]

        def _one_run(tag):
            """command -> plan -> two steps -> loop closure, exactly as the webapp drives it."""
            with T.run(kind="command", command=cmd, scene=scene, level="L2", lang="en",
                       agent_id="gemma3:4b", config={"model": "gemma3:4b", "rag_enabled": False,
                                                     "loop_enabled": True, "arm": "plan_fresh"}) as rid:
                T.emit("command.received", summary=cmd, payload={"command": cmd, "tag": tag})
                eps, tool_ids = [], []
                for ix, st in enumerate(steps):
                    with T.span(f"execute_tool {st['action']}", kind="TOOL", op="execute_tool",
                                step_ix=ix, scene=scene) as sid:
                        tool_ids.append(sid)
                        T.record_action("MoveAhead", kwargs={"moveMagnitude": 0.25}, success=True)
                        r = fb.log_step(scene, cmd, "L2", st["action"], {"objectType": st["object"]},
                                        results[ix]["ok"], results[ix]["error"], None, None,
                                        run_id=rid, step_ix=ix, span_id=sid)
                        eps.append(r.get("episode_id"))
                rep_out = improve.close_loop(kb, mem, scene=scene, command=cmd, level="L2", arm="plan_fresh",
                                             executed_steps=steps, step_results=results,
                                             goal={"applicable": True, "ok": True, "target": "Apple",
                                                   "held": None},
                                             rag_enabled=False, loop_enabled=True, report_path=rep,
                                             run_id=rid, episode_ids=eps)
                return rid, tool_ids, eps, rep_out

        rid1, tools1, eps1, rep1 = _one_run("first")
        T.flush(10)
        eff = T.span_effects(tools1[0], include_descendants=True)
        _check("T-48a a plan step's descendants carry the episode write it caused",
               any(w["store"] == "mem" and w["table_or_kind"] == "episodes"
                   and w["pointer"] == str(eps1[0]) and w["run_id"] == rid1 for w in eff),
               f"effects={[(w['store'], w['table_or_kind'], w['pointer']) for w in eff]}")

        pid = rep1["writes"][0]["id"] if rep1["writes"] else None
        art_id = f"kb:procedure:{pid}"
        c1 = T.artifact_causes(art_id)
        _check("T-48b the KB entry the loop wrote is an INSERT caused by the close_loop span of THIS run",
               rep1["writes"] and rep1["writes"][0]["action"] == "insert" and len(c1) == 1
               and c1[0]["verdict"] == "insert" and c1[0]["run_id"] == rid1
               and c1[0]["span_name"] == "close_loop" and c1[0]["run_command"] == cmd,
               f"writes={rep1['writes']} causes={[(c['verdict'], c['span_name']) for c in c1]}")

        rid2, tools2, eps2, rep2 = _one_run("second")
        T.flush(10)
        c2 = T.artifact_causes(art_id)
        arts = _sql(p, "SELECT * FROM artifacts WHERE artifact_id=?", (art_id,))
        _check("T-48c the SAME procedure re-learned by a DIFFERENT run: ONE artifact, TWO causes, "
               "insert then reinforce, revision=2",
               rep2["writes"] and rep2["writes"][0]["action"] == "reinforce"
               and rep2["writes"][0]["id"] == pid and len(arts) == 1 and arts[0]["revision"] == 2
               and len(c2) == 2 and [c["verdict"] for c in c2] == ["insert", "reinforce"]
               and [c["run_id"] for c in c2] == [rid1, rid2] and rid1 != rid2,
               f"action={rep2['writes'][0]['action'] if rep2['writes'] else '?'} causes={len(c2)} "
               f"rev={arts[0]['revision'] if arts else '?'} runs={[c['run_id'][:8] for c in c2]}")

        # The reverse direction, from the action side: what did the loop span write?
        loop_sid = rep1.get("trace_span_id")
        back = T.span_effects(loop_sid)
        _check("T-48d the reverse query is exact: span_effects(close_loop span) returns exactly its writes",
               loop_sid and back and {w["artifact_id"] for w in back} ==
               {art_id, f"file:report:{T._safe_ref(os.path.basename(rep) + '#1')}"},
               f"span={str(loop_sid)[:8]} wrote={[w['artifact_id'] for w in back]}")

        up = T.lineage("art:" + art_id, direction="up", depth=8)
        upset = {n["id"] for n in up["nodes"]}
        down = T.lineage(f"run:{rid1}", direction="down", depth=8)
        downset = {n["id"] for n in down["nodes"]}
        _check("T-49 lineage(up) from the entry reaches its close_loop span, its run and its agent; "
               "lineage(down) from the run reaches the entry",
               f"span:{loop_sid}" in upset and f"run:{rid1}" in upset
               and "agent:env:ai2thor" in upset
               and any(n.startswith("art:mem:episode:") for n in upset)
               and ("art:" + art_id) in downset,
               f"up={sorted(upset)[:6]} down_has_entry={('art:' + art_id) in downset}")

        with open(rep, "r", encoding="utf-8") as fh:
            journal = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
        orig12 = {"ts", "scene", "command", "level", "arm", "retrieved", "executed_steps", "outcome",
                  "writes", "related_ids", "reindexed", "loop_enabled"}
        ref = T._safe_ref(os.path.basename(rep) + "#1")
        line_art = _sql(p, "SELECT * FROM artifacts WHERE artifact_id=?", (f"file:report:{ref}",))
        _check("T-50 the loop report now carries run_id, is registered as a file:report artifact, and still "
               "has all 12 original keys",
               len(journal) == 2 and journal[0].get("run_id") == rid1 and journal[1].get("run_id") == rid2
               and all(orig12 <= set(j) for j in journal) and len(line_art) == 1,
               f"lines={len(journal)} run_id={str(journal[0].get('run_id'))[:8] if journal else '?'} "
               f"artifact={len(line_art)}")

        err = "PickupObject failed: Mug|+01.2|-00.3 not visible"
        bad = [{"ok": True, "error": None}, {"ok": False, "error": err}]
        fcmd = "pick up the mug"
        fsteps = [{"action": "go_to", "object": "Mug"}, {"action": "pickup", "object": "Mug"}]
        lids = []
        for i in range(2):
            with T.run(kind="command", command=fcmd, scene=scene, level="L1") as rf:
                with T.span("execute_tool pickup", kind="TOOL", op="execute_tool", step_ix=1,
                            scene=scene) as sf:
                    r = fb.log_step(scene, fcmd, "L1", "pickup", {"objectType": "Mug"}, False, err,
                                    None, None, run_id=rf, step_ix=1, span_id=sf)
                rl = improve.close_loop(kb, mem, scene=scene, command=fcmd, level="L1", arm="plan_fresh",
                                        executed_steps=fsteps, step_results=bad,
                                        goal={"applicable": False}, rag_enabled=False, loop_enabled=True,
                                        report_path=rep, run_id=rf, episode_ids=[r.get("episode_id")])
                lids.append((rf, rl, r.get("episode_id"), sf))
        T.flush(10)
        lesson_id = lids[0][1]["writes"][0]["id"]
        lart = f"kb:lesson:{lesson_id}"
        lcauses = T.artifact_causes(lart)
        want_key = f"{knowledge.normalize_command(fcmd)}|{knowledge._error_class(err)}"
        keys = {c["dedup_key"] for c in lcauses if c["dedup_key"]}
        verds = [c["verdict"] for c in lcauses if c["verdict"] != "update"]
        sig = f"pickup|{mem._error_class(err)}|Mug"
        fail_edges = _sql(p, "SELECT * FROM edges WHERE src=? AND rel='wasDerivedFrom'",
                          (f"art:mem:failure:{T._safe_ref(sig)}",))
        lesson_edges = _sql(p, "SELECT * FROM edges WHERE src=? AND rel='wasDerivedFrom'", (f"art:{lart}",))
        _check("T-51 a failure writes a lesson keyed on (normalized command | error class), reinforced on the "
               "second identical failure",
               lids[0][1]["writes"][0]["action"] == "insert"
               and lids[1][1]["writes"][0]["action"] == "reinforce"
               and lids[1][1]["writes"][0]["id"] == lesson_id
               and verds == ["insert", "reinforce"] and keys == {want_key},
               f"actions={[x[1]['writes'][0]['action'] for x in lids]} verdicts={verds} keys={keys} "
               f"want={want_key!r}")
        _check("T-51b the lesson and the failure counter both link back to the failing ACTION's episode",
               len(fail_edges) >= 1 and fail_edges[0]["dst"] == f"art:mem:episode:{lids[0][2]}"
               and any(e["dst"] == f"art:mem:episode:{lids[0][2]}" for e in lesson_edges),
               f"failure->{[e['dst'] for e in fail_edges]} lesson->{[e['dst'] for e in lesson_edges]}")

        now = time.time()
        old = now - 90 * 24 * 3600.0
        mem.remember_location(scene, "Spatula", receptacle="Drawer", track="oracle", ts=old)
        mem.remember_location(scene, "Spatula", receptacle="Sink", track="oracle", ts=now)
        with T.run(kind="maint", command="prune") as rp:
            with T.span("prune spatial", kind="MAINT", op="maintenance", actor="system:memory"):
                pruned = mem.prune(scene=scene, min_confidence=0.05, now=now)
        T.flush(10)
        dels = T.query_writes(limit=100, store="mem", table_or_kind="spatial", verdict="delete")["writes"]
        payloads = [json.loads(w["before_json"] or "{}") for w in dels]
        _check("T-52 prune records one delete write PER removed id, each carrying the raw AND decayed "
               "confidence (information that exists nowhere else on disk)",
               pruned["spatial_removed"] >= 1 and len(dels) == pruned["spatial_removed"]
               and all("confidence" in b and "decayed" in b for b in payloads)
               and all(w["run_id"] == rp for w in dels)
               and all(b["decayed"] < 0.05 <= b["confidence"] for b in payloads),
               f"removed={pruned} writes={len(dels)} before={payloads[:1]}")
        mem.close()

    # ================================================================= GROUP I — existing gates (opt-in)
    def group_regress(self):
        T = self.T
        T.close()
        gates = ["_phase7_selftest.py", "_phase6_selftest.py", "_phase6a_selftest.py", "_eval_selftest.py"]
        for flag in ("1", "0"):
            bad = []
            for g in gates:
                env = dict(os.environ, TRACE_ENABLED=flag,
                           TRACE_DB=os.path.join(self.tmp, f"regress_{flag}.db"))
                try:
                    r = subprocess.run([sys.executable, os.path.join(self.here, g)], env=env,
                                       capture_output=True, text=True, timeout=900)
                    if r.returncode != 0:
                        bad.append(f"{g} rc={r.returncode}")
                except Exception as exc:
                    bad.append(f"{g} {exc!r}")
            _check(f"T-53 the four existing self-tests still pass with TRACE_ENABLED={flag}",
                   not bad, "; ".join(bad))


    # ============================================================= GROUP I — the remediated write paths
    # Every check here corresponds to a defect found by review: a fix without a test is a fix that regresses.
    def group_i(self, improve, knowledge, memory, feedback):
        T = self.T

        # ---- T-54 the ledger is not droppable ----------------------------------------------------------
        p = self.fresh("I1.db", async_writes=True)
        with T._lock:                       # freeze the writer out of the ring, then saturate it
            qmax_before = T._CFG["queue_max"]
            T._CFG["queue_max"] = 40
            T._writer_stop = True
        for i in range(200):                # 200 DEBUG events into a 40-slot ring
            T.emit("console.log", summary=f"filler {i}", level="DEBUG")
        art = T.record_write(store="kb", table_or_kind="fact", pointer=99, verdict="insert",
                             why="written while the queue was saturated", source="human")
        T.link("art:" + art, "agent:human:arash", "wasAttributedTo", note="under pressure")
        ops = [o for o, _params, _dbg in list(T._queue)]
        with T._lock:
            T._CFG["queue_max"] = qmax_before
            T._writer_stop = False
        T._ensure_writer()
        _t0 = time.time()
        flushed = T.flush(5)
        flush_s = time.time() - _t0
        wrote = _sql(p, "SELECT COUNT(*) n FROM writes WHERE pointer='99'")[0]["n"]
        edged = _sql(p, "SELECT COUNT(*) n FROM edges WHERE src=?", ("art:" + art,))[0]["n"]
        _check("T-54 queue saturation drops DEBUG events but NEVER a writes/artifact/edge row: the "
               "auditability evidence itself is not droppable",
               T._dropped > 0 and "W" in ops and "A" in ops and "G" in ops and wrote == 1 and edged >= 1,
               f"dropped={T._dropped} ops={sorted(set(ops))} writes={wrote} edges={edged}")
        _check("T-54b a drop is ACCOUNTED for, so flush() still completes instead of waiting for a commit "
               "that can never arrive",
               flushed is True and flush_s < 2.0, f"flushed={flushed} in {flush_s:.2f}s")

        # ---- T-55 / T-56 ambient context across the webapp's thread topology --------------------------
        p = self.fresh("I2.db")
        cmd_run = T.start_run(kind="command", command="A: put the apple in the fridge", scene="FloorPlan1")
        fb_holder = {}

        def _http_thread():                 # a rating/teach arrives on an HTTP handler thread mid-command
            fb_holder["run"] = T.start_run(kind="feedback", command="B: human teaching")
            T.end_run(fb_holder["run"], status="OK")

        t = threading.Thread(target=_http_thread, name="http")
        t.start()
        t.join(10)
        worker_ctx = {}

        def _worker_thread():               # the AI2-THOR worker owns no run of its own
            worker_ctx["run"] = (T.current() or {}).get("run_id")
            T.record_write(store="mem", table_or_kind="episodes", pointer=7, verdict="insert",
                           why="step executed", source="environment")

        w = threading.Thread(target=_worker_thread, name="worker")
        w.start()
        w.join(10)
        T.end_run(cmd_run, status="OK")
        T.flush(5)
        owner = _sql(p, "SELECT run_id FROM writes WHERE pointer='7'")[0]["run_id"]
        _check("T-55 a feedback run opened on another thread does NOT steal the process default from the "
               "open command run: the worker's episode is filed under the command",
               worker_ctx.get("run") == cmd_run and owner == cmd_run,
               f"worker saw {str(worker_ctx.get('run'))[:8]}, write says {str(owner)[:8]}, "
               f"command={cmd_run[:8]}, feedback={str(fb_holder.get('run'))[:8]}")

        z = T.start_run(kind="command", command="Z")
        a2 = T.start_run(kind="command", command="A2")
        T.end_run(z, status="OK")           # Z finishes FIRST; A2 is still open
        T.end_run(a2, status="OK")
        after = {}

        def _later():
            after["run"] = (T.current() or {}).get("run_id")

        lt = threading.Thread(target=_later, name="later")
        lt.start()
        lt.join(10)
        _check("T-56 end_run never reinstates an ALREADY FINISHED run as the process default (an orphan is "
               "correct; a wrong causal graph is not)",
               after.get("run") in (None, cmd_run) and after.get("run") != z,
               f"resolved={str(after.get('run'))[:8]} z={z[:8]} a2={a2[:8]}")

        # ---- T-57 writes.step_ix rides the ambient span ------------------------------------------------
        p = self.fresh("I3.db")
        d = self.sub("I3")
        mem = memory.Memory(path=os.path.join(d, "memory.db"))
        T.MEM_DB_PATH = mem.path
        fb = feedback.EnvFeedback(mem, enabled=True)
        rid = T.start_run(kind="command", command="pick up the apple", scene="FloorPlan1")
        for ix in (0, 1):
            fb.log_step("FloorPlan1", command="pick up the apple", level="L1", action="pickup",
                        args={"object": "Apple"}, success=True, error=None, before=None, after=None,
                        run_id=rid, step_ix=ix)
        T.end_run(rid, status="OK")
        T.flush(5)
        rows = _sql(p, "SELECT step_ix, pointer FROM writes WHERE table_or_kind='episodes' ORDER BY id")
        _check("T-57 writes.step_ix is populated from the enclosing plan-step span, so two back-to-back "
               "executions of the same command are separable in `episodes`",
               len(rows) == 2 and [r["step_ix"] for r in rows] == [0, 1],
               f"rows={rows}")

        # ---- T-58 two stores do not fuse into one artifact ---------------------------------------------
        p = self.fresh("I4.db")
        d = self.sub("I4")
        kb_a = knowledge.KnowledgeBase(path=os.path.join(d, "kb_a.json"))
        kb_b = knowledge.KnowledgeBase(path=os.path.join(d, "kb_b.json"))
        T.KB_PATH, T.KB_AUDIT_PATH = kb_a.path, kb_a.audit_path
        ea = kb_a.add_fact("apples live in the fridge", why="A", source="human")
        eb = kb_b.add_fact("a DIFFERENT store's entry number one", why="B", source="human")
        T.flush(5)
        ids = [r["artifact_id"] for r in _sql(p, "SELECT artifact_id FROM writes ORDER BY id")]
        arts = _sql(p, "SELECT artifact_id, n_causes FROM artifacts ORDER BY artifact_id")
        det = T.artifact([r["artifact_id"] for r in arts if "@" in r["artifact_id"]][0]) or {}
        _check("T-58 entry #1 of two different knowledge bases stays TWO artifacts (ids restart at 1 after "
               "every reset, so a shared id would fuse unrelated histories)",
               ea.get("id") == eb.get("id") == 1 and len(set(ids)) == 2 and len(arts) == 2
               and all(a["n_causes"] == 1 for a in arts) and det.get("foreign_store") is True
               and det.get("live") is None,
               f"ids={sorted(set(ids))} n_causes={[a['n_causes'] for a in arts]} "
               f"foreign={det.get('foreign_store')}")

        # ---- T-59 read-only endpoints do not create the store ------------------------------------------
        T.close()
        ghost = os.path.join(self.tmp, "never_created.db")
        T.configure(db_path=ghost)          # configure DOES open (it is the write-side entry point)
        T.close()
        os.remove(ghost)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(ghost + suffix):
                os.remove(ghost + suffix)
        st = T.status()
        stt = T.stats()
        _check("T-59 status()/stats() are read-only: they report a missing store instead of creating it, "
               "applying the schema and spawning a writer thread",
               not os.path.exists(ghost) and st.get("pending") is True and st.get("events") == 0
               and isinstance(stt, dict) and stt.get("error"),
               f"exists={os.path.exists(ghost)} pending={st.get('pending')} reason={str(st.get('reason'))[:40]}")

        # ---- T-60 an unknown order_by is reported, not silently swapped --------------------------------
        p = self.fresh("I5.db")
        T.KB_PATH = kb_a.path
        bad_kb = T.db_rows("kb.facts", order_by="1; DROP TABLE events")
        bad_tr = T.db_rows("trace.events", order_by="no_such_column")
        good = T.db_rows("trace.events", order_by="seq")
        alive = {r["name"] for r in _sql(p, "SELECT name FROM sqlite_master WHERE type='table'")}
        _check("T-60 db_rows REPORTS an unknown order_by column instead of silently falling back (and still "
               "cannot reach SQL: events and edges survive)",
               bad_kb.get("error") and bad_tr.get("error") and not good.get("error")
               and {"events", "edges"} <= alive,
               f"kb={str(bad_kb.get('error'))[:40]} trace={str(bad_tr.get('error'))[:40]}")

        # ---- T-61 the loop-report pointer is derived from the append itself ----------------------------
        d = self.sub("I6")
        jr = os.path.join(d, "reports.jsonl")
        with open(jr, "w", encoding="utf-8") as fh:
            fh.write('{"a": 1}\n\n{"a": 2}\n')          # note the BLANK line: the viewer numbers it too
        n1 = improve._append_report_line(jr, json.dumps({"a": 3}) + "\n")
        outs = []
        thr = [threading.Thread(target=lambda i=i: outs.append(
            improve._append_report_line(jr, json.dumps({"a": 4, "i": i}) + "\n"))) for i in range(8)]
        for t2 in thr:
            t2.start()
        for t2 in thr:
            t2.join(10)
        lines = T._read_lines(jr)
        by_no = {no: s for no, s in lines}
        _check("T-61 every loop report gets its OWN line number, blank lines included, and 8 concurrent "
               "appends produce 8 DISTINCT pointers that resolve to the right lines",
               n1 == 4 and json.loads(by_no[4])["a"] == 3 and len(set(outs)) == 8 and None not in outs
               and all(json.loads(by_no[no])["i"] == i for i, no in enumerate(outs)),
               f"n1={n1} concurrent={sorted(outs)} lines={len(lines)}")

        # ---- T-62 install_thor_hooks actually records primitives and scene resets ----------------------
        p = self.fresh("I7.db")

        class _StubAgent:
            scene = "FloorPlan1"
            success = True
            error = None

            def do(self, action, **kw):
                self.last = (action, kw)
                return {"ran": action}

            def reset(self, scene=None):
                return f"reset {scene}"

        mod = types.SimpleNamespace(ThorAgent=_StubAgent)
        installed = T.install_thor_hooks(mod)
        ag = _StubAgent()
        rid = T.start_run(kind="command", command="pick up the apple", scene="FloorPlan1")
        with T.span("execute_tool pickup", kind="TOOL", op="execute_tool", step_ix=0):
            ret = ag.do("PickupObject", objectId="Apple|+01.0")
            for _ in range(5):
                ag.do("MoveAhead", moveMagnitude=0.25)
            ag.do("Pass")                                  # filtered: a 40 Hz no-op must cost nothing
        was_reset = ag.reset("FloorPlan3")
        T.end_run(rid, status="OK")
        T.flush(5)
        acts = _sql(p, "SELECT COUNT(*) n FROM events WHERE type='action.executed'")[0]["n"]
        spans = _sql(p, "SELECT COUNT(*) n FROM spans WHERE kind='ACTION'")[0]["n"]
        resets = _sql(p, "SELECT COUNT(*) n FROM events WHERE type='scene.reset'")[0]["n"]
        _check("T-62 install_thor_hooks records every AI2-THOR primitive as an ACTION span under the plan "
               "step, collapses the nav repeat, filters Pass, and never changes what do() returns",
               "ThorAgent.do" in (installed.get("installed") or []) and ret == {"ran": "PickupObject"}
               and was_reset == "reset FloorPlan3" and acts == 2 and spans == 2 and resets == 1,
               f"installed={installed} action.executed={acts} ACTION spans={spans} resets={resets}")

        self.group_j_journal()
        self.group_i_webapp(improve, knowledge, memory, feedback)

    def group_j_journal(self):
        """The store must not run WAL on a filesystem that cannot host it.

        `sim/trace.db` lives beside the code, which under WSL means `/mnt/d` -- a Windows drive over DrvFs, with
        no working shared-memory file. WAL there does not fail cleanly: it writes for a while, then every read
        and write starts returning `unable to open database file`, and the store is left corrupt. It happened
        twice on 2026-08-22, the second time within minutes of a clean rebuild.
        """
        T = self.T
        _check("T-73 a real Linux path may use WAL", T._wal_is_safe("/home/arash/thor/trace.db") is True)
        _check("T-74 a Windows path may use WAL", T._wal_is_safe(r"D:\x\trace.db") is True)
        is_wsl = False
        try:
            with open("/proc/version", "r", encoding="utf-8", errors="replace") as fh:
                is_wsl = "microsoft" in fh.read().lower()
        except Exception:
            is_wsl = False
        if os.name == "posix" and is_wsl:
            _check("T-75 a WSL-mounted Windows drive may NOT use WAL",
                   T._wal_is_safe("/mnt/d/Personal/x/trace.db") is False)
        else:
            _check("T-75 /mnt is only suspect under WSL (not WSL here, so WAL stays allowed)",
                   T._wal_is_safe("/mnt/d/Personal/x/trace.db") is True)
        p = self.fresh("J1.db")
        T.emit("t", summary="open the store")
        T.flush(5)
        mode = (T.status() or {}).get("journal_mode")
        _check("T-76 the chosen journal mode is reported, so a regression is visible in /trace_stats",
               mode in ("WAL", "TRUNCATE", "DELETE", "MEMORY"), f"journal_mode={mode!r}")
        actual = str(list(_sql(p, "PRAGMA journal_mode")[0].values())[0]).upper()
        _check("T-77 and it is the mode the file actually has", actual == (mode or "").upper(),
               f"reported={mode!r} actual={actual!r}")

        # A reader cached on ANOTHER thread must not survive the store being recycled. `/reset?scope=all`
        # deletes trace.db and the writer reopens a fresh one under the SAME name -- and `_rconn` cached a
        # connection PER THREAD keyed only on the path, so every HTTP handler thread kept its handle on the
        # removed inode. Observed live: the file held 625 events while /trace_stats reported 0.
        #
        # The test recycles the store WITHOUT deleting it, because Windows refuses to unlink a file with open
        # handles -- which is precisely why this only ever bit on Linux, where the unlink succeeds and the
        # stale handles go on reading a file that no longer has a name. What is asserted is the mechanism:
        # after a close/reopen the generation moves and a thread's cached connection is REBUILT, not reused.
        import threading as _th2
        gate1, gate2 = _th2.Event(), _th2.Event()
        seen = {}

        def reader():
            c1 = T._rconn()
            seen["first"] = c1
            gate1.set()
            gate2.wait(5)
            c2 = T._rconn()
            seen["second"] = c2
            seen["rebuilt"] = (c2 is not c1)
            try:
                seen["reads"] = None if c2 is None else list(c2.execute("SELECT COUNT(*) FROM events"))[0][0]
            except Exception as exc:
                seen["reads"] = f"ERR {exc}"

        t = _th2.Thread(target=reader, daemon=True)
        t.start()
        _check("T-78 a reader thread opens the store", gate1.wait(5) and seen.get("first") is not None,
               str(seen.get("first")))
        before_gen = T._CFG.get("generation")
        T.close()
        T.emit("t", summary="after the recycle")        # reopens
        T.flush(5)
        gate2.set()
        t.join(5)
        _check("T-79 recycling the store moves the generation",
               T._CFG.get("generation") != before_gen, f"{before_gen} -> {T._CFG.get('generation')}")
        _check("T-80 so a thread's cached read connection is rebuilt, not reused",
               seen.get("rebuilt") is True, str({k: v for k, v in seen.items() if k != "first"}))
        _check("T-81 and it can still read", isinstance(seen.get("reads"), int), str(seen.get("reads")))

    def group_i_webapp(self, improve, knowledge, memory, feedback):
        """The end-to-end exhibit: ONE command through the real webapp plan path, with the simulator, the LLM
        and the retriever stubbed. This is what the review found missing entirely — webapp.py had no
        write-side instrumentation at all, so every recorded write was an orphan and 14 of the 20 event types
        were never emitted by any code path."""
        T = self.T
        p = self.fresh("I8.db")
        d = self.sub("I8")
        try:
            import webapp as W
        except Exception as exc:
            _check("T-63 webapp imports and is instrumented", False, repr(exc))
            return

        kb = knowledge.KnowledgeBase(path=os.path.join(d, "kb.json"))
        mem = memory.Memory(path=os.path.join(d, "memory.db"))
        T.KB_PATH, T.KB_AUDIT_PATH, T.MEM_DB_PATH = kb.path, kb.audit_path, mem.path
        T.LOOP_REPORTS_PATH = improve.LOOP_REPORTS_PATH = os.path.join(d, "reports.jsonl")
        seed = kb.add_procedure("pick up the apple", [{"action": "pickup", "object": "Apple"}],
                                why="seed", source="human")

        class _Ag:                                  # the simulator, reduced to what one plan step touches
            scene = "FloorPlan1"
            success = True
            error = None

            def snapshot(self):
                return {"held": "Apple"}

            def spatial_snapshot(self):
                return [{"object_type": "Apple", "object_id": "Apple|1", "receptacle": "CounterTop"}]

            def resolve(self, t):
                return {"objectId": "Apple|+01.0|+00.9|-00.6", "objectType": "Apple"}

            def object_by_id(self, oid):
                return {"objectId": oid, "objectType": "Apple", "visible": True, "isPickedUp": False}

            def pickup(self, oid, force=False):
                self.success = True
                return True

            def held_object(self):
                return {"objectType": "Apple"}

            # `_approach_then` now ALWAYS approaches and then edges in (AI2-THOR's `visible` means "in frame and
            # within 1.5 m", not "near"), so both of these are on the path of every manipulation step. They are
            # generators: the catch-all below returns None, which `yield from` cannot consume.
            def walk_to(self, oid, **kw):
                return iter(())

            def close_in(self, oid, **kw):
                return iter(())

            def objects(self):
                return [{"objectType": "Apple", "objectId": "Apple|1"}]

            def __getattr__(self, name):
                """Everything else the plan repairer may reach for (walk_to, move_ahead, put_in, ...) succeeds
                silently — this fixture is about the instrumentation, not about AI2-THOR's semantics."""
                if name.startswith("_"):
                    raise AttributeError(name)

                def _ok(*a, **kw):
                    object.__setattr__(self, "success", True)
                    return None
                return _ok

        agent = _Ag()
        stop = threading.Event()

        def _fake_worker():                         # the single AI2-THOR thread, minus AI2-THOR
            while not stop.is_set():
                try:
                    job = W._jobs.get(timeout=0.05)
                except Exception:
                    continue
                try:
                    r = job(agent)
                    if hasattr(r, "__next__"):
                        for _ in r:
                            pass
                except Exception:
                    pass

        saved = (W._kb, W._memory, W._feedback, W.RAG_ENABLED, W._retrieve_memory, W._retrieve_memory_hits,
                 W._active_exploration, W.al.make_provider)
        wt = threading.Thread(target=_fake_worker, name="fakeworker", daemon=True)
        wt.start()
        try:
            W._kb, W._memory = kb, mem
            W._feedback = feedback.EnvFeedback(mem, enabled=True)
            W.RAG_ENABLED = False                   # keeps close_loop's reindex (a network call) out of a test
            W._retrieve_memory = lambda text: "known: Apple is on the CounterTop"
            W._retrieve_memory_hits = lambda text: [{"id": seed.get("id"), "kind": "procedures"}]
            W._active_exploration = lambda text, force=False: None
            reply = json.dumps({"say": "picking up the apple", "level": "L1",
                                "plan": [{"action": "pickup", "object": "Apple"}]})
            W.al.make_provider = lambda *a, **kw: types.SimpleNamespace(chat=lambda model, msgs: reply)
            with W._lock:
                W._state["scene"] = "FloorPlan1"
            W._plan_and_run("pick up the apple", "gemma3:4b", "http://127.0.0.1:11434")
        finally:
            stop.set()
            wt.join(5)
            (W._kb, W._memory, W._feedback, W.RAG_ENABLED, W._retrieve_memory, W._retrieve_memory_hits,
             W._active_exploration, W.al.make_provider) = saved
        T.flush(10)

        types_seen = {r["type"] for r in _sql(p, "SELECT DISTINCT type FROM events")}
        want = {"run.started", "command.received", "retrieval.performed", "llm.call", "plan.produced",
                "step.started", "step.finished", "memory.write", "kb.write", "loop.closed", "console.log",
                "run.ended"}
        runs = _sql(p, "SELECT * FROM runs")
        t_run = runs[0]["started_ts"] if runs else 0
        # The ONLY unattributed write must be the fixture's own seed procedure, written before the run opened.
        orphan_w = _sql(p, "SELECT COUNT(*) n FROM writes WHERE run_id IS NULL AND ts >= ?", (t_run,))[0]["n"]
        pre_w = _sql(p, "SELECT COUNT(*) n FROM writes WHERE run_id IS NULL AND ts < ?", (t_run,))[0]["n"]
        total_w = _sql(p, "SELECT COUNT(*) n FROM writes")[0]["n"]
        _check("T-63 one webapp command produces ONE run carrying its command/scene/level/lang/arm, and every "
               "event type the console can emit — the 14 that no code path emitted before",
               want <= types_seen and len(runs) == 1 and runs[0]["command"] == "pick up the apple"
               and runs[0]["level"] == "L1" and runs[0]["lang"] == "en"
               and runs[0]["scene"] == "FloorPlan1" and runs[0]["status"] == "OK" and runs[0]["arm"],
               f"missing={sorted(want - types_seen)} runs={len(runs)} "
               f"row={ {k: runs[0][k] for k in ('command', 'level', 'lang', 'arm', 'status')} if runs else None}")
        _check("T-64 NOTHING the command wrote is an orphan any more (the review measured 4079/4079 writes "
               "with run_id IS NULL on the author's own store)",
               total_w > 4 and orphan_w == 0 and pre_w == 1,
               f"writes={total_w} orphans_during_run={orphan_w} before_run={pre_w}")

        rid = runs[0]["run_id"] if runs else ""
        proc = [r for r in _sql(p, "SELECT * FROM writes WHERE table_or_kind='procedure' ORDER BY id")]
        eps = [r["pointer"] for r in _sql(p, "SELECT pointer FROM writes WHERE table_or_kind='episodes'")]
        derived = _sql(p, "SELECT src, dst FROM edges WHERE rel='wasDerivedFrom'")
        kb_to_ep = [e for e in derived if e["src"].startswith("art:kb:") and ":episode:" in e["dst"]]
        causes = T.artifact_causes(proc[-1]["artifact_id"]) if proc else []
        _check("T-65 the thesis exhibit is reproducible end to end: the procedure the loop wrote names the "
               "run that produced it AND carries wasDerivedFrom edges to the episodes of the steps executed",
               proc and proc[-1]["run_id"] == rid and kb_to_ep
               and all(e["dst"].split(":")[-1] in eps for e in kb_to_ep)
               and causes and causes[-1]["run_command"] == "pick up the apple"
               and causes[-1]["span_name"] == "close_loop",
               f"proc_run={str(proc[-1]['run_id'])[:8] if proc else None} run={rid[:8]} "
               f"episodes={eps} kb->episode edges={[(e['src'], e['dst']) for e in kb_to_ep]}")

        steps = _sql(p, "SELECT * FROM spans WHERE op='execute_tool' ORDER BY seq")
        eff = T.span_effects(steps[0]["span_id"], include_descendants=True) if steps else []
        rep = _sql(p, "SELECT * FROM writes WHERE store='file' ORDER BY id")
        _check("T-66 'what did this action write?' answers for a real plan step, and the loop report is "
               "addressable as file:report:<journal>#<line> with its OWN line",
               steps and steps[0]["step_ix"] == 0 and eff
               and any(w["table_or_kind"] == "episodes" for w in eff)
               and rep and rep[-1]["pointer"].endswith("#1") and rep[-1]["run_id"] == rid,
               f"steps={len(steps)} effects={[w['table_or_kind'] for w in eff]} "
               f"report={rep[-1]['pointer'] if rep else None}")


def main() -> int:
    _stub_ai2thor()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    tmp = tempfile.mkdtemp(prefix="trace_selftest_")
    regress = "--regress" in sys.argv
    tr = None
    try:
        try:
            import trace as tr                      # sim/trace.py — shadows the stdlib coverage module
            if getattr(tr, "TRACE_API_VERSION", 0) < 1:
                raise ImportError("resolved the STDLIB trace module, not sim/trace.py")
            import feedback
            import improve
            import knowledge
            import memory
            # Data safety: the browse paths must NEVER point at the real research stores while testing.
            tr.KB_PATH = os.path.join(tmp, "fixture_kb.json")
            tr.KB_AUDIT_PATH = os.path.join(tmp, "fixture_kb_audit.log")
            tr.MEM_DB_PATH = os.path.join(tmp, "fixture_memory.db")
            tr.LOOP_REPORTS_PATH = os.path.join(tmp, "fixture_loop_reports.jsonl")
            improve.LOOP_REPORTS_PATH = os.path.join(tmp, "fixture_loop_reports.jsonl")

            s = Suite(tmp, here, tr)
            s.group_a()
            s.group_b()
            s.group_c()
            s.group_d(improve, knowledge, memory, feedback)
            s.group_e()
            s.group_f()
            s.group_g(improve, knowledge, memory)
            s.group_h(improve, knowledge, memory, feedback)
            s.group_i(improve, knowledge, memory, feedback)
            if regress:
                s.group_regress()
        except Exception as exc:                    # a crash mid-run still yields a table + non-zero exit
            import traceback as _tb
            _check("unexpected exception", False, repr(exc) + " | " + _tb.format_exc().splitlines()[-3][:90])
    finally:
        try:
            if tr is not None:
                tr.close()
        except Exception:
            pass
        # Read connections opened on worker threads (T-36/T-38) are released only when those thread-locals are
        # collected; on Windows an open handle blocks the delete, so collect and retry rather than leak a dir.
        import gc
        for _ in range(6):
            gc.collect()
            shutil.rmtree(tmp, ignore_errors=True)
            if not os.path.isdir(tmp):
                break
            time.sleep(0.2)

    width = max(len(n) for n, _, _ in _results)
    print("=" * 110)
    print("TRACE SELFTEST  (observability plane: temp stores only, no AI2-THOR, no LLM, no network)")
    print("-" * 110)
    for name, ok, detail in _results:
        tail = f"  | {detail[:110]}" if detail else ""
        print(f" [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}{tail if not ok else ''}")
    passed = sum(1 for _, ok, _ in _results if ok)
    print("-" * 110)
    print(f" {passed}/{len(_results)} checks passed")
    print("=" * 110)
    if passed == len(_results):
        print(f"TRACE SELFTEST OK  ({passed}/{len(_results)})")
        return 0
    for name, ok, detail in _results:
        if not ok:
            print(f"  FAILED: {name}\n          {detail}")
    print(f"TRACE SELFTEST FAILED  ({passed}/{len(_results)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
