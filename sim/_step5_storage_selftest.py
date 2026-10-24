"""Self-test for step 5 / A1-storage -- `sim/ltm_store.py` + the `sim/ltm.py` facade.

Offline, Windows `python` from the REPO ROOT, temp directories only (never sim/trace.db, sim/memory.db or the default
sim/step5/ltm.db), no simulator, no WSL, no Ollama, no network:

    python sim/_step5_storage_selftest.py

Proves (docs/step5-plan.md sections 4, 9, 11): lazy creation + rollback journal (never WAL); versioned migrations with a
pre-migration backup and refusal of a newer schema; restart; atomic evidence + claims + write_log in ONE transaction;
rollback at every fault point; crash between commit and trace export (no loss, exactly one export against a deduping
destination); duplicate / reordered / truncated / conflicting commit streams; privileged-field rejection; snapshot,
verification, restore and frozen-state equality; integrity checks; import hygiene.
"""
import hashlib
import itertools
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                    # append, not insert: sim/trace.py must not shadow the stdlib `trace`

import ltm                                # noqa: E402
import ltm_store as ls                    # noqa: E402
from ltm_store import (ConflictError, FrozenError, MissingRowError, OrderError, SchemaError, Store, StoreError,  # noqa: E402
                       TruncatedCommitError, WhitelistError)

REPO = Path(HERE).parent
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:                # wrong exception type
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


def fresh(tmp, name="a", **kw):
    return Store(Path(tmp) / name / "ltm.db", **kw)


def ep_ops(i):
    """A consolidation-shaped commit body for practice episode i: place + natural location claim + trial + episode row."""
    return [
        {"table": "place_instance", "op": "insert",
         "values": {"place": "pl:1", "kitchen": "FloorPlan1", "frame": "f0", "epoch": 0, "rtype": "Fridge",
                    "cx": 1.0, "cy": 0.5, "cz": 2.0, "extent": {"w": 0.9, "h": 1.8}, "created_E": 1},
         "evidence": {"rule": "R3", "mutation": "create", "summary": {"basis": "box_only"}, "src_ev": ["ab" * 6]}},
        {"table": "location_claim", "op": "upsert",
         "values": {"claim": f"lc:{i}", "kitchen": "FloorPlan1", "epoch": 0, "frame": "f0", "otype": "Apple",
                    "place": "pl:1", "xyz": [1.0, 0.5, 2.0], "verified": "verified", "provenance": "natural",
                    "conf": 0.9, "E_created": "$E", "E_support": "$E", "contradictions": 0, "status": "current"},
         "evidence": {"rule": "R2", "mutation": "create", "summary": {"views": 2, "basis": "inside_inspection"},
                      "src_ev": [hashlib.sha256(str(i).encode()).hexdigest()[:12]], "conf": 0.9}},
        {"table": "search_trial", "op": "insert",
         "values": {"u": f"u:{i}", "kitchen": "FloorPlan1", "epoch": 0, "otype": "Apple", "place": "pl:1",
                    "rtype": "Fridge", "outcome": "positive", "E": "$E"},
         "evidence": {"rule": "R3", "mutation": "create", "summary": {"trial": i}, "src_ev": []}},
        {"table": "episodes", "op": "insert",
         "values": {"episode": f"ep:{i}", "kitchen": "FloorPlan1", "mode": "practice", "world": "reset", "epoch": 0,
                    "level": "L1", "lang": "en", "E": "$E", "exec_ok": True, "goal_ok": True, "invalid": False,
                    "stats": {"prims": 12}}},
    ]


def ep_commit(i):
    return Store.build_commit(f"consol:ep:{i}", "consolidation", ep_ops(i), episode=f"ep:{i}", seq=i, advance_clock=True)


def tables_files(tmp):
    return sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file())


def main():
    sim_dir = Path(HERE)
    guard = {n: (sim_dir / n).stat().st_mtime if (sim_dir / n).exists() else None for n in ("memory.db",)}   # trace.db is written by the concurrent step-4 workflow: guarded by the source check instead
    default_before = ls.DEFAULT_DB.exists()
    tmp = tempfile.mkdtemp(prefix="s5storage_")
    try:
        # ---------------------------------------------------------------- 1. lazy creation, journal, pragmas
        s = fresh(tmp, "lazy")
        check("lazy: constructor creates nothing", not s.path.exists() and not s.path.parent.exists())
        s.ensure()
        check("lazy: ensure() creates file", s.path.exists())
        check("journal mode is delete (never WAL)", s.pragmas["journal_mode"] == "delete", s.pragmas)
        check("synchronous=EXTRA, FK on", s.pragmas["synchronous"] == 3 and s.pragmas["foreign_keys"] == 1)
        check("schema is latest", s.version == ls.SCHEMA_LATEST)
        check("drvfs path policy still delete", ls.journal_policy("/mnt/d/x/ltm.db")["mode"] == "delete"
              and "drvfs" in ls.journal_policy("/mnt/d/x/ltm.db")["reason"])
        check("no -wal/-shm files exist", not any(f.endswith(("-wal", "-shm")) for f in tables_files(tmp)))
        sid = s.store_id()
        check("store_id minted", len(sid) == 20)
        s.close()
        # a stray WAL database is demoted back to DELETE
        wal = Path(tmp) / "wal" / "ltm.db"
        wal.parent.mkdir()
        c = sqlite3.connect(str(wal))
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE x(a)")
        c.commit()
        c.close()
        check("fixture is really WAL", wal.read_bytes()[18] == 2)
        # opening it as a store fails on missing schema? it must at least be demoted first
        s = Store(wal)
        try:
            s.ensure()
        except Exception:
            pass
        s.close()
        check("stray WAL header demoted to rollback journal", wal.read_bytes()[18] == 1)

        # ---------------------------------------------------------------- 2. migrations
        s1 = fresh(tmp, "mig", target_version=1)
        r = s1.commit("m:1", "admin", [{"table": "episodes", "op": "insert", "values": {
            "episode": "ep:0", "kitchen": "FloorPlan1", "mode": "practice", "world": "reset", "epoch": 0, "E": 0}}])
        check("v1 db accepts v1 columns", r["status"] == "committed" and s1.version == 1)
        check("v1 db rejects a v2 column",
              raises(WhitelistError, s1.commit, "m:2", "admin",
                     [{"table": "retrieval_log", "op": "insert", "values": {"retrieval": "r:1", "channel": "lexical"}}]))
        h1 = s1.state_hash()
        s1.close()
        s2 = Store(Path(tmp) / "mig" / "ltm.db")
        s2.ensure()
        check("reopen at latest migrates v1 -> v2", s2.version == 2)
        check("pre-migration backup written", (Path(tmp) / "mig" / "ltm.db.premigrate-v1").exists())
        check("migrated data intact", s2.get("episodes", episode="ep:0") is not None and s2.clock() == 0)
        check("v2 column now writable",
              s2.commit("m:3", "admin", [{"table": "retrieval_log", "op": "insert", "values": {
                  "retrieval": "r:1", "episode": "ep:0", "channel": "lexical", "query": {"a": 1}}}])["status"] == "committed")
        check("schema_version rows 1,2", [r["version"] for r in s2.select("schema_version")] == [1, 2])
        check("migration did not change old rows' hash base", h1 != s2.state_hash())
        s2.close()
        check("downgrade refused", raises(SchemaError, Store(Path(tmp) / "mig" / "ltm.db", target_version=1).ensure))
        c = sqlite3.connect(str(Path(tmp) / "mig" / "ltm.db"))
        c.execute("INSERT INTO schema_version VALUES (99,'future')")
        c.commit()
        c.close()
        check("newer-than-code schema refused", raises(SchemaError, Store(Path(tmp) / "mig" / "ltm.db").ensure))
        check("bad target_version refused", raises(SchemaError, Store, Path(tmp) / "z.db", target_version=7))

        # ---------------------------------------------------------------- 3. one atomic commit, restart
        s = fresh(tmp, "atom")
        r = s.apply(ep_commit(1))
        check("commit ok, E == seq", r["status"] == "committed" and r["E"] == 1 and s.clock() == 1)
        check("one write_log row per op", len(s.select("write_log")) == 4 and len(r["write_log"]) == 4)
        check("one evidence row per claim mutation", s.count("evidence") == 3 and len(r["evidence"]) == 3)
        check("evidence stamped with E and rule", all(e["E"] == 1 and e["rule"] for e in s.select("evidence")))
        check("$E substituted", s.get("location_claim", claim="lc:1")["E_support"] == 1)
        check("json column decoded", s.get("location_claim", claim="lc:1")["xyz"] == [1.0, 0.5, 2.0])
        check("integrity clean", s.check_integrity() == [], s.check_integrity())
        h = s.state_hash()
        s.close()
        s = Store(Path(tmp) / "atom" / "ltm.db")
        check("restart: same state hash", s.state_hash() == h)
        check("restart: clock persisted", s.clock() == 1)
        check("restart: duplicate event no-op", s.apply(ep_commit(1))["status"] == "duplicate" and s.state_hash() == h)
        # atomic rollback: 3rd op fails -> nothing persists
        bad = ep_ops(2)[:2] + [{"table": "episodes", "op": "update", "key": {"episode": "ep:nope"}, "values": {"E": 1}}]
        check("failing op rolls the whole commit back",
              raises(MissingRowError, s.commit, "bad:1", "consolidation", bad, episode="ep:2", seq=2, advance_clock=True)
              and s.state_hash() == h and s.clock() == 1)
        check("insert of existing key with different values is a conflict",
              raises(ConflictError, s.commit, "bad:2", "admin", [{"table": "place_instance", "op": "insert", "values": {
                  "place": "pl:1", "kitchen": "FloorPlan1", "rtype": "Cabinet"}, "evidence": {
                      "rule": "R3", "mutation": "create", "summary": {}}}]) and s.state_hash() == h)
        check("insert identical row is a no-op, not an error",
              s.commit("ok:1", "admin", [{"table": "episodes", "op": "insert", "values": {
                  "episode": "ep:1", "kitchen": "FloorPlan1"}}])["status"] == "committed")
        check("foreign key to a missing place rolls back",
              raises(ConflictError, s.commit, "fk:1", "admin", [{"table": "location_claim", "op": "insert", "values": {
                  "claim": "lc:x", "place": "pl:404"}, "evidence": {"rule": "R2", "mutation": "create", "summary": {}}}]))
        check("no row from the failed fk commit", s.get("location_claim", claim="lc:x") is None)
        check("update changes a column", s.commit("up:1", "admin", [{"table": "location_claim", "op": "update", "key": {
            "claim": "lc:1"}, "values": {"contradictions": 1, "status": "maybe"}, "evidence": {
                "rule": "R6", "mutation": "contradict", "summary": {"views": 2}, "src_ev": []}}])["status"] == "committed"
              and s.get("location_claim", claim="lc:1")["contradictions"] == 1)
        check("evidence history kept per mutation", len(s.select("evidence", where={"claim": "location_claim/lc:1"})) == 2)
        check("integrity still clean", s.check_integrity() == [], s.check_integrity())
        s.close()

        # ---------------------------------------------------------------- 4. fault points: rollback / crash recovery
        for point in ("ops_done", "pre_receipt", "pre_commit"):
            fs = fresh(tmp, f"f-{point}", fault_hook=lambda p, point=point: (_ for _ in ()).throw(RuntimeError("boom")) if p == point else None)
            fs.ensure()
            h0 = fs.state_hash()
            ok = raises(RuntimeError, fs.apply, ep_commit(1))
            check(f"crash at {point}: nothing persisted", ok and fs.state_hash() == h0 and fs.clock() == 0 and fs.count("write_log") == 0)
            fs.fault_hook = None
            check(f"crash at {point}: retry commits cleanly", fs.apply(ep_commit(1))["status"] == "committed")
            fs.close()

        # crash AFTER commit, before the trace export: no loss, exactly one export
        for probe in (True, False):
            tr = ltm.ListTracer(has_probe=probe)
            state = {"armed": True}

            def hook(p, state=state):
                if p == "post_commit" and state["armed"]:
                    state["armed"] = False
                    raise RuntimeError("process died after COMMIT")
            cs = fresh(tmp, f"crash-{probe}", tracer=tr, fault_hook=hook)
            died = raises(RuntimeError, cs.apply, ep_commit(1))
            check(f"post-commit crash (probe={probe}): data survived", died and cs.get("episodes", episode="ep:1") is not None)
            check(f"post-commit crash (probe={probe}): nothing exported yet", len(tr.records) == 0 and len(cs.pending_exports()) == 4)
            cs.close()
            cs = Store(Path(tmp) / f"crash-{probe}" / "ltm.db", tracer=tr)       # restart
            check(f"restart retry is a duplicate commit (probe={probe})", cs.apply(ep_commit(1))["status"] == "duplicate")
            out = cs.export_pending()
            check(f"export after restart sends all 4 (probe={probe})", out["emitted"] == 4 and len(tr.records) == 4, out)
            cs.export_pending()
            check(f"second export is a no-op (probe={probe})", len(tr.records) == 4 and tr.emit_calls == 4 and cs.pending_exports() == [])
            check(f"trace keys namespaced by store id (probe={probe})", all(k.startswith(cs.store_id() + ":") for k in tr.records))
            cs.close()

        # crash after emit but before the ack is stored -> at-least-once; the deduping destination keeps it exactly-once
        for probe in (True, False):
            tr = ltm.ListTracer(has_probe=probe)
            state = {"n": 0}

            def hook2(p, state=state):
                if p == "export_after_emit":
                    state["n"] += 1
                    if state["n"] == 2:
                        raise RuntimeError("died after emit, before ack")
            es = fresh(tmp, f"emit-{probe}", fault_hook=hook2)
            es.apply(ep_commit(1))
            check(f"export crash (probe={probe}) surfaces to explicit caller", raises(RuntimeError, es.export_pending, tr))
            es.fault_hook = None
            es.export_pending(tr)
            check(f"retry exports every row exactly once at the destination (probe={probe})", len(tr.records) == 4)
            check(f"retry left no un-acked rows (probe={probe})", es.pending_exports() == [])
            if not probe:
                check("no-probe destination saw a re-send, but dedupe kept one record", tr.emit_calls == 5 and len(tr.records) == 4)
            else:
                check("probe destination saw no re-send", tr.emit_calls == 4)
            es.close()

        # failing destination never fails the commit
        tr = ltm.ListTracer(fail_on=lambda k: (_ for _ in ()).throw(ConnectionError("destination down")))
        ms = fresh(tmp, "down", tracer=tr)
        check("tracer outage does not fail the commit", ms.apply(ep_commit(1))["status"] == "committed" and ms.mirror_errors)
        tr.fail_on = None
        check("outage recovers on next export", ms.export_pending()["emitted"] == 4 and len(tr.records) == 4)
        # destination refuses a different payload for an existing key
        bad_tr = ltm.ListTracer()
        k0 = ms.trace_key("consol:ep:1", 0)
        bad_tr.records[k0] = {"other": 1}
        ms2 = fresh(tmp, "down2")
        check("destination rejects differing payload under the same key",
              raises(StoreError, bad_tr.emit, k0, {"other": 2}))
        ms.close()
        ms2.close()

        # ---------------------------------------------------------------- 5. duplicate / reordered / truncated streams
        commits = [ep_commit(i) for i in range(1, 7)]
        base = fresh(tmp, "ordered")
        for c in commits:
            base.apply(c)
        want = base.state_hash()
        check("ordered replay integrity", base.check_integrity() == [])
        rng = random.Random(5)
        perms = [list(reversed(commits))] + [rng.sample(commits, len(commits)) for _ in range(9)]
        same = 0
        for n, p in enumerate(perms):
            ps = fresh(tmp, f"perm{n}")
            for c in p:
                ps.apply(c)
            same += ps.state_hash() == want and ps.clock() == 6
            ps.close()
        check("10 permutations of a complete stream give an identical canonical state", same == 10, f"{same}/10")
        dupes = fresh(tmp, "dupes")
        stream = commits + commits[2:5] + list(reversed(commits))
        results = [dupes.apply(c)["status"] for c in stream]
        check("duplicates anywhere are ignored", results.count("committed") == 6 and dupes.state_hash() == want)
        # conflicting duplicate
        alt = ep_commit(3)
        alt["ops"] = ep_ops(3)
        alt["ops"][1]["values"]["conf"] = 0.1
        alt = Store.build_commit(alt["event_id"], "consolidation", alt["ops"], episode="ep:3", seq=3, advance_clock=True)
        check("conflicting duplicate (same event id, new content) fails", raises(ConflictError, dupes.apply, alt)
              and dupes.state_hash() == want)
        # same episode under another event id; same seq for another episode
        other = Store.build_commit("consol:other", "consolidation", ep_ops(3), episode="ep:3", seq=33, advance_clock=True)
        check("episode already consolidated -> conflict", raises(ConflictError, dupes.apply, other))
        clash = Store.build_commit("consol:clash", "consolidation", ep_ops(77), episode="ep:77", seq=3, advance_clock=True)
        check("seq reused by another episode -> conflict", raises(ConflictError, dupes.apply, clash))
        check("conflicts left the state untouched", dupes.state_hash() == want)
        # truncated
        t = ep_commit(8)
        cut = dict(t, ops=t["ops"][:3])
        check("truncated ops (declared n_ops) rejected", raises(TruncatedCommitError, dupes.apply, cut))
        cut2 = dict(t, ops=t["ops"][:3], n_ops=3)
        check("truncated ops (digest) rejected", raises(TruncatedCommitError, dupes.apply, cut2))
        check("a truncated episode wrote nothing", dupes.get("episodes", episode="ep:8") is None and dupes.state_hash() == want)
        check("empty op list rejected", raises(WhitelistError, dupes.commit, "e:0", "admin", []))
        # explicit-clock ordering
        check("expect_clock mismatch -> OrderError", raises(OrderError, dupes.commit, "o:1", "admin", [{
            "table": "episodes", "op": "insert", "values": {"episode": "ep:o"}}], expect_clock=2))
        check("expect_clock match commits", dupes.commit("o:2", "admin", [{
            "table": "episodes", "op": "insert", "values": {"episode": "ep:o"}}], expect_clock=6)["status"] == "committed")
        # clock without seq is +1 and persisted
        ns = fresh(tmp, "noseq")
        a = ns.commit("n:1", "consolidation", [{"table": "episodes", "op": "insert", "values": {"episode": "ep:a", "E": "$E"}}],
                      episode="ep:a", advance_clock=True)
        b = ns.commit("n:2", "consolidation", [{"table": "episodes", "op": "insert", "values": {"episode": "ep:b", "E": "$E"}}],
                      episode="ep:b", advance_clock=True)
        check("auto clock is +1 per advancing commit", (a["E"], b["E"], ns.clock()) == (1, 2, 2))
        check("episode with no claims still advances E", ns.get("episodes", episode="ep:b")["E"] == 2)
        check("non-advancing commit does not move the clock", ns.commit("n:3", "reset_world", [{
            "table": "epochs", "op": "insert", "values": {"kitchen": "FloorPlan1", "epoch": 1, "E": "$E"}}])["E"] == 2
              and ns.clock() == 2)
        check("advancing commit needs an episode", raises(WhitelistError, ns.commit, "n:4", "consolidation", [{
            "table": "episodes", "op": "insert", "values": {"episode": "ep:c"}}], advance_clock=True))

        # ---------------------------------------------------------------- 6. whitelist / privileged fields
        ws = fresh(tmp, "white")
        ws.ensure()
        h0 = ws.state_hash()

        def put(table, values, ev=None, op="insert"):
            o = {"table": table, "op": op, "values": values}
            if ev is not None:
                o["evidence"] = ev
            return ws.commit("w:" + hashlib.md5(repr((table, values)).encode()).hexdigest()[:8], "admin", [o])
        good_ev = {"rule": "R3", "mutation": "create", "summary": {"n": 1}, "src_ev": []}
        check("unknown column rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "objectId": "x"}))
        check("privileged column group rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "group": "g1"}))
        check("metadata-ish column rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "metadata": 1}))
        check("goal predicate column rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "goal": "Apple in Fridge"}))
        check("errors_logged column rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "errors_logged": 3}))
        check("privileged json key objectId rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "stats": {"objectId": "a"}}))
        check("privileged json key group rejected (nested)", raises(WhitelistError, put, "episodes", {"episode": "e1", "stats": {"a": {"b": [{"group": 1}]}}}))
        check("goal_ok inside json rejected", raises(WhitelistError, put, "episodes", {"episode": "e1", "stats": {"goal_ok": True}}))
        check("goal_ok admitted only as a declared column", put("episodes", {"episode": "e1", "goal_ok": True})["status"] == "committed")
        check("simulator id string rejected in a token field", raises(WhitelistError, put, "episodes", {"episode": "Apple|+01.20|+00.90|-01.30"}))
        check("simulator id in a json value rejected", raises(WhitelistError, put, "episodes", {"episode": "e2", "stats": {"x": "Apple|+1.0|"}}))
        check("_physics suffix rejected", raises(WhitelistError, put, "epochs", {"kitchen": "FloorPlan1_physics", "epoch": 1}))
        check("error sentence rejected in json (not an enumerated token)", raises(WhitelistError, put, "episodes", {
            "episode": "e3", "stats": {"msg": "Object not found in scene!"}}))
        check("non-ASCII rejected in token fields", raises(WhitelistError, put, "episodes", {"episode": "e4", "kitchen": "آشپزخانه"}))
        check("homoglyph / full-width id rejected", raises(WhitelistError, put, "episodes", {"episode": "ｅp1"}))
        check("place ids must be agent-minted pl:<n>", raises(WhitelistError, put, "place_instance", {
            "place": "Fridge_1", "kitchen": "FloorPlan1"}, good_ev))
        check("closed enum enforced", raises(WhitelistError, put, "episodes", {"episode": "e5", "mode": "oracle"}))
        check("type check: real must be finite", raises(WhitelistError, put, "place_instance", {
            "place": "pl:9", "cx": float("nan")}, good_ev))
        check("type check: int rejects float", raises(WhitelistError, put, "episodes", {"episode": "e6", "epoch": 1.5}))
        check("claim table requires evidence", raises(WhitelistError, put, "place_instance", {"place": "pl:2", "rtype": "Fridge"}))
        check("evidence mutation enum enforced", raises(WhitelistError, put, "place_instance", {"place": "pl:2"}, dict(good_ev, mutation="invent")))
        check("evidence src_ev must be hashes", raises(WhitelistError, put, "place_instance", {"place": "pl:2"}, dict(good_ev, src_ev=["Apple|1|"])))
        check("evidence free text rejected", raises(WhitelistError, put, "place_instance", {"place": "pl:2"}, dict(good_ev, summary={"why": "x"})))
        check("evidence extra field rejected", raises(WhitelistError, put, "place_instance", {"place": "pl:2"}, dict(good_ev, detail="x")))
        check("internal tables are not writable", all(raises(WhitelistError, put, t, {"a": 1}) for t in ("evidence", "write_log", "commit_receipt", "meta", "export_ack")))
        check("SQL-ish table name rejected", raises(WhitelistError, put, "episodes; DROP TABLE meta", {"episode": "e7"}))
        check("SQL-ish column name rejected", raises(WhitelistError, put, "episodes", {"episode) VALUES (1); --": 1}))
        check("update must name the full primary key", raises(WhitelistError, ws.commit, "w:u1", "admin", [{
            "table": "episodes", "op": "update", "key": {}, "values": {"E": 1}}]))
        check("update cannot rewrite the primary key", raises(WhitelistError, ws.commit, "w:u2", "admin", [{
            "table": "episodes", "op": "update", "key": {"episode": "e1"}, "values": {"episode": "e9"}}]))
        check("unknown op field rejected", raises(WhitelistError, ws.commit, "w:u3", "admin", [{
            "table": "episodes", "op": "insert", "values": {"episode": "e8"}, "sql": "x"}]))
        check("unknown commit kind rejected", raises(WhitelistError, ws.commit, "w:k", "sneaky", [{
            "table": "episodes", "op": "insert", "values": {"episode": "e8"}}]))
        check("bool column rejects strings", raises(WhitelistError, put, "episodes", {"episode": "e10", "invalid": "yes"}))
        check("Persian procedure text admitted (NFC, ZWNJ)", put("procedure", {
            "proc": "pr:1", "kitchen": "FloorPlan1", "intent_text": "سیب را از یخچال بردار‌و بیاور",
            "intent_key": [{"verb": "pickup", "otype": "Apple", "negated": False}]}, dict(good_ev, rule="R7"))["status"] == "committed")
        check("bidi override in text rejected", raises(WhitelistError, put, "procedure", {
            "proc": "pr:2", "intent_text": "abc‮def"}, good_ev))
        check("zero-width space in text rejected", raises(WhitelistError, put, "procedure", {
            "proc": "pr:3", "intent_text": "a​b"}, good_ev))
        check("sim id inside Gemma hypothesis prose rejected", raises(WhitelistError, put, "hypothesis_audit", {
            "hyp": "h:1", "prose": "the Apple|+01.0|+00.5| was behind"}))
        check("hypothesis prose lives only in hypothesis_audit", put("hypothesis_audit", {
            "hyp": "h:2", "lesson": "ls:1", "prose": "target was likely occluded", "accepted": 0, "reason": "no_evidence_id"})["status"] == "committed")
        check("prose column is not a column of lesson", raises(WhitelistError, put, "lesson", {"lesson": "ls:2", "prose": "x"}, good_ev))
        check("rejected writes left the state unchanged (except the admitted ones)",
              ws.count("episodes") == 1 and ws.count("procedure") == 1 and ws.count("hypothesis_audit") == 1)
        check("whitelist state still integral", ws.check_integrity() == [], ws.check_integrity())
        ws.close()

        # ---------------------------------------------------------------- 7. snapshot / restore / frozen
        src = fresh(tmp, "snapsrc")
        for c in commits:
            src.apply(c)
        man = src.snapshot(Path(tmp) / "snap1", extra={"params_sha256": "abc"})
        check("manifest hash equals live canonical hash", man["state_hash"] == src.state_hash())
        check("manifest carries E_frozen, schema, extra", man["E_frozen"] == 6 and man["schema_version"] == 2 and man["extra"] == {"params_sha256": "abc"})
        check("manifest integrity empty", man["integrity"] == [])
        check("snapshot refuses to overwrite", raises(StoreError, src.snapshot, Path(tmp) / "snap1"))
        check("snapshot verifies", Store.verify_snapshot(Path(tmp) / "snap1") == [])
        src.apply(ep_commit(9))
        check("live store moved on, snapshot unchanged", Store.verify_snapshot(Path(tmp) / "snap1") == []
              and Store.read_manifest(Path(tmp) / "snap1")["state_hash"] == man["state_hash"] != src.state_hash())
        fz_file = Path(tmp) / "snap1" / "ltm.db"
        before_bytes = hashlib.sha256(fz_file.read_bytes()).hexdigest()
        fz = Store.open_frozen(Path(tmp) / "snap1", Path(tmp) / "tl" / "testlog.db")
        check("frozen opens read-only", fz.readonly and fz.clock() == 6)
        check("frozen: write refused", raises(FrozenError, fz.apply, ep_commit(20)))
        check("frozen: raw SQL write refused by the engine", raises(sqlite3.OperationalError, fz._open().execute, "DELETE FROM meta"))
        check("frozen: reads work", fz.get("location_claim", claim="lc:3") is not None)
        check("frozen: test logs go to the separate testlog", fz.log_test("retrieval_log", {
            "retrieval": "r:1", "episode": "tep:1", "query": {"otype": "Apple"}, "candidates": [{"id": "pl:1"}], "E": 6})["status"] == "committed")
        check("frozen: only log tables accepted in testlog", raises(WhitelistError, fz.log_test, "episodes", {"episode": "x"}))
        check("frozen: testlog row exists in testlog.db only", fz.testlog.count("retrieval_log") == 1 and fz.count("retrieval_log") == 0)
        check("frozen state hash == manifest hash after tests", fz.state_hash() == man["state_hash"])
        fz.close()
        fz.testlog.close()
        check("frozen file bytes unchanged after tests", hashlib.sha256(fz_file.read_bytes()).hexdigest() == before_bytes)
        check("no journal left beside the snapshot", not (Path(str(fz_file) + "-journal")).exists())
        rs = Store.restore_snapshot(Path(tmp) / "snap1", Path(tmp) / "restored" / "ltm.db")
        check("restore: identical canonical state", rs.state_hash() == man["state_hash"] and rs.clock() == 6)
        check("restore: is writable and continues the clock", rs.apply(ep_commit(7))["E"] == 7)
        check("restore refuses an existing target", raises(StoreError, Store.restore_snapshot, Path(tmp) / "snap1", Path(tmp) / "restored" / "ltm.db"))
        rs.close()
        # tamper detection
        shutil.copytree(Path(tmp) / "snap1", Path(tmp) / "snapbad")
        c = sqlite3.connect(str(Path(tmp) / "snapbad" / "ltm.db"))
        c.execute("UPDATE location_claim SET conf=0.01 WHERE claim='lc:1'")
        c.commit()
        c.close()
        check("tampered snapshot fails verification", len(Store.verify_snapshot(Path(tmp) / "snapbad")) >= 1)
        check("tampered snapshot cannot be opened frozen", raises(StoreError, Store.open_frozen, Path(tmp) / "snapbad", None))
        check("missing manifest fails verification", Store.verify_snapshot(Path(tmp) / "nowhere") != [])
        # export acks / store_id do not enter the canonical hash
        exp = fresh(tmp, "acks", tracer=ltm.ListTracer())
        exp.apply(ep_commit(1))
        h_acked = exp.state_hash()
        noack = fresh(tmp, "noacks")
        noack.apply(ep_commit(1))
        check("canonical hash independent of export acks and store_id", h_acked == noack.state_hash() and exp.store_id() != noack.store_id())
        # integrity detects tampering with the authoritative log
        c = sqlite3.connect(str(Path(tmp) / "noacks" / "ltm.db"))
        c.execute("UPDATE write_log SET payload='{}' WHERE wl=1")
        c.execute("DELETE FROM evidence WHERE claim_table='search_trial'")
        c.commit()
        c.close()
        noack.close()
        problems = Store(Path(tmp) / "noacks" / "ltm.db").check_integrity()
        check("integrity flags a rewritten write_log row", any("payload hash mismatch" in p for p in problems), problems)
        check("integrity flags a claim without evidence", any("without any evidence" in p for p in problems), problems)
        src.close()
        exp.close()

        # ---------------------------------------------------------------- 8. facade
        lt = ltm.LongTermMemory(Path(tmp) / "facade" / "ltm.db", kitchen="FloorPlan1", tracer=ltm.ListTracer())
        check("facade: params loaded with hash", len(lt.params["_sha256"]) == 64 and lt.params["version"] == 1)
        check("facade: nothing created until first use", not (Path(tmp) / "facade").exists())
        check("facade: epoch starts at 0", lt.current_epoch() == 0)
        r1 = lt.reset_world("blk1-ep1")
        check("facade: reset_world -> epoch 1", r1["epoch"] == 1 and lt.current_epoch() == 1)
        r1b = lt.reset_world("blk1-ep1")
        check("facade: reset retry is idempotent (no double reset)", r1b["status"] == "duplicate" and r1b["epoch"] == 1 and lt.current_epoch() == 1)
        check("facade: second reset -> epoch 2", lt.reset_world("blk1-ep2")["epoch"] == 2)
        check("facade: reset mirrored to the tracer", len(lt.store.tracer.records) == 2)
        lt.apply(ep_commit(1))
        sm = lt.snapshot(Path(tmp) / "facade-snap")
        check("facade: snapshot manifest records params hash and channels", sm["extra"]["params_sha256"] == lt.params["_sha256"]
              and sm["extra"]["channels"] == ["lexical"])
        lt.close()
        ft = ltm.LongTermMemory.open_frozen(Path(tmp) / "facade-snap", Path(tmp) / "fl" / "testlog.db")
        check("facade: frozen refuses reset_world", raises(StoreError, ft.reset_world, "zz"))
        check("facade: frozen kitchen comes from the manifest", ft.kitchen == lt.kitchen)
        ft.close()
        check("facade: frozen open refuses a kitchen that disagrees with the manifest",
              raises(StoreError, ltm.LongTermMemory.open_frozen, Path(tmp) / "facade-snap", Path(tmp) / "fl2" / "testlog.db",
                     kitchen="FloorPlan6"))
        check("facade: no silent default kitchen", raises(StoreError, ltm.LongTermMemory, Path(tmp) / "nok" / "ltm.db"))
        ltm.LongTermMemory.clear_memory(Path(tmp) / "facade" / "ltm.db")
        check("facade: clear_memory removes the db", not (Path(tmp) / "facade" / "ltm.db").exists())
        lt2 = ltm.LongTermMemory(Path(tmp) / "facade" / "ltm.db", kitchen="FloorPlan1")
        check("facade: cleared memory restarts empty at E=0", lt2.clock() == 0 and lt2.current_epoch() == 0)
        lt2.close()
        st = Store(Path(tmp) / "atom" / "ltm.db").stats()
        check("stats report rows, bytes, pragmas", st["rows"]["location_claim"] == 1 and st["bytes"] > 0 and st["pragmas"]["journal_mode"] == "delete")

        # ---------------------------------------------------------------- 9. hygiene: no simulator / network at import
        code = (
            "import sys, socket, builtins\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('ai2thor','requests','urllib3','ultralytics','torch','numpy','cv2'):\n"
            "            raise ImportError('blocked '+name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "def boom(*a, **k): raise RuntimeError('socket opened')\n"
            "socket.socket = boom; socket.create_connection = boom\n"
            f"sys.path.append({HERE!r})\n"
            "import ltm, ltm_store\n"
            "bad = [m for m in ('ai2thor','requests','urllib.request','http.client','ultralytics','torch') if m in sys.modules]\n"
            "print('LOADED_BAD', bad)\n")
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=60)
        check("importing ltm / ltm_store starts no controller and opens no socket", out.returncode == 0 and "LOADED_BAD []" in out.stdout,
              out.stdout + out.stderr)
        src_text = (Path(HERE) / "ltm_store.py").read_text(encoding="utf-8") + (Path(HERE) / "ltm.py").read_text(encoding="utf-8")
        check("source never names the live databases or WAL", "trace.db" not in src_text.replace("sim/trace.db", "") and "memory.db" not in src_text.replace("sim/memory.db", "")
              and "journal_mode=WAL" not in src_text)

        # ---------------------------------------------------------------- 10. nothing leaked outside temp dirs
        check("default sim/step5/ltm.db not created by the selftest", ls.DEFAULT_DB.exists() == default_before)
        for n, mt in guard.items():
            now = (sim_dir / n).stat().st_mtime if (sim_dir / n).exists() else None
            check(f"sim/{n} untouched", now == mt)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = _pass + _fail
    print(f"{_pass}/{total}")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
