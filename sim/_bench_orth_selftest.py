"""Headless gate for `sim/bench_rag.py` section F -- the tab.orth micro-benchmark re-run on the real dataset.

No AI2-THOR, no Ollama, no GPU, no network. Stdlib only. Every file the test writes lives in a
`tempfile.TemporaryDirectory()`; the repo's number sources and stores are hashed (and stat'ed) before and after.

Check groups:

  A  import + isolation     -- bench_rag imports; section F's public surface exists; no socket is ever opened.
  B  section B unchanged    -- hoisting B's engines to module level did not move a single tab.orth number, and
                               the published B_persian rows still equal a fresh recomputation.
  C  perturbations          -- each condition does exactly what its spec says; the normaliser undoes every
                               covered one on every Persian dataset command, and does NOT undo `zwnj_joined`.
  D  probe construction     -- 102 memories / 51 Persian queries / gold = byte-identical scene copies (`records`);
                               66 / 33 / single gold (`unique`); per-level counts.
  E  engines + metrics      -- the tie-set helper agrees with section B's own tie counter; the deterministic
                               winner is always inside the tie set; normalised engines are exactly invariant
                               under every covered perturbation; the raw scan tokenizer already splits at ZWNJ,
                               and raw FTS5 keeps its top-1 under ZWNJ -> space (lower ranks reorder).
  F  end-to-end section F   -- full row grid, metric sanity, determinism, no skipped engine, and (when the
                               results file exists) the published numbers equal a fresh run.
  G  write discipline       -- `--section F` writes only --orth-out; `--no-write` writes nothing; F alone never
                               rewrites sim/bench_rag_results.json; `--section B --no-write` still runs.

Run it (from the repo root):  python sim/_bench_orth_selftest.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import socket
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ["TRACE_ENABLED"] = "0"
os.environ["TRACE_DB"] = os.path.join(tempfile.gettempdir(), "bench_orth_selftest_trace.db")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- no network, ever: any socket use is recorded and refused
_SOCKET_ATTEMPTS: list = []
_real_socket = socket.socket


class _NoSocket(_real_socket):
    def connect(self, *a, **k):                     # noqa: D401
        _SOCKET_ATTEMPTS.append(a)
        raise OSError("network disabled by _bench_orth_selftest")

    def connect_ex(self, *a, **k):
        _SOCKET_ATTEMPTS.append(a)
        return 1


socket.socket = _NoSocket


def _no_create_connection(*a, **k):
    _SOCKET_ATTEMPTS.append(a)
    raise OSError("network disabled by _bench_orth_selftest")


socket.create_connection = _no_create_connection

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def _fingerprint(path):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    with open(path, "rb") as fh:
        return (hashlib.sha256(fh.read()).hexdigest(), st.st_size, st.st_mtime_ns)


GUARDED = [os.path.join(HERE, p) for p in (
    "bench_rag_results.json", "bench_orth_dataset_results.json", "knowledge_base.json", "memory.db",
    os.path.join("dataset", "commands.jsonl"), "loop_reports.jsonl", "trace.db")]


def _strip_latency(rows):
    return [{k: v for k, v in r.items() if k != "latency_p50_ms"} for r in rows]


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def main() -> int:
    before = {p: _fingerprint(p) for p in GUARDED}
    sim_listing = sorted(os.listdir(HERE))

    # ============================================================== A
    print("A  import + isolation")
    import bench_rag as b
    check("A1 bench_rag imports", hasattr(b, "bench_persian"))
    for sym in ("bench_orth_dataset", "build_orth_probes", "add_harakat", "ORTH_CONDITIONS", "ORTH_ENGINES",
                "ORTH_CONDITION_SPEC", "ORTH_RESULTS_PATH", "_overlap_engine", "_fts_engine"):
        check(f"A2 exposes {sym}", hasattr(b, sym))
    check("A3 eight conditions, spec for each",
          list(b.ORTH_CONDITIONS) == ["control", "arabic_yeh", "arabic_kaf", "harakat", "zwnj_space",
                                      "zwnj_joined", "cross_no_harakat", "cross_orthography"]
          and set(b.ORTH_CONDITIONS) == set(b.ORTH_CONDITION_SPEC))
    check("A4 F's results path is NOT the A-E results path",
          os.path.abspath(b.ORTH_RESULTS_PATH) != os.path.abspath(b.RESULTS_PATH)
          and os.path.basename(b.ORTH_RESULTS_PATH) == "bench_orth_dataset_results.json")

    # ============================================================== B
    print("B  section B unchanged by the engine hoist")
    syn_cross = _strip_latency(_quiet(b.bench_persian, True))
    syn_ctrl = _strip_latency(_quiet(b.bench_persian, False))
    expected_cross = [  # tab.orth, as published (docs/thesis/fasl4.tex, B_persian cross_orthography)
        ("shipped: token-overlap scan", 0.7, 0.9, 7), ("+ fa_normalize (index & query)", 1.0, 1.0, 2),
        ("SQLite FTS5 bm25 (raw)", 0.75, 0.95, None), ("SQLite FTS5 bm25 + fa_normalize", 1.0, 1.0, None)]
    got = [(r["engine"], r["recall@1"], r["recall@5"], r["mean_ties@1"]) for r in syn_cross]
    check("B1 cross-orthography rows reproduce tab.orth exactly", got == expected_cross, str(got))
    got = [(r["engine"], r["recall@1"], r["recall@5"], r["mean_ties@1"]) for r in syn_ctrl]
    check("B2 control rows: every engine 1.00/1.00, overlap ties 2",
          got == [(e, 1.0, 1.0, 2 if t is not None else None) for e, _, _, t in expected_cross], str(got))
    check("B3 section F's engine names are section B's engine names",
          tuple(r["engine"] for r in syn_cross) == tuple(b.ORTH_ENGINES))
    if os.path.exists(b.RESULTS_PATH):
        with open(b.RESULTS_PATH, encoding="utf-8") as fh:
            pub = json.load(fh).get("B_persian") or []
        check("B4 published B_persian == fresh recomputation (latency aside)",
              _strip_latency(pub) == syn_cross + syn_ctrl)
    else:
        print("  [SKIP] B4 no sim/bench_rag_results.json on disk")

    # ============================================================== C
    print("C  perturbations")
    with open(b.DATASET, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    fa_cmds = [r["command"] for r in recs if r["lang"] == "fa"]
    C = b.ORTH_CONDITIONS
    check("C1 arabic_yeh leaves no Persian yeh and adds no other change",
          all("ی" not in C["arabic_yeh"](s) and C["arabic_yeh"](s).replace("ي", "ی") == s
              for s in fa_cmds))
    check("C2 arabic_kaf leaves no keheh and adds no other change",
          all("ک" not in C["arabic_kaf"](s) and C["arabic_kaf"](s).replace("ك", "ک") == s
              for s in fa_cmds))
    dam = "ُ"
    check("C3 harakat inserts exactly ONE damma, removable back to the original",
          all(C["harakat"](s).count(dam) == 1 and C["harakat"](s).replace(dam, "") == s for s in fa_cmds))
    check("C4 harakat is deterministic", all(C["harakat"](s) == C["harakat"](s) for s in fa_cmds))
    check("C5 harakat lands on the longest word, after its first letter",
          b.add_harakat("سیب را در یخچال بگذار") == "سیب را در ی" + dam + "خچال بگذار",
          b.add_harakat("سیب را در یخچال بگذار"))
    check("C6 harakat leaves a text with no >=3-letter word unchanged", b.add_harakat("را در") == "را در")
    check("C7 zwnj_space / zwnj_joined remove every ZWNJ",
          all("‌" not in C[c](s) for s in fa_cmds for c in ("zwnj_space", "zwnj_joined")))
    covered = ("control", "arabic_yeh", "arabic_kaf", "harakat", "zwnj_space", "cross_no_harakat",
               "cross_orthography")
    bad = [(c, s) for c in covered for s in fa_cmds if b.fa_normalize(C[c](s)) != b.fa_normalize(s)]
    check("C8 the normaliser undoes every covered perturbation on every Persian command", not bad, str(bad[:2]))
    n_zwnj = sum("‌" in s for s in fa_cmds)
    n_unrec = sum(b.fa_normalize(C["zwnj_joined"](s)) != b.fa_normalize(s) for s in fa_cmds)
    check("C9 zwnj_joined is NOT undone, on exactly the commands that contain a ZWNJ",
          n_zwnj == n_unrec == 20, f"zwnj={n_zwnj} unrecovered={n_unrec}")
    check("C10 cross_orthography == harakat, then yeh, kaf, ZWNJ->space",
          all(C["cross_orthography"](s) == C["cross_no_harakat"](C["harakat"](s)) for s in fa_cmds))
    check("C11 cross_orthography alters every Persian query", all(C["cross_orthography"](s) != s for s in fa_cmds))

    # ============================================================== D
    print("D  probe construction on the real dataset")
    pr = b.build_orth_probes(b.DATASET, "records")
    pu = b.build_orth_probes(b.DATASET, "unique")
    check("D1 records: 102 memories = 51 FA + 51 EN, 51 Persian queries",
          len(pr["docs"]) == 102 and pr["doc_langs"].count("fa") == 51 and pr["doc_langs"].count("en") == 51
          and len(pr["queries"]) == 51)
    check("D2 records: each query's gold contains its own record and only byte-identical texts",
          all(pr["doc_ids"].index(q["id"]) in q["gold"] and all(pr["docs"][j] == q["command"] for j in q["gold"])
              for q in pr["queries"]))
    sizes = sorted(len(q["gold"]) for q in pr["queries"])
    check("D3 records: 9 commands shared by all 3 kitchens -> 27 queries with a 3-member gold set, 24 singletons",
          sizes.count(3) == 27 and sizes.count(1) == 24, str(sizes))
    check("D4 records: levels L1 21 / L2 15 / L3 15",
          {lv: sum(q["level"] == lv for q in pr["queries"]) for lv in ("L1", "L2", "L3")}
          == {"L1": 21, "L2": 15, "L3": 15})
    check("D5 unique: 66 memories (33 FA + 33 EN), 33 queries, single gold each",
          len(pu["docs"]) == 66 and pu["doc_langs"].count("fa") == 33 and len(pu["queries"]) == 33
          and all(len(q["gold"]) == 1 for q in pu["queries"]) and len(set(pu["docs"])) == 66)
    check("D6 English memories are never gold",
          all(pr["doc_langs"][j] == "fa" for q in pr["queries"] for j in q["gold"]))
    try:
        b.build_orth_probes(b.DATASET, "bogus")
        check("D7 unknown variant raises", False)
    except ValueError:
        check("D7 unknown variant raises", True)

    # ============================================================== E
    print("E  engines + metrics")
    syn_docs = [d for d, _ in b._PROBE] + b._PROBE_EN + [d + " (تکرار)" for d, _ in b._PROBE]
    syn_q = [q for _, q in b._PROBE]
    for label, docs, qs in (("section-B corpus", syn_docs, syn_q),
                            ("dataset records", pr["docs"], [C["cross_orthography"](q["command"])
                                                             for q in pr["queries"]])):
        for norm_name, dt, norm in (("raw", [b.tokens(d) for d in docs], lambda s: s),
                                    ("normalised", [b.tokens(b.fa_normalize(d)) for d in docs], b.fa_normalize)):
            ties: list = []
            rank = b._overlap_engine(dt, norm, ties)
            for q in qs:
                rank(q)
            sizes_t = [len(rank.top_tied(q)) for q in qs if rank.top_tied(q)]
            check(f"E1 top_tied agrees with section B's tie counter ({label}, {norm_name})",
                  ties == sizes_t, f"{ties[:5]} vs {sizes_t[:5]}")
    for name, rank in b._orth_engines(pr["docs"]):
        for cond in b.ORTH_CONDITIONS:
            qs = [C[cond](q["command"]) for q in pr["queries"]]
            ok = all((not rank(q)) or rank(q)[0] in rank.top_tied(q) for q in qs)
            if not ok:
                break
        check(f"E2 deterministic winner is inside the tie set, every condition ({name})", ok)
    for probes in (pr, pu):
        eng_v = dict(b._orth_engines(probes["docs"]))
        for name in (b.ORTH_ENGINES[1], b.ORTH_ENGINES[3]):
            rank = eng_v[name]
            ctrl = [rank(q["command"]) for q in probes["queries"]]
            inv = all([rank(C[c](q["command"])) for q in probes["queries"]] == ctrl for c in covered)
            check(f"E3 normalised engine returns the control ranking under every covered perturbation, per query "
                  f"({probes['variant']}, {name})", inv)
    engines = dict(b._orth_engines(pr["docs"]))
    check("E4 the raw scan tokenizer already splits at ZWNJ (why zwnj_space is a no-op for it)",
          all(b.tokens(C["zwnj_space"](s)) == b.tokens(s) for s in fa_cmds))
    # Raw FTS5 is NOT rank-invariant under zwnj_space: the query builder splits on whitespace only, so a ZWNJ word
    # stays ONE quoted term (an FTS5 phrase over its unicode61 pieces) while a spaced word becomes two OR terms.
    # Lower ranks reorder; the top-1 memory does not change on this dataset. Pin exactly that, no more.
    for probes in (pr, pu):
        rank = dict(b._orth_engines(probes["docs"]))[b.ORTH_ENGINES[2]]
        qs = probes["queries"]
        top1_same = all(rank(C["zwnj_space"](q["command"]))[:1] == rank(q["command"])[:1] for q in qs)
        some_reorder = any(rank(C["zwnj_space"](q["command"])) != rank(q["command"]) for q in qs)
        check(f"E5 raw FTS5 under zwnj_space: same top-1 on every query, lower ranks DO reorder (phrase vs OR) "
              f"({probes['variant']})", top1_same and some_reorder)

    # ============================================================== F
    print("F  section F end to end")
    doc1 = _quiet(b.bench_orth_dataset, b.DATASET, reps=3)
    rows = doc1["rows"]
    grid = {(r["variant"], r["condition"], r["engine"]) for r in rows}
    check("F1 full grid: 2 variants x 8 conditions x 4 engines, each once",
          len(rows) == 64 and len(grid) == 64)
    sane = all(0.0 <= r["recall@1"] <= r["recall@5"] <= 1.0 and 0.0 <= r["recall@1_tie_expected"] <= 1.0
               and (r["recall@1_ci"][0] is None or r["recall@1_ci"][0] <= r["recall@1"] <= r["recall@1_ci"][1])
               for r in rows)
    check("F2 metric sanity (0 <= r@1 <= r@5 <= 1, CI brackets r@1)", sane)
    check("F3 control changes 0 queries; cross_orthography changes all of them",
          all(r["n_changed"] == 0 for r in rows if r["condition"] == "control")
          and all(r["n_changed"] == r["n_queries"] for r in rows if r["condition"] == "cross_orthography"))
    byk = {(r["variant"], r["condition"], r["engine"]): r for r in rows}
    check("F4 normalised engines: every covered condition scores exactly as control",
          all(byk[(v, c, e)]["recall@1"] == byk[(v, "control", e)]["recall@1"]
              and byk[(v, c, e)]["recall@5"] == byk[(v, "control", e)]["recall@5"]
              for v in b.ORTH_VARIANTS for c in covered for e in (b.ORTH_ENGINES[1], b.ORTH_ENGINES[3])))
    check("F5 the orthographic failure is real: raw engines lose recall@1 under cross_orthography",
          all(byk[(v, "cross_orthography", e)]["recall@1"] < byk[(v, "control", e)]["recall@1"]
              for v in b.ORTH_VARIANTS for e in (b.ORTH_ENGINES[0], b.ORTH_ENGINES[2])))
    check("F6 no engine skipped (section B has no dense/hybrid engine) and the reason is recorded",
          doc1["skipped_engines"] == [] and "embedding" in doc1["skipped_note"])
    check("F7 dataset fingerprint and shape recorded",
          len(doc1["dataset"]["sha256"]) == 64 and doc1["dataset"]["variants"]["records"]["n_queries"] == 51)
    check("F8 synthetic reference read from the published file", doc1["synthetic_reference"]["available"]
          == os.path.exists(b.RESULTS_PATH))

    # ============================================================== G
    print("G  write discipline")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "orth.json")
        main_before = _fingerprint(b.RESULTS_PATH)
        rc = _quiet(b.main, ["--section", "F", "--orth-out", out])
        check("G1 --section F exits 0 and writes --orth-out", rc == 0 and os.path.exists(out))
        check("G2 --section F leaves sim/bench_rag_results.json byte-identical (and unmodified)",
              _fingerprint(b.RESULTS_PATH) == main_before)
        if os.path.exists(out):
            with open(out, encoding="utf-8") as fh:
                doc2 = json.load(fh)
            check("G3 deterministic: the CLI run equals the direct run (latency aside)",
                  _strip_latency(doc2["rows"]) == _strip_latency(json.loads(json.dumps(rows))))
        published = os.path.join(HERE, "bench_orth_dataset_results.json")
        if os.path.exists(published):
            with open(published, encoding="utf-8") as fh:
                pub = json.load(fh)
            check("G4 published sim/bench_orth_dataset_results.json == a fresh run (latency aside)",
                  _strip_latency(pub["rows"]) == _strip_latency(json.loads(json.dumps(rows)))
                  and pub["dataset"]["sha256"] == doc1["dataset"]["sha256"])
        else:
            print("  [SKIP] G4 sim/bench_orth_dataset_results.json not generated yet")

        out2 = os.path.join(td, "orth_nowrite.json")
        real = b.bench_orth_dataset
        b.bench_orth_dataset = lambda *a, **k: {"rows": []}   # the write gate is what is under test here
        try:
            rc = _quiet(b.main, ["--section", "F", "--no-write", "--orth-out", out2])
        finally:
            b.bench_orth_dataset = real
        check("G5 --no-write writes nothing", rc == 0 and not os.path.exists(out2)
              and _fingerprint(b.RESULTS_PATH) == main_before)
        rc = _quiet(b.main, ["--section", "B", "--no-write"])
        check("G6 --section B --no-write still runs, writes nothing",
              rc == 0 and _fingerprint(b.RESULTS_PATH) == main_before)
        check("G7 the temp dir holds only what the test asked for", sorted(os.listdir(td)) == ["orth.json"])

    after = {p: _fingerprint(p) for p in GUARDED}
    check("G8 guarded repo files byte-identical (sha256, size, mtime)", before == after,
          str([os.path.basename(p) for p in GUARDED if before[p] != after[p]]))
    check("G9 nothing new written into sim/", sorted(os.listdir(HERE)) == sim_listing)
    check("G10 no socket was ever opened", not _SOCKET_ATTEMPTS, str(_SOCKET_ATTEMPTS[:3]))

    total = len(PASS) + len(FAIL)
    print(f"\n_bench_orth_selftest: {len(PASS)}/{total} passed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
