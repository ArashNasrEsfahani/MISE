"""Retrieval-cost benchmark: how long does grounding actually take, where does the time go, and does
the chunkless probe actually retrieve better than the index it replaces?

The thesis's ablation axes (``--rag off|keyword|hybrid|probe|agentic|full``) change *quality*; this
module measures their **cost** as well, so "faster retrieval" can be argued with numbers instead of
adjectives. Six sections (``all`` = A-E; F runs only when named):

  A. **Scaling** — how do the shipped keyword scan and a stdlib SQLite **FTS5/BM25** index scale with
     knowledge-base size (10^2 .. 10^5 entries): index build time, query p50/p95, index bytes.
  B. **Persian robustness** — what Persian orthographic variation (Arabic yeh/kaf, harakat, ZWNJ)
     costs recall@1/@5 on a *labelled probe set*, and how much a Lucene-style normalizer recovers.
  C. **Dense scan** — brute-force cosine over N x d vectors, pure Python vs numpy, at 768/256/64 dims.
  D. **Sidecar I/O** — the cost of loading ``knowledge_base_embeddings.json`` (JSON floats) versus a
     packed float32+base64 sidecar. A fresh :class:`rag.HybridRAG` re-reads it on every construction.
  E. **Retrieval quality and cost on a frozen probe set** — the Phase-10 head-to-head. Every ``--rag``
     arm over the same ~110-entry fixture knowledge base seeded from the real bilingual dataset:
     recall@{1,3,5,10}, MRR@10, nDCG@10, precision@5, scene precision, top-score ties, against
     per-query latency p50/p95/p99/max, one-time setup cost, on-disk index bytes, and the characters
     and tokens each arm injects into the planner prompt.
  F. **Persian robustness on the real dataset** — section B's four engines and metrics re-run with the
     real 102 bilingual commands as the memory and the 51 Persian commands as queries, under eight
     deterministic conditions (control; Arabic yeh; Arabic kaf; one damma; ZWNJ -> space; ZWNJ deleted;
     yeh+kaf+ZWNJ; and the combined cross-orthography mix with a damma), in a ``records`` (gold = byte-identical scene copies) and a
     ``unique`` (one gold per distinct command) variant. It writes ONLY
     ``sim/bench_orth_dataset_results.json`` (``--orth-out``) and never touches the A-E results file.
     Gated by ``sim/_bench_orth_selftest.py``; write-up in ``docs/orth-dataset-benchmark.md``.

**Section E's gold labels are judge-free by construction** (design section 12.2): the 102 dataset ids
are 51 stems x {-en,-fa} verified translation pairs, so seeding the knowledge base from the English
half and querying with the Persian half gives one analytically-known gold per query — and an
analytically-known **0.00** keyword baseline, since a Persian query and an English record share no
tokens. Graded relevance for nDCG comes from overlap between a record's stored plan labels and the
query's gold plan labels, computed with ``eval.labels`` (imported, not reimplemented). This is a
*proxy*: it measures cross-lingual self-retrieval and plan similarity, not whether a retrieved memory
actually helped the planner. Only the SR/PC/TDS harness measures that. Section E exists to make the
retrieval axis interpretable and tunable *before* an expensive eval run.

**No LLM call and no simulator, ever.** The ``agentic`` arm is benched with a `ReplayProvider` that
returns a recorded/derived pivot offline; its model calls are *counted* and priced at ``--llm-ms``
(an assumption, stated as one, never a measurement). The ``hybrid`` arm needs Ollama for its dense
half; when Ollama is not reachable that arm is **skipped with a message**, not crashed and not
silently degraded into a second keyword arm.

**What section E does NOT establish** — three caveats added after review, each now measured rather
than merely disclaimed, because the first version of this file produced numbers that read stronger
than they were:

  E1b **index-coverage sensitivity.** ``agentic_rag._entry_haystack`` indexes a record's ``plan``
     steps and ``tags``; ``knowledge.search`` (the ``keyword`` baseline) does not index the plan. The
     T2 graded-relevance labels are *derived from* the plan. So the arm under test searches a field
     the baseline cannot see and is graded on that same field. E1b re-runs ``probe`` with the plan
     (then plan + tags) stripped from the haystack and prints the delta, which is the amount to
     discount before calling the gap a retrieval-mechanism effect.

  E4 **held-out Persian vocabulary.** The entire cross-lingual result is produced by
     ``agentic_rag.FA_TYPE_LEXICON``, a Persian -> AI2-THOR-type table hand-authored *from this
     dataset*. Every in-sample FA figure in E1/E3 is therefore fitted. E4 deletes, per query, exactly
     the lexicon entries that query uses, and re-measures. That is the generalisation number.

  E4 is also **the only configuration in this file where the multi-hop loop actually runs.** On the
     shipped lexicon the sufficiency gate is satisfied at hop 0 for nearly every query, so ``agentic``
     and ``probe`` come out near-identical and an ``agentic`` vs ``probe`` delta read off E1 would be
     a delta over a mechanism that never fired. The pivot replayed on the hops is still an OFFLINE
     ORACLE, so E4's agentic row is the loop's **ceiling**, not a live result.

Latency is reported with a **percentile-bootstrap 95% CI** (2000 resamples, pinned seed) and the
results JSON records a ``host`` block (CPU / OS / interpreter / which optional packages are
importable), because a millisecond figure without its machine is not a measurement. The one-time
``import chromadb`` that ``HybridRAG.__init__``'s ``chroma_available()`` probe triggers is timed and
reported **separately** from retriever construction: it is a site-packages cost, not an index build,
and folding it into setup made "removes a 585 ms index tax" look like a retrieval result.

Stdlib-first, per CLAUDE.md: imports under a bare-stdlib Python (numpy is imported LAZILY inside the
dense benchmark and reported as "n/a" when missing; ``tiktoken`` is used only if it happens to be
importable). Sections A-D build synthetic fixtures in memory; section E builds a throwaway
KnowledgeBase inside a ``tempfile.TemporaryDirectory()``. It **never** writes
``knowledge_base.json``, ``knowledge_base_embeddings.json``, ``memory.db`` or any ``eval_*.json``,
and reads ``dataset/commands.jsonl`` read-only. The only files it writes in the repo are its own
``sim/bench_rag_results.json`` for sections A-E (change with ``--out``) and, for section F alone,
``sim/bench_orth_dataset_results.json`` (change with ``--orth-out``); ``--no-write`` suppresses both.

Run:  ``python sim/bench_rag.py``                      (everything)
      ``python sim/bench_rag.py --section E``          (just the Phase-10 head-to-head)
      ``python sim/bench_rag.py --quick``              (CI-ish: skips the 100k rows, fewer reps)
      ``python sim/bench_rag.py --section E --emit-probes bench_probes.jsonl``
      ``python sim/bench_rag.py --section F``          (tab.orth on the real dataset; ~seconds, stdlib)
"""
from __future__ import annotations

import argparse
import array
import base64
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATASET = os.path.join(HERE, "dataset", "commands.jsonl")
RESULTS_PATH = os.path.join(HERE, "bench_rag_results.json")

# Tracing OFF before any sibling import. `sim/knowledge.py` carries a tracing hook into `sim/trace.py`, so
# seeding the throwaway fixture knowledge base (110 entries x several arms x several sections) appended
# hundreds of KB to the LIVE sim/trace.db on every benchmark run -- another workstream's store, polluted with
# synthetic bench records, and a write inside the timed region besides. TRACE_ENABLED=0 is trace.py's
# documented opt-out; TRACE_DB is additionally redirected to scratch as a belt-and-braces measure.
os.environ.setdefault("TRACE_ENABLED", "0")
os.environ.setdefault("TRACE_DB", os.path.join(tempfile.gettempdir(), "bench_rag_trace.db"))

# ------------------------------------------------------------------ Persian normalization
# ONE normaliser for the whole repo: agentic_rag.normalize_text is the Lucene ArabicNormalizer +
# PersianNormalizer implementation the retriever itself uses, so sections B and E cannot drift apart.
# The local table below survives only as a standalone fallback (bench_rag must run from a bare
# checkout even if the sibling module is missing).
_FOLD = {
    "ي": "ی", "ى": "ی", "ے": "ی",   # arabic yeh / alef maksura / yeh barree -> farsi yeh
    "ك": "ک",                                            # arabic kaf -> keheh
    "ة": "ه", "ۀ": "ه",                        # teh marbuta / heh+hamza -> heh
    "آ": "ا", "أ": "ا", "إ": "ا",    # alef madda / hamza above / hamza below -> alef
    "ـ": "",                                                   # tatweel (kashida)
    "‌": " ", "‍": "", "‎": "", "‏": "",        # ZWNJ -> space; other bidi marks dropped
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}
_HARAKAT = "ًٌٍَُِّْٰ"
_TABLE = str.maketrans({**_FOLD, **{c: None for c in _HARAKAT}})


def _fa_normalize_fallback(text) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).translate(_TABLE)


try:                                                   # the shipped normaliser, when the sibling is present
    from agentic_rag import normalize_text as fa_normalize   # noqa: F401
    _NORMALIZER = "agentic_rag.normalize_text"
except Exception:                                      # pragma: no cover - bare-checkout fallback
    fa_normalize = _fa_normalize_fallback
    _NORMALIZER = "bench_rag local fallback (agentic_rag unavailable)"


# ------------------------------------------------------------------ the shipped scorer (copied, not imported)
# knowledge.KnowledgeBase._tokens, reproduced so the benchmark can run standalone and so a change there shows up
# as a *diff* here rather than silently moving the baseline.
_STOP = {"the", "a", "an", "to", "in", "on", "of", "and", "it", "is", "with", "into", "then",
         "put", "go", "get", "please", "up", "down", "from", "at", "for", "this", "that"}


def tokens(text: str) -> set:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in str(text)).split()
            if w and w not in _STOP and len(w) > 1}


def _timed(fn, reps: int):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


def _pct(values, q: float) -> float:
    """Nearest-rank percentile over an already-collected sample (no interpolation, no numpy)."""
    xs = sorted(values or [0.0])
    if len(xs) == 1:
        return float(xs[0])
    return float(xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))])


def bootstrap_ci(values, stat=None, reps: int = 2000, level: float = 0.95, seed: int = 20260816):
    """Percentile bootstrap CI for a statistic over `values`. -> ``(lo, hi)``, or ``(None, None)``.

    Stdlib ``random`` with a PINNED seed, so the interval is reproducible from the same sample. `stat`
    defaults to the mean, which is what every quality column in section E is (recall@k and nDCG are
    per-query numbers averaged over the probe set), and applies equally to a latency percentile if one
    is passed in. n < 5 returns ``(None, None)`` rather than a fake interval.

    This exists because the results JSON was quoted with point estimates only while section 14.1 of the
    design doc referred to "the benchmark's bootstrap CI", which did not exist. An n=102 probe set puts
    roughly +/-0.04 on a recall figure near 0.9 — enough that a 0.02 arm-to-arm delta is noise, which is
    exactly the mistake the `agentic` vs `probe` comparison invites.
    """
    xs = [float(v) for v in (values or []) if v is not None]
    n = len(xs)
    if n < 5:
        return (None, None)
    fn = stat or (lambda s: sum(s) / len(s))
    rng = random.Random(seed)
    draws = []
    for _ in range(int(reps)):
        draws.append(fn([xs[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    a = (1.0 - level) / 2.0
    lo = draws[max(0, int(math.floor(a * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1.0 - a) * len(draws))) - 1)]
    return (round(float(lo), 4), round(float(hi), 4))


def _stats_ms(values) -> dict:
    xs = list(values or [])
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None,
                "p50_ci": [None, None]}
    lo, hi = bootstrap_ci(xs, stat=lambda s: _pct(s, 0.50))
    return {"n": len(xs), "p50": round(_pct(xs, 0.50), 4), "p95": round(_pct(xs, 0.95), 4),
            "p99": round(_pct(xs, 0.99), 4), "max": round(max(xs), 4),
            "mean": round(sum(xs) / len(xs), 4), "p50_ci": [lo, hi]}


def host_info() -> dict:
    """CPU / OS / interpreter identification for the results JSON.

    Absent from every prior run, which made the recorded latency numbers unattributable: a p50 in
    milliseconds means nothing without the machine it was measured on, and the 585 ms "keyword setup"
    figure in particular turned out to be a one-time ``import chromadb`` whose presence depends entirely
    on the host's site-packages.
    """
    import platform
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python_impl": platform.python_implementation(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "chromadb_importable": _module_importable("chromadb"),
            "numpy_importable": _module_importable("numpy"),
            "tiktoken_importable": _module_importable("tiktoken")}


def _module_importable(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ================================================================================================
# A. scaling: scan vs FTS5
# ================================================================================================
_EN = ["put", "apple", "fridge", "slice", "bread", "knife", "clean", "plate", "sink", "toaster", "microwave",
       "open", "close", "pick", "place", "counter", "mug", "coffee", "egg", "pan", "cabinet", "drawer",
       "lettuce", "tomato", "bowl", "stove", "burner", "faucet", "trash", "towel"]
_FA = ["سیب", "یخچال", "نان", "چاقو",
       "بشقاب", "سینک", "توستر",
       "مایکروویو", "باز", "بردار",
       "بگذار", "پیشخوان", "لیوان",
       "قهوه", "ماهیتابه", "کابینت",
       "کشو", "کاهو", "گوجه", "کاسه",
       "اجاق", "سطل", "حوله", "تمیز"]
_QUERIES = ["put the apple in the fridge",
            "سیب را در یخچال بگذار",
            "slice bread knife",
            "بشقاب را تمیز کن"]


def _synth(n: int) -> list:
    rnd = random.Random(7)
    return [f"record{i} " + " ".join(rnd.sample(_EN, 6) + rnd.sample(_FA, 6)) for i in range(n)]


def bench_scaling(sizes) -> list:
    print("### A. corpus scaling -- shipped keyword scan vs stdlib SQLite FTS5/BM25\n")
    print("| entries | tokenize corpus (ms) | scan p50/p95 (ms) | FTS5 build (ms) | FTS5 query p50/p95 (ms) | "
          "FTS5 index bytes |")
    print("|---|---|---|---|---|---|")
    rows = []
    for n in sizes:
        docs = _synth(n)
        t = time.perf_counter()
        dts = [tokens(d) for d in docs]
        build_tok = (time.perf_counter() - t) * 1000

        reps = 200 if n <= 1_000 else (40 if n <= 10_000 else 8)
        counter = [0]

        def scan():
            q = tokens(_QUERIES[counter[0] % len(_QUERIES)])
            counter[0] += 1
            scored = [(len(q & dt), i) for i, dt in enumerate(dts)]
            scored = [s for s in scored if s[0]]
            scored.sort(key=lambda s: (-s[0], s[1]))
            return scored[:5]
        p50s, p95s = _timed(scan, reps)

        con = sqlite3.connect(":memory:")
        t = time.perf_counter()
        con.execute('create virtual table f using fts5(body, tokenize="unicode61")')
        con.executemany("insert into f(body) values (?)", ((d,) for d in docs))
        con.commit()
        build_fts = (time.perf_counter() - t) * 1000
        idx_bytes = list(con.execute("select sum(length(block)) from f_data"))[0][0] or 0
        counter[0] = 0

        def fts():
            q = fa_normalize(_QUERIES[counter[0] % len(_QUERIES)])
            counter[0] += 1
            terms = " OR ".join(f'"{w}"' for w in q.split() if len(w) > 1)
            list(con.execute("select rowid, bm25(f) from f where f match ? order by bm25(f) limit 5", (terms,)))
        p50f, p95f = _timed(fts, reps)
        con.close()
        print(f"| {n:,} | {build_tok:.1f} | {p50s:.3f} / {p95s:.3f} | {build_fts:.0f} | {p50f:.3f} / {p95f:.3f} "
              f"| {idx_bytes:,} |")
        rows.append({"entries": n, "tokenize_ms": round(build_tok, 3), "scan_p50_ms": round(p50s, 4),
                     "scan_p95_ms": round(p95s, 4), "fts_build_ms": round(build_fts, 3),
                     "fts_p50_ms": round(p50f, 4), "fts_p95_ms": round(p95f, 4),
                     "fts_index_bytes": idx_bytes})
    print()
    return rows


# ================================================================================================
# B. Persian probe set (labelled)
# ================================================================================================
_PROBE = [
    ("سیب را در یخچال بگذار",
     "سيب را در يخچال بگذار"),
    ("نان را با چاقو ببر",
     "نان را با چاقو ببُر"),
    ("بشقاب را تمیز کن",
     "بشقاب را تميز كن"),
    ("در یخچال را باز کن",
     "در يخچال را باز كن"),
    ("لیوان را روی پیشخوان بگذار",
     "ليوان را روي پيشخوان بگذار"),
    ("قهوه را در ماگ بریز",
     "قهوه را در ماگ بريز"),
    ("تخم‌مرغ را در ماهیتابه بگذار",
     "تخم مرغ را در ماهيتابه بگذار"),
    ("کابینت را باز کن",
     "كابينت را باز كن"),
    ("کشو را ببند", "كشو را ببند"),
    ("کاهو را بشوی", "كاهو را بشوي"),
    ("گوجه را خرد کن",
     "گوجه را خُرد كن"),
    ("کاسه را در سینک بگذار",
     "كاسه را در سينك بگذار"),
    ("اجاق گاز را روشن کن",
     "اجاق گاز را روشن كن"),
    ("شیر آب را ببند", "شير آب را ببند"),
    ("سطل زباله را خالی کن",
     "سطل زباله را خالي كن"),
    ("حوله را بردار", "حوله را بردار"),
    ("توستر را روشن کن", "توستر را روشن كن"),
    ("مایکروویو را باز کن",
     "مايكروويو را باز كن"),
    ("چاقو را بردار", "چاقو را بردار"),
    ("میز را تمیز کن", "ميز را تميز كن"),
]
_PROBE_EN = ["put the apple in the fridge", "slice the bread with a knife", "clean the plate",
             "open the fridge door", "put the mug on the counter", "pour coffee into the mug",
             "put the egg in the pan", "open the cabinet", "close the drawer", "wash the lettuce",
             "chop the tomato", "put the bowl in the sink", "turn on the stove", "turn off the faucet",
             "empty the trash", "pick up the towel", "turn on the toaster", "open the microwave",
             "pick up the knife", "clean the table"]


def _overlap_engine(doc_toks, norm=lambda s: s, ties=None):
    """Section B's token-overlap ranker, hoisted verbatim so section F reuses the SAME engine.

    ``rank(q, k=5)`` -> doc indices, best first; a tie on the overlap count goes to the earliest-inserted
    document. When `ties` is a list, each call with at least one match appends how many documents share the
    top score (the ``mean_ties@1`` column). ``rank.top_tied(q)`` returns that tied index set WITHOUT
    touching `ties` -- section F uses it for the tie-aware recall; it is never on the timed path.
    """
    def scored_for(q):
        qt = tokens(norm(q))
        scored = [(len(qt & dt), i) for i, dt in enumerate(doc_toks)]
        return [s for s in scored if s[0]]

    def rank(q, k=5):
        scored = scored_for(q)
        scored.sort(key=lambda s: (-s[0], s[1]))     # deterministic: earliest-inserted wins a tie
        if scored and ties is not None:
            ties.append(sum(1 for s in scored if s[0] == scored[0][0]))
        return [i for _, i in scored[:k]]

    def top_tied(q):
        scored = scored_for(q)
        if not scored:
            return set()
        best = max(s[0] for s in scored)
        return {i for s, i in scored if s == best}

    rank.top_tied = top_tied
    return rank


def _fts_engine(corpus, norm):
    """Section B's SQLite FTS5 (unicode61) + ``bm25()`` ranker, hoisted verbatim for reuse by section F.

    ``rank(q, k=5)`` -> rowids (= corpus indices) by bm25. ``rank.top_tied(q)`` -> the rowids whose bm25
    equals the best one (relative tolerance 1e-9); used only by section F, never on the timed path.
    """
    con = sqlite3.connect(":memory:")
    con.execute('create virtual table f using fts5(body, tokenize="unicode61")')
    con.executemany("insert into f(rowid, body) values (?, ?)", list(enumerate(corpus)))

    def match_expr(q):
        terms = [t for t in (norm(q)).replace('"', " ").split() if len(t) > 1]
        return " OR ".join(f'"{t}"' for t in terms) if terms else None

    def rank(q, k=5):
        m = match_expr(q)
        if not m:
            return []
        try:
            return [r[0] for r in con.execute(
                "select rowid from f where f match ? order by bm25(f) limit ?", (m, k))]
        except sqlite3.OperationalError:
            return []

    def top_tied(q):
        m = match_expr(q)
        if not m:
            return set()
        try:
            rows = list(con.execute("select rowid, bm25(f) from f where f match ?", (m,)))
        except sqlite3.OperationalError:
            return set()
        if not rows:
            return set()
        best = min(s for _, s in rows)                 # bm25() is negated: lower is better
        tol = 1e-9 * max(1.0, abs(best))
        return {i for i, s in rows if abs(s - best) <= tol}

    rank.top_tied = top_tied
    return rank


def bench_persian(cross_orthography: bool = True) -> list:
    docs = [d for d, _ in _PROBE] + _PROBE_EN + [d + " (تکرار)" for d, _ in _PROBE]
    queries = [q if cross_orthography else d for d, q in _PROBE]
    gold = list(range(len(_PROBE)))
    rows = []

    def report(name, ranker, ties=None):
        h1 = sum(1 for i, q in enumerate(queries) if (r := ranker(q)) and r[0] == gold[i])
        h5 = sum(1 for i, q in enumerate(queries) if gold[i] in (ranker(q) or [])[:5])
        tie_val = statistics.mean(ties) if ties else None      # snapshot BEFORE the timing loop re-runs it
        tie = f"{tie_val:.1f}" if tie_val is not None else "-"
        if ties:
            ties.clear()
        lat = _timed(lambda: ranker(queries[0]), 100)[0]
        if ties:
            ties.clear()
        n = len(queries)
        print(f"| {name} | {h1/n:.2f} | {h5/n:.2f} | {lat:.3f} | {tie} |")
        rows.append({"engine": name, "cross_orthography": cross_orthography,
                     "recall@1": round(h1 / n, 4), "recall@5": round(h5 / n, 4),
                     "latency_p50_ms": round(lat, 4),
                     "mean_ties@1": round(tie_val, 3) if tie_val is not None else None})

    label = "cross-orthography (Arabic yeh/kaf, harakat, ZWNJ dropped)" if cross_orthography \
        else "control: query typed in the SAME orthography as the memory"
    print(f"### B. Persian probe -- {len(docs)} memories, {len(queries)} queries, {label}\n")
    print("| engine | recall@1 | recall@5 | latency p50 (ms) | mean docs tied at top score |")
    print("|---|---|---|---|---|")

    ties: list = []

    report("shipped: token-overlap scan", _overlap_engine([tokens(d) for d in docs], ties=ties), ties)
    report("+ fa_normalize (index & query)",
           _overlap_engine([tokens(fa_normalize(d)) for d in docs], fa_normalize, ties), ties)
    report("SQLite FTS5 bm25 (raw)", _fts_engine(docs, lambda s: s))
    report("SQLite FTS5 bm25 + fa_normalize", _fts_engine([fa_normalize(d) for d in docs], fa_normalize))
    print()
    return rows


# ================================================================================================
# F. section B's orthographic probe, re-run on the REAL bilingual dataset
# ================================================================================================
#: Section F writes here and ONLY here. It never touches RESULTS_PATH (the thesis's number source).
ORTH_RESULTS_PATH = os.path.join(HERE, "bench_orth_dataset_results.json")

_YEH_FA, _YEH_AR = "ی", "ي"          # Persian yeh  -> Arabic yeh
_KAF_FA, _KAF_AR = "ک", "ك"          # keheh        -> Arabic kaf
_ZWNJ = "‌"
_DAMMA = "ُ"                               # the diacritic section B's hand-written variants used (ببُر, خُرد)


def add_harakat(text: str) -> str:
    """Insert ONE damma into the query: after the first letter of its longest word (letters only, >= 3;
    ties -> the first such word). Deterministic and text-only, so the same command always gets the same
    variant. Section B put a damma on a content word in 2 of its 20 hand-written queries; this puts one on
    every query, on the word most likely to be a content word -- a stress version of the same perturbation.
    """
    words = str(text).split(" ")
    best = None
    for i, w in enumerate(words):
        n = sum(1 for c in w if c.isalpha())
        if n >= 3 and (best is None or n > best[0]):
            best = (n, i)
    if best is None:
        return str(text)
    w = words[best[1]]
    for j, c in enumerate(w):
        if c.isalpha():
            words[best[1]] = w[:j + 1] + _DAMMA + w[j + 1:]
            break
    return " ".join(words)


#: Every condition is a pure str -> str function of the ORIGINAL command. `cross_orthography` is the analogue of
#: section B's cross-orthography mix (Arabic yeh/kaf everywhere, a damma, ZWNJ dropped to a space) and is the row
#: to set against tab.orth; the single-perturbation rows decompose it. `zwnj_joined` (ZWNJ deleted, words fused)
#: is OUTSIDE the normaliser's contract on purpose: it shows the failure the normaliser cannot repair.
ORTH_CONDITIONS = {
    "control": lambda s: s,
    "arabic_yeh": lambda s: s.replace(_YEH_FA, _YEH_AR),
    "arabic_kaf": lambda s: s.replace(_KAF_FA, _KAF_AR),
    "harakat": add_harakat,
    "zwnj_space": lambda s: s.replace(_ZWNJ, " "),
    "zwnj_joined": lambda s: s.replace(_ZWNJ, ""),
    "cross_no_harakat": lambda s: s.replace(_YEH_FA, _YEH_AR).replace(_KAF_FA, _KAF_AR).replace(_ZWNJ, " "),
    "cross_orthography": lambda s: add_harakat(s).replace(_YEH_FA, _YEH_AR).replace(_KAF_FA, _KAF_AR)
                                                 .replace(_ZWNJ, " "),
}

ORTH_CONDITION_SPEC = {
    "control": "query typed in the SAME orthography as the memory (unperturbed)",
    "arabic_yeh": "every Persian yeh U+06CC -> Arabic yeh U+064A",
    "arabic_kaf": "every keheh U+06A9 -> Arabic kaf U+0643",
    "harakat": "one damma U+064F after the first letter of the longest word (>= 3 letters)",
    "zwnj_space": "every ZWNJ U+200C -> space (section B's 'ZWNJ dropped')",
    "zwnj_joined": "every ZWNJ U+200C deleted, fusing the words (stress; the normaliser does not claim this)",
    "cross_no_harakat": "arabic_yeh + arabic_kaf + zwnj_space, no damma (section B's synthetic probe put a damma "
                        "on only 2 of 20 queries, so this is the closer like-for-like row)",
    "cross_orthography": "harakat + arabic_yeh + arabic_kaf + zwnj_space (the tab.orth analogue)",
}

ORTH_ENGINES = ("shipped: token-overlap scan", "+ fa_normalize (index & query)",
                "SQLite FTS5 bm25 (raw)", "SQLite FTS5 bm25 + fa_normalize")

ORTH_VARIANTS = ("records", "unique")


def build_orth_probes(dataset_path=DATASET, variant: str = "records") -> dict:
    """-> ``{"docs", "doc_ids", "doc_langs", "queries": [{id, command, level, gold}]}`` from the real dataset.

    ``records``: the memory is ALL dataset commands (51 FA + 51 EN, file order -- the 102-record analogue of
    section B's 60-memory bilingual corpus); one query per Persian record. Nine Persian commands occur
    verbatim in all three kitchens, so a query's gold set is every memory whose text is byte-identical to its
    command: a scene-blind lexical retriever cannot tell those apart, and must not be charged for it.
    ``unique``: duplicate command texts collapsed (first occurrence kept); one query per distinct Persian
    command, exactly one gold each -- section B's single-gold shape, with no free scene-copy ties.

    English siblings are memories (inert distractors, as section B's 20 English documents were), never gold:
    a Persian query shares no token with its English translation, so they cannot be hit lexically.
    """
    with open(dataset_path, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    if variant == "records":
        keep = recs
    elif variant == "unique":
        seen, keep = set(), []
        for r in recs:
            if r["command"] not in seen:
                seen.add(r["command"])
                keep.append(r)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    docs = [r["command"] for r in keep]
    queries = []
    for r in keep:
        if r.get("lang") != "fa":
            continue
        gold = sorted(j for j, d in enumerate(docs) if d == r["command"])
        queries.append({"id": r["id"], "command": r["command"], "level": r.get("level"), "gold": gold})
    return {"variant": variant, "docs": docs, "doc_ids": [r["id"] for r in keep],
            "doc_langs": [r.get("lang") for r in keep], "queries": queries}


def _orth_engines(docs) -> list:
    """The four section-B engines, built by the SAME factories section B uses, over `docs`."""
    return [
        (ORTH_ENGINES[0], _overlap_engine([tokens(d) for d in docs])),
        (ORTH_ENGINES[1], _overlap_engine([tokens(fa_normalize(d)) for d in docs], fa_normalize)),
        (ORTH_ENGINES[2], _fts_engine(docs, lambda s: s)),
        (ORTH_ENGINES[3], _fts_engine([fa_normalize(d) for d in docs], fa_normalize)),
    ]


def _orth_measure(rank, queries_text, probes, changed, reps: int) -> dict:
    """Section B's recall@1 / recall@5 / latency / ties, generalised to a gold SET, plus honest extras."""
    qs = probes["queries"]
    h1, h5, tie_exp, tie_sizes, no_match = [], [], [], [], 0
    for i, q in enumerate(queries_text):
        gold = set(qs[i]["gold"])
        r = rank(q) or []
        h1.append(1.0 if r and r[0] in gold else 0.0)
        h5.append(1.0 if gold & set(r[:5]) else 0.0)
        top = rank.top_tied(q)
        if top:
            tie_sizes.append(len(top))
            tie_exp.append(len(top & gold) / len(top))
        else:
            no_match += 1
            tie_exp.append(0.0)
    lat = _timed(lambda: rank(queries_text[0]), reps)[0] if queries_text else 0.0
    n = len(queries_text) or 1
    ch = [h1[i] for i in range(len(h1)) if changed[i]]
    by_level = {}
    for lvl in sorted({q["level"] for q in qs}):
        idx = [i for i, q in enumerate(qs) if q["level"] == lvl]
        by_level[lvl] = {"n": len(idx), "recall@1": round(sum(h1[i] for i in idx) / len(idx), 4)}
    lo, hi = bootstrap_ci(h1)
    return {"recall@1": round(sum(h1) / n, 4), "recall@1_ci": [lo, hi], "recall@5": round(sum(h5) / n, 4),
            "recall@1_tie_expected": round(sum(tie_exp) / n, 4),
            "recall@1_changed_only": round(sum(ch) / len(ch), 4) if ch else None,
            "mean_ties@1": round(statistics.mean(tie_sizes), 3) if tie_sizes else None,
            "n_no_match": no_match, "latency_p50_ms": round(lat, 4), "recall@1_by_level": by_level}


def _repo_rel(path) -> str:
    """`path` relative to the repo root with forward slashes; the absolute path if it is on another drive."""
    try:
        return os.path.relpath(os.path.abspath(path), os.path.dirname(HERE)).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(path)


def _synthetic_reference() -> dict:
    """Section B's published rows, read READ-ONLY from RESULTS_PATH (the thesis's number source), if present."""
    try:
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            rows = json.load(fh).get("B_persian") or []
    except Exception:
        rows = []
    return {"source": _repo_rel(RESULTS_PATH), "available": bool(rows), "rows": rows}


def bench_orth_dataset(dataset_path=DATASET, reps: int = 100, variants=ORTH_VARIANTS, verbose: bool = True) -> dict:
    """Section F: section B's four engines and metrics over the real 102-record dataset. Pure stdlib; no model,
    no embedding server, no simulator, no network. Reads `dataset_path` read-only and returns the results doc;
    the caller decides whether to write it (to ORTH_RESULTS_PATH, never RESULTS_PATH)."""
    out = print if verbose else (lambda *a, **k: None)
    with open(dataset_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    rows, shapes = [], {}
    for variant in variants:
        probes = build_orth_probes(dataset_path, variant)
        qs = probes["queries"]
        shapes[variant] = {"n_docs": len(probes["docs"]),
                           "n_docs_fa": sum(1 for x in probes["doc_langs"] if x == "fa"),
                           "n_docs_en": sum(1 for x in probes["doc_langs"] if x == "en"),
                           "n_queries": len(qs),
                           "n_queries_with_multi_gold": sum(1 for q in qs if len(q["gold"]) > 1),
                           "queries_by_level": {lv: sum(1 for q in qs if q["level"] == lv)
                                                for lv in sorted({q["level"] for q in qs})}}
        engines = _orth_engines(probes["docs"])
        out(f"### F. Persian orthography on the real dataset -- variant `{variant}`: {len(probes['docs'])} memories "
            f"({shapes[variant]['n_docs_fa']} FA + {shapes[variant]['n_docs_en']} EN), {len(qs)} Persian queries\n")
        out("| condition | engine | changed | recall@1 [95% CI] | recall@5 | recall@1 tie-expected | "
            "mean ties@1 | no match | latency p50 (ms) |")
        out("|---|---|---|---|---|---|---|---|---|")
        for cond, fn in ORTH_CONDITIONS.items():
            texts = [fn(q["command"]) for q in qs]
            changed = [t != q["command"] for t, q in zip(texts, qs)]
            for name, rank in engines:
                m = _orth_measure(rank, texts, probes, changed, reps)
                row = {"variant": variant, "condition": cond, "engine": name,
                       "cross_orthography": cond != "control", "n_queries": len(qs),
                       "n_changed": sum(changed), **m}
                rows.append(row)
                ci = m["recall@1_ci"]
                ci_s = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else ""
                ties = f"{m['mean_ties@1']:.1f}" if m["mean_ties@1"] is not None else "-"
                out(f"| {cond} | {name} | {sum(changed)}/{len(qs)} | {m['recall@1']:.2f}{ci_s} | "
                    f"{m['recall@5']:.2f} | {m['recall@1_tie_expected']:.2f} | {ties} | {m['n_no_match']} | "
                    f"{m['latency_p50_ms']:.4f} |")
        out()

    ref = _synthetic_reference()
    if ref["rows"]:
        out("### F. headline -- recall@1 / recall@5 / mean ties@1, synthetic tab.orth vs the real dataset\n")
        out("| engine | condition | synthetic (B_persian, 20 q / 60 docs) | dataset `records` | dataset `unique` |")
        out("|---|---|---|---|---|")

        def fmt(r):
            if not r:
                return "n/a"
            t = r.get("mean_ties@1")
            return f"{r['recall@1']:.2f} / {r['recall@5']:.2f} / {('%.1f' % t) if t is not None else '-'}"
        for name in ORTH_ENGINES:
            for cond, flag in (("cross_orthography", True), ("cross_no_harakat", True), ("control", False)):
                syn = next((r for r in ref["rows"] if r.get("engine") == name
                            and bool(r.get("cross_orthography")) == flag), None)
                cells = [fmt(next((r for r in rows if r["variant"] == v and r["condition"] == cond
                                   and r["engine"] == name), None)) for v in ORTH_VARIANTS]
                out(f"| {name} | {cond} | {fmt(syn)} | {cells[0]} | {cells[1]} |")
        out("\n(synthetic FTS5 rows carry no tie count: section B never measured one.)\n")

    return {"section": "F_orth_dataset", "ts": time.time(), "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version, "host": host_info(),
            "dataset": {"path": _repo_rel(dataset_path), "sha256": sha, "variants": shapes},
            "normalizer": _NORMALIZER, "engines": list(ORTH_ENGINES),
            "skipped_engines": [],
            "skipped_note": "none: section B (tab.orth) compares four lexical engines only; no dense or hybrid "
                            "engine exists in it, so nothing needs an embedding server and nothing was skipped",
            "conditions": ORTH_CONDITION_SPEC,
            "metrics": {"recall@1": "top-ranked memory is in the query's gold set (section B's definition, "
                                    "generalised from one gold index to a gold set); deterministic tie-break",
                        "recall@1_ci": "percentile bootstrap, 2000 reps, seed 20260816, over per-query hits",
                        "recall@5": "any gold memory in the top 5",
                        "recall@1_tie_expected": "expected recall@1 if ties at the top score were broken "
                                                 "uniformly at random: |gold & top-tied| / |top-tied|",
                        "recall@1_changed_only": "recall@1 over the queries the perturbation actually altered",
                        "mean_ties@1": "mean number of memories sharing the top score, over queries with >= 1 "
                                       "match (section B measured it for the overlap engines only)",
                        "latency_p50_ms": f"section B's method: p50 of {reps} calls on the first query"},
            "tie_break": {"overlap": "earliest-inserted memory (dataset file order)",
                          "fts5": "SQLite's order among equal bm25() values"},
            "synthetic_reference": ref, "rows": rows}


# ================================================================================================
# C. dense scan
# ================================================================================================
def bench_dense(sizes) -> list:
    print("### C. brute-force cosine scan (no ANN index)\n")
    print("| vectors x dim | pure-python p50 (ms) | numpy matmul p50 (ms) |")
    print("|---|---|---|")
    try:
        import numpy as np
    except Exception:
        np = None
    rnd = random.Random(3)
    rows = []
    for n, dim in sizes:
        base = [[rnd.random() for _ in range(dim)] for _ in range(min(n, 10_000))]
        vecs = base * max(1, n // len(base))
        q = [rnd.random() for _ in range(dim)]
        reps = 20 if n <= 1_000 else (5 if n <= 10_000 else 2)

        def py():
            return sorted((sum(a * b for a, b in zip(q, v)) for v in vecs), reverse=True)[:5]
        p50p, _ = _timed(py, reps)
        if np is None:
            p50n = float("nan")
        else:
            M = np.asarray(vecs, dtype="float32")
            M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            qv = np.asarray(q, dtype="float32")
            qv /= np.linalg.norm(qv)
            p50n, _ = _timed(lambda: np.argpartition(-(M @ qv), 5)[:5], 20)
        shown = "n/a (numpy absent)" if p50n != p50n else f"{p50n:.3f}"
        print(f"| {n:,} x {dim} | {p50p:.2f} | {shown} |")
        rows.append({"vectors": n, "dim": dim, "python_p50_ms": round(p50p, 4),
                     "numpy_p50_ms": None if p50n != p50n else round(p50n, 4)})
    print()
    return rows


# ================================================================================================
# D. embedding sidecar I/O
# ================================================================================================
def bench_sidecar(specs) -> list:
    print("### D. embedding sidecar load cost (JSON floats vs packed float32+base64)\n")
    print("| entries x dim | JSON bytes | json.load p50 (ms) | packed bytes | packed load p50 (ms) |")
    print("|---|---|---|---|---|")
    rnd = random.Random(11)
    tmp = tempfile.mkdtemp(prefix="bench_rag_")
    rows = []
    for n, dim in specs:
        vecs = {str(i): [round(rnd.uniform(-1, 1), 6) for _ in range(dim)] for i in range(n)}
        pj = os.path.join(tmp, f"s{n}_{dim}.json")
        with open(pj, "w") as fh:
            json.dump({"model": "nomic-embed-text", "vectors": vecs}, fh)
        p50j, _ = _timed(lambda: json.load(open(pj)), 7)

        pb = os.path.join(tmp, f"s{n}_{dim}.packed.json")
        with open(pb, "w") as fh:
            json.dump({"model": "nomic-embed-text", "dtype": "f4",
                       "vectors": {k: base64.b64encode(array.array("f", v).tobytes()).decode()
                                   for k, v in vecs.items()}}, fh)

        def load_packed():
            blob = json.load(open(pb))
            return {k: array.array("f", base64.b64decode(v)) for k, v in blob["vectors"].items()}
        p50b, _ = _timed(load_packed, 7)
        jb, bb = os.path.getsize(pj), os.path.getsize(pb)
        print(f"| {n} x {dim} | {jb:,} | {p50j:.1f} | {bb:,} | {p50b:.1f} |")
        rows.append({"entries": n, "dim": dim, "json_bytes": jb, "json_load_p50_ms": round(p50j, 3),
                     "packed_bytes": bb, "packed_load_p50_ms": round(p50b, 3)})
        os.remove(pj)
        os.remove(pb)
    os.rmdir(tmp)
    print()
    return rows


# ================================================================================================
# E. retrieval quality and cost on a frozen probe set  (the Phase-10 head-to-head)
# ================================================================================================
MODES = ("off", "keyword", "hybrid", "probe", "agentic", "full")

#: L1 gold goals name a final STATE; this is the state -> single-action table the fixture uses to give
#: every L1 record a stored plan (design section 12.1). No LLM, no simulator.
_GOAL2ACT = {("isOpen", True): "open", ("isOpen", False): "close",
             ("isPickedUp", True): "pickup", ("isToggled", True): "toggle_on",
             ("isToggled", False): "toggle_off", ("temperature", "Hot"): "heat_object",
             ("temperature", "Cold"): "cool_object", ("isSliced", True): "slice",
             ("isCooked", True): "cook_food"}


def gold_plan(rec) -> list:
    """The plan stored for a dataset record, derived from its gold ONLY (no LLM, no simulator).

    L2 -> one step per ``gold['sequence']`` label, in order.
    L3 -> the same over ``sorted(gold['actions'])`` (a set has no order; sorting keeps it deterministic).
    L1 -> ONE step from ``gold['goal'][0]`` via ``_GOAL2ACT`` (``parentReceptacles`` -> pick_and_place).
    """
    g = rec.gold or {}
    if rec.level == "L2":
        labs = list(g.get("sequence") or [])
    elif rec.level == "L3":
        labs = sorted(g.get("actions") or [])
    else:
        labs = []
        for a in (g.get("goal") or [])[:1]:
            key, val = a.get("key"), a.get("value")
            if key == "parentReceptacles":
                labs.append(f"pick_and_place {a.get('type')}")
            else:
                act = _GOAL2ACT.get((key, val))
                if act:
                    labs.append(f"{act} {a.get('type')}")
    plan = []
    for lab in labs:
        t = str(lab).split()
        if len(t) >= 2:
            plan.append({"action": t[0], "object": " ".join(t[1:])})
    return plan


def build_fixture_kb(dataset_path, tmpdir, lang="en", scenes=None, seed=0):
    """-> ``(kb, meta)``. A throwaway KnowledgeBase seeded from the dataset + skill docs + kitchen priors.

    One ``procedure`` per `lang` record (plan = :func:`gold_plan`, scope = the record's scene,
    ``source='environment'``, ``confidence=0.9``, tags = the gold object types, ``uses`` spread
    deterministically over 1..3); one ``lesson`` for every 4th record; every ``skills.skill_docs()``
    line and every ``seed_memory.KITCHEN_FACTS`` entry as a global ``prior`` fact. Result ~110 entries,
    matching the realistic knowledge base the surveys measured against.

    `meta` records exactly which KB entry id came from which dataset record id — that mapping IS the
    T1 gold label. Fully deterministic: `seed` exists only to pin the signature, nothing samples.
    """
    import knowledge
    import eval as ev                                   # loads without ai2thor (proved by _eval_selftest)

    records = ev.load_dataset(dataset_path, lang="all", limit=None)
    if scenes:
        records = [r for r in records if r.scene in set(scenes)]
    kb = knowledge.KnowledgeBase(path=os.path.join(tmpdir, "bench_kb.json"))
    now = 1_750_000_000.0                               # pinned; nothing here depends on wall-clock time
    meta = {"kb_lang": lang, "entries": {}, "by_record": {}, "n_lessons": 0, "n_facts": 0}
    half = [r for r in records if r.lang == lang]
    for i, r in enumerate(half):
        plan = gold_plan(r)
        e = kb.add_procedure(r.command, plan, why="bench fixture", scene=r.scene,
                             tags=sorted({s["object"] for s in plan}),
                             source="environment", ts=now, confidence=0.9)
        e["uses"] = 1 + (i % 3)
        meta["by_record"][r.id] = e["id"]
        meta["entries"][str(e["id"])] = {"record": r.id, "kind": "procedures",
                                         "labels": ev.labels(plan), "scene": r.scene}
        if i % 4 == 0 and plan:
            le = kb.add_lesson(r.command,
                               f"{plan[0]['action']} {plan[0]['object']} failed: precondition unmet",
                               f"make sure the {plan[0]['object']} is reachable first",
                               why="bench fixture", scene=r.scene, source="environment",
                               ts=now, confidence=0.6)
            meta["entries"][str(le["id"])] = {"record": r.id, "kind": "lessons", "labels": [],
                                              "scene": r.scene}
            meta["n_lessons"] += 1
    try:
        import skills
        for doc in skills.skill_docs():
            kb.add_fact(doc, why="bench fixture", source="prior", ts=now, confidence=1.0)
            meta["n_facts"] += 1
    except Exception as exc:                             # pragma: no cover
        print(f"  [note] skills.skill_docs() unavailable ({exc!r}); facts under-seeded")
    try:
        import seed_memory
        for text, tags in seed_memory.KITCHEN_FACTS:
            kb.add_fact(text, why="bench fixture", tags=list(tags), source="prior", ts=now, confidence=1.0)
            meta["n_facts"] += 1
    except Exception as exc:                             # pragma: no cover
        print(f"  [note] seed_memory.KITCHEN_FACTS unavailable ({exc!r}); facts under-seeded")
    meta["stats"] = kb.stats()
    return kb, meta


def build_probes(dataset_path, meta, kb_lang="en", query_langs=("en", "fa"), scenes=None) -> list:
    """-> the frozen probe set. Three judge-free gold tiers (design section 12.2).

    T1 TWIN (binary, primary) — the 102 ids are 51 stems x {-en,-fa} verified translation pairs, so the
       single gold id for a query is the KB entry seeded from its twin. For ``query_lang='fa'`` against an
       English-seeded KB, pure keyword recall is **analytically 0.00**: zero token intersection.
    T2 LABEL OVERLAP (graded) — ``grade(e) = |labels(e.plan) & gold_labels(q)| / |gold_labels(q)|``,
       computed with ``eval.labels`` / ``eval.action_label``. Relevant iff ``grade >= 0.5`` and
       ``scope in {'global', q.scene}``. Feeds nDCG@10 with real graded relevance and no annotation.
    T3 HARD NEGATIVES — the same-label entry seeded from a DIFFERENT scene, and entries sharing one object
       type but no action. Feeds precision@5 and the scene-scoping test.

    Deterministic: nothing samples. ``--emit-probes`` writes it as JSONL and the sha256 is printed, so an
    accidental dataset edit cannot silently move the baseline.
    """
    import eval as ev

    records = ev.load_dataset(dataset_path, lang="all", limit=None)
    if scenes:
        records = [r for r in records if r.scene in set(scenes)]
    entries = meta["entries"]
    stems = {}
    for r in records:
        stems.setdefault(r.id.rsplit("-", 1)[0], {})[r.lang] = r

    probes = []
    for stem in sorted(stems):
        pair = stems[stem]
        src = pair.get(kb_lang)
        if src is None:
            continue
        gold_id = meta["by_record"].get(src.id)
        if gold_id is None:
            continue
        for qlang in query_langs:
            q = pair.get(qlang)
            if q is None:
                continue
            gold_labels = set(ev.labels(gold_plan(q)))
            # A synthetic OBSERVATION: the object types a planner standing in this kitchen would see for
            # this command, taken from the record's own gold. There is no simulator here, so this is a
            # STAND-IN -- but the sufficiency gate is defined over the OBSERVATION, and benching the
            # agentic arm with an empty one would silently make every gate pass at hop 0 and turn the arm
            # into a second copy of `probe`.
            obs_types = sorted({str(a.get("type")) for a in ((q.gold or {}).get("goal") or [])
                                if a.get("type")}
                               | {" ".join(str(l).split()[1:]) for l in gold_labels})
            graded, hard_neg = {}, []
            for eid, info in entries.items():
                labs = set(info.get("labels") or [])
                if not gold_labels or not labs:
                    continue
                grade = len(labs & gold_labels) / float(len(gold_labels))
                scoped = info.get("scene") in (None, "global", q.scene)
                if grade >= 0.5 and scoped:
                    graded[eid] = round(grade, 4)
                elif grade >= 0.5 and not scoped:
                    hard_neg.append(int(eid))            # T3a: right labels, wrong kitchen
                elif 0.0 < grade < 0.5:
                    hard_neg.append(int(eid))            # T3b/c: partial overlap only
            graded[str(gold_id)] = 1.0                   # the twin is relevance 1 by construction
            probes.append({"qid": f"{stem}-{qlang}", "query": q.command, "lang": qlang,
                           "level": q.level, "scene": q.scene, "gold_ids": [gold_id],
                           "observation": ", ".join(t for t in obs_types if t),
                           "graded": graded, "hard_neg": sorted(set(hard_neg))})
    return probes


def probes_sha256(probes) -> str:
    blob = "\n".join(json.dumps(p, sort_keys=True, ensure_ascii=False) for p in probes)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ReplayProvider:
    """An `llm.LLMProvider`-shaped stub that REPLAYS a pivot offline. Zero network, zero model calls.

    The agentic arm's extra hops are a model cost, and a benchmark that actually called the model would
    be measuring Ollama, not retrieval. So the pivot reply is either read from ``--pivot-replay FILE``
    (``{qid: {"types": [...], "verbs": [...]}}``) or derived from the probe's own gold object types —
    an **oracle** pivot, i.e. an upper bound on what the model could return, and it is labelled as such
    in the output. Calls are counted and priced separately at ``--llm-ms``.
    """

    def __init__(self, table=None):
        self.table = table or {}
        self.next_key = None
        self.calls = 0

    def chat(self, model, messages, **kw):
        self.calls += 1
        payload = self.table.get(self.next_key) or {"types": [], "verbs": [], "ids": []}
        return json.dumps({"types": list(payload.get("types") or [])[:6],
                           "verbs": list(payload.get("verbs") or [])[:4],
                           "ids": list(payload.get("ids") or [])[:4]})


def _ollama_up(endpoint: str, timeout: float = 1.5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def _tiktoken_counter():
    """`tiktoken` if it happens to be importable, else None. Never a dependency."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s or ""))
    except Exception:
        return None


def _dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _score_ranking(ranked_ids, probe, k_max=10) -> dict:
    """recall@{1,3,5,10} / MRR@10 / nDCG@10 / precision@5 / scene precision for one query."""
    gold = set(probe["gold_ids"])
    graded = {int(k): float(v) for k, v in (probe.get("graded") or {}).items()}
    out = {}
    for k in (1, 3, 5, 10):
        out[f"recall@{k}"] = 1.0 if gold & set(ranked_ids[:k]) else 0.0
    rr = 0.0
    for i, eid in enumerate(ranked_ids[:k_max], start=1):
        if eid in gold:
            rr = 1.0 / i
            break
    out["MRR@10"] = rr
    gains = [graded.get(eid, 0.0) for eid in ranked_ids[:k_max]]
    ideal = sorted(graded.values(), reverse=True)[:k_max]
    idcg = _dcg(ideal)
    out["nDCG@10"] = (_dcg(gains) / idcg) if idcg > 0 else 0.0
    top5 = ranked_ids[:5]
    out["precision@5"] = (sum(1 for e in top5 if graded.get(e, 0.0) >= 0.5) / 5.0) if top5 else 0.0
    return out


def _deep_hits(mode, retr, probe, kb_size):
    """The arm's FULL candidate pool for one query (k = |KB|), for the ranking-degeneracy statistic.

    Untimed and best-effort: a failure returns [] and ties@1 simply reports 0 for that query rather than
    aborting the run. `full` is already the whole KB, so its pool is its output.
    """
    try:
        if mode == "off" or retr is None:
            return []
        if mode == "full":
            return retr.dump(scene=probe["scene"])
        if mode in ("keyword", "hybrid"):
            return retr.search(probe["query"], scene=probe["scene"], k=kb_size,
                               alpha=(1.0 if mode == "keyword" else 0.5))
        return retr.search(probe["query"], scene=probe["scene"], k=kb_size,
                           observation=probe.get("observation", ""), model="replay")
    except Exception:
        return []


def _run_mode(mode, probes, kb_path, tmpdir, args, replay_table, type_pairs=None, heldout=False):
    """Run one --rag arm over the whole probe set. -> a result dict (or a 'skipped' dict).

    `type_pairs` (see ``agentic_rag.held_out_type_pairs``) replaces the object-type lexicon for the
    probe/agentic arms. It is how section E4 measures HELD-OUT Persian vocabulary; `None` = the shipped,
    dataset-fitted lexicon.
    """
    import agentic_rag as ar
    import knowledge

    tik = _tiktoken_counter()
    endpoint_dead = "http://127.0.0.1:1"                # guarantees the keyword arm touches no network

    # ---- setup, split in two because the halves have different owners: every arm pays `kb_load_ms`
    # (parsing the one JSON store), and only some arms pay `retriever_ms` (sidecar parse, index build,
    # and -- a real measured surprise -- HybridRAG.__init__'s `chroma_available()` probe, which IMPORTS
    # chromadb when it happens to be installed on the host).
    t0 = time.perf_counter()
    kb = knowledge.KnowledgeBase(path=kb_path)
    kb_load_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    retr = None
    provider = None
    index_bytes = 0
    note = ""
    import_ms = 0.0
    if mode in ("keyword", "hybrid"):
        import rag as ragmod
        # SPLIT OUT the one-time `import chromadb`. HybridRAG.__init__ probes chroma_available(), and on a
        # host where chromadb is installed that probe IMPORTS it -- 585 ms in the first recorded run. That is
        # a process-level module import, not an index build, not a staleness cost and not retrieval: run the
        # benchmark in a process where chromadb is already imported (or on a host without it) and the 585 ms
        # vanishes. Reporting it inside `retriever_ms` made "probe removes a 585 ms index tax" look like a
        # retrieval result when it is a site-packages result, so it is measured and reported separately and
        # `retriever_ms` below is the honest, import-free setup number.
        t_imp = time.perf_counter()
        try:
            ragmod.HybridRAG.chroma_available()
        except Exception:
            pass
        import_ms = (time.perf_counter() - t_imp) * 1000.0
        t1 = time.perf_counter()
        sidecar = os.path.join(tmpdir, f"bench_sidecar_{mode}.json")
        endpoint = args.endpoint if mode == "hybrid" else endpoint_dead
        retr = ragmod.HybridRAG(kb=kb, endpoint=endpoint, embed_path=sidecar)
        if mode == "hybrid":
            n = retr.reindex()                          # the index build IS part of hybrid's setup cost
            note = f"reindexed {n} entries"
            index_bytes = os.path.getsize(sidecar) if os.path.exists(sidecar) else 0
    elif mode in ("probe", "agentic", "full"):
        provider = ReplayProvider(replay_table) if mode == "agentic" else None
        retr = ar.AgenticRAG(kb=kb, provider=provider, model="replay",
                             hops=(0 if mode != "agentic" else args.agentic_hops),
                             k=args.k, pivot_mode=("lex" if mode != "agentic" else "llm"),
                             rank_mode=args.rank, quota_mode="typed",
                             gate=(mode == "agentic"),
                             cache=(args.cache and not heldout),   # the cache key omits the lexicon
                             compact=True,
                             level_policy="est", budget_chars=args.budget,
                             type_pairs=type_pairs)
        _fp = ar.kb_fingerprint(kb)                     # the agentic arms' whole "index build"
        index_bytes = 0
    retriever_ms = (time.perf_counter() - t1) * 1000.0
    setup_ms = kb_load_ms + retriever_ms
    kb_size = max(1, len(ar.kb_entries(kb)))

    per_query, lat, cold = [], [], None
    llm_calls = 0
    reps = max(1, int(args.reps))
    for rep in range(reps):
        for p in probes:
            if provider is not None:
                provider.next_key = p["qid"]
            if heldout and retr is not None and hasattr(retr, "type_pairs"):
                # LEAVE-THIS-QUERY'S-VOCABULARY-OUT: delete from the lexicon exactly the surface forms this
                # command uses, then retrieve. Set per query, so `type_pairs` has to be swapped in the loop;
                # the probe cache is forced off for this run because its key does not include the lexicon.
                retr.type_pairs = ar.held_out_type_pairs(surface=p["query"])
            t = time.perf_counter()
            try:
                if mode == "off":
                    hits = []
                elif mode in ("keyword", "hybrid"):
                    hits = retr.search(p["query"], scene=p["scene"], k=args.k,
                                       alpha=(1.0 if mode == "keyword" else 0.5))
                elif mode == "full":
                    hits = retr.dump(scene=p["scene"])
                else:
                    hits = retr.search(p["query"], scene=p["scene"], k=args.k,
                                       observation=p.get("observation", ""), model="replay")
            except Exception as exc:                    # an arm that crashes is reported, not hidden
                note = (note + f" | ERROR {exc!r}").strip(" |")
                hits = []
            dt = (time.perf_counter() - t) * 1000.0
            lat.append(dt)
            if cold is None:
                cold = dt
            if rep:                                     # score once; later reps are timing samples only
                continue

            ranked = [(h.get("entry") or {}).get("id") for h in hits]
            block = _render_block(hits)
            row = _score_ranking(ranked, p)
            in_scope = [h for h in hits if (h.get("entry") or {}).get("scope",
                        (h.get("entry") or {}).get("scene") or "global") in ("global", p["scene"])]
            row["scene_precision"] = (len(in_scope) / len(hits)) if hits else 0.0
            # ties@1 is RANKING DEGENERACY, so it has to be measured over the CANDIDATE POOL, not over the
            # k hits that survived truncation. Computed over the returned hits it was structurally capped at
            # k: the `keyword` arm returned 4.1 hits on average so its ties@1 could never exceed 4.1, and the
            # `full` arm scored 67.33 purely because dump() gives every record score 0.0 and returns 67 of
            # them. The column measured how many hits came back, not how degenerate the ordering was.
            # `_deep_hits` re-runs the SAME arm at k = |KB| outside the timing loop; nothing here is timed.
            deep = _deep_hits(mode, retr, p, kb_size)
            top = max([float(h.get("score") or 0.0) for h in deep], default=0.0)
            row["ties@1"] = float(sum(1 for h in deep if float(h.get("score") or 0.0) == top)) if deep else 0.0
            row["pool"] = float(len(deep))
            row["n_returned"] = float(len(hits))
            row["block_chars"] = float(len(block))
            row["est_tokens"] = float(_est_tokens(block))
            row["tiktoken"] = float(tik(block)) if tik else None
            row["hard_neg@5"] = float(sum(1 for e in ranked[:5] if e in set(p.get("hard_neg") or [])))
            # Telemetry, so "the agentic arm spent no hops" reads as a MEASUREMENT rather than a bug.
            tele = retr.last_telemetry() if hasattr(retr, "last_telemetry") else {}
            row["hops"] = float(tele.get("hops") or 0) if tele else None
            row["gate_stopped_at_hop0"] = (1.0 if (tele.get("gate_stopped") and not tele.get("hops"))
                                           else 0.0) if tele else None
            row["qid"], row["lang"], row["level"] = p["qid"], p["lang"], p["level"]
            row["est_level"] = tele.get("level") if tele else None
            per_query.append(row)
    if provider is not None:
        llm_calls = provider.calls // reps

    cache_hit_rate = None
    if mode in ("probe", "agentic", "full") and getattr(retr, "cache", None) is not None:
        st = retr.cache.stats()
        tot = st["hits"] + st["misses"]
        cache_hit_rate = round(st["hits"] / tot, 4) if tot else None

    return {"mode": mode, "skipped": False, "note": note, "setup_ms": round(setup_ms, 3),
            "kb_load_ms": round(kb_load_ms, 3), "retriever_ms": round(retriever_ms, 3),
            "module_import_ms": round(import_ms, 3), "kb_size": kb_size,
            "index_bytes": index_bytes, "latency_ms": _stats_ms(lat), "cold_ms": round(cold or 0.0, 4),
            "llm_calls_total": llm_calls, "llm_calls_per_query": round(llm_calls / max(1, len(probes)), 3),
            "assumed_model_ms": round(llm_calls / max(1, len(probes)) * args.llm_ms, 1),
            "cache_hit_rate": cache_hit_rate, "heldout_lexicon": bool(heldout), "per_query": per_query,
            "overall": _aggregate(per_query), "by_lang": _group(per_query, "lang"),
            "by_level": _group(per_query, "level")}


def _render_block(hits) -> str:
    """The memory block these hits would inject, via eval's single source of truth when available."""
    if not hits:
        return ""
    try:
        import eval as ev
        return ev._render_hits(hits)
    except Exception:                                    # pragma: no cover
        import agentic_rag as ar
        return "\n".join(ar._default_render(h.get("kind", ""), h.get("entry") or {}) for h in hits)


def _est_tokens(text) -> int:
    try:
        import agentic_rag as ar
        return ar.est_tokens(text)
    except Exception:                                    # pragma: no cover
        s = str(text or "")
        n_ascii = sum(1 for c in s if ord(c) < 128)
        return int(math.ceil(n_ascii / 4.0 + (len(s) - n_ascii) / 1.5))


_METRIC_KEYS = ("recall@1", "recall@3", "recall@5", "recall@10", "MRR@10", "nDCG@10", "precision@5",
                "scene_precision", "ties@1", "pool", "n_returned", "block_chars", "est_tokens", "tiktoken",
                "hard_neg@5", "hops", "gate_stopped_at_hop0")


#: Columns a write-up is likely to quote as a headline. Only these get a bootstrap CI, because the CI is
#: 2000 resamples per column and the point of the interval is to stop a 0.02 delta being read as a result.
_CI_KEYS = ("recall@1", "recall@5", "recall@10", "MRR@10", "nDCG@10", "precision@5")


def _aggregate(rows) -> dict:
    out = {"n": len(rows)}
    for key in _METRIC_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = round(sum(vals) / len(vals), 4) if vals else None
        if key in _CI_KEYS:
            lo, hi = bootstrap_ci(vals)
            out[key + "_ci"] = [lo, hi]
    return out


def _group(rows, field) -> dict:
    buckets = {}
    for r in rows:
        buckets.setdefault(r.get(field), []).append(r)
    return {k: _aggregate(v) for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def bench_section_e(args) -> dict:
    """The Phase-10 head-to-head. Returns a JSON-able result dict; prints the readable tables."""
    print("### E. retrieval quality and cost on a frozen probe set (Phase-10 head-to-head)\n")
    if not os.path.exists(args.dataset):
        print(f"  SKIPPED: dataset not found at {args.dataset}. Section E needs the bilingual "
              f"dataset to build its judge-free gold labels.\n")
        return {"skipped": True, "reason": "dataset absent"}
    try:
        import agentic_rag                              # noqa: F401
        import knowledge                                # noqa: F401
        import eval                                     # noqa: F401,A004
    except Exception as exc:
        print(f"  SKIPPED: sibling modules unavailable ({exc!r}).\n")
        return {"skipped": True, "reason": repr(exc)}

    tmpdir = tempfile.mkdtemp(prefix="bench_rag_E_")
    try:
        kb, meta = build_fixture_kb(args.dataset, tmpdir, lang=args.kb_lang, scenes=args.scenes)
        probes = build_probes(args.dataset, meta, kb_lang=args.kb_lang,
                              query_langs=tuple(args.query_langs), scenes=args.scenes)
        if args.limit:
            probes = probes[:int(args.limit)]
        sha = probes_sha256(probes)
        print(f"  fixture KB: {meta['stats']} = {sum(meta['stats'].values())} entries seeded from the "
              f"{args.kb_lang.upper()} half of {os.path.basename(args.dataset)}")
        print(f"  probe set : {len(probes)} queries ({'/'.join(args.query_langs)}), "
              f"sha256[:16]={sha}, {args.reps} timing rep(s) -> {len(probes) * args.reps} timings/arm")
        print(f"  normaliser: {_NORMALIZER}")
        if args.emit_probes:
            with open(args.emit_probes, "w", encoding="utf-8") as fh:
                for p in probes:
                    fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"  probes written to {args.emit_probes}")

        # The agentic arm's pivot: replayed from a file, or derived offline from each probe's gold types.
        replay_table = {}
        if args.pivot_replay and os.path.exists(args.pivot_replay):
            with open(args.pivot_replay, "r", encoding="utf-8") as fh:
                replay_table = json.load(fh)
            pivot_src = f"replayed from {os.path.basename(args.pivot_replay)}"
        else:
            import eval as ev
            for p in probes:
                labs = meta["entries"].get(str(p["gold_ids"][0]), {}).get("labels") or []
                types = [ev.canon_type(" ".join(str(l).split()[1:])) for l in labs]
                verbs = [str(l).split()[0] for l in labs]
                replay_table[p["qid"]] = {"types": [t for t in types if t][:6],
                                          "verbs": [v for v in verbs if v][:4], "ids": []}
            pivot_src = "ORACLE (derived offline from each probe's gold object types)"
        print(f"  agentic pivot: {pivot_src} -- ZERO live model calls; hop calls are counted and priced "
              f"at --llm-ms {args.llm_ms:.0f} ms")

        ollama = _ollama_up(args.endpoint)
        modes = [m for m in MODES if m in args.modes]
        results = {}
        for mode in modes:
            if mode == "hybrid" and not ollama:
                print(f"  [skip] --rag hybrid: no Ollama at {args.endpoint}. Its dense half cannot be "
                      f"measured offline; run this section again with the server up rather than "
                      f"reading a keyword-only number as 'hybrid'.")
                results[mode] = {"mode": mode, "skipped": True,
                                 "reason": f"Ollama unreachable at {args.endpoint}"}
                continue
            results[mode] = _run_mode(mode, probes, kb.path, tmpdir, args, replay_table)

        _print_quality(results)
        _print_cost(results, args)
        _print_by(results, "by_lang", "language")
        _print_by(results, "by_level", "level")
        haystack = _bench_haystack_sensitivity(probes, kb.path, tmpdir, args, replay_table)
        heldout = _bench_heldout_lexicon(probes, kb.path, tmpdir, args, replay_table, results)
        _print_headline(results, args, haystack, heldout)
        _print_notes(results, ollama, args, haystack, heldout)
        return {"skipped": False, "probes_sha256": sha, "n_probes": len(probes),
                "kb_stats": meta["stats"], "kb_lang": args.kb_lang,
                "query_langs": list(args.query_langs), "reps": args.reps,
                "ollama": ollama, "pivot_source": pivot_src, "normalizer": _NORMALIZER,
                "E1b_haystack_sensitivity": haystack, "E4_heldout_lexicon": heldout,
                "modes": {m: {k: v for k, v in r.items() if k != "per_query"}
                          for m, r in results.items()}}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _bench_haystack_sensitivity(probes, kb_path, tmpdir, args, replay_table) -> dict:
    """**E1b.** How much of the probe arm's quality is INDEX COVERAGE rather than retrieval mechanism?

    `agentic_rag._entry_haystack` indexes a record's ``plan`` steps and ``tags``; ``knowledge.search`` (the
    ``keyword`` arm) indexes only command/text/lesson/problem/tags. Meanwhile `build_probes` defines T2
    relevance as ``|labels(e.plan) & gold_labels(q)| / |gold_labels(q)|`` and `build_fixture_kb` sets
    ``tags = {s['object'] for s in plan}``. So part of the headline gap is the arm under test searching a
    field the baseline cannot see, *and* being graded on that same field — not chunkless retrieval.

    This re-runs `probe` over the identical probe set with the plan steps, and then plan steps + tags,
    stripped from the haystack, and reports the delta. It is a DISCLOSURE, not a correction: the shipped
    arm really does index the plan, and indexing more of a record is a legitimate design choice. What it
    is not is evidence for chunkless probing per se, and the number below is how much to discount.
    """
    import agentic_rag as ar

    orig = ar._entry_haystack

    def _make(drop_plan=True, drop_tags=False):
        def _hay(entry):
            e = dict(entry or {})
            if drop_plan:
                e.pop("plan", None)
            if drop_tags:
                e.pop("tags", None)
            return orig(e)
        return _hay

    out = {}
    try:
        for label, fn in (("shipped", None),
                          ("no_plan", _make(True, False)),
                          ("no_plan_no_tags", _make(True, True))):
            if fn is not None:
                ar._entry_haystack = fn
            else:
                ar._entry_haystack = orig
            r = _run_mode("probe", probes, kb_path, tmpdir, args, replay_table)
            out[label] = {"overall": r["overall"], "by_lang": r["by_lang"]}
    except Exception as exc:                             # pragma: no cover
        out["error"] = repr(exc)
    finally:
        ar._entry_haystack = orig                        # ALWAYS restored

    print("#### E1b. index-coverage sensitivity: how much of `probe`'s gain survives when the haystack "
          "stops indexing the field the gold labels are derived from\n")
    print("| probe haystack | recall@1 all | recall@1 fa | nDCG@10 all | nDCG@10 fa |")
    print("|---|---|---|---|---|")
    for label in ("shipped", "no_plan", "no_plan_no_tags"):
        r = out.get(label)
        if not r:
            continue
        o, fa = r["overall"], (r["by_lang"].get("fa") or {})
        print(f"| {label} | {_fmt(o['recall@1'])} | {_fmt(fa.get('recall@1'))} | "
              f"{_fmt(o['nDCG@10'])} | {_fmt(fa.get('nDCG@10'))} |")
    print()
    return out


def _bench_heldout_lexicon(probes, kb_path, tmpdir, args, replay_table, results) -> dict:
    """**E4.** Does the Persian bridge generalise to vocabulary the lexicon has never seen?

    The whole cross-lingual result rests on `agentic_rag.FA_TYPE_LEXICON`, a Persian -> AI2-THOR-type
    table hand-authored FROM the evaluation set (see the DISCLOSURE on that constant). Ablating it drops
    FA recall@1 from 0.824 to 0.000 with 51/51 empty result sets, so no in-sample Persian number is a
    generalisation claim. This section takes the strict held-out measurement instead: for every query,
    delete from the lexicon exactly the surface forms THAT query uses, then retrieve.

    It also happens to be the only configuration in this file where the multi-hop loop is exercised at
    all. On the shipped lexicon the gate is satisfied at hop 0 for 102/102 queries, so `agentic` and
    `probe` are byte-identical and the loop -- the actual novelty over `probe` -- contributes exactly
    nothing measurable. With the query's own vocabulary withheld the lexical pivot returns no types, the
    deterministic gate fails, and the hop budget is drawn on. The pivot replayed on those hops is still
    the OFFLINE ORACLE (derived from each probe's gold types), so E4's `agentic` row is an **upper bound
    on a live model**, not a live result -- exactly like E1's. What it does establish is that the
    machinery runs, how many hops it costs, and how much of the held-out gap it can close at best.
    """
    out = {"probe": None, "agentic": None}
    try:
        for mode in ("probe", "agentic"):
            if mode in args.modes:
                out[mode] = _run_mode(mode, probes, kb_path, tmpdir, args, replay_table, heldout=True)
    except Exception as exc:                             # pragma: no cover
        out["error"] = repr(exc)

    print("#### E4. HELD-OUT Persian vocabulary (each query's own lexicon entries deleted before retrieval)\n")
    print("| --rag | lexicon | n | recall@1 all | recall@1 fa | recall@5 fa | mean hops | model calls/q |")
    print("|---|---|---|---|---|---|---|---|")
    for mode in ("probe", "agentic"):
        base = results.get(mode)
        if base and not base.get("skipped"):
            o, fa = base["overall"], (base["by_lang"].get("fa") or {})
            print(f"| {mode} | shipped (IN-SAMPLE) | {o['n']} | {_fmt(o['recall@1'])} | "
                  f"{_fmt(fa.get('recall@1'))} | {_fmt(fa.get('recall@5'))} | "
                  f"{_fmt(o.get('hops'), '.2f')} | {base['llm_calls_per_query']:.2f} |")
        r = out.get(mode)
        if r and not r.get("skipped"):
            o, fa = r["overall"], (r["by_lang"].get("fa") or {})
            print(f"| {mode} | HELD OUT | {o['n']} | {_fmt(o['recall@1'])} | "
                  f"{_fmt(fa.get('recall@1'))} | {_fmt(fa.get('recall@5'))} | "
                  f"{_fmt(o.get('hops'), '.2f')} | {r['llm_calls_per_query']:.2f} |")
    print()
    return {k: ({kk: vv for kk, vv in v.items() if kk != "per_query"} if isinstance(v, dict) else v)
            for k, v in out.items()}


def _fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def _print_quality(results):
    print("\n#### E1. retrieval quality (T1 twin gold, binary; T2 label-overlap grades for nDCG)\n")
    print("| --rag | n | recall@1 | recall@5 (95% CI) | recall@10 | MRR@10 | nDCG@10 (95% CI) | P@5 | "
          "scene prec | ties@1 (pool) | hits |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for mode, r in results.items():
        if r.get("skipped"):
            print(f"| {mode} | - | *skipped: {r.get('reason')}* | | | | | | | | |")
            continue
        o = r["overall"]
        print(f"| {mode} | {o['n']} | {_fmt(o['recall@1'])} | {_fmt(o['recall@5'])}{_ci(o, 'recall@5')} "
              f"| {_fmt(o['recall@10'])} | {_fmt(o['MRR@10'])} | {_fmt(o['nDCG@10'])}{_ci(o, 'nDCG@10')} | "
              f"{_fmt(o['precision@5'])} | {_fmt(o['scene_precision'])} | "
              f"{_fmt(o['ties@1'], '.2f')} ({_fmt(o.get('pool'), '.1f')}) | "
              f"{_fmt(o['n_returned'], '.1f')} |")
    print()


def _print_cost(results, args):
    tik_col = any((r.get("overall") or {}).get("tiktoken") is not None
                  for r in results.values() if not r.get("skipped"))
    print("#### E2. retrieval cost (latency is retrieval-path only; model time is priced separately)\n")
    head = ("| --rag | kb load ms | retriever ms | index bytes | cold ms | p50 ms | p95 ms | p99 ms | "
            "max ms | block chars | est tokens |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    if tik_col:
        head += " tiktoken |"
        sep += "---|"
    head += " model calls/q | +assumed model ms | cache hit |"
    sep += "---|---|---|"
    print(head)
    print(sep)
    for mode, r in results.items():
        if r.get("skipped"):
            print(f"| {mode} | *skipped* | | | | | | | | | |" + (" |" if tik_col else "") + " | | |")
            continue
        o, L = r["overall"], r["latency_ms"]
        row = (f"| {mode} | {r['kb_load_ms']:.2f} | {r['retriever_ms']:.2f} | {r['index_bytes']:,} | "
               f"{r['cold_ms']:.3f} | {_fmt(L['p50'])} | {_fmt(L['p95'])} | {_fmt(L['p99'])} | "
               f"{_fmt(L['max'])} | {_fmt(o['block_chars'], '.0f')} | {_fmt(o['est_tokens'], '.0f')} |")
        if tik_col:
            row += f" {_fmt(o['tiktoken'], '.0f')} |"
        row += (f" {r['llm_calls_per_query']:.2f} | {r['assumed_model_ms']:.0f} | "
                f"{_fmt(r['cache_hit_rate'], '.2f')} |")
        print(row)
    print()
    agentic_arms = [(m, r) for m, r in results.items()
                    if m in ("probe", "agentic") and not r.get("skipped")]
    if agentic_arms:
        print("#### E2b. the retrieval program's own decisions (from AgenticRAG.last_telemetry())\n")
        print("| --rag | mean hops | gate stopped at hop 0 | mean model calls/q |")
        print("|---|---|---|---|")
        for mode, r in agentic_arms:
            o = r["overall"]
            print(f"| {mode} | {_fmt(o['hops'], '.2f')} | {_fmt(o['gate_stopped_at_hop0'], '.2f')} | "
                  f"{r['llm_calls_per_query']:.2f} |")
        print()


def _print_by(results, key, label):
    print(f"#### E3. quality by {label}\n")
    print(f"| --rag | {label} | n | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 | est tokens |")
    print("|---|---|---|---|---|---|---|---|---|")
    for mode, r in results.items():
        if r.get("skipped"):
            continue
        for bucket, o in (r.get(key) or {}).items():
            print(f"| {mode} | {bucket} | {o['n']} | {_fmt(o['recall@1'])} | {_fmt(o['recall@5'])} | "
                  f"{_fmt(o['recall@10'])} | {_fmt(o['MRR@10'])} | {_fmt(o['nDCG@10'])} | "
                  f"{_fmt(o['est_tokens'], '.0f')} |")
    print()


def _ci(o, key):
    lo, hi = (o.get(key + "_ci") or [None, None])
    return "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"


def _print_headline(results, args, haystack=None, heldout=None):
    """The one comparison the section exists to make, spelled out so nobody has to eyeball the table.

    Every claim here is bounded by what was ACTUALLY measured in this run. Three things the previous
    version got wrong and this one states explicitly:
      * latency — `probe` is compared to the arm that ran, with the direction of the result stated even
        when it loses. The `hybrid` comparison is not made at all unless `hybrid` ran.
      * setup — the one-time `import chromadb` is reported separately from retriever construction, so
        "removes a 585 ms index tax" cannot be read off a module import.
      * quality — the Persian gap is labelled IN-SAMPLE and paired with the E4 held-out number.
    """
    kw, pb = results.get("keyword"), results.get("probe")
    if not kw or not pb or kw.get("skipped") or pb.get("skipped"):
        return
    kp50, pp50 = kw["latency_ms"]["p50"], pb["latency_ms"]["p50"]
    ratio = pp50 / max(1e-9, kp50)
    faster = "FASTER" if ratio < 1.0 else "SLOWER"
    print("**Headline.** (every number below is from THIS run; nothing is quoted from an arm that was skipped)")
    print(f"- retrieval-path latency p50: keyword {kp50:.4f} ms {_lat_ci(kw)} vs probe {pp50:.4f} ms "
          f"{_lat_ci(pb)} -- probe is {ratio:.2f}x, i.e. **{faster}** per query; p95 {kw['latency_ms']['p95']:.3f} "
          f"vs {pb['latency_ms']['p95']:.3f}.")
    print(f"- setup, split: keyword kb_load {kw['kb_load_ms']:.2f} ms + retriever {kw['retriever_ms']:.2f} ms "
          f"(+ a ONE-TIME {kw.get('module_import_ms', 0.0):.0f} ms `import chromadb` reported separately, which "
          f"is a site-packages cost, not an index build) vs probe kb_load {pb['kb_load_ms']:.2f} ms + retriever "
          f"{pb['retriever_ms']:.2f} ms. On-disk index {kw['index_bytes']:,} B vs {pb['index_bytes']:,} B.")
    hy = results.get("hybrid")
    if hy and not hy.get("skipped"):
        print(f"- vs hybrid: p50 {hy['latency_ms']['p50']:.4f} ms, setup {hy['retriever_ms']:.1f} ms, "
              f"index {hy['index_bytes']:,} B, recall@5 {hy['overall']['recall@5']:.3f}.")
    else:
        print("- vs hybrid: **NOT MEASURED** in this run (arm skipped, no Ollama). No speed or quality claim "
              "against `hybrid` may be made from this file.")
    print(f"- recall@5 overall: keyword {kw['overall']['recall@5']:.3f}{_ci(kw['overall'], 'recall@5')} vs probe "
          f"{pb['overall']['recall@5']:.3f}{_ci(pb['overall'], 'recall@5')}.  Persian half: "
          f"{kw['by_lang'].get('fa', {}).get('recall@5', 0.0):.3f} vs "
          f"{pb['by_lang'].get('fa', {}).get('recall@5', 0.0):.3f} -- **IN-SAMPLE**: that gap is produced by "
          f"FA_TYPE_LEXICON, a table hand-authored from this very dataset. See E4 for the held-out number.")
    ag = results.get("agentic")
    if ag and not ag.get("skipped"):
        d = ag["overall"]["recall@1"] - pb["overall"]["recall@1"]
        hops = ag["overall"].get("hops") or 0.0
        if hops <= 0.0:
            print("- agentic vs probe: the multi-hop loop spent ZERO hops and the two arms are numerically "
                  "identical. That is not a null result about the loop -- it is the loop never running. Do "
                  "not report a delta from this row.")
        else:
            print(f"- agentic vs probe: +{d:.3f} recall@1 for {hops:.2f} hops/query "
                  f"({ag['llm_calls_per_query']:.2f} model calls, priced at +{ag['assumed_model_ms']:.0f} ms). "
                  f"The pivot on those hops is the OFFLINE ORACLE, so this is the loop's CEILING, not a live "
                  f"result; with an n=102 probe set a delta below ~0.04 is inside the bootstrap CI either way.")
    ho = ((heldout or {}).get("probe") or {})
    hoa = ((heldout or {}).get("agentic") or {})
    if ho and hoa and not ho.get("skipped") and not hoa.get("skipped"):
        print(f"- the loop's clearest measured contribution is on HELD-OUT vocabulary (E4), which is the only "
              f"configuration where the gate actually fails at hop 0: recall@1 {ho['overall']['recall@1']:.3f} "
              f"(probe) -> {hoa['overall']['recall@1']:.3f} (agentic), FA half "
              f"{(ho['by_lang'].get('fa') or {}).get('recall@1', 0.0):.3f} -> "
              f"{(hoa['by_lang'].get('fa') or {}).get('recall@1', 0.0):.3f}, at "
              f"{hoa['overall'].get('hops') or 0.0:.2f} hops/query. Oracle-pivoted; still a ceiling.")
    if ho and not ho.get("skipped"):
        print(f"- held-out Persian vocabulary (E4): probe recall@1 falls "
              f"{pb['overall']['recall@1']:.3f} -> {ho['overall']['recall@1']:.3f} overall and "
              f"{pb['by_lang'].get('fa', {}).get('recall@1', 0.0):.3f} -> "
              f"{ho['by_lang'].get('fa', {}).get('recall@1', 0.0):.3f} on the FA half. THAT is the "
              f"generalisation figure; the in-sample one is not.")
    hs = (haystack or {})
    if hs.get("shipped") and hs.get("no_plan"):
        a, b = hs["shipped"]["overall"], hs["no_plan"]["overall"]
        fa_a = (hs["shipped"]["by_lang"].get("fa") or {})
        fa_b = (hs["no_plan"]["by_lang"].get("fa") or {})
        print(f"- index-coverage discount (E1b): dropping the plan steps from the probe haystack -- the field "
              f"the T2 gold labels are derived from and the keyword baseline cannot see -- moves recall@1 "
              f"{a['recall@1']:.3f} -> {b['recall@1']:.3f} (FA {fa_a.get('recall@1', 0.0):.3f} -> "
              f"{fa_b.get('recall@1', 0.0):.3f}). Subtract that before calling the gap a mechanism effect.")
    print(f"- injected memory: keyword {kw['overall']['est_tokens']:.0f} est tokens vs probe "
          f"{pb['overall']['est_tokens']:.0f}"
          + (f" vs full {results['full']['overall']['est_tokens']:.0f} (the stuff-everything control)."
             if results.get("full") and not results["full"].get("skipped") else "."))
    print()


def _lat_ci(r):
    lo, hi = (r["latency_ms"].get("p50_ci") or [None, None])
    return "" if lo is None else f"(95% CI {lo:.4f}-{hi:.4f})"


def _print_notes(results, ollama, args, haystack=None, heldout=None):
    print("**Reading this table. Five caveats, all load-bearing.**")
    print("1. `keyword` recall on the FA half is the *analytically known* 0.00: a Persian query and an "
          "English-authored record share no tokens. Any FA number above zero is the pivot working -- and "
          "the pivot is a LOOKUP TABLE hand-authored from this dataset (agentic_rag.FA_TYPE_LEXICON). "
          "Every in-sample FA number in E1/E3 is therefore fitted; section E4 is the held-out one.")
    print("2. `full` is not a retrieval arm. It dumps every in-scope record, so its rank columns are an "
          "artefact of insertion order; its honest column is `est tokens`, and that is the CAG "
          "upper-bound control the retrieval axis is measured against.")
    print("3. `agentic`'s pivot is REPLAYED offline (see the header line). Its quality column is therefore "
          "an upper bound on the live arm, and its true cost is `p50 ms` PLUS `+assumed model ms`, which "
          f"is an assumption ({args.llm_ms:.0f} ms/call), never a measurement.")
    print("4. `ties@1` is measured over the arm's FULL candidate pool (k = |KB|, retrieved outside the "
          "timing loop), not over the k hits that survived truncation -- otherwise the column would just "
          "be reporting how many hits came back.")
    print("5. Latency excludes the planner call entirely; it is the retrieval path only, on the host "
          "recorded in the results JSON's `host` field.")
    ag, pb = results.get("agentic"), results.get("probe")
    if ag and not ag.get("skipped") and (ag["overall"].get("hops") or 0.0) == 0.0:
        print("- `agentic` spent ZERO hops on every query here, so on the SHIPPED lexicon it is numerically "
              "identical to `probe` and this section measures NOTHING about the multi-hop loop. Two honest "
              "readings, both required: (a) the gate works -- the lexical FA->EN pivot had already covered "
              "every OBSERVATION-present object type, so the budget was never drawn on and the extra "
              "machinery is free when unneeded; (b) the loop is consequently UNTESTED end to end on this "
              "fixture, and an `agentic` vs `probe` delta on it is structurally 0, not a null result. "
              "Section E4 is where the gate actually fails and the hops fire.")
    hb = ((heldout or {}).get("agentic") or {})
    hp = ((heldout or {}).get("probe") or {})
    if hb and not hb.get("skipped"):
        print(f"- E4 is where the loop runs: with each query's own vocabulary withheld, `agentic` spends "
              f"{hb['overall'].get('hops') or 0.0:.2f} hops/query ({hb['llm_calls_per_query']:.2f} model "
              f"calls) and reaches recall@1 {hb['overall']['recall@1']:.3f} against `probe`'s "
              f"{(hp.get('overall') or {}).get('recall@1', 0.0):.3f} on the same held-out setting. Both are "
              f"ORACLE-pivoted, so this is the loop's CEILING, not its live value.")
    if not ollama:
        print(f"- `hybrid` was SKIPPED: no Ollama at {args.endpoint}. The dense half cannot be measured "
              f"offline and must not be reported as if it had been. In particular no 'faster than hybrid' "
              f"claim is supported by this run.")
    print()


# ================================================================================================
# MAIN
# ================================================================================================
def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="CI-ish run: skip the 100k-entry / 100k-vector rows and use 1 timing rep")
    ap.add_argument("--section", default="all",
                    help="A|B|C|D|E|F or a combination like 'AE' (default: all = A-E; F runs only when named "
                         "and writes its own --orth-out file)")
    ap.add_argument("--dataset", default=DATASET, help="bilingual dataset (READ-ONLY)")
    ap.add_argument("--kb-lang", default="en", choices=("en", "fa"),
                    help="which half of the dataset seeds the fixture knowledge base")
    ap.add_argument("--query-langs", default="en,fa", help="comma-separated query languages for the probe set")
    ap.add_argument("--scenes", default="", help="comma-separated FloorPlan filter (default: all three)")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated --rag arms to bench")
    ap.add_argument("--k", type=int, default=10, help="retrieval depth (>= 10 so recall@10 is meaningful)")
    ap.add_argument("--reps", type=int, default=3, help="timing repetitions of the whole probe set")
    ap.add_argument("--limit", type=int, default=0, help="cap the probe set (0 = all)")
    ap.add_argument("--rank", default="provenance", choices=("plain", "provenance", "provenance+utility"))
    ap.add_argument("--agentic-hops", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--budget", type=int, default=6000, help="memory-block character budget")
    ap.add_argument("--cache", action="store_true",
                    help="enable the probe cache (OFF by default: timing runs must not measure a dict lookup)")
    ap.add_argument("--llm-ms", type=float, default=1200.0,
                    help="ASSUMED wall-clock per model call, used to price the agentic arm's hops")
    ap.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint (hybrid arm only)")
    ap.add_argument("--pivot-replay", default="", help="JSON {qid: {types, verbs}} replayed for --rag agentic")
    ap.add_argument("--emit-probes", default="", help="write the frozen probe set to this JSONL path")
    ap.add_argument("--out", default=RESULTS_PATH, help="machine-readable results path")
    ap.add_argument("--orth-out", default=ORTH_RESULTS_PATH,
                    help="section F's results path (never the A-E results file)")
    ap.add_argument("--no-write", action="store_true", help="do not write the results JSON(s)")
    args = ap.parse_args(argv)

    args.query_langs = [s for s in args.query_langs.split(",") if s]
    args.scenes = [s for s in args.scenes.split(",") if s]
    args.modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    if args.quick:
        args.reps = 1
    # `all` stays A-E so a full run is unchanged; F is opt-in and has its own output file.
    sections = set("ABCDE") if args.section.lower() == "all" else set(args.section.upper()) & set("ABCDEF")
    main_sections = sections & set("ABCDE")

    sizes = [100, 1_000, 10_000] + ([] if args.quick else [100_000])
    dense = [(100, 768), (1_000, 768), (10_000, 768), (10_000, 256), (10_000, 64)] + \
            ([] if args.quick else [(100_000, 768)])

    _host = host_info()
    print(f"# bench_rag -- python {sys.version.split()[0]}, sqlite {sqlite3.sqlite_version}, "
          f"numpy {'present' if _has_numpy() else 'absent'}, "
          f"tiktoken {'present' if _tiktoken_counter() else 'absent'}, "
          f"sections {''.join(sorted(sections))}{' (quick)' if args.quick else ''}")
    print(f"# host -- {_host['platform']} | {_host['processor']} | {_host['cpu_count']} logical CPUs | "
          f"chromadb {'installed' if _host['chromadb_importable'] else 'absent'}")
    print("# latency figures are HOST-SPECIFIC and must be quoted with this line; CIs are a 2000-rep "
          "percentile bootstrap, seed 20260816\n")

    doc = {"ts": time.time(), "python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
           "numpy": _has_numpy(), "quick": bool(args.quick), "sections": sorted(main_sections),
           "argv": sys.argv[1:], "host": _host,
           "ci": {"method": "percentile bootstrap", "reps": 2000, "level": 0.95, "seed": 20260816}}
    if "A" in sections:
        doc["A_scaling"] = bench_scaling(sizes)
    if "B" in sections:
        doc["B_persian"] = bench_persian(cross_orthography=True) + bench_persian(cross_orthography=False)
    if "C" in sections:
        doc["C_dense"] = bench_dense(dense)
    if "D" in sections:
        doc["D_sidecar"] = bench_sidecar([(100, 768), (500, 768), (2_000, 768), (500, 256)])
    if "E" in sections:
        doc["E_retrieval"] = bench_section_e(args)
    if "F" in sections:
        orth = bench_orth_dataset(args.dataset)
        orth["argv"] = sys.argv[1:]
        if not args.no_write:
            try:
                tmp = args.orth_out + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(orth, fh, ensure_ascii=False, indent=2)
                os.replace(tmp, args.orth_out)
                print(f"section F results -> {args.orth_out}")
            except Exception as exc:
                print(f"WARNING: could not write {args.orth_out}: {exc!r}")

    print("note: no live Ollama round-trip is included in any latency column above -- the embedding call "
          "is network/model bound and must be measured separately against a running server "
          "(see docs/research/rag-fast-retrieval.md).")
    if not main_sections:
        return 0                                       # F alone must never (re)write the A-E results file
    if not args.no_write:
        try:
            tmp = args.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, args.out)
            print(f"machine-readable results -> {args.out}")
        except Exception as exc:
            print(f"WARNING: could not write {args.out}: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
