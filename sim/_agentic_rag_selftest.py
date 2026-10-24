"""Pure Phase-10 gate for the chunkless agentic retriever (sim/agentic_rag.py).

No AI2-THOR, no Ollama, no network, no numpy, no live store. Runs on a bare Windows box with the
stock `python` interpreter: numpy / chromadb / ai2thor are POISONED in `sys.modules` before the
module under test is imported (the `_phase7_selftest.py` stubbing pattern), every fixture lives in a
`tempfile.TemporaryDirectory()`, and the repo's own `knowledge_base.json` / `memory.db` /
`dataset/commands.jsonl` are hashed before and after to prove nothing was touched.

Check groups (docs/agentic-rag-design.md section 13, plus one group the design implies but does not
enumerate):

  A  import purity           -- agentic_rag imports with numpy/chromadb/ai2thor absent and drags in
                                NOTHING from the repo; `import rag` still works; the section-5 public
                                API exists with the documented parameter names.
  B  bilingual normalisation -- FA/AR yeh + kaf folding, harakat, ZWNJ, tatweel, Persian/Arabic-Indic
                                digits, bilingual stopwords, byte-compat with knowledge._tokens.
  C  manifest (level 1)      -- exact line format, truncation, tag suffix, scope filter, budget drops
                                the LOWEST-ranked rows, the '... (+N more)' marker.
  D  probe (level 2)         -- whole-word matching, the Persian lookaround path, cross-orthography
                                matching on BOTH sides, plan-field haystack, deterministic order.
  E  progressive disclosure  -- probes return lines not bodies, kb_read is byte-identical to eval's
                                template, the budget is a hard bound, the manifest is cheaper than
                                stuffing every body.
  F  provenance ranking      -- plain is byte-reproducible; source / scope / confidence-decay move the
                                score in the claimed direction; utility saturates; supersession.
  G  typed quota             -- lessons are never crowded out; backfill; no id twice.
  H  the bilingual pivot     -- the FA->EN object-type bridge with ZERO model calls; `auto` spends a
                                call only on a Persian command the lexicon cannot bridge.
  I  gate + hop bound        -- the deterministic sufficiency gate, and a HARD hop cap on an
                                adversarial query that can never satisfy it.
  J  probe cache             -- exact invalidation on the KB content hash; an exact cache can never
                                change what is retrieved.
  K  plan verifier           -- unknown_action / unknown_object, alias-insensitive, never raises.
  L  degradation             -- provider absent / provider raising / knowledge base raising: results,
                                a logged warning, no exception, and the never-worse-than-keyword floor.
  M  hit-shape + eval        -- the HybridRAG hit-dict contract and the eval.py memory-block contract.
  N  data safety             -- repo stores byte-identical, nothing written outside the temp dir, and
                                no socket is ever opened.
  O  ablation switches       -- every --rag / --rank / --agentic-* switch genuinely changes the
                                output. A switch that is a no-op is a bug, not a feature.

Checks that depend on Phase-10 edits to `sim/eval.py` (owned by another agent in this pass) are
FEATURE-DETECTED and reported as SKIP rather than FAIL when those edits are not on disk yet.

Run it:  python sim/_agentic_rag_selftest.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATASET = os.path.join(HERE, "dataset", "commands.jsonl")

# ------------------------------------------------------------------------------------------------
# TRACING OFF, BEFORE ANY SIBLING IMPORT. `sim/knowledge.py` carries a tracing hook into `sim/trace.py`
# (both owned by the observability workstream), so merely BUILDING the fixtures below used to append
# ~270 KB to the live sim/trace.db on every run -- while N1/N2 reported PASS, because N1's guarded list
# did not name trace.db and N2 only diffed the SET OF PATHS, which a pre-existing store is invisible to.
# Run the same suite in a clean copy where trace.db does not exist yet and N2 failed with
# ['.../trace.db', '-shm', '-wal']. The emit itself is not ours to change; the ISOLATION is this file's
# job, and `TRACE_ENABLED=0` is trace.py's documented opt-out (its module docstring: "creates no file").
# TRACE_DB is additionally pointed at a scratch path so that even a future code path that ignores the
# enable flag cannot land in the repo store. Both are set before `import knowledge` anywhere below.
# Regression-tested by N1/N2, which now compare SIZE AND MTIME, not just the set of paths.
# ------------------------------------------------------------------------------------------------
_TRACE_SCRATCH = os.path.join(tempfile.gettempdir(), "agentic_rag_selftest_trace.db")
os.environ["TRACE_ENABLED"] = "0"
os.environ["TRACE_DB"] = _TRACE_SCRATCH

# ------------------------------------------------------------------------------------------------
# PASS / FAIL / SKIP recorder (same shape as sim/_eval_selftest.py).
# ------------------------------------------------------------------------------------------------
_RESULTS = []          # (name, status) with status in {"PASS","FAIL","SKIP"}


def _record(name, status, note=""):
    _RESULTS.append((name, status))
    line = "  [%-4s] %s" % (status, name)
    if note:
        line += "  (%s)" % note
    print(line)


def check(name, cond, note=""):
    ok = bool(cond)
    _record(name, "PASS" if ok else "FAIL", "" if ok else note)
    return ok


def skip(name, note):
    _record(name, "SKIP", note)


#: Why the dataset-backed checks have no records. Set once in main(). The old note said "dataset absent",
#: which sent a reader to sim/dataset/commands.jsonl -- a file that is present and fine; the real cause is
#: almost always that `sim/eval.py` failed to import, because `ev.load_dataset` is what reads it.
_NO_RECORDS_REASON = ["dataset records unavailable"]


def _no_records_reason():
    return _NO_RECORDS_REASON[0]


def approx(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


class _WarnTrap(logging.Handler):
    """Captures agentic_rag's degradation warnings so 'logs a warning' can be ASSERTED, not assumed."""

    def __init__(self):
        logging.Handler.__init__(self, level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def reset(self):
        self.records = []


_TRAP = _WarnTrap()


# ================================================================================================
# GROUP A -- import purity and module hygiene (4)
# ================================================================================================
_POISON = ("numpy", "chromadb", "ai2thor", "ai2thor.controller", "ai2thor.platform")


def group_a():
    """Poison the optional/heavy deps, then import the module under test. MUST run before anything
    else imports a repo module, or A2's "nothing from the repo was dragged in" claim is vacuous."""
    for name in _POISON:
        sys.modules[name] = None            # the None sentinel makes `import <name>` raise ImportError

    ar = None
    try:
        import agentic_rag as ar            # noqa: F401
        check("A1 import agentic_rag with numpy/chromadb/ai2thor poisoned", True)
    except Exception as exc:
        check("A1 import agentic_rag with numpy/chromadb/ai2thor poisoned", False, repr(exc))
        return None

    leaked = [n for n in ("knowledge", "rag", "memory", "improve") if n in sys.modules]
    loaded = [n for n in ("numpy", "chromadb") if sys.modules.get(n) is not None]
    check("A2 import pulls in no knowledge/rag/memory/improve/numpy/chromadb",
          not leaked and not loaded, f"leaked={leaked} loaded={loaded}")

    try:
        import rag as _rag                  # noqa: F401
        cycle_free = getattr(_rag, "HybridRAG", None) is not None
        check("A3 `import rag` still succeeds under the same poisoning (no import cycle)", cycle_free)
    except Exception as exc:
        check("A3 `import rag` still succeeds under the same poisoning (no import cycle)", False, repr(exc))

    # A4: every public name of the design's section 5, with the documented leading parameter names.
    wanted_names = [
        "FA_FOLD", "HARAKAT", "_FOLD_TABLE", "STOP_EN", "STOP_FA", "STOP_ALL", "PROVENANCE",
        "HALF_LIFE_KB", "RANK_WEIGHTS", "STALE_PENALTY", "MANIFEST_TITLE_CHARS", "DEFAULT_BUDGET_CHARS",
        "DEFAULT_BODY_CHARS", "FULL_BUDGET_CHARS", "MAX_HOPS", "KINDS", "PIVOT_SYSTEM",
        "FA_TYPE_LEXICON", "EN_TYPE_LEXICON", "FA_VERB_LEXICON", "EN_VERB_LEXICON", "LEVEL_POLICY",
        "normalize_text", "tokens", "is_persian", "kb_entries", "manifest_line", "kb_manifest",
        "kb_grep", "kb_read", "kb_fingerprint", "utility", "rank_key", "rank_hits", "quota_select",
        "lexical_pivot", "llm_pivot", "pivot", "coverage", "sufficient", "verify_plan", "ProbeCache",
        "estimate_level", "compact", "kb_read_trim", "est_tokens", "keyword_fallback", "AgenticRAG",
        "level_budget", "held_out_type_pairs", "_L2_INSTRUMENT_MARKERS",
    ]
    wanted_sig = {
        "normalize_text": ["text"],
        "tokens": ["text", "stop"],
        "is_persian": ["text"],
        "kb_entries": ["kb", "scene", "kinds"],
        "manifest_line": ["kind", "entry", "width"],
        "kb_manifest": ["kb", "scene", "kinds", "limit", "budget_chars", "rank_mode", "now"],
        "kb_grep": ["kb", "terms", "scene", "kinds", "limit", "whole_word"],
        "kb_read": ["kb", "ids", "budget_chars", "render"],
        "kb_fingerprint": ["kb"],
        "utility": ["entry", "util_sidecar"],
        "rank_key": ["kind", "entry", "scene", "sim", "now", "util_sidecar", "superseded", "weights"],
        "rank_hits": ["hits", "scene", "now", "mode", "util_sidecar", "kb", "weights"],
        "quota_select": ["hits", "k", "quota", "mode"],
        "lexical_pivot": ["command", "scene_types", "alias_map", "type_pairs"],
        "llm_pivot": ["command", "provider", "model", "manifest_text", "observation", "timeout", "options"],
        "pivot": ["command", "provider", "model", "scene_types", "mode", "manifest_text", "observation",
                  "timeout", "type_pairs"],
        "coverage": ["hits", "terms", "scene", "observation"],
        "sufficient": ["hits", "terms", "scene", "observation", "min_hits"],
        "verify_plan": ["plan", "observation_types", "action_names"],
        "estimate_level": ["command", "piv", "type_pairs"],
        "level_budget": ["level", "hops", "k", "budget_chars"],
        "held_out_type_pairs": ["surface", "drop_keys", "base"],
        "compact": ["hits", "budget_chars", "render"],
        "kb_read_trim": ["hits", "budget_chars"],
        "est_tokens": ["text"],
        "keyword_fallback": ["kb", "query", "scene", "k"],
    }
    missing = [n for n in wanted_names if not hasattr(ar, n)]
    bad_sig = []
    for fn_name, params in wanted_sig.items():
        fn = getattr(ar, fn_name, None)
        if fn is None:
            continue
        got = list(inspect.signature(fn).parameters)
        if got != params:
            bad_sig.append(f"{fn_name}{tuple(got)}")
    for meth, params in (("search", ["self", "query", "scene", "k", "alpha", "level", "observation",
                                     "model", "scene_types"]),
                         ("dump", ["self", "scene", "budget_chars"]),
                         ("recall_text", ["self", "query", "scene", "k"])):
        got = list(inspect.signature(getattr(ar.AgenticRAG, meth)).parameters)
        if got != params:
            bad_sig.append(f"AgenticRAG.{meth}{tuple(got)}")
    check("A4 every section-5 public name exists with the documented parameter names",
          not missing and not bad_sig, f"missing={missing} bad={bad_sig}")

    logging.getLogger("agentic_rag").addHandler(_TRAP)
    return ar


# ================================================================================================
# Fixtures (built ONCE, in a temp dir; the repo's own stores are never opened for writing).
# ================================================================================================
_FA_CMD = "سیب را در یخچال بگذار"                # "put the apple in the fridge", pure Persian
_FA_LESSON_CMD = "نان را برش بزن"                # "slice the bread"
_LONG_TITLE = ("Put the apple on the counter top and then close the fridge door before you "
               "turn on the light switch")


def build_fixture(ar, knowledge, tmp, now):
    """The hand-built bilingual KB used by groups B-K, M and O. Deterministic; `now` is pinned."""
    kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "fixture_kb.json"))
    ids = {}
    ids["apple"] = kb.add_procedure(
        "put the apple in the fridge",
        [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}],
        why="fixture", scene="FloorPlan1", tags=["apple", "fridge"],
        source="environment", ts=now, confidence=0.9)["id"]
    ids["knife"] = kb.add_lesson(
        "slice the bread", "SliceObject failed: not holding a knife",
        "pick up a Knife before slicing", why="fixture", scene="FloorPlan1",
        source="human", ts=now, confidence=0.8)["id"]
    ids["faucet"] = kb.add_fact(
        "Running the faucet does not clean objects; use CleanObject.",
        why="fixture", tags=["sink", "clean"], source="prior",
        ts=now - 14 * 24 * 3600.0, confidence=1.0)["id"]
    ids["fa_proc"] = kb.add_procedure(
        _FA_CMD, [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}],
        why="fixture", scene="FloorPlan2", tags=["سیب", "یخچال"],
        source="demo", ts=now, confidence=0.7)["id"]
    ids["fa_lesson"] = kb.add_lesson(
        _FA_LESSON_CMD, "برش خوردن نان ناموفق بود", "اول چاقو را بردار",
        why="fixture", scene="FloorPlan2", source="environment", ts=now, confidence=0.6)["id"]
    ids["notags"] = kb.add_fact(
        "The LightSwitch is next to the kitchen door.", why="fixture",
        tags=None, source="prior", ts=now, confidence=1.0)["id"]
    ids["long"] = kb.add_procedure(
        _LONG_TITLE, [{"action": "pick_and_place", "object": "Apple", "to": "CounterTop"}],
        why="fixture", scene="FloorPlan3", tags=["apple", "countertop"],
        source="environment", ts=now, confidence=0.5)["id"]
    return kb, ids


_GOAL2ACT = {("isOpen", True): "open", ("isOpen", False): "close",
             ("isPickedUp", True): "pickup", ("isToggled", True): "toggle_on",
             ("isToggled", False): "toggle_off", ("temperature", "Hot"): "heat_object",
             ("temperature", "Cold"): "cool_object", ("isSliced", True): "slice",
             ("isCooked", True): "cook_food"}


def gold_plan(rec) -> list:
    """The plan stored for a dataset record, derived from its OWN gold (no LLM, no simulator).

    Shared verbatim with `sim/bench_rag.py` so the benchmark's fixture and this gate's fixture are the
    same corpus (design section 12.1).
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


def build_dataset_kb(knowledge, records, tmp, now, lang="en"):
    """~110-entry KB seeded from the `lang` half of the dataset + skill docs + kitchen priors.

    Returns (kb, twin) where twin maps a dataset record id -> the KB entry id seeded from it.
    """
    kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "dataset_kb.json"))
    twin = {}
    half = [r for r in records if r.lang == lang]
    for i, r in enumerate(half):
        plan = gold_plan(r)
        e = kb.add_procedure(r.command, plan, why="fixture", scene=r.scene,
                             tags=sorted({s["object"] for s in plan}),
                             source="environment", ts=now, confidence=0.9)
        e["uses"] = 1 + (i % 3)                       # deterministic spread so `uses` is not constant
        twin[r.id] = e["id"]
        if i % 4 == 0 and plan:
            kb.add_lesson(r.command,
                          f"{plan[0]['action']} {plan[0]['object']} failed: precondition unmet",
                          f"make sure the {plan[0]['object']} is reachable first",
                          why="fixture", scene=r.scene, source="environment", ts=now, confidence=0.6)
    try:
        import skills
        for doc in skills.skill_docs():
            kb.add_fact(doc, why="fixture", source="prior", ts=now, confidence=1.0)
    except Exception:
        pass
    try:
        import seed_memory
        for text, tags in seed_memory.KITCHEN_FACTS:
            kb.add_fact(text, why="fixture", tags=list(tags), source="prior", ts=now, confidence=1.0)
    except Exception:
        pass
    return kb, twin


class StubProvider:
    """An `llm.LLMProvider`-shaped stub. Counts calls; replies with a canned string, or raises."""

    def __init__(self, reply='{"types": ["Microwave"], "verbs": ["heat"], "ids": []}', boom=None):
        self.calls = 0
        self.reply = reply
        self.boom = boom

    def chat(self, model, messages, **kw):
        self.calls += 1
        if self.boom is not None:
            raise self.boom
        return self.reply


# ================================================================================================
# GROUP B -- bilingual normalisation (10)
# ================================================================================================
_MIXED = [
    "سیب را در یخچال بگذار", "سيب را در يخچال بگذار", "سَیب", "گوجه‌فرنگی", "گوجه فرنگی",
    "مـیـز", "۱۲۳", "١٢٣", "پاک کن", "پاك كن", "Put the apple in the fridge",
    "Slice the BREAD with a knife!", "نان را با چاقو برش بزن", "تخم‌مرغ", "کاهو را بشوی",
    "چراغ را خاموش کن", "open the microwave", "  spaced   out  ", "ماکروویو را باز کن", "",
]


def group_b(ar, knowledge, records):
    nt = ar.normalize_text
    check("B1 normalize_text is idempotent over 20 mixed FA/EN strings",
          all(nt(nt(s)) == nt(s) for s in _MIXED))
    check("B2 Persian yeh and Arabic yeh normalise equal (سیب == سيب)",
          nt("سیب") == nt("سيب") and nt("سیب") != "")
    check("B3 keheh and Arabic kaf normalise equal (پاک == پاك)", nt("پاک") == nt("پاك"))
    check("B4 harakat is STRIPPED, not treated as a word boundary (سَیب -> سیب)",
          nt("سَیب") == nt("سیب") and ar.tokens("سَیب") == ar.tokens("سیب"))
    check("B5 ZWNJ folds to a space (گوجه‌فرنگی tokenizes like گوجه فرنگی)",
          ar.tokens("گوجه‌فرنگی") == ar.tokens("گوجه فرنگی") and len(ar.tokens("گوجه‌فرنگی")) == 2)
    check("B6 tatweel is removed (مـیـز -> میز)", nt("مـیـز") == "میز")
    check("B7 Persian ۱۲۳ and Arabic-Indic ١٢٣ fold to ASCII 123",
          nt("۱۲۳") == "123" and nt("١٢٣") == "123")
    toks = ar.tokens(_FA_CMD)
    check("B8 STOP_FA fires: tokens('سیب را در یخچال بگذار') drops را and در",
          "را" not in toks and "در" not in toks and "سیب" in toks and "یخچال" in toks, str(sorted(toks)))
    ascii_strings = ["put the apple in the fridge", "Slice the BREAD with a knife!", "open the microwave",
                     "turn on the toaster", "a an the of", "x1 y2 z3", "clean the plate",
                     "pick up the mug", "close the cabinet drawer", "wash lettuce",
                     "PUT THE EGG IN THE PAN", "heat the potato", "cool the apple down",
                     "throw the bottle in the trash", "serve me breakfast", "make a cold snack",
                     "toast the bread", "fill the mug with coffee", "empty the garbage can", ""]
    bad = [s for s in ascii_strings if ar.tokens(s, stop=ar.STOP_EN) != knowledge._tokens(s)]
    check("B9 tokens(x, stop=STOP_EN) == knowledge._tokens(x) on 20 ASCII strings", not bad, str(bad[:2]))
    if not records:
        skip("B10 is_persian True for every FA command and False for every EN command", _no_records_reason())
    else:
        wrong = [r.id for r in records if ar.is_persian(r.command) != (r.lang == "fa")]
        check("B10 is_persian True for every FA command and False for every EN command (n=%d)" % len(records),
              not wrong, str(wrong[:3]))


# ================================================================================================
# GROUP C -- manifest (6)
# ================================================================================================
def group_c(ar, kb, ids, now):
    by_id = {e.get("id"): (kind, e) for kind, e in ar.kb_entries(kb)}
    kp, ep = by_id[ids["apple"]]
    kl, el = by_id[ids["knife"]]
    kf, ef = by_id[ids["faucet"]]
    exact = (
        ar.manifest_line(kp, ep) == f"#{ids['apple']} P [FloorPlan1] put the apple in the fridge | apple,fridge"
        and ar.manifest_line(kl, el) == f"#{ids['knife']} L [FloorPlan1] slice the bread"
        and ar.manifest_line(kf, ef) == (f"#{ids['faucet']} F [global] Running the faucet does not clean "
                                         f"objects; use CleanObject. | sink,clean"))
    check("C1 manifest_line matches '#{id} {k} [{scope}] {title} | {tags}' for P / L / F", exact,
          repr(ar.manifest_line(kp, ep)))
    kL, eL = by_id[ids["long"]]
    long_line = ar.manifest_line(kL, eL)
    check("C2 a title longer than `width` is truncated with a trailing '...'; a short one is not",
          long_line.endswith("...") is False and "..." in long_line
          and _LONG_TITLE[:ar.MANIFEST_TITLE_CHARS] + "..." in long_line
          and "..." not in ar.manifest_line(kl, el), repr(long_line))
    kn, en_ = by_id[ids["notags"]]
    check("C3 the ' | tags' suffix is absent when the record has no tags",
          " | " not in ar.manifest_line(kn, en_), repr(ar.manifest_line(kn, en_)))
    _t2, rows2 = ar.kb_manifest(kb, scene="FloorPlan2", now=now)
    got = {r["id"] for r in rows2}
    check("C4 kb_manifest(scene='FloorPlan2') excludes FloorPlan1 rows and keeps global ones",
          ids["apple"] not in got and ids["fa_proc"] in got and ids["faucet"] in got, str(sorted(got)))
    full_text, full_rows = ar.kb_manifest(kb, now=now)
    ranked_ids = [r["id"] for r in full_rows]
    budget = max(len(full_rows[0]["line"]), len(full_text) // 2)
    text, rows = ar.kb_manifest(kb, budget_chars=budget, now=now)
    body = [ln for ln in text.split("\n") if not ln.startswith("... (+")]
    dropped_ids = ranked_ids[len(rows):]
    check("C5 a tight budget keeps rows == body lines and drops the LOWEST-ranked records first",
          len(rows) == len(body) and 0 < len(rows) < len(full_rows)
          and [r["id"] for r in rows] == ranked_ids[:len(rows)]
          and set(dropped_ids) == set(ranked_ids) - {r["id"] for r in rows},
          f"rows={len(rows)} body={len(body)} of {len(full_rows)}")
    markers = [ln for ln in text.split("\n") if ln.startswith("... (+")]
    check("C6 the '... (+N more records not shown)' marker appears once and reports the correct N",
          len(markers) == 1
          and markers[0] == f"... (+{len(full_rows) - len(rows)} more records not shown)", str(markers))


# ================================================================================================
# GROUP D -- probe (7)
# ================================================================================================
def group_d(ar, kb, ids):
    hits = ar.kb_grep(kb, "apple")
    got = {h["entry"]["id"] for h in hits}
    check("D1 kb_grep('apple') finds the apple procedure and not the knife lesson",
          ids["apple"] in got and ids["knife"] not in got, str(sorted(got)))
    check("D2 whole-word matching: 'app' does not match a record containing only 'apple'",
          ar.kb_grep(kb, "app") == [])
    fa_hits = ar.kb_grep(kb, sorted(ar.tokens(_FA_CMD)))
    fa_ids = {h["entry"]["id"] for h in fa_hits}
    shared_ascii = ar.tokens(_FA_CMD, stop=ar.STOP_EN) & ar.tokens("put the apple in the fridge")
    check("D3 a pure-Persian query retrieves the Persian record (Unicode lookaround path, 0 shared ASCII tokens)",
          ids["fa_proc"] in fa_ids and not shared_ascii and not any(c.isascii() and c.isalpha()
                                                                    for c in _FA_CMD),
          f"ids={sorted(fa_ids)} shared_ascii={shared_ascii}")
    check("D4 an Arabic-yeh query matches a Persian-yeh record (normalisation on BOTH sides)",
          {h["entry"]["id"] for h in ar.kb_grep(kb, "سيب")} == {ids["fa_proc"]})
    multi = ar.kb_grep(kb, ["apple", "fridge", "knife", "clean"])
    order = [(h["n_match"], h["entry"]["id"]) for h in multi]
    expected = sorted(order, key=lambda t: (-t[0], t[1]))
    stable = all([(h["n_match"], h["entry"]["id"])
                  for h in ar.kb_grep(kb, ["apple", "fridge", "knife", "clean"])] == order
                 for _ in range(100))
    check("D5 hits sort by n_match desc then id asc, and the order is stable over 100 repeats",
          order == expected and stable, str(order))
    plan_hits = {h["entry"]["id"] for h in ar.kb_grep(kb, "pick_and_place")}
    cmd = next(e for _k, e in ar.kb_entries(kb) if e["id"] == ids["apple"])["command"]
    check("D6 the `plan` field is part of the haystack (a command that never says 'pick_and_place')",
          ids["apple"] in plan_hits and "pick_and_place" not in cmd, str(sorted(plan_hits)))
    check("D7 `limit` truncates, and asking for more than exist never raises",
          len(ar.kb_grep(kb, ["apple", "fridge", "knife", "clean", "light"], limit=1)) == 1
          and len(ar.kb_grep(kb, "apple", limit=999)) >= 1
          and ar.kb_grep(kb, "zzzznotpresent", limit=999) == [])


# ================================================================================================
# GROUP E -- progressive disclosure and budget (5)
# ================================================================================================
def group_e(ar, ev, kb, ids):
    hits = ar.kb_grep(kb, "apple")
    check("E1 kb_grep returns LINES, never bodies ('body' not in hit)",
          all("body" not in h for h in hits) and all("line" in h for h in hits))

    read = ar.kb_read(kb, [ids["apple"]])
    hit = {"kind": read[0]["kind"], "entry": read[0]["entry"]}
    if ev is None:
        skip("E2 kb_read body is byte-identical to eval's memory-line template", "sim/eval.py unimportable")
    else:
        renderer = getattr(ev, "render_hit_body", None)
        if callable(renderer):
            ref = renderer(hit["kind"], hit["entry"])
            src = "eval.render_hit_body"
        else:
            ref = ev._render_hits([hit]).splitlines()[1]
            src = "eval._render_hits line"
        check(f"E2 kb_read body is byte-identical to {src}", read[0]["body"] == ref,
              f"{read[0]['body']!r} != {ref!r}")

    all_ids = [e.get("id") for _k, e in ar.kb_entries(kb)]
    bodies = [ar._default_render(k, e) for k, e in ar.kb_entries(kb)]
    budget = len(bodies[0]) + len(bodies[1]) - 1              # room for one body, not two
    got = ar.kb_read(kb, all_ids, budget_chars=budget)
    total = sum(len(g["body"]) for g in got)
    partial = [g for g in got if g["body"] != ar._default_render(g["kind"], g["entry"])]
    check("E3 kb_read stops before exceeding budget_chars and never emits a partial body",
          total <= budget and len(got) == 1 and not partial, f"total={total} budget={budget} n={len(got)}")

    mixed = ar.kb_read(kb, [999999, ids["apple"], "not-an-id", ids["knife"]], budget_chars=10_000)
    check("E4 kb_read skips unknown / hallucinated ids silently and returns the rest",
          [g["entry"]["id"] for g in mixed] == [ids["apple"], ids["knife"]],
          str([g["entry"]["id"] for g in mixed]))

    manifest_text, _rows = ar.kb_manifest(kb, budget_chars=100_000)
    t_man, t_bodies = ar.est_tokens(manifest_text), ar.est_tokens("\n".join(bodies))
    check("E5 progressive disclosure is cheaper than stuffing: est_tokens(manifest) < est_tokens(all bodies)",
          t_man < t_bodies, f"manifest={t_man} bodies={t_bodies}")


# ================================================================================================
# GROUP F -- provenance ranking (7)
# ================================================================================================
def group_f(ar, knowledge, tmp, now):
    kb = None
    hits = [{"kind": "facts", "entry": {"id": 5, "source": "prior"}, "n_match": 1, "matched": ["x"]},
            {"kind": "facts", "entry": {"id": 2, "source": "prior"}, "n_match": 3, "matched": ["x"]},
            {"kind": "facts", "entry": {"id": 9, "source": "prior"}, "n_match": 1, "matched": ["x"]}]
    plain = [h["entry"]["id"] for h in ar.rank_hits(hits, scene="FloorPlan1", now=now, mode="plain", kb=kb)]
    repeat = [h["entry"]["id"] for h in ar.rank_hits(hits, scene="FloorPlan1", now=now, mode="plain")]
    check("F1 rank_hits(mode='plain') orders exactly by (n_match desc, id asc) and is byte-reproducible",
          plain == [2, 5, 9] and plain == repeat, str(plain))

    base = {"confidence": 1.0, "ts": now, "scope": "FloorPlan1"}
    s_h, _ = ar.rank_key("facts", dict(base, id=1, source="human"), "FloorPlan1", 0.5, now=now)
    s_e, _ = ar.rank_key("facts", dict(base, id=2, source="environment"), "FloorPlan1", 0.5, now=now)
    check("F2 all else equal, source='human' outranks source='environment'", s_h > s_e, f"{s_h} vs {s_e}")

    prov = {"confidence": 1.0, "ts": now, "source": "human"}
    s_scene, _ = ar.rank_key("facts", dict(prov, id=1, scope="FloorPlan1"), "FloorPlan1", 0.5, now=now)
    s_glob, _ = ar.rank_key("facts", dict(prov, id=2, scope="global"), "FloorPlan1", 0.5, now=now)
    s_out, _ = ar.rank_key("facts", dict(prov, id=3, scope="FloorPlan9"), "FloorPlan1", 0.5, now=now)
    check("F3 scope==scene outranks 'global', which outranks an out-of-scene record",
          s_scene > s_glob > s_out, f"{s_scene} / {s_glob} / {s_out}")

    fresh = dict(prov, id=1, scope="FloorPlan1")
    stale = dict(fresh, ts=now - 14 * 24 * 3600.0)
    sf, pf = ar.rank_key("facts", fresh, "FloorPlan1", 0.5, now=now)
    ss, ps = ar.rank_key("facts", stale, "FloorPlan1", 0.5, now=now)
    check("F4 confidence decays with a 7-day half-life: 14 days old -> conf term x 0.5**2, score strictly lower",
          ss < sf and approx(ps["conf"], pf["conf"] * 0.5 ** 2, 1e-6),
          f"conf {pf['conf']} -> {ps['conf']}")

    u_more = ar.utility({"id": 1}, {"1": {"helpful": 5, "harmful": 0}})
    u_less = ar.utility({"id": 1}, {"1": {"helpful": 1, "harmful": 0}})
    u_harm = ar.utility({"id": 1}, {"1": {"helpful": 1, "harmful": 5}})
    u_zero = ar.utility({"id": 1}, {"1": {"helpful": 0, "harmful": 0}})
    check("F5 utility is monotone in helpful, anti-monotone in harmful, and in (0,1) at 0/0",
          u_more > u_less > u_harm and 0.0 < u_zero < 1.0 and approx(u_zero, 0.5),
          f"{u_more}/{u_less}/{u_harm}/{u_zero}")

    u100 = ar.utility({"id": 1, "occurrences": 100})
    u1000 = ar.utility({"id": 1, "occurrences": 1000})
    check("F6 the reinforcement proxy SATURATES: occ=1000 stays < 1.0 and within 0.01 of occ=100",
          u1000 < 1.0 and abs(u1000 - u100) < 0.01, f"{u100} -> {u1000}")

    kb2 = knowledge.KnowledgeBase(path=os.path.join(tmp, "supersede_kb.json"))
    weak = kb2.add_lesson("wash the mug", "failed", "do x", why="f", ts=now - 100, confidence=0.5)
    tough = kb2.add_lesson("dry the mug", "failed", "do y", why="f", ts=now - 100, confidence=0.5)
    weak["occurrences"], tough["occurrences"] = 1, 9
    p1 = kb2.add_procedure("wash the mug", [{"action": "wash", "object": "Mug"}], why="f",
                           ts=now, confidence=0.9)
    p2 = kb2.add_procedure("dry the mug", [{"action": "dry", "object": "Mug"}], why="f",
                           ts=now, confidence=0.9)
    p1["occurrences"], p2["occurrences"] = 3, 3
    sup_hits = [{"kind": "lessons", "entry": weak, "n_match": 1, "matched": ["mug"]},
                {"kind": "lessons", "entry": tough, "n_match": 1, "matched": ["mug"]}]
    ranked = {h["entry"]["id"]: h for h in ar.rank_hits(sup_hits, scene=None, now=now,
                                                        mode="provenance", kb=kb2)}
    hw, ht = ranked[weak["id"]], ranked[tough["id"]]
    # NOT self-referential. The old form computed its expected value from ar.STALE_PENALTY -- the very
    # constant it was meant to pin -- so setting STALE_PENALTY = 1.0 (supersession neutralised entirely)
    # still gave 83/83 PASS. Three independent assertions now: the constant is a real DEMOTION (a literal
    # open interval, not a value read back from the module), the demotion is applied, and a superseded
    # lesson ranks STRICTLY BELOW an otherwise-identical non-superseded twin -- the behaviour the design
    # actually claims, which no choice of constant can satisfy vacuously.
    twin_a = dict(weak)                                    # identical in every ranking input ...
    twin_b = dict(weak)
    twin_b["id"] = 999_001
    twin_b["command"] = "no procedure supersedes this one"  # ... except that nothing supersedes it
    twins = ar.rank_hits([{"kind": "lessons", "entry": twin_a, "n_match": 1, "matched": ["mug"]},
                          {"kind": "lessons", "entry": twin_b, "n_match": 1, "matched": ["mug"]}],
                         scene=None, now=now, mode="provenance", kb=kb2)
    tw = {h["entry"]["id"]: h for h in twins}
    a_score, b_score = tw[weak["id"]]["score"], tw[999_001]["score"]
    check("F7 supersession DEMOTES: 0 < STALE_PENALTY < 1 (literal bound), the penalty is applied, and a "
          "superseded lesson ranks strictly below an otherwise-identical non-superseded twin",
          0.0 < float(ar.STALE_PENALTY) < 1.0
          and hw["rank_parts"]["superseded"] is True and ht["rank_parts"]["superseded"] is False
          and approx(hw["score"], round(ar.STALE_PENALTY * hw["rank_parts"]["base"], 4), 1e-4)
          and a_score < b_score
          and [h["entry"]["id"] for h in twins] == [999_001, weak["id"]],
          f"penalty={ar.STALE_PENALTY} weak={hw['score']} base={hw['rank_parts']['base']} "
          f"tough={ht['score']} twins={a_score} vs {b_score}")

    # F8: the `weights` override REPLACES the vector (missing keys -> 0.0), so {"sim": 1.0} really is
    # plain similarity. It used to UPDATE the defaults, leaving prov/conf/util/scope at 0.10/0.10/0.15/0.10
    # for a weight sum of 1.45; two entries with identical sim then scored 0.9350 vs 0.5000 while the
    # docstring promised a tie. Every term individually ablatable is an API claim, so it is asserted here.
    rich = {"id": 1, "source": "human", "confidence": 1.0, "ts": now, "occurrences": 9, "scope": "FloorPlan1"}
    poor = {"id": 2, "source": "prior", "confidence": 0.0, "ts": 0.0, "occurrences": 0, "scope": "other"}
    s_rich, _p = ar.rank_key("procedures", rich, "FloorPlan1", sim=0.5, now=now, weights={"sim": 1.0})
    s_poor, _p = ar.rank_key("procedures", poor, "FloorPlan1", sim=0.5, now=now, weights={"sim": 1.0})
    no_prov = {k: v for k, v in ar.RANK_WEIGHTS.items() if k != "prov"}
    s_np1, np1 = ar.rank_key("procedures", rich, "FloorPlan1", sim=0.5, now=now, weights=no_prov)
    check("F8 weights={'sim':1.0} recovers PLAIN similarity exactly (both entries tie at sim) and an "
          "omitted key is a 0.0 weight, not the default",
          approx(s_rich, 0.5, 1e-9) and approx(s_poor, 0.5, 1e-9)
          and approx(np1["weights"]["prov"], 0.0) and approx(np1["weights"]["sim"], ar.RANK_WEIGHTS["sim"]),
          f"rich={s_rich} poor={s_poor} prov_w={np1['weights']['prov']}")


# ================================================================================================
# GROUP G -- typed quota (3)
# ================================================================================================
def group_g(ar):
    pool = [{"kind": "procedures", "entry": {"id": i}, "n_match": 3} for i in range(10)]
    pool.append({"kind": "lessons", "entry": {"id": 99}, "n_match": 1})
    sel = ar.quota_select(pool, k=5, quota=(2, 2, 1), mode="typed")
    mixed = ar.quota_select(pool, k=5, quota=(2, 2, 1), mode="mixed")
    check("G1 typed quota returns the lone lesson out of 10 procedures + 1 lesson (mixed does not)",
          99 in {h["entry"]["id"] for h in sel} and 99 not in {h["entry"]["id"] for h in mixed},
          str([h["entry"]["id"] for h in sel]))
    check("G2 unused quota slots backfill in rank order until exactly k (or the pool) is reached",
          len(sel) == 5 and [h["entry"]["id"] for h in sel] == [0, 1, 2, 3, 99]
          and len(ar.quota_select(pool[:2], k=5, quota=(2, 2, 1))) == 2,
          str([h["entry"]["id"] for h in sel]))
    dup = pool + [dict(pool[0]), dict(pool[1]), {"kind": "lessons", "entry": {"id": 99}, "n_match": 1}]
    out = [h["entry"]["id"] for h in ar.quota_select(dup, k=8, quota=(2, 2, 1), mode="typed")]
    out_m = [h["entry"]["id"] for h in ar.quota_select(dup, k=8, mode="mixed")]
    check("G3 no entry id ever appears twice, even when the input contains duplicates",
          len(out) == len(set(out)) and len(out_m) == len(set(out_m)), str(out))


# ================================================================================================
# GROUP H -- the bilingual pivot (5)
# ================================================================================================
def group_h(ar, records):
    en = ar.lexical_pivot("Put the apple in the fridge")
    check("H1 lexical_pivot(EN) -> types >= {Apple, Fridge}, a placement verb, ZERO model calls",
          {"Apple", "Fridge"} <= set(en["types"])
          and ({"pick_and_place"} & set(en["verbs"]) or {"put_in"} & set(en["verbs"]))
          and en["source"] == "lex", str(en))
    fa = ar.lexical_pivot(_FA_CMD)
    check("H2 lexical_pivot(FA) -> types >= {Apple, Fridge}: the FA->EN bridge with ZERO model calls",
          {"Apple", "Fridge"} <= set(fa["types"]) and fa["source"] == "lex", str(fa))

    dead_en = StubProvider(boom=ConnectionError("no provider"))
    p_en = ar.pivot("Put the apple in the fridge", provider=dead_en, model="m", mode="auto")
    dead_fa = StubProvider(boom=ConnectionError("no provider"))
    p_fa = ar.pivot("لطفا آن را انجام بده", provider=dead_fa, model="m", mode="auto")
    check("H3 pivot(mode='auto') spends 0 model calls on English and at most 1 on an unbridgeable Persian command",
          dead_en.calls == 0 and dead_fa.calls <= 1 and p_en["source"] == "lex"
          and p_fa["source"].endswith("lex") and p_fa["terms"],
          f"en={dead_en.calls} fa={dead_fa.calls} src={p_fa['source']}")

    # H4 used to ASSERT that FA_TYPE_LEXICON covers every gold object type in sim/dataset/commands.jsonl.
    # That is a regression test that GUARANTEES a fit to the evaluation set: the lexicon was hand-authored
    # from that file, ablating it drops FA recall@1 from 0.824 to 0.000 (51/51 empty result sets), and a
    # green check saying "coverage complete" reads as validation of a bridge that is really a lookup table.
    # The suite must not be complicit in that. H4 now REPORTS coverage as a number (never fails on it) and
    # H4b asserts the thing that is actually falsifiable: the held-out path exists, is honest (deleting a
    # query's own vocabulary really does remove those types), and does not mutate module state.
    if not records:
        skip("H4 report FA_TYPE_LEXICON coverage of the dataset's gold object types",
             "sim/eval.py did not import, so the dataset records could not be loaded")
    else:
        need = set()
        for r in records:
            g = r.gold or {}
            for a in (g.get("goal") or []):
                if a.get("type"):
                    need.add(a["type"])
            for lab in list(g.get("sequence") or []) + list(g.get("actions") or []):
                t = str(lab).split()
                if len(t) >= 2:
                    need.add(" ".join(t[1:]))
        have = set(ar.FA_TYPE_LEXICON.values())
        cov = len(need & have) / float(len(need)) if need else 0.0
        check("H4 FA_TYPE_LEXICON coverage of the dataset's gold object types is REPORTED, not enforced "
              "(coverage=%.3f of n=%d; this is an IN-SAMPLE fit, see the DISCLOSURE on FA_TYPE_LEXICON)"
              % (cov, len(need)),
              True, "")

    before = list(ar._TYPE_PAIRS)
    q = "سیب را در یخچال بگذار."
    full = ar.lexical_pivot(q)
    held = ar.lexical_pivot(q, type_pairs=ar.held_out_type_pairs(surface=q))
    fold = ar.lexical_pivot(q, type_pairs=ar.held_out_type_pairs(drop_keys=["یخچال"]))
    check("H4b held_out_type_pairs really withholds vocabulary (leave-this-query-out empties the type "
          "list; a named fold removes only its own key) and never mutates the module lexicon",
          set(full["types"]) == {"Apple", "Fridge"} and held["types"] == []
          and set(fold["types"]) == {"Apple"} and list(ar._TYPE_PAIRS) == before,
          f"full={full['types']} held={held['types']} fold={fold['types']}")

    sink_none = ar.lexical_pivot("Wash the mug in the sink.")
    sink_live = ar.lexical_pivot("Wash the mug in the sink.",
                                 scene_types=["Sink", "SinkBasin", "Mug", "Faucet"])
    sink_fa = ar.lexical_pivot("ماگ را در سینک بشوی.")
    check("H5b a live scene objectType list may only ADD surface forms, never redefine a built-in one: "
          "a kitchen holding both Sink and SinkBasin still pivots 'sink' to the canonical SinkBasin, so "
          "eval (which passes scene_types) and bench (which does not) retrieve identically, and the FA "
          "twin resolves to the same language-neutral type",
          sink_none["types"] == sink_live["types"] == sink_fa["types"]
          and "SinkBasin" in sink_live["types"],
          f"none={sink_none['types']} live={sink_live['types']} fa={sink_fa['types']}")

    bad = []
    for reply in ("not json at all, sorry", "", "```json\nnope\n```", "[1,2,3]", None):
        prov = StubProvider(reply=reply) if reply is not None else StubProvider(boom=TimeoutError("t/o"))
        try:
            res = ar.llm_pivot("x", prov, "m")
        except Exception as exc:
            bad.append(f"{reply!r} raised {exc!r}")
            continue
        if res.get("source") != "llm-failed" or res.get("types") or res.get("ids"):
            bad.append(f"{reply!r} -> {res}")
    none_prov = ar.llm_pivot("x", None, "m")
    check("H5 llm_pivot on garbage / prose / empty / a raising provider returns source='llm-failed', never raises",
          not bad and none_prov["source"] == "llm-failed", str(bad[:2]))


# ================================================================================================
# GROUP I -- the sufficiency gate and the hop bound (6)
# ================================================================================================
def group_i(ar, kb, ids):
    hits = ar.kb_grep(kb, ["Apple", "Fridge"], scene="FloorPlan1")
    ok1, r1 = ar.sufficient(hits, ["Apple", "Fridge"], "FloorPlan1", observation="Apple, Fridge, CounterTop")
    check("I1 sufficient() is True when an in-scope hit covers every OBSERVATION-present queried type",
          ok1 is True and r1["ok"] is True and r1["in_scene"] >= 1 and not r1["types_uncovered"], str(r1))
    ok2, r2 = ar.sufficient(hits, ["Apple", "Fridge", "Microwave"], "FloorPlan1",
                            observation="Apple, Fridge, Microwave")
    check("I2 it is False and names the uncovered term in reason['terms_missed']",
          ok2 is False and "Microwave" in r2["terms_missed"] and "Microwave" in r2["types_uncovered"], str(r2))
    ok3, r3 = ar.sufficient(hits, ["Apple", "Unicorn"], "FloorPlan1", observation="Apple, Fridge")
    check("I3 a term absent from the OBSERVATION never blocks the gate",
          ok3 is True and "Unicorn" in r3["terms_missed"] and "Unicorn" not in r3["types_uncovered"], str(r3))

    st = StubProvider()
    ag = ar.AgenticRAG(kb=kb, provider=st, model="m", hops=2, gate=True, cache=False)
    ag.search("put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge, CounterTop")
    t = ag.last_telemetry()
    check("I4 gate on + a satisfiable hop 0 -> ZERO provider.chat calls and telemetry['hops'] == 0",
          st.calls == 0 and t["hops"] == 0 and t["llm_calls"] == 0 and t["gate_stopped"] is True,
          f"calls={st.calls} tele={t['hops']}/{t['llm_calls']}")

    st2 = StubProvider()
    ag2 = ar.AgenticRAG(kb=kb, provider=st2, model="m", hops=2, gate=False, pivot_mode="llm",
                        level_policy="off", cache=False)
    ag2.search("Make me a cold snack.", scene="FloorPlan1", observation="Apple, Fridge")
    st3 = StubProvider()
    ag3 = ar.AgenticRAG(kb=kb, provider=st3, model="m", hops=2, gate=True, pivot_mode="llm", cache=False)
    ag3.search("Prepare something nice with the unicorn and the dragon.", scene="FloorPlan1",
               observation="Unicorn, Dragon")           # ADVERSARIAL: the gate can never be satisfied
    check("I5 gate off (and an unsatisfiable gate) spend EXACTLY the hop budget: 2 calls, never more",
          st2.calls == 2 and st3.calls == 2 and ag2.last_telemetry()["hops"] == 2
          and ag3.last_telemetry()["hops"] == 2, f"gate-off={st2.calls} adversarial={st3.calls}")

    st4 = StubProvider()
    ag4 = ar.AgenticRAG(kb=kb, provider=st4, model="m", hops=99, gate=False, pivot_mode="llm",
                        level_policy="off", cache=False)
    ag4.search("Prepare something nice with the unicorn.", scene="FloorPlan1", observation="Unicorn")
    st5 = StubProvider()
    ag5 = ar.AgenticRAG(kb=kb, provider=st5, model="m", hops=99, gate=False, pivot_mode="llm",
                        level_policy="est", cache=False)
    ag5.search("Open the fridge.", scene="FloorPlan1", observation="Fridge")     # L1 policy -> 0 hops
    # `st4.calls == ar.MAX_HOPS` alone was self-referential: MAX_HOPS = 99 gave 83/83 PASS, i.e. a retrieval
    # hop cap 33x over budget (up to 99 LIVE model calls per query) was invisible to the suite. The literal
    # bound is asserted first, then the clamp against it.
    check("I6 MAX_HOPS <= 3 (literal budget bound) and hops clamps to "
          "min(self.hops, MAX_HOPS, level policy): 99 -> MAX_HOPS, L1 -> 0",
          ar.MAX_HOPS <= 3 and st4.calls == ar.MAX_HOPS and st4.calls <= 3 and st5.calls == 0
          and ag5.last_telemetry()["level"] == "L1",
          f"MAX_HOPS={ar.MAX_HOPS} cap={st4.calls} L1={st5.calls}")


# ================================================================================================
# GROUP J -- the probe cache (3)
# ================================================================================================
def _ids(hits):
    return [(h.get("kind"), (h.get("entry") or {}).get("id"), h.get("score")) for h in hits]


def group_j(ar, knowledge, tmp, now):
    kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "cache_kb.json"))
    kb.add_procedure("put the apple in the fridge",
                     [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}],
                     why="f", scene="FloorPlan1", tags=["apple"], source="environment", ts=now)
    kb.add_fact("Apples live in the Fridge.", why="f", tags=["apple"], source="prior", ts=now)

    ag = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=True)
    a = ag.search("put apple in fridge", scene="FloorPlan1", k=5)
    b = ag.search("put apple in fridge", scene="FloorPlan1", k=5)
    st = ag.cache.stats()
    check("J1 the same query twice with cache=True yields identical hits and exactly one cache hit",
          _ids(a) == _ids(b) and st["hits"] == 1 and st["misses"] == 1, str(st))

    fp_before = ar.kb_fingerprint(kb)
    kb.add_fact("A brand new fact about the Kettle.", why="f", tags=["kettle"], source="prior", ts=now)
    fp_after = ar.kb_fingerprint(kb)
    ag.search("put apple in fridge", scene="FloorPlan1", k=5)
    st2 = ag.cache.stats()
    check("J2 adding an entry changes kb_fingerprint and the next probe is a MISS (exact invalidation)",
          fp_before != fp_after and st2["misses"] == st["misses"] + 1 and st2["hits"] == st["hits"],
          f"{fp_before} -> {fp_after}, {st2}")

    agn = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=False)
    agc = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=True)
    check("J3 cache=False produces byte-identical hits to cache=True (an exact cache cannot change retrieval)",
          _ids(agn.search("put apple in fridge", scene="FloorPlan1", k=5))
          == _ids(agc.search("put apple in fridge", scene="FloorPlan1", k=5))
          and agn.cache is None and agn.last_telemetry()["cache"] is None)


# ================================================================================================
# GROUP K -- the deterministic plan verifier (4)
# ================================================================================================
def group_k(ar):
    v = ar.verify_plan([{"action": "teleport", "object": "Apple"}], {"Apple"}, {"pickup", "open"})
    check("K1 an action outside the skill library is flagged 'unknown_action'",
          v == [{"step": 1, "kind": "unknown_action", "value": "teleport"}], str(v))
    v2 = ar.verify_plan([{"action": "pickup", "object": "Unicorn"}], {"Apple", "Fridge"}, {"pickup"})
    check("K2 an object type absent from the OBSERVATION is flagged 'unknown_object'",
          v2 == [{"step": 1, "kind": "unknown_object", "value": "Unicorn"}], str(v2))
    clean = ar.verify_plan([{"action": "pickup", "object": "Apple"}], {"Apple"}, {"pickup"})
    check("K3 a clean plan returns []; None / a non-list / an empty plan return [] without raising",
          clean == [] and ar.verify_plan(None, {"Apple"}, {"pickup"}) == []
          and ar.verify_plan("nonsense", {"Apple"}, {"pickup"}) == []
          and ar.verify_plan([], {"Apple"}, {"pickup"}) == []
          and ar.verify_plan([None, 3], {"Apple"}, {"pickup"}) == [])
    v4 = ar.verify_plan([{"action": "pick_and_place", "object": "fridge", "to": "counter top"}],
                        {"Fridge", "CounterTop"}, {"pick_and_place"})
    check("K4 object comparison is case- and alias-insensitive ('fridge' is clean when 'Fridge' is observed)",
          v4 == [], str(v4))


# ================================================================================================
# GROUP L -- degradation and the never-worse-than-keyword floor (5)
# ================================================================================================
class _BrokenBoth:
    """A KnowledgeBase whose .data AND .search both raise: the worst case the retriever must survive."""
    path = "broken.json"

    @property
    def data(self):
        raise RuntimeError("knowledge base data exploded")

    def search(self, query, limit=5):
        raise RuntimeError("knowledge base search exploded")


class _BrokenData:
    """.data raises but .search works: the degradation path must land on keyword_fallback, not on []."""
    path = "broken.json"

    def __init__(self, entry):
        self._entry = entry

    @property
    def data(self):
        raise RuntimeError("knowledge base data exploded")

    def search(self, query, limit=5):
        return [{"kind": "facts", "score": 2.0, "entry": self._entry}]


def group_l(ar, kb, ids, dskb, records):
    ag = ar.AgenticRAG(kb=kb, provider=None, hops=2, cache=False)
    hits = ag.search("put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge")
    t = ag.last_telemetry()
    check("L1 provider=None searches successfully with llm_calls == 0 and no exception",
          hits and t["llm_calls"] == 0 and t["degraded"] is None, str(t))

    _TRAP.reset()
    dead = StubProvider(boom=ConnectionError("Ollama unreachable"))
    ag2 = ar.AgenticRAG(kb=kb, provider=dead, model="m", hops=2, gate=False, pivot_mode="llm",
                        level_policy="off", cache=False)
    try:
        hits2 = ag2.search("put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge")
        raised = None
    except Exception as exc:
        hits2, raised = [], exc
    t2 = ag2.last_telemetry()
    check("L2 a provider raising ConnectionError -> hop-0 hits kept, telemetry['degraded'] set, a WARNING "
          "logged, nothing raised",
          raised is None and hits2 and t2["degraded"] and any("llm pivot unavailable" in m
                                                              for m in _TRAP.records),
          f"raised={raised!r} n={len(hits2)} degraded={t2['degraded']} warns={_TRAP.records[:1]}")

    _TRAP.reset()
    fact = {"id": 7, "text": "a fallback fact", "source": "prior", "confidence": 1.0}
    try:
        agb = ar.AgenticRAG(kb=_BrokenBoth(), provider=None, cache=False)
        hb = agb.search("apple", scene="FloorPlan1")
        agd = ar.AgenticRAG(kb=_BrokenData(fact), provider=None, cache=False)
        hd = agd.search("apple", scene=None)
        raised2 = None
    except Exception as exc:
        hb = hd = None
        raised2 = exc
    check("L3 a KnowledgeBase that raises -> keyword_fallback-shaped results or [], never an exception",
          raised2 is None and hb == [] and hd and hd[0]["entry"]["id"] == 7
          and set(hd[0]) >= {"kind", "score", "keyword", "cosine", "entry"}
          and agb.last_telemetry()["degraded"] and agd.last_telemetry()["degraded"],
          f"raised={raised2!r} hb={hb} hd={hd}")

    # ALL records, not records[:20]. The truncated loop is precisely what hid the real superset violation
    # this contract exists to catch: `Pick up the lettuce.` (fp2-l1-pickup-lettuce-en) is record #21+ in the
    # dataset, and `knowledge.search` matched two lessons whose body reads "pick_and_place Bread failed: ..."
    # that `kb_grep` could not, because re.compile(r"\bpick\b") treats `_` as a word character. Twenty
    # records is not a floor; it is a spot check that reads like one. `k=10_000` on the floor side because
    # a superset claim over a top-5 slice is a weaker claim than the one the docstring makes.
    if dskb is None or not records:
        skip("L4 the probe pool is a SUPERSET of the keyword floor over EVERY dataset command",
             "sim/eval.py did not import, so the dataset records could not be loaded")
    else:
        misses = []
        for r in records:
            floor = {(h.get("entry") or {}).get("id")
                     for h in ar.keyword_fallback(dskb, r.command, scene=r.scene, k=10_000)}
            piv = ar.pivot(r.command, mode="lex")
            terms = ar._ordered_unique(list(piv.get("terms") or []) + sorted(ar.tokens(r.command)))
            pool = {(h.get("entry") or {}).get("id")
                    for h in ar.kb_grep(dskb, terms, scene=r.scene, limit=200)}
            if not floor <= pool:
                misses.append((r.id, sorted(floor - pool)))
        check("L4 the probe pool is a SUPERSET of the keyword floor over EVERY dataset command (n=%d) "
              "-- the infrastructural floor" % len(records), not misses, str(misses[:3]))

    check("L4b whole-word probing treats `_` as a boundary (so `pick` finds `pick_and_place`, matching "
          "knowledge.search's tokenizer) and Arabic punctuation as a boundary (so `بردار` finds `بردار؟`)",
          bool(ar._term_pattern("pick").search("pick_and_place bread failed"))
          and not ar._term_pattern("pick").search("pickup the apple")
          and bool(ar._term_pattern("بردار").search(ar.normalize_text("سیب را بردار؟")))
          and bool(ar._term_pattern("کن").search(ar.normalize_text("مایکروویو گرم کن، سپس")))
          and bool(ar._term_pattern("pick and place").search("a pick and place step")))

    a = ag.search("put apple in fridge", scene="FloorPlan1", alpha=0.5)
    b = ag.search("put apple in fridge", scene="FloorPlan1", alpha=None)
    c = ag.search("put apple in fridge", scene="FloorPlan1", alpha=0.0)
    check("L5 `alpha` is accepted and IGNORED (drop-in polymorphism with HybridRAG.search)",
          _ids(a) == _ids(b) == _ids(c), str(_ids(a)))


# ================================================================================================
# GROUP M -- hit-shape contract and eval integration (6)
# ================================================================================================
def group_m(ar, ev, kb, ids):
    ag = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=False)
    hits = ag.search("apple fridge knife clean light", scene="FloorPlan1", k=5)
    check("M1 every hit has kind in KINDS and a dict entry carrying an id (S8's load-bearing contract)",
          hits and all(h.get("kind") in ar.KINDS and isinstance(h.get("entry"), dict)
                       and h["entry"].get("id") is not None for h in hits),
          str([(h.get("kind"), (h.get("entry") or {}).get("id")) for h in hits]))
    check("M2 every hit carries score / keyword / cosine (cosine == 0.0), so _render_hits cannot degrade",
          all(isinstance(h.get("score"), float) and isinstance(h.get("keyword"), float)
              and h.get("cosine") == 0.0 for h in hits),
          str([(h.get("score"), h.get("keyword"), h.get("cosine")) for h in hits]))

    hybrid_like = [{"kind": h["kind"], "score": h["score"], "keyword": h["keyword"],
                    "cosine": h["cosine"], "entry": h["entry"]} for h in hits]
    if ev is None:
        for nm in ("M3 eval._render_hits(agentic hits) is byte-identical to the equivalent hybrid hits' block",
                   "M4 build_memory_block(rag_mode='nonsense') raises ValueError (the silent-hybrid trap is closed)",
                   "M5 build_memory_block(rag_mode='off') returns ('', [], telemetry) with telemetry['mode']=='off'"):
            skip(nm, "sim/eval.py unimportable")
        phase10 = False
        sig = {}
    else:
        block_a, block_b = ev._render_hits(hits), ev._render_hits(hybrid_like)
        check("M3 eval._render_hits(agentic hits) is byte-identical to the equivalent hybrid hits' block",
              block_a == block_b and block_a.startswith("**Relevant memory**")
              and "- **" in block_a, repr(block_a[:70]))
        sig = inspect.signature(ev.build_memory_block).parameters
        phase10 = "agentic" in sig
    if ev is not None and not phase10:
        skip("M4 build_memory_block(rag_mode='nonsense') raises ValueError (the silent-hybrid trap is closed)",
             "eval.py Phase-10 edits not on disk yet (agent A2)")
        skip("M5 build_memory_block(rag_mode='off') returns ('', [], telemetry) with telemetry['mode']=='off'",
             "eval.py Phase-10 edits not on disk yet (agent A2)")
    elif ev is not None:
        try:
            ev.build_memory_block(None, None, "FloorPlan1", "put apple in fridge", "nonsense")
            raised = None
        except ValueError as exc:
            raised = exc
        except Exception as exc:
            raised = exc
        check("M4 build_memory_block(rag_mode='nonsense') raises ValueError (the silent-hybrid trap is closed)",
              isinstance(raised, ValueError), repr(raised))
        out = ev.build_memory_block(None, None, "FloorPlan1", "put apple in fridge", "off")
        check("M5 build_memory_block(rag_mode='off') returns ('', [], telemetry) with telemetry['mode']=='off'",
              isinstance(out, tuple) and len(out) == 3 and out[0] == "" and out[1] == []
              and isinstance(out[2], dict) and out[2].get("mode") == "off", str(out)[:120])

    t_ok = ag.last_telemetry()
    agb = ar.AgenticRAG(kb=_BrokenBoth(), provider=None, cache=False)
    agb.search("apple")
    t_bad = agb.last_telemetry()
    need = set(ar._TELEMETRY_KEYS)
    check("M6 last_telemetry() carries every documented key after a successful AND after a failed search",
          need <= set(t_ok) and need <= set(t_bad) and t_bad["degraded"],
          f"missing_ok={sorted(need - set(t_ok))} missing_bad={sorted(need - set(t_bad))}")


# ================================================================================================
# GROUP O -- every ablation switch genuinely changes the output (9)
# ================================================================================================
def group_o(ar, knowledge, tmp, now, kb, ids, dskb):
    # O1 needs a fixture where plain order and provenance order provably disagree.
    kb2 = knowledge.KnowledgeBase(path=os.path.join(tmp, "rank_kb.json"))
    lo = kb2.add_fact("apple note one", why="f", tags=["apple"], source="prior",
                      ts=now - 30 * 24 * 3600.0, confidence=0.2, scope="FloorPlan9")
    hi = kb2.add_fact("apple note two", why="f", tags=["apple"], source="human",
                      ts=now, confidence=1.0, scope="FloorPlan1")
    hits = ar.kb_grep(kb2, "apple", scene=None)
    o_plain = [h["entry"]["id"] for h in ar.rank_hits(hits, scene="FloorPlan1", now=now, mode="plain")]
    o_prov = [h["entry"]["id"] for h in ar.rank_hits(hits, scene="FloorPlan1", now=now,
                                                     mode="provenance", kb=kb2)]
    o_none = [h["entry"]["id"] for h in ar.rank_hits(hits, scene="FloorPlan1", now=now, mode="none")]
    check("O1 --rank plain vs provenance changes the ORDER (and mode='none' preserves insertion order)",
          o_plain == [lo["id"], hi["id"]] and o_prov == [hi["id"], lo["id"]]
          and o_none == [h["entry"]["id"] for h in hits],
          f"plain={o_plain} prov={o_prov}")

    pool = [{"kind": "procedures", "entry": {"id": i}, "n_match": 3} for i in range(10)]
    pool.append({"kind": "lessons", "entry": {"id": 99}, "n_match": 1})
    check("O2 --agentic-quota typed vs mixed changes the SELECTION",
          ar.quota_select(pool, k=5, mode="typed") != ar.quota_select(pool, k=5, mode="mixed"))

    st_on = StubProvider()
    st_off = StubProvider()
    kw = {"kb": kb, "model": "m", "hops": 2, "pivot_mode": "llm", "level_policy": "off", "cache": False}
    ar.AgenticRAG(provider=st_on, gate=True, **kw).search(
        "put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge")
    ag_off = ar.AgenticRAG(provider=st_off, gate=False, **kw)
    ag_off.search("put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge")
    check("O3 --agentic-gate on vs off changes the number of model calls (the gate's saving is real)",
          st_on.calls < st_off.calls and st_off.calls == 2, f"on={st_on.calls} off={st_off.calls}")

    ag_c = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=True)
    ag_n = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=False)
    ag_c.search("apple", scene="FloorPlan1")
    ag_n.search("apple", scene="FloorPlan1")
    check("O4 --agentic-cache on vs off changes the TELEMETRY (dict of stats vs None) but not the hits",
          isinstance(ag_c.last_telemetry()["cache"], dict) and ag_n.last_telemetry()["cache"] is None)

    kb3 = knowledge.KnowledgeBase(path=os.path.join(tmp, "compact_kb.json"))
    for _ in range(3):                       # three phrasings of ONE lesson: length without distinctness
        kb3.add_lesson("slice the bread", "SliceObject failed: not holding a knife",
                       "pick up a Knife before slicing", why="f", scene="FloorPlan1",
                       source="environment", ts=now, confidence=0.6)
    kb3.add_procedure("slice the bread", [{"action": "slice", "object": "Bread"}], why="f",
                      scene="FloorPlan1", tags=["bread"], source="environment", ts=now)
    on = ar.AgenticRAG(kb=kb3, provider=None, hops=0, compact=True, cache=False)
    off = ar.AgenticRAG(kb=kb3, provider=None, hops=0, compact=False, cache=False)
    h_on = on.search("slice the bread", scene="FloorPlan1", k=5)
    h_off = off.search("slice the bread", scene="FloorPlan1", k=5)
    check("O5 --agentic-compact on vs off changes the block: near-duplicate lessons merge",
          len(h_on) < len(h_off) and any(h.get("why_line") for h in h_on),
          f"on={len(h_on)} off={len(h_off)}")

    ag_lex = ar.AgenticRAG(kb=kb, provider=None, hops=0, pivot_mode="lex", cache=False)
    ag_poff = ar.AgenticRAG(kb=kb, provider=None, hops=0, pivot_mode="off", cache=False)
    ag_lex.search(_FA_CMD, scene="FloorPlan1", k=5)
    ag_poff.search(_FA_CMD, scene="FloorPlan1", k=5)
    t_lex, t_off = ag_lex.last_telemetry(), ag_poff.last_telemetry()
    check("O6 --agentic-pivot lex vs off changes the probe TERMS (the FA->EN types disappear)",
          set(t_lex["terms"]) > set(t_off["terms"]) and {"Apple", "Fridge"} <= set(t_lex["terms"])
          and not ({"Apple", "Fridge"} & set(t_off["terms"])),
          f"lex={t_lex['terms']} off={t_off['terms']}")

    if dskb is None:
        skip("O7 --agentic-level est vs oracle vs off changes k / hops / the returned memories",
             _no_records_reason())
    else:
        est = ar.AgenticRAG(kb=dskb, provider=None, hops=0, k=6, level_policy="est", cache=False)
        ora = ar.AgenticRAG(kb=dskb, provider=None, hops=0, k=6, level_policy="oracle", cache=False)
        uni = ar.AgenticRAG(kb=dskb, provider=None, hops=0, k=6, level_policy="off", cache=False)
        q = "Open the fridge."
        h_est = est.search(q, scene="FloorPlan1")
        h_ora = ora.search(q, scene="FloorPlan1", level="L3")
        h_uni = uni.search(q, scene="FloorPlan1")
        check("O7 --agentic-level est vs oracle vs off changes the level, k and the returned memories",
              est.last_telemetry()["level"] == "L1" and est.last_telemetry()["level_source"] == "est"
              and ora.last_telemetry()["level"] == "L3" and ora.last_telemetry()["level_source"] == "oracle"
              and uni.last_telemetry()["level_source"] == "off"
              and len(h_est) < len(h_ora) and len(h_uni) >= len(h_est),
              f"est={len(h_est)} oracle={len(h_ora)} off={len(h_uni)}")

    # ---- O7b/O7c/O7d: every CLI knob must MOVE the retriever in the direction its help text promises.
    # All three were no-ops before this pass, because LEVEL_POLICY held absolute caps rather than fractions:
    #  * --agentic-budget could only ever SHRINK the block (6000 / 20000 / 40000 all produced the identical
    #    695-char block on the bench fixture; only a value below the level cap changed anything),
    #  * --agentic-hops 3 was unreachable under the default --agentic-level est (L3 topped out at 2 while
    #    MAX_HOPS is 3, so argparse offered a choice that could not take effect),
    #  * the documented L1 k=3 / L2 k=5 / L3 k=6 was dead in every eval run, because build_memory_block
    #    always passes an explicit k=5 and an explicit k used to win outright over the policy.
    # A switch that cannot change the output is a bug, not a feature -- this group's whole premise.
    lb = {lv: ar.level_budget(lv, 2, 5, 6000) for lv in ("L1", "L2", "L3")}
    check("O7b level_budget at the shipped defaults (hops=2, k=5, budget=6000) reproduces the documented "
          "table exactly: L1 0/3/2400, L2 1/5/4800, L3 2/6/6000",
          (lb["L1"]["hops"], lb["L1"]["k"], lb["L1"]["budget_chars"]) == (0, 3, 2400)
          and (lb["L2"]["hops"], lb["L2"]["k"], lb["L2"]["budget_chars"]) == (1, 5, 4800)
          and (lb["L3"]["hops"], lb["L3"]["k"], lb["L3"]["budget_chars"]) == (2, 6, 6000),
          str(lb))
    check("O7c --agentic-hops 3 is REACHABLE (L3 of a 3-hop budget is 3, still clamped by MAX_HOPS) and "
          "--agentic-budget is monotone upward at every level, never only downward",
          ar.level_budget("L3", 3, 5, 6000)["hops"] == 3
          and ar.level_budget("L3", 99, 5, 6000)["hops"] == ar.MAX_HOPS
          and ar.level_budget("L1", 2, 5, 12000)["budget_chars"] > lb["L1"]["budget_chars"]
          and ar.level_budget("L3", 2, 5, 12000)["budget_chars"] > lb["L3"]["budget_chars"]
          and ar.level_budget("L1", 2, 10, 6000)["k"] > lb["L1"]["k"],
          f"hops3={ar.level_budget('L3', 3, 5, 6000)['hops']} "
          f"budget12k_L1={ar.level_budget('L1', 2, 5, 12000)['budget_chars']}")

    # A KB where BOTH an L1 and an L3 command have more candidates than either level's k, so the only thing
    # that can cap the returned count is the level policy itself.
    kkb = knowledge.KnowledgeBase(path=os.path.join(tmp, "levelk_kb.json"))
    for i in range(12):
        kkb.add_fact(f"apple note {i}: the Apple and the Fridge are both in this kitchen",
                     why="f", tags=["apple", "fridge"], source="prior", ts=now)
    k_ex = ar.AgenticRAG(kb=kkb, provider=None, hops=0, k=5, level_policy="est", cache=False,
                         quota_mode="mixed")
    n_l1 = len(k_ex.search("Open the fridge.", scene=None, k=5))                  # L1 -> k_eff 3
    l1_tele = k_ex.last_telemetry()
    n_l3 = len(k_ex.search("Prepare something with the apple.", scene=None, k=5))  # L3 -> k_eff 6
    l3_tele = k_ex.last_telemetry()
    check("O7d the level policy's k is LIVE on the eval path: an explicit search(k=5) -- exactly what "
          "build_memory_block passes -- still yields fewer memories at L1 (k_eff 3) than at L3 (k_eff 6)",
          l1_tele["level"] == "L1" and l3_tele["level"] == "L3" and n_l1 == 3 and n_l3 == 6,
          f"L1={n_l1}@{l1_tele['level']} L3={n_l3}@{l3_tele['level']}")

    probe = ar.AgenticRAG(kb=kb, provider=None, hops=0)
    agentic = ar.AgenticRAG(kb=kb, provider=None, hops=2)
    check("O8 --rag probe vs agentic is a real distinction: hops=0 reports mode 'probe', hops>0 'agentic'",
          probe.mode == "probe" and agentic.mode == "agentic")

    # ---- O10: --rank provenance+utility must not be a duplicate of --rank provenance. It was: the
    # mechanism it depends on (rag.credit, which writes <kb stem>_utility.json) did not exist, so
    # AgenticRAG._load_utility always returned {} and both spellings took the same rank_key branch --
    # and the eval call site swallowed the resulting AttributeError with a bare `except Exception: pass`,
    # so the run looked healthy. Asserted end to end here, against a sidecar in the TEMP dir.
    import rag as _ragmod
    ukb = knowledge.KnowledgeBase(path=os.path.join(tmp, "credit_kb.json"))
    u1 = ukb.add_fact("the mug lives on the counter", why="f", tags=["mug"], source="prior", ts=now)
    u2 = ukb.add_fact("the mug is sometimes in the sink", why="f", tags=["mug"], source="prior", ts=now)
    have_credit = callable(getattr(_ragmod, "credit", None))
    if have_credit:
        _ragmod.credit(ukb.path, [u2["id"]], True, why="selftest", ts=now)     # u2 helped ...
        for _ in range(4):
            _ragmod.credit(ukb.path, [u1["id"]], False, why="selftest", ts=now)   # ... u1 misled
    upath = _ragmod.utility_path(ukb.path) if have_credit else ""
    ag_prov = ar.AgenticRAG(kb=ukb, provider=None, hops=0, rank_mode="provenance", cache=False)
    ag_util = ar.AgenticRAG(kb=ukb, provider=None, hops=0, rank_mode="provenance+utility", cache=False)
    ids_prov = [(h.get("entry") or {}).get("id") for h in ag_prov.search("mug", scene=None, k=5)]
    ids_util = [(h.get("entry") or {}).get("id") for h in ag_util.search("mug", scene=None, k=5)]
    check("O10 --rank provenance+utility is a REAL arm, not a duplicate: rag.credit exists, writes the "
          "utility sidecar, and the credited ranking differs from the uncredited one",
          have_credit and os.path.exists(upath) and bool(ag_util.util)
          and ar.utility(u2, ag_util.util) > ar.utility(u1, ag_util.util)
          and ids_util == [u2["id"], u1["id"]] and ids_prov != ids_util,
          f"credit={have_credit} util={getattr(ag_util, 'util', None)} prov={ids_prov} util_ids={ids_util}")

    ag_full = ar.AgenticRAG(kb=kb, provider=None, hops=0, cache=False)
    dumped = ag_full.dump()
    t_full = ag_full.last_telemetry()
    searched = ag_full.search("apple", scene=None, k=5)
    check("O9 --rag full dumps EVERY in-scope record (stamped rag_full) and differs from a retrieval arm",
          len(dumped) == len(ar.kb_entries(kb)) and all(h.get("rag_full") for h in dumped)
          and len(dumped) > len(searched) and t_full["mode"] == "full",
          f"full={len(dumped)} search={len(searched)}")


# ================================================================================================
# GROUP N -- data safety (3)
# ================================================================================================
#: Repo stores this suite must leave byte-identical. `trace.db` (+ its -shm/-wal siblings) and the
#: `*_utility.json` retrieval-credit sidecar are here because BOTH are written by sibling modules this
#: suite imports -- trace.db by knowledge.py's tracing hook, the sidecar by rag.credit -- and neither was
#: guarded before. A store that already exists on disk is exactly the case the old path-set check could
#: not see, which is why N1 now covers them explicitly and N2 compares content, not names.
_GUARDED = ["knowledge_base.json", "knowledge_base_embeddings.json", "knowledge_base_audit.log",
            "knowledge_base_utility.json", "memory.db", "loop_reports.jsonl",
            "trace.db", "trace.db-shm", "trace.db-wal",
            os.path.join("dataset", "commands.jsonl")]


def _digest(path):
    """Content + size + mtime. Size is included so a same-length-different-bytes edit and an APPEND are
    both caught even if the hash read is short-circuited by an error."""
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
        return (hashlib.sha256(blob).hexdigest()[:16]
                + f":{len(blob)}:{os.path.getmtime(path):.0f}")
    except FileNotFoundError:
        return "absent"
    except Exception as exc:
        return f"unreadable:{exc}"


def snapshot_stores():
    snap = {p: _digest(os.path.join(HERE, p)) for p in _GUARDED}
    for name in sorted(os.listdir(HERE)):
        if name.startswith("eval_") and name.endswith(".json"):
            snap[name] = _digest(os.path.join(HERE, name))
    return snap


def snapshot_tree():
    """``{path: (size, mtime)}`` for sim/ and the repo root, minus bytecode caches (which are not data).

    A MAPPING, not a set of names: the old set-of-paths version passed vacuously against any store that
    already existed, so a run that appended 270 KB to sim/trace.db reported "no file was created" and was
    telling the truth about the wrong thing. Comparing (size, mtime) catches creation AND growth AND
    in-place rewrite, which is what "nothing outside the temp dir was touched" has to mean.
    """
    out = {}
    for root in (HERE, os.path.dirname(HERE)):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "node_modules")]
            for f in filenames:
                if f.endswith((".pyc", ".pyo")):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    st = os.stat(p)
                    out[p] = (st.st_size, round(st.st_mtime, 3))
                except OSError:
                    out[p] = ("gone", 0)
            if root != HERE:
                break                                   # repo root: top level only (docs/ etc. are large)
    return out


def group_n(ar, kb, ids, stores_before, tree_before):
    after = snapshot_stores()
    changed = [k for k, v in stores_before.items() if after.get(k) != v]
    check("N1 knowledge_base.json + sidecars + memory.db + trace.db + the dataset are byte-identical "
          "afterwards (size and mtime too)", not changed, str(changed))

    tree_after = snapshot_tree()
    created = sorted(set(tree_after) - set(tree_before))
    modified = sorted(p for p in set(tree_after) & set(tree_before) if tree_after[p] != tree_before[p])
    check("N2 no file outside the temporary directory was created OR modified (size+mtime compared, so a "
          "pre-existing store that merely GREW is caught)",
          not created and not modified,
          f"created={created[:3]} modified={modified[:3]}")

    import socket as _sock
    real = _sock.socket

    class _NoSocket:
        def __init__(self, *a, **kw):
            raise AssertionError("agentic_rag opened a socket")

    _sock.socket = _NoSocket
    try:
        ag = ar.AgenticRAG(kb=kb, provider=None, hops=2, cache=False)
        hits = ag.search("put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge")
        _text, _rows = ar.kb_manifest(kb)
        _dump = ag.dump()
        _recall = ag.recall_text("slice the bread", scene="FloorPlan1")
        sock_err = None
    except Exception as exc:
        hits, sock_err = [], exc
    finally:
        _sock.socket = real
    check("N3 AgenticRAG never opens a socket (socket.socket stubbed to raise; search/manifest/dump/recall OK)",
          sock_err is None and bool(hits), repr(sock_err))


# ================================================================================================
# MAIN
# ================================================================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 100)
    print("PHASE-10 AGENTIC-RAG SELF-TEST  (pure: no ai2thor / Ollama / numpy / network / repo store)")
    print("=" * 100)

    stores_before = snapshot_stores()
    tree_before = snapshot_tree()

    ar = group_a()
    if ar is None:
        return _summary_and_exit()

    import knowledge                                   # AFTER group A, which asserts it was not dragged in
    try:                                               # sim/eval.py, not the builtin. Another agent may be
        import eval as ev                              # mid-edit; a broken sibling must SKIP, not crash us.
    except Exception as exc:                           # noqa: A004
        print(f"  [note] sim/eval.py is not importable right now ({exc!r}); "
              f"its integration checks will SKIP")
        ev = None

    records = []
    if ev is None:
        _NO_RECORDS_REASON[0] = ("sim/eval.py failed to import, so ev.load_dataset could not run "
                                 "(the dataset file itself is fine)")
    else:
        try:
            records = ev.load_dataset(DATASET, lang="all", limit=None)
            if not records:
                _NO_RECORDS_REASON[0] = f"ev.load_dataset({DATASET}) returned 0 records"
        except Exception as exc:
            records = []
            _NO_RECORDS_REASON[0] = f"ev.load_dataset raised {exc!r}"

    tmp = tempfile.mkdtemp(prefix="agentic_rag_selftest_")
    now = 1_750_000_000.0                              # pinned: nothing here depends on wall-clock time
    try:
        kb, ids = build_fixture(ar, knowledge, tmp, now)
        dskb = None
        if records:
            try:
                dskb, _twin = build_dataset_kb(knowledge, records, tmp, now)
            except Exception as exc:
                print(f"  [note] dataset fixture unavailable: {exc!r}")

        group_b(ar, knowledge, records)
        group_c(ar, kb, ids, now)
        group_d(ar, kb, ids)
        group_e(ar, ev, kb, ids)
        group_f(ar, knowledge, tmp, now)
        group_g(ar)
        group_h(ar, records)
        group_i(ar, kb, ids)
        group_j(ar, knowledge, tmp, now)
        group_k(ar)
        group_l(ar, kb, ids, dskb, records)
        group_m(ar, ev, kb, ids)
        group_o(ar, knowledge, tmp, now, kb, ids, dskb)
    except Exception as exc:                           # a crash mid-run still prints a table + exits non-zero
        import traceback
        traceback.print_exc()
        check("unexpected exception during the run", False, repr(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    group_n(ar, kb, ids, stores_before, tree_before)   # AFTER the temp dir is gone
    return _summary_and_exit()


def _summary_and_exit():
    """A SKIP is not a pass. Exit 1 on any skip as well as any failure.

    The old summary printed "N/M PASS -- AGENTIC-RAG SELFTEST OK" and exited 0 no matter how much had been
    skipped. Making sim/eval.py unimportable -- the exact mid-edit state this file's header anticipates, and
    an everyday occurrence while several workstreams edit concurrently -- silently skipped 8 of 83 checks
    (the whole eval-integration contract plus every dataset-derived assertion) and still printed a green
    last line with exit 0. A gate that reports OK with 10% of itself unexecuted is not a gate.
    """
    n_pass = sum(1 for _, s in _RESULTS if s == "PASS")
    n_fail = sum(1 for _, s in _RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s in _RESULTS if s == "SKIP")
    print("-" * 100)
    print("checks: %d total | %d PASS | %d FAIL | %d SKIP" % (len(_RESULTS), n_pass, n_fail, n_skip))
    if n_fail:
        print("FAILED: " + "; ".join(name for name, s in _RESULTS if s == "FAIL"))
    if n_skip:
        print("SKIPPED (counted as NOT OK -- the suite did not run in full):")
        for name, s in _RESULTS:
            if s == "SKIP":
                print("   - " + name)
    if n_fail or n_skip:
        print("%d/%d PASS, %d FAIL, %d SKIP -- AGENTIC-RAG SELFTEST NOT OK"
              % (n_pass, len(_RESULTS), n_fail, n_skip))
        print("=" * 100)
        sys.exit(1)
    print("%d/%d PASS -- AGENTIC-RAG SELFTEST OK" % (n_pass, len(_RESULTS)))
    print("=" * 100)
    sys.exit(0)


if __name__ == "__main__":
    main()
