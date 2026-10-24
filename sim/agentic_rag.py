"""Chunkless, index-free, provenance-ranked *agentic* retrieval over the agent's KnowledgeBase.

`rag.HybridRAG` treats retrieval as a preprocessing step: embed the query, cosine it against a cached vector per
entry, take top-k. That buys paraphrase recall at the price of an index that goes stale, an embedding round-trip on
every query, and a ranking signal (cosine) that throws away everything the knowledge base actually knows about a
memory — who wrote it, when, how confident it was, which kitchen it was proven in. This module takes the other
road, the one the corpus size justifies (~100 authored records, ~46 KB): retrieval becomes a **bounded, auditable
program over whole records**. Nothing is embedded, nothing is indexed, nothing can go stale.

Pieces:
  - :func:`normalize_text` / :func:`tokens` / :func:`is_persian` — the bilingual front door. NFKC plus Lucene-style
    Persian/Arabic folding (Arabic yeh/kaf -> Persian, teh-marbuta -> heh, harakat and tatweel stripped, ZWNJ ->
    space, Arabic-Indic and Persian digits -> ASCII), applied to the **query and the record**, plus a bilingual
    stopword list so `را` stops scoring like a content word.
  - :func:`kb_manifest` / :func:`kb_grep` / :func:`kb_read` — the glob / grep / read ladder, i.e. progressive
    disclosure. Level 1 is a one-line-per-record metadata manifest (~17.5 tokens/record, so at this corpus size the
    manifest *is* the index); Level 2 is a regex probe over whole record bodies returning matching LINES; Level 3
    expands the full provenance-tagged body of explicitly named ids only, under a character budget.
  - :func:`rank_key` / :func:`rank_hits` / :func:`utility` / :func:`quota_select` — ranking by **provenance**
    instead of cosine: source trust x confidence with a seven-day half-life x scene specificity x retrieval
    utility x a supersession penalty, every term separately weighted and separately reportable, plus a typed
    retrieval budget so three near-duplicate procedures cannot evict the lesson that prevents the failure.
  - :func:`lexical_pivot` / :func:`llm_pivot` / :func:`pivot` — query decomposition into **language-neutral
    AI2-THOR object types**. A regex cannot cross Persian -> an English-authored memory; an object type can,
    because both languages' commands denote the same identifier. The lexical pivot does it with zero model calls;
    the LLM pivot is one strict-JSON call, used only when the lexical one provably cannot bridge.
  - :func:`coverage` / :func:`sufficient` / :func:`verify_plan` / :class:`ProbeCache` — the deterministic
    sufficiency gate (never the model's own judgement), the deterministic plan verifier, and an exact-match probe
    cache keyed on the KB content hash (so invalidation is exact rather than heuristic).
  - :class:`AgenticRAG` — the program itself, exposing ``search()``/``recall_text()`` with the SAME shape as
    :class:`rag.HybridRAG` so no call site has to branch, plus ``dump()`` for the whole-knowledge-base control arm
    and ``last_telemetry()`` for the audit record of the last retrieval.

**Degradation contract.** Every failure path lands on :func:`keyword_fallback` (``kb.search`` only: no network, no
embeddings, no model) and stamps ``telemetry['degraded']``. ``--rag agentic`` must never be able to score worse
than ``--rag keyword`` for infrastructural reasons — only for retrieval-quality reasons. A missing provider, a
provider that raises, unparseable model JSON, a raising KnowledgeBase: all of them warn and keep going.

Stdlib-first, per CLAUDE.md: this module imports under a bare-stdlib Python and opens no socket at import time.
It imports NOTHING from the repo at module scope — `knowledge`, `rag`, `memory`, `improve` and `agent_llm` are
imported LAZILY inside the functions that need them, and every one of those imports is optional (a local fallback
covers it). numpy and chromadb are never imported at all: there is no dense half to compute. ``import agentic_rag``
therefore succeeds with no numpy / chromadb / Ollama present, and every public function below is usable offline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata

_log = logging.getLogger("agentic_rag")


def _warn(msg: str):
    _log.warning("agentic_rag: %s", msg)


# ==============================================================================================================
# Constants
# ==============================================================================================================
KINDS: tuple = ("procedures", "lessons", "facts")

#: Persian/Arabic orthographic folding, toward PERSIAN forms. The direction is arbitrary as long as the index side
#: and the query side agree — and here they do, because both go through :func:`normalize_text`.
FA_FOLD: dict = {
    # --- yeh family -> Persian yeh (U+06CC)
    "ي": "ی",   # Arabic yeh
    "ى": "ی",   # alef maksura
    "ے": "ی",   # yeh barree
    "ئ": "ی",   # yeh with hamza above
    # --- kaf family -> keheh (U+06A9)
    "ك": "ک",   # Arabic kaf
    "ڪ": "ک",   # swash kaf
    # --- heh family
    "ة": "ه",   # teh marbuta
    "ۀ": "ه",   # heh with yeh above
    "ہ": "ه",   # heh goal
    # --- alef family
    "آ": "ا",   # alef with madda
    "أ": "ا",   # alef with hamza above
    "إ": "ا",   # alef with hamza below
    "ٱ": "ا",   # alef wasla
    "ؤ": "و",   # waw with hamza above
    # --- joiners, bidi marks, decorations
    "ـ": "",         # tatweel (kashida)
    "‌": " ",        # ZWNJ -> space, so گوجه‌فرنگی and گوجه فرنگی tokenize identically
    "‍": "",         # ZWJ
    "‎": "", "‏": "", "؜": "", "﻿": "",   # LRM / RLM / ALM / BOM
    "ٔ": "", "ٕ": "",                               # free-standing hamza above/below
    # --- Arabic-Indic digits -> ASCII
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # --- Persian (extended Arabic-Indic) digits -> ASCII
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

#: Arabic diacritics (harakat). Stripped, NOT treated as a word boundary: `سَیب` must normalise to `سیب`.
HARAKAT: str = "ًٌٍَُِّْٰ"

_FOLD_TABLE: dict = str.maketrans({**FA_FOLD, **{c: None for c in HARAKAT}})

#: A COPY of ``knowledge._STOP``, deliberately not an import: `sim/knowledge.py` is owned by another workstream, so
#: a change there must show up here as a diff rather than silently moving this module's baseline.
STOP_EN: frozenset = frozenset({
    "the", "a", "an", "to", "in", "on", "of", "and", "it", "is", "with", "into", "then",
    "put", "go", "get", "please", "up", "down", "from", "at", "for", "this", "that",
})

#: Persian function words. Without these, `را` matches every Persian record against every Persian query.
STOP_FA: frozenset = frozenset({
    "را", "به", "در", "از", "که", "با", "این", "آن", "برای", "است", "بر", "تا", "هم", "یک",
    "کن", "کند", "بکن", "بده", "شود", "می", "ای", "های", "ها", "و", "یا", "اگر", "هر",
})

STOP_ALL: frozenset = STOP_EN | STOP_FA

#: Mirrors ``memory.PROVENANCE``: how much a memory's origin is trusted.
PROVENANCE: dict = {"human": 3, "demo": 2, "environment": 1, "prior": 0}

#: Seven days. Procedures and lessons are durable; ``memory.py``'s one-hour half-life is for *spatial* beliefs,
#: which decay because objects move — a lesson about needing a knife does not.
HALF_LIFE_KB: float = 7 * 24 * 3600.0

#: The five ranking terms. They sum to 1 and each is in [0,1], so ``{"sim": 1.0, ...}`` recovers plain similarity
#: ranking exactly — which is what makes every term individually ablatable.
RANK_WEIGHTS: dict = {"sim": 0.55, "prov": 0.10, "conf": 0.10, "util": 0.15, "scope": 0.10}

STALE_PENALTY: float = 0.5

MANIFEST_TITLE_CHARS: int = 70
DEFAULT_BUDGET_CHARS: int = 6000        # manifest + bodies, ~1.5-2k tokens
DEFAULT_BODY_CHARS: int = 1600          # expansion budget for kb_read
FULL_BUDGET_CHARS: int = 48000          # --rag full hard cap (~12k tok); overflow logs a loud warning
MAX_HOPS: int = 3

_TAGS_CHARS: int = 40                   # manifest tag-list truncation
_LINE_WINDOW: int = 120                 # kb_grep match-window width


# ==============================================================================================================
# Bilingual normalisation
# ==============================================================================================================
def normalize_text(text) -> str:
    """NFKC + Persian/Arabic orthographic folding + harakat/tatweel removal. Pure stdlib, ~0.5 us/command.

    Lucene's ``ArabicNormalizer`` + ``PersianNormalizer``, folded toward PERSIAN forms (the direction is arbitrary
    as long as index and query agree — and both sides of :func:`kb_grep` go through this function). Whitespace is
    collapsed, so ZWNJ folding cannot leave double spaces behind.

    Idempotent by construction: no folding output is itself a folding input, so
    ``normalize_text(normalize_text(x)) == normalize_text(x)``. ASCII input is returned unchanged apart from
    whitespace collapsing, which is what keeps :func:`tokens` byte-compatible with ``knowledge._tokens``.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(_FOLD_TABLE)
    return " ".join(s.split())


def tokens(text, stop: frozenset | None = None) -> set:
    """``normalize_text`` -> lowercase -> non-alphanumeric to space -> split -> drop `stop` and 1-char tokens.

    `stop` defaults to :data:`STOP_ALL` (bilingual). With ``stop=STOP_EN`` this is byte-compatible with
    ``knowledge._tokens`` on ASCII input, so the keyword half of ``--rag hybrid`` does not shift when `rag.py`
    starts normalising queries.
    """
    st = STOP_ALL if stop is None else stop
    norm = normalize_text(text)
    flat = "".join(c.lower() if c.isalnum() else " " for c in norm)
    return {w for w in flat.split() if w and w not in st and len(w) > 1}


def is_persian(text) -> bool:
    """True when >= 25% of the alphabetic characters are in the Arabic Unicode block (U+0600-U+06FF)."""
    s = str(text or "")
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return False
    fa = sum(1 for c in alpha if "؀" <= c <= "ۿ")
    return (fa / len(alpha)) >= 0.25


def _is_persian_term(term: str) -> bool:
    """Cheap per-term test: does `term` contain any Arabic-block character at all."""
    return any("؀" <= c <= "ۿ" for c in str(term or ""))


#: The word-character class used for BOTH sides of a whole-word probe. ``[^\W_]`` is "Unicode word character,
#: minus the underscore": letters and digits in every script (so Persian letters count) and NOT ``_`` and NOT
#: punctuation (so ARABIC COMMA U+060C, ARABIC SEMICOLON U+061B and ARABIC QUESTION MARK U+061F are boundaries).
#:
#: Two defects this replaces, both found by review and both regression-tested in `_agentic_rag_selftest.py`:
#:   1. ``\b`` treats ``_`` as a word character, so ``\bpick\b`` could not match the token ``pick`` inside the
#:      skill name ``pick_and_place`` — while ``knowledge.search`` (the ``--rag keyword`` floor this arm must be
#:      a superset of) splits on every non-alphanumeric and DOES match it. That broke the never-worse-than-keyword
#:      contract for infrastructural, not retrieval-quality, reasons.
#:   2. The previous Persian lookaround ``(?<![\w؀-ۿ])`` put the whole Arabic block — punctuation included — into
#:      the "word character" class, so a Persian term adjacent to Persian punctuation could never match. The
#:      shipped dataset contains three such records (``fp{1,2,3}-l2-heat-potato-fa``, ``…گرم کن، سپس…``), where
#:      the verb immediately before the comma failed to match its own record body.
_WORD_CHAR: str = r"[^\W_]"
_BOUNDARY_L: str = r"(?<!" + _WORD_CHAR + r")"
_BOUNDARY_R: str = r"(?!" + _WORD_CHAR + r")"


def _term_pattern(term: str, whole_word: bool = True):
    """Compiled, case-insensitive pattern matching one already-normalised `term`.

    ONE boundary strategy for both scripts — ``(?<![^\\W_])term(?![^\\W_])`` — because Python's ``\\b`` is wrong
    at both ends of this problem: it is unreliable at Arabic-script boundaries (the script has no cased word
    characters and the ZWNJ/harakat we just folded away used to sit on those boundaries), and it treats ``_`` as
    a word character, which hid whole-token matches inside underscore-joined skill names. See :data:`_WORD_CHAR`.
    """
    esc = re.escape(term)
    if not whole_word:
        return re.compile(esc, re.IGNORECASE | re.UNICODE)
    return re.compile(_BOUNDARY_L + esc + _BOUNDARY_R, re.IGNORECASE | re.UNICODE)


# --------------------------------------------------------------------------------------------------------------
# Local copies of KnowledgeBase helpers. `sim/knowledge.py` is owned by another workstream; these are COPIES on
# purpose, so a change there surfaces as a behavioural diff here instead of silently moving this module.
# --------------------------------------------------------------------------------------------------------------
def _normalize_command(text) -> str:
    """Copy of ``knowledge.normalize_command`` (lowercase, punctuation -> space, collapse whitespace)."""
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in str(text or "")).split())


def _plan_signature(plan) -> tuple:
    """Copy of ``knowledge.plan_signature``: the ordered (action, object) tuple identifying WHAT a plan does."""
    sig = []
    for s in (plan or []):
        if isinstance(s, dict):
            sig.append((str(s.get("action") or "").lower().strip(),
                        str(s.get("object") or "").lower().strip()))
    return tuple(sig)


def _error_class(error) -> str:
    """Copy of ``knowledge._error_class``: coarse deterministic error bucket used for lesson dedup."""
    s = str(error or "").lower()
    s = re.sub(r"[|]", " ", s)
    s = re.sub(r"[-+]?\d*\.?\d+", " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:48].strip()


def _entry_scope(entry: dict, default: str = "global") -> str:
    """An entry's effective scope: explicit ``scope``, else its ``scene``, else 'global'."""
    return str((entry or {}).get("scope") or (entry or {}).get("scene") or default)


def _in_scope(entry: dict, scene: str | None) -> bool:
    """Byte-identical to ``rag.HybridRAG._in_scope``: no scene filter -> everything; else 'global' or a match."""
    if not scene:
        return True
    sc = _entry_scope(entry)
    return sc == "global" or sc == scene


# ==============================================================================================================
# The three tools — manifest (glob), probe (grep), read
# ==============================================================================================================
def kb_entries(kb, scene: str | None = None, kinds: tuple = KINDS) -> list:
    """``[(kind, entry), ...]`` for every in-scope entry, in insertion order.

    The scope rule is byte-identical to ``rag.HybridRAG._in_scope``. `kb` is duck-typed: anything with a ``.data``
    mapping of ``kind -> [entry, ...]`` works, which is what keeps this module free of a `knowledge` import.
    """
    out = []
    data = getattr(kb, "data", None) or {}
    for kind in kinds:
        for e in (data.get(kind) or []):
            if isinstance(e, dict) and _in_scope(e, scene):
                out.append((kind, e))
    return out


def _entry_title(kind: str, entry: dict) -> str:
    """The record's headline text: its command (procedures/lessons) or its text (facts), whitespace-collapsed."""
    raw = entry.get("command") if kind in ("procedures", "lessons") else entry.get("text")
    if raw in (None, ""):
        raw = entry.get("text") or entry.get("command") or entry.get("lesson") or ""
    return " ".join(str(raw).split())


def manifest_line(kind: str, entry: dict, width: int = MANIFEST_TITLE_CHARS) -> str:
    """One terse, semantically distinct line for one record: ``#{id} {k} [{scope}] {title} | {tags}``.

    ``k`` is ``'P'|'L'|'F'``; ``scope`` is the entry's scope/scene/'global'; ``title`` is truncated to `width` with
    a trailing ``'...'`` when cut; ``tags`` is a comma-joined list truncated to 40 chars, and the ``' | tags'``
    suffix is omitted entirely when the record has none. Measured at ~17.5 tokens/record, so the whole 106-entry
    knowledge base costs ~1.9k tokens — at this corpus size progressive-disclosure Level 1 can *be* the index
    rather than a sample of it.
    """
    entry = entry or {}
    k = {"procedures": "P", "lessons": "L", "facts": "F"}.get(kind, "?")
    title = _entry_title(kind, entry)
    if len(title) > width:
        title = title[:width] + "..."
    line = f"#{entry.get('id')} {k} [{_entry_scope(entry)}] {title}"
    tags = ",".join(str(t) for t in (entry.get("tags") or []))
    if tags:
        line += " | " + tags[:_TAGS_CHARS]
    return line


def kb_manifest(kb, scene: str | None = None, kinds: tuple = KINDS, limit: int | None = None,
                budget_chars: int = DEFAULT_BUDGET_CHARS, rank_mode: str = "provenance",
                now: float | None = None) -> tuple:
    """-> ``(text, rows)``. Progressive-disclosure LEVEL 1: metadata only, never bodies.

    `rows` is ``[{"id", "kind", "scope", "line", "entry"}]`` ordered by :func:`rank_hits` under `rank_mode`, so
    that when the budget truncates it is always the LOWEST-ranked records that are dropped — never an arbitrary
    insertion-order tail. `text` is the joined lines capped at `budget_chars`; when it is truncated a final
    ``'... (+N more records not shown)'`` marker line is appended and `rows` is truncated to match (the marker is
    a rendering artefact and is deliberately not a row).
    """
    cands = [{"kind": kind, "entry": e, "n_match": 0, "matched": []}
             for kind, e in kb_entries(kb, scene=scene, kinds=kinds)]
    ranked = rank_hits(cands, scene=scene, now=now, mode=rank_mode, kb=kb)
    if limit:
        ranked = ranked[:int(limit)]

    rows, lines, used = [], [], 0
    dropped = 0
    for h in ranked:
        e = h["entry"]
        line = manifest_line(h["kind"], e)
        cost = len(line) + (1 if lines else 0)
        if used + cost > budget_chars:
            dropped = len(ranked) - len(rows)
            break
        lines.append(line)
        used += cost
        rows.append({"id": e.get("id"), "kind": h["kind"], "scope": _entry_scope(e), "line": line, "entry": e})
    text = "\n".join(lines)
    if dropped:
        text += ("\n" if text else "") + f"... (+{dropped} more records not shown)"
    return text, rows


def _entry_haystack(entry: dict) -> str:
    """Everything about one record that a probe may match, normalised as ONE string.

    Command/text/lesson/problem/why, then tags, then each plan step as ``"action object"`` — so a procedure whose
    command never contains the word ``pick_and_place`` is still found by probing for it. The whole concatenation
    goes through :func:`normalize_text` (not just the text fields), so Persian *tags* normalise too; on ASCII
    input that is a no-op, which keeps the English side byte-stable.
    """
    e = entry or {}
    parts = [str(e.get(f, "") or "") for f in ("command", "text", "lesson", "problem", "why")]
    parts += [str(t) for t in (e.get("tags") or [])]
    for s in (e.get("plan") or []):
        if isinstance(s, dict):
            parts.append(f"{s.get('action', '') or ''} {s.get('object', '') or ''}".strip())
    return normalize_text(" ".join(p for p in parts if p))


def _norm_terms(terms) -> list:
    """Normalise a str-or-iterable of terms into an ordered, de-duplicated list of non-empty strings."""
    if terms is None:
        return []
    if isinstance(terms, str):
        terms = [terms]
    out, seen = [], set()
    for t in terms:
        s = normalize_text(t).strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def kb_grep(kb, terms, scene: str | None = None, kinds: tuple = KINDS,
            limit: int = 20, whole_word: bool = True) -> list:
    """PROBE (progressive-disclosure LEVEL 2). Lexical match over WHOLE records; returns LINES, never bodies.

    `terms` is a str or an iterable of str. Every term is normalised and escaped; ASCII terms match on ``\\b``
    boundaries, Persian terms on Unicode lookarounds (see :func:`_term_pattern`). Because the haystack is
    normalised too, a query written with Arabic yeh finds a record written with Persian yeh — the defect that
    makes ``knowledge.search`` blind to half its own corpus.

    Returns ``[{"kind", "entry", "matched", "line", "n_match"}]`` where `line` is a 120-character window of the
    haystack centred on the first match, sorted by ``n_match`` desc then entry id asc (stable and deterministic),
    truncated to `limit`. Measured at 0.25 ms over 106 entries.
    """
    tlist = _norm_terms(terms)
    if not tlist:
        return []
    pats = [(t, _term_pattern(t, whole_word=whole_word)) for t in tlist]

    hits = []
    for kind, e in kb_entries(kb, scene=scene, kinds=kinds):
        hay = _entry_haystack(e)
        if not hay:
            continue
        matched, first = [], None
        for term, pat in pats:
            m = pat.search(hay)
            if m:
                matched.append(term)
                if first is None or m.start() < first:
                    first = m.start()
        if not matched:
            continue
        start = max(0, (first or 0) - _LINE_WINDOW // 2)
        hits.append({"kind": kind, "entry": e, "matched": matched,
                     "line": hay[start:start + _LINE_WINDOW], "n_match": len(matched)})
    hits.sort(key=lambda h: (-h["n_match"], _id_key(h["entry"])))
    return hits[:limit] if limit else hits


def _id_key(entry: dict):
    """Sort-stable numeric id (ids are ints in this KB; anything else sorts last, deterministically)."""
    try:
        return (0, int((entry or {}).get("id")))
    except Exception:
        return (1, str((entry or {}).get("id")))


def _default_render(kind: str, entry: dict) -> str:
    """One rendered memory line. A private copy of ``eval.render_hit_body``'s per-kind templates, VERBATIM.

    Byte-identical to `eval._render_hits`' inlined templates so a memory block built by this module stays
    comparable with the existing ``eval_*.json`` runs. `eval.build_memory_block` passes its own
    ``render_hit_body`` in via ``kb_read(render=...)`` so there is exactly one template in the live path.
    """
    e = entry or {}
    try:
        conf = float(e.get("confidence", 1.0))
    except Exception:
        conf = 1.0
    tag = f"_(src={e.get('source', '?')}, conf={conf:.2f})_"
    if kind == "procedures":
        steps = " ; ".join(f"{s.get('action')} {s.get('object') or ''}".strip()
                           for s in (e.get("plan") or []) if isinstance(s, dict))
        return f"- **worked before** -- \"{e.get('command')}\" -> {steps} {tag}"
    if kind == "lessons":
        return f"- **lesson** -- for \"{e.get('command')}\": {e.get('lesson')} {tag}"
    return f"- **fact** -- {e.get('text')} {tag}"


def _index_by_id(kb, kinds: tuple = KINDS) -> dict:
    """``{id: (kind, entry)}`` over the whole store (no scope filter — ids are explicit here)."""
    idx = {}
    data = getattr(kb, "data", None) or {}
    for kind in kinds:
        for e in (data.get(kind) or []):
            if isinstance(e, dict):
                idx[e.get("id")] = (kind, e)
    return idx


def kb_read(kb, ids, budget_chars: int = DEFAULT_BODY_CHARS, render=None) -> list:
    """LEVEL 3 expansion: full provenance-tagged bodies for EXPLICITLY NAMED ids only, under a budget.

    Returns ``[{"kind", "entry", "body"}]`` in the order `ids` were given, stopping *before* the cumulative body
    length would exceed `budget_chars` — a partially rendered body is never emitted. Unknown ids are skipped
    silently: they may be model-supplied and therefore hallucinated. `render` is a ``(kind, entry) -> str``
    callable defaulting to :func:`_default_render`, so `eval.py` can pass its own single source of truth.
    """
    rend = render or _default_render
    idx = _index_by_id(kb)
    out, used = [], 0
    for raw in (ids or []):
        found = idx.get(raw)
        if found is None:
            try:                                  # models emit "12" as often as 12
                found = idx.get(int(raw))
            except Exception:
                found = None
        if found is None:
            continue
        kind, entry = found
        try:
            body = rend(kind, entry)
        except Exception as exc:
            _warn(f"could not render entry {entry.get('id')!r}: {exc}")
            continue
        if used + len(body) > budget_chars:
            break
        out.append({"kind": kind, "entry": entry, "body": body})
        used += len(body)
    return out


def kb_fingerprint(kb) -> str:
    """First 16 hex chars of the sha256 of the whole store — the cache key that makes memoisation EXACT.

    The knowledge base is one small JSON file, so a content hash is both cheap (sub-millisecond at 46 KB) and
    *exactly* invalidating: the opposite of the embedding sidecar's situation, where a stale index is invisible
    until it silently returns a previous generation's vector for a different memory.
    """
    try:
        blob = json.dumps(getattr(kb, "data", {}) or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception as exc:
        _warn(f"could not fingerprint the knowledge base ({exc}); caching disabled for this call")
        return ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ==============================================================================================================
# Provenance-aware ranking
# ==============================================================================================================
def utility(entry: dict, util_sidecar: dict | None = None) -> float:
    """Laplace-smoothed retrieval utility in (0,1).

    ``(helpful + 1) / (helpful + harmful + 2)`` when the entry has a record in `util_sidecar` (written by
    ``rag.credit`` from the loop reports), else the reinforcement proxy ``1 - 1/(1 + occurrences + uses)``.

    The proxy **saturates** instead of growing without bound, which is the point: ``knowledge.search`` adds
    ``0.1 * uses`` to a raw overlap of 1-4 *inside* the scorer, before `rag.py`'s per-query min-max normalisation,
    so a procedure reinforced 30 times contributes +3.0 and rescales every other candidate to near zero.
    """
    e = entry or {}
    if util_sidecar:
        eid = e.get("id")
        rec = util_sidecar[str(eid)] if str(eid) in util_sidecar else util_sidecar.get(eid)
        if isinstance(rec, dict):
            try:
                h = float(rec.get("helpful", 0) or 0)
                x = float(rec.get("harmful", 0) or 0)
                return (h + 1.0) / (h + x + 2.0)
            except Exception:
                pass
    try:
        n = float(e.get("occurrences", 0) or 0) + float(e.get("uses", 0) or 0)
    except Exception:
        n = 0.0
    return 1.0 - 1.0 / (1.0 + max(0.0, n))


def rank_key(kind: str, entry: dict, scene: str | None, sim: float, now: float | None = None,
             util_sidecar: dict | None = None, superseded: bool = False,
             weights: dict | None = None) -> tuple:
    """-> ``(score, parts)``. Provenance ranking: five terms in [0,1], weights summing to 1.

    ::

        score = ( w_sim   * sim
                + w_prov  * PROVENANCE.get(source, 0) / 3
                + w_conf  * confidence * 0.5 ** ((now - ts) / HALF_LIFE_KB)
                + w_util  * utility(entry, util_sidecar)
                + w_scope * (1.0 if scope == scene else 0.6 if scope == 'global' else 0.0)
                ) * (STALE_PENALTY if superseded else 1.0)

    `weights` **REPLACES** the whole weight vector: any of the five keys it omits is taken as 0.0, so
    ``weights={"sim": 1.0}`` recovers plain similarity ranking exactly and ``weights={k: v for k, v in
    RANK_WEIGHTS.items() if k != "prov"}`` ablates provenance alone. (It used to *update* the defaults, which
    meant ``{"sim": 1.0}`` silently left prov/conf/util/scope at 0.10/0.10/0.15/0.10 for a weight sum of 1.45 —
    the documented "recovers plain similarity exactly" contract was false, and a caller ablating one term kept
    the other four. Review finding; regression-tested as F8.) Unknown keys are ignored.

    `parts` carries each term separately (plus the unpenalised ``base``) so the eval telemetry can show WHY a
    record ranked where it did — the auditability requirement applied to retrieval and not only to writes.
    """
    e = entry or {}
    if weights:
        w = {k: 0.0 for k in RANK_WEIGHTS}
        w.update({k: float(v) for k, v in weights.items() if k in RANK_WEIGHTS})
    else:
        w = dict(RANK_WEIGHTS)
    t_now = time.time() if now is None else float(now)

    prov = float(PROVENANCE.get(str(e.get("source") or ""), 0)) / 3.0
    try:
        conf0 = float(e.get("confidence", 1.0))
    except Exception:
        conf0 = 1.0
    try:
        ts = float(e.get("ts", t_now))
    except Exception:
        ts = t_now
    age = max(0.0, t_now - ts)
    decay = 0.5 ** (age / HALF_LIFE_KB) if HALF_LIFE_KB > 0 else 1.0
    conf = max(0.0, min(1.0, conf0)) * decay
    util = utility(e, util_sidecar)
    sc = _entry_scope(e)
    scope_term = 1.0 if (scene and sc == scene) else (0.6 if sc == "global" else 0.0)
    s = max(0.0, min(1.0, float(sim or 0.0)))

    base = (w["sim"] * s + w["prov"] * prov + w["conf"] * conf + w["util"] * util + w["scope"] * scope_term)
    score = base * (STALE_PENALTY if superseded else 1.0)
    parts = {"sim": round(s, 4), "prov": round(prov, 4), "conf": round(conf, 4),
             "util": round(util, 4), "scope": round(scope_term, 4),
             "superseded": bool(superseded), "base": base, "weights": w, "kind": kind}
    return score, parts


def _supersession_map(kb) -> dict:
    """``{normalized command -> newest procedure entry}``, built ONCE per :func:`rank_hits` call.

    Implements the design note's stated-but-unimplemented intent: when a later *success* exists for the same
    command, an older lesson about failing it is probably obsolete — but only if the procedure is also at least as
    confident and at least as reinforced, so a lesson seen many times is never demoted by one lucky run.
    """
    out = {}
    data = getattr(kb, "data", None) or {}
    for p in (data.get("procedures") or []):
        if not isinstance(p, dict):
            continue
        key = _normalize_command(p.get("command"))
        if not key:
            continue
        cur = out.get(key)
        try:
            ts = float(p.get("ts", 0) or 0)
        except Exception:
            ts = 0.0
        if cur is None or ts > float(cur.get("ts", 0) or 0):
            out[key] = p
    return out


def _is_superseded(kind: str, entry: dict, sup: dict) -> bool:
    if kind != "lessons" or not sup:
        return False
    p = sup.get(_normalize_command((entry or {}).get("command")))
    if not p:
        return False

    def _f(d, key, default=0.0):
        try:
            return float(d.get(key, default) or default)
        except Exception:
            return float(default)

    return (_f(p, "ts") > _f(entry, "ts")
            and _f(entry, "confidence", 1.0) <= _f(p, "confidence", 1.0)
            and _f(entry, "occurrences", 1.0) <= _f(p, "occurrences", 1.0))


def rank_hits(hits, scene: str | None = None, now: float | None = None, mode: str = "provenance",
              util_sidecar: dict | None = None, kb=None, weights: dict | None = None) -> list:
    """Rank a :func:`kb_grep` / :func:`kb_manifest` hit list; returns a NEW list of shallow-copied hits.

    ``mode='plain'``      -> sort by ``(n_match desc, id asc)``. The control arm; byte-reproducible.
    ``mode='provenance'`` -> sort by :func:`rank_key` desc, with ``sim = n_match / max(n_match)``.
    ``mode='none'``       -> insertion order, unchanged (for measuring how much ordering matters at all).

    ``mode`` may also carry the ``'provenance+utility'`` spelling used by ``--rank``; the utility term is always
    computed, the suffix only signals that a `util_sidecar` is expected. When `kb` is given, lesson supersession
    is evaluated once per call. Each returned hit gains ``score``, ``rank_parts``, ``keyword`` (= sim) and
    ``cosine`` (always 0.0 — this retriever never embeds), preserving the HybridRAG hit-dict contract.
    """
    src = list(hits or [])
    if mode == "none":
        return [dict(h) for h in src]

    nmax = max([float(h.get("n_match", 0) or 0) for h in src], default=0.0) or 1.0
    sup = _supersession_map(kb) if (kb is not None and mode != "plain") else {}
    out = []
    for h in src:
        e = h.get("entry") or {}
        kind = h.get("kind", "")
        sim = float(h.get("n_match", 0) or 0) / nmax
        superseded = _is_superseded(kind, e, sup)
        if mode == "plain":
            score, parts = float(sim), {"sim": round(sim, 4), "mode": "plain", "superseded": False}
        else:
            score, parts = rank_key(kind, e, scene, sim, now=now, util_sidecar=util_sidecar,
                                    superseded=superseded, weights=weights)
        nh = dict(h)
        nh["score"] = round(float(score), 4)
        nh["keyword"] = round(float(sim), 4)
        nh["cosine"] = 0.0
        nh["rank_parts"] = parts
        out.append(nh)

    if mode == "plain":
        out.sort(key=lambda h: (-float(h.get("n_match", 0) or 0), _id_key(h.get("entry") or {})))
    else:
        out.sort(key=lambda h: (-float(h.get("score", 0.0)), _id_key(h.get("entry") or {})))
    return out


def quota_select(hits, k: int = 5, quota: tuple = (2, 2, 1), mode: str = "typed") -> list:
    """Typed retrieval budget: procedures / lessons / facts get their own slots before any backfill.

    ``mode='mixed'`` -> the first `k` distinct records in rank order. ``mode='typed'`` -> up to ``quota[0]``
    procedures, ``quota[1]`` lessons and ``quota[2]`` facts in rank order, then unused slots backfilled from the
    remainder in rank order until `k` records are chosen. Duplicates by entry id are removed FIRST, so one record
    can never occupy two slots. The output preserves rank order.

    Rationale: Reflexion caps its reflection buffer at Omega = 1-3 for exactly this reason, and A-MemGuard's
    dual-memory structure consults lessons BEFORE acting — a flat k=5 lets three near-duplicate procedures evict
    the lesson that would have prevented the failure.
    """
    seen, pool = set(), []
    for h in (hits or []):
        eid = (h.get("entry") or {}).get("id")
        key = eid if eid is not None else id(h)
        if key in seen:
            continue
        seen.add(key)
        pool.append(h)
    k = max(0, int(k))
    if mode != "typed":
        return pool[:k]

    caps = {"procedures": int(quota[0]) if len(quota) > 0 else 0,
            "lessons": int(quota[1]) if len(quota) > 1 else 0,
            "facts": int(quota[2]) if len(quota) > 2 else 0}
    chosen, taken = set(), {"procedures": 0, "lessons": 0, "facts": 0}
    for i, h in enumerate(pool):
        if len(chosen) >= k:
            break
        kind = h.get("kind", "")
        if kind in caps and taken[kind] < caps[kind]:
            chosen.add(i)
            taken[kind] += 1
    for i, _h in enumerate(pool):                  # backfill unused slots in rank order
        if len(chosen) >= k:
            break
        chosen.add(i)
    return [h for i, h in enumerate(pool) if i in chosen]


# ==============================================================================================================
# Query decomposition — the bilingual object-type pivot
# ==============================================================================================================
PIVOT_SYSTEM: str = (
    "You extract search keys for a robot's memory lookup. Reply with ONLY a JSON object of exactly this shape "
    "(no prose, no markdown):\n"
    '{"types": ["<AI2-THOR object type>", ...], "verbs": ["<action name>", ...], "ids": [<int>, ...]}\n'
    "types must be English AI2-THOR object types (Fridge, Knife, Apple, Microwave, ...). verbs must be action "
    "names from the robot's skill list (pickup, open, close, slice, heat, cool, cook, wash, pick_and_place, "
    "put_in, toggle_on, toggle_off). ids are record numbers from the MEMORY INDEX you want expanded, or []. "
    "Return at most 6 types, 4 verbs and 4 ids. Never invent object types that are not in the OBSERVATION."
)

#: Persian surface form -> canonical AI2-THOR object type. DATA, hand-authored once from
#: `sim/dataset/commands.jsonl` and versioned here. Multi-word keys are matched as phrases (ZWNJ folds to a space,
#: so `گوجه‌فرنگی` and `گوجه فرنگی` are the same key), longest first, and a matched span is consumed — which is why
#: `سیب زمینی` yields Potato and not Apple.
#:
#: **DISCLOSURE — this table is FITTED TO THE EVALUATION SET.** It was written by reading the Persian half of
#: `sim/dataset/commands.jsonl`, so every Persian retrieval number measured with it in place is an IN-SAMPLE
#: number and must be reported as one. Ablating the table drops Persian recall@1 from 0.824 to 0.000 with 51/51
#: empty result sets: normalisation, provenance ranking, the manifest, the gate and the hops contribute exactly
#: zero to the cross-lingual gain — the lexicon IS the bridge. The self-test used to *assert* full coverage of
#: the dataset's gold types, which guaranteed the fit and made the regression suite complicit in it; it now
#: REPORTS coverage and asserts the held-out path instead (check H4).
#:
#: The honest question ("does this generalise to Persian words not in the table?") is answered by
#: :func:`held_out_type_pairs` + `bench_rag.py --section E`'s **E4 held-out-vocabulary** run, which deletes from
#: the lexicon exactly the keys each query uses and re-measures. That is the number a write-up may quote as
#: generalisation; 0.824 is not.
FA_TYPE_LEXICON: dict = {
    "یخچال": "Fridge", "فریزر": "Fridge",
    "مایکروویو": "Microwave", "ماکروویو": "Microwave", "مایکروفر": "Microwave",
    "سیب زمینی": "Potato", "سیب": "Apple",
    "گوجه فرنگی": "Tomato", "گوجه": "Tomato",
    "نان": "Bread", "چاقو": "Knife", "کاهو": "Lettuce",
    "تخم مرغ": "Egg", "تخم‌مرغ": "Egg",
    "لیوان": "Mug", "ماگ": "Mug", "فنجان": "Cup", "کاسه": "Bowl", "بشقاب": "Plate",
    "سینک": "SinkBasin", "ظرفشویی": "SinkBasin", "دستشویی": "SinkBasin",
    "شیر آب": "Faucet", "شیر": "Faucet",
    "پیشخوان": "CounterTop", "کانتر": "CounterTop", "میز": "CounterTop",
    "توستر": "Toaster", "نان برشته کن": "Toaster",
    "چراغ": "LightSwitch", "لامپ": "LightSwitch", "کلید برق": "LightSwitch", "کلید چراغ": "LightSwitch",
    "کتاب": "Book",
    "تلفن همراه": "CellPhone", "تلفن": "CellPhone", "موبایل": "CellPhone", "گوشی": "CellPhone",
    "بطری شراب": "WineBottle", "شراب": "WineBottle", "بطری": "Bottle",
    "تابه": "Pan", "ماهیتابه": "Pan", "قابلمه": "Pot",
    "اجاق": "StoveBurner", "گاز": "StoveBurner", "فر": "StoveBurner",
    "سطل زباله": "GarbageCan", "سطل آشغال": "GarbageCan", "زباله": "GarbageCan", "آشغال": "GarbageCan",
    "کابینت": "Cabinet", "کشو": "Drawer", "قاشق": "Spoon", "چنگال": "Fork",
    "نمکدان": "SaltShaker", "فلفل": "PepperShaker", "قهوه ساز": "CoffeeMachine",
    "حوله": "DishSponge", "اسفنج": "DishSponge", "صندلی": "Chair", "سبد": "Bowl",
}

#: English surface form -> canonical AI2-THOR object type. Covers the dataset's gold types plus the receptacles
#: they name; `agent_llm.TYPE_ALIASES` is layered on top of this at call time (lazily imported).
EN_TYPE_LEXICON: dict = {
    "apple": "Apple", "book": "Book", "bowl": "Bowl", "bread": "Bread", "loaf": "Bread",
    "cell phone": "CellPhone", "cellphone": "CellPhone", "phone": "CellPhone", "mobile": "CellPhone",
    "cup": "Cup", "egg": "Egg", "fridge": "Fridge", "refrigerator": "Fridge",
    "knife": "Knife", "lettuce": "Lettuce",
    "light switch": "LightSwitch", "lightswitch": "LightSwitch", "light": "LightSwitch", "lamp": "LightSwitch",
    "microwave": "Microwave", "mug": "Mug", "pan": "Pan", "pot": "Pot", "potato": "Potato",
    "toaster": "Toaster", "tomato": "Tomato",
    "wine bottle": "WineBottle", "winebottle": "WineBottle", "wine": "WineBottle", "bottle": "Bottle",
    "counter top": "CounterTop", "countertop": "CounterTop", "counter": "CounterTop",
    "plate": "Plate", "dish": "Plate",
    "sink": "SinkBasin", "sink basin": "SinkBasin", "basin": "SinkBasin",
    "faucet": "Faucet", "tap": "Faucet",
    "garbage can": "GarbageCan", "garbage": "GarbageCan", "trash": "GarbageCan", "trash can": "GarbageCan",
    "bin": "GarbageCan", "stove": "StoveBurner", "burner": "StoveBurner", "stove burner": "StoveBurner",
    "cabinet": "Cabinet", "drawer": "Drawer", "spoon": "Spoon", "fork": "Fork", "kettle": "Kettle",
    "sponge": "DishSponge", "chair": "Chair", "shelf": "Shelf",
}

#: Persian verb/phrase -> canonical skill name (the same vocabulary `PLANNER_SYSTEM` uses).
FA_VERB_LEXICON: dict = {
    "باز کن": "open", "بازکن": "open", "باز کردن": "open",
    "ببند": "close", "بستن": "close", "درش را ببند": "close",
    "بردار": "pickup", "برداشتن": "pickup", "بگیر": "pickup",
    "بگذار": "put_in", "بذار": "put_in", "قرار بده": "put_in", "بگذارید": "put_in",
    "روشن کن": "toggle_on", "روشن": "toggle_on",
    "خاموش کن": "toggle_off", "خاموش": "toggle_off",
    "برش بزن": "slice", "برش": "slice", "خرد کن": "slice", "قاچ کن": "slice",
    "گرم کن": "heat", "داغ کن": "heat", "داغ": "heat",
    "خنک کن": "cool", "سرد کن": "cool", "خنک": "cool",
    "بپز": "cook", "بپزید": "cook", "پختن": "cook",
    "بشوی": "wash", "بشور": "wash", "شستن": "wash",
    "تمیز کن": "clean", "پر کن": "fill", "خالی کن": "empty",
    "بیاور": "pick_and_place", "سرو کن": "pick_and_place", "ببر": "pick_and_place",
    "پیدا کن": "search_for", "پیدا کردن": "search_for", "پیداکن": "search_for",
    "بگرد": "search_for", "جستجو کن": "search_for", "جست و جو کن": "search_for",
    "دنبال": "search_for", "کجاست": "search_for", "کجا است": "search_for", "پیدا": "search_for",
}

#: English verb/phrase -> canonical skill name.
EN_VERB_LEXICON: dict = {
    "open": "open", "close": "close", "shut": "close",
    "pick up": "pickup", "pickup": "pickup", "pick": "pickup", "take": "pickup", "grab": "pickup",
    "put": "put_in", "place": "put_in", "set": "put_in",
    "move": "pick_and_place", "bring": "pick_and_place", "serve": "pick_and_place", "carry": "pick_and_place",
    "turn on": "toggle_on", "switch on": "toggle_on", "toggle on": "toggle_on",
    "turn off": "toggle_off", "switch off": "toggle_off", "toggle off": "toggle_off",
    "slice": "slice", "cut": "slice", "chop": "slice",
    "heat": "heat", "warm": "heat", "microwave": "heat", "toast": "heat",
    "cool": "cool", "chill": "cool",
    "cook": "cook", "wash": "wash", "rinse": "wash", "clean": "clean",
    "fill": "fill", "empty": "empty", "drop": "drop", "throw": "throw",
    "find": "search_for", "search": "search_for", "search for": "search_for", "look for": "search_for",
    "locate": "search_for", "where is": "search_for", "where's": "search_for",
}

#: Coordination markers (L2) and goal/abstraction markers (L3) for :func:`estimate_level`.
_L2_MARKERS: tuple = ("and then", "then", "after", "and", "و", "سپس", "بعد")
_L3_MARKERS: tuple = ("make", "prepare", "tidy", "clean up", "serve", "breakfast", "something",
                      "درست کن", "آماده کن", "مرتب کن", "سرو کن", "خوراکی")
#: INSTRUMENT markers. `Slice the bread with the knife` names one verb but is an ordered chain (pick up the
#: knife, then slice), i.e. L2 — the surface cue is the instrumental preposition, not a second verb.
_L2_INSTRUMENT_MARKERS: tuple = ("with", "using", "با", "توسط")


def _phrase_keys(mapping: dict) -> list:
    """Normalised (key, value) pairs ordered longest-phrase-first, so `سیب زمینی` wins over `سیب`."""
    pairs = []
    for k, v in (mapping or {}).items():
        nk = normalize_text(k).lower()
        if nk:
            pairs.append((nk, v))
    pairs.sort(key=lambda kv: (-len(kv[0].split()), -len(kv[0])))
    return pairs


def _merged_pairs(base: list, extras: dict) -> list:
    """`base` (precomputed at import) plus any caller/`agent_llm` extras, re-ordered longest-phrase-first."""
    if not extras:
        return base
    pairs = _phrase_keys(extras) + list(base)
    pairs.sort(key=lambda kv: (-len(kv[0].split()), -len(kv[0])))
    return pairs


def _match_lexicon(surface: str, pairs, consumed: list | None = None) -> list:
    """Find every lexicon entry present in the normalised `surface`, longest phrase first, consuming spans.

    Returns ``[(position, value), ...]`` sorted by position. `consumed` (a list of ``[start, end]`` spans) is
    shared across lexicons by the caller so a span matched as an object type is not re-read as a verb. The
    ``key not in surface`` pre-filter is a strictly necessary condition for the (literal) pattern to match, and it
    keeps a ~200-entry lexicon at a handful of regex calls per command instead of 200.
    """
    spans = consumed if consumed is not None else []
    found = []
    for key, val in pairs:
        if key not in surface:
            continue
        for m in _term_pattern(key).finditer(surface):
            s, e = m.span()
            if any(not (e <= cs or s >= ce) for cs, ce in spans):
                continue
            spans.append([s, e])
            found.append((s, val))
    found.sort(key=lambda pv: pv[0])
    return found


def _ordered_unique(values) -> list:
    out, seen = [], set()
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _scene_type_lexicon(scene_types) -> dict:
    """Surface forms for a live ``objectType`` list: the type itself plus its space-separated CamelCase form."""
    out = {}
    for t in (scene_types or []):
        s = str(t or "").strip()
        if not s:
            continue
        out[s.lower()] = s
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s).lower()
        if spaced != s.lower():
            out[spaced] = s
    return out


#: Precomputed at import so a command costs a handful of substring checks rather than a full lexicon sort.
_TYPE_PAIRS: list = _phrase_keys({**EN_TYPE_LEXICON, **FA_TYPE_LEXICON})
_VERB_PAIRS: list = _phrase_keys({**EN_VERB_LEXICON, **FA_VERB_LEXICON})

#: The surface forms the BUILT-IN lexicons already resolve. Discovered extras (a live AI2-THOR ``objectType``
#: list, `agent_llm.TYPE_ALIASES`) may only ADD keys, never redefine one of these — otherwise a kitchen that
#: contains both ``Sink`` and ``SinkBasin`` rewrites the English surface form "sink" from the canonical
#: ``SinkBasin`` to the literal ``Sink``, whole-word grep stops matching records that name ``SinkBasin``, and the
#: two halves of the same bilingual command stop resolving to the same language-neutral type (the Persian side
#: has no scene-derived surface form and keeps ``SinkBasin``). `eval.py` always passes live `scene_types` and
#: `bench_rag.py` never does, so before this rule the benchmark measured a strictly different retriever from the
#: one the eval harness ran. Review finding; regression-tested as H5b.
_BUILTIN_TYPE_KEYS: frozenset = frozenset(k for k, _v in _TYPE_PAIRS)


def held_out_type_pairs(surface: str = "", drop_keys=None, base: list | None = None) -> list:
    """A COPY of the object-type lexicon with some keys deleted — the generalisation harness for the disclosure
    on :data:`FA_TYPE_LEXICON`.

    Two ways to ask the question, both deterministic and both side-effect free (the module-level lexicon is
    never mutated):

    - ``held_out_type_pairs(surface=command)`` — leave-THIS-QUERY's-vocabulary-out: every lexicon key that
      occurs in `command` is removed, so the pivot must bridge that command with vocabulary it has never been
      shown. This is the strict answer to "what happens on a Persian noun the table does not contain".
    - ``held_out_type_pairs(drop_keys=[...])`` — remove a named fold, for k-fold sweeps.

    Pass the result to :func:`lexical_pivot` / :func:`pivot` / :class:`AgenticRAG` as `type_pairs`.
    """
    pairs = list(base if base is not None else _TYPE_PAIRS)
    drop = {normalize_text(k).lower() for k in (drop_keys or []) if str(k or "").strip()}
    if surface:
        norm = normalize_text(surface).lower()
        drop |= {key for key, _v in pairs if key and _term_pattern(key).search(norm)}
    if not drop:
        return pairs
    return [(k, v) for k, v in pairs if k not in drop]


def lexical_pivot(command: str, scene_types=None, alias_map: dict | None = None,
                  type_pairs: list | None = None) -> dict:
    """-> ``{"types": [...], "verbs": [...], "ids": [], "source": "lex"}``. ZERO LLM calls, zero network.

    The FA->EN bridge, done with data instead of a model: Persian and English surface forms both resolve onto the
    same **language-neutral AI2-THOR object type**, which is an affordance specific to embodied agents that generic
    text RAG does not have. Lexicons are consulted longest-phrase-first over the normalised command with matched
    spans consumed, then `scene_types` (the live objectType list `eval.py` already computes) and
    ``agent_llm.TYPE_ALIASES`` (imported lazily; optional) are layered on. Verbs resolve the same way.

    **Discovered extras may only ADD surface forms, never redefine a built-in one** (see
    :data:`_BUILTIN_TYPE_KEYS`): the canonical lexicon is the language-neutral contract both halves of a bilingual
    pair are resolved against, and a live scene list must not be able to move it under one half only. An explicit
    `alias_map` from the caller is the one exception — that is deliberate, stated intent — and it is applied last.

    `type_pairs` overrides the built-in object-type lexicon wholesale (see :func:`held_out_type_pairs`); it is
    how the held-out-vocabulary measurement is taken without mutating module state.

    Tokens that resolve to nothing appear in neither list — :func:`pivot` falls back to raw content tokens when
    both come back empty.
    """
    surface = normalize_text(command).lower()
    if not surface:
        return {"types": [], "verbs": [], "ids": [], "source": "lex"}
    base_pairs = _TYPE_PAIRS if type_pairs is None else list(type_pairs)

    discovered: dict = {}
    try:                                   # optional: the planner's own alias table, and its Persian sibling
        import agent_llm as _al            # lazy: keeps `import agentic_rag` repo-free
        for extra in (getattr(_al, "TYPE_ALIASES", None), getattr(_al, "FA_TYPE_ALIASES", None)):
            if isinstance(extra, dict):
                discovered.update(extra)
    except Exception:
        pass                               # agent_llm is optional; the built-in lexicons stand alone
    discovered.update(_scene_type_lexicon(scene_types))
    # Drop every discovered surface form the BUILT-IN lexicon defines: additive only. Filtering against the full
    # built-in key set (not against `base_pairs`) is deliberate — under a held-out `type_pairs` a deleted key must
    # not sneak back in through the live scene list, or the generalisation measurement would leak its own answer.
    extras = {k: v for k, v in discovered.items()
              if normalize_text(k).lower() not in _BUILTIN_TYPE_KEYS}
    if alias_map:                          # explicit caller intent: this one IS allowed to override
        extras.update(alias_map)

    spans: list = []
    types = _ordered_unique(v for _p, v in _match_lexicon(surface, _merged_pairs(base_pairs, extras), spans))
    verbs = _ordered_unique(v for _p, v in _match_lexicon(surface, _VERB_PAIRS, spans))
    return {"types": types, "verbs": verbs, "ids": [], "source": "lex"}


def _parse_json_object(content: str) -> dict:
    """Tolerant JSON-object scrape. Prefers ``agent_llm.parse_json_object`` when that module exposes it."""
    try:
        import agent_llm as _al
        fn = getattr(_al, "parse_json_object", None)
        if callable(fn):
            obj = fn(content)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", str(content or ""), re.S)
        if not m:
            raise ValueError(f"no JSON object in model reply: {str(content)[:200]}")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model reply was not a JSON object")
    return obj


def llm_pivot(command: str, provider, model: str, manifest_text: str = "", observation: str = "",
              timeout: float = 30.0, options: dict | None = None) -> dict:
    """ONE single-shot strict-JSON call decomposing the command into object types, verbs and record ids.

    NOT function calling and NOT multi-turn: BFCL puts 4B-class models at ~35% multi-turn tool-calling accuracy,
    while this repo's ``parse_plan`` already proves the model survives a single JSON reply. Temperature and seed
    are pinned so the retrieval program stays deterministic given a fixed model.

    Raises nothing: on ANY failure (no provider, connection error, prose instead of JSON, wrong shape) it returns
    ``{"types": [], "verbs": [], "ids": [], "source": "llm-failed", "error": str}`` and the caller degrades.
    """
    out = {"types": [], "verbs": [], "ids": [], "source": "llm"}
    if provider is None or not hasattr(provider, "chat"):
        return {**out, "source": "llm-failed", "error": "no provider"}
    user = (f"MEMORY INDEX:\n{manifest_text}\n\nOBSERVATION:\n{observation}\n\nCOMMAND: {command}")
    try:
        content = provider.chat(model, [{"role": "system", "content": PIVOT_SYSTEM},
                                        {"role": "user", "content": user}],
                                timeout=timeout,
                                options=options or {"temperature": 0.0, "seed": 0})
        obj = _parse_json_object(content)
    except Exception as exc:
        _warn(f"llm pivot unavailable ({exc}); falling back to the lexical pivot")
        return {**out, "source": "llm-failed", "error": str(exc)}

    def _strs(v, n):
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in (v or []) if str(x).strip()][:n] if isinstance(v, (list, tuple)) else []

    out["types"] = _strs(obj.get("types"), 6)
    out["verbs"] = _strs(obj.get("verbs"), 4)
    ids = []
    raw_ids = obj.get("ids")
    if isinstance(raw_ids, (list, tuple)):
        for x in raw_ids[:4]:
            try:
                ids.append(int(x))
            except Exception:
                continue
    out["ids"] = ids
    return out


def pivot(command: str, provider=None, model: str | None = None, scene_types=None, mode: str = "auto",
          manifest_text: str = "", observation: str = "", timeout: float = 30.0,
          type_pairs: list | None = None) -> dict:
    """Dispatcher over the query-decomposition strategies. ``mode`` in ``{'off','lex','llm','auto'}``.

    ``'off'``  -> no decomposition at all (the probe runs on raw content tokens).
    ``'lex'``  -> :func:`lexical_pivot` only; zero model calls, zero network.
    ``'llm'``  -> :func:`llm_pivot`, falling back to the lexical pivot when it fails (``source='llm-failed->lex'``).
    ``'auto'`` -> lexical first, and a model call ONLY when the lexical pivot found no object types AND the command
    is Persian — the one case where a lexical probe provably cannot cross into an English-authored memory. On the
    English half of the dataset ``auto`` therefore costs zero extra calls, which is exactly the asymmetry the
    thesis should report.

    Always returns a ``"terms"`` list: types + verbs, or the raw content tokens when both are empty.
    """
    m = (mode or "auto").strip().lower()
    if m == "off":
        res = {"types": [], "verbs": [], "ids": [], "source": "off"}
        res["terms"] = sorted(tokens(command))
        return res

    if m == "llm":
        res = llm_pivot(command, provider, model or "", manifest_text=manifest_text,
                        observation=observation, timeout=timeout)
        if res.get("source") == "llm-failed":
            lex = lexical_pivot(command, scene_types=scene_types, type_pairs=type_pairs)
            res = {**lex, "source": "llm-failed->lex", "error": res.get("error", "")}
    elif m == "auto":
        res = lexical_pivot(command, scene_types=scene_types, type_pairs=type_pairs)
        if not res["types"] and is_persian(command):
            got = llm_pivot(command, provider, model or "", manifest_text=manifest_text,
                            observation=observation, timeout=timeout)
            if got.get("source") == "llm":
                res = {"types": _ordered_unique(got["types"]),
                       "verbs": _ordered_unique(list(res["verbs"]) + list(got["verbs"])),
                       "ids": got["ids"], "source": "llm"}
            else:
                res = {**res, "source": "llm-failed->lex", "error": got.get("error", "")}
    else:
        res = lexical_pivot(command, scene_types=scene_types, type_pairs=type_pairs)

    terms = _ordered_unique(list(res.get("types") or []) + list(res.get("verbs") or []))
    res["terms"] = terms or sorted(tokens(command))
    return res


# ==============================================================================================================
# Gate, verifier, cache
# ==============================================================================================================
def coverage(hits, terms, scene: str | None, observation: str = "") -> dict:
    """What the current hit set does and does not cover. Pure; no LLM, no network, sub-millisecond.

    -> ``{"in_scene", "terms_hit", "terms_missed", "types_in_obs", "types_uncovered"}``. A term counts as covered
    when some hit matched it explicitly (``hit['matched']``) or when the term occurs in that hit's record body.
    ``types_in_obs`` are the queried terms the planner can actually see in the OBSERVATION — the only ones whose
    absence from memory is worth another hop.
    """
    tlist = _norm_terms(terms)
    hs = list(hits or [])
    in_scene = sum(1 for h in hs if _in_scope(h.get("entry") or {}, scene))

    obs_norm = normalize_text(observation)
    covered = set()
    for h in hs:
        for t in (h.get("matched") or []):
            covered.add(normalize_text(t).lower())
    uncovered_check = [t for t in tlist if t.lower() not in covered]
    if uncovered_check:
        hays = [_entry_haystack(h.get("entry") or {}) for h in hs]
        for t in uncovered_check:
            pat = _term_pattern(t)
            if any(pat.search(hay) for hay in hays):
                covered.add(t.lower())

    terms_hit = [t for t in tlist if t.lower() in covered]
    terms_missed = [t for t in tlist if t.lower() not in covered]
    types_in_obs = [t for t in tlist if obs_norm and _term_pattern(t).search(obs_norm)]
    types_uncovered = [t for t in types_in_obs if t.lower() not in covered]
    # Did the query decomposition produce ANY term naming something the agent can actually SEE? This is the
    # "is there a bridge at all" signal, and it is separate from "is every visible thing covered". A Persian
    # command whose vocabulary the lexicon does not contain yields only raw Persian tokens, none of which can
    # appear in an English OBSERVATION -- so `types_in_obs` came back empty, `types_uncovered` was vacuously
    # empty too, and the gate declared even an EMPTY result set sufficient. The gate was blind to exactly the
    # case the LLM pivot exists to rescue, which is the mechanical reason the multi-hop loop spent 0 hops on
    # 102/102 benchmark queries and `agentic` was byte-identical to `probe`. Review finding.
    return {"in_scene": in_scene, "terms_hit": terms_hit, "terms_missed": terms_missed,
            "types_in_obs": types_in_obs, "types_uncovered": types_uncovered,
            "observation_present": bool(obs_norm), "bridged": (not obs_norm) or bool(types_in_obs)}


def sufficient(hits, terms, scene: str | None, observation: str = "", min_hits: int = 1) -> tuple:
    """-> ``(bool, reason)``. The DETERMINISTIC sufficiency gate. Stop when BOTH hold:

    (a) at least `min_hits` hits are in scope (``'global'`` or this scene); AND
    (b) every queried term that names an object type **present in the OBSERVATION** is covered by some hit — a
        term the planner can already see does not also need to be remembered; AND
    (c) the query decomposition **bridged at all** — when there IS an observation, at least one queried term
        names something in it. (c) is a review fix: without it, a query whose terms are all in a script the
        observation is not written in satisfied (b) vacuously, and the gate never asked for a hop in the one
        situation the hop exists for.

    Termination is this gate plus a hard hop cap, never the model's own judgement: Search-R1's learned sufficiency
    needs training we do not have and CRAG's evaluator is a fine-tuned T5 we will not ship. `reason` is the
    :func:`coverage` dict plus the two booleans, and goes straight into the telemetry so the gate's stop-rate is
    reportable.
    """
    cov = coverage(hits, terms, scene, observation=observation)
    enough = cov["in_scene"] >= int(min_hits)
    covered = not cov["types_uncovered"]
    bridged = bool(cov.get("bridged", True))
    ok = bool(enough and covered and bridged)
    reason = dict(cov)
    reason["enough_hits"] = enough
    reason["types_covered"] = covered
    reason["ok"] = ok
    return ok, reason


def verify_plan(plan, observation_types, action_names) -> list:
    """-> ``[{"step", "kind", "value"}]`` violations. ZERO LLM calls.

    Enforces deterministically the two rules ``agent_llm.PLANNER_SYSTEM`` only states in prose ("Use ONLY object
    TYPES that appear in the OBSERVATION list. Never invent objects."): an action outside the skill library is
    ``unknown_action``; an object/receptacle type absent from the OBSERVATION is ``unknown_object``. `step` is
    1-based. Comparison is case- and alias-insensitive.

    `action_names` and `observation_types` are supplied BY THE CALLER (`eval.py` builds them from
    ``skills.skills_manifest()`` and its ``_NAV_HELPERS``) — this module imports neither. An empty
    `observation_types` disables the object check rather than flagging everything: no observation is not evidence
    of absence.
    """
    if not isinstance(plan, (list, tuple)):
        return []
    names = {str(a).strip().lower() for a in (action_names or []) if str(a).strip()}
    obs = set()
    for t in (observation_types or []):
        s = str(t or "").strip()
        if s:
            obs.add(s.lower())
            obs.add(str(_canonical_type(s) or s).lower())

    out = []
    for i, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        if names and action and action.lower() not in names:
            out.append({"step": i, "kind": "unknown_action", "value": action})
        if not obs:
            continue
        for field in ("object", "to"):
            val = step.get(field)
            if val in (None, "", False):
                continue
            cand = str(val).strip()
            canon = str(_canonical_type(cand) or cand)
            if cand.lower() not in obs and canon.lower() not in obs:
                out.append({"step": i, "kind": "unknown_object", "value": cand})
    return out


def _canonical_type(name):
    """Canonical AI2-THOR type for a surface form, via the built-in lexicons and (lazily) `agent_llm`."""
    if not name:
        return None
    key = normalize_text(name).strip().lower()
    if not key:
        return None
    try:
        import agent_llm as _al
        for extra in (getattr(_al, "TYPE_ALIASES", None), getattr(_al, "FA_TYPE_ALIASES", None)):
            if isinstance(extra, dict) and key in extra:
                return extra[key]
    except Exception:
        pass
    return EN_TYPE_LEXICON.get(key) or FA_TYPE_LEXICON.get(key) or str(name).strip()


class ProbeCache:
    """Exact-match memoisation of probe results, keyed on ``(kb_fingerprint, scene, kinds, whole_word, terms)``.

    NOT a semantic cache. An exact-match cache cannot change *what* is retrieved, so it cannot contaminate an
    ablation whose dependent variable is retrieval quality; a similarity-threshold cache silently could, which is
    why §14 rejects it. It exists because the harness runs the same 102 commands across models, scenes and arms.
    **Disable it for timing runs** (``--agentic-cache off``) or the benchmark measures a dict lookup.

    Thread-safe (a lock around a bounded FIFO/LRU map), so the webapp can share one retriever across handlers.
    """

    def __init__(self, maxsize: int = 512):
        self.maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._map: dict = {}
        self._order: list = []
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: tuple):
        with self._lock:
            if key in self._map:
                self._hits += 1
                try:
                    self._order.remove(key)
                    self._order.append(key)
                except ValueError:
                    pass
                return self._map[key]
            self._misses += 1
            return None

    def put(self, key: tuple, value) -> None:
        with self._lock:
            if key not in self._map:
                self._order.append(key)
            self._map[key] = value
            while len(self._order) > self.maxsize:
                old = self._order.pop(0)
                self._map.pop(old, None)
                self._evictions += 1

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses,
                    "evictions": self._evictions, "size": len(self._map)}

    def clear(self) -> None:
        with self._lock:
            self._map.clear()
            self._order.clear()


# ==============================================================================================================
# Level policy
# ==============================================================================================================
def _any_marker(surface: str, markers) -> bool:
    return any(_term_pattern(normalize_text(m).lower()).search(surface) for m in markers)


def estimate_level(command: str, piv: dict | None = None, type_pairs: list | None = None) -> str:
    """-> ``'L1' | 'L2' | 'L3'``. Deterministic, bilingual, LLM-free heuristic over the NORMALISED command.

    Rules, in order:
      1. a goal/abstraction marker, or ZERO concrete object types                      -> **L3**
      2. a coordination marker AND >= 2 distinct action verbs                          -> **L2**
      3. a coordination marker AND >= 3 distinct object types (``X and Y into Z``)     -> **L3**
      4. an instrumental marker AND >= 2 types AND >= 1 verb (``slice X with Y``)      -> **L2**
      5. otherwise                                                                     -> **L1**

    Rules 3 and 4 are review fixes. The two-rule version scored **0.794** on the 102-record dataset with every
    error one-directional (9/30 L2 and 12/30 L3 demoted to L1, nothing ever over-estimated), which silently gave
    21 higher-level records the atomic retrieval budget — 0 hops and the smallest quota — so the multi-hop loop
    this arm exists to test never ran for them. The five-rule version, together with the word-boundary fix in
    :func:`_term_pattern` (which is what finally lets ``گرم کن،`` match its own verb), scores **1.000** on all
    102 records, EN and FA alike.

    **1.000 IS AN IN-SAMPLE FIT, NOT AN ACCURACY.** The marker lists in :data:`_L2_MARKERS` /
    :data:`_L3_MARKERS` / :data:`_L2_INSTRUMENT_MARKERS` were authored and then *revised* by reading the errors
    this function made on `sim/dataset/commands.jsonl`. A perfect score on the set you tuned against measures
    nothing except that the tuning converged; the honest reading is "level estimation is not currently a source
    of error in the reported numbers", and NOT "level estimation is solved". No held-out command set exists, so
    no generalisation figure can be quoted. ``--agentic-level oracle`` (read the dataset label) and
    ``--agentic-level off`` (one uniform policy) bracket this arm from above and below; with the estimator at
    1.000 the est-vs-oracle gap is 0 by construction, which is itself only an in-sample statement.

    This exists so the retriever does NOT read ``rec.level`` by default: a deployed agent never receives the
    dataset's label, so reading it would be an oracle.

    `piv` optionally supplies an already-computed :func:`lexical_pivot` result, so the retriever pays for the
    lexicon scan once per command rather than twice.
    """
    surface = normalize_text(command).lower()
    if not surface:
        return "L1"
    piv = piv or lexical_pivot(command, type_pairs=type_pairs)
    n_verbs = len(set(piv["verbs"]))
    n_types = len(set(piv["types"]))
    if _any_marker(surface, _L3_MARKERS) or n_types == 0:
        return "L3"
    has_coord = _any_marker(surface, _L2_MARKERS)
    if has_coord and n_verbs >= 2:
        return "L2"
    if has_coord and n_types >= 3:
        return "L3"
    if n_types >= 2 and n_verbs >= 1 and _any_marker(surface, _L2_INSTRUMENT_MARKERS):
        return "L2"
    return "L1"


#: Retrieval is level-conditioned, per the thesis's first-class L1/L2/L3 hierarchy: nothing to decompose at L1
#: (the useful memory is a *lesson*, and a big block only invites context rot on a 4B model); an ordered chain at
#: L2 benefits from a stored procedure and one refine hop; only abstract goals justify a second hop and facts.
#:
#: hops / k / budget are **FRACTIONS of the caller's own setting**, not absolute caps. That is a review fix: as
#: absolute caps, ``--agentic-hops 3`` was a no-op (L3 capped at 2 while ``MAX_HOPS`` is 3, so argparse offered a
#: choice that could not take effect), ``--agentic-budget`` could only ever SHRINK the block (6000/20000/40000 all
#: produced the identical 695-char block), and the documented L1 k=3 / L2 k=5 / L3 k=6 was dead in every eval run
#: because `eval.build_memory_block` always passes an explicit ``k=5``. As fractions, every knob is monotone in
#: its CLI value and the level policy still bites: at the shipped defaults (hops=2, k=5, budget=6000) the
#: fractions reproduce the original table exactly — L1 0 hops / k 3 / 2400 chars, L2 1 / 5 / 4800, L3 2 / 6 /
#: 6000 — while ``--agentic-hops 3`` now genuinely reaches 3 at L3 and ``--agentic-budget 12000`` genuinely
#: doubles the block. `quota` stays absolute: it is a per-KIND shape, not a magnitude.
LEVEL_POLICY: dict = {
    #      hop_frac  k_frac  quota      budget_frac      (defaults hops=2 k=5 budget=6000 -> the original table)
    "L1": {"hop_frac": 0.0, "k_frac": 0.6, "quota": (1, 2, 0), "budget_frac": 0.4},
    "L2": {"hop_frac": 0.5, "k_frac": 1.0, "quota": (2, 2, 1), "budget_frac": 0.8},
    "L3": {"hop_frac": 1.0, "k_frac": 1.2, "quota": (2, 2, 2), "budget_frac": 1.0},
}


def level_budget(level: str, hops: int, k: int, budget_chars: int) -> dict:
    """Resolve :data:`LEVEL_POLICY`'s fractions against the caller's own hops / k / budget.

    -> ``{"hops", "k", "quota", "budget_chars"}``, all absolute. ``hops`` is floored (L2 of a 1-hop budget is 0
    hops, not a rounded-up 1) and additionally clamped to :data:`MAX_HOPS`; ``k`` and ``budget_chars`` are
    rounded and floored at 1.
    """
    pol = LEVEL_POLICY.get(level, LEVEL_POLICY["L2"])
    return {"hops": min(int(MAX_HOPS), int(max(0, int(hops)) * float(pol["hop_frac"]))),
            "k": max(1, int(round(max(1, int(k)) * float(pol["k_frac"])))),
            "quota": tuple(pol["quota"]),
            "budget_chars": max(1, int(round(int(budget_chars) * float(pol["budget_frac"]))))}


# ==============================================================================================================
# Compaction and the token proxy
# ==============================================================================================================
def compact(hits, budget_chars: int = DEFAULT_BUDGET_CHARS, render=None) -> list:
    """Context compaction of the selected hits: dedup, merge, drop duplicates, then truncate to a budget.

    Near-duplicate **lessons** (same normalised command + error class) collapse into the highest-ranked one,
    carrying an ``(xN)`` marker in ``why_line`` so nothing is silently discarded; **procedures** whose plan
    signature duplicates a higher-ranked one are dropped; the survivors are truncated to `budget_chars` of
    rendered body, lowest-ranked first. Models hit context *cliffs* rather than degrading smoothly, and semantic
    distinctness matters more than raw length — three phrasings of one lesson is length without distinctness.
    """
    rend = render or _default_render
    out, seen_ids = [], set()
    lesson_at: dict = {}
    proc_sigs = set()
    for h in (hits or []):
        e = h.get("entry") or {}
        eid = e.get("id")
        if eid is not None and eid in seen_ids:
            continue
        kind = h.get("kind", "")
        if kind == "lessons":
            key = (_normalize_command(e.get("command")), _error_class(e.get("error_key", e.get("problem"))))
            if key in lesson_at:
                prev = lesson_at[key]
                prev["_merged"] = int(prev.get("_merged", 1)) + 1
                prev["why_line"] = f"(x{prev['_merged']})"
                continue
        elif kind == "procedures":
            sig = _plan_signature(e.get("plan"))
            if sig and sig in proc_sigs:
                continue
            proc_sigs.add(sig)
        nh = dict(h)
        if eid is not None:
            seen_ids.add(eid)
        if kind == "lessons":
            lesson_at[(_normalize_command(e.get("command")),
                       _error_class(e.get("error_key", e.get("problem"))))] = nh
        out.append(nh)

    kept, used = [], 0
    for h in out:
        try:
            body = rend(h.get("kind", ""), h.get("entry") or {})
        except Exception:
            body = ""
        if used + len(body) > budget_chars and kept:
            break
        kept.append(h)
        used += len(body)
    for h in kept:
        h.pop("_merged", None)
    return kept


def est_tokens(text: str) -> int:
    """Deterministic stdlib token PROXY — stated openly as a proxy, never as a tokenizer.

    ``ceil(n_ascii / 4 + n_nonascii / 1.5)``, calibrated against the measured 32.5 tok/record (EN) and
    46.3 tok/record (FA) figures: Persian costs far more tokens per character under a BPE trained on English.
    When `tiktoken` happens to be importable `bench_rag.py` prints both columns side by side; this module never
    imports it.
    """
    s = str(text or "")
    n_ascii = sum(1 for c in s if ord(c) < 128)
    n_other = len(s) - n_ascii
    return int(math.ceil(n_ascii / 4.0 + n_other / 1.5))


# ==============================================================================================================
# The floor
# ==============================================================================================================
def keyword_fallback(kb, query: str, scene: str | None = None, k: int = 5) -> list:
    """The floor every degradation path lands on: ``kb.search`` -> scope filter -> top-k. No network, no model.

    Shaped exactly like a `HybridRAG` hit (``{kind, score, keyword, cosine: 0.0, entry}``), so a caller cannot
    tell the difference structurally — which is the whole point of the never-worse-than-keyword contract. This is
    what ``--rag keyword`` would have produced, modulo the query-side normalisation applied here.
    """
    try:
        raw = kb.search(normalize_text(query), limit=10_000)
    except Exception as exc:
        _warn(f"keyword fallback failed ({exc}); returning no memories")
        return []
    scoped = [h for h in (raw or []) if _in_scope(h.get("entry") or {}, scene)]
    kw_max = max([float(h.get("score", 0.0) or 0.0) for h in scoped], default=0.0) or 1.0
    out = []
    for h in scoped[:max(0, int(k))]:
        kw = float(h.get("score", 0.0) or 0.0) / kw_max
        out.append({"kind": h.get("kind"), "score": round(kw, 4), "keyword": round(kw, 4),
                    "cosine": 0.0, "entry": h.get("entry") or {}})
    return out


# ==============================================================================================================
# The retriever
# ==============================================================================================================
_TELEMETRY_KEYS = ("mode", "hops", "llm_calls", "pivot_source", "terms", "manifest_rows", "manifest_chars",
                   "probe_ms", "rank_ms", "retrieval_ms", "gate_stopped", "gate_reason", "block_chars",
                   "est_tokens", "cache", "degraded", "level", "level_source", "retrieved_ids",
                   "rank_parts_top")


def _blank_telemetry(mode: str) -> dict:
    return {"mode": mode, "hops": 0, "llm_calls": 0, "pivot_source": None, "terms": [],
            "manifest_rows": 0, "manifest_chars": 0, "probe_ms": 0.0, "rank_ms": 0.0, "retrieval_ms": 0.0,
            "gate_stopped": None, "gate_reason": None, "block_chars": 0, "est_tokens": 0, "cache": None,
            "degraded": None, "level": None, "level_source": None, "retrieved_ids": [], "rank_parts_top": {}}


class AgenticRAG:
    """Chunkless, index-free, provenance-ranked retrieval over a KnowledgeBase — a bounded PROGRAM, not an agent.

    Exposes :meth:`search` / :meth:`recall_text` with the SAME shape as :class:`rag.HybridRAG`, so no call site has
    to branch, plus :meth:`dump` for the whole-knowledge-base control arm and :meth:`last_telemetry` for the audit
    record. The retrieval program is: normalise -> lexical pivot -> probe whole records -> rank by provenance ->
    typed quota -> sufficiency gate; and only if the gate says the block is inadequate, up to `hops` bounded model
    calls that decompose the command into object types and probe again. **Total model calls never exceed `hops`**
    (the pivot's call IS a hop), and any exception anywhere lands on :func:`keyword_fallback`.
    """

    def __init__(self, kb=None, path: str = "knowledge_base.json", provider=None, model: str | None = None,
                 hops: int = 2, k: int = 5, pivot_mode: str = "auto", rank_mode: str = "provenance",
                 quota_mode: str = "typed", gate: bool = True, cache: bool = True, compact: bool = True,
                 level_policy: str = "est", budget_chars: int = DEFAULT_BUDGET_CHARS,
                 util_path: str | None = None, timeout: float = 30.0, weights: dict | None = None,
                 type_pairs: list | None = None):
        """`kb=None` lazily builds a `KnowledgeBase` from `path` (mirroring ``HybridRAG.__init__``).

        `util_path` defaults to ``<kb stem>_utility.json`` and is READ-ONLY here (only ``rag.credit`` writes it);
        a missing or unreadable file is tolerated silently, so the utility term simply falls back to the
        reinforcement proxy.
        """
        if kb is None:
            import knowledge          # lazy: keeps `import agentic_rag` pure-stdlib and repo-free
            kb = knowledge.KnowledgeBase(path=path)
        self.kb = kb
        self.provider = provider
        self.model = model
        self.hops = max(0, int(hops))
        self.k = max(1, int(k))
        self.pivot_mode = (pivot_mode or "auto").strip().lower()
        self.rank_mode = (rank_mode or "provenance").strip().lower()
        self.quota_mode = (quota_mode or "typed").strip().lower()
        self.gate = bool(gate)
        self.compact = bool(compact)
        self.level_policy = (level_policy or "est").strip().lower()
        self.budget_chars = int(budget_chars)
        self.timeout = float(timeout)
        self.weights = dict(weights) if weights else None
        # None = the shipped (dataset-fitted) object-type lexicon. A list overrides it wholesale; see
        # `held_out_type_pairs` and the DISCLOSURE on FA_TYPE_LEXICON — this is the hook the held-out
        # Persian-vocabulary measurement uses, and it never mutates module state.
        self.type_pairs = list(type_pairs) if type_pairs is not None else None
        self.cache_on = bool(cache)
        self.cache = ProbeCache() if self.cache_on else None
        self.util_path = util_path or (os.path.splitext(getattr(kb, "path", path) or path)[0] + "_utility.json")
        self.util = self._load_utility()
        self._lock = threading.Lock()
        self._telemetry = _blank_telemetry(self.mode)

    # ------------------------------------------------------------------ small helpers
    @property
    def mode(self) -> str:
        """The ablation arm this object implements: ``'probe'`` with no hop budget, else ``'agentic'``."""
        return "probe" if self.hops == 0 else "agentic"

    def _load_utility(self) -> dict:
        """Read the retrieval-credit sidecar — but ONLY under ``rank_mode='provenance+utility'``.

        This gate is what makes ``--rank provenance`` and ``--rank provenance+utility`` two different arms
        instead of two spellings of one. Previously the sidecar was loaded unconditionally and the class
        docstring rationalised it ("the utility term is always computed, the suffix only signals that a
        util_sidecar is expected"), which meant the ablation could not vary anything even once `rag.credit`
        existed. Now:

          ``provenance``          -> utility term = the reinforcement proxy ``1 - 1/(1 + occurrences + uses)``
          ``provenance+utility``  -> utility term = measured retrieval credit, Laplace-smoothed

        Best-effort in every other respect: a missing file is the normal case on a fresh knowledge base.
        """
        if "utility" not in self.rank_mode:
            return {}
        try:
            if self.util_path and os.path.exists(self.util_path):
                with open(self.util_path, "r", encoding="utf-8") as fh:
                    blob = json.load(fh)
                if isinstance(blob, dict):
                    return blob.get("utility", blob) or {}
        except Exception as exc:
            _warn(f"could not read utility sidecar {self.util_path}: {exc}; using the reinforcement proxy")
        return {}

    def _level_for(self, command: str, level: str | None, piv: dict | None = None) -> tuple:
        """-> ``(level, level_source)``. 'oracle' trusts a caller-supplied label; 'est' never reads one."""
        if self.level_policy == "off":
            return (level or "L2"), "off"
        if self.level_policy == "oracle" and level in LEVEL_POLICY:
            return level, "oracle"
        return estimate_level(command, piv=piv, type_pairs=self.type_pairs), "est"

    def _policy(self, level: str, k: int | None = None) -> dict:
        """The level's retrieval policy resolved against THIS retriever's hops / k / budget.

        ``--agentic-level off`` is the uniform control: the caller's settings verbatim, no level conditioning.
        `k` lets the caller's per-call ``search(k=...)`` be the base the level fraction scales, so the documented
        L1 k=3 / L2 k=5 / L3 k=6 is live in the eval path (where ``k=5`` is always passed explicitly) instead of
        being silently overridden by it.
        """
        base_k = int(k) if k else self.k
        if self.level_policy == "off":
            return {"hops": self.hops, "k": max(1, base_k), "quota": (2, 2, 1),
                    "budget_chars": self.budget_chars}
        return level_budget(level, self.hops, base_k, self.budget_chars)

    def clear_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear()

    def last_telemetry(self) -> dict:
        """The audit record for the LAST :meth:`search` (or :meth:`dump`). Every key of §5.8 is always present."""
        with self._lock:
            return dict(self._telemetry)

    def _publish(self, tele: dict) -> dict:
        for key in _TELEMETRY_KEYS:
            tele.setdefault(key, None)
        with self._lock:
            self._telemetry = dict(tele)
        return tele

    # ------------------------------------------------------------------ the probe (cached)
    def _probe(self, terms, scene, kinds=KINDS, whole_word=True, limit=200) -> list:
        """One :func:`kb_grep` call, memoised on the KB's content hash when caching is on.

        `limit` is deliberately far above any `k`: the probe's job is to produce the CANDIDATE POOL, and that pool
        is what has to remain a superset of what ``--rag keyword`` would have matched (verified over all 102
        dataset commands). Ranking, the typed quota and compaction then narrow it — those are retrieval-quality
        decisions, which is the only kind of difference this arm is allowed to have against the keyword floor.
        """
        key = None
        if self.cache is not None:
            key = (kb_fingerprint(self.kb), scene, tuple(kinds), bool(whole_word),
                   tuple(_norm_terms(terms)), int(limit))
            got = self.cache.get(key)
            if got is not None:
                return list(got)
        hits = kb_grep(self.kb, terms, scene=scene, kinds=kinds, limit=limit, whole_word=whole_word)
        if key is not None:
            self.cache.put(key, list(hits))
        return hits

    # ------------------------------------------------------------------ the program
    def search(self, query: str, scene: str | None = None, k: int | None = None, alpha: float | None = None,
               *, level: str | None = None, observation: str = "", model: str | None = None,
               scene_types=None) -> list:
        """-> up to `k` hits ``{kind, score, keyword, cosine, entry, matched, hop, rank_parts}``.

        `alpha` is ACCEPTED AND IGNORED (there is no dense half) purely so this object is drop-in polymorphic with
        ``HybridRAG.search``; `model` overrides the constructor model per record, because the eval harness sweeps
        ``--models`` and the pivot must use the same model as the planner.

        Program: hop 0 is fully deterministic (normalise -> lexical pivot -> probe -> provenance rank -> typed
        quota -> gate). While the gate is unsatisfied and hops remain, ONE strict-JSON model call re-decomposes the
        command seeded with the uncovered terms, the probe re-runs, and the gate is re-evaluated. The hop cap is
        ``min(self.hops, MAX_HOPS, level_budget(level, self.hops, ...)['hops'])`` and is unconditional. Any exception anywhere
        degrades to :func:`keyword_fallback` with ``telemetry['degraded']`` set.
        """
        t0 = time.perf_counter()
        tele = _blank_telemetry(self.mode)
        try:
            # ---- hop 0: deterministic. The LEXICAL pivot only — an LLM pivot IS a hop and is budgeted as one.
            piv = pivot(query, provider=None, model=model or self.model or "", scene_types=scene_types,
                        mode=("off" if self.pivot_mode == "off" else "lex"), type_pairs=self.type_pairs)
            tele["pivot_source"] = piv.get("source")
            # The level estimate always uses a LEXICAL pivot, even under `--agentic-pivot off`: the two are
            # separate ablation axes and switching the pivot off must not silently make every command look L3.
            lvl, lvl_src = self._level_for(query, level, piv=(piv if self.pivot_mode != "off" else None))
            # The caller's k is the BASE the level fraction scales, not an override of it: `k` stays the
            # caller's contract with HybridRAG.search (double it and you get roughly twice the memories at every
            # level), while L1 still retrieves less than L3 as the design says. Before this fix `k` won outright
            # and the documented L1 k=3 / L2 k=5 / L3 k=6 was dead in every eval run, because
            # `eval.build_memory_block` always passes an explicit k=5.
            pol = self._policy(lvl, k=k)
            tele["level"], tele["level_source"] = lvl, lvl_src
            k_eff = max(1, int(pol["k"]))
            budget = int(pol["budget_chars"])
            hop_cap = min(self.hops, MAX_HOPS, int(pol["hops"]))
            mdl = model or self.model or ""

            # The raw content tokens are ALWAYS part of the probe: they are exactly what `knowledge.search`
            # matches on, so probing for them is what makes this arm a superset of `--rag keyword` (the
            # never-worse-than-keyword contract) rather than merely a different retriever.
            terms = _ordered_unique(list(piv.get("terms") or []) + sorted(tokens(query)))
            tele["terms"] = list(terms)

            t_probe = time.perf_counter()
            raw_hits = self._probe(terms, scene)
            tele["probe_ms"] = round((time.perf_counter() - t_probe) * 1000.0, 3)

            t_rank = time.perf_counter()
            ranked = rank_hits(raw_hits, scene=scene, mode=self.rank_mode, util_sidecar=self.util,
                               kb=self.kb, weights=self.weights)
            sel = quota_select(ranked, k=k_eff, quota=tuple(pol["quota"]), mode=self.quota_mode)
            tele["rank_ms"] = round((time.perf_counter() - t_rank) * 1000.0, 3)
            # first_seen: which hop DISCOVERED each record. Kept outside the hit dicts because quota_select and
            # rank_hits rebuild them every hop, and "found at hop 1" is an audit fact about the record, not about
            # this particular copy of its hit.
            first_seen = {(h.get("entry") or {}).get("id"): 0 for h in raw_hits}
            for h in sel:
                h["hop"] = 0

            # Gate OFF means "always spend the full hop budget" — that is what isolates the gate's saving as its
            # own measurement, so the loop below must not be short-circuited by a satisfied gate that is not
            # being consulted. gate_stopped is then None, never False: nothing was asked.
            ok, reason = sufficient(sel, terms, scene, observation=observation) if self.gate \
                else (False, {"gate": "off"})
            tele["gate_stopped"], tele["gate_reason"] = (bool(ok) if self.gate else None), reason

            # ---- hops 1..N: bounded, one strict-JSON model call each, seeded with what is still uncovered.
            manifest_text = ""
            hop = 0
            while (not ok) and hop < hop_cap:
                if not self._should_spend_hop(query, piv):
                    break
                if not manifest_text:
                    manifest_text, _rows = kb_manifest(self.kb, scene=scene, budget_chars=budget,
                                                       rank_mode=self.rank_mode)
                    tele["manifest_chars"] = len(manifest_text)
                hop += 1
                tele["llm_calls"] += 1
                missed = (reason or {}).get("terms_missed") or []
                seed = query if not missed else f"{query}\n(missing: {', '.join(missed)})"
                got = llm_pivot(seed, self.provider, mdl, manifest_text=manifest_text,
                                observation=observation, timeout=self.timeout)
                if got.get("source") != "llm":
                    tele["pivot_source"] = "lex+llm-failed"
                    tele["degraded"] = f"llm_pivot: {got.get('error', 'failed')}"
                    break
                tele["pivot_source"] = "lex+llm"
                new_terms = _ordered_unique(list(got.get("types") or []) + list(got.get("verbs") or []))
                terms = _ordered_unique(list(terms) + new_terms)
                tele["terms"] = list(terms)
                more = self._probe(new_terms, scene) if new_terms else []
                more += kb_read(self.kb, got.get("ids") or [], budget_chars=budget)
                merged = {}
                for h in list(raw_hits) + list(more):
                    e = h.get("entry") or {}
                    prev = merged.get(e.get("id"))
                    if prev is None or h.get("n_match", 0) > prev.get("n_match", 0):
                        merged[e.get("id")] = {"kind": h.get("kind"), "entry": e,
                                               "matched": h.get("matched") or [],
                                               "line": h.get("line", ""),
                                               "n_match": int(h.get("n_match", 1) or 1)}
                raw_hits = list(merged.values())
                for eid in merged:
                    first_seen.setdefault(eid, hop)
                ranked = rank_hits(raw_hits, scene=scene, mode=self.rank_mode, util_sidecar=self.util,
                                   kb=self.kb, weights=self.weights)
                sel = quota_select(ranked, k=k_eff, quota=tuple(pol["quota"]), mode=self.quota_mode)
                for h in sel:
                    h["hop"] = first_seen.get((h.get("entry") or {}).get("id"), hop)
                ok, reason = sufficient(sel, terms, scene, observation=observation) if self.gate \
                    else (False, {"gate": "off"})
                tele["gate_stopped"], tele["gate_reason"] = (bool(ok) if self.gate else None), reason
            tele["hops"] = hop

            if self.compact:
                sel = compact(sel, budget_chars=budget)
            else:
                sel = kb_read_trim(sel, budget)

            tele["manifest_rows"] = len(kb_entries(self.kb, scene=scene))
            block = "\n".join(_default_render(h.get("kind", ""), h.get("entry") or {}) for h in sel)
            tele["block_chars"] = len(block)
            tele["est_tokens"] = est_tokens(block)
            tele["retrieved_ids"] = [(h.get("entry") or {}).get("id") for h in sel]
            tele["rank_parts_top"] = (sel[0].get("rank_parts") if sel else {}) or {}
            tele["cache"] = self.cache.stats() if self.cache is not None else None
            tele["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            self._publish(tele)
            return sel[:k_eff]
        except Exception as exc:                       # never worse than keyword for INFRASTRUCTURAL reasons
            _warn(f"agentic retrieval failed ({exc}); degrading to keyword-only")
            tele["degraded"] = f"agentic: {exc}"
            hits = keyword_fallback(self.kb, query, scene=scene, k=int(k or self.k))
            tele["retrieved_ids"] = [(h.get("entry") or {}).get("id") for h in hits]
            tele["cache"] = self.cache.stats() if self.cache is not None else None
            tele["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            self._publish(tele)
            return hits

    def _should_spend_hop(self, query: str, piv: dict) -> bool:
        """Whether an LLM hop is allowed at all, given ``--agentic-pivot`` and ``--agentic-gate``.

        ``off``/``lex`` never spend one. ``llm`` always may. ``auto`` spends one only when the lexical pivot found
        no object types AND the command is Persian — the one case a lexical probe provably cannot bridge — EXCEPT
        with the gate disabled, where the arm's definition is "always spend the full hop budget" so that the gate's
        saving can be isolated as its own measurement.
        """
        if self.provider is None:
            return False
        if self.pivot_mode in ("off", "lex"):
            return False
        if self.pivot_mode == "llm" or not self.gate:
            return True
        return (not (piv.get("types") or [])) and is_persian(query)

    # ------------------------------------------------------------------ the control arm
    def dump(self, scene: str | None = None, budget_chars: int = FULL_BUDGET_CHARS) -> list:
        """``--rag full``: EVERY in-scope record as hits, in insertion order. No ranking, no probe, no LLM.

        Insertion order on purpose: an append-only knowledge base then EXTENDS a cached prompt prefix instead of
        invalidating it. (The self-improving loop still breaks that: ``upsert_lesson``'s in-place reinforcement
        mutates an existing record, so CAG and the loop are structurally opposed — a finding worth reporting.)
        Overflowing `budget_chars` logs a loud warning and truncates; hits are stamped ``rag_full`` so no report
        can ever conflate this control with a retrieval arm.
        """
        t0 = time.perf_counter()
        tele = _blank_telemetry("full")
        hits, used, dropped = [], 0, 0
        for kind, e in kb_entries(self.kb, scene=scene):
            body = _default_render(kind, e)
            if used + len(body) > budget_chars:
                dropped += 1
                continue
            used += len(body)
            hits.append({"kind": kind, "entry": e, "score": 0.0, "keyword": 0.0, "cosine": 0.0,
                         "matched": [], "hop": 0, "rank_parts": {}, "rag_full": True})
        if dropped:
            _warn(f"--rag full: knowledge base exceeds the {budget_chars}-char budget; "
                  f"{dropped} record(s) DROPPED — the 'stuff everything' control is no longer complete")
        tele["manifest_rows"] = len(hits) + dropped
        tele["block_chars"] = used
        tele["est_tokens"] = est_tokens("\n".join(_default_render(h["kind"], h["entry"]) for h in hits))
        tele["retrieved_ids"] = [(h.get("entry") or {}).get("id") for h in hits]
        tele["gate_stopped"] = True
        tele["gate_reason"] = {"gate": "n/a (full dump)"}
        tele["degraded"] = f"truncated: {dropped} record(s) over budget" if dropped else None
        tele["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        self._publish(tele)
        return hits

    # ------------------------------------------------------------------ prompt-facing text
    def recall_text(self, query: str, scene: str | None = None, k: int = 5) -> str:
        """Markdown block of the most relevant memories (with provenance), shaped exactly like
        ``HybridRAG.recall_text`` — em-dash variant — so the webapp path is unaffected if the objects are swapped.
        """
        hits = self.search(query, scene=scene, k=k)
        if not hits:
            return ""
        lines = ["**Relevant memory** (retrieved — most relevant first):"]
        for h in hits:
            e = h.get("entry") or {}
            try:
                conf = float(e.get("confidence", 1.0))
            except Exception:
                conf = 1.0
            tag = f"_(src={e.get('source', '?')}, conf={conf:.2f})_"
            if h.get("kind") == "procedures":
                steps = " ; ".join(f"{s.get('action')} {s.get('object') or ''}".strip()
                                   for s in (e.get("plan") or []) if isinstance(s, dict))
                lines.append(f"- **worked before** — \"{e.get('command')}\" -> {steps} {tag}")
            elif h.get("kind") == "lessons":
                lines.append(f"- **lesson** — for \"{e.get('command')}\": {e.get('lesson')} {tag}")
            else:
                lines.append(f"- **fact** — {e.get('text')} {tag}")
        return "\n".join(lines)


def kb_read_trim(hits, budget_chars: int) -> list:
    """Truncate a ranked hit list to `budget_chars` of rendered body, lowest-ranked first (no merging).

    The ``--agentic-compact off`` path: the character budget is a hard constraint of the prompt, not a compaction
    feature, so it is enforced either way — what compaction adds on top is the dedup/merge.
    """
    kept, used = [], 0
    for h in (hits or []):
        body = _default_render(h.get("kind", ""), h.get("entry") or {})
        if used + len(body) > budget_chars and kept:
            break
        kept.append(h)
        used += len(body)
    return kept


# ==============================================================================================================
# Offline self-demo — mirrors sim/rag.py's __main__: no Ollama, no numpy, no network, nothing outside a temp dir.
# ==============================================================================================================
if __name__ == "__main__":
    import tempfile

    import knowledge                                  # lazy on purpose: only the demo needs the real KB class

    class _DeadProvider:
        """A provider that is always down — the offline equivalent of rag.py's endpoint http://127.0.0.1:1."""

        def __init__(self):
            self.calls = 0

        def chat(self, model, messages, **opts):
            self.calls += 1
            raise ConnectionError("no LLM provider reachable (offline self-demo)")

    with tempfile.TemporaryDirectory() as tmp:
        kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "_ar_demo.json"))
        now = time.time()
        kb.add_procedure("put the apple in the fridge",
                         [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}],
                         why="succeeded", scene="FloorPlan1", tags=["apple", "fridge"],
                         source="environment", ts=now, confidence=0.9)
        kb.add_lesson("slice the bread", "SliceObject failed: not holding a knife",
                      "pick up a Knife before slicing", scene="FloorPlan1",
                      source="human", ts=now, confidence=0.8)
        kb.add_fact("Running the faucet does not clean objects; use CleanObject.",
                    tags=["sink", "clean"], source="prior", ts=now - 14 * 24 * 3600, confidence=1.0)
        kb.add_procedure("سیب را داخل یخچال بگذار",
                         [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}],
                         why="succeeded", scene="FloorPlan2", tags=["سیب", "یخچال"],
                         source="demo", ts=now)

        print("== normalisation ==")
        print("  arabic yeh == persian yeh :", normalize_text("سيب") == normalize_text("سیب"))
        print("  harakat stripped          :", normalize_text("سَیب") == normalize_text("سیب"))
        print("  ZWNJ folded               :", tokens("گوجه‌فرنگی") == tokens("گوجه فرنگی"))
        print("  digits folded             :", normalize_text("۱۲۳ ١٢٣"))
        print("  stopwords (fa)            :", sorted(tokens("سیب را در یخچال بگذار")))
        print("  is_persian                :", is_persian("سیب را بردار"), is_persian("Pick up the apple."))

        print("\n== manifest (level 1: metadata only) ==")
        text, rows = kb_manifest(kb)
        print(text)
        print(f"  rows={len(rows)}  chars={len(text)}  est_tokens={est_tokens(text)}")

        print("\n== probe (level 2: lines, never bodies) ==")
        for h in kb_grep(kb, ["Apple", "Fridge"]):
            print(f"  #{h['entry']['id']} {h['kind']:<10} n_match={h['n_match']} matched={h['matched']}")
        print("  persian term crosses orthography:",
              [h["entry"]["id"] for h in kb_grep(kb, "سيب")])       # ARABIC yeh query, PERSIAN yeh record

        print("\n== provenance ranking ==")
        for h in rank_hits(kb_grep(kb, ["apple", "knife", "clean"]), scene="FloorPlan1", now=now, kb=kb):
            p = h["rank_parts"]
            print(f"  #{h['entry']['id']} {h['kind']:<10} score={h['score']:.4f}  "
                  f"sim={p['sim']} prov={p['prov']} conf={p['conf']} util={p['util']} scope={p['scope']}")

        print("\n== pivot (the FA->EN bridge, zero LLM calls) ==")
        print("  en:", lexical_pivot("Put the apple in the fridge"))
        print("  fa:", lexical_pivot("سیب را در یخچال بگذار"))
        print("  levels:", estimate_level("Open the fridge."),
              estimate_level("نان را با چاقو برش بزن و در یخچال بگذار"),
              estimate_level("یک خوراکی خنک برایم آماده کن."))

        print("\n== the program, with an UNREACHABLE provider ==")
        dead = _DeadProvider()
        ag = AgenticRAG(kb=kb, provider=dead, model="gemma3:4b", hops=2, gate=True)
        hits = ag.search("put apple in fridge", scene="FloorPlan1", observation="Apple, Fridge, CounterTop")
        for h in hits:
            print(f"  #{h['entry']['id']} {h['kind']:<10} score={h['score']} hop={h['hop']}")
        tele = ag.last_telemetry()
        print("  telemetry:", {kk: tele[kk] for kk in ("mode", "level", "hops", "llm_calls", "pivot_source",
                                                       "gate_stopped", "block_chars", "est_tokens")})
        assert tele["llm_calls"] <= 2, "the hop cap is unconditional"
        assert set(_TELEMETRY_KEYS) <= set(tele), "every telemetry key must always be present"

        print("\n  persian query (lexical bridge; the dead provider must NOT break it):")
        hits_fa = ag.search("سیب را در یخچال بگذار", scene="FloorPlan2", observation="Apple, Fridge")
        print("   ", [(h["kind"], h["entry"]["id"]) for h in hits_fa], "| calls:", dead.calls)

        print("\n== degradation floor ==")
        print("  keyword_fallback:", [(h["kind"], h["entry"]["id"])
                                      for h in keyword_fallback(kb, "apple fridge", scene="FloorPlan1")])

        print("\n== --rag full control arm ==")
        full = ag.dump()
        print(f"  {len(full)} records, est_tokens={ag.last_telemetry()['est_tokens']}")

        print("\n== plan verifier ==")
        print("  ", verify_plan([{"action": "pick_and_place", "object": "fridge"},
                                 {"action": "teleport", "object": "Apple"},
                                 {"action": "pickup", "object": "Unicorn"}],
                                {"Apple", "Fridge"}, {"pick_and_place", "pickup"}))

        print("\n== recall_text ==")
        print(ag.recall_text("wash and slice the bread", scene="FloorPlan1"))

    print("\nagentic_rag self-demo OK (offline: no numpy, no Ollama, no network, no repo store touched)")
