"""Hybrid (keyword + vector) retrieval over the agent's KnowledgeBase — the RAG grounding layer.

`knowledge.KnowledgeBase` gives auditable storage and a *keyword* `search()`. This module adds the missing half of
real RAG: **dense semantic retrieval**, fused with the keyword score, so the planner is grounded by memories that
are relevant in *meaning* even when they share no exact words (important for the bilingual Persian/English dataset).

Pieces:
  - :func:`embed` — text -> vectors via a local **Ollama** server (`/api/embeddings`), stdlib-urllib transport.
  - :class:`HybridRAG` — wraps a `KnowledgeBase`; ranks entries by
        ``score = alpha * keyword_overlap + (1 - alpha) * cosine(query, entry)``
    where the keyword term reuses `KnowledgeBase.search` scoring (normalized to [0,1] so ``alpha`` trades off
    cleanly against the cosine, which is also in [0,1]). Entry vectors are cached in a JSON **sidecar** next to
    `knowledge_base.json`. **If Ollama is unreachable, retrieval degrades gracefully to keyword-only** and logs a
    warning — the agent keeps working, just without the semantic half.

Stdlib-first, per CLAUDE.md: this module imports under a bare-stdlib Python. numpy is imported LAZILY inside the
cosine helper; chromadb is only ever probed inside a guarded import (never at module load); Ollama is reached over
plain HTTP, not a client library. ``import rag`` therefore succeeds with no numpy / chromadb / Ollama present.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

_log = logging.getLogger("rag")


def _warn(msg: str):
    _log.warning("rag: %s", msg)


# --------------------------------------------------------------------------- embeddings
def embed(texts, endpoint: str = "http://localhost:11434",
          model: str = "nomic-embed-text", timeout: float = 60.0) -> list:
    """Embed a string or list of strings -> ``list[list[float]]`` via Ollama ``/api/embeddings``.

    Uses only stdlib ``urllib``. Raises :class:`RuntimeError` with a clear message if the server is unreachable
    or returns no vector (callers that want graceful degradation should catch it — :class:`HybridRAG` does).
    """
    if isinstance(texts, str):
        texts = [texts]
    base = endpoint.rstrip("/")
    out = []
    for t in texts:
        req = urllib.request.Request(
            base + "/api/embeddings",
            data=json.dumps({"model": model, "prompt": str(t)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
        except Exception as exc:
            raise RuntimeError(
                f"embedding request to Ollama at {base} failed (model {model!r}): {exc}. "
                f"Is Ollama running and reachable, and is the model pulled (`ollama pull {model}`)?"
            ) from exc
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError(f"Ollama returned an empty embedding for model {model!r} "
                               f"(pull it with `ollama pull {model}`)")
        out.append(vec)
    return out


def _cosine(a, b) -> float:
    """Cosine similarity of two vectors (numpy imported LAZILY so the module loads without numpy)."""
    import numpy as np
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# --------------------------------------------------------------------------- hybrid retriever
class HybridRAG:
    """Keyword+vector retrieval over a :class:`knowledge.KnowledgeBase`, with a JSON embedding sidecar.

    Construct with an existing ``kb`` or a ``path`` to build one. Entry vectors are computed lazily on
    :meth:`add` (and on demand for keyword candidates during :meth:`search`) and cached in a sidecar
    ``<knowledge_base>_embeddings.json``. The first embedding failure flips the retriever to keyword-only for the
    rest of the session (so we don't pay repeated connection errors).
    """

    def __init__(self, kb=None, path: str = "knowledge_base.json",
                 endpoint: str = "http://localhost:11434", model: str = "nomic-embed-text",
                 embed_path: str | None = None):
        if kb is None:
            import knowledge  # lazy: keep `import rag` pure-stdlib and avoid a hard import-time dependency
            kb = knowledge.KnowledgeBase(path=path)
        self.kb = kb
        self.endpoint = endpoint
        self.model = model
        self.embed_path = embed_path or (os.path.splitext(self.kb.path)[0] + "_embeddings.json")
        self.embeddings: dict = {}     # str(entry_id) -> vector
        self._embed_ok = True          # flips False after the first embedding failure (keyword-only thereafter)
        self.chroma = self.chroma_available()  # capability probe only; current store is the JSON sidecar
        self._load_embeddings()

    # ------------------------------------------------------------- optional chroma backend (future)
    @staticmethod
    def chroma_available() -> bool:
        """Probe for an optional Chroma backend WITHOUT importing it at module load.

        The current vector store is the JSON sidecar; a future Chroma-backed store can branch on this. The import
        is attempted only here, inside a try/except, so ``import rag`` never requires ``chromadb``.
        """
        try:
            import chromadb  # noqa: F401  (optional dependency, only probed)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------- sidecar persistence
    def _load_embeddings(self):
        if not os.path.exists(self.embed_path):
            return
        try:
            with open(self.embed_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception as exc:
            _warn(f"could not read embedding sidecar {self.embed_path}: {exc}")
            return
        if blob.get("model") and blob.get("model") != self.model:
            _warn(f"embedding sidecar was built with model {blob.get('model')!r} != {self.model!r}; ignoring it")
            return
        self.embeddings = blob.get("vectors", {}) or {}

    def _save_embeddings(self):
        tmp = self.embed_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"model": self.model, "vectors": self.embeddings}, fh)
            os.replace(tmp, self.embed_path)
        except Exception as exc:
            _warn(f"could not write embedding sidecar {self.embed_path}: {exc}")

    # ------------------------------------------------------------- embedding of stored entries
    @staticmethod
    def _entry_text(e: dict) -> str:
        """Canonical text to embed for an entry: its command/text/lesson/problem, tags, and any plan steps."""
        parts = [str(e.get(f, "")) for f in ("command", "text", "lesson", "problem")]
        parts += list(e.get("tags", []) or [])
        for s in e.get("plan", []) or []:
            parts.append(f"{s.get('action', '')} {s.get('object', '') or ''}".strip())
        return " ".join(p for p in parts if p).strip()

    def _embed_entry(self, entry: dict, save: bool = True):
        """Embed and cache one entry's vector; return it, or None if embeddings are unavailable."""
        if not self._embed_ok:
            return None
        try:
            vec = embed([self._entry_text(entry)], self.endpoint, self.model)[0]
        except Exception as exc:
            self._embed_ok = False
            _warn(f"embeddings unavailable ({exc}); falling back to keyword-only retrieval")
            return None
        self.embeddings[str(entry.get("id"))] = vec
        if save:
            self._save_embeddings()
        return vec

    def reindex(self, force: bool = False) -> int:
        """Embed every entry missing a cached vector (best-effort). Returns the number newly embedded."""
        n = 0
        for kind in ("procedures", "lessons", "facts"):
            for e in self.kb.data.get(kind, []):
                if not self._embed_ok:
                    break
                if force or str(e.get("id")) not in self.embeddings:
                    if self._embed_entry(e, save=False) is not None:
                        n += 1
        if n:
            self._save_embeddings()
        return n

    # ------------------------------------------------------------- write
    def add(self, kind: str, text: str, source: str, scope: str = "global",
            confidence: float = 1.0, **fields) -> dict:
        """Add an entry of ``kind`` ('procedure'|'lesson'|'fact', singular or plural) and index its embedding.

        ``text`` is the entry's primary content (the command for procedures/lessons, the fact text for facts);
        the rest of each kind's fields are passed through ``**fields`` (``plan=``, ``problem=``, ``lesson=``,
        ``command=``, ``scene=``, ``tags=``, ``ts=``, ``why=``). Provenance (``source``/``scope``/``confidence``)
        is forwarded to the KnowledgeBase so the write stays auditable.
        """
        k = kind.lower().rstrip("s")
        why = fields.pop("why", "")
        ts = fields.pop("ts", None)
        if k == "procedure":
            entry = self.kb.add_procedure(
                fields.pop("command", text), fields.pop("plan", []), why=why,
                scene=fields.pop("scene", None), tags=fields.pop("tags", None),
                source=source, ts=ts, confidence=confidence, scope=scope)
        elif k == "lesson":
            entry = self.kb.add_lesson(
                fields.pop("command", text), fields.pop("problem", ""), fields.pop("lesson", text),
                why=why, scene=fields.pop("scene", None),
                source=source, ts=ts, confidence=confidence, scope=scope)
        elif k == "fact":
            entry = self.kb.add_fact(
                text, why=why, tags=fields.pop("tags", None),
                source=source, ts=ts, confidence=confidence, scope=scope)
        else:
            raise ValueError(f"unknown kind {kind!r} (expected 'procedure', 'lesson', or 'fact')")
        self._embed_entry(entry)   # best-effort; degrades to keyword-only if Ollama is down
        return entry

    # ------------------------------------------------------------- read
    @staticmethod
    def _in_scope(entry: dict, scene: str | None) -> bool:
        """An entry is in scope when no scene filter is given, or its scope is 'global' or matches `scene`."""
        if not scene:
            return True
        sc = entry.get("scope") or entry.get("scene") or "global"
        return sc == "global" or sc == scene

    def search(self, query: str, scene: str | None = None, k: int = 5, alpha: float = 0.5) -> list:
        """Hybrid-rank memories for `query`; return up to `k` ``{kind, score, keyword, cosine, entry}`` dicts.

        ``score = alpha * keyword + (1 - alpha) * cosine`` where ``keyword`` is `KnowledgeBase.search`'s score
        normalized to [0,1] across candidates and ``cosine`` is in [0,1]. With ``alpha=1`` (or no Ollama) this is
        pure keyword search; with ``alpha=0`` it is pure semantic over entries that have a cached embedding.
        Entries can be restricted to a `scene` (keeps that scene's entries plus 'global' ones).
        """
        # Keyword scores for every matching entry (reuse KnowledgeBase scoring), keyed by entry id.
        kw_map = {h["entry"].get("id"): h["score"] for h in self.kb.search(query, limit=10_000)}

        # Query embedding (only if the semantic half is wanted and Ollama is up).
        qvec = None
        if alpha < 1.0 and self._embed_ok:
            try:
                qvec = embed([query], self.endpoint, self.model)[0]
            except Exception as exc:
                self._embed_ok = False
                _warn(f"embeddings unavailable ({exc}); falling back to keyword-only retrieval")

        cands = []     # (kind, entry, kw_raw, cos)
        dirty = False
        for kind in ("procedures", "lessons", "facts"):
            for e in self.kb.data.get(kind, []):
                if not self._in_scope(e, scene):
                    continue
                kw_raw = float(kw_map.get(e.get("id"), 0.0))
                cos = 0.0
                if qvec is not None:
                    ev = self.embeddings.get(str(e.get("id")))
                    if ev is None and kw_raw > 0 and self._embed_ok:
                        ev = self._embed_entry(e, save=False)   # lazily index keyword candidates
                        dirty = dirty or ev is not None
                    if ev is not None:
                        try:
                            cos = max(0.0, _cosine(qvec, ev))
                        except Exception as exc:      # e.g. numpy missing -> drop the semantic half cleanly
                            self._embed_ok = False
                            qvec = None
                            _warn(f"cosine unavailable ({exc}); falling back to keyword-only retrieval")
                if kw_raw <= 0.0 and cos <= 0.0:
                    continue
                cands.append([kind, e, kw_raw, cos])
        if dirty:
            self._save_embeddings()

        kw_max = max((c[2] for c in cands), default=0.0) or 1.0
        results = []
        for kind, e, kw_raw, cos in cands:
            kw = kw_raw / kw_max
            score = alpha * kw + (1.0 - alpha) * cos
            results.append({"kind": kind, "score": round(score, 4),
                            "keyword": round(kw, 4), "cosine": round(cos, 4), "entry": e})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]

    def recall_text(self, query: str, scene: str | None = None, k: int = 5) -> str:
        """Markdown block of the most relevant memories (with provenance), for injection into a planner prompt."""
        hits = self.search(query, scene=scene, k=k)
        if not hits:
            return ""
        lines = ["**Relevant memory** (retrieved — most relevant first):"]
        for h in hits:
            e = h["entry"]
            tag = f"_(src={e.get('source', '?')}, conf={float(e.get('confidence', 1.0)):.2f})_"
            if h["kind"] == "procedures":
                steps = " ; ".join(f"{s.get('action')} {s.get('object') or ''}".strip()
                                   for s in e.get("plan", []))
                lines.append(f"- **worked before** — \"{e.get('command')}\" -> {steps} {tag}")
            elif h["kind"] == "lessons":
                lines.append(f"- **lesson** — for \"{e.get('command')}\": {e.get('lesson')} {tag}")
            else:
                lines.append(f"- **fact** — {e.get('text')} {tag}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- retrieval credit assignment
def utility_path(kb_path: str) -> str:
    """``<kb stem>_utility.json`` — the sidecar :func:`credit` writes and ``agentic_rag.utility`` reads."""
    return os.path.splitext(str(kb_path or "knowledge_base.json"))[0] + "_utility.json"


def load_utility(kb_path: str) -> dict:
    """Read the retrieval-credit sidecar. -> ``{entry_id_as_str: {"helpful": int, "harmful": int}}``, ``{}``
    when the file is absent or unreadable (the normal case on a fresh knowledge base)."""
    path = utility_path(kb_path)
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:
        _warn(f"could not read utility sidecar {path}: {exc}; utility falls back to the reinforcement proxy")
        return {}
    if not isinstance(blob, dict):
        return {}
    util = blob.get("utility", blob)
    return util if isinstance(util, dict) else {}


def credit(kb_path: str, retrieved, success: bool, *, why: str = "", ts: float | None = None) -> dict:
    """Assign RETRIEVAL credit for one executed command and persist it. -> the updated utility map.

    This is the write half of ``rank_key``'s utility term and the whole difference between
    ``--rank provenance`` and ``--rank provenance+utility``. Without it the two ablation arms are byte-identical
    (`agentic_rag.utility` finds no sidecar and falls back to the reinforcement proxy in both), i.e. the sweep
    reports a null result about a mechanism that was never switched on. That was the state on disk until this
    review pass; the eval call site had always been there, wrapped in a bare ``except Exception: pass``.

    `retrieved` is the eval harness's retrieved-id list — either bare ids or ``{"id": ..., "kind": ...}`` dicts.
    Every retrieved entry gets ``helpful += 1`` when the record succeeded and ``harmful += 1`` when it did not:
    coarse, symmetric, and deliberately not a learned weight. ``agentic_rag.utility`` then reads it as the
    Laplace-smoothed ``(helpful + 1) / (helpful + harmful + 2)``, so one bad outcome cannot zero a record and one
    good outcome cannot pin it at 1.0.

    **Auditable, like every other self-improving-loop write** (CLAUDE.md): the sidecar keeps a bounded
    ``log`` of the last 2000 credit events (ts, ids, outcome, why) so a utility score can always be traced back
    to the runs that produced it, and it is written atomically via a ``.tmp`` + ``os.replace``.

    Never raises: an unwritable path warns and returns the in-memory map.
    """
    import time as _time

    path = utility_path(kb_path)
    blob = {"utility": {}, "log": []}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                got = json.load(fh)
            if isinstance(got, dict):
                blob["utility"] = got.get("utility", got if "log" not in got else {}) or {}
                blob["log"] = got.get("log") or []
    except Exception as exc:
        _warn(f"could not read utility sidecar {path}: {exc}; starting a fresh one")

    ids = []
    for r in (retrieved or []):
        eid = r.get("id") if isinstance(r, dict) else r
        if eid is None:
            continue
        key = str(eid)
        if key not in ids:
            ids.append(key)

    field = "helpful" if success else "harmful"
    util = blob["utility"] if isinstance(blob["utility"], dict) else {}
    for key in ids:
        rec = util.get(key)
        if not isinstance(rec, dict):
            rec = {"helpful": 0, "harmful": 0}
        try:
            rec[field] = int(rec.get(field, 0) or 0) + 1
        except Exception:
            rec[field] = 1
        rec.setdefault("helpful", 0)
        rec.setdefault("harmful", 0)
        util[key] = rec
    blob["utility"] = util

    if ids:
        blob["log"] = (list(blob["log"]) + [{"ts": float(ts if ts is not None else _time.time()),
                                             "ids": ids, "outcome": bool(success),
                                             "why": str(why or "eval record outcome")}])[-2000:]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        _warn(f"could not write utility sidecar {path}: {exc}")
    return util


if __name__ == "__main__":
    # Self-test that runs WITHOUT Ollama or numpy: it must degrade to keyword-only and still retrieve.
    import knowledge

    kb = knowledge.KnowledgeBase(path="_rag_test.json")
    rag = HybridRAG(kb=kb, endpoint="http://127.0.0.1:1")   # unreachable on purpose -> keyword-only
    rag.add("procedure", "put the apple in the fridge",
            source="environment", scope="FloorPlan1", tags=["apple", "fridge"],
            plan=[{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}], why="succeeded")
    rag.add("lesson", "slice the bread", source="environment", scope="FloorPlan1",
            problem="SliceObject failed: not holding a knife", lesson="pick up a Knife before slicing")
    rag.add("fact", "Running the faucet does not clean objects; use CleanObject.",
            source="prior", tags=["sink", "clean"], confidence=0.9)
    print("embed_ok (expect False, no Ollama):", rag._embed_ok)
    print("search 'apple fridge':", rag.search("put apple in fridge"))
    print("scene filter FloorPlan2 (apple proc is FloorPlan1, fact is global):",
          [h["kind"] for h in rag.search("apple knife clean", scene="FloorPlan2")])
    print("\nrecall:\n" + rag.recall_text("wash and slice the bread"))
    for p in ("_rag_test.json", "_rag_test_audit.log", "_rag_test_embeddings.json"):
        if os.path.exists(p):
            os.remove(p)
    print("\nrag self-test OK (keyword-only fallback)")
