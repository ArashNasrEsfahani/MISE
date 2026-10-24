"""Knowledge base for the self-improving loop — the agent's long-term memory.

This is the thesis's third contribution (a self-improving agent): after acting, the agent writes back what it
learned — **procedures** (a command that succeeded + the plan that worked), **lessons** (a command that failed + why
+ what to do differently), and **facts** (durable observations about the world). Future commands first *retrieve*
relevant entries (the RAG grounding) before planning, so the agent gets better with experience.

Storage is a single JSON file (a vector store comes later — `search()` is keyword-based here; the hybrid
vector retriever lives in `rag.py` and reuses this scoring). **Every write is auditable**: it is appended to a
human-readable audit log with a timestamp, the entry's `source`, and a `why`, so the proof-of-concept can be
evaluated (per CLAUDE.md: self-improving-loop writes must be auditable).

Every entry also carries **provenance** so retrieval and evaluation can weigh memories by where they came from
and when:
  - ``source``     — one of ``'human' | 'demo' | 'environment' | 'prior'`` (who/what produced the entry).
  - ``ts``         — float epoch seconds (``time.time()``); recency signal, distinct from the human ``time`` string.
  - ``confidence`` — 0..1 trust weight (default 1.0).
  - ``scope``      — a ``FloorPlan`` name the memory is specific to, or ``'global'`` (default).
All four are OPTIONAL keyword args on the write API with sensible defaults, so existing callers are unaffected.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

# ---------------------------------------------------------------------- observability sidecar (optional)
# `trace` is ALSO a stdlib module (the coverage tracer). Because sim/ sits ahead of the stdlib on sys.path a
# plain `import trace` resolves to sim/trace.py here, but a differently ordered sys.path would silently hand us
# the stdlib module — hence the TRACE_API_VERSION guard. When the sidecar is absent (bare-stdlib environment,
# `sim/` not on the path) every hook below degrades to a no-op and this module behaves exactly as before.
try:                                    # pragma: no cover - environment dependent
    import trace as _trace              # sim/trace.py — the observability plane
    if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
        _trace = None                   # we resolved the STDLIB trace module: no instrumentation
except Exception:
    _trace = None


def _t(name, *a, **kw):
    """Call ``trace.<name>(*a, **kw)``, swallowing EVERYTHING — mirrors _audit's IO guard.

    The observability plane is a sidecar: a raising hook here would break a knowledge-base write, which is the
    one thing the thesis's self-improving loop cannot tolerate. Returns None on any failure.

    The tracer function is named by STRING and resolved lazily on purpose: writing ``_t(_trace.emit, ...)``
    would evaluate the attribute at the call site, raising ``AttributeError`` before this guard can run
    whenever the sidecar is absent — i.e. exactly in the bare-stdlib environment this indirection protects.
    """
    if _trace is None:
        return None
    try:
        fn = getattr(_trace, name, None)
        return None if fn is None else fn(*a, **kw)
    except Exception:
        return None


# Mirror every audit line into `events` as its own row? Default on (D1: one place shows everything); set
# TRACE_AUDIT_MIRROR=0 for a long batch where the duplicate narration is pure volume.
AUDIT_MIRROR = str(os.environ.get("TRACE_AUDIT_MIRROR", "1")).strip().lower() not in ("0", "false", "no", "off")


def _actor(source):
    """PROV agent id for a provenance `source` ('human'|'demo'|'environment'|'prior'), or None."""
    if _trace is None:
        return None
    try:
        return getattr(_trace, "_SOURCE_ACTOR", {}).get(str(source or ""))
    except Exception:
        return None


# Words too common to be useful for keyword retrieval.
_STOP = {"the", "a", "an", "to", "in", "on", "of", "and", "it", "is", "with", "into", "then",
         "put", "go", "get", "please", "up", "down", "from", "at", "for", "this", "that"}


def _tokens(text: str) -> set:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in str(text)).split()
            if w and w not in _STOP and len(w) > 1}


def normalize_command(text) -> str:
    """Canonical form of a command for dedup keying: lowercase, punctuation -> space, collapsed whitespace.

    "Put the Apple in the fridge!!" and "put   the apple in the fridge" normalize identically, so the Phase-7
    loop reinforces ONE stored entry instead of accumulating near-duplicates (the KB-side analogue of
    memory.py's funnel keys).
    """
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in str(text or "")).split())


def plan_signature(plan) -> tuple:
    """Order-preserving tuple of (action, object) pairs identifying WHAT a plan does.

    Object ids/coordinates and auxiliary step keys are ignored, so two runs that solved a command with the
    same actions on the same object types share a signature and dedup into one procedure. Non-dict steps are
    skipped defensively.
    """
    sig = []
    for s in (plan or []):
        if isinstance(s, dict):
            sig.append((str(s.get("action") or "").lower().strip(),
                        str(s.get("object") or "").lower().strip()))
    return tuple(sig)


def _error_class(error) -> str:
    """Coarse deterministic error bucket for lesson dedup.

    Byte-identical to memory.Memory._error_class (lowercase, strip ids/digits/punctuation, collapse
    whitespace, truncate 48) so lesson keys here and failure signatures in memory.py agree — a duplicated
    algorithm on purpose: knowledge.py stays dependency-free.
    """
    import re
    s = str(error or "").lower()
    s = re.sub(r"[|]", " ", s)                     # object-id pipe separators
    s = re.sub(r"[-+]?\d*\.?\d+", " ", s)          # digits / coordinates / ids
    s = re.sub(r"[^a-z ]+", " ", s)                # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()             # collapse whitespace
    return s[:48].strip()


class KnowledgeBase:
    """JSON-backed long-term memory with auditable writes and keyword retrieval."""

    def __init__(self, path: str = "knowledge_base.json", audit_path: str | None = None):
        self.path = path
        self.audit_path = audit_path or (os.path.splitext(path)[0] + "_audit.log")
        self.data: dict = {"procedures": [], "lessons": [], "facts": []}
        self._seq = 0
        # Phase-7 makes the loop's plan thread a KB writer alongside the HTTP-handler writers (/teach, /rating,
        # /demo); a reentrant lock serializes every read-modify-write + _save so concurrent writes cannot
        # interleave a torn tmp file (which _load would then silently drop) or resize a dict mid-serialization.
        self._lock = threading.RLock()
        self._load()

    # ----------------------------------------------------------------- storage
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                for k in self.data:
                    self.data[k] = loaded.get(k, [])
                self._seq = max([e.get("id", 0) for kind in self.data.values() for e in kind] + [0])
            except Exception as exc:
                # NEVER silently wipe long-term memory on a parse error: surface it and preserve the file so a
                # corrupt-but-recoverable KB is not overwritten by the next _save (which starts from empty data).
                print(f"[knowledge] WARNING: could not load {self.path}: {exc!r}; "
                      f"starting empty (existing file preserved)", file=sys.stderr)
                try:
                    backup = self.path + ".corrupt"
                    if not os.path.exists(backup):
                        import shutil
                        shutil.copy2(self.path, backup)
                        print(f"[knowledge] backed up unreadable KB to {backup}", file=sys.stderr)
                except Exception:
                    pass

    def _save(self):
        with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)

    def _audit(self, op: str, summary: str, why: str, source: str | None = None):
        src = f"  <{source}>" if source else ""
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [{op}]{src}  {summary}"
                + (f"   WHY: {why}" if why else ""))
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        # K5: mirror the human-readable line into the structured trail verbatim, so the two can never diverge.
        # Gated (TRACE_AUDIT_MIRROR=0 turns it off) because it is the single largest event class — ~49% of all
        # rows — and every structured write already carries the SAME line on `writes.audit_line`. The mirror
        # earns its keep only for the audit ops that produce no `writes` row of their own.
        if AUDIT_MIRROR:
            _t("emit", "audit.line", summary=line[:200], level="DEBUG", actor=_actor(source),
               payload={"store": "kb", "op": op, "source": source, "summary": summary, "why": why,
                        "path": self.audit_path, "line": line})
        return line

    def _add(self, kind: str, entry: dict, why: str, source: str = "human",
             ts: float | None = None, confidence: float = 1.0, scope: str | None = None) -> dict:
        with self._lock:
            self._seq += 1
            prov = {"source": source, "ts": float(ts) if ts is not None else time.time(),
                    "confidence": float(confidence), "scope": scope or "global"}
            entry = {"id": self._seq, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **prov, **entry}
            self.data[kind].append(entry)
            self._save()
            line = self._audit("ADD " + kind[:-1],
                               entry.get("command") or entry.get("text") or str(entry.get("id")),
                               why, source=source)
            # K1: the SINGLE insert choke point — every KB create in the repo funnels through here (the loop,
            # human teaching, demos, ratings, rag.add, seed_memory, even the legacy Tk control_gui), so one hook
            # attributes them all without touching a public signature.
            _t("record_write", store="kb", table_or_kind=kind[:-1], pointer=entry.get("id"),
               verdict="insert", why=why, source=source, scope=entry.get("scope"),
               confidence=entry.get("confidence"), occurrences=entry.get("occurrences", 1),
               summary=entry.get("command") or entry.get("text") or f"#{entry.get('id')}",
               content=entry, audit_line=line,
               # No int()/float() coercion at this call site on purpose: `_add` did none before, and a hook
               # must not introduce a new way for a KB insert to raise.
               after={"occurrences": entry.get("occurrences", 1), "uses": entry.get("uses"),
                      "confidence": entry.get("confidence"), "scope": entry.get("scope")},
               store_path=self.path)
            return entry

    # -------------------------------------------------------------- write API
    def add_procedure(self, command: str, plan: list, why: str = "", scene: str | None = None,
                      tags: list | None = None, source: str = "environment", ts: float | None = None,
                      confidence: float = 1.0, scope: str | None = None) -> dict:
        """Record a command and the plan that successfully accomplished it.

        Provenance defaults to ``source='environment'`` (the agent learned this by acting in the sim); ``scope``
        falls back to ``scene`` when not given, so a plan proven in ``FloorPlan1`` is tagged to it.
        """
        return self._add("procedures", {"command": command, "plan": plan, "scene": scene,
                                        "tags": tags or [], "uses": 1}, why,
                         source=source, ts=ts, confidence=confidence, scope=scope or scene)

    def add_lesson(self, command: str, problem: str, lesson: str, why: str = "",
                   scene: str | None = None, source: str = "environment", ts: float | None = None,
                   confidence: float = 1.0, scope: str | None = None) -> dict:
        """Record a failure and the lesson learned, so the same mistake can be avoided next time.

        Defaults to ``source='environment'``; ``scope`` falls back to ``scene`` when not given.
        """
        return self._add("lessons", {"command": command, "problem": problem, "lesson": lesson,
                                     "scene": scene}, why,
                         source=source, ts=ts, confidence=confidence, scope=scope or scene)

    def add_fact(self, text: str, why: str = "", tags: list | None = None, source: str = "human",
                 ts: float | None = None, confidence: float = 1.0, scope: str | None = None) -> dict:
        """Record a durable observation about the world/objects.

        Defaults to ``source='human'`` (a hand-authored/observed fact); pass ``source='prior'`` when seeding from
        skill docs. ``scope`` defaults to ``'global'`` since facts are usually scene-independent.
        """
        return self._add("facts", {"text": text, "tags": tags or []}, why,
                         source=source, ts=ts, confidence=confidence, scope=scope)

    def reinforce(self, entry_id: int):
        """Bump a procedure's use-count when it's reused successfully (cheap relevance signal).

        This used to be the one KB mutation with NO audit line at all — an invisible edit of research data that
        silently changes retrieval ranking for every future command. It now writes a ``REINFORCE procedure``
        audit line (a new op string; nothing asserts on it) and a structured ``verdict='update'`` trace write.
        """
        with self._lock:
            for p in self.data["procedures"]:
                if p.get("id") == entry_id:
                    before = int(p.get("uses", 1))
                    p["uses"] = before + 1
                    self._save()
                    # K4: close the silent-mutation hole (audit line + trace write).
                    line = self._audit("REINFORCE procedure", f"uses {before} -> {p['uses']}",
                                       "procedure reused", source=p.get("source"))
                    _t("record_write", store="kb", table_or_kind="procedure", pointer=p.get("id"),
                       verdict="update", why="procedure reused", source=p.get("source"),
                       scope=p.get("scope"), confidence=p.get("confidence"),
                       occurrences=int(p.get("occurrences", 1)), summary=p.get("command"),
                       audit_line=line, before={"uses": before}, after={"uses": p["uses"]},
                       store_path=self.path)
                    return p
        return None

    # ------------------------------------------- write API: dedup upserts (Phase-7 loop closure)
    def _reinforce_entry(self, entry: dict, ts: float | None, confidence: float) -> int:
        """Shared reinforce mutation: bump 'occurrences', refresh 'ts'/'time'/'confidence'. Returns the count.

        Only counters/recency/trust change — the entry's text and plan stay untouched, so any cached
        embedding vector in the RAG sidecar remains valid (no re-embed needed on reinforce).
        """
        n = entry["occurrences"] = int(entry.get("occurrences", 1)) + 1
        entry["ts"] = float(ts) if ts is not None else time.time()
        entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        c = float(entry.get("confidence", 1.0))
        # Grow toward 1 like memory.observe_location (conf + (1-conf)*0.5), but never below the fresh signal.
        entry["confidence"] = round(min(1.0, max(float(confidence), c + (1.0 - c) * 0.5)), 4)
        return n

    def upsert_procedure(self, command: str, plan: list, why: str = "", scene: str | None = None,
                         tags: list | None = None, source: str = "environment", ts: float | None = None,
                         confidence: float = 1.0, scope: str | None = None) -> dict:
        """Insert a procedure, or REINFORCE the existing one with the same (normalized command, plan signature).

        The KB-side analogue of memory.observe_location's change-gated upsert: re-learning the same solution
        bumps 'occurrences' (dedup counter) and 'uses' (retrieval-ranking boost, mirroring reinforce()) and
        refreshes ts/confidence instead of appending a near-duplicate row. Both paths are audited (insert via
        add_procedure's 'ADD procedure', reinforce via an explicit 'UPSERT procedure' line — reinforce() alone
        writes no audit). Returns {'action': 'insert'|'reinforce', 'id': int, 'entry': dict}.
        """
        key = (normalize_command(command), plan_signature(plan))
        with self._lock:
            for e in self.data["procedures"]:
                if (normalize_command(e.get("command")), plan_signature(e.get("plan"))) == key:
                    # Snapshot the counters BEFORE the mutation so the trace can show the delta (K2).
                    occ_before = int(e.get("occurrences", 1))
                    uses_before = int(e.get("uses", 1))
                    conf_before = float(e.get("confidence", 1.0))
                    n = self._reinforce_entry(e, ts, confidence)
                    e["uses"] = uses_before + 1                # proven plans rank higher in search()
                    self._save()
                    line = self._audit("UPSERT procedure", f"reinforced '{command}' (x{n})", why, source=source)
                    # K2: a MERGE, not an insert — no new id, counters moved. The dedup key is COMPUTED and never
                    # persisted on the entry, so record it here: it is the only answer to "why did this merge?".
                    _t("record_write", store="kb", table_or_kind="procedure", pointer=e.get("id"),
                       verdict="reinforce", why=why, source=source, scope=e.get("scope"),
                       confidence=e.get("confidence"), occurrences=n, summary=e.get("command"),
                       audit_line=line, dedup_key=f"{key[0]}|{key[1]!r}",
                       before={"occurrences": occ_before, "uses": uses_before, "confidence": conf_before},
                       after={"occurrences": n, "uses": e["uses"], "confidence": e.get("confidence")},
                       store_path=self.path)
                    return {"action": "reinforce", "id": e.get("id"), "entry": e}
            entry = self.add_procedure(command, plan, why=why, scene=scene, tags=tags, source=source,
                                       ts=ts, confidence=confidence, scope=scope)
            return {"action": "insert", "id": entry.get("id"), "entry": entry}

    def upsert_lesson(self, command: str, problem: str, lesson: str, why: str = "",
                      scene: str | None = None, source: str = "environment", ts: float | None = None,
                      confidence: float = 1.0, scope: str | None = None, error: str | None = None) -> dict:
        """Insert a lesson, or REINFORCE the existing one with the same (normalized command, error class).

        The KB-side analogue of memory.record_failure's signature counter: the same command failing the same
        WAY again bumps 'occurrences' and refreshes ts/confidence; the first-seen 'problem'/'lesson' text is
        kept (record_failure's first_seen semantics — also keeps the cached embedding valid). Audited on both
        paths. Returns {'action': 'insert'|'reinforce', 'id': int, 'entry': dict}.

        ``error`` is the RAW agent error to key the dedup on. When given it is used instead of ``problem``:
        callers build a human-readable ``problem`` like ``"<step label> failed: <error>"``, and the step label
        would otherwise consume the whole 48-char ``_error_class`` window, collapsing distinct failure modes of
        the same command into one lesson (storing the wrong remediation). Keying on the raw error keeps
        "fridge is closed" and "receptacle is full" as separate lessons. Falls back to ``problem`` for callers
        that pass only the two-string form (backward compatible). The dedup KEY is stored on the entry as
        ``error_key`` so a later match reuses the same basis even though only ``problem`` text is persisted.
        """
        basis = error if error is not None else problem
        key = (normalize_command(command), _error_class(basis))
        with self._lock:
            for e in self.data["lessons"]:
                e_basis = e.get("error_key", e.get("problem"))
                if (normalize_command(e.get("command")), _error_class(e_basis)) == key:
                    occ_before = int(e.get("occurrences", 1))
                    conf_before = float(e.get("confidence", 1.0))
                    n = self._reinforce_entry(e, ts, confidence)
                    self._save()
                    line = self._audit("UPSERT lesson", f"reinforced '{command}' (x{n})", why, source=source)
                    # K3a: lesson MERGE — keyed on (normalized command, error class of the RAW error).
                    _t("record_write", store="kb", table_or_kind="lesson", pointer=e.get("id"),
                       verdict="reinforce", why=why, source=source, scope=e.get("scope"),
                       confidence=e.get("confidence"), occurrences=n, summary=e.get("command"),
                       audit_line=line, dedup_key=f"{key[0]}|{key[1]}",
                       before={"occurrences": occ_before, "confidence": conf_before},
                       after={"occurrences": n, "confidence": e.get("confidence")},
                       store_path=self.path)
                    return {"action": "reinforce", "id": e.get("id"), "entry": e}
            entry = self.add_lesson(command, problem, lesson, why=why, scene=scene, source=source,
                                    ts=ts, confidence=confidence, scope=scope)
            entry["error_key"] = str(basis or "")              # remember the dedup basis for future matches
            self._save()
            # K3b: the error_key patch is a SECOND save that no audit line covers, and the entry is briefly
            # persisted without its dedup basis. Record it as an explicit update so the gap is visible.
            _t("record_write", store="kb", table_or_kind="lesson", pointer=entry.get("id"),
               verdict="update", why="error_key patch: persist the dedup basis after insert",
               source=source, scope=entry.get("scope"), confidence=entry.get("confidence"),
               summary=entry.get("command"), dedup_key=f"{key[0]}|{key[1]}",
               before={"error_key": None}, after={"error_key": entry["error_key"]},
               store_path=self.path)
            return {"action": "insert", "id": entry.get("id"), "entry": entry}

    # --------------------------------------------------------------- read API
    def search(self, query: str, kinds: tuple = ("procedures", "lessons", "facts"), limit: int = 5) -> list:
        """Return entries most relevant to `query` (keyword overlap), best first.

        Same call shape a vector retriever will use later — only the scoring changes. Each result is
        {kind, score, entry}.
        """
        q = _tokens(query)
        if not q:
            return []
        scored = []
        with self._lock:                                          # snapshot-consistent read vs concurrent writers
            for kind in kinds:
                for e in list(self.data.get(kind, [])):
                    hay = " ".join(str(e.get(f, "")) for f in ("command", "text", "lesson", "problem"))
                    hay += " " + " ".join(e.get("tags", []))
                    overlap = len(q & _tokens(hay))
                    if overlap:
                        score = overlap + 0.1 * e.get("uses", 0)  # mild preference for proven procedures
                        scored.append({"kind": kind, "score": round(score, 2), "entry": e})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]

    def recall_text(self, query: str, limit: int = 3) -> str:
        """Compact text block of the most relevant memories, for injection into a planner prompt."""
        hits = self.search(query, limit=limit)
        lines = []
        for h in hits:
            e = h["entry"]
            if h["kind"] == "procedures":
                steps = " ; ".join(f"{s.get('action')} {s.get('object') or ''}".strip() for s in e.get("plan", []))
                lines.append(f"- (worked before) \"{e.get('command')}\" -> {steps}")
            elif h["kind"] == "lessons":
                lines.append(f"- (lesson) for \"{e.get('command')}\": {e.get('lesson')}")
            else:
                lines.append(f"- (fact) {e.get('text')}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {k: len(v) for k, v in self.data.items()}


if __name__ == "__main__":
    kb = KnowledgeBase(path="_kb_test.json")
    kb.add_procedure("put the apple in the fridge",
                     [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}],
                     why="succeeded 1/1 steps", scene="FloorPlan1", tags=["apple", "fridge"])
    kb.add_lesson("slice the bread", "SliceObject failed: agent not holding a knife",
                  "pick up a Knife before slicing", why="step 2 failed", scene="FloorPlan1")
    kb.add_fact("Running the faucet does not clean objects; use CleanObject.", tags=["sink", "clean"])
    print("stats:", kb.stats())
    print("search 'apple fridge':", kb.search("put apple in fridge"))
    print("\nrecall for 'wash and slice bread':\n" + kb.recall_text("wash and slice the bread"))

    # ----- Phase-7 upserts: insert-then-reinforce dedup (procedures keyed by command+plan, lessons by command+error).
    r1 = kb.upsert_procedure("wash the mug", [{"action": "wash", "object": "Mug"}],
                             why="loop closure: succeeded", scene="FloorPlan1")
    assert r1["action"] == "insert", f"new procedure should insert, got {r1!r}"
    r2 = kb.upsert_procedure("Wash the mug!", [{"action": "wash", "object": "Mug"}],
                             why="loop closure: succeeded again")
    assert r2["action"] == "reinforce" and r2["id"] == r1["id"], f"same command+plan should reinforce, got {r2!r}"
    assert r2["entry"]["occurrences"] == 2 and r2["entry"]["uses"] == 2, f"counters should bump, got {r2['entry']!r}"
    r3 = kb.upsert_procedure("wash the mug", [{"action": "wash", "object": "Cup"}], why="different plan")
    assert r3["action"] == "insert", "a different plan signature must NOT collapse into the old procedure"
    # Backward compat: an add_procedure-era entry (no 'occurrences' field) reinforces cleanly.
    r0 = kb.upsert_procedure("Put the apple in the fridge",
                             [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}], why="re-learned")
    assert r0["action"] == "reinforce" and r0["entry"]["occurrences"] == 2, f"legacy entry should reinforce, got {r0!r}"
    l1 = kb.upsert_lesson("slice the tomato", "SliceObject failed: Knife|+01.0|-00.3 not visible",
                          "go to the knife first", why="loop closure: failed", scene="FloorPlan1")
    assert l1["action"] == "insert", f"new lesson should insert, got {l1!r}"
    l2 = kb.upsert_lesson("Slice the tomato", "SliceObject failed: Knife|+02.4|-01.1 not visible",
                          "go to the knife first", why="loop closure: failed again")
    assert l2["action"] == "reinforce" and l2["entry"]["occurrences"] == 2, \
        f"same command+error class should reinforce, got {l2!r}"
    # Two DIFFERENT failure modes of the same command, behind a long shared step label, must NOT collapse:
    # keying on the raw `error` (not the label-prefixed `problem`) keeps them distinct (Phase-7 dedup fix).
    long_label = "pick and place the apple onto the counter top failed"
    e1 = kb.upsert_lesson("put the apple on the counter", f"{long_label}: Fridge is closed",
                          "open the fridge first", error="Fridge is closed", scene="FloorPlan1")
    e2 = kb.upsert_lesson("put the apple on the counter", f"{long_label}: receptacle is full",
                          "clear the counter first", error="receptacle is full", scene="FloorPlan1")
    assert e1["action"] == "insert" and e2["action"] == "insert" and e1["id"] != e2["id"], \
        f"distinct errors under a shared label must be separate lessons, got {e1['action']}/{e2['action']}"
    e3 = kb.upsert_lesson("put the apple on the counter", f"{long_label}: Fridge is closed",
                          "open the fridge first", error="Fridge is closed", scene="FloorPlan1")
    assert e3["action"] == "reinforce" and e3["id"] == e1["id"], \
        f"the same raw error should reinforce its own lesson, got {e3!r}"
    with open(kb.audit_path, "r", encoding="utf-8") as fh:
        _audit_text = fh.read()
    assert "UPSERT procedure" in _audit_text and "UPSERT lesson" in _audit_text, "upserts must be audited"
    print("upserts:", kb.stats(), "| procedure x", r2["entry"]["occurrences"], "| lesson x", l2["entry"]["occurrences"])

    os.remove("_kb_test.json")
    if os.path.exists("_kb_test_audit.log"):
        os.remove("_kb_test_audit.log")
    print("knowledge self-test OK")
