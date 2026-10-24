"""Step 5 / A1 - long-term-memory facade (docs/step5-plan.md section 9).

At stage A1 this exposes the STORAGE layer only: parameters, the store, reset_world, snapshot / frozen open, the tracer
mirror. Working memory, consolidation, retrieval and the adapter are later stages (A2..A7) and plug into the same
`LongTermMemory` object without changing the storage contract.

Offline by construction: imports only the standard library and sim/ltm_store.py; starts no simulator, no socket; the
database path is injected (default sim/step5/ltm.db is created lazily and never used by the selftests).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

try:                                    # run as a script from the repo root (`python sim/x.py`) or imported as package
    from ltm_store import (DEFAULT_DB, Store, StoreError, canon, sha256_hex, file_sha256)
except ImportError:                     # pragma: no cover
    from sim.ltm_store import (DEFAULT_DB, Store, StoreError, canon, sha256_hex, file_sha256)

PARAMS_PATH = Path(__file__).resolve().parent / "step5" / "params.json"


def load_params(path=None) -> dict:
    """The step-5 parameter registry plus its content hash (every run records the hash)."""
    p = Path(path) if path else PARAMS_PATH
    raw = p.read_bytes()
    reg = json.loads(raw.decode("utf-8"))
    reg["_sha256"] = sha256_hex(raw)
    return reg


def retr_words(lang: str) -> dict:
    try:
        from ltm_retrieval import WORDS
    except ImportError:                     # pragma: no cover
        from sim.ltm_retrieval import WORDS
    return WORDS[lang]


def _real_counts(counter, text) -> dict:
    """Per-tokenizer real count of `text` ({tokenizer: n}); a plain callable counter reports a single 'count'."""
    if hasattr(counter, "counts"):
        return {k: int(v) for k, v in counter.counts(text).items()}
    return {"count": int(counter(text))}


def a12_measure(blocks, counter, candidates=None) -> dict:
    """A12 measurement from built blocks (token counts only; no outcome field is read). `blocks` = iterable of memory_block results
    (any mix of languages, levels and models). Returns H = the counter's max real planner-header cost, head = the max real cost of
    the block's own fixed first line (it is paid once per block, so it belongs to the fixed cost: H_eff = H + head is what the
    allocation's H argument should be), r = the max real cost of one place
    record line incl. its separator, s = the max real cost of one task-type section's fixed lines ("Where to look for X:" +
    EXPLORE, separators included), each a max over every block, tokenizer and language seen. Feed r and s to a12_allocation.

    E-fix: the built blocks only hold the records their (old) budget RETAINED, and the longest records are exactly the ones an old
    budget drops, so r read from them alone is biased low. A12 defines r over EVERY eligible measured record: pass
    `candidates=[(ltm, wm), ...]` (the dev / C3 working memories) and every learned place candidate of every block type is also rendered
    ON ITS OWN, in EN and FA, at every level, under an unbounded budget, and enters r and s. `n_candidates_measured` counts them;
    `n_records` stays the number of retained records read from `blocks` (`n_retained`)."""
    blocks = list(blocks)
    n_cand = 0
    if candidates is not None:
        extra_blocks = []
        for ltm_, wm_ in candidates:
            lines = ltm_.place_candidate_lines(wm_)
            n_cand += len({(x["otype"], x["id"]) for x in lines})
            extra_blocks.extend(lines)
        blocks = blocks + extra_blocks
    r_max = s_max = n_rec = n_blk = 0
    for b in blocks:
        if not b.get("text"):
            continue
        n_blk += 1
        lg = b.get("lang", "en")
        W = retr_words(lg)
        for x in b.get("records") or []:
            if str(x["id"]).startswith("proc/"):
                continue
            n_rec += 1
            r_max = max(r_max, max(_real_counts(counter, x["text"] + "\n").values()))
        pre = W["where"].split("{t}")[0]
        for ln in b["text"].split("\n"):
            if pre and ln.startswith(pre):
                s_max = max(s_max, max(_real_counts(counter, ln + "\n" + W["explore"] + "\n").values()))
    head = max(max(_real_counts(counter, retr_words(lg)["head"] + "\n").values()) for lg in ("en", "fa"))
    H = int(counter.header_cost)
    return {"H": H, "head": int(head), "H_eff": H + int(head), "r": int(r_max), "s": int(s_max), "n_records": n_rec, "n_retained": n_rec,
            "n_candidates_measured": n_cand, "n_blocks": n_blk}


def a12_coverage(blocks_by_lang, n_place_candidates, level, counter, alloc) -> dict:
    """A12 minimum-coverage criterion (prereg s16; capacity, not a success rate). Outcome-blind: reads only a built block and
    token counts.  ok iff
      * the level is memory-incapable (exempt, reported), or fewer than N_L learned place candidates exist (exempt, reported), or the
        block holds >= N_L whole place records in EVERY language (record ids are identical across tokenizers by construction since
        the counter is the max over tokenizers; identity across languages is checked here), AND
      * the complete `header + block` fits B_L under EVERY tokenizer x language (the complete-prefix check).
    blocks_by_lang: {"en": block, "fa": block} as returned by LongTermMemory.memory_block. alloc: a12_allocation() output."""
    N = int(alloc["N"][level]) if alloc.get("N") else {"L1": 1, "L2": 2, "L3": 3}[level]
    B = int(alloc["B"][level])
    inc = level in (alloc.get("incapable") or [])

    def place_ids(b):
        return [r["id"] for r in (b.get("records") or []) if not str(r["id"]).startswith("proc/")]
    held = {lg: len(place_ids(b)) for lg, b in blocks_by_lang.items()}
    ids = {lg: place_ids(b) for lg, b in blocks_by_lang.items()}
    same = len({tuple(v) for v in ids.values()}) <= 1
    whole = all(r.get("text") and r["text"] in b["text"] for b in blocks_by_lang.values() for r in (b.get("records") or []))
    hdr = getattr(counter, "header", "")
    real = {lg: (_real_counts(counter, hdr + "\n" + b["text"]) if b.get("text") else {}) for lg, b in blocks_by_lang.items()}
    fits = all(v <= B for per in real.values() for v in per.values())
    if inc:
        reason = "memory-incapable level"
    elif int(n_place_candidates) < N:
        reason = "fewer than N place candidates"
    else:
        reason = None
    cap_ok = reason is not None or all(h >= N for h in held.values())
    ok = bool(cap_ok and fits and same and whole)
    return {"ok": ok, "N": N, "B": B, "held": held, "exempt": reason, "real": real, "ids_identical": same, "whole_records": bool(whole),
            "complete_fits": bool(fits), "n_candidates": int(n_place_candidates), "level": level}


def a12_freeze_check(items, alloc, counter, post_freeze=False) -> dict:
    """Freeze-gate / pre-execution check over many prepared blocks. items: [{"id","level","n_place_candidates","blocks":{lang: block}}].
    Pre-freeze (dev presentations + non-test C3 blocks): not ok blocks the freeze. Post-freeze (every S* test prompt, before
    execution): not ok blocks execution (`blocks_execution`) and needs a dated amendment; frozen parameters are never altered.
    Empty-block rates per level x language are reported, never decisive. An empty item list is a failure (a vacuous pass is not a pass)."""
    rows, fails = [], []
    for it in items:
        c = a12_coverage(it["blocks"], it["n_place_candidates"], it["level"], counter, alloc)
        rows.append(dict(c, id=it.get("id")))
        if not c["ok"]:
            fails.append({"id": it.get("id"), "level": it["level"], "held": c["held"], "N": c["N"], "complete_fits": c["complete_fits"],
                          "ids_identical": c["ids_identical"], "whole_records": c["whole_records"]})
    by_level = {}
    for r in rows:
        d = by_level.setdefault(r["level"], {"n": 0, "ok": 0, "exempt": 0, "empty": {}})
        d["n"] += 1
        d["ok"] += r["ok"]
        d["exempt"] += r["exempt"] is not None
    for it in items:
        d = by_level[it["level"]]
        for lg, b in it["blocks"].items():
            e = d["empty"].setdefault(lg, [0, 0])
            e[0] += 0 if b.get("text") else 1
            e[1] += 1
    for d in by_level.values():
        d["empty_rate"] = {lg: round(a / n, 4) for lg, (a, n) in d.pop("empty").items()}
    ok = bool(rows) and not fails
    return {"ok": ok, "n": len(rows), "failures": fails, "by_level": by_level, "memory_incapable": list(alloc.get("incapable") or []),
            "stage": "post_freeze" if post_freeze else "pre_freeze", "blocks_execution": bool(post_freeze and not ok),
            "blocks_freeze": bool((not post_freeze) and not ok), "reason": None if ok else ("no_items" if not rows else "coverage_failed"),
            "outcome_blind": True}


class ListTracer:
    """In-memory mirror destination honouring the destination contract: emit() de-duplicates atomically under its key
    and rejects a differing payload for an existing key. Used by tests and as the reference implementation of the
    contract a real trace adapter must meet."""

    def __init__(self, name: str = "list", has_probe: bool = True, fail_on=None):
        self.name = name
        self.records: dict[str, dict] = {}
        self.emit_calls = 0
        self._has_probe = has_probe
        self.fail_on = fail_on          # callable(key) -> raise to simulate an outage
        self._lock = threading.Lock()
        if not has_probe:
            self.has = None             # type: ignore[assignment]  (destination offers no probe)

    def has(self, key: str) -> bool:    # replaced by None when has_probe is False
        return key in self.records

    def emit(self, key: str, record: dict):
        self.emit_calls += 1
        if self.fail_on:
            self.fail_on(key)
        with self._lock:
            if key in self.records:
                if canon(self.records[key]) != canon(record):
                    raise StoreError(f"destination already holds a different record under {key}")
                return                  # idempotent: durable unique key
            self.records[key] = record


class LongTermMemory:
    """Facade. `db_path` is always injectable; `tracer=None` means no mirroring (the default)."""

    _overlay = None                 # B1: RAM-only test epoch (frozen stores only); class default so open_frozen needs no init
    _retriever_obj = None
    fragment_ledger = None          # C3: sidecar of fragment failed reuses (ltm_fragments.FragmentLedger); None = empty
    last_proc_report = None         # C3: what the procedures channel decided for the last memory_block (audit / coverage)
    package_builder = None          # B7: ltm_package.PackageBuilder for the "package" channel; None = the channel is unavailable (default off)

    def __init__(self, db_path=None, kitchen: str | None = None, params: dict | None = None, tracer=None,
                 embedder=None, count_tokens=None, fault_hook=None, target_version=None):
        if not kitchen:                 # A7-review: no silent FloorPlan1 default - the kitchen scope is always declared
            raise StoreError("LongTermMemory needs an explicit kitchen label")
        self.kitchen = kitchen
        self.params = params if params is not None else load_params()
        self.embedder = embedder
        self.count_tokens = count_tokens
        kw = {} if target_version is None else {"target_version": target_version}
        self.store = Store(db_path if db_path is not None else DEFAULT_DB, tracer=tracer, fault_hook=fault_hook, **kw)
        self.frozen = False

    # -- passthrough ------------------------------------------------------------------------------------------
    def commit(self, *a, **k):
        return self.store.commit(*a, **k)

    def apply(self, c):
        return self.store.apply(c)

    def clock(self) -> int:
        return self.store.clock()

    def state_hash(self) -> str:
        return self.store.state_hash()

    def export_pending(self, tracer=None, limit=None):
        return self.store.export_pending(tracer, limit)

    def check_integrity(self):
        return self.store.check_integrity()

    def stats(self) -> dict:
        s = self.store.stats()
        s["params_sha256"] = self.params.get("_sha256")
        return s

    def close(self):
        self.store.close()

    # -- worlds -----------------------------------------------------------------------------------------------
    def current_epoch(self) -> int:
        if self._overlay is not None:
            return self._overlay["epoch"]
        rows = self.store.select("epochs", where={"kitchen": self.kitchen}, order="-epoch", limit=1)
        return rows[0]["epoch"] if rows else 0

    def frame_of_epoch(self, epoch: int):
        """The coordinate frame declared for `epoch` (None when the epoch is 0 / unknown). Honours the test overlay."""
        if self._overlay is not None and epoch == self._overlay["epoch"]:
            return self._overlay["frame"]
        rows = self.store.select("epochs", where={"kitchen": self.kitchen, "epoch": epoch}) if epoch else []
        return rows[0]["frame"] if rows else None

    def begin_test_epoch(self, frame: str = "f0", map_rev: str = "m0") -> dict:
        """B1 (plan B-e): an independent test on a FROZEN snapshot starts at snapshot epoch + 1, held in RAM only (a
        frozen store refuses reset_world). Every test calls this on its own fresh read-only copy, so test order can
        never accumulate resets; practice claims then read as 'maybe' (T2), exactly as after a task reset."""
        if not self.frozen:
            raise StoreError("begin_test_epoch is for frozen stores; practice uses reset_world")
        self._overlay = None
        rows = self.store.select("epochs", where={"kitchen": self.kitchen}, order="-epoch", limit=1)
        base = rows[0]["epoch"] if rows else 0
        self._overlay = {"epoch": base + 1, "frame": frame, "map_rev": map_rev, "base": base}
        return dict(self._overlay)

    def end_test_epoch(self) -> None:
        self._overlay = None

    def reset_world(self, reset_id: str, reason: str = "task_reset", frame: str = "f0", map_rev: str = "m0") -> dict:
        """Declare a world reset: epoch += 1, locations become 'maybe' from the next epoch (rules in the retrieval stage).
        Idempotent per `reset_id`: a retry after a crash neither double-resets nor loses the reset."""
        if self.frozen:
            raise StoreError("frozen store")
        c = self.store.select("commit_receipt", where={"event_id": f"reset:{reset_id}"})
        if c:                                       # already committed: report the existing epoch row
            return {"status": "duplicate", "event_id": f"reset:{reset_id}", "epoch": self._epoch_of(f"reset:{reset_id}")}
        epoch = self.current_epoch() + 1
        res = self.store.commit(f"reset:{reset_id}", "reset_world", [
            {"table": "epochs", "op": "insert",
             "values": {"kitchen": self.kitchen, "epoch": epoch, "frame": frame, "map_rev": map_rev, "reason": reason,
                        "E": "$E"}}], stamp=self.store.clock())
        res["epoch"] = epoch
        return res

    def _epoch_of(self, event_id: str) -> int | None:
        rows = self.store.select("write_log", where={"event_id": event_id})
        if not rows:
            return None
        return json.loads(rows[0]["payload"])["values"]["epoch"]

    # -- A2: episodes, consolidation, retrieval -------------------------------------------------------------------
    def begin_episode(self, episode: str, mode: str = "practice", world: str = "reset"):
        """A fresh WorkingMemory. The caller feeds it events (`wm.ingest`) starting with an `episode_start`; nothing is
        written to the store until `end_episode`. `mode`/`world` are only remembered for convenience."""
        try:
            from ltm_wm import WorkingMemory
        except ImportError:                 # pragma: no cover
            from sim.ltm_wm import WorkingMemory
        wm = WorkingMemory(episode)
        wm.declared_mode, wm.declared_world = mode, world
        return wm

    def end_episode(self, wm, admission=None, extensions=(), useful=None) -> dict:
        """Consolidate (practice, complete, valid episodes only) in ONE transaction; test / probe / frozen write nothing."""
        try:
            from ltm_consolidate import consolidate
        except ImportError:                 # pragma: no cover
            from sim.ltm_consolidate import consolidate
        return consolidate(self, wm, admission=admission, extensions=extensions, useful=useful)

    def where_is(self, otype: str, wm=None, e_now=None, prior=None, channels=None) -> dict:
        try:
            from ltm_retrieve import where_is
        except ImportError:                 # pragma: no cover
            from sim.ltm_retrieve import where_is
        return where_is(self, otype, wm=wm, e_now=e_now, prior=prior, channels=channels)

    # -- B1: the composed memory block (the ONE renderer input of the planner contract, gate 12.11) ------------------
    def retriever(self):
        if self._retriever_obj is None:
            try:
                from ltm_retrieval import Retriever
            except ImportError:             # pragma: no cover
                from sim.ltm_retrieval import Retriever
            self._retriever_obj = Retriever(params=self.params, embedder=self.embedder, count_tokens=self.count_tokens)
        return self._retriever_obj

    def block_types(self, wm) -> list:
        """Object types the task is about, in order of first mention, from the parsed task concepts (never the gold plan).
        Only movable ('useful') types: where a fridge is never needs a memory line."""
        try:
            from ltm_consolidate import load_useful_types
        except ImportError:                 # pragma: no cover
            from sim.ltm_consolidate import load_useful_types
        useful = load_useful_types()
        tc = wm.first("task_context") if wm is not None else None
        cpt = (tc or {}).get("concepts") or {}
        roles = cpt.get("roles") or []
        out = []
        if cpt.get("ok"):                                  # a fully resolved parse: the roles, in order
            for r in roles:
                for f in ("obj", "src"):
                    t = r.get(f)
                    if t in useful and t not in out:
                        out.append(t)
        if not cpt.get("ok") and tc and tc.get("text"):   # any parse the lexicon cannot fully resolve: surface type mentions, in order
            try:
                import ltm_concepts as lc
            except ImportError:             # pragma: no cover
                from sim import ltm_concepts as lc
            toks, lex, i = lc.prep(tc["text"]), lc.default_lexicon(), 0
            while i < len(toks):
                n, t = lc._match_type(toks, i, lex)
                if n:
                    if t in useful and t not in out:
                        out.append(t)
                    i += n
                else:
                    i += 1
        return out

    def proc_candidates(self, wm, holding=None) -> dict:
        """C3 (docs/step5-plan.md s13 C2-b..c): the procedure lines the plan-time gate offers for the CURRENT instruction -
        whole-intent trusted procedures (frozen 8.3 gate, `Retriever.retrieve_procedures`) first, then role-level fragments in
        instruction order. Reads only: the store, the parsed concepts of this command, the declared hand state."""
        try:
            import ltm_fragments as lf
            import ltm_concepts as lc
            from ltm_retrieval import load_records
        except ImportError:                 # pragma: no cover
            from sim import ltm_fragments as lf
            from sim import ltm_concepts as lc
            from sim.ltm_retrieval import load_records
        tc = wm.first("task_context") if wm is not None else None
        cpt = (tc or {}).get("concepts") or {}
        rep = {"whole": [], "fragments": [], "rejected": [], "resolved": bool(cpt.get("ok")), "n_roles": len(cpt.get("roles") or []),
               "n_unhidden_fragments_total": None}
        if not (cpt.get("roles")):
            return {"items": [], "report": rep}
        retr = self.retriever()
        items = []
        recs = load_records(self.store, self.kitchen)["procedures"]
        if recs and cpt.get("ok"):
            res = retr.retrieve_procedures(cpt, recs, kitchen=self.kitchen, wm_snapshot={"holding": holding}, mode="practice" if not self.frozen else "test")
            rep["rejected"] = [{"id": r["id"], "first": r["first"]} for r in res["rejected"]][:20]
            for r in res["steer"]:
                items.append({"id": r["proc"], "kind": "procedure", "rec": r})
                rep["whole"].append(r["proc"])
        idx = lf.build_index(self.store, self.kitchen, self.fragment_ledger)
        rep["index"] = {"n_fragments": len(idx["fragments"]), "discards": idx["discards"]}
        rep["n_unhidden_fragments_total"] = sum(1 for e in idx["fragments"].values() if not e["hidden"])
        roles = [r for r in (cpt.get("roles") or [])] if cpt.get("ok") else []
        for e in lf.candidates_for_roles(idx, self.kitchen, roles):
            items.append({"id": e["fid"], "kind": "fragment", "frag": e})
            rep["fragments"].append({"fid": e["fid"], "support": e["support"], "procs": e["procs"][:3], "sig": [list(x) for x in e["sig"]]})
        return {"items": items, "report": rep}

    def place_candidate_count(self, wm, channels=("places",)) -> int:
        """Learned place candidates over the block's types (what A12 coverage compares against N_L); no counter involved."""
        ch = frozenset(c for c in channels if c != "procedures")
        return sum(len(self.where_is(t, wm, channels=ch)["candidates"]) for t in self.block_types(wm)) if ch else 0

    def place_candidate_lines(self, wm, channels=("places",), levels=("L1", "L2", "L3")) -> list:
        """E-fix (A12 measurement): every learned place candidate of every block type, each rendered ON ITS OWN under an unbounded budget,
        in EN and FA, at each level. Returns block-shaped dicts {text, records, lang, level, otype, id} that `a12_measure` reads. Nothing
        is dropped by any retention budget, so the longest records reach the measurement. Outcome-blind; no counter involved."""
        retr = self.retriever()
        ch = frozenset(c for c in channels if c != "procedures")
        out = []
        for t in (self.block_types(wm) if ch else []):
            ans = self.where_is(t, wm, channels=ch)
            for c in ans["candidates"]:
                sub = dict(ans, candidates=[c])
                for lv in levels:
                    for lg in ("en", "fa"):
                        b = retr.memory_block(where=sub, level=lv, lang=lg, otype=t, budget_tokens=10 ** 6)
                        out.append({"text": b["text"], "records": b["records"], "lang": lg, "level": lv, "otype": t, "id": f"{t}/{c['id']}"})
        return out

    def package_block(self, wm, level: str = "L2", lang: str = "en", holding=None, summary_lines=()) -> dict:
        """B7 (PACKAGE-SPEC 14.1): the assistance package of ONE presentation, shaped like memory_block's result.

        Reads only the instruction text of the task context, its language, the holding state and the summary lines; `level` is echoed
        for the caller and is NOT an input to retrieval. Refuses unless a PackageBuilder was set (`ltm.package_builder = ...`) AND its
        params say enabled=true: the hook is default-off until the amendment pins it, then switched on for every arm at once."""
        b = self.package_builder
        if b is None:
            raise ValueError("channel 'package' needs ltm.package_builder (an ltm_package.PackageBuilder); none is set")
        if not b.params.get("enabled"):
            raise ValueError("channel 'package' is disabled (package_params.json enabled=false): the amendment switches it on for every arm together")
        tc = wm.first("task_context") if wm is not None else None
        text = (tc or {}).get("text")
        if not text:
            raise ValueError("channel 'package' needs a task_context event with the instruction text in the working memory")
        try:
            import ltm_package as LPK
        except ImportError:             # pragma: no cover
            from sim import ltm_package as LPK
        pkg = b.build(LPK.make_query(text, lang, holding, summary_lines))
        return LPK.memory_block_compat(pkg, level=level)

    def memory_block(self, wm, level: str = "L2", lang: str = "en", channels=("places",), types=None,
                     budget_tokens=None, joint: bool = True, holding=None, retention=None, measurement_only: bool = False,
                     summary_lines=()) -> dict:
        """Where-to-look lines for every task type under ONE aggregate token budget; learned candidates only (the channel filter
        runs inside where_is, before competition and top-k). `text == ""` when nothing learned applies, so the planner prompt
        is then byte-identical to the no-memory prompt.

        Retention is decided JOINTLY for English and Persian (`joint=True`): a type's section keeps its first k candidates for
        the largest k that renders completely in BOTH languages inside the budget, so EN and FA always retain the same record
        identities in the same order with the same qualifiers, also after truncation. Whole records only; the first type that
        does not fit stops the block (later types are dropped, never reordered ahead). The where lines get `where_share` of the
        aggregate budget, as declared in the registry, not of each section.

        `retention` (A12, prereg s16): None reads the registry switch retrieval.retention (absent = "c2e", the behaviour above). "a12":
        places get the allowance A_L, procedures B_L - A_L, the planner header is paid once, every fit test is the REAL count
        (max over tokenizers) of the complete `header + block`, and the retained ids are identical across models and languages.

        B7: when "package" is among the channels the result is the assistance package (`package_block`); it cannot be mixed with the
        place / procedure channels, because the package already contains their records under its own budget."""
        if "package" in channels:
            if set(channels) != {"package"}:
                raise ValueError("channel 'package' cannot be combined with other channels: %s" % sorted(channels))
            return self.package_block(wm, level, lang, holding, summary_lines)
        retr = self.retriever()
        mode = retention if retention is not None else (retr.params.get("retrieval", {}).get("retention") or "c2e")
        if mode not in ("c2e", "a12"):
            raise ValueError(f"unknown retention {mode!r}")
        a12 = mode == "a12"
        A = B = None
        if a12:
            cfg = retr.params.get("retrieval", {}).get("a12")
            ctr = self.count_tokens
            if not cfg:
                raise ValueError("retention a12 needs the A12 block under the retrieval section of the registry (python sim/final_test/matrix/retention_a12.py --write)")
            if not joint:
                raise ValueError("retention a12 is joint over languages by definition")
            if not (callable(ctr) and hasattr(ctr, "header_cost") and hasattr(ctr, "header")):
                raise ValueError("retention a12 needs the real max-token counter (header_cost + header); an estimate cannot back it")
            tks = set(getattr(ctr, "tokenizers", None) or [])
            if not ("gemma" in tks and len(tks) >= 2) and not measurement_only:      # A12: the max over Gemma AND the API tokenizer
                raise ValueError(f"retention a12 needs the max over BOTH tokenizers (Gemma and the API one); the counter has {sorted(tks)}. "
                                 "A single-tokenizer counter is allowed only with measurement_only=True (practice / rehearsal, recorded)")
            if getattr(retr, "_count", None) is not ctr:
                raise ValueError("retention a12: the cached Retriever was built with a different counter than self.count_tokens "
                                 "(set count_tokens BEFORE the first retriever() / memory_block call)")
            B = int(budget_tokens) if budget_tokens is not None else int(cfg["B"][level])
            A = min(B, int(cfg["A"][level]))
        types = list(types) if types is not None else self.block_types(wm)
        ch = frozenset(channels)
        want_proc = "procedures" in ch
        ch = frozenset(c for c in ch if c != "procedures")          # where_is knows only the place channels
        # E-fix: "c2e" always reads the REGISTERED budgets, even when the registry carries an a12 block that raised retr.budgets
        total = int(budget_tokens if budget_tokens is not None else (retr.budgets[level] if a12 else retr.registered_budgets[level]))
        T = retr.count_tokens
        total = max(1, total - int(getattr(self.count_tokens, "header_cost", 0) or 0))   # C2-e: the prompt header is paid once, out of the budget
        shrinks = 0

        def full(txt):                       # the complete planner-visible text: header + block (what the real counter must fit)
            return self.count_tokens.header + "\n" + txt

        def fits_places(txt):
            return T(full(txt)) <= A if a12 else T(txt) <= total
        langs = ("en", "fa") if joint else (lang,)
        share = float(retr.where_share)
        head = {lg: retr_words(lg)["head"] for lg in langs}
        parts = {lg: [] for lg in langs}                       # accepted section bodies per language
        recs, dropped, shown, stopped = [], [], [], False

        def used_where():
            return max((T("\n".join(parts[lg])) if parts[lg] else 0) for lg in langs)

        def text_of(lg, extra=None):
            body = parts[lg] + ([extra] if extra else [])
            return head[lg] + "\n" + "\n".join(body) if body else ""

        for t in (types if ch else []):
            ans = self.where_is(t, wm, channels=ch)
            cands = ans["candidates"]
            if not cands:
                continue
            if stopped:
                dropped += [{"id": f"{t}/{c['id']}", "reason": "after_budget_stop", "tokens": 0} for c in cands]
                continue
            if a12:
                bud = int(-(-A // share)) + 1                      # the inner block never limits; the complete-count test below does
            else:
                left = int(total * share) - used_where()
                bud = max(1, int(-(-max(left, 1) // share)) + 1) if left > 0 else 1
            got = None
            for k in range(len(cands), 0, -1):
                sub = dict(ans, candidates=cands[:k])
                blks = {lg: retr.memory_block(where=sub, level=level, lang=lg, otype=t, budget_tokens=bud) for lg in langs}
                if any((not b["records"]) or b["dropped"] or b.get("overflow") for b in blks.values()):
                    continue
                secs = {lg: b["text"].split("\n", 1)[1] for lg, b in blks.items()}
                if not all(fits_places(text_of(lg, secs[lg])) for lg in langs):
                    shrinks += a12
                    continue
                got = (k, blks, secs)
                break
            if got is None:
                stopped = True
                dropped += [{"id": f"{t}/{c['id']}", "reason": "no_record_fits", "tokens": 0} for c in cands]
                continue
            k, blks, secs = got
            for lg in langs:
                parts[lg].append(secs[lg])
            shown.append(t)
            recs.append([dict(r, id=f"{t}/{r['id']}", otype=t) for r in blks[lang if lang in blks else langs[0]]["records"]])
            dropped += [{"id": f"{t}/{c['id']}", "reason": "budget", "tokens": 0} for c in cands[k:]]
        # ---- C3: the procedures channel (whole-intent procedures + fragments). Retention: the largest prefix of whole records
        # whose COMPLETE block fits `total` in EVERY language (and, through the injected counter, under every tokenizer).
        proc_recs = []
        if want_proc:
            pc = self.proc_candidates(wm, holding)
            self.last_proc_report = pc["report"]
            items = pc["items"]
            if items:
                try:
                    import ltm_fragments as lf
                except ImportError:         # pragma: no cover
                    from sim import ltm_fragments as lf
                lex = retr.lex
                lines = {lg: [retr.render_proc(i["rec"], lg) if i["kind"] == "procedure" else lf.render_fragment(i["frag"], lg, lex)
                              for i in items] for lg in langs}
                hdr = {lg: retr_words(lg)["procs"] for lg in langs}
                best = 0
                for k in range(len(items), 0, -1):
                    sec = {lg: hdr[lg] + "\n" + "\n".join(lines[lg][:k]) for lg in langs}
                    if a12:
                        okk = all(T(full(text_of(lg, sec[lg]))) <= B and T(sec[lg]) <= B - A for lg in langs)
                    else:
                        okk = all(T(text_of(lg, sec[lg])) <= total for lg in langs)
                    if okk:
                        best = k
                        break
                    shrinks += a12
                for lg in langs:
                    if best:
                        parts[lg].append(hdr[lg] + "\n" + "\n".join(lines[lg][:best]))
                lg0 = lang if lang in langs else langs[0]
                proc_recs = [{"id": f"proc/{i['id']}", "section": "procs", "kind": i["kind"], "otype": None,
                              "text": lines[lg0][n]} for n, i in enumerate(items[:best])]
                dropped += [{"id": f"proc/{i['id']}", "reason": "budget", "tokens": 0} for i in items[best:]]
        lg0 = lang if lang in parts else langs[0]
        text = text_of(lg0)
        records = [r for rr in recs for r in rr] + proc_recs
        extra = {}
        if a12:
            real = {lg: T(full(text_of(lg))) if parts[lg] else 0 for lg in langs}
            extra = {"retention": "a12", "A": A, "B": B, "H": int(self.count_tokens.header_cost), "prefix_shrinks": shrinks,
                     "tokens_with_header": real[lg0], "complete_ok": all(v <= B for v in real.values()),
                     "tokenizers": sorted(getattr(self.count_tokens, "tokenizers", None) or []), "measurement_only": bool(measurement_only)}
            total = B
        return {**extra, "text": text, "records": records, "dropped": dropped, "tokens": T(text) if text else 0, "budget": total,
                "proc_report": self.last_proc_report if want_proc else None, "types": shown, "level": level, "lang": lang, "channels": sorted(ch | ({"procedures"} if want_proc else set())), "empty": not text, "joint": joint,
                "token_claim": retr.token_claim, "tokenizer": retr.tokenizer, "epoch": self.current_epoch()}

    def lesson_for(self, wm, failure_key: dict, available, retries_left: int, revalidating=()):
        """Zero or one lesson remedy for a recovery attempt (plan 7.5; mode-checked: tentative lessons only in practice).
        Thin facade over ltm_lessons.lesson_for (A7-review), so workflow B's recovery hook needs only this object."""
        try:
            from ltm_lessons import lesson_for
        except ImportError:                 # pragma: no cover
            from sim.ltm_lessons import lesson_for
        return lesson_for(self, wm, failure_key, available, retries_left, revalidating)

    def unmerge(self, place: str, reason: str = "manual_correction") -> dict:
        try:
            from ltm_consolidate import unmerge
        except ImportError:                 # pragma: no cover
            from sim.ltm_consolidate import unmerge
        return unmerge(self, place, reason)

    def habits(self, otype: str, e_now=None, general_validation=None) -> dict:
        """Derived habit state for one type: kitchen preferences + the general-habit decision (for audit / tests)."""
        try:
            from ltm_habits import alias_map, collapse_trials, general_habit, kitchen_prefs
        except ImportError:                 # pragma: no cover
            from sim.ltm_habits import alias_map, collapse_trials, general_habit, kitchen_prefs
        E = self.clock() if e_now is None else int(e_now)
        col = collapse_trials(self.store.select("search_trial"), alias_map(self.store))
        return {"kitchen": kitchen_prefs(col, self.params, E, otype, self.kitchen),
                "general": general_habit(col, self.params, E, otype, validation_kitchen=general_validation)}

    # -- snapshot / frozen --------------------------------------------------------------------------------------
    def snapshot(self, directory, overwrite: bool = False) -> dict:
        extra = {"params_sha256": self.params.get("_sha256"), "kitchen": self.kitchen,
                 "channels": ["lexical"]}                       # dense only after gate G-dense (plan 8.5)
        res = self.store.snapshot(directory, extra=extra, overwrite=overwrite)
        if self.fragment_ledger is not None:                       # C3: the sidecar travels with (and is frozen by) the snapshot
            try:
                from ltm_fragments import LEDGER_NAME
            except ImportError:             # pragma: no cover
                from sim.ltm_fragments import LEDGER_NAME
            res = dict(res)
            res["fragments_sha256"] = self.fragment_ledger.save(directory)
        return res

    @classmethod
    def open_frozen(cls, directory, testlog_path, kitchen: str | None = None, params=None) -> "LongTermMemory":
        """The kitchen comes from the snapshot manifest; a caller-given kitchen must agree with it (A7-review)."""
        man_k = (Store.read_manifest(directory).get("extra") or {}).get("kitchen")
        if kitchen and man_k and kitchen != man_k:
            raise StoreError(f"snapshot is for kitchen {man_k!r}, not {kitchen!r}")
        kitchen = kitchen or man_k
        if not kitchen:
            raise StoreError("snapshot manifest names no kitchen; pass kitchen= explicitly")
        self = cls.__new__(cls)
        self.kitchen = kitchen
        self.params = params if params is not None else load_params()
        self.embedder = None
        self.count_tokens = None
        self.store = Store.open_frozen(directory, testlog_path)
        self.frozen = True
        try:
            from ltm_fragments import FragmentLedger
        except ImportError:                 # pragma: no cover
            from sim.ltm_fragments import FragmentLedger
        self.fragment_ledger = FragmentLedger.load(directory)
        return self

    def log_test(self, table: str, values: dict) -> dict:
        return self.store.log_test(table, values)

    @staticmethod
    def clear_memory(db_path) -> None:
        """Arm `none`: remove the database file (a new empty one is created lazily at the same injected path)."""
        p = Path(db_path)
        for suffix in ("", "-journal"):
            q = Path(str(p) + suffix)
            if q.exists():
                q.unlink()
