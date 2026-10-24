"""Step 5 / A4 - retrieval for procedures and lessons + the budgeted memory block (docs/step5-plan.md sections 8.2-8.6).

Offline, deterministic, no LLM, no wall clock, no randomness. NEW file; nothing in step 4 and no existing ltm_* module is edited.

    exact-key path   `locate()` / `habit_prefs()` delegate to ltm_retrieve.where_is / ltm_habits (SQL keyed lookups; the embedder is
                     never called on this path - tested with a spy).
    lexical channel  clause-indexed, role-bound concept tokens (ltm_concepts) with idf weighting, polarity always explicit; surface
                     tokens count 0.25 and can NEVER establish eligibility.
    dense channel    behind `Embedder` (`model_rev`, `embed(list[str]) -> vectors`); EmbeddingCache keyed (sha256(key text), model_rev);
                     a vector is published only against the key hash it embedded (late writes for a superseded key are rejected);
                     lazily used, any failure degrades to lexical and the log says so.
    RRF              score = sum_c 1/(rrf_k + rank_c), 1-based ranks, quantised to 1e-6, ties by record id.
    steering gate    DOMINATES the fusion. Applied to every retrieved record (no top-K before the gate). Retrieval scores, channel
                     availability, record length and insertion order can never make an ineligible record steer (tested).
    budgets          tokens per level; where_is lines first (where_share), whole records only, prefix STOP per section, mandatory EXPLORE
                     line reserved, oversize records excluded with a logged reason, the FINAL serialised block is what is counted.
    logs             one retrieval record per query: retrieved vs eligible vs rejected(+first failed condition) vs rendered vs dropped
                     vs applied (mark_applied is idempotent and monotone: selected < attempted < succeeded).

Contracts this module assumes of A5 (procedures / lessons records; see the ledger):
    procedure row: proc, kitchen, intent_key ({"roles":[...]} or a role list), intent_text, body ({"steps":[{"verb","otype","to","pre"}],
                   "pre": optional whole-procedure preconditions}), status, success_uses, failed_reuses, optional boolean `contradicted`.
    lesson row:    lesson, kitchen, conditions {"cause","verb","otype","rtype","state"}, action, status, helpful_units, unhelpful_units,
                   E_created, revalidate. Hypothesis prose is never read, never rendered.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import unicodedata
from pathlib import Path

try:
    import ltm_concepts as lc
    import ltm_retrieve as lrt
except ImportError:                     # pragma: no cover
    from sim import ltm_concepts as lc
    from sim import ltm_retrieve as lrt

RETRIEVAL_VERSION = "1"
SCORE_QUANT = 1e-6                      # dense scores are quantised before ranking (runtime float noise must not reorder)
SURFACE_WEIGHT = 0.25                   # surface tokens are a recall aid only
PROC_SHARE = 0.6                        # declared: share of the post-where remainder offered to procedures (lessons get the rest + unspent)
RECORD_CAP_FRAC = 0.6                   # declared: a single record above this fraction of the block budget is excluded (oversize)
MAX_APPS_PER_FAILURE = 1                # at most one lesson remedy per failure event
PROC_STATUS_STEER = ("trusted",)
PROC_STATUS_SUGGEST = ("candidate", "demoted")
STATE_ORDER = {"selected": 1, "attempted": 2, "succeeded": 3}
ACTION_IMPLIES = {"open_first": "open"}
FAILURE_FIELDS = ("cause", "verb", "otype", "rtype", "state")

# fixed bilingual renderings (enumerated values only; never free text)
TIER_LABEL = {
    "en": {"T0": "current sighting", "T1": "placed there by me", "T2": "maybe, older evidence", "T3": "habit, often found there",
           "T3b": "doubtful, contradicted {c}x", "T4": "default guess"},
    "fa": {"T0": "مشاهدهٔ فعلی", "T1": "خودم آنجا گذاشته‌ام", "T2": "شاید، شواهد قدیمی", "T3": "عادت، معمولاً آنجا است",
           "T3b": "مشکوک، {c} بار رد شده", "T4": "حدس پیش‌فرض"},
}
CAUSE_NAME = {
    "en": {"not_detected": "target not detected", "action_refused": "action refused", "unreachable": "unreachable",
           "wrong_target": "wrong target", "postcondition_unconfirmed": "result unconfirmed", "budget_exhausted": "budget exhausted",
           "plan_goal_unmet": "goal unmet", "unknown": "unknown failure"},
    "fa": {"not_detected": "هدف دیده نشد", "action_refused": "عمل رد شد", "unreachable": "دسترسی ممکن نبود",
           "wrong_target": "هدف اشتباه", "postcondition_unconfirmed": "نتیجه تأیید نشد", "budget_exhausted": "بودجه تمام شد",
           "plan_goal_unmet": "هدف محقق نشد", "unknown": "شکست نامشخص"},
}
ACTION_NAME = {
    "en": {"try_other_instance": "try another instance", "change_pose": "change pose", "approach_closer": "approach closer",
           "crouch": "crouch", "open_first": "open it first", "search_elsewhere": "search elsewhere", "none": "no remedy"},
    "fa": {"try_other_instance": "نمونهٔ دیگری را امتحان کن", "change_pose": "حالت بدن را عوض کن", "approach_closer": "نزدیک‌تر شو",
           "crouch": "خم شو", "open_first": "اول بازش کن", "search_elsewhere": "جای دیگری بگرد", "none": "بدون راه‌حل"},
}
WORDS = {
    "en": {"head": "Memory (may be outdated; verify before acting):", "where": "Where to look for {t}:", "explore": "Otherwise explore.",
           "any": "any {n}", "spot": "a spot near {x}", "age": "age {a}", "support": "support {s}", "contra": "contradicted {c}x",
           "neg": "{p}: not detected after a qualified inspection ({a} episodes ago)",
           "neg_task": "{p}: already inspected in this task", "procs": "Worked before:", "proc": "For \"{k}\" ({st}, {n} successes): {steps}",
           "sugg": "Unverified suggestion for \"{k}\" ({st}): {steps}", "lessons": "Failure hints:",
           "lesson": "If \"{c}\" while trying to {v} {o}: {a} ({st}; helpful {h}, unhelpful {u})", "into": "into", "then": "then",
           "status": {"trusted": "trusted", "candidate": "candidate", "demoted": "demoted", "retired": "retired"},
           "lstatus": {"supported": "supported", "tentative": "tentative", "contradicted": "contradicted", "retired": "retired"}},
    "fa": {"head": "حافظه (ممکن است قدیمی باشد؛ پیش از عمل بررسی کن):", "where": "جای {t} را اینجا بگرد:", "explore": "در غیر این صورت جست‌وجو کن.",
           "any": "هر {n}", "spot": "نقطه‌ای نزدیک {x}", "age": "{a} اپیزود پیش", "support": "پشتیبانی {s}", "contra": "{c} بار رد شده",
           "neg": "{p}: پس از بازرسی معتبر دیده نشد ({a} اپیزود پیش)",
           "neg_task": "{p}: در همین کار بازرسی شده", "procs": "پیش‌تر جواب داده:", "proc": "برای «{k}» ({st}، {n} موفقیت): {steps}",
           "sugg": "پیشنهاد تأییدنشده برای «{k}» ({st}): {steps}", "lessons": "نکته‌های شکست:",
           "lesson": "اگر «{c}» هنگام {v} {o}: {a} ({st}؛ مفید {h}، نامفید {u})", "into": "در", "then": "سپس",
           "status": {"trusted": "معتبر", "candidate": "نامزد", "demoted": "تنزل‌یافته", "retired": "بازنشسته"},
           "lstatus": {"supported": "تأییدشده", "tentative": "آزمایشی", "contradicted": "ردشده", "retired": "بازنشسته"}},
}


A12_N_DEFAULT = {"L1": 1, "L2": 2, "L3": 3}      # declared minimum whole place records per level (prereg A12; a design choice)


def a12_allocation(H, r, N=None, B=None, where_share=0.4, cap_factor=2.0, s=0) -> dict:
    """A12 (prereg s16, 2026-10-11): token-derived places allowance. Pure; no I/O; kitchen-independent.

    need_L = H + N_L * (r + s)  (H = max real header cost, r = max real cost of one place record incl. separators, s = optional
    worst-case per-record section overhead: a record may sit alone in its own task-type section, which also pays the "Where to look
    for X:" line and the EXPLORE line; default 0 = the amendment as written, and the coverage check then decides).  A_L = max(where_share * B_L, need_L).  If need_L > B_L the budget
    is raised to the smallest multiple of 10 that fits, at most cap_factor x the registered value, and A_L is recomputed; if it still
    does not fit the level is memory-incapable (E-fix: B_L then stays at its REGISTERED value, A_L = where_share * B_L, the cap that
    would not have sufficed is only reported under "not_applied"; nothing is tuned further).
    Returns {"A","B","N","need","raised","incapable","not_applied"}; no outcome ever enters."""
    N = dict(A12_N_DEFAULT if N is None else N)
    B = dict({"L1": 150, "L2": 300, "L3": 450} if B is None else B)

    def up(x):
        return int(math.ceil(round(float(x), 6)))
    out = {"A": {}, "B": {}, "N": {}, "need": {}, "raised": {}, "incapable": [], "not_applied": {}}
    for lv in sorted(B):
        b0, n = int(B[lv]), int(N.get(lv, A12_N_DEFAULT.get(lv, 1)))
        need = up(H + n * (r + s))
        cap = int(cap_factor * b0)
        b = b0
        if need > b0:
            if need <= cap:
                b = min(-(-need // 10) * 10, cap)
                out["raised"][lv] = [b0, b]
            else:                                   # E-fix: a memory-incapable level is NOT raised (A12: "not tuned further")
                out["incapable"].append(lv)
                out["not_applied"][lv] = [b0, cap]  # the cap that would not have sufficed; the registered B_L stays
        out["B"][lv], out["N"][lv], out["need"][lv] = b, n, need
        out["A"][lv] = up(where_share * b) if lv in out["incapable"] else max(up(where_share * b), need)
    return out


def quant(x) -> float:
    return round(float(x) / SCORE_QUANT) * SCORE_QUANT


def sha(text) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", str(text)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# token counting
# ---------------------------------------------------------------------------------------------------------------
def estimate_tokens(text, params=None) -> int:
    """Declared fallback: ceil(latin_chars/lcpt + persian_chars*factor/lcpt). An ESTIMATE: no token-compliance claim."""
    r = (params or {}).get("retrieval", {}) if params else {}
    lcpt = float(r.get("latin_chars_per_token", {}).get("value", 4))
    fac = float(r.get("persian_token_factor", {}).get("value", 1.66))
    fa = sum(1 for c in str(text) if "؀" <= c <= "ۿ")
    latin = len(str(text)) - fa
    return int(math.ceil(latin / lcpt + fa * fac / lcpt))


# ---------------------------------------------------------------------------------------------------------------
# dense channel plumbing
# ---------------------------------------------------------------------------------------------------------------
class Embedder:
    """Interface. `model_rev` identifies immutable weights AND configuration (tokenizer, pooling, prefix, truncation, dim)."""
    model_rev = "none"

    def embed(self, texts) -> list:                   # pragma: no cover - interface
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic fake for tests: hashed character trigrams of the NORMALISED text -> unit vector. Plumbing only; it makes no claim
    about bilingual retrieval quality."""

    def __init__(self, dim=64, rev="fake-hash-v1"):
        self.dim, self.model_rev, self.calls = dim, rev, 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            s = " ".join(lc.prep(t))
            v = [0.0] * self.dim
            for i in range(max(1, len(s) - 2)):
                h = int(hashlib.sha256(s[i:i + 3].encode("utf-8")).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class BgeM3Embedder(Embedder):
    """Real dense channel (G-dense candidate). LAZY: nothing heavy is imported until the first embed(). CPU only (CUDA hidden), threads and
    max_seq_length capped, loaded from the local Hugging Face cache at a PINNED revision. Meant for the isolated venv
    (sim/step5/.venv_bge); never the WSL environment, never the GPU the simulator needs. `model_rev` encodes the weights revision and the
    configuration so a cache entry is never reused across configurations."""
    REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

    def __init__(self, threads=4, max_len=128, cache_dir=None):
        self.threads, self.max_len, self.cache_dir = threads, max_len, cache_dir
        self.model_rev = f"BAAI/bge-m3@{self.REVISION[:12]}|cpu|maxlen{max_len}|cls|l2norm"
        self._model = None

    def _load(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""              # must precede the torch import
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from pathlib import Path as _P
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(self.threads)
        root = _P(self.cache_dir) if self.cache_dir else _P.home() / ".cache/huggingface/hub"
        m = SentenceTransformer(str(root / "models--BAAI--bge-m3" / "snapshots" / self.REVISION), device="cpu")
        m.max_seq_length = self.max_len
        self._model = m

    def embed(self, texts) -> list:
        if self._model is None:
            self._load()
        return [list(map(float, v)) for v in self._model.encode(list(texts), batch_size=16, normalize_embeddings=True)]


def _valid_vec(v, dim=None) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) > 0 and (dim is None or len(v) == dim)
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) and any(abs(x) > 0 for x in v))


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class EmbeddingCache:
    """(sha256(key text), model_rev) -> vector; optional sqlite persistence at an INJECTED path."""

    def __init__(self, embedder: Embedder, path=None):
        self.embedder, self.path = embedder, path
        self.mem: dict = {}
        self.dim = None
        self.hits = self.misses = self.rejected = 0
        self.index: dict = {}                          # record id -> (key hash, vec): what a record is CURRENTLY embedded as
        if path:
            c = sqlite3.connect(str(path))
            c.execute("CREATE TABLE IF NOT EXISTS vec(h TEXT, rev TEXT, dim INT, blob BLOB, PRIMARY KEY(h, rev))")
            for h, rev, dim, blob in c.execute("SELECT h, rev, dim, blob FROM vec"):
                self.mem[(h, rev)] = list(struct.unpack(f"<{dim}d", blob))
            c.commit()
            c.close()

    def _store(self, h, vec):
        self.mem[(h, self.embedder.model_rev)] = vec
        if self.path:
            c = sqlite3.connect(str(self.path))
            c.execute("INSERT OR REPLACE INTO vec VALUES(?,?,?,?)", (h, self.embedder.model_rev, len(vec), struct.pack(f"<{len(vec)}d", *vec)))
            c.commit()
            c.close()

    def vectors(self, texts) -> list:
        """Vectors for `texts` in order; missing ones are embedded in ONE batch. Raises ValueError on an invalid vector."""
        rev = self.embedder.model_rev
        hs = [sha(t) for t in texts]
        need = [(i, t) for i, (t, h) in enumerate(zip(texts, hs)) if (h, rev) not in self.mem]
        self.hits += len(texts) - len(need)
        self.misses += len(need)
        if need:
            got = self.embedder.embed([t for _, t in need])
            if len(got) != len(need):
                raise ValueError("embedder returned a different number of vectors")
            for (i, t), v in zip(need, got):
                v = [float(x) for x in v]
                if not _valid_vec(v, self.dim):
                    self.rejected += 1
                    raise ValueError("invalid embedding (non-finite, zero, or wrong dimension)")
                self.dim = len(v)
                self._store(hs[i], v)
        return [self.mem[(h, rev)] for h in hs]

    def publish(self, rid, key_hash, vec, current_key_hash) -> bool:
        """Atomic publish for one record: accepted only if `key_hash` is still the record's CURRENT key hash (late writes rejected)."""
        if key_hash != current_key_hash or not _valid_vec(vec, self.dim):
            self.rejected += 1
            return False
        self.index[rid] = (key_hash, list(vec))
        return True

    def sync(self, items: dict) -> dict:
        """items: record id -> key text. Re-embeds exactly the records whose key hash changed. Returns {id: vector}."""
        out, redo = {}, []
        for rid, text in sorted(items.items()):
            h = sha(text)
            cur = self.index.get(rid)
            if cur and cur[0] == h:
                out[rid] = cur[1]
            else:
                redo.append((rid, text, h))
        if redo:
            vecs = self.vectors([t for _, t, _ in redo])
            for (rid, text, h), v in zip(redo, vecs):
                if self.publish(rid, h, v, sha(items[rid])):
                    out[rid] = v
        return out


# ---------------------------------------------------------------------------------------------------------------
# record views
# ---------------------------------------------------------------------------------------------------------------
def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def intent_roles(rec, lex) -> list:
    ik = _json(rec.get("intent_key"))
    roles = ik.get("roles") if isinstance(ik, dict) else ik
    if isinstance(roles, list) and roles and all(isinstance(r, dict) for r in roles):
        return [{k: r.get(k, "-" if k != "neg" else 0) for k in lc.ROLE_FIELDS} for r in roles]
    if rec.get("intent_text"):
        return lc.parse(rec["intent_text"], lex)["roles"]          # fail-closed parse; may contain '?'
    return []


def failure_text(fk, lex) -> str:
    n = lambda t: lex.type_name(t, "en") if t in lex.data["types"] else str(t)
    nf = lambda t: lex.type_name(t, "fa") if t in lex.data["types"] else str(t)
    return (f"{fk.get('cause')} {fk.get('verb')} {n(fk.get('otype'))} {fk.get('rtype')} {fk.get('state')} / "
            f"{fk.get('cause')} {fk.get('verb')} {nf(fk.get('otype'))} {fk.get('rtype')} {fk.get('state')}")


def failure_tokens(fk) -> list:
    return [f"f.{k}={fk.get(k)}" for k in FAILURE_FIELDS] + ["f.sig=" + "/".join(str(fk.get(k)) for k in FAILURE_FIELDS)]


# ---------------------------------------------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------------------------------------------
def rrf_fuse(rankings: dict, rrf_k: int) -> list:
    """rankings: channel -> ordered list of ids (best first). -> [(id, rrf, {channel: rank})] sorted by (-quantised score, id)."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    score, ranks = {}, {}
    for ch in sorted(rankings):
        seen = set()
        for r, rid in enumerate(rankings[ch], start=1):
            if rid in seen:
                continue
            seen.add(rid)
            score[rid] = score.get(rid, 0.0) + 1.0 / (rrf_k + r)
            ranks.setdefault(rid, {})[ch] = r
    return sorted(((i, quant(s), ranks[i]) for i, s in score.items()), key=lambda x: (-x[1], str(x[0])))


# ---------------------------------------------------------------------------------------------------------------
# preconditions (injectable) and body consistency
# ---------------------------------------------------------------------------------------------------------------
def _atom(a):
    if isinstance(a, dict):
        return a.get("kind"), a.get("otype") or a.get("rtype"), a.get("state")
    s = str(a)
    if ":" in s:
        k, _, rest = s.partition(":")
        if "=" in rest:
            t, _, st = rest.partition("=")
            return k, t, st
        return k, rest, None
    return s, None, None


def default_pre_eval(atoms, snap) -> tuple:
    """-> ('hold'|'fail'|'unknown', failed-atom-text). Unknown atoms or missing snapshot keys are 'unknown' (fail closed)."""
    if snap is None:
        return ("unknown", "no_wm_snapshot")
    for a in atoms or []:
        k, t, st = _atom(a)
        if k == "hand_empty":
            if "holding" not in snap:
                return ("unknown", "hand_empty")
            if snap["holding"] not in (None, ""):
                return ("fail", "hand_empty")
        elif k == "holding":
            if "holding" not in snap:
                return ("unknown", f"holding:{t}")
            if snap["holding"] != t:
                return ("fail", f"holding:{t}")
        elif k == "located":
            if "located" not in snap:
                return ("unknown", f"located:{t}")
            if t not in snap["located"]:
                return ("fail", f"located:{t}")
        elif k == "container_state":
            cs = snap.get("container_state")
            if cs is None or t not in cs:
                return ("unknown", f"container_state:{t}")
            if cs[t] != st:
                return ("fail", f"container_state:{t}={st}")
        else:
            return ("unknown", f"atom:{k}")
    return ("hold", None)


def body_steps(rec) -> list:
    b = _json(rec.get("body"))
    st = b.get("steps") if isinstance(b, dict) else b
    return [s for s in st if isinstance(s, dict)] if isinstance(st, list) else []


def body_pre(rec) -> list:
    b = _json(rec.get("body"))
    if isinstance(b, dict) and b.get("pre") is not None:
        return list(b["pre"])
    st = body_steps(rec)
    return list(st[0].get("pre") or []) if st else []


#: A10 (C3, 2026-10-11): concept-role verbs are abstract ("pick", "put", "find"); plan steps carry TOOL names ("pickup",
#: "pick_and_place", "put_in", "search_for"). Before A10 `_step_matches` compared them verbatim, so a pick / put role could
#: never match its own body and the whole-intent gate (8.3) was closed for every such procedure. The mapping is fixed by the
#: tool semantics (honest_skills.TOOLS), not tuned on data.
ROLE_TOOLS = {"pick": ("pickup", "pick_and_place"), "put": ("pick_and_place", "put_in"), "find": ("search_for", "go_to")}


def step_matches_role(step, role) -> bool:
    v = str(step.get("verb", ""))
    rv = role["verb"]
    ok_verb = (v == rv or v in ROLE_TOOLS.get(rv, ())
               or (rv == "toggle" and v in (f"toggle_{role['req']}", f"toggle:{role['req']}", "toggle")))
    if not ok_verb:
        return False
    if v == "put_in" and rv == "put":                  # put_in's object is the RECEPTACLE
        return role.get("dest") not in (None, "-", "?") and step.get("otype") == role["dest"]
    if v == "pick_and_place" and rv in ("pick", "put") and role.get("dest") not in (None, "-", "?") and rv == "put":
        return step.get("otype") == role["obj"] and step.get("to") == role["dest"]
    return step.get("otype") == role["obj"]


_step_matches = step_matches_role


def body_consistent(rec, roles) -> bool:
    """Intent <-> body: every non-negated role appears as a step, in order; a negated role's verb+object never appears."""
    steps = body_steps(rec)
    if not steps or not roles:
        return False
    pos = 0
    for r in roles:
        if r["neg"]:
            if any(_step_matches(s, r) for s in steps):
                return False
            continue
        j = next((i for i in range(pos, len(steps)) if _step_matches(steps[i], r)), None)
        if j is None:
            return False
        pos = j + 1
    return True


def contradicted(rec) -> bool:
    """Explicit contradiction state: the `contradicted` flag (A5 sets it for an unresolved contradiction), or failed reuses that have
    caught up with successes, or a demoted/retired status."""
    if rec.get("contradicted"):
        return True
    fr = int(rec.get("failed_reuses") or 0)
    if fr > 0 and fr >= int(rec.get("success_uses") or 0):
        return True
    return rec.get("status") in ("demoted", "retired")


# ---------------------------------------------------------------------------------------------------------------
# application ledger (bounded, idempotent, monotone)
# ---------------------------------------------------------------------------------------------------------------
class ApplicationLedger:
    def __init__(self):
        self.by_failure: dict = {}                      # failure event id -> [lesson ids reserved]
        self.scored: set = set()                        # (episode, lesson) first eligible application per episode
        self.state: dict = {}                           # (retrieval id, record id) -> state

    def reserve(self, failure_event, lesson, episode) -> bool:
        cur = self.by_failure.setdefault(failure_event, [])
        if len(cur) >= MAX_APPS_PER_FAILURE and lesson not in cur:
            return False
        if lesson not in cur:
            cur.append(lesson)
        return True

    def first_scored(self, episode, lesson) -> bool:
        """True exactly once per (episode, lesson): the scored application."""
        k = (episode, lesson)
        if k in self.scored:
            return False
        self.scored.add(k)
        return True

    def mark(self, retrieval, record, state) -> str:
        if state not in STATE_ORDER:
            raise ValueError(f"unknown application state {state!r}")
        k = (retrieval, record)
        cur = self.state.get(k)
        if cur is None or STATE_ORDER[state] > STATE_ORDER[cur]:
            self.state[k] = state
        return self.state[k]


# ---------------------------------------------------------------------------------------------------------------
# the retriever
# ---------------------------------------------------------------------------------------------------------------
class Retriever:
    def __init__(self, params=None, lexicon=None, embedder: Embedder | None = None, cache: EmbeddingCache | None = None,
                 count_tokens=None, tokenizer_name=None, channels=None, sink=None):
        if params is None:
            try:
                from ltm import load_params
            except ImportError:                 # pragma: no cover
                from sim.ltm import load_params
            params = load_params()
        self.params = params
        self.lex = lexicon or lc.default_lexicon()
        self.embedder = embedder
        self.cache = cache or (EmbeddingCache(embedder) if embedder is not None else None)
        self._count = count_tokens
        self.tokenizer = tokenizer_name or ("injected" if count_tokens else "estimate")
        r = params["retrieval"]
        self.configured = list(channels if channels is not None else r["channels"]["value"])
        self.rrf_k = int(r["rrf_k"]["value"])
        self.top_k = int(r["top_k_records"]["value"])
        self.where_share = float(r["where_share"]["value"])
        self.budgets = dict(r["budget_tokens"]["value"])
        self.registered_budgets = dict(self.budgets)
        if r.get("retention") == "a12" and r.get("a12"):          # A12: the (possibly raised) token-derived budgets
            self.budgets.update({k: int(v) for k, v in r["a12"]["B"].items()})
        self.sink = sink if sink is not None else []
        self.ledger = ApplicationLedger()
        self.seq = 0
        self.logs: dict = {}

    # -- exact-key path -------------------------------------------------------------------------------------------
    def locate(self, ltm, otype, wm=None, e_now=None, prior=None) -> dict:
        """Locations / habits: SQL keyed lookups only (no embeddings, no lexical channel, no LLM)."""
        return lrt.where_is(ltm, otype, wm=wm, e_now=e_now, prior=prior)

    def habit_prefs(self, ltm, otype, e_now=None) -> dict:
        return ltm.habits(otype, e_now=e_now)

    # -- tokens ---------------------------------------------------------------------------------------------------
    def count_tokens(self, text) -> int:
        return int(self._count(text)) if self._count else estimate_tokens(text, self.params)

    @property
    def token_claim(self) -> bool:
        return self._count is not None

    # -- channels -------------------------------------------------------------------------------------------------
    def active_channels(self) -> tuple:
        act = [c for c in self.configured if c == "lexical"]
        reason = None
        if "dense" in self.configured:
            if self.embedder is None or self.cache is None:
                reason = "dense_unavailable"
            else:
                act.append("dense")
        if "lexical" not in act:
            act.insert(0, "lexical")
            reason = (reason + ",lexical_forced") if reason else "lexical_forced"
        return act, reason

    def _lexical(self, qc, qs, entries) -> list:
        n = len(entries)
        df: dict = {}
        for e in entries:
            for t in e["ctok"]:
                df[t] = df.get(t, 0) + 1
        qw = {t: lc.idf(df.get(t, 0), n) for t in qc}
        qtot = sum(qw.values())
        stot = float(len(qs))
        out = []
        for e in entries:
            cov = (sum(w for t, w in qw.items() if t in e["ctok"]) / qtot) if qtot else 0.0
            sc = (len(qs & e["stok"]) / stot) if stot else 0.0
            score = quant(cov + SURFACE_WEIGHT * sc)
            if score > 0:
                out.append({"id": e["id"], "score": score, "all_match": bool(qc) and set(qc) <= e["ctok"]})
        out.sort(key=lambda x: (-x["score"], str(x["id"])))
        return out

    def _dense(self, qtext, entries) -> list:
        vecs = self.cache.sync({e["id"]: e["text"] for e in entries})
        q = self.cache.vectors([qtext])[0]
        out = [{"id": rid, "score": quant(_cos(q, v))} for rid, v in vecs.items()]
        out = [x for x in out if x["score"] > 0]
        out.sort(key=lambda x: (-x["score"], str(x["id"])))
        return out

    def rank(self, qc, qs, qtext, entries) -> dict:
        """-> {"channels": {...}, "fused": [(id, rrf, ranks)], "scores": {id: {channel: score}}, "all_match": {id: bool}, "fallback": reason}"""
        act, fallback = self.active_channels()
        rankings, scores, allm = {}, {}, {}
        lex = self._lexical(qc, qs, entries)
        rankings["lexical"] = [x["id"] for x in lex]
        for x in lex:
            scores.setdefault(x["id"], {})["lexical"] = x["score"]
            allm[x["id"]] = x["all_match"]
        if "dense" in act:
            try:
                d = self._dense(qtext, entries)
                rankings["dense"] = [x["id"] for x in d]
                for x in d:
                    scores.setdefault(x["id"], {})["dense"] = x["score"]
            except Exception as ex:                       # degrade, and SAY so
                act = [c for c in act if c != "dense"]
                fallback = (fallback + "," if fallback else "") + "dense_error:" + type(ex).__name__
        return {"channels": {"configured": list(self.configured), "active": act}, "fused": rrf_fuse(rankings, self.rrf_k),
                "scores": scores, "all_match": allm, "fallback": fallback}

    # -- procedures -----------------------------------------------------------------------------------------------
    def _proc_entry(self, rec) -> dict:
        roles = intent_roles(rec, self.lex)
        ctok = set(lc.concept_tokens(roles)) if roles else set()
        text = lc.key_text(roles, self.lex) if roles else str(rec.get("intent_text") or "")
        return {"id": rec["proc"], "rec": rec, "roles": roles, "ctok": ctok, "text": text,
                "stok": set(lc.surface_tokens(text + " " + str(rec.get("intent_text") or "")))}

    def gate_procedure(self, e, q, kitchen, snap, pre_eval) -> dict:
        """Every conjunct is evaluated and recorded (`failed` in a fixed order); `first` is the first failure."""
        rec, failed = e["rec"], []
        if rec.get("kitchen") != kitchen:
            failed.append("scope")
        st = rec.get("status")
        status_ok = st in PROC_STATUS_STEER
        if not status_ok:
            failed.append("status")
        if contradicted(rec):
            failed.append("contradiction")
        roles_ok = bool(q["ok"]) and lc.roles_equal(q["roles"], e["roles"])
        if not roles_ok:
            failed.append("roles" if q["ok"] else "query_unresolved")
        body_ok = body_consistent(rec, e["roles"]) if e["roles"] and lc.roles_known(e["roles"]) else False
        if not body_ok:
            failed.append("body_intent")
        pres, pdetail = (pre_eval or default_pre_eval)(body_pre(rec), snap)
        if pres != "hold":
            failed.append("preconditions" if pres == "fail" else "preconditions_unknown")
        # a suggestion needs every conjunct except the status itself (and, for a demoted record, the contradiction its demotion implies)
        ignore = {"status"} | ({"contradiction"} if st == "demoted" else set())
        suggest = st in PROC_STATUS_SUGGEST and not [f for f in failed if f not in ignore]
        steer = not failed
        return {"id": e["id"], "steer": steer, "suggest": (not steer) and suggest, "failed": failed,
                "first": failed[0] if failed else None, "pre_detail": pdetail}

    def retrieve_procedures(self, query, records, *, kitchen, wm_snapshot=None, mode="test", episode=None, E=0, pre_eval=None,
                            top_k=None) -> dict:
        lex = self.lex
        q = query if isinstance(query, dict) else lc.parse(query, lex)
        qtext = lc.key_text(q["roles"], lex) if q["roles"] else str(query if isinstance(query, str) else "")
        qc = lc.concept_tokens(q["roles"]) if q["roles"] else []
        qs = set(lc.surface_tokens(qtext if isinstance(query, dict) else (query + " " + qtext)))
        entries = [self._proc_entry(r) for r in records]
        rk = self.rank(qc, qs, qtext, entries)
        by_id = {e["id"]: e for e in entries}
        decisions, steer, sugg, rejected = {}, [], [], []
        for rid, rrf, ranks in rk["fused"]:                          # the gate sees EVERY retrieved record (no top-K first)
            d = self.gate_procedure(by_id[rid], q, kitchen, wm_snapshot, pre_eval)
            decisions[rid] = d
            if d["steer"]:
                steer.append(rid)
            elif d["suggest"]:
                sugg.append(rid)
            else:
                rejected.append({"id": rid, "first": d["first"], "failed": d["failed"]})
        k = self.top_k if top_k is None else int(top_k)
        res = {"kind": "procedure", "query": q, "query_key_text": qtext, "steer": [by_id[i]["rec"] for i in steer[:k]],
               "suggestions": [by_id[i]["rec"] for i in sugg[:k]], "decisions": decisions, "rank": rk, "rejected": rejected,
               "n_records": len(records), "mode": mode}
        res["log"] = self._new_log("procedure", episode, E, q, rk, steer[:k], sugg[:k], rejected, decisions)
        return res

    # -- lessons --------------------------------------------------------------------------------------------------
    def _lesson_entry(self, rec) -> dict:
        fk = _json(rec.get("conditions")) or {}
        return {"id": rec["lesson"], "rec": rec, "fk": fk, "ctok": set(failure_tokens(fk)), "text": failure_text(fk, self.lex),
                "stok": set(lc.surface_tokens(failure_text(fk, self.lex)))}

    def retrieve_lessons(self, failure_key, records, *, kitchen, mode="test", roles=None, available=None, episode=None, E=0,
                         failure_event=None, step_i=None, attempt_k=None) -> dict:
        fk = {k: failure_key.get(k) for k in FAILURE_FIELDS}
        unknown = [k for k, v in fk.items() if v in (None, "", "?", "*")]
        entries = [self._lesson_entry(r) for r in records]
        rk = self.rank(failure_tokens(fk), set(), failure_text(fk, self.lex), entries) if not unknown else \
            {"channels": {"configured": list(self.configured), "active": self.active_channels()[0]}, "fused": [], "scores": {},
             "all_match": {}, "fallback": None}
        by_id = {e["id"]: e for e in entries}
        eligible, rejected, decisions = [], [], {}
        order_ids = [i for i, _, _ in rk["fused"]]
        for rid in order_ids:                                           # gate over every retrieved lesson
            e = by_id[rid]
            rec, failed = e["rec"], []
            if rec.get("kitchen") != kitchen:
                failed.append("scope")
            if any(e["fk"].get(k) != fk[k] for k in FAILURE_FIELDS):
                failed.append("failure_key")
            eff = rec.get("status")
            if eff == "supported" and rec.get("revalidate"):
                eff = "tentative"                                       # downgrade BEFORE any priority sorting
            if eff == "supported":
                pass
            elif eff == "tentative":
                if mode != "practice":
                    failed.append("tentative_outside_practice")
            else:
                failed.append("status")
            act = rec.get("action")
            if act in (None, "none"):
                failed.append("no_remedy")
            implied = ACTION_IMPLIES.get(act)
            if implied and roles and any(r.get("neg") and r.get("verb") == implied for r in roles):
                failed.append("forbidden_by_instruction")
            if available is not None and act not in available:
                failed.append("remedy_unavailable")
            decisions[rid] = {"id": rid, "failed": failed, "first": failed[0] if failed else None, "eff_status": eff}
            if failed:
                rejected.append({"id": rid, "first": failed[0], "failed": failed})
            else:
                eligible.append(rid)
        applied, reason = None, None
        if unknown:
            reason = "query_field_unknown:" + ",".join(unknown)
        elif eligible:
            ordered = sorted(eligible, key=lambda i: (0 if decisions[i]["eff_status"] == "supported" else 1,
                                                      int(by_id[i]["rec"].get("E_created") or 0), str(i)))
            actions = {by_id[i]["rec"]["action"] for i in ordered}
            if len(actions) > 1:
                reason = "conflicting_remedies"
            elif failure_event is not None and not self.ledger.reserve(failure_event, ordered[0], episode):
                reason = "application_bound"
            else:
                applied = ordered[0]
        res = {"kind": "lesson", "failure_key": fk, "eligible": eligible, "applied": by_id[applied]["rec"] if applied else None,
               "abstain_reason": reason, "decisions": decisions, "rejected": rejected, "rank": rk, "mode": mode,
               "related": [i for i in order_ids if i not in eligible][:5]}
        res["log"] = self._new_log("lesson", episode, E, {"roles": [], "ok": not unknown, "reasons": [reason] if reason else []}, rk,
                                   [applied] if applied else [], [], rejected, decisions, failure_key=fk, eligible=eligible)
        return res

    def recheck(self, item, record_now, kitchen, snap_now, query=None, pre_eval=None) -> bool:
        """Re-run the whole gate just before execution (WM or record version may have changed). Procedures only."""
        q = query if isinstance(query, dict) else lc.parse(query, self.lex)
        return bool(self.gate_procedure(self._proc_entry(record_now), q, kitchen, snap_now, pre_eval)["steer"])

    # -- logging --------------------------------------------------------------------------------------------------
    def _new_log(self, kind, episode, E, q, rk, steer, sugg, rejected, decisions, failure_key=None, eligible=None) -> dict:
        self.seq += 1
        rid = f"rt-{episode or 'x'}-{self.seq}"
        log = {"retrieval": rid, "kind": kind, "episode": episode, "E": int(E),
               "versions": {"retrieval": RETRIEVAL_VERSION, "parser": lc.PARSER_VERSION, "lexicon": self.lex.sha[:16],
                            "embedder": getattr(self.embedder, "model_rev", None), "tokenizer": self.tokenizer,
                            "rrf_k": self.rrf_k, "token_claim": self.token_claim},
               "channels": rk["channels"], "fallback": rk["fallback"],
               "query": {"roles": [lc.role_sig(r) for r in q.get("roles", [])], "ok": bool(q.get("ok")),
                         "reasons": list(q.get("reasons", []))[:8]},
               "retrieved": [{"id": i, "rrf": s, "rank": r, "score": rk["scores"].get(i, {})} for i, s, r in rk["fused"]],
               "eligible": list(eligible) if eligible is not None else list(steer), "suggested": list(sugg),
               "rejected": rejected, "rendered": [], "dropped": [], "applied": {}}
        if failure_key:
            log["query"]["failure"] = {k: str(v) for k, v in failure_key.items()}
        self.logs[rid] = log
        return log

    def mark_applied(self, retrieval, record, state) -> str:
        """selected < attempted < succeeded; idempotent; never downgrades. Returns the resulting state."""
        cur = self.ledger.mark(retrieval, record, state)
        self.logs[retrieval]["applied"][record] = cur
        return cur

    def log_row(self, log, cap=20) -> dict:
        """A `retrieval_log` whitelist row: ids, enums and numbers only (no free text)."""
        cands = [{"id": c["id"], "rrf": round(c["rrf"], 6), "ranks": c["rank"]} for c in log["retrieved"][:cap]]
        q = dict(log["query"])
        q["versions"] = {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in log["versions"].items() if v is not None}
        q["eligible"] = [str(i) for i in log["eligible"]][:cap]
        q["rejected"] = [{"id": r["id"], "first": r["first"]} for r in log["rejected"][:cap]]
        q["rendered"] = [str(i) for i in log["rendered"]][:cap]
        q["dropped"] = [{"id": d["id"], "reason": d["reason"]} for d in log["dropped"][:cap]]
        q["applied"] = [{"id": str(k), "state": v} for k, v in sorted(log["applied"].items())]
        q["fallback"] = (log["fallback"] or "none").replace(",", ";")[:60]
        q["channels_active"] = list(log["channels"]["active"])
        return {"retrieval": log["retrieval"], "episode": log["episode"], "query": q, "candidates": cands, "E": log["E"],
                "channel": "_".join(log["channels"]["active"])[:40]}

    def flush(self, log, writer=None):
        """Hand the finished log row to `writer(table, values)` (e.g. ltm.log_test) or the injected sink list."""
        row = self.log_row(log)
        if writer is not None:
            return writer("retrieval_log", row)
        self.sink.append(row)
        return row

    # -- rendering ------------------------------------------------------------------------------------------------
    def _place_name(self, c, lang) -> str:
        w = WORDS[lang]
        rt = c.get("rtype")
        nm = self.lex.type_name(rt, lang) if rt in self.lex.data["types"] else (rt or "")
        if c["kind"] == "rtype":
            return w["any"].format(n=nm or str(c["id"]))
        if c["kind"] == "xyz":
            return w["spot"].format(x=str(c["id"]).replace("xyz:", ""))
        return f"{nm} ({c['id']})" if nm else str(c["id"])

    def render_where(self, cand, i, lang) -> str:
        w = WORDS[lang]
        label = TIER_LABEL[lang][cand["tier"]].format(c=cand.get("contradictions", 0))
        q = [label]
        if cand.get("age") is not None:
            q.append(w["age"].format(a=cand["age"]))
        q.append(w["support"].format(s=cand.get("support", 0)))
        if cand.get("contradictions") and cand["tier"] != "T3b":
            q.append(w["contra"].format(c=cand["contradictions"]))
        if "deferred" in str(cand.get("status", "")):
            q.append("deferred")
        return f"{i}. {self._place_name(cand, lang)} [{'; '.join(q)}]"

    def render_suppressed(self, s, lang) -> str:
        w = WORDS[lang]
        if s["reason"] == "inspected_this_task":
            return w["neg_task"].format(p=s["place"])
        return w["neg"].format(p=s["place"], a=s["age"])

    def _steps_text(self, rec, lang) -> str:
        w, parts = WORDS[lang], []
        for s in body_steps(rec):
            v = self.lex.verb_name(str(s.get("verb")).split(":")[0].replace("toggle_", "toggle"), "-", lang) if s.get("verb") else "?"
            ob = s.get("otype")
            nm = self.lex.type_name(ob, lang) if ob in self.lex.data["types"] else str(ob or "")
            to = s.get("to")
            tn = self.lex.type_name(to, lang) if to in self.lex.data["types"] else str(to or "")
            parts.append(f"{v} {nm}" + (f" {w['into']} {tn}" if to and to != "-" else ""))
        return f" {w['then']} ".join(parts)

    def render_proc(self, rec, lang, suggestion=False) -> str:
        w = WORDS[lang]
        roles = intent_roles(rec, self.lex)
        k = lc.render_key(roles, self.lex, lang) if roles and lc.roles_known(roles) else "?"
        st = w["status"].get(rec.get("status"), str(rec.get("status")))
        if suggestion:
            return w["sugg"].format(k=k, st=st, steps=self._steps_text(rec, lang))
        return w["proc"].format(k=k, st=st, n=int(rec.get("success_uses") or 0), steps=self._steps_text(rec, lang))

    def render_lesson(self, rec, eff_status, lang) -> str:
        w = WORDS[lang]
        fk = _json(rec.get("conditions")) or {}
        ob = fk.get("otype")
        return w["lesson"].format(
            c=CAUSE_NAME[lang].get(fk.get("cause"), "?"), v=self.lex.verb_name(fk.get("verb"), "-", lang),
            o=self.lex.type_name(ob, lang) if ob in self.lex.data["types"] else str(ob), a=ACTION_NAME[lang].get(rec.get("action"), "?"),
            st=w["lstatus"].get(eff_status, str(eff_status)), h=int(rec.get("helpful_units") or 0), u=int(rec.get("unhelpful_units") or 0))

    def memory_block(self, where=None, procs=None, lessons=None, *, level="L2", lang="en", otype=None, render_suggestions=False,
                     budget_tokens=None) -> dict:
        """Budgeted, deterministic block. Counts the FINAL serialised text. Whole records only; a record's qualifiers are inside its line."""
        w = WORDS[lang]
        total = int(budget_tokens if budget_tokens is not None else self.budgets[level])
        cap = max(1, int(total * RECORD_CAP_FRAC))
        T = self.count_tokens
        dropped, kept, sections = [], [], []

        def fill(header, items, budget, section):
            """items: [(id, text)] in rank order. Prefix STOP at the first record that does not fit. Returns lines used."""
            lines = [header] if header else []
            used = T("\n".join(lines)) if lines else 0
            out = []
            stopped = False
            for rid, text in items:
                tk = T(text)
                if tk > cap:
                    dropped.append({"id": rid, "reason": "oversize", "tokens": tk})
                    continue
                if stopped or used + T("\n") + tk > budget:
                    stopped = True
                    dropped.append({"id": rid, "reason": "budget", "tokens": tk})
                    continue
                out.append((rid, text, section))
                used += tk + T("\n")
            return lines, out

        head = w["head"]
        remaining = total - T(head)
        blocks = []
        if remaining < 0:
            return {"text": "", "records": [], "dropped": [], "tokens": 0, "budget": total, "overflow": "head_exceeds_budget",
                    "token_claim": self.token_claim, "tokenizer": self.tokenizer}
        used_where = 0
        if where is not None:
            sub = w["where"].format(t=self.lex.type_name(otype or where.get("otype"), lang))
            explore = w["explore"]
            mand = T(sub) + T(explore) + 2 * T("\n")
            wb = int(total * self.where_share)
            items = [(c["id"], self.render_where(c, i, lang)) for i, c in enumerate(where["candidates"], 1)]
            supp = [(f"sup:{s['place']}", self.render_suppressed(s, lang)) for s in where["explore"].get("suppressed", [])]
            if mand > min(wb, remaining):
                dropped += [{"id": i, "reason": "mandatory_exceeds_budget", "tokens": 0} for i, _ in items + supp]
                blocks.append(("where", [sub], [], explore))
            else:
                _, got = fill(None, items + supp, min(wb, remaining) - mand, "where")
                blocks.append(("where", [sub], got, explore))
                used_where = T(sub) + T(explore) + sum(T(t) + T("\n") for _, t, _ in got)
        rest = remaining - used_where
        if procs is not None:
            pitems = [(r["proc"], self.render_proc(r, lang)) for r in procs["steer"]]
            if render_suggestions:
                pitems += [(r["proc"], self.render_proc(r, lang, True)) for r in procs["suggestions"]]
            pb = int(rest * (PROC_SHARE if lessons is not None else 1.0))
            ph = w["procs"]
            if pitems and T(ph) + T("\n") < pb:
                _, got = fill(None, pitems, pb - T(ph) - T("\n"), "procs")
                blocks.append(("procs", [ph], got, None))
                rest -= (T(ph) + T("\n") + sum(T(t) + T("\n") for _, t, _ in got)) if got else 0
            else:
                dropped += [{"id": i, "reason": "budget", "tokens": T(t)} for i, t in pitems]
        if lessons is not None and lessons.get("applied") is not None:
            r = lessons["applied"]
            eff = lessons["decisions"][r["lesson"]]["eff_status"]
            lh = w["lessons"]
            if T(lh) + T("\n") < rest:
                _, got = fill(None, [(r["lesson"], self.render_lesson(r, eff, lang))], rest - T(lh) - T("\n"), "lessons")
                blocks.append(("lessons", [lh], got, None))
            else:
                dropped.append({"id": r["lesson"], "reason": "budget", "tokens": 0})

        def serialise(bl):
            lines = [head]
            for name, hdr, recs, tail in bl:
                if not recs and name != "where":
                    continue
                lines += hdr + [t for _, t, _ in recs]
                if tail:
                    lines.append(tail)
            return "\n".join(lines)

        text = serialise(blocks)
        trimmed = False
        while T(text) > total:                                              # final check on the serialised block
            tail_i = next((i for i in range(len(blocks) - 1, -1, -1) if blocks[i][2]), None)
            if tail_i is None:
                break
            name, hdr, recs, tail = blocks[tail_i]
            rid, t, _ = recs[-1]
            dropped.append({"id": rid, "reason": "final_trim", "tokens": T(t)})
            blocks[tail_i] = (name, hdr, recs[:-1], tail)
            trimmed = True
            text = serialise(blocks)
        recs_out = [{"id": rid, "section": s, "text": t, "tokens": T(t)} for _, _, recs, _ in blocks for rid, t, s in recs]
        n = T(text)
        return {"text": text, "records": recs_out, "dropped": dropped, "tokens": n, "budget": total, "trimmed": trimmed,
                "overflow": ("over_budget" if n > total else None), "token_claim": self.token_claim, "tokenizer": self.tokenizer,
                "level": level, "lang": lang}

    def finish_block(self, block, *logs):
        """Write rendered/dropped ids into the retrieval log(s) of this block."""
        rendered = [r["id"] for r in block["records"]]
        for lg in logs:
            if lg is None:
                continue
            lg["rendered"] = [i for i in rendered if i in {x["id"] for x in lg["retrieved"]}]
            lg["dropped"] = [d for d in block["dropped"] if d["id"] in {x["id"] for x in lg["retrieved"]}]


# ---------------------------------------------------------------------------------------------------------------
# bilingual key / alias generation (no LLM)
# ---------------------------------------------------------------------------------------------------------------
def generate_aliases(otype, lex=None) -> dict:
    lex = lex or lc.default_lexicon()
    e = lex.data["types"][otype]
    en = sorted({" ".join(lc.prep(a)) for a in e["en"] + e.get("en_plural", [])})
    fa = sorted({" ".join(lc.prep(a)) for a in e["fa"]})
    return {"en": en, "fa": fa}


def load_records(store, kitchen) -> dict:
    """Procedure / lesson rows of one kitchen from an A1 store (read only)."""
    return {"procedures": store.select("procedure", where={"kitchen": kitchen}),
            "lessons": store.select("lesson", where={"kitchen": kitchen})}
