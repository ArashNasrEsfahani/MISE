"""M2: the experience block of the FS arms (prereg s6.4; SPEC s13 M2). Same source, equal token budget.

The block is built from the K = 10 practice attempts of order 0 and ALWAYS holds all ten attempts:
  Part 1  one compact line per attempt, oldest first: instruction (practice language) | plan as `action object>to` tokens |
          per-step outcome (`ok` or the cause code) | admission verdict (success / failure). Failures are kept.
  Part 2  sightings: one line per (object type -> receptacle type), times seen, attempts, newest attempt first, filled with
          COMPLETE lines until the level's token allowance is used up (a prefix; the first line that does not fit stops it).
If part 1 alone does not fit, plan tokens are shortened uniformly for all attempts (verb and type kept, `to` dropped) and that is
recorded (`shortened`). If it still does not fit, FSBlockInfeasible is raised: nothing is dropped silently.

The allowance and the counter are the memory block's: `budget` = the final A12 B_L; the count is the max over the tokenizers of
(EXPERIENCE_HEADER + block), exactly the text the planner sees. No outcome of the TEST phase ever enters; the source is practice
(training) data, so its outcomes are allowed. Test items are refused by C.guard_items.

    python sim/final_test/matrix/fs_block.py --write --budget L1=900,L2=1100,L3=1300 [--ollama URL] [--allow-missing-anthropic]
    python sim/final_test/matrix/fs_block.py --check
Python 3.10 syntax only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                    # noqa: E402  (also puts sim/ on sys.path)

LEVELS = ("L1", "L2", "L3")
FS_VERSION = "fs-v1"          # default and history; "fs-v2" (render_fs_v2) is switched on only by I-pkg / A24.1
FS_VERSIONS = ("fs-v1", "fs-v2")
K = 10
SIGHT_HEAD = "Sightings (object -> place, times seen, attempts, newest first):"
ATT_HEAD = "Attempts (oldest first; instruction | plan | steps | result):"


class FSBlockInfeasible(RuntimeError):
    pass


def experience_header() -> str:
    from planners.contract import EXPERIENCE_HEADER
    return EXPERIENCE_HEADER


# ------------------------------------------------------------------------------------------------ counting
def real_counts(counter, text: str) -> dict:
    """Per-tokenizer real count; a plain callable counter reports a single 'count' (the same rule as ltm._real_counts)."""
    if hasattr(counter, "counts"):
        return {k: int(v) for k, v in counter.counts(text).items()}
    return {"count": int(counter(text))}


def real_tokens(counter, header: str, text: str) -> tuple:
    c = real_counts(counter, header + "\n" + text) if text else {}
    return (max(c.values()) if c else 0), c


# ------------------------------------------------------------------------------------------------ rows
def _verdict(r: dict) -> str:
    """success / failure, or the explicit token `crash` / `inapplicable` for a STARTED attempt that never produced an admission
    (prereg 6.2 / 6.4: K counts every started attempt and the block always holds all ten; such an attempt is shown, not dropped)."""
    v = r.get("verdict")
    if isinstance(v, str) and v.lower() in ("success", "failure"):
        return v.lower()
    st = r.get("status")
    if st in ("crash", "inapplicable"):
        return st
    if st == "rejected":                              # a plan rejected after the sweep: a started attempt, admission False
        return "failure"
    a = r.get("admission", r.get("applied"))
    if isinstance(a, dict):
        a = a.get("admitted", a.get("ok"))
    if isinstance(a, bool):
        return "success" if a else "failure"
    g = r.get("goal_ok")
    if isinstance(g, bool):
        return "success" if g else "failure"
    raise ValueError("practice row has no verdict / admission / goal_ok")


def _plan_steps(r: dict) -> list:
    p = r.get("plan")
    if isinstance(p, dict):                           # c3-style: row["plan"]["plan"]
        p = p.get("plan")
    out = []
    for s in p or []:
        if isinstance(s, dict):
            out.append((str(s.get("action") or ""), s.get("object"), s.get("to")))
        else:
            s = list(s) + [None] * 4
            out.append((str(s[0] or ""), s[1], s[2]))
    return out


def _outcomes(r: dict, n_steps: int) -> list:
    steps = r.get("exec_steps", r.get("steps"))
    if not steps or r.get("plan_not_executed"):
        return ["not_run"] * n_steps
    out = []
    for s in steps:
        out.append("ok" if s.get("ok") else str(s.get("cause") or "failed"))
    return out


def _sights(r: dict) -> list:
    """Sightings of one attempt as (object type, place type). practice_k writes dicts {otype, in_rtype, state}; the older
    {type, receptacle|place} spelling and "a -> b" strings are read too. A row whose sightings carry NO recognised key is refused
    (a silent drop would give FS less experience than +M and bias H4 without any error)."""
    out = []
    for s in r.get("sightings") or []:
        if isinstance(s, dict):
            if not ({"otype", "type"} & set(s) and {"in_rtype", "receptacle", "place"} & set(s)):
                raise ValueError(f"practice attempt {r.get('attempt')}: a sighting carries no recognised keys {sorted(s)}")
            t = s.get("otype", s.get("type"))
            p = s.get("in_rtype", s.get("receptacle", s.get("place")))
        elif isinstance(s, str) and "->" in s:
            t, p = [x.strip() for x in s.split("->", 1)]
        else:
            t, p = s[0], s[1]
        if t and p:
            out.append((str(t), str(p)))
    return out


def _attempt_no(r: dict) -> int:
    return int(r.get("attempt", r.get("n", -1)))


def in_source(r: dict) -> bool:
    """THE one filter of the FS source: order 0, attempt number 1..K (practice_k numbers its attempts 1-based; the older 0..K-1
    spelling is still accepted by normalise_rows when it is complete)."""
    return r.get("order", 0) == 0 and 0 <= _attempt_no(r) <= K


def normalise_rows(rows) -> list:
    """Order-0, the K attempts exactly once each, oldest first. practice_k.practice_order numbers attempts 1..K (the ONE place that
    maps them to the internal 0-based index is here); a complete legacy 0..K-1 numbering is accepted too. Anything else is refused
    (no partial source)."""
    rows = [r for r in rows if in_source(r)]
    by = {}
    for r in rows:
        a = _attempt_no(r)
        if a in by:
            raise ValueError(f"duplicate practice attempt {a} in order 0")
        by[a] = r
    if sorted(by) == list(range(1, K + 1)):
        by = {a - 1: r for a, r in by.items()}
    if sorted(by) != list(range(K)):
        raise ValueError(f"FS source needs the K={K} order-0 attempts 1..{K}; got {sorted(by)}")
    out = []
    for a in range(K):
        r = by[a]
        text = r.get("instruction") or r.get("instruction_text")
        if not text:
            raise ValueError(f"practice attempt {a} carries no verbatim instruction")
        steps = _plan_steps(r)
        out.append({"attempt": a, "item": r.get("item"), "lang": r.get("lang", "en"), "instruction": str(text).strip(),
                    "steps": steps, "outcomes": _outcomes(r, len(steps)), "verdict": _verdict(r), "sights": _sights(r)})
    return out


def source_sha(norm: list) -> str:
    return C.sha256_json(norm)


# ------------------------------------------------------------------------------------------------ rendering
def _tok(s) -> str:
    return str(s).replace(" ", "_")


def _step(s: tuple, short: bool) -> str:
    act, obj, to = s
    t = act + (" " + _tok(obj) if obj else "")
    if to and not short:
        t += ">" + _tok(to)
    return t


def _attempt_line(a: dict, short: bool) -> str:
    plan = "; ".join(_step(s, short) for s in a["steps"]) or "(no plan)"
    steps = " ".join(a["outcomes"]) or "-"
    return f"{a['attempt'] + 1}. {a['lang']} | {a['instruction']} | {plan} | {steps} | {a['verdict']}"


def _sight_lines(norm: list) -> list:
    agg = {}
    for a in norm:
        for key in sorted(set(a["sights"])):
            agg.setdefault(key, []).append(a["attempt"] + 1)
    items = [(k, v) for k, v in agg.items()]
    items.sort(key=lambda kv: (-max(kv[1]), -len(kv[1]), kv[0]))          # newest attempt first, then most seen, then name
    return [f"{t} -> {p} x{len(v)} (attempts {','.join(map(str, v))})" for (t, p), v in items]


def _assemble(att_lines: list, sight_lines: list) -> str:
    parts = [ATT_HEAD] + att_lines
    if sight_lines:
        parts += [SIGHT_HEAD] + sight_lines
    return "\n".join(parts)


def render(practice_rows_o0_k10, level: str, counter, budget: int, *, namespace: str = "practice-v1", root=None,
           guard: bool = True) -> dict:
    """-> {text, real_tokens, counts, budget, level, retained:{attempts,outcomes,sightings,sightings_available}, shortened,
    shortening, source_sha, sha256}. real_tokens counts header + block (what the planner sees) under the counter."""
    if level not in LEVELS:
        raise ValueError(f"level {level!r}")
    norm = normalise_rows(practice_rows_o0_k10)
    if guard:
        C.guard_items([a["item"] for a in norm], namespace, root=root)
    header = experience_header()
    budget = int(budget)

    def fits(text):
        n, c = real_tokens(counter, header, text)
        return n <= budget, n, c

    short = False
    atts = [_attempt_line(a, False) for a in norm]
    ok, n, c = fits(_assemble(atts, []))
    if not ok:
        short = True
        atts = [_attempt_line(a, True) for a in norm]
        ok, n, c = fits(_assemble(atts, []))
        if not ok:
            raise FSBlockInfeasible(f"{level}: ten attempts need {n} tokens even without `to` (budget {budget})")
    sl = _sight_lines(norm)
    kept = 0
    best = (n, c)
    for i in range(1, len(sl) + 1):
        ok, n2, c2 = fits(_assemble(atts, sl[:i]))
        if not ok:
            break
        kept, best = i, (n2, c2)
    text = _assemble(atts, sl[:kept])
    out = {"level": level, "text": text, "real_tokens": best[0], "counts": best[1], "budget": budget,
           "retained": {"attempts": len(atts), "outcomes": sum(len(a["outcomes"]) for a in norm), "sightings": kept,
                        "sightings_available": len(sl)},
           "shortened": short, "shortening": "drop_to" if short else "none", "source_sha": source_sha(norm)}
    out["sha256"] = C.sha256_text(text)
    return out


# ================================================================================================ fs-v2 (PACKAGE-SPEC s10; D6)
# Additive. Nothing above this line changes: fs-v1 stays byte-identical and is still the default (FS_VERSION).
#
# fs-v2 = the SAME example retriever as the package (B5 ExampleIndex, practice corpus only), the SAME 2,000-token cap and counter
# discipline, RAW rendering, successes AND failures eligible, and nothing else (no briefing, profile, places, procedure or template
# content). It is per presentation and identical for G+FS and O+FS. The level label, item id, reference plan and goal are never inputs.
FS2_VERSION = "fs-v2"
FS2_CORPUS_SCHEMA = "final-test.fs_corpus/2"
FS2_BUDGET = 2000
#: a-priori ranking weights, the same as the package's (package_params.json "weights"); dense is 0 without an embedder (recorded).
FS2_WEIGHTS = {"exact": 0.6, "lexical": 0.25, "dense": 0.15}
FS2_IRRELEVANT_MIN_DENSE = 0.5
_FORBIDDEN_QUERY_KEYS = ("level", "item", "item_id", "reference_plan", "plan", "goal", "gold")


def _words(s: str) -> list:
    out, cur = [], []
    for ch in str(s).lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _bm25_norm(docs: list, qwords: list, k1: float = 1.5, b: float = 0.75) -> list:
    """Plain BM25 over the candidate set, normalised to [0,1] by the best score (0 everywhere when nothing matches)."""
    import math
    toks = [_words(d) for d in docs]
    n = len(toks)
    if not n or not qwords:
        return [0.0] * n
    avg = (sum(len(t) for t in toks) / n) or 1.0
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    raw = []
    for t in toks:
        sc = 0.0
        for w in set(qwords):
            f = t.count(w)
            if not f:
                continue
            idf = max(0.0, math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)))
            sc += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avg))
        raw.append(sc)
    m = max(raw)
    return [r / m if m > 0 else 0.0 for r in raw]


def _query_check(query: dict) -> None:
    bad = [k for k in _FORBIDDEN_QUERY_KEYS if k in query]
    if bad:
        raise ValueError(f"FS query carries forbidden keys {bad}: retrieval may use the instruction, language, holding and observed types only")


def rank_examples(cands: list, query: dict, *, embedder=None, weights=None) -> tuple:
    """-> (ranked, dropped). Same rule as the package's example component (PACKAGE-SPEC 9.3 steps 3-4, example component only):
    score = w_exact*match + w_lexical*bm25_norm(lex + text) + w_dense*cos; irrelevant = match == 0 and dense < 0.5; order = score desc,
    prio asc, id asc. Without an embedder dense = 0 (recorded by the caller). `ranked` = [(cand, score)]."""
    w = dict(FS2_WEIGHTS)
    w.update(weights or {})
    # Lexical scoring uses LANGUAGE-NEUTRAL tokens only (the query's type and operation keys against the record's keys and English
    # lex), never the instruction's surface words: an EN and an FA presentation of one command then rank, and so retain, the same
    # ids (the spec s10 early-check number that must be 0). Only when the parse gave no key at all does it fall back to the
    # instruction words, and that is reported (`lexical_source`).
    docs = []
    for c in cands:
        r = c.get("rec") or {}
        k = r.get("keys") or {}
        docs.append(" ".join(list(k.get("types") or []) + list(k.get("ops") or []) + [(r.get("lex") or {}).get("en", "")]))
    qw = _words(" ".join(list(query.get("types") or []) + list(query.get("ops") or [])))
    src_ = "keys"
    if not qw:
        qw, src_ = _words(query.get("instruction", "")), "instruction_fallback"
    rank_examples.last_lexical_source = src_
    lex = _bm25_norm(docs, qw)
    dense = [0.0] * len(cands)
    if embedder is not None and cands:
        dense = [float(x) for x in embedder(query, cands)]
    ranked, dropped = [], []
    for c, lx, dn in zip(cands, lex, dense):
        r = c.get("rec") or {}
        m = float(c.get("match", 0.0))
        if m <= 0 and dn < FS2_IRRELEVANT_MIN_DENSE:
            dropped.append({"id": r.get("id"), "reason": "irrelevant"})
            continue
        ranked.append((c, w["exact"] * m + w["lexical"] * lx + w["dense"] * dn))
    ranked.sort(key=lambda cs: (-round(cs[1], 12), (cs[0].get("rec") or {}).get("prio", 0), str((cs[0].get("rec") or {}).get("id"))))
    return ranked, dropped


def _verdict_of(rec: dict) -> str:
    v = rec.get("verdict")
    if not v and isinstance(rec.get("example"), dict):
        v = rec["example"].get("verdict")
    if not v and isinstance(rec.get("provenance"), dict):          # C4-open (Astra a4): ExampleIndex.to_rec keeps it in provenance
        v = rec["provenance"].get("verdict")
    return str(v or "unknown").lower()


def render_fs_v2(query: dict, index, counter, budget: int = FS2_BUDGET, *, headers=None, embedder=None, weights=None, ranker=None) -> dict:
    """The fs-v2 experience block of ONE presentation.

    -> {version, text, real_tokens, counts, budget, retained:[ids], retained_kinds:{verdict:n}, dropped:[{id,reason}], empty, lang,
        ranker:{weights, dense}, sha256}.
    `index.candidates(query, include_failures=True)` supplies pkg.cand/1 rows; `index.render(rec, lang, raw=True)` renders one whole
    line (the raw practice-language line). Whole lines only, taken as the longest prefix of the ranked list that fits the budget
    under EVERY tokenizer of the counter in BOTH languages (header + block, the text the planner sees); the first line that does
    not fit stops the block. Nothing qualifies -> text "" (no padding, no header); that is reported as `empty`, never hidden.
    `ranker` (C4-open, Astra a2): callable(cands, query) -> (ranked [(cand, score)], dropped) replacing rank_examples, so the locked
    runner can rank with the PACKAGE's own scorer (treatment.FSRenderer); None keeps rank_examples (fs-v2 as specified, unchanged)."""
    _query_check(query)
    budget = int(budget)
    hdr = dict({"en": experience_header(), "fa": experience_header()}, **(headers or {}))
    cands = list(index.candidates(query, include_failures=True))
    if ranker is None:
        ranked, dropped = rank_examples(cands, query, embedder=embedder, weights=weights)
        rk = {"weights": dict(FS2_WEIGHTS, **(weights or {})), "dense": embedder is not None,
              "lexical_source": getattr(rank_examples, "last_lexical_source", "keys")}
    else:
        ranked, dropped = ranker(cands, query)
        dropped = list(dropped)
        rk = {"ranker": getattr(ranker, "name", "injected"), "dense": embedder is not None, "lexical_source": "keys"}
    lines, kept, kinds = [], [], {}
    for pos, (c, _score) in enumerate(ranked):
        rec = c["rec"]
        line = str(index.render(rec, query.get("lang", "en"), raw=True)).strip()
        if not line or "\n" in line:
            dropped.append({"id": rec.get("id"), "reason": "not_a_whole_line"})
            continue
        v = _verdict_of(rec)
        if v != "unknown" and not line.lower().endswith(v):
            # Astra H-B8-fs: a clipped line must never make a failed attempt look successful. The raw line ends with its verdict.
            dropped.append({"id": rec.get("id"), "reason": "verdict_not_in_line"})
            continue
        it = rec.get("item") or (rec.get("provenance") or {}).get("item")
        if it and str(it) in line:
            raise ValueError("an FS line carries the item id (plumbing leak): " + str(rec.get("id")))
        trial = "\n".join(lines + [line])
        if any(real_tokens(counter, hdr[lg], trial)[0] > budget for lg in ("en", "fa")):
            dropped.append({"id": rec.get("id"), "reason": "budget"})
            dropped += [{"id": c2["rec"].get("id"), "reason": "budget_after_stop"} for c2, _s in ranked[pos + 1:]]
            break
        lines.append(line)
        kept.append(rec.get("id"))
        kinds[v] = kinds.get(v, 0) + 1
    text = "\n".join(lines)
    n, cnt = 0, {}
    if text:
        for lg in ("en", "fa"):
            n2, c2 = real_tokens(counter, hdr[lg], text)
            if n2 >= n:
                n, cnt = n2, c2
    out = {"version": FS2_VERSION, "text": text, "real_tokens": n, "counts": cnt, "budget": budget, "retained": kept,
           "retained_kinds": kinds, "dropped": dropped, "empty": not text, "lang": query.get("lang"), "candidates": len(cands),
           "ranked": len(ranked), "ranker": rk}
    out["sha256"] = C.sha256_text(text)
    return out


def render_fs(version: str, *args, **kw) -> dict:
    """Version switch. fs-v1 is the per-level block (render); fs-v2 is render_fs_v2. Unknown versions are refused."""
    if version == "fs-v1":
        return render(*args, **kw)
    if version == FS2_VERSION:
        return render_fs_v2(*args, **kw)
    raise ValueError(f"FS version {version!r} not in {FS_VERSIONS}")


def parity(query_a: dict, query_b: dict, index, counter, budget: int = FS2_BUDGET, **kw) -> dict:
    """EN and FA presentations of one command must retain the same ids (the early-check number that must be 0 differences)."""
    a = render_fs_v2(query_a, index, counter, budget, **kw)
    b = render_fs_v2(query_b, index, counter, budget, **kw)
    return {"same": a["retained"] == b["retained"], "a": a["retained"], "b": b["retained"]}


# ------------------------------------------------------------------------------------------------ fs_corpus.json (replaces fs_blocks.json)
def fs_corpus_doc(corpus: list, practice_ns: str, params: dict) -> dict:
    """{schema: final-test.fs_corpus/2, corpus_sha, practice_ns, params_sha, sha256}. Refused for any namespace but a practice one."""
    if not str(practice_ns).startswith("practice"):
        raise C.TestItemRefused("the fs-v2 corpus is built from practice-pool items only (a practice namespace)")
    doc = {"schema": FS2_CORPUS_SCHEMA, "corpus_sha": C.sha256_json(corpus), "practice_ns": practice_ns,
           "params_sha": C.sha256_json(params)}
    doc["sha256"] = C.self_sha(doc)
    return doc


def check_corpus_file(path: str, corpus=None, params=None) -> list:
    """Offline integrity of fs_corpus.json: shape, self-sha, and (when given) that it still matches the corpus / params."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"unreadable: {type(e).__name__}"]
    p = []
    if d.get("schema") != FS2_CORPUS_SCHEMA:
        p.append("wrong schema")
    if d.get("sha256") != C.self_sha(d):
        p.append("self-sha mismatch")
    if corpus is not None and d.get("corpus_sha") != C.sha256_json(corpus):
        p.append("corpus changed")
    if params is not None and d.get("params_sha") != C.sha256_json(params):
        p.append("params changed")
    return p


# ------------------------------------------------------------------------------------------------ early check (spec s10)
def early_check(queries: list, index, counter, budget: int = FS2_BUDGET, package_tokens=None, **kw) -> dict:
    """Diagnostic only. `queries` = presentations on NON-TEST items ({"instruction","lang","holding","observed","pair"}). Per language:
    blocks, empty rate, mean/max FS tokens against the budget, mean package tokens when given ({(instruction, lang): tokens}), the
    share of blocks holding >= 1 failure example; and the number of EN/FA pairs (same `pair` key) whose retained ids differ."""
    per, byp = {}, {}
    for q in queries:
        r = render_fs_v2(q, index, counter, budget, **kw)
        lg = q.get("lang", "en")
        s = per.setdefault(lg, {"n": 0, "empty": 0, "tokens": [], "with_failure": 0, "pkg": []})
        s["n"] += 1
        s["empty"] += int(r["empty"])
        s["tokens"].append(r["real_tokens"])
        s["with_failure"] += int(any(k in ("failure", "crash") for k in r["retained_kinds"]))
        if package_tokens and (q.get("instruction"), lg) in package_tokens:
            s["pkg"].append(package_tokens[(q["instruction"], lg)])
        if q.get("pair"):
            byp.setdefault(q["pair"], {})[lg] = r["retained"]
    out = {"budget": budget, "languages": {}}
    for lg, s in per.items():
        t = s["tokens"]
        out["languages"][lg] = {"n": s["n"], "empty_rate": round(s["empty"] / s["n"], 3), "fs_tokens_mean": round(sum(t) / len(t), 1),
                                "fs_tokens_p50": sorted(t)[len(t) // 2], "fs_tokens_max": max(t),
                                "utilisation_mean": round(sum(t) / len(t) / budget, 3), "share_with_failure": round(s["with_failure"] / s["n"], 3),
                                "package_tokens_mean": (round(sum(s["pkg"]) / len(s["pkg"]), 1) if s["pkg"] else None)}
    out["en_fa_pairs"] = sum(1 for v in byp.values() if len(v) == 2)
    out["en_fa_id_mismatch"] = sum(1 for v in byp.values() if len(v) == 2 and v.get("en") != v.get("fa"))
    return out


# ------------------------------------------------------------------------------------------------ files
def build(rows, budgets: dict, counter, *, namespace: str = "practice-v1", root=None, guard: bool = True) -> dict:
    blocks = {lv: render(rows, lv, counter, budgets[lv], namespace=namespace, root=root, guard=guard) for lv in LEVELS}
    srcs = {b["source_sha"] for b in blocks.values()}
    assert len(srcs) == 1
    doc = {"schema": "final-test.fs_blocks/1", "spec": C.SPEC_VERSION, **blocks, "source_sha": srcs.pop(),
           "counter": {"tokenizers": list(getattr(counter, "tokenizers", ["count"])), "header": experience_header(),
                       "budgets": {lv: int(budgets[lv]) for lv in LEVELS}}}
    doc["sha256"] = C.self_sha(doc)
    return doc


def check_file(path: str) -> list:
    """Offline integrity of an fs_blocks.json: self-sha, text sha, stored counts within budget. Returns problems."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"unreadable: {type(e).__name__}"]
    p = []
    if d.get("sha256") != C.self_sha(d):
        p.append("self-sha mismatch")
    for lv in LEVELS:
        b = d.get(lv) or {}
        if C.sha256_text(b.get("text", "")) != b.get("sha256"):
            p.append(f"{lv}: text sha mismatch")
        if not b.get("real_tokens", 10 ** 9) <= b.get("budget", -1):
            p.append(f"{lv}: tokens over budget")
        if (b.get("retained") or {}).get("attempts") != K:
            p.append(f"{lv}: not all ten attempts retained")
        if C.mask_problems(b.get("text", "")) or C.key_leak(b.get("text", "")):
            p.append(f"{lv}: masking / key leak")
    return p


def _parse_budget(s: str) -> dict:
    if os.path.exists(s):
        d = json.load(open(s, encoding="utf-8"))
        return {lv: int((d.get("B") or d)[lv]) for lv in LEVELS}
    out = dict(kv.split("=") for kv in s.split(","))
    return {lv: int(out[lv]) for lv in LEVELS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--budget", help="L1=n,L2=n,L3=n or a json with {B:{L1..L3}} (the final A12 allocation)")
    ap.add_argument("--namespace", default="practice-v1")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--allow-missing-anthropic", action="store_true")
    a = ap.parse_args(argv)
    P = C.paths()
    if a.check:
        pr = check_file(P["fs_blocks"])
        print("OK" if not pr else "PROBLEMS: " + "; ".join(pr))
        return 0 if not pr else 1
    if not a.write:
        ap.error("--write or --check")
    if a.allow_test or a.namespace != "practice-v1":
        raise C.TestItemRefused("FS blocks are built from practice-pool items only (namespace practice-v1)")
    if not a.budget:
        ap.error("--budget is required: the final A12 B_L per level")
    rows = C.Journal(C.journal_path(a.namespace, "practice")).rows()
    rows = [r for r in rows if r.get("order") == 0 and in_source(r)]
    sys.path.insert(0, os.path.join(C.SIM, "final_test", "c3"))
    from real_counter import RealMaxCounter
    ctr = RealMaxCounter(os.path.join(C.paths()["matrix_work"], "tokcache.fs_block.json"), experience_header(), a.ollama,
                         allow_missing_anthropic=a.allow_missing_anthropic)
    doc = build(rows, _parse_budget(a.budget), ctr)
    ctr.save()
    if a.dry_run:
        print(json.dumps({lv: dict(doc[lv]["retained"], tokens=doc[lv]["real_tokens"]) for lv in LEVELS}))
        return 0
    os.makedirs(os.path.dirname(P["fs_blocks"]), exist_ok=True)
    if os.path.exists(P["fs_blocks"]):
        old = json.load(open(P["fs_blocks"], encoding="utf-8"))
        if old.get("sha256") != doc["sha256"] and not os.environ.get("FS_BLOCK_OVERWRITE"):
            raise SystemExit("fs_blocks.json exists and differs; set FS_BLOCK_OVERWRITE=1 only before the freeze")
    with open(P["fs_blocks"], "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("wrote", P["fs_blocks"], doc["sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
