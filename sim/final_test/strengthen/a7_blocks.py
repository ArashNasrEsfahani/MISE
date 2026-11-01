"""A7: render the two fixed-procedure arms for the three kitchens -> docs/final-test/strengthen/A/a7/blocks.jsonl (+ fixed_texts.json).

    python sim/final_test/strengthen/gpu_lock.py run A -- python sim/final_test/strengthen/a7_blocks.py      (Gemma token counts need Ollama)

Gates (A2's, through ablate_blocks.Builder, cache-only counts): every captured S* / none request re-renders to its captured sha, every recorded
S* block parses and renders back byte-identically, and its recorded token counts equal the cached counts. A7 gate: every procedure record line
of every recorded block equals the store's line for that id (so the fixed text uses exactly the lines the package builder writes).

Token counts: Gemma only (the planner's tokenizer, Ollama prompt_eval_count over a one-token baseline, header included), because the Anthropic
count_tokens endpoint refused on 2026-11-01 (account credit balance too low). Recorded in every row ("tokenizers": ["gemma"]).
Checks on every block, as Builder.row: Persian character, D2-excluded type, masking gate on the whole prompt, package <= 2000 (Gemma);
fixed_proc_300: the block (rephrasing + fixed part) <= the longest recorded procedure section of the kitchen's full-memory blocks (Gemma).
Written atomically once all three kitchens are done; a stop before that leaves no file and the re-run redoes it (about two minutes).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a7_common as A7             # noqa: E402
import ablate_blocks as AB          # noqa: E402
import ablate_common as A          # noqa: E402

import common as C                 # noqa: E402
import plan as PL                  # noqa: E402
import treatment as T              # noqa: E402
from planners import prompt_sha256  # noqa: E402
from planners.contract import PACKAGE_HEADER  # noqa: E402

TOKCACHE = os.path.join(A7.WORK, "tokcache_gemma.json")


class GemmaCounter:
    """RealMaxCounter restricted to the Gemma tokenizer (private cache work/A7/tokcache_gemma.json, seeded with the Gemma counts of A's cache)."""

    def __init__(self):
        from real_counter import RealMaxCounter
        if not os.path.exists(TOKCACHE):
            seed = {}
            if os.path.exists(AB.TOKCACHE):
                with open(AB.TOKCACHE, encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        if "gemma_raw" in v:
                            seed[k] = {"gemma_raw": v["gemma_raw"], "chars": v.get("chars")}
            A.atomic_write_json(TOKCACHE, seed)
        self.rc = RealMaxCounter(TOKCACHE, PACKAGE_HEADER, allow_missing_anthropic=True)
        self.rc._key = None                        # Gemma only (the Anthropic endpoint refuses: credit balance)
        self.rc.tokenizers = ["gemma"]
        self.n = 0

    def gemma(self, text: str) -> int:
        """Gemma tokens of the block with the package header (Builder.counts_of convention); 0 for an empty block."""
        if not text:
            return 0
        v = int(self.rc.counts(PACKAGE_HEADER + "\n" + text)["gemma"])
        self.n += 1
        if self.n % 50 == 0:
            self.rc.save()
        return v

    def save(self):
        self.rc.save()


def row(b: AB.Builder, g: GemmaCounter, arm: str, p: str, recs: list, *, limit: int | None, extra: dict) -> dict:
    """One block row in A2's schema (Builder.row), Gemma-only counts, with the A7 length rule instead of the 300 quota."""
    r = b.cap[p]
    text = A.render_block({"records": recs, "bundle": []})
    msgs = b.contract.messages(r["instruction"], PL.summary_of(r), r.get("holding"), memory_block=text or None)
    sha = prompt_sha256(msgs)
    n = g.gemma(text)
    probs = []
    if n > int(b.cfg["package_max_tokens"]):
        probs.append(f"package {n} > {b.cfg['package_max_tokens']} (gemma)")
    if limit is not None and n > limit:
        probs.append(f"block {n} > the kitchen's longest recorded procedure section {limit} (gemma)")
    if any("؀" <= ch <= "ۿ" for ch in text):
        probs.append("Persian character in the block")
    if T.d2_on(b.cfg) and T._ts().mentions_excluded(text):
        probs.append("excluded type in the block")
    mk = C.mask_problems("\n".join(m["content"] for m in msgs))
    if mk:
        probs.append(f"masking gate: {mk}")
    same = "S*" if sha == r["requests"]["S*"] else ("none" if sha == r["requests"]["none"] else None)
    full_ids = list(r["blocks"]["S*"]["ids"])
    ids = [x["id"] for x in recs]
    out = {"ns": b.ns, "scene": A.KITCHENS[b.ns], "arm": arm, "kind": A.ARMS[arm]["kind"], "pres": p, "item": r["item"], "lang": r["lang"],
           "level": r["level"], "text": text, "text_sha": A.sha256_text(text) if text else None, "empty": not text, "ids": ids,
           "removed": [i for i in full_ids if i not in ids], "added": [i for i in ids if i not in full_ids],
           "counts": {"gemma": n} if text else {}, "real_max": n, "retrieved_max": n, "tokenizers": ["gemma"],
           "full_counts": r["blocks"]["S*"].get("counts") or {},
           "checks": {"ok": not probs, "problems": probs, "retrieved_cap_exempt": arm == "fixed_proc", "length_limit_gemma": limit},
           "request_sha": sha, "same_request_as": same, "planned": same is None, "a_phase": A7.PHASE}
    out.update(extra)
    return out


def kitchen(ns: str, cnt: AB.Counter, g: GemmaCounter) -> tuple:
    b = AB.Builder(cnt, ns)
    gates = b.gates()
    store = A7.store_records(ns, cnt)
    by_id = {x["id"]: x["line"] for x in store}
    mism, ctx = [], []
    for p, pb in b.parsed.items():
        for x in pb["records"]:
            if x["comp"] == "procedure" and x["id"] in by_id and by_id[x["id"]] != x["line"]:
                # the builder's per-instruction note ("... Needed here: a Microwave must be closed when toggle_on runs."): the fixed text
                # keeps the store's plain line, as a fixed prompt cannot know the instruction
                (ctx if x["line"].startswith(by_id[x["id"]] + " Needed here: ") else mism).append((p, x["id"]))
    if mism:
        raise SystemExit(f"GATE store lines: {len(mism)} recorded procedure lines differ from the store (first {mism[0]})")
    gates["store_lines_checked"] = sum(1 for pb in b.parsed.values() for x in pb["records"] if x["id"] in by_id)
    gates["instruction_specific_notes_not_in_fixed_text"] = sorted(ctx)
    # ---- arm 1: the whole store
    fixed_all = list(store)
    # ---- arm 2: length-matched, ranked by how many recorded full blocks of this kitchen contain the record
    freq = {x["id"]: 0 for x in store}
    sections = []
    for p, pb in b.parsed.items():
        procs = [x for x in pb["records"] if x["comp"] == "procedure" and not x["id"].endswith(A.REQUEST_SUFFIX)]
        for x in procs:
            if x["id"] in freq:
                freq[x["id"]] += 1
        if procs:
            sections.append(procs)
    # the length limit: the longest recorded procedure section (rephrasing included, as the arm's block includes it)
    limit = 0
    for p, pb in b.parsed.items():
        sec = [x for x in pb["records"] if x["comp"] == "procedure"]
        if sec:
            limit = max(limit, g.gemma(A.render_block({"records": sec, "bundle": []})))
    ranked = sorted(store, key=lambda x: (-freq[x["id"]], x["id"]))
    reqs = {p: [x for x in pb["records"] if x["id"].endswith(A.REQUEST_SUFFIX)] for p, pb in b.parsed.items()}
    longest = max(reqs, key=lambda p: (g.gemma(A.render_block({"records": reqs[p], "bundle": []})) if reqs[p] else 0, p))
    fixed_m = []
    for x in ranked:
        if freq[x["id"]] == 0:
            break                                  # never retrieved in this kitchen: not part of the most-often-useful prefix
        trial = fixed_m + [x]
        if g.gemma(A.render_block({"records": reqs[longest] + trial, "bundle": []})) <= limit:
            fixed_m = trial
        else:
            break                                  # a prefix of the ranking (no skipping ahead to shorter records)
    order = {x["id"]: i for i, x in enumerate(store)}
    fixed_m = sorted(fixed_m, key=lambda x: order[x["id"]])          # rendered in the store's own order, like arm 1
    rows = []
    fixed_doc = {}
    for arm, fixed, lim in (("fixed_proc", fixed_all, None), ("fixed_proc_300", fixed_m, limit)):
        ftext = A.render_block({"records": fixed, "bundle": []})
        fixed_doc[arm] = {"ids": [x["id"] for x in fixed], "text": ftext, "text_sha": A.sha256_text(ftext), "gemma_tokens": g.gemma(ftext),
                          "n_records": len(fixed)}
        for p in sorted(b.cap):
            rows.append(row(b, g, arm, p, reqs[p] + fixed, limit=lim,
                            extra={"fixed_sha": fixed_doc[arm]["text_sha"], "has_rephrasing": bool(reqs[p])}))
        g.save()
    fixed_doc["fixed_proc_300"].update(limit_gemma=limit, longest_rephrasing_pres=longest,
                                       ranking=[{"id": x["id"], "blocks_containing": freq[x["id"]]} for x in ranked])
    fixed_doc["store"] = {"n_records": len(store), "conditional_left_out": "affordance lines ending in '-request'",
                          "sha": A.sha256_text(json.dumps(store, sort_keys=True))}
    return rows, gates, fixed_doc


def main() -> int:
    if not AB.lock_held_by_a():
        print("refused: token counting uses Ollama; run under `gpu_lock.py run A -- ...`", file=sys.stderr)
        return 2
    if os.path.exists(A7.BLOCKS):
        print("a7/blocks.jsonl exists: nothing to do (A7 never rewrites a finished result)")
        return 0
    if A7.stop_requested():
        return 77
    cnt = AB.Counter(live=False)                   # gates: cache-only counts of the RECORDED blocks (both tokenizers, as recorded)
    g = GemmaCounter()
    rows, gates, fixed = [], {}, {}
    try:
        for ns in A7.NAMESPACES:
            rs, gates[ns], fixed[ns] = kitchen(ns, cnt, g)
            rows += rs
            for arm in A7.NEW_ARMS:
                mine = [r for r in rs if r["arm"] == arm]
                print(f"{ns} {arm}: fixed {fixed[ns][arm]['n_records']} records / {fixed[ns][arm]['gemma_tokens']} gemma tokens; "
                      f"changed requests {sum(1 for r in mine if r['same_request_as'] is None)}, "
                      f"failed checks {sum(1 for r in mine if not r['checks']['ok'])}", flush=True)
    finally:
        g.save()
        AB.unload_gemma()
    shas = {ns: fixed[ns]["store"]["sha"] for ns in fixed}
    bad = [r for r in rows if not r["checks"]["ok"]]
    if bad:
        print(f"{len(bad)} block(s) failed their checks (first {bad[0]['ns']} {bad[0]['arm']} {bad[0]['pres']}: {bad[0]['checks']['problems']}); "
              "nothing written", file=sys.stderr)
        return 1
    os.makedirs(A7.DOCS, exist_ok=True)
    A.atomic_write_json(A7.FIXED, {"kitchens": fixed, "gates": gates, "store_identical_across_kitchens": len(set(shas.values())) == 1,
                                   "tokenizers": ["gemma"], "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    A.atomic_write_jsonl(A7.BLOCKS, rows)
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
