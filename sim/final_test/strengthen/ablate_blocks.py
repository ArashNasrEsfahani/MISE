"""A2: render the ablated Scene Memory blocks offline for the 250 development-kitchen test presentations (no simulator, no planner call).

    python sim/final_test/strengthen/gpu_lock.py run A -- python sim/final_test/strengthen/ablate_blocks.py [--ns test-v1 ...]

Inputs (read only): docs/final-test/matrix/test-v1/capture.jsonl (instruction, prompt summary, holding, the recorded S* block and the
captured request shas), the frozen ft-v2 stores of FloorPlan15 / FloorPlan17 (sim/final_test/snapshots_ft2/...), the frozen type list.

Output: docs/final-test/strengthen/A/blocks.jsonl (one row per arm x presentation, rewritten atomically after every arm) and
blocks_summary.json. Token counts: the matrix's own RealMaxCounter (Gemma through Ollama, Anthropic count_tokens, free), header included,
max over tokenizers, with a private cache seeded from the three test namespaces' caches (sim/final_test/strengthen/work/A/tokcache.json).
A count that is not cached needs Ollama, so the script refuses to count live unless the strengthen GPU lock is held by task A.

Gates before anything is written: (1) every captured S* / none request re-renders to its captured sha with PlannerContract (plan.py's
request-identity gate); (2) every recorded S* block parses into one line per record and renders back byte-identically; (3) the recorded
token counts of every S* block equal the counter's counts.
Checks on every rendered block, as treatment.Packager.render does for the original: real max (header included) <= package_max_tokens,
retrieved part <= the profile cap (300), no Persian character, no D2-excluded type, and the masking gate on the whole prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate_common as A          # noqa: E402

import common as C                 # noqa: E402
import plan as PL                  # noqa: E402
import treatment as T              # noqa: E402
from planners import prompt_sha256  # noqa: E402
from planners.contract import PACKAGE_HEADER  # noqa: E402

OUT = os.path.join(A.DOCS, "blocks.jsonl")
SUMMARY = os.path.join(A.DOCS, "blocks_summary.json")
TOKCACHE = os.path.join(A.WORK, "tokcache.json")
SEED_CACHES = [os.path.join(A.ROOT, "sim", "final_test", "matrix", "work", n, "tokcache.json") for n in ("test-v1", "test-k2-v1", "test-k3-v1")]
TYPE_RX = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z]+)(?![A-Za-z])")
RESUME_CMD = "python sim/final_test/strengthen/gpu_lock.py run A -- python sim/final_test/strengthen/ablate_blocks.py --ns {ns}"


class Counter:
    """RealMaxCounter on the private cache; saves the cache atomically every 40 live counts. live=False -> a miss is an error."""

    def __init__(self, live: bool):
        from real_counter import RealMaxCounter
        if not os.path.exists(TOKCACHE):
            merged = {}
            for p in SEED_CACHES:
                with open(p, encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        merged.setdefault(k, {}).update(v)
            A.atomic_write_json(TOKCACHE, merged)
        self.rc = RealMaxCounter(TOKCACHE, PACKAGE_HEADER)
        self.header = PACKAGE_HEADER
        self.tokenizers = self.rc.tokenizers
        if self.tokenizers != ["gemma", "anthropic"]:
            raise RuntimeError("the counter needs both tokenizers (THESIS_ANTHROPIC_API_KEY missing?)")
        self.live, self._n = live, 0
        if not live:
            def miss(txt):
                raise RuntimeError("token count not cached and live counting is off (run under the GPU lock): " + txt[:60])
            self.rc._gemma_raw = miss
            self.rc._anth_raw = miss

    def counts(self, text: str) -> dict:
        before = sum(self.rc.n_calls.values())
        out = self.rc.counts(text)
        if sum(self.rc.n_calls.values()) != before:
            self._n += 1
            if self._n % 40 == 0:
                self.rc.save()
        return out

    def __call__(self, text):
        return self.rc(text)

    def save(self):
        self.rc.save()

    @property
    def live_calls(self) -> dict:
        return dict(self.rc.n_calls)


def lock_held_by_a() -> bool:
    p = os.path.join(A.ROOT, "docs", "final-test", "strengthen", "gpu.lock", "owner.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("task") == "A"
    except (OSError, ValueError):
        return False


def types_in(text: str, vocab: set) -> set:
    return {m for m in TYPE_RX.findall(text or "") if m in vocab}


class Builder:
    def __init__(self, counter: Counter, ns: str):
        self.cnt, self.ns = counter, ns
        self.cap = A.load_capture(ns)
        self.fam = A.family_map()
        self.types, self.types_src = A.load_types(ns)
        self.vocab = set(self.types)
        self.contract = PL.make_contract(self.types, ns)
        self.cfg = T.config(ns)
        self.rows = []
        self.parsed = {}

    # ---------------------------------------------------------------- gates
    def gates(self) -> dict:
        n_req = 0
        for p, r in self.cap.items():
            for s in ("S*", "none"):
                if prompt_sha256(PL.render_messages(self.contract, r, s)) != r["requests"][s]:
                    raise SystemExit(f"GATE request identity: {p} / {s} does not re-render to its captured sha")
                n_req += 1
            b = r["blocks"]["S*"]
            self.parsed[p] = A.parse_block(b["text"], b["ids"])          # raises on any grammar mismatch
            if b["text"]:
                got = self.cnt.counts(PACKAGE_HEADER + "\n" + b["text"])
                if {k: int(v) for k, v in got.items()} != {k: int(v) for k, v in (b.get("counts") or {}).items()}:
                    raise SystemExit(f"GATE token counts: {p} recorded {b.get('counts')} vs counter {got}")
        return {"requests_reproduced": n_req, "blocks_parsed": len(self.parsed),
                "nonempty_blocks": sum(1 for r in self.cap.values() if r["blocks"]["S*"]["text"]), "types_source": self.types_src}

    # ---------------------------------------------------------------- one row
    def counts_of(self, text: str) -> dict:
        return {k: int(v) for k, v in self.cnt.counts(PACKAGE_HEADER + "\n" + text).items()} if text else {}

    def row(self, arm: str, p: str, text: str, ids: list, *, removed=(), extra=None, retrieved_cap_exempt=False) -> dict:
        r = self.cap[p]
        msgs = self.contract.messages(r["instruction"], PL.summary_of(r), r.get("holding"), memory_block=text or None)
        sha = prompt_sha256(msgs)
        cts = self.counts_of(text)
        real = max(cts.values()) if cts else 0
        body = text
        i = text.find(A.BUNDLE_START)
        if i >= 0:
            body = text[:i].rstrip("\n")
        retrieved = (max(self.counts_of(body).values()) if body else 0) if i >= 0 else real
        probs = []
        if real > int(self.cfg["package_max_tokens"]):
            probs.append(f"package {real} > {self.cfg['package_max_tokens']}")
        if retrieved > 300 and not retrieved_cap_exempt:
            probs.append(f"retrieved part {retrieved} > 300")
        if any("؀" <= ch <= "ۿ" for ch in text):
            probs.append("Persian character in the block")
        if T.d2_on(self.cfg) and T._ts().mentions_excluded(text):
            probs.append("excluded type in the block")
        mk = C.mask_problems("\n".join(m["content"] for m in msgs))
        if mk:
            probs.append(f"masking gate: {mk}")
        same = "S*" if sha == r["requests"]["S*"] else ("none" if sha == r["requests"]["none"] else None)
        out = {"ns": self.ns, "scene": A.KITCHENS[self.ns], "arm": arm, "kind": A.ARMS[arm]["kind"], "pres": p, "item": r["item"], "lang": r["lang"], "level": r["level"],
               "text": text, "text_sha": A.sha256_text(text) if text else None, "empty": not text, "ids": list(ids), "removed": list(removed),
               "counts": cts, "real_max": real, "retrieved_max": retrieved, "full_counts": r["blocks"]["S*"].get("counts") or {},
               "checks": {"ok": not probs, "problems": probs, "retrieved_cap_exempt": bool(retrieved_cap_exempt)},
               "request_sha": sha, "same_request_as": same, "planned": arm == "rep" or same is None}
        if extra:
            out.update(extra)
        return out

    # ---------------------------------------------------------------- arms
    def edit_arm(self, arm: str) -> list:
        out = []
        for p in sorted(self.cap):
            pb = self.parsed[p]
            new, removed = A.edit_block(pb, arm)
            text = A.render_block(new)
            ids = [x["id"] for x in new["records"]] + (A.bundle_ids(self.cap[p]["blocks"]["S*"]["ids"]) if new["bundle"] else [])
            out.append(self.row(arm, p, text, ids, removed=removed))
        return out

    def rep_arm(self) -> list:
        return [self.row("rep", p, self.cap[p]["blocks"]["S*"]["text"], self.cap[p]["blocks"]["S*"]["ids"]) for p in sorted(self.cap)]

    def kitchen_arm(self, arm: str) -> list:
        """The other kitchen's frozen stores, the same query. One declared filter: a practice EXAMPLE whose quoted instruction is Persian is
        never offered (the package contract refuses any Persian character; in the other kitchen's own test the ranking never put such an
        example first, here the development kitchen's observations sometimes do). It can only change a presentation whose package would
        otherwise be refused (the example quota is 1); the filtered candidates are counted per row."""
        ns = A.ARMS[arm]["ns"]
        log = []

        class NoPersianExamples:
            def __init__(self, src):
                self.src = src

            def __getattr__(self, k):
                return getattr(self.src, k)

            def candidates(self, *a, **k):
                out = []
                for c in self.src.candidates(*a, **k):
                    rec = c.get("rec") if isinstance(c, dict) and "rec" in c else c
                    txt = str(((rec or {}).get("text") or {}).get("en") or "")
                    if any("؀" <= ch <= "ۿ" for ch in txt):
                        log.append((rec or {}).get("id"))
                        continue
                    out.append(c)
                return out

        class Pk(T.Packager):
            def stores(self, key, db_path, rows):
                st = super().stores(key, db_path, rows)
                if "example" in st and not isinstance(st["example"], NoPersianExamples):
                    st["example"] = NoPersianExamples(st["example"])
                return st

        pk = Pk(ns, self.cnt)
        d = T.source_dir(ns, "S*")
        rows = T.read_rows(d)
        db = os.path.join(d, "ltm.db")
        src = {"snapshot_dir": os.path.relpath(d, A.ROOT).replace(os.sep, "/"), "snapshot_sha": C.dir_sha(d), "rows_sha": T.rows_sha(rows),
               "rows_n": len(rows), "scene": C.ns_scene(ns)}
        out = []
        for p in sorted(self.cap):
            r = self.cap[p]
            del log[:]
            try:
                pkg = pk.render("S*", db, rows, item=r["item"], instruction=r["instruction"], lang=r["lang"], holding=r.get("holding"),
                                summary_lines=list(r.get("prompt_summary") or []))
                text, err = pkg["text"] or "", None
            except T.TreatmentError as e:
                pkg, text, err = {}, "", str(e)[:300]
            ids = list(pkg.get("kept") or [])
            row = self.row(arm, p, text, ids, extra={"kitchen": src, "pkg_sha": pkg.get("pkg_sha"), "store_shas": pkg.get("store_shas"),
                                                     "package_tokens": pkg.get("tokens"), "packager_error": err,
                                                     "persian_examples_filtered": sorted(set(x for x in log if x)),
                                                     "same_text_as_full": text == r["blocks"]["S*"]["text"]})
            if err:
                row["checks"]["ok"] = False
                row["checks"]["problems"].append("packager: " + err)
            elif pkg.get("tokens", {}).get("counts") and {k: int(v) for k, v in pkg["tokens"]["counts"].items()} != row["counts"]:
                raise SystemExit(f"{arm} {p}: packager counts differ from the row counts")
            out.append(row)
        return out

    def irrelevant_arm(self, no_request_rows: list) -> list:
        """Each non-empty block is replaced by records retrieved for OTHER instructions (another family), with the restatement and the
        need bundle of the donor left out, no record line in common with the target block and (tier 1) no object type in common, matched
        to the target's Gemma token count (the planner's tokenizer; Anthropic as tie-break). A single donor block when one lies within the
        tolerance, else a merge of whole donor blocks (records grouped by section, headers once)."""
        don = {}
        for r in no_request_rows:
            pb = A.parse_block(r["text"], r["ids"])
            if not pb["records"]:
                continue
            don[r["pres"]] = pb["records"]
        dtext = {q: A.render_block({"records": recs, "bundle": []}) for q, recs in don.items()}
        dcnt = {q: self.counts_of(t) for q, t in dtext.items()}
        dlines = {q: {x["line"] for x in recs} for q, recs in don.items()}
        dtypes = {q: types_in(dtext[q], self.vocab) for q in don}
        hdr = self.counts_of(".")
        out = []
        for p in sorted(self.cap):
            r = self.cap[p]
            full = r["blocks"]["S*"]
            if not full["text"]:
                out.append(self.row("irrelevant", p, "", [], extra={"donors": [], "match": None}))
                continue
            tg, ta = int(full["counts"]["gemma"]), int(full["counts"]["anthropic"])
            plines = {x["line"] for x in self.parsed[p]["records"]} | set(self.parsed[p]["bundle"])
            ptypes = types_in(full["text"], self.vocab)
            fam = self.fam[r["item"]]
            base = [q for q in don if self.fam[self.cap[q]["item"]] != fam and not (dlines[q] & plines)]
            tier1 = [q for q in base if not (dtypes[q] & ptypes)]
            tol = max(3, round(0.05 * tg))

            def key(q):
                return (abs(dcnt[q]["gemma"] - tg), abs(dcnt[q]["anthropic"] - ta), A.sha256_text(f"{A.SEED}|{p}|{q}"))
            pick, tier = None, None
            for tname, pool in (("1", tier1), ("2", base)):
                if pool:
                    b = min(pool, key=key)
                    if abs(dcnt[b]["gemma"] - tg) <= tol:
                        pick, tier = [b], tname
                        break
            if pick is None:                                        # merge whole donor blocks (tier 1 first, if it is not tiny)
                # a NEED target's block lists every inventory type, so type-disjoint donors hardly exist: tier 2 then
                pool, tname = (tier1, "1") if (len(tier1) >= 10 and not self.parsed[p]["bundle"]) else (base, "2")
                cur = [min(pool, key=lambda q: (dcnt[q]["gemma"] > tg, abs(dcnt[q]["gemma"] - tg), A.sha256_text(f"{A.SEED}|{p}|{q}")))]

                def merged(qs):
                    seen_, recs_ = set(), []
                    for qq in qs:
                        for x in don[qq]:
                            if x["line"] not in seen_:
                                seen_.add(x["line"])
                                recs_.append(x)
                    return A.render_block({"records": [x for s in A.SECTIONS for x in recs_ if x["comp"] == s], "bundle": []}), seen_
                txt, used = merged(cur)
                cur_g = self.counts_of(txt)["gemma"]
                rate = cur_g / max(1, len(txt))                      # tokens per character, for the next donor's estimate
                best = (abs(cur_g - tg), list(cur))
                for _ in range(40):
                    if abs(cur_g - tg) <= tol or cur_g > tg:
                        break
                    rest = [q for q in pool if q not in cur and (dlines[q] - used)]
                    if not rest:
                        break
                    q = min(rest, key=lambda q: (abs(cur_g + rate * sum(len(l) + 1 for l in dlines[q] - used) - tg),
                                                 A.sha256_text(f"{A.SEED}|{p}|{q}")))
                    cur.append(q)
                    txt, used = merged(cur)
                    cur_g = self.counts_of(txt)["gemma"]
                    if abs(cur_g - tg) < best[0]:
                        best = (abs(cur_g - tg), list(cur))
                pick, tier = best[1], tname + "-merge"
            recs = []
            seen = set()
            for q in pick:
                for x in don[q]:
                    if x["line"] not in seen:
                        seen.add(x["line"])
                        recs.append(x)
            recs = [x for s in A.SECTIONS for x in recs if x["comp"] == s]          # section order (within a section: donor order)
            text = A.render_block({"records": recs, "bundle": []})
            row = self.row("irrelevant", p, text, [x["id"] for x in recs], retrieved_cap_exempt=bool(self.parsed[p]["bundle"]),
                           extra={"donors": pick, "match": {"tier": tier, "target": {"gemma": tg, "anthropic": ta}, "tolerance_gemma": tol,
                                                            "n_tier1": len(tier1), "n_base": len(base)}})
            row["match"]["delta"] = {k: row["counts"].get(k, 0) - v for k, v in (("gemma", tg), ("anthropic", ta))}
            out.append(row)
        return out


def summarise(rows: list, gates: dict, counter: Counter) -> dict:
    by = {}
    for r in rows:
        d = by.setdefault(f"{r['ns']}/{r['arm']}", {"n": 0, "nonempty": 0, "changed_request": 0, "same_as_full": 0, "same_as_none": 0, "planned": 0,
                                     "checks_failed": 0, "real_max_max": 0})
        d["n"] += 1
        d["nonempty"] += not r["empty"]
        d["changed_request"] += r["same_request_as"] is None
        d["same_as_full"] += r["same_request_as"] == "S*"
        d["same_as_none"] += r["same_request_as"] == "none"
        d["planned"] += bool(r["planned"])
        d["checks_failed"] += not r["checks"]["ok"]
        d["real_max_max"] = max(d["real_max_max"], r["real_max"])
        if r["arm"] == "irrelevant" and r.get("match"):
            d.setdefault("abs_delta_gemma", []).append(abs(r["match"]["delta"]["gemma"]))
            d.setdefault("abs_delta_anthropic", []).append(abs(r["match"]["delta"]["anthropic"]))
            d.setdefault("tiers", {}).setdefault(r["match"]["tier"], 0)
            d["tiers"][r["match"]["tier"]] += 1
    for d in by.values():
        for k in ("abs_delta_gemma", "abs_delta_anthropic"):
            if k in d:
                v = sorted(d[k])
                d[k] = {"mean": round(sum(v) / len(v), 2), "median": v[len(v) // 2], "max": v[-1], "n": len(v)}
    return {"gates": gates, "arms": by, "counter_live_calls": counter.live_calls, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arm_labels": {k: v["label"] for k, v in A.ARMS.items()}}


def prefetch_anthropic(namespaces, threads: int = 4) -> dict:
    """No GPU: the Anthropic counts (free count_tokens) of every edit-arm text and donor text, written into the private cache, so the
    GPU-locked pass only spends Ollama time. The Gemma side is untouched here (Ollama is never called)."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from real_counter import RealMaxCounter
    if not os.path.exists(TOKCACHE):
        Counter(live=False)                                           # seeds the private cache
    rc = RealMaxCounter(TOKCACHE, PACKAGE_HEADER)
    texts = set()
    for ns in namespaces:
        for p, r in A.load_capture(ns).items():
            pb = A.parse_block(r["blocks"]["S*"]["text"], r["blocks"]["S*"]["ids"])
            for arm in ("no_request", "no_guide", "no_procedures", "no_inventory", "no_examples_templates"):
                new, _ = A.edit_block(pb, arm)
                t = A.render_block(new)
                if t:
                    texts.add(PACKAGE_HEADER + "\n" + t)
                    i = t.find(A.BUNDLE_START)
                    if i >= 0 and t[:i].rstrip("\n"):
                        texts.add(PACKAGE_HEADER + "\n" + t[:i].rstrip("\n"))
                if arm == "no_request":
                    d = A.render_block({"records": new["records"], "bundle": []})
                    if d:
                        texts.add(PACKAGE_HEADER + "\n" + d)
    texts.add(PACKAGE_HEADER + "\n" + ".")
    import hashlib
    todo = [t for t in sorted(texts) if "anth_raw" not in rc.cache.get(hashlib.sha256(t.encode("utf-8")).hexdigest()[:24], {})]
    lk = threading.Lock()
    rc.baseline()
    done = [0]

    def one(t):
        n = rc._anth_raw(t)
        with lk:
            e = rc.cache.setdefault(hashlib.sha256(t.encode("utf-8")).hexdigest()[:24], {})
            e["anth_raw"], e["chars"] = n, len(t)
            done[0] += 1
            if done[0] % 100 == 0:
                rc.save()
                print(f"  anthropic counts {done[0]} / {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, todo))
    rc.save()
    return {"texts": len(texts), "counted": len(todo), "calls": dict(rc.n_calls)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-live", action="store_true", help="cache-only token counts (no Ollama / API): fails on a miss")
    ap.add_argument("--prefetch-anthropic", action="store_true", help="no GPU: fill the Anthropic counts only, then exit")
    ap.add_argument("--ns", nargs="+", default=[A.NS], choices=sorted(A.KITCHENS))
    a = ap.parse_args(argv)
    if a.prefetch_anthropic:
        print(json.dumps(prefetch_anthropic(a.ns)))
        return 0
    live = not a.no_live
    if live and not lock_held_by_a():
        print("refused: live token counting uses Ollama; run under `gpu_lock.py run A -- ...`", file=sys.stderr)
        return 2
    cnt = Counter(live)
    bad_all = 0
    gates_all = {}
    try:
        for ns in a.ns:
            bad_all += run_ns(ns, cnt, gates_all)
    finally:
        if live:
            unload_gemma()
    return 0 if not bad_all else 1


def unload_gemma(endpoint: str = "http://127.0.0.1:11434"):
    """Free the GPU before the lock is released: keep_alive 0 for the counter's model (the counter's requests use Ollama's default)."""
    import plan as PL_
    from real_counter import OLLAMA_MODEL
    left = PL_.OllamaCtl(endpoint).unload(OLLAMA_MODEL)
    print(f"gemma unloaded; still loaded: {left}", flush=True)


def run_ns(ns: str, cnt: Counter, gates_all: dict) -> int:
    b = Builder(cnt, ns)
    gates = b.gates()
    gates_all[ns] = gates
    print(ns, "gates ok:", json.dumps(gates), flush=True)
    keep = [r for r in A.read_jsonl(OUT) if r.get("ns") != ns]        # other kitchens' rows stay as they are
    rows = []
    order = [x for x in ["rep", "no_request", "no_guide", "no_procedures", "no_inventory", "no_examples_templates", "kitchen_fp1",
                         "kitchen_fp15", "kitchen_fp17", "irrelevant"] if x in A.arms_for(ns)]
    done = []
    for arm in order:
        t0 = time.time()
        if arm == "rep":
            rs = b.rep_arm()
        elif A.ARMS[arm]["kind"] == "edit":
            rs = b.edit_arm(arm)
        elif A.ARMS[arm]["kind"] == "kitchen":
            rs = b.kitchen_arm(arm)
        else:
            rs = b.irrelevant_arm([r for r in rows if r["arm"] == "no_request"])
        rows += rs
        cnt.save()
        A.atomic_write_jsonl(OUT, keep + rows)                        # checkpoint: every finished arm
        A.atomic_write_json(SUMMARY, summarise(keep + rows, gates_all, cnt))
        n_ch = sum(1 for r in rs if r["same_request_as"] is None)
        done.append(f"A2 arm {arm}: {n_ch} changed requests, {sum(1 for r in rs if not r['checks']['ok'])} check failures")
        print(f"{ns} {arm}: rows {len(rs)} changed {n_ch} failed {sum(1 for r in rs if not r['checks']['ok'])} ({time.time() - t0:.0f} s; "
              f"live counts {cnt.live_calls})", flush=True)
        A.update_resume(f"A2 render {ns} (running)", ["A1 design written (A/design.md)"] + done, 0.0, RESUME_CMD.format(ns=ns),
                        ["A2 is deterministic: a re-run recomputes every arm from the cached token counts and rewrites blocks.jsonl atomically."])
    bad = [r for r in rows if not r["checks"]["ok"]]
    print(ns, "total rows", len(rows), "check failures", len(bad))
    for r in bad[:10]:
        print("  FAIL", r["arm"], r["pres"], r["checks"]["problems"])
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
