"""Selftest for fs-v2 (PACKAGE-SPEC s10, stage B8): the FS control redesign. Offline: a fake ExampleIndex, fake two-tokenizer counter,
a synthetic kitchen ("SynthKitchen") with a different fixture set. Python 3.10 syntax.

    python sim/_pkg_fs_selftest.py            # gate
    python sim/_pkg_fs_selftest.py --early    # spec s10 early check on the 116 non-test instructions (stand-in index; diagnostic only)
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import common as C            # noqa: E402
import fs_block as F          # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


class Ctr:
    tokenizers = ["words", "chars3"]

    def counts(self, text):
        return {"words": len(text.split()), "chars3": len(text) // 3}

    def __call__(self, text):
        return max(self.counts(text).values())


class FakeIndex:
    """Stand-in for B5's ExampleIndex: candidates(query, include_failures) -> pkg.cand/1; render(rec, lang, raw)."""

    def __init__(self, exs):
        self.exs = exs
        self.calls = []

    def candidates(self, query, include_failures=False):
        self.calls.append(include_failures)
        qops = set(query.get("ops") or [])
        out = []
        for e in self.exs:
            if not include_failures and e["verdict"] != "success":
                continue
            sig = set(e["ops_sig"])
            m = (len(sig & qops) / len(sig | qops)) if (sig | qops) else 0.0
            out.append({"rec": {"id": e["id"], "component": "example", "kitchen": e["kitchen"], "keys": {"types": [], "ops": e["ops_sig"], "rels": []},
                                "lex": {"en": " ".join(e["ops_sig"]), "fa": ""}, "text": {"en": e["instruction"], "fa": e["instruction"]},
                                "verdict": e["verdict"], "prio": e["attempt"]}, "match": m, "why": []})
        return out

    def render(self, rec, lang, raw=False):
        assert raw is True
        e = next(x for x in self.exs if x["id"] == rec["id"])
        return f"{e['attempt']}. {e['lang']} | {e['instruction']} | {'; '.join(e['ops_sig'])} | {e['verdict']}"


def mk(kitchen, specs):
    return [{"id": f"example/{kitchen}/ns:o0:a{i}", "kitchen": kitchen, "attempt": i, "lang": lg, "instruction": ins, "ops_sig": ops,
             "verdict": v} for i, (lg, ins, ops, v) in enumerate(specs, 1)]


FP = mk("FloorPlan1", [("en", "put the apple in the fridge", ["pick_and_place"], "success"),
                       ("en", "close the fridge", ["close"], "success"),
                       ("en", "open the cabinet", ["open"], "failure"),
                       ("fa", "یخچال را باز کن", ["open"], "success"),
                       ("en", "slice the bread", ["slice"], "failure")])
SYNTH = mk("SynthKitchen", [("en", "open the second fridge", ["open"], "success"), ("en", "turn on burner 4", ["toggle_on"], "failure"),
                            ("en", "open the drawer", ["open"], "failure")])
Q = {"schema": "pkg.query/1", "instruction": "open the fridge", "lang": "en", "holding": None, "observed": [{"type": "Fridge", "state": "closed"}],
     "ops": ["open"], "types": ["Fridge"]}
CTR = Ctr()

# ------------------------------------------------------------------------------------------------ static
src = open(F.__file__, encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok = True
except SyntaxError:
    ok = False
check("fs_block parses as python 3.10", ok)
check("this selftest parses as python 3.10", (ast.parse(open(__file__, encoding="utf-8").read(), feature_version=(3, 10)) is not None))
check("default version is fs-v1", F.FS_VERSION == "fs-v1" and "fs-v2" in F.FS_VERSIONS)
ast_fn = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in ("render_fs_v2", "rank_examples")]
body_src = "\n".join(ast.get_source_segment(src, n) for n in ast_fn)
check("fs-v2 never reads a level / item / goal / plan key of the QUERY", not any(f'query["{k}"]' in body_src or f'query.get("{k}"' in body_src
                                                             for k in ("level", "item", "goal", "plan", "reference_plan")))
check("no simulator / network import", not any(m in src for m in ("import ai2thor", "import requests", "anthropic.Anthropic")))

# ------------------------------------------------------------------------------------------------ v1 stays byte-identical
pre = os.path.join(HERE, "final_test", "_checkpoints", "H-B8-fs", "fs_block.py.pre")
if os.path.exists(pre):
    import types
    old = types.ModuleType("fs_block_pre")
    old.__file__ = pre
    exec(compile(open(pre, encoding="utf-8").read(), pre, "exec"), old.__dict__)
    rows = []
    for a in range(1, 11):
        rows.append({"order": 0, "attempt": a, "item": f"g-x{a}", "lang": "en" if a % 2 else "fa", "instruction": f"do thing {a}", "verdict": "success" if a % 3 else "failure",
                     "plan": [["open", "Fridge", None, None]], "exec_steps": [{"ok": True}], "sightings": [{"otype": "Egg", "in_rtype": "Fridge", "state": None}]})
    a1 = old.render(rows, "L2", CTR, 1200, guard=False)
    a2 = F.render(rows, "L2", CTR, 1200, guard=False)
    a3 = F.render_fs("fs-v1", rows, "L2", CTR, 1200, guard=False)
    check("fs-v1 output byte-identical to the pre-change module (and via the switch)", a1 == a2 == a3, "differs")
else:
    check("pre-change copy present", False, pre)

# ------------------------------------------------------------------------------------------------ v2 core
idx = FakeIndex(FP)
r = F.render_fs_v2(Q, idx, CTR, 2000)
check("includes failures: index asked with include_failures=True", idx.calls == [True])
check("keeps successes and failures", set(r["retained_kinds"]) >= {"success", "failure"}, r["retained_kinds"])
check("irrelevant dropped (no key hit)", any(d["reason"] == "irrelevant" for d in r["dropped"]) and len(r["retained"]) == 2, r["dropped"])
check("ranked: both 'open' examples first, then close? (close/slice irrelevant)", r["retained"][0].endswith("a3") or r["retained"][0].endswith("a4"), r["retained"])
check("whole lines only (no line split)", all(l.count("|") == 3 for l in r["text"].split("\n")), r["text"])
check("under budget with header", r["real_tokens"] <= 2000 and r["budget"] == 2000)
check("sha256 = text sha", r["sha256"] == C.sha256_text(r["text"]))
check("raw: the practice-language line is kept (FA example in a EN query)", "fa |" in r["text"])
check("deterministic", F.render_fs_v2(Q, FakeIndex(FP), CTR, 2000) == r)
check("tie order stable: prio then id", [c for c in r["retained"]] == F.render_fs_v2(Q, FakeIndex(list(reversed(FP))), CTR, 2000)["retained"])

# budget: prefix, stop at first non-fit
hdr_n = F.real_tokens(CTR, F.experience_header(), "x")[0]
one = F.real_tokens(CTR, F.experience_header(), r["text"].splitlines()[0])[0]
tiny = F.render_fs_v2(Q, FakeIndex(FP), CTR, one + 1)
check("tiny budget keeps a prefix only", len(tiny["retained"]) == 1 and tiny["real_tokens"] <= one + 1, tiny["retained"])
check("rest reported dropped for budget", sum(d["reason"].startswith("budget") for d in tiny["dropped"]) == 1, tiny["dropped"])
none_fit = F.render_fs_v2(Q, FakeIndex(FP), CTR, 3)
check("nothing fits -> empty block, no header, tokens 0", none_fit["empty"] and none_fit["text"] == "" and none_fit["real_tokens"] == 0)
nomatch = F.render_fs_v2(dict(Q, instruction="zzz", ops=["unknown_op"]), FakeIndex(FP), CTR, 2000)
check("no candidate -> empty, not padded", nomatch["empty"] and nomatch["retained"] == [])
big = mk("FloorPlan1", [("en", "open the fridge number %d with a very long extra clause to take up tokens" % i, ["open"], "failure" if i % 2 else "success") for i in range(1, 60)])
rb = F.render_fs_v2(Q, FakeIndex(big), CTR, 400)
check("large corpus truncated under budget in both languages", rb["real_tokens"] <= 400 and 0 < len(rb["retained"]) < 59, rb["real_tokens"])
hdr_fa = {"fa": "ثبت‌های تمرین " * 40}
rfa = F.render_fs_v2(Q, FakeIndex(big), CTR, 400, headers=hdr_fa)
check("joint EN/FA fit: a longer FA header shrinks the block for both", len(rfa["retained"]) < len(rb["retained"]), (len(rfa["retained"]), len(rb["retained"])))
check("counts reported per tokenizer", set(rb["counts"]) == {"words", "chars3"})

# EN/FA parity: the same ids for the two presentations of one command
qa = dict(Q, lang="en", instruction="open the fridge")
qb = dict(Q, lang="fa", instruction="یخچال را باز کن")
check("lexical source is the neutral keys", F.render_fs_v2(qb, FakeIndex(FP), CTR)["ranker"]["lexical_source"] == "keys")
check("EN/FA parity when the index sees the same structure", F.parity(qa, qb, FakeIndex(FP), CTR)["same"])

# forbidden inputs
for k, v in (("level", "L3"), ("item", "g-l1-x"), ("goal", []), ("reference_plan", [])):
    check(f"query with {k!r} is refused", raises(lambda k=k, v=v: F.render_fs_v2(dict(Q, **{k: v}), FakeIndex(FP), CTR), ValueError))

# no cross-component content: only the index is consulted
check("only lines from the example index appear", all(any(e["instruction"] in l for e in FP) for l in r["text"].split("\n")))

# synthetic kitchen
rs = F.render_fs_v2(dict(Q, instruction="open the second fridge"), FakeIndex(SYNTH), CTR, 2000)
check("SynthKitchen: works, failures included, ids carry the kitchen", set(rs["retained_kinds"]) == {"success", "failure"} and all("SynthKitchen" in i for i in rs["retained"]), rs["retained"])
check("no scene constant in code", "FloorPlan" not in body_src and "g-l" not in body_src)

# ranker: dense hook, recorded
rd = F.render_fs_v2(Q, FakeIndex(FP), CTR, 2000, embedder=lambda q, cs: [0.9 if "slice" in c["rec"]["text"]["en"] else 0.0 for c in cs])
check("dense hook recorded and rescues a key-less but near example", rd["ranker"]["dense"] is True and any("a5" in i for i in rd["retained"]), rd["retained"])
check("without an embedder dense is recorded off", r["ranker"]["dense"] is False)

# Astra H-B8-fs: clipping cannot hide a failure; plumbing cannot carry the item id
class ClipIndex(FakeIndex):
    def render(self, rec, lang, raw=False):
        return super().render(rec, lang, raw=raw).rsplit(" | ", 1)[0]          # verdict clipped off


rc = F.render_fs_v2(Q, ClipIndex(FP), CTR)
check("a line whose verdict was clipped is dropped, never shown", rc["empty"] and any(d["reason"] == "verdict_not_in_line" for d in rc["dropped"]), rc["dropped"])


class LeakIndex(FakeIndex):
    def candidates(self, query, include_failures=False):
        cs = super().candidates(query, include_failures)
        for c in cs:
            c["rec"]["item"] = "g-l9-secret-item"
        return cs

    def render(self, rec, lang, raw=False):
        return super().render(rec, lang, raw=raw).replace("|", "| g-l9-secret-item |", 1)


check("an FS line carrying the item id is refused", raises(lambda: F.render_fs_v2(Q, LeakIndex(FP), CTR), ValueError))
check("rank_examples is public (the package can share the one ranking)", callable(F.rank_examples))

# ------------------------------------------------------------------------------------------------ corpus file
params = {"budget": 2000, "weights": F.FS2_WEIGHTS}
doc = F.fs_corpus_doc(FP, "practice-v2", params)
check("fs_corpus doc has the schema and fields", set(doc) == {"schema", "corpus_sha", "practice_ns", "params_sha", "sha256"} and doc["schema"] == "final-test.fs_corpus/2")
check("non-practice namespace refused", raises(lambda: F.fs_corpus_doc(FP, "test-v1", params), C.TestItemRefused))
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "fs_corpus.json")
    json.dump(doc, open(p, "w", encoding="utf-8"))
    check("corpus file verifies", F.check_corpus_file(p, FP, params) == [])
    check("changed corpus detected", "corpus changed" in F.check_corpus_file(p, FP[:-1], params))
    check("changed params detected", "params changed" in F.check_corpus_file(p, FP, dict(params, budget=1900)))
    d2 = dict(doc, practice_ns="x")
    json.dump(d2, open(p, "w", encoding="utf-8"))
    check("tampered file detected", "self-sha mismatch" in F.check_corpus_file(p))
check("missing file reported", F.check_corpus_file(os.path.join(tempfile.gettempdir(), "no-such-fs-corpus.json")) != [])

# early_check plumbing
qs = [dict(Q, lang="en", pair="p1"), dict(Q, lang="fa", instruction="یخچال را باز کن", pair="p1"), dict(Q, instruction="zzz", ops=["zz"], lang="en", pair="p2")]
ec = F.early_check(qs, FakeIndex(FP), CTR, package_tokens={("open the fridge", "en"): 800})
check("early_check: per-language stats, empty rate, failure share, parity", ec["languages"]["en"]["n"] == 2 and ec["languages"]["en"]["empty_rate"] == 0.5
      and ec["languages"]["en"]["share_with_failure"] == 0.5 and ec["en_fa_id_mismatch"] == 0 and ec["languages"]["en"]["package_tokens_mean"] == 800.0, ec)


# ------------------------------------------------------------------------------------------------ early check on real non-test data
def _early():
    sys.path.insert(0, HERE)
    import ltm_concepts as LC
    P = C.paths()
    split = json.load(open(P["split"], encoding="utf-8"))
    nontest = set(split["nontest"])
    ds = json.load(open(P["dataset"], encoding="utf-8"))["items"]
    items = [i for i in ds if i["id"] in nontest]
    # stand-in B5: the 10 order-0 practice rows of practice-v1 (the only practice corpus that exists before B5)
    rows = [r for r in C.Journal(C.journal_path("practice-v1", "practice")).rows() if r.get("order") == 0 and F.in_source(r)]
    norm = F.normalise_rows(rows)
    vmap = {"put": ["pick_and_place", "put_in"], "place": ["pick_and_place", "put_in"]}
    exs = []
    for a in norm:
        ops = sorted({s[0] for s in a["steps"]})
        ok_ = all(o == "ok" for o in a["outcomes"])
        exs.append({"id": f"example/standin/practice-v1:o0:a{a['attempt'] + 1}", "kitchen": "standin", "attempt": a["attempt"] + 1, "lang": a["lang"],
                    "instruction": a["instruction"], "ops_sig": ops, "verdict": a["verdict"]})
    idx = FakeIndex(exs)
    queries = []
    for it in items:
        for lg in ("en", "fa"):
            pr = LC.parse(it[lg])
            ops = sorted({o for r_ in pr["roles"] for o in vmap.get(r_["verb"], [r_["verb"]])})
            queries.append({"instruction": it[lg], "lang": lg, "holding": None, "observed": [], "ops": ops, "pair": it["id"]})
    ec = F.early_check(queries, idx, CTR, 2000)
    # real header and a crude counter: this measures plumbing and token scale only
    print(json.dumps({"stand_in_corpus_rows": len(exs), "failures_in_corpus": sum(e["verdict"] != "success" for e in exs), **ec}, indent=1))
    return ec


if __name__ == "__main__":
    if "--early" in sys.argv:
        _early()
    print(f"{N['pass']}/{N['pass'] + N['fail']}")
    sys.exit(0 if not N["fail"] else 1)
