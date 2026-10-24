"""C4-open gate (a): the ft-v2 FS block (fs-v2) in the locked runner (treatment.FSRenderer + capture._render_ft2; prereg A24 item 4, A25 item 2
table; Astra docs/final-test/_astra/C4-open.txt Q-a). Offline: Windows python, repo root; no simulator, no LLM, no network.

    python sim/_ftm_fsv2_selftest.py

  1  same corpus, same scorer, same counter, same maximum as the package (the Packager's own example store object; PackageBuilder._rank;
     FT2["package_max_tokens"]); the A17 query (no goal-schema keys; no global toggled); successes AND failures eligible
  2  the REAL practice-v1 S* store + its order-0 rows: deterministic bytes, within 2,000 under every tokenizer, whole raw lines,
     every retained line ends with its own verdict, no Knife / alias, the leave-out of the presentation's own item (logged)
  3  synthetic corpora: test presentation + own-item record -> integrity failure; a test-citing record anywhere in the corpus -> failure;
     D2 alias filter (knife / knives / چاقو dropped, ButterKnife / butter knife / چاقوی کره kept); duplicate / missing ids fail; empty
     block reasons (eligibility / filtered / budget); a budget prefix of whole lines
  4  capture ft-v2: FS without fs_blocks.json, FS-only capture still needs S*, an empty FS block = the none request bytes (G-iso),
     identical bytes for G+FS and O+FS (one block), ft-v1 capture still reads fs_blocks.json (legacy path untouched)
  5  fs_block: _verdict_of reads provenance; render_fs_v2 with no ranker is the old fs-v2 (36/36 gate unchanged)
  6  INFO early signal (dev, non-test items, practice-v1 store, fake counter): empty rate, tokens, EN/FA retained-id agreement
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import types as _types

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                     # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                                    # noqa: E402
import treatment as T                                                 # noqa: E402
import fs_block as FB                                                 # noqa: E402
import capture as K                                                   # noqa: E402
import ltm_package as LP                                              # noqa: E402
import ltm_examples as LE                                             # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)[:400]))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


class FakeCounter:
    tokenizers = ["gemma", "anthropic"]

    def counts(self, text):
        return {"gemma": len(text) // 4 + 1, "anthropic": len(text) // 3 + 1}

    def __call__(self, text):
        return max(self.counts(text).values()) if text else 0

    def save(self):
        pass


MAN = C.load_manifest()
TEST_IDS = set(MAN["test"]) | set(MAN["test_uncapped_not_scored"])
GEN = {i["id"]: i for i in json.load(open(os.path.join(ROOT, "sim", "dataset", "general.json"), encoding="utf-8"))["items"]}
SNAP_DB = os.path.join(ROOT, "sim", "final_test", "snapshots", "o0", "k10", "ltm.db")
ROWS10 = LP.load_practice_rows("practice-v1")
SUMMARY = ["Apple [?] on/in CounterTop at (1.00, 2.00), step 3", "Fridge [closed] on/in Floor at (2.00, 1.00), step 4",
           "Microwave [closed] on/in CounterTop at (1.50, 2.50), step 6", "Bread [?] on/in CounterTop at (1.20, 2.10), step 7",
           "Lettuce [?] on/in Fridge at (2.00, 1.00), step 8", "SinkBasin [?] on/in Sink at (0.50, 2.50), step 8"]


def packager(ns="rehearsal-v3", test_side=None, factory=None, cfg=None):
    return T.Packager(ns, FakeCounter(), cfg=cfg or T.config(), test_side=test_side, stores_factory=factory)


def render(fr, item, lang, instr=None, rows=ROWS10, db=SNAP_DB, key="S*"):
    return fr.render(key, db, rows, item=item, instruction=instr if instr is not None else GEN[item][lang], lang=lang, holding=None,
                     summary_lines=SUMMARY)


# ================================================================================================ 1 static / same machinery
print("== 1 same corpus, scorer, counter and maximum as the package")
check("1.1 the switch pins fs-v2 and ONE maximum for package and FS", T.FT2["fs_version"] == "fs-v2" and T.FT2["package_max_tokens"] == 2000)
check("1.2 fs_block.py and the A17 lexicon are part of the treatment identity",
      "sim/final_test/matrix/fs_block.py" in T.IDENTITY_FILES and "sim/step5/lexicon/concepts.json" in T.IDENTITY_FILES)
pk = packager()
fr = T.FSRenderer(pk, test_ids=TEST_IDS)
check("1.3 the FS renderer refuses to exist without the namespace's packager", raises(lambda: T.FSRenderer(None), T.TreatmentError))
check("1.4 an ft-v2 switch that names another FS version is refused",
      raises(lambda: T.FSRenderer(packager(cfg=dict(T.config(), fs_version="fs-v1"))), T.TreatmentError))
st = pk.stores("S*", SNAP_DB, ROWS10)
it0 = "g-l1-pickup-bread"
g0 = LP.USE_GOAL_SCHEMA
o_en = render(fr, it0, "en")
check("1.5 FS uses the Packager's OWN example store object (same corpus before ranking; one cache)",
      pk.stores("S*", SNAP_DB, ROWS10)["example"] is st["example"] and o_en["corpus_sha"] == st["example"].sha())
check("1.6 the FS ranker is the package's scorer (PackageBuilder._rank, example section)", o_en["ranker"].get("ranker") == "ltm_package.PackageBuilder._rank(example)")
q_fs = T.fs_query(GEN[it0]["fa"], "fa", None, SUMMARY)
check("1.7 the FS query is the A17-only query (ltm_package._make_query_legacy), no global toggled",
      q_fs == LP._make_query_legacy(GEN[it0]["fa"], "fa", None, SUMMARY) and LP.USE_GOAL_SCHEMA == g0)
cnt = {"diff": 0, "n": 0}
for iid in sorted(MAN["nontest"])[:60]:
    for lg in ("en", "fa"):
        if GEN.get(iid, {}).get(lg):
            cnt["n"] += 1
            cnt["diff"] += int(T.fs_query(GEN[iid][lg], lg, None, SUMMARY) != LP.make_query_profile(T.FT2["package_profile"], GEN[iid][lg], lg, None, SUMMARY))
print("   INFO the +M (goal-schema) query differs from the FS (A17) query on %d of %d non-test presentations" % (cnt["diff"], cnt["n"]))
check("1.8 the FS query never carries a forbidden key (level / item / plan / goal / gold)", not (set(q_fs) & set(FB._FORBIDDEN_QUERY_KEYS)))
# the ranker is literally the package's: same candidates -> same order as PackageBuilder._rank
prm = T.effective_params(T.FT2["package_profile"], cfg=T.config())
b = LP.PackageBuilder("FloorPlan1", st, pk.counter, prm, categories=pk.categories())
qq = T.fs_query(GEN[it0]["en"], "en", None, SUMMARY)
cands = list(st["example"].candidates(qq, include_failures=True))
r1 = [c["rec"]["id"] for c, _s in fr.ranker(st, prm)(cands, qq)[0]]
r2 = [c["rec"]["id"] for c, _s in b._rank("example", cands, qq)[0]]
check("1.9 same candidates -> the FS order equals PackageBuilder._rank's order (weights, need discount, irrelevance, tie-break)", r1 == r2 and len(cands) > 0, (r1, r2))
check("1.10 failures are eligible for FS (include_failures=True), the package keeps successes only",
      any(c["rec"]["provenance"].get("verdict") == "failure" for c in cands)
      and not any(c["rec"]["provenance"].get("verdict") == "failure" for c in st["example"].candidates(qq, include_failures=False)))

# ================================================================================================ 2 the real store
print("== 2 the real practice-v1 S* store")
o_en2 = render(fr, it0, "en")
check("2.1 deterministic: re-render gives identical bytes and ids", o_en2["text"] == o_en["text"] and o_en2["retained"] == o_en["retained"])
outs = {}
for iid in sorted(MAN["dev_pilot"]):
    for lg in ("en", "fa"):
        outs[(iid, lg)] = render(fr, iid, lg)
nonempty = [o for o in outs.values() if o["text"]]
check("2.2 the dev presentations render (some FS blocks are non-empty)", len(nonempty) > 0, len(nonempty))
check("2.3 every block within the 2,000 maximum under every tokenizer, header included",
      all(o["real_tokens"] <= 2000 and o["max_tokens"] == 2000 for o in outs.values()))
lines_ok = True
for o in nonempty:
    for ln in o["text"].split("\n"):
        if not (ln.endswith("| success") or ln.endswith("| failure")):
            lines_ok = False
check("2.4 whole raw lines only, each ending with its own verdict", lines_ok)
check("2.5 no excluded type or alias in any FS text", not any(T._ts().mentions_excluded(o["text"]) or T.excluded_alias_hits(o["text"]) for o in outs.values()))
check("2.6 retained kinds use the real verdicts (no 'unknown': Astra a4)", all("unknown" not in o["retained_kinds"] for o in outs.values()))
own = [r["item"] for r in ROWS10]
o_own = render(fr, own[3], ROWS10[3]["lang"])
own_id = "example/%s/practice-v1:o0:a%d" % (LP.EXAMPLE_LABEL, ROWS10[3]["attempt"])
check("2.7 the presentation's own practice example is left out and logged (non-test side), never retained",
      any(e["component"] == "example" and e["id"] == own_id for e in o_own["leave_out"]) and own_id not in o_own["retained"], o_own["leave_out"])
check("2.8 every block records corpus / query / params shas, the parser, the ranker and the Persian flag",
      all(o.get("corpus_sha") and o.get("query_sha") and o.get("params_sha") and o.get("parser", "").startswith("A17")
          and "has_persian" in o for o in outs.values()))
print("   INFO dev blocks: %d of %d non-empty; with Persian text: %d; empty reasons: %s" % (
    len(nonempty), len(outs), sum(o["has_persian"] for o in outs.values()),
    sorted({o["empty_reason"] for o in outs.values() if not o["text"]})))

# ================================================================================================ 3 synthetic corpora
print("== 3 synthetic corpora")


def rows_with(mut):
    rs = copy.deepcopy(ROWS10)
    mut(rs)
    return rs


def factory_for(rows):
    def f(db, _rows):
        return {"example": LE.ExampleIndex(LE.build_corpus(rows, LP.EXAMPLE_LABEL, guard=False))}
    return f


tid = sorted(MAN["test"])[0]


def mk(rows, test_side=False, ns="rehearsal-v3"):
    p = packager(ns=ns, test_side=test_side, factory=factory_for(rows))
    return T.FSRenderer(p, test_ids=TEST_IDS)


rows_a = rows_with(lambda rs: None)
fr_t = mk(rows_a, test_side=True)
check("3.1 a TEST presentation whose own item has a practice example -> integrity failure (never a silent drop)",
      raises(lambda: render(fr_t, rows_a[3]["item"], rows_a[3]["lang"], instr=rows_a[3]["instruction"], rows=rows_a), T.TreatmentError))


def put_test(rs):
    rs[0]["item"] = tid
rows_b = rows_with(put_test)
fr_b = mk(rows_b)
check("3.2 a test-citing record ANYWHERE in the corpus fails the whole FS source (Astra a1), even for an unrelated query",
      raises(lambda: render(fr_b, it0, "en", rows=rows_b), T.TreatmentError))
check("3.3 corpus_test_citations lists it", T.corpus_test_citations(mk(rows_b).pk.stores("S*", SNAP_DB, rows_b)["example"], TEST_IDS)[0]["items"] == [tid])


def knives(rs):
    rs[3]["instruction"] = "pick up the knife and the bread"
    rs[4]["instruction"] = "چاقو را بردار"
    rs[8]["instruction"] = "pick up the butter knife"
rows_c = rows_with(knives)
fr_c = mk(rows_c)
oc = render(fr_c, "g-l1-pickup-butter-knife", "en", instr="pick up the bread", rows=rows_c)
dropped_ids = {d["id"] for d in oc["d2_dropped"]}
ex_c = fr_c.pk.stores("S*", SNAP_DB, rows_c)["example"]
log_c = []
kept_c = [c["rec"]["id"] for c in T._FSIndex(ex_c, ex_c, d2=True, log=log_c).candidates(T.fs_query("pick up the bread", "en", None, []))]
raw_c = [c["rec"]["id"] for c in ex_c.candidates(T.fs_query("pick up the bread", "en", None, []), include_failures=True)]
a = lambda n: "example/%s/practice-v1:o0:a%d" % (LP.EXAMPLE_LABEL, n)          # noqa: E731
check("3.4 D2: practice lines with 'knife' (lower case) and 'چاقو' are dropped BEFORE ranking and logged; 'butter knife' is kept",
      a(4) in raw_c and a(5) in raw_c and a(9) in raw_c and a(4) not in kept_c and a(5) not in kept_c and a(9) in kept_c
      and {d["id"] for d in log_c} == {a(4), a(5)} and {a(4), a(5)} <= dropped_ids and not ({a(4), a(5)} & set(oc["retained"])), (raw_c, kept_c, log_c))
check("3.5 alias detector: knife / Knife / knives / چاقو / چاقوی تیز hit; ButterKnife / butter knife / چاقوی کره / کارد کره do not",
      all(T.excluded_alias_hits(t) for t in ("pick up the knife", "Knife", "two knives", "چاقو را بردار", "چاقوی تیز"))
      and not any(T.excluded_alias_hits(t) for t in ("ButterKnife", "pick up the butter knife", "Butter Knives", "چاقوی کره را بردار", "کارد کره", "pickup ButterKnife")))
check("3.6 the alias check is a superset of type_scope.mentions_excluded on plan tokens", all(
    bool(T.excluded_alias_hits(t)) >= bool(T._ts().mentions_excluded(t)) for t in ("pickup Knife; slice Bread", "pickup ButterKnife", "Knife>Sink")))
# empty reasons
o_empty = render(mk(rows_a), it0, "en", instr="zzz qqq", rows=rows_a)
check("3.7 nothing eligible -> empty text, reason 'eligibility', no padding", o_empty["text"] == "" and o_empty["empty_reason"] == "eligibility", o_empty)
small = T.FSRenderer(packager(factory=factory_for(rows_a), cfg=dict(T.config(), package_max_tokens=30)), test_ids=TEST_IDS)
o_small = render(small, it0, "en", rows=rows_a)
check("3.8 a maximum below the first line -> empty, reason 'budget'", o_small["text"] == "" and o_small["empty_reason"] == "budget", o_small["empty_reason"])
full = render(mk(rows_a), it0, "en", rows=rows_a)
if full["text"].count("\n") >= 1:
    first = full["text"].split("\n")[0]
    mid = FakeCounter().counts(FB.experience_header() + "\n" + first)
    cap_ = max(mid.values()) + 2
    pre = T.FSRenderer(packager(factory=factory_for(rows_a), cfg=dict(T.config(), package_max_tokens=cap_)), test_ids=TEST_IDS)
    op = render(pre, it0, "en", rows=rows_a)
    check("3.9 a tight maximum keeps the longest PREFIX of whole lines (the first line only)", op["text"] == first and op["real_tokens"] <= cap_)
else:
    check("3.9 a tight maximum keeps the longest PREFIX of whole lines (needs >= 2 lines; got 1, prefix = whole)", True)


class Dup:
    corpus = [{"id": "x"}, {"id": "x"}]

    def sha(self):
        return "d"
check("3.10 a duplicate corpus id fails loudly", raises(lambda: T._FSIndex(Dup(), Dup(), d2=True, log=[]), T.TreatmentError))
fi = T._FSIndex(st["example"], st["example"], d2=True, log=[])
check("3.11 a record id missing from the corpus fails loudly", raises(lambda: fi.render({"id": "nope"}, "en"), T.TreatmentError))
check("3.12 FS never renders the package's (non-raw) form", raises(lambda: fi.render({"id": st["example"].corpus[0]["id"]}, "en", raw=False), T.TreatmentError))

# ================================================================================================ 4 capture ft-v2
print("== 4 capture ft-v2 (fake executor)")
from planners.contract import PlannerContract, EXPERIENCE_HEADER, prompt_sha256   # noqa: E402
import honest_skills as hs                                                         # noqa: E402
import plan as PLN                                                                 # noqa: E402
TYPES49 = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))["types"]
c2 = PlannerContract(T.planner_types(TYPES49), tools=hs.TOOLS, version="ft-v2")
tmp = tempfile.mkdtemp(prefix="ftm_fsv2_")
try:
    d = T.snapshot_dir("S*", tmp)
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(SNAP_DB, os.path.join(d, "ltm.db"))
    T.write_rows(d, ROWS10)

    class FakeOM:
        def __init__(self, lines):
            self.lines = list(lines)

        def summary(self, limit=40):
            return self.lines[:limit] if limit else list(self.lines)

    class FakeEx:
        frame, map_rev = "f0", "m0"

        def __init__(self, lines):
            self.last = {"skills": _types.SimpleNamespace(om=FakeOM(lines))}

    br = {"records_sha": "r", "n_records": 1, "allow_unapproved": True, "hint_table_sha": "h"}

    def mk_cap(only, fs_doc=None, ns="rehearsal-v3", items=None):
        cap = K.Capturer(namespace=ns, executor=FakeEx(SUMMARY), contract=c2, counter=FakeCounter(), open_sources=None, items=items or GEN,
                         manifest=MAN, fs_doc=fs_doc, root=tmp, only_sources=only, log=lambda *a, **k: None,
                         packager=T.Packager(ns, FakeCounter(), root=tmp))
        cap.ident = T.identity(ns, briefing=br)
        return cap

    post = {"summary": SUMMARY, "summary_sha": C.sha256_text("\n".join(SUMMARY)), "episode_id": "ep:1", "sweep_prims": 3}
    cap = mk_cap(["none", "S*", "FS"])
    cap.cur = {"pres": C.pres_id(it0, "en"), "level": GEN[it0]["level"], "lang": "en", "world": "canon"}
    out = cap.render_presentation(None, GEN[it0], None, post)
    check("4.1 ft-v2 capture renders FS with NO fs_blocks.json (fs_doc=None; Astra a6)", "FS" in out["requests"] and cap.fs_doc is None)
    fsb = out["blocks"]["FS"]
    check("4.2 the FS block records the fs-v2 metadata (ids, dropped, leave-out, D2, reason, shas, ranker, Persian flag)",
          fsb["fs"]["version"] == "fs-v2" and fsb["ids"] == fsb["fs"]["retained"] and fsb["fs"]["corpus_sha"] and "has_persian" in fsb["fs"]
          and fsb["source"]["rows_sha"] == T.rows_sha(ROWS10))
    m_fs = PLN.render_messages(c2, dict(out, treatment="ft-v2"), "FS")
    check("4.3 G+FS and O+FS read ONE capture block: plan re-renders the captured FS sha (identical bytes for both arms)",
          prompt_sha256(m_fs) == out["requests"]["FS"])
    if fsb["text"]:
        check("4.4 the FS request = none + exactly one EXPERIENCE_HEADER block", m_fs[1]["content"].count(EXPERIENCE_HEADER) == 1)
    else:
        check("4.4 an empty FS block renders no block (FS request = none request)", out["requests"]["FS"] == out["requests"]["none"])
    cap_fs = mk_cap(["none", "FS"])
    check("4.5 FS-only capture still needs (opens, hashes, checks) the S* snapshot", "S*" in cap_fs.needed_sources())
    items_e = dict(GEN)
    items_e["g-zz-empty"] = dict(GEN[it0], id="g-zz-empty", en="zzz qqq", fa="zzz qqq")
    cap_e = mk_cap(["none", "FS"], items=items_e)
    cap_e.cur = {"pres": C.pres_id("g-zz-empty", "en"), "level": "L1", "lang": "en", "world": "canon"}
    out_e = cap_e.render_presentation(None, items_e["g-zz-empty"], None, post)
    check("4.6 an empty FS block: FS request bytes == none request bytes; G-iso passes; reason recorded",
          out_e["requests"]["FS"] == out_e["requests"]["none"] and out_e["blocks"]["FS"]["empty"] and out_e["blocks"]["FS"]["fs"]["empty_reason"])
    capn = mk_cap(["none", "FS"])
    capn.fs_renderer = None
    capn.cur = dict(cap.cur)
    check("4.7 an ft-v2 capture without an FS renderer refuses the FS source", raises(lambda: capn.render_presentation(None, GEN[it0], None, post), K.CaptureAbort))
    FS_DOC = {lv: {"text": "Earlier: pickup Mug, put_in Mug Sink", "real_tokens": 10, "counts": {}, "budget": 2000} for lv in ("L1", "L2", "L3")}
    cleg = PlannerContract(TYPES49, tools=hs.TOOLS)
    leg = K.Capturer(namespace="rehearsal-v1", executor=FakeEx(SUMMARY), contract=cleg, counter=FakeCounter(), open_sources=None, items=GEN,
                     manifest=MAN, fs_doc=FS_DOC, root=tmp, only_sources=["none", "FS"], log=lambda *a, **k: None)
    leg.cur = dict(cap.cur)
    outl = leg.render_presentation(None, GEN[it0], None, post)
    check("4.8 ft-v1 capture still reads fs_blocks.json (legacy per-level fs-v1 text, untouched)",
          outl["blocks"]["FS"]["text"] == FS_DOC[GEN[it0]["level"]]["text"] and leg.fs_renderer is None)
    leg2 = K.Capturer(namespace="rehearsal-v1", executor=FakeEx(SUMMARY), contract=cleg, counter=FakeCounter(), open_sources=None, items=GEN,
                      manifest=MAN, fs_doc=None, root=tmp, only_sources=["none", "FS"], log=lambda *a, **k: None)
    check("4.9 ft-v1 without fs_blocks.json has no FS source (unchanged rule) and needs no S* for it", "S*" not in leg2.needed_sources())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ================================================================================================ 5 fs_block
print("== 5 fs_block")
check("5.1 _verdict_of reads the verdict from provenance (ExampleIndex.to_rec)", FB._verdict_of({"provenance": {"verdict": "failure"}}) == "failure"
      and FB._verdict_of({"verdict": "success"}) == "success" and FB._verdict_of({}) == "unknown")
src = open(os.path.join(HERE, "final_test", "matrix", "fs_block.py"), encoding="utf-8").read()
check("5.2 render_fs_v2 without a ranker still ranks with rank_examples (fs-v2 as specified)", "if ranker is None:" in src and "rank_examples(cands, query" in src)

# ================================================================================================ 6 early signal (INFO only)
print("== 6 INFO early signal: non-test items, practice-v1 S* store, fake counter")
fr6 = T.FSRenderer(packager(), test_ids=TEST_IDS)
agree = both_empty = pairs = 0
toks, empty, persian, n = [], 0, 0, 0
for iid in sorted(MAN["nontest"]):
    it = GEN.get(iid)
    if not it or not it.get("en") or not it.get("fa"):
        continue
    r_en, r_fa = render(fr6, iid, "en"), render(fr6, iid, "fa")
    pairs += 1
    agree += int(r_en["retained"] == r_fa["retained"])
    both_empty += int(not r_en["retained"] and not r_fa["retained"])
    for r in (r_en, r_fa):
        n += 1
        empty += int(not r["text"])
        persian += int(r["has_persian"])
        toks.append(r["real_tokens"])
print("   INFO %d presentations: empty %.0f %%, FS tokens mean %.0f max %d (fake counter), Persian-containing blocks %d; EN/FA retained-id "
      "agreement %d / %d pairs (both empty: %d)" % (n, 100.0 * empty / max(1, n), sum(toks) / max(1, len(toks)), max(toks or [0]), persian,
                                                     agree, pairs, both_empty))

print("\n_ftm_fsv2_selftest: %d/%d passed" % (N["pass"], N["pass"] + N["fail"]))
sys.exit(0 if N["fail"] == 0 else 1)
