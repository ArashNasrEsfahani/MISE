"""H4 selftest: the EN/FA block-record-id difference on g-l2-empty-mug-sink (C3).

Run from the repo root with Windows `python sim/final_test/_h4_selftest.py` (offline, no simulator, no LLM, no network).

Claims checked (each is an assertion):
 1. The C3 gate `bilingual_ids` differs for exactly this item (P and PF), nothing else.
 2. Cause: the concept lexicon lists the Persian word for mug as a Cup alias and gives Mug no Persian alias, so the FA text parses to
    Cup (and the elided object of the second clause is '?'), while EN parses to Mug. Memory is keyed by object type.
 3. The records are not "different ids for one record": the EN block holds Mug/pl:1; the FA block was EMPTY (no record retrieved for Cup).
 4. Pairing / scoring do not read block ids: c3_analyze joins on (item, lang) and scores `sr` from the row; block ids are used only by
    the `bilingual_ids` gate.
 5. Consequence in the data: both arms of the FA twin ran on an empty block, so P == PF == G0 for FA (same request, same plan), and the
    EN twin is the only P/PF presentation that actually used memory. It can bias the P-vs-G0 contrast ONLY via the retrieval being absent
    in FA (a coverage effect that the analysis already reports as `empty_blocks_by_arm_level_lang`), not via a wrong pairing.
 6. Scope: how many dataset items have a different set of target object types in EN vs FA parse (the real exposure).
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import ltm_concepts as lc                     # noqa: E402

FAILS = []
N = 0


def check(name, cond, detail=""):
    global N
    N += 1
    if not cond:
        FAILS.append(name)
    print(("ok   " if cond else "FAIL ") + name + (("  " + str(detail)) if detail and not cond else ""))


def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as fh:
        return json.load(fh)


ITEM = "g-l2-empty-mug-sink"
res = load("docs/final-test/c3/c3_results.json")
gen = load("sim/dataset/general.json")
by = {i["id"]: i for i in gen["items"]}
it = by[ITEM]

# 1 ---------------------------------------------------------------------------------------------------------------
bi = res["bilingual_ids"]
check("1 only this item differs (P and PF)", sorted(bi["differ"]) == [ITEM + ":P", ITEM + ":PF"], bi)
check("1 all other pairs identical", bi["identical"] == bi["pairs"] - 2, bi)

# 2 ---------------------------------------------------------------------------------------------------------------
# W4a (prereg A17): the cause is checked on the PRE-fix lexicon/parser (kept in _checkpoints/W4a-fa-lexicon) and the fix on the current ones.
import importlib.util                           # noqa: E402
CK = os.path.join(ROOT, "sim", "final_test", "_checkpoints", "W4a-fa-lexicon")
_sp = importlib.util.spec_from_file_location("ltm_concepts_pre_w4a", os.path.join(CK, "ltm_concepts.py"))
old = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(old)
olex = old.Lexicon.load(os.path.join(CK, "concepts.json"))
lex = lc.default_lexicon()
fa_word = it["fa"].split()[0]                                   # the Persian word for the mug as written in the record
check("2 (pre-fix) FA word was a Cup alias", fa_word in olex.aliases_for("Cup"), olex.aliases_for("Cup"))
check("2 (pre-fix) Mug's only Persian alias was the loan word", [a for a in olex.aliases_for("Mug") if re.search("[؀-ۿ]", a)] == ["ماگ"])
ope, opf = old.parse(it["en"], olex), old.parse(it["fa"], olex)
check("2 (pre-fix) FA parsed to Cup, not ok (no_object)", {r["obj"] for r in opf["roles"]} - {"-", "?"} == {"Cup"} and not opf["ok"] and "no_object" in opf["reasons"], opf)
check("2 (pre-fix) EN parsed to Mug, ok", ope["ok"] and {r["obj"] for r in ope["roles"]} == {"Mug"})
fa_mug = [a for a in lex.aliases_for("Mug") if re.search("[؀-ۿ]", a)]
check("2 (fix) Mug has the everyday word and the loan word; Cup keeps only the cup word",
      fa_word in fa_mug and "ماگ" in fa_mug and [a for a in lex.aliases_for("Cup") if re.search("[؀-ۿ]", a)] == ["فنجان"], fa_mug)
check("2 (fix) Mug.name_fa is the everyday word, Cup.name_fa stays distinct", lex.type_name("Mug", "fa") == fa_word and lex.type_name("Cup", "fa") == "فنجان")
pe, pf = lc.parse(it["en"]), lc.parse(it["fa"])
check("2 (fix) FA parses to Mug twice, with a destination, and is ok", pe["ok"] and pf["ok"] and
      [(r["verb"], r["obj"], r["dest"]) for r in pf["roles"]] == [("empty", "Mug", "-"), ("put", "Mug", "Sink")], pf)
check("2 (fix) the elided object was inherited and the parser says so", "fa_elided_object_inherited" in pf["reasons"] and pf["parser"] == lc.PARSER_VERSION)
check("2 (fix) English is not given the rule (a pronoun-less elision still fails closed)", not lc.parse("Pick up the apple and put in the fridge")["ok"])

# 3 ---------------------------------------------------------------------------------------------------------------
rows = [json.loads(x) for x in open(os.path.join(ROOT, "docs/final-test/c3/eval.jsonl"), encoding="utf-8") if x.strip()]
mine = {(r["lang"], r["arm"]): r for r in rows if r["item"] == ITEM}
blk = lambda lg, a: (mine[(lg, a)]["plan"] or {}).get("block")      # noqa: E731
for a in ("P", "PF"):
    check(f"3 EN {a} block holds one place record for Mug", blk("en", a)["ids"] == ["Mug/pl:1"], blk("en", a)["ids"])
    check(f"3 FA {a} block is empty (no record for Cup)", blk("fa", a)["empty"] and blk("fa", a)["ids"] == [], blk("fa", a))
check("3 record ids are type-keyed ('<Type>/pl:n'): the same record can only be named one way", blk("en", "P")["ids"][0].startswith("Mug/"))

# 4 ---------------------------------------------------------------------------------------------------------------
src = open(os.path.join(ROOT, "sim/final_test/c3/c3_analyze.py"), encoding="utf-8").read()
uses = [m.start() for m in re.finditer(r'\[\s*"ids"\s*\]|\.get\(\s*"ids"', src)]
lines = {src.count("\n", 0, u) + 1 for u in uses}
bid_start = src.index("# bilingual ids")
bid_end = src.index("# token budget compliance")
bid_lines = set(range(src.count("\n", 0, bid_start) + 1, src.count("\n", 0, bid_end) + 1))
other = lines - bid_lines
check("4 block ids are read only in the bilingual_ids gate and the proc/ line counter (never in pairing or scoring)",
      bool(lines) and all("proc/" in src.splitlines()[n - 1] for n in other), (sorted(lines), sorted(bid_lines)))
check("4 pairing joins on (item, lang) / item id", "item_score" in src and "app[(item, lg)]" in src)
# no 'sr' depends on the block: identical goal_ok when the request/plan is identical (G0 vs P vs PF in FA)
fa = {a: mine[("fa", a)] for a in ("G0", "P", "PF")}
check("4 FA twin: P, PF and G0 got the same plan", len({json.dumps(fa[a]["plan"]["plan"]) for a in fa}) == 1)
check("4 FA twin: scoring fields identical across arms", len({(fa[a]["goal_ok"], fa[a]["sr"]) for a in fa}) == 1)

# 5 ---------------------------------------------------------------------------------------------------------------
check("5 FA P request_sha differs from EN (different language, expected)", mine[("fa", "P")]["plan"]["request_sha"] != mine[("en", "P")]["plan"]["request_sha"])
emp = res["empty_blocks_by_arm_level_lang"]
check("5 empty FA blocks are already reported in the funnel (not hidden)", isinstance(emp, dict) and len(emp) > 0)

# 6 ---------------------------------------------------------------------------------------------------------------
# W4a: scope after the fix, on the 116 NON-TEST items only (parse of instruction text, no outcome); before-values come from the coverage report.
split = load("docs/final-test/split_manifest.json")
nontest = set(split["nontest"])
sel = [i for i in gen["items"] if i["id"] in nontest]
diff_types, mug_cup, parse_fa_ok = [], [], 0
for i in sel:
    e, f = lc.parse(i["en"]), lc.parse(i["fa"])
    es = {r["obj"] for r in e["roles"]} - {"-", "?"}
    fs = {r["obj"] for r in f["roles"]} - {"-", "?"}
    if es != fs:
        diff_types.append(i["id"])
        if "Mug" in es and "Cup" in fs:
            mug_cup.append(i["id"])
    parse_fa_ok += 1 if f["ok"] else 0
print(f"INFO non-test items={len(sel)} EN/FA target-type sets differ={len(diff_types)} (Mug->Cup={len(mug_cup)}) FA parse ok={parse_fa_ok}")
cov = load("docs/final-test/w4a_fa_coverage.json")
check("6 116 non-test items", len(sel) == 116 == len(nontest))
check("6 no Mug->Cup mislabel is left", len(mug_cup) == 0 and ITEM not in diff_types, mug_cup)
check("6 coverage matches the recorded report (FA parse ok 35 -> >= 56; same types 105 -> >= 112)",
      cov["before"]["fa_ok"] == 35 and cov["after"]["fa_ok"] == parse_fa_ok >= 56 and cov["after"]["same_types"] == 116 - len(diff_types) >= 112,
      (cov["after"]["fa_ok"], parse_fa_ok, len(diff_types)))
check("6 English parses are unchanged by the pass", cov["en_roles_changed_ids"] == [])
check("6 the report reads no outcome field", cov["outcome_fields_read"] == [])

print(f"\n{N - len(FAILS)}/{N} checks passed")
sys.exit(1 if FAILS else 0)
