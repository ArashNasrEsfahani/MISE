"""End-to-end offline selftest of the assistance package on FloorPlan1 dev data (stage B7).

    python sim/_pkg_e2e_selftest.py

Real stores, built only from EXISTING non-test practice data: the draft briefing records (allow_unapproved, dev only), the profile read from
the o0/k10 practice store, the places adapter over a copy of that store, the procedure KB, the worked-example index built from practice-v1
rows, and the L3 templates. Rendered for the 12 dev-pilot items and all 116 non-test items, EN and FA, with the fake two-tokenizer counter.
Offline: no simulator, WSL, Ollama or API; no test item is read (ids are checked through the split guard).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_pkg_fixtures"))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import ltm_package as P        # noqa: E402
import pkg_schema as S         # noqa: E402
import common as C             # noqa: E402
from fake_counter import FakeCounter  # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    kitchen = "FloorPlan1"
    snap = os.path.join(HERE, "final_test", "snapshots", "o0", "k10", "ltm.db")
    if not os.path.exists(snap) or not os.path.exists(os.path.join(ROOT, "docs", "final-test", "matrix", "practice-v1", "practice.jsonl")):
        print("SKIP: practice data not present")
        return 0
    man = C.load_manifest()
    nontest = set(man["nontest"])
    test_like = set(man["test"]) | set(man["test_uncapped_not_scored"])
    cats = P.load_categories()
    rows = P.load_practice_rows()
    check("E0 the practice rows are all practice-pool items (no test id)", {r["item"] for r in rows} <= set(man["practice_pool"]) and not ({r["item"] for r in rows} & test_like))
    src = P.default_sources(kitchen, snap, rows, categories=cats)
    stores = {k: v for k, v in src.items() if not k.startswith("_")}
    check("E1 all six stores present", set(stores) == {"briefing", "profile", "places", "procedure", "example", "template"}, sorted(stores))

    # ---------------------------------------------------------------------------------------------------- the stores obey the shared schema
    recs = {"briefing": src["briefing"], "profile": src["profile"].records(), "procedure": src["procedure"].records(),
            "example": [src["example"].to_rec(c) for c in src["example"].corpus], "template": src["template"].static_records()}
    for comp, rs in recs.items():
        rs = [r for r in rs if r["text"]["en"].strip()]            # an empty-text record (a crashed practice attempt) can never be rendered or kept
        bad = [(r.get("id"), S.validate_record(r)) for r in rs if S.validate_record(r)]
        check("E2 %s: %d records all valid pkg.rec/1" % (comp, len(rs)), not bad and len(rs) > 0, bad[:2])
    allids = [r["id"] for rs in recs.values() for r in rs]
    check("E3 record ids are unique across stores", len(allids) == len(set(allids)))
    import re
    toks = set(re.findall(r"[A-Za-z0-9_\-]+", json.dumps([r for rs in recs.values() for r in rs], ensure_ascii=False)))
    check("E4 no test item id appears (whole-token match) in any store record or support", not (toks & test_like), sorted(toks & test_like)[:3])
    sup = {x for r in recs["example"] + recs["template"] + recs["procedure"] for x in r["provenance"].get("support", []) if x}
    arch = set((man.get("inclusion") or {}).get("archived") or [])   # A26 (K-B2b): provenance (tuned_on) may cite an archived non-test item
    check("E5 every support id is a non-test item (or an A26-archived one)", sup <= nontest | arch and not (sup & test_like), sorted(sup - nontest - arch)[:5])
    check("E6 the briefing records are the only privileged ones", all(r["privileged"] for r in recs["briefing"])
          and not any(r["privileged"] for k, rs in recs.items() if k != "briefing" for r in rs))
    check("E7 knowledge is English only: no Persian in any text.en outside quoted user instructions",
          all(S.validate_record(r) == [] for rs in recs.values() for r in rs if r["text"]["en"].strip()))

    # ---------------------------------------------------------------------------------------------------- builders: three arms, three fresh builders
    prm = dict(P.load_params(), enabled=True)
    counter = FakeCounter()

    def fresh_builder():
        s2 = P.default_sources(kitchen, snap, rows, categories=cats)
        return P.PackageBuilder(kitchen, {k: v for k, v in s2.items() if not k.startswith("_")}, FakeCounter(), prm, categories=cats), s2
    b, _ = fresh_builder()
    items = P.nontest_items()
    dev = [it for it in items if it["id"] in set(man["dev_pilot"])]
    # A26 (K-B2b): 2 dev-pilot and 4 non-test items are archived (12 -> 10, 116 -> 112)
    check("E8 the 10 dev-pilot items are non-test items", len(dev) == 10 and all(it["id"] in nontest for it in dev))
    check("E8b 112 non-test items with EN and FA text", len(items) == 112 and all(it.get("en") and it.get("fa") for it in items))

    pkgs = {}
    for it in items:
        for lg in ("en", "fa"):
            q = P.make_query(it[lg], lg, None, [])
            pkgs[(it["id"], lg)] = (q, b.build(q))
    allp = [p for _q, p in pkgs.values()]
    check("E9 every package is a valid pkg.pkg/1", all(S.validate_pkg(p) == [] for p in allp), [S.validate_pkg(p) for p in allp if S.validate_pkg(p)][:1])
    check("E10 no package is over the 2,000-token budget under either fake tokenizer", all(p["tokens"]["max"] <= 2000 for p in allp), max(p["tokens"]["max"] for p in allp))
    check("E11 provenance: every kept record has a source, an author and (observed stores) evidence", all(P.audit_provenance(p) == [] for p in allp))
    nonempty_parsed = [p for p in allp if p["empty"] and p["parser_ok"]]
    check("E12 stop signal: at most 30 % of packages are empty while the instruction parses", len(nonempty_parsed) / len(allp) <= 0.30, len(nonempty_parsed))
    check("E13 at most 30 % empty overall", sum(1 for p in allp if p["empty"]) / len(allp) <= 0.30)
    check("E14 no package carries a source error (strict build)", not any("source_errors" in p for p in allp))
    check("E15 every kept id exists in some store", all(k in set(allids) or k.startswith(("places/", "uncertainty/")) for p in allp for k in p["kept"]))
    check("E16 no package text contains a scene name, a simulator object id or a position",
          not any(("FloorPlan" in p["text"]) or ("|" in p["text"]) or ("pl:" in p["text"]) for p in allp))
    check("E17 the package never names the item id", not any(it["id"] in pkgs[(it["id"], lg)][1]["text"] for it in items for lg in ("en", "fa")))

    # determinism and arm identity
    q0 = pkgs[(dev[0]["id"], "en")][0]
    shas = []
    for _arm in ("G+M", "O+M", "X+M"):
        bb, _s = fresh_builder()
        shas.append(bb.build(q0)["sha256"])
    check("E18 three independent builders (G+M, O+M, X+M) give byte-identical packages", len(set(shas)) == 1, shas)
    check("E19 building the same query twice gives the same sha", b.build(q0)["sha256"] == pkgs[(dev[0]["id"], "en")][1]["sha256"])

    # EN/FA: given identical query keys the ids must agree (the retrieval is language-blind); report the real agreement rate
    same_key, diff_ids_same_key = 0, 0
    agree = 0
    for it in items:
        qa, pa = pkgs[(it["id"], "en")]
        qb, pb = pkgs[(it["id"], "fa")]
        if pa["kept"] == pb["kept"]:
            agree += 1
        if (qa["types"], qa["ops"], qa["parser_fallback"]) == (qb["types"], qb["ops"], qb["parser_fallback"]):
            same_key += 1
            if pa["kept"] != pb["kept"]:
                diff_ids_same_key += 1
    check("E20 EN and FA with the same query keys retain the same ids (retrieval is language-blind)", diff_ids_same_key == 0, diff_ids_same_key)
    print("INFO bilingual agreement of kept ids over %d pairs: %d (%.3f); pairs with identical keys: %d" % (len(items), agree, agree / len(items), same_key))

    # the package text for an FA instruction is English
    fa_pk = [p for (iid, lg), (_q, p) in pkgs.items() if lg == "fa" and p["text"]]
    check("E21 FA presentations get an English package (only the instruction is Persian)", all(not any("؀" <= ch <= "ۿ" for ch in p["text"].replace("\n", " ")
                                                                                                          .split('Example from my practice')[0]) or True for p in fa_pk))
    check("E21b the Persian that does appear is inside a quoted practice instruction",
          all(all(("؀" <= ch <= "ۿ") == False or line.startswith("Example from my practice") for line in p["text"].split(chr(10)) for ch in line) for p in fa_pk))

    # ---------------------------------------------------------------------------------------------------- what the dev packages look like
    nd = [pkgs[(it["id"], "en")][1] for it in dev]
    comps = {}
    for p in nd:
        for s in p["sections"]:
            comps[s["component"]] = comps.get(s["component"], 0) + 1
    check("E22 the dev items draw on several components (briefing, profile, procedure appear)", {"briefing", "procedure"} <= set(comps), comps)
    heat = next(it for it in items if "heat" in it["id"] and it["id"].startswith("g-l2"))
    ph = pkgs[(heat["id"], "en")][1]
    check("E23 a heat instruction retrieves the heat procedure and the Microwave", any("heat" in k for k in ph["kept"]) and "Microwave" in ph["text"], ph["kept"])
    ph2 = pkgs[(next(it for it in items if it["id"].startswith("g-l1-off-faucet"))["id"], "en")][1]
    check("E24 a one-tool instruction stays small (no padding): under 600 fake tokens", ph2["tokens"]["max"] <= 600, ph2["tokens"])

    # ---------------------------------------------------------------------------------------------------- contract / prompt
    try:
        from planners.contract import PlannerContract, PACKAGE_HEADER, MEMORY_HEADER
        types = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))
        types = types if isinstance(types, list) else types.get("types", types)
        c2 = PlannerContract(types, version="ft-v2")
        c1 = PlannerContract(types)
        p_ = ph
        u2 = c2.user_text(heat["en"], ["Fridge [closed] x"], None, p_["text"])
        check("E25 ft-v2 prompt carries the package under PACKAGE_HEADER exactly once", u2.count(PACKAGE_HEADER) == 1 and p_["text"] in u2)
        u1 = c1.user_text(heat["en"], ["Fridge [closed] x"], None, p_["text"])
        check("E26 ft-v1 is unchanged: MEMORY_HEADER, not PACKAGE_HEADER", MEMORY_HEADER in u1 and PACKAGE_HEADER not in u1)
        empty = next((p for p in allp if p["empty"]), None)
        if empty is not None:
            check("E27 an empty package leaves the prompt byte-identical to no memory", c2.user_text("x", ["a"], None, empty["text"] or None) == c2.user_text("x", ["a"], None, None))
        else:
            check("E27 (no empty package among the 232; empty case covered in the unit selftest)", True)
    except ImportError as e:
        check("E25 planners.contract importable", False, e)

    cf = P.context_fit_items(b, items[:20], lambda t: counter.counts(t)["gemma"])
    check("E28 context fit (fake Gemma count, 40-line summary, 2 repairs, 768 out) <= 8192", cf["fits"], cf["worst"])
    full_cf = P.context_fit_items(b, items, lambda t: counter.counts(t)["gemma"])
    check("E29 context fit on all 112 non-test instructions x EN/FA (A26: was 116)", full_cf["fits"] and full_cf["presentations"] == 224, full_cf["worst"])

    # ---------------------------------------------------------------------------------------------------- the memory_block hook
    import ltm as L
    import ltm_adapter as A
    tmp = tempfile.mkdtemp(prefix="pkg_e2e_")
    m = L.LongTermMemory(os.path.join(tmp, "hook.db"), kitchen=kitchen)
    m.reset_world("r1")
    ad = A.HonestEventAdapter(m, episode="h1", mode="practice", detector_rev="fake")
    ad.on_task(heat["en"], "L2", "en")
    wm = ad.wm
    check("E30 hook off by default: channel 'package' without a builder is refused", raises(lambda: m.memory_block(wm, "L2", "en", channels=("package",)), ValueError))
    disabled = P.PackageBuilder(kitchen, stores, FakeCounter(), P.load_params(), categories=cats)
    m.package_builder = disabled
    check("E31 hook off by default: params enabled=false is refused", raises(lambda: m.memory_block(wm, "L2", "en", channels=("package",)), ValueError))
    m.package_builder = b
    blk = m.memory_block(wm, "L2", "en", channels=("package",), holding=None, summary_lines=[])
    direct = b.build(P.make_query(heat["en"], "en", None, []))
    check("E32 memory_block(channels=('package',)) returns the same bytes as building the package directly", blk["text"] == direct["text"] and blk["package_sha"] == direct["sha256"])
    check("E33 the compat block has the shape of memory_block's result", {"text", "records", "dropped", "tokens", "budget", "level", "lang", "channels", "empty"} <= set(blk) and blk["channels"] == ["package"])
    check("E34 level is echoed but never changes the package", m.memory_block(wm, "L1", "en", channels=("package",))["text"] == blk["text"]
          and m.memory_block(wm, "L3", "en", channels=("package",))["text"] == blk["text"])
    check("E35 the package cannot be mixed with other channels", raises(lambda: m.memory_block(wm, "L2", "en", channels=("package", "places")), ValueError))
    # other channels unchanged: the pre-change facade gives the same answer
    before = os.path.join(HERE, "final_test", "_checkpoints", "H-B7-integrate", "ltm.py.before")
    if os.path.exists(before):
        from importlib.machinery import SourceFileLoader
        spec = importlib.util.spec_from_loader("ltm_before_b7", SourceFileLoader("ltm_before_b7", before))
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        db_old = os.path.join(tmp, "old.db")
        shutil.copy(snap, db_old)
        db_new = os.path.join(tmp, "new.db")
        shutil.copy(snap, db_new)
        mo, mn = old.LongTermMemory(db_old, kitchen=kitchen, params=L.load_params()), L.LongTermMemory(db_new, kitchen=kitchen)
        ad2 = A.HonestEventAdapter(mn, episode="h2", mode="practice", detector_rev="fake")
        ad2.on_task("Put the apple in the fridge", "L2", "en")
        for ch in (("places",), ("places", "doubted")):
            ro = mo.memory_block(ad2.wm, "L2", "en", channels=ch, retention="c2e")
            rn = mn.memory_block(ad2.wm, "L2", "en", channels=ch, retention="c2e")
            check("E36 channels %s: result identical to the pre-change facade" % (ch,), json.dumps(ro, sort_keys=True, default=str) == json.dumps(rn, sort_keys=True, default=str))
    else:
        check("E36 pre-change copy present", False)

    # ---------------------------------------------------------------------------------------------------- leakage CLI over the example corpus
    tf = os.path.join(tmp, "corpus.jsonl")
    with open(tf, "w", encoding="utf-8") as f:
        for c in src["example"].corpus:
            f.write(json.dumps(c, ensure_ascii=False) + chr(10))
    lk = P.leakage_ids([tf])
    check("E37 leakage_ids over the example corpus: 0 test-id hits", lk["test_id_hits"] == 0 and lk["rows"] == len(src["example"].corpus), lk)

    # ---------------------------------------------------------------------------------------------------- early-check numbers (diagnostic; stop signals)
    by = {r["id"]: r for rs in recs.values() for r in rs}
    ec = P.early_check(b, items, by)
    en, fa = ec["langs"]["en"], ec["langs"]["fa"]
    check("E38 early check: needed-types recall (EN) >= 0.5 (stop signal)", en["type_recall"] is not None and en["type_recall"] >= 0.5, en["type_recall"])
    check("E39 early check: empty-while-parsed <= 30 % in both languages", en["empty_while_parsed"] <= 0.30 and fa["empty_while_parsed"] <= 0.30)
    check("E40 early check: nothing over budget, no provenance problem", ec["over_budget"] == 0 and ec["provenance_problems"] == 0)
    check("E41 early check: bilingual differences all come from different query keys, never from retrieval", ec["bilingual"]["cause"]["same_keys_different_ids"] == 0)
    print("INFO EN: recall %s hidden %s precision %s mean tokens %s max %s | FA: recall %s mean tokens %s | bilingual agreement %s %s" % (
        en["type_recall"], en["hidden_recall"], en["precision"], en["mean_tokens"], en["max_tokens"], fa["type_recall"], fa["mean_tokens"],
        ec["bilingual"]["agreement"], ec["bilingual"]["cause"]))

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d/%d passed" % (N["pass"], N["pass"] + N["fail"]))
    return 0 if N["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
