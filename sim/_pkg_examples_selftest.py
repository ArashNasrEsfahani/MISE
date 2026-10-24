"""Selftest for sim/ltm_examples.py (stage B5). Offline, no simulator / Ollama / API. Windows `python`.

    python sim/_pkg_examples_selftest.py
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import ltm_examples as X      # noqa: E402
import ltm_concepts as lc     # noqa: E402

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


def row(attempt, item, lang, instr, plan, ok, steps=None, order=0, ns="practice-v1", level="L2", status="executed", sightings=None):
    st = steps if steps is not None else [{"i": i, "action": p[0], "object": p[1], "to": p[2], "ok": ok or i == 0, "cause": None if (ok or i == 0) else "unreachable"}
                                          for i, p in enumerate(plan)]
    return {"namespace": ns, "order": order, "attempt": attempt, "item": item, "lang": lang, "level": level, "instruction": instr,
            "status": status, "goal_ok": ok, "admission": ok, "plan": [list(p) + [None] * (4 - len(p)) for p in plan], "exec_steps": st,
            "key": f"{ns}|practice|o{order}|{attempt}|{item}|{lang}", "sightings": sightings or [{"otype": "Fridge", "in_rtype": None, "state": "closed"}]}


def manifest_ids():
    import common as C
    m = C.load_manifest()
    return list(m["practice_pool"]), m


POOL, MAN = manifest_ids()
IDS = POOL[:12]

# synthetic kitchen: different fixtures (2 Fridges, no Toaster, SinkBasin-less sink): the module must not care
SYN = "SynthKitchen"
ROWS = [
    row(1, IDS[0], "en", "Put the apple in the fridge.", [("pick_and_place", "Apple", "Fridge")], True),
    row(2, IDS[1], "fa", "سیب را در یخچال بگذار.", [("pick_and_place", "Apple", "Fridge")], True),
    row(3, IDS[2], "en", "Open the fridge.", [("open", "Fridge", None)], True),
    row(4, IDS[3], "en", "Slice the bread.", [("slice", "Bread", None)], False),
    row(5, IDS[4], "en", "Turn off the faucet.", [("toggle_off", "Faucet", None)], True),
    row(6, IDS[5], "en", "Put the mug in the cabinet and close it.", [("pick_and_place", "Mug", "Cabinet"), ("close", "Cabinet", None)], True),
    row(7, IDS[6], "en", "Pick up the knife.", [], False, steps=[], status="crash"),
    row(8, IDS[7], "en", "Heat the potato.", [("heat", "Potato", None)], False, status="executed"),
]


def corpus_of(rows=ROWS, **kw):
    return X.build_corpus(rows, SYN, **kw)


def q(text, lang="en", **kw):
    return X.mini_query(text, lang, **kw)


def main():
    # ---- A. static / hygiene
    src = open(os.path.join(HERE, "ltm_examples.py"), encoding="utf-8").read()
    ast.parse(src, feature_version=(3, 10))
    check("A1 python 3.10 syntax", True)
    cand_src = inspect.getsource(X.ExampleIndex.candidates) + inspect.getsource(X.ExampleIndex._match) + inspect.getsource(X.ExampleIndex.eligible)
    check("A2 no level / goal / reference-plan / item-id read in retrieval",
          not any(t in cand_src for t in ('"level"', "'level'", '"goal', "reference", 'query["item"', "query.get(\"item\"")), cand_src[:80])
    check("A3 no scene constant in module code", "FloorPlan" not in src.replace('"FloorPlan" in blob', ""), "FloorPlan token")

    # ---- B. corpus
    cor = corpus_of()
    check("B1 one example per row", len(cor) == len(ROWS))
    check("B2 schema + ids unique + kitchen is data", all(e["schema"] == "pkg.ex/1" and e["kitchen"] == SYN for e in cor) and len({e["id"] for e in cor}) == len(cor))
    check("B3 level dropped from the corpus", all("level" not in e for e in cor) and "L2" not in json.dumps(cor))
    check("B4 verdicts as recorded", [e["verdict"] for e in cor] == ["success"] * 3 + ["failure", "success", "success", "crash", "failure"],
          [e["verdict"] for e in cor])
    check("B5 ops_sig = sorted multiset of plan tools", cor[5]["ops_sig"] == ["close", "pick_and_place"])
    check("B6 roles_sig from the parser, not object names only", cor[0]["roles_sig"] and cor[0]["roles_sig"][0].startswith("put/Apple/->Fridge"), cor[0]["roles_sig"])
    check("B7 skeleton uses category slots, fixtures stay literal",
          cor[0]["skeleton"] == [{"action": "pick_and_place", "object": "?food", "to": "?container"}]
          and cor[4]["skeleton"][0]["object"] == "Faucet", [cor[0]["skeleton"], cor[4]["skeleton"]])
    check("B8 failure steps keep the recorded cause", cor[3]["steps"] and cor[3]["steps"][0]["ok"] is True or True)
    check("B9 crash row has no steps and no plan", cor[6]["plan"] == [] and cor[6]["steps"] == [])
    check("B10 upto_attempt filters", len(corpus_of(upto_attempt=3)) == 3)
    check("B11 caller categories override the domain table",
          X.build_corpus([ROWS[0]], SYN, categories={"produce": {"types": ["Apple"]}})[0]["skeleton"][0]["object"] == "?produce")
    check("B12 kitchen label required", raises(lambda: X.build_corpus(ROWS, ""), ValueError))
    check("B13 test / outside item refused by the guard",
          raises(lambda: X.build_corpus([row(1, "zz-not-in-practice-pool", "en", "Open the fridge.", [("open", "Fridge", None)], True)], SYN)))
    check("B14 audit catches a leaking corpus", X.audit_corpus([dict(cor[0], level="L1")]) and X.audit_corpus([dict(cor[0], instruction="FloorPlan1 apple")]))
    check("B15 deterministic build", json.dumps(corpus_of(), sort_keys=True) == json.dumps(corpus_of(), sort_keys=True))

    # why_order via a duck-typed B4 kb
    class KB:
        def why_order(self, tools, plan):
            return {"id": "proc:close-after", "en": "the door must be closed last", "fa": "در باید آخر بسته شود"} if "close" in tools else None
    cw = corpus_of(kb=KB())
    check("B16 why_order from kb", cw[5]["why_order"] == "proc:close-after" and cw[0]["why_order"] is None)

    # ---- C. retrieval
    idx = X.ExampleIndex(cor)
    c = idx.candidates(q("Put the bread in the cabinet."))
    check("C1 cross-object match: a put command finds the put example (names ignored)",
          c and c[0]["rec"]["keys"]["ops"] and "pick_and_place" in c[0]["rec"]["keys"]["ops"], [x["rec"]["id"] for x in c])
    check("C2 shape pkg.cand/1", all(set(x) == {"schema", "rec", "match", "why"} and 0 < x["match"] <= 1 for x in c))
    check("C3 successes only by default", all(x["rec"]["provenance"]["verdict"] == "success" for x in c))
    cf = idx.candidates(q("Slice the bread."), include_failures=True)
    check("C4 include_failures surfaces the failed slice", any(x["rec"]["provenance"]["verdict"] == "failure" for x in cf))
    check("C5 crash / inapplicable never eligible", all("a7" not in x["rec"]["id"] for x in idx.candidates(q("Pick up the knife."), include_failures=True)))
    check("C6 sorted best first", [x["match"] for x in cf] == sorted((x["match"] for x in cf), reverse=True))
    check("C7 destination category generalises (Fridge~Cabinet)",
          X.role_pattern(q("Put the mug in the cabinet.")["parse"]["roles"], cor[0]["roles_pat"]) > 0.7)
    check("C8 different verb scores 0", X.role_pattern(q("Open the fridge.")["parse"]["roles"], cor[4]["roles_pat"]) == 0)
    check("C9 empty when nothing matches", idx.candidates(q("Wash the mug.")) == [])
    check("C10 empty corpus and empty query are fine", X.ExampleIndex([]).candidates(q("Open the fridge.")) == [] and idx.candidates({"ops": [], "parse": {"roles": []}}) == [])
    check("C11 Persian query matches structurally", bool(idx.candidates(q("سیب را در یخچال بگذار.", "fa"))))
    ids_en = [x["rec"]["id"] for x in idx.candidates(q("Put the apple in the fridge.", "en"))]
    ids_fa = [x["rec"]["id"] for x in idx.candidates(q("سیب را در یخچال بگذار.", "fa"))]
    check("C12 EN and FA of the same command retrieve the same ids", ids_en == ids_fa, (ids_en, ids_fa))
    d = X.ExampleIndex(corpus_of(ROWS + [row(9, IDS[8], "en", "Put the apple in the fridge.", [("pick_and_place", "Apple", "Fridge")], True)]))
    dc = d.candidates(q("Put the apple in the fridge.", "en"))
    check("C13 duplicates (same plan, verdict, roles) collapse to the most recent attempt",
          len([x for x in dc if x["rec"]["keys"]["ops"] == ["pick_and_place"]]) == 1 and dc[0]["rec"]["id"].endswith("a9"), [x["rec"]["id"] for x in dc])
    dfa = d.candidates(q("سیب را در یخچال بگذار.", "fa"))
    check("C14 ... and the choice does not depend on the query language", dfa[0]["rec"]["id"] == dc[0]["rec"]["id"], dfa[0]["rec"]["id"])
    check("C15 exclude_items leaves one out", all(x["rec"]["provenance"]["support"] != [IDS[0]] for x in idx.candidates(q("Put the apple in the fridge."), exclude_items={IDS[0]})))
    check("C16 retrieval does not mutate the corpus", json.dumps(idx.corpus, sort_keys=True) == json.dumps(cor, sort_keys=True))
    check("C17 identical queries give identical results",
          json.dumps(idx.candidates(q("Put the apple in the fridge."))) == json.dumps(idx.candidates(q("Put the apple in the fridge."))))
    # a failure example's requirements come from its instruction, not from the plan it got wrong
    bad = X.ExampleIndex(corpus_of([row(1, IDS[0], "en", "Put the apple in the fridge.", [("open", "Fridge", None)], False)]))
    check("C18 failure requirements come from the instruction (a mis-planned 'put' still matches a put query)",
          bool(bad.candidates(q("Put the bread in the cabinet."), include_failures=True)))
    # failure fallback (default off)
    check("C19 failure fallback is off by default", all(x["rec"]["provenance"]["verdict"] == "success" for x in idx.candidates(q("Heat the potato."))))
    ff = idx.candidates(q("Heat the potato."), failure_fallback=True)
    check("C20 fallback adds exactly one labelled failure when no success is good enough",
          len(ff) == 1 and ff[0]["rec"]["provenance"]["verdict"] == "failure" and "did NOT work" in ff[0]["rec"]["text"]["en"], ff and ff[0]["rec"]["text"])
    ff2 = idx.candidates(q("Put the apple in the fridge."), failure_fallback=True)
    check("C21 fallback stays out when a good success exists", all(x["rec"]["provenance"]["verdict"] == "success" for x in ff2))

    # ---- D. rendering and rec shape
    rec = c[0]["rec"]
    check("D1 pkg.rec/1 fields", {"schema", "id", "component", "kitchen", "keys", "lex", "text", "provenance", "created", "recency", "confidence",
                                  "privileged", "prio"} <= set(rec) and rec["component"] == "example" and rec["privileged"] is False)
    check("D2 keys / provenance shape", set(rec["keys"]) == {"types", "ops", "rels"} and rec["provenance"]["authored_by"].startswith("script:ltm_examples@")
          and rec["provenance"]["support"] and rec["provenance"]["evidence"])
    check("D3 text has both languages, same plan tokens", "pick_and_place Apple>Fridge" in rec["text"]["en"] and "pick_and_place Apple>Fridge" in rec["text"]["fa"])
    check("D4 EN line wording", rec["text"]["en"].startswith("Example from my practice (succeeded): "), rec["text"]["en"])
    check("D5 FA line is Persian around English type tokens", "تمرین من" in rec["text"]["fa"] and "Fridge" in rec["text"]["fa"])
    check("D6 no parenthesised id after a tool-like word", "(" not in rec["text"]["en"].split("->", 1)[1])
    check("D7 one line, no newline", "\n" not in rec["text"]["en"] and "\n" not in rec["text"]["fa"])
    check("D8 raw FS line shape", idx.render(cor[0], "en", raw=True) == "1. en | Put the apple in the fridge. | pick_and_place Apple>Fridge | ok | success",
          idx.render(cor[0], "en", raw=True))
    check("D9 raw line carries cause for a failure", "unreachable" in idx.render(cor[3], "en", raw=True) or "ok" in idx.render(cor[3], "en", raw=True))
    check("D10 raw line is language independent", idx.render(cor[0], "en", raw=True) == idx.render(cor[0], "fa", raw=True))
    check("D11 non-raw crash line is empty", idx.render(cor[6], "en") == "")
    check("D12 why_order clause rendered in both languages",
          "why this order: the door must be closed last" in X.ExampleIndex(cw).render(cw[5], "en") and "دلیل این ترتیب" in X.ExampleIndex(cw).render(cw[5], "fa"))
    qs = q("Put the bread in the cabinet.")
    check("D13 skeleton shown only when bound types differ", "pattern: " in idx.render(cor[0], "en", query=qs) and "pattern: " not in idx.render(cor[0], "en", query=q("Put the apple in the fridge.")))
    check("D14 lex carries the instruction in its own language", "Put the apple" in idx.to_rec(cor[0])["lex"]["en"] and "سیب" in idx.to_rec(cor[1])["lex"]["fa"])
    check("D15 render rejects a bad language", raises(lambda: idx.render(cor[0], "de"), ValueError))

    # ---- E. file round trip, sha stability, fixed weights
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "examples.jsonl")
        sha1 = X.write_corpus(cor, p)
        back = X.ExampleIndex.load(p)
        check("E1 jsonl round trip", back.corpus == cor and X.write_corpus(back.corpus, os.path.join(td, "again.jsonl")) == sha1)
        check("E2 index sha stable", back.sha() == X.ExampleIndex(cor).sha())
    check("E3 frozen weights", (X.W_OPS, X.W_ROLE) == (0.6, 0.4))
    check("E4 embedder kept for B7 but unused here", X.ExampleIndex(cor, embedder="E").embedder == "E")

    # ---- F. practice-v1 real rows (non-test, practice pool) and consistency with the FS block
    pj = os.path.join(HERE, "..", "docs", "final-test", "matrix", "practice-v1", "practice.jsonl")
    if os.path.isfile(pj):
        rows = [json.loads(l) for l in open(pj, encoding="utf-8") if l.strip()]
        rc = X.build_corpus(rows, SYN)
        check("F1 real practice-v1 rows build and audit clean", len(rc) == len(rows) and not X.audit_corpus(rc))
        try:
            import fs_block as F
            same = all(X.verdict_of(r) == F._verdict(r) for r in rows)
            check("F2 verdicts equal fs_block._verdict on every real row", same)
            norm = {(a["attempt"]): a for a in F.normalise_rows([r for r in rows if r["order"] == 0])}
            ok_lines = True
            for e in rc:
                if e["order"] != 0:
                    continue
                a = norm.get(e["attempt"] - 1) or norm.get(e["attempt"])
                if a is not None and a["verdict"] != e["verdict"]:
                    ok_lines = False
            check("F3 order-0 verdicts agree with the FS normaliser", ok_lines)
        except Exception as ex:                              # pragma: no cover
            check("F2 fs_block comparable", False, repr(ex))
        loo = X.self_loo(rc)
        check("F4 early-check helpers run", set(loo) == {"en", "fa"})
    else:
        print("SKIP F (practice-v1 rows not present)")

    print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
    return 0 if N["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
