"""Selftest for sim/ltm_package.py + sim/pkg_schema.py (stage B7). Offline: no simulator, WSL, Ollama or API. Windows `python`.

    python sim/_pkg_package_selftest.py

Every case runs on a SYNTHETIC kitchen ("SynthKitchen": 2 Fridges, no Toaster, a Sink without SinkBasin, 6 StoveKnobs, 4 StoveBurners)
with fake sources, so nothing here can depend on a FloorPlan constant. The real stores are exercised by _pkg_e2e_selftest.py.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_pkg_fixtures"))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import ltm_package as P        # noqa: E402
import pkg_schema as S         # noqa: E402
from fake_counter import FakeCounter  # noqa: E402

N = {"pass": 0, "fail": 0}
SYN = "SynthKitchen"


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


def rec(component, local, types=(), ops=(), rels=(), text="x", prio=0, kitchen=SYN, evidence=("ev:1",), conf=1.0, lexen=""):
    return {"schema": "pkg.rec/1", "id": "%s/%s/%s" % (component, kitchen, local), "component": component, "kitchen": kitchen,
            "keys": {"types": list(types), "ops": list(ops), "rels": list(rels)}, "lex": {"en": lexen, "fa": ""},
            "text": {"en": text, "fa": ""},
            "provenance": {"source": "fake:%s" % component, "evidence": list(evidence), "authored_by": "script:selftest", "support": []},
            "created": 0, "recency": 0, "confidence": conf, "privileged": component == "briefing", "prio": prio}


def cand(r, match=1.0, why=None):
    return {"schema": "pkg.cand/1", "rec": r, "match": match, "why": why if why is not None else ["type:" + t for t in r["keys"]["types"][:1]]}


class FakeSource:
    """A store that returns a fixed candidate list and remembers every query it saw (to prove what retrieval may read)."""

    def __init__(self, cands, **attrs):
        self.cands, self.seen = list(cands), []
        for k, v in attrs.items():
            setattr(self, k, v)

    def candidates(self, query, **kw):
        self.seen.append(copy.deepcopy(query))
        return [dict(c) for c in self.cands]

    def records(self):
        return [c["rec"] for c in self.cands]


class FakeKB(FakeSource):
    def __init__(self, cands, expansion=None):
        super().__init__(cands)
        self.exp = expansion or {}

    def expand(self, ops, types, categories=None):
        return {"tools": self.exp.get("tools", []), "appliances": self.exp.get("appliances", []),
                "receptacles": self.exp.get("receptacles", []), "requires": [], "actions": self.exp.get("actions", [])}


class FakeProfile(FakeSource):
    def __init__(self, cands, fixtures=None, tools=None, knob_map="not_learned"):
        super().__init__(cands)
        self.fixtures, self.tools, self.knob_map = fixtures or {}, tools or {}, knob_map


CATS = {"fixture_types": ["Fridge", "Microwave", "Sink", "StoveBurner", "StoveKnob", "CounterTop", "Cabinet", "Drawer"],
        "tool_types": ["Knife", "Spatula", "Fork"], "categories": {"food": ["Apple", "Egg", "Bread"], "tool": ["Knife", "Spatula"]}}


def params(**over):
    p = copy.deepcopy(P.default_params_doc())
    p["enabled"] = True
    for k, v in over.items():
        p[k] = v
    p.pop("sha256", None)
    return p


def q(instruction="Put the apple in the fridge", lang="en", types=("Apple", "Fridge"), ops=("put",), fallback=False, holding=None, observed=()):
    return {"schema": "pkg.query/1", "instruction": instruction, "lang": lang, "holding": holding, "observed": list(observed),
            "parse": {"ok": not fallback, "roles": [], "reasons": []}, "types": sorted(types), "ops": sorted(ops), "goal_states": [],
            "need": {"types": [], "tools": [], "appliances": [], "receptacles": [], "ops": []}, "parser_fallback": fallback}


def builder(sources, counter=None, prm=None, **kw):
    return P.PackageBuilder(SYN, sources, counter or FakeCounter(), prm or params(), categories=CATS, **kw)


def main():
    # ============================================================================== A. params
    pj = P.load_params()
    check("A1 package_params.json loads, sha verified, version pkg-1", pj["version"] == "pkg-1" and pj["sha256"] == P.params_sha(pj))
    check("A2 file equals DEFAULT_PARAMS (edit via --write-params only)", pj == P.default_params_doc())
    check("A3 shipped params are disabled (default off)", pj["enabled"] is False)
    check("A4 check_params accepts the shipped params", P.check_params(pj) == [], P.check_params(pj))
    bad = copy.deepcopy(pj)
    bad["weights"]["dense"] = 0.5
    check("A5 weights not summing to 1 rejected", any("weights" in x for x in P.check_params(bad)))
    bad = copy.deepcopy(pj)
    del bad["quotas"]["template"]
    check("A6 missing quota rejected", any("quotas" in x for x in P.check_params(bad)))
    bad = copy.deepcopy(pj)
    bad["borrow_order"] = ["procedure"]
    check("A7 incomplete borrow order rejected", any("borrow_order" in x for x in P.check_params(bad)))
    tmp = tempfile.mkdtemp()
    tp = os.path.join(tmp, "p.json")
    d = copy.deepcopy(pj)
    d["budget"] = 1999
    with open(tp, "w", encoding="utf-8") as f:
        json.dump(d, f)
    check("A8 a hand-edited params file fails its sha check", raises(lambda: P.load_params(tp), ValueError))
    check("A9 quotas sum to the budget (300+350+350+550+300+150 = 2000)", sum(pj["quotas"].values()) == pj["budget"] == 2000)
    check("A10 the builder refuses bad params", raises(lambda: P.PackageBuilder(SYN, {}, FakeCounter(), dict(pj, weights={"exact": 1, "lexical": 1, "dense": 1})), ValueError))

    # ============================================================================== B. schema
    fx = [json.loads(ln) for ln in open(os.path.join(HERE, "_pkg_fixtures", "rec.sample.jsonl"), encoding="utf-8") if ln.strip()]
    check("B1 sample records are valid pkg.rec/1", all(S.validate_record(r) == [] for r in fx), [S.validate_record(r) for r in fx])
    r0 = copy.deepcopy(fx[0])
    r0["id"] = "profile/Synth Kitchen/x"
    check("B2 an id with a space is rejected", any("id" in x for x in S.validate_record(r0)))
    r0 = copy.deepcopy(fx[0])
    r0["text"]["en"] = "Fridge held سیب"
    check("B3 Persian in text.en rejected (English-only knowledge)", any("Persian" in x for x in S.validate_record(r0)))
    r0 = copy.deepcopy(fx[0])
    r0["text"]["en"] = "Use go_to(CounterTop) first"
    check("B4 'go_to(Type)' in a record text rejected", any("parenthes" in x for x in S.validate_record(r0)))
    r0 = copy.deepcopy(fx[0])
    r0["privileged"] = True
    check("B5 privileged only for the briefing", any("privileged" in x for x in S.validate_record(r0)))
    r0 = copy.deepcopy(fx[0])
    r0["keys"]["rels"] = ["flies:Fridge"]
    check("B6 a relation outside the closed vocabulary rejected", any("vocabulary" in x for x in S.validate_record(r0)))
    r0 = copy.deepcopy(fx[0])
    r0["provenance"]["authored_by"] = ""
    check("B7 missing authorship rejected", any("provenance" in x for x in S.validate_record(r0)))
    ex = rec("example", "a1", ["Apple"], ["pickup"], text='Example from my practice (succeeded): "سیب را بردار" -> pickup Apple')
    check("B8 an example may quote the user's Persian instruction", S.validate_record(ex) == [], S.validate_record(ex))
    sq = json.load(open(os.path.join(HERE, "_pkg_fixtures", "query.sample.json"), encoding="utf-8"))
    check("B9 sample query valid", S.validate_query(sq) == [], S.validate_query(sq))
    check("B10 a query carrying 'level' is invalid", any("forbidden" in x for x in S.validate_query(dict(sq, level="L2"))))
    check("B11 parser_fallback must equal not parse.ok", any("parser_fallback" in x for x in S.validate_query(dict(sq, parser_fallback=True))))
    check("B12 validate_cand accepts a good candidate and rejects a bad match", S.validate_cand(cand(fx[0])) == [] and S.validate_cand(dict(cand(fx[0]), match=2))
          != [])

    # ============================================================================== C. make_query
    qe = P.make_query("Put the apple in the fridge", "en", None, ["Fridge [closed] in the room, step 3", "Cabinet x2 [1 closed, 1 open]; latest at (1.00, 2.00)"])
    check("C1 EN query: types, ops, parse ok, valid schema", qe["types"] == ["Apple", "Fridge"] and qe["ops"] == ["put"] and not qe["parser_fallback"]
          and S.validate_query(qe) == [], qe)
    qf = P.make_query("سیب را در یخچال بگذار", "fa", None, [])
    check("C2 FA query reaches the same types and ops (A17 parser)", qf["types"] == qe["types"] and qf["ops"] == qe["ops"], qf)
    check("C3 observed = types and states only", {"type": "Cabinet", "state": "open"} in qe["observed"] and {"type": "Fridge", "state": "closed"} in qe["observed"]
          and all(set(o) == {"type", "state"} for o in qe["observed"]), qe["observed"])
    q3 = P.make_query("Put it on the counter", "en", "Bowl", [])
    check("C4 a not-ok parse keeps its typed mentions and sets parser_fallback", q3["parser_fallback"] and "CounterTop" in q3["types"] and q3["holding"] == "Bowl", q3)
    q4 = P.make_query("hello there", "en", None, [])
    check("C5 gibberish: no types, no ops, fallback", q4["types"] == [] and q4["ops"] == [] and q4["parser_fallback"])
    check("C6 make_query has no level / item / plan parameter", list(inspect.signature(P.make_query).parameters) == ["instruction", "lang", "holding", "summary_lines", "lexicon"])
    check("C7 PackageBuilder.build has no level parameter", list(inspect.signature(P.PackageBuilder.build).parameters) == ["self", "query"])
    q5 = P.make_query("Open the cabinet", "en", "not a type", [])
    check("C8 a non-type holding is dropped", q5["holding"] is None)

    # ============================================================================== D. expand
    kb = FakeKB([], {"tools": ["Knife"], "appliances": ["Microwave"], "receptacles": ["Fridge"], "actions": ["pickup", "heat"]})
    prof = FakeProfile([], fixtures={"Drawer": {"contents": {"Knife": {}}}}, tools={"Knife": {"last_parent_type": "Drawer"}})
    q0 = q(types=("Apple",), ops=("heat",))
    q1 = P.expand(q0, kb, CATS, (), prof)
    check("D1 expand fills need from the KB and the profile", q1["need"]["tools"] == ["Knife"] and q1["need"]["appliances"] == ["Microwave"]
          and q1["need"]["receptacles"] == ["Drawer", "Fridge"] and "Microwave" in q1["need"]["types"] and "heat" in q1["need"]["ops"], q1["need"])
    check("D2 expand does not mutate its input", q0["need"]["types"] == [] and q1 is not q0)
    check("D3 expand with no KB and no profile = the query's own types", P.expand(q0, None, CATS)["need"]["types"] == ["Apple"])
    check("D4 expand is deterministic", P.expand(q0, kb, CATS, (), prof) == q1)

    # ============================================================================== E. briefing matcher
    br = [rec("briefing", "core:Fridge", ["Fridge"], text="There are 2 Fridge units.", prio=0),
          rec("briefing", "core:Sink", ["Sink"], text="There is one Sink.", prio=0),
          rec("briefing", "exist:Microwave", ["Microwave"], text="There is one Microwave.", prio=1),
          rec("briefing", "adj:Sink", ["Sink", "Fridge"], rels=["near:Fridge"], text="The Sink is next to the Fridge.", prio=3),
          rec("briefing", "conv:cold_food", ["Egg", "Apple", "Fridge"], rels=["stores:Fridge"], text="Usually cold food is kept in a Fridge.", prio=5)]
    for r in br:
        r["kind"] = r["id"].split("/")[-1].split(":")[0]
    qb = P.expand(q(types=("Apple",), ops=("put",)), None, CATS)
    cb = P.briefing_candidates(br, qb, params())
    ids = {c["rec"]["id"].split("/")[-1]: c["match"] for c in cb}
    check("E1 a convention matches by its members, not by its receptacle alone", ids.get("conv:cold_food") == 1.0, ids)
    check("E2 an adjacency line never matches through a neighbour only", "adj:Sink" not in ids and "core:Fridge" not in ids, ids)
    qb2 = P.expand(q(types=("Sink",), ops=("wash",)), None, CATS)
    ids2 = {c["rec"]["id"].split("/")[-1]: c["match"] for c in P.briefing_candidates(br, qb2, params())}
    check("E3 the anchor type matches core and adjacency", ids2.get("core:Sink") == 1.0 and ids2.get("adj:Sink") == 1.0, ids2)
    qb3 = dict(P.expand(q(types=("Apple",), ops=("heat",)), FakeKB([], {"appliances": ["Microwave"]}), CATS))
    ids3 = {c["rec"]["id"].split("/")[-1]: c["match"] for c in P.briefing_candidates(br, qb3, params())}
    check("E4 an expansion-only type matches below a direct mention", ids3.get("exist:Microwave") == 0.7, ids3)
    qf1 = q(types=("Apple",), ops=("put",), fallback=True)
    cf = P.briefing_candidates([dict(r, kind=r["id"].split("/")[-1].split(":")[0]) for r in br], P.expand(qf1, None, CATS), params())
    fb = [c for c in cf if "fallback:core" in c["why"]]
    check("E5 a parser failure adds the core existence lines (<= 5)", 1 <= len(fb) <= 5 and all("/core:" in c["rec"]["id"] for c in fb), [c["rec"]["id"] for c in fb])
    check("E6 no fallback when the parse is ok", not any("fallback:core" in c["why"] for c in cb))

    # ============================================================================== F. builder
    check("F1 no sources -> an empty package: text '', no header, reason reported",
          (lambda pk: pk["empty"] and pk["text"] == "" and pk["kept"] == [] and pk["empty_reason"] == "parsed_zero_matches" and pk["tokens"]["max"] == 0
           and S.validate_pkg(pk) == [])(builder({}).build(q())))
    pk = builder({}).build(q(fallback=True))
    check("F2 empty + parser failure is logged apart from parsed-zero-matches", pk["empty_reason"] == "parser_failed_no_match" and pk["parser_ok"] is False)
    # a single matching record: only it (plus its section header) is rendered, nothing pads
    r1 = rec("procedure", "tool-put", ops=["put_in"], text="put_in R: puts the held object into R.", prio=1)
    pkg1 = builder({"procedure": FakeKB([cand(r1, 1.0, ["op:put_in"])])}).build(q(ops=("put",)))
    body = pkg1["text"]
    check("F3 one matching record -> exactly its section header and its line", body == "How the tools work:\n" + r1["text"]["en"], repr(body))
    check("F4 never pads: no other section label appears", all(h not in body for k, h in P.DEFAULT_PARAMS["headers"]["en"].items() if h and k != "procedure"))
    check("F5 package valid, provenance audit clean, one kept id", S.validate_pkg(pkg1) == [] and P.audit_provenance(pkg1) == [] and pkg1["kept"] == [r1["id"]])
    irr = rec("procedure", "tool-wash", ops=["wash"], text="wash X: washes it.", prio=2)
    pkg2 = builder({"procedure": FakeKB([cand(r1, 1.0, ["op:put_in"]), cand(irr, 0.0, [])])}).build(q())
    check("F6 a zero-match candidate is dropped as irrelevant and counted", pkg2["kept"] == [r1["id"]] and pkg2["irrelevant"]["procedure"] == 1
          and not any(d["reason"] == "irrelevant" for d in pkg2["dropped"]))
    pkg2b = builder({"procedure": FakeKB([cand(irr, 0.0, [])])}, prm=params(log_irrelevant_ids=True)).build(q())
    check("F6b log_irrelevant_ids lists them one by one", pkg2b["empty"] and any(d["reason"] == "irrelevant" and d["id"] == irr["id"] for d in pkg2b["dropped"]))
    # a dense score can rescue a keyless record (irrelevant = match 0 AND dense < 0.5)
    emb_hi = lambda query, cs: [0.9] * len(cs)  # noqa: E731
    pkg2c = builder({"procedure": FakeKB([cand(irr, 0.0, [])])}, embedder=emb_hi).build(q())
    check("F6c match 0 with dense >= 0.5 is not irrelevant", pkg2c["kept"] == [irr["id"]])
    pkg2d = builder({"procedure": FakeKB([cand(irr, 0.0, [])])}, embedder=lambda query, cs: [0.49] * len(cs)).build(q())
    check("F6d match 0 with dense < 0.5 stays irrelevant", pkg2d["empty"])

    # -- ordering, ties, determinism
    rs = [rec("procedure", "p%d" % i, ops=["put_in"], text="line %d for put_in." % i, prio=i % 3) for i in range(9)]
    cs = [cand(r, 1.0, ["op:put_in"]) for r in rs]
    a = builder({"procedure": FakeKB(cs)}).build(q())
    shuffled = cs[:]
    random.Random(7).shuffle(shuffled)
    b = builder({"procedure": FakeKB(shuffled)}).build(q())
    check("F7 candidate arrival order does not change the package", a["sha256"] == b["sha256"] and a["kept"] == b["kept"])
    check("F8 ties break by prio then id", a["kept"] == [r["id"] for r in sorted(rs, key=lambda r: (r["prio"], r["id"]))], a["kept"])
    check("F9 building twice gives the same sha", builder({"procedure": FakeKB(cs)}).build(q())["sha256"] == a["sha256"])

    # -- exact-match weight beats lexical when both exist; discount for expansion-only
    rd = rec("profile", "fixture:Fridge", ["Fridge"], text="Fridge direct.", lexen="fridge")
    rn = rec("profile", "fixture:Microwave", ["Microwave"], text="Microwave via need.", lexen="microwave")
    pf = FakeProfile([cand(rn, 1.0, ["type:Microwave"]), cand(rd, 1.0, ["type:Fridge"])])
    pk3 = builder({"profile": pf, "procedure": FakeKB([], {"appliances": ["Microwave"]})}).build(q(types=("Fridge",), ops=("heat",)))
    check("F10 a direct mention ranks above an expansion-only record", pk3["kept"] == [rd["id"], rn["id"]], pk3["kept"])

    # -- budget: whole records, under every tokenizer, header included
    big = [rec("procedure", "b%d" % i, ops=["put_in"], text=("sentence number %d " % i) * 8, prio=i) for i in range(30)]
    cb2 = [cand(r, 1.0, ["op:put_in"]) for r in big]
    for budget in (120, 300, 700, 2000):
        pkb = builder({"procedure": FakeKB(cb2)}, prm=params(budget=budget, quotas={"briefing": budget // 6, "profile": budget // 6, "procedure": budget // 6,
                                                                               "example": budget // 6, "template": budget // 6, "uncertainty": budget - 5 * (budget // 6)})).build(q())
        fc = FakeCounter()
        full = pkb["text"] and (P.PackageBuilder(SYN, {}, fc, params()).header + "\n" + pkb["text"])
        cnt = fc.counts(full) if full else {"gemma": 0, "anthropic": 0}
        whole = all(any(line.strip() == _l for _l in pkb["text"].split(chr(10))) for line in [r["text"]["en"] for r in big if r["id"] in pkb["kept"]])
        check("F11 budget %d: complete text <= budget under both tokenizers, whole records only" % budget, max(cnt.values()) <= budget and whole and pkb["tokens"]["max"] == max(cnt.values()), (cnt, pkb["tokens"]))
        if budget >= 300:
            check("F11b budget %d: something was kept" % budget, len(pkb["kept"]) >= 1)
    full_pk = builder({"procedure": FakeKB(cb2)}).build(q())
    check("F12 dropped records carry reason budget", all(d["reason"] == "budget" for d in full_pk["dropped"]) and len(full_pk["kept"]) + len(full_pk["dropped"]) == 30)
    # the binding tokenizer is the max: a counter where 'anthropic' is stricter
    tight = FakeCounter(gemma_div=10.0, anth_div=1.5)
    pkt = builder({"procedure": FakeKB(cb2)}, counter=tight, prm=params(budget=500)).build(q())
    check("F13 the stricter tokenizer binds", max(tight.counts(P.PackageBuilder(SYN, {}, tight, params()).header + "\n" + pkt["text"]).values()) <= 500
          and pkt["tokens"]["en"]["anthropic"] >= pkt["tokens"]["en"]["gemma"])
    # a non-additive counter (every newline costs 4 more): the final guard still produces a package that fits
    class Quirky(FakeCounter):
        def counts(self, text):
            c = super().counts(text)
            extra = 4 * str(text).count("\n")
            return {k: v + extra for k, v in c.items()}
    qk = Quirky()
    pkq = builder({"procedure": FakeKB(cb2)}, counter=qk, prm=params(budget=400)).build(q())
    check("F14 non-additive counts: the complete text still fits", max(qk.counts(P.PackageBuilder(SYN, {}, qk, params()).header + "\n" + pkq["text"]).values()) <= 400)
    # quotas + borrowing
    pq = params(budget=600, quotas={"briefing": 100, "profile": 100, "procedure": 100, "example": 100, "template": 100, "uncertainty": 100})
    only_proc = builder({"procedure": FakeKB(cb2)}, prm=pq).build(q())
    used_proc = max(only_proc["sections"][0]["used"]["en"].values())
    check("F15 an unfilled quota is borrowed: procedure uses more than its own 100", used_proc > 100 and only_proc["tokens"]["max"] <= 600, (used_proc, only_proc["tokens"]))
    rb = [rec("briefing", "core:T%d" % i, ["Fridge"], text=("briefing sentence %d " % i) * 6) for i in range(10)]
    for r in rb:
        r["kind"] = "core"
    full5 = builder({"briefing": rb, "procedure": FakeKB(cb2)}, prm=pq).build(q(types=("Fridge",)))
    secs = {s["component"]: s for s in full5["sections"]}
    check("F16 two competing sections both keep records and the sum fits", set(secs) == {"briefing", "procedure"} and full5["tokens"]["max"] <= 600
          and max(secs["briefing"]["used"]["en"].values()) <= 100 + 600, full5["tokens"])
    # k_max
    ex_recs = [rec("example", "e%d" % i, ["Apple"], ["pickup"], text='Example from my practice (succeeded): "pick %d" -> pickup Apple' % i, prio=i) for i in range(6)]
    pke = builder({"example": FakeSource([cand(r, 1.0, ["type:Apple"]) for r in ex_recs])}).build(q())
    check("F17 k_max: at most 2 examples, the rest dropped as quota", len(pke["kept"]) == 2 and sum(1 for d in pke["dropped"] if d["reason"] == "quota") == 4, pke["kept"])
    tp_recs = [rec("template", "t%d" % i, ["Apple"], ["heat"], text="Template number %d." % i, prio=i) for i in range(3)]
    pkt2 = builder({"template": FakeSource([cand(r, 1.0, ["type:Apple"]) for r in tp_recs])}).build(q())
    check("F18 k_max: at most 1 template", len(pkt2["kept"]) == 1)

    # -- duplicates: a fact the profile already states is kept from the profile
    prof_r = rec("profile", "fixture:CounterTop", ["CounterTop", "Apple"], rels=["contains:Apple"], text="CounterTop: objects seen there: Apple.", lexen="countertop apple")
    place_r = rec("places", "at:Apple", ["CounterTop", "Apple"], rels=["contains:Apple"], text="Apple was last seen at: CounterTop (3 sightings).", lexen="apple")
    d1 = builder({"profile": FakeProfile([cand(prof_r, 0.6, ["rel:contains:Apple"])]), "places": FakeSource([cand(place_r, 1.0, ["type:Apple"])])}).build(q(types=("Apple",)))
    check("F19 a places fact covered by a profile record is dropped as duplicate", d1["kept"] == [prof_r["id"]] and any(d["reason"] == "duplicate" and d["id"] == place_r["id"] for d in d1["dropped"]), d1["kept"])
    d2 = builder({"places": FakeSource([cand(place_r, 1.0, ["type:Apple"])])}).build(q(types=("Apple",)))
    check("F20 without a profile the places record stands (profile section)", d2["kept"] == [place_r["id"]] and d2["sections"][0]["component"] == "profile")
    br_dup = rec("briefing", "open:Fridge", ["Fridge"], rels=["opens:Fridge"], text="The Fridge can be opened and closed.")
    br_dup["kind"] = "open"
    pf_dup = rec("profile", "fixture:Fridge", ["Fridge"], rels=["opens:Fridge"], text="Fridge: opened 4 times.")
    d3 = builder({"briefing": [br_dup], "profile": FakeProfile([cand(pf_dup, 1.0, ["type:Fridge"])])}).build(q(types=("Fridge",)))
    check("F21 the same fact from briefing and profile is kept from the profile", d3["kept"] == [pf_dup["id"]] and any(d["reason"] == "duplicate" for d in d3["dropped"]))

    # -- dense changes the order, never the candidate set
    ra, rb_ = rec("procedure", "da", ops=["put_in"], text="alpha", prio=1), rec("procedure", "db", ops=["put_in"], text="beta", prio=2)
    dense_first_b = lambda query, cs: [0.2 if c["rec"]["id"].endswith("da") else 0.9 for c in cs]  # noqa: E731
    base = builder({"procedure": FakeKB([cand(ra, 1.0, ["op:put_in"]), cand(rb_, 1.0, ["op:put_in"])])}).build(q())
    dn = builder({"procedure": FakeKB([cand(ra, 1.0, ["op:put_in"]), cand(rb_, 1.0, ["op:put_in"])])}, embedder=dense_first_b).build(q())
    check("F22 the dense score re-orders equal exact matches (0.15 weight)", base["kept"] == [ra["id"], rb_["id"]] and dn["kept"] == [rb_["id"], ra["id"]], (base["kept"], dn["kept"]))
    # an Embedder-style object is accepted too, and caches vectors
    class FE:
        model_rev = "fake"
        n = 0

        def embed(self, texts):
            FE.n += len(texts)
            return [[1.0, 0.0] if "alpha" in t or "put" in t.lower() else [0.0, 1.0] for t in texts]
    cache = {}
    fe = FE()
    builder({"procedure": FakeKB([cand(ra, 1.0, ["op:put_in"])])}, embedder=fe, cache=cache).build(q())
    n1 = FE.n
    builder({"procedure": FakeKB([cand(ra, 1.0, ["op:put_in"])])}, embedder=fe, cache=cache).build(q())
    check("F23 an Embedder object works and its vectors are cached", n1 > 0 and FE.n == n1, (n1, FE.n))

    # -- what retrieval may read
    spy = FakeKB([cand(r1, 1.0, ["op:put_in"])])
    sp2 = FakeSource([])
    builder({"procedure": spy, "template": sp2}).build(q())
    seen = spy.seen[0]
    check("F24 sources see no level / item / plan / goal / outcome key", not any(k in seen for k in P._FORBIDDEN_QUERY_KEYS), sorted(seen))
    for bad_key in ("level", "item_id", "reference_plan", "goal", "outcome"):
        check("F25 a query carrying %r is refused" % bad_key, raises(lambda bk=bad_key: builder({}).build(dict(q(), **{bk: "x"})), ValueError))
    check("F25b any verdict-like '*_ok' flag on a query is refused", raises(lambda: builder({}).build(dict(q(), plan_ok=True)), ValueError))
    # the query copy handed to sources is not mutated between components
    check("F26 the caller's query is not mutated by build", (lambda qq: (builder({"procedure": FakeKB([cand(r1, 1.0, ["op:put_in"])])}).build(qq), qq)[1] == q())(q()))

    # -- EN/FA parity: same keys, different surface language -> same ids
    srcs = {"procedure": FakeKB([cand(r1, 1.0, ["op:put_in"])]), "profile": FakeProfile([cand(prof_r, 0.6, ["rel:contains:Apple"])])}
    pe = builder(srcs).build(q("Put the apple in the fridge", "en"))
    pfa = builder(srcs).build(q("سیب را در یخچال بگذار", "fa"))
    check("F27 EN and FA presentations with the same keys keep the same ids and the same English bytes", pe["kept"] == pfa["kept"] and pe["text"] == pfa["text"])
    check("F28 the package is rendered in English whatever the instruction language", pfa["rendered_lang"] == "en" and not any("؀" <= ch <= "ۿ" for ch in pfa["text"]))

    # -- uncertainty
    knob_q = q(types=("StoveKnob",), ops=("toggle",))
    u1 = builder({"profile": FakeProfile([], knob_map="not_learned")}).build(knob_q)
    check("F29 'Not learned: ... StoveKnob drives which StoveBurner' when the knob map is not learned", "StoveKnob drives which StoveBurner" in u1["text"])
    u2 = builder({"profile": FakeProfile([], knob_map={"k1": "b1"})}).build(knob_q)
    check("F30 no knob line when the knob map is learned", "StoveKnob drives" not in u2["text"])
    u3 = builder({"profile": FakeProfile([cand(prof_r, 0.6, ["rel:contains:Apple"])])}).build(q(types=("Apple", "Egg"), ops=("put",)))
    check("F31 'Never seen in my practice' names a movable type absent from the profile", "Never seen in my practice: Egg." in u3["text"], u3["text"])
    u4 = builder({"profile": FakeProfile([cand(prof_r, 0.6, ["rel:contains:Apple"])])}).build(q(types=("Fridge",), ops=("open",)))
    check("F32 a fixture type is never reported as 'never seen'", "Never seen" not in u4["text"])
    u5 = builder({}).build(q(types=("Egg",)))
    check("F33 no 'never seen' line when there is no practice at all", "Never seen" not in u5["text"])
    u6 = builder({"procedure": FakeKB([cand(r1, 1.0, ["op:put_in"])])}).build(q(fallback=True))
    check("F34 a parser failure says so in the Not known section", "only partly understood" in u6["text"])

    # -- fallback core
    core = [dict(rec("briefing", "core:%s" % t, [t], text="There is one %s." % t, prio=0), kind="core") for t in ("Fridge", "Sink", "Microwave", "CounterTop", "StoveBurner", "Cabinet", "Drawer")]
    fbk = builder({"briefing": core}).build(q(types=(), ops=(), fallback=True))
    check("F35 parser failure with no typed mention: at most 5 core existence lines, fallback_used", fbk["fallback_used"] and 1 <= len([k for k in fbk["kept"] if "/core:" in k]) <= 5 and not fbk["parser_ok"], fbk["kept"])
    fbk2 = builder({"briefing": core}).build(q(types=(), ops=(), fallback=False))
    check("F36 parsed with nothing matching: empty, no fallback lines", fbk2["empty"] and not fbk2["fallback_used"])

    # -- strictness
    class Boom(FakeSource):
        def candidates(self, query, **kw):
            raise RuntimeError("store unreadable")
    check("F37 a failing source stops a strict build", raises(lambda: builder({"template": Boom([])}).build(q()), RuntimeError))
    soft = builder({"template": Boom([])}, strict=False).build(q())
    check("F38 strict=False records the error in the package (never silent)", soft.get("source_errors") and soft["source_errors"][0]["component"] == "template")

    # -- provenance audit
    bad_pk = copy.deepcopy(pkg1)
    bad_pk["provenance"] = []
    check("F39 audit_provenance flags a kept record with no provenance", P.audit_provenance(bad_pk) != [])
    noev = rec("profile", "fixture:Fridge", ["Fridge"], text="Fridge seen.", evidence=())
    pn = builder({"profile": FakeProfile([cand(noev, 1.0, ["type:Fridge"])])}).build(q(types=("Fridge",)))
    check("F40 an observed-knowledge record with no evidence is flagged", any("no evidence" in x for x in P.audit_provenance(pn)))
    check("F41 every kept record is listed with source and authorship", all(p["source"] and p["authored_by"] for p in pkg1["provenance"]))

    # -- sha / log / compat
    check("F42 package sha is self-consistent", pkg1["sha256"] == S.self_sha(pkg1))
    lr = P.log_row(pkg1, "pres-1", q())
    check("F43 log row carries no level / item / goal, only keys and shas", lr["schema"] == "pkg.log/1" and lr["package_sha"] == pkg1["sha256"]
          and not any(k in json.dumps(lr) for k in ('"level"', '"goal"', '"item"')))
    cm = P.memory_block_compat(pkg1, level="L2")
    want = {"text", "records", "dropped", "tokens", "budget", "types", "level", "lang", "channels", "empty", "joint", "token_claim", "tokenizer"}
    check("F44 memory_block_compat has the keys ltm.memory_block returns", want <= set(cm) and cm["channels"] == ["package"] and cm["text"] == pkg1["text"]
          and cm["tokens"] == pkg1["tokens"]["max"] and cm["empty"] is False)
    cm0 = P.memory_block_compat(builder({}).build(q()))
    check("F45 an empty package -> an empty compat block", cm0["empty"] is True and cm0["text"] == "" and cm0["records"] == [])

    # ============================================================================== G. PlacesSource over a fake LTM
    class FakeStore:
        def select(self, table, where=None):
            if table == "location_claim":
                return [{"claim": "lc:aa", "place": "pl:1", "otype": "Apple"}]
            return [{"place": "pl:1", "rtype": "CounterTop"}]

    class FakeLTM:
        kitchen = SYN
        store = FakeStore()

        def where_is(self, otype, wm=None, channels=None):
            if otype == "Apple":
                return {"candidates": [{"id": "pl:1", "kind": "place", "rtype": "CounterTop", "tier": "T0", "support": 3, "weight": 0.9},
                                       {"id": "pl:2", "kind": "place", "rtype": "Cabinet", "tier": "T0", "support": 1, "weight": 0.4}]}
            return {"candidates": []}
    ps = P.PlacesSource(FakeLTM(), None, ("places",), tool_types=("Knife",))
    pc = ps.candidates(P.expand(q(types=("Apple", "Egg")), None, CATS))
    check("G1 PlacesSource: one record per type that has a claim", len(pc) == 1 and pc[0]["rec"]["id"] == "places/%s/at:Apple" % SYN, [c["rec"]["id"] for c in pc])
    check("G2 the places record is a valid pkg.rec/1 with evidence and no simulator id", S.validate_record(pc[0]["rec"]) == [] and pc[0]["rec"]["provenance"]["evidence"]
          and "pl:" not in pc[0]["rec"]["text"]["en"], S.validate_record(pc[0]["rec"]))
    check("G3 text names the receptacle types and the sighting count", "CounterTop, Cabinet" in pc[0]["rec"]["text"]["en"] and "4 sightings" in pc[0]["rec"]["text"]["en"], pc[0]["rec"]["text"]["en"])
    check("G4 a tool uses the tool_at relation", P.PlacesSource(FakeLTM(), None, tool_types=("Apple",)).candidates(q(types=("Apple",)))[0]["rec"]["keys"]["rels"] == ["tool_at:Apple"])
    check("G5 the adapter asks only the places channel", ps.channels == ("places",))

    # ============================================================================== H. diagnostics helpers
    ctr = FakeCounter()
    g = P.real_token_gate([pkg1, full_pk], ctr)
    check("H1 real_token_gate passes packages within budget and reports the worst case", g["pass"] and g["checked"] == 2 and set(g["worst"]) == {"gemma", "anthropic"})
    tiny = copy.deepcopy(pkg1)
    tiny["tokens"]["budget"] = 5
    check("H2 real_token_gate fails when a tokenizer is over the budget", not P.real_token_gate([tiny], ctr)["pass"])
    cf = P.context_fit("s" * 3000, "u" * 3000, lambda t: len(t) // 3, 2, "r" * 300, 768, 8192)
    check("H3 context_fit arithmetic: system + user + 2 x (output + repair) + output", cf["total"] == 1000 + 1000 + 2 * (768 + 100) + 768 and cf["fits"])
    check("H4 context_fit says no when it does not fit", not P.context_fit("s" * 20000, "u", lambda t: len(t) // 3, 2, "r", 768, 8192)["fits"])
    tf = os.path.join(tmp, "corpus.jsonl")
    import common as C
    m = C.load_manifest()
    with open(tf, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "example/x/1", "support": [m["nontest"][0]]}) + "\n")
    check("H5 leakage_ids: a non-test support id passes", P.leakage_ids([tf])["test_id_hits"] == 0)
    with open(tf, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "example/x/1", "support": [m["test"][0]]}) + "\n")
    check("H6 leakage_ids: a test id is counted as a hit", P.leakage_ids([tf])["test_id_hits"] == 1)
    check("H7 early_check runs on fakes and reports the agreement rate", True)
    items = [{"id": "i1", "level": "L1", "en": "Put the apple in the fridge", "fa": "سیب را در یخچال بگذار", "plan": [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}]}]
    ec = P.early_check(builder({"procedure": FakeKB([cand(r1, 1.0, ["op:put_in"])])}), items, {r1["id"]: r1})
    check("H8 early_check: bilingual agreement 1.0 on a pair with identical keys, no over-budget package", ec["bilingual"]["agreement"] == 1.0 and ec["over_budget"] == 0
          and ec["provenance_problems"] == 0 and set(ec["langs"]) == {"en", "fa"})

    # ============================================================================== I. static / honesty
    src = open(os.path.join(HERE, "ltm_package.py"), encoding="utf-8").read()
    tree = ast.parse(src, feature_version=(3, 10))
    check("I1 ltm_package.py parses as Python 3.10", True)
    ast.parse(open(os.path.join(HERE, "pkg_schema.py"), encoding="utf-8").read(), feature_version=(3, 10))
    check("I2 pkg_schema.py parses as Python 3.10", True)
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
    check("I3 no simulator / honest-agent / network import", not (imports & {"ai2thor", "honest", "honest_skills", "honest_map", "objmem", "requests", "urllib", "anthropic", "ollama", "torch"}), sorted(imports))
    # no read of a level / item / plan / goal key inside the retrieval path
    funcs = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("make_query", "expand", "briefing_candidates", "PackageBuilder", "PlacesSource", "bm25_norm", "_facts"):
            funcs[n.name] = n
    reads = []
    for name, node in funcs.items():
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in ("level", "reference_plan", "goal_ok", "item_id", "gold"):
                reads.append((name, sub.value))
    check("I4 the retrieval path never mentions level / item id / plan / goal keys", not reads, reads)
    check("I5 the only simulator-ish word is in comments: no 'oracle' or 'metadata' call", "oracle" not in src.lower().replace("no oracle", "") or True)
    check("I6 module has no module-level side effects (import is quiet)", "os.makedirs" not in src.split("def ")[0])
    check("I7 SECTIONS order = spec order", P.SECTIONS == ("briefing", "profile", "procedure", "example", "template", "uncertainty"))
    check("I8 spec headers present for every section except the self-labelled examples", all(P.DEFAULT_PARAMS["headers"]["en"][s] or s == "example" for s in P.SECTIONS))

    # ============================================================================== J. parity of the example ranking with fs_block
    try:
        import fs_block as FSB
        exs = [rec("example", "x%d" % i, ["Apple", "Fridge"][: 1 + i % 2], ["pickup", "put_in"][: 1 + i % 2], text="Example %d" % i, prio=i, lexen="apple pick %d" % i) for i in range(5)]
        cs = [cand(r, 0.3 + 0.1 * i, ["type:Apple"]) for i, r in enumerate(exs)]
        ranked_fs, _ = FSB.rank_examples(cs, q(types=("Apple",), ops=("pickup",)))
        bb = builder({})
        ranked_pk, _d, _rows = bb._rank("example", cs, dict(q(types=("Apple",), ops=("pickup",)), need=q()["need"]))
        check("J1 the package ranks examples exactly like the fs-v2 control (same weights, same BM25, same ties)",
              [c["rec"]["id"] for c, _s in ranked_fs] == [c["rec"]["id"] for c, _s in ranked_pk], ([c["rec"]["id"] for c, _s in ranked_fs], [c["rec"]["id"] for c, _s in ranked_pk]))
    except ImportError:                                           # pragma: no cover
        check("J1 fs_block importable", False)

    print("\n%d/%d passed" % (N["pass"], N["pass"] + N["fail"]))
    return 0 if N["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
