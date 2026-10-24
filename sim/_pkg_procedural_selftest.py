"""Self-test for B4 (package spec pkg-1 s7): sim/ltm_procedural.py + sim/step5/procedures_kb.json.

Offline, Windows `python` from the REPO ROOT, no simulator / WSL / Ollama / API:

    python sim/_pkg_procedural_selftest.py

Covers: the JSON equals its build; check_static is clean against the executor contract and CATCHES planted defects
(wrong hand effect, bad route, kitchen name, test-item support, EN/FA token drift); one entry per honest tool;
executor facts agree with honest_skills' source; records are valid pkg.rec/1; candidates/expand behave on dev-style
queries, ignore level/item id, and never use the kitchen; the verification file path (B9 rows) switches entries off;
a synthetic kitchen (different fixtures and a different food category) works through caller data only.
"""
import ast
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

SIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM))

import ltm_procedural as P                                   # noqa: E402

_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print("  FAIL  %s %s" % (name, detail))


def top(cands, n=3):
    return [(c["rec"]["id"].split("/")[-1], c["match"]) for c in cands[:n]]


def match_of(cands, local):
    return next(c["match"] for c in cands if c["rec"]["id"].endswith("/" + local))


def rec_shape(r):
    need = {"schema", "id", "component", "kitchen", "keys", "lex", "text", "provenance", "created", "recency",
            "confidence", "privileged", "prio"}
    ok = need <= set(r) and r["schema"] == "pkg.rec/1" and r["component"] == "procedure" and r["kitchen"] == "*"
    ok = ok and set(r["keys"]) == {"types", "ops", "rels"} and r["privileged"] is False and r["confidence"] == 1.0
    ok = ok and r["text"]["en"].strip() and r["text"]["fa"].strip() and r["id"].isascii()
    ok = ok and set(r["provenance"]) >= {"source", "evidence", "authored_by", "support"}
    return bool(ok)


def main():
    import honest_skills as hs
    from planners import procedures_check as PC
    V = PC.validator()                                        # injected (the library never loads the executor)
    tools = list(hs.TOOLS)
    types = P.detector_types()
    contract = P.load_exec_contract()
    check("the contract's tool set equals honest_skills.TOOLS", P.contract_tools(contract) == sorted(tools))
    kb = P.load_kb(verified=False)

    # ---------------------------------------------------------------- 1 build / file
    src = (SIM / "ltm_procedural.py").read_text(encoding="utf-8")
    try:
        ast.parse(src, feature_version=(3, 10))
        ok310 = True
    except SyntaxError as e:                                  # pragma: no cover
        ok310 = False
        print(e)
    check("module parses as python 3.10", ok310)
    check("importing the module did not load the simulator", "ai2thor" not in sys.modules)
    check("no FloorPlan constant in the module source", not re.search(r"FloorPlan\d", src))
    fresh = P.build_kb()
    on_disk = json.loads(P.KB_PATH.read_text(encoding="utf-8"))
    strip = lambda d: {**{k: v for k, v in d.items() if k not in ("sha256", "entries")},
                       "entries": [{k: v for k, v in e.items() if k != "source"} for e in d["entries"]]}
    check("procedures_kb.json equals build_kb() (modulo the source hashes)", strip(fresh) == strip(on_disk))
    check("document sha256 verifies", on_disk["sha256"] == P._self_sha(on_disk))
    check("schema pkg.procs/1, kitchen *", on_disk["schema"] == "pkg.procs/1" and on_disk["kitchen"] == "*")
    blob = json.dumps(on_disk, ensure_ascii=False)
    check("no FloorPlan / FP token in the JSON", not re.search(r"FloorPlan|\bFP\d", blob))
    ids = [e["id"] for e in kb.entries]
    check("entry ids unique", len(ids) == len(set(ids)))
    check("entry schema pkg.proc/1", all(e["schema"] == "pkg.proc/1" for e in kb.entries))
    tool_entries = [e for e in kb.entries if not e["composite"]]
    comp = [e for e in kb.entries if e["composite"]]
    check("one tool entry per honest tool (22)", sorted(e["op"] for e in tool_entries) == sorted(tools) and len(tools) == 22,
          str(len(tool_entries)))
    check("composites cover slice/heat/microwave/chill/put-away/wash/coffee/stove/toast/empty/fill",
          {"slice-with-butterknife", "heat-then-place",   # A26: was slice-with-knife
           "microwave-put-close", "chill", "put-away-close", "wash-dish", "coffee",
           "stove-pan", "toast", "empty-to-sink", "fill-container"} <= {e["id"].split(":")[1] for e in comp})

    # ---------------------------------------------------------------- 2 static check
    problems = kb.check_static(contract, tools, types, validator=V)
    check("check_static == [] against the executor contract (%s)" % contract.get("_origin"), problems == [], str(problems[:5]))
    check("a validator was available", kb.last_validator is not None, str(kb.last_validator))
    check("without an injected validator check_static is NOT clean (no silent skip)",
          any("no validator supplied" in p for p in kb.check_static(contract, tools, types)))
    # the checked-in sample (B1) too, loaded through an explicit path (never a silent default)
    c2 = P.load_exec_contract(P.SAMPLE_CONTRACT_PATH)
    check("check_static clean against %s" % P.SAMPLE_CONTRACT_PATH.name, kb.check_static(c2, tools, types, validator=V) == [],
          str(kb.check_static(c2, tools, types, validator=V)[:3]))
    check("the default contract is the generated file only", P.EXEC_PATHS == (P.EXEC_CONTRACT_PATH,) and
          contract["_origin"] == str(P.EXEC_CONTRACT_PATH))
    with tempfile.TemporaryDirectory() as td:
        try:
            P.load_exec_contract(Path(td) / "missing.json")
            check("a missing contract file raises (fail closed)", False)
        except FileNotFoundError as ex:
            check("a missing contract file raises (fail closed) and names the regenerate command", P.REGENERATE_CONTRACT in str(ex))
        bad = json.loads(P.EXEC_CONTRACT_PATH.read_text(encoding="utf-8"))
        bad["tools"]["heat"]["hand"] = "none"
        (Path(td) / "tampered.json").write_text(json.dumps(bad), encoding="utf-8")
        try:
            P.load_exec_contract(Path(td) / "tampered.json")
            check("a contract whose content hash does not verify raises", False)
        except ValueError:
            check("a contract whose content hash does not verify raises", True)

    # negative controls (each plants ONE defect in a copy)
    def planted(mut):
        d = copy.deepcopy(on_disk)
        mut(d)
        k2 = P.KB(d)
        d.pop("sha256", None)
        k2.doc.pop("sha256", None)
        return k2.check_static(contract, tools, types, validator=V)

    c_bad = copy.deepcopy(contract)
    c_bad["tools"]["heat"]["hand"] = "none"
    check("planted: contract says heat keeps nothing in hand -> flagged", any("heat" in p for p in kb.check_static(c_bad, tools, types, validator=V)))
    c_bad2 = copy.deepcopy(contract)
    c_bad2["tools"]["put_in"]["requires_held"] = False
    check("planted: contract says put_in needs no held object -> flagged", bool(kb.check_static(c_bad2, tools, types, validator=V)))
    check("planted: route with an unknown tool -> flagged",
          bool(planted(lambda d: d["entries"][0]["route"].__setitem__(0, {"action": "teleport", "object": "?item", "to": None, "arg": None}))))
    check("planted: put_in with nothing held (hand trace) -> flagged",
          bool(planted(lambda d: [e for e in d["entries"] if e["id"] == "proc:tool-put-in"][0].__setitem__("route", [{"action": "put_in", "object": "?place", "to": None, "arg": None}]))))
    check("planted: kitchen name in text -> flagged",
          bool(planted(lambda d: d["entries"][3]["text"].__setitem__("en", d["entries"][3]["text"]["en"] + " FloorPlan9"))))
    check("planted: realism item inside requires -> flagged",
          bool(planted(lambda d: d["entries"][7]["requires"].append({"cond": "hold a ButterKnife", "kind": "realism", "type": "ButterKnife"}))))
    check("planted: executor requirement that maps to nothing -> flagged",
          bool(planted(lambda d: d["entries"][5]["requires"].append({"cond": "the sky is blue", "kind": "executor"}))))
    check("planted: EN type token missing in FA -> flagged",
          bool(planted(lambda d: [e for e in d["entries"] if e["id"] == "proc:chill"][0]["text"].__setitem__("fa", "بدون نوع"))))
    check("planted: tool written as name(...) in text -> flagged",
          bool(planted(lambda d: d["entries"][0]["text"].__setitem__("en", "pickup(Apple) takes it"))))
    check("planted: absence claim in text -> flagged",
          bool(planted(lambda d: d["entries"][0]["text"].__setitem__("en", "pickup: the object is absent here"))))
    check("planted: a missing tool entry -> flagged", bool(planted(lambda d: d["entries"].pop(0))))
    check("planted: declared hand_end contradicts the trace -> flagged",
          bool(planted(lambda d: [e for e in d["entries"] if e["id"] == "proc:tool-pickup"][0].__setitem__("hand_end", "empty"))))

    # test-item guard on `support`, with a SYNTHETIC manifest (no real test id is used here)
    old_split = P.SPLIT_PATH
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "split.json"
        fake.write_text(json.dumps({"nontest": ["a-ok"], "practice_pool": ["b-ok"], "dev_pilot": [], "test": ["t-secret"],
                                    "test_uncapped_not_scored": []}), encoding="utf-8")
        P.SPLIT_PATH = fake
        try:
            check("support guard: a test id is flagged", any("TEST item" in x for x in P._support_problems(["t-secret"])))
            check("support guard: a non-test id passes", P._support_problems(["a-ok", "b-ok"]) == [])
            check("support guard: an unknown id is flagged", bool(P._support_problems(["zzz"])))
        finally:
            P.SPLIT_PATH = old_split
    sup = sorted({s for e in kb.entries for s in e["provenance"]["support"]})
    check("every support id is a non-test item (real manifest)", P._support_problems(sup) == [], str(P._support_problems(sup)[:3]))
    _ds = json.loads((SIM / "dataset" / "general.json").read_text(encoding="utf-8"))
    # A26 (K-B2b): provenance (tuned_on) may cite an item archived by A26; it stays in the dataset file under "archived"
    check("support ids all exist in the dataset (items, or A26 archived)",
          set(sup) <= {i["id"] for i in _ds["items"]} | {i["id"] for i in _ds.get("archived", [])},
          str(sorted(set(sup) - {i["id"] for i in _ds["items"]} - {i["id"] for i in _ds.get("archived", [])})))
    check("spec_only flag matches an empty support", all(e["provenance"]["spec_only"] == (not e["provenance"]["support"]) for e in kb.entries))

    # ---------------------------------------------------------------- 3 agreement with honest_skills (only what the skills implement)
    hsrc = (SIM / "honest_skills.py").read_text(encoding="utf-8")
    m = re.search(r'if action in \("heat", "cook"\):(.*?)if action == "wash":(.*?)if action == "fill":', hsrc, re.S)
    check("found the heat/cook and wash dispatch in honest_skills", m is not None)
    if m:
        heat_src, wash_src = m.group(1), m.group(2)
        for t in ("heat", "cook"):
            fx = [r["fixture"] for r in kb.by_id["proc:tool-" + t]["requires"] if r.get("fixture")]
            check("%s: required fixture(s) %s are used by the dispatch" % (t, fx), fx and all('"%s"' % f in heat_src for f in fx))
        fx = [r["fixture"] for r in kb.by_id["proc:tool-wash"]["requires"] if r.get("fixture")]
        check("wash: required fixtures %s are used by the dispatch" % fx, fx and all('"%s"' % f in wash_src for f in fx))
        check("wash leaves the item in the sink (put SinkBasin, no pickup afterwards)", '_put("SinkBasin")' in wash_src and "pickup" not in wash_src)
        check("heat picks the food up again (the food ends in hand)", '"pickup"' in heat_src and kb.by_id["proc:tool-heat"]["hand_end"] == "held")
    check("every tool the KB routes through exists in honest_skills.TOOLS",
          all(st["action"] in tools for e in kb.entries for st in e["route"]))

    # ---------------------------------------------------------------- 4 records
    recs = kb.records()
    check("one record per active entry", len(recs) == len(kb.entries))
    check("all records are valid pkg.rec/1", all(rec_shape(r) for r in recs), str([r["id"] for r in recs if not rec_shape(r)][:3]))
    check("record ids unique", len({r["id"] for r in recs}) == len(recs))
    check("records deterministic", json.dumps(kb.records(), sort_keys=True) == json.dumps(P.load_kb(verified=False).records(), sort_keys=True))
    check("record keys.types are detector types", all(set(r["keys"]["types"]) <= set(types) for r in recs))
    check("record rels use the closed vocabulary", all(x.split(":")[0] in P.RELS for r in recs for x in r["keys"]["rels"]))
    check("heat is separate from cook (distinct records, distinct effect tokens)",
          "effect:hot" in kb.by_id["proc:tool-heat"]["keys"]["rels"] and "effect:cooked" in kb.by_id["proc:tool-cook"]["keys"]["rels"])
    # A26 (K-B2b) replaces: "slice realism is not an executor requirement" (requires all executor AND a realism item present). Under
    # A26 the ButterKnife in hand IS an executor fact (contract slice.requires_tool), so it moved from realism to requires.
    check("slice (A26): the ButterKnife requirement is an executor requirement on the contract, no realism item left",
          [(r["kind"], r.get("tool"), r.get("field"), r.get("value")) for r in kb.by_id["proc:tool-slice"]["requires"]]
          == [("executor", "slice", "requires_tool", ["ButterKnife"])] and not kb.by_id["proc:tool-slice"]["realism"])
    ct = lambda t: -(-len(t) // 3)
    tok = {r["id"]: (ct(r["text"]["en"]), int(ct(r["text"]["fa"]) * 1.7)) for r in recs}
    check("every record is short (fake count <= 120 tokens, FA weighted 1.7)", all(max(v) <= 120 for v in tok.values()), str(max(tok.values())))
    print("  info  record tokens (fake counter): total EN %d / FA(x2) %d; max EN %d" % (
        sum(v[0] for v in tok.values()), sum(v[1] for v in tok.values()), max(v[0] for v in tok.values())))

    # ---------------------------------------------------------------- 5 candidates
    q_heat = {"instruction": "heat the egg", "lang": "en", "types": ["Egg", "Microwave"], "ops": ["heat"]}
    c = kb.candidates(q_heat)
    check("heat egg: heat tool and heat-then-place match 1.0",
          match_of(c, "tool-heat") == 1.0 and match_of(c, "heat-then-place") == 1.0, str(top(c, 5)))
    check("heat egg: cook is only a 0.5 confusion hit", match_of(c, "tool-cook") == 0.5)
    check("heat egg: microwave-put-close surfaces as a 0.5 confusion hit", match_of(c, "microwave-put-close") == 0.5)
    check("heat egg: chill does not match", match_of(c, "chill") == 0.0)
    q_chill = {"instruction": "put the apple in the fridge and close it", "lang": "en", "types": ["Apple", "Fridge"], "ops": ["put", "close"]}
    c = kb.candidates(q_chill)
    check("put apple in fridge + close: chill 1.0 and first among composites",
          match_of(c, "chill") == 1.0 and [x["rec"]["id"].split("/")[-1] for x in c if x["match"] == 1.0 and x["rec"]["prio"] < 100][0] == "chill")
    check("... put-away-close needs a Cabinet/Drawer: 0.0", match_of(c, "put-away-close") == 0.0)
    check("... discard needs a GarbageCan: 0.0", match_of(c, "discard") == 0.0)
    check("... pick_and_place and close tools match", match_of(c, "tool-pick-and-place") >= 0.7 and match_of(c, "tool-close") >= 0.7)
    q_fb = {"instruction": "???", "lang": "en", "types": ["Microwave"], "ops": [], "parser_fallback": True}
    c = kb.candidates(q_fb)
    check("parser failure with a typed mention: only 0.5 type hits, never 1.0",
          max(x["match"] for x in c) == 0.5 and match_of(c, "tool-heat") == 0.5, str(top(c, 3)))
    q_none = {"instruction": "hello", "lang": "en", "types": [], "ops": []}
    check("no typed key, no op -> every match is 0 and include_zero=False returns nothing",
          all(x["match"] == 0 for x in kb.candidates(q_none)) and kb.candidates(q_none, include_zero=False) == [])
    q_on = {"types": ["StoveBurner"], "ops": ["toggle"], "parse": {"roles": [{"verb": "toggle", "req": "on"}]}}
    c = kb.candidates(q_on)
    check("turn on the stove: toggle_on and the stove composite match", match_of(c, "tool-toggle-on") == 1.0 or match_of(c, "tool-toggle-on") >= 0.7)
    check("... stove-pan composite fits via StoveBurner", match_of(c, "stove-pan") == 1.0)
    q_a = dict(q_heat)
    q_b = dict(q_heat, level="L3", item="g-l3-secret", gold=[{"action": "heat"}], goal="x", kitchen="K9")
    check("level / item / gold / goal / kitchen fields do not change the candidates",
          json.dumps(kb.candidates(q_a), sort_keys=True) == json.dumps(kb.candidates(q_b), sort_keys=True))
    check("candidate shape pkg.cand/1", all(set(x) == {"rec", "match", "why"} and 0 <= x["match"] <= 1 for x in kb.candidates(q_heat)))
    order = [(-x["match"], x["rec"]["prio"], x["rec"]["id"]) for x in kb.candidates(q_heat)]
    check("candidates ordered by match desc, prio, id", order == sorted(order))
    check("a match of 0 has no 'why'", all(x["why"] == [] for x in kb.candidates(q_heat) if x["match"] == 0))

    # ---------------------------------------------------------------- 6 expand
    e = kb.expand(["slice"], ["Apple"])
    # A26 (K-B2b) replaces: check("expand slice: Knife as a tool", "Knife" in e["tools"], str(e))
    check("expand slice (A26): the ButterKnife as the tool, no Knife", "ButterKnife" in e["tools"] and "Knife" not in e["tools"], str(e))
    e = kb.expand(["heat"], ["Egg"])
    check("expand heat: Microwave appliance, executor requirement listed", e["appliances"] == ["Microwave"] and "a Microwave exists" in e["requires"], str(e))
    e = kb.expand(["wash"], ["Plate"])
    check("expand wash: SinkBasin + Faucet", {"SinkBasin", "Faucet"} <= set(e["appliances"]), str(e))
    e = kb.expand(["put"], ["Apple", "Fridge"])
    check("expand put apple fridge: Fridge receptacle", "Fridge" in e["receptacles"], str(e))
    e = kb.expand(["toggle_on"], ["StoveBurner"])
    check("expand toggle on stove: StoveKnob appliance", "StoveKnob" in e["appliances"], str(e))
    e = kb.expand([], [])
    check("expand nothing: nothing", all(e[k] == [] for k in ("tools", "appliances", "receptacles", "requires", "actions")))
    check("expand is deterministic", kb.expand(["put", "close"], ["Bread", "Fridge"]) == kb.expand(["close", "put"], ["Fridge", "Bread"]))
    check("expand never returns an unlisted type", all(t in types for k in ("tools", "appliances", "receptacles")
                                                         for t in kb.expand(["put", "heat", "wash", "slice", "toggle"], ["Fridge", "Egg"])[k]))

    # ---------------------------------------------------------------- 7 instantiate
    r = kb.instantiate("proc:heat-then-place", {"?food": "Egg", "?place": "Bowl"})
    check("instantiate heat-then-place", [s["action"] for s in r] == ["heat", "pick_and_place"] and r[1]["to"] == "Bowl" and r[1]["object"] == "Egg", str(r))
    for bad in ({"?food": "Egg"}, {"?food": "Egg", "?place": "Bowl", "?zzz": "x"}):
        try:
            kb.instantiate("proc:heat-then-place", bad)
            check("instantiate rejects %s" % bad, False)
        except ValueError:
            check("instantiate rejects %s" % bad, True)
    same_ok = True
    for e in kb.entries:
        bs = kb.sample_bindings(e["id"], 2)
        same_ok &= len(bs) >= 1 and (len(bs) == 2 or not e["slots"] or all(len(s["samples"]) == 1 for s in e["slots"].values()))
        same_ok &= kb.sample_bindings(e["id"], 2) == bs
    check("sample_bindings: deterministic, two distinct bindings where the slots allow it", same_ok)
    check("two distinct bindings exist for every composite with slots (B9 needs 2 configurations)",
          all(len(kb.sample_bindings(e["id"], 2)) == 2 for e in comp if e["slots"]))
    a, b = kb.sample_bindings("proc:chill", 2)
    check("the 2 chill bindings are distinct", a != b)

    # ---------------------------------------------------------------- 8 verification results (B9 rows)
    def row(eid, item, cls="feasible"):
        return {"item": item, "mode": "procedures", "requirement": eid, "class": cls}
    k = P.load_kb(verified=False)
    k.apply_verified([row("proc:chill", "a"), row("proc:chill", "b"), row("proc:toast", "a"), row("proc:coffee", "a"),
                      row("proc:coffee", "b", "executor_defect"), row("proc:stove-pan", "a", "navigation")])
    check("2 feasible configurations -> verified.wsl True", k.by_id["proc:chill"]["verified"]["wsl"] is True)
    check("1 feasible configuration only -> stays unverified (None)", k.by_id["proc:toast"]["verified"]["wsl"] is None and k.by_id["proc:toast"]["active"])
    check("an infeasible row -> wsl False and inactive", k.by_id["proc:coffee"]["verified"]["wsl"] is False and not k.by_id["proc:coffee"]["active"])
    check("... and the inactive entry is never a record or a candidate",
          all("coffee" not in r["id"] for r in k.records()) and all("coffee" not in c["rec"]["id"] for c in k.candidates({"types": ["CoffeeMachine"], "ops": ["toggle"]})))
    check("an inconclusive (navigation) row neither verifies nor deactivates", k.by_id["proc:stove-pan"]["active"] and k.by_id["proc:stove-pan"]["verified"]["wsl"] is None)
    k3 = P.load_kb(verified=False)
    k3.apply_verified([row("proc:chill", "a", "executor_defect"), row("proc:chill", "a"), row("proc:chill", "b"),
                       row("proc:toast", "a", "api_cannot_express"), {**row("proc:wash-dish", "a", "executor_defect"), "preconditions_ok": False},
                       row("proc:wash-dish", "b"), row("proc:wash-dish", "c")])
    check("the LAST row per (entry, configuration) counts (a fixed defect re-run verifies)", k3.by_id["proc:chill"]["verified"]["wsl"] is True and k3.by_id["proc:chill"]["active"])
    check("api_cannot_express deactivates", not k3.by_id["proc:toast"]["active"])
    check("a row whose preconditions were unmet is ignored", k3.by_id["proc:wash-dish"]["verified"]["wsl"] is True)
    check("fixture-dependent composites say availability is unknown (Astra H-B4 Q2)",
          all("availability unknown" in kb.by_id["proc:" + i]["text"]["en"] for i in ("coffee", "toast", "stove-pan")))
    check("an entry with no rows is untouched", k.by_id["proc:wash-dish"]["verified"]["wsl"] is None and k.by_id["proc:wash-dish"]["active"])
    k2 = P.load_kb(verified=False)
    k2.apply_verified(None, {"proc:toast": {"wsl": True}, "proc:chill": {"wsl": False}})
    check("results-file map applies", k2.by_id["proc:toast"]["verified"]["wsl"] is True and not k2.by_id["proc:chill"]["active"])
    with tempfile.TemporaryDirectory() as td:
        old = (P.FEAS_PATH, P.VERIFIED_PATH)
        P.FEAS_PATH = Path(td) / "feas.jsonl"
        P.VERIFIED_PATH = Path(td) / "nope.json"
        P.FEAS_PATH.write_text("\n".join(json.dumps(x) for x in [row("proc:chill", "a"), row("proc:chill", "b")]) + "\n", encoding="utf-8")
        try:
            check("load_kb reads B9's jsonl", P.load_kb().by_id["proc:chill"]["verified"]["wsl"] is True)
        finally:
            P.FEAS_PATH, P.VERIFIED_PATH = old
    check("the checked-in KB starts with verified.wsl null everywhere (B9 fills it, not B4)",
          all(e["verified"] == {"static": False, "wsl": None} for e in on_disk["entries"]))

    # ---------------------------------------------------------------- 9 synthetic kitchen (different fixtures / categories), caller data only
    synth_cats = {"food": {"types": ["Mango", "Egg"]}, "cutlery": {"types": ["Fork"], "receptacles": ["Drawer"]}}
    c = kb.candidates({"types": ["Mango", "Microwave"], "ops": ["heat"]})
    check("synthetic: Mango is not food by default -> heat-then-place still fits via the Microwave type", match_of(c, "heat-then-place") == 1.0)
    e = kb.expand(["heat", "put"], ["Mango", "Fridge"], synth_cats)
    check("synthetic: caller categories feed the fit (Mango food -> heat-then-place expands the Microwave)",
          "Microwave" in e["appliances"], str(e))
    e0 = kb.expand(["heat", "put"], ["Mango", "Fridge"], None)
    check("synthetic: ... and without them a food-only fit would not (chill still adds nothing wrong)", "Microwave" in e0["appliances"])
    synth_doc = copy.deepcopy(on_disk)
    synth_doc["categories"]["food"] = ["Mango"]
    synth_doc.pop("sha256")
    ks = P.KB(synth_doc)
    cs = ks.candidates({"types": ["Mango"], "ops": ["slice"]})
    check("synthetic KB (different category data): slice fits only the new members", match_of(cs, "tool-slice") in (0.7, 1.0))
    check("synthetic: a 2-Fridge / no-Toaster kitchen needs no change - records carry no kitchen",
          all(r["kitchen"] == "*" for r in kb.records()) and "Synth" not in blob)
    check("candidates do not depend on a kitchen label",
          json.dumps(kb.candidates(dict(q_heat, kitchen="SynthKitchen")), sort_keys=True) == json.dumps(kb.candidates(q_heat), sort_keys=True))

    # ---------------------------------------------------------------- 11 round-1 additions (I-F2): recipes, holding, afford
    contract = P.load_exec_contract()

    def acts(eid, b, hold=None, tail=False):
        r, n = kb.holding_route(eid, b, hold, contract, tail)
        return [x["action"] + ":" + str(x["object"]) + (">" + x["to"] if x.get("to") else "") for x in r], n
    check("recipe: microwave put -> pick_and_place, close",
          acts("proc:microwave-put-close", {"?item": "Egg"})[0] == ["pick_and_place:Egg>Microwave", "close:Microwave"])
    check("recipe: microwave run -> put, close, THEN toggle_on (never on before closed)",
          acts("proc:microwave-run", {"?item": "Egg"})[0] == ["pick_and_place:Egg>Microwave", "close:Microwave", "toggle_on:Microwave"])
    check("recipe: microwave take-out tail = off, open, pickup",
          acts("proc:microwave-run", {"?item": "Egg"}, tail=True)[0][3:] == ["toggle_off:Microwave", "open:Microwave", "pickup:Egg"])
    check("recipe: fridge put -> put, close", acts("proc:chill", {"?food": "Apple"})[0] == ["pick_and_place:Apple>Fridge", "close:Fridge"])
    check("recipe: stove -> put on burner, knob on (burner is switched by its knob)",
          acts("proc:stove-pan", {"?vessel": "Pan"})[0] == ["pick_and_place:Pan>StoveBurner", "toggle_on:StoveKnob"])
    check("recipe: stove tail switches the KNOB off", acts("proc:stove-pan", {"?vessel": "Pan"}, tail=True)[0][-1] == "toggle_off:StoveKnob")
    check("recipe: sink -> dish goes to SinkBasin; wash is a single step",
          acts("proc:dish-to-sink", {"?dish": "Cup"})[0] == ["pick_and_place:Cup>SinkBasin"]
          and acts("proc:wash-dish", {"?dish": "Mug"})[0] == ["wash:Mug"])
    check("recipe: faucet-run says wash does it itself", kb.by_id["proc:faucet-run"]["text"]["en"].count("wash") >= 1)
    check("tails are contract-clean",
          all(not P._contract_route_problems(kb.holding_route(e["id"], kb.sample_bindings(e["id"], 1)[0], None, contract, True)[0], contract)
              for e in kb.entries if e.get("tail")))
    heat_t = kb.by_id["proc:tool-heat"]["text"]["en"] + kb.by_id["proc:tool-cook"]["text"]["en"]
    check("heat/cook text: no 'in your hand' wording; says the cycle includes close and take-out and is not a put",
          "in your hand" not in heat_t and "not 'put X in the microwave'" in heat_t and "door closed" in heat_t)
    check("heat/cook text is warm/heat only", "or cook the food" not in heat_t)
    check("heat contract fact still true: the executor really returns the food to the hand",
          kb.by_id["proc:tool-heat"]["hand_end"] == "held")
    r, n = kb.holding_route("proc:chill", {"?food": "Apple"}, "Apple", contract)
    check("holding the target: pick_and_place -> put_in Fridge (+ close)",
          [x["action"] for x in r] == ["put_in", "close"] and r[0]["object"] == "Fridge" and n == ["holding_target"], str(r))
    r, n = kb.holding_route("proc:chill", {"?food": "Apple"}, "Mug", contract)
    check("holding another object: release first with the contract's drop (no kitchen constant), then the recipe",
          [(x["action"], x["object"]) for x in r][:2] == [("drop", None), ("pick_and_place", "Apple")] and n == ["holding_other"], str(r))
    r2, _ = kb.holding_route("proc:chill", {"?food": "Apple"}, "Mug", contract, release_to="SinkBasin")
    check("holding another object with a receptacle the caller has seen: put_in that receptacle",
          [(x["action"], x["object"]) for x in r2][:1] == [("put_in", "SinkBasin")])
    nodrop = {"tools": {k: v for k, v in contract["tools"].items() if k != "drop"}}
    r3, _ = P.adapt_route([{"action": "pick_and_place", "object": "Apple", "to": "Fridge", "arg": None}], "Mug", nodrop)
    check("no drop in the contract: documented CounterTop fallback", (r3[0]["action"], r3[0]["object"]) == ("put_in", P.RELEASE_FALLBACK))
    r, n = kb.holding_route("proc:chill", {"?food": "Apple"}, None, contract)
    check("empty hand: recipe unchanged, and NO search_for is added (the skills search themselves)",
          [x["action"] for x in r] == ["pick_and_place", "close"] and not n)
    r, n = kb.holding_route("proc:find-then-take", {"?item": "Fork"}, "Fork", contract)          # A26: was Knife
    check("find-then-take while holding the target: no pickup left, search_for kept as the asked find",
          [x["action"] for x in r] == ["search_for"] and n == ["holding_target"], str(r))
    # A26 (K-B2b): proc:slice-with-knife -> proc:slice-with-butterknife, the held tool is the ButterKnife
    r, n = kb.holding_route("proc:slice-with-butterknife", {"?food": "Apple"}, "ButterKnife", contract)
    check("slice while already holding the ButterKnife: no second pickup", [x["action"] for x in r] == ["slice"], str(r))
    r, n = kb.holding_route("proc:slice-with-butterknife", {"?food": "Apple"}, "Apple", contract)
    check("slice holding the food: drop it, then pickup ButterKnife (one hand)", [x["action"] for x in r][:2] == ["drop", "pickup"]
          and r[1]["object"] == "ButterKnife", str(r))
    r, n = kb.holding_route("proc:microwave-put-close", {"?item": "Egg"}, "Egg", contract)
    check("microwave put while holding the egg: put_in Microwave, close",
          [(x["action"], x["object"]) for x in r] == [("put_in", "Microwave"), ("close", "Microwave")])
    skip = ("proc:tool-put-in", "proc:tool-drop", "proc:tool-close-all")
    bad = []
    for e in kb.entries:
        if e["id"] in skip:
            continue
        for b in kb.sample_bindings(e["id"], 2):
            for h in (None, "Apple", "Mug", "ButterKnife"):          # A26: was Knife
                rr = kb.holding_route(e["id"], b, h, contract)[0]
                pr = P._contract_route_problems(rr, contract) + P.trace_hand(rr, contract, h)[1]
                if pr:
                    bad.append((e["id"], h, pr[:1]))
    check("every adapted route over every entry/holding is contract-clean", not bad, str(bad[:3]))
    txt, n = kb.render("proc:chill", {"?food": "Apple"}, "Apple", contract=contract)
    check("render uses the adapted route", txt == "put_in Fridge, then close Fridge." and n == ["holding_target"], txt)
    check("holding rules are in the KB, three of them",
          [x["when"] for x in kb.doc["holding_rules"]] == ["holding_target", "holding_other", "location_unknown"])
    af = lambda w: kb.afford(w)
    # A26 (K-B2b) replaces: "afford: cut/slice/chop -> Knife or ButterKnife" (tools == ["Knife", "ButterKnife"])
    check("afford (A26): cut/slice/chop -> the ButterKnife only",
          all(af(w) and af(w)["tools"] == ["ButterKnife"] for w in ("cut", "slice", "chopping", "sliced")))
    check("afford: flip -> Spatula", af("flip")["tools"] == ["Spatula"] and af("flipping")["tools"] == ["Spatula"])
    check("afford: chill -> Fridge with the chill recipe", af("chill")["appliances"] == ["Fridge"] and af("chilled")["recipe"] == "proc:chill")
    check("afford: heat -> Microwave or StoveBurner", af("heat")["appliances"] == ["Microwave", "StoveBurner"] and af("warm")["key"] == "heat")
    check("afford: unknown word -> None", af("zorble") is None and af("") is None)
    check("afford: every row's tools/appliances are detector types and its recipe exists",
          all(set(a["tools"]) | set(a["appliances"]) <= set(types) and (not a["recipe"] or a["recipe"] in kb.by_id) for a in kb.afford_map.values()))
    check("afford: every word maps to a row", all(v in kb.afford_map for v in kb.afford_words.values()))
    ar = kb.afford_records()
    check("afford records are valid pkg.rec/1, English only (fa = en), kitchen-free",
          all(rec_shape(r) and r["kitchen"] == "*" and r["text"]["fa"] == r["text"]["en"] for r in ar) and len(ar) == len(kb.afford_map) + sum(1 for a in kb.afford_map.values() if a.get("request")))
    check("afford records are NOT in records() (retrieval owner opts in)", not any("afford" in r["id"] for r in kb.records()))
    check("afford: no record text carries a kitchen name or an item id", not re.search(r"FloorPlan|g-l\d", json.dumps([{k: v for k, v in r.items() if k != "provenance"} for r in ar])))
    check("afford: tuned_on items are listed as provenance.support (leave-out can drop them)",
          all(r["provenance"]["support"] == list(kb.afford_map[r["id"].rsplit("afford-", 1)[1].replace("-request", "")].get("tuned_on") or []) for r in ar)
          and any(r["provenance"]["support"] for r in ar))
    check("entries cite only non-test support", all(not P._support_problems(e["provenance"].get("support", [])) for e in kb.entries))

    # ---------------------------------------------------------------- 10 early check (diagnostic)
    rep = P.early_check(validator=V)
    check("early check: check_static clean on the real KB", rep["check_static"] == [])
    check("early check: 112 non-test items (116 minus the 4 A26-archived), test items excluded", rep["n_items"] == 112, rep["n_items"])
    print("  info  early check: EN adds-a-useful-type %s, FA %s (diagnostic only)" % (rep["en"], rep["fa"]))

    print("\n_pkg_procedural_selftest: %d/%d" % (_pass, _pass + _fail))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
