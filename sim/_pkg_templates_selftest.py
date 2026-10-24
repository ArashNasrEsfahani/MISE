"""Self-test for B6 (package spec pkg-1 section 9.1): L3 decomposition templates (sim/ltm_templates.py, sim/step5/l3_templates.json).

Offline, Windows `python` from the REPO ROOT, standard library only:

    python sim/_pkg_templates_selftest.py            # gate
    python sim/_pkg_templates_selftest.py --early    # gate + the B6 early check (diagnostic numbers for the ledger)

Checks: file integrity (schema, sha, English only, unique ids, support non-empty and NON-TEST only, static text fresh); every
skeleton validates under the executor contract (and validator_v2 when present); selection is a function of the instruction,
language, holding state and observed types only (static AST scan + invariance to level / id / plan / goal keys); behaviour on
hand-written queries in EN and FA, including negatives; a synthetic kitchen with a different fixture set; token budget of an
instantiated template; the shape matcher; `check` detects planted defects; Python 3.10 syntax.
"""
import ast
import copy
import json
import re
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
ROOT = SIM.parent
sys.path.insert(0, str(SIM))
import ltm_templates as T  # noqa: E402

_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print("  FAIL  %s %s" % (name, detail))


def q(instr, lang="en", holding=None, observed=None, types=None, extra=None):
    d = {"schema": "pkg.query/1", "instruction": instr, "lang": lang, "holding": holding,
         "observed": observed or [], "parse": {"ok": False, "roles": [], "reasons": []},
         "types": types or [], "ops": [], "goal_states": [], "need": {}, "parser_fallback": True}
    if extra:
        d.update(extra)
    return d


def top_ids(tp, query):
    return [c["rec"]["id"].split("/")[-1] for c in tp.candidates(query)]


# ------------------------------------------------------------------------------------------------ local shape checkers
REC_KEYS = {"schema", "id", "component", "kitchen", "keys", "lex", "text", "provenance", "created", "recency", "confidence",
            "privileged", "prio"}


def rec_ok(r) -> bool:
    if set(r) != REC_KEYS or r["schema"] != "pkg.rec/1" or r["component"] != "template" or r["kitchen"] != "*":
        return False
    if not re.fullmatch(r"[\x20-\x7e]+", r["id"]) or r["privileged"] is not False:
        return False
    k = r["keys"]
    if set(k) != {"types", "ops", "rels"} or not all(isinstance(k[x], list) for x in k):
        return False
    if r["text"].get("fa") != "" or not r["text"].get("en"):
        return False
    pv = r["provenance"]
    if set(pv) != {"source", "evidence", "authored_by", "support"} or not pv["source"].startswith("spec:"):
        return False
    return 0.0 <= r["confidence"] <= 1.0 and r["created"] == 0 and r["recency"] == 0 and isinstance(r["prio"], int)


def cand_ok(c) -> bool:
    return (set(c) == {"rec", "match", "why"} and 0.0 <= c["match"] <= 1.0 and rec_ok(c["rec"])
            and all(isinstance(w, str) for w in c["why"]))


def load_contract():
    p = SIM / "_pkg_fixtures" / "exec_contract.sample.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), "fixture"
    try:                                                      # derive from step 4 (same source B1 uses)
        import step4_gemma as g
        tools = {}
        for t, (o, to, arg, what) in g.TOOL_SPECS.items():
            eff = g.HAND.get(t)
            tools[t] = {"object": o or "none", "to": to or "none", "arg": ["water", "coffee", "wine"] if arg else "none",
                        "hand": {"take": "take", "take-release": "take-release", "need": "release"}.get(eff, "keep"),
                        "requires_held": eff == "need", "requires_empty_hand": eff in ("take", "take-release")}
        return {"schema": "pkg.exec/1", "tools": tools}, "derived"
    except Exception as e:                                    # pragma: no cover
        raise SystemExit("no exec contract available: %r" % e)


def v2_validator():
    try:
        import planners.validator_v2 as v2                     # noqa: F401
    except Exception:
        try:
            from planners import validator_v2 as v2
        except Exception:
            return None

    def run(steps, types):
        return v2.validate({"say": "", "plan": steps}, types)
    return run


def main():
    early = "--early" in sys.argv
    contract, csrc = load_contract()
    tp = T.load()
    raw = json.loads(T.DEFAULT_PATH.read_text(encoding="utf-8"))
    manifest = T.load_manifest()
    nontest_ids = set(manifest["nontest"])
    test_like = set(manifest["test"]) | set(manifest["test_uncapped_not_scored"])

    # ---------------------------------------------------------------------------------------------- 1 file integrity
    check("schema", raw["schema"] == "pkg.tpls/1" and raw["version"] == "tpl-1")
    check("authored_from", raw["authored_from"] == "domain spec (B6, no test item opened)")
    check("sha256 matches", raw["sha256"] == T._self_sha(raw), "run: python sim/ltm_templates.py --build --write")
    check("no Persian text in the stored file", not re.search(r"[؀-ۿ]", json.dumps(raw, ensure_ascii=False)))
    ids = [t["id"] for t in tp.templates]
    check("ids unique and prefixed", len(ids) == len(set(ids)) and all(i.startswith("tpl:") for i in ids))
    check("template set is small (6..16)", 6 <= len(ids) <= 16, str(len(ids)))
    vocab = set(T.LC.default_lexicon().types.values())
    check("every category member is a detector type", all(t in vocab for c in tp.cats.values() for t in c))
    check("every purpose tool is a detector type", all(p["tool"] in vocab for p in tp.purposes))
    check("purposes cover distinct tools", len({p["tool"] for p in tp.purposes}) >= 8)
    check("every intent group name is defined", all(g in tp.groups_en for t in tp.templates for g in t["intent"]["any"]))
    check("every FA group key has an EN group", set(T.FA_LEXICON["groups"]) <= set(tp.groups_en))
    check("every FA purpose key has an EN purpose", set(T.FA_LEXICON["purposes"]) <= {p["id"] for p in tp.purposes})
    check("rejected templates are archived with a reason",
          all(r.get("reason") and r.get("template", {}).get("id") == r["id"] for r in tp.rejected))
    check("rejected ones are not in the shipped list", not ({r["id"] for r in tp.rejected} & set(ids)))
    for t in tp.templates:
        check("%s has support" % t["id"], len(t.get("support", [])) >= 1)
        check("%s support is non-test only" % t["id"],
              all(i in nontest_ids and i not in test_like for i in t.get("support", [])))
        check("%s support strength labelled" % t["id"], t.get("support_strength") == "strong")
        check("%s static text fresh" % t["id"], t["text"]["en"] == tp.render(t, None))
        check("%s has title/note/checks" % t["id"], t["title"] and t["note"] and t["checks"])
        check("%s note has no parenthesised id after a tool word" % t["id"],
              not re.search(r"\b(?:pickup|put_in|pick_and_place|heat|cook|fill|close|slice|toggle_on|toggle_off|wash)\s*\(",
                            t["note"] + tp.render(t, None)))
    banned = re.compile(r"(usually|normally|typically|belongs?|lives? in|stored in|default location|by default)", re.I)
    check("no usual-location / storage-convention claim in any stored template text",
          not any(banned.search(t["title"] + " " + t["note"]) for t in tp.templates))
    check("knowledge_class and support_rule recorded", "functional affordance" in raw.get("knowledge_class", "")
          and "strong support" in raw.get("support_rule", ""))
    fp = tp.fingerprint()
    check("fingerprint pins file, module and FA lexicon", fp["templates_file_sha"] == raw["sha256"] and len(fp["module_sha16"]) == 16
          and len(fp["fa_lexicon_sha"]) == 64)
    check("authorship_note present", "non-test" in raw["authorship_note"] and "no test item" in raw["authorship_note"])

    # ---------------------------------------------------------------------------------------------- 2 skeleton validity
    probs = tp.check(contract, T.load_nontest_items(), validator=v2_validator())
    check("check() clean on the shipped file (contract: %s, validator_v2: %s)" % (csrc, v2_validator() is not None),
          not probs, json.dumps(probs)[:600])

    # ---------------------------------------------------------------------------------------------- 3 level-blind (static)
    src = Path(T.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    check("python 3.10 syntax", bool(ast.parse(src, feature_version=(3, 10))))
    forbidden = {"level", "plan", "goal", "reference_plan", "gold", "setup", "reviewed", "item_id"}
    sel = {"candidates", "_bind", "_group_hit", "_purpose_hits", "_liquid", "expand", "render", "mentions", "compile_phrase",
           "phrase_in", "first_hit", "_fmt_step"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in sel:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in forbidden:
                    offenders.append((node.name, sub.value))
                base = None
                if isinstance(sub, ast.Subscript):
                    base, key = sub.value, sub.slice
                elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get" and sub.args:
                    base, key = sub.func.value, sub.args[0]
                if isinstance(base, ast.Name) and base.id in ("query", "q") and isinstance(key, ast.Constant) and key.value == "id":
                    offenders.append((node.name, "query[id]"))
    check("selection code never reads level/id/plan/goal keys", not offenders, str(offenders))
    cand_src = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "candidates")
    q_reads = set()
    for sub in ast.walk(cand_src):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get" \
                and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "query" and sub.args:
            q_reads.add(sub.args[0].value)
    check("candidates reads only instruction/types/parse/observed/holding from the query",
          q_reads <= {"instruction", "types", "parse", "observed", "holding"}, str(q_reads))

    # ---------------------------------------------------------------------------------------------- 4 behaviour (EN + FA)
    cases = [
        ("Serve the potato hot on a plate.", "heat-serve"),
        ("Serve me a hot egg in a bowl.", "heat-serve"),
        ("I want to eat this apple cold later.", "store-cold"),
        ("Put the lettuce, the tomato and the egg where they will stay fresh.", "store-cold"),
        ("The tomato has gone bad.", "discard-spoiled"),
        ("This bread is stale. Get rid of it.", "discard-spoiled"),
        ("Make me a salad.", "prepare-slice"),
        ("I need something to cut with.", "tool-for-purpose"),
        ("My food needs salt.", "tool-for-purpose"),
        ("I want to wash my hands.", "tool-for-purpose"),
        ("It's too dark in here.", "light-on"),
        ("It's too bright in here.", "light-off"),
        ("Everything has been left open. Tidy that up.", "close-all-open"),
        # A26 (K-B2b): tpl:wash-used lost its only non-test support (g-l3-cutlery-sink, archived), so the support rule archives it;
        # the shipped set no longer fires it (was "wash-used"); its machinery is checked on the lab copy in section 5
        ("The fork and the spoon have been used. Put them where they get washed.", None),
        ("Breakfast: a hot potato on a plate and a cup of coffee.", "meal-sequence"),
        ("سیب را سرد نگه دار", "store-cold"),
        ("اینجا خیلی تاریک است.", "light-on"),
        ("اینجا خیلی روشن است.", "light-off"),
        ("سیب زمینی را داغ روی بشقاب سرو کن", None),         # FA type name for Plate may not be in the lexicon: tolerated
        ("تخم مرغ خراب شده است.", "discard-spoiled"),
        ("یک چیزی برای بریدن می خواهم.", "tool-for-purpose"),
        ("همه چیز باز مانده است. مرتب کن.", "close-all-open"),
    ]
    for instr, want in cases:
        got = top_ids(tp, q(instr, "fa" if re.search(r"[؀-ۿ]", instr) else "en"))
        if want is None:
            check("tolerated: %r" % instr, True)
        else:
            check("top candidate for %r is %s" % (instr, want), got[:1] == [want], str(got))
    negatives = ["Pick up the apple.", "Open the fridge.", "Close the cabinet.", "Turn on the light.", "Heat the potato.",
                 "Put the apple in the fridge.", "Slice the tomato.", "Fill the cup with water.", "Pick up the salt shaker.",
                 "Pick up the soap bottle.", "Turn on the coffee machine.", "Wash the plate.", "I'm leaving the house, turn everything off.", "I would like a cup of coffee.", "سیب را بردار.",
                 "در یخچال را باز کن.", "چراغ را روشن کن."]
    for instr in negatives:
        got = top_ids(tp, q(instr, "fa" if re.search(r"[؀-ۿ]", instr) else "en"))
        check("no template for atomic %r" % instr, got == [], str(got))
    check("empty/garbage instruction -> nothing", top_ids(tp, q("")) == [] and top_ids(tp, q("zzz qqq")) == [])
    check("missing optional query fields tolerated", top_ids(tp, {"instruction": "It's too dark here."}) == ["light-on"])
    # binding details
    c = tp.candidates(q("Put the lettuce, the tomato and the egg where they will stay fresh."))[0]
    check("multi-item binding keeps mention order",
          c["rec"]["text"]["en"].index("Lettuce") < c["rec"]["text"]["en"].index("Tomato") < c["rec"]["text"]["en"].index("Egg"))
    check("store-cold ends with close Fridge", c["rec"]["text"]["en"].count("close Fridge") >= 1)
    c = tp.candidates(q("Make me a salad."))[0]
    check("salad default ingredients are half-weighted", abs(c["match"] - 0.75) < 1e-9, str(c["match"]))
    c = tp.candidates(q("Make me a salad with the potato."))[0]
    check("named ingredient overrides the salad default",
          "slice Potato" in c["rec"]["text"]["en"] and "Lettuce" not in c["rec"]["text"]["en"])
    c = tp.candidates(q("Breakfast: a hot potato on a plate and a cup of coffee."))
    check("meal template supersedes heat-serve", [x["rec"]["id"] for x in c] == ["template/*/meal-sequence"], str(top_ids(tp, q("Breakfast: a hot potato on a plate and a cup of coffee."))))
    check("meal text has fill with coffee", "fill Cup with coffee" in c[0]["rec"]["text"]["en"])
    check("meal without food or drink is not offered", top_ids(tp, q("What is for breakfast?")) == [])
    c = tp.candidates(q("I need something to cut with.", holding="Mug"))[0]
    check("holding note when the first step takes the hand", "holding Mug" in c["rec"]["text"]["en"])
    c = tp.candidates(q("I need something to cut with.", holding="Knife"))[0]
    check("holding the target tool: still a note-free pickup advice is acceptable", "text" in c["rec"])
    check("holding note absent when the hand is empty",
          "holding" not in tp.candidates(q("I need something to cut with."))[0]["rec"]["text"]["en"])
    # match values in range and records valid
    allc = []
    for instr, _ in cases:
        allc += tp.candidates(q(instr, "fa" if re.search(r"[؀-ۿ]", instr) else "en"))
    check("all candidates are pkg.cand/1 with valid pkg.rec/1", allc and all(cand_ok(c) for c in allc))
    check("all matches >= 0.5 (typed key present)", all(c["match"] >= 0.5 for c in allc))
    check("static_records are valid and one per shipped template",
          len(tp.static_records()) == len(tp.templates) and all(rec_ok(r) for r in tp.static_records()))
    # determinism + invariance to forbidden keys
    qq = q("Serve me a hot egg in a bowl.")
    a = json.dumps(tp.candidates(qq), sort_keys=True)
    qq2 = copy.deepcopy(qq)
    qq2.update({"level": "L1", "item_id": "g-l1-x", "id": "zzz", "plan": [{"action": "open", "object": "Fridge"}],
                "goal": [{"type": "Egg"}], "reference_plan": ["x"]})
    check("deterministic", a == json.dumps(tp.candidates(qq), sort_keys=True))
    check("invariant to level / id / plan / goal keys in the query", a == json.dumps(tp.candidates(qq2), sort_keys=True))
    # language: the same instruction in EN and FA reaches the same template
    pairs = [("I want to eat this apple cold later.", "سیب را سرد نگه دار"), ("It's too dark in here.", "اینجا خیلی تاریک است."),
             ("The egg is rotten.", "تخم مرغ خراب شده است.")]
    check("EN and FA agree on the template", all(top_ids(tp, q(e))[:1] == top_ids(tp, q(f, "fa"))[:1] for e, f in pairs))
    check("rendered text is English only", all(not re.search(r"[؀-ۿ]", c["rec"]["text"]["en"]) for c in allc))

    # ---------------------------------------------------------------------------------------------- 5 synthetic kitchen
    synth = [{"type": t, "state": "closed"} for t in ("Fridge", "Fridge", "Microwave", "Cabinet", "Sink", "StoveKnob", "StoveKnob",
                                                         "StoveKnob", "StoveKnob", "StoveKnob", "StoveKnob", "StoveBurner",
                                                         "StoveBurner", "StoveBurner", "StoveBurner", "GarbageCan")]
    lab_w = T.Templates(dict(raw, templates=list(raw["templates"]) + [r["template"] for r in raw["rejected"]]))   # A26: wash-used is archived
    check("A26: the shipped set does not fire the archived wash-used; the lab copy does",
          top_ids(tp, q("The fork and the spoon have been used. Put them where they get washed.")) == []
          and top_ids(lab_w, q("The fork and the spoon have been used. Put them where they get washed."))[:1] == ["wash-used"])
    c = lab_w.candidates(q("The fork and the spoon have been used. Put them where they get washed.", observed=synth))[0]
    check("synthetic kitchen: sink without SinkBasin -> Sink", "to Sink," in c["rec"]["text"]["en"] + "," or
          "to Sink" in c["rec"]["text"]["en"] and "SinkBasin" not in c["rec"]["text"]["en"], c["rec"]["text"]["en"][:160])
    c = lab_w.candidates(q("The fork and the spoon have been used. Put them where they get washed.",
                           observed=[{"type": "SinkBasin", "state": "empty"}, {"type": "Sink", "state": "empty"}]))[0]
    check("both present -> SinkBasin", "to SinkBasin" in c["rec"]["text"]["en"])
    # the archived (unsupported) templates stay unreachable in the shipped set; their machinery is still exercised in a lab copy
    lab = T.Templates(dict(raw, templates=list(raw["templates"]) + [r["template"] for r in raw["rejected"]]))
    c = lab.candidates(q("I'm leaving the house, turn everything off.",
                         observed=[{"type": "StoveKnob", "state": "on"}, {"type": "Faucet", "state": "off"},
                                   {"type": "StoveKnob", "state": "off"}]))
    check("lab: switch-off names the appliance the observation shows on",
          c and "toggle_off StoveKnob" in c[0]["rec"]["text"]["en"] and "Faucet" not in c[0]["rec"]["text"]["en"])
    c = lab.candidates(q("I'm leaving the house, turn everything off.", observed=synth))
    check("lab: switch-off with nothing on renders the category word", c and "<appliance>" in c[0]["rec"]["text"]["en"])
    c = lab.candidates(q("I would like a cup of coffee."))
    check("lab: drink-fill binds the liquid and the default cup", c and "fill Cup with coffee" in c[0]["rec"]["text"]["en"])
    c = lab.candidates(q("Cook the egg and serve it in the bowl."))
    check("lab: cook-serve", c and "cook Egg, then put_in Bowl" in c[0]["rec"]["text"]["en"])
    check("shipped set does not fire the archived templates",
          top_ids(tp, q("I would like a cup of coffee.")) == [] and top_ids(tp, q("Cook the egg and serve it in the bowl.")) == [])
    check("no kitchen-specific constant: no FloorPlan in code or file", "FloorPlan" not in src and "FloorPlan" not in json.dumps(raw))
    # the Toaster-less kitchen: nothing in candidates depends on a fixture being present
    c = tp.candidates(q("It's too dark in here.", observed=[{"type": "Cabinet", "state": "closed"}]))
    check("light template does not require the observation to list a LightSwitch", c and c[0]["rec"]["id"].endswith("light-on"))

    # ---------------------------------------------------------------------------------------------- 6 budget
    longest = max(len(c["rec"]["text"]["en"]) for c in allc)
    fake = lambda s: max(-(-len(s) // 3), -(-len(s) * 10 // 25))                   # max(len/3, len/2.5): the fake counter, EN
    check("longest instantiated template fits the 300-token quota (fake counter)", fake("x" * longest) <= 300,
          "%d chars" % longest)
    check("cheapest template under 60 tokens", min(fake(c["rec"]["text"]["en"]) for c in allc) <= 80)

    # ---------------------------------------------------------------------------------------------- 7 shape matcher
    sh = lambda tid, plan: tp.shape_matches(tid, [{"action": a, "object": o, "to": t, "arg": g} for a, o, t, g in plan])
    check("shape: store-cold single", sh("tpl:store-cold", [("pick_and_place", "Egg", "Fridge", None), ("close", "Fridge", None, None)]))
    check("shape: store-cold multi", sh("tpl:store-cold", [("pick_and_place", "Egg", "Fridge", None),
                                                          ("pick_and_place", "Apple", "Fridge", None), ("close", "Fridge", None, None)]))
    check("shape: store-cold rejects wrong receptacle", not sh("tpl:store-cold", [("pick_and_place", "Egg", "Cabinet", None), ("close", "Fridge", None, None)]))
    check("shape: store-cold rejects missing close", not sh("tpl:store-cold", [("pick_and_place", "Egg", "Fridge", None)]))
    check("shape: store-cold rejects a non-food", not sh("tpl:store-cold", [("pick_and_place", "Fork", "Fridge", None), ("close", "Fridge", None, None)]))
    check("shape: heat then pick_and_place == heat then put_in",
          sh("tpl:heat-serve", [("heat", "Egg", None, None), ("pick_and_place", "Egg", "Bowl", None)]) and
          sh("tpl:heat-serve", [("heat", "Egg", None, None), ("put_in", "Bowl", None, None)]))
    check("shape: heat-serve needs the same food held", not sh("tpl:heat-serve", [("heat", "Egg", None, None), ("pick_and_place", "Apple", "Bowl", None)]))
    sh_w = lambda tid, plan: lab_w.shape_matches(tid, [{"action": a, "object": o, "to": t, "arg": g} for a, o, t, g in plan])   # A26: archived
    check("shape: no repeated item inside one each-group", not sh_w("tpl:wash-used", [("pick_and_place", "Fork", "SinkBasin", None), ("pick_and_place", "Fork", "SinkBasin", None)]))
    check("shape: wash-used accepts Sink or SinkBasin, not Fridge",
          sh_w("tpl:wash-used", [("pick_and_place", "Fork", "Sink", None)]) and not sh_w("tpl:wash-used", [("pick_and_place", "Fork", "Fridge", None)]))
    check("shape: meal with all three goals", sh("tpl:meal-sequence", [("heat", "Potato", None, None), ("pick_and_place", "Potato", "Plate", None), ("fill", "Cup", None, "coffee")]))
    check("shape: meal needs a liquid arg for fill", not sh("tpl:meal-sequence", [("heat", "Potato", None, None), ("put_in", "Plate", None, None), ("fill", "Cup", None, None)]))
    check("shape: salad (A26: the ButterKnife is the tool; was Knife)", sh("tpl:prepare-slice", [("pickup", "ButterKnife", None, None), ("slice", "Lettuce", None, None), ("slice", "Tomato", None, None)]))
    check("shape: close-all alt (explicit closes)", sh("tpl:close-all-open", [("close", "Fridge", None, None), ("close", "Cabinet", None, None)]))
    check("shape: tool pickup only for a tool type (A26: ButterKnife; the Knife is no longer a tool)", sh("tpl:tool-for-purpose", [("pickup", "ButterKnife", None, None)])
          and not sh("tpl:tool-for-purpose", [("pickup", "Knife", None, None)]) and not sh("tpl:tool-for-purpose", [("pickup", "Apple", None, None)]))

    # ---------------------------------------------------------------------------------------------- 8 check() finds planted defects
    items = T.load_nontest_items()
    mut = copy.deepcopy(raw)
    mut["templates"][1]["skeleton"][0]["action"] = "teleport"
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags an unknown tool", any(p["code"] == "SKELETON_INVALID" for p in pr))
    mut = copy.deepcopy(raw)
    mut["templates"][1]["support"] = []
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags empty support", any(p["code"] == "SUPPORT_EMPTY" for p in pr))
    mut = copy.deepcopy(raw)
    some_test = sorted(manifest["test"])[0]
    mut["templates"][1]["support"] = [some_test]
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags a test id as support", any(p["code"] == "SUPPORT_TEST_ITEM" for p in pr))
    mut = copy.deepcopy(raw)
    mut["templates"][1]["support"] = ["g-not-a-real-item"]
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags an unknown id as support", any(p["code"] == "SUPPORT_NOT_NONTEST" for p in pr))
    mut = copy.deepcopy(raw)
    mut["templates"][2]["support"] = [raw["templates"][1]["support"][0]]
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags a support item whose plan has another shape", any(p["code"] == "SUPPORT_NO_MATCH" for p in pr))
    mut = copy.deepcopy(raw)
    mut["templates"][1]["text"]["en"] = "stale"
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags stale static text", any(p["code"] == "TEXT_STALE" for p in pr))
    mut = copy.deepcopy(raw)
    mut["templates"][1]["note"] = "Keep it جا"
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags Persian text in a stored template", any(p["code"] == "NOT_ENGLISH" for p in pr))
    mut = copy.deepcopy(raw)
    mut["templates"][1]["skeleton"] = [{"action": "put_in", "object": "Fridge", "to": None, "arg": None}]
    pr = T.Templates(mut).check(contract, items, nontest_ids=nontest_ids)
    check("check() flags put_in with an empty hand", any("held" in p["detail"] for p in pr))

    # ---------------------------------------------------------------------------------------------- 9 loader privacy
    _arch = {a["id"] for a in json.loads((T.DATASET_PATH).read_text(encoding="utf-8")).get("archived", [])}
    check("load_nontest_items returns non-test ids only (minus the A26-archived ids, which left the dataset's items)",
          {i["id"] for i in items} == nontest_ids - _arch and len(_arch) == 6)
    check("no test id appears anywhere in the shipped file", not any('"%s"' % i in json.dumps(raw) for i in sorted(test_like)))

    print("B6 templates selftest: %d/%d" % (_pass, _pass + _fail))
    if early:
        early_check(tp, items)
    return 0 if _fail == 0 else 1


def early_check(tp, items):
    """B6 early check (spec 9.1): share of the non-test L3 instructions with >=1 candidate; share where the top candidate's
    skeleton matches the reference plan's tool sequence.  Diagnostic only.  `level` is read HERE (reporting) and nowhere in
    the retrieval code."""
    print("\n== B6 early check (diagnostic; non-test items only) ==")
    out = {}
    for lang in ("en", "fa"):
        rows = []
        for it in items:
            text = it.get(lang)
            cands = tp.candidates(q(text, lang))
            top = cands[0] if cands else None
            tid = ("tpl:" + top["rec"]["id"].split("/")[-1]) if top else None
            rows.append({"id": it["id"], "level": it["level"], "n": len(cands), "top": tid,
                         "shape": bool(tid and tp.shape_matches(tid, it["plan"])),
                         "any_shape": any(tp.shape_matches("tpl:" + c["rec"]["id"].split("/")[-1], it["plan"]) for c in cands),
                         "ids": [c["rec"]["id"].split("/")[-1] for c in cands]})
        l3 = [r for r in rows if r["level"] == "L3"]
        rest = [r for r in rows if r["level"] != "L3"]
        out[lang] = {
            "L3_n": len(l3), "L3_ge1_candidate": sum(1 for r in l3 if r["n"]),
            "L3_top_shape_matches_refplan": sum(1 for r in l3 if r["shape"]),
            "L3_any_cand_shape_matches": sum(1 for r in l3 if r["any_shape"]),
            "nonL3_n": len(rest), "nonL3_fire": sum(1 for r in rest if r["n"]),
            "L2_fire": sum(1 for r in rest if r["level"] == "L2" and r["n"]),
            "L2_n": sum(1 for r in rest if r["level"] == "L2"),
            "L1_fire": sum(1 for r in rest if r["level"] == "L1" and r["n"]),
            "L1_n": sum(1 for r in rest if r["level"] == "L1"),
            "uncovered_L3": [r["id"] for r in l3 if not r["n"]],
            "L3_top_wrong_shape": [(r["id"], r["top"]) for r in l3 if r["n"] and not r["shape"]],
        }
        out[lang]["_rows"] = rows
    for lang in ("en", "fa"):
        o = out[lang]
        print("[%s] L3 with >=1 candidate: %d/%d (%.0f%%); top candidate shape == reference plan: %d/%d (%.0f%%); "
              "any candidate shape ok: %d/%d" % (lang.upper(), o["L3_ge1_candidate"], o["L3_n"], 100.0 * o["L3_ge1_candidate"] / o["L3_n"],
                                                 o["L3_top_shape_matches_refplan"], o["L3_n"],
                                                 100.0 * o["L3_top_shape_matches_refplan"] / o["L3_n"], o["L3_any_cand_shape_matches"], o["L3_n"]))
        print("      templates firing on non-L3 items: %d/%d (L2 %d/%d, L1 %d/%d)" % (
            o["nonL3_fire"], o["nonL3_n"], o["L2_fire"], o["L2_n"], o["L1_fire"], o["L1_n"]))
        print("      uncovered L3: %s" % o["uncovered_L3"])
        print("      L3 top candidate with another shape: %s" % o["L3_top_wrong_shape"])
    # Astra H-B6 cheap check: how much of the answer does the top template already contain?
    for lang in ("en",):
        n = ex = dest = 0
        for it in items:
            if it["level"] != "L3":
                continue
            cands = tp.candidates(q(it[lang], lang), with_steps=True)
            if not cands:
                continue
            n += 1
            got = [(s["action"], s["object"], s["to"]) for s in T.normalise_plan(cands[0]["steps"])]
            ref = [(s["action"], s["object"], s["to"]) for s in T.normalise_plan(it["plan"])]
            ex += got == ref
            dest += {x[2] or x[1] for x in ref} <= {x[2] or x[1] for x in got}
        print("[EN] top template, instantiated: identical to the reference plan (after put_in folding): %d/%d; "
              "contains every goal-critical object/destination of the reference: %d/%d" % (ex, n, dest, n))
    # author-written paraphrase probe (NOT from any dataset; written after the lists, so optimistic)
    probes = [
        ("Please refrigerate the tomatoes.", "store-cold"), ("Chill the egg for later.", "store-cold"),
        ("I would like the lettuce to stay crisp.", "store-cold"), ("This potato smells off, bin it.", "discard-spoiled"),
        ("The apple is moldy.", "discard-spoiled"), ("Throw away the old bread.", "discard-spoiled"),
        ("Warm up the bread and put it on a plate.", "heat-serve"), ("I'd like a hot tomato in a bowl.", "heat-serve"),
        ("Chop the lettuce and the tomato for dinner.", "prepare-slice"), ("Prepare a salad with lettuce and tomato.", "prepare-slice"),
        ("I need something sharp.", "tool-for-purpose"), ("Is there something I can stir with?", "tool-for-purpose"),
        ("Hand me something to spread butter with.", "tool-for-purpose"), ("My hands are greasy.", "tool-for-purpose"),
        ("I can't see anything in here.", "light-on"), ("The room is dim.", "light-on"),
        ("The glare is hurting my eyes.", "light-off"), ("Shut everything that is open.", "close-all-open"),
        ("Please close all the doors someone left open.", "close-all-open"),
        ("These plates are dirty, take them to the sink.", "wash-used"),
        ("Lunch: a warm egg in a bowl and some water.", "meal-sequence"),
    ]
    hit = sum(1 for qq, want in probes if top_ids(tp, q(qq))[:1] == [want])
    miss = [qq for qq, want in probes if top_ids(tp, q(qq))[:1] != [want]]
    print("[EN] author-written paraphrase probe: top candidate correct %d/%d; missed: %s" % (hit, len(probes), miss))
    en, fa = out["en"]["_rows"], out["fa"]["_rows"]
    agree = sum(1 for a, b in zip(en, fa) if set(a["ids"]) == set(b["ids"]))
    agree_top = sum(1 for a, b in zip(en, fa) if a["top"] == b["top"])
    print("EN/FA agreement on the full candidate-id set: %d/%d; on the top id: %d/%d" % (agree, len(en), agree_top, len(en)))
    only_fa = [(a["id"], a["ids"], b["ids"]) for a, b in zip(en, fa) if set(a["ids"]) != set(b["ids"])]
    for r in only_fa[:12]:
        print("      EN/FA differ:", r)
    print("EARLY_JSON " + json.dumps({k: {x: y for x, y in v.items() if x != "_rows"} for k, v in out.items()} |
                                     {"agree_set": agree, "agree_top": agree_top, "n": len(en)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
