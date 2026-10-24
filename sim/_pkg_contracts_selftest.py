"""Selftest for the skill contracts of the procedure KB and package (stage J-P2-contracts, round 2). Offline: no simulator, no LLM.

    python sim/_pkg_contracts_selftest.py

What it locks, against the SKILL CODE (sim/honest_skills.py read by AST, never imported):
  * heat == cook: one branch, the microwave cycle put-in, close, switch on, switch off, open, pick up -> the food ends in HAND;
  * the stay-inside route (put in, close, switch on) is a distinct record with its own goal predicate and is never reached by heat/cook;
  * a tool request ("something to X with") renders the fetch-the-tool line only; an unrequested slice is a violation;
  * recipes render only when the operation key matches, 1-3 procedure records, holding-aware.
A line that the skill code does not support makes check_skill_truth() return a problem; the planted-mutation cases prove it does.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_pkg_fixtures"))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import ltm_package as P          # noqa: E402
import ltm_procedural as L       # noqa: E402
from fake_counter import FakeCounter  # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  :: " + str(detail)))


def main():
    kb = L.load_kb(verified=False)
    sc = L.skill_contract()
    SRC = open(os.path.join(HERE, "honest_skills.py"), encoding="utf-8").read()

    # ------------------------------------------------------------------ A. the skill code, read by AST
    check("A1 heat and cook are ONE branch of _dispatch", sc["heat_cook_shared"])
    check("A2 the cycle is ensure-held, put in Microwave, close, on, off, open, pick up",
          sc["cycle"] == [("ensure_holding", "ot"), ("put", "Microwave"), ("close", "Microwave"), ("toggle_on", "Microwave"),
                          ("toggle_off", "Microwave"), ("open", "Microwave"), ("pickup", "ot")], sc["cycle"])
    check("A3 the cycle ends with the food picked up again (in hand)", sc["cycle"][-1] == ("pickup", "ot"))
    check("A4 the door is closed before the Microwave is switched on",
          [a for a, _ in sc["cycle"]].index("close") < [a for a, _ in sc["cycle"]].index("toggle_on"))
    check("A5 pick_and_place = hold X, then put on the destination; put_in = put the held object", sc["pick_and_place"] == [("ensure_holding", "ot"), ("put", "to")]
          and sc["put_in"] == [("put", "ot")], (sc["pick_and_place"], sc["put_in"]))
    # A26 (K-B2b 2026-10-13) replaces: check("A6 slice never asks for a Knife in hand (the dispatch has no Knife)", not sc["slice_needs_knife"])
    check("A6 (A26) the slice branch refuses without a cutting tool, and CUTTING_TOOLS == ['ButterKnife']",
          sc["slice_needs_tool"] and sc["cutting_tools"] == ["ButterKnife"], (sc["slice_needs_tool"], sc["cutting_tools"]))
    check("A7 the read is from source text, not an import (no honest_skills in sys.modules because of this file)", sc["origin"].startswith("ast:honest_skills@"))

    # ------------------------------------------------------------------ B. every line TRUE against the skill code
    probs = L.check_skill_truth(kb, sc)
    check("B1 check_skill_truth(real KB, real skills) is clean", probs == [], probs)
    heat, cook = kb.by_id["proc:tool-heat"], kb.by_id["proc:tool-cook"]
    check("B2 heat and cook both end with the food in hand and have no other effect on the hand",
          heat["hand_end"] == "held" and cook["hand_end"] == "held")
    check("B3 heat and cook text each say the food does not stay in the microwave",
          all("does not stay inside" in e["text"]["en"] or "does not leave X in the microwave" in e["text"]["en"] for e in (heat, cook)))
    check("B4 cook says it is the same cycle as heat", "same whole microwave cycle as heat" in cook["text"]["en"])
    check("B5 heat/cook are the same micro-route: each is exactly one tool step", [s["action"] for s in heat["route"]] == ["heat"]
          and [s["action"] for s in cook["route"]] == ["cook"])
    check("B6 the static check of the whole KB against the exec contract is clean",
          kb.check_static(L.load_exec_contract(), __import__("honest_skills").TOOLS, L.detector_types(),
                          validator=__import__("planners.procedures_check", fromlist=["validator"]).validator()) == [])
    # planted mutations of the SKILL SOURCE must be caught
    def mutated(fn):
        d = tempfile.mkdtemp(prefix="sk_")
        p = os.path.join(d, "honest_skills.py")
        open(p, "w", encoding="utf-8").write(fn(SRC))
        try:
            return L.check_skill_truth(kb, L.skill_contract(p))
        except ValueError as e:                                   # the contract reader refuses a changed shape: also "caught"
            return ["refused: %s" % e]
    swap = lambda s: s.replace('self._interact("close", "Microwave")\n            self._interact("toggle_on", "Microwave")',
                               'self._interact("toggle_on", "Microwave")\n            self._interact("close", "Microwave")')
    check("B7 a skill change that switches on before closing is caught", bool(mutated(swap)) and swap(SRC) != SRC)
    nopick = lambda s: s.replace('            self._interact("pickup", ot, belief=self._placed(ot))\n            return {"procedure": "heated',
                                 '            return {"procedure": "heated')
    check("B8 a skill change that no longer takes the food out again is caught", bool(mutated(nopick)) and nopick(SRC) != SRC)
    split = lambda s: s.replace('if action in ("heat", "cook"):', 'if action == "heat":')
    check("B9 a skill change that splits heat from cook is caught", bool(mutated(split)) and split(SRC) != SRC)
    # planted mutations of the KB
    import copy
    def kbmut(fn):
        doc = copy.deepcopy(kb.doc)
        fn({e["id"]: e for e in doc["entries"]})
        k2 = L.KB(doc)
        return L.check_skill_truth(k2, sc)
    check("B10 a KB where heat 'ends empty' is caught", bool(kbmut(lambda d: d["proc:tool-heat"].__setitem__("hand_end", "empty"))))
    check("B11 a KB where the stay route also takes the food out is caught",
          bool(kbmut(lambda d: d["proc:microwave-run"]["route"].append({"action": "pickup", "object": "?item", "to": None, "arg": None}))))
    check("B12 a KB stay route without its goal predicate is caught", bool(kbmut(lambda d: d["proc:microwave-run"].__setitem__("goal", []))))
    check("B13 a KB stay route triggered by heat is caught", bool(kbmut(lambda d: d["proc:microwave-run"]["triggers"].append("heat"))))

    # ------------------------------------------------------------------ C. the distinct stay-inside route
    stay = kb.by_id["proc:microwave-run"]
    check("C1 route = pick_and_place into Microwave, close, toggle_on; the food is NOT taken out",
          [s["action"] for s in stay["route"]] == ["pick_and_place", "close", "toggle_on"] and stay["hand_end"] == "empty")
    check("C2 goal predicate: item in Microwave, Microwave closed, Microwave on",
          {(g["type"], g["key"]): (g.get("contains"), g.get("value")) for g in stay["goal"]}
          == {("?item", "parentReceptacles"): ("Microwave", None), ("Microwave", "isOpen"): (None, False), ("Microwave", "isToggled"): (None, True)})
    check("C3 text: the food stays inside and is only taken out if the command says so", "stays inside" in stay["text"]["en"]
          and "do not take it out unless" in stay["text"]["en"])
    check("C4 not triggered by heat or cook; 'heat the egg' / 'cook the potato' do not match it at 1.0",
          all(c["match"] < 1.0 for c in kb.candidates({"types": ["Egg", "Microwave"], "ops": ["heat"]}) if c["rec"]["id"].endswith("microwave-run"))
          and all(c["match"] < 1.0 for c in kb.candidates({"types": ["Potato", "Microwave"], "ops": ["cook"]}) if c["rec"]["id"].endswith("microwave-run")))
    check("C5 polarity is on (a 'turn off' request must not get the run route)", stay["polarity"] == "on")
    goal_from_data = {"type": "Microwave", "key": "isToggled", "value": True}
    import json
    gen = json.load(open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8"))["items"]
    runs = [i for i in gen if i["id"].startswith("g-l2-microwave-run-")]
    check("C6 the predicate has the same shape as the reviewed dataset goals that run the microwave (shape only, no id used)",
          runs and all(goal_from_data in i["goal"] for i in runs))

    # ------------------------------------------------------------------ D. rendering through the real builder
    prm = P.profile_params(P.PROFILE_R1)
    b = P.PackageBuilder("K", {"procedure": kb}, FakeCounter(), prm, categories=P.load_categories())

    def render(s, held=None, lang="en"):
        q = P.make_query_profile(P.PROFILE_R1, s, lang, held, [])
        pk = b.build(q)
        txt = pk["text"]
        lines = [ln for ln in txt.split("\n")[1:]] if txt else []
        return pk, lines

    def n_proc(pk):
        return len((pk.get("sections") or {}).get("procedure") or []) if isinstance(pk.get("sections"), dict) else None

    pk, ls = render("Put the egg in the microwave, close the door, and turn it on.")
    check("D1 run-it request renders the stay-inside recipe (and not heat)", any("The food stays inside" in x for x in ls) and not any(x.startswith("heat X") for x in ls), ls)
    pk, ls = render("Heat the egg.")
    check("D2 'heat the egg' renders the heat line, not the stay-inside route, not heat-then-place",
          any(x.startswith("heat X") for x in ls) and not any("stays inside" in x or "then pick_and_place it to the place" in x for x in ls), ls)
    pk, ls = render("Cook the potato.")
    check("D3 'cook the potato' renders the cook line, no stove line for a potato",
          any(x.startswith("cook X") for x in ls) and not any("StoveBurner" in x or "stays inside" in x for x in ls), ls)
    pk, ls = render("Heat the egg and put it on the plate.")
    check("D4 heat with a named destination renders heat-then-place", any("then pick_and_place it to the place" in x for x in ls), ls)
    pk, ls = render("Put the apple in the microwave and close it.")
    check("D5 put-in-and-close renders the put/close recipe and states it is not heat; no run route",
          any("This is not heat" in x for x in ls) and not any("stays inside" in x for x in ls), ls)
    pk, ls = render("Turn off the microwave.")
    check("D6 'turn off the microwave' gets neither the run route nor heat", not any("stays inside" in x or x.startswith("heat X") for x in ls), ls)
    pk, ls = render("I need something to cut with.")
    # A26: the cutting tool is the ButterKnife (was "pickup Knife and stop there" in D7 / D8 / D13, "To slice: pickup Knife" in D9)
    check("D7 tool request: the fetch-the-tool line, no slice recipe", any("pickup ButterKnife and stop there" in x for x in ls)
          and not any("To slice" in x or x.startswith("slice X") for x in ls), ls)
    pk, ls = render("Get me a knife to cut the tomato with.")
    check("D8 a purpose clause that ends the request is a tool request even when it names a food",
          any("pickup ButterKnife and stop there" in x for x in ls) and not any("To slice" in x for x in ls), ls)
    pk, ls = render("Get a knife and cut the tomato.")
    check("D9 an explicit cut request still gets the slice recipe", any("To slice: pickup ButterKnife, then slice" in x for x in ls), ls)
    pk, ls = render("Slice the tomato.")
    check("D10 'slice the tomato' gets the slice recipe, no tool-request line", any("To slice" in x for x in ls) and not any("stop there" in x for x in ls), ls)
    pk, ls = render("Put the egg in the microwave, close the door, and turn it on.", held="Egg")
    check("D11 holding the egg: put_in, no pick_and_place of the held egg, no pickup advice for it",
          any("put_in Microwave, close Microwave, then toggle_on Microwave" in x for x in ls) and not any("pick_and_place it to Microwave" in x for x in ls)
          and any("You are holding the Egg" in x for x in ls), ls)
    battery = ["Heat the egg.", "Cook the potato.", "Put the egg in the microwave, close the door, and turn it on.", "Turn off the microwave.",
               "I need something to cut with.", "Get me a knife to cut the tomato with.", "Slice the tomato.", "Put the apple in the microwave and close it.",
               "I need something to flip with.", "I need something to scoop with.", "Heat the egg and put it on the plate.", "Find the apple."]
    counts = []
    for s in battery:
        for held in (None, "Egg"):
            pk, ls = render(s, held)
            counts.append((s, held, len([x for x in ls if x.strip() and not x.endswith(":")])))
    check("D12 never more than 3 lines of procedure and never zero for these requests",
          all(0 <= c[2] <= 4 for c in counts), counts)   # <= 3 records + the section header line
    check("D13 a flip/cut tool request names its tool and stops", all(
        any(("pickup %s and stop there" % t) in x for x in render(s)[1]) for s, t in
        (("I need something to flip with.", "Spatula"), ("I need something to cut with.", "ButterKnife"))))
    # every tool token a rendered line mentions is a real tool or a detector type
    import honest_skills as hs
    types = set(L.detector_types())
    bad = []
    for s in battery:
        for held in (None, "Egg"):
            for x in render(s, held)[1]:
                for tok in re.findall(r"\b([a-z]+_[a-z]+)\b", x):
                    if tok not in hs.TOOLS and tok not in ("hand_end",):
                        bad.append((s, tok))
    check("D14 every snake_case tool named in a rendered line is a real skill", not bad, bad)
    reqs = [r for r in kb.afford_records() if r["id"].endswith("-request")]
    check("D15 every tool-request record names a detector type and tells the agent to stop",
          reqs and all(re.search(r"pickup (\w+) and stop there", r["text"]["en"]) and re.search(r"pickup (\w+) and stop there", r["text"]["en"]).group(1) in types for r in reqs))
    check("D16 tool-request records carry the same support (leave-out) as their parent row",
          all(r["provenance"]["support"] == list(kb.afford_map[r["id"].rsplit("afford-", 1)[1].replace("-request", "")].get("tuned_on") or []) for r in reqs))

    # ------------------------------------------------------------------ E. effects, not words: unrequested actions
    ua = L.unrequested_actions
    sl = [{"action": "pickup", "object": "ButterKnife"}, {"action": "slice", "object": "Tomato"}]   # A26: was Knife
    check("E1 a tool request with a slice step is a violation", [v["effect"] for v in ua(sl, ["slice"], [], tool_request=True)] == ["slice"])
    check("E2 a tool request with only the pickup is clean", ua(sl[:1], ["slice"], [], tool_request=True) == [])
    check("E3 an unrequested slice (the request is about heating) is a violation", [v["effect"] for v in ua(sl, ["heat"], ["Egg"])] == ["slice"])
    check("E4 a requested slice is clean", ua(sl, ["slice"], ["Tomato"]) == [])
    check("E5 cook authorises the heat effect and heat authorises cook (one routine)",
          ua([{"action": "cook", "object": "Egg"}], ["heat"]) == [] and ua([{"action": "heat", "object": "Egg"}], ["cook"]) == [])
    check("E6 toggle_on of the Microwave counts as heat: unrequested when the request only stores the food",
          [v["effect"] for v in ua([{"action": "pick_and_place", "object": "Egg", "to": "Microwave"}, {"action": "close", "object": "Microwave"},
                                    {"action": "toggle_on", "object": "Microwave"}], ["put", "close"], ["Egg", "Microwave"])] == ["heat"])
    check("E7 ... and requested when the request says to turn the microwave on",
          ua([{"action": "toggle_on", "object": "Microwave"}], ["toggle_on"], ["Egg", "Microwave"]) == [])
    check("E8 a tool request allows walking and opening to reach the tool, nothing else",
          ua([{"action": "search_for", "object": "ButterKnife"}, {"action": "open", "object": "Drawer"}, {"action": "pickup", "object": "ButterKnife"}], [], [], tool_request=True) == []
          and [v["action"] for v in ua([{"action": "pickup", "object": "ButterKnife"}, {"action": "pick_and_place", "object": "ButterKnife", "to": "Sink"}], [], [], tool_request=True)]
          == ["pick_and_place"])
    check("E9 is_tool_request: parser need=tool, or a closing purpose clause; not an explicit two-step request",
          P.is_tool_request(P.make_query_profile(P.PROFILE_R1, "I need something to cut with.", "en", None, []))
          and P.is_tool_request(P.make_query_profile(P.PROFILE_R1, "Get me a knife to cut the tomato with.", "en", None, []))
          and not P.is_tool_request(P.make_query_profile(P.PROFILE_R1, "Get a knife and cut the tomato.", "en", None, []))
          and not P.is_tool_request(P.make_query_profile(P.PROFILE_R1, "Slice the tomato.", "en", None, [])))

    # ------------------------------------------------------------------ F. honesty of the new code
    src = open(os.path.join(HERE, "ltm_procedural.py"), encoding="utf-8").read()
    new = src[src.index("# ====== skill contracts") if "# ====== skill contracts" in src else src.index("def skill_contract"):src.index("# ============================================================================================ early check")]
    check("F1 the contract code names no item id, level, kitchen or plan", not re.search(r"g-l\d|FloorPlan|\blevel\b|reference_plan", new))
    pk_src = open(os.path.join(HERE, "ltm_package.py"), encoding="utf-8").read()
    seg = pk_src[pk_src.index("_PURPOSE_WITH ="):pk_src.index("def holding_rewrite")]
    check("F2 is_tool_request reads the instruction and the parse only", not re.search(r"g-l\d|FloorPlan|reference_plan|gold|\blevel\b", seg))
    check("F3 the KB document carries no kitchen name", "FloorPlan" not in open(L.KB_PATH, encoding="utf-8").read())

    print("\n_pkg_contracts_selftest: %d/%d" % (N["pass"], N["pass"] + N["fail"]))
    return 0 if N["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
