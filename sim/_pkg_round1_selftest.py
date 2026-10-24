"""Selftest for the round-1 integration (stage I-round1-integrate): package PROFILES, the affordance route, schema-aware empty reasons,
recipe subsumption, and the signal measure's independent contradiction checker. Offline: no simulator, WSL, Ollama or API.

    python sim/_pkg_round1_selftest.py

Synthetic kitchen only for the builder checks (helpers of _pkg_package_selftest / _pkg_precision_selftest).
"""
from __future__ import annotations

import ast
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_pkg_fixtures"))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
sys.path.insert(0, os.path.join(HERE, "final_test", "signal"))
import ltm_package as P        # noqa: E402
import pkg_schema as S         # noqa: E402
import _pkg_package_selftest as T  # noqa: E402
import _pkg_precision_selftest as PT  # noqa: E402
from _pkg_package_selftest import check, CATS  # noqa: E402


class AffordKB(PT.SmartKB):
    """The smart fake KB plus the affordance API the real procedure KB has (afford_key / afford / afford_records / by_id)."""
    WORDS = {"chill": "chill", "cold": "chill", "flip": "flip"}

    def __init__(self, cands, by_id=None):
        super().__init__(cands)
        self.by_id = by_id or {}

    def afford_key(self, w):
        return self.WORDS.get(str(w or "").lower())

    def afford(self, w):
        k = self.afford_key(w)
        if k == "chill":
            return {"key": k, "tools": [], "appliances": ["Fridge"], "recipe": "proc:chill", "text": "x"}
        if k == "flip":
            return {"key": k, "tools": ["Spatula"], "appliances": [], "recipe": None, "text": "x"}
        return None

    def afford_records(self):
        r = PT.prec_rec("afford-flip", "To flip or turn food over use the Spatula: pickup Spatula first.", ["flip"], ["Spatula"])
        r["id"] = "procedure/*/afford-flip"
        return [r]


def intent_query(instruction, types, ops, key, aff=(), fallback=False, status="ok", holding=None):
    qq = PT.query(instruction, types, ops, holding=holding)
    qq["intent"] = {"status": status, "key": key, "affordance_types": list(aff), "ops": list(ops), "types": list(types),
                    "holding": holding, "fallback": fallback, "reason": ""}
    return qq


def main():
    # -- profiles ------------------------------------------------------------------------------------------------------------
    shipped = P.load_params()
    leg = P.profile_params(P.PROFILE_LEGACY)
    r1 = P.profile_params(P.PROFILE_R1)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("enabled", "profile", "goal_schema", "sha256")}
    check("R1 the legacy profile is the shipped params (only enabled / profile / goal_schema / sha differ)", strip(leg) == strip(shipped)
          and leg["enabled"] and leg["goal_schema"] is False and leg["profile"] == "pkg-1" and leg["budget"] == 2000, strip(leg).keys())
    check("R2 the r1 profile: precision block on, budget 300, goal schema on, neutral headers", r1["precision"]["enabled"] and r1["budget"] == 300
          and r1["goal_schema"] is True and "How the tools work" not in json.dumps(r1["headers"]) and r1["headers"]["en"]["procedure"] == "Steps that work here:", r1["headers"])
    check("R3 both profile param sets are valid and carry a matching sha", all(P.check_params(x) == [] and x["sha256"] == P.params_sha(x) for x in (leg, r1)),
          [P.check_params(x) for x in (leg, r1)])
    check("R4 the shipped file is untouched: switch stays off, no precision block, sha verifies", shipped["enabled"] is False and "precision" not in shipped
          and shipped["sha256"] == P.params_sha(shipped))
    try:
        P.profile_params("pkg-9")
        bad = False
    except ValueError:
        bad = True
    check("R5 an unknown profile name is refused", bad)
    check("R6 profile_params does not mutate its base", P.profile_params(P.PROFILE_R1, base=shipped) and "precision" not in shipped and shipped["budget"] == 2000)

    # -- queries per profile --------------------------------------------------------------------------------------------------
    import inspect
    check("R7 make_query keeps its pinned signature (no level / item / plan parameter)", list(inspect.signature(P.make_query).parameters)
          == ["instruction", "lang", "holding", "summary_lines", "lexicon"])
    ql = P.make_query_profile(P.PROFILE_LEGACY, "I want to eat this apple cold later.", "en", None, [])
    q1 = P.make_query_profile(P.PROFILE_R1, "I want to eat this apple cold later.", "en", None, [])
    check("R8 legacy profile = the legacy query (no intent); r1 = the goal-schema query", "intent" not in ql and "chill" in q1["ops"] and "Fridge" in q1["intent"]["affordance_types"]
          and ql == P._make_query_legacy("I want to eat this apple cold later.", "en", None, []), (ql.get("ops"), q1.get("ops")))
    qf = P.make_query_profile(P.PROFILE_R1, "I need something to flip the pancakes with.", "en", None, [])
    check("R9 'something to flip with' reads as a tool need whose affordance type is the Spatula (not a goal)", "flip" in qf["ops"] and qf["intent"]["affordance_types"] == ["Spatula"]
          and not qf["goal_states"], qf["intent"])
    try:
        P.make_query_profile("x", "a", "en", None, [])
        bad = False
    except ValueError:
        bad = True
    check("R10 make_query_profile refuses an unknown profile", bad)

    # -- affordance route -----------------------------------------------------------------------------------------------------
    chill = PT.prc("chill", "To chill food: pick_and_place it to Fridge, then close Fridge.", ["close", "put"], ["Fridge"])
    kbx = AffordKB(PT.PROC)
    qc = intent_query("I want to eat this apple cold later.", ("Apple",), ("chill", "eat"),
                      [{"op": "eat", "obj": "Apple", "mood": "desire"}, {"op": "chill", "obj": "Apple", "via_state": "cold"}], aff=("Fridge",))
    pc = PT.build(qc, procs=kbx)
    check("R11 a word with a recipe in the affordance map makes the recipe a direct match even when the parser op list lacks its trigger",
          "To chill food" in pc["text"] and not pc["empty"], pc["text"])
    qno = copy.deepcopy(qc)
    qno["intent"]["key"][1]["neg"] = True
    pno = PT.build(qno, procs=AffordKB(PT.PROC))
    check("R12 a negated item does not route", "To chill food" not in pno["text"], pno["text"])
    qfl = intent_query("I need something to flip the pancakes with.", (), ("flip",), [{"op": "flip", "obj": None, "need": "tool"}], aff=("Spatula",))
    pfl = PT.build(qfl, procs=AffordKB(PT.PROC))
    check("R13 a word without a recipe brings its affordance line (and the line names the action)", "Spatula" in pfl["text"] and "pickup Spatula" in pfl["text"], pfl["text"])
    pleg = PT.build(qc, procs=AffordKB(PT.PROC), prm=T.params())
    check("R14 the legacy params never use the affordance route", "To chill food" not in pleg["text"] and "precision" not in pleg, pleg["text"])
    qnoint = PT.query("I want the apple cold", ("Apple",), ("eat",))
    check("R15 a query without an intent keeps the plain candidates (no affordance route)", "To chill food" not in PT.build(qnoint, procs=AffordKB(PT.PROC))["text"])

    # -- empty reason from the schema verdict ---------------------------------------------------------------------------------
    qe = intent_query("sing a song", (), (), [], status="ok", fallback=False)
    qe["parser_fallback"] = True
    qe["parse"]["ok"] = False
    pe = PT.build(qe, procs=PT.kb())
    check("R16 the schema read the request, nothing matched: 'parsed_zero_matches', not 'parser_failed'", pe["empty"] and pe["empty_reason"] == "parsed_zero_matches", pe["empty_reason"])
    qe2 = intent_query("sing a song", (), (), [], status="unsupported", fallback=True)
    qe2["parser_fallback"] = True
    qe2["parse"]["ok"] = False
    pe2 = PT.build(qe2, procs=PT.kb())
    check("R17 the schema could not read it: 'parser_failed_no_match' and flagged unsupported_request", pe2["empty"] and pe2["empty_reason"] == "parser_failed_no_match"
          and pe2["precision"]["unsupported_request"], (pe2["empty_reason"], pe2.get("precision")))

    # -- subsumption and close_all --------------------------------------------------------------------------------------------
    put_close = PT.prc("microwave-put-close", "To put something in the Microwave: pick_and_place it to Microwave, then close Microwave.", ["close", "put"], ["Microwave"])
    run = PT.prc("microwave-run", "To run the Microwave with something in it: pick_and_place it to Microwave, close Microwave, then toggle_on Microwave.", ["toggle", "put"], ["Microwave"])
    by_id = {"e:put-close": {"route": [{"action": "pick_and_place", "object": "?f", "to": "Microwave"}, {"action": "close", "object": "Microwave", "to": None}]},
             "e:run": {"route": [{"action": "pick_and_place", "object": "?f", "to": "Microwave"}, {"action": "close", "object": "Microwave", "to": None},
                                 {"action": "toggle_on", "object": "Microwave", "to": None}]}}
    put_close["rec"]["provenance"]["evidence"] = ["e:put-close"]
    run["rec"]["provenance"]["evidence"] = ["e:run"]
    close_all = PT.prc("tool-close-all", "close_all: closes every container you have seen open.", ["close", "close_all"])
    qm = PT.query("Put the egg in the microwave, close it and turn it on", ("Egg", "Microwave"), ("close", "put", "toggle"))
    pm = PT.build(qm, procs=AffordKB([put_close, run, close_all] + [c for c in PT.PROC if c["rec"]["id"].endswith(("tool-toggle-on", "tool-pick-and-place"))], by_id))
    lg = {d["id"].split("/")[-1]: d["why"] for d in pm["precision"]["log"]}
    kept = PT.ids(pm)
    check("R18 a recipe that is the leading part of a longer matched recipe is dropped as a duplicate", "microwave-run" in kept and "microwave-put-close" not in kept
          and any(d["id"].endswith("microwave-put-close") and d["reason"] == "duplicate" for d in pm["dropped"]), (kept, pm["dropped"]))
    check("R19 close_all is not offered when a recipe already states close", "tool-close-all" not in kept, kept)
    pm2 = PT.build(qm, procs=AffordKB([put_close, run], {}))
    check("R20 without routes in the source nothing is dropped as subsumed (both recipes stay under the recipe cap)", len([k for k in PT.ids(pm2) if k.startswith("microwave")]) == 2, PT.ids(pm2))

    # -- the real procedure KB through the real goal schema -------------------------------------------------------------------
    import ltm_procedural as LPR
    realkb = LPR.load_kb()
    cats = copy.deepcopy(CATS)
    for txt, lang, hold, must in (("I want to eat this apple cold later.", "en", None, "Fridge"),
                                  ("I need something to flip the pancakes with.", "en", None, "Spatula"),
                                  ("I need something to flip the pancakes with.", "en", None, "pickup Spatula")):
        qq = P.make_query_profile(P.PROFILE_R1, txt, lang, hold, [])
        pk = T.builder({"briefing": list(PT.BRIEF), "procedure": realkb}, prm=P.precision_params(base=T.params())).build(qq)
        check("R21 real KB + real schema: %r is not empty and states %r" % (txt[:34], must), not pk["empty"] and must in pk["text"], pk["text"])
    # EN / FA of one request retrieve the same records
    qa = P.make_query_profile(P.PROFILE_R1, "I want to eat this apple cold later.", "en", None, [])
    qb = P.make_query_profile(P.PROFILE_R1, "می‌خوام این سیب رو سرد بخورم.", "fa", None, [])
    pa = T.builder({"briefing": list(PT.BRIEF), "procedure": realkb}, prm=P.precision_params(base=T.params())).build(qa)
    pb = T.builder({"briefing": list(PT.BRIEF), "procedure": realkb}, prm=P.precision_params(base=T.params())).build(qb)
    check("R22 the package is English for a Persian instruction and the records agree (FA is only the instruction)", pa["rendered_lang"] == pb["rendered_lang"] == "en"
          and not any("؀" <= ch <= "ۿ" for ch in pb["text"]) and pa["kept"] == pb["kept"], (pa["kept"], pb["kept"]))
    check("R23 the packages are valid, under 300 tokens, provenance complete", all(S.validate_pkg(x) == [] and x["tokens"]["max"] <= 300 and P.audit_provenance(x) == [] for x in (pa, pb)))
    # determinism
    pa2 = T.builder({"briefing": list(PT.BRIEF), "procedure": realkb}, prm=P.precision_params(base=T.params())).build(qa)
    check("R24 the build is deterministic (same sha)", pa["sha256"] == pa2["sha256"])

    # -- the honesty of the new code ------------------------------------------------------------------------------------------
    src = open(os.path.join(HERE, "ltm_package.py"), encoding="utf-8").read()
    new_src = src[src.index("def _procedure_with_afford"):src.index("def _dense")] + src[src.index("def _drop_subsumed"):src.index("def _precision")]
    check("R25 the new retrieval code reads no reference plan, level, item id or goal label", not any(w in new_src for w in ('"reference_plan"', '"level"', '"item_id"', '"gold"', "g-l1", "g-l2", "g-l3", "FloorPlan")))
    try:
        ast.parse(src, feature_version=(3, 10))
        ok310 = True
    except SyntaxError:
        ok310 = False
    check("R26 ltm_package.py parses with Python 3.10 syntax (WSL imports it)", ok310)

    # -- the signal measure's contradiction checker ---------------------------------------------------------------------------
    import signal_pkg_r1 as R
    cats2 = {"receptacle_types": ["Cabinet", "CounterTop", "Fridge", "SinkBasin"], "fixture_types": ["Cabinet", "CounterTop", "Fridge", "SinkBasin"]}
    qh = PT.query("Put the bowl on the counter", ("Bowl", "CounterTop"), ("put",), holding="Bowl", dest="CounterTop")
    check("R27 contradictions: refetching the held object is flagged", R.contradictions("To place it: pickup Bowl, then put_in CounterTop.", qh, "Bowl", cats2)
          and R.contradictions("pick_and_place Bowl to CounterTop.", qh, "Bowl", cats2)[0]["rule"] == "holding:refetch")
    check("R28 contradictions: a line that says what to do when already holding is not flagged", not R.contradictions(
        "X is already in your hand: do not pickup it again; go straight to put_in R.", qh, "Bowl", cats2))
    conv = "Usually (general convention, not checked today): dishes and cups (Plate, Bowl, Cup, Mug) are kept in a Cabinet or on a CounterTop."
    check("R29 contradictions: a storage habit naming another receptacle than the stated destination is flagged, not for a search",
          R.contradictions(conv, qh, "Bowl", cats2) and R.contradictions(conv, qh, "Bowl", cats2)[0]["rule"] == "storage:conflicts_destination"
          and not R.contradictions(conv, PT.query("Find the bowl and put it on the counter", ("Bowl", "CounterTop"), ("find", "put"), dest="CounterTop"), None, cats2))
    check("R30 contradictions: a line about other objects is not flagged", not R.contradictions(
        "Usually (general convention, not checked today): cold food (Egg, Lettuce) are kept in a Fridge.", qh, "Bowl", cats2))

    print("\n%d/%d passed" % (T.N["pass"], T.N["pass"] + T.N["fail"]))
    return 0 if T.N["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
