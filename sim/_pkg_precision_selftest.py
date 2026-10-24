"""Selftest for the precision package (stage I-F3-retrieval): sim/ltm_package.py precision pass + sim/kitchen_briefing.py natural lines.
Offline: no simulator, WSL, Ollama or API. Windows `python`.

    python sim/_pkg_precision_selftest.py

Synthetic kitchen only (the helpers of _pkg_package_selftest), so nothing here can depend on a FloorPlan constant.
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_pkg_fixtures"))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import ltm_package as P        # noqa: E402
import kitchen_briefing as KB  # noqa: E402
import pkg_schema as S         # noqa: E402
import _pkg_package_selftest as T  # noqa: E402
from _pkg_package_selftest import rec, cand, FakeKB, FakeProfile, FakeSource, CATS, SYN, check  # noqa: E402
from fake_counter import FakeCounter  # noqa: E402


def brec(kind, t, text, types=None, ops=(), rels=()):
    r = rec("briefing", "%s:%s" % (kind, t), types or [t], ops, rels, text=text, prio=1)
    r["kind"] = kind
    return r


BRIEF = [
    brec("core", "Microwave", "There is one Microwave."),
    brec("core", "CounterTop", "There are 3 CounterTop units."),
    brec("core", "Fridge", "There is one Fridge."),
    brec("exist", "StoveKnob", "There are 4 StoveKnob units."),
    brec("place", "Microwave", "The Microwave is in the south of the kitchen."),
    brec("adj", "Microwave", "The Microwave is next to the CounterTop.", ["Microwave", "CounterTop"], rels=["near:CounterTop"]),
    brec("open", "Microwave", "The Microwave can be opened and closed, and switched on and off.", ops=["open", "close", "toggle_on", "toggle_off"]),
    brec("open", "Fridge", "The Fridge can be opened and closed.", ops=["open", "close"]),
    brec("open", "StoveKnob", "The StoveKnob can be switched on and off.", ops=["toggle_on", "toggle_off"]),
    brec("conv", "food", "Usually (general convention, not checked today): cold food (Egg, Lettuce) are kept in a Fridge.",
         ["Egg", "Lettuce", "Fridge"], rels=["stores:Fridge"]),
    brec("conv", "dishes", "Usually (general convention, not checked today): dishes (Bowl, Plate) are kept in a Cabinet.",
         ["Bowl", "Plate", "Cabinet"], rels=["stores:Cabinet"]),
]


def prec_rec(local, text, ops=(), types=(), rels=()):
    return rec("procedure", local, types, ops, rels, text=text, kitchen="*")


def prc(local, text, ops, types=()):
    r = prec_rec(local, text, ops, types)
    r["id"] = "procedure/*/" + local
    return cand(r, 1.0, ["op:" + (ops[0] if ops else "x")])


PROC = [
    prc("tool-put-in", "put_in R: puts the HELD object into or onto the receptacle R; with nothing held it is an error.", ["put", "put_in"]),
    prc("tool-pick-and-place", "pick_and_place X to R: picks X up and puts it into R; the hand is empty afterwards.", ["pick", "pick_and_place", "put"]),
    prc("tool-pickup", "pickup: walks to the object and picks it up; the hand must be empty first.", ["pick", "pickup"]),
    prc("tool-close", "close X: closes it.", ["close"]),
    prc("tool-close-all", "close_all: closes every container you have seen open.", ["close", "close_all"]),
    prc("tool-toggle-on", "toggle_on X: switches it on.", ["toggle", "toggle_on"]),
    prc("tool-toggle-off", "toggle_off X: switches it off.", ["toggle", "toggle_off"]),
    prc("microwave-put-close", "To run the Microwave: pick_and_place the food to Microwave, close Microwave, then toggle_on Microwave.",
        ["close", "put", "toggle"], ["Microwave"]),
    prc("chill", "To chill food: pick_and_place it to Fridge, then close Fridge.", ["close", "put"], ["Fridge"]),
]


def query(instruction, types, ops, holding=None, fallback=False, goals=(), dest=None):
    qq = T.q(instruction, "en", types, ops, fallback, holding)
    qq["goal_states"] = list(goals)
    if dest:
        qq["parse"]["roles"] = [{"verb": "put", "obj": None, "dest": dest}]
    return qq


class SmartKB(FakeKB):
    """Emulates the real procedure KB: a candidate is a direct match (1.0) only when one of its ops is asked and, for a recipe with
    types, one of those types is in the request; everything else comes back with match 0 (include_zero)."""

    def candidates(self, query, **kw):
        self.seen.append(copy.deepcopy(query))
        qo, qt = set(query.get("ops") or []), set(query.get("types") or [])
        out = []
        for c in self.cands:
            r = c["rec"]
            ro, rt = set(r["keys"]["ops"]), set(r["keys"]["types"])
            hit = bool(ro & qo) and (not rt or bool(rt & qt))
            out.append(dict(c, match=1.0 if hit else 0.0))
        return out

    def expand(self, ops, types, categories=None):
        apps = [t for t in types if t in ("Microwave", "Toaster")]
        recs = [t for t in types if t in ("Fridge", "Cabinet")]
        return {"tools": [], "appliances": apps, "receptacles": recs, "requires": [], "actions": list(ops)}


def kb():
    return SmartKB(PROC)


def build(qq, procs=None, places=None, profile=None, brief=BRIEF, prm=None, **kw):
    src = {"briefing": list(brief), "procedure": procs if procs is not None else kb()}
    if places is not None:
        src["places"] = places
    if profile is not None:
        src["profile"] = profile
    return T.builder(src, prm=prm or P.precision_params(base=T.params()), **kw).build(qq)


def ids(pkg):
    return [i.split("/")[-1] for i in pkg["kept"]]


def main():
    pp = P.precision_params(base=T.params())
    check("P1 precision params valid, hashed, enabled, budget 300", P.check_params(pp) == [] and pp["sha256"] == P.params_sha(pp)
          and pp["precision"]["enabled"] and pp["budget"] == 300, P.check_params(pp))
    check("P2 precision off by default: the shipped params carry no precision block", "precision" not in P.load_params())
    base = build(query("Put the egg in the microwave, close the door and turn it on", ("Egg", "Microwave"), ("close", "put", "toggle")),
                 prm=T.params())
    check("P3 without the block the package has no 'precision' key (original behaviour)", "precision" not in base)

    # -- the microwave request ------------------------------------------------------------------------------------------------
    qm = query("Put the egg in the microwave, close the door and turn it on", ("Egg", "Microwave"), ("close", "put", "toggle"))
    pm = build(qm)
    txt = pm["text"]
    check("P4 the microwave package is valid and under the 300 budget", S.validate_pkg(pm) == [] and pm["tokens"]["max"] <= 300, (S.validate_pkg(pm), pm["tokens"]))
    check("P5 one natural kitchen line: ops of the Microwave, no compass, no 'units', no counts of unrelated fixtures",
          "Microwave (it opens and closes, and switches on and off)" in txt and "south" not in txt and "units" not in txt and "CounterTop" not in txt
          and "Fridge" not in txt.split("Steps")[0], txt)
    check("P6 no generic storage habit (cold food in a Fridge) when the goal names the Microwave as destination", "convention" not in txt, txt)
    check("P7 the recipe is kept and the tool lines it already states are dropped (close, toggle_on, pick_and_place not repeated)",
          "microwave-put-close" in ids(pm) and not any(i in ids(pm) for i in ("tool-close", "tool-toggle-on", "tool-pick-and-place")), ids(pm))
    check("P8 polarity: 'turn it on' never brings toggle_off", "tool-toggle-off" not in ids(pm))
    check("P9 no uncertainty note, no parser note", "Not known" not in txt and "partly understood" not in txt)
    check("P10 provenance is complete for every kept record", P.audit_provenance(pm) == [], P.audit_provenance(pm))
    lg = {d["id"].split("/")[-1]: d["why"] for d in pm["precision"]["log"]}
    check("P11 every drop is logged with a reason", lg.get("core:Microwave") == "folded_into_kitchen_line" and lg.get("conv:food") == "conflicts_destination", lg)

    # -- holding --------------------------------------------------------------------------------------------------------------
    qh = query("You are holding the egg. Put it in the microwave.", ("Egg", "Microwave"), ("put",), holding="Egg")
    ph = build(qh)
    check("P12 holding: the rule line is first, put_in is offered, pickup and pick_and_place are gone",
          [i for i in ids(ph) if i != "fixtures"][0] == "holding-rule" and "tool-put-in" in ids(ph) and "tool-pickup" not in ids(ph) and "tool-pick-and-place" not in ids(ph), ids(ph))
    check("P13 holding: the recipe is rewritten to put_in (never 'pick_and_place ... to')", "pick_and_place the food to" not in ph["text"]
          and "put_in Microwave" in ph["text"], ph["text"])
    check("P14 holding: the rewrite is logged and kept in the record evidence", any(d["why"] == "rewritten_for_holding" for d in ph["precision"]["log"]))
    pn = build(query("Put the apple in the fridge", ("Apple", "Fridge"), ("put",)))
    check("P15 nothing held: put_in is not offered (it needs a held object), pick_and_place is", "tool-put-in" not in ids(pn) and "pick_and_place it to Fridge" in pn["text"], ids(pn))
    check("P16 holding_rewrite is a pure text rule", P.holding_rewrite("a: pick_and_place it to Fridge, then close Fridge.", "Egg") == "a: put_in Fridge, then close Fridge."
          and P.holding_rewrite("pick_and_place it to Fridge", None) == "pick_and_place it to Fridge")

    # -- storage habits, learned places -----------------------------------------------------------------------------------------
    qf = query("Find the lettuce and pick it up", ("Lettuce",), ("find", "pick"))
    pf = build(qf)
    check("P17 a search keeps the storage habit for the searched object", "cold food" in pf["text"], pf["text"])
    pl = rec("places", "at:Lettuce", ["CounterTop", "Lettuce"], (), ["contains:Lettuce"], text="Lettuce was last seen at: CounterTop (3 sightings).")
    pl["id"] = "places/%s/at:Lettuce" % SYN
    prof = rec("profile", "fx:Sink", ["Sink"], (), (), text="Sink: seen at 2 places; objects seen there: Lettuce; last seen in practice attempt 9 of 10.")
    prof2 = rec("profile", "fx:Stove", ["StoveBurner"], (), (), text="StoveBurner: seen at 1 place; last seen in practice attempt 2 of 10.")
    pls = FakeSource([cand(pl, 1.0, ["type:Lettuce"])])
    prf = FakeProfile([cand(prof, 1.0, ["type:Sink"]), cand(prof2, 1.0, ["type:StoveBurner"])])
    pp2 = build(qf, places=pls, profile=prf)
    check("P18 a places answer replaces the generic habit and the profile lines (practice beats convention)",
          "last seen at: CounterTop" in pp2["text"] and "cold food" not in pp2["text"] and "objects seen there" not in pp2["text"]
          and "StoveBurner: seen" not in pp2["text"], pp2["text"])
    qn = query("Put the apple in the fridge", ("Apple", "Fridge"), ("put",))
    pp3 = build(qn, places=pls, profile=prf)
    check("P19 without a search no learned place or profile line is rendered", "last seen" not in pp3["text"] and "seen at" not in pp3["text"], pp3["text"])

    # -- Astra I-F3: the destination veto must not erase the SOURCE habit of a search ----------------------------------------------
    a = build(query("Find the egg and put it in the microwave", ("Egg", "Microwave"), ("find", "put")))
    b = build(query("Find the egg and put it in the fridge", ("Egg", "Fridge"), ("find", "put")))
    c = build(query("Put the egg in the microwave", ("Egg", "Microwave"), ("put",)))
    check("P32 find X and put it in Y: the storage habit for X survives whatever Y is; without a search the conflicting habit is vetoed",
          "cold food" in a["text"] and "cold food" in b["text"] and "cold food" not in c["text"], (a["text"], c["text"]))

    # -- dedup, cap, budget, unsupported ------------------------------------------------------------------------------------------
    dup = [dict(PROC[0]), dict(PROC[0], rec=dict(PROC[0]["rec"], id="procedure/*/tool-put-in-copy"))]
    pd = build(query("Put it away", ("Apple",), ("put",), holding="Apple"), procs=SmartKB(dup))
    check("P20 identical rendered text is kept once", sum(1 for i in ids(pd) if i.startswith("tool-put-in")) == 1, ids(pd))
    many = [prc("tool-x%d" % i, "do%d X: step number %d of the long list." % (i, i), ["put"]) for i in range(8)]
    pc = build(query("Put the apple in the fridge", ("Apple", "Fridge"), ("put",)), procs=SmartKB(many))
    check("P21 at most procedure_max (3) procedure records", sum(1 for i in ids(pc) if i.startswith("tool-x")) <= 3, ids(pc))
    pe = build(query("hello there", (), (), fallback=True))
    check("P22 an unsupported request is an empty package with the flag, not a 'partly understood' line", pe["empty"] and pe["precision"]["unsupported_request"]
          and S.validate_pkg(pe) == [], pe.get("precision"))
    pb = build(qm, prm=P.precision_params(base=T.params(), budget=40))
    check("P23 the hard budget still holds (40 fake tokens)", pb["tokens"]["max"] <= 40 and S.validate_pkg(pb) == [], pb["tokens"])
    check("P24 deterministic: same query, same sha", build(qm)["sha256"] == pm["sha256"])
    check("P25 the query is not mutated and carries no new key", set(qm) == set(T.q()) , sorted(set(qm) ^ set(T.q())))

    # -- bounded expansion -----------------------------------------------------------------------------------------------------
    big = FakeKB([], {"tools": ["Knife", "Spatula", "Fork"], "appliances": ["Microwave", "Toaster", "CoffeeMachine"],
                      "receptacles": ["Fridge", "Cabinet", "Drawer", "Sink"], "actions": ["put"]})
    e0 = P.expand(T.q(types=("Apple",), ops=("put",)), big, CATS)
    e1 = P.expand(T.q(types=("Apple",), ops=("put",)), big, CATS, bound={"tools": 1, "appliances": 2, "receptacles": 2, "profile_receptacles": False})
    check("P26 unbounded expansion is unchanged, bounded expansion is cut per kind",
          len(e0["need"]["appliances"]) == 3 and len(e1["need"]["tools"]) == 1 and len(e1["need"]["appliances"]) == 2
          and len(e1["need"]["receptacles"]) == 2, (e0["need"], e1["need"]))

    # -- the briefing helpers ----------------------------------------------------------------------------------------------------
    f = KB.fixture_facts(BRIEF)
    check("P27 fixture_facts reads count, ops and neighbours from the records", f["Microwave"]["n"] == 1 and f["CounterTop"]["n"] == 3
          and "toggle_on" in f["Microwave"]["ops"] and f["Microwave"]["near"] == ["CounterTop"], f["Microwave"])
    line = KB.natural_fixture_line(f, ["Microwave", "Fridge", "Nothing"])
    check("P28 natural_fixture_line is plain English, skips unknown fixtures, no compass", line.startswith("In this kitchen: one Microwave") and "Nothing" not in line
          and "south" not in line, line)
    check("P29 with_near renders a relative position, off by default", "next to CounterTop" in KB.natural_fixture_line(f, ["Microwave"], with_near=True)
          and "next to" not in line)
    check("P30 prefix='' capitalises the line", KB.natural_fixture_line(f, ["Fridge"], prefix="").startswith("One Fridge"))
    src = open(os.path.join(HERE, "ltm_package.py"), encoding="utf-8").read()
    prec_src = src[src.index("precision package (I-F3)"):src.index("class PackageBuilder:")]
    check("P31 the precision rules contain no reference-plan, level or item-id key", not any(w in prec_src for w in ('"reference_plan"', '"level"', '"item_id"', '"gold"')))

    print("\n%d/%d passed" % (T.N["pass"], T.N["pass"] + T.N["fail"]))
    return 0 if T.N["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
