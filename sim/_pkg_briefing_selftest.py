"""Selftest of sim/kitchen_briefing.py (B2). Offline, Windows python, no simulator / WSL / Ollama / API.

    python sim/_pkg_briefing_selftest.py
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kitchen_briefing as kb                                                  # noqa: E402

N = {"ok": 0, "bad": 0}


def check(name, cond, extra=""):
    if cond:
        N["ok"] += 1
    else:
        N["bad"] += 1
        print("FAIL:", name, extra)


def obj(t, x, z, **k):
    d = {"objectType": t, "position": {"x": x, "y": 0.9, "z": z}, "openable": False, "toggleable": False, "receptacle": False,
         "pickupable": False, "moveable": False, "isOpen": True, "isToggled": True, "objectId": t + "|%+.2f|%+.2f|%+.2f" % (x, 0.9, z),
         "controlledObjects": ["StoveBurner|secret"], "parentReceptacles": None}
    d.update(k)
    return d


def fixture_kitchen():
    """A recorded-style metadata fixture: a mixed kitchen with traps (pickupable / movable clutter, hidden state)."""
    o = [obj("Fridge", -2.1, 1.0, openable=True, receptacle=True, moveable=False),
         obj("Microwave", -1.0, 2.2, openable=True, toggleable=True, receptacle=True, moveable=True),
         obj("CoffeeMachine", -0.4, 2.3, toggleable=True, receptacle=True, moveable=True),
         obj("Sink", 0.6, 2.3), obj("SinkBasin", 0.62, 2.31, receptacle=True, parentReceptacles=["Sink|+00.60|+00.90|+02.30"]),
         obj("Faucet", 0.6, 2.5, toggleable=True)]
    for i, (x, z) in enumerate([(1.0, 2.0), (1.3, 2.0), (1.0, 2.5), (1.3, 2.5)]):
        o.append(obj("StoveBurner", x, z, receptacle=True, parentReceptacles=["CounterTop|+01.00|+00.90|+02.00"]))
        o.append(obj("StoveKnob", x, z + 0.4, toggleable=True))
    o += [obj("CounterTop", 0.0, 2.4, receptacle=True), obj("CounterTop", 1.2, 2.4, receptacle=True)]
    for i in range(6):
        o.append(obj("Cabinet", -2.0 + i * 0.9, 2.6 if i % 2 else -0.5, openable=True, receptacle=True))
    for i in range(3):
        o.append(obj("Drawer", i * 0.5, -0.4, openable=True, receptacle=True))
    # traps that must NOT reach the output
    o += [obj("Apple", -1.0, 2.1, pickupable=True, moveable=True), obj("Knife", 0.1, 2.4, pickupable=True),
          obj("Stool", 0.9, 0.2, moveable=True, receptacle=True), obj("DiningTable", 1.5, 0.5, moveable=True, receptacle=True),
          obj("GarbageBag", 1.0, 0.8, moveable=True)]
    return o


def synth_kitchen():
    """SynthKitchen: 2 Fridges, no Toaster, a Sink WITHOUT SinkBasin, 6 StoveKnobs, 4 StoveBurners, a GarbageCan, no Drawer."""
    o = [obj("Fridge", -3.0, 0.0, openable=True, receptacle=True), obj("Fridge", 3.0, 0.0, openable=True, receptacle=True),
         obj("Sink", 0.0, 3.0), obj("GarbageCan", -0.5, 3.1, receptacle=True, moveable=True)]
    for i in range(4):
        o.append(obj("StoveBurner", 1.0 + 0.4 * i, 3.0, receptacle=True))
    for i in range(6):
        o.append(obj("StoveKnob", 1.0 + 0.3 * i, 3.4, toggleable=True))
    o += [obj("CounterTop", 0.5, 3.0, receptacle=True), obj("Cabinet", 2.0, 3.2, openable=True, receptacle=True)]
    o.append(obj("Mug", 0.5, 3.0, pickupable=True))
    return o


def main() -> int:
    conv = kb.load_conventions()
    tpl = kb.load_templates()
    vocab = kb.detector_vocab()
    check("vocab loaded", vocab and "Fridge" in vocab)
    check("conv schema", conv["schema"] == "pkg.conv/1" and conv["source"] == "domain conventions, not observed")
    check("fixture/tool lists intersect vocab", set(kb.fixture_types_allowed(conv, vocab)) <= vocab)

    # ---- whitelist
    lay = kb.extract_layout(fixture_kitchen(), "TestKitchen", vocab=vocab, extracted="t")
    types = {f["type"] for f in lay["fixtures"]}
    for bad in ("Apple", "Knife", "Stool", "DiningTable", "GarbageBag"):
        check("trap dropped: " + bad, bad not in types)
    txt = json.dumps(lay)
    for bad in ("Apple", "Knife", "Stool", "DiningTable", "GarbageBag", "isOpen", "isToggled", "controlledObjects", "secret", "objectId", "|"):
        check("trap absent from layout text: " + bad, bad not in txt)
    check("moveable-flag appliances kept", {"Microwave", "CoffeeMachine"} <= types)
    check("pickupable+fixture-type object dropped", "Apple" not in types)
    fx = {f["type"]: f for f in lay["fixtures"]}
    check("counts", fx["StoveBurner"]["n"] == 4 and fx["StoveKnob"]["n"] == 4 and fx["Cabinet"]["n"] == 6 and fx["Drawer"]["n"] == 3)
    check("cells quantised to 0.5", all(abs(c[0] * 2 - round(c[0] * 2)) < 1e-9 and abs(c[1] * 2 - round(c[1] * 2)) < 1e-9
                                          for f in lay["fixtures"] for c in f["cells"]))
    check("parent type never read (ids stay unread)", fx["SinkBasin"]["parent_type"] is None and fx["StoveBurner"]["parent_type"] is None)
    check("flags", fx["Fridge"]["openable"] and not fx["Fridge"]["toggleable"] and fx["Microwave"]["toggleable"])
    check("layout sha self-consistent", lay["sha256"] == kb.self_sha(lay) and lay["schema"] == "pkg.layout/1")
    check("whitelist recorded", "position.x" in lay["fields_whitelist"] and not any("isOpen" in w for w in lay["fields_whitelist"]))
    # input mutated state must not change the layout (state fields are never read)
    k2 = fixture_kitchen()
    for o in k2:
        o["isOpen"] = False
        o["isToggled"] = False
        o["controlledObjects"] = []
    check("layout independent of object state", kb.extract_layout(k2, "TestKitchen", vocab=vocab, extracted="t")["sha256"] == lay["sha256"])
    # and independent of movable/pickupable clutter
    k3 = [o for o in fixture_kitchen() if not (o["pickupable"] or o["objectType"] in ("Stool", "DiningTable", "GarbageBag"))]
    check("layout independent of clutter", kb.extract_layout(k3, "TestKitchen", vocab=vocab, extracted="t")["sha256"] == lay["sha256"])

    # ---- records, both kitchens
    out = {}
    for name, objs in (("TestKitchen", fixture_kitchen()), ("SynthKitchen", synth_kitchen())):
        L = kb.extract_layout(objs, name, vocab=vocab, extracted="t")
        R = kb.build_records(L, conv, tpl, vocab)
        out[name] = (L, R)
        ids = [r["id"] for r in R]
        check(name + " ids unique+ascii", len(ids) == len(set(ids)) and all(i.isascii() for i in ids))
        check(name + " deterministic", kb.build_records(L, conv, tpl, vocab) == R)
        check(name + " all privileged briefing", all(r["privileged"] and r["component"] == "briefing" and r["kitchen"] == name
                                                    and r["confidence"] == 1.0 and r["created"] == 0 and r["schema"] == "pkg.rec/1" for r in R))
        check(name + " en+fa nonempty", all(r["text"]["en"] and r["text"]["fa"] for r in R))
        check(name + " fa has every en type token (0 missing)", kb.fa_missing_tokens(R, vocab) == 0)
        check(name + " no unfilled placeholders", not any("{" in r["text"][l] for r in R for l in ("en", "fa")))
        check(name + " no id-like or parenthesised-tool text", not any(re.search(r"[A-Za-z]\(", r["text"]["en"]) or "|" in r["text"]["en"] for r in R))
        check(name + " no digits-with-positions leak", not any(re.search(r"-?\d+\.\d", r["text"]["en"]) for r in R))
        check(name + " no state words", not any(re.search(r"\b(is open|is closed|currently|right now|inside the)\b", r["text"]["en"]) for r in R))
        check(name + " no knob map", not any("controls" in r["keys"]["rels"] or "knob" in r["text"]["en"].lower() and "burner" in r["text"]["en"].lower()
                                              for r in R))
        check(name + " rels in closed set", all(x.split(":")[0] in ("near", "opens", "toggles", "stores") for r in R for x in r["keys"]["rels"]))
        have = {f["type"] for f in L["fixtures"]}
        for r in R:
            if r["kind"] in ("core", "exist", "place", "open"):
                check(name + " rec type exists " + r["id"], r["keys"]["types"][0] in have)
        # conventions only for existing receptacles
        for r in (x for x in R if x["kind"] == "conv"):
            recs = [x.split(":")[1] for x in r["keys"]["rels"]]
            check(name + " conv receptacles exist " + r["id"], all(t in have for t in recs))
            check(name + " conv wording", ("general convention" in r["text"]["en"] and "not checked today" in r["text"]["en"] and r["text"]["en"].startswith("Usual place for")))
        core = [r for r in R if r["kind"] == "core"]
        check(name + " core prio 0", core and all(r["prio"] == 0 and r["id"].split("/")[-1].startswith("core:") for r in core))

    L, R = out["TestKitchen"]
    ids = {r["id"].split("/", 2)[2] for r in R}
    check("test: core Fridge/Microwave/SinkBasin/StoveBurner/CounterTop",
          {"core:Fridge", "core:Microwave", "core:SinkBasin", "core:StoveBurner", "core:CounterTop", "core:Sink"} <= ids)
    check("test: burner count line", any(r["id"].endswith("core:StoveBurner") and "4" in r["text"]["en"] for r in R))
    check("test: toaster absent", not any("Toaster" in r["text"]["en"] for r in R))
    check("test: microwave open+toggle", any(r["id"].endswith("open:Microwave") and "switched" in r["text"]["en"] and "opened" in r["text"]["en"] for r in R))
    check("test: StoveBurner toggle-free (receptacle only)", not any(r["id"].endswith("open:StoveBurner") for r in R))
    adj = {r["id"].split(":")[-1]: r for r in R if r["kind"] == "adj"}
    check("test: Microwave next to CoffeeMachine", "Microwave" in adj and "CoffeeMachine" in adj["Microwave"]["text"]["en"])
    check("test: adjacency cap", all(len(r["keys"]["types"]) - 1 <= kb.ADJ_MAX for r in adj.values()))
    check("test: spread cabinets", any(r["id"].endswith("place:Cabinet") and "spread" in r["text"]["en"] for r in R))
    check("test: cutlery convention", any(r["id"].endswith("conv:cutlery") and "Drawer" in r["text"]["en"] for r in R))

    L, R = out["SynthKitchen"]
    ids = {r["id"].split("/", 2)[2] for r in R}
    check("synth: two fridges", any(r["id"].endswith("core:Fridge") and "2" in r["text"]["en"] and "units" in r["text"]["en"] for r in R))
    check("synth: no toaster/coffee/microwave lines", not any(w in r["text"]["en"] for r in R for w in ("Toaster", "CoffeeMachine", "Microwave")))
    check("synth: sink without basin -> core:Sink, no SinkBasin", "core:Sink" in ids and not any("SinkBasin" in i for i in ids))
    check("synth: 6 knobs 4 burners", any(r["id"].endswith("exist:StoveKnob") and "6" in r["text"]["en"] for r in R)
          and any(r["id"].endswith("core:StoveBurner") and "4" in r["text"]["en"] for r in R))
    check("synth: no drawer convention for cutlery -> uses CounterTop only",
          any(r["id"].endswith("conv:cutlery") and "Drawer" not in r["text"]["en"] and "CounterTop" in r["text"]["en"] for r in R))
    check("synth: cleaning conv has no SinkBasin", not any(r["id"].endswith("conv:cleaning") and "SinkBasin" in r["text"]["en"] for r in R))
    check("synth: no code branch on name (renaming changes only the kitchen field)",
          [r["text"] for r in kb.build_records(dict(L, scene="Other"), conv, tpl, vocab)] == [r["text"] for r in R])
    src = open(os.path.join(HERE, "kitchen_briefing.py"), encoding="utf-8").read()
    check("source has no scene-name constants", not re.search(r"FloorPlan\d|SynthKitchen", src.split('"""', 2)[2]))

    # ---- markdown
    for lg in ("en", "fa"):
        md = kb.render_markdown(L, R, lg, tpl)
        check("md " + lg + " draft marker", "DRAFT" in md and "SynthKitchen" in md)
        check("md " + lg + " has all lines", all(r["text"][lg] in md for r in R))

    # ---- files: render, overrides, approval, history
    d = tempfile.mkdtemp(prefix="b2st_")
    try:
        L0 = out["TestKitchen"][0]
        with open(os.path.join(d, "TestKitchen.layout.json"), "w", encoding="utf-8") as f:
            json.dump(L0, f, ensure_ascii=False)
        r1 = kb.render_scene("TestKitchen", out_dir=d)
        check("render writes files", all(os.path.exists(os.path.join(d, "TestKitchen." + e)) for e in ("en.md", "fa.md", "records.jsonl", "overrides.json")))
        b1 = open(os.path.join(d, "TestKitchen.records.jsonl"), "rb").read()
        kb.render_scene("TestKitchen", out_dir=d)
        check("render byte-deterministic", open(os.path.join(d, "TestKitchen.records.jsonl"), "rb").read() == b1)
        try:
            kb.load_records("TestKitchen", out_dir=d)
            check("refuses unapproved", False)
        except kb.BriefingNotApproved:
            check("refuses unapproved", True)
        recs = kb.load_records("TestKitchen", allow_unapproved=True, out_dir=d)
        check("allow_unapproved loads all", len(recs) == r1["records"])
        victim = recs[0]["id"]
        edit = recs[5]["id"]
        kb.save_overrides("TestKitchen", {"version": 1, "approved_by": "Arash", "approved_at": "t",
                                          "edits": {victim: None, edit: {"en": "EDITED en", "fa": "EDITED fa"}}}, out_dir=d)
        recs2 = kb.load_records("TestKitchen", out_dir=d)
        check("approved loads", len(recs2) == len(recs) - 1)
        check("null deletes", all(r["id"] != victim for r in recs2))
        e = [r for r in recs2 if r["id"] == edit][0]
        check("edit applied + provenance", e["text"] == {"en": "EDITED en", "fa": "EDITED fa"} and e["provenance"]["authored_by"] == "arash-override:v1")
        check("history kept", os.path.exists(os.path.join(d, "_history", "TestKitchen.overrides.v0.json")))
        kb.save_overrides("TestKitchen", {"version": 2, "approved_by": "Arash", "approved_at": "t", "edits": {}}, out_dir=d)
        check("history keeps every version", len(os.listdir(os.path.join(d, "_history"))) == 2)
        # tampered layout refused
        bad = dict(L0)
        bad["fixtures"] = L0["fixtures"][1:]
        with open(os.path.join(d, "TestKitchen.layout.json"), "w", encoding="utf-8") as f:
            json.dump(bad, f)
        try:
            kb.render_scene("TestKitchen", out_dir=d)
            check("tampered layout refused", False)
        except ValueError:
            check("tampered layout refused", True)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---- early check B2 (section 5.5): lines per kind, tokens with the fake counter; real briefings when present
    def fake(t, fa=False):
        n = math.ceil(len(t) / 3)
        return int(n * 1.7) if fa else n
    for name in ("SynthKitchen", "TestKitchen"):
        L, R = out[name]
        kinds = {}
        for r in R:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        en = sum(fake(r["text"]["en"]) for r in R)
        fa = sum(fake(r["text"]["fa"], True) for r in R)
        print("early check %-13s lines/kind %s  fake tokens EN %d FA %d  fa-missing-type-tokens %d" % (name, kinds, en, fa, kb.fa_missing_tokens(R, vocab)))
    for scene in ("FloorPlan1", "FloorPlan15", "FloorPlan17"):
        p = os.path.join(kb.OUT_DIR, scene + ".records.jsonl")
        if os.path.exists(p):
            R = kb.load_records(scene, allow_unapproved=True)
            check(scene + " real records: fa has every en type token", kb.fa_missing_tokens(R, vocab) == 0)
            check(scene + " real records: no clutter types", not any(w in r["text"]["en"] for r in R if r["kind"] != "conv" for w in ("Apple", "Knife", "Stool", "DiningTable", "GarbageBag")))
            check(scene + " real records: unapproved (draft) until Arash signs",
                  json.load(open(os.path.join(kb.OUT_DIR, scene + ".overrides.json"), encoding="utf-8")).get("approved_by") in (None, "Arash"))

    print("%d ok, %d bad" % (N["ok"], N["bad"]))
    return 0 if N["bad"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
