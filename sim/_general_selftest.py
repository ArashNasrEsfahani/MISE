"""Offline gate for the general instruction set (sim/general_set.py, gen_general.py, eval's applicability gate).

Pure: no simulator, no LLM, no network. Run from the repo root:   python sim/_general_selftest.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval as ev  # noqa: E402
import gen_general  # noqa: E402
import general_set as gs  # noqa: E402
import levels  # noqa: E402

PASS = FAIL = 0


def ok(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # ---- DSL round trips ---------------------------------------------------------------------------------
    for line in ("pick_and_place Apple -> Fridge", "fill Mug = water", "toggle_on Toaster", "crouch",
                 "put_in SinkBasin", "close_all"):
        ok(f"step round-trip {line!r}", gs.format_step(gs.parse_step(line)) == line)
    for line in ("Apple in Fridge", "Fridge.isOpen = false", "holding Knife", "any Cabinet.isOpen = true",
                 "all Drawer.isOpen = false", "Mug.fillLiquid = water", "Fridge.distance <= 1.5",
                 "Apple.temperature = Cold", "Agent.isStanding = false"):
        ok(f"goal round-trip {line!r}", gs.format_goal(gs.parse_goal(line)) == line)
    ok("step arg is typed", gs.parse_step("fill Mug = water")["arg"] == "water")
    ok("goal bool is typed", gs.parse_goal("Fridge.isOpen = false")["value"] is False)
    ok("goal bound is numeric", gs.parse_goal("Fridge.distance <= 1.5")["max"] == 1.5)
    for bad in ("", "Apple Fridge", "Fridge.isOpen", "Fridge.distance <= near"):
        try:
            gs.parse_goal(bad)
            ok(f"bad goal rejected {bad!r}", False)
        except ValueError:
            ok(f"bad goal rejected {bad!r}", True)
    try:
        gs.parse_lines("open Fridge\nthis is not a step at all\n", gs.parse_step)
        ok("parse_lines names the line", False)
    except ValueError as exc:
        ok("parse_lines names the line", "line 2" in str(exc), str(exc))

    # ---- the seed ---------------------------------------------------------------------------------------
    seed = [gs.normalize(it) for it in gen_general.build_all()]
    inv = gs.load_inventory()
    actions = gs.known_actions()
    by = {lv: [it for it in seed if it["level"] == lv] for lv in gs.LEVELS}
    ok("seed >= 100 L1", len(by["L1"]) >= 100, str(len(by["L1"])))
    ok("seed >= 100 L2", len(by["L2"]) >= 100, str(len(by["L2"])))
    ok("seed >= 50 L3", len(by["L3"]) >= 50, str(len(by["L3"])))
    ok("seed ids unique", len({it["id"] for it in seed}) == len(seed))
    ok("seed english unique", len({it["en"] for it in seed}) == len(seed))
    ok("seed persian unique", len({it["fa"] for it in seed}) == len(seed))
    bad = [(it["id"], e) for it in seed for e in gs.validate_item(it, actions, inv)[0]]
    ok("every seed item validates", not bad, str(bad[:3]))
    ok("every item is bilingual", all(it["en"] and gs._ARABIC_RE.search(it["fa"]) for it in seed))
    ok("no persian in the english", not any(gs._ARABIC_RE.search(it["en"]) for it in seed))
    ok("every item has a reference plan", all(it["plan"] for it in seed))
    l1_long = [it["id"] for it in by["L1"] if len(gs.plan_labels(it["plan"])) > 1]
    ok("L1 = one function", not l1_long, str(l1_long[:3]))
    l2_short = [it["id"] for it in by["L2"] if len(gs.plan_labels(it["plan"])) < 2]
    ok("L2 = a chain of >= 2", not l2_short, str(l2_short[:3]))
    # A goal that names a state every kitchen STARTS in must come with a setup, or doing nothing scores SR = 1.
    trivially_true = {("isOpen", False), ("isToggled", False), ("isDirty", False), ("isFilledWithLiquid", False),
                      ("isPickedUp", False)}
    lazy = [it["id"] for it in seed if not it["setup"]
            and all((a.get("key"), a.get("value")) in trivially_true and a.get("type") != "LightSwitch"
                    for a in it["goal"])]
    ok("no goal is true at reset without a setup", not lazy, str(lazy[:4]))
    sliced = [it["id"] for it in seed for a in it["goal"]
              if a.get("key") in ("isSliced", "isBroken") and a.get("q") != "any"]
    ok("slice/break goals are quantified (the two-Bread coin flip)", not sliced, str(sliced[:3]))

    if inv:
        full = [it for it in seed if gs.applicability(it, inv)["n"] == len(inv["scenes"])]
        ok("most of the set applies in every kitchen", len(full) >= 0.9 * len(seed), f"{len(full)}/{len(seed)}")
        drawer = next(it for it in seed if it["id"] == "g-l1-open-drawer")
        ap = gs.applicability(drawer, inv)
        ok("applicability names the kitchens without a Drawer",
           ap["n"] == 28 and set(ap["missing"]) == {"FloorPlan14", "FloorPlan29"}, str(ap["n"]))
        cold = {"id": "x-cold", "level": "L3", "en": "cold", "fa": "سرد", "setup": [], "plan": [gs.parse_step("cool Apple")],
                "goal": [gs.parse_goal("Apple.temperature = Cold")]}
        pre = gs.presatisfied_scenes(cold, inv)
        ok("presatisfied finds the apples that spawn in the fridge", "FloorPlan2" in pre and "FloorPlan1" not in pre,
           str(pre))

    # ---- records + the loader -----------------------------------------------------------------------------
    doc = {"version": 1, "items": seed, "deleted": []}
    recs = gs.to_records(doc)
    ok("two records per item", len(recs) == 2 * len(seed))
    ok("records interleave languages", [r["lang"] for r in recs[:4]] == ["en", "fa", "en", "fa"])
    ok("records are scene-less", all("scene" not in r for r in recs))
    for r in recs[:40]:
        levels.CommandRecord.from_dict(r).validate()
    l2 = next(r for r in recs if r["id"] == "g-l2-fridge-apple-en")
    ok("L2 gold sequence is derived from the plan",
       l2["gold"]["sequence"] == ["pick_and_place Apple", "close Fridge"], str(l2["gold"].get("sequence")))
    fill = next(r for r in recs if r["id"] == "g-l2-stove-water-pot-en")
    ok("fill is labelled in the manifest vocabulary", fill["gold"]["sequence"][0] == "fill_with_liquid Pot")
    l3 = next(r for r in recs if r["id"] == "g-l3-salad-fa")
    ok("L3 gold is an action set", sorted(l3["gold"]["actions"]) == ["pickup Knife", "slice Lettuce", "slice Tomato"])
    ok("setup travels with the record",
       next(r for r in recs if r["id"] == "g-l1-close-fridge-en")["setup"][0]["action"] == "open")

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "general.json")
        gs.save(dict(doc), p)
        loaded = ev.load_dataset(p)
        ok("eval.load_dataset reads the general set", len(loaded) == 2 * len(seed))
        ok("eval.load_dataset lang filter", len(ev.load_dataset(p, lang="fa")) == len(seed))
        ok("loaded records carry setup", any(r.setup for r in loaded))
        ok("the JSONL dataset still loads", len(ev.load_dataset(ev.DEFAULT_DATASET)) == 102)

        # ---- store: add / rename / duplicate / delete, all audited --------------------------------------
        audit = os.path.join(tmp, "audit.jsonl")
        new = {"id": "t-open-oven", "level": "L1", "en": "Open the oven.", "fa": "در فر را باز کن.",
               "plan": [gs.parse_step("open Microwave")], "goal": [gs.parse_goal("Microwave.isOpen = true")]}
        r = gs.upsert(new, path=p, audit_path=audit)
        ok("add", r["ok"] and r["op"] == "add", str(r.get("errors")))
        ok("duplicate id refused", not gs.upsert(new, path=p, audit_path=audit)["ok"])
        ren = dict(new, id="t-open-oven-2", en="Open the oven door.")
        r = gs.upsert(ren, original_id="t-open-oven", path=p, audit_path=audit)
        ids = [it["id"] for it in gs.load(p)["items"]]
        ok("rename replaces, not duplicates", r["ok"] and "t-open-oven-2" in ids and "t-open-oven" not in ids)
        ok("invalid item refused", not gs.upsert(dict(new, id="t-x", level="L9"), path=p, audit_path=audit)["ok"])
        ok("missing persian refused", not gs.upsert(dict(new, id="t-y", fa=""), path=p, audit_path=audit)["ok"])
        ok("L2 without a plan refused",
           not gs.upsert(dict(new, id="t-z", level="L2", plan=[]), path=p, audit_path=audit)["ok"])
        # ---- the review mark: survives an edit, is audited, and only writes when it changes
        ok("check", gs.set_reviewed("t-open-oven-2", True, path=p, audit_path=audit)["ok"])
        ok("check is stored", next(i for i in gs.load(p)["items"] if i["id"] == "t-open-oven-2").get("reviewed") is True)
        gs.set_reviewed("t-open-oven-2", True, path=p, audit_path=audit)          # no change -> no audit line
        r = gs.upsert(dict(ren, en="Open the oven door, please."), original_id="t-open-oven-2", path=p, audit_path=audit)
        ok("an edit does not un-check", r["ok"] and r["item"].get("reviewed") is True)
        ok("checking a missing id fails cleanly", not gs.set_reviewed("nope", True, path=p, audit_path=audit)["ok"])
        ok("uncheck", gs.set_reviewed("t-open-oven-2", False, path=p, audit_path=audit)["ok"]
           and "reviewed" not in next(i for i in gs.load(p)["items"] if i["id"] == "t-open-oven-2"))
        csv_text = gs.export_csv(gs.load(p))
        ok("csv export is bilingual, one row per instruction",
           csv_text.count("\n") == len(gs.load(p)["items"]) + 1 and "سیب را بردار." in csv_text
           and csv_text.startswith("id,level,english,persian,"))
        ok("delete", gs.delete("t-open-oven-2", path=p, audit_path=audit)["ok"])
        ok("delete is remembered", "t-open-oven-2" in gs.load(p)["deleted"])
        ok("delete of a missing id fails cleanly", not gs.delete("nope", path=p, audit_path=audit)["ok"])
        ops = [json.loads(ln)["op"] for ln in open(audit, encoding="utf-8")]
        ok("every write is audited", ops == ["add", "update", "review", "update", "unreview", "delete"], str(ops))

    # ---- editor payload -> item ---------------------------------------------------------------------------
    it = gs.from_editor({"id": "t-e", "level": "L2", "en": "a", "fa": "ب", "setup_text": "open Fridge\n",
                         "plan_text": "# comment\nclose Fridge\ntoggle_off Faucet", "goal_text": "Fridge.isOpen = false"})
    ok("editor text -> item", len(it["plan"]) == 2 and len(it["setup"]) == 1 and it["goal"][0]["value"] is False)
    v = gs.view(seed[0], inv)
    ok("view round-trips through the editor",
       gs.normalize(gs.from_editor(dict(v, id=v["id"])))["goal"] == seed[0]["goal"])
    try:
        gs.from_editor({"plan_text": "open Fridge\nnot a valid step line here", "goal_text": "x"})
        ok("editor error names the field", False)
    except ValueError as exc:
        ok("editor error names the field", str(exc).startswith("plan, line 2"), str(exc))

    # ---- eval: quantifiers, bounds, gate helpers -------------------------------------------------------------
    world = {"Cabinet": [{"isOpen": False}, {"isOpen": True}], "Fridge": [{"isOpen": False, "distance": 1.2}],
             "Bread": [{"isSliced": False}, {"isSliced": True}]}
    one = lambda t, k: (world.get(t) or [{}])[0].get(k)           # "nearest" = the first instance
    every = lambda t, k: [o.get(k) for o in world.get(t, [])]
    g = lambda line: [gs.parse_goal(line)]
    ok("unquantified reads ONE instance", ev.sr_score(g("Cabinet.isOpen = true"), one, every) is False)
    ok("any finds the open cabinet", ev.sr_score(g("any Cabinet.isOpen = true"), one, every) is True)
    ok("all fails while one is open", ev.sr_score(g("all Cabinet.isOpen = false"), one, every) is False)
    ok("any settles the two-Bread coin flip", ev.sr_score(g("any Bread.isSliced = true"), one, every) is True)
    ok("all over zero instances is False", ev.sr_score(g("all Drawer.isOpen = false"), one, every) is False)
    ok("quantifier without get_all degrades to the old reader",
       ev.sr_score(g("any Cabinet.isOpen = true"), one) is False)
    ok("bound holds", ev.sr_score(g("Fridge.distance <= 1.5"), one, every) is True)
    ok("bound fails", ev.sr_score(g("Fridge.distance <= 1.0"), one, every) is False)
    ok("bound on a missing value is False", ev.sr_score(g("Oven.distance <= 9"), one, every) is False)

    rec = levels.CommandRecord.from_dict(next(r for r in recs if r["id"] == "g-l2-drawer-fork-en"))
    ok("gold_types collects goal + labels", ev.gold_types(rec) == ["Drawer", "Fork"], str(ev.gold_types(rec)))
    ok("missing_types: absent Drawer", ev.missing_types(["Drawer", "Fork"], ["Fork", "Fridge"]) == ["Drawer"])
    ok("missing_types is exact: a Potato is not a Pot", ev.missing_types(["Pot"], ["Potato"]) == ["Pot"])
    ok("missing_types canonicalises aliases", ev.missing_types(["sink", "fridge"], ["SinkBasin", "Fridge"]) == [])
    agent = levels.CommandRecord.from_dict(next(r for r in recs if r["id"] == "g-l1-crouch-en"))
    ok("Agent is never looked up in the scene", ev.gold_types(agent) == [])

    rows = [{"level": "L1", "lang": "en", "model": "m", "scene": "s", "SR": True},
            {"level": "L1", "lang": "en", "model": "m", "scene": "s", "SR": False},
            {"level": "L1", "lang": "en", "model": "m", "scene": "s", "SR": None, "skipped": "object does not exist"}]
    b = ev.aggregate(rows)["overall"]
    ok("skipped records are outside every mean", b["n"] == 2 and b["n_skipped"] == 1 and b["SR"] == 0.5, str(b))

    print(f"checks: {PASS + FAIL} total | {PASS} PASS | {FAIL} FAIL")
    print("GENERAL SET SELFTEST OK" if not FAIL else "GENERAL SET SELFTEST FAILED")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
