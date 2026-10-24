"""Gate for sim/goal_schema.py and its query integration (stage I-F1-goalschema). Offline: no simulator, LLM, clock or network.

    python sim/_goal_schema_selftest.py

Groups: A purity and robustness, B span and provenance integrity (no unstated goals), C golden cases (hand-authored EN/FA paraphrases,
including the contrastive set: fact vs wish vs negation), D signal-set and non-test-set measures, E query integration, F no leakage.
The golden cases were written from general kitchen language, not from the dataset; they are regression anchors, not a held-out set.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import goal_schema as G           # noqa: E402
import ltm_concepts as LC         # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("PASS", name)
    else:
        FAIL += 1
        print("FAIL", name, "--", detail)


# (text, lang, status, canonical key with unknown-noun surfaces)
GOLDEN = [
    ('Please put two apples in the fridge.', 'en', 'ok', 'put(Apple>Fridge)'),
    ("Don't open the microwave.", 'en', 'ok', 'open(Microwave>None!)'),
    ('Put both mugs in the sink, then turn on the faucet.', 'en', 'ok', 'put(Mug>Sink) toggle:on(Faucet>None)'),
    ('Before you close the drawer, put the fork inside.', 'en', 'ok', 'put(Fork>Drawer) close(Drawer>None)'),
    ('Heat the soup.', 'en', 'ok', 'heat(soup>None)'),
    ('Take the spatula and flip the egg.', 'en', 'ok', 'pick(Spatula>None) flip(Egg>None)'),
    ('Wash the cup and put it on the counter.', 'en', 'ok', 'wash(Cup>None) put(Cup>CounterTop)'),
    ('Warm up the bread in the microwave.', 'en', 'ok', 'heat(Bread>Microwave)'),
    ('Take the apple out of the fridge and slice it.', 'en', 'ok', 'pick(Apple>None<Fridge) slice(Apple>None)'),
    ('Turn the stove off.', 'en', 'ok', 'toggle:off(StoveBurner>None)'),
    ('Switch on the light.', 'en', 'ok', 'toggle:on(LightSwitch>None)'),
    ('Sing a song.', 'en', 'unsupported', ''),
    ('Make me a coffee.', 'en', 'ok', 'make(coffee>None)'),
    ("I'd like the lettuce chilled.", 'en', 'ok', 'chill(Lettuce>None^cold)'),
    ('Put the plate on the table.', 'en', 'ok', 'put(Plate>CounterTop)'),
    ('Can you pick up the knife and cut the bread with it?', 'en', 'ok', 'pick(knife>None) slice(Bread>None)'),  # A26 (K-B2b): bare knife -> '?' (was pick(Knife>None))
    ('I need something to stir with.', 'en', 'unsupported', ''),
    ('Close all the cabinets.', 'en', 'ok', 'close(Cabinet>None)'),
    ('لیوان را در سینک بگذار.', 'fa', 'ok', 'put(Mug>Sink)'),
    ('درِ یخچال را باز نکن.', 'fa', 'ok', 'open(Fridge>None!)'),
    ('سیب را از یخچال بردار و برش بزن.', 'fa', 'ok', 'pick(Apple>None<Fridge) slice(Apple>None)'),
    ('دو تا لیوان را روی میز بگذار.', 'fa', 'ok', 'put(Mug>CounterTop)'),
    ('چراغ را روشن کن.', 'fa', 'ok', 'toggle:on(LightSwitch>None)'),
    ('نان را در مایکروویو گرم کن.', 'fa', 'ok', 'heat(Bread>Microwave)'),
    ('می\u200cخواهم کاهو را خنک بخورم.', 'fa', 'ok', 'eat(Lettuce>None) chill(Lettuce>None^cold)'),
    ('برای بریدن نان یک چاقو لازم دارم.', 'fa', 'ok', 'slice(Bread>None#item=?)'),  # A26 (K-B2b): bare knife -> '?' (was #item=Knife)
    ('یک آهنگ بخوان.', 'fa', 'unsupported', ''),
    ('همه کابینت\u200cها را ببند.', 'fa', 'ok', 'close(Cabinet>None)'),
    ('قبل از بستن کشو، چنگال را داخلش بگذار.', 'fa', 'ok', 'put(Fork>Drawer) close(Drawer>None)'),
    ('تخم\u200cمرغ را بپز.', 'fa', 'ok', 'cook(Egg>None)'),
    ('بشقاب را بشور و در کابینت بگذار.', 'fa', 'ok', 'wash(Plate>None) put(Plate>Cabinet)'),
    ('Put the egg in the microwave, close it and turn it on.', 'en', 'ok', 'put(Egg>Microwave) close(Microwave>None) toggle:on(Microwave>None)'),
    ('شیر آب را روشن کن.', 'fa', 'ok', 'toggle:on(Faucet>None)'),
    ('Fill the kettle with water.', 'en', 'ok', 'fill(Kettle>None)'),
    ('The bowl is dirty. Wash it.', 'en', 'ok', 'wash(Bowl>None)'),
    ('کاسه کثیف است؛ آن را بشور.', 'fa', 'ok', 'wash(Bowl>None)'),
    ('The apple is cold.', 'en', 'indirect', 'None(Apple>None?cold)'),
    ('I want the apple cold.', 'en', 'ok', 'chill(Apple>None^cold)'),
    ("Don't chill the apple.", 'en', 'ok', 'chill(Apple>None!)'),
    ('Make sure the apple is cold.', 'en', 'ok', 'chill(Apple>None^cold)'),
    ('The apple is already cold, leave it.', 'en', 'indirect', 'None(Apple>None?cold)'),
    ('The microwave is on.', 'en', 'indirect', 'None(Microwave>None?on)'),
    ('Turn off the microwave.', 'en', 'ok', 'toggle:off(Microwave>None)'),
    ('Do not turn on the stove.', 'en', 'ok', 'toggle:on(StoveBurner>None!)'),
    ('The stove is off.', 'en', 'indirect', 'None(StoveBurner>None?off)'),
    ('Never open the fridge.', 'en', 'ok', 'open(Fridge>None!)'),
    ('سیب سرد است.', 'fa', 'indirect', 'None(Apple>None?cold)'),
    ('سیب را سرد می\u200cخواهم.', 'fa', 'ok', 'chill(Apple>None^cold)'),
    ('سیب را سرد نکن.', 'fa', 'ok', 'chill(Apple>None!)'),
    ('مایکروویو روشن است.', 'fa', 'indirect', 'None(Microwave>None?on)'),
    ('اجاق را روشن نکن.', 'fa', 'ok', 'toggle:on(StoveBurner>None!)'),
    ('فریزر را باز نکن.', 'fa', 'ok', 'open(Fridge>None!)'),
    ("I don't want the bread cold.", 'en', 'indirect', ''),
    ('The knife is on the counter.', 'en', 'unsupported', ''),  # A26 (K-B2b): bare knife -> '?' (was indirect: a fact about an unknown thing)
    ('Is the fridge open?', 'en', 'indirect', ''),
    ('چاقو روی کانتر است.', 'fa', 'unsupported', ''),
    ("Put the apple in the fridge but don't close it.", 'en', 'ok', 'put(Apple>Fridge) close(Fridge>None!)'),
    ('Could you please grab the mug and place it in the cabinet?', 'en', 'ok', 'pick(Mug>None) put(Mug>Cabinet)'),
    ('Pick up the pot, put it on the stove and turn the burner on.', 'en', 'ok', 'pick(Pot>None) put(Pot>StoveBurner) toggle:on(StoveBurner>None)'),
    ('Put the cup in the sink and run the tap.', 'en', 'ok', 'put(Cup>Sink) toggle:on(Faucet>None)'),
    ('Open the fridge, take out the tomato and close the fridge.', 'en', 'ok', 'open(Fridge>None) pick(Tomato>None) close(Fridge>None)'),
    ('Cool the lettuce down.', 'en', 'ok', 'chill(Lettuce>None)'),
    ("I'm hungry.", 'en', 'unsupported', ''),
    ('Turn off the light and the toaster.', 'en', 'ok', 'toggle:off(LightSwitch>None) toggle:off(Toaster>None)'),
    ('Put the knife and the fork in the drawer.', 'en', 'ok', 'put(knife>Drawer) put(Fork>Drawer)'),  # A26 (K-B2b): bare knife -> '?' (was put(Knife>Drawer))
    ('Take the plate to the table.', 'en', 'ok', 'put(Plate>CounterTop)'),
    ('Slice the tomato with the knife.', 'en', 'ok', 'slice(Tomato>None)'),  # A26 (K-B2b): bare knife -> '?' (was @Knife; an unknown tool is not a key)
    ('The mug is full. Empty it into the sink.', 'en', 'ok', 'empty(Mug>Sink)'),
    ('After washing the pan, put it in the cabinet.', 'en', 'ok', 'wash(Pan>None) put(Pan>Cabinet)'),
    ('Put the spoon in the sink after you use it.', 'en', 'ok', 'put(Spoon>Sink)'),
    ("I'd like a cup of coffee.", 'en', 'ok', 'None(Cup>None#item)'),
    ('Stop the stove.', 'en', 'ok', 'toggle:off(StoveBurner>None)'),
    ('Where is the apple?', 'en', 'indirect', ''),
    ('Please find the bread.', 'en', 'ok', 'find(Bread>None)'),
    ('بشقاب و لیوان را در سینک بگذار.', 'fa', 'ok', 'put(Plate>Sink) put(Mug>Sink)'),
    ('یخچال را باز کن، گوجه را بردار و درش را ببند.', 'fa', 'ok', 'open(Fridge>None) pick(Tomato>None) close(Fridge>None)'),
    ('لطفاً تخم\u200cمرغ را بشکن.', 'fa', 'ok', 'break(Egg>None)'),
    ('نان را با چاقو برش بزن.', 'fa', 'ok', 'slice(Bread>None)'),  # A26 (K-B2b): bare knife -> '?' (was @Knife)
    ('کتری را پر کن و روی اجاق بگذار.', 'fa', 'ok', 'fill(Kettle>None) put(Kettle>StoveBurner)'),
    ('چراغ و توستر را خاموش کن.', 'fa', 'ok', 'toggle:off(LightSwitch>None) toggle:off(Toaster>None)'),
    ('سیب کجاست؟', 'fa', 'indirect', ''),
    ('گرسنه\u200cام.', 'fa', 'unsupported', ''),
    ('ماهیتابه را بردار و روی اجاق بگذار.', 'fa', 'ok', 'pick(Pan>None) put(Pan>StoveBurner)'),
    ('بعد از شستن تابه، آن را در کابینت بگذار.', 'fa', 'ok', 'wash(Pan>None) put(Pan>Cabinet)'),
    ('کاهو را در یخچال بگذار.', 'fa', 'ok', 'put(Lettuce>Fridge)'),
    ('لیوان پر است؛ آن را در سینک خالی کن.', 'fa', 'ok', 'empty(Mug>Sink)'),
    ('غذا سرد شده است.', 'fa', 'indirect', 'None(None>None?cold)'),
    ('I want my soup hot.', 'en', 'indirect', 'None(None>None?hot)'),
    ('سوپ را داغ می\u200cخواهم.', 'fa', 'indirect', 'None(None>None?hot)'),
    ('Throw away the rotten egg.', 'en', 'ok', 'discard(Egg>None)'),
    ('تخم\u200cمرغ گندیده را دور بینداز.', 'fa', 'ok', 'discard(Egg>None)'),
]


def _all_spans(g):
    out = []

    def walk(x):
        if isinstance(x, dict):
            if "tok" in x and "text" in x:
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for k in ("entities", "operations", "final_predicates", "needs", "facts", "desires", "affordances", "wanted_states", "negations", "temporal"):
        walk(g[k])
    return out


def main() -> int:
    # ===================================================================================== A. purity, robustness
    txt = "Put the egg in the microwave, close the door, and turn it on."
    g1, g2 = G.parse_goal(txt, "en"), G.parse_goal(txt, "en")
    check("A1 deterministic: same text, same schema", json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True))
    odd = ["", "   ", "...", "؟؟", "asdf qwer", "a" * 500, "و و و", "را", "1234", "Put put put", "،،،", "\n\n", "turn turn on off", "!!!"]
    ok = True
    for t in odd:
        try:
            g = G.parse_goal(t)
            ok = ok and g["status"] in ("ok", "indirect", "partial", "unsupported") and isinstance(g["reason"], str)
        except Exception as e:                                  # pragma: no cover
            ok = False
            print("   crash on", repr(t), e)
    check("A2 odd inputs never crash and always carry a status", ok)
    check("A3 empty / gibberish / unmodelled is 'unsupported' with a reason (the defined unsupported path)",
          all(G.parse_goal(t)["status"] == "unsupported" and G.parse_goal(t)["reason"] for t in ("", "asdf qwer", "...", "یک آهنگ بخوان.")))
    check("A4 the language is detected from the script when not given", G.parse_goal("سیب را بردار")["lang"] == "fa" and G.parse_goal("Pick up the apple")["lang"] == "en")
    check("A5 schema and version tags", g1["schema"] == G.SCHEMA == "goal.schema/1" and g1["version"] == G.VERSION)

    # ===================================================================================== B. integrity
    texts = [(t, l) for (t, l, _s, _c) in GOLDEN]
    span_ok = type_ok = fp_ok = aff_ok = True
    bad = []
    for t, l in texts:
        g = G.parse_goal(t, l)
        toks = g["tokens"]
        for sp in _all_spans(g):
            a, b = sp["tok"]
            if not (0 <= a <= b <= len(toks)) or " ".join(toks[a:b]) != sp["text"]:
                span_ok = False
                bad.append((t, sp))
        for e in g["entities"]:
            if e.get("type") and e["type"] != "*" and not e.get("span"):
                type_ok = False
        ents = {e.get("type") for e in g["entities"]}
        for f in g["final_predicates"]:
            if not f.get("span") or f.get("from") not in ("operation", "stated_state", "wish", "declarative"):
                fp_ok = False
            if f.get("type") not in ents and not (g["context"]["holding"] and f.get("type") == g["context"]["holding"]["type"]):
                fp_ok = False
                bad.append((t, f))
        for a in g["affordances"]:
            if not a.get("via") or not a.get("rule", "").startswith("affordance:"):
                aff_ok = False
    check("B1 every span is a real slice of the token list (no invented text)", span_ok, bad[:2])
    check("B2 every typed entity carries a span", type_ok)
    check("B3 every final predicate has a span, a stated source, and a type that appears in the instruction", fp_ok, bad[:2])
    check("B4 every affordance is reached through a word span, never free-floating", aff_ok)
    aff_in_fp = []
    for t, l in texts:
        g = G.parse_goal(t, l)
        stated = {e.get("type") for e in g["entities"]}
        for a in g["affordances"]:
            if a["type"] not in stated and any(f.get("type") == a["type"] or f.get("target") == a["type"] for f in g["final_predicates"]):
                aff_in_fp.append((t, a["type"]))
    check("B5 an affordance type the instruction does not name never appears in final_predicates", not aff_in_fp, aff_in_fp[:2])
    for t, l in [("The apple is cold.", "en"), ("I'm hungry.", "en"), ("It's too dark in here.", "en"), ("The tomato has gone bad.", "en"),
                 ("Is the fridge open?", "en"), ("سیب سرد است.", "fa"), ("The knife is on the counter.", "en")]:
        g = G.parse_goal(t, l)
        check("B6 no operation, goal or wish is made up from a bare fact / question / statement: %r" % t,
              not g["operations"] and not g["final_predicates"] and not g["wanted_states"] and not [k for k in G.retrieval_key(g) if k.get("via_state") or k.get("dest")],
              json.dumps(G.retrieval_key(g)))
    g = G.parse_goal("The apple is cold.", "en")
    check("B7 'the apple is cold' is a fact, with no chill step and no Fridge", g["status"] == "indirect" and not G.affordance_types(g)
          and all(k.get("via_state") is None for k in G.retrieval_key(g)))
    g = G.parse_goal("I want the apple cold.", "en")
    check("B8 'I want the apple cold' is a wish: chill is flagged via_state, Fridge is an affordance, the final predicate is 'wish'",
          any(k.get("via_state") == "cold" for k in G.retrieval_key(g)) and G.affordance_types(g) == ["Fridge"]
          and [f["from"] for f in g["final_predicates"]] == ["wish"] and not g["operations"])
    g = G.parse_goal("Don't chill the apple.", "en")
    check("B9 'don't chill the apple' is a negated operation with no final predicate and no via_state", g["operations"][0]["neg"] and not g["final_predicates"]
          and all("via_state" not in k for k in G.retrieval_key(g)))
    g = G.parse_goal("I don't want the bread cold.", "en")
    check("B10 a negated wish makes no wish and no key", not g["wanted_states"] and G.retrieval_key(g) == [] and g["facts"][0]["kind"] == "negated_wish")
    g = G.parse_goal("Put it on the counter", "en", holding="Bowl")
    check("B11 a caller-supplied holding state resolves the pronoun and is marked inherited", g["operations"][0]["obj"][0]["type"] == "Bowl"
          and g["operations"][0]["obj"][0]["inherited"])
    g = G.parse_goal("You are holding the bowl. Put it on the counter.", "en")
    check("B12 'You are holding X' is context, not a goal; the pronoun resolves to it", g["context"]["holding"]["type"] == "Bowl" and len(g["operations"]) == 1
          and G.canonical(g) == "put(Bowl>CounterTop)")
    g = G.parse_goal("A cabinet has been left open. Close it.", "en")
    check("B13 a stated state of an earlier mention is not inherited as the goal of the pronoun", G.canonical(g) == "close(Cabinet>None)"
          and not [f for f in g["final_predicates"] if f.get("from") == "stated_state"])
    check("B14 FA elided object is inherited and flagged",
          G.parse_goal("کاهو را پیدا کن و بردار.", "fa")["operations"][1]["obj"][0].get("inherited") is True)

    # ===================================================================================== C. golden cases
    bad = []
    for t, l, status, canon in GOLDEN:
        g = G.parse_goal(t, l)
        if g["status"] != status or G.canonical(g, surfaces=True) != canon:
            bad.append((t, g["status"], G.canonical(g, surfaces=True), "expected", status, canon))
    check("C1 golden cases (%d EN/FA paraphrases + the contrastive set)" % len(GOLDEN), not bad, bad[:3])
    en = sum(1 for _t, l, _s, _c in GOLDEN if l == "en")
    check("C2 both languages are covered", en >= 30 and len(GOLDEN) - en >= 25, (en, len(GOLDEN)))
    pairs = [("Put the apple in the fridge.", "سیب را داخل یخچال بگذار."), ("Turn on the stove.", "اجاق را روشن کن."),
             ("Heat the bread in the microwave.", "نان را در مایکروویو گرم کن."),
             ("Wash the plate and put it in the cabinet.", "بشقاب را بشور و در کابینت بگذار."), ("I want the lettuce cold.", "کاهو را سرد می‌خواهم."),
             ("Don't open the fridge.", "در یخچال را باز نکن."), ("Put both mugs in the sink.", "هر دو لیوان را در سینک بگذار."),
             ("Take the apple out of the fridge and slice it.", "سیب را از یخچال بردار و برش بزن.")]
    check("C3 EN / FA twins give the same key", all(G.canonical(G.parse_goal(a, "en")) == G.canonical(G.parse_goal(b, "fa")) for a, b in pairs),
          [(a, G.canonical(G.parse_goal(a, "en")), G.canonical(G.parse_goal(b, "fa"))) for a, b in pairs])
    g = G.parse_goal("Before you close the drawer, put the fork inside.", "en")
    check("C4 'before X, Y' reorders: Y first, explicit temporal edge", G.canonical(g) == "put(Fork>Drawer) close(Drawer>None)" and g["temporal"][0]["rel"] == "before"
          and g["temporal"][0]["explicit"] and g["order"] == [1, 0])
    g = G.parse_goal("Open the microwave, then turn on the toaster.", "en")
    check("C5 'then' is an explicit temporal edge in list order", g["temporal"][0]["explicit"] and g["order"] == [0, 1])
    check("C6 quantities: 'two apples', 'both mugs', 'all the cabinets'",
          [q["quantity"] for q in G.parse_goal("Put two apples in the fridge and both mugs in the sink.", "en")["quantities"]] == [2, "all"]
          and G.parse_goal("Close all the cabinets.", "en")["quantities"][0]["quantity"] == "all")
    check("C7 negation is carried on the operation in both languages",
          G.parse_goal("Never open the fridge.", "en")["operations"][0]["neg"] and G.parse_goal("فریزر را باز نکن.", "fa")["operations"][0]["neg"])
    check("C8 a tap has no door: open/close on a Faucet is a switch", G.canonical(G.parse_goal("شیر آب را باز کن.", "fa")) == "toggle:on(Faucet>None)")
    check("C9 FA lexicon gaps closed: table, turn it on, spoon is a Spoon",
          G.canonical(G.parse_goal("کاسه را روی میز بگذار.", "fa")) == "put(Bowl>CounterTop)"
          and G.canonical(G.parse_goal("تخم‌مرغ را داخل مایکروویو بگذار و روشنش کن.", "fa")) == "put(Egg>Microwave) toggle:on(Microwave>None)"
          and ("قاشق", "Spoon") in G.glossary(G.parse_goal("قاشق را بشور.", "fa")))
    g = G.parse_goal("I need something to cut with.", "en")
    check("C10 a tool need keeps the operation, with the knife family as affordance only", g["needs"][0]["op"] == "slice"
          and set(G.affordance_types(g)) == {"ButterKnife"} and not g["operations"] and not g["final_predicates"])   # A26: was {"Knife", "ButterKnife"}
    g = G.parse_goal("I need a knife to cut the bread.", "en")
    # A26 (K-B2b): the bare knife is still the wished item, but it stays unknown ("?"; was type Knife, canonical "#item=Knife")
    check("C11 'a knife to cut the bread': the knife is the wished item (unknown, '?'), cutting is its purpose, not a step", not g["operations"]
          and g["needs"][0]["type"] is None and g["needs"][0].get("surface") == "knife" and G.canonical(g) == "slice(Bread>None#item=?)", G.canonical(g))
    g = G.parse_goal("The fork, the spoon and the knife have been used. Put them where they get washed.", "en")
    check("C12 cutlery: three puts to the function 'washed', Sink / SinkBasin as affordance, nothing in final_predicates",
          G.canonical(g) == "put(Fork>None~washed) put(Spoon>None~washed)" and set(G.affordance_types(g)) == {"Sink", "SinkBasin"}   # A26: the knife no longer resolves (was a third put(Knife>None~washed))
          and g["final_predicates"] == [])

    # ===================================================================================== D. measures
    import ltm_package as P
    root = os.path.dirname(HERE)
    _ds = json.load(open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8"))
    items = {i["id"]: i for i in _ds["items"] + [a["item"] for a in _ds.get("archived", [])]}   # A26: the historical signal sets cite archived items
    sig = json.load(open(os.path.join(root, "docs", "final-test", "kitchen_memory", "signal_set.json"), encoding="utf-8"))
    st = [G.parse_goal(items[p["item"]][p["lang"]], p["lang"])["status"] for p in sig["presentations"]]
    check("D1 the 24 signal presentations: all 'ok', no fallback", st.count("ok") == 24 and len(st) == 24, st)
    legacy_fb = [p for p in sig["presentations"] if not LC.parse(items[p["item"]][p["lang"]])["ok"]]
    # A26 (K-B2b): the lexicon no longer has a Knife concept, so the LEGACY parser falls back on one more knife presentation (was 15)
    check("D2 the legacy parser fell back on 16 of the 24 (reference point; 15 before the A26 lexicon)", len(legacy_fb) == 16, len(legacy_fb))
    nt = P.nontest_items()
    sts, agree, dis, indirect_ok = [], 0, [], True
    for it in nt:
        a, b = G.parse_goal(it["en"], "en"), G.parse_goal(it["fa"], "fa")
        sts += [a["status"], b["status"]]
        if G.canonical(a) == G.canonical(b):
            agree += 1
        else:
            dis.append(it["id"])
        for g in (a, b):
            if g["status"] == "indirect" and (g["operations"] or g["needs"]):
                indirect_ok = False
    check("D3 non-test items (116 x EN+FA): zero fallbacks (partial / unsupported)", sts.count("partial") + sts.count("unsupported") == 0, {s: sts.count(s) for s in set(sts)})
    check("D4 non-test items: EN/FA key agreement >= 0.98", agree / len(nt) >= 0.98, (agree, len(nt), dis))
    # Arash decision (8), K-B2b: the egg item's Persian text now says to close the door, so its EN/FA keys agree (was the one disagreement)
    check("D5 no EN/FA key disagreement on the non-test items (the egg FA text was fixed, decision 8)", dis == [], dis)
    check("D6 'indirect' is rare and only on statements without an operation word", sts.count("indirect") <= 12 and indirect_ok, sts.count("indirect"))

    # ===================================================================================== E. query integration
    q = P.make_query("I want to eat this apple cold later.", "en", None, [])
    check("E1 make_query carries the compact `intent` key, not the raw schema", q["intent"]["schema"] == "goal.schema/1" and "tokens" not in json.dumps(q["intent"]))
    check("E2 intent.key is the retrieval key: eat (desire) + chill via_state", [k["op"] for k in q["intent"]["key"]] == ["eat", "chill"]
          and q["intent"]["key"][1]["via_state"] == "cold" and q["intent"]["affordance_types"] == ["Fridge"])
    check("E3 legacy fields keep their contract: parser_fallback == not parse.ok", q["parser_fallback"] == (not q["parse"]["ok"]))
    check("E4 types / ops / goal_states are unioned with the schema", "Apple" in q["types"] and "chill" in q["ops"] and {"type": "Apple", "state": "cold"} in q["goal_states"])
    try:
        P._query_check(q)
        forb = True
    except ValueError:
        forb = False
    check("E5 the query passes the forbidden-key check (no level / item / plan / gold)", forb)
    P.USE_GOAL_SCHEMA = False
    q0 = P.make_query("I want to eat this apple cold later.", "en", None, [])
    P.USE_GOAL_SCHEMA = True
    check("E6 USE_GOAL_SCHEMA=False restores the legacy query exactly", "intent" not in q0 and q0["ops"] == [] and q0["types"] == ["Apple"])
    q = P.make_query("Put the egg in the microwave, close the door, and turn it on.", "en", None, [])
    check("E7 toggle maps to the legacy tokens toggle and toggle_on", "toggle_on" in q["ops"] and "toggle" in q["ops"])
    q_fa = P.make_query("سیب را سرد می‌خواهم.", "fa", None, [])
    q_en = P.make_query("I want the apple cold.", "en", None, [])
    check("E8 EN and FA twins give the identical intent key", q_fa["intent"]["key"] == q_en["intent"]["key"])
    q = P.make_query("hello there", "en", None, [])
    check("E9 unsupported request: intent.status 'unsupported', fallback flagged with a reason",
          q["intent"]["status"] == "unsupported" and q["intent"]["fallback"] and q["intent"]["reason"])
    q = P.make_query("The tomato has gone bad.", "en", None, [])
    check("E10 indirect request: not a schema fallback, the key is the stated state only", q["intent"]["status"] == "indirect" and not q["intent"]["fallback"]
          and q["intent"]["key"] == [{"op": None, "obj": "Tomato", "dest": None, "state": "bad"}])

    # ===================================================================================== F. no leakage
    src = open(os.path.join(HERE, "goal_schema.py"), encoding="utf-8").read()
    check("F1 goal_schema.py reads no dataset, plan, goal or simulator", not re.search(r"general\.json|commands\.jsonl|reference_plan|\bgold\b|ai2thor|import requests|urllib|ollama", src))
    check("F2 no dataset item id appears in goal_schema.py", not re.search(r"g-l[123]-", src))
    check("F3 no FloorPlan constant in goal_schema.py", "FloorPlan" not in src)

    # ===================================================================================== G. J-P1-parser: EN / FA twins and contrasts
    # (EN, FA, expected key). Written from general kitchen language, not from any dataset item.
    TWINS = [
        ("Fetch a knife.", "یک چاقو بیاور.", "pick(?>None)"),                   # A26 (K-B2b): was pick(Knife>None)
        ("Bring me a spoon.", "برایم یک قاشق بیاور.", "pick(Spoon>None)"),
        ("Get me a fork.", "یک چنگال به من بده.", "pick(Fork>None)"),
        ("Give me the butter knife.", "کارد کره را به من بده.", "pick(ButterKnife>None)"),
        ("Pick up the knife.", "کارد را بردار.", "pick(?>None)"),               # A26 (K-B2b): was pick(Knife>None)
        ("Pick up the butter knife.", "چاقوی کره را بردار.", "pick(ButterKnife>None)"),
        ("Get the lettuce from the fridge.", "کاهو را از یخچال بردار.", "pick(Lettuce>None<Fridge)"),
        ("Slice the potato.", "سیب‌زمینی را برش بزن.", "slice(Potato>None)"),
        ("Cut the bread.", "نان را ببر.", "slice(Bread>None)"),
        ("Slice the potato.", "سیب زمینی را خرد کن.", "slice(Potato>None)"),
        ("Do not slice the bread.", "نان را نبر.", "slice(Bread>None!)"),
        ("Do not slice the potato.", "سیب‌زمینی را برش نزن.", "slice(Potato>None!)"),
        ("Do not wash the apple.", "سیب را نشور.", "wash(Apple>None!)"),
        ("Bring something to slice with.", "یک چیزی بیاور که با آن برش بزنم.", "slice(None>None#tool)"),
        ("Bring me something to cut with.", "چیزی برای بریدن بیاور.", "slice(None>None#tool)"),
        ("Heat the apple in the microwave.", "سیب را توی مایکروفر گرم کن.", "heat(Apple>Microwave)"),
        ("Turn on the microwave.", "مایکروویو را راه بینداز.", "toggle:on(Microwave>None)"),
        ("Run the microwave.", "مایکروویو را روشن کن.", "toggle:on(Microwave>None)"),
        ("Cool the apple.", "سیب را خنک کن.", "chill(Apple>None)"),
        ("Chill the apple.", "سیب را سرد کن.", "chill(Apple>None)"),
        ("Flip the egg.", "تخم‌مرغ را برگردان.", "flip(Egg>None)"),
        ("Flip the egg.", "تخم‌مرغ را پشت و رو کن.", "flip(Egg>None)"),
        ("Bring me the pan.", "ماهیتابه را بیاور.", "pick(Pan>None)"),
        ("Bring me the pot.", "قابلمه را بیاور.", "pick(Pot>None)"),
        ("Put the spatula in the drawer.", "کفگیر را در کشو بگذار.", "put(Spatula>Drawer)"),
    ]
    bad = []
    for en, fa, want in TWINS:
        ke, kf = G.canonical(G.parse_goal(en, "en")), G.canonical(G.parse_goal(fa, "fa"))
        if not (ke == kf == want):
            bad.append((en, ke, kf, want))
    check("G1 %d EN/FA twins give the identical expected key" % len(TWINS), not bad, str(bad[:3]))
    # contrasts: wording that differs in meaning must NOT collapse to one key
    CONTRAST = [("Fetch a knife.", "Slice the potato."), ("Slice the potato.", "Do not slice the potato."),
                ("Bring something to slice with.", "Slice the potato."), ("Bring something to slice with.", "Fetch a knife."),
                ("Pick up the fork.", "Pick up the spoon."), ("Pick up the knife.", "Pick up the butter knife."),
                ("Heat the apple in the microwave.", "Cool the apple."), ("Flip the egg.", "Slice the egg."),
                ("Turn on the microwave.", "Turn off the microwave.")]
    same = [(a, b) for a, b in CONTRAST if G.canonical(G.parse_goal(a, "en")) == G.canonical(G.parse_goal(b, "en"))]
    check("G2 meaning contrasts stay distinct in English", not same, str(same))
    CONTRAST_FA = [("یک چاقو بیاور.", "سیب‌زمینی را برش بزن."), ("نان را ببر.", "نان را نبر."), ("چنگال را بردار.", "قاشق را بردار."),
                   ("چاقو را بردار.", "کارد کره را بردار."), ("سیب را گرم کن.", "سیب را خنک کن."), ("مایکروویو را روشن کن.", "مایکروویو را خاموش کن."),
                   ("یک چیزی بیاور که با آن برش بزنم.", "یک چاقو بیاور.")]
    same = [(a, b) for a, b in CONTRAST_FA if G.canonical(G.parse_goal(a, "fa")) == G.canonical(G.parse_goal(b, "fa"))]
    check("G3 meaning contrasts stay distinct in Persian", not same, str(same))
    # spelling and spacing variants land on one key
    VARIANTS = [["سیب‌زمینی را برش بزن", "سیب زمینی را برش بزن", "سیب‌زمینی  را   برش بزن!"],
                ["مایکروویو را روشن کن", "مایکروفر را روشن کن", "ماکروویو را روشن کن", "مایکرو ویو را روشن کن"],
                ["ماهیتابه را بردار", "ماهی‌تابه را بردار", "تابه را بردار"], ["تخم‌مرغ را برگردان", "تخم مرغ را برگردان"],
                ["چاقو را بردار", "كارد را بردار"]]
    bad = [v for v in VARIANTS if len({G.canonical(G.parse_goal(t, "fa")) for t in v}) != 1]
    check("G4 Persian spelling / spacing / orthography variants give one key", not bad, str(bad))
    ev = ["Slice the potato", "slice the potato", "SLICE THE POTATO!", "Please slice the potato.", "Slice  the   potato"]
    check("G5 English case / spacing / politeness variants give one key", len({G.canonical(G.parse_goal(t, "en")) for t in ev}) == 1)
    # ambiguity policy: ببر is cut OR carry. Declared, recorded, never silent.
    g_cut, g_carry = G.parse_goal("نان را ببر", "fa"), G.parse_goal("نان را ببر روی میز", "fa")
    check("G6 'ببر' with no destination = cut, with a destination = carry; the choice is recorded in schema['ambiguities']",
          G.canonical(g_cut) == "slice(Bread>None)" and G.canonical(g_carry).startswith("put(Bread>CounterTop)")
          and g_cut["ambiguities"] and g_cut["ambiguities"][0]["chosen"] == "slice" and g_carry["ambiguities"][0]["chosen"] == "put",
          G.canonical(g_carry))
    g = G.parse_goal("ملاقه را بذار توی کشو", "fa")
    check("G7 an unknown noun stays a visible '?' (no silent guess to a known type)", "put(?>Drawer)" == G.canonical(g))
    check("G8 'get' as a state or idiom is not a fetch", G.canonical(G.parse_goal("Get rid of the egg.", "en")) != "pick(Egg>None)"
          and "pick" not in G.canonical(G.parse_goal("I want the apple to get cold.", "en")))
    k9 = G.canonical(G.parse_goal("Fetch a knife but do not slice the bread.", "en"))
    check("G9 negation is scoped: fetch the knife, do not slice", k9 == "pick(?>None) slice(Bread>None!)", k9)   # A26: was pick(Knife>None)
    check("G10 a fetch of a tool never creates a slice step", "slice" not in G.canonical(G.parse_goal("Fetch a knife.", "en"))
          and "slice" not in G.canonical(G.parse_goal("یک چاقو بیاور", "fa")))
    # Astra J-P1-parser review (docs/final-test/kitchen_memory/astra/J-P1-parser.txt): the five risks, each with its contrast test
    ck = lambda t, l: G.canonical(G.parse_goal(t, l))
    check("G11 ببر takes its destination from ITS OWN clause (cut, then put) vs carry-to",
          ck("نان را ببر و در بشقاب بگذار", "fa") == ck("Cut the bread and put it on the plate", "en") == "slice(Bread>None) put(Bread>Plate)"
          and ck("نان را به بشقاب ببر", "fa") == "put(Bread>Plate)", ck("نان را ببر و در بشقاب بگذار", "fa"))
    # A26 (K-B2b): the sample object is the fork (was the knife, which no longer resolves to a type); the check is about the recipient
    ops_en = G.parse_goal("Give me the fork", "en")["operations"][0]
    ops_fa = G.parse_goal("چنگال را به من بده", "fa")["operations"][0]
    check("G12 the recipient of a hand-over survives in the schema (not in the key); a plain pick has none",
          ops_en.get("recipient") == ops_fa.get("recipient") == "speaker" and not G.parse_goal("Pick up the fork", "en")["operations"][0].get("recipient")
          and ck("Give me the fork", "en") == ck("Pick up the fork", "en"))
    check("G13 tool need vs item need: 'with' / 'با آن' = tool; 'آن را' = the thing cut",
          ck("چیزی بیاور که با آن ببرم", "fa") == ck("Bring something to slice with", "en") == "slice(None>None#tool)"
          and ck("چیزی بیاور که آن را ببرم", "fa") == "slice(None>None#item)"
          and G.parse_goal("Bring something to slice", "en")["ambiguities"][0]["kind"] == "need_role")
    ke_l, ke_w = G.retrieval_key(G.parse_goal("Bring a ladle", "en")), G.retrieval_key(G.parse_goal("Bring a whisk", "en"))
    check("G14 two different unknown nouns keep their own surface in the key (not one shared '?')",
          ke_l[0].get("obj_surface") == "ladle" and ke_w[0].get("obj_surface") == "whisk" and ke_l != ke_w
          and G.retrieval_key(G.parse_goal("ملاقه را بیاور", "fa"))[0].get("obj_surface") != G.retrieval_key(G.parse_goal("همزن را بیاور", "fa"))[0].get("obj_surface"))
    check("G15 negation does not leak across clauses (EN and FA)",
          ck("Don't slice the bread; heat the soup", "en") == ck("نان را نبر؛ سوپ را گرم کن", "fa") == "slice(Bread>None!) heat(?>None)"
          and ck("Slice the bread; don't heat the soup", "en") == ck("نان را ببر؛ سوپ را گرم نکن", "fa") == "slice(Bread>None) heat(?>None!)")
    print("\n%d/%d passed" % (PASS, PASS + FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
