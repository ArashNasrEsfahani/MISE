"""Regenerates sim/step5/lexicon/concepts.json (the A4 concept lexicon). Authored from general English/Persian knowledge
and the 49-class detector list; NOT derived from any dataset command, plan or evaluation item. Run from the repo root:
    python sim/step5/lexicon/_build_concepts.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
classes = json.loads((HERE / "useful_types.json").read_text(encoding="utf-8"))["detector_classes"]
# A26 item 1 (K-B2b 2026-10-13): no concept for an excluded type (type_scope.EXCLUDED_TYPES = Knife). The bare words knife /
# چاقو / کارد then resolve to nothing ("?", A26 3(b)); the butter-knife phrases keep resolving to ButterKnife.
EXCLUDED = ("Knife",)
ALL_CLASSES = list(classes)
classes = [c for c in classes if c not in EXCLUDED]

EN_EXTRA = {
    "Fridge": ["refrigerator", "fridge"], "Cabinet": ["cupboard", "cabinet"], "CounterTop": ["counter", "countertop", "counter top", "worktop"],
    "GarbageCan": ["trash can", "garbage can", "bin", "trash bin", "garbage bin", "rubbish bin"], "StoveBurner": ["stove", "burner", "stove burner", "cooktop"],
    "StoveKnob": ["stove knob", "knob", "burner knob"], "CoffeeMachine": ["coffee machine", "coffee maker"], "Faucet": ["faucet", "tap"],
    "LightSwitch": ["light switch"], "SinkBasin": ["sink basin", "basin"], "Sink": ["sink"], "PaperTowelRoll": ["paper towel roll", "paper towel", "paper towels"],
    "DishSponge": ["dish sponge", "sponge"], "SoapBottle": ["soap bottle", "soap dispenser", "dish soap"], "WineBottle": ["wine bottle"],
    "ShelvingUnit": ["shelving unit"], "HousePlant": ["house plant", "houseplant", "plant"], "CreditCard": ["credit card", "card"],
    "ButterKnife": ["butter knife"], "SaltShaker": ["salt shaker", "salt"], "PepperShaker": ["pepper shaker", "pepper"],
    "Microwave": ["microwave", "microwave oven"], "Egg": ["egg"],
}
FA = {
    "Apple": ["سیب"], "Book": ["کتاب"], "Bottle": ["بطری"], "Bowl": ["کاسه"], "Bread": ["نان"], "ButterKnife": ["کارد کره", "چاقوی کره"],
    "Cabinet": ["کابینت", "کمد"], "CoffeeMachine": ["قهوه ساز", "دستگاه قهوه"], "CounterTop": ["کانتر", "میز کار"],
    "CreditCard": ["کارت اعتباری", "کارت بانکی"], "Cup": ["فنجان"], "DishSponge": ["اسفنج ظرفشویی", "اسفنج"], "Drawer": ["کشو"],
    "Egg": ["تخم مرغ"], "Faucet": ["شیر آب"], "Floor": ["کف", "زمین"], "Fork": ["چنگال"], "Fridge": ["یخچال"],
    "GarbageCan": ["سطل زباله", "سطل اشغال", "سطل آشغال"], "HousePlant": ["گیاه"], "Kettle": ["کتری"], "Knife": ["چاقو", "کارد"],
    "Lettuce": ["کاهو"], "LightSwitch": ["کلید برق", "کلید چراغ"], "Microwave": ["مایکروویو", "مایکروفر"], "Mug": ["لیوان", "ماگ"],
    "Pan": ["تابه", "ماهیتابه"], "PaperTowelRoll": ["دستمال کاغذی", "حوله کاغذی"], "PepperShaker": ["فلفل پاش", "فلفل دان", "فلفل"],
    "Plate": ["بشقاب"], "Pot": ["قابلمه", "دیگ"], "Potato": ["سیب زمینی"], "SaltShaker": ["نمک پاش", "نمکدان", "نمک"],
    "Shelf": ["قفسه"], "ShelvingUnit": ["واحد قفسه", "قفسه بندی"], "Sink": ["سینک", "ظرفشویی"], "SinkBasin": ["حوضچه سینک"],
    "SoapBottle": ["بطری صابون", "صابون مایع"], "Spatula": ["کفگیر", "اسپاتول"], "Spoon": ["قاشق"], "Statue": ["مجسمه"],
    "Stool": ["چهارپایه"], "StoveBurner": ["اجاق", "شعله اجاق"], "StoveKnob": ["دکمه اجاق", "دستگیره اجاق"], "Toaster": ["توستر"],
    "Tomato": ["گوجه فرنگی", "گوجه"], "Vase": ["گلدان"], "Window": ["پنجره"], "WineBottle": ["بطری شراب"],
}


def camel(name):
    out = ""
    for i, ch in enumerate(name):
        if ch.isupper() and i:
            out += " "
        out += ch.lower()
    return out


def plural(w):
    if w.endswith("knife"):
        return w[:-5] + "knives"
    if w.endswith(("s", "x", "ch", "sh", "o")):
        return w + "es"
    if w.endswith("y") and w[-2:-1] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


types = {}
for c in classes:
    assert c in FA, c
    en = sorted(set([camel(c)] + [camel(c).replace(" ", "")] + EN_EXTRA.get(c, [])))
    en_pl = sorted({plural(x) for x in en if not x.endswith("s")})
    types[c] = {"en": en, "en_plural": en_pl, "fa": FA[c], "name_en": camel(c), "name_fa": FA[c][0]}
# A26: the excluded types' surface words, kept OUT of `types` (the parser never resolves them) and listed only for the D2 alias text
# check (matrix/treatment.excluded_alias_hits: no agent-facing text may name an excluded type or its aliases)
excluded_aliases = {}
for c in ALL_CLASSES:
    if c in EXCLUDED:
        en = sorted(set([camel(c)] + [camel(c).replace(" ", "")] + EN_EXTRA.get(c, [])))
        excluded_aliases[c] = {"en": en, "en_plural": sorted({plural(x) for x in en if not x.endswith("s")}), "fa": FA[c],
                               "note": "A26 excluded type: alias list for the D2 text check only; never a parser concept"}

LIGHT_POS = ["کن", "کنید", "بکن", "بکنید", "بکنین", "کنین"]
LIGHT_NEG = ["نکن", "نکنید", "نکنی", "نکنین"]
GIVE_POS = ["بده", "بدهید", "بدین", "بدی"]
GIVE_NEG = ["نده", "ندهید", "ندین"]


def fa_light(nouns, pos=LIGHT_POS, neg=LIGHT_NEG):
    p, n = [], []
    for w in nouns:
        for x in pos:
            p += [f"{w} {x}", f"{w}{x}"]
        for x in neg:
            n += [f"{w} {x}", f"{w}{x}"]
    return p, n


def fa_verb(nouns=(), simple_pos=(), simple_neg=(), give=(), inf=()):
    p, n = fa_light(nouns)
    gp, gn = fa_light(give, GIVE_POS, GIVE_NEG)
    return {"fa": sorted(set(p + gp + list(simple_pos) + list(inf))), "fa_neg": sorted(set(n + gn + list(simple_neg)))}


verbs = {
    "pick": {"en": ["pick up", "pick", "take", "grab", "get", "pickup", "lift"],
             "en_ing": ["picking up", "picking", "taking", "grabbing", "getting", "lifting"],
             **fa_verb(simple_pos=["بردار", "بردارید", "بگیر", "بگیرید", "بر دار"], simple_neg=["برندار", "برندارید", "نگیر", "نگیرید"],
                       inf=["برداشتن", "گرفتن"])},
    "put": {"en": ["put", "place", "set", "store", "keep"], "en_ing": ["putting", "placing", "setting", "storing", "keeping"],
            **fa_verb(simple_pos=["بگذار", "بگذارید", "بذار", "بذارید", "بزار", "بزارید", "بنه"],
                      simple_neg=["نگذار", "نگذارید", "نذار", "نذارید", "نزار"], give=["قرار"], inf=["گذاشتن", "قرار دادن"])},
    "open": {"en": ["open"], "en_ing": ["opening"], **fa_verb(nouns=["باز"], inf=["باز کردن"])},
    "close": {"en": ["close", "shut"], "en_ing": ["closing", "shutting"],
              **fa_verb(nouns=["بسته"], simple_pos=["ببند", "ببندید"], simple_neg=["نبند", "نبندید"], inf=["بستن", "بسته کردن"])},
    "slice": {"en": ["slice", "cut"], "en_ing": ["slicing", "cutting"],
              **fa_verb(nouns=["خرد", "قطعه"], give=["برش"], inf=["برش دادن", "خرد کردن"])},
    "toggle:on": {"en": ["turn on", "switch on", "power on"], "en_ing": ["turning on", "switching on"],
                  **fa_verb(nouns=["روشن"], inf=["روشن کردن"])},
    "toggle:off": {"en": ["turn off", "switch off", "power off"], "en_ing": ["turning off", "switching off"],
                   **fa_verb(nouns=["خاموش"], inf=["خاموش کردن"])},
    "toggle": {"en": ["turn", "switch", "toggle"], "en_ing": ["turning", "switching", "toggling"], "fa": [], "fa_neg": []},
    "find": {"en": ["find", "locate", "search for", "look for"], "en_ing": ["finding", "locating", "searching for", "looking for"],
             **fa_verb(nouns=["پیدا"], simple_pos=["بگرد", "بگردید"], simple_neg=["نگرد", "نگردید"], inf=["پیدا کردن"])},
    "heat": {"en": ["heat", "warm", "heat up", "warm up"], "en_ing": ["heating", "warming", "heating up", "warming up"],
             **fa_verb(nouns=["گرم"], inf=["گرم کردن"])},
    "cook": {"en": ["cook", "bake"], "en_ing": ["cooking", "baking"],
             **fa_verb(simple_pos=["بپز", "بپزید"], simple_neg=["نپز", "نپزید"], inf=["پختن"])},
    "wash": {"en": ["wash", "clean", "rinse"], "en_ing": ["washing", "cleaning", "rinsing"],
             **fa_verb(nouns=["تمیز"], simple_pos=["بشور", "بشویید", "بشوی"], simple_neg=["نشور", "نشویید", "نشوی"], inf=["شستن"])},
    "fill": {"en": ["fill"], "en_ing": ["filling"], **fa_verb(nouns=["پر"], inf=["پر کردن"])},
    "empty": {"en": ["empty out", "pour out"], "en_ing": ["emptying out", "pouring out"], **fa_verb(nouns=["خالی"], inf=["خالی کردن"])},
    "drop": {"en": ["drop"], "en_ing": ["dropping"],
             **fa_verb(nouns=["رها"], simple_pos=["بینداز", "بیندازید", "بیفکن"], simple_neg=["نینداز", "نیندازید"], inf=["انداختن"])},
    "break": {"en": ["break", "smash"], "en_ing": ["breaking", "smashing"],
              **fa_verb(simple_pos=["بشکن", "بشکنید"], simple_neg=["نشکن", "نشکنید"], inf=["شکستن"])},
}
names = {
    "pick": ["pick up", "برداشتن"], "put": ["put", "گذاشتن"], "open": ["open", "باز کردن"], "close": ["close", "بستن"],
    "slice": ["slice", "برش دادن"], "toggle": ["toggle", "روشن یا خاموش کردن"], "find": ["find", "پیدا کردن"], "heat": ["heat", "گرم کردن"],
    "cook": ["cook", "پختن"], "wash": ["wash", "شستن"], "fill": ["fill", "پر کردن"], "empty": ["empty", "خالی کردن"],
    "drop": ["drop", "انداختن"], "break": ["break", "شکستن"],
}
req_names = {"on": ["turn on", "روشن کردن"], "off": ["turn off", "خاموش کردن"]}
adjectives = {
    "open": {"en": ["open", "opened"], "fa": ["باز"]}, "closed": {"en": ["closed", "shut"], "fa": ["بسته"]},
    "sliced": {"en": ["sliced"], "fa": ["خرد شده", "برش خورده"]}, "clean": {"en": ["clean", "washed"], "fa": ["تمیز", "شسته"]},
    "dirty": {"en": ["dirty"], "fa": ["کثیف"]}, "hot": {"en": ["hot", "heated"], "fa": ["داغ", "گرم"]},
    "cold": {"en": ["cold", "chilled"], "fa": ["سرد"]}, "filled": {"en": ["filled", "full"], "fa": ["پر"]},
    "empty": {"en": ["empty"], "fa": ["خالی"]}, "broken": {"en": ["broken"], "fa": ["شکسته"]},
    "on": {"en": ["turned on", "switched on", "lit"], "fa": ["روشن"]}, "off": {"en": ["turned off", "switched off"], "fa": ["خاموش"]},
}
lex = {
    "_doc": "A4 concept lexicon (docs/step5-plan.md 8.2). Authored from general knowledge + the 49 detector classes (useful_types.json); "
            "NOT derived from any dataset command or reference plan. Regenerate with sim/step5/lexicon/_build_concepts.py.",
    "version": "1", "source": "authored (general EN/FA knowledge); detector class list from useful_types.json",
    "dataset_fitted": False, "fp1_tuned": False, "ontology_verified": False,
    "types": types, "excluded_aliases": excluded_aliases, "verbs": verbs, "names": names, "req_names": req_names, "adjectives": adjectives,
    "negation": {"en": ["not", "dont", "never", "without", "no", "avoid", "doesnt", "wont", "shouldnt", "mustnt", "cant", "cannot"],
                 "fa": ["نه", "بدون", "هرگز", "نباید", "هیچ وقت", "هیچوقت"]},
    "prep_dest": {"en": ["in", "into", "inside", "inside of", "on", "onto", "on top of", "to", "in to"],
                  "fa": ["در", "توی", "تو", "داخل", "درون", "روی", "بالای", "به"]},
    "prep_src": {"en": ["from", "out of", "off of"], "fa": ["از", "بیرون از"]},
    "accusative_fa": ["را"],
    "pronouns": {"en": ["it", "them", "that", "this", "one", "those", "these"], "fa": ["آن", "آنها", "اون", "اینها", "همین", "این"]},
    "quantifiers": {"en": ["all", "both", "every", "each", "two", "three", "four", "some", "any", "several", "2", "3", "4"],
                    "fa": ["همه", "هر", "دو", "سه", "چند", "چندتا", "هردو", "2", "3", "4"]},
    "complex_markers": {"en": ["before", "after", "until", "if", "unless", "when", "while", "once", "because", "or", "instead", "otherwise"],
                        "fa": ["قبل", "اگر", "مگر", "وقتی", "یا", "تا", "چون"]},
    "complex_bigrams_fa": ["پس از", "بعد از", "قبل از", "به جای"],
    "hard_sep": {"en": ["then", "but", "afterwards", "after that", "and then"], "fa": ["سپس", "بعد", "بعد از آن", "ولی", "اما", "و سپس", "بعدش"]},
    "soft_sep": {"en": ["and"], "fa": ["و"]},
    "determiners": {"en": ["the", "a", "an", "my", "your", "our", "that", "this", "those", "these", "of", "please"], "fa": ["یک", "این", "آن", "لطفا"]},
    # W4a (prereg A17): FA partitive "one of the X" is dropped before parsing; FA elided object (second clause inherits the previous
    # clause's object) applies only between these carry-chain verbs (never open / close / toggle).
    "partitive_fa": ["یکی از"],
    # "در X را باز/ببند" = "open/close the door of X": the leading noun 'در' (door; also the preposition 'in') is dropped so X is the object.
    "door_of_fa": {"words": ["در"], "verbs": ["open", "close"]},
    "elided_object_fa": {"source_verbs": ["find", "pick", "heat", "cook", "wash", "slice", "fill", "empty", "put", "drop"],
                         "target_verbs": ["pick", "put", "drop", "heat", "cook", "wash", "slice", "fill", "empty", "break"]},
    "fa_light_verb_tokens": LIGHT_POS + LIGHT_NEG + GIVE_POS + GIVE_NEG,
}
out = HERE / "concepts.json"
out.write_text(json.dumps(lex, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("wrote", out.name, len(types), "types", len(verbs), "verbs")
