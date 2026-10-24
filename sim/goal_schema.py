"""Instruction -> goal schema (stage I-F1-goalschema, 2026-10-12).

A deterministic, rule-based reading of one instruction (English or Persian) into a small schema:
    entities, operations (with object / source / destination / tool / quantity / negation), final predicates, temporal order,
    plus needs ("something to cut with"), facts ("the fork has been used") and desires ("I want to eat this apple cold").
Every element carries a SUPPORTING SPAN (token range + surface text of the normalised instruction). Nothing is invented:

  * a goal element exists only because a span says it;
  * an AFFORDANCE (chill -> Fridge, cut -> ButterKnife) is reached through an operation / state / function word, is kept in the separate
    `affordances` list with the span of that word, and is never copied into `final_predicates`;
  * a state word that implies an operation ("cold" -> chill) is flagged `via_state` in the retrieval key; it never becomes a step;
  * a pronoun or an elided Persian object is resolved to an earlier span and flagged `inherited` / `via`;
  * a sentence that states a fact or a wish but names no operation is `indirect` (the facts are kept; no goal is made up);
  * a sentence with nothing usable is `unsupported` (a defined path with a reason), never a silent empty parse.

Knowledge is English only (types, operations, affordances); the Persian side is a lexicon of surface forms that lands on the same
English keys, so the same request in both languages gives the same `retrieval_key`. All tables are authored from general kitchen /
AI2-THOR knowledge. They are NOT derived from any dataset instruction, reference plan or goal.

No simulator, LLM, clock or randomness: the same text always gives the same schema.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    import ltm_concepts as LC
except ImportError:  # pragma: no cover
    from sim import ltm_concepts as LC

def N(s: str) -> str:
    """Normalise a Persian literal exactly as the instruction text is normalised (alef-madda folding, ZWNJ, ...)."""
    return " ".join(LC.prep(s))


SCHEMA = "goal.schema/1"
VERSION = "1"
H, S = LC.H, LC.S
P = "§P"          # Persian enclitic object pronoun split off its stem

# ------------------------------------------------------------------------------------------------------------ type knowledge
OPENABLE = frozenset({"Microwave", "Fridge", "Cabinet", "Drawer"})
TOGGLEABLE = frozenset({"Microwave", "StoveBurner", "StoveKnob", "Faucet", "LightSwitch", "CoffeeMachine", "Toaster"})
FIXTURES = frozenset({"Microwave", "Fridge", "Cabinet", "Drawer", "StoveBurner", "StoveKnob", "Faucet", "LightSwitch", "CoffeeMachine",
                      "Toaster", "Sink", "SinkBasin", "CounterTop", "Shelf", "ShelvingUnit", "GarbageCan", "Floor", "Window", "Stool"})
# an open/close word aimed at these is, physically, a switch (a tap has no door)
OPEN_AS_TOGGLE = frozenset({"Faucet"})

# Type aliases the shared concept lexicon lacks (general words for the same AI2-THOR types). The shared lexicon file is hash-pinned
# elsewhere and is not edited; these are looked up too and win only when longer than the lexicon match.
EXTRA_TYPES = {
    ("table",): ("CounterTop", "alias_ambiguous"), ("tabletop",): ("CounterTop", "alias_ambiguous"),
    ("kitchen", "counter"): ("CounterTop", ""), ("stovetop",): ("StoveBurner", ""), ("hob",): ("StoveBurner", ""),
    ("cooker",): ("StoveBurner", ""), ("oven",): ("Microwave", "alias_ambiguous"),
    ("freezer",): ("Fridge", "alias_ambiguous"), ("trash",): ("GarbageCan", ""), ("rubbish",): ("GarbageCan", ""),
    ("garbage",): ("GarbageCan", ""), ("dustbin",): ("GarbageCan", ""),
    ("lamp",): ("LightSwitch", "alias_ambiguous"), ("light",): ("LightSwitch", ""), ("lights",): ("LightSwitch", ""),
    ("coffee", "maker"): ("CoffeeMachine", ""), ("scrubber",): ("DishSponge", ""),
    ("butter", "knife"): ("ButterKnife", ""),
    # Persian
    ("میز",): ("CounterTop", "alias_ambiguous"), ("میز", N("آشپزخانه")): ("CounterTop", ""),
    ("سطل",): ("GarbageCan", "alias_ambiguous"), ("چراغ",): ("LightSwitch", ""), ("لامپ",): ("LightSwitch", ""),
    ("فریزر",): ("Fridge", "alias_ambiguous"), ("اجاق", "گاز"): ("StoveBurner", ""),
    ("قهوه", "ساز"): ("CoffeeMachine", ""), ("ظرف", "شویی"): ("Sink", ""),
    ("کارد", "کره", "خوری"): ("ButterKnife", ""), ("چاقوی", "کره"): ("ButterKnife", ""),
    ("فلفل", "پاش"): ("PepperShaker", ""), ("نمک", "پاش"): ("SaltShaker", ""),
    ("بطری", "صابون"): ("SoapBottle", ""), ("صابون",): ("SoapBottle", "alias_ambiguous"),
    ("اسفنج", "ظرف", "شویی"): ("DishSponge", ""), ("کفگیر",): ("Spatula", ""),
    # J-P1-parser: spelling / spacing variants and common synonyms (general kitchen vocabulary, not read from any dataset item)
    ("ماکروویو",): ("Microwave", ""), ("مایکرویو",): ("Microwave", ""), ("ماکروفر",): ("Microwave", ""), ("مایکرو", "ویو"): ("Microwave", ""),
    ("مایکرو", "فر"): ("Microwave", ""), ("ماکرو", "ویو"): ("Microwave", ""), ("ماکرو", "فر"): ("Microwave", ""),
    ("کفچه",): ("Spatula", ""), ("ماهی", "تابه"): ("Pan", ""), ("قابلمه",): ("Pot", ""),
    ("کارد", "کره"): ("ButterKnife", ""), ("چاقو", "کره"): ("ButterKnife", ""), ("چاقوی", "کره", "خوری"): ("ButterKnife", ""),
    ("چنگل",): ("Fork", ""),
    ("frying", "pan"): ("Pan", ""), ("skillet",): ("Pan", ""), ("saucepan",): ("Pot", ""),
    # A26 item 1 / 3(b) (K-B2b 2026-10-13): a knife phrase that is not a butter knife resolves to NO type ("?"); the phrase is
    # still consumed as one unknown noun so that "table knife" does not become a table (was ("Knife", "alias_ambiguous"))
    ("cutlery", "knife"): (None, "excluded_type"), ("table", "knife"): (None, "excluded_type"),
}
EXTRA_MAX = max(len(k) for k in EXTRA_TYPES)

# ------------------------------------------------------------------------------------------------------------ operations
# canonical English operation keys; toggle carries on/off. Phrases are normalised tokens (LC.prep output).
_OPS_EN = {
    "put": ["put", "place", "set", "store", "leave", "put away", "lay", "stash"],
    "pick": ["pick", "pick up", "take", "grab", "fetch", "collect", "lift", "bring", "carry"],
    "find": ["find", "locate", "look for", "search for", "spot"],
    "toggle:on": ["turn on", "switch on", "power on", "start", "light up", "run"],
    "toggle:off": ["turn off", "switch off", "power off", "shut off", "stop"],
    "open": ["open"], "close": ["close", "shut"],
    "slice": ["slice", "cut", "chop", "dice", "carve", "cut up"],
    "heat": ["heat", "warm", "reheat", "heat up", "warm up", "microwave"],
    "cook": ["cook", "bake", "fry", "boil", "grill", "roast"],
    "chill": ["chill", "cool", "refrigerate", "cool down"],
    "wash": ["wash", "rinse", "wash up"],
    "scrub": ["scrub", "scour"],
    "fill": ["fill", "fill up"], "empty": ["empty", "empty out", "pour out"], "drop": ["drop"],
    "break": ["break", "smash", "crack"],
    "flip": ["flip", "turn over", "flip over"],
    "eat": ["eat", "consume"], "drink": ["drink", "sip"],
    "serve": ["serve", "plate up"], "make": ["make", "prepare", "create"], "_ensure": ["make sure"],
    "keep": ["keep", "save", "preserve"],
    "discard": ["throw away", "throw out", "get rid of", "dispose of", "discard", "toss"],
    "tidy": ["tidy", "tidy up", "organize", "organise", "arrange", "straighten", "clean up"],
    "season": ["season"],
}
_SINGLE_EN = {ph: op for op, phs in _OPS_EN.items() if not op.startswith("_") for ph in phs if " " not in ph and op not in ("toggle:on", "toggle:off")}

# phrasal verbs whose particle may be separated from the verb by a pronoun / short object ("turn it on", "pick it up")
_PHRASAL = {"turn": {"on": "toggle:on", "off": "toggle:off", "over": "flip"}, "switch": {"on": "toggle:on", "off": "toggle:off"},
            "power": {"on": "toggle:on", "off": "toggle:off"}, "shut": {"off": "toggle:off"}, "pick": {"up": "pick"},
            "put": {"away": "put"}, "throw": {"away": "discard", "out": "discard"}, "tidy": {"up": "tidy"},
            "clean": {"up": "tidy"}, "warm": {"up": "heat"}, "heat": {"up": "heat"}, "cool": {"down": "chill"},
            "fill": {"up": "fill"}, "get": {"rid": "discard"}, "cut": {"up": "slice"}, "flip": {"over": "flip"}}
_WANT_EN = {"want", "wants", "need", "needs", "wish", "like", "prefer", "require", "requires", "d", "ll", "ensure"}
_WANT_FA = {"خواهم", "میخواهم", "میخواهیم", "میخوام", "لازم", "نیاز", "دوست", "میخواهد", "می"}

# Persian simple verbs: surface forms (imperative, subjunctive, infinitive) -> op
_VERBS_FA = {
    "put": ["بگذار", "بگذارید", "بذار", "بذارید", "بگذارم", "بگذاری", "گذاشتن", "بزار", "بگذارند"],
    "pick": ["بردار", "بردارید", "بردارم", "برداشتن", "بگیر", "بگیرید", "بگیرم", "گرفتن",
             "بیاور", "بیاورید", "بیاورم", "بیار", "بیارید", "بیارم", "بیاری", N("آوردن"), N("بیاورین"), N("بیاوریم")],
    "break": ["بشکن", "بشکنید", "بشکنم", "شکستن"],
    "close": ["ببند", "ببندید", "ببندم", "بستن"],
    "wash": ["بشور", "بشویید", "بشوی", "بشویم", "شستن", "بشوییم"],
    "cook": ["بپز", "بپزید", "بپزم", "پختن"],
    "eat": ["بخور", "بخورید", "بخورم", "خوردن", "بخوریم", "بخورد"],
    "drink": ["بنوش", "بنوشید", "بنوشم", "نوشیدن"],
    "drop": ["بینداز", "بیندازید", "بیفکن", "بیندازم", "انداختن"],
    "empty": ["بریز", "بریزید", "بریزم"],
    "flip": ["برگرداندن", "برگردان", "برگردانید", "برگردانم"],
    "scrub": ["سابیدن", "بساب", "بسابید", "بسابم"],
    "slice": ["بریدن"],
    "find": ["بیاب", "بیابید", "یافتن"],
    "season": [],
}
# noun + light verb ("X کن" ...): noun -> op
_LIGHT_NOUN_FA = {
    "روشن": "toggle:on", "خاموش": "toggle:off", "باز": "open", "بسته": "close", "گرم": "heat", "داغ": "heat", "سرد": "chill",
    "خنک": "chill", "تمیز": "wash", "پاک": "wash", "خالی": "empty", "پر": "fill", "پیدا": "find", "برش": "slice", "سرو": "serve",
    "درست": "make", N("آماده"): "make", "مرتب": "tidy", "رها": "drop", "قرار": "put", "سرخ": "cook", "خرد": "slice", "نگه": "keep",
}
_LIGHT_NOUN_FA.update({"تکه": "slice", "وارونه": "flip", "استارت": "toggle:on", "اجرا": "toggle:on"})   # J-P1-parser: "تکه کن", "وارونه کن", "استارت بزن", "اجرا کن"
_LIGHT_VERBS_FA = {"کن", "کنید", "بکن", "بکنید", "کنم", "بکنم", "کردن", "کنین", "بزن", "بزنید", "بزنم", "بده", "بدهید", "بدم", "بدین",
                   "دار", "داری", "دارید", "دارم", "بدهم", "شو", "شود", "شوید"}
_NEG_LIGHT_FA = {"نکن", "نکنید", "نکنی", "نکنین", "نزن", "نزنید", "نده", "ندهید"}
_NEG_EN = {"not", "dont", "never", "without", "avoid", "doesnt", "wont", "shouldnt", "mustnt", "cant", "cannot", "no"}
_NEG_FA = {"نه", "نباید", "هرگز", "نمی"}
_NEG_VERBS_FA = {"نگذار", "نذار", "نبند", "نشکن", "نخور", "نینداز", "نشور", "نپز", "نبر", "نیفکن"}
# J-P1-parser: the operation a negated simple verb refuses ("نبر" = do not cut): it is emitted as NEG + OP(neg) so "do not slice the bread"
# and "نان را نبر" give the same key
_NEG_VERB_OP_FA = {"نگذار": "put", "نذار": "put", "نبند": "close", "نشکن": "break", "نخور": "eat", "نینداز": "drop", "نشور": "wash",
                   "نپز": "cook", "نبر": "slice", "نیفکن": "drop", "نبریدش": "slice"}
# "ببر" is two Persian verbs: بریدن (cut) and بردن (carry). POLICY (declared, not guessed silently): with a destination word in the same
# sentence it is "carry to" (-> put); without one it is "cut" (-> slice). Either way the choice is recorded in schema['unresolved'].
_BORDAN_FA = {"ببر", "ببرید", "ببری", "ببرم", "نبر"}
_DEST_WORDS_FA = {"در", "توی", "تو", "داخل", "درون", "روی", "بالای", "به", "زیر", "کنار"}
_SUBJ_EN = {"nobody", "someone", "somebody", "everyone", "everybody", "anyone"}
_EVERY_EN = {"everything", "anything"}
_EVERY_FA = {"همهچیز"}

# Persian verbal nouns usable after "برای" ("for V-ing")
_VNOUN_FA = {"بریدن": "slice", "برگرداندن": "flip", "سابیدن": "scrub", "خوردن": "eat", "شستن": "wash", "پختن": "cook", "نوشیدن": "drink",
             "شکستن": "break", "بستن": "close", "برداشتن": "pick", "گذاشتن": "put", "پیدا": "find"}

# --------------------------------------------------------------------------------------------------------------------- states
_STATES = {
    "cold": ["cold", "chilled", "cool", "iced", "سرد", "خنک"], "hot": ["hot", "heated", "warm", "داغ", "گرم"],
    "clean": ["clean", "washed", "تمیز", "شسته"], "dirty": ["dirty", "used", "unwashed", "کثیف", "استفاده شده"],
    "open": ["open", "opened", "باز"], "closed": ["closed", "shut", "بسته"],
    "on": ["روشن", "turned on", "switched on", "powered on", "running"], "off": ["خاموش", "turned off", "switched off", "powered off"],
    "sliced": ["sliced", "برش خورده", "خرد شده"], "fresh": ["fresh", "تازه"],
    "bad": ["bad", "rotten", "stale", "spoiled", "wilted", "gone bad", "خراب", "گندیده", "بیات", "پلاسیده"],
    "dark": ["dark", "dim", "تاریک"], "bright": ["bright", "lit"],
    "empty": ["empty", "خالی"], "filled": ["full", "filled", "پر"], "broken": ["broken", "شکسته"],
}
_STATE_PH: dict = {}
for _k, _ph in _STATES.items():
    for _p in _ph:
        _STATE_PH.setdefault(tuple(LC.prep(_p)), _k)
_STATE_MAX = max(len(k) for k in _STATE_PH)
# an operation word read in a copular / participle context is a state, not a request ("has been left open", "the tap is on")
_OP_AS_STATE = {"open": "open", "close": "closed", "wash": "clean", "toggle:on": "on", "toggle:off": "off", "heat": "hot", "chill": "cold",
                "slice": "sliced", "break": "broken"}
_STATE_OP = {"cold": "chill", "hot": "heat", "clean": "wash", "sliced": "slice", "open": "open", "closed": "close",
             "on": "toggle:on", "off": "toggle:off"}                       # stated state -> the operation that reaches it (via_state)

# ------------------------------------------------------------------------------------------------------------- affordances
# operation / state / function word -> candidate English types, reached through that word. A hint, never a stated goal.
# role: dest = put the thing in/at it, tool = hold it, appliance = operate it
AFFORDANCE = {
    "chill": [("Fridge", "dest")], "cold": [("Fridge", "dest")], "cool": [("Fridge", "dest")], "stay_fresh": [("Fridge", "dest")],
    "heat": [("Microwave", "appliance"), ("StoveBurner", "appliance")], "hot": [("Microwave", "appliance"), ("StoveBurner", "appliance")],
    "cook": [("StoveBurner", "appliance"), ("Microwave", "appliance")],
    "slice": [("ButterKnife", "tool")], "flip": [("Spatula", "tool")], "scrub": [("DishSponge", "tool")],
    "washed": [("Sink", "dest"), ("SinkBasin", "dest")], "dirty": [("Sink", "dest")],
    "discard": [("GarbageCan", "dest")],
    "eat:soup": [("Spoon", "tool")], "eat": [("Fork", "tool"), ("Spoon", "tool")],
    "season": [("SaltShaker", "tool"), ("PepperShaker", "tool")],
    "dark": [("LightSwitch", "appliance")], "bright": [("LightSwitch", "appliance")], "bad": [("GarbageCan", "dest")],
    "wash:hands": [("SoapBottle", "tool"), ("Faucet", "appliance")],
}
_FN_EN = {"washed": "washed", "cleaned": "washed", "wash": "washed", "cool": "cool", "cold": "cool", "chilled": "cool", "cooled": "cool",
          "fresh": "stay_fresh", "kept": "stay_fresh", "stored": "stay_fresh"}
_FN_FA = {"شسته": "washed", "خنک": "cool", "سرد": "cool", "تازه": "stay_fresh", "تمیز": "washed"}
_WHERE_EN = {"where", "somewhere", "someplace"}
_WHERE_FA = {"جایی"}

_PRON_EN = {"it", "them", "this", "that", "these", "those", "one"}
_PRON_PL_EN = {"them", "these", "those"}
_PRON_FA = {N("آن"), N("آنها"), "اون", "اینها", "همین", "این"}
_PRON_PL_FA = {N("آنها"), "اینها"}
_DEM_EN = {"this", "that", "these", "those"}
_DEM_FA = {N("آن"), "این", "اون", "همین"}
_DET_EN = set(LC.default_lexicon().data["determiners"]["en"]) | {"some", "any", "another", "your", "my", "our", "his", "her", "their", "its"}
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "1": 1, "2": 2, "3": 3, "4": 4}
_ALL = {"all", "every", "each", "both", "همه", "هردو", "تمام", "هر"}
_DEST_EN = {"in", "into", "inside", "on", "onto", "to", "at", "under", "within"}
_SRC_EN = {"from", "off"}
_DEST_FA = {"در", "توی", "تو", "داخل", "درون", "روی", "بالای", "به", "زیر", "کنار"}
_SRC_FA = {"از", "بیرون"}
_AUX_EN = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "get", "gets", "got", "gone", "will", "can", "could",
           "would", "should", "might", "may", "do", "does", "did", "am", "stay", "left", "remains", "remain", "became", "become", "its"}
_AUX_FA = {"است", "هستند", "شده", "شد", "شود", "اند", "بود", "هست", "بوده", "مانده", "میماند", "میمانند", "میشوند", "میشود", "شدهاند", "ماند"}
_STOP_EN = _DET_EN | _AUX_EN | {"you", "i", "me", "we", "please", "of", "so", "very", "too", "up", "here", "in", "there", "that", "which", "who",
                                 "for", "with", "can", "just", "now", "later", "tomorrow", "today", "dinner", "lunch", "breakfast",
                                 "later", "s", "ll", "d", "somewhere", "something", "anything", "right", "away", "out", "down",
                                 "then", "also", "again", "still", "already", "ever", "bright", "dark", "much", "more"}
_STOP_FA = _AUX_FA | {"من", "برای", "برایم", "برام", "برامون", "بهم", "برایش", "را", "که", "با", "اینجا", "خیلی", "کم", "دیگر", "بعدا", "فردا", "شام", "ناهار", "صبحانه", "یک", "این",
                      N("آن"), "چیزی", "لطفا", "هم", "هنوز", "قبلا", "ها", "م", "ت", "ش", "یم", "ام", "تا", "عدد", "دانه", "تکه", "تایی", "دارم", "دارد", "دارند", "می"}


# --------------------------------------------------------------------------------------------------------------------------
def _is_fa(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in str(text))


def _span(toks, i, j) -> dict:
    return {"tok": [i, j], "text": " ".join(toks[i:j])}


def _match_extra(toks, i):
    for n in range(min(EXTRA_MAX, len(toks) - i), 0, -1):
        k = tuple(toks[i:i + n])
        if k in EXTRA_TYPES:
            return n, EXTRA_TYPES[k]
    return 0, None


def _match_type_here(toks, i, lex):
    n, t = LC._match_type(toks, i, lex)
    n2, e = _match_extra(toks, i)
    if n2 > n:
        return n2, e[0], e[1]
    if n:
        return n, t, ""
    return 0, None, ""


# FA enclitic object pronoun on a known stem: روشنش کن -> [§P روشن کن]; درش -> [در §P]
_FA_ENCLITIC_STEMS = {"روشن", "خاموش", "نگه", "مرتب", "باز", "بسته", "سرد", "گرم", "داغ", "خنک", "تمیز", "پاک", N("آماده"), "ببند", "بردار",
                      "بگذار", "بشور", "برش", "پیدا", "خالی", "پر", "درست", "بذار", "ببرش", "ببر", "بیاور", "بیار", "تکه", "وارونه", "بخور", "بشکن", "داخل", "درون"}


def _fa_split(toks: list) -> list:
    out, k = [], 0
    while k < len(toks):
        t = toks[k]
        if t == "از" and k + 1 < len(toks) and toks[k + 1] in ("شرش", "شر"):
            out.append(P if toks[k + 1] == "شرش" else "§X")                  # "get rid of it": drop the idiom's از شر, keep the pronoun
            k += 2
            continue
        done = False
        if t not in (H, S):
            for suf in ("ش", "شان", "شون"):
                stem = t[:-len(suf)]
                if t.endswith(suf) and len(stem) >= 2 and (stem in _FA_ENCLITIC_STEMS or stem == "در"):
                    if stem in ("در", "داخل", "درون"):
                        out.extend([stem, P])
                    else:
                        out.extend([P, stem])
                    done = True
                    break
        if not done:
            out.append(t)
        k += 1
    return [x for x in out if x != "§X"]


# ---------------------------------------------------------------------------------------------------------------------- events
def _scan(toks, lex, fa: bool) -> list:
    """Left-to-right mention scan -> events {k, i, j, ...}. Longest match first; no overlaps."""
    ev, i, n = [], 0, len(toks)
    while i < n:
        w = toks[i]
        if w == H:
            ev.append({"k": "SENT", "i": i, "j": i + 1}); i += 1; continue
        if w == S:
            ev.append({"k": "COMMA", "i": i, "j": i + 1}); i += 1; continue
        if w == P:
            ev.append({"k": "PRON", "i": i, "j": i + 1, "pl": False, "enclitic": True}); i += 1; continue
        m, t, note = _match_type_here(toks, i, lex)
        if m and t is None:                                   # A26: an excluded-type phrase = plain unknown words, no type
            ev.extend({"k": "WORD", "i": x, "j": x + 1, "word": toks[x]} for x in range(i, i + m)); i += m; continue
        if m:
            ev.append({"k": "TYPE", "i": i, "j": i + m, "type": t, "note": note}); i += m; continue
        if fa:
            took = _scan_fa_op(toks, i)
            if took:
                ev.append(took); i = took["j"]; continue
        else:
            took = _scan_en_op(toks, i, lex)
            if took:
                ev.extend(took["events"]); i = took["j"]; continue
        if not fa and w in ("on", "off") and i > 0 and toks[i - 1] in _AUX_EN and (i + 1 == n or toks[i + 1] in (H, S, "and", "but", "then")):
            ev.append({"k": "STATE", "i": i, "j": i + 1, "state": w}); i += 1; continue           # "the tap is on"
        # states (longest)
        hit = 0
        for ln in range(min(_STATE_MAX, n - i), 0, -1):
            k = tuple(toks[i:i + ln])
            if k in _STATE_PH:
                ev.append({"k": "STATE", "i": i, "j": i + ln, "state": _STATE_PH[k]}); i += ln; hit = ln; break
        if hit:
            continue
        if fa and w in _NEG_VERBS_FA:
            op = _NEG_VERB_OP_FA.get(w)
            if w == "نبر" and _has_dest_fa(toks, i):
                op = "put"
            ev.append({"k": "NEG", "i": i, "j": i + 1})
            if op:
                ev.append({"k": "OP", "i": i, "j": i + 1, "op": op, "neg": True, **({"amb": "bordan"} if w == "نبر" else {})})
            i += 1; continue
        if (not fa and w in _NEG_EN) or (fa and w in _NEG_FA):
            ev.append({"k": "NEG", "i": i, "j": i + 1}); i += 1; continue
        if ((not fa and w in _DEM_EN) or (fa and w in _DEM_FA)) and i + 1 < n and _match_type_here(toks, i + 1, lex)[0]:
            i += 1; continue                                                          # "this apple" / "این سیب": a determiner
        if not fa and w == "to" and i + 1 < n and _scan_en_op(toks, i + 1, lex):
            ev.append({"k": "INF", "i": i, "j": i + 1}); i += 1; continue             # infinitive marker "want to eat"
        if not fa and w == "out" and i + 1 < n and toks[i + 1] == "of":
            ev.append({"k": "SRC", "i": i, "j": i + 2, "word": "out of"}); i += 2; continue
        if (not fa and w in _PRON_EN) or (fa and w in _PRON_FA):
            ev.append({"k": "PRON", "i": i, "j": i + 1, "pl": (w in _PRON_PL_EN) if not fa else (w in _PRON_PL_FA), "word": w})
            i += 1; continue
        if fa and w == "را":
            ev.append({"k": "ACC", "i": i, "j": i + 1}); i += 1; continue
        if fa and w == "ها" and ev and ev[-1]["k"] == "PRON":
            ev[-1]["pl"] = True; ev[-1]["j"] = i + 1; i += 1; continue
        if fa and w == "یکی" and i + 1 < n and toks[i + 1] == "از":
            ev.append({"k": "INDEF", "i": i, "j": i + 2}); i += 2; continue            # partitive "one of the ..."
        if not fa and w == "one" and i + 1 < n and toks[i + 1] == "of":
            ev.append({"k": "INDEF", "i": i, "j": i + 2}); i += 2; continue
        if (not fa and w in _DEST_EN) or (fa and w in _DEST_FA):
            ev.append({"k": "DEST", "i": i, "j": i + 1, "word": w}); i += 1; continue
        if (not fa and w in _SRC_EN) or (fa and w in _SRC_FA):
            ev.append({"k": "SRC", "i": i, "j": i + 1, "word": w}); i += 1; continue
        if (not fa and w == "with") or (fa and w == "با"):
            ev.append({"k": "WITH", "i": i, "j": i + 1}); i += 1; continue
        if (not fa and w in _WHERE_EN) or (fa and w in _WHERE_FA):
            ev.append({"k": "WHERE", "i": i, "j": i + 1}); i += 1; continue
        if fa and w == "که":
            ev.append({"k": "REL", "i": i, "j": i + 1}); i += 1; continue
        if fa and w == "کم" and i + 1 < n and toks[i + 1] in ("دارد", "دارم", "دارند", "دارید"):
            ev.append({"k": "WANT", "i": i, "j": i + 2, "word": "کم دارد"}); i += 2; continue
        if (not fa and w in _WANT_EN) or (fa and w in _WANT_FA):
            if fa and w == "می":
                if i + 1 < n and toks[i + 1] in ("خواهم", "خواهد", "خواهیم"):
                    ev.append({"k": "WANT", "i": i, "j": i + 2, "word": "می خواهم"}); i += 2; continue
                ev.append({"k": "WORD", "i": i, "j": i + 1, "word": w}); i += 1; continue
            ev.append({"k": "WANT", "i": i, "j": i + 1, "word": w}); i += 1; continue
        if (not fa and (w in _EVERY_EN)) or (fa and w == "همه" and i + 1 < n and toks[i + 1] == "چیز"):
            ev.append({"k": "EVERY", "i": i, "j": i + (2 if fa else 1)}); i += (2 if fa else 1); continue
        if (not fa and w in _SUBJ_EN) or (fa and w in ("کسی", "هیچکس")):
            ev.append({"k": "SUBJ", "i": i, "j": i + 1, "word": w}); i += 1; continue
        if fa and w == "هر" and i + 1 < n and toks[i + 1] == "دو":
            ev.append({"k": "QUANT", "i": i, "j": i + 2, "all": True}); i += 2; continue                      # "هر دو" = both
        if w in _ALL:
            ev.append({"k": "QUANT", "i": i, "j": i + 1, "all": True}); i += 1; continue
        if w in _NUM and not (not fa and w == "one"):
            ev.append({"k": "QUANT", "i": i, "j": i + 1, "n": _NUM[w]}); i += 1; continue
        if (not fa and w in ("a", "an")) or (fa and w == "یک"):
            ev.append({"k": "INDEF", "i": i, "j": i + 1}); i += 1; continue
        if (not fa and w == "door"):
            ev.append({"k": "DOOR", "i": i, "j": i + 1}); i += 1; continue
        if not fa and w == "but":
            ev.append({"k": "CONJ", "i": i, "j": i + 1, "then": False, "but": True}); i += 1; continue
        if (not fa and w in ("and", "also", "plus")) or (fa and w == "و"):
            ev.append({"k": "CONJ", "i": i, "j": i + 1, "then": False}); i += 1; continue
        if (not fa and w in ("then", "afterwards", "next")) or (fa and w in ("سپس", "بعدش", "بعدا")) or (fa and w == "بعد" and not
                                                                                                      (i + 1 < n and toks[i + 1] == "از")):
            ev.append({"k": "CONJ", "i": i, "j": i + 1, "then": True}); i += 1; continue
        if w in ("first", "اول", "ابتدا", "نخست"):
            ev.append({"k": "FIRST", "i": i, "j": i + 1}); i += 1; continue
        if fa and w in ("قبل", "پس", "بعد") and i + 1 < n and toks[i + 1] == "از":
            ev.append({"k": "ORDER", "i": i, "j": i + 2, "word": {"قبل": "before", "پس": "after", "بعد": "after"}[w]}); i += 2; continue
        if (not fa and w in ("before", "after")):
            ev.append({"k": "ORDER", "i": i, "j": i + 1, "word": w}); i += 1; continue
        ev.append({"k": "WORD", "i": i, "j": i + 1, "word": w}); i += 1
    _ops_as_states(ev, toks, fa)
    return ev


def _gives_to_me_fa(toks, i) -> bool:
    """'X را به من بده' / 'بهم بده' / 'برایم ... بده' = hand it to the speaker = fetch. A bare 'بده' stays a light verb."""
    a = i
    while a > 0 and toks[a - 1] != H:
        a -= 1
    seg = toks[a:i]
    return "بهم" in seg or "برایم" in seg or "برام" in seg or any(seg[k] == "به" and seg[k + 1] in ("من", "ما") for k in range(len(seg) - 1))


_CLAUSE_EDGE_FA = {H, S, "و", "سپس", "بعدش", "بعدا", "اما", "ولی", "بعد"}


def _has_dest_fa(toks, i) -> bool:
    """A destination word in the SAME CLAUSE as toks[i] (clause = between hard / soft separators and 'و', 'سپس' ...). A destination that
    belongs to a later verb ('نان را ببر و در بشقاب بگذار') is not evidence that ببر is 'carry'."""
    a = i
    while a > 0 and toks[a - 1] not in _CLAUSE_EDGE_FA:
        a -= 1
    b = i
    while b + 1 < len(toks) and toks[b + 1] not in _CLAUSE_EDGE_FA:
        b += 1
    return any(t in _DEST_WORDS_FA for t in toks[a:b + 1])


def _scan_fa_op(toks, i):
    n, w = len(toks), toks[i]
    nxt = toks[i + 1] if i + 1 < n else None
    if w == "روشن" and i > 0 and toks[i - 1] in ("خیلی", "بسیار", "کمی", "زیادی"):
        return {"k": "STATE", "i": i, "j": i + 1, "state": "bright"}
    if w in _LIGHT_NOUN_FA and nxt is not None:
        if nxt in _LIGHT_VERBS_FA:
            return {"k": "OP", "i": i, "j": i + 2, "op": _LIGHT_NOUN_FA[w], "neg": False}
        if nxt in _NEG_LIGHT_FA:
            return {"k": "OP", "i": i, "j": i + 2, "op": _LIGHT_NOUN_FA[w], "neg": True}
    if w == "دور" and nxt in ("بینداز", "بیندازید", "بریز", "بریزید", "بیفکن", "بنداز"):
        return {"k": "OP", "i": i, "j": i + 2, "op": "discard", "neg": False}
    if w == "خلاص" and nxt in ("شو", "شود", "کن"):
        return {"k": "OP", "i": i, "j": i + 2, "op": "discard", "neg": False}
    if w == "برش" and nxt in ("بزن", "بده", "بزنید", "بدهید", "بدم", "بزنم", "بدهم"):
        return {"k": "OP", "i": i, "j": i + 2, "op": "slice", "neg": False}
    if w == "قرار" and nxt in ("بده", "بدهید", "بدم", "بدین"):
        return {"k": "OP", "i": i, "j": i + 2, "op": "put", "neg": False}
    if w in _NEG_VERBS_FA:
        return None
    if w in _BORDAN_FA:
        carry = _has_dest_fa(toks, i)
        return {"k": "OP", "i": i, "j": i + 1, "op": "put" if carry else "slice", "neg": False, "amb": "bordan"}
    if w in ("بده", "بدهید", "بدین", "بدی", "بدم") and _gives_to_me_fa(toks, i):
        return {"k": "OP", "i": i, "j": i + 1, "op": "pick", "neg": False}
    if w == "راه" and nxt in ("بینداز", "بیندازید", "بنداز", "بیندازم", "بیفتد"):
        return {"k": "OP", "i": i, "j": i + 2, "op": "toggle:on", "neg": False}
    if w == "پشت" and nxt == "و" and i + 3 < n and toks[i + 2] == "رو" and toks[i + 3] in _LIGHT_VERBS_FA | _NEG_LIGHT_FA:
        return {"k": "OP", "i": i, "j": i + 4, "op": "flip", "neg": toks[i + 3] in _NEG_LIGHT_FA}
    for op, forms in _VERBS_FA.items():
        if w in forms:
            return {"k": "OP", "i": i, "j": i + 1, "op": op, "neg": False}
    if w == "خوردن" or w == "گرم":
        pass
    return None


def _scan_en_op(toks, i, lex):
    n, w = len(toks), toks[i]
    # separable phrasal verb: turn it on / pick it up
    if w in _PHRASAL:
        for look in range(1, 5):
            if i + look >= n:
                break
            if toks[i + look] in _PHRASAL[w]:
                between = toks[i + 1:i + look]
                if all(_np_only(between, lex)) if between else True:
                    if between and any(x in (H, S) for x in between):
                        break
                    # a phrasal particle directly after the verb is handled by the multi-word phrase below
                    if look > 1:
                        evs = [{"k": "OP", "i": i, "j": i + 1, "op": _PHRASAL[w][toks[i + look]], "neg": False, "particle": i + look}]
                        # keep the between-words as ordinary events, mark the particle as consumed
                        inner = _scan(between, lex, False) if between else []
                        for e in inner:
                            e2 = dict(e)
                            e2["i"] += i + 1
                            e2["j"] += i + 1
                            evs.append(e2)
                        evs.append({"k": "SKIP", "i": i + look, "j": i + look + 1})
                        return {"events": evs, "j": i + look + 1}
                break
    if w in ("get", "give", "hand", "fetch") and _fetch_np_follows(toks, i, lex):
        return {"events": [{"k": "OP", "i": i, "j": i + 1, "op": "pick", "neg": False}], "j": i + 1}
    best = None
    for op, phs in _OPS_EN.items():
        for ph in phs:
            p = tuple(LC.prep(ph))
            if tuple(toks[i:i + len(p)]) == p and (best is None or len(p) > best[0]):
                best = (len(p), op)
    if best is None and w.endswith("ing") and len(w) > 5:                  # gerund: "after washing the pan"
        stem = w[:-3]
        for cand in (stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else None):
            if cand and cand in _SINGLE_EN:
                return {"events": [{"k": "OP", "i": i, "j": i + 1, "op": _SINGLE_EN[cand], "neg": False, "gerund": True}], "j": i + 1}
    if best and best[1] == "_ensure":
        return {"events": [{"k": "WANT", "i": i, "j": i + best[0], "word": "make sure"}], "j": i + best[0]}
    if best:
        return {"events": [{"k": "OP", "i": i, "j": i + best[0], "op": best[1], "neg": False}], "j": i + best[0]}
    return None


_RECIPIENT_EN = {"me", "us", "him", "her", "them"}


def _fetch_np_follows(toks, i, lex) -> bool:
    """'get / give / hand (me) the X': a fetch ONLY when a kitchen type follows (determiners and a recipient are skipped); 'get cold', 'get rid of'
    and 'gets washed' are not fetches."""
    k = i + 1
    while k < len(toks) and (toks[k] in _DET_EN or toks[k] in _RECIPIENT_EN or toks[k] in ("a", "an", "the", "some", "one")):
        k += 1
    return k < len(toks) and LC._match_type(toks, k, lex)[0] > 0 or (k < len(toks) and _match_extra(toks, k)[0] > 0)


def _np_only(words, lex):
    k = 0
    while k < len(words):
        x = words[k]
        if x in _DET_EN or x in _PRON_EN or x in ("a", "an", "the"):
            k += 1
            yield True
            continue
        m, t, _ = _match_type_here(words, k, lex)
        if m and t is None:                     # A26: an excluded-type phrase is unknown words, like a bare "knife"
            k += m
            yield False
            continue
        if m:
            k += m
            yield True
            continue
        yield False
        k += 1


def _ops_as_states(ev, toks, fa):
    """An operation word in a copular / participle context is a state: 'has been left open', 'the tap is on'."""
    for idx, e in enumerate(ev):
        if e["k"] != "OP" or fa:
            continue
        prev = toks[e["i"] - 1] if e["i"] > 0 else None
        if prev in _AUX_EN and e["op"] in _OP_AS_STATE:
            e["k"], e["state"] = "STATE", _OP_AS_STATE[e["op"]]


# --------------------------------------------------------------------------------------------------------------- the parser
def parse_goal(text: str, lang: Optional[str] = None, holding: Optional[str] = None, lexicon=None) -> dict:
    """text -> goal schema (see the module docstring). `holding` is the type the agent is told it holds (optional context)."""
    lex = lexicon or LC.default_lexicon()
    fa = (lang == "fa") if lang in ("en", "fa") else _is_fa(text)
    lang = "fa" if fa else "en"
    toks0 = LC.prep(text)
    toks = _fa_split(toks0) if fa else toks0
    ev = _scan(toks, lex, fa)
    skip = {e["i"] for e in ev if e["k"] == "SKIP"}
    ev = [e for e in ev if e["k"] != "SKIP" and not (e["k"] == "WORD" and e["i"] in skip)]
    out = {"schema": SCHEMA, "version": VERSION, "lang": lang, "text": text, "tokens": toks,
           "context": {"holding": None}, "entities": [], "operations": [], "final_predicates": [], "temporal": [], "needs": [],
           "facts": [], "desires": [], "affordances": [], "quantities": [], "negations": [], "unresolved": [], "ambiguities": [], "wanted_states": [],
           "status": "unsupported", "reason": ""}
    st = {"toks": toks, "fa": fa, "lex": lex, "out": out, "focus": [], "gid": 0}
    if holding and isinstance(holding, str):
        out["context"]["holding"] = {"type": holding, "span": None, "source": "caller"}
        st["focus"].append({"type": holding, "span": None, "role": "held", "group": -1})
    sents, cur = [], []
    for e in ev:
        if e["k"] == "SENT":
            if cur:
                sents.append(cur)
            cur = []
        else:
            cur.append(e)
    if cur:
        sents.append(cur)
    st["qmark"] = str(text).strip().endswith(("?", "؟"))
    for k, sent in enumerate(sents):
        _sentence(st, sent, last=(k == len(sents) - 1))
    _finish(st)
    return out


def _compatible(op: str, tp: str) -> bool:
    if tp == "*":
        return op in ("tidy", "open", "close", "discard")
    if op in ("open", "close"):
        return tp in OPENABLE
    if op in ("toggle:on", "toggle:off"):
        return tp in TOGGLEABLE
    if op in ("put", "pick", "drop", "empty", "fill", "break", "slice", "heat", "cook", "wash", "chill", "eat", "drink", "flip", "keep",
              "scrub", "discard", "serve", "find"):
        return tp not in FIXTURES
    return True


def _resolve_pron(st, op, plural, group):
    """Latest earlier mention compatible with the operation; objects before destinations; plural -> the latest coordinated group."""
    pool = [f for f in st["focus"] if f.get("type") and f.get("group") != group and _compatible(op, f["type"])]
    for roles in (("obj", "held", "subject", "universal"), None):
        cand = [f for f in pool if roles is None or f.get("role") in roles]
        if cand:
            f = cand[-1]
            if plural and f.get("set") is not None:
                return [x for x in cand if x.get("set") == f["set"]]
            return [f]
    return []


def _resolve_container(st, group):
    for f in reversed(st["focus"]):
        if f.get("type") in OPENABLE | {"Sink", "SinkBasin", "CounterTop", "GarbageCan", "Shelf", "StoveBurner", "Plate", "Bowl", "Pot", "Pan"}                 and f.get("group") != group:
            return f
    return None


_EXCL_WORDS = None


def _excluded_words() -> frozenset:
    """A26 (K-B2b): the surface words of the excluded types (concepts.json `excluded_aliases`, outside `types`): they never resolve to a
    type, but in a need clause ("I need a knife to ...") they still mark the wished item, which then stays unknown ("?")."""
    global _EXCL_WORDS
    if _EXCL_WORDS is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step5", "lexicon", "concepts.json")
        try:
            with open(p, encoding="utf-8") as f:
                ex = json.load(f).get("excluded_aliases") or {}
        except OSError:
            ex = {}
        _EXCL_WORDS = frozenset(w for e in ex.values() for k in ("en", "en_plural", "fa") for w in e.get(k, []) if " " not in w)
    return _EXCL_WORDS


def _need_item(st, cl, group, wants, ops) -> bool:
    """'I need a knife to cut the bread' / FA 'برای بریدن نان یک چاقو لازم دارم': the wished item is the thing named next to the wish; the
    operation is its purpose, not a step to execute."""
    toks, fa, out = st["toks"], st["fa"], st["out"]
    types = [e for e in cl if e["k"] == "TYPE"]
    main = ops[-1] if fa else ops[0]
    needed = []
    stop = _STOP_FA if fa else _STOP_EN

    def _noun(e):                       # A26 (K-B2b): a word of an EXCLUDED type ("knife" / «چاقو») can be the wished item, as "?"
        return e["k"] == "WORD" and toks[e["i"]] in _excluded_words()
    if not fa:
        inf = next((e for e in cl if e["k"] == "INF"), None)
        if inf is not None:
            needed = [e for e in types if wants[0]["i"] < e["i"] < inf["i"]] or                 [e for e in cl if _noun(e) and wants[0]["i"] < e["i"] < inf["i"]][:1]
    else:
        if len(types) >= 2 and toks[main["i"]].endswith("ن") and main["i"] < wants[0]["i"]:
            needed = [types[-1]]
        elif toks[main["i"]].endswith("ن") and main["i"] < wants[0]["i"]:
            needed = [e for e in cl if _noun(e) and max([main["i"]] + [t["i"] for t in types]) < e["i"] < wants[0]["i"]][-1:]
    if not needed:
        return False
    for e in needed:
        if e["k"] == "WORD":
            d = {"type": None, "surface": toks[e["i"]], "span": _span(toks, e["i"], e["j"]), "role": "needed", "group": group,
                 "unknown_noun": True}
        else:
            d = _mk_entity(toks, e, group, "needed", {})
        out["entities"].append(d)
        st["focus"].append(d)
    purpose_objs = []
    for e in types:
        if e in needed:
            continue
        d = _mk_entity(toks, e, group, "obj", {})
        out["entities"].append(d)
        purpose_objs.append(d)
    out["needs"].append({"kind": "item", "type": needed[0].get("type"), "span": _span(toks, needed[0]["i"], needed[0]["j"]),
                         **({"surface": toks[needed[0]["i"]]} if needed[0]["k"] == "WORD" else {}),
                         "cue": _span(toks, wants[0]["i"], wants[0]["j"]), "op": main["op"], "op_span": _span(toks, main["i"], main["j"]),
                         "for": purpose_objs, "purpose": True})
    return True


_Q_EN = {"is", "are", "was", "were", "does", "do", "did", "has", "have", "what", "who", "which", "how", "why", "when", "where", "whether"}
_Q_FA = {N("آیا"), "چه", "کجا", "کجاست", "چرا", "کی", "چطور", "چگونه", "کدام", "چند", "چیست"}


def _is_question(st, sent, last: bool) -> bool:
    toks, fa = st["toks"], st["fa"]
    first = next((toks[e["i"]] for e in sent if e["k"] not in ("COMMA", "CONJ", "SENT")), None)
    if first is None:
        return False
    if fa:
        return (first == N("آیا")) or (last and bool(st.get("qmark")) and any(toks[e["i"]] in _Q_FA for e in sent))
    return first in _Q_EN and last and bool(st.get("qmark"))


def _sentence(st, sent, last=False):
    toks, fa, out = st["toks"], st["fa"], st["out"]
    if _is_question(st, sent, last):
        out["facts"].append({"kind": "question", "span": _span(toks, sent[0]["i"], sent[-1]["j"])})
        return
    words = [toks[e["i"]] for e in sent]
    has_op = any(e["k"] == "OP" for e in sent)
    # holding statement: "You are holding the bowl" / "کاسه در دست توست"
    held_marker = ("holding" in words) if not fa else ("دست" in words and not has_op)
    if held_marker:
        t = next((e for e in sent if e["k"] == "TYPE"), None)
        if t:
            d = {"type": t["type"], "span": _span(toks, t["i"], t["j"]), "role": "held", "group": st["gid"]}
            st["gid"] += 1
            out["entities"].append(d)
            st["focus"].append(d)
            out["context"]["holding"] = {"type": t["type"], "span": d["span"], "source": "instruction"}
            return
    for cl in _clauses(sent, fa):
        st["gid"] += 1
        _clause(st, cl, st["gid"])


def _clauses(sent, fa):
    """Segment at separators; a segment without an operation joins the previous one (EN, verb-first) or the next one (FA, verb-last).
    A segment with several operations and no want / relative marker splits at each operation."""
    segs, cur = [], []
    for e in sent:
        if not fa and e["k"] == "ORDER" and cur and any(x["k"] not in ("COMMA", "CONJ") for x in cur):
            segs.append(cur)
            cur = []
        cur.append(e)
        if e["k"] in ("COMMA", "CONJ") and e is not sent[-1]:
            segs.append(cur[:-1])
            cur = [e]
    segs.append(cur)
    segs = [s for s in segs if s]
    merged = []
    if fa:
        carry = []
        for s in segs:
            if any(e["k"] == "OP" for e in s):
                merged.append(carry + s)
                carry = []
            else:
                carry = carry + s
        if carry:
            if merged:
                merged[-1] = merged[-1] + carry
            else:
                merged.append(carry)
    else:
        for s in segs:
            lead = next((e for e in s if e["k"] not in ("COMMA", "CONJ")), None)
            if any(e["k"] == "OP" for e in s) or not merged or (lead is not None and lead["k"] == "ORDER"):
                merged.append(list(s))
            else:
                merged[-1].extend(s)
    out = []
    for s in merged:
        out.extend(_split_multi(s, fa))
    return out


def _split_multi(seg, fa):
    ops = [k for k, e in enumerate(seg) if e["k"] == "OP"]
    if len(ops) < 2 or any(e["k"] in ("WANT", "WHERE", "REL") for e in seg):
        return [seg]
    # J-P1-parser: "bring me something to cut with" / "چیزی برای بریدن بیاور": a purpose phrase, ONE need, not two operations
    if any(e["k"] == "WORD" and e.get("word") in (("چیزی", "ابزاری", "وسیله") if fa else ("something", "anything", "tool", "tools")) for e in seg):
        return [seg]
    if fa:
        cuts = [k + 1 for k in ops[:-1] if not (seg[k]["k"] == "OP" and k > 0 and seg[k - 1]["k"] == "SRC")]      # "قبل از بستن کشو"
    else:
        cuts = [k for k in ops[1:]]
    out, start = [], 0
    for c in cuts:
        out.append(seg[start:c])
        start = c
    out.append(seg[start:])
    return [s for s in out if s]


def _mk_entity(toks, e, group, role, pending):
    d = {"type": e["type"], "span": _span(toks, e["i"], e["j"]), "role": role, "group": group}
    if e.get("note"):
        d["note"] = e["note"]
    if pending.get("indef"):
        d["indefinite"] = True
        pending["indef"] = False
    if pending.get("quant") is not None:
        d["quantity"] = pending["quant"]
        pending["quant"] = None
    if pending.get("state"):
        d["state"], d["state_span"] = pending["state"]
        pending["state"] = None
    return d


def _clause(st, cl, group):
    toks, fa, out, lex = st["toks"], st["fa"], st["out"], st["lex"]
    wants = [e for e in cl if e["k"] == "WANT"]
    where = next((e for e in cl if e["k"] == "WHERE"), None)
    rel = next((e for e in cl if e["k"] == "REL"), None)
    rel_i = (where["i"] if where is not None else None) if not fa else (rel["i"] if rel is not None else None)
    ops_all = [e for e in cl if e["k"] == "OP"]
    ops = [e for e in ops_all if rel_i is None or (e["i"] < rel_i)] if rel_i is not None else ops_all
    # relative clause "where they get washed" / "که شسته می‌شوند": its words name a function of the destination
    fn = _function_in(cl, toks, fa, where, rel) if (where is not None or rel is not None) else None
    every = next((e for e in cl if e["k"] == "EVERY"), None)
    has_subj = any(e["k"] == "SUBJ" for e in cl)
    # ---- needs --------------------------------------------------------------------------------------------------------
    thing = next((k for k, e in enumerate(cl) if e["k"] == "WORD" and toks[e["i"]] in (("something", "anything", "tool", "tools") if not fa else ("چیزی", "ابزاری", "وسیله"))), None)
    if thing is not None and (wants or ops_all) and where is None:
        _need(st, cl, thing, group)
        return
    if wants and not ops and [e for e in cl if e["k"] == "TYPE"] and not [e for e in cl if e["k"] == "STATE"]:
        # "my food needs salt": the wish names an item
        t = [e for e in cl if e["k"] == "TYPE"]
        for e in t:
            d = _mk_entity(toks, e, group, "needed", {})
            out["entities"].append(d)
            st["focus"].append(d)
            out["needs"].append({"kind": "item", "type": e["type"], "span": d["span"], "cue": _span(toks, wants[0]["i"], wants[0]["j"])})
            for tp, role in AFFORDANCE.get("season", []) if e["type"] in ("SaltShaker", "PepperShaker") else []:
                pass
        return
    if wants and ops and _need_item(st, cl, group, wants, ops):
        return
    core0 = next((e for e in cl if e["k"] not in ("COMMA", "CONJ", "INF")), None)
    if not ops and core0 is not None and core0["k"] == "ORDER":                  # "after you use it": a condition on an unmodelled event
        out["facts"].append({"kind": "condition", "rel": core0["word"], "span": _span(toks, cl[0]["i"], cl[-1]["j"])})
        return
    if not ops or has_subj:
        _statement(st, cl, group, wants, where, fn, ops, every)
        return
    # ---- one operation clause -----------------------------------------------------------------------------------------
    main = ops[-1] if fa else ops[0]
    op_key = main["op"]
    neg = bool(main.get("neg")) or any(e["k"] == "NEG" for e in cl)
    rec = {"op": op_key, "neg": neg, "span": _span(toks, main["i"], main["j"]), "mood": "desire" if wants else "imperative",
           "obj": [], "src": None, "dest": None, "tool": None}
    if op_key == "pick":                                  # "give me X" / "به من بده": the recipient is kept in the schema (not in the retrieval key)
        cw = [toks[e["i"]] for e in cl]
        if (not fa and any(x in ("me", "us") for x in cw)) or (fa and (any(x in ("بهم", "برایم", "برام") for x in cw) or any(
                cw[q] == "به" and cw[q + 1] in ("من", "ما") for q in range(len(cw) - 1)))):
            rec["recipient"] = "speaker"
    if main.get("amb"):
        out["ambiguities"].append({"word": toks[main["i"]], "kind": main["amb"], "chosen": op_key, "other": "put" if op_key == "slice" else "slice",
                                   "policy": "destination word in the sentence = carry (put), otherwise cut (slice)", "span": rec["span"]})
    prev_prep, door_pending, np_words = None, None, []
    pending: dict = {}
    last_ent = None
    out_set = group * 10
    for k, e in enumerate(cl):
        kd = e["k"]
        if rel_i is not None and e["i"] >= rel_i:
            break
        if kd != "WORD" and np_words:
            _flush_np(st, rec, np_words, prev_prep, fa, toks)
            np_words = []
        if kd == "OP":
            continue
        if kd in ("DEST", "SRC", "WITH"):
            nxt = cl[k + 1] if k + 1 < len(cl) else None
            if fa and kd == "DEST" and e.get("word") == "در" and nxt is not None and (
                    nxt["k"] == "PRON" and nxt.get("enclitic") or (op_key in ("open", "close") and nxt["k"] == "TYPE")):
                door_pending = e                                     # "در مایکروویو را باز کن", "درش را ببند": a door, not a destination
                if nxt["k"] == "PRON":
                    cl[k + 1] = dict(nxt, k="SKIPPRON")
                prev_prep = None
                continue
            prev_prep = {"DEST": "dest", "SRC": "src", "WITH": "with"}[kd]
            continue
        if kd == "DOOR":
            door_pending = e
            continue
        if kd == "INDEF":
            pending["indef"] = True
            continue
        if kd == "QUANT":
            pending["quant"] = "all" if e.get("all") else e["n"]
            continue
        if kd == "EVERY":
            d = {"type": "*", "universal": True, "span": _span(toks, e["i"], e["j"]), "role": "universal", "group": group}
            rec["obj"].append(d)
            out["entities"].append(d)
            st["focus"].append(d)
            continue
        if kd == "STATE":
            # bound to the nearest noun: forward ("a hot potato") only if a type follows before the verb, else backward
            if _binds_forward(cl, k, toks, fa):
                pending["state"] = (e["state"], _span(toks, e["i"], e["j"]))
            elif last_ent is not None:
                last_ent.setdefault("state", e["state"])
                last_ent["state_span"] = _span(toks, e["i"], e["j"])
            elif rec["obj"]:
                rec["obj"][-1].setdefault("state", e["state"])
                rec["obj"][-1]["state_span"] = _span(toks, e["i"], e["j"])
            else:
                pending["state"] = (e["state"], _span(toks, e["i"], e["j"]))
            continue
        if kd == "TYPE":
            role = prev_prep or "obj"
            d = _mk_entity(toks, e, group, role, pending)
            if role == "obj":
                d["set"] = out_set
                rec["obj"].append(d)
            elif role == "dest":
                rec["dest"] = d
            elif role == "src":
                rec["src"] = d
            else:
                rec["tool"] = d
            out["entities"].append(d)
            st["focus"].append(d)
            last_ent = d
            prev_prep = None
            continue
        if kd == "PRON":
            if prev_prep == "with":
                rec["tool"] = {"type": None, "via": _span(toks, e["i"], e["j"]), "unresolved": "pronoun_tool"}
                prev_prep = None
                continue
            if door_pending is not None and e.get("enclitic"):
                continue
            if prev_prep in ("dest", "src"):
                # "in it", "داخلش": the container just mentioned
                tgt = _resolve_container(st, group)
                if tgt is not None:
                    d = {"type": tgt["type"], "span": tgt.get("span"), "via": _span(toks, e["i"], e["j"]), "inherited": True,
                         "role": prev_prep, "group": group}
                    if prev_prep == "dest":
                        rec["dest"] = d
                    else:
                        rec["src"] = d
                prev_prep = None
                continue
            tgt = _resolve_pron(st, op_key, e.get("pl", False), group)
            if tgt:
                for f in tgt:
                    d = {"type": f["type"], "span": f.get("span"), "via": _span(toks, e["i"], e["j"]), "inherited": True, "role": "obj",
                         "group": group}
                    rec["obj"].append(d)
            else:
                rec["obj"].append({"type": None, "via": _span(toks, e["i"], e["j"]), "unresolved": "pronoun", "role": "obj", "group": group})
            continue
        if kd == "WORD":
            np_words.append(e)
            continue
    if np_words:
        _flush_np(st, rec, np_words, prev_prep, fa, toks)
    if prev_prep == "dest" and rec["dest"] is None and not fn:              # a bare "inside": the container just mentioned
        tgt = _resolve_container(st, group)
        if tgt is not None:
            last_dest = next((e for e in reversed(cl) if e["k"] == "DEST"), None)
            rec["dest"] = {"type": tgt["type"], "span": tgt.get("span"), "via": _span(toks, last_dest["i"], last_dest["j"]), "inherited": True,
                           "role": "dest", "group": group}
    # a door: "close the door" -> the latest earlier openable thing
    if door_pending is not None and op_key in ("open", "close") and not [x for x in rec["obj"] if x.get("type")]:
        tgt = _resolve_pron(st, op_key, False, group)
        if tgt:
            f = tgt[0]
            rec["obj"].append({"type": f["type"], "span": f.get("span"), "via": _span(toks, door_pending["i"], door_pending["j"]),
                               "inherited": True, "door": True, "role": "obj", "group": group})
    # an elided object (Persian drops it; English "put in the sink" after a state sentence): the previous clause's object
    if not [x for x in rec["obj"] if x.get("type") or x.get("surface")] and op_key in (
            "put", "pick", "heat", "cook", "wash", "slice", "fill", "empty", "drop", "break", "chill", "discard", "keep", "flip", "scrub",
            "tidy", "serve", "eat", "drink") and not rec["obj"]:
        tgt = _resolve_pron(st, op_key, False, group)
        if tgt and fa:
            f = tgt[0]
            rec["obj"].append({"type": f["type"], "span": f.get("span"), "via": rec["span"], "inherited": True, "elided": True, "role": "obj",
                               "group": group})
    # destination described by a function
    if fn:
        rec["dest"] = {"type": None, "fn": fn[0], "span": fn[1], "kind": "function"}
        for tp, role in AFFORDANCE.get(fn[0], []):
            out["affordances"].append({"via": fn[1], "word": fn[0], "type": tp, "role": role, "rule": "affordance:" + fn[0]})
    if pending.get("state") and rec["obj"]:
        rec["obj"][-1].setdefault("state", pending["state"][0])
        rec["obj"][-1]["state_span"] = pending["state"][1]
    if rec["op"] == "pick" and rec["dest"] is not None and rec["dest"].get("type"):          # "take the plate to the table"
        rec["op"], rec["op_note"] = "put", "carry"
    # open / close aimed at a tap is a switch
    if op_key in ("open", "close") and any(x.get("type") in OPEN_AS_TOGGLE for x in rec["obj"]):
        rec["op"] = "toggle:on" if op_key == "open" else "toggle:off"
        rec["op_note"] = "tap_has_no_door"
    # explicit order cues
    then_cue = next((e for e in cl if e["k"] == "CONJ" and e.get("then")), None)
    if then_cue is not None:
        rec["after_cue"] = _span(toks, then_cue["i"], then_cue["j"])
    od = next((e for e in cl if e["k"] == "ORDER"), None)
    if od is not None:
        core = [x for x in cl if x["k"] not in ("COMMA", "CONJ", "INF")]
        rec["order"] = {"word": od["word"], "clause_initial": bool(core) and core[0] is od, "span": _span(toks, od["i"], od["j"])}
    first = next((e for e in cl if e["k"] == "FIRST"), None)
    if first is not None:
        rec["order_cue"] = _span(toks, first["i"], first["j"])
    if toks[main["i"]] == "leave" and not fa and rec["dest"] is None and not fn:         # "leave it": keep as it is, no action
        out["facts"].append({"kind": "leave_as_is", "span": rec["span"], "obj": [x.get("type") for x in rec["obj"]]})
        return
    if neg:
        out["negations"].append({"op": rec["op"], "span": rec["span"]})
    if wants:
        out["desires"].append({"span": _span(toks, wants[0]["i"], wants[0]["j"]), "op": rec["op"]})
    out["operations"].append(rec)
    # affordances reached through the operation word, and through a stated state
    key = rec["op"]
    if key == "eat" and any((x.get("surface") or "").startswith(("soup", "سوپ")) for x in rec["obj"]):
        key = "eat:soup"
    if key == "wash" and any((x.get("surface") or "").startswith(("hand", "دست")) for x in rec["obj"]):
        key = "wash:hands"
    if key in AFFORDANCE and key not in ("eat", "eat:soup"):
        for tp, role in AFFORDANCE[key]:
            out["affordances"].append({"via": rec["span"], "word": key, "type": tp, "role": role, "rule": "affordance:" + key})
    for x in rec["obj"]:
        s = x.get("state")
        if s and s in AFFORDANCE and x.get("state_span"):
            for tp, role in AFFORDANCE[s]:
                out["affordances"].append({"via": x["state_span"], "word": s, "type": tp, "role": role, "rule": "affordance:" + s})


def _binds_forward(cl, k, toks, fa) -> bool:
    """An adjective directly before its noun ('a hot potato') binds forward; anywhere else it describes the noun before it."""
    if fa:
        return False
    for x in cl[k + 1:]:
        if x["k"] in ("INDEF", "QUANT") or (x["k"] == "WORD" and toks[x["i"]] in _STOP_EN):
            continue
        return x["k"] == "TYPE"
    return False


def _flush_np(st, rec, words, prev_prep, fa, toks):
    """Unknown content words in an object slot are kept (surface only, no type) so nothing is silently dropped."""
    content = [e for e in words if toks[e["i"]] not in (_STOP_FA if fa else _STOP_EN) and len(toks[e["i"]]) > 1]
    if not content or prev_prep not in (None,):
        return
    surf = " ".join(toks[e["i"]] for e in content)
    if rec["obj"] and all(x.get("type") for x in rec["obj"]) and not fa:
        return
    rec["obj"].append({"type": None, "surface": surf, "span": _span(toks, content[0]["i"], content[-1]["j"]), "role": "obj", "unknown_noun": True})


def _function_in(cl, toks, fa, where, rel):
    """('washed' | 'cool' | 'stay_fresh', span) from the words of a relative clause ('where they get washed')."""
    start = (where or rel)["i"]
    # Persian puts the verb phrase ('شسته می‌شوند') after که; جایی comes before the main verb
    for e in cl:
        if e["i"] < start and not (fa and where is not None):
            continue
        if fa and rel is not None and e["i"] < rel["i"]:
            continue
        w = toks[e["i"]]
        if e["k"] == "STATE" and e["state"] in ("clean", "cold", "fresh"):
            f = {"clean": "washed", "cold": "cool", "fresh": "stay_fresh"}[e["state"]]
            return f, _span(toks, min(start, e["i"]), e["j"])
        if e["k"] == "OP" and not fa and e["op"] in ("wash", "chill"):
            return ("washed" if e["op"] == "wash" else "cool"), _span(toks, start, e["j"])
        if not fa and w in _FN_EN and e["k"] in ("WORD", "OP", "STATE"):
            return _FN_EN[w], _span(toks, start, e["j"])
        if fa and w in _FN_FA:
            return _FN_FA[w], _span(toks, start, e["j"])
    return None


def _need(st, cl, thing, group):
    """'I need something to cut (the X) with' / 'something I can eat soup with' / FA 'چیزی برای بریدن لازم دارم'."""
    toks, fa, out = st["toks"], st["fa"], st["out"]
    ops = [e for e in cl if e["k"] == "OP"]
    purpose = [e for e in ops if e["op"] != "pick"]                         # "bring something to slice with": the purpose is slicing, not bringing
    op_ev = (purpose[-1] if (fa and purpose) else (purpose[0] if purpose else None)) or ((ops[-1] if fa else ops[0]) if ops else None)
    op = op_ev["op"] if op_ev else None
    if op is None:
        for e in cl:
            w = toks[e["i"]]
            if fa and w in _VNOUN_FA:
                op, op_ev = _VNOUN_FA[w], e
                break
    if op is None:
        out["needs"].append({"kind": "tool", "op": None, "span": _span(toks, cl[0]["i"], cl[-1]["j"]), "unresolved": "operation"})
        return
    objs = []
    for e in cl:
        if e["k"] == "TYPE":
            d = _mk_entity(toks, e, group, "obj", {})
            objs.append(d)
            out["entities"].append(d)
    stop = _STOP_FA if fa else _STOP_EN
    skip_words = {"something", "anything", "need", "needs", "tool", "tools", "to", "with", "چیزی", "ابزاری", "لازم", "نیاز", "برای", "با", "که",
                  "می", "خواهم", N("آن"), "ها", "م"} | set(_VNOUN_FA)
    for e in cl:
        if e["k"] == "WORD":
            w = toks[e["i"]]
            if w not in stop and w not in skip_words and len(w) > 1 and (op_ev is None or e["i"] != op_ev["i"]):
                objs.append({"type": None, "surface": w, "span": _span(toks, e["i"], e["j"]), "unknown_noun": True, "role": "obj"})
    kind = "tool"
    cw = [toks[e["i"]] for e in cl]
    marked_tool = ("with" in cw) if not fa else ("با" in cw)
    if fa and "را" in cw[thing:] and "با" not in cw:                  # "چیزی بیاور که آن را ببرم": the thing is what gets cut, not the cutter
        kind = "item"
    elif not marked_tool:
        out["ambiguities"].append({"word": cw[thing], "kind": "need_role", "chosen": "tool", "other": "item",
                                   "policy": "a purpose phrase without 'with' / 'با آن' is read as a tool need", "span": _span(toks, cl[thing]["i"], cl[-1]["j"])})
    need = {"kind": kind, "op": op, "for": objs, "span": _span(toks, cl[thing]["i"], cl[-1]["j"]),
            "op_span": _span(toks, op_ev["i"], op_ev["j"]) if op_ev else None}
    out["needs"].append(need)
    key = op
    if op == "eat" and any((o.get("surface") or "").startswith(("soup", "سوپ")) for o in objs):
        key = "eat:soup"
    for tp, role in ([] if kind == "item" else AFFORDANCE.get(key, [])):
        out["affordances"].append({"via": need["op_span"] or need["span"], "word": key, "type": tp, "role": role, "rule": "affordance:" + key})


def _statement(st, cl, group, wants, where, fn, ops, every):
    """A clause with no usable operation: a fact ('the fork has been used'), a wish ('I like my bread cold'), a declared target
    ('a hot potato on a plate') or something indirect ('it is too dark'). Nothing here becomes a goal by itself."""
    toks, fa, out = st["toks"], st["fa"], st["out"]
    ents, pending, dest_next = [], {}, False
    last_subj = None
    sentence_span = _span(toks, cl[0]["i"], cl[-1]["j"])
    declarative = []
    for e in cl:
        kd = e["k"]
        if kd == "INDEF":
            pending["indef"] = True
        elif kd == "QUANT":
            pending["quant"] = "all" if e.get("all") else e["n"]
        elif kd == "STATE":
            if last_subj is not None and not dest_next and not _binds_forward(cl, cl.index(e), toks, fa):
                # a state after a coordinated list ("the fork, the spoon and the knife have been used") describes each of them
                grp = [d for d in ents if d["role"] == "subject" and "state" not in d and d.get("set") == last_subj.get("set")]
                for d in (grp if cl.index(e) == len(cl) - 1 or all(x["k"] in ("WORD", "STATE") for x in cl[cl.index(e):]) else [last_subj]):
                    d["state"] = e["state"]
                    d["state_span"] = _span(toks, e["i"], e["j"])
                if last_subj.get("state") is None:
                    last_subj["state"], last_subj["state_span"] = e["state"], _span(toks, e["i"], e["j"])
            else:
                pending["state"] = (e["state"], _span(toks, e["i"], e["j"]))
        elif kd == "DEST":
            dest_next = True
        elif kd == "TYPE":
            role = "dest" if dest_next else "subject"
            d = _mk_entity(toks, e, group, role, pending)
            d["set"] = group * 10
            out["entities"].append(d)
            st["focus"].append(d)
            ents.append(d)
            if role == "subject":
                last_subj = d
                declarative.append({"subject": d, "dest": None})
            elif declarative:
                declarative[-1]["dest"] = d
            dest_next = False
        elif kd == "EVERY":
            d = {"type": "*", "universal": True, "span": _span(toks, e["i"], e["j"]), "role": "universal", "group": group}
            out["entities"].append(d)
            st["focus"].append(d)
            ents.append(d)
            last_subj = d
        elif kd == "SUBJ":
            pass
    subjects = [x for x in ents if x["role"] in ("subject", "universal")]
    # an event statement ("nobody will eat it"): kept as a fact, never a goal
    if ops and any(e["k"] == "SUBJ" for e in cl):
        o = ops[0]
        out["facts"].append({"kind": "event", "op": o["op"], "neg": any(e["k"] == "NEG" for e in cl) or bool(o.get("neg")),
                             "span": sentence_span})
        for x in subjects:
            pass
        return
    if where is not None and fn and subjects:
        out["facts"].append({"kind": "place_by_function", "fn": fn[0], "span": sentence_span,
                             "subjects": [{"type": s["type"], "span": s["span"]} for s in subjects]})
        return
    pairs = [p for p in declarative if p["dest"] is not None]
    copula = any(e["k"] == "WORD" and toks[e["i"]] in (("is", "are", "was", "were") if not fa else ("است", "هستند", "هست", "بود")) for e in cl)
    if pairs and copula:
        out["facts"].append({"kind": "relation", "span": sentence_span, "rel": "in",
                             "pairs": [{"type": p["subject"]["type"], "target": p["dest"]["type"]} for p in pairs]})
        return
    if pairs:
        out["facts"].append({"kind": "declarative_target", "span": sentence_span,
                             "subjects": [{"type": p["subject"]["type"], "state": p["subject"].get("state"), "span": p["subject"]["span"],
                                           "dest": {"type": p["dest"]["type"], "span": p["dest"]["span"]}} for p in pairs],
                             "others": [{"type": p["subject"]["type"], "state": p["subject"].get("state"), "span": p["subject"]["span"]}
                                        for p in declarative if p["dest"] is None]})
        return
    if not ents and not any(e["k"] == "STATE" for e in cl):
        words = [toks[e["i"]] for e in cl if e["k"] == "WORD"]
        if words:
            out["facts"].append({"kind": "unparsed", "span": sentence_span, "words": words})
        return
    # state statements and wishes about a stated state
    states = [e for e in cl if e["k"] == "STATE"]
    if wants and subjects and any(e["k"] == "NEG" for e in cl):
        out["facts"].append({"kind": "negated_wish", "span": sentence_span, "about": [{"type": d["type"], "state": d.get("state")} for d in subjects]})
        return
    if wants and subjects:
        wanted = []
        for s in subjects:
            if s.get("state"):
                wanted.append({"type": s["type"], "state": s["state"], "span": s["state_span"], "cue": _span(toks, wants[0]["i"], wants[0]["j"])})
        if wanted:
            out["wanted_states"].extend(wanted)
            out["desires"].append({"span": _span(toks, wants[0]["i"], wants[0]["j"]), "op": None,
                                   "about": [{"type": w["type"], "state": w["state"]} for w in wanted]})
            for w in wanted:
                for tp, role in AFFORDANCE.get(w["state"], []):
                    out["affordances"].append({"via": w["span"], "word": w["state"], "type": tp, "role": role, "rule": "affordance:" + w["state"]})
            return
    for s in states:
        tgt = next((d for d in reversed(subjects) if d.get("state") == s["state"]), None) or (subjects[-1] if subjects else None)
        fact = {"kind": "state", "state": s["state"], "span": _span(toks, s["i"], s["j"])}
        if tgt:
            fact["type"] = tgt["type"]
            fact["subject_span"] = tgt["span"]
        elif any(x["k"] == "PRON" for x in cl) and st["focus"]:
            fact["type"] = st["focus"][-1].get("type")
            fact["inherited"] = True
        else:
            fact["type"] = None
        out["facts"].append(fact)
        for tp, role in AFFORDANCE.get(s["state"], []) if s["state"] in ("dirty", "dark", "bright", "bad") else []:
            out["affordances"].append({"via": fact["span"], "word": s["state"], "type": tp, "role": role, "rule": "affordance:" + s["state"] + ":fact"})
    if wants and not states:
        out["desires"].append({"span": _span(toks, wants[0]["i"], wants[0]["j"]), "op": None, "about": [{"type": d["type"]} for d in subjects]})


# ------------------------------------------------------------------------------------------------------------- post-processing
_OP_NEEDS_OBJ = {"put", "pick", "open", "close", "toggle:on", "toggle:off", "slice", "heat", "cook", "wash", "fill", "empty", "break", "chill",
                 "flip", "scrub", "discard", "find", "tidy"}


def _finish(st) -> None:
    out, toks = st["out"], st["toks"]
    ops = out["operations"]
    for k, o in enumerate(ops):
        o["index"] = k
    edges = {}
    for k in range(1, len(ops)):
        cue = ops[k].get("after_cue")
        edges[(k - 1, k)] = {"before": k - 1, "after": k, "rel": "then", "explicit": cue is not None, "span": cue}
    for k, o in enumerate(ops):                              # "Before A, B" / "A after B": the order word reverses or confirms the list order
        od = o.get("order")
        if not od:
            continue
        partner = k + 1 if od["clause_initial"] and k == 0 else k - 1 if k > 0 else None
        if partner is None or not (0 <= partner < len(ops)):
            continue
        # "before X" in this clause: the partner clause's action comes first; "after X": this clause's action comes first
        first, second = (partner, k) if od["word"] == "before" else (k, partner)
        edges.pop((min(k, partner), max(k, partner)), None)
        edges[(first, second)] = {"before": first, "after": second, "rel": od["word"], "explicit": True, "span": od["span"]}
    out["temporal"] = [edges[key] for key in sorted(edges, key=lambda t: (t[0], t[1]))]
    order = list(range(len(ops)))
    for e in out["temporal"]:
        a, b = e["before"], e["after"]
        if order.index(a) > order.index(b):
            order.remove(a)
            order.insert(order.index(b), a)
    out["order"] = order
    fp = []
    for o in ops:
        if o.get("neg"):
            continue
        objs = [x for x in o["obj"] if x.get("type") and x["type"] != "*"]
        op = o["op"]
        for x in objs:
            if op == "put" and o.get("dest") and o["dest"].get("type"):
                fp.append({"type": x["type"], "rel": "in", "target": o["dest"]["type"], "span": o["span"], "from": "operation"})
            if op == "serve" and o.get("dest") and o["dest"].get("type"):
                fp.append({"type": x["type"], "rel": "in", "target": o["dest"]["type"], "span": o["span"], "from": "operation"})
            if x.get("state") and x.get("state_span"):
                fp.append({"type": x["type"], "state": x["state"], "span": x["state_span"], "from": "stated_state"})
            for opk, stt in (("pick", "held"), ("slice", "sliced"), ("toggle:on", "on"), ("toggle:off", "off"), ("open", "open"),
                             ("close", "closed")):
                if op == opk:
                    fp.append({"type": x["type"], **({"rel": "held"} if stt == "held" else {"state": stt}), "span": o["span"], "from": "operation"})
        d = o.get("dest")
        if d and d.get("type") and d.get("state") and d.get("state_span"):
            fp.append({"type": d["type"], "state": d["state"], "span": d["state_span"], "from": "stated_state"})
    for f in out["facts"]:
        if f.get("kind") == "declarative_target":
            for s in f["subjects"]:
                if s.get("state"):
                    fp.append({"type": s["type"], "state": s["state"], "span": s["span"], "from": "declarative"})
                fp.append({"type": s["type"], "rel": "in", "target": s["dest"]["type"], "span": f["span"], "from": "declarative"})
    for w in out["wanted_states"]:
        fp.append({"type": w["type"], "state": w["state"], "span": w["span"], "from": "wish"})
    seen, uniq = set(), []
    for p in fp:
        k = json.dumps({a: p.get(a) for a in ("type", "rel", "target", "state")}, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    out["final_predicates"] = uniq
    for e in out["entities"]:
        if e.get("quantity") is not None:
            out["quantities"].append({"type": e.get("type"), "quantity": e["quantity"], "span": e["span"]})
    for o in ops:
        if o["op"] in _OP_NEEDS_OBJ and not [x for x in o["obj"] if x.get("type") or x.get("surface")]:
            out["unresolved"].append({"op": o["index"], "slot": "obj", "span": o["span"]})
        elif [x for x in o["obj"] if x.get("unresolved")]:
            out["unresolved"].append({"op": o["index"], "slot": "obj", "span": o["span"], "why": "pronoun"})
        if o["op"] == "put" and not o.get("dest"):
            out["unresolved"].append({"op": o["index"], "slot": "dest", "span": o["span"]})
    needs_ok = [n for n in out["needs"] if not n.get("unresolved")]
    declared = any(f.get("kind") == "declarative_target" for f in out["facts"])
    if ops and not out["unresolved"]:
        out["status"], out["reason"] = "ok", ""
    elif ops:
        out["status"], out["reason"] = "partial", "an operation has an unresolved argument"
    elif needs_ok or declared or out["wanted_states"]:
        out["status"], out["reason"] = "ok", ""
    elif out["facts"] and all(f.get("kind") == "unparsed" for f in out["facts"]) and not out["desires"] and not out["needs"]:
        out["status"] = "unsupported"
        out["reason"] = "no known operation, need or state in: " + " ".join(sum([f.get("words", []) for f in out["facts"]], []))[:80]
    elif out["needs"] and all(n.get("unresolved") for n in out["needs"]):
        out["status"], out["reason"] = "unsupported", "a tool is wanted for an operation outside the known set"
    elif out["facts"] or out["desires"] or out["needs"]:
        out["status"], out["reason"] = "indirect", "facts or wishes without an operation word: no goal is made up"
    else:
        out["status"], out["reason"] = "unsupported", "no operation, need, wish, fact or declared target found"
    seen, aff = set(), []
    for a in out["affordances"]:
        k = (a["type"], a["role"], a["word"])
        if k not in seen:
            seen.add(k)
            aff.append(a)
    out["affordances"] = aff
    out["fallback"] = out["status"] in ("partial", "unsupported")


# ------------------------------------------------------------------------------------------------------------- retrieval key
def retrieval_key(schema: dict) -> list:
    """A light key that is NOT rendered: [{op, obj, dest}] in instruction order, English canonical names only.

    op is the operation key ('put', 'toggle:on' ...); obj / dest are types or None; a function destination ('washed') is dest_fn.
    A stated state with an operation that reaches it adds {op, obj, via_state: True} when no listed operation already does.
    Needs give {'op': <wanted operation>, 'need': 'tool'}. Facts add nothing except a stated state when nothing else is there."""
    key = []
    ops = schema.get("operations", [])
    for o in [ops[k] for k in (schema.get("order") or range(len(ops)))]:
        objs = [x for x in o.get("obj", []) if x.get("type") or x.get("surface")] or [None]
        dest = o.get("dest") or {}
        for x in objs:
            item = {"op": o["op"], "obj": (x or {}).get("type") if x else None, "dest": dest.get("type")}
            if x and x.get("surface") and not x.get("type"):
                item["obj_surface"] = x["surface"]
            if x and x.get("type") == "*":
                item["obj"], item["scope"] = None, "all"
            if dest.get("fn"):
                item["dest_fn"] = dest["fn"]
            if (o.get("src") or {}).get("type"):
                item["src"] = o["src"]["type"]
            if (o.get("tool") or {}).get("type"):
                item["tool"] = o["tool"]["type"]
            if o.get("neg"):
                item["neg"] = True
            if o.get("mood") == "desire":
                item["mood"] = "desire"
            key.append(item)
    have = {(k["op"], k.get("obj")) for k in key}
    for o in schema.get("operations", []):
        for x in o.get("obj", []):
            s = x.get("state")
            op2 = _STATE_OP.get(s)
            if op2 and x.get("type") and (op2, x["type"]) not in have and not o.get("neg"):
                key.append({"op": op2, "obj": x["type"], "dest": None, "via_state": s})
                have.add((op2, x["type"]))
    for w in schema.get("wanted_states", []):
        op2 = _STATE_OP.get(w["state"])
        if op2 and (op2, w["type"]) not in have:
            key.append({"op": op2, "obj": w["type"], "dest": None, "via_state": w["state"]})
            have.add((op2, w["type"]))
    for f in schema.get("facts", []):
        if f.get("kind") == "declarative_target":
            for s in f["subjects"]:
                key.append({"op": "put", "obj": s["type"], "dest": s["dest"]["type"]})
                op2 = _STATE_OP.get(s.get("state"))
                if op2 and (op2, s["type"]) not in have:
                    key.append({"op": op2, "obj": s["type"], "dest": None, "via_state": s["state"]})
        elif f.get("kind") == "place_by_function":
            for s in f["subjects"]:
                key.append({"op": "put", "obj": s["type"], "dest": None, "dest_fn": f["fn"]})
    for n in schema.get("needs", []):
        if n.get("op") and n.get("kind") == "item" and (n.get("type") or n.get("surface")):
            po = next((x.get("type") for x in n.get("for", []) if x.get("type")), None)
            key.append({"op": n["op"], "obj": po, "dest": None, "need": "item", "need_type": n.get("type") or "?"})   # "?" = unknown noun (A26)
        elif n.get("op"):
            key.append({"op": n["op"], "obj": None, "dest": None, "need": n.get("kind", "tool")})
        elif n.get("type"):
            key.append({"op": None, "obj": n["type"], "dest": None, "need": "item"})
    if not key:
        for f in schema.get("facts", []):
            if f.get("kind") == "state" and f.get("state"):
                key.append({"op": None, "obj": f.get("type"), "dest": None, "state": f["state"]})
    return key


def canonical(schema: dict, surfaces: bool = False) -> str:
    """Language-free comparison string of the key (execution order). Unknown nouns (no type) are compared only when surfaces=True."""
    parts = []
    for k in retrieval_key(schema):
        obj = k.get("obj") or k.get("scope") or (k.get("obj_surface") if surfaces else ("?" if k.get("obj_surface") else None))
        parts.append("%s(%s>%s%s%s%s%s%s%s%s)" % (k.get("op"), obj, k.get("dest"), "~" + k["dest_fn"] if k.get("dest_fn") else "",
                                                 ("<" + k["src"] if k.get("src") else "") + ("@" + k["tool"] if k.get("tool") else ""), "!" if k.get("neg") else "",
                                                 "?" + k["state"] if k.get("state") else "", "^" + k["via_state"] if k.get("via_state") else "",
                                                 "#" + k["need"] if k.get("need") else "", "=" + k["need_type"] if k.get("need_type") else ""))
    return " ".join(parts)


def key_types(schema: dict) -> list:
    return sorted({e["type"] for e in schema.get("entities", []) if e.get("type") and e["type"] != "*"})


def affordance_types(schema: dict) -> list:
    return sorted({a["type"] for a in schema.get("affordances", [])})


def schema_ops(schema: dict) -> list:
    out = []
    for k in retrieval_key(schema):
        if k.get("op") and k["op"] not in out:
            out.append(k["op"])
    return out


def glossary(schema: dict) -> list:
    """[(surface text, English type)] for every typed mention, so a non-English request can be shown with the types it resolved to
    (the Persian word for a spoon is a Spoon, not a Spatula). Surfaces are the normalised tokens of the instruction."""
    seen, out = set(), []
    for e in schema.get("entities", []):
        sp, t = e.get("span"), e.get("type")
        if sp and t and t != "*" and (sp["text"], t) not in seen:
            seen.add((sp["text"], t))
            out.append((sp["text"], t))
    return out


def compact(schema: dict) -> dict:
    """The small JSON-safe form that travels in a package query (no raw tokens, no spans)."""
    return {"schema": SCHEMA, "status": schema["status"], "key": retrieval_key(schema), "affordance_types": affordance_types(schema),
            "ops": schema_ops(schema), "types": key_types(schema), "holding": (schema.get("context", {}).get("holding") or {}).get("type"),
            "fallback": bool(schema.get("fallback")), "reason": schema.get("reason", "")}


# ------------------------------------------------------------------------------------------------- round 6: the need trigger
# Amendment R6-D1 (2026-10-13, stage K-A2; design docs/final-test/kitchen_memory/ROUND6-NEED-DESIGN.md, accepted by Astra).
# ONE meaning, the same in English and Persian. A request is a NEED when it is
#   (a) a need / want cue plus an indefinite placeholder with a purpose, and no satisfying object named
#       ("I need something to flip the pancakes with", «چیزی برای بریدن لازم دارم», «یه چیزی می‌خوام که ...»), or
#   (b) a desired state of a named object with no operation stated that reaches it ("I want to eat this apple cold later",
#       «سیب را سرد می‌خواهم»).
# Negations, questions and mere preferences ("I like cold apples") never fire; no single word (cold, warm, سرد, گرم) fires by itself.
# The trigger names no type and rewrites nothing: the requested meaning is kept, the planner chooses the means. The cue lists live in
# sim/step5/lexicon/need_cues.json (general usage; no item id, no type, no per-language target map).
NEED_LEX_PATH = os.path.join(HERE, "step5", "lexicon", "need_cues.json")
_NEED_LEX = None


def need_lexicon() -> dict:
    """The need-cue lexicon, Persian entries normalised like the instruction text (loaded once)."""
    global _NEED_LEX
    if _NEED_LEX is None:
        raw = json.load(open(NEED_LEX_PATH, encoding="utf-8"))
        lex = {"version": raw.get("version")}
        for lang in ("en", "fa"):
            d = {}
            for k, v in raw[lang].items():
                if k == "cue_bigram":
                    d[k] = [tuple(x) for x in v]
                else:
                    d[k] = {(N(w) if lang == "fa" else w) for w in v}
            lex[lang] = d
        _NEED_LEX = lex
    return _NEED_LEX


def _need_sentences(text: str) -> list:
    """[(tokens, is_question_mark)] per sentence of the raw text."""
    import re as _re
    out = []
    for s in _re.split(r"(?<=[.!?؟;؛])\s+", str(text or "").strip()):
        if not s.strip():
            continue
        q = s.rstrip().endswith(("?", "؟"))
        toks = [t for t in LC.prep(s) if t not in (H, S)]
        if toks:
            out.append((toks, q))
    return out


def _cue_at(toks, d, fa) -> list:
    """Indices of real need / want cues (not preference words)."""
    idx = []
    for i, w in enumerate(toks):
        if w in d["cue"]:
            idx.append(i)
    if not fa:
        for i in range(len(toks) - 1):
            if (toks[i], toks[i + 1]) in d["cue_bigram"]:
                idx.append(i + 1)
    return sorted(set(idx))


def _cue_negated(toks, i, d, fa) -> bool:
    if fa:
        if i > 0 and toks[i - 1] in d["cue_prefix_negation"]:
            return True
        if any(w in d["negation_after_cue"] for w in toks[i + 1:i + 3]):
            return True
        return any(w in d["negation"] for w in toks)
    return any(w in d["negation"] for w in toks[:i + 3])


def _sentence_question(toks, q, d, fa) -> bool:
    if fa:
        return toks[0] in d["question_first"] or q
    return q or (toks[0] in d["question_first"] and q)


_POST_SKIP = {"later", "tomorrow", "today", "please", "again", "را", "بعدا", "فردا", "لطفا", "دوباره"}


def _post_object_states(sc: dict, fa: bool) -> list:
    """[(type, state)] for a state word that follows a typed object as its complement ('this tomato warm', «گوجه را گرم»), within
    three tokens of the same clause. It is an operation instead when it is followed by a Persian light verb ('گرم کنم') or, in
    English, when the same word is an operation verb placed before its object ('to warm the tomato')."""
    toks = sc.get("tokens") or []
    out = []
    for e in sc.get("entities") or []:
        t, sp = e.get("type"), (e.get("span") or {}).get("tok")
        if not t or t == "*" or not sp:
            continue
        j = sp[1]
        k = j
        while k < len(toks) and k < j + 4 and toks[k] not in (H, S):
            w = toks[k]
            if w in _POST_SKIP or w in _DET_EN:
                k += 1
                continue
            st = _STATE_PH.get((w,))
            if st and not (fa and k + 1 < len(toks) and toks[k + 1] in _LIGHT_VERBS_FA):
                out.append((t, st))
            break
    return out


def need_trigger(text: str, lang: Optional[str] = None, schema: Optional[dict] = None) -> dict:
    """-> {"fires": bool, "kind": "placeholder" | "desired_state" | None, "reason": str, "cue": str | None, "lex": version}.
    Reads the instruction text and its own parse only (the parse is made here when not given)."""
    return _need_trigger_core(text, lang, schema)[0]


# Round 6b (amended protocol R6b, ROUND6B-FIXES.md section 2): the thing / place split of a placeholder NEED. It is decided by the
# placeholder token alone, from the frozen, language-symmetric lists of sim/step5/lexicon/need_cues_r6b.json (a copy of r6-v1 plus
# placeholder_thing / placeholder_place). The r6b file is read only when need_trigger_r6b is called, so need_trigger (r6) is unchanged.
NEED_LEX_R6B_PATH = os.path.join(HERE, "step5", "lexicon", "need_cues_r6b.json")
_NEED_LEX_R6B = None


def need_lexicon_r6b() -> dict:
    """The r6b placeholder lists {lang: {"thing": set, "place": set}} + version, Persian entries normalised (loaded once)."""
    global _NEED_LEX_R6B
    if _NEED_LEX_R6B is None:
        raw = json.load(open(NEED_LEX_R6B_PATH, encoding="utf-8"))
        lex = {"version": raw.get("version")}
        for lang in ("en", "fa"):
            nf = (lambda w: N(w)) if lang == "fa" else (lambda w: w)
            lex[lang] = {"thing": {nf(w) for w in raw[lang]["placeholder_thing"]}, "place": {nf(w) for w in raw[lang]["placeholder_place"]}}
        _NEED_LEX_R6B = lex
    return _NEED_LEX_R6B


def need_trigger_r6b(text: str, lang: Optional[str] = None, schema: Optional[dict] = None) -> dict:
    """need_trigger's verdict (same dict, same r6 lexicon) plus "placeholder": "thing" | "place" | None (None unless kind is
    "placeholder") and "placeholder_lex" (the r6b lexicon version). Same inputs as need_trigger: the text, the language, its own parse."""
    res, tok = _need_trigger_core(text, lang, schema)
    lx = need_lexicon_r6b()
    out = dict(res, placeholder=None, placeholder_lex=lx.get("version"))
    if res.get("fires") and res.get("kind") == "placeholder" and tok is not None:
        fa = (lang == "fa") if lang in ("en", "fa") else _is_fa(text)
        d = lx["fa" if fa else "en"]
        out["placeholder"] = "thing" if tok in d["thing"] else ("place" if tok in d["place"] else None)
    return out


def _need_trigger_core(text: str, lang: Optional[str] = None, schema: Optional[dict] = None):
    """-> (need_trigger verdict, the placeholder token that fired or None). The r6 logic, unchanged."""
    lexd = need_lexicon()
    fa = (lang == "fa") if lang in ("en", "fa") else _is_fa(text)
    d = lexd["fa" if fa else "en"]
    sc = schema if isinstance(schema, dict) else parse_goal(text, "fa" if fa else "en")
    res = {"fires": False, "kind": None, "reason": "no_need_cue", "cue": None, "lex": lexd.get("version")}
    real_cue = False
    reasons = []
    for toks, q in _need_sentences(text):
        cues = _cue_at(toks, d, fa)
        if not cues:
            if any(w in d["preference"] for w in toks):
                reasons.append("preference_only")
            continue
        if _sentence_question(toks, q, d, fa):
            reasons.append("question")
            continue
        if all(_cue_negated(toks, i, d, fa) for i in cues):
            reasons.append("negated")
            continue
        real_cue = True
        ci = [i for i in cues if not _cue_negated(toks, i, d, fa)][0]
        # (a) placeholder + purpose, no satisfying object named
        for p, w in enumerate(toks):
            if w not in d["placeholder"]:
                continue
            if not fa and p < ci:
                continue
            after = toks[p + 1:]
            purpose = (after[:1] and after[0] in d["purpose_after_placeholder"]) if not fa else any(x in d["purpose_after_placeholder"] for x in after)
            if not purpose:
                continue
            if any(n.get("type") for n in sc.get("needs") or []):
                reasons.append("satisfying_object_named")
                continue
            return dict(res, fires=True, kind="placeholder", reason="need_cue+placeholder", cue=toks[ci]), w
    if real_cue:
        # (b) a desired state of a named object, with no operation stated that reaches it
        imperative = [o for o in sc.get("operations") or [] if not o.get("neg") and o.get("mood") != "desire"]
        if imperative:
            return dict(res, reason="operation_stated"), None
        desired = []
        for o in sc.get("operations") or []:
            if o.get("neg") or o.get("mood") != "desire":
                continue
            for x in o.get("obj") or []:
                s = x.get("state")
                if s and x.get("type") and x["type"] != "*" and _STATE_OP.get(s) != o.get("op"):
                    desired.append((x["type"], s))
        for w in sc.get("wanted_states") or []:
            if w.get("type") and w.get("state"):
                desired.append((w["type"], w["state"]))
        desired += _post_object_states(sc, fa)
        if desired:
            return dict(res, fires=True, kind="desired_state", reason="desired_state_without_operation", cue=desired[0][1]), None
        reasons.append("need_cue_without_placeholder_or_desired_state")
    if reasons:
        res["reason"] = reasons[-1] if "negated" not in reasons and "question" not in reasons else ("negated" if "negated" in reasons else "question")
    return res, None


if __name__ == "__main__":                                  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--lang", default=None)
    a = ap.parse_args()
    print(json.dumps(parse_goal(a.text, a.lang), ensure_ascii=False, indent=1))
