"""B4 (package spec pkg-1, section 7) - procedural knowledge base: what each honest tool needs and does, and a few
composite routes. Pure, offline, no LLM, no simulator, no kitchen constant.

    kb = load_kb()                          # sim/step5/procedures_kb.json (+ B9's verification results, if present)
    kb.check_static(exec_contract, tools, types, validator)  -> [problem, ...]   ([] = ok; validator injected by the caller,
                                            planners/procedures_check.py: this module never loads the executor, even transitively)
    kb.expand(ops, types, categories)       -> {"tools", "appliances", "receptacles", "requires", "actions"}
    kb.records()                            -> [pkg.rec/1]  (component "procedure", kitchen "*")
    kb.candidates(query)                    -> [pkg.cand/1] (structural match only; no lexical / dense scoring)
    kb.instantiate(entry_id, binding)       -> [step, ...]  (route with slots bound; for B9)

Honesty: every entry is a statement about the TOOLS (taken from honest_skills / step4_gemma through the executor
contract) or generic household realism, never about a kitchen's current state. Executor facts and realism advice are
kept apart (`kind: executor` vs `kind: realism`); a realism item is never a validator rule. Hidden state (a door that
may be locked, an object that may not be there) is never asserted.

The JSON is GENERATED from DEFINITIONS below (`python sim/ltm_procedural.py --build`); the selftest checks the file
against the build. Python 3.10 syntax (WSL imports this module).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KB_PATH = HERE / "step5" / "procedures_kb.json"
FEAS_PATH = ROOT / "docs" / "final-test" / "kitchen_memory" / "l3_feasibility.jsonl"      # B9 rows (mode "procedures")
VERIFIED_PATH = ROOT / "docs" / "final-test" / "kitchen_memory" / "procedures_verified.json"   # optional {id: {"wsl": bool}}
EXEC_CONTRACT_PATH = HERE / "planners" / "exec_contract.json"          # GENERATED: python sim/planners/validator_v2.py --write
SAMPLE_CONTRACT_PATH = HERE / "_pkg_fixtures" / "exec_contract.sample.json"  # fixture: only through an explicit path
EXEC_PATHS = (EXEC_CONTRACT_PATH,)
SPLIT_PATH = ROOT / "docs" / "final-test" / "split_manifest.json"
TYPES_PATH = ROOT / "docs" / "final-test" / "matrix" / "detector_types.json"

SCHEMA_KB = "pkg.procs/1"
SCHEMA_PROC = "pkg.proc/1"
SCHEMA_REC = "pkg.rec/1"
KB_VERSION = "procs-1"

#: closed relation list (spec 3.4) - the ones B4 emits
RELS = ("contains", "tool_at", "requires", "appliance", "controls", "near", "opens", "toggles", "stores", "effect",
        "confuse")

#: static vocabulary (domain spec, written once; not tuned on any kitchen). Types are detector classes.
VOCAB = {
    "tool_types": ["ButterKnife", "Spatula", "Fork", "Spoon", "Ladle"],        # A26 item 1 (K-B2b): Knife out of scope
    "appliance_types": ["Microwave", "CoffeeMachine", "Toaster", "StoveBurner", "StoveKnob", "Faucet", "SinkBasin",
                        "Sink"],
    "receptacle_types": ["Fridge", "Microwave", "Cabinet", "Drawer", "GarbageCan", "SinkBasin", "Sink", "CounterTop",
                         "Plate", "Bowl", "Pot", "Pan", "Toaster", "CoffeeMachine", "StoveBurner", "DiningTable",
                         "Shelf"],
}
CATEGORIES = {
    "food": ["Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato"],
    "sliceable": ["Apple", "Bread", "Lettuce", "Potato", "Tomato"],
    "dish": ["Bowl", "Cup", "Mug", "Plate", "Pan", "Pot", "Fork", "Spoon", "ButterKnife", "Spatula",
             "Kettle"],
    "fillable": ["Bottle", "Bowl", "Cup", "Kettle", "Mug", "Pot", "WineBottle"],
    "vessel": ["Pot", "Pan"],
    "breakable": ["Egg", "Mug", "Plate", "Cup", "Bowl", "Vase", "WineBottle"],
    "dirtyable": ["Bowl", "Cup", "Mug", "Plate", "Pan", "Pot"],
    "openable": ["Fridge", "Cabinet", "Drawer", "Microwave"],
    "toggleable": ["CoffeeMachine", "Faucet", "LightSwitch", "Microwave", "StoveKnob", "Toaster"],
    "stowable": ["Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato", "Bowl", "Cup", "Mug", "Plate", "Pan", "Pot",
                 "Fork", "Spoon", "ButterKnife", "Spatula", "SaltShaker", "PepperShaker", "SoapBottle",
                 "DishSponge", "Kettle"],
    "discardable": ["Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato", "Cup", "Mug", "Plate", "Bowl",
                    "PaperTowelRoll"],
    "receptacle": ["Fridge", "Microwave", "Cabinet", "Drawer", "GarbageCan", "SinkBasin", "Sink", "CounterTop", "Plate",
                   "Bowl", "Pot", "Pan", "Toaster", "CoffeeMachine", "StoveBurner", "DiningTable", "Shelf"],
    "pickupable": ["Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato", "Bowl", "Cup", "Mug", "Plate", "Pan",
                   "Pot", "Fork", "Spoon", "ButterKnife", "Spatula", "SaltShaker", "PepperShaker",
                   "SoapBottle", "DishSponge", "Kettle", "Bottle", "WineBottle", "Book", "Vase"],
}
#: parser verb (ltm_concepts) -> entry ops it can trigger; the query's ops may be parser verbs or tool names
VERB_OPS = {
    "pick": ["pickup", "pick_and_place"], "put": ["put_in", "pick_and_place"], "open": ["open"], "close": ["close"],
    "slice": ["slice"], "toggle": ["toggle_on", "toggle_off"], "find": ["search_for", "go_to"], "heat": ["heat"],
    "cook": ["cook"], "wash": ["wash"], "fill": ["fill"], "empty": ["empty"], "drop": ["drop"], "break": ["break"],
    "clean": ["clean"], "dirty": ["dirty"],
}
FA_NOTE_TOKENS = ()      # nothing kitchen specific

#: Affordance map (I-F2, round 1): reached through the OPERATION word (English; the parser maps a Persian verb to an
#: English operation first). Each row is general household knowledge about which tool or appliance a verb or a goal
#: word needs, plus the recipe entry that carries the steps. Never a kitchen fact, never derived from a reference plan.
#: `tools` = held tools (detector types); `appliances` = where it happens, in order of preference.
AFFORD = {
    "cut":       {"tools": ["ButterKnife"], "appliances": [], "recipe": "proc:slice-with-butterknife", "tuned_on": ['g-l3-tool-cut', 'g-l3-tool-flip', 'g-l3-cutlery-sink'],
                  "en": "To cut or slice something use the ButterKnife: pickup ButterKnife first."},   # A26 items 2-3 (K-B2b)
    "flip":      {"tools": ["Spatula"], "appliances": [], "recipe": None, "tuned_on": ['g-l3-tool-cut', 'g-l3-tool-flip', 'g-l3-cutlery-sink'],
                  "en": "To flip or turn food over use the Spatula: pickup Spatula first."},
    "spread":    {"tools": ["ButterKnife"], "appliances": [], "recipe": None, "tuned_on": ['g-l3-tool-cut', 'g-l3-tool-flip', 'g-l3-cutlery-sink'],
                  "en": "To spread something use the ButterKnife: pickup ButterKnife first."},
    "scoop":     {"tools": ["Spoon"], "appliances": [], "recipe": None, "tuned_on": ['g-l3-tool-cut', 'g-l3-tool-flip', 'g-l3-cutlery-sink'],
                  "en": "To scoop or stir liquid or soup use the Spoon: pickup Spoon first."},
    "stab":      {"tools": ["Fork"], "appliances": [], "recipe": None, "tuned_on": ['g-l3-tool-cut', 'g-l3-tool-flip', 'g-l3-cutlery-sink'],
                  "en": "To pick food up on a prong use the Fork (pickup Fork first); the Spoon is for liquid, the Spatula for flipping."},
    "chill":     {"tools": [], "appliances": ["Fridge"], "recipe": "proc:chill", "tuned_on": ['g-l3-chilled-apple'],
                  "en": "To chill or keep food cold put it in the Fridge, then close Fridge."},
    "heat":      {"tools": [], "appliances": ["Microwave", "StoveBurner"], "recipe": "proc:heat-then-place", "tuned_on": ['g-l2-microwave-run-egg', 'g-l2-microwave-run-potato'],
                  "en": "To warm food use heat (the Microwave); a Pot or Pan is warmed on the StoveBurner with its "
                        "StoveKnob."},
    "boil":      {"tools": [], "appliances": ["StoveBurner"], "recipe": "proc:stove-pan",
                  "en": "To boil or fry use a Pot or Pan on the StoveBurner, then toggle_on StoveKnob."},
    "toast":     {"tools": [], "appliances": ["Toaster"], "recipe": "proc:toast",
                  "en": "To toast bread put it in the Toaster, then toggle_on Toaster."},
    "brew":      {"tools": [], "appliances": ["CoffeeMachine"], "recipe": "proc:coffee",
                  "en": "To make coffee put a Mug in the CoffeeMachine, then toggle_on CoffeeMachine."},
    "wash":      {"tools": [], "appliances": ["SinkBasin", "Faucet"], "recipe": "proc:wash-dish",
                  "en": "To wash a dish use wash (SinkBasin and Faucet); it switches the water itself."},
    "discard":   {"tools": [], "appliances": ["GarbageCan"], "recipe": "proc:discard",
                  "en": "To throw something away put it in the GarbageCan."},
}
#: Tool-request rule (J-P2, Astra 2026-10-13): "something to X with" asks for the TOOL only. The line names the tool and says
#: to stop; it never tells the agent to perform X. Generated from the rows above, so the tool names cannot drift.
_REQ_DO = {"cut": ("cut", "slice or cut anything"), "flip": ("flip", "flip anything"), "spread": ("spread", "spread anything"),
           "scoop": ("scoop or stir", "scoop or stir anything"), "stab": ("spear food", "spear anything")}
for _k, _a in AFFORD.items():
    if _a["tools"] and _k in _REQ_DO:
        _a["request"] = ("You were asked only for something to %s with: pickup %s and stop there. Do not %s unless the "
                         "command also says to." % (_REQ_DO[_k][0], _a["tools"][0], _REQ_DO[_k][1]))
#: word -> AFFORD key (lower case, inflections handled by afford_key)
AFFORD_WORDS = {
    "cut": "cut", "slice": "cut", "chop": "cut", "dice": "cut", "carve": "cut", "trim": "cut",
    "flip": "flip", "turn over": "flip", "toss": "flip",
    "spread": "spread", "butter": "spread",
    "scoop": "scoop", "stir": "scoop", "ladle": "scoop", "serve soup": "scoop",
    "stab": "stab", "spear": "stab",
    "chill": "chill", "cool": "chill", "cold": "chill", "refrigerate": "chill", "freeze": "chill",
    "keep cold": "chill", "keep fresh": "chill",
    "heat": "heat", "warm": "heat", "reheat": "heat", "microwave": "heat", "cook": "heat", "bake": "heat",
    "boil": "boil", "fry": "boil", "simmer": "boil", "saute": "boil",
    "toast": "toast", "brew": "brew", "coffee": "brew",
    "wash": "wash", "rinse": "wash", "scrub": "wash",
    "discard": "discard", "throw": "discard", "trash": "discard", "bin": "discard",
}
#: the holding-state rule (spec 3.1): what to do first, by what the hand holds relative to the target X
HOLDING_RULES = (
    {"when": "holding_target", "en": "X is already in your hand: do not pickup it again; go straight to put_in R "
                                     "(not pick_and_place).",
     "action": "skip_pickup_use_put_in"},
    {"when": "holding_other", "en": "Your hand holds something else: put it down first (put_in on a receptacle you "
                                    "have seen, else drop), then pickup or pick_and_place X.",
     "action": "release_first"},
    {"when": "location_unknown", "en": "pickup and pick_and_place walk and search by themselves; use search_for X "
                                       "first only when the command is to find it.",
     "action": "search_only_if_asked"},
)
#: Release of a non-target held object (Astra I-F2: no kitchen constant): put_in on a receptacle the CALLER passes (one
#: the agent has actually seen), else the contract's `drop`, and only without either, a CounterTop fallback.
RELEASE_FALLBACK = "CounterTop"


# ============================================================================================ definitions
def _ex(cond, **kw):
    d = {"cond": cond, "kind": "executor"}
    d.update(kw)
    return d


def _re(cond, **kw):
    d = {"cond": cond, "kind": "realism"}
    d.update(kw)
    return d


def _st(action, obj=None, to=None, arg=None):
    return {"action": action, "object": obj, "to": to, "arg": arg}


def _tool(op, ops, en, fa, route, slots, *, requires=(), realism=(), effects=(), hand_end=None, then=None,
          confuse=(), applies=None, lex=("", ""), prio=0):
    return {"id": "proc:tool-" + op.replace("_", "-"), "op": op, "ops": list(ops) + [op], "composite": False,
            "applies_to": applies or {"categories": [], "types": []}, "requires": list(requires),
            "realism": list(realism), "effects": list(effects), "route": route, "slots": slots, "hand_end": hand_end,
            "then": then, "confuse": list(confuse), "text": {"en": en, "fa": fa}, "lex": {"en": lex[0], "fa": lex[1]},
            "support": [], "prio": 100 + prio}


def _comp(eid, op, triggers, en, fa, route, slots, applies, *, requires=(), realism=(), effects=(), hand_end=None,
          then=None, confuse=(), support=(), lex=("", ""), prio=0, extra_ops=(), tail=(), goal=None, polarity=None, needs_dest=False):
    return {"id": "proc:" + eid, "op": op, "ops": list(triggers) + list(extra_ops), "triggers": list(triggers),
            "composite": True, "applies_to": applies, "requires": list(requires), "realism": list(realism),
            "effects": list(effects), "route": route, "slots": slots, "hand_end": hand_end, "then": then,
            "confuse": list(confuse), "text": {"en": en, "fa": fa}, "lex": {"en": lex[0], "fa": lex[1]},
            "support": list(support), "prio": 10 + prio, "tail": list(tail), "goal": list(goal or []),
            "polarity": polarity, "needs_dest": bool(needs_dest)}


def _slot(cat, samples):
    return {"category": cat, "samples": list(samples)}


def _support(*ids):
    return list(ids)


def definitions():
    """The entries, in prio order. Returns a fresh list of dicts."""
    T = []
    T.append(_tool(
        "pickup", ["pick"],
        "pickup: walks to the object and picks it up; the hand must be empty first. To carry it to a place use "
        "pick_and_place.",
        "pickup: به شیء می‌رود و آن را برمی‌دارد؛ دست باید از قبل خالی باشد. برای بردن آن به جای دیگر از "
        "pick_and_place استفاده کنید.",
        [_st("pickup", "?item")], {"?item": _slot("pickupable", ["Apple", "Mug", "Fork"])},
        requires=[_ex("the hand is empty", tool="pickup", field="hand", value="take")],
        effects=[{"hand": "?item ends in hand"}], hand_end="held", lex=("pick up take grab", "برداشتن بردار"), prio=0))
    T.append(_tool(
        "put_in", ["put"],
        "put_in R: puts the HELD object into or onto the receptacle R (opens R if closed, and may leave its door open). The "
        "object field is the receptacle, not the held item; with nothing held it is an error.",
        "put_in R: شیئی را که در دست دارید داخل یا روی R می‌گذارد (اگر بسته باشد باز می‌کند). فیلد object خود "
        "گیرنده است نه شیء در دست؛ اگر دست خالی باشد خطا است.",
        [_st("pickup", "?item"), _st("put_in", "?place")],
        {"?item": _slot("stowable", ["Apple", "Mug"]), "?place": _slot("receptacle", ["CounterTop", "SinkBasin"])},
        requires=[_ex("something is held", tool="put_in", field="requires_held", value=True)],
        effects=[{"hand": "ends empty"}], hand_end="empty", lex=("put place set down", "گذاشتن بگذار"), prio=1))
    T.append(_tool(
        "pick_and_place", ["put", "pick"],
        "pick_and_place X to R: picks X up and puts it into or onto R (it walks, searches and opens R itself, and may "
        "leave its door open); the hand is empty afterwards. Closing R is a separate step. If X is already in your hand "
        "use put_in R.",
        "pick_and_place X to R: X را برمی‌دارد و داخل یا روی R می‌گذارد (خودش راه می‌رود، جستجو می‌کند و R را باز "
        "می‌کند)؛ بعد از آن دست خالی است. بستن R گام جدا است. اگر X در دست است از put_in R استفاده کنید.",
        [_st("pick_and_place", "?item", "?place")],
        {"?item": _slot("stowable", ["Apple", "Mug"]), "?place": _slot("receptacle", ["CounterTop", "Fridge"])},
        requires=[_ex("the hand is empty or holds X", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"hand": "ends empty"}], hand_end="empty", lex=("put place move carry store", "گذاشتن بگذار ببر"),
        prio=2))
    T.append(_tool(
        "open", ["open"],
        "open X: opens a fridge, cabinet, drawer or microwave; it stays open until a close X step. pick_and_place and "
        "put_in already open a closed target themselves.",
        "open X: یخچال، کابینت، کشو یا مایکروویو را باز می‌کند. pick_and_place و put_in هدف بسته را خودشان باز "
        "می‌کنند.",
        [_st("open", "?target")], {"?target": _slot("openable", ["Fridge", "Cabinet"])},
        effects=[{"state": "open", "type": "?target"}], hand_end="same", lex=("open", "باز کردن باز کن"), prio=3))
    T.append(_tool(
        "close", ["close"],
        "close X: closes it. It is its own step: add it when the command says to close something or to leave it "
        "closed.",
        "close X: آن را می‌بندد. این یک گام جدا است: وقتی دستور بستن یا بسته گذاشتن می‌خواهد آن را اضافه کنید.",
        [_st("close", "?target")], {"?target": _slot("openable", ["Fridge", "Drawer"])},
        effects=[{"state": "closed", "type": "?target"}], hand_end="same", lex=("close shut", "بستن ببند"), prio=4))
    T.append(_tool(
        "toggle_on", ["toggle"],
        "toggle_on X: switches on a faucet, stove knob, coffee machine, toaster, microwave or light. A burner is "
        "switched through its StoveKnob. A Microwave only switches on while its door is closed.",
        "toggle_on X: شیر آب، knob اجاق، دستگاه قهوه، توستر، مایکروویو یا چراغ را روشن می‌کند. شعله از راه "
        "StoveKnob روشن می‌شود. مایکروویو فقط با درِ بسته روشن می‌شود: اول close Microwave.",
        [_st("toggle_on", "?target")], {"?target": _slot("toggleable", ["Faucet", "Toaster"])},
        effects=[{"state": "on", "type": "?target"}], hand_end="same",
        lex=("turn on switch on start", "روشن کردن روشن کن"), prio=5))
    T.append(_tool(
        "toggle_off", ["toggle"],
        "toggle_off X: switches it off again (faucet, stove knob, coffee machine, toaster, microwave, light).",
        "toggle_off X: آن را دوباره خاموش می‌کند (شیر آب، knob اجاق، دستگاه قهوه، توستر، مایکروویو، چراغ).",
        [_st("toggle_off", "?target")], {"?target": _slot("toggleable", ["Faucet", "Toaster"])},
        effects=[{"state": "off", "type": "?target"}], hand_end="same",
        lex=("turn off switch off stop", "خاموش کردن خاموش کن"), prio=6))
    T.append(_tool(
        "slice", ["slice"],
        "slice X: slices it. The executor needs the ButterKnife in hand; pickup ButterKnife first.",
        "slice X: آن را برش می‌دهد. اجراکننده کارد کره‌خوری را در دست لازم دارد؛ اول pickup ButterKnife.",
        [_st("slice", "?food")], {"?food": _slot("sliceable", ["Apple", "Tomato"])},
        requires=[_ex("the ButterKnife is in hand", tool="slice", field="requires_tool", value=["ButterKnife"])],
        effects=[{"state": "sliced", "type": "?food"}], hand_end="same", lex=("cut slice chop", "برش بریدن ببر"), prio=7,
        applies={"categories": ["sliceable"], "types": []}))
    T.append(_tool(
        "break", ["break"], "break X: breaks or cracks it (for example an Egg, Mug or Plate).",
        "break X: آن را می‌شکند (مثلا Egg، Mug یا Plate).",
        [_st("break", "?target")], {"?target": _slot("breakable", ["Egg", "Mug"])},
        effects=[{"state": "broken", "type": "?target"}], hand_end="same", lex=("break crack smash", "شکستن بشکن"),
        prio=8, applies={"categories": ["breakable"], "types": []}))
    T.append(_tool(
        "dirty", ["dirty"], "dirty X: makes it dirty.", "dirty X: آن را کثیف می‌کند.",
        [_st("dirty", "?target")], {"?target": _slot("dirtyable", ["Plate", "Mug"])},
        effects=[{"state": "dirty", "type": "?target"}], hand_end="same", lex=("dirty soil", "کثیف کردن کثیف کن"),
        prio=9, applies={"categories": ["dirtyable"], "types": []}))
    T.append(_tool(
        "clean", ["clean"],
        "clean X: cleans it where it is. To rinse something in the sink use wash instead.",
        "clean X: آن را همان‌جا که هست تمیز می‌کند. برای شستن در سینک از wash استفاده کنید.",
        [_st("clean", "?target")], {"?target": _slot("dirtyable", ["Plate", "Mug"])},
        effects=[{"state": "clean", "type": "?target"}], hand_end="same", lex=("clean scrub wipe", "تمیز کردن تمیز کن"),
        confuse=[{"with": "wash", "note": "wash carries the item to the sink and rinses it there; clean acts in place"}],
        prio=10, applies={"categories": ["dirtyable"], "types": []}))
    T.append(_tool(
        "fill", ["fill"],
        "fill X arg: fills the container with a liquid; arg is water, coffee or wine (water when omitted).",
        "fill X arg: ظرف را با مایع پر می‌کند؛ arg یکی از water، coffee یا wine است (پیش‌فرض water).",
        [_st("fill", "?container", None, "?liquid")],
        {"?container": _slot("fillable", ["Mug", "Cup"]), "?liquid": _slot("liquid", ["water", "coffee"])},
        requires=[_ex("arg is one of water, coffee, wine", tool="fill", field="arg", value="liquid")],
        effects=[{"state": "filled", "type": "?container"}], hand_end="same", lex=("fill pour water coffee", "پر کردن پر کن"),
        prio=11, applies={"categories": ["fillable"], "types": []}))
    T.append(_tool(
        "empty", ["empty"], "empty X: pours the liquid out of the container.", "empty X: مایع داخل ظرف را خالی می‌کند.",
        [_st("empty", "?container")], {"?container": _slot("fillable", ["Mug", "Cup"])},
        effects=[{"state": "empty", "type": "?container"}], hand_end="same", lex=("empty pour out", "خالی کردن خالی کن"),
        prio=12, applies={"categories": ["fillable"], "types": []}))
    T.append(_tool(
        "heat", ["heat"],
        "heat X: runs the whole microwave cycle itself (in, door closed, on, out again). Use it only when the command "
        "is to warm or heat the food. It is not 'put X in the microwave': X does not stay inside.",
        "heat X: کل دور مایکروویو را خودش انجام می‌دهد (گذاشتن، بستن در، روشن کردن، بیرون آوردن). فقط برای دستور "
        "گرم کردن غذا. این «X را در مایکروویو بگذار» نیست: X داخل نمی‌ماند.",
        [_st("heat", "?food")], {"?food": _slot("food", ["Egg", "Potato"])},
        requires=[_ex("a Microwave exists", fixture="Microwave"),
                  _ex("the hand is empty or holds the food", tool="heat", field="hand", value="take")],
        effects=[{"state": "hot", "type": "?food"}, {"hand": "?food ends in hand"}], hand_end="held",
        then="to place it somewhere, follow with put_in or pick_and_place",
        confuse=[{"with": "cook", "note": "cook also uses the Microwave and leaves the food in hand; 'put X in the "
                                          "microwave and close it' is pick_and_place + close, not heat"}],
        lex=("heat warm reheat microwave", "گرم کردن گرم کن مایکروویو"), prio=13,
        applies={"categories": ["food"], "types": []}))
    T.append(_tool(
        "cook", ["cook"],
        "cook X: the same whole microwave cycle as heat (put in, close, switch on, take out). Use it only when the "
        "command is to cook the food; it does not leave X in the microwave.",
        "cook X: همان دور کامل مایکروویو مثل heat (گذاشتن، بستن در، روشن کردن، بیرون آوردن). فقط وقتی دستور پختن "
        "غذا است از آن استفاده کنید؛ X را در مایکروویو نمی‌گذارد.",
        [_st("cook", "?food")], {"?food": _slot("food", ["Egg", "Potato"])},
        requires=[_ex("a Microwave exists", fixture="Microwave"),
                  _ex("the hand is empty or holds the food", tool="cook", field="hand", value="take")],
        effects=[{"state": "cooked", "type": "?food"}, {"hand": "?food ends in hand"}], hand_end="held",
        then="to place it somewhere, follow with put_in or pick_and_place",
        confuse=[{"with": "heat", "note": "heat and cook share one routine here; only the goal word differs"}],
        lex=("cook bake", "پختن بپز"), prio=14, applies={"categories": ["food"], "types": []}))
    T.append(_tool(
        "wash", ["wash"],
        "wash X: rinses the item in the sink (SinkBasin and Faucet) and leaves it there; the hand is empty "
        "afterwards.",
        "wash X: شیء را در سینک (SinkBasin و Faucet) می‌شوید و همان‌جا رها می‌کند؛ بعد از آن دست خالی است.",
        [_st("wash", "?item")], {"?item": _slot("dish", ["Plate", "Mug"])},
        requires=[_ex("a SinkBasin exists", fixture="SinkBasin"), _ex("a Faucet exists", fixture="Faucet"),
                  _ex("the hand is empty or holds the item", tool="wash", field="hand", value="take-release")],
        effects=[{"state": "clean", "type": "?item"}, {"hand": "ends empty"}], hand_end="empty",
        confuse=[{"with": "clean", "note": "clean acts in place; wash carries the item to the sink"}],
        lex=("wash rinse dishes", "شستن بشوی"), prio=15, applies={"categories": ["dish"], "types": []}))
    T.append(_tool(
        "go_to", ["find"], "go_to X: walks to the object and looks at it.", "go_to X: به شیء می‌رود و به آن نگاه می‌کند.",
        [_st("go_to", "?target")], {"?target": _slot("pickupable", ["Apple", "Mug"])}, hand_end="same",
        lex=("go walk to", "رفتن برو"), prio=16))
    T.append(_tool(
        "search_for", ["find"],
        "search_for X: looks for the object in likely places and opens closed containers; it succeeds only if the "
        "object is actually seen.",
        "search_for X: شیء را در جاهای محتمل می‌گردد و محفظه‌های بسته را باز می‌کند؛ فقط وقتی موفق است که شیء "
        "واقعا دیده شود.",
        [_st("search_for", "?target")], {"?target": _slot("pickupable", ["Apple", "Mug"])}, hand_end="same",
        lex=("find search look for locate", "پیدا کردن پیدا کن بگرد"), prio=17))
    T.append(_tool(
        "drop", ["drop"], "drop: lets go of what you are holding (needs a held object).",
        "drop: چیزی را که در دست دارید رها می‌کند (باید شیئی در دست باشد).",
        [_st("pickup", "?item"), _st("drop")], {"?item": _slot("pickupable", ["Apple", "Mug"])},
        requires=[_ex("something is held", tool="drop", field="requires_held", value=True)],
        effects=[{"hand": "ends empty"}], hand_end="empty", lex=("drop let go release", "رها کردن رها کن"), prio=18))
    T.append(_tool(
        "close_all", ["close"], "close_all: closes every container you have seen open.",
        "close_all: هر محفظه‌ای را که باز دیده‌اید می‌بندد.",
        [_st("close_all")], {}, hand_end="same", lex=("close all tidy", "بستن همه"), prio=19))
    T.append(_tool(
        "crouch", [], "crouch: crouches to see low places; stand undoes it.",
        "crouch: خم می‌شود تا جاهای پایین را ببیند؛ stand آن را برمی‌گرداند.",
        [_st("crouch")], {}, hand_end="same", lex=("crouch low", "خم شدن"), prio=20))
    T.append(_tool(
        "stand", [], "stand: stands up again after crouch.", "stand: بعد از crouch دوباره می‌ایستد.",
        [_st("stand")], {}, hand_end="same", lex=("stand up", "ایستادن"), prio=21))

    C = []
    C.append(_comp(
        "slice-with-butterknife", "slice", ["slice"],
        "To slice: pickup ButterKnife, then slice the food.",
        "برای برش: اول pickup ButterKnife، بعد slice روی غذا.",
        [_st("pickup", "ButterKnife"), _st("slice", "?food")], {"?food": _slot("sliceable", ["Apple", "Tomato", "Potato"])},
        {"categories": ["sliceable"], "types": ["ButterKnife"]},
        requires=[_ex("the hand is empty to take the ButterKnife", tool="pickup", field="hand", value="take"),
                  _ex("the ButterKnife is in hand", tool="slice", field="requires_tool", value=["ButterKnife"])],
        effects=[{"state": "sliced", "type": "?food"}, {"hand": "ButterKnife stays in hand"}], hand_end="held",
        support=_support("g-l2-slice-apple-potato", "g-l2-slice-bread-tomato", "g-l2-slice-potato-tomato",
                         "g-l2-slice-tomato-lettuce", "g-l3-salad"),
        lex=("slice cut chop salad butter", "برش بریدن سالاد کره‌خوری"), prio=0))   # A26: was "knife" / "چاقو"
    C.append(_comp(
        "heat-then-place", "heat", ["heat", "cook"],
        "To warm food and then place it: heat the food (it does the whole microwave cycle), then pick_and_place it to "
        "the place.",
        "برای گرم کردن و بعد گذاشتن غذا: heat روی غذا (کل دور مایکروویو را خودش انجام می‌دهد)، بعد pick_and_place "
        "به مقصد.",
        [_st("heat", "?food"), _st("pick_and_place", "?food", "?place")],
        {"?food": _slot("food", ["Egg", "Potato", "Tomato"]), "?place": _slot("receptacle", ["Plate", "Bowl", "CounterTop"])},
        {"categories": ["food"], "types": ["Microwave"]},
        requires=[_ex("a Microwave exists", fixture="Microwave"),
                  _ex("heat leaves the food in hand", tool="heat", field="hand", value="take")],
        effects=[{"state": "hot", "type": "?food"}, {"hand": "ends empty"}], hand_end="empty",
        confuse=[{"with": "cook", "note": "heat and cook share one routine; a command to put food in the microwave "
                                          "and close it is not a heat"}],
        support=_support("g-l2-heat-egg-bowl", "g-l2-heat-potato-plate", "g-l2-heat-apple-plate",
                         "g-l2-heat-bread-counter-top", "g-l2-heat-potato-counter-top", "g-l2-heat-tomato-plate",
                         "g-l3-breakfast"),
        lex=("heat warm microwave place plate bowl", "گرم کردن مایکروویو بشقاب کاسه"), prio=1, needs_dest=True))
    C.append(_comp(
        "microwave-put-close", "pick_and_place", ["put"],
        "To put something in the Microwave: pick_and_place it to Microwave, then close Microwave. This is not heat: "
        "heat takes the food out again.",
        "برای گذاشتن چیزی در Microwave: pick_and_place به Microwave، بعد close Microwave. این heat نیست: heat غذا را "
        "دوباره بیرون می‌آورد.",
        [_st("pick_and_place", "?item", "Microwave"), _st("close", "Microwave")],
        {"?item": _slot("stowable", ["Apple", "Mug", "Bread"])}, {"categories": [], "types": ["Microwave"]},
        requires=[_ex("a Microwave exists", fixture="Microwave"),
                  _ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "closed", "type": "Microwave"}, {"hand": "ends empty"}], hand_end="empty",
        confuse=[{"with": "heat", "note": "heat/cook warm the food and return it to your hand; this only stores it"},
                 {"with": "cook", "note": "heat/cook warm the food and return it to your hand; this only stores it"}],
        support=_support("g-l2-microwave-apple", "g-l2-microwave-bread", "g-l2-microwave-egg", "g-l2-microwave-mug",
                         "g-l2-microwave-potato", "g-l2-microwave-tomato", "g-l2-microwave-run-bread",
                         "g-l2-microwave-run-egg", "g-l2-microwave-run-potato"),
        lex=("put microwave close store", "گذاشتن مایکروویو ببند"), prio=2, extra_ops=["close"]))
    C.append(_comp(
        "microwave-run", "toggle_on", ["toggle"],
        "To run the Microwave with something inside: pick_and_place it to Microwave, close Microwave, then toggle_on "
        "Microwave (close first). The food stays inside; do not take it out unless the command says to (then: "
        "toggle_off Microwave, open Microwave, pickup it). Done when it is inside, the door is closed and it is on.",
        "برای اجرای Microwave با چیزی داخل: pick_and_place به Microwave، close Microwave، toggle_on Microwave. غذا "
        "داخل می‌ماند؛ فقط اگر دستور گفت بیرون بیاورید (toggle_off، open، pickup). تمام شد: داخل، در بسته، روشن.",
        [_st("pick_and_place", "?item", "Microwave"), _st("close", "Microwave"), _st("toggle_on", "Microwave")],
        {"?item": _slot("stowable", ["Egg", "Potato", "Bread"])}, {"categories": [], "types": ["Microwave"]},
        requires=[_ex("a Microwave exists", fixture="Microwave"),
                  _ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "on", "type": "Microwave"}, {"state": "closed", "type": "Microwave"},
                 {"hand": "ends empty"}], hand_end="empty",
        tail=[_st("toggle_off", "Microwave"), _st("open", "Microwave"), _st("pickup", "?item")],
        goal=[{"type": "?item", "key": "parentReceptacles", "contains": "Microwave"},
              {"type": "Microwave", "key": "isOpen", "value": False},
              {"type": "Microwave", "key": "isToggled", "value": True}], polarity="on",
        confuse=[{"with": "heat", "note": "heat does the whole cycle and takes the food out; this leaves it inside"}],
        support=_support("g-l2-microwave-run-bread", "g-l2-microwave-run-egg", "g-l2-microwave-run-potato"),
        lex=("microwave run turn on switch on start close put", "مایکروویو روشن کن ببند"), prio=2, extra_ops=["close", "put"]))
    C.append(_comp(
        "faucet-run", "toggle_on", ["toggle"],
        "To run the water: toggle_on Faucet; toggle_off Faucet stops it. wash already switches the Faucet on and off "
        "itself, so do not add it to a wash.",
        "برای باز کردن آب: toggle_on Faucet؛ toggle_off Faucet آن را می‌بندد. wash خودش Faucet را روشن و خاموش "
        "می‌کند، پس آن را به wash اضافه نکنید.",
        [_st("toggle_on", "Faucet")], {}, {"categories": [], "types": ["Faucet"]},
        requires=[_ex("a Faucet exists", fixture="Faucet")],
        effects=[{"state": "on", "type": "Faucet"}], hand_end="same",
        tail=[_st("toggle_off", "Faucet")],
        confuse=[{"with": "wash", "note": "wash runs the Faucet itself"}],
        support=[], lex=("faucet water tap run turn on", "شیر آب باز کن"), prio=2))
    C.append(_comp(
        "chill", "pick_and_place", ["put"],
        "To chill food: pick_and_place it to Fridge, then close Fridge.",
        "برای سرد نگه داشتن غذا: pick_and_place به Fridge، بعد close Fridge.",
        [_st("pick_and_place", "?food", "Fridge"), _st("close", "Fridge")],
        {"?food": _slot("stowable", ["Apple", "Bread", "Tomato"])}, {"categories": [], "types": ["Fridge"]},
        requires=[_ex("a Fridge exists", fixture="Fridge"),
                  _ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "closed", "type": "Fridge"}, {"hand": "ends empty"}], hand_end="empty",
        support=_support("g-l2-fridge-apple", "g-l2-fridge-bread", "g-l2-fridge-lettuce", "g-l2-fridge-potato",
                         "g-l2-fridge-tomato", "g-l3-groceries", "g-l2-find-place-bread-fridge", "g-l3-chilled-apple"),
        lex=("put fridge cold chill store groceries close", "گذاشتن یخچال سرد ببند"), prio=3, extra_ops=["close"]))
    C.append(_comp(
        "put-away-close", "pick_and_place", ["put"],
        "To put something away in a Cabinet or Drawer: pick_and_place it there, then close it.",
        "برای جمع کردن چیزی در Cabinet یا Drawer: pick_and_place به آن‌جا، بعد آن را ببندید.",
        [_st("pick_and_place", "?item", "?container"), _st("close", "?container")],
        {"?item": _slot("stowable", ["Bowl", "SaltShaker", "Spatula"]), "?container": _slot("openable", ["Cabinet", "Drawer"])},
        {"categories": [], "types": ["Cabinet", "Drawer"]},
        requires=[_ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "closed", "type": "?container"}, {"hand": "ends empty"}], hand_end="empty",
        support=_support("g-l2-cabinet-bowl", "g-l2-cabinet-pepper-shaker", "g-l2-cabinet-salt-shaker",
                         "g-l2-cabinet-soap-bottle", "g-l2-drawer-butter-knife", "g-l2-drawer-spatula"),
        lex=("put away store cabinet drawer close tidy", "جمع کردن کابینت کشو ببند"), prio=4, extra_ops=["close"]))
    C.append(_comp(
        "discard", "pick_and_place", ["put"],
        "To throw something away: pick_and_place it to GarbageCan.",
        "برای دور انداختن چیزی: pick_and_place به GarbageCan.",
        [_st("pick_and_place", "?item", "GarbageCan")], {"?item": _slot("discardable", ["Egg", "Tomato", "Bread"])},
        {"categories": [], "types": ["GarbageCan"]},
        requires=[_ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"hand": "ends empty"}], hand_end="empty",
        support=_support("g-l3-bad-tomato", "g-l3-rotten-egg", "g-l3-stale-bread", "g-l3-wilted-lettuce",
                         "g-l1-place-egg-garbage-can"),
        lex=("throw away discard garbage trash bin rotten", "دور انداختن سطل زباله"), prio=5))
    C.append(_comp(
        "dish-to-sink", "pick_and_place", ["put"],
        "To put dishes in the sink: pick_and_place each one to SinkBasin.",
        "برای گذاشتن ظرف‌ها در سینک: برای هر کدام pick_and_place به SinkBasin.",
        [_st("pick_and_place", "?dish", "SinkBasin")], {"?dish": _slot("dish", ["Cup", "Fork", "Spatula"])},
        {"categories": [], "types": ["SinkBasin", "Sink"]},
        requires=[_ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"hand": "ends empty"}], hand_end="empty",
        support=_support("g-l3-cutlery-sink", "g-l2-find-place-cup-sink-basin", "g-l2-find-place-dish-sponge-sink-basin",
                         "g-l2-find-place-fork-sink-basin", "g-l2-find-place-spatula-sink-basin",
                         "g-l1-place-apple-sink-basin"),
        lex=("put dishes sink basin cutlery", "ظرف سینک"), prio=6))
    C.append(_comp(
        "empty-to-sink", "empty", ["empty"],
        "To pour out a container and put it in the sink: empty it, then pick_and_place it to SinkBasin.",
        "برای خالی کردن ظرف و گذاشتن آن در سینک: empty، بعد pick_and_place به SinkBasin.",
        [_st("empty", "?container"), _st("pick_and_place", "?container", "SinkBasin")],
        {"?container": _slot("fillable", ["Mug", "Cup"])}, {"categories": ["fillable"], "types": ["SinkBasin"]},
        requires=[_ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "empty", "type": "?container"}, {"hand": "ends empty"}], hand_end="empty",
        support=_support("g-l2-empty-mug-sink"),
        lex=("empty pour sink", "خالی کردن سینک"), prio=7))
    C.append(_comp(
        "wash-dish", "wash", ["wash"],
        "To wash a dish: wash it (SinkBasin and Faucet needed); it stays in the sink and the hand ends empty.",
        "برای شستن ظرف: wash (به SinkBasin و Faucet نیاز دارد)؛ ظرف در سینک می‌ماند و دست خالی است.",
        [_st("wash", "?dish")], {"?dish": _slot("dish", ["Plate", "Mug", "Pan"])},
        {"categories": ["dish"], "types": ["SinkBasin", "Faucet"]},
        requires=[_ex("a SinkBasin exists", fixture="SinkBasin"), _ex("a Faucet exists", fixture="Faucet"),
                  _ex("the hand is empty or holds the dish", tool="wash", field="hand", value="take-release")],
        effects=[{"state": "clean", "type": "?dish"}, {"hand": "ends empty"}], hand_end="empty",
        confuse=[{"with": "clean", "note": "clean acts in place; wash carries the dish to the sink"}],
        support=[], lex=("wash dishes sink soap", "شستن ظرف سینک"), prio=8))
    C.append(_comp(
        "fill-container", "fill", ["fill"],
        "To fill a container: fill it with arg water, coffee or wine.",
        "برای پر کردن ظرف: fill با arg برابر water، coffee یا wine.",
        [_st("fill", "?container", None, "?liquid")],
        {"?container": _slot("fillable", ["Mug", "Cup"]), "?liquid": _slot("liquid", ["water", "coffee"])},
        {"categories": ["fillable"], "types": []},
        requires=[_ex("arg is one of water, coffee, wine", tool="fill", field="arg", value="liquid")],
        effects=[{"state": "filled", "type": "?container"}], hand_end="same",
        support=_support("g-l3-breakfast"), lex=("fill water coffee pour", "پر کردن آب قهوه"), prio=9))
    C.append(_comp(
        "coffee", "toggle_on", ["toggle", "fill", "put"],
        "To make coffee: pick_and_place a Mug to CoffeeMachine, then toggle_on CoffeeMachine. Needs a CoffeeMachine; "
        "availability unknown, check what you see.",
        "برای درست کردن قهوه: pick_and_place یک Mug به CoffeeMachine، بعد toggle_on CoffeeMachine. به CoffeeMachine "
        "نیاز دارد؛ وجود آن معلوم نیست، به آنچه می‌بینید نگاه کنید.",
        [_st("pick_and_place", "Mug", "CoffeeMachine"), _st("toggle_on", "CoffeeMachine")], {},
        {"categories": [], "types": ["CoffeeMachine"]},
        requires=[_ex("a CoffeeMachine exists", fixture="CoffeeMachine"),
                  _ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "on", "type": "CoffeeMachine"}, {"hand": "ends empty"}], hand_end="empty",
        support=[], lex=("coffee machine mug brew", "قهوه دستگاه قهوه ماگ"), prio=10))
    C.append(_comp(
        "toast", "toggle_on", ["toggle", "put"],
        "To toast bread: pick_and_place Bread to Toaster, then toggle_on Toaster. Needs a Toaster; availability "
        "unknown, check what you see.",
        "برای برشته کردن نان: pick_and_place از Bread به Toaster، بعد toggle_on Toaster. به Toaster نیاز دارد؛ وجود "
        "آن معلوم نیست، به آنچه می‌بینید نگاه کنید.",
        [_st("pick_and_place", "Bread", "Toaster"), _st("toggle_on", "Toaster")], {},
        {"categories": [], "types": ["Toaster"]},
        requires=[_ex("a Toaster exists", fixture="Toaster"),
                  _ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "on", "type": "Toaster"}, {"hand": "ends empty"}], hand_end="empty",
        support=[], lex=("toast bread toaster", "نان برشته توستر"), prio=11))
    C.append(_comp(
        "stove-pan", "toggle_on", ["toggle", "put", "cook"],
        "To warm a Pot or Pan on the stove: pick_and_place it to StoveBurner, then toggle_on StoveKnob. Needs a "
        "StoveBurner and a StoveKnob; availability unknown, check what you see.",
        "برای گرم کردن Pot یا Pan روی اجاق: pick_and_place به StoveBurner، بعد toggle_on StoveKnob. به StoveBurner "
        "و StoveKnob نیاز دارد؛ وجود آن‌ها معلوم نیست، به آنچه می‌بینید نگاه کنید.",
        [_st("pick_and_place", "?vessel", "StoveBurner"), _st("toggle_on", "StoveKnob")],
        {"?vessel": _slot("vessel", ["Pot", "Pan"])}, {"categories": ["vessel"], "types": ["StoveBurner", "StoveKnob"]},
        tail=[_st("toggle_off", "StoveKnob")],
        requires=[_ex("a StoveBurner exists", fixture="StoveBurner"), _ex("a StoveKnob exists", fixture="StoveKnob"),
                  _ex("pick_and_place leaves the hand empty", tool="pick_and_place", field="hand", value="take-release")],
        effects=[{"state": "on", "type": "StoveKnob"}, {"hand": "ends empty"}], hand_end="empty",
        confuse=[{"with": "heat", "note": "heat uses the Microwave; a pot on the stove needs the StoveKnob"}],
        support=_support("g-l1-on-stove-knob", "g-l1-off-stove-knob"),
        lex=("stove burner knob pot pan boil fry", "اجاق شعله قابلمه تابه"), prio=12))
    C.append(_comp(
        "find-then-take", "pickup", ["find", "pick"],
        "To find and take something: search_for it, then pickup (the hand must be empty).",
        "برای پیدا کردن و برداشتن چیزی: search_for، بعد pickup (دست باید خالی باشد).",
        [_st("search_for", "?item"), _st("pickup", "?item")], {"?item": _slot("pickupable", ["Fork", "Mug", "Lettuce"])},
        {"categories": [], "types": []},
        requires=[_ex("the hand is empty", tool="pickup", field="hand", value="take")],
        effects=[{"hand": "?item ends in hand"}], hand_end="held",
        support=_support("g-l2-find-pickup-bowl", "g-l2-find-pickup-egg",   # A26: g-l2-find-pickup-knife archived
                         "g-l2-find-pickup-lettuce", "g-l2-find-pickup-mug", "g-l2-find-pickup-pan",
                         "g-l2-find-pickup-soap-bottle", "g-l2-find-pickup-spoon"),
        lex=("find pick up search look for take", "پیدا کردن برداشتن بگرد"), prio=13))
    C.append(_comp(
        "tidy-up-close", "close_all", ["close"],
        "To close everything left open: close_all, or close each container by name.",
        "برای بستن هرچه باز مانده: close_all یا بستن هر محفظه با نامش.",
        [_st("close_all")], {}, {"categories": ["openable"], "types": [], "min_fit": 2}, effects=[{"state": "closed", "type": "?all"}],
        hand_end="same", support=_support("g-l3-left-open"),
        lex=("close all open left open tidy", "بستن همه باز"), prio=14))
    return T + C


# ============================================================================================ small helpers
def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _file_sha16(p) -> str:
    try:
        return _sha16(Path(p).read_bytes())
    except OSError:
        return "missing"


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _self_sha(obj: dict) -> str:
    return hashlib.sha256(canon({k: v for k, v in obj.items() if k != "sha256"}).encode("utf-8")).hexdigest()


def _fixed(v):
    return v if isinstance(v, str) and v and not v.startswith("?") else None


def _entry_keys(e) -> dict:
    types = set(e["applies_to"].get("types") or [])
    for st in e["route"]:
        for f in ("object", "to"):
            if _fixed(st.get(f)):
                types.add(st[f])
    for r in e["requires"] + e["realism"]:
        if r.get("fixture"):
            types.add(r["fixture"])
        if r.get("type"):
            types.add(r["type"])
    rels = []
    for r in e["requires"] + e["realism"]:
        t = r.get("fixture") or r.get("type")
        if t:
            rels.append(("appliance:" if t in VOCAB["appliance_types"] else "requires:") + t)
    for ef in e["effects"]:
        if ef.get("state"):
            rels.append("effect:" + ef["state"])
    for c in e["confuse"]:
        rels.append("confuse:" + c["with"])
    ops = sorted(set(e["ops"]))
    return {"types": sorted(types), "ops": ops, "rels": sorted(set(rels))}


def build_kb() -> dict:
    """The KB document, generated from definitions(). `source` records the honest_skills / step4_gemma hashes."""
    src = "domain spec + honest_skills@%s + step4_gemma@%s" % (_file_sha16(HERE / "honest_skills.py"),
                                                              _file_sha16(HERE / "step4_gemma.py"))
    entries = []
    for e in definitions():
        d = {"schema": SCHEMA_PROC, "id": e["id"], "op": e["op"], "composite": e["composite"],
             "triggers": list(e.get("triggers") or e["ops"]),
             "applies_to": e["applies_to"], "requires": e["requires"], "realism": e["realism"], "effects": e["effects"],
             "route": e["route"], "slots": e["slots"], "hand_end": e["hand_end"], "then": e["then"],
             "confuse": e["confuse"], "text": e["text"], "lex": e["lex"], "keys": _entry_keys(e),
             "tail": list(e.get("tail") or []), "goal": list(e.get("goal") or []), "polarity": e.get("polarity"),
             "needs_dest": bool(e.get("needs_dest")),
             "source": src, "provenance": {"support": e["support"], "spec_only": not e["support"]},
             "verified": {"static": False, "wsl": None}, "active": True, "prio": e["prio"]}
        entries.append(d)
    doc = {"schema": SCHEMA_KB, "version": KB_VERSION, "kitchen": "*", "vocab": VOCAB, "categories": CATEGORIES,
           "verb_ops": VERB_OPS, "afford": AFFORD, "afford_words": AFFORD_WORDS,
           "holding_rules": list(HOLDING_RULES), "entries": entries}
    doc["sha256"] = _self_sha(doc)
    return doc


# ============================================================================================ contract
REGENERATE_CONTRACT = "python sim/planners/validator_v2.py --write"


def load_exec_contract(path=None) -> dict:
    """The executor contract pkg.exec/1: sim/planners/exec_contract.json, or the file at an explicit `path` (fixtures).

    The contract is GENERATED on the planner side from the executor (planners/validator_v2.build_exec_contract; _pkg_contract_selftest
    compares it with a fresh derivation). This module never imports the executor to derive it and never substitutes another file:
    a missing file, a wrong schema or a content hash that does not verify raises (fail closed)."""
    p = Path(path) if path else EXEC_CONTRACT_PATH
    if not p.exists():
        raise FileNotFoundError("no executor contract at %s; regenerate it with `%s`" % (p, REGENERATE_CONTRACT))
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema") != "pkg.exec/1" or not isinstance(d.get("tools"), dict) or not d["tools"]:
        raise ValueError("%s is not a pkg.exec/1 contract with tools" % p)
    if d.get("sha256") != _contract_sha(d):
        raise ValueError("%s: content hash does not verify; regenerate it with `%s`" % (p, REGENERATE_CONTRACT))
    d["_origin"] = str(p)
    return d


def _contract_sha(d: dict) -> str:
    """sha256 of the canonical JSON without 'sha256' / '_origin' (planners/validator_v2.sha_of convention)."""
    body = {k: v for k, v in d.items() if k not in ("sha256", "_origin")}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def contract_tools(contract: dict) -> list:
    """The tool names of the executor contract (generated from honest_skills.TOOLS; same set)."""
    return sorted((contract or {}).get("tools") or {})


def _hand_effect(spec: dict) -> str:
    h = spec.get("hand") or "none"
    if spec.get("requires_held") and h in ("none", "keep"):
        h = "release"
    return "release" if h == "need" else h


def trace_hand(route, contract, holding=None):
    """Predicted hand after each step from the contract. Returns (final_hand, problems)."""
    tools = contract.get("tools") or {}
    hand, probs = holding, []
    for i, st in enumerate(route):
        spec = tools.get(st["action"])
        if spec is None:
            probs.append("step %d: tool %r not in the contract" % (i, st["action"]))
            continue
        eff = _hand_effect(spec)
        obj = st.get("object")
        if eff in ("take", "take-release") and hand and hand != obj:
            probs.append("step %d: %s while holding %s" % (i, st["action"], hand))
        if spec.get("requires_held") and not hand:
            probs.append("step %d: %s needs a held object" % (i, st["action"]))
        if eff == "take":
            hand = obj
        elif eff in ("take-release", "release"):
            hand = None
    return hand, probs


def _contract_route_problems(route, contract, liquids=("water", "coffee", "wine")):
    probs = []
    tools = contract.get("tools") or {}
    for i, st in enumerate(route):
        spec = tools.get(st["action"])
        if spec is None:
            probs.append("step %d: unknown tool %r" % (i, st["action"]))
            continue
        for fld in ("object", "to"):
            rule = spec.get(fld, "none")
            v = st.get(fld)
            if rule == "type" and not v:
                probs.append("step %d: %s needs %s" % (i, st["action"], fld))
            if rule == "none" and v:
                probs.append("step %d: %s takes no %s" % (i, st["action"], fld))
        if spec.get("arg", "none") == "none" and st.get("arg"):
            probs.append("step %d: %s takes no arg" % (i, st["action"]))
        if spec.get("arg", "none") not in ("none", None) and st.get("arg") and st["arg"] not in liquids:
            probs.append("step %d: bad arg %r" % (i, st["arg"]))
        if st.get("action") == "pick_and_place" and st.get("object") == st.get("to"):
            probs.append("step %d: pick_and_place onto itself" % i)
    return probs


# ============================================================================================ holding-aware routes
def fmt_step(st) -> str:
    a, o, t, g = st.get("action"), st.get("object"), st.get("to"), st.get("arg")
    if a == "pick_and_place":
        return "pick_and_place %s to %s" % (o, t)
    if a == "fill" and g:
        return "fill %s %s" % (o, g)
    return a if not o else "%s %s" % (a, o)


def adapt_route(route, holding, contract=None, release_to=None):
    """Rewrite a route for what the hand holds NOW, from the skill contract (no search step is ever added).
    Returns (new_route, notes). notes name the branch taken: holding_target / holding_other."""
    tools = (contract or {}).get("tools") or {}
    hand, out, notes = holding, [], []
    for st in route:
        a, o = st["action"], st.get("object")
        spec = tools.get(a) or {}
        eff = _hand_effect(spec) if spec else "none"
        if a == "pickup" and hand == o:
            notes.append("holding_target")                  # honest pickup of the held object is a no-op
            continue
        if a == "pick_and_place" and hand == o:
            notes.append("holding_target")
            out.append({"action": "put_in", "object": st["to"], "to": None, "arg": None})
            hand = None
            continue
        if eff in ("take", "take-release") and hand and hand != o:
            notes.append("holding_other")
            if release_to:
                out.append({"action": "put_in", "object": release_to, "to": None, "arg": None})
            elif "drop" in tools:
                out.append({"action": "drop", "object": None, "to": None, "arg": None})
            else:
                out.append({"action": "put_in", "object": RELEASE_FALLBACK, "to": None, "arg": None})
            hand = None
        out.append(dict(st))
        if eff == "take":
            hand = o
        elif eff in ("take-release", "release"):
            hand = None
    return out, sorted(set(notes))


def route_text(route, holding_note=(), lang="en") -> str:
    steps = [fmt_step(s) for s in route]
    if not steps:
        return "Nothing to do: it is already in your hand." if lang == "en" else "Nothing to do."
    return ", then ".join(steps) + "."


# ============================================================================================ query helpers
def _cat_members(kb_cats: dict, extra) -> dict:
    out = {k: set(v) for k, v in kb_cats.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict):
            mem = list(v.get("types") or [])
        elif isinstance(v, (list, tuple, set)):
            mem = list(v)
        else:
            continue
        out.setdefault(k, set()).update(mem)
    return out


def _query_ops(ops, roles=None) -> set:
    """Ops as a set of entry-op tokens: tool names stay, parser verbs expand, toggle direction from roles."""
    out = set()
    for o in ops or []:
        out.add(o)
    for r in roles or []:
        v = r.get("verb")
        if v == "toggle":
            req = r.get("req")
            if req == "on":
                out.add("toggle_on")
            elif req == "off":
                out.add("toggle_off")
    return out


class KB:
    def __init__(self, doc: dict, path=None):
        self.doc = doc
        self.path = str(path) if path else None
        self.entries = [dict(e, verified=dict(e.get("verified") or {}), provenance=dict(e.get("provenance") or {}))
                        for e in doc["entries"]]
        self.by_id = {e["id"]: e for e in self.entries}
        self.categories = doc.get("categories") or CATEGORIES
        self.vocab = doc.get("vocab") or VOCAB
        self.verb_ops = doc.get("verb_ops") or VERB_OPS
        self.afford_map = doc.get("afford") or AFFORD
        self.afford_words = doc.get("afford_words") or AFFORD_WORDS
        self.sha = doc.get("sha256") or _self_sha(doc)
        self.last_validator = None

    # ------------------------------------------------------------------ verification results (B9)
    #: B9 classes that deactivate an entry (a confirmed failure of the ROUTE itself); `perception` / `navigation` are
    #: inconclusive (they say nothing about the route) and a row with preconditions_ok == False is inapplicable.
    ROUTE_FAILS = ("executor_defect", "api_cannot_express")

    def apply_verified(self, rows=None, results=None):
        """Fill verified.wsl from B9's rows (mode "procedures", requirement = entry id, class "feasible") or a
        {id: {"wsl": bool}} map (Astra H-B4-procedural Q3). The last row per (entry, configuration) counts. True needs
        >= 2 distinct feasible configurations and no route-attributable failure; a route-attributable failure
        (executor_defect / api_cannot_express with the preconditions met) sets wsl False and active False. Inconclusive
        and inapplicable rows change nothing. An entry with no usable row is left untouched (wsl None)."""
        last = {}
        for r in rows or []:
            if r.get("mode") != "procedures" or r.get("requirement") not in self.by_id:
                continue
            if r.get("preconditions_ok") is False:
                continue
            last[(r["requirement"], r.get("item"))] = r
        by = {}
        for (eid, _item), r in last.items():
            by.setdefault(eid, []).append(r)
        for eid, rs in by.items():
            e = self.by_id[eid]
            bad = [r for r in rs if r.get("class") in self.ROUTE_FAILS]
            ok_items = {r.get("item") for r in rs if r.get("class") == "feasible"}
            if bad:
                e["verified"]["wsl"] = False
                e["active"] = False
            elif len(ok_items) >= 2:
                e["verified"]["wsl"] = True
        for eid, v in (results or {}).items():
            if eid in self.by_id and isinstance(v, dict) and v.get("wsl") is not None:
                self.by_id[eid]["verified"]["wsl"] = bool(v["wsl"])
                if v["wsl"] is False:
                    self.by_id[eid]["active"] = False
        self._recs = None
        return self

    # ------------------------------------------------------------------ static check
    def check_static(self, exec_contract: dict, tools: list, types: list, validator=None) -> list:
        """[problem] ([] = ok). `validator` = (name, fn) with fn(plan_obj, types, holding=None, tools=...) -> {"ok", "errors"}, injected
        by the caller (planners/procedures_check.validator()); without one the check is NOT complete and says so (no silent skip)."""
        probs = []
        tools, types = list(tools), set(types)
        seen = set()
        contract_tools = (exec_contract or {}).get("tools") or {}
        vname, vfn = validator if validator else (None, None)
        self.last_validator = vname
        if vfn is None:
            probs.append("no validator supplied: the route validation did not run (use planners/procedures_check.py)")
        if self.doc.get("schema") != SCHEMA_KB:
            probs.append("document schema %r != %s" % (self.doc.get("schema"), SCHEMA_KB))
        if self.doc.get("sha256") and self.doc["sha256"] != _self_sha(self.doc):
            probs.append("document sha256 does not match its content")
        for e in self.entries:
            eid = e.get("id", "?")
            if eid in seen:
                probs.append("%s: duplicate id" % eid)
            seen.add(eid)
            for f in ("op", "applies_to", "requires", "effects", "route", "text", "keys", "source", "verified"):
                if f not in e:
                    probs.append("%s: missing field %s" % (eid, f))
            if not e.get("route") or not isinstance(e.get("route"), list):
                probs.append("%s: empty route" % eid)
                continue
            for lang in ("en", "fa"):
                if not (e.get("text") or {}).get(lang, "").strip():
                    probs.append("%s: text.%s empty" % (eid, lang))
            # keys
            for t in e["keys"].get("types", []):
                if t not in types:
                    probs.append("%s: keys.types %r is not a detector type" % (eid, t))
            for o in e["keys"].get("ops", []):
                if o not in tools and o not in self.verb_ops:
                    probs.append("%s: keys.ops %r is neither a tool nor a parser verb" % (eid, o))
            for r in e["keys"].get("rels", []):
                if r.split(":", 1)[0] not in RELS:
                    probs.append("%s: relation %r not in RELS" % (eid, r))
            # applies_to
            for t in e["applies_to"].get("types", []):
                if t not in types:
                    probs.append("%s: applies_to type %r is not a detector type" % (eid, t))
            for c in e["applies_to"].get("categories", []):
                if c not in self.categories:
                    probs.append("%s: unknown category %r" % (eid, c))
            # requires: executor items must map to the contract or a fixture type; realism never to a validator rule
            for r in e["requires"]:
                if r.get("kind") != "executor":
                    probs.append("%s: requires item %r must be kind executor (realism has its own list)" % (eid, r.get("cond")))
                    continue
                if r.get("fixture"):
                    if r["fixture"] not in types:
                        probs.append("%s: fixture %r is not a detector type" % (eid, r["fixture"]))
                elif r.get("tool") and r.get("field"):
                    spec = contract_tools.get(r["tool"])
                    if spec is None:
                        probs.append("%s: requires tool %r not in the contract" % (eid, r["tool"]))
                    else:
                        got = spec.get(r["field"], "<absent>") if r["field"] != "arg" else \
                            ("liquid" if spec.get("arg", "none") not in ("none", None) else "none")
                        if r["field"] == "hand":
                            got = _hand_effect(spec)
                        if got != r["value"]:
                            probs.append("%s: requires %s.%s == %r but the contract says %r"
                                         % (eid, r["tool"], r["field"], r["value"], got))
                else:
                    probs.append("%s: executor requirement %r maps to neither a fixture nor a contract field" % (eid, r.get("cond")))
            for r in e.get("realism", []):
                if r.get("kind") != "realism":
                    probs.append("%s: realism item %r has kind %r" % (eid, r.get("cond"), r.get("kind")))
                if r.get("type") and r["type"] not in types:
                    probs.append("%s: realism type %r is not a detector type" % (eid, r["type"]))
            # route: action names, contract rules, validator, hand trace
            for i, st in enumerate(e["route"]):
                if st.get("action") not in tools:
                    probs.append("%s: route step %d action %r not in the tool list" % (eid, i, st.get("action")))
            bound_sets = self.sample_bindings(eid, 3)
            for b in bound_sets:
                route = self.instantiate(eid, b)
                for t in [x for st in route for x in (st.get("object"), st.get("to")) if x]:
                    if t not in types:
                        probs.append("%s: bound type %r is not a detector type" % (eid, t))
                for p in _contract_route_problems(route, exec_contract):
                    probs.append("%s: %s (binding %s)" % (eid, p, canon(b)))
                hand, hp = trace_hand(route, exec_contract)
                for p in hp:
                    probs.append("%s: %s (binding %s)" % (eid, p, canon(b)))
                he = e.get("hand_end")
                if he == "empty" and hand is not None:
                    probs.append("%s: declared hand_end empty but the trace ends holding %s" % (eid, hand))
                if he == "held" and hand is None:
                    probs.append("%s: declared hand_end held but the trace ends empty" % (eid))
                if vfn is not None:
                    try:
                        res = vfn({"plan": route}, types, holding=None, tools=tools)
                    except TypeError:
                        res = vfn({"plan": route}, types, None, tools)
                    if not res.get("ok"):
                        probs.append("%s: validator rejects the route (%s) binding %s"
                                     % (eid, "; ".join(str(x.get("reason") or x.get("code")) for x in res.get("errors", [])), canon(b)))
            # text hygiene
            blob = canon(e)
            if re.search(r"FloorPlan|\bFP\d|Synth", blob):
                probs.append("%s: kitchen name in entry" % eid)
            if re.search(r"-?\d+\.\d+", canon({"t": e["text"], "r": e["route"], "r2": e["requires"]})):
                probs.append("%s: a coordinate-like number in the entry" % eid)
            for lang in ("en", "fa"):
                txt = (e.get("text") or {}).get(lang, "")
                if re.search(r"\b(%s)\(" % "|".join(map(re.escape, tools)), txt):
                    probs.append("%s: text.%s writes a tool as name(...)" % (eid, lang))
                if re.search(r"not here|absent|not in this kitchen|is missing from", txt, re.I):
                    probs.append("%s: text.%s asserts absence" % (eid, lang))
            ten = set(re.findall(r"[A-Za-z]+", e["text"]["en"])) & types
            tfa = set(re.findall(r"[A-Za-z]+", e["text"]["fa"])) & types
            if ten != tfa:
                probs.append("%s: type tokens differ EN %s vs FA %s" % (eid, sorted(ten - tfa), sorted(tfa - ten)))
            distinct = [t for t in tools if "_" in t or t in ("pickup", "heat", "cook", "wash")]
            tools_en = {t for t in distinct if re.search(r"\b%s\b" % re.escape(t), e["text"]["en"])}
            tools_fa = {t for t in distinct if re.search(r"\b%s\b" % re.escape(t), e["text"]["fa"])}
            if tools_en != tools_fa:
                probs.append("%s: tool tokens differ EN %s vs FA %s" % (eid, sorted(tools_en - tools_fa), sorted(tools_fa - tools_en)))
            for c in e.get("confuse", []):
                if c.get("with") not in tools:
                    probs.append("%s: confuse.with %r is not a tool" % (eid, c.get("with")))
            # support must not be a test item
            for s in (e.get("provenance") or {}).get("support", []):
                probs += ["%s: support %s" % (eid, x) for x in _support_problems([s])]
        # coverage: one entry per tool
        ops_with_tool = {e["op"] for e in self.entries if not e.get("composite")}
        for t in tools:
            if t not in ops_with_tool:
                probs.append("no tool entry for %r" % t)
        return probs

    # ------------------------------------------------------------------ route binding
    def slot_names(self, eid):
        return sorted((self.by_id[eid].get("slots") or {}).keys())

    def sample_bindings(self, eid, n=2):
        """Deterministic distinct bindings from the slot samples (first n of the cartesian product)."""
        slots = self.by_id[eid].get("slots") or {}
        names = sorted(slots)
        if not names:
            return [{}]
        prod = itertools.product(*[slots[k]["samples"] for k in names])
        out = []
        for combo in prod:
            out.append(dict(zip(names, combo)))
            if len(out) >= n:
                break
        return out

    def instantiate(self, entry_id: str, binding: dict) -> list:
        e = self.by_id[entry_id]
        slots = set((e.get("slots") or {}).keys())
        extra = set(binding or {}) - slots
        if extra:
            raise ValueError("unknown slot(s) %s for %s" % (sorted(extra), entry_id))
        out = []
        for st in e["route"]:
            d = {}
            for k in ("action", "object", "to", "arg"):
                v = st.get(k)
                if isinstance(v, str) and v.startswith("?"):
                    if v not in (binding or {}):
                        raise ValueError("slot %s of %s is not bound" % (v, entry_id))
                    v = binding[v]
                d[k] = v
            out.append(d)
        return out

    # ------------------------------------------------------------------ matching
    def _members(self, categories):
        return _cat_members(self.categories, categories)

    def _fit(self, e, qtypes: set, cats: dict) -> bool:
        a = e.get("applies_to") or {}
        types, cs = set(a.get("types") or []), a.get("categories") or []
        if not types and not cs:
            return True
        hit = set(types & qtypes)
        for c in cs:
            hit |= set(cats.get(c, ())) & qtypes
        return len(hit) >= int(a.get("min_fit", 1))

    def _trigger_hit(self, e, qops: set) -> bool:
        trig = set(e.get("triggers") or e.get("ops") or [])
        if trig & qops:
            return True
        # a parser verb in the query hits any entry whose triggers contain a tool it maps to
        for q in qops:
            if set(self.verb_ops.get(q, ())) & trig:
                return True
        for t in trig:                      # a tool name in the query hits an entry triggered by its parser verb
            if set(self.verb_ops.get(t, ())) & qops:
                return True
        return False

    def _match(self, e, qops, qtypes, cats, no_ops):
        why = []
        hit = self._trigger_hit(e, qops)
        fit = self._fit(e, qtypes, cats)
        if hit:
            why += ["op:" + o for o in sorted(set(e.get("triggers") or e.get("ops")) & set(self._expand_ops(qops)))]
            if e.get("composite"):
                if fit:
                    return 1.0, why + ["fit"]
                return 0.0, []
            return (1.0, why + ["fit"]) if fit else (0.7, why)
        confused = [c["with"] for c in e.get("confuse", []) if c["with"] in self._expand_ops(qops)]
        if confused and fit:
            return 0.5, ["rel:confuse:" + c for c in sorted(set(confused))]
        if no_ops:
            appl = set(e["keys"].get("types", [])) & set(self.vocab["appliance_types"]) & qtypes
            if appl:
                return 0.5, ["type:" + t for t in sorted(appl)]
        return 0.0, []

    def _expand_ops(self, qops):
        out = set(qops)
        for q in qops:
            out |= set(self.verb_ops.get(q, ()))
        return out

    def candidates(self, query: dict, include_zero: bool = True) -> list:
        qtypes = set(query.get("types") or [])
        qtypes |= set((query.get("need") or {}).get("types") or [])
        roles = ((query.get("parse") or {}).get("roles")) or []
        qops = _query_ops(query.get("ops"), roles)
        no_ops = not qops or bool(query.get("parser_fallback"))
        cats = self._members(None)
        out = []
        recs = {r["id"]: r for r in self.records()}
        for e in self.entries:
            if not e.get("active", True):
                continue
            m, why = self._match(e, qops, qtypes, cats, no_ops)
            if m == 0 and not include_zero:
                continue
            out.append({"rec": recs["procedure/*/" + e["id"].split(":", 1)[1]], "match": m, "why": why})
        out.sort(key=lambda c: (-c["match"], c["rec"]["prio"], c["rec"]["id"]))
        return out

    # ------------------------------------------------------------------ expansion
    def expand(self, ops: list, types: list, categories: dict = None) -> dict:
        cats = self._members(categories)
        qops, qtypes = _query_ops(ops), set(types or [])
        tools, apps, recs, reqs, acts = set(), set(), set(), set(), set()
        for e in self.entries:
            if not e.get("active", True):
                continue
            if not self._trigger_hit(e, qops):
                continue
            if e.get("composite") and not self._fit(e, qtypes, cats):
                continue
            acts.add(e["op"])
            for st in e["route"]:
                acts.add(st["action"])
                for f in ("object", "to"):
                    t = _fixed(st.get(f))
                    if t:
                        self._classify(t, tools, apps, recs)
            for r in e["requires"] + e["realism"]:
                t = r.get("fixture") or r.get("type")
                if t:
                    self._classify(t, tools, apps, recs)
                if r.get("kind") == "executor" and r.get("cond"):
                    reqs.add(r["cond"])
        return {"tools": sorted(tools), "appliances": sorted(apps), "receptacles": sorted(recs),
                "requires": sorted(reqs), "actions": sorted(acts)}

    def _classify(self, t, tools, apps, recs):
        if t in self.vocab["tool_types"]:
            tools.add(t)
        if t in self.vocab["appliance_types"]:
            apps.add(t)
        if t in self.vocab["receptacle_types"]:
            recs.add(t)

    # ------------------------------------------------------------------ affordance map (operation word -> tool/appliance)
    def afford_key(self, word):
        """AFFORD key for an English operation word (inflections stripped); None when the word is unknown."""
        w = re.sub(r"[^a-z ]+", " ", str(word or "").lower()).strip()
        w = re.sub(r"\s+", " ", w)
        if not w:
            return None
        cands = [w, w.rstrip("s")]
        for suf in ("ing", "ed", "es", "d"):
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                st = w[:-len(suf)]
                cands += [st, st + "e"]
                if len(st) > 2 and st[-1] == st[-2]:            # chopping -> chopp -> chop
                    cands.append(st[:-1])
        for cand in cands:
            if cand in self.afford_words:
                return self.afford_words[cand]
        return None

    def afford(self, word):
        """{"key", "tools", "appliances", "recipe", "text"} for an operation word, or None. Pure lookup."""
        k = self.afford_key(word)
        if k is None:
            return None
        a = self.afford_map[k]
        rid = a.get("recipe")
        if rid and (rid not in self.by_id or not self.by_id[rid].get("active", True)):
            rid = None
        return {"key": k, "tools": list(a["tools"]), "appliances": list(a["appliances"]), "recipe": rid,
                "text": a["en"], "request": a.get("request")}

    def afford_records(self):
        """One pkg.rec/1 per AFFORD row, component "procedure" (not part of records(): the retrieval owner opts in)."""
        me = "script:ltm_procedural@" + _file_sha16(__file__)
        out = []
        for k in sorted(self.afford_map):
            a = self.afford_map[k]
            words = sorted(w for w, v in self.afford_words.items() if v == k)
            out.append({
                "schema": SCHEMA_REC, "id": "procedure/*/afford-" + k, "component": "procedure", "kitchen": "*",
                "keys": {"types": sorted(set(a["tools"]) | set(a["appliances"])), "ops": [k] + words,
                         "rels": ["requires:" + t for t in a["tools"]] + ["appliance:" + t for t in a["appliances"]]},
                "lex": {"en": " ".join(words), "fa": ""},
                "text": {"en": a["en"], "fa": a["en"]},      # stored in English only (override)
                "provenance": {"source": "kb:" + self.sha[:16], "evidence": ["afford:" + k], "authored_by": me,
                               # dev items whose cells led to this row's wording: item-level leave-out must drop it (I-close)
                               "support": list(a.get("tuned_on") or [])},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": 30})
            if a.get("request"):                  # the tool-request line (J-P2): same keys, text = fetch the tool and stop
                r2 = json.loads(json.dumps(out[-1]))
                r2["id"] = "procedure/*/afford-" + k + "-request"
                r2["text"] = {"en": a["request"], "fa": a["request"]}
                r2["provenance"]["evidence"] = ["afford:" + k, "tool-request"]
                out.append(r2)
        return out

    # ------------------------------------------------------------------ holding-aware routes
    def holding_route(self, entry_id, binding, holding=None, contract=None, with_tail=False, release_to=None):
        """The entry's route for the hand's actual state. `holding` = held object type or None. Returns (route, notes).
        Rules (HOLDING_RULES, aligned with honest_skills): a pickup of what is already held is dropped; a
        pick_and_place of the held object becomes put_in; something else in the hand is put down first
        (release_to receptacle, else drop); no search_for is added (the skills search by themselves)."""
        base = self.instantiate(entry_id, binding)
        if with_tail and self.by_id[entry_id].get("tail"):
            tb = dict(binding or {})
            e = self.by_id[entry_id]
            for st in e["tail"]:
                d = {}
                for k in ("action", "object", "to", "arg"):
                    v = st.get(k)
                    if isinstance(v, str) and v.startswith("?"):
                        v = tb[v]
                    d[k] = v
                base.append(d)
        return adapt_route(base, holding, contract or load_exec_contract(), release_to)

    def render(self, entry_id, binding, holding=None, lang="en", with_tail=False, contract=None, release_to=None):
        """One-line recipe text from the adapted route (English; FA keeps tool names). Returns (text, notes)."""
        route, notes = self.holding_route(entry_id, binding, holding, contract, with_tail, release_to)
        return route_text(route, holding_note=notes, lang=lang), notes

    # ------------------------------------------------------------------ records
    def records(self) -> list:
        if getattr(self, "_recs", None) is not None and self._recs_key == tuple(e.get("active", True) for e in self.entries):
            return [dict(r) for r in self._recs]
        me = "script:ltm_procedural@" + _file_sha16(__file__)
        out = []
        for e in self.entries:
            if not e.get("active", True):
                continue
            local = e["id"].split(":", 1)[1]
            out.append({
                "schema": SCHEMA_REC, "id": "procedure/*/" + local, "component": "procedure", "kitchen": "*",
                "keys": {"types": list(e["keys"]["types"]), "ops": list(e["keys"]["ops"]), "rels": list(e["keys"]["rels"])},
                "lex": {"en": e["lex"]["en"] + " " + e["op"].replace("_", " "), "fa": e["lex"]["fa"]},
                "text": {"en": e["text"]["en"], "fa": e["text"]["fa"]},
                "provenance": {"source": "kb:" + self.sha[:16], "evidence": [e["id"]], "authored_by": me,
                               "support": list(e["provenance"].get("support", []))},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": e["prio"]})
        self._recs, self._recs_key = out, tuple(e.get("active", True) for e in self.entries)
        return [dict(r) for r in out]


def _support_problems(ids) -> list:
    try:
        m = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    ok = set(m.get("nontest", [])) | set(m.get("practice_pool", [])) | set(m.get("dev_pilot", []))
    test = set(m.get("test", [])) | set(m.get("test_uncapped_not_scored", []))
    out = []
    for s in ids:
        if s in test:
            out.append("%s is a TEST item" % s)
        elif s not in ok:
            out.append("%s is not a non-test item id" % s)
    return out


def load_kb(path=None, verified: bool = True) -> KB:
    p = Path(path) if path else KB_PATH
    doc = json.loads(p.read_text(encoding="utf-8"))
    kb = KB(doc, p)
    if verified:
        rows = []
        if FEAS_PATH.exists():
            for ln in FEAS_PATH.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        pass
        res = {}
        if VERIFIED_PATH.exists():
            try:
                res = json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))
            except ValueError:
                res = {}
        kb.apply_verified(rows, res)
    return kb


def detector_types() -> list:
    return list(json.loads(TYPES_PATH.read_text(encoding="utf-8"))["types"])


# ============================================================================================ skill contracts (J-P2)
def skill_contract(path=None) -> dict:
    """What honest_skills.py REALLY does for the composite tools, read from its source by AST (no import, no simulator).

    -> {"cycle": [(kind, arg), ...], "heat_cook_shared": bool, "pick_and_place": [...], "put_in": [...],
        "slice_needs_tool": bool, "cutting_tools": [type]}. `cycle` is the heat/cook branch of ``_dispatch`` in order: ("ensure_holding", "ot"),
    ("put", "Microwave"), ("close", "Microwave"), ("toggle_on", "Microwave"), ("toggle_off", ...), ("open", ...), ("pickup", "ot").
    heat and cook are ONE branch (``action in ("heat", "cook")``); the fact is read, not assumed."""
    import ast
    p = Path(path or (HERE / "honest_skills.py"))
    tree = ast.parse(p.read_text(encoding="utf-8"))
    disp = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_dispatch":
            disp = n
            break
    if disp is None:
        raise ValueError("honest_skills._dispatch not found")

    def lit(a):
        return a.value if isinstance(a, ast.Constant) else (a.id if isinstance(a, ast.Name) else "?")

    def calls(body):
        out = []
        for st in body:
            if isinstance(st, (ast.Expr, ast.Return)) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Attribute):
                f, args = st.value.func.attr, [lit(a) for a in st.value.args]
                if f == "_interact":
                    out.append((args[0], args[1] if len(args) > 1 else None))
                elif f == "_put":
                    out.append(("put", args[0] if args else None))
                elif f == "_ensure_holding":
                    out.append(("ensure_holding", args[0] if args else None))
        return out

    def consts(t):
        return {c.value for c in ast.walk(t) if isinstance(c, ast.Constant) and isinstance(c.value, str)}

    def branch(pred):
        for n in ast.walk(disp):
            if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and pred(consts(n.test)):
                return n
        return None
    hc = branch(lambda c: {"heat", "cook"} <= c)
    pp = branch(lambda c: c == {"pick_and_place"})
    pi = branch(lambda c: c == {"put_in"})
    if hc is None or pp is None or pi is None:
        raise ValueError("honest_skills._dispatch: heat/cook, pick_and_place or put_in branch not found")
    # A26 item 2 (K-B2b): the slice branch refuses without a cutting tool, and the module's CUTTING_TOOLS set (read, not assumed)
    sl = branch(lambda c: c == {"slice"})
    needs_tool = sl is not None and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                        and n.func.attr == "_cutting_tool_check" for n in ast.walk(sl))
    cutting = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUTTING_TOOLS" for t in n.targets):
            cutting = sorted(c.value for c in ast.walk(n.value) if isinstance(c, ast.Constant) and isinstance(c.value, str))
    return {"cycle": calls(hc.body), "heat_cook_shared": consts(hc.test) == {"heat", "cook"},
            "pick_and_place": calls(pp.body), "put_in": calls(pi.body), "slice_needs_tool": needs_tool,
            "cutting_tools": cutting,
            "origin": "ast:honest_skills@" + _file_sha16(p)}


def check_skill_truth(kb, sc: dict = None) -> list:
    """Every claim a procedure line makes about heat / cook / pick_and_place / put_in / the stay-inside route, checked against the
    skill code. -> [problem]; [] = every checked line is TRUE."""
    sc = sc or skill_contract()
    probs = []
    cyc = sc["cycle"]
    want = [("ensure_holding", "ot"), ("put", "Microwave"), ("close", "Microwave"), ("toggle_on", "Microwave"),
            ("toggle_off", "Microwave"), ("open", "Microwave"), ("pickup", "ot")]
    if cyc != want:
        probs.append("the heat/cook cycle in honest_skills is %s, not the one the KB describes" % (cyc,))
    if not sc["heat_cook_shared"]:
        probs.append("heat and cook are no longer one branch in honest_skills")
    names = [a for a, _ in cyc]
    if "close" in names and "toggle_on" in names and names.index("close") > names.index("toggle_on"):
        probs.append("toggle_on precedes close in the cycle")
    ends_in_hand = bool(cyc) and cyc[-1] == ("pickup", "ot")
    for t in ("heat", "cook"):
        e = kb.by_id.get("proc:tool-" + t)
        if e is None:
            probs.append("no tool entry for %s" % t)
            continue
        if (e["hand_end"] == "held") != ends_in_hand:
            probs.append("proc:tool-%s hand_end %r contradicts the cycle (food back in hand: %s)" % (t, e["hand_end"], ends_in_hand))
        if [s["action"] for s in e["route"]] != [t]:
            probs.append("proc:tool-%s route is not the single %s step" % (t, t))
        txt = e["text"]["en"]
        if "door closed" not in txt and "put in, close" not in txt:
            probs.append("proc:tool-%s text does not state the door is closed" % t)
        if "does not stay inside" not in txt and "does not leave X in the microwave" not in txt:
            probs.append("proc:tool-%s text does not say the food does not stay in the microwave" % t)
        if "microwave" not in txt.lower():
            probs.append("proc:tool-%s text does not name the microwave" % t)
    if sc["pick_and_place"] != [("ensure_holding", "ot"), ("put", "to")]:
        probs.append("pick_and_place in honest_skills is %s" % (sc["pick_and_place"],))
    if sc["put_in"] != [("put", "ot")]:
        probs.append("put_in in honest_skills is %s" % (sc["put_in"],))
    if not sc["slice_needs_tool"]:
        probs.append("honest_skills._dispatch no longer refuses slice without a cutting tool: the 'ButterKnife in hand' line is stale")
    if sc["cutting_tools"] != ["ButterKnife"]:
        probs.append("honest_skills.CUTTING_TOOLS is %r, not ['ButterKnife'] as the slice lines say" % (sc["cutting_tools"],))
    for eid in ("proc:tool-slice", "proc:slice-with-butterknife"):
        e = kb.by_id.get(eid)
        if e is None or "ButterKnife" not in e["text"]["en"]:
            probs.append("%s does not state that the ButterKnife is needed in hand" % eid)
    stay = kb.by_id.get("proc:microwave-run")
    if stay is None:
        probs.append("no stay-inside route proc:microwave-run")
        return probs
    prefix = list(cyc[1:4])                                     # put, close, toggle_on: the stay-inside route is the cycle's first half
    route = [("put" if st["action"] in ("pick_and_place", "put_in") else st["action"],
              st["to"] if st["action"] == "pick_and_place" else st["object"]) for st in stay["route"]]
    if route != prefix:
        probs.append("stay-inside route %s is not the cycle's first half %s" % (route, prefix))
    tail = [(st["action"], st["object"]) for st in stay.get("tail") or []]
    if tail != [("toggle_off", "Microwave"), ("open", "Microwave"), ("pickup", "?item")]:
        probs.append("stay-inside tail %s is not the cycle's second half" % (tail,))
    if stay["hand_end"] != "empty":
        probs.append("stay-inside route must end with an empty hand (hand_end=%r)" % stay["hand_end"])
    goal = {(g["type"], g["key"]): g for g in stay.get("goal") or []}
    if goal.get(("Microwave", "isToggled"), {}).get("value") is not True or \
            goal.get(("Microwave", "isOpen"), {}).get("value") is not False or \
            goal.get(("?item", "parentReceptacles"), {}).get("contains") != "Microwave":
        probs.append("stay-inside goal predicate must be: item in Microwave, Microwave closed, Microwave on")
    if {"heat", "cook"} & set(stay.get("triggers") or []):
        probs.append("the stay-inside route must not be triggered by heat/cook")
    txt = stay["text"]["en"]
    if "stays inside" not in txt or "do not take it out unless" not in txt:
        probs.append("stay-inside text must say the food stays inside and is not taken out unless asked")
    return probs


#: effect classes of a plan action (J-P2, Astra: check effects, not words). toggle_on of the Microwave heats what is inside it.
_EFFECT_OF = {"slice": "slice", "heat": "heat", "cook": "heat", "wash": "wash", "break": "break", "fill": "fill", "empty": "empty",
              "clean": "wash"}
#: request op -> the effect class it authorises (parser verbs and tool names)
_REQUESTED_AS = {"slice": "slice", "cut": "slice", "chop": "slice", "heat": "heat", "cook": "heat", "warm": "heat", "bake": "heat",
                 "microwave": "heat", "wash": "wash", "clean": "wash", "rinse": "wash", "scrub": "wash", "break": "break",
                 "fill": "fill", "empty": "empty"}
#: actions a pure tool request may contain (getting to the tool)
_ACQUIRE = {"pickup", "go_to", "search_for", "open", "close", "close_all", "look_around", "scan_for"}


def unrequested_actions(plan, ops, types=(), tool_request: bool = False) -> list:
    """Violations: effects the plan produces that the request does not ask for. `ops` are the request's operation words (tool names
    or parser verbs), `types` its object types. A tool request ("something to cut with") authorises NO effect: only the acquisition
    steps (pickup, walking, opening). An unrequested slice is a violation like any other.
    -> [{"effect", "action", "why"}]. A toggle_on of the Microwave counts as the effect 'heat'."""
    allowed = set()
    if not tool_request:
        for o in ops or []:
            o = str(o).replace(":", "_")
            if o in _REQUESTED_AS:
                allowed.add(_REQUESTED_AS[o])
            if o in ("toggle", "toggle_on") and "Microwave" in set(types or ()):
                allowed.add("heat")
    out = []
    for st in plan or []:
        a = st.get("action")
        eff = _EFFECT_OF.get(a) or ("heat" if a == "toggle_on" and st.get("object") == "Microwave" else None)
        if eff and eff not in allowed:
            out.append({"effect": eff, "action": a, "why": "tool_request_asks_for_the_tool_only" if tool_request else "unrequested_effect"})
        elif tool_request and not eff and a not in _ACQUIRE:
            out.append({"effect": "other", "action": a, "why": "tool_request_asks_for_the_tool_only"})
    return out


# ============================================================================================ early check (diagnostic)
def early_check(lang_list=("en", "fa"), validator=None) -> dict:
    """B4 early check: check_static == [] and, over the non-test instructions, the share where `expand` adds a
    tool/appliance type the reference plan uses. Diagnostic only (D9); reads non-test items only."""
    sys.path.insert(0, str(HERE))
    import ltm_concepts as lc
    kb = load_kb(verified=False)
    types, contract = detector_types(), load_exec_contract()
    problems = kb.check_static(contract, contract_tools(contract), types, validator)
    m = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    gen = {i["id"]: i for i in json.loads((HERE / "dataset" / "general.json").read_text(encoding="utf-8"))["items"]}
    test = set(m["test"]) | set(m.get("test_uncapped_not_scored", []))
    ids = [i for i in m["nontest"] if i in gen and i not in test]
    rep = {"n_items": len(ids), "check_static": problems, "validator": kb.last_validator,
           "contract_origin": contract.get("_origin")}
    for lang in lang_list:
        added_any = used = parsed = 0
        for iid in ids:
            it = gen[iid]
            p = lc.parse(it[lang])
            roles = p["roles"]
            qtypes = sorted({r[k] for r in roles for k in ("obj", "src", "dest") if r[k] not in ("-", "?", "PRON")})
            ops = sorted({r["verb"] for r in roles if r["verb"] != "?"})
            parsed += bool(p["ok"])
            ex = kb.expand(ops + [("toggle_on" if r.get("req") == "on" else "toggle_off") for r in roles if r["verb"] == "toggle"],
                           qtypes)
            adds = (set(ex["tools"]) | set(ex["appliances"])) - set(qtypes)
            plan_types = {t for st in it["plan"] for t in (st.get("object"), st.get("to")) if t}
            for st in it["plan"]:       # fixtures the executor itself uses for a plan action (heat -> Microwave ...)
                ent = kb.by_id.get("proc:tool-" + st["action"].replace("_", "-"))
                for rq in (ent or {}).get("requires", []):
                    if rq.get("fixture"):
                        plan_types.add(rq["fixture"])
            added_any += bool(adds)
            used += bool(adds & plan_types)
        rep[lang] = {"parsed_ok": parsed, "expand_adds_something": added_any, "adds_a_type_the_plan_uses": used,
                     "share_of_items": round(used / max(1, len(ids)), 3)}
    return rep


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="B4 procedural knowledge base")
    ap.add_argument("--build", action="store_true", help="write sim/step5/procedures_kb.json from definitions()")
    a = ap.parse_args(argv)
    if a.build:
        doc = build_kb()
        KB_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("wrote", KB_PATH, "entries", len(doc["entries"]), "sha", doc["sha256"][:16])
    # --check / --early: python sim/planners/procedures_check.py [--early] (it injects the planner validator)
    return 0


if __name__ == "__main__":
    sys.exit(main())
