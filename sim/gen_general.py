"""Seed the GENERAL bilingual instruction set -> sim/dataset/general.json  (see sim/general_set.py).

>= 100 L1 atomic, >= 100 L2 sequential, >= 50 L3 abstract-goal instructions, each in English and Persian, none
tied to one kitchen. The vocabulary is grounded in `dataset/scene_inventory.json` (a live scan of all 30 iTHOR
kitchens): 35 object types exist in every kitchen and the set is built almost entirely from those, plus three
common ones (Drawer 28/30, DiningTable 16/30, Kettle 15/30). Where a kitchen lacks the object, the harness
reports the record as skipped -- "object does not exist in this scene" -- rather than failing the model.

This file is a SEED, not the source of truth: the console edits general.json directly. So a re-run only ADDS
ids that are missing -- it never overwrites an edited instruction and never resurrects a deleted one.
`--reset` discards the file and regenerates everything.

Run (pure stdlib, from the repo root):   python sim/gen_general.py [--reset]
"""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import general_set as gs  # noqa: E402

S = gs.parse_step
G = gs.parse_goal

# ---------------------------------------------------------------------------------------------------------------
# VOCABULARY  objectType -> (english noun, persian noun)
# ---------------------------------------------------------------------------------------------------------------
N = {
    "Apple": ("apple", "سیب"), "Bread": ("bread", "نان"), "Tomato": ("tomato", "گوجه‌فرنگی"),
    "Potato": ("potato", "سیب‌زمینی"), "Egg": ("egg", "تخم‌مرغ"), "Lettuce": ("lettuce", "کاهو"),
    "Mug": ("mug", "لیوان"), "Cup": ("cup", "فنجان"), "Bowl": ("bowl", "کاسه"), "Plate": ("plate", "بشقاب"),
    "Pan": ("pan", "تابه"), "Pot": ("pot", "قابلمه"), "Kettle": ("kettle", "کتری"),
    "Knife": ("knife", "چاقو"), "ButterKnife": ("butter knife", "کارد کره‌خوری"), "Fork": ("fork", "چنگال"),
    "Spoon": ("spoon", "قاشق"), "Spatula": ("spatula", "کفگیر"), "DishSponge": ("sponge", "اسفنج ظرف‌شویی"),
    "SoapBottle": ("soap bottle", "بطری صابون"), "SaltShaker": ("salt shaker", "نمکدان"),
    "PepperShaker": ("pepper shaker", "فلفل‌پاش"),
    "Fridge": ("fridge", "یخچال"), "Microwave": ("microwave", "مایکروویو"), "Toaster": ("toaster", "توستر"),
    "CoffeeMachine": ("coffee machine", "قهوه‌ساز"), "Faucet": ("faucet", "شیر آب"),
    "StoveKnob": ("stove", "اجاق گاز"), "StoveBurner": ("stove", "اجاق گاز"),
    "Cabinet": ("cabinet", "کابینت"), "Drawer": ("drawer", "کشو"), "GarbageCan": ("trash can", "سطل زباله"),
    "SinkBasin": ("sink", "سینک"), "CounterTop": ("counter", "کانتر"), "LightSwitch": ("light", "چراغ"),
    "DiningTable": ("dining table", "میز غذاخوری"),
}
en = lambda t: N[t][0]
fa = lambda t: N[t][1]

# Where something ends up: objectType -> (english phrase, persian phrase)
AT = {
    "SinkBasin": ("in the sink", "داخل سینک"), "CounterTop": ("on the counter", "روی کانتر"),
    "GarbageCan": ("in the trash can", "داخل سطل زباله"), "Fridge": ("in the fridge", "داخل یخچال"),
    "Microwave": ("in the microwave", "داخل مایکروویو"), "Plate": ("on the plate", "روی بشقاب"),
    "Bowl": ("in the bowl", "داخل کاسه"), "Pan": ("in the pan", "داخل تابه"), "Pot": ("in the pot", "داخل قابلمه"),
    "Cabinet": ("in a cabinet", "داخل یکی از کابینت‌ها"), "Drawer": ("in a drawer", "داخل یکی از کشوها"),
    "StoveBurner": ("on the stove", "روی اجاق گاز"), "CoffeeMachine": ("in the coffee machine", "داخل قهوه‌ساز"),
    "DiningTable": ("on the dining table", "روی میز غذاخوری"),
}
LIQ = {"water": "آب", "coffee": "قهوه", "wine": "شراب"}

PICKUPABLE = ["Apple", "Bread", "Tomato", "Potato", "Egg", "Lettuce", "Mug", "Cup", "Bowl", "Plate", "Pan", "Pot",
              "Knife", "ButterKnife", "Fork", "Spoon", "Spatula", "DishSponge", "SoapBottle", "SaltShaker",
              "PepperShaker", "Kettle"]
DIRTYABLE = ["Bowl", "Cup", "Mug", "Pan", "Plate", "Pot"]
MULTI = {"Cabinet", "Drawer", "StoveKnob", "StoveBurner"}      # several instances per kitchen -> quantified goals


def slug(t):
    return "".join(("-" + c.lower()) if c.isupper() and i else c.lower() for i, c in enumerate(t))


def _any(goal_line):
    """Slicing or breaking REPLACES the object, and leaves a second object of the ORIGINAL type behind: after
    `slice Bread` a kitchen holds the old Bread (isSliced = true) AND a new end-piece whose type is also Bread
    (isSliced = false), a centimetre apart. An un-quantified goal reads whichever is nearest -- a coin flip
    that flips again when the agent walks away. "Some Bread is sliced" is what the instruction means."""
    if goal_line.endswith((".isSliced = true", ".isBroken = true")) and not goal_line.startswith(("any ", "all ")):
        return "any " + goal_line
    return goal_line


def item(level, stem, en_text, fa_text, plan, goal, setup=()):
    return {"id": f"g-{level.lower()}-{stem}", "level": level, "en": en_text, "fa": fa_text,
            "setup": [S(s) for s in setup], "plan": [S(s) for s in plan], "goal": [G(_any(g)) for g in goal]}


def q(t, word):
    """`any`/`all` prefix for a goal on a multi-instance type, nothing for a singleton."""
    return f"{word} " if t in MULTI else ""


# ---------------------------------------------------------------------------------------------------------------
# L1 -- one command -> one function
# ---------------------------------------------------------------------------------------------------------------
def build_l1():
    out = []
    A = lambda *a, **k: out.append(item("L1", *a, **k))

    for t in PICKUPABLE:
        A(f"pickup-{slug(t)}", f"Pick up the {en(t)}.", f"{fa(t)} را بردار.", [f"pickup {t}"], [f"holding {t}"])

    OPEN = {"Fridge": ("Open the fridge.", "در یخچال را باز کن.", "Close the fridge.", "در یخچال را ببند."),
            "Microwave": ("Open the microwave.", "در مایکروویو را باز کن.",
                          "Close the microwave.", "در مایکروویو را ببند."),
            "Cabinet": ("Open a cabinet.", "یکی از کابینت‌ها را باز کن.",
                        "A cabinet has been left open. Close it.", "یکی از کابینت‌ها باز مانده است؛ آن را ببند."),
            "Drawer": ("Open a drawer.", "یکی از کشوها را باز کن.",
                       "A drawer has been left open. Close it.", "یکی از کشوها باز مانده است؛ آن را ببند.")}
    for t, (oe, of, ce, cf) in OPEN.items():
        A(f"open-{slug(t)}", oe, of, [f"open {t}"], [f"{q(t, 'any')}{t}.isOpen = true"])
        A(f"close-{slug(t)}", ce, cf, [f"close {t}"], [f"{q(t, 'all')}{t}.isOpen = false"], setup=[f"open {t}"])

    ON = {"Microwave": None, "Toaster": None, "CoffeeMachine": None, "StoveKnob": None,
          "Faucet": ("Turn on the faucet.", "شیر آب را باز کن.", "Turn off the faucet.", "شیر آب را ببند.")}
    for t, special in ON.items():
        one, onf, offe, offf = special or (f"Turn on the {en(t)}.", f"{fa(t)} را روشن کن.",
                                           f"Turn off the {en(t)}.", f"{fa(t)} را خاموش کن.")
        A(f"on-{slug(t)}", one, onf, [f"toggle_on {t}"], [f"{q(t, 'any')}{t}.isToggled = true"])
        A(f"off-{slug(t)}", offe, offf, [f"toggle_off {t}"], [f"{q(t, 'all')}{t}.isToggled = false"],
          setup=[f"toggle_on {t}"])
    # The room light is the one device that starts ON in every kitchen, so here it is "on" that needs the setup.
    A("off-light", "Turn off the light.", "چراغ را خاموش کن.", ["toggle_off LightSwitch"],
      ["LightSwitch.isToggled = false"])
    A("on-light", "Turn on the light.", "چراغ را روشن کن.", ["toggle_on LightSwitch"],
      ["LightSwitch.isToggled = true"], setup=["toggle_off LightSwitch"])

    GO = [("Fridge", "fridge", "یخچال"), ("Microwave", "microwave", "مایکروویو"), ("Toaster", "toaster", "توستر"),
          ("CoffeeMachine", "coffee machine", "قهوه‌ساز"), ("SinkBasin", "sink", "سینک"),
          ("GarbageCan", "trash can", "سطل زباله"), ("LightSwitch", "light switch", "کلید چراغ"),
          ("StoveBurner", "stove", "اجاق گاز"), ("DiningTable", "dining table", "میز غذاخوری")]
    for t, e, f in GO:
        # "within reach" (AI2-THOR's 1.5 m interaction range), not `visible`: from the spawn point the fridge is
        # already visible in many kitchens, and a goal that is true before the agent moves scores nothing.
        A(f"goto-{slug(t)}", f"Go to the {e}.", f"به سمت {f} برو.", [f"go_to {t}"],
          [f"{q(t, 'any')}{t}.distance <= 1.5"])

    for t in ("Egg", "Mug", "Plate"):
        A(f"break-{slug(t)}", f"Break the {en(t)}.", f"{fa(t)} را بشکن.", [f"break {t}"], [f"{t}.isBroken = true"])

    for t in DIRTYABLE:
        A(f"dirty-{slug(t)}", f"Make the {en(t)} dirty.", f"{fa(t)} را کثیف کن.", [f"dirty {t}"],
          [f"{t}.isDirty = true"])
        A(f"clean-{slug(t)}", f"The {en(t)} is dirty. Clean it.", f"{fa(t)} کثیف است؛ تمیزش کن.", [f"clean {t}"],
          [f"{t}.isDirty = false"], setup=[f"dirty {t}"])

    for t in ("Bowl", "Cup", "Mug", "Pot"):
        for liq, lfa in LIQ.items():
            A(f"fill-{slug(t)}-{liq}", f"Fill the {en(t)} with {liq}.", f"{fa(t)} را با {lfa} پر کن.",
              [f"fill {t} = {liq}"], [f"{t}.isFilledWithLiquid = true", f"{t}.fillLiquid = {liq}"])
    A("fill-kettle-water", "Fill the kettle with water.", "کتری را با آب پر کن.", ["fill Kettle = water"],
      ["Kettle.isFilledWithLiquid = true", "Kettle.fillLiquid = water"])
    for t in ("Bowl", "Cup", "Mug", "Pot", "Kettle"):
        A(f"empty-{slug(t)}", f"Empty the {en(t)}.", f"{fa(t)} را خالی کن.", [f"empty {t}"],
          [f"{t}.isFilledWithLiquid = false"], setup=[f"fill {t} = water"])

    for t in ("Apple", "Bread", "Potato", "Fork", "Spoon", "Spatula", "DishSponge", "SoapBottle"):
        A(f"drop-{slug(t)}", f"Drop the {en(t)}.", f"{fa(t)} را رها کن.", ["drop"], [f"{t}.isPickedUp = false"],
          setup=[f"pickup {t}"])

    # With the knife already in hand, slicing IS one function (the two-step version is L2).
    for t in ("Apple", "Bread", "Tomato", "Potato", "Lettuce"):
        A(f"slice-{slug(t)}", f"Slice the {en(t)}.", f"{fa(t)} را برش بزن.", [f"slice {t}"],
          [f"{t}.isSliced = true"], setup=["pickup Knife"])

    HELD = [("Apple", "SinkBasin"), ("Lettuce", "SinkBasin"), ("Fork", "SinkBasin"), ("Spoon", "SinkBasin"),
            ("Cup", "SinkBasin"), ("Tomato", "CounterTop"), ("Bread", "CounterTop"), ("Mug", "CounterTop"),
            ("Plate", "CounterTop"), ("Bowl", "CounterTop"), ("Egg", "GarbageCan"), ("Potato", "GarbageCan")]
    for t, r in HELD:
        A(f"place-{slug(t)}-{slug(r)}", f"You are holding the {en(t)}. Put it {AT[r][0]}.",
          f"{fa(t)} در دست توست؛ آن را {AT[r][1]} بگذار.", [f"put_in {r}"], [f"{t} in {r}"],
          setup=[f"pickup {t}"])

    A("crouch", "Crouch down.", "چمباتمه بزن.", ["crouch"], ["Agent.isStanding = false"])
    A("stand", "Stand up.", "بلند شو و بایست.", ["stand"], ["Agent.isStanding = true"], setup=["crouch"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# L2 -- an explicit, ordered chain of functions
# ---------------------------------------------------------------------------------------------------------------
def build_l2():
    out = []
    A = lambda *a, **k: out.append(item("L2", *a, **k))

    for t in ("Apple", "Bread", "Tomato", "Potato", "Egg", "Lettuce"):
        A(f"fridge-{slug(t)}", f"Put the {en(t)} in the fridge and close the door.",
          f"{fa(t)} را داخل یخچال بگذار و درش را ببند.",
          [f"pick_and_place {t} -> Fridge", "close Fridge"], [f"{t} in Fridge", "Fridge.isOpen = false"])
    for t in ("Potato", "Bread", "Apple", "Egg", "Tomato", "Mug"):
        A(f"microwave-{slug(t)}", f"Put the {en(t)} in the microwave and close it.",
          f"{fa(t)} را داخل مایکروویو بگذار و درش را ببند.",
          [f"pick_and_place {t} -> Microwave", "close Microwave"], [f"{t} in Microwave", "Microwave.isOpen = false"])
    for t in ("Potato", "Bread", "Egg"):
        A(f"microwave-run-{slug(t)}", f"Put the {en(t)} in the microwave, close the door, and turn it on.",
          f"{fa(t)} را داخل مایکروویو بگذار، درش را ببند و روشنش کن.",
          [f"pick_and_place {t} -> Microwave", "close Microwave", "toggle_on Microwave"],
          [f"{t} in Microwave", "Microwave.isOpen = false", "Microwave.isToggled = true"])

    for t in ("Apple", "Bread", "Tomato", "Potato", "Lettuce"):
        A(f"slice-{slug(t)}", f"Pick up the knife and slice the {en(t)}.", f"چاقو را بردار و {fa(t)} را برش بزن.",
          ["pickup Knife", f"slice {t}"], [f"{t}.isSliced = true"])
    for a, b in (("Tomato", "Lettuce"), ("Bread", "Tomato"), ("Apple", "Potato"), ("Potato", "Tomato")):
        A(f"slice-{slug(a)}-{slug(b)}", f"Take the knife, slice the {en(a)}, then slice the {en(b)}.",
          f"چاقو را بردار، اول {fa(a)} و بعد {fa(b)} را برش بزن.",
          ["pickup Knife", f"slice {a}", f"slice {b}"], [f"{a}.isSliced = true", f"{b}.isSliced = true"])

    for t, r in (("Potato", "Plate"), ("Bread", "CounterTop"), ("Apple", "Plate"), ("Tomato", "Plate"), ("Egg", "Bowl"),
                 ("Potato", "CounterTop")):
        A(f"heat-{slug(t)}-{slug(r)}", f"Heat the {en(t)} in the microwave, then put it {AT[r][0]}.",
          f"{fa(t)} را در مایکروویو گرم کن، سپس {AT[r][1]} بگذار.",
          [f"heat {t}", f"pick_and_place {t} -> {r}"], [f"{t}.temperature = Hot", f"{t} in {r}"])
    for t, r in (("Bread", "CounterTop"), ("Apple", "Plate"), ("Tomato", "Plate"), ("Potato", "CounterTop"),
                 ("Lettuce", "CounterTop"), ("Egg", "Bowl")):
        A(f"cool-{slug(t)}-{slug(r)}", f"Chill the {en(t)} in the fridge, then put it {AT[r][0]}.",
          f"{fa(t)} را در یخچال خنک کن، سپس {AT[r][1]} بگذار.",
          [f"cool {t}", f"pick_and_place {t} -> {r}"], [f"{t}.temperature = Cold", f"{t} in {r}"])
    for r in ("Plate", "Bowl", "Pan", "CounterTop"):
        A(f"cook-potato-{slug(r)}", f"Cook the potato and put it {AT[r][0]}.",
          f"سیب‌زمینی را بپز و {AT[r][1]} بگذار.",
          ["cook Potato", f"pick_and_place Potato -> {r}"], ["Potato.isCooked = true", f"Potato in {r}"])

    for t in DIRTYABLE:
        A(f"wash-{slug(t)}", f"The {en(t)} is dirty. Wash it and put it on the counter.",
          f"{fa(t)} کثیف است؛ آن را بشوی و روی کانتر بگذار.",
          [f"wash {t}", f"pick_and_place {t} -> CounterTop"], [f"{t}.isDirty = false", f"{t} in CounterTop"],
          setup=[f"dirty {t}"])

    for t in ("SoapBottle", "SaltShaker", "PepperShaker", "Plate", "Bowl"):
        A(f"cabinet-{slug(t)}", f"Put the {en(t)} away in a cabinet and close it.",
          f"{fa(t)} را داخل یکی از کابینت‌ها بگذار و درش را ببند.",
          [f"pick_and_place {t} -> Cabinet", "close Cabinet"], [f"{t} in Cabinet", "all Cabinet.isOpen = false"])
    for t in ("Fork", "Spoon", "Knife", "ButterKnife", "Spatula"):
        A(f"drawer-{slug(t)}", f"Put the {en(t)} in a drawer and close it.",
          f"{fa(t)} را داخل یکی از کشوها بگذار و آن را ببند.",
          [f"pick_and_place {t} -> Drawer", "close Drawer"], [f"{t} in Drawer", "all Drawer.isOpen = false"])

    for t in ("Egg", "Knife", "Mug", "Bowl", "SoapBottle", "Spoon", "Lettuce", "Pan"):
        A(f"find-pickup-{slug(t)}", f"Find the {en(t)} and pick it up.", f"{fa(t)} را پیدا کن و بردار.",
          [f"search_for {t}", f"pickup {t}"], [f"holding {t}"])
    for t, r in (("Fork", "SinkBasin"), ("Cup", "SinkBasin"), ("DishSponge", "SinkBasin"), ("Bread", "Fridge"),
                 ("Spatula", "SinkBasin")):
        A(f"find-place-{slug(t)}-{slug(r)}", f"Find the {en(t)} and put it {AT[r][0]}.",
          f"{fa(t)} را پیدا کن و {AT[r][1]} بگذار.",
          [f"search_for {t}", f"pick_and_place {t} -> {r}"], [f"{t} in {r}"])

    for a, b, r in (("Fork", "Spoon", "SinkBasin"), ("Cup", "Bowl", "SinkBasin"), ("Plate", "Mug", "SinkBasin"),
                    ("Egg", "Tomato", "GarbageCan"), ("Apple", "Potato", "GarbageCan"),
                    ("Apple", "Lettuce", "Fridge"), ("Tomato", "Egg", "Fridge")):
        A(f"two-{slug(a)}-{slug(b)}-{slug(r)}", f"First put the {en(a)} {AT[r][0]}, then the {en(b)}.",
          f"اول {fa(a)} و بعد {fa(b)} را {AT[r][1]} بگذار.",
          [f"pick_and_place {a} -> {r}", f"pick_and_place {b} -> {r}"], [f"{a} in {r}", f"{b} in {r}"])

    A("coffee-machine", "Put the mug in the coffee machine and turn it on.",
      "لیوان را داخل قهوه‌ساز بگذار و آن را روشن کن.",
      ["pick_and_place Mug -> CoffeeMachine", "toggle_on CoffeeMachine"],
      ["Mug in CoffeeMachine", "CoffeeMachine.isToggled = true"])
    for t in ("Pan", "Pot"):
        A(f"stove-{slug(t)}", f"Put the {en(t)} on the stove and turn the stove on.",
          f"{fa(t)} را روی اجاق گاز بگذار و اجاق را روشن کن.",
          [f"pick_and_place {t} -> StoveBurner", "toggle_on StoveKnob"],
          [f"{t} in StoveBurner", "any StoveKnob.isToggled = true"])
    for t in ("Pot", "Kettle"):
        A(f"stove-water-{slug(t)}", f"Fill the {en(t)} with water, put it on the stove, and turn the stove on.",
          f"{fa(t)} را با آب پر کن، روی اجاق گاز بگذار و اجاق را روشن کن.",
          [f"fill {t} = water", f"pick_and_place {t} -> StoveBurner", "toggle_on StoveKnob"],
          [f"{t}.isFilledWithLiquid = true", f"{t} in StoveBurner", "any StoveKnob.isToggled = true"])

    for t, liq, r in (("Cup", "water", "CounterTop"), ("Mug", "coffee", "CounterTop"), ("Bowl", "water", "SinkBasin"),
                      ("Cup", "wine", "DiningTable"), ("Mug", "water", "Microwave"), ("Cup", "water", "Fridge")):
        A(f"fill-place-{slug(t)}-{liq}-{slug(r)}", f"Fill the {en(t)} with {liq} and put it {AT[r][0]}.",
          f"{fa(t)} را با {LIQ[liq]} پر کن و {AT[r][1]} بگذار.",
          [f"fill {t} = {liq}", f"pick_and_place {t} -> {r}"], [f"{t}.fillLiquid = {liq}", f"{t} in {r}"])

    # -- two and three unrelated atomics, where ORDER is the whole content of the instruction ------------------
    CHAINS = [
        ("light-off-open-fridge", "Turn off the light, then open the fridge.",
         "چراغ را خاموش کن، سپس در یخچال را باز کن.",
         ["toggle_off LightSwitch", "open Fridge"], ["LightSwitch.isToggled = false", "Fridge.isOpen = true"]),
        ("open-microwave-toaster-on", "Open the microwave, then turn on the toaster.",
         "در مایکروویو را باز کن، سپس توستر را روشن کن.",
         ["open Microwave", "toggle_on Toaster"], ["Microwave.isOpen = true", "Toaster.isToggled = true"]),
        ("coffee-on-pickup-mug", "Turn on the coffee machine, then pick up the mug.",
         "قهوه‌ساز را روشن کن، سپس لیوان را بردار.",
         ["toggle_on CoffeeMachine", "pickup Mug"], ["CoffeeMachine.isToggled = true", "holding Mug"]),
        ("faucet-on-pickup-sponge", "Turn on the faucet, then pick up the sponge.",
         "شیر آب را باز کن، سپس اسفنج ظرف‌شویی را بردار.",
         ["toggle_on Faucet", "pickup DishSponge"], ["Faucet.isToggled = true", "holding DishSponge"]),
        ("open-fridge-pickup-apple", "Open the fridge, then pick up the apple.",
         "در یخچال را باز کن، سپس سیب را بردار.",
         ["open Fridge", "pickup Apple"], ["Fridge.isOpen = true", "holding Apple"]),
        ("toaster-then-coffee", "Turn on the toaster and then the coffee machine.",
         "اول توستر و بعد قهوه‌ساز را روشن کن.",
         ["toggle_on Toaster", "toggle_on CoffeeMachine"], ["Toaster.isToggled = true", "CoffeeMachine.isToggled = true"]),
        ("open-fridge-microwave", "Open the fridge and then the microwave.",
         "اول در یخچال و بعد در مایکروویو را باز کن.",
         ["open Fridge", "open Microwave"], ["Fridge.isOpen = true", "Microwave.isOpen = true"]),
        ("stove-then-microwave", "Turn on the stove, then turn on the microwave.",
         "اجاق گاز را روشن کن، سپس مایکروویو را روشن کن.",
         ["toggle_on StoveKnob", "toggle_on Microwave"], ["any StoveKnob.isToggled = true", "Microwave.isToggled = true"]),
        ("salt-then-light-off", "Pick up the salt shaker, then turn off the light.",
         "نمکدان را بردار، سپس چراغ را خاموش کن.",
         ["pickup SaltShaker", "toggle_off LightSwitch"], ["holding SaltShaker", "LightSwitch.isToggled = false"]),
        ("open-cabinet-drawer", "Open a cabinet, then open a drawer.",
         "اول یکی از کابینت‌ها و بعد یکی از کشوها را باز کن.",
         ["open Cabinet", "open Drawer"], ["any Cabinet.isOpen = true", "any Drawer.isOpen = true"]),
        ("faucet-on-light-off", "Turn on the faucet, then turn off the light.",
         "شیر آب را باز کن، سپس چراغ را خاموش کن.",
         ["toggle_on Faucet", "toggle_off LightSwitch"], ["Faucet.isToggled = true", "LightSwitch.isToggled = false"]),
        ("break-egg-pickup-fork", "Break the egg, then pick up the fork.",
         "تخم‌مرغ را بشکن، سپس چنگال را بردار.",
         ["break Egg", "pickup Fork"], ["Egg.isBroken = true", "holding Fork"]),
        ("dirty-pan-faucet-on", "Make the pan dirty, then turn on the faucet.",
         "تابه را کثیف کن، سپس شیر آب را باز کن.",
         ["dirty Pan", "toggle_on Faucet"], ["Pan.isDirty = true", "Faucet.isToggled = true"]),
        ("open-microwave-pickup-potato", "Open the microwave, then pick up the potato.",
         "در مایکروویو را باز کن، سپس سیب‌زمینی را بردار.",
         ["open Microwave", "pickup Potato"], ["Microwave.isOpen = true", "holding Potato"]),
        ("toaster-on-open-cabinet", "Turn on the toaster, then open a cabinet.",
         "توستر را روشن کن، سپس یکی از کابینت‌ها را باز کن.",
         ["toggle_on Toaster", "open Cabinet"], ["Toaster.isToggled = true", "any Cabinet.isOpen = true"]),
        ("three-appliances", "Turn on the toaster, the coffee machine and the microwave, in that order.",
         "به‌ترتیب توستر، قهوه‌ساز و مایکروویو را روشن کن.",
         ["toggle_on Toaster", "toggle_on CoffeeMachine", "toggle_on Microwave"],
         ["Toaster.isToggled = true", "CoffeeMachine.isToggled = true", "Microwave.isToggled = true"]),
        ("three-doors", "Open the fridge, then the microwave, then a cabinet.",
         "به‌ترتیب در یخچال، در مایکروویو و یکی از کابینت‌ها را باز کن.",
         ["open Fridge", "open Microwave", "open Cabinet"],
         ["Fridge.isOpen = true", "Microwave.isOpen = true", "any Cabinet.isOpen = true"]),
        ("light-fridge-egg", "Turn off the light, open the fridge, and pick up the egg.",
         "چراغ را خاموش کن، در یخچال را باز کن و تخم‌مرغ را بردار.",
         ["toggle_off LightSwitch", "open Fridge", "pickup Egg"],
         ["LightSwitch.isToggled = false", "Fridge.isOpen = true", "holding Egg"]),
        ("knife-bread-toaster", "Pick up the knife, slice the bread, then turn on the toaster.",
         "چاقو را بردار، نان را برش بزن و سپس توستر را روشن کن.",
         ["pickup Knife", "slice Bread", "toggle_on Toaster"], ["Bread.isSliced = true", "Toaster.isToggled = true"]),
        ("mug-water-sink", "Pick up the mug, fill it with water, and put it in the sink.",
         "لیوان را بردار، با آب پرش کن و داخل سینک بگذار.",
         ["pickup Mug", "fill Mug = water", "put_in SinkBasin"], ["Mug.fillLiquid = water", "Mug in SinkBasin"]),
        ("open-all-cabinets-light", "Open all the cabinets, then turn off the light.",
         "همهٔ کابینت‌ها را باز کن، سپس چراغ را خاموش کن.",
         ["open_all Cabinet", "toggle_off LightSwitch"], ["all Cabinet.isOpen = true", "LightSwitch.isToggled = false"]),
        ("open-all-drawers-fork", "Open all the drawers, then pick up the fork.",
         "همهٔ کشوها را باز کن، سپس چنگال را بردار.",
         ["open_all Drawer", "pickup Fork"], ["all Drawer.isOpen = true", "holding Fork"]),
        ("all-burners-microwave", "Turn on every stove burner, then turn on the microwave.",
         "همهٔ شعله‌های اجاق گاز را روشن کن، سپس مایکروویو را روشن کن.",
         ["turn_on_all StoveKnob", "toggle_on Microwave"], ["all StoveKnob.isToggled = true", "Microwave.isToggled = true"]),
    ]
    for stem, e, f, plan, goal in CHAINS:
        A(stem, e, f, plan, goal)

    # -- chains that undo a state the kitchen is first PUT in (every kitchen starts closed / off / clean) -------
    UNDO = [
        ("close-fridge-faucet-off", "Close the fridge, then turn off the faucet.",
         "در یخچال را ببند، سپس شیر آب را ببند.",
         ["close Fridge", "toggle_off Faucet"], ["Fridge.isOpen = false", "Faucet.isToggled = false"],
         ["open Fridge", "toggle_on Faucet"]),
        ("toaster-coffee-off", "Turn off the toaster, then the coffee machine.",
         "اول توستر و بعد قهوه‌ساز را خاموش کن.",
         ["toggle_off Toaster", "toggle_off CoffeeMachine"],
         ["Toaster.isToggled = false", "CoffeeMachine.isToggled = false"],
         ["toggle_on Toaster", "toggle_on CoffeeMachine"]),
        ("close-microwave-light-on", "Close the microwave, then turn the light back on.",
         "در مایکروویو را ببند، سپس چراغ را دوباره روشن کن.",
         ["close Microwave", "toggle_on LightSwitch"], ["Microwave.isOpen = false", "LightSwitch.isToggled = true"],
         ["open Microwave", "toggle_off LightSwitch"]),
        ("stove-off-close-fridge", "Turn off the stove, then close the fridge.",
         "اجاق گاز را خاموش کن، سپس در یخچال را ببند.",
         ["toggle_off StoveKnob", "close Fridge"], ["all StoveKnob.isToggled = false", "Fridge.isOpen = false"],
         ["toggle_on StoveKnob", "open Fridge"]),
        ("clean-plate-bowl", "Clean the plate, then clean the bowl.", "اول بشقاب و بعد کاسه را تمیز کن.",
         ["clean Plate", "clean Bowl"], ["Plate.isDirty = false", "Bowl.isDirty = false"],
         ["dirty Plate", "dirty Bowl"]),
        ("empty-mug-sink", "Empty the mug and put it in the sink.", "لیوان را خالی کن و داخل سینک بگذار.",
         ["empty Mug", "pick_and_place Mug -> SinkBasin"], ["Mug.isFilledWithLiquid = false", "Mug in SinkBasin"],
         ["fill Mug = coffee"]),
        ("empty-cup-wine", "Empty the cup, then fill it with wine.", "فنجان را خالی کن، سپس با شراب پرش کن.",
         ["empty Cup", "fill Cup = wine"], ["Cup.fillLiquid = wine"], ["fill Cup = water"]),
        ("close-all-light-off", "Close everything that is open, then turn off the light.",
         "هرچه باز است را ببند، سپس چراغ را خاموش کن.",
         ["close_all", "toggle_off LightSwitch"],
         ["Fridge.isOpen = false", "Microwave.isOpen = false", "LightSwitch.isToggled = false"],
         ["open Fridge", "open Microwave"]),
    ]
    for stem, e, f, plan, goal, setup in UNDO:
        A(stem, e, f, plan, goal, setup=setup)
    return out


# ---------------------------------------------------------------------------------------------------------------
# L3 -- a need or an end state; the actions are never named and must be reasoned out
# ---------------------------------------------------------------------------------------------------------------
def build_l3():
    knife = ["pickup Knife"]
    to = lambda t, r: f"pick_and_place {t} -> {r}"
    ROWS = [
        # --- temperature as a wish ---
        ("chilled-apple", "I'd like a chilled apple.", "یک سیب خنک می‌خواهم.",
         ["cool Apple"], ["Apple.temperature = Cold"], []),
        ("cold-tomato", "This tomato should be served cold.", "این گوجه‌فرنگی باید سرد سرو شود.",
         ["cool Tomato"], ["Tomato.temperature = Cold"], []),
        ("cold-bread", "I prefer my bread cold.", "من نان را سرد می‌پسندم.",
         ["cool Bread"], ["Bread.temperature = Cold"], []),
        ("cold-potato", "The potato needs to cool down properly.", "سیب‌زمینی باید حسابی خنک شود.",
         ["cool Potato"], ["Potato.temperature = Cold"], []),
        ("warm-potato", "I'd like a warm potato.", "یک سیب‌زمینی گرم می‌خواهم.",
         ["heat Potato"], ["Potato.temperature = Hot"], []),
        ("warm-bread", "Warm up some bread for me.", "کمی نان برایم گرم کن.",
         ["heat Bread"], ["Bread.temperature = Hot"], []),
        ("hot-tomato", "I feel like eating a hot tomato.", "هوس گوجه‌فرنگی داغ کرده‌ام.",
         ["heat Tomato"], ["Tomato.temperature = Hot"], []),
        ("warm-apple", "The apple is too cold to eat. Take the chill off it.", "سیب برای خوردن زیادی سرد است؛ گرمش کن.",
         ["heat Apple"], ["Apple.temperature = Hot"], []),
        # --- serving ---
        ("serve-hot-potato", "Serve the potato hot on a plate.", "سیب‌زمینی را داغ در یک بشقاب سرو کن.",
         ["heat Potato", to("Potato", "Plate")], ["Potato.temperature = Hot", "Potato in Plate"], []),
        ("serve-cold-apple", "Serve me a chilled apple on a plate.", "یک سیب خنک در بشقاب برایم بیاور.",
         ["cool Apple", to("Apple", "Plate")], ["Apple.temperature = Cold", "Apple in Plate"], []),
        ("serve-hot-egg", "Serve me a hot egg in a bowl.", "یک تخم‌مرغ داغ در کاسه برایم سرو کن.",
         ["heat Egg", to("Egg", "Bowl")], ["Egg.temperature = Hot", "Egg in Bowl"], []),
        ("serve-cold-tomato", "Serve a cold tomato on a plate.", "یک گوجه‌فرنگی سرد در بشقاب سرو کن.",
         ["cool Tomato", to("Tomato", "Plate")], ["Tomato.temperature = Cold", "Tomato in Plate"], []),
        ("serve-cooked-potato", "Dinner is a cooked potato in a bowl.", "شام، سیب‌زمینی پخته در کاسه است.",
         ["cook Potato", to("Potato", "Bowl")], ["Potato.isCooked = true", "Potato in Bowl"], []),
        # --- cutting ---
        ("apple-slices", "I want some apple slices.", "چند برش سیب می‌خواهم.",
         knife + ["slice Apple"], ["Apple.isSliced = true"], []),
        ("tomato-for-salad", "Get the tomato ready for a salad.", "گوجه‌فرنگی را برای سالاد آماده کن.",
         knife + ["slice Tomato"], ["Tomato.isSliced = true"], []),
        ("bread-for-breakfast", "Cut up some bread for breakfast.", "کمی نان برای صبحانه آماده کن؛ باید تکه شده باشد.",
         knife + ["slice Bread"], ["Bread.isSliced = true"], []),
        ("potato-for-frying", "The potato has to be in pieces before I can fry it.",
         "سیب‌زمینی باید قبل از سرخ‌کردن تکه‌تکه شود.",
         knife + ["slice Potato"], ["Potato.isSliced = true"], []),
        ("shred-lettuce", "The lettuce is too big to eat like this.", "کاهو این‌طوری برای خوردن زیادی بزرگ است.",
         knife + ["slice Lettuce"], ["Lettuce.isSliced = true"], []),
        ("salad", "Make me a salad.", "برایم سالاد درست کن.",
         knife + ["slice Lettuce", "slice Tomato"], ["Lettuce.isSliced = true", "Tomato.isSliced = true"], []),
        ("sandwich-prep", "Prepare the ingredients for a sandwich.", "مواد لازم برای ساندویچ را آماده کن.",
         knife + ["slice Bread", "slice Tomato", "slice Lettuce"],
         ["Bread.isSliced = true", "Tomato.isSliced = true", "Lettuce.isSliced = true"], []),
        ("omelette-egg", "I need a cracked egg for the omelette.", "برای املت یک تخم‌مرغ شکسته لازم دارم.",
         ["break Egg"], ["Egg.isBroken = true"], []),
        ("toast", "Make some toast.", "نان تست درست کن.",
         knife + ["slice Bread", "toggle_on Toaster"], ["Bread.isSliced = true", "Toaster.isToggled = true"], []),
        # --- drinks ---
        ("coffee", "Make me a coffee.", "برایم قهوه درست کن.",
         [to("Mug", "CoffeeMachine"), "toggle_on CoffeeMachine"],
         ["Mug in CoffeeMachine", "CoffeeMachine.isToggled = true"], []),
        ("boil-water", "Boil some water.", "کمی آب بجوشان.",
         ["fill Pot = water", to("Pot", "StoveBurner"), "toggle_on StoveKnob"],
         ["Pot.isFilledWithLiquid = true", "Pot in StoveBurner", "any StoveKnob.isToggled = true"], []),
        ("tea-kettle", "Get the kettle going for tea.", "کتری را برای چای آماده کن و روی اجاق بگذار تا جوش بیاید.",
         ["fill Kettle = water", to("Kettle", "StoveBurner"), "toggle_on StoveKnob"],
         ["Kettle.isFilledWithLiquid = true", "Kettle in StoveBurner", "any StoveKnob.isToggled = true"], []),
        ("thirsty", "I'm thirsty. Bring me a cup of water.", "تشنه‌ام؛ یک فنجان آب برایم بیاور.",
         ["fill Cup = water", "pickup Cup"], ["Cup.fillLiquid = water", "holding Cup"], []),
        ("wine", "Pour me some wine.", "کمی شراب برایم بریز.",
         ["fill Cup = wine"], ["Cup.fillLiquid = wine"], []),
        ("chilled-water", "I want a cup of water chilling in the fridge.", "یک فنجان آب می‌خواهم که در یخچال خنک شود.",
         ["fill Cup = water", to("Cup", "Fridge"), "close Fridge"],
         ["Cup.fillLiquid = water", "Cup in Fridge", "Fridge.isOpen = false"], []),
        ("stale-coffee", "This coffee has gone cold and stale. Get rid of it.",
         "این قهوه سرد و مانده شده است؛ دورش بریز.",
         ["empty Mug"], ["Mug.isFilledWithLiquid = false"], ["fill Mug = coffee"]),
        ("old-water", "The cup has old water in it.", "داخل فنجان آب مانده است.",
         ["empty Cup"], ["Cup.isFilledWithLiquid = false"], ["fill Cup = water"]),
        # --- washing up ---
        ("dishes", "Do the dishes.", "ظرف‌ها را بشوی.",
         ["wash Plate", "wash Bowl"], ["Plate.isDirty = false", "Bowl.isDirty = false"], ["dirty Plate", "dirty Bowl"]),
        ("dirty-cups", "The cups are dirty. Take care of them.", "لیوان و فنجان کثیف‌اند؛ رسیدگی کن.",
         ["wash Mug", "wash Cup"], ["Mug.isDirty = false", "Cup.isDirty = false"], ["dirty Mug", "dirty Cup"]),
        ("cookware", "Clean up the cookware after dinner.", "بعد از شام ظرف‌های پخت‌وپز را تمیز کن.",
         ["wash Pan", "wash Pot"], ["Pan.isDirty = false", "Pot.isDirty = false"], ["dirty Pan", "dirty Pot"]),
        # --- throwing away ---
        ("bad-tomato", "The tomato has gone bad.", "گوجه‌فرنگی خراب شده است.",
         [to("Tomato", "GarbageCan")], ["Tomato in GarbageCan"], []),
        ("stale-bread", "This bread is stale. Get rid of it.", "این نان بیات شده است؛ از شرش خلاص شو.",
         [to("Bread", "GarbageCan")], ["Bread in GarbageCan"], []),
        ("rotten-egg", "The egg is rotten.", "تخم‌مرغ گندیده است.",
         [to("Egg", "GarbageCan")], ["Egg in GarbageCan"], []),
        ("wilted-lettuce", "The lettuce has wilted and nobody will eat it.", "کاهو پلاسیده شده و دیگر کسی آن را نمی‌خورد.",
         [to("Lettuce", "GarbageCan")], ["Lettuce in GarbageCan"], []),
        ("old-produce", "The potato and the apple are both past their best.", "سیب‌زمینی و سیب هر دو کهنه شده‌اند.",
         [to("Potato", "GarbageCan"), to("Apple", "GarbageCan")], ["Potato in GarbageCan", "Apple in GarbageCan"], []),
        # --- putting away ---
        ("groceries", "Put the lettuce, the tomato and the egg where they will stay fresh.",
         "کاهو، گوجه‌فرنگی و تخم‌مرغ را جایی بگذار که تازه بمانند.",
         [to("Lettuce", "Fridge"), to("Tomato", "Fridge"), to("Egg", "Fridge")],
         ["Lettuce in Fridge", "Tomato in Fridge", "Egg in Fridge"], []),
        ("keep-cold", "Store the apple and the potato somewhere cold.", "سیب و سیب‌زمینی را در جای سرد نگه‌داری کن.",
         [to("Apple", "Fridge"), to("Potato", "Fridge")], ["Apple in Fridge", "Potato in Fridge"], []),
        ("cutlery-sink", "The fork, the spoon and the knife have been used. Put them where they get washed.",
         "چنگال، قاشق و چاقو استفاده شده‌اند؛ آن‌ها را جایی بگذار که شسته می‌شوند.",
         [to("Fork", "SinkBasin"), to("Spoon", "SinkBasin"), to("Knife", "SinkBasin")],
         ["Fork in SinkBasin", "Spoon in SinkBasin", "Knife in SinkBasin"], []),
        ("cutlery-away", "The butter knife and the spatula are lying around. Put them where cutlery is kept.",
         "کارد کره‌خوری و کفگیر همین‌طور رها شده‌اند؛ آن‌ها را جایی بگذار که قاشق و چنگال نگه‌داری می‌شود.",
         [to("ButterKnife", "Drawer"), to("Spatula", "Drawer")], ["ButterKnife in Drawer", "Spatula in Drawer"], []),
        ("set-table", "Set the table for one.", "میز را برای یک نفر بچین.",
         [to("Plate", "DiningTable"), to("Fork", "DiningTable"), to("Knife", "DiningTable")],
         ["Plate in DiningTable", "Fork in DiningTable", "Knife in DiningTable"], []),
        ("seasoning-away", "Tidy the seasonings away into a cabinet.", "ادویه‌ها را مرتب داخل کابینت بگذار.",
         [to("SaltShaker", "Cabinet"), to("PepperShaker", "Cabinet")],
         ["SaltShaker in Cabinet", "PepperShaker in Cabinet"], []),
        # --- the state of the room ---
        ("too-dark", "It's too dark in here.", "اینجا خیلی تاریک است.",
         ["toggle_on LightSwitch"], ["LightSwitch.isToggled = true"], ["toggle_off LightSwitch"]),
        ("too-bright", "It's too bright in here.", "اینجا خیلی روشن است.",
         ["toggle_off LightSwitch"], ["LightSwitch.isToggled = false"], []),
        ("save-power", "Save electricity. Make sure no appliance has been left on.",
         "در مصرف برق صرفه‌جویی کن؛ مطمئن شو هیچ وسیله‌ای روشن نمانده است.",
         ["toggle_off Toaster", "toggle_off CoffeeMachine"],
         ["Toaster.isToggled = false", "CoffeeMachine.isToggled = false"],
         ["toggle_on Toaster", "toggle_on CoffeeMachine"]),
        ("leaving-house", "We are leaving the house. Make the kitchen safe.",
         "داریم از خانه بیرون می‌رویم؛ آشپزخانه را ایمن کن.",
         ["toggle_off StoveKnob", "toggle_off Faucet"], ["all StoveKnob.isToggled = false", "Faucet.isToggled = false"],
         ["toggle_on StoveKnob", "toggle_on Faucet"]),
        ("fridge-and-water", "Someone left the fridge open and the water running.",
         "یک نفر در یخچال را باز و شیر آب را باز گذاشته است.",
         ["close Fridge", "toggle_off Faucet"], ["Fridge.isOpen = false", "Faucet.isToggled = false"],
         ["open Fridge", "toggle_on Faucet"]),
        ("left-open", "Everything has been left open. Tidy that up.", "همه‌چیز باز مانده است؛ مرتبش کن.",
         ["close Fridge", "close Microwave", "close Cabinet"],
         ["Fridge.isOpen = false", "Microwave.isOpen = false", "all Cabinet.isOpen = false"],
         ["open Fridge", "open Microwave", "open Cabinet"]),
        # --- cooking set-up ---
        ("pan-heating", "Get a pan heating on the stove.", "یک تابه روی اجاق بگذار تا داغ شود.",
         [to("Pan", "StoveBurner"), "toggle_on StoveKnob"], ["Pan in StoveBurner", "any StoveKnob.isToggled = true"], []),
        ("pot-heating", "Start the pot on the stove for soup.", "قابلمه را برای سوپ روی اجاق بار بگذار.",
         [to("Pot", "StoveBurner"), "toggle_on StoveKnob"], ["Pot in StoveBurner", "any StoveKnob.isToggled = true"], []),
        ("breakfast", "Breakfast: a hot potato on a plate and a cup of coffee.",
         "صبحانه: یک سیب‌زمینی داغ در بشقاب و یک فنجان قهوه.",
         ["heat Potato", to("Potato", "Plate"), "fill Cup = coffee"],
         ["Potato.temperature = Hot", "Potato in Plate", "Cup.fillLiquid = coffee"], []),
        # --- the right tool, never named ---
        ("tool-cut", "I need something to cut with.", "چیزی برای بریدن لازم دارم.",
         ["pickup Knife"], ["holding Knife"], []),
        ("tool-soup", "I need something to eat my soup with.", "چیزی می‌خواهم که با آن سوپم را بخورم.",
         ["pickup Spoon"], ["holding Spoon"], []),
        ("tool-flip", "I need something to flip the pancakes with.", "چیزی برای برگرداندن پنکیک‌ها لازم دارم.",
         ["pickup Spatula"], ["holding Spatula"], []),
        ("tool-salt", "My food needs salt.", "غذایم نمک کم دارد.",
         ["pickup SaltShaker"], ["holding SaltShaker"], []),
        ("tool-scrub", "I need something to scrub the dishes with.", "چیزی برای سابیدن ظرف‌ها لازم دارم.",
         ["pickup DishSponge"], ["holding DishSponge"], []),
        ("tool-soap", "I want to wash my hands.", "می‌خواهم دست‌هایم را بشویم.",
         ["pickup SoapBottle"], ["holding SoapBottle"], []),
    ]
    return [item("L3", stem, e, f, plan, goal, setup=setup) for stem, e, f, plan, goal, setup in ROWS]


def build_all():
    return build_l1() + build_l2() + build_l3()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    seed = build_all()
    inv = gs.load_inventory()
    actions = gs.known_actions()
    ids = Counter(it["id"] for it in seed)
    dup = [i for i, n in ids.items() if n > 1]
    assert not dup, f"duplicate ids: {dup}"
    bad = 0
    for it in seed:
        errs, _ = gs.validate_item(gs.normalize(it), actions, inv)
        for e in errs:
            bad += 1
            print(f"INVALID {it['id']}: {e}")
    assert not bad, f"{bad} seed validation error(s)"

    doc = {"version": 1, "items": [], "deleted": []} if "--reset" in argv else gs.load()
    have = {it["id"] for it in doc["items"]}
    gone = set(doc.get("deleted") or [])
    added = [gs.normalize(it) for it in seed if it["id"] not in have and it["id"] not in gone]
    doc["items"] += added
    gs.save(doc)

    lv = Counter(it["level"] for it in doc["items"])
    print(f"seed: {len(seed)} instructions  |  added {len(added)}, kept {len(have)} existing, "
          f"{len(gone)} deleted ids left alone")
    print(f"general.json now holds {len(doc['items'])}: L1 {lv['L1']} / L2 {lv['L2']} / L3 {lv['L3']} "
          f"(x2 languages = {2 * len(doc['items'])} records)")
    if inv:
        full = sum(1 for it in doc["items"] if gs.applicability(it, inv)["n"] == len(inv["scenes"]))
        print(f"{full}/{len(doc['items'])} apply in all {len(inv['scenes'])} scanned kitchens; "
              f"the rest are skipped where the object is missing.")


if __name__ == "__main__":
    main()
