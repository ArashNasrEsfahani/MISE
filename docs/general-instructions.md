# The general instruction set

A second, **scene-agnostic** bilingual evaluation set: 280 instructions (110 L1 · 113 L2 · 57 L3; seeded as 288 — 111 · 118 · 59 — and hand-edited since 2026-09-27, see `general_audit.jsonl`), each in
English and Persian — 576 records — that can be given in any iTHOR kitchen. It complements the Phase-9 dataset
(`sim/dataset/commands.jsonl`, 102 records), where every record is grounded in exactly one of three kitchens.

It is viewable and **editable in the console**. The Evaluate view has three tabs — **Instructions** (the
default: this set), *Run & results*, and *Scene-grounded set* (the read-only Phase-9 browser). On Instructions:

| Control | What it does |
|---|---|
| level chips `all · L1 · L2 · L3` | filter by level; each chip shows its count, its tooltip how many are checked |
| `show:` | everything · not checked yet · checked · not in every kitchen · needs a setup |
| search | English, فارسی or id |
| **check** (per row) | marks the instruction as read and correct in both languages; drives the progress bar (`37 of 288 checked`). Survives an edit. Reviewing 288 × 2 records takes several sittings — *show: not checked yet* is where the next one starts |
| **edit** | opens the row in place: id, level, English, فارسی, setup / reference plan / goal. `Ctrl+Enter` saves, `Esc` cancels |
| **duplicate** | a prefilled *add* form — the quick way to write a variant |
| **delete** | two clicks (arm, then confirm); an **undo** toast restores it for 9 s |
| **+ add instruction** | an empty form, level preset to the active chip |
| **export CSV** | the whole set as one bilingual table (UTF-8 with BOM, so Excel keeps the Persian) |

Every add / update / check / delete is appended to `general_audit.jsonl`.

> **If the panel is empty or has no buttons, the console is stale.** `sim/web/*` is re-read from disk on every
> request but `webapp.py` is loaded once, so a console started before a code change serves the new page on an
> old server and `/general` returns 404. The page shows a red "console is newer than the running server" bar
> (`UI_API` / `WEBAPP_API`, now **4**), and `START-TESTING.bat` now detects it and defaults to restarting.

| File | Role |
|---|---|
| `sim/dataset/general.json` | the set itself — the source of truth; the console edits it in place |
| `sim/dataset/general_audit.jsonl` | one line per add / update / delete (before + after) |
| `sim/dataset/scene_inventory.json` | live scan of all 30 kitchens: types, affordances, **initial states** |
| `sim/general_set.py` | store, validation, the goal/plan text format, record expansion (pure stdlib) |
| `sim/gen_general.py` | the seed. Re-running only **adds missing ids** — never overwrites an edit, never resurrects a delete. `--reset` regenerates |
| `sim/scene_inventory.py` | re-scan the kitchens (WSL) |
| `sim/validate_general.py` | run every **reference plan** in the simulator, no LLM (WSL) |
| `sim/_general_selftest.py` | offline gate, 85/85 |

## 1. What an item is

One item = one instruction in both languages, because the pair shares a gold and is edited as a unit.

```
id     g-l2-fridge-apple            level  L2
en     Put the apple in the fridge and close the door.
fa     سیب را داخل یخچال بگذار و درش را ببند.
setup  (none)
plan   pick_and_place Apple -> Fridge
       close Fridge
goal   Apple in Fridge
       Fridge.isOpen = false
```

* **`plan` is the reference plan; the gold labels are derived from it** by `eval.labels` — the function that
  labels a *generated* plan. L2 `sequence` and L3 `actions` are therefore in the scorer's vocabulary by
  construction (`fill` → `fill_with_liquid`), and the item is executable, which is what makes §4 possible.
* **`setup`** runs before the agent plans. Every kitchen starts with every door closed, every appliance off,
  the light on and nothing dirty or filled (verified over all 30 — `scene_inventory.json`). So "close the
  fridge" with goal `Fridge.isOpen = false` is true before the agent moves; `setup: open Fridge` gives the
  instruction something to do. The model never sees the setup except through the OBSERVATION it changes.
* **`goal`** is what Success Rate is judged against, at every level.

### The text format (what the editor's three boxes hold — one entry per line)

```
step   action [Object] [-> Destination] [= arg]      pick_and_place Apple -> Fridge · fill Mug = water · crouch
goal   Type.key = value                              Fridge.isOpen = false · Apple.temperature = Cold
       Type in Receptacle                            Apple in Fridge
       holding Type                                  holding Knife
       any|all Type.key = value                      any Cabinet.isOpen = true · all Drawer.isOpen = false
       Type.key <= n   (or >=)                       Fridge.distance <= 1.5
       Agent.key = value                             Agent.isStanding = false
```

A save is validated server-side and a rejected one names the offending line. Errors block (unknown action,
missing Persian, an L2/L3 without a plan); warnings do not (object in no scanned kitchen, goal already true at
the start in *n* kitchens, an L1 whose plan has two scored steps).

## 2. Vocabulary: what "general" rests on

`scene_inventory.py` booted FloorPlan1–30. **35 object types exist in all 30 kitchens**; the set is built from
those plus three common ones — `Drawer` 28/30, `DiningTable` 16/30, `Kettle` 15/30. 270 of 288 items apply
in every kitchen; the console shows each item's count (`28/30 kitchens · no Drawer`).

An instruction need not apply everywhere. Where a kitchen lacks the object the record is **skipped** — see §3.

## 3. The applicability gate (`eval.py --gate on`, the default)

Before scoring a record the harness now checks, against the live scene:

1. **every object the record depends on exists** (exact type — a Potato is not a Pot) → else
   `skipped: object does not exist in this scene: Drawer`;
2. it runs the **setup**; a failed setup step → skipped;
3. **the goal is not already true** → else `skipped: goal already true before the agent acted`.

A skipped record has no SR/PC/TDS and is left out of every mean; aggregates report it as `n_skipped` beside `n`,
and the results doc carries `n_skipped` + `skip_reasons`. Scoring it 0 would charge the model for the dataset's
choice of scene; scoring it 1 would credit the model for doing nothing. `--gate off` scores everything, for a
like-for-like re-run of results made before the gate existed.

Two smaller additions to `sr_score`, both inert for records that do not use them:

* **`any` / `all`** — an un-quantified assertion is judged on *one* object, the nearest of its type. That is
  exact for a Fridge and arbitrary for a Cabinet (2–28 per kitchen).
* **`<=` / `>=`** — for "go to the fridge" (`Fridge.distance <= 1.5`); `visible` is already true from the spawn
  point in many kitchens.

## 4. Reference-plan validation — the executor ceiling

`validate_general.py` does what `_score_record` does around the model call (reset → gate → setup), then
executes the item's **own reference plan** and checks the goal. `ok` = the instruction is scoreable and SR = 1
is attainable; `skipped` = the gate would skip it here; **`FAIL` = the gold is unreachable — a defect in the
item or the executor, never in a model.** A record with unreachable gold scores 0 for every model forever, and
no results table distinguishes that from a hard instruction. ~1 s per item, no LLM, no GPU contention.

```bash
cd /mnt/d/Personal/Master*/sim
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python validate_general.py --scene FloorPlan5
```

Results are in `sim/dataset/general_validation_<scene>.json`. **Run it for a scene before evaluating there**,
and after editing the set.

**Status (2026-09-20).** FloorPlan1: **278 ok · 10 skipped · 0 FAIL** (skipped = no DiningTable there, or the
goal is already true at reset — e.g. the fridge is within 1.5 m of the spawn point). No other kitchen has been
validated yet. **Open:** in FloorPlan5, `g-l1-pickup-lettuce` makes the simulator stop responding — python and
Unity both idle at 0 % CPU, indefinitely. The validator has a per-record watchdog (`--timeout`, default 60 s:
marks the record `HUNG`, restarts the simulator, carries on), but **`eval.py` has no such watchdog**, so that one
record would freeze an evaluation run in FloorPlan5 forever. Undiagnosed.

### What the first run found (2026-09-20) — all in shared code, so they affect the Phase-9 results too

| # | Finding | Fix |
|---|---|---|
| 1 | **`pick_and_place` / `put_in` never opened a closed receptacle.** "Target openable Receptacle is CLOSED" — while `PLANNER_SYSTEM` tells the model high-level actions handle "their own walking, opening", and every kitchen starts with every door closed. 41 of 288 reference plans failed on this alone. | `ThorAgent.put_in` opens the destination first |
| 2 | **`isSliced` was a coin flip.** After `slice Bread` the scene holds the old `Bread` (`isSliced = true`) *and* a new end-piece whose type is also `Bread` (`isSliced = false`), a centimetre apart; the goal read whichever was nearer, and flipped when the agent walked off. | general set uses `any X.isSliced = true` |
| 3 | `fill` / `empty` are offered by `PLANNER_SYSTEM` but were never dispatched by `_exec_one` — a plan using them silently did nothing. | dispatched; labelled `fill_with_liquid` / `empty_liquid` |
| 4 | `drop` was refused whenever the held object touched anything, the agent included. | `force=True`, like every other manipulation |
| 5 | Every walk ended in a refused `LookUp`/`LookDown` ("degree value must be a multiple of 0.1"), so `go_to` reported failure on arrival. | `_pitch_to` rounds to 0.1° |
| 6 | "close the drawer" went to the *nearest* drawer, usually a closed one, not the open one. | `_resolve_for` prefers an instance the action can change |
| 7 | `heat_object` ignored a refused put into the microwave and "heated" an empty oven (intermittent). | one retry from a fresh approach |

### Consequences for the published Phase-9 numbers — *not yet acted on*

Checked offline against the inventory and the gold in `commands.jsonl`:

* **6 of 102 records have a goal that is already true at reset** (`fp1-l2-wash-mug`, `fp2-l3-cold-snack`,
  `fp3-l2-wash-bowl`, en + fa). The apple spawns inside the fridge in FloorPlan2, so "make me a cold snack"
  scores SR = 1 for an empty plan. With `--gate on` these are now skipped.
* **12 records carry an un-quantified `isSliced` goal** (finding 2): their SR has been partly chance.
* **Every record whose gold is `pick_and_place X` into the Fridge** (`*-l2-fridge-*`, `*-l3-tidy-fridge`) had a
  reference plan the executor could not run (finding 1). A model could only succeed by emitting an explicit
  `open Fridge` first — i.e. by *deviating* from the gold it was scored against on PC/TDS.

None of the nine `sim/eval_*.json` files has been re-run. The SR numbers in them were produced by the old
executor and are **not comparable** with runs on this tree; PC/TDS are unaffected (they score the plan, not its
execution). `commands.jsonl` itself was left untouched.

## 5. Running an evaluation on the general set

Console: Evaluate → dataset **general instructions** → list the scenes (`FloorPlan1,FloorPlan5`); every
instruction runs once per scene. CLI:

```bash
python sim/eval.py --dataset sim/dataset/general.json --scenes FloorPlan1,FloorPlan5 \
    --models gemma3:4b --rag agentic --lang all
```

576 records × scenes × models is a long run (the published 102-record runs took tens of minutes per model).
`--limit` cuts evenly across languages — records are interleaved en/fa per item.
