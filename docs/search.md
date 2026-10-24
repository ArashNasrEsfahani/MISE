# `search_for` — a systematic scene search

Gate: `python sim/_search_selftest.py` → **108/108**, headless (no ai2thor, GPU, Ollama or ultralytics).

## The nine lines this exists because of

```
15:16:14  active search: turning to look (detector only — no scene metadata)
15:16:15  perceived: 10 object type(s) seen by YOLO
15:16:26  agent: I'm looking for the apple.  [1 step(s)]
15:16:27  look around: ok
15:16:29  bandit reward: environment=1.00
15:16:29  loop: insert procedure #4 — command succeeded
```

Three separate failures:

1. **The sweep was invisible.** `sweep_detections` called `look_around(turn=45)` — 8 yields, and the worker
   renders one frame per yield, so a full 360° search took **8 frames** (~0.2 s). 45° was chosen to keep YOLO to
   8 inferences, and that same number silently set the frame rate of the search.
2. **There was no search.** The planner's only perception verbs, `look_around` and `scan_for`, both spin *in
   place*. Neither can find anything outside the current field of view, and `scan_for` resolves its target from
   metadata — so it only ever "finds" what the simulator already told it.
3. **It reported success having seen nothing.** `_live_goal_check` understood only pickup goals, so the run was
   scored from its steps: `look around: ok` → `environment=1.00` → the loop stored a proven procedure for a run
   in which no apple was ever seen. Two such procedures are still in `knowledge_base.json` (ids 3 and 4).

Failure 3 had a second edge: the bandit's `replay_procedure` arm matches on command similarity, so the *next*
`find the apple` would replay that degenerate procedure and skip planning entirely.

---

## 1. The policy — `sim/search.py`

Pure, stdlib-only, GPU-free. `rank_stops()` turns what the agent knows into an ordered list of places to go,
each carrying **why** it is on the list. The authority order *is* the claim that retrieval changes behaviour:

| rank | source | beats |
|---|---|---|
| `here` | always first — one free sweep before walking anywhere | — |
| `recall` | `memory.query_location` — this type was actually observed somewhere | everything |
| `receptacle` | a container the **KB** associates with the target (weight 0.95) | the prior |
| `receptacle` | the curated `RECEPTACLE_PRIOR` (Apple→Fridge 0.90, CounterTop 0.70, …) | guessing |
| `frontier` | farthest-first over discovered nav cells, so the room still gets covered | nothing left |

Distance is a real term, not a tie-break: for a Mug the prior prefers `Cabinet` (0.70) to `CounterTop` (0.60),
but a cabinet 8 m away loses to a counter 1 m away. Walking past the likely place to reach the
slightly-more-likely one is worse search, not better.

`sweep_schedule(turn, detect_every)` is the fix for failure 1 — rotation granularity and detection cadence
become independent numbers:

```
sweep_schedule(45, 45)  ->  8 frames,  8 inferences   # what shipped, and what nobody could see
sweep_schedule( 8, 45)  -> 45 frames,  8 inferences   # watchable, same cost
```

`search_budget(speed)` scales thoroughness with the motion profile (4/6/8 stops for fast/normal/cinematic), each
time-capped. `not_found_lesson()` produces the write-back for a failed search.

**What the policy never reads:** AI2-THOR's containment graph (`receptacleObjectIds` / `parentReceptacles`) for
the target. That is the answer; walking to it afterwards is not a search, and under `--perception perceived` it
would make the ablation measure nothing. Receptacle *positions* do come from the simulator — `walk_to` needs a
nav target, as it does for every other skill here — but *whether the apple is in the fridge* is only ever
established by looking.

## 2. The skill — `ThorAgent.search_for`

A generator, one rendered frame per yield. Per stop: walk → open if closed (the door swings) → sweep → look.
Stops at the first sighting. One sensor abstraction covers both perception arms:

- **perceived** → `detect_objects(detector)`, the target type present in this frame's detections;
- **oracle** → `find(type, visible_only=True)`, AI2-THOR's `visible` flag, which is range-gated at 1.5 m and is
  therefore a *sighting* rather than a lookup.

Every knowledge source is optional and independently guarded, so the skill still runs on a cold agent with no
memory, no KB and no detector — it just falls further down the authority order.

It sets `self.last_search` = `{found, type, receptacle, stops_planned, stops_visited, visited, elapsed, reason}`.
That is what makes failure 3 fixable: the caller can score the command on whether the thing was *seen*.

`sim/thor_actions.py` now guards its `ai2thor` import, which is what lets the real generator be bound to a fake
agent and driven headlessly. Constructing a live `ThorAgent` off-GPU still fails loudly with the original error.

## 3. The wiring — `sim/webapp.py`

- `_search_target(command)` — bilingual via `agentic_rag.lexical_pivot`, so `find the apple` and
  `سیب را پیدا کن` both resolve to `Apple`. It deliberately does **not** pass `scene_types`: "which types exist
  in this room" is exactly the oracle perceived mode withholds, and letting it in through the command parser
  would hand the search half its answer.
- `_live_goal_check` gains a search branch — **failure 3**. A search succeeds iff `last_search.found`.
- Write-back: found → `memory.observe_location(...)` (the **single-object** writer; `observe_scene(spatial_snapshot())`
  reads ground-truth `parentReceptacles` for all 77 objects and would record the whole kitchen's true layout
  whether or not the agent looked). Not found → a lesson naming the places ruled out.
- `_repair_plan_for_command` upgrades a search command's `look_around`/`scan_for` into `search_for`. This runs
  **before** the pickup early-return, and it is what neutralises the two degenerate procedures already in the
  KB without anyone hand-editing their own knowledge base.

## 4. Where the name had to be registered

The action vocabulary is maintained by hand in six places and nothing validates it end to end:

| file | what |
|---|---|
| `sim/skills.py` | `SkillSpec("search_for", "L2", …)` — the manifest name must equal the `ThorAgent` method name |
| `sim/agent_llm.py` | `PLANNER_SYSTEM`'s PERCEPTION block |
| `sim/webapp.py` | the `comp` dispatch table in `_exec_one` |
| `sim/webapp.py` + `sim/eval.py` | the approach-step set, **in lockstep** — otherwise the console and the harness repair the same plan differently |
| `sim/agentic_rag.py` | EN + FA verb lexicons |
| `sim/eval.py` `_NAV_HELPERS` | **deliberately not touched** — see below |

## 5. Integrity — the evaluation action space changed

`search_for` is a **scored** action. Unlike `look_around`/`scan_for` it leaves where the agent is standing,
opens containers, and determines whether the object is found at all, so it is a real step in the plan and counts
in Procedural Correctness and Task Decomposition Score.

That changes the action space PC/TDS are computed over. A gold plan of `[go_to, pickup]` is now compared against
a generated `[search_for, pickup]` — one Levenshtein insertion on PC, one false positive on TDS. **Every PC/TDS
number in the nine `sim/eval_*.json` files was computed without it and is no longer directly comparable.**

Two mitigations ship with it:

- `sim/eval.py --score-search off` restores the old space, so a historical run can be reproduced like for like;
- the effective set is recorded in the results JSON as `config.scored_actions`, so no future reader has to guess
  which space produced a number.

## 6. Three bugs the FIRST live run found

None of these were visible off-GPU. All three are now covered by the gate.

**The plan and the goal check disagreed on the target.** `find the wine bottle` → the bilingual pivot resolved
`WineBottle`, the LLM planned `search_for Bottle`, the search found a Bottle, and the goal check reported
"WineBottle was never seen". Plan repair now retargets a search step naming a type the command never asked for,
while leaving alone a step that names one of several requested types (`find the apple and the knife`).

**A not-found lesson was read as a positive affinity.** The worst of the three. A failed search writes
*"WineBottle was NOT at: Cabinet, Floor, Shelf, CounterTop, Fridge"*; `kb_affinity` saw the target and five
receptacle names in one sentence and returned five learned affinities — so the next search walked straight back
round the same five places, **with the confidence of knowledge**. The self-improving loop was teaching itself to
repeat a failure. `is_negative()` now keeps such sentences out of affinity and `kb_exclusions()` turns them into
a demotion instead:

```
search [3/6]: WineBottle is usually at/in a SinkBasin
search [4/6]: WineBottle was NOT here last time - checking again only after everywhere else
```

Demoted, not dropped — someone may have put it back.

**The sink was never searched.** AI2-THOR ships both `Sink` (the fixture) and `SinkBasin` (the basin things
actually rest in), and the scene reports the basin. A prior keyed on `"Sink"` matched nothing, so the sink was
silently absent from every route. `RECEPTACLE_ALIASES` fixes it, and `Shelf` — missing from every prior — was
added at the same time.

## 7. Three more from live use

**It left the doors open.** A search opened the fridge to look inside and never shut it. `search_for` now
tracks what it opened and closes all of it — including the container the object was found in, because a
follow-up "pick it up" re-opens that by itself (below). Verified live: **0 containers left open** after a search
that found the egg inside the fridge.

**Reaching for something inside a closed container.** `_open_parent` walks to the target's parent receptacle and
swings it open before the action. This is execution-side — it reads `parentReceptacles` for an object the agent
has *already been told to act on*, the same latitude every skill takes when it resolves an objectId. It is the
planner's view of the room that perceived mode restricts, not how a decided action reaches its target.

**Pickups happened from too far away.** Two causes, both fixed:

- `walk_to` took `poses[0]` from `GetInteractablePoses`, which guarantees only that the object is *reachable*
  from that pose, not that it is near. `_closest_pose` now picks the pose nearest the object.
- `_approach_then` skipped the walk entirely whenever the object was flagged `visible` — which in AI2-THOR means
  "in frame and within 1.5 m", not "near". It now always approaches, then `close_in` edges forward until a step
  fails or stops closing the gap.

Measured live: picking up an apple 1.1 m away moved the agent **1.12 m**.

## 8. What is not verified off-GPU

The 105 checks drive the real generator against a fake controller: ranking authority, the frame/inference split,
walking, opening, skipping the unreachable, the scene guard on recalled beliefs, both sensors, bilingual intent,
plan repair, negative-lesson handling, and the honesty property. They cannot show that YOLO recognises a real
apple, how the walk looks on screen, or the true timing. Those are missions **C6c** and **C6d** in
`HOW-TO-TEST.html`.

Live-verified 2026-08-22 (FloorPlan1, gemma3:4b): a `find the apple` that replayed degenerate procedure #3 and
was upgraded in flight; a six-stop `find the wine bottle` that opened the cabinet and the fridge, found nothing
and wrote the lesson; and a re-run of the same search that correctly demoted what the first one ruled out.
