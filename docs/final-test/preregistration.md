# Final test: does specialisation beat model complexity? (preregistration)

Status: **LOCKED v1 (P0, 2026-10-10)**. It was written before any final-test planner call, execution or result
existed, and Astra's referee review (12 points, section 17) has been folded in. From now on, every change is a dated
**amendment** in section 16, with a reason, and it is reported in the thesis. A change made after any test outcome has
been seen keeps the outcome but loses the "preregistered" label for whatever it touches. One item is still open
**before the matrix**, not before this lock: Arash's audit of the family boundaries (gate 12.8). If that audit
regroups anything, it is an amendment made before any outcome exists.

Sources: `docs/research/final-test-astra-review.txt` (Astra's design review; this protocol follows it point by point,
and section 17 lists where it deviates and why), `docs/step5-plan.md`, `docs/step5-progress.md`,
`docs/step4-honest-agent.md`, `sim/step4_gemma.py`, `sim/llm.py`, `sim/step5_eval.py`, `sim/eval.py`,
`sim/dataset/general.json`. Locked code: `sim/final_test_split.py` (split), `sim/final_test_power.py` (inference
functions and the precision simulation). Frozen split: `docs/final-test/split_manifest.json`. Ledger:
`docs/final-test/progress.md`.

---

## 1. The question, stated narrowly

Arash's hope is that **specialisation beats model complexity**: a small local model that has practised in this
kitchen and remembers what it learnt should plan as well as a much larger general model that has not. This protocol
is built so that the data can say **no**. A refuted or inconclusive result is a valid thesis result and is reported in
full.

**What is tested** (Astra, wording kept): *"In FloorPlan1, does Gemma 4 E2B with frozen experiential memory achieve task
success comparable to or higher than specified frontier planners without that memory?"*

What this compares is **two configured whole systems with different prior experience**: the Gemma system has
practised in this kitchen, and its memory acts both in the planner's prompt and in the executor's search. The frontier
system has not practised. A positive result shows that, in this kitchen, *this* practice history plus memory let a
small local model reach the named frontier configuration's task success. It does **not** show that model complexity
matters less in general, and it does not show that memory made Gemma's *planning* competitive (section 14).

Model sizes, stated correctly: Gemma 4 E2B has **2.3 B effective parameters, 5.1 B including embeddings** (model
card). It runs locally as `gemma4:e2b-it-q4_K_M` (Ollama digest `b37049369adf`, Q4_K_M, about 4.6 GB, confirmed by Arash
2026-10-06). The frontier models are hosted, and their parameter counts are not public.

## 2. Hypotheses

Notation: `SR(arm)` is the item-level success rate of an arm (section 9.1). `+M` means "with the frozen memory
snapshot", `0` means "without long-term memory". G = Gemma, O = Opus 5.5, S = Sonnet 5.5.

| id | type | contrast | decision rule |
|---|---|---|---|
| **H1** | **primary, confirmatory** | `D1 = SR(G+M) − SR(O0)` | non-inferiority (H0: D1 ≤ −5 pp) first; superiority (H0: D1 ≤ 0) only if non-inferiority holds. Fixed sequence, so no extra alpha. The exact rule is in section 9.3 |
| H2a/b/c | secondary | memory gain per model: `SR(G+M) − SR(G0)`, `SR(S+M) − SR(S0)`, `SR(O+M) − SR(O0)` | one-sided, H0: gain ≤ 0 (Holm family, section 9.4) |
| H3a/b | secondary | `SR(O+M) − SR(G+M)`, `SR(S+M) − SR(G+M)` | two-sided, H0: difference = 0 (Holm family) |
| H3c | secondary | `SR(G+M) − SR(S0)` | non-inferiority, H0: ≤ −5 pp (Holm family; it enters Holm as min(1, 2p), keeping H1's one-sided .025 convention) |
| H4a/b | secondary | structured memory vs a same-source, equal-token-budget experience control: `SR(G+M) − SR(G+FS)`, `SR(O+M) − SR(O+FS)` | two-sided (Holm family) |
| I | secondary (estimate only) | model × memory interaction `(G+M − G0) − (O+M − O0)` | 95 % CI reported, not tested; a CI that includes 0 does **not** show the interaction is absent |
| E1–E9 | exploratory | Haiku 4.5; one OpenRouter model; Codex CLI systems; learning curve over K; executor-only memory effect; relocation probe; cost, latency, VRAM and energy per success; strict SR; per-level and per-language rates | descriptive, CIs where defined, no test |

If H2 shows that memory helps every model, the headline still stands, but its explanation changes: memory would then be
a general aid, not something special to the small model. The interaction I and H3 say which of the two it is.

**Margin (5 pp).** A **prespecified practical tolerance**: one task in twenty. It is fixed now, before any data, and
it is not claimed to be the threshold below which differences stop mattering. **Expected precision** comes from a
simulation with the actual 42 test families and their sizes (`python sim/final_test_power.py`, section 9.6). The 95 %
half-width is 5–14 pp, depending on how often the two arms disagree (10–30 % of presentations) and how strongly
families cluster (ICC 0–0.6). At a true difference of 0, H1's non-inferiority is shown only 10–48 % of the time. At
+5 pp it is shown 27–97 % of the time, and at +10 pp 53–100 %. So **H1 can succeed only if Gemma+M is actually ahead**,
and "inconclusive" is a likely outcome. "Not significant" is never written as "equal".

## 3. Systems and arms

### 3.1 Planner models (identity and decoding, frozen)

| key | model | where | decoding (frozen) |
|---|---|---|---|
| G | `gemma4:e2b-it-q4_K_M`, digest `b37049369adf` | Ollama on the laptop GPU | `step4_gemma.OPTIONS`: temperature 0, seed 7, top_k 1, num_ctx 8192, num_predict 768, `think=false`, JSON-Schema `format` = `PLAN_SCHEMA` |
| O | `claude-opus-5-5` (**primary frontier comparator**, designated now) | Anthropic API | thinking cannot be disabled on this model; adaptive thinking at the vendor-default effort `medium`, set explicitly; sampling parameters are not accepted by the API; structured output (`output_config.format`) = `PLAN_SCHEMA`; `max_tokens` 8000 |
| S | `claude-sonnet-5-5` | Anthropic API | adaptive thinking at the vendor-default effort `high`, set explicitly; structured output = `PLAN_SCHEMA`; `max_tokens` 8000 |
| H | `claude-haiku-4-5-20251001` (exploratory) | Anthropic API | no thinking (the model's default), temperature 0, structured output = `PLAN_SCHEMA`; `max_tokens` 2000 |
| R | one cheap OpenRouter model (exploratory, chosen in the cost pilot before any test call, recorded by id + provider) | OpenRouter | temperature 0, JSON mode if offered |
| X-* | Codex CLI `gpt-6-sol`, `gpt-6-astra`, `gpt-6-luna`, `gpt-5.6-terra` (exploratory, **reported separately**, section 3.4) | `codex exec` | CLI defaults (recorded); `--output-schema` = `PLAN_SCHEMA` |

- Equal numeric settings do not make sampling equivalent across vendors (Astra). We fix each model's settings and
  record them; we do not claim the decoding is "the same".
- **These are deployment configurations, not matched compute.** The frontier models think (they cannot be told not
  to), and Gemma does not. Gemma runs quantised with a 768-token output budget. The comparison is between systems as
  configured, and it never claims matched compute or intrinsic capability.
- **Effort is protected.** If the cost pilot (section 10) projects that the matrix exceeds the approved budget, the
  exploratory arms are cut first, in this order: R, X-\*, H, learning-curve seeds 1–2, the relocation probe. Only after
  that do O and S drop to effort `low`. That choice uses pilot cost only, never pilot success. If it happens, every H1
  statement names it ("Opus 5.5 at low effort").
- **Equal interface opportunity (dev only).** Every model gets the same bounded chance to clear interface failures on
  the 12 dev items: output truncation, context fit and validator compatibility. A model's output budget (`num_predict` /
  `max_tokens`) is raised only if it truncates on dev, and the change is recorded. A prompt or validator change made for
  one model applies to all of them. The step-4 prompt and validator were developed with Gemma, which is disclosed; this
  dev check is the mitigation.
- **No server-side model fallback.** The Anthropic `fallbacks` parameter stays **off**: an answer from a different
  model would contaminate the arm. A `refusal` stop reason is recorded and counts as an invalid plan for that arm.
- Transport errors (HTTP 429 / 5xx / timeout / connection) are retried up to 3 times with backoff. They are not
  planner attempts. Section 13 covers what happens when a call still fails.
- `stop_reason = max_tokens`, an Ollama `done_reason = length` or an empty reply is a planner error. It goes through
  the single repair, like any invalid reply (`step4_gemma.GemmaPlanner._call`).

### 3.2 Arms

| arm | planner | long-term memory | prompt block | executor search chooser |
|---|---|---|---|---|
| **G0, G+M** | Gemma | none / snapshot S\* | — / memory block | prior / `LtmChooser` |
| **S0, S+M** | Sonnet 5.5 | none / S\* | — / memory block | prior / `LtmChooser` |
| **O0, O+M** | Opus 5.5 | none / S\* | — / memory block | prior / `LtmChooser` |
| G+FS, O+FS (same-source, equal-token-budget experience control) | Gemma, Opus | S\* for search only | experience block from the **same** practice episodes, same token allowance (6.4) | `LtmChooser` (same S\*) |
| H0, H+M (exploratory) | Haiku 4.5 | none / S\* | as above | as above |
| R0, R+M (exploratory) | OpenRouter model | none / S\* | as above | as above |
| X-\*0, X-\*+M (exploratory, separate) | Codex CLI systems | none / S\* | as above | as above |
| P0, P+M (diagnostic, no LLM) | the item's reference plan | none / S\* | — | prior / `LtmChooser` |
| G+M/p, O+M/p (exploratory ablation, no new calls) | the **cached** G+M / O+M plans | S\* in the prompt only | (already in the plan) | prior |

- The **six core arms** are G/S/O × {0, +M}. All `+M` arms get the **identical** frozen snapshot S\* (section 6).
- The P arms show how much search memory helps when the plans are the reference plans. They use gold plans, so they
  are diagnostics only. They are never a planner result, and they are never used to build memory. They **cannot**
  split the memory effect for generated plans (Astra).
- The `/p` ablation does that split for generated plans. It replays the cached G+M and O+M plans with the prior
  chooser, so `+M − +M/p` is the search channel and `+M/p − 0` the prompt channel, for these plans (exploratory).
- The FS arms use the **same** frozen chooser as `+M`, so H4 compares only how the experience reaches the prompt
  (a structured memory block or raw attempt records), with search held equal. This answers Astra's question: does the
  structured memory beat simply showing the model the same experience?

### 3.3 The LLM's role

The LLM is the **planner only**: one plan, then at most one repair, then the plan is executed. The search chooser is
**deterministic in every arm** (`honest_skills.prior_chooser` without memory, `LtmChooser` over `where_is` with
memory). The LLM chooser (`GemmaPlanner.chooser`) is **not** used. So **within a memory condition, only planning
differs between models**. Between memory conditions, both planning and search differ, which makes H1 a whole-system
comparison.
Lesson hypotheses (`Hypothesizer`) run only during practice (step 5, section 7.4), never in a test.

### 3.4 Codex CLI systems (exploratory, never pooled with the API arms)

A Codex model is a **model plus its CLI scaffold** (Astra). Each call is one fresh `codex exec -m <model>
--skip-git-repo-check --ephemeral -s read-only -C <empty temp dir> --json --output-schema <PLAN_SCHEMA file> -o
<file>`, with the prompt on stdin. The temp directory is created per call, holds nothing but the schema file, and is
deleted afterwards. The prompt is byte-identical to the API arms' prompt (system and user text concatenated with fixed
headers). There is no repository, no gold, no personal memory and no browsing; web access is disabled where the CLI
allows it, and the setting is recorded. The archive keeps the CLI version, config, the JSONL event stream, usage and
wall time. Because decoding is not controllable, the sample is the REPEAT subset (30 items × EN + FA) × {0, +M}, with 3
independent calls each. All 3 are reported, and the **first** call (r = 0) is the scored one. The best call is never
selected. Results go in their own table, labelled "model + CLI system". **Conflict of interest:** `gpt-6-astra` also
reviewed this protocol. That is disclosed, and it is one reason why the Codex systems are exploratory.

## 4. The planner / executor contract (identical in every arm)

| element | frozen content |
|---|---|
| instruction | the item's `en` or `fa` text, verbatim |
| system prompt | `step4_gemma.SYSTEM`, frozen as prompt version `ft-v1` (the step-4 version at freeze time, hashed). The tool list is `honest_skills.TOOLS` with `TOOL_SPECS`; the type list is the detector's class list (49 types; an ontology, not a room inventory) |
| user message | `step4_gemma.USER`: the **within-episode observation summary** after the fixed start sweep (`HonestSkills.start_episode` → `objmem` summary), what the agent was told it holds, and the command |
| memory block (`+M` arms) | one block appended to the user message under a fixed header, rendered deterministically by the step-5 facade (`memory_block(wm, level, lang)`: `where_is` lines plus whichever procedure / lesson channels are active in S\*), within `retrieval.budget_tokens[level]`. **No LLM** produces or reorders it in tests: `reorder_within` is off |
| experience block (FS arms) | same position, same header style, same token allowance (counted with the same counter). Content and order are fixed in section 6.4 |
| plan schema | `step4_gemma.PLAN_SCHEMA` (`{"say", "plan": [{action, object, to, arg}]}`), validated by `step4_gemma.validate` |
| repair policy | at most **1** repair re-prompt with the validator's reasons (they name only tools and types). Still invalid → the plan is **not executed** and the presentation fails |
| replanning | **none**. The plan is fixed before execution (open loop at the plan level, closed loop inside the skills). This keeps the cached-plan replay valid (Astra) |
| action limits | `MAX_STEPS` 12 plan steps; the executor's 3 shared retries per step; the step-4 primitive cap and a per-episode wall-clock cap (both values frozen at freeze time and recorded) |
| walked map | **shared baseline infrastructure**, present in every arm: the pre-built `FloorPlan1.map.json` (0.25 m free-space grid, no objects), restored read-only per test. It is not part of the memory treatment |
| within-episode memory | `objmem` working memory (this episode's own sightings) in **every** arm, wiped at episode end |
| perception | the step-3 stack (`TwoStageDetector` + StateEstimator), weights frozen by sha256 |
| honesty | the step-4 gateway (`sim/honest.py`) with its declared exceptions; 0 decision-path oracle reads; evaluator reads only inside `h.scoring()` |

**Isolation audit (Astra: "the no-memory arm cannot silently access persistent experience").** Before the matrix, the
`0` arms run on the 12 dev items twice: once with no `ltm.db` on disk, and once with S\* present at its path and a
populated `sim/step4/honest_memory.db`. The primitive traces, prompts and outcomes must be identical (gate G-iso).
Executor caches checked: `objmem` per-episode tables, `opened_at`, `put_*` rejection lists, the blocked-edge set, and
the search ranking inputs.

## 5. Tasks, units and the split

Dataset: `sim/dataset/general.json` (277 items; each item is one task written in EN and FA). The split is computed by
`sim/final_test_split.py` (master seed `20261010`) and frozen in `docs/final-test/split_manifest.json` with its sha256.

**Resolving "277 instructions".** There are **277 items = 554 bilingual presentations**, but far fewer independent
tasks:

| unit | definition | count |
|---|---|---|
| item | one task, EN + FA twins together | 277 (267 applicable in FloorPlan1) |
| task group | step-5 rule: union-find over items with the same goal signature or the same plan concept key | 209 |
| **task family** (the unit of the split and of the bootstrap) | the task-group links, **plus** same level and same ordered action skeleton ("near-identical template": *pick up the apple* ~ *pick up the bread*) | **57** among the 267 applicable items (16 L1, 32 L2, 9 L3 by majority level) |

So Astra's target of 120 held-out groups (about 40 per level) is **not available** at the family level. Following
Astra's fallback ("use all eligible groups if fewer remain"), every family that is not needed for practice or
development is a test family.

**Rules** (`sim/final_test_split.py`):
1. **Applicability exclusion** (a scene fact, not an outcome): the 10 items the FloorPlan1 applicability gate skipped in
   step 4 (7 goals already true at reset, 3 need a DiningTable). That leaves 267.
2. **Families** as defined in the table above.
3. **Exposure.** Families containing any item an LLM planner has already been run on (the step-4 s6 pilot: 6 items;
   the chooser check: 2; console runs: 2, overlapping) are forced to **non-test**. This forces 3 families, including the
   36-item pick-up family.
4. **Draw.** Within each majority-level stratum, the first ⌈0.25 n⌉ families in a seeded order are non-test (forced
   families count towards the quota). Every other family is test.
5. **Cap.** At most **6 items per test family** (seeded order inside the family), so no family carries more than about
   5 % of the weight. The 26 capped-out items are listed and never scored.

**Resulting split** (manifest sha256 `09c7da23…`; computed on the dataset only, no outcomes involved):

| set | families | items (L1 / L2 / L3) | presentations |
|---|---|---|---|
| **test** | **42** (12 / 24 / 6 by majority level) | **125** (47 / 50 / 28) | **250** |
| non-test = practice pool | 15 | 116 (46 / 47 / 23) | — |
| dev pilot (inside non-test) | — | 12 (4 / 4 / 4) | 24 |
| REPEAT subset (inside test) | — | 30 (10 / 10 / 10) | 60 |
| RELOCATION probe (inside REPEAT) | — | 10 | 20 |

- **L3 is thin.** Only 6 test families have L3 as their majority level, although 28 test items are L3. Per-level
  results are descriptive.
- **Disclosed looks.** Before freezing, P0 computed the counts for non-test fractions of 0.25 and 0.30, with and
  without the exposure rule. 0.25 was kept because it leaves more test families. No planner outcome exists for any
  final-test arm. The step-4 reference-plan executions (s5) exist for every item, and they were not consulted per
  split.
- **FloorPlan1 is development data for the executor** (step 4 was fixed on it). "Held-out" here means held-out
  **tasks** for the planner and memory, in a known kitchen. It never means a held-out kitchen.
- **Dev items** (the 12 in the pilot, and any non-test item) are the only items allowed for prompt checks, contract
  debugging and the cost pilot. Test items are never used for tuning. If a test item reveals a bug, the fix is an
  amendment and the item keeps its outcome.

## 6. Memory, practice budget K and snapshots

### 6.1 What "memory" is
The step-5 long-term memory (`sim/ltm*.py`, `docs/step5-plan.md`) with exactly the **channels that passed their
workflow-B acceptance gates** at freeze time (B3: places, at minimum; B5/B6: habits, lessons, procedures, only if
accepted). The active channel set and every parameter (`sim/step5/params.json` sha256) are written into the snapshot
manifest **before any test call**. If only places pass, the memory is places-only, and the thesis says "location
memory", not "experiential memory".

### 6.2 Practice
- **Who practises: the Gemma system.** It is the specialised system, and it builds its memory from its own attempts.
  Practice is Gemma-planned (G arm settings), with the honest executor, on **practice-pool items only**, in the
  canonical FloorPlan1 initial world, which is restored per episode with the item's setup.
- **K = started practice attempts overall.** Every started attempt counts, including invalid plans, rejected plans,
  timeouts, crashes and inapplicable items, so that "completed" cannot hide extra experience (this tightens step-5
  A6). The practice order is fixed by the master seed: practice-pool items **without replacement**, stratified
  round-robin by level (L1, L2, L3, ...), with each item's language drawn by seed. The action caps are the same as in
  tests. Admission follows the step-5 rule R7, unchanged. The actual practice cost (episodes, primitives, wall time,
  Gemma calls) is recorded.
- **Memory channels** are chosen by their **predeclared** workflow-B acceptance criteria (`docs/step5-progress.md`,
  the B3 criteria and their B5/B6 successors), evaluated on non-test data before this matrix. They are never chosen
  from gains on test items, and the primary snapshot is never chosen from observed gains.
- The privileged terminal goal check is used after each practice episode for admission only (the step-5 honesty
  statement, wording unchanged). Reference plans and goal predicates never enter memory. Replay-derived (gold)
  procedures are **never** loaded (step-5 A7).

### 6.3 Snapshots
- Nested snapshots at **K ∈ {0, 1, 3, 5, 10}** from that one order. **S\* = the K = 10 snapshot of practice order 0** is
  the memory of every `+M` arm in the main matrix. The primary intervention is ten practice attempts, and the
  conclusions are conditional on this budget and this one snapshot.
- **Learning curve (E4):** G+M and O+M at K ∈ {0, 1, 3, 5, 10} on the REPEAT subset, with **3 practice orders** (seeds
  0, 1, 2). Order 0 at K = 10 is shared with the main matrix.
- Tests open a fresh read-only copy (`LongTermMemory.open_frozen`). The sha256 of the snapshot before and after every
  test must be equal (gate). The testlog is separate per arm.
- **Leakage gate:** the snapshot dump must contain no test item id and no test instruction text (EN or FA, after
  normalisation). Locations of objects that test tasks also use **are allowed**: that is the intervention (Astra).
- Every stored claim has an experience trace (`write_log` + evidence rows). Nothing in S\* is hand-written or generated
  by a frontier model.

### 6.4 Experience block for the FS arms (same source, equal token budget)
It is built from the same K = 10 practice episodes of order 0, and it always holds **all ten attempts**. Part 1 is one
compact line per attempt, oldest first, with a fixed template: instruction (in its practice language) · the plan Gemma
made, as `action object→to` tokens · per-step outcome (`ok` or the cause code) · the admission verdict (success /
failure). Failures are kept. Part 2 lists sightings: one line per (object type → receptacle type, times seen, in which
attempts), from WM sightings, newest attempt first, filled until the token allowance of that level's memory block is
used up (the same counter). If part 1 alone does not fit, the plan tokens are shortened uniformly (verb and type are
kept, `to` is dropped), and that is recorded. Retained counts (attempts, outcomes, sightings) are reported per level.
The block is rendered once, cached, and identical for G and O.

## 7. Execution protocol

Planning and execution are separated, so that planner variation and simulator variation can be told apart (Astra).

**Phase A — observation capture (simulator, no LLM).** For every test presentation (and REPEAT / RELOCATION world):
restore the canonical world, apply the item setup, run the fixed start sweep, and record the observation summary, the
told holding and the private world-state hash. Do it twice for the 12 dev items, to check that reset is reproducible
(gate). For `+M` arms, render the memory block from S\* and the post-sweep WM; render the FS block. Every prompt is
written to the prompt cache with its sha256. The observation part is **byte-identical across arms** of one
presentation.

**Phase B — planning (no simulator).** Every (arm × presentation) call runs from the prompt cache in a seeded, randomised
order. Arms are interleaved over time in blocks, and the queue respects rate limits. Raw response, usage, latency, stop
reason, parsed plan, validation, repair and final verdict are all stored. Anthropic primary calls may go through the
Message Batches API (50 % price; same models). Latency is then measured only on the synchronous calls (the pilot and
the REPEAT repeats). On the REPEAT subset every arm gets **3 independent calls** (r = 0, 1, 2); **r = 0 is the scored
call everywhere**, and r = 1, 2 measure planner variance only.

**Phase C — execution (simulator, plans cached, no LLM, so only one GPU user).** Each accepted plan runs from a freshly
restored copy of the same world: restore, setup, start sweep, plan. The world hash must equal Phase A's (checked inside
the scoring scope), and a sweep-summary mismatch is logged as execution variance. The **execution unit** is
(normalised plan, snapshot, chooser, presentation, restored state, observation summary, caps, simulator seed policy).
Plans are compared after normalisation with action order and every execution-relevant argument (`action`, `object`,
`to`, `arg`) kept. Arms whose execution units are identical **share one execution**, because the simulator gets
identical inputs. This is a declared paired-benchmark design. It does not remove execution noise: identical plans get
perfectly correlated outcomes, which mostly affects same-condition model contrasts (Astra). Distinct plans of one
presentation use the same presentation seed where the simulator takes one. Deduplication is validated on REPEAT, where
every unit is also run from independent restores (below). Simulator time is reported as a shared benchmarking cost,
not as the cost of deploying each arm on its own. Execution order is a seeded random
permutation of presentation blocks, with a random arm order inside each block. On the REPEAT subset every distinct
execution unit runs **3 times** (execution variance); run 0 is the scored one.

**Phase D — scoring.** Goal predicates are evaluated inside `h.scoring()` after the episode. The gold file is read only
by the scorer process. Planner processes (API clients, Ollama calls, Codex temp dirs) receive only the prompt and have
no read path to `general.json`. Outcomes go to a sealed journal. No aggregate over test outcomes is computed until
Phase C is complete, or until a stopping rule fires (section 13).

**Relocation probe (E6).** The 10 RELOCATION items are also run in a relocated world (`InitialRandomSpawn`, one fixed
seed declared in the manifest, objects moved) for G0, G+M, O0 and O+M, with S\* unchanged (now stale). It measures
stale-memory harm, and whether a frontier model gains anything from knowing the standard FloorPlan1 layout.

## 8. Metrics

- **SR (primary, authoritative):** the item's goal predicates (`general.json → goal`, state predicates written before
  this study; any plan that reaches the state counts, which covers acceptable alternatives on L3) are true at episode
  end. Invalid plans, refusals, timeouts and unexecuted plans count as **0**. Denominators always include them.
  SR is **simulator-goal success**. It does not require realistic prerequisites.
- **Strict SR (secondary):** goal met **and** every step ok.
- **Known unrealistic successes (reported separately):** slicing goals met while the agent's own action history shows
  no Knife or ButterKnife held at the slice step. AI2-THOR slices empty-handed, and step 4 saw 19 such cases. They count
  in SR, are counted per arm, and a "prerequisite-realistic SR" that excludes them is reported as exploratory.
- **PC / TDS (diagnostic only)** on the **generated** plan (first-pass and final), against the item's reference plan,
  with `sim/eval.py` `pc_score` / `tds_score`. Step label = `action:object:to:arg`. `search_for`, `look_around` and
  `scan_for` are stripped from **both** sequences (high-level tools search on their own in the honest executor; locked
  now). PC is scored separately from the executed trace. A successful alternative plan can score badly on PC; TDS
  ignores order.
- **Plan validity:** schema-valid first pass, accepted after normalisation, repaired, rejected; invalid arguments;
  refusals; truncations.
- **Failure categories:** planner side (no JSON, schema, unknown action / type, hand trace, refusal, truncation, timeout,
  valid but wrong = all steps ok and goal false) and executor side (step-4 / step-5 cause taxonomy).
- **Efficiency:** planner calls, input / output / thinking tokens, cache reads, USD at list price on the run date,
  latency p50 / p95 per call and per episode, primitives, walking metres, opens, retries, all reported **over all
  attempts and among successes**. **Cost per success** = total spend (failed attempts, repairs and amortised practice
  included) / successes. Practice cost is Gemma's local time and energy.
- **Local footprint (Gemma path):** peak VRAM (nvidia-smi at 1 Hz) and GPU energy (integrated `power.draw`), declared
  as GPU-only. There is no cloud energy comparison. "On-device" is claimed only if the whole inference and retrieval
  path ran locally, which is checked.
- **Memory mechanism coverage:** fraction of test targets with a non-EXPLORE `where_is` candidate, gate-pass and
  application counts, stale visits.
- **Reference-plan baseline under this executor (context):** the P0 arm's honest reference-plan SR on the same items.
  Step 4 measured .749 on all 267. It is **not** a ceiling: an alternative plan can succeed where the reference plan
  fails, and `+M` also changes how well the executor does.

## 9. Statistics

1. **Estimand.** For arm a, `SR(a)` = the mean over the 125 capped, unexposed test items of the item score, where the
   item score = the mean of its EN and FA presentations (r = 0, execution run 0). Items are equally weighted, and the
   family cap limits any family to 6 items. **Sensitivity:** the family-weighted SR (each family's mean counts once).
2. **Interval (display).** Paired **cluster bootstrap over the 42 test families**: 20,000 resamples with replacement,
   seed 20261010. Each resample keeps every item, language and arm of a family together, so arm pairing is preserved.
   The interval is the percentile 95 % interval of the difference (`final_test_power.cluster_ci`). Family-level
   win / tie / loss counts are reported alongside.
3. **Tests (decisions).** One-sided **wild cluster bootstrap with the null imposed** (`final_test_power.wcr_p`):
   family totals, Webb 6-point weights, B = 9,999, a cluster-robust t statistic with the small-sample factor F/(F−1),
   the p-value counted with the observed statistic included, and the null boundary itself imposed (δ0 = −5 pp or 0).
   Two-sided p = min(1, 2·min(p_greater, p_less)). **H1 wording** (`final_test_power.h1_decision`) needs **both** the
   wild test (p < .025) and the percentile bound to agree, because both were measured anticonservative at the margin
   when discordance is low and clustering high (calibration, item 6):

   | result | wording |
   |---|---|
   | superiority test rejects and L > 0 | Gemma+M **superior** to Opus 5.5 without memory, in FloorPlan1, under this contract |
   | non-inferiority test rejects and L > −5 pp | Gemma+M **non-inferior within 5 pp**; if also U < 0 and the "lower" test rejects, add "with statistically lower SR" |
   | "materially inferior" test (H0: D1 ≥ −5 pp) rejects and U < −5 pp | Gemma+M **materially inferior** (worse by more than 5 pp) |
   | anything else | **non-inferiority unresolved** (inconclusive) |

   A difference exactly at the margin does not establish non-inferiority.
4. **Multiplicity.** H1 stands alone (one-sided .025 per test, fixed sequence). The 8 secondary tests (H2a–c, H3a–c,
   H4a–b) form one Holm–Bonferroni family at α = .05, using the same wild test with the null values from section 2;
   H3c enters as min(1, 2p). That family's error control is **separate** from H1's; there is no study-wide .05 claim.
   I and every E are estimates without tests.
5. **Variance decomposition** (REPEAT subset): planner variance (3 calls: plan-identity rate, SR spread) and execution
   variance (3 runs per cached plan: outcome agreement), reported separately.
6. **Precision and calibration (published now; not a sample-size rule).** `python sim/final_test_power.py` simulates
   the actual family sizes across discordance d ∈ {.1, .2, .3}, family ICC ρ ∈ {0, .3, .6} and EN/FA twin agreement .7
   (the results are in section 2). `--calib` gives the type-I error at the margin (true D1 = −5 pp, nominal .025,
   1,000 simulations). The wild test gives .022–.042 when d ≥ .2, and .039–.102 when d = .1. The percentile interval
   gives .031–.099. The conjunction rule in item 3 is the mitigation, and the residual inflation in the
   low-discordance, high-clustering regime is disclosed. The pilot's discordance and ICC estimates are reported next to
   this table. N is fixed at all eligible test items and is never increased after outcomes are seen.
7. **Per level and per language:** rates with family-bootstrap CIs, descriptive. The thesis's per-level metrics (L1 SR,
   L2 PC, L3 TDS) are reported as diagnostics next to SR.

## 10. Cost pilot and spending ceilings

- **Smoke (earlier stage, already capped):** Anthropic ≤ 1 USD total, OpenRouter ≤ 0.10 USD. Keys come only from the
  environment (`ANTHROPIC_API_KEY`; OpenRouter from the file named by `OPENROUTER_ENV_FILE`), and they are never
  written anywhere. If `ANTHROPIC_API_KEY` is unset, the live Anthropic parts are skipped and that is reported.
- **Cost pilot (needs Arash's go):** the 12 dev items × EN + FA × {G0, G+M, O0, O+M, S0, S+M}, synchronous calls,
  memory from a dev-only snapshot (practice on non-test items, same procedure). It measures billed input, output and
  thinking tokens, retries, repairs and cache use, plus Gemma latency and VRAM. **Hard ceiling 3 USD**; the runner
  stops at the ceiling.
- **Illustration, to be replaced by the pilot:** at 3,000 input and 1,000 output tokens per call, Opus 5.5 costs about
  $0.032 per call, Sonnet 5.5 about $0.016 and Haiku 4.5 about $0.008. Main matrix: O (3 arms) 750 calls ≈ $24,
  S (2 arms) 500 ≈ $8, H (2 arms) 500 ≈ $4, plus REPEAT repeats ≈ $5, plus the learning curve ≈ $6. That totals
  **≈ $47 before repairs and before real thinking usage**, and roughly half that with Batches.
- **Matrix budget, reserved first.** The pilot projection × 1.5 for the **complete primary comparison** (the six core
  arms, with repairs, transport retries and the REPEAT repeats) is put to Arash, who approves and reserves it before the
  matrix starts. The control and exploratory arms get their own, separate allowance, and they are cut first (3.1). The
  runner keeps a running total per provider.
- **OpenRouter (exploratory):** the remaining ~1.4 USD allows at most one cheap model on the REPEAT subset (30 items ×
  2 languages × {0, +M}, 1 call each), with a hard stop at 1.00 USD total spend. Otherwise it is omitted, and the
  thesis says so.
- **Codex CLI:** no per-call dollar price (subscription). Tokens are recorded from the event stream.

## 11. Contamination controls

- **Masking gate:** no cached prompt may contain `FloorPlan`, `AI2-THOR`, `ai2thor`, `iTHOR`, `Unity`, `_physics`, or
  an AI2-THOR objectId pattern (`[A-Za-z]+\|[-0-9.]+\|`). This is checked on every prompt before Phase B.
- The instruction set was written for this thesis (2026-09) and was never published, so it is unlikely to be in any
  pretraining data. That is a mitigation, not a proof.
- A frontier model could still know the standard FloorPlan1 layout. Plans carry no coordinates, so that knowledge can
  only act through plan structure (for example an extra `open Fridge`). The relocation probe (E6) is the check; it
  mitigates rather than proves the absence of contamination (Astra).
- Gold isolation: section 7 Phase D.

## 12. Gates before any test outcome is opened

All are recorded in the ledger with their evidence. Any failure blocks the matrix.

1. **Freeze:** code hashes (all `sim/` modules on the planning and execution path), prompt version `ft-v1` text,
   `PLAN_SCHEMA`, `params.json`, detector weights, map file, S\* manifest, split manifest, model ids, Ollama digest and
   version, CLI version: all in one `freeze.json` with sha256.
2. **Bilingual retrieval:** for the 12 dev items, the FA instruction yields the same memory-block records as the EN one
   (`concepts` is language-normalised).
3. **Honest perception:** 0 violations, 0 oracle reads and 0 map-build reads in all pilot episodes.
4. **Reset reproducibility:** identical world hashes and sweep summaries on the twice-captured dev items.
5. **Evaluator correctness:** reference plans on the dev items reproduce their step-4 outcomes up to known variance;
   an empty plan scores 0 on every applicable item; goal predicates are evaluated only in `scoring()`.
6. **Masking, isolation (G-iso), leakage** (6.3) and **snapshot hash** gates.
7. **Spend guard:** dry run of the budget counter on the pilot's real usage numbers.
8. **Family boundary audit (Arash):** the manifest lists every family. Arash checks representative families and the
   boundary cases: near-identical tasks that escaped grouping through a different plan skeleton or level label, and
   transitive links that merged different tasks (for example the 15-item slice family, which spans L1–L3). Any
   regrouping is a dated amendment that re-runs `sim/final_test_split.py` with a new manifest sha, and it happens
   before any test call.

## 13. Exclusions, missing data and stopping rules

- **Pre-specified exclusions only:** the 10 inapplicable items and the 26 capped-out items. Nothing else is excluded
  after the fact.
- **Planner infrastructure failure** (transport errors after 3 retries): re-queued in up to 2 later sessions. If it is
  still missing, it counts as failure in the primary analysis. Infrastructure failures are reported separately per arm,
  together with a **paired sensitivity analysis** that drops every presentation where either arm of the contrast had
  one. A cloud transport failure scored as 0 can favour Gemma without showing better planning, so this sensitivity is
  reported next to H1.
- **INVALID episode** (step-4 audit violation, oracle or map-build read): this is an integrity defect, not a task
  failure. Execution stops for that presentation, the cause is corrected (an amendment), and **all arms** of that
  presentation are re-run as matched executions. A presentation that cannot be made valid is reported by count and
  dropped from every arm, which keeps the pairing.
- **Simulator crash / hang:** the episode is re-run from a restored world (max 2). If it still fails, it counts as
  failure and is reported.
- **Fail fast (abort the run):** any snapshot hash change, any decision-path oracle read in a valid-marked episode, any
  prompt failing the masking gate, any planner process found with file access to the gold. Fix, write the amendment,
  restart the phase. Outcomes already collected are kept but labelled.
- **Budget exhausted:** the primary comparison's budget is reserved in advance (section 10), so it should not run out.
  If it does, the run stops, and the results are reported as an **incomplete study**: estimates over the presentations
  that were completed, without the confirmatory H1 decision. Exploratory work is always stopped first.
- **No early stopping for efficacy or futility.** Results are not looked at during the run. Only the integrity
  counters (violations, hashes, spend, error rate) are monitored, every 20 minutes during long runs.

## 14. What can and cannot be claimed

| result | allowed claim | not allowed |
|---|---|---|
| H1 superior | "In FloorPlan1, on these held-out task families, the configured Gemma 4 E2B system with frozen memory from one 10-attempt practice history had higher task success than Opus 5.5 (effort as configured) without memory, under this contract." | "Specialisation beats model complexity" in general; other kitchens or real homes; "Gemma is as capable as Opus"; "memory made Gemma's planning competitive" (H1 is whole-system) |
| H1 non-inferior | the same, with "**was non-inferior within 5 pp** to" (never "matched" or "equal") | the same |
| H2 positive for all models | "Memory improved each named configuration here" | "Memory helps Gemma no more than the others" (a positive H2 does not exclude a larger Gemma benefit, and an estimate-only interaction cannot show it is absent) |
| H3: O+M > G+M | "With the same memory, Opus 5.5 was still better": stated plainly, even if H1 holds. Use model names, never "the larger model" (frontier sizes are unknown) | hiding it behind H1 |
| H4: no difference | "**No difference was established** between the structured memory and the same experience given as records" (no equivalence test was preregistered) | "structured memory adds nothing"; claiming the memory architecture as the reason for H1 |
| inconclusive / inferior | stated in the abstract, with the CI | dropping or softening it |

Further bounds: **one kitchen** means within-kitchen transfer to held-out tasks; held-out kitchens need their own study
(step-5 B8). The memory gives location knowledge by construction, so the information advantage is part of the
treatment. Tasks whose goal is fully stated (all of this dataset) are not preference-dependent tasks. No cloud energy
claim. "On-device" only as verified (section 8). Results are for the frozen executor, and honest-executor failures
affect every arm. The headline is about specified, configured systems, one frozen experience history, these task
families and one kitchen. It is not about model complexity itself.

## 15. Archive (kept with the results)

`freeze.json`, the split manifest, the snapshot S\* with its manifest (and the learning-curve snapshots), the prompt
cache, raw responses, usage, latency, plans, validation, execution journals, scoring journal, Codex event streams, CLI
and Ollama versions, nvidia-smi logs, the spend ledger, and every amendment. No API key appears in any of them; a
grep gate checks the archive for key prefixes before it is written.

## 16. Amendments

All amendments below were made **before any final-test planner call on a test item, any practice episode and any
test outcome existed**, so none of them is informed by results. Source: the P3 fairness review
(`docs/final-test/progress.md` P3; Astra `docs/final-test/_astra/P3-fairness.*`).

**A1 (2026-10-10, P2 finding, recorded in P3): Codex output schema.** The bare `PLAN_SCHEMA` is rejected by the OpenAI
strict output mode (HTTP 400, every Codex model, P2 smoke). The Codex arms now receive the same transformed schema the
Anthropic arms receive (`additionalProperties: false` on every object, type lists as `anyOf`;
`codex_provider.strict_schema`). The type-list part is semantically identical; forbidding extra keys is a stronger
constraint than the bare schema, and its effect is unmeasured (Astra P2). Gemma's Ollama `format` still carries the
bare schema (step-4 behaviour). Whether Gemma gets the same strict transform (`OllamaPlanner(strict_schema=True)`,
Astra P3 prefers it) is decided on the 12 dev items in the cost pilot, under 3.1's equal-interface rule, and recorded
as a further amendment **before practice starts**. The JSON-only OpenRouter arm stays exploratory and disclosed.

**A2 (2026-10-10, P3): Codex isolation from the user's CLI configuration.** In P2 every Codex call silently loaded
the user's `~/.codex/config.toml` (reasoning effort `high`, a personality and a service tier) and the user's
execpolicy rules, which 3.4 did not record. Every Codex call now adds `--ignore-user-config --ignore-rules` and pins
`-c model_reasoning_effort="high"` (the value P2 ran with; Astra P3). This replaces 3.4's "CLI defaults (recorded)"
for the effort; the effective flags are part of the recorded decoding and of the cache key. Checked live once with
`gpt-6-luna` (valid plan, 0 tool events). The CLI's own scaffolding (about 16 k input tokens per call) is still not
controlled; the arms stay exploratory and separate.

**A3 (2026-10-10, P3): call counts and the cost illustration (section 10).** The illustration under-counted the
design. The enumerated call plan (`python sim/final_test_callplan.py`, from the frozen manifest) gives these
first-pass calls: Opus 1,810 (main 750, REPEAT repeats 360, learning curve 660, relocation 40), Sonnet 740, Haiku 740,
Gemma 1,810, OpenRouter 120, Codex 1,440 (4 models). At the illustrative 3,000 input + 1,000 output tokens this is
about Opus $58 + Sonnet $12 + Haiku $6 ≈ **$76 before repairs** (not $47), roughly half with Batches. The **primary
comparison** (six core arms incl. their REPEAT repeats) is 740 Opus + 740 Sonnet calls ≈ $35 at the same assumption.
The protocol is unchanged; only the illustration is corrected. The pilot measurement still replaces it (section 10).

**A4 (2026-10-10, P3): added gates before practice / the matrix** (they only tighten section 12):
- **12.9 Gemma model identity and provenance.** Ollama reports `parent_model =
  gemma4-validation-20260929:e2b-imatrix-attention-q6-q8-draft2` for the tag `gemma4:e2b-it-q4_K_M` (step-4 ledger,
  s6 provenance flag, still open). Section 1's "confirmed by Arash 2026-10-06" is therefore not enough. Before any
  practice episode: record `ollama show --modelfile` and the full digest, and have Arash state where the weights came
  from. A digest fixes the artefact, not its lineage (Astra P3). If it is a local derivative (for example a custom
  imatrix quantisation), every claim names it as such ("a locally quantised derivative of Gemma 4 E2B"), and any
  tuning on this project's data would make G a different system, which must be declared before practice.
- **12.10 Workflow-B data.** Channel acceptance (B3, B5, B6) is training: it may use **only final-test non-test
  items** (`split_manifest.json → nontest`). The step-5 manifest of `sim/step5_eval.py` is filtered to those items
  before any B run; no test-family item may appear in any B journal.
- **12.11 One prompt renderer.** Practice, the cost pilot and the test all render prompts through
  `sim/planners/contract.PlannerContract` (memory block after the command, under `MEMORY_HEADER`), with
  `reorder_within` off. The step-5 plan's alternative (memory lines appended into the observation summary, Gemma
  reordering) is not used for anything that feeds S\* or a test.
- **12.12 Type list.** The prompt's type list is the class list of the frozen detector weights (49 types), read from
  the weights. It is never derived from reference plans. (The P2 smoke used 36 types collected from reference plans;
  that was a smoke shortcut, and no smoke output is test data.)
- **12.13 Cache namespaces.** Each phase has its own response-cache namespace (`pilot-v1`, `practice-v1`, `test-v1`);
  the pilot never reuses a cached reply, and the test namespace starts empty. Independent repeats stay independent
  (distinct `replicate` in the key). A per-call `max_tokens` override is part of the cache key (P3 adapter fix).

**A5 (2026-10-10, P3): a representative cost pilot.** The pilot's dev-only memory snapshot is built with the same
budget as S\* (K = 10 started attempts on non-test items that are not dev items), so the memory-block length matches the
test. Arms are interleaved per presentation, so a stop at the 3 USD ceiling still measures every arm.

## 17. Astra review

Astra (`gpt-6-astra` via `codex exec`, isolated temp dir) reviewed draft v0.9 once, as a methods referee. Prompt, raw
output and answer: `docs/final-test/_astra/P0-prereg.{prompt,raw,}.txt`. Every point was adopted:

| # | Astra's point (short) | what changed |
|---|---|---|
| 1 | Memory acts in planning **and** in search: H1 is whole-system; H4 confounds representation with executor access | FS arms now use the same frozen `LtmChooser`; new exploratory `/p` replay ablation; the wording "only planning differs between models **within a memory condition**" (1, 3.2, 3.3) |
| 2 | Keep families as the unit; 42 is acceptable but gives no power guarantee; audit the boundaries; family-weighted sensitivity | gate 12.8 (Arash's family audit, before the matrix); family-weighted SR sensitivity (9.1); estimand restricted to the capped, unexposed set |
| 3 | The H1 table overlapped and wrongly called any significant disadvantage a refutation | new 4-row table: superior / non-inferior (± "statistically lower") / materially inferior / unresolved; margin equality is not non-inferiority (9.3) |
| 4 | Precision claims too confident; simulate with the real family sizes; drop the "laptop user" justification | `sim/final_test_power.py` (precision, power and calibration with the 42 real family sizes); margin restated as a prespecified tolerance (2) |
| 5 | "p-values from the same bootstrap" was not executable | frozen null-imposed wild cluster bootstrap test (`wcr_p`) with its statistic, weights, tails and boundary; H2 one-sided, H3a/b and H4 two-sided, H3c against −5 pp via min(1, 2p); secondary control is separate from H1. Calibration found the percentile CI anticonservative at the margin, so H1 now needs **both** the test and the interval (9.3, 9.6) |
| 6 | FS dropped whole episodes, so it was not equal experience | FS always holds all ten attempts in compact form, sightings fill the rest; renamed "same-source, equal-token-budget experience control"; retained counts reported (6.4) |
| 7 | K = 10 is fine if chosen before outcomes, but "fairest" is advocacy; count every started attempt; freeze sampling and channel acceptance | K = started attempts; sampling without replacement, round-robin by level, seeded language; channels chosen by their predeclared B-gate criteria; neutral wording (6.2, 6.3) |
| 8 | Decoding asymmetry: fine for deployment systems, not matched compute; cut exploratory arms before lowering effort; equal dev-only interface opportunity; the prompt was Gemma-developed | 3.1: "deployment configurations, not matched compute"; the cut order; the dev-only interface check for every model; disclosure |
| 9 | Shared execution is acceptable only for execution-equivalent units; validate it; report it as shared cost | full execution-unit key, common presentation seed, validation on REPEAT with independent restores, cost wording (7 C) |
| 10 | .749 is not a ceiling; strict SR does not catch knife-less slicing | "reference-plan baseline", not a ceiling; SR = simulator-goal success; known unrealistic successes counted separately, plus an exploratory prerequisite-realistic SR (8) |
| 11 | Spending stop and failure coding undermine the fixed sample | the primary budget is reserved first, exploratory cut first, "incomplete study" if it still runs out; paired infrastructure-failure sensitivity; INVALID → correct and re-run matched, never scored as task failure (10, 13) |
| 12 | Claims exceeded the tests | claim table rewritten: "non-inferior within 5 pp" (never "matched"), "no difference established", model names instead of "larger model", whole-system wording (14) |

Deviations from Astra's earlier design review (`docs/research/final-test-astra-review.txt`) that remain, with
reasons. Astra saw all three in the referee review:
- **K = 10 is primary, not K = 5.** It was fixed before any outcome exists. Astra: "K = 5 has no intrinsic
  methodological privilege". K = 5 is still on the learning curve.
- **The task family includes the action skeleton.** This follows "near-identical templates together" literally, and it
  gives 42 test families, not 120 groups. That is the honest count; Astra: keep it, and do not switch units to
  manufacture independence.
- **Shared execution for identical plans**, with the conditions of point 9.

### A6 (2026-10-10, before any outcome) -- OpenRouter arm dropped (Arash)
Arash's budget decision: no OpenRouter model. Planners in the study are ONLY Gemma 4 E2B (local, Ollama), the Anthropic
API models (claude-opus-5-5, claude-sonnet-5-5, claude-haiku-4-5-20251001) and the Codex CLI systems (gpt-6-sol,
gpt-6-astra, gpt-6-luna, gpt-5.6-terra; exploratory, reported separately). Every OpenRouter row in sections 4, 10 and the
call plan is void; no OpenRouter result is reported. The A1 decision (Gemma strict vs bare JSON schema) is being made on
non-test practice items by sim/final_gemma_schema_test.py and is recorded as A7.

### A7 (2026-10-10, before any outcome) -- Gemma gets the STRICT JSON schema (decides A1)
Test: sim/final_gemma_schema_test.py, 30 NON-TEST practice items x EN/FA = 60 presentations per arm, fixed fake
observation, cache off (docs/final-test/gemma_schema_test.json). Bare vs strict schema through Ollama `format`:
identical outcomes on every presentation (valid 56/60 both, first-pass 43/60 both, 13 repairs both, EN 28/30 and FA 28/30
both, mean PC .616 both; per level L1 22/22, L2 26/30, L3 8/8 valid in both). Strict was slightly faster (mean 0.83 s vs
1.13 s, p95 1.69 vs 1.90 s). Decision: Gemma uses the strict schema, so every arm shares ONE canonical output contract
(Astra's P2/P3 advice); no measured cost to Gemma. `OllamaPlanner(strict_schema=True)` is the final-test setting.

### A8 (2026-10-10, before any outcome) -- gate 12.9 (Gemma provenance) answered by Arash
Arash: the Gemma weights are a publicly available quantized build (q4_K_M) of Gemma 4 E2B, downloaded from the internet
(not a local fine-tune). Ollama tag gemma4:e2b-it-q4_K_M, digest b37049369adf; Ollama's parent_model string
"gemma4-validation-20260929:e2b-imatrix-attention-q6-q8-draft2" names that upstream quantization. Thesis wording: "a public
4-bit (q4_K_M) quantization of Gemma 4 E2B served by Ollama"; cite the download page/source URL once confirmed.

### A9 (2026-10-11, before any acceptance, cost-pilot, practice or held-out test outcome) -- the memory treatment, defined (C2-design)
Substantive amendment to the intervention and its learning policy (sections 6.1, 6.2, 8). Full text: `docs/step5-plan.md` s13 "C2-design";
Astra: `docs/final-test/_astra/C2-design.*` (9 decisions: 1 adopted, 8 changed, all applied). Evidence that informed it, disclosed:
development outcomes C1 (a nonempty places block changed Gemma's plans in 4/6 presentations) and B2 (hint goals: Cup -51 %, DishSponge +50 %
primitives), plus the structural fact that family construction makes exact whole-intent procedure matches between non-test and test items rare.
- **Candidate treatment** = `places` + `procedures` (whole-intent with the frozen gate, plus role-level FRAGMENTS of admitted Gemma practice
  procedures, exact role key, support >= 1, rendered as conditional, qualified advice with prerequisites). **Lessons are not part of the
  treatment** (recorded in practice, never rendered or applied). Habits, negatives, dense retrieval: dormant.
- **hint_goals** (executor search policy): OFF, decided before acceptance and held fixed; it can only change through one prespecified fix +
  one screen (C3b) before C4, on Arash's order.
- **Practice reads memory online** through the same renderer and gate (6.2 "G arm settings" means G+M with the live practice LTM).
- **Token budget**: block retention by the maximum REAL token count over Gemma and Anthropic x EN/FA (identical record ids everywhere); freeze
  gate: every S\* block <= its level budget in both tokenizers; the FS block (6.4) is filled with the same counter.
- **Acceptance (C4)**, on non-test minus dev items only: practice on families A, evaluate on families B (seeded split 20261011); arms G0 /
  G+P / G+PF; procedures added iff integrity, coverage >= 10 % of B presentations (mean of 3 practice orders) and a harm screen (net paired
  loss <= max(1, 0.05 n_B) item-equivalents on validity and order-0 executed SR) pass. These are engineering tolerances, not evidence of
  non-harm or benefit. Fallback: places-only, and 6.1's wording "location memory".
- **Claim wording**: "experiential memory" only if procedure lines render on >= 10 % of test presentations; any procedure claim is about
  transfer of familiar components to held-out task families. H1 stays a whole-system comparison; nothing here is a test of memory's effect.

### Section 16 amendments, continued (W0-amend, 2026-10-11)
A10-A15 below were all written on 2026-10-11, **before any new outcome**: no held-out test item has been called, practice for S\* has not
started, and none of the runs these amendments govern (W2 re-capture, W3 acceptance, W4 practice, W5 matrix) exists. Development outcomes that
were already known when they were written are disclosed in each one (C3 one-order screen on non-test items; C4 cost pilot on the 12 dev items).
Decisions are Arash's (ledger "Arash decisions after the restart"); Astra was consulted on A12 and A14 (`docs/final-test/_astra/W0-amend.*`,
both CHANGE, both applied as written below). **Family audit 12.8: APPROVED by Arash on 2026-10-11** (all 57 families in
`docs/final-test/families-review.html`; record `docs/final-test/families_approved.json`); no family was regrouped, so the split manifest
and its sha are unchanged and gate 12.8 is passed.

### A10 (2026-10-11, before any new outcome) -- verb-alias fix in the procedure gate (C3)
Found while building C3, fixed before any C3 outcome existed (ledger C3-channels). The concept verbs `pick`, `put` and `find` produced by
`concepts(text)` never equalled the tool names in plan steps, so the frozen whole-intent gate (step5-plan
8.3) was closed for every pick / put / find procedure: a bug, not a design choice. Fix in `sim/ltm_retrieval.py`: one fixed, hand-written
mapping `ROLE_TOOLS` (pick -> pickup, pick_and_place; put -> pick_and_place, put_in; find -> search_for, go_to), tested for the wrong object, a composite step and negation
(`sim/_step5_fragments_selftest.py`). Nothing else in the gate changed. Mirrored in `docs/step5-plan.md` s13 (A10).

### A11 (2026-10-11, before any new outcome) -- the planner runs AFTER the fixed start sweep (C5 H1; restores section 4)
Section 4 registers the user message as the within-episode observation summary after the fixed start sweep. The C3 and C4 runs did not do
this: `step5_live.LiveExecutor` called the plan hook before `HonestSkills.start_episode`, so every C3 / C4 prompt carried an empty summary
(disclosed in pilot.md s9). This amendment does not change the design; it binds the implementation to it:
- in every final-test run (W2 re-capture and rehearsal, W3 acceptance, W4 practice, W5 matrix) the plan hook is called after `start_episode`
  and the fixed start sweep, with the post-sweep `objmem` summary and the told holding in the user message; the memory block (`+M`) is
  rendered from S\* (or the practice LTM, C2-f) and the post-sweep working memory; the FS block likewise;
- the new order sits behind a flag that the final-test runners set; the old order stays reproducible for the C3 / C4 records, which keep
  their label "pre-A11, no observation summary";
- Phase A captures the summary; Phase C, before it executes a cached plan, compares its own post-sweep state and summary with the Phase A
  capture of that presentation (criteria of A14); any breach blocks that execution;
- consequences, all before any test call: the 24 dev presentations are re-captured and their real token counts and costs re-measured
  (Gemma `prompt_eval_count`, free Anthropic `count_tokens`); the C4 projection is re-checked on the new prompts; the end-to-end dev
  rehearsal runs on the new order; W3 runs on the new order (all three orders). The full cost pilot is not repeated (Astra C5 Q2).

### A12 (2026-10-11, before any new outcome; decided after the pilot, disclosed) -- retention: token accounting plus a minimum capacity
Why: under C2-e with the registered budgets the L1 places share (0.4 x 150 = 60 tokens) can never hold one record: header + one place record
costs 63-66 real tokens in EN and 118-125 in FA (max over Gemma / Anthropic; from saved C3 / pilot block data, no outcome field read), so
16/24 `+M` pilot prompts were empty (every L1 one). This was seen in the pilot, so the amendment is post-pilot. It is decided **only from token
counts**, never from success, validity or any other outcome, and it is reported as such in the thesis.
- **Declared minimum capacities N_L = 1 (L1), 2 (L2), 3 (L3) whole place records.** These minima are design choices, not something the token
  measurements establish (Astra W0 Q1). Numerical allowances and any budget increase are computed solely from token measurements.
- **Measurement (W2, before practice):** H = the maximum real header cost and r = the maximum serialised cost of one place record
  (separators included), each over every eligible measured record and every {Gemma, Anthropic} x {EN, FA} combination, on the re-captured
  dev blocks (A11) and the non-test C3 blocks. Additive accounting is not assumed to guarantee fit: the real count of every complete prefix is
  also checked.
- **Allocation:** the places allowance becomes A_L = max(where_share x B_L, H + N_L x r); procedures get B_L - A_L (C2-e otherwise
  unchanged: largest prefix of whole records under the maximum real count, places first, record ids identical across models and languages).
- **Budget:** if H + N_L x r > B_L, B_L is raised to the smallest multiple of 10 that fits, at most 2 x its registered value (the registered
  range of `retrieval.budget_tokens`); A_L is then recomputed from the final B_L. If it still does not fit, that level is declared
  **memory-incapable**, reported as such, exempt from the coverage check below, and not tuned further. A budget increase is a further change
  within the registered range and is recorded as part of this amendment.
- **Minimum-coverage criterion (capacity, not a rate target):** before the freeze, on all 24 re-captured dev presentations and the non-test
  C3 blocks, with the actual complete-block counts: whenever >= N_L place candidates exist, the block holds >= N_L whole place records in every
  tokenizer x language. A failure blocks the freeze. After the freeze the same outcome-blind check runs on every S\* test prompt before
  execution; a failure blocks execution and needs a dated amendment before any further outcome; test-time checks never alter frozen
  parameters. Empty-block rates per level / language / arm are reported, never decisive.
- The computed H, r, A_L and B_L go into `sim/step5/params.json` and `freeze.json` with the reason "A12, token-derived". The rule is
  kitchen-independent; the measured values come from FloorPlan1 blocks and are marked FP1-measured.

### A13 (2026-10-11, before any new outcome) -- procedures: the registered 3-order acceptance on the final implementation (C5 H2b)
C3 was a one-order pilot screen (procedures inactive: coverage 0 %), not the C2-h acceptance. Arash's decision: run the registered C2-h
acceptance as written (step5-plan s13 C2-h, **unchanged**: data, split seed 20261011, arms G0 / G+P / G+PF, 3 practice orders 0 / 1 / 2 x K = 10,
online reads, decision rule integrity -> coverage >= 10 % (mean of 3 orders) -> harm screen) on the **final implementation**, i.e. after A10,
A11 and A12, with all three orders re-run (the C3 order-0 run is pre-A11 and is not reused). Procedures enter S\* only if C2-h passes; otherwise
S\* is places-only and the thesis says "location memory" (with a dormant procedure channel). The C3 outcomes are known and are disclosed as the
reason this acceptance is run on the final implementation; they are not part of the decision.

### A14 (2026-10-11, before any new outcome; decided after the pilot, disclosed) -- gate 12.4 replaced, with its tolerance protocol (C5 H3)
Why: as written, 12.4 failed in the pilot (strict world hash equal in 14/48 paired captures) because one loose object settled up to 0.44 m
differently between restores; nothing else moved. AI2-THOR physics is not bit-deterministic across resets. The gate is replaced; no object
is named and no kitchen constant enters it.
- **New 12.4 wording (Astra C5 Q4):** across repeated independent restores, all task-relevant symbolic predicates, object identities,
  containment / support relations, accessibility states and agent poses must match. Continuous pose differences must satisfy prospectively
  fixed tolerances calibrated on dev restores. Any residual settling must leave task-relevant observations, action feasibility and evaluator
  labels unchanged. Any breach fails the gate.
- **(a) Ceilings first.** Before any calibration restore, a dated amendment (A14a) states numerical position and rotation ceilings justified
  by task geometry, observation thresholds and action / evaluation sensitivity (for example the map grid step, the interaction distance, the
  detector's visibility rule), independently of the observed maximum settling (Astra W0 Q2). A settling of 0.44 m is not acceptable merely
  because only one object moves.
- **(b) Calibration (W2):** the 12 dev items x 5 independent restores each (60 restores: restore, setup, fixed start sweep), all 10 pairs per
  item; per object the position and rotation delta. Tolerances tau_pos / tau_rot = the observed maxima rounded up to 0.05 m / 5 deg, **provided
  they do not exceed the ceilings; otherwise the gate fails** (the remedy is then a reset fix applied in every arm, as a new amendment, not a
  wider tolerance). Agent pose must match exactly. Ceilings and tolerances are written to `freeze.json` before (c).
- **(c) Validation re-run:** each of the 24 dev presentations is captured twice from fresh restores (48 captures, 24 comparison pairs).
  PASS iff in 24/24 pairs: (i) the symbolic part is identical (object identities and types, receptacle parents, containment / support, open /
  toggled / sliced / dirty / filled / broken states, the set of objects seen in the sweep); (ii) the post-sweep observation summary is
  byte-identical; (iii) evaluator labels at reset are identical; (iv) the reference-plan precondition checks give the same verdicts
  (feasibility); (v) every continuous delta is within tau. Any breach fails; a fix is a new amendment and the protocol restarts at (b) with new
  restores.
- **(d)** Before each Phase C execution, its post-sweep state is compared with its saved Phase A capture under the same criteria; any
  breach blocks that execution (logged; section 13's crash / re-run rule applies). The strict world hash stays recorded as a diagnostic.
- The sample sizes (60 restores, 24 pairs, 100 % pass) are an engineering gate. The pairwise comparisons are dependent; they establish
  neither universal reproducibility nor a small failure probability, and the thesis says so.

### A15 (2026-10-11, before any new outcome) -- Anthropic ceilings and the cost guard (C5 H7; replaces section 10's reserve)
Arash's approval, one total per scope, replacing the C4 proposal of a 15.07 USD primary reserve (Astra C5 Q7: below the core projection
with the thinking tail):
- **Hard ceilings:** **25 USD** for the six core arms incl. their REPEAT repeats (projection 10.05 USD + thinking tail 11.1 at +500 output
  tokens per call + retries and revalidation) and **45 USD for the whole call plan** (projection 21.18 + tail 22.2). Controls and
  exploratory arms are cut first (3.1). Not counted: Codex (subscription), Gemma (local), simulator time, free `count_tokens` calls.
- **Enforcement from the first call:** the SQLite spend ledger reserves the worst case of each block before the block starts and refuses a
  block that would cross a ceiling. The key comes only from `THESIS_ANTHROPIC_API_KEY` through `sim/planners/keys.py` and is never printed
  or written (this replaces section 10's `ANTHROPIC_API_KEY`).
- **Projected-completion check** from the first call, and every 60 Opus calls a comparison of the running cost per Opus call with the pilot's
  0.0089 USD; above 2 x, the run pauses for Arash's decision. Pauses happen only at block boundaries, so arms stay balanced, and resumption is
  outcome-blind (only spend, tokens and integrity counters are looked at). If a ceiling is reached anyway, section 13's "incomplete study"
  rule applies.

### A12a (2026-10-11, before any W2 re-capture or A12 re-measurement; token accounting only) -- the allocation needs the block's fixed lines
Found while building E2 (ledger E2-retention): a place record is never alone in a block. The block pays its fixed first line (`head`) once,
and each task-type section pays its own "Where to look for X:" line plus the EXPLORE line (`s`) in addition to the records. With H = header only
and s = 0, the literal A12 allocation under-provisions L3 (2 of 3 records fit on the dev fixture) and the coverage check catches it. Refinement,
decided from token counts only: H_eff = H + head (max real counts, both tokenizers, EN/FA) and the need of N whole records in separate sections is
H_eff + N x (r + s); `a12_allocation(H_eff, r, s=s)` computes A_L = max(where_share x B_L, H_eff + N_L x (r + s)); everything else in A12 is unchanged
(budget raise rule, incapable levels, complete-prefix check, coverage criterion). The measurement reads place candidates of the frozen snapshot,
rendered alone, for every non-test instruction in EN and FA; no outcome field is read. Disclosure: this is a post-pilot refinement of A12, measured on FloorPlan1 blocks. Astra W2 check: CONFIRM, provided `head` is counted once, `s` covers both section lines (the "Where to look" line and the EXPLORE line) and the assembled blocks pass the existing token-budget and coverage checks in both tokenizers and both languages (they do: complete-prefix check, A12).

### A14a (2026-10-11, before any A14 calibration restore) -- numerical ceilings and the summary-coordinate rule
No restore for A14 b has been taken at the time of writing. The ceilings are fixed before calibration; they are informed by implementation scales and by
the pilot's known 0.44 m settle (they are not independent of that knowledge, and "a priori" is not claimed).
- **Position ceiling 0.10 m; rotation ceiling 15 deg.** These are design limits, not guarantees of unchanged cell assignment, container identification,
  visibility or task equivalence; those properties remain subject to the unchanged checks (symbolic part, summary, labels, feasibility). Scales they were set against:
  the navigation grid cell is 0.25 m (`honest_map.CELL`), "same container" uses a 0.25 m radius (`ID_SAME_D`), a distinct viewpoint needs 0.20 m, 20 deg of yaw or
  15 deg of pitch (`objmem.VIEW_*`). The pilot's 0.44 m settle exceeds the ceiling and would fail.
- **Summary rule (a relaxation of the byte identity of A14 c-ii, stated as such).** The observation summary lines contain the agent's depth-derived (x, z)
  estimate of each sighted object, to two decimals. Byte identity imposes an additional constraint on these reported coordinates even when the physical displacement
  is within tau. The comparison is therefore: replace only the designated "(x, z)" pair of each line by a placeholder; the lines (types, states, places, step numbers,
  order, count) must then be byte-identical; each coordinate pair must satisfy the Euclidean distance between the corresponding reported (x, z) pairs <= tau_pos_m;
  a malformed or unmatched coordinate field fails. Calibration takes the maximum over BOTH the positional discrepancy of the objects and this summary-coordinate
  distance, with the same procedure later used in validation and in Phase C; rounding (0.05 m) and the ceilings apply afterwards. The planners keep seeing the
  full Phase A summary text unchanged; only the comparison changes. Astra W2 check (docs/final-test/_astra/W2-rehearsal-A14a.*): CHANGE applied as above.
- Unchanged from A14: 12 dev items x 5 restores (10 pairs per item), tau = observed maxima rounded up to 0.05 m / 5 deg, failed if a symbolic breach or tau
  exceeds a ceiling (final), 24 validation pairs at 100 %, one-shot validation, agent pose exact, labels and feasibility identical, strict world hash a diagnostic.

### A14b (2026-10-11, after the first A14a calibration FAILED; chosen after inspecting its results; disclosed) -- agent pose precision and the scope of the summary comparison
**What happened.** The A14a calibration (12 dev items x 5 restores, 120 pairs; the registered one-shot) FAILED: 23/120 comparison pairs breached its symbolic criteria. Agent-pose
discrepancies occurred in 12 pairs and summary discrepancies in 21 pairs, with 10 pairs exhibiting both. Objects, flags, parents / contents, accessibility, seen types, evaluator labels and
reference-plan feasibility were identical in all 120 pairs; the largest object position delta was 0.04 m and the largest rotation delta 0.029 deg (the pilot's 0.44 m settle did not occur).
(a) The agent pose differed by at most 4e-4 (m / deg), e.g. yaw 317.62 vs 317.6201: "exact to 4 decimals" fell on a rounding boundary. (b) The summary differed only in rows the planner's tracker keeps for
ambiguous fixtures (e.g. "Cabinet [closed] on/in CounterTop ... step 8" vs "Cabinet [closed] on/in Bowl ... step 7", Floor rows). The failed file is kept as
`docs/final-test/matrix/rehearsal-v1/calib/reset_tolerances.A14a-FAILED.json`; the failed result and the A14a rules are reported in the thesis. A14b was chosen after inspecting these results and changes the
operational definition of equivalence. Fresh calibration and validation assess only the amended definition; they do not retrospectively establish that A14a passed.
**Changes (everything not named here is unchanged from A14 / A14a, incl. the ceilings 0.10 m / 15 deg):**
1. **Agent pose** is compared with an absolute tolerance of 0.001 m on each position coordinate and an angular distance of at most 0.01 deg (geodesic, wraparound handled); horizon and the standing flag stay exact.
   These tolerances were selected after inspecting the failed calibration and are frozen before new captures; they are not calibrated.
2. **Summary scope (a narrower claim, stated as such).** The A14a summary rule (placeholder-equal text, Euclidean coordinate distance <= tau_pos) is applied to the summary rows whose leading object type is
   TASK-RELEVANT by a frozen dependency rule, read from the captured state itself: T0 = the object / destination types of the item's reference-plan steps (feasibility record) plus the object types named by its goal clauses
   (labels); RELEVANT = T0 plus the types of every transitive container ancestor (parent-receptacle chain) of every instance of a T0 type. The relevant set is part of the symbolic state and must be identical across restores.
   This gate establishes equivalence of specified task-state properties, not equivalence of planner inputs. Excluded summary rows remain visible to planners and may affect their decisions. The rule was written down before it was evaluated on any data.
   Astra preferred comparing all planner-visible rows; that choice would have failed the attempt on tracker nondeterminism that is not a property of the reset, which is why the narrower, explicitly labelled claim was taken.
3. **Diagnostics, never criteria.** For every comparison the number of differing non-relevant summary rows and their object types are stored; calibration and validation report pairs affected, rows differing and the
   pairs per type, separately for the failed A14a calibration (computed afterwards under the amended rule), the new calibration and the validation. The 120 pairs are not 120 independent experimental units; the validation
   consists of fresh captures, and independence is claimed at the item level only if the items are also held out (they are not: the dev items are reused).
4. **Freeze.** Before any new calibration capture, A14b's comparison rules, scope, tolerances, sampling procedure and failure consequences are frozen. Calibration may determine only the quantities designated for calibration
   (tau_pos_m, tau_rot_deg). Failure of the new calibration or of the validation terminates this gate attempt; no further revision and retest may qualify this experiment under this protocol.
Astra W2 check: `docs/final-test/_astra/W2-rehearsal-A14b.*` (CHANGE applied: disclosure wording, the freeze clause, the frozen dependency rule with container ancestors, the pose-tolerance justification; Astra's
preferred all-rows comparison is not adopted, for the reason in item 2).

### A16 (2026-10-12, before W4 practice and before any new outcome; Arash's decision of 2026-10-11, W2 finding) -- one fixed token budget for the memory block and the FS block
**Trigger (W2 rehearsal, disclosed):** the experience block of the FS control (6.4) holds all ten practice attempts and needs >= 492 real tokens even with the plan
shortened, while the A12 allowances were B_L = 150 / 300 / 450, so the FS control was not buildable as registered. Separately, under 150 tokens every L1 memory block was empty.
**Decision (Arash: "Give them all a high budget so that this won't be a problem anymore"):**
- B = **800 real tokens at every level** (L1 = L2 = L3), counted as the maximum over the Gemma and API tokenizers of header + block, for BOTH the memory block (+M) and the FS block.
  The "equal token budget" claim of 6.4 therefore holds again (same B for both). FS keeps all ten attempts.
- A12 / A12a are unchanged as a rule: token accounting only, never outcomes. With B = 800 the places allowance is A_L = max(0.4 x 800, need_L) = 320 at L1 and L2 and 329 at L3 (need_L3 = 59 + 3 x 90 = 329 exceeds 320;
  the registry block states A), the procedures section gets the rest (B - A_L = 480 / 480 / 471). H, head, r, s are the A12 measurements (token counts only, unchanged).
- Supersedes the per-level 150 / 300 / 450 as the A12 B_L; the registered `budget_tokens` leaf stays in the registry as history (the A12 block overrides it, as A12 states).
  Prompts stay far inside num_ctx 8192 (Gemma mean prompt 1.9 k tokens in the W2 re-measure).
- Disclosure: decided after the pilot and after the W2 rehearsal, from a feasibility finding, not from an outcome. W3's procedures decision (S\* places-only) did not depend on B:
  the procedure coverage was 0/54 because no procedure matched (gate on the whole intent, no fragment), not because of the budget; a larger B can only raise exposure.
  The A12 minimum-coverage criterion (>= N_L whole records in every tokenizer x language) is re-checked on the real S\* before the freeze.
- Not changed by A16 (still open for Arash): the G-iso reading, the cost-guard baseline, the `--partial` execute semantics.

### A17 (2026-10-12, before W4 practice, before any freeze and before any test outcome; Arash's TODO of 2026-10-12) -- Persian concept lexicon and parser (stage W4a)
**Trigger (disclosed):** the E4-h4 finding (`sim/final_test/h4_report.md`): the memory lexicon listed the everyday Persian word for mug (لیوان) as a Cup alias, so the FA twin of every mug item built its
memory block for Cup instead of Mug, and a Persian second clause with an elided object ("X را بردار و در سینک بگذار") was not understood. **Exposure, stated plainly:** the E4 audit that found this ran the parser
over all 277 applicable general items, which include test items; it read only parse outputs (EN/FA type sets differed on 58 items, 19 of them Mug to Cup; FA parse not ok on 206), no outcome, and nobody read the
test texts to design a rule. The rules below were chosen from the 116 non-test items only (the work-order for this stage was written in the ledger on 2026-10-12 before the pass, with the two criteria below).
So this is not "blind to the test set" in the strict sense; it is outcome-blind, and the test-set parse statistics were seen only in that one aggregate.
**Changes (all in the memory parser; none touches a prompt template, a planner, a score or a split):**
1. Lexicon: Cup.fa = [فنجان]; Mug.fa = [لیوان, ماگ], Mug.name_fa = لیوان; SaltShaker.fa += نمک and PepperShaker.fa += فلفل (parity with the English aliases "salt" and "pepper").
2. Parser (PARSER_VERSION 1 to 2), Persian-only rules, driven by new lexicon keys (`partitive_fa`, `door_of_fa`, `elided_object_fa`):
   (a) the token sequence "یکی از" (one of) is removed, inside a clause only, on whole-token match, so از is not read as a source preposition;
   (b) a whole token "در" immediately followed by a type mention and را, in a clause whose verb is open or close, is removed ("open/close the door of X": X becomes the object);
   (c) an elided object: a clause whose object is missing (reason no_object, no pronoun, no unresolved mention, not negated, verb in pick/put/drop/heat/cook/wash/slice/fill/empty/break) inherits the
   PREVIOUS role's object, only when that role is not negated, has a resolved object type (never '?', '-' or a pronoun) and its verb is in find/pick/heat/cook/wash/slice/fill/empty/put/drop. The previous
   role's object is the uniquely resolved, still-active referent along the carry chain; after open/close/toggle or any '?' the rule does not fire and the clause stays unparsed. Continuation after put/drop is
   allowed because a put leaves the same object as the subject of the next carry verb ("put X in A, then take it again"); it never moves the referent to the container.
   English keeps failing closed and is unchanged. The rule makes the Persian parse succeed where the English one cannot (English needs an explicit "it"), so FA parse-ok may exceed EN parse-ok.
3. Everything else in the parser and the lexicon is unchanged (the build-script diff is only the items above). The new lexicon sha256 replaces the old one in `freeze.json`; the lexicon-hash gates are re-pinned.
**Acceptance criteria for the coverage pass (outcome-blind; stated in the ledger work-order, tightened here after Astra's check, W4a):** measured ONLY on the 116 non-test items (split manifest `nontest`), on
instruction text only. (i) FA parse-ok rate and (ii) the share of items whose EN and FA parses give the same target-type set, reported before and after; plus the safeguards (iii) EN roles unchanged for all 116
items, (iv) for every item where both parses are ok, the ordered (verb, object) sequences agree, and any disagreement is listed and explained, (v) every inherited object is a type the English parse also names,
(vi) paired categories (both ok / FA only / EN only / neither) and the residual failures are reported. No plan, score, goal result or test-item text is read or used to choose a rule. A rule is kept only if it is
a lexicon entry or a language-level construct (no item id, no FloorPlan1 constant). Stopping rule: the pass ends when the remaining failures are shared with English (fail-closed by design) or need a construct
this grammar does not cover; those are reported, not patched. **Grammar scope is frozen with this amendment: no further parser or lexicon change after this date unless a new dated amendment is written before
the freeze, and none after the test parse statistics have been looked at again.**
**Result (`docs/final-test/w4a_fa_coverage.json`, `sim/final_test/w4a_fa_coverage.py`; 116 non-test items):** FA parse ok 35 to 56 (EN ok 53, unchanged); same target types 105 to 112; EN roles changed for 0 items;
paired categories before: both ok 33 / EN only 20 / FA only 2 / neither 61, after: 48 / 5 / 8 / 55; both-ok verb/object disagreements 3 before and 3 after, all pre-existing and with the same object type
(the faucet: English "turn on" is the toggle verb and Persian "باز/ببند" is open/close; "empty the mug" parses to one role in English and two in Persian); 20 items inherit an object, 0 of them to a type English does
not name. Criterion (iv) is therefore met only with those three disclosed disagreements, which this amendment accepts: they are verb-level, not object-level, and do not change the memory key's object type.
**Known remaining limits (reported, not fixed):** "درش را ببند" (close its door) and other pronoun or possessive antecedents fail closed in both languages; stative "X در دست توست" (holding) clauses;
ordered enumerations ("اول X و بعد Y را برش بزن"); the colloquial "میز" for a counter; abstract L3 statements without an action verb. The test set's parse coverage is computed by this same frozen parser at the freeze and
reported per language.
**Consequences:** the C3 FA rows for Mug/Cup items are stale and are not reused; the W4 practice run and every later capture use the new lexicon. Equal token budgets do not guarantee equal coverage or usable memory:
FA parse success is at least EN's on the non-test items (56/116 vs 53/116), and the analysis reports parse coverage and empty/partial blocks per language, so a language difference in the memory effect may reflect this
asymmetry and is not claimed from the memory arm alone. Astra check: `docs/final-test/_astra/W4a-fa-lexicon.*` (CHANGE on all three points; applied: exposure and chronology wording, frozen scope and stopping rule,
the semantic guards (iv)/(v)/(vi), the one-sided coverage sentence replaced).

### A18 (2026-10-12, written by the main session from Arash's own words in the conversation; before freeze.json is final and before any test outcome) -- whole-plan ceiling, Codex main arms, run-all, replay-hook fix
1. **Whole call-plan ceiling 55 USD** (A15 said 45). Arash: "Sure", answering whether to raise the overall cap to about 55 USD after the W2 token re-measure.
   - After A11 the prompts are about 44 % larger. The thinking-tail projection is about 50 USD for the whole plan.
   - The core six arms incl. REPEAT stay capped at 25 USD.
   - The section 3.1 cut order still applies if a projection exceeds a ceiling.
2. **Codex `gpt-6-sol` becomes a main, executed comparator, with and without memory: arms X0 and X+M.** Arash: "one of the main arms should be from codex", then chose gpt-6-sol and "with and without memory".
   - X0 and X+M use the same renderer, tools, observation, budgets (A16), memory snapshot S* and executor as S and O.
   - They add secondary hypotheses to the section 2 Holm family:
     - **H5a** `SR(X+M) - SR(X0)` (memory gain, one-sided, H0: gain <= 0);
     - **H5b** `SR(X+M) - SR(G+M)` (two-sided, H0: difference = 0);
     - **H5c** `SR(G+M) - SR(X0)` (non-inferiority, H0: <= -5 pp, entering Holm as min(1, 2p) like H3c).
   - H1 (primary, G+M vs O0) is unchanged.
   - CLI caveats are reported with every X result (A2): no temperature or seed control, CLI context overhead, ChatGPT-account rate limits.
   - Codex costs no API money. The other three Codex models stay exploratory (section 3.4).
3. **Every registered arm is executed** (Arash: "Run them all"). No arm is scored on its plan only. Plans are made first (plan-first mode, if its dev safety test passes). Identical plans in identical restored worlds run once and their result is shared (execution-unit dedup, as registered).
4. **Execute replay hook** (W2 defect 1): the pre-execution check compares the told holding plus the A14b state check, not the full summary hash, which blocked about 90 % of executions over coordinate jitter. Arash approved it: "do it".

### A19 (2026-10-12, main session, Arash's own answers; before freeze.json is final and before any test outcome) -- G-iso rule and cost-guard baseline
1. **G-iso compares what matters, not raw poses.** Arash chose "Compare what matters".
   - The registered strict digest cannot pass: the absent-vs-absent control also differs in 12 of 12 runs, through coordinate jitter in the sweep event stream and the summary hash.
   - New rule: for each presentation, the no-memory (`0`) arm must give identical
     (i) prompt request bytes,
     (ii) run status,
     (iii) goal reached,
     (iv) executed step list (action, object type, ok flag),
     (v) audit counts (violations, oracle reads)
     under the conditions memory files absent, present, absent.
   - Continuous poses and event timings are excluded.
   - Pass rule: all selected presentations are identical on (i)-(v), with n = 12 seeded dev presentations (as registered).
   - It is re-validated on fresh dev runs BEFORE the freeze is final.
   - If the absent-vs-absent control differs on (i)-(v), the result is 'inconclusive' and is reported as such.
2. **Cost-guard baseline.** Arash chose "2 x new cost".
   - The guard pauses W5 for Arash's decision when the running mean cost per Opus call exceeds 2 x the expected cost under the post-A11 prompts: 2 x 0.012 = **0.024 USD**, replacing 0.0178.
   - The 25 / 55 USD ceilings (A15, A18) are unchanged.

### A20 (2026-10-12, main session, Arash's own answer "Fix + re-validate"; before freeze.json is final and before any test outcome) -- G-iso audit component and control noise
- Run 1 of the A19 validation is kept as registered: INCONCLUSIVE.
- Two corrections, applied to the NEXT fresh validation only:
  1. Component (v) "audit counts" means what A19's text names: **violations, oracle reads and map-build reads**. Motion bookkeeping counters (pose, actuator, action-resolve) are excluded. These are allowed accesses whose counts follow the trajectory length. The A19 implementation compared the whole counter dict; that was a coding slip.
  2. **Pass**: absent-vs-present shows ZERO differences on (i)-(v) for all n = 12 presentations, AND the absent-vs-absent control differs on at most 2 of 12 presentations, only on (iv) steps. Such a control difference is recorded as simulator noise and reported.
     **Inconclusive**: the control differs on more than 2, or on (i), (ii), (iii) or (v).
     **Fail**: any absent-vs-present difference.
- One more fresh validation (n = 12, absent / present / absent) decides the gate. If that is inconclusive too, the gate stays pass=false; it is reported with the component evidence and the test proceeds.

### A20.1 (2026-10-12, main session; after Astra's review docs/final-test/_astra/A18-A20-review.txt; written before the fresh G-iso run reports) -- honest chronology and precedence
1. **Chronology (correcting A20's wording).**
   - A20 was written AFTER seeing development-validation results (G-iso run 1 and the diagnostic run 2), and BEFORE any test outcome. It is not "before any outcome".
   - Two parts of A20 are corrections; one part is a relaxation:
     - Excluding motion bookkeeping corrects an implementation slip: A19 named violations and oracle reads.
     - Adding map-build reads is a STRENGTHENING beyond A19's text.
     - The control tolerance (at most 2 of 12, steps only) is a pragmatic engineering screen chosen after seeing 1 control difference. Arash kept it, with full disclosure. It is not a statistically calibrated tolerance: 0 of 12 absent-vs-present mismatches still allows a one-sided 95 % upper bound of about 22 % on the mismatch rate.
   - Control differences are reported as "unexplained execution variability consistent with simulator noise".
   - Runs 1 and 2 are kept and published.
2. **G-iso run rules.**
   - The present run is compared with the FIRST absent run.
   - Absent-vs-present differences take precedence over an inconclusive control.
   - All three executions really run (no dedup, no cached result).
   - The no-memory request is rebuilt through the full construction path.
   - Audit sensitivity is shown once: an injected forbidden read must raise the counted violation or oracle count.
   - No further result-driven retries.
   - If the fresh run is INCONCLUSIVE, the gate is not passed. W5 may start only with Arash's explicit written waiver, and claims about memory isolation are narrowed. If it FAILS, W5 does not start until the cause is investigated.
3. **Budget precedence (Arash: "Caps win, cut order fixed").**
   - The 25 / 55 USD caps are hard. If a projection exceeds a cap, arms are cut in the fixed order of section 3.1 (controls, then Haiku, then the exploratory Codex models), at block boundaries and blind to outcomes.
   - The core six arms and the gpt-6-sol arms X0 / X+M are never cut.
   - "Every registered arm is executed" (A18.3) holds within the caps.

### A21 (2026-10-12, stage G-blockers; written AFTER development results and BEFORE any test outcome; Astra review: docs/final-test/_astra/A21-review.txt) -- the freeze blockers closed

**Honest chronology.** Everything below was decided after this development evidence existed and before any test item was touched: the three G-iso runs (A19, A20, A20.1), the Codex isolation probe of 2026-10-12 (`docs/final-test/matrix/x_isolation.json`), the dev simulations in `sim/final_test_power_secondary.py` (`power_secondary.json`, `ni_calibration.json`) and the call-plan projection (`final_projection.json`). Nothing here was chosen on a test outcome (there is none). Rules that go beyond the earlier text are marked **NEW**; the rest writes down what the code already did or clarifies A15, A18 and A19. Statements about code are backed by selftests (ledger entry "STAGE G-blockers"). Astra reviewed a first draft (verdict CHANGE); this text applies its points: the non-inferiority guard was re-validated and tightened, "served model" became "CLI-configured model", the budget reading in G was corrected, and the failure precedence in A.1 was added. This amendment passes no gate and does not change the G-iso verdict.

**A. Blocked and failed runs (replaces the open question; extends section 13).**
1. *Infrastructure* means the run never got a fair test of the plan through no fault of the plan or the agent: the restored world differs from Phase A (blocked state), a simulator crash or hang, or a planner transport failure after 3 transport retries (provider timeouts are 240 s for Gemma and the Anthropic API and 300 s for Codex, and count as transport failures). *Agent-caused* means a plan the validator rejected or the executor's own limits: wrong or invalid plan, refusal, step failures, the 900 s episode wall cap, the 3,000 primitive cap. An honesty violation is neither: it is an INVALID episode (section 13, unchanged).
2. **NEW precedence (one cell, one class, first match wins):** (1) INVALID episode; (2) planner transport failure after the retries = `infra_planner`; (3) a rejected plan (no JSON, schema, unknown action or type, refusal, truncation) = `agent`, nothing executed; (4) restored world not equal to Phase A, or a simulator crash or hang AT ANY TIME, including after the first step = `infra_exec` once the 3 attempts are used (the run is discarded and redone from a fresh restore); (5) step failures and the executor's own caps = `agent`.
3. *Retries:* execution attempts 0, 1, 2, each from a fresh restore (unchanged); planner cells re-queued in at most 2 later sessions (unchanged). The 60 s detector hang guard kills the process (exit 75); the interrupted attempt is repeated without using one of the 3 attempts; a unit that hangs on 3 successive launches is declared a terminal crash by Arash with a ledger note.
4. *Scoring and denominators (unchanged):* an infrastructure failure still missing after the retries counts as failure (SR 0) in the primary analysis and stays in every denominator. Every score row carries its class (`infra_planner`, `infra_exec`, `agent`, `ok`, `budget`, `missing`).
5. **NEW sensitivity:** next to every contrast the report gives the estimate after dropping, from BOTH arms, every presentation where EITHER arm has an infrastructure failure of either kind (the registered sensitivity covered planner failures only). **NEW flag:** an arm with more than 5 % infrastructure cells among its scored cells is marked "infrastructure-sensitive"; the report shows both wordings side by side. The primary analysis never changes.
6. A blocked run is never a G-iso pass (a pass needs every presentation measured).

**B. The cost guard, written out (clarifies A15 and A19; NEW checks marked).**
- *Unit:* one billed Opus plan cell = the first transmission plus its repair turn plus any billed transport retry. Cache replays are not cells. *Numerator:* the sum of `cost_usd` over all attempts of the cell, including cells that ended rejected or infra-failed but were billed. *Denominator:* the number of billed Opus cells (cost above 0); a transport failure that was never billed is not counted.
- *Threshold and sample:* the guard pauses above 0.024 USD (A19). There is no mean check before 60 billed Opus cells; after that it runs at block boundaries each time 60 more exist. It is a tripwire, not a forecast: it can fire only after the cells that crossed the line were already paid for.
- **NEW:** the check applies to the cumulative mean AND to the mean of the last 60 billed Opus cells.
- *In flight:* a transmission in flight is a ledger reservation at its worst case; it counts against both the core and the total ceiling until it settles or is released. The runner is sequential.
- **NEW:** at every block boundary the ledger's settled paid total must equal the sum of `cost_usd` over all plan-journal rows within max(0.05 USD, 2 % of the larger total); otherwise it pauses. After W5, and at every pause, Arash compares the ledger with the provider's own usage page; a gap above 5 % is reported with the results.
- *Responses to a pause (exactly three):* continue unchanged; record the next group of the registered cut order (codex, haiku, curve12, reloc) and then continue; stop and declare the study incomplete (section 13). **NEW:** every pause decision is outcome-blind (only spend, tokens, errors and integrity counters); in test-v1 `--ack-pause` needs `--pause-response`, a note of at least 20 characters naming that evidence, and `--outcome-blind`. Changing a prompt, a model, a budget, the roster outside the cut order, or the ceilings is a protocol change: an amendment and a restart of the phase.

**C. The replay precondition (completes A18.4; A14b unchanged).** A cached plan runs only if the restored world equals the Phase A world on everything the task can depend on: the A14/A14a/A14b state comparison (object identities and types, containment, open/closed, picked-up, sliced and the other flags, accessibility, relevant summary rows, seen types, labels, reference-plan feasibility, per-object pose within tau, agent pose to numerical precision), AND the **actual** holding read from the simulator's own picked-up flags must equal Phase A's, AND the told holding must equal Phase A's and agree with the actual holding. The hook refuses to run a plan unless that check ran and passed. Evidence (selftest, no simulator needed because the check is a pure function of two states): on a synthetic kitchen built with the real `reset_gate.capture_state`, 300 seeded harmless-jitter cases pass and 22 seeded fault classes (each five times, with and without jitter) are rejected; on the **24 real Phase A states** captured from the live simulator (four of them with an object really in the hand), identical restores and 96 seeded jitters pass and 196 seeded faults (dropped or swapped held object, opened/closed, displaced 0.25 m, removed object, cleared parent, sliced flag, agent moved 0.2 m, wrong told holding) are all rejected. An injection inside a live simulator run remains recommended for the next dev rehearsal.

**D. Execution sharing (confirms A18.3 and section 7; freezes the key; NEW reporting).**
- *The key* is frozen with a golden value: the normalised plan (action, object, to, arg, order kept), presentation, chooser class (prior or snapshot), snapshot sha, restored state, post-sweep summary sha, caps and seed policy. Arms with equal keys share one simulator run. Every score row carries its `unit`; every exec record lists its plan `cells`; analysis refuses a complete journal in which a score row is not linked to its execution.
- *Dependence:* a shared outcome is ONE realisation, not agreement of two runs, and the difference of two arms that share a unit is 0 by construction. This does not bias the contrast if equal keys imply the same distribution of outcomes (it can only reduce variance, as common random numbers do), and the inference already clusters on the 42 families, which contain every unit. **NEW reporting:** per arm the scored cells, distinct units and the share of cells that share a unit with another arm; per contrast the presentations with a shared unit and the discordance; a **distinct-units-only sensitivity** that drops them. That estimate is performance CONDITIONAL on non-shared executions: it changes the population and cannot validate or replace the full-sample result. "Independent executions" are counted as distinct units, never as cells.
- *Costs:* planner USD is charged to the arm whose call made the plan; the wall time of a shared run is split equally among the scored cells that share it; simulator time is otherwise a shared benchmarking cost (section 7).
- The A18.3 wording "every registered arm is executed" holds within the ceilings (A20.1.3).

**E. The Codex X arms (gpt-6-sol): isolation, identity, version (completes A18.2).**
- *Clean sessions:* every call is a fresh ephemeral session in an empty temporary directory with the user's configuration and rules ignored, API-key variables removed, the read-only sandbox, sixteen CLI features switched off (apps, shell tool, browser, computer use, multi-agent, plugins, hooks, image generation, skill search, image viewer, sleep, unified exec, goals, remote plugins, in-app browser) and the reasoning effort fixed at "high". Measured on codex-cli 0.162.1: the default session lists 24 tools and bills about 14.1 k input tokens for a one-word prompt, the hardened one 15 tools and 9.0 k. Some built-in tools (code-mode exec, apply_patch, MCP resource listing) cannot be switched off from the command line: their use is detected, not prevented. The CLI adds about 10.6 k input tokens of scaffolding to every call.
- *Isolation checks:* (1) per call, any non-message item in the event stream counts as a tool event and flags the cell; (2) the probe `x_isolation_probe.py` places random-token files outside the call's directory in three places and asks the model to read them with any tool: 9 calls, none returned a token; the six canary-read calls produced 24 visible tool events (MCP resource listing). This is the gate **X-iso**, valid for the exact flag set and CLI version it measured. The gold dataset is never used as a canary. These checks are limited evidence: they cannot show the absence of invisible tools or access.
- *Identity:* the session facts are read from the CLI's own session event (model, effort, approval policy, sandbox, network) and logged per call as the **CLI-configured model** (`cli_configured_model`; `served_model` stays empty). **NEW:** any difference from the frozen values is an identity violation and aborts the planning phase (fail fast, section 13). The model the server finally answers with is NOT observable through the CLI.
- *Version:* the CLI version in freeze.json is pinned; in the test namespace a different version skips the Codex planner before any call.
- *Fixed retries:* one transmission per attempt, no transport retry of our own (the CLI has invisible ones); a transport failure is `infra` and is handled by A.
- **NEW sensitivity:** for H5a-c the report adds the estimate after dropping presentations whose X cell had a tool event or an identity problem; the primary results keep flagged calls.
- **Required wording with every X result (printed by the analysis):** "X used codex-cli 0.162.1 configured for gpt-6-sol at high reasoning effort (the CLI-configured model; the model the server finally answered with was unobservable). Built-in tools remained available: isolation monitoring covered the events the CLI emitted, not invisible calls. The isolation probe (9 calls; 6 canary-read calls produced 24 visible tool events, mostly MCP resource listing) returned no secret token. Primary results include calls flagged for tool events; paired exclusions are reported separately. CLI scaffolding (about 10.6 k input tokens per call) and subscription execution are part of this comparator; 'free' means no incremental API expenditure only." A tool-free model-only claim is not made.

**F. The analysis plan, frozen (completes A18.2 and section 9).**
1. *H1* stands alone: one-sided .025, the wild test and the percentile bound must agree, fixed sequence. It is NOT a member of the Holm family, so adding H5a-c does not change H1's inference. There is no study-wide .05 claim. H1 is NOT covered by the guard in item 4; its registered calibration table (section 9.6) stands, and the report prints H1's observed discordance with a caveat when it is below .25.
2. *The Holm family* has m = 11 members (H2a-c, H3a-c, H4a-b, H5a-c) at alpha .05. Adding three members makes every adjusted p of the registered eight larger or equal; the report prints the m = 8 adjustment next to the m = 11 verdict. The strictest step is .05/11 = .0045 (was .05/8 = .0063).
3. *p-values:* superiority members use the one-sided wild cluster bootstrap with the null imposed at 0; two-sided members use min(1, 2 min(p_greater, p_less)); the non-inferiority members **H3c and H5c** use p1 = the one-sided wild test of H0: D <= -5 pp with the **boundary** imposed (never a test of D = 0), and enter Holm as min(1, 2 p1). The doubling puts a one-sided .025 non-inferiority test on the two-sided .05 scale of the family; at the first Holm step the raw p1 level is therefore .05/22 = .00227.
4. **NEW guard, validated on dev simulations (`ni_calibration.json`):** if the observed discordance of H3c or H5c (the share of presentations in which the two arms' SR differ; a presentation with a shared execution is concordant) is below **.25**, its p is set to 1 (not rejected, m stays 11). Evidence: type-I error of the complete non-inferiority procedure at the null boundary D = -5 pp, 4,000 simulations per cell (Monte Carlo SE about .001), the real 42 family sizes, EN/FA agreement .7, discordance .05-.30, family ICC 0 / .3 / .6. Without a guard the rejection rate at the first Holm step (nominal .00227) is up to .14 at discordance .05, .054 at .10, .027 at .15, .009 at .20 and .0075 at .25. With the guard at .25 the largest first-step rate in the grid is .0022 and the largest rate at the most lenient Holm step (nominal .025) is .0213, that is at or below nominal within Monte Carlo error. Guards at .20 and at .15 were tried first and left first-step rates of .0037 and .0047 (1.6 and 2.1 times nominal); the percentile-interval conjunction used for H1 does not repair the low-discordance regime. A non-inferiority result for H3c or H5c is still labelled approximate. The guard can only remove rejections; its price is power when the true discordance is .20-.25.
5. *Pairing and clustering:* the unit of analysis is the item (mean of its EN and FA presentations); pairing is by presentation; the cluster is the family. Sensitivities: family-weighted, infrastructure-paired (A.5), distinct-units-only (D), X-isolation-paired (E).
6. *Power note (planning aid, not a sample-size rule; `power_secondary.json`, 1,000 simulations per cell, the real 42 families, 125 items, EN/FA agreement .7).* At the first Holm step a single superiority test rejects with probability 0.99 / 0.84 / 0.50 for a true D = +10 pp at discordance .10 and ICC 0 / .3 / .6, 0.52 / 0.31 / 0.18 at discordance .20, and 0.30 / 0.16 / 0.10 at discordance .30; at D = +5 pp it is at most 0.23. A non-inferiority member can establish "within 5 pp" at a true D = 0 with probability only 0.02-0.23, and at D = +5 pp 0.08-0.80 (before the guard, which removes the cases below .25 discordance). So H3c, H5c and most superiority members will often be "unresolved"; an unresolved result is not evidence of inferiority or of no effect. The 5 pp margin remains a prespecified tolerance, not a claim about practical importance.

**G. Final cost projection and precedence (completes A15, A18 and A20.1.3).** `docs/final-test/matrix/final_projection.json` (generated by `final_test_callplan.py --final-projection`) prices the final roster (12 main arms incl. REPEAT r = 1, 2, the learning curve and the relocation block), the real block sizes (S*/curve 149 tokens, FS 759), a thinking tail, repair and failed-billable-call rates. USD, core / other / total against the ceilings 25 / 55:
| scenario | core | other | total | result |
|---|---|---|---|---|
| S1 pilot-like (no tail) | 14.2 | 16.8 | 31.0 | fits, margin 24.0 |
| S2 reference (+500 tail tokens per call, 15 % repairs, 3 % failed calls) | 27.4 | 31.7 | 59.0 | the core arms do NOT fit their 25 USD ceiling (over by 2.35); cutting the exploratory arms brings the total to 54.8, but the core cannot be cut, so the core arms would not all complete |
| S3 stress (+1000 tail, 25 % repairs, 6 % failed calls, 800-token blocks) | 45.5 | 52.6 | 98.0 | no cut fits; the study would be reported incomplete |
The "codex" cut group means ONLY the exploratory models X-gpt-6-astra, X-gpt-6-luna and X-gpt-5.6-terra (free); X0 and X+M are main arms and are never cut; the first group that saves USD is "haiku". The hard ledger ceilings and the incomplete-study outcome (section 13) are the expenditure controls; they are not a guarantee that the core arms complete. The 0.024 USD guard (about +600 tail tokens per Opus call) is a tripwire that fires after 60 cells at a block boundary; it is no assurance of a pause before the S2 regime has been spent. Whether the thinking tail appears is unknown (the pilot saw almost none; Opus adaptive thinking cannot be switched off). The caps win; the cut order is fixed; core arms, Gemma, FS arms and X0 / X+M are never cut. Simulator time: at most 11,240 episodes, 59.4 h at 19 s each without sharing; the dev rehearsal shared 22 % of plan cells (212 cells, 165 units), which would give about 46 h. The earlier "10-13 h" estimate is withdrawn: it was not computed from the call plan. All USD figures use list prices and the post-A11 priors; none is an outcome.

**H. What is still open (not decided here).** G-iso stays as recorded (A20.1 pass on dev). For Arash: the family audit, the freeze check, and the budget decision (accept the S2 risk to the 25 USD core ceiling, or raise it by an amendment before the first call). freeze.json must be regenerated after this amendment and `gate_evidence.py --write` run so that `gates/X-iso.json` exists; an injection of faults inside a live simulator run is still recommended in the next dev rehearsal.

### A22 (2026-10-12, main session from Arash's own answers; after Astra's review docs/final-test/_astra/PF-findings.txt; written AFTER development results and BEFORE any test call or test outcome) -- caps, run time, plan-first, H1 validity
1. **Caps: core 30 USD, whole call plan 65 USD.** Arash: "$30 core / $65 total". This replaces A15 and A18's 25 / 55.
   - Reason: with the +500-token reasoning-tail scenario (S2) the core six arms project 27.4 USD and the whole plan 59.0 USD.
   - S2 is a sensitivity scenario, not a forecast. The caps do not cover S3. The cost tripwires stay and do not guarantee completion.
   - The core roster is unchanged. Pause and resume decisions are outcome-blind, at block boundaries. No reasoning-token limit is introduced, because one would change the evaluated agents.
2. **Run time.** The earlier "10-13 h" estimate is withdrawn.
   - The call plan gives at most 11,240 episodes x about 19 s = **59.4 h** of simulator execution. About 46 h is expected if the test deduplicates like the dev rehearsal (22 %); that figure is conditional, not a commitment.
   - Arash: "Run everything". Every registered arm, with the registered REPEAT, is executed on one simulator. Plans-only scoring is not used.
3. **Plan-first scheduling (PF-safety-test, qualified pass on dev):**
   - W5 planning may run in parallel with at most 4 Anthropic, 2 Codex and 1 Gemma, completing before any execution. The serial path is the fallback.
   - Disclosed limits: exactly-once billing is not guaranteed; killed attempts may be charged without a journal result; replacement attempts are recorded separately; live rate-limit behaviour is unverified; Haiku showed 2/48 plan-content differences between identical requests.
4. **H1 validity (Arash: "Fix + validate H1 now").** The registered wild cluster non-inferiority test can be anticonservative at low discordance, so it is not used as is.
   - Kept: the contrast, estimand, family clustering, the 5-pp margin (H0: Delta <= -0.05, Delta = p(G+M) - p(O0)) and the standalone one-sided alpha = .025.
   - Before freeze.json is final, the COMPLETE H1 decision rule, including any observed-data guard, is validated by simulation at H1's actual level, on the real test family counts, sizes, pairing, weights and dependence. Null-boundary and low-discordance cases are included, with Monte Carlo uncertainty reported.
   - The acceptance criterion and the fallback are written before the simulation is run.
   - A compatible lower confidence bound must exceed -0.05 to claim non-inferiority. A guard failure means "inconclusive", never "inferior".
   - If no defensible procedure passes, H1 becomes estimation-only and the confirmatory non-inferiority claim is withdrawn.
   - The validated method is recorded in a follow-up amendment A22.1, with its exact procedure.
   - No test data are touched by this work.

### A23 (2026-10-12, main session from Arash's instruction "remove these two"; Astra review docs/final-test/_astra/A23-remove-S-H.txt; written AFTER development results and BEFORE any test call; held-out-test-outcome-blind, NOT data-blind) -- Sonnet and Haiku removed
1. **Removed from the test protocol entirely:** Claude Sonnet 5.5 (S0, S+M) and Claude Haiku 4.5 (H0, H+M).
   - Reason: feasibility and focus. The registered matrix was up to about 11,200 simulator episodes (about 2-2.5 days on one simulator). The revised workload is recomputed by the callplan and recorded in freeze.json.
   - Prior knowledge, disclosed: Sonnet and Haiku dev-pilot results had been seen (every frontier arm had 24/24 first-pass plan validity). Plan validity does not establish task success, and those observations were not used to pick which families to remove. The choice is about workload and a focus on the local-vs-frontier question.
2. **Remaining main arms (8):**
   - G0, G+M, O0, O+M: the **core four**, under the core cost cap of 30 USD (A22);
   - G+FS and O+FS: the structured-memory vs experience control;
   - X0 and X+M: gpt-6-sol via the Codex CLI, A18.
   - The three other Codex models stay exploratory.
3. **Hypotheses.**
   - **H1 is unchanged:** G+M vs O0, 5-pp non-inferiority, one-sided alpha .025, standalone and outside the Holm family. Its H1-only validation (A22.4) is unaffected by this change, because it uses the task-family structure, not the arm list.
   - **Withdrawn before testing, with their IDs kept for traceability:** H2b (S memory gain), H3b (S+M vs G+M) and H3c (G+M vs S0).
   - **Revised secondary Holm family, frozen with explicit membership:**
     - H2a SR(G+M)-SR(G0) and H2c SR(O+M)-SR(O0);
     - H3a SR(O+M)-SR(G+M);
     - H4a SR(G+M)-SR(G+FS) and H4b SR(O+M)-SR(O+FS);
     - H5a SR(X+M)-SR(X0), H5b SR(X+M)-SR(G+M) and H5c SR(G+M)-SR(X0), non-inferiority.
     - That gives **m = 8**, alpha .05, and all other rules as registered. The smaller m changes the thresholds, which is disclosed.
     - The interaction estimate I (G vs O) is unchanged.
4. **Blocks.**
   - REPEAT r = 1, 2: the 8 remaining main arms.
   - Curve: G+M and O+M, unchanged.
   - Relocation: G0, G+M, O0 and O+M, unchanged.
   - Exploratory Codex: unchanged.
   - Removed families cannot re-enter. Their rows are deleted from the callplan, manifests, budget totals and analysis.
5. **Claims lost:**
   - no confirmatory evidence on Sonnet's memory benefit, on S+M vs G+M, or on G+M vs S0;
   - no Haiku conclusions;
   - no "mid-size frontier" claim.
   - The absence of these arms is not evidence of equivalence or failure.

### A24 (2026-10-12, stage H-A24, from docs/final-test/kitchen_memory/FINAL-PLAN.md as approved by Arash; Astra review docs/final-test/_astra/H-A24.txt, verdict CHANGE, 9 points applied; written AFTER development results and BEFORE any test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- the privileged briefing + experience/knowledge package, the common interface, hypotheses, acceptance and budgets

**0. Chronology and what was seen (disclosure).** No test item has been planned, executed or scored under this protocol, and no test outcome exists. This amendment is a development-informed redesign of the treatment before confirmatory testing; it is not the original specification. It was decided after the following evidence had been seen by Arash and by the sessions that wrote it:
- Cost pilot on the 12 dev items x EN/FA (24 presentations per arm): executed successes G0 13/24, G+M 13/24, O0 18/24, O+M 18/24; Gemma L3 0/8; Gemma first-pass plan validity 19-21/24, every frontier and Codex arm 24/24 (Sonnet and Haiku included; those arms are removed by A23).
- One-order memory screens on non-test practice-pool items: C3 SR .704 (G0, 38/54) vs .685 (places memory, 37/54); W3 order 0: G0 32/54 vs places 36/54 executed successes. These are descriptive differences from one order; they do not establish a reproducible benefit. Also the A13 procedure gate and the PF harm screen (net loss 0).
- Dev rehearsals rehearsal-v1 and rehearsal-v2 (Gemma plan validity 68/72 vs 69/72 main r0 cells), the PF-safety test, the three G-iso runs (A19, A20, A20.1), the Codex isolation probe, the dev cost and power simulations of A21 and the H1-validation simulations of A22.4 (simulated data only).
- The read-only diagnosis in docs/final-test/kitchen_memory/ (diagnose.py, diagnosis.jsonl, diagnosis_summary.txt, DESIGN.md section 1): the observation summary every arm sees was cut at 40 lines, sorted by type name, in 210/210 dev prompts checked, so Microwave, SinkBasin, StoveKnob, Knife and other types after "F" never reached any planner; in 44/48 rehearsal presentations a type the task needs was missing from it; the places memory block was empty in 12/24 rehearsal-v1 and 17/24 pilot presentations; Gemma's failures were mostly wrong action (cook confusion), wrong type token (including Persian mappings), invalid format, L3 decomposition, and detection failures that Opus also had.
- Earlier step-4/step-5 FloorPlan1 development results (docs/step4-progress.md, docs/step5-progress.md).
- **Test-content exposure (not outcomes).** Before the split of 2026-10-10, all general-set items, including those that became test items, were written and hand-edited by Arash; their reference plans were executed with no LLM in step 4 (honest .749 vs oracle .996 over 267 applicable items) and by validate_general; step 4 also had a 6-item Gemma pilot. Arash's family audit (gate 12.8) read the test families. The absence of test outcomes therefore does not mean the absence of exposure to test content.
- **Evidence manifest.** The dated entries of docs/final-test/progress.md are the linked evidence manifest: they identify every development run and analysis available to the decision-makers, including unsuccessful runs, their outputs, and when they were inspected.
Consequence: the claim "location memory" or "experiential memory" is withdrawn for the final test; the C3/W3/pilot memory results are history under the old treatment and are reported as such. The trigger is Arash's decision of 2026-10-12 ("we have to improve the gemma memory then about the basic info about the kitchen"; "we have to compensate that using a lot of rag and general info about each kitchen") and his research goal, recorded verbatim in the ledger the same day.

**1. The treatment, renamed and defined (replaces A9 and section 6.1 for +M; rewords section 1's question).**
- +M is now the **"privileged briefing + experience/knowledge package"** (the "package"), not "memory". Its effect is the effect of the whole bundle; no component's effect can be identified from the test, and none is claimed.
- Section 1's question becomes: *"In FloorPlan1, does Gemma 4 E2B with the frozen privileged briefing + experience/knowledge package achieve task success comparable to or higher than Opus 5.5 without it?"* H1 compares assisted Gemma with unassisted Opus as configured systems; it says nothing about intrinsic model capability or model size in general.
- **Components** (each a separately labelled section of the package, in this fixed order; every record carries its source, provenance, time of creation and a confidence or support value):
  1. **Welcome briefing (privileged: a metadata-derived, homeowner-approved briefing).** A fixed, kitchen-independent script writes a human-style tour of the kitchen in English and Persian from the kitchen's static layout and its usual storage at the default reset (fixtures, their relative positions, and which categories are normally kept where). It never states the current position or state of a movable object in an episode. Static defaults may predict episode locations; the briefing is never regenerated from an episode reset or current state. It is **privileged information acquisition**, declared as the ONLY privileged channel: privileged task information enters the planner only through this channel, and metadata used exclusively for resets, scoring and isolated reference audits cannot reach planning or retrieval. In the prompt it is labelled as what the homeowner said, not as something the agent saw. Arash approves each kitchen's briefing before the freeze (item 9).
  2. **Observed kitchen knowledge (honest).** The kitchen profile (fixtures, last-seen states, contents detected when the agent opened a container, tools and their last-seen parents, the agent's own action outcomes, a knob-to-burner mapping only if the agent saw a burner change after toggling a knob) and the places records, learned only from the honest practice episodes through the step-5 store with write_log and evidence rows. Contents and locations are historical hypotheses ("last seen ... in practice"); what the agent sees now wins.
  3. **Procedural knowledge base.** Operator requirements and effects (for example: slicing needs a supported knife held; heating goes through a supported appliance route and its preconditions; heating is not cooking), written once from the domain specification and the honest skills' implementation, scene-independent, and frozen only after each entry is checked against the executable skills.
  4. **Worked examples.** Short, successful, parameterised examples (instruction, relevant initial conditions, a valid plan, the prerequisite that explains its order) taken from the fresh practice episodes only, each with its practice-episode provenance. If any example plan was produced by a model other than Gemma, that teacher contribution and its cost are disclosed.
  5. **L3 decomposition templates.** Generic parameterised templates (single goal, ordered transformations, multi-goal dependencies with subgoals, prerequisites, intermediate placements, hand occupancy and completion checks), written from the domain specification and selected from the instruction text, never from the benchmark level label.
- **Never in the package:** simulator metadata (except through the declared briefing), reference plans, goal predicates, test item ids, test instructions or their paraphrases, and examples or templates derived from inspecting test items or test families.
- **Retrieval (frozen, deterministic, model-independent, no LLM; Arash's principle of 2026-10-12, "a lot of info and use only the best when needed").** Parse the instruction (the frozen A17 concept parser, EN/FA) into objects, operations and goal state; expand through the stored relations to the needed tools, receptacles, appliances and prerequisites; gather candidates by exact keys, lexical match and bge-m3 dense retrieval; rerank with a fixed deterministic scorer under per-component quotas with borrowing. No padding. A frozen eligibility rule (fixed before acceptance) says when evidence for an instruction exists; an empty component without eligible evidence is legitimate and stays empty, and only a missing eligible record counts as a delivery failure. Every presentation logs what was retrieved, what was dropped and why.
- **Budget:** at most **2,000 tokens** for the whole package including its headers, counted as the maximum over the Gemma and Anthropic tokenizers, binding in the more expensive language (Persian); this replaces A16's 800-token B for the +M arms. Whole records only; the first record that does not fit stops its component. Quotas, borrowing and the counter are pinned in freeze.json by hash. Gemma's num_ctx stays 8192: the full prompt plus the package plus two repair turns plus the 768-token output must fit, and this is measured on the real prompts (a context-fit gate), not estimated.
- **Fairness:** the package is rendered once per presentation by deterministic code; **identical bytes** go to G+M, O+M and X+M, from the same practice corpus and the same retrieval inputs. The 0 arms get no package and no briefing.

**2. The common interface (every arm, including the 0 arms; extends section 4 and section 3.3).**
- Every arm gets the same system prompt (a new version, ft-v2), the same tool contract (exact skill names, argument types, valid combinations and one minimal valid example), the same JSON schema, the same plan validator, the same execution limits and the same currently available observations. None of this is part of the package.
- **Repairs: at most 2** repair re-prompts (was 1), with the validator's specific reasons. The validator rules are identical for every arm and use only the common observations plus the common contract. It checks syntax, skill existence, argument consistency and prerequisites, propagating known state changes through the preceding planned actions. It distinguishes unknown from false: an unknown prerequisite is never treated as observed false. Its messages never disclose a scene fact that was not already in that arm's prompt. Still invalid after the second repair: not executed, and the presentation fails (agent-caused, A21.A).
- **Observation summary:** the 40-line truncation is fixed for every arm by a deterministic overflow rule that does not lengthen the list (DESIGN.md section 2: drop Floor lines; group same-type fixtures with counts and states; one line per type before any second line; A-Z order). If more than 40 types remain after that, the first 39 types in A-Z order are kept and line 40 states how many types were omitted; the exact rule is pinned in A24.1. An optional fixed Persian-to-type glossary from the frozen A17 lexicon is given either to every arm's FA presentations or to none (Arash decides before the freeze; never to +M only).
- Model identifiers, quantisation, reasoning settings and CLI configuration stay as frozen in section 3.1, A18 and A21.E; external tools and hidden conversation persistence stay disabled.
- The exact frozen texts (prompt ft-v2, tool contract, validator rules, summary rule, glossary yes/no) are pinned by hash in freeze.json and listed in a follow-up amendment **A24.1** BEFORE any acceptance data exist. Every Phase A capture, G-iso, the token gates and the call plan are redone under ft-v2; dev numbers under ft-v1 are not comparable and are kept as history.

**3. Arms (unchanged, 8; A23).** G0, G+M, O0, O+M (core four), G+FS, O+FS, X0, X+M (gpt-6-sol via the Codex CLI). The three other Codex models stay exploratory. "+M" in every arm name now means the package of item 1.

**4. FS redesign (replaces section 6.4 and A16 for the FS arms).** The FS block holds **practice worked examples only**: it selects from the same fresh examples corpus as the package BEFORE ranking, with the same relevance scorer and the same token counter, under the same maximum of 2,000 tokens, and has **no** briefing, profile, places, procedural or template sections. It is identical for G+FS and O+FS. A shared maximum is an allowance match, not actual token matching: actual token use per presentation is reported next to the package's. FS now controls for an examples-only alternative, not for every component of +M. Any specification change before acceptance gets a timestamped amendment.

**5. Hypotheses (replaces section 2 and A23.3 where they differ).**
- **H1 unchanged (primary, confirmatory):** D1 = SR(G+M) - SR(O0), non-inferiority with margin 5 pp (H0: D1 <= -0.05), one-sided alpha .025, standalone, outside any Holm family, with the procedure validated under A22.4 and recorded in A22.1. If no procedure passes that validation, H1 is estimation-only and no confirmatory claim is made.
- **Superiority (confirmatory, fixed sequence):** tested only if H1's non-inferiority is established; H0: D1 <= 0, one-sided .025, with the same procedure. H1-only validation does not validate superiority at zero: superiority needs its own validity evidence at its null before acceptance, or it is estimation-only.
- **Secondary family:** H2a SR(G+M) - SR(G0), H2c SR(O+M) - SR(O0) and H3a SR(O+M) - SR(G+M) remain confirmatory-secondary under this development-informed amendment. This explicitly supersedes the approved plan's exploratory wording for within-model gains. Removing H4a/b and H5a-c reduces Holm membership from eight to three (alpha .05) and relaxes its thresholds, which is disclosed. H2a/H2c estimate within-model **package** effects (never isolated memory effects); H3a estimates the model difference under common assistance. Each member's null, direction and procedure are pinned before acceptance, and each needs suitable validity evidence (A22.4-style simulation); a member without it is estimation-only. Holm cannot repair an invalid component p-value. No study-wide .05 error-control claim is made across the separate primary and secondary families.
- **Exploratory only (no confirmatory conclusion can come from them):** equivalence |SR(G+M) - SR(O+M)| < 5 pp by TOST (two one-sided tests at .05, i.e. the 90 % interval inside +/-5 pp); the interaction I = (G+M - G0) - (O+M - O0); H4a/b (package vs FS); H5a-c (X arms); every subgroup (level, language). "Not significantly different" is never written as "equivalent". Illustration (Astra): at 10 % discordance even an independent-item approximation gives a 90 % half-width of about 4.7 pp before family clustering, so equivalence is expected to be hard to show.
- The item-level estimand (mean of EN and FA), the weighting, pairing by presentation, family clustering and the infrastructure rules of A21 are unchanged.

**6. L3 executor-feasibility audit (before the freeze; non-test tasks only).** Shared failure can fake parity. Before the freeze, every L3 requirement used by the non-test L3 tasks is checked against the executable honest skills (the reference plans of non-test L3 items run through the executor, no LLM). A failure caused by a general executor defect may be repaired before the freeze; the repair applies to every arm, and everything it touches is re-validated. Feasibility classifications are fixed before acceptance and cannot be revised to excuse reference-plan failures. The audit supports feasibility only for the requirements examined; it does not certify every held-out task. Originally eligible test tasks are kept even if difficult. **No post-result exclusions**: no test item is dropped, reweighted or reclassified after any test outcome. The audit's counts (feasible / infeasible per requirement) are reported. If every arm has low L3 success on the test, the report states that L3 parity says nothing about capability.

**7. Acceptance gate (criteria fixed now, before any of its data exist).**
- *Set:* 24 fresh non-test items, 8 per level, each presented in EN and FA, giving 48 presentations per arm; drawn by the master seed from non-test items not used in practice, the dev pilot, the C3/W3 screens or any rehearsal. Item and family ids, scene seeds and execution order are saved before any call. No presentation is replaced because of its outcome.
- *Arms:* all 8 main arms (Opus calls are paid and count against item 10).
- *Frozen first:* all acceptance rules, retrieval quotas, the evidence-eligibility rule, relevance annotations, feasibility classifications and infrastructure-retry rules are frozen before acceptance.
- *Hard criteria (all must hold):* (a) 0 honesty-gateway violations and 0 decision-path oracle reads; (b) leakage gate 0: no test item id, test instruction text (EN or FA, normalised), test-family paraphrase, or example or template derived from test inspection in the package corpus, the FS corpus or the briefings; (c) 100 % of package records trace to their source (briefing version, practice write_log evidence, the practice episode of each worked example, or the frozen procedural/template file); (d) reset and isolation correct: snapshot and corpus hashes unchanged before and after, and the 0 arms receive no package bytes (G-iso style comparison); (e) every reference plan of an item classified feasible executes; (f) per arm, at least **46/48** plans pass the full common validator (not merely the JSON schema) after at most 2 repairs; (g) package delivery: within 2,000 tokens in 100 % of +M presentations, identical bytes across G+M, O+M and X+M, and no eligible record missing; (h) FS blocks within the same maximum and identical across G+FS and O+FS; (i) measured full-context fit for every Gemma presentation, including two repair turns.
- *Reported, not gated:* retrieval relevance (recall and precision of "needed facts" derived from non-test reference plans and goals; references never populate, rank or repair the package), the empty-package rate per language, tokens per component, and SR per arm.
- **Never gated on Gemma beating Opus**; no SR value is a pass criterion.
- *One correction cycle:* acceptance presentations are development data from their first inspection. After a hard-gate failure, one documented correction cycle addressing that failure is allowed; changes may not be selected using SR or arm differences. The affected artifacts are re-frozen and all eight arms are re-run once on the same 48 presentations. Both attempts are reported completely. A second hard-gate failure stops this protocol. Neither a successful acceptance nor the re-run provides independent evidence of treatment effectiveness.

**8. Early small checks and a proceed/stop decision (NEW; Arash 2026-10-12: "make sure to test stuff even small not time consuming tests, to make sure we are going in the right path").**
- Every build step runs its offline selftests and a small, cheap dev check (free Gemma planning on the dev items, scored by the frozen semantic rubric of DESIGN.md 5.2(b)) before the next step starts. These checks may guide development before acceptance; every iteration is logged in the ledger the same day.
- After acceptance, the report gives Arash the **paired G+M - O0 difference** on the acceptance set, with an interval accounting for translations and families, and **scenario-based power over a preregistered range of effects, discordance and family dependence (the A22.1 tables); this is not an estimated probability that H1 will pass.**
- Arash may then make a development-outcome-informed proceed/stop decision before any test call. This decision does not alter the treatment, hypotheses, sample, analysis or execution rules. The acceptance results and the decision are reported even if testing stops; a stop is reported as "test not run", never as a result. Any subsequent redesign needs a new amendment and the disclosure of all acceptance evidence.

**9. The welcome briefings: Arash's approval.** Before the freeze, Arash reads each kitchen's briefing (EN and FA) as the homeowner, checks its provenance and the bilingual semantic consistency, and approves it. His edits are kept and versioned; the approved version's hash goes into freeze.json. Astra checks the briefings and the package corpus for leakage and for knowledge the stated sources do not support. The briefing script is one template for every kitchen; nothing FloorPlan1-specific is in the code.

**10. Budgets (replaces A22 item 1's caps).**
- **Anthropic total cap: 100 USD** for the whole study (replaces A22's 65). Every Anthropic dollar already spent counts, including all historical Sonnet and Haiku charges; no further calls to those arms are planned (their code is archived, not deleted, A23).
- **Cost gate before the first test call:** spend so far + the remaining-cost projection under the A21.G S2 scenario (S2 is a sensitivity scenario, not a guaranteed upper bound) at the new prompt and package sizes, covering practice, acceptance, the possible complete acceptance re-run, the rehearsal, every retained block, up to 2 repair turns per cell and billed reasoning, **<= 80 USD**, which leaves a **20 USD contingency** inside the cap. If the gate fails, the scope is cut before testing in the cut order of section 3.1 and A20.1.3, outcome-blind; the core four arms, Gemma, the FS arms and X0/X+M are never cut. If the permitted cuts cannot satisfy the cost gate while keeping the protected arms, testing does not begin under this protocol.
- If the 100 USD cap is reached during the test, the run stops at the next block boundary (outcome-blind), and the study is reported incomplete under section 13 with every completed cell analysed as registered.
- The core-four ceiling, the whole-plan ceiling and the A21.B tripwire threshold are re-derived from the new call-plan projection and recorded in A24.1 before any acceptance call, all within the 100 USD cap.
- **Package budget:** hard maximum 2,000 tokens (item 1). **FS:** the same maximum and counter (item 4).

**11. Practice, re-validation and order of work.** Before fresh practice: the allowed practice families are locked and every test family is prohibited (no paraphrases, examples or templates from test inspection); the briefings, the procedural and template content, the retrieval rules and the practice collection/update policy are frozen, including the bootstrap package used at the start of practice and any permitted within-practice updates. Fresh practice snapshots are then collected under ft-v2 (section 6.2 rules otherwise), and the resulting corpus is frozen before acceptance. Then: token gates, leakage gate, all Phase A captures, G-iso (A20.1), the L3 audit (item 6), the acceptance gate (item 7), the proceed/stop decision (item 8), the cost gate (item 10), the freeze (hashes of code, prompts, package corpus, briefings, detector, analysis and A24/A24.1), a non-test dress rehearsal (a substantive fix invalidates the freeze), Arash's sign-off, and only then the locked test (paired scene seeds across arms, randomised execution order, frozen package, reset between presentations, blind scoring, no accuracy peeking, no selective reruns, all planned presentations analysed).

**12. FloorPlan15 and FloorPlan17 (separate registration).** The two other kitchens are a later replication with G0, G+M, O0 and O+M only and the identical procedures, registered before detector adaptation, practice collection or evaluation in either kitchen; all subsequent adaptation and practice outcomes are disclosed. Their detectors are trained under a declared adaptation protocol (about 50 GPU-hours together). Their packages come from their own briefings and their own practice, with the same code. They are not called "unseen kitchens" without that qualification; conclusions extend to those two kitchens only.

**13. Claims, by outcome (replaces section 14 where it differs).**
- H1 passes, superiority fails: "Assisted Gemma was non-inferior to unassisted Opus within 5 pp on FloorPlan1."
- Superiority passes: "Assisted Gemma outperformed unassisted Opus on FloorPlan1."
- H1 fails: "Non-inferiority was not established." Material inferiority is claimed only if the interval lies wholly below -5 pp.
- Exploratory equivalence passes: "exploratory evidence of +/-5 pp equivalence on FloorPlan1"; otherwise "equivalence not established".
- Package gains and the interaction are package effects, never isolated memory effects.
- If every arm has low success, parity says nothing about capability.
- An inference-gate failure removes confirmation for the affected hypothesis; a material integrity failure removes confirmation for the affected analyses. An H1 failure alone does not erase otherwise valid secondary inference.
- **No outcome shows that frontier models generally add little value.**


### A24.2 (2026-10-12, main session from Arash's own words; text corrected by Astra docs/final-test/_astra/A24.2.txt; written after development results, BEFORE any test call or outcome) -- cost-effectiveness outcomes and English-only knowledge
Arash: "we don't have to beat opus necessarily ... we can work with how much less costly it is based on dollar even if accuracy is a little low"; "the knowledge is only saved in english. The only persian thing is user input". The laptop purchase price used for hardware allocation is declared by Arash before the first test call and recorded in freeze.json.

**1. Cost-effectiveness.** This amendment records decisions made after A24 was drafted and is locked before the first final-test call. Cost-effectiveness outcomes are registered key secondary outcomes, reported descriptively with 95% task-family cluster-bootstrap CIs. Resampling preserves matched observations across arms and EN/FA presentations; all arm comparisons use the same task weights.

Planning cost includes all incurred planning attempts, repairs, and arm-specific parsing, embedding, retrieval and package preparation. Shared preparation costs are allocated across the registered test workload using a rule fixed before testing. Executor/simulator costs are excluded. These outcomes therefore describe planning cost, not end-to-end task cost; shared executor software does not guarantee equal execution expenditure.

API planning costs use a dated, archived list-price schedule, applying the relevant input, cached-input and output rates to billed usage, including reasoning and repairs without double counting. Local Gemma planning cost includes measured, non-overlapping GPU and CPU energy plus hardware allocation:

- Energy cost = sampled power integrated over planning time, in joules, divided by 3,600,000, multiplied by USD 0.20/kWh.
- Hardware allocation = declared laptop purchase price / 6,000 operating hours × attributable planning hours.

The energy estimate covers GPU and CPU components. Hardware allocation assumes three years at 2,000 operating hours/year. Electricity-only results and the hardware component are also reported separately; the headline comparison includes both. CIs describe task-sampling uncertainty conditional on these prices and accounting assumptions.

Outcomes are: (a) total planning USD / successful tasks per arm; (b) retained-success-rate ratio R = SR(G+M)/SR(O0), and cost ratio C = mean planning USD per attempted task for G+M / the corresponding mean for O0; and (c) a Pareto plot of success rate against mean planning USD per attempted task, showing all arms. Absolute success rates and planning latency, including repairs and failed attempts, are reported.

Positive cost with zero successes gives infinite USD/success; zero/zero is undefined. Undefined bootstrap ratios are not silently discarded. If valid required CI bounds cannot be obtained, the reading rule cannot be satisfied.

The thresholds are author-chosen interpretation criteria, fixed before final-test data. If H1 is not established, the thesis may state: “Gemma with the package retained at least 80% of unassisted Opus’s success rate at at most 10% of its specified planning cost” only when the lower 95% CI bound of R ≥ 0.80 and the upper 95% CI bound of C ≤ 0.10. Otherwise, estimates and CIs are reported without that claim.

**2. English-only knowledge.** All stored knowledge records, rendered packages and welcome briefings are English for both EN and FA presentations. Only user input varies between English and Persian. FA input accesses knowledge through the A17 parser and cross-lingual bge-m3. Package-budget checks use the English rendering under the existing budget rule.

For matched EN/FA inputs, report the proportion retrieving identical record-ID sets, with both-empty retrievals identified separately. Also report EN and FA success rates and their paired difference with CIs. The bilingual claim is limited to handling EN/FA user input with a shared English knowledge store and English briefings.

### A24.1-H1 (2026-10-12, stage H-H1-stats; the A22.1 record that A22 item 4 and A24 item 5 call for, and the H1-statistics part of A24.1; written after SIMULATION results only, BEFORE any acceptance data and any test outcome; Astra reviews docs/final-test/_astra/H-H1-stats.procedure.txt and H-H1-stats.results.txt, both CHANGE, all points applied) -- the H1 decision rule h1-rule-1, its validation, power, and the exploratory equivalence rule

**1. What happened (disclosure).**
- The audit of A22.4 (docs/final-test/h1_validation/audit.json) found that no earlier number validated the registered H1 rule (final_test_power.h1_decision = wild cluster test AND percentile bound, plus the fixed sequence) as a whole; its wild half alone reached .2165 at H1's level .025 at discordance .05.
- The frozen G-H1-validate screen (protocol in docs/final-test/progress.md, frozen before any simulation; sim/final_test_h1.py, protocol sha 414982f540f7e4e4; 146 configurations x 4,000 simulated datasets on the 42 real family sizes; no outcome data) required every acceptance null error to be <= .0275. **All five candidates failed**: the registered rule C0 had 67 failing cells (worst NI error .2645), bootstrap-t C2 118 (worst .107), CR2-Satterthwaite C5 68 (worst .196), C2 with the K >= 20 guard (C3) 14 (worst .033), C5 with the guard (C6) 5 (worst .0305). Its prespecified consequence was estimation-only H1.
- Before observing any acceptance or test outcome, that consequence is amended to permit ONE additional, versioned procedure, h1-rule-1 (candidate C7). Its guard (K >= 20) and its one-sided computational level (.0175 per tail) were fixed after a disclosed EXPLORATORY simulation (own seeds 20261016 * 1000 + index, 10,000 datasets per configuration on the same 146 configurations; worst null error .0315 at .025, .0284 at .0225, .0253 at .02, .0222 at .0175, .0192 at .015; docs/final-test/h1_validation/calib_explore_summary.C6.json). Those simulations are not acceptance evidence. C7 belongs to a revised protocol; it does not count as passing the original screen. C7 underwent one fresh, prespecified validation (item 3); any failure would have made H1 estimation-only, with no further candidate or recalibration. The registered rule C0 is never used for any claim.
- Development history (old treatment; not an estimate of the revised treatment's effect and not a basis for predicting test power): the development pilot recorded 13/24 (G+M) versus 18/24 (O0) executed successes across 12 items and two languages in one order (A24 item 0).

**2. The H1 rule h1-rule-1 (C7), exactly.**
- Data: per item, the mean of its used paired presentation differences SR(G+M) - SR(O0) (EN and FA; a presentation is used only if both arms have it; an item with none is dropped); per family f the total S_f and the number of kept items n_f; families with n_f = 0 are dropped; N = sum n_f; K = the number of families with at least one discordant used presentation (analyze.item_diffs / discordant_families, unchanged).
- Estimate: est = sum S_f / N (item-weighted). CR2 standard error (intercept-only model of the item differences, working independence, families as clusters): se^2 = sum_f (S_f - n_f est)^2 / (1 - n_f/N) / N^2; Satterthwaite degrees of freedom nu = (sum_f g_f'g_f)^2 / sum_{f,h} (g_f'g_h)^2 with g_f'g_h = c_f c_h (delta_fh n_f - n_f n_h / N), c_f = (1 - n_f/N)^(-1/2) (nu = 27.47 on the real sizes without missing data).
- Bounds: L = est - t_nu(.9825) se, U = est + t_nu(.9825) se. The target one-sided H1 error level remains .025; the revised procedure uses the computational tail level alpha* = .0175. The interval is reported as a "nominal 96.5 % CR2-Satterthwaite interval; actual coverage is not assured" (item 3).
- Wordings (strict inequalities): non-inferior iff L > -.05; superior iff non-inferior AND L > 0 (fixed sequence); materially inferior iff U < -.05; the add-on "SR also statistically lower" iff non-inferior AND U < 0; otherwise "non-inferiority unresolved (inconclusive)".
- Inconclusive (no claim of any kind, never inferior) when: K < 20 (guard), se = 0 or not finite, or nu / the critical value is not finite. Interval threshold crossings do not authorise claims when the guard or another validity condition fails.
- Each primary result reports the estimate, the nominal interval when defined, K, the guard status, the retained item and family counts, and the reason for any inconclusive outcome.
- The margin (5 pp), the estimand, the pairing and the family clustering are unchanged.
- Deployable code: sim/final_test_h1_stats.py `h1_rule_c7` (PROCEDURE_VERSION = "h1-rule-1"), called by sim/final_test/matrix/analyze.py `h1_block` (`H1_PROCEDURE`).

**3. Validation (Stage V100, protocol frozen in the ledger before it ran).**
- Grid: the frozen 146 configurations unchanged plus 80 frozen challenge configurations added before the run (guard transitions where P(K >= 20) is between about .1 and .9; the neighbourhood of the screen's high-discordance failures at d .5-.8 in both skew directions with Delta -.10 / -.05 / 0 / +.05; effect allocation to large vs small families; MCAR missingness 3 % and 6 % per arm-presentation crossed with transition and heterogeneous cells; a probit neighbourhood at p_O .5 / .6 / .7) = 226 configurations; 100,000 simulated datasets in EVERY configuration; seeds 20261013 * 1000 + index (never used before); stage sha e739bc27df577ba7. The RNG scheme is numpy default_rng(base + configuration index) with a distinct base per stage; selftests verified distinct stage/configuration seeds and agreement between the deployable and the simulation implementations.
- Null sets: NI error over Delta <= -.05; materially-inferior error over Delta >= -.05; superiority error over Delta <= 0; negative claims (materially inferior or "statistically lower") over Delta >= 0. Errors and power count ALL generated datasets (a guard failure is no claim). The MNAR stress cells are report-only.
- Gate (all-or-nothing): PASS iff every acceptance null cell has a one-sided 95 % Clopper-Pearson upper bound <= .0275 AND a point rate <= .025.
- **Result: PASS.** 628 acceptance null cells, 0 failures. Largest rate .0210 (materially-inferior error at Delta -.05, d .60, rho .9; Monte Carlo SE .00045; 95 % CP [.0201, .0219]); largest one-sided CP upper bound .0218; largest NI error .0192 and largest superiority error .0189 (both probit p_O .5 heterogeneous challenge cells); largest negative-claim error at Delta >= 0 .0065; MNAR stress <= .0003. Every cell with its rate, MC SE and CP interval: docs/final-test/h1_validation/v100_summary.C7.json.
- The primary item-weighted H1 decision procedure passed the prespecified simulation-validation gate across the frozen acceptance configurations. This establishes neither universal Type I error control nor nominal interval coverage; the per-cell intervals are not simultaneous.
- Superiority, tested only after non-inferiority, has its own null cells in this validation (Delta = 0, -.05, -.10 and challenge cells); its largest error is .0189. This is the separate validity evidence A24 item 5 requires.
- Coverage was evaluated among datasets with a defined interval, without applying the guard: median .964, minimum .82, and below .95 in 36 of 225 configurations (225 because in the 226th, Z-A-identical with no discordance, no interval is ever defined; overall 244,947 of 22.6 million datasets had an undefined interval, mostly at discordance .02-.05). The lowest coverage occurred in low-discordance, strongly dependent, all-or-nothing-family cells, with excessive lower-bound misses. No claims occurred in those cells in V100 because the guard blocked them; this is an observation in these simulations, not a mathematical guarantee. Conditional coverage among guard-passing datasets was not evaluated. All reported intervals are labelled "nominal 96.5 % CR2-Satterthwaite interval; actual coverage is not assured", and low-discordance results additionally flag the observed severe undercoverage.

**4. Views.** The primary H1 view (item-weighted) is the one that passed the gate. The family-weighted and infra-paired views use the same rule and are sensitivities only (not separately validated); a disagreement is reported, never used to change the wording.

**5. Power (scenario-based; docs/final-test/h1_validation/power_summary.C7.json, 78 scenarios x 20,000; seeds 20261014 * 1000 + index).**
- Delta = 0 (parity): non-inferiority was established in 0-15 % of simulated datasets under the beta-binomial model (mean .055 over d .10-.40 x family dependence rho 0 / .3 / .6) and in 3-17 % under the probit model. This design has a very low probability of establishing non-inferiority under parity in the evaluated scenarios: mean power .055 corresponds to a mean 94.5 % inconclusive rate. This scenario average is not a forecast for the experiment. An inconclusive result cannot establish inferiority, equivalence, or the absence of a meaningful difference.
- Delta = +5 pp: non-inferiority 0-53 % (about .50 at d .2-.3 without family dependence, <= .13 at rho .6), superiority 0-17 %. Delta = +10 pp: non-inferiority up to .89, superiority up to .59. Delta = -10 pp: "materially inferior" in 0-18 %.
- In the evaluated low-discordance (d <= .10) or strongly dependent (rho .6) scenarios the guard almost never passed, producing predominantly inconclusive H1 outcomes.
- Budget note recorded for Arash before any test spending: if the main purpose of the test expenditure is to demonstrate non-inferiority when performance is equal, these simulations indicate that the current design is unlikely to achieve it; proceeding should reflect the value of the measurements and the other planned analyses even if H1 remains unresolved. No hypothesis is changed by this note.

**6. Exploratory equivalence E1 (A24 item 5; never confirmatory).** |SR(G+M) - SR(O+M)| < 5 pp is assessed by a separate rule E1: the same CR2 + Satterthwaite algebra with the plain 90 % interval (t_nu(.95); TOST at .05 per side; the C7 calibration is NOT transferred), the same guard K >= 20 and the same handling of undefined se / degrees of freedom (inconclusive); "exploratory evidence of +/-5 pp equivalence" iff L > -.05 and U < +.05, otherwise "equivalence not established". Simulation (222 scenarios x 20,000, seeds 20261015 * 1000 + index; docs/final-test/h1_validation/equiv_summary.E1.json): its largest observed simulated error rate (an equivalence claim when |Delta| >= .05, at and outside both boundaries, including guard transitions and probit cells) was .0068; the largest observed power at |Delta| <= .025 was .018. Equivalence claims were very rare across the evaluated near-equivalence scenarios; this is not a universal bound on power. The report states the interval and its containment, and "not significantly different" is never written as "equivalent".

**7. Evidence and code (hashes go into freeze.json).** sim/final_test_h1_stats.py (V100 ran from the copy sim/final_test/_checkpoints/H-H1-stats/final_test_h1_stats.py.frozen-before-V100), sim/final_test_h1.py (screen, unchanged), docs/final-test/h1_validation/{screen,calib_explore,v100,power,equiv}*.json[l] and logs; selftests sim/_ftm_h1_stats_selftest.py (35), sim/_ftm_h1_selftest.py (34), sim/_ftm_analyze_h1_selftest.py (19: analyze's H1 decision equals h1_rule_c7 on the same family totals). The ledger entries "STAGE H-H1-stats" are the evidence manifest of this amendment.

**8. Not changed by this amendment.** The margin, the target alpha, the estimand, the item and family definitions, the claims table of A24 item 13 (the H1 interval is printed as the nominal 96.5 % interval), and the secondary Holm family. This amendment provides no validity evidence for H2a, H2c or H3a (A24 item 5 still requires it before acceptance) and none for any subgroup. The other parts of A24.1 (ft-v2 texts, validator, summary rule, glossary, package quotas, ceilings) are recorded by their own stages.

### A25 (2026-10-13, stage J-A25, from Arash's round-2 decisions and Astra's round-2 list docs/final-test/kitchen_memory/astra/round2-plan.txt; Astra review docs/final-test/kitchen_memory/astra/J-A25.txt, verdict CHANGE, all 9 required changes and the 3 suggestions applied; second look docs/final-test/kitchen_memory/astra/J-A25.round2.txt, 6 applied / 3 partly, its 3 blocking insertions applied verbatim in items 5.5, 7 and 10; written AFTER development results (signal rounds 1-5) and BEFORE any acceptance call and any test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- round-2 to round-5 package and executor changes, arm allocation, skill contracts, the fresh development set, acceptance exposure, regression matrix, budgets, versions and the road to the locked test

**0. Chronology and status (disclosure).** This amendment follows observed development failures; it is not part of the original specification. It is prospective only for what has not yet happened (briefing approval, practice, acceptance, freeze, rehearsal, sign-off, locked test). Development-informed mechanisms may legitimately enter the frozen confirmatory treatment: the untouched test evaluates that final bundle, not whether its development was independent of observed failures. No test item has been planned, executed or scored under any version of this treatment. **No acceptance item has received an LLM-generated plan or a formal acceptance evaluation. Two reserved items received reference-plan execution, and g-l3-tool-salt additionally informed executor development; their text and reference plans, and those of the other reserved items, were also read by offline builders (item 5.3). Acceptance is therefore partly development-exposed; the locked test remains outcome-blind.**
- **Round 1 is exploratory.** The round-1 fixes (I-F1 goal schema, I-F2 recipes and affordance map, I-F3 precision pass; profile pkg-1.r1) were written after reading the three swing cells of the 24 dev signal presentations (12 items x EN/FA, FloorPlan1). Round 1 executed G0 13, GP 17, O0 19 of 24 (docs/final-test/kitchen_memory/round1.json). The I-close review then found part of that gain in-sample (recipe rows tuned on the items they helped) and removed those rows from their own items (`tuned_on` provenance).
- **Rounds 2-5 (development; the original 24 kept as a regression set, plus a fresh dev set of 22 presentations from round 2 on, item 5).** Executed goals G0 / GP / O0:
  - round 2 (new executor, pkg-1.r1 + I-close/J fixes): original 14 / 16 / 19, fresh 16 / 16 / 22. NOT MET. Two original GP cells received an affordance line tuned on them (a leave-out leak found in round 3, fixed by F0); round 2 is reported as observed, with that protocol violation, not corrected by subtraction.
  - round 3 (pkg-1.r3): original 14 / 15 / 19, fresh 16 / 20 / 22 (rep2 fresh 15 / 22 / 21). NOT MET on the original set.
  - round 4 (pkg-1.r4): original 14 / 16 / 19, fresh 16 / 22 / 22 (rep2 identical except fresh GP 21). NOT MET on the original set (short by 1).
  - round 5 (pkg-1.r4 unchanged, executor tidy correction T1, item 2.2): original 16 / 17 / 19 in both runs; fresh 15 / 22 / 20 (run 1) and 16 / 20 / 22 (rep2). Development target met with all arm-eligible mechanisms enabled (UNIVERSAL) in both runs; the STRICT condition (item 2.5) fails on the original set (15 of 24 in both runs). 0 honesty-gateway violations and 0 oracle reads in every cell of every round.
  - **Replication status:** rounds 3-5 each had one predeclared paired repeat that re-executed the SAME plans on the SAME seeds; round 5 re-used the frozen round-4 plans (prompts and package unchanged); Opus plans in rounds 3-5 came from the response cache. These repeats measure execution noise only; they are not independent planner replications.
- **Development target and its status.** Arash's development goal (2026-10-12; raised to 90 % for the main session on 2026-10-13) was GP >= 0.85 x O0 and GP >= G0 with no new violations, on the original 24 AND on the fresh set. It is a development criterion only, a count ratio on development data; it does not enter any hypothesis, gate or claim of this protocol, and passing it is not evidence of generalisation or an "as good as Opus" claim. The 90 % aim was not reached on the original set (17/19 = .895 in goals, i.e. GP 17 vs ceil(0.90 x 19) = 18).
- **Development budget used (Astra round 2: at most 3 further rounds after round 2 unless Arash extends):** rounds 3, 4 and 5 were all used; the budget is exhausted. No further repair is chosen from the residual failures of these 46 presentations (Astra round5-repair). Anthropic spend in workflows I-J: round 1 0.22 USD, round 2 about 0.30 USD (cap 1.5 per round), rounds 3-5 0.00; the study-wide total is read from the A15 cost guard ledger at the A24 item 10 cost gate. Simulator time and Gemma compute of the rounds are in the per-round files; they are development cost, not test cost.
- **Residual failures, named (not repaired):** the only Knife in FloorPlan1 has zero rendered pixels from every agent pose checked (EXEC-DIAG), so tool-cut and the Knife step of cutlery-sink (4 original presentations) are unreachable for every honest arm, Opus included; the chilled-apple "put in the Fridge and close it" row is tuned on that item and stays left out for it (2 presentations); the Persian text of g-l2-microwave-run-egg does not ask to close the door while the English text does (a dataset mismatch for Arash; not patched in the parser or the package); Persian tool-flip plans `search_for Pan` instead of picking up the Spatula, and no general mechanism remains that does not hard-wire "flip -> Spatula".
- **Evidence manifest:** the dated entries of docs/final-test/progress.md from "ROUND 1 CHECK" to "Q-round5 CHECK DONE" and the files they name (round1-5.json, diagnosis_r3/r4/r5.jsonl, EXEC-DIAG.md, signal_set_r2_fresh.json, the Astra files under docs/final-test/kitchen_memory/astra/).

**1. What the treatment +M now is (extends A24 item 1; English-only per A24.2).** **+M is a bundled intervention comprising English knowledge, parser-derived instruction assistance, and privileged briefing-guided executor search. H1 compares Gemma with this bundle against Opus without it. Package contrasts do not isolate memory, translation, knowledge, or search-policy effects.** Every planner-side element below is rendered by deterministic code, once per presentation, as identical bytes for G+M, O+M and X+M; the 0 arms and the FS arms receive none of it.
1. **Persian dictionary inside the parser (Arash, option (b), 2026-10-13).** sim/goal_schema.py maps Persian kitchen words and phrasings (objects, verbs, negated verbs, microwave start phrases, purpose clauses, spelling variants; Fork / Spoon / Knife / ButterKnife kept distinct) to the same English action and object keys as the English sentence. Declared policies: an unknown noun stays a visible "?" and is never mapped to a known type; ببر is carry only with a destination in its own clause, otherwise cut, and every use is recorded as an ambiguity; "something to X with" is one tool need, never an X step. The dictionary itself is never rendered. **Provenance:** the entries were written as general kitchen language, but the measurement parsed the EN/FA text of all 116 non-test items (acceptance items included) and the round-2 diagnosis read the dev failure cells, so the dictionary is not independent of observed instructions. **Provenance records distinguish item inspection, diagnostic evaluation, and changes informed by their results. Absence of an item ID is not evidence of independence.** EN/FA key agreement on the 116 non-test items: 115 of 116 (the miss is the egg text mismatch) -- an in-sample consistency figure, not an accuracy estimate.
2. **Parser-derived English instruction assistance (an EXPANSION beyond option (b); Astra round3/round4-repair).** Lines rendered from the parser's reading of the user's own instruction: **F1** "This request names: <movable types>." plus, for a purpose need, "A tool that fits the request: <type>." (from the English affordance table; suppressed when that operation's KB row is left out for the item); **F2** a stated fact whose parser affordance is a destination (bad -> GarbageCan) reaches the pre-existing discard line, with the state shown as read; **G1** (`request_ops`) "The request in English: ..." (keeps negation, arguments, destinations, order and stated attributes; falls back to the F1 line whenever the parser is not "ok", left anything unresolved, met an ambiguous alias or an untyped object; never renders an implied operation such as cold -> chill, never a tool type for a vague purpose; fixtures as plain nouns). The parser's role thus changes from retrieval-only to planner assistance for +M. **For ALL parser-derived assistance actually delivered (F1 and F2 always; G1 if accepted), this supersedes A24 item 2's rule that a Persian-to-type glossary is given "to every arm's FA presentations or to none (never to +M only)".** These lines are a component of +M, appear in EN and FA presentations alike, and are part of every package effect (H1, H2a, H2c). No glossary is given to the 0 or FS arms. G1 enters the final treatment only if Arash accepts this expansion in writing before practice (item 11.1); otherwise `request_ops` is off for every +M presentation, and the value is pinned in freeze.json.
3. **Procedural lines checked against the skill code (item 4).** Recipe and rule lines may only state what sim/honest_skills.py does: F3 contract facts ("A Microwave only switches on while its door is closed"; put_in / pick_and_place "may leave its door open"; an explicit open stays open until a close step); G2 precondition closure (a requested, non-negated toggle_on of a Microwave keeps the close line with "Needed here: a Microwave must be closed when toggle_on runs"). No order advice tuned on the microwave items is written.
4. **Leave-out (F0) and provenance.** The item-level leave-out filters affordance rows and tool-request lines as well as candidates. For the **test** items no exclusion is needed only if the provenance check (A24 item 7(b), widened here) shows that no store record, example or template was derived from them; the check covers support citations, `tuned_on` lists and the dated inspection log, not just item ids. For the **acceptance** items the stores DO cite them (item 5.3), so each acceptance presentation uses the same item-level leave-out (support + `tuned_on`) as the dev rounds, and the acceptance report lists the records dropped per item.
5. **Package profile and budget.** Profile `pkg-1.r4` (A24 item 1 retrieval + goal-schema query + precision pass, 1-3 task records, 300-token cap inside A24's 2,000-token maximum; real tokens = max over the Gemma and Anthropic tokenizers). Development maximum observed 296 tokens; no Persian character in any package.

**2. Arm allocation (declared).**
| element | 0 arms (G0, O0, X0) | +M arms (G+M, O+M, X+M) | FS arms |
|---|---|---|---|
| ft-v2 prompt, tool contract, validator, <= 2 repairs | yes | yes | yes |
| parser dictionary (retrieval key) and F1 / F2 / F3 / G2 lines; G1 if accepted | no | yes | no |
| briefing-guided search order (executor hook B) | no | yes | no |
| container-aware search E3, protruding-pixel aim E1a, identity check E2, tidy correction T1 | yes | yes | yes |
| purpose-family substitution P | off | off | off |
1. **Briefing-guided search ONLY for the +M arms (Arash).** The executor puts the container types that the approved welcome briefing names as the usual storage of the searched type FIRST in its search order; every other place stays in the search. **Wrong hints preserve eventual search coverage only when the episode budget permits; they can reduce success under finite budgets.** The hook can only be built with `allow_briefing_hints=True` (a constructor without it raises); each use is logged (`briefing_log`, `search_info["briefing"]`). It is part of +M because the briefing is the declared privileged channel (A24 item 1.1) and is not supplied to the 0 arms. The development rounds used the UNAPPROVED kitchen-1 draft briefing; the test uses only the approved version.
2. **Shared executor upgrades, ALL arms (Arash; Astra: a shared executor that repairs planner omissions narrows what is measured about planning).** E3 container-aware memory: while a container stands open, every movable type the robot's own detector finds inside is written to that inspection record; a later search for that type revisits it first (at most 2 revisits). E1a aim: the pickup pixel is chosen among pixels that protrude from a plane fitted to a ring around the box (threshold max(0.8 %, 3 robust sigmas)); one attempt sends at most 3 pixels, the legacy centre point always the last (J-regress fix for hollow objects). E2 identity check (item 6). T1 tidy correction (round 5): the end-of-step tidy closes a door this agent opened implicitly before a later toggle_on of the same appliance whose contract is "switches on only while closed" (`SWITCH_ON_CLOSED = {Microwave}`); explicit opens, toggle_off and every other type unchanged. T1 is an executor-policy correction found through development failures; it also helps Gemma alone (G0 14 -> 16 on the original set) and is not a package effect.
3. **Purpose-family substitution (P) stays OFF for every arm** (whether a ButterKnife slices in the simulator is unverified).
4. **Decisions use observed data only.** None of E1a, E2, E3, T1 or B reads simulator object metadata; they use the agent's own images, depth, detections and its own record of what it opened. Simulator positions serve only resets, scoring and isolated audits.
5. **UNIVERSAL and STRICT (development reporting only).** UNIVERSAL = all arm-eligible mechanisms enabled (B stays +M-only, P stays off). STRICT = each mechanism off for the items that motivated it (request_ops off on potato and tool-flip, precondition_closure off on egg, T1 off on the whole egg and potato items). STRICT is a development sensitivity of the 46 presentations. Acceptance and test use one frozen configuration: UNIVERSAL, or UNIVERSAL without request_ops (item 1.2).

**3. Exploratory Codex models are plan-only (Arash, 2026-10-12).** GPT6-Astra, GPT6-Luna and GPT5.6-Terra still plan (0 and +M) and are scored on plan validity, PC and TDS only; they are NOT executed in the simulator. GPT6-Sol (X0, X+M) stays a main, executed arm. The call plan, plan-first and analyze.py are updated before freeze.json; no hypothesis changes (the exploratory Codex arms were never confirmatory).

**4. Skill contracts and success predicates (locked; gate sim/_pkg_contracts_selftest.py).** The contract is read from sim/honest_skills.py by AST (never imported); planted changes to the skill source or to the KB are caught.
- **heat = cook:** ONE route: hold the food, put it in the Microwave, close, switch on, switch off, open, pick the food up: it ends back in the hand. Success predicate: the item's goal as written (for example "Potato isCooked = true", "Potato in Plate").
- **Stay-inside route (proc:microwave-run):** put in, close, switch on; the food stays inside and is taken out only if the command says so. Predicate: item in Microwave, Microwave closed, Microwave on. Never triggered by heat/cook; dropped for a "turn off" request.
- **Fridge as destination:** put in, then close (the close is a separate step; put does not close). Predicate: the item's goal (it may require the door closed).
- **Tool request ("something to X with", "a knife to cut the tomato with"):** pickup <Tool> and stop. Predicate: the tool is held. Any unrequested action is an instruction violation, counted per cell by `unrequested_actions(plan, ops, types, tool_request)` on effects (toggle_on of a Microwave = heat; heat = cook; an unrequested slice counts). It is reported next to the goal outcome; the primary outcome (item 10) is unchanged.
- `pick_and_place` ensures holding, then puts; `search_for` / `go_to` search exhaustively and never pick up; a pickup of a specialist-class type is identity-checked (item 6).

**5. The fresh development set and the acceptance split (J-fresh-set; docs/final-test/kitchen_memory/signal_set_r2_fresh.json; builder sim/final_test/signal/build_fresh_r2.py, master seed 20261010).**
1. **Pool:** non-test, reviewed, EN and FA present, reference plan, ok in the FloorPlan1 validation, minus the 12 signal items, manifest-exposed items, practice items, C3/W3 screen items and inapplicable items: 40 items (L1 18, L2 17, L3 5). Existing dataset items only; no new instruction text.
2. **Split (acceptance reserved FIRST):** within each hard kind a seeded shuffle alternates items between acceptance and fresh. Acceptance 19 items (L1 8, L2 8, L3 3); fresh 11 items = 22 presentations (L1 4, L2 5, L3 2); spare 10 items used by nobody (19 + 11 + 10 = 40; all ids in the file). Overlap of fresh with the 24, acceptance, practice and screens: none. Fresh items share families with practice items (fresh items, not held-out families) and are development data from their first inspection.
3. **Exposure of the reserved acceptance items (none planned by any model):**
| exposure | acceptance items |
|---|---|
| reference plan executed, no LLM (A24 item 6 L3 audit) | g-l3-tool-salt, g-l3-tool-soup |
| executor diagnosis and fixes informed by it (J-exec-diag, J-R1: occlusion, identity check, aim) | g-l3-tool-salt |
| cited as support in sim/step5/procedures_kb.json (builders read non-test instructions and reference plans) | 8 L2 items (cabinet-pepper-shaker, find-pickup-bowl, find-pickup-mug, find-place-dish-sponge-sink-basin, find-place-spatula-sink-basin, fridge-potato, microwave-bread, microwave-mug) |
| cited as support in sim/step5/l3_templates.json | 8 items (pickup-butter-knife, pickup-cup, pickup-soap-bottle, pickup-spatula, fridge-potato, cold-bread, tool-salt, tool-soup) |
| text parsed in aggregate measurements (goal schema, parser, wide package sanity over the 116 non-test items) | all 19 |
| none of the above except aggregate parsing | g-l1-pickup-pot, g-l1-pickup-tomato, g-l1-place-plate-counter-top, g-l1-place-spoon-sink-basin |
4. **Acceptance composition: an explicit, dated choice by Arash before practice (no silent default).** A24 item 7 registered 24 fresh items, 8 per level, 48 presentations; only 3 unused non-test L3 items exist, and **g-l3-tool-salt and g-l3-tool-soup remain mandatory regression checks but cannot establish acceptance coverage on unseen L3 items.** Arash chooses ONE of: (a) restore the registered composition with newly authored, independently reviewed L3 items (non-test, non-practice families, EN/FA, reference plans), added before any acceptance call, so that 8 L3 items without executor exposure exist; or (b) amend the size and level coverage explicitly: the acceptance set is the 19 reserved items (38 presentations per arm), the unseen-L3 coverage requirement is waived and stated as waived, and every denominator-dependent threshold keeps its registered pass RATE: criterion (f) becomes at least **37/38** valid plans per arm (46/48 = .958; 37/38 = .974 is the nearest count not below it); criteria (a)-(e) and (g)-(i) are 0-tolerance or 100 % criteria and do not depend on the denominator. No used item is recycled into acceptance.
5. **Item-level provenance ledger (before practice).** Before practice, complete a dated item-level ledger for all 19 reserved items covering inspection, diagnostic evaluation and resulting changes, marking unknown provenance explicitly. The 15 support-cited items remain development-exposed after leave-out; exclusion removes attributable records, not prior influence on shared parser or template design.

**6. Identity check: policy and costs (all arms).** After a successful pickup of Fork, Spoon, ButterKnife, Knife, Spatula, SaltShaker, PepperShaker, Plate, Pot or Pan: one extra look at the hand. **One look = one camera frame passed through BOTH detector models (primary YOLO26s and specialist YOLO26m), 0 simulator primitives**; it does not write to memory. Only detections within 12 cm of the nearest depth can be the held object. A different specialist-class type at confidence >= 0.50 with the asked type not within 0.15 => drop and a retryable wrong_target, which costs one of the step's 3 retries; at most 2 swap-backs per interaction, the third wrong object fails the step honestly. Doubt keeps the pickup. The dropped object is not placed back (no scene-restoration claim). Measured on the 40 step-4 items: 6 looks, 0 extra primitives, 0 swaps, none wrong; isolated item timing 63.1 s with vs 64.7 s without (within noise). Per episode `identity_stats` (checks, detector_calls, wrong, unclear, swaps, extra_prims) are logged and reported per arm.

**7. Execution budgets and charges (identical for every arm; frozen with the executor).** Retries per step `MAX_RETRIES = 3`; primitives per top-level step `STEP_PRIM_CAP = 700`; extra search primitives per step `SEARCH_PRIM_CAP = 2500`; primitives per episode `EP_PRIM_CAP = 3000`; container inspections per search `MAX_INSPECT = 40`; place instances per exhaustive search `MAX_EXH_PLACES = 80`; repositions after "not visible" `MAX_REPOSITION = 3`. Charges: each refused aim pixel = 2 cheap primitives toward the step cap; an identity look = 0 primitives and one two-model detector pass; a swap-back = one drop primitive plus one retry; a wrong briefing hint = the visits it costs, inside the same caps. Planner side: <= 2 repair turns (A24 item 2), Gemma num_ctx 8192 and 768 output tokens, Opus max 8000 tokens with adaptive thinking (effort medium); package <= 300 tokens of the 2,000 maximum. Before practice, freeze a budget table for every executed arm, including X and FS, specifying planning context/input and output limits, repair limits, search and primitive limits, and perception limits or an explicit absence of a separate cap (currently there is no separate perception cap: detector passes are bounded only through the primitive, viewpoint and retry caps above). Count both detector-model invocations per look, charge every aim attempt and retry, and report detector compute alongside elapsed time, Gemma energy and API spend, per arm.

**8. Executor regression matrix (before the freeze, and after any executor change).** Offline (sim/_step4_exec_r1_selftest.py, _step4_tidy_e1_selftest.py, _step4_act_selftest.py): drawer/door closed -> open -> look -> detect -> pick, open container, low and high cabinet, occlusion (salt behind pepper), empty container, wrong hint, unreachable object, exhausted search budget, visible but never interactable (reported as action_refused / retry_exhausted, not not_detected), salt/pepper swap and false rejection, T1 on/off. Live, FloorPlan1 reference plans, no LLM: the 40 step-4 items (round 5: 40/40 goal, 39 strict, 0 violations, 0 oracle reads) and the L3 audit (26 of 28 classes unchanged; 2 known flaky perception items that T1 cannot reach). A change after which a previously ok item fails reproducibly is reverted.

**9. Visibility audit (before acceptance).** Every type a non-test item needs is checked for rendered pixels from reachable poses in all three kitchens (FloorPlan1, 15, 17), with simulator data used only as an audit aid, never in agent decisions. Results feed the A24 item 6 feasibility classification, fixed before acceptance. No test item is removed, replaced or reweighted because of it (A24 item 6); an unobservable object in a test item is reported as a shared ceiling.

**10. Primary endpoint and analysis (restated; A24 item 5 and A24.1-H1 control any conflict with development text).** Primary outcome: binary goal success of each executed presentation (registered goal predicate, blind scoring). Estimand: per item, the mean over its paired EN and FA presentations of SR(G+M) - SR(O0); item-weighted over items, families as clusters. H1: non-inferiority with margin 5 pp, H0: Delta <= -0.05, decided by h1-rule-1: non-inferior iff the lower bound of the nominal 96.5 % CR2-Satterthwaite interval (computational tail .0175) exceeds -0.05, with the guard K >= 20 discordant families; guard failure, se = 0 or non-finite degrees of freedom = inconclusive, never inferior. Superiority only after non-inferiority (fixed sequence). A presentation is used only if both arms have it; an item with none is dropped; missing data, infrastructure retries and stopping follow A21 and section 13; the test size is the registered test set (no interim looks, no extension, no selective reruns). The A24.2 retained-success ratio R = SR(G+M)/SR(O0) and cost ratio C are key secondary, descriptive, with cluster-bootstrap CIs and A24.2's zero-denominator rule (positive cost with zero successes = infinite; zero/zero undefined, never silently dropped). For R, SR(O0) = 0 with SR(G+M) > 0 gives infinity; both zero gives undefined. Before practice, a binding analysis addendum (A25.1) must state the numerical test size, the C formula and its undefined cases, and the registered missing-data and infrastructure-retry rules, resolving the conflict between paired exclusions and "every planned presentation analysed" (every planned presentation enters the analysis record; the addendum states exactly when a presentation counts as missing rather than failed). Overlap (whether G+M succeeds on the items O0 solves) is reported descriptively from the paired table; it is not a decision rule. **The development ratio and every development threshold stay entirely outside these decisions.**

**11. Pinned decisions, correction cycle and the untouched final-test procedure.**
1. **Before practice, pinned in writing (dated ledger entry) and then in freeze.json:** G1 accepted or not (item 1.2); the approved English briefings (A24 item 9; A24.2); the instruction corrections Arash makes (for example the Persian egg text; any corrected text is fixed before practice and applies to every arm); the acceptance composition (item 5.4); and the permitted correction categories below.
2. Visibility audit (item 9); L3 feasibility classification fixed.
3. Fresh practice snapshots under ft-v2 with the frozen bootstrap package (A24 item 11); the examples corpus frozen.
4. **Acceptance** (A24 item 7 as set by item 5.4), all main arms. **Acceptance permits one correction cycle for predefined implementation or contract failures only** (a hard criterion (a)-(i) failing because of a bug, a missing or untraceable record, a token or context overflow, an isolation or reset fault, or a skill-contract mismatch); **outcomes cannot select treatment variants or trigger efficacy-driven tuning. Preserve and report both attempts. Failure after the permitted cycle stops progression.**
5. Proceed/stop decision (A24 item 8); cost gate (A24 item 10).
6. **Freeze**, with complete hashes and configurations: executor, package code and stores, parser (goal_schema and lexicon), briefings, examples and FS selection, L3 templates, prompts and validator, planners (model ids, quantisation, settings, tokenizer versions), Codex/X configuration, detectors, scorer, analysis code, response-cache policy (no cached plan from development is reused in the test), randomisation seeds and execution order, A24, A24.1, A24.2 and A25.
7. **Non-test dress rehearsal.** A substantive fix (any change to a frozen hash) invalidates the freeze: the affected gates (token and context fit, leakage/provenance, G-iso, the executor regression matrix of item 8 if the executor changed, the package gates if the package changed) are re-run, a new freeze is written and Arash signs off again. Then Arash's sign-off.
8. **Locked test:** paired scene seeds across arms, randomised execution order, frozen package, reset between presentations, blind scoring, no accuracy peeking, no selective reruns, every planned presentation analysed.
Executor and tool changes stop at the freeze. Neither the original 24 nor the fresh 22 is used again for any repair.

**12. Versions (development state at round 5; the final values are recorded at the freeze, item 11.6).**
- Executor sim/honest_skills.py sha256 5f18e5d9ebb4fca1a3e035ea87be2aa577fe84ea2a7f354ee3b2ec142c1781ce (rounds 2-4: b601d68f...; pre-round-2 copy in sim/final_test/_checkpoints/J-R1-executor/).
- Package: sim/ltm_package.py 14717db3..., sim/ltm_procedural.py 82e0e672..., sim/step5/procedures_kb.json f1c0c762..., sim/goal_schema.py 82feb2f2...; profile pkg-1.r4 params sha 41133f8d002f55fa (LP.profile_params(LP.PROFILE_R4)); leave-out sim/final_test/signal/signal_pkg_plan.py ae24f5f6...; dev scorer sim/final_test/signal/score_equiv.py c199165a... (close-folding fix of round 2). Full copies: sim/final_test/_checkpoints/Q-round5/.
- Planners: Gemma `gemma4:e2b-it-q4_K_M` (Ollama); Opus `claude-opus-5-5`. Detectors: the FloorPlan1 pair of step 3.
- The development rounds ran the planner BEFORE the start sweep (no observation summary in any prompt, all arms equal). The test runs it AFTER the sweep (A11, A24 item 2), so development numbers do not transfer to the test configuration and are not used to predict it.

**13. Claims (narrowed).** Every +M effect is the effect of the whole bundle of item 1 (knowledge, parser-derived assistance, briefing-guided search); no component's effect is identified. Executor gains (E1a, E2, E3, T1) are shared and never attributed to the package. Rounds 1-5 are reported as development history, labelled tuned where they were tuned, and are never cited as evidence that Gemma + package matches Opus.

### A27 (2026-10-13, stage P-A27, from Arash's acceptance of Astra's run-time review docs/final-test/_astra/runtime-cuts.txt; DRAFT -- PENDING ARASH SIGN-OFF AT THE FREEZE; written AFTER development results and BEFORE any acceptance call, test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- the run-time plan
**Status.** Arash approved the plan in words: "Accept astra's vision. it's logical." (ledger entry "ARASH DECISION: accept Astra's run-time plan", 2026-10-13). This text is a DRAFT until he signs it off at the freeze (A25 item 11.6). If he declines, the code returns to the pre-A27 copies in sim/final_test/_checkpoints/P-A27/. The number A26 is kept for the round-6 draft of stage K-B2 (Knife archive, cutting tool in hand).
1. **Kept unchanged.**
   - All 8 main arms (G0, G+M, O0, O+M, G+FS, O+FS, X0, X+M) on all 250 test presentations at r = 0.
   - Primary scoring on r = 0 / run 0; bilingual pairing; family clustering; frozen memory; H1 (h1-rule-1) and the fixed sequence; the Holm family of A23.
   - The relocation block (G0, G+M, O0, O+M on the 20 relocation presentations).
   - The executor's default controls: the P0 / P+M reference plans and the G+M/p, O+M/p replays.
   - The exact execution-unit dedup (the A21 D key). It is not broadened.
   - Cost accounting, the 80 USD gate and the 100 USD cap (A24.10).
2. **REPEAT (replaces "every arm gets 3 independent calls" in sections 6 and 7, the "3 runs per cached plan" of the variance decomposition, A23 item 4 "REPEAT r = 1, 2: the 8 remaining main arms", and the declared 3-run narrowing in execute.py).**
   - Planner calls r = 1, 2 are made only for the core four (G0, G+M, O0, O+M). G+FS, O+FS, X0 and X+M have the r = 0 call only.
   - Execution: on a REPEAT presentation, every distinct execution unit that carries a core-four cell of block main or repeat, or a /p replay of a core +M cell, runs **2** times (was 3), each from an independent restore. This is "3 planner calls and 2 executions per distinct plan".
   - Every other unit runs once: FS, X0 / X+M, curve, relocation and P0 / P+M. A unit shared by a core cell and another cell is still one execution that runs twice.
   - The variance decomposition becomes: planner variance from the 3 calls, execution variance from the 2 runs per distinct plan, core four only.
   - Lost: no repeatability estimate for FS or X0 / X+M, and a weaker execution-noise estimate. H5a-c use r = 0 and are unaffected.
3. **Exploratory Codex models are plan-only.** This confirms A25 item 3 and replaces A22 item 2 ("Run everything ... Plans-only scoring is not used") for these three models only.
   - GPT6-Astra, GPT6-Luna and GPT5.6-Terra still plan (0 and +M, r = 0, 1, 2 on the REPEAT presentations).
   - They are never executed. They are reported as planning diagnostics: plan validity, first-pass validity, PC and TDS, in docs/final-test/matrix/<namespace>/plan_diagnostics.json.
   - They have no SR (not even 0), are not in score.jsonl, and are never pooled with the executed arms. plan_diagnostics.json is sealed with the score seal (its sha is checked on every resume); it is always written, empty when every plan-only cell is cut.
   - Lost: no embodied success, retained-accuracy or USD-per-success comparison for these three models.
4. **Learning curve (replaces the 3 practice orders of section 6.3 E4).**
   - Practice order 0 only: K = 1, 3, 5, plus the main K = 10 (= S*, shared with the main matrix); K = 0 is the 0 arm.
   - Orders 1 and 2 are dropped. Practice runs `practice_k.py --orders 0`.
   - Three points on different orders were rejected, because K and order would be confounded.
   - Lost: practice-order robustness of the curve. The cut group curve12 is now empty and stays in the registered cut order.
5. **Rejected (Astra and Arash):** lower resolution or render quality; early stopping or new action / time caps; sequential or adaptive testing; two simulators; overlapping planning with execution; broadening the dedup.
6. **Workload.** Source: docs/final-test/matrix/runtime_a27.json (`python sim/final_test_callplan.py --runtime-a27 --write`). Counts are executions before dedup, at 19 s per episode (the practice-v1 mean, 533 s / 28 episodes). "Before A27" is the pre-A27 implementation baseline: A25 item 3 had already declared the exploratory Codex models plan-only, but the code still executed them.

| | before A27 | A27 |
|---|---|---|
| planner cells (first pass) | 5,440 | 4,000 (1,080 of them plan-only Codex) |
| Opus first-pass calls | 1,810 | 1,210 (core 740 unchanged) |
| listed matrix (plan arms) | 8,320 episodes = 43.9 h | 3,640 = 19.2 h (2,000 main + 1,200 additional REPEAT execution overhead of the core four [240 extra main-block + 960 repeat-block executions] + 360 curve + 80 relocation) |
| with P0 / P+M (540) and /p (A27: 1,140; before: 1,500) controls | 10,360 = 54.7 h | 5,320 = 28.1 h |
| uniform dedup 0.778 (dev rehearsal, all blocks) | 8,060 = 42.5 h | 4,139 = 21.8 h |
| dev per-block dedup (main 0.623, curve 1.0; blocks the rehearsal did not run kept at 1.0) | 8,777 = 46.3 h | 4,053 = 21.4 h |
| cost-gate projection (S2, package-maximum blocks, spend 0 USD) | 69.20 USD | 47.94 USD |

   - The dedup ratios come from the dev rehearsal (Gemma, reference and replay arms only; no Opus, Codex, REPEAT block or relocation). They are assumptions, not test numbers.
   - Scope: these episode counts cover the test-v1 execution matrix only. NOT counted: practice, the acceptance gate and its possible re-run, the dress and thermal rehearsals, Phase A capture (its own timing), the G-iso audit (a full default G-iso adds about 36 executions, ~0.19 h), crash and blocked-state re-runs, hang-guard relaunches and cooling. The USD projection includes acceptance and rehearsal allowances; the episode counts do not.
   - Provisional allowance: **25-32 h elapsed per kitchen** for the test matrix, including capture and operational overhead. This is not a measured bound.
7. **Before the freeze (Astra; part of the A25 item 11 road):**
   - measure dedup per block;
   - profile reset / setup / sweep / navigation / detection / logging;
   - measure mean and tail durations, restore failures and retries;
   - run a sustained thermal rehearsal, including small objects, containers and difficult navigation;
   - judge any optimisation on trajectories, detections and goal outcomes, not on SR alone.
   - A shared execution is a benchmark saving, not a cheaper independent deployment, and a shared observation, never an independent replication. Per-arm planning charges and allocated learning costs are kept.
8. **Code (stage P-A27; the frozen hashes are recorded at the freeze).**
   - One source for the numbers: sim/final_test/matrix/common.py (RUNTIME_PLAN, REPEAT_ARMS, EXEC_RUNS_REPEAT, CURVE_SNAPSHOTS, plan_only, repeat_exec_arm).
   - Users: plan.py (enumeration), capture.py (curve sources), execute.py (units, runs, plan-only skip), score.py (plan_diagnostics.json), sim/final_test_callplan.py (cells, projection, execution_enumeration).
   - analyze.py does not read plan_diagnostics.json yet. Its Codex table is empty because no X-* row reaches score.jsonl. That reader is owed before the freeze.
   - The gates are listed in the ledger entry of stage P-A27. Astra reviewed the implementation (docs/final-test/_astra/P-A27.{prompt.txt,txt}): AGREE WITH CHANGES; its 5 points are applied (sealed plan-diagnostics file, scope of the counts, the wording in items 3, 6 and 7, the selftest comment) except the analyze.py reader, which is owed.
9. **A27 gate wording (DRAFT - PENDING ARASH SIGN-OFF AT THE FREEZE; added by the main session 2026-10-13 from docs/final-test/FREEZE-READINESS.md table H; Astra review docs/final-test/_astra/A26-gates.txt points 5-7, applied).** These replacements activate with A27 at the freeze, independently of A26. The superseded wording stays as historical text. The ledger records the dated sign-off, this gate-text revision and the freeze identifiers. Written after development results and the freeze-readiness audit, before any acceptance or test planning, execution or scoring.
   - **Snapshot gate (12.6; replaces the three-order requirement).** Under A27 the snapshot gate checks hashes and nesting for order 0 at K = 1, 3, 5, 10. K = 0 uses the registered empty-memory baseline; the freeze specifies and hashes its representation. K = 10 is the same S* snapshot used by the main matrix. Orders 1 and 2 are not freeze inputs.
   - **A21 G final projection.** `final_projection.json` is regenerated from the A27 call plan (`python sim/final_test_callplan.py --final-projection`) at the freeze; the pre-A27 file is not a valid freeze input. Its ceilings are the A24 item 10 values (80 USD gate, 100 USD cap).
   - **A27 consistency.** Before the freeze, the call plan, the workload counts, the projection and the budget gates must agree on A27: REPEAT applies only to the core four with two executions, Codex is plan-only, and the learning curve uses order 0 only. The projection records the generating code and call-plan hashes and the pricing and runtime assumptions.

### A26 (2026-10-13, stage K-B2a; DRAFT - PENDING ARASH; from Arash's decisions of 2026-10-13 (ledger entries "ARASH DECISION: slicing needs a cutting tool in hand" and the K-A1 Knife decision); Astra review docs/final-test/kitchen_memory/astra/round6-knife-removal.txt, verdict ACCEPT WITH CHANGES, all applied below; written AFTER development results (rounds 1-5, the round-6 Gemma planning signal, K-B1 feasibility, P-vis audit) and BEFORE any acceptance call and any test call or test outcome; held-out-test-outcome-blind, NOT data-blind; placed after A27 only because A27 was appended first) -- Knife removed from task scope, the ButterKnife is the cutting tool, slicing needs it in hand

**Status.** This is a DRAFT. It authorizes nothing until Arash approves it in a dated ledger entry. It becomes active only as part of one coherent successor version (executor, perception filter, package, parser, validator, generated contract, scorer, dataset) that passes the gates of item 7 and is written into a new freeze (A25 item 11.6). Until then the round-5 / R6-D1D2 configuration stays the development configuration.

**0. Chronology and motivation (disclosure).** As of 2026-10-13 no acceptance or test item has been planned, executed or scored under any treatment version. Development failures and diagnostics were already known when this amendment was written, and they motivated it: in FloorPlan1 the only Knife sits clipped inside drawer geometry and renders 0 px from every honest pose (EXEC-DIAG; P-vis CONFIRMED_ZERO, 2 x 1200 valid frames), so every arm, Opus included, failed the Knife cells in rounds 1-5. That is evidence about one FloorPlan1 instance, not evidence that Knives are unusable in every kitchen. Task materials (instructions, reference plans, goals) of test and test_uncapped items WERE inspected to write the proposals of item 4; "no test execution" does not mean the materials are unseen.

**1. Knife out of task scope (all kitchens, all arms; extends R6-D1D2 / D2).** The Knife type is treated as not existing for the agent: it is removed from the planner type vocabulary, the package, the uses table, the homeowner inventory, the briefings, and the detections and object memory exposed to the agent (one global filter, `sim/type_scope.py` EXCLUDED_TYPES = ("Knife",)). The same filter applies to every arm. The detectors are NOT retrained: Knife boxes are dropped at the exposure boundary, never remapped to another type. Documented exception: the specialist's Knife class stays INSIDE the identity check (A25 item 6) as an internal verification class; a specialist "Knife" on a held object means the ButterKnife identity is not confirmed (a generic identity-check failure is returned to the agent; the pickup is not silently approved). The sealed step-3 holdout result (.924) is the result of the original, unfiltered detector evaluation; it does not measure the filtered pipeline.

**2. The ButterKnife is the cutting tool; slice needs it in hand (shared executor, all arms).** CUTTING_TOOLS = {ButterKnife}. The honest executor refuses `slice` unless the type the agent honestly holds (established by its own pickup or setup record and, for a specialist-class type, the identity check) is in CUTTING_TOOLS; cause `no_cutting_tool`, checked before any primitive, 0 primitives charged, not retried within the step; refusals count against the existing per-step and per-episode budgets (A25 item 7), so they cannot loop freely. Missing or uncertain tool identity never authorizes a slice, and no other execution path may slice. **This is an experimenter-imposed realism convention, not an AI2-THOR constraint** (ai2thor 5.0.0 slices with empty hands, a Knife, a ButterKnife or a Spoon: 88/88 cells, K-B1) and not a validation of real-world cutting. Purpose substitution (A25 item 2.3) stays OFF.

**3. Planner side.** (a) Package and procedures say "pick up the ButterKnife, then slice"; this line is a contract fact once item 2 is active and is checked by the contract gates against the executor source. (b) Parser: the bare words knife / چاقو / کارد stay unresolved ("?", A25 item 1.1); they never become ButterKnife. Explicit butter-knife terms resolve normally. (c) Validator: NO new repair code. A plan that slices without the ButterKnife in hand is accepted if otherwise valid and is refused by the shared executor; the submitted plan is what PC/TDS score. The executor accounts for the start holding and any put/drop between pickup and slice. (d) The prompt example 'never "pickup(Knife)"' (sim/planners/contract.py) and the slice note in validator_v2.py / exec_contract.json change only together with item 2, the contract regenerated by build_exec_contract, in the same successor version.

**4. Dataset (PROPOSALS in sim/dataset/knife_to_butterknife_proposals.json; Arash checks every item before practice).** One fixed, exhaustive transformation: every Knife in a reference plan or oracle setup becomes ButterKnife; an instruction that names the knife is reworded to the butter knife (EN "butter knife", FA کارد کره‌خوری, as in the existing ButterKnife items); goals (isSliced) are unchanged; ids are kept. Items whose GOAL names the Knife are archived, not re-meant: g-l1-pickup-knife and g-l2-drawer-knife (duplicates of existing ButterKnife items), g-l2-find-pickup-knife, g-l3-cutlery-sink, g-l3-tool-cut, and g-l3-set-table (stays excluded_inapplicable). No replacement item is introduced by this amendment; any replacement would be a NEW id with its own approved family/split assignment and a duplicate check against the reserved acceptance items. Affected: 9 test + 9 test_uncapped_not_scored + 9 dev nontest + 2 excluded = 29 items; 13 test-side items change their derived PC/TDS gold (reference plan); the 5 L1 slice items change only the oracle setup (gold unchanged); 5 items change text (g-l2-knife-bread-toaster and the 4 dev "Take the knife" items). general.json (956f0ae8...) and split_manifest.json (c1ec1b62...) are kept as archived originals; if approved, a versioned successor dataset and an effective inclusion manifest are written (archived ids removed, no reshuffle of unaffected assignments), and id integrity, split membership, family / duplicate checks, EN/FA consistency, denominators and gold generation are re-run.

**5. Comparability with rounds 1-5.** Rounds 1-5 are reported as observed, with their original denominators (original 24, fresh 22) and the old rules (Knife in scope, a slice without a tool counted as ok). The D2 original 20 (round 5: GP 17 / G0 16 / O0 19 / STRICT 15) is a retrospectively selected subset, reported next to the 24, never instead of it. Results under the new executor are a different protocol: they are not pooled with rounds 1-5, and differences are not attributed to planning. Historical traces are re-annotated, not re-credited: the 19 step-4 "slice successes that skip the knife" and every earlier empty-handed slice are listed as old-rule successes that the new rule would refuse; new-rule numbers come only from new matched runs. Development regression reports give old-rule and new-rule counts side by side; every loss caused only by the new rule is listed by presentation and labelled expected, and every other loss is unexplained and counts against the zero-loss gate. STRICT keeps its definition and gets the same executor semantics.

**6. Scorer (prereg s8).** `unrealistic_slice` stays an independently computed invariant from the recorded holding at each executed slice; it is not hard-coded to 0 nor inferred from goal success. Under item 2 it should be 0 for every executed slice; any non-zero value is an executor fault.

**7. Activation gates (before the new freeze).** Offline: contracts, exec_r1, act, rules, tidy_e1, package and procedural, goal_schema, planners, score, filter leakage (no Knife in any agent-facing text or exposed detection), setup / gold consistency, family checks. Live, FloorPlan1 only, reference plans, no LLM, after auditing the item list against the reserved acceptance, test and test_uncapped ids (none of those may run): the step-4 regression items and the dev slice items, each slice in three hand states (ButterKnife held -> ok; empty -> no_cutting_tool; other object -> no_cutting_tool), plus a specialist Knife-confusion case; 0 violations and zero oracle reads DURING the honest episode (setup is oracle world preparation, logged separately, with the same initial-state disclosure for every arm). All affected hashes are re-frozen and incompatible response caches invalidated. The exact edit list and order are in docs/final-test/kitchen_memory/K-B2b-TODO.md.

**8. Claims.** The final estimand concerns the amended task scope and the shared executor. Applying the change to every arm does not show that relative arm advantages are unchanged. Nothing about Knife handling in other kitchens is claimed. P-vis disclosure: the agent code was not frozen when the held-out kitchens were inspected; that inspection was for eligibility only and changed nothing in the agent. K-B1 used FloorPlan1 plus FloorPlan2/5/10 (none of them held out) as simulator testing aids only.

**9. A26 gate wording (DRAFT - PENDING ARASH; main session 2026-10-13, from docs/final-test/FREEZE-READINESS.md summary point 6 and table H; Astra review docs/final-test/_astra/A26-gates.txt, verdict CHANGE, all 7 points applied).** These replacements activate only with the approved, validated A26 successor freeze. The superseded wording stays in the file as clearly labelled historical text. If A26 is not approved, the old wording stands and the D2 filter must be off in the frozen runner; one freeze never mixes the two. A27's snapshot, workload and projection changes are recorded and activated under A27 (A27 item 9), independently of this item. The ledger records the dated approval, this gate-text revision and the successor-freeze identifiers. These revisions were informed by development results and the freeze-readiness audit; test task materials had been inspected, as disclosed in item 0. They were written before any acceptance or test planning, execution or scoring.
- **12.12 Type list (replaces A4's 12.12; in section 4's system-prompt row the type-list clause "the type list is the detector's class list (49 types; an ontology, not a room inventory)" is replaced by "the type list is the 12.12 list").** The prompt's type list is the ordered class list extracted from the frozen primary detector weights of the kitchen, minus the types in `sim/type_scope.EXCLUDED_TYPES` (item 1; now ("Knife",)), order kept. It is an ontology, not a room inventory, and it is never derived from reference plans. For FloorPlan1 the detector has 49 classes and the planner is shown 48. The gate passes only if all of these hold: (a) the ordered class list extracted from the frozen weights matches the registered count (FloorPlan1: 49) and hash, using the registered extraction and serialization method, and the weights hash also matches; (b) the shown list equals `type_scope.filter_types(classes)` exactly (FloorPlan1: 48), and the system text contains that list; (c) within each kitchen and frozen configuration, the rendered type list is byte-identical across arms; (d) no agent-facing text on the frozen inputs names an excluded type (system text, tool contract, package, FS block, briefing, the observation summaries of the captures; the user's instruction is verbatim, section 4, and is not filtered). Matching distinguishes Knife from ButterKnife and covers the excluded-type names and the aliases used in those inputs. This text check supplements the detection and object-memory filter gates of item 7. The documented exception of item 1 (the specialist's Knife class inside the identity check) is not agent-facing. freeze.json records the detector count, the shown count, the excluded-types tuple, the sha of type_scope.py, and the ordered source-list and shown-list hashes with their extraction and serialization method. The FloorPlan15 / FloorPlan17 freezes use the same rule with their own detectors; their counts are fixed in their registration, never assumed from Kitchen 1.
- **12.5 Evaluator (replaces the first clause of 12.5 for the successor version; items 2, 4 and 5).** The successor reference plans and setups are those approved under item 4; archived and excluded presentations are omitted. Before running this gate, each expected outcome change attributable solely to the new slicing prerequisite is registered, with its historical trace evidence, together with the previously specified allowance for known variance. Every other outcome difference fails the gate. Changed outcomes are established by new runs, not credited by re-annotation. At every executed slice of the 12.5 run, `unrealistic_slice` (item 6) is 0. "An empty plan scores 0 on every applicable item" is unchanged. The requirement that goal predicates are evaluated only in scoring() remains unchanged.
- **12.8 Family boundary audit (adds to 12.8; item 4 and the visibility addendum).** The effective inclusion manifest preserves the original split assignments and family memberships, except for explicitly approved regrouping, and applies the approved archival and presentation-specific exclusion records. A kitchen-specific exclusion does not remove an id from other kitchens. Arash re-checks every family affected by archival, exclusion, or a change to instructions, reference plans or setups, checks for newly created duplicates across split boundaries, and records the result in a dated ledger entry. `sim/final_test_split.py` is re-run, with a new manifest sha, only if a family is regrouped.
- **Section 8, known unrealistic successes.** The original "Known unrealistic successes" paragraph applies only to historical results under the old executor (rounds 1-5 and step 4; item 5). Under A26, `unrealistic_slice` independently flags each executed slice whose recorded holding does not establish an honestly confirmed ButterKnife in hand. Any such event is an executor fault: it fails pre-freeze validation; after the freeze it is disclosed and handled under the registered fault policy, without silently deleting episodes or changing SR denominators.

**10. The shared slice tool line (BK-LINE; DRAFT - APPROVED BY ARASH 2026-10-13, SIGN AT THE FREEZE; from Arash's decision of 2026-10-13, ledger entry "ARASH DECISIONS: butter-knife rule for every arm = YES; overnight GO; save every step", item (1); Astra review docs/final-test/_astra/BK-LINE.txt, verdict ACCEPT WITH CHANGES, both changes applied; written AFTER development results including the round-6b check, and BEFORE the final freeze and any acceptance or test call or outcome; held-out-test-outcome-blind, NOT data-blind).** Every arm's shared ft-v2 tool line for `slice` (G0, O0, X0 and the +M arms; same bytes, English only) states the item-2 executor contract. The meaning text is generated by `sim/planners/validator_v2.build_exec_contract` from `requires_tool` (= honest_skills.CUTTING_TOOLS while CUT_RULE_ON), never typed per type, and exec_contract.json / ft_v2.json are regenerated by `validator_v2.py --write`. Old line: `slice it. Hand: the hand is unchanged.` New line: `slice it; you must be holding a ButterKnife. Hand: the hand is unchanged.` (Astra: keep "slice"; no "pickup ButterKnife first", which can read as an unconditional pickup when the ButterKnife is already held; the empty-hand rule stays on the pickup line.) ft-v1 (step4_gemma / contract.TOOL_SPECS, the frozen step-4 text) is unchanged. The validation rules are unchanged (item 3(c): no new repair code; only the contract builder in validator_v2.py changed). The package's own slice line and the procedure "To slice: pickup ButterKnife, then slice the food." are unchanged; they are now redundant with the shared line, and the package rows are not rebuilt. Disclosure: (a) this changes the SHARED prompt after the round-6b development check; round 6b ran on the old line and its result stands as development evidence under the old prompt, it is not re-run and it is not a same-prompt comparison with the final test; the final test freezes the new line (new ft-v2 texts_sha256 and exec_contract sha, recorded in freeze.json). (b) Every arm now receives a fact that before was supplied only by the package (+M), so the package's incremental information on slice items, and possibly its measured effect there, is smaller than in rounds 1-6b; package-effect contrasts are interpreted on the new shared prompt only. (c) The round-6b freeze pins these files; after the recorded round-6b result its offline tripwires run in a CLOSED mode (sim/final_test/signal/r6b/r6b_closed.py): each post-run change of a pinned file is registered with its frozen sha, an archived copy of the frozen bytes and the approved new sha, and the frozen r6b gate keeps refusing a re-run on the new tree.

**A26 addendum: visibility-audit criterion and the Kitchen 1 Knife exclusions** (main session, 2026-10-13, from Arash's approval in words; ledger entry "ARASH DECISION: visibility-audit criterion and Kitchen 1 Knife exclusions APPROVED")
- **Criterion.** A presentation is excluded only if a type its goal needs is CONFIRMED_ZERO: 0 px from every honest robot pose, confirmed with the method of docs/final-test/kitchen_memory/VISIBILITY-AUDIT.md. MARGINAL, HARD_TO_SEE and DETECTOR_MISS are reported descriptively and never excluded.
- **Kitchen 1.** The Knife-goal items listed in VISIBILITY-AUDIT.md are excluded for every arm.
- **Kitchens 2 and 3.** No exclusion. The Kitchen 3 Mug (HARD_TO_SEE) and Egg (MARGINAL) are reported only.
- **When.** Applied before the final manifest freeze and before any arm outcome is seen.
- **Disclosure.** Kitchens 2 and 3 were inspected for object observability only, while the agent code was not yet frozen; nothing in the agent, the dataset or the packages was changed from that inspection.

### A28 (2026-10-13, stage k2k3-A28; DRAFT - APPROVED IN WORDS BY ARASH, NOT YET SIGNED; from Arash's approval-packet decision 13 (ledger entry "ARASH DECISIONS (approval packet ...)", 2026-10-13, items #1-#7) and decision 12, based on docs/kitchen-datasets-k2k3-plan.md (Astra review docs/final-test/_astra/k2k3-detectors.txt); Astra review of this text docs/final-test/_astra/A28-k2k3.txt; written AFTER FloorPlan1 development results and the P-vis eligibility audit, BEFORE any FloorPlan15 / FloorPlan17 probe, capture, training, practice or evaluation, and BEFORE any acceptance or test call or outcome in any kitchen) -- the FloorPlan15 / FloorPlan17 perception adaptation protocol (the perception part of the A24 item 12 registration)

**Status.** Arash approved the plan in words (decision 13: "#1 approve the plan as the draft FP15/17 adaptation protocol; write the registration next"). This text is the registration of the perception part. It authorizes nothing until Arash signs it in a dated ledger entry, and that entry must exist before the first FloorPlan15 or FloorPlan17 probe. It does not itself register the agent-side replication (briefings, packages, practice, task set, the executor's per-kitchen inputs, runner plumbing); that is a separate amendment. **Timing of that amendment [Astra A28 1; this tightens the plan, which deferred it until before practice].** Before either kitchen's first probe, the shared executor / E3 version, the task-selection and evaluation rules, and the briefing / package construction procedures for both kitchens are frozen and signed in that separate amendment. Later plumbing may implement those specifications but may not introduce choices informed by either kitchen's adaptation data or results. FloorPlan15 findings cannot change FloorPlan17's protocol. Registering the perception part (A28) and the agent part as two dated texts is a staged registration; it is declared here as an amendment to A24 item 12's timing, and both texts precede any detector adaptation. Code needed by this protocol is listed in docs/kitchen-datasets-k2k3-TODO.md; nothing in it is built yet.

**0. Chronology and disclosure.** Before this text: FloorPlan15 and FloorPlan17 were inspected once for eligibility (P-vis visibility audit: instance-mask pixel counts, types and item ids only, no frames saved; docs/final-test/kitchen_memory/VISIBILITY-AUDIT.md), and their object inventory is in sim/dataset/scene_inventory.json (a live scan of all 30 kitchens). Nothing in the agent, the dataset or the packages was changed from that inspection. The agent code is not frozen yet. No frame of either kitchen has been captured, labelled or trained on.

**1. What is allowed, and the wording.**
- Labelled capture and detector / state-estimator training in FloorPlan15 and FloorPlan17 are **perception adaptation**, named and allowed by A24 item 12. Labels come from simulator ground truth (instance masks, metadata), as in Kitchen 1; it is dataset construction.
- Capture data, frames, logs, `probe.json`, audits and perception reports **never reach the agent's inputs, memory, briefings, packages, prompts, task selection, executor rules or evaluation design**. The agent loads only the frozen artefacts of item 9 (weights, state calibration, the look-alike map). A read-only grep check of the agent loaders (no path under `~/thor-data/`, no perception report) is a gate before the first agent episode in that kitchen.
- Applying a registered rule to a kitchen's data is allowed; inventing or adjusting a rule after seeing that kitchen's data or results is not.
- **Human-information barrier [Astra A28 2].** Before Arash views either kitchen's capture images or hand-review items, both kitchens' task sets and briefing content are frozen (hashed in a dated ledger entry), or their preparation and approval are assigned to someone unexposed to adaptation data under a frozen construction rule. Every image / crop / context-thumbnail inspection is logged (who, kitchen, episode, frame, reason, finding). The read-only loader grep (above) cannot detect information carried by a person; it is a supplementary check only.
- Thesis wording: "FloorPlan15 and FloorPlan17 were held out from agent development and tuning, except for the preregistered perception adaptations, including the executor's look-alike map; they received kitchen-specific perception adaptation under a preregistered protocol. Conclusions extend to these two kitchens." The P-vis eligibility audit and its scope are disclosed. Never claimed: zero-shot transfer, completely unseen kitchens, registration before any inspection.

**2. Order and timing (decision 13 #7).**
- Kitchen order fixed: **FloorPlan15, then FloorPlan17**.
- Nothing in this protocol runs before the FloorPlan1 final test is complete (its last registered test execution and scoring done). It never runs at the same time as any other GPU job: one simulator or one training job at a time on the 8 GB card, the exclusive lock `/home/arash/sim.lock`, no console, no Ollama model and no other project's GPU process (checked with `nvidia-smi` and the process list before every session).
- Blocking before the first probe: Arash's dated sign-off of this text; the plumbing of docs/kitchen-datasets-k2k3-TODO.md (part A) built with its Kitchen 1 regression proof (byte-identical Kitchen 1 seeds, paths and `state_calib.json`; the step-3 selftests green); the hashes of item 3 recorded.

**3. Frozen inputs (recorded with sha256 in the ledger before the first probe).** Base checkpoints `yolo26s.pt`, `yolo26m.pt`, `yolo11s-cls.pt`; ultralytics 8.4.83; the torch / CUDA versions; every script used by items 4-9 (capture, balance, training, evaluation, layout check, map build, state review); the plumbing commit.

**4. Capture (reused from Kitchen 1 unchanged).**
- Main capture: protocol policy v3-2 / quota v3-1 (`kd.CAPTURE`): 168 episodes (128 train / 20 val / 20 test), 110 frames, brightness-only lighting on 20 %, no material randomisation, held-view rules, render guard, restart-on-glitch, fail-fast; container-aware capture (`open_parents`). Top-ups (`kd.topup_train_plan`, cap 24 train episodes) and the close protocol (`kd.close_plan`, 12 val_close + 12 test_close). Ambiguity capture: protocol amb-v1 + amendment 1 (`kd.CAPTURE_AMB`, `AMB_VIEWS`; 70 episodes: train 40 / val 10 / test 10 / holdout 10).
- Seeds `stable_seed(scene, protocol, split, index)` with the distinct protocol strings of each pipeline (main, `topup-train`, close, `amb-v1`, `amb-v1-holdout`), so each kitchen gets its own layouts. Caps are fixed; a shortfall is logged and reported, never filled by extra episodes; no floor is lowered.
- **Class list** = the probe registry (sorted base types present in the scene, `vocab.base_type`), fixed at probe time, before capture, and recorded with its hash in a dated ledger entry. Expected from the scene inventory: FloorPlan15 43 classes, FloorPlan17 45; the probe value is authoritative and is never assumed from Kitchen 1. The detectors are trained WITH Knife (A26 item 1). Shown to the planner: the class list minus `type_scope.EXCLUDED_TYPES` (A26 item 9, 12.12); the shown count is recorded at the same time.
- **Unattended capture (decision 13 #3).** Probe, main capture, top-ups / close and ambiguity capture are simulator jobs. Each may run unattended only with Arash's authorization **for that run**, given in a dated ledger entry naming the kitchen and the stage, before it starts. The hang guard, render guard, fail-fast, the WSL-reset driver and the 20-minute status monitor stay on.
- **Human checks during capture.** Logs, audit numbers and label checks are allowed. Frames are sampled only by the fixed rule: frame 0 and frame 55 of every 8th committed episode, TRAIN split only, each entry written to an inspection log (episode, frame, reason, finding). VAL / TEST / holdout frames are never viewed by a person (manifest counts and hashes only); the one exception is automated metadata access for the registered integrity checks (the layout check of item 5, counts and hashes), which never renders, displays or builds lists from those records. Incidental layout knowledge must not feed into briefings, packages, prompts, tasks or code; the barrier of item 1 applies, and where practical the session that views frames does not edit agent code.

**5. Data gates and repairs.**
- The Kitchen 1 audit and detection gate (`check_capture`, `stage_audit`, `stage_yolo`: `gate_type_target` 500/60/60, `gate_instance_target` 20/3/3) apply unchanged.
- Data-quality gate failure (separate from performance): the registered response is the existing top-up within its cap (24 train episodes, train split only). If the gate still fails, training proceeds with `--allow-failed-audit` only when the failure is a train-label shortfall, and the shortfall is reported. A structural FAIL (missing files, repicks, corruption) stops that kitchen for a documented repair.
- Repairs are infrastructure only: render-guard redos, restart-on-glitch, hang-guard redo, resume from checkpoint, an episode recorded failed after 6 redos. A code bug that corrupts data: the affected episodes are marked invalid (records kept), the bug is fixed, the episodes are re-captured from the same seeds under one uniform rule, and it is disclosed. **[Astra A28 7]** Infrastructure corrections require independently documented failure evidence, an enumerated affected set and preserved original records. Performance alone never justifies a correction. Corrections made after any score was inspected are reported separately as post-inspection corrections and cannot restore confirmatory status to an opened holdout; an opened holdout is never called unseen again.
- **Cross-pipeline layout check** before training: a layout fingerprint per episode (sorted object `name` + position after the spawn rounded to 0.01 m, from the stored frame records); no val / test / holdout layout may equal any train layout across all pipelines. **A persistent layout collision or a non-TRAIN quota failure stops that kitchen; no discretionary reseeding, split reassignment or additional capture is permitted** (same-seed recapture cannot resolve a deterministic collision). The stop is reported. Disclosed by design: 25 % of main-capture TRAIN episodes keep the kitchen's default arrangement (`p_spawn` 0.75), and the agent's evaluation episodes start from that default arrangement; val / test / holdout always shuffle. This is stated next to every agent result in that kitchen.

**6. Models (Kitchen 1 recipes, unchanged).**
- Primary: YOLO26s, MuSGD, 60 epochs, patience 20, 640 px, batch 16, seed 0 (`train_kitchen.RECIPE`, `MODELS["yolo26s"]`). A 1-epoch smoke into a scratch root first (weights never written to `sim/weights`).
- Specialist: YOLO26m, MuSGD, 60 epochs, patience 20, 640 px, batch 8 (nbs 64), seed 0, the full amb train set only, selected on amb VAL mAP50-95. Specialist targets **frozen, not re-selected**: `kd.AMB_TARGETS` (Fork, Spoon, ButterKnife, Knife, Spatula, SaltShaker, PepperShaker, Plate, Pot, Pan) and `kd.AMB_PAIRS`; disclosed that Kitchen 1 found them from its own VAL errors. The VAL rule `select_ambiguity_types` runs only after every choice is frozen, is reported descriptively, and never changes targets, groups or executor behaviour.
- **Control model: in, in both kitchens (decision 13 #2).** YOLO26s fine-tuned on primary-train + amb-train, 20 epochs, lr0 .002, pooled-VAL selection. It is never dropped in one kitchen only.
- Deployment rule: gated strict replacement while the robot holds an object up (`perception.TwoStageDetector`), as in Kitchen 1.
- **Selection rule, every model [Astra A28 8].** The deployed weights are the `best.pt` that the frozen script copies: ultralytics 8.4.83's default fitness on that model's own VAL list (primary: primary VAL; specialist: amb VAL, the plan's "selection on amb VAL mAP50-95"; control: the pooled VAL of `train_amb.py control`; each state model: its axis VAL, top-1 classification fitness), early stopping by the patience above, ties as ultralytics resolves them (the first epoch reaching the best fitness is kept). These rules are fixed jointly for both kitchens before the first probe.
- Reduced diagnostics (declared): no YOLO11s baseline and no 10 / 20-episode specialist learning curve in these kitchens; no per-kitchen claim about model family or data sufficiency is made.
- No Kitchen 1 weights, frames or live poses are used in another kitchen.

**7. State estimator (decision 13 #4: new hand labels per kitchen; this replaces the plan's "reuse the Kitchen 1 size floor").**
- Same code and recipe as Kitchen 1: 7 per-axis YOLO11s-cls models, 128 px, 30 epochs, patience 10, batch 128, seed 0. Crops from TRAIN / VAL episodes only, written to a per-kitchen path (never the shared Kitchen 1 `state_cls`).
- **Size floor per kitchen from Arash's own new blind hand labels in that kitchen.** The Kitchen 1 tool and rule: `sim/state_review.py make` (n = 300, seed 0, stratified over axis x true value x object size x distance, TRAIN episodes of that kitchen only; the page never contains the simulator's answer; "can't tell" is an answer; visible = answered and right), then `score`; the floor per axis is the Kitchen 1 rule of `state_eval.size_floors` (items sorted by box area, the smallest area at which a rolling window of 8 reaches >= 50 % visible; if never, the largest area). The labels are made after capture and before the VAL threshold fit. **[Astra A28 3]** The unchanged tool shows a padded crop with the object outlined **and a small full-frame context thumbnail** of the same TRAIN frame; this exposure is registered and falls under the barrier and inspection log of item 1. "Blind" means blind to the simulator's answers, not to kitchen identity or layout. Arash's `answers.json` is finalized and hashed before `score.json`, the ground-truth answers or the fitted floors are revealed; no score-informed relabelling and no replacement sample. An axis with fewer than 8 labelled items (one rolling window) gets no floor: that axis always abstains in that kitchen and is reported as zero coverage (the floor code's fallback of 0 is not used). The rolling-window floor is a registered operating rule, not a validated visibility guarantee. The missing decision-13 #4 note supplies no additional authorization; Arash's sign-off explicitly adopts the 300-item, seed-0, TRAIN-only procedure described here. The Kitchen 1 floor is not used in these kitchens.
- Thresholds per kitchen on that kitchen's **VAL** only, by the Kitchen 1 rule (per axis and predicted value, the lowest confidence threshold whose accepted VAL answers reach >= 95 % precision with >= 20 accepted; `TARGET_PREC` .95, `MIN_ACCEPT` 20). Edge case: fewer than 20 accepted VAL answers for a value, or .95 precision not reachable at any threshold -> that value always abstains ("unknown"), reported as zero coverage, never relaxed.
- Precision and abstention coverage are reported on untouched TEST episodes. Wording: "only preregistered perception calibration is permitted; Kitchen 1 agent constants remain frozen."

**8. Per-kitchen look-alike map (decision 13 #6, with decision 12).**
- E3 (DESSTA look-alike re-check: pick up a possible look-alike and re-check it with the specialist) uses, in each kitchen, a **registered per-kitchen Ambiguous Objects Map M** built from that kitchen's own primary detector. This is an agent-facing perception adaptation (it changes which boxes the executor picks up and re-checks) and is disclosed as such.
- Rule (the one frozen for Kitchen 1 in `sim/step4_lookalike_map.json`, applied unchanged, including its two restrictions) **[Astra A28 5]**: run the frozen primary on that kitchen's **VAL** split only (TEST never read) with the Kitchen 1 error-breakdown method (`sim/kitchen_v3/error_breakdown.py`, operating confidence .25). For each ground-truth box of class X: correct if a prediction of class X overlaps it at IoU >= .5; otherwise a classification error if any prediction of another class overlaps it at IoU >= .5, and the confused label Y is that of the greatest-overlap such prediction. Per X only the **three largest** confusion counts are kept (sorted by count, descending; ties keep the order of first occurrence, Python's stable sort). M[X][Y] = count / n_X; Y enters M[X] iff that rate >= theta = 0.02, tested before rounding (rates stored rounded to 4 decimals); n_X = 0 gives no entries. X ranges over the frozen specialist targets minus excluded types (Knife), i.e. the nine types Fork, Spoon, ButterKnife, Spatula, SaltShaker, PepperShaker, Plate, Pot, Pan. Y need not be one of the nine: any non-surface, non-excluded label may enter; Knife never enters as X or Y. The provenance file keeps counts and denominators for **all nine** targets, empty rows included (the Kitchen 1 writer kept counts only for targets with entries; this is a reporting addition, not a rule change). The same E3 constants as Kitchen 1.
- **Scope of the map [Astra A28 4].** It is a fixed, VAL-derived, disclosed agent-facing adaptation. It implements decision 12 as a bounded inspection mechanism for the nine specialist targets; it does not establish disambiguation of every object in the kitchen. VAL is reused for checkpoint selection and for this perception-policy fit, so VAL error statistics and the map are not independent performance evidence; generalization is assessed only on frozen, untouched evaluation data.
- The map is built after the primary is frozen and before any agent episode in that kitchen, written with its provenance (source file and sha256, weights sha256, split, operating confidence, theta, exclusions, raw counts and denominators), and frozen with the detector hashes (item 9). An empty map is a valid, reported result; a missing or malformed map file is a configuration error, never a silent empty map. The map is never rebuilt after an agent outcome in that kitchen is seen.
- The Kitchen 1 map is never used in FloorPlan15 / FloorPlan17. Which E3 executor version runs there is fixed by the agent-side amendment; this item fixes only the map and its rule.

**9. What VAL decides, the freeze, and evaluation.**
- VAL decides only: the checkpoint (the selection rule of item 6), the state thresholds (item 7) and the look-alike map (item 8). No model family, recipe, class list, target or E3 constant changes.
- **Freeze before evaluation.** Before TEST, CLOSE-TEST, LIVE, the fused amb TEST, the state TEST or the holdout is opened: sha256 of every deployed `.pt` (primary, specialist, control, 7 state models), the state calibration, the size-floor label files, the fusion settings, the look-alike map, the split manifests / list files, the data YAMLs and the relevant configuration and dependency code (not just the entry scripts) **[Astra A28 8]**, written to `sim/kitchen_v3/<scene>/frozen_weights.sha256` and a dated ledger entry.
- Evaluation, once, after the freeze: primary TEST, CLOSE-TEST, LIVE (150 live poses by the Kitchen 1 rule `train_kitchen.frozen_poses`: seed 20260929 over that kitchen's reachable positions, excluding any pose within 0.3 m and 20 deg of a recorded TRAIN camera pose; written to `live_eval_poses.json` and hashed before the first primary evaluation; a smoke run never writes to the real root), fused amb TEST, state TEST. Then the **sealed holdout** once (`train_amb.py holdout`): primary endpoint held-target naming success (IoU >= .5, conf >= .25, a miss is a failure), with held non-target false positives, all-class / target AP and a paired episode bootstrap (primary vs gated vs control). The holdout episodes (protocol `amb-v1-holdout`, index 1-10) are never built into any list before the freeze.
- **No outcome-driven retraining.** A poor TEST, LIVE or holdout score triggers nothing: no extra epochs, data, model, threshold or map change. The `kd.ACCEPTANCE` targets (macro AP50 >= .90, AP50-95 >= .70, state P/R >= .95) are reported against, not gates.
- Smoke and LIVE checks run the detector only (no planner, no executor, no agent task), so they expose no agent-task outcome. **No agent outcome in a kitchen is seen before its detectors, calibration and map are frozen.**

**10. GPU, heat and attendance (decision 13 #5: the proposed values as they are).**
- Training (primary, specialist, control, state) runs only while Arash is at the laptop and stops before he leaves: a job-specific stop (for example `pkill -f "[s]im/train_kitchen.py train --scene FloorPlan15"`), then a check that `_ckpt/<run>/last.pt` loads and records an epoch / training state at least as far as the last logged epoch (modification time alone is not enough) **[Astra A28 6]**. Resume = the same command. State models (`train_state.py`) have no resume: a stopped axis restarts from scratch with the same seed and recipe, the partial run is discarded and logged (no code change). Never train and open a Controller in one process. Launch WSL from PowerShell.
- Thermal watch every session: `nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv -l 60` to a log file. **Pause values (decision 13 #5 registers the proposed values as they are; this replaces the plan's prerequisite of a Kitchen 1 validation run; they are operational rules, not validated hardware protection): GPU temperature >= 87 C in 5 consecutive 60-s samples, or an SM clock below 1,000 MHz in 5 consecutive samples with utilization.gpu >= 90 %; no telemetry for 5 minutes also pauses.** A pause is a job-specific stop with the checkpoint check above; the job resumes from its checkpoint once the resume condition holds. **Resume condition: [OPEN -- a numeric cooldown value was not proposed; Arash fixes it before his sign-off].** A heat pause is an infrastructure event: logged, never an outcome, and it triggers no change of recipe.
- Laptop plugged in, Windows "best performance" power mode, hard surface, free vents. Batch sizes as Kitchen 1 (16 primary, 8 specialist, 128 state).

**11. Budget.** Estimate (not a cap; A24 item 12's "about 50 GPU-hours together" is replaced by this estimate): **about 51-59 GPU hours for both kitchens with the control model**, before unexpected redos (per kitchen 25.3-29.3 h: training components about 13.9-15.9 h, simulator / evaluation components about 11.4-13.4 h (Astra A28's corrected sums of plan section 8), plus the hand-label session, which uses no GPU). Elapsed time is longer (WSL resets, attended-session gaps). Order of resource-only cuts, decided now: none. This replaces the plan's possible control cut: the control model is kept in both kitchens (decision 13 #2); if the budget cannot be met, the delay is reported, nothing is cut, and no step is shortened. **[Astra A28 9]** Resource interruption cannot select which kitchen or which outcomes are reported. If completion becomes impossible, every completed and incomplete component and all observed results are reported, and the incomplete study is not presented as a completed two-kitchen replication. Actual totals are reported.

**12. What is reported per kitchen.** Data: episodes planned / kept / failed, frames per split, labels per class, shortfalls, redos, aborts, disk, the layout check, the inspection log, absent distractor types. Models: epochs run, best epoch, hours, VAL / TEST / CLOSE / LIVE mAP50 and mAP50-95, per-class AP, the descriptive VAL ambiguity selection. Specialist: held-target naming success (primary vs gated vs control) with CI on the sealed holdout, held non-target FP. State: the size floors and the hand-label counts, precision and coverage per axis value on TEST. Look-alike map: the full map with counts. Budget: training, simulator and elapsed hours, every heat pause. All hashes. Every number is labelled "FloorPlan15 (adapted detector)" or "FloorPlan17 (adapted detector)" and never pooled with Kitchen 1.

**13. Claims.** Perception results describe these two kitchens with their own adapted detectors; they are not evidence about unadapted transfer. A specialist-vs-control claim is made per kitchen only from that kitchen's sealed holdout, and it compares the registered systems: model capacity and training schedules differ, so the difference is not attributed to specialization alone. Nothing about the agent in FloorPlan15 / FloorPlan17 is registered here.

**14. Astra review (docs/final-test/_astra/A28-k2k3.{prompt.txt a16fe56a..., txt f28f276e..., raw.txt 795377bc...}).** Verdict ACCEPT WITH CHANGES, 9 required changes, all applied above and marked [Astra A28 n]; the 3 optional suggestions applied (thesis wording, specialist-vs-control wording, budget sums). Code facts behind changes 3, 5 and 6 were checked in the source (`state_review.py` context thumbnail; `error_breakdown.py` top-3 truncation and greatest-overlap rule; `train_state.py` has no resume). Items Arash must still fix before signing: the heat-pause resume condition (item 10), and his choice under item 1's barrier (freeze both kitchens' task sets and briefings before he views any image, or assign them to an unexposed person). The 0.01 m fingerprint rounding (item 5) and the "fewer than 8 items -> axis abstains" rule (item 7) are the main session's proposals made to meet Astra's request for exact rules; Arash confirms or changes them at sign-off.

**15. Dated timing change by Arash (2026-10-14 00:25; ledger entry "ARASH DECISIONS: (a) he sees the freeze summary first; Kitchens 2/3 data capture 07:30-10:30"; recorded by stage K23-PREP; DRAFT -- revised after Astra's review docs/final-test/_astra/K23-PREP.txt (verdict REJECT, 10 required changes, all applied and marked [Astra K23 n]); it authorizes nothing until Arash signs the record of 15.6).** Arash's words: "in the time between 7:30 and 10:30 maybe you can start doing some very careful (checking each step along the way to make sure the class balance and the quality of pics are good) data collecting for kitchen 2 and 3."

15.1 **What changes.** Item 2's timing ("Nothing in this protocol runs before the FloorPlan1 final test is complete") is amended for the perception data CAPTURE of item 4 only: in the window 2026-10-14 07:30-10:30, BEFORE the Kitchen 1 locked test, FloorPlan15 first (item 2 order). In this release only three pipelines exist: the probe, a thermal pause test that writes no data, and the main capture (`kd.episode_plan`, 168 episodes). Top-ups / close and the ambiguity capture are NOT part of this change; each needs its own verified tooling and authorization later [Astra K23 10]. Training, smoke epochs, list building, evaluation, hand labels, the map and any perception freeze keep item 2's timing and item 10's attendance rule. Nothing else in A28 changes unless Arash's signed record (15.6) names the item and the amendment explicitly and prospectively; a general statement is not an exemption [Astra K23 1].

15.2 **Preconditions (any one missing = no capture).** (a) The Kitchen 1 freeze is complete and its freeze summary written, Arash has read it (his record says so), and no Kitchen 1 job uses the GPU, the simulator or /home/arash/sim.lock. (b) The K23 gates pass, including the Kitchen 1-unchanged gate (FloorPlan1 seeds byte-identical; the 47-file frozen treatment manifest; the Kitchen 1 weights and reports byte-identical; the Kitchen 1 stores unchanged by size + mtime for large files, sha256 for small files and file counts per directory -- this is evidence of no write, not a byte comparison of every image). (c) The capture runs alone on the GPU (item 2). (d) Each kitchen's class list is fixed by its probe and recorded with its sha256 in a dated ledger entry before that kitchen's first capture episode (item 4).

15.3 **How it runs.** The registered Kitchen 1 capture code and configuration (`kd.CAPTURE`, `kd.episode_plan`, `stable_seed`) called unchanged by a batch wrapper; batches of 8 episodes; no ad-hoc pose changes, no reseeding, no split reassignment, no extra episodes. Batching reuses the capture's registered resume (state saved per episode); it reuses the protocol, it does not claim pixel-identical frames versus one uninterrupted process (a simulator restart changes renderer history) [Astra K23 Q2]. The capture's own guards stay on (render guard, restart-on-glitch, per-episode hang watchdog, fail-fast + the 8-episode check_capture checkpoint); plus a whole-batch timeout bounded by an absolute dated deadline, and the item-10 thermal watch, which must be running and reporting complete telemetry before a batch starts and is supervised for the whole batch (guard missing or dead = stop) [Astra K23 6]. The WSL-reset driver of item 4 runs at the start of the window and after a render-guard burst or a hang only, each time after a preflight shows that no other job holds the sim lock or runs in WSL; a preflight failure means no reset and a stop [Astra K23 2].

15.4 **Checking each step.** After every batch -- and also after an abort or a timeout, before any retry -- an automated quality check runs on every newly committed episode: class balance; image quality (blur, exposure, corruption signatures, near-duplicates); mask-size distribution; labels re-derived from the stored instance masks; image / depth / segmentation / paired-render file pairing in both directions; the class-ID mapping in stored registry order and the probe's scene; exact duplicates by hash; failed episodes; check_capture with scene and root explicit; the layout check (15.5); Kitchen 1 unchanged. Thresholds come from the Kitchen 1 dataset (operational screening values, not validated guarantees). A FAIL or a quality WARN, a missing / unreadable / stale QC result, or a QC that did not cover every new episode STOPS the capture; the batch stays open until a person records a resolution after consulting Astra (Arash's rule). A stop never leads to a pose change, a reseed or a threshold change. At the end of the main capture the held-out type / instance floors of the detection gate are checked as counts; a shortfall stops the kitchen (item 5) [Astra K23 4, 7].

15.4a **Ruling R1: `pace` is a NOTE (Arash, 2026-10-14, decision 7 of docs/final-test/MORNING-DECISIONS.html, ledger entry "ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)"; Astra K23-PREP r3 ruling R1; registered BEFORE any FloorPlan15 / FloorPlan17 probe or capture).** The `pace` result of the frozen Kitchen 1 verifier check_capture (episodes keeping < 20 % of their views, and the longest such streak) is classified as a NOTE: its counts and its streak are written in every QC report and in the final report (15.7), and it does not stop the capture. Reason: it measures accepted / attempted throughput, not image quality (on accepted Kitchen 1 data it reported 14 such episodes, longest streak 6). The classification is narrow: it covers `pace` only. Every other check_capture WARN and every quality, coverage, no-progress, failed-episode, layout and deadline stop of 15.4, 15.5 and 15.7 stays a stop, and no threshold changes. Code status: the K23 QC still treats `pace` as a stop; the change is made and tested in the bounded K23 repair cycle of 15.8 (c), before any capture.

15.5 **Pixel access and the layout fingerprint (registered here before any probe).** (a) [Astra K23 8] The K23 QC decodes pixels of TRAIN frames only. Registered automated exception: the frozen Kitchen 1 verifier check_capture, run inside the capture every 8 episodes and by the QC, decodes small grey thumbnails of the last 8 committed episodes of every split, VAL / TEST included, for its near-duplicate test. It is automated integrity checking only: nothing is displayed, no person or model sees those pixels, nothing is selected, trained or tuned on them. Human viewing is limited to item 4's registered TRAIN sample and logged; in this window Arash views no capture image unless 15.6 (iii) allows it. (b) [Astra K23 9] The item-5 fingerprint is implemented as a REDUCED fingerprint: for each episode, the probe's settable instances (every probed instance except the knob-driven controlled types), each as name + objectId position after the spawn at 0.01 m, read before that object's state is set. Stateless objects are not recorded, so two layouts that differ only in stateless objects count as equal (a false collision is possible; a hidden one is not). Rules: every record must carry only expected names with well-formed positions; a held-out record collides with a train record when they agree on every shared name and share at least 75 % of the expected names; a pair sharing fewer names is undecidable; collisions, undecidable pairs, malformed or unexpected records, held-out duplicates and failed held-out spawns all stop the kitchen for investigation. Kitchen 1 (development data) check: 285 episodes over all five pipelines, every record complete (44 names), 0 collisions.

15.6 **What Arash signs before 07:30 (the authorization record) [Astra K23 1, Q1].** One dated ledger entry whose heading contains its own title, mirrored in `sim/kitchen_v3/k23/authorization.json` (verified by the supervisor: date, window, scene, pipelines, the sha256 of every K23 script, the two ledger headings, every item below closed): (i) his sign-off of A28 with this item 15, for FloorPlan15 probe + thermal test + main capture, 2026-10-14 07:30-10:30, with the script hashes; (ii) the thermal resume condition -- proposed: **any thermal pause ends this capture window, no restart** -- and which threshold applies: A28 item 10's 87 C (registered) or the Kitchen 1 night chain's 85 C (stricter); (iii) the item-1 human barrier -- proposed: **no person views any FloorPlan15 / FloorPlan17 capture image in this window** (all checks automated; the registered TRAIN sample is listed, not viewed), or freeze and hash both kitchens' task sets and briefings first, or delegate preparation and approval to an unexposed person; (iv) the A28 status-line requirement that the agent-side amendment be frozen and signed before either kitchen's first probe -- either it is signed, or he amends that deadline explicitly for this capture-only window (stating that no agent-side choice is made in the window and that, under (iii), no person sees capture data); (v) the explicit deferral, for this window, of the not-yet-built training / evaluation plumbing (TODO A1, A3, A5-A11, A14-A16) and of the item-3 frozen inputs that only training uses (base checkpoints, torch / CUDA versions), which item 2 otherwise requires before the first probe; the capture-path scripts are hashed in the record. If any of (i)-(v) is not signed, no capture runs.

15.7 **Stops and reporting.** 10:30 is a hard, checkpointed stop: no batch starts unless it can finish, be checked and be cleaned up before the deadline; a batch still running is ended before it and its partial episode is discarded and redone on resume (item 5 repair rule). Committed but unchecked episodes are reported as unchecked, never as passed. The final report states committed / failed / checked episodes against 168, every stop, redo, reset and pause, and that the capture is partial (FloorPlan17 is not started in this window). The barrier of item 1, the gates and repair rules of item 5, item 10 and item 12 are unchanged.

15.8 **Dated decision by Arash (2026-10-14, decisions 5 and 6 of docs/final-test/MORNING-DECISIONS.html, ledger entry "ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)"): the window of this item is not used; the capture moves to a later window.** His recorded decisions: (5) "move it to a new window after the Kitchen 1 freeze (dated amendment to A28 item 15 at that time). If round 4 says REJECT, one more bounded offline repair cycle before that window." (6) "decide together with the new window, on the then-current script hashes." Effect: (a) the 2026-10-14 07:30-10:30 window of 15.1, 15.6 and 15.7 is vacated; nothing was probed, tested or captured in it (no authorization record exists; K23-PREP Astra r4 = REJECT). (b) The capture moves to a new window that starts only after the Kitchen 1 freeze (15.2 (a) unchanged). That window, its dated deadline and the authorization record of 15.6 ((i)-(v), on the script hashes current then) are set together by a dated, prospective amendment to this item at that time; until that amendment exists, item 15 authorizes no probe, thermal test or capture, and item 2's timing stands. (c) Round 4 said REJECT, so one more bounded offline repair cycle (the r4 required changes and R1 of 15.4a in code) runs before that window. (d) Nothing else in this item changes now. This entry is the record of the decision, not the amendment that sets the window.

15.9 **Dated amendment by Arash (2026-10-14 ~14:25; ledger entry "2026-10-14 ~14:25 -- ARASH DECISION: start Kitchen 2 data capture now, in parallel, when the GPU is free"; written by stage K2-PARALLEL on 2026-10-14, BEFORE any FloorPlan15 / FloorPlan17 probe, thermal test or capture; prospective).** Arash's words: "can't you do a little more parallelization? maybe when the gpu is free, gather databases of kitchen 2 for now in parallel." The ledger records it as overriding decision 5's timing, as his dated authorization per run (A28 #3 / item 15.6), with five binding conditions, quoted from that entry: "1. First the bounded K23 repair cycle (decision 5) to an Astra ACCEPT of the capture plumbing (round-4 findings). No capture on rejected plumbing. 2. The simulator only when NO Kitchen 1 job needs it. The Kitchen 1 path (acceptance -> freeze -> test) always has priority: the capture yields at the next batch boundary as soon as a Kitchen 1 stage wants the lock, checkpointed, and resumes later. 3. No `wsl --shutdown` and no global kills while any other job exists. 4. Capture only, no training (training only while Arash is present). 5. QC after every batch; registered policies only; R1 pace = NOTE."
- (a) **Timing.** For the FloorPlan15 capture of 15.1 (probe, thermal test, main capture only; FloorPlan17 is not part of this amendment) the "after the Kitchen 1 freeze" timing of 15.8 (b) is replaced: the capture may run BEFORE the Kitchen 1 freeze, in parallel with Kitchen 1 stages that do not use the simulator. Precondition 15.2 (a) is replaced, for this capture only, by condition 2: before EVERY batch no Kitchen 1 stage needs the simulator, the GPU or /home/arash/sim.lock and no job waits for the lock; when one does, the capture stops at the batch boundary with its state saved (the registered per-episode resume of 15.3) and resumes only after that stage no longer needs the simulator. A batch never starts while the sim lock is held (the batch takes it with `flock -n`). 15.2 (b)-(d), 15.3-15.5 and 15.7 are unchanged; 15.4a (R1) applies as registered.
- (b) **Authorization record.** The record of 15.6 carries `precondition: parallel_k1_priority`, the exact heading of Arash's decision entry above (`parallel_decision_entry`) and `kitchen1_priority: true` in place of the freeze-summary heading and `freeze_summary_read_by_arash`; everything else of 15.6 is unchanged (date, window, scene, pipelines, the sha256 of every K23 script including the new ownership library `k23_own.sh`, every item (i)-(v) closed). The supervisor refuses any other basis.
- (c) **What this amendment does not do.** It does not sign A28 and does not itself choose items 15.6 (ii)-(iv); decision 9 requires A28 to be signed before any FloorPlan15 probe. Before the first probe Arash closes, in one dated ledger entry, (i) his sign-off of A28 with items 15, 15.4a and 15.9 for the FloorPlan15 probe, thermal test and main capture, the window (date, start, end = the hard deadline) and the script hashes; (ii) the thermal rule (proposed: any thermal pause ends the window, no restart; the implemented threshold is 87 C, A28 item 10 -- 85 C would need a guard change, a re-hash and a new review); (iii) the human barrier (proposed: no person views any FloorPlan15 capture image; all checks automated; the registered TRAIN sample is listed, not viewed); (iv) the agent-side amendment deadline (proposed: amended for this capture-only window, with the statements that no agent-side choice is made in it and that, under (iii), no person sees capture data); (v) the deferral of the not-yet-built training / evaluation plumbing and of the training-only frozen inputs -- condition 4 ("Capture only, no training") already covers the training part. Until that entry and the record mirroring it exist, no probe, thermal test or capture runs.
- (d) **Plumbing.** Condition 1 is met only by an Astra ACCEPT of the repaired capture plumbing (K23-PREP round 5 on the code of stage K2-PARALLEL: thermal closure persisted before any fallible validation and real evidence above every induced claim; per-episode quota-skip provenance and validated QC report contents and resolutions; cleanup of every process of the invocation on every exit; ownership proven per process by an inherited environment token, never by an archived pid; R1 in code). A `wsl --shutdown` (the person-invoked `-ResetWslNow`) is not used while any other job exists (condition 3); the emergency stop is job-specific.
- (e) Nothing else in A28 changes.

### A24.1-T (2026-10-13, stage A24.1-pins; DRAFT - PENDING ARASH (sign-off at the freeze; approval needed BEFORE fresh practice, item 0); the treatment-wiring part of A24.1 that the C4 runner integration left open (ledger "STAGE C4-integration -- DONE", OPEN item 2; Astra C4 review docs/final-test/_astra/C4-integration.txt, Q1, Q2, Q4); Astra review of this text docs/final-test/_astra/A24.1-pins.txt, verdict CHANGE, 8 required changes and 3 optional suggestions, all applied; written AFTER development results and BEFORE any practice-v2, acceptance or test call or outcome; held-out-test-outcome-blind, NOT data-blind) -- the planner-observation summary, the omitted-count line, P+M and /p under ft-v2, and the test-v1 flip

**0. Status, scope and deadline.** This is a DRAFT; it describes sim/final_test/matrix/treatment.py as it is now (sha256 f14a550a...). **These treatment choices require dated approval before fresh practice and remain fixed through acceptance, subject only to the registered correction cycle. They are recorded in the final freeze and included in Arash's final sign-off after the dress rehearsal.** The same deadline applies to the package-profile choice (D5: pkg-1.r4 / pkg-1.r6 / pkg-1.r6b) and to the other switch values (`mechanisms_off_all`, `type_scope`). It pins four things A24 item 2 and the C4 review left to A24.1: (1) the source of the planner-observation summary; (2) the exact omitted-count line; (3) what P+M and /p mean under ft-v2, with the executor search policy of every arm; (4) the switch of the test namespace from ft-v1 to ft-v2 and the treatment record in freeze.json. The other parts of A24.1 (prompt / tool contract / validator texts by hash, glossary, package quotas, the re-derived ceilings, A12) and the H1 part (A24.1-H1) are separate. Where this text and an older section disagree for an ft-v2 namespace, this text controls. Legacy matrix namespaces practice-v1 and rehearsal-v1 / v2 remain unchanged. Development rounds retain their recorded configurations and outcomes as history. The D2 type filter used below stays conditional on the approved A26 successor (A26 item 9); this addendum does not itself approve A26.

**1. The planner-observation summary: source (every arm, ft-v2 only).**
- Source: the honest agent's own complete object-memory list right after the fixed start sweep (A11), read from the executor's skills object as `om.summary(limit=None)`: one line per belief whose status is trusted, placed or held, ordered by type A-Z and newest step first inside a type, in the form `<Type> [<state>] <where>[ at (x, z)], step <n>` (the coordinates are omitted when the belief has none). Nothing is read from simulator metadata.
- Completeness check: capture refuses the row unless the first lines of that full list are byte-identical to the executor's own 40-line post-sweep summary (the same belief state), and unless the list is demonstrably complete (the executor's list may be shorter than the full list only when it was cut at 40).
- Rendering: (a) the D2 type filter (skipped when type_scope is "none"), applied BEFORE compaction, so an excluded type is neither shown nor counted; (b) **after D2 filtering, count distinct non-Floor types before truncation. With at most 40 types, call `compact(..., limit=40, agent_xz=None)`. Otherwise compact without an effective truncation limit, retain the first line for each of the first 39 types A-Z, and append the omitted-count line (item 2).** `planners/summary_v2.compact` drops Floor lines; turns several sightings of one fixture type into one group line with counts and states; gives one line per type before any type gets a second; prints A-Z by type. With `agent_xz` None a group line says "latest at", never "nearest at". The result is the prompt summary.
- The same prompt summary is given to every arm of a presentation (one capture row per presentation; identical bytes). Prompt compaction does not change executor memory. The chooser receives `om.summary()` with its default 40-line limit.
- Two hashes are kept per capture row: the executor's 40-line summary sha stays the integrity and execution-unit-key value (unit-key bytes unchanged); the prompt-summary sha is recorded next to it (`prompt_summary_sha`), together with `observed_types` (the sorted unique type names parsed from the prompt summary; states and positions are not stored in this field) and the full-list length. The package query separately parses types and states (never positions) from the same prompt summary.
- Development vs test: the development rounds 1-5 / r6 / r6b ran the planner BEFORE the start sweep, so their prompts had no observation summary at all (the prompt shows "(nothing yet)") in every arm. Acceptance, practice-v2 and the test run it AFTER the sweep with the summary above (A11, A25 item 12). Development numbers therefore do not transfer to the test configuration.

**2. The omitted-count line (exact text).** When more than 40 distinct types remain after the D2 filter (Floor and excluded types not counted), the first 39 types in A-Z order keep exactly one line each (the first line `compact` gives that type: its group line for a fixture seen at least twice, otherwise its newest sighting) and line 40 is exactly

`(+N more object types seen; not listed)`

where N is the number of remaining types not listed (a plain integer, English in every presentation, EN and FA alike). With 40 or fewer types the line is absent. The list never exceeds 40 lines. The line names no type and does not enter `observed_types`. Omitted types contribute no current-observation entries through this summary or its parsed retrieval inputs. They may still be named in the ontology, instruction, or independently eligible package records; package retrieval does not receive the omitted belief lines.

**3. Executor search policy and the meanings of P+M and /p (ft-v2 only; supersedes, for ft-v2, the search-memory allocations of the section 3.2 table: the "long-term memory" and "executor search chooser" columns of every row, including FS's "S\* for search only", and the rows P0 / P+M and G+M/p / O+M/p).**
- **No ft-v2 arm uses `LtmChooser`.** In matrix evaluation (acceptance, rehearsal, test) every arm executes with the prior chooser (`honest_skills.prior_chooser`) and no long-term memory store is opened during execution; practice memory reaches the +M arms only through the package text in the prompt (A24 item 1). Practice-v2 itself still runs the prior chooser + hook B while its learning store consolidates from the honest event stream.
- **Briefing-guided search (hook B, A25 item 2.1)** is ON for: the +M main arms (G+M, O+M, X+M), the learning-curve arms G+M@o0kK and O+M@o0kK (K = 1, 3, 5, 10), and P+M. It is OFF for G0, O0, X0, G+FS, O+FS, P0 and the /p replays. Hook B uses only the approved Kitchen 1 briefing (A25 item 2.1; D10). Its hint table (searched type -> the storage container types the briefing names, D2-filtered on both sides) is hashed, and a hook-on execution unit carries the extra key field `exec_policy = "briefing:<first 16 hex of the hint-table sha>"`, so a hook-on unit never shares an execution with a hook-off one. G-iso fails when a selected audit unit carries a policy or produces briefing-log entries. Policy-bearing units are filtered out before selection, so this is not a global check of every 0-arm unit.
- **P0** = the item's reference plan, prior chooser, no hook (unchanged diagnostic, no LLM).
- **P+M** = the item's reference plan, prior chooser, **hook B on**. It no longer means "reference plan + S\* `LtmChooser`". P+M - P0 is the effect of briefing-guided search when the plan is the reference plan; it is a diagnostic, never a planner result, and it is not used to build memory.
- **G+M/p, O+M/p** = the cached G+M / O+M plans (planned WITH the package in the prompt), replayed with the prior chooser and **hook B off**, no new planner call. Since every ft-v2 arm already uses the prior chooser, "/p" now removes hook B, not `LtmChooser`. +M - +M/p estimates the effect of enabling hook B conditional on package-generated plans; +M/p - +M estimates its removal. +M/p - 0 compares those plans without hook B with the 0 plans. **These exploratory conditional search-policy diagnostics qualify A24 item 1 and A25 item 13's blanket no-component-identification wording. They do not decompose the main package effect.** They are descriptive and never isolate memory.
- Scoring: when reconstructing a missing unit mapping, `score.derive_unit` uses the treatment-aware chooser and derives the policy from the arm and `hint_table_sha` loaded from `units.json`; otherwise scoring uses the recorded cell-to-unit mapping.

**4. The test-v1 switch and the freeze.json treatment record.**
- While `treatment.TEST_V1_TREATMENT = "ft-v1"`, test-v1 runs ft-v1, and any test entry that finds an ft-v2 treatment record in freeze.json is refused.
- The flip: (i) the one constant `TEST_V1_TREATMENT` is set to `"ft-v2"` (a one-line change in treatment.py; nothing else in the switch `FT2` changes in that edit); (ii) the treatment identity is computed in a fresh process AFTER that edit (treatment.py is one of the hashed source files and the identity also records `test_v1_treatment`; there is no circular dependency); (iii) the identity is written to freeze.json under the key `"treatment"`; (iv) the gates and the dress rehearsal run against that freeze.json.
- **The flip changes every ft-v2 treatment identity** (practice-v2 and rehearsal-v3 included), through both the treatment.py hash and `test_v1_treatment`. Earlier artifacts retain their original identities and must not be relabelled or resumed under the new identity. **OPEN for Arash (choose before practice):** (A) flip at the freeze, as Astra's C4 Q1 (b) proposed; then, before practice, the freeze workflow must specify how the provenance and eligibility of the pre-flip practice-v2 snapshots and acceptance artifacts for the final freeze are verified across this transition (today the snapshot manifests record their identity but capture does not re-check it); or (B) flip once, before fresh practice, so practice, acceptance and test share one identity; a test entry stays refused until freeze.json carries the matching `"treatment"` record, so no test call becomes possible earlier. The main session proposes (B), the simpler of the two; Astra named it as the alternative.
- **Arash's choice, recorded 2026-10-14 (decision 9, ledger entry "ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)"): (B), confirmed as carried out.** The flip was made once, on 2026-10-13, in stage TREAT-FREEZE step 3 (ledger), on the order of his overnight go of that day (ledger "ARASH DECISIONS: butter-knife rule for every arm = YES; overnight GO; save every step"), BEFORE fresh practice-v2. Practice-v2 then ran under the flipped identity d5831abc...; the later A31 re-pin to 7a2ee211... is bound to that practice by A31's replayed equivalence certificate, not by option (A)'s cross-flip check, which is therefore not built. Disclosure: item 5 asked for a dated approval of items 0-4 before fresh practice; the dated go that preceded practice was the overnight go above, and this explicit confirmation of item 4 was given after practice-v2.
- The `"treatment"` record is the output of `treatment.identity("test-v1")` with mask `"universal"`: `version` ("ft-v2"), `switch_sha` (canonical sha of `FT2`), `profile` (the one pinned package profile), `mask` ("universal"), `params_sha` (the effective package params incl. the D2 setting and `mechanisms_off_all`), `files` (the sha256 of each of the 47 identity files in `treatment.IDENTITY_FILES`, including code, contracts, configuration and model weights; **wording corrected 2026-10-14 from "20" (Arash, decision 9, ledger entry "ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)")**: the list held 20 files when this draft was written and grew to 47 when the executor, the perception weights, the providers and the contract files were added (Astra TREAT-FREEZE 6); the A31 re-pinned identity 7a2ee211... covers these 47 files, and `FT2["pins"]` is covered through `switch_sha`), `briefing` (`records_sha`, `n_records`, `allow_unapproved` = false, `hint_table_sha`), `test_v1_treatment` ("ft-v2") and `sha` (canonical sha of all of the above).
- After the flip, the capture, planning and execution entry / resume checks compare the computed treatment identity with `freeze.json` and validate the rows supplied to their treatment checks (no resume across a treatment change, no mixed journals). This is not yet an end-to-end guarantee covering scoring and every journal: `score.py` does not recompute or check the identity. The approved briefing is required (a draft briefing is refused outside rehearsal-v3). STRICT is refused in test-v1.
- Any change to a value covered by the identity after the freeze is a substantive fix under A25 item 11.7 (new freeze, re-run gates, new sign-off).
- Code status (not built; for the C6 / matrix owners): writing the `"treatment"` record into freeze.json (FREEZE-READINESS C6); a scoring-side identity check; under option (A), the cross-flip provenance check of snapshots and acceptance artifacts.

**5. Sign-off.** Arash approves items 0-4 in a dated ledger entry before fresh practice (choosing (A) or (B) in item 4) and confirms them at the freeze, together with A24.1's other parts; edits he makes are recorded with their date. Status 2026-10-14: Arash confirmed item 4 option (B) as carried out and requested correction of the file count to 47. This explicit confirmation followed practice-v2; it does not retrospectively establish dated pre-practice approval of all items 0-4. Decision 9 also calls for reading A32 today; this entry does not attest that reading is complete. The remaining signatures are due at the freeze summary, with A28 and A29 signed before any FloorPlan15 probe.

### A29 (2026-10-13, stage BATCH-A29; DRAFT - APPROVED IN WORDS BY ARASH, SIGN AT THE FREEZE; from Arash's decision of 2026-10-13 (ledger entry "ARASH DECISION: Kitchens 2/3 = 4 core arms; Batch API for all kitchens; A29 before the Kitchen 1 freeze"), based on Astra's design review docs/final-test/_astra/k2k3-design-cuts.txt; Astra review of this text docs/final-test/_astra/A29-review.{prompt.txt,txt,raw.txt}, verdict ACCEPT WITH CHANGES, all 9 required changes applied and marked [Astra A29 n]; written AFTER FloorPlan1 development results, the BATCH-A29 live batch test (3 dev requests) and the A28 draft, BEFORE the Kitchen 1 freeze, BEFORE any acceptance or test call or outcome in any kitchen and BEFORE any FloorPlan15 / FloorPlan17 probe, capture, practice or evaluation; held-out-test-outcome-blind, NOT data-blind) -- Kitchens 2/3 scope, the replication sample and analysis, the Batch API policy, study-wide accounting and budget contingencies

**0. Status.** Arash approved the design in words ("do all these", accepting the main session's suggestion after Astra). This text is a DRAFT until he signs it in a dated ledger entry at the Kitchen 1 freeze. The signature is needed before the first Kitchen 1 test call (items 4-7 bind Kitchen 1 too) and before any FloorPlan15 / FloorPlan17 agent-side step. A29 is the agent-side and analysis part of the A24 item 12 registration; A28 is the perception part. A29 does not change Kitchen 1's design, hypotheses, H1 rule or claims. **New element needing explicit approval at signature [Astra A29 8]:** the spend-only resizing of the replication kitchens after Kitchen 1 and after FloorPlan15 (item 6) is a prospectively specified cost-adaptive rule; it goes beyond "choose the size outcome-blind before testing".

**1. Kitchens 2/3 scope (nothing in Kitchen 1 changes).**
- FloorPlan15, then FloorPlan17 (after the FloorPlan1 test, A28 item 2), run **G0, G+M, O0 and O+M at K = 10, practice order 0, planner replicate r = 0 only**. Each kitchen has its own frozen briefing and its own K = 10 practice package (A24 item 12). Everything else is common with Kitchen 1: code, ft-v2 contract, validator, 2 repairs, executor rules, scoring and treatment rules. Perception per A28; EN/FA pairing; L1-L3.
- The free **P0 / P+M** reference-plan controls run on every presentation of both kitchens. The `/p` cached-plan replays do not run there.
- **Kitchen 1 only** (no cross-kitchen claim): O+FS and G+FS (H4), REPEAT and the planner / executor variance analysis, the learning curve K = 1, 3, 5, the relocation probe, X0 / X+M and the three exploratory Codex models.
- One execution per presentation and arm (the registered primary run 0). The A21 D execution-unit dedup applies unchanged.

**2. The replication sample.**
- **What is frozen when [Astra A29 1].** The selection algorithm, the candidate sizes, the seed, the applicability protocol and the budget decision rule are frozen before the Kitchen 1 test (by this text and its code hashes). Both kitchens' applicable pools and all candidate sample manifests are generated and hashed before practice in either kitchen and before the human exposure prohibited by A28 item 1, unless A28's unexposed-preparer alternative is used. Later budget decisions select one of these existing manifests; they never redraw one.
- **Pool [Astra A29 2].** The items of the registered **test** split (split_manifest.json `test` after the A26 inclusion manifest; family membership = the split manifest's `families` lists, pinned by the split manifest sha256 and the inclusion manifest sha256) that are applicable in that kitchen (next point). Practice, acceptance, dev and reserved items are never in the pool. This is **transfer of the registered task set to new kitchens, not independent task replication**: the test-family teaching embargo stays in force for both kitchens' briefings, packages and practice, and nothing in either kitchen may be changed on the basis of Kitchen 1 item outcomes.
- **Applicability [Astra A29 3].** Applicability means simulator feasibility under the registered reference-validation procedure, independent of detector accuracy and model responses: the item's reference plan, executed by the frozen final-test executor with the registered setup, reset and scoring rules (the versions recorded at the freeze; the oracle reference executor of `sim/validate_general.py` is not by itself that procedure), reaches the goal, and the goal is false **after the registered setup, before the reference plan**. Seeds, setup timing, the per-item timeout and the infrastructure retries are those of the frozen executor and are recorded with the result. Infrastructure errors leave eligibility unresolved; they do not establish inapplicability (unresolved items are retried under the same rule, then reported). Reference failures and exclusions are reported by item, family and level. This reference-only validation is explicitly authorized at the stage of the first bullet; its outcomes are kept separate from scored agent outcomes and are unavailable to briefing, package and practice construction. The claim is accordingly narrowed to **applicable, reference-solvable tasks**.
- **Sampler (replaces the draft's largest-remainder procedure) [Astra A29 2].** One ordered sample per kitchen: sort the eligible families by descending applicable-item count, ties by family id; select one uniformly random item per family without replacement. Thereafter, at each position t+1, select the non-empty level maximizing (t+1) p_l - m_l, where p_l is the level's share of the applicable pool and m_l the number of that level already selected; break ties L1, L2, L3; draw uniformly among that level's remaining items. Sorted item ids and the frozen RNG implementation (`C.rng("A29-sample", kitchen)`, MASTER_SEED 20261010) are used throughout. Candidate samples are the first min(50, pool), the first min(60, pool), and all applicable items; each item enters with both languages, so the candidate sizes are 100, 120 and up to **250 presentations (primary)**. The three candidates are nested. If the pool has fewer than 125 items, "all applicable items" is the primary sample and the shortfall is reported.
- Estimates are unweighted item averages for the selected task sets, not sampling-weighted estimates for the entire eligible pool [Astra A29 2].
- The size used is chosen by item 6 only.

**3. Replication analysis (exploratory; estimation-focused).**
- **Per kitchen**, on the registered primary run: the paired contrasts G+M - O0 (the H1 contrast), G+M - G0, O+M - O0 and G+M - O+M, each as the h1-rule-1 item-weighted estimate with the nominal 96.5 % CR2-Satterthwaite interval (A24.1-H1 item 2 algebra; family clusters; guard K >= 20; "actual coverage is not assured"), plus EN / FA and L1 / L2 / L3 descriptive rates. h1-rule-1 was validated on Kitchen 1's family sizes only (A24.1-H1 item 3).
- **Pooled replication summary.** est = (est_15 + est_17) / 2: the equal-weight average of the two kitchens' estimates, kitchen as a fixed factor (one intercept per kitchen; no random kitchen effect; no Kitchen 1 data). Variance: the CR2 cluster variance of this linear combination under that working model (identity working covariance), with **families as clusters across both kitchens** (a family present in both kitchens is ONE cluster). With S_fk the sum and n_fk the number of kept item differences of family f in kitchen k, N_k = sum_f n_fk, a_k = 1 / (2 N_k) and c_fk = (1 - n_fk / N_k)^(-1/2) (the leverage term is kitchen-specific):
  se^2 = sum_f [ sum_k a_k c_fk (S_fk - n_fk est_k) ]^2.
  Degrees of freedom [Astra A29 4]: B_fh = sum_k a_k^2 c_fk c_hk [ 1(f = h) n_fk - n_fk n_hk / N_k ], nu = (sum_f B_ff)^2 / sum_{f,h} B_fh^2; the interval uses t_nu(.9825). A family missing from a kitchen contributes zero there. For a separately specified one-kitchen analysis the kitchen weight is set to 1, which recovers A24.1-H1; a missing replication kitchen does not silently convert the registered two-kitchen summary into a one-kitchen result. Before use, the implementation is checked against a standard CR2 implementation on unequal-size, shared-family and missing-data examples.
- **Wording [Astra A29 5].** For G+M - O0, report separately whether the exploratory **average** non-inferiority criterion (pooled lower bound > -5 pp) and the exploratory **each-kitchen** criterion (both kitchen-specific lower bounds > -5 pp; a conjunction) are met. Neither constitutes a simulation-validated confirmatory conclusion; the thesis says "met the exploratory non-inferiority criterion on average across the two replication kitchens" / "in each replication kitchen", never "was shown non-inferior". The other three contrasts are estimation-only; their intervals are not simultaneous. The guard (K >= 20 families with a discordant presentation, counted across both kitchens for the pooled summary, within the kitchen for a kitchen result) and a finite positive se with defined degrees of freedom are required for **any** threshold statement, inferiority included ("upper bound < -5 pp"). Intervals are reported whenever they are defined, even when the guard fails, but then no threshold statement is made. No superiority statement is made from Kitchens 2/3. A criterion that is not met is reported as "not met" for that criterion only.
- **Power [Astra A29 5].** The 0-17 % parity power range of A24.1-H1 item 5 concerns the simulated Kitchen 1 configurations. Power and coverage for these replication samples and their pooled estimator are unknown; >= 80 % power has not been demonstrated. Kitchens 2/3 are therefore registered as **estimation-focused, exploratory replication**; an unmet criterion is never reported as evidence of no difference. With a fall-back size, the language and level analyses are descriptive only.
- Kitchen 1 results are never pooled with Kitchens 2/3. Claims extend to "two kitchens held out from agent development, after the registered perception adaptation (A28) and kitchen-specific teaching" and to the applicable, reference-solvable tasks of item 2. When applicability differs between the kitchens, the common-item intersection is also reported descriptively (it never replaces the primary samples).
- Missing / infra handling, exclusions and stopping rules: as Kitchen 1 (section 13, A21, A24). The A24.2 cost-effectiveness outcomes are reported per kitchen.

**4. Batch / caching policy (all three kitchens).**
- **Every paid Anthropic planner call in Kitchens 1, 2 and 3 is made through the Message Batches API** (`sim/planners/anthropic_batch.py`, `sim/final_test/matrix/plan_batch.py`; `plan.py --batch`, `planfirst.py --batch`), inside plan-first (all planning before any execution, A22).
- **Equivalence is procedural [Astra A29 6].** The batch and synchronous tiers are the same planning treatment: the same model, the same request parameters and the same repair algorithm, with fresh stochastic generations. Byte identity refers to each request's Messages parameters (`AnthropicPlanner.build_body`: model, system, messages, output_config format + effort, adaptive thinking, max_tokens), excluding the batch envelope; the decoding, the response-cache key and the plan journal record are the same (gate `sim/_planners_batch_selftest.py`). Repairs are identical conditional on the same preceding reply. The BATCH-A29 three-request live check establishes the integration, **not equal response distributions**. Resume recovers the original generation; it does not generate or select a replacement answer.
- `custom_id` = the plan cell key when the API accepts it (^[A-Za-z0-9_-]{1,64}$), otherwise `ck_` + sha256(cell key)[:56]; every submission records custom_id, cell, cache key, ledger row id and request sha256.
- **Repairs** are further batch rounds: round 0 = every cell's first call; round k = the k-th repair of the cells still invalid, the repair frame built by the same `PlannerRunner` code from the same failed reply (at most 2 repairs, A24 item 2). A chunk of whole presentation blocks (registered seeded order) runs to completion before the next. Frozen limits: chunk hold <= 10 USD and <= 200 cells.
- **Retries [Astra A29 6].** Retries apply only to requests with a confirmed terminal unbilled result. `errored` with `invalid_request_error` = a transport failure at once (as a synchronous 4xx); `expired`, `canceled` or another `errored` type = unbilled, resubmitted in a later round at most 3 times (the synchronous unbilled-retry rule), then the cell is `infra`, re-queued in at most 2 later sessions (section 13). The retry count is restored from the journal on every restart. A request whose result line is absent is **unresolved**: its reservation stays open, it is not retried, and the run stops for reconciliation. A batch not ended 26 h after submission is cancelled once; its succeeded requests are collected and settled before any resubmission or synchronous fallback. A killed run re-attaches to its submitted batches and never re-submits them.
- **Cost guard.** At submission: the chunk's worst case (first call + 2 repairs per cell, at the batch price) is held per scope and every request is reserved at its own pre-call reserve before the batch is created (a refusal sends nothing); holds and request reservations represent the same liability without double counting. At result: succeeded requests settle at their billed batch cost, unbilled ones are released, and the registered guard (A15 / A19 / A21 B) runs at every chunk boundary on actual spend. An ambiguous batch creation (timeout / 5xx after sending) keeps its reservations open and stops the run for reconciliation with the provider's batch list. The study-wide constraint of item 6 applies on top.
- **Synchronous fallback.** Only if batch creation or status polling fails (HTTP errors or no response) for more than 24 h; ordinary batch processing time is not unavailability. Then the remaining cells of that stage may be planned synchronously (same request parameters), after a revised remaining-cost projection and with reservations at synchronous prices; the tier is recorded per call (`billing_tier`), and the decision is recorded with its spend evidence, outcome-blind.
- **No prompt caching** is added (no `cache_control`; it would change the request parameters). Any cache-read tokens the API reports are recorded and priced, never assumed. Provider prompt caching is distinct from the response cache (recovery / replay of an earlier generation); REPEAT and every registered planner cell still need fresh registered generations. A prompt-caching policy would need its own amendment before the first test call.
- **Cost and latency are outcomes [Astra A29 9].** Reported per kitchen and arm: billed USD, tokens in / out, calls per cell, and a **standard-price sensitivity: each call's recorded usage repriced at standard synchronous rates** (only eligible batch charges are doubled; historical and synchronous charges keep their applicable rates). Batch wall time is reported once per round and per stage, with its arm composition and throughput, never summed as independent per-call latency; it is a planning-throughput property, not a deployment latency. The three synchronous dev calls of the live check (2.2-3.2 s) are illustrative request timings, not deployment-latency estimates. Gemma and Codex are reported as 0 USD marginal API charge, distinguished from electricity, hardware and subscription costs.
- Live check (dev items only, BATCH-A29 step 2; docs/final-test/batch_live/live_test.json): identical first-request sha and input tokens 3/3; valid 3/3 in both tiers; batch billed 0.50 x its list price and 0.49 x the synchronous cost of the same requests; usage.service_tier "batch"; ledger settled = journal, 0 open.

**5. Study-wide accounting.**
- The A24.10 gate (80 USD) and cap (100 USD) apply to the **whole study**: every Anthropic dollar spent so far (all ledgers under sim/ and docs/, historical Sonnet / Haiku charges included, copied ledgers counted once; 2.76 USD recorded on 2026-10-13 including the 0.083 USD live check) plus Kitchen 1 (practice, acceptance with its possible re-run, rehearsal, the retained blocks) plus Kitchens 2 and 3 (their rehearsal allowance, 10 % of their retained cells, and their test cells).
- The recorded total is reconciled with the provider's usage page by Arash at the freeze, after each kitchen and at every pause, over the matching study scope and dates; unresolved commitments are kept during provider reporting lag; the larger value is used and a gap above 5 % is reported.
- Projection: `python sim/final_test_callplan.py --study-a29 --spent <USD> --i0 <tokens> --write` -> docs/final-test/matrix/study_projection_a29.json: the ft-v2 3-call model (first + up to 2 repairs, growing conversation; the `cost_gate_ft2` terms), every +M / FS block at the 2,000-token maximum, S2 as the gate scenario, the batch price, and I0 = the measured ft-v2 first-call input (4,121 tokens in the live check, above the provisional 2,953). It is regenerated at the freeze with the final captures' I0.

**6. Budget gate and contingencies.**
- **Cost-adaptive design, stated [Astra A29 8].** This is a prospectively specified cost-adaptive design. Decisions use only aggregate accounting counters (ledger and provider totals, token counts, billed cell counts), never success scores or item-level performance. Cost can correlate with performance (repairs and token use can reflect task difficulty, also across shared families), so outcome-independent sampling and unconditional interval coverage are not claimed.
- **Study-wide pre-submission constraint [Astra A29 7].** Before any paid submission, reconciled study-wide spend plus unresolved commitments plus the proposed additional worst-case liability must not exceed USD 100. Otherwise no new submission occurs; existing requests are collected and settled. This wiring (one study-wide liability check shared by every namespace, ahead of the per-namespace ledger) and its verification are required **before the first Kitchen 1 test call**.
- **At the freeze:** if the item 5 projection at 250 presentations per replication kitchen is <= 80 USD, both kitchens use the 250-presentation manifest; otherwise 120; otherwise 100. If even 100 does not fit, the Kitchen 1 cut order (A23 / A27: codex, curve12, reloc) is applied to Kitchen 1 first; if that still does not fit, testing does not begin under this protocol (A24.10). The core four are never cut in any kitchen.
- **After the Kitchen 1 test, before FloorPlan15 planning, and after FloorPlan15, before FloorPlan17 planning [Astra A29 8]:** the remaining-cost projection is recomputed as: reconciled study-wide spend + unresolved commitments + for each remaining replication kitchen, sum over its cells of the measured mean billed USD per cell **of that arm** (O0 and O+M separately; numerator = every billed attempt of the arm's main-block core cells, repaired, rejected and infra-failed cells included; denominator = the number of billed cells), taken from Kitchen 1 at the first checkpoint and from FloorPlan15 at the second (FloorPlan15's measurements replace Kitchen 1's), plus that kitchen's 10 % rehearsal allowance. At each checkpoint the largest permitted size no greater than the current one that keeps the projection <= 80 USD is selected (250, 120, 100); the selection is recorded before any agent outcome of the next kitchen exists. If 100 does not fit, planning pauses and the remaining replication kitchen(s) are reported as not run (the study is reported incomplete for the replication; section 13). Unequal kitchen sizes are allowed; the pooled analysis weights the kitchens equally regardless, and their different task compositions are disclosed.
- During any kitchen: the registered cost guard applies; reaching the study-wide constraint stops new submissions, and the study is reported incomplete (section 13) with every completed cell analysed as registered.
- **Projection at the time of writing** (docs/final-test/matrix/study_projection_a29.json; spend 2.76 USD; I0 4,121): S2 batch **67.03 USD at 250** (Kitchen 1 38.85, FloorPlan15 12.71, FloorPlan17 12.71; margin 12.97 to the gate) -> **passes; 250 presentations per replication kitchen**. 53.81 at 120; 51.78 at 100. Sensitivities (reported, not gated): standard price 131.29 at 250; S1 48.26; S3 stress 106.41; every cell making all 3 calls 166.16; repair output x3 70.61. **S2 is a planning scenario, not an upper bound or a quantified probability of completion. The author accepts the disclosed risk of an incomplete study under higher costs [Astra A29 9].** The S3 and every-cell-3-calls sensitivities exceed the 100 USD cap; the study-wide constraint, the guard and the resizing rule are what protect the cap.

**7. Code (hashes recorded at the freeze).** Built (stage BATCH-A29): sim/planners/anthropic_batch.py, sim/final_test/matrix/plan_batch.py, the `--batch` options of plan.py and planfirst.py, the `--study-a29` projection of sim/final_test_callplan.py (`k23_cells`, `study_projection_a29`), gate sim/_planners_batch_selftest.py (87 checks), live check sim/final_test/batch_live/live_batch_test.py. **Owed before the first Kitchen 1 test call:** the study-wide pre-submission liability check of item 6. **Owed before either replication kitchen's practice:** the item 2 sampler and the applicability validation under the frozen executor (pool + candidate manifests, hashed), the per-kitchen namespaces and call-plan wiring. **Owed before the replication analysis:** the pooled estimator of item 3 in analyze.py with its CR2 cross-check.

**8. Astra review (docs/final-test/_astra/A29-review.{prompt.txt, txt, raw.txt}).** Verdict ACCEPT WITH CHANGES; 9 required changes, all applied above (1 timing of rule vs manifests and A28's deadline; 2 sampler, family membership pin, unweighted estimand, transfer wording; 3 applicability under the frozen executor; 4 pooled degrees of freedom and the one-kitchen weight; 5 exploratory wording, guard for every threshold statement, power; 6 procedural equivalence, confirmed-unbilled retries, absent results unresolved, persisted retry counts, the 24 h definition; 7 study-wide pre-submission constraint; 8 cost-adaptive statement and the exact recomputation; 9 S2 is not a bound, the standard-price repricing, batch latency reporting). The two code gaps of change 6 were closed in the same stage (absent result -> open reservation + stop; retry counts restored from the journal; both in the gate). Optional suggestions applied: the common-item intersection (item 3) and the CR2 cross-check (item 3).

### A30 (2026-10-13, stage PRACTICE-V2; DRAFT - APPROVED BY ARASH 2026-10-13, SIGN AT THE FREEZE; from Arash's written decision in the ledger entry "ARASH DECISION: practice leakage gate = option (A); then STOP" ("Go for A."); Astra reviews docs/final-test/_astra/PRACTICE-V2-needles.txt (instruction-field exemption, ACCEPT WITH CHANGES, 5 rules), PRACTICE-V2-reqline.txt (verdict B: exempting the request line is a protocol amendment needing written approval) and PRACTICE-V2-optA.txt (the implementation, ACCEPT); written AFTER practice-v2 attempts 1-2 and BEFORE practice-v2 attempt 3, BEFORE any acceptance call and any test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- the practice-snapshot leakage gate: two field-scoped exemptions in the stored practice rows (amends section 6.3 "Leakage gate" and A24 item 7(b))

**Why.** An ft-v2 practice snapshot stores its practice rows (`practice_rows.jsonl`: the instruction the planner was given and the package text it saw), because the K-prefix of rows travels with the snapshot. The registered gate is a substring test over the whole snapshot. Two legitimate kinds of text in those rows contain a shorter TEST sentence: (1) a practice instruction can embed one (practice item g-l2-slice-bread-tomato contains the test sentences of g-l1-slice-bread, g-l1-slice-tomato, g-l2-slice-bread and g-l2-slice-tomato, and the reserved FA sentence of g-l1-pickup-butter-knife); (2) the G1 line that A25 item 1.2 adds to every +M package ("The request in English: ...", the parser's rendering of the user's own words) can contain one: for practice attempt 2 (g-l2-microwave-apple, FA) it reads "The request in English: put Apple to the microwave; close the microwave.", which holds the whole normalised text of test item g-l1-close-microwave. Neither is derived from a test item; both are practice-own text. The literal gate would fail K = 3, 5 and 10 on (2) and K = 10 on (1).

**1. The exemptions (ft-v2 practice snapshots only).** In the snapshot's own stored practice rows, and nowhere else, two kinds of text are excluded from INSTRUCTION-TEXT matching:
- (a) the `instruction` field of row n, only if row n is at position n of the PINNED seeded practice schedule (`C.rng("practice", order)` over the pinned pool; never the snapshot's own manifest), its item and language equal that schedule entry, its item is no test, test-uncapped or reserved acceptance id, and the field equals the pinned dataset text in that language byte for byte;
- (b) ONE line of `block.text` of the same row, only if row n already passed (a) and the line is byte-identical to the G1 line recomputed from that practice item's own pinned text by the same code path and format as the package (`render_request(_parse_request({instruction, lang, holding}))`, holding = the hand state the item's pinned setup leaves when every setup step succeeds). Two or more G1 lines in one row, a G1 line where the recompute gives none, or a G1 line that differs from the recompute in any byte make the exemption invalid.

**2. Everything else stays refused.** Test and reserved ids are matched as whole tokens on the UNMASKED bytes and on the raw store dump, with no exemption. The store dump, every other field of a row, every other line of `block.text`, and every other file get no text exemption: the same practice instruction or the same G1 line copied into a lesson, a package record, another field or another file is a hit. Normalisation variants (case, punctuation, spacing) are not exempt. Any invalid exemption leaves the rows file unmasked and is itself a hit (`exemption_invalid`), so the gate fails closed; a failing gate aborts the practice before the attempt's row is journaled. The ft-v1 gate (practice-v1, rehearsal-v1) is unchanged byte for byte.

**3. Audit.** Every exemption is recorded in the snapshot manifest's leakage result: per instruction {item, lang, field, text_sha}, per request line {item, lang, field "n.block.text#request_line", text_sha, count}, and the counts `n` and `reqline_n`. The audit is checked against the pinned schedule prefix.

**4. What this does not change.** The treatment (fingerprint d5831abc..., G1 on, the package text) is unchanged: the G1 line is shown to the planner exactly as before, in practice and in the test. The gate code lives in `sim/final_test/matrix/practice_k.py`, outside the treatment identity; its hash is recorded at the freeze (gate 12.1). A24 item 7(b) (acceptance leakage gate 0) applies the same two exemptions, and only these, to the stored practice rows inside a practice snapshot; for the package corpus, the FS corpus and the briefings it is unchanged. The disclosure stands that acceptance and test items share broad families with practice items (acceptance tests unseen ITEMS, not unseen families).

**5. Evidence.** Gate `sim/_ftm_practice_reserved_selftest.py` (65 checks: sections 5 and 6 hold the positive and negative cases of (a) and (b)); offline prediction on the real ft-v2 package text: K = 1/3/5/10 pass with 1/2/4/7 exempt request lines; the live rows 1-2 fail the literal gate on g-l1-close-microwave only and pass with (b).

### A31 (2026-10-13, stage A31; DRAFT - APPROVED BY ARASH 2026-10-13, SIGN AT THE FREEZE; from Arash's decision in the ledger entry "ARASH DECISION: A31 AUTHORIZED, looser leakage checker" ("fix this. make it loose so that it's not so strict"); Astra reviews docs/final-test/_astra/FREEZE-GATES-B1.txt (Q1 re-pin with whole ids, Q2 practice kept only with a replayed equivalence proof, Q3 narrow audited spans) and docs/final-test/_astra/A31-design.txt (design, no REJECT; every CHANGE applied); written AFTER three false alarms of the leakage checks on DEVELOPMENT data (dev namespace rehearsal-v3, FREEZE-GATES steps 3-4) and AFTER practice-v2, BEFORE any acceptance call and BEFORE any test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- the leakage checks: whole-id matching (treatment re-pin) and a looser instruction-text rule for rendered blocks (amends A24 item 7(b), A25 item 1.4 and the capture leakage gate of SPEC s12 M4)

**Disclosure.** This is a rule change made after false alarms, not after any acceptance or test result. No acceptance or test item had been planned, executed or scored when it was written or implemented. It loosens the instruction-text rule on purpose (Arash's decision): a test sentence embedded as a FRAGMENT of a longer sentence is no longer a hit. It does not touch the gold-plan / goal checks.

**Why (the three false alarms).**
- B1: `treatment._cites` (evidence clause) and `treatment.corpus_test_citations` matched test ids by plain substring. Practice example a8's evidence is the row key "practice-v2|practice|o0|8|g-l2-slice-bread-tomato|en"; test id g-l2-slice-bread is a prefix of practice id g-l2-slice-bread-tomato, so every FS render with S* aborted, and in the test the same rule would have made every +M cell of g-l2-slice-bread an integrity failure when a8 is a candidate. Four test ids are prefixes of practice-pool ids (g-l2-slice-apple / -bread / -potato / -tomato).
- B2: the practice instruction a8 "Take the butter knife, slice the bread, then slice the tomato." contains the test sentences "slice the bread" / "slice the tomato" as fragments; the capture scan of RENDERED blocks flagged its FS line (A30 covers stored practice rows only).
- B3: the package's G1 line of dev item g-l2-microwave-run-egg (FA), "The request in English: put Egg to the microwave; close the microwave; switch on the microwave.", the parser's English rendering of the presentation's OWN words, contains test g-l1-close-microwave; offline count: 21-33 of 250 test presentations would abort the same way.

**1. Whole ids (inside the treatment identity; re-pin).** In the evidence clause of `_cites` and of `corpus_test_citations`, an item id counts only as a WHOLE id: `treatment.id_in(id, text)` = the id not preceded or followed by a character of the ASCII id alphabet [A-Za-z0-9_-] (so "|", ":", "/", space, quotes and the string ends are boundaries), case-insensitive (re.ASCII | re.I; this also catches case variants the old rule missed). support, tuned_on and item stay exact set membership. Nothing else in the treatment changes. Old fingerprint d5831abcbf89e393fd79807dfc674dd1fb54d27b864955f958c6dd86f6a4910f, new fingerprint 7a2ee211b85cc534d7677291fce80b04e92cd24a860f1ddd7665185b3a03b67d; the only changed file of the 47 is sim/final_test/matrix/treatment.py (89c38daf... -> ed311a5b...); both manifests and the exact diff are in sim/final_test/diag/a31/repin.json. freeze.json records the new identity (still treatment-only-unvalidated until the full freeze).

**2. Practice-v2 is kept (equivalence certificate).** practice-v2 was generated under d5831abc...; that stays its provenance (the new fingerprint does not retroactively describe its generation). Certificate sim/final_test/diag/a31/equiv_replay.json: five full replays (attempts 1, 2, 4, 6, 10, whose pre-attempt store exists: a new empty store and the byte-identical K=1, K=3, K=5 and pre-attempt-10 dbs) render identical packages under the old and the new code in text, kept ids, dropped records, leave-out log, mask, D2, need rule and every leave-out decision, and the new code reproduces the journaled package of each of them byte for byte; plus a complete equivalence argument covering attempts 3, 5, 7, 8, 9: every static and example record of the exact row prefix compared exhaustively, and the db-derived records (profile, places) proved unable to differ by their id grammar (evidence ids 24 hex, claims lc: + 16 hex, negative evidence ng: + 16 hex, fallback where_is:<Type>; none contains "-", every practice id does), on the generating source lines and on all 6 available db states. corpus_test_citations is never called in practice; no snapshot, consolidation or practice-leakage code uses the changed functions; the four snapshots verify unchanged. Equivalent, so the snapshots are kept (no new practice run).

**3. The rendered-block instruction rule (capture.py, outside the identity; ft-v2 namespaces only).**
- (a) Ids: test, test-uncapped and the 29 reserved acceptance ids (pinned reservation sources) are matched as WHOLE ids (`treatment.id_in`) on the UNMASKED block text, with no exemption.
- (b) Texts: a test, test-uncapped or reserved acceptance instruction (EN or FA) is a hit only as WHOLE delimiter-defined normalised segments: both the block and the instruction are split on CRLF / CR / LF, . ! ? ; : |, the straight double quote, the curly double quotes U+201C / U+201D, the guillemets U+00AB / U+00BB, the Arabic semicolon U+061B, the Arabic question mark U+061F and the arrow "->" (commas, Latin or Arabic, are not delimiters); each segment is normalised as before (NFKC, lower case, Persian letter variants, no diacritics / ZWNJ / punctuation); a hit = the instruction's segment tuple occurs as a contiguous run of whole block segments. These delimiters are operational boundaries, not all linguistic sentence ends. The former 8-character floor is dropped (the whole-segment condition replaces it).
- (c) Exempt spans (removed before (b); each becomes a non-matching barrier, so removal never joins text or makes segments consecutive): (i) G1: the exact span <R> of "The request in English: <R>." where the line is recomputed in capture from the presentation's own pinned instruction, language, told holding and prompt summary by the package's own code path (make_query_profile, the goal-schema parser, render_request, the same format string); (ii) practice examples: the instruction field of each example the block's own render log lists (package `kept`, FS `retained`), mapped through the same source's practice rows, only if the row's item is no test / test-uncapped / reserved id, is in its namespace's pinned pool and the row's instruction equals the pinned dataset text byte for byte; the full rendered wrapper ('"<instruction>" -> ', the FA guillemet form, or FS "<attempt>. <lang> | <instruction> | ") must occur exactly once and only the instruction's characters are exempt.
- (d) Fail closed (integrity abort, never a silent pass): more than one G1 line, a G1 line that is not byte-identical to the recompute, a G1 line where the recompute gives none, an example that does not map to a valid row, a wrapper not found exactly once, a block showing more practice examples than its render log lists, overlapping spans, or an unloadable reserved set.
- (e) Audit: every exemption is recorded per block in capture.jsonl (`leak_exempt`: kind, example id, item, language, character offsets, sha of the exempted text, source, block sha, rows sha, rule "A31-v1").
- (f) The ft-v1 namespaces keep the pre-A31 scan unchanged (`leakage_problems_legacy`). The practice-snapshot gate of A30 is unchanged.

**4. What stays strict / what A31 does not do.** Whole test and acceptance ids anywhere, test and acceptance text outside the exemptions, every integrity check above. A31 changes id and instruction-text detection only. Gold-plan / goal checks remain a separate, required C11 obligation; this capture audit does not establish C11 compliance. Embedded instruction FRAGMENTS are now permitted unless they form complete matching segments outside the exemptions (this is the loosening Arash chose).

**5. Evidence.** Gates: _ftm_treatment_selftest section 9 (the 4 slice collisions do not match; real citations in every clause still do; a8 passes the test-side leave-out while a real citation still raises; the real K=10 corpus is clean) and _ftm_capture_selftest t_a31 (the a8 FS and package lines allowed; the egg-FA G1 line caught without and allowed with its exemption, its recompute equal to the line; 5 planted real leaks caught: a test sentence outside the span, a whole test id beside a span, a row-key id, a Persian test sentence, a reserved id; multi-segment and barrier cases; 9 fail-closed aborts). Pins: capture.py and practice_k.py are pinned separately by gate 12.1 at the freeze. A30 and A31 are signed together at the freeze; the applicable gates are re-run under the final identity.

### A32 (2026-10-14, stage C11-AND-REG; DRAFT - PENDING ARASH'S SIGNATURE AT THE FREEZE (with A30 / A31); written by the night chain while Arash slept, under his night rules; Astra reviews docs/final-test/_astra/C11-RULING.{prompt.txt,txt,raw.txt} (Q1 CHANGE / Q2 AGREE / Q3 AGREE conditionally / Q4 CHANGE / Q5 AGREE; no treatment change required) and docs/final-test/_astra/A32-REG.{prompt.txt,raw.txt} (ACCEPT WITH CHANGES, all applied); written AFTER development results and BEFORE any acceptance call and any test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- the C11 clarifications and the acceptance registration

**Disclosure.** No acceptance or test item has been planned, executed or scored. This amendment changes no treatment file (fingerprint 7a2ee211... unchanged) and no registered rule; it records (1) a dated clarification of how a gold-plan overlap found by the C11 audit is disposed, (2) a required supplemental table, and (3) the acceptance registration that A24 item 7 says must be saved before the first acceptance call.

**1. C11 audit (sim/final_test/diag/c11/c11_audit.py; report c11_audit.json).** Coverage: every channel a planner receives (system prompt; user message = start-sweep summary + holding + own command + package / FS block; repair turn = the model's own reply + the fb-1 feedback text, which the validator builds from the reply, types, holding, tools and observed types only; the Codex flat prompt) is inventoried from the code; on the dev rehearsal all 200 captured requests and the one repair turn re-render byte for byte to their recorded hashes, and the bound bytes carry 0 test / reserved ids and 0 instruction segments (own command and the audited A31 spans exempt). Quarantine: no reserved or test item reaches practice; the item-level leave-out drops exactly the records that cite the target; 7 records citing another reserved item survive for the other reserved items (the registered A25 item 4 rule is target-only; Astra Q2: it stands, with disclosure); support_plan_only is retrospective shape correspondence (every shipped template ships without any reserved item, and its text comes from its definition only). Controls: 24 / 24. **C11 is not yet passed** (Astra Q3): it passes only when this audit is re-run clean under the final identity at the freeze AND the same checks are clean on the acceptance captures (request hashes, every request and repair scanned, exemptions validated, kept / dropped listings reconciled).

**2. Ruling R-G1 (dated clarification, Astra Q1).** Practice attempt a8 (non-test practice item g-l2-slice-bread-tomato, family g-l2-slice-apple-potato, role nontest) succeeded with Gemma's own plan "pickup ButterKnife; slice Bread; slice Tomato"; its first two steps are the whole 2-step gold plan of TEST items g-l2-slice-bread and g-l3-bread-for-breakfast (test family g-l1-slice-apple). Goals differ (a8: bread and tomato sliced; tests: bread sliced); provenance is clean; every arm's system prompt also says "slice it; you must be holding a ButterKnife". The a8 example can be retrieved into FS and +M blocks whenever it ranks; in the dev rehearsal it reached 4 dev presentations x 2 runs (this is not the test exposure count). Census over the whole K = 10 example corpus and every test / uncapped / reserved gold plan of >= 2 steps: this is the only containment. Ruling: a distinctive-gold hit is a disclosed PRACTICE-ADJACENT overlap, not a leak, only if a practice example the block itself lists carries the gold as a contiguous sub-plan and (a) it is a practice attempt of a pinned non-test item in a non-test family, (b) its provenance cites no test or reserved item, (c) the overlap is the example's own content (a strict superset plan, a different goal, an instruction that is no forbidden text). R-G1 never overrides the id / text gates; any other gold hit is investigated as a possible leak and blocks a pass until resolved.

**3. Supplemental table (REQUIRED, Astra Q1; registered here, before any test outcome).** Next to every primary contrast of the locked test, the estimate without the 4 practice-adjacent presentations (g-l2-slice-bread en / fa, g-l3-bread-for-breakfast en / fa), labelled supplemental and descriptive. The primary analysis keeps all 250 presentations.

**4. Acceptance registration (docs/final-test/acceptance_registration.json, self-sha sealed; builder sim/final_test/acceptance_v2/register_a25.py).** Namespace acceptance-v2. Items: the 24 of slate (i) (16 reserved L1 / L2 + 8 new L3; 8 / 8 / 8) with 2 spares in order, each with content hashes of its EN, FA, setup, plan and goal. Families (registered split-manifest link rule): 3 reserved families (g-l1-pickup-apple, g-l1-place-apple-sink-basin, g-l2-cabinet-bowl) and 8 new-item families (no new item links to any split family). World: FloorPlan1 canon layout, canonical reset + the pinned setup, the A14 reset gate; capture once, execute once, r = 0. Seeds: master 20261010; stateless order keys sha256("20261010|acceptance-v2|exec|<pres>") and per-presentation arm keys; Gemma decoding as frozen (temperature 0, top_k 1, seed 7). Arms: the 8 main arms (48 presentations each) + P0 (reference plan, first unit of each presentation) for criterion (e). Plan first: Gemma, then Opus through the Batch API (after the fake over-cap rejection test and the A29 study-wide check), then Codex; execution in key order. Feasibility: all 26 classified feasible, never revised; the 8 new L3 by the live reference check (VALIDATE-L3), the 16 reserved by the same live check before the first call (Astra A32-REG Q4). Replacement: only for a new L3 item that fails its live check, before the first call, by the first PASSING spare in order; an unresolved check or too few passing spares blocks acceptance; the final roster is re-sealed before the first call. Decision rule: PASS iff every A24 item 7 hard criterion (a)-(i) holds, completeness holds (exactly the 48 registered presentations per arm; a missing, stale, skipped or truncated row or audit field fails) and the acceptance-capture C11 checks are clean; criterion (b) under A31 + R-G1; criterion (c) adds the per-presentation listing of kept records naming any reserved item; infrastructure classes and retries as in A21 / A29; one correction cycle only for a documented defect in a predefined category whose remedy keeps the treatment (an (f) shortfall alone does not authorize one); both attempts reported; never tuned; Arash's proceed / stop decision cannot redefine the verdict. Spend: A24 guard + A29 check; ceilings 8 USD per attempt, 16 USD for the namespace (S2 Batch projection 6.27 USD); the night's authorization is 20 USD.

**5. Owed before the first acceptance call** (listed in the registration): the acceptance-v2 matrix plumbing (outside the identity; gates; recorded at the freeze); VALIDATE-L3 finished and the roster re-sealed; the live reference check of the 16 reserved items; the A25 item-9 visibility audit for the new types and the ACC-V2b alternative-route probes; A24.1 and the A24 item 7 "frozen first" rules recorded at the freeze; the fake over-cap rejection test; Arash's go.

### A33 (2026-10-14, stage PAPERWORK; APPROVED BY ARASH 2026-10-14 (decision 8 of docs/final-test/MORNING-DECISIONS.html, ledger entry "ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)"), SIGN AT THE FREEZE (with A30-A32); the P10 amendment that FREEZE-READINESS D13 calls for; Astra review of this text docs/final-test/_astra/PAPERWORK.{prompt.txt,txt,raw.txt}; written AFTER development results and BEFORE any acceptance call and any test call or test outcome; prospective; held-out-test-outcome-blind, NOT data-blind) -- the hardware price of the A24.2 cost outcomes: a reference device price

**Disclosure.** No scored acceptance run or locked-test run has occurred, and no A24.2 cost outcome has been computed from those runs. This statement does not exclude registered reference-plan feasibility checks of acceptance candidates. This amendment changes no treatment file (fingerprint 7a2ee211... unchanged), no planner and no accounting rule; it replaces only the declared basis of the hardware price. The price itself (399 USD) is the number the analysis plan has carried since stage P-cost (2026-10-13).

**1. What changes (amends A24.2's opening paragraph and item 1).** A24.2 says the hardware allocation uses "the declared laptop purchase price". Prospectively, that is replaced by a **reference price: the NVIDIA Jetson Orin Nano Super Developer Kit (8 GB), USD 399** (ledger 2026-10-12 "Hardware price for A24.2": Arash asked for "the minimum system to run this"; 399 USD is NVIDIA's price after the July 2026 increase). The rule is otherwise unchanged: hardware allocation = 399 USD / 6,000 operating hours x attributable planning hours (three years at 2,000 operating hours / year). A24.2's sentence "The laptop purchase price used for hardware allocation is declared by Arash before the first test call and recorded in freeze.json" now reads: the reference price and its basis `amendment:A33` are recorded in freeze.json `a24_2_cost` before the first test call.

**2. What the price is and is not.** It is the price of a plausible onboard computer for a home robot. It is not the price of the machine the study runs on, and the study has not shown that this system (the Gemma planner with the package, the honest executor and the perception models) runs on that device.

**3. Disclosure (stated wherever the cost result is reported).** Planning time is measured on the run laptop (RTX 4060, 8 GB). The headline hardware allocation combines those measured hours with the USD 399 reference-device price. The energy term requires measured, non-overlapping laptop GPU and CPU energy over the attributable planning interval; C5 has not yet implemented this measurement. Until it is available, Gemma cost remains UNKNOWN and the A24.2 statement remains blocked.

**4. Sensitivity row (registered, reported next to the headline).** The analysis view `hw_time_3x` (analysis plan `a24_2_cost.views`, "3x hardware-time sensitivity; measured laptop energy held fixed"): the hardware allocation recomputed with 3 x the measured planning hours, as the sensitivity scenario approved by Arash; the factor 3 is a registered assumption, not a measured Jetson slowdown or a validated prediction. It is not an estimate of the reference device's energy or of total operating cost. It is descriptive: the A24.2 reading rule (lower 95 % bound of R >= 0.80 and upper 95 % bound of C <= 0.10) is applied to the headline view only (analysis plan `claim.view` = "headline").

**5. Records.** `analyze.A24_2_COST.hardware_price_basis` = "amendment:A33" (was "reference_device_pending"), with this disclosure in `hardware_price_disclosure`; the ft-v2 freeze builder copies the basis into freeze.json `a24_2_cost` and checks that this heading exists in this file. The basis is part of the hashed analysis plan, so the analysis-plan sha changes (49eab233... before this amendment); the plan is frozen at the freeze (A21 F). The freeze gate `a24_2_cost` no longer waits for D13; it still needs the energy method (FREEZE-READINESS C5: measured, non-overlapping GPU and CPU energy over the planning interval, not yet built) before it can pass. Without it the Gemma cost stays UNKNOWN and the A24.2 statement stays blocked, as registered.

**6. Sign-off.** Arash approved this choice in words on 2026-10-14 (decision 8: "Jetson Orin Nano Super, 399 USD, as a reference price via a prospective amendment, with the ~3x planning-time sensitivity row and the disclosure that planning time is measured on the laptop."); he signs this text at the freeze summary with A30-A32.

### A24.1-C (2026-10-14, stage A24.1-COST, entered by stage RUNNER-FINAL; APPROVED BY ARASH 2026-10-14 ~11:40 (ledger entry "ARASH APPROVAL: everything needed from him for this part (owed item 4 / A24.1-C)": "approve everything from me for this part and get to the next stage."), SIGNATURE PENDING -- he signs this text at the freeze summary with A30-A33 (docs/final-test/signatures.json; freeze gate 'signatures'); the cost part of A24.1 that A24 item 10 and A32 item 5 (owed item 4 of the acceptance registration) call for; the text below is the Astra-accepted r4 draft docs/final-test/A24.1-COST-DRAFT.md (sha256 8044fe76...) without its draft heading, unchanged; Astra reviews docs/final-test/_astra/A24.1-COST-r1..r4.txt (r4 = ACCEPT); written AFTER development results and BEFORE any acceptance call and any test call or test outcome; held-out-test-outcome-blind, NOT data-blind) -- the re-derived Kitchen 1 ceilings, the A21.B tripwire, and the A24 item 7 "frozen first" inventory

**0. Status and what this text does.**
- A24 item 10 says the core-four ceiling, the whole-plan ceiling and the A21.B tripwire threshold "are re-derived from the new call-plan projection and recorded in A24.1 before any acceptance call, all within the 100 USD cap". That has not been done yet. Today `common.py` holds three old values:
  - the A22 core ceiling of 30 USD, as a placeholder;
  - a test-v1 whole-plan ceiling equal to the 100 USD study cap;
  - the A19 tripwire of 0.024 USD per billed Opus cell, set for the one-repair, synchronous-price ft-v1 prompts.
- **Status: a reviewed draft only.** No approval, no prerequisite record, no acceptance result and no test result is established by this text or by its review.
- **What is computed and what is chosen.** The projections are deterministic results of the inputs and the cost model below. Three things are NEW policy choices: using S3, rounding up to 5 USD, and the revised tripwire formula. They were proposed after development results and after looking at these projections, before any acceptance or test outcome.
- Script: `sim/final_test/_checkpoints/A24.1-COST/derive_a24_1_cost.py`. It writes `derivation.json` in the same folder: the full bundle, with inputs, code hashes, the ledger inventory, and raw and rounded values.
  - It is offline: no Anthropic call, no simulator, and it writes only inside that folder.
  - Totals are computed before display rounding, so a displayed component can differ from its displayed total by one cent. Study subtotals and the illustrative namespace ceilings are also computed from unrounded projections, with display rounding applied only afterwards. The bundle keeps the unrounded study projections next to their displayed values.
- Treatment 7a2ee211, dataset 7df9a925 and freeze.json 068814ab are unchanged.

**1. Inputs.**
- *Price.* Every paid Kitchen 1 planner call goes through the Message Batches API (A29 item 4), at 0.50 x list.
  - Opus 5.5 list price: 4 / 20 USD per million input / output tokens. Batch price: 2 / 10.
  - BATCH-A29 live check: 0.50 x list on 3/3 requests.
  - No prompt caching (A29 item 4).
- *Call model (ft-v2; C4-open step 3, Astra Q-b).*
  - Call k of a cell: input I0 + B + (k-1)(R + F + P), output A + tail.
  - Expected cell = [c1 + p1 c2 + p1 p2 c3] x (1 + failed).
  - **p2 is fixed by this text:** S1 .25, S2 .5, S3 1.0. Per scenario, (p1 repair share / failed / tail) are the A21.G values: S1 .15 / .03 / 0; S2 .15 / .03 / +500 tokens per call; S3 .25 / .06 / +1000.
  - **These are scenario assumptions:** the repair probabilities, the failed-billing surcharge and the reasoning tails. (1 + failed) models a proportional extra billed cost; it is not a measured distribution of transport retries. The model holds the replayed reply length R at 284 even when billed output grows. S3 and the "all three calls" sensitivity are not token-maximum bounds and do not guarantee completion.
- *Prompt sizes* (FREEZE-GATES step 4; `sim/final_test/diag/freeze_gates/cost_projection.json` sha 001a02f8...):

| term | tokens | what it is |
|---|---|---|
| I0 | **3,754** | the observed maximum of 7 real post-sweep ft-v2 `none` requests (count_tokens; minimum 3,315). An observed maximum, **not an upper bound over future requests** |
| B(+M) | **1,158** | the observed maximum S* / curve block, header included |
| B(+FS) | **572** | the whole K = 10 FS corpus as one block, which is an upper bound of any FS block |
| F | 51 | the repair frame |
| P | **1,200** | the fb-1 allowance. `planners/runner.py` (registration frozen_refs 11cfa254...) caps the problem text at 600 folded characters (pinned). The Anthropic factor of 2.0 tokens per character is a conservative modelling allowance (measured maximum 1.02, on 16 adversarial families). It is an empirical calibration, not a demonstrated tokenizer-wide bound. `problem_token_bound("anthropic")` = 1,200 |
| A = R | 284 | a development prior (the largest billed Opus output, thinking included). No post-sweep Opus output exists yet |

  - **Blocks per scenario:** S1 and S2 use B(+M) = 1,158 and B(+FS) = 572. **S3 uses the maximum block size, 1,158, for both** (`callplan._ft2_block`). The S3 label's "800-token blocks" wording is obsolete.
  - Sensitivity 1: I0 = 4,121, the BATCH-A29 live-check first request on older capture rows (step 3b).
  - Sensitivity 2: **repair-call base output A x 3, reasoning tail unchanged** (step 3). The repair calls bill 3 x 284 + tail output tokens instead of 284 + tail. The first call is unchanged and the replayed reply R stays 284. This sensitivity does **not** triple the whole billed repair output. Tripling the whole S3 repair output would instead project 37.18 USD core / 61.59 USD total (Astra r2; not a scenario of this text).
- *Scope (A27 / A29).* Kitchen 1 test-v1 = `final_test_callplan.cells()` under A27. The paid cells are Opus only:
  - **core** = O0 and O+M in the main and REPEAT blocks: 250 + 250 + 120 + 120 = **740 cells**;
  - **other** = O+FS main, the curve O+M at K = 1, 3, 5, and relocation O0 / O+M: 250 + 3 x 60 + 20 + 20 = **470 cells**.
  - Gemma, Codex (plan-only), P0 / P+M and the /p replays cost 0 USD. Every cell has up to 2 repairs.
  - Acceptance-v2 has its own registered ceilings (8 USD per attempt, 16 for the namespace; A32), and so do the dress rehearsal and FloorPlan15 / FloorPlan17 (A29). None of them is charged to test-v1. In A29, "r = 0" means one planner replicate per presentation (no REPEAT). The replication cells still have up to 2 repairs.
- *Spend so far.* The reported ledger liability is 3.435213 USD: 3.352099 settled plus 0.083114 open, over 10 repo ledgers (`acc_common.study_spend()`; archived copies are over-counted, which is the safe direction). **The derivation uses 3.44 USD.** A29 recorded 2.76, and the larger value is used. Arash still has to reconcile it with the provider usage page (A29 item 5).

**2. Per-cell cost, S2, batch price (USD).**
| arm | block | input tokens, calls 1 / 2 / 3 | output per call | call 1 / 2 / 3 | expected cell | all three modelled calls, incl. failure surcharge |
|---|---|---|---|---|---|---|
| O0 | 0 | 3,754 / 5,289 / 6,824 | 784 | .015348 / .018418 / .021488 | **.02031** | .05691 |
| O+M | 1,158 | 4,912 / 6,447 / 7,982 | 784 | .017664 / .020734 / .023804 | **.02324** | .06407 |
| O+FS | 572 | 4,326 / 5,861 / 7,396 | 784 | .016492 / .019562 / .022632 | **.02176** | .06045 |

Worked example: O0 call 1 costs (2 x 3,754 + 10 x 784) / 10^6 = 0.015348 USD. Its S2 expected cell cost is (0.015348 + 0.15 x 0.018418 + 0.15 x 0.5 x 0.021488) x 1.03 = 0.020313969 USD, displayed as 0.02031. Acceptance-v2 (O0, O+M and O+FS equally weighted) averages 0.0217692 USD per planned cell.

**3. Kitchen 1 test-v1 projection by ledger scope (USD; 740 core + 470 other planned paid cells).**
| scenario | price | core | other | total | mean per planned paid cell |
|---|---|---|---|---|---|
| S1 | batch | 10.94 | 7.20 | 18.14 | .0150 |
| **S2 (gate scenario)** | **batch** | **16.11** | **10.49** | **26.61** | **.02199** |
| S2 | batch, repair-call A x 3 (tail unchanged) | 17.09 | 11.11 | 28.20 | .0233 |
| S3 stress | batch | **27.11** | 18.01 | **45.12** | .0373 |
| S3 | batch, repair-call A x 3 (tail unchanged) | 29.34 | 19.43 | 48.76 | .0403 |
| S2 | list (synchronous) | 32.23 | 20.99 | 53.21 | .0440 |
| S2 | batch, all three calls in every cell | 44.76 | 29.06 | 73.83 | .0610 |
| S3 | list | 54.22 | 36.02 | 90.24 | .0746 |

3b. **Sensitivity, I0 = 4,121 (batch):**
- S2: core 16.80, other 10.93, total 27.73.
- S3: core 27.97, other 18.56, total **46.53** (mean .0385).
- **Combined, I0 = 4,121 plus repair-call A x 3:** S3 core **30.20**, total **50.18** (mean .0415). This **exceeds both proposed ceilings**, by 0.20 and 0.18 USD. It is disclosed, not used: the rule takes the base S3 scenario at the measured I0.

**4. The study gate (A24.10 / A29 item 6; S2, batch, spent 3.44).**
- **62.93 USD** with FloorPlan15 / FloorPlan17 at 250 presentations each (unrounded 62.929175842; margin 17.07 to the 80 USD gate). At 120 presentations: 50.47 (50.473832656). At 100: 48.56 (48.557626012).
- Of the 62.93:
  - Kitchen 1 is 35.54: retained blocks 26.61, acceptance with its possible re-run 6.27, rehearsal 2.66.
  - FloorPlan15 and FloorPlan17 are 11.98 each.
- This restates the FREEZE-GATES 62.25 USD figure, which used spent = 2.76.
- The current S2 gate provisionally supports 250 presentations.

**5. Proposed ceilings (test-v1 namespace).**
- *Rule (NEW).* For the core scope and for the whole test-v1 scope separately: round the **unrounded S3 batch projection** up to the next multiple of 5 USD. Only displayed results are rounded. There is no separate "other" ceiling: "other" is bounded by the total and by the A15 core-reserve rule.
  - This amendment replaces the earlier core and whole-plan figures and their S2-based rationale (A22 item 1: 30 / 65 USD, chosen from the S2 projections; A24 item 10 replaced the 65 with the 100 USD study cap).
  - Why S3: the core four may never be removed by a discretionary budget cut (A24.10). At A22's list prices S3 did not fit inside the cap; at batch prices it does.
  - **Protection does not guarantee completion.** Reservations or actual costs can still exhaust a ceiling, and then the registered pause or incomplete-study procedure applies.
- **Core-four ceiling = 30 USD.** The unrounded S3 core is 27.1100406, and the next multiple of 5 is 30. The number equals the A22 placeholder; it is now derived. Margins below 30: S2 core 16.11 by 13.89; S3 core by 2.89; S3 with I0 4,121 (27.97) by 2.03; S3 with repair-call A x 3 (29.34) by 0.66. The combined I0 4,121 + A x 3 sensitivity (30.20) exceeds the ceiling (step 3b).
- **Whole-plan (test-v1 total) ceiling = 50 USD**, down from 100. The unrounded S3 total is 45.1203045, and the next multiple of 5 is 50. While the core is unfinished, "other" may use only 50 - 30 = 20 USD; the S3 "other" projection is 18.0102639.
- *Mixed ceiling-and-projection affordability check (under the 100 USD cap).*
  - Spent liability plus the acceptance-v2 and test-v1 ceilings is 3.44 + 16 + 50 = 69.44 USD, which leaves 30.56 USD.
  - The remaining S2 projections are 2.66064244 USD for the Kitchen 1 rehearsal and 11.976291525 USD for each replication namespace. Together they total 26.61322549 USD (displayed 26.61), which is inside the 30.56 USD that remains. The two replication namespaces together project 23.95258305 USD (displayed 23.95).
  - This does **not** show that all namespace ceilings fit inside 100 USD together. For illustration only (not a recorded allocation), the same rule applied to the rest would give: Kitchen 1 rehearsal 5 USD; each replication namespace 25 USD at 250 presentations (stack 124.44 USD), or 10 USD at 120 (stack 94.44 USD).
  - The S3 projection of the rest does not fit at 250: 4.51203045 + 2 x 20.1493545 = 44.81 USD. At 120, the remaining S3 projection is 4.51203045 + 2 x 9.67169016 = 23.85541077 USD, displayed as 23.86 USD. The hard protection of the 100 USD cap is A29's study-wide pre-submission check. That check is wired for every paid namespace whose ledger sits in this repo's matrix work dir (`acc_common.guard_problems` -> `paid_namespace_of_repo`), so test-v1 is covered. The rehearsal and replication namespaces do not exist yet, so their coverage is verified when they are built (A29 item 7).
- *Resizing.* Any later resizing follows A29 item 6 only: its timing, its trigger, its manifest-selection rule, and only permitted cost and integrity counters. This text authorizes no resizing after replication outcomes are seen, and no change to the sample of a namespace that has started. Regenerated study gates count the actual committed liability plus the remaining work, without counting a completed phase twice.
- *Not covered (disclosed).*
  - A full synchronous fallback for Kitchen 1 (S2 at list price: core 32.23 > 30, total 53.21 > 50) does not fit these ceilings. A29 item 4 allows a fallback only after a revised projection, so a fallback of that size needs an amendment before it starts.
  - The ceilings of the rehearsal and replication namespaces are not set here. They are owed when those namespaces are built (A29 item 7).

**6. Proposed tripwire (A21.B / A19).**
- *Reference.* The S2 batch projection is 26.6064244 USD over 1,210 planned paid Opus cells, which gives 0.0219887805 USD per planned cell. Rounded to the nearest 0.001 USD (ties to even), the reference is 0.022.
  - We use this as the billed-cell reference under the modelling assumption that every planned paid cell is charged something.
  - Unbilled terminal cells are excluded from the guard's actual denominator, so in operation the two means need not be equal.
- *Threshold.* A19's form, 2 x reference, would give 0.044. The ceiling-to-cell ratios are 30 / 740 = 0.0405 and 50 / 1,210 = 0.0413.
  - These ratios are only average budget allowances for the originally planned scopes. Their minimum is used as a conservative heuristic threshold.
  - A pooled running mean cannot show whether the core or the rest of the plan can be completed. Example: if core cells average 0.041 USD (30.34 USD in total, above the core ceiling) and other cells average 0.020, the pooled mean is only 0.0328, below the threshold. That case needs the separate scope-specific projection and reservation checks, which A21.B already has: projected completion per scope, core and total.
- **Proposed: pause above 0.040 USD per billed Opus cell** = floor to 0.001 of min(2 x reference, core ceiling / planned core cells, total ceiling / planned paid cells). This is NEW and extends A19. **0.040 is not 2 x 0.022.**
- *Unchanged in A21.B:*
  - the numerator, denominator and minimum sample of 60;
  - the cumulative AND the last-60 window;
  - checks at block / chunk boundaries only;
  - the projection and reconciliation checks;
  - the three pause responses;
  - outcome-blind decisions.
- *The old value.* The old 0.024 threshold is about 9.1 % above the proposed 0.022 reference. The S2 expected means do not exceed it. Actual cumulative or last-60 means could exceed it, but these projections do not measure how likely that is.
- *Scenario averages against 0.040* (`derivation.json` step 7; these compare scenario averages with the threshold; they are not simulations of the cumulative or last-60 guard):
  - **below:** S1 batch .015; S2 batch .022; S2 repair-call base output A x 3, reasoning tail unchanged: .023; S3 batch .037; S3 at I0 4,121 .0385;
  - **above:** S2 list .044; all three calls .046-.079; S3 list .075; **and S3 with repair-call A x 3, .0403, although that regime fits both ceilings** (29.34 / 48.76).
  - So these comparisons do not show that every trajectory that exceeds a ceiling trips the mean guard, or that every trajectory within the ceilings avoids it.
- *Limits.*
  - It is a tripwire, not a forecast.
  - The guard reads the cumulative mean and the last-60 billed-cell mean at eligible boundaries only. A transient crossing between checks can disappear before it is read and never cause a pause. A 200-cell batch chunk (A29 item 4: at most 200 cells and at most 10 USD) can contain an expensive 60-cell window that is no longer in the last-60 window when the chunk ends.
  - The one-chunk delay bound holds only under verified submission order and for a crossing that lasts until an eligible check.
  - The reservation and study-cap checks are separate protection.

**7. The A24 item 7 "frozen first" requirements: proposed bindings and status (shas taken 2026-10-14).** The table below names the artefact proposed to bind each frozen-first requirement. Current artefact hashes do not, on their own, show that the prerequisite record has been approved, written or validated. Those steps are still pending. The machine record will carry full digests, artefact paths, hash scope, status and approval provenance; the short digests here are for reading only.

| requirement (A24 item 7) | governing text | proposed binding (hash scope) | status |
|---|---|---|---|
| all acceptance rules | A24 item 7 (criteria (a)-(i), one correction cycle); A32 item 4 (decision rule, completeness, C11 on acceptance captures, R-G1, the infrastructure exception for (f), the replacement rule, order and seeds, spend) | `docs/final-test/acceptance_registration.json` (whole file 2622f6c2...; self-sha fef2bd3e... over every section) and the runner that implements it: `sim/final_test/acceptance_v2/acc_runner.py` + `sim/final_test/matrix/acc_common.py` (re-pinned at the pre-acceptance record) | registered (A32); the runner pins are pending, and acc_runner.py is being edited by another stage (ACC-G) |
| retrieval quotas | A24 item 1 / A25 | `sim/step5/package_params.json` 9a07f2d2... (whole file), inside the treatment identity 7a2ee211... | frozen through the identity; record pending |
| the evidence-eligibility rule | A24 item 1 / A25 item 4 | `sim/ltm_package.py` da29b4a9... (whole file), inside the identity | frozen through the identity; record pending |
| relevance annotations | A24 item 7 "reported, not gated" | **No manual relevance-annotation file exists.** The diagnostic proxy `ltm_package.needed_facts` takes the object-type set (the `object` and `to` fields) and the action-name set (`action`) from the pinned reference plans. It does **not** read goal fields. `ltm_package.covered` / `early_check` define coverage and reporting. The proxy never feeds retrieval and is never an acceptance pass criterion. Inputs: the plans of dataset 7df9a925... and the registration's per-item plan hashes. Code: ltm_package da29b4a9... | NEW: this text registers the narrower plan-only proxy in place of A24's "reference plans and goals" wording (needs approval) |
| feasibility classifications | A24 item 6; A32 item 4 | (a) acceptance: the registration's `feasibility.classification` maps **26 items = the 24 sealed roster items + the 2 spare L3 items**, all "feasible", never revised. The sealed roster (24, 0 replacements) and its live reference-check basis (2 fresh restores, PASS) sit in the registration's roster section (self-sha fef2bd3e...). Classification is separate from evidence that the reference plans execute: that is criterion (e) at acceptance, plus the live checks of owed items 1-2. (b) The test L3 audit (A24 item 6): `docs/final-test/kitchen_memory/l3_feasibility_summary.json` 59c41148... is **stale** (built on dataset 956f0ae8, before A26). The latest item-level audit rows are `sim/final_test/diag/spec_inhand_apply/regress/l3_rows_apply.jsonl` 6b7d266d... (23 non-test jobs, executor ea860610 = the current honest_skills.py: 18 feasible, the rest perception / unreviewed) | (a) registered; (b) **PENDING**: no fixed item-level classification record exists under the current dataset and executor |
| infrastructure-retry rules | A21 A (classes, precedence, retries, hang guard); A29 item 4 (batch retries: confirmed unbilled only; an absent result is unresolved; persisted retry counts; the 26 h cancel) | `common.BLOCKED_RULE` + `RUN_CLASSES` + `INFRA_FLAG_RATE`: digest a8f15ee9... = `common.sha256_json([BLOCKED_RULE, RUN_CLASSES, INFRA_FLAG_RATE])` (sha256 of the canonical JSON, the scope freeze.py uses for `blocked_run_rule`), plus `sim/final_test/matrix/plan_batch.py` 9079ef1a... (whole file; it is not in the registration's frozen_refs) | **PENDING**: the current freeze.json (`treatment-only-unvalidated`) holds no `blocked_run_rule` or `cost_guard` entry |

**8. Order of work and the tuning lock.**
- *Pre-acceptance prerequisite record (before the first acceptance call):*
  - Arash approves the derivation rule, the input-selection procedure, the scenario assumptions, the numerical ceilings and the tripwire.
  - The required record is written and validated.
  - That record binds the applicable code and all six frozen-first requirements of step 7, the PENDING ones included.
- *Implementation after approval (nothing here is done yet):*
  - `common.py` test-v1 ceilings become core 30 / total 50.
  - `OPUS_REF_USD_PER_CALL` becomes 0.022, and `OPUS_GUARD_USD_PER_CALL` becomes an explicit 0.040.
  - `COST_GUARD_SPEC` becomes version ftm-guard-3:
    - the unit is one billed Opus cell, meaning the first transmission plus **up to two repair turns** plus every billed transport attempt;
    - the in-flight wording covers A29 batch execution (chunk holds, request reservations, unresolved results), not "the runner is sequential";
    - the `cost_guard` docstring and messages say "USD per billed Opus cell" and no longer claim the threshold is 2 x the reference.
  - Boundary tests use the explicit threshold: the cumulative and recent-window checks each on their own, the 60-cell minimum, and aggregation of repaired and failed cells. `gate_evidence` 12.7 (whose probes are multiples of the reference) is updated and re-run.
  - The owed item 4 validator (`acc_runner._owed_frozen_first`, owned by the acc_runner stage) has to check much more than "the tripwire and frozen_first fields are non-empty". Required: it verifies structured records for all six frozen-first requirements, resolves and checks their artefact digests, compares the recorded guard specification with the active one, and verifies both the acceptance ceilings and the derived test-v1 ceilings. Missing or stale prerequisite bindings stop the first acceptance call.
  - A24.1-C goes into preregistration.md after A33 (by that file's owner), with Arash's signature.
- *Final test freeze (after acceptance):*
  - The study's remaining-cost projection is regenerated and the recorded assumptions and ceilings are checked.
  - That check does not authorize raising a ceiling or changing the tripwire. A required change follows the applicable amendment and phase-restart rules, with all acceptance evidence available at that time disclosed.
- *Tuning lock:*
  - Acceptance results, SR, arm differences, and feasibility successes or failures may not choose the ceiling rule, the scenario, the rounding step, the reference, the threshold or the capture subset used here.
  - The registered acceptance correction cycle stays limited to its listed hard-gate failures.
  - Changing these values after the first test call is a protocol change (A21.B "not allowed").

**9. Implementation status (added by RUNNER-FINAL, 2026-10-14; not part of the reviewed text).** Item 0 above describes the state when the draft was written. Since Arash's approval: `common.py` holds the test-v1 ceilings core 30 / total 50, `OPUS_REF_USD_PER_CALL` 0.022, the explicit `OPUS_GUARD_USD_PER_CALL` 0.040 and `COST_GUARD_SPEC` ftm-guard-3, with the boundary tests (`_ftm_common_selftest`) and `gate_evidence` 12.7; freeze.json carries the `cost_guard` and `blocked_run_rule` entries (still a treatment-only, unvalidated freeze); the test L3 feasibility record is `docs/final-test/test_l3_feasibility.json`; `acc_runner._owed_frozen_first` checks the record by content. The machine record is `docs/final-test/a24_1_record.json`.

### A38 (2026-10-17; SIGNED BY ARASH 2026-10-17: "I sign the K1-weight transfer amendment for Kitchens 2 and 3."; from Arash's idea the same day ("Our models for kitchen 1 already have gained some knowledge about this simulator environment, can we use those weights and knowledge to not start over training for kitchen 2 and 3?"); Astra reviews docs/final-test/_astra/K23-TRANSFER-r1.txt (design) and K23-PARALLEL-r1.txt item 2 (bindings, all applied); written AFTER the FloorPlan15 main capture (168 episodes) and 8 top-up episodes, its automated QC and the pre-capture eligibility inspection, and BEFORE any Kitchen 2 / 3 training or any perception-performance inspection; prospective, joint for both kitchens) -- Kitchen-1-initialized perception adaptation (amends A28: the original COCO checkpoints and the ban on reusing Kitchen 1 weights)
1. **Primary method, both kitchens:** every Kitchen 2 / 3 model is initialized from the FROZEN Kitchen 1 model of the same role, each kitchen independently from the same Kitchen 1 files (never Kitchen 2 -> Kitchen 3). Sources (sha256): primary FloorPlan1.yolo26s.pt 4418f63f...b90f54; specialist FloorPlan1.amb26m.pt b02c5d05...13b356; pooled-data control: initialized from THAT kitchen's adapted primary (A28's control rule, unchanged); state classifiers FloorPlan1.state_{broken 39c02b0a..., cooked 7eb96a17..., dirty c802083a..., filled cfa9e536..., on 752f9b61..., open a0e110a0..., sliced 638c6711...}.pt (full hashes in sim/final_test/_checkpoints/K2-TOPUP/k1_source_hashes.sha256). The YOLO11s baseline omission of A28 is unchanged.
2. **Layers.** Primary detector: backbone / neck / box layers loaded; BOTH YOLO26 classification branches (one-to-many and one-to-one heads) re-initialized for the kitchen's ordered class list (no semantic remapping). Specialist and state classifiers: outputs retained ONLY when the ordered label list is identical to Kitchen 1's (asserted in code); any mismatch re-initializes that output layer (recorded), never silently retained. The transfer code writes an auditable list of loaded / re-initialized parameters (name, shape) per model; its file hash and the list are frozen with each kitchen's perception freeze.
3. **Training.** Fresh optimizer AND scheduler (resume False, no inherited epochs / LR state); epochs, patience, batch, image size, seed, splits and VAL-based selection exactly as A28 / Kitchen 1. State calibration (size floors from Arash's new hand labels, VAL thresholds) refit per kitchen.
4. **One registered sensitivity (FloorPlan15 only):** a COCO-initialized YOLO26s primary on identical data and settings. Both FP15 primary checkpoints are frozen before any TEST evaluation. Paired endpoints: TEST mAP50 and mAP50-95 (paired bootstrap over TEST episodes, 95 % CI), per-class AP50, epochs run / best epoch, GPU-hours. The COCO run never selects the deployed model and cannot trigger further runs.
5. **Claims and disclosure.** Wording: "Kitchen-1-initialized supervised perception adaptation"; held out from agent development; Kitchen 1 / 2 / 3 perception comparisons do not have equal training exposure and are descriptive. No time saving is promised. Disclosed: the FP15 capture, its automated QC, the eligibility inspection and the 8 unsupervised top-up episodes (ledger K2-TOPUP INCIDENT) all preceded this amendment.

### A39 (2026-10-20; APPROVED BY ARASH 2026-10-20 ~16:15 (AskUserQuestion, his choice: "Approve, append once (Recommended)"); Astra review docs/final-test/k23/_astra/K23-PREREG-APPEND-r2.txt (r1 CHANGES REQUIRED, applied); written AFTER the FloorPlan15 A29 applicability run and A29 sample (2026-10-20 11:59-12:15) and BEFORE any Kitchen 2 / 3 practice, rehearsal, agent freeze, paid planning call or test outcome; Kitchen-2/3-agent-outcome-blind: no Kitchen 2 / 3 agent outcome exists) -- the Kitchen 2 / 3 register: A29.1 and the K23 records, appended once

**0. What this is.** One append that records, before the Kitchen 2 / 3 agent freezes, the procedures and decisions under which Kitchens 2 / 3 (FloorPlan15, FloorPlan17) are run. Each part names its source record; abbreviated SHA-256 prefixes are shown. Nothing in Kitchen 1's design, data, results or claims changes. The Kitchen 2 / 3 agent freezes (docs/final-test/k23/freeze_<scene>.json) bind the sha256 of this preregistration as it stands after this append. Earlier records that bind an older preregistration sha were already stale after the A38 append of 2026-10-17: Kitchen 1's freeze.json, signatures.json (through which the A36 / A36-IMPL approval validators compare the preregistration), docs/final-test/matrix/gates_ft2/leakage.json and the C11 audit (all 031349dd...), and reset_tolerances.json (df4d3852..., read only when ceilings are initialised; the A14a section it hashes is unchanged). The A36 implementation record, the A37 alignment record, the treatment pins, the Kitchen 1 manifest and the Kitchen 2 perception freeze do not bind the preregistration. This append changes no satisfied binding, and no section an existing reader parses (A12 basis line, A14a, A24.1-C, A33).

#### A29.1 (written 2026-10-17 22:47 as docs/final-test/k23/A29.1-applicability-draft.md, before any applicability run; adopted by A39) -- applicability procedure for Kitchens 2 / 3

A prospective clarification of A29 item 2 "Applicability" (Astra docs/final-test/_astra/K23-A29-APPLIC-r1.txt; runner review K23-A29-RUNNER-r1). It changes no sampler, size, hypothesis or analysis rule; covered by approve-all-6 item 5 (the agent-side amendment).

1. **Interpretation.** "Independent of detector accuracy and model responses" means: no model is called (the item's registered reference plan replaces the planner) and no detector-performance threshold is used to select items. The execution itself uses the kitchen's frozen perception stack (A38 detectors, calibration, look-alike map, walked map), so a detector-caused failure can make an item ineligible; the thesis states this dependence. Oracle-perception execution is not used.
2. **Executor.** The frozen final-test executor (matrix LiveExecutor, honest mode, console policy) with the registered restore / reset / setup checks, scoring block, 12-step / 3-retry / 3000-primitive / 900 s caps (C.CAPS), condition P0 without learned memory, the reference plan given as the plan.
3. **Timing.** After the kitchen's perception freeze (A28: no agent-task outcome is seen before it); before practice in either kitchen. Both kitchens' pools and nested 50 / 60 / all manifests (a29_sample.py) are hashed before either kitchen's practice starts.
4. **Runs.** One primary execution per item, EN as the canonical validation presentation (the reference plan is language-independent; a registered 10-item FA equivalence sample, C.rng("A29-equiv", kitchen), is run and its agreement reported; the scored P0 / P+M controls still run both languages). Only infrastructure failures (crash, hang) are re-run: one run + up to 2 reruns per retry round; round 1 is the primary pass, a later round is started explicitly; an interrupted invocation resumes its own round. No repeat of a valid run, no vote.
5. **Classes.** applicable = goal false after setup AND true after the reference plan, within the caps. inapplicable = object_absent, goal_true_at_setup, step_cap (reference plan longer than 12 steps) or reference_execution_failure (a valid execution that does not reach the goal, including a 900 s wall-cap overrun; cause recorded). unresolved = infrastructure after the retries, or a setup failure, which is never excluded automatically (reviewed and resolved by a recorded decision). INVALID = an honesty-invariant break: the run stops, the evidence is kept, the item stays unresolved until a registered correction. The sampler refuses any item that is not applicable / inapplicable.
6. **Records.** Per attempt: class, cause, round, attempt, pre / post goal truth, step rows with causes, primitives, audit, sweep primitives, the captured post-sweep reset state hash, wall time, world_seed (a label). A binding row, written before the first execution and required to match on every resume, holds the hashes of the executor / scorer / reference / dataset / split / inclusion files, the caps and the verified perception freeze. Runner: sim/final_test/matrix/a29_applicability.py.
7. **Quarantine.** The results file is read only by a29_sample.py and by freeze / audit verification (the evaluator gate of the Kitchen 2 / 3 agent freeze); briefing, package and practice builders never read it; the test-family embargo stays in force.
8. **As applied (record).** FloorPlan15: round 1 on 2026-10-20 11:59-12:11 (sim/kitchen_v3/k23/a29_run.sh), under the superseding FloorPlan15 perception freeze record 4a139f55... (written coco_pending, see A39.2): 125 test items, 100 applicable, 25 inapplicable (all reference_execution_failure), 0 unresolved (applicability record sha256_json 957b7e32..., docs/final-test/k23/applicability/FloorPlan15/applicability.json); sample frozen 12:15 (docs/final-test/k23/a29_sample_FloorPlan15.json, self-sha 41aced3d...): pool 100 (L1 47 / L2 35 / L3 18; 40 families) < 125, so "all" is the primary sample (200 presentations; shortfall 25 reported). The registered 10-item FA equivalence run of item 4 (a29_applicability.py --equivalence) had not been run for FloorPlan15 when this text was written. Each kitchen's FA equivalence report is required before that kitchen's agent freeze (reference plans only; it does not change eligibility or the frozen sample). This is a manual prerequisite: the agent-freeze evaluator gate does not check it. FloorPlan17 follows the same procedure after its perception freeze; its record and sample are named in its agent freeze.

#### A39.1 The A29 item 6 size decision as implemented

- Tool sim/final_test/matrix/k23_size_decision.py writes docs/final-test/k23/a29_size_decision.json (read by k23_ns.a29_roster). Sizes are the frozen nested candidates (all / first_60 / first_50 = up to 250 / 120 / 100 presentations); a decision selects an existing candidate and never redraws or grows.
- **after-k1** (the Kitchen 1 score sealed, both A29 manifests frozen, no test-k2-v1 plan row): one size for both kitchens, never above the freeze-time selection (docs/final-test/matrix/study_projection_a29.json, 1c54dd38...: 250 passed, i.e. "all").
- **after-fp15** (test-k2-v1 has a full plan_first_complete event and no open ledger reservation; no test-k3-v1 plan row): FloorPlan17 alone, never larger. "After FloorPlan15" is read as "after FloorPlan15's paid planning is complete": Kitchen 2 execution is unbilled, so every FloorPlan15 cost counter is final then (Astra K23-TOOLING-G3).
- Projection: reconciled study-wide liability + for each remaining kitchen its presentations x (the measuring kitchen's mean billed USD per O0 / O+M canonical r = 0 cell, every billed attempt, from its ledger) + 10 % rehearsal allowance net of that kitchen's rehearsal spend already billed; gate C.COST_GATE_USD (80). Nothing fits: planning pauses and the remaining kitchen(s) are reported not run.
- **Spend basis (Arash, 2026-10-20 ~16:15):** ledger-only accounting at after-k1: the reconciled ledger liability, recomputed when the checkpoint runs (29.96 USD when Arash approved this; spend added later, such as rehearsals, is included then); the provider usage-page reconciliation of A29 item 5 is deferred to the FloorPlan17 checkpoint (disclosed).

#### A39.2 A38 item 4 order (the FloorPlan15 COCO-start sensitivity)

- The FloorPlan15 perception freeze was written --coco-pending (2026-10-18 16:46; superseded 2026-10-19 13:32 by record 4a139f55..., coco_pending kept). Arash, 2026-10-20 ~12:40 (AskUserQuestion, after Astra K23-PENDING-MODELS-r1): "No, keep the current order": the COCO run and its append (k23_freeze.py --scene FloorPlan15 --add-coco) stay before the Kitchen 2 agent test; the Kitchen 3 control model trains before the Kitchen 3 freeze.
- Arash, 2026-10-20 ~16:15: the reading "practice-pool practice is not a TEST evaluation under A38 item 4" is NOT adopted. Kitchen 2 practice and rehearsal therefore also start only after the COCO run has finished and the COCO-start checkpoint is appended to the FloorPlan15 perception freeze (after which sim/kitchen_v3/k23/k23_coco_meta.py rewrites the record metadata, verifying that exactly that one line was appended). The append preserves the FloorPlan15 applicability evidence and the frozen sample: they stay bound to the predecessor record 4a139f55...; the successor record is the predecessor plus one line (the COCO-start checkpoint, which the agent never uses), every previously listed file keeps its hash, and neither the eligibility evidence nor the sample is rewritten. Both record identities (predecessor and successor) are named in the FloorPlan15 agent freeze. Disclosed: the FloorPlan15 A29 applicability run (A29.1 item 8) ran while the record was coco_pending; it used reference plans only, no planner and no learned memory.
- Resulting order: FloorPlan17 sample -> FloorPlan17 practice -> FloorPlan17 rehearsal -> FloorPlan15 COCO run -> --add-coco -> FloorPlan15 practice -> rehearsal -> size decision + agent freeze -> FloorPlan15 test -> after-fp15 decision -> FloorPlan17 agent freeze -> FloorPlan17 test (A29 item 1: FloorPlan15 is tested before FloorPlan17).

#### A39.3 Co-run of Kitchen 2 / 3 training with Kitchen 2 / 3 capture

Arash, 2026-10-18 ~18:50: "Yes, run both at once (Recommended)". A narrow, revocable exemption (record sim/kitchen_v3/k23/k23_corun.json, ddeb03b9..., until 2026-10-25 23:59; Astra K23-CORUN-r1 / r2): a Kitchen 2 / 3 training job may run while Kitchen 2 / 3 capture batches run; Gemma inactive throughout; Kitchen 1 returning stops training too; capture thermal closure and the training heat / telemetry stops kept; budget 1024 + 512 MiB; any out-of-memory error ends the co-run. Nothing else is exempted (practice, rehearsal and test never run beside training).

#### A39.4 FloorPlan17 Sink structural exclusion (K3-SINK-1)

Arash, 2026-10-17 ~21:15: "Approve the FloorPlan17 Sink exception." (Astra K3-SINK-r1 option A). The registered FloorPlan17 class list stays 45 classes (registry sha aa959ba2..., A38 audit unchanged). A versioned, evidence-linked exception exempts exactly FloorPlan17 / Sink from classes_seen, coverage projections and the TRAIN / VAL / TEST type floors; fail-closed (any Sink label or Sink object id in a frame's segmentation voids it and stops); SinkBasin keeps its ordinary requirements; reported as "44/44 observable types; 1 structurally excluded (Sink)", Sink AP = N/A; Sink stays in the B5 detector ontology (detector_types_FloorPlan17.json, 45 types; the planner sees 44 after the D2 Knife filter). Evidence: no Sink id in any frame's segmentation colours; a 120-view simulator sweep (SinkBasin 17x, Sink 0x).

#### A39.5 The Kitchen 2 / 3 agent-side merge under route B

- The paired Kitchen 1 equivalence run (part B) is NOT_EQUIVALENT by the rule fixed before it: the successor's diagnostic capture was refused by the old Kitchen 1 freeze (105 of 110 blocking items), and the microwave audit.errors counter differs between runs (descriptively, the extra-send incidence at explicit close steps varies as much within the baseline: 17/34 vs 13/24, Fisher p = 0.79); all 80 compared outcomes and step sequences are equal.
- Arash, 2026-10-19 ~08:15: "Repeat diagnostic, then exception (Recommended)". The SC2 diagnostic (isolated copy) passed on attempt 2 (2026-10-19 13:14-13:16; verdict_sc2.json SC2-PASS; Astra K23-SC2-r1 "SC2 VALID WITH NOTES"; attempt 1 stopped on a staging bug, 0 planner calls). Arash's conditional engineering exception partb_exception.json (3d8e0417...) was accepted by Astra K23-ROUTE-B-r1 ("EXCEPTION ACCEPTED"). Merge 30cf1a8 (2026-10-19). Equivalence is not claimed; the microwave-counter uncertainty is accepted, not resolved.
- Consequences recorded: the Kitchen 1 manifest successor (sim/final_test/diag/k23/k1_manifest_successor.json, self-sha 737277db...; k1_unchanged PASS 5/5); the superseding FloorPlan15 perception freeze record 4a139f55... (only sim/honest_skills.py changed; Astra K23-FREEZE-SUPERSEDE-r1). Subsequent Kitchen 2 / 3 applicability and agent-freeze evidence binds the merged implementation.

#### A39.6 The A37 paid-planning hold for Kitchens 2 / 3

Every paid Kitchen 2 / 3 reservation or hold is refused (k23_ns.a37_hold_problems) until the reviewed alignment record docs/final-test/k23/a37_alignment.json (self-sha b8a4dc57...) holds: it binds the eight aligned test files, the eight exercised runtime files, the gate evidence of the commit that ran them and Astra's unconditional acceptance (K23-A37-ALIGN-completion-r2, "RULING: ACCEPTED"). Applied on the branch ef46d89 / 4e59248 / 08167e9 (Astra K23-A37-HOLD-r3), lifted on master 68de553 (2026-10-19). A change to any bound file re-arms the hold. Kitchen 1's A37 (synchronous API for the Kitchen 1 plan; docs/final-test/amendments/A37.md) is unchanged; approve-all-6 item 3 extends the synchronous API to Kitchens 2 / 3.

#### A39.7 Kitchen 2 / 3 entry points and the replication design as implemented

- Merged on master f5721cb (Astra K23-TOOLING-G1-r3 and K23-TOOLING-G3-r3, "RULING: ACCEPTED"); new files only (no byte of the A37-bound, treatment-pinned, perception-frozen or Kitchen 1 manifest files changed): k23_practice.py (practice-kN-v1 through the unchanged practice_k.py engine: order 0, K = 10; the kitchen's own briefing hints; live run refused unless both A29 samples, the final package, the perception freeze and the treatment pins hold), k23_run.py (rehearsal-kN-v1 / test-kN-v1 through the unchanged capture / plan / planfirst / execute / score modules), k23_size_decision.py (A39.1), k23_freeze_agent.py, k23_coco_meta.py, and the launchers under sim/final_test/k23_run/ (one sim lock, hang guard; a fresh simulator per 300 executions, approve-all-6 item 3).
- Design forced in code (A29 item 1): planning arms G0, G+M, O0, O+M (stages free + core0, synchronous Opus); P0 / P+M at execution; canonical world, run 1, r = 0, one execution; no FS, curve, relocation, REPEAT, Codex or /p replays. Test pause acknowledgements need a response, an outcome-blind attestation and a >= 20-character note (the test-v1 rule).
- Budget: Kitchen 2 + Kitchen 3 Opus <= 70 USD together (approve-all-6 item 4; namespaces 32 + 3 per kitchen), the study cap 180 USD, the A29 study-wide pre-submission check.
- **Agent freeze** (k23_freeze_agent.py -> docs/final-test/k23/freeze_<scene>.json, read on every test entry by k23_ns.replication_freeze_status and k23_entry.test_entry_problems): written only when every gate passes: treatment pins, A37 hold, perception (+ the FloorPlan15 COCO append), class registration (gate 12.12 against the pre-capture probe registry), both A29 samples, the roster (FloorPlan17 only after after-fp15), the final package, practice (10 order-0 attempts, the K = 10 snapshot with its leakage gate), the rehearsal (complete and sealed, no violation), bilingual retrieval 12.2, honesty 12.3, G-iso (live in the kitchen, n = 12), masking / leakage, the 2,000-token package maximum, the measured context fit, evaluator 12.5 (this kitchen's A29 applicability + the rehearsal P0 rows; the empty-plan control from Kitchen 1's bound 12.5 evidence), reset tolerances 12.4 / A14 (the validated, kitchen-independent tolerances; per-kitchen effect measured by every execution's state check, A36), spend 12.7, families 12.8, the served Gemma digest 12.9, Workflow-B 12.10, renderer 12.11, an empty test namespace 12.13, fb-1, this A29.1 text in the preregistration, the dataset, the budget and the Kitchen 2 / 3 selftests. Every Kitchen 1 freeze gate has a recorded disposition (current kitchen evidence, shared bound evidence, or not applicable: A12 superseded, X-iso, acceptance and the acceptance registration are Kitchen 1 only).
- Sign-off: Arash's standing pre-signature (approve-all-6 item 1) is recorded only when every gate passed, the generated freeze summary is unchanged, and Astra's answer is an unconditional "RULING: ACCEPTED" naming the kitchen and the record's self-sha; any gate failure waits for Arash.
- The perception freeze records list ~/thor-data files by WSL paths; on Windows (where planning runs) the K23 entry points verify the same files through the WSL share (k23_entry.perception_problems).

#### A39.8 Energy (C5, A24.2) in Kitchens 2 / 3

Arash, 2026-10-20 ~16:15 (the Kitchen 1 rule; Astra K23-TOOLING-G2): the Gemma energy meter is on for every Kitchen 2 / 3 test plan; execute and score of a test namespace refuse incomplete strict energy coverage unless Arash records a waiver (energy_waivers.jsonl beside the namespace's journals). Practice energy is not metered (as in Kitchen 1); Kitchen 1's approved estimate rule for its 10 unmeasured practice rows does not extend to Kitchens 2 / 3, so their practice energy is reported as unknown.
