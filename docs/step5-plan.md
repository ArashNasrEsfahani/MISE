# Step 5 — the memory database (frozen spec)

Status: **FROZEN v1 (A0-spec, 2026-10-10)**, Astra's spec review (14 points) folded in. Later changes are
*amendments* (section 13), dated and reasoned, never silent edits. Sources: `docs/research/step5-memory-sota.md`
(SOTA + "RAG method choice"), `docs/research/step5-astra-review.txt` (design review),
`docs/research/step5-astra-build-review.txt` (build review), `docs/step4-honest-agent.md`, `sim/objmem.py`,
`sim/honest.py`, `sim/honest_skills.py`. Astra on this spec: `sim/step5/_astra/A0-spec.txt`. Parameters:
`sim/step5/params.json`. Ledger: `docs/step5-progress.md`.

**What step 5 adds.** A per-task **working memory** (WM, wiped at task end) and a **long-term memory** (LTM,
`ltm.db`) that keeps only *useful* claims ("don't save every interaction, that's not how humans work" — Arash).
Every claim mutation has one compact evidence row and an authoritative `write_log` row. Search and Gemma read **one**
shared answer, `where_is(type)`. Procedures and lessons are retrieved deterministically (no LLM by default).

**Honesty statement (exact wording for the thesis):** *"Decisions use permitted observations. During practice, a
privileged terminal goal check is used, after the task, for three things only: admitting a procedure, counting a
failed procedure reuse, and labelling a plan failure (`plan_goal_unmet`)."* Step 4's declared exceptions (pose,
actuator bool, action-resolve pixel grounding, declared priors, FOV) still apply. Because the verdict reaches memory,
"no simulator information ever influences decisions" is **false** and must not be written. Experimental lesson
remedies (7.5) do influence decisions and are reported as such.

**Generalisation rule.** No FloorPlan1-specific algorithm. Every threshold lives in the registry with a physical
reason or a declared engineering choice; FP1-tuned values carry `fp1_tuned: true`; dependencies on FP1 artefacts are
listed in 5.3. FloorPlan1 is development data: no unbiased FP1 performance claim. Parameters freeze before any
held-out kitchen.

**Workflow split.** **A (offline)**: built and tested with Windows `python`, temp DBs, mocked Gemma / embedder /
tokenizer, no simulator, no WSL, no Ollama, no write to `sim/trace.db` / `sim/memory.db`, no edits to step-4 modules.
**B (live)**: adapter wiring, live validation, evaluation runs (section 12). Claims about association accuracy,
lesson benefit, token compliance and generalisation are reserved for B. Offline completion is never reported as
live validation.

---

## 1. Event contract (what working memory receives)

### 1.1 Envelope (every event)

```
{ "schema": "s5.ev/1", "ev": <int, strictly increasing per episode>, "kind": <kind>,
  "kitchen": <harness label>, "epoch": <int world epoch>, "frame": <coordinate-frame id>, "map_rev": <map hash>,
  "episode": <id>, "t": <agent primitive index>, ...kind fields }
```

- `kitchen`: the harness's label (e.g. "FloorPlan1"), a declared scope key, never parsed.
- `frame`: a **stable coordinate-frame id** (the agent's map origin/start pose, minted once per map); `map_rev`: the
  map file's content hash, which changes with exploration. Geometry is compared only within one `frame`.
- `group` (task group) is **evaluator-only**: it encodes gold task relationships and never reaches WM or LTM.
- **Episode clock `E`** = number of completed, eligible practice episodes in this LTM. Each such episode advances
  `E` by 1 inside its consolidation commit, **including episodes that produce no claim**; supports written by that
  commit are stamped with the new `E`, so they start at age 0. In checkpoint evaluation `E` is **frozen**
  (`E_frozen` from the manifest), so test order cannot change rankings. Wall-clock time never enters a weight.

### 1.2 Field producers (provenance, not just shape)

| producer | fields | rule |
|---|---|---|
| detector on RGB (step-3 stack) | `otype`, `state`, `conf`, `detector`, `bbox` | `state` is the StateEstimator word or `?`, never a metadata flag |
| own depth + own pose (back-projection, `objmem` math) | `xyz`, `xyz_ok`, `extent`, `range_m`, `frac`, `thirds`, `support` | computed by the adapter from the honest `Obs`; never from metadata positions |
| agent-minted counters | `track` (`tr:<n>`), `place` (`pl:<n>`), `view` (`vw:<n>`), `obs`, `ev`, `app` | minted by WM/adapter counters; regex-validated; **never derived from, hashed from or keyed by a simulator id** |
| actuator bool | `ok`, `refused` | `lastActionSuccess` only; the error text is dropped by the adapter |
| step-4 skill outcome codes | `cause`, `under`, `verify`, `verdict` (inspection) | structured codes; B must emit `under` directly, never parse error text |
| harness (declared) | `kitchen`, `episode`, `mode`, `world`, `reset_world`, `admission_verdict` | task context and the single privileged verdict |

Adapter selftests (A, with a fake honest agent) feed forbidden inputs (sim ids, metadata xyz, oracle state) and
must see them rejected or never read.

### 1.3 Kinds and admissible fields (closed schema: unknown keys rejected, types checked)

| kind | fields |
|---|---|
| `episode_start` | `mode` (practice / test / probe), `world` (reset / persistent) |
| `task_context` | `level` (L1/L2/L3), `lang` (en/fa), `text` (the instruction as given to the agent), `concepts` (computed by `ltm_retrieve.concepts(text)`, never from the gold plan) |
| `step_start` | `i`, `verb`, `otype`, `to`, `pre` (precondition snapshot: holding, target located y/n, container state seen) |
| `attempt` | `i`, `k` (0 = first try, 1..3 recoveries), `recovery` (fresh_point / sidestep / step_back / lesson / procedure), `source` (default / `lesson:<id>` / `proc:<id>`), `ok`, `cause` |
| `application` | `app`, `record` (procedure or lesson id), `i`, `k`, `pre` (snapshot at application), `retrieval` (retrieval_log id) |
| `sighting` | `otype`, `state`, `conf`, `detector`, `track`, `xyz`, `xyz_ok`, `view`, `obs`, `in_rtype` + `in_basis` (inside_inspection / depth_consistent / box_only / none), `in_xyz` |
| `receptacle_seen` | `rtype`, `xyz`, `extent`, `view`, `obs`, `n_boxes_same_type` (in this frame), `covers` (other receptacle sightings this one box spans) |
| `inspection` | `rtype`, `xyz`, `open_ev` (ev of the contemporaneous open), `verdict` (found / visible / partly / not-visible / unsure), `frac`, `thirds`, `support`, `range_m`, `views`, `crouched`, `detected` (types), `targets` (types searched), `detector_rev` |
| `open` / `close` | `rtype`, `xyz`, `ok`, `by_me`, `state_seen` (open / closed / ?) |
| `pickup` | `track`, `otype`, `ok` |
| `put` | `track`, `otype`, `rtype`, `xyz`, `ok`, `resighted` |
| `drop` | `track`, `otype`, `xyz`, `ok`, `resighted` |
| `refusal` | `verb`, `otype`, `rtype`, `i`, `k`, `cause` — no error text |
| `contradiction` | `track`, `otype`, `views` (≥ 2 distinct; the objmem miss rule), `src` (the ev ids it used) |
| `step_outcome` | `i`, `ok`, `cause`, `under`, `retries`, `prims` |
| `terminal` | `exec_ok`, `agent_verdict` (success / failure / unknown), `prims`, `opens`, `walk_m`, `invalid` |
| `admission_verdict` | `goal_ok`, `tag: "privileged-terminal-goal-check"` — practice only, after `terminal` |
| `reset_world` | `reason` |
| `episode_end` | — |

`invalid` is an **instrumentation-integrity** flag (step-4 audit: violation, oracle or map-build read), never task
correctness. An invalid episode consolidates nothing. Places are resolved by WM (`ltm_identity`) from
`rtype + xyz + extent`; the in-receptacle *basis* is certified by the adapter from step-4 outputs (inspection result,
depth test, `place_of`).

### 1.4 Forbidden (rejected by `ltm_events.validate`)

Simulator `objectId`s and anything matching `[A-Za-z]+\|[-0-9.]+\|`, `_physics`, or a metadata key name;
`audit.errors_logged`, `audit.counts`; any error or event text (`detail`, `why`, step-4 `events` strings,
`r["err"]`); oracle states, goal predicates and gold plans, `state_readers` output, scene object lists,
`reachable_positions`, metadata positions, visibility flags, metadata receptacle contents, `group`. Strings are
allowed only in enumerated fields (closed class list, closed action list, enum words, minted ids) and in
`task_context.text`. Gemma's lesson hypothesis prose (7.4) is stored only in the audit table, scanned with the same
patterns, and **never rendered into retrieval or prompts**; only its validated structured action is usable.

### 1.5 Derived identifiers and coverage

- **Configuration unit** `U = H(kitchen, epoch, otype, canonical place)`. One discovery / search trial per `U`
  (repeated frames, revisits, unchanged re-sightings in the same epoch map to the same `U`).
- **Inspection coverage** per (place, type) = `qualified` iff: `verdict == visible` (see-in fraction ≥ `vis_ok` over
  ≥ `min_thirds` thirds), `support ≥ inspect.min_support`, the open is **contemporaneous** (`open_ev` is an `open.ok`
  or `state_seen == open` in this episode with no `close` of that place between it and the inspection),
  `range_m ≤ max_range_m`, and the type is capable for **this detector revision** (`detector_capability[detector_rev]
  [otype].recall ≥ min_type_recall`; unknown → not capable). `partly` → `partial`; else `none`.
- **Placement**: `put.ok` / `drop.ok` → placement; `verified` iff `resighted`, else `actuator`.

### 1.6 Reset boundaries

`reset_world` (harness-signalled, declared) increments `epoch`. A persistent world has no reset between episodes.
Each task starts a fresh WM; the walked map persists within a sequence and is restored per independent test (4).

### 1.7 Replay rules

Offline batches (copied, hashed step-4 JSONL via `ltm_events.from_step4_row`, which whitelists) are sorted by
(episode, ev) and validated. Identical duplicates are ignored; **conflicting duplicates fail**; an episode without
`terminal` + `episode_end` is incomplete and makes **no consolidation commit**. Missing observations are *unknown*,
never inferred. Replay validates transformations only, not association accuracy, coverage or usefulness.

---

## 2. Consolidation eligibility (`ltm_consolidate.py`)

Runs once at `episode_end` for complete, valid practice episodes, processing WM events **in chronological order**.
One SQLite transaction commits: the **episode receipt** (episode id unique → re-commit is a no-op), the clock
increment, claims, evidence and write_log. Test/probe episodes never write LTM. Each rule id is cited in its rows.

- **R1 useful types.** Claims only for the *useful-type list*: a **predeclared public ontology** — the AI2-THOR
  pickupable / movable object types (public documentation) ∩ the kitchen detector's declared class list, in
  `sim/step5/lexicon/useful_types.json` (sourced, hashed; **not** derived from evaluation plans). Plus this episode's
  targets. Everything else dies with WM.
- **R2 location history per track.** Chronological: a sighting sets the track's location; a successful pickup
  **supersedes** it (the old location is not retained as current); a `put` sets a placement (verified / actuator);
  a `drop` without re-detection → uncertain placement (actuator); a re-sighting refreshes. At episode end, one
  location or placement claim per track from the final state, if `xyz_ok` and `conf ≥ consolidate.min_conf`. Held
  tracks produce none.
- **R3 natural discovery (independent of R2).** The first sighting of a track **before any agent intervention on it**
  (natural provenance, `in_basis ∈ {inside_inspection, depth_consistent}`, a resolved canonical place) records the
  search trial for `U` with outcome *positive* — even if the object was later picked up. Once per `U`, ever
  (idempotent table).
- **R3a self-placement provenance.** `agent_placed` covers put, drop, and **ambiguous reacquisition** (a sighting of
  type T within `identity.placed_reacquire_m` of a place where the agent put or dropped a T in this epoch). Such
  sightings never create a natural trial. A `reset_world` breaks the provenance. Places where the agent removed or
  placed a T this epoch are **agent-disturbed** for T: no natural trial for T there this epoch.
- **R4 qualified negative.** A qualified inspection of a non-disturbed place, for a searched type, records the trial
  for `U` with outcome *qualified-negative* if `U` has no positive outcome; it is retained cross-task as *"<T> not
  detected in <place> after a qualified inspection (epoch e)"* — never "<place> is empty". A later positive sighting
  or placement of T there **supersedes** it (outcome becomes positive; the negative row is marked superseded).
  `partial` / `none` stay in WM only, as *deferred* (visit later), not ruled out.
- **R5 placement claims** go to `placements` only; they never touch habits. A verified placement re-sighted becomes
  a location claim with `agent_placed` provenance (retrievable, not a habit).
- **R6 contradiction.** A `contradiction` event, or a qualified negative at a place holding a current location claim
  of T, adds **one** contradiction per claim per episode (deduplicated by shared source ev ids).
- **R7 procedure admission.** Only if `admission_verdict.goal_ok` **and** `terminal.exec_ok` (every step ok)
  **and** not `invalid`. Goal met with a failed step (e.g. the 18 knife-less slices) → no procedure;
  `stats.goal_without_plan += 1`.
- **R8 lessons**: section 7.
- **R9 evidence.** One compact evidence row **per claim mutation** (create, support, contradict, supersede, retire,
  reactivate, merge, unmerge): `{claim, rule, mutation, episode, epoch, E, summary (enumerated fields),
  src_ev (hashes), conf, consolidation_version}`.

Never retained: frames, per-frame sightings, poses beyond place geometry, refusal or error text, map cells.

## 3. Identity (`ltm_identity.py`)

- Every claim carries `kitchen`, `epoch`, `frame`.
- **Place association** for a `receptacle_seen` of type R with centre c: candidates = places of type R within
  `place_gate_m[R]`. **No candidate** → create a new place. **One candidate**, or the best is nearer than the second
  by ≥ `ambig_margin_m` → associate. **Otherwise abstain** (`place = null`, candidates kept in WM); abstained
  sightings neither create places nor count trials.
- **Regions** (registry `region_types`): own sightings of the surface type whose grounded points lie within
  `region_link_m` of each other and within `region_height_m` in height form one connected region; an object claim on
  a region also stores its position.
- **Merge needs positive duplicate evidence**: two same-type places merge only if within `merge_gate_m` **and** a
  single detected box in one frame covered both centres (`covers`), and they were never seen as separate boxes in
  one frame (`n_boxes_same_type ≥ 2` with each centre in a different box) nor opened as separate fronts. Missing
  evidence keeps them separate. Adjacent identical drawers therefore stay separate unless one box ever spanned both.
- **Aliases**: a merge writes `alias(place → canonical)`; claims, trials, negatives and counts keep their original
  place id and resolve to canonical **at read time** (distinct canonical `U` → no double counting). Unmerge deletes
  the alias row (write_logged); all evidence survives.
- **Object instances** are agent tracks inside one epoch; LTM stores types at places, not object identity across
  epochs. Simulator ids never appear.
- **Frame change** (new `frame`): old-frame geometric claims become `maybe` until re-observed and re-associated.

## 4. reset_world, clear_memory, snapshot / restore

| operation | locations | negatives | placements | habits / lessons / procedures | clock |
|---|---|---|---|---|---|
| `reset_world` (practice) | read as `maybe` from the next epoch (via `n_resets`, 5.1) | earlier-epoch negatives not retrieved (kept) | earlier-epoch → `historical` (not current, provenance broken) | unchanged; geometry-dependent lessons flagged `revalidate` | unchanged |
| persistent world | stay current, decay with `E` | decay with `H_negative` | stay current | unchanged | +1 per practice episode |
| `clear_memory` | new empty `ltm.db` at the injected path (arm `none`) | | | | 0 |
| `snapshot(dir)` | SQLite backup copy + **manifest**: sha256 of a canonical dump, `E_frozen`, params sha, lexicon shas, map file copy + `frame`/`map_rev`, detector revision, code hashes, retrieval channel config | | | | frozen |
| test from snapshot | DB opened **read-only** (`mode=ro`); reset/epoch interpretation in an in-RAM **episode overlay**; retrieval / eligibility / application / diagnosis logs go to a separate per-test `testlog.db`; map + mutable agent state restored per independent test | | | | frozen |

Hash equality of the snapshot before and after every test is a gate. Persistent-world evaluation restores **once
per sequence** and advances WM, map and world consistently within it. Retirement is **computed on read**, never
written during tests. `ltm.db` paths are always injected (per run / block / arm); journal mode `DELETE` (never WAL)
on any `/mnt/...` or DrvFs path. Legacy `sim/memory.db` is never imported. The tracer is injectable (`None`
default) and mirrored **after** commit with the write_log id as idempotency key; delivery is at-least-once, and the
mirror checks the destination for the key before emitting (exactly-once needs that destination check; the local
`trace_exported` flag alone is not enough).

## 5. Decay, contradiction, tiers, registry

### 5.1 Weight

```
w = w0[kind, verification] * 2^(-(E_now - E_support) / H[kind]) * gamma^c * rho^n_resets
```

`E_support` = `E` stamped on the last *observational* support (new sightings, trials, applications with an
observed outcome; retrieval alone never refreshes); `c` = qualified contradictions since that support;
`n_resets = epoch_now - epoch_of_support` (applied once, locations and placements only). `w < retire_below` →
retired from retrieval (computed on read; history kept; a new support reactivates). Nothing is deleted.

### 5.2 Tiers (rank by tier, then weight)

| tier | contents |
|---|---|
| T0 | current-epoch location claims, `verified` (≥ 2 distinct views, or `in_basis` inside_inspection / depth_consistent) before `trusted` (one view), `c = 0` |
| T1 | current-epoch placements: verified, then actuator-only |
| T2 | `maybe` locations (earlier epochs, other frame) and current claims with `c = 1` (downgrade) |
| T3 | kitchen habits by `pref` |
| T3b | claims with `c ≥ 2` ("doubted") |
| T4 | fixed declared prior (`search.prior_for`) |
| EXPLORE | always present; lists qualified-suppressed places last, so suppression is overridable |

Recent verified evidence therefore always outranks a habit, and contradictions demote by tier, not only by weight.

### 5.3 Registry and FP1 dependencies

All values, sweep grids, units, reasons and FP1 flags: `sim/step5/params.json`; code reads only from it; a gate
asserts every leaf has `value`, `grid` or `range`, `reason`, `fp1_tuned`. Half-lives (practice episodes): location 3,
placement 2, negative 1, habit 30, lesson 20, procedure 50 — **declared defaults, not validated constants**; the
robustness table sweeps each over its grid on dev data only. FP1 dependencies (`params.json → fp1_dependencies`):
the step-3 detector and its capability table, `consolidate.min_conf`, `inspect.vis_ok` / `min_support` (from step
4), `objmem` geometry constants, the detector class list. Capability tables are versioned per detector revision; the
report states which kitchens each detector was trained on.

## 6. Habits (empirical search preferences)

Search trials (R3/R4): one per configuration unit `U` with outcome positive **or** qualified-negative (positive
supersedes), excluding agent-disturbed places. Kitchen habit for `(kitchen, otype, rtype)` aggregates trials over all
canonical places of type R; `(kitchen, otype, place)` uses one place:
`pref = (P + a * p0) / (P + N + a)`, with `P`, `N` the decayed positive / negative trial counts (each decays from the
`E` of its commit with `H_habit`), `a = habits.prior_strength`, `p0` = the `search.prior_for(otype)` weight for R
normalised to sum 1 over its listed receptacle types (an unlisted type gets `habits.p0_unlisted`). A
policy-dependent **empirical search preference**, not a probability. **Kitchen level: ACTIVE.**

General habit `(otype, rtype)`: built and synthetically tested, **INACTIVE** (`habits.general_active: false`).
Promotion is only *evaluated*: ≥ `min_kitchens` kitchens each with ≥ `min_discoveries_per_kitchen` positive
trials; equal weight per kitchen (mean of kitchen `pref`s); disagreement kept (`spread`). Activation also needs a
pre-registered validation on a third kitchen excluded from construction and tuning (paired steps-to-find with the
general prior < fixed prior). With one kitchen the report states **0 eligible general habits**.

## 7. Procedures, failure diagnosis and lessons

### 7.1 Procedures
A procedure has an **intent key** (the `task_context.concepts` of the instruction it solved, 8.2) and a **body** (the
executed sequence of verbs + types + destination types, repairs included, with per-step observable preconditions:
holding / hand empty, target located, container state seen). Status: candidate → trusted → demoted → retired.
Admission R7 (privileged use 1). **Trusted** after `procedures.trust_after` successes in distinct episode units
(distinct episode and distinct task item or epoch; independence is not claimed statistically). **Failed reuse**
(privileged use 2): an `application` with preconditions holding, then a failed step in the reused part or
`goal_ok == false` → `failed_reuses += 1`; `failed_reuses ≥ success_uses` → demoted (suggestion only);
`failed_reuses ≥ success_uses + procedures.retire_margin` → retired. A procedure whose preconditions do not hold now
is not offered.

### 7.2 Failure diagnosis (after every failed step and failed episode; tests store nothing)
1. deterministic cause from `step_outcome.cause` / `attempt.cause` (+ `under`) through the taxonomy;
2. evidence packet = numbered WM event ids of that step (enumerated fields only);
3. optional Gemma hypothesis (7.4); 4. lesson create / recurrence / no-op (7.5). Not every failure yields a lesson.

### 7.3 Cause taxonomy (observable, fixed)

| code | step-4 source | meaning |
|---|---|---|
| `not_detected` | not_detected | target not seen within budget (≠ absent) |
| `action_refused` | action_refused | actuator bool false |
| `unreachable` | unreachable | no own-map route / out of reach |
| `wrong_target` | wrong_target | the point acted on something else |
| `postcondition_unconfirmed` | state_unrecognised | result could not be confirmed |
| `budget_exhausted` | retry_exhausted / primitive cap (with `under`) | retries or primitives spent |
| `plan_goal_unmet` | all steps ok, `goal_ok == false` | plan failure (privileged use 3, practice only) |
| `unknown` | anything else | — |

### 7.4 Gemma hypothesis contract (`Hypothesizer.propose(packet) -> dict`, mockable)
Output `{"cause": code, "hypothesis": ≤ hypothesis_max_chars, "evidence": [ids], "action": one of try_other_instance
| change_pose | approach_closer | crouch | open_first | search_elsewhere | none, "alternative": optional action}` or
`{"insufficient_evidence": true}`. **Rejected** (logged, no lesson) if an evidence id is not in the packet, the cause
contradicts a known deterministic cause, the action is outside the list, or the prose trips 1.4. The prose is audit
only (1.4). Without Gemma: `lessons.default_action[cause]`.

### 7.5 Lessons: lifecycle and the promotion-deadlock resolution
Record: `conditions` = structured **local failure key** (cause, verb, otype, rtype, state seen), `action`, evidence
refs, status, `helpful_units`, `unhelpful_units`, `recurrences`, origin episode.

- **Creation**: deterministic cause ≠ unknown and a valid action (hypothesis or default ≠ none) and no lesson with
  the same conditions; otherwise a recurrence on the existing lesson (evidence of the problem, not of the remedy).
- **Experimental remedies (resolves the deadlock).** In **practice** episodes only (enforced in code by `mode`),
  at a recovery attempt whose local failure key matches a non-retired lesson and whose action maps to an available
  recovery, the attempt uses that remedy (`attempt.recovery = lesson`), inside the existing 3-retry budget (no extra
  budget, no prohibition). Selection is frozen: supported before tentative, then older `E` of creation, then id; at
  most **one lesson remedy per recovery attempt**. These are labelled *experimental recovery choices*; they do
  influence decisions. In tests, only supported lessons apply, and nothing is counted back.
- **Scoring the application** (the first eligible application per lesson per episode is the scored one), with this
  precedence: episode aborted or the outcome unobservable → **neutral**; the same cause recurs in the step after the
  application → **unhelpful**; otherwise the step ends ok → **helpful**; the step fails with a different cause →
  **neutral**; a later different remedy produced the success → **neutral**.
- **Units**: helpful / unhelpful are counted in distinct episode units (distinct episode, and distinct task item or
  epoch, excluding the origin episode). Distinct units prevent duplicate credit; they do not establish statistical
  independence. Results show *recovery association*, not causal remedy efficacy.
- **Transitions**: tentative → supported iff `helpful_units ≥ promote_after` and `helpful_units > unhelpful_units`;
  supported → contradicted iff `unhelpful_units ≥ helpful_units` (not used in tests; still a practice experiment);
  contradicted → supported iff it regains `helpful_units > unhelpful_units`; tentative or contradicted → retired iff
  `unhelpful_units - helpful_units ≥ retire_margin`; `revalidate` (after reset or frame change for geometry lessons)
  → treated as tentative until one helpful unit. Lessons never become prohibitions and never create habit counts.

### 7.6 False-lesson rate
Outcomes per audited lesson: **false** (cause wrong, or the remedy does not address the cause), **correct**, or
**insufficient evidence** (reported separately). "Remedy not demonstrated effective" is not "false".
- **Supported-set FLR** = false / all lessons supported at the checkpoint (all audited). Wilson 95 % interval.
- **All-created FLR** = false / all lessons ever created (any status). Audit every lesson if ≤ `lessons.audit_cap`;
  otherwise stratify by status with known inclusion probabilities, a Horvitz–Thompson estimate and a stratified
  bootstrap interval (no plain Wilson on a weighted estimate).
The adjudicator (Arash, blind to arm) sees the evidence packet plus a privileged offline replay summary; audited
lesson versions are hashed and frozen; adjudication never feeds learning. A separate small labelled failure audit
measures Gemma hypothesis accuracy.

## 8. Retrieval (deterministic, no LLM by default)

### 8.1 `where_is(otype, wm) -> [Candidate] (≤ where_k) + EXPLORE`
Keyed SQL by tier (5.2). WM masks: places inspected this task (`qualified` suppressed to EXPLORE; `partial` /
`none` deferred to the end of their tier). Candidate: `id (place or rtype), tier, source, age, support,
contradictions, status, weight, provenance`. Search walks the list in order. Gemma receives the same list rendered as
≤ `where_k` short lines and may **reorder only within T2–T4** (it cannot lift a habit above current evidence, add a
place, or route); the adapter passes Gemma's final ordered candidate ids to search. **Verify before act**: a
candidate is a navigation goal; manipulation needs a current sighting.

### 8.2 Concepts (no LLM)
`concepts(text)` → ordered **action roles** `[(verb, otype, destination, state, negated)]` in instruction order,
with `?` for unknown fields. Types via the closed type lexicon (EN canonical + FA aliases, constrained to the class
list, `agentic_rag.normalize_text` normaliser), verbs via the action lexicon, state words (on/off, open/closed,
sliced, clean/dirty, hot/cold, filled/empty), destination (to-argument), negation scoped to the role it modifies (EN
not / don't / without / never; FA نـ- verb prefix, نه, بدون, هرگز). Record key text (for lexical / dense) is rendered
from the structured roles by fixed EN and FA templates — never by an LLM, never copied from free text. Lexicons:
`sim/step5/lexicon/*.json`, sourced and hashed; dataset-fitted entries flagged and disclosed.

### 8.3 Procedures and lessons: channels, RRF, steering gate
Two query kinds: **intent queries** (instruction → procedures) and **local failure queries** (failure key at a
recovery attempt → lessons; enables cross-task remedy transfer). Lexical: normalised token overlap on key text.
Dense: bge-m3 on the same text (8.5). RRF with `rrf_k`. Then the **steering gate dominates RRF**. A procedure may
steer only if: its intent roles equal the query's roles exactly on every known field (verb, type, destination,
state, negation; `?` never matches); same kitchen scope; status trusted; no unresolved contradiction; body
preconditions hold in current WM. A lesson applies only through 7.5 (exact local failure key, status rules,
practice-only for tentative, enforced in code). Order = gate-pass by RRF, then labelled suggestions (candidate
procedures whose preconditions hold) by RRF. Reranker: ablation only. The report gives gate-pass and application
coverage per arm, so an arm with zero uses is not read as failed memory.

### 8.4 Budgets
One memory block per prompt, budget in tokens per level (`retrieval.budget_tokens`); `where_is` lines take up to
`retrieval.where_share` of it first, procedures/lessons get the rest. Whole records only; a record's qualifiers
(status, support, age, "tentative", "not detected after a qualified inspection") are atomic with it. Counting uses an
injected Gemma tokenizer (`count_tokens`); without it the declared estimate `ceil(latin/4 + persian*1.66/4)` is
used and **no token-compliance claim is made** (B must measure with the real tokenizer).

### 8.5 bge-m3 — separate acceptance gate (G-dense)
The dense channel is an injectable `embed(texts) -> vectors`; A tests a mock only. G-dense (B): isolated environment
(never `~/thor-env`), CPU forced in that process, threads and `max_length` capped, pinned package versions and model
revision, embedding cache keyed by (key-text hash, model revision) and invalidated on change, query latency p50/p95
and peak RAM measured; accepted iff hybrid recall ≥ lexical recall on a bilingual key-matching set. Until accepted,
runs declare `channels: ["lexical"]`; lexical-only and hybrid runs are never pooled.

### 8.6 Logs (separate tables; in tests, in `testlog.db`)
`retrieval_log` (query, roles, candidates, ranks, scores), `eligibility_log` (gate decision and the failed
condition), `application_log` (record used, step, attempt, outcome class). Retrieval earns no credit; only an
application with an observed outcome does.

## 9. Modules, files, APIs, adapter

New files only (A): `sim/ltm_events.py` (schema, producers, `validate`, `from_step4_row`, replay sort/dedupe),
`sim/ltm_wm.py` (WorkingMemory: chronological ingest, units, coverage, masks), `sim/ltm_identity.py` (association,
abstain, regions, merge/alias), `sim/ltm_store.py` (schema, migrations, transactions, receipts, write_log, snapshot
manifest, read-only open, journal mode), `sim/ltm_consolidate.py` (R1–R9), `sim/ltm_decay.py` (weights, tiers;
pure), `sim/ltm_habits.py`, `sim/ltm_lessons.py` (taxonomy, Hypothesizer, procedures, lessons, FLR),
`sim/ltm_retrieve.py` (where_is, concepts, lexical, dense interface, RRF, gate, budgets, render), `sim/ltm.py`
(facade), `sim/ltm_adapter.py` (B interface + fake honest agent; no step-4 import at module load), `sim/ltm_eval.py`
(protocol runner, synthetic world first). Data: `sim/step5/params.json`, `sim/step5/lexicon/`, `sim/step5/_astra/`,
`sim/step5/_checkpoints/`.

```python
ltm = LongTermMemory(db_path, kitchen, params=load_params(), tracer=None, embedder=None, count_tokens=None)
wm  = ltm.begin_episode(episode, mode, world)          # -> WorkingMemory (fresh)
wm.ingest(event)                                       # validated; forbidden fields rejected
ltm.where_is(otype, wm)                                # -> [Candidate] + EXPLORE
ltm.memory_block(wm, level, lang)                      # -> rendered lines + record ids (budgeted)
ltm.lesson_for(wm, failure_key)                        # -> lesson remedy or None (7.5 rules, mode-checked)
ltm.end_episode(wm, admission=None)                    # one txn; returns receipt + write_log ids
ltm.reset_world(reason); ltm.snapshot(dir); LongTermMemory.open_frozen(dir, testlog_path)
```

Adapter (finalised in B; A ships the interface and a fake-honest-agent implementation): `HonestEventAdapter(ltm,
wm)` with `on_task(text, level, lang)`, `on_step_start`, `on_attempt`, `on_observation(obs, om_result)`,
`on_receptacle`, `on_inspection(rec)`, `on_action(verb, ok, ...)`, `on_step_end(step_result)`,
`on_terminal(row)`, `on_admission(goal_ok)`; `LtmChooser(ltm, wm)` implementing step 4's
`chooser(target, type_list, memory_summary) -> ordered place types` plus instance hints; `memory_block` for
`step4_gemma`'s prompt; `lesson_for` as a recovery-choice hook.

## 10. Evaluation protocol (runner built in A, runs in B)

- **Arms (cumulative)**: `none` (no cross-task memory: same WM, map, detector, fixed priors, budgets) / `places` /
  `places+habits` / `+lessons&plans`; a plans-only arm only if a separate lesson claim is made. Reference-plan runs
  (fixed executable plans: search/execution effects) and Gemma-planned runs are reported separately. The online
  loop is separate from teach-then-freeze.
- **Worlds**: reset (`InitialRandomSpawn`, matched seeds across arms) and persistent (sequences from matched initial
  states and task orders, analysed as whole sequences).
- **Blocks**: 5 paired blocks (practice order × world seed), 10 if affordable; all 5 block differences reported;
  bootstrap over blocks / whole sequences, never over correlated episodes.
- **Splits**: task group = union-find over items sharing a goal signature (types + predicates) or a plan concept
  key; EN/FA of an item together; component sizes and a frozen split manifest published before any run; grouping
  predicates evaluator-only. Two labelled protocols: **P-familiar** (held-out episodes / worlds of practised task
  concepts — where procedure reuse can apply) and **P-composition** (group-held-out). Lesson transfer across groups
  is tested through local failure queries.
- **Learning curve**: K ∈ {0, 1, 3, 5, 10}; checkpoints are frozen snapshots forked from training; each test opens a
  fresh read-only copy; frozen-hash equality asserted.
- **Metrics**: SR / PC / TDS per level and language; steps-to-find, walking (m), opens, stale visits (a retrieved
  place visited where the target was not), retries, consolidation latency, memory size (rows, bytes), FLR,
  gate-pass and application coverage; costs reported alongside failures.
- **Minimal core first (B3)**: WM + positive place evidence + exact-key `where_is` + audited consolidation + frozen
  snapshots; **none vs places at K = 0 and 1**, fixed executable plans, reset world; SR, primitives-to-find, opens,
  stale visits. Habits, dense, diagnosis, procedures and the console panel wait until this works.

## 11. Test plan (offline gates, Windows `python` from the repo root)

`sim/_step5_<topic>_selftest.py`, each printing `N/N`: `events`, `store`, `identity`, `consolidate`, `decay`,
`habits`, `lessons`, `retrieve`, `adapter`, `adversarial`, `isolation`, `eval`, `params`.
**Adversarial gates**: reordered complete batch → same DB; identical duplicate → no-op; conflicting duplicate →
fail; truncated episode → no commit; crash between commit and trace export → no loss, one export on retry (mock
destination dedupe); migration + restart; adjacent identical drawers never merged without spanning-box evidence;
erroneous merge reversible with no double counts; moved object → contradiction, tier downgrade, new claim;
pickup supersedes location; unseen put → actuator placement; unqualified or non-contemporaneous negative never
stored cross-task; repeated-discovery inflation (100 frames, 5 revisits = 1 trial); repeated failed inspections = 1
trial; self-placement and ambiguous reacquisition never move a natural habit; stale embeddings invalidated on
model/key change; Persian orthographic variants (ي/ی, ك/ک, ZWNJ, diacritics) keep the key, negation (برندار vs
بردار) changes it; token overflow drops whole records, never qualifiers; unsupported hypothesis references
rejected; hypothesis prose never appears in a rendered block; failed procedure preconditions → not offered;
tentative lesson never applied in test mode; privileged-field contamination (objectId, metadata xyz, oracle
state, errors_logged, error text, goal predicates, `group`) rejected at ingest; frozen-state hash equality before /
after testing; every `ltm*` import starts no controller and opens no socket (`ai2thor`, `socket`, `urllib`,
`requests` monkeypatched to raise).

## 12. Workflow B outline (live, after A)

B0 re-read step-4's final state (after its bug-fix workflow lands); Astra check. B1 adapter wiring into
`honest_skills` (events, structured `under`, `LtmChooser`, recovery hook) and `step4_gemma` (memory block) behind a
flag defaulting off; step-4 gates stay green. B2 live smoke (FloorPlan1, a few items, temp `ltm.db`, 0 violations).
B3 minimal core (none vs places, K = 0/1, fixed plans, reset world, 5 blocks). B4 G-dense acceptance. B5 habits +
decay, persistent world. B6 diagnosis + lessons + procedures; Gemma runs; real-tokenizer budget check. B7 console
panel (API bump). B8 held-out kitchens (FloorPlan6/8/15/17) with frozen parameters; general-habit promotion check.
20-minute updates on long runs; fail-fast invariants per episode; Arash's go before each section.

## 13. Amendments
(none yet — v1 already includes Astra's A0 spec review, `sim/step5/_astra/A0-spec.txt`)

**2026-10-10 (A6-eval, Astra `sim/step5/_astra/A6-eval.txt`)** — evaluation runner `sim/step5_eval.py`:
- K counts completed practice *attempts* (failures and inapplicable included); recorded exposure per checkpoint.
- Two labelled reset worlds: `reset_fresh` (test world seeds disjoint from practice; only structural knowledge can transfer) and
  `reset_shared` (layouts recur from a per-block pool; object-location transfer possible). Never pooled.
- P-familiar tests use item ids disjoint from practice; translation transfer (same item, other language) is the separate
  P-translation protocol. A giant task-group component makes P-composition *infeasible* (declared in the manifest), never repaired.
- Practice order places groups that have held-out familiar tests first (`familiar_first`, stable partition of the seeded order) so
  that small K can exercise concept reuse; practised-concept coverage per K is reported.
- Arm `none` keeps WM/map/sensing/priors and loses cross-episode memory; it does not learn, so its K=0 test is its baseline for
  every K (`none_at_every_K` adds a determinism control). Arm trajectories differ: results are the total effect of learning with memory.
- Persistent worlds are a separate online-adaptation experiment (fresh memory per sequence, continue after failure, unit = sequence).
- Bootstrap units are blocks (learning curve) or whole sequences (persistent); successful-only cost rows are descriptive.

**2026-10-10 (A7-review, Opus; Astra `sim/step5/_astra/A7-review.txt`)** — no design change, three tightenings:
- 1.3 "practice only, after `terminal`" is now enforced by `WorkingMemory.ingest`: `admission_verdict` before `terminal`, outside a
  practice episode or twice is refused, and after `terminal` only `admission_verdict` / `episode_end` are admitted (the verdict is
  structurally unavailable while the agent acts).
- The kitchen scope is always declared: `LongTermMemory(...)` has no default kitchen; `open_frozen` reads it from the manifest and
  refuses a mismatch.
- 5.3's registry gate now exists (`sim/_step5_params_selftest.py`); `procedures.retire_margin` / `lessons.retire_margin` got real
  reasons (values unchanged). params.json sha256[:16] e9dd82146ec3bd89.

**2026-10-10 (B0-plan, Opus; Astra `sim/step5/_astra/B0-plan.txt`) — the concrete workflow-B plan.** It replaces section 12's
B1-B3 outline and the kickoff checklist's command lines where they differ; B4-B8 keep their order. Astra's 12 answers are folded in
(1 adopted as proposed, 11 tightened, none rejected).

*B-a. Data (prereg gate 12.10).* B builds `step5_eval`'s split over `split_manifest.json → nontest` **minus** `dev_pilot` (104
items), so the 12 dev items stay free of memory for the cost pilot. Membership is asserted at run start **and on every resume**,
before any episode executes (refuse, not warn); no test-family item (incl. the 26 capped-out ones) may appear in any B journal. B2
smoke has its own allowlist (dev items allowed: debugging) and a disposable `ltm.db`; nothing from B2 is kept.

*B-b. Modules and flags (all new behaviour default OFF; step-4 gates unchanged; pre-change copies in
`sim/step5/_checkpoints/B1/`).*

| piece | where | flag / default |
|---|---|---|
| `HonestEventAdapter` | new `sim/ltm_adapter.py` (no step-4 import at load) | used only when `memory != "none"` |
| ObjectMemory proxy | in the adapter: forwards `new_episode / update / on_pickup / on_put / on_drop` to the real `objmem` and emits `sighting / receptacle_seen / pickup / put / drop` from objmem's report + Track xyz (own depth + own pose) | — |
| structured event hook | `HonestSkills(..., on_event=None)`: one call per open / close / inspection / search start + end, enumerated fields only (Astra: post-step lists lose transient events; `opened_by_me` forgets closed doors) | `None` = no call, code path unchanged |
| instance hints | `HonestSkills(..., place_hints=None)`: reorders instances **and** the exhaustive policy's openable fronts / stops (`_plan_items`) by distance to a hinted claim xyz associated to receptacle geometry; a hint is a navigation goal only, manipulation still needs a current sighting | `None` = unchanged |
| `LtmChooser(ltm, wm, channels)` | adapter | fallback `prior_chooser`; with empty memory its output is **identical** to `prior_chooser`'s |
| facade `LongTermMemory.memory_block(wm, level, lang, channels)` | `sim/ltm.py` | `where_is` over every type of `concepts(text)` under **one aggregate budget**; returns text + record ids / order / qualifiers |
| live executor `make_executor(memory, policy="console", hints, on_event)` | new `sim/step5_live.py`, mirrors the `step4_refplans` loop (refplans not edited) | `memory="none"` |
| `world_kind="canonical"`, protocol `P-location`, item allowlist | `sim/step5_eval.py` | off (A6 selftest unchanged) |

Event provenance: counters `tr:/pl:/vw:/ob:` minted by the adapter (objmem track ids mapped, never copied); `in_basis` from
`place_of` is `box_only` (stronger bases only with observation evidence: inspection result / depth test); `step_outcome.under` is a
structured code or `unknown`, never parsed from `why`; attempt-level (per-k recovery) events are not emitted in the core.
Forbidden-source tests (offline, `sim/_step5_adapter_selftest.py`): a fake Obs / report / step result carrying objectIds, metadata
positions, oracle state, error text and `group` yields none of it in the stream, and the stream validates end to end. Static guard:
the adapter, LtmChooser and the decision path of `step5_live` never name the controller, `metadata`, `last_event` or
`state_readers`; `h.scoring()` is allowed only in the executor's scoring block (goal verdict → `admission` bool).

*B-c. One renderer (gate 12.11).* Memory reaches a planner only as `PlannerContract(memory_block=<facade memory_block text>)`,
`reorder_within` off; `step4_gemma.build_messages` / `memory_summary` and `GemmaPlanner.chooser` are never used for memory (static
guard). LtmChooser is a deterministic search input, not a prompt. In B3 (reference plans, no planner) the block is rendered and
logged per test for the budget and bilingual checks only.

*B-d. Minimal-core channels.* `channels={"places"}` = WM + positive location claims (T0 current epoch, T2 `maybe`) + audited
consolidation + exact-key `where_is` + frozen snapshots. The channel filter applies **before** candidate competition, suppression
and top-k truncation; only learned location candidates precede `prior_chooser`'s order. OFF: T3 habits, cross-task negatives
(detector_capability is empty anyway), T1 placements as candidates, merges (association runs, merging does not), dense, lessons,
procedures, attempt events, Gemma reordering. No `LessonsExtension` is passed, so reference-plan practice cannot admit a
procedure; a guard refuses any `ltm.db` holding an episode receipt that is not a B episode id (no replay seed). B memory is never
S\*: S\* needs fresh Gemma practice (final-test run order 3).

*B-e. World and epochs.* The honest executor resets to the **canonical** FloorPlan1 world + the item's setup (as step 4 and the
final test do); no InitialRandomSpawn (random-spawn worlds deferred, labelled). Every practice episode signals `reset_world`, so
older claims read as `maybe` and placements as historical. Tests run on a frozen snapshot through an **explicit epoch/frame
overlay**: each independent test starts at snapshot epoch + 1 (frozen `reset_world` raises today; the overlay lives in RAM), test
order never accumulates resets, and the map + mutable executor state are restored identically per test. Declared limitation: an
item setup can move an object, so practice evidence can be stale in a canonical test world; this is measured (stale visits), not
filtered.

*B-f. B2 live smoke + prerequisite trace (Astra A7).* WSL, one controller, temp `ltm.db` in the Linux home, 2-4 items, first
episodes deep-checked by hand. Required before any B3 scoring: one printed live chain — observation `ev` → location claim + its
`write_log` / evidence row → `where_is` candidate (T0/T2) → LtmChooser order / hint → the search's first visit → sighting — plus a
route diff showing the hint changed the actual route. Per-episode fail-fast: 0 audit violations, 0 oracle reads, every event
validates, writes only under the injected path, invalid episodes consolidate nothing, frozen hash unchanged in tests.

*B-g. B3 design (`--preset core`).* Arms `none` vs `places`; K ∈ {0, 1}; `world_kind=canonical`; protocol `P-location`: test items =
the step-5 split's test items (item ids disjoint from practice), stratified by level, **identical draws across arms and K**; 5
paired blocks, 8 test items each; each test item executed once in one seeded language, the same for every arm (fixed reference
plans execute identically in EN and FA; no bilingual-execution claim). Cells per block: none-K0, places-K0 (determinism control),
places-K1, plus 1 practice episode (~25 episodes, ~30-40 min; ~3 h in all; 20-minute updates). Executor policy = the console /
exhaustive policy of the step-4 final regression, recorded in the config hash.

*B-h. Metrics (instrumented at every internal search, composite skills included).* Per search: found / censored, the actual cap,
`steps_to_find` (primitives from search start to the target sighting; censored at the cap), visits split into memory-ranked vs
other, unreachable vs visited, **stale visits** (a memory-ranked place visited where the target was not detected). Per episode:
opens (agent opens, kept separate from search), walking m, primitives, SR (`goal_ok`, scoring only). Aggregation (predeclared):
episode → block mean per cell; effect = places-K1 − none-K0 per block; all 5 block differences + bootstrap labelled low-n;
reported whatever the sign. Coverage per test = fraction of its searched types with a **learned** T0/T2 candidate (T4 priors
excluded; gold-plan types used evaluator-side only).

*B-i. B3 acceptance (predeclared; decides whether `places` is an S\* channel).* Integrity: 0 violations / oracle reads / rejected
events, frozen hash equal before and after every test, `memory.db` / `trace.db` untouched, places-K0 == none-K0 on every paired
test (traces, opens, outcome). Mechanism: learned candidates traceable to their evidence; application coverage > 0 (at least one
test search whose route the memory changed); Arash hand-checks a sample of place associations; merges = 0. Practice episodes are
not required to yield claims, and zero-yield attempts are never replaced (K = started attempts). Bilingual check (gate 12.2 style,
offline): EN and FA render the same retained record identities, order and qualifiers, also after truncation. `places` is accepted
iff integrity + mechanism pass; the effect is reported, not required.

*B-j. Deferred (dormant flags; each later with its own acceptance, one at a time).* Real-tokenizer budget check — **before the
cost pilot / any S\* prompt-budget claim**, independent of B6 (Astra); B4 G-dense; B5 habits + decay sweeps + persistent world; B6
diagnosis / lessons / procedures / attempt events; B7 console panel (API 7 → 8); B8 held-out kitchens; random-spawn worlds;
Gemma-planned B runs. Known limits (Astra): started-attempt accounting across crashes must be journaled before an episode starts;
the 5 blocks reuse items, so blocks are not independent samples of the item population.

**2026-10-10 (B1-wire, Opus; Astra `sim/step5/_astra/B1-wire.txt`) — what B1 built and where it differs from B0-plan.** Offline only (no
simulator ran); pre-change copies in `sim/step5/_checkpoints/B1/`.
- `sim/ltm_adapter.py` (new): `HonestEventAdapter`, `ObjectMemoryProxy`, `LtmChooser`, `FakeHonestAgent`. It imports **no step-4 module at
  all** (stricter than B0's "none at module load"): `objmem` (as `geometry`) and the prior chooser (as `fallback`) are injected by the
  executor, so the honesty gate scans it like every `ltm*.py`; the gate got two declared entries (the adapter's single `on_admission` door for
  the verdict; the adapter selftest may import the step-4 fake kitchen).
- Facade: `LongTermMemory.memory_block(wm, level, lang, channels, types, budget_tokens, joint=True)`, `begin_test_epoch / end_test_epoch /
  frame_of_epoch` (RAM-only overlay on frozen stores), `where_is(..., channels=)`. Channels (`places, doubted, placements, habits, negatives,
  prior`) are applied before competition / suppression / top-k; `None` = A-stage behaviour. **Retention is joint for EN and FA** (largest
  k that renders completely in both): with per-language fitting the Persian block lost a record at budgets where English kept it (found by
  the sweep, fixed). The where lines get `where_share` of the *aggregate* budget; the first type that does not fit stops the block.
  Parse not ok -> block types are the surface mentions of movable types.
- `HonestSkills(on_event=None, place_hints=None)`: open / close / inspection / search start+end events, a structured `_Fail.under`
  (exposed in `detail["under"]` only when the flag is on), `HINT_BOOST = 4` on plan items within a hint's radius, stable sort in the bounded
  policy. Flag off => the pre-B1 copy and the current module send identical simulator calls and return identical results (selftest 1).
- `sim/step5_eval.py`: `location_config` (B3 preset `--preset location`; **`core_config` is unchanged**, so B0's "`--preset core`" wording
  in B-g means this preset), `P-location`, world kind `canonical`, `epoch_overlay`, `test_langs`, `practice_reset` (the RUNNER declares the
  reset before the read-only check; the executor never writes), allowlist gate 12.10 (`final_test_allowlist`, `assert_items_allowed`, checked
  at construction = start and every resume, plus `allowlist_sha` in the header), `--executor-args`. New config fields are hidden from
  `to_dict()` while default, so A-stage config hashes are unchanged (asserted against the pre-B1 copy).
- `sim/step5_live.py` (new): `run_episode_core` (testable offline), `LiveExecutor`, `make_executor`, the smoke CLI, `chain_report`,
  `route_diff`, `render_planner_messages` (the only way memory reaches a planner: `PlannerContract(memory_block=...)`), `assert_b_memory`
  (refuses any receipt whose episode is not `ep:<24 hex>` / `smoke-...`, i.e. replay seeds). **Everything defaults OFF** (memory none,
  hints off, `--memory off`, `--hints off`); the smoke CLI allows only nontest minus dev_pilot unless `--allow-dev-pilot` is passed
  (debugging; recorded in every row; B3 never). `step4_refplans.py` is untouched (selftest compares bytes).
- Astra's 10 decisions: 6 adopted, 5 changed and applied (callbacks other than contract refusals recorded and fatal; chooser failures
  recorded and fatal; cross-frame candidates give type advice but no coordinate hint; parse-failure types; defaults OFF; per-test fresh
  frozen copy + hash check + overlay after boot in the CLI; first-detection primitive and the actual search cap recorded; strict allowlist;
  `chain_report` / `route_diff` rewritten to use fields that exist).
- Known limits carried to B2/B3: (1) R2 means a *manipulated* object leaves no location claim (a pickup supersedes it; the natural sighting is
  a search trial, i.e. the dormant habit channel), so `places` learns from unmanipulated sightings - application coverage must be measured, not
  assumed; (2) a place centre is the ground point of a detection box on a door, so hints match within 2 x the type's place gate
  (`HINT_GATE_FACTOR`, declared) and the matching plan item is a measured front; (3) non-container place visits are not counted as visits;
  (4) the real-tokenizer budget check is still open (the block is counted with the registry estimator; EN/FA identity is by construction).

**2026-10-10 (B2-live, Opus; Astra `sim/step5/_astra/B2-live.txt`) - amendments from the first live runs.** (1) B-i's "places-K0 == none-K0 on every
paired test" is replaced by an exact OFFLINE control (adapter selftest) plus a LIVE distributional control (R = 4 repeats per item and arm,
+-10 % engineering margin on mean primitives and +-10 points SR, "inconclusive" otherwise), because the live simulator is not deterministic
(same item, same code: pickup-spoon 2/6 successes). (2) Live runs set `TRACE_ENABLED=0`; sim/trace.db is quarantined (already corrupt; the
first live runs touched it). (3) B1's hint (a weight boost on known fronts) cannot change a route when the remembered container has not been
seen yet; the flag `hint_goals` (default OFF) adds a look goal at the remembered point. It halved a Cup search (78 vs 158 primitives) and
worsened a DishSponge search (+50 %), so it is not accepted as a B3 channel yet. (4) The K = 1 development protocol is "exploratory,
adaptively selected familiar-type" (practice item of the test's target type, mapping frozen in the B2c script). (5) Habits and negatives stay
dormant. Numbers and files: docs/step5-progress.md "B2-live".

**2026-10-11 (C2-design, Opus; Astra `docs/final-test/_astra/C2-design.{prompt,raw,}.txt`) - how experience reaches the planner in the final
test.** Design only (no code yet, no run). Written before any acceptance, cost-pilot, practice or held-out test outcome. Evidence that informed
it, disclosed: the structural analysis below; C1 (a nonempty places block changed Gemma's plans in 4/6 presentations, n = 6); B2 (hint goals:
Cup -51 % primitives, DishSponge +50 %). Mirrors prereg amendment A9.

*C2-a. Why the frozen gates are (nearly) empty at test.* Families merge items with equal gold-plan concept keys or the same level + skeleton,
so an exact whole-intent match (8.3) between a practised non-test item and a test item is rare, and at K = 10 single-pass practice a
whole-intent procedure rarely reaches `trusted` (2 distinct units). Lessons need the same failure key to recur in >= 2 later episodes and an
executor recovery hook that was never wired. "Rare", not "impossible" (Astra): two same-intent items can sit in one family; attainable support
is measured (C4 reports the support distribution), not assumed.

*C2-b. Procedure FRAGMENTS (new; role level).* From each admitted procedure (R7 unchanged: privileged terminal goal check true, every step ok,
episode valid; practice only) one fragment per instruction role: the plan steps after the previous role's matched step up to and including
this role's matched step (the existing `body_consistent` alignment). Kept only if (1) the alignment is unambiguous: every prefix step names the
role's object type or its destination type (e.g. `go_to Fridge, open Fridge, put_in Apple Fridge`), otherwise the fragment is discarded and
logged; (2) non-trivial: > 1 step (a one-step fragment repeats the role and adds nothing). Body = plan-level steps only (executor retries are
inside a step and never become steps). A fragment keeps its per-step observable preconditions (holding / hand empty, container state seen)
and its provenance (procedure id, episode, write_log id). Fragment key = that one role, exact on verb, type, destination, state, negation
(`?` never matches; FA/EN through `concepts`). Support = distinct episode units (7.1 rules) whose admitted procedure contains an identical
fragment (same key, same step signature). Fragment failed reuse = a later practice episode whose plan contains the identical fragment and
one of its steps fails (any cause); failed reuses >= support -> hidden (contradicted). Scope of any claim: **transfer of familiar components
to held-out task families**, not of unseen components.

*C2-c. Plan-time gate (test and practice identical).* Per fully resolved role of the current instruction: same kitchen label; key exact;
support >= 1; fragment not contradicted; parent procedure not demoted / retired. Preconditions are NOT checked at plan time (nothing has
been done yet); they are rendered as prerequisites. Rendering is conditional and qualified, never an order: support >= 2 "To <role>: this
worked N times: <steps> (needs: <preconditions>)"; support 1 "... worked once, unconfirmed: ...". Whole-intent procedures keep the frozen
8.3 gate (trusted only) inside the same `procedures` channel. One fragment per role (highest support, then oldest E, then id); roles in
instruction order. Fixed EN and FA templates, no LLM, no instruction text copied.

*C2-d. Lessons: DROPPED from the final-test treatment (Astra D3).* Plan-time lesson application would be a new mechanism needing its own
acceptance (a remedy appearing in a plan is exposure, not help; `search_for` is not "search elsewhere"). Lessons are still consolidated in
practice (default actions, no Hypothesizer, audit only) and are never rendered or applied in practice or test; the executor recovery hook
stays unwired. Thesis wording: "failure lessons are recorded but not used in the final test".

*C2-e. Budget (replaces the estimator for every block that reaches a planner, places included).* Retention is decided by the MAXIMUM real
token count over {Gemma (`prompt_eval_count` difference), Anthropic (`count_tokens`)} x {EN, FA} of the block incl. its header: the largest
prefix of whole records (places first within `where_share`, then procedures) that fits the level budget under that maximum, so record ids are
identical across models and languages. Measured again after truncation; gate at freeze: every S\* block of every test presentation <= its
budget for both tokenizers. Codex arms receive the same text; their tokens are reported, not enforced. FS (H4): filled with the same counter
to <= the same allowance (6.4 truncation rule unchanged); its real size per presentation is reported next to the +M block's.

*C2-f. Practice reads memory online (Astra D5: adopt).* Attempt n sees the block built from attempts 1..n-1 through the same renderer and the
same gate (C2-c); rendering, update and gate rules are frozen before any practice episode. H4's FS block comes from the same S\* episodes.

*C2-g. hint_goals.* Decided BEFORE C4 and held fixed through C4, the cost pilot, practice and the test. Default OFF. Only if Arash orders it:
ONE prespecified fix ("open the front nearest the remembered point first") and ONE screen (stage C3b): cup / tool-scrub / sponge-sink + 3
new non-test, non-dev items, 4 repeats, accepted iff no item has > +10 % mean primitives vs hints-off and SR is not lower on any item;
otherwise OFF for the whole study. No second fix on the same items.

*C2-h. Acceptance of the `procedures` channel (stage C4; frozen now).* Data: non-test minus dev_pilot. The 15 non-test families are split by a
seeded draw (seed 20261011) alternating B / A within majority-level strata, no role-overlap optimisation (A/B role overlap reported).
Practice on A, evaluate on B (B capped at 6 items per family, as the test; illustrative: A 7 families / 63 items, B 8 families / 27 items =
54 presentations). Practice: 3 orders (seeds 0, 1, 2) x K = 10 started Gemma attempts on A, online (C2-f), executor policy per C2-g, places +
procedures on, one snapshot per order; the best order is never selected. Evaluation on B: one observation capture per presentation (shared
by every arm and order); arms G0 (no LTM), G+P (places only), G+PF (places + procedures); Gemma strict schema (A7), cache off. Planning over
all 3 orders; executed SR on **order 0 only** (deduplicated execution units; P and PF share snapshot and chooser, so they differ only through
the plan); execution conclusions are limited to order 0. Repeatability: G+P is called twice (r = 0, 1) on order 0 to measure the baseline
plan-identity rate. Request identity: where the PF block text equals the P block text the two requests must be byte-identical (gate); output
differences there are planner variance.
Decision rule (engineering tolerances, NOT evidence of non-harm or benefit; item-level estimand as prereg 9.1, EN/FA paired):
1. integrity: 0 audit violations, 0 decision-path oracle reads (the declared terminal admission check is the only privileged read),
   snapshot hashes unchanged, no B item id or text in any A snapshot, EN/FA identical record ids, single-renderer static guard;
2. coverage: mean over the 3 orders of the share of B presentations with >= 1 procedure line >= 10 %; otherwise "inactive at K = 10" and
   NOT added;
3. harm screen PF vs P: net paired loss <= max(1, 0.05 n_B) item-equivalents for first-pass validity, final validity and executed SR
   (order 0); any breach -> NOT added;
4. reported, never decisive: PC / TDS (diagnostic; they penalise valid alternatives), primitives, opens, walking, raw paired failures,
   family-cluster bootstrap CIs, P vs G0 (first Gemma-planned evidence for places), support distribution, discarded fragments by reason.
Fallback: places-only ("location memory"). If accepted, the thesis may say "experiential memory" only if procedure lines render on >= 10 % of
TEST presentations (mechanism coverage, prereg 8); otherwise "location memory with a dormant procedure channel". Coverage is exposure only,
not use, correctness or benefit. Any outcome-driven revision after C4 needs fresh acceptance data or is labelled exploratory.

*C2-i. Order of work.* C3 offline build (fragment extraction, gate, renderer, real-token retention, A/B split script, selftests; pre-change
copies) -> optional C3b (hint-goal fix + screen, only on Arash's order) -> C4 acceptance run (WSL, Arash's go) -> cost pilot (dev snapshot
built with the final channel set, A5) -> S\* practice. Channel set, params sha and these rules go into the S\* manifest before any test call.

**2026-10-11 (W0-amend, Opus; mirrors prereg s16 A10, A12, A13; written before any new outcome)** - three amendments to this spec.
- *A10 (verb-alias fix in the 8.3 gate, found and fixed in C3 before any C3 outcome).* Concept-role verbs are abstract (`pick`, `put`,
  `find`) while plan steps carry tool names, so `_step_matches` never matched and the whole-intent gate was closed for every pick / put / find
  procedure. Fix: the fixed mapping `ltm_retrieval.ROLE_TOOLS` (pick -> pickup, pick_and_place; put -> pick_and_place, put_in; find ->
  search_for, go_to); tests for the wrong object, a composite step and negation in `sim/_step5_fragments_selftest.py`. Nothing else changed.
- *A12 (C2-e allocation; post-pilot, token counts only).* Declared minimum place-record capacities N_L = 1 / 2 / 3 (L1 / L2 / L3; design
  choices). Places allowance A_L = max(where_share x B_L, H + N_L x r) with H, r the maximum real header / one-record costs over {Gemma,
  Anthropic} x {EN, FA} measured on the re-captured dev and non-test C3 blocks; procedures get B_L - A_L. If H + N_L x r > B_L, B_L rises to
  the smallest multiple of 10 that fits, at most 2 x (the registered range); otherwise the level is memory-incapable and reported. Minimum-
  coverage (capacity) check before the freeze and on every S\* test prompt; values recorded in params.json as "A12, token-derived"
  (FP1-measured). Full text: prereg A12.
- *A13 (C2-h).* The C2-h acceptance runs unchanged on the final implementation (after A10, A11, A12), all three orders re-run; the C3 order-0
  run was a one-order pilot screen and is not reused. Procedures enter S\* only if C2-h passes; otherwise places-only.
