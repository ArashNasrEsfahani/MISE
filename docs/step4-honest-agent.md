# Step 4 — the honest agent (FloorPlan1)

Status 2026-10-06: **built, offline-verified and run live on FloorPlan1 (integration verified, not a
performance claim on unseen data).** Live: the s4 smoke ran 4/4 with 0 violations; the s5 reference-plan runs
(no LLM) ran all 277 general-set items in both arms; the s6 Gemma pilot ran 6 items. Every live run had 0 recorded
violations and 0 prohibited oracle reads *under the implemented checks*. Numbers and caveats: "Results" below.
Spec: `docs/step4-plan.md`. Ledger: `docs/step4-progress.md`.

## What "honest" means, stated exactly

In honest mode the agent decides from **RGB + depth run through the step-3 perception stack** (`TwoStageDetector`
plus the StateEstimator), **its own map, its own object memory and its own action history**. It also relies on these
explicitly declared exceptions:

| Read | Reason tag | Why it is allowed |
|---|---|---|
| agent position / yaw / horizon / standing / camera position | `pose` | Self-pose comes from the simulator. This is a stated assumption: there is no learned odometry. |
| `lastActionSuccess` | `actuator` | Action feedback. The error text is logged for the failure breakdown, and decisions see only `refused`. |
| `GetObjectInFrame` return at a pixel fixed beforehand | `action-resolve` | This is how the simulator's arm works. The id stays opaque: it is used once and never stored. |
| `GetReachablePositions` | `map-build` | Used only for the coverage report after the map is built. |
| goal and gate state | `scoring` | Readable only inside `with h.scoring()`, where acting is impossible. |
| ground truth | `oracle` | Used only by the oracle comparison arm. |
| detector class list, `search.prior_for`, camera FOV 90° | (priors / calibration) | Fixed knowledge, not scene contents. The class list is specific to FloorPlan1. |

So the claim for the thesis is: *RGB-D decisions with simulator proprioception, action feedback, declared priors
and pixel-to-action grounding*. A literal "camera only" claim would be wrong.

## Modules

`sim/honest.py` is the only gateway to simulator data. It holds `HonestAgent` and `MetaAudit` with a field
whitelist per reason. The other modules each add one piece:

- `sim/honest_map.py`: a 0.25 m grid built by depth-first walking. Live result: 100 % coverage, 0 violations.
- `sim/objmem.py` plus 2 tables in `sim/memory.py`: depth-grounded sightings, a trust/contradiction rule and
  held/placed beliefs. Live: 54 of 58 positions within 0.10 m.
- `sim/honest_skills.py`: acting by screen point, 3 shared retries, and search through a chooser.
- `sim/step4_refplans.py`: the honest arm and the oracle arm over the general set.

## The s7 audit: trying to disprove the claim

Every metadata read on an honest path was searched for and classified: `objects()`, `object_by_id`,
`event.metadata`, objectId selection, `GetShortestPath`, `interactable_poses`, `resolve`/`find`,
`type_list`, `["objects"]` and direct ThorAgent attributes.

- **Decision code (`honest_skills`, `honest_map`, `objmem`): no leak found.** These modules touch the simulator
  only through `HonestAgent.pose / move / act / observe`.
- **Harness code** (`step4_refplans`, `step4_smoke`) makes these reads, and all of them are allowed:
  - `ag.type_list`: the applicability gate (scoring, before the episode).
  - `ev._reset_world`: lifecycle.
  - `ev.run_steps`: oracle setup before `start_episode`, and the oracle arm.
  - `ev.state_readers`: the goal check inside `h.scoring()`.
  - `ag.field_of_view`: camera calibration.
  - `ag.stop`: lifecycle.

  The smoke script checks the goal after each step. This is report-only and is never passed to the skills.
- **Gaps fixed:**
  1. A non-strict audit used to return privileged data after recording a violation. It now withholds the data.
  2. A non-strict audit also still executed actions in the scoring phase. Such actions are now never sent.
  3. The static guard missed reads through `ag.` and `sim_agent.` and `md["objects"]`. It now has a stricter
     token list for the decision modules and a list of allowed harness reads, each with its reason.
  4. An honest episode with a violation, an oracle read or a map-build read would have been scored as FAIL. It
     is now `INVALID` and excluded from every rate and from the pairing.
  5. The oracle-state helper was moved inside the adapter region.
- **New evidence beyond grep:** an invariance test runs the real skills on two fake worlds. The second world
  renames every id, adds decoys the camera never sees, and gives a metadata list with wrong state flags. Both
  worlds produce identical primitive traces, outcomes, memory and history.

## Results (live, FloorPlan1, 2026-10-06)

All are FloorPlan1 only. FloorPlan1 is now **development data**: the honest code was fixed twice from what these
runs showed (s4-live and s5-honest), so none of this is a held-out estimate.

**s4 smoke** (`sim/step4_smoke.py`): open Fridge, pickup Apple, put it in the Fridge, toggle the Faucet on. The
first run was 3/4 (the put aimed at the swung fridge door). After the fix: 4/4 ok, 4/4 goals true, 0 retries,
0 violations, twice. This shows basic integration only; the spawn cell is already in reach of fridge and apple.

**s5 reference-plan execution** (`sim/step4_refplans.py`, no LLM; the plan is the item's reference plan). Of 277
items, the same 10 are skipped in both arms (7 goals already true at reset, 3 need a DiningTable FloorPlan1 lacks),
leaving 267 applicable:

| arm (code hash) | ok / applicable | L1 | L2 | L3 |
|---|---|---|---|---|
| oracle (`e4132d596afe5606`; forces actions, reads metadata) | 266/267 = **.996** | 107/107 = 1.000 | 103/104 = .990 | 56/56 = 1.000 |
| honest, before the s5 fixes (`bdf33b2b893c93dc`) | 198/267 = .742 | 88/107 = .822 | 76/104 = .731 | 34/56 = .607 |
| honest, final (`e2377920f9f822c2`) | 200/267 = **.749** | 93/107 = .869 | 73/104 = .702 | 34/56 = .607 |
| honest, final, strict (goal met **and** every step ok) | 181/267 = .678 | 92/107 = .860 | 63/104 = .606 | 26/56 = .464 |

- Paired over the 267: both ok 199, oracle only 67, honest only 1, both fail 0. 0 invalid episodes.
- Honest failure causes (final): retry_exhausted 38, not_detected 20, unreachable 5, goal_unmet 4. Counting
  the cause underneath each retry_exhausted: not_detected 35, action_refused 18, wrong_target 5, unreachable 5.
  Mean 0.75 retries and 100.1 primitives per item.
- **The strict row exists because 19 honest successes are simulator-defined**: they are slicing items where the
  goal was met although a step failed, 18 of them a failed `pickup Knife`. AI2-THOR lets SliceObject work
  empty-handed, so these do not show knife use. Every oracle success has all steps ok.
- **4 cook-potato items vary from run to run in the oracle arm** (0 to 2 failures in 4 runs each). The single
  honest-only success is one of them. The oracle run does not reproduce the September baseline (263/10/4) only
  because of these items.
- This is an **execution comparison under two information and action regimes**. Perception, targeting,
  navigation, recovery and `forceAction` all differ at once (the oracle forces actions), so the gap is not
  attributable to perception alone. Planning, language and RAG are not tested here. Each arm ran once on frozen
  code, so this is not a reproducible performance estimate.

**s6 Gemma pilot** (`sim/step4_gemma_trial.py`, gemma4:e2b-it-q4_K_M, think off, temperature 0): 4 of 6 goals
met (L1 2/2, L2 2/2, L3 0/2). All 6 accepted plans executed without a failed step, so both L3 failures are
planning failures. 0 violations. The 6 items were run 3 times while two integration bugs were fixed, so this
is **integration evidence, not an evaluation**. The model's provenance needs Arash's confirmation (see the ledger,
s8-wrap).

## Open items

1. **No held-out result yet.** A frozen protocol on other kitchens (or unseen FloorPlan1 items) is needed before
   any performance or generalisation claim. Planning has only the 6-item pilot.
2. Honest weaknesses seen live: thin flat objects (Spatula: the box-centre point resolves to the CounterTop),
   look-alike pickups (SaltShaker vs PepperShaker; the specialist only runs while holding), Faucet toggle_off
   while water runs, and search breadth (Cup and Plate in upper cabinets, Fork and Spoon in a far drawer).
3. `opened_at` is keyed by object type, not by container track. Blocked edges learned live are not saved to
   the map file.
4. The `reachable_positions` and `scoring()` doors can still be called at runtime. They are kept out of decision
   code by the static guard, and the s5 scoring invalidates any episode that uses them. They are not blocked
   while the agent is acting.
5. The heat and cook composites in the oracle path never check their outcome; that is why the potato items
   vary.
