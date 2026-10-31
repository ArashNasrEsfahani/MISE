# Final-test results

This file holds the recorded results for the protocol in `docs/final-test/preregistration.md`. The protocol state for
the development kitchen is frozen by `docs/final-test/freeze.json`, and the held-out kitchens have their own records,
`docs/final-test/k23/freeze_FloorPlan15.json` and `docs/final-test/k23/freeze_FloorPlan17.json`.

| Name used here | AI2-THOR scene | Role |
|---|---|---|
| Development | FloorPlan1 | development kitchen |
| Held-out A | FloorPlan15 | held-out kitchen |
| Held-out B | FloorPlan17 | held-out kitchen |

Two analyses exist:

- **Registered.** Every presentation counts. A simulator restoration failure counts as a failure.
- **Re-executed (post-hoc).** A presentation interrupted by a restoration failure runs again with the same saved plan.
  Presentations still not restored after three rescue passes are left out, and a paired contrast drops them from both
  conditions. Failures that happen after execution starts stay failures.

Condition codes, as used in the analysis files: G = Gemma 4 E2B, O = Opus 5.5, X = GPT6-Sol, P = reference plans (no
planner). `0` = without Scene Memory, `+M` = with Scene Memory. Scene Memory is the frozen memory block, called the
package in the protocol and code. `+FS` = solved examples only, `/p` = replay of memory-conditioned plans without guided
search, E4B = Gemma 4 E4B, Q9B = Qwen 3.5 9B. A presentation is one item in one language, English or Persian. pp =
percentage points.

## 1. Registered results

Success rates are percentages. Intervals are 95 %.

### Development (FloorPlan1)

| Condition | Code | Overall [95 % CI] | Atomic (L1) | Sequential (L2) | Abstract (L3) | PC | TDS |
|---|---|---|---|---|---|---|---|
| Gemma 4, no memory | G0 | 54.0 [44.5, 63.4] | 69.1 | 57.0 | 23.2 | 0.612 | 0.340 |
| Gemma 4, memory | G+M | 68.4 [59.3, 76.8] | 86.2 | 78.0 | 21.4 | 0.808 | 0.391 |
| Opus 5.5, no memory | O0 | 78.0 [69.9, 84.7] | 89.4 | 73.0 | 67.9 | 0.877 | 0.794 |
| Opus 5.5, memory | O+M | 82.0 [76.1, 87.3] | 88.3 | 81.0 | 73.2 | 0.910 | 0.851 |
| GPT6-Sol, no memory | X0 | 76.4 [69.9, 82.4] | 89.4 | 74.0 | 58.9 | 0.902 | 0.714 |
| GPT6-Sol, memory | X+M | 80.4 [73.8, 86.3] | 90.4 | 83.0 | 58.9 | 0.899 | 0.757 |
| Reference plans | P0 | 87.2 [82.0, 91.9] | 91.5 | 84.0 | 85.7 | --- | --- |

PC = Procedural Correctness. TDS = Task Decomposition Score.

### Held-out A (FloorPlan15)

| Condition | Code | Overall [95 % CI] | Atomic (L1) | Sequential (L2) | Abstract (L3) |
|---|---|---|---|---|---|
| Gemma 4, no memory | G0 | 63.0 [51.8, 73.8] | 75.5 | 68.6 | 19.4 |
| Gemma 4, memory | G+M | 77.0 [66.2, 86.5] | 87.2 | 87.1 | 30.6 |
| Opus 5.5, no memory | O0 | 87.5 [81.6, 92.6] | 91.5 | 87.1 | 77.8 |
| Opus 5.5, memory | O+M | 88.0 [77.5, 95.8] | 88.3 | 91.4 | 80.6 |
| GPT6-Sol, no memory | X0 | 88.0 [82.8, 92.6] | 93.6 | 87.1 | 75.0 |
| GPT6-Sol, memory | X+M | 86.5 [75.9, 94.6] | 88.3 | 88.6 | 77.8 |
| Reference plans | P0 | 93.0 [88.6, 96.5] | 95.7 | 88.6 | 94.4 |

### Held-out B (FloorPlan17)

| Condition | Code | Overall [95 % CI] | Atomic (L1) | Sequential (L2) | Abstract (L3) |
|---|---|---|---|---|---|
| Gemma 4, no memory | G0 | 59.4 [48.1, 69.7] | 69.6 | 65.3 | 23.7 |
| Gemma 4, memory | G+M | 70.8 [60.6, 80.3] | 77.2 | 86.1 | 26.3 |
| Opus 5.5, no memory | O0 | 86.1 [81.4, 90.4] | 84.8 | 88.9 | 84.2 |
| Opus 5.5, memory | O+M | 86.1 [78.7, 92.5] | 88.0 | 88.9 | 76.3 |
| GPT6-Sol, no memory | X0 | 83.2 [75.0, 89.8] | 88.0 | 76.4 | 84.2 |
| GPT6-Sol, memory | X+M | 81.2 [72.3, 88.4] | 88.0 | 75.0 | 76.3 |
| Reference plans | P0 | 93.1 [89.2, 96.6] | 91.3 | 91.7 | 100.0 |

The analysis file labels the held-out kitchens an exploratory, estimation-focused replication. Their registered design has
four core conditions, and the GPT6-Sol rows come from an addendum outside it:
`docs/final-test/k23/decisions/K23-CODEX-ADDENDUM.json`.

### Paired contrasts, development kitchen

| Contrast | Code | Estimate (pp) | 95 % CI (pp) | Holm p | Presentations / items |
|---|---|---|---|---|---|
| Gemma 4, memory effect | H2a, G+M - G0 | +14.4 | [+7.4, +21.6] | 0.0006 | 250 / 125 |
| Opus 5.5, memory effect | H2c, O+M - O0 | +4.0 | [-1.0, +10.8] | 0.1114 | 250 / 125 |
| GPT6-Sol, memory effect (exploratory) | H5a, X+M - X0 | +4.0 | [0.0, +8.5] | --- | 250 / 125 |
| Opus 5.5 minus Gemma 4, same memory | H3a, O+M - G+M | +13.6 | [+7.0, +20.8] | 0.0012 | 250 / 125 |

## 2. The protocol's highest bar

Development kitchen, 125 items, 250 presentations. Source: `docs/final-test/matrix/test-v1/analysis.md`.

| Comparison | Code | Margin | Estimate | Nominal 96.5 % interval | Families with a discordant presentation | Result |
|---|---|---|---|---|---|---|
| Gemma 4 with memory minus Opus 5.5 without memory | H1, G+M - O0 | 5 pp | -9.6 pp | [-21.0, +1.8] | 20 (guard: at least 20) | not established (protocol wording: non-inferiority unresolved, inconclusive) |

- The protocol labels this comparison primary. It is the highest bar the study set: a small local model compared, across
  all instruction levels, with a frontier model.
- Section 2 of the protocol says that at a true difference of 0, non-inferiority is shown only 10-48 % of the time, that
  it can succeed only if the small model is actually ahead, and that an inconclusive result is likely. This result was
  therefore expected to be likely.
- Amendment A24.1-H1, item 5, gives the figures for the decision rule that was finally applied (`h1-rule-1`). At a true
  difference of 0, non-inferiority came out in 0-15 % of simulated datasets under a beta-binomial model and 3-17 % under a
  probit model.
- This wording claims neither non-inferiority, inferiority nor equivalence. Sensitivity views and the protocol's A36
  amended subset analysis are printed in `analysis.md` and not repeated here.

## 3. Re-executed analysis: presentations analysed

Each cell gives presentations analysed, with presentations left out in brackets.

| Condition | Code | Development (of 250) | Held-out A (of 200) | Held-out B (of 202) |
|---|---|---|---|---|
| Gemma 4, no memory | G0 | 247 (3) | 199 (1) | 200 (2) |
| Gemma 4, memory | G+M | 249 (1) | 199 (1) | 200 (2) |
| Opus 5.5, no memory | O0 | 247 (3) | 199 (1) | 199 (3) |
| Opus 5.5, memory | O+M | 248 (2) | 199 (1) | 200 (2) |
| Reference plans, no memory | P0 | 248 (2) | 199 (1) | 200 (2) |
| Reference plans, memory | P+M | 249 (1) | 199 (1) | 201 (1) |
| GPT6-Sol, no memory | X0 | 248 (2) | 199 (1) | 198 (4) |
| GPT6-Sol, memory | X+M | 249 (1) | 199 (1) | 200 (2) |
| Gemma 4, solved examples only | G+FS | 247 (3) | - | - |
| Opus 5.5, solved examples only | O+FS | 247 (3) | - | - |
| Gemma 4, replay without guided search | G+M/p | 248 (2) | - | - |
| Opus 5.5, replay without guided search | O+M/p | 246 (4) | - | - |
| Gemma 4 E4B, no memory | E4B0 | 247 (3) | 200 (0) | 201 (1) |
| Gemma 4 E4B, memory | E4B+M | 248 (2) | 199 (1) | 200 (2) |
| Qwen 3.5 9B, no memory | Q9B0 | 249 (1) | 199 (1) | 200 (2) |
| Qwen 3.5 9B, memory | Q9B+M | 250 (0) | 199 (1) | 200 (2) |

Across the first twelve rows above, the presentations left out are 27 of 3,000 (development), 8 of 1,600 (held-out A) and
18 of 1,616 (held-out B).

## 4. Rescue passes

A unit is one execution. It can serve several conditions and several experiments, so unit counts and presentation counts
differ. A rescue pass reran each unit whose latest state was a restoration failure or a crash, with its saved plan, a fresh
simulator boot and the same reset check, for up to three passes.

| Kitchen | Unique units tried | Recovered | Still blocked |
|---|---|---|---|
| Development (FloorPlan1) | 47 | 33 | 14 |
| Held-out A (FloorPlan15) | 19 | 8 | 11 |
| Held-out B (FloorPlan17) | 40 | 29 | 11 |
| **All** | **106** | **70** | **36** |

Each cell below gives units tried / recovered / still blocked.

| Kitchen | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---|
| Development (FloorPlan1) | 47 / 16 / 31 | 31 / 7 / 24 | 24 / 10 / 14 |
| Held-out A (FloorPlan15) | 19 / 4 / 15 | 15 / 3 / 12 | 12 / 1 / 11 |
| Held-out B (FloorPlan17) | 40 / 17 / 23 | 23 / 8 / 15 | 15 / 4 / 11 |
| **All** | **106 / 37 / 69** | **69 / 18 / 51** | **51 / 15 / 36** |

Cause: scene restoration failed its state check. The final data hold no crash (`exec_crash`: 0), and every blocked state
is flagged. The differences recorded by the check were in the observed types and scene summary (development), the agent
pose and scene summary (held-out A), and the observed types, scene summary and its coordinates (held-out B).

## 5. Experiments

| Experiment | Kitchens | Size | Planners and memory | Handling |
|---|---|---|---|---|
| Main | Development | Test set, 125 items, 250 presentations per condition | Gemma 4, Opus 5.5, GPT6-Sol, each with and without Scene Memory | One analysis rule for all conditions (section 3) |
| Main | Held-out A | Applicable test subset, 100 items, 200 presentations per condition | As above | As above |
| Main | Held-out B | Applicable test subset, 101 items, 202 presentations per condition | As above | As above |
| Reference plans | All | The same test presentations in each kitchen | No planner, no guided search | As above |
| Solved examples only | Development | Test set, 250 presentations | Gemma 4 given only solved examples (the Opus 5.5 counterpart is counted in section 3) | As above |
| Search replay | Development | Test set, 250 presentations each | Plans generated with Scene Memory are kept and guided search is off | As above |
| Practice | All | Outside the test set, 10 attempted episodes per kitchen | Gemma 4; memory builds up, then is frozen | Every started attempt counts |
| Repeat | Development | Separate subset, 30 items, 60 presentations | Core conditions, 3 planning repetitions, 2 executions per distinct plan | Execution agreement reported on its own |
| Recognition | All | Sealed holdouts, 10 episodes per kitchen, 526 / 517 / 494 targets | Primary detector, further-trained primary, active disambiguation | A target with no qualifying box counts as a failure |
| Component ablation | All | 9,172 plan cells, 743 new executions, 40 restoration re-executions, 60 noise re-executions (58 completed) | Gemma 4 and Opus 5.5, with memory components removed or replaced | Changed plans run again, identical plans reuse their execution |
| Larger local planners | All | 848 new executions out of 2,586 units | Gemma 4 E4B and Qwen 3.5 9B, with and without Scene Memory | Identical executions reused |
| Restoration re-execution | All | First-round targets 101 / 43 / 64 (development / A / B), then three rescue passes | Main planners and reference plans | Saved plans; unresolved restorations left out |
| Plan checker | All | Sealed plans 651 with and 648 without Scene Memory; flagged 52 and 110; changed 27 and 75 | Gemma 4, with and without Scene Memory | Post-hoc; one repair per flagged plan, valid repairs kept, 50 new executions |
| Rendering check | Held-out A | 54 execution units in the rendering-fault window | Affected conditions only | Diagnostic rerun only; finished original executions stay in the results |

Counts are per language. Executions shared between conditions stay linked when resampling.

## 6. Models, settings and where things are

| Planner | Identifier and settings | Recorded in |
|---|---|---|
| Gemma 4 E2B | `gemma4:e2b-it-q4_K_M`, Ollama 0.32.15, temperature 0, thinking off, context 8192, output 768, RTX 4060 laptop GPU with 8 GB | `docs/final-test/freeze.json` (keys `models`, `ollama`), `sim/planners/ollama_provider.py` |
| Opus 5.5 | `claude-opus-5-5`, Anthropic API, effort medium, output limit 8000 | `docs/final-test/freeze.json` (key `models`), `sim/planners/anthropic_provider.py` |
| GPT6-Sol | `gpt-6-sol`, codex-cli 0.162.1, high reasoning effort; the serving model is not observable through the CLI | `docs/final-test/freeze.json` (keys `x_arms`, `codex_cli_version`), `sim/planners/codex_provider.py`, `docs/final-test/matrix/x_isolation.json` |
| Gemma 4 E4B | `gemma4:e4b-it-qat`, Ollama 0.32.15, E2B settings with a strict JSON schema | no record here, see below |
| Qwen 3.5 9B | `qwen3.5:9b`, Ollama 0.32.15, E2B settings with a strict JSON schema | no record here, see below |

| Topic | Files |
|---|---|
| Prompts | `sim/planners/contract.py`, `sim/planners/ft_v2.json`, `sim/planners/summary_v2.py`, `sim/planners/validator_v2.py`, `sim/final_test/matrix/treatment.py`, `sim/final_test/matrix/fs_block.py` |
| Memory schemas | `sim/pkg_schema.py`, `sim/ltm_store.py`, `sim/ltm_package.py`, `sim/step5/package_params.json`, `docs/step5-plan.md` |
| Skill definitions | `sim/honest_skills.py`, `sim/planners/exec_contract.json`, `sim/honest.py`, `docs/step4-honest-agent.md` |
| Dataset | `sim/dataset/general.json`, `sim/dataset/README.md`, `docs/general-instructions.md` |
| Splits | `docs/final-test/split_manifest.json`, `docs/final-test/families_approved.json`, `docs/final-test/inclusion_manifest_a26.json`, `docs/final-test/k23/a29_sample_FloorPlan15.json`, `docs/final-test/k23/a29_sample_FloorPlan17.json` |
| Evaluation scripts | `sim/final_test/matrix/score.py`, `sim/final_test/matrix/analyze.py`, `sim/final_test/matrix/k23_analyze.py`, `sim/final_test/matrix/k23_codex.py`, `sim/final_test_h1_stats.py`, `sim/final_test/reset_gate.py`, `docs/final-test/reset_tolerances.json`, `scripts/reproduce_analysis.py` |
| Registered results | `docs/final-test/matrix/test-v1/analysis.md`, `docs/final-test/k23/analysis/analysis_k23.md`, `docs/final-test/k23/codex/test-k2-v1/summary.codex.json`, `docs/final-test/k23/codex/test-k3-v1/summary.codex.json` |
| Rendering check | `docs/final-test/k23/decisions/K23-WINDOW-RERUN.json`, `docs/final-test/k23/rerun/test-k2-v1/analysis_rerun.md` |
| Detector training | `sim/train_kitchen.py`, `sim/train_amb.py`, `docs/kitchen-datasets.md` |

### Not in this repository

- The code, per-run records and rates of the re-executed analysis, including the rescue summary behind section 4 and the
  decision record for the executor revision in section 7.
- The result records of the component ablation, plan checker and larger local planners experiments, including the run
  settings of Gemma 4 E4B and Qwen 3.5 9B.
- The result records of the recognition experiment, and the simulation summary behind the power figures in section 2
  (`docs/final-test/h1_validation/power_summary.C7.json`).
- Detector weights and training datasets, published on Hugging Face (see `README.md`).
- Frozen memory snapshots and gate evidence. The freeze check therefore refuses to run in this repository, as
  `README.md` explains.

## 7. Notes

- **Detector training.** Detectors were trained per kitchen, including the two held-out kitchens. Amendment A38 of the
  protocol records the starting weights for FloorPlan15 and FloorPlan17.
- **Executor revision.** Re-executions in the development kitchen ran on a later executor revision than the original run
  (treatment identity `9799088` against `7a2ee211`). The authors judged it functionally unchanged. Amendment A39.5 of the
  protocol records that the paired check of the two revisions was NOT_EQUIVALENT under its pre-fixed rule, with all 80
  compared outcomes and step sequences equal, and that equivalence is not claimed. Five first-round re-executions and 7
  rescue units passed the reset check only after the later identity was mapped to the original one.
