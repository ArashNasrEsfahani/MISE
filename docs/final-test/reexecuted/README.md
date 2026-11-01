# Re-executed analysis

This directory recomputes the post-hoc **re-executed** analysis of the final test from the records in this repository.

## What it is

In the registered analysis, a presentation counts as a failure when the simulator could not restore its starting scene.
The re-executed analysis treats those presentations differently:

- A presentation interrupted by a scene-restoration failure was executed again, with the same saved plan, from the same
  restored state. Presentations that were still not restored got up to three rescue passes.
- A presentation that was still not restored after the rescue passes is left out. In a paired contrast it is left out
  of both conditions.
- Only interruptions before the first action (a failed scene restoration, or a simulator crash) are handled this way.
  A failure after execution started stays a failure.

No plan was generated again. The re-executions reused the saved plans, and the scores come from the registered scorer
(`sim/final_test/matrix/score.py`) and the registered estimators (`sim/final_test/matrix/analyze.py`,
`sim/final_test/matrix/k23_analyze.py`).

## Regenerate the results

From the repository root:

```
python docs/final-test/reexecuted/run.py
```

This rewrites `final_results.json`, `l3_breakdown.json` and `joint_bootstrap.json` in this directory, so `git status`
shows whether anything changed. Add `--out DIR` to write them somewhere else.

- Runtime: about 2 minutes on a laptop CPU (113 s measured).
- Needs Python with numpy and scipy; scipy is not optional, it gives the t quantiles of the held-out intervals (tested
  with Python 3.14.0, numpy 2.4.4, scipy 1.17.1).
- No simulator, no language model, no GPU and no network.

The script stops without writing `final_results.json` if a check fails. The checks: the same code must reproduce the
stored values of two earlier analyses (the adjusted analysis of the sealed rows, and the no-exclusion analysis of the
first re-execution round), the input hashes must match, and the restoration flags must be consistent.

## What the output holds

`final_results.json`:

| Key | Content |
|---|---|
| `rates` | success rates per kitchen, condition, level and language, with 95 % intervals |
| `three_kitchen_gain`, `three_kitchen_means` | equal-weight means over the three kitchens |
| `level_diffs`, `k1_contrasts_holm_path`, `k23_contrasts_cr2`, `heldout_mean_cr2` | paired contrasts |
| `memory_effect_triples` | memory effect of each planner, and Gemma 4 with memory minus Opus 5.5 without, per kitchen and held-out mean, as [estimate, lower, upper] in pp |
| `counts`, `count_ranges`, `rescue`, `rerun_summary` | presentations analysed and left out, and the rescue passes |
| `ablation` | follow-up: memory components removed or replaced |
| `plan_checker` | follow-up: plan checker |
| `goal_library` | follow-up: goal library |
| `larger_planners` | follow-up: larger local planners (Gemma 4 E4B, Qwen 3.5 9B) |
| `failure_breakdown`, `diagnostics`, `language`, `controls`, `contamination_sensitivity` | further breakdowns |
| `gates` | the checks above and their results |

`l3_breakdown.json` explains why Abstract (L3) instructions fail: each failure of Gemma 4 and Opus 5.5 with Scene Memory
is classed by walking its recorded plan symbolically from the recorded state at reset.

`joint_bootstrap.json` gives the three-kitchen intervals from a joint family bootstrap (key `joint_ci95_pp` of each
contrast): families recur across kitchens, so three-kitchen intervals resample each family once and apply the draw in
every kitchen where it appears. The same file keeps the per-kitchen independent intervals of `final_results.json`
(`existing_ci95_pp`) for comparison, and covers the memory gain, the gain in each language, the larger local planners and
the memory components.

## Files

| File | Role |
|---|---|
| `run.py` | the command above |
| `compute_final.py` | the re-executed analysis (writes `final_results.json`) |
| `l3_breakdown.py` | the Abstract failure breakdown (writes `l3_breakdown.json`) |
| `joint_bootstrap.py` | three-kitchen intervals with the joint family bootstrap (writes `joint_bootstrap.json`) |
| `adjusted.py`, `paired.py`, `noexcl.py` | loading, flag filter, paired contrasts, shared helpers |
| `rescue.py` | reader of the rescue passes |
| `plan_walk.py` | the symbolic plan walk used by `l3_breakdown.py` |
| `inputs/` | extracts of records that are not in this repository in full |
| `inputs/MANIFEST.json` | every input file with its sha256; for each extract, its source and the rule that made it |

The record files the scripts read as they are live at their usual paths: `docs/final-test/matrix/<ns>/outcome.sealed.jsonl`,
`docs/final-test/k23/codex/<ns>/outcome.codex.jsonl`, `docs/final-test/k23/rerun/test-k2-v1/`, and
`docs/final-test/strengthen/R/` (first re-execution round) and `docs/final-test/strengthen/N/` (larger local planners and
rescue passes). The namespaces are `test-v1` (development, FloorPlan1), `test-k2-v1` (held-out A, FloorPlan15) and
`test-k3-v1` (held-out B, FloorPlan17).

## Limits

- The memory-component, plan-checker and goal-library experiments also stored rows of their adjusted version. Those rows
  are not included, so the checks that reproduce the adjusted version of those three experiments are skipped. The
  values in the output do not use them.
- The event logs are reduced to their attempt-start rows, the only rows the analysis reads.
- The Abstract breakdown reads the state at reset from `inputs/l3_reset_states.json`, extracted from the capture records,
  which hold the planner prompts and are not published.

## Fixed-procedure prompt comparison (A7)

`python docs/final-test/reexecuted/a7_fixed.py` regenerates the fixed-procedure prompt comparison (fixed full store and fixed short
procedure texts against Scene Memory, retrieved procedures with rephrasing, rephrasing only and no memory) from
`inputs/rows/A7.no-exclusion.jsonl` and checks every value against `docs/final-test/strengthen/A/a7/analysis.json`. The arms, fixed texts,
blocks, plans and execution journals are in `docs/final-test/strengthen/A/a7/`, and the code that produced them in
`sim/final_test/strengthen/a7_*.py` (with the A6 / ablation modules it builds on).
