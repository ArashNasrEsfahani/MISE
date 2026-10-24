# Final-test matrix: build spec (stage D-spec, 2026-10-11, spec version `ftm-1`)

This is the contract for the W1 build. About 14 builders work in parallel. Each one edits **only** the files listed
for them in section 13. If you need a change in someone else's file, write it in your notes; do not make it yourself.
The registered design is binding: `docs/final-test/preregistration.md` (LOCKED v1 + A1-A15; sha256[:16] `ca81dad361a64781`
at W0) and `docs/step5-plan.md` s13. This spec only turns that design into interfaces. If the spec and the
preregistration disagree, the preregistration wins; report the conflict in your notes.

The shared code is `sim/final_test/matrix/common.py`. It is implemented and tested (`python sim/_ftm_common_selftest.py`,
89/89, offline). Import it like this, from any matrix module, run from the repo root:

```python
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # sim/final_test/matrix
import common as C                              # also puts sim/ on sys.path (planners, ltm, step5_live, ...)
```

## 0. Rules for every builder

- **No test item, ever, in W1.** Selftests use dev items, practice-pool items or synthetic ids. `C.guard_items` refuses
  test items unless `--allow-test`, namespace `test-v1`, a valid `freeze.json` and `families_approved.json` all hold.
- **Offline gates.** Every module has its own selftest (section 13). It runs on Windows `python` with no simulator,
  no WSL, no Ollama and no API. Simulator code is imported lazily, inside functions (as `step5_live` does).
- **Python 3.10.** WSL runs `~/thor-env` (python 3.10); Windows runs 3.14. No syntax newer than 3.10 in any module
  that WSL imports (capture, execute, practice_k, reset_gate, step5_live, ltm*). Selftests can check this with
  `ast.parse(src, feature_version=(3, 10))`.
- **One renderer.** Every prompt is built by `planners.contract.PlannerContract` (gate 12.11). Memory reaches a prompt
  only as `memory_block=` and the FS block only as `fewshot_block=`.
- **Honesty.** The decision path never reads simulator truth. Reads of world state for gates (12.4) and scoring
  happen only inside `h.scoring()` in functions named `_score*` (the static gate in `_step5_adapter_selftest.py`).
- **Generalise.** No FloorPlan1 constants in rules. Values measured on FloorPlan1 are marked `FP1-measured` where
  they are stored.
- **Keys.** Anthropic calls only through `planners.keys.get_anthropic_key` (`THESIS_ANTHROPIC_API_KEY`). Never print,
  log or write a key. Every journal, cache and archive file must pass `C.key_leak(text) == False`.
- **Save after every step.** Journals are append-only and fsync'd (`C.Journal`). Each stage appends short ledger
  entries to `docs/final-test/progress.md`, with its stage id in the heading. Pre-change copies of any existing file
  you edit go under `sim/final_test/_checkpoints/<stage>/`.
- **No git commit.** Never touch `docs/thesis/`.

## 1. Pipeline at a glance

| step | where | module | input | output |
|---|---|---|---|---|
| practice + nested snapshots (W4) | WSL + Gemma | `practice_k.py` | practice pool | `sim/final_test/snapshots/o{0,1,2}/k{0,1,3,5,10}/`, `practice.jsonl` |
| FS blocks (W4) | Windows (counter: Ollama + free count_tokens) | `fs_block.py` | `o0` practice rows K = 10 | `docs/final-test/matrix/fs_blocks.json` |
| freeze (W4) | Windows | `freeze.py` | everything frozen | `docs/final-test/freeze.json` |
| Phase A capture | WSL, no LLM | `capture.py` | items, snapshots, FS blocks | `capture.jsonl`, `states/` |
| Phase B planning | Windows, no simulator | `plan.py` | `capture.jsonl` only | `plan.jsonl`, response cache, ledger |
| Phase C execution | WSL, no LLM | `execute.py` | `plan.jsonl`, `capture.jsonl`, `states/` | `exec.jsonl` (integrity), `outcome.sealed.jsonl` |
| Phase D scoring | Windows | `score.py` | plan + exec + sealed outcomes + gold | `score.jsonl` + `score.seal.json` |
| monitor (every 20 min) | either | `monitor.py` | integrity journals + ledger, never outcomes | `monitor.jsonl` |
| analysis (W6) | Windows | `analyze.py` | sealed `score.jsonl` (or `--synthetic`) | `analysis.json` + tables |
| orchestration | either | `matrix_run.py` | CLI | runs the phases above in order, with preflight checks |

## 2. Paths and namespaces

All paths come from `C.paths(root=None)`; never hard-code them.

| what | path |
|---|---|
| matrix code | `sim/final_test/matrix/*.py` |
| reset gate | `sim/final_test/reset_gate.py` (+ `sim/final_test/reset_calib.py`, the WSL calibration runner) |
| work dir (binary, per namespace) | `sim/final_test/matrix/work/<ns>/` : `ledger.db` (+ `.jsonl` audit), `cache/`, `tokcache.json`, `<phase>.lock` |
| journals (per namespace) | `docs/final-test/matrix/<ns>/` : `practice.jsonl`, `capture.jsonl`, `plan.jsonl`, `exec.jsonl`, `outcome.sealed.jsonl`, `score.jsonl`, `score.seal.json`, `events.jsonl`, `monitor.jsonl`, `states/`, `analysis.json` |
| snapshots | `sim/final_test/snapshots/o<order>/k<K>/` (`C.snapshot_dir("S*")` = `o0/k10`) |
| FS blocks | `docs/final-test/matrix/fs_blocks.json` |
| freeze | `docs/final-test/freeze.json` |
| tolerances | `docs/final-test/reset_tolerances.json` |
| family approval | `docs/final-test/families_approved.json` (exists: Arash, 2026-10-11) |

`C.journal_path(ns, kind)` gives a journal path. The kinds are `practice capture plan exec outcome score events monitor`.

**Namespaces** (`C.NAMESPACES`). A namespace fixes the split set it may touch, its ceilings and its response-cache
namespace (prereg A4 12.13). The test namespace starts empty, and no namespace replays another's cache.

| namespace | items (split manifest key) | Anthropic ceilings (core / total) | use |
|---|---|---|---|
| `rehearsal-v1` | `dev_pilot` (12 items, 24 presentations) | 1.00 / 1.00 USD | W2 dev re-capture, rehearsal, A12 / A14 measurements |
| `practice-v1` | `practice_pool` (116 non-test items) | 0 / 0 (Gemma only) | W4 practice |
| `test-v1` | `test` (125 items; REPEAT / RELOCATION inside) | **30 / 65 USD** (A22; were 25 / 45 under A15, 25 / 55 under A18) | W5 matrix |

In this workflow, paid dev calls stay at 1 USD or less in total, so `rehearsal-v1` is capped at 1 USD. Capped-out
items (`test_uncapped_not_scored`) are refused in every namespace.

## 3. Arms, memory sources, blocks, scopes

`C.arm_info(arm)` returns `{planner, prompt, chooser}`:

- `planner` is `G | S | O | H` (`C.PLANNER_SPEC` maps these to `planners.make_planner` specs), `X:<codex model>`,
  `ref` (the reference plan, P arms) or `replay:<arm>` (the cached plans of that arm, `/p` arms).
- `prompt` is the **memory source** of the prompt block: `none`, `S*`, `FS` or `o<order>k<K>`.
- `chooser` is `prior` (`honest_skills.prior_chooser`) or a snapshot source for `LtmChooser`.

| arms | prompt | chooser |
|---|---|---|
| `G0 S0 O0 H0`, `X0`, `X-<m>0` | none | prior |
| `G+M S+M O+M H+M`, `X+M`, `X-<m>+M` | S\* | S\* |
| `G+FS O+FS` | FS | S\* |
| `P0` / `P+M` (reference plans, no call) | none | prior / S\* |
| `G+M/p O+M/p` (cached +M plans, no call) | (in the plan) | prior |
| curve `G+M@o<o>k<K>`, `O+M@o<o>k<K>` (K in 1, 3, 5, 10) | o<o>k<K> | o<o>k<K> |

A18/A19 (2026-10-12): `X0` / `X+M` are Codex `gpt-6-sol` as a MAIN, executed comparator (blocks `main` and `repeat`, pass `xmain`, scope `free`,
never cut; Holm members H5a-c). `X-<m>0` / `X-<m>+M` exist only for the three exploratory Codex models (`C.CODEX_EXPLORATORY`). The cost guard pauses above
0.024 USD per Opus call (`C.OPUS_GUARD_USD_PER_CALL`, A19). G-iso (`execute.py --g-iso`, schema `ftm.g_iso/2`) compares components (i) request bytes,
(ii) status, (iii) goal reached, (iv) executed steps (action, object type, ok), (v) audit counts, under absent / present / absent; poses and timings are excluded.

K = 0 is the `0` arm. `o0k10` is S\*, so curve cells at (0, 10) are the main-matrix cells: shared, never re-called.

**Blocks** (`C.BLOCKS`): `main` (test, canon world, r = 0), `repeat` (REPEAT subset, r = 1, 2), `curve`,
`reloc` (relocated world, r = 0), `codex`.

**Ledger scope** (`C.scope_of(arm, block)`): `core` is the six core arms in `main` and `repeat`, including their
repairs. `other` is every other paid call (FS, Haiku, curve, reloc). `free` means no Anthropic spend (Gemma,
Codex, P, /p).

## 4. Ids and the execution unit

- Presentation: `C.pres_id(item, lang, world)` gives `"<world>:<item>:<lang>"`, with world `canon | reloc`.
- Cell ids: `C.cell_id(ns, phase, *parts)`.
  - capture: `cell_id(ns, "capture", pres, f"c{run}")`. Runs 1 and 2 are only for the dev 12.4 re-capture; the
    matrix uses run 1.
  - plan: `cell_id(ns, "plan", arm, pres, f"r{r}")`.
  - exec: `f"{ns}|exec|{unit}|run{run}"`.
  - score: `cell_id(ns, "score", arm, pres, f"r{r}", f"run{run}")`.
- Plans are compared after `C.norm_plan(steps)`, which gives `[[action, object, to, arg], ...]`. Order and every
  argument are kept.
- **Execution unit** (prereg s7 C): `C.execution_unit_key(plan=, pres=, chooser=, snapshot_sha=, restored_state=,
  summary_sha=, caps=C.CAPS, seed_policy=C.SEED_POLICY)`.
  - `restored_state` is the world id plus the setup sha that the restore produces: `"canon"` or `"reloc:<seed>"`,
    then `"+" + sha256(canon(setup))[:16]`. The executor builds it from the item; the planner never sees the setup.
  - `summary_sha` is the Phase A capture's `summary_sha` (run 1).
  - A prior chooser carries no snapshot sha.
  - Arms with equal unit keys share one execution.
  - REPEAT presentations: each distinct unit runs **3 times** (runs 0, 1, 2, each from an independent restore). Run 0
    is the scored one.
  - `C.CAPS` (12 plan steps, 3 retries per step, 3000 primitives, 900 s per episode) are recorded again in
    `freeze.json`.

## 5. Seeded orders (master seed 20261010)

- `C.rng(*labels)` gives a `random.Random` seeded from sha256 of the master seed plus the labels. It is stable across
  machines and python versions. Never use `random.seed()` or the global RNG.
- Phase B: `C.plan_blocks(pres_ids, cells_of, ns, block_name)` gives `[(pres, [cell, ...])]`. Presentations come in
  a seeded order. All cells (arm, r) of one presentation are one **block**, in a seeded order, so arms are interleaved
  in time and a stop between blocks leaves every arm balanced.
- Phase C: `C.exec_blocks({pres: [unit, ...]}, ns, block_name)` gives a seeded permutation of presentation blocks,
  with a seeded unit order inside each one.
- Practice order (M1): `C.rng("practice", order)`. Draw practice-pool items without replacement, round-robin by
  level (L1, L2, L3, ...), and draw each item's language with the same RNG (prereg 6.2).

## 6. Journals and record schemas

`C.Journal(path)`: `.append(rec)` (fsync'd), `.rows()`, `.keys(field="key")`, `.latest()`.

- A torn last line (a crash mid-write) is ignored on read and moved to `<path>.torn` on the next append.
- A torn line in the middle raises `JournalCorrupt`. That is fail fast.
- Every record is built with `C.make_record(kind, **fields)`. It adds `schema`, `spec` and `ts`, checks the required
  fields and legal status values, and enforces the **firewall**: no gold key (`goal gold setup reference_plan
  ref_plan goal_predicates`) and no outcome key (`goal_ok exec_ok sr strict_sr success step_ok steps_ok admission`)
  anywhere inside a capture, plan, exec, events or monitor record. Exec records also may not carry `steps`.
- Extra fields are allowed. The table lists the minimum.

| kind | required fields (beyond the envelope) | notes |
|---|---|---|
| practice | key, namespace, order, attempt, item, lang, level, status, snapshot_after | training data, so outcomes are allowed (admission verdict, plan, steps) |
| capture | key, namespace, pres, item, lang, level, world, capture_run, instruction, holding, summary (list of str), summary_sha, blocks `{source: {text, ids, real_tokens, budget, empty, coverage}}`, requests `{source: request_sha}`, state_ref (path under `states/`), world_hash (strict, diagnostic), audit | `instruction` is the verbatim EN / FA text. Sources are `none`, `S*`, curve sources (REPEAT presentations only) and `FS`. No evaluator label goes in the record; those are only in the state file |
| plan | key, namespace, arm, block, pres, r, session, request_sha, status (`ok rejected infra ceiling`), plan, first_plan, first_ok, repaired, calls, cached_calls, tokens_in, tokens_out, cost_usd, model, scope | also: say, normalised, decoding, attempts (raw, usage, latency, stop_reason, thinking tokens, served model). `session` 0, 1, 2 is the infra re-queue count (prereg s13) |
| exec | key, namespace, unit, run, attempt, pres, status (`done not_executed blocked_state crash invalid`), state_check `{ok, breaches}`, audit, prims, wall_s, snapshot_sha_before, snapshot_sha_after | integrity only. `outcome` holds the results |
| outcome (sealed) | key (same as exec), namespace, unit, run, goal_ok, exec_ok, steps | written only by `execute.py`. Read only by `score.py` |
| score | key, namespace, arm, pres, r, run, unit, sr, strict_sr, valid, infra | also: unrealistic_slice, pc_first, pc_final, tds_first, tds_final, level, lang, family |
| events | event, phase | for example `phase_complete`, `ceiling`, `pause_cost_guard`, `block_refused`, `abort_invalid`, `lock_stolen` |
| monitor | tick, counters | integrity counters only |

**Phase completion.** A phase that finishes all its cells appends `events {event: "phase_complete", phase, n,
journal_sha}`. Downstream phases refuse to start without it. The only exception is `score.py --partial`, which needs
a recorded stopping-rule event (prereg s13 "incomplete study").

## 7. The test-item refusal guard

`C.guard_items(item_ids, ns, allow_test=False)` must be called before any batch of capture, plan or execute, and
again on every resume. It refuses (`TestItemRefused`) in four cases:

- any item outside the namespace's set;
- any capped-out item;
- any test item in a namespace other than `test-v1`;
- any test item when one of these is missing: the `--allow-test` flag, a valid `freeze.json`
  (`C.freeze_status()`: schema `final-test.freeze/1`, self-sha, every gate `pass: true`,
  `arash_check: {by: "Arash", date}`, split sha, prereg sha equal to the file on disk, and every hashed code file
  unchanged), or `families_approved.json` with `approved_by: "Arash"`.

Every CLI that can touch test items exposes `--allow-test` and passes it through unchanged. A selftest must show the
refusal in each case. W5 is gated on Arash's freeze check, which only Arash can record (section 12, M3).

## 8. Gold isolation, masking, keys

- **Planner processes** (`plan.py` and anything it calls: Ollama, Anthropic, Codex temp dirs) read **only**
  `capture.jsonl`, `fs_blocks.json` (already inside capture) and the frozen type list. They never open
  `sim/dataset/general.json`, `states/`, `outcome.sealed.jsonl` or `score.jsonl`. `_ftm_plan_selftest.py` checks this
  statically (no such path strings, no `gold_of`, no `general_set` import) and at runtime (the poisoned-file test from
  M8). Codex temp dirs hold only the schema file (prereg 3.4).
- **Gold readers.** Only `execute.py` (in WSL: setup, reference plans for the P arms, goal predicates inside
  `h.scoring()`) and `score.py` (reference plans for PC / TDS, slice realism) read the gold. `capture.py` reads
  items for setup and text, but writes no gold field (the firewall in section 6).
- **Masking** (prereg s11): `C.mask_problems(text)` on every rendered prompt, before it is written to capture and
  again in `plan.py` before any call. Any hit aborts. `C.MASK` adds `+` to the registered objectId class, because ids
  carry signed coordinates (`Apple|+01.20|...`) and the registered pattern misses them. This is strictly tighter, and
  it is disclosed.
- **Keys** (prereg s15): `C.key_leak(text)` over every file a phase writes, before `phase_complete`.

## 9. Ledger, ceilings, cost guard (A15)

- `C.open_ledger(ns)` gives a `C.ScopedLedger` at `work/<ns>/ledger.db`, with the namespace ceilings. Pass it as
  `ledger=` to `planners.make_planner`.
- Every paid call runs inside `with ledger.block(scope, worst_usd):`. That call holds the block's worst case **before
  the block starts**, and refuses with `BlockRefused` if it would cross the scope ceiling or the total. Per-call
  reservations inside the block are checked without double-counting the hold, and the hold is released at the end.
- A paid reservation without a scope is refused. Holds left by a dead process are released when the ledger is
  reopened.
- Worst case of a block: `C.block_worst_case(n_calls, max reserve per call)`, which counts every first transmission
  plus one repair, each at `provider.reserve_usd(messages, None)`.
- At **block boundaries only**, run `C.cost_guard(opus_calls=, opus_usd=, checked_at=, spent=, remaining=,
  mean_cost=, ceiling=)`. Every 60 billed Opus calls it compares the mean with 2 x 0.0089 USD, and it checks that the
  projected completion stays under the ceiling. `pause` means: append `events pause_cost_guard`, stop with exit
  code 3, and wait for Arash. Resume is outcome-blind: only spend, tokens and integrity counters are looked at.
- `remaining` comes from M10's call plan (`final_test_callplan.py --json`), minus the done cells.
- The order of passes in `test-v1` is: core six r = 0, then core REPEAT r = 1, 2, then FS, curve, reloc, Haiku, Codex.
  Exploratory work is cut first (prereg 3.1).

## 10. Resume, locks, fail fast

- **One writer** per (namespace, phase): `with C.PhaseLock(ns, phase, steal=args.steal_lock):`. A stolen lock is
  logged as `events lock_stolen`.
- **Done** means that the latest record for the key has a terminal status. Resume skips done keys and redoes
  everything else. A crash mid-cell leaves no record, so the cell is redone.
  - plan: `ok` and `rejected` are terminal. `infra` is re-queued in up to 2 later sessions (`--requeue-infra`), and
    after the third it is terminal (scored 0, reported). `ceiling` is not terminal.
  - exec: `done` and `not_executed` are terminal. `crash` and `blocked_state` are re-run from a fresh restore, up to
    attempt 2, then they are terminal (a failure, reported). `invalid` (a broken honesty invariant) **aborts the phase**
    (`events abort_invalid`). It needs an amendment, then all arms of that presentation are re-run matched (prereg s13).
- A planner replay after resume comes from the response cache (`cached=True`, 0 USD) and is never a new observation.
- Fail-fast aborts: a snapshot sha change, a decision-path oracle read, a masking hit, a key leak, gold access by a
  planner process, a corrupt journal, or a capture / prompt sha mismatch between Phase A and Phase B.

## 11. Interfaces for the E stages

### E1 - A11 in `sim/step5_live.py` (the planner after the fixed start sweep)

```python
LiveExecutor(memory="none"|"places", ..., plan_after_sweep: bool = False,
             state_capture: bool = False, state_check=None)
executor.plan_hook(req, it, holding, ltm, post=None) -> {"steps": [...] | None, "info": {...}}
```

- `plan_after_sweep=False` is the old order. It must stay byte-for-byte as it is, so C3 / C4 stay reproducible, and
  `post` is then `None`. With `True`, the order is: reset, applicability, setup, `initial_state_hash`, build the agent,
  adapter, chooser and skills, `sk.start_episode(...)` (the fixed sweep), then `post`, then `state_check`, then
  `plan_hook`, then the steps. `run_episode_core` gets an optional `started=` (the `start_episode` result) so that it
  does not sweep twice.
- `post` dict (what the hook receives):
  - `summary`: `sk.om.summary()` right after the sweep. This list is the prompt's `memory_summary`.
  - `summary_sha`: `C.sha256_text("\n".join(summary))`.
  - `holding`: the told holding.
  - `events`: the adapter's sweep event stream.
  - `wm`: the live adapter's working memory, or `None` without LTM.
  - `episode_id`, `sweep_prims`, `seed_tracks`, `world_hash`.
- New helper `render_blocks(post, sources: dict[str, LongTermMemory], level, lang, channels, holding) -> {source: block}`
  renders the memory block of **several** frozen LTMs from the same post-sweep evidence. It feeds `post["events"]` into
  a fresh test-mode WM per source, works when the executor ran with `memory="none"`, never touches the live WM, and is
  deterministic. A selftest shows that for the live LTM it equals `ltm.memory_block(post["wm"], ...)`.
- `state_capture=True`: inside a `_score_post_sweep_state` function (`with h.scoring():`), call
  `reset_gate.capture_state(...)` (E3) and put the result in `res.extra["post_sweep_state"]`. It is **never** passed
  to the hook.
- `state_check(state, post) -> {"ok", "breaches"}` (M6 passes `reset_gate.compare` against the Phase A state). If it
  is not ok, nothing is planned or executed: `res.extra["blocked_state"] = breaches`.
- A plan that is not executed (rejected / infra) under the new order: no step runs, there is **no scoring call**
  (`goal_ok=False`, which is known because the goal is false at reset), the episode closes normally with
  `exec_ok=False`, and `res.extra["plan_not_executed"]=True`. In practice it counts as a started attempt, and its
  honest sweep sightings are consolidated with admission False. **E1: confirm this one point with Astra**, because it
  changes what a rejected practice attempt leaves in memory compared with the pre-A11 C3 runs.
- Gates: `_step5_adapter_selftest.py` (static honesty), plus a new `sim/_step5_live_a11_selftest.py` (old order
  unchanged; new order sweeps once; the hook sees the summary; `render_blocks` equality; no state reaches the hook).

### E2 - A12 retention in `sim/ltm.py` and `sim/ltm_retrieval.py` (+ `sim/final_test/matrix/retention_a12.py`, `sim/step5/params.json`)

- Pure: `ltm_retrieval.a12_allocation(H, r, N={"L1":1,"L2":2,"L3":3}, B={...}, where_share, cap_factor=2.0) ->
  {"A": {L: int}, "B": {L: int}, "raised": {L: [old, new]}, "incapable": [L, ...]}`. A_L = max(where_share x B_L,
  H + N_L x r). If H + N_L x r > B_L, raise B_L to the smallest multiple of 10 that fits, at most 2 x the registered
  value, then recompute A_L. If it still does not fit, the level is `incapable`.
- `LongTermMemory.memory_block(..., retention=None)`: `None` reads `params["retrieval"]["retention"]` (absent means
  `"c2e"`, the old behaviour). With `"a12"`, places get A_L, procedures get B_L - A_L, the header is paid once, the
  counter is the max-real one, and record ids are identical across models and languages. Then a **complete-prefix
  check**: the real count of `MEMORY_HEADER + "\n" + text` must be <= B_L in every tokenizer x language. If not,
  shrink the prefix and record `prefix_shrinks`.
- `ltm.a12_coverage(blocks_by_lang, n_place_candidates, level, counter, alloc) -> {"ok", "N", "held", "exempt",
  "real"}`. It is ok iff fewer than N_L candidates exist, or the block holds >= N_L whole place records in every
  tokenizer x language. Incapable levels are `exempt`.
- Counter: `final_test/c3/real_counter.RealMaxCounter`, unchanged and frozen, with the cache at
  `work/<ns>/tokcache.json`.
- CLI `retention_a12.py --measure --capture <rehearsal capture.jsonl> --c3 docs/final-test/c3/eval.jsonl [--write]`.
  It measures H and r (max over tokenizers x languages, separators included), computes the allocation, runs the
  coverage check on all 24 re-captured dev presentations and the C3 blocks, and writes
  `docs/final-test/matrix/a12_measure.json`. `--write` adds `retrieval.retention = "a12"` and `retrieval.a12 = {H, r,
  N, A, B, raised, incapable, reason: "A12, token-derived", fp1_measured: true}` to `params.json` (copy the old file
  to checkpoints first; the old sha is `e9dd82146ec3bd89`). It needs Ollama and free count_tokens (W2).
- Gate: `sim/_step5_retention_a12_selftest.py`, plus every existing `_step5_*` selftest still green.

### E3 - gate 12.4 in `sim/final_test/reset_gate.py` (+ `reset_calib.py`, `docs/final-test/reset_tolerances.json`)

All functions are pure over plain dicts; the executor reads the world for them inside `h.scoring()`.

```python
capture_state(objects: list[dict], agent: dict, *, summary: list[str], seen_types: list[str],
              labels: dict, feasibility: dict) -> dict      # schema "final-test.reset-state/1"
feasibility_of(state: dict, ref_plan: list) -> dict          # per reference step: precondition verdicts (A14 c iv)
compare(a: dict, b: dict, tol: dict) -> {"ok": bool, "breaches": [...], "max_dpos": float, "max_drot": float}
calibrate(restores: dict[str, list[dict]], ceilings: dict) -> dict   # tau = observed max rounded up to 0.05 m / 5 deg; fail above a ceiling
load_tolerances(path=None, require="calibrated") -> dict
write_tolerances(d, path=None)                                       # self-sha (C.self_sha)
```

- **Identity.** Use the object `name`, not the `objectId` (`InitialRandomSpawn` renames ids; see the AI2-THOR quirks).
- **Symbolic part** (must be identical): type, receptacle parents, containment / support, open / toggled / sliced /
  dirty / filled / broken, accessibility, the set of types seen in the sweep, the summary (byte-identical), the labels
  (evaluator labels at reset) and the feasibility verdicts.
- **Continuous part**: per-object position and rotation within tau. The agent pose must match exactly.
- `reset_tolerances.json`: `{schema: "final-test.reset-tolerances/1", amendment: "A14a", ceilings: {pos_m, rot_deg,
  justification, dated}, calibration: {items, restores_per_item: 5, pairs, observed_max_pos_m, observed_max_rot_deg},
  tau_pos_m, tau_rot_deg, agent_pose: "exact", status: "ceilings" | "calibrated" | "failed", validation: {pairs: 24,
  pass, ok}, sha256}`.
- The ceilings come from a dated prereg amendment **A14a**, written before any calibration restore. E3 drafts the
  numbers and their geometric justification (grid step, interaction distance, detector visibility rule) in its notes.
  The amendment text is appended by whoever owns the prereg text in that workflow, not by E3's code.
- `reset_calib.py` (WSL, run in W2 by the stage that owns the simulator) does 12 dev items x 5 restores, then the
  24-pair validation.
- Gate: `sim/_ftm_reset_gate_selftest.py` (synthetic states: a settled loose object above or below tau, a changed
  parent, a changed open flag, renamed ids, agent pose drift).

### E4 - H4 (EN / FA record-id mismatch on `g-l2-empty-mug-sink`, C3)

- Offline analysis of `docs/final-test/c3/eval.jsonl` blocks and `ltm.concepts` on both texts:
  `sim/final_test/h4_enfa.py`, writing `docs/final-test/h4_enfa.json` and a ledger entry.
- Show the cause, and show that it cannot corrupt (a) EN / FA pairing (the item score is the mean of the twins),
  (b) retrieval (ids identical by construction under C2-e / A12) or (c) scoring.
- If it is a retrieval bug, propose the fix to E2 in your notes. Do not edit ltm.
- Gate: `sim/_ftm_h4_selftest.py`.

## 12. Interfaces for the M stages and the integrator

Each CLI takes `--namespace`, `--allow-test`, `--steal-lock`, and `--dry-run` where it makes sense.

- **M1 `practice_k.py`** (WSL, Gemma; `--orders 0,1,2 --k 0,1,3,5,10 --namespace practice-v1`).
  - Practice follows prereg 6.2, on G+M settings with the live practice LTM read online (C2-f), under the A11 order,
    with `PlannerRunner` + `PlannerContract` and the cache namespace `practice-v1`.
  - K counts started attempts. The snapshot after attempt K goes to `C.snapshot_dir(f"o{o}k{K}")` (K = 0 is an empty
    LTM). Each snapshot has a manifest `{order, k, state_hash, params_sha, channels, attempts: [item ids], leakage: {ok,
    hits}, sha256}`.
  - Leakage gate: no test item id and no normalised test instruction (EN / FA) in the snapshot dump.
  - The practice cost is recorded: episodes, primitives, wall time, Gemma calls.
- **M2 `fs_block.py`**: `render(practice_rows_o0_k10, level, counter, budget) -> {text, real_tokens, retained:
  {attempts, outcomes, sightings}, shortened}` per prereg 6.4. The budget is the final A12 B_L. `--write` writes
  `fs_blocks.json` `{L1, L2, L3, source_sha, counter, sha256}`. The block is identical for G and O.
- **M3 `freeze.py`**.
  - `--write` builds `freeze.json`: `{schema: "final-test.freeze/1", created, prereg_sha256, split_sha256,
    families_approved_sha256, code: {relpath: sha}, prompt_version, system_text_sha, plan_schema_sha, params_sha,
    a12, detector_weights, map_sha, snapshots: {"S*": {...}, curve...}, fs_blocks_sha, reset_tolerances_sha, caps,
    reloc_seed, models, ollama: {digest, version}, codex_cli_version, gates: {id: {pass, evidence}}, arash_check:
    null, sha256}`.
  - `code` lists every `sim/` module on the planning and execution path (import closure of capture, plan, execute
    and score, plus `final_test/c3/real_counter.py`).
  - Gates: 12.1-12.13, G-iso, A12 coverage, A14 validation, snapshot hash, leakage.
  - `--check` re-verifies. `--record-arash-check` is run **only** on Arash's explicit instruction; it fills
    `arash_check` and re-seals the file. It refuses if the gates are not all passed.
- **M4 `capture.py`** (WSL; `--items dev|test --runs 1[,2] --worlds canon[,reloc]`).
  - Uses `LiveExecutor(memory="none", plan_after_sweep=True, state_capture=True)` and a capture hook that returns
    `steps=None`.
  - Per presentation it renders, via `render_blocks` and `PlannerContract`, the requests for `none`, `S*`, `FS` and,
    for REPEAT presentations, the curve sources. It checks masking, runs the A12 coverage check on `S*`, writes the
    state file to `states/<pres>.c<run>.json`, and appends the capture record.
  - The relocation world uses `reloc_seed` from freeze (or the rehearsal value).
- **M5 `plan.py`** (Windows; `--pass gemma|core|fs|curve|reloc|haiku|codex [--r 0,1,2] [--requeue-infra]`).
  - Reads capture run 1 only. It re-renders each prompt and asserts the sha equals the captured request sha, then
    applies masking.
  - Loops over `C.plan_blocks`, each block inside `ledger.block(...)`, with `PlannerRunner(replicate=r)` and the
    response cache under `work/<ns>/cache` and the namespace name.
  - Runs `C.cost_guard` at block boundaries and appends plan records.
  - Ollama is used only for the Gemma pass (check whether it is running first; stop it only if you started it).
  - Offline selftest with a fake provider. Never a real paid call in the selftest.
- **M6 `execute.py`** (WSL).
  - Builds units from plan `ok` rows, the P arms (reference plans) and the /p replays. Dedups by unit key. Runs 3 times
    on REPEAT. Uses the `C.exec_blocks` order.
  - Each run uses `LiveExecutor(memory=..., plan_after_sweep=True, state_capture=True, state_check=<reset_gate.compare
    vs Phase A state>)` with a replay hook. The hook checks `post["summary_sha"]` against the capture.
  - Takes the snapshot sha before and after every LTM run, writes exec + outcome records, and handles crash / blocked
    re-runs (max 2) and `invalid` aborts.
  - Also owns the **G-iso** audit (`--g-iso`: the `0` arms on dev, with no `ltm.db` and then with S\* +
    `honest_memory.db` present; traces, prompts and outcomes must be identical).
- **M7 `score.py`** (Windows).
  - Needs `phase_complete exec`. Maps each plan cell to its unit and outcome. Rejected / infra / unexecuted plans get
    SR 0, with the flags set.
  - Computes strict SR, `unrealistic_slice` (no Knife / ButterKnife held at the slice step) and PC / TDS with
    `eval.pc_score` / `tds_score`, with `search_for look_around scan_for` stripped from both sequences.
  - Writes `score.jsonl` and `score.seal.json` `{sha256 of score.jsonl, n, complete, created}`.
  - Also owns **12.5** (`--evaluator-check`: reference plans on dev reproduce step 4 up to known variance; the empty
    plan scores 0).
- **M8 `monitor.py`** (`--once | --every 1200`).
  - Counters: violations, oracle / map-build reads, snapshot sha changes, state-check breaches, crash / infra rates,
    spend per scope vs projection, progress per phase.
  - Never opens `outcome.sealed.jsonl` or `score.jsonl`. The selftest proves it with poisoned files that raise if read.
- **M9 `analyze.py`** (`--synthetic | --sealed <ns>/score.jsonl`).
  - Verifies the seal. Computes H1 (`final_test_power.h1_decision` with `cluster_ci` and `wcr_p`), the Holm family
    (H2a-c, H3a-c, H4a-b), the family-weighted sensitivity, the paired infra-failure sensitivity, per level / language
    results, the REPEAT variance decomposition, and cost per success (ledger + practice cost). The Codex table is kept
    separate.
  - `--synthetic` runs the whole analysis on generated outcomes with known effects, first.
- **M10 `sim/final_test_callplan.py`**: remove the OpenRouter rows (A6). Use `C.ARMS`, `C.BLOCKS` and `C.scope_of`
  names. `--json` gives the cells per (block, arm, scope) that M5 uses for `remaining`.
- **Integrator `matrix_run.py`**.
  - `--phase capture|plan|execute|score|monitor|analyze --namespace ns --items dev|test [--allow-test]`.
  - Preflight: `guard_items`, freeze status for test, `phase_complete` dependencies and the lock. Then it dispatches.
  - `--inject-interrupt N` stops a phase after N cells, for the rehearsal resume test.
  - `sim/_final_matrix_selftest.py` runs every `_ftm_*` selftest plus an offline end-to-end with fakes: capture rows,
    then plan with a fake provider, then execute with a fake executor, then score, then analyze `--synthetic`, with an
    interrupt and resume in each phase.

## 13. Ownership

| id | owns (edit only these) | selftest |
|---|---|---|
| D-spec | `sim/final_test/matrix/{SPEC.md,common.py}` | `sim/_ftm_common_selftest.py` |
| E1 | `sim/step5_live.py` | `sim/_step5_live_a11_selftest.py` (+ `_step5_adapter_selftest.py` green) |
| E2 | `sim/ltm.py`, `sim/ltm_retrieval.py`, `sim/final_test/matrix/retention_a12.py`, `sim/step5/params.json` (A12 values, in W2) | `sim/_step5_retention_a12_selftest.py` |
| E3 | `sim/final_test/reset_gate.py`, `sim/final_test/reset_calib.py`, `docs/final-test/reset_tolerances.json` | `sim/_ftm_reset_gate_selftest.py` |
| E4 | `sim/final_test/h4_enfa.py`, `docs/final-test/h4_enfa.json` | `sim/_ftm_h4_selftest.py` |
| M1 | `sim/final_test/matrix/practice_k.py` | `sim/_ftm_practice_k_selftest.py` |
| M2 | `sim/final_test/matrix/fs_block.py` | `sim/_ftm_fs_block_selftest.py` |
| M3 | `sim/final_test/matrix/freeze.py` | `sim/_ftm_freeze_selftest.py` |
| M4 | `sim/final_test/matrix/capture.py` | `sim/_ftm_capture_selftest.py` |
| M5 | `sim/final_test/matrix/plan.py` | `sim/_ftm_plan_selftest.py` |
| M6 | `sim/final_test/matrix/execute.py` | `sim/_ftm_execute_selftest.py` |
| M7 | `sim/final_test/matrix/score.py` | `sim/_ftm_score_selftest.py` |
| M8 | `sim/final_test/matrix/monitor.py` | `sim/_ftm_monitor_selftest.py` |
| M9 | `sim/final_test/matrix/analyze.py` | `sim/_ftm_analyze_selftest.py` |
| M10 | `sim/final_test_callplan.py` | `sim/_ftm_callplan_selftest.py` |
| integrator | `sim/final_test/matrix/matrix_run.py`, `sim/_final_matrix_selftest.py` | `sim/_final_matrix_selftest.py` |

Shared, and owned by no W1 builder (do not edit): `common.py` (D-spec; ask for changes in your notes),
`sim/planners/*`, `sim/final_test/c3/real_counter.py`, the preregistration, `split_manifest.json`,
`families_approved.json`, `docs/thesis/`.

**Build order.** The interfaces above are fixed, so all stages can build at the same time. M4 and M6 depend on E1 /
E3 only through the signatures here, and they test with fakes. The integration happens in `_final_matrix_selftest.py`
and later in the W2 rehearsal.

Gates for W1 (all offline): `python sim/_final_matrix_selftest.py`, `python sim/_planners_selftest.py`, every
`python sim/_step5_*_selftest.py`, `python sim/_step4_gemma_selftest.py` and `python sim/_step4_rules_selftest.py`.

## 14. Open points (decide or confirm, not built into common)

1. **E1 + Astra**: what a rejected plan leaves in memory under the A11 order (section 11, E1).
2. **A14a** ceilings (numbers + justification): drafted by E3, appended to the prereg as a dated amendment before W2's
   calibration restores.
3. **12.4 labels / feasibility**: E3 defines `feasibility_of` from the symbolic state only, so no simulator action is
   needed. If a check would need an action, it is out of scope; report it.
4. **Mask pattern**: common's `+` fix (section 8) should be named in the thesis disclosures, next to the pilot
   masking result.
5. W6's command in the C5 ledger (`analyze.py --sealed docs/final-test/matrix/score.jsonl`) becomes
   `--sealed docs/final-test/matrix/test-v1/score.jsonl`, because of the per-namespace folders.

## 15. M-fix2 changes (2026-10-11, before any test-item call)

Fixes from the three matrix reviews (fairness, statistics, robustness). Each is covered by `_final_matrix_selftest.py` (block "M-fix2").

1. **FS source.** `fs_block` takes practice_k's real journal: attempts numbered 1..K, sightings `{otype, in_rtype, state}`, a crash /
   rejected / inapplicable started attempt gets the verdict `crash` / `failure` / `inapplicable`. A sighting with no recognised key is refused.
2. **Independent planner calls.** The Phase B cache key also hashes the cell (`plan.CellCache`): every arm x r is its own call, so an empty
   S* or curve block can no longer hand the second arm of a presentation the first arm's answer. Resume replays a cell's own answer.
3. **Exploratory cut.** `plan.py --cut codex,haiku,curve12,reloc --note "<why>"` (Arash only, registered order, core / gemma / fs cannot be
   named) records `exploratory_cut`. Cut cells leave plan (`phase_complete` carries `cut_groups` / `cut_cells`), execute and score (the seal
   records `cut_groups` / `n_cut_cells`); they are never scored 0. `curve12` = the order-1 and order-2 learning-curve arms only.
4. **Replay rows.** execute writes the `/p` cells of rejected / infra +M plans to `units.json` (`unexecuted_replay`); score gives each an
   SR-0 row (`plan_status` of the replayed plan). `units.json` also carries `cut_groups` and `repeat_rule`.
5. **Unrecoverable INVALID presentations.** `execute.py --declare-unrecoverable <A-id>` records `invalid_unrecoverable`; execute stops waiting
   for it, score flags every row `invalid_episode`, analysis drops it from every arm. `valid` still means "reached scoring".
6. **Analysis.** An incomplete seal prints `incomplete study: no confirmatory H1 decision (prereg s13)` and rejects nothing. `superior`
   needs non-inferiority too (fixed sequence, applied in `analyze.fixed_sequence`). A complete test-v1 seal must cover every core and FS arm for
   every registered item x language. Per arm: prerequisite-realistic SR and the slice-flag counts (True / unknown / slice rows).
7. **Stopping rule.** Only `stopping_rule` / `incomplete_study` events (`plan.py --declare-incomplete`, `by: Arash`) allow `score --partial`.
   `pause_cost_guard` and `ceiling` do not.
8. **Dataset pin.** `freeze_artifact_problems` checks the dataset sha; capture (test-v1: manifest + freeze; every namespace: records it in
   `capture_start`) and execute (compares with that record) abort when it moved. Execute also checks `restored_state` against the capture row.
9. **Capture.** A12 coverage in test-v1 fails if any presentation is unchecked (and the run aborts without `retrieval.a12`); the channels must
   equal `freeze.json["channels"]` (freeze records them from the S* practice manifest); a setup failure is re-run twice, then a CANONICAL
   `inapplicable` presentation fails gate `applicable` (no `phase_complete`), a RELOCATED one is excluded from every arm (listed in
   `phase_complete.reloc_inapplicable`). `--allow-missing-anthropic` is refused in test-v1; tokenizers are recorded. `matrix_run` forwards the study key
   to WSL through WSLENV (`common.wsl_key_env`, value never printed) and requires `--ollama` for capture.
10. **Budget.** Worst-case holds are per CELL, not per presentation (a 4-cell core block held ~1.2 USD against a 1 USD rehearsal ceiling).
    Pauses stay at presentation boundaries. While core cells remain, the `other` scope may use only what the core ceiling leaves unreserved
    inside the total (`ScopedLedger(core_done=...)`). Default `--pass` is `gemma,core` (never mixes paid scopes). Stale open per-call
    reservations are listed by the monitor and released by `plan.py --reconcile-open "<note>"` (audited). `SEED_POLICY` is `none`
    (the honest executor takes no simulator seed; `world_seed` is the label 0 in capture and execute).
11. **Keys.** The study key is `THESIS_ANTHROPIC_API_KEY` only (`common.thesis_key`); a paid pass without it is a clean skip (exit 2).
12. **Dev REPEAT / relocation sets.** `common.repeat_items` / `reloc_items`: test-v1 = the split's sets; rehearsal-v1 = a declared seeded dev
    subset of 4 items each, so W2 can interrupt and resume the REPEAT, curve (if snapshots exist), Codex, three-run and relocation paths.
13. **Small.** `execute` indexes the exec journal once (`cell_state` no longer re-reads it); structural-vs-free-text key-leak rule shared in
    `common.leak_paths` (a hyphenated word in a step cause is no leak; the exact secret is); `final_test_callplan.py --remaining <ns>` feeds the monitor.
14. **Declared, needs a dated amendment before W5.** (a) REPEAT x3 applies to the units of the `main` and `repeat` blocks; curve / Codex / reloc
    units run once (`units.json.repeat_rule`; 3x would add ~2,900 simulator runs of exploratory arms). (b) A canonical presentation that is
    inapplicable after two re-runs stops Phase A (no silent exclusion); a relocated one is excluded and listed.

## A21 additions (stage G-blockers, 2026-10-12; the registered text is prereg A21)

1. **Blocked and failed runs (A21 A).** `common.BLOCKED_RULE`, `RUN_CLASSES`, `run_class()`, `INFRA_FLAG_RATE`. Score rows carry `run_class`
   (`infra_planner`, `infra_exec`, `agent`, `ok`, `budget`, `missing`), `exec_attempt`, `x_tool_events`, `x_identity_problem`. `analyze.is_infra`
   covers planner transport failure AND execution infrastructure failure; the paired sensitivity drops either-arm infra from both arms; an arm above
   5 % infra cells is flagged `infrastructure_sensitive`.
2. **Cost guard (A21 B).** `common.COST_GUARD_SPEC` (hashed in freeze), `reconcile_spend`, `cost_guard(..., opus_recent, ledger_usd, journal_usd)`.
   plan.py counts BILLED Opus cells only (cost > 0), keeps the per-cell cost list and the journal total, checks the cumulative mean and the last-60 window
   every 60 billed cells and the ledger-vs-journal reconciliation at every boundary. `--ack-pause NOTE --pause-response continue|cut|stop`, and in
   test-v1 `--outcome-blind` plus a note of at least 20 characters (`Plan._ack_problem`).
3. **Replay precondition (A21 C).** `execute.actual_holding`, `told_matches_actual`, `replay_precondition(state_a, state_now, told_a, told_now, tol)` =
   `reset_gate.compare` + actual holding equal + told holding equal and consistent. `state_check` runs it before the hook; the hook refuses to run a plan
   unless it passed. Fault injection: `_ftm_execute_selftest.py` (synthetic kitchen and the 24 real Phase A states).
4. **Sharing (A21 D).** `common.UNIT_KEY_SPEC`, `UNIT_KEY_GOLDEN`, `unit_key_selfcheck()` (the key function is frozen). analyze.py: `dependence_block`,
   `_dep_row`, `execution_block` (wall time split among the cells of a shared run), `execution_link_problems` (a complete journal with an unlinked score
   row is refused), the distinct-units-only sensitivity.
5. **X arms (A21 E).** `planners/codex_provider.py`: `HARDEN_FEATURES` (16), `SESSION_LOG`, `parse_session`, `identity_problems`, `tool_events`,
   `pin_cli_version`, extra keys `cli_configured_model` (`served_model` stays None), `session`, `identity_violation`, `identity_problems`. plan.py aborts
   with `abort_codex_identity` and takes the pin from freeze.json in test-v1. `x_isolation_probe.py` writes `docs/final-test/matrix/x_isolation.json`;
   `gate_evidence.py` turns it into the gate `X-iso` (freeze.py `EVIDENCE_GATES`, `GATE_ORDER`). `analyze.X_DISCLOSURE` is printed with every X result.
6. **Analysis plan (A21 F).** `analyze.ANALYSIS_PLAN` / `analysis_plan_sha()` (in freeze and in every analysis.json), `LOW_DISCORDANCE` = .25 guard for
   H3c / H5c (p := 1), `holm_m8_note`, H5c p from the -5 pp boundary. `sim/final_test_power_secondary.py` (power at the first Holm step; `--ni-calib`
   for the guard study) writes `power_secondary.json` and `ni_calibration.json`.
7. **Projection (A21 G).** `final_test_callplan.py --final-projection` (`final_projection()`, `SCENARIOS`, `measured_blocks()`, `observed_sharing()`)
   writes `docs/final-test/matrix/final_projection.json`.
8. **freeze.json** now also pins `cost_guard`, `blocked_run_rule`, `dedup`, `analysis_plan`, `power_secondary_sha256`, `final_projection_sha256`,
   `x_arms` and the 12-arm roster. It was NOT rewritten in this stage and is stale.
