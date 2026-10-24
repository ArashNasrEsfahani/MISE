# Per-kitchen perception datasets (step 3 of 7) — design, status, how to resume

**STOPPED 2026-10-04 20:35 (Arash: GPU full).** Everything closed (console, simulator, Ollama's llama-server, WSL).
Step 3 for FloorPlan1 is complete incl. the state estimator (see `docs/live-view.md` "State estimator in the console").
Console fixes today: two detectors + states in the detect view, render watchdog (auto-restart on red/blue/white
glitch). Restart the console with `START-TESTING.bat` (or `wsl -d Ubuntu-22.04 -- bash ~/run_webapp.sh`, then
http://<wsl-ip>:8000 — close old tabs first). Next: Arash tests the console, then step 4 on his go.

**STEP 3 COMPLETE FOR FLOORPLAN1 (2026-10-04 18:00).** Results page: `docs/fp1-detector-results.html`
(published artifact https://claude.ai/artifact/LhYyUiqu1KQt8oiPmvvCES). Sealed holdout (10 episodes, 526 held targets):
held-target top-label success primary .483 | control .791 | **gated primary+specialist .924** (+.441 [.393, .485] vs
primary; +.134 [.101, .163] vs control); held non-target FP .000 / .017 / .017. Main-test all-frame AP approximately
retained (all-49 .907 -> .908). Frozen weights `sim/weights/v3/FloorPlan1.{yolo26s,amb26m,yolo26s.control}.pt`
(sha256 in `sim/kitchen_v3/FloorPlan1/frozen_weights.sha256`). Console uses both (API 6), checked live. Open: state
estimator (needs Arash's hand labels, page `sim/kitchen_v3/FloorPlan1/state_review/index.html`). Next: step 4
(movement/interaction), FloorPlan1 only, on Arash's go.

**DECISION 2026-10-04 (Arash): FloorPlan1 only through ALL remaining steps (3-7); the other kitchens come after the whole system is finished.**

## 2026-09-30 — v3: FloorPlan1 recaptured FROM SCRATCH (supersedes everything below for planning)

**Why.** Arash watched the v2c capture: "the same object from the same place in the same angle several times";
"too much red"; wants "a lot of data for each object with different backgrounds", "most data in the original
colours", "each object held with many different backgrounds". Measured on v2c FloorPlan1: only 38 % original colours
+ lighting, 30 % held frames, Potato held 514x vs Spatula 18x, up to 9 pickups of one object per episode, 5.1 %
near-duplicates. **Decision (Arash):** "recapture from scratch. Perfect our method on kitchen 1, create and train
BOTH models and datasets; if it beats my paper (81.6 primary / 93.8 secondary mAP50), apply the same method to the
other kitchens." v2c data (`~/thor-data/kitchen_v2c/`) is superseded — do not train on it.

**What changed (dataset `kitchen_v3`, policy `v3-1`, quota `v3-1`; `kd.CAPTURE_V2C` keeps the old config):**
- no material randomisation; brightness-only lighting (0.85-1.15) on 20 % of episodes;
- 128 train / 20 val / 20 test episodes; type target 1,500 train labels; >= 150 distinct viewpoints per type
  (cell = 0.5 m x 0.5 m x 45 deg yaw, `view_cell`) earns training-frame gain;
- `held_views` rewritten: per-INSTANCE held deficit (60 frames, 30 distinct background cells), ONE pickup per
  instance per episode, each view from a standing spot far from that object's earlier held spots, yaw chosen to face
  >= 5 other labelled objects, object rotated AND moved across the frame; held frames capped at 12 % per episode;
- `check_capture` new invariants: repick (FAIL), held_share (FAIL), colours (FAIL on any material episode),
  held_context, held_backgrounds, near_duplicates (FAIL > 4 %).
- Driver: `sim/kitchen_v3/capture_all.ps1` (FloorPlan1 only) -> `~/capture_one_v3.sh`; log
  `sim/kitchen_v3/capture_all.log`; data `~/thor-data/kitchen_v3/FloorPlan1/`. Resume = rerun the driver: every
  completed episode is saved (frames.jsonl + capture_state.json) and an interrupted one is redone from its seed.

**STATUS 2026-09-30 02:18 — STOPPED by Arash ("save and stop").** FloorPlan1 v3: **49 of 168 episodes done**
(0-48; 5,390 frames; 0 failed), verified with `check_capture`: no FAIL, 0 missing files, 0 orphans. Original colours
82 %, held 13.1 %, 0 repicks, near-duplicates 0.2 %, all 77 instances in every split. Lagging: Cup/Pot held
backgrounds (4/6) and rare states (Egg broken=0, Potato sliced) — the latter are top-up work.
Fixes made during the run (all in `build_kitchen_data.py`, selftest 139/139): (1) a dropped object settles
(5 x Pass) before the next view; (2) a view or paired flip render whose renders disagree then settle, AND do so again
at the same view in a fresh process, is scene MOTION, not a glitch — accepted (episode 14 view 107 had burned 3
redos); (3) policy v3-2 (from episode 32): held pick weight = deficit squared, and a held target inside a closed
receptacle has the receptacle opened before pickup.
**RESUME:** run `sim/kitchen_v3/capture_all.ps1` (hidden:
`Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"sim\kitchen_v3\capture_all.ps1"`);
it resets WSL, re-probes (~1 min, looks frozen), and continues at episode 49.

**STATUS 2026-09-30 17:21 — FloorPlan1 v3 main capture COMPLETE.** 168/168 episodes, 0 failed, 27 glitch/motion
redos; **18,465 frames (train 14,065 / val 2,200 / test 2,200), 283,391 train labels**; `check_capture`: no FAIL.
Original colours + lighting 78 %; held 13.2 %; 0 repicks; **held per object 57-64 frames against 54-59 distinct
backgrounds for every one of the 31 pickupable objects**; 82/82 state combinations at the floor; viewpoints >= 150
for 47/49 types (SinkBasin 71, StoveKnob 115). Open: 6 small types under 1,500 train labels (Fork 899, Spoon 1,118,
CreditCard 1,161, Knife 1,289, Egg 1,434, Spatula 1,439); rare states (Potato cooked/sliced, Egg broken=0: 36-69 of
150 crops; val/test thinner) — matter for the state estimator, not the type-only detectors; near-duplicates 2.4 % in
the last 8 episodes (late, saturated episodes). NOT yet run: top-up + close protocol, stage_yolo, training.

**STATUS 2026-10-01 00:50 — STOPPED by Arash during YOLO26s training.** Balance done (top-ups 24 + close 12+12;
yolo_primary: train 16,299 / val 2,200 / test 2,200 / val_close 1,320 / test_close 1,320; detection gate PASS). Smoke
(1 epoch) passed: TEST mAP50 .732, LIVE .944 (10 poses). YOLO26s stopped after **48/60 epochs** — best VAL epoch 39:
mAP50 .906 / mAP50-95 .803. Checkpoint saved at `~/thor-data/kitchen_v3/_ckpt/FloorPlan1.yolo26s/` (epoch 48,
results.csv). Recipe now 60 epochs / patience 20 (same gradient steps as the reviewed 150 on v2c; speed measured).
**RESUME:** run `sim/kitchen_v3/train_only.ps1` (hidden Start-Process) — it restarts `ckpt_backup.sh`, resumes
YOLO26s at epoch 49 from the checkpoint, evaluates (test / close / 150 live poses), then trains + evaluates YOLO11s.
A model with `sim/kitchen_v3/FloorPlan1/eval_<model>.json` is skipped.

**STATUS 2026-10-01 16:02 — STOPPED by Arash during YOLO11s training.**
- **YOLO26s DONE** (60 epochs, best VAL epoch 52 mAP50 .905 / 50-95 .806): **TEST mAP50 .907 / 50-95 .803**, CLOSE-TEST
  .913 / .821 (beats the paper's 81.6 primary — different dataset, indicative only). Weakest TEST AP50: Fork .58,
  Spoon .61, ButterKnife .64, Spatula .68, PepperShaker .74, Knife .74, SaltShaker .75 -> disambiguation candidates.
  Its LIVE number (.958) is INVALID — only 10 poses (a smoke run froze a 10-pose list on the real root; file moved
  to `live_eval_poses.smoke10.json`, code now regenerates a short list). Re-run: `train_kitchen.py eval --scene
  FloorPlan1 --model yolo26s` after YOLO11s has drawn the 150 poses.
- **YOLO11s at 26/60 epochs** (best VAL epoch 25 mAP50 .894 / 50-95 .775); checkpoint `~/thor-data/kitchen_v3/_ckpt/
  FloorPlan1.yolo11s/`. RESUME = run `sim/kitchen_v3/train_only.ps1` (skips YOLO26s, resumes YOLO11s at epoch 27).

**STATUS 2026-10-02 01:25 — FloorPlan1 PRIMARY MODELS DONE** (both 60 epochs; same data, recipe, 150 frozen live poses):

| model | TEST mAP50 / 50-95 | CLOSE-TEST | LIVE (150 poses) |
|---|---|---|---|
| **YOLO26s** (MuSGD, NMS-free) | **.907 / .803** | **.913 / .821** | **.927 / .811** |
| YOLO11s (SGD) | .900 / .790 | .908 / .813 | .909 / .762 |

YOLO26s is the primary for FloorPlan1. Paper primary (YOLOv5s, 69 classes, other data): .816 mAP50 — indicative only.
Weakest TEST classes in BOTH models: Fork, Spoon, ButterKnife, Spatula, Knife, SaltShaker, PepperShaker
(AP50 .56-.77) -> disambiguation set. Caveat: live "classes_scored" differs (40 vs 31) — check before citing live.
Weights: `sim/weights/v3/FloorPlan1.{yolo26s,yolo11s}.pt`; reports `sim/kitchen_v3/FloorPlan1/eval_*.json`.

**STATUS 2026-10-03 18:02 — STOPPED by Arash during the ambiguity (specialist) training.** Spec + amendment:
`docs/ambiguity-spec.md` (Astra step3j-o).
- Coverage live set rebuilt (300 poses, near-training views): YOLO26s .971 / .822, YOLO11s .976 / .814 — sanity
  check only; realistic set too thin (118 boxes) — neither cited as generalisation.
- VAL error breakdown of the primary: `sim/kitchen_v3/FloorPlan1/val_errors_yolo26s.json` (oracle relabel .910->.936).
- Ambiguity capture DONE: `~/thor-data/kitchen_v3_amb/FloorPlan1/` — 69/70 episodes (3024 failed), 6,880 frames
  (held target 3,726); yolo_amb train 3,899 / val 997 / test 1,009; holdout 975 sealed (manifest only).
- Specialist YOLO26m learning curve: ep10 DONE (VAL mAP50 .705 / 50-95 .615, weights `sim/weights/v3/
  FloorPlan1.amb26m.ep10.pt`); ep20 stopped after 2 epochs (checkpoint `~/thor-data/kitchen_v3/_ckpt/FloorPlan1.amb26m.ep20/`).
- **RESUME:** run `sim/kitchen_v3/amb_train.ps1 -Cmd lcurve` (resumes ep20, then ep39), then `-Cmd control`, then
  `-Cmd eval` — consult Astra between steps (Arash's rule). 20-minute status: Monitor `sim/kitchen_v3/status20_amb.sh`.

**STATUS 2026-10-04 02:00 — STOPPED by Arash.** Specialist learning curve: ep10 DONE (VAL .705 / .615), ep20 DONE
(VAL .801 / .709), **ep39 at 38/60 epochs** (best VAL mAP50 .841 / 50-95 .738 at epoch 38; checkpoint
`~/thor-data/kitchen_v3/_ckpt/FloorPlan1.amb26m.ep39/`). Console upgraded for the two detectors (WEBAPP_API/UI_API 6;
`docs/live-view.md` "The two detectors in the console") — not yet tested live.
**RESUME:** `sim/kitchen_v3/amb_train.ps1 -Cmd lcurve` (skips ep10/ep20, resumes ep39 at epoch 39; then copies it to
`FloorPlan1.amb26m.pt`), Astra check, then `-Cmd control`, Astra check, then `-Cmd eval`. FloorPlan1 only through steps
3-7 (Arash 2026-10-04).

**Next after the capture:** close-protocol val/test + top-ups -> primary YOLO set -> train YOLO26s + YOLO11s ->
evaluate -> disambiguation set from the primary's VAL errors -> specialist -> compare with the paper.

---


Last updated: 2026-09-29 20:35. **Status: STOPPED by Arash ("save and stop"). Everything is saved and verified.**

| kitchen | state |
|---|---|
| FloorPlan1 | **complete** — 10,467 frames, verified (check_capture: no FAIL). Top-ups NOT yet run. |
| FloorPlan6 | **73 of 96 episodes** done (8,016 frames, verified consistent: 0 missing, 0 orphans). Episodes **39, 48, 56 are marked failed** ("render glitch on every redo" — the glitch came back on all 6 redos); 20 not started. |
| FloorPlan8/15/17 | not started |

**RESUME PLAN (Arash's go of 2026-09-29 ~19:30, in this order):**
1. Clear FloorPlan6's failed episodes so they are retried (nothing running): in
   `~/thor-data/kitchen_v2c/FloorPlan6/capture_state.json` remove keys 39/48/56 from `failed_episodes`.
2. Relaunch the capture driver `sim/kitchen_v2c/capture_all.ps1` (see §5) AND the hold watcher
   `scratchpad/hold_after_fp6.ps1` (or edit `capture_all.ps1` to stop after FloorPlan6) — FloorPlan6 resumes at its
   missing episodes; the driver must NOT continue to FloorPlan8 (Arash wants FloorPlan1+6 trained first).
3. Smoke-test the whole balance+train chain: `scratchpad/smoke_balance_train.sh` (tiny budgets, separate root).
4. Run `sim/kitchen_v2c/balance_train.ps1` (copy `balance_train.sh` to `~` first; remove its wait-for-HELD loop if
   the run was not held by the watcher): balance FloorPlan1 then FloorPlan6 (train top-up to every head/combination,
   cap 24; exactly 12+12 `heldout-close-v1` episodes; audit; detection gate + split lists), then train + evaluate
   YOLO26s and YOLO11s per kitchen (`sim/train_kitchen.py`; TEST, CLOSE-TEST, LIVE at 150 frozen poses).
5. Report results vs targets (macro AP50 >= .90, AP50-95 >= .70), then Arash decides on FloorPlan8/15/17.

Built today and NOT yet run live: `stage_yolo` detection gate + split lists, the `topup` stage (train top-up +
close protocol), the StoveKnob state exclusion (quota axis-2 recount on resume), `sim/train_kitchen.py`.

**Render corruption (found 2026-09-29, fixed).** Arash's red/blue glitches were in the saved data: whole-frame red
casts, white wash-outs, and flat magenta/cyan "segmentation-colour" leaks (a different pass's colours in the RGB
frame). Evidence: in 16 of 18 paired cases a same-pose re-render moments later was clean; 0 corrupt renders in 600
static + 3,090 capture-like renders on a FRESH WSL. Cause: a WSLg/Mesa-D3D12 display degraded by earlier forced
simulator restarts (all of the old FloorPlan1 was captured in such a session). Fixes: (1) `render_guard` — every
kept view (and every paired render) must be two agreeing fresh renders, per whole image and per 80 px tile, with no
corruption signature (`frame_anomaly`: pure white, whole-image colour cast); (2) fail-fast when > 10 % of views need
a re-render; (3) the Windows driver runs `wsl --shutdown` before every kitchen and on a guard abort (resume, max 2).
Post-hoc filtering could not separate subtle red casts from legitimate red materials, so by Arash's decision
(option B) FloorPlan1 was **recaptured** from scratch; the old one is archived (`~/thor-data/kitchen_v2c/
FloorPlan1.v2c-pre-guard`, reports in `sim/kitchen_v2c/FloorPlan1.v2c-pre-guard/`, incl. its quarantine of 122
frames). First 3 new episodes: 0 corrupted frames of 330, guard re-render on ~2 % of views, 38/38 pairs valid.
Still to do in step 4: the same guard in the live console stream and the agent's perception path.
**13:09 — the safety net fired once.** In fresh-FloorPlan1 episode 3 the display degraded (guard rejected a render on
50 % of views, 8 skipped); the fail-fast abort fired, the driver reset WSL and resumed at episode 4 — no hands.
Gap found: the invariant check ran AFTER the episode was committed, so episode 3 (110 frames) was kept. All its frames
passed the guard and the corruption scan finds none, but by rule it should be redone. Fixed (per-episode invariants
now run BEFORE commit; the 8-episode verifier after) — effective from the next kitchen's process. **Done 13:54:** episode 3 removed (111 rows, backups `*.before_redo_ep3_*`) and recaptured clean.

**13:50 — the run stopped itself after 3 aborts (episodes 3, 29, 36).** Measured on the 3,960 kept frames: re-renders
happen on ~2-3 % of views WHATEVER is in view (no-animation views 2.9 % vs running tap 1.9 %, lit burner 2.6 %) —
real glitches at a base rate, in bursts, and a fresh WSL does NOT prevent them (two bursts came < 10 min after a
reset). Every kept frame scanned clean, i.e. the guard works; the 10 % stop threshold was too sensitive. Changed:
after any disagreement the guard now requires THREE identical consecutive renders (closes the "two identical
corrupted renders" gap); the run aborts only when an episode needs a re-render on > 30 % of views or SKIPS > 5 %.
Relaunched 13:52, resuming FloorPlan1. Bursts then recurred every ~15 min (episodes 49 at 14:11, 61 at 14:26 — both
discarded pre-commit and redone after a WSL reset, as designed). The fixed budget of 3 resets per kitchen would have
stopped the run, so the driver is now PROGRESS-based: it keeps resetting WSL and resuming while each run commits at
least one new episode, and stops only after two consecutive runs that commit nothing (a genuinely broken display) or
any non-guard abort. Swapped in at 14:32 (the in-progress episode was redone).

**Root-cause experiment (2026-09-29 16:07-18:38, `sim/glitch_experiment.py`, results in `~/thor-data/glitch_exp/`).**
Capture-like views, one factor at a time, 20-30 min each on a fresh WSL. Every run starts CLEAN, then at an
unpredictable moment the simulator PROCESS flips into a degraded state and stays there until restarted:
A rgb-only 44 % glitched (onset ~1,780 views), B rgb+seg 7.2 % (~2,600), C full 6.1 % (~1,770), D full + 100 ms
pacing 16.2 % (~220 — slowing down does not help). Not the render passes, not pacing, not a fixed frame count, not
animated objects, not NVIDIA driver resets (1 driver event today, unrelated), not the WSL display (a WSL reset helped
only because it restarts Unity). Restarting the process: F every 1,500 views 0.2 %; **G at the first glitch
0.02 % (1 in ~5,000)**. Exact component inside the GPU path (Unity -> Mesa D3D12 -> WSLg) not pinned; the
software-rendering control (variant E) was not run. **Fix adopted (Arash, "G is ok"):** at the first guard-detected
glitch the episode is discarded, the simulator process restarted (~17 s) and the episode redone from its seed
(`RenderGlitch`, `restart_on_glitch`, max 6 redos, then the episode is recorded failed); forced-glitch test passed.
Capture resumed 18:34 (FloorPlan1 complete, FloorPlan6 started).

**Why v2c (2026-09-29).** Arash saw the agent "looking stuck" near the end of FloorPlan1 in v2b. Two causes, both
fixed and verified: (1) **AI2-THOR's `InitialRandomSpawn` renames every object it moves** (objectId encodes
position — 29–30 of 77 ids change per shuffle; the `name` field stays identical and unique). v2b looked objects up
by id, so in every shuffled episode (all val/test, 75 % of train) the apple, egg, potato, mugs … were never put
into a state — only 28 of 48 settable objects were set — and per-instance targets for vanished ids could never be
met. (2) Those unreachable targets made late train episodes **grind** to the 550-attempt cap (~3 min to keep 35–44
frames), which on screen looks exactly like a frozen agent. Labels were never wrong (always read from metadata);
coverage of movable objects' states was. Fixes: identity by `name` everywhere (`instance_key`); a per-episode
**state-coverage gate** (the audit FAILS below 90 % — v2c smoke: 44/44 in every episode incl. shuffled); an
**anti-grind exit** (a train episode ends when its last 150 attempts kept ≤ 8 frames; train episodes are skipped
once every train target is met); a per-episode **hang guard** (10 min → restart the simulator, redo once, then
record the episode as failed — verified by forcing it). **Fail-fast (2026-09-29, after FloorPlan1 had been captured
three times):** every episode checks its invariants (state coverage ≥ 90 %, ≤ 5 % failed state settings) and the
full read-only verifier `sim/check_capture.py` (files, coverage, movable objects really vary in shuffled episodes,
instances by name, held views differ, pair validity, pace, projected per-state coverage) runs every 8 episodes
inside the capture (`checkpoints/after_NNN.json`); any FAIL aborts the run ("ABORTED (fail-fast)") and the launcher
does not start the next kitchen. Run the verifier by hand any time: `python sim/check_capture.py --scene <scene>`.
v2b data (`~/thor-data/kitchen_v2b/`) is superseded: do
not train on it.

**Train view policy v2c-2 (2026-09-29 ~01:36, from episode 41 of FloorPlan1; approved by Arash after Astra step3d).**
The first checkpoints showed 53/84 state keys projecting below 150 crops: states were set, but small objects were
rarely >= 900 px. Train episodes now add an explicit state-targeting view (35 %: object chosen by its CURRENT state's
deficit, shot standing from 0.45-1.0 m so the camera looks into vessels) and choose held objects by state deficit.
Val/test protocol unchanged (verified byte-identical view mix). Every episode record carries `policy`
(`v2c-2` / `heldout-v2c-1`; no field = v2c-1, FloorPlan1 episodes 0-40); every (re)start appends its config to
`capture_config_history.jsonl`. **To stop a run safely** use self-safe patterns:
`wsl -d Ubuntu-22.04 -- bash -c "pkill -f '[c]apture_all'; pkill -f '[b]uild_kitchen'; sleep 3; pkill -f '[t]hor-Linux64'"`.

**Per-head counting v2c-3 (2026-09-29 ~02:13, from FloorPlan1 episode 69; Arash's go after Astra step3e).** The state
quota counts PER HEAD — `Type|axis=value` (`vocab.axis_keys`; liquid only when filled; a broken vessel keeps only
its broken head) — the unit the multi-head state estimator learns; per-object head weight is normalised. Full
combinations stay as an ENFORCED floor (>= 10 large train crops from >= 3 episodes). Counted the old way, 44/84
combinations projected short; per head, 20/89 (all small objects: whole egg, apple slices, potato, water in a cup) —
anything still short at the end of the plan gets close-up top-up TRAIN episodes, never lower targets. The resume
recounted all stored records (`quota recounted to axis-1 …`, totals identical) — no recapture. Val/test protocol
unchanged (re-verified).

Arash's 7-step order: (1) platform check — done; (2) human-verified 3-level instructions — done; **(3) per-kitchen
datasets + two detectors per kitchen — in progress**; (4) agent movement/interaction logic, esp. hard instructions;
(5) memory database design; (6) how much interaction before the final test; (7) final test.

Working rule (memory `feedback-astra-then-go`): before building any section, get a Codex `gpt-6-astra` second opinion
(`codex exec --model gpt-6-astra -s read-only`), explain it simply, and wait for Arash's go. Progress update every
20 min during long runs; verify after every major step.

---

## 1. What was decided (with Arash's go)

| decision | choice | why |
|---|---|---|
| kitchens | **FloorPlan1, 6, 8, 15, 17** | every one of the 277 instructions is applicable in ≥1 (269 in all five — the max of any 5-subset); FloorPlan1 = published continuity; FloorPlan5 excluded (lettuce pickup hangs the sim). `python sim/build_kitchen_data.py select-scenes` re-derives it. |
| executability | verified by running every reference plan (no LLM) | `sim/dataset/general_validation_FloorPlan{1,6,8,15,17}.json`: **all 277 succeed in ≥1 kitchen**. FAIL counts 4 / 16 / 35 / 15 / 11 are agent-side (placement "no valid position" 48, occupied burner 14, heat → not Hot 6, potato not cooked 3, FloorPlan8 DirtyObject Bowl NullReference 5 + fill 5) → **step 4 work**. Old FloorPlan1 validation kept as `general_validation_FloorPlan1.2026-09-20.json`. |
| architecture | **option (b)** | detectors predict object TYPES only; a separate pooled state estimator (ConvNeXt-Tiny crop classifier, value + observability heads, may answer *unknown*) predicts per-object state. Astra rejected "side-on vessel = empty" labels. |
| detectors | primary **YOLO26s**; close-view specialist **YOLO26m vs l** piloted on FloorPlan1; **YOLO11s** baseline | mirrors the paper's s/l; YOLO26 is the newest release (Jan 2026; YOLO27 unreleased). |
| training recipe | ultralytics **8.4.83 pinned**, optimizer **MuSGD pinned** (`auto` silently switches AdamW↔MuSGD at 10k iterations), `end2end` to be pinned, state-safe augmentation from `yolo_recipe.py`, RFS only as an ablation arm | Astra |
| specialist types | a-priori small/confusable list + primary **VALIDATION** AP50 < .90 or > 5 % confusion (both sides), min support 20 — never test | Astra: test-set leakage |
| acceptance (pre-declared) | macro AP50 ≥ .90, AP50-95 ≥ .70, task-critical state P/R ≥ .95 on observable crops at validation-chosen thresholds; abstentions reported as coverage, never as correct; bootstrap CIs by episode | Astra |

Astra's full opinions: `C:\Users\arash\AppData\Local\Temp\claude\D--Personal-Master-s-Project\a53bc8f7-...\scratchpad\ASTRA_OPINION_step3{,b,c}.md`
(scratchpad — may be cleaned; the decisions above are the record).

## 2. Bugs found and fixed today (all verified)

- **RGB/BGR swap in live detection** — `perception.Detector.detect` passed AI2-THOR RGB frames to ultralytics, which
  assumes BGR. Every live detection before 2026-09-28 (incl. the FloorPlan1 oracle mAP50 0.49) was colour-swapped.
  Fixed (hands BGR). **The 0.49 baseline must be re-measured** (`sim/oracle_eval.py`) before it is cited.
- Capture (smoke tests + first full run): microwave step order (switch off before opening); held object not dropped
  on an early episode end; held-object rotation undone by turning the agent afterwards (now: turn agent, then
  rotate + move the object — silhouettes verified to differ); **broken Plate/Mug shards are not attributed to the
  object in instance segmentation** → vessels are never broken during capture (`vocab.CAPTURE_UNSUPPORTED`; state
  stays in the schema, estimator must say *unknown*; affects 2 of 277 goals); StoveBurner is knob-driven
  (`vocab.CONTROLLED_TYPES`, never set directly); `set_state` could report success for a missing object; resume left
  orphan files of an interrupted episode (fixed + tested 2026-09-28 19:10).

## 3. The code

| file | role | gate |
|---|---|---|
| `sim/vocab.py` | type folding (AppleSliced→Apple), state schema (applicable / value; observability assigned later), composite targets, verified action recipes | `python sim/_kitchen_data_selftest.py` **77/77** |
| `sim/kitchen_data.py` | pure policy: scene selection, fixed episode plan with stable SHA-256 seeds, per-split Quota (type / instance / state), opportunity-based suppression, audit, RFS, specialist-type selection, configs (`CAPTURE`, `TRAIN`, `ACCEPTANCE`) | same |
| `sim/build_kitchen_data.py` | live driver: `probe` (per-instance reachable states, verified), `capture` (resumable per episode), `audit`, `yolo` (refuses a failed audit) | smoke tests in WSL |

Capture design (v2b): 96 episodes per kitchen (train 64 / val 16 / test 16) × 110 frames = 10,560 frames; each episode
re-randomises placement (always for val/test), lighting, materials; irreversible states at episode start, reversible
states re-rolled every 22 frames (5 rounds); views aimed by deficit (train) or uniformly (val/test); 30 % close views,
10 % held "floating" views, 15 % paired renders (noise + one-axis flip, validity recorded) as observability evidence;
every frame stores RGB, depth (mm PNG), instance segmentation + colour→object map, camera pose/intrinsics, every
object's box/px/distance/lineage/state record/rim elevation.

## 4. Where the data is and what state it is in

Linux home in WSL (`\\wsl.localhost\Ubuntu-22.04\home\arash\thor-data\`):

| path | status |
|---|---|
| `kitchen_v2c/<scene>/` | **current version** — capture started 2026-09-29 00:41 (FloorPlan1 → 6 → 8 → 15 → 17). Check `sim/kitchen_v2c/capture_all.log`. |
| `kitchen_v2c/smoke/` | smoke test data — disposable |
| `kitchen_v2b/` | **superseded — do NOT train on it** (movable objects stayed in their default state in shuffled episodes; FloorPlan1 complete, FloorPlan6 partial). Its FloorPlan1 audit (`sim/kitchen_v2b/FloorPlan1/audit.json`) documents the problem. |
| `kitchen_v2/FloorPlan1/` | **first full run — do NOT train on it** (contains unlabelled shards of broken vessels; 48/88 state keys short). Kept as the record behind the v2b changes. Its audit: `sim/kitchen_v2/FloorPlan1/audit.json`. |

Repo copies of small reports: `sim/kitchen_v2c/<scene>/probe.json` (and `audit.json` once a kitchen finishes);
log `sim/kitchen_v2c/capture_all.log`; launcher `sim/kitchen_v2c/capture_all.sh`.

## 5. How to resume

1. Make sure nothing else holds the GPU/simulator (`wsl -d Ubuntu-22.04 -- pgrep -af thor-Linux64`).
2. Resume the capture (continues from the last completed episode of the current kitchen, then the rest; ~1.5 h per kitchen):

   ```powershell
   wsl -d Ubuntu-22.04 -- bash -c "cp '/mnt/d/Personal/Master'\''s Project/sim/kitchen_v2c/capture_all.sh' ~/capture_all.sh"
   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d Ubuntu-22.04 -- bash /home/arash/capture_all.sh'
   ```

   **Never** launch with `setsid nohup` — the launching wsl.exe exits and WSL kills the job a minute later.
   Watch `sim/kitchen_v2c/capture_all.log`.
3. After each kitchen: verify `audit.json` (shortfalls by unit/split, missing types, unrealised states incl.
   re-rolls, `pairs.valid_evidence`, `tried_never_observed`) and look at sample frames with boxes drawn; compare
   state coverage with v2 FloorPlan1 (48/88 train state keys short).

## 6. What comes after the capture (each needs Arash's go)

1. Re-measure the old FloorPlan1 detector with the RGB fix (`sim/oracle_eval.py`) — the honest baseline.
2. Hand-label a stratified sample of crops ("is the state visible?") → calibrate the observability rule (Astra:
   paired renders are evidence, not ground truth). **The tool is built (2026-09-28) and tested**: `sim/state_review.py`
   + page template `sim/review/state_review.html`. Blind (the page never contains the simulator's answer; the reviewer
   says what they SEE, incl. "Can't tell"; visible = answered AND right), stratified over axis × true value × object
   size × distance, TRAIN episodes only, ~300 items ≈ 20–30 min. Use it after all five kitchens are captured:

   ```bash
   python sim/state_review.py make --n 300 --root //wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v2c \
       --out "sim/kitchen_v2c/review"             # then double-click sim/kitchen_v2c/review/index.html
   # answer (keys 1-5), press Export, save answers.json into that folder, then:
   python sim/state_review.py score --dir sim/kitchen_v2c/review --answers sim/kitchen_v2c/review/answers.json
   ```

   Answers autosave in the browser (resume any time). `truth.json` stays next to the page but is never loaded by it.
3. `yolo` stage (only on a passed audit) → FloorPlan1 pilot training: YOLO26s primary, YOLO26m vs l specialist,
   YOLO11s baseline, capture-balancing vs +RFS; measure real epoch time and VRAM with the simulator stopped.
4. Freeze the recipe, train the other four kitchens; train the pooled state estimator.
5. Three frozen evaluations: operational views, balanced state/occlusion challenge set, full
   primary → approach → specialist workflow.
