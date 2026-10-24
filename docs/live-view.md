# Live view: move-by-move motion, and the senses switch

Two console features, both added because of one observation from a real run:

```
13:33:58  agent: Okay, I will open the fridge.  [1 step(s)]
13:33:58  open Fridge: ok
```

The fridge opened inside a single second — and, more to the point, inside a **single rendered frame**. You could
read that it happened; you could not watch it happen. Meanwhile the console's `perception: oracle | perceived`
control had been wired to nothing since Phase 4: it set a flag, reported the flag in `/status`, and changed no
behaviour whatsoever.

- **§1 Move-by-move** makes the robot's motion something you can actually watch.
- **§2 The senses switch** makes "perceived" mean what it says.

Gate: `python sim/_motion_selftest.py` → **88/88**, headless on Windows (no ai2thor, GPU, Ollama or ultralytics).

> **Restart the console after editing `sim/webapp.py`.** `sim/web/*` is re-read from disk on every request;
> `webapp.py` is loaded once at process start. Reloading the browser alone therefore pairs a **new UI with an old
> server** — the new controls 404, the new `/status` fields are absent, and every new feature looks broken rather
> than absent. That is not hypothetical: it ate the first real run of this work. §3 is the fix.

---

## 1. Move-by-move

### Why frames went missing

Three independent causes, all of which had to be fixed:

**(a) The stream sampled instead of following.** `/stream` looped on `time.sleep(1/30)` and re-sent whatever
`_state["jpeg"]` happened to hold. At any worker rate that is not exactly 30 fps that silently drops frames
(worker faster) or duplicates them (worker slower) — and the worker's rate swings with resolution, the active
view, and whether YOLO is running.

**(b) The generators yielded too coarsely.** The worker renders exactly **one frame per generator `yield`**, so
the yield granularity of `walk_to` / `_rotate_to` *is* the frame rate of the robot's movement. At the old
defaults — 30° per turn chunk, 0.25 m per step — crossing a kitchen and turning to face a fridge is a handful of
frames. Nothing was dropped there; there was simply never anything in between to see.

**(c) Manipulation has no animation at all.** AI2-THOR applies `PickupObject`, `SliceObject`, `ToggleObjectOn`
and `OpenObject` in one controller step. The fridge is shut in frame *N* and wide open in frame *N+1*. No amount
of streaming fixes that, because the intermediate states were never rendered.

### What changed

**Sequenced fan-out.** The worker publishes each frame through `_publish_frame()`, which bumps `_state["seq"]`
and appends to a bounded ring (`STREAM_BACKLOG = 24`). Each `/stream` client blocks on a condition variable and
writes every `seq` it has not written yet (`_stream_pending`). A fresh connection starts at the newest frame
rather than replaying the backlog into a browser that just opened the page. A client that falls further behind
than the backlog is fast-forwarded — and the frames it will never see are **counted** into
`/status.motion.dropped`, because a stream that silently skips motion is the exact failure this replaced.

**A motion profile the whole skill tree inherits.** `ThorAgent.motion = {"step", "turn"}` is now the default
source for `walk_to`, `_rotate_to`, `turn_to_object` and `scan_for`. That matters because the composite skills
(`heat_object`, `pick_and_place`, `wash_object`, …) call `walk_to` with no arguments — threading `step`/`turn`
through a dozen signatures that never wanted to know about them would have been the alternative. `walk_to`'s
step budget became a **path length** (~20 m) rather than a fixed 80 steps, so the finer profiles don't strand
the agent mid-kitchen with `"arrived"` never reached.

| profile | turn | step | pacing | hold | openness steps |
|---|---|---|---|---|---|
| `fast` *(default since 2026-10-05, see "Painfully slow" below)* | 30° | 0.25 m | none | 0 | 1 |
| `normal` | 15° | 0.15 m | none | 3 | 6 |
| `cinematic` | 8° | 0.08 m | 50 ms/frame | 8 | 12 |

`fast` reproduces the previous behaviour exactly — same turn chunk, same step, no holds, one-shot open/close.
It is there so any measurement taken before this change stays reproducible.

**Instantaneous primitives are held.** `_approach_then` renders `hold` frames either side of the action, so the
state change is something you watch rather than a jump between two frames. `_hold` does nothing at `hold = 0`.

**Doors swing.** `_animate_openness` drives `OpenObject`'s `openness` from where the door is now to the target
in profile-sized increments, one rendered frame each. Two details that matter:

- it never asks for `openness = 0` mid-swing, and lands a close with the canonical `CloseObject`;
- if the simulator refuses an intermediate openness it stops stepping and throws once to the full target, so the
  **end state is identical to the un-animated path either way**.

**A move-by-move readout.** Every label a running generator yields (`stepping`, `turning`, `arrived`,
`opening`, `searching · 7 types seen`) is published per frame as `/status.motion.label` and shown in a chip at
the bottom of the camera. So the feed is watchable *and* readable, move by move.

**Pacing.** `cinematic`'s 50 ms/frame delay applies **only while a plan generator is running** — idle and manual
WASD drive stay at full speed.

### The teleports hiding inside "smooth" locomotion

Reported after the first fix landed: *some* movement still felt abrupt. Three teleports were doing it, and a
teleport is the one thing that cannot be animated — every one is a hard cut in the feed.

- **Every `walk_to` ENDED in one.** Its last act was `look_at_object`, which reaches the right pose by
  teleporting: yaw and pitch snap in a single controller step. Replaced by `aim_at_object`, which rotates and
  then tilts over several frames to the identical pose. `turn_to_object` got the same treatment.
- **A blocked step hopped.** On any failed `move_ahead`, `walk_to` teleported to the waypoint. It now retries at
  half and quarter step first (`squeezing past`) and only hops if both fail.
- **A frontier stop teleported outright.** `search_for` jumped to unexplored parts of the room. New
  `walk_to_point(x, z)` walks there using the same corner-following as `walk_to`.

### Where to find it

- Console: **Navigation → motion** (`fast` / `normal` / `cinematic`).
- API: `GET /motion?speed=cinematic` → `{"ok", "speed", "profiles"}`.
- Telemetry: `/status.motion` = `{speed, label, seq, dropped, note}`.

---

## 2. The senses switch

### What "perceived" now means

`_PERCEPTION["mode"]` used to be inert. It is now the thing it always claimed to be:

| | oracle | perceived |
|---|---|---|
| OBSERVATION block | `ag.describe(30)` — AI2-THOR's ground-truth object list | only what this scene's trained YOLO detector finds |
| how the room is surveyed | metadata read | the agent **turns 360° and looks**, running the detector at each stop |
| pre-plan search | `_active_exploration` (turns toward objects looked up in metadata) | the detector sweep *is* the search |
| held object | metadata | metadata — proprioception, not vision |
| retrieval / RAG | unchanged | unchanged (orthogonal axes) |
| execution + goal check | simulator | simulator |

That last row is the honest limit, and the UI says so in the button's tooltip. This is the **observation** axis:
it changes what the planner *knows*, not how an action resolves an `objectId` once the plan exists. It matches
`sim/eval.py --perception` exactly, where SR is always scored from oracle metadata — which is what makes the
console and the eval harness comparable.

### Degradation is loud

If the scene has no trained weights, or ultralytics is unavailable, or inference throws, perceived mode falls
back to oracle — and says so, in three places at once: the transcript (`perception DEGRADED to oracle: …`), the
red note under the button, and `/status.perception_degraded`. A silent fallback would leave a "detector only"
badge sitting over an oracle observation, which would make the whole switch a lie and any measurement taken
through it worthless.

`/status.perception_ready` is a cheap up-front answer to "could this even run here?" — it is false before you
have run **scan & learn** and **train detector** for the current scene.

### One weights bug fixed on the way

`weights_path()` in `sim/perception.py` is now the single resolver for a scene's `.pt` file, and both the console
and `sim/eval.py` call it. AI2-THOR names the live scene `FloorPlan1_physics` while the dataset/eval side names
it `FloorPlan1`, and the file is written under whichever spelling did the scan. `eval.py` looked only for
`weights/<scene>.pt`, so **`--perception perceived` degraded to oracle on a machine with trained weights sitting
right there** — silently, since the old code had no way to report it.

No published number is affected: every record in all nine `sim/eval_*.json` files declares
`"perception": "oracle"` (checked 2026-08-22). The arm was never exercised, rather than exercised wrongly — which
also means the perception ablation the thesis wants is still **unrun**, and now runnable for the first time.

### Where to find it

- Console: the **scene info** button under the command box (and the `perception` segmented control in
  *Scene & setup* — the two stay in sync; both drive `/perception`).
- Turning it on also switches the camera to **Detect**, so you see the boxes the planner is reasoning from.
  Suppress that with `/perception?mode=perceived&view=0`. Detect view runs YOLO every frame, so expect lower fps.
- API: `GET /perception?mode=oracle|perceived` → `{"ok", "mode", "degraded", "ready"}`.

---

## 2b. Reset

`GET /reset?scope=session|all`, and a **reset** button in the topbar that arms into the two scopes rather than
opening a browser dialog — the choice *is* the confirmation, and a modal would block the page.

**`session`** — reloads the kitchen to its starting state (objects, doors, agent pose, empty hands) and clears
the transcript, plan, demo and search outcome. It deliberately does **not** touch motion detail, perception
mode, camera view, model or endpoint: those are the operator's choices, not run state, and silently reverting
them is how a reset button becomes something people are afraid to press.

**`all`** — everything above, plus what the agent has **learned**: the knowledge base and its embedding
sidecar, `memory.db` (spatial beliefs, episodes, perception labels, the discovered nav map **and the bandit's
weights**), the loop reports, the audit logs and the trace store. Each store is closed before its file is
removed, each step is guarded independently, and the response says exactly what was cleared and what failed:

```json
{"ok": true, "scope": "all",
 "wiped": {"cleared": ["knowledge base", "memory + bandit", "loop reports", "trace store"], "failed": []}}
```

The endpoint returns **409** while a command is running — wiping the stores under a live plan would race every
write it is still making — and **400** on an unknown scope.

Verified live: 10 procedures / 9 lessons → 0 / 0, `memory.db` and the loop reports and audit logs gone, and the
trace store recreated with `write_errors: 0`. (Taken against a backup and restored afterwards.)

## 3. Two things the first live run exposed

### A stale server is now impossible to miss

`webapp.py` carries `WEBAPP_API` and reports it as `/status.api`; `app.js` carries `UI_API` and compares. On a
mismatch — including an **absent** field, which is exactly what an old server sends — a red banner is inserted at
the top of the page:

> This console is newer than the running server (ui v2 vs server v≤1). Restart it — new controls will 404 and new
> panels will stay empty until you do.

Bump both numbers together whenever `web/` starts depending on a route or `/status` field that an older
`webapp.py` lacks. The self-test asserts they are equal, so they cannot silently drift.

### No plan step finishes in silence

Only the `_approach_then` actions (`pickup`, `open`, `slice`, …) wrote a `<action> <ObjectType>: ok` line. Every
composite (`scan_for`, `pick_and_place`, `heat`, `wash`, …) and every nav primitive finished without one — so a
one-step plan produced a transcript that went straight from the plan to the reward with no evidence in between
that anything had happened:

```
14:23:17  agent: I am searching for the apple.  [1 step(s)]
14:23:19  bandit reward: environment=1.00
```

`_tracked_step` now logs any step whose action is not in `_SELF_LOGGING_ACTIONS`, in the same documented format:

```
14:23:17  agent: I am searching for the apple.  [1 step(s)]
14:23:18  scan for → Apple: ok
14:23:19  bandit reward: environment=1.00
```

Note what this does *not* fix: `scan_for` only rotates in place toward an object it already resolved, so
"find the apple" is a small turn and little else. In **perceived** mode it is preceded by the full 360° detector
sweep, which is the part worth watching.

---

## 4. The trace store cannot run WAL on `/mnt/d`

**Root cause of two destroyed stores, found 2026-08-22.** `sim/trace.db` sits beside the code, which under WSL
means `/mnt/d` — a Windows drive over **DrvFs**. `trace.py` opened it with `PRAGMA journal_mode=WAL`. WAL needs a
shared-memory index (`-shm`) backed by working `mmap` and POSIX advisory locks, and DrvFs provides neither
reliably.

It does not fail cleanly. The store opens, accepts writes for a while, then every read and write starts
returning `unable to open database file` — with the file already structurally corrupt. The tell is that
**Windows-side Python could read the very file WSL-side Python could not open**, which points at the `-shm`
layer rather than the database.

Measured on the second occurrence, minutes after a clean rebuild:

```
write_errors   5  ->  12  ->  3149
/trace_runs    ok  ->  threw after 5 rows  ->  "unavailable"
count(*) on events                        ->  unreadable
```

`_wal_is_safe()` now picks the journal mode from the filesystem: WAL on a real Linux path or on native Windows,
a rollback journal (`TRUNCATE`) on a WSL-mounted Windows drive. That costs reader/writer concurrency, which for
an append-only log written a few hundred rows per command is a price worth paying for a store that survives.

The chosen mode is reported in `/trace_stats` as `tracer.journal_mode` **because it is not a detail** — if that
ever reads `WAL` while `db_path` starts with `/mnt/`, this regressed. Gated by T-73…T-77.

After the fix, three commands including a full six-stop search: `journal_mode: TRUNCATE`, `write_errors: 0`,
`integrity: ok`, every read endpoint serving.

## 5. If `/trace` shows no runs

Discovered 2026-08-22 on the live store: `sim/trace.db` had a **corrupt index**. The signature is specific —
`/trace_stats` and `/trace_tail` work and report a non-zero run count, but `/trace_runs` returns
`{"error": "unavailable"}`, because `query_runs` orders by an indexed column and `@_safe` swallows the real
SQLite error. A table scan (`count(*)`) reads fine; anything going through the damaged B-tree throws
`database disk image is malformed`.

```
python sim/repair_trace_db.py                   # inspect only — safe with the console running
python sim/repair_trace_db.py --fix             # back up, then REINDEX; console must be stopped
python sim/repair_trace_db.py --salvage [src]   # row-copy into trace.db.salvaged; never swaps it in
```

What the damage turned out to be, on the live store: **rootpage 29 = the `writes` table itself** and
**rootpage 34 = its `ix_writes_run` index**, plus pages cross-linked between the `runs` / `spans` / `events` /
`edges` trees ("2nd reference to page N") — the signature of an interrupted write. `REINDEX` and `VACUUM INTO`
both **failed**, and they always will on that shape: each walks the existing b-trees, which is precisely what is
broken. Hence `--salvage`, which copies **rows** into a fresh schema so SQLite builds clean trees as it goes.

**Judge a damaged store by what it can do, not by `count(*)`.** Ours claimed 47 runs; enumerating them threw
after **5**. The salvage returned **25** that all read cleanly. `count(*)` and `MAX(rowid)` resolve through
different (differently damaged) trees and openly disagree — `runs` reported `MAX(rowid)=39` against
`count(*)=47`, and `artifacts` reported 498 against 517 rows actually readable. Both are hints, not facts.

Two behaviours in the salvage that earned their place the hard way:

- a block scan that throws is **resumed past** the damaged region rather than abandoned;
- where that under-recovers, every rowid is retried **individually** — a single-row read touches one leaf page,
  so only rows physically on a broken page are lost. Without this pass, one bad page at the front of a small
  table discards the whole table: the first attempt returned **0 of 47** runs for exactly that reason.

The script refuses to write while the console answers on :8000, copies to `trace.db.bak-<stamp>` before touching
anything, and never deletes. The swap is left to you on purpose — promoting a salvaged provenance store over the
original is a decision about evidence, not a decision about a file:

```
mv sim/trace.db sim/trace.db.corrupt
mv sim/trace.db.salvaged sim/trace.db
```

---

## What is *not* verified

`sim/_motion_selftest.py` runs against fakes that reproduce the controller surface and the worker's job
contract. It proves the primitive sequences, the frame accounting, the degradation path and the profile wiring.
It cannot prove — and nothing off-GPU can — that a swung fridge door *looks* right, that YOLO finds the right
objects in a real frame, or the end-to-end frame timing. Those are Track C in `HOW-TO-TEST.html`.

## The two detectors in the console (step 3, 2026-10-04)

When `sim/weights/v3/<scene>.yolo26s.pt` exists the console uses the step-3 perception stack instead of the legacy
single `weights/<scene>.pt` (`perception.TwoStageDetector`, `webapp._detector_for`):

- **primary** YOLO26s (49 type-only classes) on every frame;
- **inspection specialist** YOLO26m (`<scene>.amb26m.pt`; Fork, Spoon, ButterKnife, Knife, Spatula, SaltShaker,
  PepperShaker, Plate, Pot, Pan) — **auto**: takes over those 10 classes only while the agent is holding an object
  (gated strict replacement, the deployment rule in `docs/ambiguity-spec.md`); **off**: primary only; **always**:
  specialist on every frame (the ungated ablation, not the deployment rule).

Where: Scene & setup → *perception* row → new **inspection** switch (auto / off / always) and a `detectors:` line
(primary · inspection model · "INSPECTING now" while it is active). In the **detect** camera view the specialist's
boxes are **amber and starred (`* Fork 0.91`)**, the primary's teal. Endpoints: `/inspect?mode=auto|off|always`,
`/status.detectors`. API bumped to 6 (`WEBAPP_API` = `UI_API` = 6).

How to test (FloorPlan1): set perception = perceived, view = detect; pick up a fork / salt shaker / plate and hold it
in view — its box turns amber (specialist); drop it — teal again. Switch inspection off/always to compare.
The specialist file appears only after the step-3 run finishes (`FloorPlan1.amb26m.pt`, copied from the 39-episode
run); until then the line says "inspection not trained yet" and only the primary runs.

## Render watchdog in the console (2026-10-04)

The simulator process on this machine degrades at a random moment (whole-frame red/blue cast or white frames; see
`docs/kitchen-datasets.md`). The console now checks the camera twice a second (`webapp._glitch_check`): three corrupted
checks in a row that survive a forced re-render restart the simulator on a fresh process, in the same scene and pose
(a held object and a running plan are dropped; the log says so). A cast that returns straight after a restart at the
same pose is treated as scene content (e.g. a close-up of a red object) until the agent moves. `/status.render` =
`{glitch, restarts}`. If the display itself is degraded (glitches right after every restart), restart WSL
(`wsl --shutdown`, or START-TESTING.bat option 4 then 1).

## State estimator in the console (2026-10-04)

Option (b)'s third piece: 7 per-axis classifiers (YOLO11s-cls, 128 px crops of each DETECTED box;
`sim/state_data.py`, `sim/train_state.py`, weights `sim/weights/v3/FloorPlan1.state_<axis>.pt`) calibrated on detector
crops (`sim/state_eval.py` -> `sim/kitchen_v3/FloorPlan1/state_calib.json`): a state is shown only above the box size
from which Arash could see it in his 300 blind hand labels and above a VAL threshold for >= 95 % precision per axis
and value; otherwise "<axis>?" (unknown). "broken" is judged for eggs only (vessels were never broken in capture).
In the detect view: "Fridge 0.99 · closed", "Lettuce · sliced". TEST (detector crops, matched): accuracy among
answered .93-1.00, coverage open .92 / sliced .90 / broken .98 / on .37 / cooked .41 / filled .40 / dirty .31.
Caveat: on unmatched detections (no same-type GT at IoU .5 — e.g. slice pieces, duplicate boxes) states are still
asserted, so end-to-end false assertions exist (counts in state_calib.json).

## "Painfully slow and glitchy" — root-cause review (2026-10-05, Astra perf review)

- **Proven cause of the freeze:** the simulator (Unity) hung/died; the console's worker thread hit the 100 s backend
  timeout, raised, and EXITED — the HTTP server kept serving the last frame and a stale fps, so the console looked
  slow but was dead. Fixed: `webapp._worker` is now a supervisor (`_worker_loop` inside it): any backend failure
  stops the running action (never replays it), drops queued actions, kills Unity and starts a fresh simulator in the
  same scene (backoff 2 -> 30 s). Step timeout 25 s (`SIM_STEP_TIMEOUT_S`, passed at controller start).
  `/status.sim = {state ok|stalled|recovering|starting, frame_age_s, generation, recoveries, error}`; fps reads 0 when
  frames are > 3 s old; the stage chip says "simulator not responding / crashed — restarting". Tested: kill -9 of
  Unity -> back at ~59 fps in 34 s.
- **Fixed earlier the same day:** detect-view models moved off the render loop (10 -> ~41 fps); MJPEG backlog 24 -> 4
  + 192 KB socket buffer (a slow browser drifted seconds behind); glitch watchdog counts 3 corrupted frames in 3 s
  (intermittent corruption never met "3 in a row").
- **Hypothesis, not proven:** the corruption (red/white casts, wrong view, silhouettes) and the Unity death are made
  worse by OTHER GPU work on the laptop (seen: another project's headless-Chrome GPU process rendering video into
  ffmpeg; Ollama's llama-server holding VRAM). To confirm: matched fresh-start trials (alone / Chrome rendering / Ollama
  idle / detector on-off) logging step latency, frame age, corruption + restarts, Windows GPU engine + memory, WSL RAM,
  dmesg OOM, Windows driver-reset events. Until then: avoid running other GPU-heavy jobs while using the console.
- **Measured A/B against the July console (git HEAD), 2026-10-05** (`sim/kitchen_v3/console_bench.py`,
  `console_ab_win.ps1`, from Windows): single moves equal (~0.07 s click-to-picture); the regression was in OBJECT
  actions — open fridge 0.16 s (old) vs 0.97 s, close 0.15 vs 1.16, go-to 0.47 vs 1.30. Causes: (1) the August motion
  profiles defaulted to "normal" (door swung in 6 steps, 15 deg / 0.15 m increments, hold frames) -> default is now
  "fast"; (2) `_approach_then` re-planned a walk (interactable-pose query) even when the agent stood at the object ->
  skipped when the object is visible and within 1.5 m (AI2-THOR's interaction range). After: open .12 s, close .10 s,
  go-to .88-1.11 s, 41 fps delivered (old 29.5).

## Console v2 and honest mode (2026-10-06)

The console was redesigned (spec `docs/console-v2-spec.md`, ledger `docs/console-v2-progress.md`) and gained an
**Agent: Oracle | Honest** switch in the command bar. Server and page are **API 7** (`WEBAPP_API` = `UI_API` = 7).

- **Three workspaces:** Run (camera · plan ticket · mind tabs), Evaluate (Instructions / Run & results /
  Scene-grounded set, same as before), Setup (scene, perception, planner models, simulator, capture library).
  Theme: system → light → dark (top bar), shared with `/trace`.
- **Honest mode** (FloorPlan1 only): type an instruction in English or Persian, or pick one with *From the set*.
  Gemma (`gemma4:e2b-it-q4_K_M`) plans with the honest tools; `HonestAgent` executes through a four-name port
  (`sim/honest_console.py`, scanned by `_step4_rules_selftest`). The ticket shows each step with retries, the cause
  and the underlying cause; the stamps say Executed / Goal (dataset items only) / Audit. The mind tabs show the
  agent's own walked map, its object memory, the audit (violations, reads per reason) and, separately labelled,
  the simulator's truth for display. Entering honest mode reloads the scene once with depth on. While it is on,
  object actions, batch open/close, scan/explore and the eval runner are refused; manual driving is locked while
  the agent acts. Items that need a setup are greyed out (run them with `sim/step4_gemma_trial.py`).
- **Live check (2026-10-06):** "سیب را بردار" → pickup Apple, done, 0 violations; "put the apple in the fridge"
  → pick_and_place Apple→Fridge, done, apple on the fridge shelf, 0 violations; dataset item g-l1-pickup-bread →
  failed honestly (3 retries, the open fridge door was in the way), goal *unmet*, 0 violations; Stop →
  "Stopping…" → cancelled. Oracle mode: move/rotate, motion speed, Open Fridge, chat plan, Detect, Evaluate,
  `/trace`, Reset all worked. Screenshots: `sim/step4/_checkpoints/console-v2/verify/`.
- **Planner fix found live:** Gemma wrote `put_in {object: Apple, to: Fridge}`; the validator dropped `to` and the
  agent searched for an *Apple receptacle*. `step4_gemma.validate` (prompt `s6-v4`, Astra-reviewed) now rewrites
  it to `put_in Fridge` only when the plan's own hand trace predicts the Apple is held, otherwise rejects (→ one
  repair). The ticket says "accepted after argument normalisation" and lists the validator's notes.
- **Open it at `http://localhost:8000`** from Windows. The WSL IP (`http://<hostname -I>:8000`) works too, but on
  this machine the Windows→WSL NAT stalls some browser connects for 1–9 s (`ERR_CONNECTION_TIMED_OUT`, curl is
  unaffected); WSL's localhost relay does not.
