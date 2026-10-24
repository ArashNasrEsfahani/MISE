# FloorPlan1 ambiguity (specialist) detector — frozen specification (step 3, 2026-10-03)

Arash's design (as in his paper): a second **detector** trained on a targeted ambiguity dataset. Reviewed with
Codex/gpt-6-astra (step3j, step3k, step3l). Decided from the primary's VALIDATION errors only (TEST never read).

## Targets (10)
Fork, Spoon, ButterKnife, Knife, Spatula, SaltShaker, PepperShaker, Plate, Pot, Pan.
Evidence (VAL, YOLO26s): shakers ~pure classification confusion (SaltShaker -> PepperShaker 19 %); utensils
classification 12-19 % + misses 15-30 %; Plate -> Pot, Pan -> Pot. Faucet / SinkBasin deferred (localisation).

## Data — `~/thor-data/kitchen_v3_amb/FloorPlan1/` (separate store; never mixed into the primary's frames.jsonl)
- Protocol `amb-v1`. Episodes and splits fixed BEFORE capture, one layout (InitialRandomSpawn, p = 1.0) per episode,
  layout seed derived from (scene, protocol, split, index) — a split never shares a layout seed with another:
  train 40, val 6, test 6, **final holdout 6** (sealed: not built into any list until every choice is frozen).
  110 frames per episode: ~4,400 / 660 / 660 / 660.
- Appearance as kitchen_v3: original colours, brightness-only lighting on 20 % of episodes.
- Partner placement: at setup, for 2 random confusable pairs per episode (Salt/Pepper, Fork/Spoon,
  Spoon/ButterKnife, ButterKnife/Knife, Fork/ButterKnife, Spatula/ButterKnife, Plate/Pot, Pan/Pot), the partner is
  placed 8-20 cm from the target on the same surface (`PlaceObjectAtPoint`); other targets stay where the layout put
  them (alone).
- View mix per episode (fixed proportions): close 0.3-0.8 m aimed at a target 30 %; ordinary 1.0-2.5 m aimed at a
  target 30 %; held target (one pickup per instance per episode, new spot per view, rotated + moved) 15 %; target-free
  25 % (aimed at a non-target object or random; kept only if NO target box is visible — hard negatives with
  look-alike distractors such as Cup, Bottle, Mug, Book, CreditCard, Statue).
- Labels: ALL 49 classes stored per frame (same labeller, masks, depth). The specialist's labels are derived:
  target classes only, every visible target instance labelled, everything else background.
- No quota-driven frame declining (fixed protocol); empty-view drop and the render guard as in kitchen_v3.

## Models
- Specialist: YOLO26m, 10 classes, same recipe as the primary (60 epochs, patience 20, 640, batch 16 if it fits),
  selection on amb VAL. Learning curve: train at 25 / 50 / 100 % of the train episodes (cheap check of sufficiency).
- Control: the primary (YOLO26s) fine-tuned on primary-train + amb-train with all 49-class labels (original-data
  replay included), 20 epochs, selection on primary-val + amb-val.

## Fusion (frozen on VAL)
Strict replacement: drop every primary prediction of a target class, append the specialist's predictions; primary
non-target predictions unchanged. Duplicate / conflicting target boxes reported separately.

## Evaluation (full images, low-confidence AP; operating P/R at .25)
- amb-test and primary-test (49-class GT): primary alone vs primary + specialist vs control; all-49-class and
  target-class mAP50 / mAP50-95, per-class AP, non-target regressions, latency; paired episode bootstrap (95 % CI).
- Final: the sealed amb holdout, once, after everything is frozen.
- Disclosure: primary-TEST informed the YOLO26s-vs-YOLO11s choice; the paper's .938 is context, not a benchmark.

## Amendment 1 (2026-10-03, Arash; reviewed Astra step3m-n) — the specialist is an INSPECTION model
Arash: "the ambiguity model is used when the object is held up, so there should be a lot of shots of the object in
hand." Seeds, splits and the sealed holdout are unchanged; val/test/holdout start at index 1 (index 0 layouts were
looked at in smoke runs).
- View schedule per episode (110): **held target 60** (one pickup per target instance, 6 views per pickup, each from
  a new standing spot; 40 % of views face the target's look-alike partner; object rotated and moved for inspection:
  ahead -0.15..+0.05 m, right +-0.15, up 0..+0.15, horizon -5..20 deg; held object's visible mask >= 800 px at 640x640),
  **held non-target 12** (no specialist box for the held object; visible background targets still labelled), in-place
  close 14 / ordinary 10 / target-free 14 (>= 4 labelled non-targets, no repeated camera cell). Held views last.
- Every frame records requested and realised view type, camera-to-target distance, inspection state, pickup id and
  (for held targets) whether a look-alike partner is visible. Shortfalls are logged, not refilled (~95-105 frames).
- **Deployment rule = GATED fusion:** on frames where the robot holds an object up for inspection (robot state, not
  simulator identity) the specialist replaces target-class predictions; otherwise the primary is used unchanged.
  Ungated replacement is reported as an ablation.
- **Primary endpoint:** held-target naming success (top-confidence prediction on the held box, IoU >= .5, conf >= .25;
  a miss is a failure), with the held-non-target false-positive rate; plus target AP on inspection frames and on all
  frames, all 49 classes, latency; paired episode bootstrap.

## Capture result + training freeze (2026-10-03 17:00; Astra step3o)
- 69 of 70 episodes retained; train 39 of 40 (episode 3024 excluded: the render never settled on several views after six
  redos — a technical exclusion, not assumed random). 6,880 frames: held target 3,726, held non-target 828, target-free
  900, requested-close 765 (realised close <= 0.8 m: 96; 669 realised ordinary — close-range coverage is limited, no
  broad close-range claim), ordinary 661. Splits: train 3,899 / val 997 / test 1,009 / holdout 975 (sealed).
  Look-alike partner visible in 35 % of held-target frames.
- Specialist: YOLO26m, MuSGD, 60 epochs, patience 20, imgsz 640, physical batch 8 with ultralytics nbs=64 (accumulation
  to ~64 images per update), seed 0, ultralytics 8.4.83; nested learning curve on the first 10 / 20 / 39 train episodes
  (frozen order, 3024 removed), each from the same COCO checkpoint, all selected on the full amb VAL (mAP50-95).
- Control: YOLO26s primary weights, MuSGD lr0 .002, 20 epochs, batch 16, selection = ultralytics fitness (mAP50-95 over
  the 49 classes) on the POOLED validation list (2,200 primary-val + 997 amb-val images, unweighted).

## Learning curve result (2026-10-04 13:42; Astra step3p)
Specialist YOLO26m, all selected on amb VAL (997 images, mixed view types — not the inspection endpoint):
10 episodes (997 img) .705 / .615; 20 (1,988) .801 / .709; **39 (3,899) .850 / .769** -> the specialist
(`sim/weights/v3/FloorPlan1.amb26m.pt`). Diminishing observed gains (+.096 then +.049 mAP50), not saturation; fixed
epochs mean larger subsets also got more updates, so this is a curve for the recipe, not a pure data-volume effect.
The best checkpoint was epoch 59 of 60 (late gains overlap the close_mosaic phase): "the best checkpoint occurred near
the training limit; convergence was not established" — no post-hoc extension. More data = future work.
Control verified before launch: 20,198 train / 3,197 pooled-val images, 0 duplicates, 49 class ids identical to the
primary. Results are reported as a comparison of PRESCRIBED SYSTEMS (size, label scope, exposure, selection differ),
with primary-test retention next to held-target naming success.
