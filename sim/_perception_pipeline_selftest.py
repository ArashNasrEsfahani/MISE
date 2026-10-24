"""Self-test for `sim/perception_pipeline.py` — the before/after comparison that decides whether a new detector
gets promoted. Headless: no ai2thor, torch, ultralytics or GPU.

    python sim/_perception_pipeline_selftest.py        # from the REPO ROOT

Only the pure half is testable off-GPU (the comparison, its rendering, the console guard, import hygiene). The
scan/train/verify sequence itself is the thing the pipeline exists to run for real.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def section(t):
    print(f"\n{t}")


section("1. import hygiene")
heavy = ("torch", "ultralytics", "ai2thor", "cv2")
before = {m for m in heavy if m in sys.modules}
import perception_pipeline as PP                                     # noqa: E402
check("importing the pipeline pulls in no heavy dependency",
      {m for m in heavy if m in sys.modules} == before, str({m for m in heavy if m in sys.modules} - before))

section("2. the before/after comparison")
import oracle_eval as OE                                             # noqa: E402

old = OE.Tally()
old.add([("Egg_whole", (0, 0, 19, 19))], [])                                        # never detected
old.add([("Apple_whole", (0, 0, 19, 19))], [("Apple_whole", (0, 0, 19, 19), 0.9)])  # found
old.add([("Fridge_open", (0, 0, 99, 99))], [("Fridge_closed", (0, 0, 99, 99), 0.8)])  # state swap
new = OE.Tally()
new.add([("Egg_whole", (0, 0, 19, 19))], [("Egg_whole", (0, 0, 19, 19), 0.7)])      # now found
new.add([("Apple_whole", (0, 0, 19, 19))], [])                                      # REGRESSION
new.add([("Fridge_open", (0, 0, 99, 99))], [("Fridge_open", (0, 0, 99, 99), 0.8)])  # state now right
cmp = PP.compare(old.summary(), new.summary())
rows = {r["cls"]: r for r in cmp["rows"]}

check("a class the new model now finds shows a positive recall delta",
      rows["Egg_whole"]["recall_delta"] == 1.0, str(rows["Egg_whole"]))
check("a class it lost shows a negative delta", rows["Apple_whole"]["recall_delta"] == -1.0,
      str(rows["Apple_whole"]))
check("regressions are listed explicitly", [r["cls"] for r in cmp["regressions"]] == ["Apple_whole"],
      str(cmp["regressions"]))
check("the worst regression is the FIRST row — what got worse is read before the headline",
      cmp["rows"][0]["cls"] == "Apple_whole", cmp["rows"][0]["cls"])
check("state swaps are compared", cmp["state_swaps_before"] == 1 and cmp["state_swaps_after"] == 0,
      str((cmp["state_swaps_before"], cmp["state_swaps_after"])))
check("never-detected counts are compared", cmp["never_detected_before"] >= 1)
check("a class present on only one side is None, not 0.0",
      rows["Fridge_closed"]["recall_after"] is None and rows["Fridge_closed"]["recall_before"] is None,
      str(rows["Fridge_closed"]))
check("mAP delta is computed when both sides have one",
      cmp["mAP50_delta"] is not None and abs(cmp["mAP50_delta"] - (cmp["mAP50_after"] - cmp["mAP50_before"])) < 1e-12)
empty = PP.compare(OE.Tally().summary(), OE.Tally().summary())
check("two empty runs compare without crashing, and report no mAP", empty["mAP50_delta"] is None)
check("top limits the rows", len(PP.compare(old.summary(), new.summary(), top=1)["rows"]) == 1)

section("3. rendering")
txt = PP.render_comparison(cmp)
check("names the comparison honestly", "same poses" in txt and "ground truth" in txt)
check("reports how many classes got worse", "got WORSE" in txt)
check("is pure ASCII (WSL and PowerShell consoles)", all(ord(c) < 128 for c in txt))
check("an empty comparison still renders", "BEFORE vs AFTER" in PP.render_comparison(empty))

section("4. the console guard")
check("console_is_running returns a bool and never raises", isinstance(PP.console_is_running(port=1), bool))
check("an unbound port reads as not running", PP.console_is_running(port=1) is False)

section("5. it cannot overwrite the deployed detector")
src = open(os.path.join(HERE, "perception_pipeline.py"), encoding="utf-8").read()
check("new weights go to a stamped, base-named file", 'f"{scene_name}.{stem}.{stamp}.pt"' in src)
check("the new scan goes to a separate dataset root", '"datasets_v2"' in src)
check("it refuses to start while the console holds the GPU", "console_is_running() and not args.force" in src)
check("it frees the renderer before training", "agent.controller.stop()" in src)
check("the baseline is persisted before training starts", ".baseline.json" in src
      and src.index(".baseline.json") < src.index("gen_dataset.train_scene"))
check("a reused baseline must match seed and pose count", "(prior.get(\"seed\"), prior.get(\"poses\")) != (args.seed, args.poses)" in src)
check("training batch is overridable (shared GPU)", "batch=(args.batch or None)" in src)
gd = open(os.path.join(HERE, "gen_dataset.py"), encoding="utf-8").read()
check("training restores CUDA_VISIBLE_DEVICES (else the verify step's simulator cannot start)",
      '_cvd_before = os.environ.get("CUDA_VISIBLE_DEVICES")' in gd and 'os.environ.pop("CUDA_VISIBLE_DEVICES", None)' in gd
      and gd.index("_cvd_before = ") < gd.index("results = model.train(") < gd.index('os.environ.pop("CUDA_VISIBLE_DEVICES"'))
check("the pipeline can skip straight to verification of existing weights", "--verify-weights" in src)
check("the ground-truth gate is a recorded option, applied to the agent AND the area filter",
      "visibility_distance=args.gt_visibility" in src and "min_area=args.gt_min_area" in src
      and '"gt": {"visibility": args.gt_visibility, "min_area": args.gt_min_area}' in src)
check("a reused baseline must have been scored under the same ground-truth gate", "prior_gt != report[\"gt\"]" in src)
oe = open(os.path.join(HERE, "oracle_eval.py"), encoding="utf-8").read()
check("oracle_eval's CLI exposes and records the same gate",
      "--gt-visibility" in oe and "--gt-min-area" in oe and '"gt": {"visibility": args.gt_visibility' in oe)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("PERCEPTION-PIPELINE SELFTEST OK")
