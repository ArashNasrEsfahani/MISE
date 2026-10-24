"""Re-scan, re-train, and prove it: one command that answers "is the new detector actually better?"

    # inside WSL, with the web console STOPPED (one AI2-THOR controller at a time on the 8 GB card)
    python sim/perception_pipeline.py --scene FloorPlan1

What it does, in order, and why each step is where it is:

  1. BASELINE  — score the existing detector with `oracle_eval` at N freshly sampled poses. First, so the
                 comparison is against the model that is actually deployed, measured the same way.
  2. SCAN      — a full `scan_scene` with the container pass on, into a NEW dataset root. The June dataset is
                 left untouched: it is what the thesis detector was trained on.
  3. TRAIN     — fine-tune the new recipe's base (yolo11s by default) into a NEW weights file. The deployed
                 `weights/<scene>.pt` is never overwritten here; promoting the new model is a separate, human
                 decision made after reading step 4.
  4. VERIFY    — score the new detector at the SAME poses (same seed, scene reloaded clean first), and print
                 the two side by side.

Why the oracle and not a val split: the June dataset's own benchmark reports a leakage probe of 0.98 — its val
frames sit next to their training frames — so its mAP50 0.732 was measured on near-copies of what it trained on.
AI2-THOR's instance masks give a perfect label at any pose, so both models are scored against the simulator at
poses neither has seen. See docs/perception-upgrade.md.

The controller is stopped before training and a fresh one started for verification: YOLO training and a live
Unity renderer on one 8 GB card is exactly the combination the recipe's batch size had to guess about.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def console_is_running(port: int = 8000) -> bool:
    """True if the web console answers. It owns an AI2-THOR controller; a second one would contend for the GPU
    and, worse, both would be driving the SAME scene state the evaluation depends on."""
    import http.client
    try:
        h = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        h.request("GET", "/status")
        ok = h.getresponse().status == 200
        h.close()
        return ok
    except Exception:
        return False


def compare(before: dict, after: dict, top: int = 20) -> dict:
    """Side-by-side per-class comparison of two `oracle_eval.Tally.summary()` results. Pure; tested headless.

    Rows are ordered by the size of the recall change, largest regression first — the thing a reader most needs
    to see before promoting a model is what got WORSE, not the headline mean. A class missing from one side is
    reported with `None`, never 0.0, so "never appeared" is not confused with "never detected".
    """
    b = {r["cls"]: r for r in (before or {}).get("rows", [])}
    a = {r["cls"]: r for r in (after or {}).get("rows", [])}
    rows = []
    for cls in sorted(set(b) | set(a)):
        rb, ra = b.get(cls) or {}, a.get(cls) or {}
        r0, r1 = rb.get("recall"), ra.get("recall")
        delta = (r1 - r0) if (r0 is not None and r1 is not None) else None
        rows.append({"cls": cls, "gt_before": rb.get("gt", 0), "gt_after": ra.get("gt", 0),
                     "recall_before": r0, "recall_after": r1, "recall_delta": delta,
                     "ap_before": rb.get("ap"), "ap_after": ra.get("ap")})
    rows.sort(key=lambda r: (r["recall_delta"] is None, r["recall_delta"] if r["recall_delta"] is not None else 0.0))
    swaps = lambda s: sum(v.get("swapped", 0) for v in (s or {}).get("state_swaps", {}).values())
    m0, m1 = (before or {}).get("mAP50"), (after or {}).get("mAP50")
    return {
        "mAP50_before": m0, "mAP50_after": m1,
        "mAP50_delta": (m1 - m0) if (m0 is not None and m1 is not None) else None,
        "never_detected_before": len((before or {}).get("never_detected", [])),
        "never_detected_after": len((after or {}).get("never_detected", [])),
        "state_swaps_before": swaps(before), "state_swaps_after": swaps(after),
        "regressions": [r for r in rows if r["recall_delta"] is not None and r["recall_delta"] < 0],
        "rows": rows if not top else rows[:top],
    }


def render_comparison(cmp: dict) -> str:
    """The side-by-side as text. ASCII only — it is printed to WSL and PowerShell consoles."""
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"
    d = lambda v: "   -  " if v is None else f"{v:+6.3f}"
    L = ["BEFORE vs AFTER - same poses, scored against simulator ground truth", ""]
    L.append(f"  mAP50 (tested classes)   {f(cmp['mAP50_before'])} -> {f(cmp['mAP50_after'])}   ({d(cmp['mAP50_delta'])})")
    L.append(f"  classes never detected   {cmp['never_detected_before']:>6} -> {cmp['never_detected_after']:<6}")
    L.append(f"  state swaps              {cmp['state_swaps_before']:>6} -> {cmp['state_swaps_after']:<6}")
    L.append(f"  classes that got WORSE   {len(cmp['regressions'])}")
    L.append("")
    L.append(f"  {'class':<26}{'recall before':>14}{'after':>8}{'delta':>8}{'AP50 before':>13}{'after':>8}")
    for r in cmp["rows"]:
        L.append(f"  {r['cls']:<26}{f(r['recall_before']):>14}{f(r['recall_after']):>8}{d(r['recall_delta']):>8}"
                 f"{f(r['ap_before']):>13}{f(r['ap_after']):>8}")
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--poses", type=int, default=80, help="evaluation poses per model (same poses for both)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-weights", default="",
                    help="detector to beat (default: the deployed weights/<scene>.pt)")
    ap.add_argument("--dataset-root", default="", help="where the NEW scan goes (default: sim/datasets_v2)")
    ap.add_argument("--base", default="", help="YOLO base checkpoint (default: yolo_recipe's, i.e. yolo11s.pt)")
    ap.add_argument("--epochs", type=int, default=0, help="0 = the recipe's default")
    ap.add_argument("--skip-scan", action="store_true", help="reuse an existing dataset under --dataset-root")
    ap.add_argument("--no-containers", action="store_true", help="scan WITHOUT the container pass (for an ablation)")
    ap.add_argument("--max-positions", type=int, default=24, help="coverage positions (8 frames each)")
    ap.add_argument("--per-class-target", type=int, default=40, help="frames per class the balance pass aims for")
    ap.add_argument("--max-containers", type=int, default=10)
    ap.add_argument("--balance-frames", type=int, default=12, help="balance frames per starved class, per scan")
    ap.add_argument("--max-state-flips", type=int, default=16, help="objects flipped to photograph both state halves")
    ap.add_argument("--batch", type=int, default=0,
                    help="training batch (0 = the recipe's). Lower it when another process shares the GPU: an "
                         "oversubscribed 8 GB card under WSL pages to host memory and training stalls, not fails")
    ap.add_argument("--baseline-json", default="",
                    help="reuse a baseline summary written by an earlier run instead of re-scoring the old model")
    ap.add_argument("--gt-visibility", type=float, default=1.5,
                    help="ground-truth visibility gate (m). Must match a labeller: 1.5 = June scan, 6.0 = this scan")
    ap.add_argument("--gt-min-area", type=int, default=400,
                    help="ground-truth minimum box area (px^2): 400 = June scan, 150 = this scan")
    ap.add_argument("--verify-weights", default="",
                    help="skip scan and training: score these already-trained weights against the baseline")
    ap.add_argument("--force", action="store_true", help="run even if the console answers on :8000")
    args = ap.parse_args(argv)

    if console_is_running() and not args.force:
        print("STOP: the web console is running on :8000. It owns an AI2-THOR controller on the same GPU and the")
        print("same scene state this evaluation depends on. Stop it first (or pass --force if you are sure).")
        return 2

    import oracle_eval as OE
    import perception
    import gen_dataset
    import yolo_recipe
    import thor_actions as ta

    weights_dir = os.path.join(HERE, "weights")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    baseline = args.baseline_weights or perception.weights_path(args.scene, weights_dir)
    if not baseline:
        print(f"no deployed detector for {args.scene!r} to use as the baseline - pass --baseline-weights")
        return 2
    root = args.dataset_root or os.path.join(HERE, "datasets_v2")
    base = args.base or yolo_recipe.resolve_base()
    report = {"scene": args.scene, "seed": args.seed, "poses": args.poses, "started": stamp,
              "gt": {"visibility": args.gt_visibility, "min_area": args.gt_min_area},
              "baseline_weights": baseline, "dataset_root": root, "base": base}

    def new_agent():
        return ta.ThorAgent(scene=args.scene, render_instance_seg=True, visibility_distance=args.gt_visibility)

    def score(agent, weights, label):
        det = perception.Detector(weights)
        agent.reset(agent.scene)                         # the SAME clean scene for both models
        poses = OE.sample_poses(agent, args.poses, seed=args.seed)
        box = {}
        print(f"\n[{label}] scoring {os.path.basename(weights)} at {len(poses)} poses (seed {args.seed})", flush=True)
        for i, msg in enumerate(OE.evaluate_live(agent, det, poses, min_area=args.gt_min_area,
                                                 on_pose=lambda _i, t: box.__setitem__("t", t))):
            if i % 10 == 0 or i == len(poses) - 1:
                print("   " + msg, flush=True)
        s = box["t"].summary() if "t" in box else OE.Tally().summary()
        print(OE.render_report(s, f"{args.scene} [{label}]", top=15), flush=True)
        return s

    # ---- 1. baseline --------------------------------------------------------------------------------------
    agent = new_agent()
    scene_name = agent.scene
    if args.baseline_json:
        with open(args.baseline_json, encoding="utf-8") as fh:
            prior = json.load(fh)
        prior_gt = prior.get("gt") or {"visibility": 1.5, "min_area": 400}
        if prior_gt != report["gt"]:
            print(f"STOP: {args.baseline_json} was scored under ground-truth gate {prior_gt}, not {report['gt']}")
            return 2
        if (prior.get("seed"), prior.get("poses")) != (args.seed, args.poses):
            print(f"STOP: {args.baseline_json} was scored at seed={prior.get('seed')} poses={prior.get('poses')}, "
                  f"not seed={args.seed} poses={args.poses} - the comparison would not be at the same poses")
            return 2
        report["before"] = prior["before"]
        report["baseline_reused_from"] = args.baseline_json
        print(f"[baseline] reused from {args.baseline_json}", flush=True)
    else:
        report["before"] = score(agent, baseline, "baseline")
        # Persist the baseline NOW: training is the step most likely to be interrupted, and re-scoring the old
        # model costs a simulator session.
        with open(os.path.join(HERE, f"perception_pipeline_{args.scene}_{stamp}.baseline.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({k: report[k] for k in ("scene", "seed", "poses", "gt", "baseline_weights", "before")}, fh,
                      ensure_ascii=False, indent=2, default=str)

    # ---- 2. scan -------------------------------------------------------------------------------------------
    dataset_dir = os.path.join(root, scene_name)
    if args.verify_weights:
        args.skip_scan = True
    if not args.skip_scan:
        print(f"\n[scan] {scene_name} -> {dataset_dir}  (containers: {not args.no_containers})", flush=True)
        t0 = time.time()
        n = 0
        scan_kw = {"max_positions": args.max_positions, "per_class_target": args.per_class_target,
                   "max_containers": args.max_containers, "balance_frames": args.balance_frames,
                   "max_state_flips": args.max_state_flips}
        report["scan"] = dict(scan_kw, containers=not args.no_containers)
        for msg in agent.scan_scene(dataset_root=root, do_containers=not args.no_containers,
                                    do_destructive=False, **scan_kw):
            n += 1
            # the plan, the state/balance outcomes and the coverage verdict are always printed; the rest is sampled
            if n % 25 == 0 or str(msg).startswith(("scan ", "estimate", "containers:", "states", "balance",
                                                   "coverage:", "histogram")):
                print(f"   {msg}", flush=True)
        report["scan_seconds"] = round(time.time() - t0, 1)
        print(f"[scan] done in {report['scan_seconds']} s", flush=True)
    if not os.path.isfile(os.path.join(dataset_dir, "data.yaml")):
        print(f"no data.yaml under {dataset_dir} - the scan produced nothing to train on")
        return 3

    # Free the renderer before training: yolo11s at batch 8 alongside a live Unity on 8 GB is a guess.
    try:
        agent.controller.stop()
    except Exception:
        pass
    agent = None

    # ---- 3. train ------------------------------------------------------------------------------------------
    if args.verify_weights:
        weights_out = os.path.abspath(args.verify_weights)
        report["train"] = {"weights_path": weights_out, "skipped": "verify-only run (--verify-weights)"}
        if not os.path.isfile(weights_out):
            print(f"no weights at {weights_out}")
            return 4
    else:
        stem = os.path.splitext(os.path.basename(base))[0]
        weights_out = os.path.join(weights_dir, f"{scene_name}.{stem}.{stamp}.pt")
        print(f"\n[train] base {base} -> {weights_out}", flush=True)
        t0 = time.time()
        res = gen_dataset.train_scene(scene_name, dataset_dir, weights_out, base=base,
                                      epochs=(args.epochs or None), batch=(args.batch or None))
        report["train_seconds"] = round(time.time() - t0, 1)
        report["train"] = {k: res.get(k) for k in ("weights_path", "base", "n_images", "recipe")}
        report["train"]["summary"] = (res.get("metrics") or {}).get("summary")
        if not os.path.isfile(weights_out):
            print("training produced no weights file - see the ultralytics output above")
            return 4
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)     # see gen_dataset.train_scene: AI2-THOR misreads it under WSL

    # ---- 4. verify -----------------------------------------------------------------------------------------
    agent = new_agent()
    report["after"] = score(agent, weights_out, "new")
    cmp = compare(report["before"], report["after"], top=30)
    report["comparison"] = cmp
    print("\n" + render_comparison(cmp), flush=True)

    out = os.path.join(HERE, f"perception_pipeline_{args.scene}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote {out}")
    print(f"the deployed detector was NOT replaced. To promote the new one, copy {os.path.basename(weights_out)}")
    print(f"over weights/{scene_name}.pt (the Train button archives the old file first; a manual copy does not).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
