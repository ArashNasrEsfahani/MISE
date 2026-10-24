"""Offline selftest of k23_perception_test.py (WSL thor-env: ultralytics' ap_per_class and fork workers; no GPU, no simulator, no
Kitchen 2/3 data). Checks: whole-episode resampling; the paired bootstrap on ultralytics' ap_per_class (full sample == ul_map, identical
systems -> [0, 0], a better system -> CI above 0, parallel == serial); the freeze verification; the evaluation-code binding; the step
order and once-only rules; an unfinished attempt needs a documented recovery that moves partial outputs aside (an opened holdout stays
recorded); incomplete split metrics and a coverage-pose shortfall fail the step with the report kept; LIVE persists the realistic
result before coverage and a recovered attempt runs coverage only; the training-time parser marks unmatched STARTs."""
import json
import os
import random
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np                  # noqa: E402
import k23_perception_test as P     # noqa: E402

ok_n = fail_n = 0


def check(name, cond):
    global ok_n, fail_n
    if cond:
        ok_n += 1
    else:
        fail_n += 1
        print("FAIL", name)


def refused(fn, exc=RuntimeError):
    try:
        fn()
        return False
    except exc:
        return True


def synth_stats(rng, n_img, quality):
    """Per-image validator-like statistics: tp [n x 10] monotone over IoU thresholds, conf, pred_cls, target_cls."""
    out = []
    for i in range(n_img):
        nt = rng.randint(0, 5)
        tcls = np.array([rng.randint(0, 3) for _ in range(nt)], dtype=float)
        preds = []
        for c in tcls:
            hit = rng.random() < quality
            k = rng.randint(3, 10) if hit else 0                         # matched at the first k IoU thresholds
            preds.append(([True] * k + [False] * (10 - k), rng.uniform(.4, 1) if hit else rng.uniform(0, .6), c))
        for _ in range(rng.randint(0, 2)):
            preds.append(([False] * 10, rng.uniform(0, .5), float(rng.randint(0, 3))))
        out.append({"tp": np.array([p[0] for p in preds], dtype=bool).reshape(-1, 10), "conf": np.array([p[1] for p in preds]),
                    "pred_cls": np.array([p[2] for p in preds]), "target_cls": tcls, "im_name": f"FloorPlan99_e{i // 6:03d}_{i:04d}.jpg"})
    return out


def main():
    rng = random.Random(3)
    # A. whole-episode resampling
    eps = [i // 6 for i in range(60)]
    res, n_eps = P.episode_resamples(eps, n=50, seed=0)
    by = {}
    for i, e in enumerate(eps):
        by.setdefault(e, []).append(i)
    whole = all(sorted(idx) == sorted(i for e in [eps[j] for j in idx[::6]] for i in by[e]) and len(idx) == 60 for idx in res)
    check("resamples draw whole episodes (every drawn episode contributes all its images, n_eps draws each)", n_eps == 10 and whole)
    check("resampling is deterministic for a seed", P.episode_resamples(eps, n=50, seed=0)[0] == res)

    # B. the paired bootstrap on ultralytics' ap_per_class
    from ultralytics.utils.metrics import ap_per_class
    a = synth_stats(rng, 72, .55)
    b = synth_stats(random.Random(3), 72, .55)                           # same images, regenerated -> identical to `a`
    better = synth_stats(random.Random(3), 72, .55)
    for s in better:                                                      # every true positive also at the strictest IoUs
        s["tp"] = np.repeat(s["tp"][:, :1], 10, axis=1)
    e72 = [int(s["im_name"].split("_e")[1].split("_")[0]) for s in a]
    full = list(range(72))
    cat = {k: np.concatenate([s[k] for s in a], 0) for k in ("tp", "conf", "pred_cls", "target_cls")}
    ap = ap_per_class(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[5]
    check("ul_map == ap_per_class on the concatenated statistics (mAP50, mAP50-95)",
          P.ul_map(a, full) == (float(ap[:, 0].mean()), float(ap.mean())))
    same = P.paired_bootstrap(a, b, e72, n=40, workers=1)
    check("identical systems: zero difference, CI [0, 0]", same["diff_point"] == {"mAP50": 0.0, "mAP50_95": 0.0}
          and same["boot_diff"]["mAP50"]["ci95"] == [0.0, 0.0] and same["boot_diff"]["mAP50_95"]["ci95"] == [0.0, 0.0])
    bt = P.paired_bootstrap(a, better, e72, n=200, workers=1)
    check("full-sample point values == ul_map", bt["point"]["a"]["mAP50_95"] == P.ul_map(a, full)[1]
          and bt["point"]["b"]["mAP50_95"] == P.ul_map(better, full)[1])
    check("the stricter-IoU-better system: mAP50-95 difference > 0 with the CI above 0, mAP50 unchanged",
          bt["diff_point"]["mAP50_95"] > 0 and bt["boot_diff"]["mAP50_95"]["ci95"][0] > 0 and abs(bt["diff_point"]["mAP50"]) < 1e-12)
    par = P.paired_bootstrap(a, better, e72, n=200, workers=3)
    check("parallel (3 fork workers) == serial", json.dumps(par, sort_keys=True) == json.dumps(bt, sort_keys=True))
    check("misaligned statistics are refused", refused(lambda: P.paired_bootstrap(a, better[:-1], e72, n=5, workers=1), ValueError))

    # C-E. freeze, code binding, order, recovery, failing steps (temp folder; module paths redirected)
    tmp = tempfile.mkdtemp(prefix="k23perc_")
    SC = "FloorPlan17"
    for d in ("rep", "docs", "code", "work", "thor", "train"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    f1 = os.path.join(tmp, "a.pt")
    open(f1, "wb").write(b"weights")
    rec, meta, fz = os.path.join(tmp, "rep", "rec.sha256"), os.path.join(tmp, "rep", "meta.json"), os.path.join(tmp, "docs", "freeze.json")
    open(rec, "w").write(f"{P.sha_file(f1)}  {f1}\n")
    json.dump({"test_evaluation_blocked": False, "coco_pending": False}, open(meta, "w"))
    json.dump({"perception": {"record_sha256": P.sha_file(rec), "meta_sha256": P.sha_file(meta)}}, open(fz, "w"))
    P.record_paths = lambda scene: (rec, meta, fz)
    P.work = lambda scene: os.path.join(tmp, "work")
    P.REPO = os.path.join(tmp, "code")
    P.CODE_FILES = ("w.py", "r.sh")
    for c in P.CODE_FILES:
        open(os.path.join(P.REPO, c), "w").write("code " + c)
    poses = [os.path.join(tmp, "live_eval_poses.json"), os.path.join(tmp, "live_cover_poses.json")]
    outs = {"poses": poses, "primary": [os.path.join(tmp, "eval_yolo26s.json")], "fused": [os.path.join(tmp, "amb_eval.json")],
            "state": [os.path.join(tmp, "state_test.json")], "holdout": [os.path.join(tmp, "amb_eval_final.json")]}
    P.outputs = lambda scene: outs
    check("verify: a consistent freeze passes", P.verify_freeze(SC)["files_verified"] == 1)
    check("a step before `bind` is refused", refused(lambda: P.begin(SC, "poses")))
    P.bind_code(SC)
    check("bind runs once", refused(lambda: P.bind_code(SC)))
    open(os.path.join(P.REPO, "w.py"), "a").write("#")
    check("code changed after bind is refused", refused(lambda: P.begin(SC, "poses")))
    open(os.path.join(P.REPO, "w.py"), "w").write("code w.py")
    check("order: primary before poses is refused", refused(lambda: P.begin(SC, "primary")))
    check("coco is not part of FloorPlan17's protocol", refused(lambda: P.begin(SC, "coco")))

    os.environ["K23P_STAMP"] = "T0"

    def fin(step):
        P.finish(SC, step)
        P.supervisor_outcome(SC, step, "T0", True)

    # poses: a coverage shortfall fails the step (stubbed Kitchen 1 modules)
    class Agent:
        def reset(self, s): pass

        def stop(self): pass
    sys.modules["build_kitchen_data"] = types.SimpleNamespace(make_agent=lambda s: Agent())
    state = {"cover": 299, "live": 150, "live_calls": 0, "fail_cover": False, "close_err": False}

    def frozen_poses(scene, agent):
        open(poses[0], "w").write(json.dumps([0] * state["live"]))
        return [0] * state["live"]

    def coverage_poses(scene, agent):
        open(poses[1], "w").write(json.dumps([0] * state["cover"]))
        return [0] * state["cover"]

    def evaluate(scene, model, live=False):
        r = {k: {"mAP50": .9, "mAP50_95": .8} for k in ("val", "test", "test_close")}
        if state["close_err"]:
            r["test_close"] = {"error": "boom"}
        json.dump(r, open(outs["primary"][0], "w"))
        return r

    def live_eval(scene, model, which):
        state["live_calls"] += 1
        if which == "cover" and state["fail_cover"]:
            raise OSError("simulator died")
        return {"mAP50": .7, "mAP50_95": .5, "poses": 150, "skipped": 0, "gt_boxes": 9}
    fake_tk = types.SimpleNamespace(LIVE_POSES=150, COVER_POSES=300, frozen_poses=frozen_poses, coverage_poses=coverage_poses,
                                    evaluate=evaluate, live_eval=live_eval,
                                    _dump=lambda p, o: json.dump(o, open(p, "w")))
    P.tk_mod = lambda: fake_tk
    P.begin(SC, "poses")
    check("a coverage-pose shortfall fails the step (StepFailed)", refused(lambda: P.step_poses(SC), P.StepFailed))
    check("an unfinished attempt is refused without a recovery", refused(lambda: P.begin(SC, "poses")))
    check("a recovery needs a reason and evidence", refused(lambda: P.recover(SC, "poses", "", "")))
    r = P.recover(SC, "poses", "test: induced shortfall", "selftest stub")
    check("recovery moves the partial outputs aside with hashes (never deleted)", set(r["moved"]) == {os.path.basename(p) for p in poses}
          and all(os.path.exists(os.path.join(r["moved_to"], n)) for n in r["moved"]) and not any(os.path.exists(p) for p in poses))
    state["cover"], state["live"] = 300, 127
    P.begin(SC, "poses")
    pr = P.step_poses(SC)
    check("a rule-limited live set (the 20,000-draw cap: 127 / 150) with full coverage is accepted and flagged",
          pr["live_poses"] == 127 and pr.get("live_rule_limited") is True and pr["cover_poses"] == 300)
    P.finish(SC, "poses", {"result": pr})
    P.supervisor_outcome(SC, "poses", "T0", True)
    check("once-only: poses again is refused", refused(lambda: P.begin(SC, "poses")))
    open(poses[0], "w").write("[1]")
    check("a pose file changed after hashing is refused", refused(lambda: P.begin(SC, "primary")))
    open(poses[0], "w").write(json.dumps([0] * 127))
    # primary: a CLOSE-TEST error keeps the report and fails the step
    state["close_err"] = True
    P.begin(SC, "primary")
    check("an incomplete CLOSE-TEST fails the step (StepFailed), report kept", refused(lambda: P.step_primary(SC), P.StepFailed)
          and os.path.exists(outs["primary"][0]))
    P.recover(SC, "primary", "test: induced CLOSE error", "selftest stub")
    state["close_err"] = False
    P.begin(SC, "primary")
    P.step_primary(SC)
    fin("primary")
    # live: realistic persisted before coverage; a recovered attempt runs coverage only
    state["fail_cover"] = True
    P.begin(SC, "live")
    check("a coverage failure raises", refused(lambda: P.step_live(SC), OSError))
    d = json.load(open(outs["primary"][0]))
    check("the realistic LIVE result was persisted before coverage started", "live" in d and "live_cover" not in d)
    P.recover(SC, "live", "test: simulator died", "selftest stub")
    state["fail_cover"], state["live_calls"] = False, 0
    P.begin(SC, "live")
    out = P.step_live(SC)
    check("the recovered LIVE attempt runs coverage only", state["live_calls"] == 1 and out["realistic_reused_from_recovered_attempt"])
    check("LIVE reads the rule-limited set back as saved (train_kitchen.LIVE_POSES set to the recorded 127)",
          fake_tk.LIVE_POSES == 127 and out["live_poses_used_from_rule"] == 127)
    fin("live")
    # supervisor acceptance (Astra r3 4): a step_done counts only with ok=1 for the same attempt stamp, or a documented disposition
    os.environ["K23P_STAMP"] = "S1"
    P.begin(SC, "fused")
    open(outs["fused"][0], "w").write("{}")
    P.finish(SC, "fused")
    check("a finished step without a supervisor outcome is not done; it and the next step are refused",
          "fused" not in P.done_steps(SC) and refused(lambda: P.begin(SC, "fused")) and refused(lambda: P.begin(SC, "state")))
    P.supervisor_outcome(SC, "fused", "OTHER", True)
    P.supervisor_outcome(SC, "fused", "S1", False)
    check("an ok outcome of another attempt, or a failed outcome of this one, does not accept it", "fused" not in P.done_steps(SC))
    check("accept needs a reason and evidence", refused(lambda: P.accept(SC, "fused", "", "")))
    P.accept(SC, "fused", "guard rc logged as killed after the step finished", "selftest stub")
    check("a documented disposition accepts the kept result (no re-run)", "fused" in P.done_steps(SC))
    os.environ["K23P_STAMP"] = "S2"
    P.begin(SC, "state")
    open(outs["state"][0], "w").write("{}")
    P.finish(SC, "state")
    r2 = P.recover(SC, "state", "test: supervisor failed", "selftest stub")
    check("recover on an unaccepted result moves it aside and makes the step runnable again",
          "state_test.json" in r2["moved"] and not os.path.exists(outs["state"][0]) and not refused(lambda: P.begin(SC, "state")))
    open(outs["state"][0], "w").write("{}")
    P.finish(SC, "state")
    P.supervisor_outcome(SC, "state", "S2", True)
    check("an ok outcome of the same attempt accepts the step", "state" in P.done_steps(SC))
    # holdout: an opened seal stays recorded through a recovery
    os.environ["K23P_STAMP"] = "S3"
    P.begin(SC, "holdout")
    P.ledger(SC, event="holdout_seal_opened", n_images=1)
    open(outs["holdout"][0], "w").write("{}")
    check("a holdout attempt that opened the seal and died is refused without a recovery", refused(lambda: P.begin(SC, "holdout")))
    rr = P.recover(SC, "holdout", "test: crash", "selftest stub")
    check("the holdout recovery records that the seal was already opened", rr["holdout_already_opened"] and rr["note"])
    del os.environ["K23P_STAMP"]
    open(f1, "wb").write(b"changed")
    check("verify: a changed frozen file is refused", refused(lambda: P.verify_freeze(SC)))
    open(f1, "wb").write(b"weights")
    json.dump({"test_evaluation_blocked": True}, open(meta, "w"))
    check("verify: a meta that blocks TEST (or no longer matches the agent freeze) is refused", refused(lambda: P.verify_freeze(SC)))
    # re-binding the evaluation code: refused once a step is done, allowed (old manifest moved aside) before any step
    check("rebind is refused once a step of the kitchen is done", refused(lambda: P.rebind_code(SC, "t", "t")))
    json.dump({"test_evaluation_blocked": False, "coco_pending": False}, open(meta, "w"))         # undo the tamper checks above
    json.dump({"perception": {"record_sha256": P.sha_file(rec), "meta_sha256": P.sha_file(meta)}}, open(fz, "w"))
    w2 = os.path.join(tmp, "work2")
    os.makedirs(w2, exist_ok=True)
    P.work = lambda scene: w2
    P.bind_code(SC)
    check("rebind needs a reason and evidence", refused(lambda: P.rebind_code(SC, "", "")))
    rb = P.rebind_code(SC, "test: code changed before any step", "selftest")
    check("rebind before any step moves the old manifest aside and binds the current code",
          rb["event"] == "eval_code_bound" and any(r.get("event") == "eval_code_unbound" for r in P.ledger_rows(SC))
          and os.path.exists(P.code_manifest_path(SC)))
    # training-time parser
    P.HERE, P.THOR, P.SIM = tmp, os.path.join(tmp, "thor"), tmp
    os.makedirs(os.path.join(tmp, SC, "train"), exist_ok=True)
    open(os.path.join(tmp, SC, "train", "train_m.log"), "w").write(
        "2026-10-18 10:00:00 START\n2026-10-18 11:00:00 END rc=0\n2026-10-18 12:00:00 START\n2026-10-18 13:00:00 START\n2026-10-18 13:30:00 END rc=0\n")
    tr = P.training_record(SC, "m")
    check("training time: matched sessions summed, unmatched STARTs counted, total marked incomplete",
          tr["matched_sessions_wall_hours"] == 1.5 and tr["unmatched_starts"] == 1 and tr["gpu_hours_complete"] is False)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{ok_n}/{ok_n + fail_n}")
    return 1 if fail_n else 0


if __name__ == "__main__":
    sys.exit(main())
