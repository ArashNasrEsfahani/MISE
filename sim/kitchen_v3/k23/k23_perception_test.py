"""Kitchen 2/3 perception TEST evaluations (A28 item 9, A38 item 4), after each kitchen's perception freeze.

Arash's go (2026-10-21 morning): asked "Perception TEST evaluations: no Kitchen 2/3 script exists yet ... I'll build them with you",
he answered "you can do all of these three now", then "go for it all. i want everything done.", then "Ok while it's running ...".

A thin wrapper: Kitchen 1's evaluation code stays byte-identical and is called as it is, pointed at the kitchen through module
globals inside this process (the k23_amb_train.py pattern):
  sim/train_kitchen.py   frozen_poses / coverage_poses / evaluate / live_eval      (primary TEST, CLOSE-TEST, LIVE)
  sim/train_amb.py       evaluate(False) / evaluate(True)                          (fused amb TEST, the sealed holdout)
  sim/kitchen_v3/k23/k23_state_calib.py test                                       (state TEST)
Steps, each ONCE, in the registered order (A28 item 9), per kitchen, FloorPlan15 before FloorPlan17 (A28 item 2):
  bind     the hashes of this evaluation code (wrapper, runner, selftest), linked to the frozen record (Astra K23-PERCEPTION-TEST-r1 9)
  poses    (simulator) the 150 LIVE poses (seed 20260929, Kitchen 1 rule) and the 300 coverage poses, both complete, written and
           HASHED before the first primary evaluation
  primary  (GPU)  primary VAL / TEST / CLOSE-TEST                     -> sim/kitchen_v3/<scene>/eval_yolo26s.json
  coco     (GPU, FloorPlan15 only, A38 item 4) the COCO-initialized primary: ultralytics VAL / TEST / CLOSE (eval_yolo26s_coco.json)
           + a paired episode bootstrap of the ultralytics TEST mAP50 and mAP50-95 (COCO - K1-init, 95 % CI) on the validator's own
           per-image statistics -> coco_sensitivity.json; it never selects the deployed model
  live     (simulator) LIVE realistic (persisted at once) then coverage, added to eval_yolo26s.json; pose hashes re-checked
  fused    (GPU)  fused amb TEST (primary / gated / ungated / control)  -> sim/kitchen_v3/<scene>/amb_eval.json
  state    (GPU)  state TEST                                           -> sim/kitchen_v3/<scene>/state_test.json
  holdout  (GPU)  the sealed holdout, once, last                        -> sim/kitchen_v3/<scene>/amb_eval_final.json
Every step first verifies the frozen record (every listed sha256), the evaluation-code hashes, the order and the pose hashes, and
refuses when it ran before, when its output exists, or when an earlier attempt started and did not finish: another attempt needs a
documented infrastructure recovery (`recover`: reason + evidence; partial outputs moved aside and hashed, never deleted; an opened
holdout stays recorded as opened). A step whose result is incomplete (e.g. CLOSE-TEST error) keeps its report and stops.
Ledger: sim/kitchen_v3/k23/<scene>/perception_test/ledger.jsonl. No result triggers anything (A28 item 9).

    (WSL thor-env, repo root, MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0, under /home/arash/sim.lock: run_perception_test.sh)
    python sim/kitchen_v3/k23/k23_perception_test.py <step> --scene FloorPlan15
    python sim/kitchen_v3/k23/k23_perception_test.py recover --scene FloorPlan15 --step live --reason "..." --evidence "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.dirname(SIM)
sys.path.insert(0, SIM)
sys.path.insert(0, HERE)
SCENES = ("FloorPlan15", "FloorPlan17")
STEPS = ("poses", "primary", "coco", "live", "fused", "state", "holdout")
ORDER = {"FloorPlan15": STEPS, "FloorPlan17": tuple(s for s in STEPS if s != "coco")}
THOR = os.path.join(os.path.expanduser("~"), "thor-data")
BOOT_N, BOOT_SEED = 1000, 0
CODE_FILES = ("sim/kitchen_v3/k23/k23_perception_test.py", "sim/kitchen_v3/k23/run_perception_test.sh",
              "sim/kitchen_v3/k23/run_perception_step.sh", "sim/kitchen_v3/k23/_k23_perception_test_selftest.py",
              "sim/kitchen_v3/k23/_run_perception_step_selftest.sh")


class StepFailed(RuntimeError):
    """A step ran but its result is incomplete: its report is kept, no step_done is written, the chain stops."""


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rep_dir(scene: str) -> str:
    return os.path.join(SIM, "kitchen_v3", scene)


def work(scene: str) -> str:
    d = os.path.join(HERE, scene, "perception_test")
    os.makedirs(d, exist_ok=True)
    return d


def outputs(scene: str) -> dict:
    r = rep_dir(scene)
    return {"poses": [os.path.join(THOR, "kitchen_v3", scene, "live_eval_poses.json"), os.path.join(THOR, "kitchen_v3", scene, "live_cover_poses.json")],
            "primary": [os.path.join(r, "eval_yolo26s.json")],
            "coco": [os.path.join(r, "eval_yolo26s_coco.json"), os.path.join(r, "coco_sensitivity.json")],
            "fused": [os.path.join(r, "amb_eval.json")], "state": [os.path.join(r, "state_test.json")],
            "holdout": [os.path.join(r, "amb_eval_final.json")]}


# ------------------------------------------------------------------------------------------------ ledger
def ledger(scene: str, **kw) -> dict:
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), scene=scene, **kw)
    with open(os.path.join(work(scene), "ledger.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return rec


def ledger_rows(scene: str) -> list:
    p = os.path.join(work(scene), "ledger.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def step_states(scene: str) -> dict:
    """Per step, from the ledger in order (Astra r3 4): `open` = a step_start with no step_done / recovery yet; `pending` = a step_done
    whose attempt (stamp) has no accepting supervisor outcome (run_perception_step.sh: guard + cleanup accepted) and no accepting
    disposition; `done` = accepted. An infra_recovery resets the step."""
    st = {}
    for r in ledger_rows(scene):
        s, ev = r.get("step"), r.get("event")
        if not s:
            continue
        d = st.setdefault(s, {"open": None, "pending": None, "done": False})
        if ev == "step_start":
            d.update(open=r, pending=None, done=False)
        elif ev == "step_done":
            d.update(open=None, pending=r, done=False)
        elif ev == "supervisor_outcome" and d["pending"] is not None and r.get("stamp") and r.get("stamp") == d["pending"].get("stamp"):
            if r.get("ok") is True:
                d.update(pending=None, done=True)
        elif ev == "supervisor_disposition" and d["pending"] is not None and r.get("accepted") is True:
            d.update(pending=None, done=True)
        elif ev == "infra_recovery":
            d.update(open=None, pending=None, done=False)
    return st


def done_steps(scene: str) -> set:
    return {s for s, d in step_states(scene).items() if d["done"]}


def open_attempt(scene: str, step: str):
    """The last step_start of `step` with neither a later step_done nor a later infra_recovery for it, else None."""
    return (step_states(scene).get(step) or {}).get("open")


def pending_acceptance(scene: str, step: str):
    return (step_states(scene).get(step) or {}).get("pending")


def supervisor_outcome(scene: str, step: str, stamp: str, ok: bool, **kw) -> dict:
    """Written by run_perception_step.sh for every launched attempt (ok only when the step exited 0, the guard's outcome was valid and
    the owned session was cleaned up)."""
    return ledger(scene, event="supervisor_outcome", step=step, stamp=stamp, ok=bool(ok), **kw)


def accept(scene: str, step: str, reason: str, evidence: str) -> dict:
    """A documented disposition that KEEPS a completed result whose supervisor acceptance is missing or failed (no re-run)."""
    if not reason.strip() or not evidence.strip():
        raise RuntimeError("REFUSED: a disposition needs a reason and evidence")
    p = pending_acceptance(scene, step)
    if p is None:
        raise RuntimeError(f"REFUSED: {step} has no result awaiting acceptance")
    return ledger(scene, event="supervisor_disposition", step=step, stamp=p.get("stamp"), accepted=True, reason=reason, evidence=evidence)


# ------------------------------------------------------------------------------------------------ freeze + code verification
def record_paths(scene: str) -> tuple:
    return (os.path.join(rep_dir(scene), "frozen_weights.sha256"), os.path.join(rep_dir(scene), "frozen_weights.meta.json"),
            os.path.join(REPO, "docs", "final-test", "k23", f"freeze_{scene}.json"))


def verify_freeze(scene: str, *, hasher=sha_file) -> dict:
    """The kitchen's perception freeze holds: record + meta are the ones the agent freeze binds, TEST is unblocked, and every listed
    file still has its frozen sha256 (paths relative to the repo root, or absolute). Raises RuntimeError on any problem."""
    rec, meta, fz = record_paths(scene)
    for p in (rec, meta, fz):
        if not os.path.exists(p):
            raise RuntimeError(f"REFUSED: {p} missing (the kitchen is not frozen, A28 item 9)")
    bind = (json.load(open(fz, encoding="utf-8")).get("perception") or {})
    rs, ms = hasher(rec), hasher(meta)
    if bind.get("record_sha256") != rs or bind.get("meta_sha256") != ms:
        raise RuntimeError(f"REFUSED: the frozen record / meta differ from freeze_{scene}.json (record {rs[:12]}, meta {ms[:12]})")
    m = json.load(open(meta, encoding="utf-8"))
    if m.get("test_evaluation_blocked") or m.get("coco_pending"):
        raise RuntimeError("REFUSED: the frozen meta still blocks TEST evaluation (A38 item 4)")
    bad, n = [], 0
    for line in open(rec, encoding="utf-8"):
        if not line.strip():
            continue
        want, path = line.split(None, 1)
        path = path.strip()
        full = path if os.path.isabs(path) else os.path.join(REPO, path)
        n += 1
        if not os.path.exists(full) or hasher(full) != want:
            bad.append(path)
    if bad:
        raise RuntimeError(f"REFUSED: {len(bad)} frozen file(s) differ from the record (first {bad[0]})")
    return {"record_sha256": rs, "meta_sha256": ms, "files_verified": n}


def code_manifest_path(scene: str) -> str:
    return os.path.join(work(scene), "eval_code.sha256")


def code_hashes(hasher=sha_file) -> dict:
    return {p: hasher(os.path.join(REPO, p)) for p in CODE_FILES}


def bind_code(scene: str, *, hasher=sha_file) -> dict:
    """Additive evaluation manifest (Astra K23-PERCEPTION-TEST-r1 9): the evaluation code's hashes, linked to the frozen record;
    written once before the first step; every step re-checks it (the code cannot change mid-protocol)."""
    p = code_manifest_path(scene)
    if os.path.exists(p):
        raise RuntimeError(f"REFUSED: {p} exists (the evaluation code is bound once)")
    v = verify_freeze(scene, hasher=hasher)
    hs = code_hashes(hasher)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for k, h in hs.items():
            f.write(f"{h}  {k}\n")
    return ledger(scene, event="eval_code_bound", code=hs, manifest_sha256=hasher(p), freeze=v)


def rebind_code(scene: str, reason: str, evidence: str, *, hasher=sha_file) -> dict:
    """A documented re-binding of the evaluation code for a kitchen in which NO step is done yet (the old manifest is moved aside,
    never deleted); refused once any step of that kitchen is done (its results stay bound to the code that produced them)."""
    if not reason.strip() or not evidence.strip():
        raise RuntimeError("REFUSED: a re-binding needs a reason and evidence")
    if done_steps(scene):
        raise RuntimeError(f"REFUSED: {scene} has done steps {sorted(done_steps(scene))}: its evaluation code cannot be re-bound")
    p = code_manifest_path(scene)
    if not os.path.exists(p):
        raise RuntimeError("REFUSED: nothing bound yet (use bind)")
    dest = os.path.join(work(scene), "aborted", time.strftime("%Y%m%dT%H%M%S") + "_eval_code")
    os.makedirs(dest, exist_ok=True)
    old = hasher(p)
    shutil.move(p, os.path.join(dest, os.path.basename(p)))
    ledger(scene, event="eval_code_unbound", old_manifest_sha256=old, moved_to=dest, reason=reason, evidence=evidence)
    return bind_code(scene, hasher=hasher)


def verify_code(scene: str, *, hasher=sha_file) -> None:
    p = code_manifest_path(scene)
    if not os.path.exists(p):
        raise RuntimeError("REFUSED: the evaluation code is not bound yet (run `bind` first)")
    want = {l.split(None, 1)[1].strip(): l.split()[0] for l in open(p, encoding="utf-8") if l.strip()}
    bound = [r for r in ledger_rows(scene) if r.get("event") == "eval_code_bound"]
    if not bound or bound[-1].get("manifest_sha256") != hasher(p):
        raise RuntimeError("REFUSED: eval_code.sha256 differs from its ledger entry")
    now = code_hashes(hasher)
    if now != want:
        raise RuntimeError(f"REFUSED: the evaluation code changed after it was bound ({sorted(k for k in now if now[k] != want.get(k))})")


def begin(scene: str, step: str, *, hasher=sha_file) -> dict:
    """Order + once-only + recovery + freeze + code + pose checks; returns the verification record. Raises RuntimeError to refuse."""
    if step not in ORDER[scene]:
        raise RuntimeError(f"REFUSED: step {step} is not part of {scene}'s protocol")
    done = done_steps(scene)
    if step in done:
        raise RuntimeError(f"REFUSED: step {step} is already done for {scene} (each evaluation runs once)")
    missing = [s for s in ORDER[scene][:ORDER[scene].index(step)] if s not in done]
    if missing:
        raise RuntimeError(f"REFUSED: {scene} {step} needs {missing} first (A28 item 9 order)")
    oa = open_attempt(scene, step)
    if oa is not None:
        raise RuntimeError(f"REFUSED: an attempt of {step} started at {oa['ts']} and did not finish: record an infrastructure recovery "
                           "(`recover --reason ... --evidence ...`) before another attempt")
    pa = pending_acceptance(scene, step)
    if pa is not None:
        raise RuntimeError(f"REFUSED: {step} finished at {pa['ts']} but its supervisor acceptance is missing or failed: `accept` (keeps the "
                           "result) or `recover` (moves it aside) with a reason and evidence")
    for p in outputs(scene).get(step, []):
        if os.path.exists(p):
            raise RuntimeError(f"REFUSED: {p} exists although the ledger has no {step} step_done (inspect it; never overwritten)")
    v = verify_freeze(scene, hasher=hasher)
    verify_code(scene, hasher=hasher)
    if step != "poses":
        check_poses(scene, hasher=hasher)
    ledger(scene, event="step_start", step=step, stamp=os.environ.get("K23P_STAMP"), freeze=v)
    return v


def check_poses(scene: str, *, hasher=sha_file) -> None:
    pz = [r for r in ledger_rows(scene) if r.get("event") == "step_done" and r.get("step") == "poses"]
    now = {os.path.basename(p): (hasher(p) if os.path.exists(p) else None) for p in outputs(scene)["poses"]}
    if not pz or now != pz[-1].get("outputs"):
        raise RuntimeError("REFUSED: the LIVE pose files are missing or changed after they were hashed")


def finish(scene: str, step: str, extra=None) -> dict:
    outs = outputs(scene).get(step, []) if step != "live" else outputs(scene)["primary"]
    return ledger(scene, event="step_done", step=step, stamp=os.environ.get("K23P_STAMP"),
                  outputs={os.path.basename(p): sha_file(p) for p in outs}, **(extra or {}))


def recover(scene: str, step: str, reason: str, evidence: str) -> dict:
    """Infrastructure recovery of an unfinished attempt (A28 item 5 / Astra r1 6): reason + independent evidence required; the
    step's partial outputs are MOVED (never deleted) to perception_test/aborted/<ts>_<step>/ with their hashes; an opened holdout
    stays recorded as opened. For `live`, a persisted realistic LIVE result is kept in eval_yolo26s.json (the next attempt runs
    coverage only)."""
    if not reason.strip() or not evidence.strip():
        raise RuntimeError("REFUSED: a recovery needs a reason and independent evidence (A28 item 5: performance alone never justifies it)")
    oa = open_attempt(scene, step) or pending_acceptance(scene, step)
    if oa is None:
        raise RuntimeError(f"REFUSED: {step} has no unfinished or unaccepted attempt to recover")
    if step == "live" and pending_acceptance(scene, step) is not None:
        raise RuntimeError("REFUSED: a finished LIVE result lives inside eval_yolo26s.json: keep it with `accept`, or handle it by hand")
    dest = os.path.join(work(scene), "aborted", time.strftime("%Y%m%dT%H%M%S") + f"_{step}")
    moved = {}
    for p in (outputs(scene).get(step, []) if step != "live" else []):
        if os.path.exists(p):
            os.makedirs(dest, exist_ok=True)
            moved[os.path.basename(p)] = sha_file(p)
            shutil.move(p, os.path.join(dest, os.path.basename(p)))
    opened = any(r.get("event") == "holdout_seal_opened" for r in ledger_rows(scene))
    return ledger(scene, event="infra_recovery", step=step, attempt_started=oa["ts"], reason=reason, evidence=evidence, moved=moved,
                  moved_to=dest if moved else None, holdout_already_opened=opened,
                  note=("the holdout was opened in the recovered attempt: every result from it is reported as a post-opening re-run "
                        "(A28 item 5)") if (step == "holdout" and opened) else None)


# ------------------------------------------------------------------------------------------------ Kitchen 1 modules, pointed at the kitchen
def point_amb(scene: str):
    import train_amb as TA
    TA.SCENE = scene
    TA.AMB = os.path.join(THOR, "kitchen_v3_amb", scene)
    TA.PRIM = os.path.join(THOR, "kitchen_v3", scene, "yolo_primary")
    TA.REPORTS = rep_dir(scene)                               # amb_eval*.json beside the kitchen's frozen record, as Kitchen 1's
    TA.PRIMARY_W = os.path.join(TA.WEIGHTS, f"{scene}.yolo26s.pt")
    return TA


def tk_mod():
    import train_kitchen as tk
    return tk


def require_split_metrics(rep: dict, what: str) -> None:
    """Astra r1 5: VAL / TEST / CLOSE-TEST must each hold numeric mAP50 and mAP50-95; else the report is kept and the step fails."""
    bad = [k for k in ("val", "test", "test_close")
           if not isinstance(rep.get(k), dict) or not all(isinstance(rep[k].get(m), float) for m in ("mAP50", "mAP50_95"))]
    if bad:
        raise StepFailed(f"{what}: incomplete split metrics {bad} ({ {k: rep.get(k, {}).get('error') for k in bad} }); report kept, step not done")


# ------------------------------------------------------------------------------------------------ steps
def step_poses(scene: str) -> dict:
    import build_kitchen_data as bk
    tk = tk_mod()
    agent = bk.make_agent(scene)
    try:
        agent.reset(scene)
        live = tk.frozen_poses(scene, agent)
        cover = tk.coverage_poses(scene, agent)
    finally:
        agent.stop()
    # The registered rule (train_kitchen.frozen_poses, unchanged) stops at 150 poses OR after its 20,000-draw cap; fewer than 150 poses
    # therefore means the cap was reached: the rule's own output in that kitchen (FloorPlan17: dense training cameras), disclosed as a
    # shortfall, never topped up by another rule. Coverage must be complete (Astra r1 7). Ledger record K23-PT-POSES-CAP.
    if len(cover) != tk.COVER_POSES or not 0 < len(live) <= tk.LIVE_POSES:
        raise StepFailed(f"pose sets incomplete: {len(live)} / {tk.LIVE_POSES} live, {len(cover)} / {tk.COVER_POSES} coverage")
    hashes = {os.path.basename(p): sha_file(p) for p in outputs(scene)["poses"]}
    with open(os.path.join(work(scene), "poses.sha256"), "w", encoding="utf-8", newline="\n") as f:
        for p in outputs(scene)["poses"]:
            f.write(f"{hashes[os.path.basename(p)]}  {p}\n")
    out = {"live_poses": len(live), "cover_poses": len(cover)}
    if len(live) < tk.LIVE_POSES:
        out.update(live_rule_limited=True, live_registered=tk.LIVE_POSES,
                   note="the Kitchen 1 rule reached its 20,000-draw cap (0.3 m / 20 deg exclusion of TRAIN cameras): its output is used as it "
                        "is; disclosed shortfall")
    return out


def live_pose_count(scene: str) -> int:
    """The LIVE pose count the poses step recorded (the rule's output). The live step sets train_kitchen.LIVE_POSES to it in-process so a
    rule-limited set is read back as saved instead of being redrawn and rewritten (the redraw would be identical; the hash check stays)."""
    pz = [r for r in ledger_rows(scene) if r.get("event") == "step_done" and r.get("step") == "poses"]
    return int(pz[-1]["result"]["live_poses"])


def step_primary(scene: str) -> dict:
    rep = tk_mod().evaluate(scene, "yolo26s", live=False)
    require_split_metrics(rep, f"{scene} yolo26s")
    return {"test_mAP50": rep["test"]["mAP50"], "test_mAP50_95": rep["test"]["mAP50_95"],
            "close_mAP50": rep["test_close"]["mAP50"], "close_mAP50_95": rep["test_close"]["mAP50_95"]}


def step_live(scene: str) -> dict:
    tk = tk_mod()
    path = outputs(scene)["primary"][0]
    rep = json.load(open(path, encoding="utf-8"))
    if "live_cover" in rep:
        raise RuntimeError("REFUSED: eval_yolo26s.json already holds the coverage LIVE result")
    reused = "live" in rep                                               # persisted by an attempt that died during coverage
    n_live = live_pose_count(scene)
    if n_live < tk.LIVE_POSES:
        tk.LIVE_POSES = n_live                                           # rule-limited set: read back as saved (see live_pose_count)
    if not reused:
        rep["live"] = tk.live_eval(scene, "yolo26s", "realistic")
        check_poses(scene)
        tk._dump(path, rep)                                              # Astra r1 6: persisted before coverage starts
    rep["live_cover"] = tk.live_eval(scene, "yolo26s", "cover")
    check_poses(scene)                                                   # Astra r1 7: neither pose set was regenerated
    tk._dump(path, rep)
    out = {k: {"mAP50": rep[k]["mAP50"], "mAP50_95": rep[k]["mAP50_95"], "poses_used": rep[k]["poses"] - rep[k]["skipped"],
               "gt_boxes": rep[k]["gt_boxes"]} for k in ("live", "live_cover")}
    out["realistic_reused_from_recovered_attempt"] = reused
    out["live_poses_used_from_rule"] = n_live
    return out


def step_fused(scene: str) -> dict:
    TA = point_amb(scene)
    rep = TA.evaluate(False)
    if set(rep.get("sets", {})) != {"amb_test", "primary_test"}:
        raise StepFailed(f"fused evaluation incomplete: sets {sorted(rep.get('sets', {}))}")
    return {k: {"held_target_success": v["held_target_success"], "held_counts": v["held_counts"]} for k, v in rep["sets"].items()}


def step_state(scene: str) -> dict:
    import k23_state_calib as SC
    doc = SC.test(scene)
    return {"state_test_written": True, "calib_sha256": doc["calib_sha256"]}


def holdout_images(TA) -> list:
    meta = TA._frame_meta(os.path.join(TA.AMB, "frames.jsonl"))
    return sorted(n for n, m in meta.items() if m["protocol"] == "amb-v1-holdout")


def step_holdout(scene: str) -> dict:
    TA = point_amb(scene)
    man = json.load(open(os.path.join(TA.AMB, "holdout_manifest.json"), encoding="utf-8"))
    imgs = holdout_images(TA)
    want = sorted(os.path.basename(x).rsplit(".", 1)[0] for x in man["images"])
    if not man.get("sealed") or man.get("protocol") != "amb-v1-holdout" or imgs != want or len(imgs) != man.get("n"):
        raise RuntimeError("REFUSED: the holdout frames differ from the sealed manifest")
    ledger(scene, event="holdout_seal_opened", n_images=len(imgs), manifest_sha256=sha_file(os.path.join(TA.AMB, "holdout_manifest.json")),
           note="an opened holdout is never called unseen again (A28 item 5)")
    rep = TA.evaluate(True)
    s = rep["sets"]["amb_holdout"]
    if s["images"] != len(imgs):
        raise StepFailed(f"holdout scored {s['images']} images, the manifest has {len(imgs)}")
    return {"images": s["images"], "episodes": s["episodes"], "held_counts": s["held_counts"], "held_target_success": s["held_target_success"],
            "held_nontarget_fp_rate": s["held_nontarget_fp_rate"], "boot_gated_minus_primary": s["boot_held_success_gated_minus_primary"],
            "boot_gated_minus_control": s["boot_held_success_gated_minus_control"]}


# ------------------------------------------------------------------------------------------------ A38 item 4: the COCO sensitivity
def val_with_stats(weights: str, data_yaml: str, split: str = "test"):
    """The ultralytics validator exactly as train_kitchen.split_metrics runs it (imgsz 640, batch 16, conf .001, iou .7, max_det 300,
    end2end), with its per-image statistics captured (tp [n x 10 IoUs], conf, pred_cls, target_cls, image name) by wrapping
    DetMetrics.update_stats in this process only. Returns (metrics, [per-image stat dicts])."""
    import tempfile
    from ultralytics.utils import metrics as UM
    tk = tk_mod()
    cap, orig = [], UM.DetMetrics.update_stats

    def upd(self, stat):
        cap.append({k: stat[k] for k in ("tp", "conf", "pred_cls", "target_cls", "im_name")})
        return orig(self, stat)
    UM.DetMetrics.update_stats = upd
    try:
        m = tk._yolo(weights).val(data=data_yaml, split=split, imgsz=tk.RECIPE["imgsz"], batch=16, plots=False, verbose=False,
                                  end2end=True, project=tempfile.mkdtemp(prefix="k23_val_"), **tk.EVAL)
    finally:
        UM.DetMetrics.update_stats = orig
    return m, cap


def episode_resamples(episodes: list, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple:
    """Whole-episode resamples with replacement: episodes[i] is image i's episode; each resample is the list of image indices of the
    drawn episodes (an episode drawn k times contributes all its images k times). -> (resamples, n_episodes)."""
    by = {}
    for i, e in enumerate(episodes):
        by.setdefault(e, []).append(i)
    eps = sorted(by)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append([i for e in (rng.choice(eps) for _ in eps) for i in by[e]])
    return out, len(eps)


def ul_map(stats: list, idx, ap_fn=None) -> tuple:
    """(mAP50, mAP50-95) of the images `idx` with ultralytics' own ap_per_class (the validator's DetMetrics.process rule)."""
    import numpy as np
    if ap_fn is None:
        from ultralytics.utils.metrics import ap_per_class as ap_fn
    cat = {k: np.concatenate([stats[i][k] for i in idx], 0) for k in ("tp", "conf", "pred_cls", "target_cls")}
    ap = ap_fn(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[5]
    return (float(ap[:, 0].mean()), float(ap.mean())) if len(ap) else (0.0, 0.0)


_BOOT = {}


def _boot_chunk(chunk):
    a, b, fn = _BOOT["a"], _BOOT["b"], _BOOT.get("ap_fn")
    out = []
    for idx in chunk:
        ma, mb = ul_map(a, idx, fn), ul_map(b, idx, fn)
        out.append((mb[0] - ma[0], mb[1] - ma[1]))
    return out


def paired_bootstrap(stats_a: list, stats_b: list, episodes: list, n: int = BOOT_N, seed: int = BOOT_SEED, workers=None, ap_fn=None) -> dict:
    """Paired episode bootstrap of TEST mAP50 / mAP50-95 (b - a), A38 item 4: both systems on the SAME resampled episodes; the metric is
    ultralytics' ap_per_class on the validator's per-image statistics. Resamples are drawn once (seed) in this process; workers only
    evaluate them (fork), so the result does not depend on parallelism."""
    from concurrent.futures import ProcessPoolExecutor
    if len(stats_a) != len(stats_b) or len(stats_a) != len(episodes):
        raise ValueError("per-image statistics of the two systems do not align")
    res, n_eps = episode_resamples(episodes, n, seed)
    full = list(range(len(episodes)))
    pa, pb = ul_map(stats_a, full, ap_fn), ul_map(stats_b, full, ap_fn)
    _BOOT.update(a=stats_a, b=stats_b, ap_fn=ap_fn)
    k = workers if workers is not None else max(1, min(8, (os.cpu_count() or 4) - 2))
    if k <= 1:
        diffs = _boot_chunk(res)
    else:
        chunks = [res[i::k] for i in range(k)]
        with ProcessPoolExecutor(max_workers=k) as ex:
            parts = list(ex.map(_boot_chunk, chunks))
        diffs = [None] * len(res)
        for i, part in enumerate(parts):
            for j, d in enumerate(part):
                diffs[i + j * k] = d

    def ci(v):
        v = sorted(v)
        return {"mean": sum(v) / len(v), "ci95": [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]}
    return {"point": {"a": {"mAP50": pa[0], "mAP50_95": pa[1]}, "b": {"mAP50": pb[0], "mAP50_95": pb[1]}},
            "diff_point": {"mAP50": pb[0] - pa[0], "mAP50_95": pb[1] - pa[1]},
            "boot_diff": {"mAP50": ci([d[0] for d in diffs]), "mAP50_95": ci([d[1] for d in diffs])},
            "episodes": n_eps, "images": len(episodes), "resamples": n, "seed": seed,
            "method": "paired whole-episode bootstrap; ultralytics 8.4.83 ap_per_class (101-point interpolated AP, the validator's own "
                      "matching) on the validator's per-image statistics; full-sample values equal the validator's"}


def training_record(scene: str, model: str) -> dict:
    """Epochs run and the best epoch from the run's RETAINED results.csv (~/thor-data/kitchen_v3/<scene>/yolo_primary/runs/<scene>.<model>/;
    the _ckpt copy is deleted after training) under ultralytics 8.4.83's fitness = mAP50-95 (the first epoch reaching the maximum);
    the deployed checkpoint is read too, but a final archive stores epoch -1 / best_fitness None, so checkpoint verification is then
    reported as unavailable (Astra r2 6). Training wall time from the log's START / END lines; unmatched STARTs are counted and the
    total is then marked incomplete (a lower bound, not complete GPU-hours)."""
    import csv
    import re
    out = {"model": model}
    w = os.path.join(SIM, "weights", "v3", f"{scene}.{model}.pt")
    try:
        import torch
        ck = torch.load(w, map_location="cpu", weights_only=False)
        ep, bf = ck.get("epoch"), ck.get("best_fitness")
        out["checkpoint"] = ({"epoch_0based": ep, "best_fitness": bf} if isinstance(ep, int) and ep >= 0 else
                             {"verification": "unavailable", "why": f"the deployed archive stores epoch={ep}, best_fitness={bf}"})
    except Exception as e:                                                # noqa: BLE001
        out["checkpoint"] = {"verification": "unavailable", "why": f"{type(e).__name__}: {e}"[:200]}
    rc = os.path.join(THOR, "kitchen_v3", scene, "yolo_primary", "runs", f"{scene}.{model}", "results.csv")
    if os.path.exists(rc):
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(rc, encoding="utf-8"))]
        fit = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        bi = max(range(len(fit)), key=lambda i: (fit[i], -i))           # the first epoch reaching the best fitness
        out.update(epochs_run=len(rows), best_epoch_1based=int(float(rows[bi]["epoch"])), best_epoch_source="results.csv (mAP50-95 fitness)",
                   best_val_mAP50=float(rows[bi]["metrics/mAP50(B)"]), best_val_mAP50_95=fit[bi], results_csv=rc)
        ckd = out.get("checkpoint") or {}
        if isinstance(ckd.get("epoch_0based"), int):
            out["best_epoch_agrees_with_checkpoint"] = ckd["epoch_0based"] + 1 == out["best_epoch_1based"]
    else:
        out["results_csv"] = {"missing": rc}
    lg = os.path.join(HERE, scene, "train", f"train_{model}.log")
    if os.path.exists(lg):
        txt = open(lg, encoding="utf-8", errors="replace").read()
        ev = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (START|END rc=\d+)", txt)
        secs, start, unmatched = 0.0, None, 0
        for ts, kind in ev:
            t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            if kind == "START":
                unmatched += start is not None
                start = t
            elif start is not None:
                secs += t - start
                start = None
        unmatched += start is not None
        out.update(train_log=lg, sessions=sum(1 for _t, k in ev if k == "START"), matched_sessions_wall_hours=round(secs / 3600, 3),
                   unmatched_starts=unmatched, gpu_hours_complete=unmatched == 0,
                   timing_note=None if unmatched == 0 else "sessions without an END line are not timed: the total is a LOWER BOUND")
    return out


def step_coco(scene: str) -> dict:
    tk = tk_mod()
    TA = point_amb(scene)
    tk.MODELS["yolo26s_coco"] = dict(tk.MODELS["yolo26s"])            # same recipe, the frozen COCO-initialized weights
    rep = tk.evaluate(scene, "yolo26s_coco", live=False)
    require_split_metrics(rep, f"{scene} yolo26s_coco")
    k1 = json.load(open(outputs(scene)["primary"][0], encoding="utf-8"))
    yml = os.path.join(TA.PRIM, "data.yaml")
    ma, sa = val_with_stats(os.path.join(TA.WEIGHTS, f"{scene}.yolo26s.pt"), yml)
    mb, sb = val_with_stats(os.path.join(TA.WEIGHTS, f"{scene}.yolo26s_coco.pt"), yml)
    if [s["im_name"] for s in sa] != [s["im_name"] for s in sb]:
        raise StepFailed("the two validators saw the TEST images in different orders")
    n_test = sum(1 for l in open(os.path.join(TA.PRIM, "test.txt"), encoding="utf-8") if l.strip())
    full = list(range(len(sa)))
    chk = {"k1init": (ul_map(sa, full), (float(ma.box.map50), float(ma.box.map)), (k1["test"]["mAP50"], k1["test"]["mAP50_95"])),
           "coco": (ul_map(sb, full), (float(mb.box.map50), float(mb.box.map)), (rep["test"]["mAP50"], rep["test"]["mAP50_95"]))}
    off = {s: c for s, c in chk.items() if any(abs(x - y) > 1e-6 for t in (c[1], c[2]) for x, y in zip(c[0], t))}
    if len(sa) != n_test or off:
        raise StepFailed(f"full-sample agreement failed (images {len(sa)} / {n_test}; {off})")
    boot = paired_bootstrap(sa, sb, [TA._episode(s["im_name"]) for s in sa])
    boot["direction"] = "coco - k1init"
    per = {c: {"k1init_ap50": k1["test"]["per_class"].get(c, {}).get("ap50"), "coco_ap50": v.get("ap50")}
           for c, v in sorted(rep["test"]["per_class"].items())}
    doc = {"scene": scene, "label": f"{scene} (adapted detector): A38 item 4 COCO-start sensitivity (never selects the deployed model)",
           "test_ultralytics": {"k1init": {"mAP50": k1["test"]["mAP50"], "mAP50_95": k1["test"]["mAP50_95"]},
                                "coco": {"mAP50": rep["test"]["mAP50"], "mAP50_95": rep["test"]["mAP50_95"]}},
           "full_sample_agreement": {s: {"bootstrap_stats": c[0], "validator": c[1], "eval_json": c[2]} for s, c in chk.items()},
           "paired_bootstrap": boot, "per_class_ap50": per,
           "training": {"k1init": training_record(scene, "yolo26s"), "coco": training_record(scene, "yolo26s_coco")},
           "weights_sha256": {"k1init": sha_file(os.path.join(TA.WEIGHTS, f"{scene}.yolo26s.pt")),
                              "coco": sha_file(os.path.join(TA.WEIGHTS, f"{scene}.yolo26s_coco.pt"))}}
    tk._dump(outputs(scene)["coco"][1], doc)
    return {"test_ultralytics": doc["test_ultralytics"], "boot_diff": boot["boot_diff"]}


RUN = {"poses": step_poses, "primary": step_primary, "coco": step_coco, "live": step_live, "fused": step_fused, "state": step_state,
       "holdout": step_holdout}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS + ("bind", "rebind", "verify", "status", "recover", "accept", "supervisor"))
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--step", dest="rstep", choices=STEPS, help="recover / accept / supervisor: the step concerned")
    ap.add_argument("--reason", default="")
    ap.add_argument("--evidence", default="")
    ap.add_argument("--stamp", default="", help="supervisor: the attempt's stamp")
    ap.add_argument("--ok", type=int, choices=(0, 1), default=0, help="supervisor: 1 = step rc 0, valid guard outcome, cleanup done")
    ap.add_argument("--detail", default="", help="supervisor: rc / guard / cleanup detail (JSON)")
    a = ap.parse_args(argv)
    if a.step == "status":
        ss = step_states(a.scene)
        print(json.dumps({"done": sorted(done_steps(a.scene)), "order": ORDER[a.scene],
                          "open": {s: d["open"]["ts"] for s, d in ss.items() if d["open"]},
                          "pending_acceptance": {s: d["pending"]["ts"] for s, d in ss.items() if d["pending"]}}))
        return 0
    if a.step == "supervisor":
        if not a.rstep or not a.stamp:
            print("REFUSED: supervisor needs --step and --stamp")
            return 2
        try:
            det = json.loads(a.detail) if a.detail else {}
        except ValueError:
            det = {"raw": a.detail[:500]}
        print(json.dumps(supervisor_outcome(a.scene, a.rstep, a.stamp, bool(a.ok), detail=det)))
        return 0
    try:
        if a.step == "accept":
            print(json.dumps(accept(a.scene, a.rstep or "", a.reason, a.evidence)))
            return 0
        if a.step == "rebind":
            print(json.dumps(rebind_code(a.scene, a.reason, a.evidence)))
            return 0
        if a.step == "verify":
            print(json.dumps(verify_freeze(a.scene)))
            return 0
        if a.step == "bind":
            print(json.dumps(bind_code(a.scene)))
            return 0
        if a.step == "recover":
            print(json.dumps(recover(a.scene, a.rstep or "", a.reason, a.evidence)))
            return 0
        begin(a.scene, a.step)
    except RuntimeError as e:
        print(e)
        return 2
    t0 = time.time()
    try:
        res = RUN[a.step](a.scene)
    except StepFailed as e:
        ledger(a.scene, event="step_failed", step=a.step, why=str(e)[:500], wall_s=round(time.time() - t0, 1))
        print(f"STEP FAILED: {e}")
        return 3
    rec = finish(a.scene, a.step, {"result": res, "wall_s": round(time.time() - t0, 1)})
    print(json.dumps(rec, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
