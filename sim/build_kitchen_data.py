"""Capture the per-kitchen perception dataset (option b): one full-provenance frame store per kitchen, from which the
primary detector, the close-view specialist and the pooled state estimator all take their training data.

Continues the recipe of Nasr Esfahani et al., Intelligent Service Robotics 19:71 (2026) §2.2 — randomised placement
and materials, many angles, small/far objects filtered, "floating" held views for the disambiguation model — and
applies both Codex/gpt-6-astra reviews of 2026-09-28 (scratchpad ASTRA_OPINION_step3*.md):

* the detectors see TYPES only; every visible instance is boxed whatever its state;
* every (object, axis) stores applicability and simulator value; OBSERVABILITY is not decided here — the record
  keeps the evidence (box/mask size, distance, rim elevation, paired renders) and a later calibration against the
  author's hand labels decides it;
* every state an object is put into is VERIFIED on that object (or its slices), and intended vs realised is kept;
* episodes, splits and seeds are fixed in advance from a stable hash; val/test views are sampled by a fixed
  protocol, never by what training lacks;
* a frame record carries camera intrinsics + extrinsics, depth (mm PNG), the instance-segmentation image and its
  colour->object map, lineage of sliced/cracked pieces, and the episode's action outcomes;
* the audit's verdict is written next to the data, and training refuses a failed audit.

Stages::

    probe     per-INSTANCE: every composite state each object can really reach (verified), eye heights, speed
    capture   the frame store (resumable per episode)
    audit     coverage contract per split -> audit.json (PASS/FAIL, every shortfall named)
    yolo      write the primary detector's YOLO view of the store (fixed type registry; nothing dropped)

    python sim/build_kitchen_data.py plan                                   # Windows OK, runs nothing
    python sim/build_kitchen_data.py run --scenes FloorPlan1 --smoke        # WSL: capture-only smoke audit

Data: ``~/thor-data/kitchen_v2[/smoke]/<scene>/`` (Linux home: fast IO, no apostrophe for ultralytics). Reports
are copied to ``sim/kitchen_v2/<scene>/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import kitchen_data as kd  # noqa: E402
import vocab  # noqa: E402

STAGES = ("probe", "capture", "audit", "yolo")      # + "quarantine" (run on demand)
#: Dataset VERSION. v2 = first full FloorPlan1 run (2026-09-28 16:25-17:51; kept for the record, not for training:
#: its frames contain unlabelled shards of broken vessels). v2b = within-episode state re-rolls, no vessel breaking,
#: 16+16 held-out episodes, opportunity-based suppression. Never resume or mix across versions.
#: v2c = objects identified by their stable NAME (a layout shuffle renames objectIds, so v2b left every shuffled
#: movable object in its default state), anti-grind saturation exit, per-episode hang guard, state-coverage gate.
DATASET_VERSION = "kitchen_v3"
#: TRAIN view-policy version, recorded in every episode record. v2c-1: episodes captured before 2026-09-29 ~01:30
#: (no record field = v2c-1). v2c-2: explicit close-range state-targeting views + state-deficit held views (train
#: only; held-out protocol unchanged = "heldout-v2c-1"). Approved by Arash after an Astra review (step3d).
#: v2c-3: the balancer counts state PER HEAD ('Type|axis=value') with normalised per-object weight, plus an enforced
#: floor per full combination (>= 10 crops from >= 3 episodes); existing records recounted, no recapture (Astra
#: step3e, Arash's go 2026-09-29 ~02:05).
#: v2c-4: render guard (every kept view = two agreeing clean renders; paired renders guarded too), 2026-09-29.
POLICY_VERSION = "v3-2"
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "thor-data", DATASET_VERSION)
REPORT_ROOT = os.path.join(HERE, DATASET_VERSION)
FRAME = 640
FOV = 90.0
EDGE_SLIVER_PX = 3
IRREVERSIBLE = {"sliced", "cooked", "broken"}


# ======================================================================================================================
# Utilities.
# ======================================================================================================================
def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _log(scene, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {scene}: {msg}", flush=True)


def _publish(scene, name, obj):
    _dump(os.path.join(REPORT_ROOT, scene, name), obj)


def _versions() -> dict:
    out = {}
    for mod in ("ai2thor", "ultralytics", "torch", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import subprocess
        out["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                                         timeout=10).stdout.strip()
    except Exception:
        out["git_head"] = None
    return out


def instance_key(obj: dict, objs_by_id: dict = None) -> str:
    """The STABLE identity of an object across layout shuffles: its AI2-THOR ``name``.

    ``objectId`` encodes the object's position, so ``InitialRandomSpawn`` RENAMES every object it moves (verified
    2026-09-29 in FloorPlan1: 29-30 of 77 ids change per shuffle, while all 77 ``name`` values stay identical and
    unique). Keying anything on objectId across a shuffle silently loses those objects — which is how the v2b
    capture left every shuffled apple, egg and mug in its default state. A slice/crack piece counts toward its
    parent's identity when the parent is still in the scene."""
    t = obj.get("objectType") or ""
    parent = parent_of(obj.get("objectId") or "", t)
    if parent and objs_by_id and parent in objs_by_id:
        return objs_by_id[parent].get("name") or parent
    return obj.get("name") or obj.get("objectId")


def parent_of(oid: str, otype: str):
    """Lineage of a derived piece: ``'Apple|..|AppleSliced_3'`` -> ``'Apple|..'``; None for an original object."""
    if otype in vocab.DERIVED_TYPES and "|" in oid:
        return oid.rsplit("|", 1)[0]
    return None


# ======================================================================================================================
# Agent + state setting.
# ======================================================================================================================
def make_agent(scene):
    import thor_actions as ta
    return ta.ThorAgent(scene=scene, width=FRAME, height=FRAME, field_of_view=FOV,
                        render_instance_seg=True, render_depth=True)


def _pieces(agent, oid):
    return [o for o in agent.objects() if o["objectId"].startswith(oid + "|")
            and o.get("objectType") in vocab.DERIVED_TYPES]


def set_state(agent, oid: str, target: dict) -> dict:
    """Put object ``oid`` into composite ``target``; follow slices/cracks for later steps; VERIFY on the object and
    its pieces. Returns ``{'target', 'steps', 'errors', 'realised', 'pieces'}`` — never raises."""
    obj = agent.object_by_id(oid) or {}
    steps = vocab.actions_for(target, vocab.state_record(obj))
    errors, ids = [], [oid]
    for action, kw in steps:
        nxt = []
        for tid in ids:
            ev = agent.do(action, objectId=tid, forceAction=True, **kw)
            if not ev.metadata.get("lastActionSuccess"):
                errors.append([action, tid, (ev.metadata.get("errorMessage") or "")[:160]])
            if action in ("SliceObject", "BreakObject"):
                kids = [o["objectId"] for o in _pieces(agent, tid)]
                nxt.extend(kids or [tid])
            else:
                nxt.append(tid)
        ids = nxt
    # verify: the original if it still exists as itself, else every piece it became. A missing object is a
    # FAILURE (Astra, 2026-09-28: the old filtered all(...) reported success for an object that no longer existed).
    pieces = _pieces(agent, oid)
    orig = agent.object_by_id(oid)
    check = pieces if pieces else ([orig] if orig else [])
    ok = bool(check) and all(vocab.realised(target, vocab.state_record(o)) for o in check)
    return {"target": target, "steps": [s[0] for s in steps], "errors": errors, "realised": ok,
            "pieces": [o["objectId"] for o in pieces]}


# ======================================================================================================================
# The labeller (instance masks, not the range-gated `visible` flag).
# ======================================================================================================================
def label_frame(agent, cfg=kd.CAPTURE) -> list:
    """``[{oid, type, orig_type, parent, box, px, dist, state, state_key, rim_elev, held}]`` for every object
    with enough rendered pixels. No state-based rejection: every visible instance gets its type label."""
    import numpy as np
    import perception
    frame = agent.frame
    if frame is None:
        return []
    h, w = frame.shape[0], frame.shape[1]
    masks = agent.instance_masks or {}
    objs = {o["objectId"]: o for o in agent.objects()}
    cam = agent.event.metadata.get("cameraPosition")
    out = []
    for oid, mask in masks.items():
        o = objs.get(oid)
        if o is None:
            continue
        m = np.asarray(mask, dtype=bool)
        px = int(m.sum())
        if px < cfg["min_mask_px"]:
            continue
        box = perception.box_from_mask(m)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1 + 1) * (y2 - y1 + 1) < cfg["min_box_area"]:
            continue
        if ((x1 <= 0 or x2 >= w - 1) and (x2 - x1 + 1) <= EDGE_SLIVER_PX) or \
           ((y1 <= 0 or y2 >= h - 1) and (y2 - y1 + 1) <= EDGE_SLIVER_PX):
            continue
        st = vocab.state_record(o)
        parent = parent_of(oid, o["objectType"])
        out.append({"oid": oid, "name": o.get("name"), "inst": instance_key(o, objs),
                    "type": vocab.base_type(o["objectType"]), "orig_type": o["objectType"],
                    "parent": parent, "box": [x1, y1, x2, y2], "px": px,
                    "dist": round(float(o.get("distance") or 0.0), 3), "state": st,
                    "state_key": vocab.state_key(o["objectType"], st),
                    "rim_elev": (round(vocab.rim_elevation_deg(cam, o), 2)
                                 if vocab.rim_elevation_deg(cam, o) is not None else None),
                    "held": bool(o.get("isPickedUp")), "state_min_px": cfg["state_min_px"]})
    return out


def frame_anomaly(frame) -> str | None:
    """Content check for the corruption signatures found 2026-09-29: an all-white frame, or a whole-image colour
    cast (one channel's mean far above the other two). Returns the reason, or None."""
    import numpy as np
    a = np.asarray(frame, dtype=np.float32)
    if float(((a > 245).all(-1)).mean()) >= 0.99:
        return "pure_white"
    m = a.reshape(-1, 3).mean(0)
    if min(m[0] - m[1], m[0] - m[2]) > 60:
        return "red_cast"
    if min(m[2] - m[0], m[2] - m[1]) > 60:
        return "blue_cast"
    return None


#: Why the most recent render_guard call failed: {"stable": renders agreed but the content check fired,
#: "anomaly": the signature}. Read by the capture loop to tell legitimate content from a glitch (see RenderGlitch).
GUARD_LAST = {"stable": False, "anomaly": None}


def render_guard(agent, cfg=kd.CAPTURE, allow_anomaly: bool = False):
    """Re-render the current pose until two CONSECUTIVE fresh renders agree and the frame shows no corruption
    signature; the agent's event then holds the accepted RGB, masks and depth. Returns ``(ok, renders)``.

    Why: on this machine (Unity over WSLg / Mesa D3D12) ~0.85 % of renders come back with a corrupted colour
    channel — a same-pose re-render moments later is clean (16 of 18 paired cases, 2026-09-29) — and a few come back
    pure white. Agreement proves stability, not correctness (Astra), so the content check runs too, and the
    comparison is also made per tile, so a local corruption cannot hide in a whole-image average. Physics settling
    can legitimately change a frame; that just costs another render until it is at rest."""
    import numpy as np
    n_max = int(cfg.get("guard_max_renders", 8))
    tile = int(cfg.get("guard_tile", 80))
    # after ANY disagreement, require this many consecutive agreeing renders (i.e. 3 identical frames) — closes the
    # "two identical corrupted renders" gap (Astra step3f); clean views still cost one extra render
    confirm = int(cfg.get("guard_confirm_after_retry", 2))
    prev = np.array(agent.frame, dtype=np.float32, copy=True)
    disagreed, streak, anomaly = False, 0, None
    for n in range(1, n_max + 1):
        agent.do("Pass")
        cur = np.array(agent.frame, dtype=np.float32, copy=True)
        d = np.abs(cur - prev)
        h, w = d.shape[0] // tile * tile, d.shape[1] // tile * tile
        tiles = d[:h, :w].reshape(h // tile, tile, w // tile, tile, 3).mean(axis=(1, 3, 4))
        agree = float(d.mean()) < cfg.get("guard_mae", 1.0) and float(tiles.max()) < cfg.get("guard_tile_mae", 6.0)
        anomaly = None if allow_anomaly else frame_anomaly(cur)
        if agree and anomaly is None:
            streak += 1
            if streak >= (confirm if disagreed else 1):
                return True, n
        else:
            if not agree:
                disagreed = True
            streak = 0
        prev = cur
    GUARD_LAST.update(stable=not disagreed, anomaly=anomaly)
    return False, n_max


def camera_record(agent) -> dict:
    md = agent.event.metadata
    pose = agent.agent_pose()
    return {"position": md.get("cameraPosition"), "agent": pose["position"], "yaw": pose["rotation"]["y"],
            "horizon": pose["horizon"], "standing": pose.get("standing", True),
            "fov": FOV, "w": FRAME, "h": FRAME}


def save_frame(agent, ds: str, split: str, name: str, sub: str = "") -> None:
    """RGB (JPEG q95), depth (uint16 mm PNG), instance segmentation (lossless PNG)."""
    import numpy as np
    from PIL import Image
    base = os.path.join(ds, sub) if sub else ds
    for kind in ("images", "depth", "seg"):
        os.makedirs(os.path.join(base, kind, split), exist_ok=True)
    Image.fromarray(agent.frame).save(os.path.join(base, "images", split, name + ".jpg"), quality=95)
    d = agent.depth_frame
    if d is not None:
        Image.fromarray(np.clip(np.asarray(d) * 1000.0, 0, 65535).astype(np.uint16)).save(
            os.path.join(base, "depth", split, name + ".png"))
    s = agent.instance_segmentation_frame
    if s is not None:
        Image.fromarray(np.asarray(s)).save(os.path.join(base, "seg", split, name + ".png"))


def seg_colors(agent, labels) -> dict:
    """colour -> objectId for the labelled objects, so masks can be recovered from the stored seg PNG."""
    inv = {}
    for col, oid in (agent.color_to_object_id or {}).items():
        inv.setdefault(oid, list(col) if not isinstance(col, str) else col)
    return {lab["oid"]: inv.get(lab["oid"]) for lab in labels}


# ======================================================================================================================
# Stage: probe.
# ======================================================================================================================
def stage_probe(scene: str, root: str) -> dict:
    """Which composite states can EACH object instance reach? Verified on the instance itself (or its pieces).

    Reversible-only targets are tried in place and undone by the next target; any target touching an irreversible
    axis is tried from a fresh reset. Also measures eye heights (one-teleport aiming) and render speed."""
    agent = make_agent(scene)
    try:
        agent.reset(scene)
        objs = agent.objects()
        registry = sorted({vocab.base_type(o["objectType"]) for o in objs})
        # identities are NAMES (stable across layout shuffles); the reset-time id is kept only for reference
        instances = sorted(instance_key(o) for o in objs)
        reset_ids = {instance_key(o): o["objectId"] for o in objs}
        pos = agent.reachable_positions()
        p = pos[len(pos) // 2]
        agent.teleport_full(p["x"], p["y"], p["z"], 0.0, horizon=0.0, standing=True)
        eye_stand = agent.event.metadata["cameraPosition"]["y"] - agent.agent_pose()["position"]["y"]
        agent.teleport_full(p["x"], p["y"], p["z"], 0.0, horizon=0.0, standing=False)
        eye_crouch = agent.event.metadata["cameraPosition"]["y"] - agent.agent_pose()["position"]["y"]
        rng = random.Random(kd.stable_seed(scene, "probe"))
        t0 = time.time()
        for _ in range(30):
            q = rng.choice(pos)
            agent.teleport_full(q["x"], q["y"], q["z"], round(rng.uniform(0, 360), 1), horizon=0.0, standing=True)
        s_view = (time.time() - t0) / 30.0

        reach, fail, t_reset = {}, {}, []
        for o in objs:
            if o["objectType"] in vocab.DERIVED_TYPES or not vocab.axes_of(o["objectType"]):
                continue
            oid = o["objectId"]
            name = instance_key(o)
            targets = vocab.composite_targets(o["objectType"], capture=True)
            agent.reset(scene)
            for tg in sorted(targets, key=lambda t: any(t.get(a) for a in IRREVERSIBLE)):
                if any(tg.get(a) for a in IRREVERSIBLE):
                    t1 = time.time()
                    agent.reset(scene)
                    t_reset.append(time.time() - t1)
                r = set_state(agent, oid, tg)
                key = vocab.target_key(o["objectType"], tg)
                if r["realised"]:
                    reach.setdefault(name, []).append({"target": tg, "key": key})
                else:
                    fail.setdefault(name, []).append({"target": tg, "key": key, "errors": r["errors"][:3]})
            _log(scene, f"probe {oid[:48]:48s} {len(reach.get(name, []))}/{len(targets)} states reachable")
        keys = sorted({x["key"] for v in reach.values() for x in v})
        # stateless types still get one state key each, so every type has state support targets = its identity
        result = {"scene": scene, "registry": registry, "instances": instances, "reset_ids": reset_ids,
                  "identity": "name", "n_reachable": len(pos),
                  "eye_stand": eye_stand, "eye_crouch": eye_crouch, "seconds_per_view": s_view,
                  "seconds_per_reset": (sum(t_reset) / len(t_reset)) if t_reset else None,
                  "reachable": reach, "unreachable": fail, "state_keys": keys, "versions": _versions()}
    finally:
        agent.stop()
    _dump(os.path.join(root, scene, "probe.json"), result)
    _publish(scene, "probe.json", result)
    return result


# ======================================================================================================================
# Stage: capture.
# ======================================================================================================================
def setup_episode(agent, scene, ep, probe, quota, cfg=kd.CAPTURE) -> dict:
    """Reset, randomise (seeded), and put every stateful instance into a verified state."""
    rng = random.Random(ep["seed"])
    agent.reset(scene)
    rec = {"episode": ep["episode"], "split": ep["split"], "seed": ep["seed"], "spawn": None,
           "policy": POLICY_VERSION if ep["split"] == "train" else (ep.get("protocol") or "heldout-v2c-1"),
           "protocol": ep.get("protocol") or "main",
           "lighting": None, "materials": None, "objects": {}, "rounds": []}
    p_spawn = cfg["p_spawn"] if ep["split"] == "train" else cfg.get("p_spawn_heldout", cfg["p_spawn"])
    if rng.random() < p_spawn:
        s = rng.randrange(1 << 30)
        ev = agent.do("InitialRandomSpawn", randomSeed=s, forceVisible=False, numPlacementAttempts=5,
                      placeStationary=True)
        rec["spawn"] = {"seed": s, "ok": bool(ev.metadata.get("lastActionSuccess"))}
    if rng.random() < cfg["p_lighting"]:
        ev = agent.do("RandomizeLighting", brightness=tuple(cfg.get("lighting_brightness", (0.7, 1.3))),
                      randomizeColor=False, synchronized=False)
        rec["lighting"] = {"ok": bool(ev.metadata.get("lastActionSuccess")),
                           "err": ev.metadata.get("errorMessage", "")[:120]}
    if rng.random() < cfg["p_materials"]:
        ev = agent.do("RandomizeMaterials")
        rec["materials"] = {"ok": bool(ev.metadata.get("lastActionSuccess"))}
    rec["objects"] = assign_states(agent, ep, probe, quota, rng, reversible_only=False)
    rec["state_coverage"] = state_coverage(rec["objects"], probe, agent)
    rec["settled_metadata"] = _settle(agent)
    return rec


def _settle(agent) -> dict:
    for _ in range(3):
        agent.do("Pass")                                # let physics settle (spills, dropped pieces)
    return {o["objectId"]: vocab.state_key(o["objectType"], vocab.state_record(o))
            for o in agent.objects() if vocab.axes_of(o["objectType"])}


def assign_states(agent, ep, probe, quota, rng, reversible_only: bool) -> dict:
    """Put every stateful instance into a verified composite state.

    ``reversible_only`` (a within-episode re-roll): only targets COMPATIBLE with the object's current
    irreversible values (:func:`vocab.compatible` — a sliced apple stays sliced), so only open/on/dirty/fill
    change; objects that no longer exist as themselves (sliced into pieces) are skipped. Train picks by deficit,
    held-out uniformly (independent of training deficits)."""
    out = {}
    # resolve every probed identity (a NAME) to the object's CURRENT id — a layout shuffle renames what it moves
    by_name = {o.get("name"): o for o in agent.objects() if o.get("objectType") not in vocab.DERIVED_TYPES}
    for name in sorted(probe["reachable"]):
        obj = by_name.get(name)
        if obj is None or vocab.base_type(obj["objectType"]) in vocab.CONTROLLED_TYPES:
            continue
        oid = obj["objectId"]
        base = vocab.base_type(obj["objectType"])
        cands = [(x["target"], x["key"],
                  vocab.axis_keys(base, {ax: {"value": x["target"].get(ax)} for ax in vocab.axes_of(base)}))
                 for x in probe["reachable"][name]]
        if reversible_only:
            if _pieces(agent, oid):
                continue
            cur = vocab.state_record(obj)
            cands = [c for c in cands if vocab.compatible(c[0], cur)]
        if not cands:
            continue
        tg, key, _heads = quota.pick_target(cands, rng) if ep["split"] == "train" else rng.choice(cands)
        r = set_state(agent, oid, tg)
        r["key"] = key
        r["oid"] = oid
        out[name] = r
    return out


def state_coverage(assigned: dict, probe: dict, agent) -> dict:
    """How many of the kitchen's settable objects actually received a state this episode — the check that would
    have caught the v2b renaming bug on the first episode (28 of 48 were set; every movable one was missed)."""
    present = {o.get("name"): o for o in agent.objects() if o.get("objectType") not in vocab.DERIVED_TYPES}
    settable = [n for n in probe["reachable"] if n in present
                and vocab.base_type(present[n]["objectType"]) not in vocab.CONTROLLED_TYPES]
    missed = sorted(n for n in settable if n not in assigned)
    return {"settable": len(settable), "set": len(settable) - len(missed), "missed": missed[:12]}


def choose_view(agent, positions, probe, quota, split, rng, cfg=kd.CAPTURE, protocol=None):
    """Pick a pose and teleport. Returns ``(pose, kind)`` or ``(None, kind)`` if refused."""
    objs = [o for o in agent.objects() if o.get("position")]
    if protocol == kd.CLOSE_PROTOCOL:
        # pre-declared held-out close protocol: a STATEFUL object chosen UNIFORMLY, 0.45-1.0 m, standing
        stateful = [o for o in objs if vocab.axes_of(o["objectType"])
                    and vocab.base_type(o["objectType"]) not in vocab.CONTROLLED_TYPES]
        if stateful:
            return _state_view(agent, positions, probe, quota, rng, stateful, cfg, uniform=True), "close_protocol"
    r = rng.random()
    kind = "random" if r < cfg["p_random_view"] else ("close" if r < cfg["p_random_view"] + cfg["p_close"] else "aimed")
    if split == "train" and objs and rng.random() < cfg.get("p_state_view", 0.0):
        stateful = [o for o in objs if vocab.axes_of(o["objectType"])
                    and vocab.base_type(o["objectType"]) not in vocab.CONTROLLED_TYPES]
        if stateful:
            return _state_view(agent, positions, probe, quota, rng, stateful, cfg), "state"
    if kind == "random" or not objs:
        p = rng.choice(positions)
        pose = {"x": p["x"], "y": p["y"], "z": p["z"], "rot": round(rng.uniform(0, 360), 1),
                "hor": round(rng.uniform(-10, 40), 1), "standing": rng.random() >= cfg["p_crouch"], "target": None}
    else:
        if split == "train":
            by_id = {o["objectId"]: o for o in objs}
            w = [quota.object_weight(vocab.base_type(o["objectType"]), instance_key(o, by_id),
                                     *_state_keys(o)) for o in objs]
        else:
            w = [1.0] * len(objs)
        o = rng.choices(objs, weights=w, k=1)[0]
        c = (o.get("axisAlignedBoundingBox") or {}).get("center") or o["position"]
        lo, hi = cfg["close_band"] if kind == "close" else cfg["dist_band"]
        d = rng.uniform(lo, hi)
        ranked = sorted(positions, key=lambda q: abs(math.hypot(q["x"] - c["x"], q["z"] - c["z"]) - d))
        p = rng.choice(ranked[:6])
        standing = rng.random() >= cfg["p_crouch"]
        eye = p["y"] + (probe["eye_stand"] if standing else probe["eye_crouch"])
        dx, dy, dz = c["x"] - p["x"], c["y"] - eye, c["z"] - p["z"]
        yaw = math.degrees(math.atan2(dx, dz)) + rng.gauss(0.0, (cfg["yaw_jitter"] / (3 if kind == "close" else 2)))
        hor = -math.degrees(math.atan2(dy, max(math.hypot(dx, dz), 1e-6))) + rng.uniform(-cfg["horizon_jitter"],
                                                                                           cfg["horizon_jitter"])
        pose = {"x": p["x"], "y": p["y"], "z": p["z"], "rot": round(yaw % 360.0, 1),
                "hor": round(max(-30.0, min(60.0, hor)), 1), "standing": standing, "target": o["objectId"]}
    agent.teleport_full(pose["x"], pose["y"], pose["z"], pose["rot"], horizon=pose["hor"], standing=pose["standing"])
    return (pose if agent.success else None), kind


def _state_keys(o) -> tuple:
    """(per-head keys, combination key) of an object's CURRENT state — what the balancer weighs."""
    rec = vocab.state_record(o)
    return vocab.axis_keys(o["objectType"], rec), vocab.state_key(o["objectType"], rec)


def _state_view(agent, positions, probe, quota, rng, stateful, cfg, uniform=False):
    """TRAIN only: aim at the object whose CURRENT state is most short of crops, from close range, standing.

    Standing matters for vessels: the eye is ~0.67 m above the agent's base, so from 0.45-1.0 m a counter-height mug
    is seen from roughly 30-50 degrees above its rim — the liquid is in view, not hidden behind the wall of the cup."""
    if uniform:
        o = rng.choice(stateful)
    else:
        w = [quota.state_weight(*_state_keys(o)) for o in stateful]
        o = rng.choices(stateful, weights=w, k=1)[0]
    c = (o.get("axisAlignedBoundingBox") or {}).get("center") or o["position"]
    d = rng.uniform(*cfg["state_band"])
    ranked = sorted(positions, key=lambda q: abs(math.hypot(q["x"] - c["x"], q["z"] - c["z"]) - d))
    p = rng.choice(ranked[:4])
    eye = p["y"] + probe["eye_stand"]
    dx, dy, dz = c["x"] - p["x"], c["y"] - eye, c["z"] - p["z"]
    yaw = math.degrees(math.atan2(dx, dz)) + rng.gauss(0.0, cfg["yaw_jitter"] / 4)
    hor = -math.degrees(math.atan2(dy, max(math.hypot(dx, dz), 1e-6))) + rng.uniform(-6, 6)
    pose = {"x": p["x"], "y": p["y"], "z": p["z"], "rot": round(yaw % 360.0, 1),
            "hor": round(max(-30.0, min(60.0, hor)), 1), "standing": True, "target": o["objectId"],
            "state_target": vocab.state_key(o["objectType"], vocab.state_record(o))}
    agent.teleport_full(pose["x"], pose["y"], pose["z"], pose["rot"], horizon=pose["hor"], standing=True)
    return pose if agent.success else None


def view_cell(agent) -> list:
    """Where the camera stands and looks: (0.5 m x-cell, 0.5 m z-cell, 45-degree yaw bin) — the unit of
    "a different background" (v3: distinct viewpoints per type, distinct held backgrounds per object)."""
    pose = agent.agent_pose()
    p = pose["position"]
    return [int(math.floor(p["x"] / 0.5)), int(math.floor(p["z"] / 0.5)),
            int(((pose["rotation"]["y"] % 360.0) + 22.5) // 45) % 8]


def held_views(agent, positions, probe, rng, cfg=kd.CAPTURE, quota=None, split="val", picked=None, types=None,
               face_map=None):
    """GENERATOR (policy v3-1): pick ONE object up and show it, in hand, against several DIFFERENT backgrounds.

    Arash 2026-09-30: v2c picked "the same object from the same place at the same angle several times"; measured,
    Potato was held 514x and SoapBottle 9x, one object was re-picked up to 9x per episode, and the agent often faced
    a bare wall. Now:
      * object: TRAIN by per-INSTANCE held deficit (frames and distinct background cells), val/test uniform; an
        instance is picked up at most ONCE per episode (``picked``);
      * each view: the agent (still holding it — probed: a teleport keeps the grip) moves to a reachable spot FAR
        from every spot this object was already held at, then turns to a yaw with >= ``held_min_context`` other
        labelled objects in view (no bare walls);
      * the object is rotated and moved within the frame (MoveHeldObject ahead -0.1..0.25, right +-0.25, up +-0.1 m
        — probed: accepted, and the box moves across the image), so it is not always the same size in the same place.
    Always drops the object after."""
    picked = picked if picked is not None else set()
    objs = agent.objects()
    by_id = {o["objectId"]: o for o in objs}
    cands = [o for o in objs if o.get("pickupable") and o.get("position")
             and instance_key(o, by_id) not in picked and (types is None or o["objectType"] in types)]
    if not cands:
        return
    if split == "train" and quota is not None:
        # SQUARED deficit (v3-2, 2026-09-30): after 24 train episodes the plain deficit left SaltShaker at 2 held
        # frames and Bread at 31 — the weights differed by only ~2x while most objects were still far from target
        w = [max(0.02, quota.held_deficit(instance_key(o, by_id))) ** 2 for o in cands]
        o = rng.choices(cands, weights=w, k=1)[0]
    else:
        o = rng.choice(cands)
    inst = instance_key(o, by_id)
    picked.add(inst)
    oid, c = o["objectId"], o["position"]
    p = rng.choice(sorted(positions, key=lambda q: math.hypot(q["x"] - c["x"], q["z"] - c["z"]))[:4])
    face = math.degrees(math.atan2(c["x"] - p["x"], c["z"] - p["z"])) % 360.0
    agent.teleport_full(p["x"], p["y"], p["z"], round(face, 1), horizon=30.0, standing=True)
    ev = agent.do("PickupObject", objectId=oid, forceAction=True)
    if not ev.metadata.get("lastActionSuccess"):
        # an object inside a CLOSED cabinet / drawer / fridge cannot be picked up: open what holds it, retry once
        opened = False
        for rid in (o.get("parentReceptacles") or []):
            r = by_id.get(rid)
            if r and r.get("openable") and not r.get("isOpen"):
                opened |= bool(agent.do("OpenObject", objectId=rid, openness=1.0,
                                        forceAction=True).metadata.get("lastActionSuccess"))
        if opened:
            ev = agent.do("PickupObject", objectId=oid, forceAction=True)
        if not ev.metadata.get("lastActionSuccess"):
            return
    need = int(cfg.get("held_min_context", 5))
    used = [tuple(u) for u in ((quota.held_pos.get(inst) if quota is not None else None) or [])]
    try:
        partners = [x for x in agent.objects() if x.get("position") and face_map
                    and x["objectType"] in face_map.get(o["objectType"], ())]
        for _ in range(cfg["held_views"]):
            face_pt = None
            if partners and rng.random() < float(cfg.get("held_face_partner", 0.0)):
                # amb-v1: hold the target WITH its look-alike in view (salt shaker in hand, pepper shaker behind)
                fo = agent.object_by_id(rng.choice(partners)["objectId"])
                if fo:
                    face_pt = (fo.get("axisAlignedBoundingBox") or {}).get("center") or fo["position"]
                    dd = rng.uniform(0.7, 1.6)
                    ranked = sorted(positions, key=lambda q_: abs(math.hypot(q_["x"] - face_pt["x"],
                                                                             q_["z"] - face_pt["z"]) - dd))
                    q = rng.choice(ranked[:5])
            if face_pt is None:
                # 1. a standing spot far from where this object was held before (best few of a random sample)
                sample = rng.sample(positions, min(len(positions), 24))
                if used:
                    sample.sort(key=lambda q: -min(math.hypot(q["x"] - u[0], q["z"] - u[1]) for u in used))
                    q = rng.choice(sample[:3])
                else:
                    q = rng.choice(sample)
            # 2. a yaw that faces content: keep the try with the most OTHER labelled objects; stop at a good one
            best = None
            for _t in range(int(cfg.get("held_yaw_tries", 6))):
                if face_pt is not None:
                    dx, dz = face_pt["x"] - q["x"], face_pt["z"] - q["z"]
                    dy = face_pt["y"] - (q["y"] + probe["eye_stand"])
                    pose = (round((math.degrees(math.atan2(dx, dz)) + rng.gauss(0, 12)) % 360, 1),
                            round(max(-10.0, min(50.0, -math.degrees(math.atan2(dy, max(math.hypot(dx, dz), 1e-6)))
                                                 + rng.uniform(-5, 5))), 1), True)
                else:
                    hz = cfg.get("held_horizon", (0, 30))
                    pose = (round(rng.uniform(0, 360), 1), round(rng.uniform(*hz), 1), rng.random() >= 0.15)
                agent.teleport_full(q["x"], q["y"], q["z"], pose[0], horizon=pose[1], standing=pose[2])
                h = agent.object_by_id(oid)
                if not agent.success or not h or not h.get("isPickedUp"):
                    continue
                n_ctx = sum(1 for l in label_frame(agent, cfg) if l["oid"] != oid)
                if best is None or n_ctx > best[0]:
                    best = (n_ctx, pose)
                if n_ctx >= need:
                    break
            if best is None or best[0] < max(1, need - 2):
                continue                                   # only bare views from this spot: try another spot
            if best[1] != pose:
                agent.teleport_full(q["x"], q["y"], q["z"], best[1][0], horizon=best[1][1], standing=best[1][2])
            h = agent.object_by_id(oid)
            if not agent.success or not h or not h.get("isPickedUp"):
                continue
            # 3. rotate it (halve on a refused collision), then move it across the frame — AFTER the last agent turn,
            #    which would otherwise reset the grip
            p_, y_, r_ = rng.uniform(-90, 90), rng.uniform(-180, 180), rng.uniform(-45, 45)
            rotated = False
            for scale in (1.0, 0.5, 0.25):
                ev = agent.do("RotateHeldObject", pitch=round(p_ * scale, 1), yaw=round(y_ * scale, 1),
                              roll=round(r_ * scale, 1))
                if ev.metadata.get("lastActionSuccess"):
                    rotated = scale
                    break
            hm = cfg.get("held_move") or {"ahead": (-0.1, 0.25), "right": (-0.25, 0.25), "up": (-0.1, 0.1)}
            mv = (rng.uniform(*hm["ahead"]), rng.uniform(*hm["right"]), rng.uniform(*hm["up"]))
            moved = False
            for scale in (1.0, 0.5):
                ev = agent.do("MoveHeldObject", ahead=round(mv[0] * scale, 2), right=round(mv[1] * scale, 2),
                              up=round(mv[2] * scale, 2), forceVisible=False)
                if ev.metadata.get("lastActionSuccess"):
                    moved = scale
                    break
            used.append((round(q["x"], 2), round(q["z"], 2)))
            yield {"held": oid, "inst": inst, "x": q["x"], "y": q["y"], "z": q["z"], "cell": view_cell(agent),
                   "context": best[0], "rotated": rotated, "moved": moved}
    finally:
        try:
            agent.do("DropHandObject", forceAction=True)
            for _ in range(5):
                agent.do("Pass")                          # let the dropped object come to rest before the next view
        except Exception:
            pass                                          # the controller is already gone (caller stopped it)


def paired_render(agent, ds, split, name, labels, rng, cfg=kd.CAPTURE, allow_anomaly=False):
    """Observability EVIDENCE for one reversible axis of one object in the current view.

    Renders: ``noise`` (same state, re-rendered: the pixel noise floor), ``flip`` (axis flipped), then flips back
    and verifies the object AND every other object's state returned. Frames go to ``pairs/`` — never into
    detector training. Returns the pair record, or None when nothing qualifies."""
    cands = []
    for lab in labels:
        if lab["px"] < cfg["state_min_px"] or lab["held"]:
            continue
        for ax in vocab.REVERSIBLE:
            if lab["state"].get(ax, {}).get("applicable"):
                cands.append((lab, ax))
    if not cands:
        return None
    lab, ax = rng.choice(cands)
    oid, v = lab["oid"], bool(lab["state"][ax]["value"])
    before = {o["objectId"]: vocab.state_key(o["objectType"], vocab.state_record(o)) for o in agent.objects()}
    liquid = lab["state"].get("liquid", {}).get("value") or rng.choice(vocab.LIQUIDS)
    on_a, off_a = vocab.REVERSIBLE[ax]
    fwd, back = (off_a, on_a) if v else (on_a, off_a)
    kw_fwd = {"fillLiquid": liquid} if fwd == "FillObjectWithLiquid" else ({"openness": 1.0} if fwd == "OpenObject" else {})
    kw_back = {"fillLiquid": liquid} if back == "FillObjectWithLiquid" else ({"openness": 1.0} if back == "OpenObject" else {})
    g1, n1 = render_guard(agent, cfg, allow_anomaly=allow_anomaly)   # the "noise" render must be clean
    if (n1 > 1 or not g1) and cfg.get("restart_on_glitch"):
        raise RenderGlitch(f"paired render of {name}: guard needed {n1} renders")
    if not g1:
        return None
    save_frame(agent, ds, split, name + "_noise", "pairs")
    ev = agent.do(fwd, objectId=oid, forceAction=True, **kw_fwd)
    rec = {"base": name, "oid": oid, "axis": ax, "from": v, "action": fwd, "ok": bool(ev.metadata.get("lastActionSuccess")),
           "err": (ev.metadata.get("errorMessage") or "")[:120]}
    if rec["ok"]:
        g2, n2 = render_guard(agent, cfg, allow_anomaly=allow_anomaly)
        # a state flip (door, tap, burner) can set things moving; a flip render that disagrees then settles, AND
        # does so again at the same frame in a fresh process, is motion — accepted (see the episode loop)
        motion_ok = g2 and ("flip:" + name) in cfg.get("_allow_motion_pairs", ())
        if (n2 > 1 or not g2) and cfg.get("restart_on_glitch") and not motion_ok:
            raise RenderGlitch(f"paired flip render of {name}: guard needed {n2} renders", view="flip:" + name,
                               converged=bool(g2))
        rec["flip_guarded"] = g2
        if not g2:
            rec["ok"] = False
            rec["err"] = "render guard rejected the flip render"
    if rec["ok"]:
        save_frame(agent, ds, split, name + "_flip", "pairs")
        rec["flip_labels"] = [{k: x[k] for k in ("oid", "box", "px", "state_key")} for x in label_frame(agent, cfg)]
    ev = agent.do(back, objectId=oid, forceAction=True, **kw_back)
    rec["back_ok"] = bool(ev.metadata.get("lastActionSuccess"))
    rec["back_err"] = (ev.metadata.get("errorMessage") or "")[:120]
    after = {o["objectId"]: vocab.state_key(o["objectType"], vocab.state_record(o)) for o in agent.objects()}
    rec["restored"] = after.get(oid) == before.get(oid)
    if not rec["restored"]:
        rec["key_before"], rec["key_after"] = before.get(oid), after.get(oid)
    rec["collateral"] = sorted(k for k in set(before) | set(after) if k != oid and before.get(k) != after.get(k))
    # only a clean, isolated, fully-restored flip counts as observability EVIDENCE for one axis
    rec["valid_evidence"] = bool(rec["ok"] and rec["back_ok"] and rec["restored"] and not rec["collateral"])
    return rec


def stage_capture(scene: str, root: str, cfg=kd.CAPTURE, plan_override=None, stop_when=None) -> dict:
    probe = _load(os.path.join(root, scene, "probe.json"))
    if not probe:
        raise RuntimeError(f"{scene}: run the probe stage first")
    ds = os.path.join(root, scene)
    meta_path, state_path = os.path.join(ds, "frames.jsonl"), os.path.join(ds, "capture_state.json")
    plan = plan_override if plan_override is not None else kd.episode_plan(scene, cfg["episodes"])
    state = _load(state_path)

    def fresh_quota():
        heads, combos = kd.reachable_keys(probe)
        q = kd.Quota(kd.build_targets(probe["registry"], probe["instances"], heads, cfg, combos),
                     int(cfg.get("combo_min_episodes") or 0))
        q.suppress_after = int(cfg.get("suppress_after") or 0)
        q.vp_target = int(cfg.get("viewpoint_target") or 0)
        q.held_target, q.held_bg_target = int(cfg.get("held_target") or 0), int(cfg.get("held_bg_target") or 0)
        return q

    if state:
        done_eps = set(state["done_episodes"])
        _truncate(meta_path, ds, done_eps)
        if (state["quota"].get("version") or "") != kd.QUOTA_VERSION:
            # the COUNTING UNIT changed (v2c-3: per-head keys + combination floor): rebuild the targets from the
            # probe and recount every stored frame — no recapture; opportunity counts carry over
            old = state["quota"]
            quota = fresh_quota()
            quota.tried, quota.hit = dict(old.get("tried") or {}), dict(old.get("hit") or {})
            with open(meta_path, encoding="utf-8") as fh:
                quota.replay(json.loads(l) for l in fh)
            _log(scene, f"quota recounted to {kd.QUOTA_VERSION} from {quota.frames} stored frames")
        else:
            quota = kd.Quota.from_dict(state["quota"])
    else:
        quota = fresh_quota()
        done_eps = set()
        for f in (meta_path,):
            if os.path.exists(f):
                os.remove(f)
    # every (re)start APPENDS its exact config + code revision — the first run's config is never overwritten
    # (Astra: a resume used to replace capture_config.json, losing the settings the earlier episodes were made with)
    with open(os.path.join(ds, "capture_config_history.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": POLICY_VERSION,
                             "resume_from_done": sorted(done_eps), "cfg": cfg, "versions": _versions()},
                            default=str) + "\n")
    if not os.path.exists(os.path.join(ds, "capture_config.json")):
        _dump(os.path.join(ds, "capture_config.json"), {"cfg": cfg, "plan": plan, "versions": _versions()})
    est = kd.estimate_capture_h(probe["seconds_per_view"], probe.get("seconds_per_reset") or 5.0, cfg)
    _log(scene, f"capture: {len(plan)} episodes ({cfg['episodes']}), {cfg['frames_per_episode']} frames each, "
                f"{len(probe['registry'])} types, {len(probe['state_keys'])} state keys; est. {est} h")
    failed = dict((state or {}).get("failed_episodes") or {})
    glitch_redos = dict((state or {}).get("glitch_redos") or {})
    anomaly_accepted = dict((state or {}).get("anomaly_accepted") or {})
    agent = make_agent(scene)
    try:
        for ep in plan:
            if ep["episode"] in done_eps or str(ep["episode"]) in failed:
                continue
            if stop_when is not None and stop_when(quota):
                _log(scene, f"stop rule met before episode {ep['episode']} — every train head/combination at target")
                break
            if ep["split"] == "train" and not str(ep.get("protocol", "")).startswith("amb")                     and quota.split_done("train"):
                # every TRAIN target is met: further train episodes would only grind (see _run_episode)
                _log(scene, f"ep {ep['episode']:3d} [train] skipped: every train target already met")
                done_eps.add(ep["episode"])
                _dump(state_path, {"quota": quota.to_dict(), "done_episodes": sorted(done_eps),
                                   "failed_episodes": failed, "glitch_redos": glitch_redos,
                               "anomaly_accepted": anomaly_accepted})
                continue
            result = None
            hangs = glitches = 0
            seen_stable = set()
            while hangs < 2 and glitches <= int(cfg.get("glitch_max_redos", 6)):
                try:
                    with _watchdog(cfg.get("episode_timeout_s", 600)):
                        runner = _run_amb_episode if str(ep.get("protocol", "")).startswith("amb") else _run_episode
                        result = runner(agent, scene, ds, ep, probe, quota, cfg)
                    break
                except (EpisodeHung, RenderGlitch) as exc:
                    if isinstance(exc, EpisodeHung):
                        hangs += 1
                        why = f"HUNG (> {cfg.get('episode_timeout_s', 600)} s, attempt {hangs})"
                    else:
                        glitches += 1
                        why = f"RENDER GLITCH ({exc}; redo {glitches})"
                        # CONTENT, not a glitch: the renders were stable and only the colour check fired, AND the
                        # same view does it again in a fresh process. Real glitches are transient (a fresh process
                        # renders the view clean — 2026-09-29 experiment); a close-up of a red wall is not. Accept
                        # that one view (agreement still required) and record it.
                        if exc.stable and exc.anomaly:
                            key = (ep["episode"], exc.view)
                            if key in seen_stable:
                                cfg = dict(cfg, _allow_anomaly_views=set(cfg.get("_allow_anomaly_views", ())) | {key})
                                anomaly_accepted.setdefault(str(ep["episode"]), []).append(
                                    {"view": exc.view, "signature": exc.anomaly})
                                why += f" — same view flagged again in a fresh process: legitimate content "                                        f"({exc.anomaly}), accepted for this view"
                            seen_stable.add(key)
                        # MOTION, not a glitch (2026-09-30, FloorPlan1 v3 episode 14 view 107, 3 redos): the renders
                        # disagreed and then settled on 3 identical frames, and the SAME view does it again in a fresh
                        # process — something in the scene was still moving (physics settling). A transient glitch
                        # does not recur at the same view after a restart. Accept the settled render for that view.
                        if exc.converged:
                            key = ("motion", ep["episode"], exc.view)
                            if key in seen_stable and isinstance(exc.view, str):
                                cfg = dict(cfg, _allow_motion_pairs=set(cfg.get("_allow_motion_pairs", ()))
                                           | {exc.view})
                                why += " — same flip render again in a fresh process: scene motion, accepted"
                            elif key in seen_stable:
                                cfg = dict(cfg, _allow_motion_views=set(cfg.get("_allow_motion_views", ()))
                                           | {(ep["episode"], exc.view)})
                                why += " — same view again in a fresh process: scene motion, settled render accepted"
                            seen_stable.add(key)
                    _log(scene, f"ep {ep['episode']:3d} {why} — restarting the simulator, discarding the partial "
                                f"episode and redoing it from its seed")
                    agent = _restart_agent(agent, scene)
                    st_ = _load(state_path)
                    quota = kd.Quota.from_dict(st_["quota"]) if st_ else fresh_quota()
                    _truncate(meta_path, ds, done_eps)
                    glitch_redos[str(ep["episode"])] = glitch_redos.get(str(ep["episode"]), 0) + int(
                        isinstance(exc, RenderGlitch))
            if result is None:
                failed[str(ep["episode"])] = "hung twice" if hangs >= 2 else "render glitch on every redo"
                _dump(state_path, {"quota": quota.to_dict(), "done_episodes": sorted(done_eps),
                                   "failed_episodes": failed, "glitch_redos": glitch_redos,
                               "anomaly_accepted": anomaly_accepted})
                continue
            setup, rows, attempts, accepted, n_pairs, stats = result
            # per-episode invariants BEFORE the episode is committed: an episode that fails them (e.g. recorded on
            # a degrading display) is discarded — its files are orphans the resume deletes — and redone, instead
            # of being kept (2026-09-29: episode 3 of the fresh FloorPlan1 was committed, then the abort fired)
            _fail_fast(scene, ds, ep, setup, dict(cfg, check_every=0), len(done_eps) + 1, stats)
            with open(meta_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"episode_record": setup, "attempts": attempts, "accepted": accepted,
                                     "pairs": n_pairs, "stats": stats}) + "\n")
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            done_eps.add(ep["episode"])
            _dump(state_path, {"quota": quota.to_dict(), "done_episodes": sorted(done_eps),
                               "failed_episodes": failed, "glitch_redos": glitch_redos,
                               "anomaly_accepted": anomaly_accepted})
            _checkpoint(scene, ds, cfg, len(done_eps))          # full verifier on COMMITTED data, every N eps
            unreal = sum(1 for v in setup["objects"].values() if not v["realised"])
            cov = setup.get("state_coverage") or {}
            _log(scene, f"ep {ep['episode']:3d} [{ep['split']:5s}] +{accepted}/{attempts} views, {n_pairs} pairs, "
                        f"states set {cov.get('set')}/{cov.get('settable')}, {unreal} unrealised"
                        + (", SATURATED" if stats.get("saturated") else "") + f", frames {quota.frames}")
    finally:
        try:
            agent.stop()
        except Exception:
            pass
    return stage_audit(scene, root)


class CaptureAborted(RuntimeError):
    """A data invariant broke: stop NOW rather than finish a kitchen of bad data (FloorPlan1 was captured three
    times because each defect surfaced only in the end-of-kitchen audit)."""


def _fail_fast(scene, ds, ep, setup, cfg, n_done, stats=None):
    """Per-episode invariants (cheap, every episode) + the full :mod:`check_capture` verifier every
    ``check_every`` episodes. Any FAIL raises :class:`CaptureAborted`; the completed episodes stay saved and the
    run resumes after the defect is fixed."""
    cov = setup.get("state_coverage") or {}
    if cov.get("settable") and cov["set"] / cov["settable"] < cfg["min_state_coverage"]:
        raise CaptureAborted(f"episode {ep['episode']}: only {cov['set']}/{cov['settable']} settable objects got a "
                             f"state (missed e.g. {cov.get('missed', [])[:4]})")
    objs = list(setup["objects"].values()) + [v for rd in setup.get("rounds", []) for v in rd["objects"].values()]
    bad = sum(1 for v in objs if not v["realised"])
    if objs and bad / len(objs) > 0.05:
        raise CaptureAborted(f"episode {ep['episode']}: {bad}/{len(objs)} state settings did not take")
    st = stats or {}
    if st.get("views_guarded", 0) >= 50:
        rate = st.get("guard_retries", 0) / st["views_guarded"]
        skip = st.get("guard_skipped", 0) / st["views_guarded"]
        if rate > cfg.get("guard_max_retry_rate", 0.30) or skip > cfg.get("guard_max_skip_rate", 0.05):
            raise CaptureAborted(f"episode {ep['episode']}: the render guard had to reject a render on {rate:.0%} of "
                                 f"views ({st.get('guard_skipped', 0)} skipped) — the renderer is failing; check the "
                                 f"WSLg display (a `wsl --shutdown` has cleared this before) before resuming")
    _checkpoint(scene, ds, cfg, n_done)


def _checkpoint(scene, ds, cfg, n_done):
    """Every ``check_every`` committed episodes: the full :mod:`check_capture` verifier; any FAIL aborts."""
    every = int(cfg.get("check_every") or 0)
    if every and n_done % every == 0:
        import check_capture
        rep = check_capture.check(ds, cfg)
        _log(scene, f"CHECKPOINT after {n_done} episodes — {'FAIL' if rep['fail'] else 'ok'}\n"
                    + check_capture.render(rep))
        _dump(os.path.join(ds, "checkpoints", f"after_{n_done:03d}.json"), rep)
        if rep["fail"]:
            raise CaptureAborted(f"checkpoint after {n_done} episodes failed: "
                                 + "; ".join(f"{x['check']}: {x['detail']}" for x in rep["results"]
                                             if x["status"] == "FAIL"))


class RenderGlitch(Exception):
    def __init__(self, msg, view=None, stable=False, anomaly=None, converged=False):
        super().__init__(msg)
        self.view, self.stable, self.anomaly = view, stable, anomaly
        self.converged = converged                # renders disagreed, then 3 agreed: motion OR a transient glitch

    """The render guard saw a glitched render. The simulator PROCESS is degraded from that moment until restarted
    (glitch experiment 2026-09-29: every run started clean, flipped into a degraded state after 220-2,600 views and
    stayed there; restarting at the first glitch — variant G — gave 1 glitched view in 4,848 vs 6-44 % without).
    The episode is discarded, the simulator restarted, and the episode redone from its seed."""


class EpisodeHung(BaseException):
    """Raised by the watchdog. BaseException so no per-step ``except Exception`` can swallow it."""


class _watchdog:
    """SIGALRM-based wall-clock limit for one episode (Linux; a no-op where SIGALRM does not exist).

    A stuck controller pipe read blocks forever with python and Unity both idle (seen in FloorPlan5 during
    validation); the alarm interrupts the read so the episode can be discarded and redone on a fresh simulator."""

    def __init__(self, seconds):
        self.seconds = int(seconds or 0)

    def __enter__(self):
        import signal
        if self.seconds and hasattr(signal, "SIGALRM"):
            def _raise(*_):
                raise EpisodeHung()
            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        import signal
        if self.seconds and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


def _restart_agent(agent, scene):
    import subprocess
    try:
        agent.stop()
    except BaseException:
        pass
    subprocess.run(["pkill", "-f", "thor-Linux64"], check=False)
    time.sleep(3)
    return make_agent(scene)


def _run_episode(agent, scene, ds, ep, probe, quota, cfg):
    """One episode: set up the world, then take views until the frame budget, the attempt cap, or SATURATION.

    Saturation (train only): when the last ``saturation_window`` attempts kept at most ``saturation_keep`` frames,
    the episode has nothing left to add — every remaining short target is one this layout cannot show — and it
    ends early instead of grinding to the attempt cap. (2026-09-29: late v2b episodes took 550 views over ~3 min to
    keep 35-44 frames; watching the simulator it looked stuck.) Held-out episodes never saturate: they are sampled
    by a fixed protocol, not by what training lacks."""
    if True:
            rng = random.Random(kd.stable_seed(scene, "views", ep["episode"]))
            setup = setup_episode(agent, scene, ep, probe, quota, cfg)
            positions = agent.reachable_positions()
            rows, accepted, attempts, n_pairs = [], 0, 0, 0
            stats = {"refused": 0, "empty_dropped": 0, "declined": 0, "saturated": False}
            rnd, rnd_acc, rnd_att = 0, 0, 0
            every = int(cfg.get("reroll_every") or 0)
            win, keep_max = int(cfg.get("saturation_window", 150)), int(cfg.get("saturation_keep", 8))
            recent = []                                   # 1 per attempt: was a frame kept?
            picked, n_held = set(), 0                     # v3: one pickup per instance; held share capped
            held_max = int(round(cfg.get("held_cap", 1.0) * cfg["frames_per_episode"]))
            while accepted < cfg["frames_per_episode"] and attempts < cfg["frames_per_episode"] * 5:
                if ep["split"] == "train" and len(recent) >= win and sum(recent[-win:]) <= keep_max:
                    stats["saturated"] = True
                    break
                kept_before = accepted
                # within-episode state round: re-roll the reversible axes (after any held view was put down)
                if every and (rnd_acc >= every or rnd_att >= every * 5):
                    rnd += 1
                    rnd_acc = rnd_att = 0
                    objs_r = assign_states(agent, ep, probe, quota, rng, reversible_only=True)
                    setup["rounds"].append({"round": rnd, "at_frame": accepted, "objects": objs_r,
                                            "settled_metadata": _settle(agent)})
                attempts += 1
                rnd_att += 1
                if (ep.get("protocol") != kd.CLOSE_PROTOCOL and n_held < held_max
                        and rng.random() < cfg["p_held"]):
                    views = ((v, "held") for v in held_views(agent, positions, probe, rng, cfg, quota, ep["split"],
                                                             picked))
                else:
                    pose, kind = choose_view(agent, positions, probe, quota, ep["split"], rng, cfg,
                                             protocol=ep.get("protocol"))
                    if pose is None:
                        stats["refused"] += 1
                        continue
                    views = iter([(pose, kind)])
                for pose, kind in views:
                    # render guard BEFORE labelling: RGB, masks and depth then all come from the accepted render
                    vidx = stats.get("views_guarded", 0)          # deterministic within an episode's redo
                    allow = (ep["episode"], vidx) in cfg.get("_allow_anomaly_views", ())
                    g_ok, g_n = render_guard(agent, cfg, allow_anomaly=allow)
                    stats["views_guarded"] = vidx + 1
                    if allow:
                        stats.setdefault("anomaly_accepted_views", []).append(vidx)
                    motion_ok = g_ok and (ep["episode"], vidx) in cfg.get("_allow_motion_views", ())
                    if (g_n > 1 or not g_ok) and cfg.get("restart_on_glitch") and not motion_ok:
                        raise RenderGlitch(f"episode {ep['episode']}, view {vidx}: guard needed {g_n} renders"
                                           + (f" ({GUARD_LAST['anomaly']}, renders stable)" if GUARD_LAST["stable"]
                                              and not g_ok else ""),
                                           view=vidx, stable=bool(GUARD_LAST["stable"] and not g_ok),
                                           anomaly=GUARD_LAST["anomaly"] if not g_ok else None,
                                           converged=bool(g_ok))
                    if g_n > 1:
                        stats["guard_retries"] = stats.get("guard_retries", 0) + 1
                    if not g_ok:
                        stats["guard_skipped"] = stats.get("guard_skipped", 0) + 1
                        continue
                    labels = label_frame(agent, cfg)
                    # an observation OPPORTUNITY for the targeted object's state: seen large enough, or not
                    tgt = (pose or {}).get("target")
                    if tgt:
                        tobj = agent.object_by_id(tgt)
                        if tobj and vocab.axes_of(tobj["objectType"]):
                            key = vocab.state_key(tobj["objectType"], vocab.state_record(tobj))
                            quota.note_target(key, any(l["oid"] == tgt and l["px"] >= cfg["state_min_px"]
                                                       for l in labels))
                    if kind == "held" and not any(l["oid"] == pose["held"] and l["px"] >= 200 for l in labels):
                        # the held object was moved out of frame / hidden behind the counter: not a held view
                        # (v3 smoke 2026-09-30: 1 of 63 held frames had no box for the held object)
                        stats["held_unseen"] = stats.get("held_unseen", 0) + 1
                        continue
                    if not labels and rng.random() >= cfg["p_empty"]:
                        stats["empty_dropped"] += 1
                        continue
                    cell = view_cell(agent)
                    g = quota.gain(labels, ep["split"])
                    if ep["split"] == "train":
                        g += quota.vp_gain(labels, cell)
                        if kind == "held":                # a held view is taken FOR the held object's deficit
                            g += 2.0 * quota.held_deficit(pose["inst"])
                    if ep["split"] == "train" and labels and not kd.accept_train(g, rng, cfg):
                        stats["declined"] += 1
                        continue
                    name = f"{scene}_e{ep['episode']:03d}_{accepted:04d}"
                    save_frame(agent, ds, ep["split"], name)
                    quota.add(labels, ep["split"], ep["episode"])
                    if ep["split"] == "train":
                        quota.add_vp(labels, cell)
                        if kind == "held":
                            quota.add_held(pose["inst"], pose["cell"], (pose["x"], pose["z"]))
                    if kind == "held":
                        n_held += 1
                    row = {"img": name, "split": ep["split"], "episode": ep["episode"], "kind": kind, "guard": g_n,
                           "pose": pose, "camera": camera_record(agent), "cell": cell, "gain": round(g, 3),
                           "seg_colors": seg_colors(agent, labels), "labels": labels}
                    if kind != "held" and rng.random() < cfg["p_pair"]:
                        pr = paired_render(agent, ds, ep["split"], name, labels, rng, cfg, allow_anomaly=allow)
                        if pr:
                            row["pair"] = pr
                            n_pairs += 1
                    row["round"] = rnd
                    row["protocol"] = ep.get("protocol") or "main"
                    rows.append(row)
                    accepted += 1
                    rnd_acc += 1
                    if accepted >= cfg["frames_per_episode"]:
                        break
                if hasattr(views, "close"):
                    views.close()                         # a held object is always put down before the next view
                recent.append(1 if accepted > kept_before else 0)
            return setup, rows, attempts, accepted, n_pairs, stats


# ======================================================================================================================
# Ambiguity (specialist) capture — protocol amb-v1, frozen in docs/ambiguity-spec.md.
# ======================================================================================================================
def place_partners(agent, rng, cfg) -> list:
    """For ``partner_pairs`` random confusable pairs present in the layout, put the partner next to the target on the
    same surface: surface CLEARANCE (bounding-box edge to edge, in the floor plane) drawn from ``partner_gap``, so a
    pot and a plate are spaced like a fork and a spoon. Verified after physics settles; failures are recorded."""
    by_type = {}
    for o in agent.objects():
        if o.get("position") and o.get("pickupable"):
            by_type.setdefault(o["objectType"], []).append(o)
    pairs = [p for p in kd.AMB_PAIRS if p[0] in by_type and p[1] in by_type]
    rng.shuffle(pairs)
    out = []
    for a_t, b_t in pairs:
        if sum(1 for x in out if x["ok"]) >= int(cfg.get("partner_pairs", 2)):
            break
        a, b = rng.choice(by_type[a_t]), rng.choice(by_type[b_t])
        if rng.random() < 0.5:
            a, b = b, a
        ab, bb = a.get("axisAlignedBoundingBox") or {}, b.get("axisAlignedBoundingBox") or {}
        ac, asz, bsz = ab.get("center") or a["position"], ab.get("size") or {}, bb.get("size") or {}
        ra = 0.5 * max(asz.get("x", 0.1), asz.get("z", 0.1))
        rb = 0.5 * max(bsz.get("x", 0.1), bsz.get("z", 0.1))
        gap = rng.uniform(*cfg.get("partner_gap", (0.08, 0.20)))
        floor_y = ac["y"] - 0.5 * asz.get("y", 0.0) + 0.01
        rec = {"target": a["objectId"], "partner": b["objectId"], "pair": [a_t, b_t], "gap": round(gap, 3), "ok": False}
        for k in range(8):
            ang = rng.uniform(0, 2 * math.pi)
            d = ra + rb + gap
            pt = {"x": ac["x"] + d * math.cos(ang), "y": floor_y, "z": ac["z"] + d * math.sin(ang)}
            ev = agent.do("PlaceObjectAtPoint", objectId=b["objectId"], position=pt)
            if not ev.metadata.get("lastActionSuccess"):
                continue
            for _ in range(5):
                agent.do("Pass")
            nb = agent.object_by_id(b["objectId"])
            if nb and math.hypot(nb["position"]["x"] - pt["x"], nb["position"]["z"] - pt["z"]) < 0.15 \
                    and abs(nb["position"]["y"] - floor_y) < 0.25:
                rec.update(ok=True, tries=k + 1, point={k_: round(v, 3) for k_, v in pt.items()})
                break
        out.append(rec)
    return out


def open_parents(agent, obj) -> list:
    """Open every closed openable receptacle holding ``obj`` (a layout shuffle hides many targets in cabinets)."""
    by_id = {o["objectId"]: o for o in agent.objects()}
    opened = []
    for rid in (obj.get("parentReceptacles") or []):
        r = by_id.get(rid)
        if r and r.get("openable") and not r.get("isOpen"):
            if agent.do("OpenObject", objectId=rid, openness=1.0, forceAction=True).metadata.get("lastActionSuccess"):
                opened.append(rid)
    return opened


def aim_pose(agent, positions, probe, obj, band, rng, cfg):
    """Stand ``band`` metres from ``obj`` and look at it (jittered); teleport. Returns the pose or None."""
    c = (obj.get("axisAlignedBoundingBox") or {}).get("center") or obj["position"]
    d = rng.uniform(*band)
    ranked = sorted(positions, key=lambda q: abs(math.hypot(q["x"] - c["x"], q["z"] - c["z"]) - d))
    p = rng.choice(ranked[:6])
    standing = rng.random() >= cfg["p_crouch"]
    eye = p["y"] + (probe["eye_stand"] if standing else probe["eye_crouch"])
    dx, dy, dz = c["x"] - p["x"], c["y"] - eye, c["z"] - p["z"]
    yaw = math.degrees(math.atan2(dx, dz)) + rng.gauss(0.0, cfg["yaw_jitter"] / 3)
    hor = -math.degrees(math.atan2(dy, max(math.hypot(dx, dz), 1e-6))) + rng.uniform(-8, 8)
    pose = {"x": p["x"], "y": p["y"], "z": p["z"], "rot": round(yaw % 360.0, 1),
            "hor": round(max(-30.0, min(60.0, hor)), 1), "standing": standing, "target": obj["objectId"]}
    agent.teleport_full(pose["x"], pose["y"], pose["z"], pose["rot"], horizon=pose["hor"], standing=standing)
    return pose if agent.success else None


def _amb_guard(agent, ep, cfg, stats):
    """The render guard with the same glitch / content / motion handling as the main capture."""
    vidx = stats.get("views_guarded", 0)
    allow = (ep["episode"], vidx) in cfg.get("_allow_anomaly_views", ())
    g_ok, g_n = render_guard(agent, cfg, allow_anomaly=allow)
    stats["views_guarded"] = vidx + 1
    motion_ok = g_ok and (ep["episode"], vidx) in cfg.get("_allow_motion_views", ())
    if (g_n > 1 or not g_ok) and cfg.get("restart_on_glitch") and not motion_ok:
        raise RenderGlitch(f"episode {ep['episode']}, view {vidx}: guard needed {g_n} renders",
                           view=vidx, stable=bool(GUARD_LAST["stable"] and not g_ok),
                           anomaly=GUARD_LAST["anomaly"] if not g_ok else None, converged=bool(g_ok))
    if g_n > 1:
        stats["guard_retries"] = stats.get("guard_retries", 0) + 1
    if not g_ok:
        stats["guard_skipped"] = stats.get("guard_skipped", 0) + 1
    return g_ok, g_n


def _run_amb_episode(agent, scene, ds, ep, probe, quota, cfg):
    """One amb-v1 episode: a fresh layout, partners placed, then the FIXED view schedule (kd.amb_schedule).
    Each slot is retried up to kd.AMB_SLOT_TRIES times until its condition holds (close/ordinary: the aimed target is
    visible; free: no target visible at all; held: the held target is visible); a slot that never holds is logged."""
    rng = random.Random(kd.stable_seed(scene, "amb-views", ep["episode"]))
    setup = setup_episode(agent, scene, ep, probe, quota, cfg)
    setup["partners"] = place_partners(agent, rng, cfg)
    setup["settled_after_partners"] = _settle(agent)
    positions = agent.reachable_positions()
    targets = set(kd.AMB_TARGETS)
    rows, accepted, attempts = [], 0, 0
    stats = {"refused": 0, "slot_failed": {}, "slots": {}}
    seen_t = {t: 0 for t in kd.AMB_TARGETS}
    pickup_no = 0

    def _partner_visible(pose, labels):
        held = next((l for l in labels if l["oid"] == (pose or {}).get("held")), None)
        if not held:
            return False
        mates = {b for a, b in kd.AMB_PAIRS if a == held["type"]} | {a for a, b in kd.AMB_PAIRS if b == held["type"]}
        return any(l["type"] in mates and l["oid"] != held["oid"] for l in labels)
    views_of, used_views, free_cells = {}, set(), set()   # per-instance view count; (target, cell); negative cells
    schedule = kd.amb_schedule(rng)

    def keep(kind, pose, labels, g_n):
        nonlocal accepted
        name = f"{scene}_e{ep['episode']}_{accepted:04d}"
        save_frame(agent, ds, ep["split"], name)
        quota.add(labels, ep["split"], ep["episode"])
        for l in labels:
            if l["type"] in seen_t:
                seen_t[l["type"]] += 1
        dist = None
        tid = (pose or {}).get("held") or (pose or {}).get("target")
        tob = agent.object_by_id(tid) if tid else None
        if tob:
            cam = agent.event.metadata.get("cameraPosition") or {}
            c = (tob.get("axisAlignedBoundingBox") or {}).get("center") or tob["position"]
            dist = round(math.dist((cam.get("x", 0), cam.get("y", 0), cam.get("z", 0)), (c["x"], c["y"], c["z"])), 2)
        realised = kind if kind in ("free", "held", "held_other") else ("close" if dist is not None and dist <= 0.8 else "ordinary")
        rows.append({"img": name, "split": ep["split"], "episode": ep["episode"], "kind": kind, "guard": g_n,
                     "requested_view": kind, "realised_view": realised, "target_dist_m": dist,
                     # deployment-time gate state (the robot knows it is holding something up for inspection) and,
                     # for evaluation only, the held object, its pickup and whether a look-alike partner is in view
                     "inspection": kind.startswith("held"), "pickup": pickup_no if kind.startswith("held") else None,
                     "partner_visible": _partner_visible(pose, labels) if kind == "held" else None,
                     "pose": pose, "camera": camera_record(agent), "cell": view_cell(agent),
                     "seg_colors": seg_colors(agent, labels), "labels": labels, "round": 0,
                     "protocol": ep["protocol"]})
        stats["slots"][kind] = stats["slots"].get(kind, 0) + 1
        accepted += 1

    for kind in [k for k in schedule if not k.startswith("held")]:
        done, failed, slot_opened = False, set(), []
        for _try in range(kd.AMB_SLOT_TRIES):
            attempts += 1
            objs = [o for o in agent.objects() if o.get("position")]
            if kind in ("close", "ordinary"):
                # balanced over target classes, but a failed object is not retried in this slot (smoke 2026-10-03:
                # the least-seen target was usually one shut inside a cabinet, and 8 retries on it all failed)
                by_id_ = {o["objectId"]: o for o in objs}
                cands = [o for o in objs if o["objectType"] in targets and instance_key(o, {}) not in failed
                         and views_of.get(instance_key(o, {}), 0) < kd.AMB_MAX_VIEWS_PER_OBJECT]
                if not cands:
                    break
                # prefer objects in the open: one shut in a cabinet is shown only when its class has no other
                # instance in view-able places (smoke 2026-10-03: the same fridge shelf, tiny, 5 times in a row)
                def shut(o):
                    return any((by_id_.get(r) or {}).get("openable") and not (by_id_.get(r) or {}).get("isOpen")
                               for r in (o.get("parentReceptacles") or []))
                free_c = [o for o in cands if not shut(o)]
                cands = free_c if free_c and rng.random() < 0.85 else cands
                order = sorted({o["objectType"] for o in cands}, key=lambda t: (seen_t[t], rng.random()))
                t = order[0]
                obj = rng.choice([o for o in cands if o["objectType"] == t])
                opened = open_parents(agent, obj)
                slot_opened += opened
                obj = agent.object_by_id(obj["objectId"]) or obj
                pose = aim_pose(agent, positions, probe, obj,
                                cfg["close_band"] if kind == "close" else cfg["dist_band"], rng, cfg)
                if pose is not None:
                    pose["opened"] = opened
                need_px = 400 if kind == "close" else 15
            else:
                dis = [o for o in objs if o["objectType"] in kd.AMB_DISTRACTORS
                       and views_of.get(instance_key(o, {}), 0) < 3]
                if dis and rng.random() < 0.7:
                    pose = aim_pose(agent, positions, probe, rng.choice(dis), (0.4, 2.0), rng, cfg)
                else:
                    p = rng.choice(positions)
                    pose = {"x": p["x"], "y": p["y"], "z": p["z"], "rot": round(rng.uniform(0, 360), 1),
                            "hor": round(rng.uniform(-10, 40), 1), "standing": rng.random() >= cfg["p_crouch"],
                            "target": None}
                    agent.teleport_full(pose["x"], pose["y"], pose["z"], pose["rot"], horizon=pose["hor"],
                                        standing=pose["standing"])
                    pose = pose if agent.success else None
            if pose is None:
                stats["refused"] += 1
                continue
            g_ok, g_n = _amb_guard(agent, ep, cfg, stats)
            if not g_ok:
                continue
            labels = label_frame(agent, cfg)
            if kind == "free":
                # a hard NEGATIVE: no target anywhere, but real content (>= 4 objects) — not a bare wall
                if any(l["type"] in targets for l in labels) or len(labels) < 4                         or tuple(view_cell(agent)) in free_cells:
                    continue
            elif (pose["target"], tuple(view_cell(agent))) in used_views:
                continue                                                    # same object from the same spot again
            elif not any(l["oid"] == pose["target"] and l["px"] >= need_px for l in labels):
                failed.add(instance_key(agent.object_by_id(pose["target"]) or {"objectId": pose["target"]}, {}))
                continue
            keep(kind, pose, labels, g_n)
            if kind == "free":
                free_cells.add(tuple(view_cell(agent)))
                if pose.get("target"):
                    ik = instance_key(agent.object_by_id(pose["target"]) or {"objectId": pose["target"]}, {})
                    views_of[ik] = views_of.get(ik, 0) + 1
            else:
                ik = instance_key(agent.object_by_id(pose["target"]) or {"objectId": pose["target"]}, {})
                views_of[ik] = views_of.get(ik, 0) + 1
                used_views.add((pose["target"], tuple(view_cell(agent))))
            done = True
            break
        if not done:
            stats["slot_failed"][kind] = stats["slot_failed"].get(kind, 0) + 1
        for rid in slot_opened:
            agent.do("CloseObject", objectId=rid, forceAction=True)

    # held views LAST (a dropped object lands elsewhere): first non-target objects in hand ("held_other": the
    # specialist must not call a held mug a pot), then the targets — the specialist's real use case is an object held
    # up to the camera (Arash, 2026-10-03), so most of the episode is held targets: one pickup per instance, several
    # views per pickup, each from a new spot facing content or the target's look-alike partner
    partner_of = {}
    for a_, b_ in kd.AMB_PAIRS:
        partner_of.setdefault(a_, set()).add(b_)
        partner_of.setdefault(b_, set()).add(a_)
    pickup_no = 0
    for kind, types, quota_n in (("held_other", None, kd.AMB_VIEWS["held_other"]),
                                 ("held", targets, kd.AMB_VIEWS["held"])):
        if types is None:
            types = {o["objectType"] for o in agent.objects() if o.get("pickupable")} - targets
        n_held, picked, pickups = 0, set(), 0
        tcfg = dict(cfg, held_views=kd.AMB_HELD_PER_PICKUP, held_face_partner=0.4 if kind == "held" else 0.0)
        while n_held < quota_n and pickups < 20:
            pickups += 1
            pickup_no += 1
            before = n_held
            gen = held_views(agent, positions, probe, rng, tcfg, None, ep["split"], picked, types=types,
                             face_map=partner_of if kind == "held" else None)
            for pose in gen:
                attempts += 1
                g_ok, g_n = _amb_guard(agent, ep, cfg, stats)
                if not g_ok:
                    continue
                labels = label_frame(agent, cfg)
                if not any(l["oid"] == pose["held"] and l["px"] >= kd.AMB_HELD_MIN_PX for l in labels):
                    continue
                keep(kind, pose, labels, g_n)
                n_held += 1
                if n_held >= quota_n:
                    break
            gen.close()
            if n_held == before and not any(o.get("pickupable") and o["objectType"] in types
                                            and instance_key(o, {}) not in picked for o in agent.objects()):
                break
        if n_held < quota_n:
            stats["slot_failed"][kind] = quota_n - n_held
    stats["targets_seen"] = seen_t
    return setup, rows, attempts, accepted, 0, stats


def _truncate(meta_path, ds, done_eps):
    """On resume, drop rows and files of any episode not recorded as complete.

    Files are removed by the EPISODE NUMBER in their name (``<scene>_e048_...``), not only by meta rows: an episode
    interrupted mid-way has written images but not yet its rows (rows are appended when the episode ends), and those
    orphans would otherwise survive the resume (found 2026-09-28 when the v2b capture was stopped in episode 48)."""
    import re
    for sub in ("", "pairs"):
        for kind in ("images", "depth", "seg"):
            for split in ("train", "val", "test"):
                d = os.path.join(ds, sub, kind, split)
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    m = re.search(r"_e(\d{3,})_", f)
                    if m and int(m.group(1)) not in done_eps:
                        os.remove(os.path.join(d, f))
    if not os.path.exists(meta_path):
        return
    keep = []
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ep = r["episode_record"]["episode"] if "episode_record" in r else r.get("episode")
            if ep in done_eps:
                keep.append(line)
            elif "img" in r:
                for sub, kind, ext in (("", "images", ".jpg"), ("", "depth", ".png"), ("", "seg", ".png")):
                    p = os.path.join(ds, kind, r["split"], r["img"] + ext)
                    if os.path.exists(p):
                        os.remove(p)
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.writelines(keep)


# ======================================================================================================================
# Stage: quarantine — frames captured before the render guard existed.
# ======================================================================================================================
def stage_quarantine(scene: str, root: str, cfg=kd.CAPTURE) -> dict:
    """Mark every stored frame whose RGB shows a corruption signature (:func:`frame_anomaly`) as quarantined — in
    ``frames.jsonl``, reversibly: the files stay, a backup of the records is kept — then recount the quota without
    them. Every consumer (replay, check_capture, the yolo view, state_review) skips quarantined rows. For data
    captured before the render guard (FloorPlan1 v2c episodes 0-79, 2026-09-29); the residual (subtle casts below
    the threshold) is estimated by a random-sample visual audit, not assumed zero (Astra step3f)."""
    import numpy as np
    from PIL import Image
    ds = os.path.join(root, scene)
    path = os.path.join(ds, "frames.jsonl")
    shutil.copyfile(path, path + f".pre_quarantine_{time.strftime('%Y%m%d_%H%M%S')}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    reasons = {}
    for r in rows:
        if "img" not in r:
            continue
        a = np.asarray(Image.open(os.path.join(ds, "images", r["split"], r["img"] + ".jpg")).convert("RGB"))
        why = frame_anomaly(a)
        if why:
            r["quarantined"] = why
            reasons[why] = reasons.get(why, 0) + 1
        elif r.get("quarantined"):
            r.pop("quarantined")
        if r.get("pair"):
            bad = []
            for suf in ("_noise", "_flip"):
                p = os.path.join(ds, "pairs", "images", r["split"], r["img"] + suf + ".jpg")
                if os.path.exists(p) and frame_anomaly(np.asarray(Image.open(p).convert("RGB"))):
                    bad.append(suf)
            if bad:
                r["pair"]["valid_evidence"] = False
                r["pair"]["quarantined"] = bad
                reasons["pair"] = reasons.get("pair", 0) + 1
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(path + ".tmp", path)
    st = _load(os.path.join(ds, "capture_state.json"))
    q = kd.Quota.from_dict(st["quota"])
    q.replay(rows)
    st["quota"] = q.to_dict()
    _dump(os.path.join(ds, "capture_state.json"), st)
    n_img = sum(1 for r in rows if "img" in r)
    rep = {"scene": scene, "frames": n_img, "quarantined": reasons, "frames_after": dict(q.frames),
           "rule": "frame_anomaly: pure_white >= 99% px, or whole-image channel cast > 60"}
    _dump(os.path.join(ds, "quarantine.json"), rep)
    _publish(scene, "quarantine.json", rep)
    _log(scene, f"quarantine: {reasons} of {n_img} frames; quota recounted -> {q.frames}")
    return rep


# ======================================================================================================================
# Stage: audit.
# ======================================================================================================================
def stage_audit(scene: str, root: str) -> dict:
    ds = os.path.join(root, scene)
    probe = _load(os.path.join(ds, "probe.json"))
    st = _load(os.path.join(ds, "capture_state.json"))
    quota = kd.Quota.from_dict(st["quota"])
    unreachable = sorted({x["key"] for v in probe["unreachable"].values() for x in v} - set(probe["state_keys"]))
    rep = kd.audit(quota, probe["registry"], unreachable)
    # state-setting and paired-render health, from the episode records
    unreal, pairs, restored, collateral, valid, rounds, unreal_r = 0, 0, 0, 0, 0, 0, 0
    low_cov, saturated = [], 0
    fpath = os.path.join(ds, "frames.jsonl")
    lines = open(fpath, encoding="utf-8").readlines() if os.path.exists(fpath) else []   # no episode completed yet
    if True:
        for line in lines:
            r = json.loads(line)
            if "episode_record" in r:
                er = r["episode_record"]
                unreal += sum(1 for v in er["objects"].values() if not v["realised"])
                cov = er.get("state_coverage")
                if cov and cov["settable"] and cov["set"] / cov["settable"] < kd.CAPTURE["min_state_coverage"]:
                    low_cov.append({"episode": er["episode"], **cov})
                if r.get("stats", {}).get("saturated"):
                    saturated += 1
                for rd in er.get("rounds", []):
                    rounds += 1
                    unreal_r += sum(1 for v in rd["objects"].values() if not v["realised"])
            elif r.get("pair"):
                pairs += 1
                restored += int(bool(r["pair"].get("restored")))
                collateral += int(bool(r["pair"].get("collateral")))
                valid += int(bool(r["pair"].get("valid_evidence")))
    rep["state_setting_unrealised"] = unreal
    rep["low_state_coverage_episodes"] = low_cov
    rep["saturated_episodes"] = saturated
    rep["failed_episodes"] = st.get("failed_episodes") or {}
    if low_cov or rep["failed_episodes"]:
        rep["ok"] = False
    rep["rerolls"] = {"rounds": rounds, "unrealised": unreal_r}
    rep["pairs"] = {"n": pairs, "restored": restored, "with_collateral_changes": collateral,
                    "valid_evidence": valid}
    rep["counts"] = quota.counts
    _dump(os.path.join(ds, "audit.json"), rep)
    _publish(scene, "audit.json", {k: v for k, v in rep.items() if k != "counts"})
    _log(scene, f"AUDIT {'PASS' if rep['ok'] else 'FAIL'}: {rep['n_short']} shortfalls, missing types "
                f"{rep['missing_types'] or 'none'}, {unreal} unrealised states, rerolls {rep['rerolls']}, "
                f"pairs {rep['pairs']}, tried-never-observed {len(rep['tried_never_observed'])}, "
                f"low-coverage episodes {len(low_cov)}, saturated {saturated}, failed {len(rep['failed_episodes'])}")
    return rep


# ======================================================================================================================
# Stage: yolo — the DETECTOR's view of the store (type-only labels, explicit split lists).
# ======================================================================================================================
def detection_gate(ds: str, probe: dict, cfg=kd.CAPTURE) -> dict:
    """The detector's gate (Astra step3h): recount DETECTION units (every type, every instance) from MAIN-protocol
    frames only — close-protocol episodes cannot hide a main-protocol gap — plus integrity (no failed / low-coverage
    episodes, every row's files present). State-unit shortfalls do NOT gate the detector; they gate the state
    estimator."""
    rows, eps = [], []
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            (eps if "episode_record" in r else rows).append(r)
    main = [r for r in rows if not r.get("quarantined") and r.get("protocol", "main") != kd.CLOSE_PROTOCOL]
    gcfg = dict(cfg, type_target=cfg.get("gate_type_target", cfg["type_target"]),
                instance_target=cfg.get("gate_instance_target", cfg["instance_target"]))
    q = kd.Quota(kd.build_targets(probe["registry"], probe["instances"], [], gcfg))
    for r in main:
        q.add(r["labels"], r["split"])
    short = {u: {s: {k: [q.counts[u][s].get(k, 0), n] for k, n in tg.items() if q.counts[u][s].get(k, 0) < n}
                 for s, tg in q.targets[u].items()} for u in ("type", "inst")}
    n_short = sum(len(v) for u in short.values() for v in u.values())
    missing = sorted(set(probe["registry"]) - {k for s in q.counts["type"].values() for k in s})
    st = _load(os.path.join(ds, "capture_state.json")) or {}
    failed = st.get("failed_episodes") or {}
    low = [e["episode_record"]["episode"] for e in eps
           if (e["episode_record"].get("state_coverage") or {}).get("settable")
           and e["episode_record"]["state_coverage"]["set"] / e["episode_record"]["state_coverage"]["settable"]
           < cfg["min_state_coverage"]]
    missing_files = [r["img"] for r in rows if not r.get("quarantined")
                     and not os.path.exists(os.path.join(ds, "images", r["split"], r["img"] + ".jpg"))]
    ok = not n_short and not missing and not failed and not low and not missing_files
    return {"ok": ok, "detection_short": short, "n_short": n_short, "missing_types": missing,
            "failed_episodes": failed, "low_coverage_episodes": low, "missing_files": missing_files[:10]}


def stage_yolo(scene: str, root: str, allow_failed_audit: bool = False) -> dict:
    ds = os.path.join(root, scene)
    probe = _load(os.path.join(ds, "probe.json"))
    gate = detection_gate(ds, probe)
    _publish(scene, "detection_gate.json", gate)
    if not gate["ok"] and not allow_failed_audit:
        raise RuntimeError(f"{scene}: detection gate FAILED ({gate['n_short']} type/instance shortfalls, missing types "
                           f"{gate['missing_types']}, failed {list(gate['failed_episodes'])}, low-coverage "
                           f"{gate['low_coverage_episodes']}) — refusing to build a training set "
                           f"(--allow-failed-audit overrides, and the override is recorded)")
    registry = probe["registry"]
    cid = {t: i for i, t in enumerate(registry)}
    out = os.path.join(ds, "yolo_primary")
    lists = {"train": [], "val": [], "test": [], "val_close": [], "test_close": []}
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "img" not in r or r.get("quarantined"):
                continue
            for kind in ("images", "labels"):
                os.makedirs(os.path.join(out, kind, r["split"]), exist_ok=True)
            src = os.path.join(ds, "images", r["split"], r["img"] + ".jpg")
            dst = os.path.join(out, "images", r["split"], r["img"] + ".jpg")
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            lines = []
            for lab in r["labels"]:
                x1, y1, x2, y2 = lab["box"]
                lines.append(f"{cid[lab['type']]} {(x1 + x2 + 1) / 2 / FRAME:.6f} {(y1 + y2 + 1) / 2 / FRAME:.6f} "
                             f"{(x2 - x1 + 1) / FRAME:.6f} {(y2 - y1 + 1) / FRAME:.6f}")
            with open(os.path.join(out, "labels", r["split"], r["img"] + ".txt"), "w", encoding="utf-8") as f2:
                f2.write("\n".join(lines) + ("\n" if lines else ""))
            close = r.get("protocol") == kd.CLOSE_PROTOCOL
            key = r["split"] + ("_close" if close and r["split"] in ("val", "test") else "")
            lists[key].append(dst)
    for k, v in lists.items():
        with open(os.path.join(out, f"{k}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(v)) + "\n")
    names = [f"  {i}: {t}" for i, t in enumerate(registry)]
    for yaml_name, val_list, test_list in (("data.yaml", "val.txt", "test.txt"),
                                          ("data_close.yaml", "val_close.txt", "test_close.txt")):
        with open(os.path.join(out, yaml_name), "w", encoding="utf-8") as fh:
            fh.write("\n".join([f"path: {out}", "train: train.txt", f"val: {val_list}", f"test: {test_list}", "",
                                f"nc: {len(registry)}", "names:"] + names) + "\n")
    rep = {"images": {k: len(v) for k, v in lists.items()}, "classes": len(registry), "gate": gate["ok"],
           "gate_overridden": bool(not gate["ok"] and allow_failed_audit), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    _dump(os.path.join(out, "build.json"), rep)
    _publish(scene, "yolo_build.json", rep)
    _log(scene, f"yolo_primary: {rep['images']}, {len(registry)} classes, detection gate {'PASS' if gate['ok'] else 'FAIL'}")
    return rep


def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(x + "\n" for x in lines))          # an empty label file = a true negative frame


def stage_yolo_amb(scene: str, root: str) -> dict:
    """Build the specialist's training set from the amb-v1 store (docs/ambiguity-spec.md).

    ``yolo_amb/``   10 target classes only (kd.AMB_TARGETS); every target instance labelled, the rest background.
    ``yolo_amb49/`` the same frames with all 49 type labels — for the full-pipeline evaluation and the control.
    ``data_plus.yaml`` (in yolo_amb49) = the CONTROL's set: primary train + amb train (49 labels; replay of the original
    data included), validation = primary val + amb val. Holdout episodes (protocol amb-v1-holdout) go into NO list;
    only a sealed manifest records them."""
    ds = os.path.join(root, scene)
    probe = _load(os.path.join(ds, "probe.json"))
    registry = probe["registry"]
    cid49 = {t: i for i, t in enumerate(registry)}
    cid10 = {t: i for i, t in enumerate(kd.AMB_TARGETS)}
    outs = {"amb": os.path.join(ds, "yolo_amb"), "amb49": os.path.join(ds, "yolo_amb49")}
    lists = {k: {"train": [], "val": [], "test": []} for k in outs}
    holdout = []
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "img" not in r or r.get("quarantined"):
                continue
            if r.get("protocol") == kd.AMB_HOLDOUT:
                holdout.append(r["img"])
                continue
            src = os.path.join(ds, "images", r["split"], r["img"] + ".jpg")
            for k, out in outs.items():
                for kind in ("images", "labels"):
                    os.makedirs(os.path.join(out, kind, r["split"]), exist_ok=True)
                dst = os.path.join(out, "images", r["split"], r["img"] + ".jpg")
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
                cmap = cid10 if k == "amb" else cid49
                lines = []
                for lab in r["labels"]:
                    if lab["type"] not in cmap:
                        continue
                    x1, y1, x2, y2 = lab["box"]
                    lines.append(f"{cmap[lab['type']]} {(x1 + x2 + 1) / 2 / FRAME:.6f} {(y1 + y2 + 1) / 2 / FRAME:.6f} "
                                 f"{(x2 - x1 + 1) / FRAME:.6f} {(y2 - y1 + 1) / FRAME:.6f}")
                _write_lines(os.path.join(out, "labels", r["split"], r["img"] + ".txt"), lines)
                lists[k][r["split"]].append(dst)
    for k, out in outs.items():
        for sp, v in lists[k].items():
            _write_lines(os.path.join(out, f"{sp}.txt"), sorted(v))
        names = kd.AMB_TARGETS if k == "amb" else registry
        _write_lines(os.path.join(out, "data.yaml"),
                     [f"path: {out}", "train: train.txt", "val: val.txt", "test: test.txt", "", f"nc: {len(names)}",
                      "names:"] + [f"  {i}: {t}" for i, t in enumerate(names)])
    prim = os.path.join(DEFAULT_ROOT, scene, "yolo_primary")
    a49 = outs["amb49"]
    for sp in ("train", "val"):
        rows_ = [l.strip() for l in open(os.path.join(prim, f"{sp}.txt")) if l.strip()] + lists["amb49"][sp]
        _write_lines(os.path.join(a49, f"plus_{sp}.txt"), rows_)
    _write_lines(os.path.join(a49, "data_plus.yaml"),
                 [f"path: {a49}", "train: plus_train.txt", "val: plus_val.txt", "test: test.txt", "",
                  f"nc: {len(registry)}", "names:"] + [f"  {i}: {t}" for i, t in enumerate(registry)])
    _dump(os.path.join(ds, "holdout_manifest.json"), {"sealed": True, "protocol": kd.AMB_HOLDOUT,
                                                       "images": sorted(holdout), "n": len(holdout)})
    rep = {k: {sp: len(v) for sp, v in d.items()} for k, d in lists.items()}
    rep["holdout_sealed"] = len(holdout)
    _dump(os.path.join(ds, "yolo_amb_build.json"), rep)
    _log(scene, f"yolo_amb: {rep}")
    return rep


# ======================================================================================================================
# Driver.
# ======================================================================================================================
SMOKE = {"episodes": {"train": 3, "val": 1, "test": 1}, "frames_per_episode": 30, "held_cap": 0.4,
         "type_target": {"train": 5, "val": 1, "test": 1}, "instance_target": {"train": 1, "val": 0, "test": 0},
         "state_target": {"train": 2, "val": 0, "test": 0}, "p_pair": 0.5, "p_held": 0.4, "reroll_every": 4,
         "suppress_after": 5, "saturation_window": 20, "saturation_keep": 1, "episode_timeout_s": 300}


SMOKE_AMB = {"episode_timeout_s": 400}


def run(scenes, stages, root, smoke=False, allow_failed_audit=False):
    cfg = dict(kd.CAPTURE, **(SMOKE if smoke else {}))
    root = os.path.join(root, "smoke") if smoke else root
    for scene in scenes:
        for st in stages:
            _log(scene, f"=== {st} ===")
            if st == "probe":
                if not _load(os.path.join(root, scene, "probe.json")):
                    stage_probe(scene, root)
            elif st == "capture":
                stage_capture(scene, root, cfg)
            elif st == "audit":
                stage_audit(scene, root)
            elif st == "topup":
                stage_capture(scene, root, cfg, plan_override=kd.topup_train_plan(scene, kd.TOPUP_TRAIN_MAX),
                              stop_when=kd.train_complete)
                stage_capture(scene, root, cfg, plan_override=kd.close_plan(scene, kd.CLOSE_EPISODES))
            elif st == "amb":
                # the ambiguity store shares the kitchen's probe (same scene, same reachable states)
                os.makedirs(os.path.join(root, scene), exist_ok=True)
                pp = os.path.join(root, scene, "probe.json")
                if not os.path.exists(pp):
                    import shutil
                    src = os.path.join(DEFAULT_ROOT, scene, "probe.json")
                    shutil.copyfile(src, pp)
                acfg = dict(kd.CAPTURE_AMB, **(SMOKE_AMB if smoke else {}))
                plan = kd.amb_plan(scene)
                if smoke:
                    plan = [e for e in plan if e["episode"] in (3000, 3100, 3300)]
                stage_capture(scene, root, acfg, plan_override=plan)
            elif st == "yolo_amb":
                stage_yolo_amb(scene, root)
            elif st == "quarantine":
                stage_quarantine(scene, root, cfg)
            elif st == "yolo":
                stage_yolo(scene, root, allow_failed_audit)
            else:
                raise ValueError(f"unknown stage {st!r}; stages are {STAGES}")


def cmd_select_scenes():
    inv = _load(os.path.join(HERE, "dataset", "scene_inventory.json"))
    items = _load(os.path.join(HERE, "dataset", "general.json"))["items"]
    sel = kd.select_scenes(kd.feasibility(items, inv), inv["scenes"])
    print(f"{sel['n_items']} instructions -> {', '.join(sel['scenes'])}: feasible somewhere "
          f"{sel['feasible_somewhere']}, everywhere {sel['feasible_everywhere']}, nowhere {sel['uncovered'] or 'none'}")
    _dump(os.path.join(REPORT_ROOT, "scene_selection.json"), sel)
    return sel


def cmd_plan():
    c = kd.CAPTURE
    n = sum(c["episodes"].values())
    print(f"per kitchen: {n} episodes {c['episodes']} x {c['frames_per_episode']} frames = {n * c['frames_per_episode']}"
          f" frames; targets type {c['type_target']}, instance {c['instance_target']}, state {c['state_target']}")
    print(f"estimate at 0.25 s/view, 5 s/episode: {kd.estimate_capture_h(0.25, 5.0)} h capture per kitchen "
          f"(the probe measures the real speed)")
    print(f"training (pinned ultralytics {kd.ULTRALYTICS_VERSION}): {json.dumps(kd.TRAIN)}")
    print(f"acceptance: {json.dumps(kd.ACCEPTANCE)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select-scenes")
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--scenes", default=",".join(kd.DEFAULT_SCENES))
    r.add_argument("--stages", default=",".join(STAGES))
    r.add_argument("--root", default=DEFAULT_ROOT)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--allow-failed-audit", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "select-scenes":
        cmd_select_scenes()
    elif a.cmd == "plan":
        cmd_plan()
    else:
        try:
            run([s for s in a.scenes.split(",") if s], [s for s in a.stages.split(",") if s], a.root, a.smoke,
                a.allow_failed_audit)
        except CaptureAborted as exc:
            print(f"[{time.strftime('%H:%M:%S')}] ABORTED (fail-fast): {exc}", flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
