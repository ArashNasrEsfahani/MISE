"""Deep, read-only verification of a kitchen capture WHILE it runs — so a defect is caught in the first episodes,
not after a whole kitchen (FloorPlan1 was captured three times because each defect surfaced only at the end).

Checks, each PASS / WARN / FAIL with the evidence:

1. files      every frame row has its image, depth and segmentation file; no orphans from completed episodes
2. coverage   every episode put >= min_state_coverage of the settable objects into a verified state
3. movable    in SHUFFLED episodes, movable stateful objects (apple, egg, mug …) really show non-default states
              — the check that would have caught the v2b renaming bug directly
4. unrealised share of state settings that did not take (physics can undo a few; many means a bug)
5. instances  every object instance (by NAME) has labels, per split, so far
6. held       held "floating" views of one pickup actually differ (a reset grip gives identical silhouettes)
7. pairs      paired renders restored cleanly (valid evidence)
8. pace       accepted / attempted views per episode (a collapse means grinding)
9. projection at the current rate, will every state key reach its TRAIN target by the end of the plan?

    python sim/check_capture.py --scene FloorPlan1                       # WSL (reads ~/thor-data/<version>)
    python sim/check_capture.py --scene FloorPlan1 --root //wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v2c

Exit code 0 = no FAIL, 1 = at least one FAIL. :func:`check` is also called by the capture itself after its
first episodes (the early checkpoint), which ABORTS the run on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import kitchen_data as kd  # noqa: E402
import vocab  # noqa: E402

MOVABLE_TYPES = {"Apple", "Tomato", "Lettuce", "Bread", "Potato", "Egg", "Mug", "Cup", "Bowl", "Plate", "Pot", "Pan",
                 "Bottle", "WineBottle", "Kettle"}


def _load_rows(ds):
    eps, frames = [], []
    path = os.path.join(ds, "frames.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("quarantined"):
                    continue                        # corrupted render, excluded (build_kitchen_data quarantine)
                (eps if "episode_record" in r else frames).append(r)
    return eps, frames


def check(ds: str, cfg: dict = kd.CAPTURE, deep_files: bool = True) -> dict:
    probe = json.load(open(os.path.join(ds, "probe.json"), encoding="utf-8"))
    eps, frames = _load_rows(ds)
    res = []

    def add(name, status, detail):
        res.append({"check": name, "status": status, "detail": detail})

    if not eps:
        add("progress", "WARN", "no completed episode yet")
        return {"results": res, "fail": False}
    done = {e["episode_record"]["episode"] for e in eps}

    # 1. files
    if deep_files:
        missing = [r["img"] for r in frames if not all(
            os.path.exists(os.path.join(ds, k, r["split"], r["img"] + x))
            for k, x in (("images", ".jpg"), ("depth", ".png"), ("seg", ".png")))]
        orphans = 0
        for k in ("images", "depth", "seg"):
            for s in ("train", "val", "test"):
                d = os.path.join(ds, k, s)
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        m = re.search(r"_e(\d{3,})_", f)
                        if m and int(m.group(1)) not in done and int(m.group(1)) < max(done):
                            orphans += 1
        add("files", "FAIL" if missing or orphans else "PASS",
            f"{len(frames)} frame rows; {len(missing)} missing files; {orphans} orphans in completed range")

    # 2. coverage
    low = []
    for e in eps:
        c = e["episode_record"].get("state_coverage")
        if c is None:
            low.append((e["episode_record"]["episode"], "no coverage record"))
        elif c["settable"] and c["set"] / c["settable"] < cfg["min_state_coverage"]:
            low.append((e["episode_record"]["episode"], f"{c['set']}/{c['settable']}"))
    add("coverage", "FAIL" if low else "PASS",
        f"{len(eps)} episodes; below {cfg['min_state_coverage']:.0%}: {low[:5] or 'none'}")

    # 3. movable objects in shuffled episodes really vary
    shuffled = {e["episode_record"]["episode"] for e in eps if (e["episode_record"].get("spawn") or {}).get("ok")}
    keys_by_type, default_only = {}, []
    for r in frames:
        if r["episode"] not in shuffled:
            continue
        for lab in r["labels"]:
            if lab["type"] in MOVABLE_TYPES and vocab.axes_of(lab["orig_type"]):
                keys_by_type.setdefault(lab["type"], set()).add(lab["state_key"])
    for t, ks in sorted(keys_by_type.items()):
        if len(ks) < 2:
            default_only.append(t)
    status = "PASS"
    if shuffled and len(keys_by_type) and len(default_only) > len(keys_by_type) / 2:
        status = "FAIL"
    elif default_only and len(shuffled) >= 4:
        status = "WARN"
    add("movable", status, f"{len(shuffled)} shuffled episodes; movable types with a single state so far: "
                           f"{default_only or 'none'}; distinct states per type: "
                           f"{ {t: len(k) for t, k in sorted(keys_by_type.items())} }")

    # 4. unrealised
    n_set = n_bad = 0
    for e in eps:
        er = e["episode_record"]
        for objs in [er["objects"]] + [rd["objects"] for rd in er.get("rounds", [])]:
            n_set += len(objs)
            n_bad += sum(1 for v in objs.values() if not v["realised"])
    rate = n_bad / n_set if n_set else 0.0
    add("unrealised", "FAIL" if rate > 0.05 else ("WARN" if rate > 0.01 else "PASS"),
        f"{n_bad} of {n_set} state settings did not take ({rate:.1%}; physics can undo a few)")

    # 5. instances by name, per split
    seen = {"train": set(), "val": set(), "test": set()}
    for r in frames:
        for lab in r["labels"]:
            seen[r["split"]].add(lab.get("inst") or lab["oid"])
    inst = set(probe["instances"])
    miss = {s: sorted(inst - v) for s, v in seen.items() if v}
    n_miss_train = len(miss.get("train", []))
    add("instances", "FAIL" if len(done) >= 10 and n_miss_train > len(inst) * 0.2 else
        ("WARN" if any(miss.values()) else "PASS"),
        "; ".join(f"{s}: {len(inst) - len(m)}/{len(inst)} seen" + (f" (missing e.g. {m[:3]})" if m else "")
                  for s, m in miss.items()))

    # 6. held views differ
    groups = {}
    for r in frames:
        if r.get("kind") == "held":
            h = (r["pose"] or {}).get("held")
            lab = next((l for l in r["labels"] if l["oid"] == h), None)
            if lab:
                groups.setdefault((r["episode"], h), []).append(tuple(lab["box"]) + (lab["px"],))
    same = sum(1 for v in groups.values() if len(v) > 1 and len(set(v)) == 1)
    multi = sum(1 for v in groups.values() if len(v) > 1)
    add("held", "FAIL" if multi and same / multi > 0.3 else ("WARN" if same else "PASS"),
        f"{multi} held sequences; {same} with identical silhouettes in every view")

    # 7. pairs
    prs = [r["pair"] for r in frames if r.get("pair")]
    valid = sum(1 for p in prs if p.get("valid_evidence"))
    add("pairs", "WARN" if prs and valid / len(prs) < 0.85 else "PASS",
        f"{len(prs)} paired renders, {valid} valid evidence")

    # 8. pace
    slow = [(e["episode_record"]["episode"], e["accepted"], e["attempts"]) for e in eps
            if e["attempts"] and e["accepted"] / e["attempts"] < 0.2]
    run = 0
    worst_run = 0
    for e in sorted(eps, key=lambda e: e["episode_record"]["episode"]):
        run = run + 1 if (e["attempts"] and e["accepted"] / e["attempts"] < 0.2) else 0
        worst_run = max(worst_run, run)
    add("pace", "WARN" if worst_run >= 3 else "PASS",
        f"episodes keeping < 20% of views: {len(slow)} (longest streak {worst_run}); "
        f"saturated: {sum(1 for e in eps if e.get('stats', {}).get('saturated'))}")

    # 9. projection of TRAIN per-head state keys (v2c-3 unit), from the RECENT train episodes' rate — not the
    #    whole-run average, which mixes older policies (Astra step3e) — plus the combination floor
    train_eps = sorted(e["episode_record"]["episode"] for e in eps if e["episode_record"]["split"] == "train")
    n_train_done, n_train_plan = len(train_eps), cfg["episodes"]["train"]
    recent = set(train_eps[-8:])
    have, recent_n, combo, combo_eps = {}, {}, {}, {}
    for r in frames:
        if r["split"] != "train":
            continue
        for lab in r["labels"]:
            if lab["px"] < cfg["state_min_px"] or not lab.get("state"):
                continue
            for k in vocab.axis_keys(lab.get("orig_type") or lab["type"], lab["state"]):
                have[k] = have.get(k, 0) + 1
                if r["episode"] in recent:
                    recent_n[k] = recent_n.get(k, 0) + 1
            sk = lab.get("state_key") or ""
            if sk and not sk.endswith("|-"):
                combo[sk] = combo.get(sk, 0) + 1
                combo_eps.setdefault(sk, set()).add(r["episode"])
    heads, combos = kd.reachable_keys(probe)
    target = cfg["state_target"]["train"]
    if n_train_done >= 3:
        left = max(0, n_train_plan - n_train_done)
        rate = {k: recent_n.get(k, 0) / max(1, len(recent)) for k in heads}
        proj = {k: have.get(k, 0) + rate[k] * left for k in heads}
        short = sorted((k for k in heads if proj[k] < target), key=lambda k: proj[k])
        add("projection", "WARN" if short else "PASS",
            f"after {n_train_done}/{n_train_plan} train episodes, {len(short)} of {len(heads)} per-head state keys "
            f"project below {target} at the last {len(recent)} episodes' rate: "
            + ", ".join(f"{k}~{proj[k]:.0f}" for k in short[:6]))
    else:
        add("projection", "WARN", f"only {n_train_done} train episodes — too early to project")
    floor, min_eps = int(cfg.get("combo_floor") or 0), int(cfg.get("combo_min_episodes") or 0)
    if floor:
        low = sorted(k for k in combos if combo.get(k, 0) < floor or len(combo_eps.get(k, ())) < min_eps)
        add("combos", "WARN" if low else "PASS",
            f"{len(combos) - len(low)}/{len(combos)} reachable combinations have >= {floor} large train crops from "
            f">= {min_eps} episodes; below: " + ", ".join(f"{k} ({combo.get(k, 0)}/{len(combo_eps.get(k, ()))}ep)"
                                                          for k in low[:5]))

    # 10-14. v3 diversity invariants (Arash 2026-09-30: "the same object from the same place in the same angle
    #         several times"; "most of the data should be from the original colours"; "each object held with many
    #         different backgrounds")
    if cfg.get("held_cap"):
        _diversity(ds, cfg, eps, frames, add, deep_files)

    return {"results": res, "fail": any(x["status"] == "FAIL" for x in res), "episodes": len(eps)}


def _diversity(ds, cfg, eps, frames, add, deep_files):
    main = [r for r in frames if (r.get("protocol") or "main") == "main"]
    # 10. one pickup per instance per episode: at most held_views frames of one instance in one episode
    per = {}
    for r in main:
        if r.get("kind") == "held":
            k = (r["episode"], (r["pose"] or {}).get("inst") or (r["pose"] or {}).get("held"))
            per[k] = per.get(k, 0) + 1
    over = sorted((k, n) for k, n in per.items() if n > cfg["held_views"])
    add("repick", "FAIL" if over else "PASS",
        f"{len(per)} (episode, object) held sequences; over {cfg['held_views']} frames: {over[:4] or 'none'}")
    # 11. held share
    n_held = sum(1 for r in main if r.get("kind") == "held")
    share = n_held / len(main) if main else 0.0
    add("held_share", "FAIL" if len(eps) >= 4 and share > cfg["held_cap"] + 0.03 else "PASS",
        f"{n_held}/{len(main)} frames held ({share:.1%}; cap {cfg['held_cap']:.0%})")
    # 12. original colours: frames from episodes with no material and no lighting randomisation
    orig_eps = {e["episode_record"]["episode"] for e in eps
                if not e["episode_record"].get("materials") and not e["episode_record"].get("lighting")}
    mats = sum(1 for e in eps if e["episode_record"].get("materials"))
    o_share = sum(1 for r in frames if r["episode"] in orig_eps) / len(frames) if frames else 0.0
    add("colours", "FAIL" if mats else ("WARN" if len(eps) >= 8 and o_share < 0.7 else "PASS"),
        f"{o_share:.0%} of frames in original colours + lighting; {mats} material-randomised episodes")
    # 13. held context: the held object is seen against content, not bare walls
    ctx = sorted((r["pose"] or {}).get("context", 0) for r in main if r.get("kind") == "held")
    bare = sum(1 for c in ctx if c < 3)
    add("held_context", "WARN" if ctx and bare / len(ctx) > 0.1 else "PASS",
        f"held views: median {ctx[len(ctx) // 2] if ctx else '-'} other objects in view; {bare} with < 3")
    # 14. held background diversity per instance (train) and near-duplicate frames (recent episodes)
    bg = {}
    for r in main:
        if r["split"] == "train" and r.get("kind") == "held":
            ps = r["pose"] or {}
            bg.setdefault(ps.get("inst"), set()).add(tuple(ps.get("cell") or ()))
    if bg:
        lo = sorted(bg.items(), key=lambda kv: len(kv[1]))[:4]
        add("held_backgrounds", "PASS", f"{len(bg)} objects held; distinct background cells min "
            + ", ".join(f"{k} {len(v)}" for k, v in lo))
    if deep_files:
        import numpy as np
        from PIL import Image
        recent = sorted({r["episode"] for r in frames})[-8:]
        dup = tot = 0
        for e in recent:
            th = []
            for r in frames:
                if r["episode"] != e:
                    continue
                p = os.path.join(ds, "images", r["split"], r["img"] + ".jpg")
                if os.path.exists(p):
                    th.append(np.asarray(Image.open(p).convert("L").resize((32, 32)), dtype=np.float32))
            for i in range(len(th)):
                tot += 1
                if any(float(np.abs(th[i] - th[j]).mean()) < 6.0 for j in range(i)):
                    dup += 1
        rate = dup / tot if tot else 0.0
        add("near_duplicates", "FAIL" if tot >= 200 and rate > 0.04 else ("WARN" if rate > 0.02 else "PASS"),
            f"{dup}/{tot} frames of the last {len(recent)} episodes near-duplicate an earlier frame "
            f"of the same episode ({rate:.1%})")


def render(rep: dict) -> str:
    return "\n".join(f"  {x['status']:4s} {x['check']:10s} {x['detail']}" for x in rep["results"])


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v2c"))
    ap.add_argument("--no-files", action="store_true", help="skip the file-existence scan (faster over \\\\wsl$)")
    a = ap.parse_args(argv)
    rep = check(os.path.join(a.root, a.scene), deep_files=not a.no_files)
    print(f"{a.scene}: {rep.get('episodes', 0)} episodes — {'FAIL' if rep['fail'] else 'no FAIL'}")
    print(render(rep))
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
