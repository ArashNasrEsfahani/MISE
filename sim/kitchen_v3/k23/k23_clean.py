"""K23 dataset cleaning before training (Kitchens 2/3), Astra K2-CLEAN-r1 (accept with conditions).

Marks frame rows of a K23 store (main or amb) as quarantined, reusing the frozen Kitchen 1 MECHANISM of
build_kitchen_data.stage_quarantine (a "quarantined" field on frames.jsonl rows; the frozen list builders skip such rows)
but not its function (it would clear earlier flags). Reversible: an immutable timestamped backup is written first, the new
file replaces the old one atomically, every quarantined row carries {"clean": {"reason", "dup_of", "rule"}}.

Rules, in this fixed precedence (a frame gets the first that applies; earlier flags are never cleared):
  1. xsplit_dup  -- a held-out frame listed in <k23>/<scene>/heldout_exclusions.json (Astra K2-XSPLIT-r1): any split.
  2. blur        -- TRAIN only: k23_qc.laplacian_var(k23_qc.gray(rgb)) < frame_limits.lapvar_low of the frozen thresholds
                    (qc_thresholds_k1.json for the main store, qc_thresholds_k1_amb.json for amb) (K23-BLUR-1).
  3. xsplit_near_dup -- TRAIN only (K2-PARITY, Astra K2-PARITY-r1 3): the check_capture thumbnail rule (< 6.0) against a
                    HELD-OUT frame (val/test, any protocol) shot from the SAME camera spot (3-D |dpos| <= 5 cm, |dyaw| and
                    |dhorizon| <= 5 deg) AND the same annotations (identical (type, state) multiset, every box
                    matched at IoU >= 0.8; validated 2026-10-17: thumbnail + pose alone flagged 937 pairs, 826 of
                    them with different objects present); the TRAIN copy is quarantined, held-out membership kept.
  4. exact_dup   -- TRAIN only: byte-identical (sha256) to an EARLIER retained TRAIN frame (any episode); keeps the first.
  5. near_dup    -- TRAIN only, Arash 2026-10-16: the check_capture thumbnail (grey, 32x32 resize, mean abs diff < 6.0)
                    against EARLIER RETAINED frames of the same episode. NOTE (Astra r1 1): check_capture compares with every
                    earlier frame; "compare with earlier retained frames, keep the first" is THIS cleaner's selection
                    policy (frame order = frames.jsonl order). A frame already removed by rules 1-3 never suppresses a
                    later frame.
VAL / TEST / close / amb-holdout frames are not filtered by rules 2-4 (held-out populations stay comparable with Kitchen 1;
only the leakage exclusion touches them). A row's paired render follows its parent ("pair.quarantined").

    python sim/kitchen_v3/k23/k23_clean.py --scene FloorPlan15 --store main [--dry-run]      (WSL, thor-env python)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SIM)
sys.path.insert(0, HERE)
VERSION = "k23-clean-4"
RULES = ("xsplit_dup", "blur", "xsplit_near_dup", "exact_dup", "near_dup")
POSE_M, POSE_DEG = 0.05, 5.0          # xsplit_near_dup confirmation: same camera spot (Astra K2-PARITY-r1 3)


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def store_dir(scene: str, store: str) -> str:
    root = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3_amb" if store == "amb" else "kitchen_v3")
    return os.path.join(root, scene)


def thresholds_path(store: str) -> str:
    return os.path.join(HERE, "qc_thresholds_k1_amb.json" if store == "amb" else "qc_thresholds_k1.json")


def _cam(r: dict):
    c = r.get("camera") or {}
    pos = c.get("position") or {}
    if "x" not in pos or c.get("yaw") is None:
        return None
    return (float(pos["x"]), float(pos["z"]), float(c["yaw"]) % 360.0, float(c.get("horizon") or 0.0),
            float(pos.get("y") or 0.0))


def _labels(r: dict) -> list:
    return [(x["type"], x["box"], x.get("state_key")) for x in r.get("labels") or []]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / ua if ua else 0.0


def same_annotations(A: list, B: list, thr: float = 0.8) -> bool:
    """Identical multiset of (type, state_key) AND a PERFECT one-to-one matching of boxes (bipartite, augmenting paths;
    an edge = same type, same state_key, IoU >= thr). Astra K2-PARITY-r2 2-3: order-independent; state evidence agrees."""
    import collections
    if collections.Counter((t, s) for t, _, s in A) != collections.Counter((t, s) for t, _, s in B):
        return False
    adj = [[j for j, (t2, b2, s2) in enumerate(B) if t2 == t and s2 == s and _iou(b, b2) >= thr] for t, b, s in A]
    match_b = {}

    def aug(i, seen) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_b or aug(match_b[j], seen):
                match_b[j] = i
                return True
        return False
    return all(aug(i, set()) for i in range(len(A)))


def held_index(rows: list, ds: str, load) -> dict:
    """{(round(x,1), round(z,1)): [(img, cam, thumb32)]} of every held-out (val/test, any protocol) frame not quarantined."""
    import numpy as np
    idx = {}
    for r in rows:
        if "img" not in r or r["split"] == "train" or r.get("quarantined"):
            continue
        cam = _cam(r)
        p = os.path.join(ds, "images", r["split"], r["img"] + ".jpg")
        if cam is None or not os.path.exists(p):
            continue
        t = np.asarray(load(p).convert("L").resize((32, 32)), dtype=np.float32)
        idx.setdefault((round(cam[0], 1), round(cam[1], 1)), []).append((r["img"], cam, t, _labels(r)))
    return idx


def held_twin(r: dict, t, idx: dict):
    """The held-out frame this TRAIN frame near-duplicates FROM THE SAME CAMERA SPOT (|dpos| <= 5 cm, |dyaw|, |dhorizon|
    <= 5 deg) with the check_capture thumbnail rule (mean abs diff < 6.0), else None. Conservative by construction: a
    look-alike view from a different spot (e.g. another object state) is never matched."""
    import numpy as np
    cam = _cam(r)
    if cam is None:
        return None
    x, z, yaw, hor, y = cam
    for dx in (-0.1, 0.0, 0.1):
        for dz in (-0.1, 0.0, 0.1):
            for img, c, u, lab in idx.get((round(x + dx, 1), round(z + dz, 1)), ()):
                dyaw = abs((yaw - c[2] + 180.0) % 360.0 - 180.0)
                if (((x - c[0]) ** 2 + (z - c[1]) ** 2 + (y - c[4]) ** 2) ** 0.5 <= POSE_M and dyaw <= POSE_DEG
                        and abs(hor - c[3]) <= POSE_DEG and float(np.abs(t - u).mean()) < 6.0
                        and same_annotations(_labels(r), lab)):
                    return img
    return None


def decide(rows: list, ds: str, lapvar_low: float, excluded, img_loader=None) -> dict:
    """{row index: (reason, dup_of)} for rows that become quarantined now (rows already quarantined are skipped)."""
    import numpy as np
    from PIL import Image
    import k23_qc
    load = img_loader or (lambda p: Image.open(p))
    out = {}
    seen_sha = {}
    thumbs = {}
    held = held_index(rows, ds, load)                      # held-out thumbnails by camera cell (rule 3)
    for i, r in enumerate(rows):
        if "img" not in r or r.get("quarantined"):
            continue
        img = r["img"]
        if img in excluded:
            out[i] = ("xsplit_dup", excluded[img] if isinstance(excluded, dict) else None)   # r2 4: the TRAIN partner
            continue
        if r["split"] != "train":
            continue
        p = os.path.join(ds, "images", "train", img + ".jpg")
        if not os.path.exists(p):
            continue                                    # a missing file is the verifier's job, never silently 'clean'
        im = load(p)
        rgb = np.asarray(im.convert("RGB"))
        if k23_qc.laplacian_var(k23_qc.gray(rgb)) < lapvar_low:
            out[i] = ("blur", None)
            continue
        t = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32)       # check_capture's thumbnail
        twin = held_twin(r, t, held)
        if twin is not None:
            out[i] = ("xsplit_near_dup", twin)
            continue
        h = sha_file(p)
        if h in seen_sha:
            out[i] = ("exact_dup", seen_sha[h])
            continue
        prev = thumbs.setdefault(r["episode"], [])
        hit = next((name for name, u in prev if float(np.abs(t - u).mean()) < 6.0), None)
        if hit is not None:
            out[i] = ("near_dup", hit)
            continue
        seen_sha[h] = img
        prev.append((img, t))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    ap.add_argument("--store", required=True, choices=("main", "amb"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ds = store_dir(a.scene, a.store)
    fj = os.path.join(ds, "frames.jsonl")
    thp = thresholds_path(a.store)
    th = json.load(open(thp, encoding="utf-8"))
    exp = os.path.join(HERE, a.scene, "heldout_exclusions.json")
    if a.store == "main" and not os.path.exists(exp):        # r2 5: a missing exclusions file is an error, never 'none'
        print(f"REFUSED: {exp} missing"); return 2
    excl = ({e["frame"]: e.get("duplicate_of") for e in json.load(open(exp, encoding="utf-8"))["entries"]}
            if os.path.exists(exp) else {})
    rows = [json.loads(l) for l in open(fj, encoding="utf-8")]
    before = sha_file(fj)
    dec = decide(rows, ds, float(th["frame_limits"]["lapvar_low"]), excl)
    counts = {}
    for i, (why, dup) in dec.items():
        r = rows[i]
        counts.setdefault(r["split"], {}).setdefault(why, 0)
        counts[r["split"]][why] += 1
        r["quarantined"] = why
        r["clean"] = {"reason": why, "dup_of": dup, "rule": VERSION}
        if isinstance(r.get("pair"), dict) and not r["pair"].get("quarantined"):     # r2 4: keep an earlier reason
            r["pair"]["quarantined"] = "parent:" + why
            r["pair"]["valid_evidence"] = False
    kept = {}
    inst = {}
    for r in rows:
        if "img" in r and not r.get("quarantined"):
            kept[r["split"]] = kept.get(r["split"], 0) + 1
            if r["split"] == "train":
                for lab in r.get("labels") or []:
                    inst[lab["type"]] = inst.get(lab["type"], 0) + 1
    total = {}
    for r in rows:
        if "img" in r:
            total[r["split"]] = total.get(r["split"], 0) + 1
    marked = {r["img"] for r in rows if "img" in r and r.get("quarantined")}
    missing_excl = sorted(set(excl) - marked) if a.store == "main" else []
    rep = {"what": f"K23 cleaning {a.scene} {a.store} store ({VERSION}; Astra K2-CLEAN-r1)", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "dry_run": a.dry_run, "store": ds, "frames_jsonl_sha256_before": before,
           "thresholds": {"file": os.path.basename(thp), "sha256": sha_file(thp), "lapvar_low": th["frame_limits"]["lapvar_low"]},
           "exclusions": {"file": exp, "sha256": sha_file(exp) if os.path.exists(exp) else None, "n": len(excl),
                          "not_found_or_unmarked": missing_excl},
           "cleaner_sha256": sha_file(os.path.abspath(__file__)), "rules": RULES,
           "frames_total_by_split": total, "newly_quarantined_by_split_reason": counts, "frames_kept_by_split": kept,
           "train_instances_kept": dict(sorted(inst.items())),
           "quarantined": [{"img": rows[i]["img"], "split": rows[i]["split"], "reason": w, "dup_of": d} for i, (w, d) in sorted(dec.items())]}
    out_dir = os.path.join(HERE, a.scene, "clean")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"clean_{a.store}_{stamp}{'_dry' if a.dry_run else ''}.json")
    if missing_excl:                                        # r2 5: refuse BEFORE any replacement
        rep["refused"] = f"exclusions not found / unmarked: {missing_excl}"
        json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
        print(rep["refused"]); return 1
    with open(path, "w", encoding="utf-8") as fh:          # r2 5: the manifest exists before the store changes
        json.dump(dict(rep, state="planned"), fh, indent=1)
    if not a.dry_run and dec:
        bak = fj + f".pre_clean_{stamp}"
        shutil.copyfile(fj, bak)
        os.chmod(bak, 0o444)                              # immutable backup
        tmp = fj + ".clean_tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, fj)
        rep["backup"] = bak
    rep["frames_jsonl_sha256_after"] = sha_file(fj)
    rep["state"] = "dry-run" if a.dry_run else ("applied" if dec else "no-op")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("frames_total_by_split", "newly_quarantined_by_split_reason", "frames_kept_by_split")}))
    print(f"exclusions not found/unmarked: {missing_excl}; report {path}")
    return 0 if not missing_excl else 1


if __name__ == "__main__":
    raise SystemExit(main())
