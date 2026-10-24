"""Per-batch quality check for the FloorPlan15 / FloorPlan17 perception capture (K23). Read-only on the data.

After each capture batch it reports, against thresholds DERIVED FROM THE KITCHEN 1 DATASET (``baseline`` mode, run once
over FloorPlan1's main store in 8-episode windows, the batch size):

  class balance   train instances per class (batch + cumulative), min / median, classes under the gate floor (500)
                  and the capture aim (1,500) at the current rate, classes never seen, the Kitchen 1 rate per batch
                  for the shared classes; val / test counts (counts only)
  image quality   (TRAIN frames only, every ``stride``-th frame of an episode) blur = variance of the Laplacian,
                  exposure = mean luminance + clipped-pixel share, the capture's own corruption signatures
                  (build_kitchen_data.frame_anomaly), near-duplicates inside an episode (check_capture's 32x32 rule)
  masks / labels  mask pixel-size distribution and the share of very small objects; label sanity against the stored
                  instance-segmentation masks (pixel count + box re-derived from the mask); metadata sanity (type in
                  the registry, box inside the frame, the capture's min sizes)
  pairing / ids   every frame row has its image / depth / seg file and no file lacks a row; the class-ID mapping
                  (id = index in the probe registry, as stage_yolo writes it) is complete and equals the expected list
  duplicates      exact duplicate images by sha256 (all splits; hashes only), cross-split duplicates are a FAIL
  integrity       check_capture (scene + root explicit), the layout fingerprint check, Kitchen 1 unchanged

Verdict (Astra K23-PREP 4, 7): FAIL = a hard rule failed; WARN = a quality warning (Kitchen 1 band exceeded); NOTE = an
expected-shortfall notice only (projections early in the capture). FAIL and WARN both STOP the supervisor, and the batch
stays OPEN until a person writes qc/resolution_batch_NNN.json; every later QC run refuses while it is open. A missing or
unreadable report is a STOP, never a PASS.

Barrier (A28 item 4): this QC decodes pixels of TRAIN frames only; for VAL / TEST it uses rows, counts, file existence
and sha256. Registered exception (A28 item 15): the capture's own check_capture verifier (frozen Kitchen 1 code, also
called here) decodes small grey thumbnails of the last 8 committed episodes of EVERY split for its near-duplicate test;
automated, nothing displayed, no person or model sees them, nothing is selected or tuned on them. The registered human inspection sample (frame 0 and frame 55 of every 8th committed episode, TRAIN only)
is LISTED in the report (names only); nothing is displayed.

    # once, offline, Kitchen 1 (development data) -> thresholds
    python sim/kitchen_v3/k23/k23_qc.py baseline --ds //wsl.localhost/Ubuntu-22.04/home/arash/thor-data/kitchen_v3/FloorPlan1
    # after each batch (WSL; scene and root always explicit)
    python sim/kitchen_v3/k23/k23_qc.py batch --scene FloorPlan15 --root /home/arash/thor-data/kitchen_v3
    exit 0 = PASS (NOTE lines = expected shortfall notices only), 1 = FAIL, 3 = quality WARN, 2 = could not run,
    5 = an earlier FAIL / WARN unresolved, 4 = nothing new; everything but 0 STOPS the supervisor
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.dirname(SIM)
for p in (SIM, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import kitchen_data as kd  # noqa: E402
import vocab  # noqa: E402
import k23_decide  # noqa: E402  (report_content_problems, valid_resolution; no heavy imports)

THRESHOLDS = os.path.join(HERE, "qc_thresholds_k1.json")
WINDOW = 8                 # episodes per Kitchen 1 window = per K23 batch
STRIDE = 4                 # every 4th TRAIN frame of an episode is decoded
SMALL_PX = 400             # "very small object": mask < 20 x 20 px (reported; capture keeps >= 60 px)
FRAME = 640
GATE_TRAIN = kd.CAPTURE["gate_type_target"]["train"]       # 500
AIM_TRAIN = kd.CAPTURE["type_target"]["train"]             # 1,500
N_TRAIN_MAIN = kd.CAPTURE["episodes"]["train"]             # 128
MIN_EPS_FOR_PROJECTION = 24
MIN_EPS_FOR_UNSEEN = 16
# exit codes: 0 PASS (NOTE lines allowed), 1 FAIL, 2 could not run, 3 quality WARN (stop and consult),
# 4 nothing new to check, 5 an earlier FAIL / WARN is unresolved
EXIT_WARN, EXIT_NOTHING_NEW, EXIT_UNRESOLVED = 3, 4, 5
# check_capture checks whose WARN is an expected-shortfall notice while the capture is partial (everything else stops)
CC_NOTE_CHECKS = ("progress", "projection", "instances", "combos")   # state-target projections + combination floors (top-up work)
# R1 (Arash 2026-10-14, decision 7; A28 15.4a): check_capture's `pace` (throughput, not image quality) is a NOTE
R1_NOTE_CHECKS = ("pace",)
# K23-AMD-2 (Arash 2026-10-15, amendment_K23-AMD-2.json): on these scenes the exposure SHARE check is reported as a NOTE
# (never stops) because the scene is naturally brighter than FloorPlan1. exposure_extreme, blur, near_dup and every
# other check are unchanged.
AMD2_EXPOSURE_NOTE_SCENES = ("FloorPlan15", "FloorPlan17")   # FP17: Arash 2026-10-17 ~18:25 ("approve. It's the same treatment you gave Kitchen 2.")
# cc_pairs denominator (Arash 2026-10-16 15:45 IST, ledger entry "2026-10-16 15:45 IST -- ARASH DECISION: start Kitchen 2
# ...", arash_rulings_k23.json cc_pairs_denominator_20261016): on these scenes check_capture's `pairs` WARN is judged with
# the K23-AMD-1 Blinds Open/Close pairs (never sent, recorded failed by design) EXCLUDED from the denominator; both values
# are reported. Same 0.85 threshold; every other pair (incl. other Blinds pairs) counts; check_capture.py is unchanged.
AMD1_PAIRS_EXCL_SCENES = ("FloorPlan15",)
PAIRS_MIN = 0.85
# Duplicates ruling (Arash 2026-10-16 16:40 IST, ledger "2026-10-16 16:40 IST -- ARASH RULING (K2/K3 capture): duplicates
# never stop the capture; removed before training; datasets of similar size"): on these scenes the duplicate checks
# (check_capture near_duplicates, near_dup, dup_exact) never stop the capture at any rate; the found severity and rate
# are recorded in a NOTE; duplicate frames are removed before training. dup_cross_split (split leakage) and every other
# check are unchanged. FloorPlan17 is added only when its capture is authorized.
DUP_RULING_SCENES = ("FloorPlan15", "FloorPlan17")   # FP17: approve-all-6 item 2 + Astra K3-BATCH1-r1 (2026-10-17)
DUP_RULING_CHECKS = ("cc_near_duplicates", "near_dup", "dup_exact")
DUP_RULING = "ARASH RULING 2026-10-16 16:40 (duplicates never stop; removed before training)"


# Delegated amendment K23-BLUR-1 (main session 2026-10-16 ~17:35 under Arash's 16:52 delegation, Astra K2-K1TEST-r3 Q2; prompted by
# batch 006; Arash ~17:30: "be softer on kitchen 2 as its lightning situation is vastly brighter than kitchen 1"): for FloorPlan15 a
# blur WARN is recorded as a NOTE (measured value + original WARN kept), never a stop; the blur FAIL line is unchanged and still stops;
# blurry frames (same frozen per-frame blur test) are removed before training, counts per split reported.
BLUR_NOTE_SCENES = ("FloorPlan15", "FloorPlan17")    # FP17: approve-all-6 item 2 + Astra K3-BATCH1-r1 (2026-10-17)
BLUR_RULING = "K23-BLUR-1 2026-10-16 (delegated; blur WARN = NOTE, FAIL unchanged; blurry frames removed before training)"


# Delegated standing ruling K23-TOPUP-CC-1 (main session 2026-10-17 under Arash's delegation and his "Go for the top ups.";
# Astra K23-PARALLEL-r1 item 1): during a verified topup / close run (the supervisor's QC call carries --all-protocols for
# those pipelines only), an authentic cc_projection WARN -- the end-of-main TRAIN shortfall the capped top-up exists to
# remedy -- is recorded as found and reported as a NOTE (never a stop); every FAIL, every other check and missing / malformed
# evidence keep their treatment; the terminal deficits and the stop reason (quota met / cap exhausted) are reported at the end.
TOPUP_CC_RULING = "K23-TOPUP-CC-1 2026-10-17 (delegated; cc_projection WARN = NOTE during topup/close only; FAIL unchanged)"


def topup_cc_ruling(out: list, active: bool) -> list:
    if not active:
        return out
    return [("NOTE", n, f"{TOPUP_CC_RULING}; found {st}: {d}"[:300]) if n == "cc_projection" and st == "WARN"
            else (st, n, d) for st, n, d in out]


# K2-AMB (Arash 2026-10-17 "Approve 1–5." item 2; Astra K2-AMB-r1 5): the amb store follows a FIXED plan (Kitchen 1's amb
# capture was not quota-gated: its shortfalls were reported, never refilled). Main-quota coverage checks are therefore
# reported as NOTE in amb runs only; integrity, duplicates across splits, pixels and every other check keep their rules.
AMB_NOTE_CHECKS = ("cc_projection", "cc_instances", "cc_combos", "cc_progress", "class_balance", "classes_seen")
AMB_RULING = "K2-AMB 2026-10-17 (fixed-plan amb coverage reported, not gated, as Kitchen 1 amb)"


# K3-SINK-1 (Arash 2026-10-17 ~21:15 "Approve the FloorPlan17 Sink exception."; Astra K3-SINK-r1 option A): a registered type that
# is STRUCTURALLY UNRENDERED in a scene (FloorPlan17's Sink: only its child SinkBasin is ever segmented; simulator sweep 120 views
# Sink 0x) is exempt from classes_seen, the coverage projections and the type / instance floors -- ONLY while the fail-closed scan
# finds no label of that type and no segmentation of that object anywhere in the store. The registered list is unchanged.
STRUCTURAL_EXCLUSIONS = {"FloorPlan17": ("Sink",)}
STRUCTURAL_RULING = "K3-SINK-1 2026-10-17 (Arash; structurally unrendered type exempt from coverage floors; fail-closed)"


def structural_scan(ds: str, scene: str) -> dict:
    """Fail-closed evidence for STRUCTURAL_EXCLUSIONS: any label of an excluded type, or any segmentation key of the excluded
    OBJECT itself ('<Type>|x|y|z', exactly 4 '|' parts -- its children such as '<Type>|x|y|z|SinkBasin' are other objects)."""
    types = STRUCTURAL_EXCLUSIONS.get(scene, ())
    hits = {"labels": 0, "seg": 0}
    if not types:
        return {"types": (), "hits": hits, "ok": False}
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if '"img"' not in line:
                continue
            r = json.loads(line)
            hits["labels"] += sum(1 for x in r.get("labels") or [] if x.get("type") in types)
            hits["seg"] += sum(1 for k in (r.get("seg_colors") or {}) if k.split("|")[0] in types and len(k.split("|")) == 4)
    return {"types": types, "hits": hits, "ok": hits["labels"] == 0 and hits["seg"] == 0}


def dup_identity(r: dict) -> str:
    """Split identity for duplicate detection: the sealed amb holdout (episodes 3300-3399) is its own split."""
    return "holdout" if 3300 <= int(r.get("episode", -1)) < 3400 else r["split"]


def layout_evidence(qdir: str, qstate: dict, amb: bool) -> tuple:
    """(provenance, qc_dir, batches) for layout_fingerprint: always the MAIN store's own QC evidence (K2-AMB r2 3)."""
    if not amb:
        return dict(qstate.get("skips") or {}), qdir, int(qstate.get("batches", 0))
    l_dir = os.path.join(os.path.dirname(qdir), "qc")
    try:
        mq = json.load(open(os.path.join(l_dir, "qc_state.json"), encoding="utf-8"))
    except Exception:
        mq = {}
    return dict(mq.get("skips") or {}), l_dir, int(mq.get("batches", -1))


def amb_ruling(out: list, active: bool) -> list:
    if not active:
        return out
    return [("NOTE", n, f"{AMB_RULING}; found {st}: {d}"[:300])
            if st in ("WARN", "FAIL") and (n in AMB_NOTE_CHECKS or n.endswith("_floors") or n.endswith("_counts"))
            else (st, n, d) for st, n, d in out]


def dup_ruling(scene: str, out: list) -> list:
    if scene not in DUP_RULING_SCENES and scene not in BLUR_NOTE_SCENES:
        return out
    res = []
    for st, n, d in out:
        if scene in DUP_RULING_SCENES and n in DUP_RULING_CHECKS and st in ("WARN", "FAIL"):
            res.append(("NOTE", n, f"{DUP_RULING}; found {st}: {d}"[:300]))
        elif scene in BLUR_NOTE_SCENES and n == "blur" and st == "WARN":
            res.append(("NOTE", n, f"{BLUR_RULING}; found {st}: {d}"[:300]))
        else:
            res.append((st, n, d))
    return res


def amd1_pairs(ds: str) -> dict:
    """Paired renders in <ds>/frames.jsonl: total, valid, and the K23-AMD-1 Blinds Open/Close pairs (excluded)."""
    n = valid = ex = ex_valid = 0
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for l in fh:
            p = json.loads(l).get("pair")
            if not p:
                continue
            n += 1
            v = bool(p.get("valid_evidence"))
            valid += v
            if str(p.get("oid") or "").startswith("Blinds|") and p.get("action") in ("OpenObject", "CloseObject"):
                ex += 1
                ex_valid += v
    return {"total": n, "valid": valid, "amd1_blinds": ex, "amd1_blinds_valid": ex_valid}
SKIP_RE = r"\] {scene}: ep\s+(\d+) \[train\] skipped: every train target already met"


# ======================================================================================================================
# Reading.
# ======================================================================================================================
def iter_episodes(ds: str):
    """Yield ``(episode_row, [frame rows])`` per episode in file order, without holding the whole store in memory.
    A frame row belongs to the episode record written right before it (the capture commits a record + its rows)."""
    cur, rows = None, []
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "episode_record" in r:
                if cur is not None:
                    yield cur, rows
                cur, rows = r, []
            elif not r.get("quarantined"):
                rows.append(r)
    if cur is not None:
        yield cur, rows


def load_rows(ds: str, episodes: set | None = None):
    eps, frames = [], []
    for e, rs in iter_episodes(ds):
        if episodes is None or e["episode_record"]["episode"] in episodes:
            eps.append(e)
            frames += [r for r in rs if r["episode"] == e["episode_record"]["episode"]]
    return eps, frames


def counts_only(ds: str, keep) -> dict:
    """Cumulative counts (no pixels, no rows kept): episodes / frames per split and instances per split x type."""
    out = {"episodes_by_split": {}, "frames_by_split": {}, "instances": {"train": {}, "val": {}, "test": {}},
           "episode_split": {}, "committed": [], "inst_counts": {}}
    out["records_seen"] = []
    for e, rs in iter_episodes(ds):
        er = e["episode_record"]
        out["records_seen"].append(er["episode"])
        if not keep(er["episode"]):
            continue
        out["committed"].append(er["episode"])
        out["episode_split"][er["episode"]] = er["split"]
        out["episodes_by_split"][er["split"]] = out["episodes_by_split"].get(er["split"], 0) + 1
        for r in rs:
            out["frames_by_split"][r["split"]] = out["frames_by_split"].get(r["split"], 0) + 1
            d = out["instances"].setdefault(r["split"], {})
            di = out["inst_counts"].setdefault(r["split"], {})
            for lab in r["labels"]:
                d[lab["type"]] = d.get(lab["type"], 0) + 1
                k = lab.get("inst") or lab.get("oid")
                di[k] = di.get(k, 0) + 1
    return out


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ======================================================================================================================
# Pure image metrics (numpy only).
# ======================================================================================================================
def gray(rgb):
    import numpy as np
    a = np.asarray(rgb, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def laplacian_var(g) -> float:
    """Variance of the 4-neighbour Laplacian of a grey image (the usual blur measure; low = blurred / flat)."""
    lap = (g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def exposure(g) -> tuple:
    """(mean luminance 0-255, share of clipped pixels: <= 5 or >= 250)."""
    return float(g.mean()), float(((g <= 5) | (g >= 250)).mean())


def thumb32(g):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.fromarray(g.astype("uint8")).resize((32, 32)), dtype=np.float32)


def mask_stats(seg, colour) -> tuple:
    """(pixel count, [x1, y1, x2, y2] or None) of one colour in an instance-segmentation image."""
    import numpy as np
    m = (seg[..., 0] == colour[0]) & (seg[..., 1] == colour[1]) & (seg[..., 2] == colour[2])
    n = int(m.sum())
    if not n:
        return 0, None
    ys, xs = np.nonzero(m)
    return n, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / ua if ua else 0.0


def label_problems(lab: dict, registry: set, cfg=kd.CAPTURE) -> list:
    """Metadata sanity of one stored label (no pixels)."""
    out = []
    if lab.get("type") not in registry:
        out.append("type_not_in_registry")
    if vocab.base_type(lab.get("orig_type") or "") != lab.get("type"):
        out.append("type_fold")
    b = lab.get("box") or []
    if len(b) != 4 or not (0 <= b[0] <= b[2] < FRAME and 0 <= b[1] <= b[3] < FRAME):
        out.append("box_outside_frame")
    else:
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        if area < cfg["min_box_area"]:
            out.append("box_too_small")
        if int(lab.get("px") or 0) > area:
            out.append("px_exceeds_box")
    if int(lab.get("px") or 0) < cfg["min_mask_px"]:
        out.append("mask_too_small")
    return out


KINDS = (("images", ".jpg"), ("depth", ".png"), ("seg", ".png"))


def expected_files(frames: list) -> set:
    """Every file the capture writes for these frame rows (save_frame + paired_render): relative paths."""
    exp = set()
    for r in frames:
        for kind, ext in KINDS:
            exp.add(f"{kind}/{r['split']}/{r['img']}{ext}")
        pr = r.get("pair")
        if pr:
            sufs = ["_noise"] + (["_flip"] if pr.get("ok") else [])
            for suf in sufs:
                for kind, ext in KINDS:
                    exp.add(f"pairs/{kind}/{r['split']}/{r['img']}{suf}{ext}")
    return exp


def episode_of(fname: str):
    try:
        return int(fname.split("_e")[1].split("_")[0])
    except (IndexError, ValueError):
        return None


def pairing(ds: str, frames: list, epset: set) -> dict:
    exp = expected_files(frames)
    missing = sorted(f for f in exp if not os.path.exists(os.path.join(ds, f)))
    orphans = []
    for sub in ("", "pairs/"):
        for kind, _ext in KINDS:
            for s in ("train", "val", "test"):
                d = os.path.join(ds, sub + kind, s)
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    if episode_of(f) in epset and f"{sub}{kind}/{s}/{f}" not in exp:
                        orphans.append(f"{sub}{kind}/{s}/{f}")
    return {"expected_files": len(exp), "missing_files": len(missing), "missing_examples": missing[:5],
            "orphans": len(orphans), "orphan_examples": sorted(orphans)[:5]}


# ======================================================================================================================
# Metrics for one set of episodes.
# ======================================================================================================================
def measure(ds: str, eps: list, frames: list, registry: list, stride: int = STRIDE, decode: bool = True,
            limits: dict | None = None) -> dict:
    """Every metric for one batch / window. ``limits`` = the per-frame cut-offs (blur, exposure) from the baseline;
    None in the baseline's first pass (the raw per-frame values are then returned for the percentile fit)."""
    import numpy as np
    from PIL import Image
    import build_kitchen_data as bkd
    reg = set(registry)
    train = [r for r in frames if r["split"] == "train"]
    out = {"episodes": sorted(e["episode_record"]["episode"] for e in eps),
           "episodes_by_split": {}, "frames_by_split": {}}
    for e in eps:
        s = e["episode_record"]["split"]
        out["episodes_by_split"][s] = out["episodes_by_split"].get(s, 0) + 1
    for r in frames:
        out["frames_by_split"][r["split"]] = out["frames_by_split"].get(r["split"], 0) + 1
    # ---- class balance (counts)
    per = {s: {} for s in ("train", "val", "test")}
    for r in frames:
        d = per.setdefault(r["split"], {})
        for lab in r["labels"]:
            d[lab["type"]] = d.get(lab["type"], 0) + 1
    out["instances"] = per
    # ---- metadata label sanity + mask size distribution (TRAIN labels)
    probs = {}
    pxs = []
    for r in frames:
        for lab in r["labels"]:
            for p in label_problems(lab, reg):
                probs[p] = probs.get(p, 0) + 1
            if lab.get("oid") not in (r.get("seg_colors") or {}):
                probs["no_seg_colour"] = probs.get("no_seg_colour", 0) + 1
    for r in train:
        pxs += [int(l["px"]) for l in r["labels"] if l["type"] != "Floor"]
    n_lab = sum(len(r["labels"]) for r in frames)
    out["labels"] = {"n": n_lab, "per_frame": n_lab / len(frames) if frames else 0.0, "problems": probs}
    if pxs:
        q = np.percentile(pxs, [5, 25, 50, 75, 95]).tolist()
        out["mask_px"] = {"n": len(pxs), "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
                          "share_small": sum(1 for v in pxs if v < SMALL_PX) / len(pxs)}
    else:
        out["mask_px"] = {"n": 0, "share_small": 0.0}
    out["held_share"] = (sum(1 for r in frames if r.get("kind") == "held") / len(frames)) if frames else 0.0
    # ---- pairing: rows <-> files, BOTH directions, main frames and paired renders (existence only; every split)
    out["pairing"] = pairing(ds, frames, set(out["episodes"]))
    # ---- exact duplicates (sha256 of image bytes, every split; hashes only)
    hashes = {}
    for r in frames:
        p = os.path.join(ds, "images", r["split"], r["img"] + ".jpg")
        if os.path.exists(p):
            # K2-AMB r2 5: the sealed amb holdout (episodes 3300-3399, stored under split 'test', protocol AMB_HOLDOUT) is
            # its own split for duplicate identity, so a holdout copy of an ordinary amb-test frame is cross-split
            ident = dup_identity(r)
            hashes.setdefault(sha_file(p), []).append((ident, r["img"]))
    dups = [v for v in hashes.values() if len(v) > 1]
    out["exact_dup"] = {"n_frames_dup": sum(len(v) - 1 for v in dups),
                        "cross_split": sum(1 for v in dups if len({s for s, _ in v}) > 1),
                        "share": (sum(len(v) - 1 for v in dups) / len(frames)) if frames else 0.0}
    out["_hashes"] = {h: v[0] for h, v in hashes.items()}
    # ---- pixels: TRAIN frames only, every stride-th frame of each episode
    sample = []
    by_ep = {}
    for r in train:
        by_ep.setdefault(r["episode"], []).append(r)
    for e, rs in sorted(by_ep.items()):
        sample += [(e, r) for i, r in enumerate(rs) if i % stride == 0]
    out["sample"] = {"stride": stride, "n": len(sample)}
    if not decode:
        return out
    lv, mean_l, clip, anomalies, nd, nd_tot = [], [], [], {}, 0, 0
    mism, checked = 0, 0
    thumbs = {}
    for e, r in sample:
        p = os.path.join(ds, "images", "train", r["img"] + ".jpg")
        sp = os.path.join(ds, "seg", "train", r["img"] + ".png")
        if not os.path.exists(p):
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        g = gray(rgb)
        lv.append(laplacian_var(g))
        m, c = exposure(g)
        mean_l.append(m)
        clip.append(c)
        an = bkd.frame_anomaly(rgb)
        if an:
            anomalies[an] = anomalies.get(an, 0) + 1
        t = thumb32(g)
        prev = thumbs.setdefault(e, [])
        nd_tot += 1
        if any(float(np.abs(t - u).mean()) < 6.0 for u in prev):
            nd += 1
        prev.append(t)
        if os.path.exists(sp):
            seg = np.asarray(Image.open(sp).convert("RGB"))
            cols = r.get("seg_colors") or {}
            for lab in r["labels"]:
                col = cols.get(lab["oid"])
                if not col:
                    continue
                checked += 1
                n, box = mask_stats(seg, col)
                if box is None or abs(n - int(lab["px"])) > max(2, 0.01 * int(lab["px"])) \
                        or box_iou(box, lab["box"]) < 0.95:
                    mism += 1
    out["pixels"] = {"n": len(lv), "anomalies": anomalies, "near_dup": nd, "near_dup_share": nd / nd_tot if nd_tot else 0.0,
                     "mask_checked": checked, "mask_mismatch": mism,
                     "mask_mismatch_share": mism / checked if checked else 0.0}
    if lv:
        out["pixels"].update(lapvar_p50=float(np.median(lv)), mean_lum_p50=float(np.median(mean_l)),
                             clip_p50=float(np.median(clip)))
    if limits is None:
        out["_raw"] = {"lapvar": lv, "mean_lum": mean_l, "clip": clip}
    else:
        n = len(lv) or 1
        out["pixels"]["blur_low_share"] = sum(1 for v in lv if v < limits["lapvar_low"]) / n
        out["pixels"]["exposure_out_share"] = sum(
            1 for m, c in zip(mean_l, clip) if m < limits["lum_low"] or m > limits["lum_high"]
            or c > limits["clip_high"]) / n
        out["pixels"]["extreme_frames"] = sum(1 for m, c in zip(mean_l, clip) if m < 10 or m > 245 or c > 0.5)
    return out


# ======================================================================================================================
# Baseline (Kitchen 1) and thresholds.
# ======================================================================================================================
def windows(plan_eps: list, done: set) -> list:
    ids = [e for e in plan_eps if e in done]
    return [set(ids[i:i + WINDOW]) for i in range(0, len(ids) - WINDOW + 1, WINDOW)]


def cmd_baseline(ds: str, out_path: str, stride: int, amb: bool = False) -> int:
    import numpy as np
    probe = json.load(open(os.path.join(ds, "probe.json"), encoding="utf-8"))
    scene = probe["scene"]
    main_ids = [e["episode"] for e in kd.episode_plan(scene, kd.CAPTURE["episodes"])]
    if amb:
        # K2-AMB (Astra K2-AMB-r1 4): the Kitchen 1 AMB store's committed amb-plan TRAIN episodes only (sealed holdout,
        # val and test never read); same measure(), window and percentile rules; incomplete last window omitted
        done = set(json.load(open(os.path.join(ds, "capture_state.json"), encoding="utf-8")).get("done_episodes") or [])
        main_ids = [e["episode"] for e in kd.amb_plan(scene) if e["split"] == "train" and e["episode"] in done]
    main_set = set(main_ids)
    t0 = time.time()
    per_w, raw = [], {"lapvar": [], "mean_lum": [], "clip": []}
    buf_e, buf_f = [], []
    seen, first_all, n_main = set(), None, 0
    reg = set(probe["registry"])
    for e, rs in iter_episodes(ds):
        eid = e["episode_record"]["episode"]
        if eid not in main_set:
            continue
        n_main += 1
        seen |= {l["type"] for r in rs if r["split"] == "train" for l in r["labels"]}
        if first_all is None and seen >= reg:
            first_all = n_main
        buf_e.append(e)
        buf_f += rs
        if len(buf_e) == WINDOW:
            m = measure(ds, buf_e, buf_f, probe["registry"], stride=stride)
            for k in raw:
                raw[k] += m["_raw"][k]
            m.pop("_raw")
            m.pop("_hashes")
            per_w.append(m)
            print(f"  window {len(per_w)} episodes {min(m['episodes'])}-{max(m['episodes'])}: "
                  f"{m['sample']['n']} frames decoded ({time.time() - t0:.0f} s)", flush=True)
            buf_e, buf_f = [], []
    wins = per_w
    lim = {"lapvar_low": float(np.percentile(raw["lapvar"], 1)),
           "lum_low": float(np.percentile(raw["mean_lum"], 0.5)),
           "lum_high": float(np.percentile(raw["mean_lum"], 99.5)),
           "clip_high": float(np.percentile(raw["clip"], 99.5))}
    # second pass over the cached raw values -> per-window shares at those cut-offs
    wstats = []
    pos = 0
    for m in per_w:
        n = m["pixels"]["n"]
        lv = raw["lapvar"][pos:pos + n]
        ml = raw["mean_lum"][pos:pos + n]
        cl = raw["clip"][pos:pos + n]
        pos += n
        nn = n or 1
        tr_eps = m["episodes_by_split"].get("train", 0) or 1
        wstats.append({
            "episodes": [min(m["episodes"]), max(m["episodes"])],
            "blur_low_share": sum(1 for v in lv if v < lim["lapvar_low"]) / nn,
            "exposure_out_share": sum(1 for a, c in zip(ml, cl) if a < lim["lum_low"] or a > lim["lum_high"]
                                      or c > lim["clip_high"]) / nn,
            "near_dup_share": m["pixels"]["near_dup_share"],
            "mask_mismatch_share": m["pixels"]["mask_mismatch_share"],
            "anomalies": sum(m["pixels"]["anomalies"].values()),
            "label_problems": sum(m["labels"]["problems"].values()),
            "exact_dup_share": m["exact_dup"]["share"],
            "cross_split_dup": m["exact_dup"]["cross_split"],
            "share_small": m["mask_px"]["share_small"], "mask_p50": m["mask_px"].get("p50"),
            "labels_per_frame": m["labels"]["per_frame"], "held_share": m["held_share"],
            "train_per_episode": {t: c / tr_eps for t, c in m["instances"]["train"].items()},
            "train_episodes": m["episodes_by_split"].get("train", 0)})

    def band(k):
        v = [w[k] for w in wstats if w[k] is not None]
        return {"min": min(v), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)), "max": max(v)}
    rate = {}
    for w in wstats:
        for t, v in w["train_per_episode"].items():
            rate.setdefault(t, []).append(v)
    res = {"what": ("K23 QC thresholds derived from the Kitchen 1 (FloorPlan1, development data) "
                    + ("AMB store, committed amb-plan TRAIN episodes" if amb else "main store") +
                    f", {len(wins)} windows of {WINDOW} consecutive episodes"),
           "amb": bool(amb), "source_episode_ids": list(main_ids),
           "frames_jsonl_sha256": sha_file(os.path.join(ds, "frames.jsonl")),
           "source": ds, "frames_jsonl_size": os.path.getsize(os.path.join(ds, "frames.jsonl")),
           "built": time.strftime("%Y-%m-%d %H:%M:%S"), "stride": stride, "window": WINDOW,
           "frame_limits": lim, "bands": {k: band(k) for k in (
               "blur_low_share", "exposure_out_share", "near_dup_share", "mask_mismatch_share", "anomalies",
               "label_problems", "exact_dup_share", "cross_split_dup", "share_small", "mask_p50",
               "labels_per_frame", "held_share")},
           "k1_train_labels_per_train_episode": {t: {"median": float(np.median(v)), "min": min(v), "max": max(v)}
                                                 for t, v in sorted(rate.items())},
           "k1_episodes_until_every_type_seen": first_all, "windows": wstats}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"baseline -> {out_path} (sha {sha_file(out_path)[:16]}); frame limits {lim}; every type seen after "
          f"{first_all} committed episodes")
    for k, b in res["bands"].items():
        print(f"  {k:20s} min {b['min']:.4f} p50 {b['p50']:.4f} p90 {b['p90']:.4f} max {b['max']:.4f}")
    return 0


# ======================================================================================================================
# Verdict for one K23 batch.
# ======================================================================================================================
def rule_upper(value, band, name, out):
    """Bad-is-high share: WARN above the Kitchen 1 worst window; FAIL above max(2 x worst, worst + .05)."""
    worst = band["max"]
    hard = max(2 * worst, worst + 0.05)
    st = "FAIL" if value > hard else ("WARN" if value > worst else "PASS")
    out.append((st, name, f"{value:.4f} (Kitchen 1 windows: p50 {band['p50']:.4f}, worst {worst:.4f}; FAIL > {hard:.4f})"))


def verdict(scene: str, batch: dict, cum: dict, th: dict, expected: dict, registry: list, extra: dict) -> list:
    out = []
    # ---- integrity first
    # the registry exactly AS STORED (order = class id, as stage_yolo writes it) and the probe's own scene
    ok_ids = registry == expected["classes"] and extra.get("probe_scene") == scene
    out.append(("PASS" if ok_ids else "FAIL", "class_ids",
                f"{len(registry)} classes = probe registry in stored order (id = index, as stage_yolo); expected "
                f"{expected['n_classes']}; identical incl. order: {registry == expected['classes']}; probe scene "
                f"{extra.get('probe_scene')!r}"))
    fe = extra.get("failed_episodes") or {}
    out.append(("FAIL" if fe else "PASS", "failed_episodes", f"episodes recorded failed by the capture: "
                f"{fe or 'none'} (an infrastructure repair; surfaced for a decision, never silently kept)"))
    lp = batch["labels"]["problems"]
    out.append(("FAIL" if lp else "PASS", "label_meta", f"label metadata problems: {lp or 'none'}"))
    pr = batch["pairing"]
    out.append(("FAIL" if pr["missing_files"] or pr["orphans"] else "PASS", "pairing",
                f"{pr['expected_files']} expected files (image / depth / seg + paired renders): {pr['missing_files']} "
                f"missing, {pr['orphans']} files without a row {pr['missing_examples'] + pr['orphan_examples']}"))
    xd = extra["cross_batch_dups"]
    cs = batch["exact_dup"]["cross_split"] + xd["cross_split"]
    out.append(("FAIL" if cs else "PASS", "dup_cross_split", f"exact duplicate images across splits: {cs} "
                f"(within batch {batch['exact_dup']['cross_split']}, vs earlier batches {xd['cross_split']})"))
    b = th["bands"]
    rule_upper(batch["exact_dup"]["share"] + xd["share"], b["exact_dup_share"], "dup_exact", out)
    for name, ok, detail in extra["integrity"]:
        out.append(("PASS" if ok else "FAIL", name, detail))
    # check_capture WARNs (Astra K23-PREP r2 7): expected-shortfall checks are NOTEs, every other WARN stops
    for x in extra.get("cc_warn") or []:
        if x["check"] in R1_NOTE_CHECKS:
            continue                                   # reported below, every batch (R1)
        # expected-shortfall checks (Astra K23-PREP r2 7: progress / projection / instances / combos) are NOTEs only
        # while the main capture is partial; at its end every check_capture WARN stops (15.4a: every other WARN stays
        # a stop -- the classification in force before R1; read narrowly, a shortfall that remains at the end stops)
        st = "NOTE" if x["check"] in CC_NOTE_CHECKS and not extra.get("main_complete") else "WARN"
        if x.get("status") == "FAIL":
            st = "FAIL"
        # K3-SINK-1 also exempts the INSTANCE floor (Astra K3-TOPUP-r1 2026-10-18: "correct cc_instances separately to honour the
        # exact exemption"): a check_capture instances WARN whose unseen instances (counted here, every split) are ALL of a
        # structurally exempt type, while the fail-closed scan is clean, is a NOTE. Any other unseen instance keeps the WARN.
        if x["check"] == "instances" and st == "WARN":
            sx0 = extra.get("structural") or {}
            ex0 = tuple(sx0.get("types") or ()) if sx0.get("ok") else ()
            ic0 = cum.get("inst_counts", {})
            miss = sorted({i for s in ("train", "val", "test") for i in extra.get("instances_expected", [])
                           if ic0.get(s, {}).get(i, 0) == 0})
            if ex0 and miss and all(i.split("_")[0] in ex0 for i in miss):
                out.append(("NOTE", "cc_instances", f"{STRUCTURAL_RULING}: only structurally exempt instances unseen {miss[:4]} "
                            f"(Astra K3-TOPUP-r1); check_capture: {str(x['detail'])[:120]}"))
                continue
        pa = extra.get("cc_pairs_amd1")
        if x["check"] == "pairs" and scene in AMD1_PAIRS_EXCL_SCENES:
            if not isinstance(pa, dict) or not pa.get("total"):
                out.append(("WARN", "cc_pairs", f"{str(x['detail'])[:160]} (AMD-1 exclusion could not be computed)"))
                continue
            den = pa["total"] - pa["amd1_blinds"]
            num = pa["valid"] - pa["amd1_blinds_valid"]
            r_ex = num / den if den else 0.0
            out.append(("WARN" if r_ex < PAIRS_MIN else "NOTE", "cc_pairs",
                        f"cc_pairs denominator (Arash 2026-10-16; K23-AMD-1 Blinds pairs excluded): {num}/{den} = "
                        f"{r_ex:.3f} (min {PAIRS_MIN}); registered all-pairs value {pa['valid']}/{pa['total']} = "
                        f"{pa['valid'] / pa['total']:.3f} ({pa['amd1_blinds']} AMD-1 Blinds pairs); check_capture: "
                        f"{str(x['detail'])[:80]}"))
            continue
        out.append((st, "cc_" + x["check"], str(x["detail"])[:220]))
    # R1 (Arash 2026-10-14, A28 15.4a): check_capture's `pace` is a NOTE -- its counts and streak in EVERY report, never a
    # stop. Narrow: only `pace`; a pace result that check_capture did not produce (or a FAIL) is not hidden (FAIL).
    pc = extra.get("cc_pace")
    if not isinstance(pc, dict) or pc.get("status") not in ("PASS", "WARN"):
        out.append(("FAIL", "cc_pace", f"check_capture produced no usable pace result ({str(pc)[:120]})"))
    else:
        out.append(("NOTE", "cc_pace",
                    f"R1 (A28 15.4a, reported, does not stop): check_capture pace {pc['status']}: "
                    f"{str(pc.get('detail'))[:200]}"))
    # ---- image quality (TRAIN sample)
    px = batch.get("pixels") or {}
    if not px.get("n"):
        out.append(("FAIL" if batch["episodes_by_split"].get("train") else "NOTE", "pixels",
                    "no TRAIN frame decoded in this batch"))
    else:
        rule_upper(px["blur_low_share"], b["blur_low_share"], "blur", out)
        if scene in AMD2_EXPOSURE_NOTE_SCENES:
            tmp = []
            rule_upper(px["exposure_out_share"], b["exposure_out_share"], "exposure", tmp)
            st0, _, det0 = tmp[0]
            out.append(("NOTE", "exposure", f"K23-AMD-2 (reported, does not stop; would be {st0}): {det0}"))
        else:
            rule_upper(px["exposure_out_share"], b["exposure_out_share"], "exposure", out)
        out.append(("FAIL" if px["extreme_frames"] else "PASS", "exposure_extreme",
                    f"{px['extreme_frames']} frames near black / white or > 50 % clipped"))
        na = sum(px["anomalies"].values())
        out.append(("FAIL" if na > b["anomalies"]["max"] else "PASS", "corruption",
                    f"capture corruption signatures {px['anomalies'] or 'none'} (Kitchen 1 worst window "
                    f"{b['anomalies']['max']})"))
        rule_upper(px["near_dup_share"], b["near_dup_share"], "near_dup", out)
        mm = px["mask_mismatch"]
        out.append(("FAIL" if mm else "PASS", "mask_vs_label",
                    f"{mm} of {px['mask_checked']} labels disagree with their stored instance mask (px > 1 % or box "
                    f"IoU < .95); Kitchen 1: 0 in every window -> any disagreement is an integrity stop"))
        out.append(("INFO", "pixel_medians", f"Laplacian var p50 {px.get('lapvar_p50', 0):.1f}, mean luminance p50 "
                    f"{px.get('mean_lum_p50', 0):.1f}, clipped p50 {px.get('clip_p50', 0):.4f}; "
                    f"{px['n']} TRAIN frames decoded, {px['mask_checked']} masks re-derived"))
    mp = batch["mask_px"]
    sb = b["share_small"]
    st = "WARN" if mp["n"] and not (sb["min"] - 0.05 <= mp["share_small"] <= sb["max"] + 0.05) else "PASS"
    out.append((st, "mask_sizes", f"TRAIN masks {mp['n']}: p5 {mp.get('p5', 0):.0f} p50 {mp.get('p50', 0):.0f} "
                f"p95 {mp.get('p95', 0):.0f} px; small (< {SMALL_PX} px) {mp['share_small']:.3f} (Kitchen 1 windows "
                f"{sb['min']:.3f}-{sb['max']:.3f})"))
    # ---- class balance (cumulative TRAIN, main protocol)
    sx = extra.get("structural") or {}
    excl = tuple(sx.get("types") or ()) if sx.get("ok") else ()
    if sx.get("types"):
        out.append(("NOTE" if sx.get("ok") else "FAIL", "structural_exclusion",
                    f"{STRUCTURAL_RULING}: {list(sx['types'])} "
                    + (f"exempt ({len(registry) - len(excl)}/{len(registry) - len(excl)} observable types gated; fail-closed scan: "
                       f"0 labels, 0 segmentations)" if sx.get("ok") else
                       f"VOID: the excluded type appeared ({sx.get('hits')}) -> full registry applies; STOP")))
    full_registry, registry = registry, [t for t in registry if t not in excl]
    tr = cum["instances"]["train"]
    n_tr = cum["episodes_by_split"].get("train", 0)
    n_all = sum(cum["episodes_by_split"].values())
    counts = [tr.get(t, 0) for t in registry]
    unseen = [t for t in registry if not tr.get(t)]
    st = "FAIL" if unseen and n_all >= MIN_EPS_FOR_UNSEEN else ("NOTE" if unseen else "PASS")
    out.append((st, "classes_seen", f"{len(registry) - len(unseen)}/{len(registry)} classes have TRAIN labels after "
                f"{n_all} episodes; never seen: {unseen or 'none'} (Kitchen 1: every type seen after "
                f"{th.get('k1_episodes_until_every_type_seen')} episodes; FAIL from {MIN_EPS_FOR_UNSEEN})"))
    if n_tr:
        proj = {t: tr.get(t, 0) * N_TRAIN_MAIN / n_tr for t in registry}
        under_gate = sorted((t for t in registry if proj[t] < GATE_TRAIN), key=lambda t: proj[t])
        under_aim = sorted((t for t in registry if proj[t] < AIM_TRAIN), key=lambda t: proj[t])
        st = "FAIL" if under_gate and n_all >= MIN_EPS_FOR_PROJECTION else ("NOTE" if under_gate or under_aim else "PASS")
        out.append((st, "class_balance",
                    f"TRAIN instances min {min(counts)} / median {statistics.median(counts):.0f} after {n_tr} train "
                    f"episodes; projected to {N_TRAIN_MAIN}: under the gate floor {GATE_TRAIN}: "
                    f"{[(t, round(proj[t])) for t in under_gate] or 'none'}; under the aim {AIM_TRAIN}: "
                    f"{[(t, round(proj[t])) for t in under_aim[:12]] or 'none'} (FAIL from {MIN_EPS_FOR_PROJECTION} "
                    f"episodes; registered response = the capped TRAIN top-up, nothing else)"))
    complete = bool(extra.get("main_complete"))
    for s in ("val", "test"):
        n_s = cum["episodes_by_split"].get(s, 0)
        if n_s:
            c = cum["instances"].get(s, {})
            tgt = kd.CAPTURE["gate_type_target"][s]
            itgt = kd.CAPTURE["gate_instance_target"][s]
            if complete:
                # end of the main capture: the registered held-out floors (the detection gate's), counted, not projected
                low = sorted(t for t in registry if c.get(t, 0) < tgt)
                ic = cum.get("inst_counts", {}).get(s, {})
                ilow = sorted(i for i in extra.get("instances_expected", []) if ic.get(i, 0) < itgt
                              and i.split("_")[0] not in excl)
                out.append(("FAIL" if low or ilow else "PASS", f"{s}_floors",
                            f"{s} at the end of the main capture: types under {tgt}: {low or 'none'}; instances under "
                            f"{itgt}: {ilow[:8] or 'none'} (a held-out shortfall cannot be topped up: STOP, A28 item 5)"))
            else:
                proj = {t: c.get(t, 0) * kd.CAPTURE["episodes"][s] / n_s for t in registry}
                low = sorted((t for t in registry if proj[t] < tgt), key=lambda t: proj[t])
                out.append(("NOTE" if low else "PASS", f"{s}_counts",
                            f"{s} instances (counts only) after {n_s} episodes; projected under {tgt}: "
                            f"{[(t, round(proj[t])) for t in low] or 'none'} (checked as a hard floor at the end of "
                            f"the main capture)"))
    return amb_ruling(topup_cc_ruling(dup_ruling(scene, out), bool(extra.get("topup_close_run"))),
                      bool(extra.get("amb_run")))


def inspection_due(state_path: str, episodes_split: dict) -> list:
    """Registered human sample (A28 item 4): frame 0 and frame 55 of every 8th COMMITTED episode, TRAIN only."""
    st = json.load(open(state_path, encoding="utf-8"))
    order = list(st.get("done_episodes") or [])
    due = []
    for i, e in enumerate(order):
        if (i + 1) % 8 == 0 and episodes_split.get(e) == "train":
            due.append(e)
    return due


def evidence_problems(qdir: str, qstate: dict, scene: str) -> list:
    """Why the QC state does NOT reconcile with the reports on disk (empty = it does). Astra K23-PREP r3 2: the
    'nothing new' answer must rest on reports that exist, are unchanged (sha256 recorded at write time), belong to this
    scene and batch number, cover every episode the state calls covered, and are PASS or validly resolved."""
    import re
    out = []
    n_b = qstate.get("batches", 0)
    covered = qstate.get("covered")
    if not (isinstance(n_b, int) and not isinstance(n_b, bool) and n_b >= 0) or not isinstance(covered, list):
        return ["qc_state.json: 'batches' / 'covered' unreadable"]
    files = {f for f in os.listdir(qdir) if re.fullmatch(r"batch_\d{3}\.json", f)}
    expect = {f"batch_{i:03d}.json" for i in range(1, n_b + 1)}
    if files - expect:
        out.append(f"report file(s) the QC state does not know: {sorted(files - expect)[:3]}")
    hashes = qstate.get("reports") or {}
    resolved, rprob = bound_resolutions(qdir, qstate)
    out += rprob
    seen = set()
    for i in range(1, n_b + 1):
        f = f"batch_{i:03d}.json"
        p = os.path.join(qdir, f)
        if f not in files:
            out.append(f"{f} missing")
            continue
        h = hashes.get(str(i))
        if not isinstance(h, str) or len(h) != 64:
            out.append(f"{f}: no sha256 recorded in qc_state.json")
        elif sha_file(p) != h:
            out.append(f"{f}: changed after it was written")
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception as exc:
            out.append(f"{f}: unreadable ({type(exc).__name__})")
            continue
        if r.get("scene") != scene or r.get("batch") != i:
            out.append(f"{f}: scene / batch {r.get('scene')} / {r.get('batch')} != {scene} / {i}")
        eps = r.get("episodes")
        if not isinstance(eps, list) or not all(isinstance(e, int) and not isinstance(e, bool) for e in eps):
            out.append(f"{f}: episode list unreadable")
            continue
        seen |= set(eps)
        cp = k23_decide.report_content_problems(r)        # r4 2: the CONTENTS, not only the file, are the evidence
        if cp:
            out.append(f"{f}: contents invalid: {cp[:3]}")
        if r.get("verdict") != "PASS" and i not in resolved:
            out.append(f"{f}: verdict {r.get('verdict')!r} and not resolved")
        if r.get("verdict") == "PASS" and i in resolved:
            out.append(f"{f}: a resolution is recorded for a PASS report (a resolution binds exactly one non-PASS batch)")
    miss = sorted(set(covered) - seen)
    if miss:
        out.append(f"episodes the QC state calls covered but no report checked: {miss[:5]}")
    return out


def bound_resolutions(qdir: str, qstate: dict) -> tuple:
    """({batch numbers validly resolved}, [problems]). r4 2: a recorded resolution counts only when it names its batch
    (an integer), its file is exactly qc/resolution_batch_NNN.json of THIS qc folder, still exists, matches the sha256
    recorded when it was accepted, and is (still) a valid person-written resolution with an Astra file. One resolution
    per batch; anything else is a problem (and resolves nothing)."""
    ok, out = set(), []
    for x in qstate.get("resolved") or []:
        if not isinstance(x, dict) or not (isinstance(x.get("batch"), int) and not isinstance(x.get("batch"), bool)):
            out.append(f"resolution record unreadable: {str(x)[:80]}")
            continue
        n = x["batch"]
        want = os.path.join(qdir, f"resolution_batch_{n:03d}.json")
        rp = x.get("resolution")
        if not isinstance(rp, str) or os.path.normcase(os.path.basename(rp)) != os.path.basename(want):
            out.append(f"resolution of batch {n}: names {rp!r}, not {os.path.basename(want)}")
            continue
        if not os.path.exists(want):
            out.append(f"resolution of batch {n}: {os.path.basename(want)} missing")
            continue
        h = x.get("resolution_sha256")
        if not isinstance(h, str) or sha_file(want) != h:
            out.append(f"resolution of batch {n}: changed after it was accepted (or no sha256 recorded)")
            continue
        why = k23_decide.valid_resolution(want)
        if why:
            out.append(f"resolution of batch {n}: not valid ({why[:2]})")
            continue
        if n in ok:
            out.append(f"resolution of batch {n}: recorded twice")
            continue
        ok.add(n)
    return ok, out


def skip_provenance(ds: str, qstate: dict, done: set, records: set, covered: set, log_text: str, plan_scene: str,
                    scene: str, qdir: str | None = None) -> tuple:
    """(new provenance entries, problems) for the 'done' episodes WITHOUT a record (r4 2). An earlier quota skip is
    proven only by the provenance this QC recorded WHEN IT FIRST SAW IT; "the quota is met now" does not prove that an
    earlier missing record was a skip (a captured record could have vanished). First sight: the episode is not one an
    earlier report checked (that would be a vanished record), the capture logged the exact quota-skip line for it in
    this scene, and the stored quota demonstrates it (layout_fingerprint.quota_skip_unexplained). Later sightings
    must find that entry. A covered episode that lost its record, or a provenance entry whose episode now has a record
    or is no longer 'done', is a problem."""
    import re
    import layout_fingerprint as lf
    prov = qstate.get("skips") or {}
    if not isinstance(prov, dict):
        return {}, ["qc_state.json: 'skips' unreadable"]
    out, new = [], {}
    logged = {int(m) for m in re.findall(SKIP_RE.format(scene=re.escape(scene)), log_text or "")}
    st = json.load(open(os.path.join(ds, "capture_state.json"), encoding="utf-8"))
    missing = sorted(done - records)
    und = lf.quota_skip_unexplained(st, missing, plan_scene) if missing else {}
    try:
        qsha = hashlib.sha256(json.dumps(st.get("quota"), sort_keys=True).encode()).hexdigest()
    except Exception:
        qsha = None
    for e in missing:
        if e in und:
            out.append(f"episode {e} 'done' without a record: {und[e]}")
        elif str(e) in prov:
            continue                               # validated (structure + its log line still present) below
        elif e in covered:
            out.append(f"episode {e} was checked by an earlier report but its record is gone")
        elif e not in logged:
            out.append(f"episode {e} 'done' without a record, no provenance and no quota-skip log line in capture.log")
        else:
            # r7: the caller RETAINS this quota as qc/skip_evidence_<ep>.json (k23_decide.write_skip_evidence)
            new[str(e)] = {"episode": e, "first_seen_batch": int(qstate.get("batches", 0)) + 1,
                           "quota": st.get("quota"), "quota_sha256": qsha}
    for k in prov:
        bad_ = k23_decide.skip_entry_problems(k, prov[k], scene, log_text, qdir=qdir,     # r5 2 + r7: every entry
                                              batches=qstate.get("batches", 0))         # + its retained evidence
        if bad_:
            out += bad_
            continue
        e = int(k)
        if e in records:
            out.append(f"episode {e} has skip provenance but now has a record")
        elif e not in done:
            out.append(f"episode {e} has skip provenance but is no longer 'done'")
    vanished = sorted(covered - records)
    if vanished:
        out.append(f"episode(s) an earlier report checked no longer have a record: {vanished[:5]}")
    return new, out


THRESHOLDS_AMB = os.path.join(HERE, "qc_thresholds_k1_amb.json")


def cmd_batch(a) -> int:
    ds = os.path.join(a.root, a.scene)
    probe = json.load(open(os.path.join(ds, "probe.json"), encoding="utf-8"))
    if getattr(a, "amb", False):
        if a.thresholds == THRESHOLDS:
            a.thresholds = THRESHOLDS_AMB
        if not os.path.basename(os.path.normpath(a.root)).endswith("_amb"):
            print(f"{a.scene}: --amb needs the amb store root (got {a.root}) -> STOP (could not run)")
            return 2
        a.out_dir = a.out_dir or os.path.join(SIM, "kitchen_v3", "k23", a.scene, "qc_amb")
    th = json.load(open(a.thresholds, encoding="utf-8"))
    import k23_classes
    expected = json.load(open(k23_classes.EXPECTED, encoding="utf-8"))["scenes"][a.scene]
    qdir = a.out_dir or os.path.join(SIM, "kitchen_v3", "k23", a.scene, "qc")
    os.makedirs(qdir, exist_ok=True)
    qstate_p = os.path.join(qdir, "qc_state.json")
    qstate = json.load(open(qstate_p, encoding="utf-8")) if os.path.exists(qstate_p) else \
        {"covered": [], "hashes": {}, "batches": 0}
    # an unresolved FAIL / WARN of an earlier batch blocks every later QC run (and so every later batch) until a
    # person writes qc/resolution_batch_NNN.json (who, when, finding, decision) -- Astra K23-PREP 4
    op = qstate.get("open")
    if op and not a.only_episodes:
        res = os.path.join(qdir, f"resolution_batch_{op['batch']:03d}.json")
        import k23_decide
        why = k23_decide.valid_resolution(res) if os.path.exists(res) else [f"no {res}"]
        if why:
            print(f"{a.scene}: batch {op['batch']} ended {op['verdict']} and is UNRESOLVED ({why}) -> STOP")
            return EXIT_UNRESOLVED
        qstate.setdefault("resolved", []).append(dict(op, resolution=res, resolution_sha256=sha_file(res)))
        qstate["open"] = None
    main_ids = {e["episode"] for e in kd.episode_plan(a.dry_scene_alias or a.scene, kd.CAPTURE["episodes"])}
    if a.amb:                                             # K2-AMB: this store's plan is the amb plan (incl. sealed holdout)
        main_ids = {e["episode"] for e in kd.amb_plan(a.dry_scene_alias or a.scene)}
    # evidence, not the cached list (Astra K23-PREP r3 2): every earlier report must exist, match its recorded hash,
    # be of this scene, and together cover `covered`; every non-PASS one must be resolved; capture_state must agree with
    # the episode records (a 'done' episode without a record = a DEMONSTRATED quota skip only)
    st_cap0 = json.load(open(os.path.join(ds, "capture_state.json"), encoding="utf-8"))
    done0 = set(st_cap0.get("done_episodes") or [])
    keep = (lambda e: e in done0) if a.all_protocols else (lambda e: e in main_ids and e in done0)
    cum = counts_only(ds, keep)
    new_skips = {}
    if not a.only_episodes:
        prob = evidence_problems(qdir, qstate, a.scene)
        # r4 2: per-episode skip provenance, recorded at first sight and reconciled on every later run
        cap_log = a.capture_log or os.path.join(os.path.dirname(qdir), "capture.log")
        try:
            with open(cap_log, encoding="utf-8", errors="replace") as fh:
                log_text = fh.read()
        except OSError:
            log_text = ""
        new_skips, sprob = skip_provenance(ds, qstate, done0, set(cum["records_seen"]),
                                           set(qstate.get("covered") or []), log_text, a.dry_scene_alias or a.scene,
                                           a.scene, qdir=qdir)
        prob += sprob
        if prob:
            print(f"{a.scene}: QC EVIDENCE DOES NOT RECONCILE -> STOP (could not run): {prob[:5]}")
            return 2
        # every 'done' episode is accounted for: checked by a report, newly committed (checked now), or a proven skip
        if new_skips:
            import k23_decide
            new_skips = {k: k23_decide.write_skip_evidence(qdir, a.scene, int(k), v["quota"], v["first_seen_batch"])
                         for k, v in new_skips.items()}
            bad_ev = [p_ for k, v in new_skips.items() for p_ in k23_decide.skip_entry_problems(
                k, v, a.scene, log_text, qdir=qdir, batches=qstate.get("batches", 0))]
            if bad_ev:                                    # e.g. a stale evidence file of another quota: never adopted
                print(f"{a.scene}: QC EVIDENCE DOES NOT RECONCILE -> STOP (could not run): {bad_ev[:5]}")
                return 2
            qstate.setdefault("skips", {}).update(new_skips)
            with open(qstate_p + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(qstate, fh)
            os.replace(qstate_p + ".tmp", qstate_p)
            print(f"{a.scene}: quota-skip provenance recorded for {sorted(int(k) for k in new_skips)} (capture log line "
                  f"+ stored quota; first sight)")
    uncommitted = sorted(set(cum["records_seen"]) - done0)
    if uncommitted:
        print(f"{a.scene}: {len(uncommitted)} record(s) not in capture_state (uncommitted; the capture's resume removes "
              f"them) are NOT checked: {uncommitted[:5]}")
    committed = cum["committed"]
    covered = set(qstate["covered"])
    new = [e for e in committed if e not in covered]
    if not new and not a.recheck:
        print(f"{a.scene}: no new committed episode since the last QC report -> nothing to check")
        return EXIT_NOTHING_NEW
    nset = set(new) if new else set(committed[-WINDOW:])
    if a.only_episodes:                                   # dry runs / re-checks of a named range only
        lo, hi = (int(x) for x in a.only_episodes.split("-"))
        nset = {e for e in committed if lo <= e <= hi}
    b_eps, b_frs = load_rows(ds, nset)
    batch = measure(ds, b_eps, b_frs, probe["registry"], stride=th["stride"], limits=th["frame_limits"])
    # exact duplicates against earlier batches (hashes only)
    old = qstate["hashes"]
    cross = {"n": 0, "cross_split": 0}
    split_of = {(r["split"], r["img"]) for r in b_frs}
    for h, (s, img) in batch["_hashes"].items():
        if h in old and tuple(old[h]) not in split_of:
            cross["n"] += 1
            cross["cross_split"] += int(old[h][0] != s)
    cross["share"] = cross["n"] / len(b_frs) if b_frs else 0.0
    # integrity: check_capture (scene + root explicit), layout, Kitchen 1 unchanged
    integ = []
    import check_capture
    rep = check_capture.check(ds, kd.CAPTURE_AMB if a.amb else kd.CAPTURE)
    fails = [f"{x['check']}: {x['detail'][:120]}" for x in rep["results"] if x["status"] == "FAIL"
             and not (a.scene in DUP_RULING_SCENES and x["check"] == "near_duplicates")
             and not (getattr(a, "amb", False) and x["check"] in CC_NOTE_CHECKS)]   # K2-AMB r2 2: reported as cc_<check>
    cc_amb_cov = [x for x in rep["results"] if x["status"] == "FAIL" and getattr(a, "amb", False)
                  and x["check"] in CC_NOTE_CHECKS]
    cc_dup_fail = [x for x in rep["results"] if x["status"] == "FAIL" and x["check"] == "near_duplicates"
                   and a.scene in DUP_RULING_SCENES]           # reported as cc_near_duplicates (ruling NOTE) instead
    integ.append(("check_capture", not fails, f"check_capture({ds}): {'no FAIL' if not fails else fails}"))
    import layout_fingerprint as lf
    # K2-AMB r2 3: the layout check reconciles the MAIN store's quota skips; in an amb run it gets the main store's own
    # QC evidence (qc/qc_state.json), never the amb store's (amb skips are not exempt: any amb 'done' without a record stops)
    l_prov, l_dir, l_bat = layout_evidence(qdir, qstate, getattr(a, "amb", False))
    lrep = lf.report(a.scene, os.path.dirname(os.path.normpath(a.root)), dry_alias=a.dry_scene_alias,
                     provenance=l_prov, qc_dir=l_dir, qc_batches=l_bat)
    integ.append(("layout", lrep["ok"], lf.summary(lrep)))
    import k1_unchanged
    try:
        base = json.load(open(k1_unchanged.BASELINE, encoding="utf-8"))
        bad, n = k1_unchanged.manifest_check()
        now = k1_unchanged.snapshot(True)
        d = []
        for k in ("seeds", "weights", "reports", "data"):
            d += k1_unchanged._diff(base[k], now[k], k)
        integ.append(("kitchen1_unchanged", not bad and not d,
                      f"manifest {n} files changed {bad or 'none'}; seeds/weights/reports/stores "
                      f"{'identical' if not d else d[:4]}"))
    except Exception as exc:                                  # a check that cannot run is a FAIL, never a PASS
        integ.append(("kitchen1_unchanged", False, f"could not run: {exc!r}"))
    st_cap = json.load(open(os.path.join(ds, "capture_state.json"), encoding="utf-8"))
    finished = set(st_cap.get("done_episodes") or []) | {int(k) for k in (st_cap.get("failed_episodes") or {})}
    lines = verdict(a.scene, batch, cum, th, expected, list(probe["registry"]),
                    {"cross_batch_dups": cross, "integrity": integ, "probe_scene": probe.get("scene"),
                     "failed_episodes": st_cap.get("failed_episodes") or {},
                     "cc_warn": [x for x in rep["results"] if x["status"] == "WARN"] + cc_dup_fail + cc_amb_cov,
                     "cc_pace": next((x for x in rep["results"] if x.get("check") == "pace"), None),
                     "cc_pairs_amd1": amd1_pairs(ds),
                     "main_complete": main_ids <= finished, "structural": structural_scan(ds, a.scene), "topup_close_run": bool(a.all_protocols), "amb_run": bool(a.amb), "instances_expected": list(probe.get("instances") or [])})
    statuses = {st for st, _, _ in lines}
    verdict_s = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    ok = verdict_s == "PASS"
    due = [e for e in inspection_due(os.path.join(ds, "capture_state.json"), cum["episode_split"]) if e in nset]
    qstate["batches"] += 1
    n_b = qstate["batches"]
    k1_rate = th["k1_train_labels_per_train_episode"]
    n_btr = batch["episodes_by_split"].get("train", 0)
    rep_out = {"scene": a.scene, "batch": n_b, "checked": time.strftime("%Y-%m-%d %H:%M:%S"),
               "verdict": verdict_s, "episodes": sorted(nset), "only_episodes": a.only_episodes, "amb": bool(a.amb),
               "amb_holdout_episodes": sorted(e for e in nset if 3300 <= e < 3400) if a.amb else [],
               "amb_holdout_frames": sum(1 for r in b_frs if 3300 <= int(r.get("episode", -1)) < 3400) if a.amb else 0,
               "amb_test_frames_excl_holdout": (sum(1 for r in b_frs if r["split"] == "test"
                                                    and not 3300 <= int(r.get("episode", -1)) < 3400) if a.amb else None),
               "episodes_by_split": batch["episodes_by_split"], "frames_by_split": batch["frames_by_split"],
               "lines": [{"status": s, "check": n, "detail": d} for s, n, d in lines],
               "train_instances_batch": batch["instances"]["train"],
               "train_instances_cumulative": cum["instances"]["train"],
               "expected_batch_from_k1": {t: round(k1_rate[t]["median"] * n_btr, 1) for t in sorted(probe["registry"])
                                          if t in k1_rate},
               "val_test_instances_cumulative": {s: cum["instances"].get(s, {}) for s in ("val", "test")},
               "mask_px": batch["mask_px"], "pixels": batch.get("pixels"), "pairing": batch["pairing"],
               "exact_dup": batch["exact_dup"], "exact_dup_vs_earlier": cross,
               "inspection_due_train_frames": [f"{a.scene}_e{e:03d}_{f:04d}" for e in due for f in (0, 55)],
               "thresholds": {"file": a.thresholds, "sha256": sha_file(a.thresholds)},
               "code_sha256": sha_file(os.path.abspath(__file__))}
    path = os.path.join(qdir, f"batch_{n_b:03d}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rep_out, fh, indent=1)
    os.replace(path + ".tmp", path)
    qstate.setdefault("reports", {})[str(n_b)] = sha_file(path)          # the evidence evidence_problems() checks
    qstate["covered"] = sorted(set(qstate["covered"]) | nset)
    if verdict_s != "PASS" and not a.only_episodes:
        qstate["open"] = {"batch": n_b, "verdict": verdict_s, "report": path}
    for h, v in batch["_hashes"].items():
        qstate["hashes"].setdefault(h, list(v))
    with open(qstate_p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(qstate, fh)
    os.replace(qstate_p + ".tmp", qstate_p)
    txt = [f"K23 QC {a.scene} batch {n_b} ({time.strftime('%H:%M')}): episodes {sorted(nset)} "
           f"{batch['episodes_by_split']} -> {verdict_s}"
           + ("" if ok else " (STOP: a person reads this, consults Astra, and writes qc/resolution_batch_NNN.json)")]
    txt += [f"  {s:4s} {n:18s} {d}" for s, n, d in lines]
    if due:
        txt.append(f"  inspection due (registered TRAIN sample, log every view): {rep_out['inspection_due_train_frames']}")
    txt.append(f"  report {path}")
    with open(os.path.join(qdir, "summary.log"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    return {"PASS": 0, "FAIL": 1, "WARN": EXIT_WARN}[verdict_s]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--ds", required=True, help="the Kitchen 1 main store (read-only)")
    b.add_argument("--out", default=THRESHOLDS)
    b.add_argument("--stride", type=int, default=STRIDE)
    b.add_argument("--amb", action="store_true", help="K2-AMB: baseline over the Kitchen 1 amb store's TRAIN episodes")
    c = sub.add_parser("batch")
    c.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    c.add_argument("--root", required=True, help="e.g. /home/arash/thor-data/kitchen_v3 (always explicit)")
    c.add_argument("--thresholds", default=THRESHOLDS)
    c.add_argument("--out-dir", default=None)
    c.add_argument("--recheck", action="store_true", help="re-check the last window when nothing is new")
    c.add_argument("--only-episodes", default=None, help="lo-hi: check exactly these committed episodes")
    c.add_argument("--dry-scene-alias", default=None, help="dry run only: the store's real scene (e.g. FloorPlan1)")
    c.add_argument("--all-protocols", action="store_true", help="include top-up / close episodes (later stages)")
    c.add_argument("--amb", action="store_true", help="K2-AMB: the amb store (--root = the amb root), amb plan, amb "
                                                       "thresholds (qc_thresholds_k1_amb.json), main-quota checks NOTE")
    c.add_argument("--capture-log", default=None, help="the capture log with the quota-skip lines (default: "
                                                       "<out-dir>/../capture.log)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "baseline":
            return cmd_baseline(a.ds, a.out, a.stride, amb=a.amb)
        return cmd_batch(a)
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"K23 QC could not run: {exc!r} -> STOP (a missing QC result is never a PASS)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
