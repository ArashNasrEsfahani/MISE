"""Score a trained per-scene detector against a held-out split, and say **which class** is failing and **why**.

"The models are dogshit at detecting" is not a number, and nothing that is not a number can be improved on
purpose. This module turns that complaint into a per-class ledger: precision / recall / AP50 / AP50-95 for every
class in the scene's vocabulary, printed **worst-first**, with the two columns that usually explain the failure
sitting right next to the score:

  * **how many training frames that class actually got.** A class the scan photographed four times has not been
    learned, and no change of backbone or recipe will fix it — the fix is another scan. The report prints the
    Spearman correlation between per-class AP50 and per-class training-frame count, which is the whole
    "is this a model problem or a data problem?" question reduced to one signed number.
  * **whether the two halves of a STATE pair are being told apart.** Our classes are ``<objectType>_<state>``
    (:mod:`perception`), so a detector that finds every fridge but calls the open one closed scores a respectable
    mAP and is *worthless to this thesis* — the agent's whole reason for looking is to learn the state. The
    state-pair section reads the confusion matrix and reports, per object family, what fraction of the correctly
    *recognised* instances got the *state* wrong, plus which direction it errs in.

It also counts the classes it could **not** score at all — zero held-out instances, or zero training frames —
instead of reporting them as 0.00 and letting them drag the mean down anonymously. A mAP averaged over the 41
classes that appear in val is a different number from one averaged over all 53 declared classes, and the header
says which it is.

Failure modes are loud on purpose. Missing weights, a missing dataset, an empty split, or a weights/data.yaml
class-list mismatch (which silently misattributes *every* per-class score, because the ids shift) raise
:class:`BenchError` with the path and the command that would fix it. This benchmark never prints zeros for
"I could not measure that".

Scoring the detector needs ultralytics + the GPU, so it runs in WSL::

    # stop the console first if the card is tight -- one CUDA consumer at a time
    wsl -d Ubuntu-22.04 --cd "/mnt/d/Personal/Master's Project" -- \\
        /home/arash/thor-env/bin/python sim/bench_detect.py --scene FloorPlan1 --top 25

Half the diagnosis is a property of the DATASET, not the model — which classes were starved, which were never
photographed, which state half the scan never produced, and whether the held-out split is near-duplicates of
the training frames. ``--counts-only`` reports exactly that half, needs no GPU, no ultralytics and no trained
weights, and takes about a second on Windows. Run it *before* paying for a train::

    python sim/bench_detect.py --scene FloorPlan1 --counts-only    # from the repo root, Windows python
    python sim/_bench_detect_selftest.py                           # headless gate, no torch

``ultralytics``/``torch`` are imported **lazily inside** :func:`run_val`, so ``import bench_detect`` and every
pure helper in this file work on bare Windows python (mirrors ``perception.py`` / ``gen_dataset.py``). The
arithmetic — aggregation, ranking, state-pair confusion, the confusion-matrix orientation inference — is pure
and covered by ``sim/_bench_detect_selftest.py`` with synthetic inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import perception                                                     # noqa: E402  (stdlib-only at import)

#: Below this many training frames a class is called *starved*. Detection fine-tuning on a frozen indoor scene
#: needs a couple of dozen views per class before the head has seen the object at more than one scale/angle;
#: under that, per-class AP is dominated by which handful of frames happened to be captured. It is a reporting
#: threshold, not a claim about a cliff — the Spearman figure is the quantitative version of the same question.
STARVED_FRAMES = 20

#: ultralytics `val` defaults. conf=0.001 is what makes mAP a proper area-under-PR rather than a single-threshold
#: score; iou=0.7 is the NMS IoU (unrelated to the 0.50:0.95 matching thresholds mAP sweeps).
#:
#: NB the CONFUSION MATRIX is not built at this threshold: ultralytics' `ConfusionMatrix` substitutes 0.25 when
#: conf is None or 0.001, so the state-pair section below is measured at the same 0.25 that
#: `perception.Detector.detect` uses in deployment. That is the number we want there — raising `--conf` moves
#: both the mAP columns and the confusion matrix, so change it only deliberately.
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class BenchError(RuntimeError):
    """A precondition this benchmark refuses to paper over — missing weights, dataset, split, or class list."""


# ----------------------------------------------------------------------------------------------------------------------
# data.yaml — a deliberately small parser (no pyyaml dependency; the file is written by build_data_yaml).
# ----------------------------------------------------------------------------------------------------------------------
def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def parse_data_yaml(path: str) -> dict:
    """Read an ultralytics ``data.yaml`` into ``{'path', 'splits': {...}, 'names': {id: name}, 'nc'}``.

    Handles the block form :func:`gen_dataset.build_data_yaml` writes (``names:`` then ``  0: Apple_whole``) plus
    the inline list/dict forms ultralytics also accepts, so a hand-edited file still parses. Raises
    :class:`BenchError` rather than returning an empty class list — a benchmark that silently scores against
    zero classes is worse than one that stops.
    """
    if not os.path.isfile(path):
        raise BenchError(f"data.yaml not found at {path!r}. Scan the scene first "
                         f"(webapp 'scan & learn', or sim/_phase4_fullscan.py) so the dataset exists.")

    cfg = {"path": None, "splits": {}, "names": {}, "nc": None}
    in_names = False
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if in_names and line[:1].isspace():
                key, sep, val = line.strip().partition(":")
                if sep:
                    try:
                        cfg["names"][int(key.strip().lstrip("-").strip())] = _unquote(val)
                    except ValueError:
                        pass                       # a sequence item ('- Apple') has no id; handled below
                elif line.strip().startswith("- "):
                    cfg["names"][len(cfg["names"])] = _unquote(line.strip()[2:])
                continue
            in_names = False
            key, sep, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if not sep:
                continue
            if key == "names":
                if not val:
                    in_names = True
                elif val.startswith("[") or val.startswith("{"):
                    body = val.strip("[]{} ")
                    for i, item in enumerate(p for p in body.split(",") if p.strip()):
                        k, s, v = item.partition(":")
                        if s:
                            try:
                                cfg["names"][int(k.strip())] = _unquote(v)
                                continue
                            except ValueError:
                                pass
                        cfg["names"][i] = _unquote(item)
            elif key == "path":
                cfg["path"] = _unquote(val)
            elif key in ("train", "val", "test"):
                cfg["splits"][key] = _unquote(val)
            elif key == "nc":
                try:
                    cfg["nc"] = int(val)
                except ValueError:
                    pass

    if not cfg["names"]:
        raise BenchError(f"{path!r} declares no classes (`names:` missing or unparseable). "
                         f"Re-run gen_dataset.build_data_yaml() for this scene.")
    if cfg["nc"] is None:
        cfg["nc"] = len(cfg["names"])
    return cfg


def split_dirs(dataset_dir: str, cfg: dict, split: str):
    """``(images_dir, labels_dir)`` for ``split``, resolved against the dataset ON DISK.

    Deliberately ignores the yaml's own ``path:`` for counting: that field records the absolute path of the
    machine that did the scan (``/mnt/d/...`` under WSL), so a report run from Windows — or from a copied
    dataset — would count nothing. ultralytics still resolves images through ``path:`` when it loads the set,
    which is why :func:`effective_data_yaml` rewrites that field into a temp copy before val runs.

    The labels directory follows ultralytics' own rule: the last ``images`` path component becomes ``labels``.
    """
    rel = (cfg.get("splits") or {}).get(split)
    if not rel:
        have = ", ".join(sorted((cfg.get("splits") or {}))) or "(none)"
        raise BenchError(f"data.yaml declares no {split!r} split (it has: {have}). "
                         f"Use --split with one of those, or re-scan so a held-out split exists.")
    images = rel if os.path.isabs(rel) else os.path.join(dataset_dir, rel)
    parts = os.path.normpath(images).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    labels = os.sep.join(parts)
    return images, labels


def effective_data_yaml(dataset_dir: str, cfg: dict, out_path: str) -> str:
    """Write a copy of ``cfg`` at ``out_path`` with ``path:`` repointed at ``dataset_dir``; returns ``out_path``.

    The shipped ``data.yaml`` hard-codes the scanning machine's absolute path, and ultralytics resolves every
    image through it — so a dataset that has been moved (or is being read from the other side of the WSL mount)
    fails inside the dataloader with a confusing "no labels found". Writing a corrected copy into a temp
    directory fixes that without touching the research data under ``sim/datasets/``.
    """
    lines = [f"path: {os.path.abspath(dataset_dir)}"]
    for key in ("train", "val", "test"):
        if cfg.get("splits", {}).get(key):
            lines.append(f"{key}: {cfg['splits'][key]}")
    lines += ["", f"nc: {cfg['nc']}", "names:"]
    for i in sorted(cfg["names"]):
        lines.append(f"  {i}: {cfg['names'][i]}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


# ----------------------------------------------------------------------------------------------------------------------
# Ground truth on disk — per-class frame and instance counts, straight out of the YOLO label files.
# ----------------------------------------------------------------------------------------------------------------------
def count_labels(labels_dir: str, n_classes=None) -> dict:
    """Per-class counts for one split, or ``None`` when ``labels_dir`` does not exist.

    Returns ``{'instances': {id: n}, 'frames': {id: n}, 'n_files', 'n_empty', 'n_instances', 'bad_lines',
    'unknown_ids'}``. *instances* counts boxes; *frames* counts images containing at least one box of the class —
    the second is the honest measure of how much a class was photographed, because twelve boxes spread over
    twelve viewpoints teach far more than twelve boxes in one frame.
    """
    if not os.path.isdir(labels_dir):
        return None
    instances, frames = {}, {}
    n_files = n_empty = bad = 0
    unknown = set()
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        n_files += 1
        seen = set()
        with open(os.path.join(labels_dir, fn), "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    bad += 1
                    continue
                if n_classes is not None and not (0 <= cid < int(n_classes)):
                    unknown.add(cid)
                    bad += 1
                    continue
                instances[cid] = instances.get(cid, 0) + 1
                seen.add(cid)
        if seen:
            for cid in seen:
                frames[cid] = frames.get(cid, 0) + 1
        else:
            n_empty += 1
    return {"instances": instances, "frames": frames, "n_files": n_files, "n_empty": n_empty,
            "n_instances": sum(instances.values()), "bad_lines": bad, "unknown_ids": sorted(unknown)}


def count_images(images_dir: str) -> int:
    """Number of image files directly under ``images_dir`` (0 if it is missing)."""
    if not os.path.isdir(images_dir):
        return 0
    return sum(1 for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS))


def frame_indices(images_dir: str) -> list:
    """Capture indices parsed out of ``<scene>_<index>_<hash>.jpg`` filenames, sorted; ``[]`` if unparseable.

    :func:`gen_dataset.capture_frame` prefixes each filename with the capture ordinal, which is the only record
    on disk of *when in the trajectory* a frame was taken — and that is what :func:`split_adjacency` needs.
    """
    out = []
    if not os.path.isdir(images_dir):
        return out
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        for part in os.path.splitext(fn)[0].split("_"):
            if part.isdigit():
                out.append(int(part))
                break
    return sorted(out)


def split_adjacency(train_idx, val_idx) -> dict:
    """How often a held-out frame sits directly beside a training frame in the capture order.

    A *heuristic leakage probe*, not a proof. A split taken 1-in-5 by FRAME puts near-duplicates of every val
    image into train (the agent moved a few degrees between them), so val mAP measures memorisation rather than
    generalisation and every number in this report reads high. A split taken by capture SITE does not. The
    fraction below is ~1.0 for the former and near 0 for the latter, so it tells the reader which kind of number
    they are looking at. Returns ``{'n_val', 'adjacent', 'fraction', 'verdict'}``; ``fraction`` is ``None`` when
    the filenames carry no capture index to compare.
    """
    val_idx = sorted(val_idx or [])
    train = set(train_idx or [])
    if not val_idx or not train:
        return {"n_val": len(val_idx), "adjacent": 0, "fraction": None, "verdict": "unknown (no capture indices)"}
    adjacent = sum(1 for i in val_idx if (i - 1) in train or (i + 1) in train)
    frac = adjacent / float(len(val_idx))
    if frac >= 0.8:
        verdict = "by-FRAME split: val frames neighbour their own training frames -- read every score as optimistic"
    elif frac >= 0.3:
        verdict = "partly adjacent -- some val frames neighbour training frames"
    else:
        verdict = "by-SITE split: val frames come from viewpoints train never saw"
    return {"n_val": len(val_idx), "adjacent": adjacent, "fraction": frac, "verdict": verdict}


# ----------------------------------------------------------------------------------------------------------------------
# Metric extraction — pure over an ultralytics DetMetrics-shaped object (so it is testable without torch).
# ----------------------------------------------------------------------------------------------------------------------
def extract_val_metrics(results, names: dict) -> dict:
    """Pull per-class P/R/AP50/AP50-95, the summary scalars and the confusion matrix out of a val result.

    **The index-convention trap** (the same one documented in :func:`gen_dataset._train_metrics`): a
    ``DetMetrics`` exposes ``.maps`` indexed by CLASS ID, but the vectors on the nested ``.box`` metric
    (``p``, ``r``, ``ap50``, ``ap``) are indexed by POSITION in ``.ap_class_index``. Reading one with the other's
    convention reports one class's score under another class's name and looks entirely plausible. Everything
    per-class here goes through ``ap_class_index``; ``.maps`` is used only as a fallback for AP50-95.

    Returns ``{'per_class': {name: {p, r, ap50, ap}}, 'summary': {...}, 'matrix': [[...]] | None,
    'scored_ids': [...], 'errors': [...]}``. Missing pieces degrade to ``None`` with a recorded reason instead
    of raising — a partial metrics object is still worth reporting, as long as the gap is visible.
    """
    out = {"per_class": {}, "summary": {}, "matrix": None, "scored_ids": [], "errors": []}
    src = results

    try:
        rd = getattr(src, "results_dict", None)
        if isinstance(rd, dict):
            out["summary"] = {k: float(v) for k, v in rd.items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    except Exception as exc:
        out["errors"].append(f"summary: {exc}")

    try:
        idx = getattr(src, "ap_class_index", None)
        idx = [] if idx is None else [int(i) for i in list(idx)]
        out["scored_ids"] = idx
        box = getattr(src, "box", None)
        vectors = {}
        for key, attr in (("p", "p"), ("r", "r"), ("ap50", "ap50"), ("ap", "ap")):
            vals = getattr(box, attr, None) if box is not None else None
            vectors[key] = None if vals is None else list(vals)
        maps = getattr(src, "maps", None)
        maps = None if maps is None else list(maps)

        for pos, cid in enumerate(idx):
            name = names.get(cid, str(cid))
            row = {}
            for key in ("p", "r", "ap50", "ap"):
                vec = vectors[key]
                row[key] = float(vec[pos]) if vec is not None and pos < len(vec) else None
            if row["ap"] is None and maps is not None and cid < len(maps):
                row["ap"] = float(maps[cid])      # .maps is CLASS-ID indexed -- only safe used this way
            out["per_class"][name] = row
    except Exception as exc:
        out["errors"].append(f"per-class: {exc}")

    try:
        cm = getattr(src, "confusion_matrix", None)
        mat = getattr(cm, "matrix", None) if cm is not None else None
        if mat is not None:
            out["matrix"] = [[float(v) for v in row] for row in (mat.tolist() if hasattr(mat, "tolist") else mat)]
    except Exception as exc:
        out["errors"].append(f"confusion matrix: {exc}")

    return out


def compare_names(model_names, yaml_names) -> tuple:
    """``(ok, message)`` for "do the weights and the dataset agree on what class id N means?".

    This is the quietest way for the whole report to be wrong. Class ids are assigned in
    :func:`gen_dataset.capture_frame` in first-seen-sorted order, so a re-scan that finds one new object type
    renumbers everything after it. Validating those weights against the new labels then scores ``Fridge_open``'s
    predictions against ``Fork``'s ground truth, and prints a table of confident, per-class, wrong numbers.
    """
    m = {int(k): str(v) for k, v in dict(model_names or {}).items()}
    y = {int(k): str(v) for k, v in dict(yaml_names or {}).items()}
    if not m:
        return True, "weights carry no class names; assuming they match data.yaml"
    if m == y:
        return True, ""
    diffs = [f"{i}: weights={m.get(i, '-')!r} data.yaml={y.get(i, '-')!r}"
             for i in sorted(set(m) | set(y)) if m.get(i) != y.get(i)]
    return False, (f"weights and data.yaml disagree on {len(diffs)} class id(s) — every per-class number would be "
                   f"misattributed. Retrain this scene against the current dataset, or benchmark the dataset the "
                   f"weights were trained on.\n    " + "\n    ".join(diffs[:12]) +
                   ("\n    ..." if len(diffs) > 12 else ""))


# ----------------------------------------------------------------------------------------------------------------------
# Confusion matrix — orientation, then the state-pair arithmetic that the thesis actually cares about.
# ----------------------------------------------------------------------------------------------------------------------
def _transpose(matrix):
    return [list(col) for col in zip(*matrix)]


def infer_orientation(matrix, gt_by_id: dict) -> dict:
    """Work out which axis of an ultralytics confusion matrix is GROUND TRUTH, by checking it against the labels.

    ultralytics stores ``matrix[predicted, truth]`` with a background row/column at index ``nc``, so every
    ground-truth instance lands somewhere in its own COLUMN (matched, swapped, or in the background row). The
    column sums must therefore equal the instance counts we independently counted off the label files. Rather
    than trusting a library convention that a future release could flip — and which cannot be verified from a
    machine without torch — we test both axes against those counts and adopt whichever matches.

    Returns ``{'orientation': 'pred_rows'|'gt_rows', 'confident': bool, 'col_err', 'row_err', 'why'}``. When
    neither axis matches, the closer one is used and ``confident`` is False, which the report prints as a
    warning: the state-pair numbers below it are then suspect rather than wrong-and-silent.
    """
    if not matrix:
        return {"orientation": "pred_rows", "confident": False, "col_err": None, "row_err": None,
                "why": "no confusion matrix available"}
    n = len(matrix)
    nc = n - 1                                    # last index is ultralytics' background bucket
    col = [sum(matrix[r][c] for r in range(n)) for c in range(nc)]
    row = [sum(matrix[r][c] for c in range(n)) for r in range(nc)]
    want = [float(gt_by_id.get(i, 0)) for i in range(nc)]
    col_err = sum(abs(a - b) for a, b in zip(col, want))
    row_err = sum(abs(a - b) for a, b in zip(row, want))
    if col_err <= row_err:
        return {"orientation": "pred_rows", "confident": col_err < 1e-6, "col_err": col_err, "row_err": row_err,
                "why": ("columns match the counted ground truth exactly" if col_err < 1e-6 else
                        f"columns are the closer axis but miss the counted ground truth by {col_err:.0f} instance(s)")}
    return {"orientation": "gt_rows", "confident": row_err < 1e-6, "col_err": col_err, "row_err": row_err,
            "why": ("rows match the counted ground truth exactly (transposed convention)" if row_err < 1e-6 else
                    f"rows are the closer axis but miss the counted ground truth by {row_err:.0f} instance(s)")}


def state_families(names: dict) -> dict:
    """``{objectType: [(id, class_name, state), ...]}`` for every type whose classes carry a state suffix.

    Sorted by state so a family's members are in a stable order. Types with no state axis (``Sink``, ``Fork``)
    are not families and are excluded — there is nothing to confuse them with.
    """
    fams = {}
    for cid, name in sorted(names.items()):
        t, s = perception.split_class_name(str(name))
        if not s:
            continue
        fams.setdefault(t, []).append((int(cid), str(name), s))
    for t in fams:
        fams[t].sort(key=lambda e: e[2])
    return fams


def missing_state_classes(names: dict) -> dict:
    """``{objectType: [class names the state axis defines but the dataset never produced]}``.

    Read off :data:`perception.STATE_AXES`, so it names the *exact* class the next scan has to create — the
    single most actionable line this report can print. A vocabulary containing ``Fridge_closed`` and not
    ``Fridge_open`` means the scan never opened the fridge, and the detector has no way to answer the one
    question the agent opens a fridge to ask.
    """
    present = {str(n) for n in names.values()}
    types = {perception.split_class_name(str(n))[0] for n in names.values()}
    out = {}
    for t in sorted(types):
        spec = perception.STATE_AXES.get(t)
        if not spec:
            continue
        wanted = [f"{t}_{spec[True]}", f"{t}_{spec[False]}"]
        gap = [w for w in wanted if w not in present]
        if gap and len(gap) < len(wanted):        # one half present, the other never photographed
            out[t] = gap
    return out


def state_pair_confusion(names: dict, matrix, orientation: str = "pred_rows") -> list:
    """Per-family state accuracy read off the confusion matrix. ``[]`` when no matrix is available.

    For each object family (``Fridge_open`` / ``Fridge_closed``, ...) with at least two classes:

      * ``gt``       — held-out instances of the family (any of its classes),
      * ``correct``  — right type AND right state,
      * ``swapped``  — right type, WRONG state: predicted one half of the pair where the other was true,
      * ``missed``   — not detected at all (ultralytics' background row),
      * ``other``    — confused with a class outside the family,
      * ``swap_rate``— ``swapped / (correct + swapped)``: of the instances whose *object* the detector got
        right, the fraction whose *state* it got wrong. This is the number that decides whether a state-aware
        detector is doing its job; a family can sit at AP50 0.9 and swap_rate 0.5 at the same time, which reads
        as a good detector and behaves as a coin flip.
      * ``worst``    — the single heaviest off-diagonal cell, as ``(predicted, truth, n)``, because "it always
        says closed" and "it is 50/50 in both directions" call for different fixes.

    Families whose classes are all single-state are returned too (with ``swap_rate=None``), flagged
    ``single_state``: they are unmeasurable *and* that is itself the finding.
    """
    fams = state_families(names)
    if not fams:
        return []
    out = []
    m = _transpose(matrix) if (matrix and orientation == "gt_rows") else matrix
    n = len(m) if m else 0
    nc = n - 1 if n else 0

    for t in sorted(fams):
        members = fams[t]
        ids = [cid for cid, _n, _s in members if cid < nc]
        entry = {"type": t, "classes": [nm for _c, nm, _s in members],
                 "single_state": len(members) < 2, "gt": None, "correct": None, "swapped": None,
                 "missed": None, "other": None, "swap_rate": None, "worst": None}
        if not m or not ids:
            out.append(entry)
            continue
        gt = sum(m[r][c] for c in ids for r in range(n))
        correct = sum(m[c][c] for c in ids)
        swapped = sum(m[p][g] for g in ids for p in ids if p != g)
        missed = sum(m[nc][g] for g in ids)
        entry.update({"gt": gt, "correct": correct, "swapped": swapped, "missed": missed,
                      "other": gt - correct - swapped - missed})
        if correct + swapped > 0:
            entry["swap_rate"] = swapped / float(correct + swapped)
        if len(ids) > 1:
            cells = [(m[p][g], p, g) for g in ids for p in ids if p != g]
            best = max(cells)
            if best[0] > 0:
                entry["worst"] = (names.get(best[1], str(best[1])), names.get(best[2], str(best[2])), best[0])
        out.append(entry)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Per-class rows, ranking, and the diagnosis attached to each.
# ----------------------------------------------------------------------------------------------------------------------
#: Buckets, in the order they are ranked. A class the detector was never shown is a *dataset* failure and cannot
#: be compared with one that was shown and still fails; a class with no held-out instances was not measured at
#: all. Mixing the three into one AP-sorted list is how "it's bad" happens.
BUCKETS = ("untrained", "scored", "unmeasured")


def build_rows(names: dict, train_counts, val_counts, per_class: dict) -> list:
    """One dict per declared class, joining the on-disk counts to the measured scores.

    ``train_counts``/``val_counts`` are :func:`count_labels` results (or ``None`` when that split has no labels
    directory); ``per_class`` is :func:`extract_val_metrics`'s ``per_class``, keyed by class NAME.
    """
    tf = (train_counts or {}).get("frames", {})
    ti = (train_counts or {}).get("instances", {})
    vf = (val_counts or {}).get("frames", {})
    vi = (val_counts or {}).get("instances", {})
    rows = []
    for cid in sorted(names):
        name = str(names[cid])
        t, s = perception.split_class_name(name)
        got = per_class.get(name) or {}
        scored = bool(got) and got.get("ap50") is not None
        train_frames = tf.get(cid, 0) if train_counts is not None else None
        row = {
            "id": int(cid), "name": name, "type": t, "state": s,
            "train_frames": train_frames,
            "train_instances": ti.get(cid, 0) if train_counts is not None else None,
            "val_frames": vf.get(cid, 0) if val_counts is not None else None,
            "val_instances": vi.get(cid, 0) if val_counts is not None else None,
            "precision": got.get("p"), "recall": got.get("r"),
            "ap50": got.get("ap50"), "ap": got.get("ap"),
            "scored": scored,
        }
        if train_frames == 0:
            row["bucket"] = "untrained"
        elif scored:
            row["bucket"] = "scored"
        else:
            row["bucket"] = "unmeasured"
        rows.append(row)
    return rows


def diagnose(row: dict, swap_by_type: dict = None, starved: int = STARVED_FRAMES) -> str:
    """One short, mechanical reason this class is where it is in the ranking — never a vague adjective."""
    swap_by_type = swap_by_type or {}
    tf, vi = row.get("train_frames"), row.get("val_instances")
    if row["bucket"] == "untrained":
        if vi:
            return f"NEVER photographed in train, {vi} held-out instance(s) — the scan missed it"
        return "0 frames in either split — declared but never seen"
    if row["bucket"] == "unmeasured":
        if not vi:
            return "no held-out instances — unscored, quality unknown"
        return f"{vi} held-out instance(s) but no score was reported"

    rate = swap_by_type.get(row.get("type"))
    if rate is not None and rate >= 0.25:
        return f"state confusion {rate:.0%} within {row['type']} — found, but the state is a guess"
    if (row.get("recall") or 0.0) <= 0.05:
        return f"never detected ({vi} held-out instance(s))"
    if tf is not None and tf < starved:
        return f"starved: {tf} training frame(s)"
    if (row.get("recall") or 0.0) >= 0.7 and (row.get("precision") or 1.0) < 0.5:
        return f"over-fires: precision {row['precision']:.2f} at recall {row['recall']:.2f}"
    if (row.get("ap50") or 0.0) < 0.5:
        return "low AP with adequate data — a model/recipe problem, not a data one"
    return ""


def rank_rows(rows: list) -> list:
    """Worst-first ordering: untrained classes, then scored classes by ascending AP50, then the unscored.

    Deterministic to the last tiebreak (class name), so two runs of the same detector produce byte-identical
    tables and a diff between two runs is meaningful.
    """
    order = {b: i for i, b in enumerate(BUCKETS)}
    return sorted(rows, key=lambda r: (order.get(r.get("bucket"), 9),
                                       r["ap50"] if r.get("ap50") is not None else 0.0,
                                       r["recall"] if r.get("recall") is not None else 0.0,
                                       r["train_frames"] if r.get("train_frames") is not None else 0,
                                       r["name"]))


def coverage(rows: list) -> dict:
    """Headline counts: how much of the declared vocabulary this benchmark could actually speak about.

    Two means, deliberately. ``mean_ap50_measured`` averages only the classes ultralytics scored, and so must
    equal its reported mAP50 — :func:`assemble` checks that and warns if it does not, because a mismatch means
    the per-class join is landing scores on the wrong names. ``mean_ap50_all`` counts every declared class the
    detector cannot find as the 0.0 it behaves like. The first is the number the literature quotes; the second
    is the number the agent lives with, and the gap between them is how much of the vocabulary is missing.
    """
    scored = [r for r in rows if r["bucket"] == "scored"]
    measured = [r for r in rows if r.get("ap50") is not None]
    return {
        "n_classes": len(rows),
        "n_scored": len(scored),
        "n_measured": len(measured),
        "n_untrained": sum(1 for r in rows if r["bucket"] == "untrained"),
        "n_no_val": sum(1 for r in rows if (r.get("val_instances") or 0) == 0),
        "n_unmeasured": sum(1 for r in rows if r["bucket"] == "unmeasured"),
        "n_starved": sum(1 for r in rows if (r.get("train_frames") or 0) and r["train_frames"] < STARVED_FRAMES),
        "n_zero_recall": sum(1 for r in measured if (r.get("recall") or 0.0) <= 0.05),
        "mean_ap50_measured": _mean([r["ap50"] for r in measured]),
        # Undefined, not 0.0, when nothing was measured at all: scoring zero classes is not a score of zero.
        "mean_ap50_all": _mean([(r["ap50"] or 0.0) for r in rows]) if measured else None,
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / float(len(vals)) if vals else None


def _ranks(xs):
    """Average (tie-corrected) ranks, 1-based — the input to Spearman's rho."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman rank correlation, or ``None`` for fewer than 3 usable pairs / a constant series.

    Used for AP50 against training-frame count. It is the report's answer to "is this a model problem or a data
    problem?": a strong positive rho says per-class quality is tracking how many frames each class got, and the
    next scan buys more than the next backbone. Rank-based rather than Pearson because frame counts are heavily
    skewed (a CounterTop is in every frame, a Statue in four) and one such outlier would otherwise set the
    correlation on its own.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return None
    return num / math.sqrt(dx * dy)


def topup_plan(rows: list, families: list, missing: dict, limit: int = 12) -> dict:
    """What the next scan should go and photograph, grouped by the mechanism that would fix it.

    Consumed by a human today (and, in the obvious next step, by :mod:`scan_plan`'s per-class target counts):
    ``never_photographed`` wants a container opened or a state flipped, ``starved`` wants more viewpoints of an
    object already in view, ``state_blind`` wants both halves of a pair shot from more angles, and
    ``unscorable`` wants the class to appear in the held-out split at all before any claim is made about it.
    """
    never = [r["name"] for r in rank_rows(rows) if r["bucket"] == "untrained"]
    starved = [f"{r['name']}({r['train_frames']})" for r in rank_rows(rows)
               if r.get("train_frames") and r["train_frames"] < STARVED_FRAMES]
    blind = [f"{f['type']}({f['swap_rate']:.0%})" for f in sorted(
        (f for f in families if f.get("swap_rate") is not None and f["swap_rate"] >= 0.25),
        key=lambda f: -f["swap_rate"])]
    # Keyed on the held-out count itself, NOT on the "unmeasured" bucket. The bucket also catches a class that was
    # simply not scored this run — every class, under --counts-only — and listing a class with twelve held-out
    # instances under "no held-out instances" is exactly the kind of confident wrong line this file exists to stop.
    # Classes with no training frames are omitted: `never_photographed` already owns them, with a stronger fix.
    unscorable = [r["name"] for r in rank_rows(rows)
                  if not (r.get("val_instances") or 0) and (r.get("train_frames") is None
                                                            or r["train_frames"] > 0)]
    def cap(items):
        return items[:limit] + ([f"(+{len(items) - limit} more)"] if len(items) > limit else [])

    return {
        "never_photographed": cap(never),
        "missing_state_halves": {t: v for t, v in sorted(missing.items())},
        "starved": cap(starved),
        "state_blind": cap(blind),
        "unscorable": cap(unscorable),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Rendering.
# ----------------------------------------------------------------------------------------------------------------------
def _f(v, spec=".3f", dash="-"):
    return dash if v is None else format(v, spec)


def _i(v, dash="-"):
    return dash if v is None else str(int(v))


WIDTH = 108


def _rule(label: str) -> str:
    return f"-- {label} " + "-" * max(4, WIDTH - len(label) - 4)


def format_report(report: dict, top: int = 20) -> str:
    """The whole printed benchmark as one string (so the self-test can assert on it, and it can be logged)."""
    L = []
    cfg, cov = report["config"], report["coverage"]
    L.append("=" * WIDTH)
    L.append(f"detector benchmark — {cfg['scene']}  (split: {cfg['split']})")
    L.append("=" * WIDTH)
    if cfg.get("counts_only"):
        L.append("COUNTS ONLY — no model was run. Every P/R/AP column below is a dash, not a zero; only the "
                 "dataset-side")
        L.append("             findings (frame counts, starved classes, missing state halves) are real.")
    L.append(f"weights   {cfg['weights'] or '(none — this scene has not been trained yet)'}")
    L.append(f"dataset   {cfg['dataset_dir']}")
    L.append(f"held-out  {report['split_stats']['n_images']} images, "
             f"{report['split_stats']['n_instances']} instances, "
             f"{cov['n_classes'] - cov['n_no_val']}/{cov['n_classes']} classes represented")
    s = report.get("summary") or {}
    if cov["n_measured"]:
        L.append(f"overall   mAP50 {_f(s.get('mAP50'))}   mAP50-95 {_f(s.get('mAP50-95'))}   "
                 f"P {_f(s.get('precision'))}   R {_f(s.get('recall'))}   "
                 f"(averaged over the {cov['n_measured']} classes that appear in this split, not all "
                 f"{cov['n_classes']})")
    else:
        L.append("overall   not measured — no class was scored (see the coverage block below)")
    for w in report.get("warnings", []):
        L.append(f"WARNING   {w}")

    L.append("")
    L.append(_rule("coverage"))
    pairs = [
        ("classes declared in data.yaml", str(cov["n_classes"]), ""),
        ("ZERO training frames", str(cov["n_untrained"]), "the detector was never shown these"),
        ("ZERO held-out instances", str(cov["n_no_val"]), "cannot be scored - excluded from the means"),
        (f"starved (< {STARVED_FRAMES} training frames)", str(cov["n_starved"]), ""),
        ("scored but never detected (R<=.05)", str(cov["n_zero_recall"]), ""),
        ("mean AP50 over classes in this split", _f(cov["mean_ap50_measured"]),
         "reconciles with the mAP50 above" if cov["n_measured"] else ""),
        ("mean AP50 counting unscored as 0", _f(cov["mean_ap50_all"]),
         "the honest whole-vocabulary figure" if cov["n_measured"] else ""),
    ]
    rho = report.get("ap_vs_frames_rho")
    pairs.append(("AP50 ~ train-frames Spearman rho", _f(rho, "+.3f"),
                  "not enough scored classes to correlate" if rho is None else
                  ("positive = per-class quality tracks how many frames the class got: scan more, "
                   "before retraining differently" if rho > 0.3 else
                   "weak - per-class quality is NOT explained by frame count alone")))
    adj = report.get("split_adjacency") or {}
    pairs.append(("held-out frames next to a train frame", _f(adj.get("fraction"), ".2f"), adj.get("verdict") or ""))
    for label, value, why in pairs:
        L.append(f"  {label:<38} {value:>7}   {('(' + why + ')') if why else ''}".rstrip())

    L.append("")
    L.append(_rule(f"worst classes first (showing {min(top, len(report['rows']))} of {len(report['rows'])})"))
    L.append(f"  {'#':>3} {'class':<24} {'trainF':>6} {'trainI':>6} {'valI':>5} "
             f"{'P':>6} {'R':>6} {'AP50':>6} {'AP':>6}  why")
    for n, r in enumerate(report["rows"][:top], 1):
        L.append((f"  {n:>3} {r['name']:<24} {_i(r['train_frames']):>6} {_i(r['train_instances']):>6} "
                  f"{_i(r['val_instances']):>5} {_f(r['precision'], '.3f'):>6} {_f(r['recall'], '.3f'):>6} "
                  f"{_f(r['ap50'], '.3f'):>6} {_f(r['ap'], '.3f'):>6}  {r.get('why', '')}").rstrip())

    L.append("")
    L.append(_rule("state pairs: can it tell the two halves apart?"))
    fams = report.get("state_pairs") or []
    if not fams:
        L.append("  (no state-bearing classes in this vocabulary — nothing to confuse)")
    else:
        L.append(f"  {'family':<16} {'classes':<38} {'gt':>5} {'ok':>5} {'swap':>5} {'miss':>5} "
                 f"{'swap%':>6}  worst direction")
        for f in fams:
            if f.get("single_state"):
                L.append(f"  {f['type']:<16} {', '.join(f['classes']):<38} "
                         f"{'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>6}  "
                         f"ONLY ONE STATE EVER PHOTOGRAPHED — the axis is untrained")
                continue
            worst = f.get("worst")
            note = (f"says {worst[0]} when it is {worst[1]} ({int(worst[2])}x)" if worst else "")
            L.append((f"  {f['type']:<16} {', '.join(f['classes']):<38} {_i(f['gt']):>5} {_i(f['correct']):>5} "
                      f"{_i(f['swapped']):>5} {_i(f['missed']):>5} "
                      f"{(_f(f['swap_rate'], '.1%') if f['swap_rate'] is not None else '-'):>6}  {note}").rstrip())
        L.append("  swap% = of the instances whose OBJECT was recognised, the fraction whose STATE was wrong.")
        L.append("  A state-aware detector with a high swap% is worthless to the agent however good its mAP is.")

    miss = report.get("missing_state_halves") or {}
    if miss:
        L.append("")
        L.append("  state halves the dataset never produced (the scan never opened/flipped these):")
        for t, gap in sorted(miss.items()):
            L.append(f"    {t:<16} missing {', '.join(gap)}")

    plan = report.get("topup_plan") or {}
    L.append("")
    L.append(_rule("top-up plan for the next scan"))
    for key, label in (("never_photographed", "never photographed"),
                       ("starved", "starved (< %d frames)" % STARVED_FRAMES),
                       ("state_blind", "state-blind families"),
                       ("unscorable", "no held-out instances")):
        items = plan.get(key) or []
        L.append(f"  {label:<22}: " + (", ".join(items) if items else "(none)"))
    L.append("=" * WIDTH)
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------------
# Input resolution — everything that can fail before a GPU is touched, fails here, by name.
# ----------------------------------------------------------------------------------------------------------------------
def _scene_spellings(scene: str):
    """AI2-THOR reports ``FloorPlan1_physics``; the dataset/eval side says ``FloorPlan1``. Try both, either way."""
    base = str(scene or "").strip()
    if not base:
        return []
    stem = base[:-len("_physics")] if base.endswith("_physics") else base
    out = []
    for cand in (base, stem, stem + "_physics"):
        if cand not in out:
            out.append(cand)
    return out


def dataset_path(scene: str, datasets_dir: str) -> str:
    """Resolve a scene's dataset directory (tolerating both scene spellings), or ``''``."""
    for cand in _scene_spellings(scene):
        p = os.path.join(datasets_dir, cand)
        if os.path.isdir(p):
            return p
    return ""


def resolve_inputs(scene=None, dataset_dir=None, weights=None, split="val", root=None,
                   require_weights=True) -> dict:
    """Locate + validate the dataset, the weights and the split. Raises :class:`BenchError`, never returns zeros.

    Every message names the path it looked for and the command that would produce it, because the two realistic
    causes of getting here are "this scene was never scanned" and "this scene was never trained", and those have
    different fixes.

    ``require_weights=False`` (what ``--counts-only`` passes) makes an untrained scene legal, because the moment
    the dataset-side diagnosis is worth the most is *before* spending forty minutes training on a scan that never
    photographed half the vocabulary. Weights are still resolved and reported when they happen to exist; they are
    simply not a precondition for counting label files.
    """
    root = root or HERE
    datasets_dir = os.path.join(root, "datasets")
    weights_dir = os.path.join(root, "weights")

    if not dataset_dir:
        if not scene:
            raise BenchError("pass --scene (e.g. --scene FloorPlan1) or --dataset-dir.")
        dataset_dir = dataset_path(scene, datasets_dir)
        if not dataset_dir:
            have = sorted(os.listdir(datasets_dir)) if os.path.isdir(datasets_dir) else []
            raise BenchError(
                f"no dataset for scene {scene!r} under {datasets_dir!r} "
                f"(looked for {', '.join(_scene_spellings(scene))}). Scanned scenes: {have or '(none)'}. "
                f"Scan the scene first: webapp 'scan & learn', or sim/_phase4_fullscan.py inside WSL.")
    if not os.path.isdir(dataset_dir):
        raise BenchError(f"dataset directory {dataset_dir!r} does not exist.")
    if not scene:
        scene = os.path.basename(os.path.normpath(dataset_dir))   # --dataset-dir alone still names the run

    cfg = parse_data_yaml(os.path.join(dataset_dir, "data.yaml"))

    if not weights:
        weights = perception.weights_path(scene, weights_dir)
        if not weights and require_weights:
            have = sorted(f for f in os.listdir(weights_dir) if f.endswith(".pt")) \
                if os.path.isdir(weights_dir) else []
            raise BenchError(
                f"no trained weights for scene {scene!r} under {weights_dir!r} "
                f"(looked for {', '.join(s + '.pt' for s in _scene_spellings(scene))}). "
                f"Trained scenes: {have or '(none)'}. Train this scene first: webapp 'train detector', or "
                f"gen_dataset.train_scene(...) — see sim/yolo_recipe.py for what that downloads. "
                f"To diagnose the DATASET without training, re-run with --counts-only.")
    if weights:
        if not os.path.isfile(weights):
            raise BenchError(f"weights file {weights!r} does not exist.")
        if os.path.getsize(weights) == 0:
            raise BenchError(f"weights file {weights!r} is empty (0 bytes) — the last train did not finish. "
                             f"Retrain.")

    images_dir, labels_dir = split_dirs(dataset_dir, cfg, split)
    n_images = count_images(images_dir)
    if n_images == 0:
        raise BenchError(
            f"the {split!r} split is empty: no images under {images_dir!r}. A benchmark against zero held-out "
            f"images would print zeros, not a measurement. Re-scan the scene so a held-out split exists.")
    val_counts = count_labels(labels_dir, n_classes=cfg["nc"])
    if val_counts is None:
        raise BenchError(f"no labels directory at {labels_dir!r} for the {split!r} split — the images have no "
                         f"ground truth, so nothing can be scored.")
    if val_counts["n_instances"] == 0:
        raise BenchError(
            f"the {split!r} split has {n_images} image(s) but ZERO labelled instances "
            f"({labels_dir!r} is all empty files). Every AP would be undefined; re-scan the scene.")
    if val_counts["unknown_ids"]:
        raise BenchError(
            f"the {split!r} labels reference class id(s) {val_counts['unknown_ids']} that data.yaml does not "
            f"declare (nc={cfg['nc']}). The labels and the class list are out of sync — re-run "
            f"gen_dataset.build_data_yaml() for this scene, or re-scan.")

    train_images, train_labels = (None, None)
    if cfg["splits"].get("train"):
        train_images, train_labels = split_dirs(dataset_dir, cfg, "train")

    return {"scene": scene, "dataset_dir": dataset_dir, "weights": weights, "split": split, "cfg": cfg,
            "images_dir": images_dir, "labels_dir": labels_dir, "n_images": n_images,
            "val_counts": val_counts, "train_images": train_images, "train_labels": train_labels}


# ----------------------------------------------------------------------------------------------------------------------
# The run itself — the only part that needs ultralytics.
# ----------------------------------------------------------------------------------------------------------------------
def run_val(weights: str, data_yaml: str, split: str = "val", imgsz: int = DEFAULT_IMGSZ,
            batch: int = DEFAULT_BATCH, device=0, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
            project=None):
    """Load ``weights`` and run ultralytics validation; returns ``(results, model_names)``.

    Lazy-imports ultralytics (which pulls torch) — never at module top, so this file imports on Windows.
    """
    try:
        from ultralytics import YOLO  # lazy: heavy (pulls torch); keep module import light
    except Exception as exc:
        raise BenchError(
            f"ultralytics is required to run the detector but failed to import ({exc}). This benchmark needs the "
            f"WSL venv and the GPU: wsl -d Ubuntu-22.04 --cd \"/mnt/d/Personal/Master's Project\" -- "
            f"/home/arash/thor-env/bin/python sim/bench_detect.py --scene <scene>") from exc
    import shutil
    import tempfile

    load_path = weights
    if "'" in os.path.abspath(weights):
        # ultralytics strips apostrophes from absolute weight paths (".../Master's Project/..." -> "Masters
        # Project") and then cannot open them. Same workaround as perception.Detector.__init__.
        load_path = os.path.join(tempfile.gettempdir(), "thor_bench_" + os.path.basename(weights))
        shutil.copyfile(weights, load_path)

    model = YOLO(load_path)
    runs_root = project or tempfile.mkdtemp(prefix="yolo_val_")
    try:
        results = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=device,
                            conf=conf, iou=iou, plots=False, verbose=False,
                            project=runs_root, name="bench", exist_ok=True)
    finally:
        if project is None:
            shutil.rmtree(runs_root, ignore_errors=True)
    return results, dict(getattr(model, "names", {}) or {})


def assemble(inputs: dict, raw: dict, model_names=None) -> dict:
    """Join the resolved inputs and the extracted metrics into the full report dict (pure — no ultralytics).

    Split out from :func:`bench` so the entire aggregation path is exercised headlessly with synthetic metrics.
    """
    cfg = inputs["cfg"]
    names = cfg["names"]
    warnings = []

    train_counts = count_labels(inputs["train_labels"], n_classes=cfg["nc"]) if inputs.get("train_labels") else None
    if train_counts is None:
        warnings.append("no train labels directory found — the training-frame columns are blank, so the "
                        "'starved class' diagnosis cannot be made.")

    if model_names is not None:                # None = no model was loaded (counts-only); {} = it had no names
        ok, msg = compare_names(model_names, names)
        if not ok:
            raise BenchError(msg)
        if msg:
            warnings.append(msg)

    orient = infer_orientation(raw.get("matrix"), inputs["val_counts"]["instances"])
    if raw.get("matrix") and not orient["confident"]:
        warnings.append(f"confusion-matrix orientation is not certain ({orient['why']}) — the state-pair "
                        f"numbers below are indicative, not exact.")
    families = state_pair_confusion(names, raw.get("matrix"), orient["orientation"])
    swap_by_type = {f["type"]: f["swap_rate"] for f in families if f.get("swap_rate") is not None}

    rows = build_rows(names, train_counts, inputs["val_counts"], raw.get("per_class") or {})
    for r in rows:
        r["why"] = diagnose(r, swap_by_type)
    ranked = rank_rows(rows)

    scored = [r for r in rows if r["bucket"] == "scored" and r.get("train_frames") is not None]
    rho = spearman([r["train_frames"] for r in scored], [r["ap50"] for r in scored])

    summary = raw.get("summary") or {}
    def _pick(*keys):
        for k in keys:
            if k in summary:
                return summary[k]
        return None

    # Integrity check on the join itself: ultralytics' mAP50 IS the mean of the per-class AP50s it reported, so
    # if our per-class table disagrees with it, we are attaching scores to the wrong class names and every line
    # below is confidently wrong. Cheap, and it fails the one way the report cannot otherwise detect.
    cov = coverage(rows)
    mAP50 = _pick("metrics/mAP50(B)", "mAP50")
    if mAP50 is not None and cov["mean_ap50_measured"] is not None \
            and abs(cov["mean_ap50_measured"] - mAP50) > 0.005:
        warnings.append(f"per-class AP50 averages to {cov['mean_ap50_measured']:.3f} but ultralytics reports "
                        f"mAP50 {mAP50:.3f} — the per-class join disagrees with the model's own summary, so "
                        f"treat every per-class number below as suspect.")

    missing = missing_state_classes(names)
    report = {
        "config": {"scene": inputs["scene"], "dataset_dir": inputs["dataset_dir"], "weights": inputs["weights"],
                   "split": inputs["split"], "starved_frames": STARVED_FRAMES},
        "split_stats": {"n_images": inputs["n_images"], "n_instances": inputs["val_counts"]["n_instances"],
                        "n_label_files": inputs["val_counts"]["n_files"],
                        "n_empty_labels": inputs["val_counts"]["n_empty"]},
        "train_stats": ({"n_label_files": train_counts["n_files"], "n_instances": train_counts["n_instances"]}
                        if train_counts else None),
        "summary": {"mAP50": _pick("metrics/mAP50(B)", "mAP50"), "mAP50-95": _pick("metrics/mAP50-95(B)", "mAP50-95"),
                    "precision": _pick("metrics/precision(B)", "precision"),
                    "recall": _pick("metrics/recall(B)", "recall"), "fitness": summary.get("fitness")},
        "summary_raw": summary,
        "rows": ranked,
        "coverage": cov,
        "ap_vs_frames_rho": rho,
        "state_pairs": families,
        "missing_state_halves": missing,
        "matrix_orientation": orient,
        "split_adjacency": split_adjacency(frame_indices(inputs.get("train_images") or ""),
                                           frame_indices(inputs["images_dir"])),
        "warnings": warnings + list(raw.get("errors") or []),
    }
    report["topup_plan"] = topup_plan(rows, families, missing)
    return report


def bench(scene=None, dataset_dir=None, weights=None, split="val", imgsz=DEFAULT_IMGSZ, batch=DEFAULT_BATCH,
          device=0, conf=DEFAULT_CONF, iou=DEFAULT_IOU, root=None, counts_only=False) -> dict:
    """Full benchmark: resolve inputs, run validation, return the report dict. Needs ultralytics.

    ``counts_only=True`` skips the model entirely and reports only what the DATASET says — per-class frame
    counts, starved classes, classes with no held-out instances, and the state halves the scan never produced.
    That half of the diagnosis needs no GPU and no ultralytics, so it runs on Windows in a second; the report is
    stamped ``counts_only`` and every measured column is a dash, so it can never be mistaken for a scored run.
    """
    import tempfile

    inputs = resolve_inputs(scene=scene, dataset_dir=dataset_dir, weights=weights, split=split, root=root,
                            require_weights=not counts_only)
    if counts_only:
        raw, model_names = {"per_class": {}, "summary": {}, "matrix": None, "scored_ids": [], "errors": []}, None
    else:
        with tempfile.TemporaryDirectory(prefix="bench_detect_") as td:
            data_yaml = effective_data_yaml(inputs["dataset_dir"], inputs["cfg"], os.path.join(td, "data.yaml"))
            results, model_names = run_val(inputs["weights"], data_yaml, split=split, imgsz=imgsz, batch=batch,
                                           device=device, conf=conf, iou=iou)
        raw = extract_val_metrics(results, inputs["cfg"]["names"])
    report = assemble(inputs, raw, model_names)
    report["config"].update({"imgsz": imgsz, "batch": batch, "conf": conf, "iou": iou, "device": str(device),
                             "counts_only": bool(counts_only)})
    return report


# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="",
                    help="FloorPlan name; resolves sim/datasets/<scene> + sim/weights/<scene>.pt")
    ap.add_argument("--dataset-dir", default="", help="explicit dataset directory (overrides --scene)")
    ap.add_argument("--weights", default="", help="explicit .pt to benchmark (overrides --scene)")
    ap.add_argument("--split", default="val", help="held-out split declared in data.yaml (default: val)")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="keep low while AI2-THOR holds the same GPU (default: 8)")
    ap.add_argument("--device", default="0", help="CUDA index, or 'cpu'")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF,
                    help="detection threshold for mAP (default 0.001; the confusion matrix uses 0.25 regardless)")
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU (default: 0.7)")
    ap.add_argument("--counts-only", action="store_true",
                    help="skip the model: report only the dataset-side findings (per-class frame counts, "
                         "starved classes, missing state halves). No GPU, no ultralytics.")
    ap.add_argument("--top", type=int, default=20, help="rows in the worst-first table (0 = all)")
    ap.add_argument("--out", default="", help="results JSON path (default: sim/bench_detect_<scene>.json)")
    ap.add_argument("--no-write", action="store_true", help="do not write the results JSON")
    args = ap.parse_args(argv)

    device = args.device if args.device == "cpu" else (int(args.device) if args.device.isdigit() else args.device)
    try:
        report = bench(scene=args.scene or None, dataset_dir=args.dataset_dir or None,
                       weights=args.weights or None, split=args.split, imgsz=args.imgsz, batch=args.batch,
                       device=device, conf=args.conf, iou=args.iou, counts_only=args.counts_only)
    except BenchError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    print(format_report(report, top=(len(report["rows"]) if args.top <= 0 else args.top)))

    if not args.no_write:
        stem = report["config"]["scene"] + ("_counts" if args.counts_only else "")
        out = args.out or os.path.join(HERE, f"bench_detect_{stem}.json")
        try:
            tmp = out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, out)
            print(f"machine-readable results -> {out}")
        except Exception as exc:
            print(f"WARNING: could not write {out}: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ===================================================================== PUBLIC API ====================================
# Per-class detection quality for a trained per-scene detector. ultralytics/torch are imported LAZILY inside
# run_val(), so this module imports on bare Windows python; every helper below is pure and covered by
# `python sim/_bench_detect_selftest.py` (run from the REPO ROOT).
#
# BenchError                        Raised for every unmeasurable precondition; main() prints it and exits 2.
# STARVED_FRAMES = 20               Below this many training frames a class is reported as starved.
#
# DATA
#   parse_data_yaml(path) -> {'path', 'splits': {train/val/test}, 'names': {id: name}, 'nc'}
#   split_dirs(dataset_dir, cfg, split) -> (images_dir, labels_dir)   resolved on disk, NOT via the yaml's path:
#   effective_data_yaml(dataset_dir, cfg, out_path) -> path           temp copy with path: repointed for ultralytics
#   count_labels(labels_dir, n_classes=None) -> {'instances', 'frames', 'n_files', 'n_empty', 'n_instances',
#                                                'bad_lines', 'unknown_ids'} | None
#   count_images(images_dir) -> int      frame_indices(images_dir) -> [int]
#   split_adjacency(train_idx, val_idx) -> {'n_val', 'adjacent', 'fraction', 'verdict'}   leakage heuristic
#
# METRICS (pure over a DetMetrics-shaped object)
#   extract_val_metrics(results, names) -> {'per_class': {name: {p, r, ap50, ap}}, 'summary', 'matrix',
#                                           'scored_ids', 'errors'}
#   compare_names(model_names, yaml_names) -> (ok, message)   guards against silently misattributed class ids
#   infer_orientation(matrix, gt_by_id) -> {'orientation', 'confident', 'col_err', 'row_err', 'why'}
#   state_families(names) -> {type: [(id, name, state)]}     missing_state_classes(names) -> {type: [names]}
#   state_pair_confusion(names, matrix, orientation) -> [{type, classes, gt, correct, swapped, missed, other,
#                                                         swap_rate, worst, single_state}]
#
# ROWS / RANKING
#   build_rows(names, train_counts, val_counts, per_class) -> [row]   row.bucket in BUCKETS
#   diagnose(row, swap_by_type=None) -> str      rank_rows(rows) -> [row]  (worst first, deterministic)
#   coverage(rows) -> {...}   spearman(xs, ys) -> float|None   topup_plan(rows, families, missing) -> {...}
#
# RUN
#   resolve_inputs(scene, dataset_dir, weights, split, root=None, require_weights=True) -> dict
#                                     Raises BenchError for anything unmeasurable. require_weights=False lets an
#                                     UNTRAINED scene through, for the counts-only diagnosis.
#   run_val(weights, data_yaml, split, ...) -> (results, model_names)         (needs ultralytics)
#   assemble(inputs, raw, model_names=None) -> report                         (pure)
#   bench(..., counts_only=False) -> report   counts_only skips the model and reports the dataset half only
#                                             (no GPU, no ultralytics, no trained weights needed), stamped as such
#   format_report(report, top=20) -> str      main(argv=None) -> int
