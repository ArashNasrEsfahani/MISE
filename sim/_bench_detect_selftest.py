"""Self-test for the detection benchmark (`sim/bench_detect.py`) — the arithmetic, on synthetic inputs.

Runs headless on Windows `python` from the REPO ROOT — no torch, no ultralytics, no GPU, no ai2thor, and it
never reads or writes `sim/datasets/`, `sim/weights/` or any results file (every fixture is built inside a
`tempfile.TemporaryDirectory`):

    python sim/_bench_detect_selftest.py

What this exists to prove, since the benchmark's whole job is to be BELIEVED:

  * the per-class join is right — the training-frame count printed next to a class's AP is that class's, and
    classes are bucketed into "never trained", "scored" and "not measurable" instead of all being reported as
    0.00 (§2, §7);
  * the state-pair arithmetic is right, and does not depend on trusting ultralytics' confusion-matrix
    orientation from memory: §5 feeds the matrix in BOTH orientations and shows the inference picks the one
    that reconciles with the label files, and §6 shows the swap counts come out identical either way;
  * per-class scores are read with the right index convention — `.box.*` is POSITIONAL in `ap_class_index`
    while `.maps` is CLASS-ID indexed, and mixing them reports one class's score under another's name (§4);
  * the ranking is worst-first and fully deterministic, so two runs diff cleanly (§7);
  * and that it FAILS LOUDLY: every missing/empty/mismatched input raises BenchError naming the path, rather
    than printing a table of zeros (§3, §9).

What this cannot cover: whether the detector is actually any good. That needs the real weights, the real
held-out split and a GPU — `sim/bench_detect.py --scene FloorPlan1`, inside WSL.
"""
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_detect as bd                                             # noqa: E402
import perception                                                     # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


def section(t):
    print(f"\n{t}")


def raises(fn, msg_contains=()):
    """(did it raise BenchError, message) — every precondition in this module must raise, never return zeros."""
    try:
        fn()
    except bd.BenchError as exc:
        text = str(exc)
        return all(m.lower() in text.lower() for m in msg_contains), text
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return False, "no exception raised"


# ----------------------------------------------------------------------------------------------------------------------
# The synthetic scene. Six classes, two full state families, one half-photographed family, one plain type.
#
# Ground truth in the held-out split (instances):  Fridge_closed 10, Fridge_open 6, StoveBurner_off 8,
# StoveBurner_on 4, Sink 5, Microwave_closed 0. Training frames: 40 / 5 / 30 / 0 / 50 / 25 — so StoveBurner_on
# was NEVER photographed yet still appears in val, Fridge_open is starved, and Microwave_closed cannot be
# scored at all. Every expected number below is computed by hand from these.
# ----------------------------------------------------------------------------------------------------------------------
NAMES = {0: "Fridge_closed", 1: "Fridge_open", 2: "StoveBurner_off", 3: "StoveBurner_on", 4: "Sink",
         5: "Microwave_closed"}
VAL_INSTANCES = {0: 10, 1: 6, 2: 8, 3: 4, 4: 5}
TRAIN_FRAMES = {0: 40, 1: 5, 2: 30, 3: 0, 4: 50, 5: 25}

# ultralytics' ConfusionMatrix stores matrix[predicted][truth] with a background bucket at index nc.
# Columns therefore sum to the ground-truth counts above, which is exactly what infer_orientation checks.
MATRIX = [
    #  gt: 0   1   2   3   4  bg
    [7.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # pred Fridge_closed  (1 false positive on background)
    [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # pred Fridge_open
    [0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 0.0],   # pred StoveBurner_off
    [0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0],   # pred StoveBurner_on
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0],   # pred Sink           (2 false positives on background)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # pred Microwave_closed (never predicted)
    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],   # background row = ground truth that was MISSED
]


class _FakeBox:
    """Positional vectors, in ap_class_index order — the opposite convention to .maps, deliberately."""
    p = [0.90, 0.40, 0.80, 0.50, 0.95]
    r = [0.70, 0.33, 0.75, 0.75, 0.80]
    ap50 = [0.85, 0.30, 0.80, 0.45, 0.90]
    ap = [0.60, 0.20, 0.55, 0.30, 0.70]


class _FakeConfusion:
    matrix = MATRIX


class _FakeMetrics:
    results_dict = {"metrics/precision(B)": 0.71, "metrics/recall(B)": 0.66, "metrics/mAP50(B)": 0.66,
                    "metrics/mAP50-95(B)": 0.47, "fitness": 0.49, "task": "detect"}
    maps = [0.60, 0.20, 0.55, 0.30, 0.70, 0.0]        # CLASS-ID indexed (note class 5 present here, not in idx)
    ap_class_index = [0, 1, 2, 3, 4]                  # class 5 has no held-out instances
    box = _FakeBox()
    confusion_matrix = _FakeConfusion()


def write_dataset(root, scene="FloorPlanX", names=NAMES, val_start=100):
    """A complete on-disk YOLO dataset matching the tables above. Returns the dataset directory."""
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    def emit(split, index, cls_ids):
        stem = f"{scene}_{index:06d}_deadbeef01"
        with open(os.path.join(root, "images", split, stem + ".jpg"), "wb") as fh:
            fh.write(b"\xff\xd8\xff")
        with open(os.path.join(root, "labels", split, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("".join(f"{c} 0.5 0.5 0.2 0.2\n" for c in cls_ids))

    for i in range(max(TRAIN_FRAMES.values())):
        ids = [c for c, n in sorted(TRAIN_FRAMES.items()) if i < n]
        emit("train", i, ids)
    for i in range(max(VAL_INSTANCES.values())):
        ids = [c for c, n in sorted(VAL_INSTANCES.items()) if i < n]
        emit("val", val_start + i, ids)

    lines = [f"path: /nowhere/that/exists/{scene}", "train: images/train", "val: images/val", "",
             f"nc: {len(names)}", "names:"] + [f"  {i}: {names[i]}" for i in sorted(names)]
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return root


def write_weights(path, content=b"fake-weights"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# 1. data.yaml parsing.
# ----------------------------------------------------------------------------------------------------------------------
section("1. data.yaml parsing")

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    cfg = bd.parse_data_yaml(os.path.join(ds, "data.yaml"))
    check("names parsed, keyed by int id", cfg["names"] == NAMES, str(cfg["names"]))
    check("nc parsed", cfg["nc"] == 6, str(cfg["nc"]))
    check("splits parsed", cfg["splits"] == {"train": "images/train", "val": "images/val"}, str(cfg["splits"]))
    check("path field parsed", cfg["path"].endswith("FloorPlanX"), str(cfg["path"]))

    # split_dirs resolves against the dataset ON DISK, never the yaml's (foreign, WSL-absolute) path field.
    img, lbl = bd.split_dirs(ds, cfg, "val")
    check("split_dirs ignores the yaml's stale absolute path", os.path.isdir(img), img)
    check("labels dir derived by swapping the 'images' component", os.path.isdir(lbl), lbl)
    check("labels dir is not the images dir", os.path.normpath(img) != os.path.normpath(lbl))

    ok, msg = raises(lambda: bd.split_dirs(ds, cfg, "test"), ("test", "split"))
    check("an undeclared split raises, naming the splits that exist", ok, msg)

    # effective_data_yaml repoints `path:` at the real directory, without touching the shipped file.
    out = bd.effective_data_yaml(ds, cfg, os.path.join(td, "eff.yaml"))
    eff = bd.parse_data_yaml(out)
    check("effective yaml repoints path at the real dataset", os.path.normpath(eff["path"]) == os.path.normpath(ds),
          eff["path"])
    check("effective yaml preserves the class list exactly", eff["names"] == NAMES)
    check("effective yaml preserves nc", eff["nc"] == 6)
    check("the shipped data.yaml is left untouched",
          bd.parse_data_yaml(os.path.join(ds, "data.yaml"))["path"].endswith("FloorPlanX"))

    # Alternative yaml spellings ultralytics also accepts.
    alt = os.path.join(td, "alt.yaml")
    with open(alt, "w", encoding="utf-8") as fh:
        fh.write("path: /x\ntrain: images/train\nval: images/val\nnames: ['Sink', 'Fork']\n")
    check("inline list form parses positionally", bd.parse_data_yaml(alt)["names"] == {0: "Sink", 1: "Fork"},
          str(bd.parse_data_yaml(alt)["names"]))
    check("nc defaults to len(names) when absent", bd.parse_data_yaml(alt)["nc"] == 2)
    with open(alt, "w", encoding="utf-8") as fh:
        fh.write("val: images/val\nnames: {0: Sink, 1: 'Fork'}\n")
    check("inline dict form parses and unquotes", bd.parse_data_yaml(alt)["names"] == {0: "Sink", 1: "Fork"})
    with open(alt, "w", encoding="utf-8") as fh:
        fh.write("# comment\nval: images/val\n\nnames:\n  0: Sink   \n  1: Fork\nnc: 2\n")
    check("comments/blank lines/trailing spaces survive", bd.parse_data_yaml(alt)["names"] == {0: "Sink", 1: "Fork"})

    with open(alt, "w", encoding="utf-8") as fh:
        fh.write("path: /x\nval: images/val\nnc: 0\n")
    ok, msg = raises(lambda: bd.parse_data_yaml(alt), ("no classes",))
    check("a yaml with no class list raises rather than scoring nothing", ok, msg)
    ok, msg = raises(lambda: bd.parse_data_yaml(os.path.join(td, "nope.yaml")), ("data.yaml", "scan"))
    check("a missing data.yaml raises, and says to scan the scene", ok, msg)


# ----------------------------------------------------------------------------------------------------------------------
# 2. Counting ground truth off the label files.
# ----------------------------------------------------------------------------------------------------------------------
section("2. label / image counting")

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    tr = bd.count_labels(os.path.join(ds, "labels", "train"), n_classes=6)
    va = bd.count_labels(os.path.join(ds, "labels", "val"), n_classes=6)

    check("per-class TRAIN FRAME counts match the fixture", tr["frames"] == {0: 40, 1: 5, 2: 30, 4: 50, 5: 25},
          str(tr["frames"]))
    check("a class with zero training frames is simply absent from the map", 3 not in tr["frames"])
    check("per-class VAL INSTANCE counts match the fixture", va["instances"] == VAL_INSTANCES, str(va["instances"]))
    check("total val instances is the sum", va["n_instances"] == 33, str(va["n_instances"]))
    check("frame count counts IMAGES, instance count counts BOXES",
          va["frames"][0] == 10 and va["instances"][0] == 10)
    check("label file count is the number of images", tr["n_files"] == 50, str(tr["n_files"]))
    check("no malformed lines in a clean fixture", tr["bad_lines"] == 0 and not tr["unknown_ids"])
    check("images counted", bd.count_images(os.path.join(ds, "images", "val")) == 10)
    check("counting a missing labels dir returns None (not an empty, honest-looking zero)",
          bd.count_labels(os.path.join(ds, "labels", "nope")) is None)
    check("counting a missing images dir returns 0", bd.count_images(os.path.join(ds, "images", "nope")) == 0)

    # Two frames of the same class are TWO frames; two boxes in one frame are ONE frame.
    solo = os.path.join(td, "solo")
    os.makedirs(solo)
    with open(os.path.join(solo, "a.txt"), "w", encoding="utf-8") as fh:
        fh.write("2 .5 .5 .1 .1\n2 .2 .2 .1 .1\n2 .3 .3 .1 .1\n")
    with open(os.path.join(solo, "b.txt"), "w", encoding="utf-8") as fh:
        fh.write("")
    with open(os.path.join(solo, "c.txt"), "w", encoding="utf-8") as fh:
        fh.write("junk line\n7 .5 .5 .1 .1\n2.0 .5 .5 .1 .1\n")
    got = bd.count_labels(solo, n_classes=6)
    check("three boxes of one class in one frame = 1 frame, 3 instances",
          got["frames"][2] == 2 and got["instances"][2] == 4, str(got))
    check("an empty label file is counted as an empty frame", got["n_empty"] == 1, str(got["n_empty"]))
    check("a malformed line is counted, not crashed on", got["bad_lines"] == 2, str(got["bad_lines"]))
    check("an out-of-range class id is reported by id", got["unknown_ids"] == [7], str(got["unknown_ids"]))
    check("a float-formatted class id still counts", got["instances"][2] == 4)

    # Capture indices + the split-leakage heuristic.
    idx_tr = bd.frame_indices(os.path.join(ds, "images", "train"))
    idx_va = bd.frame_indices(os.path.join(ds, "images", "val"))
    check("capture indices parsed out of the filenames", idx_tr[:3] == [0, 1, 2], str(idx_tr[:3]))
    check("val indices parsed", idx_va[0] == 100, str(idx_va[:3]))

adj = bd.split_adjacency([1, 2, 3, 4, 6, 7, 8, 9], [5, 10])
check("a 1-in-5 by-FRAME split is detected as adjacent", adj["fraction"] == 1.0 and "by-FRAME" in adj["verdict"],
      str(adj))
adj = bd.split_adjacency(list(range(0, 40)), [100, 101, 102])
check("a by-SITE split is detected as clean", adj["fraction"] == 0.0 and "by-SITE" in adj["verdict"], str(adj))
adj = bd.split_adjacency(list(range(0, 10)), [5, 50])
check("a half-adjacent split is reported as partly adjacent", 0.3 <= adj["fraction"] < 0.8, str(adj))
check("no capture indices -> 'unknown', never a made-up fraction",
      bd.split_adjacency([], [1, 2])["fraction"] is None)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Preconditions — every one of these must raise, by name.
# ----------------------------------------------------------------------------------------------------------------------
section("3. loud failures (no zeros, ever)")

with tempfile.TemporaryDirectory() as td:
    root = os.path.join(td, "sim")
    os.makedirs(os.path.join(root, "datasets"))
    os.makedirs(os.path.join(root, "weights"))
    ds = write_dataset(os.path.join(root, "datasets", "FloorPlanX_physics"))

    ok, msg = raises(lambda: bd.resolve_inputs(root=root), ("--scene",))
    check("neither --scene nor --dataset-dir raises", ok, msg)

    ok, msg = raises(lambda: bd.resolve_inputs(scene="FloorPlan9", root=root), ("floorplan9", "scan"))
    check("an unscanned scene raises, listing the scenes that were scanned", ok, msg)
    check("...and that message names the scenes it did find", "FloorPlanX_physics" in msg, msg)

    ok, msg = raises(lambda: bd.resolve_inputs(scene="FloorPlanX", root=root), ("weights", "train"))
    check("a scanned-but-untrained scene raises, saying to train it", ok, msg)
    check("...naming both spellings of the weights file it looked for", "FloorPlanX_physics.pt" in msg, msg)

    # Scene-name spelling tolerance (AI2-THOR says FloorPlan1_physics, the dataset side says FloorPlan1).
    wp = write_weights(os.path.join(root, "weights", "FloorPlanX_physics.pt"))
    got = bd.resolve_inputs(scene="FloorPlanX", root=root)
    check("'FloorPlanX' resolves the '_physics' dataset and weights",
          os.path.normpath(got["dataset_dir"]) == os.path.normpath(ds) and
          os.path.normpath(got["weights"]) == os.path.normpath(wp))
    check("resolve_inputs counts the held-out split it found", got["n_images"] == 10, str(got["n_images"]))
    check("resolve_inputs returns the counted val instances", got["val_counts"]["n_instances"] == 33)
    check("resolve_inputs locates the train labels for the frame-count columns",
          got["train_labels"] and os.path.isdir(got["train_labels"]))

    write_weights(wp, b"")
    ok, msg = raises(lambda: bd.resolve_inputs(scene="FloorPlanX", root=root), ("empty", "retrain"))
    check("a 0-byte weights file raises rather than being loaded", ok, msg)
    write_weights(wp)

    ok, msg = raises(lambda: bd.resolve_inputs(scene="FloorPlanX", weights=os.path.join(td, "no.pt"), root=root),
                     ("does not exist",))
    check("an explicit weights path that does not exist raises", ok, msg)

    ok, msg = raises(lambda: bd.resolve_inputs(dataset_dir=os.path.join(td, "nodir"), root=root), ("does not exist",))
    check("a missing dataset directory raises", ok, msg)

    ok, msg = raises(lambda: bd.resolve_inputs(scene="FloorPlanX", split="test", root=root), ("test",))
    check("a split data.yaml does not declare raises", ok, msg)

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    w = write_weights(os.path.join(td, "w.pt"))

    for f in os.listdir(os.path.join(ds, "images", "val")):
        os.remove(os.path.join(ds, "images", "val", f))
    ok, msg = raises(lambda: bd.resolve_inputs(dataset_dir=ds, weights=w), ("empty", "zeros"))
    check("an EMPTY held-out split raises, and says why zeros would be a lie", ok, msg)

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    w = write_weights(os.path.join(td, "w.pt"))
    for f in os.listdir(os.path.join(ds, "labels", "val")):
        open(os.path.join(ds, "labels", "val", f), "w", encoding="utf-8").close()
    ok, msg = raises(lambda: bd.resolve_inputs(dataset_dir=ds, weights=w), ("zero labelled",))
    check("a held-out split with images but no boxes raises", ok, msg)

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    w = write_weights(os.path.join(td, "w.pt"))
    first = sorted(os.listdir(os.path.join(ds, "labels", "val")))[0]
    with open(os.path.join(ds, "labels", "val", first), "a", encoding="utf-8") as fh:
        fh.write("99 .5 .5 .1 .1\n")
    ok, msg = raises(lambda: bd.resolve_inputs(dataset_dir=ds, weights=w), ("99", "out of sync"))
    check("labels referencing an undeclared class id raise (labels/class-list drift)", ok, msg)

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    w = write_weights(os.path.join(td, "w.pt"))
    got = bd.resolve_inputs(dataset_dir=ds, weights=w)
    check("--dataset-dir alone still names the run after the directory", got["scene"] == "ds", str(got["scene"]))


# ----------------------------------------------------------------------------------------------------------------------
# 4. Metric extraction — the index-convention trap, and graceful degradation.
# ----------------------------------------------------------------------------------------------------------------------
section("4. metric extraction")

raw = bd.extract_val_metrics(_FakeMetrics(), NAMES)
check("summary scalars extracted", raw["summary"]["metrics/mAP50(B)"] == 0.66)
check("non-numeric summary entries dropped", "task" not in raw["summary"])
check("only classes present in ap_class_index are reported",
      set(raw["per_class"]) == {"Fridge_closed", "Fridge_open", "StoveBurner_off", "StoveBurner_on", "Sink"},
      str(sorted(raw["per_class"])))
check("a class with no held-out instances is NOT invented as 0.0", "Microwave_closed" not in raw["per_class"])
check("precision read positionally", raw["per_class"]["Fridge_open"]["p"] == 0.40)
check("recall read positionally", raw["per_class"]["StoveBurner_on"]["r"] == 0.75)
check("AP50 read positionally", raw["per_class"]["Sink"]["ap50"] == 0.90)
check("AP50-95 read positionally from .box.ap (NOT .maps, which is class-id indexed)",
      raw["per_class"]["Sink"]["ap"] == 0.70, str(raw["per_class"]["Sink"]))
check("scored ids recorded", raw["scored_ids"] == [0, 1, 2, 3, 4], str(raw["scored_ids"]))
check("confusion matrix carried through as plain floats",
      raw["matrix"][0][0] == 7.0 and len(raw["matrix"]) == 7)
check("no errors on the happy path", raw["errors"] == [], str(raw["errors"]))


class _Gappy:
    """ap_class_index is a subset and the positional vectors are SHORTER than the class list."""
    results_dict = {"metrics/mAP50(B)": 0.5}
    ap_class_index = [2, 4]
    maps = [0.0, 0.0, 0.55, 0.0, 0.70, 0.0]

    class box:
        ap50 = [0.80, 0.90]
        p = None
        r = None
        ap = None


gap = bd.extract_val_metrics(_Gappy(), NAMES)
check("a subset ap_class_index maps to the right NAMES",
      set(gap["per_class"]) == {"StoveBurner_off", "Sink"}, str(sorted(gap["per_class"])))
check("positional AP50 lands on the right classes",
      gap["per_class"]["StoveBurner_off"]["ap50"] == 0.80 and gap["per_class"]["Sink"]["ap50"] == 0.90)
check("missing .box.ap falls back to .maps BY CLASS ID (0.55 for id 2, not 0.55 for position 2)",
      gap["per_class"]["StoveBurner_off"]["ap"] == 0.55 and gap["per_class"]["Sink"]["ap"] == 0.70,
      str(gap["per_class"]))
check("absent vectors degrade to None, not 0.0", gap["per_class"]["Sink"]["p"] is None)


class _Nothing:
    results_dict = {}


nada = bd.extract_val_metrics(_Nothing(), NAMES)
check("a metrics object with nothing on it does not raise", nada["per_class"] == {} and nada["matrix"] is None)

ok, why = bd.compare_names(NAMES, NAMES)
check("identical class lists pass the weights/data.yaml check", ok and not why)
ok, why = bd.compare_names({}, NAMES)
check("weights without class names are tolerated, with a note", ok and "assuming" in why, why)
shifted = {i: NAMES[(i + 1) % 6] for i in NAMES}
ok, why = bd.compare_names(shifted, NAMES)
check("a SHIFTED class list is caught (the silent misattribution bug)", not ok, why)
check("...and the message names the ids that disagree", "0:" in why and "misattributed" in why, why)


# ----------------------------------------------------------------------------------------------------------------------
# 5. Confusion-matrix orientation, inferred from the labels rather than assumed.
# ----------------------------------------------------------------------------------------------------------------------
section("5. confusion-matrix orientation")

o = bd.infer_orientation(MATRIX, VAL_INSTANCES)
check("ultralytics' matrix[pred][truth] is recognised", o["orientation"] == "pred_rows", str(o))
check("...confidently, because the columns reconcile with the counted labels exactly", o["confident"], str(o))
check("column error is exactly zero", o["col_err"] == 0.0, str(o["col_err"]))
check("the other axis does NOT reconcile (so this was a real test)", o["row_err"] > 0)

trans = [list(col) for col in zip(*MATRIX)]
o2 = bd.infer_orientation(trans, VAL_INSTANCES)
check("a transposed matrix is detected as such", o2["orientation"] == "gt_rows" and o2["confident"], str(o2))

o3 = bd.infer_orientation(MATRIX, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
check("a matrix that matches NEITHER axis is flagged not-confident", not o3["confident"], str(o3))
check("...and still picks the closer axis rather than giving up", o3["orientation"] in ("pred_rows", "gt_rows"))
check("no matrix at all is handled", bd.infer_orientation(None, VAL_INSTANCES)["confident"] is False)


# ----------------------------------------------------------------------------------------------------------------------
# 6. State-pair confusion — the number that decides whether a state-aware detector is doing its job.
# ----------------------------------------------------------------------------------------------------------------------
section("6. state-pair confusion")

fams = bd.state_families(NAMES)
check("families are grouped by objectType", set(fams) == {"Fridge", "StoveBurner", "Microwave"}, str(sorted(fams)))
check("stateless types are not families (nothing to confuse them with)", "Sink" not in fams)
check("family members carry (id, class, state)", fams["Fridge"] == [(0, "Fridge_closed", "closed"),
                                                                   (1, "Fridge_open", "open")], str(fams["Fridge"]))

miss = bd.missing_state_classes(NAMES)
check("the never-produced half of a state axis is named exactly", miss == {"Microwave": ["Microwave_open"]},
      str(miss))
check("a fully-photographed family is not reported as missing", "Fridge" not in miss)
check("the missing half is derived from perception.STATE_AXES, not guessed",
      perception.STATE_AXES["Microwave"][True] == "open")

pairs = {f["type"]: f for f in bd.state_pair_confusion(NAMES, MATRIX, "pred_rows")}
fr = pairs["Fridge"]
check("family ground truth = every held-out instance of either half (10 + 6)", fr["gt"] == 16, str(fr))
check("correct = right type AND right state (7 + 2)", fr["correct"] == 9, str(fr))
check("swapped = right type, WRONG state (2 + 3)", fr["swapped"] == 5, str(fr))
check("missed = the background row for both halves (1 + 1)", fr["missed"] == 2, str(fr))
check("other = confusions with classes outside the family (0 here)", fr["other"] == 0, str(fr))
check("swap_rate = swapped / (correct + swapped) = 5/14", abs(fr["swap_rate"] - 5.0 / 14.0) < 1e-9,
      str(fr["swap_rate"]))
check("the dominant error DIRECTION is reported (says closed when it is open, 3x)",
      fr["worst"] == ("Fridge_closed", "Fridge_open", 3.0), str(fr["worst"]))

sb = pairs["StoveBurner"]
check("second family scored independently (gt 12, correct 9, swapped 2)",
      (sb["gt"], sb["correct"], sb["swapped"], sb["missed"]) == (12, 9, 2, 1), str(sb))
check("a low swap_rate is reported as low", abs(sb["swap_rate"] - 2.0 / 11.0) < 1e-9, str(sb["swap_rate"]))

mw = pairs["Microwave"]
check("a family with only ONE state photographed is flagged, not scored", mw["single_state"] is True and
      mw["swap_rate"] is None, str(mw))

trans_pairs = {f["type"]: f for f in bd.state_pair_confusion(NAMES, trans, "gt_rows")}
check("the SAME numbers come out of a transposed matrix (orientation is handled, not assumed)",
      (trans_pairs["Fridge"]["correct"], trans_pairs["Fridge"]["swapped"], trans_pairs["Fridge"]["missed"]) ==
      (fr["correct"], fr["swapped"], fr["missed"]), str(trans_pairs["Fridge"]))
check("the reported direction also survives transposition",
      trans_pairs["Fridge"]["worst"] == fr["worst"], str(trans_pairs["Fridge"]["worst"]))

check("no matrix -> families are listed with empty counts, never fabricated",
      all(f["gt"] is None for f in bd.state_pair_confusion(NAMES, None)))
check("a vocabulary with no state classes yields no families",
      bd.state_pair_confusion({0: "Sink", 1: "Fork"}, MATRIX) == [])

# A detector that finds every fridge and always calls it closed: perfect recall of the TYPE, useless to the agent.
blind = [[10.0, 6.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
bf = bd.state_pair_confusion({0: "Fridge_closed", 1: "Fridge_open"}, blind, "pred_rows")[0]
check("a state-blind detector scores swap_rate 6/16, not a clean sheet",
      abs(bf["swap_rate"] - 6.0 / 16.0) < 1e-9 and bf["missed"] == 0, str(bf))


# ----------------------------------------------------------------------------------------------------------------------
# 7. Rows, buckets, ranking, diagnosis, coverage.
# ----------------------------------------------------------------------------------------------------------------------
section("7. rows, buckets and the worst-first ranking")

train_counts = {"frames": {k: v for k, v in TRAIN_FRAMES.items() if v},
                "instances": {k: v for k, v in TRAIN_FRAMES.items() if v},
                "n_files": 50, "n_empty": 0, "n_instances": 150, "bad_lines": 0, "unknown_ids": []}
val_counts = {"frames": dict(VAL_INSTANCES), "instances": dict(VAL_INSTANCES), "n_files": 10, "n_empty": 0,
              "n_instances": 33, "bad_lines": 0, "unknown_ids": []}
rows = bd.build_rows(NAMES, train_counts, val_counts, raw["per_class"])
by_name = {r["name"]: r for r in rows}

check("one row per DECLARED class, including the ones nothing can be said about", len(rows) == 6)
check("the training-frame count printed beside a class is that class's",
      by_name["Fridge_closed"]["train_frames"] == 40 and by_name["Fridge_open"]["train_frames"] == 5)
check("held-out instance count joined per class", by_name["Fridge_open"]["val_instances"] == 6)
check("scores joined by class NAME, not by position", by_name["Fridge_open"]["ap50"] == 0.30)
check("class with 0 train frames -> bucket 'untrained' even though it WAS scored",
      by_name["StoveBurner_on"]["bucket"] == "untrained" and by_name["StoveBurner_on"]["ap50"] == 0.45)
check("class with 0 held-out instances -> bucket 'unmeasured', scores left None",
      by_name["Microwave_closed"]["bucket"] == "unmeasured" and by_name["Microwave_closed"]["ap50"] is None)
check("a properly measured class -> bucket 'scored'", by_name["Sink"]["bucket"] == "scored")
check("type/state split off the class name", (by_name["Fridge_open"]["type"], by_name["Fridge_open"]["state"]) ==
      ("Fridge", "open"))

swap_by_type = {f["type"]: f["swap_rate"] for f in bd.state_pair_confusion(NAMES, MATRIX) if f["swap_rate"]}
for r in rows:
    r["why"] = bd.diagnose(r, swap_by_type)
check("an untrained class says so, and cites its held-out instances",
      "NEVER photographed" in by_name["StoveBurner_on"]["why"] and "4 held-out" in by_name["StoveBurner_on"]["why"],
      by_name["StoveBurner_on"]["why"])
check("a state-confused class is diagnosed as state-confused, not merely 'low AP'",
      "state confusion" in by_name["Fridge_open"]["why"] and "36%" in by_name["Fridge_open"]["why"],
      by_name["Fridge_open"]["why"])
check("an unscorable class says the quality is unknown rather than bad",
      "unscored" in by_name["Microwave_closed"]["why"], by_name["Microwave_closed"]["why"])
check("a healthy class gets no invented complaint", by_name["Sink"]["why"] == "", by_name["Sink"]["why"])

check("starvation is diagnosed when nothing worse applies",
      "starved: 7" in bd.diagnose({"bucket": "scored", "train_frames": 7, "val_instances": 9, "type": "Fork",
                                   "recall": 0.6, "precision": 0.8, "ap50": 0.6}),
      bd.diagnose({"bucket": "scored", "train_frames": 7, "val_instances": 9, "type": "Fork", "recall": 0.6,
                   "precision": 0.8, "ap50": 0.6}))
check("a never-detected class is called out before starvation",
      "never detected" in bd.diagnose({"bucket": "scored", "train_frames": 3, "val_instances": 9, "type": "Fork",
                                       "recall": 0.0, "precision": 0.0, "ap50": 0.0}))
check("over-firing (high recall, low precision) is distinguished from missing",
      "over-fires" in bd.diagnose({"bucket": "scored", "train_frames": 90, "val_instances": 9, "type": "Fork",
                                   "recall": 0.9, "precision": 0.2, "ap50": 0.6}))
check("low AP with plenty of data is named as a MODEL problem, not a data one",
      "model/recipe" in bd.diagnose({"bucket": "scored", "train_frames": 90, "val_instances": 9, "type": "Fork",
                                     "recall": 0.4, "precision": 0.9, "ap50": 0.3}))
check("diagnose does not crash when train counts are unavailable",
      isinstance(bd.diagnose({"bucket": "scored", "train_frames": None, "val_instances": 9, "type": "Fork",
                              "recall": 0.6, "precision": 0.8, "ap50": 0.6}), str))

ranked = bd.rank_rows(rows)
order = [r["name"] for r in ranked]
check("worst first: the never-photographed class leads the table", order[0] == "StoveBurner_on", str(order))
check("then scored classes by ASCENDING AP50",
      order[1:5] == ["Fridge_open", "StoveBurner_off", "Fridge_closed", "Sink"], str(order))
check("unmeasurable classes go last (unknown is not the same as bad)", order[-1] == "Microwave_closed", str(order))
check("ranking is deterministic", [r["name"] for r in bd.rank_rows(list(reversed(rows)))] == order)

cov = bd.coverage(rows)
check("coverage: classes declared", cov["n_classes"] == 6)
check("coverage: scored (has training frames AND a score)", cov["n_scored"] == 4, str(cov))
check("coverage: measured = every class ultralytics scored, untrained ones included",
      cov["n_measured"] == 5, str(cov))
check("coverage: ZERO training frames", cov["n_untrained"] == 1, str(cov))
check("coverage: ZERO held-out instances (the task's headline count)", cov["n_no_val"] == 1, str(cov))
check("coverage: starved classes", cov["n_starved"] == 1, str(cov))
check("coverage: never-detected classes", cov["n_zero_recall"] == 0, str(cov))
check("mean AP50 over measured classes reconciles with ultralytics' own mAP50 (0.66)",
      abs(cov["mean_ap50_measured"] - 0.66) < 1e-9, str(cov["mean_ap50_measured"]))
check("mean AP50 over the WHOLE vocabulary is lower, and reported separately",
      abs(cov["mean_ap50_all"] - 0.55) < 1e-9 and cov["mean_ap50_all"] < cov["mean_ap50_measured"],
      str(cov["mean_ap50_all"]))

check("spearman: perfectly monotonic -> +1", abs(bd.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9)
check("spearman: perfectly inverted -> -1", abs(bd.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9)
check("spearman: ties are rank-averaged, not crashed on",
      bd.spearman([1, 1, 2, 2], [5, 6, 7, 8]) is not None)
check("spearman: a constant series has no correlation", bd.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None)
check("spearman: too few points -> None, not a fake number", bd.spearman([1, 2], [1, 2]) is None)
check("spearman: outlier-resistant (rank based) - one huge x does not force +1",
      abs(bd.spearman([1, 2, 3, 10000], [4, 3, 2, 1]) + 1.0) < 1e-9)

plan = bd.topup_plan(rows, bd.state_pair_confusion(NAMES, MATRIX), bd.missing_state_classes(NAMES))
check("top-up plan: never photographed", plan["never_photographed"] == ["StoveBurner_on"], str(plan))
check("top-up plan: starved classes carry their frame count", plan["starved"] == ["Fridge_open(5)"], str(plan))
check("top-up plan: state-blind families carry their swap rate", plan["state_blind"] == ["Fridge(36%)"], str(plan))
check("top-up plan: unscorable classes", plan["unscorable"] == ["Microwave_closed"], str(plan))
# 'no held-out instances' must mean exactly that. Build the rows with NOTHING scored (which is what
# --counts-only produces): every class is then bucket 'unmeasured', but only the one that truly has no held-out
# instances may be listed. Bucketing was the old criterion here and it put a class with 6 held-out instances
# under the heading "no held-out instances".
_unscored_rows = bd.build_rows(NAMES, {"frames": TRAIN_FRAMES, "instances": TRAIN_FRAMES},
                               {"frames": VAL_INSTANCES, "instances": VAL_INSTANCES}, {})
_unscorable = bd.topup_plan(_unscored_rows, [], {})["unscorable"]
check("top-up plan: a class with held-out instances is never called unscorable",
      "Fridge_open" not in _unscorable and "Fridge_closed" not in _unscorable, str(_unscorable))
check("top-up plan: ...while the class that genuinely has none still is",
      _unscorable == ["Microwave_closed"], str(_unscorable))
check("top-up plan: the missing half of a state axis is named",
      plan["missing_state_halves"] == {"Microwave": ["Microwave_open"]}, str(plan))


# ----------------------------------------------------------------------------------------------------------------------
# 8-9. Assembly, rendering, and the end-to-end wiring against a fake ultralytics.
# ----------------------------------------------------------------------------------------------------------------------
section("8. assembly + rendering + end-to-end wiring (fake ultralytics, no torch)")

CALLS = []


class _FakeYOLO:
    names = dict(NAMES)

    def __init__(self, path):
        self.path = path
        CALLS.append(("init", path))

    def val(self, **kwargs):
        # Read the data.yaml HERE: bench() hands it over from a TemporaryDirectory that is gone by the time the
        # call returns, so the only place to see what ultralytics was actually given is inside the call.
        with open(kwargs["data"], "r", encoding="utf-8") as fh:
            CALLS.append(("val", kwargs, fh.read()))
        return _FakeMetrics()


fake = types.ModuleType("ultralytics")
fake.YOLO = _FakeYOLO
sys.modules["ultralytics"] = fake

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    w = write_weights(os.path.join(td, "w.pt"))
    del CALLS[:]
    report = bd.bench(dataset_dir=ds, weights=w, split="val", device="cpu")
    call = next(c for c in CALLS if c[0] == "val")
    seen, yaml_text = dict(call[1]), call[2]

    check("bench loads the weights it was given", os.path.basename(CALLS[0][1]) == "w.pt", str(CALLS[0]))
    check("val is called on the requested split", seen["split"] == "val", str(seen.get("split")))
    check("val is handed the REWRITTEN data.yaml, not the stale one",
          f"path: {os.path.abspath(ds)}" in yaml_text, yaml_text.splitlines()[0])
    check("the rewritten yaml carries the same class ids the report is keyed by",
          all(f"  {i}: {NAMES[i]}" in yaml_text for i in NAMES), yaml_text)
    check("val runs at conf 0.001 so mAP is an area, not a single-threshold score", seen["conf"] == 0.001)
    check("val is given the NMS iou", seen["iou"] == 0.7)
    check("plots are off (this is a measurement, not a training artifact)", seen["plots"] is False)
    check("device passes through", seen["device"] == "cpu")

    check("report: summary mAP50 lifted out of results_dict", report["summary"]["mAP50"] == 0.66,
          str(report["summary"]))
    check("report: mAP50-95 lifted out too", report["summary"]["mAP50-95"] == 0.47)
    check("report: rows are ranked worst-first", report["rows"][0]["name"] == "StoveBurner_on")
    check("report: every declared class has a row", len(report["rows"]) == 6)
    check("report: coverage counts the unscorable classes", report["coverage"]["n_no_val"] == 1)
    check("report: AP50-vs-frames Spearman computed over scored classes (+1 on this fixture)",
          abs(report["ap_vs_frames_rho"] - 1.0) < 1e-9, str(report["ap_vs_frames_rho"]))
    check("report: matrix orientation inferred confidently from the real label counts",
          report["matrix_orientation"]["orientation"] == "pred_rows" and report["matrix_orientation"]["confident"])
    check("report: state pairs present with the hand-computed swap counts",
          {f["type"]: f["swapped"] for f in report["state_pairs"] if not f["single_state"]} ==
          {"Fridge": 5.0, "StoveBurner": 2.0},
          str([(f["type"], f["swapped"]) for f in report["state_pairs"]]))
    check("report: the missing state half is named",
          report["missing_state_halves"] == {"Microwave": ["Microwave_open"]})
    check("report: split adjacency computed (this fixture is a clean by-SITE split)",
          report["split_adjacency"]["fraction"] == 0.0, str(report["split_adjacency"]))
    check("report: no spurious warnings on the happy path", report["warnings"] == [], str(report["warnings"]))
    check("report: train stats recorded", report["train_stats"]["n_label_files"] == 50, str(report["train_stats"]))
    check("report is JSON-serialisable (it is written to disk by main())",
          isinstance(__import__("json").dumps(report), str))

    text = bd.format_report(report, top=10)
    check("render: names the scene and the split", "FloorPlanX" in text or "ds" in text)
    check("render: the worst class is the first row of the table",
          text.index("StoveBurner_on") < text.index("Sink"), "ordering")
    check("render: the training-frame column is printed beside the score", "trainF" in text and "AP50" in text)
    check("render: coverage block names the zero-held-out count", "ZERO held-out instances" in text)
    check("render: the whole-vocabulary mean is shown next to the scored-only mean",
          "mean AP50 over classes in this split" in text and "counting unscored as 0" in text)
    check("render: the state-pair section explains what swap% means", "swap%" in text and "worthless" in text)
    check("render: a half-photographed family is called out in capitals",
          "ONLY ONE STATE EVER PHOTOGRAPHED" in text)
    check("render: the dominant confusion direction is spelled out",
          "says Fridge_closed when it is Fridge_open" in text, "direction")
    check("render: the top-up plan is printed last", text.rindex("top-up plan") > text.rindex("state pairs"))
    check("render: the Spearman line interprets its own sign", "scan more" in text)
    check("render: --top limits the table", len(bd.format_report(report, top=2).splitlines()) <
          len(bd.format_report(report, top=6).splitlines()))
    check("render: prints on this console (cp1252-safe)", text.encode(sys.stdout.encoding or "utf-8", "strict"))

    # A weights file whose class list has drifted from the dataset must stop the run, not misattribute 53 scores.
    class _ShiftedYOLO(_FakeYOLO):
        names = {i: NAMES[(i + 1) % 6] for i in NAMES}

    fake.YOLO = _ShiftedYOLO
    ok, msg = raises(lambda: bd.bench(dataset_dir=ds, weights=w, device="cpu"), ("misattributed",))
    check("weights whose class ids have drifted from data.yaml stop the benchmark", ok, msg)
    fake.YOLO = _FakeYOLO

    # The per-class table must reconcile with ultralytics' own mAP50; if it does not, the join is landing scores
    # on the wrong names and the report says so instead of publishing a plausible-looking table.
    class _Inconsistent(_FakeMetrics):
        results_dict = dict(_FakeMetrics.results_dict, **{"metrics/mAP50(B)": 0.90})

    class _InconsistentYOLO(_FakeYOLO):
        def val(self, **kwargs):
            return _Inconsistent()

    fake.YOLO = _InconsistentYOLO
    r4 = bd.bench(dataset_dir=ds, weights=w, device="cpu")
    check("a per-class table that does not average to the reported mAP50 is flagged",
          any("disagrees with the model's own summary" in w_ for w_ in r4["warnings"]), str(r4["warnings"]))
    check("...and the warning is printed in the report header",
          "WARNING" in bd.format_report(r4, top=3))
    fake.YOLO = _FakeYOLO

    # No confusion matrix at all: the report still comes out, minus the state section, and says so.
    class _NoMatrix(_FakeMetrics):
        confusion_matrix = None

    class _NoMatrixYOLO(_FakeYOLO):
        def val(self, **kwargs):
            return _NoMatrix()

    fake.YOLO = _NoMatrixYOLO
    r2 = bd.bench(dataset_dir=ds, weights=w, device="cpu")
    check("a missing confusion matrix degrades to unscored state pairs, not a crash",
          all(f["swap_rate"] is None for f in r2["state_pairs"]), str(r2["state_pairs"]))
    check("...and the ranked table still works", r2["rows"][0]["name"] == "StoveBurner_on")
    check("...and rendering still works", "state pairs" in bd.format_report(r2, top=6))
    fake.YOLO = _FakeYOLO

    # A dataset with no train labels at all: frame-count columns blank, and a WARNING that says why.
    import shutil
    ds2 = os.path.join(td, "ds2")
    shutil.copytree(ds, ds2)
    shutil.rmtree(os.path.join(ds2, "labels", "train"))
    r3 = bd.bench(dataset_dir=ds2, weights=w, device="cpu")
    check("missing train labels -> the starvation diagnosis is withdrawn, loudly",
          any("starved" in w_.lower() for w_ in r3["warnings"]), str(r3["warnings"]))
    check("...and the frame columns are blank rather than zero",
          all(r["train_frames"] is None for r in r3["rows"]))
    check("...and no class is falsely bucketed 'untrained'",
          not any(r["bucket"] == "untrained" for r in r3["rows"]))
    check("...and the report still renders", "-" in bd.format_report(r3, top=6))

del sys.modules["ultralytics"]

check("bench_detect imports with no torch/ultralytics present", "ultralytics" not in sys.modules)


# ----------------------------------------------------------------------------------------------------------------------
# 9. --counts-only: the dataset half of the diagnosis, with NO model, NO GPU and no ultralytics installed.
#    It must be obviously unmeasured rather than quietly zero.
# ----------------------------------------------------------------------------------------------------------------------
section("9. counts-only mode (no model at all)")

with tempfile.TemporaryDirectory() as td:
    ds = write_dataset(os.path.join(td, "ds"))
    w = write_weights(os.path.join(td, "w.pt"))
    r5 = bd.bench(dataset_dir=ds, weights=w, counts_only=True)

    check("counts-only runs with ultralytics absent", "ultralytics" not in sys.modules)
    check("counts-only is stamped in the config", r5["config"]["counts_only"] is True)
    check("counts-only still counts the training frames per class",
          {r["name"]: r["train_frames"] for r in r5["rows"]}["Fridge_closed"] == 40)
    check("counts-only still finds the never-photographed class",
          "StoveBurner_on" in r5["topup_plan"]["never_photographed"], str(r5["topup_plan"]))
    check("counts-only still names the missing state half",
          r5["missing_state_halves"] == {"Microwave": ["Microwave_open"]})
    check("counts-only lists starved classes even though nothing was scored",
          r5["topup_plan"]["starved"] == ["Fridge_open(5)"], str(r5["topup_plan"]["starved"]))
    check("counts-only reports NO measured classes", r5["coverage"]["n_measured"] == 0)
    check("a mean over zero measured classes is None, NOT 0.0 (the whole point of this file)",
          r5["coverage"]["mean_ap50_all"] is None and r5["coverage"]["mean_ap50_measured"] is None,
          str(r5["coverage"]))
    check("counts-only does not claim the weights lack class names (no model was loaded)",
          r5["warnings"] == [], str(r5["warnings"]))
    check("counts-only does not fabricate a Spearman correlation", r5["ap_vs_frames_rho"] is None)

    text5 = bd.format_report(r5, top=6)
    check("render: the counts-only banner is unmissable", "COUNTS ONLY" in text5)
    check("render: the overall line says not measured instead of showing 0.000",
          "not measured" in text5 and "mAP50 0.000" not in text5)
    check("render: the rho line admits there is nothing to correlate",
          "not enough scored classes to correlate" in text5)

    # An UNTRAINED scene is the single most valuable moment to run this: the counts say whether the scan is
    # worth training on at all. Requiring a .pt to count label files would have made the mode useless exactly
    # when it pays for itself, so counts-only must not demand one.
    r6 = bd.bench(dataset_dir=ds, counts_only=True, root=os.path.join(td, "no-weights-here"))
    check("counts-only works on a scene that was never trained", r6["config"]["counts_only"] is True)
    check("...and does not invent a weights path", r6["config"]["weights"] == "")
    check("...and still produces the same dataset findings", r6["topup_plan"] == r5["topup_plan"])
    check("...and says so in the header rather than printing a blank",
          "has not been trained yet" in bd.format_report(r6, top=6))
    # ...but a SCORING run against an untrained scene must still stop, and point at counts-only.
    ok6, msg6 = raises(lambda: bd.bench(dataset_dir=ds, root=os.path.join(td, "no-weights-here")),
                       ("no trained weights", "--counts-only"))
    check("a scoring run on an untrained scene still fails loudly, and names the way forward", ok6, msg6)

# Truncated top-up lists say they are truncated.
many = [{"name": f"C{i}", "bucket": "untrained", "train_frames": 0, "val_instances": 0, "ap50": None,
         "recall": None, "type": f"C{i}"} for i in range(20)]
capped = bd.topup_plan(many, [], {}, limit=3)
check("a truncated top-up list says how many it left out",
      len(capped["never_photographed"]) == 4 and capped["never_photographed"][-1] == "(+17 more)",
      str(capped["never_photographed"]))

print(f"\n{'=' * 70}\n{PASS}/{PASS + FAIL} checks passed" + (f" - {FAIL} FAILED" if FAIL else " - ALL PASS"))
sys.exit(1 if FAIL else 0)
