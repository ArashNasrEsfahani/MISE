"""Self-test for the per-scene detector's base-model selection and training recipe (`sim/yolo_recipe.py`).

Runs headless on Windows `python` from the REPO ROOT — no torch, no ultralytics, no GPU, no ai2thor:

    python sim/_detector_selftest.py

What this exists to prove:

  * the DEFAULT base is `yolo11s.pt`, not the nano `yolov8n.pt` that produced the detector the user called
    unusable (§1);
  * `$THOR_YOLO_BASE` re-points it without a code edit, and an explicit `base=` still beats the env — so
    `base='yolov8n.pt'` keeps reproducing the published Phase-4 numbers (§1, §4);
  * the download manifest is honest about what a first train pulls over the network (§2);
  * no augmentation in the recipe can change the pixels that carry a STATE label — the failure mode that makes
    `StoveBurner_on` and `StoveBurner_off` the same class as far as the loss is concerned (§3);
  * and that all of it actually REACHES `model.train()`, rather than being a well-documented dict nobody passes
    (§4) — checked by binding a fake `ultralytics` module and driving the real `gen_dataset.train_scene`.

§5 covers the metric extraction, including the index-convention trap: `DetMetrics.maps` is indexed by class id
while `DetMetrics.box.ap50` is indexed by position in `ap_class_index`, so mixing them silently reports one
class's score under another class's name.

What this cannot cover: whether yolo11s actually detects better than yolov8n in the kitchen. That needs a real
train in WSL — `sim/_phase4_fullscan.py`, Track C in HOW-TO-TEST.html.
"""
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_dataset                                                    # noqa: E402
import yolo_recipe as yr                                              # noqa: E402

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


# ----------------------------------------------------------------------------------------------------------------------
# 1. Base selection: default, env override, explicit argument, custom paths.
# ----------------------------------------------------------------------------------------------------------------------
section("1. base selection")

check("default base is yolo11s.pt (small, not nano)", yr.DEFAULT_BASE == "yolo11s.pt", yr.DEFAULT_BASE)
check("default resolves with an empty env", yr.resolve_base(env={}) == "yolo11s.pt", yr.resolve_base(env={}))
check("default base is a known, fetchable asset", yr.DEFAULT_BASE in yr.BASES)

check("default is a real upgrade on the old nano base",
      yr.BASES["yolo11s.pt"]["coco_map"] > yr.BASES["yolov8n.pt"]["coco_map"] + 5.0,
      f"{yr.BASES['yolo11s.pt']['coco_map']} vs {yr.BASES['yolov8n.pt']['coco_map']}")
check("default stays small enough to share an 8 GB card (< 12M params)",
      yr.BASES[yr.DEFAULT_BASE]["params_m"] < 12.0, str(yr.BASES[yr.DEFAULT_BASE]["params_m"]))

check("env override applies", yr.resolve_base(env={yr.BASE_ENV: "yolo11m.pt"}) == "yolo11m.pt")
check("env override accepts a bare stem (adds .pt)", yr.resolve_base(env={yr.BASE_ENV: "yolo11m"}) == "yolo11m.pt",
      yr.resolve_base(env={yr.BASE_ENV: "yolo11m"}))
check("blank env is ignored", yr.resolve_base(env={yr.BASE_ENV: "   "}) == yr.DEFAULT_BASE)
check("missing env key is ignored", yr.resolve_base(env={"OTHER": "x"}) == yr.DEFAULT_BASE)

check("explicit base beats the env",
      yr.resolve_base("yolov8n.pt", env={yr.BASE_ENV: "yolo11l.pt"}) == "yolov8n.pt")
check("old yolov8n.pt is still reachable explicitly", yr.resolve_base("yolov8n.pt", env={}) == "yolov8n.pt")
check("old yolov8n.pt is still reachable via env", yr.resolve_base(env={yr.BASE_ENV: "yolov8n.pt"}) == "yolov8n.pt")
check("a custom checkpoint path passes through untouched",
      yr.resolve_base("/home/arash/mine/best.pt", env={}) == "/home/arash/mine/best.pt")
check("a .yaml (from-scratch) spec keeps its suffix", yr.resolve_base("yolo11s.yaml", env={}) == "yolo11s.yaml")
check("empty-string base falls back rather than resolving to '.pt'",
      yr.resolve_base("", env={}) == yr.DEFAULT_BASE, yr.resolve_base("", env={}))

# Checkpoint caching: known assets get a cache path, everything else is left to ultralytics.
tmp_cache = os.path.join(tempfile.gettempdir(), "thor_yolo_cache_selftest")
env_cache = {yr.CACHE_ENV: tmp_cache}
check("known asset resolves into the cache dir",
      yr.checkpoint_path("yolo11s.pt", env=env_cache) == os.path.join(tmp_cache, "yolo11s.pt"))
check("cache dir is created on demand", os.path.isdir(tmp_cache))
check("a path-like base is not re-homed into the cache",
      yr.checkpoint_path(os.path.join("some", "dir", "best.pt"), env=env_cache).endswith("best.pt")
      and tmp_cache not in yr.checkpoint_path(os.path.join("some", "dir", "best.pt"), env=env_cache))
check("an unknown asset name is left for ultralytics to resolve",
      yr.checkpoint_path("yolo99z.pt", env=env_cache) == "yolo99z.pt")
check("an apostrophe in the cache dir disables caching (ultralytics strips it)",
      yr.checkpoint_path("yolo11s.pt", env={yr.CACHE_ENV: os.path.join(tempfile.gettempdir(), "Master's")})
      == "yolo11s.pt")
check("cache dir defaults under the home directory",
      yr.cache_dir(env={}).startswith(os.path.expanduser("~")), yr.cache_dir(env={}))


# ----------------------------------------------------------------------------------------------------------------------
# 2. Download manifest — what a first train pulls, and how big.
# ----------------------------------------------------------------------------------------------------------------------
section("2. first-train download manifest")

man = yr.download_manifest(env={})
files = [m["file"] for m in man]
check("manifest names the base checkpoint", "yolo11s.pt" in files, str(files))
check("manifest declares the base size (~19 MB)",
      18.0 <= (man[0]["mb"] or 0) <= 21.0, str(man[0]["mb"]))
# The AMP-check checkpoint is whatever the INSTALLED ultralytics hard-codes, not a name we get to choose: 8.4.83's
# checks.py runs yolo26n.pt (and the June Phase-4 train left a 5 544 453-byte copy of exactly that in the repo root).
check("manifest includes the AMP-check nano model", yr.AMP_CHECKPOINT in files, str(files))
check("and that model is the one ultralytics 8.4.83 actually fetches", yr.AMP_CHECKPOINT == "yolo26n.pt",
      yr.AMP_CHECKPOINT)
check("manifest includes the plotting font", "Arial.ttf" in files, str(files))
check("total first-train download stays under 30 MB", sum(m["mb"] or 0 for m in man) < 30.0)
check("every manifest row explains itself", all(m.get("why") for m in man))
check("manifest follows the env override",
      "yolo11m.pt" in [m["file"] for m in yr.download_manifest(env={yr.BASE_ENV: "yolo11m"})])
check("a custom checkpoint is reported as not-downloaded",
      yr.download_manifest("/tmp/mine.pt", env={})[0]["mb"] is None)
check("choosing the nano base does not double-list it",
      [m["file"] for m in yr.download_manifest("yolo11n.pt", env={})].count("yolo11n.pt") == 1)
check("every catalog entry is fully specified",
      all({"params_m", "gflops", "coco_map", "download_mb"} <= set(v) for v in yr.BASES.values()))


# ----------------------------------------------------------------------------------------------------------------------
# 3. The recipe — values, and the state-safety invariants.
# ----------------------------------------------------------------------------------------------------------------------
section("3. training recipe")

kw = yr.train_kwargs(n_images=333)
check("epochs default is 150 (was a hard-coded 50)", kw["epochs"] == 150 == yr.DEFAULT_EPOCHS, str(kw["epochs"]))
check("patience is set and below the epoch ceiling, so early stopping can actually fire",
      0 < kw["patience"] < kw["epochs"], f"{kw['patience']}/{kw['epochs']}")
check("imgsz 640 matches the 600-640 px rendered frames", kw["imgsz"] == 640)
check("batch is halved to 8 to coexist with AI2-THOR on 8 GB", kw["batch"] == 8)
check("batch is NOT AutoBatch (-1), which OOMs against a live simulator", kw["batch"] != -1)
check("amp is on (halves activation memory)", kw["amp"] is True)
check("cosine LR schedule for the long fine-tune", kw["cos_lr"] is True)
check("seeded", kw["seed"] == 0)
check("deterministic", kw["deterministic"] is True)
check("plots on (confusion matrix / PR curves are copied back for the thesis)", kw["plots"] is True)
check("no lr0 override (optimizer='auto' would silently ignore it)", "lr0" not in kw)
check("no optimizer override (auto picks AdamW + an LR for a small set)", "optimizer" not in kw)

# State-corrupting augmentation must be pinned OFF. Each of these can change the pixels that carry the state label
# while the label itself stays put -> label noise, not regularization.
for key in ("flipud", "degrees", "shear", "perspective", "mixup", "cutmix", "copy_paste", "bgr", "erasing"):
    check(f"state-corrupting aug '{key}' pinned to 0", kw[key] == 0, f"{key}={kw[key]}")

check("hsv_v tightened below the ultralytics default of 0.4 (the on/off glow cue)",
      0 < kw["hsv_v"] < 0.4, str(kw["hsv_v"]))
check("hsv_s tightened below the ultralytics default of 0.7 (the cooked/filled colour cue)",
      0 < kw["hsv_s"] < 0.7, str(kw["hsv_s"]))
check("hsv_h tightened below the ultralytics default of 0.015",
      0 < kw["hsv_h"] < 0.015, str(kw["hsv_h"]))
check("some photometric jitter is kept (renderer gamma/shading does vary)",
      kw["hsv_v"] > 0 and kw["hsv_s"] > 0)
check("horizontal flip KEPT — no chiral classes, so it is state-preserving", kw["fliplr"] == 0.5)
check("scale jitter kept — the agent genuinely views objects at 0.5x-2x distance", kw["scale"] == 0.5)
check("mosaic reduced below the default 1.0 but not removed", 0 < kw["mosaic"] < 1.0, str(kw["mosaic"]))
check("close_mosaic lengthened past the default 10", kw["close_mosaic"] > 10, str(kw["close_mosaic"]))
check("close_mosaic still leaves a clean tail inside the schedule", kw["close_mosaic"] < kw["epochs"])

check("small dataset is cached in RAM (dataset lives on a slow drvfs mount)",
      yr.train_kwargs(n_images=333)["cache"] == "ram")
check("large dataset falls back to disk reads",
      yr.train_kwargs(n_images=yr.RAM_CACHE_MAX_IMAGES + 1)["cache"] is False)
check("unknown dataset size does not enable the RAM cache", yr.train_kwargs()["cache"] is False)
check("explicit cache= wins", yr.train_kwargs(n_images=10, cache="disk")["cache"] == "disk")

check("explicit epochs wins over the default", yr.train_kwargs(epochs=7)["epochs"] == 7)
check("explicit imgsz/batch/patience/seed win",
      yr.train_kwargs(imgsz=320, batch=2, patience=5, seed=11) ==
      {**yr.train_kwargs(), "imgsz": 320, "batch": 2, "patience": 5, "seed": 11})
check("overrides are applied last and win", yr.train_kwargs(epochs=7, overrides={"epochs": 9})["epochs"] == 9)
check("overrides can add a knob the recipe does not set", yr.train_kwargs(overrides={"freeze": 10})["freeze"] == 10)
check("train_kwargs returns a fresh dict each call (no shared mutable default)",
      yr.train_kwargs() is not yr.train_kwargs())
kw["mosaic"] = 99
check("mutating a returned recipe does not poison STATE_SAFE_AUG", yr.train_kwargs()["mosaic"] != 99)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Wiring — the recipe reaches ultralytics. A fake `ultralytics` module stands in for torch.
# ----------------------------------------------------------------------------------------------------------------------
section("4. train_scene wiring (fake ultralytics, no torch)")

CALLS = []


class _FakeBox:
    ap50 = [0.90, 0.40]
    r = [0.88, 0.30]


class _FakeMetrics:
    results_dict = {"metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7, "metrics/mAP50(B)": 0.75,
                    "metrics/mAP50-95(B)": 0.6, "fitness": 0.61}
    maps = [0.71, 0.0, 0.22]          # indexed by CLASS ID
    ap_class_index = [0, 2]           # class 1 was never seen in val
    box = _FakeBox()


class _FakeYOLO:
    names = {0: "Fridge_open", 1: "Fridge_closed", 2: "StoveBurner_on"}

    def __init__(self, base):
        self.base = base
        CALLS.append(("init", base))

    def train(self, **kwargs):
        CALLS.append(("train", kwargs))
        run = os.path.join(kwargs["project"], kwargs["name"])
        os.makedirs(os.path.join(run, "weights"), exist_ok=True)
        with open(os.path.join(run, "weights", "best.pt"), "wb") as fh:
            fh.write(b"fake-weights")
        res = _FakeMetrics()
        res.save_dir = run
        return res


fake = types.ModuleType("ultralytics")
fake.YOLO = _FakeYOLO
sys.modules["ultralytics"] = fake


def make_dataset(root, n_images=4):
    """A minimal on-disk dataset: data.yaml + n dummy jpgs, enough for train_scene's preconditions."""
    img_dir = os.path.join(root, "images", "train")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(n_images):
        with open(os.path.join(img_dir, f"f{i}.jpg"), "wb") as fh:
            fh.write(b"\xff\xd8\xff")
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("path: .\ntrain: images/train\nval: images/val\nnc: 3\n"
                 "names:\n  0: Fridge_open\n  1: Fridge_closed\n  2: StoveBurner_on\n")
    return root


def run_train(env_overrides=None, **kwargs):
    """Drive the real gen_dataset.train_scene against the fake ultralytics; returns (result, train_kwargs_seen)."""
    del CALLS[:]
    saved = {k: os.environ.get(k) for k in (yr.BASE_ENV, yr.CACHE_ENV)}
    os.environ[yr.CACHE_ENV] = tmp_cache
    if env_overrides:
        os.environ.update(env_overrides)
    else:
        os.environ.pop(yr.BASE_ENV, None)
    try:
        with tempfile.TemporaryDirectory() as td:
            ds = make_dataset(os.path.join(td, "ds"))
            out = os.path.join(td, "weights", "FloorPlanX.pt")
            res = gen_dataset.train_scene("FloorPlanX", ds, out, **kwargs)
            res["_weights_exists"] = bool(res.get("weights_path")) and os.path.exists(res["weights_path"])
            res["_runs_copied"] = os.path.isdir(os.path.join(ds, "runs", "FloorPlanX"))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    seen = dict(next(c[1] for c in CALLS if c[0] == "train"))
    init = next(c[1] for c in CALLS if c[0] == "init")
    return res, seen, init


res, seen, init = run_train()
check("train_scene loads the default base", os.path.basename(init) == "yolo11s.pt", init)
check("the base is loaded from the cache dir", init == os.path.join(tmp_cache, "yolo11s.pt"), init)
check("result reports the resolved base", res.get("base") == "yolo11s.pt", str(res.get("base")))
check("result echoes the recipe actually used", isinstance(res.get("recipe"), dict) and res["recipe"]["epochs"] == 150)
check("result reports the dataset size it trained on", res.get("n_images") == 4, str(res.get("n_images")))
check("best.pt is copied to weights_out", res["_weights_exists"])
check("run artifacts are copied back under the dataset dir", res["_runs_copied"])

check("data.yaml path is passed", seen["data"].endswith("data.yaml"))
check("epochs reaches ultralytics", seen["epochs"] == 150, str(seen["epochs"]))
check("patience reaches ultralytics", seen["patience"] == yr.DEFAULT_PATIENCE)
check("imgsz reaches ultralytics", seen["imgsz"] == 640)
check("batch reaches ultralytics", seen["batch"] == 8)
check("seed + deterministic reach ultralytics", seen["seed"] == 0 and seen["deterministic"] is True)
check("cos_lr reaches ultralytics", seen["cos_lr"] is True)
check("RAM cache reaches ultralytics for this small set", seen["cache"] == "ram")
check("the whole state-safe augmentation block reaches ultralytics",
      all(seen.get(k) == v for k, v in yr.STATE_SAFE_AUG.items()),
      str({k: seen.get(k) for k in yr.STATE_SAFE_AUG if seen.get(k) != yr.STATE_SAFE_AUG[k]}))
check("no duplicate-keyword crash from data/device/project/name",
      {"device", "project", "name", "exist_ok"} <= set(seen))

res2, seen2, init2 = run_train(env_overrides={yr.BASE_ENV: "yolo11m"})
check("env override reaches train_scene", res2.get("base") == "yolo11m.pt", str(res2.get("base")))
check("env override reaches the model constructor", os.path.basename(init2) == "yolo11m.pt", init2)

res3, seen3, init3 = run_train(env_overrides={yr.BASE_ENV: "yolo11m"}, base="yolov8n.pt")
check("explicit base beats the env inside train_scene", res3.get("base") == "yolov8n.pt", str(res3.get("base")))
check("explicit yolov8n.pt reaches the constructor (old results stay reproducible)",
      os.path.basename(init3) == "yolov8n.pt", init3)

res4, seen4, _ = run_train(epochs=3, imgsz=320, batch=2, patience=1, seed=7,
                           overrides={"mosaic": 0.0, "workers": 2})
check("explicit epochs/imgsz/batch/patience/seed reach ultralytics",
      (seen4["epochs"], seen4["imgsz"], seen4["batch"], seen4["patience"], seen4["seed"]) == (3, 320, 2, 1, 7),
      str([seen4[k] for k in ("epochs", "imgsz", "batch", "patience", "seed")]))
check("overrides reach ultralytics and win", seen4["mosaic"] == 0.0 and seen4["workers"] == 2)
check("state-safe pins survive an unrelated override", seen4["flipud"] == 0.0 and seen4["hsv_v"] == 0.20)

# Preconditions fail loudly and early, not 150 epochs in.
with tempfile.TemporaryDirectory() as td:
    empty = os.path.join(td, "empty")
    os.makedirs(empty)
    try:
        gen_dataset.train_scene("X", empty, os.path.join(td, "w.pt"))
        check("missing data.yaml raises", False, "no exception")
    except FileNotFoundError as exc:
        check("missing data.yaml raises FileNotFoundError naming the file", "data.yaml" in str(exc))
    except Exception as exc:
        check("missing data.yaml raises FileNotFoundError", False, type(exc).__name__)

    noimg = os.path.join(td, "noimg")
    os.makedirs(noimg)
    with open(os.path.join(noimg, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("nc: 0\n")
    try:
        gen_dataset.train_scene("X", noimg, os.path.join(td, "w.pt"))
        check("empty image tree raises before training", False, "no exception")
    except RuntimeError as exc:
        check("empty image tree raises before training, naming the cause", "no images" in str(exc).lower())
    except Exception as exc:
        check("empty image tree raises RuntimeError", False, type(exc).__name__)

check("count_images walks both splits", yr.count_images(os.path.join(HERE, "does-not-exist")) == 0)


# ----------------------------------------------------------------------------------------------------------------------
# 5. Metrics — per-class quality is reported, with the right index convention.
# ----------------------------------------------------------------------------------------------------------------------
section("5. metric extraction")

m = gen_dataset._train_metrics(_FakeMetrics(), _FakeYOLO("x"))
check("summary scalars extracted", m["summary"]["metrics/mAP50(B)"] == 0.75)
check("per-class AP50-95 keyed by class NAME", set(m["per_class_ap"]) == {"Fridge_open", "StoveBurner_on"},
      str(m.get("per_class_ap")))
check("per-class AP50-95 read by CLASS ID from .maps",
      m["per_class_ap"]["Fridge_open"] == 0.71 and m["per_class_ap"]["StoveBurner_on"] == 0.22,
      str(m["per_class_ap"]))
check("per-class AP50 read POSITIONALLY from .box.ap50",
      m["per_class_ap50"] == {"Fridge_open": 0.90, "StoveBurner_on": 0.40}, str(m.get("per_class_ap50")))
check("per-class recall reported (separates 'never found' from 'found, mislabelled')",
      m["per_class_recall"] == {"Fridge_open": 0.88, "StoveBurner_on": 0.30}, str(m.get("per_class_recall")))
check("a class absent from val is not reported", "Fridge_closed" not in m["per_class_ap"])
check("no error on the happy path", "error" not in m, str(m.get("error")))


class _Bare:
    results_dict = {"metrics/mAP50(B)": 0.5, "task": "detect"}


bare = gen_dataset._train_metrics(_Bare(), _FakeYOLO("x"))
check("non-numeric summary entries are dropped", "task" not in bare["summary"])
check("missing per-class data degrades gracefully", "per_class_ap" not in bare and "error" not in bare,
      str(bare))

del sys.modules["ultralytics"]

print(f"\n{'=' * 70}\n{PASS}/{PASS + FAIL} checks passed" + (f" — {FAIL} FAILED" if FAIL else " — ALL PASS"))
sys.exit(1 if FAIL else 0)
