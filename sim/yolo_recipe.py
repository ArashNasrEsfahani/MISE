"""Which YOLO checkpoint the per-scene detector fine-tunes, and with what training recipe (Phase 4, detector quality).

:mod:`gen_dataset` owns the *data* (auto-labelled frames -> ``data.yaml``); this module owns the two decisions that
actually determine whether the resulting detector works: **which pretrained backbone we start from** and **what
hyper-parameters we hand ultralytics**. It is kept separate so both are inspectable and unit-testable without
importing ``ultralytics``/``torch`` (the self-test ``sim/_detector_selftest.py`` runs on bare Windows python).

Stdlib-only at module top, like :mod:`perception`/:mod:`gen_dataset` — ``import yolo_recipe`` costs nothing.

Why this file exists
--------------------
Until now ``train_scene`` fine-tuned ``yolov8n.pt`` with stock ultralytics defaults. Both halves were wrong for
this problem:

* **yolov8n is the nano model** — the smallest, weakest member of the family (3.2 M params, 37.3 COCO mAP50-95).
  It was designed for 80 *coarse, visually distinct* COCO classes. Our task is the opposite: ~50-60 classes that
  come in **confusable pairs** (``Fridge_open`` / ``Fridge_closed``, ``Apple_sliced`` / ``Apple_whole``,
  ``StoveBurner_on`` / ``StoveBurner_off``), where the two members share an object identity and differ only by a
  local geometry or appearance cue. Telling those apart is a feature-*resolution* problem, and nano's backbone is
  capacity-starved for it.
* The stock augmentation recipe **corrupts state labels** — see :data:`STATE_SAFE_AUG` below.

Run ``python sim/yolo_recipe.py`` to print the resolved base, the exact first-train download manifest (file
names + sizes) and the full recipe. ``sim/_detector_selftest.py`` is the headless gate over all of it.
"""
from __future__ import annotations

import os

# ----------------------------------------------------------------------------------------------------------------------
# Base checkpoint catalog.
#
# The CURATED SUBSET of detection checkpoints considered for this task — not everything 8.4.83 can fetch. ``coco_map``
# is the published COCO val2017 mAP50-95 at 640 px, ``download_mb`` the size of the file pulled from the ultralytics
# GitHub release on first use. These are documentation *and* the source of the download manifest printed by
# ``python sim/yolo_recipe.py``.
#
# NOT LISTED, deliberately: the installed 8.4.83 also ships and can fetch ``yolo26{n,s,m,l,x}.pt`` (its own AMP check
# runs yolo26n — see :data:`AMP_CHECKPOINT`), and yolo26 is the newer architecture. It is absent here because nobody
# on this project has measured it on this dataset and its published figures are not in hand; entering guessed numbers
# would put a fabricated COCO mAP into the thesis's own documentation. ``THOR_YOLO_BASE=yolo26s`` still WORKS — an
# uncatalogued name is passed through to ultralytics untouched — but it bypasses :func:`checkpoint_path`, so the
# download lands in the CWD instead of the cache. Benchmark it against yolo11s before adopting it, then add the row.
# ----------------------------------------------------------------------------------------------------------------------
BASES = {
    # YOLO11 (2024 architecture: C3k2 blocks + a C2PSA attention stage). Strictly better accuracy-per-FLOP than v8.
    "yolo11n.pt":  {"params_m": 2.6,  "gflops": 6.5,   "coco_map": 39.5, "download_mb": 5.4},
    "yolo11s.pt":  {"params_m": 9.4,  "gflops": 21.5,  "coco_map": 47.0, "download_mb": 19.2},
    "yolo11m.pt":  {"params_m": 20.1, "gflops": 68.0,  "coco_map": 51.5, "download_mb": 38.8},
    "yolo11l.pt":  {"params_m": 25.3, "gflops": 86.9,  "coco_map": 53.4, "download_mb": 49.0},
    "yolo11x.pt":  {"params_m": 56.9, "gflops": 194.9, "coco_map": 54.7, "download_mb": 109.3},
    # YOLOv8 — kept reachable so the Phase-4 results already in the thesis stay reproducible.
    "yolov8n.pt":  {"params_m": 3.2,  "gflops": 8.7,   "coco_map": 37.3, "download_mb": 6.2},
    "yolov8s.pt":  {"params_m": 11.2, "gflops": 28.6,  "coco_map": 44.9, "download_mb": 21.5},
    "yolov8m.pt":  {"params_m": 25.9, "gflops": 78.9,  "coco_map": 50.2, "download_mb": 49.7},
}

#: The nano checkpoint ultralytics' own ``check_amp()`` fetches and runs before the first CUDA train. It is NOT the
#: base we fine-tune and NOT a knob — it is version-specific machinery, hard-coded in 8.4.83's
#: ``ultralytics/utils/checks.py``. Named here so :func:`download_manifest` tells the truth about what the network
#: round actually pulls; ``overrides={'amp': False}`` skips the fetch entirely. Size measured from the copy 8.4.83
#: already left in the repo root during the June Phase-4 run (5 544 453 bytes).
AMP_CHECKPOINT = "yolo26n.pt"
AMP_CHECKPOINT_MB = 5.3

# ----------------------------------------------------------------------------------------------------------------------
# THE DEFAULT BASE — yolo11s.
#
# The regime this has to be right for: an 8 GB laptop RTX 4060 **shared with a live AI2-THOR controller** (Unity holds
# ~2-2.5 GB), a per-scene dataset of a few hundred to a couple thousand 640x640 frames (FloorPlan1_physics: 268 train /
# 65 val), and ~53 state-aware classes.
#
#   * Why not stay on nano (yolov8n / yolo11n). On a 268-image set almost all of the final accuracy comes from the
#     *pretrained representation*, not from what 268 images can teach; we are re-fitting a head and lightly adapting a
#     backbone. yolo11s brings +9.7 COCO mAP50-95 over yolov8n (47.0 vs 37.3) for that representation. And the part of
#     our task that fails today — separating a class PAIR that shares an object identity — is exactly what a 2.6-3.2 M
#     param backbone cannot encode across 53 classes.
#   * Why not go bigger (yolo11m / yolo11l). Three independent reasons, any one of which is disqualifying:
#       - VRAM: 11m at 640/batch 16 wants ~7-8 GB, which does not coexist with the simulator; the user would have to
#         stop the console for every retrain, and the console is the thesis demo.
#       - Data: 20 M params fitted on <300 images is a textbook overfit; the val split here is *frames sampled from the
#         same scan trajectory*, so val mAP would flatter it rather than catch it.
#       - Latency: the detector runs per-frame inside the live loop (``perception.Detector.detect``) on the SAME GPU as
#         the simulator. 11s is ~2-3 ms/frame on a 4060; 11m is 2-3x that plus the memory pressure.
#   * Why 11s specifically and not v8s. Same size tier (9.4 M vs 11.2 M params), fewer FLOPs (21.5 vs 28.6), and +2.1
#     COCO mAP. There is no axis on which v8s wins.
#
# Bigger is not automatically better here — the ceiling is set by 268 images and a shared 8 GB card, and yolo11s is the
# largest checkpoint that respects both while still tripling the pretrained-representation quality.
# ----------------------------------------------------------------------------------------------------------------------
DEFAULT_BASE = "yolo11s.pt"

#: Environment override for the base checkpoint (``THOR_YOLO_BASE=yolo11m`` or a path to a local ``.pt``). Lets the
#: webapp's Train button and the WSL scripts be re-pointed without editing code; an explicit ``base=`` argument still
#: wins, so ``base='yolov8n.pt'`` continues to reproduce the published Phase-4 numbers.
BASE_ENV = "THOR_YOLO_BASE"

#: Where downloaded base checkpoints are kept, so the ~19 MB fetch happens once per machine instead of once per CWD.
#: Defaults to ``~/.cache/thor-yolo``; deliberately OUTSIDE the repo (the repo path contains an apostrophe, which
#: ultralytics strips from absolute weight paths — see ``perception.Detector.__init__``).
CACHE_ENV = "THOR_YOLO_CACHE"

# ----------------------------------------------------------------------------------------------------------------------
# Recipe defaults. Every value that departs from the ultralytics 8.4.83 default carries the reason.
# ----------------------------------------------------------------------------------------------------------------------
#: 150, up from the previous hard-coded 50. The last run (yolov8n, 50 epochs, 268 images) was still improving when the
#: schedule ended: box_loss 0.56 and falling, mAP50 0.732. 50 epochs was a budget, not convergence. Fine-tuning a
#: larger backbone on a small set needs a longer schedule; :data:`DEFAULT_PATIENCE` stops it early when it plateaus, so
#: this is a ceiling rather than a cost.
DEFAULT_EPOCHS = 150

#: 50, down from ultralytics' 100. With a 150-epoch ceiling, patience=100 would never fire — early stopping would be
#: decorative. 50 is long enough to ride out the plateau that follows the mosaic-close step (see ``close_mosaic``) and
#: short enough to reclaim real time. ultralytics keeps ``best.pt`` regardless of where it stops, and we copy best.pt.
DEFAULT_PATIENCE = 50

#: 640: the scan renders 600x600 (webapp) or 640x640 (``_phase4_fullscan``) square frames, so 640 matches the source.
#: Going higher only upsamples pixels the renderer never produced; going lower would drop the small state cues
#: (a StoveKnob, a lit burner, liquid in a mug) below the detector's stride-8 feature map.
DEFAULT_IMGSZ = 640

#: 8, half the ultralytics default of 16, because the GPU is SHARED with a live AI2-THOR controller. yolo11s at
#: 640/batch 8 with AMP fits in ~2.5-3 GB, which coexists with Unity's ~2-2.5 GB on an 8 GB card; batch 16 does not.
#: This costs nothing statistically: ultralytics accumulates gradients to its nominal batch of 64
#: (``accumulate = round(64 / batch)``), so the effective optimizer batch is unchanged.
#:
#: NOT AutoBatch (``batch=-1``). AutoBatch probes free VRAM once, at start, and targets a fraction of *total* memory —
#: with the simulator already resident and free to allocate more mid-run, it picks a batch that OOMs later.
#: Stopping the console first and passing ``batch=16`` is the right move for a standalone retrain.
DEFAULT_BATCH = 8

#: Fixed seed + ``deterministic=True`` so a rerun of the same scan reproduces the same detector — required for the
#: ablation tables to mean anything. These match the ultralytics defaults today; they are passed explicitly so a
#: future default flip cannot silently un-reproduce the thesis results.
DEFAULT_SEED = 0

#: Cache decoded images in RAM below this many images. The dataset lives under ``/mnt/d`` (a drvfs mount), where
#: per-epoch JPEG reads are the slow part of a small-dataset epoch; 1000 x 640x640x3 is ~1.2 GB, which is affordable
#: inside WSL. Above the threshold we fall back to on-disk reads rather than risk the WSL memory ceiling.
RAM_CACHE_MAX_IMAGES = 1000

# ----------------------------------------------------------------------------------------------------------------------
# Augmentation — the half of the recipe that was actively HURTING us.
#
# Our labels are ``<objectType>_<state>``. An augmentation that changes the pixels carrying the *state* cue without
# changing the label is not regularization, it is label noise: it teaches the detector that the cue is irrelevant.
# Walking the ultralytics defaults against :data:`perception.STATE_AXES`:
#
#   isToggled (StoveBurner/Faucet/Toaster/CoffeeMachine/LightSwitch/Television -> on|off)
#       The cue is a GLOW / brightness / colour difference and nothing else. ultralytics' default ``hsv_v=0.4`` jitters
#       value by +-40% and ``hsv_s=0.7`` saturation by +-70% — enough to render an "off" burner brighter than a lit one
#       in the same epoch. This is the single most label-corrupting default for our class set. -> hsv_v 0.4 -> 0.2,
#       hsv_s 0.7 -> 0.3.
#   isCooked (colour/texture shift) and isFilledWithLiquid (a dark liquid region inside a vessel)
#       Both are low-contrast colour cues; the same value/saturation jitter washes them out. ``hsv_h`` carries the
#       raw-vs-cooked hue shift, so it is tightened 0.015 -> 0.010 rather than left alone.
#   isFilledWithLiquid also depends on GRAVITY — liquid sits at the bottom of the vessel. Any vertical flip or large
#       rotation produces a physically impossible frame and moves the cue. -> ``flipud`` and ``degrees`` pinned to 0.
#   isOpen / isSliced / isBroken are geometry cues, so geometric jitter that preserves gravity is safe and useful:
#       ``scale=0.5`` (the agent genuinely views objects from 0.5x-2x distance) and ``translate=0.1`` are kept.
#
# Horizontal flip (``fliplr=0.5``) is KEPT: no class in the vocabulary is chiral — a mirrored open fridge is still an
# open fridge — so it doubles the effective dataset for free. This is the one flip that is safe.
#
# ``mixup``/``cutmix``/``copy_paste``/``bgr``/``erasing`` are pinned to 0 even though they are already 0 today, because
# each is specifically dangerous here and a future ultralytics default change must not be able to enable them:
# mixup/cutmix alpha-blend two frames, ghosting a *closed* fridge over an *open* one under a single label; bgr swaps
# channels (a red tomato becomes blue, destroying every colour-based state cue); erasing can blank exactly the patch
# that carries the cue.
#
# ``mosaic`` is reduced 1.0 -> 0.5 rather than removed. Mosaic tiles four frames into one 640 canvas, so every object
# is seen at roughly half scale with synthetic seams — on a fixed indoor scene with fine state cues that manufactures
# unrealistically small objects, but it remains the strongest regularizer available on a 268-image set. At 0.5, half
# the batches are real frames at deployment scale and half are mosaics. ``close_mosaic`` is raised 10 -> 20 so the tail
# of a full-length run trains purely on the deployment distribution. (Caveat, stated honestly: if early stopping fires
# before ``epochs - close_mosaic``, that clean tail never happens. That is the accepted cost of patience=50.)
# ----------------------------------------------------------------------------------------------------------------------
STATE_SAFE_AUG = {
    "hsv_h": 0.010,       # tightened: hue carries the raw/cooked and liquid-colour cues
    "hsv_s": 0.30,        # was 0.7 — saturation jitter washes out cooked/filled cues
    "hsv_v": 0.20,        # was 0.4 — value jitter flips the on/off (glow) axis outright
    "degrees": 0.0,       # gravity-dependent cues (liquid level) must not rotate
    "translate": 0.10,    # safe: pure framing jitter
    "scale": 0.50,        # safe and wanted: the agent really does view objects at 0.5x-2x distance
    "shear": 0.0,         # the camera is a fixed-FOV pinhole; sheared views are unreachable in deployment
    "perspective": 0.0,   # ditto
    "flipud": 0.0,        # would invert gravity (and the kitchen)
    "fliplr": 0.5,        # KEPT: no chiral classes, so mirroring is state-preserving and doubles the data
    "bgr": 0.0,           # channel swap destroys every colour-based state cue
    "mosaic": 0.50,       # was 1.0 — see the note above
    "close_mosaic": 20,   # was 10 — longer clean tail at deployment scale
    "mixup": 0.0,         # blends two frames -> ghosts one state over another under one label
    "cutmix": 0.0,        # same failure mode, patch-wise
    "copy_paste": 0.0,    # needs segmentation masks; pastes objects into impossible contexts
    "erasing": 0.0,       # classification-only in ultralytics today, but pinned: it can blank the state cue
}

#: Image file extensions counted by :func:`count_images` (ultralytics' own detection-image set, lower-cased).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ----------------------------------------------------------------------------------------------------------------------
# Base selection.
# ----------------------------------------------------------------------------------------------------------------------
def resolve_base(base=None, env=None) -> str:
    """Return the base checkpoint to fine-tune. Precedence: explicit ``base`` > ``$THOR_YOLO_BASE`` > :data:`DEFAULT_BASE`.

    ``env`` defaults to :data:`os.environ` (injectable for tests). A bare stem (``'yolo11m'``) gains its ``.pt``; a
    value that already names a file (``'yolo11m.pt'``, ``'/path/to/custom.pt'``, ``'yolo11s.yaml'`` for from-scratch)
    is passed through untouched, so an arbitrary local checkpoint is always reachable.
    """
    env = os.environ if env is None else env
    picked = base if base is not None and str(base).strip() else None
    if picked is None:
        raw = (env.get(BASE_ENV) or "").strip()
        picked = raw or DEFAULT_BASE
    picked = str(picked).strip()
    if not os.path.splitext(picked)[1]:
        picked += ".pt"       # 'yolo11m' -> 'yolo11m.pt'; ultralytics only resolves the suffixed asset name
    return picked


def base_info(name: str) -> dict:
    """Catalog entry for ``name`` (``{params_m, gflops, coco_map, download_mb}``), or ``{}`` for a custom checkpoint."""
    return dict(BASES.get(os.path.basename(str(name)), {}))


def cache_dir(env=None) -> str:
    """Directory downloaded base checkpoints are kept in (``$THOR_YOLO_CACHE`` or ``~/.cache/thor-yolo``)."""
    env = os.environ if env is None else env
    return (env.get(CACHE_ENV) or "").strip() or os.path.join(os.path.expanduser("~"), ".cache", "thor-yolo")


def checkpoint_path(name: str, env=None) -> str:
    """Absolute path under :func:`cache_dir` to load ``name`` from, or ``name`` unchanged when caching can't apply.

    Handing ultralytics an absolute path whose *basename* is a known release asset makes it download to that path, so
    the fetch happens once per machine rather than once per working directory (today a bare name lands a stray
    ``yolo11s.pt`` in whatever CWD the webapp was started from). Returns ``name`` untouched when it is already a path,
    is not a known asset, or when the cache directory is unusable — including the apostrophe case: ultralytics strips
    apostrophes from absolute weight paths, so a cache dir under ``Master's Project`` would silently break the load.
    """
    if os.path.dirname(str(name)):
        return str(name)                       # caller gave a real path already
    if os.path.basename(str(name)) not in BASES:
        return str(name)                       # unknown asset: let ultralytics resolve it however it likes
    root = cache_dir(env)
    if "'" in os.path.abspath(root):
        return str(name)
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        return str(name)
    return os.path.join(root, str(name))


def download_manifest(base=None, env=None) -> list:
    """What a first train on a fresh machine pulls over the network, as ``[{file, mb, why}]``.

    One network round is required the first time; every later train reuses :func:`cache_dir` and is fully offline.
    """
    name = resolve_base(base, env=env)
    info = base_info(name)
    items = []
    if info:
        items.append({"file": name, "mb": info["download_mb"],
                      "why": f"base checkpoint being fine-tuned ({info['params_m']}M params, "
                             f"COCO mAP50-95 {info['coco_map']})"})
    else:
        items.append({"file": name, "mb": None, "why": "custom checkpoint — supplied by you, not downloaded"})
    # ultralytics' check_amp() validates mixed precision on CUDA by running a nano model, and fetches it if absent.
    # The checkpoint it uses is version-specific: 8.4.83's checks.py hard-codes yolo26n.pt (read from the installed
    # source, and confirmed by the stray yolo26n.pt the June Phase-4 train left in the repo root — 5.29 MB on disk).
    if os.path.basename(name) != AMP_CHECKPOINT:
        items.append({"file": AMP_CHECKPOINT, "mb": AMP_CHECKPOINT_MB,
                      "why": "ultralytics check_amp() sanity test on the first CUDA train (skipped by amp=False)"})
    items.append({"file": "Arial.ttf", "mb": 0.8,
                  "why": "label font for the plotted curves/confusion matrix (plots=True)"})
    return items


# ----------------------------------------------------------------------------------------------------------------------
# Recipe.
# ----------------------------------------------------------------------------------------------------------------------
def count_images(dataset_dir: str) -> int:
    """Number of image files under ``dataset_dir/images/**`` (used to size the RAM cache); 0 if the tree is missing."""
    root = os.path.join(dataset_dir, "images")
    n = 0
    for cur, _dirs, files in os.walk(root):
        n += sum(1 for f in files if f.lower().endswith(IMAGE_EXTS))
    return n


def train_kwargs(n_images=None, epochs=None, imgsz=None, batch=None, patience=None, seed=DEFAULT_SEED,
                 cache=None, overrides=None) -> dict:
    """Build the full ultralytics ``model.train(**kwargs)`` mapping for a per-scene detector.

    ``None`` for any knob means "use the documented default". Everything the recipe deliberately sets is passed
    explicitly (including values that merely match today's ultralytics defaults) so the run is self-describing and
    immune to upstream default changes; the returned dict is echoed back by
    :func:`gen_dataset.train_scene` under ``'recipe'``. ``overrides`` is applied last and wins, so a caller can still
    experiment without editing this file.

    Note the optimizer is left at ``'auto'`` and ``lr0`` is deliberately NOT set: with ``optimizer='auto'`` ultralytics
    picks AdamW plus a dataset-size-derived learning rate for small runs like ours and overrides any ``lr0`` we pass,
    so setting one would be a silently-ignored lie in the recorded recipe.
    """
    n_images = 0 if n_images is None else int(n_images)
    if cache is None:
        cache = "ram" if 0 < n_images <= RAM_CACHE_MAX_IMAGES else False

    kw = {
        "epochs": DEFAULT_EPOCHS if epochs is None else int(epochs),
        "patience": DEFAULT_PATIENCE if patience is None else int(patience),
        "imgsz": DEFAULT_IMGSZ if imgsz is None else int(imgsz),
        "batch": DEFAULT_BATCH if batch is None else batch,
        "cache": cache,
        "cos_lr": True,        # smooth decay over a long fine-tune; the default linear schedule leaves the LR high
                               # exactly where a patience-triggered stop is most likely to land
        "amp": True,           # halves activation memory — non-negotiable while the simulator holds the same card
        "seed": int(seed),
        "deterministic": True,
        "val": True,
        "plots": True,         # the run's confusion matrix / PR curves are copied back into datasets/<scene>/runs and
                               # are what the thesis reports per-class detector quality from
        "verbose": True,
    }
    kw.update(STATE_SAFE_AUG)
    if overrides:
        kw.update(overrides)
    return kw


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    base = resolve_base()
    info = base_info(base)
    print(f"base checkpoint : {base}" + (f"  ({info['params_m']}M params, {info['gflops']} GFLOPs, "
                                         f"COCO mAP50-95 {info['coco_map']})" if info else "  (custom)"))
    print(f"override with   : {BASE_ENV}=yolo11m   (or any .pt path)")
    print(f"cache directory : {cache_dir()}")
    print("\nfirst-train downloads (needs network ONCE; later trains are offline):")
    total = 0.0
    for it in download_manifest():
        mb = it["mb"]
        total += mb or 0.0
        print(f"  {it['file']:14s} {'?' if mb is None else f'{mb:6.1f} MB'}   {it['why']}")
    print(f"  {'':14s} {total:6.1f} MB   total")
    print("\nrecipe (dataset of 333 images):")
    for k, v in sorted(train_kwargs(n_images=333).items()):
        print(f"  {k:14s} {v}")


# ===================================================================== PUBLIC API ====================================
# DEFAULT_BASE = 'yolo11s.pt'   BASE_ENV = 'THOR_YOLO_BASE'   CACHE_ENV = 'THOR_YOLO_CACHE'
# DEFAULT_EPOCHS = 150   DEFAULT_PATIENCE = 50   DEFAULT_IMGSZ = 640   DEFAULT_BATCH = 8   DEFAULT_SEED = 0
# BASES: {checkpoint_name: {params_m, gflops, coco_map, download_mb}} — the fetchable detection checkpoints.
# STATE_SAFE_AUG: the augmentation block, chosen so no augmentation can change the pixels carrying a STATE label.
#
# resolve_base(base=None, env=None) -> str
#     Explicit base > $THOR_YOLO_BASE > DEFAULT_BASE; a bare stem gains '.pt'; paths pass through.
# base_info(name) -> dict            Catalog entry, or {} for a custom checkpoint.
# cache_dir(env=None) -> str         $THOR_YOLO_CACHE or ~/.cache/thor-yolo.
# checkpoint_path(name, env=None) -> str
#     Absolute path under cache_dir() to download/load a known asset from, else `name` unchanged (custom path,
#     unknown asset, unusable cache dir, or an apostrophe in the cache path — which ultralytics mangles).
# download_manifest(base=None, env=None) -> [{file, mb, why}]
#     What a first train pulls over the network (base + yolo11n.pt for check_amp + Arial.ttf).
# count_images(dataset_dir) -> int   Images under dataset_dir/images/** (sizes the RAM cache).
# train_kwargs(n_images=None, epochs=None, imgsz=None, batch=None, patience=None, seed=0, cache=None,
#              overrides=None) -> dict
#     The full model.train(**kwargs) mapping. None = documented default; `overrides` applied last and wins.
