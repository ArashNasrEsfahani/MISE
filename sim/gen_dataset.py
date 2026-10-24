"""Auto-label AI2-THOR frames into a YOLO dataset and train a **per-scene**, **state-aware** detector (Phase 4).

The thesis uses a per-scene detector: on entering a `FloorPlan` the agent (re)scans the room, this module turns each
egocentric RGB frame into a YOLO-format training example using AI2-THOR's *ground-truth* instance masks (so no human
boxing is ever needed), and :func:`train_scene` fits ``weights/<scene>.pt`` on the result. Classes are the
**stateful** labels from :mod:`perception` (``Fridge_open`` vs ``Fridge_closed``, ``Apple_sliced`` vs ``Apple_whole``,
…), so a single detection answers both *what* and *what state*.

Pipeline per scene::

    class_to_id = {}                                   # one stable name -> id map for the whole scene
    # (recommended) pre-seed a fully scene-deterministic ordering up front:
    for n in perception.scene_class_names(agent.objects()):
        class_to_id.setdefault(n, len(class_to_id))
    for i, _ in enumerate(agent.look_around()):        # spin in place, capturing frames
        gen_dataset.capture_frame(agent, scene, out_dir, split='train', class_to_id=class_to_id, index=i)
    gen_dataset.build_data_yaml(out_dir, class_to_id)
    gen_dataset.train_scene(scene, out_dir, 'weights/%s.pt' % scene)

The detector's *quality* knobs — which pretrained checkpoint we fine-tune and with what hyper-parameters — live in
the sibling :mod:`yolo_recipe`, so they can be inspected and unit-tested without ultralytics installed. The default
base is ``yolo11s.pt`` (not the old ``yolov8n.pt``); ``$THOR_YOLO_BASE`` overrides it and an explicit ``base=``
argument beats both. ``python sim/yolo_recipe.py`` prints the resolved base, the one-time download manifest
and the recipe; ``python sim/_detector_selftest.py`` is the headless gate.

Stdlib-/numpy-first per CLAUDE.md: the only module-top imports are stdlib (``os``) plus the light, stdlib-only
siblings :mod:`perception` and :mod:`yolo_recipe`. ``numpy``/``PIL`` are imported lazily inside :func:`capture_frame`,
and ``ultralytics``/``torch`` only inside :func:`train_scene`, so ``import gen_dataset`` succeeds with none of the
heavy deps installed (mirrors ``perception.py``/``rag.py``/``memory.py``).
"""
from __future__ import annotations

import os

import perception
import yolo_recipe

# A box that touches a frame edge and is no thicker than this (px) perpendicular to that edge is a partial-object
# *sliver* (a thin strip clipped at the border) — its bbox is uninformative, so it is dropped from the dataset.
EDGE_SLIVER_PX = 3

# Default JPEG quality for written frames (high; we want the detector to learn from near-lossless images).
JPEG_QUALITY = 95


# ----------------------------------------------------------------------------------------------------------------------
# Internal helpers (pure, numpy-light).
# ----------------------------------------------------------------------------------------------------------------------
def _instance_masks(agent) -> dict:
    """``{objectId: bool HxW mask}`` from the agent, tolerant of ``instance_masks`` being a property or a method."""
    m = getattr(agent, "instance_masks", None)
    if callable(m):
        m = m()
    return m or {}


def _is_edge_sliver(box, w: int, h: int, thick: int = EDGE_SLIVER_PX) -> bool:
    """True if ``box=(x1,y1,x2,y2)`` hugs a frame edge as a thin sliver (clipped partial object)."""
    x1, y1, x2, y2 = box
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    touch_vert = x1 <= 0 or x2 >= w - 1        # against the left/right border
    touch_horiz = y1 <= 0 or y2 >= h - 1       # against the top/bottom border
    return (touch_vert and bw <= thick) or (touch_horiz and bh <= thick)


def _frame_uid(frame, index):
    """Reproducible per-frame id: content hash (deterministic, collision-resistant), index-prefixed if given."""
    import hashlib
    import numpy as np
    h = hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()[:10]
    return h if index is None else f"{int(index):06d}_{h}"


# ----------------------------------------------------------------------------------------------------------------------
# Capture — one frame -> one (image, label) pair.
# ----------------------------------------------------------------------------------------------------------------------
def capture_frame(agent, scene: str, out_dir: str, split: str = "train", class_to_id=None,
                  min_area: int = 400, min_visible_frac: float = 0.0, index=None, memory=None,
                  source: str = "environment") -> dict:
    """Auto-label the agent's current RGB frame into a YOLO example under ``out_dir``.

    Reads the agent's instance masks + object metadata; for each VISIBLE object with a mask it computes a tight bbox
    (:func:`perception.box_from_mask`), drops it if too small (``min_area``, inclusive-corner pixel area), an
    edge-hugging sliver, or under-filled (``min_visible_frac`` = mask pixels / bbox area), and labels the survivor
    with its stateful class (:func:`perception.class_name`).

    ``class_to_id`` (a ``{class_name: id}`` dict, created empty if ``None``) is **mutated in place**: any new class is
    appended in deterministic (sorted) order so ids stay stable across calls. The RGB frame is written to
    ``out_dir/images/<split>/<scene>_<uid>.jpg`` and the YOLO labels to ``out_dir/labels/<split>/<scene>_<uid>.txt``
    (``<uid>`` is a content hash, optionally prefixed by ``index`` for ordering). If a ``Memory`` is passed, the
    detections are also recorded via :meth:`memory.add_perception_label` (auditable, ``source`` provenance).

    Returns ``{'written': bool, 'image_path': str|None, 'n_boxes': int, 'classes': [str]}``. A frame yielding no
    usable boxes writes nothing and returns ``written=False, image_path=None``.
    """
    import numpy as np
    from PIL import Image

    if class_to_id is None:
        class_to_id = {}

    frame = agent.frame
    if frame is None:
        return {"written": False, "image_path": None, "n_boxes": 0, "classes": []}
    frame = np.asarray(frame)
    h, w = frame.shape[0], frame.shape[1]

    masks = _instance_masks(agent)

    # Collect surviving (class_name, box) pairs for the visible, masked, non-trivial objects in view.
    kept: list = []
    for o in agent.visible_objects():
        mask = masks.get(o.get("objectId"))
        if mask is None:
            continue
        box = perception.box_from_mask(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area < min_area:
            continue
        if _is_edge_sliver(box, w, h):
            continue
        if min_visible_frac > 0.0:
            mask_px = int(np.asarray(mask, dtype=bool).sum())
            if mask_px / float(area) < min_visible_frac:
                continue
        kept.append((perception.class_name(o), box))

    if not kept:
        return {"written": False, "image_path": None, "n_boxes": 0, "classes": []}

    # Extend the class->id map deterministically (sorted) so ids are stable regardless of object iteration order.
    for cls in sorted({c for c, _ in kept}):
        if cls not in class_to_id:
            class_to_id[cls] = len(class_to_id)

    lines = [perception.yolo_line(class_to_id[c], box, w, h) for c, box in kept]

    uid = _frame_uid(frame, index)
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    image_path = os.path.join(img_dir, f"{scene}_{uid}.jpg")
    label_path = os.path.join(lbl_dir, f"{scene}_{uid}.txt")

    Image.fromarray(frame).save(image_path, quality=JPEG_QUALITY)
    with open(label_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    classes = sorted({c for c, _ in kept})
    if memory is not None:
        boxes = [{"cls": c, "bbox_xyxy": list(box)} for c, box in kept]
        memory.add_perception_label(
            image_path, scene=scene, boxes=boxes, source=source,
            why=f"auto-labeled {len(kept)} box(es) from instance masks for {scene} ({split})")

    return {"written": True, "image_path": image_path, "n_boxes": len(kept), "classes": classes}


# ----------------------------------------------------------------------------------------------------------------------
# Dataset manifest.
# ----------------------------------------------------------------------------------------------------------------------
def build_data_yaml(out_dir: str, class_to_id) -> str:
    """Write ``out_dir/data.yaml`` (ultralytics format) from a ``{class_name: id}`` map; returns the yaml path.

    Emits ``path`` (absolute ``out_dir``), ``train: images/train``, ``val: images/val``, ``nc`` and the ``id -> name``
    ``names`` mapping (sorted by id, the inverse of ``class_to_id``).
    """
    id_to_name = {i: n for n, i in class_to_id.items()}
    lines = [
        f"path: {os.path.abspath(out_dir)}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(id_to_name)}",
        "names:",
    ]
    for i in sorted(id_to_name):
        lines.append(f"  {i}: {id_to_name[i]}")

    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return yaml_path


# ----------------------------------------------------------------------------------------------------------------------
# Train — fit a per-scene YOLO model and stash its best weights at weights/<scene>.pt.
# ----------------------------------------------------------------------------------------------------------------------
def _find_best(results, model) -> str | None:
    """Locate the ``best.pt`` produced by an ultralytics training run (via results.save_dir or trainer.best)."""
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        cand = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.exists(cand):
            return cand
    best = getattr(getattr(model, "trainer", None), "best", None)
    if best and os.path.exists(str(best)):
        return str(best)
    return None


def _train_metrics(results, model) -> dict:
    """Best-effort extraction of per-class AP + summary scalars from an ultralytics ``DetMetrics`` object.

    ``model.train()`` returns a ``DetMetrics`` (verified on ultralytics 8.4.83) exposing ``.results_dict`` (a
    ``{str: float}`` of precision/recall/mAP50/mAP50-95/fitness) and ``.maps`` (a numpy array of per-class mAP50-95
    indexed by class id) + ``.ap_class_index``. If ``results`` lacks them, fall back to ``model.metrics``.

    Returns ``per_class_ap`` (AP50-95), ``per_class_ap50`` and ``per_class_recall``. The last two matter for
    diagnosing a *state-aware* detector: mAP50-95 conflates localisation with recognition, whereas an agent that
    cannot find the fridge at all fails differently from one that finds it and calls it closed when it is open —
    per-class recall separates "never detected" from "detected, mislabelled", and AP50 isolates recognition from
    box-tightness. Both are read positionally off ``DetMetrics.box`` (aligned with ``ap_class_index``), unlike
    ``.maps`` which is indexed by class id — mixing the two conventions up silently misattributes every score.
    """
    out: dict = {}
    src = results if hasattr(results, "results_dict") else (getattr(model, "metrics", None) or results)

    # Summary scalars. NB: never do `np_array or default` (raises "ambiguous truth value") — test None + len.
    try:
        rd = getattr(src, "results_dict", None)
        if isinstance(rd, dict):
            out["summary"] = {k: float(v) for k, v in rd.items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    except Exception as exc:
        out["error"] = str(exc)

    # Per-class AP50-95: maps[class_id] -> AP, for the class ids listed in ap_class_index.
    try:
        maps = getattr(src, "maps", None)
        idx = getattr(src, "ap_class_index", None)
        names = dict(getattr(model, "names", {}) or {})
        if maps is not None and idx is not None:
            maps, idx = list(maps), list(idx)
            per_class = {}
            for ci in idx:
                ci = int(ci)
                ap = maps[ci] if ci < len(maps) else None
                per_class[names.get(ci, str(ci))] = None if ap is None else float(ap)
            if per_class:
                out["per_class_ap"] = per_class

            # AP50 and recall live on the nested `box` Metric and are indexed by POSITION in ap_class_index.
            box = getattr(src, "box", None)
            for key, attr in (("per_class_ap50", "ap50"), ("per_class_recall", "r")):
                vals = getattr(box, attr, None) if box is not None else None
                if vals is None:
                    continue
                vals = list(vals)
                got = {names.get(int(ci), str(int(ci))): float(vals[i])
                       for i, ci in enumerate(idx) if i < len(vals)}
                if got:
                    out[key] = got
    except Exception as exc:
        out.setdefault("error", str(exc))

    return out


def train_scene(scene: str, dataset_dir: str, weights_out: str, base=None,
                epochs=None, imgsz=None, device=0, batch=None, patience=None, seed=None,
                overrides=None, run_name=None) -> dict:
    """Train a YOLO detector on ``dataset_dir/data.yaml`` and copy its ``best.pt`` to ``weights_out``.

    **Base checkpoint.** ``base=None`` (the default) resolves through :func:`yolo_recipe.resolve_base`: an explicit
    ``base=`` wins, else ``$THOR_YOLO_BASE``, else :data:`yolo_recipe.DEFAULT_BASE` (``yolo11s.pt``). Passing
    ``base='yolov8n.pt'`` therefore still reproduces the published Phase-4 run exactly. The checkpoint is loaded from
    :func:`yolo_recipe.checkpoint_path` so the one-time download is cached per machine instead of landing a stray
    ``.pt`` in whatever directory the caller happened to start in; if that path cannot be used we fall back to the
    bare asset name and let ultralytics resolve it.

    **First train needs the network once** — see ``python sim/yolo_recipe.py`` for the exact manifest (yolo11s.pt
    ~19 MB, plus yolo26n.pt ~5.3 MB for ultralytics 8.4.83's AMP sanity check and Arial.ttf ~0.8 MB for the plots).

    **Recipe.** Every other knob comes from :func:`yolo_recipe.train_kwargs` (``None`` = documented default);
    ``overrides`` is applied last for one-off experiments. Lazy-imports ``ultralytics`` (which pulls ``torch``) —
    never at module top.

    Returns ``{'weights_path', 'metrics', 'base', 'recipe', 'n_images'}``: ``metrics`` carries per-class AP/AP50/recall
    + summary scalars when ultralytics exposes them, and ``base``/``recipe`` make the result self-describing, so a
    logged run can be reproduced (and the thesis can state what produced a number) without guessing.
    """
    try:
        from ultralytics import YOLO  # lazy: heavy (pulls torch); keep module import light
    except Exception as exc:
        raise RuntimeError(
            f"ultralytics is required to train a scene detector but failed to import ({exc}). "
            f"Install it in the WSL venv (`pip install ultralytics`)."
        ) from exc
    import shutil
    import tempfile

    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml!r}; call build_data_yaml(...) first.")

    n_images = yolo_recipe.count_images(dataset_dir)
    if n_images == 0:
        # Fail here with the cause, rather than opaquely inside ultralytics' dataloader after the base is downloaded.
        raise RuntimeError(f"no images under {os.path.join(dataset_dir, 'images')!r}; scan the scene before training.")

    base = yolo_recipe.resolve_base(base)
    kwargs = yolo_recipe.train_kwargs(n_images=n_images, epochs=epochs, imgsz=imgsz, batch=batch,
                                      patience=patience, seed=yolo_recipe.DEFAULT_SEED if seed is None else seed,
                                      overrides=overrides)

    load_from = yolo_recipe.checkpoint_path(base)
    try:
        model = YOLO(load_from)
    except Exception:
        model = YOLO(base)     # cache path unusable (offline, permissions, a name ultralytics resolves differently)

    # ultralytics mangles run-output paths containing an apostrophe (".../Master's Project/...": it saves best.pt to the
    # real path but reloads it from an apostrophe-stripped one, crashing final_eval). Train into a clean temp dir, then
    # copy the weights — and best-effort the run artifacts (curves, confusion matrix) — back under the dataset dir.
    runs_root = tempfile.mkdtemp(prefix="yolo_runs_")
    # ultralytics' select_device() WRITES os.environ["CUDA_VISIBLE_DEVICES"] and never puts it back. AI2-THOR reads
    # that variable when a Controller starts: seeing it set, it assigns a gpu_device and takes its CloudRendering
    # Vulkan-mapping path, which has no Vulkan device under WSL and raises "Could not find a Vulkan device
    # corresponding to the CUDA device". So any simulator started in this process AFTER a training run died (the
    # perception pipeline's verify step did, 2026-09-15). Snapshot and restore it.
    _cvd_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        run_name = run_name or scene     # distinct per model, so two models' run artifacts don't overwrite
        results = model.train(data=data_yaml, device=device, project=runs_root, name=run_name, exist_ok=True, **kwargs)

        best = _find_best(results, model)
        metrics = _train_metrics(results, model)
        out_parent = os.path.dirname(os.path.abspath(weights_out)) or "."
        os.makedirs(out_parent, exist_ok=True)
        if best:
            shutil.copyfile(best, weights_out)
        else:
            weights_out = None  # training produced no locatable best.pt

        save_dir = getattr(results, "save_dir", None)
        if save_dir and os.path.isdir(str(save_dir)):
            dest = os.path.join(dataset_dir, "runs", run_name)
            shutil.rmtree(dest, ignore_errors=True)
            try:
                shutil.copytree(str(save_dir), dest)
            except Exception:
                pass
    finally:
        shutil.rmtree(runs_root, ignore_errors=True)
        if _cvd_before is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _cvd_before

    return {"weights_path": weights_out, "metrics": metrics, "base": base, "recipe": kwargs, "n_images": n_images}


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "gen_dataset.py — auto-label AI2-THOR frames into a per-scene YOLO dataset and train a stateful detector.\n"
        "This module does NOT capture or train on import/run; drive it from agent code inside WSL, e.g.:\n"
        "\n"
        "  import gen_dataset, perception\n"
        "  c2i = {}\n"
        "  for n in perception.scene_class_names(agent.objects()):   # optional: scene-deterministic id order\n"
        "      c2i.setdefault(n, len(c2i))\n"
        "  for i, _ in enumerate(agent.look_around()):               # spin in place, grabbing frames\n"
        "      gen_dataset.capture_frame(agent, scene, out_dir, split='train', class_to_id=c2i, index=i)\n"
        "  gen_dataset.build_data_yaml(out_dir, c2i)\n"
        "  gen_dataset.train_scene(scene, out_dir, f'weights/{scene}.pt')\n"
    )


# ===================================================================== PUBLIC API ====================================
# capture_frame(agent, scene, out_dir, split='train', class_to_id=None, min_area=400, min_visible_frac=0.0,
#               index=None, memory=None, source='environment') -> dict
#     Auto-label the agent's current RGB frame. For each VISIBLE object with an instance mask, compute a tight bbox
#     (perception.box_from_mask); drop boxes with inclusive-corner pixel area < min_area, edge-hugging slivers
#     (EDGE_SLIVER_PX), or fill ratio (mask_px / bbox_area) < min_visible_frac; label survivors with
#     perception.class_name (stateful '<Type>_<state>'). MUTATES class_to_id ({name: id}), appending new classes in
#     sorted order for stable ids. Writes out_dir/images/<split>/<scene>_<uid>.jpg and
#     out_dir/labels/<split>/<scene>_<uid>.txt (perception.yolo_line); <uid> is a content hash, index-prefixed if given.
#     If memory is passed, records detections via memory.add_perception_label (auditable). Lazy-imports numpy + PIL.
#     Returns {'written': bool, 'image_path': str|None, 'n_boxes': int, 'classes': [str]}.
#
# build_data_yaml(out_dir, class_to_id) -> str
#     Write out_dir/data.yaml in ultralytics format (path=abs(out_dir), train: images/train, val: images/val, nc, and
#     the id->name `names` map, inverse of class_to_id, sorted by id). Returns the yaml path.
#
# train_scene(scene, dataset_dir, weights_out, base=None, epochs=None, imgsz=None, device=0, batch=None,
#             patience=None, seed=None, overrides=None) -> dict
#     Lazy-import ultralytics YOLO (pulls torch; never at module top). Train on dataset_dir/data.yaml and copy the
#     run's best.pt to weights_out (e.g. weights/<scene>.pt). base=None -> yolo_recipe.resolve_base() (explicit arg >
#     $THOR_YOLO_BASE > 'yolo11s.pt'); every other None -> the documented yolo_recipe default; `overrides` is applied
#     last. The checkpoint is loaded from yolo_recipe.checkpoint_path() so the first-train download is cached per
#     machine. Returns {'weights_path': str|None, 'metrics': {...}, 'base': str, 'recipe': {...}, 'n_images': int};
#     metrics carries 'per_class_ap' (AP50-95), 'per_class_ap50', 'per_class_recall' + summary scalars.
