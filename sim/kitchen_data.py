"""Planning for the per-kitchen perception datasets — every decision the live generator makes, minus the simulator.

Option (b), approved 2026-09-28 after two Codex/gpt-6-astra reviews: two type-only YOLO detectors per kitchen
(primary + close-view specialist) and one state estimator pooled over the five kitchens. This module owns the
policy so each rule is unit-tested on bare Windows python (``python sim/_kitchen_data_selftest.py``):

* :func:`select_scenes` — the five kitchens, from the instruction set and the 30-kitchen scan;
* :func:`episode_plan` — every episode's split and seed fixed IN ADVANCE, from a stable hash (no ``hash()``,
  which is salted per process), interleaved so even a partial run has all three splits;
* :class:`Quota` — per-SPLIT coverage targets for detection (instances per type, per object instance) and for
  states (crops per realised state combination); train views chase deficits, val/test views do not;
* :func:`audit` — the contract; a failed audit BLOCKS training unless the operator overrides it on the record;
* :func:`repeat_factors` — LVIS repeat-factor sampling, an ablation arm against capture balancing alone;
* :func:`select_ambiguity_types` — the specialist's types, from the primary's VALIDATION results (never test).

Stdlib only.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random

import vocab

# ----------------------------------------------------------------------------------------------------------------------
# Configuration. Recorded verbatim in every run manifest.
# ----------------------------------------------------------------------------------------------------------------------
#: Chosen 2026-09-28 by :func:`select_scenes` over the 277-item general set: every instruction feasible (by the
#: harness's applicability rule) in >=1 of them, 269 in all five — the maximum over every 5-kitchen subset.
#: FloorPlan1 kept for continuity with the published work; FloorPlan5 excluded (lettuce pickup hangs the sim).
#: Applicability is not executability: `validate_general.py` runs every reference plan in each of them.
DEFAULT_SCENES = ("FloorPlan1", "FloorPlan6", "FloorPlan8", "FloorPlan15", "FloorPlan17")

#: ultralytics is pinned: YOLO26 behaviour (optimizer auto-switch, end-to-end head) is version-specific.
ULTRALYTICS_VERSION = "8.4.83"

CAPTURE = {
    # episodes are the unit of splitting and of randomisation. Held-out raised 8 -> 16 each after the first full
    # FloorPlan1 run left small types short in val/test (Plate 24/60 test, Fork 27/60 val); Astra noted +25% would
    # not close that gap. Pre-declared here, before the re-capture, independent of any model's errors.
    "episodes": {"train": 64, "val": 16, "test": 16},
    "frames_per_episode": 110,
    # reversible axes (open, on, dirty, filled/liquid) are re-rolled every this many accepted frames -> 5 state
    # rounds per episode; irreversible axes (sliced, cooked, cracked) only at the episode start. A round also ends
    # after `reroll_every * 5` attempts, so an unproductive round cannot stall the episode.
    "reroll_every": 22,
    # a state key whose object was targeted this many times without a single large-enough crop stops pulling the
    # camera (floor weight) and is listed in the audit as "tried but never observed"; retries continue at the floor
    "suppress_after": 30,
    # held-out episodes ALWAYS shuffle placement (train keeps the default arrangement 25% of the time), so no
    # val/test episode reproduces a train arrangement
    "p_spawn_heldout": 1.0,
    # anti-grind: a TRAIN episode whose last `saturation_window` attempts kept <= `saturation_keep` frames ends
    # early (2026-09-29: late v2b episodes spent 550 views / ~3 min to keep 35-44 frames and looked stuck)
    "saturation_window": 150, "saturation_keep": 8,
    # hang guard: an episode running longer than this is discarded and redone once on a fresh simulator
    "episode_timeout_s": 600,
    # an episode must put at least this share of the kitchen's settable objects into a verified state, or the
    # audit FAILS (would have caught the v2b renaming bug — 28 of 48 set — on its first episode)
    "min_state_coverage": 0.90,
    # the full check_capture verifier runs every this many episodes inside the capture; any FAIL aborts the run
    "check_every": 8,
    # detection targets, per split: instances of each base TYPE, and labels of each object INSTANCE
    "type_target": {"train": 500, "val": 60, "test": 60},
    "instance_target": {"train": 20, "val": 3, "test": 3},
    # state targets, per split: crops of each realised composite state, counted only when the object's box is at
    # least `state_min_px` (a 30x30 px object cannot show a liquid, whatever its label)
    "state_target": {"train": 150, "val": 25, "test": 25},
    "state_min_px": 900,
    # domain randomisation per episode (paper §2.2.1)
    "p_spawn": 0.75, "p_lighting": 0.6, "p_materials": 0.2,
    # view geometry
    "dist_band": (0.5, 4.0), "p_crouch": 0.2, "yaw_jitter": 25.0, "horizon_jitter": 12.0, "p_random_view": 0.15,
    # close-view share (feeds the specialist + state estimator) and held "floating" views (paper Fig. 4)
    "p_close": 0.30, "close_band": (0.45, 1.2), "p_held": 0.10, "held_views": 3,
    # TRAIN-only explicit state-targeting branch (policy v2c-2, 2026-09-29): this share of views picks the object by
    # its STATE deficit alone and shoots it from `state_band`, standing (so the camera looks down into vessels);
    # held objects are also chosen by state deficit in train. Val/test keep the original protocol. Reason: after 16
    # train episodes 53/84 state keys projected short — states were set, but small objects were rarely >= 900 px.
    "p_state_view": 0.35, "state_band": (0.45, 1.0),
    # policy v2c-3: `state_target` now counts PER HEAD ('Type|axis=value'); every reachable full combination must
    # still reach `combo_floor` large train crops from >= `combo_min_episodes` distinct train episodes
    "combo_floor": 10, "combo_min_episodes": 3,
    # render guard (2026-09-29): a view is kept only when two consecutive fresh renders agree (whole-image MAE
    # < guard_mae AND every guard_tile x guard_tile tile < guard_tile_mae) and the frame shows no corruption
    # signature (pure white / whole-image colour cast); at most guard_max_renders renders, else the view is skipped
    "guard_mae": 1.0, "guard_tile": 80, "guard_tile_mae": 6.0, "guard_max_renders": 8,
    "guard_confirm_after_retry": 2,
    # fail fast only on real danger. Measured 2026-09-29: glitches occur at a ~2-3 % base rate in bursts even on a
    # fresh WSL (not explained by animated objects), and every frame the guard kept scanned clean (3,960/3,960) —
    # so re-renders are the guard WORKING, not a failure. Abort when an episode needed a re-render on > 30 % of views
    # (a burst like the 50 % one) or had to SKIP > 5 % of views.
    "guard_max_retry_rate": 0.30, "guard_max_skip_rate": 0.05,
    # variant G (Arash, 2026-09-29): at the FIRST glitched render the episode is discarded, the simulator process
    # restarted and the episode redone from its seed (1 glitched view in 4,848 vs 6-44 % without restarts)
    "restart_on_glitch": True, "glitch_max_redos": 6,
    # negatives: frames with no labelled object are kept at this rate (detectors must learn "nothing here")
    "p_empty": 0.02,
    # train frames: accept if they add ANY deficit, else with this probability (natural mix)
    "p_background": 0.08,
    # paired renders for state-observability evidence: share of accepted frames, reversible axes only
    "p_pair": 0.15,
    # label filter (paper: very small/far objects excluded); px at 640x640
    "min_box_area": 144, "min_mask_px": 60,
}

TRAIN = {
    "primary":    {"base": "yolo26s.pt", "epochs": 150, "patience": 40, "batch": 16, "imgsz": 640,
                   "optimizer": "MuSGD", "lr0": 0.01},
    # the specialist is piloted as m vs l on FloorPlan1; the winner is frozen for the other four kitchens
    "specialist": {"base": "yolo26m.pt", "alt_base": "yolo26l.pt", "epochs": 150, "patience": 40, "batch": 8,
                   "imgsz": 640, "optimizer": "MuSGD", "lr0": 0.01},
    "baseline":   {"base": "yolo11s.pt", "epochs": 150, "patience": 40, "batch": 16, "imgsz": 640,
                   "optimizer": "SGD", "lr0": 0.01},
    "rfs_t": 0.10,
}

#: Small / mutually confusable types — the specialist's vocabulary before any model exists (paper Table 6:
#: Spoon/Fork/ButterKnife at 0-25 % before disambiguation). Validation evidence adds to it; it never subtracts.
A_PRIORI_AMBIGUOUS = (
    "Knife", "ButterKnife", "Fork", "Spoon", "Spatula", "Ladle", "SaltShaker", "PepperShaker",
    "Mug", "Cup", "Bowl", "Pot", "Bottle", "WineBottle", "Apple", "Tomato", "Potato", "Egg", "Bread",
    "CreditCard", "Pen", "Pencil", "DishSponge", "SoapBottle",
)
AMBIGUITY_AP50 = 0.90          # a type below this VALIDATION AP50 joins the specialist
AMBIGUITY_CONFUSION = 0.05     # ... as do both sides of a pair confusing more than this share of GT
AMBIGUITY_MIN_SUPPORT = 20     # ... but only on at least this many validation instances

#: Pre-declared acceptance targets (Astra, 2026-09-28; engineering targets, not universal standards).
ACCEPTANCE = {"macro_ap50": 0.90, "macro_ap50_95": 0.70, "critical_state_precision": 0.95,
              "critical_state_recall": 0.95,
              "note": "state P/R on OBSERVABLE crops at validation-chosen thresholds; abstentions reported "
                      "separately as coverage, never counted as correct"}


# ----------------------------------------------------------------------------------------------------------------------
# 1. Scene selection.
# ----------------------------------------------------------------------------------------------------------------------
def feasibility(items, inventory) -> dict:
    """``{item_id: set(scenes)}`` where every referenced type exists and the goal is not already true at reset —
    the evaluation harness's own rule, imported, not re-implemented."""
    import eval as _eval
    import general_set as gs
    scenes = inventory.get("scenes") or []
    per = inventory.get("per_scene") or {}
    out = {}
    for it in items:
        need = gs.referenced_types(it)
        pre = set(gs.presatisfied_scenes(it, inventory))
        out[it["id"]] = {s for s in scenes if not _eval.missing_types(need, list(per.get(s) or {})) and s not in pre}
    return out


def select_scenes(feas: dict, scenes, k: int = 5, prefer=("FloorPlan1",), exclude=("FloorPlan5",)) -> dict:
    """Best k-subset, lexicographically: (1) items feasible somewhere, (2) items feasible everywhere (a larger
    common panel for paired comparisons), (3) preferred kitchens included, (4) lowest scene numbers."""
    pool = [s for s in scenes if s not in set(exclude)]
    best = None
    for combo in itertools.combinations(pool, k):
        cs = set(combo)
        key = (sum(1 for v in feas.values() if v & cs), sum(1 for v in feas.values() if cs <= v),
               sum(1 for s in combo if s in prefer), tuple(-_num(s) for s in combo))
        if best is None or key > best[0]:
            best = (key, combo)
    key, combo = best
    cs = set(combo)
    return {"scenes": list(combo), "n_items": len(feas), "feasible_somewhere": key[0],
            "feasible_everywhere": key[1],
            "uncovered": sorted(i for i, v in feas.items() if not (v & cs)),
            "partial": {i: sorted(cs - v, key=_num) for i, v in feas.items() if (v & cs) and not cs <= v},
            "excluded": list(exclude)}


def _num(scene: str) -> int:
    d = "".join(ch for ch in scene if ch.isdigit())
    return int(d) if d else 0


# ----------------------------------------------------------------------------------------------------------------------
# 2. Episodes.
# ----------------------------------------------------------------------------------------------------------------------
def stable_seed(*parts) -> int:
    """A 31-bit seed from a SHA-256 of the parts — identical in every process, unlike ``hash()``."""
    return int(hashlib.sha256("#".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def episode_plan(scene: str, episodes: dict) -> list:
    """``[{'episode', 'split', 'seed'}]`` fixed in advance: splits interleaved evenly over the index so a run cut
    short still has every split; seeds from :func:`stable_seed`, so episode e of scene s is the same world in
    every run on every machine."""
    n = sum(episodes.values())
    slots = []
    for split in ("val", "test"):
        k = episodes.get(split, 0)
        for j in range(k):
            slots.append((int((j + (0.33 if split == "val" else 0.66)) * n / max(k, 1)), split))
    taken = {}
    for pos, split in sorted(slots):
        while pos in taken:
            pos = (pos + 1) % n
        taken[pos] = split
    return [{"episode": e, "split": taken.get(e, "train"), "seed": stable_seed(scene, "episode", e)}
            for e in range(n)]


# ----------------------------------------------------------------------------------------------------------------------
# 3. Coverage bookkeeping.
# ----------------------------------------------------------------------------------------------------------------------
class Quota:
    """Per-split counts against per-split targets, for three units:

    * ``type``  — detector labels per base type;
    * ``inst``  — detector labels per object instance present at reset (every object must be seen);
    * ``state`` — state-estimator crops per realised composite state key (:func:`vocab.state_key`).

    Train frames are chosen by :meth:`gain` (deficit-driven); val/test frames are not — they are sampled by a fixed
    protocol and only COUNTED here, so held-out data is independent of what training happened to lack.
    """

    #: ``state`` counts PER HEAD — ``'Type|axis=value'`` keys (:func:`vocab.axis_keys`), the unit the multi-head
    #: state estimator learns (policy v2c-3, 2026-09-29; before, it counted full combinations, which split one
    #: mug's examples eight ways). ``combo`` keeps the full combinations as an ENFORCED floor — each reachable
    #: combination at least ``combo_floor`` large crops from at least ``combo_min_episodes`` episodes — so no
    #: combination is absent and co-occurrence confounds ("wine only ever in a dirty mug") stay visible (Astra
    #: step3e).
    UNITS = ("type", "inst", "state", "combo")

    def __init__(self, targets: dict, combo_min_episodes: int = 0):
        # targets: {'type': {split: {key: n}}, 'inst': {...}, 'state': {...}, 'combo': {...}}
        self.targets = {u: {s: dict(v) for s, v in targets.get(u, {}).items()} for u in self.UNITS}
        self.counts = {u: {s: {} for s in ("train", "val", "test")} for u in self.UNITS}
        self.frames = {"train": 0, "val": 0, "test": 0}
        # observation OPPORTUNITIES per combination key: times a view targeted an object in that state, and how
        # many of those produced a large-enough crop (Astra: count opportunities, not selections)
        self.tried, self.hit = {}, {}
        self.suppress_after = 0
        self.combo_min_episodes = int(combo_min_episodes)
        self.combo_eps = {}                      # combination key -> sorted list of TRAIN episodes it appeared in
        # v3: per-type distinct viewpoints; per-instance held frames, held background cells and held standing spots
        self.vp_target, self.vp = 0, {}          # type -> [cell, ...] (train)
        self.held_target, self.held_bg_target = 0, 0
        self.held_n, self.held_bg, self.held_pos = {}, {}, {}

    def note_target(self, key: str, observed: bool) -> None:
        self.tried[key] = self.tried.get(key, 0) + 1
        if observed:
            self.hit[key] = self.hit.get(key, 0) + 1

    def suppressed(self, key: str) -> bool:
        return bool(self.suppress_after) and self.tried.get(key, 0) >= self.suppress_after \
            and not self.hit.get(key)

    def deficit(self, unit: str, split: str, key: str) -> float:
        t = self.targets[unit].get(split, {}).get(key)
        if not t:
            return 0.0
        if unit == "combo":
            if self.suppressed(key):
                return 0.0
            d = 1.0 - self.counts["combo"][split].get(key, 0) / float(t)
            if self.combo_min_episodes and split == "train":
                d = max(d, 1.0 - len(self.combo_eps.get(key, ())) / float(self.combo_min_episodes))
            return max(0.0, d)
        return max(0.0, 1.0 - self.counts[unit][split].get(key, 0) / float(t))

    def state_value(self, axis_keys, combo_key, split: str = "train", suppressed_scale: float = 0.0) -> float:
        """Deficit value of ONE object's state: the MEAN of its head deficits (normalised, so an object with four
        heads does not pull four times harder than one with one — Astra step3e) plus its combination's floor
        deficit (a suppressed combination contributes ``suppressed_scale`` of its raw deficit)."""
        heads = [self.deficit("state", split, k) for k in axis_keys]
        v = sum(heads) / len(heads) if heads else 0.0
        if combo_key:
            if self.suppressed(combo_key):
                t = self.targets["combo"].get(split, {}).get(combo_key)
                raw = max(0.0, 1.0 - self.counts["combo"][split].get(combo_key, 0) / float(t)) if t else 0.0
                v += suppressed_scale * raw
            else:
                v += self.deficit("combo", split, combo_key)
        return v

    def state_weight(self, axis_keys, combo_key=None, floor: float = 0.02) -> float:
        """How much a TRAIN view aimed at an object in this state is worth. A suppressed combination keeps a
        quarter of its deficit, so the explicit state-targeting branch can still revisit it from close up (Astra:
        suppression must not stop the policy from retrying exactly the states it was built for)."""
        if isinstance(axis_keys, str):                       # a lone key (older call sites / tests)
            axis_keys, combo_key = [axis_keys], combo_key
        return max(floor, self.state_value(axis_keys, combo_key, "train", suppressed_scale=0.25))

    def gain(self, labels, split: str = "train") -> float:
        """Deficit value of a frame. ``labels`` carry ``type``, ``oid``/``inst``, ``orig_type``, ``state``, ``px``."""
        g, seen = 0.0, set()
        for lab in labels:
            for unit, key, w in self._keys(lab):
                if (unit, key) in seen:
                    continue
                seen.add((unit, key))
                g += w * self.deficit(unit, split, key)
        return g

    def add(self, labels, split: str, episode=None) -> None:
        self.frames[split] += 1
        for lab in labels:
            for unit, key, _w in self._keys(lab):
                c = self.counts[unit][split]
                c[key] = c.get(key, 0) + 1
                if unit == "combo" and split == "train" and episode is not None:
                    eps = self.combo_eps.setdefault(key, [])
                    if episode not in eps:
                        eps.append(episode)

    @staticmethod
    def _keys(lab):
        """(unit, key, weight) for one label. Head keys share one unit of weight (normalised per object)."""
        yield "type", lab["type"], 1.0
        # instance identity is the object's NAME (stable across layout shuffles), not its objectId
        yield "inst", lab.get("inst") or lab["oid"], 1.0
        if lab.get("px", 0) < lab.get("state_min_px", 0) or not lab.get("state"):
            return
        heads = vocab.axis_keys(lab.get("orig_type") or lab["type"], lab["state"])
        for k in heads:
            yield "state", k, 1.0 / len(heads)
        sk = lab.get("state_key") or ""
        if heads and sk and not sk.endswith("|-"):
            yield "combo", sk, 1.0

    def object_weight(self, type_: str, inst: str, axis_keys=(), combo_key=None) -> float:
        """How much it is worth aiming the camera at this object in a TRAIN episode right now."""
        if isinstance(axis_keys, str):                       # legacy call: a single state key
            axis_keys, combo_key = [], axis_keys
        return (self.deficit("type", "train", type_) + self.deficit("inst", "train", inst)
                + self.state_value(list(axis_keys), combo_key) + 0.02)

    def pick_target(self, candidates, rng: random.Random, floor: float = 0.05):
        """Deficit-weighted choice among ``[(target, combo_key, axis_keys)]`` (or legacy ``(target, key)``) for a
        TRAIN episode; ``floor`` keeps every state possible after its targets are met."""
        w = []
        for c in candidates:
            heads = c[2] if len(c) > 2 else []
            w.append(max(floor, self.state_value(heads, c[1])))
        return rng.choices(candidates, weights=w, k=1)[0]

    def replay(self, rows) -> None:
        """Recount every unit from stored frame records (after a change of counting unit — no recapture)."""
        self.counts = {u: {s: {} for s in ("train", "val", "test")} for u in self.UNITS}
        self.frames = {"train": 0, "val": 0, "test": 0}
        self.combo_eps = {}
        self.vp, self.held_n, self.held_bg, self.held_pos = {}, {}, {}, {}
        for r in rows:
            if "img" in r and not r.get("quarantined"):
                self.add(r["labels"], r["split"], r.get("episode"))
                if r["split"] == "train":
                    self.add_vp(r["labels"], r.get("cell"))
                    ps = r.get("pose") or {}
                    if r.get("kind") == "held" and ps.get("inst"):
                        self.add_held(ps["inst"], ps.get("cell"), (ps["x"], ps["z"]) if "x" in ps else None)

    def split_done(self, split: str) -> bool:
        return all(self.deficit(u, split, k) == 0 for u in self.UNITS for k in self.targets[u].get(split, {}))

    # ---- v3: viewpoints and held views
    def vp_deficit(self, type_: str) -> float:
        if not self.vp_target:
            return 0.0
        return max(0.0, 1.0 - len(self.vp.get(type_, ())) / float(self.vp_target))

    def vp_gain(self, labels, cell) -> float:
        """Value of a TRAIN frame for viewpoint diversity: each type seen from a cell it has not been seen from."""
        if cell is None or not self.vp_target:
            return 0.0
        return sum(0.5 * self.vp_deficit(t) for t in {l["type"] for l in labels}
                   if list(cell) not in self.vp.get(t, []))

    def add_vp(self, labels, cell) -> None:
        if cell is None:
            return
        for t in {l["type"] for l in labels}:
            cells = self.vp.setdefault(t, [])
            if list(cell) not in cells:
                cells.append(list(cell))

    def held_deficit(self, inst: str) -> float:
        if not self.held_target:
            return 0.0
        a = max(0.0, 1.0 - self.held_n.get(inst, 0) / float(self.held_target))
        b = max(0.0, 1.0 - len(self.held_bg.get(inst, ())) / float(self.held_bg_target or 1))
        return max(a, b)

    def add_held(self, inst: str, cell, pos) -> None:
        self.held_n[inst] = self.held_n.get(inst, 0) + 1
        if cell is not None and list(cell) not in self.held_bg.setdefault(inst, []):
            self.held_bg[inst].append(list(cell))
        if pos is not None:
            self.held_pos.setdefault(inst, []).append([round(pos[0], 2), round(pos[1], 2)])

    def to_dict(self) -> dict:
        return {"version": QUOTA_VERSION, "targets": self.targets, "counts": self.counts, "frames": self.frames,
                "tried": self.tried, "hit": self.hit, "suppress_after": self.suppress_after,
                "combo_min_episodes": self.combo_min_episodes, "combo_eps": self.combo_eps,
                "vp_target": self.vp_target, "vp": self.vp, "held_target": self.held_target,
                "held_bg_target": self.held_bg_target, "held_n": self.held_n, "held_bg": self.held_bg,
                "held_pos": self.held_pos}

    @classmethod
    def from_dict(cls, d: dict) -> "Quota":
        q = cls(d["targets"], d.get("combo_min_episodes", 0))
        for u in cls.UNITS:
            q.counts[u].update(d["counts"].get(u, {}))
        q.frames = d["frames"]
        q.tried, q.hit = dict(d.get("tried") or {}), dict(d.get("hit") or {})
        q.suppress_after = int(d.get("suppress_after") or 0)
        q.combo_eps = {k: list(v) for k, v in (d.get("combo_eps") or {}).items()}
        q.vp_target, q.vp = int(d.get("vp_target") or 0), dict(d.get("vp") or {})
        q.held_target, q.held_bg_target = int(d.get("held_target") or 0), int(d.get("held_bg_target") or 0)
        q.held_n, q.held_bg = dict(d.get("held_n") or {}), dict(d.get("held_bg") or {})
        q.held_pos = dict(d.get("held_pos") or {})
        return q


#: Counting-unit version of a saved Quota. "axis-1" = per-head state keys + combination floor (v2c-3). A saved
#: quota of any other version is REBUILT from the probe and replayed from frames.jsonl on resume.
QUOTA_VERSION = "v3-1"     # axis-2 (2026-09-29): StoveKnob state heads removed (vocab.STATE_EXCLUDED)


def build_targets(types, instances, axis_keys, cfg: dict = CAPTURE, combo_keys=()) -> dict:
    """Per-split targets: every type in the kitchen, every instance present at reset, every per-head state key the
    probe proved reachable, and (train only) the floor for every reachable full combination."""
    return {
        "type": {s: {t: n for t in types} for s, n in cfg["type_target"].items()},
        "inst": {s: {o: n for o in instances} for s, n in cfg["instance_target"].items()},
        "state": {s: {k: n for k in axis_keys} for s, n in cfg["state_target"].items()},
        "combo": {"train": {k: int(cfg.get("combo_floor", 0)) for k in combo_keys}} if cfg.get("combo_floor") else {},
    }


def reachable_keys(probe: dict) -> tuple:
    """(per-head keys, combination keys) the probe proved reachable, from its verified targets."""
    heads, combos = set(), set()
    for entries in probe.get("reachable", {}).values():
        for x in entries:
            base = x["key"].split("|", 1)[0]
            if any(b == base for b, _ax in vocab.STATE_EXCLUDED):
                continue                                  # e.g. StoveKnob: no state-estimator targets
            combos.add(x["key"])
            rec = {ax: {"value": x["target"].get(ax)} for ax in vocab.axes_of(base)}
            heads.update(vocab.axis_keys(base, rec))
    return sorted(heads), sorted(combos)


#: Top-up protocols (Arash's go 2026-09-29, Astra step3g/3h), appended AFTER a kitchen's main plan:
#: * TRAIN top-up: ids 1000+, the ordinary v2c-4 train policy (deficit-driven), until every train head and combination
#:   meets its floor, capped at TOPUP_TRAIN_MAX episodes; residuals are reported, never extended silently.
#: * "heldout-close-v1": ids 2000+, exactly CLOSE_EPISODES val + CLOSE_EPISODES test, each view aimed at a STATEFUL
#:   object chosen UNIFORMLY (never by deficit or by any model's errors), 0.45-1.0 m, standing; excluded from the main
#:   detector val/test and reported as a separate close-view benchmark. Pre-declared before any run.
TOPUP_TRAIN_MAX = 24
CLOSE_EPISODES = 12
CLOSE_PROTOCOL = "heldout-close-v1"


def topup_train_plan(scene: str, n: int = TOPUP_TRAIN_MAX) -> list:
    return [{"episode": 1000 + i, "split": "train", "seed": stable_seed(scene, "topup-train", i), "protocol": "topup"}
            for i in range(n)]


def close_plan(scene: str, n: int = CLOSE_EPISODES) -> list:
    out = []
    for i in range(2 * n):
        split = "val" if i % 2 == 0 else "test"
        out.append({"episode": 2000 + i, "split": split, "seed": stable_seed(scene, CLOSE_PROTOCOL, i),
                    "protocol": CLOSE_PROTOCOL})
    return out


def train_state_complete(quota) -> bool:
    """Every TRAIN per-head key and combination floor met (the top-up's stop rule)."""
    return all(quota.deficit(u, "train", k) == 0 for u in ("state", "combo")
               for k in quota.targets[u].get("train", {}))


def train_complete(quota) -> bool:
    """v3 top-up stop rule: every TRAIN type, per-head state key and combination floor met."""
    return all(quota.deficit(u, "train", k) == 0 for u in ("type", "state", "combo")
               for k in quota.targets[u].get("train", {}))


def accept_train(gain: float, rng: random.Random, cfg: dict = CAPTURE) -> bool:
    """A train frame is kept if it adds any deficit at all (no threshold a lone short class could fall under),
    else with ``p_background`` so the natural class mix is still represented."""
    return gain > 0.0 or rng.random() < cfg["p_background"]


# ----------------------------------------------------------------------------------------------------------------------
# 4. Audit — blocks training when it fails.
# ----------------------------------------------------------------------------------------------------------------------
def audit(quota: Quota, registry_types, unreachable_states=()) -> dict:
    """Every shortfall, named, per split and unit. ``ok`` is False if ANY target is unmet — type, instance or
    state — in ANY split, or a registry type never appears. Nothing is averaged away."""
    short = {u: {s: {} for s in ("train", "val", "test")} for u in Quota.UNITS}
    for u in Quota.UNITS:
        for s, tg in quota.targets[u].items():
            for k, n in tg.items():
                got = quota.counts[u][s].get(k, 0)
                if got < n:
                    short[u][s][k] = [got, n]
                elif u == "combo" and s == "train" and quota.combo_min_episodes \
                        and len(quota.combo_eps.get(k, ())) < quota.combo_min_episodes:
                    short[u][s][k] = [got, n, f"only {len(quota.combo_eps.get(k, ()))} episodes"]
    seen_types = {k for s in quota.counts["type"].values() for k in s}
    missing = sorted(set(registry_types) - seen_types)
    n_short = sum(len(v) for u in short.values() for v in u.values())
    tr = quota.counts["type"]["train"]
    vals = [tr.get(t, 0) for t in registry_types] or [0]
    never = sorted(k for k in quota.tried if quota.suppressed(k))
    return {"ok": not missing and n_short == 0, "missing_types": missing, "short": short, "n_short": n_short,
            "tried_never_observed": {k: quota.tried[k] for k in never},
            "unreachable_states": sorted(unreachable_states), "frames": dict(quota.frames),
            "type_imbalance_max_over_min": (max(vals) / min(vals)) if min(vals) > 0 else None}


# ----------------------------------------------------------------------------------------------------------------------
# 5. Repeat-factor sampling (Gupta et al., LVIS, CVPR 2019) — an ablation arm, not the default.
# ----------------------------------------------------------------------------------------------------------------------
def repeat_factors(image_classes: dict, t: float) -> dict:
    n = len(image_classes)
    if n == 0:
        return {}
    freq = {}
    for classes in image_classes.values():
        for c in set(classes):
            freq[c] = freq.get(c, 0) + 1
    rc = {c: max(1.0, math.sqrt(t / (k / n))) for c, k in freq.items()}
    return {img: max([rc[c] for c in set(cl)] or [1.0]) for img, cl in image_classes.items()}


def expand_repeats(factors: dict, seed: int = 0) -> list:
    rng = random.Random(seed)
    out = []
    for img in sorted(factors):
        r = factors[img]
        out.extend([img] * max(1, int(math.floor(r)) + (1 if rng.random() < (r - math.floor(r)) else 0)))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 6. Specialist types — from VALIDATION only.
# ----------------------------------------------------------------------------------------------------------------------
def select_ambiguity_types(registry_types, val_ap50=None, val_support=None, val_confusion=None,
                           a_priori=A_PRIORI_AMBIGUOUS) -> dict:
    """``val_ap50``/``val_support``: ``{type: value}`` on the primary's VALIDATION split; ``val_confusion``:
    ``{(true, pred): fraction of true's GT}``. Both sides of a confused pair join. Evidence on fewer than
    :data:`AMBIGUITY_MIN_SUPPORT` instances is recorded but does not decide."""
    types = set(registry_types)
    sup = val_support or {}
    why = {}
    for t in a_priori:
        if t in types:
            why.setdefault(t, []).append("a priori small/confusable")
    for t, ap in (val_ap50 or {}).items():
        if t in types and ap is not None and ap < AMBIGUITY_AP50:
            if sup.get(t, 0) >= AMBIGUITY_MIN_SUPPORT:
                why.setdefault(t, []).append(f"val AP50 {ap:.2f} < {AMBIGUITY_AP50}")
    for (a, b), f in (val_confusion or {}).items():
        if a != b and f > AMBIGUITY_CONFUSION and sup.get(a, 0) >= AMBIGUITY_MIN_SUPPORT:
            for t in (a, b):
                if t in types:
                    why.setdefault(t, []).append(f"val confusion {a}->{b} {f:.0%}")
    return {"types": sorted(why), "why": {t: sorted(set(v)) for t, v in sorted(why.items())},
            "source": "validation split"}


# ----------------------------------------------------------------------------------------------------------------------
# 7. Estimates are printed as estimates; the probe measures render speed, the pilot measures training speed.
# ----------------------------------------------------------------------------------------------------------------------
def estimate_capture_h(seconds_per_view: float, seconds_per_episode: float, cfg: dict = CAPTURE,
                       accept_rate: float = 0.5) -> float:
    n_ep = sum(cfg["episodes"].values())
    views = n_ep * cfg["frames_per_episode"] / accept_rate
    return round((views * seconds_per_view * (1 + cfg["p_pair"] * 3) + n_ep * seconds_per_episode) / 3600.0, 2)


# ===================================================================== PUBLIC API ====================================
# DEFAULT_SCENES, CAPTURE, TRAIN, ACCEPTANCE, A_PRIORI_AMBIGUOUS, ULTRALYTICS_VERSION
# feasibility(items, inventory);  select_scenes(feas, scenes, k=5, prefer, exclude)
# stable_seed(*parts);  episode_plan(scene, episodes) -> [{episode, split, seed}]
# Quota(targets) .gain(labels, split) .add .object_weight .pick_target .split_done .to_dict/.from_dict
# build_targets(types, instances, state_keys, cfg);  accept_train(gain, rng, cfg)
# audit(quota, registry_types, unreachable_states) -> {'ok', 'short', 'missing_types', ...}
# repeat_factors / expand_repeats;  select_ambiguity_types(registry, val_ap50, val_support, val_confusion)
# estimate_capture_h(seconds_per_view, seconds_per_episode, cfg)


# ======================================================================================================================
# Dataset v3 (Arash, 2026-09-30): recapture FloorPlan1 from scratch as the pilot. His requirements: a LOT of data for
# EACH object with DIFFERENT backgrounds; MOST data in the kitchen's ORIGINAL colours; EACH object HELD against MANY
# different backgrounds. Measured on v2c: only 38 % of frames were original colours+lighting, held views were 30 % of
# frames and chased state deficits (Potato held 514x, SoapBottle 9x), backgrounds often bare walls. Astra step3i.
# ======================================================================================================================
CAPTURE_V2C = dict(CAPTURE)
CAPTURE = dict(CAPTURE_V2C)
CAPTURE.update({
    "episodes": {"train": 128, "val": 20, "test": 20},
    # appearance: NO material randomisation (original colours); mild brightness-only variation on 20 % of episodes
    "p_materials": 0.0, "p_lighting": 0.20, "lighting_brightness": (0.85, 1.15),
    # more data per object, spread over places
    "type_target": {"train": 1500, "val": 150, "test": 150},
    "instance_target": {"train": 60, "val": 6, "test": 6},
    "viewpoint_target": 150,                 # distinct (0.5 m cell x 45 deg yaw) viewpoints per TYPE (train)
    # held views v3: per-INSTANCE quota, one pickup per instance per episode, a new standing spot per view (farthest
    # from that object's earlier held spots), facing content (>= held_min_context other labelled objects)
    "held_target": 60, "held_bg_target": 30, "held_views": 3, "held_min_context": 5, "held_yaw_tries": 6,
    "held_cap": 0.12, "p_held": 0.25,
    # The detector's GATE keeps the floor Astra approved for v2c (step3h). The v3 capture targets above (1,500 labels,
    # 60 per instance) are what the capture AIMS for; a small tool that ends at 900 labels is reported, not a reason
    # to refuse to train. Held-out main-protocol splits cannot be topped up, so their floor must be the gate's.
    "gate_type_target": {"train": 500, "val": 60, "test": 60},
    "gate_instance_target": {"train": 20, "val": 3, "test": 3},
})


# ======================================================================================================================
# Ambiguity (specialist) dataset — protocol amb-v1, frozen in docs/ambiguity-spec.md (Astra step3j/k/l, 2026-10-03).
# ======================================================================================================================
AMB_PROTOCOL = "amb-v1"
AMB_HOLDOUT = "amb-v1-holdout"                     # stored under split "test", sealed: never built into a list early
AMB_TARGETS = ("Fork", "Spoon", "ButterKnife", "Knife", "Spatula", "SaltShaker", "PepperShaker", "Plate", "Pot", "Pan")
AMB_PAIRS = (("SaltShaker", "PepperShaker"), ("Fork", "Spoon"), ("Spoon", "ButterKnife"), ("ButterKnife", "Knife"),
             ("Fork", "ButterKnife"), ("Spatula", "ButterKnife"), ("Plate", "Pot"), ("Pan", "Pot"))
AMB_DISTRACTORS = ("Cup", "Bottle", "Mug", "Book", "CreditCard", "Statue", "Vase", "WineBottle", "Bowl", "Kettle",
                   "SoapBottle", "PaperTowelRoll", "DishSponge")
AMB_EPISODES = {"train": 40, "val": 10, "test": 10, "holdout": 10}
# Arash 2026-10-03: "the ambiguity model is used when the object is held up, so there should be a lot of shots of the
# object in hand" -> held-dominant: 60 held targets + 12 non-targets in hand + 38 in-place views (110 per episode)
AMB_VIEWS = {"close": 14, "ordinary": 10, "free": 14, "held_other": 12, "held": 60}
AMB_HELD_PER_PICKUP = 6                    # views per pickup, each from a new standing spot
AMB_HELD_MIN_PX = 800                      # "held UP to the camera": the held object must be big in the frame
AMB_SLOT_TRIES = 8
AMB_MAX_VIEWS_PER_OBJECT = 6               # aimed (close + ordinary) views of one instance per episode
CAPTURE_AMB = dict(CAPTURE)
CAPTURE_AMB.update({"episodes": {k: v for k, v in AMB_EPISODES.items() if k != "holdout"},
                    "frames_per_episode": sum(AMB_VIEWS.values()), "p_spawn": 1.0, "p_spawn_heldout": 1.0,
                    "p_pair": 0.0, "held_cap": 0.70, "close_band": (0.3, 0.8), "dist_band": (1.0, 2.5),
                    "partner_pairs": 2, "partner_gap": (0.08, 0.20),
                    # held up for inspection: closer and higher in the frame, camera level-ish (smoke 2026-10-03: with
                    # the main capture's ranges the held object often sat small at the bottom of the frame)
                    "held_move": {"ahead": (-0.15, 0.05), "right": (-0.15, 0.15), "up": (0.0, 0.15)},
                    "held_horizon": (-5, 20)})


def amb_plan(scene: str) -> list:
    """Episodes fixed before capture: one InitialRandomSpawn layout per episode, seed derived from
    (scene, protocol, split, index), so no two splits share a layout seed. Holdout episodes carry split "test" for
    storage and protocol AMB_HOLDOUT; they are excluded from every list until the final evaluation."""
    base = {"train": 3000, "val": 3100, "test": 3200, "holdout": 3300}
    out = []
    for split, n in AMB_EPISODES.items():
        # index 0 of val/test/holdout = layouts already looked at in the smoke runs (2026-10-03): never evaluated on
        first = 0 if split == "train" else 1
        for i in range(first, first + n):
            out.append({"episode": base[split] + i, "split": "test" if split == "holdout" else split,
                        "seed": stable_seed(scene, AMB_PROTOCOL, split, i),
                        "protocol": AMB_HOLDOUT if split == "holdout" else AMB_PROTOCOL})
    return out


def amb_schedule(rng: random.Random) -> list:
    """The episode's fixed view schedule: non-held kinds shuffled, held views appended at the end (a held object is
    dropped elsewhere, so it must not disturb the other views of the layout)."""
    head = [k for k, n in AMB_VIEWS.items() if not k.startswith("held") for _ in range(n)]
    rng.shuffle(head)
    return head + ["held_other"] * AMB_VIEWS["held_other"] + ["held"] * AMB_VIEWS["held"]
