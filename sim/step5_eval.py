"""Step 5 / A6-eval - the evaluation runner for the memory database (docs/step5-plan.md section 10).

CODE ONLY (workflow A): nothing here starts a simulator, WSL python, Ollama or a socket. The live executor is an INJECTED
interface (`Executor.execute(EpisodeRequest) -> EpisodeResult`); workflow B plugs step4_refplans / HonestSkills into it.
Tests use a fake executor (sim/_step5_eval_selftest.py).

What it builds, in the order a run uses it
  1. task groups + a frozen split manifest      `build_split`, `write_manifest`, `load_manifest`
       union-find over (goal signature | plan concept key); EN/FA of one item together; practice / P-familiar /
       P-composition / P-translation sets with disjoint item ids; a giant component makes the composition protocol
       INFEASIBLE (declared, never silently repaired).
  2. paired plans                                `make_block_plan`, `make_sequences`
       every random choice is a stateless function of (master seed, namespace, ids) -> arms see identical practice
       orders, practice world seeds, test episodes and test world seeds (verified after setup by initial-state hash).
  3. the runner                                  `EvalRunner`
       learning curve: per (world kind, block, arm) one training run; K in {0,1,3,5,10} = completed practice ATTEMPTS;
       each K snapshot is frozen and every test opens a fresh read-only copy (state hash asserted equal before / after).
       persistent worlds: whole sequences from matched starts, memory consolidated after every episode, checkpointed.
       a JSON-lines journal makes the run resumable; the report is rebuilt from the journal alone.
  4. metrics + report                            `build_report`, `render_markdown`, `write_report`
       SR / PC / TDS per level and language, steps-to-find, walking, opens, stale visits, retries, consolidation latency,
       memory size, FLR, gate-pass / application coverage; paired block differences with bootstrap CIs over BLOCKS
       (or whole sequences), never over episodes.

Honesty: a report built from a fake executor says so; offline completion is never reported as live validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
SCHEMA = "s5.eval/1"
GROUPING_RULES = "goal-signature+plan-concept-key/1"

# ----------------------------------------------------------------------------------------------------------- arms
ARMS = {
    "none": frozenset(),                                   # same WM / map / sensing / priors / budgets, no cross-task memory
    "places": frozenset({"places"}),
    "places+habits": frozenset({"places", "habits"}),
    "full": frozenset({"places", "habits", "lessons", "plans"}),   # "+lessons&plans"
}
ARM_ORDER = ["none", "places", "places+habits", "full"]
LEVELS = ["L1", "L2", "L3"]
LANGS = ["en", "fa"]
WORLD_KINDS = ("reset_fresh", "reset_shared")              # learning-curve worlds; persistent worlds are separate
CANONICAL = "canonical"                                    # B1: the executor's canonical FloorPlan world + the item's setup (no random spawn)
PRIMARY = {"L1": "sr", "L2": "pc", "L3": "tds"}            # spec section 10 / THESIS section 6
METRICS = ["sr", "pc", "tds", "goal_ok", "found", "steps_to_find", "walking_m", "opens", "stale_visits", "retries", "steps"]
COST_METRICS = ["steps_to_find", "walking_m", "opens", "stale_visits", "retries", "steps"]


class EvalError(Exception):
    pass


class InvariantError(EvalError):
    """A fail-fast invariant failed (frozen store changed, practice executor wrote to memory, ...). Aborts the run."""


class JournalMismatch(EvalError):
    pass


class AllowlistError(EvalError):
    """Gate 12.10: an item outside the workflow-B allowlist (or inside the final test) reached a run or a journal."""


class GroupingInfeasible(EvalError):
    pass


# ------------------------------------------------------------------------------------------------------- utilities
def canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else str(b).encode("utf-8")).hexdigest()


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _eid(cell: str) -> str:
    """Stable, charset-safe episode id for a cell (consolidation event ids `cons:<episode>` stay idempotent on replay)."""
    return "ep:" + sha256_hex(cell)[:24]


def derive(master, *parts) -> int:
    """Stateless 63-bit seed from (master, namespace, ids...). No RNG state exists anywhere, so resume needs none."""
    return int(sha256_hex(canon([master, *parts]))[:15], 16)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _num(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return None


# ----------------------------------------------------------------------------------------------- dataset + groups
def load_items(path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    d = json.loads(raw.decode("utf-8"))
    items = d["items"] if isinstance(d, dict) else d
    for it in items:
        for k in ("id", "level", "en", "fa", "plan", "goal"):
            if k not in it:
                raise EvalError(f"item {it.get('id')} lacks '{k}'")
    return items, sha256_hex(raw)


def goal_signature(item) -> str:
    """Sorted canonical clauses: types, keys, typed values AND operators (`contains`, `any`, ...) are all kept; `setup` is
    deliberately excluded (it is a starting condition, not the goal)."""
    return canon(sorted(canon(c) for c in item["goal"]))


def plan_concept_key(item) -> str:
    """Ordered (action, object, to, arg): `to` and `arg` are KEPT so 'put apple in fridge' != 'put apple on counter'."""
    return canon([[s.get("action"), s.get("object"), s.get("to"), s.get("arg")] for s in item["plan"]])


class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, k):
        while self.p[k] != k:
            self.p[k] = self.p[self.p[k]]
            k = self.p[k]
        return k

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def task_groups(items) -> list[list[str]]:
    """Union-find components over item ids. An EN/FA pair is ONE item, so translations are together by construction."""
    uf = _UF([it["id"] for it in items])
    by_goal, by_plan = {}, {}
    for it in items:
        by_goal.setdefault(goal_signature(it), []).append(it["id"])
        by_plan.setdefault(plan_concept_key(it), []).append(it["id"])
    for table in (by_goal, by_plan):
        for ids in table.values():
            for other in ids[1:]:
                uf.union(ids[0], other)
    comps = {}
    for it in items:
        comps.setdefault(uf.find(it["id"]), []).append(it["id"])
    return sorted((sorted(v) for v in comps.values()), key=lambda v: v[0])


def _group_level(ids, level_of):
    cnt = {}
    for i in ids:
        cnt[level_of[i]] = cnt.get(level_of[i], 0) + 1
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_split(items, dataset_sha, master_seed, params_sha=None, composition_frac=0.25, familiar_test_frac=0.34,
                giant_frac=0.40, min_groups_for_composition=4) -> dict:
    """The frozen split. Practice, P-familiar, P-composition and P-translation use DISJOINT item ids (translation
    transfer, i.e. the same item in the other language, is a separate protocol)."""
    level_of = {it["id"]: it["level"] for it in items}
    groups = task_groups(items)
    n_items = len(items)
    largest = max(len(g) for g in groups)
    feasible = (largest / n_items <= giant_frac) and len(groups) >= min_groups_for_composition
    reason = None
    if largest / n_items > giant_frac:
        reason = f"giant component: {largest}/{n_items} items ({largest / n_items:.0%} > {giant_frac:.0%}); concept definition must be revised"
    elif len(groups) < min_groups_for_composition:
        reason = f"only {len(groups)} groups"
    gl = {g[0]: _group_level(g, level_of) for g in groups}

    comp_groups: set = set()
    if feasible:
        for lvl in LEVELS:
            stratum = sorted((g for g in groups if gl[g[0]] == lvl),
                             key=lambda g: derive(master_seed, "split", "comp", g[0]))
            if len(stratum) >= 2:
                for g in stratum[:max(1, int(math.ceil(composition_frac * len(stratum))))]:
                    comp_groups.add(g[0])

    practice_items, familiar_items, comp_items = [], [], []
    for g in groups:
        if g[0] in comp_groups:
            comp_items += g
            continue
        if len(g) >= 2:
            order = sorted(g, key=lambda i: derive(master_seed, "split", "fam", i))
            n_test = min(len(g) - 1, max(1, int(math.ceil(familiar_test_frac * len(g)))))
            familiar_items += order[:n_test]
            practice_items += order[n_test:]
        else:
            practice_items += g
    practice = []
    for iid in sorted(practice_items):
        lang = LANGS[derive(master_seed, "split", "plang", iid) % 2]
        practice.append({"item": iid, "lang": lang})
    translation = [{"item": p["item"], "lang": LANGS[1 - LANGS.index(p["lang"])]} for p in practice]
    m = {
        "schema": SCHEMA, "kind": "split_manifest", "grouping_rules": GROUPING_RULES, "dataset_sha256": dataset_sha,
        "params_sha256": params_sha, "master_seed": master_seed,
        "settings": {"composition_frac": composition_frac, "familiar_test_frac": familiar_test_frac,
                     "giant_frac": giant_frac, "min_groups_for_composition": min_groups_for_composition},
        "groups": [{"id": g[0], "items": g, "level": gl[g[0]], "heldout_composition": g[0] in comp_groups} for g in groups],
        "composition": {"feasible": feasible, "reason": reason},
        "practice": practice,
        "test_familiar": sorted(set(familiar_items)),
        "test_composition": sorted(comp_items) if feasible else [],
        "test_translation": translation,
        "stats": {"n_items": n_items, "n_groups": len(groups), "largest_group": largest,
                  "largest_fraction": round(largest / n_items, 4),
                  "group_sizes": _hist([len(g) for g in groups]), "n_practice_items": len(practice),
                  "n_test_familiar_items": len(familiar_items), "n_test_composition_items": len(comp_items) if feasible else 0},
    }
    m["sha256"] = sha256_hex(canon({k: v for k, v in m.items() if k != "sha256"}))
    validate_manifest(m)
    return m


def _hist(xs):
    h = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return h


def validate_manifest(m) -> None:
    """Hash + disjointness invariants (a violation is a leak of test items into practice)."""
    if m.get("sha256") != sha256_hex(canon({k: v for k, v in m.items() if k != "sha256"})):
        raise EvalError("split manifest hash mismatch")
    pr = {p["item"] for p in m["practice"]}
    fam, comp = set(m["test_familiar"]), set(m["test_composition"])
    if pr & fam or pr & comp or fam & comp:
        raise EvalError("split manifest: practice / familiar / composition item sets overlap")
    comp_g = {g["id"] for g in m["groups"] if g["heldout_composition"]}
    for g in m["groups"]:
        if g["id"] in comp_g and set(g["items"]) & pr:
            raise EvalError("a composition-held-out group has practised items")
    if comp and not m["composition"]["feasible"]:
        raise EvalError("composition test set present although composition is infeasible")


def write_manifest(m, path) -> None:
    """Freeze: refuses to overwrite a manifest whose hash differs (published before any run)."""
    p = Path(path)
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("sha256") != m["sha256"]:
            raise EvalError(f"{p} already holds a different frozen manifest")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_manifest(path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(m)
    return m


# ------------------------------------------------------------------------------------------------------- config
@dataclass
class EvalConfig:
    run_id: str = "run"
    master_seed: int = 20261010
    kitchen: str = "FloorPlan1"
    arms: tuple = ("none", "places", "places+habits", "full")
    Ks: tuple = (0, 1, 3, 5, 10)
    blocks: int = 5
    protocols: tuple = ("P-familiar", "P-composition")       # + "P-translation" (reported separately)
    world_kinds: tuple = ("reset_fresh", "reset_shared")
    tests_per_block: int = 8                                  # ITEMS per block per protocol (x2 languages)
    layout_pool: int = 3                                      # layouts per block in 'reset_shared'
    familiar_first: bool = True                               # practice order puts groups with held-out familiar tests first
    none_at_every_K: bool = False                             # False: `none` does not learn, K=0 test is its baseline
    plan_source: str = "reference"                            # "reference" (fixed executable plans) | "gemma"
    n_sequences: int = 5
    seq_len: int = 8
    persistent: bool = True
    check_practice_readonly: bool = True
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 7
    low_n_blocks: int = 8
    # --- workflow B (B1): all default OFF and absent from to_dict() while default
    epoch_overlay: bool = False                               # tests on a frozen snapshot run at snapshot epoch + 1 (RAM only)
    test_langs: str = "both"                                  # P-location: "seeded" = one seeded language per test item, same for every arm
    practice_reset: bool = False                              # the RUNNER declares reset_world before each practice episode (the executor never writes)

    _B_DEFAULTS = {"epoch_overlay": False, "test_langs": "both", "practice_reset": False}

    def to_dict(self):
        d = asdict(self)
        for k in ("arms", "Ks", "protocols", "world_kinds"):
            d[k] = list(d[k])
        for k, v in self._B_DEFAULTS.items():
            if d.get(k) == v:
                d.pop(k, None)
        return d

    def sha(self):
        return sha256_hex(canon(self.to_dict()))


def config_from_params(params: dict, **over) -> EvalConfig:
    ev = params.get("eval", {})
    kw = {}
    if "blocks" in ev:
        kw["blocks"] = int(ev["blocks"]["value"])
    if "K" in ev:
        kw["Ks"] = tuple(int(k) for k in ev["K"]["value"])
    if "protocols" in ev:
        kw["protocols"] = tuple(ev["protocols"]["value"])
    kw.update(over)
    return EvalConfig(**kw)


def core_config(params: dict | None = None, **over) -> EvalConfig:
    """Minimal core first (plan section 10 / B3): none vs places at K = 0 and 1, fixed reference plans, reset world,
    5 paired blocks. Habits, lessons, persistent worlds and composition wait until this works."""
    base = dict(arms=("none", "places"), Ks=(0, 1), blocks=5, protocols=("P-familiar",), plan_source="reference",
                persistent=False)
    base.update(over)
    cfg = config_from_params(params or {}, **base)
    return cfg


def location_config(params: dict | None = None, **over) -> EvalConfig:
    """B3 (plan s13 B0-plan B-g): none vs places, K in {0, 1}, canonical world, protocol P-location, fixed reference plans, 5 paired
    blocks of 8 test items, each executed once in one seeded language (identical for every arm), epoch overlay on."""
    base = dict(arms=("none", "places"), Ks=(0, 1), blocks=5, protocols=("P-location",), world_kinds=(CANONICAL,),
                plan_source="reference", persistent=False, tests_per_block=8, epoch_overlay=True, test_langs="seeded",
                practice_reset=True)
    base.update(over)
    return config_from_params(params or {}, **base)


# ---------------------------------------------------------------------------------------------- gate 12.10 allowlist
def final_test_allowlist(path) -> tuple[frozenset, frozenset, str]:
    """(allowed, forbidden, sha) from the final test's frozen split manifest: allowed = nontest minus the dev-pilot items
    (the pilot keeps memory-free dev data), forbidden = every test-family item incl. the capped-out ones."""
    raw = Path(path).read_bytes()
    m = json.loads(raw.decode("utf-8"))
    allowed = frozenset(m["nontest"]) - frozenset(m["dev_pilot"])
    forbidden = frozenset(m["test"]) | frozenset(m.get("test_uncapped_not_scored") or [])
    if allowed & forbidden:
        raise AllowlistError(f"final-test manifest lists items as both non-test and test: {sorted(allowed & forbidden)[:3]}")
    return allowed, forbidden, sha256_hex(raw)


def assert_items_allowed(ids, allowed, forbidden=frozenset(), where="run") -> None:
    """Refuse (never warn): every id must be in `allowed` and none in `forbidden`."""
    ids = set(ids)
    bad_f = sorted(ids & set(forbidden))
    if bad_f:
        raise AllowlistError(f"{where}: final-test item(s) {bad_f[:5]} are not allowed in workflow B (gate 12.10)")
    bad = sorted(ids - set(allowed))
    if bad:
        raise AllowlistError(f"{where}: item(s) {bad[:5]} are outside the workflow-B allowlist (gate 12.10)")


# ----------------------------------------------------------------------------------------------------------- plans
def _stratified_items(item_ids, level_of, n, rng):
    by = {}
    for i in sorted(item_ids):
        by.setdefault(level_of[i], []).append(i)
    for v in by.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by.values()):
        for lvl in sorted(by):
            if by[lvl] and len(out) < n:
                out.append(by[lvl].pop())
    return out


def make_block_plan(manifest, items, cfg: EvalConfig, block: int, world_kind: str) -> dict:
    """Everything random in a block, identical for every arm. Seed namespaces are separate: split / practice / test /
    policy. `practised_at_K[k]` says which test concepts (groups) the first k practice attempts exposed."""
    level_of = {it["id"]: it["level"] for it in items}
    group_of = {i: g["id"] for g in manifest["groups"] for i in g["items"]}
    s_b = derive(cfg.master_seed, "block", block, world_kind)
    maxK = max(cfg.Ks) if cfg.Ks else 0
    pool = [dict(p) for p in manifest["practice"]]
    random.Random(derive(s_b, "practice_order")).shuffle(pool)
    if cfg.familiar_first:
        # Declared design: a small K can only show concept reuse (P-familiar) if the first attempts exercise concepts that
        # have held-out partners. Stable partition of the seeded order; coverage per K is reported, not hidden.
        fam_groups = {group_of[i] for i in manifest["test_familiar"]}
        pool = [p for p in pool if group_of[p["item"]] in fam_groups] + [p for p in pool if group_of[p["item"]] not in fam_groups]
    layouts = [derive(s_b, "layout", j) for j in range(cfg.layout_pool)]

    def ws_practice(pos):
        if world_kind == CANONICAL:
            return 0                                   # one canonical world; the seed is unused by the executor
        return derive(s_b, "ws_practice", pos) if world_kind == "reset_fresh" else layouts[derive(s_b, "lp", pos) % len(layouts)]

    def ws_test(tag, item, lang):
        # lang is NOT part of the world seed: the EN and FA variants of one test item run in the SAME world (paired languages)
        if world_kind == CANONICAL:
            return 0
        return derive(s_b, "ws_test", tag, item) if world_kind == "reset_fresh" else layouts[derive(s_b, "lt", tag, item) % len(layouts)]

    practice = []
    for pos, p in enumerate(pool[:maxK], 1):
        practice.append({"pos": pos, "item": p["item"], "lang": p["lang"], "level": level_of[p["item"]],
                         "world_seed": ws_practice(pos), "policy_seed": derive(s_b, "policy", "practice", pos)})
    practised_groups = {k: sorted({group_of[p["item"]] for p in practice[:k]}) for k in cfg.Ks}

    tests = {}
    for proto in cfg.protocols:
        rng = random.Random(derive(s_b, "test_pick", proto))
        if proto == "P-familiar":
            cover = set(practised_groups.get(maxK, []))
            cand = [i for i in manifest["test_familiar"] if group_of[i] in cover]
            ids = _stratified_items(cand, level_of, cfg.tests_per_block, rng)
            eps = [(i, lg) for i in ids for lg in LANGS]
        elif proto == "P-composition":
            ids = _stratified_items(manifest["test_composition"], level_of, cfg.tests_per_block, rng) if manifest["composition"]["feasible"] else []
            eps = [(i, lg) for i in ids for lg in LANGS]
        elif proto == "P-location":
            # B1: location transfer needs no practised concept; held-out item ids (disjoint from practice), stratified by level,
            # one SEEDED language per item (stateless in item id => identical for every arm and K)
            pool_t = list(manifest["test_familiar"]) + (list(manifest["test_composition"]) if manifest["composition"]["feasible"] else [])
            ids = _stratified_items(pool_t, level_of, cfg.tests_per_block, rng)
            eps = [(i, LANGS[derive(s_b, "loc_lang", i) % 2]) for i in ids] if cfg.test_langs == "seeded" else [(i, lg) for i in ids for lg in LANGS]
        elif proto == "P-translation":
            tr = [t for t in manifest["test_translation"] if t["item"] in {p["item"] for p in practice}]
            eps = [(t["item"], t["lang"]) for t in tr]
        else:
            raise EvalError(f"unknown protocol {proto}")
        tests[proto] = [{"item": i, "lang": lg, "level": level_of[i], "group": group_of[i],
                         "world_seed": ws_test(proto, i, lg), "policy_seed": derive(s_b, "policy", "test", proto, i, lg),
                         "practised_at_K": {str(k): group_of[i] in practised_groups[k] for k in cfg.Ks}}
                        for i, lg in eps]
    if world_kind == "reset_fresh":
        pseeds = {p["world_seed"] for p in practice}
        tseeds = {t["world_seed"] for ts in tests.values() for t in ts}
        if pseeds & tseeds:
            raise EvalError("reset_fresh: a test world seed collides with a practice world seed")
    return {"block": block, "world_kind": world_kind, "seed": s_b, "practice": practice, "tests": tests,
            "layouts": layouts if world_kind == "reset_shared" else None}


def make_sequences(manifest, items, cfg: EvalConfig) -> list[dict]:
    """Persistent-world sequences: same task order and the same initial seed for every arm."""
    level_of = {it["id"]: it["level"] for it in items}
    pool = sorted({p["item"] for p in manifest["practice"]} | set(manifest["test_familiar"]))
    seqs = []
    for s in range(cfg.n_sequences):
        seed = derive(cfg.master_seed, "seq", s)
        rng = random.Random(derive(seed, "order"))
        ids = list(pool)
        rng.shuffle(ids)
        ids = ids[:cfg.seq_len]
        tasks = [{"pos": n, "item": i, "lang": LANGS[derive(seed, "lang", i) % 2], "level": level_of[i],
                  "policy_seed": derive(seed, "policy", n)} for n, i in enumerate(ids, 1)]
        seqs.append({"seq": s, "initial_seed": derive(seed, "init"), "tasks": tasks})
    return seqs


# --------------------------------------------------------------------------------------------- executor interface
@dataclass
class EpisodeRequest:
    cell: str
    episode_id: str                    # stable per cell -> consolidation `cons:<episode>` is idempotent on replay
    run_id: str
    item: dict
    lang: str
    level: str
    mode: str                          # "practice" | "test"
    world: str                         # "reset" | "persistent"
    world_kind: str                    # reset_fresh | reset_shared | persistent
    world_seed: int
    policy_seed: int
    arm: str
    features: frozenset                # which memory channels the executor may read (empty for `none`)
    memory: Any                        # LongTermMemory (mutable in practice, FROZEN in test) or None. Read-only for the executor
    plan_source: str
    kitchen: str
    world_token: Any = None            # persistent worlds: the previous episode's `world_token_out`
    position: int | None = None
    block: int | None = None
    K: int | None = None
    protocol: str | None = None


@dataclass
class EpisodeResult:
    sr: Optional[float] = None
    pc: Optional[float] = None
    tds: Optional[float] = None
    goal_ok: Optional[bool] = None
    found: Optional[bool] = None
    steps_to_find: Optional[float] = None
    walking_m: Optional[float] = None
    opens: Optional[float] = None
    stale_visits: Optional[float] = None
    retries: Optional[float] = None
    steps: Optional[float] = None
    timed_out: bool = False
    inapplicable: Optional[str] = None  # e.g. "goal true at reset" / "object absent": excluded from means, counted
    initial_state_hash: Optional[str] = None   # hash of the world AFTER setup (pairing audit)
    world_token_out: Any = None
    wm: Any = None                      # WorkingMemory to consolidate (practice only); never journaled
    admission: Optional[bool] = None    # the privileged terminal verdict, passed on as a bool only
    gate_pass: Optional[bool] = None    # a memory record passed the steering gate
    applied: Optional[bool] = None      # a retrieved record was actually used
    extra: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "wm"}


class Executor:
    """Interface. `kind` is recorded in the report ('fake' | 'live'). Contract: never write to `req.memory` (the runner
    consolidates `result.wm`); report costs up to termination for failures too; report `inapplicable` instead of a score
    when the goal already holds / an object is absent."""
    kind = "abstract"
    frame = "f0"                       # B1: coordinate frame / map revision this executor's events carry
    map_rev = "m0"

    def execute(self, req: EpisodeRequest) -> EpisodeResult:   # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ memory factory
class LtmFactory:
    """Default factory over the real LongTermMemory (imported lazily so this module stays importable on its own)."""

    def __init__(self, kitchen="FloorPlan1", params=None):
        self.kitchen, self.params = kitchen, params

    def _mods(self):
        sys.path.append(str(HERE))
        import ltm as _ltm
        import ltm_store as _st
        return _ltm, _st

    def create(self, db_path):
        _ltm, _ = self._mods()
        return _ltm.LongTermMemory(db_path, kitchen=self.kitchen, params=self.params)

    def restore(self, snap_dir, db_path):
        _ltm, st = self._mods()
        s = st.Store.restore_snapshot(snap_dir, db_path)
        s.close()
        return _ltm.LongTermMemory(db_path, kitchen=self.kitchen, params=self.params)

    def open_frozen(self, snap_dir, testlog_path):
        _ltm, _ = self._mods()
        return _ltm.LongTermMemory.open_frozen(snap_dir, testlog_path, kitchen=self.kitchen, params=self.params)

    def snapshot(self, mem, directory):
        return mem.snapshot(directory, overwrite=True)

    def verify(self, directory):
        _, st = self._mods()
        return st.Store.verify_snapshot(directory)

    def stats(self, mem):
        return mem.stats()

    def state_hash(self, mem):
        return mem.state_hash()


# ----------------------------------------------------------------------------------------------------- journal
class Journal:
    """Append-only JSON lines, fsync'd. First write per cell wins (replayed practice cells after a resume do not
    duplicate). A torn last line is ignored. A header pins config / manifest / params hashes: a mismatch refuses to mix."""

    def __init__(self, path):
        self.path = Path(path)
        self.cells: dict[str, dict] = {}
        self.order: list[str] = []
        self.torn = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            data = self.path.read_bytes()
            for ln in data.split(b"\n"):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln.decode("utf-8"))
                except ValueError:
                    self.torn = True
                    continue
                if r["cell"] not in self.cells:
                    self.cells[r["cell"]] = r
                    self.order.append(r["cell"])
            self._needs_nl = bool(data) and not data.endswith(b"\n")
        else:
            self._needs_nl = False

    def has(self, cell):
        return cell in self.cells

    def get(self, cell):
        return self.cells.get(cell)

    def header(self):
        return self.cells.get("__header__")

    def append(self, cell, kind, payload) -> bool:
        if cell in self.cells:
            return False
        rec = {"cell": cell, "kind": kind, "payload": payload}
        line = (("\n" if self._needs_nl else "") + json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr) + "\n").encode("utf-8")
        self._needs_nl = False
        with open(self.path, "ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.cells[cell] = rec
        self.order.append(cell)
        return True

    def records(self, kind=None):
        return [self.cells[c]["payload"] for c in self.order if c != "__header__" and (kind is None or self.cells[c]["kind"] == kind)]


# ------------------------------------------------------------------------------------------------------- runner
class EvalRunner:
    def __init__(self, cfg: EvalConfig, items, manifest, executor, workdir, factory=None, params_sha=None,
                 extensions=None, flr_provider: Callable | None = None, items_path_sha=None,
                 allow_items=None, forbid_items=frozenset(), allow_sha=None):
        validate_manifest(manifest)
        self.allow_items, self.forbid_items = (frozenset(allow_items) if allow_items is not None else None), frozenset(forbid_items)
        if self.allow_items is not None:                      # gate 12.10: BEFORE anything is created or run
            assert_items_allowed({it["id"] for it in items}, self.allow_items, self.forbid_items, "items")
            assert_items_allowed({i for g in manifest["groups"] for i in g["items"]}, self.allow_items, self.forbid_items, "manifest")
        self.cfg, self.items, self.manifest, self.ex = cfg, items, manifest, executor
        self.by_id = {it["id"]: it for it in items}
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.factory = factory or LtmFactory(cfg.kitchen)
        self.extensions = extensions or {}            # arm -> consolidation extensions (A5 ops)
        self.flr_provider = flr_provider
        self.journal = Journal(self.work / "journal.jsonl")
        hdr = {"config_sha": cfg.sha(), "manifest_sha": manifest["sha256"], "params_sha": params_sha, "config": cfg.to_dict(),
               "executor_kind": getattr(executor, "kind", "unknown"), "dataset_sha256": manifest["dataset_sha256"]}
        if self.allow_items is not None:
            hdr["allowlist_sha"] = allow_sha
        old = self.journal.header()
        if self.allow_items is not None:                      # ... and on EVERY resume, before any episode executes
            for r in self.journal.records():
                it = r.get("item")
                if it is not None:
                    assert_items_allowed([it], self.allow_items, self.forbid_items, "journal record")
        if old is None:
            self.journal.append("__header__", "header", hdr)
        else:
            o = old["payload"]
            if self.allow_items is not None and o.get("allowlist_sha") != allow_sha:
                raise JournalMismatch(f"journal allowlist_sha differs ({o.get('allowlist_sha')} vs {allow_sha}): refusing to mix runs")
            for k in ("config_sha", "manifest_sha", "params_sha", "dataset_sha256"):
                if o.get(k) != hdr[k]:
                    raise JournalMismatch(f"journal header {k} differs ({o.get(k)} vs {hdr[k]}): refusing to mix runs")
        self.stop_after = None                        # test hook: raise after N executor calls
        self.calls = 0

    # -- helpers -----------------------------------------------------------------------------------------------
    def _call(self, req):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise KeyboardInterrupt("simulated stop")
        return self.ex.execute(req)

    def _cell(self, *parts):
        return "|".join(str(p) for p in (self.cfg.run_id, *parts))

    def _snap(self, mem, final: Path):
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        self.factory.snapshot(mem, tmp)
        bad = self.factory.verify(tmp)
        if bad:
            raise InvariantError("snapshot failed verification: " + "; ".join(bad))
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        return json.loads((final / "manifest.json").read_text(encoding="utf-8"))

    def _snap_valid(self, d: Path) -> bool:
        return d.exists() and (d / "manifest.json").exists() and not self.factory.verify(d)

    def _fresh_db(self, tag) -> Path:
        p = self.work / "work" / tag / "ltm.db"
        if p.parent.exists():
            shutil.rmtree(p.parent)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _request(self, **kw):
        return EpisodeRequest(run_id=self.cfg.run_id, plan_source=self.cfg.plan_source, kitchen=self.cfg.kitchen, **kw)

    # -- learning curve ----------------------------------------------------------------------------------------
    def run_learning_curve(self):
        for block in range(self.cfg.blocks):                          # block-major: partial runs hold complete blocks
            for wk in self.cfg.world_kinds:
                plan = make_block_plan(self.manifest, self.items, self.cfg, block, wk)
                for arm in self.cfg.arms:
                    self._run_arm(plan, arm)

    def _tests_for(self, plan, K):
        for proto in self.cfg.protocols:
            for t in plan["tests"][proto]:
                yield proto, t

    def _run_arm(self, plan, arm):
        cfg, wk, block = self.cfg, plan["world_kind"], plan["block"]
        feats = ARMS[arm]
        ck = lambda k: self.work / "ckpt" / wk / f"b{block}" / arm.replace("+", "_") / f"K{k}"
        if arm == "none":
            ks = list(cfg.Ks) if cfg.none_at_every_K else [min(cfg.Ks)]
            for k in ks:
                self._run_tests(plan, arm, k, None, None)
            return
        Ks = sorted(cfg.Ks)
        # resume: the largest K whose snapshot verifies AND whose ckpt record is journaled
        done = [k for k in Ks if self.journal.has(self._cell("ckpt", wk, block, arm, k)) and self._snap_valid(ck(k))]
        tag = f"lc-{wk}-b{block}-{arm.replace('+', '_')}"
        if done:
            k0 = max(done)
            mem = self.factory.restore(ck(k0), self._fresh_db(tag))
        else:
            k0 = 0
            mem = self.factory.create(self._fresh_db(tag))
            if 0 in Ks:
                self._checkpoint(plan, arm, 0, mem, ck(0), [])
        lat = []
        try:
            if 0 in Ks or done:
                self._run_tests(plan, arm, k0, ck(k0), None)
            i = k0
            maxK = max(Ks)
            while i < maxK:
                i += 1
                p = plan["practice"][i - 1]
                lat.append(self._practice_attempt(plan, arm, p, mem))
                if i in Ks:
                    self._checkpoint(plan, arm, i, mem, ck(i), lat)
                    self._run_tests(plan, arm, i, ck(i), None)
        finally:
            try:
                mem.close()
            except Exception:
                pass

    def _practice_attempt(self, plan, arm, p, mem):
        wk, block = plan["world_kind"], plan["block"]
        cell = self._cell("practice", wk, block, arm, p["pos"], p["item"], p["lang"])
        req = self._request(cell=cell, episode_id=_eid(cell), item=self.by_id[p["item"]], lang=p["lang"], level=p["level"],
                            mode="practice", world="reset", world_kind=wk, world_seed=p["world_seed"],
                            policy_seed=p["policy_seed"], arm=arm, features=ARMS[arm], memory=mem, block=block,
                            position=p["pos"], K=p["pos"])
        if self.cfg.practice_reset:          # B-e: every practice episode is a new epoch; written by the runner, before the check
            mem.reset_world(_eid(cell), "task_reset", frame=getattr(self.ex, "frame", "f0"), map_rev=getattr(self.ex, "map_rev", "m0"))
        h0 = self.factory.state_hash(mem) if self.cfg.check_practice_readonly else None
        res = self._call(req)
        if h0 is not None and self.factory.state_hash(mem) != h0:
            raise InvariantError(f"executor wrote to memory during practice cell {cell}")
        return self._consolidate_and_journal(cell, arm, mem, res, dict(kind="practice", world_kind=wk, block=block, arm=arm,
                                                                          pos=p["pos"], item=p["item"], lang=p["lang"],
                                                                          level=p["level"], world_seed=p["world_seed"]))

    def _consolidate_and_journal(self, cell, arm, mem, res: EpisodeResult, meta) -> float:
        latency, status = 0.0, None
        if mem is not None and res.wm is not None and not res.inapplicable:
            t0 = time.perf_counter()
            out = mem.end_episode(res.wm, admission=res.admission, extensions=self.extensions.get(arm, ()))
            latency = time.perf_counter() - t0
            status = out.get("status")
        elif res.inapplicable:
            status = "inapplicable"
        rec = dict(meta, **res.public())
        rec.update(consolidation_status=status, consolidation_latency_s=round(latency, 6),
                   counted_attempt=True)
        self.journal.append(cell, meta["kind"], rec)
        return latency

    def _checkpoint(self, plan, arm, k, mem, final: Path, lat):
        wk, block = plan["world_kind"], plan["block"]
        man = self._snap(mem, final)
        prev = self.journal.get(self._cell("ckpt", wk, block, arm, k))
        if prev and prev["payload"]["state_hash"] != man["state_hash"]:
            raise InvariantError(f"non-deterministic replay: checkpoint {wk}/b{block}/{arm}/K{k} differs from the journaled one")
        st = self.factory.stats(mem)
        recs = [r for r in self.journal.records("practice") if r["world_kind"] == wk and r["block"] == block and r["arm"] == arm and r["pos"] <= k]
        statuses = {}
        for r in recs:
            statuses[r["consolidation_status"]] = statuses.get(r["consolidation_status"], 0) + 1
        payload = {"kind": "ckpt", "world_kind": wk, "block": block, "arm": arm, "K": k, "state_hash": man["state_hash"],
                   "file_sha256": man["file_sha256"], "E": man.get("E_frozen"),
                   "memory_rows": sum((st.get("rows") or {}).values()), "memory_bytes": st.get("bytes"),
                   "consolidation_latency_mean_s": mean([r["consolidation_latency_s"] for r in recs if r["consolidation_status"] == "committed"]),
                   "consolidation_latency_max_s": max([r["consolidation_latency_s"] for r in recs] or [0.0]),
                   "consolidation_statuses": statuses,
                   "exposure": {"items": [r["item"] for r in recs], "langs": [r["lang"] for r in recs]},
                   "flr": self.flr_provider(mem) if self.flr_provider else None}
        self.journal.append(self._cell("ckpt", wk, block, arm, k), "ckpt", payload)

    def _run_tests(self, plan, arm, k, snap_dir: Path | None, _unused):
        wk, block = plan["world_kind"], plan["block"]
        pending = []
        for proto, t in self._tests_for(plan, k):
            cell = self._cell("test", wk, block, arm, proto, f"K{k}", t["item"], t["lang"], t["world_seed"])
            if not self.journal.has(cell):
                pending.append((proto, t, cell))
        if not pending:
            return
        testlog = None
        if snap_dir is not None:
            testlog = self.work / "testlogs" / f"{wk}-b{block}-{arm.replace('+', '_')}-K{k}.db"
            testlog.parent.mkdir(parents=True, exist_ok=True)
        for proto, t, cell in pending:
            frozen, h0, f0, h1 = None, None, None, None
            if snap_dir is not None:
                frozen = self.factory.open_frozen(snap_dir, testlog)
                h0, f0 = self.factory.state_hash(frozen), file_sha(Path(snap_dir) / "ltm.db")
                if self.cfg.epoch_overlay:                      # B1: every test is independent: snapshot epoch + 1, RAM only
                    frozen.begin_test_epoch(frame=getattr(self.ex, "frame", "f0"), map_rev=getattr(self.ex, "map_rev", "m0"))
            req = self._request(cell=cell, episode_id=_eid(cell), item=self.by_id[t["item"]], lang=t["lang"], level=t["level"],
                                mode="test", world="reset", world_kind=wk, world_seed=t["world_seed"],
                                policy_seed=t["policy_seed"], arm=arm, features=ARMS[arm], memory=frozen, block=block, K=k,
                                protocol=proto)
            res = self._call(req)
            if frozen is not None:
                h1, f1 = self.factory.state_hash(frozen), file_sha(Path(snap_dir) / "ltm.db")
                try:
                    frozen.close()
                except Exception:
                    pass
                if h1 != h0 or f1 != f0:
                    raise InvariantError(f"frozen memory changed during test cell {cell}")
            rec = dict(kind="test", world_kind=wk, block=block, arm=arm, protocol=proto, K=k, item=t["item"], lang=t["lang"],
                       level=t["level"], world_seed=t["world_seed"], group=t["group"] if "group" in t else None,
                       practised_at_K=t["practised_at_K"].get(str(k)) if t.get("practised_at_K") else None,
                       frozen_hash_before=h0, frozen_hash_after=h1, memory_read_only=frozen is not None)
            rec.update(res.public())
            rec["counted_attempt"] = False
            self.journal.append(cell, "test", rec)

    # -- persistent worlds -------------------------------------------------------------------------------------
    def run_persistent(self):
        seqs = make_sequences(self.manifest, self.items, self.cfg)
        for sq in seqs:                                               # sequence-major, arms inside
            for arm in self.cfg.arms:
                self._run_sequence(sq, arm)

    def _run_sequence(self, sq, arm):
        s = sq["seq"]
        n = len(sq["tasks"])
        ck = lambda p: self.work / "ckpt" / "persistent" / f"s{s}" / arm.replace("+", "_") / f"P{p}"
        cellp = lambda p, t=None: self._cell("seq", s, arm, p, *( [t["item"], t["lang"]] if t else []))
        mem, token, start = None, None, 0
        if arm != "none":
            done = [p for p in range(1, n + 1) if any(c.startswith(self._cell("seq", s, arm, p, "")) for c in self.journal.cells)
                    and self._snap_valid(ck(p))]
            tag = f"seq-s{s}-{arm.replace('+', '_')}"
            if done:
                start = max(done)
                mem = self.factory.restore(ck(start), self._fresh_db(tag))
            else:
                mem = self.factory.create(self._fresh_db(tag))
        else:
            done = [p for p in range(1, n + 1) if any(c.startswith(self._cell("seq", s, arm, p, "")) for c in self.journal.cells)]
            start = 0
            while start < n and any(c.startswith(self._cell("seq", s, arm, start + 1, "")) for c in self.journal.cells):
                start += 1
        if start:
            prev = [r for c, r in self.journal.cells.items() if c.startswith(self._cell("seq", s, arm, start, ""))][0]["payload"]
            token = prev.get("world_token_out")
        try:
            for t in sq["tasks"][start:]:
                p = t["pos"]
                cell = cellp(p, t)
                req = self._request(cell=cell, episode_id=_eid(cell), item=self.by_id[t["item"]], lang=t["lang"], level=t["level"],
                                    mode="practice", world="persistent", world_kind="persistent",
                                    world_seed=sq["initial_seed"], policy_seed=t["policy_seed"], arm=arm,
                                    features=ARMS[arm], memory=mem, world_token=token, position=p)
                h0 = self.factory.state_hash(mem) if (mem is not None and self.cfg.check_practice_readonly) else None
                res = self._call(req)
                if h0 is not None and self.factory.state_hash(mem) != h0:
                    raise InvariantError(f"executor wrote to memory in sequence cell {cell}")
                self._consolidate_and_journal(cell, arm, mem, res, dict(kind="seq", world_kind="persistent", seq=s, arm=arm, pos=p,
                                                                        item=t["item"], lang=t["lang"], level=t["level"],
                                                                        world_seed=sq["initial_seed"]))
                token = res.world_token_out
                if mem is not None:
                    self._snap(mem, ck(p))
        finally:
            if mem is not None:
                try:
                    mem.close()
                except Exception:
                    pass

    def run_all(self):
        self.run_learning_curve()
        if self.cfg.persistent:
            self.run_persistent()
        return self.journal


# ---------------------------------------------------------------------------------------------------- statistics
def bootstrap_ci(values, resamples=10000, seed=7, alpha=0.05):
    r = _boot(tuple(round(v, 12) for v in values), resamples, seed, alpha)
    return list(r) if r else None


@lru_cache(maxsize=None)
def _boot(values, resamples, seed, alpha):
    return _boot_impl(values, resamples, seed, alpha)


def _boot_impl(values, resamples, seed, alpha):
    """Percentile CI of the MEAN over the given units (blocks or whole sequences). The same (n, seed, resamples) always
    draws the same index sets, so a family of paired statistics (e.g. K-to-K gains) shares one resampling of the units."""
    n = len(values)
    if n < 2:
        return None
    rng = random.Random(seed)
    ms = []
    for _ in range(resamples):
        s = 0.0
        for _j in range(n):
            s += values[rng.randrange(n)]
        ms.append(s / n)
    ms.sort()
    lo = ms[min(len(ms) - 1, int(alpha / 2 * resamples))]
    hi = ms[max(0, int((1 - alpha / 2) * resamples) - 1)]
    return [lo, hi]


def _sel(rec, level, lang):
    return (level in (None, "all") or rec.get("level") == level) and (lang in (None, "all") or rec.get("lang") == lang)


def _val(rec, metric, successful_only):
    if rec.get("inapplicable"):
        return None
    if successful_only:
        ok = rec.get("goal_ok")
        ok = (rec.get("sr") or 0) >= 1 if ok is None else ok
        if not ok:
            return None
    return _num(rec.get(metric))


def block_values(recs, metric, level="all", lang="all", successful_only=False, unit="block"):
    by = {}
    for r in recs:
        if not _sel(r, level, lang):
            continue
        v = _val(r, metric, successful_only)
        if v is not None:
            by.setdefault(r[unit], []).append(v)
    return {b: mean(v) for b, v in by.items()}


def paired_comparison(recs_a, recs_ref, metric, level, lang, resamples, seed, low_n, successful_only=False, unit="block"):
    va = block_values(recs_a, metric, level, lang, successful_only, unit)
    vr = block_values(recs_ref, metric, level, lang, successful_only, unit)
    units = sorted(set(va) & set(vr))
    diffs = [va[b] - vr[b] for b in units]
    return {"n_units": len(units), "diffs": {str(b): round(d, 6) for b, d in zip(units, diffs)},
            "mean_diff": mean(diffs), "ci95": bootstrap_ci(diffs, resamples, seed) if len(diffs) >= 2 else None,
            "n_positive": sum(1 for d in diffs if d > 0), "n_negative": sum(1 for d in diffs if d < 0),
            "low_n": len(diffs) < low_n, "mean_a": mean([va[b] for b in units]), "mean_ref": mean([vr[b] for b in units])}


# ---------------------------------------------------------------------------------------------------- reporting
def pairing_audit(tests: list[dict]) -> dict:
    """Initial-state hash (after setup) must agree across arms / K for the same episode. Seeds equal is not enough."""
    seen, bad, checked = {}, [], 0
    for r in tests:
        h = r.get("initial_state_hash")
        if h is None:
            continue
        key = (r["world_kind"], r.get("block", r.get("seq")), r.get("protocol"), r.get("pos") if r.get("kind") != "test" else None,
               r["item"], r["lang"], r["world_seed"])
        if key in seen and seen[key][0] != h:
            bad.append({"key": list(key), "a": seen[key][1], "b": r["arm"], "K": r.get("K")})
        seen.setdefault(key, (h, r["arm"]))
        checked += 1
    return {"checked": checked, "violations": bad, "ok": not bad}


def _rates(recs, field_):
    v = [r[field_] for r in recs if r.get(field_) is not None and not r.get("inapplicable")]
    return {"n": len(v), "rate": mean([1.0 if x else 0.0 for x in v])}


def build_report(journal_path, cfg_override: EvalConfig | None = None, manifest=None) -> dict:
    j = Journal(journal_path)
    hdr = j.header()["payload"]
    cfgd = hdr["config"]
    resamples = cfgd["bootstrap_resamples"]
    seed = cfgd["bootstrap_seed"]
    low_n = cfgd["low_n_blocks"]
    tests, practice, ckpts, seqs = j.records("test"), j.records("practice"), j.records("ckpt"), j.records("seq")
    kind = hdr["executor_kind"]
    caveats = [
        "Inference is conditional on the fixed task split (manifest sha %s); it does not generalise to unseen tasks." % hdr["manifest_sha"][:12],
        "Arm trajectories differ (each arm practises with its own memory): differences are the TOTAL effect of learning with memory, "
        "not memory content under equal experience.",
        "K counts completed practice attempts (failures and inapplicable included).",
        "reset_fresh: only structural knowledge (receptacles, habits) can transfer; reset_shared: layouts recur, so object-location "
        "transfer is possible (layout recurrence). They are reported separately and must not be pooled.",
        "Successful-only cost rows are descriptive: success selects different episodes per arm.",
        "Fewer than %d blocks: confidence intervals are weak (low-n); all raw block differences are listed." % low_n,
    ]
    if cfgd["plan_source"] == "reference":
        caveats.append("Fixed reference plans: PC/TDS are ceiling checks of execution, not evidence of better planning or bilingual understanding.")
    if kind != "live":
        caveats.append("EXECUTOR KIND = %r: these numbers come from a non-live executor and are NOT evidence about the agent." % kind)
    report = {"schema": SCHEMA, "run_id": cfgd["run_id"], "executor_kind": kind, "config": cfgd, "config_sha": hdr["config_sha"],
              "manifest_sha": hdr["manifest_sha"], "params_sha": hdr["params_sha"], "dataset_sha256": hdr["dataset_sha256"],
              "caveats": caveats, "journal_cells": len(j.order), "journal_torn": j.torn}

    # pairing + invariants
    report["pairing"] = pairing_audit(tests + practice + seqs)
    report["invariants"] = {"frozen_tests": sum(1 for t in tests if t.get("memory_read_only")),
                            "frozen_hash_mismatches": sum(1 for t in tests if t.get("memory_read_only") and t.get("frozen_hash_before") != t.get("frozen_hash_after"))}

    groups = sorted({(t["world_kind"], t["protocol"]) for t in tests})
    arms = [a for a in ARM_ORDER if a in cfgd["arms"]]
    Ks = sorted({t["K"] for t in tests})
    means, comps, gains, coverage = [], [], [], []
    selectors = [("all", "all")] + [(l, "all") for l in LEVELS] + [("all", g) for g in LANGS]
    for wk, proto in groups:
        for arm in arms:
            for K in Ks:
                rows = _arm_tests(tests, wk, proto, arm, K, cfgd)
                if not rows:
                    continue
                for lvl, lg in selectors + [(l, g) for l in LEVELS for g in LANGS]:
                    sub = [r for r in rows if _sel(r, lvl, lg)]
                    if not sub:
                        continue
                    app = [r for r in sub if not r.get("inapplicable")]
                    row = {"world_kind": wk, "protocol": proto, "arm": arm, "K": K, "level": lvl, "lang": lg,
                           "n_episodes": len(sub), "n_inapplicable": len(sub) - len(app),
                           "n_blocks": len({r["block"] for r in app})}
                    for m in ["sr", "pc", "tds", "goal_ok", "found"] + COST_METRICS:
                        bv = block_values(sub, m)
                        row[m] = mean(bv.values()) if bv else None
                        row["n_" + m] = sum(1 for r in app if _num(r.get(m)) is not None)
                    for m in COST_METRICS:
                        bv = block_values(sub, m, successful_only=True)
                        row[m + "_succ_only"] = mean(bv.values()) if bv else None
                    row["gate_pass"] = _rates(sub, "gate_pass")
                    row["applied"] = _rates(sub, "applied")
                    if proto == "P-familiar":
                        pr = [r for r in sub if r.get("practised_at_K") is not None]
                        row["practised_coverage"] = mean([1.0 if r["practised_at_K"] else 0.0 for r in pr]) if pr else None
                    means.append(row)
        # paired comparisons against `none`
        for arm in arms:
            if arm == "none":
                continue
            for K in Ks:
                a = _arm_tests(tests, wk, proto, arm, K, cfgd)
                ref = _arm_tests(tests, wk, proto, "none", K, cfgd)
                if not a or not ref:
                    continue
                for lvl, lg in selectors:
                    for m in ["sr", "pc", "tds", "goal_ok"] + COST_METRICS:
                        c = paired_comparison(a, ref, m, lvl, lg, resamples, seed, low_n)
                        if c["n_units"]:
                            comps.append(dict(c, world_kind=wk, protocol=proto, arm=arm, ref="none", K=K, level=lvl, lang=lg, metric=m,
                                              successful_only=False))
                    for m in COST_METRICS:
                        c = paired_comparison(a, ref, m, lvl, lg, resamples, seed, low_n, successful_only=True)
                        if c["n_units"]:
                            comps.append(dict(c, world_kind=wk, protocol=proto, arm=arm, ref="none", K=K, level=lvl, lang=lg, metric=m,
                                              successful_only=True, descriptive=True))
                # K-to-K gain within an arm (same resampling of blocks => pairing preserved)
                for K0 in Ks:
                    if K0 >= K:
                        continue
                    a0 = _arm_tests(tests, wk, proto, arm, K0, cfgd)
                    if not a0:
                        continue
                    for m in ["sr", "pc", "tds"]:
                        g = paired_comparison(a, a0, m, "all", "all", resamples, seed, low_n)
                        if g["n_units"]:
                            gains.append(dict(g, world_kind=wk, protocol=proto, arm=arm, K=K, K_from=K0, metric=m))
    report["means"], report["comparisons"], report["k_gains"] = means, comps, gains
    cp = [dict(c) for c in ckpts]
    report["checkpoints"] = _ckpt_summary(cp, arms)
    report["training"] = _training_summary(practice)
    report["persistent"] = _persistent_summary(seqs, arms, resamples, seed, low_n)
    report["inapplicable"] = {"tests": sum(1 for t in tests if t.get("inapplicable")), "practice": sum(1 for t in practice if t.get("inapplicable")),
                              "sequence": sum(1 for t in seqs if t.get("inapplicable"))}
    return report


def _arm_tests(tests, wk, proto, arm, K, cfgd):
    rows = [t for t in tests if t["world_kind"] == wk and t["protocol"] == proto and t["arm"] == arm and t["K"] == K]
    if not rows and arm == "none" and not cfgd["none_at_every_K"]:
        rows = [t for t in tests if t["world_kind"] == wk and t["protocol"] == proto and t["arm"] == "none"]   # K-independent baseline
    return rows


def _ckpt_summary(cp, arms):
    out = []
    for arm in arms:
        for wk in sorted({c["world_kind"] for c in cp}):
            for K in sorted({c["K"] for c in cp}):
                sub = [c for c in cp if c["arm"] == arm and c["world_kind"] == wk and c["K"] == K]
                if not sub:
                    continue
                flr = [c["flr"] for c in sub if c.get("flr")]
                out.append({"arm": arm, "world_kind": wk, "K": K, "n_blocks": len(sub),
                            "memory_rows": mean([c["memory_rows"] for c in sub]), "memory_bytes": mean([c["memory_bytes"] for c in sub if c["memory_bytes"] is not None]),
                            "consolidation_latency_mean_s": mean([c["consolidation_latency_mean_s"] for c in sub if c["consolidation_latency_mean_s"] is not None]),
                            "consolidation_latency_max_s": max(c["consolidation_latency_max_s"] for c in sub),
                            "flr": flr[0] if len(flr) == 1 else (flr or "not measured")})
    return out


def _training_summary(practice):
    by = {}
    for r in practice:
        by.setdefault((r["world_kind"], r["arm"]), []).append(r)
    out = []
    for (wk, arm), v in sorted(by.items()):
        srs = [_num(r.get("sr")) for r in v if _num(r.get("sr")) is not None and not r.get("inapplicable")]
        out.append({"world_kind": wk, "arm": arm, "n_attempts": len(v), "n_inapplicable": sum(1 for r in v if r.get("inapplicable")),
                    "sr": mean(srs), "statuses": _hist([r.get("consolidation_status") for r in v])})
    return out


def _persistent_summary(seqs, arms, resamples, seed, low_n):
    if not seqs:
        return None
    per = {}
    for r in seqs:
        per.setdefault((r["arm"], r["seq"]), []).append(r)
    out = {"unit": "whole sequence (fresh memory per sequence, matched initial seed and task order across arms)",
           "independence": "sequences are independent only conditional on the fixed task pool and distinct initial seeds",
           "failure_continuation": "the sequence continues after a failed task; inapplicable positions are counted and excluded",
           "arms": {}, "comparisons": [], "by_position": []}
    seq_rec = []
    for (arm, s), rs in sorted(per.items()):
        for m in ["sr", "pc", "tds", "goal_ok", "found"] + COST_METRICS:
            vals = [_val(r, m, False) for r in rs]
            vals = [v for v in vals if v is not None]
            seq_rec.append({"arm": arm, "seq": s, "metric": m, "value": mean(vals), "n": len(vals), "n_inapplicable": sum(1 for r in rs if r.get("inapplicable")), "n_positions": len(rs)})
    for arm in arms:
        out["arms"][arm] = {m: mean([x["value"] for x in seq_rec if x["arm"] == arm and x["metric"] == m and x["value"] is not None])
                            for m in ["sr", "pc", "tds", "goal_ok"] + COST_METRICS}
    for arm in arms:
        if arm == "none":
            continue
        for m in ["sr", "pc", "tds", "goal_ok"] + COST_METRICS:
            va = {x["seq"]: x["value"] for x in seq_rec if x["arm"] == arm and x["metric"] == m and x["value"] is not None}
            vr = {x["seq"]: x["value"] for x in seq_rec if x["arm"] == "none" and x["metric"] == m and x["value"] is not None}
            units = sorted(set(va) & set(vr))
            diffs = [va[u] - vr[u] for u in units]
            if diffs:
                out["comparisons"].append({"arm": arm, "ref": "none", "metric": m, "n_units": len(units), "mean_diff": mean(diffs),
                                           "ci95": bootstrap_ci(diffs, resamples, seed), "low_n": len(diffs) < low_n,
                                           "diffs": {str(u): round(d, 6) for u, d in zip(units, diffs)}})
    for arm in arms:
        for pos in sorted({r["pos"] for r in seqs}):
            vals = [_num(r.get("sr")) for r in seqs if r["arm"] == arm and r["pos"] == pos and not r.get("inapplicable") and _num(r.get("sr")) is not None]
            out["by_position"].append({"arm": arm, "pos": pos, "sr": mean(vals), "n": len(vals)})
    return out


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(rep: dict) -> str:
    L = [f"# Step 5 evaluation report - {rep['run_id']}", "",
         f"- executor kind: **{rep['executor_kind']}**  ·  plan source: {rep['config']['plan_source']}  ·  blocks: {rep['config']['blocks']}  ·  K: {rep['config']['Ks']}",
         f"- config sha {rep['config_sha'][:12]}  ·  manifest sha {rep['manifest_sha'][:12]}  ·  params sha {(rep['params_sha'] or '-')[:12]}  ·  dataset sha {rep['dataset_sha256'][:12]}",
         f"- journal cells {rep['journal_cells']}  ·  pairing audit: {'OK' if rep['pairing']['ok'] else 'VIOLATIONS %d' % len(rep['pairing']['violations'])} "
         f"({rep['pairing']['checked']} checked)  ·  frozen tests {rep['invariants']['frozen_tests']} (hash mismatches {rep['invariants']['frozen_hash_mismatches']})", "",
         "## Caveats (read first)", ""] + [f"- {c}" for c in rep["caveats"]] + [""]
    L += ["## Means per arm (all levels, both languages; block means)", "",
          "| world | protocol | arm | K | n eps | n skip | SR | PC (L2) | TDS (L3) | steps-to-find | walk m | opens | stale | retries |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["means"]:
        if r["level"] == "all" and r["lang"] == "all":
            L.append(f"| {r['world_kind']} | {r['protocol']} | {r['arm']} | {r['K']} | {r['n_episodes']} | {r['n_inapplicable']} | {_f(r['sr'])} | {_f(r['pc'])} | {_f(r['tds'])} | "
                     f"{_f(r['steps_to_find'])} | {_f(r['walking_m'])} | {_f(r['opens'])} | {_f(r['stale_visits'])} | {_f(r['retries'])} |")
    L += ["", "## Primary metric per level (L1 SR, L2 PC, L3 TDS)", "", "| world | protocol | arm | K | lang | L1 SR | L2 PC | L3 TDS |", "|---|---|---|---|---|---|---|---|"]
    idx = {(r["world_kind"], r["protocol"], r["arm"], r["K"], r["level"], r["lang"]): r for r in rep["means"]}
    for (wk, pr_, arm, K, lvl, lg), r in sorted(idx.items()):
        if lvl == "all" and lg in LANGS:
            g = lambda l: _f(idx.get((wk, pr_, arm, K, l, lg), {}).get(PRIMARY[l]))
            L.append(f"| {wk} | {pr_} | {arm} | {K} | {lg} | {g('L1')} | {g('L2')} | {g('L3')} |")
    L += ["", "## Paired differences vs `none` (all levels, both languages; mean over blocks, bootstrap 95% CI over blocks)", "",
          "| world | protocol | arm | K | metric | mean diff | 95% CI | blocks | +/- | per-block diffs | note |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        if c["level"] == "all" and c["lang"] == "all" and not c["successful_only"] and c["metric"] in ("sr", "steps_to_find", "walking_m", "opens", "stale_visits"):
            ci = "-" if c["ci95"] is None else f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"
            note = "low-n" if c["low_n"] else ""
            L.append(f"| {c['world_kind']} | {c['protocol']} | {c['arm']} | {c['K']} | {c['metric']} | {_f(c['mean_diff'])} | {ci} | {c['n_units']} | {c['n_positive']}/{c['n_negative']} | "
                     f"{', '.join(f'{k}:{v:+.2f}' for k, v in c['diffs'].items())} | {note} |")
    if rep["k_gains"]:
        L += ["", "## Learning gains within an arm (K vs earlier K)", "", "| world | protocol | arm | metric | K from -> to | mean diff | 95% CI | blocks |", "|---|---|---|---|---|---|---|---|"]
        for g in rep["k_gains"]:
            ci = "-" if g["ci95"] is None else f"[{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]"
            L.append(f"| {g['world_kind']} | {g['protocol']} | {g['arm']} | {g['metric']} | {g['K_from']} -> {g['K']} | {_f(g['mean_diff'])} | {ci} | {g['n_units']} |")
    L += ["", "## Memory size, consolidation latency, FLR (checkpoints)", "", "| arm | world | K | rows | bytes | latency mean s | latency max s | FLR |", "|---|---|---|---|---|---|---|---|"]
    for c in rep["checkpoints"]:
        L.append(f"| {c['arm']} | {c['world_kind']} | {c['K']} | {_f(c['memory_rows'], 1)} | {_f(c['memory_bytes'], 0)} | {_f(c['consolidation_latency_mean_s'], 4)} | {_f(c['consolidation_latency_max_s'], 4)} | {c['flr'] if isinstance(c['flr'], str) else 'measured'} |")
    if rep["persistent"]:
        p = rep["persistent"]
        L += ["", "## Persistent worlds", "", f"- unit: {p['unit']}", f"- {p['independence']}", f"- {p['failure_continuation']}", "",
              "| arm | metric | mean diff vs none | 95% CI | sequences | low-n |", "|---|---|---|---|---|---|"]
        for c in p["comparisons"]:
            if c["metric"] in ("sr", "steps_to_find", "walking_m", "opens", "stale_visits"):
                ci = "-" if c["ci95"] is None else f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"
                L.append(f"| {c['arm']} | {c['metric']} | {_f(c['mean_diff'])} | {ci} | {c['n_units']} | {c['low_n']} |")
    L += ["", f"Inapplicable (skipped, excluded from every mean): {rep['inapplicable']}", ""]
    return "\n".join(L)


def write_report(journal_path, out_dir) -> dict:
    rep = build_report(journal_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


# ------------------------------------------------------------------------------------------------------ CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5 evaluation runner (offline code; live executor injected)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="build + freeze the split manifest")
    m.add_argument("--items", default=str(HERE / "dataset" / "general.json"))
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=20261010)
    m.add_argument("--allow-final-test-manifest", default=None,
                   help="gate 12.10: keep only the final test's non-test, non-dev-pilot items")
    r = sub.add_parser("report", help="rebuild report.md/json from a journal")
    r.add_argument("--journal", required=True)
    r.add_argument("--out", required=True)
    n = sub.add_parser("run", help="run with an injected executor factory 'module:function' (workflow B)")
    n.add_argument("--executor", required=True)
    n.add_argument("--workdir", required=True)
    n.add_argument("--manifest", required=True)
    n.add_argument("--items", default=str(HERE / "dataset" / "general.json"))
    n.add_argument("--preset", choices=["core", "location", "full"], default="core")
    n.add_argument("--executor-args", default="{}", help="JSON keyword arguments for the executor factory, e.g. '{\"memory\": \"places\", \"hints\": true}'")
    n.add_argument("--allow-final-test-manifest", default=None,
                   help="gate 12.10: refuse any item outside nontest minus dev_pilot (checked at start and on every resume)")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        items, sha = load_items(a.items)
        if a.allow_final_test_manifest:
            allowed, forbidden, _ = final_test_allowlist(a.allow_final_test_manifest)
            items = [it for it in items if it["id"] in allowed]
            assert_items_allowed({it["id"] for it in items}, allowed, forbidden, "manifest build")
        man = build_split(items, sha, a.seed)
        write_manifest(man, a.out)
        print(json.dumps(man["stats"], indent=1), man["composition"], man["sha256"][:16])
    elif a.cmd == "report":
        write_report(a.journal, a.out)
        print("report written to", a.out)
    else:
        import importlib
        sys.path.append(str(HERE))
        mod, fn = a.executor.split(":")
        ex = getattr(importlib.import_module(mod), fn)(**json.loads(a.executor_args))
        items, _ = load_items(a.items)
        man = load_manifest(a.manifest)
        sys.path.append(str(HERE))
        import ltm as _ltm
        P = _ltm.load_params()
        cfg = core_config(P) if a.preset == "core" else location_config(P) if a.preset == "location" else config_from_params(P)
        allow, forbid, allow_sha = (None, frozenset(), None)
        if a.allow_final_test_manifest:
            allow, forbid, allow_sha = final_test_allowlist(a.allow_final_test_manifest)
            items = [it for it in items if it["id"] in allow]
        runner = EvalRunner(cfg, items, man, ex, a.workdir, factory=LtmFactory(cfg.kitchen, P), params_sha=P.get("_sha256"),
                            allow_items=allow, forbid_items=forbid, allow_sha=allow_sha)
        runner.run_all()
        write_report(Path(a.workdir) / "journal.jsonl", Path(a.workdir) / "report")
        print("done; report in", Path(a.workdir) / "report")


if __name__ == "__main__":
    _main()
