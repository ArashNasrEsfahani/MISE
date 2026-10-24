"""A29 item 2 sampler for the replication kitchens (FloorPlan15, FloorPlan17): one ordered sample per kitchen from the
applicable test pool, frozen BEFORE practice in either kitchen (preregistration A29 item 2, Astra A29 1-3).

Pool = the effective split manifest's `test` items that are applicable in the kitchen (applicability = the registered
reference-validation result, supplied as a file; this module never decides applicability). Order:
  1. eligible families (>= 1 applicable test item) sorted by descending applicable-item count, ties by family id; one
     uniformly random item per family (from its sorted applicable ids);
  2. then at each position t+1 the non-empty level maximizing (t+1) p_l - m_l (p_l = the level's share of the pool, m_l =
     selected so far), ties L1, L2, L3; one uniformly random item of that level (sorted remaining ids).
RNG: C.rng("A29-sample", kitchen) (MASTER_SEED 20261010), one stream for the whole order. Candidates (nested): the first
min(50, pool), the first min(60, pool), all applicable items; each item enters with both languages. A pool under 125
items makes "all" the primary sample and the shortfall is reported.

    python sim/final_test/matrix/a29_sample.py --kitchen FloorPlan15 --applicable <applicability.json>
(writes docs/final-test/k23/a29_sample_<kitchen>.json once; an existing manifest is never overwritten)
applicability.json: {"kitchen": ..., "items": {"<id>": "applicable" | "inapplicable" | "unresolved", ...}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

KITCHENS = ("FloorPlan15", "FloorPlan17")
LEVELS = ("L1", "L2", "L3")
CANDIDATES = (50, 60)
PRIMARY_TARGET = 125
VERSION = "a29-sample-1"


def frozen_path(kitchen: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", f"a29_sample_{kitchen}.json")


def item_levels(dataset_path: str) -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        return {it["id"]: it["level"] for it in json.load(f)["items"]}


def next_level(live: list, cnt: dict, n: int, m: dict, t1: int) -> str:
    """argmax over the non-empty levels of (t+1) p_l - m_l, in exact integers ((t+1) n_l - N m_l); ties L1, L2, L3."""
    return max(live, key=lambda l: (t1 * cnt[l] - n * m[l], -LEVELS.index(l)))


def ordered_sample(kitchen: str, pool: list, family_of: dict, level_of: dict) -> list:
    """The A29 ordered sample (pure). pool: applicable test ids; family_of / level_of: id -> family / level."""
    if kitchen not in KITCHENS:
        raise ValueError(f"A29 sampler is registered for {KITCHENS} only")
    pool = sorted(set(pool))
    missing = [i for i in pool if i not in family_of or level_of.get(i) not in LEVELS]
    if missing:
        raise ValueError(f"items without a family or a level: {missing[:5]}")
    r = C.rng("A29-sample", kitchen)
    fams = {}
    for i in pool:
        fams.setdefault(family_of[i], []).append(i)
    order = []
    for fam in sorted(fams, key=lambda f: (-len(fams[f]), f)):
        order.append(r.choice(sorted(fams[fam])))
    rest = sorted(set(pool) - set(order))
    n = len(pool)
    cnt = {l: sum(1 for i in pool if level_of[i] == l) for l in LEVELS}      # exact integer deficits (t+1) n_l - N m_l
    m = {l: sum(1 for i in order if level_of[i] == l) for l in LEVELS}
    while rest:
        t1 = len(order) + 1
        best = next_level([l for l in LEVELS if any(level_of[i] == l for i in rest)], cnt, n, m, t1)
        pick = r.choice([i for i in rest if level_of[i] == best])
        order.append(pick)
        rest.remove(pick)
        m[best] += 1
    return order


def build(kitchen: str, applicability: dict, root: str | None = None) -> dict:
    if applicability.get("kitchen") != kitchen:
        raise ValueError("applicability file is for another kitchen")
    man = C.load_manifest(root)
    test = set(man["test"])
    family_of = {i: f["family"] for f in man["families"] for i in f["items"]}
    level_of = item_levels(C.paths(root)["dataset"])
    status = applicability["items"]
    unknown = sorted(test - set(status))
    if unknown:
        raise ValueError(f"applicability missing for {len(unknown)} test items (e.g. {unknown[:3]}): resolve or record them first")
    unresolved = sorted(i for i in test if status[i] not in ("applicable", "inapplicable"))
    if unresolved:
        raise ValueError(f"{len(unresolved)} test items not resolved to applicable / inapplicable (e.g. {unresolved[:3]}; "
                         "A29: retry under the same rule before sampling)")
    pool = sorted(i for i in test if status[i] == "applicable")
    order = ordered_sample(kitchen, pool, family_of, level_of)
    cands = {f"first_{k}": order[:min(k, len(order))] for k in CANDIDATES}
    cands["all"] = list(order)
    rep = {"version": VERSION, "kitchen": kitchen, "rng": f"C.rng('A29-sample', '{kitchen}') MASTER_SEED {C.MASTER_SEED}",
           "split_manifest_sha256": man.get("sha256"), "inclusion": man.get("inclusion", {}).get("sha256"),
           "applicability_sha256": C.sha256_json(applicability), "n_test": len(test), "n_pool": len(pool),
           "inapplicable": sorted(i for i in test if status[i] == "inapplicable"),
           "pool_levels": {l: sum(1 for i in pool if level_of[i] == l) for l in LEVELS},
           "n_families": len({family_of[i] for i in pool}), "order": order, "candidates": cands,
           "presentations": {k: 2 * len(v) for k, v in cands.items()},
           "primary_note": None if len(pool) >= PRIMARY_TARGET else
           f"pool {len(pool)} < {PRIMARY_TARGET}: 'all' is the primary sample (shortfall {PRIMARY_TARGET - len(pool)} items)"}
    rep["sha256"] = C.self_sha(rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitchen", required=True, choices=KITCHENS)
    ap.add_argument("--applicable", required=True)
    a = ap.parse_args(argv)
    with open(a.applicable, encoding="utf-8") as f:
        rep = build(a.kitchen, json.load(f))
    out = frozen_path(a.kitchen)                              # ONE authoritative manifest per kitchen (no --out)
    if os.path.exists(out):
        print(f"REFUSED: {out} exists (A29: later decisions select an existing manifest, never redraw)")
        return 2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("kitchen", "n_pool", "pool_levels", "n_families", "presentations", "primary_note")}), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
