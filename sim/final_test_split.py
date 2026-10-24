"""Final test (specialisation vs model complexity) -- the frozen task-family split. Pure, offline, stdlib only.

Preregistration: docs/final-test/preregistration.md section 5. Run from the repo root:

    python sim/final_test_split.py                     # print the counts
    python sim/final_test_split.py --write             # write docs/final-test/split_manifest.json (refuses to overwrite)

Rules (all locked in the preregistration; changing any of them is a dated amendment):
  1. Applicability exclusion (scene-level, not outcome-level): the 10 items the FloorPlan1 applicability gate skipped in
     the step-4 s5 run (goal already true at reset, or the scene has no DiningTable).
  2. Task FAMILY = union-find over items sharing (a) the goal signature, (b) the plan concept key (step5_eval rules), or
     (c) the same level AND the same ordered action skeleton ("near-identical template": pickup Apple ~ pickup Bread).
     EN/FA twins are one item, so they are together by construction.
  3. Families containing an item any LLM planner has already seen (step-4 s6 pilot, chooser check, console runs) are
     forced to NON-TEST before the seeded draw.
  4. Per family level stratum (majority level), the first ceil(NONTEST_FRAC * n) families in a seeded order are non-test
     (forced ones count towards that quota). Everything else is TEST.
  5. Test items per family are capped at TEST_CAP (seeded order inside the family) -- more independent families, fewer
     near-duplicates. Uncapped leftovers are reported, never scored.
  6. Fixed subsets, all seeded: DEV_PILOT (12 non-test items, 4 per item level) for prompt checks and the cost pilot;
     REPEAT (30 test items, 10 per level) for 3 planner calls per arm and 3 executions per cached plan; RELOCATION (10
     test items, stratified) for the contamination / stale-memory probe; the practice pool = all non-test items.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import step5_eval as se  # noqa: E402  (pure module: no simulator, no socket)

SCHEMA = "final-test.split/1"
MASTER_SEED = 20261010
NONTEST_FRAC = 0.25
TEST_CAP = 6
N_DEV_PER_LEVEL = 4
N_REPEAT_PER_LEVEL = 10
N_RELOCATION = 10
#: A26 (K-B2b 2026-10-13): the split was cut from general.json v1 956f0ae8..., now archived; the successor (archived ids removed,
#: docs/final-test/inclusion_manifest_a26.json) never regroups, so the split is rebuilt from the archived original
DATASET = HERE / "dataset" / "archive" / "general.v1.956f0ae8.json"
OUT = ROOT / "docs" / "final-test" / "split_manifest.json"
LEVELS = ("L1", "L2", "L3")

#: rule 1 -- skipped by the applicability gate in sim/step4/refplans_honest.jsonl (status "skipped", both arms)
INAPPLICABLE = frozenset({
    "g-l1-goto-dining-table", "g-l1-goto-fridge", "g-l1-goto-toaster", "g-l2-cabinet-plate", "g-l2-drawer-fork",
    "g-l2-drawer-knife", "g-l2-drawer-spoon", "g-l2-fill-place-cup-wine-dining-table", "g-l2-fridge-egg",
    "g-l3-set-table"})
#: rule 3 -- items an LLM planner has already been run on (sim/step4/s6_trial*.jsonl, sim/step4/console_runs.jsonl)
EXPOSED = frozenset({
    "g-l1-pickup-apple", "g-l1-pickup-bread", "g-l1-close-cabinet", "g-l2-fridge-apple", "g-l3-chilled-apple",
    "g-l3-cold-tomato", "g-l1-pickup-dish-sponge", "g-l1-pickup-potato"})


def skeleton(item) -> str:
    return se.canon([item["level"], [s.get("action") for s in item["plan"]]])


def families(items) -> list[list[dict]]:
    uf = se._UF([it["id"] for it in items])
    keyed = collections.defaultdict(list)
    for it in items:
        keyed[("goal", se.goal_signature(it))].append(it["id"])
        keyed[("plan", se.plan_concept_key(it))].append(it["id"])
        keyed[("skel", skeleton(it))].append(it["id"])
    for ids in keyed.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
    comp = collections.defaultdict(list)
    for it in items:
        comp[uf.find(it["id"])].append(it)
    return sorted((sorted(v, key=lambda i: i["id"]) for v in comp.values()), key=lambda v: v[0]["id"])


def _order(ids, *ns):
    return sorted(ids, key=lambda i: (se.derive(MASTER_SEED, "final-test", *ns, i), i))


def _stratified(items, per_level, *ns):
    out = []
    for lv in LEVELS:
        out += _order([i["id"] for i in items if i["level"] == lv], *ns, lv)[:per_level]
    return sorted(out)


def build(dataset=DATASET) -> dict:
    items, ds_sha = se.load_items(dataset)
    all_ids = {it["id"] for it in items}
    missing = (INAPPLICABLE | EXPOSED) - all_ids
    if missing:
        raise SystemExit(f"declared ids not in the dataset: {sorted(missing)}")
    eligible = [it for it in items if it["id"] not in INAPPLICABLE]
    fams = families(eligible)
    level_of = {it["id"]: it["level"] for it in eligible}
    fam_level = {f[0]["id"]: se._group_level([i["id"] for i in f], level_of) for f in fams}
    fam_of = {i["id"]: f[0]["id"] for f in fams for i in f}
    forced = {fam_of[e] for e in EXPOSED}
    nontest = set(forced)
    for lv in LEVELS:
        stratum = [f[0]["id"] for f in fams if fam_level[f[0]["id"]] == lv]
        need = math.ceil(NONTEST_FRAC * len(stratum))
        have = sum(1 for fid in stratum if fid in nontest)
        rest = _order([fid for fid in stratum if fid not in nontest], "fam")
        nontest.update(rest[:max(0, need - have)])
    test_items, uncapped, nontest_items = [], [], []
    fam_rows = []
    for f in fams:
        fid = f[0]["id"]
        ids = [i["id"] for i in f]
        if fid in nontest:
            nontest_items += ids
            role = "nontest-forced" if fid in forced else "nontest"
        else:
            order = _order(ids, "cap", fid)
            test_items += order[:TEST_CAP]
            uncapped += order[TEST_CAP:]
            role = "test"
        fam_rows.append({"family": fid, "level": fam_level[fid], "role": role, "items": ids,
                         "levels": dict(collections.Counter(level_of[i] for i in ids))})
    by_id = {it["id"]: it for it in eligible}
    test_objs = [by_id[i] for i in test_items]
    repeat = _stratified(test_objs, N_REPEAT_PER_LEVEL, "repeat")
    reloc_pool = [by_id[i] for i in repeat]
    relocation = sorted(_order(_stratified(reloc_pool, math.ceil(N_RELOCATION / 3), "reloc"), "reloc-cut")[:N_RELOCATION])
    dev = _stratified([by_id[i] for i in nontest_items], N_DEV_PER_LEVEL, "dev")
    m = {
        "schema": SCHEMA, "dataset": "sim/dataset/general.json", "dataset_sha256": ds_sha,
        "master_seed": MASTER_SEED, "rules": {"nontest_frac": NONTEST_FRAC, "test_cap": TEST_CAP,
                                             "family_links": ["goal_signature", "plan_concept_key", "level+action_skeleton"],
                                             "grouping_code": "sim/step5_eval.py (goal_signature, plan_concept_key, _UF)"},
        "excluded_inapplicable": sorted(INAPPLICABLE), "exposed_items": sorted(EXPOSED),
        "families": fam_rows,
        "test": sorted(test_items), "test_uncapped_not_scored": sorted(uncapped), "nontest": sorted(nontest_items),
        "practice_pool": sorted(nontest_items), "dev_pilot": dev, "repeat_subset": repeat, "relocation_probe": relocation,
        "stats": {
            "n_items": len(items), "n_eligible": len(eligible), "n_families": len(fams),
            "families_by_level": dict(collections.Counter(fam_level.values())),
            "test_families": sum(1 for r in fam_rows if r["role"] == "test"),
            "test_families_by_level": dict(collections.Counter(r["level"] for r in fam_rows if r["role"] == "test")),
            "test_items": len(test_items), "test_items_by_level": dict(collections.Counter(level_of[i] for i in test_items)),
            "test_presentations": 2 * len(test_items), "uncapped": len(uncapped),
            "nontest_families": sum(1 for r in fam_rows if r["role"] != "test"), "nontest_items": len(nontest_items),
            "nontest_items_by_level": dict(collections.Counter(level_of[i] for i in nontest_items)),
        },
    }
    m["sha256"] = hashlib.sha256(se.canon({k: v for k, v in m.items() if k != "sha256"}).encode()).hexdigest()
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(ROOT)} (never overwrites)")
    a = ap.parse_args(argv)
    m = build()
    print(json.dumps(m["stats"], indent=1))
    print("dev_pilot", m["dev_pilot"])
    print("sha256", m["sha256"])
    if a.write:
        if OUT.exists():
            old = json.loads(OUT.read_text(encoding="utf-8"))
            if old.get("sha256") != m["sha256"]:
                raise SystemExit(f"{OUT} exists with a different sha256; a change is an amendment -- refusing")
            print("unchanged:", OUT)
            return 0
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
