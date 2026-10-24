"""Per-kitchen class lists for FloorPlan15 / FloorPlan17 (A28 item 4) -- EXPECTED values and the probe check.

A28 item 4: the class list IS the probe registry (sorted base types present in the scene, ``vocab.base_type``), fixed at
probe time, before capture, recorded with its hash in a dated ledger entry; the probe value is authoritative. This file
only (a) writes the EXPECTED lists from the 30-kitchen scene inventory (sim/dataset/scene_inventory.json), and (b) after
the probe, compares the probe registry with them and prints the registry sha256 for the ledger. A mismatch is not
"fixed" here: it stops the run for a consult (the probe stays authoritative).

    python sim/kitchen_v3/k23/k23_classes.py expected                       # -> sim/kitchen_v3/k23/expected_classes.json
    python sim/kitchen_v3/k23/k23_classes.py check --probe <probe.json>     # exit 0 match, 1 mismatch

Method check: for FloorPlan1 the inventory rule gives exactly the Kitchen 1 probe registry (49 types).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
if SIM not in sys.path:
    sys.path.insert(0, SIM)

import vocab  # noqa: E402

SCENES = ("FloorPlan15", "FloorPlan17")
EXPECTED = os.path.join(HERE, "expected_classes.json")
INVENTORY = os.path.join(SIM, "dataset", "scene_inventory.json")


def excluded_types() -> tuple:
    # sim/type_scope.py is in the frozen Kitchen 1 manifest: read, never edited
    import type_scope
    return tuple(type_scope.EXCLUDED_TYPES)


def registry_sha(types) -> str:
    return hashlib.sha256(json.dumps(sorted(types), separators=(",", ":")).encode()).hexdigest()


def inventory_classes(scene: str, inv: dict) -> dict:
    per = inv["per_scene"][scene]
    counts = {}
    for t, rec in per.items():
        b = vocab.base_type(t)
        counts[b] = counts.get(b, 0) + int((rec or {}).get("count") or 1)
    return dict(sorted(counts.items()))


def build_expected() -> dict:
    inv = json.load(open(INVENTORY, encoding="utf-8"))
    ex = excluded_types()
    k1_probe = json.load(open(os.path.join(SIM, "kitchen_v3", "FloorPlan1", "probe.json"), encoding="utf-8"))
    k1 = sorted(inventory_classes("FloorPlan1", inv))
    out = {"what": "EXPECTED class lists from the scene inventory; the probe registry is authoritative (A28 item 4)",
           "inventory": os.path.relpath(INVENTORY, os.path.dirname(SIM)).replace("\\", "/"),
           "inventory_generated": inv.get("generated_ts"),
           "excluded_types": list(ex),
           "method_check_FloorPlan1": {"inventory_rule": len(k1), "probe_registry": len(k1_probe["registry"]),
                                       "identical": k1 == sorted(k1_probe["registry"])},
           "scenes": {}}
    for s in SCENES:
        c = inventory_classes(s, inv)
        types = sorted(c)
        out["scenes"][s] = {"n_classes": len(types), "n_shown": len([t for t in types if t not in ex]),
                            "classes": types, "registry_sha256": registry_sha(types),
                            "instances_per_class": c, "n_instances": sum(c.values()),
                            "not_in_FloorPlan1": sorted(set(types) - set(k1)),
                            "FloorPlan1_types_absent": sorted(set(k1) - set(types)),
                            "amb_targets_present": [t for t in ("Fork", "Spoon", "ButterKnife", "Knife", "Spatula",
                                                                "SaltShaker", "PepperShaker", "Plate", "Pot", "Pan")
                                                    if t in c]}
    return out


def check_probe(probe_path: str, expected: dict) -> dict:
    p = json.load(open(probe_path, encoding="utf-8"))
    scene = p["scene"]
    reg = sorted(p["registry"])
    exp = expected["scenes"].get(scene)
    res = {"scene": scene, "n_probe": len(reg), "probe_registry_sha256": registry_sha(reg),
           "n_shown": len([t for t in reg if t not in expected["excluded_types"]])}
    if exp is None:
        res.update(ok=False, reason=f"{scene} is not a K23 kitchen")
    else:
        res.update(ok=reg == exp["classes"], n_expected=exp["n_classes"],
                   missing_vs_expected=sorted(set(exp["classes"]) - set(reg)),
                   extra_vs_expected=sorted(set(reg) - set(exp["classes"])))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("expected")
    c = sub.add_parser("check")
    c.add_argument("--probe", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "expected":
        out = build_expected()
        with open(EXPECTED, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        for s, v in out["scenes"].items():
            print(f"{s}: {v['n_classes']} classes ({v['n_shown']} shown), {v['n_instances']} instances, "
                  f"registry sha {v['registry_sha256'][:16]}")
        print(f"method check FloorPlan1: {out['method_check_FloorPlan1']}")
        return 0
    res = check_probe(a.probe, json.load(open(EXPECTED, encoding="utf-8")))
    print(json.dumps(res, indent=1))
    print(f"CLASS LIST {res['scene']}: {'MATCH' if res['ok'] else 'MISMATCH -> stop and consult'} "
          f"({res['n_probe']} classes, sha {res['probe_registry_sha256'][:16]})")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
