"""K23 list building with the K3-SINK-1 structural exception (Arash 2026-10-17 ~21:15; Astra K3-SINK-r1 option A).

The frozen Kitchen 1 detection gate (build_kitchen_data.detection_gate) runs UNCHANGED. Only when the fail-closed
structural scan (k23_qc.structural_scan) is clean, the shortfalls of the scene's structurally unrendered types (and of
their instances, '<Type>_<id>') are removed from that gate; every other shortfall, missing type, failed / low-coverage
episode or missing file still refuses. If the adjusted gate passes, the frozen stage_yolo builds the lists with its
recorded override; both the original and the adjusted gate are written. Scenes without an exception: plain stage_yolo.

    python sim/kitchen_v3/k23/k23_lists.py --scene FloorPlan17          (WSL thor-env)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)


def adjust_gate(gate: dict, excl: tuple) -> dict:
    """The gate without the shortfalls of the excluded types / their instances (pure; for tests)."""
    g = copy.deepcopy(gate)
    is_ex = lambda k: k in excl or str(k).split("_")[0] in excl
    removed = []
    for u, splits in (g.get("detection_short") or {}).items():
        for s, keys in splits.items():
            for k in [k for k in keys if is_ex(k)]:
                removed.append(f"{u}/{s}/{k}")
                keys.pop(k)
    g["missing_types"] = [t for t in g.get("missing_types") or [] if t not in excl]
    g["n_short"] = sum(len(v) for u in g["detection_short"].values() for v in u.values())
    g["ok"] = (not g["n_short"] and not g["missing_types"] and not g.get("failed_episodes")
               and not g.get("low_coverage_episodes") and not g.get("missing_files"))
    g["structural_exception"] = {"types": list(excl), "removed": removed}
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    a = ap.parse_args(argv)
    import build_kitchen_data as bkd
    import k23_qc
    root = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3")
    ds = os.path.join(root, a.scene)
    probe = json.load(open(os.path.join(ds, "probe.json"), encoding="utf-8"))
    out = os.path.join(HERE, a.scene, "clean", f"detection_gate_k23_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sx = k23_qc.structural_scan(ds, a.scene)
    if not sx["types"]:
        res = bkd.stage_yolo(a.scene, root)
        json.dump({"scene": a.scene, "structural": sx, "stage_yolo": "plain (no exception)"}, open(out, "w"), indent=1, default=str)
        print(json.dumps({"lists": "built", "exception": None}))
        return 0
    gate = bkd.detection_gate(ds, probe)
    adj = adjust_gate(gate, tuple(sx["types"])) if sx["ok"] else None
    rec = {"scene": a.scene, "ruling": k23_qc.STRUCTURAL_RULING, "structural_scan": sx, "gate_original": gate,
           "gate_adjusted": adj, "built": False}
    if not sx["ok"]:
        rec["refused"] = "structural exception VOID (the excluded type appeared): the full gate applies"
    elif not adj["ok"]:
        rec["refused"] = "adjusted gate still FAILS"
    else:
        bkd.stage_yolo(a.scene, root, allow_failed_audit=True)      # frozen builder; its override is recorded there too
        rec["built"] = True
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: rec.get(k) for k in ("built", "refused")}), out)
    return 0 if rec["built"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
