"""Verify a built K23 YOLO dataset before training (Astra K2-XSPLIT-r1, K2-CLEAN-r1 item 5): every image of every list is
hashed (sha256 of the resolved file); FAIL on any exact duplicate between the TRAIN list and any held-out list (val, test,
val_close, test_close, and any extra list), on any held-out frame of heldout_exclusions.json still listed, and on any
quarantined frame still listed. Reuse of the SAME frame inside two held-out views is reported (not leakage).

    python sim/kitchen_v3/k23/k23_verify_lists.py --scene FloorPlan15 [--dataset yolo_primary] [--store main|amb]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_LISTS = ("train", "plus_train")


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("FloorPlan15", "FloorPlan17"))
    ap.add_argument("--dataset", default="yolo_primary")
    ap.add_argument("--store", default="main", choices=("main", "amb"))
    a = ap.parse_args(argv)
    root = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3_amb" if a.store == "amb" else "kitchen_v3", a.scene)
    dd = os.path.join(root, a.dataset)
    lists = {os.path.splitext(f)[0]: [l.strip() for l in open(os.path.join(dd, f), encoding="utf-8") if l.strip()]
             for f in sorted(os.listdir(dd)) if f.endswith(".txt")}
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    hashes = {}
    for name, paths in lists.items():
        for p in paths:
            hashes.setdefault(sha_file(os.path.realpath(p)), []).append((name, stem(p)))
    is_tr = lambda n: n in TRAIN_LISTS                 # yolo_amb49: plus_train = primary train + amb train (the control's TRAIN)
    leak = [v for v in hashes.values() if any(is_tr(n) for n, _ in v) and any(not is_tr(n) for n, _ in v)]
    reuse = [v for v in hashes.values() if len(v) > 1 and v not in leak]
    exp = os.path.join(HERE, a.scene, "heldout_exclusions.json")
    excl = {e["frame"] for e in json.load(open(exp, encoding="utf-8"))["entries"]} if os.path.exists(exp) else set()
    listed = {stem(p) for n, ps in lists.items() if not is_tr(n) for p in ps}
    excl_listed = sorted(excl & listed)
    # quarantine of BOTH stores (Astra K23-AMB-TRAIN-r1 item 2: pooled lists carry main-store frames)
    quar = {}
    for st in ("kitchen_v3", "kitchen_v3_amb"):
        fj = os.path.join(os.path.expanduser("~"), "thor-data", st, a.scene, "frames.jsonl")
        if not os.path.exists(fj):
            continue
        for line in open(fj, encoding="utf-8"):
            r = json.loads(line)
            if "img" in r and r.get("quarantined"):
                quar[r["img"]] = r["quarantined"]
    all_listed = {stem(p) for ps in lists.values() for p in ps}
    quar_listed = sorted(set(quar) & all_listed)
    # the sealed amb holdout (metadata only): none of its frames in any list
    hm = os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3_amb", a.scene, "holdout_manifest.json")
    sealed = set(json.load(open(hm, encoding="utf-8"))["images"]) if os.path.exists(hm) else set()
    holdout_listed = sorted(sealed & all_listed)
    ok = not leak and not excl_listed and not quar_listed and not holdout_listed
    rep = {"what": f"K23 list verification {a.scene} {a.store}/{a.dataset}", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "ok": ok, "lists": {n: len(v) for n, v in lists.items()}, "n_unique_images": len(hashes),
           "train_heldout_exact_duplicates": leak[:50], "n_train_heldout_exact_duplicates": len(leak),
           "heldout_exclusions_still_listed": excl_listed, "quarantined_still_listed": quar_listed[:50],
           "sealed_holdout_n": len(sealed), "sealed_holdout_listed": holdout_listed[:50],
           "n_quarantined_in_store": len(quar), "heldout_view_reuse": len(reuse),
           "code_sha256": sha_file(os.path.abspath(__file__))}
    out = os.path.join(HERE, a.scene, "clean", f"verify_{a.store}_{a.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("ok", "lists", "n_train_heldout_exact_duplicates", "heldout_exclusions_still_listed",
                                          "quarantined_still_listed", "sealed_holdout_listed", "heldout_view_reuse")}))
    print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
