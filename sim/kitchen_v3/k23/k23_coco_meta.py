"""Successor metadata for the FloorPlan15 perception freeze after `k23_freeze.py --scene FloorPlan15 --add-coco` (A38 item 4).

k23_freeze.py --add-coco (unchanged, it is bound by the K2 perception freeze record) appends the COCO-start sensitivity checkpoint line to
sim/kitchen_v3/FloorPlan15/frozen_weights.sha256 and sets coco_pending / test_evaluation_blocked to false, but leaves the metadata's
record_sha256 and n_files describing the record BEFORE the append (Astra noted this, 2026-10-20). This NEW tool verifies the append and
writes the successor metadata:

  checks   the record minus its LAST line hashes to the metadata's record_sha256 (so the append changed nothing else); the last line names
           sim/weights/v3/FloorPlan15.yolo26s_coco.pt with the file's current sha256; the metadata says coco_pending = false and
           test_evaluation_blocked = false; n_files + 1 = the record's line count.
  writes   the previous metadata is copied to sim/kitchen_v3/FloorPlan15/supersession/<ts>-coco/frozen_weights.meta.json; the metadata gets
           record_sha256 / n_files of the record as it now stands plus a "coco_append" entry (previous record sha and count, the COCO
           sha, the time, this tool). Nothing else changes; the record itself is never edited.

    python sim/kitchen_v3/k23/k23_coco_meta.py              (check only; exit 0 = the append is clean and the metadata is already current)
    python sim/kitchen_v3/k23/k23_coco_meta.py --write      (after --add-coco: verify, archive, write the successor metadata)
Runs anywhere the record's repo files are readable (the COCO checkpoint is a repo file). Python 3.10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCENE = "FloorPlan15"
COCO_REL = f"sim/weights/v3/{SCENE}.yolo26s_coco.pt"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def paths(root: str | None = None) -> dict:
    d = os.path.join(root or ROOT, "sim", "kitchen_v3", SCENE)
    return {"dir": d, "record": os.path.join(d, "frozen_weights.sha256"), "meta": os.path.join(d, "frozen_weights.meta.json"),
            "root": root or ROOT}


def inspect(root: str | None = None) -> dict:
    """-> {state: 'current' | 'needs_successor' | 'invalid', problems, ...}. 'current' = the metadata already describes the record."""
    P = paths(root)
    out = {"problems": []}
    try:
        raw = open(P["record"], "rb").read()
        meta = json.load(open(P["meta"], encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"state": "invalid", "problems": [f"record / metadata unreadable ({type(e).__name__})"]}
    lines = [ln for ln in raw.split(b"\n") if ln.strip()]
    cur_sha, n = sha_bytes(raw), len(lines)
    body = raw[:-1] if raw.endswith(b"\n") else raw
    cut = body.rfind(b"\n") + 1                                  # the raw bytes before the LAST line (byte-exact, blank lines included)
    prefix, last = raw[:cut], raw[cut:]
    out.update(record_sha256=cur_sha, n_lines=n, meta_record_sha256=meta.get("record_sha256"), meta_n_files=meta.get("n_files"))
    if meta.get("coco_pending") is not False or meta.get("test_evaluation_blocked") is not False:
        out["problems"].append("the metadata still says coco_pending / test_evaluation_blocked: run k23_freeze.py --scene FloorPlan15 --add-coco first")
    parts = last.decode("utf-8", "replace").split("  ")
    if not raw.endswith(b"\n") or len(parts) != 2 or parts[1] != COCO_REL + "\n" or len(parts[0]) != 64:
        out["problems"].append(f"the record's last line is not exactly '<sha256>  {COCO_REL}'")
    else:
        want = parts[0]
        cp = os.path.join(P["root"], *COCO_REL.split("/"))
        if not os.path.exists(cp) or sha_file(cp) != want:
            out["problems"].append("the COCO checkpoint is missing or differs from its record line")
        out["coco_sha256"] = want
    if out["problems"]:
        out["state"] = "invalid"
        return out
    if meta.get("record_sha256") == cur_sha and meta.get("n_files") == n:
        out["state"] = "current"
        return out
    if sha_bytes(prefix) != meta.get("record_sha256") or meta.get("n_files") != n - 1:
        out["problems"].append("the record minus its last line is not the record the metadata describes (something besides the COCO "
                               "line changed, or the metadata is not the pre-append one)")
        out["state"] = "invalid"
        return out
    out["state"] = "needs_successor"
    return out


def write(root: str | None = None, now: str | None = None) -> dict:
    P = paths(root)
    st = inspect(root)
    if st["state"] != "needs_successor":
        raise RuntimeError(f"not written: state {st['state']} {st['problems']}")
    meta = json.load(open(P["meta"], encoding="utf-8"))
    ts = now or time.strftime("%Y%m%dT%H%M%S")
    arch = os.path.join(P["dir"], "supersession", f"{ts}-coco")
    os.makedirs(arch, exist_ok=True)
    shutil.copy2(P["meta"], os.path.join(arch, "frozen_weights.meta.json"))
    meta["coco_append"] = {"previous_record_sha256": meta.get("record_sha256"), "previous_n_files": meta.get("n_files"),
                           "coco": COCO_REL, "coco_sha256": st["coco_sha256"], "ts": ts, "tool": "sim/kitchen_v3/k23/k23_coco_meta.py",
                           "archive": os.path.relpath(arch, P["root"]).replace(os.sep, "/")}
    meta["record_sha256"], meta["n_files"] = st["record_sha256"], st["n_lines"]
    tmp = P["meta"] + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, P["meta"])
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FloorPlan15 perception freeze: successor metadata after --add-coco")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    st = inspect()
    print(json.dumps(st, indent=1))
    if not a.write:
        return 0 if st["state"] == "current" else 1
    try:
        m = write()
    except RuntimeError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"written": True, "record_sha256": m["record_sha256"], "n_files": m["n_files"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
