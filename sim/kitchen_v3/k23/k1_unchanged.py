"""K23-PREP gate: prove Kitchen 1 (FloorPlan1) is unchanged by the FloorPlan15 / FloorPlan17 capture plumbing.

Read-only. Checks, each PASS / FAIL:

1. manifest   the 47-file frozen Kitchen 1 treatment manifest (sim/final_test/diag/a31/repin.json "new.files"): every
              file's sha256 equals the pinned value.
2. seeds      kd.episode_plan / kd.amb_plan / kd.topup_train_plan / kd.close_plan for FloorPlan1 give byte-identical
              seeds (canonical sha of each plan) to the recorded baseline.
3. weights    sim/kitchen_v3/FloorPlan1/frozen_weights.sha256 and the weights it names are unchanged.
4. reports    the published Kitchen 1 reports under sim/kitchen_v3/FloorPlan1/ (probe, audit, state_calib, val_errors, ...)
              are byte-identical to the baseline.
5. data       the Kitchen 1 stores ~/thor-data/kitchen_v3/FloorPlan1 and ~/thor-data/kitchen_v3_amb/FloorPlan1 (and the
              shared state_cls/summary.json): size + mtime of the big files, sha256 of the small ones, the file count
              of every image / seg / depth split directory. Nothing is ever written there.
6. defaults   the Kitchen 1 capture scripts still resolve FloorPlan1 when called without a scene.

    python sim/kitchen_v3/k23/k1_unchanged.py --record    # once, BEFORE any K23 change -> baseline json
    python sim/kitchen_v3/k23/k1_unchanged.py             # any time after: exit 0 = all PASS, 1 = a FAIL

Runs on Windows (reads the WSL store through \\\\wsl.localhost) or inside WSL (reads ~/thor-data directly).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.dirname(SIM)
if SIM not in sys.path:
    sys.path.insert(0, SIM)

BASELINE = os.path.join(REPO, "sim", "final_test", "_checkpoints", "K23-PREP", "k1_baseline.json")
REPIN = os.path.join(REPO, "sim", "final_test", "diag", "a31", "repin.json")
#: K23 merge (docs/final-test/k23/merge-drafts/K1-UNCHANGED-SUCCESSOR.md): the recorded successor of the A31 manifest after the
#: authorized re-pin of six Kitchen 1 files; the A31 record itself stays the historical record and is never rewritten
SUCCESSOR = os.path.join(REPO, "sim", "final_test", "diag", "k23", "k1_manifest_successor.json")


def thor_data() -> str:
    if os.name == "nt":
        return "//wsl.localhost/Ubuntu-22.04/home/arash/thor-data"
    return os.path.join(os.path.expanduser("~"), "thor-data")


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_plans() -> dict:
    import kitchen_data as kd
    s = "FloorPlan1"
    return {"episode_plan": sha_obj(kd.episode_plan(s, kd.CAPTURE["episodes"])),
            "amb_plan": sha_obj(kd.amb_plan(s)),
            "topup_train_plan": sha_obj(kd.topup_train_plan(s, kd.TOPUP_TRAIN_MAX)),
            "close_plan": sha_obj(kd.close_plan(s, kd.CLOSE_EPISODES)),
            "capture_cfg": sha_obj({k: v for k, v in kd.CAPTURE.items()}),
            "capture_amb_cfg": sha_obj({k: v for k, v in kd.CAPTURE_AMB.items()})}


SMALL = 8 << 20          # files up to 8 MB are hashed; larger ones are size + mtime


def store_snapshot(d: str) -> dict:
    out = {}
    if not os.path.isdir(d):
        return {"_missing": True}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            st = os.stat(p)
            rec = {"size": st.st_size, "mtime": int(st.st_mtime)}
            if st.st_size <= SMALL:
                rec["sha256"] = sha_file(p)
            out[name] = rec
        elif os.path.isdir(p):
            if name in ("images", "depth", "seg", "pairs"):
                out[name + "/"] = {s: len(os.listdir(os.path.join(p, s))) for s in sorted(os.listdir(p))
                                   if os.path.isdir(os.path.join(p, s))}
            else:
                out[name + "/"] = {"n": len(os.listdir(p)), "mtime": int(os.stat(p).st_mtime)}
    return out


def reports_snapshot() -> dict:
    d = os.path.join(SIM, "kitchen_v3", "FloorPlan1")
    out = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            out[name] = sha_file(p)
    return out


def weights_snapshot() -> dict:
    out = {"frozen_weights.sha256": sha_file(os.path.join(SIM, "kitchen_v3", "FloorPlan1", "frozen_weights.sha256"))}
    wd = os.path.join(SIM, "weights", "v3")
    for name in sorted(os.listdir(wd)):
        if name.startswith("FloorPlan1."):
            out[name] = sha_file(os.path.join(wd, name))
    return out


def defaults_check() -> list:
    """The Kitchen 1 capture scripts, called with no scene, still mean FloorPlan1."""
    probs = []
    kv = os.path.join(SIM, "kitchen_v3")
    sh = open(os.path.join(kv, "amb_capture.sh"), encoding="utf-8").read()
    if "${1:-FloorPlan1}" not in sh and "--scenes FloorPlan1" not in sh:
        probs.append("amb_capture.sh: no FloorPlan1 default")
    ps = open(os.path.join(kv, "amb_capture.ps1"), encoding="utf-8").read()
    if '$Scene = "FloorPlan1"' not in ps and "FloorPlan1: ep " not in ps:
        probs.append("amb_capture.ps1: no FloorPlan1 default")
    st = open(os.path.join(kv, "status20_amb.sh"), encoding="utf-8").read()
    if "${1:-FloorPlan1}" not in st and "kitchen_v3_amb/FloorPlan1/capture_state.json" not in st:
        probs.append("status20_amb.sh: no FloorPlan1 default")
    ca = open(os.path.join(kv, "capture_all.ps1"), encoding="utf-8").read()
    if '@("FloorPlan1")' not in ca:
        probs.append("capture_all.ps1: default scenes changed")
    return probs


def snapshot(with_data: bool = True) -> dict:
    td = thor_data()
    snap = {"seeds": seed_plans(), "weights": weights_snapshot(), "reports": reports_snapshot()}
    if with_data:
        snap["data"] = {"kitchen_v3/FloorPlan1": store_snapshot(os.path.join(td, "kitchen_v3", "FloorPlan1")),
                        "kitchen_v3_amb/FloorPlan1": store_snapshot(os.path.join(td, "kitchen_v3_amb", "FloorPlan1")),
                        "kitchen_v3/state_cls": store_snapshot(os.path.join(td, "kitchen_v3", "state_cls"))}
    return snap


def manifest_path():
    """-> (path, problems): the manifest in force. The successor is selected only when it exists AND names the A31 record's
    current sha as the one it supersedes AND its self sha is valid; a present but unbound successor is a FAIL (never a silent
    fallback to the A31 record)."""
    if not os.path.exists(SUCCESSOR):
        return REPIN, []
    s = json.load(open(SUCCESSOR, encoding="utf-8"))
    probs = []
    if (s.get("supersedes") or {}).get("sha256") != sha_file(REPIN):
        probs.append("successor manifest: not bound to the A31 record on disk")
    body = {k: v for k, v in s.items() if k != "sha256"}
    if s.get("sha256") != hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        probs.append("successor manifest: self sha invalid (edited?)")
    return SUCCESSOR, probs


def manifest_check() -> list:
    path, bad = manifest_path()
    d = json.load(open(path, encoding="utf-8"))
    for f, h in d["new"]["files"].items():
        p = os.path.join(REPO, f)
        if not os.path.exists(p):
            bad.append(f"{f}: missing")
        elif sha_file(p) != h:
            bad.append(f"{f}: sha changed")
    return bad, len(d["new"]["files"])


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: new")
            elif k not in b:
                out.append(f"{path}/{k}: gone")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", action="store_true", help="write the baseline (only before any K23 change)")
    ap.add_argument("--no-data", action="store_true", help="skip the WSL store snapshot (no WSL access)")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default=None, help="also write the report json here")
    a = ap.parse_args(argv)
    if a.record:
        if os.path.exists(a.baseline):
            print(f"refusing: baseline exists ({a.baseline}); it is recorded once")
            return 2
        snap = snapshot(not a.no_data)
        os.makedirs(os.path.dirname(a.baseline), exist_ok=True)
        json.dump(snap, open(a.baseline, "w", encoding="utf-8"), indent=1, sort_keys=True)
        print(f"baseline written: {a.baseline} (sha {sha_file(a.baseline)[:16]})")
        return 0
    base = json.load(open(a.baseline, encoding="utf-8"))
    res = []
    bad, n = manifest_check()
    res.append(("manifest", not bad, f"{os.path.relpath(manifest_path()[0], REPO)}: {n} files; changed: {bad or 'none'}"))
    now = snapshot(not a.no_data and "data" in base)
    for k in ("seeds", "weights", "reports") + (("data",) if "data" in now else ()):
        d = _diff(base[k], now[k])
        res.append((k, not d, "byte-identical" if not d else "; ".join(d[:8])))
    probs = defaults_check()
    res.append(("defaults", not probs, "FloorPlan1 defaults kept" if not probs else "; ".join(probs)))
    ok = all(r[1] for r in res)
    for name, good, detail in res:
        print(f"  {'PASS' if good else 'FAIL'} {name:9s} {detail}")
    print(f"KITCHEN 1 UNCHANGED: {'PASS' if ok else 'FAIL'} ({sum(r[1] for r in res)}/{len(res)})")
    if a.out:
        json.dump({"ok": ok, "results": [{"check": r[0], "pass": r[1], "detail": r[2]} for r in res]},
                  open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
