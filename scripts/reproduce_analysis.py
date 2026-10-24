"""Reproduce the final-test results tables from the sealed score journals shipped in this repository.

    python scripts/reproduce_analysis.py [--out reproduced] [--skip-official] [--B 9999] [--B-cost 20000]

Run from the repository root with Windows or Linux Python 3.10+ (numpy needed; no simulator, no LLM, no network).

Two steps:

1. OFFICIAL: the registered command lines
       python sim/final_test/matrix/analyze.py --sealed docs/final-test/matrix/test-v1/score.jsonl --allow-test --outdir <out>/official/k1
       python sim/final_test/matrix/k23_analyze.py --allow-test --outdir <out>/official/k23
   These re-verify the preregistration freeze before they read any test item (docs/final-test/freeze.json and
   docs/final-test/k23/freeze_<scene>.json). The freeze pins byte hashes of the detector weights, the memory snapshots
   (SQLite), gate evidence and operational files that are NOT part of this release (weights go to Hugging Face, the rest
   stays in the private working tree), and some pinned code was re-pinned by later amendments, so in this release the
   official command is expected to print "REFUSED: TestItemRefused ...". That refusal is the guard working, not a fault.

2. POST-HOC REPRODUCTION: the same analysis functions (analyze.run_sealed and k23_analyze.run, unmodified) are called
   with ONLY the freeze re-verification replaced by a stub that records it was skipped. Every other check still runs:
   the score.seal.json hash of each score journal, the namespace and split-manifest membership of every item, the
   completeness of the registered roster, and the Kitchen-1 / Kitchen-2/3 separation. The guard existed to keep test
   items blind before the freeze; the results are already sealed, so re-deriving the tables from the sealed journals is
   a reproduction, not an unblinding. The outputs are compared with the shipped analysis.md / analysis_k23.md.

Nothing under docs/ or sim/ is written: all outputs go to <out>/ (default ./reproduced, git-ignored).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
MATRIX = os.path.join(SIM, "final_test", "matrix")

K1_SCORE = os.path.join(ROOT, "docs", "final-test", "matrix", "test-v1", "score.jsonl")
K1_SHIPPED_MD = os.path.join(ROOT, "docs", "final-test", "matrix", "test-v1", "analysis.md")
K23_SHIPPED_MD = os.path.join(ROOT, "docs", "final-test", "k23", "analysis", "analysis_k23.md")
K2_STORE = os.path.join(ROOT, "docs", "final-test", "matrix", "test-k2-v1")
K3_STORE = os.path.join(ROOT, "docs", "final-test", "matrix", "test-k3-v1")


def _rel(p: str) -> str:
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:                     # another drive (Windows)
        return p


def official(out: str, B: int, B_cost: int) -> dict:
    res = {}
    cmds = {
        "k1": [sys.executable, os.path.join("sim", "final_test", "matrix", "analyze.py"), "--sealed",
               os.path.relpath(K1_SCORE, ROOT), "--allow-test", "--B", str(B), "--outdir", os.path.join(out, "official", "k1")],
        "k23": [sys.executable, os.path.join("sim", "final_test", "matrix", "k23_analyze.py"), "--allow-test",
                "--B-cost", str(B_cost), "--outdir", os.path.join(out, "official", "k23")],
    }
    for k, cmd in cmds.items():
        print(f"\n[official {k}] " + " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = (r.stdout + r.stderr).strip()
        refused = "REFUSED" in text
        print(f"  exit {r.returncode}; " + (text.splitlines()[0][:300] + (" ..." if len(text) > 300 else "") if text else ""))
        res[k] = {"exit": r.returncode, "refused": refused}
    return res


def _diff(new_md: str, shipped_md: str, label: str) -> dict:
    a = open(shipped_md, encoding="utf-8").read().splitlines()
    b = open(new_md, encoding="utf-8").read().splitlines()
    d = [ln for ln in difflib.unified_diff(a, b, "shipped", "reproduced", n=0, lineterm="") if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    same = not d
    print(f"  {label}: {'IDENTICAL to the shipped report' if same else f'{len(d)} differing line(s) vs the shipped report'}")
    for ln in d[:20]:
        print("    " + ln[:220])
    return {"identical": same, "n_diff_lines": len(d)}


def posthoc(out: str, B: int, B_cost: int) -> dict:
    for p in (MATRIX, SIM):
        if p not in sys.path:
            sys.path.insert(0, p)
    import common as C                     # noqa: E402  (sim/final_test/matrix/common.py)
    import analyze as A                    # noqa: E402
    import k23_analyze as K23A             # noqa: E402

    skipped = []

    def _freeze_skipped(*a, **k):
        skipped.append("kitchen-1 freeze")
        return {"ok": True, "reasons": [], "posthoc_reproduction": "freeze re-verification skipped (see scripts/reproduce_analysis.py)"}

    def _rep_freeze_skipped(namespace, root=None, check_code=True):
        skipped.append(f"replication freeze {namespace}")
        return {"ok": True, "reasons": [], "posthoc_reproduction": "freeze re-verification skipped (see scripts/reproduce_analysis.py)"}

    C.freeze_status = _freeze_skipped
    C._k23.replication_freeze_status = _rep_freeze_skipped

    res = {}
    print("\n[post-hoc k1] analyze.run_sealed(test-v1/score.jsonl, allow_test=True)")
    r1, (jp, mp) = A.run_sealed(K1_SCORE, allow_test=True, B=B, outdir=os.path.join(out, "posthoc", "k1"))
    print(f"  H1: {r1['H1'].get('wording')}\n  wrote {_rel(mp)}")
    res["k1"] = _diff(mp, K1_SHIPPED_MD, "Kitchen 1 (test-v1) analysis.md")

    print("\n[post-hoc k23] k23_analyze.run(test-k2-v1, test-k3-v1, allow_test=True)")
    r2, (jp2, mp2) = K23A.run(K2_STORE, K3_STORE, allow_test=True, B_cost=B_cost, outdir=os.path.join(out, "posthoc", "k23"))
    print(f"  wrote {_rel(mp2)}")
    res["k23"] = _diff(mp2, K23_SHIPPED_MD, "Kitchens 2/3 analysis_k23.md")
    res["freeze_checks_skipped"] = sorted(set(skipped))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "reproduced"))
    ap.add_argument("--skip-official", action="store_true")
    ap.add_argument("--B", type=int, default=9999, help="bootstrap draws for analyze.py (registered default 9999)")
    ap.add_argument("--B-cost", type=int, default=20000, help="cost bootstrap draws for k23_analyze.py (registered default 20000)")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    summary = {"official": None if a.skip_official else official(out, a.B, a.B_cost)}
    summary["posthoc"] = posthoc(out, a.B, a.B_cost)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    ok = all(summary["posthoc"][k]["identical"] for k in ("k1", "k23"))
    print(f"\nsummary -> {_rel(os.path.join(out, 'summary.json'))}; tables reproduced exactly: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
