"""Kitchen 3 G-iso: Arash's written waiver with a narrowed claim (decision record K23-GISO-WAIVER-FloorPlan17) and the ONLY condition
under which the K23 agent freeze accepts a non-pass G-iso. NEW module; no preregistration edit: A20.1 already provides the route ("an
INCONCLUSIVE gate is not passed (A20.1: W5 only with Arash's explicit written waiver, claims about memory isolation narrowed)").

The result (rehearsal-k3-v1, 2026-10-20 21:0x; archived sim/final_test/_checkpoints/K3-GISO-INCONCLUSIVE): status inconclusive, n 12,
diff [] (no absent-vs-present difference), blocked 3 (g-l1-place-bowl-counter-top en/fa, g-l2-microwave-run-egg en), control noise on one
blocked presentation, and "request probe differs from the Phase A none-arm bytes" -- the K23 execute path rendered the probe with Kitchen
1's type list (a tooling bug, fixed in k23_run.py for Kitchen 2). Arash, ~21:40 (AskUserQuestion): "Written waiver, narrowed claim
(Recommended)", under A20.1.

The freeze accepts the G-iso of a kitchen ONLY when: the kitchen is FloorPlan17 (rehearsal-k3-v1); the report is sealed (self-sha), of that
namespace, n = n_registered = 12; status is exactly "inconclusive" (a "fail" -- or anything else -- is never waivable); diff == [] and no
absent-vs-present difference in any component; the report's file sha = the archived copy's sha = the decision's g_iso_sha256; and the
decision record is Arash's, valid (self-sha), with this module's choice, rule and claim. The freeze summary then states the narrowed claim.

    python sim/final_test/matrix/k23_giso_waiver.py --check
    python sim/final_test/matrix/k23_giso_waiver.py --write --by Arash --date YYYY-MM-DD --said "<his words>"
Python 3.10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

SCENE = "FloorPlan17"
NS = "rehearsal-k3-v1"
DECISION_ID = f"K23-GISO-WAIVER-{SCENE}"
SCHEMA = "k23.decision/1"
CHOICE = "written-waiver-narrowed-claim"
RULE = ("A20.1: the inconclusive Kitchen 3 G-iso (rehearsal-k3-v1) is waived in writing by Arash with a narrowed claim; accepted by the "
        "agent freeze only for FloorPlan17, only for status 'inconclusive' with diff == [] and no absent-vs-present difference, only for the "
        "archived result named here; a 'fail' is never waivable")
CLAIM = ("Kitchen 3 memory isolation shown on 9/12 presentations; 3 unmeasured because of restore blocking; the Phase A request tie-in was "
         "broken by a tooling bug (fixed for Kitchen 2)")
ARCHIVE = "sim/final_test/_checkpoints/K3-GISO-INCONCLUSIVE/g_iso.json"
GISO_N = 12


def _p(rel, root=None):
    return os.path.join(C.paths(root)["root"], *rel.split("/"))


def report_path(root: str | None = None) -> str:
    return os.path.join(C.ns_docs(NS, root), "g_iso.json")


def decision_path(root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "decisions", f"{DECISION_ID}.json")


def _sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def decision_sha(root: str | None = None):
    return _sha(decision_path(root))


def decision_problems(root: str | None = None) -> list:
    p = decision_path(root)
    if not os.path.exists(p):
        return [f"no decision record ({os.path.relpath(p, C.paths(root)['root'])})"]
    try:
        d = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"decision record unreadable ({type(e).__name__})"]
    out = []
    if d.get("schema") != SCHEMA or d.get("id") != DECISION_ID or d.get("sha256") != C.self_sha(d):
        out.append("decision record schema / id / self-sha mismatch")
    if d.get("choice") != CHOICE or d.get("rule") != RULE or d.get("claim") != CLAIM or d.get("scene") != SCENE or d.get("namespace") != NS:
        out.append("decision record choice / rule / claim / scene / namespace are not this module's")
    if d.get("by") != "Arash" or not d.get("date") or not str(d.get("said") or "").strip():
        out.append("decision record is not Arash's attributed record (by / date / said)")
    if d.get("archive") != ARCHIVE or not d.get("g_iso_sha256"):
        out.append("decision record does not name the archived G-iso result")
    return out


def report_problems(rep_: dict, file_sha: str | None, root: str | None = None) -> list:
    """The G-iso report qualifies for THIS waiver (beyond the decision record itself)."""
    why = []
    if rep_.get("sha256") != C.self_sha(rep_) or rep_.get("namespace") != NS:
        why.append("G-iso report not sealed / not rehearsal-k3-v1")
    if rep_.get("status") != "inconclusive":
        why.append(f"G-iso status {rep_.get('status')!r}: only 'inconclusive' is waivable (a 'fail' never)")
    if int(rep_.get("n") or 0) != GISO_N or int(rep_.get("n_registered") or 0) != GISO_N:
        why.append(f"G-iso n {rep_.get('n')} / registered {rep_.get('n_registered')} (must be {GISO_N})")
    comps = rep_.get("components") if isinstance(rep_.get("components"), dict) else {}
    if rep_.get("diff") != [] or rep_.get("diff_components") or any(int((v or {}).get("absent_vs_present") or 0) for k, v in comps.items()
                                                                     if k in ("request", "audit")):
        why.append("G-iso shows an absent-vs-present difference (diff not empty): never waivable")
    arch = _sha(_p(ARCHIVE, root))
    try:
        d = json.load(open(decision_path(root), encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    if not file_sha or file_sha != arch or file_sha != d.get("g_iso_sha256"):
        why.append(f"the G-iso report ({str(file_sha)[:12]}) is not the archived result ({str(arch)[:12]}) the waiver names "
                   f"({str(d.get('g_iso_sha256'))[:12]})")
    return why


def waiver_problems(root: str | None = None) -> list:
    """[] when the current rehearsal-k3-v1 G-iso report is waived by a valid record of Arash's (every condition above)."""
    p = report_path(root)
    try:
        rep_ = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"G-iso report unreadable ({type(e).__name__})"]
    return decision_problems(root) + report_problems(rep_, _sha(p), root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K23 Kitchen 3 G-iso waiver (decision record)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--by"), ap.add_argument("--date"), ap.add_argument("--said")
    a = ap.parse_args(argv)
    if a.check:
        why = waiver_problems()
        print(json.dumps({"decision": DECISION_ID, "waiver_holds": not why, "problems": why, "claim": CLAIM}, indent=1, ensure_ascii=False))
        return 0 if not why else 1
    if a.by != "Arash" or not a.date or not (a.said or "").strip():
        print("REFUSED: --write needs --by Arash --date YYYY-MM-DD --said \"<his words>\"", file=sys.stderr)
        return 2
    out = decision_path()
    if os.path.exists(out):
        print(f"REFUSED: {out} exists (a decision is recorded once)", file=sys.stderr)
        return 2
    rp = report_path()
    try:
        rep_ = json.load(open(rp, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"REFUSED: G-iso report unreadable ({type(e).__name__})", file=sys.stderr)
        return 2
    fsha, arch = _sha(rp), _sha(_p(ARCHIVE))
    if fsha != arch:
        print(f"REFUSED: the current G-iso report ({str(fsha)[:12]}) is not the archived one ({str(arch)[:12]})", file=sys.stderr)
        return 2
    d = {"schema": SCHEMA, "id": DECISION_ID, "choice": CHOICE, "rule": RULE, "claim": CLAIM, "scene": SCENE, "namespace": NS,
         "archive": ARCHIVE, "g_iso_sha256": fsha, "g_iso_self_sha": rep_.get("sha256"), "status": rep_.get("status"), "n": rep_.get("n"),
         "blocked": rep_.get("blocked"), "control_noise": rep_.get("control_noise"), "diff": rep_.get("diff"),
         "inconclusive_reasons": rep_.get("inconclusive_reasons"), "by": a.by, "date": a.date, "said": a.said.strip(),
         "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "basis": "prereg A20.1 (W5 only with Arash's explicit written waiver; claims narrowed)"}
    if rep_.get("status") != "inconclusive" or rep_.get("diff") != []:
        print("REFUSED: only an 'inconclusive' G-iso with diff == [] can be waived", file=sys.stderr)
        return 2
    d["sha256"] = C.self_sha(d)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", os.path.relpath(out, C.paths(None)["root"]), "sha", decision_sha()[:16])
    why = waiver_problems()
    print("waiver holds" if not why else "WARNING: " + "; ".join(why))
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
