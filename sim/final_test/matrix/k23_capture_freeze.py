"""K23 capture: the frozen-artefact reference of capture.py for Kitchens 2 / 3 = the historical Kitchen 1 freeze AS SUPERSEDED BY the
authorized K23 re-pin. NEW module; capture.py, common.py and docs/final-test/freeze.json are not changed (freeze.json is the signed
historical record: docs/final-test/k23/repin.json lists it under historical_records_not_rewritten).

Why: capture.py (unchanged, Kitchen 1) loads docs/final-test/freeze.json and, for a validated freeze, aborts when a frozen artefact's
bytes differ from the freeze (common.freeze_artifact_problems). The historical freeze (sha 6ab8ee5f..., A37 re-freeze signed by Arash)
records exec_contract.json and ft_v2.json as they were BEFORE the K23 re-pin; the re-pin (Arash 2026-10-18 ~20:50 "Yes, authorize it
(Recommended)", docs/final-test/k23/repin.json; merged by route B, prereg A39.5) regenerated exactly those two files (contract_change:
only derived_from.honest_skills / exec_contract_sha256 + self shas; texts, tools and policies unchanged). Every K23 capture
(rehearsal-kN-v1, test-kN-v1) therefore aborts with "artefact exec_contract.json / ft_v2.json changed after the freeze" (rehearsal-k3-v1,
2026-10-20 20:28). The SC2 diagnostic met the same abort and used a successor freeze that replaced ONLY these values, each checked
against the approved before -> after shas (docs/final-test/k23/merge-drafts/sc2/make_sc2_freeze.py ARTEFACTS; Astra K23-SC2-r1).

Rule (fail closed, all or nothing): when -- and only when -- freeze.json is byte-for-byte the historical freeze 6ab8ee5f..., the re-pin
record is Arash's authorization and lists both files with exactly these before / after shas, and each historical artefact entry names
the expected path with exactly the "before" sha, the capture receives an in-memory view of the freeze in which those two artefact shas
are the re-pinned "after" shas. Nothing else differs (every other artefact, dataset, params, tolerances, weights, map, snapshot and key
is checked by capture exactly as before; a third changed artefact, a reverted file, or another freeze still aborts). The view is never
written to disk and carries no signature; the K23 test entry authority stays the kitchen's agent freeze (k23_ns.replication_freeze_status).
Every K23 capture records the view it used in the namespace's events journal (event k23_capture_freeze_view, before capture_start); the
agent freeze checks that every rehearsal capture_start was preceded by an applied view (journal_problems).

    python sim/final_test/matrix/k23_capture_freeze.py --check --namespace rehearsal-k3-v1 [--root <repo>]
        (offline, live-faithful: the capture's own artefact step -- snapshot_freeze_problems(need) + freeze_artifact_problems -- on the
         historical freeze (the abort) and on the view (what the capture now sees); nothing is written)
Python 3.10.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

HIST_FREEZE_SHA = "6ab8ee5fbef6c1184d490afae416d7395a1b69540d69f9db562559c7c793c959"     # docs/final-test/freeze.json (A37 re-freeze, signed)
REPIN = "docs/final-test/k23/repin.json"
#: label in freeze.json "artefacts" -> (path, historical sha = re-pin "before", approved sha = re-pin "after")   (= make_sc2_freeze.ARTEFACTS)
ARTEFACTS = {"exec_contract.json": ("sim/planners/exec_contract.json",
                                    "93047cc3252a02b6921ae426a9c5e8a01e3532ac28a6208f724a918675d5b9c6",
                                    "f6900b3e66bc2db0fac63d1c505d119ccfe01162369b18bb9a58d8795c7fd1e6"),
             "ft_v2.json": ("sim/planners/ft_v2.json",
                            "d80e2066c20061ee7f13f64bfcfa22c7a1ea52c24a2bed42b7609456ed919795",
                            "70bb32eb3327be3526a9a5d65a60c7edd6a54468e050ae9d2038e2d15f0d17e3")}
EVENT = "k23_capture_freeze_view"
RULE = ("K23 capture: Kitchen 1 freeze 6ab8ee5f as superseded by the authorized K23 re-pin (docs/final-test/k23/repin.json) for "
        "exec_contract.json and ft_v2.json only; in memory, unsigned, all-or-nothing")


def _sha_file(p: str):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _root(root):
    return C.paths(root)["root"]


def repin_problems(root: str | None = None) -> list:
    """The re-pin record must be Arash's authorization and list both artefacts with exactly the before / after shas of ARTEFACTS."""
    p = os.path.join(_root(root), *REPIN.split("/"))
    try:
        rp = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{REPIN} unreadable ({type(e).__name__})"]
    why = []
    if rp.get("schema") != "k23.repin/1" or (rp.get("authorization") or {}).get("by") != "Arash":
        why.append(f"{REPIN}: not the k23.repin/1 record of Arash's authorization")
    if not any("freeze.json" in str(x) for x in rp.get("historical_records_not_rewritten") or []):
        why.append(f"{REPIN}: freeze.json is not listed as a historical record that is not rewritten")
    before, after = rp.get("identity_file_shas_before") or {}, rp.get("identity_file_shas_after") or {}
    for label, (path, b, a) in ARTEFACTS.items():
        if path not in (rp.get("identity_files_changed") or []):
            why.append(f"{REPIN}: {path} is not a re-pinned file")
        if before.get(path) != b or after.get(path) != a:
            why.append(f"{REPIN}: {path} before / after shas differ from {b[:12]} -> {a[:12]}")
    return why


def view(fz: dict, root: str | None = None) -> tuple:
    """(freeze-for-capture, info). The re-pin view when every condition holds; otherwise the freeze unchanged (capture stays strict)."""
    r = _root(root)
    info = {"historical_freeze_sha256": _sha_file(C.paths(root)["freeze"]), "repin_sha256": _sha_file(os.path.join(r, *REPIN.split("/"))),
            "applied": False, "why": [], "rule": RULE}
    if not fz:
        info["why"].append("no freeze.json")
        return fz, info
    if info["historical_freeze_sha256"] != HIST_FREEZE_SHA:
        info["why"].append(f"freeze.json is not the historical Kitchen 1 freeze {HIST_FREEZE_SHA[:16]}: no substitution")
        return fz, info
    rp = repin_problems(root)
    if rp:
        info["why"] += rp
        return fz, info
    v = copy.deepcopy(fz)
    subs = {}
    for label, (path, b, a) in ARTEFACTS.items():
        ent = (v.get("artefacts") or {}).get(label)
        if not isinstance(ent, dict) or ent.get("path") != path or ent.get("sha256") != b:
            info["why"].append(f"{label}: the historical entry is not {path} @ {b[:12]}: no substitution")
            return fz, info
        ent["sha256"] = a
        subs[label] = {"path": path, "historical": b, "repinned": a}
    info.update(applied=True, substitutions=subs)
    return v, info


def journal_problems(ns: str, root: str | None = None) -> list:
    """For the agent freeze: every capture_start of `ns` was preceded (since the previous capture_start) by an applied view event with the
    historical freeze sha and exactly the ARTEFACTS substitutions."""
    p = C.journal_path(ns, "events", root)
    rows = C.Journal(p).rows() if os.path.exists(p) else []
    want = {lb: {"path": pa, "historical": b, "repinned": a} for lb, (pa, b, a) in ARTEFACTS.items()}
    ok_since, why, n = False, [], 0
    for e in rows:
        if e.get("event") == EVENT:
            ok_since = bool(e.get("applied") and e.get("historical_freeze_sha256") == HIST_FREEZE_SHA and e.get("substitutions") == want)
        elif e.get("event") == "capture_start":
            n += 1
            if not ok_since:
                why.append(f"{ns}: capture_start #{n} without a preceding applied {EVENT} event (re-pin view of the Kitchen 1 freeze)")
            ok_since = False
    return why


_STATE = {"installed": False, "info": None}


def install(root: str | None = None):
    """Patch capture.load_freeze in THIS process (k23_run's capture path only; K23 namespaces only). Returns the view info (applied or not);
    an unapplied view leaves capture exactly as strict as before."""
    import capture as CP
    if not _STATE["installed"]:
        orig = CP.load_freeze

        def load_freeze(root_=None):
            v, info = view(orig(root_), root_)
            _STATE["info"] = info
            return v
        load_freeze.k23_capture_freeze = True
        load_freeze.k23_orig = orig
        CP.load_freeze = load_freeze
        _STATE["installed"] = True
    _, info = view(CP.load_freeze.k23_orig(root), root)
    _STATE["info"] = info
    return info


def record_event(ns: str, info: dict, root: str | None = None) -> None:
    """The view used, in the namespace's events journal, before the capture runs (the audit trail the agent freeze checks)."""
    os.makedirs(C.ns_docs(ns, root), exist_ok=True)
    C.Journal(C.journal_path(ns, "events", root)).append(C.make_record(
        "events", event=EVENT, phase="capture", namespace=ns, applied=bool(info.get("applied")),
        historical_freeze_sha256=info.get("historical_freeze_sha256"), repin_sha256=info.get("repin_sha256"),
        substitutions=info.get("substitutions"), why=info.get("why") or [], rule=RULE))


def capture_need(ns: str, root: str | None = None) -> list:
    """The sources the live capture of `ns` checks (capture.main's own construction, as its --dry-run builds it)."""
    import k23_entry as E
    import k23_run as R
    import treatment as T
    CP = R.admit_capture(ns)
    manifest, items = C.load_manifest(root), CP.load_items(root)
    ft2 = T.is_ft2(ns)
    cap = CP.Capturer(namespace=ns, executor=None, contract=None, counter=None, open_sources=None, items=items, manifest=manifest,
                      fs_doc=None, channels=(), allow_test=E.kind(ns) == "test", root=root, worlds=("canon",), runs=(1,),
                      reloc_seed=None, reloc_all=False, only_sources=None, use_curve=False, use_fs=False, freeze={})
    assert ft2 == cap.ft2
    return cap.needed_sources()


def check(ns: str, root: str | None = None) -> dict:
    """The capture's artefact step (capture.py: freeze_status None -> snapshot_freeze_problems(need) + freeze_artifact_problems) on the
    historical freeze and on the view. Read-only."""
    import k23_entry as E
    if E.kind(ns) not in ("rehearsal", "test"):
        raise ValueError(f"{ns!r} is not a K23 rehearsal / test namespace")
    import capture as CP
    orig = getattr(CP.load_freeze, "k23_orig", CP.load_freeze)
    fz = orig(root)
    try:
        need, need_note = capture_need(ns, root), None
    except RuntimeError as e:                                   # a test roster resolves only after the A29 item-6 size decision
        need, need_note = [], f"capture sources unresolved ({e}); artefact step checked without snapshot entries"
    branch = "checked" if (fz and fz.get("freeze_status") is None) else ("not_checked" if fz else "no_freeze")
    hist = C.snapshot_freeze_problems(need, root, fz) + C.freeze_artifact_problems(fz, root) if branch == "checked" else []
    v, info = view(fz, root)
    new = C.snapshot_freeze_problems(need, root, v) + C.freeze_artifact_problems(v, root) if branch == "checked" else []
    return {"namespace": ns, "need": need, "need_note": need_note, "branch": branch, "historical_problems": hist, "view_problems": new, "view": info,
            "capture_aborts_before": bool(hist), "capture_aborts_with_view": bool(new),
            "abort_message_before": ("frozen artefact(s) changed: " + "; ".join(hist[:6])) if hist else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K23 capture: Kitchen 1 freeze as superseded by the authorized re-pin (offline check)")
    ap.add_argument("--check", action="store_true", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    r = check(a.namespace, a.root)
    print(json.dumps(r, indent=1, ensure_ascii=False))
    return 0 if (r["view"]["applied"] and not r["capture_aborts_with_view"]) else 1


if __name__ == "__main__":
    sys.exit(main())
