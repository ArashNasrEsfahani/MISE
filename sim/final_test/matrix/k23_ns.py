"""K23 (Kitchens 2 / 3) namespaces and the A29 roster, kept out of common.py (its size gate) and merged into C.NAMESPACES at import
(Astra K23-NAMESPACES-r1 items 1-2; ledger 2026-10-18 "K2/K3 namespace design"). Kitchen 1 entries are never touched.

Per replication kitchen: practice (Gemma only, 0 USD), rehearsal (the frozen pipeline on dev items; no outcome-driven tuning), test
(the A29 roster). Ceilings = the operational split of Arash's 70 USD Opus approval (approve-all-6 item 4): 35 per kitchen.
"""
from __future__ import annotations

import json
import os

REPLICATION_SCENES = ("FloorPlan15", "FloorPlan17")
REPLICATION_USD = 70.0                    # Arash approve-all-6 item 4: Opus K2 + K3 together (incl. rehearsal)
A29_ROSTER = "a29_roster"


def _ns(items, core, total, scene, **kw):
    return dict({"items": items, "ceilings": {"core": core, "total": total}, "archived_arms": (), "treatment": "ft-v2", "scene": scene}, **kw)


K23_NAMESPACES = {
    "practice-k2-v1": _ns("practice_pool", 0.0, 0.0, "FloorPlan15"),
    "rehearsal-k2-v1": _ns("dev_pilot", 3.0, 3.0, "FloorPlan15"),
    "test-k2-v1": _ns(A29_ROSTER, 32.0, 32.0, "FloorPlan15", replication_test=True),
    "practice-k3-v1": _ns("practice_pool", 0.0, 0.0, "FloorPlan17"),
    "rehearsal-k3-v1": _ns("dev_pilot", 3.0, 3.0, "FloorPlan17"),
    "test-k3-v1": _ns(A29_ROSTER, 32.0, 32.0, "FloorPlan17", replication_test=True),
}


def _C():
    import common
    return common


def is_replication(namespace: str) -> bool:
    """A Kitchen 2 / 3 namespace (its scene is a replication kitchen)."""
    return _C().ns_scene(namespace) in REPLICATION_SCENES


def a29_files(scene: str, root: str | None = None) -> tuple:
    d = os.path.join(_C().paths(root)["docs"], "k23")
    return os.path.join(d, f"a29_sample_{scene}.json"), os.path.join(d, "a29_size_decision.json")


def a29_roster(namespace: str, root: str | None = None) -> list:
    """The test items of a replication test namespace = the candidate of the kitchen's FROZEN A29 sample manifest named by the recorded
    A29 item-6 size decision. Validated: kitchen, manifest self-hash, the decision's manifest hash, split + inclusion hashes, unique
    ids, nested candidates. Missing / mismatching -> RuntimeError (never a default, never 'all')."""
    C = _C()
    scene = C.ns_scene(namespace)
    mp, dp = a29_files(scene, root)
    if not os.path.exists(mp):
        raise RuntimeError(f"no frozen A29 sample for {scene} ({mp})")
    with open(mp, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("kitchen") != scene or C.self_sha(m) != m.get("sha256"):
        raise RuntimeError(f"A29 sample {mp}: wrong kitchen or self-hash mismatch")
    man = C.load_manifest(root)
    if m.get("split_manifest_sha256") != man.get("sha256") or m.get("inclusion") != (man.get("inclusion") or {}).get("sha256"):
        raise RuntimeError("A29 sample: written for another split / inclusion manifest")
    order = list(m.get("order") or [])
    if len(set(order)) != len(order) or set(order) - set(man["test"]):
        raise RuntimeError("A29 sample: duplicate or non-test ids")
    cands = m.get("candidates") or {}
    for name, ids in cands.items():
        if list(ids) != order[:len(ids)]:
            raise RuntimeError(f"A29 sample: candidate {name} is not a prefix of the order")
    if not os.path.exists(dp):
        raise RuntimeError(f"no recorded A29 item-6 size decision ({dp})")
    with open(dp, encoding="utf-8") as f:
        dec = (json.load(f).get("kitchens") or {}).get(scene) or {}
    if dec.get("manifest_sha256") != m["sha256"] or dec.get("candidate") not in cands:
        raise RuntimeError(f"A29 size decision for {scene}: missing, or not for this manifest / candidate")
    return list(cands[dec["candidate"]])


FREEZE_SCHEMA = "final-test.freeze-k23/1"


def freeze_path(scene: str, root: str | None = None) -> str:
    return os.path.join(_C().paths(root)["docs"], "k23", f"freeze_{scene}.json")


def perception_problems(scene: str, root: str | None = None) -> list:
    """The kitchen's perception freeze record (sim/kitchen_v3/<scene>/frozen_weights.sha256, k23_freeze.py): every listed file
    (repo-relative or absolute) must still have its hash; a FloorPlan15 record must not be COCO-pending for TEST evaluation is NOT
    required here (the agent test uses the deployed models only)."""
    C = _C()
    base = C.paths(root)["root"]
    p = os.path.join(base, "sim", "kitchen_v3", scene, "frozen_weights.sha256")
    if not os.path.exists(p):
        return [f"{scene}: perception freeze record missing ({p})"]
    bad = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        sha, rel = line.split(None, 1)
        rel = rel.strip()
        fp = rel if os.path.isabs(rel) else os.path.join(base, rel)
        try:
            if C.sha256_file(fp) != sha:
                bad.append(rel)
        except OSError:
            bad.append(rel)
    return [f"{scene}: frozen perception changed / missing: {bad[:5]}"] if bad else []


def replication_freeze_status(namespace: str, root: str | None = None, check_code: bool = True) -> dict:
    """K23 build plan step 4 (Astra K23-NAMESPACES-r1 items 4-5): a replication test namespace runs TEST items only under ITS kitchen's
    freeze docs/final-test/k23/freeze_<scene>.json: schema, self-sha, scene + namespace, every gate passed, the sign-off (Arash's
    check, or his standing pre-signature of approve-all-6 item 1 together with an Astra review file and a written freeze summary),
    split + preregistration shas, unchanged code, the kitchen's verified perception freeze, the pinned A29 roster, unchanged
    snapshots. Kitchen 1's freeze.json is never consulted here (and never satisfies this)."""
    C = _C()
    P, reasons = C.paths(root), []
    scene = C.ns_scene(namespace)
    if not is_replication(namespace):
        return {"ok": False, "reasons": [f"{namespace} is not a replication namespace"]}
    p = freeze_path(scene, root)
    try:
        with open(p, encoding="utf-8") as f:
            fz = json.load(f)
    except (OSError, ValueError) as e:
        return {"ok": False, "reasons": [f"{os.path.basename(p)} missing or unreadable: {type(e).__name__}"]}
    if fz.get("schema") != FREEZE_SCHEMA:
        reasons.append("freeze schema")
    if fz.get("sha256") != C.self_sha(fz):
        reasons.append("freeze self-sha mismatch (edited after writing)")
    if fz.get("scene") != scene or namespace not in (fz.get("namespaces") or ()):
        reasons.append(f"freeze is not for {scene} / {namespace}")
    gates = fz.get("gates") or {}
    failed = sorted(g for g, v in gates.items() if not (isinstance(v, dict) and v.get("pass") is True))
    if not gates or failed:
        reasons.append(f"gates not all passed: {failed or 'none recorded'}")
    chk, pre = fz.get("arash_check") or {}, fz.get("standing_presignature") or {}
    signed = chk.get("by") == "Arash" and chk.get("date")
    presigned = (pre.get("basis") == "approve-all-6 item 1" and pre.get("astra_review") and pre.get("freeze_summary")
                 and all(os.path.exists(os.path.join(P["root"], pre[k])) for k in ("astra_review", "freeze_summary")))
    if not (signed or presigned):
        reasons.append("no sign-off (Arash's check, or the approve-all-6 item 1 pre-signature with Astra review + freeze summary)")
    if fz.get("split_sha256") != C.SPLIT_SHA256:
        reasons.append("split sha differs")
    try:
        if fz.get("prereg_sha256") != C.sha256_file(P["prereg"]):
            reasons.append("preregistration changed after the freeze")
    except OSError:
        reasons.append("preregistration unreadable")
    if check_code:
        for rel, sha in sorted((fz.get("code") or {}).items()):
            try:
                if C.sha256_file(os.path.join(P["root"], rel)) != sha:
                    reasons.append(f"code changed after the freeze: {rel}")
            except OSError:
                reasons.append(f"frozen code file missing: {rel}")
        if not fz.get("code"):
            reasons.append("no code hashes in the freeze")
    reasons += perception_problems(scene, root)
    try:
        roster = a29_roster(namespace, root)
        if C.sha256_json(roster) != fz.get("a29_roster_sha256"):
            reasons.append("A29 roster differs from the frozen one")
    except RuntimeError as e:
        reasons.append(str(e))
    import treatment as T
    for src, sha in sorted((fz.get("snapshots") or {}).items()):
        d = T.source_dir(namespace, src, root)
        if not os.path.isdir(d) or C.dir_sha(d) != sha:
            reasons.append(f"snapshot {src} of {scene} differs from the freeze")
    if not fz.get("snapshots"):
        reasons.append("no snapshot hashes in the freeze")
    return {"ok": not reasons, "reasons": reasons}


#: K23 A37 paid-planning HOLD (Astra FP1-EQUIV-r5 point 7 / GATE-DISPOSITIONS A; K23-A37-ALIGN-r1 points 4-8): every PAID K2 / K3
#: reservation or batch hold is refused until the reviewed A37 test alignment is recorded and still holds. The record (written by
#: docs/final-test/k23/merge-drafts/a37/make_a37_alignment_record.py) binds the EXACT eight aligned test files, the versioned expected gate
#: totals, the _acc_runner 5.7 / 5.7b / 5.19 PASS evidence (log sha), the runtime files the gates exercised (a later runtime change
#: re-arms the hold), the commit that ran them, the run_gates artifact + logs that show all of it (re-checked here, K23-A37-HOLD-r1), and
#: an UNCONDITIONAL, case-bound Astra acceptance.
A37_RECORD = "k23/a37_alignment.json"
A37_SCHEMA = "k23.a37-alignment/3"
A37_CASE = "K23-A37-ALIGN"
A37_TESTS = ("sim/_ftm_callplan_selftest.py", "sim/_ftm_common_selftest.py", "sim/_ftm_monitor_selftest.py", "sim/_ftm_plan_selftest.py",
             "sim/_ftm_freeze_selftest.py", "sim/_ftm_treatment_selftest.py", "sim/_final_matrix_selftest.py", "sim/_acc_runner_selftest.py")
#: versioned expected totals of the seven A37 gates (alignment round 2): passed == total == these, nothing else lifts the hold
A37_EXPECTED = {"_ftm_callplan_selftest": 87, "_ftm_common_selftest": 120, "_ftm_monitor_selftest": 47, "_ftm_plan_selftest": 128,
                "_ftm_freeze_selftest": 121, "_ftm_treatment_selftest": 101, "_final_matrix_selftest": 150}
A37_ACC_LINES = ("PASS 5.7 ", "PASS 5.7b ", "PASS 5.19 ")
A37_ANSWER_MARK = "=== ANSWER (verbatim) ==="          # astra_call.py saves the prompt, then this marker, then Astra's answer
A37_RUNTIME = ("sim/final_test/matrix/common.py", "sim/final_test/matrix/monitor.py", "sim/final_test/matrix/plan.py",
               "sim/final_test_callplan.py", "sim/final_test/matrix/acc_common.py", "sim/final_test/matrix/k23_ns.py",
               "sim/final_test/matrix/freeze.py", "sim/final_test/matrix/treatment.py")


def a37_sha(path: str) -> str:
    """sha256 of a file's bytes with CRLF normalised to LF: the record binds CONTENT, so a checkout that rewrites line endings (git's
    autocrlf, the trap the K23 merge rehearsal found) neither lifts nor wrongly keeps the hold."""
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


#: selftests only (Astra K23-A37-ALIGN-r1 point 5): TEMPORARY roots whose fake-provider ledgers are exempt from the hold. Only a path
#: under the system temp directory can be registered, so a real paid run (repo work dir, or any alternate root outside temp) never is.
_A37_TEST_ROOTS: set = set()


def a37_exempt_test_root(path: str) -> str:
    import tempfile
    p = os.path.normcase(os.path.abspath(path))
    tmp = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
    if not p.startswith(tmp + os.sep):
        raise ValueError(f"A37 hold: only a temporary test root can be exempted (not {path})")
    _A37_TEST_ROOTS.add(p)
    return p


def a37_review_answer(full: str) -> str:
    """Astra's ANSWER in an astra_call.py review file: the text after its ONE answer-marker line. The saved prompt names the case and the
    commit too, so it never counts; no marker, a marker that is not a line of its own, or more than one marker (one inside the prompt)
    -> '' (no answer, the hold stays: Astra K23-A37-HOLD-r1 point 4, r2 point 1)."""
    lines = full.splitlines()
    at = [i for i, ln in enumerate(lines) if ln.strip() == A37_ANSWER_MARK]
    if full.count(A37_ANSWER_MARK) != 1 or len(at) != 1:
        return ""
    return "\n".join(lines[at[0] + 1:])


def _a37_counts(last: str):
    """(passed, total) from a gate's summary line, as run_gates.counts reads it; None when it has no counts."""
    import re
    m = re.search(r"(\d+)\s*/\s*(\d+)", last or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+) passed, (\d+) failed", last or "")
    return (int(m.group(1)), int(m.group(1)) + int(m.group(2))) if m else None


def _a37_evidence_problems(P: dict, rec: dict) -> list:
    """Astra K23-A37-HOLD-r1 points 1-3: the record's gate claims re-derived from the run_gates artifact it names and that artifact's logs
    (both committed next to each other: <label>.json + logs_<label>/<gate>.log). Every one of the seven gates and _acc_runner must have
    run on the record's commit (before = after), on the record's runtime + test bytes (before = after), and its log must hash to the
    artifact's log sha; the seven show rc 0 and passed == total == expected in their own summary line; the _acc_runner log has the exact
    PASS 5.7 / 5.7b / 5.19 lines."""
    H = "A37 paid-planning hold: "
    gf = rec.get("gates_file") or {}
    ap = os.path.join(P["root"], *str(gf.get("path") or "-").split("/"))
    try:
        if a37_sha(ap) != gf.get("sha256"):
            return [H + "the gate artifact changed after the record"]
        with open(ap, encoding="utf-8") as f:
            art = json.load(f)
    except (OSError, ValueError) as e:
        return [H + f"the gate artifact is missing or unreadable ({type(e).__name__})"]
    logs = os.path.join(os.path.dirname(ap), "logs_" + os.path.basename(ap)[:-5])
    tests, rt, gates = rec.get("tests") or {}, rec.get("runtime") or {}, rec.get("gates") or {}
    want = {**rt, **tests}
    out = []
    for g in list(A37_EXPECTED) + ["_acc_runner_selftest"]:
        a = art.get(g) or {}
        rel = f"sim/{g}.py"
        if not rec.get("commit") or not (a.get("head") == a.get("head_after") == rec.get("commit")):
            out.append(H + f"{g} did not run on the record's commit (or the commit moved during the gate)")
        m0, m1 = a.get("runtime") or {}, a.get("runtime_after") or {}
        if m0 != m1 or any(m0.get(r) != s for r, s in want.items()) or not want:
            out.append(H + f"{g} did not run on the recorded runtime / test bytes (or they changed during the gate)")
        if a.get("src_sha_lf") != tests.get(rel):
            out.append(H + f"{g}: the gate file that ran is not the recorded one")
        try:
            with open(os.path.join(logs, g + ".log"), encoding="utf-8", errors="replace") as f:
                stdout = f.read().partition("\n--- stderr ---\n")[0]
            if a37_sha(os.path.join(logs, g + ".log")) != a.get("log_sha"):
                out.append(H + f"{g}: its log does not hash to the artifact's log sha")
        except OSError:
            out.append(H + f"{g}: its log is missing")
            continue
        lines = [ln.strip() for ln in stdout.splitlines()]
        if g == "_acc_runner_selftest":
            if not all(any(ln.startswith(x) for ln in stdout.splitlines()) for x in A37_ACC_LINES) \
                    or rec.get("acc_runner_log_sha256") != a.get("log_sha"):
                out.append(H + "the _acc_runner log does not show PASS 5.7 / 5.7b / 5.19, or is not the recorded log")
            continue
        n = A37_EXPECTED[g]
        r = gates.get(g) or {}
        if (a.get("rc") != 0 or _a37_counts(a.get("last")) != (n, n) or str(a.get("last") or "").strip() not in lines
                or (r.get("rc"), r.get("last"), r.get("log_sha")) != (a.get("rc"), a.get("last"), a.get("log_sha"))):
            out.append(H + f"{g}: not rc 0 at {n}/{n} in its own log, or the record differs from the artifact")
    return out


def a37_hold_problems(root: str | None = None) -> list:
    """[] once the A37 alignment record is valid and still holds; otherwise the reasons (the hold stays)."""
    C = _C()
    P = C.paths(root)
    p = os.path.join(P["docs"], *A37_RECORD.split("/"))
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        return [f"A37 paid-planning hold: no valid alignment record ({os.path.basename(p)}: {type(e).__name__})"]
    return a37_record_problems(rec, root) if isinstance(rec, dict) else ["A37 paid-planning hold: the alignment record is not an object"]


def a37_record_problems(rec: dict, root: str | None = None) -> list:
    """The checks of a37_hold_problems on a record dict against the tree at `root` (also used by the record writer before it writes)."""
    C = _C()
    P = C.paths(root)
    out = []
    if rec.get("schema") != A37_SCHEMA or rec.get("sha256") != C.self_sha(rec):
        out.append("A37 paid-planning hold: alignment record schema / self-sha")
    g = rec.get("gates") or {}
    if set(g) != set(A37_EXPECTED) or any((v.get("rc"), (v.get("counts") or {}).get("passed"), (v.get("counts") or {}).get("total"))
                                          != (0, A37_EXPECTED[k], A37_EXPECTED[k]) for k, v in g.items() if k in A37_EXPECTED):
        out.append("A37 paid-planning hold: the recorded gates are not exactly the seven A37 gates at their expected totals")
    if sorted(rec.get("acc_runner_a37_checks") or []) != sorted(x.strip() for x in A37_ACC_LINES) \
            or not all((rec.get("acc_runner_a37_checks") or {}).values()) or not rec.get("acc_runner_log_sha256"):
        out.append("A37 paid-planning hold: the _acc_runner 5.7 / 5.7b / 5.19 PASS evidence is missing")
    out += _a37_evidence_problems(P, rec)
    tests = rec.get("tests") or {}
    if sorted(tests) != sorted(A37_TESTS):
        out.append("A37 paid-planning hold: the record does not bind exactly the eight aligned test files")
    rt = rec.get("runtime") or {}
    if sorted(rt) != sorted(A37_RUNTIME):
        out.append("A37 paid-planning hold: the record does not bind the exercised runtime files")
    for rel, sha in sorted(list(tests.items()) + list(rt.items())):
        try:
            if a37_sha(os.path.join(P["root"], *rel.split("/"))) != sha:
                out.append(f"A37 paid-planning hold: {rel} changed after the alignment record")
        except OSError:
            out.append(f"A37 paid-planning hold: {rel} missing")
    rv = rec.get("astra_review") or {}
    rp = os.path.join(P["root"], *str(rv.get("path") or "-").split("/"))
    try:
        txt = a37_review_answer(open(rp, encoding="utf-8").read())
        first = next((ln.strip() for ln in txt.splitlines() if ln.startswith("RULING:")), "")
        if (a37_sha(rp) != rv.get("sha256") or first != "RULING: ACCEPTED" or A37_CASE not in txt
                or not rec.get("commit") or str(rec["commit"])[:7] not in txt):
            out.append("A37 paid-planning hold: Astra's review is changed, not an UNCONDITIONAL 'RULING: ACCEPTED', or not bound to this "
                       "case and commit")
    except OSError:
        out.append("A37 paid-planning hold: Astra's review file missing")
    return out


def a37_hold_for_ledger(ledger_path: str, usd: float) -> list:
    """The K23 checks of ONE paid reservation / hold on any ledger whose directory names a K2 / K3 namespace, under ANY root, run first in
    acc_common.guard_problems (so a registered guard does not bypass them: Astra K23-A37-ALIGN-r1 point 4). Free (0 USD) entries pass.
    (1) Alternate roots (point 5): a paid K2 / K3 ledger must be THIS repo's <matrix work>/<namespace>/ledger.db, where the study-wide and
    70 USD replication caps apply; anywhere else it is refused, always (not only while the hold stands), unless it sits under a registered
    temporary selftest root (fake providers). (2) The A37 hold: refused until this repo's alignment record is valid and still holds."""
    if float(usd) <= 0:
        return []
    ap = os.path.abspath(ledger_path)
    lp = os.path.normcase(ap)
    if any(lp.startswith(r + os.sep) for r in _A37_TEST_ROOTS):     # a registered TEMPORARY selftest root (fake providers only)
        return []
    C = _C()
    ns = C.namespace_of_dir(os.path.basename(os.path.dirname(ap)))
    if not ns or not is_replication(ns):
        return []
    work = os.path.normcase(os.path.abspath(C.paths(None)["matrix_work"]))
    if os.path.basename(ap) != "ledger.db" or os.path.normcase(os.path.dirname(os.path.dirname(ap))) != work:
        return [f"K23: a paid {ns} entry on a ledger outside this repo's matrix work dir ({ap}) is refused: alternate-root paid K2 / K3 "
                f"operation is not supported (Astra K23-A37-ALIGN-r1 point 5); nothing sent"]
    return a37_hold_problems()


def replication_problems(ledger_path: str, own_usd: float, spend: dict) -> list:
    """Astra K23-NAMESPACES-r1 item 6: a paid reservation on a K2/K3 ledger also needs the COMBINED replication liability (every
    replication namespace's ledger, reconciled by study_spend, + this ledger's own liability incl. the pending call) <= REPLICATION_USD.
    Run inside the study-wide boundary (acc_common.guard_problems). Empty list = inside."""
    C = _C()
    own_ns = C.namespace_of_dir(os.path.basename(os.path.dirname(os.path.abspath(ledger_path))))
    if not own_ns or not is_replication(own_ns):
        return []
    others = 0.0
    for rel, v in (spend.get("ledgers") or {}).items():
        ns = C.namespace_of_dir(os.path.basename(os.path.dirname(rel)))
        if ns and is_replication(ns):
            others += float(v.get("liability", 0.0))
    tot = others + float(own_usd)
    if tot > REPLICATION_USD + 1e-12:
        return [f"replication budget: K2+K3 liability {others:.4f} + this ledger {float(own_usd):.4f} = {tot:.4f} > {REPLICATION_USD:.2f} USD "
                f"(Arash approve-all-6 item 4)"]
    return []
