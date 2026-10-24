"""Selftest for sim/final_test/acceptance_v2/acc_attempt2.py (stage ACC-ATTEMPT2, 2026-10-14): attempt 2 = the ONE registered correction
cycle of acceptance-v2. Offline (Windows python), no simulator, no LLM, no network. Every write goes to a temporary tree under
<repo>/acceptance_bundles/_selftest (same volume, gitignored); the real acceptance-v2 / -a2 docs + work dirs and the real bundle dir must stay
byte-identical. Sections 1-3 and 13 hold in every state of the study (before / after attempt 1); sections 4-11b stub decide / the bundle
replay / the identity / the owed state on purpose (mechanics and refusals); section 12 runs the WHOLE chain UNSTUBBED on a full copy of the
tracked tree: a real attempt-1 verdict, an identity-changing correction through Arash's go and a2_start, the a2 journals, the real attempt-2
verdict and both_attempts (real decide, real bundle replays in fresh interpreters, the real identity; the real owed state, which section 12
replaces by a clean one ONLY if the study's own owed state refuses at the time, and says so). The gate summary is the one synthesized
fixture: the gate run cannot run inside one of its own gates.

    python sim/_acc_attempt2_selftest.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
sys.path.insert(0, os.path.join(HERE, "final_test", "acceptance_v2"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import common as C                                       # noqa: E402
import acc_common as AC                                  # noqa: E402
import acc_runner as A                                   # noqa: E402
import acc_attempt2 as A2                                # noqa: E402
import treatment as T                                    # noqa: E402

N = {"ok": 0, "fail": 0}
SCRATCH = os.path.join(ROOT, A2.BUNDLES, "_selftest")


def check(name, cond, detail=""):
    N["ok" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  :: {str(detail)[:400]}"))


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def real_snapshot():
    out = {}
    bases = [C.ns_docs(ns) for ns in C.ACCEPTANCE_NAMESPACES] + [C.ns_work(ns) for ns in C.ACCEPTANCE_NAMESPACES]
    bases += [os.path.join(ROOT, A2.BUNDLES, ns) for ns in C.ACCEPTANCE_NAMESPACES]
    for base in bases:
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                q = os.path.join(dp, fn)
                out[os.path.relpath(q, ROOT)] = A.file_sha(q)
    out["exists"] = str([os.path.isdir(b) for b in bases])
    return out


REAL0 = real_snapshot()
REG = A.registration()
ROSTER = AC.acceptance_roster(None, REG)
os.makedirs(SCRATCH, exist_ok=True)
SCRATCH_LOCK = os.path.join(SCRATCH, "study_spend.lock")
A2.STUDY_LOCK_OVERRIDE = SCRATCH_LOCK                     # the finalizations below never touch the real study-wide lock


def tmpdir(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=SCRATCH)


# ------------------------------------------------------------------------------------------------ 1 the namespace
# ------------------------------------------------------------------------------------------------ A36 successor state (Astra A36-IMPL-freeze-r2)
# acc_attempt2.py is NOT changed. After the A34-A2 attempt-2 tree, the signed A36 implementation moved tracked code. The historical checks
# (6.20a fingerprint, 6.20g exception) stay bound to AUTHENTICATED historical evidence (the replay-verified attempt-2 verdict + its bundle);
# the current tree may differ from them ONLY through the A36 successor branch: replay-verified terminal NOT PASS / NOT PASS under a valid A36
# waiver, a valid A36 implementation record with Arash's A36-IMPL approval, the complete structured changed-set / pin / freeze.json succession
# checks (freeze_ft2.a34_a36_exception) and an intact succession record. Without the record on disk the branch refuses (the gate stays red).
import freeze_ft2 as FT                                  # noqa: E402

_SUCC_CACHE = {}


class _RealImpls:
    """The successor / history checks always run on the REAL replay / decision code (never on this selftest's fixture stubs)."""
    def __enter__(self):
        self.saved = (A2.compute_decision, A2.owed_state, A2.identity_parts, A2.REPLAY)
        A2.compute_decision, A2.owed_state, A2.identity_parts, A2.REPLAY = REAL
        A2._REPLAY_MEMO.clear()

    def __exit__(self, *exc):
        A2.compute_decision, A2.owed_state, A2.identity_parts, A2.REPLAY = self.saved
        A2._REPLAY_MEMO.clear()
        return False


def a36_successor(sigs=None):
    """-> (ok, reasons, refrozen {rel: sha}). sigs: an in-memory signatures.json (fixtures only); None = the file on disk (cached)."""
    if sigs is None and "disk" in _SUCC_CACHE:
        return _SUCC_CACHE["disk"]
    why = []
    with _RealImpls():
        rec, g = FT.acceptance_record(ROOT, sigs)         # both verdicts replayed in their bundles (verdict_problems(k, replay=True))
    att = rec.get("attempts") or {}
    if not (g.get("status") == "WAIVED" and g.get("pass") is True and rec.get("verdict") == "NOT PASS" and rec.get("terminal") is True
            and all(a and a.get("intact") and a.get("verdict") == "NOT PASS" for a in (att.get("1"), att.get("2")))):
        why.append(f"acceptance is not the replay-verified terminal NOT PASS / NOT PASS under a valid A36 waiver ({g.get('status')}, {rec.get('verdict')})")
    refz, w = FT.a34_a36_exception(ROOT, sigs)
    why += w
    if not refz:
        why.append("the re-freeze exception holds nothing")
    imp = FT.a36_impl_check(ROOT, sigs)
    if not imp["ok"]:
        why += [f"A36-IMPL: {x}" for x in imp["reasons"]]
    st = FT.succession_record(ROOT, sigs, imp)
    if not st["ok"]:
        why += [f"succession: {x}" for x in st["reasons"]]
    out = (not why, why, refz if not why else {})
    if sigs is None:
        _SUCC_CACHE["disk"] = out
    return out


def hist_fingerprint_problems():
    """The pinned new fingerprint against AUTHENTICATED history: the attempt-2 verdict (replay-verified) records it and its preserved bundle
    code reproduces it."""
    SA = A2.SIGNED_AMENDMENT
    v2 = A2._json(A2.verdict_path(2)) or {}
    out = []
    if v2.get("rule_fingerprint") != SA["rule_fingerprint_new"]:
        out.append("the attempt-2 verdict does not record the pinned new fingerprint")
    with _RealImpls():
        vp2 = A2.verdict_problems(2, replay=True)
    if vp2:
        out.append("the attempt-2 verdict is not intact (bundle replay)")
    if A2.rule_fingerprint(A2.bundle_dir(2))["sha"] != SA["rule_fingerprint_new"]:
        out.append("the attempt-2 bundle code does not reproduce the pinned new fingerprint")
    return out


def fingerprint_moved_files():
    """Files of every module the decision-rule closure references (now or in the attempt-2 bundle) whose bytes differ from the bundle."""
    bd = A2.bundle_dir(2)
    fn, fb = A2.rule_fingerprint(), A2.rule_fingerprint(bd)
    names = set(fn["names"]) | set(fb["names"])
    i_now, i_b = A2.module_index(), A2.module_index(bd)
    moved = set()
    for m in set(i_now) | set(i_b):
        if not any(n.startswith(m + ".") for n in names):
            continue
        pn, pb = i_now.get(m), i_b.get(m)
        rel_ = os.path.relpath(pn or pb, ROOT if pn else bd).replace(os.sep, "/")
        if not pn or not pb or A.file_sha(pn) != A.file_sha(pb):
            moved.add(rel_)
    return sorted(moved)


def p_620a(sigs=None):
    """6.20a, state-aware -> (ok, detail)."""
    SA = A2.SIGNED_AMENDMENT
    hp = hist_fingerprint_problems()
    disk = A2.rule_fingerprint()["sha"]
    if hp:
        return False, hp
    if disk == SA["rule_fingerprint_new"]:
        return True, "historical state: the disk fingerprint is the pinned new one"
    ok, why, refz = a36_successor(sigs)
    moved = fingerprint_moved_files()
    unexplained = [r for r in moved if r not in refz]
    return (ok and bool(moved) and not unexplained), {"disk": disk[:12], "successor": why[:4], "moved": moved, "unexplained": unexplained}


_TREE_ONLY = ("tracked files changed outside the signed set:", "freeze.json changed beyond the added amendment_freezes entry",
              "freeze.json does not carry exactly", "freeze.json's amendment freeze is not")


def p_620g(rec_real, v1_real, sigs=None):
    """6.20g, state-aware -> (ok, detail): [] on the signed tree; else every problem is a tree-state one AND the A36 successor branch holds."""
    ap = A2.amendment_problems(rec_real, v1_real)
    if ap == []:
        return True, "historical state"
    tree_only = all(x.startswith(_TREE_ONLY) or x.endswith(" is not at its pinned sha") for x in ap)
    ok, why, _ = a36_successor(sigs)
    return tree_only and ok, {"amendment_problems": ap, "successor": why[:4]}


#: gates added after the frozen REQUIRED_GATES manifest (2026-10-14) by the signed A36's reviewed implementation: new files since the A34-A2
#: base commit and in the A36 code scope (freeze_ft2.A36_CODE_SCOPE); every other addition / any missing required gate is refused
A36_ADDED_GATES = {"_ftm_a36_selftest": "the A36 analysis (excluded_reset, paired common subset, ANALYSIS_PLAN ftm-analysis-3)",
                   "_ftm_freeze_gates_selftest": "the ft-v2 freeze evidence gates (G-iso / 12.5 / tokens / context / leakage final identity)"}


def gate_set_problems(tree):
    out = []
    req = set(A2.REQUIRED_GATES)
    if sorted(req - set(tree)):
        out.append(f"required gates missing: {sorted(req - set(tree))}")
    extra = set(tree) - req - set(A2.GATES_NOT_OFFLINE) - set(A36_ADDED_GATES)
    if extra:
        out.append(f"unexplained additions: {sorted(extra)}")
    gone = set(A36_ADDED_GATES) - set(tree)
    if gone:
        out.append(f"accounted A36 gates missing: {sorted(gone)}")
    for g in A36_ADDED_GATES:
        rel_ = f"sim/{g}.py"
        if g in req or rel_ not in FT.A36_CODE_SCOPE or FT._git_blob_sha(FT.A36_BASE_COMMIT, rel_) is not None:
            out.append(f"{g} is not an A36 addition (new since the base commit, in the A36 code scope)")
    return out


# ---- a temporary A36 implementation record + in-memory approval (fixture; the real record / signatures.json are never written)
_IMPL0, _REV0 = FT.A36_IMPL_FILE, dict(FT.A36_REQUIRED_REVIEWS)


def impl_fixture_on(td):
    rel_ = os.path.relpath(td, ROOT).replace(os.sep, "/")
    open(os.path.join(td, "final-review.txt"), "w", encoding="utf-8").write("**VERDICT: ACCEPT** (selftest fixture, not a review)\n")
    FT.A36_IMPL_FILE = rel_ + "/A36.impl.json"
    FT.A36_REQUIRED_REVIEWS = dict(_REV0, **{FT.A36_FINAL_REVIEW: rel_ + "/final-review.txt"})
    srec = FT.a36_signature(ROOT)[1]
    rec = {"schema": FT.A36_IMPL_SCHEMA, "amendment": "A36", "signature_key": "A36", "approval_key": FT.A36_IMPL_KEY, "a36_file": FT.A36_FILE,
           "a36_file_sha256": srec["file_sha256"], "base_commit": FT.A36_BASE_COMMIT, "written": "2026-10-16T00:30:00",
           "files": {r: {"before": FT._git_blob_sha(FT.A36_BASE_COMMIT, r), "after": FT._sha(ROOT, r)} for r in FT.A36_CODE_SCOPE},
           "reviews": [{"id": rid, "file": p_, "sha256": FT._sha(ROOT, p_),
                        "verdict": FT._review_verdict(open(os.path.join(ROOT, *p_.split("/")), encoding="utf-8").read())}
                       for rid, p_ in FT.A36_REQUIRED_REVIEWS.items()],
           "reviewed_commits": [{"commit": FT.A36_GATES_FIX_COMMIT}]}
    rec["sha256"] = C.self_sha(rec)
    p_ = os.path.join(ROOT, *FT.A36_IMPL_FILE.split("/"))
    open(p_, "w", encoding="utf-8", newline="\n").write(json.dumps(rec, indent=1) + "\n")
    sig = json.load(open(os.path.join(ROOT, "docs", "final-test", "signatures.json"), encoding="utf-8"))
    sig[FT.A36_IMPL_KEY] = {"by": "Arash", "date": "2026-10-16", "said": "selftest fixture (in memory only)",
                            "prereg_sha256": A.file_sha(C.paths()["prereg"]), "file": FT.A36_IMPL_FILE, "file_sha256": A.file_sha(p_),
                            "authorized_at": "2026-10-16T00:05:00+03:30", "record_created_at": rec["written"],
                            "authorization_preceded_record_creation": True, "authorization_kind": "pre-approval (selftest fixture)"}
    return sig


def impl_fixture_off():
    FT.A36_IMPL_FILE, FT.A36_REQUIRED_REVIEWS = _IMPL0, dict(_REV0)


print("== 1 namespace acceptance-v2-a2")
n2 = C.NAMESPACES.get(A2.NS2) or {}
check("1.1 a2: roster items, 8 USD total + attempt ceiling, study guard, ft-v2, attempt 2 of acceptance-v2",
      n2.get("items") == C.ACCEPTANCE_ROSTER and n2.get("ceilings") == {"total": 8.0} and n2.get("attempt_ceiling") == 8.0
      and n2.get("study_guard") is True and n2.get("treatment") == "ft-v2" and n2.get("attempt_of") == A2.NS1 and n2.get("attempt") == 2, n2)
check("1.2 attempt 1's namespace entry is unchanged (16 total, 8 per attempt)",
      C.NAMESPACES[A2.NS1] == {"items": "acceptance_roster", "ceilings": {"total": 16.0}, "attempt_ceiling": 8.0, "archived_arms": (),
                               "treatment": "ft-v2", "study_guard": True})
check("1.3 separate journals, docs, work (ledger, cache) and bundle dirs",
      C.ns_docs(A2.NS2) != C.ns_docs(A2.NS1) and C.ns_work(A2.NS2) != C.ns_work(A2.NS1) and A2.bundle_dir(1) != A2.bundle_dir(2)
      and C.journal_path(A2.NS2, "plan") != C.journal_path(A2.NS1, "plan") and A2.ledger_path(2) != A2.ledger_path(1))
check("1.4 the registered 16 USD acceptance total = 2 x 8", REG["spend"]["ceiling_usd_namespace"] == 16.0 and REG["spend"]["ceiling_usd_per_attempt"] == 8.0)
check("1.5 guard_items refuses the roster in a2 without an armed session", raises(lambda: C.guard_items([ROSTER[0]], A2.NS2), C.TestItemRefused))
AC.arm_session([], by="selftest", namespace=A2.NS1)
a1_ok = not raises(lambda: C.guard_items([ROSTER[0]], A2.NS1), C.TestItemRefused)
a2_no = raises(lambda: C.guard_items([ROSTER[0]], A2.NS2), C.TestItemRefused)
AC.arm_session([], by="selftest", namespace=A2.NS2)
a2_ok = not raises(lambda: C.guard_items([ROSTER[0]], A2.NS2), C.TestItemRefused)
a1_no = raises(lambda: C.guard_items([ROSTER[0]], A2.NS1), C.TestItemRefused)
AC.arm_session([], by="selftest", namespace=A2.NS2, lifecycle=lambda: ["record edited"])
lc_no = raises(lambda: C.guard_items([ROSTER[0]], A2.NS2), C.TestItemRefused)
AC.disarm_session()
AC.SESSION.update(armed=True, by="legacy")                # a session armed the old way (no namespace) is an attempt-1 session only
legacy = (not raises(lambda: C.guard_items([ROSTER[0]], A2.NS1), C.TestItemRefused)) and raises(lambda: C.guard_items([ROSTER[0]], A2.NS2), C.TestItemRefused)
AC.disarm_session()
check("1.5b the session is armed for ONE namespace and its lifecycle is re-checked on every guard call",
      a1_ok and a2_no and a2_ok and a1_no and lc_no and legacy, (a1_ok, a2_no, a2_ok, a1_no, lc_no, legacy))
m_ = C.load_manifest()
check("1.6 a2 runs the same canon-only design: no repeat / relocation items", C.repeat_items(A2.NS2, m_) == [] and C.reloc_items(A2.NS2, m_) == [])
check("1.7 ns_of: 1 -> acceptance-v2, 2 -> acceptance-v2-a2, 3 does not exist",
      A2.ns_of(1) == A2.NS1 and A2.ns_of(2) == A2.NS2 and A.ns_of(2) == A2.NS2 and raises(lambda: A2.ns_of(3), ValueError))
check("1.8 a2 is ft-v2, not a draft-briefing / strict-mask namespace", T.is_ft2(A2.NS2) and A2.NS2 not in T.DEV_DRAFT_NAMESPACES + T.STRICT_NAMESPACES)
check("1.9 the treatment identity does not depend on the namespace name (a1 == a2 on the same code)",
      T.identity(A2.NS1)["sha"] == T.identity(A2.NS2)["sha"])
check("1.10 the same registered order serves both attempts", A.main_order(ROSTER) == REG["execution_order"]["main_order"] and not A.order_problems(REG, ROSTER))

# ------------------------------------------------------------------------------------------------ 2 the real repo: refusals (any state)
print("== 2 the real repo: attempt 2 refused without the whole chain")
pf2 = A.preflight(attempt=2, identity=False)
check("2.1 preflight(attempt=2) refuses on the real repo (no correction record exists)", pf2 and all(x.startswith("attempt 2") for x in pf2), pf2)
check("2.2 preflight(attempt=3) refuses: no such attempt", any("does not exist" in x for x in A.preflight(attempt=3, identity=False)))
check("2.3 arm_guard(attempt 2) / (attempt 3) refuse",
      raises(lambda: A.arm_guard(None, 2, path=os.path.join(SCRATCH, "nope", "ledger.db"), record=False), A.Refused)
      and raises(lambda: A.arm_guard(None, 3, record=False), A.Refused))
calls = []
orig = (A.make_plan, A.make_execute, A.capture_live, A.fake_overcap)
A.make_plan = lambda *a, **k: calls.append(("plan", a, k))
A.make_execute = lambda *a, **k: calls.append(("execute", a, k))
A.capture_live = lambda *a, **k: calls.append(("capture", a, k))
A.fake_overcap = lambda *a, **k: calls.append(("fake", a, k))
try:
    rcs = [A.main(["plan", "--stage", "opus", "--attempt", "2"]), A.main(["plan", "--stage", "gemma", "--attempt", "2"]),
           A.main(["execute", "--attempt", "2"]), A.main(["capture", "--attempt", "2"]),
           A.main(["fake-overcap", "--on-namespace", "--attempt", "2"]), A.main(["score", "--attempt", "2"]),
           A.main(["check", "--attempt", "2"]), A.main(["c11", "--attempt", "2"]), A.main(["seal-roster", "--attempt", "2"]),
           A.main(["plan", "--stage", "gemma", "--attempt", "3"]), A.main(["verdict", "--attempt", "2"])]
finally:
    A.make_plan, A.make_execute, A.capture_live, A.fake_overcap = orig
check("2.4 CLI --attempt 2 / 3: every command refused (rc 2) and NOTHING dispatched (mocked)", rcs == [2] * 11 and calls == [], (rcs, calls))
check("2.5 correction-status runs read-only (rc 0)", A.main(["correction-status"]) == 0)
p_ = subprocess.run([sys.executable, os.path.join(HERE, "final_test", "acceptance_v2", "acc_runner.py"), "verdict", "--attempt", "2"],
                    capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
check("2.6 the REAL CLI entry (python acc_runner.py verdict --attempt 2) refuses cleanly: rc 2, REFUSED, no traceback (Astra r1 11)",
      p_.returncode == 2 and "REFUSED" in p_.stdout and "Traceback" not in (p_.stderr + p_.stdout), (p_.returncode, p_.stdout[-200:], p_.stderr[-300:]))
had_v1 = os.path.exists(A2.verdict_path(1))
rc_v = A.main(["verdict", "--attempt", "1"])
check("2.7 verdict --attempt 1 without --write writes nothing", rc_v in (0, 2) and os.path.exists(A2.verdict_path(1)) == had_v1, rc_v)

# ------------------------------------------------------------------------------------------------ 3 fingerprint + protection
print("== 3 rule fingerprint, tracked / protected files")
fp = A2.rule_fingerprint()
check("3.1 the fingerprint covers decide / evidence / c11_scan / source_inventory, their callees across modules (C11 -> capture leakage)",
      fp == A2.rule_fingerprint() and {"acc_runner.decide", "acc_runner.evidence", "acc_runner.c11_scan", "acc_runner.source_inventory",
                                       "acc_runner.provenance_problem", "common.cell_id"} <= set(fp["names"])
      and any(n.startswith("c11_audit.") for n in fp["names"]) and any(n.startswith("capture.") for n in fp["names"]), fp["names"][:10])
check("3.1b the (g) recomputation is inside the fingerprint when it exists", not hasattr(A, "g_recompute") or "acc_runner.g_recompute" in fp["names"])
CODE_DIRS = [("sim",), ("sim", "planners"), ("sim", "final_test", "matrix"), ("sim", "final_test", "acceptance_v2"),
             ("sim", "final_test", "diag", "c11"), ("sim", "final_test", "c3")]


def code_copy():
    td = tmpdir("fp_")
    for parts in CODE_DIRS:
        s, d = os.path.join(ROOT, *parts), os.path.join(td, *parts)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(s):
            if f.endswith(".py") or (parts == ("sim", "planners") and f.endswith(".json")):
                shutil.copyfile(os.path.join(s, f), os.path.join(d, f))
    return td


TDF = code_copy()
try:
    base = A2.rule_fingerprint(TDF)["sha"]

    def edited(rel, old, new):
        q = os.path.join(TDF, *rel.split("/"))
        src = open(q, encoding="utf-8").read()
        assert src.count(old) >= 1, (rel, old)
        open(q, "w", encoding="utf-8").write(src.replace(old, new, 1))
        got = A2.rule_fingerprint(TDF)["sha"]
        open(q, "w", encoding="utf-8").write(src)
        return got
    same = edited("sim/final_test/acceptance_v2/acc_runner.py", "    ok = all(not v for v in crit.values())",
                  "    ok = all(not v for v in crit.values())  # a comment only") == base
    tuned = edited("sim/final_test/acceptance_v2/acc_runner.py", "if n_ok < 46:", "if n_ok < 45:") != base
    callee = edited("sim/final_test/acceptance_v2/acc_runner.py", "def kb_record_problem(x: dict, inv: dict):",
                    "def kb_record_problem(x: dict, inv: dict):\n    x = dict(x)") != base
    plumbing = edited("sim/final_test/acceptance_v2/acc_runner.py", "def capture_live(ollama: str, attempt: int = 1) -> int:",
                      "def capture_live(ollama: str, attempt: int = 1) -> int:\n    x = 1") == base
    cap_src = open(os.path.join(TDF, "sim", "final_test", "matrix", "capture.py"), encoding="utf-8").read()
    leak = [n for n in fp["names"] if n == "capture.leakage_problems"]
    leak_moves = bool(leak)
    if leak:
        fn = "leakage_problems"
        leak_moves = edited("sim/final_test/matrix/capture.py", f"def {fn}(", f"def {fn}(*_a, **_k):\n    return []\n\n\ndef _was_{fn}(") != base
    pin = edited("sim/final_test/matrix/treatment.py", T.FT2["pins"]["executor"]["sim/honest_skills.py"], "0" * 64) == base
    check("3.2 a comment does not move it; the (f) threshold 46 -> 45 does; a callee does; capture plumbing does not",
          same and tuned and callee and plumbing, (same, tuned, callee, plumbing))
    check("3.3 replacing a capture-module measurement the rule reaches (via c11_audit's alias) moves it (Astra r1 1)", leak_moves, leak)
    pkg = edited("sim/ltm_package.py", "class PackageBuilder", "PB_FIX = 1" + chr(10) * 3 + "class PackageBuilder") == base
    idin = edited("sim/final_test/matrix/treatment.py", "def id_in(", "def id_in(*_a):" + chr(10) + "    return False" + chr(10) * 3 + "def _old_id_in(") != base
    check("3.4 re-pinning an executor file in treatment.FT2 does NOT move it (Astra r1 5: re-pinning stays possible)", pin)
    check("3.4b treatment code (ltm_package.PackageBuilder) is NOT in the rule hash; the leakage match treatment.id_in IS",
          pkg and idin and "ltm_package.PackageBuilder" in fp["treatment_refs"] and "treatment.id_in" in fp["names"], (pkg, idin))
    edge = edited("sim/final_test/matrix/treatment.py", '_ID_EDGE = "A-Za-z0-9_-"', '_ID_EDGE = "A-Za-z0-9_:-"') != base
    check("3.4c a CONSTANT a measurement definition depends on (treatment._ID_EDGE, used by id_in) moves it (Astra r2 2)",
          edge and "treatment._ID_EDGE" in fp["names"] and T.id_in("g-l2-slice-bread", "prefix:g-l2-slice-bread"), edge)
    pc0 = A2.planner_config_sha(TDF)["sha"]

    def pc_edited(rel, old, new):
        q = os.path.join(TDF, *rel.split("/"))
        src = open(q, encoding="utf-8").read()
        assert src.count(old) >= 1, (rel, old)
        open(q, "w", encoding="utf-8").write(src.replace(old, new, 1))
        got = A2.planner_config_sha(TDF)["sha"]
        open(q, "w", encoding="utf-8").write(src)
        return got
    model = pc_edited("sim/planners/__init__.py", '"opus": ("anthropic", "claude-opus-5-5")', '"opus": ("anthropic", "claude-sonnet-5-5")') != pc0
    dflt = pc_edited("sim/planners/codex_provider.py", 'sandbox: str = "read-only"', 'sandbox: str = "workspace-write"') != pc0
    spec = pc_edited("sim/final_test/matrix/common.py", 'CODEX_MAIN = "gpt-6-sol"', 'CODEX_MAIN = "gpt-6-luna"') != pc0
    cmt = pc_edited("sim/planners/__init__.py", "def make_planner(", "# a comment\ndef make_planner(") == pc0
    check("3.4d the model selection digest: Opus -> Sonnet in planners.ARMS, a provider default, the Codex main model move it; a comment "
          "does not (Astra r2 3)", model and dflt and spec and cmt and pc0 == A2.planner_config_sha()["sha"], (model, dflt, spec, cmt))
    try:
        t0 = A2.prompt_texts_sha(TDF)
        q_c = os.path.join(TDF, "sim", "planners", "contract.py")
        src_c = open(q_c, encoding="utf-8").read()
        open(q_c, "w", encoding="utf-8").write(src_c.replace("SYSTEM_V2 = ", "SYSTEM_V2 = 'EDITED ' + ", 1))
        t1 = A2.prompt_texts_sha(TDF)
        open(q_c, "w", encoding="utf-8").write(src_c)
        txt_ok = t0 == A2.prompt_texts_sha() and t1 != t0 and json.load(open(os.path.join(ROOT, "sim", "planners", "ft_v2.json"),
                                                                                encoding="utf-8")).get("texts_sha256") == t0
        txt_info = (t0[:12], t1[:12])
    except Exception as e:                                     # noqa: BLE001
        txt_ok, txt_info = False, f"{type(e).__name__}: {e}"
    check("3.4e the RENDERED prompt texts are recomputed by a fresh interpreter: an edit of contract.SYSTEM_V2 changes them although no "
          "stored sha moved (Astra r2 7)", txt_ok, txt_info)
    body = pc_edited("sim/planners/anthropic_provider.py", "        super().__init__(model, **kw)", "        super().__init__('claude-sonnet-5-5', **kw)") != pc0
    base_m = pc_edited("sim/planners/base.py", "        self.model = model", "        self.model = 'gemma3:4b'") != pc0
    req = pc_edited("sim/planners/anthropic_provider.py", '"model": self.model', '"model": "claude-sonnet-5-5"') != pc0
    harmless = pc_edited("sim/planners/anthropic_provider.py", "    def decoding(self) -> dict:", "    def decoding(self) -> dict:\n        _unused = 1") == pc0
    check("3.4f RUNNER-FINAL (Astra r3 interim 1): a constructor BODY re-selecting the model, the base class's model assignment and a "
          "request body's model key each move the model-selection digest; an unrelated provider line does not", body and base_m and req and harmless,
          (body, base_m, req, harmless))
    try:
        r0 = A2.rendered_prompts_sha(TDF)
        src_c = open(q_c, encoding="utf-8").read()
        open(q_c, "w", encoding="utf-8").write(src_c.replace("            return SYSTEM_V2.format(", "            return 'EDITED ' + SYSTEM_V2.format(", 1))
        r1, t2 = A2.rendered_prompts_sha(TDF), A2.prompt_texts_sha(TDF)
        open(q_c, "w", encoding="utf-8").write(src_c)
        rend_ok, rend_info = r0 == A2.rendered_prompts_sha() and r1 != r0 and t2 == t0, (r0[:12], r1[:12], t2[:12])
    except Exception as e:                                     # noqa: BLE001
        rend_ok, rend_info = False, f"{type(e).__name__}: {e}"
    check("3.4g RUNNER-FINAL (Astra r3 interim 4): an edit of PlannerContract.system_text moves the RENDERED messages digest although every "
          "frozen text constant (prompt_texts_sha) is unchanged", rend_ok, rend_info)
    rf0 = A2.rendering_fingerprint(TDF)["sha"]

    def rf_edited(rel, old, new):
        q = os.path.join(TDF, *rel.split("/"))
        src = open(q, encoding="utf-8").read()
        assert src.count(old) >= 1, (rel, old)
        open(q, "w", encoding="utf-8").write(src.replace(old, new, 1))
        got = A2.rendering_fingerprint(TDF)["sha"], A2.rendered_prompts_sha(TDF)
        open(q, "w", encoding="utf-8").write(src)
        return got
    hold_rf, hold_rp = rf_edited("sim/planners/contract.py", 'holding=holding or "nothing"', 'holding=holding or "empty"')
    summ_rf, _ = rf_edited("sim/final_test/matrix/treatment.py", "def prompt_summary(full_lines, cfg: dict | None = None) -> list:",
                           "def prompt_summary(full_lines, cfg: dict | None = None) -> list:\n    full_lines = list(full_lines)[:1]")
    tools_rf, _ = rf_edited("sim/final_test/matrix/plan.py", "return PlannerContract(T.planner_types(types), tools=hs.TOOLS, version=T.FT2_VERSION)",
                            "return PlannerContract(T.planner_types(types)[:5], tools=hs.TOOLS, version=T.FT2_VERSION)")
    valid_rf, _ = rf_edited("sim/planners/validator_v2.py", "def ", "def _selftest_extra():\n    return 1\n\n\ndef ")
    check("3.4h RUNNER-FINAL (Astra r3b 1): the rendering CODE is frozen: the empty-hand wording (holding or 'empty') moves the rendering "
          "fingerprint AND the rendered probe (empty / occupied hands, EN / FA, each block alone); the treatment prompt summary and the "
          "contract's type list move the fingerprint; the repairable validator does not; real tree == copy",
          hold_rf != rf0 and hold_rp != A2.rendered_prompts_sha(TDF) and summ_rf != rf0 and tools_rf != rf0 and valid_rf == rf0
          and rf0 == A2.rendering_fingerprint()["sha"] and not A2.rendering_fingerprint()["missing"],
          (hold_rf[:8], summ_rf[:8], tools_rf[:8], valid_rf[:8], rf0[:8]))
finally:
    shutil.rmtree(TDF, ignore_errors=True)
prot = ["docs/final-test/acceptance_registration.json", "docs/final-test/acceptance_rulings.json", "docs/final-test/acceptance_owed.json",
        "docs/final-test/acceptance_authorization.json", "docs/final-test/freeze.json", "sim/dataset/general.json", "sim/step5/package_params.json",
        "sim/step5/procedures_kb.json", "sim/step5/lexicon/concepts.json", "sim/weights/v3/FloorPlan1.yolo26s.pt",
        "sim/final_test/snapshots_ft2/o0/k10/ltm.db", "sim/final_test/matrix/score.py", "sim/final_test/diag/c11/c11_audit.py",
        "sim/final_test/matrix/context_fit.py", A2.SELF_REL, "docs/final-test/matrix/fs_blocks.json",
        "docs/final-test/kitchen_memory/briefings/FloorPlan1.layout.json", "sim/planners/__init__.py"]
free = ["sim/honest_skills.py", "sim/honest.py", "sim/planners/validator_v2.py", "sim/planners/exec_contract.json", "sim/ltm_package.py",
        "sim/planners/contract.py", "sim/final_test/matrix/plan.py", "sim/final_test/matrix/treatment.py",
        "sim/final_test/acceptance_v2/acc_runner.py"]
check("3.5 protected: registration, rulings, owed, authorization, freeze.json, dataset, stores / params / lexicons, briefings, weights, S*, "
      "scorer, C11, context fit, FS blocks, this module, planners/__init__.py (model selection; Astra r2 3)", all(A2.is_protected(x) for x in prot), [x for x in prot if not A2.is_protected(x)])
check("3.6 changeable (documented): executor, validator, exec contract, package code, contract.py (its prompt texts stay pinned through "
      "ft_v2.json), phases, treatment.py, acc_runner plumbing", not any(A2.is_protected(x) for x in free),
      [x for x in free if A2.is_protected(x)])
cs = A2.code_state()
check("3.7 the code state tracks identity files, frozen refs, plumbing, planners/__init__.py and the briefings (selftests excluded)",
      set(T.IDENTITY_FILES) <= set(cs) and set(REG["frozen_refs"]["files"]) <= set(cs) and "sim/planners/__init__.py" in cs
      and any(x.startswith("docs/final-test/kitchen_memory/briefings/") for x in cs) and not any(A2.is_selftest(x) for x in cs),
      sorted(set(T.IDENTITY_FILES) - set(cs))[:4])
cfg_ = copy.deepcopy(T.FT2)
T.FT2["pins"]["executor"]["sim/honest_skills.py"] = "0" * 64
pin_only = A2.ft2_core_sha()
T.FT2["pins"]["switches"]["honest_skills"]["IDENT_ON"] = False
switched = A2.ft2_core_sha()
T.FT2.clear()
T.FT2.update(cfg_)
check("3.8 FT2 core: an executor re-pin keeps it, a switch flip changes it", pin_only == A2.ft2_core_sha() and switched != A2.ft2_core_sha())

# ------------------------------------------------------------------------------------------------ the temporary study tree (stubbed sections)
COPY = sorted(set(["docs/final-test/acceptance_registration.json", "docs/final-test/acceptance_rulings.json", "sim/dataset/general.json",
                   "docs/final-test/inclusion_manifest_a26.json", "sim/planners/ft_v2.json"] + list(REG["frozen_refs"]["files"])
                  + list(A.OWED_CODE) + [A2.SELF_REL]))
LEDGER_TITLE = "ARASH DECISION: correction cycle authorized (selftest heading)"
GATE_NAMES = list(A2.REQUIRED_GATES)


def write_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def jl(p, rows, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def terminal_journals(td, ns, *, ok_keys=()):
    """A finished run's shape: capture_gates + execute phase_complete events, every registered planner cell terminal."""
    jl(C.journal_path(ns, "capture", td), [{"pres": "x"}])
    jl(C.journal_path(ns, "exec", td), [{"unit": "u"}])
    jl(C.journal_path(ns, "outcome", td), [{"unit": "u"}])
    rows = [{"key": C.cell_id(ns, "plan", a, p, "r0"), "status": "ok" if C.cell_id(ns, "plan", a, p, "r0") in ok_keys else "rejected"}
            for p in A.presentations(ROSTER) for a in C.MAIN_ARMS]
    jl(C.journal_path(ns, "plan", td), rows)
    jl(C.journal_path(ns, "events", td), [{"event": "capture_gates"}, {"event": "phase_complete", "phase": "execute"}], mode="a")


def restore_frozen(td):
    """A34-A2 + A35 (signed 2026-10-15): the real tree now carries re-frozen frozen refs (execute.py); a fixture is a PRE-correction tree, so a
    frozen ref that moved since the registration is restored to its registered bytes from the attempt-1 bundle (when that copy matches)."""
    _reg = json.load(open(os.path.join(td, "docs", "final-test", "acceptance_registration.json"), encoding="utf-8"))
    for rel_r, want in ((_reg.get("frozen_refs") or {}).get("files") or {}).items():
        d, b = os.path.join(td, *rel_r.split("/")), os.path.join(ROOT, "acceptance_bundles", A2.NS1, *rel_r.split("/"))
        if os.path.isfile(d) and A.file_sha(d) != want and os.path.isfile(b) and A.file_sha(b) == want:
            shutil.copyfile(b, d)


def make_root():
    td = tmpdir("acc_a2_")
    for rel in COPY:
        s, d = os.path.join(ROOT, *rel.split("/")), os.path.join(td, *rel.split("/"))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
    restore_frozen(td)
    with open(os.path.join(td, "docs", "final-test", "progress.md"), "w", encoding="utf-8") as f:
        f.write(f"# ledger\n\n## 2026-10-14 -- {LEDGER_TITLE}\n- go\n")
    for rel in ("sim/honest_skills.py", "sim/ltm_package.py"):
        q = os.path.join(td, *rel.split("/"))
        open(q, "w", encoding="utf-8").write("X = 1\n")
    for g in GATE_NAMES:
        open(os.path.join(td, "sim", g + ".py"), "w").write("# placeholder gate\n")
    os.makedirs(C.ns_work(A2.NS1, td), exist_ok=True)
    open(os.path.join(C.ns_work(A2.NS1, td), "tokcache.json"), "w").write("{}\n")
    terminal_journals(td, A2.NS1, ok_keys={C.cell_id(A2.NS1, "plan", "G0", A.presentations(ROSTER)[1], "r0")})
    return td


CELLS = {C.cell_id(A2.NS1, "plan", a, p, "r0"): (a, p) for p in A.presentations(ROSTER) for a in C.MAIN_ARMS}
G0_REJ = C.cell_id(A2.NS1, "plan", "G0", A.presentations(ROSTER)[0], "r0")
G0_OK = C.cell_id(A2.NS1, "plan", "G0", A.presentations(ROSTER)[1], "r0")
O0_REJ = C.cell_id(A2.NS1, "plan", "O0", A.presentations(ROSTER)[0], "r0")
DEC_FAIL = {"schema": A.SCHEMA + "/decision", "verdict": "NOT PASS", "attempt": 1,
            "criteria": {k: {"pass": k not in ("d", "f"), "reasons": (["capture canon:x:en: per-row G-iso not ok"] if k == "d" else
                                                                      ["G0: 44 / 48 valid plans (< 46)"] if k == "f" else []),
                             "n_reasons": 1 if k in ("d", "f") else 0} for k in ("completeness", "a", "b", "c", "d", "e", "f", "g", "h", "i", "c11")},
            "valid_plans_per_arm": {}, "listing_c": {}, "sr_descriptive_not_gated": {}}
PASS_DEC = dict(copy.deepcopy(DEC_FAIL), verdict="PASS", criteria={k: {"pass": True, "reasons": [], "n_reasons": 0} for k in DEC_FAIL["criteria"]})
STUB = {1: copy.deepcopy(DEC_FAIL), 2: dict(copy.deepcopy(DEC_FAIL), attempt=2)}
REPLAY_STUB = {1: None, 2: None}                         # None = replay returns the same decision as compute
REAL = (A2.compute_decision, A2.owed_state, A2.identity_parts, A2.REPLAY)
IDENT0 = {"sha": A.TREATMENT_SHA, "params_sha": "p" * 64, "briefing": {"records_sha": "b"}, "profile": "pkg-1.r6b", "mask": "universal"}
STUB_ID = {"v": dict(IDENT0)}
OWED = {"v": {"problems": [], "error": None}}


def stub_on():
    A2.compute_decision = lambda attempt, root=None: copy.deepcopy(STUB[attempt])
    A2.owed_state = lambda root=None: copy.deepcopy(OWED["v"])
    A2.identity_parts = lambda ns, root=None: dict(STUB_ID["v"])
    A2.REPLAY = lambda attempt, root=None: copy.deepcopy(REPLAY_STUB[attempt] if REPLAY_STUB[attempt] is not None else STUB[attempt])
    A2._REPLAY_MEMO.clear()


def stub_off():
    A2.compute_decision, A2.owed_state, A2.identity_parts, A2.REPLAY = REAL
    A2._REPLAY_MEMO.clear()


def rel(td, p):
    return os.path.relpath(p, td).replace(os.sep, "/")


def good_record(td, v1, *, files=None, defects=None):
    files = files or {}
    rec = {"schema": A2.CORRECTION_SCHEMA, "date": "2026-10-15", "written_by": "main session (selftest)",
           "attempt1_verdict": {"path": rel(td, A2.verdict_path(1, td)), "sha256": A.file_sha(A2.verdict_path(1, td))},
           "declarations": {k: True for k in A2.DECLARATIONS},
           "defects": defects or [{"category": "isolation_or_reset_fault", "criterion": "d", "attempt1_reasons": ["capture canon:x:en: per-row G-iso not ok"],
                                   "traced_to": [{"path": "sim/honest_skills.py", "sha256": v1["code_state"]["sim/honest_skills.py"], "what": "reset path"}],
                                   "description": "the reset left a package byte in a 0 arm"}],
           "remedy": {"description": "fix the reset", "files": files,
                      "diff": {"sha256": C.sha256_text(A2.make_diff(v1, td))}},
           "refrozen": {"treatment_identity": A.TREATMENT_SHA, "frozen_refs": {}}}
    rec["sha256"] = C.self_sha(rec)
    return rec


def save_record(td, rec):
    rec = dict(rec)
    rec["sha256"] = C.self_sha(rec)
    write_json(A2.correction_path(td), rec)
    return rec


def gates_for(td, rec):
    gdir = os.path.join(td, "gates")
    os.makedirs(gdir, exist_ok=True)
    gs = os.path.join(gdir, "summary.txt")
    open(gs, "w", encoding="utf-8").write("".join(f"{g} rc=0 :: ok\n" for g in A2.required_gates(td)) + "DONE\n")
    need = sorted(set(A.OWED_CODE) | {A2.SELF_REL} | set(rec["remedy"]["files"]) | set(A2.gate_sources(td)))
    code = {r_: (A.file_sha(os.path.join(td, *r_.split("/"))) if os.path.isfile(os.path.join(td, *r_.split("/"))) else "absent") for r_ in need}
    rec["gates"] = {"summary": {"path": rel(td, gs), "sha256": A.file_sha(gs)}, "code": code}
    return gs


def astra_for(td, rec, text="ACCEPT\n\nthe record is eligible\n"):
    ad = os.path.join(td, "docs", "final-test", "_astra")
    os.makedirs(ad, exist_ok=True)
    av = os.path.join(ad, "CORRECTION-r1.txt")
    rec["astra"] = {"verdict_file": rel(td, av)}
    open(av, "w", encoding="utf-8").write(text)
    write_json(av[:-4] + ".code.json", {"verdict_sha256": A.file_sha(av), "correction_core_sha": A2.correction_core_sha(rec)})
    return av


def full_chain(td, v1):
    """remedy honest_skills.py, the record, gates, Astra, Arash -> the saved record."""
    hs = os.path.join(td, "sim", "honest_skills.py")
    open(hs, "w", encoding="utf-8").write("X = 2  # fixed\n")
    now = A2.code_state(td)
    files = {"sim/honest_skills.py": {"before": v1["code_state"]["sim/honest_skills.py"], "after": now["sim/honest_skills.py"], "role": "remedy",
                                      "why": "the reset fix"}}
    rec = good_record(td, v1, files=files)
    rec["refrozen"]["treatment_identity"] = STUB_ID["v"]["sha"]
    gates_for(td, rec)
    astra_for(td, rec)
    rec = save_record(td, rec)
    A2.record_auth2(by="Arash", date="2026-10-15", said="go for the correction cycle", ledger_entry=LEDGER_TITLE, root=td)
    return rec


stub_on()
TDS = []
try:
    # ------------------------------------------------------------------------------------------------ 4 verdicts
    print("== 4 verdicts: terminal, locked, bundled, replayed, exclusive, closing")
    TD = make_root()
    TDS.append(TD)
    lk = os.path.join(C.ns_work(A2.NS1, TD), "plan.lock")
    os.makedirs(os.path.dirname(lk), exist_ok=True)
    open(lk, "w").write("{}")
    check("4.1 a running phase (plan.lock held) refuses the verdict", raises(lambda: A2.record_verdict(1, TD, write=True), A.Refused)
          and not os.path.exists(A2.verdict_path(1, TD)))
    os.remove(lk)
    evp = C.journal_path(A2.NS1, "events", TD)
    keep_ev = open(evp, "rb").read()
    open(evp, "w").write(json.dumps({"event": "capture_gates"}) + "\n")
    check("4.2 a run that is not terminal (no execute phase_complete) refuses the verdict", raises(lambda: A2.record_verdict(1, TD, write=True), A.Refused))
    open(evp, "wb").write(keep_ev)
    STUB[1] = dict(copy.deepcopy(DEC_FAIL), criteria=dict(DEC_FAIL["criteria"], g={"pass": False, "reasons": ["PENDING: x"], "n_reasons": 1}))
    check("4.3 a PENDING criterion refuses the verdict", raises(lambda: A2.record_verdict(1, TD, write=True), A.Refused))
    STUB[1] = copy.deepcopy(DEC_FAIL)
    REPLAY_STUB[1] = dict(copy.deepcopy(DEC_FAIL), verdict="PASS")
    check("4.4 a bundle replay that differs from the live decision refuses (and leaves no bundle)",
          raises(lambda: A2.record_verdict(1, TD, write=True), A.Refused) and not os.path.exists(A2.bundle_dir(1, TD)))
    REPLAY_STUB[1] = None
    orig_cd = A2.compute_decision

    def moving(attempt, root=None):
        jl(C.journal_path(A2.NS1, "exec", root), [{"late": 1}], mode="a")
        return copy.deepcopy(STUB[attempt])
    A2.compute_decision = moving
    check("4.5 evidence that moves while the verdict is computed refuses", raises(lambda: A2.record_verdict(1, TD, write=True), A.Refused)
          and not os.path.exists(A2.verdict_path(1, TD)))
    A2.compute_decision = orig_cd
    v1 = A2.record_verdict(1, TD, write=True)
    check("4.6 the attempt-1 verdict: written once with evidence, code state, rule fingerprint, ceilings, ledgers, bundle manifest",
          os.path.exists(A2.verdict_path(1, TD)) and v1["verdict"] == "NOT PASS" and v1["evidence"].get("plan.jsonl") and v1["rule_fingerprint"] == fp["sha"]
          and v1["ceilings"] == A2.ceilings_state() and v1["ledgers"]["1"]["exists"] is False and v1["bundle"]["n_files"] > 10
          and os.path.isfile(os.path.join(A2.bundle_dir(1, TD), "sim", "honest_skills.py")))
    check("4.7 a second verdict is refused (immutable, created exclusively)", raises(lambda: A2.record_verdict(1, TD, write=True), A.Refused))
    check("4.8 attempt 1 is CLOSED: preflight(attempt 1), arm_guard(attempt 1) and its lifecycle refuse",
          A2.attempt_closed(1, TD) and any("CLOSED" in x for x in A.preflight(TD, attempt=1, identity=False))
          and raises(lambda: A.arm_guard(TD, 1, path=os.path.join(TD, "x", "ledger.db"), record=False), A.Refused)
          and A2.attempt1_lifecycle(TD)(0.0) != [])
    check("4.9 the intact FAIL verdict is accepted (bundle + replay)", A2.attempt1_fail_problems(TD) == [], A2.attempt1_fail_problems(TD))
    A2._REPLAY_MEMO.clear()
    REPLAY_STUB[1] = dict(copy.deepcopy(DEC_FAIL), criteria=dict(DEC_FAIL["criteria"], a={"pass": False, "reasons": ["x"], "n_reasons": 1}))
    check("4.10 a replay that does not reproduce EVERY criterion (even an unquoted one) is refused (Astra r1 7)",
          any("does not reproduce" in x for x in A2.attempt1_fail_problems(TD)))
    REPLAY_STUB[1] = None
    A2._REPLAY_MEMO.clear()
    bf = os.path.join(A2.bundle_dir(1, TD), "sim", "honest_skills.py")
    keep_b = open(bf, "rb").read()
    open(bf, "w").write("X = 99\n")
    check("4.11 a bundle file changed after the verdict is refused", any("bundle files changed" in x for x in A2.attempt1_fail_problems(TD, replay=False)))
    open(bf, "wb").write(keep_b)
    vp = A2.verdict_path(1, TD)
    vd = json.load(open(vp, encoding="utf-8"))
    os.replace(vp, vp + ".orig")
    write_json(vp, dict(vd, verdict="PASS"))
    check("4.12 an edited verdict file is refused (self sha)", any("self-sha" in x for x in A2.attempt1_fail_problems(TD, replay=False)))
    os.remove(vp)
    os.replace(vp + ".orig", vp)
    jl(C.journal_path(A2.NS1, "exec", TD), [{"late": 1}], mode="a")
    check("4.13 an attempt-1 artefact changed after the verdict is refused", any("artefacts changed" in x for x in A2.attempt1_fail_problems(TD, replay=False)))
    lines = open(C.journal_path(A2.NS1, "exec", TD), encoding="utf-8").read().splitlines(keepends=True)[:-1]
    open(C.journal_path(A2.NS1, "exec", TD), "w", encoding="utf-8", newline="\n").write("".join(lines))
    check("4.14 restored -> intact again", A2.attempt1_fail_problems(TD) == [], A2.attempt1_fail_problems(TD))
    TD2 = make_root()
    TDS.append(TD2)
    STUB[1] = copy.deepcopy(PASS_DEC)
    A2.record_verdict(1, TD2, write=True)
    check("4.15 an attempt-1 PASS: no correction cycle", any("only after an attempt-1 FAIL" in x for x in A2.attempt1_fail_problems(TD2)))
    STUB[1] = copy.deepcopy(DEC_FAIL)

    # ------------------------------------------------------------------------------------------------ 5 eligibility record
    print("== 5 correction-eligibility record")
    rec0 = good_record(TD, v1)
    check("5.1 a complete record is eligible", A2.eligibility_record_problems(rec0, v1, TD) == [], A2.eligibility_record_problems(rec0, v1, TD))

    def mut(fn):
        r = copy.deepcopy(rec0)
        fn(r)
        r["sha256"] = C.self_sha(r)
        return A2.eligibility_record_problems(r, v1, TD)
    F_R = ["G0: 44 / 48 valid plans (< 46)"]
    cases = {
        "5.2 a category outside the predefined five": lambda r: r["defects"][0].update(category="efficacy"),
        "5.3 a criterion that passed in attempt 1": lambda r: r["defects"][0].update(criterion="a"),
        "5.4 a reason not quoted from the verdict": lambda r: r["defects"][0].update(attempt1_reasons=["made up"]),
        "5.5 no reasons": lambda r: r["defects"][0].update(attempt1_reasons=[]),
        "5.6 traced to a file at another sha": lambda r: r["defects"][0]["traced_to"][0].update(sha256="0" * 64),
        "5.7 traced to an untracked file": lambda r: r["defects"][0]["traced_to"][0].update(path="sim/nowhere.py"),
        "5.8 nothing traced": lambda r: r["defects"][0].update(traced_to=[]),
        "5.9 a missing declaration": lambda r: r["declarations"].pop("treatment_kept"),
        "5.10 a false declaration": lambda r: r["declarations"].update(acceptance_not_tuned=False),
        "5.11 not bound to the verdict": lambda r: r["attempt1_verdict"].update(sha256="0" * 64),
        "5.12 no defect": lambda r: r.update(defects=[]),
        "5.13 completeness alone (no (a)-(i) defect)": lambda r: r["defects"][0].update(criterion="completeness"),
        "5.14 an (f) defect without cells ((f) alone never authorizes)": lambda r: r["defects"][0].update(criterion="f", attempt1_reasons=F_R),
        "5.15 an (f) defect naming a VALID plan cell": lambda r: r["defects"][0].update(criterion="f", attempt1_reasons=F_R, cells=[G0_OK]),
        "5.16 an (f) defect naming an unregistered cell key": lambda r: r["defects"][0].update(criterion="f", attempt1_reasons=F_R, cells=["cellX"]),
        "5.17 an (f) defect naming a cell of an arm no quoted reason names": lambda r: r["defects"][0].update(criterion="f", attempt1_reasons=F_R, cells=[O0_REJ]),
        "5.18 no description": lambda r: r["defects"][0].update(description=" "),
    }
    for name, fn in cases.items():
        got = mut(fn)
        check(name + " -> refused", bool(got), got)
    ok_f = mut(lambda r: r["defects"][0].update(criterion="f", attempt1_reasons=F_R, cells=[G0_REJ]))
    check("5.19 an (f) defect naming rejected cells of the arm its quoted reason names is eligible", ok_f == [], ok_f)
    pj = C.journal_path(A2.NS1, "plan", TD)
    missing_key = C.cell_id(A2.NS1, "plan", "G0", A.presentations(ROSTER)[2], "r0")
    keep_pj = open(pj, "rb").read()
    rows_ = [json.loads(x) for x in keep_pj.decode().splitlines() if x.strip() and json.loads(x)["key"] != missing_key]
    # (the verdict's evidence binds plan.jsonl; this probe calls the record validator alone, then restores the journal)
    jl(pj, rows_)
    ok_m = mut(lambda r: r["defects"][0].update(criterion="f", attempt1_reasons=F_R, cells=[missing_key]))
    open(pj, "wb").write(keep_pj)
    check("5.20 an (f) defect naming a MISSING planner cell (registered, no row) is eligible (Astra r1 9)", ok_m == [], ok_m)
    bad_self = copy.deepcopy(rec0)
    bad_self["date"] = "2026-10-16"
    check("5.21 a record edited after writing (self sha) is refused", any("self-sha" in x for x in A2.eligibility_record_problems(bad_self, v1, TD)))
    check("5.22 no record at all is refused", A2.eligibility_record_problems(None, v1, TD) != [])

    # ------------------------------------------------------------------------------------------------ 6 code diff + re-freeze
    print("== 6 what may change, what may never change, the re-freeze")
    hs = os.path.join(TD, "sim", "honest_skills.py")
    open(hs, "w", encoding="utf-8").write("X = 2  # fixed\n")
    now = A2.code_state(TD)
    files = {"sim/honest_skills.py": {"before": v1["code_state"]["sim/honest_skills.py"], "after": now["sim/honest_skills.py"], "role": "remedy",
                                      "why": "the reset fix"}}
    rec = good_record(TD, v1, files=files)
    check("6.1 a documented remedy with the GENERATED diff sha passes", A2.code_diff_problems(rec, v1, TD) == [], A2.code_diff_problems(rec, v1, TD))
    check("6.1b the generated diff is a real unified diff of the bundle bytes -> disk", "-X = 1" in A2.make_diff(v1, TD) and "+X = 2  # fixed" in A2.make_diff(v1, TD))
    r_d = copy.deepcopy(rec)
    r_d["remedy"]["diff"]["sha256"] = C.sha256_text("--- a/sim/honest_skills.py\n+++ b/sim/honest_skills.py\n")
    check("6.2 a hand-written diff (filenames only) is refused (Astra r1 10)", any("remedy.diff" in x for x in A2.code_diff_problems(r_d, v1, TD)))
    r_none = good_record(TD, v1)
    check("6.3 a change the record does not list is refused (undocumented)", any("undocumented change" in x for x in A2.code_diff_problems(r_none, v1, TD)))
    r_bad = copy.deepcopy(rec)
    r_bad["remedy"]["files"]["sim/honest_skills.py"]["before"] = "1" * 64
    check("6.4 wrong before / after shas are refused", any("before / after" in x for x in A2.code_diff_problems(r_bad, v1, TD)))
    r_pad = copy.deepcopy(rec)
    r_pad["remedy"]["files"]["sim/ltm_package.py"] = {"before": "x", "after": "x", "role": "remedy", "why": "padding"}
    check("6.5 a listed file that did not change is refused (no padding)", any("did not change" in x for x in A2.code_diff_problems(r_pad, v1, TD)))
    r_role = copy.deepcopy(rec)
    r_role["remedy"]["files"]["sim/honest_skills.py"]["role"] = "tuning"
    check("6.6 a remedy entry without a registered role / why is refused", any("role" in x for x in A2.code_diff_problems(r_role, v1, TD)))
    for prel in ("sim/step5/package_params.json", "docs/final-test/acceptance_registration.json", A2.SELF_REL, "docs/final-test/freeze.json"):
        q = os.path.join(TD, *prel.split("/"))
        old = open(q, "rb").read() if os.path.exists(q) else None
        os.makedirs(os.path.dirname(q), exist_ok=True)
        open(q, "ab").write(b"\n")
        n_ = A2.code_state(TD)
        rp = copy.deepcopy(rec)
        rp["remedy"]["files"][prel] = {"before": v1["code_state"].get(prel, "absent"), "after": n_[prel], "role": "remedy", "why": "x"}
        rp["remedy"]["diff"]["sha256"] = C.sha256_text(A2.make_diff(v1, TD, now=n_))
        got = A2.code_diff_problems(rp, v1, TD, now=n_)
        check(f"6.7 a documented change of the PROTECTED {prel} is still refused", any("PROTECTED" in x for x in got), got)
        if old is None:
            os.remove(q)
        else:
            open(q, "wb").write(old)
    check("6.8 a decision-rule fingerprint change since attempt 1 is refused", any("fingerprint" in x for x in A2.code_diff_problems(rec, dict(v1, rule_fingerprint="0"), TD)))
    check("6.9 an FT2 switch change is refused", any("FT2" in x for x in A2.code_diff_problems(rec, dict(v1, ft2_core_sha="0"), TD)))
    check("6.10 an ft_v2.json change outside its contract sha is refused", any("ft_v2.json" in x for x in A2.code_diff_problems(rec, dict(v1, ft_v2_json_core_sha="0"), TD)))
    check("6.11 changed namespace ceilings are refused", any("ceilings" in x for x in A2.code_diff_problems(rec, dict(v1, ceilings={}), TD)))
    check("6.12 a record without a remedy file is refused",
          any("no remedy file" in x for x in A2.code_diff_problems(good_record(TD, v1), dict(v1, code_state=A2.code_state(TD)), TD)))
    # ---- 6.20 the SIGNED A34-A2 + A35 exception (Arash 2026-10-15), pinned to immutable artefacts (Astra ACC2-IMPL-r1 1)
    SA = A2.SIGNED_AMENDMENT
    fpx = lambda xs: [x for x in xs if "fingerprint" in x or "signed amendment" in x]
    v1_old = dict(v1, rule_fingerprint=SA["rule_fingerprint_old"])
    _p20a = p_620a()
    check("6.20a the pinned new fingerprint IS the fingerprint of the attempt-2 code (authenticated: replay-verified verdict + its bundle) and of "
          "the code on disk -- or the disk moved ONLY through the A36 successor branch (replay-verified terminal NOT PASS, A36 + A36-IMPL, complete "
          "changed-set / succession) and only in approved A36 files", _p20a[0], _p20a[1])
    check("6.20b the pinned self core sha IS this module's", A2.self_core_sha() == SA["self_core_sha256"])
    check("6.20c amendment_block() is the pinned constants (no disk read)", A2.amendment_block(TD) == A2.amendment_block()
          and A2.amendment_block()["pins"] == SA["pins"])
    check("6.20d without the amendment block the old -> new move is refused", fpx(A2.code_diff_problems(rec, v1_old, TD)) != [])
    r_am = copy.deepcopy(rec)
    r_am["amendment"] = A2.amendment_block()
    check("6.20e on a tree that is not the signed one (the fixture), the amendment block does NOT open the guard",
          any("signed amendment" in x for x in A2.code_diff_problems(r_am, v1_old, TD)))
    for prel in (A2.SELF_REL, "docs/final-test/freeze.json", "sim/step5/package_params.json"):
        q = os.path.join(TD, *prel.split("/"))
        old = open(q, "rb").read() if os.path.exists(q) else None
        os.makedirs(os.path.dirname(q), exist_ok=True)
        open(q, "ab").write(b"\n")
        n_ = A2.code_state(TD)
        rp = copy.deepcopy(r_am)
        rp["remedy"]["files"][prel] = {"before": v1["code_state"].get(prel, "absent"), "after": n_[prel], "role": "remedy", "why": "x"}
        rp["remedy"]["diff"]["sha256"] = C.sha256_text(A2.make_diff(v1, TD, now=n_))
        got = [x for x in A2.code_diff_problems(rp, v1_old, TD, now=n_) if "PROTECTED" in x]
        check(f"6.20f an unsigned change of the PROTECTED {prel} stays refused even with the amendment block", got != [], got)
        if old is None:
            os.remove(q)
        else:
            open(q, "wb").write(old)
    # the REAL tree (the signed state; meaningful once the correction record is written)
    rec_real = A2._json(A2.correction_path()) or {}
    v1_real = A2._json(A2.verdict_path(1)) or {}
    if rec_real.get("amendment") is not None:
        now_r = A2.code_state()
        _p20g = p_620g(rec_real, v1_real)
        check("6.20g REAL tree: the signed exception applies (pins, core sha, signature, freeze.json entry) -- or every remaining problem is a "
              "tree-state one and the A36 successor branch holds (no record on disk = refused)", _p20g[0], _p20g[1])
        _impl_disk = os.path.exists(os.path.join(ROOT, *FT.A36_IMPL_FILE.split("/")))
        if not _impl_disk:
            check("6.20s1 no A36 implementation record on disk: the successor branch refuses (6.20a / 6.20g stay red until Arash's record exists)",
                  not a36_successor()[0] and not p_620g(rec_real, v1_real)[0]
                  and (not p_620a()[0] or A2.rule_fingerprint()["sha"] == SA["rule_fingerprint_new"]), a36_successor()[1][:3])
        _TF = tempfile.mkdtemp(prefix="acc2impl_", dir=os.path.join(ROOT, "sim", "final_test", "_checkpoints"))
        _verp = A2.verdict_problems
        try:
            _sig_t = impl_fixture_on(_TF)
            _s_ok = a36_successor(_sig_t)
            _a_ok, _g_ok = p_620a(_sig_t), p_620g(rec_real, v1_real, _sig_t)
            check("6.20s2 with a temporary A36-IMPL record + approval: the successor branch holds and 6.20a / 6.20g pass",
                  _s_ok[0] and _a_ok[0] and _g_ok[0], (_s_ok[1][:3], _a_ok[1], _g_ok[1]))
            _no_appr = {k: v for k, v in _sig_t.items() if k != FT.A36_IMPL_KEY}
            _no_a36 = {k: v for k, v in _sig_t.items() if k != "A36"}
            _bad_appr = dict(_sig_t, **{FT.A36_IMPL_KEY: dict(_sig_t[FT.A36_IMPL_KEY], file_sha256="0" * 64)})
            A2.verdict_problems = lambda k, root=None, replay=True, _o=_verp: (["replay differs (fixture)"] if k == 2 else _o(k, root, replay=replay))
            _rep_bad = a36_successor(_sig_t)[0]
            A2.verdict_problems = _verp
            check("6.20s3 the successor branch refuses: record without approval, a mismatched approval, no A36 signature, an attempt-2 verdict "
                  "that does not replay (file existence alone is not enough)",
                  not a36_successor(_no_appr)[0] and not a36_successor(_bad_appr)[0] and not a36_successor(_no_a36)[0] and not _rep_bad
                  and not p_620g(rec_real, v1_real, _no_appr)[0])
        finally:
            A2.verdict_problems = _verp
            impl_fixture_off()
            shutil.rmtree(_TF, ignore_errors=True)
        bad_blocks = {"fingerprint": dict(rec_real["amendment"], rule_fingerprint_new="0" * 64),
                      "pins": dict(rec_real["amendment"], pins=dict(rec_real["amendment"]["pins"], **{SA["reset_gate"]: "0" * 64})),
                      "self core": dict(rec_real["amendment"], self_core_sha256="0" * 64),
                      "protected set": dict(rec_real["amendment"], protected_changed=[A2.SELF_REL, "sim/step5/package_params.json"])}
        for nm, blk in bad_blocks.items():
            check(f"6.20h REAL tree: a forged amendment block ({nm}) is refused", A2.amendment_problems(dict(rec_real, amendment=blk), v1_real) != [])
        check("6.20i REAL tree: another attempt-1 fingerprint is refused", A2.amendment_problems(rec_real, dict(v1_real, rule_fingerprint="1" * 64)) != [])
        n2 = dict(now_r, **{SA["reset_gate"]: "0" * 64})
        check("6.20j REAL tree: a pinned file at another sha is refused", any("pinned sha" in x for x in A2.amendment_problems(rec_real, v1_real, now=n2)))
        n3 = dict(now_r, **{"sim/honest_skills.py": "0" * 64})
        check("6.20k REAL tree: a change outside the signed file set is refused", any("outside the signed set" in x
                                                                                    for x in A2.amendment_problems(rec_real, v1_real, now=n3)))
        _fz = A2._json(A2._abs(SA["freeze_json"]))
        _real_json = A2._json
        A2._json = lambda p_, _o=_real_json: (dict(_fz, extra_key=1, sha256=C.self_sha(dict(_fz, extra_key=1))) if str(p_).replace("\\", "/").endswith(
            "docs/final-test/freeze.json") and "acceptance_bundles" not in str(p_) else _o(p_))
        check("6.20l REAL tree: freeze.json changed beyond the amendment entry is refused", any("beyond" in x for x in A2.freeze_json_problems(rec_real)))
        A2._json = _real_json
        check("6.20m REAL tree: the dev validation is bound, reproduced now, after the freeze, never failed",
              A2.a34_validation_problems(rec_real) == [] or not rec_real.get("dev_validation"), A2.a34_validation_problems(rec_real))
        check("6.20n a record without a dev validation is refused by the rule-4 check", A2.a34_validation_problems(dict(rec_real, dev_validation=None)) != [])
        _ow = [x for x in A2.owed_problems2(v1_real) if x.startswith("owed item 4")]
        check("6.20o REAL tree: owed item 4's stale whole-file bindings of the SIGNED rule files are excused, nothing else", _ow == [], _ow)
        _m = "owed item 4 (a24_frozen_first): frozen-first 'x': binding sim/final_test/matrix/score.py (file) is stale (recorded 0, now abc)"
        check("6.20p an owed item-4 binding of a file the amendment does not pin still refuses", not A2._owed4_signed_rebind(_m))
        _m2 = "owed item 4 (a24_frozen_first): frozen-first 'x': binding sim/final_test/acceptance_v2/acc_runner.py (file) is stale (recorded 0, now 000000000000)"
        check("6.20q ...and one whose current sha is not the reported / pinned one refuses", not A2._owed4_signed_rebind(_m2))
    else:
        print("NOTE 6.20g-n: the real correction record does not carry the amendment yet (written by a34_record.py core)")
    rec_id = copy.deepcopy(rec)
    check("6.13 nothing re-pinned and no frozen ref moved: the identity stays the registration's pin", A2.refreeze_problems(rec, v1, TD) == [],
          A2.refreeze_problems(rec, v1, TD))
    STUB_ID["v"] = dict(IDENT0, sha="9" * 64)
    check("6.14 a refrozen identity that differs from the computed one is refused", any("treatment_identity" in x for x in A2.refreeze_problems(rec, v1, TD)))
    rec_id["refrozen"]["treatment_identity"] = "9" * 64
    check("6.15 a re-pinned identity file (honest_skills.py) with the new identity recorded passes", A2.refreeze_problems(rec_id, v1, TD) == [],
          A2.refreeze_problems(rec_id, v1, TD))
    rec_nf = copy.deepcopy(rec_id)
    rec_nf["remedy"]["files"] = {"sim/final_test/matrix/other.py": {"before": "a", "after": "b", "role": "remedy", "why": "x"}}
    check("6.16 a new identity although no identity file changed is refused", any("registration's pin" in x for x in A2.refreeze_problems(rec_nf, v1, TD)))
    STUB_ID["v"] = dict(IDENT0, sha="9" * 64, params_sha="q" * 64)
    check("6.17 changed package parameters are refused even with a recorded identity", any("params_sha" in x for x in A2.refreeze_problems(rec_id, v1, TD)))
    STUB_ID["v"] = dict(IDENT0)
    fr0 = sorted(REG["frozen_refs"]["files"])[0]
    q = os.path.join(TD, *fr0.split("/"))
    keep = open(q, "rb").read()
    open(q, "ab").write(b"\n")
    check("6.18 a moved frozen ref not listed in refrozen.frozen_refs is refused", any("frozen_refs" in x for x in A2.refreeze_problems(rec, v1, TD)))
    rec_fr = copy.deepcopy(rec)
    rec_fr["refrozen"]["frozen_refs"] = {fr0: A.file_sha(q)}
    rec_fr["remedy"]["files"][fr0] = {"before": v1["code_state"][fr0], "after": A.file_sha(q), "role": "remedy", "why": "x"}
    check("6.19 a moved frozen ref listed with its new sha (and as a remedy file) passes the re-freeze", A2.refreeze_problems(rec_fr, v1, TD) == [],
          A2.refreeze_problems(rec_fr, v1, TD))
    open(q, "wb").write(keep)
    ds = os.path.join(TD, "sim", "dataset", "general.json")
    keep = open(ds, "rb").read()
    open(ds, "ab").write(b" ")
    check("6.20 a dataset change is refused by the pins", any("pin dataset" in x for x in A2.refreeze_problems(rec, v1, TD)))
    open(ds, "wb").write(keep)

    # ------------------------------------------------------------------------------------------------ 7 gates, Astra, Arash
    print("== 7 gates (every offline gate) on the remedied code, Astra on the record, Arash's attempt-2 authorization")
    import glob as _glob
    tree_gates = {os.path.splitext(os.path.basename(x))[0] for x in _glob.glob(os.path.join(HERE, "_*selftest.py"))}
    check("7.0 the required gate set is the FROZEN manifest (independent of the tree): every offline selftest of the repo, minus the "
          "declared non-offline one (Astra r2 10) and the two gates the signed A36 implementation added (accounted: new since the base commit, "
          "in the A36 code scope); no required gate missing, no unexplained addition",
          A2.required_gates(TD) == A2.required_gates() == sorted(A2.REQUIRED_GATES) and gate_set_problems(tree_gates) == []
          and {"_acc_runner_selftest", "_acc_attempt2_selftest", "_console_honest_selftest", "_search_selftest", "_step4_act_selftest"}
          <= set(A2.REQUIRED_GATES), gate_set_problems(tree_gates))
    check("7.0b a missing required gate, an unexplained added gate, or a missing accounted A36 gate is refused",
          gate_set_problems(tree_gates - {"_acc_runner_selftest"}) != [] and gate_set_problems(tree_gates | {"_zz_new_selftest"}) != []
          and gate_set_problems(tree_gates - {"_ftm_a36_selftest"}) != [])
    gs = gates_for(TD, rec)
    check("7.1 green gates (all of them) on the current remedied code pass", A2.gates_problems(rec, TD) == [], A2.gates_problems(rec, TD))
    open(gs, "w", encoding="utf-8").write("".join(f"{g} rc={1 if g == '_acc_attempt2_selftest' else 0} :: x\n" for g in GATE_NAMES) + "DONE\n")
    r_g = copy.deepcopy(rec)
    r_g["gates"]["summary"]["sha256"] = A.file_sha(gs)
    check("7.2 a red gate is refused", any("not green" in x for x in A2.gates_problems(r_g, TD)))
    open(gs, "w", encoding="utf-8").write("".join(f"{g} rc=0 :: x\n" for g in GATE_NAMES[1:]) + "\n")
    r_g["gates"]["summary"]["sha256"] = A.file_sha(gs)
    got = A2.gates_problems(r_g, TD)
    check("7.3 a truncated summary / a missing gate is refused", any("truncated" in x for x in got) and any("lacks" in x for x in got), got)
    gs = gates_for(TD, rec)
    r_c = copy.deepcopy(rec)
    r_c["gates"]["code"]["sim/honest_skills.py"] = "0" * 64
    check("7.4 code changed after the gates ran is refused", any("changed since the gates ran" in x for x in A2.gates_problems(r_c, TD)))
    av = astra_for(TD, rec)
    check("7.5 Astra's plain ACCEPT bound to the record core passes", A2.astra_problems(rec, TD) == [], A2.astra_problems(rec, TD))
    r_a = copy.deepcopy(rec)
    r_a["defects"][0]["description"] = "changed after the review"
    check("7.6 a record changed after Astra's review is refused", any("not bound" in x for x in A2.astra_problems(r_a, TD)))
    astra_for(TD, rec, "ACCEPT WITH CHANGES\n")
    check("7.7 ACCEPT WITH CHANGES does not authorize", any("not ACCEPT" in x for x in A2.astra_problems(rec, TD)))
    astra_for(TD, rec)
    rec = save_record(TD, rec)
    check("7.8 no attempt-2 authorization yet -> refused", any("no attempt-2 authorization" in x for x in A2.auth2_problems(rec, TD)))
    check("7.9 only Arash authorizes attempt 2", raises(lambda: A2.record_auth2(by="Claude", date="d", said="s", ledger_entry=LEDGER_TITLE, root=TD), PermissionError))
    check("7.10 the authorization needs an 'ARASH ...' ledger heading",
          raises(lambda: A2.record_auth2(by="Arash", date="d", said="s", ledger_entry="nothing like it at all here", root=TD), A.Refused))
    A2.record_auth2(by="Arash", date="2026-10-15", said="go for the correction cycle", ledger_entry=LEDGER_TITLE, root=TD)
    check("7.11 Arash's attempt-2 authorization bound to the record passes", A2.auth2_problems(rec, TD) == [], A2.auth2_problems(rec, TD))
    check("7.12 an authorization bound to another record is refused", any("another correction record" in x for x in A2.auth2_problems(dict(rec, sha256="0"), TD)))

    # ------------------------------------------------------------------------------------------------ 8 preflight2 + bindings
    print("== 8 preflight2, owed items, bindings")
    pf = A2.preflight2(TD)
    check("8.1 every condition met -> preflight2 is clean", pf == [], pf)
    check("8.2 acc_runner.preflight(attempt=2) delegates to it", A.preflight(TD, attempt=2) == [])
    check("8.3 stage order on the a2 journals: Gemma needs the a2 capture", any("capture is not complete" in x for x in A2.preflight2(TD, stage="gemma")))
    check("8.4 the Opus stage needs a2's own fake over-cap event", any("acceptance-v2-a2" in x and "fake over-cap" in x for x in A2.preflight2(TD, stage="opus")))
    OWED["v"] = {"problems": [], "error": "KeyError: 'kind'"}
    check("8.5 an UNREADABLE owed state refuses (never filtered away; Astra r1 4)", any("unreadable" in x for x in A2.preflight2(TD)))
    OWED["v"] = {"problems": ["owed item 4 not recorded as done: A24.1"], "error": None}
    check("8.6 owed item 4 is still required", any("owed item 4" in x for x in A2.preflight2(TD)))
    OWED["v"] = {"problems": ["a global owed problem without an item number"], "error": None}
    check("8.7 an owed problem that names no item refuses", any("global owed problem" in x for x in A2.preflight2(TD)))
    OWED["v"] = {"problems": ["owed item 0 (plumbing_gates): x", "owed item 2 (reserved_live_check): y", "owed item 6 (arash_go): z"], "error": None}
    check("8.8 items 0 / 6 (replaced) and 1-3 (inherited from the verdict) do not refuse now", A2.preflight2(TD) == [], A2.preflight2(TD))
    OWED["v"] = {"problems": [], "error": None}
    check("8.9 a verdict whose owed state was not clean refuses", any("clean owed state" in x for x in
          A2.owed_problems2(dict(v1, owed_at_verdict={"problems": ["owed item 1 (roster_sealed): x"], "error": None}), TD)))
    A2.on_enter(TD)
    se = A2.start_event(TD)
    check("8.10 the first entry journals a2_start binding record, verdict, authorization and the code state (once)",
          se and se["correction_sha"] == rec["sha256"] and se["attempt1_verdict_sha"] == A.file_sha(A2.verdict_path(1, TD))
          and se["authorization_sha"] == json.load(open(A2.auth2_path(TD)))["sha256"] and se["code_state_sha"] == C.sha256_json(A2.code_state(TD)))
    A2.on_enter(TD)
    check("8.11 a2_start is written once; the authorization is never re-issued after it",
          sum(1 for r_ in A._rows(C.journal_path(A2.NS2, "events", TD)) if r_.get("event") == "a2_start") == 1
          and raises(lambda: A2.record_auth2(by="Arash", date="d", said="s", ledger_entry=LEDGER_TITLE, root=TD), A.Refused))
    keep_rec = open(A2.correction_path(TD), "rb").read()
    r_late = save_record(TD, dict(rec, written_by="edited after the start"))
    check("8.12 a record changed after the re-run started is refused (bindings, every guard call)",
          any("after the re-run started" in x for x in A2.bindings_problems(TD)) and any("after the re-run started" in x for x in A2.preflight2(TD)))
    open(A2.correction_path(TD), "wb").write(keep_rec)
    q = os.path.join(TD, "sim", "ltm_package.py")
    keep = open(q, "rb").read()
    open(q, "w").write("X = 3\n")
    check("8.13 code changed after the re-run started is refused (bindings)", any("code / data changed after the re-run started" in x for x in A2.bindings_problems(TD)))
    open(q, "wb").write(keep)
    check("8.14 restored -> bindings clean", A2.bindings_problems(TD) == [], A2.bindings_problems(TD))
    ev_p = C.journal_path(A2.NS2, "events", TD)
    keep_ev = open(ev_p, "rb").read()
    open(ev_p, "w").write("")
    jl(C.journal_path(A2.NS2, "capture", TD), [{"pres": "x"}])
    check("8.15 a2 rows without the a2_start event are refused", any("without the a2_start" in x for x in A2.preflight2(TD)))
    open(ev_p, "wb").write(keep_ev)
    os.remove(C.journal_path(A2.NS2, "capture", TD))

    # ------------------------------------------------------------------------------------------------ 9 spend
    print("== 9 attempt-2 spend: own ledger, combined 16, lifecycle on every submission")
    TD3 = make_root()
    TDS.append(TD3)
    LOCK = os.path.join(TD3, "study.lock")
    zero = {"settled": 0.0, "open": 0.0, "holds": 0.0, "liability": 0.0, "committed": 0.0, "ledgers": {}, "errors": []}
    p1 = A2.ledger_path(1, TD3)
    os.makedirs(os.path.dirname(p1), exist_ok=True)
    AC.register_ledger_guard(p1, attempt_base_usd=0.0, attempt_cap_usd=100.0, other_spend=lambda: dict(zero), lock_path=LOCK,
                             lifecycle=A2.attempt1_lifecycle(TD3))
    l1 = C.ScopedLedger(p1, C.NAMESPACES[A2.NS1]["ceilings"])
    with l1.use("core"):
        rid = l1.reserve("anthropic", 9.5)
    check("9.1 an attempt-1 ledger with an OPEN reservation blocks the verdict (not terminal)", raises(lambda: A2.record_verdict(1, TD3, write=True), A.Refused))
    l1.settle(rid, 9.5, {"provider": "anthropic"})
    v1b = A2.record_verdict(1, TD3, write=True)
    with l1.use("core"):
        a1_after = raises(lambda: l1.reserve("anthropic", 0.01), C.CostCeilingExceeded)
    check("9.2 a guard armed BEFORE attempt 1's verdict refuses every later attempt-1 submission (lifecycle; Astra r1 2)", a1_after)
    AC.unregister_ledger_guard(p1)
    rec3 = full_chain(TD3, v1b)
    check("9.3 the full chain on this tree is clean", A2.preflight2(TD3) == [], A2.preflight2(TD3))
    check("9.4 effective cap = min(8, 16 - 9.5) = 6.5", abs(A2.effective_cap(TD3) - 6.5) < 1e-9, A2.effective_cap(TD3))
    g2 = A.arm_guard(TD3, 2, other_spend=lambda: dict(zero))
    g2["lock"] = LOCK
    st2 = A2.attempt_start2(TD3)
    check("9.5 arm_guard(attempt 2): a2's own ledger, cap 6.5, acc_attempt_start in a2 (not a1)",
          abs(g2["attempt_cap"] - 6.5) < 1e-9 and st2 and st2["attempt1_liability"] == 9.5
          and not any(r_.get("event") == "acc_attempt_start" for r_ in A._rows(C.journal_path(A2.NS1, "events", TD3)) or []))
    A2.on_enter(TD3)
    l2 = C.ScopedLedger(A2.ledger_path(2, TD3), C.NAMESPACES[A2.NS2]["ceilings"])
    with l2.use("core"):
        rid2 = l2.reserve("anthropic", 6.4)
        ok_ = rid2 > 0
        over = raises(lambda: l2.reserve("anthropic", 0.11), C.CostCeilingExceeded)
    l2.settle(rid2, 6.4, {"provider": "anthropic"})
    check("9.6 6.4 USD fits, the next 0.11 crosses the combined 16 USD acceptance total", ok_ and over)
    keep_rec = open(A2.correction_path(TD3), "rb").read()
    save_record(TD3, dict(rec3, written_by="edited mid-run"))
    with l2.use("core"):
        rec_edit = raises(lambda: l2.reserve("anthropic", 0.01), C.CostCeilingExceeded)
    open(A2.correction_path(TD3), "wb").write(keep_rec)
    check("9.7 an ALREADY ARMED attempt-2 guard refuses the next submission once the record changes (Astra r1 2)", rec_edit)
    c_ = sqlite3.connect(p1)
    c_.execute("INSERT INTO ledger (ts, provider, state, reserved, actual) VALUES ('t', 'anthropic@core', 'settled', 0.0, 0.3)")
    c_.commit()
    c_.close()
    with l2.use("core"):
        late = raises(lambda: l2.reserve("anthropic", 0.01), C.CostCeilingExceeded)
    check("9.8 a later attempt-1 settlement refuses the next attempt-2 submission (attempt 1's ledger is bound at its verdict)", late)
    c_ = sqlite3.connect(p1)
    c_.execute("DELETE FROM ledger WHERE ts = 't'")
    c_.commit()
    c_.close()
    fo = AC.fake_overcap_test(l2)
    check("9.9 the fake over-cap test refuses on the a2 ledger and leaves nothing open", fo["ok"] and fo["refused"], fo)
    g2b = A.arm_guard(TD3, 2, other_spend=lambda: dict(zero))
    check("9.10 on restart the recorded base and cap are restored (one start event)",
          abs(g2b["attempt_cap"] - 6.5) < 1e-9 and sum(1 for r_ in A._rows(C.journal_path(A2.NS2, "events", TD3)) if r_.get("event") == "acc_attempt_start") == 1)
    AC.unregister_ledger_guard(A2.ledger_path(2, TD3))
    l2b = C.ScopedLedger(A2.ledger_path(2, TD3), C.NAMESPACES[A2.NS2]["ceilings"], release_stale=False)
    with l2b.use("core"):
        check("9.11 the a2 ledger without its guard refuses a paid reservation (fail closed)", raises(lambda: l2b.reserve("anthropic", 0.01), C.CostCeilingExceeded))

    # ------------------------------------------------------------------------------------------------ 10 phases
    print("== 10 the phases run in acceptance-v2-a2")
    try:
        pl = A.make_plan("gemma", TD3, attempt=2)
        check("10.1 make_plan(attempt 2): namespace a2, a2 plan journal, ledger, cache", pl.ns == A2.NS2 and pl.attempt == 2
              and pl.J.path == C.journal_path(A2.NS2, "plan", TD3) and pl._ledger_path == A2.ledger_path(2, TD3)
              and pl._cache_dir.startswith(C.ns_work(A2.NS2, TD3)))
        po = A.make_plan("opus", TD3, attempt=2)
        check("10.2 make_plan(opus, attempt 2) holds at most the effective cap", abs(po.batch_hold_usd - 6.5) < 1e-9, po.batch_hold_usd)
    except Exception as e:                                         # noqa: BLE001
        check("10.1 make_plan(attempt 2) builds", False, f"{type(e).__name__}: {e}")
    ph = A.make_execute(TD3, dry_run=True, attempt=2)
    check("10.3 make_execute(attempt 2): namespace a2", ph.ns == A2.NS2 and ph.attempt == 2 and ph.docs == C.ns_docs(A2.NS2, TD3))
    check("10.4 stage_prereq_problems / fake_overcap_recorded read the attempt's namespace",
          any("capture" in x for x in A.stage_prereq_problems("gemma", ROSTER, TD3, ns=A2.NS2)) and A.fake_overcap_recorded(TD3, ns=A2.NS2) is False)

    # ------------------------------------------------------------------------------------------------ 11 the attempt-2 verdict, both attempts
    print("== 11 attempt-2 verdict (full validator), both attempts reported")
    check("11.1 the attempt-2 verdict refuses before attempt 2 is terminal", raises(lambda: A2.record_verdict(2, TD3, write=True), A.Refused))
    terminal_journals(TD3, A2.NS2)
    keep_rec = open(A2.correction_path(TD3), "rb").read()
    save_record(TD3, dict(rec3, written_by="edited after execution"))
    check("11.2 a record edited after execution refuses the attempt-2 verdict (Astra r1 3)", raises(lambda: A2.record_verdict(2, TD3, write=True), A.Refused)
          and not os.path.exists(A2.verdict_path(2, TD3)))
    open(A2.correction_path(TD3), "wb").write(keep_rec)
    STUB[2] = dict(copy.deepcopy(DEC_FAIL), attempt=2)
    v2 = A2.record_verdict(2, TD3, write=True)
    b = A2.both_attempts(TD3)
    check("11.3 attempt-2 verdict NOT PASS: protocol stopped; both reported; STOPPED",
          v2["protocol_stopped"] is True and v2["correction_sha"] == rec3["sha256"] and b["acceptance"].startswith("STOPPED")
          and b["attempts"]["1"]["intact"] and b["attempts"]["2"]["intact"], b)
    check("11.4 after it: attempt 2 closed (bindings refuse), no attempt 3", any("CLOSED" in x for x in A2.bindings_problems(TD3)) and raises(lambda: A2.ns_of(3), ValueError))
    os.remove(A2.verdict_path(2, TD3))
    shutil.rmtree(A2.bundle_dir(2, TD3))
    STUB[2] = dict(copy.deepcopy(PASS_DEC), attempt=2)
    A2.record_verdict(2, TD3, write=True)
    check("11.5 an attempt-2 PASS is reported as PASS after the one correction cycle, attempt 1 kept", A2.both_attempts(TD3)["acceptance"].startswith("PASS (attempt 2")
          and A2.both_attempts(TD3)["attempts"]["1"]["verdict"] == "NOT PASS")
    vp2 = A2.verdict_path(2, TD3)
    vd2 = json.load(open(vp2, encoding="utf-8"))
    os.replace(vp2, vp2 + ".orig")
    write_json(vp2, dict(vd2, note="tampered"))
    check("11.6 a tampered PASS verdict is reported INVALID, never PASS (Astra r1 3)", A2.both_attempts(TD3)["acceptance"].startswith("INVALID"))
    os.remove(vp2)
    os.replace(vp2 + ".orig", vp2)

    # ------------------------------------------------------------------------------------------------ 11b Astra r2 probes (stubbed decide)
    print("== 11b Astra r2 probes: forged decisions, free calls, terminal states, finalization locks, history, gates, diffs")
    os.remove(A2.verdict_path(2, TD3))
    shutil.rmtree(A2.bundle_dir(2, TD3))
    STUB[2] = dict(copy.deepcopy(DEC_FAIL), attempt=2)
    A2._REPLAY_MEMO.clear()
    v2n = A2.record_verdict(2, TD3, write=True)
    forged = dict(copy.deepcopy(v2n), verdict="PASS")
    forged["decision"]["verdict"] = "PASS"
    forged.pop("sha256")
    forged["sha256"] = C.self_sha(forged)
    keep_v2 = open(vp2, "rb").read()                      # kept in memory: a backup inside the docs dir would itself be evidence
    write_json(vp2, forged)
    A2._REPLAY_MEMO.clear()
    b_f = A2.both_attempts(TD3)
    check("11b.1 an attempt-2 NOT PASS rewritten as PASS with a RECOMPUTED self sha is INVALID (its bundle replay disagrees; Astra r2 1)",
          b_f["acceptance"].startswith("INVALID") and any("does not reproduce" in x for x in b_f["attempts"]["2"]["problems"]), b_f)
    desc = copy.deepcopy(v2n)
    desc["decision"]["sr_descriptive_not_gated"] = {"G0": 0.99}
    desc.pop("sha256")
    desc["sha256"] = C.self_sha(desc)
    write_json(vp2, desc)
    A2._REPLAY_MEMO.clear()
    check("11b.2 a changed DESCRIPTIVE field (SR, not gated) is also caught: the whole decision is compared (Astra r2 1)",
          any("does not reproduce" in x for x in A2.verdict_problems(2, TD3)))
    open(vp2, "wb").write(keep_v2)
    A2._REPLAY_MEMO.clear()
    check("11b.3 the intact attempt-2 verdict binds its record, authorization, attempt-1 verdict and a clean final validator",
          v2n["final_problems"] == [] and v2n["correction_file_sha"] == A.file_sha(A2.correction_path(TD3)) and A2.verdict_problems(2, TD3) == []
          and os.path.isfile(os.path.join(A2.bundle_dir(2, TD3), "docs", "final-test", A2.CORRECTION_FILE))
          and os.path.isfile(os.path.join(A2.bundle_dir(2, TD3), "docs", "final-test", A2.AUTH2_FILE)), A2.verdict_problems(2, TD3))
    keep_rec = open(A2.correction_path(TD3), "rb").read()
    save_record(TD3, dict(rec3, written_by="edited after the verdict"))
    check("11b.4 a correction record edited after the attempt-2 verdict invalidates it", any("correction record changed" in x for x in A2.verdict_problems(2, TD3)))
    open(A2.correction_path(TD3), "wb").write(keep_rec)
    # ---- free calls / every write (Astra r2 4)
    import types as _types
    gl = os.path.join(SCRATCH, "free", A2.NS2, "ledger.db")
    os.makedirs(os.path.dirname(gl), exist_ok=True)
    AC.register_ledger_guard(gl, attempt_base_usd=0.0, attempt_cap_usd=8.0, lock_path=os.path.join(SCRATCH, "free", "s.lock"),
                             lifecycle=lambda tot: ["the record changed"])
    free_no = AC.guard_problems(_types.SimpleNamespace(path=gl), None, 0.0)
    AC.unregister_ledger_guard(gl)
    check("11b.5 a FREE reservation (0 USD) still meets the attempt lifecycle", free_no == ["the record changed"], free_no)
    import plan as _PL

    EVS = []

    class _Base:
        stage = "gemma"

        def _call_cell(self, c, rec):
            return "called"

        def ev(self, name, **kw):
            EVS.append(dict(kw, event=name))

    class _Cell(A._AccOrder, _Base):
        ns = A2.NS2
    AC.arm_session([], by="selftest", namespace=A2.NS2, lifecycle=lambda: ["record edited"])
    cell_no = raises(lambda: _Cell()._call_cell({"key": "k"}, {}), _PL.PlanAbort)
    AC.arm_session([], by="selftest", namespace=A2.NS2, lifecycle=lambda: [])
    _cell = _Cell()
    cell_ok = _cell._call_cell({"key": "k"}, {}) == "called" and _cell._call_cell({"key": "k"}, {}) == "called"
    cell_ev_ok = [e for e in EVS if e["event"] == A2.CELL_ATTEMPT_EVENT] == [{"event": A2.CELL_ATTEMPT_EVENT, "key": "k", "run_id": _cell._acc_run_id,
                                                                              "stage": _cell.stage}]
    ExC = A.exec_class()
    eo = object.__new__(ExC)
    eo.ns = A2.NS2
    AC.arm_session([], by="selftest", namespace=A2.NS2, lifecycle=lambda: ["record edited"])
    import execute as _EX
    exec_no = raises(lambda: ExC.run_one(eo, {}, 1, 0), _EX.ExecAborted)
    jp_ = C.journal_path(A2.NS2, "events", os.path.join(SCRATCH, "jw"))
    jw_no = raises(lambda: C.Journal(jp_).append({"event": "x", "phase": "plan"}), C.JournalRefused)
    AC.disarm_session()
    jw_ok = not raises(lambda: C.Journal(jp_).append({"event": "x", "phase": "plan"}), C.JournalRefused)
    check("11b.6 an armed process re-checks its lifecycle before EVERY planner submission (free Gemma / Codex too), execution attempt and "
          "journal row (Astra r2 4)", cell_no and cell_ok and exec_no and jw_no and jw_ok, (cell_no, cell_ok, exec_no, jw_no, jw_ok))
    check("11d.0 RUNNER-FINAL (Astra r3b 3): every planner submission journals ONE cell-bound cell_attempt event per cell per run (a "
          "lifecycle refusal journals none)", cell_ev_ok, EVS)
    closed_no = raises(lambda: C.Journal(C.journal_path(A2.NS1, "exec", TD3)).append({"unit": "late"}), C.JournalRefused)
    check("11b.7 a CLOSED attempt refuses every journal write, whoever writes (no session needed)", closed_no and AC.VERDICT_NAME == A2.VERDICT_NAME)
    rw_closed = raises(lambda: A.write_ns_report(A2.NS1, "c11_acceptance.json", {"x": 1}, TD3), A.Refused)
    TD5 = make_root()
    TDS.append(TD5)
    with C.PhaseLock(A2.NS1, "verdict", TD5):
        rw_locked = raises(lambda: A.write_ns_report(A2.NS1, "c11_acceptance.json", {"x": 1}, TD5), A.Refused)
    rw_ok = os.path.isfile(A.write_ns_report(A2.NS1, "c11_acceptance.json", {"x": 1}, TD5))
    os.remove(os.path.join(C.ns_docs(A2.NS1, TD5), "c11_acceptance.json"))
    check("11b.8 the scanners' report writer refuses a CLOSED attempt and waits while a verdict is computed (Astra r2 8)",
          rw_closed and rw_locked and rw_ok, (rw_closed, rw_locked, rw_ok))
    # ---- terminal states + the close-out (Astra r2 5)
    pres_ = A.presentations(ROSTER)
    MS = A2.max_session()
    check("11b.9 cell_terminal: ok / rejected terminal; infra only at its last session; ceiling / absent only through the close-out",
          A2.cell_terminal({"status": "ok"}, None) == "ok" and A2.cell_terminal({"status": "rejected"}, None) == "rejected"
          and A2.cell_terminal({"status": "infra", "session": 0}, None) is None and A2.cell_terminal({"status": "infra", "session": MS}, None) == "infra_planner"
          and A2.cell_terminal({"status": "ceiling"}, None) is None and A2.cell_terminal({"status": "ceiling"}, "budget") == "budget"
          and A2.cell_terminal(None, None) is None and A2.cell_terminal(None, "missing") == "missing" and A2.cell_terminal({"status": "ok"}, "missing") == "ok")
    pj5 = C.journal_path(A2.NS1, "plan", TD5)
    rows5 = [json.loads(x) for x in open(pj5, encoding="utf-8") if x.strip()]
    k_inf = C.cell_id(A2.NS1, "plan", "O0", pres_[0], "r0")
    k_mis = C.cell_id(A2.NS1, "plan", "G0", pres_[3], "r0")
    rows5 = [dict(r_, status="infra", session=0) if r_["key"] == k_inf else r_ for r_ in rows5 if r_["key"] != k_mis]
    jl(pj5, rows5)
    t_inf = A2.terminal_problems(1, TD5)
    check("11b.10 infra cells at session 0 (registered re-queues pending) and a missing cell keep the run NON-terminal (no verdict)",
          any("planner cell" in x for x in t_inf) and raises(lambda: A2.record_verdict(1, TD5, write=True), A.Refused), t_inf)
    co_no = raises(lambda: A2.record_closeout(1, TD5, by="selftest", reason="cannot finish"), A.Refused)
    jl(pj5, [dict(r_, status="infra", session=MS) for r_ in rows5 if r_["key"] == k_inf], mode="a")
    co_few = raises(lambda: A2.record_closeout(1, TD5, by="selftest", reason="cannot finish"), A.Refused)
    jl(C.journal_path(A2.NS1, "events", TD5), [{"event": "plan_start", "phase": "plan", "passes": ["gemma"]}] * (MS + 1), mode="a")
    co_starts = raises(lambda: A2.record_closeout(1, TD5, by="selftest", reason="cannot finish"), A.Refused)   # r3b 3: launches are not attempts
    jl(C.journal_path(A2.NS1, "events", TD5), [{"event": A2.CELL_ATTEMPT_EVENT, "key": k_mis, "run_id": f"run{i_}"} for i_ in range(MS + 1)], mode="a")
    co = A2.record_closeout(1, TD5, by="selftest", reason="the Gemma cell never produced a row after every registered session")
    check("11b.11 the close-out: refused while a registered session remains or the stage ran < MAX_SESSION + 1 times; then EVERY "
          "non-terminal cell at once (missing), written once; the run is terminal", co_no and co_few and co_starts and co == {k_mis: "missing"}
          and not any("planner cell" in x for x in A2.terminal_problems(1, TD5))
          and raises(lambda: A2.record_closeout(1, TD5, by="selftest", reason="again"), A.Refused), (co_no, co_few, co))
    # ---- the finalization holds the study-wide spend boundary (Astra r2 8)
    lk5 = os.path.join(SCRATCH, "study5.lock")
    A2.STUDY_LOCK_OVERRIDE = lk5
    open(lk5, "w").write(json.dumps({"token": "someone else", "pid": 1}))
    wait0 = AC.STUDY_LOCK_WAIT_S
    AC.STUDY_LOCK_WAIT_S = 0.3
    busy = raises(lambda: A2.record_verdict(1, TD5, write=True), A.Refused) and not os.path.exists(A2.verdict_path(1, TD5))
    os.remove(lk5)
    AC.STUDY_LOCK_WAIT_S = wait0
    v15 = A2.record_verdict(1, TD5, write=True)
    check("11b.12 the verdict waits while the study-wide spend boundary is held, then publishes under it (lock released after)",
          busy and v15["closeout"] == {k_mis: "missing"} and not os.path.exists(lk5), busy)
    A2.STUDY_LOCK_OVERRIDE = SCRATCH_LOCK
    # ---- history (Astra r2 9)
    bfile = os.path.join(A2.bundle_dir(1, TD5), "sim", "honest_skills.py")
    check("11b.13 bundle files are independent copies (no hard link)", os.stat(bfile).st_nlink == 1)
    tc = os.path.join(C.ns_work(A2.NS1, TD5), "tokcache.json")
    keep_tc = open(tc, "rb").read()
    open(tc, "w").write('{"x": 1}\n')
    check("11b.14 a work file (token cache) changed after the verdict is refused", any("work files" in x for x in A2.verdict_problems(1, TD5, replay=False)))
    open(tc, "wb").write(keep_tc)
    c_ = sqlite3.connect(p1)
    c_.execute("UPDATE ledger SET ts = 'edited' WHERE id = (SELECT MIN(id) FROM ledger)")
    c_.commit()
    c_.close()
    same_tot = A2.ledger_state(1, TD3)["liability"] == v1b["ledgers"]["1"]["liability"]
    check("11b.15 a same-total edit of an attempt-1 ledger row is visible (rows sha) to the bindings and the verdict check",
          same_tot and any("ledger" in x for x in A2.bindings_problems(TD3)) and any("ledger rows" in x for x in A2.verdict_problems(1, TD3, replay=False)))
    # ---- gates (Astra r2 10)
    gsrc = os.path.join(TD, "sim", "_acc_attempt2_selftest.py")
    keep_g = open(gsrc, "rb").read()
    os.remove(gsrc)
    gone = A2.gates_problems(rec, TD)
    open(gsrc, "wb").write(keep_g)
    check("11b.16 deleting a required gate's source refuses (the manifest is frozen, the sources are bound)",
          any("missing from the tree" in x for x in gone) and "_acc_attempt2_selftest" in A2.required_gates(TD), gone)
    # ---- byte-exact diffs (Astra r2 11)
    hsq = os.path.join(TD, "sim", "honest_skills.py")
    keep_hs = open(hsq, "rb").read()
    base_b = open(os.path.join(A2.bundle_dir(1, TD), "sim", "honest_skills.py"), "rb").read()
    results = []
    for new_b in (base_b.replace(b"\r\n", b"\n") if b"\r\n" in base_b else base_b.replace(b"\n", b"\r\n"), base_b.rstrip(b"\r\n"), b"X = 5\r\nY = 6"):
        open(hsq, "wb").write(new_b)
        n_ = A2.code_state(TD)
        patch = A2.make_diff(v1, TD, now=n_)
        results.append(bool(patch.strip()) and A2.diff_reconstruction_problems(patch, v1, TD, now=n_) == []
                       and bool(A2.diff_reconstruction_problems(patch.replace("+X", "+Z").replace("-X", "-Z"), v1, TD, now=n_)))
    open(hsq, "wb").write(keep_hs)
    check("11b.17 a CRLF <-> LF change, a dropped final newline and a mixed edit each give a non-empty patch that rebuilds the exact disk "
          "bytes; an altered patch does not (Astra r2 11)", all(results), results)
    check("11b.18 a changed model / provider selection or rendered prompt text since attempt 1 refuses (Astra r2 3 / 7)",
          any("model / provider" in x for x in A2.code_diff_problems(rec, dict(v1, planner_config_sha="0"), TD))
          and any("prompt texts" in x for x in A2.refreeze_problems(rec, dict(v1, prompt_texts_sha="0"), TD)))
    # ------------------------------------------------------------------------------------------------ 11c Astra r3 interim gaps (RUNNER-FINAL)
    print("== 11c Astra ACC-ATTEMPT2 r3 interim gaps: authorization bytes, root spellings, writers under the verdict lock, per-stage "
          "close-out, the runner's stage order, the re-queue flag")
    check("11c.0 a changed rendered-messages digest since attempt 1 refuses", any("PlannerContract renders" in x for x in
          A2.refreeze_problems(rec, dict(v1, rendered_prompts_sha="0"), TD)))
    apq = A2.auth2_path(TD3)
    keep_au = open(apq, "rb").read()
    au_ = json.load(open(apq, encoding="utf-8"))
    write_json(apq, dict(au_, said="edited after the verdict, stored sha kept"))
    vp_au = A2.verdict_problems(2, TD3, replay=False)
    open(apq, "wb").write(keep_au)
    check("11c.1 an attempt-2 authorization edited after the verdict with its STORED sha left unchanged invalidates the verdict (r3 interim 6)",
          any("authorization" in x for x in vp_au) and A2.verdict_problems(2, TD3, replay=False) == [], (vp_au, A2.verdict_problems(2, TD3, replay=False)))
    rr = os.path.abspath(TD3)
    probe = {"a": rr + "\\sim\\x.py", "b": repr(rr + "\\sim\\x.py"), "c": json.dumps(rr + "\\y"), "d": rr.replace("\\", "/") + "/z",
             "e": rr[0].swapcase() + rr[1:] + "\\w", rr + "\\k": 1}
    nz = A2.normalize(probe, TD3)
    flat = json.dumps(nz)
    check("11c.2 normalize: plain, repr (doubled backslashes), JSON-escaped, forward-slash and other-case drive spellings of the root all "
          "become <root> (r3 interim 5)", nz["a"] == "<root>\\sim\\x.py" and nz["e"] == "<root>\\w" and "<root>\\k" in nz
          and nz["d"] == "<root>/z" and "<root>" in nz["b"] and "<root>" in nz["c"] and os.path.basename(rr) not in flat, nz)
    TD6 = make_root()
    TDS.append(TD6)
    with C.PhaseLock(A2.NS1, "verdict", TD6):
        jw_lock = raises(lambda: C.Journal(C.journal_path(A2.NS1, "events", TD6)).append({"event": "late"}), C.JournalRefused)
    C.Journal(C.journal_path(A2.NS1, "events", TD6)).append({"event": "after"})
    check("11c.3 while the verdict lock is held, EVERY journal of the namespace refuses (whoever writes); after it writes go through "
          "(r3 interim 7)", jw_lock)
    seen = []

    def fake_main(argv):
        seen.append(list(argv))
        return 0
    cf_closed = A.context_fit_cmd(A2.NS1, "http://x", TD3, main=fake_main)
    with C.PhaseLock(A2.NS1, "verdict", TD6):
        cf_locked = A.context_fit_cmd(A2.NS1, "http://x", TD6, main=fake_main)
    cf_ok = A.context_fit_cmd(A2.NS1, "http://x", TD6, main=fake_main)
    other_ns = next(n_ for n_ in C.NAMESPACES if n_ not in C.ACCEPTANCE_NAMESPACES)
    cf_other = A.context_fit_cmd(other_ns, "http://x", TD6, main=fake_main)
    check("11c.4 acc_runner context-fit: refused on a CLOSED attempt and while a verdict is computed; otherwise context_fit.py's own "
          "--real --write path runs under the verdict lock; only for the acceptance namespaces (r3 interim 7)",
          cf_closed == 2 and cf_locked == 2 and cf_ok == 0 and cf_other == 2 and len(seen) == 1 and "--write" in seen[0] and "--real" in seen[0]
          and seen[0][seen[0].index("--namespace") + 1] == A2.NS1, (cf_closed, cf_locked, cf_ok, cf_other, seen))
    TD7 = make_root()
    TDS.append(TD7)
    pj7 = C.journal_path(A2.NS1, "plan", TD7)
    rows7 = [json.loads(x) for x in open(pj7, encoding="utf-8") if x.strip()]
    k_g = C.cell_id(A2.NS1, "plan", "G0", pres_[5], "r0")
    k_o = C.cell_id(A2.NS1, "plan", "O0", pres_[5], "r0")
    jl(pj7, [r_ for r_ in rows7 if r_["key"] not in (k_g, k_o)])
    sp_g = A.stage_prereq_problems("opus", ROSTER, TD7)
    o_first = raises(lambda: A2.record_closeout(1, TD7, by="selftest", reason="r", stage="opus"), A.Refused)
    jl(C.journal_path(A2.NS1, "events", TD7), [{"event": "plan_start", "phase": "plan", "passes": ["gemma"]}] * (MS + 1), mode="a")
    starts_only = raises(lambda: A2.record_closeout(1, TD7, by="selftest", reason="r"), A.Refused)
    jl(C.journal_path(A2.NS1, "events", TD7), [{"event": A2.CELL_ATTEMPT_EVENT, "key": k_g, "run_id": "r-a"}] * 3
       + [{"event": A2.CELL_ATTEMPT_EVENT, "key": k_g, "run_id": "r-b"}], mode="a")
    two_runs = raises(lambda: A2.record_closeout(1, TD7, by="selftest", reason="r"), A.Refused)
    jl(C.journal_path(A2.NS1, "events", TD7), [{"event": A2.CELL_ATTEMPT_EVENT, "key": k_g, "run_id": f"r-{x_}"} for x_ in "cd"], mode="a")
    exec_before = A.make_execute(TD7, dry_run=True).partial
    co_g = A2.record_closeout(1, TD7, by="selftest", reason="the G0 cell has no row after every registered session")
    sp_o = A.stage_prereq_problems("opus", ROSTER, TD7)
    g_twice = raises(lambda: A2.record_closeout(1, TD7, by="selftest", reason="again", stage="gemma"), A.Refused)
    check("11c.5 per-stage close-out: the runner's stage order refuses Opus while a Gemma cell is missing; an Opus close-out before the "
          "Gemma stage is terminal refuses; the Gemma close-out (all its non-terminal cells, written once) lets Opus start (r3 interim 2)",
          any("gemma" in x for x in sp_g) and o_first and co_g == {k_g: "missing"} and not any("gemma" in x for x in sp_o) and g_twice,
          (sp_g, o_first, co_g, sp_o, g_twice))
    ev7 = [json.loads(x) for x in open(C.journal_path(A2.NS1, "events", TD7), encoding="utf-8") if x.strip()]
    check("11d.1 RUNNER-FINAL (Astra r3b 3): a missing cell is closed out only after ITS OWN attempts in MAX_SESSION + 1 distinct runs "
          "(stage launches alone refuse; 2 runs refuse; repeats in one run count once)", starts_only and two_runs and co_g == {k_g: "missing"},
          (starts_only, two_runs))
    ex7 = A.make_execute(TD7, dry_run=True)
    check("11d.2 RUNNER-FINAL (Astra r3b 2): the close-out also journals the registered stopping_rule, and acc_runner's execute phase then "
          "runs on execute.py's --partial path (no plan phase_complete needed); without a close-out it does not",
          exec_before is False and ex7.partial is True and any(e.get("event") == "stopping_rule" and e.get("rule") == "acceptance_closeout" for e in ev7),
          (exec_before, ex7.partial))
    jl(pj7, [{"key": k_o, "status": "infra", "session": 0}], mode="a")
    sp_c0 = A.stage_prereq_problems("codex", ROSTER, TD7)
    jl(pj7, [{"key": k_o, "status": "infra", "session": MS}], mode="a")
    sp_c1 = A.stage_prereq_problems("codex", ROSTER, TD7)
    check("11c.6 the runner's stage order uses the verdict's terminal rule: an infra cell with a re-queue session left keeps its stage "
          "incomplete; at its last session it is terminal", any("opus" in x for x in sp_c0) and not any("opus" in x for x in sp_c1), (sp_c0, sp_c1))
    got_kw = {}

    class _FakePlan:
        def run(self):
            return 0
    mk0, pf0 = A.make_plan, A.preflight
    try:
        A.make_plan = lambda stage, root=None, **kw: (got_kw.setdefault(kw.get("attempt", 1), dict(kw)), _FakePlan())[1]
        A.preflight = lambda *a_, **k_: []
        rc_1 = A.main(["plan", "--stage", "gemma", "--requeue-infra"])
        rc_2 = A.main(["plan", "--stage", "gemma", "--attempt", "2", "--requeue-infra"])
        got_kw["plain"] = None
        rc_3 = A.main(["plan", "--stage", "opus"])
    finally:
        A.make_plan, A.preflight = mk0, pf0
    check("11c.7 the acceptance CLI passes --requeue-infra to the planner (both attempts) (r3 interim 3)",
          rc_1 == 0 and rc_2 == 0 and rc_3 == 0 and got_kw.get(1, {}).get("requeue_infra") is True and got_kw.get(2, {}).get("requeue_infra") is True,
          got_kw)
    TD8 = make_root()
    TDS.append(TD8)
    ej8 = C.journal_path(A2.NS1, "events", TD8)
    ev8 = [json.loads(x) for x in open(ej8, encoding="utf-8") if x.strip()]
    jl(ej8, [e for e in ev8 if not (e.get("event") == "phase_complete" and e.get("phase") == "execute")])
    t_open = A2.terminal_problems(1, TD8)
    no_abort = raises(lambda: A2.record_exec_closeout(1, TD8, by="selftest", reason="r"), A.Refused)
    jl(C.journal_path(A2.NS1, "exec", TD8), [{"unit": "u2", "status": "invalid", "error": "reset fault"}], mode="a")
    jl(ej8, [{"event": "abort_invalid", "phase": "execute", "pres": "p", "unit": "u2"}], mode="a")
    xc = A2.record_exec_closeout(1, TD8, by="selftest", reason="the INVALID run cannot be re-run inside this attempt")
    t_closed = A2.terminal_problems(1, TD8)
    twice = raises(lambda: A2.record_exec_closeout(1, TD8, by="selftest", reason="again"), A.Refused)
    jl(C.journal_path(A2.NS1, "exec", TD8), [{"unit": "u3", "status": "done"}], mode="a")
    t_moved = A2.terminal_problems(1, TD8)
    check("11d.3 RUNNER-FINAL (Astra r3b 2): an execute phase stopped by an INVALID run gets ONE explicit close-out (only after abort_invalid; "
          "it lists the INVALID units and binds the exec / outcome journals); the run is then terminal, and any later exec row breaks it",
          any("execute phase" in x for x in t_open) and no_abort and xc["invalid_units"] == ["u2"] and not any("execute" in x for x in t_closed)
          and twice and any("after the execute close-out" in x for x in t_moved), (t_open, t_closed, t_moved))
    up8 = os.path.join(C.ns_docs(A2.NS1, TD8), "units.json")
    open(up8, "wb").write(b"{\n")
    open(os.path.join(C.ns_docs(A2.NS1, TD8), "c11_acceptance.json"), "wb").write(b"[1, 2]\n")
    stub_off()
    try:
        ev_c = A.evidence(TD8)
        d_c = A.decide(ev_c, REG, ROSTER)
        cr_ = d_c["criteria"]["completeness"]["reasons"]
        corrupt_ok = (d_c["verdict"] == "NOT PASS" and ev_c["units"].get("_corrupt") is True and ev_c["c11"].get("_corrupt") is True
                      and any(r_.startswith("units.json is unreadable") for r_ in cr_)
                      and any(r_.startswith("c11_acceptance.json is unreadable") for r_ in cr_) and os.path.getsize(up8) == 2)
        corrupt_info = d_c["criteria"]["completeness"]["reasons"][:6]
    except Exception as e:                                     # noqa: BLE001
        corrupt_ok, corrupt_info = False, f"{type(e).__name__}: {e}"
    finally:
        stub_on()
    check("11d.4 RUNNER-FINAL (Astra r3b 4): a truncated units.json and a non-object c11 report are corrupt EVIDENCE: evidence() does not "
          "raise and decide() FAILS completeness (NOT PASS); the bytes stay on disk", corrupt_ok, corrupt_info)
    long_root = os.path.join(TD8, "acceptance_bundles", "acceptance-v2", "x" * 60)
    m_live = A._rootless(f"FileNotFoundError: [Errno 2] No such file or directory: {os.path.join(TD8, 'sim', 'final_test', 'practice_rows.jsonl')!r}", TD8)[:200]
    m_bund = A._rootless(f"FileNotFoundError: [Errno 2] No such file or directory: {os.path.join(long_root, 'sim', 'final_test', 'practice_rows.jsonl')!r}", long_root)[:200]
    check("11d.5 RUNNER-FINAL (Astra r3b 5): a diagnostic is made root-free BEFORE truncation, so a short and a long checkout root give the "
          "same 200-character text", m_live == m_bund and "practice_rows.jsonl" in m_live and "<root>" in m_live, (m_live, m_bund))
    # RUNNER-FINAL (Astra RUNNER-FINAL r1 1): a chunk refused by the ceiling journals its cell keys; such a cell closes out as BUDGET
    import plan as PL_
    import types
    seen_ev = []

    class _RefLedger:
        def block_all(self, w):
            raise C.BlockRefused("attempt ceiling 8.00 USD exhausted")

        def committed(self):
            return 8.0
    ck_ = [{"key": "kA", "scope": "core", "planner": "O", "pres": "p1"}, {"key": "kB", "scope": "core", "planner": "O", "pres": "p1"}]
    ck2_ = [{"key": "kC", "scope": "core", "planner": "O", "pres": "p2"}, {"key": "kD", "scope": "core", "planner": "O", "pres": "p2"}]
    fake_b = types.SimpleNamespace(ledger=_RefLedger(), _worst=lambda cs: 0.5 * len(cs), ev=lambda name, **kw: seen_ev.append(dict(kw, event=name)),
                                   _order=lambda: [("p1", ck_), ("p2", ck2_)], latest={"kD": {"status": "ok"}}, requeue=False)
    fake_b._stopped_keys = lambda cs: A._AccBatch._stopped_keys(fake_b, cs)
    stop_ = raises(lambda: A._AccBatch._run_chunk(fake_b, [("p1", ck_)], 0), PL_._Stop)
    TD8 = make_root()
    TDS.append(TD8)
    pj8 = C.journal_path(A2.NS1, "plan", TD8)
    rows8 = [json.loads(x) for x in open(pj8, encoding="utf-8") if x.strip()]
    k_g8 = C.cell_id(A2.NS1, "plan", "G0", pres_[5], "r0")
    jl(pj8, [r_ for r_ in rows8 if r_["key"] != k_g8])
    no_ev = raises(lambda: A2.record_closeout(1, TD8, by="selftest", reason="r"), A.Refused)
    jl(C.journal_path(A2.NS1, "events", TD8), [{"event": "block_refused", "phase": "plan", "keys": [k_g8], "detail": "attempt ceiling"}], mode="a")
    co8 = A2.record_closeout(1, TD8, by="selftest", reason="the attempt ceiling refused the chunk")
    term8 = A2.cell_terminal(None, A2.closeout_states(A2.NS1, TD8).get(k_g8))
    sp8 = A.stage_prereq_problems("opus", ROSTER, TD8)
    k_oa = C.cell_id(A2.NS1, "plan", "O0", pres_[5], "r0")
    k_ob = C.cell_id(A2.NS1, "plan", "O0", pres_[6], "r0")
    rows8b = [json.loads(x) for x in open(pj8, encoding="utf-8") if x.strip()]
    jl(pj8, [r_ for r_ in rows8b if r_["key"] not in (k_oa, k_ob)])
    jl(C.journal_path(A2.NS1, "events", TD8), [{"event": "block_refused", "phase": "plan", "keys": [k_oa, k_ob], "detail": "attempt ceiling"}], mode="a")
    co8o = A2.record_closeout(1, TD8, by="selftest", reason="the attempt ceiling refused the opus chunks", stage="opus")
    check("11d.6 RUNNER-FINAL (Astra RUNNER-FINAL r1 1, r2 1-2): the real _AccBatch._run_chunk journals the keys of the refused chunk AND of "
          "every later pending cell (a done cell excluded) and stops; a cell without a row and without a refusal still needs its sessions; one "
          "named by a refusal closes out as BUDGET at once, is TERMINAL (cell_terminal), lets the next stage start, and a later stage's refused "
          "cells close out the same way",
          stop_ and seen_ev and seen_ev[-1]["event"] == "block_refused" and seen_ev[-1].get("keys") == ["kA", "kB", "kC"] and no_ev
          and co8 == {k_g8: "budget"} and term8 == "budget" and not any("gemma" in x for x in sp8)
          and co8o == {k_oa: "budget", k_ob: "budget"}, (stop_, seen_ev[-1:], no_ev, co8, term8, sp8, co8o))
finally:
    A2.STUDY_LOCK_OVERRIDE = SCRATCH_LOCK
    AC.STUDY_LOCK_WAIT_S = 30.0
    AC.disarm_session()
    stub_off()
    for p_ in list(AC._LEDGER_GUARDS):
        if "_selftest" in p_:
            AC._LEDGER_GUARDS.pop(p_, None)
    for t_ in TDS:
        shutil.rmtree(t_, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ 12 the whole chain, UNSTUBBED
print("== 12 the whole chain UNSTUBBED on a full copy of the tracked tree: attempt-1 verdict, an identity-changing correction, Arash's go, "
      "the a2 run's journals, the attempt-2 verdict, both attempts (real decide, real bundle replays in fresh interpreters, real identity, "
      "real owed state)")
TDU = tmpdir("full_")
try:
    for rel_ in A2.tracked_files() + [f"sim/{g}.py" for g in A2.REQUIRED_GATES]:
        d_ = os.path.join(TDU, *rel_.split("/"))
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, *rel_.split("/")), d_)
    restore_frozen(TDU)
    o_ = json.load(open(os.path.join(ROOT, "docs", "final-test", "acceptance_owed.json"), encoding="utf-8"))
    for e_ in o_.values():
        evs = e_.get("evidence") if isinstance(e_, dict) else None
        for x_ in (evs if isinstance(evs, list) else [evs] if isinstance(evs, dict) else []):
            if x_ and x_.get("path") and os.path.isfile(os.path.join(ROOT, *x_["path"].split("/"))):
                d_ = os.path.join(TDU, *x_["path"].split("/"))
                os.makedirs(os.path.dirname(d_), exist_ok=True)
                shutil.copyfile(os.path.join(ROOT, *x_["path"].split("/")), d_)
    with open(os.path.join(TDU, "docs", "final-test", "progress.md"), "w", encoding="utf-8") as f:
        f.write(f"# ledger\n\n## 2026-10-14 -- {LEDGER_TITLE}\n- go\n")
    terminal_journals(TDU, A2.NS1)
    ow_u = A2.owed_state(TDU)
    if ow_u["error"] or ow_u["problems"]:                      # the study's REAL owed state is not clean right now: say so, then go on
        print(f"NOTE 12: the real owed state is not clean now ({(ow_u['error'] or ow_u['problems'][0])[:140]}); section 12 replaces ONLY "
              "the owed state by a clean one (attempt 1 could not start in this state)")
        A2.owed_state = lambda root=None: {"problems": [], "error": None}
    v1u = A2.record_verdict(1, TDU, write=True)
    check("12.1 a real attempt-1 verdict: live decide == a fresh-interpreter replay in the bundle (the bundle holds every input)",
          v1u["verdict"] == "NOT PASS" and v1u["bundle"]["n_files"] > 200 and v1u["identity"]["sha"] == T.identity(A2.NS1, TDU)["sha"])
    check("12.2 the real verdict is intact under a NEW replay", A2.attempt1_fail_problems(TDU) == [], A2.attempt1_fail_problems(TDU))
    crit_u = v1u["decision"]["criteria"]
    fc = next((k for k in A2.HARD if crit_u.get(k, {}).get("pass") is False and crit_u[k].get("reasons")), None)
    reasons_u = (crit_u.get(fc) or {}).get("reasons") or []
    # an IDENTITY-CHANGING, treatment-preserving remedy: a comment in ltm_package.py (an identity file, not executor-pinned)
    rem_rel = "sim/ltm_package.py"
    q_ = os.path.join(TDU, *rem_rel.split("/"))
    open(q_, "ab").write(b"\n# ACC-ATTEMPT2 selftest remedy (a comment)\n")
    now_ = A2.code_state(TDU)
    reg_u = A.registration(TDU)
    moved_u = {r_: now_[r_] for r_, w_ in (reg_u.get("frozen_refs") or {}).get("files", {}).items() if now_.get(r_) != w_}
    id2 = T.identity(A2.NS2, TDU)["sha"]
    files_ = {rem_rel: {"before": v1u["code_state"][rem_rel], "after": now_[rem_rel], "role": "remedy", "why": "selftest remedy"}}
    recu = {"schema": A2.CORRECTION_SCHEMA, "date": "2026-10-15", "written_by": "selftest (unstubbed)",
            "attempt1_verdict": {"path": rel(TDU, A2.verdict_path(1, TDU)), "sha256": A.file_sha(A2.verdict_path(1, TDU))},
            "declarations": {k: True for k in A2.DECLARATIONS},
            "defects": [{"category": "bug", "criterion": fc, "attempt1_reasons": reasons_u[:1],
                         "traced_to": [{"path": rem_rel, "sha256": v1u["code_state"][rem_rel], "what": "selftest"}], "description": "selftest"}],
            "remedy": {"description": "selftest", "files": files_, "diff": {"sha256": C.sha256_text(A2.make_diff(v1u, TDU))}},
            "refrozen": {"treatment_identity": id2, "frozen_refs": moved_u}}
    gates_for(TDU, recu)
    astra_for(TDU, recu)
    recu = save_record(TDU, recu)
    A2.record_auth2(by="Arash", date="2026-10-15", said="go", ledger_entry=LEDGER_TITLE, root=TDU)
    pfu = A2.preflight2(TDU)
    check("12.3 unstubbed preflight2 on an identity-changing correction (new identity re-frozen): record, byte diff, re-freeze, rendered "
          "prompt texts, model selection, gates, Astra, Arash, owed items, bindings all pass", fc and id2 != v1u["identity"]["sha"]
          and pfu == [], (fc, pfu[:4]))
    try:
        AC.arm_session([], by="selftest:attempt2", namespace=A2.NS2, lifecycle=A2.session_lifecycle(TDU))
        A2.on_enter(TDU)
        keep_rec = open(A2.correction_path(TDU), "rb").read()
        save_record(TDU, dict(recu, written_by="edited mid-run"))
        mid = raises(lambda: C.Journal(C.journal_path(A2.NS2, "exec", TDU)).append({"unit": "u"}), C.JournalRefused)
        open(A2.correction_path(TDU), "wb").write(keep_rec)
        check("12.4 a record edited mid-run: the armed attempt-2 process cannot write its next row (lifecycle at the journal)", mid)
        AC.disarm_session()
        terminal_journals(TDU, A2.NS2)
        v2u = A2.record_verdict(2, TDU, write=True)
        A2._REPLAY_MEMO.clear()
        bu = A2.both_attempts(TDU)
        check("12.5 the real attempt-2 verdict (full validator inside the locks; its bundle holds the correction record, so the replay decides "
              "against the re-frozen identity): NOT PASS = protocol stopped; both attempts intact under fresh replays",
              v2u["treatment_id"] == id2 and v2u["final_problems"] == [] and v2u["protocol_stopped"] is True and bu["acceptance"].startswith("STOPPED")
              and bu["attempts"]["1"]["intact"] and bu["attempts"]["2"]["intact"], (bu["acceptance"], bu["attempts"]["2"] and bu["attempts"]["2"]["problems"]))
        vp_u = A2.verdict_path(2, TDU)
        vd_u = json.load(open(vp_u, encoding="utf-8"))
        forged_u = dict(vd_u, verdict="PASS", decision=dict(vd_u["decision"], verdict="PASS"))
        forged_u.pop("sha256")
        forged_u["sha256"] = C.self_sha(forged_u)
        os.replace(vp_u, vp_u + ".orig")
        write_json(vp_u, forged_u)
        A2._REPLAY_MEMO.clear()
        check("12.6 unstubbed: the attempt-2 verdict rewritten as PASS (self sha recomputed) is INVALID under the real replay",
              A2.both_attempts(TDU)["acceptance"].startswith("INVALID"))
        os.remove(vp_u)
        os.replace(vp_u + ".orig", vp_u)
        check("12.7 after it attempt 2 is CLOSED: no journal write, no entry", raises(lambda: C.Journal(C.journal_path(A2.NS2, "plan", TDU)).append({"k": 1}), C.JournalRefused)
              and any("CLOSED" in x for x in A2.bindings_problems(TDU)))
    finally:
        AC.disarm_session()
        A2.owed_state = REAL[1]
    open(q_, "a", encoding="utf-8").write("# an undocumented second edit\n")
    check("12.8 unstubbed: an undocumented edit after the record refuses", any("before / after" in x or "remedy.diff" in x or "gates" in x for x in A2.preflight2(TDU)))
finally:
    A2.owed_state = REAL[1]
    A2._REPLAY_MEMO.clear()
    shutil.rmtree(TDU, ignore_errors=True)

REAL1 = real_snapshot()
check("13.1 the real acceptance-v2 / -a2 docs, work and bundle dirs are byte-identical", REAL1 == REAL0, sorted(set(REAL1.items()) ^ set(REAL0.items()))[:4])
A2.STUDY_LOCK_OVERRIDE = None
shutil.rmtree(SCRATCH, ignore_errors=True)
if not os.listdir(os.path.join(ROOT, A2.BUNDLES)):
    os.rmdir(os.path.join(ROOT, A2.BUNDLES))
print(f"\n_acc_attempt2_selftest: {N['ok']}/{N['ok'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
