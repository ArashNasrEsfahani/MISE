"""Selftest for the ft-v2 freeze plumbing (stage ACC-RUNNER, 2026-10-14; F8 of sim/final_test/diag/c11/freeze_dryrun.json): freeze_ft2.py,
gate_evidence.py --ft2, the freeze.py dispatch and the snapshot `dir` / `artefacts` checks in common.py. Offline (Windows python); reads the
real repo; every write goes to a temporary folder (evidence copies) and docs/final-test/freeze.json is never written (asserted).

    python sim/_ftm_freeze_ft2_selftest.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import common as C                                       # noqa: E402
import freeze as F                                       # noqa: E402
import freeze_ft2 as FT                                  # noqa: E402
import gate_evidence as GE                               # noqa: E402

N = {"ok": 0, "fail": 0}


def check(name, cond, detail=""):
    N["ok" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  :: {str(detail)[:300]}"))


FZ_PATH = C.paths(None)["freeze"]
FZ_SHA0 = F._sha_file(FZ_PATH)
# K23 (test-only): the real freeze_history dir exists since the A37 re-freeze; 6.12 compares its content before / after instead of
# asserting that it is absent
_HIST_DIR = os.path.join(ROOT, *FT.FREEZE_HISTORY_DIR.split("/"))
HIST0 = sorted((e, F._sha_file(os.path.join(_HIST_DIR, e))) for e in os.listdir(_HIST_DIR)) if os.path.isdir(_HIST_DIR) else None
REAL_DIR = GE.gates_ft2_dir(ROOT)
PROBES = {"ollama": {"model": F.OLLAMA_MODEL, "digest": "sha256:" + F.OLLAMA_DIGEST_PREFIX + "0" * 52, "version": "test", "parent_model": ""},
          "codex": {"version": (json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "x_isolation.json"), encoding="utf-8")) or {}).get("cli_version")}}

# ------------------------------------------------------------------------------------------------ 1 evidence reader (bound, stale = FAIL)
print("== 1 gates_ft2 evidence")
check("1.0 the ft-v2 evidence files exist (gate_evidence.py --ft2 --write)", all(os.path.isfile(os.path.join(REAL_DIR, f"{g}.json")) for g in FT.EVIDENCE_GATES_FT2))
real = {g: FT.read_evidence_ft2(ROOT, g) for g in FT.EVIDENCE_GATES_FT2}
check("1.1 every evidence file verifies against the disk (no stale input / generator / identity)",
      all(v["evidence"]["reason"] is None for v in real.values()), {g: v["evidence"]["reason"] for g, v in real.items() if v["evidence"]["reason"]})
_FIN_NS = {"G-iso": GE.FG_NS, "tokens": GE.ACC_FINAL_NS, "context": GE.ACC_FINAL_NS, "leakage": GE.ACC_FINAL_NS}   # FREEZE-GATES-FT2 / A36
_fin_ok = lambda g: (real[g]["status"] == "PASS" and real[g]["evidence"]["final_identity"]["status"] == "PASS"
                     and real[g]["evidence"]["final_identity"].get("namespace") == _FIN_NS[g]
                     and real[g]["evidence"]["final_identity"].get("required_identity") == GE.FT2_FINAL_IDENTITY)
check("1.2 the final-identity gates are PENDING, never PASS on dev evidence (PASS only on their final-identity part: G-iso freeze-gates-ft2, tokens / context / leakage acceptance-v2-a2)",
      all(_fin_ok(g) or (real[g]["status"] == "PENDING" and real[g]["evidence"]["final_identity"]["status"] == "PENDING") for g in GE.FINAL_IDENTITY_GATES))
_d125 = json.load(open(os.path.join(REAL_DIR, "12.5.json"), encoding="utf-8"))
_v125 = (_d125.get("evidence") or {}).get("verdict") or {}
check("1.3 12.5 is PENDING until freeze-gates-ft2 has scored P0 + E0 evidence, PASS only on that evidence's A26 verdict (the file's own "
      "evidence.verdict, the file verified against the disk)",
      (real["12.5"]["status"] == "PENDING" and real["12.5"]["pass"] is False)
      or (real["12.5"]["status"] == "PASS" and real["12.5"]["evidence"]["reason"] is None and _d125["status"] == "PASS"
          and _v125.get("status") == "PASS" and not _v125.get("reasons")), (real["12.5"], _v125.get("status"), _v125.get("reasons")))
TD = tempfile.mkdtemp(prefix="fzft2_")
orig_dir = GE.gates_ft2_dir
try:
    GE.gates_ft2_dir = lambda root=None: TD

    def put(gid, mut=None, raw=None):
        d = json.load(open(os.path.join(REAL_DIR, f"{gid}.json"), encoding="utf-8"))
        if mut:
            mut(d)
        with open(os.path.join(TD, f"{gid}.json"), "w", encoding="utf-8") as f:
            f.write(raw if raw is not None else json.dumps(d, ensure_ascii=False))
        return FT.read_evidence_ft2(ROOT, gid)

    g = put("12.10")
    check("1.4 an unchanged copy reads as its file status (PASS)", g["status"] == "PASS", g)

    def m_input(d):
        k = sorted(x for x in d["binding"]["inputs"] if "/" in x)[0]
        d["binding"]["inputs"][k] = "0" * 64
    g = put("12.10", m_input)
    check("1.5 a changed bound input = FAIL (stale)", g["status"] == "FAIL" and any("stale: input" in x for x in g["evidence"]["reason"]), g)

    def m_missing_input(d):
        d["binding"]["inputs"]["sim/final_test/does_not_exist.jsonl"] = "1" * 64
    g = put("12.2", m_missing_input)
    check("1.6 a bound input that no longer exists = FAIL", g["status"] == "FAIL", g)

    def m_gen(d):
        d["binding"]["generator"] = {k: "2" * 64 for k in d["binding"]["generator"]}
    g = put("leakage", m_gen)
    check("1.7 evidence written by a different generator version = FAIL", g["status"] == "FAIL" and any("generator" in x for x in g["evidence"]["reason"]), g)

    def m_ident(d):
        d["binding"]["identity"]["final"] = {"test-v1": "d5831abc" * 8}
    g = put("12.10", m_ident)
    check("1.8 evidence bound to another treatment identity = FAIL", g["status"] == "FAIL" and any("treatment identity" in x for x in g["evidence"]["reason"]), g)
    old = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "gates", "12.3.json"), encoding="utf-8"))
    g = put("12.3", raw=json.dumps(old))
    check("1.9 an ft-v1 gates/ file (schema ftm.gate/1, 2026-10-12) is never accepted", g["status"] == "FAIL" and any("ftm.gate/2" in x for x in g["evidence"]["reason"]), g)
    os.remove(os.path.join(TD, "12.3.json"))
    g = FT.read_evidence_ft2(ROOT, "12.3")
    check("1.10 a missing evidence file = FAIL", g["status"] == "FAIL" and "no evidence file" in str(g["evidence"]["reason"]))
    g = put("12.7", lambda d: d.update(synthetic=True))
    check("1.11 synthetic evidence = FAIL", g["status"] == "FAIL")
    g = put("X-iso", lambda d: d.update(**{"pass": False}))
    check("1.12 status PASS with pass false = FAIL (no hand-typed verdict)", g["status"] == "FAIL")
    g = put("12.2", lambda d: d.update(note="sk-ant-api03-" + "A" * 90))
    check("1.13 key-like text in evidence = FAIL", g["status"] == "FAIL")
    g = put("tokens", lambda d: (d.update(status="PASS", **{"pass": True}), d["final_identity"].update(status="PENDING")))
    check("1.14 a hand-edited final-identity gate PENDING -> PASS = FAIL (its final-identity part is not PASS)", g["status"] == "FAIL", g)
    FRESH = GE.ft2_evidence(ROOT, {"codex": PROBES["codex"]})
    d0 = json.load(open(os.path.join(REAL_DIR, "12.5.json"), encoding="utf-8"))
    _edit = "PASS" if d0["status"] != "PASS" else "NOT_APPLICABLE"     # 12.5 is PASS now: a hand edit to another passing status
    d0.update(status=_edit, **{"pass": True})
    open(os.path.join(TD, "12.5.json"), "w", encoding="utf-8").write(json.dumps(d0))
    g_nofresh = FT.read_evidence_ft2(ROOT, "12.5")
    g = FT.read_evidence_ft2(ROOT, "12.5", FRESH["12.5"])
    check(f"1.15 a hand-edited 12.5 status ({FRESH['12.5']['status']} -> {_edit}, pass true) disagrees with the fresh re-derivation = FAIL "
          "(without the re-derivation the edit would read as passing)", g["status"] == "FAIL"
          and any("re-derivation" in x for x in g["evidence"]["reason"]) and g_nofresh["status"] == _edit, (g, g_nofresh["status"]))
    shutil.copy(os.path.join(REAL_DIR, "12.10.json"), os.path.join(TD, "12.10.json"))
    g = FT.read_evidence_ft2(ROOT, "12.10", FRESH["12.10"])
    check("1.16 an unchanged file agrees with the fresh re-derivation", g["status"] == "PASS", g)
finally:
    GE.gates_ft2_dir = orig_dir
    shutil.rmtree(TD, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ 2 the generator itself (current evidence)
print("== 2 gate_evidence --ft2 recomputation")
gen = GE.ft2_evidence(ROOT, {"codex": PROBES["codex"]})
check("2.1 a fresh recomputation gives the same statuses as the written files", {g: v["status"] for g, v in gen.items()} ==
      {g: json.load(open(os.path.join(REAL_DIR, f"{g}.json"), encoding="utf-8"))["status"] for g in gen})
check("2.2 the dev part of G-iso / tokens / context passes (PENDING only for the final identity)",
      all(gen[g]["evidence"]["dev"]["dev_pass"] is True for g in GE.FINAL_IDENTITY_GATES))
check("2.3 the token gate uses the A24 2,000 maximum incl. the header (no ft-v1 800 budget)", "2,000" in gen["tokens"]["evidence"]["claim"]
      and not gen["tokens"]["evidence"]["dev"]["over_2000"] and max(gen["tokens"]["evidence"]["dev"]["worst_by_source"].values()) <= 2000)
gx = GE.ft2_evidence(ROOT, {"codex": {"version": "codex-cli 9.9.9"}})["X-iso"]
check("2.4 X-iso fails when the CLI now differs from the probed one", gx["status"] == "FAIL" and any("probe again" in r for r in gx["evidence"]["reasons"]))
gx = GE.ft2_evidence(ROOT, {})["X-iso"]
check("2.5 X-iso fails when the CLI version was not probed now", gx["status"] == "FAIL")
check("2.6 identities: dev rehearsal differs from the final only in the briefing approval flag (recorded, not waived)",
      gen["12.2"]["binding"]["identity"]["dev_vs_final"]["fields"] == ["briefing"]
      and gen["12.2"]["binding"]["identity"]["dev_vs_final"]["briefing_fields"] == ["allow_unapproved"])
check("2.7 the ft-v1 evidence folder is untouched", sorted(os.listdir(os.path.join(ROOT, "docs", "final-test", "matrix", "gates"))) ==
      sorted(["12.10.json", "12.2.json", "12.3.json", "12.5.json", "12.7.json", "G-iso.json", "leakage.json", "X-iso.json"]))

# ------------------------------------------------------------------------------------------------ A36-IMPL fixtures (temporary record + approval)
# A temporary implementation record + in-memory approval (Astra A36-IMPL-freeze-r1): the record lives in a scratch folder under _checkpoints
# (untracked); FT.A36_IMPL_FILE and the not-yet-written freeze-r2 review are pointed there. The real signatures.json / record are never written.
SIG_REAL = json.load(open(os.path.join(C.paths(None)["docs"], "signatures.json"), encoding="utf-8"))
_ORIG_IMPL_FILE, _ORIG_REVIEWS = FT.A36_IMPL_FILE, dict(FT.A36_REQUIRED_REVIEWS)
TI = tempfile.mkdtemp(prefix="fzft2impl_", dir=os.path.join(ROOT, "sim", "final_test", "_checkpoints"))
_TI_REL = os.path.relpath(TI, ROOT).replace(os.sep, "/")
open(os.path.join(TI, "final-review.txt"), "w", encoding="utf-8").write("**VERDICT: ACCEPT** (selftest fixture, not a review)\n")


def impl_on():
    FT.A36_IMPL_FILE = _TI_REL + "/A36.impl.json"
    FT.A36_REQUIRED_REVIEWS = dict(_ORIG_REVIEWS, **{FT.A36_FINAL_REVIEW: _TI_REL + "/final-review.txt"})


def impl_off():
    FT.A36_IMPL_FILE, FT.A36_REQUIRED_REVIEWS = _ORIG_IMPL_FILE, dict(_ORIG_REVIEWS)


def make_rec(mut=None, reseal=True):
    """Write a record the way write_impl_record.py does (minimal fields) -> its file sha. impl_on() must be active."""
    srec = FT.a36_signature(ROOT)[1]
    rec = {"schema": FT.A36_IMPL_SCHEMA, "amendment": "A36", "signature_key": "A36", "approval_key": FT.A36_IMPL_KEY, "a36_file": FT.A36_FILE,
           "a36_file_sha256": srec["file_sha256"], "base_commit": FT.A36_BASE_COMMIT, "written": "2026-10-16T00:30:00",
           "files": {r: {"before": FT._git_blob_sha(FT.A36_BASE_COMMIT, r), "after": FT._sha(ROOT, r)} for r in FT.A36_CODE_SCOPE},
           "reviews": [{"id": rid, "file": rel, "sha256": FT._sha(ROOT, rel),
                        "verdict": FT._review_verdict(open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8").read())}
                       for rid, rel in FT.A36_REQUIRED_REVIEWS.items()],
           "reviewed_commits": [{"commit": FT.A36_GATES_FIX_COMMIT}],
           "supporting_artifacts_informational": {"files": {"sim/final_test/acceptance_v2/run/chain_gates.sh": "0" * 64}}}
    rec["sha256"] = C.self_sha(rec)
    if mut:
        mut(rec)
        if reseal:
            rec["sha256"] = C.self_sha(rec)
    p = os.path.join(ROOT, *FT.A36_IMPL_FILE.split("/"))
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    return F._sha_file(p)


def approval(fsha, **over):
    e = {"by": "Arash", "date": "2026-10-16", "said": "selftest approval (in memory only)", "prereg_sha256": F._sha_file(C.paths(None)["prereg"]),
         "file": FT.A36_IMPL_FILE, "file_sha256": fsha, "authorized_at": "2026-10-16T00:05:00+03:30",
         "record_created_at": "2026-10-16T00:30:00", "authorization_preceded_record_creation": True,
         "authorization_kind": "pre-approval (selftest fixture)"}
    e.update(over)
    return dict(SIG_REAL, **{FT.A36_IMPL_KEY: e})


# ------------------------------------------------------------------------------------------------ 3 the ft-v2 build
print("== 3 build_ft2 (read-only)")
fz = FT.build_ft2(ROOT, probes=PROBES)
check("3.1 build_ft2 wrote nothing (freeze.json unchanged)", F._sha_file(FZ_PATH) == FZ_SHA0)
def _impl_sha(root):                                     # K23 (test-only): snapshot of the real A36 record for 6.12a
    p = os.path.join(root, *_ORIG_IMPL_FILE.split("/"))
    return F._sha_file(p) if os.path.isfile(p) else None


IMPL0 = _impl_sha(ROOT)
IMP_REAL = FT.a36_impl_check(ROOT)                       # until Arash writes + approves the record, 12.1 / registration fail on the A36 files
impl_on()
try:
    SIGS_A = approval(make_rec())
    fzA = FT.build_ft2(ROOT, probes=PROBES, signatures=SIGS_A)   # the same tree with a valid temporary record + approval
finally:
    impl_off()
FZ_OK = fz if IMP_REAL["ok"] else fzA


def a36_only_fail(g):
    """On the real tree without an approved record: the gate fails ONLY because the re-freeze exception is void on the A36 files."""
    v = fz["gates"][g]
    if IMP_REAL["ok"]:
        return FT.gate_ok(g, v)
    rs = v["evidence"].get("refreeze_exception_reasons") or v["evidence"].get("reasons") or []
    return v["status"] == "FAIL" and FT.gate_ok(g, fzA["gates"][g]) and any("A36 implementation record" in x for x in rs)
check("3.2 no ft-v1-only field survives (fs_blocks, channels, ft-v1 prompt, blocks_ns, ft-v1 tokens / a12)",
      not any(k in fz for k in ("fs_blocks_sha256", "channels", "prompt_version", "blocks_ns", "blocks_sha", "tokens", "a12", "system_text_sha", "detector_types")))
check("3.3 profile ft-v2; treatment = 7a2ee211 = the acceptance identity", fz["freeze_profile"] == "ft-v2" and fz["treatment"]["sha"].startswith("7a2ee211")
      and fz["gates"]["12.1"]["evidence"]["treatment_test_v1_equals_acceptance"] is True)
check("3.4 gate order = the ft-v2 list; every gate has a status (WAIVED only for the acceptance gate, A36)", list(fz["gates"]) == list(FT.GATE_ORDER_FT2)
      and all(v["status"] in ("PASS", "FAIL", "PENDING", "NOT_APPLICABLE") or (g == "acceptance" and v["status"] == "WAIVED")
              for g, v in fz["gates"].items()))
sn = fz["snapshots"]
check("3.5 F1: the 4 practice-v2 snapshots + S* (= o0k10) with their own dir under snapshots_ft2 and their registered role",
      sorted(sn) == ["S*", "o0k1", "o0k10", "o0k3", "o0k5"] and all(v["dir"].startswith("sim/final_test/snapshots_ft2/o0/k") for v in sn.values())
      and sn["S*"]["files"] == sn["o0k10"]["files"] and "S*" in sn["o0k10"]["role"] and "curve" in sn["o0k1"]["role"])
check("3.6 F1: snapshot gate passes (nesting, integrity, leakage, A31 equivalence bound to treatment.py + the practice journal)",
      fz["gates"]["snapshot"]["status"] == "PASS" and fz["practice_equivalence"]["ok"], fz["gates"]["snapshot"])
check("3.7 F1: the K = 10 manifest equals the registration's frozen ref", sn["o0k10"]["files"]["manifest.json"] ==
      json.load(open(os.path.join(ROOT, "docs", "final-test", "acceptance_registration.json"), encoding="utf-8"))["frozen_refs"]["files"]["sim/final_test/snapshots_ft2/o0/k10/manifest.json"])
pr = fz["prompt"]
check("3.8 F2: prompt version ft-v2; 49 detector classes, 48 shown, Knife excluded; 12.12 (vs the REGISTERED class hash) and 12.11 (AST callers + "
      "effective contract) PASS with no reason (Astra r2 findings closed in cycle 3)",
      pr["version"] == "ft-v2" and pr["types"]["detector_n"] == 49 and pr["types"]["shown_n"] == 48 and pr["types"]["excluded"] == ["Knife"]
      and fz["gates"]["12.12"]["status"] == "PASS" and fz["gates"]["12.11"]["status"] == "PASS"
      and not fz["gates"]["12.12"]["evidence"]["reasons"] and fz["gates"]["12.12"]["evidence"]["registered"]["ordered_sha"].startswith("db08b02f")
      and fz["gates"]["12.11"]["evidence"]["one_configuration"] and not fz["gates"]["12.11"]["evidence"]["stray_render_callers"],
      (fz["gates"]["12.12"], fz["gates"]["12.11"]))
check("3.8b no gate is held any more (ASTRA_R2_HELD empty in both modules); 12.2 / 12.3 / 12.7 carry their cycle-3 evidence",
      GE.ASTRA_R2_HELD == {} and FT.ASTRA_R2_HELD == {} and all(fz["gates"][g]["status"] in ("PASS", "FAIL") for g in ("12.2", "12.3", "12.7")),
      {g: fz["gates"][g]["status"] for g in ("12.2", "12.3", "12.7")})
check("3.9 F4: A12 is NOT_APPLICABLE with the registered basis (never a silent pass)", fz["gates"]["A12"]["status"] == "NOT_APPLICABLE"
      and fz["gates"]["A12"]["evidence"]["basis"]["prereg_line_sha"] and fz["gates"]["A12"]["evidence"]["basis"]["capture_audit"][0].startswith("not applicable"))
check("3.10 F3: tokens / context / leakage decided by their final-identity part (acceptance-v2-a2, authenticated), never PASS without it",
      all(fz["gates"][g]["status"] == ((json.load(open(os.path.join(REAL_DIR, g + ".json"), encoding="utf-8")).get("final_identity") or {}).get("status") or "PENDING")
          for g in ("tokens", "context", "leakage")))
check("3.11 F6: fb-1 reproduced from the pinned runner", fz["fb1_feedback_contract"]["ok"] and fz["gates"]["fb-1"]["status"] == "PASS")
check("3.12 F6: the acceptance registration recorded (file + self sha, frozen refs current, signature PENDING); without an approved A36 record "
      "it fails ONLY on the void re-freeze exception and passes with a valid record + approval",
      FZ_OK["acceptance_registration"]["ok"] and FZ_OK["acceptance_registration"]["signature_status"] == "PENDING ARASH"
      and FZ_OK["gates"]["registration"]["status"] == "PASS" and a36_only_fail("registration"), (fz["gates"]["registration"], fzA["gates"]["registration"]))
check("3.13 F6: the C11 audit recorded", fz["c11_audit"]["ok"] and fz["c11_audit"]["sha256"])
import analyze as AN                                     # noqa: E402
import energy as EN                                      # noqa: E402
import hashlib as _hl                                    # noqa: E402
import tempfile as _tf                                   # noqa: E402
_c0 = dict(AN.A24_2_COST)
_em = AN.A24_2_COST["energy_method"]
_dec = dict(_c0, energy_method=dict(_em, unmeasured_rule_basis="declined"))       # what Arash's 'declined' would record
check("3.14 F6: a24_2_cost = the analysis plan's numbers with the D13 basis amendment:A33 bound to its prereg heading and the C5-SIMPLE "
      "energy method (energy.METHOD_ID) bound to its real measurement (journals unchanged, every call joined); the unmeasured practice rows by "
      "Arash's ledger decision (arash-decision:2026-10-14, the approved terms) = PASS; the sealed analysis accepts the record",
      all(fz["a24_2_cost"][k] == AN.A24_2_COST[k] for k in ("usd_per_kwh", "hardware_price_usd", "operating_hours", "hardware_time_factor"))
      and fz["a24_2_cost"]["hardware_price_basis"] == "amendment:A33" and fz["a24_2_cost"]["basis_evidence"]["amendment"] == "A33"
      and len(fz["a24_2_cost"]["basis_evidence"]["prereg_heading_sha256"]) == 64
      and fz["a24_2_cost"]["energy_method"]["id"] == EN.METHOD_ID and fz["a24_2_cost"]["energy_evidence"]["ok"] is True
      and len(fz["a24_2_cost"]["energy_evidence"]["sha256"]) == 64 and fz["a24_2_cost"]["energy_evidence"]["cells"] >= 1
      and fz["gates"]["a24_2_cost"]["status"] == "PASS" and fz["gates"]["a24_2_cost"]["evidence"]["reasons"] == []
      and _em["unmeasured_rule_basis"] == "arash-decision:2026-10-14" and fz["a24_2_cost"]["unmeasured_rule"]["by"] == "Arash"
      and AN.check_cost_freeze({"a24_2_cost": fz["a24_2_cost"]})["recorded"] is True, fz["gates"]["a24_2_cost"])
# a scratch root: copies of the calibration, calibrate.py and the prereg (+ a synthetic C5 amendment); Arash decision files written here only
_td = _tf.mkdtemp(prefix="c5fz_")
_R = C.paths()["root"]
_ev = json.load(open(os.path.join(_R, _em["evidence"]), encoding="utf-8"))
shutil.copytree(os.path.join(_R, "docs", "final-test", "energy"), os.path.join(_td, "docs", "final-test", "energy"))
_pj = AN.UNMEASURED_TERMS_APPROVED["rows"]["journal"]                               # the bound practice-v2 journal (Astra C5 r5 #1)
os.makedirs(os.path.dirname(os.path.join(_td, *_pj.split("/"))), exist_ok=True)
shutil.copy(os.path.join(_R, *_pj.split("/")), os.path.join(_td, *_pj.split("/")))
_pre = os.path.join(_td, "docs", "final-test", "preregistration.md")
shutil.copy(C.paths()["prereg"], _pre)
with open(_pre, "a", encoding="utf-8") as _fh:
    _T_OK = dict(AN.UNMEASURED_TERMS_APPROVED)
    _T_WIDE = dict(_T_OK, eligible=["practice", "no_energy_row"])
    _fh.write("\n### A99 (selftest only) -- C5 unmeasured rows\nunmeasured_rule_basis = amendment:A99 (estimated at the measured rate).\n"
              "C5-UNMEASURED-TERMS: " + json.dumps(_T_OK, sort_keys=True) + "\n")
    _fh.write("\n### A97 (selftest only) -- C5 unmeasured rows, wider section\nC5-UNMEASURED-TERMS: " + json.dumps(_T_WIDE, sort_keys=True) + "\n")
    _fh.write("\n### A96 (selftest only) -- C5 unmeasured rows without a terms line\nunmeasured_rule_basis = amendment:A96; 1 W.\n")
    _fh.write("\n### A95 (selftest only) -- C5 two terms lines\nC5-UNMEASURED-TERMS: " + json.dumps(_T_OK, sort_keys=True) + "\n"
              "C5-UNMEASURED-TERMS: " + json.dumps(_T_WIDE, sort_keys=True) + "\n")
_sec = FT._prereg_section(_pre, "A99")[1]
_ssha = lambda aid: _hl.sha256(FT._prereg_section(_pre, aid)[1].encode("utf-8")).hexdigest()     # noqa: E731
_decp = os.path.join(_td, "docs", "final-test", FT.C5_DECISION_FILE)
_evp = os.path.join(_td, _em["evidence"])


def _decide(doc):
    if doc is None:
        if os.path.exists(_decp):
            os.remove(_decp)
    else:
        with open(_decp, "w", encoding="utf-8") as _fh:
            json.dump(doc, _fh)


_said = {"by": "Arash", "date": "2026-10-14", "said": "selftest only"}
_amd = dict(_said, decision="amendment", amendment="A99", amendment_section_sha256=_hl.sha256(_sec.encode("utf-8")).hexdigest(),
            terms=_T_OK, eligible_rows="practice-v2 rows only", effect_on_claim="selftest")
_um = lambda b, t=_T_OK: dict(_c0, energy_method=dict(_em, unmeasured_rule_basis=b, unmeasured_rule_terms=t))       # noqa: E731
_decide(dict(_said, decision="declined"))
_r_fake = FT.a24_2_record(_td, dict(_dec, hardware_price_basis="amendment:A98"))
_r_pend = FT.a24_2_record(_td, dict(_dec, hardware_price_basis="reference_device_pending"))
_r_junk = FT.a24_2_record(_td, dict(_dec, hardware_price_basis="jetson"))
_r_ok = FT.a24_2_record(_td, _dec)
_r_lap = FT.a24_2_record(_td, dict(_dec, hardware_price_basis="laptop_purchase_price"))
_r_und = FT.a24_2_record(_td, _um(None, None))                                 # undecided (no basis)
_r_decmis = FT.a24_2_record(_td, _um("amendment:A99"))                             # file says declined, plan says amendment
_decide(dict(_said, decision="declined", said=""))
_r_nosaid = FT.a24_2_record(_td, _dec)
_decide(None)
_r_nofile = FT.a24_2_record(_td, _dec)
_decide(_amd)
_r_ua = FT.a24_2_record(_td, _um("amendment:A99"))
_r_a33 = FT.a24_2_record(_td, _um("amendment:A33"))
_r_a32 = FT.a24_2_record(_td, _um("amendment:A32"))
_decide(dict(_amd, amendment_section_sha256="0" * 64))
_r_ush = FT.a24_2_record(_td, _um("amendment:A99"))
_decide(dict(_amd, terms=_T_WIDE))                                                  # the record approves more than the plan
_r_uw = FT.a24_2_record(_td, _um("amendment:A99"))
_decide(_amd)
_r_pw = FT.a24_2_record(_td, _um("amendment:A99", _T_WIDE))                        # the plan wider than the section + record
_r_pn = FT.a24_2_record(_td, _um("amendment:A99", None))                           # basis without terms
_r_pr = FT.a24_2_record(_td, _um("amendment:A99", dict(_T_OK, rate="assumed_w")))  # another rate
# the bound journal is READ: missing, changed bytes, or rows that do not match row_ids -> FAIL
_pjf = os.path.join(_td, *_pj.split("/"))
_pjb = open(_pjf, "rb").read()
os.rename(_pjf, _pjf + ".away")
_r_jmiss = FT.a24_2_record(_td, _um("amendment:A99"))
os.rename(_pjf + ".away", _pjf)
with open(_pjf, "ab") as _fh:
    _fh.write(b"\n")                                                              # same rows, other bytes
_r_jsha = FT.a24_2_record(_td, _um("amendment:A99"))
_lines = [l for l in _pjb.split(b"\n") if l.strip()]
with open(_pjf, "wb") as _fh:
    _fh.write(b"\n".join(_lines[:-1]) + b"\n")                                    # one approved row removed
_T_9 = dict(_T_OK, rows=dict(_T_OK["rows"], journal_sha256=_hl.sha256(open(_pjf, "rb").read()).hexdigest()))
_r_jids = AN.practice_journal_problems(_td, _T_9)                               # right sha, but 9 rows vs 10 ids
with open(_pjf, "wb") as _fh:
    _fh.write(_pjb)
_decide(_amd)
_r_jok = FT.a24_2_record(_td, _um("amendment:A99"))
_r_fake_j = AN.practice_journal_problems(_td, dict(_T_OK, rows=dict(_T_OK["rows"], journal="docs/final-test/no-such.jsonl")))
_r_zero = AN.practice_journal_problems(_td, dict(_T_OK, rows=dict(_T_OK["rows"], journal_sha256="0" * 64)))
_r_up = AN.practice_journal_problems(_td, dict(_T_OK, rows=dict(_T_OK["rows"], journal="../outside.jsonl")))
with open(_pjf, "ab") as _fh:
    _fh.write(b"null\n")                                                          # a non-object row: FAIL, never an exception
try:
    _r_null = FT.a24_2_record(_td, _um("amendment:A99"))
except Exception as _e:                                                          # noqa: BLE001
    _r_null = (None, f"raised {type(_e).__name__}", [])
with open(_pjf, "wb") as _fh:
    _fh.write(_pjb)
_decide(dict(_amd, amendment="A97", amendment_section_sha256=_ssha("A97"), terms=_T_WIDE))
_r_97 = FT.a24_2_record(_td, _um("amendment:A97"))                                 # section + record wider than the plan
_decide(dict(_amd, amendment="A96", amendment_section_sha256=_ssha("A96")))
_r_96 = FT.a24_2_record(_td, _um("amendment:A96"))                                 # a section with prose only ('1 W'), no terms line
_decide(dict(_amd, amendment="A95", amendment_section_sha256=_ssha("A95")))
_r_95 = FT.a24_2_record(_td, _um("amendment:A95"))                                 # two terms lines
_decide(dict(_said, decision="declined"))
_r_none = FT.a24_2_record(_td, dict(_dec, energy_method=None))
_r_str = FT.a24_2_record(_td, dict(_dec, energy_method="nvidia-smi board + CPU package"))
_r_id = FT.a24_2_record(_td, dict(_dec, energy_method=dict(_em, id="c5-energy-1", unmeasured_rule_basis="declined")))
_r_noev = FT.a24_2_record(_td, dict(_dec, energy_method=dict(_em, evidence="docs/final-test/energy/no-such-file.json", unmeasured_rule_basis="declined")))


def _cal(doc):
    with open(_evp, "w", encoding="utf-8") as _fh:
        json.dump(doc, _fh)
    return FT.a24_2_record(_td, _dec)


_r_copy = _cal(_ev)
_r_ad_wide = FT.a24_2_record(_td, _um("arash-decision:2026-10-14", _T_WIDE))      # the decision branch with terms wider than approved
_r_ad_ok = FT.a24_2_record(_td, _um("arash-decision:2026-10-14"))
_r_stale = _cal(dict(_ev, energy_py_sha256="0" * 64))
_r_schema = _cal(dict(_ev, schema="ftm.c5-calibration/1"))
_r_notok = _cal(dict(_ev, ok=False))
_r_part = _cal(dict(_ev, join=dict(_ev["join"], complete=False)))
_r_nojr = _cal(dict(_ev, journals={}))
_cal(_ev)
_jr = next(k for k in _ev["journals"] if "/energy." in k)
with open(os.path.join(_td, _jr), "a", encoding="utf-8") as _fh:
    _fh.write("\n")
_r_jrn = _cal(_ev)
shutil.rmtree(_td, ignore_errors=True)
_has = lambda r, s: any(s in x for x in r[2])            # noqa: E731
check("3.14c C5-SIMPLE: no energy method = PENDING; a free-text or other-id method = FAIL; missing measurement evidence = PENDING; the "
      "measurement is checked from its content (an unchanged copy = PASS; another energy.py, another schema, ok false, an incomplete join, no "
      "journals, a changed journal = PENDING)",
      _r_none[1] == "PENDING" and "energy method" in _r_none[2][0] and _r_str[1] == "FAIL" and _r_id[1] == "FAIL"
      and _r_noev[1] == "PENDING" and _has(_r_noev, "missing") and _r_copy[1] == "PASS"
      and _r_stale[1] == "PENDING" and _has(_r_stale, "energy.py changed after") and _r_schema[1] == "PENDING" and _has(_r_schema, "schema")
      and _r_notok[1] == "PENDING" and _has(_r_notok, "did not measure") and _r_part[1] == "PENDING" and _has(_r_part, "did not measure")
      and _r_nojr[1] == "PENDING" and _has(_r_nojr, "no journals") and _r_jrn[1] == "PENDING" and _has(_r_jrn, "missing or changed"),
      (_r_none[1:], _r_str[1:], _r_id[1:], _r_noev[1:], _r_copy[1:], _r_stale[1:], _r_schema[1:], _r_notok[1:], _r_part[1:], _r_nojr[1:], _r_jrn[1:]))
check("3.14c2 C5-SIMPLE: the 'arash-decision:' basis passes only with exactly the approved terms (wider terms = FAIL)",
      _r_ad_ok[1] == "PASS" and _r_ad_ok[0]["unmeasured_rule"]["by"] == "Arash" and _r_ad_wide[1] == "FAIL",
      (_r_ad_ok[1:], _r_ad_wide[1:]))
check("3.14d C5 unmeasured rule = an Arash decision record, never a heading: undecided = PENDING (ARASH DECISION); 'declined' with his record = "
      "PASS; a basis without the record = PENDING; a record without his words or with another decision = FAIL; amendment:A33 (the hardware-price "
      "amendment) and a non-C5 section (A32) = FAIL; a C5 section bound by its sha whose C5-UNMEASURED-TERMS line = the plan's terms = the "
      "record's terms, with the bound practice journal on disk matching its sha and row ids = PASS; a missing journal, changed journal "
      "bytes, a journal whose rows differ from row_ids, a path outside the repo; a wrong section sha, a record wider than the plan, a plan wider than the section, a plan without terms or with "
      "another rate, a section + record wider than the plan, a section with prose only ('1 W') and a section with two terms lines = FAIL",
      _r_und[1] == "PENDING" and _r_und[2][0].startswith("ARASH DECISION") and _r_ok[1] == "PASS"
      and _r_ok[0]["unmeasured_rule"]["rule"].startswith("declined") and _r_ok[0]["unmeasured_rule"]["by"] == "Arash"
      and _r_nofile[1] == "PENDING" and _has(_r_nofile, "does not exist") and _r_nosaid[1] == "FAIL" and _r_decmis[1] == "FAIL"
      and _r_ua[1] == "PASS" and _r_ua[0]["unmeasured_rule"]["amendment"] == "A99"
      and len(_r_ua[0]["unmeasured_rule"]["amendment_section_sha256"]) == 64
      and _r_a33[1] == "FAIL" and _has(_r_a33, "hardware-price amendment") and _r_a32[1] == "FAIL" and _has(_r_a32, "not a C5")
      and _r_ua[0]["unmeasured_rule"]["terms"] == _T_OK
      and _r_ush[1] == "FAIL" and _r_uw[1] == "FAIL" and _has(_r_uw, "terms")
      and _r_pw[1] == "FAIL" and _has(_r_pw, "C5-UNMEASURED-TERMS line") and _r_pn[1] == "FAIL" and _has(_r_pn, "unmeasured_rule_terms missing")
      and _r_pr[1] == "FAIL" and _has(_r_pr, "rate") and _r_97[1] == "FAIL" and _has(_r_97, "C5-UNMEASURED-TERMS line")
      and _r_96[1] == "FAIL" and _has(_r_96, "not a C5 unmeasured-rule amendment") and _r_95[1] == "FAIL" and _has(_r_95, "exactly one")
      and _r_jok[1] == "PASS" and _r_jmiss[1] == "FAIL" and _has(_r_jmiss, "not an existing file") and _r_jsha[1] == "FAIL"
      and _has(_r_jsha, "journal_sha256") and any("row ids" in w for w in _r_jids) and any("not an existing file" in w for w in _r_fake_j)
      and any("journal_sha256" in w for w in _r_zero) and any("not an existing file" in w for w in _r_up)
      and _r_null[1] == "FAIL" and _has(_r_null, "not a JSON object"),
      (_r_und[1:], _r_ok[1:], _r_nofile[1:], _r_nosaid[1:], _r_decmis[1:], _r_ua[1:], _r_a33[1:], _r_a32[1:], _r_ush[1:], _r_uw[1:],
       _r_pw[1:], _r_pn[1:], _r_pr[1:], _r_97[1:], _r_96[1:], _r_95[1:], _r_jok[1:], _r_jmiss[1:], _r_jsha[1:], _r_jids, _r_fake_j, _r_zero,
       _r_up, _r_null[1:]))
check("3.14b D13 basis: an amendment id without a prereg heading = FAIL; an undecided basis = PENDING (D13); a malformed basis = FAIL; "
      "A33 + an energy method (+ a decided unmeasured rule) = PASS (and the sealed analysis accepts it); laptop price = PASS",
      _r_fake[1] == "FAIL" and _r_pend[1] == "PENDING" and "D13" in _r_pend[2][0] and _r_junk[1] == "FAIL"
      and _r_ok[1] == "PASS" and _r_ok[0]["status"] == "recorded" and AN.check_cost_freeze({"a24_2_cost": _r_ok[0]}, _dec)["recorded"] is True
      and _r_lap[1] == "PASS", (_r_fake[1:], _r_pend[1:], _r_junk[1:], _r_ok[1:], _r_lap[1:]))
_cf_bad = []
for _bad in (dict(_c0, energy_method=dict(_em, id="c5-energy-1")), dict(_c0, energy_method=dict(_em, unmeasured_rule_basis="amendment:A33"))):
    try:
        AN.check_cost_freeze({"a24_2_cost": fz["a24_2_cost"]}, _bad)
        _cf_bad.append("accepted")
    except AN.AnalysisError:
        _cf_bad.append("refused")
_amd_plan = _um("amendment:A99")
_cf_unbound = AN.check_cost_freeze({"a24_2_cost": dict(fz["a24_2_cost"], energy_method=_amd_plan["energy_method"])}, _amd_plan)
_cf_bound = AN.check_cost_freeze({"a24_2_cost": _r_ua[0]}, _amd_plan)
try:
    AN.check_cost_freeze({"a24_2_cost": _r_ua[0]}, _um("amendment:A99", _T_WIDE))      # frozen terms narrower than the analysed plan's
    _cf_bad.append("accepted")
except AN.AnalysisError:
    _cf_bad.append("refused")
check("3.14e the sealed analysis refuses a freeze record whose energy method id, unmeasured rule or unmeasured terms differ from the analysis plan's, and "
      "treats an amendment-priced plan without the bound unmeasured_rule as not recorded",
      _cf_bad == ["refused", "refused", "refused"] and _cf_unbound["recorded"] is False and _cf_bound["recorded"] is True, (_cf_bad, _cf_unbound, _cf_bound))
# FIX-B2 (stale state): Arash signed A24.1-C on 2026-10-14 (docs/final-test/signatures.json). The check no longer hard-codes "nothing
# signed": it re-derives, independently of freeze_ft2, which amendments carry explicit attributed evidence bound to the prereg on disk, and
# requires the freeze record to say exactly that (signed = his words verbatim; unsigned = None; PASS only when all are signed).
_sigp = os.path.join(C.paths(None)["docs"], "signatures.json")
_sigd = json.load(open(_sigp, encoding="utf-8")) if os.path.exists(_sigp) else {}
_pre315 = F._sha_file(C.paths(None)["prereg"])
_sigd = {a: (_sigd.get(a) if isinstance(_sigd.get(a), dict) else next((v for k, v in _sigd.items() if isinstance(v, dict) and a in k.split("+")), None))
         for a in FT.SIGNED_AMENDMENTS}                           # a combined record ("A30+A31+A32+A33") signs each member
_exp_signed = {a for a in FT.SIGNED_AMENDMENTS if isinstance(_sigd.get(a), dict) and _sigd[a].get("by") == "Arash" and _sigd[a].get("date")
               and str(_sigd[a].get("said") or "").strip() and _sigd[a].get("prereg_sha256") == _pre315}
_exp_unsigned = sorted(set(FT.SIGNED_AMENDMENTS) - _exp_signed)
check("3.15 F6: A30 / A31 / A32 / A33 / A24.1-C signatures come ONLY from Arash's attributed records bound to the current prereg (never filled "
      "by code): the record matches an independent re-derivation; PENDING while any is unsigned",
      set(fz["signatures"]) == set(FT.SIGNED_AMENDMENTS) and fz["gates"]["signatures"]["evidence"]["unsigned"] == _exp_unsigned
      and fz["gates"]["signatures"]["status"] == ("PENDING" if _exp_unsigned else "PASS")
      and all((v["signature"] is None) == (a not in _exp_signed) for a, v in fz["signatures"].items())
      and all(v["signature"] == {k: _sigd[a][k] for k in ("by", "date", "said")} for a, v in fz["signatures"].items() if a in _exp_signed),
      (fz["gates"]["signatures"]["evidence"]["unsigned"], _exp_unsigned))
check("3.16 F7: namespaces incl. acceptance-v2 (attempt 8, study guard, roster items, ft-v2)", fz["namespaces"]["acceptance-v2"] ==
      {"total": 16.0, "attempt_ceiling": 8.0, "study_guard": True, "items": "acceptance_roster", "treatment": "ft-v2"}
      and fz["gates"]["12.13"]["status"] == "PASS")
check("3.17 F9: FS-v2 artefacts + writer / generator code identities recorded; 12.1 keeps the required pins",
      "fs_v2_corpus (S* example store)" in fz["artefacts"] and all(v["sha256"] for v in fz["artefacts"].values())
      and all(fz["writers"].values()) and FZ_OK["gates"]["12.1"]["status"] == "PASS" and not FZ_OK["gates"]["12.1"]["evidence"]["pins_failing"]
      and a36_only_fail("12.1"), (fz["gates"]["12.1"]["evidence"].get("pins_failing"), fzA["gates"]["12.1"]["status"]))
check("3.18 all_gates_pass is false while anything is PENDING or FAILS", fz["all_gates_pass"] is (not fz["status_summary"]["PENDING"] and not fz["status_summary"]["FAIL"]))
check("3.19 nothing FAILS on current evidence (except the acceptance precondition while A36 is unsigned, and 12.1 / registration only while "
      "the A36 implementation record is not written + approved -- and then only on the void re-freeze exception)",
      [g for g in fz["status_summary"]["FAIL"] if not (g == "acceptance" and not FT.a36_signature(ROOT)[0])
       and not (g in ("12.1", "registration") and a36_only_fail(g))] == []
      and not [g for g in fzA["status_summary"]["FAIL"] if not (g == "acceptance" and not FT.a36_signature(ROOT)[0])],
      (fz["status_summary"]["FAIL"], fzA["status_summary"]["FAIL"]))
sealed = F.seal(copy.deepcopy(fz))
check("3.20 the sealed preview carries no key-like text and verifies against freeze_artifact_problems (snapshot dirs + artefacts)",
      not C.key_leak(json.dumps(sealed)) and C.freeze_artifact_problems(sealed, ROOT) == [], C.freeze_artifact_problems(sealed, ROOT))
check("3.21 snapshot_freeze_problems reads the ft-v2 dir for S*", C.snapshot_freeze_problems(["S*"], ROOT, sealed) == [])
t = copy.deepcopy(sealed)
t["snapshots"]["S*"]["files"]["ltm.db"] = "3" * 64
check("3.22 a changed frozen snapshot file is detected through its recorded dir", any("snapshot S*/ltm.db" in x for x in C.snapshot_freeze_problems(["S*"], ROOT, t)))
t = copy.deepcopy(sealed)
t["artefacts"]["fs_block.py"]["sha256"] = "4" * 64
check("3.23 a changed frozen artefact is detected", any("artefact fs_block.py" in x for x in C.freeze_artifact_problems(t, ROOT)))

# ------------------------------------------------------------------------------------------------ 3b negative cases on copies (Astra ACC-RUNNER Q6)
print("== 3b snapshots, fb-1, signatures: negative cases")
TR = tempfile.mkdtemp(prefix="fzft2s_", dir=os.path.join(ROOT, "sim", "final_test", "_checkpoints", "ACC-RUNNER"))
try:
    for k in (1, 3, 5, 10):
        shutil.copytree(os.path.join(ROOT, "sim", "final_test", "snapshots_ft2", "o0", f"k{k}"), os.path.join(TR, "sim", "final_test", "snapshots_ft2", "o0", f"k{k}"))
    os.makedirs(os.path.join(TR, "sim", "final_test", "diag", "a31"))
    shutil.copy(os.path.join(ROOT, "sim", "final_test", "diag", "a31", "equiv_replay.json"), os.path.join(TR, "sim", "final_test", "diag", "a31"))
    sn0, why0 = FT.snapshots_ft2(TR)
    check("3b.1 an unchanged copy of the 4 snapshots verifies (hashes vs manifests, prefix nesting)", why0 == [], why0)
    with open(os.path.join(TR, "sim", "final_test", "snapshots_ft2", "o0", "k5", "ltm.db"), "ab") as f:
        f.write(b"\0")
    sn1, why1 = FT.snapshots_ft2(TR)
    check("3b.2 a changed ltm.db is caught against manifest.file_sha256", any("k5: ltm.db differs" in x for x in why1), why1)
    rows3 = os.path.join(TR, "sim", "final_test", "snapshots_ft2", "o0", "k3", "practice_rows.jsonl")
    lines = open(rows3, encoding="utf-8").read().splitlines()
    open(rows3, "w", encoding="utf-8").write("\n".join([lines[1], lines[0]] + lines[2:]) + "\n")
    sn2, why2 = FT.snapshots_ft2(TR)
    check("3b.3 reordered practice rows break the rows sha and the prefix nesting", any("k3: practice_rows.jsonl differs" in x for x in why2)
          and any("prefix" in x for x in why2), why2)
    eq = FT.equivalence_ft2(TR, sn1)
    check("3b.4 the A31 certificate no longer matches a changed snapshot directory", not eq["ok"] and any("directory sha" in x for x in eq["reasons"]), eq)
    from planners import runner as _RU
    orig_cf = _RU.cap_feedback
    _RU.cap_feedback = lambda text, cap=600: orig_cf(text, cap=599)
    try:
        fb = FT.fb1_record(ROOT)
    finally:
        _RU.cap_feedback = orig_cf
    check("3b.5 fb-1: a changed cap_feedback behaviour misses the fixed vectors", not fb["ok"] and any("fixed fb-1 vector" in x for x in fb["reasons"]))
    os.makedirs(os.path.join(TR, "docs", "final-test"))
    shutil.copy(C.paths(None)["prereg"], os.path.join(TR, "docs", "final-test", "preregistration.md"))
    pre = F._sha_file(os.path.join(TR, "docs", "final-test", "preregistration.md"))
    json.dump({"A30": {"by": "Arash", "date": "2026-10-14", "said": "signed", "prereg_sha256": "0" * 64},
               "A31": {"by": "Claude", "date": "2026-10-14", "said": "x", "prereg_sha256": pre},
               "A32": {"by": "Arash", "date": "2026-10-14", "said": "signed", "prereg_sha256": pre}},
              open(os.path.join(TR, "docs", "final-test", "signatures.json"), "w", encoding="utf-8"))
    sg = FT.signatures_record(TR)
    check("3b.6 signatures need explicit attributed evidence bound to the current prereg (wrong sha / wrong person = PENDING)",
          sg["A30"]["status"] == "PENDING ARASH" and sg["A31"]["status"] == "PENDING ARASH" and sg["A32"]["status"] == "recorded")
    json.dump({}, open(os.path.join(TR, "docs", "final-test", "signatures.json"), "w", encoding="utf-8"))
    sg0 = FT.signatures_record(TR)
    os.remove(os.path.join(TR, "docs", "final-test", "signatures.json"))
    sg_none = FT.signatures_record(TR)
    json.dump({"A24.1-C": {"by": "Arash", "date": "2026-10-14", "said": "", "prereg_sha256": pre},
               "A33": {"by": "Arash", "date": "", "said": "signed", "prereg_sha256": pre}},
              open(os.path.join(TR, "docs", "final-test", "signatures.json"), "w", encoding="utf-8"))
    sg_e = FT.signatures_record(TR)
    check("3b.6b an empty or absent signatures file signs nothing; empty words or a missing date sign nothing (FIX-B2 negative controls for 3.15)",
          all(not v["signed"] and v["signature"] is None for d_ in (sg0, sg_none) for v in d_.values())
          and not sg_e["A24.1-C"]["signed"] and not sg_e["A33"]["signed"])
finally:
    shutil.rmtree(TR, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ 4 check / record / write paths (nothing written)
print("== 4 check_ft2, record_arash_check_ft2, the freeze.py dispatch")
orig_read, orig_write = F._read_json, F._atomic_write
writes = []
try:
    F._read_json = lambda p: copy.deepcopy(sealed) if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(FZ_PATH)) else orig_read(p)
    F._atomic_write = lambda p, o: writes.append(p)
    r = FT.check_ft2(ROOT, probes=PROBES)
    _allpass = bool(sealed.get("all_gates_pass"))                # state-aware: before the final evidence some gates are PENDING; at the freeze none
    check("4.1 check_ft2 on the just-built freeze: no diff; refused only for PENDING gates (while any) + no Arash check",
          r["diff"] == [] and any("Arash" in x for x in r["reasons"])
          and (r["reasons"] == ["Arash's freeze check not recorded"] if _allpass else any("gates not passed" in x for x in r["reasons"])), r)
    bad = copy.deepcopy(sealed)
    bad["dataset_sha256"] = "5" * 64
    F._read_json = lambda p: copy.deepcopy(bad) if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(FZ_PATH)) else orig_read(p)
    r = FT.check_ft2(ROOT, probes=PROBES)
    check("4.2 a hand-edited freeze fails the self sha and differs from the disk", any("self-sha" in x for x in r["reasons"]) and "dataset_sha256" in r["diff"])
    F._read_json = lambda p: copy.deepcopy(sealed) if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(FZ_PATH)) else orig_read(p)
    _pend = copy.deepcopy(sealed)                                # a freeze whose gates do not all pass (the real one, or a copy marked so)
    _pend["all_gates_pass"] = False
    _pend = F.seal({k: v for k, v in _pend.items() if k != "sha256"})
    F._read_json = lambda p: copy.deepcopy(_pend) if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(FZ_PATH)) else orig_read(p)
    try:
        FT.record_arash_check_ft2(ROOT, by="Arash", date="2026-10-14", said="test", probes=PROBES)
        rec_refused = False
    except RuntimeError:
        rec_refused = True
    F._read_json = lambda p: copy.deepcopy(sealed) if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(FZ_PATH)) else orig_read(p)
    check("4.3 Arash's check cannot be recorded while gates are PENDING", rec_refused and not writes)
    del writes[:]
    try:
        FT.record_arash_check_ft2(ROOT, by="Claude", date="2026-10-14", said="x")
        who = False
    except PermissionError:
        who = True
    check("4.4 only by='Arash' with his words", who and not writes)
finally:
    F._read_json, F._atomic_write = orig_read, orig_write
try:
    F.main(["--write", "--legacy-ft1"])
    leg = False
except SystemExit as e:
    leg = "refused" in str(e)
check("4.5 freeze.py --write --legacy-ft1 is refused while test-v1 runs ft-v2", leg)
src = open(os.path.join(HERE, "final_test", "matrix", "freeze.py"), encoding="utf-8").read()
check("4.6 freeze.py --write / --check dispatch to freeze_ft2 while TEST_V1_TREATMENT is ft-v2",
      "FT2.write_ft2(" in src and "FT2.check_ft2(" in src and "FT2.record_arash_check_ft2(" in src)
rep = FT.dry_run(ROOT, probes=PROBES)
check("4.7 dry_run leaves freeze.json untouched and reports PASS / NOT_APPLICABLE / PENDING / FAIL", rep["freeze_json_untouched"]
      and set(rep["status_summary"]) == {"PASS", "NOT_APPLICABLE", "PENDING", "FAIL", "WAIVED"})

# ------------------------------------------------------------------------------------------------ 5 A36: acceptance NOT PASS + the narrow waiver
print("== 5 acceptance precondition + the A36 waiver (signature bound to the exact A36 file sha)")
PRE = F._sha_file(C.paths(None)["prereg"])
A36SHA = FT._sha(ROOT, FT.A36_FILE)
GOOD = {"by": "Arash", "date": "2026-10-15", "said": "sign", "context": "selftest (in memory only)",
        "prereg_sha256": PRE, "file": FT.A36_FILE, "file_sha256": A36SHA}
SIG_DISK = F._sha_file(os.path.join(C.paths(None)["docs"], "signatures.json"))


def acc(sigs):
    return FT.acceptance_record(ROOT, sigs)


def never_pass(rec):
    s_ = json.dumps(rec)
    return rec["verdict"] == "NOT PASS" and '"acceptance_recorded_as": "PASS"' not in s_ and all(
        (a or {}).get("verdict") == "NOT PASS" for a in rec["attempts"].values())


r0, g0 = acc({})
check("5.1 the real verdicts: attempts 1 and 2 intact (bundle replay), both NOT PASS, attempt 2 terminal and the one A36 names",
      r0["attempts"]["1"]["intact"] and r0["attempts"]["2"]["intact"] and r0["terminal"] is True
      and r0["attempts"]["2"]["self_sha256"] == FT.A36_VERDICT2_SELF_SHA, r0["attempts"])
check("5.2 no A36 key -> the acceptance precondition FAILS (the freeze is refused); recorded NOT PASS", g0["status"] == "FAIL" and g0["pass"] is False
      and never_pass(r0) and any("A36 not signed" in x for x in g0["evidence"]["reasons"]), g0)
other = dict(GOOD)
other.pop("file_sha256")
r_, g_ = acc({"A34-A2+A35": dict(GOOD), "A30+A31+A32+A33": dict(GOOD), "A36-DRAFT": dict(GOOD), "A36 ": dict(GOOD)})
check("5.3 only the exact key 'A36' counts (other keys holding a valid-looking record do not)", g_["status"] == "FAIL", g_)
cases = {"wrong file sha": dict(GOOD, file_sha256="0" * 64), "no file sha": other, "file sha of A34-A2_A35.md":
         dict(GOOD, file_sha256=FT._sha(ROOT, "docs/final-test/amendments/A34-A2_A35.md")),
         "another prereg": dict(GOOD, prereg_sha256="1" * 64), "not Arash": dict(GOOD, by="Claude"), "no words": dict(GOOD, said=" "),
         "no date": dict(GOOD, date=""), "names another file": dict(GOOD, file="docs/final-test/amendments/A34-A2_A35.md")}
res5 = {k: acc({"A36": v})[1]["status"] for k, v in cases.items()}
check("5.4 a wrong / missing file sha, another prereg, another person, no words / date, another file named -> FAIL",
      all(v == "FAIL" for v in res5.values()), res5)
rw, gw = acc({"A36": dict(GOOD)})
w = rw["waiver"]
check("5.5 A36 signed for this exact file -> the gate is WAIVED (pass true), never PASS; the record keeps NOT PASS + both verdict shas + "
      "the waiver (key, file sha, 'waives only the acceptance-PASS precondition') + the analysis placeholder",
      gw["status"] == "WAIVED" and gw["pass"] is True and never_pass(rw) and w["key"] == "A36" and w["file_sha256"] == A36SHA
      and w["waives"] == "waives only the acceptance-PASS precondition" and w["acceptance_recorded_as"] == "NOT PASS"
      and w["attempt1_verdict_sha256"] == r0["attempts"]["1"]["file_sha256"] and w["attempt2_verdict_sha256"] == r0["attempts"]["2"]["file_sha256"]
      and set(w["revised_analysis_identity"]["files"]) == set(FT.A36_ANALYSIS_FILES) and w["revised_analysis_identity"]["files_sha256"], gw)
check("5.6 WAIVED counts as met only for the acceptance gate (any other gate WAIVED is not passed)",
      FT.gate_ok("acceptance", gw) and not FT.gate_ok("12.5", gw) and not FT.gate_ok("acceptance", dict(gw, **{"pass": False})))
_orig_file = FT.A36_FILE
TA = tempfile.mkdtemp(prefix="fzft2a36_", dir=os.path.join(ROOT, "sim", "final_test", "_checkpoints"))
A2 = FT._acc_attempt2()
_orig_vp = A2.verdict_path
try:
    alt = os.path.join(TA, "A36-DRAFT.md")
    open(alt, "wb").write(open(os.path.join(ROOT, *_orig_file.split("/")), "rb").read() + b"\n")
    FT.A36_FILE = os.path.relpath(alt, ROOT).replace(os.sep, "/")
    check("5.7 the A36 file edited after signing (sha differs) -> FAIL", acc({"A36": dict(GOOD)})[1]["status"] == "FAIL")
    FT.A36_FILE = _orig_file
    real2 = _orig_vp(2, ROOT)                                      # tampered verdicts are COPIES; the real files are never touched
    v2 = json.load(open(real2, encoding="utf-8"))

    def with_v2(doc):
        q = os.path.join(TA, "v2.json")
        json.dump(doc, open(q, "w", encoding="utf-8"), ensure_ascii=False)
        A2.verdict_path = lambda k, root=None: q if k == 2 else _orig_vp(k, root)
        try:
            return acc({"A36": dict(GOOD)})
        finally:
            A2.verdict_path = _orig_vp
    t1 = with_v2(dict(v2, protocol_stopped=False))
    t2 = with_v2(dict(v2, verdict="PASS", decision=dict(v2["decision"], verdict="PASS")))
    t3d = dict(v2, verdict="PASS", decision=dict(v2["decision"], verdict="PASS"))
    t3d["sha256"] = C.self_sha(t3d)
    t3 = with_v2(t3d)
    t4d = dict(v2, ceilings="x")
    t4d["sha256"] = C.self_sha(t4d)
    t4 = with_v2(t4d)
    check("5.8 a tampered attempt-2 verdict (edited field; edited to PASS; edited to PASS + re-sealed; another re-sealed verdict) -> FAIL even "
          "with A36 signed; never PASS", all(g["status"] == "FAIL" for _r, g in (t1, t2, t3, t4)) and all(not str(r["verdict"]).startswith("PASS")
                                                                                                    for r, _g in (t1, t2, t3, t4)),
          [(g["status"], r["verdict"]) for r, g in (t1, t2, t3, t4)])
    A2.verdict_path = lambda k, root=None: os.path.join(TA, "absent.json") if k == 2 else _orig_vp(k, root)
    try:
        t5 = acc({"A36": dict(GOOD)})
    finally:
        A2.verdict_path = _orig_vp
    check("5.9 attempt 2 missing (attempt 1 NOT PASS only) -> FAIL; the A36 waiver needs the terminal attempt-2 verdict",
          t5[1]["status"] == "FAIL" and t5[0]["waiver"] is None, t5[1])
finally:
    FT.A36_FILE = _orig_file
    A2.verdict_path = _orig_vp
    shutil.rmtree(TA, ignore_errors=True)
fz36 = FT.build_ft2(ROOT, probes=PROBES, signatures=dict(json.load(open(os.path.join(C.paths(None)["docs"], "signatures.json"), encoding="utf-8")), A36=dict(GOOD)))  # the real records (incl. A36-IMPL) + this A36
diff36 = {g: (fz["gates"][g]["status"], fz36["gates"][g]["status"]) for g in fz["gates"]
          if g != "acceptance" and fz["gates"][g]["status"] != fz36["gates"][g]["status"]}
others36 = [g for g, v in fz36["gates"].items() if g != "acceptance" and not FT.gate_ok(g, v)]
check("5.10 build_ft2 with A36 signed: acceptance WAIVED, every other gate unchanged, all_gates_pass still needs every other gate",
      fz36["gates"]["acceptance"]["status"] == "WAIVED" and not diff36 and fz36["all_gates_pass"] == (not others36)
      and fz36["status_summary"]["WAIVED"] == ["acceptance"], (diff36, others36))
check("5.11 the freeze record carries acceptance NOT PASS (both verdict shas) and the bound waiver; no acceptance field says PASS",
      never_pass(fz36["acceptance"]) and fz36["acceptance"]["waiver"]["file_sha256"] == A36SHA
      and fz["gates"]["acceptance"]["status"] == ("WAIVED" if FT.a36_signature(ROOT)[0] else "FAIL")      # disk: A36 signed -> WAIVED, else FAIL
      and never_pass(fz["acceptance"]) and "acceptance" not in fz36["status_summary"]["PASS"] and "acceptance" not in fz["status_summary"]["PASS"])
# ------------------------------------------------------------------------------------------------ 6 A36-IMPL: record, approval, succession
print("== 6 A36 implementation record + Arash's A36-IMPL approval + the freeze succession (Astra A36-IMPL-freeze-r1)")
A2m = FT._a2_module()
_orig_cs = A2m.code_state
r6, w6 = FT.a34_a36_exception(ROOT, SIG_REAL)
_unl = next((x for x in w6 if x.startswith("tracked files changed outside")), "")
check("6.1 no record: the exception is void and the reason lists the COMPLETE structured set (all 7 unlisted files, no truncation) + both off-pin files",
      r6 == {} and all(n in _unl for n in ("a36.py", "analyze.py", "common.py", "fg_runner.py", "freeze.py", "gate_evidence.py", "score.py"))
      and "(7)" in _unl and sum("not at its pinned sha" in x for x in w6) == 2 or IMP_REAL["ok"], w6)
impl_on()
try:
    fsha = make_rec()
    good = FT.a36_impl_check(ROOT, approval(fsha))
    rA, wA = FT.a34_a36_exception(ROOT, approval(fsha))
    _changed9 = ("sim/final_test/matrix/a36.py", "sim/final_test/matrix/analyze.py", "sim/final_test/matrix/common.py", "sim/final_test/matrix/fg_runner.py",
                 "sim/final_test/matrix/freeze.py", "sim/final_test/matrix/gate_evidence.py", "sim/final_test/matrix/score.py",
                 "sim/final_test/matrix/execute.py", "sim/final_test/matrix/freeze_ft2.py")
    check("6.2 a valid record + Arash's approval of its exact sha: accepted; the exception re-freezes every changed code file at its disk sha",
          good["ok"] and not wA and all(rA.get(r) == FT._sha(ROOT, r) for r in _changed9) and set(good["files"]) == set(FT.A36_CODE_SCOPE)
          and "sim/final_test/acceptance_v2/run/chain_gates.sh" not in rA, (good["reasons"], wA))
    no_appr = FT.a36_impl_check(ROOT, SIG_REAL)
    r_, w_ = FT.a34_a36_exception(ROOT, SIG_REAL)
    check("6.3 the record without an approval is rejected (exception void)", not no_appr["ok"] and r_ == {}
          and any("A36-IMPL" in x for x in no_appr["reasons"]), no_appr["reasons"])
    bad_appr = {"other sha": approval("0" * 64), "not Arash": approval(fsha, by="Claude"), "no words": approval(fsha, said=" "),
                "another file": approval(fsha, file="docs/final-test/amendments/A36.impl.json.bak"), "another prereg": approval(fsha, prereg_sha256="1" * 64),
                "no provenance": approval(fsha, authorization_preceded_record_creation=None),
                "other record time": approval(fsha, record_created_at="2026-10-17T00:00:00")}
    res6 = {k: FT.a36_impl_check(ROOT, v)["ok"] for k, v in bad_appr.items()}
    check("6.4 a mismatched approval (other sha / not Arash / no words / another file / another prereg / no provenance / another record time) is rejected", not any(res6.values()), res6)

    def m_add(rec):
        rec["files"]["sim/final_test/matrix/plan.py"] = {"before": FT._git_blob_sha(FT.A36_BASE_COMMIT, "sim/final_test/matrix/plan.py"),
                                                         "after": FT._sha(ROOT, "sim/final_test/matrix/plan.py")}

    def m_after(rec):
        rec["files"]["sim/final_test/matrix/score.py"]["after"] = "0" * 64

    def m_before(rec):
        rec["files"]["sim/final_test/matrix/score.py"]["before"] = FT._sha(ROOT, "sim/final_test/matrix/score.py")

    def m_drop(rec):
        rec["files"].pop("sim/final_test/matrix/freeze.py")

    def m_verdict(rec):
        rec["reviews"][1]["verdict"] = "ACCEPT"

    def m_commit(rec):
        rec["reviewed_commits"] = [{"commit": "937636d"}]

    def m_base(rec):
        rec["base_commit"] = "8ed72e2e0a6826d94726350d7ffa6287b720c9f9"
    tam = {}
    for name, mut in (("added path", m_add), ("changed after", m_after), ("changed before", m_before), ("dropped path", m_drop),
                      ("review verdict edited", m_verdict), ("short commit id", m_commit), ("other base commit", m_base)):
        s_ = make_rec(mut)                                   # re-sealed AND re-approved: validation beyond the approval itself
        tam[name] = (FT.a36_impl_check(ROOT, approval(s_))["ok"], FT.a34_a36_exception(ROOT, approval(s_))[0] == {})
    s_ = make_rec(m_add, reseal=False)
    tam["added path, not re-sealed"] = (FT.a36_impl_check(ROOT, approval(s_))["ok"], FT.a34_a36_exception(ROOT, approval(s_))[0] == {})
    s_ = make_rec(m_add)
    tam["added path, old approval"] = (FT.a36_impl_check(ROOT, approval(fsha))["ok"], FT.a34_a36_exception(ROOT, approval(fsha))[0] == {})
    check("6.5 a tampered record (added / dropped path, changed after, changed before, edited review verdict, short commit id, other base; "
          "re-sealed and even re-approved; or not re-sealed; or under the old approval) is rejected and the exception is void",
          all(not ok and void for ok, void in tam.values()), tam)
    fsha = make_rec()

    def inject(extra):
        def cs(root=None, **kw):
            d = dict(_orig_cs(root, **kw))
            d.update(extra)
            return d
        return cs
    try:
        A2m.code_state = inject({"sim/final_test/matrix/zz_unreviewed.py": "5" * 64})
        r5, w5 = FT.a34_a36_exception(ROOT, approval(fsha))
        A2m.code_state = inject({"sim/final_test/matrix/plan.py": "6" * 64})
        r5b, w5b = FT.a34_a36_exception(ROOT, approval(fsha))
        A2m.code_state = inject({"sim/final_test/matrix/score.py": "7" * 64})
        r5c, w5c = FT.a34_a36_exception(ROOT, approval(fsha))
    finally:
        A2m.code_state = _orig_cs
    check("6.6 a fifth (unlisted) changed file is rejected although a valid approved record covers the other files (new file sorting after "
          "the A36 files; an existing file changed); an A36 file off its after sha is rejected",
          r5 == {} and any("zz_unreviewed.py" in x for x in w5) and r5b == {} and any("plan.py" in x for x in w5b)
          and r5c == {} and any("score.py" in x for x in w5c), (w5, w5b, w5c))
    ok_all = [g for g, v in fzA["gates"].items() if not FT.gate_ok(g, v)]
    check("6.7 build_ft2 with the valid temporary record + approval: 12.1 and registration PASS; acceptance WAIVED (never PASS); the freeze "
          "binds the record + approval (succession)", fzA["gates"]["12.1"]["status"] == "PASS" and fzA["gates"]["registration"]["status"] == "PASS"
          and fzA["gates"]["acceptance"]["status"] == ("WAIVED" if FT.a36_signature(ROOT)[0] else "FAIL")
          and fzA["succession"]["ok"] and fzA["succession"]["a36_impl"]["ok"]
          and fzA["succession"]["a36_impl"]["approval"]["file_sha256"] == SIGS_A[FT.A36_IMPL_KEY]["file_sha256"]
          and fzA["succession"]["predecessor"]["sha256"] == FZ_SHA0 and never_pass(fzA["acceptance"]), (ok_all, fzA["succession"]["reasons"]))
    print("     (with the temporary record + approval, gates not passed:", ok_all, ")")

    # ---- write -> check in a temporary overlay root (junctions to the real tree; the files of docs/final-test (freeze.json, signatures.json),
    #      docs/final-test/matrix and its practice-v2 folder copied)
    import _winapi
    OV = tempfile.mkdtemp(prefix="fzft2ov_", dir=os.path.dirname(ROOT))   # same drive as the repo (relative bindings), outside it (no junction loop)

    def _overlay():
        def mirror(src, dst, real):
            for e in os.listdir(src):
                if e == ".git":
                    continue
                s_, d_ = os.path.join(src, e), os.path.join(dst, e)
                if os.path.isdir(s_) and not os.path.islink(s_):
                    if e in real:
                        os.mkdir(d_)
                        mirror(s_, d_, real[e])
                    else:
                        _winapi.CreateJunction(s_, d_)
                elif os.path.isfile(s_) and (dst != OV or os.path.getsize(s_) < 5_000_000):   # not the top-level *.pt downloads
                    shutil.copy2(s_, d_)
        # practice-v2: analyze.py checks its journal by real path. freeze_history (K23, test-only): COPIED, never a junction -- it exists in
        # the real tree since the A37 re-freeze, and the overlay write archives the predecessor there (a junction would write the real tree)
        mirror(ROOT, OV, {"docs": {"final-test": {"matrix": {"practice-v2": {}}, "freeze_history": {}}}})

    def _rm(p):
        for e in os.scandir(p):
            if os.path.isjunction(e.path):
                os.rmdir(e.path)                             # removes the junction only, never its target
            elif e.is_dir(follow_symlinks=False):
                _rm(e.path)
            else:
                os.remove(e.path)
        os.rmdir(p)
    try:
        _overlay()
        with open(os.path.join(OV, "docs", "final-test", "signatures.json"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(approval(fsha), indent=1) + "\n")
        # K23 merge prerequisite (Astra MERGE-DRAFTS pre-read, test-only): once Arash has SIGNED freeze.json, a plain re-write must be
        # refused (the intended outcome of this ONE step: reported as a check); the overlay copy then re-writes with the documented
        # supersede path so that every later check (6.8-6.12, 5.12, 4.8) is still exercised. Runtime code unchanged; the real tree is
        # never written (OV is a temporary overlay).
        _signed = bool((F._read_json(os.path.join(OV, "docs", "final-test", "freeze.json")) or {}).get("arash_check"))
        if _signed:
            try:
                FT.write_ft2(OV, probes=PROBES)
                _refused = ""
            except PermissionError as _e:
                _refused = str(_e)
            check("6.7c a SIGNED freeze refuses a plain re-write (intended refusal: --supersede required, the old file archived)",
                  "--supersede" in _refused, _refused)
        wout = FT.write_ft2(OV, probes=PROBES, supersede=_signed)
        arc = os.path.join(OV, *FT._history_path(FZ_SHA0).split("/"))
        chk = FT.check_ft2(OV, probes=PROBES)
        rW, wW = FT.a34_a36_exception(OV)
        check("6.8 write in a temporary root: the historical freeze is archived content-addressed and named as the predecessor; the written "
              "freeze binds the A36 waiver + the record + its approval", os.path.isfile(arc) and F._sha_file(arc) == FZ_SHA0
              and wout["succession"]["predecessor"]["sha256"] == FZ_SHA0 and wout["succession"]["ok"] and wout["succession"]["a36_impl"]["ok"]
              and wout["freeze_profile"] == "ft-v2" and wout["all_gates_pass"] == (not [g for g, v in wout["gates"].items() if not FT.gate_ok(g, v)]),
              wout.get("succession", {}).get("reasons"))
        check("6.9 write -> check is stable: no diff, no failed gate, the re-freeze exceptions survive the write (successor accepted); only "
              "Arash's freeze check is owed", chk["diff"] == [] and chk["failed_gates"] == [] and rW and not wW
              and chk["reasons"] == ["Arash's freeze check not recorded"], (chk, wW))
        check("6.10 the written freeze records acceptance NOT PASS (WAIVED under A36), never PASS",
              wout["gates"]["acceptance"]["status"] == "WAIVED" and never_pass(wout["acceptance"]) and "acceptance" not in wout["status_summary"]["PASS"])
        with open(arc, "ab") as fh:
            fh.write(b" ")
        rT, wT = FT.a34_a36_exception(OV)
        check("6.11 a tampered archived predecessor voids the exception after the write", rT == {} and any("predecessor" in x for x in wT), wT)
    finally:
        _rm(OV)
    check("6.12 the real freeze.json / signatures.json are untouched by the overlay write", F._sha_file(FZ_PATH) == FZ_SHA0
          and F._sha_file(os.path.join(C.paths(None)["docs"], "signatures.json")) == SIG_DISK
          and (sorted((e, F._sha_file(os.path.join(_HIST_DIR, e))) for e in os.listdir(_HIST_DIR)) if os.path.isdir(_HIST_DIR) else None) == HIST0
          and not os.path.exists(os.path.join(ROOT, *_ORIG_IMPL_FILE.split("/")))
          or IMP_REAL["ok"])
    # K23 (test-only, Astra fixes check): 6.12's inherited `or IMP_REAL["ok"]` can bypass its integrity part, and the existing A36
    # record makes it fail regardless; 6.12a asserts the real-tree integrity UNCONDITIONALLY (before / after snapshots).
    _real_after = (F._sha_file(FZ_PATH), F._sha_file(os.path.join(C.paths(None)["docs"], "signatures.json")),
                   sorted((e, F._sha_file(os.path.join(_HIST_DIR, e))) for e in os.listdir(_HIST_DIR)) if os.path.isdir(_HIST_DIR) else None,
                   _impl_sha(ROOT))
    check("6.12a the real freeze.json, signatures.json, freeze_history and A36 record are byte-identical before / after the overlay write "
          "(unconditional)", _real_after == (FZ_SHA0, SIG_DISK, HIST0, IMPL0), _real_after)
finally:
    impl_off()
    A2m.code_state = _orig_cs
    shutil.rmtree(TI, ignore_errors=True)
check("5.12 signatures.json was only read (byte-identical)", F._sha_file(os.path.join(C.paths(None)["docs"], "signatures.json")) == SIG_DISK)
check("4.8 freeze.json is byte-identical to the start of this selftest", F._sha_file(FZ_PATH) == FZ_SHA0)

print(f"\n{N['ok']}/{N['ok'] + N['fail']} checks passed")
sys.exit(0 if N["fail"] == 0 else 1)
