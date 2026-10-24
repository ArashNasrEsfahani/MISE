"""The ft-v2 freeze builder (stage ACC-RUNNER, 2026-10-14; the freeze plumbing F1-F9 of sim/final_test/diag/c11/freeze_dryrun.json).

freeze.py's builder was written for ft-v1 (the 800-token per-level budget, A12 coverage, the 15 practice-v1 snapshots, the 49-type ft-v1
prompt, gates/ evidence from 2026-10-12). Since the test-v1 flip (TEST_V1_TREATMENT = ft-v2) a --write through that builder would look
partly green on STALE evidence (C11-AND-REG step 3). This module builds the ft-v2 freeze on top of freeze.build's generic part (frozen
files, code closure, seeds, caps, arms, cost guard, dedup, analysis plan, weights, map, probes, 12.4 / A14 / 12.8 / 12.9 / 12.13), removes
every ft-v1-only field, and adds:

  F1 the practice-v2 snapshots (snapshots_ft2/o0/k1, k3, k5, k10) with each K's registered role, their own directory, per-file shas, nesting,
     their A30 / A31 leakage gate, and the A31 equivalence certificate that binds them (practice ran under d5831abc) to the final identity;
  F2 the ft-v2 prompt (contract version ft-v2, SYSTEM_V2, every ft-v2 text, the 49 detector classes and the 48 shown types) with gate 12.12
     in the A26 wording (a)-(d) and gate 12.11 for the ft-v2 renderer;
  F3 the token gate = A24 7 (g) / (h) (<= 2,000 tokens incl. the header, max over both tokenizers) + the measured context fit (i);
  F4 A12: recorded NOT_APPLICABLE with the registered basis (A24 item 1 text + the capture audit), never silently passed;
  F5 the evidence gates read from gates_ft2/ (gate_evidence.py --ft2), every binding re-verified against the disk (stale = FAIL);
  F6 fb-1 (the committed runner contract, reproduced), the acceptance registration, the C11 audit, a24_2_cost (D13 = prereg A33; energy method C5) and the
     signature fields of A30 / A31 / A32 / A33 (PENDING ARASH);
  F7 the namespaces incl. acceptance-v2 (attempt ceiling, study guard);
  F9 the ft-v2 FS artefacts (the FS-v2 corpus = the S* example store, fs_block.py, package params) and the writer / evidence-generator
     code identities; 12.1 keeps the required code hashes (capture 9ae7036c, practice_k c47a2b36, runner 11cfa254);
  F8 is the selftest: sim/_ftm_freeze_ft2_selftest.py (stale / mismatched / missing evidence fails);
  A36 the gate `acceptance` (precondition "acceptance PASS", both verdicts authenticated by bundle replay): the terminal NOT PASS is WAIVED
     only with Arash's signatures.json["A36"] bound to the exact sha256 of docs/final-test/amendments/A36-DRAFT.md; acceptance is recorded
     as NOT PASS (both verdict shas), never PASS; nothing else is waived.
  A36-IMPL (Astra A36-IMPL-freeze-r1) the A34-A2 re-freeze exception survives the A36 code changes only for the 14 files of the A36
     implementation record (docs/final-test/amendments/A36.impl.json) at their exact after shas, judged on the complete structured
     changed-file set, and only with Arash's separate approval signatures.json["A36-IMPL"] of the record's sha256; the written ft-v2 freeze
     is the explicit successor of the historical freeze (archived in docs/final-test/freeze_history/, authenticated), so write -> check holds.

A gate is PASS, NOT_APPLICABLE (registered), PENDING (registered evidence or a decision still owed; pass = false) or FAIL. all_gates_pass
needs every gate PASS / NOT_APPLICABLE. Signatures and any reinterpretation are Arash's: this module records them as PENDING and never
fills them. D13 (the hardware price basis) is read from the analysis plan and checked against the prereg heading it names (A33).

    python sim/final_test/matrix/freeze.py --write | --check        (dispatch here while TEST_V1_TREATMENT is ft-v2)
    python sim/final_test/matrix/freeze_ft2.py --dry-run [--out PATH] (builds, writes only the report; freeze.json untouched)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                              # noqa: E402
import freeze as F                                              # noqa: E402

FT2_SCHEMA_NOTE = "ft-v2 freeze (freeze_ft2.py)"
SNAP_FT2 = ("sim", "final_test", "snapshots_ft2", "o0")
SNAP_FT2_FILES = ("manifest.json", "ltm.db", "fragments.json", "practice_manifest.json", "practice_rows.jsonl")
#: F1: the registered role of each practice-v2 snapshot (A27: order 0 only; K = 10 is S*)
K_ROLES = {1: "learning curve only (A27, order 0, K = 1)", 3: "learning curve only (A27, order 0, K = 3)",
           5: "learning curve only (A27, order 0, K = 5)", 10: "S* = the deployed memory of every +M arm and the FS-v2 example corpus (prereg 6.3)"}
#: the registered pins of gate 12.1 (A31 DONE, owed item (2))
REQUIRED_PINS = {"sim/final_test/matrix/capture.py": "9ae7036c", "sim/final_test/matrix/practice_k.py": "c47a2b36", "sim/planners/runner.py": "11cfa254"}
FT2_ENTRIES = ("sim/final_test/matrix/freeze.py", "sim/final_test/matrix/freeze_ft2.py", "sim/final_test/matrix/gate_evidence.py", "sim/final_test/matrix/fg_runner.py", "sim/final_test/matrix/a36.py",
               "sim/final_test/matrix/plan_batch.py", "sim/final_test/matrix/context_fit.py", "sim/final_test/matrix/acc_common.py",
               "sim/final_test/acceptance_v2/acc_runner.py", "sim/final_test/acceptance_v2/register_a25.py",
               "sim/final_test/acceptance_v2/validate_live.py", "sim/final_test/acceptance_v2/validate_reserved.py",
               "sim/final_test/diag/c11/c11_audit.py", "sim/final_test/matrix/practice_k.py",
               "sim/final_test/matrix/planfirst.py", "sim/final_test/matrix/energy.py")       # C5: the plan-first launcher + the energy wrapper are pinned
EVIDENCE_GATES_FT2 = ("12.2", "12.3", "12.5", "12.7", "12.10", "G-iso", "leakage", "X-iso", "tokens", "context")
GATE_ORDER_FT2 = ("12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "12.10", "12.11", "12.12", "12.13", "G-iso", "A12",
                  "A14", "snapshot", "leakage", "tokens", "context", "X-iso", "registration", "acceptance", "fb-1", "a24_2_cost", "signatures")
#: ft-v1-only fields of freeze.build that an ft-v2 freeze must NOT carry (they would look green on stale evidence)
FT1_ONLY = ("fs_blocks_sha256", "snapshots", "channels", "prompt_version", "system_template_sha", "user_template_sha", "plan_schema_sha",
            "headers_sha", "detector_types", "system_text_sha", "evidence_sha", "blocks_ns", "blocks_sha", "tokens", "a12", "_rows_doc", "gates",
            "all_gates_pass")
#: prereg A24 item 1: the sentence that retires the A16 / A12 per-level budget for the +M arms (F4 basis)
A12_BASIS_TEXT = "this replaces A16's 800-token B for the +M arms"
SIGNED_AMENDMENTS = ("A30", "A31", "A32", "A33", "A24.1-C")   # A33 = D13 (PAPERWORK 2026-10-14); A24.1-C (RUNNER-FINAL); signed at the freeze summary
SIGNATURES_FILE = "signatures.json"          # docs/final-test/: written only on Arash's instruction (never by this module)
#: fb-1 fixed vectors: input -> sha256 of the text cap_feedback must return (computed once from the pinned runner 11cfa254, 2026-10-14)
FB1_VECTORS = {"long": ("x" * 1000, "6f7704581b3d90273330f944569fe40fe4421381cb8c8d86f1e7816bcf36ed95"),
               "persian": ("line one\n" + "\u0645\u062b\u0627\u0644 \u0628\u0627 \u0686\u0627\u0642\u0648 caf\u00e9\n" * 3, "11e6288c498671a20d009e7b29b4683f71beadeb7af70f760cac55b6014f1abc"),
               "lines": ("\n".join("- problem %d: the step 'pickup Bread' needs a free hand" % i for i in range(40)), "38ac764ca0001dc21046ac0a7e3b3a9c3408234c2886b09a88859d64cec74194")}
OK_STATUSES = ("PASS", "NOT_APPLICABLE")
#: Astra ACC-RUNNER r2 held 12.12 / 12.11 at PENDING; cycle 3 closed both in code (12.12 against REGISTERED_CLASSES; 12.11 by the AST
#: renderer callers + the effective contract of every ft-v2 namespace). A gate listed here is still forced to PENDING (none now).
ASTRA_R2_HELD = {}
#: prereg A26 12.12 (a): "the ordered class list extracted from the frozen weights matches the registered count and hash, using the
#: registered extraction and serialization method, and the weights hash also matches". Registered independently of detector_types.json:
#: the ordered digest is the one the ft-v1 freeze recorded (docs/final-test/freeze.json, file 068814ab..., gate 12.12 types_sha) and the
#: weights sha is the one the treatment identity pins (7a2ee211..., files["sim/weights/v3/FloorPlan1.yolo26s.pt"]).
REGISTERED_CLASSES = {"kitchen": "FloorPlan1", "n": 49, "weights": "sim/weights/v3/FloorPlan1.yolo26s.pt",
                      "weights_sha": "4418f63f4b84060c51353bffb574d05e49042e763b36136d870c7e56e8b90f54",
                      "ordered_sha": "db08b02f3e9c19dea053ba6211f3a58182bdd9b242f53c1cc8981521ca379bd1",
                      "method": "sorted(set(YOLO(weights).names.values())) (write_detector_types.py) -> common.sha256_json(list)",
                      "registered_in": "docs/final-test/freeze.json 068814ab gate 12.12 types_sha; treatment identity 7a2ee211 weights file sha"}
#: K23 (Astra K23-AGENT-r1 2): one INDEPENDENTLY pinned registration per kitchen; Kitchen 1's entry is REGISTERED_CLASSES unchanged.
#: FloorPlan15 / FloorPlan17 are pinned only after their A38 weights are frozen (None = not registered -> gate 12.12 FAILs closed).
REGISTERED_CLASSES_BY_SCENE = {"FloorPlan1": REGISTERED_CLASSES, "FloorPlan15": None, "FloorPlan17": None}
#: 12.11: the ONLY modules that may call plan.render_messages (delivery: plan.py; audits / gates that re-render captured requests read-only)
RENDER_CALLERS = ("sim/final_test/matrix/plan.py", "sim/final_test/matrix/context_fit.py", "sim/final_test/acceptance_v2/acc_runner.py",
                  "sim/final_test/diag/c11/c11_audit.py", "sim/final_test/diag/freeze_gates/cost_projection.py", "sim/final_test/matrix/freeze_ft2.py")
FT2_CONTRACT_NAMESPACES = ("test-v1", "acceptance-v2")


def g2(status, **evidence):
    return {"pass": status in OK_STATUSES, "status": status, "evidence": evidence}


def _sha(root, rel):
    return F._sha_file(os.path.join(root, *rel.split("/")))


# ================================================================================================ F5: evidence (gates_ft2/, bound)
def read_evidence_ft2(root, gid, fresh=None):
    """One gates_ft2 file -> gate. The file's own status counts only if: schema ftm.gate/2, right gate id, not synthetic, no key-like text,
    every bound input file still has its recorded sha, the generator (gate_evidence.py) is the current file, and every recorded treatment
    identity still equals the identity recomputed now. Otherwise FAIL (stale / mismatched / missing)."""
    import gate_evidence as GE
    import treatment as T
    p = os.path.join(GE.gates_ft2_dir(root), f"{gid}.json")
    try:
        rel = F._rel(root, p)
    except ValueError:                                          # another drive (a temporary folder in a test)
        rel = p
    if not os.path.exists(p):
        return g2("FAIL", reason=f"no evidence file {rel} (gate_evidence.py --ft2 --write)", sha=None)
    raw = open(p, encoding="utf-8").read()
    try:
        d = json.loads(raw)
    except ValueError:
        return g2("FAIL", reason="evidence file unreadable", sha=F._sha_file(p))
    why = []
    if d.get("schema") != GE.GATE2_SCHEMA:
        why.append("schema is not ftm.gate/2 (an ft-v1 gates/ file is never accepted)")
    if d.get("gate") != gid:
        why.append("gate id differs")
    if d.get("synthetic"):
        why.append("synthetic evidence")
    if C.key_leak(raw):
        why.append("key-like text in evidence")
    b = d.get("binding") or {}
    for r, want in sorted((b.get("inputs") or {}).items()):
        if "/" not in r:
            continue                                            # a derived digest (e.g. the study ledgers), re-derived below
        got = _sha(root, r)
        if got != want:
            why.append(f"stale: input {r} changed since the evidence was written" if want and got else f"stale: input {r} {'missing' if want else 'appeared'}")
    for r, want in sorted((b.get("generator") or {}).items()):
        if _sha(root, r) != want:
            why.append(f"stale: the evidence generator {r} changed since the evidence was written")
    idb = b.get("identity") or {}
    for grp in ("dev", "final", "practice"):
        for ns, want in sorted((idb.get(grp) or {}).items()):
            try:
                now = T.identity(ns, root)["sha"]
            except Exception as e:                              # noqa: BLE001
                why.append(f"identity of {ns} not computable: {type(e).__name__}")
                continue
            if now != want:
                why.append(f"stale: treatment identity of {ns} is {now[:12]} now, the evidence recorded {str(want)[:12]}")
    st = d.get("status")
    if st not in ("PASS", "FAIL", "PENDING", "NOT_APPLICABLE"):
        why.append(f"unknown status {st!r}")
    if st in OK_STATUSES and d.get("pass") is not True:
        why.append("status and pass disagree")
    if gid in GE.FINAL_IDENTITY_GATES and st in OK_STATUSES and (d.get("final_identity") or {}).get("status") != "PASS":
        why.append("a final-identity gate cannot pass while its final-identity evidence is not PASS")
    if fresh is not None:                                       # the freeze re-derives every verdict (a hand-edited file disagrees)
        if fresh.get("status") != st:
            why.append(f"the file says {st}, a fresh re-derivation says {fresh.get('status')} (hand-edited or stale evidence)")
        if (fresh.get("binding") or {}).get("inputs") != b.get("inputs"):
            why.append("the bound inputs differ from a fresh re-derivation")
    status = "FAIL" if why else st
    return g2(status, reason=why or None, file_status=st, sha=F._sha_file(p), source=d.get("source"), created=d.get("created"),
              final_identity=d.get("final_identity"), owed=(d.get("evidence") or {}).get("owed") or (d.get("final_identity") or {}).get("owed"))


# ================================================================================================ F1: practice-v2 snapshots
def snapshots_ft2(root):
    base = os.path.join(root, *SNAP_FT2)
    out, why, prev, prev_rows = {}, [], -1, None
    for k in (1, 3, 5, 10):
        d = os.path.join(base, f"k{k}")
        key = f"o0k{k}"
        files = {f: F._sha_file(os.path.join(d, f)) for f in SNAP_FT2_FILES}
        ent = {"dir": F._rel(root, d), "files": files, "role": K_ROLES[k], "source": key}
        if not all(files.values()):
            why.append(f"{key}: missing {[f for f, v in files.items() if not v]}")
            out[key] = ent
            continue
        m = F._read_json(os.path.join(d, "manifest.json"))
        pm = F._read_json(os.path.join(d, "practice_manifest.json"))
        ef = m.get("E_frozen")
        ent.update(E_frozen=ef, state_hash=m.get("state_hash"), integrity=m.get("integrity"), practice_manifest_sha=pm.get("sha256"),
                   treatment=pm.get("treatment"), treatment_id=pm.get("treatment_id"), namespace=pm.get("namespace"), k=pm.get("k"),
                   leakage_ok=(pm.get("leakage") or {}).get("ok"), rows_sha=pm.get("rows_sha"), package_profile=pm.get("package_profile"))
        if m.get("integrity"):
            why.append(f"{key}: manifest lists integrity problems")
        if pm.get("sha256") != C.self_sha(pm):
            why.append(f"{key}: practice manifest self-sha mismatch")
        if (pm.get("leakage") or {}).get("ok") is not True:
            why.append(f"{key}: practice leakage gate not ok")
        if pm.get("namespace") != "practice-v2" or pm.get("treatment") != "ft-v2" or int(pm.get("k") or -1) != k or int(pm.get("order", -1)) != 0:
            why.append(f"{key}: practice manifest is not practice-v2 / ft-v2 / order 0 / K = {k}")
        if isinstance(ef, int):
            if ef < prev:
                why.append(f"{key}: E_frozen {ef} < the smaller snapshot's {prev} (not nested)")
            prev = ef
        else:
            why.append(f"{key}: no E_frozen")
        # Astra ACC-RUNNER Q6: the files themselves against the manifests that describe them
        if m.get("file_sha256") != files["ltm.db"]:
            why.append(f"{key}: ltm.db differs from manifest.file_sha256")
        if pm.get("fragments_sha256") != files["fragments.json"]:
            why.append(f"{key}: fragments.json differs from practice_manifest.fragments_sha256")
        if pm.get("rows_sha") != files["practice_rows.jsonl"]:
            why.append(f"{key}: practice_rows.jsonl differs from practice_manifest.rows_sha")
        if pm.get("store_manifest_sha256") != files["manifest.json"]:
            why.append(f"{key}: manifest.json differs from practice_manifest.store_manifest_sha256")
        rows = [ln for ln in open(os.path.join(d, "practice_rows.jsonl"), encoding="utf-8").read().splitlines() if ln.strip()]
        if len(rows) != k:
            why.append(f"{key}: {len(rows)} practice rows, K = {k}")
        if prev_rows is not None and rows[:len(prev_rows)] != prev_rows:
            why.append(f"{key}: the practice rows do not extend the smaller snapshot's rows (prefix nesting)")
        prev_rows = rows
        ent["dir_sha"] = C.dir_sha(d)
        out[key] = ent
    out["S*"] = dict(out.get("o0k10") or {}, source="o0k10")
    return out, why


def equivalence_ft2(root, snaps):
    """The A31 certificate: practice-v2 ran under d5831abc; the re-pin to 7a2ee211 is valid only with the replayed equivalence proof, bound
    to the current treatment.py, the practice journal and the snapshot directories."""
    p = os.path.join(root, "sim", "final_test", "diag", "a31", "equiv_replay.json")
    why = []
    if not os.path.exists(p):
        return {"path": None, "ok": False, "reasons": ["equiv_replay.json missing"]}
    e = F._read_json(p)
    if e.get("equivalent") is not True or e.get("problems") or not str(e.get("verdict", "")).startswith("EQUIVALENT"):
        why.append("the certificate is not EQUIVALENT")
    if e.get("new_treatment") != _sha(root, "sim/final_test/matrix/treatment.py"):
        why.append("the certificate's new treatment.py differs from the file on disk")
    pj = os.path.join(C.ns_docs("practice-v2", root), "practice.jsonl")
    if e.get("journal_sha") != F._sha_file(pj):
        why.append("the certificate's practice journal differs from docs/final-test/matrix/practice-v2/practice.jsonl")
    for k in (1, 3, 5, 10):                                     # the certificate's snapshot directories = the frozen directories
        want = ((e.get("E_snapshots") or {}).get(str(k)) or {}).get("dir_sha")
        if want != (snaps.get(f"o0k{k}") or {}).get("dir_sha"):
            why.append(f"the certificate's K = {k} directory sha differs from snapshots_ft2/o0/k{k}")
    tid = sorted({(snaps.get(f"o0k{k}") or {}).get("treatment_id") for k in (1, 3, 5, 10)} - {None})
    return {"path": F._rel(root, p), "sha256": F._sha_file(p), "verdict": e.get("verdict"), "old_treatment_py": e.get("old_treatment"),
            "new_treatment_py": e.get("new_treatment"), "snapshot_treatment_ids": tid, "ok": not why, "reasons": why}


# ================================================================================================ F2: prompt + 12.11 / 12.12
def prompt_ft2(root, types):
    import plan as PL
    import treatment as T
    import type_scope as TS
    from planners import contract as PC
    ct = PL.make_contract(types, "test-v1")
    shown = list(ct.types)
    sysx = ct.system_text()
    rec = {"version": ct.version, "system_text_sha": C.sha256_text(sysx), "system_v2_template_sha": C.sha256_text(PC.SYSTEM_V2),
           "ft_v2_texts_sha": PC.ft_v2_texts_sha(), "user_template_sha": C.sha256_text(PC.USER), "plan_schema_sha": C.sha256_json(PC.PLAN_SCHEMA),
           "package_header_sha": C.sha256_text(PC.PACKAGE_HEADER), "experience_header_sha": C.sha256_text(PC.EXPERIENCE_HEADER),
           "types": {"detector_n": len(types), "detector_sha": C.sha256_json(sorted(types)), "shown_n": len(shown), "shown_sha": C.sha256_json(shown),
                     "shown_types_sha": C.sha256_json(sorted(shown)), "excluded": list(TS.EXCLUDED_TYPES), "type_scope_sha": _sha(root, "sim/type_scope.py"),
                     "method": "sorted(set(YOLO(primary weights).names.values())) (or detector_types.json for the same weights sha) -> "
                               "treatment.planner_types = type_scope.filter_types (order kept); rendered joined by ', ' in SYSTEM_V2"}}
    return rec, ct, shown, sysx, T


def gate_1212_ft2(root, types, wsha, ct, shown, sysx, T, scene: str = "FloorPlan1"):
    """A26's 12.12 (prereg line 'replaces A4's 12.12'): (a) 49 classes from the frozen weights + their hash; (b) shown = filter_types(classes)
    = 48 and in the system text; (c) byte-identical across arms (one types sha / one system text in every dev plan row and capture row);
    (d) no agent-facing text names an excluded type (system text, tool lines, every dev package / FS block and observation summary)."""
    import type_scope as TS
    why = []
    if len(types) != F.EXPECTED_TYPES:
        why.append(f"(a) {len(types)} detector classes, the registered count is {F.EXPECTED_TYPES}")
    flt = TS.filter_types(sorted(set(types)))
    if shown != flt or len(shown) != F.EXPECTED_TYPES - len(TS.EXCLUDED_TYPES):
        why.append(f"(b) shown list ({len(shown)}) != type_scope.filter_types(classes) ({len(flt)})")
    if ", ".join(shown) not in sysx:
        why.append("(b) the system text does not contain the shown list")
    cap = [json.loads(x) for x in open(C.journal_path("rehearsal-v3", "capture", root), encoding="utf-8") if x.strip()]
    plans = [json.loads(x) for x in open(C.journal_path("rehearsal-v3", "plan", root), encoding="utf-8") if x.strip()]
    tsha = {r.get("types_sha") for r in plans}
    ssha = {(r.get("audit") or {}).get("system_sha") for r in cap}
    if tsha != {C.sha256_json(sorted(shown))}:
        why.append(f"(c) dev plan rows carry {len(tsha)} types sha(s), not the frozen list's")
    if ssha != {C.sha256_text(sysx)}:
        why.append(f"(c) dev capture rows carry {len(ssha)} system text sha(s), not the frozen text's")
    dtp = F.detector_types_path(root)
    dtd = F._read_json(dtp) if os.path.exists(dtp) else {}
    if dtd.get("weights_sha") != wsha or list(dtd.get("types") or []) != sorted(dtd.get("types") or []) or sorted(set(dtd.get("types") or [])) != sorted(set(types)):
        why.append("(a) the recorded ordered class list (detector_types.json) is not the one of these weights")
    reg = REGISTERED_CLASSES_BY_SCENE.get(scene)                # (a) vs the INDEPENDENTLY registered count, digest and weights hash
    if reg is None:
        return g2("FAIL", reason=f"no independently pinned class registration for {scene} (K23: pin it after the A38 weights freeze)")
    if len(types) != reg["n"]:
        why.append(f"(a) {len(types)} classes != the registered {reg['n']}")
    if C.sha256_json(list(dtd.get("types") or [])) != reg["ordered_sha"] or C.sha256_json(sorted(set(types))) != reg["ordered_sha"]:
        why.append("(a) the ordered class digest differs from the registered one (freeze.json 068814ab 12.12)")
    wfile = _sha(root, reg["weights"])
    if wsha != reg["weights_sha"] or wfile != reg["weights_sha"]:
        why.append("(a) the primary weights hash differs from the registered (identity-pinned) one")
    try:
        pinned = (T.identity("test-v1", root).get("files") or {}).get(reg["weights"])
    except Exception as e:                                      # noqa: BLE001
        pinned = None
        why.append(f"(a) the treatment identity could not be computed: {type(e).__name__}")
    if pinned != reg["weights_sha"]:
        why.append("(a) the treatment identity does not pin the registered weights")
    texts = {"system": sysx, "tool_lines": json.dumps(getattr(ct, "tools", None) or [], ensure_ascii=False)}
    try:
        import kitchen_briefing as KB
        for i, rec in enumerate(KB.load_records(scene, allow_unapproved=False)):
            texts[f"briefing#{i}"] = json.dumps(rec, ensure_ascii=False)
    except Exception as e:                                      # noqa: BLE001
        why.append(f"(d) the approved briefing could not be read: {type(e).__name__}")
    for r in cap:
        texts[f"{r['pres']}#c{r.get('capture_run')}#summary"] = "\n".join(r.get("prompt_summary") or [])
        for s, b in (r.get("blocks") or {}).items():
            if isinstance(b, dict) and b.get("text"):
                texts[f"{r['pres']}#c{r.get('capture_run')}#{s}"] = b["text"]
    hits = {k: T.excluded_alias_hits(v) for k, v in texts.items()}
    hits = {k: v for k, v in hits.items() if v}
    if hits:
        why.append(f"(d) excluded-type names in {len(hits)} agent-facing text(s): {sorted(hits)[:3]}")
    return g2("PASS" if not why else "FAIL", reasons=why, detector_n=len(types), shown_n=len(shown), weights_sha=wsha,
              detector_types_json_sha=F._sha_file(dtp), ordered_types_sha=C.sha256_json(list(dtd.get("types") or [])), registered=dict(reg),
              texts_checked=len(texts), dev_plan_rows=len(plans), dev_capture_rows=len(cap))


def render_callers(root, code_files) -> dict:
    """{rel: [line, ...]} of every CALL of render_messages in the code closure + the audit modules (AST, not a text search)."""
    import ast
    out = {}
    rels = sorted(set(code_files) | set(RENDER_CALLERS))
    for rel in rels:
        if "selftest" in rel:
            continue
        try:
            tree = ast.parse(open(F._abs(root, rel), encoding="utf-8").read())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
                if name == "render_messages":
                    out.setdefault(rel, []).append(node.lineno)
    return out


def effective_contracts(root, types) -> dict:
    """The contract each ft-v2 namespace actually builds (plan.make_contract): version, types and system text must be one configuration."""
    import plan as PL
    out = {}
    for ns in FT2_CONTRACT_NAMESPACES:
        ct = PL.make_contract(types, ns)
        out[ns] = {"version": ct.version, "types_sha": C.sha256_json(list(ct.types)), "system_sha": C.sha256_text(ct.system_text())}
    return out


def gate_1211_ft2(root, code_files, ct):
    from planners import contract as PC
    bad = []
    for rel in code_files:
        if rel.endswith("planners/contract.py") or "selftest" in rel:
            continue
        try:
            txt = open(F._abs(root, rel), encoding="utf-8").read()
        except OSError:
            continue
        if PC.PACKAGE_HEADER[:24] in txt or PC.EXPERIENCE_HEADER[:24] in txt:      # SYSTEM_V2 derives from step4_gemma's text by design
            bad.append(rel)
    defs, reo = [], []
    for rel in code_files:
        if "selftest" in rel or rel.endswith("matrix/freeze_ft2.py"):              # this checker names both strings itself
            continue
        try:
            txt = open(F._abs(root, rel), encoding="utf-8").read()
        except OSError:
            continue
        if "def render_messages(" in txt:
            defs.append(rel)
        if "reorder_within(" in txt and not rel.endswith("sim/ltm_retrieve.py"):
            reo.append(rel)
    callers = render_callers(root, code_files)
    stray = sorted(rel for rel in callers if rel not in RENDER_CALLERS)
    import plan as PL
    types, _src = PL.load_types(None, root)
    eff = effective_contracts(root, types)
    one_cfg = len({json.dumps(v, sort_keys=True) for v in eff.values()}) == 1 and all(v["version"] == "ft-v2" for v in eff.values()) \
        and eff["test-v1"]["system_sha"] == C.sha256_text(ct.system_text())
    ok = ct.version == "ft-v2" and not bad and defs == ["sim/final_test/matrix/plan.py"] and not reo and not stray \
        and "sim/final_test/matrix/plan.py" in callers and one_cfg
    return g2("PASS" if ok else "FAIL", contract_version=ct.version, redeclared_in=bad, render_messages_defined_in=defs,
              reorder_within_called_in=reo, render_callers=callers, stray_render_callers=stray, effective_contracts=eff, one_configuration=one_cfg,
              rule="one renderer: plan.render_messages over planners.contract (the block headers live there only), called only from the "
                   "registered modules (AST); every ft-v2 namespace builds the same effective contract; the memory reranker reorder_within is "
                   "not on the ft-v2 delivery path")


# ================================================================================================ F6 records
def fb1_record(root):
    """The committed fb-1 contract, reproduced: runner.py is the pinned file and cap_feedback gives the registered behaviour on fixed vectors."""
    from planners import runner as RU
    why = []
    rsha = _sha(root, "sim/planners/runner.py")
    if not str(rsha).startswith(REQUIRED_PINS["sim/planners/runner.py"]):
        why.append("runner.py is not the pinned 11cfa254 file")
    got = {}
    for name, (text, want) in FB1_VECTORS.items():
        out_, _info = RU.cap_feedback(text)
        got[name] = C.sha256_text(out_)
        if got[name] != want:
            why.append(f"cap_feedback({name}) = {got[name][:12]}, the fixed fb-1 vector is {str(want)[:12]}")
    if RU.FEEDBACK_CONTRACT != "fb-1" or RU.PROBLEM_TEXT_CAP_CHARS != 600:
        why.append("runner constants differ from fb-1 (contract name / 600-char cap)")
    return {"contract": RU.FEEDBACK_CONTRACT, "cap_chars": RU.PROBLEM_TEXT_CAP_CHARS, "tokens_per_char": dict(RU.PROBLEM_TOKENS_PER_CHAR),
            "omit_marker": RU.OMIT_MARKER, "runner_sha256": rsha, "vectors": got,
            "disclosure": "fb-1 repair-feedback contract (FREEZE-GATES step 2; Astra FREEZE-GATES Q1), disclosed next to the treatment id (C6 / A24.1-T)",
            "ok": not why, "reasons": why}


#: A36 implementation record (Astra A36-IMPL-freeze-r1): every tracked code file the signed A36's reviewed implementation (and the G-iso /
#: 12.5 / tokens / context / leakage gate build it required) changed after the A34-A2 / attempt-2 tree, each with its sha before (blob at
#: A36_BASE_COMMIT, null = new file) and after (the reviewed working tree). It grants the A34-A2 re-freeze exception for exactly these files
#: ONLY with Arash's separate approval signatures.json["A36-IMPL"] bound to the record's exact sha256 (A36's own signature binds the A36 text,
#: not this record). Written by sim/final_test/diag/a36/write_impl_record.py, approved by sim/final_test/diag/a36/sign_a36_impl.py (both run
#: by Arash); never written by this module.
A36_IMPL_FILE = "docs/final-test/amendments/A36.impl.json"
A36_IMPL_KEY = "A36-IMPL"
A36_IMPL_SCHEMA = "a36.impl/1"
A36_BASE_COMMIT = "eade1a18ee74e77839ef9f74c8961f8a0ab47868"      # the A34-A2 / attempt-2 launch tree
#: the fixed code scope (Astra A36-IMPL-freeze-r1 note 1: the changed matrix modules + the top-level selftests since eade1a1; freeze-r2: + the
#: state-aware _acc_attempt2_selftest.py = 15 files). Supporting
#: artefacts (launch scripts, logs, evidence) are inventoried by the record under `supporting_artifacts_informational` and grant nothing.
A36_CODE_SCOPE = ("sim/final_test/matrix/a36.py", "sim/final_test/matrix/analyze.py", "sim/final_test/matrix/score.py",
                  "sim/final_test/matrix/execute.py", "sim/final_test/matrix/common.py", "sim/final_test/matrix/fg_runner.py",
                  "sim/final_test/matrix/gate_evidence.py", "sim/final_test/matrix/freeze.py", "sim/final_test/matrix/freeze_ft2.py",
                  "sim/_ftm_a36_selftest.py", "sim/_final_matrix_selftest.py", "sim/_ftm_freeze_ft2_selftest.py",
                  "sim/_ftm_freeze_gates_selftest.py", "sim/_ftm_treatment_selftest.py", "sim/_acc_attempt2_selftest.py",
                  "sim/_acc_runner_selftest.py")
#: the reviews the record must cite (file -> the verdict that file must state); the last freeze review must ACCEPT
A36_REQUIRED_REVIEWS = {"A36-IMPL-analysis-r3": "docs/final-test/_astra/A36-IMPL-analysis-r3.txt",
                        "A36-IMPL-gates-r1": "docs/final-test/_astra/A36-IMPL-gates-r1.txt",
                        "A36-IMPL-freeze-r1": "docs/final-test/_astra/A36-IMPL-freeze-r1.txt",
                        "A36-IMPL-freeze-r2": "docs/final-test/_astra/A36-IMPL-freeze-r2.txt",
                        "A36-IMPL-freeze-r3": "docs/final-test/_astra/A36-IMPL-freeze-r3.txt"}
A36_FINAL_REVIEW = "A36-IMPL-freeze-r3"
A36_MUST_ACCEPT = ("A36-IMPL-analysis-r3", A36_FINAL_REVIEW)
#: the commit that fixed A36-IMPL-gates-r1 (12.5 reader); the record must list it among its reviewed commits
A36_GATES_FIX_COMMIT = "937636de117f825d355834b18fea17aa30acd5ed"
#: the historical freezes a re-freeze archives (content-addressed: freeze.<sha16>.json); untracked by acc_attempt2 (docs/final-test/*.json
#: is tracked only by exact name)
FREEZE_HISTORY_DIR = "docs/final-test/freeze_history"
_GIT_CACHE = {}


def _git(*args):
    """git in the repository this module lives in (the base commit is immutable, so a temporary root still checks against it)."""
    import subprocess
    k = tuple(args)
    if k not in _GIT_CACHE:
        r = subprocess.run(["git", "-C", C.ROOT, *args], capture_output=True)
        _GIT_CACHE[k] = (r.returncode, r.stdout)
    return _GIT_CACHE[k]


def _git_commit(cid) -> str | None:
    """The full id of an existing commit, else None."""
    if not isinstance(cid, str) or not re.fullmatch(r"[0-9a-f]{7,40}", cid):
        return None
    rc, out = _git("rev-parse", "--verify", "--quiet", cid + "^{commit}")
    return out.decode().strip() if rc == 0 else None


def _git_blob_sha(commit, rel):
    """sha256 of rel's bytes at commit; None if the path is absent there; raises if the commit is unknown."""
    import hashlib
    if _git_commit(commit) != commit:
        raise RuntimeError(f"unknown commit {commit}")
    rc, out = _git("show", f"{commit}:{rel}")
    return hashlib.sha256(out).hexdigest() if rc == 0 else None


def _review_verdict(text: str):
    m = re.search(r"VERDICT:\s*\**\s*([A-Z][A-Z \-]*[A-Z])", text or "")
    return m.group(1).strip() if m else None


def _load_sigs(root, sigs=None):
    if sigs is not None:
        return sigs
    p = os.path.join(C.paths(root)["docs"], SIGNATURES_FILE)
    try:
        return F._read_json(p) if os.path.exists(p) else {}
    except (OSError, ValueError):
        return {}


def a36_impl_approval(root, sigs=None) -> tuple:
    """-> (valid, binding, reasons). Arash's approval of the implementation record: signatures.json["A36-IMPL"] = {by: Arash, date, said,
    prereg_sha256 (the file on disk), file: A36_IMPL_FILE, file_sha256: the record's exact sha256 on disk, and the provenance (Astra
    A36-IMPL-freeze-r2): authorized_at, record_created_at = the record's `written`, authorization_preceded_record_creation: true
    (a pre-approval, not an inspection of an existing digest)}. Never written here."""
    e = _load_sigs(root, sigs).get(A36_IMPL_KEY)
    rsha = _sha(root, A36_IMPL_FILE)
    if not isinstance(e, dict):
        return False, None, [f"signatures.json holds no '{A36_IMPL_KEY}' record (the A36 implementation record is not approved by Arash)"]
    why = []
    if e.get("by") != "Arash" or not e.get("date") or not str(e.get("said") or "").strip():
        why.append("the A36-IMPL approval is not Arash's attributed record (by / date / said)")
    if e.get("prereg_sha256") != F._sha_file(C.paths(root)["prereg"]):
        why.append("the A36-IMPL approval is bound to another preregistration.md")
    if e.get("file") != A36_IMPL_FILE:
        why.append(f"the A36-IMPL approval does not name {A36_IMPL_FILE}")
    if not rsha or e.get("file_sha256") != rsha:
        why.append("the A36-IMPL approval's file_sha256 is not the sha256 of the implementation record on disk (another record was approved, or it changed)")
    try:
        written = (F._read_json(os.path.join(root, *A36_IMPL_FILE.split("/"))) or {}).get("written")
    except (OSError, ValueError):
        written = None
    if not e.get("authorized_at") or e.get("record_created_at") != written or not written \
            or e.get("authorization_preceded_record_creation") is not True or "pre-approval" not in str(e.get("authorization_kind") or ""):
        why.append("the A36-IMPL approval lacks its provenance (authorized_at, record_created_at = the record's written, "
                   "authorization_preceded_record_creation, authorization_kind pre-approval)")
    keys = ("by", "date", "said", "prereg_sha256", "file", "file_sha256", "authorized_at", "record_created_at",
            "authorization_preceded_record_creation", "authorization_kind")
    return not why, ({k: e.get(k) for k in keys} if not why else None), why


def a36_impl_check(root, sigs=None) -> dict:
    """The A36 implementation record, fully validated -> {ok, files {rel: after_sha} (empty unless ok), reasons, record, approval}.
    ok iff: A36 validly signed; the record intact (self sha), schema / amendment / signature key / A36 file + its signed sha / base commit
    exact; files = exactly A36_CODE_SCOPE, each `before` = its blob at the base commit (null = new) and `after` = the file on disk; every
    required review cited with its file sha and the verdict the file states (the analysis and the final freeze review ACCEPT); every
    reviewed commit a full existing id incl. the gates fix; and Arash's separate A36-IMPL approval bound to the record's sha256."""
    out = {"ok": False, "files": {}, "reasons": [], "record": {"path": A36_IMPL_FILE, "file_sha256": _sha(root, A36_IMPL_FILE)}, "approval": None}
    why = out["reasons"]
    try:
        ok36, srec, w36 = a36_signature(root, _load_sigs(root, sigs))
        p = os.path.join(root, *A36_IMPL_FILE.split("/"))
        rec = F._read_json(p) if os.path.exists(p) else None
    except Exception as e:                                  # noqa: BLE001 -- unverifiable: nothing is accepted
        why.append(f"the A36 implementation record cannot be read ({type(e).__name__})")
        return out
    if not ok36:
        why += [f"A36: {x}" for x in w36]
    if not isinstance(rec, dict):
        why.append(f"no A36 implementation record ({A36_IMPL_FILE}; Arash runs sim/final_test/diag/a36/write_impl_record.py)")
        return out
    out["record"]["self_sha256"] = rec.get("sha256")
    if rec.get("sha256") != C.self_sha(rec):
        why.append("the A36 implementation record's self sha is inconsistent (edited after writing)")
    for k, want in (("schema", A36_IMPL_SCHEMA), ("amendment", A36_KEY), ("signature_key", A36_KEY), ("a36_file", A36_FILE),
                    ("a36_file_sha256", srec.get("file_sha256")), ("base_commit", A36_BASE_COMMIT), ("approval_key", A36_IMPL_KEY)):
        if rec.get(k) != want or want is None:
            why.append(f"record {k} is not {want}")
    if not isinstance(rec.get("written"), str) or not rec.get("written"):
        why.append("record has no `written` time")
    files = rec.get("files")
    if not isinstance(files, dict) or set(files) != set(A36_CODE_SCOPE):
        extra = sorted(set(files or {}) - set(A36_CODE_SCOPE)) if isinstance(files, dict) else None
        miss = sorted(set(A36_CODE_SCOPE) - set(files or {})) if isinstance(files, dict) else None
        why.append(f"record files are not exactly the A36 code scope (extra {extra}, missing {miss})")
        files = {}
    for r, v in sorted(files.items()):
        if not isinstance(v, dict):
            why.append(f"record file {r}: not a before / after entry")
            continue
        try:
            b = _git_blob_sha(A36_BASE_COMMIT, r)
        except Exception as e:                              # noqa: BLE001
            why.append(f"record file {r}: base blob unverifiable ({type(e).__name__})")
            continue
        if v.get("before") != b:
            why.append(f"record file {r}: before sha is not its blob at {A36_BASE_COMMIT[:7]}")
        if not v.get("after") or v.get("after") != _sha(root, r):
            why.append(f"record file {r}: not on disk at its after sha")
    revs = rec.get("reviews")
    byid = {x.get("id"): x for x in revs if isinstance(x, dict)} if isinstance(revs, list) else {}
    for rid, rel in A36_REQUIRED_REVIEWS.items():
        x = byid.get(rid)
        fp = os.path.join(root, *rel.split("/"))
        if not x:
            why.append(f"record does not cite the review {rid}")
            continue
        if x.get("file") != rel or not os.path.isfile(fp) or x.get("sha256") != F._sha_file(fp):
            why.append(f"review {rid}: file / sha does not match {rel} on disk")
            continue
        stated = _review_verdict(open(fp, encoding="utf-8", errors="replace").read())
        if x.get("verdict") != stated:
            why.append(f"review {rid}: recorded verdict {x.get('verdict')!r} is not the file's {stated!r}")
        if rid in A36_MUST_ACCEPT and not str(stated or "").startswith("ACCEPT"):
            why.append(f"review {rid} does not ACCEPT ({stated!r})")
    commits = rec.get("reviewed_commits")
    ids = [c.get("commit") for c in commits if isinstance(c, dict)] if isinstance(commits, list) else []
    if not ids or any(not isinstance(c, str) or len(c) != 40 or _git_commit(c) != c for c in ids):
        why.append("reviewed_commits must be full ids of existing commits")
    if A36_GATES_FIX_COMMIT not in ids:
        why.append(f"reviewed_commits does not list the gates fix {A36_GATES_FIX_COMMIT[:7]}")
    okp, appr, wp = a36_impl_approval(root, sigs)
    out["approval"] = appr
    why += wp
    out["ok"] = not why
    if out["ok"]:
        out["files"] = {r: files[r]["after"] for r in A36_CODE_SCOPE}
    return out


def _a2_module():
    _ad = os.path.join(C.ROOT, "sim", "final_test", "acceptance_v2")
    if _ad not in sys.path:
        sys.path.insert(0, _ad)
    import acc_attempt2 as A2
    return A2


def historical_freeze_problems(A2, cr, root, path) -> list:
    """acc_attempt2.freeze_json_problems applied to `path` (the historical amendment freeze, in place or archived): it may differ from the
    attempt-1 bundle copy ONLY by the amendment_freezes["A34-A2+A35"] entry = the correction record's freeze file. [] = intact."""
    S = A2.SIGNED_AMENDMENT
    try:
        cur, old = A2._json(path), A2._json(A2._bundle_file(1, S["freeze_json"], root))
    except Exception as e:                                  # noqa: BLE001
        return [f"historical freeze unreadable ({type(e).__name__})"]
    if not isinstance(cur, dict) or not isinstance(old, dict):
        return ["historical freeze or its attempt-1 bundle copy missing"]
    out = []
    if A2._strip_amendment_freeze(cur) != A2._strip_amendment_freeze(old) or "amendment_freezes" in old:
        out.append("historical freeze changed beyond the added amendment_freezes entry")
    if cur.get("sha256") != C.self_sha(cur):
        out.append("historical freeze self-sha inconsistent")
    af = cur.get("amendment_freezes")
    fr = (cr or {}).get("freeze") or {}
    fz = A2._json(A2._abs(fr["path"], root)) if fr.get("path") else None
    if not isinstance(af, dict) or sorted(af) != [S["key"]]:
        out.append("historical freeze does not carry exactly the A34-A2+A35 amendment freeze")
    elif not isinstance(fz, dict) or af[S["key"]] != fz or fr.get("sha256") != A2.file_sha(A2._abs(fr["path"], root)):
        out.append("historical freeze's amendment freeze is not the correction record's freeze file")
    else:
        out += A2.freeze_content_problems(fz, root)
    return out


def _history_path(fsha: str) -> str:
    return f"{FREEZE_HISTORY_DIR}/freeze.{fsha[:16]}.json"


def succession_record(root, sigs=None, imp=None) -> dict:
    """The ft-v2 freeze as the explicit successor of the historical A34-A2 + A35 amendment freeze (Astra A36-IMPL-freeze-r1 blocker 3).
    predecessor = the historical freeze (archived, content-addressed, by write_ft2), the A36 waiver and the approved implementation record.
    Identical before a write (the predecessor is still freeze.json) and after it (the predecessor is the archive; the stored ft-v2 freeze
    carries it), so write -> check is stable."""
    A2 = _a2_module()
    P = C.paths(root)
    why = []
    try:
        stored = A2._json(P["freeze"]) if os.path.exists(P["freeze"]) else None      # the bytes on disk (their sha is the predecessor's)
        cr = F._read_json(os.path.join(root, "docs", "final-test", "acceptance_correction.json"))
    except (OSError, ValueError) as e:
        stored, cr = None, None
        why.append(f"freeze.json / the correction record unreadable ({type(e).__name__})")
    if isinstance(stored, dict) and stored.get("freeze_profile") == "ft-v2":
        pred = dict(((stored.get("succession") or {}).get("predecessor")) or {})
        src = os.path.join(root, *str(pred.get("path") or "-").split("/"))
        if not pred.get("path") or pred["path"] != _history_path(str(pred.get("sha256"))) or not os.path.isfile(src) \
                or F._sha_file(src) != pred.get("sha256"):
            why.append("the stored ft-v2 freeze names no archived predecessor at its sha (freeze_history)")
    else:
        fsha = F._sha_file(P["freeze"]) if os.path.exists(P["freeze"]) else None
        pred = {"path": _history_path(fsha) if fsha else None, "sha256": fsha, "self_sha256": (stored or {}).get("sha256"),
                "created": (stored or {}).get("created"), "freeze_status": (stored or {}).get("freeze_status"),
                "role": "the historical freeze: the attempt-1 freeze + the A34-A2+A35 amendment freeze (archived by write_ft2 before the ft-v2 freeze replaces it)"}
        src = P["freeze"]
    if not why and os.path.isfile(src):
        why += [f"predecessor: {x}" for x in historical_freeze_problems(A2, cr, root, src)]
    imp = imp if imp is not None else a36_impl_check(root, sigs)
    _ok36, w, _ = a36_signature(root, _load_sigs(root, sigs))
    return {"rule": "the ft-v2 freeze succeeds the historical A34-A2 + A35 amendment freeze (preserved in freeze_history, authenticated by sha and "
                    "by the freeze_json_problems rule); the A34-A2 re-freeze exception extends to the A36 implementation files only with the "
                    "signed A36 waiver and Arash's A36-IMPL approval of the exact record",
            "predecessor": pred, "waiver": {"key": A36_KEY, "file": A36_FILE, "file_sha256": w.get("file_sha256"),
                                            "signature": w.get("signature")},
            "a36_impl": {"record": imp["record"], "approval_key": A36_IMPL_KEY, "approval": imp["approval"], "files": imp["files"],
                         "ok": imp["ok"], "reasons": imp["reasons"][:8]},
            "ok": not why, "reasons": why}


def _successor_problems(root, sigs, imp) -> list:
    """freeze.json is an ft-v2 freeze that is the explicit, intact successor of the historical amendment freeze, bound to the current A36
    waiver and the current approved implementation record."""
    P = C.paths(root)
    cur = _a2_module()._json(P["freeze"])
    if not isinstance(cur, dict):
        return ["freeze.json unreadable"]
    out = []
    if cur.get("freeze_profile") != "ft-v2" or cur.get("sha256") != C.self_sha(cur):
        return ["freeze.json is neither the historical amendment freeze nor an intact ft-v2 freeze"]
    st = cur.get("succession")
    if not isinstance(st, dict):
        return ["the ft-v2 freeze.json carries no succession record"]
    now = succession_record(root, sigs, imp)
    if not now["ok"]:
        out += now["reasons"]
    for k in ("predecessor", "waiver", "a36_impl"):
        if st.get(k) != now.get(k):
            out.append(f"the ft-v2 freeze's succession {k} is not the current one")
    if not imp["ok"]:
        out.append("the A36 implementation record is not valid and approved")
    return out


def a34_a36_exception(root, sigs=None) -> tuple:
    """-> (refrozen {rel: sha}, reasons). The signed A34-A2 + A35 re-freeze exception, re-derived on the STRUCTURED changed-file set (never
    on diagnostic text): every check of acc_attempt2.amendment_problems, except that (a) the files changed outside the A34-A2 allowed set
    must ALL be A36 implementation files (complete set, no truncation), (b) a code pin may be off only for an A36 file at its exact after
    sha, (c) freeze.json is the historical amendment freeze or its explicit ft-v2 successor. {} with reasons when anything fails."""
    try:
        cr = F._read_json(os.path.join(root, "docs", "final-test", "acceptance_correction.json"))
    except (OSError, ValueError):
        return {}, ["no acceptance_correction.json"]
    if not isinstance(cr, dict) or cr.get("sha256") != C.self_sha(cr):
        return {}, ["the correction record is not intact"]
    imp = a36_impl_check(root, sigs)
    a36 = imp["files"] if imp["ok"] else {}
    out = []
    try:
        A2 = _a2_module()
        S = A2.SIGNED_AMENDMENT
        v1 = F._read_json(A2.verdict_path(1, root))
        am = cr.get("amendment")
        if am is None:
            return {}, ["the correction record names no signed amendment"]
        if am != A2.amendment_block(root):
            out.append("the record's amendment block is not the signed A34-A2 + A35 block")

        def sha(rel):
            q = A2._abs(rel, root)
            return A2.file_sha(q) if os.path.isfile(q) else "absent"
        tracked = A2._tracked_pin_paths(root)
        for rel in (S["text"], S["drafts"], S["a35_patch"]) + A2.DEV_VALIDATION_CODE:
            if rel not in S["pins"]:
                out.append(f"{rel} is not pinned")
        for rel, s_ in S["pins"].items():
            if rel not in tracked and sha(rel) != s_:
                out.append(f"{rel} is not at its pinned sha")
        if not S["pins"].get(S["a35_patch"], "").startswith(S["a35_patch_sha16"]):
            out.append("the pinned A35 patch is not the signed one")
        sig = (A2._json(A2._abs("docs/final-test/signatures.json", root)) or {}).get(S["key"]) or {}
        if (sig.get("by") != "Arash" or sig.get("said") != S["said"] or sig.get("drafts_sha256") != S["pins"].get(S["drafts"])
                or sig.get("b_patch_sha16") != S["a35_patch_sha16"]):
            out.append("signatures.json does not hold Arash's A34-A2+A35 signature binding the drafts and the A35 patch")
        if v1.get("rule_fingerprint") != S["rule_fingerprint_old"]:
            out.append("the attempt-1 verdict's rule fingerprint is not the one the amendment was signed against")
        if A2.self_core_sha(root) != S["self_core_sha256"]:
            out.append("acc_attempt2.py is not the reviewed version (core sha)")
        now = A2.code_state(root)
        changed = set(A2.changed_files(v1, root, now=now))
        code_pins = {r: s for r, s in S["pins"].items() if r in tracked}
        allowed = set(code_pins) | {A2.SELF_REL, S["freeze_json"]}
        extra = changed - allowed
        unlisted = sorted(extra - set(a36))
        if unlisted:
            out.append(f"tracked files changed outside the signed A34-A2 set and not in an approved A36 implementation record ({len(unlisted)}): {unlisted}")
        for r in sorted(extra & set(a36)):
            if now.get(r, "absent") != a36[r]:
                out.append(f"{r} is not at its A36 after sha")
        for r, s in code_pins.items():
            if now.get(r, "absent") != s and not (r in a36 and now.get(r, "absent") == a36[r]):
                out.append(f"{r} is not at its pinned sha (nor at an approved A36 after sha)")
        if S["freeze_json"] in changed:
            fjp = historical_freeze_problems(A2, cr, root, A2._abs(S["freeze_json"], root))
            if fjp:
                succ = _successor_problems(root, sigs, imp)
                if succ:
                    out += [f"freeze.json: {x}" for x in fjp] + [f"freeze.json successor: {x}" for x in succ]
        else:
            out.append("freeze.json does not carry the amendment freeze (Common terms 5)")
        pins = S["pins"]
    except Exception as e:                                  # noqa: BLE001 -- unverifiable: nothing is accepted
        return {}, [f"the amendment cannot be checked ({type(e).__name__}: {str(e)[:120]})"]
    if out:
        if extra and not imp["ok"]:
            out += [f"A36 implementation record: {x}" for x in imp["reasons"][:8]]
        return {}, out
    rf = (cr.get("refrozen") or {}).get("frozen_refs") or {}
    res = {r: s for r, s in rf.items() if pins.get(r) == s and r not in a36}
    res.update(a36)
    return res, []


def _refrozen_by_correction(root, sigs=None) -> dict:
    """{rel: sha} the acceptance correction record (A34-A2 + A35, signed 2026-10-15) re-froze, extended to the approved A36 implementation
    files at their exact after shas; {} when the exception does not hold (a34_a36_exception gives the reasons)."""
    return a34_a36_exception(root, sigs)[0]


def registration_record(root, code, sigs=None):
    import acc_common as AC
    p = AC.acceptance_registration_path(root)
    why = []
    try:
        reg = F._read_json(p)
    except (OSError, ValueError):
        return {"path": None, "ok": False, "reasons": ["acceptance_registration.json missing"]}
    why += AC.acceptance_registration_problems(reg)
    if reg.get("written_before_any_acceptance_call") is not True:
        why.append("the registration does not state it was written before any acceptance call")
    refrozen, exc_why = a34_a36_exception(root, sigs)      # A34-A2 + A35 (signed 2026-10-15) + the approved A36 implementation record
    stale = []
    for rel, want in sorted(((reg.get("frozen_refs") or {}).get("files") or {}).items()):
        if _sha(root, rel) != want and not (rel in refrozen and _sha(root, rel) == refrozen[rel]):
            stale.append(rel)
            why.append(f"frozen ref {rel} changed since the registration (stale; not re-frozen by the correction record)")
    if stale and exc_why:
        why += [f"re-freeze exception void: {x}"[:400] for x in exc_why[:6]]
    acc_cap = C.journal_path("acceptance-v2", "capture", root)
    if os.path.exists(acc_cap) and os.path.getsize(acc_cap) > 0 and not reg.get("roster"):
        why.append("acceptance-v2 has capture rows but the roster was never sealed (ordering violated)")
    return {"path": F._rel(root, p), "file_sha256": F._sha_file(p), "self_sha256": reg.get("sha256"), "amendment": reg.get("amendment"),
            "roster_sealed": bool(reg.get("roster")), "signature": reg.get("signature"),
            "signature_status": "PENDING ARASH" if not reg.get("signature") else "recorded", "ok": not why, "reasons": why}


def c11_record(root):
    p = os.path.join(root, "sim", "final_test", "diag", "c11", "c11_audit.json")
    if not os.path.exists(p):
        return {"path": None, "ok": False}
    d = F._read_json(p)
    return {"path": F._rel(root, p), "sha256": F._sha_file(p), "ok": d.get("ok") is True and not d.get("problems"), "binding": d.get("binding"),
            "scope": "C11 on the dev captures + stores; C11 on the acceptance captures is acceptance's criterion (registration c11_on_acceptance)"}


def calibration_check(root, em) -> tuple:
    """C5-SIMPLE (Arash, 2026-10-14, "simplify the energy method"): the evidence is the small real measurement of the simple method
    (sim/final_test/_checkpoints/C5-ENERGY/measure_simple.py), not a calibration. Checked: it exists, has this method id, was made with the
    energy.py on disk, its journals are unchanged, and it measured and joined every call. -> (record, problems)."""
    import hashlib
    import energy as EN
    rr = C.paths(root)["root"]
    ev = em.get("evidence") or ""
    evp = os.path.join(rr, ev) if ev else ""
    if not ev or not os.path.exists(evp):
        return {"path": ev or None, "ok": False}, [f"{ev or '(no evidence named)'} missing: run sim/final_test/_checkpoints/C5-ENERGY/measure_simple.py"]
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()     # noqa: E731
    try:
        d = F._read_json(evp)
    except Exception as e:                                   # noqa: BLE001
        return {"path": ev, "ok": False}, [f"unreadable: {type(e).__name__}"]
    why = []
    if d.get("schema") != "ftm.c5-simple-measurement/1":
        why.append("schema")
    if d.get("method") != EN.METHOD_ID:
        why.append(f"its method {d.get('method')!r} != {EN.METHOD_ID}")
    if d.get("energy_py_sha256") != sha(os.path.join(HERE, "energy.py")):
        why.append("energy.py changed after the measurement (re-run measure_simple.py)")
    js = d.get("journals") or {}
    if not js:
        why.append("no journals recorded")
    for k, s_ in js.items():
        fp = os.path.join(rr, k)
        if not os.path.exists(fp) or sha(fp) != s_:
            why.append(f"measurement journal {k} missing or changed")
    j_ = d.get("join") or {}
    if not (d.get("ok") is True and j_.get("complete") is True and int(j_.get("measured") or 0) >= 1):
        why.append("the measurement did not measure and join every call")
    rec = {"path": ev, "sha256": F._sha_file(evp), "energy_py_sha256": d.get("energy_py_sha256"), "journals": js,
           "cells": len(d.get("cells") or []), "totals": d.get("totals"), "ok": not why}
    return rec, why


C5_DECISION_FILE = "c5_unmeasured_decision.json"      # docs/final-test/: written only on Arash's instruction (never by code)


def _prereg_section(pre: str, aid: str):
    """The text of '### <aid> (' up to the next '### ' heading -> (number of such headings, text or None)."""
    if not os.path.exists(pre):
        return 0, None
    lines = open(pre, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith(f"### {aid} (")]
    if len(idx) != 1:
        return len(idx), None
    j = next((k for k in range(idx[0] + 1, len(lines)) if lines[k].startswith("### ")), len(lines))
    return 1, "\n".join(lines[idx[0]:j]).rstrip("\n")


def unmeasured_rule_check(root, em, hardware_basis: str) -> tuple:
    """C5 unmeasured rule (Astra C5 r2 #2): an ARASH decision recorded in docs/final-test/c5_unmeasured_decision.json, never inferred from a
    heading. -> (record or None, fail list, pending list). 'declined' keeps unmeasured rows UNKNOWN; 'amendment' estimates the eligible ones at
    the run's measured Gemma rate (analyze ENERGY_METHOD.unmeasured_estimate, flagged) and needs a C5 prereg section (heading names C5; not the hardware-price amendment) carrying the analysis plan's structured
    unmeasured_rule_terms verbatim as one 'C5-UNMEASURED-TERMS: <json>' line, whose sha the decision binds and whose terms the decision repeats;
    analyze.price_row then applies exactly those terms (rate, eligible classes, claim) row by row."""
    import hashlib
    import analyze as AN
    ub = str(em.get("unmeasured_rule_basis") or "")
    p = os.path.join(C.paths(root)["docs"], C5_DECISION_FILE)
    if not ub:
        return None, [], ["ARASH DECISION (C5 unmeasured rule): practice-v2 ran before C5, so its Gemma planning energy is unmeasured and the +M / "
                          "+FS arms' A24.2 cost is UNKNOWN (statement blocked). Either approve a prospective C5 amendment ESTIMATING them at the "
                          "measured Gemma rate (unmeasured_rule_basis = 'amendment:<id>' + unmeasured_rule_terms, e.g. analyze."
                          "UNMEASURED_TERMS_APPROVED) or record 'declined' (the statement stays blocked, disclosed); "
                          f"either way Arash's words go in docs/final-test/{C5_DECISION_FILE}"]
    if ub.startswith("arash-decision:"):                    # C5-SIMPLE: Arash's ledger decisions of 2026-10-14 ("energy estimate ok";
        terms = em.get("unmeasured_rule_terms")              # "simplify the energy method (C5)"); the terms are the approved ones, verbatim
        fail = [f"analysis plan: {w}" for w in AN.unmeasured_terms_problems(terms, em)]
        if terms != AN.UNMEASURED_TERMS_APPROVED:
            fail.append("unmeasured_rule_terms != analyze.UNMEASURED_TERMS_APPROVED (the terms Arash approved)")
        if not fail:
            fail += [f"unmeasured_rule_terms: {w}" for w in AN.practice_journal_problems(C.paths(root)["root"], terms)]
        return {"basis": ub, "terms": terms, "by": "Arash", "date": ub.split(":", 1)[1], "said": "energy estimate ok",
                "source": "docs/final-test/progress.md 2026-10-14 'ARASH DECISION: energy estimate for the 10 unmeasured practice rows = OK' and "
                          "'ARASH DECISION: simplify the energy method (C5)'"}, fail, []
    if ub != "declined" and not ub.startswith("amendment:"):
        return None, [f"unmeasured_rule_basis {ub!r} is neither 'declined' nor 'amendment:<id>'"], []
    if not os.path.exists(p):
        return None, [], [f"unmeasured_rule_basis {ub!r} is set but Arash's decision record docs/final-test/{C5_DECISION_FILE} does not exist"]
    try:
        dec = F._read_json(p)
    except Exception as e:                                   # noqa: BLE001
        return None, [f"{C5_DECISION_FILE} unreadable: {type(e).__name__}"], []
    fail = []
    if not (dec.get("by") == "Arash" and dec.get("date") and str(dec.get("said") or "").strip()):
        fail.append(f"{C5_DECISION_FILE}: by / date / said (Arash's own words) missing")
    want = "declined" if ub == "declined" else "amendment"
    if dec.get("decision") != want:
        fail.append(f"{C5_DECISION_FILE}: decision {dec.get('decision')!r} != the analysis plan's {want!r}")
    rec = {"basis": ub, "decision_file_sha256": F._sha_file(p), "by": dec.get("by"), "date": dec.get("date"), "said": dec.get("said")}
    if want == "amendment":
        aid = ub.split(":", 1)[1]
        n, sec = _prereg_section(C.paths(root)["prereg"], aid)
        if hardware_basis == f"amendment:{aid}":
            fail.append(f"{aid} is the hardware-price amendment; it does not authorize the C5 unmeasured rule")
        terms = em.get("unmeasured_rule_terms")                # the binding, structured terms (Astra C5 r3 #1): plan = section = record
        tp = AN.unmeasured_terms_problems(terms, em)
        if tp:
            fail += [f"analysis plan: {w}" for w in tp]
        line = "C5-UNMEASURED-TERMS: " + json.dumps(terms, sort_keys=True)
        if n != 1 or sec is None:
            fail.append(f"unmeasured_rule_basis {ub!r}: the prereg has {n} '### {aid} (' headings (need exactly 1)")
        else:
            tl = [ln for ln in sec.split(chr(10)) if ln.startswith("C5-UNMEASURED-TERMS:")]
            if "C5" not in sec.split(chr(10))[0] or len(tl) != 1:
                fail.append(f"prereg {aid} is not a C5 unmeasured-rule amendment (its heading must name C5 and its section must carry exactly one "
                            "'C5-UNMEASURED-TERMS: <json>' line)")
            elif not tp and tl[0].rstrip() != line:
                fail.append(f"prereg {aid}: its C5-UNMEASURED-TERMS line {tl[0].rstrip()!r} != the analysis plan's {line!r}")
            ssha = hashlib.sha256(sec.encode("utf-8")).hexdigest()
            rec.update(amendment=aid, amendment_section_sha256=ssha)
            if dec.get("amendment") != aid or dec.get("amendment_section_sha256") != ssha:
                fail.append(f"{C5_DECISION_FILE}: amendment / amendment_section_sha256 do not bind prereg {aid} as it is on disk")
        if dec.get("terms") != terms:
            fail.append(f"{C5_DECISION_FILE}: terms {dec.get('terms')!r} != the analysis plan's unmeasured_rule_terms {terms!r}")
        if not tp:                                             # the bound practice journal, read and re-identified (Astra C5 r5 #1)
            fail += [f"unmeasured_rule_terms: {w}" for w in AN.practice_journal_problems(C.paths(root)["root"], terms)]
        rec.update(terms=terms, eligible_rows=dec.get("eligible_rows"), effect_on_claim=dec.get("effect_on_claim"))
    else:
        rec["rule"] = "declined: unmeasured Gemma rows (practice-v2) keep an UNKNOWN cost; the A24.2 statement stays blocked"
    return rec, fail, []


def a24_2_record(root=None, cfg=None):
    """freeze.json a24_2_cost (A24.2 + A33 / D13). The hardware price basis is read from the analysis plan, never hard-coded: 'laptop_purchase_price',
    or 'amendment:<id>' whose heading '### <id> (' must exist in the prereg on disk. -> (record, gate status, reasons). The energy method (C5) is a
    separate requirement: without it the gate stays PENDING (the Gemma cost would be UNKNOWN)."""
    import analyze as AN
    cfg = AN.A24_2_COST if cfg is None else cfg
    rec = {k: cfg[k] for k in ("version", "usd_per_kwh", "hardware_price_usd", "operating_hours", "hardware_time_factor") if k in cfg}
    basis = str(cfg.get("hardware_price_basis") or "")
    rec.update(energy_method=cfg.get("energy_method"), hardware_price_basis=basis or None,
               hardware_price_source=cfg.get("hardware_price_source"), hardware_price_disclosure=cfg.get("hardware_price_disclosure"))
    fail, pend = [], []
    if basis == "laptop_purchase_price":
        rec["basis_evidence"] = "declared laptop purchase price"
    elif basis.startswith("amendment:") and re.fullmatch(r"A\d+(\.\d+)?(-[A-Z0-9]+)?", basis.split(":", 1)[1] or ""):
        aid = basis.split(":", 1)[1]
        pre = C.paths(root)["prereg"]
        heads = [ln for ln in open(pre, encoding="utf-8") if ln.startswith(f"### {aid} (")] if os.path.exists(pre) else []
        if len(heads) == 1:
            import hashlib
            rec["basis_evidence"] = {"amendment": aid, "prereg_heading_sha256": hashlib.sha256(heads[0].rstrip("\r\n").encode("utf-8")).hexdigest()}
        else:
            fail.append(f"hardware_price_basis {basis!r}: the prereg has {len(heads)} '### {aid} (' headings (need exactly 1)")
    elif basis.endswith("pending") or not basis:
        pend.append("D13 hardware price basis not decided (laptop purchase price or a prospective amendment)")
    else:
        fail.append(f"hardware_price_basis {basis!r} is neither laptop_purchase_price nor amendment:<id>")
    em = cfg.get("energy_method")
    if not em:
        pend.append("energy method not recorded (FREEZE-READINESS C5: measured, non-overlapping GPU + CPU energy capture not built); "
                    "the Gemma cost stays UNKNOWN until it is")
    else:                                                   # C5 (stage C5-ENERGY): the method must be energy.py's, with a validated real calibration
        import energy as EN
        if not isinstance(em, dict) or em.get("id") != EN.METHOD_ID:
            fail.append(f"energy_method {em.get('id') if isinstance(em, dict) else em!r} is not the producer's method {EN.METHOD_ID} (energy.py)")
        else:
            ev_rec, why = calibration_check(root, em)
            rec["energy_evidence"] = ev_rec
            if why:
                pend.append("C5 calibration evidence not valid: " + "; ".join(why[:6]))
            ur, uf, up = unmeasured_rule_check(root, em, basis)
            if ur is not None:
                rec["unmeasured_rule"] = ur
            fail += uf
            pend += up
    status = "FAIL" if fail else ("PENDING" if pend else "PASS")
    rec["status"] = ("recorded" if status == "PASS" else status) + (": " + "; ".join(fail + pend) if fail or pend else "")
    return rec, status, fail + pend


def signatures_record(root):
    """A30 / A31 / A32 / A33 signature fields: never filled here. A signature counts only with EXPLICIT attributed evidence in
    docs/final-test/signatures.json ({amendment: {by: Arash, date, said, prereg_sha256}}) whose prereg sha equals the file on disk; a heading
    without DRAFT / PENDING is not a signature (Astra ACC-RUNNER Q6)."""
    p = os.path.join(C.paths(root)["docs"], SIGNATURES_FILE)
    doc = F._read_json(p) if os.path.exists(p) else {}
    pre = F._sha_file(C.paths(root)["prereg"])
    out = {}
    for a in SIGNED_AMENDMENTS:
        e = doc.get(a) or next((v for k, v in doc.items() if isinstance(v, dict) and a in k.split("+")), None) or {}
        signed = e.get("by") == "Arash" and bool(e.get("date")) and bool(str(e.get("said") or "").strip()) and e.get("prereg_sha256") == pre
        out[a] = {"signed": signed, "signature": ({k: e.get(k) for k in ("by", "date", "said")} if signed else None),
                  "status": "recorded" if signed else "PENDING ARASH", "source": SIGNATURES_FILE if e else None}
    return out


# ================================================================================================ acceptance precondition + the A36 waiver
#: A36 (DRAFT r2; Astra A36-DRAFT-r1 READY TO SIGN) rule 1: the Kitchen 1 locked test proceeds although acceptance-v2 ended terminally NOT PASS.
#: The freeze waives ONLY the precondition "acceptance PASS", and only when docs/final-test/signatures.json holds Arash's record under the key
#: "A36" (schema of the other records: by / date / said / context / prereg_sha256, plus file + file_sha256 = the exact signed file and its sha256,
#: as "A34-A2+A35" binds its drafts) whose file_sha256 equals the A36 file on disk. Acceptance is recorded as NOT PASS, never PASS.
A36_KEY = "A36"
A36_FILE = "docs/final-test/amendments/A36-DRAFT.md"
A36_WAIVES = "waives only the acceptance-PASS precondition"
#: the attempt-2 verdict A36 names ("verdict sha eeda4ef8..., protocol_stopped: true"): its self sha must be this one
A36_VERDICT2_SELF_SHA = "eeda4ef8df839c0e0ecb6d10982b6467d7327075c9c20561114e46c3297f6789"
#: the revised analysis (A36 rules 2-6) is built by the analysis stage; the freeze binds these files by sha (placeholder identity until that
#: stage declares its own identity record)
A36_ANALYSIS_FILES = ("sim/final_test/matrix/analyze.py", "sim/final_test/matrix/a36.py", "sim/final_test/matrix/score.py",
                      "sim/final_test_h1_stats.py")
ACCEPTANCE_GATE = "acceptance"
WAIVED = "WAIVED"


def gate_ok(gid, v) -> bool:
    """A gate counts as passed: PASS / NOT_APPLICABLE; WAIVED only for the acceptance gate (A36) and only with pass = true."""
    st = (v or {}).get("status")
    return st in OK_STATUSES or (gid == ACCEPTANCE_GATE and st == WAIVED and (v or {}).get("pass") is True)


def _acc_attempt2():
    ad = os.path.join(C.ROOT, "sim", "final_test", "acceptance_v2")
    if ad not in sys.path:
        sys.path.insert(0, ad)
    import acc_attempt2 as A2
    return A2


def a36_signature(root, sigs=None) -> tuple:
    """-> (valid, record, reasons). Valid iff signatures.json["A36"] is Arash's attributed record bound to the current prereg AND to the exact
    A36 file on disk (file + file_sha256). Never written here."""
    p = os.path.join(C.paths(root)["docs"], SIGNATURES_FILE)
    if sigs is None:
        try:
            sigs = F._read_json(p) if os.path.exists(p) else {}
        except (OSError, ValueError):
            sigs = {}
    e = (sigs or {}).get(A36_KEY)
    fsha = _sha(root, A36_FILE)
    rec = {"key": A36_KEY, "file": A36_FILE, "file_sha256": fsha, "waives": A36_WAIVES, "signature": None}
    if not isinstance(e, dict):
        return False, rec, [f"signatures.json holds no '{A36_KEY}' record (A36 not signed)"]
    why = []
    if e.get("by") != "Arash" or not e.get("date") or not str(e.get("said") or "").strip():
        why.append("the A36 record is not Arash's attributed signature (by / date / said)")
    if e.get("file") != A36_FILE:
        why.append(f"the A36 record does not name the signed file {A36_FILE}")
    if e.get("prereg_sha256") != F._sha_file(C.paths(root)["prereg"]):
        why.append("the A36 record is bound to another preregistration.md")
    if not fsha:
        why.append(f"{A36_FILE} missing")
    elif e.get("file_sha256") != fsha:
        why.append(f"the A36 record's file_sha256 is not the sha256 of {A36_FILE} on disk (another text was signed, or the file changed)")
    if not why:
        rec["signature"] = {k: e.get(k) for k in ("by", "date", "said", "prereg_sha256", "file", "file_sha256")}
    return not why, rec, why


def a36_analysis_identity(root) -> dict:
    """Placeholder for the revised analysis identity (A36 'Implementation' -> Freeze): the analysis files bound by sha; filled by that stage."""
    files = {rel: _sha(root, rel) for rel in A36_ANALYSIS_FILES}
    ver = None
    try:
        m = re.search(r'^A36_VERSION\s*=\s*"([^"]+)"', open(F._abs(root, "sim/final_test/matrix/a36.py"), encoding="utf-8").read(), re.M)
        ver = m.group(1) if m else None
    except OSError:
        pass
    return {"status": "PLACEHOLDER: bound by file sha; the A36 analysis stage has not declared its own identity record", "files": files,
            "files_sha256": C.sha256_json(files), "a36_version": ver, "missing": sorted(r for r, s in files.items() if not s)}


def acceptance_record(root, sigs=None) -> tuple:
    """The freeze precondition 'acceptance PASS' (registration decision rule; A34-A2 / A35 Common Term 4) -> (record, gate).
    PASS only on an authenticated PASS verdict (bundle replay). A terminal, authenticated NOT PASS fails the precondition unless the A36 waiver
    is signed for this exact text; then the gate is WAIVED (counts as met for this decision only) and the record still says NOT PASS."""
    A2 = _acc_attempt2()
    att, why = {}, []
    for k in (1, 2):
        p = A2.verdict_path(k, root)
        if not os.path.exists(p):
            att[str(k)] = None
            continue
        try:
            v = F._read_json(p)
            probs = A2.verdict_problems(k, root, replay=True)
        except Exception as e:                                  # noqa: BLE001 -- unverifiable = not intact
            v, probs = {}, [f"unreadable ({type(e).__name__})"]
        att[str(k)] = {"path": F._rel(root, p), "file_sha256": F._sha_file(p), "self_sha256": v.get("sha256"), "verdict": v.get("verdict"),
                       "protocol_stopped": v.get("protocol_stopped"), "intact": not probs, "problems": probs[:4]}
        if probs:
            why.append(f"attempt {k} verdict not intact: {probs[0]}"[:300])
    a1, a2 = att.get("1"), att.get("2")
    if a2 and a1 and a2.get("intact") and (A2._json(A2.verdict_path(2, root)) or {}).get("attempt1_verdict_sha") != a1["file_sha256"]:
        why.append("the attempt-2 verdict is bound to another attempt-1 verdict")
    rec = {"rule": "freeze precondition: acceptance PASS (registration decision rule; never re-scored, pooled or redefined)",
           "attempts": att, "verdict": None, "terminal": False, "waiver": None}
    ok_pass = [k for k, a in (("1", a1), ("2", a2)) if a and a["intact"] and a["verdict"] == "PASS"]
    if why:
        rec["verdict"] = "INVALID"
        return rec, g2("FAIL", reasons=why)
    if ok_pass:
        rec["verdict"] = f"PASS (attempt {ok_pass[0]})"
        return rec, g2("PASS", reasons=[])
    if not a1:
        rec["verdict"] = "not decided"
        return rec, g2("PENDING", reasons=["no acceptance verdict recorded"])
    if not a2:
        rec["verdict"] = "NOT PASS (attempt 1; correction cycle possible)"
        return rec, g2("FAIL", reasons=["acceptance attempt 1 is NOT PASS; the freeze needs acceptance PASS"])
    rec["verdict"] = "NOT PASS"
    rec["terminal"] = a2.get("protocol_stopped") is True
    base = ["acceptance ended NOT PASS (attempts 1 and 2; protocol stopped): the freeze precondition 'acceptance PASS' is not met"]
    if not rec["terminal"] or a1.get("verdict") != "NOT PASS" or a2.get("verdict") != "NOT PASS":
        return rec, g2("FAIL", reasons=base + ["the verdicts are not the terminal NOT PASS / NOT PASS that A36 names"])
    valid, w, wwhy = a36_signature(root, sigs)
    if a2.get("self_sha256") != A36_VERDICT2_SELF_SHA:
        valid, wwhy = False, wwhy + [f"the attempt-2 verdict is not the one A36 names ({A36_VERDICT2_SELF_SHA[:8]}...)"]
    an = a36_analysis_identity(root)
    w.update(acceptance_recorded_as="NOT PASS", attempt1_verdict_sha256=a1["file_sha256"], attempt2_verdict_sha256=a2["file_sha256"],
             attempt2_verdict_self_sha256=a2["self_sha256"], revised_analysis_identity=an, valid=valid, reasons=wwhy)
    rec["waiver"] = w
    if not valid:
        return rec, g2("FAIL", reasons=base + [f"no valid A36 waiver: {x}" for x in wwhy])
    if an["missing"]:
        return rec, g2("PENDING", reasons=[f"A36 waiver signed, but the revised analysis files are missing: {an['missing']}"], acceptance="NOT PASS")
    return rec, {"pass": True, "status": WAIVED, "evidence": {"acceptance": "NOT PASS", "waiver": A36_KEY, "file_sha256": w["file_sha256"],
                                                              "waives": A36_WAIVES, "reasons": []}}


# ================================================================================================ build
def build_ft2(root=None, *, probes=None, now=None, stored=None, signatures=None, scene: str = "FloorPlan1"):
    """signatures: an in-memory signatures.json (selftests only); None = the file on disk."""
    import treatment as T
    P = C.paths(root)
    root = P["root"]
    base = F.build(root, counter=None, probes=probes, blocks_ns=None, stored=None, now=now)
    fz = {k: v for k, v in base.items() if k not in FT1_ONLY}
    fz["freeze_profile"] = "ft-v2"
    fz["note"] = FT2_SCHEMA_NOTE
    gates = {g: dict(base["gates"][g], status="PASS" if base["gates"][g]["pass"] else "FAIL") for g in ("12.4", "A14", "12.8", "12.9")}
    # ---- treatment (A24.1-T): identity of test-v1 = acceptance-v2 = the registration's pin
    ident_acc = T.identity("acceptance-v2", root)["sha"]
    t_ok = (fz.get("treatment") or {}).get("sha") == ident_acc
    # ---- code (12.1 + F9 writer / generator identities)
    entries = list(fz["code_entries"]) + [e for e in FT2_ENTRIES if os.path.isfile(F._abs(root, e)) and e not in fz["code_entries"]]
    files, missing, errors = F.import_closure(root, entries)
    fz["code_entries"] = entries
    fz["code"] = {rel: F._sha_file(F._abs(root, rel)) for rel in files}
    fz["writers"] = {rel: fz["code"].get(rel) for rel in ("sim/final_test/matrix/freeze.py", "sim/final_test/matrix/freeze_ft2.py",
                                                          "sim/final_test/matrix/gate_evidence.py")}
    _rf, _rf_why = a34_a36_exception(root, signatures)     # A34-A2 + A35 (signed) + approved A36 record: a pin re-frozen at its exact sha
    pin_bad = sorted(rel for rel, want in REQUIRED_PINS.items() if not str(fz["code"].get(rel) or "").startswith(want)
                     and not (rel in _rf and fz["code"].get(rel) == _rf[rel]))
    man = C.load_manifest(root)
    inc = man.get("inclusion") or {}
    ds_ok = fz["dataset_sha256"] == (inc.get("dataset_sha256") or man.get("dataset_sha256"))
    # ---- F2 prompt + types
    wdir = os.path.join(root, "sim", "weights", "v3")
    primary = os.path.join(wdir, f"{scene}.yolo26s.pt")
    dt = F.detector_types(root, primary)
    types = dt["types"] or []
    pr, ct, shown, sysx, _T = prompt_ft2(root, types) if types else ({}, None, [], "", T)
    fz["prompt"] = pr
    gates["12.12"] = gate_1212_ft2(root, types, dt["weights_sha"], ct, shown, sysx, T, scene=scene) if types else g2("FAIL", reason="class list unreadable")
    gates["12.11"] = gate_1211_ft2(root, files, ct) if ct else g2("FAIL", reason="no contract")
    for gid, why in ASTRA_R2_HELD.items():                      # Astra ACC-RUNNER r2: path stopped at the repair cap -> PENDING, never PASS
        if gates[gid]["status"] == "PASS":
            gates[gid] = dict(gates[gid], status="PENDING", **{"pass": False})
            gates[gid]["evidence"] = dict(gates[gid]["evidence"], owed="Astra ACC-RUNNER r2 finding, not closed (repair cap reached): " + why)
    gates["12.1"] = g2("PASS" if (ds_ok and not missing and not errors and fz["code"] and None not in fz["code"].values() and not pin_bad and t_ok
                                  and all(fz.get(k) for k in ("prereg_sha256", "dataset_sha256", "params_sha256", "map_sha"))
                                  and pr.get("system_text_sha") and all(v["sha256"] for v in fz["detector_weights"].values())) else "FAIL",
                       dataset_matches_split=ds_ok, n_code_files=len(fz["code"]), missing_entries=missing, parse_errors=errors,
                       required_pins=REQUIRED_PINS, pins_failing=pin_bad, treatment_test_v1_equals_acceptance=t_ok,
                       unhashed=[k for k, v in fz["code"].items() if v is None], refreeze_exception_reasons=_rf_why[:6] if pin_bad else [])
    # ---- F1 snapshots
    snaps, why = snapshots_ft2(root)
    eq = equivalence_ft2(root, snaps)
    fz["snapshots"] = snaps
    fz["practice_equivalence"] = eq
    gates["snapshot"] = g2("PASS" if (not why and eq["ok"] and (snaps.get("o0k10") or {}).get("state_hash")) else "FAIL", reasons=why + eq["reasons"],
                           S_star_state_hash=(snaps.get("o0k10") or {}).get("state_hash"), n=4, roles={k: v.get("role") for k, v in snaps.items()})
    # ---- F9 FS-v2 artefacts + the other frozen artefacts
    k10 = "/".join(SNAP_FT2) + "/k10/practice_rows.jsonl"
    arts = {"fs_v2_corpus (S* example store)": k10, "fs_block.py": "sim/final_test/matrix/fs_block.py", "package_params.json": "sim/step5/package_params.json",
            "ft_v2.json": "sim/planners/ft_v2.json", "exec_contract.json": "sim/planners/exec_contract.json",
            "acceptance_registration.json": "docs/final-test/acceptance_registration.json", "c11_audit.json": "sim/final_test/diag/c11/c11_audit.json",
            "context_fit (dev)": "docs/final-test/matrix/rehearsal-v3/context_fit.json", "g_iso (dev)": "docs/final-test/matrix/rehearsal-v3/g_iso.json",
            "token_gate (dev)": "sim/final_test/diag/freeze_gates/token_gate.json", "equiv_replay.json": "sim/final_test/diag/a31/equiv_replay.json"}
    import gate_evidence as GE
    for gid in EVIDENCE_GATES_FT2:
        arts[f"evidence {gid}"] = F._rel(root, os.path.join(GE.gates_ft2_dir(root), f"{gid}.json"))
    fz["artefacts"] = {k: {"path": v, "sha256": _sha(root, v)} for k, v in sorted(arts.items())}
    # ---- F5 evidence gates + F3 / F4
    fresh = GE.ft2_evidence(root, {"codex": (probes or {}).get("codex") or {"version": fz.get("codex_cli_version")}})
    for gid in EVIDENCE_GATES_FT2:
        gates[gid] = read_evidence_ft2(root, gid, fresh.get(gid))
    fz["evidence_sha"] = {gid: gates[gid]["evidence"].get("sha") for gid in EVIDENCE_GATES_FT2}
    cap = [json.loads(x) for x in open(C.journal_path("rehearsal-v3", "capture", root), encoding="utf-8") if x.strip()]
    a12_rows = sorted({str((r.get("audit") or {}).get("a12")) for r in cap})
    prereg = open(P["prereg"], encoding="utf-8").read()
    line = next((ln for ln in prereg.splitlines() if A12_BASIS_TEXT in ln), None)
    a12_ok = bool(line) and len(a12_rows) == 1 and a12_rows[0].startswith("not applicable")
    gates["A12"] = g2("NOT_APPLICABLE" if a12_ok else "FAIL",
                      basis={"prereg_A24_item_1": A12_BASIS_TEXT, "prereg_line_sha": C.sha256_text(line) if line else None,
                             "capture_audit": a12_rows, "astra": "C4 Q6: A12 superseded / not applicable, no manufactured coverage pass"},
                      rule="F4: recorded as NOT_APPLICABLE by the registered text, never passed silently; the ft-v2 budget is the tokens gate")
    gates["12.6"] = g2("PASS" if all(gates[g]["status"] in OK_STATUSES for g in ("snapshot", "G-iso", "leakage")) else
                       ("PENDING" if all(gates[g]["status"] in OK_STATUSES + ("PENDING",) for g in ("snapshot", "G-iso", "leakage")) else "FAIL"),
                       parts={g: gates[g]["status"] for g in ("snapshot", "G-iso", "leakage")})
    # ---- F7 namespaces (12.13 from freeze.build, plus the acceptance namespace's own fields)
    fz["namespaces"] = {k: dict(v["ceilings"], **({"attempt_ceiling": v["attempt_ceiling"]} if "attempt_ceiling" in v else {}),
                                **({"study_guard": True} if v.get("study_guard") else {}), items=v["items"], treatment=T.treatment_of(k))
                        for k, v in sorted(C.NAMESPACES.items())}
    acc = C.NAMESPACES.get("acceptance-v2") or {}
    ns_ok = base["gates"]["12.13"]["pass"] and acc.get("items") == C.ACCEPTANCE_ROSTER and acc.get("study_guard") is True
    gates["12.13"] = g2("PASS" if ns_ok else "FAIL", **base["gates"]["12.13"]["evidence"], acceptance_v2=fz["namespaces"].get("acceptance-v2"))
    # ---- F6 records
    fz["fb1_feedback_contract"] = fb1_record(root)
    gates["fb-1"] = g2("PASS" if fz["fb1_feedback_contract"]["ok"] else "FAIL", reasons=fz["fb1_feedback_contract"]["reasons"])
    fz["acceptance_registration"] = registration_record(root, fz["code"], signatures)
    fz["succession"] = succession_record(root, signatures)  # A36-IMPL-freeze-r1 3: predecessor (archived) + waiver + approved impl record
    gates["registration"] = g2("PASS" if fz["acceptance_registration"]["ok"] else "FAIL", reasons=fz["acceptance_registration"]["reasons"],
                               roster_sealed=fz["acceptance_registration"]["roster_sealed"])
    fz["c11_audit"] = c11_record(root)
    fz["acceptance"], gates["acceptance"] = acceptance_record(root, signatures)   # A36: NOT PASS is recorded as NOT PASS; waiver bound
    fz["a24_2_cost"], st_cost, why_cost = a24_2_record(root)
    gates["a24_2_cost"] = g2(st_cost, reasons=why_cost, hardware_price_basis=fz["a24_2_cost"].get("hardware_price_basis"),
                             energy_method=fz["a24_2_cost"].get("energy_method"))
    fz["signatures"] = signatures_record(root)
    unsigned = sorted(a for a, v in fz["signatures"].items() if not v["signed"])
    gates["signatures"] = g2("PASS" if not unsigned else "PENDING", unsigned=unsigned, rule="Arash signs A30 / A31 / A32 / A33 / A24.1-C at the freeze; never filled by code")
    fz["gates"] = {g: gates[g] for g in GATE_ORDER_FT2 if g in gates}
    fz["all_gates_pass"] = all(gate_ok(g, v) for g, v in fz["gates"].items()) and set(fz["gates"]) == set(GATE_ORDER_FT2)
    fz["status_summary"] = {s: sorted(g for g, v in fz["gates"].items() if v["status"] == s) for s in ("PASS", "NOT_APPLICABLE", "PENDING", "FAIL")}
    fz["status_summary"][WAIVED] = sorted(g for g, v in fz["gates"].items() if v["status"] == WAIVED)
    fz["arash_check"] = (stored or {}).get("arash_check")
    return fz


def write_ft2(root=None, *, probes=None, supersede=False, now=None):
    P = C.paths(root)
    root = P["root"]
    old = None
    if os.path.exists(P["freeze"]):
        try:
            old = F._read_json(P["freeze"])
        except ValueError:
            old = None
        if old and old.get("arash_check") and not supersede:
            raise PermissionError("freeze.json already carries Arash's check; a re-freeze needs --supersede (the old file is archived)")
        if old:                                            # archived content-addressed: the ft-v2 freeze names its predecessor by this path + sha
            fsha = F._sha_file(P["freeze"])
            arc = os.path.join(root, *_history_path(fsha).split("/"))
            os.makedirs(os.path.dirname(arc), exist_ok=True)
            if not os.path.exists(arc):
                shutil.copyfile(P["freeze"], arc)
            if F._sha_file(arc) != fsha:
                raise RuntimeError(f"freeze history {arc} exists with other bytes; refusing")
    fz = build_ft2(root, probes=probes, now=now)
    fz["arash_check"] = None
    out = F.seal(fz)
    if C.key_leak(json.dumps(out)):
        raise RuntimeError("key-like text in freeze.json; refusing to write")
    F._atomic_write(P["freeze"], out)
    return out


def check_ft2(root=None, *, probes=None):
    P = C.paths(root)
    root = P["root"]
    try:
        stored = F._read_json(P["freeze"])
    except (OSError, ValueError) as e:
        return {"ok": False, "reasons": [f"freeze.json missing or unreadable: {type(e).__name__}"], "diff": []}
    reasons = []
    if stored.get("freeze_profile") != "ft-v2":
        reasons.append("freeze.json is not an ft-v2 freeze")
    if stored.get("sha256") != C.self_sha(stored):
        reasons.append("freeze.json self-sha mismatch (edited after writing)")
    pr = probes or {"ollama": stored.get("ollama"), "codex": {"version": stored.get("codex_cli_version")}}
    fz = build_ft2(root, probes=pr, now=stored.get("created"), stored=stored)
    vol = set(F.VOLATILE) | {"_rows_doc"}
    diff = sorted(k for k in set(stored) | set(fz) if k not in vol and stored.get(k) != fz.get(k))
    bad = sorted(g for g, v in fz["gates"].items() if not gate_ok(g, v))
    if diff:
        reasons.append(f"differs from the disk: {diff}")
    if bad:
        reasons.append(f"gates not passed: {bad}")
    if not (stored.get("arash_check") or {}).get("by"):
        reasons.append("Arash's freeze check not recorded")
    return {"ok": not reasons, "reasons": reasons, "diff": diff, "failed_gates": bad}


def record_arash_check_ft2(root=None, *, by, date, said, probes=None):
    """--record-arash-check on an ft-v2 freeze: ONLY on Arash's explicit instruction (his words required); refuses unless every gate passes on a
    fresh rebuild (no PENDING: signatures, D13 and the final-identity evidence must exist first) and the file is intact."""
    P = C.paths(root)
    root = P["root"]
    if by != "Arash" or not date or not (said or "").strip():
        raise PermissionError("arash_check needs by='Arash', a date and his words (--arash-said)")
    stored = F._read_json(P["freeze"])
    if stored.get("sha256") != C.self_sha(stored):
        raise RuntimeError("freeze.json self-sha mismatch; refusing")
    if stored.get("arash_check"):
        raise PermissionError("already recorded")
    r = check_ft2(root, probes=probes)
    pending = [x for x in r["reasons"] if "Arash's freeze check" not in x]
    if pending or not stored.get("all_gates_pass"):
        raise RuntimeError("gates / files are not clean, no check recorded: " + "; ".join(pending)[:500])
    stored["arash_check"] = {"by": "Arash", "date": date, "said": said.strip()}
    out = F.seal({k: v for k, v in stored.items() if k != "sha256"})
    F._atomic_write(P["freeze"], out)
    return out


def dry_run(root=None, *, probes=None, out_path=None):
    """Build the ft-v2 freeze in memory; write ONLY the report (never freeze.json)."""
    P = C.paths(root)
    root = P["root"]
    before = F._sha_file(P["freeze"])
    fz = build_ft2(root, probes=probes)
    after = F._sha_file(P["freeze"])
    rep = {"schema": "acc-runner.freeze-dryrun-ft2/1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "freeze_json_sha_before": before,
           "freeze_json_sha_after": after, "freeze_json_untouched": before == after, "treatment": (fz.get("treatment") or {}).get("sha"),
           "dataset_sha256": fz.get("dataset_sha256"), "prereg_sha256": fz.get("prereg_sha256"), "n_code_files": len(fz["code"]),
           "status_summary": fz["status_summary"], "all_gates_pass": fz["all_gates_pass"],
           "gates": {g: {"status": v["status"], **{k: v["evidence"].get(k) for k in ("reason", "reasons", "owed", "unsigned", "file_status") if v["evidence"].get(k)}}
                     for g, v in fz["gates"].items()},
           "ft1_fields_absent": [k for k in FT1_ONLY if k in ("fs_blocks_sha256", "channels", "prompt_version", "blocks_ns", "tokens", "a12") and k not in fz],
           "preview": F.seal(dict(fz))}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    a = ap.parse_args(argv)
    probes = {"ollama": F.probe_ollama(a.ollama), "codex": F.probe_codex()}
    rep = dry_run(C.ROOT, probes=probes, out_path=a.out)
    print(json.dumps({k: rep[k] for k in ("freeze_json_untouched", "treatment", "status_summary", "all_gates_pass")}, indent=1))
    for g, v in rep["gates"].items():
        print(f"{v['status']:<15} {g}  {json.dumps({k: x for k, x in v.items() if k != 'status'}, ensure_ascii=False)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
