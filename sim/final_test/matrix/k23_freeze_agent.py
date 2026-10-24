"""K23 agent freeze: writes docs/final-test/k23/freeze_<scene>.json, the per-kitchen freeze k23_ns.replication_freeze_status reads before
ANY test item of test-k2-v1 / test-k3-v1 is captured, planned, executed or scored (ROADMAP section 5 item 4; Astra K23-NAMESPACES-r1
items 4-5). NEW module: freeze.py / freeze_ft2.py (Kitchen 1) and k23_ns.py are not changed; Kitchen 1's freeze.json is never read as
evidence here.

What the record binds (all checked again by replication_freeze_status / k23_entry.test_entry_problems on every test entry):
  schema, scene, namespaces, split + preregistration shas, the code closure of the K23 entry points (sha256 per file), the A29 roster sha
  (from the frozen sample + the recorded item-6 decision), the kitchen's practice snapshot S* (dir sha), the treatment identity, the
  dataset sha, the frozen B5 type list sha, the perception freeze record (re-verified in WSL), and the gates below. Sign-off: Arash's check,
  or his standing pre-signature (approve-all-6 item 1: "valid only if every gate passes, Astra agrees, and a freeze summary is written for
  him to read afterwards") -- recorded by --presign with the Astra review file and the summary this tool writes.

Gates (each {pass, evidence}; the record is written only when every gate passes):
  pins            treatment.pin_problems() == []
  a37_hold        k23_ns.a37_hold_problems() == []
  perception      the kitchen's perception freeze verifies (every listed file re-hashed; run in WSL); FloorPlan15: the COCO-start
                  sensitivity is appended (A38 item 4; Arash 2026-10-20 "keep the current order": before the K2 agent test)
  class_registration  the K23 form of gate 12.12: (a) the B5 list (detector_types_<scene>.json) = the probe registry of the pre-capture
                  class registration (sim/kitchen_v3/<scene>/probe.json "registry") = expected_classes.json, with the registered count, and
                  its weights sha = the perception freeze's primary; (b) shown = type_scope.filter_types(classes), its registered count, in
                  the system text; (c) every rehearsal capture row carries that system text and every rehearsal plan row that shown list;
                  (d) no excluded-type name in the system text, tool lines, the approved briefing or any rehearsal summary / block
  samples         both kitchens' A29 manifests verify (A29 item 2)
  roster          the A29 roster resolves through the recorded item-6 decision; FloorPlan17 only after the after-fp15 checkpoint
  package         uses table + homeowner inventory final (k23-v1), approved briefing (k23_entry.package_problems)
  practice        practice-kN-v1: 10 attempts of order 0 journaled, phase_complete bound to the journal, every row of the current treatment
                  identity, the K = 10 snapshot verifies (store, manifest, leakage gate), and the identity equals the test namespace's
  rehearsal       rehearsal-kN-v1: capture gates all passed + phase_complete, plan_first_complete full, execute complete, score sealed
                  complete and unchanged, no invalid / violation in the exec rows, rows of the current identity, billed <= its ceiling
  bilingual       12.2 on the rehearsal capture: EN and FA of every registered dev item give the same S* package record ids
  honesty         12.3: 0 violations / 0 oracle reads / 0 map-build reads (a missing field is never a 0) in the rehearsal capture sweeps,
                  the practice episodes, the rehearsal executions and the rehearsal G-iso executions
  g_iso           12.6 G-iso, live, in THIS kitchen: rehearsal-kN-v1/g_iso.json (execute --g-iso, n = 12) status pass, bound
  masking_leakage 12.6 masking + leakage: the rehearsal capture gates checked every presentation; the practice snapshot's leakage gate
  tokens          A24 7(g)/(h): every rehearsal package block <= the 2,000-token maximum (header included, max over tokenizers)
  context         A24 1 / 7(i): context_fit.json of the rehearsal capture MEASURED on the Gemma server and freeze-grade
  evaluator       12.5: this kitchen's A29 applicability (reference plans under the frozen executor reach the goal, goal false after setup)
                  + the rehearsal P0 rows scored; the empty-plan control is the shared, bound Kitchen 1 12.5 evidence (scoring code bound)
  reset           12.4 / A14: the validated reset tolerances (kitchen-independent numerical ceilings, shared, bound); their effect in this
                  kitchen is measured live by every execution's state check (A36 blocked-state accounting)
  spend           12.7: the shared, bound spend-guard evidence (fake over-cap, guard behaviour) + the budget gate below
  families        12.8: the approved family manifest (shared)
  gemma           12.9: the served Gemma digest starts with the registered prefix (live probe of the local Ollama; free)
  workflow_b      12.10: the shared, bound Workflow-B evidence (training data never test data)
  renderer        12.11: no K23 entry point renders a prompt itself (AST); the test namespace's effective contract recorded
  namespace       12.13: the test namespace has no capture / plan / exec row and an empty response cache at the freeze
  fb1             fb-1: the committed runner feedback contract, reproduced (freeze_ft2.fb1_record)
  amendments      the preregistration carries the A29.1 applicability text (the freeze binds the preregistration hash)
  fa_examples     Arash's decision K23-FA-EXAMPLES (Persian-practised examples rendered in English, k23_fa_examples.py) is recorded and
                  valid (it is then bound in the code map); every activation event in the kitchen's practice / rehearsal journals used
                  exactly that decision record
  fa_equivalence  A29.1 item 4 (prereg A39 A29.1 item 8: required before the kitchen's agent freeze): the kitchen's FA equivalence result
                  docs/final-test/k23/applicability/<scene>/equivalence_fa.json exists, names the kitchen and reports its agreement over
                  the registered sample size (min(EQUIV_N, applicable pool)); read through its summary fields only (kitchen, n,
                  agreement), never its item rows; the agreement is reported, not thresholded (A29.1: "its agreement reported")
  dataset         the gold dataset is the one the split's inclusion manifest names
  budget          study-wide liability check and the K2 + K3 replication cap hold with the kitchen's projected test cost added
  selftests       the K23 gates run now (subprocess, this interpreter) with rc 0
  dispositions    every Kitchen 1 freeze gate has a recorded disposition here: current kitchen evidence (a gate above), shared bound
                  evidence, or registered non-applicability (Astra K23-TOOLING-G3 3)

    python sim/final_test/matrix/k23_freeze_agent.py --scene FloorPlan15 [--dry-run]           (default: compute + print, nothing written)
    python sim/final_test/matrix/k23_freeze_agent.py --scene FloorPlan15 --write [--supersede]  (Windows: ~/thor-data via the WSL share)
    python sim/final_test/matrix/k23_freeze_agent.py --scene FloorPlan15 --presign --astra-review docs/final-test/k23/_astra/<review>.txt
    python sim/final_test/matrix/k23_freeze_agent.py --scene FloorPlan15 --record-arash-check --by Arash --date YYYY-MM-DD --arash-said "..."
    python sim/final_test/matrix/k23_freeze_agent.py --scene FloorPlan15 --check
Python 3.10; offline (file reads, the selftests as subprocesses; no simulator, no provider).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402
import k23_entry as E                                    # noqa: E402
import k23_ns as K                                       # noqa: E402

SCHEMA = K.FREEZE_SCHEMA
NS = {"FloorPlan15": {"test": "test-k2-v1", "practice": "practice-k2-v1", "rehearsal": "rehearsal-k2-v1"},
      "FloorPlan17": {"test": "test-k3-v1", "practice": "practice-k3-v1", "rehearsal": "rehearsal-k3-v1"}}
PRACTICE_K = 10
CODE_ENTRIES = tuple("sim/final_test/matrix/" + f for f in (
    "k23_run.py", "k23_practice.py", "k23_entry.py", "k23_size_decision.py", "k23_freeze_agent.py", "k23_ns.py", "k23_analyze.py",
    "k23_fa_examples.py", "k23_leakage.py", "k23_capture_freeze.py", "k23_giso_waiver.py",
    "capture.py", "plan.py", "planfirst.py", "execute.py", "score.py", "common.py", "treatment.py", "practice_k.py", "matrix_run.py",
    "energy.py", "hang_guard.py", "acc_common.py")) + ("sim/final_test/c3/real_counter.py", "sim/final_test/reset_gate.py",
                                                      "sim/planners/contract.py", "sim/final_test/k23_run/launch_k23.py",
                                                      "sim/final_test/k23_run/practice/launch.py",
                                                      "sim/final_test/k23_run/practice/launch_strict.py",
                                                      "sim/final_test/k23_run/launch_k23_strict.py")
SHELL_FILES = ("sim/final_test/k23_run/chain_k23.sh", "sim/final_test/k23_run/run_k23.sh", "sim/final_test/k23_run/practice/run.sh",
               "sim/final_test/k23_run/practice/inner.sh", "sim/final_test/k23_run/practice/run_strict.sh",
               "sim/final_test/k23_run/chain_k23_strict.sh", "sim/final_test/k23_run/run_k23_strict.sh",
               "sim/final_test/k23_run/practice/inner_strict.sh")
SELFTESTS = ("sim/final_test/matrix/_k23_practice_selftest.py", "sim/final_test/matrix/_k23_run_selftest.py",
             "sim/final_test/matrix/_k23_size_decision_selftest.py", "sim/final_test/matrix/_k23_freeze_agent_selftest.py",
             "sim/final_test/matrix/_k23_namespaces_selftest.py", "sim/final_test/matrix/_k23_scene_isolation_selftest.py",
             "sim/final_test/matrix/_k23_a37_hold_selftest.py", "sim/kitchen_v3/k23/_k23_coco_meta_selftest.py",
             "sim/final_test/matrix/_k23_fa_examples_selftest.py", "sim/final_test/matrix/_k23_leakage_selftest.py",
             "sim/final_test/matrix/_k23_capture_freeze_selftest.py", "sim/final_test/matrix/_k23_execute_types_selftest.py",
             "sim/final_test/matrix/_k23_giso_waiver_selftest.py", "sim/final_test/k23_run/_k23_strict_wrappers_selftest.py")
GATE_ORDER = ("pins", "a37_hold", "perception", "class_registration", "samples", "roster", "package", "practice", "rehearsal",
              "bilingual", "honesty", "g_iso", "masking_leakage", "tokens", "context", "evaluator", "reset", "spend", "families", "gemma",
              "workflow_b", "renderer", "namespace", "fb1", "amendments", "fa_examples", "fa_equivalence", "dataset", "budget", "selftests",
              "dispositions")
GISO_N = 12
SHARED_EVIDENCE = {"12.5": "docs/final-test/matrix/gates_ft2/12.5.json", "12.7": "docs/final-test/matrix/gates_ft2/12.7.json",
                   "12.10": "docs/final-test/matrix/gates_ft2/12.10.json"}
#: Astra K23-TOOLING-G3 3: every gate of the Kitchen 1 ft-v2 freeze (freeze_ft2 / freeze.py) -> how Kitchen 2 / 3 cover it.
K1_GATES = ("12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "12.10", "12.11", "12.12", "12.13", "A12", "A14",
            "G-iso", "X-iso", "a24_2_cost", "acceptance", "context", "fb-1", "leakage", "registration", "signatures", "snapshot", "tokens")
DISPOSITIONS = {
    "12.1": ("current", ["pins", "perception", "dataset", "gemma"], "code closure + bound data files (record 'code'), the kitchen's weights "
             "(perception record), the walked map and package files (record 'code'), split / prereg shas, model digest"),
    "12.2": ("current", ["bilingual"], "rehearsal capture of this kitchen"),
    "12.3": ("current", ["honesty"], "rehearsal capture + practice + rehearsal executions + rehearsal G-iso of this kitchen"),
    "12.4": ("shared", ["reset"], "validated A14b tolerances are kitchen-independent ceilings; per-kitchen effect = live state checks (A36)"),
    "12.5": ("current+shared", ["evaluator"], "this kitchen's A29 applicability + rehearsal P0; empty-plan control = bound K1 12.5 evidence"),
    "12.6": ("current", ["g_iso", "masking_leakage", "practice"], "G-iso live in this kitchen; masking / leakage per presentation; snapshot"),
    "12.7": ("current+shared", ["spend", "budget"], "bound K1 spend-guard evidence + this kitchen's budget check"),
    "12.8": ("shared", ["families"], "the approved family manifest (the registered task set is common)"),
    "12.9": ("current", ["gemma"], "live digest probe of the served Gemma"),
    "12.10": ("shared", ["workflow_b"], "bound K1 Workflow-B evidence (the memory machinery is common code)"),
    "12.11": ("current", ["renderer"], "AST of the K23 entry points + the test namespace's effective contract"),
    "12.12": ("current", ["class_registration"], "the K23 form against the pre-capture probe registry"),
    "12.13": ("current", ["namespace"], "empty test namespace + its ceilings at the freeze"),
    "A12": ("not_applicable", [], "superseded by the A24 package (as in Kitchen 1; the capture audit records 'not applicable')"),
    "A14": ("shared", ["reset"], "see 12.4"),
    "G-iso": ("current", ["g_iso"], "rehearsal-kN-v1 g_iso.json"),
    "X-iso": ("not_applicable", [], "no Codex arm in Kitchens 2 / 3 (A29 item 1)"),
    "a24_2_cost": ("current", [], "C5 energy metered at the test plan (k23_run energy gate before execute / score); A33 price basis shared"),
    "acceptance": ("not_applicable", [], "the acceptance block is Kitchen 1's (A29 item 1: no acceptance block in Kitchens 2 / 3)"),
    "context": ("current", ["context"], "rehearsal capture, measured"),
    "fb-1": ("current", ["fb1"], "reproduced now"),
    "leakage": ("current", ["masking_leakage", "practice"], "per presentation at capture + practice snapshot leakage gate"),
    "registration": ("not_applicable", [], "the acceptance registration is Kitchen 1's"),
    "signatures": ("current+shared", ["amendments"], "K1 signatures (signatures.json) stay in force; the K23 texts must be in the "
                   "preregistration the freeze binds"),
    "snapshot": ("current", ["practice"], "the kitchen's K = 10 snapshot"),
    "tokens": ("current", ["tokens"], "rehearsal capture blocks"),
}
A29_1_MARKER = "A29.1"
PRESIGN_BASIS = "approve-all-6 item 1"


def g(ok, **ev):
    return {"pass": bool(ok), **ev}


def _n(v) -> int:
    """An audit count: an int, or a list of entries (its length); None / missing = -1 (unmeasured, never 0)."""
    if v is None:
        return -1
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple, dict)):
        return len(v)
    return int(v)


def _latest(ns, kind, root):
    p = C.journal_path(ns, kind, root)
    return list(C.Journal(p).latest().values()) if os.path.exists(p) else []


def _shared(root, rel):
    p = os.path.join(C.paths(root)["root"], *rel.split("/"))
    try:
        d = _json(p)
    except (OSError, ValueError):
        return None, None
    return d, C.sha256_file(p)


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rows(ns, kind, root):
    p = C.journal_path(ns, kind, root)
    return C.Journal(p).rows() if os.path.exists(p) else []


def _complete_event(ns, phase, root, journal_kind):
    evs = [e for e in _rows(ns, "events", root) if e.get("event") == "phase_complete" and e.get("phase") == phase]
    if not evs:
        return None, "no phase_complete"
    jp = C.journal_path(ns, journal_kind, root)
    if evs[-1].get("journal_sha") and (not os.path.exists(jp) or C.sha256_file(jp) != evs[-1]["journal_sha"]):
        return evs[-1], f"{journal_kind}.jsonl changed after its phase_complete"
    return evs[-1], None


# ================================================================================================ gates
def gate_pins(ctx):
    import treatment as T
    p = T.pin_problems()
    return g(not p, problems=p[:6])


def gate_a37(ctx):
    p = K.a37_hold_problems(ctx["root"])
    return g(not p, problems=p[:4])


def gate_perception(ctx):
    sc, root = ctx["scene"], ctx["root"]
    p = list(E.perception_problems(sc, root))                     # the WSL-share mapping on Windows, the plain check in WSL
    rec = os.path.join(C.paths(root)["root"], "sim", "kitchen_v3", sc, "frozen_weights.sha256")
    if sc == E.COCO_SCENE:
        p += E.coco_problems(root)
    ev = {"record": E.rel(rec, root), "record_sha256": C.sha256_file(rec) if os.path.exists(rec) else None}
    meta = os.path.join(os.path.dirname(rec), "frozen_weights.meta.json")
    ev["meta_sha256"] = C.sha256_file(meta) if os.path.exists(meta) else None
    return g(not p, problems=p[:6], **ev)


def gate_class_registration(ctx):
    import k23_run as R
    import type_scope as TS
    import treatment as T
    sc, root, ns = ctx["scene"], ctx["root"], ctx["ns"]["test"]
    why = []
    tp = R.types_path(sc, root)
    try:
        dt = _json(tp)
    except (OSError, ValueError) as e:
        return g(False, problems=[f"type list unreadable: {type(e).__name__}"])
    types = list(dt.get("types") or [])
    base = C.paths(root)["root"]
    try:
        reg = sorted(set(_json(os.path.join(base, "sim", "kitchen_v3", sc, "probe.json"))["registry"]))
    except (OSError, ValueError, KeyError) as e:
        reg = None
        why.append(f"(a) probe registry unreadable ({type(e).__name__})")
    try:
        exp = _json(os.path.join(base, "sim", "kitchen_v3", "k23", "expected_classes.json"))["scenes"][sc]
    except (OSError, ValueError, KeyError) as e:
        exp = {}
        why.append(f"(a) expected_classes.json has no {sc} entry ({type(e).__name__})")
    if reg is not None and sorted(set(types)) != reg:
        why.append(f"(a) the B5 list ({len(types)}) differs from the probe registry ({len(reg)})")
    if exp and (sorted(set(exp.get("classes") or [])) != sorted(set(types)) or exp.get("n_classes") != len(types)):
        why.append("(a) the B5 list differs from expected_classes.json (classes / n_classes)")
    _t, tbad = R.type_list_problems(ctx["ns"]["rehearsal"], root)          # schema, scene, weights sha = the perception freeze's primary
    why += ["(a) " + x for x in tbad]
    shown = TS.filter_types(sorted(set(types)))
    if exp and exp.get("n_shown") != len(shown):
        why.append(f"(b) shown list {len(shown)} != the registered n_shown {exp.get('n_shown')}")
    import plan as PL
    ct = PL.make_contract(sorted(set(types)), ns)
    sysx = ct.system_text()
    if ", ".join(T.planner_types(sorted(set(types)))) not in sysx:
        why.append("(b) the system text does not contain the shown list")
    rns = ctx["ns"]["rehearsal"]
    cap = [r for r in _rows(rns, "capture", root) if int(r.get("capture_run", 1)) == 1]
    plans = list(C.Journal(C.journal_path(rns, "plan", root)).latest().values()) if os.path.exists(C.journal_path(rns, "plan", root)) else []
    ssha = {(r.get("audit") or {}).get("system_sha") for r in cap}
    tsha = {r.get("types_sha") for r in plans}                          # Astra G3 3: a missing types_sha on ANY row fails
    if not cap or ssha != {C.sha256_text(sysx)}:
        why.append(f"(c) rehearsal capture rows carry {len(ssha)} system text sha(s), not the frozen text's")
    if not plans or tsha != {C.sha256_json(sorted(T.planner_types(sorted(set(types)))))}:
        why.append(f"(c) rehearsal plan rows carry {len(tsha)} types sha(s), not the shown list's")
    texts = {"system": sysx, "tool_lines": json.dumps(getattr(ct, "tools", None) or [], ensure_ascii=False)}
    try:
        import kitchen_briefing as KB
        for i, rec in enumerate(KB.load_records(sc, allow_unapproved=False)):
            texts[f"briefing#{i}"] = json.dumps(rec, ensure_ascii=False)
    except Exception as e:                                # noqa: BLE001
        why.append(f"(d) approved briefing unreadable ({type(e).__name__})")
    for r in cap:
        texts[f"{r.get('pres')}#summary"] = "\n".join(r.get("prompt_summary") or [])
        for s, b in (r.get("blocks") or {}).items():
            if isinstance(b, dict) and b.get("text"):
                texts[f"{r.get('pres')}#{s}"] = b["text"]
    hits = {k: v for k, v in ((k, T.excluded_alias_hits(v)) for k, v in texts.items()) if v}
    if hits:
        why.append(f"(d) excluded-type names in {len(hits)} agent-facing text(s): {sorted(hits)[:3]}")
    return g(not why, problems=why[:8], n_classes=len(types), n_shown=len(shown), types_sha256=C.sha256_json(sorted(set(types))),
             weights_sha=dt.get("weights_sha"), registry_sha256=exp.get("registry_sha256"), system_text_sha=C.sha256_text(sysx),
             texts_checked=len(texts), rehearsal_capture_rows=len(cap), rehearsal_plan_rows=len(plans))


def gate_samples(ctx):
    p = E.samples_problems(ctx["root"])
    return g(not p, problems=p[:4])


def gate_roster(ctx):
    sc, root = ctx["scene"], ctx["root"]
    try:
        roster = K.a29_roster(ctx["ns"]["test"], root)
    except RuntimeError as e:
        return g(False, problems=[str(e)])
    import k23_size_decision as SD
    try:
        dec = (SD.load_record(root).get("kitchens") or {}).get(sc) or {}
    except SD.DecisionRefused as e:
        return g(False, problems=[str(e)])
    why = []
    if sc == "FloorPlan17" and dec.get("checkpoint") != "after-fp15":
        why.append("FloorPlan17: the size must be re-selected at the after-fp15 checkpoint (A29 item 6) before its freeze")
    if dec.get("roster_sha256") != C.sha256_json(roster):
        why.append("the roster differs from the decision's roster_sha256")
    ctx["roster"] = roster
    return g(not why, problems=why, candidate=dec.get("candidate"), checkpoint=dec.get("checkpoint"), n_items=len(roster),
             presentations=2 * len(roster), roster_sha256=C.sha256_json(roster))


def gate_package(ctx):
    import ltm_package as LP
    sc, root = ctx["scene"], ctx["root"]
    p = E.package_problems(sc, root)
    files = {E.rel(LP.uses_path(sc), root): C.sha256_file(LP.uses_path(sc)), E.rel(LP.inventory_path(sc), root): C.sha256_file(LP.inventory_path(sc))}
    lm = os.path.join(C.paths(root)["root"], "sim", "step5", f"lookalike_map_{sc}.json")
    if os.path.exists(lm):
        files[E.rel(lm, root)] = C.sha256_file(lm)
    return g(not p, problems=p[:4], files=files, hint_table_sha=__import__("treatment").hint_table_sha(E.briefing_table(ctx["ns"]["test"])))


def gate_practice(ctx):
    import practice_k as PK
    import treatment as T
    root, ns = ctx["root"], ctx["ns"]["practice"]
    why = []
    rows = [r for r in _rows(ns, "practice", root) if r.get("namespace") == ns and r.get("order") == 0]
    if len(rows) != PRACTICE_K or sorted(int(r.get("attempt", -1)) for r in rows) != list(range(1, PRACTICE_K + 1)):
        why.append(f"{len(rows)} order-0 practice rows, the registered K = {PRACTICE_K} needs exactly attempts 1..{PRACTICE_K}")
    evs = [e for e in _rows(ns, "events", root) if e.get("event") == "phase_complete" and e.get("phase") == "practice"]
    jp = C.journal_path(ns, "practice", root)
    if not evs or not os.path.exists(jp) or evs[-1].get("journal_sha") != C.sha256_file(jp):
        why.append("no practice phase_complete bound to the current journal")
    ident = ctx["ident"]
    jb = T.journal_problems(rows, ns, ident)
    if jb:
        why.append(f"practice rows of another treatment identity: {jb[:2]}")
    sd = T.source_dir(ctx["ns"]["test"], "S*", root)
    sb = PK.verify_snapshot_dir(sd, 0, PRACTICE_K) if os.path.isdir(sd) else ["S* snapshot missing"]
    why += [f"S*: {x}" for x in sb]
    try:
        tid = T.identity(ns, root)["sha"]
    except Exception as e:                                # noqa: BLE001
        tid = None
        why.append(f"practice identity: {type(e).__name__}")
    if tid != ident:
        why.append("the practice namespace's treatment identity differs from the test namespace's")
    why += E.order_evidence_problems(ns, "practice", root)          # prereg A39.2 (FloorPlan15): practice ran after COCO, via the wrapper
    import k23_leakage as KL
    sm = PK.read_snapshot_manifest(sd) if os.path.isdir(sd) else None
    why += KL.snapshot_rule_problems((sm or {}).get("leakage"), root)   # an own-fragment pass must name the bound decision
    snap_sha = C.dir_sha(sd) if os.path.isdir(sd) else None
    ctx["snapshots"] = {"S*": snap_sha} if snap_sha else {}
    return g(not why, problems=why[:6], attempts=len(rows), snapshot_dir=E.rel(sd, root), snapshot_dir_sha=snap_sha,
             practice_journal_sha=C.sha256_file(jp) if os.path.exists(jp) else None)


def gate_rehearsal(ctx):
    import matrix_run as MR
    import treatment as T
    root, ns = ctx["root"], ctx["ns"]["rehearsal"]
    why = []
    gates = [e for e in _rows(ns, "events", root) if e.get("event") == "capture_gates"]
    if not gates or gates[-1].get("failed") or gates[-1].get("partial"):
        why.append("rehearsal capture gates missing, failed or partial")
    for ph, jk in (("capture", "capture"), ("execute", "exec")):
        ev, bad = _complete_event(ns, ph, root, jk)
        if ev is None or bad:
            why.append(f"rehearsal {ph}: {bad}")
    pf = [e for e in _rows(ns, "events", root) if e.get("event") == "plan_first_complete"]
    if not pf or pf[-1].get("scope") != "full":
        why.append("rehearsal plan: no full plan_first_complete")
    st = MR.status(ns, root)
    if not st["phases"]["score"]["complete"]:
        why.append("rehearsal score not sealed complete")
    seal = os.path.join(C.ns_docs(ns, root), "score.seal.json")
    sp = C.journal_path(ns, "score", root)
    if os.path.exists(seal) and os.path.exists(sp) and _json(seal).get("sha256") != C.sha256_file(sp):
        why.append("rehearsal score.jsonl no longer matches its seal")
    if st.get("invalid_open"):
        why.append("an unresolved invalid execute run")
    why += E.order_evidence_problems(ns, "capture_start", root)     # prereg A39.2 (FloorPlan15): rehearsal ran after COCO, via the wrapper
    import k23_capture_freeze as KCF
    why += KCF.journal_problems(ns, root)                            # every capture_start ran on the re-pin view of the Kitchen 1 freeze
    ex = list(C.Journal(C.journal_path(ns, "exec", root)).latest().values()) if os.path.exists(C.journal_path(ns, "exec", root)) else []
    viol = sum(max(0, _n((r.get("audit") or {}).get("violations"))) for r in ex)
    inval = sum(1 for r in ex if r.get("status") == "invalid" or (r.get("audit") or {}).get("invalid"))
    if viol or inval:
        why.append(f"rehearsal exec rows: {viol} violation(s), {inval} invalid")
    rows = _rows(ns, "capture", root) + list(C.Journal(C.journal_path(ns, "plan", root)).latest().values() if os.path.exists(C.journal_path(ns, "plan", root)) else [])
    jb = T.journal_problems(rows + ex, ns, ctx["ident"])
    if jb:
        why.append(f"rehearsal rows of another treatment identity: {jb[:2]}")
    led = os.path.join(C.ns_work(ns, root), "ledger.db")
    billed = None
    if os.path.exists(led):
        lg = C.ScopedLedger(led, C.ns_check(ns)["ceilings"], release_stale=False)
        billed = lg.billed_scope()
        if billed > float(C.ns_check(ns)["ceilings"]["total"]) + 1e-9:
            why.append(f"rehearsal billed {billed} > its ceiling")
    return g(not why, problems=why[:8], exec_rows=len(ex), violations=viol, invalid=inval, billed_usd=billed)


def gate_bilingual(ctx):
    import gate_evidence as GE
    root, ns = ctx["root"], ctx["ns"]["rehearsal"]
    cap = [r for r in _rows(ns, "capture", root) if int(r.get("capture_run", 1)) == 1]
    starts = [e for e in _rows(ns, "events", root) if e.get("event") == "capture_start"]
    worlds = list((starts[-1] if starts else {}).get("worlds") or [])
    want = GE.expected_bilingual_keys(C.load_manifest(root), ns, worlds)
    bc = GE.bilingual_coverage(cap, want)
    return g(bc["ok"] and worlds == ["canon"], worlds=worlds, **{k: bc[k] for k in ("registered_keys", "pairs_compared", "missing_or_one_language",
                                                                                      "extra", "differences")})


def gate_honesty(ctx):
    root = ctx["root"]
    rns, pns = ctx["ns"]["rehearsal"], ctx["ns"]["practice"]
    tot = {"violations": 0, "oracle": 0, "map_build": 0, "invalid": 0, "missing": 0, "rows": 0}

    def add(viol, oracle, mb, invalid=0):
        vals = [_n(viol), _n(oracle), _n(mb)]
        tot["rows"] += 1
        if min(vals) < 0:
            tot["missing"] += 1
            return
        tot["violations"] += vals[0]
        tot["oracle"] += vals[1]
        tot["map_build"] += vals[2]
        tot["invalid"] += int(bool(invalid))
    for r in _rows(rns, "capture", root):
        h = (r.get("audit") or {}).get("honesty") or {}
        add(h.get("violations"), h.get("oracle"), h.get("map_build"))
    for r in _rows(pns, "practice", root) + _latest(rns, "exec", root):
        au = r.get("audit") or {}
        cnt = au.get("counts") or {}
        add(au.get("violations"), cnt.get("oracle"), cnt.get("map-build"), au.get("invalid"))
    gi = os.path.join(C.ns_docs(rns, root), "g_iso.json")
    gaud = (((_json(gi) if os.path.exists(gi) else {}).get("components") or {}).get("audit") or {})
    ok = tot["rows"] > 0 and tot["missing"] == 0 and tot["violations"] + tot["oracle"] + tot["map_build"] + tot["invalid"] == 0 \
        and gaud.get("absent_vs_present") == 0
    return g(ok, **tot, g_iso_audit=gaud)


def gate_g_iso(ctx):
    root, ns = ctx["root"], ctx["ns"]["rehearsal"]
    p = os.path.join(C.ns_docs(ns, root), "g_iso.json")
    if not os.path.exists(p):
        return g(False, problems=[f"no G-iso report ({E.rel(p, root)}): run k23_run --phase execute --namespace {ns} -- --g-iso"])
    rep_ = _json(p)
    why = []
    if rep_.get("sha256") != C.self_sha(rep_) or rep_.get("namespace") != ns:
        why.append("G-iso report not sealed / not this namespace")
    import k23_giso_waiver as GW
    waived = None
    if rep_.get("status") == "inconclusive" and ctx["scene"] == GW.SCENE:
        wp = GW.waiver_problems(root)                              # A20.1 written waiver, narrowed claim: FloorPlan17 only, diff == [] only
        if wp:
            why += [f"G-iso inconclusive and not waived: {w}" for w in wp]
        else:
            waived = {"decision": GW.DECISION_ID, "decision_sha256": GW.decision_sha(root), "claim": GW.CLAIM}
    elif rep_.get("status") != "pass" or int(rep_.get("n") or 0) < GISO_N or int(rep_.get("n_registered") or 0) != GISO_N:
        why.append(f"G-iso status {rep_.get('status')} n {rep_.get('n')} (registered {GISO_N})")
    ev = dict(status=rep_.get("status"), n=rep_.get("n"), sha256=C.sha256_file(p))
    if waived:
        ev.update(waived=True, waiver_decision_sha256=waived["decision_sha256"], narrowed_claim=waived["claim"])
    return g(not why, problems=why, **ev)


def gate_masking_leakage(ctx):
    root, ns = ctx["root"], ctx["ns"]["rehearsal"]
    ev = [e for e in _rows(ns, "events", root) if e.get("event") == "capture_gates"]
    gt = (ev[-1].get("gates") or {}) if ev else {}
    n = len({r.get("key") for r in _rows(ns, "capture", root)})
    why = []
    for k in ("masking", "leakage", "g_iso", "key_leak"):
        if (gt.get(k) or {}).get("pass") is not True:
            why.append(f"capture gate {k} not passed")
    if any(int((gt.get(k) or {}).get("n") or -1) < n for k in ("masking", "leakage")) or not n:
        why.append("masking / leakage did not check every captured presentation")
    return g(not why, problems=why, presentations=n, **{k: (gt.get(k) or {}).get("n") for k in ("masking", "leakage", "g_iso")})


def gate_tokens(ctx):
    import treatment as T
    root, ns = ctx["root"], ctx["ns"]["rehearsal"]
    mx, n, missing = 0, 0, 0
    cap = T.FT2["package_max_tokens"]
    for r in _rows(ns, "capture", root):
        for s_, b in (r.get("blocks") or {}).items():
            if s_ == "none" or not isinstance(b, dict) or not b.get("text"):
                continue
            n += 1
            if b.get("real_tokens") is None:
                missing += 1
                continue
            mx = max(mx, int(b["real_tokens"]))
    return g(n > 0 and not missing and mx <= cap, blocks=n, max_real_tokens=mx, cap=cap, missing=missing)


def gate_context(ctx):
    root, ns = ctx["root"], ctx["ns"]["rehearsal"]
    p = os.path.join(C.ns_docs(ns, root), "context_fit.json")
    if not os.path.exists(p):
        return g(False, problems=[f"no context_fit.json: run k23_run --phase context --namespace {ns} (Ollama up)"])
    d = _json(p)
    cp = C.journal_path(ns, "capture", root)
    why = []
    if d.get("freeze_grade") is not True or not str(d.get("counting") or "").startswith("MEASURED"):
        why.append(f"context fit not freeze-grade / not measured ({d.get('verdict')}, {d.get('counting')})")
    if d.get("namespace") != ns or not os.path.exists(cp) or d.get("capture_sha") != C.sha256_file(cp):
        why.append("context_fit.json is not bound to the current rehearsal capture")
    return g(not why, problems=why, verdict=d.get("verdict"), sha256=C.sha256_file(p))


def gate_evaluator(ctx):
    root, sc, rns = ctx["root"], ctx["scene"], ctx["ns"]["rehearsal"]
    why = []
    m = os.path.join(C.paths(root)["docs"], "k23", f"a29_sample_{sc}.json")
    ap = os.path.join(C.paths(root)["docs"], "k23", "applicability", sc, "applicability.json")
    try:
        man = _json(m)
        app_sha = C.sha256_json(_json(ap))
        if man.get("applicability_sha256") != app_sha:
            why.append("the A29 sample was not drawn from the current applicability record")
    except (OSError, ValueError):
        app_sha = None
        why.append("A29 sample / applicability record unreadable")
    sc_rows = _rows(rns, "score", root)
    p0 = [r for r in sc_rows if str(r.get("arm")) == "P0" or "|P0|" in str(r.get("cell") or r.get("key") or "")]
    if not p0:
        why.append("no scored rehearsal P0 row")
    d, sha = _shared(root, SHARED_EVIDENCE["12.5"])
    if not d or (d.get("status") or ("PASS" if d.get("pass") else None)) != "PASS":
        why.append("the shared Kitchen 1 12.5 evidence (empty-plan control) is missing or not PASS")
    return g(not why, problems=why, applicability_sha256=app_sha, rehearsal_p0_rows=len(p0), shared_125_sha256=sha)


def gate_reset(ctx):
    import freeze as F
    r = F.gate_tolerances(C.paths(ctx["root"])["root"])          # Astra G3-r2: the RESOLVED root (freeze.py joins it)
    return g(r["pass"], disposition="shared", **(r.get("evidence") or {}))


def gate_spend(ctx):
    d, sha = _shared(ctx["root"], SHARED_EVIDENCE["12.7"])
    ok = bool(d) and (d.get("status") or ("PASS" if d.get("pass") else None)) == "PASS"
    return g(ok, disposition="shared", evidence=SHARED_EVIDENCE["12.7"], sha256=sha)


def gate_families(ctx):
    import freeze as F
    r = F.gate_families(C.paths(ctx["root"])["root"])
    return g(r["pass"], disposition="shared", **(r.get("evidence") or {}))


def gate_gemma(ctx):
    import freeze as F
    info = (ctx.get("probes") or {}).get("ollama") if ctx.get("probes") is not None else F.probe_ollama()
    r = F.gate_ollama(info)
    return g(r["pass"], **(r.get("evidence") or {}))


def gate_workflow_b(ctx):
    d, sha = _shared(ctx["root"], SHARED_EVIDENCE["12.10"])
    ok = bool(d) and (d.get("status") or ("PASS" if d.get("pass") else None)) == "PASS"
    return g(ok, disposition="shared", evidence=SHARED_EVIDENCE["12.10"], sha256=sha)


def gate_renderer(ctx):
    import ast
    base = C.paths(ctx["root"])["root"]
    calls = {}
    for rel in CODE_ENTRIES:
        if not os.path.basename(rel).startswith("k23_"):
            continue
        p = os.path.join(base, *rel.split("/"))
        if not os.path.exists(p):
            continue
        for node in ast.walk(ast.parse(open(p, encoding="utf-8").read())):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
                if name == "render_messages":
                    calls.setdefault(rel, []).append(node.lineno)
    import plan as PL
    tl = os.path.join(C.paths(ctx["root"])["matrix_docs"], f"detector_types_{ctx['scene']}.json")
    types = sorted(set((_json(tl) if os.path.exists(tl) else {}).get("types") or []))
    ct = PL.make_contract(types, ctx["ns"]["test"]) if types else None
    return g(not calls and ct is not None, render_calls=calls, contract_version=getattr(ct, "version", None),
             system_sha=C.sha256_text(ct.system_text()) if ct else None)


def gate_namespace(ctx):
    root, ns = ctx["root"], ctx["ns"]["test"]
    rows = {k: len(_rows(ns, k, root)) for k in ("capture", "plan", "exec", "score")}
    cache = os.path.join(C.ns_work(ns, root), "cache")
    n_cache = sum(len(fs) for _d, _s, fs in os.walk(cache)) if os.path.isdir(cache) else 0
    return g(not any(rows.values()) and n_cache == 0, rows=rows, cache_files=n_cache, ceilings=C.ns_check(ns)["ceilings"])


def gate_fb1(ctx):
    import freeze_ft2 as F2
    r = F2.fb1_record(C.paths(ctx["root"])["root"])
    return g(r.get("ok"), problems=list(r.get("reasons") or [])[:4])


def gate_amendments(ctx):
    txt = open(C.paths(ctx["root"])["prereg"], encoding="utf-8").read()
    ok = any(ln.lstrip("# ").startswith(A29_1_MARKER) for ln in txt.splitlines() if ln.startswith("#"))
    return g(ok, problems=[] if ok else [f"the preregistration has no {A29_1_MARKER} heading: append the K23 texts (A29.1, ...) BEFORE the "
                                         "freeze (it binds the preregistration hash)"], prereg_sha256=C.sha256_file(C.paths(ctx["root"])["prereg"]))


def gate_fa_examples(ctx):
    import k23_fa_examples as FAX
    root = ctx["root"]
    why = list(FAX.decision_problems(root))
    sha = FAX.decision_sha(root)
    used = []
    for ns in (ctx["ns"]["practice"], ctx["ns"]["rehearsal"]):
        used += [(ns, e.get("decision_sha256")) for e in _rows(ns, "events", root) if e.get("event") == FAX.EVENT]
    other = sorted({u for u in used if u[1] != sha})
    if other:
        why.append(f"an activation used another decision record than the one bound now: {other[:3]}")
    return g(not why, problems=why, decision_sha256=sha, activations=len(used))


def equivalence_path(scene: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "applicability", scene, "equivalence_fa.json")


def gate_fa_equivalence(ctx):
    import a29_applicability as AA
    root, sc = ctx["root"], ctx["scene"]
    p = equivalence_path(sc, root)
    if not os.path.exists(p):
        return g(False, problems=[f"no FA equivalence result ({E.rel(p, root)}): run a29_applicability.py --kitchen {sc} --equivalence "
                                  "(A29.1 item 4; required before this kitchen's agent freeze)"])
    try:
        d = _json(p)
        summary = {k: d.get(k) for k in ("kitchen", "n", "agreement")}              # summary fields only, never the rows
    except (OSError, ValueError) as e:
        return g(False, problems=[f"FA equivalence result unreadable ({type(e).__name__})"])
    try:
        pool = int(_json(K.a29_files(sc, root)[0]).get("n_pool"))
    except (OSError, ValueError, TypeError):
        pool = None
    why = []
    want = min(AA.EQUIV_N, pool) if pool is not None else None
    if summary["kitchen"] != sc:
        why.append(f"the FA equivalence result names kitchen {summary['kitchen']!r}")
    if want is None or summary["n"] != want:
        why.append(f"FA equivalence n {summary['n']} != the registered sample size {want}")
    a = summary["agreement"]
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(summary["n"], int) or not 0 <= a <= summary["n"]:
        why.append(f"FA equivalence agreement {a!r} is not a count within 0..n")
    return g(not why, problems=why, **summary, sha256=C.sha256_file(p), registered_n=want)


def gate_dispositions(ctx):
    why = [k for k in K1_GATES if k not in DISPOSITIONS]
    bad = {k: [x for x in v[1] if x not in GATES] for k, v in DISPOSITIONS.items()}
    why += [f"{k}: unknown K23 gate(s) {v}" for k, v in bad.items() if v]
    return g(not why, problems=why, table={k: {"disposition": v[0], "k23_gates": v[1], "basis": v[2]} for k, v in DISPOSITIONS.items()})


def gate_dataset(ctx):
    root = ctx["root"]
    man = C.load_manifest(root)
    want = (man.get("inclusion") or {}).get("dataset_sha256") or man.get("dataset_sha256")
    got = C.sha256_file(C.paths(root)["dataset"])
    ctx["dataset_sha256"] = got
    return g(got == want, dataset_sha256=got, split_names=want)


def gate_budget(ctx):
    import acc_common as AC
    root, sc = ctx["root"], ctx["scene"]
    sp = AC.study_spend(root)
    import k23_size_decision as SD
    try:
        cps = SD.load_record(root).get("checkpoints") or []
    except SD.DecisionRefused:
        cps = []
    proj = None
    for cp in reversed(cps):
        for t in cp.get("trials") or []:
            if t.get("size") == cp.get("selected") and sc in (t.get("kitchens") or {}):
                proj = (t["kitchens"][sc] or {}).get("retained_usd")
                break
        if proj is not None:
            break
    why = []
    if proj is None:
        why.append("no projected test cost for this kitchen in the size decision record")
        proj = 0.0
    led = os.path.join(C.paths(root)["matrix_work"], C.ns_storage(ctx["ns"]["test"]), "ledger.db")
    why += AC.study_wide_problems(proj, spend=sp) + K.replication_problems(led, proj, sp)
    ceil = float(C.ns_check(ctx["ns"]["test"])["ceilings"]["total"])
    if proj > ceil:
        why.append(f"projected test cost {proj} > the namespace ceiling {ceil}")
    return g(not why, problems=why[:4], projected_test_usd=proj, study_committed_usd=sp.get("committed"), namespace_ceiling_usd=ceil)


def gate_selftests(ctx):
    if ctx.get("no_selftests"):
        return g(False, problems=["selftests not run (--no-selftests)"])
    root = C.paths(ctx["root"])["root"]
    res, why = {}, []
    for rel in SELFTESTS:
        p = os.path.join(root, *rel.split("/"))
        if os.path.basename(p) == "_k23_freeze_agent_selftest.py" and ctx.get("inside_selftest"):
            continue
        try:
            r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=1800, cwd=root,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"))
            last = (r.stdout.strip().splitlines() or [""])[-1]
            res[rel] = {"rc": r.returncode, "last": last[:120], "sha256": C.sha256_file(p)}
            if r.returncode != 0:
                why.append(f"{rel}: rc {r.returncode} ({last[:80]})")
        except subprocess.TimeoutExpired:
            res[rel] = {"rc": None, "last": "timeout"}
            why.append(f"{rel}: timeout")
    return g(not why, problems=why, runs=res, python=sys.executable)


GATES = {"pins": gate_pins, "a37_hold": gate_a37, "perception": gate_perception, "class_registration": gate_class_registration,
         "samples": gate_samples, "roster": gate_roster, "package": gate_package, "practice": gate_practice, "rehearsal": gate_rehearsal,
         "dataset": gate_dataset, "budget": gate_budget, "selftests": gate_selftests, "bilingual": gate_bilingual,
         "honesty": gate_honesty, "g_iso": gate_g_iso, "masking_leakage": gate_masking_leakage, "tokens": gate_tokens,
         "context": gate_context, "evaluator": gate_evaluator, "reset": gate_reset, "spend": gate_spend, "families": gate_families,
         "gemma": gate_gemma, "workflow_b": gate_workflow_b, "renderer": gate_renderer, "namespace": gate_namespace, "fb1": gate_fb1,
         "amendments": gate_amendments, "dispositions": gate_dispositions, "fa_equivalence": gate_fa_equivalence,
         "fa_examples": gate_fa_examples}


# ================================================================================================ build / write / sign
def build(scene: str, root: str | None = None, *, no_selftests: bool = False, inside_selftest: bool = False, now: str | None = None,
          probes: dict | None = None) -> dict:
    import treatment as T
    import freeze as F
    E.install_perception_reader()
    if scene not in NS:
        raise ValueError(f"{scene} is not a replication kitchen")
    ns = NS[scene]
    ctx = {"scene": scene, "root": root, "ns": ns, "no_selftests": no_selftests, "inside_selftest": inside_selftest, "probes": probes}
    try:
        ctx["ident"] = T.identity(ns["test"], root)["sha"]
    except Exception as e:                                # noqa: BLE001  (pins moved: the identity fails closed)
        ctx["ident"] = None
        ctx["ident_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    gates = {}
    for name in GATE_ORDER:
        try:
            gates[name] = GATES[name](ctx)
        except Exception as e:                            # noqa: BLE001  (a gate that cannot run never passes)
            gates[name] = g(False, problems=[f"gate error {type(e).__name__}: {str(e)[:200]}"])
    if ctx["ident"] is None:
        gates["pins"] = g(False, problems=[ctx.get("ident_error")] + list(gates["pins"].get("problems") or []))
    base = C.paths(root)["root"]
    files, missing, errors = F.import_closure(base, list(CODE_ENTRIES))
    code = {rel: C.sha256_file(os.path.join(base, *rel.split("/"))) for rel in files}
    code.update({rel: C.sha256_file(os.path.join(base, *rel.split("/"))) for rel in SHELL_FILES if os.path.exists(os.path.join(base, *rel.split("/")))})
    for rel in bound_files(scene):                                 # data the test reads, re-verified at every entry like code (12.1)
        if os.path.exists(os.path.join(base, *rel.split("/"))):
            code[rel] = C.sha256_file(os.path.join(base, *rel.split("/")))
    if missing or errors:
        gates["code"] = g(False, problems=[f"missing entries {missing}", f"parse errors {errors[:3]}"])
    tl = os.path.join(C.paths(root)["matrix_docs"], f"detector_types_{scene}.json")
    types = sorted(set((_json(tl) if os.path.exists(tl) else {}).get("types") or []))
    fz = {"schema": SCHEMA, "created": now or time.strftime("%Y-%m-%dT%H:%M:%S"), "scene": scene, "namespaces": [ns["test"]],
          "related_namespaces": {"practice": ns["practice"], "rehearsal": ns["rehearsal"]}, "split_sha256": C.SPLIT_SHA256,
          "prereg_sha256": C.sha256_file(C.paths(root)["prereg"]), "treatment_id": ctx["ident"],
          "dataset_sha256": ctx.get("dataset_sha256"), "detector_types_sha256": C.sha256_json(types) if types else None,
          "a29_roster_sha256": C.sha256_json(ctx["roster"]) if ctx.get("roster") is not None else None,
          "snapshots": ctx.get("snapshots") or {}, "code_entries": list(CODE_ENTRIES) + list(SHELL_FILES), "code": code,
          "perception": {k: gates["perception"].get(k) for k in ("record", "record_sha256", "meta_sha256")},
          "gates": gates, "gate_order": list(GATE_ORDER) + (["code"] if "code" in gates else []),
          "k1_gate_dispositions": {k: {"disposition": v[0], "k23_gates": v[1], "basis": v[2]} for k, v in DISPOSITIONS.items()},
          "shared_evidence": {k: {"path": v, "sha256": _shared(root, v)[1]} for k, v in SHARED_EVIDENCE.items()},
          "design": {"planning_arms": list(C.CORE_ARMS), "execution_controls": ["P0", "P+M"], "world": "canon", "r": 0, "replays": False,
                     "stages": ["free", "core0"], "tier": "synchronous (approve-all-6 item 3)", "fresh_simulator_every": 300},
          "builder": {"tool": "sim/final_test/matrix/k23_freeze_agent.py", "python": sys.executable, "platform": sys.platform}}
    fz["all_gates_pass"] = all(v.get("pass") is True for v in gates.values())
    fz["sha256"] = C.self_sha(fz)
    return fz


def bound_files(scene: str) -> tuple:
    """Data files the K23 test reads besides the identity-hashed ones, bound in the record's 'code' map (re-verified at every entry)."""
    return (f"sim/step4/{scene}.map.json", f"sim/step5/lookalike_map_{scene}.json", f"sim/step5/uses_table_{scene}.json",
            f"sim/step5/homeowner_inventory_{scene}.json", f"docs/final-test/matrix/detector_types_{scene}.json",
            f"docs/final-test/k23/a29_sample_{scene}.json", f"docs/final-test/k23/applicability/{scene}/equivalence_fa.json",
            "docs/final-test/split_manifest.json",
            "docs/final-test/families_approved.json", "docs/final-test/reset_tolerances.json", "docs/final-test/signatures.json",
            "docs/final-test/k23/decisions/K23-FA-EXAMPLES.json", "docs/final-test/k23/decisions/K23-PRACTICE-LEAK.json") + (
            (f"docs/final-test/k23/decisions/K23-GISO-WAIVER-{scene}.json", "sim/final_test/_checkpoints/K3-GISO-INCONCLUSIVE/g_iso.json")
            if scene == "FloorPlan17" else ())


def summary_path(scene: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", f"FREEZE-SUMMARY-{scene}.md")


def summary_text(fz: dict) -> str:
    sc = fz["scene"]
    lines = [f"# Kitchen agent freeze: {sc} ({fz['namespaces'][0]})", "",
             f"Written {fz['created']} by k23_freeze_agent.py. Freeze record docs/final-test/k23/freeze_{sc}.json, self-sha {fz['sha256'][:16]}.",
             "Every K23 test entry re-checks this record (k23_ns.replication_freeze_status, k23_entry.test_entry_problems).", "",
             "| gate | result | evidence |", "|---|---|---|"]
    for gname in fz.get("gate_order") or sorted(fz["gates"]):
        v = fz["gates"][gname]
        ev = "; ".join(v.get("problems") or []) or ", ".join(f"{k}={v[k]}" for k in v if k not in ("pass", "problems", "runs", "files")
                                                             and not isinstance(v[k], (dict, list)))[:300]
        lines.append(f"| {gname} | {'PASS' if v.get('pass') else 'FAIL'} | {ev} |")
    r = fz["gates"].get("roster") or {}
    lines += ["", "## What is bound", "",
              f"- treatment identity {str(fz.get('treatment_id'))[:16]}; dataset {str(fz.get('dataset_sha256'))[:16]}; B5 types {str(fz.get('detector_types_sha256'))[:16]}",
              f"- A29 roster: candidate {r.get('candidate')} ({r.get('n_items')} items, {r.get('presentations')} presentations), checkpoint {r.get('checkpoint')}",
              f"- practice snapshot S*: {fz.get('snapshots', {}).get('S*')}",
              f"- perception record {(fz.get('perception') or {}).get('record_sha256')}",
              f"- {len(fz.get('code') or {})} code files hashed; preregistration {fz['prereg_sha256'][:16]}; split {fz['split_sha256'][:16]}"]
    gi = fz["gates"].get("g_iso") or {}
    if gi.get("waived"):
        lines += ["", "## Narrowed claim (G-iso waived in writing by Arash, A20.1)", "",
                  f"G-iso status {gi.get('status')} (n {gi.get('n')}), waived by decision record {str(gi.get('waiver_decision_sha256'))[:16]}. "
                  f"The memory-isolation claim for this kitchen is narrowed to: {gi.get('narrowed_claim')}"]
    lines += ["", "## Sign-off", "",
              f"Arash's standing pre-signature ({PRESIGN_BASIS}) applies only because every gate passed, Astra agreed (review file named in "
              "the record) and this summary was written for him to read afterwards. Any gate failure stops and waits for him."]
    return "\n".join(lines) + "\n"


def _write_json(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def write(scene: str, root: str | None = None, *, supersede: bool = False, fz: dict | None = None) -> dict:
    fz = fz or build(scene, root)
    if not fz["all_gates_pass"]:
        failed = [k for k, v in fz["gates"].items() if not v.get("pass")]
        raise RuntimeError(f"not written: gates failed {failed}")
    p = K.freeze_path(scene, root)
    if os.path.exists(p):
        if not supersede:
            raise RuntimeError(f"{E.rel(p, root)} exists; --supersede archives it first")
        hist = os.path.join(os.path.dirname(p), "freeze_history")
        os.makedirs(hist, exist_ok=True)
        shutil.copy2(p, os.path.join(hist, f"freeze_{scene}.{C.sha256_file(p)[:16]}.json"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _write_json(p, fz)
    with open(summary_path(scene, root), "w", encoding="utf-8", newline="\n") as f:
        f.write(summary_text(fz))
    return fz


def _load(scene, root):
    p = K.freeze_path(scene, root)
    fz = _json(p)
    if fz.get("schema") != SCHEMA or fz.get("sha256") != C.self_sha(fz):
        raise RuntimeError("freeze record schema / self-sha mismatch")
    return fz


def presign(scene: str, astra_review: str, root: str | None = None) -> dict:
    """Arash's standing pre-signature (approve-all-6 item 1): every gate passed, Astra agreed (the review's answer has a line
    'RULING: ACCEPTED'), the freeze summary exists."""
    fz = _load(scene, root)
    base = C.paths(root)["root"]
    if not fz.get("all_gates_pass") or not all(v.get("pass") is True for v in fz["gates"].values()):
        raise RuntimeError("not every gate passed: the standing pre-signature does not apply (Arash decides)")
    ar = astra_review.replace("\\", "/")
    ap = ar if os.path.isabs(ar) else os.path.join(base, *ar.split("/"))
    sp = summary_path(scene, root)
    if not os.path.exists(ap) or not os.path.exists(sp):
        raise RuntimeError("the Astra review file or the freeze summary is missing")
    if open(sp, encoding="utf-8").read() != summary_text(fz):
        raise RuntimeError("the freeze summary is not the one generated for THIS record (re-run --write, or regenerate it)")
    ans = K.a37_review_answer(open(ap, encoding="utf-8").read())
    first = next((ln.strip() for ln in ans.splitlines() if ln.strip().startswith("RULING:")), "")
    if first != "RULING: ACCEPTED" or scene not in ans or fz["sha256"][:16] not in ans:      # Astra G3 3: bound to THIS freeze
        raise RuntimeError("the Astra review is not an unconditional 'RULING: ACCEPTED' whose answer names this kitchen AND this "
                           f"freeze's self-sha prefix {fz['sha256'][:16]}")
    fz["standing_presignature"] = {"basis": PRESIGN_BASIS, "astra_review": E.rel(ap, root), "astra_review_sha256": C.sha256_file(ap),
                                   "freeze_summary": E.rel(sp, root), "freeze_summary_sha256": C.sha256_file(sp),
                                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "presigned_sha": fz["sha256"]}
    fz["sha256"] = C.self_sha(fz)
    _write_json(K.freeze_path(scene, root), fz)
    return fz


def record_arash_check(scene: str, by: str, date: str, said: str, root: str | None = None) -> dict:
    if by != "Arash" or not date or not said:
        raise RuntimeError("--record-arash-check needs --by Arash --date YYYY-MM-DD --arash-said \"<his words>\"")
    fz = _load(scene, root)
    fz["arash_check"] = {"by": by, "date": date, "said": said[:500], "checked_sha": fz["sha256"]}
    fz["sha256"] = C.self_sha(fz)
    _write_json(K.freeze_path(scene, root), fz)
    return fz


def check(scene: str, root: str | None = None) -> dict:
    E.install_perception_reader()
    ns = NS[scene]["test"]
    st = K.replication_freeze_status(ns, root)
    ent = E.test_entry_problems(ns, root)
    return {"scene": scene, "namespace": ns, "freeze_status": st, "entry_problems": ent, "ok": bool(st.get("ok")) and not ent}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K23 agent freeze (freeze_<scene>.json)")
    ap.add_argument("--scene", required=True, choices=sorted(NS))
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--write", action="store_true")
    m.add_argument("--presign", action="store_true")
    m.add_argument("--record-arash-check", action="store_true")
    m.add_argument("--check", action="store_true")
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--no-selftests", action="store_true", help="dry run only: skip the selftest gate (it then FAILs)")
    ap.add_argument("--no-probe", action="store_true", help="dry run only: do not probe the local Ollama (gate gemma then FAILs)")
    ap.add_argument("--astra-review", default=None)
    ap.add_argument("--by"), ap.add_argument("--date"), ap.add_argument("--arash-said")
    a = ap.parse_args(argv)
    try:
        if a.check:
            r = check(a.scene)
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 1
        if a.presign:
            if not a.astra_review:
                raise RuntimeError("--presign needs --astra-review <path>")
            fz = presign(a.scene, a.astra_review)
            print(json.dumps({"presigned": True, "sha256": fz["sha256"]}))
            return 0
        if a.record_arash_check:
            fz = record_arash_check(a.scene, a.by, a.date, a.arash_said)
            print(json.dumps({"arash_check": True, "sha256": fz["sha256"]}))
            return 0
        if a.write and a.no_selftests:
            raise RuntimeError("--write always runs the selftest gate")
        if a.write and a.no_probe:
            raise RuntimeError("--write always probes the served Gemma (gate 12.9)")
        fz = build(a.scene, no_selftests=a.no_selftests, probes={"ollama": None} if a.no_probe else None)
        print(json.dumps({"scene": a.scene, "all_gates_pass": fz["all_gates_pass"],
                          "gates": {k: ("PASS" if v.get("pass") else "FAIL: " + ("; ".join(v.get("problems") or []) or json.dumps(
                              {kk: vv for kk, vv in v.items() if kk not in ("pass", "problems") and not isinstance(vv, (dict, list))}))[:240])
                                    for k, v in fz["gates"].items()}},
                         indent=1, ensure_ascii=False))
        if a.write:
            write(a.scene, supersede=a.supersede, fz=fz)
            print("wrote", E.rel(K.freeze_path(a.scene)), "and", E.rel(summary_path(a.scene)))
        return 0 if fz["all_gates_pass"] else 1
    except (RuntimeError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
