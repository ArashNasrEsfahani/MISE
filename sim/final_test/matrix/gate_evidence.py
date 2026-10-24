"""W4: write the freeze evidence files docs/final-test/matrix/gates/<id>.json (schema ftm.gate/1) from REAL files only.
(ACC-RUNNER, 2026-10-14: `--ft2` writes the ft-v2 evidence gates_ft2/<id>.json (schema ftm.gate/2, bound to its inputs); see ft2_evidence.)

Offline (any python). Reads: the dev capture on the real S* (rehearsal-v1/capture.jsonl + events), the practice journal and the 15 snapshot
manifests (practice-v1), the split manifest, the step-5 workflow-B journals, the W2 rehearsal archive (reference-plan scores, empty-plan
control, G-iso report, spend ledger). Every verdict is computed here from those files; nothing is typed by hand. G-iso is written with
pass=false: the registered strict rule cannot pass (the absent-vs-absent control differs as often as absent-vs-present), and a rule is
never reinterpreted after the fact (Astra W4, docs/final-test/_astra/W4-practice.txt).

    python sim/final_test/matrix/gate_evidence.py [--write]        (default: print the verdicts, write nothing)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

ROOT = C.ROOT
ARCH = os.path.join(ROOT, "docs", "final-test", "matrix", "_archive-W2-rehearsal-v1")
GATES = os.path.join(ROOT, "docs", "final-test", "matrix", "gates")
NOW = time.strftime("%Y-%m-%dT%H:%M:%S")


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def sha(p):
    return C.sha256_file(p)


def gate(gid, passed, evidence, source):
    return {"schema": "ftm.gate/1", "gate": gid, "pass": bool(passed), "synthetic": False, "evidence": evidence, "source": source, "created": NOW}


def x_iso_verdict(rep, flags_now: str, model: str = "gpt-6-sol") -> tuple:
    """A21 E gate X-iso: the Codex X-arm isolation probe (`x_isolation_probe.py`, docs/final-test/matrix/x_isolation.json) passed for the CURRENT flag set.
    -> (pass, reasons). Pure; the verdict is computed from the report, never typed."""
    why = []
    if not isinstance(rep, dict) or rep.get("schema") != "ftm.x-isolation/1":
        return False, ["no x_isolation.json report (run x_isolation_probe.py --write)"]
    if rep.get("model") != model:
        why.append(f"report is for {rep.get('model')!r}, not {model!r}")
    if rep.get("pass") is not True or rep.get("problems"):
        why.append("the probe itself did not pass: " + "; ".join(map(str, rep.get("problems") or ["pass is not true"]))[:200])
    if rep.get("flags_sha") != flags_now:
        why.append("the CLI flag set changed since the probe (hardening list, effort, sandbox): probe again")
    if int(rep.get("n_calls") or 0) < 5 or int(rep.get("trials") or 0) < 2:
        why.append("too few probe calls (needs identity and canary calls, 2 trials each)")
    if int(rep.get("canary_leaks") or 0) != 0:
        why.append("a canary outside the cwd was read")
    if not rep.get("cli_version"):
        why.append("CLI version not recorded")
    if sorted(rep.get("canary_places") or []) != ["repo_docs", "repo_work", "sibling_temp"]:
        why.append("canary places incomplete")
    return not why, why


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--ft2", action="store_true", help="ACC-RUNNER F5: the ft-v2 evidence (gates_ft2/; bound; never touches gates/)")
    a = ap.parse_args(argv)
    if a.ft2:
        import freeze as F
        return main_ft2(a.write, probes={"codex": F.probe_codex()})
    man = C.load_manifest()
    test = set(man["test"]) | set(man.get("test_uncapped_not_scored") or [])
    nontest = set(man["nontest"])
    cap_p = C.journal_path("rehearsal-v1", "capture")
    cap = jl(cap_p)
    ev = jl(os.path.join(C.ns_docs("rehearsal-v1"), "events.jsonl"))
    prac_p = C.journal_path("practice-v1", "practice")
    prac = jl(prac_p)
    out = {}

    # ---- 12.2 bilingual retrieval: EN and FA of one dev item give the same memory record ids, for S* and every curve snapshot
    by = {}
    for r in cap:
        for s, b in (r.get("blocks") or {}).items():
            if s in ("none", "FS"):
                continue
            by.setdefault((r["item"], s), {})[r["lang"]] = list(b.get("ids") or [])
    pairs = {k: v for k, v in by.items() if set(v) == {"en", "fa"}}
    diff = sorted("%s/%s" % k for k, v in pairs.items() if v["en"] != v["fa"])
    s_items = sorted({k[0] for k in pairs if k[1] == "S*"})
    nonempty = sum(1 for k, v in pairs.items() if k[1] == "S*" and v["en"])
    ok = not diff and len(s_items) == 12 and nonempty > 0
    out["12.2"] = gate("12.2", ok, {"claim": "for every dev item the FA instruction yields the same memory-block record ids as the EN one (S* and the 11 curve snapshots)",
                                    "dev_items": len(s_items), "pairs_compared": len(pairs), "pairs_with_records_in_S*": nonempty, "differences": diff,
                                    "parser_version": 2, "lexicon_sha256": sha(os.path.join(ROOT, "sim", "step5", "lexicon", "concepts.json")),
                                    "capture_journal_sha256": sha(cap_p)},
                       "dev capture on the real S* (namespace rehearsal-v1, 24 presentations, 2026-10-12)")

    # ---- 12.3 honesty: 0 violations, 0 oracle reads, 0 map-build reads (capture sweeps, practice episodes, empty-plan control)
    hc = {"violations": 0, "oracle": 0, "map_build": 0}
    for r in cap:
        h = (r.get("audit") or {}).get("honesty") or {}
        for k in hc:
            hc[k] += int(h.get(k) or 0)
    hp = {"violations": 0, "oracle": 0, "map_build": 0, "invalid": 0, "episodes": 0}
    for r in prac:
        au = r.get("audit")
        if au:
            hp["episodes"] += 1
            hp["violations"] += int(au.get("violations") or 0)
            hp["oracle"] += int((au.get("counts") or {}).get("oracle") or 0)
            hp["map_build"] += int((au.get("counts") or {}).get("map-build") or 0)
            hp["invalid"] += int(bool(au.get("invalid")))
    emp = jl(os.path.join(ARCH, "empty_plan_control.jsonl"))
    he = {"episodes": len(emp), "violations": sum(int(x.get("violations") or 0) for x in emp), "oracle": sum(int(x.get("oracle_reads") or 0) for x in emp)}
    ok = (sum(hc.values()) + hp["violations"] + hp["oracle"] + hp["map_build"] + hp["invalid"] + he["violations"] + he["oracle"] == 0
          and len(cap) == 24 and hp["episodes"] == 30 and he["episodes"] == 12)
    out["12.3"] = gate("12.3", ok, {"claim": "0 honesty violations, 0 oracle reads, 0 map-build reads", "dev_capture_sweeps": {"presentations": len(cap), **hc},
                                    "practice_episodes": hp, "empty_plan_control_dev": he,
                                    "files": {"capture": sha(cap_p), "practice": sha(prac_p), "empty_plan_control_W2_archive": sha(os.path.join(ARCH, "empty_plan_control.jsonl"))}},
                       "capture audit + practice audit + W2 empty-plan control")

    # ---- 12.5 evaluator: reference plans reproduce step 4 (W2 archive, scoring code unchanged since), the empty plan scores 0
    import score as S
    tmp = os.path.join(ROOT, "sim", "final_test", "_tmp_w4eval")
    d = os.path.join(tmp, "docs", "final-test", "matrix", "rehearsal-v1")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(ARCH, "score.jsonl"), d)
    try:
        ec = S.evaluator_check("rehearsal-v1", root=tmp, baseline_path=os.path.join(ROOT, S.STEP4_BASELINE), write=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    emp_ok = len(emp) == 12 and all(not x.get("goal_ok") and float(x.get("sr") or 0) == 0.0 and x.get("plan_len") == 0 for x in emp if not x.get("inapplicable"))
    agree_ok = ec["compared"] >= 30 and ec["agreement"] >= S.EVAL_MIN_AGREEMENT
    out["12.5"] = gate("12.5", agree_ok and emp_ok, {"claim": "reference plans (arm P0) on the 12 dev items reproduce the step-4 honest reference-plan outcomes; the empty plan scores 0",
                                                      "reference_plans": {k: ec[k] for k in ("compared", "agree", "agreement", "baseline_sr", "now_sr", "min_agreement", "disagreements")},
                                                      "empty_plan_control": {"items": len(emp), "successes": sum(1 for x in emp if x.get("goal_ok")), "plan_len_max": max(x.get("plan_len") or 0 for x in emp)},
                                                      "files": {"score_W2_archive": sha(os.path.join(ARCH, "score.jsonl")), "step4_baseline": sha(os.path.join(ROOT, S.STEP4_BASELINE))},
                                                      "note": "scored in the W2 dev rehearsal (2026-10-12); the evaluator and executor code is part of the frozen code hash, so a later change is caught by freeze --check"},
                       "score.evaluator_check on the W2 rehearsal scores + empty_plan_control.jsonl")

    # ---- 12.7 spend guard: reconciliation of the real ledgers and a dry run of the A15 guard on real per-call usage
    def ledger(p):
        c = sqlite3.connect(p)
        rows = c.execute("SELECT provider,state,COUNT(*),COALESCE(SUM(reserved),0),COALESCE(SUM(actual),0) FROM ledger GROUP BY provider,state").fetchall()
        c.close()
        return [{"provider": x, "state": y, "n": n, "reserved": round(r, 6), "actual": round(s, 6)} for x, y, n, r, s in rows]
    w2 = ledger(os.path.join(ARCH, "work", "ledger.db"))
    pil = ledger(os.path.join(ROOT, "sim", "final_test", "pilot", "ledger.db"))
    open_w2 = [x for x in w2 if x["state"] == "open"]
    plan_rows = jl(os.path.join(ARCH, "plan.jsonl"))
    billed = sum(float(x.get("cost_usd") or 0) for x in plan_rows if not x.get("cached") and str(x.get("arm", ""))[:1] in "OSH")
    settled = sum(x["actual"] for x in w2 if x["state"] == "settled" and x["provider"].startswith("anthropic"))
    ref = C.OPUS_REF_USD_PER_CALL
    g1 = C.cost_guard(opus_calls=60, opus_usd=60 * ref, checked_at=0, spent=5.0, remaining={"opus": 700}, mean_cost={"opus": ref}, ceiling=float(C.NAMESPACES["test-v1"]["ceilings"]["total"]))
    g2 = C.cost_guard(opus_calls=60, opus_usd=60 * 3 * ref, checked_at=0, spent=5.0, remaining={"opus": 700}, mean_cost={"opus": ref}, ceiling=float(C.NAMESPACES["test-v1"]["ceilings"]["total"]))
    g3 = C.cost_guard(opus_calls=60, opus_usd=60 * ref, checked_at=0, spent=5.0, remaining={"opus": 700}, mean_cost={"opus": 0.2}, ceiling=float(C.NAMESPACES["test-v1"]["ceilings"]["total"]))
    ok = (not open_w2) and abs(settled - billed) < 1e-4 and not g1["pause"] and g2["pause"] and g3["pause"]
    out["12.7"] = gate("12.7", ok, {"claim": "the spend ledger reconciles with the per-call costs, holds no open reservation, and the A15 guard pauses on a regime change only",
                                    "w2_rehearsal_ledger": w2, "w2_open_reservations": len(open_w2), "w2_settled_anthropic_usd": round(settled, 6),
                                    "w2_sum_of_non_cached_call_costs_usd": round(billed, 6),
                                    "pilot_ledger_note": pil, "guard_dry_run": {"at_pilot_mean": g1, "at_3x_pilot_mean": g2, "projection_over_ceiling": g3},
                                    "decided_by_arash": "A19: guard baseline 2 x 0.012 = 0.024 USD per Opus call (was 0.0178 = 2 x 0.0089 on the pre-A11 prompts)"},
                       "ScopedLedger reconciliation of the W2 rehearsal + common.cost_guard dry run (the ceiling argument is the registered test-v1 total, A22: 65 USD)")

    # ---- 12.10 workflow-B data is final-test non-test only
    pat = re.compile(r"g-l[123]-[a-z0-9\-]+")
    ids, hits, nfiles = set(), {}, 0
    for f in glob.glob(os.path.join(ROOT, "sim", "step5", "live", "**", "*"), recursive=True) + glob.glob(os.path.join(ROOT, "sim", "step5", "_b1_gates", "**", "*"), recursive=True):
        if os.path.isfile(f) and f.endswith((".json", ".jsonl")):
            nfiles += 1
            t = open(f, encoding="utf-8", errors="replace").read()
            found = set(pat.findall(t))
            ids |= found
            if found & test:
                hits[rel(f)] = sorted(found & test)[:5]
    used = {r["item"] for r in prac} | {r["item"] for r in cap}
    ok = nfiles > 0 and not hits and not (ids & test) and not (used & test) and ids <= nontest and used <= nontest
    out["12.10"] = gate("12.10", ok, {"claim": "workflow-B journals, practice and the dev capture contain only final-test non-test items", "step5_live_files_scanned": nfiles,
                                      "step5_item_ids": len(ids), "step5_ids_in_test": len(ids & test), "step5_ids_outside_nontest": len(ids - nontest),
                                      "practice_and_capture_items": len(used), "in_test": len(used & test), "files_with_test_ids": hits},
                        "regex scan of sim/step5/live + sim/step5/_b1_gates against the split manifest")

    # ---- G-iso: written as measured, not reinterpreted. A19 rule: the live dev report (ftm.g_iso/2, written by execute.py --g-iso on rehearsal-v1)
    # decides; without one, the W2 report under the old strict rule stays as the (failing) evidence.
    cg = [x for x in ev if x.get("event") == "capture_gates"][-1]["gates"]["g_iso"]
    live_p = os.path.join(C.ns_docs("rehearsal-v1"), "g_iso.json")
    live = json.load(open(live_p, encoding="utf-8")) if os.path.isfile(live_p) else None
    if live and live.get("schema") == "ftm.g_iso/2" and live.get("rule") in ("A19", "A20", "A20.1"):
        okg = (live.get("rule") == "A20.1" and live.get("status") == "pass" and (live.get("executions") or {}).get("all_real") is True
               and (live.get("audit_sensitivity") or {}).get("ok") is True and not live.get("instrumentation_missing")
               and live.get("status") == "pass" and int(live.get("n") or 0) >= int(live.get("n_registered") or 12) >= 12 and not live.get("diff")
               and not live.get("control_noise_bad") and len(live.get("control_noise") or []) <= int(live.get("control_noise_max") or 0)
               and not live.get("blocked") and not live.get("short") and live.get("sha256") == C.self_sha(live))
        out["G-iso"] = gate("G-iso", okg, {"claim": "A20.1 rule (A19 components): (i) request bytes, (ii) status, (iii) goal reached, (iv) executed steps, (v) violations + oracle + map-build reads are identical "
                                                    "under absent / present / absent memory files for every selected presentation (poses and timings excluded)",
                                           "status": live.get("status"), "n": live.get("n"), "n_registered": live.get("n_registered"), "diff": live.get("diff"),
                                           "control_noise": live.get("control_noise"), "control_noise_components": live.get("control_noise_components"), "control_noise_max": live.get("control_noise_max"), "rule": live.get("rule"), "blocked": live.get("blocked"), "components": live.get("components"),
                                           "inconclusive_reasons": live.get("inconclusive_reasons"), "executions": live.get("executions"), "audit_sensitivity": live.get("audit_sensitivity"),
                                           "instrumentation_missing": live.get("instrumentation_missing"),
                                           "strict_digest_informational": live.get("strict_digest_informational"),
                                           "report": rel(live_p), "report_sha256": sha(live_p), "prompt_level_gate_in_this_capture": cg},
                            "execute.py --g-iso fresh dev run on rehearsal-v1 (A19) + the capture G-iso gate")
    else:
        gi_p = os.path.join(ARCH, "g_iso.json")
        out["G-iso"] = gate("G-iso", False, {"claim": "INCONCLUSIVE under the preregistered strict rule (6.3 / 12): the absent-vs-absent control also differs 12/12 (coordinate jitter inside the digest). "
                                                       "The A19 rule (components i-v) has NOT been re-validated yet on fresh dev runs",
                                             "execution_level_report_W2": rel(gi_p), "execution_level_report_sha256": sha(gi_p),
                                             "prompt_level_gate_in_this_capture": cg, "decision": "run execute.py --g-iso (WSL, n = 12, three conditions) under A19; this file stays pass=false until a pass report exists"},
                            "W2 g_iso.json + capture G-iso gate of the W4 dev capture; Astra W4 advice (write pass=false)")

    # ---- leakage
    snaps = sorted(glob.glob(os.path.join(ROOT, "sim", "final_test", "snapshots", "o*", "k*", "practice_manifest.json")))
    lk = [json.load(open(p, encoding="utf-8")).get("leakage") or {} for p in snaps]
    fam = man["families"]
    mixed = [f["family"] for f in fam if {i in test for i in f["items"]} == {True, False}]
    pool = set(man["practice_pool"])
    pool_in_test_family = sorted({i for f in fam if f.get("role") == "test" for i in f["items"]} & pool)
    cap_leak = [x for x in ev if x.get("event") == "capture_gates"][-1]["gates"]["leakage"]
    ok = (len(snaps) == 15 and all(x.get("ok") and not x.get("hits") for x in lk) and cap_leak.get("pass") and not mixed and not pool_in_test_family
          and not (pool & test) and {r["item"] for r in prac} <= pool)
    out["leakage"] = gate("leakage", ok, {"claim": "no test item id or normalised test instruction in any practice snapshot or any rendered block; practice items share no family with test items",
                                          "practice_snapshots_checked": len(snaps), "snapshot_hits": sum(len(x.get("hits") or []) for x in lk),
                                          "capture_leakage_gate": cap_leak, "families_total": len(fam), "families_mixed_test_and_nontest": mixed,
                                          "practice_pool_items_in_test_families": pool_in_test_family, "practice_pool_items_in_test": len(pool & test),
                                          "practice_attempts_outside_pool": sorted({r["item"] for r in prac} - pool),
                                          "scope_note": "family disjointness is checked on the split manifest (metadata only, no test instruction text is read); locations of shared kitchen objects in memory are the intended transfer"},
                          "15 practice snapshot manifests + capture leakage gate + split manifest families")

    # ---- X-iso (A21 E): the isolation probe of the Codex X arms, valid for the current flag set only
    xp = os.path.join(ROOT, "docs", "final-test", "matrix", "x_isolation.json")
    xrep = json.load(open(xp, encoding="utf-8")) if os.path.isfile(xp) else None
    import x_isolation_probe as XP
    xok, xwhy = x_iso_verdict(xrep, XP.flags_sha("gpt-6-sol"))
    out["X-iso"] = gate("X-iso", xok, {"claim": "the gpt-6-sol X arms run in a clean ephemeral read-only session with the hardened tool surface; the session reports the requested model, "
                                               "the pinned effort, approval Never and a restricted network; canary files outside the cwd cannot be read; the CLI version is recorded",
                                       "reasons": xwhy, "cli_version": (xrep or {}).get("cli_version"), "flags_sha": (xrep or {}).get("flags_sha"),
                                       "n_calls": (xrep or {}).get("n_calls"), "canary_leaks": (xrep or {}).get("canary_leaks"),
                                       "visible_tool_events_total": (xrep or {}).get("visible_tool_events_total"), "limits": (xrep or {}).get("limits"),
                                       "report": rel(xp) if xrep else None, "report_sha256": sha(xp) if xrep else None},
                      "x_isolation_probe.py (real CLI, real provider flags) + the current provider flag set")

    for gid, g in out.items():
        print(("PASS " if g["pass"] else "FAIL ") + gid, json.dumps({k: v for k, v in g["evidence"].items() if k in ("claim", "dev_items", "differences", "practice_episodes", "reference_plans", "step5_item_ids", "practice_snapshots_checked")}, ensure_ascii=False)[:300])
    if a.write:
        os.makedirs(GATES, exist_ok=True)
        for gid, g in out.items():
            raw = json.dumps(g, indent=1, ensure_ascii=False)
            if C.key_leak(raw):
                raise SystemExit("key-like text in evidence for " + gid)
            with open(os.path.join(GATES, gid + ".json"), "w", encoding="utf-8", newline="\n") as f:
                f.write(raw + "\n")
        print("wrote", len(out), "evidence files to", rel(GATES))
    return 0


# ================================================================================================ ft-v2 evidence (stage ACC-RUNNER, F5)
#: F5 (freeze_dryrun.json gaps; Astra FREEZE-DRYRUN Q1): the ft-v2 gates are re-derived from the ft-v2 files only and written to a NEW
#: folder (gates_ft2/), so the 2026-10-12 ft-v1 files in gates/ are never overwritten and never read by the ft-v2 freeze. Every file is
#: BOUND: the sha of every input it read, the treatment identity of the namespaces it used (recomputed now), the evaluator / tokenizer /
#: CLI versions, and this generator's own sha. freeze_ft2.read_evidence re-verifies each binding against the disk (missing or mismatched
#: = FAIL). A gate whose registered evidence does not exist yet is written with status PENDING (pass false) and says what is owed.
GATE2_SCHEMA = "ftm.gate/2"
FT2_DEV_NS, FT2_PRAC_NS = "rehearsal-v3", "practice-v2"
#: A31 DONE owed item (4): these need captures under the FINAL identity (their dev evidence alone never passes them)
FINAL_IDENTITY_GATES = ("G-iso", "tokens", "context", "leakage")
FT2_GATE_IDS = ("12.2", "12.3", "12.5", "12.7", "12.10", "G-iso", "leakage", "X-iso", "tokens", "context")
#: Astra ACC-RUNNER r2 held 12.2 / 12.3 / 12.7 at PENDING; cycle 3 closed the three findings in code (12.2: coverage against the
#: REGISTERED dev set, `bilingual_coverage`; 12.3: row-level identity binding, `honesty_identity`; 12.7: the study-wide boundary and the
#: hold reconciliation, proven by `spend_probes` on every evidence run). A gate listed here is still forced to PENDING (none now).
ASTRA_R2_HELD = {}
#: A31 re-pin record: practice-v2 RAN under d5831abc...; it binds to the final identity only through the A31 equivalence certificate
PRACTICE_RUN_IDENTITY = "d5831abc"
#: FREEZE-GATES-FT2 (2026-10-15): G-iso and 12.5 under the FINAL identity come from this namespace only, bound to this identity only
#: (= freeze.json treatment.sha = the acceptance registration's pin). Any other identity, in the rows or in the live code, is a FAIL.
FG_NS = "freeze-gates-ft2"
FT2_FINAL_IDENTITY = "7a2ee211b85cc534d7677291fce80b04e92cd24a860f1ddd7665185b3a03b67d"
A26_125_REG = ("docs", "final-test", "a26_12_5_registration.json")


def _id_problems(final_sha, rows_by_kind: dict) -> list:
    """Identity binding of the FG_NS evidence: the live final identity and every row must be FT2_FINAL_IDENTITY."""
    why = [] if final_sha == FT2_FINAL_IDENTITY else [f"the live final identity {str(final_sha)[:12]} is not the registered {FT2_FINAL_IDENTITY[:12]}"]
    for kind, rows in rows_by_kind.items():
        bad = sorted({str(r.get("treatment_id"))[:12] for r in rows if r.get("treatment_id") != FT2_FINAL_IDENTITY})
        if bad:
            why.append(f"{kind} rows under another identity: {bad[:3]}")
    return why


def g_iso_final_verdict(gi, *, final_sha, cap_rows, plan_rows) -> tuple:
    """G-iso under the final identity, pure. -> (status, reasons): PENDING = no FG_NS report yet; FAIL = wrong identity / namespace /
    altered report, or a report that is not an A20.1 pass (inconclusive is not passed); PASS otherwise."""
    if not isinstance(gi, dict):
        return "PENDING", [f"no {FG_NS} g_iso.json yet (execute.py --namespace {FG_NS} --g-iso)"]
    why = _id_problems(final_sha, {"capture": cap_rows, "plan": plan_rows})
    if not cap_rows:
        why.append(f"{FG_NS} has no capture rows")
    if gi.get("namespace") != FG_NS or gi.get("sha256") != C.self_sha(gi):
        why.append("g_iso.json is not the intact FG_NS report (namespace / self sha)")
    if gi.get("treatment_id") != FT2_FINAL_IDENTITY:
        why.append(f"g_iso.json identity {str(gi.get('treatment_id'))[:12]} is not {FT2_FINAL_IDENTITY[:12]}")
    rule_ok = (gi.get("rule") == "A20.1" and gi.get("status") == "pass" and (gi.get("executions") or {}).get("all_real") is True
               and (gi.get("audit_sensitivity") or {}).get("ok") is True and not gi.get("instrumentation_missing")
               and int(gi.get("n") or 0) >= int(gi.get("n_registered") or 12) >= 12 and not gi.get("diff") and not gi.get("blocked")
               and not gi.get("short") and not gi.get("hook_b_on_zero_arm"))
    if not rule_ok:
        why.append(f"not an A20.1 pass: status {gi.get('status')!r}, diff {gi.get('diff')}, reasons {gi.get('inconclusive_reasons')}")
    return ("FAIL" if why else "PASS"), why


#: FINAL-IDENTITY tokens / context / leakage (2026-10-15, A36; Astra A36-DRAFT-r1 Q4: "Tokens/context may use verified attempt-2 evidence;
#: leakage still needs its required audit"). The evidence is the acceptance attempt-2 captures under FT2_FINAL_IDENTITY; the verdict is
#: authenticated exactly as freeze_ft2.acceptance_record does (acc_attempt2.verdict_problems, bundle replay); acceptance NOT PASS overall is
#: the A36 waiver's business, not these gates': each gate needs only ITS criteria to pass and its evidence files to be the recorded bytes.
ACC_FINAL_NS = "acceptance-v2-a2"
ACC_FINAL_CRITERIA = {"tokens": ("g", "h"), "context": ("i",), "leakage": ("c11",)}
ACC_FINAL_FILES = {"tokens": ("g_eligible.json", "capture.jsonl"), "context": ("context_fit.json", "capture.jsonl"),
                   "leakage": ("c11_acceptance.json", "capture.jsonl", "plan.jsonl")}


def _acc_file_content_problems(gid: str, docs: dict, shas: dict) -> list:
    """The evidence file's own verdict, beyond its bytes: it is a clean report for ACC_FINAL_NS on THIS capture."""
    cap = shas.get("capture.jsonl")
    if gid == "tokens":
        g = docs.get("g_eligible.json")
        if not isinstance(g, dict):
            return ["g_eligible.json unreadable"]
        ok = (g.get("namespace") == ACC_FINAL_NS and g.get("ok") is True and not g.get("problems") and g.get("sha256") == C.self_sha(g)
              and g.get("treatment_id") == FT2_FINAL_IDENTITY and g.get("capture_sha") == cap)
        return [] if ok else ["g_eligible.json is not a clean, intact ACC-G report on this capture under the final identity"]
    if gid == "context":
        c = docs.get("context_fit.json")
        if not isinstance(c, dict):
            return ["context_fit.json unreadable"]
        bad = [m for m, b in (c.get("by_model") or {}).items() for k in ("FAIL_MEASURED", "FAIL_KNOWN", "UNRESOLVED_EST") if (b.get("verdicts") or {}).get(k)]
        ok = (c.get("namespace") == ACC_FINAL_NS and c.get("gemma_verdict") == "PASS_MEASURED" and c.get("freeze_grade") is True
              and not c.get("not_freeze_grade_because") and c.get("capture_sha") == cap and not bad and int(c.get("n_checks") or 0) > 0)
        return [] if ok else [f"context_fit.json is not a freeze-grade measured Gemma pass on this capture (failing models {bad})"]
    c = docs.get("c11_acceptance.json")
    if not isinstance(c, dict):
        return ["c11_acceptance.json unreadable"]
    ok = (c.get("namespace") == ACC_FINAL_NS and c.get("ok") is True and not c.get("problems") and not c.get("system_text_hits")
          and c.get("capture_sha") == cap and c.get("plan_sha") == shas.get("plan.jsonl") and int(c.get("requests") or 0) > 0
          and c.get("bound") == c.get("requests") and c.get("first_bound") == c.get("first_transmissions") and c.get("repair_bound") == c.get("repair_turns"))
    return [] if ok else ["c11_acceptance.json is not a clean, fully bound C11 scan of this capture / plan"]


def acc_final_verdict(gid: str, v, auth_problems, *, final_sha, shas_now: dict, docs: dict) -> tuple:
    """tokens / context / leakage(acceptance part) under the final identity, pure. v = the attempt-2 verdict (None = not recorded),
    auth_problems = acc_attempt2.verdict_problems(2, replay=True), shas_now = {evidence file: current sha or None}, docs = parsed files.
    -> (status, reasons): PENDING = no verdict; FAIL = not authentic / other identity / a criterion not passing / evidence changed."""
    if not isinstance(v, dict):
        return "PENDING", [f"no {ACC_FINAL_NS} acceptance verdict recorded"]
    why = [f"verdict not authentic: {x}"[:300] for x in (auth_problems or [])]
    if final_sha != FT2_FINAL_IDENTITY:
        why.append(f"the live final identity {str(final_sha)[:12]} is not the registered {FT2_FINAL_IDENTITY[:12]}")
    if v.get("sha256") != C.self_sha(v) or v.get("namespace") != ACC_FINAL_NS or v.get("attempt") != 2:
        why.append("the verdict is not the intact attempt-2 acceptance-v2-a2 verdict (self sha / namespace / attempt)")
    if v.get("treatment_id") != FT2_FINAL_IDENTITY or (v.get("identity") or {}).get("sha") != FT2_FINAL_IDENTITY:
        why.append(f"verdict identity {str(v.get('treatment_id'))[:12]} is not {FT2_FINAL_IDENTITY[:12]}")
    if not (v.get("bundle") or {}).get("manifest_sha"):
        why.append("the verdict records no bundle")
    crit = (v.get("decision") or {}).get("criteria") or {}
    for k in ACC_FINAL_CRITERIA[gid]:
        c = crit.get(k) or {}
        if c.get("pass") is not True or c.get("reasons"):
            why.append(f"criterion {k} does not pass: {(c.get('reasons') or ['missing'])[:2]}")
    ev = v.get("evidence") or {}
    for f in ACC_FINAL_FILES[gid]:
        if shas_now.get(f) is None:
            why.append(f"{f} is missing")
        elif ev.get(f) != shas_now.get(f):
            why.append(f"{f} changed since the verdict (recorded {str(ev.get(f))[:12]}, now {shas_now[f][:12]})")
    if not why:
        why += _acc_file_content_problems(gid, docs, shas_now)
    return ("FAIL" if why else "PASS"), why


def c11_final_problems(c11, *, final_sha, want_bind: dict, script_sha, prereg_sha) -> list:
    """The C11 audit (c11_audit.py) re-run clean and bound to the FINAL identity, now: pure -> reasons (empty = clean)."""
    if not isinstance(c11, dict) or not c11:
        return ["no c11_audit.json (python sim/final_test/diag/c11/c11_audit.py)"]
    b, ctl, why = c11.get("binding") or {}, c11.get("controls") or {}, []
    if final_sha != FT2_FINAL_IDENTITY or b.get("treatment_test_v1") != FT2_FINAL_IDENTITY:
        why.append(f"c11 audit bound to identity {str(b.get('treatment_test_v1'))[:12]}, not the final {FT2_FINAL_IDENTITY[:12]}")
    if c11.get("ok") is not True or c11.get("problems") or int(ctl.get("n") or 0) <= 0 or ctl.get("passed") != ctl.get("n"):
        why.append("c11 audit not clean (ok / problems / planted controls)")
    mis = sorted(k for k, x in want_bind.items() if b.get(k) != x)
    if mis:
        why.append(f"c11 audit binding mismatch: {mis}")
    if b.get("audit_script") != script_sha or b.get("prereg") != prereg_sha:
        why.append("c11 audit ran on another c11_audit.py / preregistration than the current one: re-run it")
    return why


def acc_final_inputs(root) -> dict:
    """Load + authenticate the attempt-2 verdict once (bundle replay, as freeze_ft2.acceptance_record does) and every evidence file."""
    ad = os.path.join(root, "sim", "final_test", "acceptance_v2")
    if ad not in sys.path:
        sys.path.insert(0, ad)
    import acc_attempt2 as A2
    vp = A2.verdict_path(2, root)
    v = json.load(open(vp, encoding="utf-8")) if os.path.isfile(vp) else None
    try:
        auth = A2.verdict_problems(2, root, replay=True) if v is not None else []
    except Exception as e:                                          # noqa: BLE001 -- unverifiable = not authentic
        auth = [f"verdict unverifiable ({type(e).__name__})"]
    d = C.ns_docs(ACC_FINAL_NS, root)
    files = sorted({f for fs in ACC_FINAL_FILES.values() for f in fs})
    paths = {f: os.path.join(d, f) for f in files}
    shas = {f: (C.sha256_file(p) if os.path.isfile(p) else None) for f, p in paths.items()}
    docs = {}
    for f, p in paths.items():
        if f.endswith(".json") and os.path.isfile(p):
            try:
                docs[f] = json.load(open(p, encoding="utf-8"))
            except ValueError:
                docs[f] = None
    return {"verdict": v, "verdict_path": vp, "auth": auth, "shas": shas, "docs": docs, "paths": paths}


def acc_final_part(gid, A, final_sha) -> dict:
    st, why = acc_final_verdict(gid, A["verdict"], A["auth"], final_sha=final_sha, shas_now=A["shas"], docs=A["docs"])
    v = A["verdict"] or {}
    return {"required_identity": FT2_FINAL_IDENTITY, "namespace": ACC_FINAL_NS, "status": st, "reasons": why,
            "criteria": list(ACC_FINAL_CRITERIA[gid]), "evidence_files": {f: A["shas"].get(f) for f in ACC_FINAL_FILES[gid]},
            "verdict": {"file_sha256": C.sha256_file(A["verdict_path"]) if os.path.isfile(A["verdict_path"]) else None,
                        "self_sha256": v.get("sha256"), "verdict": v.get("verdict"), "treatment_id": v.get("treatment_id"),
                        "bundle_manifest_sha": (v.get("bundle") or {}).get("manifest_sha"), "authenticated_by_replay": v != {} and not A["auth"]},
            "acceptance_not_pass_disposition": "acceptance NOT PASS overall is waived by A36 (freeze_ft2.acceptance_record); this gate reads "
                                               "only its own criteria (Astra A36-DRAFT-r1 Q4)",
            "owed": None if st == "PASS" else f"final-identity {gid} evidence from {ACC_FINAL_NS}: {why[:1]}"}


def reg_pres(p: str) -> str:
    """A registered presentation -> the journals' presentation id (C.pres_id: world:item:lang). The A26 12.5 registration (da0b3bbe, written
    before the run, never edited) lists them as item:lang:world; the same three identifiers, reordered only; anything else is kept verbatim
    (and then simply matches no row, so it fails as not executed)."""
    parts = str(p).split(":")
    if len(parts) == 3 and parts[0] not in C.WORLDS and parts[2] in C.WORLDS:
        return f"{parts[2]}:{parts[0]}:{parts[1]}"
    return str(p)


def eval_125_verdict(score_rows, exec_rows, events, reg, reg_file_sha, baseline: dict, *, final_sha, min_agreement) -> dict:
    """Gate 12.5 under A26 (prereg A26 item 9), pure. reg = the registration (a26_12_5_registration.json) or None; baseline = step-4
    {item: goal_ok}. PENDING when nothing was scored yet; FAIL on a missing / altered / late registration, a wrong identity, a P0
    agreement below the registered variance allowance after the registered expected changes, an unrealistic slice, or an empty plan
    that is missing or succeeds; PASS otherwise."""
    rows = [r for r in score_rows or [] if r.get("scored") and r.get("arm") in ("P0", "E0")]
    out = {"p0": None, "e0": None, "reasons": []}
    if not rows:
        out["status"] = "PENDING"
        out["reasons"] = [f"no scored P0 / E0 rows in {FG_NS} yet"]
        return out
    why = out["reasons"]
    why += _id_problems(final_sha, {"exec": exec_rows or []})
    if not isinstance(reg, dict) or reg.get("sha256") != C.self_sha(reg) or reg.get("schema") != "ftm.a26-12.5-registration/1":
        why.append("the A26 12.5 registration is missing or altered")
        reg = {}
    elif reg.get("treatment_identity") != FT2_FINAL_IDENTITY or reg.get("namespace") != FG_NS:
        why.append("the registration names another identity / namespace")
    ev = [e for e in events or [] if e.get("event") == "a26_12_5_registration"]
    first_exec = min((str(r.get("ts")) for r in exec_rows or []), default=None)
    if not ev or ev[0].get("sha256") != reg_file_sha or (first_exec is not None and str(ev[0].get("ts")) > first_exec):
        why.append("no a26_12_5_registration event with this file's sha before the first execution")
    va = float((reg.get("variance_allowance") or {}).get("min_agreement") or -1)
    if abs(va - float(min_agreement)) > 1e-12:
        why.append(f"registered variance allowance {va} is not the previously specified {min_agreement}")
    exp = {x["item"]: x for x in reg.get("items") or []}
    chg = {c.get("item"): c for c in reg.get("expected_changes") or []}
    for i, x in exp.items():                     # the registered expectation = step 4 with exactly the registered changes applied
        want = bool(chg[i].get("expected_outcome")) if i in chg else baseline.get(i)
        if i not in baseline or x.get("expected_outcome") != want:
            why.append(f"{i}: registered expectation {x.get('expected_outcome')} does not follow step 4 ({baseline.get(i)}) + the registered changes")
    pres = [reg_pres(p) for p in reg.get("presentations") or []]          # the registration wrote item:lang:world; rows use C.pres_id
    p0 = {r["pres"]: r for r in rows if r["arm"] == "P0"}
    e0 = {r["pres"]: r for r in rows if r["arm"] == "E0"}
    diff, miss = [], []
    for p in pres:
        r, it = p0.get(p), C.parse_pres(p)["item"]
        if r is None or not r.get("executed"):
            miss.append(p)
        elif bool(r.get("goal_ok")) != bool((exp.get(it) or {}).get("expected_outcome")):
            diff.append({"pres": p, "now": bool(r.get("goal_ok")), "expected": (exp.get(it) or {}).get("expected_outcome"),
                         "registered_change": it in chg})
    agree = len(pres) - len(diff) - len(miss)
    share = agree / len(pres) if pres else 0.0
    out["p0"] = {"registered": len(pres), "executed": len(pres) - len(miss), "agree": agree, "agreement": round(share, 4),
                 "min_agreement": va, "unregistered_differences": [d for d in diff if not d["registered_change"]], "not_executed": miss,
                 "registered_changes": sorted(chg)}
    if not pres:
        why.append("the registration lists no presentation")
    elif share < va:
        why.append(f"P0 agreement {share:.3f} < {va} (every unregistered difference counts against it)")
    if miss:                                     # Astra A36-IMPL-gates r1 (1): the allowance covers measured differences, never a missing execution
        why.append(f"P0 not executed / not scored on {len(miss)} registered presentation(s): {miss[:4]}")
    # Astra A36-IMPL-gates r1 (2): every scored P0 / E0 row of a registered presentation needs its own exec evidence in this namespace's
    # journal (a row listing that cell, status done, identity FT2_FINAL_IDENTITY); an empty or absent journal fails
    ex_by_cell = {}
    for e in exec_rows or []:
        for c in e.get("cells") or ():
            ex_by_cell.setdefault(c, []).append(e)
    no_ev = sorted(f"{r['pres']}#{r['arm']}" for r in rows if r["pres"] in set(pres) and not any(
        e.get("status") == "done" and e.get("treatment_id") == FT2_FINAL_IDENTITY and (r.get("unit") in (None, e.get("unit")))
        for e in ex_by_cell.get(r.get("cell"), ())))
    out["exec_evidence"] = {"exec_rows": len(exec_rows or []), "scored_rows_without_exec_evidence": no_ev}
    if not exec_rows:
        why.append("the namespace's exec journal is empty or absent: scored rows have no execution evidence")
    elif no_ev:
        why.append(f"{len(no_ev)} scored P0 / E0 row(s) without a matching done exec row under {FT2_FINAL_IDENTITY[:12]}: {no_ev[:4]}")
    unreal = sorted(r["pres"] + "#" + r["arm"] for r in rows if r.get("unrealistic_slice") or int(r.get("n_slice_no_knife") or 0))
    if unreal:
        why.append(f"unrealistic_slice at an executed slice: {unreal[:4]}")
    e_bad = sorted(p for p in pres if p not in e0 or not e0[p].get("executed") or e0[p].get("plan_len") != 0)
    e_succ = sorted(p for p, r in e0.items() if r.get("goal_ok"))
    out["e0"] = {"registered": len(pres), "empty_plan_rows": sum(1 for r in e0.values() if r.get("executed") and r.get("plan_len") == 0),
                 "not_measured": e_bad, "successes": e_succ}
    if e_bad:
        why.append(f"empty-plan control missing / not executed / not empty on {len(e_bad)} presentation(s)")
    if e_succ:
        why.append(f"the empty plan succeeded on {e_succ[:4]} (goal true at reset)")
    out["status"] = "FAIL" if why else "PASS"
    return out


def expected_bilingual_keys(man: dict, ns: str, worlds) -> set:
    """The (item, world, source) keys whose package record ids must agree between EN and FA, derived from the REGISTERED dev set (split
    manifest), the namespace's registered relocation / repeat subsets and the capture's registered worlds; never from the rows."""
    items = list(man.get("dev_pilot") or [])
    reloc, rep_ = set(C.reloc_items(ns, man)), set(C.repeat_items(ns, man))
    curve = ["o%dk%d" % ok for ok in C.CURVE_SNAPSHOTS]
    out = set()
    for w in worlds:
        for i in items:
            if w != "canon" and i not in reloc:
                continue
            for src in ["S*"] + (curve if (w == "canon" and i in rep_) else []):
                out.add((i, w, src))
    return out


def bilingual_coverage(run1: list, want: set) -> dict:
    """12.2 on run-1 capture rows: every registered key has BOTH an EN and an FA row carrying a block, the ids agree, nothing extra."""
    by = {}
    for r in run1:
        for src, b in (r.get("blocks") or {}).items():
            if src in ("none", "FS") or not isinstance(b, dict):
                continue
            by.setdefault((r.get("item"), r.get("world"), src), {})[r.get("lang")] = list(b.get("ids") or [])
    missing = sorted("%s/%s/%s" % k for k in want if set(by.get(k) or {}) != {"en", "fa"})
    extra = sorted("%s/%s/%s" % k for k in by if k not in want)
    diff = sorted("%s/%s/%s" % k for k, v in by.items() if k in want and set(v) == {"en", "fa"} and v["en"] != v["fa"])
    return {"ok": bool(want) and not missing and not extra and not diff, "registered_keys": len(want), "missing_or_one_language": missing,
            "extra": extra, "differences": diff, "pairs_compared": len([k for k in want if set(by.get(k) or {}) == {"en", "fa"}])}


def honesty_identity(cap: list, prac: list, *, dev_sha: str, dev_ns: str, prac_run_sha_prefix: str, cert: dict | None, prac_journal_sha) -> dict:
    """12.3 row-level identity binding: every capture row is an ft-v2 row of the dev namespace under the dev identity; every practice row
    ran under the registered practice-run identity, and the A31 certificate binds that journal to the re-pinned identity."""
    bad_cap = sorted({str(r.get("key") or r.get("pres")) for r in cap
                      if r.get("treatment_id") != dev_sha or r.get("namespace") != dev_ns or r.get("treatment") != "ft-v2"})
    bad_prac = sorted({str(r.get("key") or r.get("attempt")) for r in prac if not str(r.get("treatment_id") or "").startswith(prac_run_sha_prefix)})
    cert_ok = bool(cert) and cert.get("equivalent") is True and cert.get("journal_sha") == prac_journal_sha
    return {"ok": not bad_cap and not bad_prac and cert_ok and bool(cap) and bool(prac), "capture_rows_off_identity": bad_cap[:10],
            "n_capture_off_identity": len(bad_cap), "practice_rows_off_identity": bad_prac[:10], "certificate_binds_practice_journal": cert_ok}


def spend_probes(td: str) -> dict:
    """12.7: the A29 accounting probes of Astra ACC-RUNNER r2, re-run on every evidence run in a temporary tree (offline, no network)."""
    import acc_common as AC
    out = {}
    lock = os.path.join(td, "probe.lock")
    sd = os.path.join(td, "study")
    for sub in ("sim/a", "sim/b", "docs/final-test"):
        os.makedirs(os.path.join(sd, *sub.split("/")), exist_ok=True)
    import shutil as _sh
    _sh.copy(C.paths(None)["split"], os.path.join(sd, "docs", "final-test", "split_manifest.json"))
    for sub in ("a", "b"):                                         # two independent ledgers, each with one identical id=1 reservation
        lg = C.ScopedLedger(os.path.join(sd, "sim", sub, "ledger.db"), {"core": 5.0, "total": 5.0})
        with lg.use("core"):
            lg.reserve("anthropic", 0.2)
    out["identical_rows_in_two_ledgers_counted_twice"] = abs(AC.study_spend(sd)["open"] - 0.4) < 1e-9
    rows = [(1, "hold@core", "open", 1.0, None), (2, "anthropic@core", "settled", 0.6, 0.3), (3, "anthropic@core", "open", 0.4, None)]
    out["hold_and_its_reservations_count_once"] = AC.reconcile_rows(rows)["liability"] == 1.0
    out["reservations_above_their_hold_count_in_full"] = AC.reconcile_rows(rows + [(4, "anthropic@core", "open", 0.5, None)])["liability"] == 1.2
    out["two_open_holds_in_one_scope_not_reconciled"] = AC.reconcile_rows(rows + [(5, "hold@core", "open", 1.0, None)])["liability"] == 2.7
    orig = AC.sqlite3.connect

    class _Bad:
        def execute(self, *a, **k):
            raise AC.sqlite3.DatabaseError("file is not a database")

        def close(self):
            pass
    AC.sqlite3.connect = lambda *a, **k: _Bad()
    try:
        sp = AC.study_spend(sd)
    finally:
        AC.sqlite3.connect = orig
    out["sqlite_master_error_refuses"] = bool(sp["errors"]) and bool(AC.study_wide_problems(0.0, spend=sp))
    gp = os.path.join(td, "acceptance-v2", "ledger.db")
    os.makedirs(os.path.dirname(gp), exist_ok=True)
    AC.register_ledger_guard(gp, attempt_base_usd=0.0, attempt_cap_usd=8.0, other_spend=lambda: {"settled": 0.0, "open": 0.0, "committed": 0.0,
                                                                                                 "errors": [], "ledgers": {}}, lock_path=lock)
    w0 = AC.STUDY_LOCK_WAIT_S
    try:
        lg = C.ScopedLedger(gp, C.NAMESPACES["acceptance-v2"]["ceilings"])
        open(lock, "w").write("{}")                                  # another submission holds the boundary
        AC.STUDY_LOCK_WAIT_S = 0.2
        with lg.use("core"):
            try:
                lg.reserve("anthropic", 0.01)
                busy = False
            except C.CostCeilingExceeded:
                busy = True
        os.remove(lock)
        out["busy_boundary_refuses_and_sends_nothing"] = busy and lg.committed() == 0.0
        with lg._lock, lg._conn() as c:
            out["check_outside_the_boundary_refuses"] = bool(AC.guard_problems(lg, c, 0.01))
        with lg.use("core"):
            out["inside_the_boundary_accepted"] = lg.reserve("anthropic", 0.01) > 0 and not os.path.exists(lock)
        # ACC-RUNNER-D (Astra ACC-RUNNER r3 NEW defect 1): the lock itself -- no age-based takeover, a waiting contender never unlinks the
        # held lock (the r3 interleaving), and release removes only a lock that carries our token
        foreign = json.dumps({"pid": 999999, "host": "elsewhere", "ts": 0, "token": "f" * 32})
        open(lock, "w").write(foreign)
        old_t = time.time() - 10 * 24 * 3600
        os.utime(lock, (old_t, old_t))
        AC.STUDY_LOCK_WAIT_S = 0.2
        n0 = lg.committed()
        with lg.use("core"):
            try:
                lg.reserve("anthropic", 0.01)
                old_busy = False
            except C.CostCeilingExceeded:
                old_busy = True
        out["old_lock_not_taken_over"] = (old_busy and lg.committed() == n0 and os.path.exists(lock) and open(lock).read() == foreign
                                          and not hasattr(AC, "STUDY_LOCK_STALE_S"))
        removed, orig_rm = [], AC.os.remove
        AC.os.remove = lambda q, *a, **k: (removed.append(q), orig_rm(q, *a, **k))[1]
        seq = iter([0.0, 0.0, 0.1, 0.2, 99.0, 99.0, 99.0])
        try:
            AC._lock_acquire(lock, 1.0, now=lambda: next(seq), sleep=lambda _s: None)
            race_busy = False
        except AC.StudyBoundaryBusy:
            race_busy = True
        finally:
            AC.os.remove = orig_rm
        out["contender_never_unlinks"] = race_busy and removed == [] and open(lock).read() == foreign
        os.remove(lock)
        own = lock + ".own"
        tok = AC._lock_acquire(own, 0.2)
        kept = not AC._lock_release(own, "0" * 32) and os.path.exists(own)
        with lg.use("core"):
            with AC.study_boundary(lg):
                open(lock, "w").write(foreign)                         # our lock replaced while held: release must leave it
        out["release_only_own_token"] = (kept and AC._lock_release(own, tok) and not os.path.exists(own)
                                         and os.path.exists(lock) and open(lock).read() == foreign)
        os.remove(lock)
    finally:
        AC.STUDY_LOCK_WAIT_S = w0
        AC.unregister_ledger_guard(gp)
    out["ok"] = all(v is True for v in out.values())
    return out


def gates_ft2_dir(root=None):
    return os.path.join(root or ROOT, "docs", "final-test", "matrix", "gates_ft2")


def _rel2(root, p):
    return os.path.relpath(p, root).replace(os.sep, "/")


def bind_inputs(root, paths) -> dict:
    return {_rel2(root, p): (C.sha256_file(p) if os.path.isfile(p) else None) for p in paths}


def ft2_identities(root=None) -> dict:
    import treatment as T
    docs = {ns: T.identity(ns, root) for ns in ("test-v1", "acceptance-v2", FT2_DEV_NS, FT2_PRAC_NS)}
    fin, dev = docs["test-v1"], docs[FT2_DEV_NS]
    diff = sorted(k for k in fin if k != "sha" and fin.get(k) != dev.get(k))
    sub = sorted(k for k in (fin.get("briefing") or {}) if (fin.get("briefing") or {}).get(k) != (dev.get("briefing") or {}).get(k))
    return {"shas": {ns: d["sha"] for ns, d in docs.items()},
            "dev_vs_final": {"fields": diff, "briefing_fields": sub,
                             "note": "the dev rehearsal identity differs from the final one only where listed; the final-identity requirement "
                                     "of A31 item (4) is NOT waived by this (a disposition is Arash's)"}}


def gate2(gid, status, evidence, source, inputs, *, root, identity=None, namespaces=(), versions=None, final=None) -> dict:
    gen = os.path.abspath(__file__)
    doc = {"schema": GATE2_SCHEMA, "gate": gid, "status": status, "pass": status in ("PASS", "NOT_APPLICABLE"), "synthetic": False,
           "evidence": evidence, "source": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "binding": {"inputs": inputs, "identity": identity or {}, "namespaces": list(namespaces), "versions": versions or {},
                       "generator": {_rel2(root, gen): C.sha256_file(gen)}}}
    if final is not None:
        doc["final_identity"] = final
    return doc


def _jl(p):
    return jl(p) if p and os.path.isfile(p) else None


def final_identity_rows(root, kind, want_sha) -> dict:
    """{namespace: rows} of every ft-v2 namespace (never test-v1 before the freeze) whose `kind` journal rows carry the FINAL identity."""
    out = {}
    for ns in ("acceptance-v2",):
        rows = _jl(C.journal_path(ns, kind, root)) if ns in C.NAMESPACES else None
        mine = [r for r in rows or [] if r.get("treatment_id") == want_sha]
        if mine:
            out[ns] = mine
    return out


def _ledger_summary(p):
    if not os.path.isfile(p):
        return []
    c = sqlite3.connect("file:%s?mode=ro" % p.replace("\\", "/"), uri=True)
    try:
        rows = c.execute("SELECT provider,state,COUNT(*),COALESCE(SUM(reserved),0),COALESCE(SUM(actual),0) FROM ledger GROUP BY provider,state").fetchall()
    finally:
        c.close()
    return [{"provider": x, "state": y, "n": n, "reserved": round(r, 6), "actual": round(s, 6)} for x, y, n, r, s in rows]


def ft2_evidence(root=None, probes=None) -> dict:
    """-> {gate id: gate doc}. Offline except the read-only CLI probe in `probes` (codex --version), passed in by the caller."""
    root = root or ROOT
    import tempfile
    import acc_common as AC
    import practice_k as PK
    import x_isolation_probe as XP
    man = C.load_manifest(root)
    test = set(man["test"]) | set(man.get("test_uncapped_not_scored") or [])
    ids = ft2_identities(root)
    final_sha, dev_sha = ids["shas"]["test-v1"], ids["shas"][FT2_DEV_NS]
    cap_p = C.journal_path(FT2_DEV_NS, "capture", root)
    plan_p = C.journal_path(FT2_DEV_NS, "plan", root)
    prac_p = C.journal_path(FT2_PRAC_NS, "practice", root)
    cap, plans, prac = _jl(cap_p) or [], _jl(plan_p) or [], _jl(prac_p) or []
    run1 = [r for r in cap if int(r.get("capture_run", 1)) == 1]
    idb = {"dev": {FT2_DEV_NS: dev_sha}, "final": {"test-v1": final_sha}, "dev_vs_final": ids["dev_vs_final"]}
    stale_dev = sorted({str(r.get("treatment_id")) for r in cap + plans if r.get("treatment_id") != dev_sha})
    out = {}

    # ---- 12.2 bilingual retrieval: EN and FA of one dev item give the same package record ids (S* and the curve snapshots), over the
    #      REGISTERED dev set (Astra ACC-RUNNER r2: a wholly missing pair must fail)
    ev_rows = _jl(C.journal_path(FT2_DEV_NS, "events", root)) or []
    starts = [e for e in ev_rows if e.get("event") == "capture_start"]
    worlds = list((starts[-1] if starts else {}).get("worlds") or [])
    want = expected_bilingual_keys(man, FT2_DEV_NS, worlds)
    bc = bilingual_coverage(run1, want)
    ok = bc["ok"] and bool(worlds) and not stale_dev
    out["12.2"] = gate2("12.2", "PASS" if ok else "FAIL",
                        {"claim": "for every REGISTERED dev item and world the FA instruction yields the same package record ids as the EN one (S* and the curve snapshots K = 1 / 3 / 5)",
                         "registered_worlds": worlds, **bc, "stale_identity_rows": stale_dev},
                        f"{FT2_DEV_NS} capture (run 1) vs split_manifest dev_pilot + the registered relocation / repeat subsets",
                        bind_inputs(root, [cap_p, C.paths(root)["split"], C.journal_path(FT2_DEV_NS, "events", root)]), root=root, identity=idb,
                        namespaces=[FT2_DEV_NS])

    # ---- 12.3 honesty: capture sweeps, practice-v2 episodes, G-iso executions (a missing field is never a 0)
    hc = {"violations": 0, "oracle": 0, "map_build": 0, "missing": 0}
    for r in cap:
        h = (r.get("audit") or {}).get("honesty")
        if not isinstance(h, dict) or any(h.get(k) is None for k in ("violations", "oracle", "map_build")):
            hc["missing"] += 1
            continue
        for k in ("violations", "oracle", "map_build"):
            hc[k] += int(h[k])
    hp = {"episodes": 0, "violations": 0, "oracle": 0, "map_build": 0, "invalid": 0, "missing": 0}
    for r in prac:
        au = r.get("audit")
        if not isinstance(au, dict):
            hp["missing"] += 1
            continue
        hp["episodes"] += 1
        cnt = au.get("counts") or {}
        if au.get("violations") is None or cnt.get("oracle") is None or cnt.get("map-build") is None:
            hp["missing"] += 1
        hp["violations"] += int(au.get("violations") or 0)
        hp["oracle"] += int(cnt.get("oracle") or 0)
        hp["map_build"] += int(cnt.get("map-build") or 0)
        hp["invalid"] += int(bool(au.get("invalid")))
    gi_p = os.path.join(C.ns_docs(FT2_DEV_NS, root), "g_iso.json")
    gi = json.load(open(gi_p, encoding="utf-8")) if os.path.isfile(gi_p) else None
    gaud = ((gi or {}).get("components") or {}).get("audit") or {}
    ev_p = C.journal_path(FT2_DEV_NS, "events", root)
    cgs = [e for e in (_jl(ev_p) or []) if e.get("event") == "capture_gates"]
    cg = (cgs[-1].get("gates") or {}) if cgs else {}
    n_keys = len({r.get("key") for r in cap})
    cover_ok = bool(cgs) and (cg.get("all_cells") or {}).get("pass") is True and int((cg.get("all_cells") or {}).get("missing", 1)) == 0 \
        and n_keys == len(cap) == int((cg.get("g_iso") or {}).get("n") or -1)
    sens = (gi or {}).get("audit_sensitivity") or {}
    cert_p = os.path.join(root, "sim", "final_test", "diag", "a31", "equiv_replay.json")
    cert = json.load(open(cert_p, encoding="utf-8")) if os.path.isfile(cert_p) else None
    hid = honesty_identity(cap, prac, dev_sha=dev_sha, dev_ns=FT2_DEV_NS, prac_run_sha_prefix=PRACTICE_RUN_IDENTITY, cert=cert,
                           prac_journal_sha=C.sha256_file(prac_p) if os.path.isfile(prac_p) else None)
    gi_bound = bool(gi) and gi.get("namespace") == FT2_DEV_NS and gi.get("sha256") == C.self_sha(gi)
    ok = (hid["ok"] and gi_bound and len(cap) > 0 and cover_ok and hc["missing"] == 0 and hc["violations"] + hc["oracle"] + hc["map_build"] == 0 and hp["episodes"] == 10 and hp["missing"] == 0
          and hp["violations"] + hp["oracle"] + hp["map_build"] + hp["invalid"] == 0 and gi is not None and gaud.get("absent_vs_present") == 0
          and (sens.get("clean_key") or {}) == {"map-build": 0, "oracle": 0, "violations": 0})
    out["12.3"] = gate2("12.3", "PASS" if ok else "FAIL",
                        {"claim": "0 honesty violations, 0 oracle reads, 0 map-build reads; a missing audit field is never a 0",
                         "dev_capture_rows": {"rows": len(cap), "distinct_cells": n_keys, "coverage_ok": cover_ok, **hc}, "practice_v2_episodes": hp,
                         "g_iso_executions": {"executions": (gi or {}).get("executions"), "audit_component": gaud, "audit_sensitivity_clean_key": sens.get("clean_key"),
                                              "bound": gi_bound},
                         "row_identity": hid},
                        f"{FT2_DEV_NS} capture audit + {FT2_PRAC_NS} practice audit + {FT2_DEV_NS} G-iso executions, each row bound to its identity",
                        bind_inputs(root, [cap_p, prac_p, gi_p, ev_p, cert_p]), root=root, identity=dict(idb, practice={FT2_PRAC_NS: ids["shas"][FT2_PRAC_NS]}),
                        namespaces=[FT2_DEV_NS, FT2_PRAC_NS])

    # ---- 12.5 evaluator (A26 item 9 wording): FG_NS P0 + E0 rows under the FINAL identity, against the registered expected changes
    import score as S
    fg_p = {k: C.journal_path(FG_NS, k, root) for k in ("capture", "plan", "exec", "events", "score")}
    reg_p = os.path.join(root, *A26_125_REG)
    reg = json.load(open(reg_p, encoding="utf-8")) if os.path.isfile(reg_p) else None
    base_p = os.path.join(root, S.STEP4_BASELINE)
    base = {r["id"]: bool(r.get("goal_ok")) for r in (_jl(base_p) or []) if r.get("status") in ("ok", "FAIL")}
    if reg and (reg.get("baseline") or {}).get("sha256") not in (None, C.sha256_file(base_p) if os.path.isfile(base_p) else None):
        base = {}                                                          # the step-4 baseline changed since the registration: nothing follows
    fg_exec = _jl(fg_p["exec"]) or []
    ev125 = eval_125_verdict(_jl(fg_p["score"]), fg_exec, _jl(fg_p["events"]), reg, C.sha256_file(reg_p) if reg else None, base,
                             final_sha=final_sha, min_agreement=S.EVAL_MIN_AGREEMENT)
    plain = S.evaluator_check(FG_NS, root=root, write=False) if os.path.isfile(fg_p["score"]) else None
    out["12.5"] = gate2("12.5", ev125["status"],
                        {"claim": "A26 12.5: reference plans (P0) under the frozen ft-v2 executor and the FINAL identity reproduce their step-4 outcomes with the "
                                  "registered expected slicing changes applied, within the previously specified variance allowance; unrealistic_slice 0; the empty "
                                  "plan (E0, same frozen execute.py path) scores 0 on every registered presentation",
                         "verdict": ev125, "required_identity": FT2_FINAL_IDENTITY, "registration": _rel2(root, reg_p),
                         "evaluator_check_unadjusted": {k: (plain or {}).get(k) for k in ("compared", "agreement", "empty_plan_rows", "empty_plan_successes", "reasons")}},
                        f"{FG_NS} score.jsonl (P0, E0) + exec identity + a26_12_5_registration.json + its event + sim/step4/refplans_honest.jsonl",
                        bind_inputs(root, [fg_p["score"], fg_p["exec"], fg_p["events"], reg_p, base_p]), root=root,
                        identity=dict(idb, final_required=FT2_FINAL_IDENTITY), namespaces=[FG_NS],
                        versions={k: C.sha256_file(os.path.join(HERE, k)) for k in ("score.py", "execute.py", "fg_runner.py")})

    # ---- 12.7 spend guard: ft-v2 ledgers reconcile, no open reservation, the guard + the A29 study-wide check behave
    lps = {ns: os.path.join(C.ns_work(ns, root), "ledger.db") for ns in (FT2_DEV_NS, FT2_PRAC_NS, "acceptance-v2")}
    led = {ns: _ledger_summary(p) for ns, p in lps.items()}
    open_ft2 = {ns: [x for x in rows if x["state"] == "open" and not x["provider"].startswith("hold@")] for ns, rows in led.items()}
    billed = sum(float(x.get("cost_usd") or 0) for x in plans if not x.get("cached"))
    settled = sum(x["actual"] for x in led[FT2_DEV_NS] if x["state"] == "settled" and x["provider"].startswith("anthropic"))
    ref, thr, tot = C.OPUS_REF_USD_PER_CALL, C.OPUS_GUARD_USD_PER_CALL, float(C.NAMESPACES["test-v1"]["ceilings"]["total"])
    g1 = C.cost_guard(opus_calls=60, opus_usd=60 * ref, checked_at=0, spent=5.0, remaining={"opus": 700}, mean_cost={"opus": ref}, ceiling=tot)
    g2 = C.cost_guard(opus_calls=60, opus_usd=60 * 3 * ref, checked_at=0, spent=5.0, remaining={"opus": 700}, mean_cost={"opus": ref}, ceiling=tot)
    g3 = C.cost_guard(opus_calls=60, opus_usd=60 * ref, checked_at=0, spent=5.0, remaining={"opus": 700}, mean_cost={"opus": 0.2}, ceiling=tot)
    # A24.1-C (RUNNER-FINAL): the EXPLICIT threshold (A37: 0.080 USD per billed Opus cell, not 2 x the 0.044 reference), each window on its own;
    # A37: the recent-window probe is written relative to the threshold (0.75 / 1.25 x thr = the former 0.030 / 0.050 at 0.040)
    gb = {"at_threshold": C.cost_guard(opus_calls=60, opus_usd=60 * thr, checked_at=0, spent=0.0, remaining={}, mean_cost={}, ceiling=tot,
                                       opus_recent=[thr] * 60)["pause"],
          "just_above_cumulative_only": C.cost_guard(opus_calls=60, opus_usd=60 * (thr + 0.0005), checked_at=0, spent=0.0, remaining={},
                                                     mean_cost={}, ceiling=tot)["pause"],
          "recent_window_only": C.cost_guard(opus_calls=600, opus_usd=540 * 0.75 * thr + 60 * 1.25 * thr, checked_at=540, spent=0.0, remaining={},
                                             mean_cost={}, ceiling=tot, opus_recent=[0.75 * thr] * 540 + [1.25 * thr] * 60)["pause"],
          "below_minimum_sample": C.cost_guard(opus_calls=59, opus_usd=59 * 0.10, checked_at=0, spent=0.0, remaining={}, mean_cost={},
                                               ceiling=tot)["pause"],
          "old_a19_level_0_025": C.cost_guard(opus_calls=60, opus_usd=60 * 0.025, checked_at=0, spent=0.0, remaining={}, mean_cost={},
                                              ceiling=tot)["pause"]}
    gb_ok = (gb == {"at_threshold": False, "just_above_cumulative_only": True, "recent_window_only": True, "below_minimum_sample": False,
                    "old_a19_level_0_025": False} and C.COST_GUARD_SPEC["version"] == "ftm-guard-4" and abs(thr - 2 * ref) > 1e-9)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "acceptance-v2", "ledger.db")
        os.makedirs(os.path.dirname(p))
        ug = C.ScopedLedger(p, C.NAMESPACES["acceptance-v2"]["ceilings"])
        with ug.use("core"):
            try:
                ug.reserve("anthropic", 0.01)
                fail_closed = False
            except Exception:                                        # noqa: BLE001
                fail_closed = True
        AC.register_ledger_guard(p, attempt_base_usd=0.0, attempt_cap_usd=C.NAMESPACES["acceptance-v2"]["attempt_ceiling"], root=root,
                                 lock_path=os.path.join(td, "fo.lock"))
        try:
            fo = AC.fake_overcap_test(C.ScopedLedger(p, C.NAMESPACES["acceptance-v2"]["ceilings"]))
        finally:
            AC.unregister_ledger_guard(p)
        probes_sp = spend_probes(td)
    sw = AC.study_spend(root)
    ok = (not any(open_ft2.values()) and abs(settled - billed) < 1e-4 and not g1["pause"] and g2["pause"] and g3["pause"] and fail_closed
          and fo.get("ok") is True and AC.study_wide_problems(0.0, spend=sw) == [] and probes_sp["ok"] and gb_ok)
    out["12.7"] = gate2("12.7", "PASS" if ok else "FAIL",
                        {"claim": "the ft-v2 ledgers reconcile with the per-call costs and hold no open reservation; the A15 / A21 B guard pauses on a regime change "
                                  "only; the A29 study-wide pre-submission check and the acceptance attempt guard refuse an over-cap submission (fake over-cap test); "
                                  "an unguarded acceptance ledger refuses paid calls",
                         "ledgers": led, "open_reservations_ft2": {k: len(v) for k, v in open_ft2.items()},
                         "rehearsal_v3_settled_anthropic_usd": round(settled, 6), "rehearsal_v3_sum_of_non_cached_call_costs_usd": round(billed, 6),
                         "guard_dry_run": {"at_ref": g1["pause"], "at_3x_ref": g2["pause"], "projection_over_ceiling": g3["pause"]},
                         "guard_boundaries_a24_1_c": dict(gb, threshold_usd_per_billed_opus_cell=thr, reference=ref, spec=C.COST_GUARD_SPEC["version"],
                                                          ok=gb_ok),
                         "fake_overcap": fo, "unguarded_acceptance_ledger_refuses": fail_closed, "a29_accounting_probes": probes_sp,
                         "study_wide": {"settled": sw["settled"], "open": sw["open"], "holds": sw.get("holds"), "committed": sw["committed"], "cap": C.ANTHROPIC_CAP_USD,
                                        "ledgers": sorted(sw["ledgers"])}},
                        "ft-v2 namespace ledgers + common.cost_guard dry run + acc_common study-wide check / fake over-cap (temporary ledger)",
                        dict(bind_inputs(root, [plan_p] + list(lps.values())), study_ledgers=C.sha256_json(sw["ledgers"])), root=root,
                        identity=idb, namespaces=[FT2_DEV_NS, FT2_PRAC_NS, "acceptance-v2"],
                        versions={"cost_guard": C.COST_GUARD_SPEC["version"], "acc_common.py": C.sha256_file(AC.__file__)})

    # ---- 12.10 practice, capture and workflow-B data are final-test non-test only
    pat = re.compile(r"g-l[123]-[a-z0-9\-]+")
    s5, hits, nfiles = set(), {}, 0
    files = sorted(glob.glob(os.path.join(root, "sim", "step5", "live", "**", "*"), recursive=True)
                   + glob.glob(os.path.join(root, "sim", "step5", "_b1_gates", "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f) and f.endswith((".json", ".jsonl"))]
    for f in files:
        nfiles += 1
        found = set(pat.findall(open(f, encoding="utf-8", errors="replace").read()))
        s5 |= found
        if found & test:
            hits[_rel2(root, f)] = sorted(found & test)[:5]
    items = {i["id"]: i for i in json.load(open(C.paths(root)["dataset"], encoding="utf-8"))["items"]}
    res = PK.reserved_acceptance(None, items)
    pool = set(man["practice_pool"]) - set(res["ids"])
    p_items, c_items = {r["item"] for r in prac}, {r["item"] for r in cap}
    ok = (nfiles > 0 and not hits and not (p_items & test) and not (c_items & test) and bool(p_items) and p_items <= pool
          and c_items <= set(man["dev_pilot"]) and not ((p_items | c_items) & set(res["ids"])))
    out["12.10"] = gate2("12.10", "PASS" if ok else "FAIL",
                         {"claim": "practice-v2 attempts lie in the effective practice pool (reserved acceptance items removed), the rehearsal-v3 capture in dev_pilot, "
                                   "and the workflow-B journals hold no test id",
                          "step5_files_scanned": nfiles, "step5_ids_in_test": len(s5 & test), "files_with_test_ids": hits,
                          "practice_items": len(p_items), "practice_outside_pool": sorted(p_items - pool), "capture_items": len(c_items),
                          "capture_outside_dev": sorted(c_items - set(man["dev_pilot"])), "reserved_sha": res["sha"]},
                         "practice-v2 journal + rehearsal-v3 capture + regex scan of sim/step5/live and _b1_gates",
                         dict(bind_inputs(root, [prac_p, cap_p, C.paths(root)["dataset"], C.paths(root)["split"]]),
                              step5_files=C.sha256_json([[_rel2(root, f), C.sha256_file(f)] for f in files])),
                         root=root, identity=idb, namespaces=[FT2_PRAC_NS, FT2_DEV_NS])

    # ---- G-iso (final identity required): the dev A20.1 pass + the final-identity part
    okg = bool(gi) and (gi.get("rule") == "A20.1" and gi.get("status") == "pass" and (gi.get("executions") or {}).get("all_real") is True
                        and (gi.get("audit_sensitivity") or {}).get("ok") is True and not gi.get("instrumentation_missing")
                        and int(gi.get("n") or 0) >= int(gi.get("n_registered") or 12) >= 12 and not gi.get("diff") and not gi.get("blocked")
                        and not gi.get("short") and gi.get("sha256") == C.self_sha(gi) and gi.get("namespace") == FT2_DEV_NS)
    fin_rows = final_identity_rows(root, "capture", final_sha)
    fgi_p = os.path.join(C.ns_docs(FG_NS, root), "g_iso.json")
    fgi = json.load(open(fgi_p, encoding="utf-8")) if os.path.isfile(fgi_p) else None
    fg_cap = _jl(C.journal_path(FG_NS, "capture", root)) or []
    fst, fwhy = g_iso_final_verdict(fgi, final_sha=final_sha, cap_rows=fg_cap, plan_rows=_jl(C.journal_path(FG_NS, "plan", root)) or [])
    fin = {"required_identity": FT2_FINAL_IDENTITY, "namespace": FG_NS, "namespaces_with_final_identity_captures": sorted(fin_rows), "status": fst,
           "reasons": fwhy, "report": {k: (fgi or {}).get(k) for k in ("status", "n", "n_registered", "diff", "control_noise", "blocked", "executions",
                                                                         "treatment_id", "sha256")},
           "owed": None if fst == "PASS" else "A31 DONE owed item (4): G-iso regenerated under the FINAL identity (FREEZE-GATES-FT2: run/chain_gates.sh)"}
    st = "FAIL" if (not okg and fst != "PASS") else fst                 # the final-identity report decides; a failing dev report never hides behind PENDING
    out["G-iso"] = gate2("G-iso", st,
                         {"claim": "A20.1 rule: request bytes, status, goal, executed steps and audit identical under absent / present / absent memory files",
                          "dev": {"status": (gi or {}).get("status"), "n": (gi or {}).get("n"), "diff": (gi or {}).get("diff"),
                                  "control_noise": (gi or {}).get("control_noise"), "executions": (gi or {}).get("executions"), "dev_pass": okg}},
                         f"{FG_NS} g_iso.json under the final identity (+ the {FT2_DEV_NS} dev report)",
                         bind_inputs(root, [fgi_p, C.journal_path(FG_NS, "capture", root), C.journal_path(FG_NS, "plan", root), gi_p, cap_p]),
                         root=root, identity=dict(idb, final_required=FT2_FINAL_IDENTITY), namespaces=[FG_NS, FT2_DEV_NS], final=fin,
                         versions={k: C.sha256_file(os.path.join(HERE, k)) for k in ("execute.py", "fg_runner.py")})

    # ---- tokens + context (final identity required): A24 7 (g) / (h) / (i) on the dev captures
    worst, over, miss, n = {}, [], [], 0
    for r in cap:
        for s, b in (r.get("blocks") or {}).items():
            if s == "none" or not isinstance(b, dict) or not b.get("text"):
                continue
            n += 1
            cts = b.get("counts") or {}
            if sorted(cts) != ["anthropic", "gemma"]:
                miss.append(f"{r['pres']}#{s}")
                continue
            m = max(int(v) for v in cts.values())
            worst[s] = max(worst.get(s, 0), m)
            if m > 2000:
                over.append(f"{r['pres']}#{s}: {m}")
    A = acc_final_inputs(root)
    acc_in = [A["verdict_path"]]
    tg_p = os.path.join(root, "sim", "final_test", "diag", "freeze_gates", "token_gate.json")
    tg = json.load(open(tg_p, encoding="utf-8")) if os.path.isfile(tg_p) else {}
    okt = n > 0 and not over and not miss and not stale_dev and tg.get("ok") is True and float((tg.get("max") or {}).get("anthropic", 9e9)) <= 2000
    ftk = acc_final_part("tokens", A, final_sha)
    out["tokens"] = gate2("tokens", ftk["status"] if okt else "FAIL",
                          {"claim": "A24 7 (g) / (h): every package and FS block <= 2,000 tokens INCLUDING its header, max over the Gemma and Anthropic tokenizers "
                                    "(replaces the ft-v1 800-token per-level budget)",
                           "dev": {"blocks": n, "worst_by_source": worst, "over_2000": over, "missing_tokenizer": miss, "dev_pass": okt,
                                   "token_gate_json": {"ok": tg.get("ok"), "max": tg.get("max"), "status": tg.get("status")}}},
                          f"{FT2_DEV_NS} capture block counts (header + block, real counter) + diag/freeze_gates/token_gate.json; FINAL: {ACC_FINAL_NS} "
                          "verdict criteria g / h (authenticated, bundle replay) + g_eligible.json + capture.jsonl",
                          bind_inputs(root, [cap_p, tg_p] + acc_in + [A["paths"][f] for f in ACC_FINAL_FILES["tokens"]]), root=root,
                          identity=dict(idb, final_required=FT2_FINAL_IDENTITY), namespaces=[FT2_DEV_NS, ACC_FINAL_NS],
                          versions={"tokenizers": tg.get("tokenizers")}, final=ftk)
    cf_p = os.path.join(C.ns_docs(FT2_DEV_NS, root), "context_fit.json")
    cf = json.load(open(cf_p, encoding="utf-8")) if os.path.isfile(cf_p) else {}
    need = {(r["pres"], s) for r in cap for s in (r.get("requests") or {}) if s in ("none", "S*", "FS")}
    got = {(c.get("pres"), c.get("source")) for c in cf.get("checks") or [] if c.get("model") == "gemma" and c.get("mode") == "measured" and c.get("fits") is True}
    okc = (cf.get("gemma_verdict") == "PASS_MEASURED" and cf.get("freeze_grade") is True and cf.get("namespace") == FT2_DEV_NS and bool(need)
           and need <= got and (cf.get("rule") or {}).get("feedback_contract") == "fb-1")
    fcx = acc_final_part("context", A, final_sha)
    out["context"] = gate2("context", fcx["status"] if okc else "FAIL",
                           {"claim": "A24 7 (i): measured full-context fit for every Gemma presentation incl. 2 repair turns at the fb-1 bound + the 768-token output",
                            "dev": {"gemma_verdict": cf.get("gemma_verdict"), "freeze_grade": cf.get("freeze_grade"), "pairs_needed": len(need),
                                    "pairs_measured_fitting": len(need & got), "missing": sorted("%s#%s" % x for x in need - got)[:10], "dev_pass": okc}},
                           f"{FT2_DEV_NS} context_fit.json (context_fit.py --real); FINAL: {ACC_FINAL_NS} verdict criterion i (authenticated, bundle "
                           "replay) + its context_fit.json + capture.jsonl",
                           bind_inputs(root, [cf_p, cap_p] + acc_in + [A["paths"][f] for f in ACC_FINAL_FILES["context"]]), root=root,
                           identity=dict(idb, final_required=FT2_FINAL_IDENTITY), namespaces=[FT2_DEV_NS, ACC_FINAL_NS],
                           versions={"feedback_contract": (cf.get("rule") or {}).get("feedback_contract")}, final=fcx)

    # ---- leakage: the C11 audit (bound to the final identity, dataset, split, snapshots) + the practice snapshot gates
    c11_p = os.path.join(root, "sim", "final_test", "diag", "c11", "c11_audit.json")
    c11 = json.load(open(c11_p, encoding="utf-8")) if os.path.isfile(c11_p) else {}
    b = c11.get("binding") or {}
    snaps = sorted(glob.glob(os.path.join(root, "sim", "final_test", "snapshots_ft2", "o0", "k*", "practice_manifest.json")))
    lk = {os.path.basename(os.path.dirname(p)): (json.load(open(p, encoding="utf-8")).get("leakage") or {}) for p in snaps}
    want_bind = {"dataset": C.sha256_file(C.paths(root)["dataset"]), "treatment_test_v1": final_sha, "treatment_practice_v2": ids["shas"][FT2_PRAC_NS],
                 "inclusion_manifest_a26": C.sha256_file(os.path.join(root, "docs", "final-test", "inclusion_manifest_a26.json")),
                 "capture": C.sha256_file(cap_p) if os.path.isfile(cap_p) else None, "plan": C.sha256_file(plan_p) if os.path.isfile(plan_p) else None}
    bmis = sorted(k for k, v in want_bind.items() if b.get(k) != v)
    snap_bind = {k: C.sha256_file(os.path.join(root, "sim", "final_test", "snapshots_ft2", "o0", k, "manifest.json")) for k in lk}
    smis = sorted(k for k, v in snap_bind.items() if (b.get("snapshots") or {}).get(k) != v)
    ctl = c11.get("controls") or {}
    ok = (c11.get("ok") is True and not c11.get("problems") and int(ctl.get("n") or 0) > 0 and ctl.get("passed") == ctl.get("n")
          and not bmis and not smis and len(lk) == 4 and all(x.get("ok") is True for x in lk.values()))
    c11_script = os.path.join(root, "sim", "final_test", "diag", "c11", "c11_audit.py")
    prereg_p = os.path.join(root, "docs", "final-test", "preregistration.md")
    cwhy = c11_final_problems(c11, final_sha=final_sha, want_bind=want_bind, script_sha=C.sha256_file(c11_script) if os.path.isfile(c11_script) else None,
                              prereg_sha=C.sha256_file(prereg_p) if os.path.isfile(prereg_p) else None)
    flk = acc_final_part("leakage", A, final_sha)
    if cwhy:                                                             # the audit half: missing = PENDING, anything else = FAIL
        miss_audit = not c11
        flk = dict(flk, status="FAIL" if (not miss_audit or flk["status"] == "FAIL") else "PENDING", reasons=cwhy + flk["reasons"],
                   owed="c11_audit.py re-run clean under the FINAL identity: " + cwhy[0])
    flk["c11_audit"] = {"status": "clean" if not cwhy else "not clean", "reasons": cwhy, "binding_treatment_test_v1": b.get("treatment_test_v1"),
                        "scope": "c11_audit.py scans the rehearsal-v3 dev capture / plan bytes (its registered scope) with the forbidden set and identity "
                                 "of the final freeze; the acceptance-v2-a2 bytes are covered by criterion c11 (c11_acceptance.json)"}
    out["leakage"] = gate2("leakage", flk["status"] if ok else "FAIL",
                           {"claim": "A31 leakage on every delivered byte of the dev captures (requests + repair turns, final forbidden set) and the stores; "
                                     "C11 quarantine R1-R4 + the planted controls; the 4 practice-v2 snapshots passed their A30 / A31 leakage gate",
                            "c11_ok": c11.get("ok"), "c11_problems": c11.get("problems"), "controls": {k: ctl.get(k) for k in ("n", "passed")},
                            "binding_mismatch": bmis, "snapshot_binding_mismatch": smis, "snapshot_leakage_ok": {k: x.get("ok") for k, x in lk.items()},
                            "dev": {"dev_pass": ok},
                            "note": "the final-identity part (final_identity) = the c11_audit.py re-run bound to the final identity + acceptance-v2-a2 criterion c11"},
                           "sim/final_test/diag/c11/c11_audit.json + snapshots_ft2/o0/k*/practice_manifest.json; FINAL: the same audit bound to the final "
                           "identity + acceptance-v2-a2 criterion c11 (authenticated verdict) + c11_acceptance.json",
                           bind_inputs(root, [c11_p] + snaps + [prereg_p] + acc_in + [A["paths"][f] for f in ACC_FINAL_FILES["leakage"]]), root=root,
                           identity=dict(idb, final_required=FT2_FINAL_IDENTITY), namespaces=[FT2_DEV_NS, FT2_PRAC_NS, ACC_FINAL_NS],
                           versions={"c11_audit.py": C.sha256_file(c11_script)}, final=flk)

    # ---- X-iso: the gpt-6-sol isolation probe, valid for the CURRENT CLI version and flag set only
    xp = os.path.join(root, "docs", "final-test", "matrix", "x_isolation.json")
    xrep = json.load(open(xp, encoding="utf-8")) if os.path.isfile(xp) else None
    fl = XP.flags_sha("gpt-6-sol")
    xok, xwhy = x_iso_verdict(xrep, fl)
    cli_now = ((probes or {}).get("codex") or {}).get("version")
    if not cli_now:
        xwhy = xwhy + ["the codex CLI version was not probed now (codex --version)"]
    elif (xrep or {}).get("cli_version") != cli_now:
        xwhy = xwhy + [f"the probe ran on {(xrep or {}).get('cli_version')!r}, the CLI now is {cli_now!r}: probe again"]
    out["X-iso"] = gate2("X-iso", "PASS" if (xok and not xwhy) else "FAIL",
                         {"claim": "the gpt-6-sol X arms run in a clean ephemeral read-only session with the hardened tool surface, for THIS CLI version and flag set",
                          "reasons": xwhy, "cli_version_report": (xrep or {}).get("cli_version"), "cli_version_now": cli_now, "flags_sha": (xrep or {}).get("flags_sha"),
                          "n_calls": (xrep or {}).get("n_calls"), "canary_leaks": (xrep or {}).get("canary_leaks")},
                         "x_isolation.json + codex --version now + the current provider flag set", bind_inputs(root, [xp]), root=root,
                         versions={"codex_cli": cli_now, "flags_sha": fl})
    for gid, why in ASTRA_R2_HELD.items():
        g = out.get(gid)
        if g and g["status"] == "PASS":
            g["status"], g["pass"] = "PENDING", False
            g["evidence"]["dev"] = dict(g["evidence"].get("dev") or {}, dev_pass=True)
            g["evidence"]["owed"] = "Astra ACC-RUNNER r2 finding, not closed (repair cap reached): " + why
    return out


def main_ft2(write: bool, probes=None, root=None) -> int:
    root = root or ROOT
    out = ft2_evidence(root, probes)
    for gid, g in out.items():
        print(f"{g['status']:<15} {gid}  {g['evidence'].get('claim', '')[:150]}")
    if write:
        d = gates_ft2_dir(root)
        os.makedirs(d, exist_ok=True)
        for gid, g in out.items():
            raw = json.dumps(g, indent=1, ensure_ascii=False)
            if C.key_leak(raw):
                raise SystemExit("key-like text in evidence for " + gid)
            with open(os.path.join(d, gid + ".json"), "w", encoding="utf-8", newline="\n") as f:
                f.write(raw + "\n")
        print("wrote", len(out), "ft-v2 evidence files to", _rel2(root, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
