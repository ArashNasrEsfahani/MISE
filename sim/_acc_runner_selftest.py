"""Selftest for sim/final_test/acceptance_v2/acc_runner.py + the acceptance-v2 plumbing in sim/final_test/matrix/common.py (stage ACC-RUNNER,
2026-10-14). Offline (Windows python), no simulator, no LLM, no network; temporary roots for every write. The real registration, dataset and
rehearsal-v3 journals are read-only inputs (the C11 scanner is checked against the C11-AND-REG audit of rehearsal-v3).

    python sim/_acc_runner_selftest.py
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
sys.path.insert(0, os.path.join(HERE, "final_test", "acceptance_v2"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import common as C                                       # noqa: E402
import acc_common as AC                                  # noqa: E402
import acc_runner as A                                   # noqa: E402

N = {"ok": 0, "fail": 0}


def check(name, cond, detail=""):
    N["ok" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  :: {str(detail)[:300]}"))


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


REG = A.registration()


def ns_snapshot():
    """{relative path: sha256} of every file in the REAL acceptance-v2 docs + work dirs (ACC-RUNNER-D: they now legitimately hold the owed
    fake over-cap test; the selftest must leave them byte-identical, not empty)."""
    out = {}
    for base in (C.ns_docs(C.ACCEPTANCE_NS), C.ns_work(C.ACCEPTANCE_NS)):
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                q = os.path.join(dp, fn)
                out[os.path.relpath(q, ROOT)] = A.file_sha(q)
    return out


NS0 = ns_snapshot()
ROSTER = AC.acceptance_roster(None, REG)
MAINS = [i["id"] for i in REG["items"] if i["role"] == "main"]
SPARES = [i["id"] for i in REG["items"] if i["role"].startswith("spare")]
L3M = [i["id"] for i in REG["items"] if i["role"] == "main" and i["level"] == "L3"]

# ------------------------------------------------------------------------------------------------ 1 registration + namespace
print("== 1 registration and namespace")
check("1.1 registration file is the C11-AND-REG seal (74b5cf56) or a valid re-seal of it",
      A.file_sha(AC.acceptance_registration_path()) == AC.ACC_REG_SEALED_FILE_SHA or REG.get("roster") is not None)
check("1.2 registration self-sha and frozen sections verify", AC.acceptance_registration_problems(REG) == [])
t = copy.deepcopy(REG)
t["decision_rule"]["f"] = ">= 40 / 48"
t["sha256"] = C.self_sha(t)
check("1.3 an edited frozen section (decision rule) is refused even with a fresh self-sha",
      any("frozen registration section" in x for x in AC.acceptance_registration_problems(t)))
t = copy.deepcopy(REG)
t["namespace"] = "test-v1"
check("1.4 an edited self-sha-less file is refused", any("self-sha" in x for x in AC.acceptance_registration_problems(t)))
ns = C.NAMESPACES.get(C.ACCEPTANCE_NS) or {}
check("1.5 namespace acceptance-v2: ceiling 16 (no A15 core reserve), attempt 8, ft-v2, study guard, no archived arms",
      ns.get("ceilings") == {"total": 16.0} and ns.get("attempt_ceiling") == 8.0 and ns.get("treatment") == "ft-v2"
      and ns.get("study_guard") is True and tuple(ns.get("archived_arms")) == (), ns)
check("1.6 registration spend section = the namespace ceilings", REG["spend"]["ceiling_usd_per_attempt"] == ns["attempt_ceiling"]
      and REG["spend"]["ceiling_usd_namespace"] == ns["ceilings"]["total"])
import treatment as T                                    # noqa: E402
check("1.7 treatment_of(acceptance-v2) = ft-v2", T.treatment_of(C.ACCEPTANCE_NS) == "ft-v2")
ia, it_ = T.identity(C.ACCEPTANCE_NS)["sha"], T.identity("test-v1")["sha"]
check("1.8 treatment identity of acceptance-v2 = test-v1 = the registered 7a2ee211", ia == it_ == REG["pins"]["treatment"] == A.TREATMENT_SHA, (ia, it_))
check("1.9 the namespace is not the draft-briefing / strict-mask namespace", C.ACCEPTANCE_NS not in T.DEV_DRAFT_NAMESPACES + T.STRICT_NAMESPACES)
_pp = A.pin_problems(REG)
_crec = {}
try:                                                       # A34-A2 + A35 (signed 2026-10-15): after attempt 1's verdict, the ONLY frozen refs that
    _crec = json.load(open(os.path.join(ROOT, "docs", "final-test", "acceptance_correction.json"), encoding="utf-8"))   # may move are the ones
except (OSError, ValueError):                              # the correction record re-freezes, at exactly their recorded new shas
    _crec = {}
_moved = ((_crec.get("refrozen") or {}).get("frozen_refs") or {}) if A._a2().attempt_closed(1) else {}
if A._a2().attempt_closed(2):                              # A36 (signed) + A36-IMPL (approved): after attempt 2's terminal verdict, refs its reviewed
    sys.path.insert(0, os.path.join(ROOT, "sim", "final_test", "matrix"))          # implementation moved count as re-frozen ONLY through
    import freeze_ft2 as _FT                                                       # freeze_ft2's validated exception (A36 + A36-IMPL +
    _moved = dict(_moved, **_FT._refrozen_by_correction(ROOT))                     # the full changed-set check), at exactly their shas
_pp_left = [x for x in _pp if not any(x == f"frozen ref {r} changed since the registration (the registration is stale)"
                                      and A.file_sha(os.path.join(ROOT, *r.split("/"))) == s for r, s in _moved.items())]
check("1.10 pins + frozen_refs match the disk (or moved exactly as the correction record re-freezes them)", _pp_left == [], _pp_left)

# ------------------------------------------------------------------------------------------------ 2 roster resolver + guards
print("== 2 roster and guards")
check("2.1 roster = the 24 registered main items while unsealed, else the sealed roster section", ROSTER == (REG["roster"]["items"] if REG.get("roster") else MAINS))
lv = {}
for i in REG["items"]:
    if i["id"] in ROSTER:
        lv[i["level"]] = lv.get(i["level"], 0) + 1
check("2.2 roster levels 8 / 8 / 8", lv == {"L1": 8, "L2": 8, "L3": 8}, lv)
check("2.2b without an armed acceptance session the guard refuses acceptance-v2 (direct plan.py / execute.py)",
      raises(lambda: C.guard_items(ROSTER, C.ACCEPTANCE_NS), C.TestItemRefused))
def guard_or_none(ids, ns, **kw):
    try:
        return C.guard_items(ids, ns, **kw)
    except Exception as e:                                  # noqa: BLE001 -- reported as a FAIL with its reason, never a crash
        return {"n": None, "n_test": None, "error": f"{type(e).__name__}: {e}"}


AC.arm_session([], by="selftest")                        # the real arming path (preflight problems = [] here), not a dict update
g = guard_or_none(ROSTER, C.ACCEPTANCE_NS)
check("2.3 with the session armed the guard admits exactly the roster in acceptance-v2", g["n"] == 24 and g["n_test"] == 0, g)
check("2.3b arming refuses when the preflight does not hold", raises(lambda: AC.arm_session(["x"], by="selftest"), RuntimeError) and AC.SESSION["armed"])
check("2.4 guard refuses a non-roster item in acceptance-v2", raises(lambda: C.guard_items(["g-l1-pickup-apple"], C.ACCEPTANCE_NS), C.TestItemRefused))
check("2.5 guard refuses a spare before it is sealed into the roster", REG.get("roster") is not None or
      raises(lambda: C.guard_items([SPARES[0]], C.ACCEPTANCE_NS), C.TestItemRefused))
new10 = [i["id"] for i in REG["items"] if i["level"] == "L3"]
bad = [(n_, i) for n_ in C.NAMESPACES if n_ != C.ACCEPTANCE_NS for i in new10 if not raises(lambda: C.guard_items([i], n_), C.TestItemRefused)]
check("2.6 the 10 new L3 items are refused in every other namespace", not bad, bad[:3])
bad = [(n_, i) for n_ in ("test-v1", "rehearsal-v1", "rehearsal-v2", "rehearsal-v3") for i in ROSTER if not raises(lambda: C.guard_items([i], n_), C.TestItemRefused)]
check("2.7 every roster item is refused in test-v1 and the rehearsal namespaces", not bad, bad[:3])
m = C.load_manifest()
check("2.8 load_manifest never carries the roster key (no other consumer sees a new id list)", C.ACCEPTANCE_ROSTER not in m)
am = AC.acceptance_manifest()
check("2.9 acceptance_manifest = effective manifest + the roster", am[C.ACCEPTANCE_ROSTER] == ROSTER and all(am[k] == m[k] for k in m))
check("2.10 no REPEAT / relocation items in acceptance-v2", C.repeat_items(C.ACCEPTANCE_NS, am) == [] and C.reloc_items(C.ACCEPTANCE_NS, am) == [])
check("2.11 main arms of acceptance-v2 = the 8 registered arms", list(C.main_arms_for(C.ACCEPTANCE_NS)) == list(C.MAIN_ARMS)
      and sorted(REG["arms"]["main"]) == sorted(C.MAIN_ARMS))

# ------------------------------------------------------------------------------------------------ 3 order + schedules
print("== 3 seeded order")
check("3.1 every registered key and arm order recomputes (52 presentations incl. spares)", A.order_problems(REG, ROSTER) == []
      and len(REG["execution_order"]["all_keys"]) == 52 and len(REG["execution_order"]["arm_order"]) == 52)
check("3.2 main order of the main roster = the registered main_order", A.main_order(MAINS) == REG["execution_order"]["main_order"])
t = copy.deepcopy(REG)
k0 = sorted(t["execution_order"]["all_keys"])[0]
t["execution_order"]["all_keys"][k0] = "0" * 64
check("3.3 a changed registered key is detected", any(k0 in x for x in A.order_problems(t, ROSTER)))
sch = A.plan_schedule(ROSTER)
check("3.4 planning schedule: 144 Gemma, 144 Opus, 96 Codex cells = 8 arms x 48", {k: len(v) for k, v in sch.items()} == {"gemma": 144, "opus": 144, "codex": 96})
check("3.5 Gemma cells in main_order x arm_order", [p for p, _ in sch["gemma"][::3]] == A.main_order(ROSTER)
      and all([a for (q, a) in sch["gemma"] if q == p] == [a for a in A.arm_order(p) if a[0] == "G"] for p in A.main_order(ROSTER)))
ex = A.exec_schedule(ROSTER)
check("3.6 execution: 48 presentations x (P0 + 8 arms), P0 first, arms = arm_key order",
      len(ex) == 48 and all(u[0] == "P0" and sorted(u[1:]) == sorted(C.MAIN_ARMS) and u[1:] == REG["execution_order"]["arm_order"][p] for p, u in ex))
swap = list(MAINS)
swap[swap.index(L3M[0])] = SPARES[0]
mo = A.main_order(swap)
check("3.7 a spare takes its own registered key; every other presentation keeps its relative order",
      [p for p in mo if SPARES[0] not in p] == [p for p in REG["execution_order"]["main_order"] if L3M[0] not in p]
      and all(REG["execution_order"]["all_keys"][p] == A.key("exec", p) for p in mo))

# ------------------------------------------------------------------------------------------------ 4 live checks + replacement rule
print("== 4 live checks and the replacement rule")
okrow = {"id": "x", "status": "ok", "goal_ok": True, "pre": {"scoring": 2, "oracle_setup": 1}, "audit": {"violations": 0, "counts": {"oracle": 0}}}
check("4.1 restore verdict PASS", A.restore_verdict(okrow) == "PASS")
check("4.2 goal not reached = FAIL", A.restore_verdict(dict(okrow, goal_ok=False, status="FAIL")) == "FAIL")
check("4.3 goal true before (scoring != 2) = FAIL", A.restore_verdict(dict(okrow, pre={"scoring": 1})) == "FAIL")
check("4.4 a violation = FAIL", A.restore_verdict(dict(okrow, audit={"violations": 1, "counts": {"oracle": 0}})) == "FAIL")
check("4.5 HUNG = UNRESOLVED", A.restore_verdict(dict(okrow, status="HUNG")) == "UNRESOLVED")
check("4.6 a missing audit field is never a 0 (UNRESOLVED)", A.restore_verdict(dict(okrow, audit={"counts": {}})) == "UNRESOLVED")
check("4.6b status FAIL with cause = exception (a crashed episode) is UNRESOLVED, never FAIL (night rule stage 3)",
      A.restore_verdict(dict(okrow, status="FAIL", goal_ok=None, cause="exception")) == "UNRESOLVED")
check("4.6c cause = hung is UNRESOLVED; a real goal failure stays FAIL",
      A.restore_verdict(dict(okrow, status="FAIL", cause="hung")) == "UNRESOLVED"
      and A.restore_verdict(dict(okrow, status="FAIL", goal_ok=False, cause="goal_unmet")) == "FAIL")


def st_all(v="PASS", **over):
    out = {i["id"]: {"verdict": v, "level": i["level"], "role": i["role"], "restores": {1: v, 2: v}} for i in REG["items"]}
    for k, x in over.items():
        out[k.replace("_", "-")]["verdict"] = x
    return out


def st(**over):
    s = st_all()
    for k, x in over.items():
        s[k]["verdict"] = x
    return s


r = A.resolve_roster(REG, st())
check("4.7 all PASS -> the main roster, no replacement", r["items"] == MAINS and not r["replacements"] and not r["blocked"])
r = A.resolve_roster(REG, st(**{L3M[2]: "FAIL"}))
check("4.8 one failed main L3 -> spare-1 in its place", r["items"][MAINS.index(L3M[2])] == SPARES[0] and not r["blocked"]
      and r["replacements"] == [{"failed": L3M[2], "spare": SPARES[0]}])
r = A.resolve_roster(REG, st(**{L3M[2]: "FAIL", SPARES[0]: "FAIL"}))
check("4.9 failed spare-1 is skipped -> spare-2", r["items"][MAINS.index(L3M[2])] == SPARES[1] and r["skipped_spares"] == [SPARES[0]] and not r["blocked"])
r = A.resolve_roster(REG, st(**{L3M[1]: "FAIL", L3M[5]: "FAIL"}))
check("4.10 two failed mains, slate order: first -> spare-1, second -> spare-2", r["replacements"] == [{"failed": L3M[1], "spare": SPARES[0]},
      {"failed": L3M[5], "spare": SPARES[1]}] and not r["blocked"])
r = A.resolve_roster(REG, st(**{L3M[1]: "FAIL", L3M[5]: "FAIL", L3M[6]: "FAIL"}))
check("4.11 more than 2 failed L3 items BLOCKS", any("> 2" in b for b in r["blocked"]))
r = A.resolve_roster(REG, st(**{MAINS[0]: "FAIL"}))
check("4.12 a failed L1 item BLOCKS (no spare for L1 / L2; never reclassified)", any("no spare exists" in b for b in r["blocked"]))
r = A.resolve_roster(REG, st(**{L3M[0]: "UNRESOLVED"}))
check("4.13 an UNRESOLVED main BLOCKS and uses no spare", r["blocked"] and not r["replacements"])
r = A.resolve_roster(REG, st(**{L3M[0]: "FAIL", SPARES[0]: "UNRESOLVED"}))
check("4.14 a needed spare that is unresolved BLOCKS", any("spare" in b for b in r["blocked"]) and not r["replacements"])
r = A.resolve_roster(REG, st(**{L3M[0]: "FAIL", L3M[1]: "FAIL", SPARES[1]: "FAIL"}))
check("4.15 too few passing spares BLOCKS", bool(r["blocked"]))
r = A.resolve_roster(REG, st(**{MAINS[3]: "MISSING"}))
check("4.16 a reserved item whose check has not run BLOCKS", any("MISSING" in b for b in r["blocked"]))

# roster section integrity
s_ok = A.roster_section(A.resolve_roster(REG, st(**{L3M[2]: "FAIL", SPARES[0]: "FAIL"})), st(**{L3M[2]: "FAIL", SPARES[0]: "FAIL"}))
check("4.17 a resolved roster section validates", AC.roster_section_problems(REG, s_ok) == [], AC.roster_section_problems(REG, s_ok))
b1 = copy.deepcopy(s_ok)
b1["items"][0] = SPARES[0]
check("4.18 an L1 item swapped for a spare is refused", bool(AC.roster_section_problems(REG, b1)))
b2 = copy.deepcopy(s_ok)
b2["skipped_spares"] = []
check("4.19 a skipped spare that is not declared is refused", bool(AC.roster_section_problems(REG, b2)))
b3 = copy.deepcopy(s_ok)
b3["items"] = b3["items"][:23]
check("4.20 a 23-item roster is refused", bool(AC.roster_section_problems(REG, b3)))
b4 = copy.deepcopy(s_ok)
b4["sealed"] = False
check("4.21 an unsealed roster section is refused", bool(AC.roster_section_problems(REG, b4)))

# seal_roster on a temporary root with synthetic live checks
TMP = tempfile.mkdtemp(prefix="acc_rt_")
try:
    os.makedirs(os.path.join(TMP, "docs", "final-test"))
    SEALED0 = os.path.join(HERE, "final_test", "_checkpoints", "ACC-RUNNER", "pre", "acceptance_registration.json")   # the C11-AND-REG seal (74b5cf56)
    for f in ("split_manifest.json", "inclusion_manifest_a26.json"):
        shutil.copy(os.path.join(ROOT, "docs", "final-test", f), os.path.join(TMP, "docs", "final-test", f))
    shutil.copy(SEALED0, os.path.join(TMP, "docs", "final-test", "acceptance_registration.json"))
    dirs = {"L3": os.path.join(TMP, "live", "acc-add"), "L1L2": os.path.join(TMP, "live", "reserved16")}

    CFG_H = A.expected_config_hash({"mode": "honest", "policy": "console", "dataset_sha256": REG["pins"]["dataset"]})[0]

    def write_live(fail=()):
        for grp, d in dirs.items():
            for k in (1, 2):
                os.makedirs(os.path.join(d, f"honest-r{k}"), exist_ok=True)
                ids_ = [i["id"] for i in REG["items"] if (i["level"] == "L3") == (grp == "L3")]
                json.dump({"mode": "honest", "restore": k, "items": ids_, "dataset_sha256": REG["pins"]["dataset"], "policy": "console",
                           "harness_sha256": A.file_sha(A.HARNESS)}, open(os.path.join(d, f"honest-r{k}", "plan.json"), "w", encoding="utf-8"))
                open(os.path.join(d, f"honest-r{k}.done"), "w").write("done\n")
                with open(os.path.join(d, f"honest-r{k}", "refplans_honest.jsonl"), "w", encoding="utf-8") as f:
                    for i in REG["items"]:
                        if (i["level"] == "L3") != (grp == "L3"):
                            continue
                        row = dict(okrow, id=i["id"], mode="honest", hash=CFG_H)   # ACC-RUNNER-D: the plan's recomputed configuration
                        if i["id"] in fail:
                            row = dict(row, status="FAIL", goal_ok=False)
                        f.write(json.dumps(row) + "\n")
    write_live(fail=(L3M[3],))
    stt = A.live_status(REG, TMP, dirs)
    check("4.22 live_status reads both restores per item", stt[L3M[3]]["verdict"] == "FAIL" and stt[MAINS[0]]["verdict"] == "PASS"
          and stt[MAINS[0]]["restores"] == {1: "PASS", 2: "PASS"})
    new = A.seal_roster(TMP, write=False, dirs=dirs)
    check("4.23 seal-roster dry run writes nothing", A.file_sha(AC.acceptance_registration_path(TMP)) == AC.ACC_REG_SEALED_FILE_SHA
          and new["roster"]["items"][MAINS.index(L3M[3])] == SPARES[0])
    A.seal_roster(TMP, write=True, dirs=dirs)
    reg2 = AC.load_acceptance_registration(TMP)
    check("4.24 sealed registration verifies (frozen sections unchanged, new self-sha)", AC.acceptance_registration_problems(reg2) == []
          and reg2["sha256"] != REG["sha256"])
    check("4.25 acceptance_roster now returns the sealed roster", AC.acceptance_roster(TMP) == new["roster"]["items"])
    check("4.26 a second seal is refused", raises(lambda: A.seal_roster(TMP, write=True, dirs=dirs), A.Refused))
    check("4.27 the spare is admitted and the replaced main refused after the seal",
          guard_or_none([SPARES[0]], C.ACCEPTANCE_NS, root=TMP)["n"] == 1 and raises(lambda: C.guard_items([L3M[3]], C.ACCEPTANCE_NS, root=TMP), C.TestItemRefused))
    shutil.copy(SEALED0, os.path.join(TMP, "docs", "final-test", "acceptance_registration.json"))
    os.makedirs(C.ns_docs(C.ACCEPTANCE_NS, TMP), exist_ok=True)
    open(C.journal_path(C.ACCEPTANCE_NS, "capture", TMP), "w").write("{}\n")
    check("4.28 after the first acceptance call (a capture row) no roster is sealed", raises(lambda: A.seal_roster(TMP, write=True, dirs=dirs), A.Refused))
    os.remove(C.journal_path(C.ACCEPTANCE_NS, "capture", TMP))
    write_live(fail=(MAINS[0],))
    check("4.29 seal refused when a reserved L1 failed", raises(lambda: A.seal_roster(TMP, write=True, dirs=dirs), A.Refused))
    write_live()
    os.remove(os.path.join(dirs["L3"], "honest-r2", "plan.json"))
    check("4.30 a restore without its predeclared plan.json is UNRESOLVED (binding)", A.live_status(REG, TMP, dirs)[L3M[0]]["verdict"] == "UNRESOLVED")
    write_live()
    check("4.31 the clean fixture binds (every item PASS)", all(v["verdict"] == "PASS" for v in A.live_status(REG, TMP, dirs).values()))
    pp = os.path.join(dirs["L3"], "honest-r1", "plan.json")
    doc = json.load(open(pp, encoding="utf-8"))
    json.dump(dict(doc, harness_sha256="0" * 64), open(pp, "w", encoding="utf-8"))
    check("4.32 a plan.json made for another harness leaves the restore UNRESOLVED (rows bound to the plan, Astra r2 Q7)",
          A.live_status(REG, TMP, dirs)[L3M[0]]["restores"][1] == "UNRESOLVED")
    write_live()
    os.remove(os.path.join(dirs["L3"], "honest-r2.done"))
    check("4.33 a restore without its .done marker is UNRESOLVED", A.live_status(REG, TMP, dirs)[L3M[0]]["verdict"] == "UNRESOLVED")
    write_live()
    with open(os.path.join(dirs["L3"], "honest-r1", "refplans_honest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(okrow, id="g-l1-pickup-apple", mode="honest", hash="h1")) + "\n")
    check("4.34 a result row the predeclared plan does not list unbinds the folder", A.live_status(REG, TMP, dirs)[L3M[0]]["restores"][1] == "UNRESOLVED")
    write_live()
    with open(os.path.join(dirs["L3"], "honest-r1", "refplans_honest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(okrow, id=L3M[1], mode="honest", hash="h2")) + "\n")
    check("4.35 rows of two harness configurations in one folder unbind it", A.live_status(REG, TMP, dirs)[L3M[0]]["restores"][1] == "UNRESOLVED")
    write_live()
    lb = A.live_binding(TMP, dirs)
    check("4.36 live_binding names the plan.json + rows sha of every restore folder",
          all(v["plan_sha"] and v["rows_sha"] for g_ in lb.values() for v in g_.values()) and sorted(lb) == ["L1L2", "L3"])
finally:
    shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ 5 spend
print("== 5 spend ceilings, study-wide check, fake over-cap")
TMP = tempfile.mkdtemp(prefix="acc_sp_")
try:
    p = os.path.join(TMP, C.ACCEPTANCE_NS, "ledger.db")
    os.makedirs(os.path.dirname(p))
    other = {"settled": 3.0, "open": 0.0, "committed": 3.0}
    led = C.ScopedLedger(p, C.NAMESPACES[C.ACCEPTANCE_NS]["ceilings"])
    with led.use("core"):
        check("5.1 an acceptance ledger WITHOUT its guard refuses a paid reservation (fail closed)",
              raises(lambda: led.reserve("anthropic", 0.01), C.CostCeilingExceeded))
    LOCK5 = os.path.join(TMP, "study5.lock")
    AC.register_ledger_guard(p, attempt_base_usd=0.0, attempt_cap_usd=8.0, other_spend=lambda: dict(other), lock_path=LOCK5)
    led = C.ScopedLedger(p, C.NAMESPACES[C.ACCEPTANCE_NS]["ceilings"])
    with led.use("core"):
        rid = led.reserve("anthropic", 7.5)
        check("5.2 a reservation inside every constraint is accepted", rid > 0)
        check("5.3 the attempt ceiling (8 USD) refuses the next 0.51", raises(lambda: led.reserve("anthropic", 0.51), C.CostCeilingExceeded))
        led.settle(rid, 0.5, {"provider": "anthropic"})
        check("5.4 settled at the billed cost frees the room", led.reserve("anthropic", 7.4) > 0)
    worker = C.ScopedLedger(p, C.NAMESPACES[C.ACCEPTANCE_NS]["ceilings"], release_stale=False)
    with worker.use("core"):
        check("5.5 a second ledger object on the same file (worker / batch driver) applies the same guard",
              raises(lambda: worker.reserve("anthropic", 0.2), C.CostCeilingExceeded))
    AC.register_ledger_guard(p, attempt_base_usd=led.committed(), attempt_cap_usd=8.0, other_spend=lambda: dict(other), lock_path=LOCK5)
    with led.use("core"):
        check("5.6 the namespace total 16 still binds under a fresh attempt base",
              raises(lambda: led.reserve("anthropic", 16.0 - led.committed() + 0.01), C.CostCeilingExceeded))
    other.update(settled=170.0, committed=170.0)
    AC.register_ledger_guard(p, attempt_base_usd=led.committed(), attempt_cap_usd=8.0, other_spend=lambda: dict(other), lock_path=LOCK5)
    with led.use("core"):
        room = 180.0 - 170.0 - led.committed()                # A37: the study cap is 180 USD (A24.10: 100); literal, not read back
        check("5.7 the A29 study-wide check (other ledgers + this one + next <= 180, A37) refuses one cent over",
              raises(lambda: led.reserve("anthropic", room + 0.01), C.CostCeilingExceeded))
    _sp = {"settled": 179.99, "open": 0.0, "committed": 179.99, "ledgers": {}, "errors": []}
    check("5.7b A37 boundary of the study-wide check: 179.99 + 0.01 = 180.00 is inside, + 0.02 is refused (the old 100 cap no longer binds)",
          AC.study_wide_problems(0.01, spend=_sp) == [] and AC.study_wide_problems(0.02, spend=_sp) != []
          and AC.study_wide_problems(0.01, spend=dict(_sp, settled=100.0, committed=100.0)) == [])
    fo = AC.fake_overcap_test(led)
    check("5.8 fake over-cap test: refused, nothing open afterwards", fo["ok"] and fo["refused"] and fo["open_holds_after"] == 0, fo)
    check("5.9 a block hold re-reads the other ledgers (refresh on every submission)", AC.ledger_guard(p)["other"]["committed"] == 170.0)
    AC.unregister_ledger_guard(p)
    q = os.path.join(TMP, "rehearsal-v3", "ledger.db")
    os.makedirs(os.path.dirname(q))
    rl = C.ScopedLedger(q, C.NAMESPACES["rehearsal-v3"]["ceilings"])
    with rl.use("core"):
        check("5.10 other namespaces are unaffected (no guard needed)", rl.reserve("anthropic", 0.1) > 0)
    led_real_files = AC.study_spend(ROOT)
    check("5.11 study_spend reads the repo ledgers (settled + open), read-only", led_real_files["committed"] > 0 and led_real_files["ledgers"])
    fo2 = A.fake_overcap()
    check("5.12 acc_runner fake-overcap (temporary ledger) refuses and leaves nothing open", fo2["ok"], fo2)
    check("5.14 attempt 2 is refused (the correction cycle is not built)", raises(lambda: A.arm_guard(None, 2, path=p, record=False), A.Refused))
    import sqlite3 as _sq
    sd = os.path.join(TMP, "study")
    os.makedirs(os.path.join(sd, "sim", "x"))
    os.makedirs(os.path.join(sd, "docs", "final-test"))
    shutil.copy(os.path.join(ROOT, "docs", "final-test", "split_manifest.json"), os.path.join(sd, "docs", "final-test"))
    l2 = C.ScopedLedger(os.path.join(sd, "sim", "x", "ledger.db"), {"core": 5.0, "total": 5.0})
    with l2.use("core"):
        l2.reserve("anthropic", 0.2)
        l2.reserve("anthropic", 0.2)
    check("5.15 study_spend counts two distinct identical reservations twice (row identity, Astra Q2)", abs(AC.study_spend(sd)["open"] - 0.4) < 1e-9, AC.study_spend(sd))
    shutil.copy(os.path.join(sd, "sim", "x", "ledger.db"), os.path.join(sd, "docs", "ledger_copy.db"))
    check("5.16 an archive COPY of a ledger is over-counted, never under-counted (Astra r2: no dedup without a ledger identity)",
          abs(AC.study_spend(sd)["open"] - 0.8) < 1e-9)
    os.remove(os.path.join(sd, "docs", "ledger_copy.db"))
    open(os.path.join(sd, "sim", "x", "notaledger.db"), "wb").write(b"garbage, not sqlite")
    check("5.17 an unreadable database file is an accounting error (it might be a cost ledger), never a silent 0",
          bool(AC.study_spend(sd)["errors"]))
    os.remove(os.path.join(sd, "sim", "x", "notaledger.db"))
    bad_db = os.path.join(sd, "sim", "x", "broken.db")
    cc = _sq.connect(bad_db)
    cc.execute("CREATE TABLE ledger (id INTEGER PRIMARY KEY)")
    cc.commit()
    cc.close()
    sp = AC.study_spend(sd)
    check("5.18 an unreadable cost ledger is an accounting error and the study-wide check refuses", sp["errors"] and AC.study_wide_problems(0.01, spend=sp), sp)
    orig_pn, orig_ss, orig_lp = AC.paid_namespace_of_repo, AC.study_spend, AC.study_lock_path
    AC.study_lock_path = lambda path: os.path.join(TMP, "study.lock")
    AC.paid_namespace_of_repo = lambda path: "rehearsal-v3"
    AC.study_spend = lambda root=None, exclude=(): {"settled": 179.99, "open": 0.0, "committed": 179.99, "ledgers": {}, "errors": []}
    try:
        l3 = C.ScopedLedger(os.path.join(TMP, "rv3", "ledger.db"), {"core": 5.0, "total": 5.0})
        with l3.use("core"):
            check("5.19 every paid namespace of the repo gets the A29 study-wide check without a registered guard (A37 cap 180: 179.99 + 0.02 "
                  "refused, + 0.01 accepted)",
                  raises(lambda: l3.reserve("anthropic", 0.02), C.CostCeilingExceeded) and l3.reserve("anthropic", 0.01) > 0)
    finally:
        AC.paid_namespace_of_repo, AC.study_spend, AC.study_lock_path = orig_pn, orig_ss, orig_lp
    check("5.20 paid_namespace_of_repo: the real work dir of a paid namespace yes, a temporary tree no",
          AC.paid_namespace_of_repo(os.path.join(C.paths(None)["matrix_work"], "test-v1", "ledger.db")) == "test-v1"
          and AC.paid_namespace_of_repo(os.path.join(TMP, "test-v1", "ledger.db")) is None
          and AC.paid_namespace_of_repo(os.path.join(C.paths(None)["matrix_work"], "practice-v2", "ledger.db")) is None)
    fc = A.batch_submissions_forecast(144, 0.32, 0.045)
    check("5.13 the worst-case hold does not fit 144 Opus cells in one 8 USD submission (chunks forecast > 1)",
          fc["fits"] and fc["chunks"] > 1 and sum(fc["sizes"]) == 144, fc)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ 6 phase subclasses (no simulator, no call)
print("== 6 phases")
# ACC-RUNNER-D: the real docs/final-test now holds Arash's rulings, his (conditional) authorization and the owed records. Sections 6-9 test
# the refusal paths and must never depend on (or be able to pass) the real preflight -- a clean real preflight would let A.main(["capture"])
# start a live phase -- so these three files are HIDDEN from the runner for the rest of this selftest; the real state is checked read-only
# at the end (section 10).
REAL_DJ = A._docs_json
HIDDEN_DOCS = (A.AUTH_FILE, A.RULINGS_FILE, A.OWED_FILE)
A._docs_json = lambda name, root=None: (None, REAL_DJ(name, root)[1]) if name in HIDDEN_DOCS else REAL_DJ(name, root)
import inspect                                            # noqa: E402
src = inspect.getsource(A._AccOrder._open)
check("6.1 the plan subclass registers the guard BEFORE any ledger exists on the file", src.index("arm_guard") < src.index("super()._open()"))
src = inspect.getsource(A._AccOrder.run)
check("6.1b a direct AccPlan.run() goes through enter() (preflight + session) before planning", src.index("enter(") < src.index("super().run()"))
check("6.1c enter() refuses now (nothing authorized, roster not sealed, protocol open)", raises(lambda: A.enter("gemma"), A.Refused))


class _FakeSel(A._AccOrder):
    root = None

    def __init__(self, sel):
        self.sel = sel


cells = [{"pres": p, "arm": a, "key": f"k|{a}|{p}"} for p in A.presentations(ROSTER) for a in C.MAIN_ARMS if a[0] == "G"]
o = _FakeSel(list(reversed(cells)))._order()
check("6.2 plan order = main_order, arms in arm_key order", [p for p, _ in o] == A.main_order(ROSTER)
      and all([c["arm"] for c in cs] == [a for a in A.arm_order(p) if a[0] == "G"] for p, cs in o))
check("6.3 a cell outside the registered presentations aborts", raises(lambda: _FakeSel(cells + [{"pres": "canon:g-l1-pickup-apple:en", "arm": "G0", "key": "x"}])._order(), RuntimeError))
import execute as EX                                      # noqa: E402
Cls = A.exec_class()
ph = Cls.__new__(Cls)
ph.root = None
p0 = A.main_order(ROSTER)[0]
units = {}
for a in ["P0"] + list(C.MAIN_ARMS):
    units["u" + a] = {"cells": [{"arm": a, "key": a}]}
units["uG0"]["cells"].append({"arm": "X0", "key": "X0"})        # a shared unit (identical plans): ranked by its earliest arm
del units["uX0"]
ph.b = {"units": units}
orig = EX.ExecutePhase.todo
EX.ExecutePhase.todo = lambda self: {0: [(p0, sorted(units))]}
try:
    got = ph.todo()[0][0][1]
finally:
    EX.ExecutePhase.todo = orig
want_arms = ["P0"] + A.arm_order(p0)
rank = {a: k for k, a in enumerate(want_arms)}
exp = sorted(units, key=lambda uk: min(rank[c["arm"]] for c in units[uk]["cells"]))
check("6.4 execution: P0 unit first, then units by their earliest registered arm", got[0] == "uP0" and got == exp, (got, exp))
EX.ExecutePhase.todo = lambda self: {0: [("canon:g-l1-pickup-apple:en", ["u"])]}
try:
    check("6.5 an execution unit outside the registered presentations aborts", raises(lambda: ph.todo(), EX.ExecAborted))
finally:
    EX.ExecutePhase.todo = orig
src = inspect.getsource(A.make_execute)
check("6.6 execute: canon only, P0 only (no P+M), no /p replays", "p_arms=(P0,)" in src and "replay=False" in src and 'worlds=("canon",)' in src)
src = inspect.getsource(A.make_plan)
check("6.7 the Opus stage uses the Batch API path; r = 0 only", "batch = stage == \"opus\"" in src and "rs=[0]" in src and A.STAGES["opus"] == ("core", "fs"))
pf = A.preflight(identity=False)
check("6.8 preflight refuses now: roster, authorization, the open one-batch protocol question, the owed items "
      "(or, after attempt 1's recorded verdict, the CLOSED attempt itself)",
      any(x.startswith("attempt 1 is CLOSED") for x in pf) or ((any("roster not sealed" in x for x in pf) or REG.get("roster")) and any("no acceptance authorization" in x for x in pf)
      and any("PROTOCOL OPEN [one-batch]" in x for x in pf) and any(x.startswith("owed item") for x in pf)), pf[:6])
fake = {"schema": A.AUTH_SCHEMA, "by": "Arash", "date": "2026-10-14", "said": "go", "attempt": 1, "registration_sha": REG["sha256"],
        "roster_sha": A.roster_sha(ROSTER), "ledger_entry": "STAGE C11-AND-REG -- STARTED: close the owed C11 items"}
fake["sha256"] = C.self_sha(fake)
orig_dj = A._docs_json
A._docs_json = lambda name, root=None: (copy.deepcopy(fake), "x") if name == A.AUTH_FILE else orig_dj(name, root)
try:
    ap_ = A.authorization_problems(REG, ROSTER, 1)
finally:
    A._docs_json = orig_dj
check("6.9 an authorization naming an unrelated progress heading (not an ARASH entry) is refused (Astra Q7)", any("ARASH" in x for x in ap_), ap_)
fake2 = dict(fake, roster_sha="0" * 64)
fake2["sha256"] = C.self_sha(fake2)
A._docs_json = lambda name, root=None: (copy.deepcopy(fake2), "x") if name == A.AUTH_FILE else orig_dj(name, root)
try:
    ap_ = A.authorization_problems(REG, ROSTER, 1)
finally:
    A._docs_json = orig_dj
check("6.9b an authorization bound to another roster is refused", any("another roster" in x for x in ap_))
check("6.9c record_authorization refuses anyone but Arash, and refuses before the roster is sealed",
      raises(lambda: A.record_authorization(by="Claude", date="d", said="s", ledger_entry="x"), (PermissionError, A.Refused))
      and (REG.get("roster") or raises(lambda: A.record_authorization(by="Arash", date="d", said="s", ledger_entry="x"), A.Refused)))
sp_ = A.stage_prereq_problems("opus", ROSTER, root=tempfile.mkdtemp(prefix="acc_sp_"))  # hermetic (ACC-V2-R1: the live journals now hold capture + Gemma)
check("6.9d stage order: Opus needs a finished capture and a finished Gemma stage", any("capture" in x for x in sp_) and any("gemma" in x for x in sp_), sp_)
check("6.9e attempt 2 is refused by the preflight", any("attempt 2" in x for x in A.preflight(attempt=2, identity=False)))
check("6.10b the CLI refuses --attempt 2 for live phases and the scorer (Astra r2 new defect 1)",
      A.main(["plan", "--stage", "gemma", "--attempt", "2"]) == 2 and A.main(["score", "--attempt", "2"]) == 2)
rul = {"one-batch": {"schema": A.RULING_SCHEMA, "key": "one-batch", "by": "Arash", "date": "2026-10-14", "said": "REJECT; remain blocked",
                     "ruling": "remain blocked", "registration_sha": REG["sha256"]}}


def _seal(e):
    e = {k: v for k, v in e.items() if k != "sha256"}
    e["sha256"] = C.self_sha(e)
    return e


A._docs_json = lambda name, root=None: (copy.deepcopy(rul), "x") if name == A.RULINGS_FILE else orig_dj(name, root)
try:
    rul["one-batch"] = _seal(rul["one-batch"])
    still = A.ruling_open("one-batch")                      # Astra's probe: "REJECT; remain blocked", no approval
    rul["one-batch"] = _seal(dict(rul["one-batch"], approved=False))
    still_f = A.ruling_open("one-batch")
    rul["one-batch"] = _seal(dict(rul["one-batch"], approved=True, registration_sha="0" * 64))
    still2 = A.ruling_open("one-batch")
    rul["one-batch"] = _seal(dict(rul["one-batch"], registration_sha=REG["sha256"]))
    lifted = not A.ruling_open("one-batch")
    rul["one-batch"] = dict(rul["one-batch"], said="edited after writing")
    still3 = A.ruling_open("one-batch")
    rul["one-batch"] = _seal(dict(rul["one-batch"], by="Claude"))
    still4 = A.ruling_open("one-batch")
finally:
    A._docs_json = orig_dj
check("6.10c Astra r2 probe: a ruling saying 'REJECT; remain blocked' (no approval) keeps the blocker", still and still_f)
check("6.10d a ruling clears a blocker only when approved AND bound to the current registration AND unedited AND by Arash",
      still2 and lifted and still3 and still4)
AC.disarm_session()
check("6.10 the CLI live phases refuse at the preflight", A.main(["plan", "--stage", "gemma"]) == 2 and A.main(["execute"]) == 2
      and A.main(["capture"]) == 2 and not AC.SESSION["armed"])
check("6.11 nothing was written into the acceptance-v2 namespace by this selftest (docs + work byte-identical to the start)",
      ns_snapshot() == NS0, sorted(set(ns_snapshot().items()) ^ set(NS0.items()))[:4])

# ------------------------------------------------------------------------------------------------ 7 decision rule (synthetic evidence)
print("== 7 decision rule")
PRES = A.presentations(ROSTER)
INV = A.source_inventory()


def synth():
    cap, plans, exe, outc, units = [], [], [], [], {}
    for p in PRES:
        item = C.parse_pres(p)["item"]
        cap.append({"pres": p, "item": item, "namespace": C.ACCEPTANCE_NS, "treatment_id": A.TREATMENT_SHA, "status": "ok", "capture_run": 1,
                    "requests": {"none": "n" + p, "S*": "m" + p, "FS": "f" + p},
                    "audit": {"honesty": {"violations": 0, "oracle": 0, "map_build": 0}, "leakage": {"ok": True}, "g_iso": {"ok": True}},
                    "blocks": {"S*": {"real_tokens": 900, "ids": ["places/x"], "package": {"parser_ok": True, "leave_out": [],
                                                                                          "provenance": [{"id": "places/x", "source": "practice:ltm/FloorPlan1",
                                                                                                          "evidence": [INV["claims"][0]],
                                                                                                          "authored_by": "script:ltm_package.PlacesSource"}]}},
                               "FS": {"real_tokens": 700, "fs": {"retained": ["example/practice-kitchen/practice-v2:o0:a1"]}}}})
        for a in list(C.MAIN_ARMS) + ["P0"]:
            k = C.cell_id(C.ACCEPTANCE_NS, "plan", a, p, "r0")
            src = C.arm_info(a)["prompt"] if a != "P0" else "none"
            if a != "P0":
                rq = {"none": "n", "S*": "m", "FS": "f"}[src] + p
                plans.append({"key": k, "arm": a, "pres": p, "status": "ok", "treatment_id": A.TREATMENT_SHA, "request_sha": rq,
                              "attempts": [{"prompt_sha256": rq, "valid": True}]})
            uk = "u" + a + p
            units[uk] = {"cells": [{"arm": a, "key": k}]}
            exe.append({"unit": uk, "run": 0, "status": "done", "audit": {"violations": 0, "counts": {"oracle": 0}}, "state_check": {"ok": True},
                        "snapshot_sha_before": None, "snapshot_sha_after": None})
            outc.append({"unit": uk, "goal_ok": True})
    checks = [{"pres": p, "source": s, "model": "gemma", "mode": "measured", "fits": True, "request_sha": {"none": "n", "S*": "m", "FS": "f"}[s] + p}
              for p in PRES for s in ("none", "S*", "FS")]
    return {"namespace": C.ACCEPTANCE_NS, "attempt": 1, "capture": cap, "plan": plans, "exec": exe, "outcome": outc,
            "capture_sha": "cs", "plan_sha": "ps", "g_ruling_open": True,
            "events": [{"event": "capture_gates", "gates": {n_: {"pass": True} for n_ in ("snapshot_hash", "all_cells", "g_iso", "masking")}}],
            "units": {"units": units}, "sources": copy.deepcopy(INV),
            "context_fit": {"namespace": C.ACCEPTANCE_NS, "gemma_verdict": "PASS_MEASURED", "freeze_grade": True, "checks": checks, "capture_sha": "cs"},
            "c11": {"ok": True, "problems": [], "unresolved_gold": [], "capture_sha": "cs", "plan_sha": "ps"}}


d = A.decide(synth(), REG, ROSTER)
check("7.1 complete clean synthetic evidence -> NOT PASS only because (g) 'no eligible record missing' is PENDING (Astra Q4)",
      d["verdict"] == "NOT PASS" and [k for k, v in d["criteria"].items() if not v["pass"]] == ["g"]
      and "PENDING" in d["criteria"]["g"]["reasons"][0], {k: v["reasons"][:2] for k, v in d["criteria"].items() if not v["pass"]})
ev_ = synth()
ev_["g_ruling_open"] = False
d_ = A.decide(ev_, REG, ROSTER)
check("7.1b a recorded ruling alone does not pass (g): without the recomputation (g_eligible.json) (g) FAILS",
      d_["verdict"] == "NOT PASS" and [k for k, v in d_["criteria"].items() if not v["pass"]] == ["g"]
      and any("g_eligible.json missing" in x for x in d_["criteria"]["g"]["reasons"]), d_["criteria"]["g"]["reasons"][:2])


def fails(mut, want):
    ev = synth()
    mut(ev)
    dd = A.decide(ev, REG, ROSTER)
    bad = sorted(k for k, v in dd["criteria"].items() if not v["pass"])
    return dd["verdict"] == "NOT PASS" and want <= set(bad), bad


def m_viol(ev):
    ev["exec"][5]["audit"]["violations"] = 1


def m_audit_missing(ev):
    ev["exec"][7]["audit"] = {"counts": {}}


def m_p0(ev):
    k = C.cell_id(C.ACCEPTANCE_NS, "plan", "P0", PRES[3], "r0")
    ev["units"]["units"] = {u: x for u, x in ev["units"]["units"].items() if x["cells"][0]["key"] != k}


def m_f(ev):
    for r in ev["plan"]:
        if r["arm"] == "O+M" and r["pres"] in PRES[:3]:
            r["status"] = "rejected"


def m_f2(ev):
    for r in ev["plan"]:
        if r["arm"] == "O+M" and r["pres"] in PRES[:2]:
            r["status"] = "rejected"


def m_tok(ev):
    ev["capture"][4]["blocks"]["S*"]["real_tokens"] = 2001


def m_req(ev):
    for r in ev["plan"]:
        if r["arm"] == "O+FS" and r["pres"] == PRES[9]:
            r["request_sha"] = "other"


def m_cf(ev):
    ev["context_fit"] = None


def m_cf2(ev):
    ev["context_fit"]["checks"] = ev["context_fit"]["checks"][1:]


def m_c11(ev):
    ev["c11"] = None


def m_gold(ev):
    ev["c11"] = {"ok": False, "problems": ["1 hit"], "unresolved_gold": [{"item": "g-l2-slice-bread"}]}


def m_stale(ev):
    ev["plan"][0]["treatment_id"] = "9b26a99f" * 8


def m_trunc(ev):
    ev["exec"].append({"_corrupt": True})


def m_extra(ev):
    ev["plan"].append(dict(ev["plan"][0], key=C.cell_id(C.ACCEPTANCE_NS, "plan", "G0", "canon:g-l1-pickup-apple:en", "r0")))


def m_missing_cell(ev):
    ev["plan"] = ev["plan"][1:]


def m_prov(ev):
    ev["capture"][2]["blocks"]["S*"]["package"]["provenance"] = []


def m_snap(ev):
    ev["events"][0]["gates"]["snapshot_hash"]["pass"] = False


def m_goal(ev):
    k = C.cell_id(C.ACCEPTANCE_NS, "plan", "P0", PRES[0], "r0")
    uk = [u for u, x in ev["units"]["units"].items() if x["cells"][0]["key"] == k][0]
    for o_ in ev["outcome"]:
        if o_["unit"] == uk:
            o_["goal_ok"] = False


def m_leak(ev):
    ev["capture"][1]["audit"]["leakage"]["ok"] = False


def m_cap_missing(ev):
    ev["capture"] = ev["capture"][1:]


def m_nocap(ev):
    ev["capture"] = None


def m_c11_plan(ev):
    ev["c11"]["plan_sha"] = "older"


def m_early_viol(ev):
    ev["exec"].append(dict(ev["exec"][3], attempt=0, audit={"violations": 2, "counts": {"oracle": 0}}))


def m_attempts(ev):
    ev["plan"][6]["attempts"] = [{}] * 5


def m_noids(ev):
    ev["capture"][5]["blocks"]["S*"].pop("ids")


def m_schema(ev):
    ev["outcome"][2].pop("goal_ok")


for name, mut, want in (("7.2 one violation -> (a)", m_viol, {"a"}), ("7.3 a missing audit field -> (a), never a 0", m_audit_missing, {"a"}),
                        ("7.4 a missing P0 row -> (e) + completeness", m_p0, {"e", "completeness"}),
                        ("7.5 45 / 48 valid plans in one arm -> (f)", m_f, {"f"}),
                        ("7.7 package 2,001 tokens -> (g)", m_tok, {"g"}), ("7.8 O+FS bytes differ from the captured FS request -> (h)", m_req, {"h"}),
                        ("7.9 no context-fit file -> (i)", m_cf, {"i"}), ("7.10 one Gemma pair not measured -> (i)", m_cf2, {"i"}),
                        ("7.11 no C11 acceptance scan -> c11", m_c11, {"c11"}), ("7.12 an unresolved gold hit -> (b) + c11 (no pass pending investigation)", m_gold, {"b", "c11"}),
                        ("7.13 a stale treatment id -> completeness", m_stale, {"completeness"}), ("7.14 a truncated row -> completeness", m_trunc, {"completeness"}),
                        ("7.15 an extra cell -> completeness", m_extra, {"completeness"}), ("7.16 a missing plan cell -> completeness + (f)", m_missing_cell, {"completeness"}),
                        ("7.17 a record without provenance -> (c)", m_prov, {"c"}), ("7.18 snapshot hash gate failed -> (d)", m_snap, {"d"}),
                        ("7.19 P0 misses the goal on a feasible item -> (e)", m_goal, {"e"}), ("7.20 a capture leakage gate not ok -> (b)", m_leak, {"b"}),
                        ("7.21 a missing capture row -> completeness", m_cap_missing, {"completeness"}),
                        ("7.22 no capture journal at all -> completeness", m_nocap, {"completeness"}),
                        ("7.25 C11 computed on an older plan journal -> c11 (stale binding)", m_c11_plan, {"c11"}),
                        ("7.26 an EARLIER violating execution attempt -> (a)", m_early_viol, {"a"}),
                        ("7.27 more planner attempts than 1 + 2 repairs -> completeness", m_attempts, {"completeness"}),
                        ("7.28 missing package record ids -> (c)", m_noids, {"c"}),
                        ("7.29 a row without a required field -> completeness", m_schema, {"completeness"})):
    ok_, bad_ = fails(mut, want)
    check(name, ok_, bad_)
ev = synth()
m_f2(ev)
dd = A.decide(ev, REG, ROSTER)
check("7.6 46 / 48 valid plans in an arm is still inside (f)", dd["criteria"]["f"]["pass"] and dd["valid_plans_per_arm"]["O+M"] == 46)
check("7.23 SR per arm is reported descriptively and never gated", "sr_descriptive_not_gated" in d and all(not k.startswith("sr") for k in d["criteria"]))
check("7.24 decide() has no threshold argument (acceptance is never tuned)", list(inspect.signature(A.decide).parameters) == ["ev", "reg", "roster", "treatment_id"])

# ------------------------------------------------------------------------------------------------ 7g criterion (g) "no eligible record missing" (ACC-G)
print("== 7g criterion (g): no eligible record missing (Arash ruling 2; synthetic + rehearsal-v3 read-only)")
import treatment as T_                                   # noqa: E402
import type_scope as TS_                                 # noqa: E402
import ltm_package as LP_                                # noqa: E402
import re as re_                                         # noqa: E402

G_SNAP = "s" * 64
G_ROWS = "r" * 64


def g_synth(ev=None):
    """synth() + a consistent S* package record on every block + the matching g_eligible.json (ruling recorded)."""
    ev = ev or synth()
    ev["g_ruling_open"] = False
    per = {}
    for r in ev["capture"]:
        b = r["blocks"]["S*"]
        p = r["pres"]
        b["text"] = "Package text of " + p
        b["sha256"] = C.sha256_text(b["text"])
        b["source"] = {"snapshot_sha": G_SNAP, "rows_sha": G_ROWS, "rows_n": 10}
        b["package"].update(pkg_sha="k" + p, params_sha="prm", leave_out=[{"component": "procedure", "id": "procedure/*/lo", "why": "support"}],
                            dropped=[{"id": "procedure/*/q", "component": "procedure", "reason": "quota"},
                                     {"id": "procedure/*/b", "component": "procedure", "reason": "budget"},
                                     {"id": "procedure/*/i", "component": "procedure", "reason": "irrelevant"},
                                     {"id": "procedure/*/d", "component": "procedure", "reason": "duplicate"}])
        per[p] = {"pres": p, "item": r["item"], "lang": C.parse_pres(p)["lang"], "inputs": {"snapshot_sha": G_SNAP, "rows_sha": G_ROWS},
                  "recomputed": {"pkg_sha": "k" + p, "text_sha": b["sha256"], "params_sha": "prm", "kept": list(b["ids"]),
                                 "dropped": copy.deepcopy(b["package"]["dropped"]), "leave_out": copy.deepcopy(b["package"]["leave_out"])},
                  "pool": [["places", "places/x"], ["procedure", "procedure/*/q"], ["procedure", "procedure/*/b"],
                           ["procedure", "procedure/*/i"], ["procedure", "procedure/*/d"], ["procedure", "procedure/*/r"]],
                  "rank_irrelevant": ["procedure/*/r"]}
    gd = {"schema": A.G_SCHEMA, "namespace": C.ACCEPTANCE_NS, "ruling": "g-eligible", "capture_sha": "cs", "snapshot_dir_sha_before": G_SNAP,
          "snapshot_dir_sha_after": G_SNAP, "token_cache_sha": "tc", "treatment_id": A.TREATMENT_SHA, "checker_sha": "chk",
          "presentations": len(per), "per_pres": per,
          "problems": [], "ok": True}
    gd["sha256"] = C.self_sha(gd)
    ev["g_eligible"] = gd
    ev["g_checker_sha"] = "chk"
    ev["g_inputs_now"] = {"snapshot_dir_sha": G_SNAP, "token_cache_sha": "tc"}
    return ev


def g_reseal(ev):
    ev["g_eligible"]["sha256"] = C.self_sha(ev["g_eligible"])


def g_reasons(ev):
    dd = A.decide(ev, REG, ROSTER)
    return dd, [k for k, v in dd["criteria"].items() if not v["pass"]], " | ".join(dd["criteria"]["g"]["reasons"])


dg, badg, _rg = g_reasons(g_synth())
check("7g.1 POSITIVE: ruling recorded + every presentation recomputed and equal, every drop recorded with a known reason -> (g) passes, "
      "verdict PASS", dg["verdict"] == "PASS" and badg == [], (badg, _rg[:300]))
ev = g_synth()
P1 = ev["capture"][1]["pres"]
ev["capture"][1]["blocks"]["S*"]["ids"] = []                               # the frozen rules deliver places/x; the block does not
_d, bad, rs = g_reasons(ev)
check("7g.2 NEGATIVE: a record the frozen rules select is neither delivered nor dropped -> (g) FAILS with MISSING eligible record",
      "g" in bad and set(bad) <= {"c", "g"} and f"{P1}: MISSING eligible record places/x" in rs, (bad, rs[:400]))
ev = g_synth()
ev["capture"][2]["blocks"]["S*"]["package"]["dropped"][0]["reason"] = "too_long"
ev["g_eligible"]["per_pres"][ev["capture"][2]["pres"]]["recomputed"]["dropped"][0]["reason"] = "too_long"
g_reseal(ev)
_d, bad, rs = g_reasons(ev)
check("7g.3 a FAKE drop reason (even one the recomputation repeats) -> (g) FAILS: unregistered reason, and the record counts as missing",
      bad == ["g"] and "unregistered reason 'too_long'" in rs and "MISSING eligible record procedure/*/q" in rs, (bad, rs[:400]))
ev = g_synth()
ev["capture"][3]["blocks"]["S*"]["package"]["dropped"] = [x for x in ev["capture"][3]["blocks"]["S*"]["package"]["dropped"] if x["reason"] != "budget"]
_d, bad, rs = g_reasons(ev)
check("7g.4 an UNRECORDED drop (the rules drop procedure/*/b for the budget; the block does not record it) -> (g) FAILS, also as MISSING",
      bad == ["g"] and "unrecorded drop: the frozen rules drop procedure/*/b" in rs and "MISSING eligible record procedure/*/b" in rs, (bad, rs[:400]))
ev = g_synth()
blk = ev["capture"][4]["blocks"]["S*"]
blk["ids"] = []
blk["package"]["dropped"].append({"id": "places/x", "component": "places", "reason": "quota"})
_d, bad, rs = g_reasons(ev)
check("7g.5 a recorded drop the frozen rules do not make (places/x 'quota' while the rules deliver it) -> (g) FAILS",
      "g" in bad and set(bad) <= {"c", "g"} and "recorded drop places/x (places, quota) is not what the frozen rules do" in rs, (bad, rs[:400]))
ev = g_synth()
ev["g_eligible"]["per_pres"][ev["capture"][5]["pres"]]["pool"].append(["example", "example/practice-kitchen/ghost"])
g_reseal(ev)
_d, bad, rs = g_reasons(ev)
check("7g.6 a store candidate that vanished without any recorded reason -> (g) FAILS", bad == ["g"] and "vanished without a recorded reason" in rs, (bad, rs[:300]))
ev = g_synth()
ev["capture"][6]["blocks"]["S*"]["package"]["leave_out"] = []
_d, bad, rs = g_reasons(ev)
check("7g.7 an unrecorded target leave-out -> (g) FAILS (unrecorded + missing)", bad == ["g"] and "unrecorded target leave-out of procedure/*/lo" in rs
      and "MISSING eligible record procedure/*/lo" in rs, (bad, rs[:300]))
ev = g_synth()
ev["capture"][7]["blocks"]["S*"]["package"]["leave_out"][0]["why"] = "felt_like_it"
ev["g_eligible"]["per_pres"][ev["capture"][7]["pres"]]["recomputed"]["leave_out"][0]["why"] = "felt_like_it"
g_reseal(ev)
_d, bad, rs = g_reasons(ev)
check("7g.8 a leave-out with an unregistered citation kind -> (g) FAILS", bad == ["g"] and "registered citation kind" in rs, (bad, rs[:300]))
for nm, mut, want in (
        ("7g.9 no g_eligible.json -> (g) FAILS", lambda e: e.update(g_eligible=None), "g_eligible.json missing"),
        ("7g.10 g_eligible.json of another capture journal -> (g) FAILS (stale)", lambda e: (e["g_eligible"].update(capture_sha="old"), g_reseal(e)), "another capture journal"),
        ("7g.11 g_eligible.json edited after writing -> (g) FAILS (self-sha)", lambda e: e["g_eligible"].update(ok=True, presentations=1), "self-sha"),
        ("7g.12 recomputed on another treatment -> (g) FAILS", lambda e: (e["g_eligible"].update(treatment_id="9b26a99f" * 8), g_reseal(e)), "recomputed on treatment"),
        ("7g.13 the snapshot changed during the recomputation -> (g) FAILS", lambda e: (e["g_eligible"].update(snapshot_dir_sha_after="x"), g_reseal(e)), "snapshot changed"),
        ("7g.14 written by another acc_runner.py -> (g) FAILS", lambda e: e.update(g_checker_sha="newer"), "another version of acc_runner.py"),
        ("7g.15 a presentation not recomputed -> (g) FAILS", lambda e: (e["g_eligible"]["per_pres"].pop(e["capture"][0]["pres"]), g_reseal(e)), "no recomputation of this presentation"),
        ("7g.16 a presentation whose recomputation errored -> (g) FAILS", lambda e: (e["g_eligible"]["per_pres"][e["capture"][0]["pres"]].update(error="GCacheMiss: x"), g_reseal(e)), "could not be recomputed"),
        ("7g.17 recomputed on other S* inputs -> (g) FAILS", lambda e: e["capture"][0]["blocks"]["S*"]["source"].update(rows_sha="other"), "other S* inputs"),
        ("7g.18 the file reports its own problems -> (g) FAILS", lambda e: (e["g_eligible"].update(ok=False, problems=["x"]), g_reseal(e)), "reported problems"),
        ("7g.19 a delivered record both delivered and dropped -> (g) FAILS", lambda e: e["capture"][0]["blocks"]["S*"]["package"]["dropped"].append(
            {"id": "places/x", "component": "places", "reason": "irrelevant"}), "both delivered and recorded as dropped"),
        ("7g.20 a delivered record drawn from no store and not builder-written -> (g) FAILS", lambda e: (
            e["g_eligible"]["per_pres"][e["capture"][0]["pres"]].update(pool=[x for x in e["g_eligible"]["per_pres"][e["capture"][0]["pres"]]["pool"] if x[1] != "places/x"]),
            g_reseal(e)), "drawn from no store"),
        # Astra ACC-G r1 P1: the delivered BYTES are hashed, not the stored field
        ("7g.38 r1: a block whose text was emptied (ids + stored sha kept) -> (g) FAILS", lambda e: e["capture"][0]["blocks"]["S*"].update(text=""),
         "stored text sha is not the sha of its delivered text"),
        ("7g.39 r1: a truncated block text (ids + stored sha kept) -> (g) FAILS", lambda e: e["capture"][0]["blocks"]["S*"].update(
            text=e["capture"][0]["blocks"]["S*"]["text"][:-5]), "stored text sha is not the sha of its delivered text"),
        ("7g.40 r1: truncated text with a matching re-computed stored sha -> (g) FAILS (differs from the recomputation)", lambda e: e["capture"][0]["blocks"]["S*"].update(
            text="Package", sha256=C.sha256_text("Package")), "delivered text differs from the recomputation"),
        # Astra ACC-G r1 P2: bindings to inputs observed on disk now
        ("7g.41 r1: the token cache changed since the recomputation -> (g) FAILS", lambda e: e["g_inputs_now"].update(token_cache_sha="tc2"), "token cache is missing or changed"),
        ("7g.42 r1: a g file without a token-cache sha -> (g) FAILS", lambda e: (e["g_eligible"].update(token_cache_sha=None), g_reseal(e)), "token cache is missing or changed"),
        ("7g.43 r1: both snapshot fields rewritten (self-sha recomputed) -> (g) FAILS against the snapshot on disk", lambda e: (
            e["g_eligible"].update(snapshot_dir_sha_before="wrong", snapshot_dir_sha_after="wrong"), g_reseal(e)), "another S* snapshot than the one on disk now"),
        ("7g.44 r1: the current inputs were not observed -> (g) FAILS", lambda e: e.update(g_inputs_now=None), "were not observed"),
        ("7g.45 r1: a presentation recomputed on another snapshot than the file's own -> (g) FAILS", lambda e: (
            e["g_eligible"]["per_pres"][e["capture"][3]["pres"]]["inputs"].update(snapshot_sha="other"), g_reseal(e)), "recomputed on another snapshot than the file's own")):
    ev = g_synth()
    mut(ev)
    _d, bad, rs = g_reasons(ev)
    check(nm, bad == ["g"] and want in rs, (bad, rs[:300]))
ev = g_synth()
e0 = ev["g_eligible"]["per_pres"][ev["capture"][0]["pres"]]
e0["recomputed"]["kept"] = ["procedure/*/request-types", "briefing/FloorPlan1/fixtures"] + e0["recomputed"]["kept"]
ev["capture"][0]["blocks"]["S*"]["ids"] = list(e0["recomputed"]["kept"])
ev["capture"][0]["blocks"]["S*"]["package"]["provenance"] += [{"id": "procedure/*/request-types", "source": "parser:goal_schema", "authored_by": "script:ltm_package.request_line"},
                                                              {"id": "briefing/FloorPlan1/fixtures", "source": "briefing:" + INV["briefing16"],
                                                               "authored_by": "script:kitchen_briefing.natural_fixture_line"}]
g_reseal(ev)
_d, bad, rs = g_reasons(ev)
check("7g.21 builder-written records (request line, kitchen line) are delivered without a store candidate and pass", "g" not in bad, (bad, rs[:300]))
# the reason vocabulary is the frozen code's own
src_lp = open(LP_.__file__, encoding="utf-8").read()
lit = set(re_.findall(r'"reason":\s*"([^"]+)"', src_lp)) | set(re_.findall(r'drop\(c,\s*"[a-z]+",\s*"[^"]+",\s*"([^"]+)"\)', src_lp))
known = set(A.G_REGISTERED_DROPS) | set(A.G_NOT_SELECTED)
check("7g.22 every drop-reason literal in ltm_package.py is a registered or a not-selected reason (+ D2 / NEED reasons by name)",
      lit and lit <= known and TS_.REASON in A.G_NOT_SELECTED and "need_rule:target_type_hint" in src_lp and "irrelevant" in src_lp, sorted(lit - known))
check("7g.23 leave-out components and citation kinds are the treatment's own (ItemLeaveOut / _cites)",
      tuple(A.G_LEAVE_OUT_COMPONENTS) == tuple(T_.LEAVE_OUT_COMPONENTS)
      and [T_._cites(pv, "g-x") for pv in ({"support": ["g-x"]}, {"tuned_on": ["g-x"]}, {"evidence": ["g-x"]})] == list(A.G_LEAVE_OUT_WHY))
check("7g.24 the registered drops are exactly quota + budget (the third registered reason, the target leave-out, has its own log)",
      sorted(A.G_REGISTERED_DROPS) == ["budget", "quota"])
# the cache-only counter never calls out
TG = tempfile.mkdtemp(prefix="accg_")
try:
    cp_ = os.path.join(TG, "tokcache.json")
    json.dump({}, open(cp_, "w", encoding="utf-8"))
    cc = A.CacheOnlyCounter(cp_, "H")
    check("7g.25 the cache-only counter raises on a text the capture never counted (no Ollama / Anthropic call)",
          raises(lambda: cc.counts("some text"), A.GCacheMiss) and cc.misses >= 1 and cc.counts("  ") == {"gemma": 1, "anthropic": 1})
    src_ar = open(A.__file__, encoding="utf-8").read()
    body = src_ar[src_ar.index("class CacheOnlyCounter"): src_ar.index("def _g_recording")]
    check("7g.26 the cache-only counter holds no network code", "urllib" not in body and "http" not in body.lower() and "requests" not in body)
    # REAL inputs, read-only: rehearsal-v3 (28 ft-v2 run-1 presentations, the same S* snapshot, its capture's own token cache)
    r3_cache = os.path.join(C.ns_work("rehearsal-v3"), "tokcache.json")
    r3_cap = C.journal_path("rehearsal-v3", "capture")
    snap_d = T_.source_dir("rehearsal-v3", "S*")
    s_before, cap_before, cache_before = C.dir_sha(snap_d), A.file_sha(r3_cap), A.file_sha(r3_cache) if os.path.exists(r3_cache) else None
    check("7g.27 rehearsal-v3's token cache exists (machine-local, gitignored work dir; needed for the real recomputation)", cache_before is not None, r3_cache)
    G3 = A.g_recompute("rehearsal-v3", identity=False)
    check("7g.28 REAL: all 28 rehearsal-v3 S* packages recompute from the frozen inputs to the captured bytes; 0 problems, 0 cache misses",
          G3.get("ok") is True and G3.get("presentations") == 28 and G3.get("problems") == [] and G3["snapshot_dir_sha_before"] == G3["snapshot_dir_sha_after"],
          (G3.get("presentations"), (G3.get("problems") or [])[:3]))
    check("7g.29 REAL: the recomputation leaves the S* snapshot, the capture journal and the token cache byte-identical",
          (C.dir_sha(snap_d), A.file_sha(r3_cap), A.file_sha(r3_cache)) == (s_before, cap_before, cache_before))
    r3 = A._g_capture_rows(A._rows(r3_cap))
    tot = {"quota": 0, "budget": 0, "leave_out": 0, "pool": 0, "rank": 0}
    for p, r in r3.items():
        pk_ = r["blocks"]["S*"]["package"]
        tot["quota"] += sum(1 for x in pk_["dropped"] if x["reason"] == "quota")
        tot["budget"] += sum(1 for x in pk_["dropped"] if x["reason"] == "budget")
        tot["leave_out"] += len(pk_["leave_out"])
        tot["pool"] += len(G3["per_pres"][p]["pool"])
    check("7g.30 REAL: the rehearsal-v3 packages exercise every registered reason (quota, budget, target leave-out) and a non-trivial pool",
          tot["quota"] > 0 and tot["budget"] > 0 and tot["leave_out"] > 0 and tot["pool"] > 100, tot)
    # real mutations through the pure check
    pq = next(p for p, r in sorted(r3.items()) if any(x["reason"] == "quota" for x in r["blocks"]["S*"]["package"]["dropped"]))
    pb = next(p for p, r in sorted(r3.items()) if any(x["reason"] == "budget" for x in r["blocks"]["S*"]["package"]["dropped"]))
    pl = next(p for p, r in sorted(r3.items()) if r["blocks"]["S*"]["package"]["leave_out"])
    check("7g.31 REAL positive: g_problems is empty on every real presentation", all(A.g_problems(r["blocks"]["S*"], G3["per_pres"][p]) == [] for p, r in r3.items()))
    b_ = copy.deepcopy(r3[pq]["blocks"]["S*"])
    qid = next(x["id"] for x in b_["package"]["dropped"] if x["reason"] == "quota")
    b_["package"]["dropped"] = [x for x in b_["package"]["dropped"] if x["id"] != qid]
    pr_ = A.g_problems(b_, G3["per_pres"][pq])
    check("7g.32 REAL unrecorded drop: a quota drop erased from a real block -> unrecorded + MISSING", any("unrecorded drop" in x and qid in x for x in pr_)
          and any(x.startswith("MISSING eligible record " + qid) for x in pr_), pr_)
    b_ = copy.deepcopy(r3[pb]["blocks"]["S*"])
    for x in b_["package"]["dropped"]:
        if x["reason"] == "budget":
            x["reason"] = "space"
    pr_ = A.g_problems(b_, G3["per_pres"][pb])
    check("7g.33 REAL fake drop reason: a real budget drop relabelled 'space' -> unregistered reason + not what the rules do + MISSING",
          any("unregistered reason 'space'" in x for x in pr_) and any("not what the frozen rules do" in x for x in pr_)
          and any(x.startswith("MISSING eligible record") for x in pr_), pr_)
    b_ = copy.deepcopy(r3[pl]["blocks"]["S*"])
    lid = b_["package"]["leave_out"][0]["id"]
    b_["package"]["leave_out"] = b_["package"]["leave_out"][1:]
    pr_ = A.g_problems(b_, G3["per_pres"][pl])
    check("7g.34 REAL unrecorded leave-out -> unrecorded + MISSING", any("unrecorded target leave-out of " + lid in x for x in pr_), pr_)
    b_ = copy.deepcopy(r3[pq]["blocks"]["S*"])
    gone_id = b_["ids"][0]
    b_["ids"] = b_["ids"][1:]
    pr_ = A.g_problems(b_, G3["per_pres"][pq])
    check("7g.35 REAL negative: a delivered record removed from a real block -> MISSING eligible record", any(x.startswith("MISSING eligible record " + gone_id) for x in pr_), pr_)
    e_ = copy.deepcopy(G3["per_pres"][pq])
    e_["pool"].append(["procedure", "procedure/*/ghost"])
    check("7g.36 REAL: a candidate the builder received but never accounted for -> vanished", any("vanished" in x for x in A.g_problems(r3[pq]["blocks"]["S*"], e_)))
    ev_r = A.evidence()
    check("7g.46 r1: evidence() observes the S* snapshot and the token cache on disk now (acceptance-v2 has no cache yet = None)",
          isinstance(ev_r.get("g_inputs_now"), dict) and ev_r["g_inputs_now"]["snapshot_dir_sha"] == C.dir_sha(T_.source_dir(C.ACCEPTANCE_NS, "S*"))
          and ev_r["g_inputs_now"]["snapshot_dir_sha"] == s_before, ev_r.get("g_inputs_now"))
    import glob as glob_
    tmp_before = set(glob_.glob(os.path.join(tempfile.gettempdir(), "pkgdb_*")) + glob_.glob(os.path.join(tempfile.gettempdir(), "accg_*")))
    G3b = A.g_recompute("rehearsal-v3", identity=False)
    check("7g.47 r1: the recomputation leaves no temp copy behind (the builder's db copy goes to a private dir, removed) and is repeatable",
          set(glob_.glob(os.path.join(tempfile.gettempdir(), "pkgdb_*")) + glob_.glob(os.path.join(tempfile.gettempdir(), "accg_*"))) - tmp_before == set()
          and G3b["per_pres"] == G3["per_pres"])
    GM = A.g_recompute("rehearsal-v3", identity=False, counter=A.CacheOnlyCounter(cp_, "H"))
    check("7g.37 REAL with an empty token cache: every presentation fails closed (recomputation error), the file is not ok",
          GM.get("ok") is False and all(v.get("error") for v in GM["per_pres"].values()) and C.dir_sha(snap_d) == s_before, (GM.get("problems") or [])[:2])
finally:
    shutil.rmtree(TG, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ 8 C11 scanner = the C11-AND-REG audit on rehearsal-v3
print("== 8 C11 delivered-bytes scan (rehearsal-v3, read-only)")
c11 = A.c11_scan("rehearsal-v3")
ref = json.load(open(os.path.join(ROOT, "sim", "final_test", "diag", "c11", "c11_audit.json"), encoding="utf-8"))
rc = ref["coverage"]["requests"]
check("8.1 every captured request re-renders to its sha (as the C11 audit)", c11["requests"] == rc["requests"] == c11["bound"] == rc["bound"], (c11["requests"], c11["bound"], rc["requests"]))
check("8.2 repair turns bound as in the C11 audit", c11["repair_turns"] == ref["coverage"]["repairs"]["turns"] == c11["repair_bound"])
check("8.3 every gold hit carries the R-G1 disposition; 0 unresolved; same hit count as the audit",
      not c11["unresolved_gold"] and len(c11["rg1_dispositioned"]) == rc["dispositioned_R_G1"], (len(c11["rg1_dispositioned"]), rc["dispositioned_R_G1"]))
check("8.4 the scan is clean on rehearsal-v3 and bound to the same forbidden set", c11["ok"] and c11["forbidden_set_sha"] == ref["binding"]["forbidden_set_sha"], c11["problems"])
# DRESS-REHEARSAL (Astra r1 4): deterministic -- the capture journal path points at a guaranteed MISSING fixture (the voided attempt's
# 11 real acceptance-v2 rows stay untouched as evidence); restored afterwards
_jp85 = C.journal_path
_miss85 = os.path.join(tempfile.mkdtemp(prefix="c11_85_"), "missing", "capture.jsonl")
try:
    C.journal_path = lambda ns, kind, root=None: _miss85 if kind == "capture" else _jp85(ns, kind, root)
    _r85 = A.c11_scan(C.ACCEPTANCE_NS)
finally:
    C.journal_path = _jp85
check("8.5 the scan without a capture journal is NOT ok ('no capture journal')", _r85["ok"] is False
      and any("no capture journal" in x for x in _r85["problems"]) and not os.path.exists(_miss85), _r85.get("problems"))
check("8.6 every first transmission (repaired or not) is bound to its captured request; the system text is scanned too",
      c11["first_transmissions"] == c11["first_bound"] == len([1 for x in open(C.journal_path("rehearsal-v3", "plan"), encoding="utf-8") if x.strip()])
      and c11["system_text_hits"] == [], (c11["first_transmissions"], c11["first_bound"]))

# ------------------------------------------------------------------------------------------------ 9 cycle 3: one test per Astra r2 probe
print("== 9 cycle 3 (Astra ACC-RUNNER r2 probes)")
import threading                                          # noqa: E402
import plan as PLN                                        # noqa: E402
import plan_batch as PB                                   # noqa: E402
import gate_evidence as GE                                # noqa: E402
import freeze_ft2 as FZ                                   # noqa: E402
import freeze as FF                                       # noqa: E402

# ---- Q1: ONE submission per chunk, registered order, both scopes held first
TMP = tempfile.mkdtemp(prefix="acc_c3_")
try:
    BP = A.plan_class(True)
    check("9.1 the acceptance Batch class uses the one-submission chunk (MRO), not plan_batch's scope-separated one",
          BP._run_chunk is A._AccBatch._run_chunk and BP.__mro__.index(A._AccBatch) < BP.__mro__.index(PB.BatchPlan))
    p0, p1 = A.main_order(ROSTER)[:2]

    def cells_of(p):
        return [{"key": f"{a}|{p}", "pres": p, "arm": a, "planner": "O", "scope": "other" if a == "O+FS" else "core"}
                for a in [x for x in A.arm_order(p) if x[0] == "O"]]
    chunk = [(p0, cells_of(p0)), (p1, cells_of(p1))]

    class FakeDriver:
        def __init__(self, led):
            self.calls, self.led, self.inside = [], led, {}

        def run(self, units, tag=""):
            self.calls.append([u["id"] for u in units])
            with self.led._lock, self.led._conn() as c:
                self.inside["open_holds"] = sorted(r[0] for r in c.execute("SELECT provider FROM ledger WHERE state='open' AND provider LIKE 'hold@%'"))
            w = {}
            for u in units:
                w[u["scope"]] = w.get(u["scope"], 0) + 0.3
            ok = []
            for sc, usd in sorted(w.items()):                # each scope reserves its WHOLE worst case inside its own hold
                with self.led.use(sc):
                    ok.append(self.led.reserve("anthropic", usd) > 0)
            self.inside["reserves_ok"] = all(ok)

    def fake_plan(cap_usd):
        lp = os.path.join(TMP, "q1-%s" % str(cap_usd).replace(".", "_"), C.ACCEPTANCE_NS, "ledger.db")
        os.makedirs(os.path.dirname(lp))
        AC.register_ledger_guard(lp, attempt_base_usd=0.0, attempt_cap_usd=cap_usd, lock_path=os.path.join(TMP, "q1.lock"),
                                 other_spend=lambda: {"settled": 0.0, "open": 0.0, "committed": 0.0, "errors": [], "ledgers": {}})
        x = A._AccBatch.__new__(type("FakeAccBatch", (A._AccBatch,), {}))
        x.ledger = C.ScopedLedger(lp, C.NAMESPACES[C.ACCEPTANCE_NS]["ceilings"])
        x.ledger.__class__ = AC.AccScopedLedger
        x.driver, x.pass_id, x.cap, x.evs = FakeDriver(x.ledger), "", {}, []
        x._worst = lambda cs: round(0.3 * len(cs), 6)
        x._runner = lambda planner: ("prov",)
        x._call_cell = lambda c, cap: None
        x.ev = lambda event, **kw: x.evs.append(event)
        return x, lp
    x, lp = fake_plan(8.0)
    x._run_chunk(chunk, 0)
    want_ids = [c["key"] for _, cs in chunk for c in cs]
    check("9.2 Astra r2 Q1 probe: two presentations -> ONE submission, cells in the registered order (main_order x arm_order)",
          x.driver.calls == [want_ids], x.driver.calls)
    import sqlite3 as _sq3
    _c = _sq3.connect(lp)
    n_open_holds = _c.execute("SELECT COUNT(*) FROM ledger WHERE state='open' AND provider LIKE 'hold@%'").fetchone()[0]
    _c.close()
    check("9.3 both scopes are held BEFORE the submission and released after it", x.driver.inside.get("open_holds") == ["hold@core", "hold@other"]
          and n_open_holds == 0, (x.driver.inside, n_open_holds))
    AC.unregister_ledger_guard(lp)
    worst_all = round(0.3 * len(want_ids), 6)
    x, lp = fake_plan(worst_all)
    x._run_chunk(chunk, 0)
    check("9.4 inside the one submission every scope reserves its whole worst case: the attempt cap = the summed hold suffices (no double counting)",
          x.driver.inside.get("reserves_ok") is True, x.driver.inside)
    AC.unregister_ledger_guard(lp)
    x, lp = fake_plan(worst_all - 0.01)
    check("9.5 a hold over the attempt cap stops BEFORE anything is submitted", raises(lambda: x._run_chunk(chunk, 0), PLN._Stop) and x.driver.calls == []
          and "block_refused" in x.evs)
    AC.unregister_ledger_guard(lp)

    class FakeBase:
        _run_chunk = PB.BatchPlan._run_chunk
    y = FakeBase()
    os.makedirs(os.path.join(TMP, "base"), exist_ok=True)
    y.ledger = C.ScopedLedger(os.path.join(TMP, "base", "ledger.db"), {"total": 16.0})
    y.driver, y.pass_id, y.cap = FakeDriver(y.ledger), "", {}
    y._worst, y._runner, y._call_cell, y.ev = (lambda cs: 0.1), (lambda planner: ("prov",)), (lambda c, cap: None), (lambda *a, **k: None)
    y.driver.run = (lambda units, tag="", d=y.driver: d.calls.append([u["id"] for u in units]))
    y._run_chunk(chunk, 0)
    check("9.6 (reproduction) the inherited plan_batch chunk WOULD submit core then O+FS separately: the acceptance class overrides it",
          len(y.driver.calls) == 2 and y.driver.calls[0] + y.driver.calls[1] != want_ids)

    # ---- ACC-V2-R1 (2026-10-15): chunks are sized LAZILY to the room left (hold - committed), as A29 / batch_submissions_forecast say
    class _FakeLed:
        def __init__(self):
            self.v = 0.0

        def committed(self):
            return self.v
    z = A._AccBatch.__new__(type("FakeAccBatchZ", (A._AccBatch,), {}))
    zp = A.main_order(ROSTER)[:6]
    z._order = lambda: [(p, cells_of(p)) for p in zp]
    z.latest, z.requeue, z._quarantined, z._quarantine_skipped, z.pass_id = {}, set(), set(), 0, ""
    z.batch_hold_usd, z.batch_max_cells, z.ledger = 2.0, 999, _FakeLed()
    z._worst = lambda cs: round(0.3 * len(cs), 6)
    _cls0 = PLN.classify
    PLN.classify = lambda c, latest, requeue: "pending"
    try:
        sizes = []
        for ch in z._chunks():
            sizes.append(len(ch))
            z.ledger.v = round(z.ledger.v + 0.5, 6)              # each settled chunk commits 0.5
        zb = type("FakeBaseZ", (), {"_chunks": PB.BatchPlan._chunks})()
        zb.__dict__.update({k: getattr(z, k) for k in ("_order", "latest", "requeue", "_quarantined", "_quarantine_skipped", "pass_id",
                                                        "batch_hold_usd", "batch_max_cells", "_worst")})
        base_sizes = [len(ch) for ch in zb._chunks()]
    finally:
        PLN.classify = _cls0
    w_pres = 0.3 * len(cells_of(zp[0]))
    check("9.7 (ACC-V2-R1) every chunk is sized to the room left after the settled spend (hold 2.0, +0.5 per chunk): sizes shrink, all "
          "presentations covered, registered order kept", sum(sizes) == 6 and sizes[0] == int(2.0 // w_pres) and sizes[1] == int(1.5 // w_pres)
          and base_sizes == [int(2.0 // w_pres)] * (6 // int(2.0 // w_pres)), (sizes, base_sizes, w_pres))

    # ---- Q2: the study-wide reservation boundary, ledger identity, sqlite errors, hold reconciliation
    lockq = os.path.join(TMP, "q2.lock")
    gp = os.path.join(TMP, "q2", C.ACCEPTANCE_NS, "ledger.db")
    os.makedirs(os.path.dirname(gp))
    AC.register_ledger_guard(gp, attempt_base_usd=0.0, attempt_cap_usd=8.0, lock_path=lockq,
                             other_spend=lambda: {"settled": 0.0, "open": 0.0, "committed": 0.0, "errors": [], "ledgers": {}})
    lg = C.ScopedLedger(gp, C.NAMESPACES[C.ACCEPTANCE_NS]["ceilings"])
    w0 = AC.STUDY_LOCK_WAIT_S
    AC.STUDY_LOCK_WAIT_S = 0.2
    try:
        open(lockq, "w").write("{}")
        with lg.use("core"):
            busy = raises(lambda: lg.reserve("anthropic", 0.01), AC.StudyBoundaryBusy)
        busy_b = raises(lambda: lg.block("core", 0.5).__enter__(), C.BlockRefused)
        os.remove(lockq)
    finally:
        AC.STUDY_LOCK_WAIT_S = w0
    check("9.7 a submission while another holds the study-wide boundary is refused and inserts nothing (reserve and block hold)",
          busy and busy_b and lg.committed() == 0.0)
    with lg._lock, lg._conn() as c:
        check("9.8 a guard check outside the boundary refuses (the check and the insert are one atomic step)", bool(AC.guard_problems(lg, c, 0.01)))
    with lg.use("core"):
        rid_ = lg.reserve("anthropic", 0.01)
    check("9.9 inside the boundary the reservation goes through and the lock is released afterwards", rid_ > 0 and not os.path.exists(lockq))
    # ---- ACC-RUNNER-D (Astra r3 NEW defect 1): no age-based takeover; release only our own token
    foreign = json.dumps({"pid": 999999, "host": "elsewhere", "ts": 0, "token": "f" * 32})
    open(lockq, "w").write(foreign)
    old_t = __import__("time").time() - 10 * 24 * 3600                  # ten days old: age proves nothing about its holder
    os.utime(lockq, (old_t, old_t))
    n0 = lg.committed()
    AC.STUDY_LOCK_WAIT_S = 0.2
    try:
        with lg.use("core"):
            old_busy = raises(lambda: lg.reserve("anthropic", 0.01), AC.StudyBoundaryBusy)
    finally:
        AC.STUDY_LOCK_WAIT_S = w0
    check("9.10 Astra r3 probe: a ten-day-old lock is NOT taken over: the submission is refused, nothing inserted, the lock is untouched",
          old_busy and lg.committed() == n0 and os.path.exists(lockq) and open(lockq).read() == foreign)
    removed = []
    orig_rm = AC.os.remove
    AC.os.remove = lambda q, *a_, **k_: (removed.append(q), orig_rm(q, *a_, **k_))[1]
    seq = iter([0.0, 0.0, 0.1, 0.2, 0.3, 99.0, 99.0, 99.0])
    try:
        race_busy = raises(lambda: AC._lock_acquire(lockq, 1.0, now=lambda: next(seq), sleep=lambda _s: None), AC.StudyBoundaryBusy)
    finally:
        AC.os.remove = orig_rm
    check("9.10b Astra r3 interleaving: a contender waiting on a held lock never unlinks it (no remove call; it fails closed)",
          race_busy and removed == [] and open(lockq).read() == foreign, removed)
    tok = AC._lock_acquire(lockq + ".own", 0.2)
    h_own = AC.lock_holder(lockq + ".own")
    check("9.10c an acquisition writes a random token + pid / host and release removes the lock only with that token",
          isinstance(tok, str) and len(tok) == 32 and h_own.get("token") == tok and h_own.get("pid") == os.getpid()
          and not AC._lock_release(lockq + ".own", "0" * 32) and os.path.exists(lockq + ".own")
          and AC._lock_release(lockq + ".own", tok) and not os.path.exists(lockq + ".own"))
    os.remove(lockq)
    with lg.use("core"):
        with AC.study_boundary(lg):
            open(lockq, "w").write(foreign)                              # someone replaced our lock while we held it
    check("9.10d a lock that no longer carries our token is left in place on release (never delete another holder's lock)",
          os.path.exists(lockq) and open(lockq).read() == foreign)
    os.remove(lockq)
    import io as _io
    import contextlib as _cl
    buf = _io.StringIO()
    rm_ls = []
    AC.os.remove = lambda q, *a_, **k_: rm_ls.append(q)
    try:
        with _cl.redirect_stdout(buf):
            rc_ls = A.main(["lock-status"])
    finally:
        AC.os.remove = orig_rm
    check("9.10e lock-status is read-only (prints the lock path and holder, removes nothing)", rc_ls == 0 and rm_ls == [] and '"holder"' in buf.getvalue())
    check("9.10f the age-based takeover constant is gone from acc_common (no stale-lock path left)", not hasattr(AC, "STUDY_LOCK_STALE_S")
          and "getmtime" not in __import__("inspect").getsource(AC._lock_acquire))
    AC.unregister_ledger_guard(gp)
    # two guarded ledgers racing for the same study headroom: exactly one is approved
    sd2 = os.path.join(TMP, "race")
    for sub in ("sim/a", "sim/b", "docs/final-test"):
        os.makedirs(os.path.join(sd2, *sub.split("/")))
    shutil.copy(os.path.join(ROOT, "docs", "final-test", "split_manifest.json"), os.path.join(sd2, "docs", "final-test"))
    pa, pb = os.path.join(sd2, "sim", "a", "ledger.db"), os.path.join(sd2, "sim", "b", "ledger.db")
    for pth in (pa, pb):
        AC.register_ledger_guard(pth, attempt_base_usd=0.0, attempt_cap_usd=8.0, study_cap_usd=1.0, lock_path=os.path.join(TMP, "race.lock"),
                                 other_spend=(lambda pth=pth: AC.study_spend(sd2, exclude=(pth,))))
    la, lb_ = C.ScopedLedger(pa, {"total": 5.0}), C.ScopedLedger(pb, {"total": 5.0})
    res, bar = {}, threading.Barrier(2)

    def go(name, led):
        bar.wait()
        try:
            with led.use("core"):
                res[name] = led.reserve("anthropic", 0.6) > 0
        except C.CostCeilingExceeded:
            res[name] = False
    ts = [threading.Thread(target=go, args=(n_, l_)) for n_, l_ in (("a", la), ("b", lb_))]
    [t_.start() for t_ in ts]
    [t_.join() for t_ in ts]
    check("9.11 two concurrent submissions on two ledgers (study cap 1.0, 0.6 each): exactly ONE is approved", sorted(res.values()) == [False, True], res)
    for pth in (pa, pb):
        AC.unregister_ledger_guard(pth)
    sd3 = os.path.join(TMP, "ident")
    for sub in ("sim/a", "sim/b", "docs/final-test"):
        os.makedirs(os.path.join(sd3, *sub.split("/")))
    shutil.copy(os.path.join(ROOT, "docs", "final-test", "split_manifest.json"), os.path.join(sd3, "docs", "final-test"))
    for sub in ("a", "b"):
        lx = C.ScopedLedger(os.path.join(sd3, "sim", sub, "ledger.db"), {"core": 5.0, "total": 5.0})
        with lx.use("core"):
            lx.reserve("anthropic", 0.2)
    sp3 = AC.study_spend(sd3)
    check("9.12 Astra r2 Q2 probe: two independent ledgers with identical id=1 reservations of 0.20 count 0.40 (no merge without a ledger identity)",
          abs(sp3["open"] - 0.4) < 1e-9 and abs(sp3["committed"] - 0.4) < 1e-9, sp3)
    orig_conn = AC.sqlite3.connect

    class _BadDB:
        def execute(self, *a, **k):
            raise AC.sqlite3.DatabaseError("database disk image is malformed")

        def close(self):
            pass
    AC.sqlite3.connect = lambda *a, **k: _BadDB()
    try:
        sp4 = AC.study_spend(sd3)
    finally:
        AC.sqlite3.connect = orig_conn
    check("9.13 Astra r2 Q2 probe: a database error on sqlite_master is an accounting error and the study-wide check refuses (never 0 spend)",
          len(sp4["errors"]) == 2 and sp4["committed"] == 0.0 and bool(AC.study_wide_problems(0.01, spend=sp4)), sp4)
    hp = os.path.join(sd3, "sim", "a", "ledger.db")
    lh = C.ScopedLedger(hp, {"core": 5.0, "total": 5.0}, release_stale=False)
    with lh.block("core", 1.0):
        lh.reserve("anthropic", 0.6)
        sp5 = AC.study_spend(sd3)
    check("9.14 Astra r2 Q2 probe: an open hold and the reservations made inside it count ONCE (max), not hold + reservations",
          abs(sp5["ledgers"]["sim/a/ledger.db"]["liability"] - 1.2) < 1e-9 and abs(sp5["committed"] - 1.4) < 1e-9, sp5["ledgers"])
    rr = [(1, "hold@core", "open", 1.0, None), (2, "anthropic@core", "settled", 0.6, 0.3), (3, "anthropic@core", "open", 0.4, None)]
    check("9.15 reconciliation: rows above their hold count in full; two open holds of one scope are not reconciled (over-count)",
          AC.reconcile_rows(rr)["liability"] == 1.0 and AC.reconcile_rows(rr + [(4, "anthropic@core", "open", 0.5, None)])["liability"] == 1.2
          and AC.reconcile_rows(rr + [(5, "hold@core", "open", 1.0, None)])["liability"] == 2.7
          and AC.reconcile_rows(rr, pending=("reserve", 1, 0.2))["liability"] == 1.0 and AC.reconcile_rows(rr, pending=("reserve", None, 0.2))["liability"] == 1.2)
    sp_ = os.path.join(TMP, "base", C.ACCEPTANCE_NS, "ledger.db")
    os.makedirs(os.path.dirname(sp_))
    ls_ = C.ScopedLedger(sp_, {"total": 16.0})
    with ls_._lock, ls_._conn() as c:
        c.execute("INSERT INTO ledger (ts, provider, state, reserved) VALUES ('t', 'hold@core', 'open', 5.0)")
        c.execute("INSERT INTO ledger (ts, provider, state, reserved, actual) VALUES ('t', 'anthropic@core', 'settled', 1.0, 0.5)")
    check("9.16 the attempt base excludes a stale hold (a hold is not spend; it would otherwise widen the attempt ceiling)", AC.attempt_base_of(sp_) == 0.5)

    # ---- Q4: source-record resolution, attempt evidence, a bound context fit, the strict first-attempt hash
    def m_src(ev):
        ev["capture"][0]["blocks"]["S*"]["package"]["provenance"][0]["source"] = "kb:deadbeefdeadbeef"

    def m_claim(ev):
        ev["capture"][1]["blocks"]["S*"]["package"]["provenance"][0]["evidence"] = ["lc:0000000000000000"]

    def m_fsid(ev):
        ev["capture"][2]["blocks"]["FS"]["fs"]["retained"] = ["example/practice-kitchen/practice-v2:o0:a99"]

    def m_inv(ev):
        ev["sources"] = None

    def m_noatt(ev):
        ev["plan"][3]["attempts"] = []

    def m_atthash(ev):
        ev["plan"][4]["attempts"] = [{"prompt_sha256": "something else"}]

    def m_cfreq(ev):
        ev["context_fit"]["checks"][0]["request_sha"] = "stale"

    def m_cfcap(ev):
        ev["context_fit"]["capture_sha"] = "an older capture journal"

    def m_cfnone(ev):
        ev["context_fit"].pop("capture_sha")
    for name, mut, want in (("9.17 Astra r2 Q4 probe: a nonexistent package source -> (c)", m_src, {"c"}),
                            ("9.18 a location claim that is not in the S* store -> (c)", m_claim, {"c"}),
                            ("9.19 Astra r2 Q4 probe: a nonexistent FS attempt id -> (c)", m_fsid, {"c"}),
                            ("9.20 no source inventory -> (c) (nothing resolves)", m_inv, {"c"}),
                            ("9.21 Astra r2 Q4 probe: a plan with no attempt evidence -> completeness", m_noatt, {"completeness"}),
                            ("9.22 a first attempt whose prompt hash is not the request sha -> completeness", m_atthash, {"completeness"}),
                            ("9.23 Astra r2 Q4 probe: a context check bound to other request bytes -> (i)", m_cfreq, {"i"}),
                            ("9.24 a context report of another capture journal -> (i)", m_cfcap, {"i"}),
                            ("9.25 a context report without its capture binding -> (i)", m_cfnone, {"i"})):
        ok_, bad_ = fails(mut, want)
        check(name, ok_, bad_)
    import inspect as _ins
    check("9.26 c11_scan: a missing first-attempt prompt hash is unbound (no '(None, want)' allowance)",
          'at0.get("prompt_sha256") == want' in _ins.getsource(A.c11_scan) and "(None, want)" not in _ins.getsource(A.c11_scan))
    inv_real = A.source_inventory()
    rows_rv3 = [json.loads(x) for x in open(C.journal_path("rehearsal-v3", "capture"), encoding="utf-8") if x.strip()]
    unres = []
    for r_ in rows_rv3:
        b_ = (r_.get("blocks") or {}).get("S*") or {}
        pv = {x_["id"]: x_ for x_ in (b_.get("package") or {}).get("provenance") or []}
        unres += [rid for rid in b_.get("ids") or [] if A.provenance_problem(pv.get(rid), inv_real)]
        unres += [rid for rid in (((r_.get("blocks") or {}).get("FS") or {}).get("fs") or {}).get("retained") or [] if rid not in set(inv_real["fs_examples"])]
    check("9.27 positive control: every S* / FS record of the 56 rehearsal-v3 captures resolves to the frozen inputs", not unres and len(rows_rv3) == 56, unres[:3])
    # ---- ACC-RUNNER-D (Astra r3 Q4): a kb: source resolves the cited KB RECORD and its evidence, not the KB digest alone
    kb_src = "kb:" + inv_real["kb16"]
    real_kb = sorted(inv_real["kb_records"])
    proc_id = next(i_ for i_ in real_kb if "afford-" not in i_)
    aff_id = next(i_ for i_ in real_kb if i_.endswith("-request"))
    auth_ = "script:ltm_procedural@x"
    p_fake = A.provenance_problem({"id": "procedure/*/NOT-A-RECORD", "component": "procedure", "source": kb_src, "evidence": ["proc:NOT-A-RECORD"],
                                   "authored_by": auth_}, inv_real)
    p_ev = A.provenance_problem({"id": proc_id, "component": "procedure", "source": kb_src, "evidence": ["proc:does-not-exist"], "authored_by": auth_}, inv_real)
    p_extra = A.provenance_problem({"id": proc_id, "component": "procedure", "source": kb_src,
                                    "evidence": inv_real["kb_records"][proc_id] + ["afford:invented"], "authored_by": auth_}, inv_real)
    p_ok = A.provenance_problem({"id": proc_id, "component": "procedure", "source": kb_src,
                                 "evidence": inv_real["kb_records"][proc_id] + ["precondition_closure:open", "rewrite:holding"], "authored_by": auth_}, inv_real)
    p_aff = A.provenance_problem({"id": aff_id, "component": "procedure", "source": kb_src, "evidence": inv_real["kb_records"][aff_id],
                                  "authored_by": auth_}, inv_real)
    check("9.27b Astra r3 probe: procedure/*/NOT-A-RECORD with the genuine KB digest does NOT resolve", bool(p_fake), p_fake)
    check("9.27c a real KB record id with evidence the KB does not hold, or an unregistered extra evidence, does not resolve", bool(p_ev) and bool(p_extra))
    check("9.27d positive controls: a KB entry record with the package's closure / holding additions, and a generated tool-request record, resolve",
          p_ok is None and p_aff is None, (p_ok, p_aff))
    n_kb = sum(1 for r_ in rows_rv3 for x_ in ((((r_.get("blocks") or {}).get("S*") or {}).get("package") or {}).get("provenance") or [])
               if str(x_.get("source") or "").startswith("kb:"))
    check("9.27e the positive control 9.27 exercised real kb: records (the rehearsal-v3 packages cite KB records)", n_kb > 0, n_kb)

    def m_kbfake(ev):
        b_ = ev["capture"][0]["blocks"]["S*"]
        b_["ids"] = b_["ids"] + ["procedure/*/NOT-A-RECORD"]
        b_["package"]["provenance"].append({"id": "procedure/*/NOT-A-RECORD", "component": "procedure", "source": kb_src,
                                            "evidence": ["proc:NOT-A-RECORD"], "authored_by": auth_})
    ok_kb, bad_kb = fails(m_kbfake, {"c"})
    check("9.27f decide(): a package citing a fake KB record under the genuine digest fails (c)", ok_kb, bad_kb)

    # ---- Q7: owed items validated by content
    reg_p = AC.acceptance_registration_path()
    fake_owed = {str(k): {"kind": A.OWED_KINDS[k], "done_by": "selftest", "evidence": [{"path": "docs/final-test/acceptance_registration.json",
                                                                                         "sha256": A.file_sha(reg_p), "role": "main"}]}
                 for k in range(len(REG["owed_before_first_call"]))}
    A._docs_json = lambda name, root=None: (copy.deepcopy(fake_owed), "x") if name == A.OWED_FILE else orig_dj(name, root)
    try:
        op = A.owed_problems(REG)
    finally:
        A._docs_json = orig_dj
    flagged = {int(x.split()[2]) for x in op if x.startswith("owed item ")}
    check("9.28 Astra r2 Q7 probe: binding EVERY owed item to the registration JSON alone leaves every item open", flagged == set(range(7)), op[:8])
    fo2_ = dict(fake_owed)
    fo2_["5"] = dict(fake_owed["5"], kind="arash_go")
    A._docs_json = lambda name, root=None: (copy.deepcopy(fo2_), "x") if name == A.OWED_FILE else orig_dj(name, root)
    try:
        op2 = A.owed_problems(REG)
    finally:
        A._docs_json = orig_dj
    check("9.29 an owed record of the wrong kind is refused", any("owed item 5" in x and "recorded as kind" in x for x in op2))
    # ACC-RUNNER-D: items 5 / 6 are bound to their OWN evidence (the quoted journal event; Arash's authorization + rulings files)
    ev_rows = [x_ for x_ in (A._rows(C.journal_path(C.ACCEPTANCE_NS, "events")) or []) if x_.get("event") == "fake_overcap_test"]
    fo_q = {"event": {"ts": "2099-01-01T00:00:00", "result": {"ok": True, "refused": True, "open_holds_after": 0}}}
    fo_soft = {"event": {"ts": "x", "result": {"ok": True, "refused": False, "open_holds_after": 0}}}
    check("9.29b owed item 5: no quoted event, a quoted event that is not in the journal, or a non-refusal result is refused",
          bool(A._owed_fake_overcap({}, None)) and bool(A._owed_fake_overcap(fo_q, None)) and bool(A._owed_fake_overcap(fo_soft, None)))
    if ev_rows:
        check("9.29c owed item 5 positive control: the real recorded fake over-cap event, quoted exactly, validates",
              A._owed_fake_overcap({"event": {"ts": ev_rows[-1]["ts"], "result": ev_rows[-1]["result"]}}, None) == [])
    rp_ = os.path.join(ROOT, "docs", "final-test", "acceptance_registration.json")
    check("9.29d owed item 6 must cite the authorization + rulings files themselves (the registration JSON is refused)",
          len(A._owed_arash_go({}, {"main": rp_, "authorization": rp_}, None, ROOT)) == 2)
    rr_ = C.paths(None)["root"]
    check("9.30 positive controls on the real evidence: items 1 / 2 (live checks bound by plan + rows shas) and 3 (visibility + routes) validate",
          A._owed_roster({"binding": A.live_binding()}, {}, REG, None, rr_, "L3") == []
          and A._owed_roster({"binding": A.live_binding()}, {}, REG, None, rr_, "L1L2") == []
          and A._owed_visibility({}, {"visibility": os.path.join(A.LIVE, "acc-add", "visibility_l3.json"),
                                      "alt_routes": os.path.join(A.ALT_DIR, "check.txt")}, REG, None, rr_) == [])
    # ---- ACC-RUNNER-D (Astra r3 Q7): restore rows must carry THE PLAN's configuration hash (recomputed), not merely one uniform hash
    d_l3 = A.CHECK_DIRS["L3"]
    pl1 = json.load(open(os.path.join(d_l3, "honest-r1", "plan.json"), encoding="utf-8"))
    rows_l3 = A._rows(os.path.join(d_l3, "honest-r1", "refplans_honest.jsonl"))
    want_h, why_h = A.expected_config_hash(pl1)
    check("9.30a positive control: the recomputed configuration of the L3 restore-1 plan equals its rows' hash (no problem)",
          why_h is None and {r_.get("hash") for r_ in rows_l3} == {want_h}
          and A.restore_binding_problems(REG, d_l3, 1, pl1["items"][0], rows_l3) == [], (want_h, why_h))
    obs = [dict(r_, hash="obsolete-config") for r_ in rows_l3]
    nohash = [{k_: v_ for k_, v_ in r_.items() if k_ != "hash"} for r_ in rows_l3]
    pb_obs = A.restore_binding_problems(REG, d_l3, 1, pl1["items"][0], obs)
    pb_no = A.restore_binding_problems(REG, d_l3, 1, pl1["items"][0], nohash)
    check("9.30b Astra r3 probe: every row with hash 'obsolete-config', or every hash removed, is REFUSED (uniform is not the plan's)",
          any("configuration" in x_ for x_ in pb_obs) and any("configuration" in x_ for x_ in pb_no), (pb_obs, pb_no))
    orig_rows = A._rows
    A._rows = lambda q: obs if q.endswith(os.path.join("honest-r1", "refplans_honest.jsonl")) and q.startswith(d_l3) else orig_rows(q)
    try:
        st_obs = A.live_status(REG)
    finally:
        A._rows = orig_rows
    check("9.30c an L3 item whose restore rows carry another configuration is not PASS in live_status", (st_obs.get(pl1["items"][0]) or {}).get("verdict") != "PASS",
          (st_obs.get(pl1["items"][0]) or {}))
    # ---- ACC-RUNNER-D (Astra r3 NEW defect 2): the required alternative routes come from alt_routes.json, not from the report
    adoc = json.load(open(os.path.join(A.ALT_DIR, "check.txt"), encoding="utf-8"))
    eq_all = [x_[0] for x_ in adoc["items"] if x_[1] == "equivalent"]
    one = dict(adoc, items=[x_ for x_ in adoc["items"] if x_[1] != "equivalent"] + [[eq_all[0], "equivalent"]])
    onep = os.path.join(TMP, "alt_one.json")
    open(onep, "w", encoding="utf-8").write(json.dumps(one))
    vp_ = os.path.join(A.LIVE, "acc-add", "visibility_l3.json")
    orig_r = A._rows
    A._rows = lambda q: [r_ for r_ in (orig_r(q) or []) if r_.get("id") not in eq_all[1:]] if os.path.dirname(os.path.dirname(q)) == A.ALT_DIR else orig_r(q)
    try:
        pv_one = A._owed_visibility({}, {"visibility": vp_, "alt_routes": onep}, REG, None, rr_)
    finally:
        A._rows = orig_r
    check("9.30d Astra r3 probe: a report (and rows) keeping 1 of the 6 equivalent routes, with the correct routes-file hash, is REFUSED",
          len(eq_all) == 6 and any("differs from alt_routes.json" in x_ for x_ in pv_one) and any("without a PASS row" in x_ for x_ in pv_one), pv_one)
    kind_ = dict(adoc, items=[[x_[0], "probe" if x_[0] == eq_all[0] else x_[1]] for x_ in adoc["items"]])
    open(onep, "w", encoding="utf-8").write(json.dumps(kind_))
    pv_kind = A._owed_visibility({}, {"visibility": vp_, "alt_routes": onep}, REG, None, rr_)
    orig_alt = A.ALT_ROUTES
    A.ALT_ROUTES = os.path.join(TMP, "no_routes.json")
    try:
        pv_nofile = A._owed_visibility({}, {"visibility": vp_, "alt_routes": os.path.join(A.ALT_DIR, "check.txt")}, REG, None, rr_)
    finally:
        A.ALT_ROUTES = orig_alt
    check("9.30e an equivalent route relabelled 'probe' in the report is refused; a missing alt_routes.json fails closed",
          any("kind changed" in x_ for x_ in pv_kind) and any("alt_routes.json missing" in x_ for x_ in pv_nofile), (pv_kind, pv_nofile))
    lb_bad = A.live_binding()
    lb_bad["L3"]["1"]["rows_sha"] = "0" * 64
    check("9.31 a live-check record whose rows sha is not the folder's refuses item 1", bool(A._owed_roster({"binding": lb_bad}, {}, REG, None, rr_, "L3")))
    gs = os.path.join(TMP, "summary.txt")
    open(gs, "w").write("\n".join(f"{g_} rc=0 :: ok" for g_ in A.OWED_GATES) + "\nDONE\n")
    code_now = {rel: A.file_sha(os.path.join(rr_, *rel.split("/"))) for rel in A.OWED_CODE}
    good0 = A._owed_plumbing({"code": code_now}, {"gates": gs}, REG, None, rr_)
    open(gs, "w").write("\n".join(f"{g_} rc={1 if k_ == 0 else 0} :: x" for k_, g_ in enumerate(A.OWED_GATES)) + "\nDONE\n")
    red0 = A._owed_plumbing({"code": code_now}, {"gates": gs}, REG, None, rr_)
    stale0 = A._owed_plumbing({"code": dict(code_now, **{A.OWED_CODE[0]: "0" * 64})}, {"gates": gs}, REG, None, rr_)
    check("9.32 owed item 0: green gates on the CURRENT plumbing code pass; a red gate or older code refuses", good0 == [] and red0 and stale0, (good0, red0[:1]))
    # ---- RUNNER-FINAL (A24.1-C step 8, owed-list 'validator'): owed item 4 checked by CONTENT (frozen_first_problems)
    sys.path.insert(0, os.path.join(ROOT, "sim", "final_test", "diag", "a24_1"))
    import update_record as UR_
    rec_now_ = json.load(open(os.path.join(ROOT, "docs", "final-test", "a24_1_record.json"), encoding="utf-8"))
    import freeze_ft2 as FT2_
    orig_sig_ = FT2_.signatures_record
    unsigned_ff = A.frozen_first_problems(UR_.build(rec_now_, REG, signed=True), REG, None, ROOT)

    def _signed(root=None):                                 # selftest stand-in for Arash's signature record (never written to disk)
        o_ = orig_sig_(root)
        o_["A24.1-C"] = {"signed": True, "signature": {"by": "Arash", "date": "selftest", "said": "selftest"}}
        return o_
    FT2_.signatures_record = _signed
    good_ff = UR_.build(rec_now_, REG, signed=True)
    p_good = A.frozen_first_problems(good_ff, REG, None, ROOT)
    check("9.32b owed item 4: a record with all six requirements BOUND, every binding resolved now, both ceiling sets, the active guard spec, "
          "the freeze entries, the test L3 record, the A24.1-C prereg text and every owed entry DONE passes", p_good == [], p_good[:3])

    def ff_mut(fn):
        d_ = copy.deepcopy(good_ff)
        fn(d_)
        return A.frozen_first_problems(d_, REG, None, ROOT)

    def _req(d_, name):
        return next(x for x in d_["frozen_first"] if x["requirement"] == name)
    muts = {
        "pending requirement": lambda d_: _req(d_, "infrastructure-retry rules").update(status_id="PENDING"),
        "stale file digest": lambda d_: _req(d_, "retrieval quotas")["binding"][0].update(sha256="0" * 64),
        "stale registration self-sha": lambda d_: _req(d_, "all acceptance rules")["binding"][1].update(sha256="1" * 64),
        "stale blocked-run digest": lambda d_: _req(d_, "infrastructure-retry rules")["binding"][0].update(sha256="2" * 64),
        "missing requirement": lambda d_: d_["frozen_first"].pop(),
        "duplicated requirement": lambda d_: d_["frozen_first"].append(copy.deepcopy(d_["frozen_first"][1])),
        "test L3 binding dropped": lambda d_: _req(d_, "feasibility classifications").update(
            binding=[b for b in _req(d_, "feasibility classifications")["binding"] if "test_l3" not in b["path"]]),
        "runner binding dropped": lambda d_: _req(d_, "all acceptance rules").update(
            binding=[b for b in _req(d_, "all acceptance rules")["binding"] if "acc_runner" not in b["path"]]),
        "unknown hash scope": lambda d_: _req(d_, "retrieval quotas")["binding"][0].update(hash_scope_id="trust_me"),
        "scope on the wrong file": lambda d_: _req(d_, "retrieval quotas")["binding"].append(
            {"path": "sim/ltm_package.py", "sha256": REG["sha256"], "hash_scope_id": "registration_self_sha"}),
        "path outside the tree": lambda d_: _req(d_, "retrieval quotas")["binding"].append(
            {"path": "../outside.json", "sha256": "0" * 64, "hash_scope_id": "file"}),
        "malformed binding": lambda d_: _req(d_, "retrieval quotas")["binding"].append("sim/ltm_package.py"),
        "frozen_first not a list": lambda d_: d_.update(frozen_first={"x": 1}),
        "test-v1 total 100": lambda d_: d_["test_v1_ceilings"].update(total=100.0),
        "acceptance ceiling 9": lambda d_: d_["ceilings"].update(attempt=9.0),
        "guard spec ftm-guard-2": lambda d_: d_["target_guard_spec"].update(version="ftm-guard-2"),
        "guard reference 0.012": lambda d_: d_["target_guard_spec"].update(reference_usd_per_billed_opus_cell=0.012),
        "guard not explicit": lambda d_: d_["target_guard_spec"].update(threshold_is_not_2x_reference=False),
        "guard unit changed": lambda d_: d_["target_guard_spec"].update(unit="one Opus call"),
        "tripwire 0.024": lambda d_: d_["tripwire"].update(pause_above_usd_per_billed_opus_cell=0.024),
        "owed entry pending": lambda d_: d_["owed"][0].update(status="PENDING"),
        "no owed list": lambda d_: d_.pop("owed"),
        "other registration": lambda d_: d_.update(registration_sha="3" * 64),
        "other pins": lambda d_: d_["pins"].update(dataset="4" * 64),
        "owed entries removed": lambda d_: d_.update(owed=[o for o in d_["owed"] if o["due"] != "before_first_acceptance_call"]),
        "owed deadline moved": lambda d_: d_["owed"][6].update(due="final_test_freeze_after_acceptance"),
    }
    miss_ = sorted(k for k, f in muts.items() if not ff_mut(f))
    check("9.32c owed item 4: each stale / missing / malformed binding, a PENDING requirement or owed entry, a changed ceiling set, guard "
          "spec or tripwire, another registration or pins, a shortened owed list or a moved deadline refuses (26 mutations)", miss_ == [], miss_)
    # environment-side: the freeze entries, the test L3 record and the prereg text are read from the tree (a temp copy of the docs)
    tdr_ = tempfile.mkdtemp(prefix="acc_ff_")
    try:
        dd_ = os.path.join(tdr_, "docs", "final-test")
        os.makedirs(dd_)
        for nm_ in ("freeze.json", "preregistration.md", "test_l3_feasibility.json", "acceptance_registration.json"):
            shutil.copyfile(os.path.join(ROOT, "docs", "final-test", nm_), os.path.join(dd_, nm_))
        orig_paths_ = C.paths

        def pths_(root=None):
            p_ = dict(orig_paths_(None))
            p_.update(freeze=os.path.join(dd_, "freeze.json"), prereg=os.path.join(dd_, "preregistration.md"))
            return p_
        good_tmp = copy.deepcopy(good_ff)
        A_tl = A.TEST_L3_FEAS
        C.paths = pths_
        try:
            base_ = A.frozen_first_problems(good_tmp, REG, None, ROOT)
            fz_ = json.load(open(os.path.join(dd_, "freeze.json"), encoding="utf-8"))
            fz_.pop("cost_guard")
            fz_["sha256"] = C.self_sha(fz_)
            json.dump(fz_, open(os.path.join(dd_, "freeze.json"), "w", encoding="utf-8"))
            no_entry = A.frozen_first_problems(good_tmp, REG, None, ROOT)
            shutil.copyfile(os.path.join(ROOT, "docs", "final-test", "freeze.json"), os.path.join(dd_, "freeze.json"))
            pre_ = open(os.path.join(dd_, "preregistration.md"), encoding="utf-8").read().replace("\n### A24.1-C (", "\n### A24.1-X (")
            open(os.path.join(dd_, "preregistration.md"), "w", encoding="utf-8").write(pre_)
            no_text = A.frozen_first_problems(good_tmp, REG, None, ROOT)
            pre2_ = open(os.path.join(ROOT, "docs", "final-test", "preregistration.md"), encoding="utf-8").read()
            i_ = pre2_.index(chr(10) + "### A24.1-C (")
            open(os.path.join(dd_, "preregistration.md"), "w", encoding="utf-8").write(pre2_[: pre2_.index(chr(10), i_ + 1) + 1])
            no_body = A.frozen_first_problems(good_tmp, REG, None, ROOT)
        finally:
            C.paths = orig_paths_
        tl_ = json.load(open(os.path.join(ROOT, A_tl), encoding="utf-8"))
        bad_tl = []
        for mut_ in (lambda t: t.update(problems=["x"]), lambda t: t["roster"]["items"].append("g-l3-not-classified"),
                     lambda t: t["dataset"].update(sha256="5" * 64)):
            t2 = copy.deepcopy(tl_)
            mut_(t2)
            t2["sha256"] = C.self_sha(t2)
            q_ = os.path.join(ROOT, "sim", "final_test", "_checkpoints", "RUNNER-FINAL", "_selftest_tl.json")   # inside the tree
            json.dump(t2, open(q_, "w", encoding="utf-8"))
            A.TEST_L3_FEAS = "sim/final_test/_checkpoints/RUNNER-FINAL/_selftest_tl.json"
            try:
                bad_tl.append(any("_selftest_tl.json" in x for x in A.frozen_first_problems(good_tmp, REG, None, ROOT)))
            finally:
                A.TEST_L3_FEAS = A_tl
                os.remove(q_)
    finally:
        shutil.rmtree(tdr_, ignore_errors=True)
    check("9.32d owed item 4 reads the tree: no cost_guard entry in freeze.json, no A24.1-C heading after A33 in the prereg refuse; the clean "
          "copy passes; a heading without the approved r4 body refuses", base_ == [] and any("cost_guard" in x for x in no_entry)
          and any("A24.1-C" in x for x in no_text) and any("approved r4 text" in x for x in no_body), (base_[:2], no_entry[:2], no_text[:2], no_body[:2]))
    check("9.32e owed item 4: a test L3 record with open problems, an unclassified roster item or another dataset refuses (self-sha recomputed)",
          bad_tl == [True, True, True], bad_tl)
    FT2_.signatures_record = orig_sig_
    check("9.32g owed item 4: without Arash's recorded A24.1-C signature (signatures.json) the record refuses (due before the first call)",
          any("not signed by Arash" in x for x in unsigned_ff) or bool(orig_sig_(None).get("A24.1-C", {}).get("signed")), unsigned_ff[:2])
    check("9.32f owed item 4: the 'record' evidence must be docs/final-test/a24_1_record.json itself",
          A._owed_frozen_first({}, {"record": reg_p}, REG, None, ROOT) == ["the 'record' evidence is not docs/final-test/a24_1_record.json"])

    # ---- --attempt: never dropped
    calls_ = []
    orig_mp, orig_me = A.make_plan, A.make_execute
    A.make_plan = lambda *a, **k: calls_.append(("plan", a, k))
    A.make_execute = lambda *a, **k: calls_.append(("execute", a, k))
    try:
        rcs = [A.main(["plan", "--stage", "opus", "--attempt", "2"]), A.main(["execute", "--attempt", "2"]), A.main(["check", "--attempt", "2"]),
               A.main(["c11", "--attempt", "2"]), A.main(["fake-overcap", "--attempt", "2"]), A.main(["seal-roster", "--attempt", "2"])]
    finally:
        A.make_plan, A.make_execute = orig_mp, orig_me
    check("9.33 Astra r2 probe: --attempt 2 is refused by EVERY command and nothing is dispatched (mocked dispatch)", rcs == [2] * 6 and calls_ == [], (rcs, calls_))

    # ---- Q5: the 12.2 / 12.3 / 12.7 evidence functions on Astra's probes
    man_ = C.load_manifest()
    run1_ = [r_ for r_ in rows_rv3 if int(r_.get("capture_run", 1)) == 1]
    want_ = GE.expected_bilingual_keys(man_, "rehearsal-v3", ["canon", "reloc"])
    one_item = [r_ for r_ in run1_ if r_["item"] == run1_[0]["item"]]
    check("9.34 12.2 positive control: the real run-1 captures cover every registered key in both languages", GE.bilingual_coverage(run1_, want_)["ok"])
    bc1 = GE.bilingual_coverage(one_item, want_)
    check("9.35 Astra r2 Q5 probe: keeping only one complete bilingual item of ten FAILS 12.2 (wholly missing pairs detected)",
          bc1["ok"] is False and len(bc1["missing_or_one_language"]) >= 20, bc1["missing_or_one_language"][:3])
    no_fa = [r_ for r_ in run1_ if not (r_["item"] == run1_[3]["item"] and r_["lang"] == "fa")]
    check("9.36 12.2: one presentation missing its FA row fails", GE.bilingual_coverage(no_fa, want_)["ok"] is False)
    prac_ = [json.loads(x) for x in open(C.journal_path("practice-v2", "practice"), encoding="utf-8") if x.strip()]
    cert_ = json.load(open(os.path.join(ROOT, "sim", "final_test", "diag", "a31", "equiv_replay.json"), encoding="utf-8"))
    import treatment as TT
    dev_sha_ = TT.identity("rehearsal-v3")["sha"]
    pj = C.sha256_file(C.journal_path("practice-v2", "practice"))
    hid_ok = GE.honesty_identity(rows_rv3, prac_, dev_sha=dev_sha_, dev_ns="rehearsal-v3", prac_run_sha_prefix=GE.PRACTICE_RUN_IDENTITY, cert=cert_, prac_journal_sha=pj)
    stale_rows = [dict(r_, treatment_id="89c38daf" * 8) for r_ in rows_rv3]
    hid_bad = GE.honesty_identity(stale_rows, prac_, dev_sha=dev_sha_, dev_ns="rehearsal-v3", prac_run_sha_prefix=GE.PRACTICE_RUN_IDENTITY, cert=cert_, prac_journal_sha=pj)
    hid_cert = GE.honesty_identity(rows_rv3, prac_, dev_sha=dev_sha_, dev_ns="rehearsal-v3", prac_run_sha_prefix=GE.PRACTICE_RUN_IDENTITY, cert=cert_, prac_journal_sha="0" * 64)
    check("9.37 12.3 positive control: every capture / practice row carries its bound identity", hid_ok["ok"], hid_ok)
    check("9.38 Astra r2 Q5 probe: changing every capture's treatment identity to an obsolete value FAILS 12.3", hid_bad["ok"] is False
          and hid_bad["n_capture_off_identity"] == 56)
    check("9.39 12.3: a practice journal not bound by the A31 certificate fails", hid_cert["ok"] is False)
    td_ = os.path.join(TMP, "probes")
    os.makedirs(td_)
    spp = GE.spend_probes(td_)
    check("9.40 12.7: the A29 accounting probes (boundary, identity, sqlite error, hold reconciliation) all hold", spp["ok"], spp)
    check("9.40b 12.7 now probes the lock itself: an old lock is not taken over, a waiting contender unlinks nothing, release needs our token",
          spp.get("old_lock_not_taken_over") is True and spp.get("contender_never_unlinks") is True and spp.get("release_only_own_token") is True, spp)

    # ---- Q6: 12.12 against the REGISTERED class hash; 12.11 by AST callers + the effective configuration
    types_, _s = PLN.load_types(None, ROOT)
    dt_ = FF.detector_types(ROOT, os.path.join(ROOT, *FZ.REGISTERED_CLASSES["weights"].split("/")))
    pr_, ct_, shown_, sysx_, T_ = FZ.prompt_ft2(ROOT, dt_["types"])
    g_ok = FZ.gate_1212_ft2(ROOT, dt_["types"], dt_["weights_sha"], ct_, shown_, sysx_, T_)
    orig_rc = dict(FZ.REGISTERED_CLASSES)
    FZ.REGISTERED_CLASSES["ordered_sha"] = "0" * 64
    try:
        g_bad = FZ.gate_1212_ft2(ROOT, dt_["types"], dt_["weights_sha"], ct_, shown_, sysx_, T_)
    finally:
        FZ.REGISTERED_CLASSES.clear()
        FZ.REGISTERED_CLASSES.update(orig_rc)
    FZ.REGISTERED_CLASSES["weights_sha"] = "1" * 64
    try:
        g_bad2 = FZ.gate_1212_ft2(ROOT, dt_["types"], dt_["weights_sha"], ct_, shown_, sysx_, T_)
    finally:
        FZ.REGISTERED_CLASSES.clear()
        FZ.REGISTERED_CLASSES.update(orig_rc)
    check("9.41 12.12 positive control: the ordered digest and the weights hash equal the independently registered ones", g_ok["status"] == "PASS", g_ok["evidence"].get("reasons"))
    check("9.42 Astra r2 Q6 probe: a class digest that differs from the REGISTERED one fails 12.12 (consistency alone is not enough)",
          g_bad["status"] == "FAIL" and any("registered" in x for x in g_bad["evidence"]["reasons"]))
    check("9.43 12.12: weights other than the registered (identity-pinned) ones fail", g_bad2["status"] == "FAIL")
    rt = os.path.join(TMP, "rc")
    os.makedirs(os.path.join(rt, "sim"))
    open(os.path.join(rt, "sim", "stray.py"), "w").write("import plan as PL\nPL.render_messages(ct, rec, 'none')\n")
    rcs_ = FZ.render_callers(rt, ["sim/stray.py"])
    check("9.44 12.11: render_messages callers are found by the AST (a stray caller outside the registered modules is listed)",
          rcs_ == {"sim/stray.py": [2]})
    g11 = FZ.gate_1211_ft2(ROOT, ["sim/final_test/matrix/plan.py", "sim/final_test/matrix/capture.py", "sim/final_test/matrix/execute.py"], ct_)
    check("9.45 12.11 positive control: only registered callers, plan.py among them, one effective contract for test-v1 = acceptance-v2",
          g11["status"] == "PASS" and g11["evidence"]["one_configuration"] and not g11["evidence"]["stray_render_callers"], {k: g11["evidence"].get(k) for k in ("stray_render_callers", "one_configuration")})
finally:
    shutil.rmtree(TMP, ignore_errors=True)
# FIX-B2 (stale state): another live process (the dress rehearsal, a gate run in parallel) may legitimately hold the repo study lock while this
# selftest runs; the claim is that THIS selftest leaves none. A lock carrying our pid + host (or one that stays unreadable) fails.
def _lock_left_by_us(p_):
    import socket as _so
    import time as _ti
    h_ = AC.lock_holder(p_)
    if h_ == {}:                                          # being written by its creator right now, or corrupt: look once more
        _ti.sleep(0.3)
        h_ = AC.lock_holder(p_)
    return h_ is not None and (h_ == {} or (h_.get("pid") == os.getpid() and h_.get("host") == _so.gethostname()))


_td46 = tempfile.mkdtemp(prefix="lock46_")
try:
    _p46 = os.path.join(_td46, AC.STUDY_LOCK_NAME)
    _neg46 = []
    _tk46 = AC._lock_acquire(_p46, 1.0)
    _neg46.append(_lock_left_by_us(_p46))                 # our own lock -> detected
    AC._lock_release(_p46, _tk46)
    _neg46.append(_lock_left_by_us(_p46))                 # released -> clean
    with open(_p46, "w") as _f46:
        _f46.write(json.dumps({"pid": os.getpid() + 1, "host": "elsewhere", "ts": 0, "token": "f" * 32}))
    _neg46.append(_lock_left_by_us(_p46))                 # another process's lock -> not ours
    with open(_p46, "w") as _f46:
        _f46.write("")
    _neg46.append(_lock_left_by_us(_p46))                 # unreadable -> fails closed
finally:
    shutil.rmtree(_td46, ignore_errors=True)
check("9.46a the lock-left predicate: own lock = left, released = clean, another process's lock = not ours, unreadable = left (fail closed)",
      _neg46 == [True, False, False, True], _neg46)
check("9.46 no study lock taken by this selftest is left in the repo work dir (a concurrent process's lock is not this selftest's)",
      not _lock_left_by_us(os.path.join(C.paths(None)["matrix_work"], AC.STUDY_LOCK_NAME)),
      AC.lock_holder(os.path.join(C.paths(None)["matrix_work"], AC.STUDY_LOCK_NAME)))

# ------------------------------------------------------------------------------------------------ 10 decision 3: the conditional authorization
print("== 10 conditional authorization (Arash decision 3) and the real rulings")
T10 = tempfile.mkdtemp(prefix="acc_d_")
try:
    for rel in A.OWED_CODE:
        os.makedirs(os.path.dirname(os.path.join(T10, *rel.split("/"))), exist_ok=True)
        shutil.copy(os.path.join(ROOT, *rel.split("/")), os.path.join(T10, *rel.split("/")))
    vd = os.path.join(T10, "docs", "final-test", "_astra")
    os.makedirs(vd)

    def verdict(name, text, code=True, code_over=None):
        q = os.path.join(vd, name)
        open(q, "w", encoding="utf-8").write(text)
        if code:
            cd = {rel: A.file_sha(os.path.join(T10, *rel.split("/"))) for rel in A.OWED_CODE}
            cd.update(code_over or {})
            json.dump({"verdict_sha256": A.file_sha(q), "code": cd}, open(q[:-4] + ".code.json", "w", encoding="utf-8"))
        return "docs/final-test/_astra/" + name
    check("10.1 astra_verdict reads the first verdict word ('**REJECT.**', 'ACCEPT', anything else = None)",
          A.astra_verdict(os.path.join(T10, *verdict("a.txt", "**REJECT.**\n\nbody", code=False).split("/"))) == "REJECT"
          and A.astra_verdict(os.path.join(T10, *verdict("b.txt", "\n**ACCEPT**\nok", code=False).split("/"))) == "ACCEPT"
          and A.astra_verdict(os.path.join(T10, *verdict("c.txt", "I think it is fine. ACCEPT", code=False).split("/"))) is None
          and A.astra_verdict(os.path.join(T10, *verdict("d.txt", "**ACCEPT WITH CHANGES**\n1. x", code=False).split("/"))) == "ACCEPT WITH CHANGES")
    amb = []
    for k_, txt_ in enumerate(("ACCEPT only after the following changes", "ACCEPT / REJECT", "**ACCEPT** -- but see item 3", "ACCEPT?",
                               "ACCEPT, REJECT", "Verdict: ACCEPT", "ACCEPT WITH CHANGES (1) x", "accepted")):
        q_ = verdict(f"amb{k_}.txt", txt_ + "\nbody\n")
        amb.append((txt_, A.astra_verdict(os.path.join(T10, *q_.split("/"))),
                    A.authorization_condition_problems({"astra_accept": [q_], "gates_green": True, "owed_records": True}, T10)))
    check("10.1c Astra ACC-RUNNER-D r1 probes: a qualified / ambiguous first line ('ACCEPT only after ...', 'ACCEPT / REJECT', ...) is no "
          "ACCEPT, and the FULL condition validator refuses each with a matching code record", all(v_ != "ACCEPT" and pr_ for _t, v_, pr_ in amb), amb)
    plain = []
    for k_, txt_ in enumerate(("ACCEPT", "**ACCEPT**", "**ACCEPT.**", "## ACCEPT", "  accept  ")):
        q_ = verdict(f"pl{k_}.txt", txt_ + "\n\nbody\n")
        plain.append((txt_, A.authorization_condition_problems({"astra_accept": [q_], "gates_green": True, "owed_records": True}, T10)))
    check("10.1d a plain ACCEPT line (markdown bold / heading / trailing period allowed) satisfies the condition when its code record matches",
          all(pr_ == [] for _t, pr_ in plain), plain)
    acw = verdict("r2w.txt", "**ACCEPT WITH CHANGES** (two small changes)\n")
    check("10.1b an ACCEPT WITH CHANGES does not satisfy the authorization condition (changes are still owed)",
          bool(A.authorization_condition_problems({"astra_accept": [acw], "gates_green": True, "owed_records": True}, T10)))
    cond = lambda files: {"astra_accept": files, "gates_green": True, "owed_records": True}   # noqa: E731
    r_none = A.authorization_condition_problems(cond(["docs/final-test/_astra/none-r1.txt", "docs/final-test/_astra/none-r2.txt"]), T10)
    rj = verdict("r1.txt", "**REJECT.**\n")
    r_rej = A.authorization_condition_problems(cond([rj]), T10)
    ac_nocode = verdict("r2n.txt", "**ACCEPT**\n", code=False)
    r_nocode = A.authorization_condition_problems(cond([rj, ac_nocode]), T10)
    ac = verdict("r2.txt", "**ACCEPT**\n")
    r_ok = A.authorization_condition_problems(cond([rj, ac]), T10)
    r_lastrej = A.authorization_condition_problems(cond([ac, rj]), T10)
    check("10.2 the authorization is NOT usable before any verdict, after a REJECT, or on an ACCEPT without its code record",
          r_none and r_rej and r_nocode, (r_none, r_rej, r_nocode))
    check("10.3 it is usable on an ACCEPT whose code record equals the plumbing code on disk; the LATEST named verdict decides",
          r_ok == [] and bool(r_lastrej), (r_ok, r_lastrej))
    with open(os.path.join(T10, *A.OWED_CODE[0].split("/")), "a", encoding="utf-8") as f:
        f.write("\n# a change after the review\n")
    r_changed = A.authorization_condition_problems(cond([rj, ac]), T10)
    check("10.4 a plumbing file changed after Astra's ACCEPT voids the authorization", any("changed since Astra reviewed" in x for x in r_changed), r_changed)
    r_shape = A.authorization_condition_problems({"astra_accept": [ac]}, T10)
    r_soft = A.authorization_condition_problems(dict(cond([ac]), gates_green=False), T10)
    check("10.5 conditions of another shape, or with the gate / owed requirement switched off, are refused", bool(r_shape) and bool(r_soft))
    check("10.6 record_authorization refuses malformed conditions", raises(lambda: A.record_authorization(
        by="Arash", date="2026-10-14", said="s", ledger_entry="ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)",
        conditions={"astra_accept": []}), A.Refused) or not REG.get("roster"))
    fk = {"schema": A.AUTH_SCHEMA, "by": "Arash", "date": "2026-10-14", "said": "go", "attempt": 1, "registration_sha": REG["sha256"],
          "roster_sha": A.roster_sha(ROSTER), "ledger_entry": "ARASH DECISIONS (MORNING-DECISIONS.html; all 9 ticked, option A each; verbatim)",
          "conditions": cond(["docs/final-test/_astra/none-r1.txt"])}
    fk["sha256"] = C.self_sha(fk)
    fk_strip = {k_: v_ for k_, v_ in fk.items() if k_ != "conditions"}
    od_ = A._docs_json
    try:
        A._docs_json = lambda name, root=None: (copy.deepcopy(fk), "x") if name == A.AUTH_FILE else od_(name, root)
        ap_c = A.authorization_problems(REG, ROSTER, 1)
        A._docs_json = lambda name, root=None: (copy.deepcopy(fk_strip), "x") if name == A.AUTH_FILE else od_(name, root)
        ap_s = A.authorization_problems(REG, ROSTER, 1)
    finally:
        A._docs_json = od_
    check("10.7 a conditional authorization with no verdict yet is refused, and stripping its conditions breaks its self-sha",
          any("condition not met" in x for x in ap_c) and any("self-sha" in x for x in ap_s), (ap_c, ap_s))
finally:
    shutil.rmtree(T10, ignore_errors=True)
# the REAL files (read-only): what Arash approved is what the runner reads
ra, _p = REAL_DJ(A.AUTH_FILE)
rr, _p2 = REAL_DJ(A.RULINGS_FILE)
A._docs_json = REAL_DJ
try:
    if ra is not None:
        base_bad = [x for x in A.authorization_problems(REG, ROSTER, 1) if "condition not met" not in x]
        check("10.8 the real authorization is Arash's, bound to this registration + roster + attempt 1 + an ARASH ledger heading, unedited",
              base_bad == [], base_bad)
        check("10.9 the real authorization is CONDITIONAL (decision 3: only after Astra ACCEPT, green gates, the owed records)",
              isinstance(ra.get("conditions"), dict) and ra["conditions"].get("gates_green") is True and ra["conditions"].get("owed_records") is True
              and bool(ra["conditions"].get("astra_accept")))
    if rr is not None:
        check("10.10 the real rulings: one-batch and g-eligible are approved, by Arash, bound to the current registration, unedited",
              A.ruling_problems("one-batch") == [] and A.ruling_problems("g-eligible") == [], (A.ruling_problems("one-batch"), A.ruling_problems("g-eligible")))
        ev10 = synth()
        ev10["g_ruling_open"] = A.ruling_open("g-eligible")
        d10 = A.decide(ev10, REG, ROSTER)
        check("10.11 the real approved (g) ruling switches (g) from PENDING to the real check: without g_eligible.json decide stays NOT PASS "
              "on (g) only", d10["verdict"] == "NOT PASS" and [k for k, v in d10["criteria"].items() if not v["pass"]] == ["g"]
              and any("g_eligible.json missing" in x for x in d10["criteria"]["g"]["reasons"]))
finally:
    A._docs_json = REAL_DJ
check("10.12 at the end, the real acceptance-v2 docs + work dirs are byte-identical to the start", ns_snapshot() == NS0)

# DRESS-REHEARSAL B1: the paid boundary's cost guard reads ns_cfg["ceilings"]["core"]; acceptance has only "total" (KeyError 'core' before)
import plan as _PLB1                                                     # noqa: E402
_orig_ge = _PLB1.Plan._guard_eval
try:
    _PLB1.Plan._guard_eval = lambda self: {"core": self.ns_cfg["ceilings"]["core"], "total": self.ns_cfg["ceilings"]["total"],
                                           "scopes": {"core": {"pause": True, "reasons": ["projected core"]}, "total": {"pause": False, "reasons": []}},
                                           "pause": True, "reasons": ["projected core"]}
    for _batch in (False, True):
        _pb = object.__new__(A.plan_class(_batch))
        _cfg = C.NAMESPACES[C.ACCEPTANCE_NS]
        _pb.ns_cfg = _cfg
        _g = _pb._guard_eval()
        check(f"11.4 B1 ({'batch' if _batch else 'plain'}): the aggregate pause / reasons come from the total scope only (Astra r1 2)",
              _g.pop("pause") is False and _g.pop("reasons") == [] and _g.pop("core_diagnostic_only") is True and bool(_g.pop("scopes")))
        check(f"11.1 B1 ({'batch' if _batch else 'plain'}): the core-scope guard of an acceptance namespace uses its one total ceiling, no KeyError",
              _g == {"core": _cfg["ceilings"]["total"], "total": _cfg["ceilings"]["total"]}, _g)
        check(f"11.2 B1 ({'batch' if _batch else 'plain'}): ns_cfg and common.NAMESPACES are left untouched", _pb.ns_cfg is _cfg
              and "core" not in C.NAMESPACES[C.ACCEPTANCE_NS]["ceilings"])
    _pb.ns_cfg = {"ceilings": {"core": 1.0, "total": 2.0}}
    _g3 = _pb._guard_eval()
    check("11.3 B1: a namespace that has a core ceiling keeps it (and its own pause rule)", _g3["core"] == 1.0 and _g3["total"] == 2.0
          and _g3["pause"] is True)
finally:
    _PLB1.Plan._guard_eval = _orig_ge

# ------------------------------------------------------------------------------------------------ 12 FIX-B2: (g) is exactly the registered text
print("== 12 FIX-B2: criterion (g) has no parser condition (registration decision_rule.g / prereg A24 item 7 (g)); parser fallback is listed, not gated")
_reg_g = (REG.get("decision_rule") or {}).get("g") or ""
check("12.0 the registered (g) text names tokens, identical bytes and missing eligible records, and no parser condition",
      "2,000 tokens" in _reg_g and "identical bytes" in _reg_g and "no eligible record missing" in _reg_g and "parser" not in _reg_g.lower(), _reg_g)
_FB = sorted(r["pres"] for r in synth()["capture"])[:3]


def _with_fb(ev, val=False):
    for r in ev["capture"]:
        if r["pres"] in _FB:
            r["blocks"]["S*"]["package"]["parser_ok"] = val
    return ev


_d12 = A.decide(_with_fb(g_synth()), REG, ROSTER)
check("12.1 POSITIVE: parser_ok false on 3 presentations of otherwise clean evidence -> (g) passes, verdict PASS",
      _d12["verdict"] == "PASS" and _d12["criteria"]["g"]["pass"], {k: v["reasons"][:2] for k, v in _d12["criteria"].items() if not v["pass"]})
check("12.2 those presentations are listed descriptively (parser_fallback_descriptive_not_gated) and no criterion is named after the parser",
      _d12["parser_fallback_descriptive_not_gated"] == _FB and not any("parser" in k for k in _d12["criteria"]), _d12["parser_fallback_descriptive_not_gated"])
for _v12 in (None, "yes", 0):
    _ev12 = g_synth()
    for r in _ev12["capture"]:
        if r["pres"] == _FB[0]:
            r["blocks"]["S*"]["package"]["parser_ok"] = _v12
    _dv = A.decide(_ev12, REG, ROSTER)
    check(f"12.3 parser_ok = {_v12!r} is listed as a fallback (only True counts as parsed) and never gates (g)",
          _dv["verdict"] == "PASS" and _dv["parser_fallback_descriptive_not_gated"] == [_FB[0]])
_d12c = A.decide(g_synth(), REG, ROSTER)
check("12.4 clean evidence lists no parser fallback", _d12c["verdict"] == "PASS" and _d12c["parser_fallback_descriptive_not_gated"] == [])


def _m12_tok(ev):
    ev["capture"][0]["blocks"]["S*"]["real_tokens"] = 2001


def _m12_notok(ev):
    ev["capture"][0]["blocks"]["S*"].pop("real_tokens")


def _m12_req(ev):
    for r in ev["plan"]:
        if r["arm"] == "X+M" and r["pres"] == ev["capture"][0]["pres"]:
            r["request_sha"] = "other"


for _nm, _mut in (("2,001 tokens", _m12_tok), ("a missing token count", _m12_notok), ("an X+M request that differs from the captured +M bytes", _m12_req)):
    _ev12 = _with_fb(g_synth())
    _mut(_ev12)
    _dn = A.decide(_ev12, REG, ROSTER)
    check(f"12.5 NEGATIVE kept: with parser fallbacks present, {_nm} still FAILS (g)", _dn["verdict"] == "NOT PASS" and not _dn["criteria"]["g"]["pass"],
          _dn["criteria"]["g"]["reasons"][:2])
_ev12 = _with_fb(g_synth())
_ev12["capture"][[r["pres"] for r in _ev12["capture"]].index(_FB[0])]["blocks"]["S*"]["ids"] = []      # places/x selected, not delivered
_dm = A.decide(_ev12, REG, ROSTER)
check("12.7 NEGATIVE kept (Astra FIX-B2-r1 Q3): parser_ok false AND a missing eligible record -> (g) FAILS as MISSING",
      _dm["verdict"] == "NOT PASS" and any(f"{_FB[0]}: MISSING eligible record places/x" in x for x in _dm["criteria"]["g"]["reasons"]),
      _dm["criteria"]["g"]["reasons"][:3])
_src12 = inspect.getsource(A.decide)
check("12.6 decide() appends no parser reason to any criterion (source scan)", "parser_ok is not true" not in _src12
      and not re_.search(r'crit\[[^\]]+\](?:\.append|\s*\+=)\([^)\n]*parser', _src12))

# ------------------------------------------------------------------------------------------------ 13 READY: fresh storage for the restarted attempt 1
print("== 13 READY: attempt 1 restarts in fresh storage (Arash ruling 1b); the logical namespace, order and the voided rows are unchanged")
import hashlib as hashlib13                               # noqa: E402
_VOID_DIRS = [os.path.join(C.paths(None)["matrix_docs"], "acceptance-v2"), os.path.join(C.paths(None)["matrix_work"], "acceptance-v2")]


def _void_snapshot():
    o = {}
    for base in _VOID_DIRS:
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                q = os.path.join(dp, fn)
                o[os.path.relpath(q, ROOT).replace(os.sep, "/")] = A.file_sha(q)
    return o


_V0 = _void_snapshot()
_PIN13 = json.load(open(os.path.join(ROOT, "sim", "final_test", "_checkpoints", "READY", "voided_storage.sha.json"), encoding="utf-8"))
check("13.1 storage: acceptance-v2's docs / work / journals live in acceptance-v2-restart1; every other namespace in its own name",
      os.path.basename(C.ns_docs(C.ACCEPTANCE_NS)) == os.path.basename(C.ns_work(C.ACCEPTANCE_NS)) == "acceptance-v2-restart1"
      and os.path.basename(os.path.dirname(C.journal_path(C.ACCEPTANCE_NS, "capture"))) == "acceptance-v2-restart1"
      and all(os.path.basename(C.ns_docs(n)) == os.path.basename(C.ns_work(n)) == n for n in C.NAMESPACES if n != C.ACCEPTANCE_NS))
check("13.2 the LOGICAL namespace is unchanged: NAMESPACES key, registration namespace, the seeded order label",
      C.ACCEPTANCE_NS == "acceptance-v2" == REG["namespace"] and "acceptance-v2" in C.NAMESPACES and "acceptance-v2-restart1" not in C.NAMESPACES
      and A.key("exec", "canon:x:en") == hashlib13.sha256("|".join([str(C.MASTER_SEED), "acceptance-v2", "exec", "canon:x:en"]).encode()).hexdigest())
check("13.3 the voided start's 15 files (incl. capture.jsonl, 11 rows) are byte-identical to the READY pin and outside the new storage",
      sorted(_V0.items()) == sorted((k.replace("\\", "/").lstrip("./"), v) for k, v in _PIN13.items())
      and len(_V0) == 15 and sum(1 for x in open(os.path.join(_VOID_DIRS[0], "capture.jsonl"), encoding="utf-8") if x.strip()) == 11
      and all(os.path.normcase(C.ns_docs(C.ACCEPTANCE_NS)) != os.path.normcase(d) for d in _VOID_DIRS), len(_V0))
_refused = []
for _d in _VOID_DIRS:
    try:
        C.Journal(os.path.join(_d, "events.jsonl")).append({"selftest": "13.4"})
        _refused.append(False)
    except C.JournalRefused:
        _refused.append(True)
_t13 = tempfile.mkdtemp(prefix="acc13_")
_tj = os.path.join(_t13, "docs", "final-test", "matrix", "acceptance-v2", "x.jsonl")
C.Journal(_tj).append({"selftest": "13.4"})
check("13.4 a journal write into either voided dir is refused (nothing written); a temporary tree's 'acceptance-v2' dir is unaffected",
      _refused == [True, True] and _void_snapshot() == _V0 and os.path.exists(_tj), _refused)
_nl = os.path.join(C.ns_work(C.ACCEPTANCE_NS), "ledger.db")
check("13.5 path -> namespace: the new storage dir AND the voided dir are acceptance-v2 (fail-closed study guard, paid namespace)",
      C.namespace_of_dir("acceptance-v2-restart1") == C.namespace_of_dir("acceptance-v2") == C.ACCEPTANCE_NS and C.namespace_of_dir("nope") is None
      and AC.guarded_namespace(_nl) == AC.guarded_namespace(os.path.join(_VOID_DIRS[1], "ledger.db")) == C.ACCEPTANCE_NS
      and AC.paid_namespace_of_repo(_nl) == C.ACCEPTANCE_NS and AC.guarded_namespace(os.path.join(C.ns_work("test-v1"), "ledger.db")) is None)
_sd = os.path.join(_t13, "docs", "final-test", "matrix", "acceptance-v2-restart1")
os.makedirs(_sd, exist_ok=True)
_p0 = AC.journal_write_problems(os.path.join(_sd, "plan.jsonl"))
with open(os.path.join(_sd, AC.VERDICT_NAME), "w") as _f13:
    _f13.write("{}")
_p1 = AC.journal_write_problems(os.path.join(_sd, "plan.jsonl"))
check("13.6 the closed-attempt journal check recognises the new storage dir (a recorded verdict there refuses every write)",
      _p0 == [] and bool(_p1) and "CLOSED" in _p1[0], (_p0, _p1))
_reg13 = A.registration()
check("13.7 registration + roster unchanged (self sha fef2bd3e..., the roster sha the authorization binds)",
      _reg13["sha256"].startswith("fef2bd3e") and A.roster_sha(AC.acceptance_roster(None, _reg13)) == json.load(
          open(os.path.join(ROOT, "docs", "final-test", "acceptance_authorization.json"), encoding="utf-8"))["roster_sha"])
shutil.rmtree(_t13, ignore_errors=True)

# ---- 14. ACC-V2 correction cycle, criterion (c): a template record resolves to ITS frozen definition (not by the spec: prefix)
_inv14 = A.source_inventory()
_t14 = _inv14["templates"]
_b14 = {"id": _t14["ids"][0], "component": "template", "source": "spec:" + _t14["sha16"],
        "authored_by": "script:ltm_templates@" + _t14["module16"], "evidence": [], "confidence": 1.0}
check("14.1 the frozen template file is pinned (spec:0abc83fba8b923df, 9 template ids)",
      _t14["sha16"] == "0abc83fba8b923df" and len(_t14["ids"]) == 9, _t14)
check("14.2 a template record of the frozen file resolves", A.provenance_problem(_b14, _inv14) is None, A.provenance_problem(_b14, _inv14))
for _k14, _v14 in (("id", "template/*/not-a-template"), ("source", "spec:0000000000000000"), ("authored_by", "script:ltm_templates@0000000000000000"),
                   ("component", "procedure"), ("evidence", ["kb:x"]), ("evidence", None), ("evidence", ""), ("evidence", {})):
    _y14 = dict(_b14, **{_k14: _v14})
    check(f"14.3 a template record with a foreign {_k14} is refused", A.provenance_problem(_y14, _inv14) is not None, _y14)
_cap14 = os.path.join(ROOT, "docs", "final-test", "matrix", "acceptance-v2-restart1", "capture.jsonl")
if os.path.isfile(_cap14):
    _n14, _bad14 = 0, []
    for _l14 in open(_cap14, encoding="utf-8"):
        _r14 = json.loads(_l14)
        _bk14 = (_r14.get("blocks") or {}).get("S*") or {}
        _pv14 = {x.get("id"): x for x in (_bk14.get("package") or {}).get("provenance") or [] if isinstance(x, dict)}
        for _i14 in _bk14.get("ids") or []:
            _x14 = _pv14.get(_i14) or {}
            if str(_x14.get("source", "")).startswith("spec:"):
                _n14 += 1
                if A.provenance_problem(_x14, _inv14):
                    _bad14.append(_i14)
    check("14.4 the six attempt-1 template records (the (c) reasons) now resolve to their frozen definitions", _n14 == 6 and not _bad14, (_n14, _bad14))

# ---- 15. A34-A2 (signed 2026-10-15): an override counts for (d) only as a recorded obs_drift; obs_drift is reported, never gated
print("== 15 A34-A2: (d) override consistency + the descriptive obs_drift report")
from final_test import reset_gate as RG15                  # noqa: E402
_br15 = [{"kind": "seen_types", "only_a": [], "only_b": ["Stool"]}]
_ob15 = {"schema": "obs-pipeline/1", "treatment_id": "7" * 64, "sweep_prims": 12,
         "health": {"detect_calls": 8, "state_calls": 8, "state_errors": 0, "configs": [["auto", "p.pt", "s.pt", True, 0.25]]}}
_dr15 = {"amendment": "A34-A2", "kinds": ["seen_types"], "types_only_a": [], "types_only_b": ["Stool"], "summary_rows_only_a": [],
         "summary_rows_only_b": [], "n_rows_differing": 0, "coord_deltas": [], "max_coord_delta_m": 0.0, "breaches": _br15, "n_breaches": 1,
         "summary_sha_a": "a" * 64, "summary_sha_b": "a" * 64, "settings": RG15.pipeline_settings(_ob15), "positional_diffs": []}
_ov15 = {"ok": True, "registered_ok": False, "override": True, "breaches": _br15, "obs_drift": _dr15, "obs_pipeline": _ob15}
check("15.1 an attempt-1 style row (no override fields) is an ordinary pass", A._a34_override_problem({"ok": True, "breaches": []}) is None)
check("15.2 a complete A34-A2 override is accepted", A._a34_override_problem(_ov15) is None)
for _nm15, _bad15 in (("no obs_drift", dict(_ov15, obs_drift=None)), ("foreign amendment", dict(_ov15, obs_drift=dict(_dr15, amendment="X"))),
                      ("non-observation breach", dict(_ov15, breaches=[{"kind": "continuous.pos"}])), ("empty breaches", dict(_ov15, breaches=[])),
                      ("override False", dict(_ov15, override=False)), ("registered_ok True + drift", dict(_ov15, registered_ok=True)),
                      ("holding breach", dict(_ov15, breaches=_ov15["breaches"] + [{"kind": "told_holding"}])),
                      ("bare obs_drift", dict(_ov15, obs_drift={"amendment": "A34-A2"})),
                      ("drift without the differing types", dict(_ov15, obs_drift=dict(_dr15, types_only_b=[]))),
                      ("no pipeline record", dict(_ov15, obs_pipeline=None)),
                      ("unhealthy pipeline", dict(_ov15, obs_pipeline=dict(_ob15, health=dict(_ob15["health"], state_errors=2)))),
                      ("pipeline settings not the drift's", dict(_ov15, obs_pipeline=dict(_ob15, sweep_prims=13)))):
    check(f"15.3 an inconsistent override is a (d) problem: {_nm15}", A._a34_override_problem(_bad15) is not None, _bad15)
_arms15 = list(C.MAIN_ARMS)
_pres15 = [f"p{i}" for i in range(4)]
_want15 = {f"{a}|{p}": (a, p) for a in _arms15 for p in _pres15}
_exe15 = {k: ({"state_check": (_ov15 if (a == _arms15[0] and p == "p0") else {"ok": True})}, {"goal_ok": p != "p3"}) for k, (a, p) in _want15.items()}
_rep15 = A.obs_drift_report(_exe15, {"p0": ({"state_check": _ov15}, {})}, _want15, _arms15, _pres15)
_c15 = f"{_arms15[0]} vs {_arms15[1]}"
check("15.4 obs_drift report: counts per arm + P0, drift-excluded SR, common no-drift subset with denominators",
      _rep15["counts"][_arms15[0]] == 1 and _rep15["counts"][_arms15[1]] == 0 and _rep15["counts"]["P0"] == 1
      and _rep15["sr_drift_excluded"][_arms15[0]] == {"sr": round(2 / 3, 4), "n": 3} and _rep15["sr_drift_excluded"][_arms15[1]] == {"sr": 0.75, "n": 4}
      and _rep15["sr_common_no_drift_subset"][_c15] == {"n": 3, "sr_a": round(2 / 3, 4), "sr_b": round(2 / 3, 4)}, _rep15)
check("15.5 the obs_drift report is in decide()'s output and labelled descriptive / not gated", "obs_drift_descriptive_not_gated" in open(
      A.__file__, encoding="utf-8").read() and "not gated" in _rep15["label"])

AC.disarm_session()
print(f"\n{N['ok']}/{N['ok'] + N['fail']} checks passed")
sys.exit(0 if N["fail"] == 0 else 1)
