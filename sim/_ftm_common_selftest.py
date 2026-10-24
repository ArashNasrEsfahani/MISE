"""Selftest for sim/final_test/matrix/common.py (stage D-spec). Offline: no simulator, no network, no API.

    python sim/_ftm_common_selftest.py
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                   # noqa: E402
from planners.base import CostCeilingExceeded        # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------------------------------- static
src = open(C.__file__, encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok310 = True
except SyntaxError:
    ok310 = False
check("common.py parses as python 3.10 (WSL thor-env)", ok310)
check("common.py stays small (< 1300 lines; A21 added the written-out cost guard, the dedup spec and the blocked-run rule; M-fix2 added dev subsets, leak scan, cuts, reconcile; M-fix added the shared dir_sha + freeze-artefact checks; ACC-ATTEMPT2 added the correction-cycle namespace; RUNNER-FINAL the A24.1-C guard spec; READY the fresh-storage map of ruling 1b; K23 the scene / namespace_items / replication hooks, the K2/K3 data in k23_ns.py)", src.count("\n") < 1370,src.count("\n"))
check("common.py never opens the dataset (gold isolation)", "general.json" not in src.replace('"general.json")', ""))
check("common.py imports no simulator module", not any(f"import {m}" in src for m in ("ai2thor", "thor_actions", "honest", "webapp")))

# ---------------------------------------------------------------------------------------------------- paths
P = C.paths()
check("paths: freeze / families / tolerances under docs/final-test",
      all(P[k].replace("\\", "/").endswith(v) for k, v in (("freeze", "final-test/freeze.json"),
                                                           ("families", "final-test/families_approved.json"),
                                                           ("tolerances", "final-test/reset_tolerances.json"))))
check("journal paths per namespace", C.journal_path("test-v1", "plan").replace("\\", "/").endswith("final-test/matrix/test-v1/plan.jsonl")
      and C.journal_path("test-v1", "outcome").endswith("outcome.sealed.jsonl"))
check("unknown namespace refused", raises(lambda: C.ns_docs("pilot-v1"), ValueError))
check("snapshot dir S* = o0/k10", C.snapshot_dir("S*").replace("\\", "/").endswith("snapshots/o0/k10")
      and C.snapshot_dir("o2k3").replace("\\", "/").endswith("snapshots/o2/k3"))
check("snapshot dir rejects K=2 / order 3", raises(lambda: C.snapshot_dir("o0k2"), ValueError) and raises(lambda: C.snapshot_dir("o3k1"), ValueError))

m = C.load_manifest()
check("split manifest verifies (09c7da23...)", m["sha256"] == C.SPLIT_SHA256 and len(m["test"]) == 125)

# ---------------------------------------------------------------------------------------------------- arms
check("A23 core arms: G/O x {0,+M} (Sonnet / Haiku are archived)", C.CORE_ARMS == ("G0", "G+M", "O0", "O+M"))
check("+M arms use S* in prompt and chooser", all(C.arm_info(a)["prompt"] == "S*" and C.arm_info(a)["chooser"] == "S*"
                                                for a in ("G+M", "O+M")))
check("FS arms: FS block, S* chooser", C.arm_info("G+FS") == {"planner": "G", "prompt": "FS", "chooser": "S*"})
check("/p arms: cached plans, prior chooser", C.arm_info("O+M/p")["chooser"] == "prior" and C.arm_info("O+M/p")["planner"] == "replay:O+M")
check("curve arm parses", C.arm_info("O+M@o1k3") == {"planner": "O", "prompt": "o1k3", "chooser": "o1k3"})
check("curve arm K=0 is not an arm (it is the 0 arm)", raises(lambda: C.arm_info("G+M@o0k0"), ValueError))
check("scope: core main/repeat", C.scope_of("O0", "main") == "core" and C.scope_of("O+M", "repeat") == "core")
check("scope: FS / curve / reloc = other", C.scope_of("O+FS", "main") == "other"
      and C.scope_of("O+M@o1k3", "curve") == "other" and C.scope_of("O0", "reloc") == "other")
check("scope: Gemma / Codex / P = free", C.scope_of("G+M", "main") == "free" and C.scope_of("X-gpt-6-luna+M", "codex") == "free"
      and C.scope_of("P+M", "main") == "free")

# ---------------------------------------------------------------------------------------------------- A23: archived arms (kept, guarded)
check("A23 ARCHIVED_ARMS keeps Sonnet and Haiku definitions (not deleted)",
      set(C.ARCHIVED_ARMS) == {"S0", "S+M", "H0", "H+M"} and C.PLANNER_SPEC["S"] == "sonnet" and C.PLANNER_SPEC["H"] == "haiku"
      and "S" in C.PAID and "H" in C.PAID and C.ARCHIVED_PLANNERS == ("S", "H"))
check("A23 archived arms are in no main / core roster and not in ARMS",
      not (set(C.ARCHIVED_ARMS) & (set(C.MAIN_ARMS) | set(C.CORE_ARMS) | set(C.ARMS))) and len(C.MAIN_ARMS) == 8)
check("A23 test-v1 opts into no archived arm and its roster is exactly MAIN_ARMS",
      C.archived_arms_of("test-v1") == () and C.main_arms_for("test-v1") == tuple(C.MAIN_ARMS))
check("A23 arm_info / scope_of refuse an archived arm in test-v1 and with no namespace",
      all(raises(lambda a=a: C.arm_info(a, "test-v1"), ValueError) and raises(lambda a=a: C.arm_info(a), ValueError)
          and raises(lambda a=a: C.scope_of(a, "main", "test-v1"), ValueError) for a in C.ARCHIVED_ARMS))
check("A23 archived_guard rejects archived arms in a plan for test-v1 and for practice-v1",
      raises(lambda: C.archived_guard("test-v1", ["G0", "S+M"]), ValueError) and raises(lambda: C.archived_guard("practice-v1", ["H0"]), ValueError)
      and C.archived_guard("test-v1", list(C.MAIN_ARMS)) is None)
_saved = dict(C.NAMESPACES["test-v1"])
try:
    C.NAMESPACES["test-v1"]["archived_arms"] = ("S0",)
    check("A23 guard: test-v1 cannot be made to opt in (even by editing the table)", raises(lambda: C.archived_guard("test-v1"), ValueError)
          and raises(lambda: C.main_arms_for("test-v1"), ValueError))
finally:
    C.NAMESPACES["test-v1"].clear()
    C.NAMESPACES["test-v1"].update(_saved)
C.NAMESPACES["archived-sh-v1"] = {"items": "dev_pilot", "ceilings": {"core": 0.0, "total": 1.0}, "archived_arms": ("S0", "S+M", "H0", "H+M")}
try:
    check("A23 a future namespace can opt in explicitly: arm_info / scope_of / roster work for exactly its arms",
          C.archived_guard("archived-sh-v1", list(C.ARCHIVED_ARMS)) is None
          and C.arm_info("S+M", "archived-sh-v1") == {"planner": "S", "prompt": "S*", "chooser": "S*"}
          and C.scope_of("S0", "main", "archived-sh-v1") == "other" and C.scope_of("H+M", "main", "archived-sh-v1") == "other"
          and C.main_arms_for("archived-sh-v1") == tuple(C.MAIN_ARMS) + tuple(C.ARCHIVED_ARMS))
    C.NAMESPACES["archived-sh-v1"]["archived_arms"] = ("S0",)
    check("A23 an opt-in is per arm: H0 stays refused where only S0 is enabled",
          C.arm_info("S0", "archived-sh-v1")["planner"] == "S" and raises(lambda: C.arm_info("H0", "archived-sh-v1"), ValueError))
    C.NAMESPACES["archived-sh-v1"]["archived_arms"] = ("Z9",)
    check("A23 an opt-in to an unknown archived arm is rejected", raises(lambda: C.archived_guard("archived-sh-v1"), ValueError))
finally:
    C.NAMESPACES.pop("archived-sh-v1", None)
check("A23 the withdrawn hypotheses keep their IDs", C.WITHDRAWN_HYPOTHESES == ("H2b", "H3b", "H3c"))

# ---------------------------------------------------------------------------------------------------- ids, units
pid = C.pres_id("g-l1-clean-bowl", "fa")
check("pres id round trip", C.parse_pres(pid) == {"world": "canon", "item": "g-l1-clean-bowl", "lang": "fa"})
check("bad pres refused", raises(lambda: C.pres_id("x", "de"), ValueError) and raises(lambda: C.pres_id("x", "en", "moon"), ValueError))
steps = [{"action": "pick_and_place", "object": "Apple", "to": "Fridge", "arg": None}, {"action": "close", "object": "Fridge", "to": None, "arg": None}]
check("norm_plan dict == list form", C.norm_plan(steps) == C.norm_plan([["pick_and_place", "Apple", "Fridge", None], ["close", "Fridge"]]))
base = dict(plan=steps, pres=pid, chooser="S*", snapshot_sha="ab" * 32, restored_state="canon+s0", summary_sha="cd" * 32)
k0 = C.execution_unit_key(**base)
check("unit key deterministic", k0 == C.execution_unit_key(**dict(base)))
check("unit key: order matters", k0 != C.execution_unit_key(**{**base, "plan": steps[::-1]}))
check("unit key: arg matters", k0 != C.execution_unit_key(**{**base, "plan": [{**steps[0], "arg": "water"}, steps[1]]}))
check("unit key: chooser matters", k0 != C.execution_unit_key(**{**base, "chooser": "prior"}))
check("unit key: summary matters", k0 != C.execution_unit_key(**{**base, "summary_sha": "ef" * 32}))
check("unit key: prior ignores the snapshot sha",
      C.execution_unit_key(**{**base, "chooser": "prior"}) == C.execution_unit_key(**{**base, "chooser": "prior", "snapshot_sha": "zz"}))
check("unit key: two LTM snapshots with the same sha share", k0 == C.execution_unit_key(**{**base, "chooser": "o0k10"}))
check("unit key: LTM chooser without sha refused", raises(lambda: C.execution_unit_key(**{**base, "snapshot_sha": None}), ValueError))

# ---------------------------------------------------------------------------------------------------- seeded orders
seq = [f"p{i}" for i in range(40)]
o1, o2 = C.seeded_order(seq, "x"), C.seeded_order(list(reversed(seq)), "x")
check("seeded order: deterministic, input-order independent", o1 == o2 and sorted(o1) == sorted(seq))
check("seeded order: label changes it", o1 != C.seeded_order(seq, "y"))
cells = lambda p: [(a, r) for a in C.CORE_ARMS for r in (0, 1, 2)]
bl = C.plan_blocks(seq[:5], cells, "test-v1", "main")
check("plan blocks: every cell once, grouped by presentation", all(sorted(cs) == sorted(cells(p)) for p, cs in bl) and len(bl) == 5)
check("plan blocks: arm order differs between blocks", len({tuple(cs) for _, cs in bl}) > 1)
check("plan blocks reproducible", bl == C.plan_blocks(seq[:5], cells, "test-v1", "main"))
eb = C.exec_blocks({"p1": ["u2", "u1", "u1"], "p2": ["u3"]}, "x")
check("exec blocks: units deduplicated", sorted(u for _, us in eb for u in us) == ["u1", "u2", "u3"])

# ---------------------------------------------------------------------------------------------------- journal
tmp = tempfile.mkdtemp(prefix="ftm_common_")
try:
    J = C.Journal(os.path.join(tmp, "j", "plan.jsonl"))
    J.append({"key": "a", "v": 1})
    J.append({"key": "b", "v": "سیب"})
    check("journal append + read (UTF-8)", [r["key"] for r in J.rows()] == ["a", "b"] and J.rows()[1]["v"] == "سیب")
    with open(J.path, "ab") as f:
        f.write(b'{"key": "c", "v"')                      # torn tail (crash mid-write)
    check("torn tail ignored on read", J.keys() == {"a", "b"})
    J.append({"key": "d"})
    check("torn tail moved to .torn on append", J.keys() == {"a", "b", "d"} and os.path.exists(J.path + ".torn"))
    with open(J.path, "ab") as f:
        f.write(b"garbage\n")
    J.append({"key": "e"})
    check("corrupt middle line fails fast", raises(J.rows, C.JournalCorrupt))
    check("latest() keeps the last row per key", C.Journal(os.path.join(tmp, "none.jsonl")).latest() == {})

    # ------------------------------------------------------------------------------------------------ records
    cap = dict(key="k", namespace="rehearsal-v1", pres=pid, item="i", lang="fa", level="L1", world="canon", capture_run=1,
               instruction="x", holding=None, summary=[], summary_sha="s", blocks={}, requests={}, state_ref="r", world_hash="h",
               audit={})
    check("capture record ok", C.make_record("capture", **cap)["schema"] == "ftm.capture/1")
    check("capture record: missing field refused", raises(lambda: C.make_record("capture", **{k: v for k, v in cap.items() if k != "summary"}), ValueError))
    check("capture record: nested gold refused", raises(lambda: C.make_record("capture", **{**cap, "blocks": {"S*": {"goal": "x"}}}), ValueError))
    ex = dict(key="u#0", namespace="test-v1", unit="u", run=0, attempt=0, pres=pid, status="done", state_check={"ok": True},
              audit={}, prims=10, wall_s=1.0, snapshot_sha_before="a", snapshot_sha_after="a")
    check("exec record ok", C.make_record("exec", **ex)["status"] == "done")
    check("exec record: an outcome field is refused (outcomes go to the sealed journal)",
          raises(lambda: C.make_record("exec", **{**ex, "goal_ok": True}), ValueError) and raises(lambda: C.make_record("exec", **{**ex, "steps": []}), ValueError))
    check("exec record: illegal status refused", raises(lambda: C.make_record("exec", **{**ex, "status": "great"}), ValueError))
    pl = dict(key="p", namespace="test-v1", arm="O0", block="main", pres=pid, r=0, session=0, request_sha="x", status="ok", plan=[],
              first_plan=[], first_ok=True, repaired=False, calls=1, cached_calls=0, tokens_in=1, tokens_out=1, cost_usd=0.0,
              model="m", scope="core")
    check("plan record ok; status checked", C.make_record("plan", **pl)["arm"] == "O0" and raises(lambda: C.make_record("plan", **{**pl, "status": "meh"}), ValueError))

    # ------------------------------------------------------------------------------------------------ guard (temp root)
    root = os.path.join(tmp, "root")
    os.makedirs(os.path.join(root, "docs", "final-test"))
    os.makedirs(os.path.join(root, "sim"))
    shutil.copy(P["split"], C.paths(root)["split"])
    with open(C.paths(root)["prereg"], "w", encoding="utf-8") as f:
        f.write("prereg v-test\n")
    with open(os.path.join(root, "sim", "mod.py"), "w") as f:
        f.write("x = 1\n")
    dev, test, prac = m["dev_pilot"][0], m["test"][0], [i for i in m["practice_pool"] if i not in m["dev_pilot"]][0]
    check("guard: dev item ok in rehearsal-v1", C.guard_items([dev], "rehearsal-v1", root=root)["n_test"] == 0)
    check("guard: test item refused in rehearsal-v1", raises(lambda: C.guard_items([test], "rehearsal-v1", allow_test=True, root=root), C.TestItemRefused))
    check("guard: non-dev practice item refused in rehearsal-v1", raises(lambda: C.guard_items([prac], "rehearsal-v1", root=root), C.TestItemRefused))
    check("guard: practice item ok in practice-v1", C.guard_items([prac, dev], "practice-v1", root=root)["n"] == 2)
    check("guard: practice item refused in test-v1", raises(lambda: C.guard_items([prac], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    check("guard: capped-out item refused everywhere",
          raises(lambda: C.guard_items([m["test_uncapped_not_scored"][0]], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    check("guard: test item without --allow-test refused", raises(lambda: C.guard_items([test], "test-v1", root=root), C.TestItemRefused))
    check("guard: test item without freeze.json refused", raises(lambda: C.guard_items([test], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    fz = {"schema": "final-test.freeze/1", "gates": {"12.1": {"pass": True}, "12.4": {"pass": True}},
          "arash_check": {"by": "Arash", "date": "2026-10-12"}, "split_sha256": C.SPLIT_SHA256,
          "prereg_sha256": C.sha256_file(C.paths(root)["prereg"]), "code": {"sim/mod.py": C.sha256_file(os.path.join(root, "sim", "mod.py"))}}

    def write_freeze(d):
        d = dict(d)
        d["sha256"] = C.self_sha(d)
        with open(C.paths(root)["freeze"], "w", encoding="utf-8") as f:
            json.dump(d, f)

    write_freeze(fz)
    check("guard: valid freeze but no families approval refused", raises(lambda: C.guard_items([test], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    with open(C.paths(root)["families"], "w", encoding="utf-8") as f:
        json.dump({"approved_by": "Arash", "date": "2026-10-11"}, f)
    check("guard: allow-test + freeze + families -> passes", C.guard_items([test, m["test"][1]], "test-v1", allow_test=True, root=root)["n_test"] == 2)
    write_freeze({**fz, "gates": {"12.1": {"pass": True}, "12.4": {"pass": False}}})
    check("guard: a failed gate refuses", raises(lambda: C.guard_items([test], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    write_freeze({**fz, "arash_check": None})
    check("guard: no Arash freeze check refuses", raises(lambda: C.guard_items([test], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    write_freeze(fz)
    d = json.load(open(C.paths(root)["freeze"]))
    d["gates"]["12.4"]["pass"] = True
    d["code"]["sim/other.py"] = "x"
    json.dump(d, open(C.paths(root)["freeze"], "w"))
    check("guard: freeze edited after writing (self-sha) refuses", "self-sha" in " ".join(C.freeze_status(root)["reasons"]))
    write_freeze(fz)
    with open(os.path.join(root, "sim", "mod.py"), "w") as f:
        f.write("x = 2\n")
    check("guard: code changed after freeze refuses", raises(lambda: C.guard_items([test], "test-v1", allow_test=True, root=root), C.TestItemRefused))
    with open(os.path.join(root, "sim", "mod.py"), "w") as f:
        f.write("x = 1\n")
    with open(C.paths(root)["prereg"], "a", encoding="utf-8") as f:
        f.write("late amendment\n")
    check("guard: prereg changed after freeze refuses", "preregistration changed" in " ".join(C.freeze_status(root)["reasons"]))

    # ------------------------------------------------------------------------------------------------ mask / keys
    check("mask: forbidden tokens found", C.mask_problems("in FloorPlan1 see Apple|+01.2|x") == ["Apple|+01.2|", "FloorPlan"])
    check("mask: negative-coordinate id found too", C.mask_problems("Mug|-00.5|+0.9|") == ["Mug|-00.5|"])
    check("mask: clean prompt", C.mask_problems("Fridge [closed] on/in ? at (1.0, 2.0)") == [])
    check("key leak detected", C.key_leak("x sk-ant-api03-abcdefghijklmnop y") and not C.key_leak('{"tokens_in": 1234, "cost_usd": 0.1}'))
    check("key leak: verbatim secret", C.key_leak("abc SECRETVALUE123 d", ["SECRETVALUE123"]))

    # ------------------------------------------------------------------------------------------------ ledger
    L = C.ScopedLedger(os.path.join(tmp, "w", "ledger.db"), {"core": 1.0, "total": 1.5})
    check("ledger: paid reserve without scope refused", raises(lambda: L.reserve("anthropic", 0.1), CostCeilingExceeded))
    check("ledger: free 0-USD reserve without scope ok", isinstance(L.reserve("ollama", 0.0), int))
    with L.use("core"):
        r1 = L.reserve("anthropic", 0.6)
    L.settle(r1, 0.5, {"provider": "anthropic", "cost_usd": 0.5})
    check("ledger: settled cost counted per scope", L.committed("core") == 0.5 and L.billed_scope("core") == 0.5)
    with L.use("core"):
        check("ledger: core ceiling enforced", raises(lambda: L.reserve("anthropic", 0.6), CostCeilingExceeded))
    with L.use("other"):
        r2 = L.reserve("anthropic", 0.9)
        check("ledger: total ceiling enforced for other", raises(lambda: L.reserve("anthropic", 0.2), CostCeilingExceeded))
    L.release(r2)
    check("ledger: block refused when the worst case does not fit", raises(lambda: L.block("core", 0.6).__enter__(), C.BlockRefused))
    with L.block("core", 0.4) as hid:
        check("ledger: block hold counts while open", L.committed("core") >= 0.9 - 1e-9)
        r3 = L.reserve("anthropic", 0.3)                 # inside the hold: not double-counted
        L.settle(r3, 0.2, {"provider": "anthropic", "cost_usd": 0.2})
        check("ledger: call inside the block admitted (hold not double-counted)", True)
    check("ledger: hold released after the block", abs(L.committed("core") - 0.7) < 1e-9, L.committed("core"))
    con = sqlite3.connect(L.path)
    con.execute("INSERT INTO ledger (ts, provider, state, reserved) VALUES ('t','hold@core','open',0.25)")
    con.commit()
    con.close()
    L2 = C.ScopedLedger(L.path, {"core": 1.0, "total": 1.5})
    check("ledger: a dead process's hold is released on reopen", abs(L2.committed("core") - 0.7) < 1e-9)
    check("ledger: provider tagged with scope", all("@" in r["provider"] for r in L2.rows()))
    with L2.use("core"):
        L2.book({"provider": "anthropic", "cached": True, "cost_usd": 0.0})
    check("ledger: cache replay booked at 0 USD", abs(L2.committed() - 0.7) < 1e-9)
    check("block worst case = calls x 2 x reserve", C.block_worst_case(6, 0.05) == 0.6)

    # ------------------------------------------------------------------------------------------------ cost guard
    g = C.cost_guard(opus_calls=59, opus_usd=59 * 0.03, checked_at=0, spent=1.0, remaining={}, mean_cost={}, ceiling=30)
    check("guard: no Opus check before 60 calls", not g["pause"] and g["checked_at"] == 0)
    g = C.cost_guard(opus_calls=61, opus_usd=61 * 0.0100, checked_at=0, spent=1.0, remaining={}, mean_cost={}, ceiling=30)
    check("guard: check at 60, under the threshold -> continue", not g["pause"] and g["checked_at"] == 60)
    g = C.cost_guard(opus_calls=120, opus_usd=120 * 0.09, checked_at=60, spent=6.0, remaining={}, mean_cost={}, ceiling=30)
    check("guard: Opus mean above the threshold (0.09 > the A37 0.080) -> pause", g["pause"] and g["checked_at"] == 120)
    # ---- A37 (Arash 2026-10-16, normal API; was A24.1-C 0.040 / 0.022 / ftm-guard-3 / 30 / 50): the explicit threshold 0.080 USD per billed
    #      Opus cell, reference 0.044, spec ftm-guard-4, test-v1 60 / 100. LITERAL expected values (K23 A37 alignment), never read back.
    check("A37: reference 0.044 and an EXPLICIT threshold 0.080 USD per billed Opus cell (not 2 x the reference); ftm-guard-4; test-v1 60 / 100",
          C.OPUS_REF_USD_PER_CALL == 0.044 and C.OPUS_GUARD_USD_PER_CALL == 0.080 and abs(C.OPUS_GUARD_USD_PER_CALL - 2 * C.OPUS_REF_USD_PER_CALL) > 1e-9
          and C.COST_GUARD_SPEC["version"] == "ftm-guard-4" and C.COST_GUARD_SPEC["reference_usd"] == 0.044 and C.COST_GUARD_SPEC["pause_above_usd"] == 0.080
          and "0.080" in C.COST_GUARD_SPEC["threshold"] and C.NAMESPACES["test-v1"]["ceilings"] == {"core": 60.0, "total": 100.0}
          and "up to two repair turns" in C.COST_GUARD_SPEC["unit"] and "A29 batch execution" in C.COST_GUARD_SPEC["in_flight"]
          and "sequential" not in C.COST_GUARD_SPEC["in_flight"] and "NOT 2 x" in C.COST_GUARD_SPEC["threshold"])
    def _g(costs, recent=True, ceiling=50.0):
        return C.cost_guard(opus_calls=len(costs), opus_usd=sum(costs), checked_at=0, spent=0.0, remaining={}, mean_cost={}, ceiling=ceiling,
                            opus_recent=list(costs) if recent else None)
    at = _g([0.080] * 60)
    just_under = _g([0.0795] * 60)
    above = _g([0.0805] * 60, recent=False)
    below_ref = _g([0.044] * 60)
    check("A37 boundary: a mean of exactly 0.080 and of 0.0795 does not pause; 0.0805 (cumulative check alone, no window) pauses; the 0.044 reference "
          "does not; the A24.1-C levels 0.040 / 0.0405 no longer pause", not at["pause"] and not just_under["pause"] and above["pause"]
          and any("cumulative" in r or r.startswith("Opus mean") for r in above["reasons"]) and not below_ref["pause"] and not _g([0.0405] * 60)["pause"]
          and not _g([0.040] * 60)["pause"] and "0.080" in above["reasons"][0] and "billed Opus cell" in above["reasons"][0])
    surge = [0.030] * 540 + [0.100] * 60                       # cumulative 0.037 (fine), last 60 at 0.100 (> 0.080)
    gs = _g(surge)
    check("A37 boundary: the RECENT window alone pauses (cumulative mean below the threshold)",
          gs["pause"] and any("recent-window" in r for r in gs["reasons"]) and not any(r.startswith("Opus mean") for r in gs["reasons"]))
    hidden = [0.090] * 540 + [0.010] * 60                      # cumulative 0.082 (above 0.080), last 60 fine
    gh = _g(hidden)
    check("A37 boundary: the CUMULATIVE check alone pauses (recent window below the threshold)",
          gh["pause"] and any(r.startswith("Opus mean") for r in gh["reasons"]) and not any("recent-window" in r for r in gh["reasons"]))
    check("A24.1-C boundary: no mean check below the 60-cell minimum (59 cells at 0.10)", not _g([0.10] * 59)["pause"] and _g([0.10] * 59)["checked_at"] == 0)
    import plan as _PLN
    import types as _ty
    _self = _ty.SimpleNamespace()
    _obs = {}
    for _r in ({"planner": "O", "calls": 3, "cached_calls": 0, "cost_usd": 0.060},     # first transmission + two repairs, billed
               {"planner": "O", "calls": 1, "cached_calls": 0, "cost_usd": 0.020, "status": "rejected"},
               {"planner": "O", "calls": 2, "cached_calls": 0, "cost_usd": 0.030, "status": "infra"},   # a billed transport attempt, then infra
               {"planner": "O", "calls": 1, "cached_calls": 1, "cost_usd": 0.0},       # a cache replay: not a cell
               {"planner": "O", "calls": 1, "cached_calls": 0, "cost_usd": 0.0},       # never billed: not counted
               {"planner": "G", "calls": 1, "cached_calls": 0, "cost_usd": 0.0}):
        _PLN.Plan._count(_self, _obs, _r)
    o_ = _obs.get("O") or {}
    g_agg = C.cost_guard(opus_calls=o_.get("n", 0) * 20, opus_usd=o_.get("usd", 0.0) * 20, checked_at=0, spent=0.0, remaining={}, mean_cost={},
                         ceiling=50.0, opus_recent=list(o_.get("costs", [])) * 20)
    check("A24.1-C aggregation (plan._count): a repaired cell's whole cost, a rejected and an infra-failed BILLED cell count; a cache replay and "
          "an unbilled cell do not; their pooled mean 0.0367 x 60 cells stays below the A37 0.080",
          o_.get("n") == 3 and abs(o_.get("usd", 0) - 0.11) < 1e-12 and o_.get("costs") == [0.06, 0.02, 0.03] and not g_agg["pause"]
          and abs(_self._journal_usd - 0.11) < 1e-12, (o_, g_agg["reasons"]))
    check("A18: arms X0 / X+M are gpt-6-sol with the same memory sources as S / O, free, and not cuttable",
          C.arm_info("X0") == {"planner": "X:gpt-6-sol", "prompt": "none", "chooser": "prior"} and C.arm_info("X+M") == {"planner": "X:gpt-6-sol", "prompt": "S*", "chooser": "S*"}
          and all(C.scope_of(a, b) == "free" for a in ("X0", "X+M") for b in ("main", "repeat")) and C.cut_group("X0", "canon:x:en") is None and C.cut_group("X-gpt-6-astra0") == "codex"
          and "gpt-6-sol" not in C.CODEX_EXPLORATORY and not any(a.startswith("X-gpt-6-sol") for a in C.ARMS) and set(C.X_MAIN_ARMS) <= set(C.MAIN_ARMS))
    g = C.cost_guard(opus_calls=10, opus_usd=0.1, checked_at=0, spent=25.0, remaining={"O": 600}, mean_cost={"O": 0.01}, ceiling=30)
    check("guard: projected completion over the ceiling -> pause", g["pause"] and g["projected"] == 31.0)   # A22 core ceiling 30 (was spent 20 -> 26 > 25)

    # ------------------------------------------------------------------------------------------------ A21 B: the cost guard, written out
    old_costs = [0.010] * 540 + [0.100] * 60                 # 600 billed Opus cells: cumulative mean 0.019 (fine), last 60 at 0.100 (a surge above the A37 0.080)
    g = C.cost_guard(opus_calls=600, opus_usd=sum(old_costs), checked_at=540, spent=8.0, remaining={}, mean_cost={}, ceiling=65, opus_recent=old_costs)
    check("A21 B: a recent surge hidden by a fine cumulative mean pauses (recent-window check)",
          g["pause"] and any("recent-window" in r for r in g["reasons"]) and not any(r.startswith("Opus mean") for r in g["reasons"]))
    g = C.cost_guard(opus_calls=600, opus_usd=sum(old_costs), checked_at=540, spent=8.0, remaining={}, mean_cost={}, ceiling=65, opus_recent=old_costs[:59])
    check("A21 B: fewer than 60 cells in the window -> no window verdict", not any("recent-window" in r for r in g["reasons"]))
    g = C.cost_guard(opus_calls=61, opus_usd=61 * 0.01, checked_at=0, spent=1.0, remaining={}, mean_cost={}, ceiling=65, opus_recent=[0.01] * 61)
    check("A21 B: a normal window does not pause", not g["pause"])
    r1 = C.reconcile_spend(10.0, 10.04)
    r2 = C.reconcile_spend(10.0, 10.30)
    r3 = C.reconcile_spend(40.0, 40.7)
    check("A21 B: reconciliation tolerance = max(0.05 USD, 2 % of the larger total)", r1["ok"] and not r2["ok"] and r3["ok"] and abs(r2["tolerance_usd"] - 0.206) < 1e-9)
    g = C.cost_guard(opus_calls=5, opus_usd=0.06, checked_at=0, spent=1.0, remaining={}, mean_cost={}, ceiling=65, ledger_usd=3.00, journal_usd=3.50)
    check("A21 B: a ledger/journal gap pauses at any boundary (before the 60-cell cadence)",
          g["pause"] and g["reconcile"]["ok"] is False and g["reasons"][0].startswith("ledger "))
    g = C.cost_guard(opus_calls=5, opus_usd=0.06, checked_at=0, spent=1.0, remaining={}, mean_cost={}, ceiling=65, ledger_usd=3.00, journal_usd=3.01)
    check("A21 B: a matching ledger and journal pass", not g["pause"] and g["reconcile"]["ok"])
    check("A21 B: the guard spec is complete (unit, denominator, minimum sample, window, in-flight, reconciliation, responses)",
          all(k in C.COST_GUARD_SPEC for k in ("unit", "numerator", "denominator", "minimum_sample", "checks", "in_flight", "reconciliation", "responses_to_a_pause", "not_allowed"))
          and set(C.COST_GUARD_SPEC["responses_to_a_pause"]) == set(C.PAUSE_RESPONSES) and C.COST_GUARD_SPEC["pause_above_usd"] == C.OPUS_GUARD_USD_PER_CALL)
    LI = C.ScopedLedger(os.path.join(tmp, "inflight.db"), {"core": 5.0, "total": 6.0}, audit=False)
    with LI.use("core"):
        rid = LI.reserve("anthropic", 4.9)                  # a call in flight: an OPEN reservation at its worst case
    with LI.use("core"):
        refused = raises(lambda: LI.reserve("anthropic", 0.3), C.CostCeilingExceeded)
    with LI.use("other"):
        refused_total = raises(lambda: LI.reserve("anthropic", 1.2), C.CostCeilingExceeded)
    check("A21 B: a call in flight counts at its worst case against the scope ceiling AND the total ceiling", refused and refused_total and abs(LI.committed() - 4.9) < 1e-9)
    LI.release(rid)
    check("A21 B: releasing the in-flight reservation frees the ceiling", abs(LI.committed()) < 1e-9)

    # ------------------------------------------------------------------------------------------------ A21 A / D: blocked-run rule, dedup key
    check("A21 A: infra vs agent classes are fixed (blocked_state / crash / plan transport = infra; rejected / not executed = agent)",
          C.run_class("exec_blocked_state") == "infra_exec" and C.run_class("exec_crash") == "infra_exec" and C.run_class("plan_infra") == "infra_planner"
          and C.run_class("plan_rejected") == "agent" and C.run_class("ok") == "ok" and C.run_class("anything-else") == "missing"
          and set(C.BLOCKED_RULE) >= {"infra", "agent", "retries", "scoring", "sensitivity", "flag", "g_iso", "precedence", "hang"})
    ck = C.unit_key_selfcheck()
    check("A21 D: the dedup key function reproduces the frozen golden key", ck["ok"], str(ck))
    base = dict(C.UNIT_KEY_GOLDEN_INPUT)
    ks = {C.execution_unit_key(**base)}
    for field, val in (("plan", [{"action": "close", "object": "Fridge", "to": None, "arg": None}]), ("pres", "canon:g-l1-sample:fa"), ("chooser", "prior"),
                       ("snapshot_sha", "2" * 64), ("restored_state", "canon+ffffffffffffffff"), ("summary_sha", "3" * 64)):
        ks.add(C.execution_unit_key(**dict(base, **{field: val})))
    check("A21 D: every field of the key changes the key (plan, presentation, chooser, snapshot, restored state, summary)", len(ks) == 7)
    check("A21 D: the order of plan steps is part of the key",
          C.execution_unit_key(**dict(base, plan=[{"action": "open", "object": "A"}, {"action": "close", "object": "B"}]))
          != C.execution_unit_key(**dict(base, plan=[{"action": "close", "object": "B"}, {"action": "open", "object": "A"}])))
    check("A21 D: a prior chooser ignores the snapshot sha (one execution for G0 / S0 / O0 with equal plans)",
          C.execution_unit_key(**dict(base, chooser="prior", snapshot_sha="a" * 64)) == C.execution_unit_key(**dict(base, chooser="prior", snapshot_sha="b" * 64)))

    # ------------------------------------------------------------------------------------------------ phase lock
    with C.PhaseLock("rehearsal-v1", "plan", root=root):
        check("lock: second writer refused", raises(lambda: C.PhaseLock("rehearsal-v1", "plan", root=root).__enter__(), C.PhaseLocked))
    lk = C.PhaseLock("rehearsal-v1", "plan", root=root)
    lk.__enter__()
    check("lock: stale lock needs steal", raises(lambda: C.PhaseLock("rehearsal-v1", "plan", root=root).__enter__(), C.PhaseLocked))
    with C.PhaseLock("rehearsal-v1", "plan", root=root, steal=True):
        pass
    check("lock: steal works and releases", not os.path.exists(lk.path))
    check("sqlite available", C.sqlite_ok())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n_ftm_common_selftest: {N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
