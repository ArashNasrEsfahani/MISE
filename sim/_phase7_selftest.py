"""Pure Phase-7 loop-closure self-test (the D14 first-tier gate).

No AI2-THOR controller is started and no network is touched: ai2thor is stubbed at import time (like
_phase6_selftest.py), the reindex check points rag at an unreachable localhost port, and every store
(KnowledgeBase JSON, Memory sqlite, bandit sqlite, loop_reports.jsonl) lives in a temp dir that is
removed afterwards. Runs on plain Windows python (stdlib + numpy, which select.py needs).

Covers the Phase-7 design points end to end:
  D3  goal-wins success signal (conf 0.9) vs all-steps-ok (conf 0.6)
  D4  the EXECUTED (post-repair) steps are stored, not the raw LLM plan
  D5  failure -> template lesson from failed-step label + agent error
  D6  dedup upserts: insert-then-reinforce for BOTH procedures and lessons
  D7  old lessons persist after a later success; related ids recorded
  D8  auto procedures scope='global', auto lessons scoped to the scene
  D9  reindex is best-effort (unreachable Ollama / broken rag must not raise)
  D2  loop off => zero KB writes, no reindex, bandit reward() frozen
  D10 replay_procedure arm eligibility gated by procedure_match
  D12 all auto-writes source='environment'
  D13 loop report dict + jsonl trail shape
plus the audit-log lines every knowledge/bandit write must leave behind.

Prints a per-check PASS/FAIL table and exits non-zero on any failure.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import types

_results: list = []          # (name, ok, detail)


def _check(name: str, cond, detail: str = "") -> bool:
    ok = bool(cond)
    _results.append((name, ok, detail))
    return ok


def _stub_ai2thor():
    """Import-time placeholder so thor_actions (lazy in improve._error_hint) never pulls the real sim."""
    ai2thor = types.ModuleType("ai2thor")
    controller = types.ModuleType("ai2thor.controller")

    class Controller:  # never instantiated here
        pass

    controller.Controller = Controller
    ai2thor.controller = controller
    sys.modules.setdefault("ai2thor", ai2thor)
    sys.modules.setdefault("ai2thor.controller", controller)


def _load_select(here: str):
    """Load local sim/select.py by path (plain `import select` resolves to the stdlib module)."""
    spec = importlib.util.spec_from_file_location("phase7_select", os.path.join(here, "select.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_checks(tmp: str, here: str):
    import improve
    import knowledge
    import memory

    logging.getLogger("rag").disabled = True     # the D9 check trips rag's expected keyword-only fallback warning
    kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "kb.json"))
    mem = memory.Memory(path=os.path.join(tmp, "memory.sqlite"))
    reports = os.path.join(tmp, "reports.jsonl")
    scene = "FloorPlan1"
    n_journal = 0            # close_loop calls that should each journal exactly one jsonl line

    # ------------------------------------------------------------------ pure helpers (D3/D6 keys)
    _check("helpers: normalize_command",
           knowledge.normalize_command("  Pick  up the APPLE!! ") == "pick up the apple")
    _check("helpers: plan_signature ignores ids/aux keys",
           knowledge.plan_signature([{"action": "go_to", "object": "Apple", "to": None},
                                     {"action": "pickup", "object": "Apple", "arg": 1}])
           == (("go_to", "apple"), ("pickup", "apple")))
    g_win = improve.decide_success([{"ok": True}], {"applicable": True, "ok": False})
    _check("signal: applicable goal WINS at conf 0.9",
           g_win == {"success": False, "source": "goal", "confidence": 0.9}, str(g_win))
    s_ok = improve.decide_success([{"ok": True}, {"ok": True}], {"applicable": False})
    _check("signal: all-steps-ok at conf 0.6",
           s_ok == {"success": True, "source": "steps", "confidence": 0.6}, str(s_ok))
    _check("signal: empty step_results is a failure",
           improve.decide_success([], None)["success"] is False)
    # Pin the D3 spec ambiguity: an applicable goal verdict WINS even over a failed step (a held-object check
    # is stronger evidence than per-step action success), so a partially-broken run that verifies gets its
    # executed steps stored. A future 'any failed step = failure' refactor must fail this check, not slip by.
    _check("signal: applicable goal wins even over a failed step (D3 resolution pinned)",
           improve.decide_success([{"ok": False}], {"applicable": True, "ok": True})
           == {"success": True, "source": "goal", "confidence": 0.9})
    _check("KB has a write lock (concurrent-writer safety)", hasattr(kb, "_lock"))

    # ------------------------------------------------------------------ success -> procedure (goal source)
    cmd_apple = "pick up the apple"
    raw_plan = [{"action": "pickup", "object": "Apple"}]                       # what the LLM said
    executed = [{"action": "go_to", "object": "Apple"},                       # what actually ran (post-repair)
                {"action": "pickup", "object": "Apple", "to": None}]
    ok2 = [{"ok": True, "error": None}, {"ok": True, "error": None}]
    goal_ok = {"applicable": True, "ok": True, "target": "Apple", "held": "Apple", "error": None}
    r_goal = improve.close_loop(kb, mem, scene=scene, command=cmd_apple, level="L1", arm="plan_fresh",
                                executed_steps=executed, step_results=ok2, goal=goal_ok,
                                retrieved_ids=[{"id": 1, "kind": "facts"}],
                                rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    _check("success(goal): procedure INSERT",
           [(w["kind"], w["action"]) for w in r_goal["writes"]] == [("procedure", "insert")],
           str(r_goal["writes"]))
    proc = kb.data["procedures"][0] if kb.data["procedures"] else {}
    _check("success(goal): entry confidence 0.9 (goal source)",
           proc.get("confidence") == 0.9 and r_goal["outcome"]["source"] == "goal",
           f"conf={proc.get('confidence')} src={r_goal['outcome']['source']}")
    _check("D4: EXECUTED (post-repair) steps stored, not raw plan",
           proc.get("plan") == executed and proc.get("plan") != raw_plan)
    _check("D8/D12: procedure scope='global', source='environment'",
           proc.get("scope") == "global" and proc.get("source") == "environment",
           f"scope={proc.get('scope')} source={proc.get('source')}")

    # ------------------------------------------------------------------ success -> procedure (steps source)
    r_steps = improve.close_loop(kb, mem, scene=scene, command="toast the bread", level="L2", arm="reuse_memory",
                                 executed_steps=[{"action": "toggle_on", "object": "Toaster"}],
                                 step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                                 rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    bread = kb.data["procedures"][-1] if kb.data["procedures"] else {}
    _check("success(steps): procedure at conf 0.6 (no applicable goal)",
           r_steps["outcome"] == {"success": True, "source": "steps", "confidence": 0.6}
           and bread.get("confidence") == 0.6, f"outcome={r_steps['outcome']}")

    # ------------------------------------------------------------------ dedup: procedure reinforce (D6)
    r_dup = improve.close_loop(kb, mem, scene=scene, command="Pick  up the APPLE!", level="L1", arm="plan_fresh",
                               executed_steps=executed, step_results=ok2, goal=goal_ok,
                               rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    _check("dedup: repeat success -> procedure REINFORCE, same id",
           r_dup["writes"] and r_dup["writes"][0]["action"] == "reinforce"
           and r_dup["writes"][0]["id"] == r_goal["writes"][0]["id"], str(r_dup["writes"]))
    _check("dedup: no duplicate procedure row, occurrences=2",
           len(kb.data["procedures"]) == 2 and kb.data["procedures"][0].get("occurrences") == 2,
           f"rows={len(kb.data['procedures'])} occ={kb.data['procedures'][0].get('occurrences')}")

    # ------------------------------------------------------------------ failure -> template lesson (D5)
    cmd_mug = "pick up the mug"
    mug_err = "PickupObject failed: Mug|+01.2|-00.3 not visible"
    for _ in range(3):       # seed memory's Phase-5 failure counter -> lesson why gets 'seen x3'
        mem.record_failure("pickup", mug_err, object_type="Mug", scene=scene, command=cmd_mug)
    mug_steps = [{"action": "go_to", "object": "Mug"}, {"action": "pickup", "object": "Mug"}]
    bad = [{"ok": True, "error": None}, {"ok": False, "error": mug_err}]
    r_fail = improve.close_loop(kb, mem, scene=scene, command=cmd_mug, level="L1", arm="plan_fresh",
                                executed_steps=mug_steps, step_results=bad, goal={"applicable": False},
                                rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    les = kb.data["lessons"][0] if kb.data["lessons"] else {}
    _check("failure: template lesson INSERT (label + agent error)",
           r_fail["writes"] and r_fail["writes"][0] == {"kind": "lesson", "id": les.get("id"),
                                                        "action": "insert", "why": r_fail["writes"][0]["why"]}
           and "pickup Mug" in les.get("problem", "") and "not visible" in les.get("problem", ""),
           f"problem={les.get('problem')!r}")
    _check("failure: memory read-only 'seen xN' note in why",
           "seen x3" in r_fail["writes"][0]["why"], r_fail["writes"][0]["why"])
    _check("D8/D12: lesson scoped to the scene, source='environment'",
           les.get("scope") == scene and les.get("source") == "environment",
           f"scope={les.get('scope')} source={les.get('source')}")

    # ------------------------------------------------------------------ dedup: lesson reinforce by error class
    bad2 = [{"ok": True, "error": None},
            {"ok": False, "error": "PickupObject failed: Mug|+03.9|-01.7 not visible"}]
    r_fail2 = improve.close_loop(kb, mem, scene=scene, command=cmd_mug, level="L1", arm="plan_fresh",
                                 executed_steps=mug_steps, step_results=bad2, goal={"applicable": False},
                                 rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    _check("dedup: same error CLASS (different coords) -> lesson REINFORCE",
           r_fail2["writes"] and r_fail2["writes"][0]["action"] == "reinforce"
           and len(kb.data["lessons"]) == 1 and kb.data["lessons"][0].get("occurrences") == 2,
           str(r_fail2["writes"]))

    # ------------------------------------------------------------------ D7: later success keeps the old lesson
    r_after = improve.close_loop(kb, mem, scene=scene, command=cmd_mug, level="L1", arm="reuse_memory",
                                 executed_steps=mug_steps, step_results=ok2,
                                 goal={"applicable": True, "ok": True, "target": "Mug", "held": "Mug"},
                                 rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    _check("D7: success after failure records superseded lesson id",
           r_after["writes"] and r_after["writes"][0]["kind"] == "procedure"
           and les.get("id") in r_after["related_ids"], f"related={r_after['related_ids']}")
    _check("D7: old lesson persists (ranking demotes, never deletes)",
           any(e.get("id") == les.get("id") for e in kb.data["lessons"]))

    # ------------------------------------------------------------------ reflect: injected llm lesson (D5)
    r_reflect = improve.close_loop(kb, mem, scene=scene, command="chop the onion", level="L1", arm="plan_fresh",
                                   executed_steps=[{"action": "slice", "object": "Onion"}],
                                   step_results=[{"ok": False, "error": "no knife equipped"}],
                                   goal={"applicable": False}, rag_enabled=False, loop_enabled=True,
                                   reflect=True, llm=lambda p: "equip a knife before chopping",
                                   report_path=reports)
    n_journal += 1
    onion = next((e for e in kb.data["lessons"] if e.get("command") == "chop the onion"), {})
    _check("reflect: injected llm lesson text replaces the template",
           onion.get("lesson") == "equip a knife before chopping" and r_reflect["writes"],
           str(onion.get("lesson")))

    # ------------------------------------------------------------------ lesson dedup keys on RAW error, not label
    # Same command + same long step label, two DIFFERENT underlying errors -> two distinct lessons. Keying on
    # the label-prefixed problem string would collapse them (its 48-char _error_class window is consumed by the
    # label), storing one wrong remediation; improve passes error=<raw error> so they stay separate.
    lbl_steps = [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}]
    e_closed = improve.close_loop(kb, mem, scene=scene, command="put the apple in the fridge", level="L2",
                                  arm="plan_fresh", executed_steps=lbl_steps,
                                  step_results=[{"ok": False, "error": "Fridge is closed"}],
                                  goal={"applicable": False}, rag_enabled=False, loop_enabled=True,
                                  report_path=reports)
    n_journal += 1
    e_full = improve.close_loop(kb, mem, scene=scene, command="put the apple in the fridge", level="L2",
                                arm="plan_fresh", executed_steps=lbl_steps,
                                step_results=[{"ok": False, "error": "receptacle is full"}],
                                goal={"applicable": False}, rag_enabled=False, loop_enabled=True,
                                report_path=reports)
    n_journal += 1
    _check("lesson dedup keys on raw error (shared long label does NOT collapse distinct failures)",
           e_closed["writes"] and e_full["writes"]
           and e_closed["writes"][0]["action"] == "insert" and e_full["writes"][0]["action"] == "insert"
           and e_closed["writes"][0]["id"] != e_full["writes"][0]["id"],
           f"{e_closed['writes']} / {e_full['writes']}")

    # ------------------------------------------------------------------ never-raises on a KB write failure
    class _BoomKB:                              # a KB whose write blows up mid-close_loop
        data = {"procedures": [], "lessons": [], "facts": []}

        def upsert_procedure(self, *a, **k):
            raise RuntimeError("kb write failed")

        def upsert_lesson(self, *a, **k):
            raise RuntimeError("kb write failed")

    boom_report = improve.close_loop(_BoomKB(), None, scene=scene, command="stow the cup", level="L1",
                                     arm="reuse_memory", executed_steps=[{"action": "pickup", "object": "Cup"}],
                                     step_results=[{"ok": True, "error": None}],
                                     goal={"applicable": True, "ok": True, "target": "Cup", "held": "Cup"},
                                     rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    _check("never-raises: a KB write failure is captured in report['error'], not propagated",
           isinstance(boom_report, dict) and "error" in boom_report and boom_report["writes"] == [],
           str(boom_report.get("error")))

    # ------------------------------------------------------------------ loop OFF => zero writes (D2)
    stats_before = kb.stats()
    r_off = improve.close_loop(kb, mem, scene=scene, command="open the fridge", level="L1", arm="plan_fresh",
                               executed_steps=[{"action": "open", "object": "Fridge"}],
                               step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                               rag_enabled=False, loop_enabled=True, report_path=reports)
    n_journal += 1
    # the loop-ON write above proves stats change when allowed; now the same call with the loop OFF:
    stats_on = kb.stats()
    r_off = improve.close_loop(kb, mem, scene=scene, command="close the fridge", level="L1", arm="plan_fresh",
                               executed_steps=[{"action": "close", "object": "Fridge"}],
                               step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                               rag_enabled=True, loop_enabled=False, report_path=reports)
    n_journal += 1
    _check("loop off: zero KB writes, kb.stats() unchanged",
           r_off["writes"] == [] and kb.stats() == stats_on and stats_on != stats_before,
           f"writes={r_off['writes']} stats={kb.stats()}")
    _check("loop off: no reindex, loop_enabled=False recorded",
           r_off["reindexed"] is None and r_off["loop_enabled"] is False)
    with open(reports, "r", encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
    _check("loop off: journaled ablation trace has writes=[]",
           lines and lines[-1]["loop_enabled"] is False and lines[-1]["writes"] == [],
           f"last={lines[-1]['command']!r}")

    # ------------------------------------------------------------------ bandit: freeze + replay gate (D2/D10)
    # select.py imports numpy at module top; a numpy-less box makes ONLY these bandit checks unrunnable, so
    # they are recorded as explicit FAILs (with the reason) while the stdlib-only D9/D13/audit checks below
    # still run -- rather than aborting the whole gate with one monolithic "unexpected exception".
    db = os.path.join(tmp, "bandit.db")
    bandit = None
    try:
        sel = _load_select(here)
        _check("select: replay_procedure appended to ARMS",
               sel.ARMS == ["reuse_memory", "plan_fresh", "explore_then_plan", "replay_procedure"], str(sel.ARMS))
        bandit = sel.Bandit(db, alpha=1.0, enabled=True, updates_enabled=False)   # loop off => frozen (D2 wiring)
        x = bandit.featurize(command=cmd_apple, scene=scene, memory_hit=1, failure_seen=0)
        dec = bandit.decide(x, scene=scene, command=cmd_apple)
        a_before = bandit._A[dec["arm"]].tolist()
        u_before = bandit._updates[dec["arm"]]
        bandit.reward(dec["decision_id"], 1.0, source="environment")
        row = bandit.conn.execute("SELECT reward FROM bandit_decisions WHERE id=?",
                                  (dec["decision_id"],)).fetchone()
        _check("loop off: bandit reward() is a frozen no-op (A/b + row untouched)",
               bandit._A[dec["arm"]].tolist() == a_before and bandit._updates[dec["arm"]] == u_before
               and row["reward"] is None and bandit.stats()["updates_enabled"] is False)
        bandit.set_updates_enabled(True)                                          # loop back on => unfreeze
        bandit.reward(dec["decision_id"], 1.0, source="environment")
        row = bandit.conn.execute("SELECT reward, source FROM bandit_decisions WHERE id=?",
                                  (dec["decision_id"],)).fetchone()
        _check("loop on: same signal re-applied after unfreeze",
               bandit._updates[dec["arm"]] == u_before + 1 and row["reward"] == 1.0
               and row["source"] == "environment", str(dict(row)))
        for pm in (None, 0.0, sel.REPLAY_MATCH_THRESHOLD - 0.01, "garbage"):
            d = bandit.decide(x, scene=scene, command=cmd_apple, procedure_match=pm)
            if "replay_procedure" in d["eligible"] or d["arm"] == "replay_procedure":
                _check("replay gate: ineligible without a close procedure match", False, f"pm={pm!r} -> {d['arm']}")
                break
        else:
            _check("replay gate: ineligible without a close procedure match", True)
        d_hi = bandit.decide(x, scene=scene, command=cmd_apple,
                             procedure_match=sel.REPLAY_MATCH_THRESHOLD, procedure_entry="proc-001")
        _check("replay gate: eligible at threshold, entry id echoed back",
               "replay_procedure" in d_hi["eligible"] and d_hi["procedure_entry"] == "proc-001"
               and d_hi["procedure_match"] == sel.REPLAY_MATCH_THRESHOLD, str(d_hi["eligible"]))
    except Exception as exc:                                                      # numpy missing / select.py broken
        detail = f"select.py/bandit unavailable: {exc!r}"
        for nm in ("select: replay_procedure appended to ARMS",
                   "loop off: bandit reward() is a frozen no-op (A/b + row untouched)",
                   "loop on: same signal re-applied after unfreeze",
                   "replay gate: ineligible without a close procedure match",
                   "replay gate: eligible at threshold, entry id echoed back"):
            _check(nm, False, detail)

    # ------------------------------------------------------------------ D9: reindex is best-effort
    import rag as ragmod
    real_hybrid = ragmod.HybridRAG

    class _Unreachable(real_hybrid):                                          # real reindex, dead endpoint
        def __init__(self, kb=None, **kw):
            kw["endpoint"] = "http://127.0.0.1:1"
            super().__init__(kb=kb, **kw)

    ragmod.HybridRAG = _Unreachable
    try:
        r_re = improve.close_loop(kb, mem, scene=scene, command="wash the pan", level="L1", arm="reuse_memory",
                                  executed_steps=[{"action": "wash", "object": "Pan"}],
                                  step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                                  rag_enabled=True, loop_enabled=True, report_path=reports)
    finally:
        ragmod.HybridRAG = real_hybrid
    n_journal += 1
    _check("D9: unreachable Ollama -> reindex degrades (0 embedded), never raises",
           r_re["reindexed"] == 0 and "error" not in r_re, f"reindexed={r_re['reindexed']!r}")

    broken = types.ModuleType("rag")                                          # rag itself blowing up
    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("rag unavailable")
    broken.HybridRAG = _Boom
    sys.modules["rag"] = broken
    try:
        r_boom = improve.close_loop(kb, mem, scene=scene, command="dry the pan", level="L1", arm="reuse_memory",
                                    executed_steps=[{"action": "wipe", "object": "Pan"}],
                                    step_results=[{"ok": True, "error": None}], goal={"applicable": False},
                                    rag_enabled=True, loop_enabled=True, report_path=reports)
    finally:
        sys.modules["rag"] = ragmod
    n_journal += 1
    _check("D9: broken rag module -> reindexed=None, write still recorded",
           r_boom["reindexed"] is None and r_boom["writes"]
           and r_boom["writes"][0]["kind"] == "procedure", str(r_boom["writes"]))

    # ------------------------------------------------------------------ D13: report + jsonl shape
    with open(reports, "r", encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
    need = {"ts", "scene", "command", "level", "arm", "retrieved", "executed_steps", "outcome",
            "writes", "related_ids", "reindexed", "loop_enabled"}
    shape_ok = len(lines) == n_journal
    for ln in lines:
        shape_ok = shape_ok and need <= set(ln) and isinstance(ln["ts"], float)
        shape_ok = shape_ok and set(ln["outcome"]) == {"success", "source", "confidence"}
        shape_ok = shape_ok and ln["outcome"]["source"] in ("goal", "steps")
        shape_ok = shape_ok and ln["outcome"]["confidence"] in (0.9, 0.6)
        shape_ok = shape_ok and (ln["reindexed"] is None or isinstance(ln["reindexed"], int))
        for w in ln["writes"]:
            shape_ok = shape_ok and {"kind", "id", "action", "why"} <= set(w)
            shape_ok = shape_ok and w["kind"] in ("procedure", "lesson") and w["action"] in ("insert", "reinforce")
    _check("D13: one jsonl line per close_loop call, all keys/shapes valid",
           shape_ok, f"{len(lines)} lines (expected {n_journal})")
    _check("D13: retrieved ids passed through verbatim",
           lines[0]["retrieved"] == [{"id": 1, "kind": "facts"}], str(lines[0]["retrieved"]))

    # ------------------------------------------------------------------ audit trails
    with open(kb.audit_path, "r", encoding="utf-8") as fh:
        audit = fh.read()
    _check("audit: KB log has ADD + UPSERT lines with <environment> provenance",
           "ADD procedure" in audit and "ADD lesson" in audit and "UPSERT procedure" in audit
           and "UPSERT lesson" in audit and "<environment>" in audit)
    baudit_path = os.path.splitext(db)[0] + "_bandit_audit.log"
    if bandit is not None and os.path.exists(baudit_path):
        with open(baudit_path, "r", encoding="utf-8") as fh:
            baudit = fh.read()
        _check("audit: bandit log has REWARD-FROZEN + UPDATES-TOGGLE lines",
               "REWARD-FROZEN" in baudit and "UPDATES-TOGGLE" in baudit)
    else:
        _check("audit: bandit log has REWARD-FROZEN + UPDATES-TOGGLE lines", False,
               "bandit unavailable (numpy/select.py)")

    if bandit is not None:
        bandit.close()
    mem.close()


def main() -> int:
    _stub_ai2thor()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    tmp = tempfile.mkdtemp(prefix="phase7_selftest_")
    try:
        try:
            _run_checks(tmp, here)
        except Exception as exc:            # a crash mid-run still yields a table + non-zero exit
            _check("unexpected exception", False, repr(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(n) for n, _, _ in _results)
    print("=" * 78)
    print("PHASE-7 SELFTEST  (pure stdlib gate: no AI2-THOR, no network, temp stores)")
    print("-" * 78)
    for name, ok, detail in _results:
        tail = f"  | {detail[:70]}" if (detail and not ok) else ""
        print(f" [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}{tail}")
    passed = sum(1 for _, ok, _ in _results if ok)
    print("-" * 78)
    print(f" {passed}/{len(_results)} checks passed")
    print("=" * 78)
    if passed == len(_results):
        print("PHASE7 SELFTEST OK")
        return 0
    print("PHASE7 SELFTEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
