"""M1 - offline selftest for sim/final_test/matrix/practice_k.py (synthetic fake executor; no simulator, no WSL, no Ollama, no API).

    python sim/_ftm_practice_k_selftest.py        (Windows python, repo root)

  1  seeded orders: deterministic, without replacement, round-robin by level, seeded languages, orders differ, pool items only
  2  normalisation + leakage gate: a planted test id / EN / FA instruction is found; the clean snapshot passes; names only ids
  3  a full small run (orders 0 and 1, K = 0 / 1 / 3 / 5 / 10): journal, snapshots, manifests, nesting, state hashes, cost, event
  4  online memory: the block of a later attempt is built from the LIVE practice memory, after the sweep (summary in the prompt)
  5  every started attempt counts (inapplicable, rejected, crash); a transport failure aborts without a row and resumes
  6  resume: a kill between consolidation and the journal line restores the checkpoint; the result equals an uninterrupted run
  7  test items are refused (injected into an order, wrong namespace); the leakage gate aborts a leaking snapshot
  8  static: python 3.10 syntax, lazy simulator imports, no key, no hard-coded item id, allow-list on the live build
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                           # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATRIX = os.path.join(HERE, "final_test", "matrix")
sys.path.insert(0, MATRIX)

import common as C                                          # noqa: E402
import practice_k as PK                                     # noqa: E402

# W2: sim/step5/params.json now carries the A12 block; this selftest uses fake single-tokenizer counters, so it reads the registry WITHOUT it.
import ltm as _L                                            # noqa: E402
_load_params_real = _L.load_params


def _load_params_c2e(path=None):
    import copy as _copy
    P = _copy.deepcopy(_load_params_real(path))
    P["retrieval"].pop("retention", None)
    P["retrieval"].pop("a12", None)
    return P


_L.load_params = _load_params_c2e

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


MAN = C.load_manifest()
GEN = {i["id"]: i for i in json.load(open(os.path.join(ROOT, "sim", "dataset", "general.json"), encoding="utf-8"))["items"]}
LEVEL = PK.level_of_items(GEN)
POOL = sorted(MAN["practice_pool"])
TEST_IDS = sorted(set(MAN["test"]) | set(MAN["test_uncapped_not_scored"]))


# ----------------------------------------------------------------------------------------------- the fakes
class FakeRunner:
    """PlannerRunner stand-in: records what it was given; deterministic verdict by the instruction text."""

    def __init__(self):
        self.calls = []
        self.fail_next = False

    def plan(self, command, memory_summary=(), holding=None, memory_block=None, fewshot_block=None, replicate=0):
        self.calls.append({"command": command, "summary": list(memory_summary), "memory_block": memory_block})
        if self.fail_next:
            self.fail_next = False
            return {"ok": False, "first_ok": False, "repaired": False, "rejected": False, "infra_failure": True, "calls": 1, "tokens_in": 0,
                    "tokens_out": 0, "plan": [], "first_plan": [], "say": "", "normalised": False, "attempts": []}
        h = sum(command.encode("utf-8")) % 5
        if h == 0:
            return {"ok": False, "first_ok": False, "repaired": False, "rejected": True, "infra_failure": False, "calls": 2, "tokens_in": 100,
                    "tokens_out": 20, "plan": [], "first_plan": [], "say": "", "normalised": False, "attempts": []}
        plan = [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]
        return {"ok": True, "first_ok": True, "repaired": False, "rejected": False, "infra_failure": False, "calls": 1, "tokens_in": 100,
                "tokens_out": 20, "plan": plan, "first_plan": plan, "say": "ok", "normalised": False, "attempts": []}


class FakeCounter:
    def counts(self, text):
        return {"gemma": len(text) // 4 + 1}

    def __call__(self, text):
        return max(self.counts(text).values()) if text else 0

    def save(self):
        pass


class SimKill(BaseException):
    pass


def _h(item_id):
    return sum(item_id.encode("utf-8"))


class FakeExec:
    """Fake of LiveExecutor.execute for practice: builds a REAL HonestEventAdapter over the practice LTM, emits a sighting, calls the
    plan hook with the A11 post dict, then ends the episode like the real executor (terminal, admission, end)."""

    frame, map_rev = "f0", "m0"

    def __init__(self, runner, contract, channels=("places",), crash_on=(), inapplicable_mod=7):
        import ltm_adapter  # noqa: F401
        self.hook = PK.PracticeHook(runner, contract, FakeCounter(), channels)
        self.plan_hook = self.hook
        self.last = None
        self.crash_on = set(crash_on)
        self.n_exec = 0
        self.inapplicable_mod = inapplicable_mod
        self.closed = False

    def execute(self, req):
        import ltm_adapter as A
        import step5_eval as se
        self.n_exec += 1
        it = req.item
        res = se.EpisodeResult()
        if it["id"] in self.crash_on:
            raise RuntimeError("simulated controller crash")
        if _h(it["id"]) % self.inapplicable_mod == 0:
            res.inapplicable = "goal already true before acting"
            return res
        ad = A.HonestEventAdapter(req.memory, episode=req.episode_id, mode="practice", world=req.world, frame=self.frame, map_rev=self.map_rev,
                                  kitchen=req.kitchen)
        ad.on_task(it[req.lang], it["level"], req.lang)
        import ltm_concepts as lc
        roles = lc.parse(it[req.lang])["roles"]
        otype = next((r["obj"] for r in roles if r.get("obj") not in (None, "-", "?")), ("Apple", "Mug", "Bread", "Tomato")[_h(it["id"]) % 4])
        for j, ot in enumerate(dict.fromkeys([otype, "Apple", "Bread", "Tomato", "Mug", "Lettuce", "Egg"])):
            ad._emit("sighting", otype=ot, state="?", conf=0.9, detector="det", track=f"tr:{j + 1}", xyz=(1.0, 0.9, 1.0 + 0.3 * j), xyz_ok=True,
                     view="vw:1", obs="ob:1", in_rtype="Fridge", in_basis="box_only", in_xyz=(1.0, 0.9, 1.1))
        summary = [f"saw {otype} in Fridge"]
        post = {"wm": ad.wm, "summary": summary, "summary_sha": C.sha256_text("\n".join(summary)), "holding": None, "events": ad.stream,
                "sweep_prims": 3}
        ph = self.plan_hook(req, it, None, req.memory, post=post)
        info = ph["info"]
        self.last = {"adapter": ad}
        if ph["steps"] is None:
            ad.on_terminal(False, 3, 0)
            ad.on_admission(False)
            ad.end()
            res.goal_ok, res.wm, res.admission = False, ad.wm, False
            res.extra = {"plan": info, "plan_not_executed": True, "plan_not_executed_reason": "rejected"}
            return res
        for i, s in enumerate(ph["steps"]):
            ad.on_step_start(i, {"action": s["action"], "object": s["object"]})
            ad.on_step_end(i, {"ok": True, "retries": 0, "detail": {"prims": 5}})
        ok = _h(it["id"]) % 3 != 0
        ad.on_terminal(ok, 8, 0)
        ad.on_admission(ok)
        ad.end()
        res.goal_ok, res.wm, res.admission = ok, ad.wm, ok
        res.extra = {"plan": info, "plan_not_executed": False, "steps": [{"ok": True, "action": "pickup"}],
                     "core": {"prims": 8, "audit": {"violations": 0, "counts": {"oracle": 0}}}}
        return res

    def close(self):
        self.closed = True


def mk(tmp, orders=(0, 1), ks=PK.KS, **kw):
    from planners.contract import PlannerContract
    runner = kw.pop("runner", None) or FakeRunner()
    ex = kw.pop("ex", None) or FakeExec(runner, PlannerContract(["Apple", "Mug", "Bread", "Tomato", "Fridge"]), **kw.pop("exkw", {}))
    pr = PK.Practice(namespace="practice-v1", orders=list(orders), ks=list(ks), executor=ex, hook=ex.hook, counter=FakeCounter(),
                     snap_root=os.path.join(tmp, "snap"), journal=os.path.join(tmp, "docs", "practice.jsonl"),
                     events=os.path.join(tmp, "docs", "events.jsonl"), work=os.path.join(tmp, "work"), log=lambda s: None, **kw)
    return pr, ex, runner


def rows_of(tmp):
    return C.Journal(os.path.join(tmp, "docs", "practice.jsonl")).rows()


# ================================================================================================ 1 orders
print("== 1 seeded orders")
o0 = PK.practice_order(POOL, LEVEL, 0)
o0b = PK.practice_order(POOL, LEVEL, 0)
o1 = PK.practice_order(POOL, LEVEL, 1)
o2 = PK.practice_order(POOL, LEVEL, 2)
check("deterministic", o0 == o0b)
check("orders differ", o0[:10] != o1[:10] and o1[:10] != o2[:10] and o0[:10] != o2[:10])
check("without replacement, whole pool", len({p["item"] for p in o0}) == len(o0) == len(POOL))
check("every item from the practice pool, none a test item", all(p["item"] in POOL for p in o0) and not (set(p["item"] for p in o0) & set(TEST_IDS)))
check("round-robin by level (L1, L2, L3, L1, ...)", [p["level"] for p in o0[:9]] == ["L1", "L2", "L3"] * 3, str([p["level"] for p in o0[:9]]))
check("attempts numbered from 1", [p["n"] for p in o0[:5]] == [1, 2, 3, 4, 5])
check("languages seeded, both appear", {p["lang"] for p in o0[:10]} == {"en", "fa"} and {p["lang"] for p in o0} <= {"en", "fa"})
check("level matches the dataset", all(GEN[p["item"]]["level"] == p["level"] for p in o0))
check("the three orders are pinned (C.rng labels)", [p["item"] for p in o0[:3]] == ["g-l1-off-microwave", "g-l2-microwave-tomato", "g-l3-bad-tomato"],
      str([p["item"] for p in o0[:3]]))

# ================================================================================================ 2 leakage
print("== 2 normalisation + leakage gate")
check("norm: case, punctuation, whitespace", PK.norm_text("  Put the  Apple, in the FRIDGE! ") == "put the apple in the fridge")
check("norm: Arabic yeh/kaf, ZWNJ, digits", PK.norm_text("كتاب‌ي ۱۲٣") == PK.norm_text("کتابی 123"))
needles = PK.test_needles(MAN, GEN)
check("needles: ids + EN/FA texts of every test item", len(needles["ids"]) == len(TEST_IDS) and len(needles["texts"]) >= 2 * len(TEST_IDS) - 4)
tmp = tempfile.mkdtemp(prefix="ftm-pk-")
try:
    import sqlite3
    d = os.path.join(tmp, "leak")
    os.makedirs(d)
    tid = TEST_IDS[0]
    con = sqlite3.connect(os.path.join(d, "ltm.db"))
    con.execute("create table t(x text)")
    con.execute("insert into t values (?)", ("lesson about " + GEN[tid]["en"].upper() + " done",))
    con.commit()
    con.close()
    g = PK.leakage_gate(d, needles)
    check("planted EN instruction (other case) is found", not g["ok"] and any(h["kind"] == "instruction" and h["item"] == tid for h in g["hits"]), str(g["hits"][:2]))
    check("hits name ids, never texts", all(set(h) <= {"kind", "item", "lang", "error"} for h in g["hits"]))
    d2 = os.path.join(tmp, "leak2")
    os.makedirs(d2)
    con = sqlite3.connect(os.path.join(d2, "ltm.db"))
    con.execute("create table t(x text)")
    con.execute("insert into t values (?)", ("episode " + tid,))
    con.commit()
    con.close()
    g2 = PK.leakage_gate(d2, needles)
    check("planted test item id is found", not g2["ok"] and any(h["kind"] == "item_id" and h["item"] == tid for h in g2["hits"]), str(g2["hits"][:2]))
    fa_id = next(i for i in TEST_IDS if GEN[i].get("fa") and len(PK.norm_text(GEN[i]["fa"])) >= PK.MIN_NEEDLE)
    d3 = os.path.join(tmp, "leak3")
    os.makedirs(d3)
    con = sqlite3.connect(os.path.join(d3, "ltm.db"))
    con.execute("create table t(x text)")
    con.execute("insert into t values (?)", (GEN[fa_id]["fa"].replace("ی", "ي") + "؟",))
    con.commit()
    con.close()
    g3 = PK.leakage_gate(d3, needles)
    check("planted FA instruction (Arabic yeh, punctuation) is found", any(h["kind"] == "instruction" and h["lang"] == "fa" for h in g3["hits"]), str(g3["hits"][:2]))

    # ============================================================================================ 3 a full small run
    print("== 3 a full run (orders 0 and 1)")
    t3 = os.path.join(tmp, "run3")
    pr, ex, runner = mk(t3)
    out = pr.run()
    rows = rows_of(t3)
    check("both orders complete", all(r["complete"] for r in out) and len(rows) == 20, str(out))
    check("journal: one row per attempt, numbered, seeded order",
          [(r["order"], r["attempt"], r["item"], r["lang"]) for r in rows if r["order"] == 0] == [(0, p["n"], p["item"], p["lang"]) for p in o0[:10]])
    check("rows carry the practice fields", all(k in rows[0] for k in ("key", "namespace", "order", "attempt", "item", "lang", "level", "status", "snapshot_after",
                                                                       "instruction", "state_hash_after", "prims", "wall_s", "calls")))
    check("snapshot_after lists exactly the K reached", {r["attempt"] for r in rows if r["order"] == 0 and r["snapshot_after"]} == {1, 3, 5, 10})
    allok = True
    for o in (0, 1):
        for k in PK.KS:
            bad = PK.verify_snapshot_dir(pr.snap_dir(o, k), o, k)
            allok = allok and not bad
    check("all 10 snapshots exist, verify, leakage ok", allok)
    m10 = PK.read_snapshot_manifest(pr.snap_dir(0, 10))
    check("manifest fields", all(k in m10 for k in ("order", "k", "state_hash", "params_sha", "channels", "attempts", "leakage", "sha256"))
          and m10["channels"] == ["places"] and m10["attempts"] == [p["item"] for p in o0[:10]] and m10["leakage"]["ok"])
    check("S* path = o0/k10", C.snapshot_dir("S*", None).endswith(os.path.join("snapshots", "o0", "k10")))
    check("nesting: each K is a prefix of the next", not PK.verify_nesting(os.path.join(t3, "snap"), 0) and not PK.verify_nesting(os.path.join(t3, "snap"), 1))
    check("K = 0 is an empty memory (no place claims)", PK.read_snapshot_manifest(pr.snap_dir(0, 0))["attempts"] == [])
    import ltm as L
    ks_state = {}
    for k in PK.KS:
        fz = L.LongTermMemory.open_frozen(pr.snap_dir(0, k), os.path.join(tmp, f"tl{k}.db"), kitchen="FloorPlan1")
        ks_state[k] = len(fz.store.select("location_claim"))
        fz.close()
    check("memory grows with K (claims at K=0 is 0, K=10 > 0, non-decreasing)", ks_state[0] == 0 and ks_state[10] > 0 and
          all(ks_state[a] <= ks_state[b] for a, b in zip(PK.KS, PK.KS[1:])), str(ks_state))
    r10 = next(r for r in rows if r["order"] == 0 and r["attempt"] == 10)
    check("snapshot K=10 state == the journaled state after attempt 10", m10["state_hash"] == r10["state_hash_after"])
    cost = PK.cost_summary(rows)
    check("cost: episodes, primitives, wall time, Gemma calls", cost["total"]["episodes"] == 20 and cost["total"]["gemma_calls"] > 0 and
          cost["by_order"]["0"]["episodes"] == 10)
    ev = C.Journal(os.path.join(t3, "docs", "events.jsonl")).rows()
    check("phase_complete event with the journal sha", len(ev) == 1 and ev[0]["event"] == "phase_complete" and ev[0]["journal_sha"] == C.sha256_file(os.path.join(t3, "docs", "practice.jsonl")))
    check("practice_cost.json written", os.path.exists(os.path.join(t3, "docs", "practice_cost.json")))
    st = PK.status("practice-v1", [0, 1], list(PK.KS), snap_root=os.path.join(t3, "snap"), journal=os.path.join(t3, "docs", "practice.jsonl"))
    check("status() ok", st["ok"], str(st))
    pr2, ex2, _ = mk(t3)
    pr2.run()
    check("re-running a finished study repeats nothing", ex2.n_exec == 0 and len(rows_of(t3)) == 20 and len(C.Journal(os.path.join(t3, "docs", "events.jsonl")).rows()) == 1)

    # ============================================================================================ 4 online memory, after the sweep
    print("== 4 online memory read, after the sweep")
    withblock = [r for r in rows if r["order"] == 0 and (r.get("block") or {}).get("n_records")]
    check("a later attempt's prompt block comes from the live practice memory", len(withblock) > 0 and all(r["attempt"] > 1 for r in withblock))
    first = next(r for r in rows if r["order"] == 0 and r["attempt"] == 1)
    check("attempt 1 (empty memory): no block, byte-identical to the no-memory prompt", first["block_sha"] is None)
    sum_calls = [c for c in runner.calls]
    check("the prompt carries the post-sweep summary (A11)", sum_calls and all(c["summary"] and c["summary"][0].startswith("saw ") for c in sum_calls))
    check("plan_order is after_sweep; summary sha recorded", all(r["plan_order"] == "after_sweep" and r["summary_sha"] for r in rows if r["status"] in ("executed", "rejected")))
    check("a memory block text starts from the planner contract (rendered by PlannerContract)", any(c["memory_block"] for c in sum_calls))
    prl = PK.PracticeHook(runner, ex.hook.contract, FakeCounter(), ("places",))
    try:
        prl(None, None, None, None, post=None)
        check("the hook refuses the pre-A11 call (post=None)", False)
    except PK.PracticeError:
        check("the hook refuses the pre-A11 call (post=None)", True)

    # ============================================================================================ 5 started attempts
    print("== 5 every started attempt counts")
    st_counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("executed", "rejected", "inapplicable", "crash")}
    check("executed, rejected and inapplicable attempts all occur and all count", st_counts["inapplicable"] > 0 and st_counts["rejected"] > 0 and
          st_counts["executed"] > 0 and sum(st_counts.values()) == 20, str(st_counts))
    check("an inapplicable attempt leaves the memory unchanged except the world reset (no consolidation)",
          all(not r["consolidated"] for r in rows if r["status"] == "inapplicable"))
    check("a rejected plan is consolidated as a failed attempt (admission False)",
          all(r["consolidated"] and r["admission"] is False for r in rows if r["status"] == "rejected"))
    t5 = os.path.join(tmp, "run5")
    crash_item = o0[1]["item"]
    pr5, ex5, runner5 = mk(t5, orders=(0,), ks=(0, 1, 3), exkw={"crash_on": [crash_item], "inapplicable_mod": 10 ** 9})
    pr5.run()
    r5 = rows_of(t5)
    cr = [r for r in r5 if r["status"] == "crash"]
    print("   crash rows:", [(r["attempt"], r["status"], r["item"], r.get("crash")) for r in r5], crash_item)
    check("a crash is a started attempt (row written, counted in K, not consolidated)", len(cr) == 1 and cr[0]["attempt"] == 2 and not cr[0]["consolidated"] and len(r5) == 3)
    check("the snapshot at K = 3 exists after a crash inside the first three", not PK.verify_snapshot_dir(pr5.snap_dir(0, 3), 0, 3))
    t5b = os.path.join(tmp, "run5b")
    runner5b = FakeRunner()
    pr5b, ex5b, _ = mk(t5b, orders=(0,), ks=(1, 3), runner=runner5b, exkw={"inapplicable_mod": 10 ** 9})
    runner5b.fail_next = True
    try:
        pr5b.run()
        check("a planner transport failure aborts the run", False)
    except Exception as e:
        check("a planner transport failure aborts the run (LiveInvariant)", type(e).__name__ == "LiveInvariant", type(e).__name__)
    check("... and writes no practice row", rows_of(t5b) == [])
    pr5c, ex5c, _ = mk(t5b, orders=(0,), ks=(1, 3), runner=runner5b, exkw={"inapplicable_mod": 10 ** 9})
    pr5c.run()
    check("... the resume repeats that attempt from attempt 1", [r["attempt"] for r in rows_of(t5b)] == [1, 2, 3])

    class BuggyRunner(FakeRunner):
        def plan(self, *a, **k):
            raise ValueError("bug in the planner glue")
    t5d = os.path.join(tmp, "run5d")
    pr5d, _, _ = mk(t5d, orders=(0,), ks=(1,), runner=BuggyRunner())
    try:
        pr5d.run()
        check("a bug inside the plan hook is fatal, not a 'crash' attempt", False)
    except Exception as e:
        check("a bug inside the plan hook is fatal, not a 'crash' attempt", type(e).__name__ == "LiveInvariant" and rows_of(t5d) == [], type(e).__name__)

    # ============================================================================================ 6 resume
    print("== 6 resume after a kill")
    t6 = os.path.join(tmp, "run6")

    def kill_at(where, o, n):
        if where == "after_consolidate" and o == 0 and n == 4:
            raise SimKill()
    pr6, ex6, _ = mk(t6, orders=(0,), ks=PK.KS, fault=kill_at)
    try:
        pr6.run()
        check("the injected kill fires", False)
    except SimKill:
        check("the injected kill fires after consolidation of attempt 4", True)
    check("journal holds attempts 1..3 only", [r["attempt"] for r in rows_of(t6)] == [1, 2, 3])
    pr6._close_mem()
    pr6b, ex6b, _ = mk(t6, orders=(0,), ks=PK.KS)
    pr6b.run()
    r6 = rows_of(t6)
    check("resume completes 10 attempts, 1..10 once each", [r["attempt"] for r in r6] == list(range(1, 11)))
    t6u = os.path.join(tmp, "run6u")
    pr6u, ex6u, _ = mk(t6u, orders=(0,), ks=PK.KS)
    pr6u.run()
    r6u = rows_of(t6u)
    check("resumed run == uninterrupted run (statuses, plans, state hashes)",
          [(r["status"], r["plan"], r["state_hash_after"]) for r in r6] == [(r["status"], r["plan"], r["state_hash_after"]) for r in r6u],
          str([(a["attempt"], a["state_hash_after"][:6], b["state_hash_after"][:6]) for a, b in zip(r6, r6u) if a["state_hash_after"] != b["state_hash_after"]][:3]))
    check("snapshot K=10 identical state in both", PK.read_snapshot_manifest(pr6b.snap_dir(0, 10))["state_hash"] == PK.read_snapshot_manifest(pr6u.snap_dir(0, 10))["state_hash"])
    check("resume re-ran only the missing attempts", ex6b.n_exec >= 1 and ex6b.n_exec <= 8)
    # a tampered memory that matches neither the journal nor the checkpoint -> fail fast
    t6t = os.path.join(tmp, "run6t")
    pr6t, ex6t, _ = mk(t6t, orders=(0,), ks=(0, 1, 3), limit=2)
    pr6t.run()
    pr6t._close_mem()
    shutil.rmtree(os.path.join(t6t, "work", "o0", "ckpt"))
    dbp = os.path.join(t6t, "work", "o0", "ltm.practice.db")
    import ltm as L2
    mm = L2.LongTermMemory(dbp, kitchen="FloorPlan1")
    mm.reset_world("tamper", "task_reset")
    mm.close()
    pr6t2, _, _ = mk(t6t, orders=(0,), ks=(0, 1, 3))
    try:
        pr6t2.run()
        check("a memory that matches neither journal nor checkpoint stops the run", False)
    except PK.PracticeError:
        check("a memory that matches neither journal nor checkpoint stops the run", True)
    # the journal must follow the seeded order
    t6o = os.path.join(tmp, "run6o")
    pr6o, _, _ = mk(t6o, orders=(0,), ks=(1,), limit=1)
    pr6o.run()
    jp = os.path.join(t6o, "docs", "practice.jsonl")
    rr = rows_of(t6o)
    rr[0]["item"] = "g-other"
    open(jp, "w", encoding="utf-8").write(json.dumps(rr[0]) + "\n")
    pr6o2, _, _ = mk(t6o, orders=(0,), ks=(1,))
    try:
        pr6o2.run()
        check("a journal that is not the seeded order stops the run", False)
    except PK.PracticeError:
        check("a journal that is not the seeded order stops the run", True)

    # ============================================================================================ 7 test items refused
    print("== 7 test items refused")
    from planners.contract import PlannerContract
    try:
        PK.Practice(namespace="test-v1", orders=[0], ks=[1], executor=None, snap_root=tmp, journal=os.path.join(tmp, "x.jsonl"), work=tmp, log=lambda s: None)
        check("namespace test-v1 refused for practice", False)
    except PK.PracticeError:
        check("namespace test-v1 refused for practice", True)
    real = PK.practice_order
    try:
        PK.practice_order = lambda pool, lv, o: [{"n": 1, "item": TEST_IDS[0], "lang": "en", "level": GEN[TEST_IDS[0]]["level"]}]
        try:
            PK.Practice(namespace="practice-v1", orders=[0], ks=[1], executor=None, snap_root=tmp, journal=os.path.join(tmp, "y.jsonl"), work=tmp, log=lambda s: None)
            check("a test item injected into an order is refused by the guard", False)
        except C.TestItemRefused:
            check("a test item injected into an order is refused by the guard", True)
        try:
            PK.Practice(namespace="practice-v1", orders=[0], ks=[1], executor=None, snap_root=tmp, journal=os.path.join(tmp, "z.jsonl"), work=tmp,
                        log=lambda s: None, allow_test=True)
            check("--allow-test does not open it (namespace practice-v1, no freeze)", False)
        except C.TestItemRefused:
            check("--allow-test does not open it (namespace practice-v1, no freeze)", True)
    finally:
        PK.practice_order = real
    # a leaking snapshot aborts
    t7 = os.path.join(tmp, "run7")
    pr7, ex7, _ = mk(t7, orders=(0,), ks=(0, 1, 3))
    leak_needles = {"ids": list(needles["ids"]), "texts": list(needles["texts"])}
    bad_item = o0[0]["item"]
    leak_needles["ids"].append(bad_item.upper())
    pr7.needles = leak_needles
    pr7.take_snapshot  # noqa: B018
    orig_gate = PK.leakage_gate
    try:
        PK.leakage_gate = lambda d, n: {"ok": False, "hits": [{"kind": "item_id", "item": "x"}], "needles": {}, "scanned": []}
        try:
            pr7.run()
            check("a leaking snapshot aborts the run", False)
        except PK.LeakageError:
            check("a leaking snapshot aborts the run (LeakageError)", True)
    finally:
        PK.leakage_gate = orig_gate
    check("... and no practice row was written for the leaking K", rows_of(t7) == [])
    # no key in any record
    all_text = "".join(open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read() for dp, _, fs in os.walk(os.path.join(t3, "docs")) for f in fs)
    check("no key-like string in the journals", not C.key_leak(all_text))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ================================================================================================ 8 static
print("== 8 static")
src = open(os.path.join(MATRIX, "practice_k.py"), encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    check("python 3.10 syntax", True)
except SyntaxError as e:
    check("python 3.10 syntax", False, str(e))
tree = ast.parse(src)
top_imports = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)} | \
              {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
check("simulator / Ollama / step-4 modules imported lazily only", not (top_imports & {"honest", "honest_skills", "thor_actions", "webapp", "ai2thor", "requests", "perception", "step5_live", "ltm", "planners"}), str(top_imports))
check("no hard-coded item id", not any(GEN_ID in src for GEN_ID in list(GEN)[:300] if len(GEN_ID) > 12))
check("no key material or key reader other than through planners", "THESIS_ANTHROPIC_API_KEY" not in src and "get_anthropic_key" not in src)
check("the live build sets an allow-list and a forbid-list from the namespace's split set", "allow=allowed" in src and "forbid=forbidden" in src and "assert_items_allowed" in src)
check("the live build runs under A11 (plan_after_sweep=True)", "plan_after_sweep=True" in src)
check("practice cache namespace is the practice namespace", "ResponseCache(os.path.join(work, \"cache\"), namespace=ns)" in src)
check("guard runs before anything and on every resume (constructor)", "C.guard_items" in src.split("def snap_dir")[0])
check("only one writer: PhaseLock around the live run", "C.PhaseLock(a.namespace, PHASE" in src)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
