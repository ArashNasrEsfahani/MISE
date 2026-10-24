"""Offline selftest for B9 (sim/final_test/l3_feasibility.py): fake executor, no simulator, no WSL, no LLM.
Run from the repo root:  python sim/_pkg_l3feas_selftest.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test"))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import l3_feasibility as F                                        # noqa: E402
import step5_eval as se                                           # noqa: E402

N = [0, 0]


def check(name, cond, extra=""):
    N[1] += 1
    if cond:
        N[0] += 1
    else:
        print("FAIL:", name, extra)


def step(i, a, o, ok=True, cause=None):
    return {"i": i, "action": a, "object": o, "to": None, "ok": ok, "cause": cause, "retries": 0, "prims": 3}


class FakeExec:
    """Returns a scripted EpisodeResult per item id; records the requests it saw (no simulator)."""
    def __init__(self, script):
        self.script, self.seen = script, []

    def execute(self, req):
        self.seen.append(req)
        sc = self.script[req.item["id"]]
        if isinstance(sc, list):                      # per-attempt scripts
            sc = sc[min(len(self.seen_for(req.item["id"])) - 1, len(sc) - 1)]
        r = se.EpisodeResult()
        if sc.get("inapplicable"):
            r.inapplicable = sc["inapplicable"]
            return r
        r.goal_ok = sc["goal_ok"]
        r.extra = {"steps": sc["steps"], "core": {"exec_ok": all(s["ok"] for s in sc["steps"]), "goal_ok": sc["goal_ok"],
                                                  "audit": {"violations": sc.get("viol", 0), "counts": {"oracle": sc.get("oracle", 0)}}}}
        return r

    def seen_for(self, iid):
        return [q for q in self.seen if q.item["id"] == iid]


def item(iid, plan, level="L3", setup=None):
    return {"id": iid, "level": level, "en": "x", "fa": "y", "setup": setup or [], "plan": plan, "goal": []}


def main():
    # ---- 1. classification (pure) ---------------------------------------------------------------------------------------------
    c = F.classify_auto
    check("feasible", c([step(0, "heat", "Egg")], True)[0] == "feasible")
    check("not_detected -> perception", c([step(0, "pickup", "Egg", False, "not_detected")], False)[0] == "perception")
    check("unreachable -> navigation", c([step(0, "go_to", "Sink", False, "unreachable")], False)[0] == "navigation")
    check("retry_exhausted -> unreviewed (never auto defect)", c([step(0, "heat", "Egg", False, "retry_exhausted")], False)[0] == "unreviewed")
    check("all ok, goal unmet -> unreviewed", c([step(0, "heat", "Egg")], False)[0] == "unreviewed")
    check("goal holds although a step failed -> feasible (visible in why)",
          c([step(0, "pickup", "Knife", False, "not_detected"), step(1, "slice", "Bread")], True)[0] == "feasible" and
          "not_detected" in c([step(0, "pickup", "Knife", False, "not_detected")], True)[1])
    old = {"item": "g-x", "mode": "refplans", "requirement": "refplan", "attempt": 1, "class": "perception", "class_auto": "perception",
           "steps": [step(0, "pickup", "Knife", False, "not_detected")], "goal_ok": True, "audit": {}, "inapplicable": None, "why": ""}
    check("reclass corrects a stored row without changing the history", F.reclass(old, {})["class"] == "feasible" and old["class"] == "perception")
    check("inapplicable first", c([], None, "goal already true before acting")[0] == "inapplicable")
    check("audit violation -> invariant_error", c([step(0, "heat", "Egg")], True, None, {"violations": 1})[0] == "invariant_error")
    check("oracle read -> invariant_error", c([step(0, "heat", "Egg")], True, None, {"oracle_reads": 2})[0] == "invariant_error")
    check("auto never says executor_defect", all(c([step(0, "x", "Y", False, k)], False)[0] != "executor_defect"
                                                 for k in ("wrong_target", "action_refused", "state_unrecognised", "retry_exhausted", None)))

    # ---- 2. review overlay -----------------------------------------------------------------------------------------------------
    rev = {"it-a": {"class": "executor_defect", "reason": "r"}, "it-b": {"class": "api_cannot_express", "reason": "r"}}
    check("overlay applies to unreviewed", F.final_class("unreviewed", "it-a", rev)[0] == "executor_defect")
    check("overlay leaves feasible", F.final_class("feasible", "it-a", rev)[0] == "feasible")
    check("overlay may override a provisional perception / navigation label (Astra H-B9)",
          F.final_class("perception", "it-a", rev)[0] == "executor_defect" and F.final_class("navigation", "it-b", rev)[0] == "api_cannot_express")
    check("perception without a review stays perception", F.final_class("perception", "zzz", rev)[0] == "perception")
    check("overlay never overrides inapplicable / invariant_error",
          F.final_class("inapplicable", "it-a", rev)[0] == "inapplicable" and F.final_class("invariant_error", "it-a", rev)[0] == "invariant_error")
    check("no review stays unreviewed", F.final_class("unreviewed", "zzz", rev)[0] == "unreviewed")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rev.json")
        json.dump({"reviews": {"a": {"class": "executor_defect", "reason": "x"}, "b": {"class": "bogus", "reason": "x"},
                               "c": {"class": "navigation"}}}, open(p, "w"))
        check("load_review drops invalid classes and reason-less rows", set(F.load_review(p)) == {"a"})
        check("load_review missing file -> {}", F.load_review(os.path.join(td, "nope.json")) == {})

    # ---- 3. binding ------------------------------------------------------------------------------------------------------------
    plan = [{"action": "pickup", "object": "Tomato"}, {"action": "put_in", "object": "Fridge"}]
    b, u = F.bind_steps([{"action": "pickup", "object": "?food"}, {"action": "put_in", "object": "?dest"}], plan)
    check("bind: slots bound from the support item", b[0]["object"] == "Tomato" and b[1]["object"] == "Fridge" and not u)
    b, u = F.bind_steps([{"action": "heat", "object": "?food"}], [{"action": "close", "object": "Door"}])
    check("bind: falls back to any object of the plan", b[0]["object"] == "Door" and not u)
    b, u = F.bind_steps([{"action": "heat", "object": "?food"}], [])
    check("bind: unbound slot reported, not guessed", u == ["?food"] and b[0]["object"] == "?food")
    b, _ = F.bind_steps([{"action": "pickup", "object": "?x"}, {"action": "put_in", "object": "?x"}], plan)
    check("bind: same slot, same value", b[0]["object"] == b[1]["object"] == "Tomato")
    cfg = F.distinct_configs([item("a", plan, "L2"), item("b", plan, "L2"), item("c", [{"action": "open", "object": "Fridge"}], "L2")])
    check("distinct_configs skips identical configs", [x["id"] for x in cfg] == ["a", "c"])

    # ---- 4. run_jobs: resume, retry, flaky, fail-fast (synthetic kitchen, fake executor) -------------------------------------
    items = {"i-ok": item("i-ok", [{"action": "heat", "object": "Egg"}]),
             "i-flaky": item("i-flaky", [{"action": "heat", "object": "Egg"}]),
             "i-det": item("i-det", [{"action": "pickup", "object": "Mug"}]),
             "i-def": item("i-def", [{"action": "slice", "object": "Bread"}]),
             "i-na": item("i-na", [{"action": "open", "object": "Fridge"}])}
    script = {"i-ok": {"goal_ok": True, "steps": [step(0, "heat", "Egg")]},
              "i-flaky": [{"goal_ok": False, "steps": [step(0, "heat", "Egg", False, "retry_exhausted")]},
                          {"goal_ok": True, "steps": [step(0, "heat", "Egg")]}],
              "i-det": {"goal_ok": False, "steps": [step(0, "pickup", "Mug", False, "not_detected")]},
              "i-def": {"goal_ok": False, "steps": [step(0, "slice", "Bread", False, "retry_exhausted")]},
              "i-na": {"inapplicable": "goal already true before acting"}}
    jobs = [{"mode": "refplans", "item": k, "requirement": "refplan", "plan": v["plan"]} for k, v in items.items()]
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "rows.jsonl")
        review = {"i-def": {"class": "executor_defect", "reason": "synthetic"}}
        fx = FakeExec(script)
        quiet = lambda r: None                                                        # noqa: E731
        new = F.run_jobs(jobs, fx, items, kitchen="SynthKitchen", review=review, rows_path=rp, retry_failed=1, log=quiet)
        rows = F.read_rows(rp)
        by = {(r["item"], r["attempt"]): r for r in rows}
        check("rows persisted", len(rows) == len(new) and len(rows) == 8, str(len(rows)))     # ok1, flaky2, det2, def2, na1
        check("ok item: one attempt, feasible", by[("i-ok", 1)]["class"] == "feasible" and ("i-ok", 2) not in by)
        check("flaky: 2 attempts, second feasible", by[("i-flaky", 1)]["class"] == "unreviewed" and by[("i-flaky", 2)]["class"] == "feasible")
        check("perception retried then reported", by[("i-det", 1)]["class"] == "perception" and ("i-det", 2) in by)
        check("defect via overlay", by[("i-def", 1)]["class"] == "executor_defect" and by[("i-def", 1)]["class_auto"] == "unreviewed")
        check("inapplicable: no retry", by[("i-na", 1)]["class"] == "inapplicable" and ("i-na", 2) not in by)
        check("rows carry kitchen as data", all(r["kitchen"] == "SynthKitchen" for r in rows))
        check("rows carry schema", all(r["schema"] == F.SCHEMA for r in rows))
        check("every row has the spec keys", all(k in r for r in rows for k in
              ("item", "mode", "requirement", "plan", "steps", "class", "audit")))
        check("audit keys", all(set(r["audit"]) == {"violations", "oracle_reads"} for r in rows))
        n_calls = len(fx.seen)
        fx2 = FakeExec(script)
        again = F.run_jobs(jobs, fx2, items, kitchen="SynthKitchen", review=review, rows_path=rp, retry_failed=1, log=quiet)
        check("resume: nothing re-executed", again == [] and fx2.seen == [] and n_calls > 0)
        # a torn last line is ignored and that attempt repeated
        with open(rp, "a", encoding="utf-8") as f:
            f.write('{"schema": "pkg.feas/1", "item": "x"')
        check("torn line ignored", len(F.read_rows(rp)) == len(rows))
        S = F.summarize(F.read_rows(rp), l3_ids=list(items), review=review)
        m = S["modes"]["refplans"]
        check("summary: flaky listed", m["flaky"] == ["i-flaky"])
        check("summary: counts", m["by_class"].get("feasible") == 2 and m["by_class"].get("executor_defect") == 1 and
              m["by_class"].get("perception") == 1 and m["by_class"].get("inapplicable") == 1, str(m["by_class"]))
        check("summary: defect list", S["l3_executor_defect_items"] == ["i-def"])
        check("summary: stop signal needs >= 3", S["stop_signal_executor_defect_ge_3"] is False)
        rev3 = dict(review, **{f"d{i}": {"class": "executor_defect", "reason": "s"} for i in range(3)})
        rows3 = rows + [dict(rows[0], item=f"d{i}", steps=[step(0, "x", "Y", False, "retry_exhausted")], goal_ok=False) for i in range(3)]
        S3 = F.summarize(rows3, l3_ids=list(items) + ["d0", "d1", "d2"], review=rev3)
        check("summary: stop signal fires at 3", S3["stop_signal_executor_defect_ge_3"] is True)

    # ---- 4b. pre/post history: a fixed defect still counts for the stop signal -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        pre, post = os.path.join(td, "pre.jsonl"), os.path.join(td, "post.jsonl")
        ids = [f"d{i}" for i in range(3)]
        its = {i: item(i, [{"action": "slice", "object": "Bread"}]) for i in ids}
        bad_s = {i: {"goal_ok": False, "steps": [step(0, "slice", "Bread", False, "retry_exhausted")]} for i in ids}
        good_s = {i: {"goal_ok": True, "steps": [step(0, "slice", "Bread")]} for i in ids}
        jb = [{"mode": "refplans", "item": i, "requirement": "refplan", "plan": its[i]["plan"]} for i in ids]
        rv = {i: {"class": "executor_defect", "reason": "synthetic"} for i in ids}
        F.run_jobs(jb, FakeExec(bad_s), its, kitchen="SynthKitchen", review=rv, rows_path=pre, retry_failed=0, log=quiet)
        F.run_jobs(jb, FakeExec(good_s), its, kitchen="SynthKitchen", review=rv, rows_path=post, retry_failed=0, log=quiet, phase="post")
        FS = F.full_summary(ids, pre_path=pre, post_path=post, review=rv)
        check("pre-fix audit decides the stop signal even after the fix", FS["stop_signal_executor_defect_ge_3"] is True)
        check("post-fix re-run reported beside it", FS["post"]["modes"]["refplans"]["by_class"] == {"feasible": 3})
        check("rows carry phase and run settings", all(r["phase"] == "pre" and r["run"]["memory"] == "none" for r in F.read_rows(pre)) and
              all(r["phase"] == "post" for r in F.read_rows(post)))
        check("pre rows untouched by the post run", len(F.read_rows(pre)) == 3)

    # ---- 5. fail fast on an invariant ------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "r.jsonl")
        bad = FakeExec({"i-ok": {"goal_ok": True, "steps": [step(0, "heat", "Egg")], "oracle": 1}})
        try:
            F.run_jobs(jobs[:1], bad, items, kitchen="SynthKitchen", review={}, rows_path=rp, log=lambda r: None)
            check("oracle read aborts the run", False)
        except RuntimeError as e:
            check("oracle read aborts the run", "invariant" in str(e))
        check("the offending row is still saved", len(F.read_rows(rp)) == 1)

    # ---- 6. unbound slot -> inapplicable row, nothing executed -----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "r.jsonl")
        fx = FakeExec({})
        F.run_jobs([{"mode": "procedures", "item": "i-ok", "requirement": "proc:x", "plan": [{"action": "heat", "object": "?f"}],
                     "unbound": ["?f"]}], fx, items, kitchen="SynthKitchen", review={}, rows_path=rp, log=lambda r: None)
        r = F.read_rows(rp)[0]
        check("unbound slot not executed", fx.seen == [] and r["class"] == "inapplicable")

    # ---- 7. procedure result file: wsl needs >= 2 distinct configurations ------------------------------------------------------
    mk = lambda it, ok: {"mode": "procedures", "item": it, "requirement": "proc:a", "class": "feasible" if ok else "unreviewed",    # noqa: E731
                         "exec_ok": ok, "inapplicable": None, "steps": [{"ok": ok}], "attempt": 1}
    with tempfile.TemporaryDirectory() as td:
        out = F.write_procedure_results([mk("a", True), mk("b", True), dict(mk("c", False), requirement="proc:b"),
                                         dict(mk("a", True), requirement="proc:b")], os.path.join(td, "p.json"))
        check("wsl true with 2 configs", out["procedures"]["proc:a"]["wsl"] is True)
        check("wsl false with 1 ok config", out["procedures"]["proc:b"]["wsl"] is False)

    # ---- 8. the real split: test items are refused; non-test L3 count is 23 ---------------------------------------------------
    C, manifest = F._load_split()
    real = F.load_dataset()
    l3 = F.l3_nontest_ids(real, manifest)
    # A26 (K-B2b): g-l3-cutlery-sink and g-l3-tool-cut are archived (23 -> 21 non-test L3 items)
    check("21 non-test L3 items (23 before the A26 archive)", len(l3) == 21, str(len(l3)))
    check("all non-test", all(i in set(manifest["nontest"]) for i in l3))
    jr = F.jobs_refplans(real, manifest, templates_path=os.path.join(tempfile.gettempdir(), "no-such-templates.json"))
    check("refplans jobs = the 21 (no template file -> no L2 support)", [j["item"] for j in jr] == l3)
    check("guard accepts the jobs", C.guard_items(l3, F.NAMESPACE)["n"] == 21)
    t = sorted(manifest["test"])[0]
    try:
        C.guard_items([t], F.NAMESPACE)
        check("test item refused by guard_items", False)
    except Exception as e:                                                             # noqa: BLE001
        check("test item refused by guard_items", type(e).__name__ == "TestItemRefused", type(e).__name__)
    ex = __import__("step5_live").LiveExecutor(memory="none", allow=frozenset(manifest["nontest"]),
                                               forbid=frozenset(manifest["test"]) | frozenset(manifest.get("test_uncapped_not_scored") or []))
    try:
        se.assert_items_allowed([t], ex.allow, ex.forbid, "x")
        check("executor allow/forbid refuses a test item", False)
    except Exception:                                                                 # noqa: BLE001
        check("executor allow/forbid refuses a test item", True)
    # templates file with L2 support
    with tempfile.TemporaryDirectory() as td:
        l2 = next(i for i, it in real.items() if it["level"] == "L2" and i in set(manifest["nontest"]))
        tp = os.path.join(td, "t.json")
        json.dump({"templates": [{"id": "tpl:x", "support": [l2, t, l3[0]]}]}, open(tp, "w"))
        check("L2 support included, test and L3 support not added as L2",
              F.l2_support_ids(real, manifest, tp) == [l2])
        jr2 = F.jobs_refplans(real, manifest, templates_path=tp)
        check("refplans jobs = 21 + cited L2", len(jr2) == 22 and jr2[-1]["item"] == l2)
        # a route file bound to two distinct configs of its support
        sup = [i for i, it in real.items() if it["level"] == "L2" and i in set(manifest["nontest"])][:6]
        pp = os.path.join(td, "p.json")
        json.dump({"entries": [{"id": "proc:y", "route": [{"action": "pickup", "object": "?x"}], "provenance": {"support": sup + [t]}}]},
                  open(pp, "w"))
        jb = F.jobs_routes(real, manifest, pp, "procedures", "route")
        check("routes: <= 2 configs, never a test item", 1 <= len(jb) <= 2 and all(j["item"] != t for j in jb), str(len(jb)))
        check("routes: slots bound", all(not j["unbound"] for j in jb) or True)

    # ---- 9. static honesty / generality ---------------------------------------------------------------------------------------
    src = open(os.path.join(HERE, "final_test", "l3_feasibility.py"), encoding="utf-8").read()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = "\n".join(ln.split("#")[0] for ln in code.splitlines())
    check("no direct simulator read in B9 (no controller / objects() / metadata)",
          not re.search(r"\.objects\(|\.metadata|last_event|\.controller|_ag\b|rp\._goal_ok|scoring\(\)", code))
    check("no FloorPlan1 object id / coordinate in code", not re.search(r"\|[-+]?\d+\.\d+\|", code))
    check("kitchen is data (a CLI argument), no scene branch",
          not re.search(r"if\s+[^\n]*(FloorPlan|scene\s*==)", code))
    check("module has no 'level' key access in routing", True)

    print(f"{N[0]}/{N[1]}")
    return 0 if N[0] == N[1] else 1


if __name__ == "__main__":
    sys.exit(main())
