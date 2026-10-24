"""Offline selftest for step 4 s5-refplans (sim/step4_refplans.py): the pure summary, cause counting and resume logic.

Windows `python`, no WSL/GPU/Ollama. Run from the repo root:  python sim/_step4_refplans_selftest.py

Covers: setup holding fold; retry_exhausted unwrapping; item cause classification; tolerant jsonl reading (a
truncated last line); resume by config hash (completed rows only, HUNG re-run); latest-row selection; per-level
tally + success; cause / underlying / step-cause histograms; mean retries over run items only; audit sums incl.
pre-episode reads; paired outcomes on ids eligible in both arms + gate disagreements; baseline reproduction flag;
the markdown report; `--report` end to end in a temp dir; the 5 smoke ids exist in the dataset and span L1-L3.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import step4_refplans as rp                                           # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def row(id_, level, status, mode="honest", h="H1", cause=None, steps=(), retries=0, prims=0, audit=None, pre=None):
    ff = next(({"i": s["i"], "cause": s["cause"], "under": s.get("under")} for s in steps if not s["ok"]), None)
    return {"id": id_, "level": level, "status": status, "mode": mode, "hash": h, "cause": cause,
            "steps": list(steps), "first_fail": ff, "retries": retries, "prims": prims,
            "audit": audit or {"counts": {"pose": 10, "actuator": 5, "action-resolve": 1, "scoring": 1, "oracle": 0},
                               "violations": 0},
            "pre": pre or {"scoring": 2, "oracle_setup": 0}}


def st(i, ok, cause=None, under=None):
    return {"i": i, "action": "open", "object": "Fridge", "ok": ok, "cause": cause, "under": under}


print("1. pure helpers")
check("holding: last successful pickup", rp.setup_holding(
    [{"action": "pickup", "object": "Mug"}, {"action": "dirty", "object": "Bowl"}],
    [{"ok": True}, {"ok": True}]) == "Mug")
check("holding: a failed pickup does not count", rp.setup_holding(
    [{"action": "pickup", "object": "Mug"}], [{"ok": False}]) is None)
check("holding: a later release clears it", rp.setup_holding(
    [{"action": "pickup", "object": "Mug"}, {"action": "put_in", "object": "Sink"}], [{"ok": True}, {"ok": True}])
      is None)
check("holding: no setup -> None", rp.setup_holding([], []) is None)
check("underlying: retry_exhausted unwraps the wrapped cause",
      rp.underlying("retry_exhausted", "not_detected: Apple not in view") == "not_detected")
check("underlying: primitive cap stays retry_exhausted",
      rp.underlying("retry_exhausted", "primitive cap 700 reached") == "retry_exhausted")
check("underlying: other causes pass through", rp.underlying("unreachable", "x") == "unreachable")
check("classify ok/skipped -> None", rp.classify({"status": "ok"}) is None and rp.classify({"status": "skipped"})
      is None)
check("classify HUNG -> hung", rp.classify({"status": "HUNG"}) == "hung")
check("classify FAIL uses the row cause", rp.classify({"status": "FAIL", "cause": "wrong_target"}) == "wrong_target")
check("classify FAIL without steps -> goal_unmet", rp.classify({"status": "FAIL"}) == "goal_unmet")
check("every step cause is an item cause", set(rp.STEP_CAUSES) <= set(rp.ITEM_CAUSES))
import honest_skills as hs                                            # noqa: E402
check("STEP_CAUSES mirror honest_skills.CAUSES", tuple(rp.STEP_CAUSES) == tuple(hs.CAUSES))
check("config_hash is order-independent and sensitive",
      rp.config_hash({"a": 1, "b": 2}) == rp.config_hash({"b": 2, "a": 1}) != rp.config_hash({"a": 1, "b": 3}))

print("\n2. jsonl + resume")
tmp = tempfile.mkdtemp(prefix="s5test_")
try:
    jl = os.path.join(tmp, "x.jsonl")
    with open(jl, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row("a", "L1", "ok")) + "\n")
        fh.write(json.dumps(row("b", "L1", "HUNG")) + "\n")
        fh.write(json.dumps(row("c", "L2", "FAIL", h="OLD")) + "\n")
        fh.write('{"id": "d", "level": "L1", "sta')                 # a run killed mid-write
    rows = rp.read_jsonl(jl)
    check("truncated last line skipped, others kept", [r["id"] for r in rows] == ["a", "b", "c"])
    check("done_ids: completed rows of the same hash only (HUNG re-run, other hash re-run)",
          rp.done_ids(rows, "H1") == {"a"})
    check("read_jsonl of a missing file -> []", rp.read_jsonl(os.path.join(tmp, "none.jsonl")) == [])
    lat = rp.latest([row("a", "L1", "HUNG"), row("a", "L1", "ok"), row("a", "L1", "HUNG")])
    check("latest prefers the last COMPLETED row over a later HUNG attempt", lat["a"]["status"] == "ok")
    lat = rp.latest([row("a", "L1", "FAIL"), row("a", "L1", "ok")])
    check("latest takes the newest completed row", lat["a"]["status"] == "ok")

    print("\n3. summaries")
    honest = [row("o1", "L1", "ok", retries=1, prims=100),
              row("o2", "L1", "FAIL", cause="not_detected", steps=[st(0, False, "not_detected", "not_detected")],
                  retries=3, prims=300),
              row("o3", "L1", "skipped", retries=0, prims=0),
              row("o4", "L2", "FAIL", cause="retry_exhausted",
                  steps=[st(0, True), st(1, False, "retry_exhausted", "wrong_target"),
                         st(2, False, "unreachable", "unreachable")], retries=3, prims=500),
              row("o5", "L2", "FAIL", cause="goal_unmet", steps=[st(0, True)], prims=200),
              row("o6", "L3", "ok", steps=[st(0, False, "state_unrecognised", "state_unrecognised")], retries=2,
                  prims=400),
              row("o7", "L3", "HUNG", cause="hung")]
    oracle = [row(r["id"], r["level"], "ok", mode="oracle", h="O1") for r in honest]
    oracle[1]["status"] = "skipped"                                  # gate disagreement on o2
    sh = rp.summarize_mode(honest)
    check("tally", sh["tally"] == {"ok": 2, "FAIL": 3, "skipped": 1, "HUNG": 1}, str(sh["tally"]))
    check("per-level L1 ok/FAIL/skip + success over run items",
          sh["per_level"]["L1"]["ok"] == 1 and sh["per_level"]["L1"]["FAIL"] == 1
          and sh["per_level"]["L1"]["skipped"] == 1 and sh["per_level"]["L1"]["success"] == 0.5,
          str(sh["per_level"]["L1"]))
    check("L3 success counts HUNG as not ok", sh["per_level"]["L3"]["success"] == 0.5)
    check("cause histogram (item level)", sh["causes"] == {"not_detected": 1, "retry_exhausted": 1,
                                                           "goal_unmet": 1, "hung": 1}, str(sh["causes"]))
    check("underlying histogram unwraps retry_exhausted -> wrong_target",
          sh["underlying"] == {"not_detected": 1, "wrong_target": 1}, str(sh["underlying"]))
    check("step-cause histogram counts every failed step (incl. one in an ok item)",
          sh["step_causes"] == {"not_detected": 1, "retry_exhausted": 1, "unreachable": 1,
                                "state_unrecognised": 1}, str(sh["step_causes"]))
    check("mean retries over the 6 run items (skipped excluded)", sh["mean_retries"] == round(9 / 6, 2),
          str(sh["mean_retries"]))
    check("audit sums incl. pre-episode reads", sh["audit"]["pose"] == 70 and sh["audit"]["pre:scoring"] == 14
          and sh["violations"] == 0, str(sh["audit"]))
    pr = rp.paired(oracle, honest)
    check("paired on ids eligible in both (o2 + o3 out)", pr["n"] == 5, str(pr))
    check("paired outcomes", pr["both_ok"] == 2 and pr["oracle_only"] == 3 and pr["honest_only"] == 0
          and pr["both_fail"] == 0, str(pr))
    check("gate disagreements listed (o2 oracle-skip, o3 honest-skip)", pr["skip_disagree"] == ["o2", "o3"], str(pr["skip_disagree"]))
    # s7 audit: an honest episode with a violation / oracle / map-build read is INVALID: never scored, never paired
    bad_v = row("v1", "L1", "ok", audit={"counts": {"pose": 3, "oracle": 0}, "violations": 1, "invalid": True})
    bad_o = row("v2", "L1", "FAIL", audit={"counts": {"pose": 3, "oracle": 2}, "violations": 0})
    bad_m = row("v3", "L2", "ok", audit={"counts": {"pose": 3, "map-build": 1}, "violations": 0})
    check("invalid(): violation, oracle read, map-build read; oracle-arm rows never",
          rp.invalid(bad_v) and rp.invalid(bad_o) and rp.invalid(bad_m) and not rp.invalid(honest[0])
          and not rp.invalid(row("x", "L1", "ok", mode="oracle", audit={"counts": {"oracle": 9}, "violations": 0})))
    sv = rp.summarize_mode(honest + [bad_v, bad_o, bad_m])
    check("invalid episodes are left out of tally / rates and listed", sv["tally"] == sh["tally"]
          and sv["per_level"]["L1"] == sh["per_level"]["L1"] and sv["invalid"] == ["v1", "v2", "v3"]
          and sv["violations"] == 1, str(sv["tally"]) + str(sv["invalid"]))
    pv = rp.paired(oracle + [row(i, "L1", "ok", mode="oracle", h="O1") for i in ("v1", "v2", "v3")],
                   honest + [bad_v, bad_o, bad_m])
    check("invalid episodes are left out of the pairing", pv["n"] == pr["n"] and pv["invalid_excluded"] ==
          ["v1", "v2", "v3"], str(pv))
    s = rp.summarize(oracle, honest)
    check("baseline flag False when the oracle tally differs", s["baseline_reproduced"] is False)
    fake_o = ([row(f"k{i}", "L1", "ok", mode="oracle") for i in range(263)]
              + [row(f"s{i}", "L2", "skipped", mode="oracle") for i in range(10)]
              + [row(f"f{i}", "L2", "FAIL", mode="oracle", cause="goal_unmet") for i in range(4)])
    check("baseline flag True on 263/10/4", rp.summarize(fake_o, [])["baseline_reproduced"] is True)
    check("baseline flag None without an oracle run", rp.summarize([], honest)["baseline_reproduced"] is None)
    md = rp.render_md(s)
    check("markdown has the level table and the causes", "| L1 |" in md and "not_detected" in md
          and "Paired on 5" in md and "execution comparison" in md)
    check("markdown renders with empty inputs", "| L3 | - | - |" in rp.render_md(rp.summarize([], [])))

    print("\n4. --report end to end + smoke picks")
    for mode, rows_ in (("oracle", oracle), ("honest", honest)):
        with open(rp.jsonl_path(mode, tmp), "w", encoding="utf-8") as fh:
            for r in rows_:
                fh.write(json.dumps(r) + "\n")
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        code = rp.main(["--report", "--out-dir", tmp])
    js = json.load(open(os.path.join(tmp, "refplans_summary.json"), encoding="utf-8"))
    check("--report writes summary json + md", code == 0 and js["honest"]["tally"]["ok"] == 2
          and os.path.exists(os.path.join(tmp, "refplans.md")))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

items = json.load(open(rp.DATASET, encoding="utf-8"))["items"]
sm = rp.pick_smoke(items)
check("all 5 smoke ids exist", len(sm) == 5, str([i["id"] for i in sm]))
check("smoke spans L1, L2, L3, a setup item and a composite",
      {i["level"] for i in sm} == {"L1", "L2", "L3"} and any(i.get("setup") for i in sm)
      and any(s["action"] in ("wash", "pick_and_place") for i in sm for s in i["plan"]))
src = open(os.path.join(HERE, "step4_refplans.py"), encoding="utf-8").read()
check("the runner calls exec_step, scoring inside h.scoring(), and never passes object ids",
      "sk.exec_step(st)" in src and "with h.scoring():" in src and "object_id" not in src)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("STEP4 REFPLANS SELFTEST OK")
