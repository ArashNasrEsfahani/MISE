"""Offline selftest of the K23 supervisor decisions (k23_decide.py): every stop / retry / done path.

    python sim/kitchen_v3/k23/_k23_decide_selftest.py
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (SIM, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import k23_capture  # noqa: E402
import k23_decide as kd  # noqa: E402
import thermal_guard as kd_tg  # noqa: E402

N = {"ok": 0, "fail": 0}


def ok(cond, name):
    N["ok" if cond else "fail"] += 1
    if not cond:
        print(f"FAIL {name}")


tmp = tempfile.mkdtemp(prefix="k23dec_")
try:
    # ------------------------------------------------------------------ auth
    for s in kd.SCRIPTS:
        open(os.path.join(tmp, s), "w").write(s)
    now = dt.datetime(2026, 10, 14, 7, 35)
    good = {"template": False, "signed_by": "Arash", "ledger_entry": "2026-10-14 07:20 -- ARASH K23 AUTHORIZATION",
            "date": "2026-10-14",
            "window": {"start": "07:30", "end": "10:30"}, "scenes": ["FloorPlan15"],
            "pipelines": ["probe", "thermaltest", "main"], "freeze_summary_entry": "2026-10-14 07:25 -- FREEZE SUMMARY",
            "freeze_summary_read_by_arash": True,
            "a28_items": {"i_signoff": "signed", "ii_resume": "a pause ends the window", "iii_barrier": "no viewing",
                          "iv_agent_side": "deadline amended", "v_deferral": "deferred"},
            "thermal": kd.thermal_policy(),
            "script_sha256": {s: kd.sha_file(os.path.join(tmp, s)) for s in kd.SCRIPTS}}
    heads = ["## 2026-10-14 07:20 -- ARASH K23 AUTHORIZATION", "## 2026-10-14 07:25 -- FREEZE SUMMARY"]
    ok(kd.check_auth(good, "FloorPlan15", "main", now, heads, tmp) == [], "auth: a complete record authorizes")

    def bad(mut, name, scene="FloorPlan15", pipe="main", when=now, hd=heads):
        r = copy.deepcopy(good)
        mut(r)
        ok(kd.check_auth(r, scene, pipe, when, hd, tmp) != [], name)
    bad(lambda r: r.update(template=True), "auth: the template is refused")
    bad(lambda r: r.pop("template"), "auth: a record without template:false is refused")
    bad(lambda r: r["a28_items"].pop("v_deferral"), "auth: item 15.6 (v) is required")
    bad(lambda r: r["thermal"].update(resume="person decides"), "auth: the thermal resume rule must be the implemented one")
    bad(lambda r: r["thermal"].update(pause_c=85), "auth: 85 C needs the guard changed first")
    bad(lambda r: r["thermal"].update(sm_min_mhz=900.0), "auth: every thermal rule value must match the guard")
    bad(lambda r: r["thermal"].pop("no_telemetry_s"), "auth: a thermal rule value left out is refused")
    bad(lambda r: r.update(thermal="pause_ends_window"), "auth: a thermal policy that is not a record is refused")
    bad(lambda r: None, "auth: 07:20 is before the window (no early start)", when=dt.datetime(2026, 10, 14, 7, 20))
    ok(kd.check_auth(good, "FloorPlan15", "main", dt.datetime(2026, 10, 14, 7, 20), heads, tmp, check_only=True) == [],
       "auth: --check-only works before the window")
    bad(lambda r: None, "auth: a heading that only CONTAINS the title is not exact",
        hd=["## 2026-10-14 07:20 -- ARASH K23 AUTHORIZATION (signed)", "## 2026-10-14 07:25 -- FREEZE SUMMARY"])
    bad(lambda r: r.update(date="2026-10-13"), "auth: yesterday's record is refused")
    bad(lambda r: None, "auth: before the window", when=dt.datetime(2026, 10, 14, 6, 0))
    bad(lambda r: None, "auth: after the window", when=dt.datetime(2026, 10, 14, 10, 31))
    bad(lambda r: None, "auth: FloorPlan17 not authorized", scene="FloorPlan17")
    bad(lambda r: r.update(pipelines=["probe"]), "auth: main not authorized")
    bad(lambda r: r.update(freeze_summary_read_by_arash=False), "auth: freeze summary not read")
    bad(lambda r: r["a28_items"].update(iv_agent_side=""), "auth: an open A28 item is refused")
    bad(lambda r: r.update(signed_by="Claude"), "auth: not signed by Arash")
    bad(lambda r: r.update(ledger_entry="ARASH K23 AUTHORIZATION"), "auth: an undated heading is refused",
        hd=["## ARASH K23 AUTHORIZATION", "## 2026-10-14 07:25 -- FREEZE SUMMARY"])
    bad(lambda r: None, "auth: no such heading", hd=["## 2026-10-14 something else"])
    bad(lambda r: r["script_sha256"].pop("k23_qc.py"), "auth: a script not pinned is refused")
    # A28 15.9: the parallel basis (Arash 2026-10-14 ~14:25) replaces the freeze-summary fields, nothing else
    PH = "2026-10-14 ~14:25 -- " + kd.PARALLEL_DECISION_TITLE
    par = dict(copy.deepcopy(good), precondition="parallel_k1_priority", parallel_decision_entry=PH,
               kitchen1_priority=True)
    for k_ in ("freeze_summary_entry", "freeze_summary_read_by_arash"):
        par.pop(k_)
    ok(kd.check_auth(par, "FloorPlan15", "main", now, heads + ["## " + PH], tmp) == [],
       "auth 15.9: a complete parallel-basis record authorizes without a freeze summary")
    for mut_, name_ in ((dict(kitchen1_priority=False), "kitchen1_priority false"),
                        (dict(kitchen1_priority="yes"), "kitchen1_priority not a Boolean"),
                        (dict(parallel_decision_entry="2026-10-14 -- something else"), "another decision entry"),
                        (dict(precondition="whenever"), "an unknown precondition"),
                        (dict(a28_items=dict(par["a28_items"], ii_resume="")), "an open 15.6 item")):
        ok(kd.check_auth(dict(par, **mut_), "FloorPlan15", "main", now, heads + ["## " + PH], tmp) != [],
           f"auth 15.9: {name_} is refused")
    ok(kd.check_auth(par, "FloorPlan15", "main", now, heads, tmp) != [],
       "auth 15.9: the decision heading must exist exactly in the ledger")
    ok(kd.check_auth(dict(par, precondition="freeze_summary"), "FloorPlan15", "main", now, heads + ["## " + PH], tmp)
       != [], "auth: the freeze-summary basis still needs its own fields")
    open(os.path.join(tmp, "k23_qc.py"), "w").write("changed")
    ok(kd.check_auth(good, "FloorPlan15", "main", now, heads, tmp) != [], "auth: a script changed after signing")

    # ------------------------------------------------------------------ pre
    S0 = {"resets": 0, "idle_runs": 0, "hang_retries": 0, "batches": 0, "last_done": None, "need_reset": True,
          "history": []}
    T = 1_000_000.0
    d = T + 3 * 3600
    S0 = kd.new_session()

    def pre(sess, now_, q=None, end=None):
        return kd.decide_pre(sess, now_, d, 30, q, "main", d if end is None else end)
    s_ = copy.deepcopy(S0)
    r = pre(s_, T)
    ok(r["action"] == "launch" and len(r["invocation"]) == 32 and s_["outstanding"] == r["invocation"],
       "pre: launch with a fresh id, recorded as outstanding before the launch")
    ok(pre(s_, T)["action"] == "stop", "pre: an outstanding invocation (no decided receipt) blocks the next launch")
    ok(pre(dict(copy.deepcopy(S0), terminal={"why": "x"}), T)["action"] == "stop", "pre: a terminal stop blocks")
    ok(pre(copy.deepcopy(S0), T, end=d + 3600)["action"] == "stop", "pre: -Deadline must equal the signed window end")
    ok(pre(copy.deepcopy(S0), T, q={"batch": 2, "verdict": "WARN"})["action"] == "stop", "pre: an unresolved QC batch stops")
    ok(pre(copy.deepcopy(S0), d - 1800)["action"] == "done", "pre: 30 min before the deadline = done")
    ok(pre(copy.deepcopy(S0), d - 3000)["action"] == "launch", "pre: 50 min before = still launches")
    ok(pre(dict(copy.deepcopy(S0), idle_runs=2), T)["action"] == "stop", "pre: two idle runs stop")
    ok(kd.decide_pre(copy.deepcopy(S0), T, d, 30, None, "main", d, ["batch_002.json: FAIL"])["action"] == "stop",
       "pre: any unresolved non-PASS QC report stops")

    # ------------------------------------------------------------------ post
    INV = "a" * 32
    QCD = os.path.join(kd.REPO, "sim", "kitchen_v3", "k23", "_selftest_qc")      # same drive as the repo (relpath)
    shutil.rmtree(QCD, ignore_errors=True)
    os.makedirs(QCD)
    LINES = ([{"status": "PASS", "check": c, "detail": ""} for c in kd.REQUIRED_CHECKS + kd.TRAIN_CHECKS
              if c != "cc_pace"]
             + [{"status": "NOTE", "check": "cc_pace", "detail": "R1"},
                {"status": "NOTE", "check": "cc_projection", "detail": ""}])
    LINES_F = LINES + [{"status": "FAIL", "check": "pairing_extra", "detail": "x"}]
    rep = os.path.join(QCD, "batch_001.json")
    json.dump({"scene": "FloorPlan15", "batch": 1, "episodes": [0, 1, 2], "episodes_by_split": kd.plan_splits("FloorPlan15", [0, 1, 2]), "verdict": "PASS", "lines": LINES},
              open(rep, "w"))
    rel = os.path.relpath(rep, kd.REPO).replace("\\", "/")
    R0 = {"invocation": INV, "scene": "FloorPlan15", "pipeline": "main", "capture_rc": 0, "qc_rc": 0, "thermal": False,
          "done_before": 0, "done_after": 3, "new_episodes": [0, 1, 2], "new_skipped": [], "failed_after": 0,
          "skipped_not_train": [], "skipped_unexplained": [],
          "guard_ok": True,
          "qc_report": rel, "qc_report_sha256": kd.sha_file(rep), "guard_burst": False, "complete": False}

    def post(rec, wsl=None, sess=None, now_=T, fb=False, pipe="main", qbad=None, markers=None):
        sess = sess if sess is not None else dict(copy.deepcopy(S0), outstanding=INV)
        w = rec["capture_rc"] if (wsl is None and isinstance(rec, dict) and "capture_rc" in rec) else (wsl or 0)
        return kd.decide_post(rec, INV, "FloorPlan15", pipe, w, sess, now_, d, fb, qbad, QCD, markers=markers), sess

    r, s1 = post(dict(R0))
    ok(r["action"] == "continue" and s1["batches"] == 1 and s1["idle_runs"] == 0, "post: QC PASS continues")
    ok(post(dict(R0), fb=True)[0]["action"] == "done", "post: first batch only stops by design")
    ok(post(dict(R0, complete=True))[0]["action"] == "done", "post: main complete")
    ok(post(None)[0]["action"] == "stop", "post: no receipt stops")
    ok(post(dict(R0, invocation="b" * 32))[0]["action"] == "stop", "post: a stale receipt stops")
    ok(post(dict(R0, scene="FloorPlan17"))[0]["action"] == "stop", "post: receipt of another scene stops")
    r2 = dict(R0)
    r2.pop("qc_rc")
    ok(post(r2)[0]["action"] == "stop", "post: a receipt without qc_rc stops (never cast to 0)")
    ok(post(dict(R0, qc_rc=True))[0]["action"] == "stop", "post: a Boolean is not an exit code")
    ok(post(dict(R0), wsl=1)[0]["action"] == "stop", "post: WSL rc != receipt rc stops")
    for q in (1, 2, 3, 4, 5, -1):
        ok(post(dict(R0, qc_rc=q))[0]["action"] == "stop", f"post: QC rc {q} with new episodes stops")
    ok(post(dict(R0, capture_rc=3, qc_rc=1, guard_burst=True))[0]["action"] == "stop",
       "post: QC checked BEFORE a render-guard retry")
    r, s1 = post(dict(R0, capture_rc=3, guard_burst=True))
    ok(r["action"] == "retry" and s1["burst_retries"] == 1, "post: QC PASS + first render-guard burst -> one resume")
    ok(post(dict(R0, capture_rc=3, guard_burst=False))[0]["action"] == "stop", "post: other fail-fast aborts stop")
    ok(post(dict(R0, qc_report="nope.json"))[0]["action"] == "stop", "post: a missing QC report stops")
    ok(post(dict(R0, qc_report_sha256="0" * 64))[0]["action"] == "stop", "post: a changed QC report stops")
    ok(post(dict(R0, new_episodes=[0, 1, 2, 9], done_after=4))[0]["action"] == "stop",
       "post: a QC report that misses a new episode stops")
    ok(post(dict(R0, new_episodes=[0, 1, 2, 9], done_after=4, new_skipped=[9]))[0]["action"] == "continue",
       "post: a quota-skipped TRAIN episode needs no QC row")
    ok(post(dict(R0, new_episodes=[0, 1, 2, 9], done_after=4, new_skipped=[9], skipped_not_train=[9]))[0]["action"]
       == "stop", "post: a held-out episode without a record stops")
    ok(post(dict(R0, new_episodes=[0, 1], done_after=3))[0]["action"] == "stop", "post: list / count mismatch stops")
    ok(post(dict(R0, new_episodes=[0, 0, 1]))[0]["action"] == "stop", "post: duplicate episode ids stop")
    r3 = dict(R0)
    r3.pop("qc_report_sha256")
    ok(post(r3)[0]["action"] == "stop", "post: a QC report without a hash stops")
    fr = os.path.join(QCD, "batch_002.json")
    json.dump({"scene": "FloorPlan15", "batch": 2, "episodes": [0, 1, 2], "episodes_by_split": kd.plan_splits("FloorPlan15", [0, 1, 2]), "verdict": "FAIL", "lines": LINES_F},
              open(fr, "w"))
    frel = os.path.relpath(fr, kd.REPO)
    ok(post(dict(R0, qc_report=frel, qc_report_sha256=kd.sha_file(fr)))[0]["action"] == "stop",
       "post: a FAIL report with qc_rc 0 still stops")
    # r2 6: a receipt with no episode list / no report hash + an existing FAIL report never continues
    r4 = dict(R0)
    r4.pop("new_episodes")
    ok(post(r4)[0]["action"] == "stop", "post: a receipt without an episode list stops")
    r5 = dict(R0)
    r5.pop("qc_report_sha256")
    ok(post(r5, qbad=kd.qc_unresolved("FloorPlan15", QCD))[0]["action"] == "stop",
       "post: no report hash + an existing FAIL report stops")
    ok(kd.qc_unresolved("FloorPlan15", QCD) == ["batch_002.json: FAIL"], "qc scan: the FAIL report is unresolved")
    ok(post(dict(R0), qbad=kd.qc_unresolved("FloorPlan15", QCD))[0]["action"] == "stop",
       "post: a clean receipt still stops while ANY earlier report is FAIL and unresolved")
    json.dump({"who": "Arash", "when": "08:10", "finding": "x", "decision": "stop"},
              open(os.path.join(QCD, "resolution_batch_002.json"), "w"))
    ok(kd.qc_unresolved("FloorPlan15", QCD) != [], "qc scan: a resolution with decision 'stop' does not release")
    json.dump({"who": "Arash", "when": "08:10", "finding": "x", "decision": "resume",
               "astra_file": "docs/final-test/_astra/K23-PREP.txt"},
              open(os.path.join(QCD, "resolution_batch_002.json"), "w"))
    ok(kd.qc_unresolved("FloorPlan15", QCD) == [], "qc scan: a valid resolution releases the FAIL report")
    open(os.path.join(QCD, "batch_003.json"), "w").write("{broken")
    ok(kd.qc_unresolved("FloorPlan15", QCD) != [], "qc scan: an unreadable report counts as unresolved")
    os.remove(os.path.join(QCD, "batch_003.json"))
    wr = os.path.join(QCD, "batch_004.json")
    json.dump({"scene": "FloorPlan15", "batch": 4, "episodes": [0, 1, 2], "episodes_by_split": kd.plan_splits("FloorPlan15", [0, 1, 2]), "verdict": "PASS",
               "lines": LINES + [{"status": "WARN", "check": "cc_pairs", "detail": ""}]}, open(wr, "w"))
    ok(post(dict(R0, qc_report=os.path.relpath(wr, kd.REPO), qc_report_sha256=kd.sha_file(wr)))[0]["action"] == "stop",
       "post: a 'PASS' report that carries a WARN line stops")
    os.remove(wr)
    er = os.path.join(QCD, "batch_005.json")
    json.dump({"scene": "FloorPlan15", "batch": 5, "episodes": [0, 1, 2], "episodes_by_split": kd.plan_splits("FloorPlan15", [0, 1, 2]), "verdict": "PASS", "lines": []}, open(er, "w"))
    ok(post(dict(R0, qc_report=os.path.relpath(er, kd.REPO), qc_report_sha256=kd.sha_file(er)))[0]["action"] == "stop",
       "post: a report with no check lines stops")
    os.remove(er)
    out_rep = os.path.join(kd.REPO, "sim", "kitchen_v3", "k23", "_selftest_batch_009.json")
    shutil.copy(rep, out_rep)
    ok(post(dict(R0, qc_report=os.path.relpath(out_rep, kd.REPO), qc_report_sha256=kd.sha_file(out_rep)))[0]["action"]
       == "stop", "post: a PASS report outside the scene's qc folder stops")
    os.remove(out_rep)
    ok(post(dict(R0, skipped_unexplained=[9], new_episodes=[0, 1, 2, 9], done_after=4, new_skipped=[9]))[0]["action"]
       == "stop", "post: a record-less 'done' episode without the quota-skip log line stops")
    r6 = dict(R0)
    r6.pop("skipped_unexplained")
    ok(post(r6)[0]["action"] == "stop", "post: a receipt without skipped_unexplained stops")
    ok(post(dict(R0, pipeline="probe", capture_rc=0, qc_rc=-1), pipe="probe")[0]["action"] == "stop",
       "post: a probe that commits episodes stops")
    ok(post(dict(R0, capture_rc=23, qc_rc=1, done_after=0, new_episodes=[]))[0]["action"] == "stop",
       "post: pre-QC of earlier commits not PASS stops (nothing captured)")
    r, s1 = post(dict(R0, capture_rc=23, qc_rc=5, done_after=0, new_episodes=[]))
    ok(r["action"] == "stop" and s1["terminal"], "post: a pre-QC stop is terminal")
    import re as _re
    logpart = ("[07:41:02] FloorPlan15: ep  12 [train] skipped: every train target already met\n"
               "[07:41:03] FloorPlan15: ep 140 [train] skipped: every train target already met\n")
    ok({int(m) for m in _re.findall(r"ep\s+(\d+) \[train\] skipped: every train target already met", logpart)}
       == {12, 140}, "receipt: the quota-skip log lines are parsed")
    r, s1 = post(dict(R0, thermal=True, capture_rc=86))
    ok(s1["terminal"] and s1["terminal"]["window_ended"], "post: a thermal pause is terminal for the window")
    # r2 2 reproduction: thermal post -> stop, then pre must NOT launch (even with a 'resolution' file)
    kd.resolve_terminal("FloorPlan15", s1)
    ok(s1["terminal"] and kd.decide_pre(s1, T, d, 30, None, "main", d)["action"] == "stop",
       "pre after a thermal pause: no launch in this window (no resolution lifts it)")
    s2 = copy.deepcopy(s1)
    post(dict(R0, capture_rc=124), sess=s2)
    ok(s2["terminal"]["window_ended"], "a later stop never clears the window-ended flag")
    r, s1 = post(None)
    ok(s1["outstanding"] == INV, "post: an unreadable receipt keeps the invocation outstanding")
    r, s1 = post(dict(R0, capture_rc=3, guard_burst=True, qc_rc=2, done_after=0, new_episodes=[]))
    ok(r["action"] == "stop", "post: zero-progress burst with QC rc 2 does not retry")
    r, s1 = post(dict(R0, capture_rc=3, guard_burst=True), sess=dict(copy.deepcopy(S0), outstanding=INV,
                                                                       burst_retries=1))
    ok(r["action"] == "stop", "post: only ONE render-burst retry per window")
    ok(post(dict(R0, thermal=True, capture_rc=86))[0]["action"] == "stop", "post: a thermal pause ends the window")
    ok(post(dict(R0, capture_rc=88, guard_ok=False))[0]["action"] == "stop", "post: guard died stops")
    ok(post(dict(R0, guard_ok=False))[0]["action"] == "stop", "post: no guard = stop even with rc 0")
    ok(post(dict(R0, failed_after=1))[0]["action"] == "stop", "post: a failed episode stops")
    r, s1 = post(dict(R0, capture_rc=9, qc_rc=-1, done_after=0, new_episodes=[], guard_ok=False))
    ok(r["action"] == "stop" and "lock busy" in r["why"] and not s1["terminal"],
       "post: lock busy (guard never started) = a plain, non-terminal stop")
    r, s1 = post(dict(R0, capture_rc=9, qc_rc=-1, done_after=0, new_episodes=[], guard_ok=False),
                 qbad=["batch_002.json: FAIL"])
    ok(r["action"] == "stop" and s1["terminal"], "post: lock busy with an unresolved FAIL report is still terminal")
    ok(post(dict(R0, capture_rc=21, qc_rc=-1, done_after=0, new_episodes=[]))[0]["action"] == "stop", "post: foreign simulator")
    ok(post(dict(R0, capture_rc=20, qc_rc=-1, done_after=0, new_episodes=[]))[0]["action"] == "done", "post: deadline, not started")
    ok(post(dict(R0, capture_rc=124), now_=d - 100)[0]["action"] == "done", "post: timeout at the deadline = hard stop")
    r, s1 = post(dict(R0, capture_rc=124))
    ok(r["action"] == "retry" and s1["hang_retries"] == 1, "post: first hang after progress -> one retry")
    ok(post(dict(R0, capture_rc=124), sess=dict(copy.deepcopy(S0), outstanding=INV, hang_retries=1))[0]["action"] == "stop",
       "post: second hang stops")
    ok(post(dict(R0, capture_rc=124, qc_rc=-1, done_after=0, new_episodes=[]))[0]["action"] == "stop",
       "post: hang without progress stops")
    r, s1 = post(dict(R0, qc_rc=4, done_after=0, new_episodes=[]))
    ok(r["action"] == "stop" and s1["idle_runs"] == 1, "post: a main batch with nothing new stops")
    ok(post(dict(R0, done_after=0, done_before=3, new_episodes=[]))[0]["action"] == "stop", "post: done count going down stops")
    ok(post(dict(R0, pipeline="probe", qc_rc=-1, done_after=0, new_episodes=[]), pipe="probe")[0]["action"] == "done",
       "post: probe done")
    # thermal test
    TT = dict(R0, pipeline="thermaltest", capture_rc=86, qc_rc=-1, thermal=True, done_after=0, new_episodes=[],
              new_skipped=[],
              pause_marker={"test": True, "induced": True, "reason": kd_tg.TEST_REASON,
                            "running_before": {"job": 1, "simulator": 1}},
              post_stop={"jobs_left": 0, "simulators_left": 0, "util_after_stop": [3, 2, 1]})
    ok(post(dict(TT), pipe="thermaltest")[0]["why"] == "THERMAL TEST PASS", "thermal test: pass")
    for mut, name in ((dict(post_stop={"jobs_left": 0, "simulators_left": 0, "util_after_stop": [90, 95, 10]}),
                       "only the last sample low"),
                      (dict(post_stop={"jobs_left": 0, "simulators_left": 0, "util_after_stop": [3, None, 1]}),
                       "a missing sample"),
                      (dict(post_stop={"jobs_left": 0, "simulators_left": 1, "util_after_stop": [3, 2, 1]}),
                       "simulator left"),
                      (dict(pause_marker={"test": True, "induced": True, "reason": kd_tg.TEST_REASON,
                                          "running_before": {"job": 1, "simulator": 0}}),
                       "simulator not running before"),
                      (dict(thermal=False), "pause did not fire")):
        ok(post(dict(TT, **mut), pipe="thermaltest")[0]["action"] == "stop", f"thermal test: {name} fails")

    # ------------------------------------------------------------------ r3 1: thermal closure on EVERY path
    RT = dict(R0, thermal=True, capture_rc=86, pause_marker={"test": False, "induced": False, "reason": "GPU temperature"})
    r, s1 = post(dict(RT), wsl=143)
    ok(r["action"] == "stop" and s1["terminal"]["window_ended"],
       "r3 1 (Astra repro): a valid thermal receipt with outer rc 143 ends the window (was 'rc mismatch', window open)")
    rpp = os.path.join(tmp, "session_resolution_1.json")
    json.dump({"who": "Arash", "when": "08:10", "finding": "x", "decision": "resume",
               "astra_file": "docs/final-test/_astra/K23-PREP.txt"}, open(rpp, "w"))
    _sd = kd.scene_dir
    kd.scene_dir = lambda s: tmp
    try:
        kd.resolve_terminal("FloorPlan15", s1)
    finally:
        kd.scene_dir = _sd
    ok(s1["terminal"] and kd.decide_pre(s1, T, d, 30, None, "main", d)["action"] == "stop",
       "r3 1: after that pause an accepted resolution still cannot allow a launch")
    os.remove(rpp)
    r, s1 = post(dict(RT, invocation="b" * 32))
    ok(r["action"] == "stop" and s1["terminal"]["window_ended"] and s1["outstanding"] == INV,
       "r3 1: a pause in an INVALID receipt (stale) still ends the window; the invocation stays outstanding")
    r, s1 = post(dict(R0), wsl=86)
    ok(s1["terminal"]["window_ended"], "r3 1: WSL rc 86 alone (receipt says no pause) ends the window")
    r, s1 = post(dict(R0), markers=[{"file": "pause_20261014-074000-cccccccc.json", "inv8": "cccccccc",
                                     "induced": False, "why": "GPU temperature"}])
    ok(r["action"] == "stop" and s1["terminal"]["window_ended"],
       "r3 1: a real pause marker of the day (another invocation) ends the window although this receipt is clean")
    r, s1 = post(dict(TT), pipe="thermaltest", wsl=143)
    ok(r["action"] == "stop" and not s1["terminal"]["window_ended"],
       "r3 1: the INDUCED pause-test trigger with outer rc 143 stops (rc mismatch) but does not end the window")
    r, s1 = post(dict(TT, pause_marker={"test": True, "induced": False, "reason": "no telemetry for 301 s",
                                        "running_before": {"job": 1, "simulator": 1}}), pipe="thermaltest")
    ok(r["action"] == "stop" and s1["terminal"]["window_ended"],
       "r3 1: a REAL pause during the pause test (telemetry lost) ends the window, never a test PASS")
    r, s1 = post(dict(TT, pause_marker={"test": True, "running_before": {"job": 1, "simulator": 1}}), pipe="thermaltest")
    ok(r["action"] == "stop" and s1["terminal"]["window_ended"], "r3 1: a test marker without 'induced' is a real pause")
    r, s1 = post(dict(TT), pipe="thermaltest", markers=[{"file": "pause_20261014-074000-aaaaaaaa.json",
                                                         "inv8": "aaaaaaaa", "induced": False, "why": "heat"}])
    ok(s1["terminal"]["window_ended"], "r3 1: a receipt copy says induced but the file marker says real -> real")
    ok(kd.real_pauses([], None, "thermaltest", 86, INV) != [], "r3 1: no receipt + rc 86 + no marker file = real")
    ok(kd.real_pauses([{"file": "f", "inv8": INV[:8], "induced": True}], None, "thermaltest", 86, INV) == [],
       "r3 1: no receipt + rc 86 + this invocation's induced test marker = the test, not a real pause")
    ok(kd.real_pauses([{"file": "f", "inv8": INV[:8], "induced": True}], None, "main", 86, INV) != [],
       "r3 1: an 'induced' marker never exempts a MAIN batch")
    # marker files: day filter, half-written, unreadable, unknown names
    TD = os.path.join(tmp, "thermal")
    os.makedirs(TD, exist_ok=True)
    json.dump({"test": True, "induced": True, "reason": kd_tg.TEST_REASON}, open(os.path.join(TD, "pause_20261014-073500-aaaaaaaa.json"), "w"))
    json.dump({"test": False, "reason": "heat"}, open(os.path.join(TD, "pause_20261012-090000-bbbbbbbb.json"), "w"))
    mks = kd.pause_markers("FloorPlan15", "2026-10-14", TD)
    ok(len(mks) == 1 and mks[0]["induced"], "markers: today's induced marker read; a marker of two days before (another window) ignored")
    open(os.path.join(TD, "pause_20261014-080000-cccccccc.json.tmp"), "w").write("{")
    open(os.path.join(TD, "pause_20261014-080500-dddddddd.json"), "w").write("{broken")
    open(os.path.join(TD, "pause_odd.json"), "w").write("{}")
    mks = kd.pause_markers("FloorPlan15", "2026-10-14", TD)
    ok(sum(1 for m in mks if not m["induced"]) == 3, "markers: half-written, unreadable and unknown-name markers are REAL")
    ok(kd.decide_pre(copy.deepcopy(S0), T, d, 30, None, "main", d, markers=mks)["action"] == "stop",
       "pre: a real marker of the day stops")
    s_ = copy.deepcopy(S0)
    kd.decide_pre(s_, T, d, 30, None, "main", d, markers=mks)
    ok(s_["terminal"]["window_ended"], "pre: ... and records the window as ended")
    ok(kd.decide_pre(copy.deepcopy(S0), T, d, 30, None, "main", d,
                     markers=[m for m in mks if m["induced"]])["action"] == "launch",
       "pre: only the induced test marker -> may launch")
    ok(kd.decide_pre(copy.deepcopy(S0), T, d, 30, None, "main", d, pending=["pending_x.json"])["action"] == "stop",
       "pre: a pending (receipt-less) batch blocks every later launch (r3 4)")
    # missing receipt through main(): a real marker still ends the window
    shutil.rmtree(TD)
    os.makedirs(TD)
    json.dump({"test": False, "reason": "heat"}, open(os.path.join(TD, "pause_20261014-074000-aaaaaaaa.json"), "w"))
    kd.scene_dir = lambda s: tmp
    try:
        for f_ in os.listdir(tmp):
            if f_.startswith("session_2026"):
                os.remove(os.path.join(tmp, f_))
        kd.main(["post", "--scene", "FloorPlan15", "--pipeline", "main", "--deadline", "2026-10-14 10:30",
                 "--invocation", INV, "--wsl-rc", "143"])
        ss = json.load(open(os.path.join(tmp, "session_20261014.json")))
        ok(ss["terminal"]["window_ended"] is True,
           "r3 1: missing receipt + a real marker -> the window is ended (closure kept without a receipt)")
    finally:
        kd.scene_dir = _sd
    # r3 3: a guard that had to be killed is never ok
    ok(post(dict(R0, guard_killed=True))[0]["action"] == "stop", "post: guard_killed in the receipt stops")
    ok(kd.GRACE_S == 300 and kd.OUTER_RESERVE_S == 60, "deadline: GRACE includes the 60 s outer reserve (r3 4)")
    ok("emergency_stop_k23.sh" in kd.SCRIPTS, "auth: the emergency stop script is pinned")

    # ------------------------------------------------------------------ r4 1: closure before fallible validation
    try:
        vb = kd.validate_receipt(dict(RT, new_episodes=[{}]), INV, "FloorPlan15", "main")
        ok(vb != [], "r4 1: validate_receipt on new_episodes=[{}] reports a problem instead of raising")
    except Exception as exc:
        ok(False, f"r4 1: validate_receipt raised on new_episodes=[{{}}] ({exc!r})")
    try:
        r, s1 = post(dict(RT, new_episodes=[{}]))
        ok(r["action"] == "stop" and s1["terminal"]["window_ended"],
           "r4 1 (Astra repro): a thermal receipt with new_episodes=[{}] ends the window (was a TypeError, window open)")
    except Exception as exc:
        ok(False, f"r4 1: decide_post raised on a malformed thermal receipt ({exc!r})")
    r, s1 = post(dict(TT, pause_marker={"test": True, "induced": False, "reason": "GPU temperature >= 87 C",
                                        "running_before": {"job": 1, "simulator": 1}}), pipe="thermaltest",
                 markers=[{"file": "pause_20261014-074000-aaaaaaaa.json", "inv8": INV[:8], "induced": True,
                           "why": kd_tg.TEST_REASON}])
    ok(r["action"] == "stop" and r["why"] != "THERMAL TEST PASS" and s1["terminal"]["window_ended"],
       "r4 1 (Astra repro): a REAL receipt marker beats an induced FILE marker (was THERMAL TEST PASS, window open)")
    ok(kd.real_pauses([{"file": "f", "inv8": INV[:8], "induced": True}],
                      {"thermal": True, "pause_marker": {"test": True, "induced": False, "reason": "heat"}},
                      "thermaltest", 86, INV) != [], "r4 1: real receipt copy + induced file = real")
    ok(kd.real_pauses([{"file": "f", "inv8": INV[:8], "induced": True}],
                      {"thermal": False, "pause_marker": {"test": False, "reason": "heat"}}, "main", 0, INV) != [],
       "r4 1: a receipt that carries a real marker copy is a pause even if it says thermal false")
    ok(kd.real_pauses(["garbage", None], {"thermal": "yes"}, "thermaltest", None, INV) != [],
       "r4 1: garbage markers / a non-boolean thermal never raise and count as real")
    ok(kd.real_pauses([], {"thermal": False}, "main", 0, INV) == [], "r4 1: a clean receipt is no pause")
    ok(kd.real_pauses([{"file": "f", "inv8": INV[:8], "induced": True}], dict(TT), "thermaltest", 86, INV) == [],
       "r4 1: the genuine induced trigger (receipt copy + this invocation's file) is still the test")
    r, s1 = post(dict(TT, pause_marker={"test": True, "induced": True, "reason": "something else",
                                        "running_before": {"job": 1, "simulator": 1}}), pipe="thermaltest")
    ok(r["action"] == "stop" and s1["terminal"]["window_ended"], "r4 1: induced flag with a non-trigger reason = real")
    # main(): a malformed thermal receipt on disk -> the closure is persisted in the session file
    shutil.rmtree(os.path.join(tmp, "thermal"), ignore_errors=True)
    for f_ in os.listdir(tmp):
        if f_.startswith("session_2026"):
            os.remove(os.path.join(tmp, f_))
    json.dump(dict(RT, new_episodes=[{}]), open(os.path.join(tmp, "last_batch.json"), "w"))
    kd.scene_dir = lambda s: tmp
    try:
        rc_m = kd.main(["post", "--scene", "FloorPlan15", "--pipeline", "main", "--deadline", "2026-10-14 10:30",
                        "--invocation", INV, "--wsl-rc", "86"])
        ss = json.load(open(os.path.join(tmp, "session_20261014.json")))
        ok(rc_m == 0 and ss["terminal"]["window_ended"] is True,
           "r4 1: main() with a malformed thermal receipt persists the window closure")
    finally:
        kd.scene_dir = _sd
        os.remove(os.path.join(tmp, "last_batch.json"))
    # ------------------------------------------------------------------ r4 2: report contents
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [1], "lines": []}) != [],
       "r4 2 (Astra repro): a PASS report without check lines is not evidence")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [1], "lines": LINES}) == [],
       "r4 2: a complete PASS report is evidence")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [1], "lines": LINES[1:]}) != [],
       "r4 2: a report missing a required check line is not evidence")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [1], "lines": LINES_F}) != [],
       "r4 2: a 'PASS' whose lines imply FAIL is not evidence")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "FAIL", "episodes": [1], "lines": LINES_F}) == [],
       "r4 2: a consistent FAIL report is valid evidence (of a FAIL)")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [], "lines": LINES}) != [],
       "r4 2: a report with no episodes is not evidence")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [1],
                                   "lines": LINES + [{"status": "PASS", "check": "layout", "detail": ""}]}) != [],
       "r4 2: a repeated required check line is not evidence")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "episodes_by_split": kd.plan_splits("FloorPlan15", [1]), "verdict": "PASS", "episodes": [1], "lines": LINES + [{"status": "OK"}]}) != [],
       "r4 2: a malformed line is not evidence")
    ok("k23_own.sh" in kd.SCRIPTS, "auth r4 3/4: the ownership library is pinned")
    # ------------------------------------------------------------------ r5 1: closure before ANY diagnostic
    TD3 = os.path.join(tmp, "thermal")
    shutil.rmtree(TD3, ignore_errors=True)
    os.makedirs(TD3)
    json.dump({"test": False, "reason": "heat"}, open(os.path.join(TD3, "pause_20261014-074000-aaaaaaaa.json"), "w"))
    for f_ in os.listdir(tmp):
        if f_.startswith("session_2026") or f_.startswith("window_ended_"):
            os.remove(os.path.join(tmp, f_))
    _pf = kd.pending_files
    kd.scene_dir = lambda s: tmp
    kd.pending_files = lambda s: (_ for _ in ()).throw(OSError("listing failed"))
    try:
        rc_m = kd.main(["post", "--scene", "FloorPlan15", "--pipeline", "main", "--deadline", "2026-10-14 10:30",
                        "--invocation", INV, "--wsl-rc", "86"])
        ss = json.load(open(os.path.join(tmp, "session_20261014.json")))
        ok(ss["terminal"]["window_ended"] is True and os.path.exists(kd.closure_path("FloorPlan15", "2026-10-14")),
           "r5 1 (Astra repro): missing receipt + real marker + a failing pending listing -> closure persisted")
        # a corrupt session file: the closure FILE is still written first
        open(os.path.join(tmp, "session_20261014.json"), "w").write("{broken")
        os.remove(kd.closure_path("FloorPlan15", "2026-10-14"))
        kd.main(["post", "--scene", "FloorPlan15", "--pipeline", "main", "--deadline", "2026-10-14 10:30",
                 "--invocation", INV, "--wsl-rc", "86"])
        ok(os.path.exists(kd.closure_path("FloorPlan15", "2026-10-14")),
           "r5 1: an unreadable session file cannot prevent the closure (closure file written first)")
        os.remove(os.path.join(tmp, "session_20261014.json"))
        _ca, _la = kd.check_auth, kd.load_auth
        kd.check_auth = lambda *a_, **k_: []
        kd.load_auth = lambda: {"date": "2026-10-14", "window": {"start": "00:00", "end": "23:59"}}
        try:
            import io as _io2
            import contextlib as _cl2
            _o = _io2.StringIO()
            with _cl2.redirect_stdout(_o):
                kd.main(["pre", "--scene", "FloorPlan15", "--pipeline", "main", "--deadline", "2026-10-14 23:59"])
            ok('"stop"' in _o.getvalue() and "closure file" in _o.getvalue(),
               "r5 1: `pre` refuses while the closure file exists (even with a fresh session)")
        finally:
            kd.check_auth, kd.load_auth = _ca, _la
    finally:
        kd.scene_dir, kd.pending_files = _sd, _pf
        shutil.rmtree(TD3, ignore_errors=True)
        for f_ in os.listdir(tmp):
            if f_.startswith("session_2026") or f_.startswith("window_ended_"):
                os.remove(os.path.join(tmp, f_))
    # ------------------------------------------------------------------ yield (A28 15.9 condition 2)
    r, s1 = post(dict(R0, capture_rc=24, qc_rc=-1, done_after=0, new_episodes=[], guard_ok=False))
    ok(r["action"] == "done" and r["why"].startswith("YIELD") and not s1["terminal"] and s1["idle_runs"] == 0,
       "15.9: a yield receipt (another job waits for the lock) = done, not terminal, not an idle run")
    ok(post(dict(R0, capture_rc=24, qc_rc=0))[0]["action"] == "stop", "15.9: a yield receipt with commits stops")
    # ------------------------------------------------------------------ r5 7: parallel basis = FloorPlan15 only
    par17 = dict(par, scenes=["FloorPlan15", "FloorPlan17"])
    ok(kd.check_auth(par17, "FloorPlan17", "main", now, heads + ["## " + PH], tmp) != [],
       "r5 7: the parallel basis never authorizes FloorPlan17")
    ok(kd.check_auth(par17, "FloorPlan15", "main", now, heads + ["## " + PH], tmp) != [],
       "r5 7: the parallel basis needs scenes exactly ['FloorPlan15']")
    # ------------------------------------------------------------------ K2-CAPTURE: a window across midnight
    mid = dict(copy.deepcopy(par), window={"start": "22:30", "end": "12:00", "end_date": "2026-10-15"})
    _hm = heads + ["## " + PH]
    _n1, _n2 = dt.datetime(2026, 10, 14, 23, 10), dt.datetime(2026, 10, 15, 11, 50)
    ok(kd.check_auth(dict(mid, date="2026-10-14"), "FloorPlan15", "main", _n1, _hm, tmp) ==
       kd.check_auth(dict(par, window={"start": "00:00", "end": "23:59"}, date="2026-10-14"), "FloorPlan15", "main",
                     _n1, _hm, tmp), "midnight: before midnight the crossing window authorizes like a same-day window")
    _b2 = kd.check_auth(dict(mid, date="2026-10-14"), "FloorPlan15", "main", _n2, _hm, tmp)
    ok(not any("today is" in x or "outside the signed window" in x for x in _b2),
       f"midnight: after midnight (11:50 on the end date) the window still holds ({_b2[:2]})")
    _b3 = kd.check_auth(dict(mid, date="2026-10-14"), "FloorPlan15", "main", dt.datetime(2026, 10, 15, 12, 0), _hm, tmp)
    ok(any("outside the signed window" in x for x in _b3), "midnight: at the end time the window is closed")
    _b4 = kd.check_auth(dict(mid, date="2026-10-14"), "FloorPlan15", "main", dt.datetime(2026, 10, 16, 1, 0), _hm, tmp)
    ok(any("today is" in x for x in _b4), "midnight: a later day is never inside")
    for _w in ({"start": "22:30", "end": "23:00", "end_date": "2026-10-16"}, {"start": "22:30", "end": "22:00"},
               {"start": "22:30", "end": "22:40", "end_date": "2026-10-13"}):
        ok(any("window unreadable" in x for x in kd.check_auth(dict(par, window=_w, date="2026-10-14"), "FloorPlan15",
                                                               "main", _n1, _hm, tmp)),
           f"midnight: an invalid window {_w} (longer than 24 h / ends before it starts) is refused")
    _td9 = tempfile.mkdtemp(prefix="k23mk_")
    try:
        for _f in ("pause_20261014-235500-abcd1234.json", "pause_20261013-235500-abcd1234.json"):
            json.dump({"reason": "GPU temperature"}, open(os.path.join(_td9, _f), "w"))
        _pm = kd.pause_markers("FloorPlan15", "2026-10-15", _td9)
        ok([x["file"] for x in _pm] == ["pause_20261014-235500-abcd1234.json"],
           "midnight: a real pause stamped before midnight still counts for a window whose deadline is the next day")
    finally:
        shutil.rmtree(_td9, ignore_errors=True)
    # ------------------------------------------------------------------ r5 2: provenance entries
    import kitchen_data as _kq
    _MQ = _kq.Quota({u: {} for u in _kq.Quota.UNITS}).to_dict()
    _UQ = _kq.Quota({"type": {"train": {"Apple": 5}}, "inst": {}, "state": {}, "combo": {}}).to_dict()
    QD = os.path.join(tmp, "qc_r7")
    os.makedirs(QD, exist_ok=True)
    GE = kd.write_skip_evidence(QD, "FloorPlan15", 12, _MQ, 1)
    LOGT = "[07:41:02] " + GE["log_line"] + "\n"
    ok(kd.skip_entry_problems("12", GE, "FloorPlan15", LOGT, qdir=QD, batches=1) == [],
       "r5 2 + r7: a complete provenance entry with its retained evidence is evidence")
    for bad_, name_ in ((None, "null"), ({}, "empty"), (dict(GE, episode=13), "wrong episode"),
                        (dict(GE, log_line="x"), "wrong line"), (dict(GE, quota_sha256=None), "no quota hash"),
                        (dict(GE, first_seen_batch=0), "no first batch"),
                        (dict(GE, quota_sha256="b" * 64), "another quota hash"),
                        (dict(GE, evidence_sha256="0" * 64), "an evidence hash mismatch"),
                        (dict(GE, evidence="../x.json"), "a foreign evidence path"),
                        (dict(GE, first_seen_batch=2), "a batch the evidence does not name")):
        ok(kd.skip_entry_problems("12", bad_, "FloorPlan15", LOGT, qdir=QD, batches=1) != [],
           f"r5 2 / r7: provenance {name_} is not evidence")
    ok(kd.skip_entry_problems("12", GE, "FloorPlan15", "", qdir=QD) != [],
       "r5 2: provenance whose log line is gone is not evidence")
    ok(kd.skip_entry_problems("12", GE, "FloorPlan17", LOGT, qdir=QD) != [],
       "r5 2: provenance of another scene is not evidence")
    ok(kd.skip_entry_problems("12", GE, "FloorPlan15", LOGT) != [], "r7: no evidence folder = not evidence (fail closed)")
    ok(kd.skip_entry_problems("12", GE, "FloorPlan15", LOGT, qdir=QD, batches=-1) != [],
       "r7: first_seen_batch after batches + 1 is not evidence")
    _GU = kd.write_skip_evidence(QD, "FloorPlan15", 14, _UQ, 1)
    ok(any("train target" in x for x in kd.skip_entry_problems("14", _GU, "FloorPlan15", "[x] " + _GU["log_line"],
                                                                 qdir=QD, batches=1)),
       "r7: retained evidence whose quota is NOT met proves no skip")
    _G2 = kd.write_skip_evidence(QD, "FloorPlan15", 12, _UQ, 7)
    ok(_G2["first_seen_batch"] == 1 and _G2["evidence_sha256"] == GE["evidence_sha256"],
       "r7: retained evidence is written once, never replaced")
    _tr1 = next(e for e in range(400) if kd.plan_splits("FloorPlan15", [e]) == {"train": 1})
    ok(kd.report_content_problems({"scene": "FloorPlan15", "verdict": "PASS", "episodes": [_tr1], "episodes_by_split": {"train": 1},
                                   "lines": [x for x in LINES if x["check"] != "blur"]}) != [],
       "r5 2: a batch with TRAIN episodes must carry the TRAIN quality lines")
    ok(kd.report_content_problems({"scene": "FloorPlan15", "verdict": "PASS", "episodes": [1], "lines": LINES}) != [],
       "r5 2: a report without episodes_by_split is not evidence")
    ok(any("registered plan" in x for x in kd.report_content_problems(
        {"scene": "FloorPlan15", "verdict": "PASS", "episodes": [_tr1], "episodes_by_split": {}, "lines": LINES})),
       "r6 2 (Astra repro): an empty split map for a TRAIN episode is not evidence")
    ok("k23_procs.py" in kd.SCRIPTS and "k1_priority_k23.py" in kd.SCRIPTS,
       "auth r5: the process helper and the priority check are pinned")

    # ------------------------------------------------------------------ r3 2: write_receipt quota-skip evidence
    import argparse as _ap
    import kitchen_data as _kdat
    _plan = _kdat.episode_plan("FloorPlan15", _kdat.CAPTURE["episodes"])
    _tr = [e["episode"] for e in _plan if e["split"] == "train"][:3]
    _va = next(e["episode"] for e in _plan if e["split"] == "val")
    SR = os.path.join(tmp, "store")
    SDS = os.path.join(SR, "FloorPlan15")
    os.makedirs(SDS, exist_ok=True)
    met_q = _kdat.Quota({u: {} for u in _kdat.Quota.UNITS}).to_dict()
    unmet_q = _kdat.Quota({"type": {"train": {"Apple": 5}}, "inst": {}, "state": {}, "combo": {}}).to_dict()

    def receipt_case(done_new, quota, logline, recs, prov=None):
        json.dump({"done": [], "failed": {}}, open(os.path.join(tmp, "snap.json"), "w"))
        json.dump({"done_episodes": done_new, "failed_episodes": {}, "quota": quota},
                  open(os.path.join(SDS, "capture_state.json"), "w"))
        with open(os.path.join(SDS, "frames.jsonl"), "w") as fh:
            for e_ in recs:
                fh.write(json.dumps({"episode_record": {"episode": e_, "split": "train", "objects": {}}}) + "\n")
        open(os.path.join(tmp, "cap.log"), "w").write(logline)
        os.makedirs(os.path.join(tmp, "qc"), exist_ok=True)
        for f_ in os.listdir(os.path.join(tmp, "qc")):
            if f_.startswith("skip_evidence_"):
                os.remove(os.path.join(tmp, "qc", f_))
        json.dump({"covered": [], "batches": 0, "skips": {str(e_): kd.write_skip_evidence(
                       os.path.join(tmp, "qc"), "FloorPlan15", e_, quota, 1) for e_ in (prov or [])}},
                  open(os.path.join(tmp, "qc", "qc_state.json"), "w"))
        ns = _ap.Namespace(snapshot=os.path.join(tmp, "snap.json"), scene="FloorPlan15", invocation=INV, pipeline="main",
                           stamp="x", capture_rc="0", qc_rc="0", guard_ok="1", guard_rc="0", guard_killed="0",
                           phase="normal", marker="", post_stop="", log=os.path.join(tmp, "cap.log"), log_offset="0")
        _sr, _sdd = kd.store_root, kd.scene_dir
        kd.store_root, kd.scene_dir = (lambda pipeline="main": SR), (lambda s: tmp)
        try:
            return kd.write_receipt(ns)
        finally:
            kd.store_root, kd.scene_dir = _sr, _sdd
    skl = f"[07:41:02] FloorPlan15: ep {_tr[2]:3d} [train] skipped: every train target already met\n"
    rr = receipt_case(_tr, met_q, skl, _tr[:2], prov=[_tr[2]])
    ok(rr["new_skipped"] == [_tr[2]] and rr["skipped_unexplained"] == [],
       "receipt: logged quota skip + stored quota met + the QC's skip provenance = a demonstrated skip")
    rr = receipt_case(_tr, met_q, skl, _tr[:2])
    ok(rr["skipped_unexplained"] == [_tr[2]],
       "receipt r4 2: logged skip + quota met but NO skip provenance recorded by the QC -> unexplained")
    rr = receipt_case(_tr, unmet_q, skl, _tr[:2])
    ok(rr["skipped_unexplained"] == [_tr[2]], "receipt: logged skip but the stored quota is NOT met -> unexplained (r3 2)")
    rr = receipt_case(_tr, met_q, "", _tr[:2])
    ok(rr["skipped_unexplained"] == [_tr[2]], "receipt: quota met but no log line -> unexplained")
    rr = receipt_case(_tr[:2] + [_va], met_q, "", _tr[:2])
    ok(rr["skipped_not_train"] == [_va], "receipt: a held-out episode without a record is listed as not TRAIN")
    rr = receipt_case(_tr[:2], met_q, "", _tr[:2])
    ok(rr["new_skipped"] == [] and rr["guard_ok"] and not rr["guard_killed"], "receipt: all recorded = nothing skipped")
    ok(kd.validate_receipt(dict(R0), INV, "FloorPlan15", "main") == [], "receipt validation unchanged for a clean one")
    shutil.rmtree(QCD, ignore_errors=True)
    rp_ = os.path.join(tmp, "res.json")
    json.dump({"who": "Arash", "when": "07:50", "finding": "x", "decision": "resume",
               "astra_file": "docs/final-test/_astra/K23-PREP.txt"}, open(rp_, "w"))
    ok(kd.valid_resolution(rp_) == [], "resolution: a complete one is valid")
    json.dump({"who": "Arash", "when": "07:50", "finding": "x", "decision": "stop"}, open(rp_, "w"))
    ok(kd.valid_resolution(rp_) != [], "resolution: decision stop / no Astra file is not a release")
    open(rp_, "w").write("{}")
    ok(kd.valid_resolution(rp_) != [], "resolution: an empty file is not a release")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------ capture wrapper refusals (no simulator)

def isolated_capture_main(args):
    """K2-TOPUP incident 2026-10-17 (Astra K2-TOPUP-INCIDENT-r1): a capture-wrapper test never touches a real store or
    simulator -- temporary roots and a fake build_kitchen_data whose every capture entry point fails at once."""
    import tempfile, types
    saved_roots, saved_mod = k23_capture.roots, sys.modules.get("build_kitchen_data")
    t = tempfile.mkdtemp(prefix="k23iso_")
    fake = types.ModuleType("build_kitchen_data")
    fake.DEFAULT_ROOT = os.path.join(t, "kv3")

    def boom(*_a, **_k):
        raise AssertionError("SELFTEST ISOLATION: a real capture / simulator call was attempted")
    fake.stage_capture = fake.stage_probe = fake.make_agent = boom
    fake.CaptureAborted = type("CaptureAborted", (Exception,), {})
    sys.modules["build_kitchen_data"] = fake
    k23_capture.roots = lambda: (os.path.join(t, "kv3"), os.path.join(t, "amb"))
    for sc in ("FloorPlan15", "FloorPlan17"):                  # an empty store with a probe: nothing to capture from
        os.makedirs(os.path.join(t, "kv3", sc))
        json.dump({"scene": sc}, open(os.path.join(t, "kv3", sc, "probe.json"), "w"))
    saved_fns = (k23_capture.check_classes, k23_capture.install_scoped_restart, k23_capture.install_hang_skip,
                 k23_capture.install_dup_ruling)
    k23_capture.check_classes = lambda *_a: True
    k23_capture.install_scoped_restart = k23_capture.install_hang_skip = k23_capture.install_dup_ruling = lambda *_a, **_k: None
    try:
        return k23_capture.main(args)
    except AssertionError as exc:
        print(exc)
        return "ISOLATION-VIOLATION"
    finally:
        k23_capture.roots = saved_roots
        (k23_capture.check_classes, k23_capture.install_scoped_restart, k23_capture.install_hang_skip,
         k23_capture.install_dup_ruling) = saved_fns
        if saved_mod is None:
            sys.modules.pop("build_kitchen_data", None)
        else:
            sys.modules["build_kitchen_data"] = saved_mod
        shutil.rmtree(t, ignore_errors=True)

# topup / close are enabled since K2-TOPUP: their acceptance is tested on a fake store in _k23_topup_selftest.py;
# here, on an isolated empty store, topup must refuse (main plan not complete) and amb stays refused
ok(isolated_capture_main(["--scene", "FloorPlan15", "--pipeline", "topup"]) == k23_capture.EXIT_REFUSED,
   "capture: topup refused before the main plan is complete (isolated)")
ok(isolated_capture_main(["--scene", "FloorPlan15", "--pipeline", "amb"]) == k23_capture.EXIT_REFUSED,
   "capture: amb refused in this release")
ok(kd.main(["pre", "--deadline", "not-a-date"]) == 2, "decide: an unreadable deadline is a STOP (rc 2)")

print(f"_k23_decide_selftest: {N['ok']}/{N['ok'] + N['fail']} {'OK' if not N['fail'] else 'FAILED'}")
sys.exit(1 if N["fail"] else 0)
