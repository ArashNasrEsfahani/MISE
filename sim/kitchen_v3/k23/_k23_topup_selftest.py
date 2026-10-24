"""Offline checks for the K2-TOPUP pipelines (Astra K2-TOPUP-r1 item 6): quota already met, batch-boundary stop,
cap exhaustion, resumed completion, close ordering, close-only receipt / QC acceptance. No simulator: build_kitchen_data
and kitchen_data are replaced by small fakes that keep the same capture_state.json contract.

    python sim/kitchen_v3/k23/_k23_topup_selftest.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
N = {"ok": 0, "bad": 0}


def check(cond, what):
    N["ok" if cond else "bad"] += 1
    if not cond:
        print("FAIL:", what)


SC = "FloorPlan15"
MAIN = [{"episode": i, "split": "train"} for i in range(4)]
TOP = [{"episode": 1000 + i, "split": "train", "protocol": "topup"} for i in range(4)]
CLOSE = [{"episode": 2000 + i, "split": "val" if i % 2 == 0 else "test", "protocol": "close"} for i in range(4)]
WORLD = {"met_after": None, "commits": 0}       # train targets met after this many committed episodes (None = never)


def fake_kd():
    kd = types.ModuleType("kitchen_data")
    kd.CAPTURE = {"episodes": 4}
    kd.TOPUP_TRAIN_MAX, kd.CLOSE_EPISODES = 4, 2
    kd.episode_plan = lambda s, n: list(MAIN)
    kd.topup_train_plan = lambda s, n=4: list(TOP)
    kd.close_plan = lambda s, n=2: list(CLOSE)
    kd.train_complete = lambda q: q["met"]
    kd.CAPTURE_AMB = {"episodes": {"train": 2}}
    kd.CLOSE_PROTOCOL = "heldout-close-v1"
    kd.accept_train = lambda *a_, **k_: "orig"
    kd.stable_seed = lambda *p: hash(p) & 0xFFFF

    class Q:
        def split_done(self, split):
            return "orig"
    kd.Quota = Q
    kd.amb_plan = lambda s: [{"episode": 3000 + i, "split": "train", "protocol": "amb"} for i in range(2)] +         [{"episode": 3300 + i, "split": "test", "protocol": "amb-v1-holdout"} for i in range(2)]
    return kd


def fake_bkd(root):
    b = types.ModuleType("build_kitchen_data")
    b.DEFAULT_ROOT = root

    class CaptureAborted(Exception):
        pass
    b.CaptureAborted = CaptureAborted

    def stage_capture(scene, r, cfg, plan_override=None, stop_when=None):
        sp = os.path.join(r, scene, "capture_state.json")
        st = json.load(open(sp)) if os.path.exists(sp) else {"done_episodes": [], "failed_episodes": {}}
        plan = plan_override if plan_override is not None else MAIN
        for e in plan:
            if e["episode"] in st["done_episodes"]:
                continue
            q = {"met": WORLD["met_after"] is not None and WORLD["commits"] >= WORLD["met_after"]}
            if stop_when is not None and stop_when(q):
                break
            st["done_episodes"].append(e["episode"])
            WORLD["commits"] += 1
            json.dump(st, open(sp, "w"))
        return {"ok": False, "n_short": 3}
    b.stage_capture = stage_capture
    b.stage_probe = lambda *a, **k: None
    b.choose_view = lambda *a, **k: (None, "stub")
    return b


def run(cap, pipeline, max_new=2):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cap.main(["--scene", SC, "--pipeline", pipeline, "--max-new", str(max_new)])
    return rc, out.getvalue()


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="k23topup_")
    try:
        root, amb = os.path.join(tmp, "kv3"), os.path.join(tmp, "amb")
        ds = os.path.join(root, SC)
        os.makedirs(ds)
        json.dump({"scene": SC}, open(os.path.join(ds, "probe.json"), "w"))
        sys.modules["kitchen_data"] = fake_kd()
        sys.modules["build_kitchen_data"] = fake_bkd(root)
        import k23_capture as cap
        cap.roots = lambda: (root, amb)
        cap.check_classes = lambda s, p: True
        cap.install_scoped_restart = lambda b, s: None
        cap.install_hang_skip = lambda b, s: None
        cap.install_dup_ruling = lambda *_a, **_k: None
        sp = os.path.join(ds, "capture_state.json")
        marker = os.path.join(ds, "topup_complete.json")
        st = lambda: json.load(open(sp))["done_episodes"]

        # 1. top-up / close refused before the main plan is complete
        json.dump({"done_episodes": [0, 1], "failed_episodes": {}}, open(sp, "w"))
        rc, o = run(cap, "topup")
        check(rc == cap.EXIT_REFUSED and "before the main capture is complete" in o, "topup refused before main done")
        rc, o = run(cap, "close")
        check(rc == cap.EXIT_REFUSED, "close refused before main done")
        check(st() == [0, 1], "nothing committed by a refused run")

        # 2. close refused before the top-up terminated (no marker)
        json.dump({"done_episodes": [0, 1, 2, 3], "failed_episodes": {}}, open(sp, "w"))
        rc, o = run(cap, "close")
        check(rc == cap.EXIT_REFUSED and "before the TRAIN top-up terminated" in o, "close refused without marker")

        # 3. batch boundary: targets never met, max_new 2 -> exactly 2 top-up episodes, not complete, no marker
        WORLD.update(met_after=None, commits=0)
        rc, o = run(cap, "topup", 2)
        check(rc == cap.EXIT_OK and st() == [0, 1, 2, 3, 1000, 1001], f"batch boundary commits 2 ({st()})")
        check("COMPLETE topup" not in o and not os.path.exists(marker), "not complete after the first batch")

        # 4. cap exhaustion: second batch uses the last 2 -> COMPLETE (plan exhausted, quotas NOT met), marker written
        rc, o = run(cap, "topup", 2)
        check("COMPLETE topup (plan exhausted; quotas NOT necessarily met)" in o, "cap exhaustion -> COMPLETE")
        check(os.path.exists(marker) and "plan exhausted" in json.load(open(marker))["why"], "marker on exhaustion")
        check(len([e for e in st() if e >= 1000]) == 4, "never more than the 24-cap plan (4 in the fake)")

        # 5. quota already met at the start: 0 new episodes, COMPLETE (every TRAIN target met)
        os.remove(marker)
        json.dump({"done_episodes": [0, 1, 2, 3], "failed_episodes": {}}, open(sp, "w"))
        WORLD.update(met_after=0, commits=0)
        rc, o = run(cap, "topup", 2)
        check(st() == [0, 1, 2, 3] and "COMPLETE topup (every TRAIN target met)" in o, "already met -> 0 new, COMPLETE")
        check(os.path.exists(marker), "marker when already met")

        # 6. resumed completion: met after 3 commits -> batch 1 commits 2 (not complete), batch 2 commits 1 then COMPLETE
        os.remove(marker)
        json.dump({"done_episodes": [0, 1, 2, 3], "failed_episodes": {}}, open(sp, "w"))
        WORLD.update(met_after=3, commits=0)
        rc, o = run(cap, "topup", 2)
        check("COMPLETE topup" not in o and st()[-2:] == [1000, 1001], "resume: first batch partial")
        WORLD["commits"] = 2                  # the fake world counts committed top-up episodes
        rc, o = run(cap, "topup", 2)
        check(st()[-1] == 1002 and "COMPLETE topup (every TRAIN target met)" in o, f"resume: completes ({st()})")

        # 7. close after the marker: batches of 2 over the 4 close episodes, COMPLETE close at the end
        WORLD.update(met_after=None, commits=0)
        rc, o = run(cap, "close", 2)
        check(rc == cap.EXIT_OK and [e for e in st() if e >= 2000] == [2000, 2001] and "COMPLETE close" not in o,
              "close batch 1")
        rc, o = run(cap, "close", 2)
        check([e for e in st() if e >= 2000] == [2000, 2001, 2002, 2003] and "COMPLETE close (plan exhausted" in o,
              "close completes")
        # 7a. parity: prerequisites, own episode range 4000+, TRAIN, patches restored after the run
        import k23_decide as d1
        _st_before_parity = open(sp).read()                # 7a must leave the fake store as it found it
        _q1 = d1.qc_unresolved
        d1.qc_unresolved = lambda *_a, **_k: []
        stx = json.load(open(sp))
        stx["done_episodes"] = sorted(set(stx["done_episodes"]) | {e["episode"] for e in TOP})
        json.dump(stx, open(sp, "w"))
        cap.parity_target_new_train = lambda ds_, k1_train=16299, retention=0.949: (100, 3)
        bk = sys.modules["build_kitchen_data"]
        orig_cv = getattr(bk, "choose_view", None)
        fkd = sys.modules["kitchen_data"]
        acc0, sd0 = fkd.accept_train if hasattr(fkd, "accept_train") else None, fkd.Quota.split_done
        rc, o = run(cap, "parity", 2)
        pe = [e for e in json.load(open(sp))["done_episodes"] if e >= 4000]
        check(rc == cap.EXIT_OK and pe == [4000, 4001], f"parity batch commits 4000-4001 ({pe}; {o[-160:]})")
        check(getattr(bk, "choose_view", None) is orig_cv and fkd.Quota.split_done is sd0
              and (acc0 is None or fkd.accept_train is acc0), "parity patches restored after the run")
        d1.qc_unresolved = lambda *_a, **_k: ["batch_009.json: WARN"]
        rc, o = run(cap, "parity", 2)
        check(rc == cap.EXIT_REFUSED and "parity before" in o, "parity refused while QC unresolved")
        d1.qc_unresolved = _q1
        open(sp, "w").write(_st_before_parity)
        # 7b. amb: own store, probe byte-identical, refused before close completes (checked above via close order)
        import k23_decide as d0                       # isolation: the scene's real QC evidence is never read here
        _qu = d0.qc_unresolved
        d0.qc_unresolved = lambda *_a, **_k: []
        rc, o = run(cap, "amb", 2)
        check(rc == cap.EXIT_REFUSED and "topup" in o, "amb refused: marker says met but no stored quota proves it")
        stx = json.load(open(sp))                     # now a really terminated top-up: its plan exhausted in the store
        stx["done_episodes"] = sorted(set(stx["done_episodes"]) | {e["episode"] for e in TOP})
        json.dump(stx, open(sp, "w"))
        rc, o = run(cap, "amb", 2)
        asp = os.path.join(amb, SC, "capture_state.json")
        check(rc == cap.EXIT_OK and json.load(open(asp))["done_episodes"] == [3000, 3001], f"amb batch 1 in its own store ({o[-200:]})")
        check(open(os.path.join(amb, SC, "probe.json"), "rb").read() == open(os.path.join(ds, "probe.json"), "rb").read(),
              "amb probe byte-identical")
        check(all(e < 3000 for e in st()), "main store untouched by amb")
        rc, o = run(cap, "amb", 2)
        check("COMPLETE amb (plan exhausted" in o and json.load(open(asp))["done_episodes"][-2:] == [3300, 3301],
              "amb completes incl. holdout episodes")
        open(os.path.join(amb, SC, "probe.json"), "w").write("{}")
        rc, o = run(cap, "amb", 2)
        check(rc == cap.EXIT_REFUSED and "byte-identical" in o, "a differing amb probe is refused")
        d0.qc_unresolved = lambda *_a, **_k: ["batch_009.json: WARN"]
        rc, o = run(cap, "amb", 2)
        check(rc == cap.EXIT_REFUSED and "QC resolved" in o, "amb refused while any QC report is unresolved")
        d0.qc_unresolved = _qu
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for m in ("kitchen_data", "build_kitchen_data", "k23_capture"):
            sys.modules.pop(m, None)

    # 8. decide side: real kitchen_data plans; close / topup receipts may commit episodes; splits known
    import k23_decide as d
    check(d.CAPTURING == ("main", "topup", "close", "amb", "parity") and set(d.CAPTURING) <= set(d.PIPELINES), "CAPTURING pipelines")
    sys.path.insert(0, d.SIM)
    import kitchen_data as kd
    cl = [e["episode"] for e in kd.close_plan(SC, kd.CLOSE_EPISODES)]
    tp = [e["episode"] for e in kd.topup_train_plan(SC, kd.TOPUP_TRAIN_MAX)]
    sp_cl = d.plan_splits(SC, cl)
    check(sp_cl == {"val": kd.CLOSE_EPISODES, "test": kd.CLOSE_EPISODES}, f"close splits {sp_cl}")
    check(d.plan_splits(SC, tp) == {"train": kd.TOPUP_TRAIN_MAX}, "top-up splits")
    base = json.load(open(os.path.join(HERE, "_fixtures", "receipt_37735a4d.json"), encoding="utf-8"))
    for pipe, eps in (("close", cl[:2]), ("topup", tp[:2])):
        rec = dict(base, pipeline=pipe, new_episodes=eps, new_skipped=[], skipped_not_train=[], skipped_unexplained=[])
        probs = [p for p in d.validate_receipt(rec, rec["invocation"], SC, pipe) if "must commit no episode" in p]
        check(not probs, f"{pipe} receipt may commit episodes ({probs})")
    rec = dict(base, pipeline="probe", new_episodes=cl[:1])
    check(any("must commit no episode" in p for p in d.validate_receipt(rec, rec["invocation"], SC, "probe")),
          "probe receipt still may not commit")
    # close with a non-PASS QC on new episodes stops like main
    sess = d.new_session()
    rec = dict(base, pipeline="close", capture_rc=0, qc_rc=3, new_episodes=cl[:2], done_before=168, done_after=170, guard_ok=True, thermal=False,
               guard_killed=False, failed_after=0, failed_episodes={}, new_skipped=[], skipped_not_train=[],
               skipped_unexplained=[], complete=False)
    try:
        r = d.decide_post(rec, rec["invocation"], SC, "close", 0, sess, 0.0, 10 ** 10, False, qc_bad=[], markers=[])
        check(r["action"] == "stop" and "QC rc 3" in r["why"], f"close + QC WARN -> stop ({r})")
    except TypeError as exc:                                   # signature drift: report, do not hide
        check(False, f"decide_post signature: {exc}")
    # 8b. session resolutions are bound to the stopped invocation (old numbered files never lift a new stop)
    rt = tempfile.mkdtemp(prefix="k23res_")
    _sd = d.scene_dir
    d.scene_dir = lambda s_: rt
    try:
        inv = "eed9129b2646420d8bf5b0e38fddebd8"
        good = {"who": "x", "when": "y", "finding": "z", "decision": "resume", "astra_file": "docs/final-test/_astra/K23-PREP.txt"}
        json.dump(good, open(os.path.join(rt, "session_resolution_1.json"), "w"))
        s1 = dict(d.new_session(), terminal={"why": "QC WARN", "invocation": inv, "window_ended": False})
        d.resolve_terminal(SC, s1)
        check(s1["terminal"] is not None, "an old numbered resolution does not lift a new stop")
        json.dump(dict(good, invocation="f" * 32), open(os.path.join(rt, f"session_resolution_{inv[:8]}.json"), "w"))
        d.resolve_terminal(SC, s1)
        check(s1["terminal"] is not None, "a resolution naming another invocation does not lift it")
        json.dump(dict(good, invocation=inv), open(os.path.join(rt, f"session_resolution_{inv[:8]}.json"), "w"))
        d.resolve_terminal(SC, s1)
        check(s1["terminal"] is None and len(s1["resolutions"]) == 1, "the bound resolution lifts it")
        s2 = dict(d.new_session(), terminal={"why": "THERMAL", "invocation": inv, "window_ended": True})
        d.resolve_terminal(SC, s2)
        check(s2["terminal"] is not None, "a thermal window end is never lifted")
    finally:
        d.scene_dir = _sd
        shutil.rmtree(rt, ignore_errors=True)
    # 8c. k23_approvals basis: both kitchens, named approvals required, window end <= 2026-10-25
    import datetime as dt
    auth = json.load(open(d.AUTH, encoding="utf-8"))
    heads = d.ledger_headings()
    now = dt.datetime.strptime(f"{auth['date']} {auth['window']['start']}", "%Y-%m-%d %H:%M") + dt.timedelta(minutes=5)
    if auth.get("precondition") == "k23_approvals":
        for sc in ("FloorPlan15", "FloorPlan17"):
            check(d.check_auth(auth, sc, "main", now, heads) == [], f"k23_approvals authorizes {sc}")
        bad = dict(auth, approval_entries=auth["approval_entries"][:1])
        check(any("approve all 6" in x for x in d.check_auth(bad, "FloorPlan17", "main", now, heads)), "both approvals needed")
        late = dict(auth, window=dict(auth["window"], end_date="2026-10-26"), date="2026-10-25")
        check(any("2026-10-25" in x for x in d.check_auth(late, "FloorPlan17", "main",
              dt.datetime(2026, 10, 25, 17, 0), heads)), "window end after 10-25 refused")
        check(d.check_auth(auth, "FloorPlan15", "amb", now, heads) == [] and "amb" in auth["pipelines"],
              "amb authorized since K2-AMB (2026-10-17)")
    # 8d. dup-ruling checkpoint: near_duplicates-only FAIL continues (NOTE); any other FAIL still aborts
    import k23_capture as kc
    fb = types.ModuleType("fake_bkd")

    class Abort(Exception):
        pass
    fb.CaptureAborted, fb.logs, fb.dumps = Abort, [], []
    fb._log = lambda s_, m: fb.logs.append(m)
    fb._dump = lambda p_, o: fb.dumps.append(p_)
    fb._checkpoint = None
    cc = types.ModuleType("check_capture")
    cc.render = lambda r: ""
    sys.modules["check_capture"] = cc
    try:
        kc.install_dup_ruling(fb, "FloorPlan15")
        cc.check = lambda ds, cfg: {"fail": True, "results": [{"status": "FAIL", "check": "near_duplicates", "detail": "5.7%"}]}
        fb._checkpoint("FloorPlan15", "/x", {"check_every": 8}, 200)
        check(len(fb.dumps) == 1 and "NOTE" in fb.logs[-1], "near_duplicates-only FAIL -> NOTE, checkpoint written")
        cc.check = lambda ds, cfg: {"fail": True, "results": [{"status": "FAIL", "check": "near_duplicates", "detail": "x"},
                                                              {"status": "FAIL", "check": "labels", "detail": "y"}]}
        try:
            fb._checkpoint("FloorPlan15", "/x", {"check_every": 8}, 208)
            check(False, "another FAIL must abort")
        except Abort as exc:
            check("labels" in str(exc), "another FAIL aborts as before")
        fb2 = types.ModuleType("fake_bkd2"); fb2._checkpoint = "orig"
        kc.install_dup_ruling(fb2, "FloorPlan1")
        check(fb2._checkpoint == "orig", "unruled scene keeps the frozen checkpoint")
    finally:
        sys.modules.pop("check_capture", None)
    # 9. K23-TOPUP-CC-1: only a cc_projection WARN, only in a topup/close run, becomes a NOTE
    import k23_qc as q
    lines = [("WARN", "cc_projection", "x"), ("FAIL", "cc_projection", "y"), ("WARN", "blur", "z"), ("PASS", "layout", "")]
    on, off = q.topup_cc_ruling(lines, True), q.topup_cc_ruling(lines, False)
    check(on[0][0] == "NOTE" and "K23-TOPUP-CC-1" in on[0][2] and "found WARN: x" in on[0][2], "cc WARN -> NOTE in topup")
    check(on[1][0] == "FAIL" and on[2][0] == "WARN" and on[3] == lines[3], "FAIL / other checks unchanged")
    check(off == lines, "main runs unchanged")
    # 10. K2-AMB QC: main-quota coverage checks -> NOTE in amb runs only; integrity / cross-split / pixels unchanged
    am = [("FAIL", "cc_projection", "a"), ("WARN", "class_balance", "b"), ("FAIL", "test_floors", "c"),
          ("WARN", "cc_combos", "d"), ("FAIL", "dup_cross_split", "e"), ("FAIL", "class_ids", "f"), ("WARN", "exposure", "g")]
    on = q.amb_ruling(am, True)
    check([x[0] for x in on] == ["NOTE", "NOTE", "NOTE", "NOTE", "FAIL", "FAIL", "WARN"], f"amb ruling scope ({on})")
    check(q.amb_ruling(am, False) == am, "amb ruling off in main runs")
    th = json.load(open(q.THRESHOLDS_AMB, encoding="utf-8"))
    check(th.get("amb") is True and all(3000 <= e < 3100 for e in th["source_episode_ids"])
          and len(th["windows"]) == 4, "K1 amb baseline: TRAIN amb episodes only, 4 full windows")
    # 11. K3-SINK-1 fail-closed scan: only the excluded OBJECT itself (4-part id) or a label of its type voids the exception
    import tempfile as _tf
    td = _tf.mkdtemp(prefix="k23sink_")
    try:
        rows = [{"img": "a", "labels": [{"type": "SinkBasin"}], "seg_colors": {"Sink|-00.01|+00.91|-00.62|SinkBasin": [1, 2, 3]}}]
        open(os.path.join(td, "frames.jsonl"), "w").write(chr(10).join(json.dumps(r) for r in rows) + chr(10))
        sc = q.structural_scan(td, "FloorPlan17")
        check(sc["ok"] and sc["types"] == ("Sink",), f"basin-only store keeps the exception ({sc})")
        rows.append({"img": "b", "labels": [], "seg_colors": {"Sink|-00.01|+00.91|-00.62": [4, 5, 6]}})
        open(os.path.join(td, "frames.jsonl"), "w").write(chr(10).join(json.dumps(r) for r in rows) + chr(10))
        check(not q.structural_scan(td, "FloorPlan17")["ok"], "a segmented Sink object voids the exception")
        check(not q.structural_scan(td, "FloorPlan15")["types"], "no exception outside FloorPlan17")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print(f"_k23_topup_selftest: {N['ok']}/{N['ok'] + N['bad']} {'OK' if not N['bad'] else 'FAILED'}")
    return 0 if not N["bad"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
