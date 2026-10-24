"""K2-AMB integration checks on a fake store (Astra K2-AMB-r2 item 6): amb receipt -> post validation against the qc_amb
folder, scene-wide QC stops across qc + qc_amb, checkpoint failure propagation (amb coverage NOTE vs integrity abort),
holdout duplicate identity. No simulator, no real store; temporary folders under this directory (same drive as REPO).

    python sim/kitchen_v3/k23/_k23_amb_selftest.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
_R = HERE
while not os.path.isdir(os.path.join(_R, "sim", "kitchen_v3", "k23")):
    _R = os.path.dirname(_R)
for _p in (os.path.join(_R, "sim"), os.path.join(_R, "sim", "kitchen_v3", "k23"), HERE):
    sys.path.insert(0, _p)                                # HERE first: the files under test
N = {"ok": 0, "bad": 0}
SC = "FloorPlan15"


def check(cond, what):
    N["ok" if cond else "bad"] += 1
    if not cond:
        print("FAIL:", what)


def main() -> int:
    import k23_decide as d
    import k23_qc as q
    import kitchen_data as kd
    tmp = os.path.join(HERE, "_amb_selftest_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    sdir, kv3 = os.path.join(tmp, "k23", SC), os.path.join(tmp, "kitchen_v3")
    os.makedirs(os.path.join(sdir, "qc_amb"))
    os.makedirs(os.path.join(sdir, "qc"))
    ads = os.path.join(kv3 + "_amb", SC)
    os.makedirs(ads)
    saved = (d.scene_dir, d.store_root)
    d.scene_dir = lambda s: os.path.join(tmp, "k23", s)
    d.store_root = lambda pipeline="main": kv3 + "_amb" if pipeline == "amb" else kv3
    try:
        amb = [e for e in kd.amb_plan(SC) if e["split"] == "train"][:8]
        eps = [e["episode"] for e in amb]
        json.dump({"done_episodes": eps, "failed_episodes": {}}, open(os.path.join(ads, "capture_state.json"), "w"))
        with open(os.path.join(ads, "frames.jsonl"), "w") as fh:
            for e in eps:
                fh.write(json.dumps({"episode_record": {"episode": e}}) + "\n")
        snap = os.path.join(tmp, "snap.json")
        json.dump({"done": [], "failed": {}}, open(snap, "w"))
        lines = [{"status": "PASS", "check": c, "detail": ""} for c in d.REQUIRED_CHECKS + d.TRAIN_CHECKS]
        rep = {"scene": SC, "batch": 1, "verdict": "PASS", "episodes": eps, "episodes_by_split": {"train": 8},
               "lines": lines, "amb": True}
        json.dump(rep, open(os.path.join(sdir, "qc_amb", "batch_001.json"), "w"))
        json.dump({"batches": 1, "covered": eps}, open(os.path.join(sdir, "qc_amb", "qc_state.json"), "w"))
        log = os.path.join(tmp, "capture.log")
        open(log, "w").write("[x] FloorPlan15: K23 amb after batch: {} \n")
        a = types.SimpleNamespace(snapshot=snap, scene=SC, pipeline="amb", invocation="a" * 32, stamp="s", capture_rc=0,
                                  qc_rc=0, guard_ok="1", guard_killed="0", guard_rc="0", phase="exit", marker="",
                                  post_stop="", log=log, log_offset="0")
        rec = d.write_receipt(a)
        check(rec["new_episodes"] == eps and rec["done_after"] == 8, f"amb receipt reads the amb store ({rec['new_episodes']})")
        check("qc_amb/batch_001.json" in rec.get("qc_report", "").replace("\\", "/"), f"receipt names the qc_amb report ({rec.get('qc_report')})")
        check(d.validate_receipt(rec, rec["invocation"], SC, "amb") == [], f"amb receipt valid ({d.validate_receipt(rec, rec['invocation'], SC, 'amb')})")
        check(d.qc_report_ok(rec, d.qc_dir_for(SC, "amb")) == [], f"amb QC report accepted in qc_amb ({d.qc_report_ok(rec, d.qc_dir_for(SC, 'amb'))})")
        check(d.qc_report_ok(rec, d.qc_dir_for(SC, "main")) != [], "the same report is NOT accepted as a main-folder report")
        sess = d.new_session()
        r = d.decide_post(rec, rec["invocation"], SC, "amb", 0, sess, 0.0, 10 ** 10, False, qc_bad=[],
                          qc_dir=d.qc_dir_for(SC, "amb"), markers=[])
        check(r["action"] == "continue", f"post: amb batch with QC PASS continues ({r})")
        # scene-wide stops: an unresolved amb WARN blocks every pipeline; an amb open batch is found
        json.dump(dict(rep, batch=2, verdict="WARN", lines=lines + [{"status": "WARN", "check": "exposure", "detail": "x"}]),
                  open(os.path.join(sdir, "qc_amb", "batch_002.json"), "w"))
        check(any(x.startswith("qc_amb/batch_002.json") for x in d.qc_unresolved(SC)), f"scene-wide unresolved incl. qc_amb ({d.qc_unresolved(SC)})")
        json.dump({"batches": 2, "covered": eps, "open": {"batch": 2, "verdict": "WARN"}},
                  open(os.path.join(sdir, "qc_amb", "qc_state.json"), "w"))
        op = d.qc_open(SC)
        check(op is not None and op.get("qc_dir") == "qc_amb", f"qc_open finds the amb open batch ({op})")
    finally:
        d.scene_dir, d.store_root = saved
        shutil.rmtree(tmp, ignore_errors=True)
    # checkpoint propagation: amb coverage FAIL -> NOTE; integrity FAIL aborts; main runs keep coverage FAILs fatal
    import k23_capture as kc
    cc = types.ModuleType("check_capture")
    cc.render = lambda r: ""
    sys.modules["check_capture"] = cc
    try:
        def fake(name):
            b = types.ModuleType(name)
            b.CaptureAborted = type("Abort", (Exception,), {})
            b._log = lambda *x: None
            b._dump = lambda *x: None
            b._checkpoint = None
            return b
        for amb_on, results, aborts, what in (
                (True, [("FAIL", "instances")], False, "amb: instances FAIL -> NOTE"),
                (True, [("FAIL", "combos"), ("FAIL", "near_duplicates")], False, "amb: coverage + dups -> NOTE"),
                (True, [("FAIL", "instances"), ("FAIL", "labels")], True, "amb: an integrity FAIL still aborts"),
                (False, [("FAIL", "instances")], True, "main: instances FAIL still aborts")):
            b = fake("fb")
            kc.install_dup_ruling(b, SC, amb=amb_on)
            cc.check = (lambda res: lambda ds, cfg: {"fail": True, "results": [{"status": s, "check": c, "detail": ""}
                                                                              for s, c in res]})(results)
            try:
                b._checkpoint(SC, "/x", {"check_every": 8}, 8)
                check(not aborts, what)
            except b.CaptureAborted:
                check(aborts, what)
    finally:
        sys.modules.pop("check_capture", None)
    # r3 1: a false 'every TRAIN target met' marker with an unfinished top-up and an unmet stored quota is refused
    t2 = os.path.join(HERE, "_amb_selftest_tmp2")
    shutil.rmtree(t2, ignore_errors=True)
    os.makedirs(t2)
    try:
        tp = kd.topup_train_plan(SC, kd.TOPUP_TRAIN_MAX)
        sp = os.path.join(t2, "capture_state.json")
        json.dump({"why": "every TRAIN target met"}, open(os.path.join(t2, "topup_complete.json"), "w"))
        fq = types.SimpleNamespace(train_complete=lambda qq: qq == "MET",
                                   Quota=types.SimpleNamespace(from_dict=lambda dd: dd.get("tag")))
        json.dump({"done_episodes": [tp[0]["episode"]], "quota": {"tag": "SHORT"}}, open(sp, "w"))
        check(kc.topup_terminated(t2, sp, tp, fq) is False, "false marker + unmet stored quota -> refused")
        json.dump({"done_episodes": [tp[0]["episode"]], "quota": {"tag": "MET"}}, open(sp, "w"))
        check(kc.topup_terminated(t2, sp, tp, fq) is True, "stored quota met -> terminated")
        json.dump({"done_episodes": [e["episode"] for e in tp], "quota": {"tag": "SHORT"}}, open(sp, "w"))
        check(kc.topup_terminated(t2, sp, tp, fq) is True, "plan exhausted -> terminated")
        open(sp, "w").write("{not json")
        check(kc.topup_terminated(t2, sp, tp, fq) is False, "malformed state -> not terminated (no exception)")
        os.remove(os.path.join(t2, "topup_complete.json"))
        check(kc.topup_terminated(t2, sp, tp, fq) is False, "no marker -> not terminated")
        # r3 3: layout evidence = the MAIN store's QC evidence in amb runs
        os.makedirs(os.path.join(t2, "qc"))
        os.makedirs(os.path.join(t2, "qc_amb"))
        json.dump({"skips": {"12": {"x": 1}}, "batches": 9}, open(os.path.join(t2, "qc", "qc_state.json"), "w"))
        prov, ldir, lb = q.layout_evidence(os.path.join(t2, "qc_amb"), {"skips": {}, "batches": 2}, True)
        check(prov == {"12": {"x": 1}} and lb == 9 and ldir.endswith("qc"), "amb run: layout gets the main store's evidence")
        prov, ldir, lb = q.layout_evidence(os.path.join(t2, "qc"), {"skips": {"5": 1}, "batches": 3}, False)
        check(prov == {"5": 1} and lb == 3, "main run: its own evidence")
    finally:
        shutil.rmtree(t2, ignore_errors=True)
    # holdout identity: episodes 3300-3399 are their own split for duplicate detection
    check(q.dup_identity({"episode": 3305, "split": "test"}) == "holdout" and
          q.dup_identity({"episode": 3205, "split": "test"}) == "test" and
          q.dup_identity({"episode": 2005, "split": "val"}) == "val", "holdout duplicate identity")
    check(q.amb_ruling([("FAIL", "dup_cross_split", "")], True)[0][0] == "FAIL", "cross-split duplicates stay FAIL in amb")
    print(f"_k23_amb_selftest: {N['ok']}/{N['ok'] + N['bad']} {'OK' if not N['bad'] else 'FAILED'}")
    return 0 if not N["bad"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
