"""Offline selftest of the K23 capture plumbing (no simulator, no GPU, no WSL needed except where noted).

    python sim/kitchen_v3/k23/_k23_selftest.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (SIM, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import k23_capture  # noqa: E402
import k23_classes  # noqa: E402
import k23_qc  # noqa: E402
import layout_fingerprint as lf  # noqa: E402
import thermal_guard as tg  # noqa: E402

N = {"ok": 0, "fail": 0}


def ok(cond, name):
    N["ok" if cond else "fail"] += 1
    if not cond:
        print(f"FAIL {name}")


# ---------------------------------------------------------------- thermal rule
def S(temp=60, sm=2000, util=50, t=0.0):
    return {"t": t, "temp": temp, "sm": sm, "util": util}


ok(tg.evaluate([S(88)] * 5, 100, 100, 0) is not None, "thermal: 5 x >= 87 C pauses")
ok(tg.evaluate([S(87)] * 5, 100, 100, 0) is not None, "thermal: exactly 87 C counts")
ok(tg.evaluate([S(88)] * 4, 100, 100, 0) is None, "thermal: 4 samples do not pause")
ok(tg.evaluate([S(88)] * 4 + [S(80)], 100, 100, 0) is None, "thermal: a cool sample breaks the run")
ok(tg.evaluate([S(88), S(80)] + [S(88)] * 4, 100, 100, 0) is None, "thermal: only the last 5 count")
ok(tg.evaluate([S(70, 900, 95)] * 5, 100, 100, 0) is not None, "thermal: SM < 1000 MHz at >= 90 % util pauses")
ok(tg.evaluate([S(70, 900, 50)] * 5, 100, 100, 0) is None, "thermal: low clock at low util is idle, no pause")
ok(tg.evaluate([S(70, 1000, 95)] * 5, 100, 100, 0) is None, "thermal: exactly 1000 MHz is not below")
ok(tg.evaluate([], 299, None, 0) is None, "thermal: 299 s without telemetry is fine")
ok(tg.evaluate([], 300, None, 0) is not None, "thermal: 300 s without telemetry pauses")
ok(tg.evaluate([S()] * 5, 1000, 600, 0) is not None, "thermal: telemetry lost after good samples pauses")
ok(tg.evaluate([S()] * 5, 100, 100, 0, test_trigger=True) is not None, "thermal: test trigger fires after 5")
ok(tg.evaluate([S()] * 4, 100, 100, 0, test_trigger=True) is None, "thermal: test trigger needs 5 samples")
ok(tg.evaluate([S(88)] * 5, 100, 100, 0, test_trigger=True).startswith("GPU temperature"),
   "thermal r3 1: a REAL heat condition wins over the test trigger")
ok(tg.evaluate([S()] * 5, 1000, 600, 0, test_trigger=True).startswith("no telemetry"),
   "thermal r3 1: lost telemetry during the pause test is a real pause")
ok(tg.is_induced(tg.TEST_REASON, True) and not tg.is_induced(tg.TEST_REASON, False)
   and not tg.is_induced("GPU temperature >= 87 C", True), "thermal r3 1: 'induced' only for the trigger itself")
ok(tg.parse_line("2026/10/14 07:31:00.000, 65, 2100, [N/A], 97", 1.0)["util"] == 97.0, "thermal: parse with N/A power")
ok(tg.parse_line("garbage", 1.0) is None, "thermal: unparsable line is no sample")
ok(tg.parse_line("2026/10/14 07:31:00.000, [N/A], 2100, 50, 97", 1.0) is None, "thermal: missing temperature = no telemetry")
ok(tg.parse_line("2026/10/14 07:31:00.000, 70, 2100, 50, [N/A]", 1.0) is None, "thermal: missing utilization = no telemetry")

# ---------------------------------------------------------------- layout fingerprint
def er(ep, split, objs, proto="main"):
    return {"episode": ep, "split": split, "protocol": proto, "spawn": {"ok": True},
            "objects": {n: {"oid": f"{n.split('_')[0]}|{x}", "target": {"sliced": i % 2 == 0}}
                        for i, (n, x) in enumerate(objs.items())}}


A = {"Apple_1": "+01.00|+00.90|-01.00", "Mug_2": "-00.50|+00.90|+00.20"}
B = {"Apple_1": "+01.10|+00.90|-01.00", "Mug_2": "-00.50|+00.90|+00.20"}
fa, _ = lf.fingerprint(er(0, "train", A))
fa2, _ = lf.fingerprint({**er(1, "train", A), "settled_metadata": {"AppleSliced_x|1": "z"}})
fb, _ = lf.fingerprint(er(2, "val", B))
ok(fa == fa2, "layout: same layout, different state draw -> same fingerprint")
ok(fa != fb, "layout: a 0.1 m move -> different fingerprint")
def row(ep, split, objs, pipe="main", spawn=True):
    e = er(ep, split, objs, pipe)
    fp, n = lf.fingerprint(e)
    return dict(store="s", episode=ep, split=split, pipeline=pipe, spawn_ok=spawn, fp=fp, n_objects=n,
                pos=lf.positions(e))


EXP = ["Apple_1", "Mug_2"]
r = lf.check([row(0, "train", A), row(3, "val", B)], EXP)
ok(r["ok"], "layout: distinct held-out layout passes")
r = lf.check([row(0, "train", A), row(3, "val", B), row(3305, "test", A, "amb-v1-holdout", spawn=True)], EXP)
ok(not r["ok"] and r["collisions"][0]["pipeline"] == "amb-v1-holdout", "layout: holdout = train layout is a collision")
r = lf.check([row(0, "train", A), row(3, "val", B, spawn=False)], EXP)
ok(not r["ok"] and r["spawn_failed_heldout"], "layout: a failed held-out spawn fails the check (investigate)")
A4 = dict(A, Egg_3="+00.10|+00.90|+00.10", Cup_4="+00.20|+00.90|+00.10")
A3 = {k: v for k, v in A4.items() if k != "Cup_4"}
r = lf.check([row(0, "train", A4), row(3, "val", A3)], EXP + ["Egg_3", "Cup_4"])
ok(not r["ok"] and r["collisions"] and r["incomplete_records"], "layout: an incomplete record still collides on the shared names")
r = lf.check([row(0, "train", {"Apple_1": "+01.00|+00.90|-01.00"}), row(3, "val", {"Mug_2": "-00.50|+00.90|+00.20"})], EXP)
ok(not r["ok"] and any("share <" in p for p in r["problems"]), "layout: uncomparable records are a problem")
r = lf.check([row(0, "train", {"Apple_1": "garbage", "Mug_2": "-00.50|+00.90|+00.20"}), row(3, "val", B)], EXP)
ok(not r["ok"] and any("malformed" in p for p in r["problems"]), "layout: malformed position caught")
ok(lf.POS_RE.match("-00.47|+01.11|00.00") is not None, "layout: AI2-THOR zero without sign accepted")
r = lf.check([row(0, "train", dict(A, Stray_9="+00.00|+00.00|+00.00")), row(3, "val", B)], EXP)
ok(not r["ok"] and any("unexpected names" in p for p in r["problems"]), "layout: unexpected name caught")
r = lf.check([row(3, "val", B), row(4, "test", B)], EXP)
ok(not r["ok"] and r["heldout_heldout_duplicates"], "layout: two held-out episodes with one layout fail")
ok(lf.pipeline_of({"protocol": "topup"}) == "topup-train", "layout: pipeline names")
r = lf.check([row(0, "train", A), row(3, "val", {"Apple_1": "+02.00|+00.90|-01.00"})], EXP)
ok(not r["ok"] and r["incomplete_records"] and not r["collisions"],
   "layout: an incomplete record fails the check even without a collision (r2 8)")
# report(): a held-out episode 'done' without a record fails; a TRAIN one is the only allowed exception
import kitchen_data as _kdat  # noqa: E402
_plan = _kdat.episode_plan("FloorPlan15", _kdat.CAPTURE["episodes"])
_val = next(e["episode"] for e in _plan if e["split"] == "val")
_tr = [e["episode"] for e in _plan if e["split"] == "train"][:2]
_td = tempfile.mkdtemp(prefix="k23lf_")
try:
    _ds = os.path.join(_td, "kitchen_v3", "FloorPlan15")
    os.makedirs(_ds)
    json.dump({"reachable": {"Apple_1": 1, "Mug_2": 1}, "reset_ids": {"Apple_1": "Apple|x", "Mug_2": "Mug|x"}},
              open(os.path.join(_ds, "probe.json"), "w"))
    with open(os.path.join(_ds, "frames.jsonl"), "w") as fh:
        fh.write(json.dumps({"episode_record": er(_tr[0], "train", A, "main")}) + chr(10))
    _met = _kdat.Quota({u: {} for u in _kdat.Quota.UNITS}).to_dict()
    _unmet = _kdat.Quota({"type": {"train": {"Apple": 5}}, "inst": {}, "state": {}, "combo": {}}).to_dict()
    json.dump({"done_episodes": [_tr[0], _tr[1]], "quota": _met}, open(os.path.join(_ds, "capture_state.json"), "w"))
    import k23_decide as _kd0
    _qcd = os.path.join(_td, "qc")
    os.makedirs(_qcd)
    _ent = _kd0.write_skip_evidence(_qcd, "FloorPlan15", _tr[1], _met, 1)
    rep_ = lf.report("FloorPlan15", _td, provenance={str(_tr[1]): _ent}, qc_dir=_qcd, qc_batches=1)
    ok(rep_["ok"] and rep_["done_without_record"]["kitchen_v3"] == [_tr[1]],
       "layout report: a TRAIN episode done without a record + stored quota met + QC skip provenance is allowed")
    rep_ = lf.report("FloorPlan15", _td, provenance={str(_tr[1]): _ent})
    ok(not rep_["ok"], "layout report r7: provenance without its evidence folder is not proof (fail closed)")
    rep_ = lf.report("FloorPlan15", _td, provenance={str(_tr[1]): dict(_ent, evidence_sha256="0" * 64)}, qc_dir=_qcd)
    ok(not rep_["ok"], "layout report r7: provenance whose retained evidence hash differs is not proof")
    rep_ = lf.report("FloorPlan15", _td, provenance={str(_tr[1]): _ent}, qc_dir=_qcd)
    ok(not rep_["ok"], "layout report r8 (Astra r7 2): no QC batch count = the first-seen bound cannot be checked")
    _ent5 = _kd0.write_skip_evidence(os.path.join(_td, "qc5"), "FloorPlan15", _tr[1], _met, 5) if os.makedirs(
        os.path.join(_td, "qc5")) is None else None
    rep_ = lf.report("FloorPlan15", _td, provenance={str(_tr[1]): _ent5}, qc_dir=os.path.join(_td, "qc5"), qc_batches=2)
    ok(not rep_["ok"], "layout report r8 (Astra r7 2): first_seen_batch after the QC's batches + 1 is not proof")
    rep_ = lf.report("FloorPlan15", _td, provenance={str(_tr[1]): None})
    ok(not rep_["ok"], "layout report r5 2: a null provenance entry is not proof")
    rep_ = lf.report("FloorPlan15", _td, provenance={})
    ok(not rep_["ok"] and any("skip provenance" in p_ for p_ in rep_["problems"]),
       "layout report r4 2: quota met NOW but no skip provenance -> not proven (a vanished record is possible)")
    rep_ = lf.report("FloorPlan15", _td)
    ok(not rep_["ok"], "layout report r4 2: no provenance given at all (stand-alone CLI) -> fail closed")
    json.dump({"done_episodes": [_tr[0], _tr[1]], "quota": _unmet}, open(os.path.join(_ds, "capture_state.json"), "w"))
    rep_ = lf.report("FloorPlan15", _td)
    ok(not rep_["ok"] and any("demonstrated quota skip" in p_ for p_ in rep_["problems"]),
       "layout report r3 2: a TRAIN episode without a record while the quota is NOT met fails")
    json.dump({"done_episodes": [_tr[0], _tr[1]]}, open(os.path.join(_ds, "capture_state.json"), "w"))
    rep_ = lf.report("FloorPlan15", _td)
    ok(not rep_["ok"], "layout report r3 2: a TRAIN episode without a record and no stored quota fails (any TRAIN is not enough)")
    ok(lf.quota_skip_unexplained({"quota": _met}, [999999], "FloorPlan15") == {999999: "not a planned episode"},
       "layout r3 2: an unplanned episode is never a quota skip")
    json.dump({"done_episodes": [_tr[0], _val]}, open(os.path.join(_ds, "capture_state.json"), "w"))
    rep_ = lf.report("FloorPlan15", _td)
    ok(not rep_["ok"] and any("not TRAIN" in p_ for p_ in rep_["problems"]),
       "layout report: a held-out episode done without a record fails (r2 8)")
finally:
    shutil.rmtree(_td, ignore_errors=True)

# ---------------------------------------------------------------- classes
exp = json.load(open(k23_classes.EXPECTED, encoding="utf-8"))
ok(exp["scenes"]["FloorPlan15"]["n_classes"] == 43 and exp["scenes"]["FloorPlan17"]["n_classes"] == 45,
   "classes: 43 / 45 expected")
ok(exp["scenes"]["FloorPlan15"]["n_shown"] == 42 and exp["scenes"]["FloorPlan17"]["n_shown"] == 44,
   "classes: shown = minus Knife")
ok(exp["method_check_FloorPlan1"]["identical"], "classes: inventory rule reproduces the Kitchen 1 probe registry")
ok(len(exp["scenes"]["FloorPlan15"]["amb_targets_present"]) == 10, "classes: all 10 specialist targets present")

# ---------------------------------------------------------------- capture wrapper (no simulator)
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


ok(isolated_capture_main(["--scene", "FloorPlan1", "--pipeline", "main"]) == k23_capture.EXIT_REFUSED,
   "capture: FloorPlan1 refused")
ok(isolated_capture_main(["--scene", "FloorPlan6", "--pipeline", "probe"]) == k23_capture.EXIT_REFUSED,
   "capture: other kitchens refused")
tmp = tempfile.mkdtemp(prefix="k23st_")
try:
    sp = os.path.join(tmp, "capture_state.json")
    json.dump({"done_episodes": [0, 1], "failed_episodes": {}}, open(sp, "w"))
    stop = k23_capture.batch_stop(sp, 2, "FloorPlan15", set(range(168)))
    ok(stop(None) is False, "capture: batch limit not reached at start")
    json.dump({"done_episodes": [0, 1, 2], "failed_episodes": {}}, open(sp, "w"))
    ok(stop(None) is False, "capture: 1 new < 2")
    json.dump({"done_episodes": [0, 1, 2], "failed_episodes": {"3": "hung twice"}}, open(sp, "w"))
    ok(stop(None) is True, "capture: a failed episode counts as finished")
    stop2 = k23_capture.batch_stop(sp, 2, "FloorPlan15", {1000, 1001})
    json.dump({"done_episodes": [0, 1, 2, 4, 5], "failed_episodes": {}}, open(sp, "w"))
    ok(stop2(None) is False, "capture: other plans' episodes do not count")
    ok(k23_capture.batch_stop(sp, 0, "x", {1})(None) is False, "capture: 0 = no limit")
    ok(k23_capture.plan_status(sp, [{"episode": e} for e in range(6)]) == {"planned": 6, "finished": 5,
                                                                           "remaining": 1}, "capture: plan status")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- pre-checkpoint cleanup (Astra K23-PREP r2 9)
import build_kitchen_data as bkd  # noqa: E402
tmp = tempfile.mkdtemp(prefix="k23cl_")
try:
    ds = os.path.join(tmp, "FloorPlan15")
    for sub in ("", "pairs"):
        for kind in ("images", "depth", "seg"):
            d = os.path.join(ds, sub, kind, "train")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "FloorPlan15_e000_0003" + ("_noise" if sub else "") + ".jpg"), "w").write("x")
    open(os.path.join(ds, "frames.jsonl"), "w").write("")
    n = k23_capture.clean_uncommitted(bkd, ds)
    ok(n == 6 and not k23_capture.episode_files(ds), "cleanup: files of an episode interrupted before the first commit removed")
    stray = os.path.join(ds, "images", "train", "FloorPlan15_partial.jpg")      # no episode number in its name
    open(stray, "w").write("x")
    try:
        k23_capture.clean_uncommitted(bkd, ds)
        ok(False, "cleanup: a surviving uncommitted file is an error, never ignored")
    except RuntimeError:
        ok(True, "cleanup: a surviving uncommitted file is an error, never ignored")
    os.remove(stray)
    json.dump({"done_episodes": []}, open(os.path.join(ds, "capture_state.json"), "w"))
    try:
        k23_capture.clean_uncommitted(bkd, ds)
        ok(False, "cleanup: refuses when a committed state exists")
    except RuntimeError:
        ok(True, "cleanup: refuses when a committed state exists")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- QC pure parts
flat = np.full((64, 64), 120.0, dtype=np.float32)
rng = np.random.default_rng(0)
noisy = rng.uniform(0, 255, (64, 64)).astype(np.float32)
ok(k23_qc.laplacian_var(flat) == 0.0 and k23_qc.laplacian_var(noisy) > 1000, "qc: Laplacian variance")
m, c = k23_qc.exposure(np.full((10, 10), 255.0))
ok(m == 255.0 and c == 1.0, "qc: exposure of a white frame")
seg = np.zeros((40, 40, 3), dtype=np.uint8)
seg[5:15, 10:30] = (1, 2, 3)
n, box = k23_qc.mask_stats(seg, (1, 2, 3))
ok(n == 200 and box == [10, 5, 29, 14], "qc: mask stats")
ok(k23_qc.mask_stats(seg, (9, 9, 9)) == (0, None), "qc: absent colour")
ok(abs(k23_qc.box_iou([0, 0, 9, 9], [0, 0, 9, 9]) - 1.0) < 1e-9 and k23_qc.box_iou([0, 0, 9, 9], [20, 20, 29, 29]) == 0,
   "qc: box IoU")
lab = {"type": "Apple", "orig_type": "Apple", "box": [10, 5, 29, 14], "px": 200}
ok(k23_qc.label_problems(lab, {"Apple"}) == [], "qc: clean label")
ok("type_not_in_registry" in k23_qc.label_problems(lab, {"Mug"}), "qc: type outside registry caught")
ok("box_outside_frame" in k23_qc.label_problems(dict(lab, box=[600, 5, 650, 14]), {"Apple"}), "qc: box outside")
ok("px_exceeds_box" in k23_qc.label_problems(dict(lab, px=10_000), {"Apple"}), "qc: px > box area")
ok("type_fold" in k23_qc.label_problems(dict(lab, orig_type="Mug"), {"Apple"}), "qc: type fold")


# ---------------------------------------------------------------- QC end to end on a synthetic store
def make_store(root):
    ds = os.path.join(root, "FloorPlan15")
    for k in ("images", "depth", "seg"):
        for s in ("train", "val"):
            os.makedirs(os.path.join(ds, k, s), exist_ok=True)
    reg = ["Apple", "Floor", "Mug"]
    json.dump({"scene": "FloorPlan15", "registry": reg}, open(os.path.join(ds, "probe.json"), "w"))
    lines = []
    r2 = np.random.default_rng(1)
    for ep, split in ((0, "train"), (1, "val")):
        lines.append(json.dumps({"episode_record": {"episode": ep, "split": split, "objects": {}}}))
        for f in range(8):
            name = f"FloorPlan15_e{ep:03d}_{f:04d}"
            img = r2.integers(0, 255, (640, 640, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(ds, "images", split, name + ".jpg"), quality=95)
            sg = np.zeros((640, 640, 3), dtype=np.uint8)
            sg[100:150, 200:260] = (5, 6, 7)
            Image.fromarray(sg).save(os.path.join(ds, "seg", split, name + ".png"))
            Image.fromarray(np.zeros((640, 640), dtype=np.uint16)).save(os.path.join(ds, "depth", split, name + ".png"))
            labels = [{"oid": "Apple|1", "type": "Apple", "orig_type": "Apple", "box": [200, 100, 259, 149],
                       "px": 3000, "state_key": "Apple|-"}]
            lines.append(json.dumps({"img": name, "split": split, "episode": ep, "kind": "free",
                                     "seg_colors": {"Apple|1": [5, 6, 7]}, "labels": labels}))
    open(os.path.join(ds, "frames.jsonl"), "w").write("\n".join(lines) + "\n")
    return ds, reg


tmp = tempfile.mkdtemp(prefix="k23qc_")
try:
    ds, reg = make_store(tmp)
    eps, frs = k23_qc.load_rows(ds)
    ok(len(eps) == 2 and len(frs) == 16, "qc e2e: streaming reader")
    lim = {"lapvar_low": 10.0, "lum_low": 20.0, "lum_high": 230.0, "clip_high": 0.5}
    m = k23_qc.measure(ds, eps, frs, reg, stride=4, limits=lim)
    ok(m["pixels"]["n"] == 2, "qc e2e: only TRAIN frames decoded (every 4th of 8 = 2)")
    ok(m["pixels"]["mask_checked"] == 2 and m["pixels"]["mask_mismatch"] == 0, "qc e2e: label = mask")
    ok(m["pairing"]["missing_files"] == 0 and m["pairing"]["orphans"] == 0, "qc e2e: pairing clean")
    ok(m["instances"]["train"]["Apple"] == 8 and m["instances"]["val"]["Apple"] == 8, "qc e2e: counts per split")
    # break things
    os.remove(os.path.join(ds, "seg", "train", "FloorPlan15_e000_0003.png"))
    shutil.copyfile(os.path.join(ds, "images", "train", "FloorPlan15_e000_0001.jpg"),
                    os.path.join(ds, "images", "val", "FloorPlan15_e001_0001.jpg"))
    Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8)).save(os.path.join(ds, "images", "train",
                                                                                "FloorPlan15_e000_0099.jpg"))
    rows = open(os.path.join(ds, "frames.jsonl")).read().splitlines()
    bad = json.loads(rows[5])
    bad["labels"][0]["px"] = 2000                                    # stored px disagrees with the mask
    rows[5] = json.dumps(bad)
    open(os.path.join(ds, "frames.jsonl"), "w").write("\n".join(rows) + "\n")
    eps, frs = k23_qc.load_rows(ds)
    m = k23_qc.measure(ds, eps, frs, reg, stride=1, limits=lim)
    ok(m["pairing"]["missing_files"] == 1, "qc e2e: missing seg file caught")
    ok(m["pairing"]["orphans"] == 1, "qc e2e: image without a row caught")
    ok(m["exact_dup"]["cross_split"] == 1, "qc e2e: cross-split duplicate caught by sha256")
    ok(m["pixels"]["mask_mismatch"] >= 1, "qc e2e: label / mask disagreement caught")
    th = {"bands": {k: {"min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0} for k in (
        "blur_low_share", "exposure_out_share", "near_dup_share", "mask_mismatch_share", "anomalies", "label_problems",
        "exact_dup_share", "cross_split_dup", "share_small", "mask_p50", "labels_per_frame", "held_share")},
          "k1_episodes_until_every_type_seen": 5}
    cum = k23_qc.counts_only(ds, lambda e: True)
    expd = {"classes": reg, "n_classes": 3}
    X = {"cross_batch_dups": {"n": 0, "cross_split": 0, "share": 0.0}, "integrity": [], "probe_scene": "FloorPlan15"}
    lines = k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, X)
    st = {n: s for s, n, _ in lines}
    ok(st["pairing"] == "FAIL" and st["dup_cross_split"] == "FAIL" and st["mask_vs_label"] == "FAIL",
       "qc verdict: broken store FAILS")
    ok(st["class_ids"] == "PASS", "qc verdict: class ids match")
    _m2 = json.loads(json.dumps(m)); _m2["pixels"].update(exposure_out_share=0.5, blur_low_share=0.5, extreme_frames=0)
    _l15 = {n: (s_, d_) for s_, n, d_ in k23_qc.verdict("FloorPlan15", _m2, cum, th, expd, reg, X)}
    _l17 = {n: s_ for s_, n, _ in k23_qc.verdict("FloorPlan17", _m2, cum, th, expd, reg, dict(X, probe_scene="FloorPlan17"))}
    ok(_l15["exposure"][0] == "NOTE" and "K23-AMD-2" in _l15["exposure"][1] and "would be FAIL" in _l15["exposure"][1],
       "qc K23-AMD-2: FloorPlan15 exposure share is a NOTE (the would-be status is reported)")
    ok(_l17["exposure"] == "NOTE", "qc K23-AMD-2: FloorPlan17 exposure share is a NOTE (Arash 2026-10-17 ~18:25)")
    ok(_l15["blur"][0] == "FAIL" and _l15["exposure_extreme"][0] == "PASS", "qc K23-AMD-2: blur unchanged on FloorPlan15")
    _m2["pixels"]["extreme_frames"] = 1
    ok({n: s_ for s_, n, _ in k23_qc.verdict("FloorPlan15", _m2, cum, th, expd, reg, X)}["exposure_extreme"] == "FAIL",
       "qc K23-AMD-2: extreme frames still FAIL on FloorPlan15")
    lines = k23_qc.verdict("FloorPlan15", m, cum, th, {"classes": reg[:2], "n_classes": 2}, reg,
                           dict(X, integrity=[("kitchen1_unchanged", False, "x")]))
    st = {n: s for s, n, _ in lines}
    ok(st["class_ids"] == "FAIL" and st["kitchen1_unchanged"] == "FAIL", "qc verdict: id map / integrity FAIL")
    ok(st["classes_seen"] in ("NOTE", "PASS"), "qc verdict: unseen classes only a NOTE before 16 episodes")
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, list(reversed(reg)), X)}
    ok(st["class_ids"] == "FAIL", "qc verdict: a permuted registry (class-ID order) FAILS")
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(X, probe_scene="FloorPlan1"))}
    ok(st["class_ids"] == "FAIL", "qc verdict: a probe of another scene FAILS")
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(X, failed_episodes={"5": "hung twice"}))}
    ok(st["failed_episodes"] == "FAIL", "qc verdict: a failed episode is surfaced as FAIL")
    PW = {"check": "pace", "status": "WARN", "detail": "episodes keeping < 20% of views: 4 (longest streak 3)"}
    lns = k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(X, cc_warn=[PW], cc_pace=PW))
    st = {n: s for s, n, _ in lns}
    ok(st.get("cc_pace") == "NOTE" and sum(1 for _, n, _ in lns if n == "cc_pace") == 1,
       "qc R1: check_capture pace WARN is ONE NOTE line (does not stop)")
    ok(any(n == "cc_pace" and "longest streak 3" in d for _, n, d in lns), "qc R1: the pace counts + streak are reported")
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg,
                                            dict(X, cc_pace={"check": "pace", "status": "PASS", "detail": "0"}))}
    ok(st.get("cc_pace") == "NOTE", "qc R1 (r5 2): a pace PASS is still ONE NOTE line (pace is always a NOTE)")
    for ck in ("progress", "projection", "instances", "combos"):
        W = {"check": ck, "status": "WARN", "detail": "x"}
        st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(X, cc_warn=[W], cc_pace=PW))}
        st2 = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg,
                                                 dict(X, cc_warn=[W], cc_pace=PW, main_complete=True))}
        ok(st.get("cc_" + ck) == "NOTE" and st2.get("cc_" + ck) == "WARN",
           f"qc r5 2: check_capture {ck} WARN = NOTE while partial (Astra r2 7), a STOP at the end of the main capture")
    for ck in ("pairs", "duplicates", "states", "balance", "files", "x_unknown"):
        W = {"check": ck, "status": "WARN", "detail": "x"}
        st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(X, cc_warn=[W], cc_pace=PW))}
        ok(st.get("cc_" + ck) == "WARN", f"qc R1 is narrow: a check_capture {ck} WARN still stops")
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, X)}
    ok(st.get("cc_pace") == "FAIL", "qc R1: no pace result from check_capture is a FAIL, never silently missing")
    PR = {"check": "pairs", "status": "WARN", "detail": "x"}
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(X, cc_warn=[PR, PW], cc_pace=PW))}
    ok(st.get("cc_pairs") == "WARN" and st.get("cc_pace") == "NOTE",
       "qc R1 is narrow: every other check_capture WARN (pairs) still stops")
    ok(k23_qc.R1_NOTE_CHECKS == ("pace",), "qc R1: only `pace` is reclassified")
    # cc_pairs denominator (Arash 2026-10-16): K23-AMD-1 Blinds pairs excluded on FloorPlan15 only, both values reported
    _PA = {"total": 272, "valid": 226, "amd1_blinds": 34, "amd1_blinds_valid": 0}
    _vl = {n: (s, d) for s, n, d in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg,
                                                   dict(X, cc_warn=[PR, PW], cc_pace=PW, cc_pairs_amd1=_PA))}
    ok(_vl["cc_pairs"][0] == "NOTE" and "226/238" in _vl["cc_pairs"][1] and "226/272" in _vl["cc_pairs"][1],
       "qc cc_pairs: FloorPlan15 226/238 = .950 with AMD-1 Blinds excluded = NOTE, both values reported")
    _vl = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(
        X, cc_warn=[PR, PW], cc_pace=PW, cc_pairs_amd1=dict(_PA, valid=190)))}
    ok(_vl["cc_pairs"] == "WARN", "qc cc_pairs: below .85 even after the exclusion (190-0)/238 still stops")
    _vl = {n: s for s, n, _ in k23_qc.verdict("FloorPlan17", m, cum, th, expd, reg,
                                            dict(X, cc_warn=[PR, PW], cc_pace=PW, cc_pairs_amd1=_PA))}
    ok(_vl["cc_pairs"] == "WARN", "qc cc_pairs: FloorPlan17 unchanged (registered denominator)")
    _vl = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg, dict(
        X, cc_warn=[{"check": "duplicates", "status": "WARN", "detail": "x"}, PW], cc_pace=PW, cc_pairs_amd1=_PA))}
    ok(_vl["cc_duplicates"] == "WARN", "qc cc_pairs: the exclusion does not touch any other check_capture WARN")
    _td2 = tempfile.mkdtemp(prefix="k23pa_")
    with open(os.path.join(_td2, "frames.jsonl"), "w", encoding="utf-8") as _fh:
        for _p in ({"oid": "Blinds|1", "action": "OpenObject", "valid_evidence": False},
                   {"oid": "Blinds|1", "action": "ToggleObjectOn", "valid_evidence": False},
                   {"oid": "Cabinet|1", "action": "OpenObject", "valid_evidence": True}, None):
            _fh.write(json.dumps({"pair": _p} if _p else {}) + "\n")
    ok(k23_qc.amd1_pairs(_td2) == {"total": 3, "valid": 1, "amd1_blinds": 1, "amd1_blinds_valid": 0},
       "qc cc_pairs: only Blinds Open/Close pairs are AMD-1 pairs")
    st = {n: s for s, n, _ in k23_qc.verdict("FloorPlan15", m, cum, th, expd, reg,
                                            dict(X, main_complete=True, instances_expected=["Apple_1"]))}
    ok(st["val_floors"] == "FAIL", "qc verdict: at completion a held-out type floor shortfall FAILS")
    # rule_upper
    out = []
    k23_qc.rule_upper(0.03, {"p50": .01, "max": .02}, "x", out)
    k23_qc.rule_upper(0.08, {"p50": .01, "max": .02}, "y", out)
    k23_qc.rule_upper(0.01, {"p50": .01, "max": .02}, "z", out)
    ok([o[0] for o in out] == ["WARN", "FAIL", "PASS"], "qc: WARN above K1 worst, FAIL above max(2x, +.05)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- r3 2: QC evidence before 'nothing new'
tmp = tempfile.mkdtemp(prefix="k23ev_")
try:
    qd = os.path.join(tmp, "qc")
    os.makedirs(qd)

    import k23_decide as _kdec
    GOODL = [{"status": "PASS", "check": c, "detail": ""} for c in _kdec.REQUIRED_CHECKS + _kdec.TRAIN_CHECKS]

    def wrep(n, eps, verdict="PASS", scene="FloorPlan15", lines=None):
        p = os.path.join(qd, f"batch_{n:03d}.json")
        ln = lines if lines is not None else (GOODL + ([{"status": verdict, "check": "x", "detail": ""}]
                                                       if verdict != "PASS" else []))
        json.dump({"scene": scene, "batch": n, "verdict": verdict, "episodes": eps,
                   "episodes_by_split": _kdec.plan_splits("FloorPlan15", eps),
                   "lines": ln}, open(p, "w"))
        return k23_qc.sha_file(p)
    h1 = wrep(1, [0, 1])
    QS = {"covered": [0, 1], "batches": 1, "hashes": {}, "reports": {"1": h1}}
    ok(k23_qc.evidence_problems(qd, QS, "FloorPlan15") == [], "qc evidence: a matching report reconciles")
    ok(any("no report checked" in p for p in k23_qc.evidence_problems(qd, dict(QS, covered=[0, 1, 2]), "FloorPlan15")),
       "qc evidence r3 2 (Astra repro): 'covered' beyond the reports is refused")
    ok(any("no sha256" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={}), "FloorPlan15")),
       "qc evidence: a report without a recorded hash is refused")
    ok(any("missing" in p for p in k23_qc.evidence_problems(qd, dict(QS, batches=2), "FloorPlan15")),
       "qc evidence: a missing report is refused")
    wrep(1, [0, 1, 5])
    ok(any("changed" in p for p in k23_qc.evidence_problems(qd, QS, "FloorPlan15")), "qc evidence: a changed report is refused")
    h1 = wrep(1, [0, 1], "FAIL")
    ok(any("not resolved" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}), "FloorPlan15")),
       "qc evidence: an unresolved FAIL report is refused")
    ok(any("resolution" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}, resolved=[{"batch": 1}]),
                                                             "FloorPlan15")),
       "qc evidence r4 2: a resolution record without its bound file / hash resolves nothing")
    rf = os.path.join(qd, "resolution_batch_001.json")
    json.dump({"who": "Arash", "when": "08:10", "finding": "x", "decision": "resume",
               "astra_file": "docs/final-test/_astra/K23-PREP.txt"}, open(rf, "w"))
    RES = [{"batch": 1, "resolution": rf, "resolution_sha256": k23_qc.sha_file(rf)}]
    ok(k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}, resolved=RES), "FloorPlan15") == [],
       "qc evidence: a FAIL report with a bound, valid resolution reconciles")
    json.dump({"who": "Arash", "when": "08:10", "finding": "x", "decision": "stop"}, open(rf, "w"))
    ok(any("changed" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}, resolved=RES), "FloorPlan15")),
       "qc evidence r4 2: a resolution changed after it was accepted resolves nothing")
    RES2 = [{"batch": 1, "resolution": rf, "resolution_sha256": k23_qc.sha_file(rf)}]
    ok(any("not valid" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}, resolved=RES2),
                                                            "FloorPlan15")),
       "qc evidence r4 2: a resolution that is not a valid 'resume' resolves nothing")
    RES3 = [{"batch": 1, "resolution": os.path.join(qd, "resolution_batch_002.json"), "resolution_sha256": "0" * 64}]
    ok(any("resolution" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}, resolved=RES3),
                                                             "FloorPlan15")),
       "qc evidence r4 2: a resolution naming another batch's file resolves nothing")
    os.remove(rf)
    h1 = wrep(1, [0, 1], scene="FloorPlan17")
    ok(any("scene" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}), "FloorPlan15")),
       "qc evidence: a report of another scene is refused")
    h1 = wrep(1, [0, 1])
    wrep(7, [3])
    ok(any("does not know" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}), "FloorPlan15")),
       "qc evidence: an extra report file is refused")
    os.remove(os.path.join(qd, "batch_007.json"))
    ok(k23_qc.evidence_problems(qd, {"covered": "x", "batches": 1}, "FloorPlan15") != [], "qc evidence: unreadable state")
    h1 = wrep(1, [0, 1], lines=[])
    ok(any("contents invalid" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}), "FloorPlan15")),
       "qc evidence r4 2 (Astra repro): a PASS report with no check lines is refused")
    h1 = wrep(1, [0, 1], lines=GOODL[2:])
    ok(any("contents invalid" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}), "FloorPlan15")),
       "qc evidence r4 2: a PASS report missing required check lines is refused")
    h1 = wrep(1, [0, 1], lines=GOODL + [{"status": "FAIL", "check": "x", "detail": ""}])
    ok(any("contents invalid" in p for p in k23_qc.evidence_problems(qd, dict(QS, reports={"1": h1}), "FloorPlan15")),
       "qc evidence r4 2: a 'PASS' report whose lines imply FAIL is refused")
    h1 = wrep(1, [0, 1])
    # end to end: cmd_batch must NOT answer 'nothing new' (rc 4) from a cached 'covered' list (Astra reproduced it)
    root = os.path.join(tmp, "root")
    ds, reg = make_store(root)
    import kitchen_data as _kd2
    _pl = _kd2.episode_plan("FloorPlan15", _kd2.CAPTURE["episodes"])
    _ids = {e["episode"] for e in _pl}
    ok({0, 1} <= _ids, "qc e2e prerequisite: synthetic episodes 0 / 1 are planned FloorPlan15 episodes")
    json.dump({"done_episodes": [0, 1], "failed_episodes": {}}, open(os.path.join(ds, "capture_state.json"), "w"))
    qd2 = os.path.join(tmp, "qc2")
    os.makedirs(qd2)
    json.dump({"covered": [0, 1], "hashes": {}, "batches": 0}, open(os.path.join(qd2, "qc_state.json"), "w"))
    a_ = ["batch", "--scene", "FloorPlan15", "--root", root, "--out-dir", qd2]
    import io as _io
    import contextlib as _cl
    _o = _io.StringIO()
    with _cl.redirect_stdout(_o):
        rc_ = k23_qc.main(a_)
    ok(rc_ == 2 and "no report checked" in _o.getvalue(),
       f"qc e2e r3 2: cached 'covered' without reports -> STOP rc 2, not 'nothing new' (got {rc_})")
    p1 = os.path.join(qd2, "batch_001.json")
    json.dump({"scene": "FloorPlan15", "batch": 1, "verdict": "PASS", "episodes": [0, 1],
               "episodes_by_split": _kdec.plan_splits("FloorPlan15", [0, 1]), "lines": GOODL}, open(p1, "w"))
    json.dump({"covered": [0, 1], "hashes": {}, "batches": 1, "reports": {"1": k23_qc.sha_file(p1)}},
              open(os.path.join(qd2, "qc_state.json"), "w"))
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a_)
    ok(rc_ == 4, f"qc e2e: reports that reconcile -> 'nothing new' rc 4 (got {rc_})")
    _trs = [e["episode"] for e in _pl if e["split"] == "train" and e["episode"] not in (0, 1)][:1]
    json.dump({"done_episodes": [0, 1] + _trs, "failed_episodes": {}}, open(os.path.join(ds, "capture_state.json"), "w"))
    _o = _io.StringIO()
    with _cl.redirect_stdout(_o):
        rc_ = k23_qc.main(a_)
    ok(rc_ == 2 and "'done' without a record" in _o.getvalue(), f"qc e2e r3 2: a 'done' TRAIN episode without a record and no demonstrated quota skip -> rc 2 (got {rc_})")
    json.dump({"done_episodes": [0], "failed_episodes": {}}, open(os.path.join(ds, "capture_state.json"), "w"))
    json.dump({"covered": [0], "hashes": {}, "batches": 1, "reports": {"1": k23_qc.sha_file(p1)}},
              open(os.path.join(qd2, "qc_state.json"), "w"))
    _o = _io.StringIO()
    with _cl.redirect_stdout(_o):
        rc_ = k23_qc.main(a_)
    ok(rc_ == 4 and "uncommitted" in _o.getvalue(),
       "qc e2e: a record not in capture_state is NOT counted as committed (listed, not checked)")
    # r4 2 (Astra repro): no reports, no records, the stored quota claims every train target met -> NOT 'nothing new'
    root3 = os.path.join(tmp, "root3")
    ds3 = os.path.join(root3, "FloorPlan15")
    os.makedirs(ds3)
    json.dump({"scene": "FloorPlan15", "registry": ["Apple"]}, open(os.path.join(ds3, "probe.json"), "w"))
    open(os.path.join(ds3, "frames.jsonl"), "w").write("")
    _tr3 = [e["episode"] for e in _pl if e["split"] == "train"][:3]
    json.dump({"done_episodes": _tr3, "failed_episodes": {},
               "quota": _kdat.Quota({u: {} for u in _kdat.Quota.UNITS}).to_dict()},
              open(os.path.join(ds3, "capture_state.json"), "w"))
    qd3 = os.path.join(tmp, "k3", "qc")
    os.makedirs(qd3)
    a3 = ["batch", "--scene", "FloorPlan15", "--root", root3, "--out-dir", qd3]
    _o = _io.StringIO()
    with _cl.redirect_stdout(_o):
        rc_ = k23_qc.main(a3)
    ok(rc_ == 2 and "no quota-skip log line" in _o.getvalue(),
       f"qc e2e r4 2 (Astra repro): no reports / records + quota 'complete' -> STOP rc 2, not 'nothing new' (got {rc_})")
    # first sight WITH the capture's log lines: provenance recorded once, then 'nothing new'
    open(os.path.join(tmp, "k3", "capture.log"), "w").write("".join(
        f"[07:41:0{i}] FloorPlan15: ep {e:3d} [train] skipped: every train target already met\n" for i, e in enumerate(_tr3)))
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a3)
    st3 = json.load(open(os.path.join(qd3, "qc_state.json")))
    ok(rc_ == 4 and sorted(int(k) for k in st3.get("skips", {})) == sorted(_tr3),
       f"qc e2e r4 2: first sight + log line + quota met -> per-episode skip provenance recorded, rc 4 (got {rc_})")
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a3)
    ok(rc_ == 4, f"qc e2e r4 2: a later run rests on the recorded, validated provenance (got {rc_})")
    import k23_decide
    _evp = os.path.join(qd3, k23_decide.SKIP_EVIDENCE.format(e=_tr3[0]))
    st3 = json.load(open(os.path.join(qd3, "qc_state.json")))
    ok(os.path.isfile(_evp) and st3["skips"][str(_tr3[0])]["evidence_sha256"] == k23_decide.sha_file(_evp),
       "qc e2e r7: first sight RETAINS qc/skip_evidence_<ep>.json and binds it by sha256")
    _evb = open(_evp, "rb").read()
    _unmet7 = _kdat.Quota({"type": {"train": {"Apple": 5}}, "inst": {}, "state": {}, "combo": {}}).to_dict()
    for _mut, _nm in ((lambda b: dict(b, quota=_unmet7), "with an unmet quota (hash differs)"),
                      (lambda b: dict(b, first_seen_batch=9), "naming another batch"), (None, "deleted")):
        if _mut is None:
            os.remove(_evp)
        else:
            json.dump(_mut(json.loads(_evb)), open(_evp, "w"), sort_keys=True)
        with _cl.redirect_stdout(_io.StringIO()):
            rc_ = k23_qc.main(a3)
        ok(rc_ == 2, f"qc e2e r7: retained skip evidence {_nm} -> STOP rc 2 (got {rc_})")
        open(_evp, "wb").write(_evb)
    # r7: evidence rewritten CONSISTENTLY (sha + quota hash re-bound in qc_state) but with an unmet quota: refused
    _st7b = json.load(open(os.path.join(qd3, "qc_state.json")))
    _st7 = json.loads(json.dumps(_st7b))
    _b7 = dict(json.loads(_evb), quota=_unmet7)
    _b7["quota_sha256"] = k23_decide.quota_sha(_unmet7)
    json.dump(_b7, open(_evp, "w"), sort_keys=True)
    _st7["skips"][str(_tr3[0])].update(evidence_sha256=k23_decide.sha_file(_evp), quota_sha256=_b7["quota_sha256"])
    json.dump(_st7, open(os.path.join(qd3, "qc_state.json"), "w"))
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a3)
    ok(rc_ == 2, f"qc e2e r7: a re-bound evidence file whose quota is NOT met proves nothing (got {rc_})")
    open(_evp, "wb").write(_evb)
    json.dump(_st7b, open(os.path.join(qd3, "qc_state.json"), "w"))
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a3)
    ok(rc_ == 4, f"qc e2e r7: evidence restored -> provenance valid again (got {rc_})")
    # r5 2 (Astra repro): a stored provenance entry that is not a valid record proves nothing
    _good = json.load(open(os.path.join(qd3, "qc_state.json")))
    for bad_entry, name_ in ((None, "null"), ({"episode": _tr3[0]}, "fields missing"),
                             (dict(_good["skips"][str(_tr3[0])], log_line="FloorPlan17: x"), "another scene's line"),
                             (dict(_good["skips"][str(_tr3[0])], episode=_tr3[1]), "another episode")):
        st_b = json.loads(json.dumps(_good))
        st_b["skips"][str(_tr3[0])] = bad_entry
        json.dump(st_b, open(os.path.join(qd3, "qc_state.json"), "w"))
        with _cl.redirect_stdout(_io.StringIO()):
            rc_ = k23_qc.main(a3)
        ok(rc_ == 2, f"qc e2e r5 2: skip provenance {name_} -> STOP rc 2 (got {rc_})")
    json.dump(_good, open(os.path.join(qd3, "qc_state.json"), "w"))
    os.remove(os.path.join(tmp, "k3", "capture.log"))
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a3)
    ok(rc_ == 2, f"qc e2e r5 2: provenance whose log line is gone from the capture log is not evidence (got {rc_})")
    open(os.path.join(tmp, "k3", "capture.log"), "w").write(
        f"[07:41:00] FloorPlan17: ep {_tr3[0]:3d} [train] skipped: every train target already met\n")
    st3["skips"] = {}
    json.dump(st3, open(os.path.join(qd3, "qc_state.json"), "w"))
    with _cl.redirect_stdout(_io.StringIO()):
        rc_ = k23_qc.main(a3)
    ok(rc_ == 2, f"qc e2e r4 2: a skip line of ANOTHER scene proves nothing (got {rc_})")
    # a record an earlier report checked has vanished -> STOP (the 'quota met now' rule alone would accept it)
    json.dump({"done_episodes": [0], "failed_episodes": {},
               "quota": _kdat.Quota({u: {} for u in _kdat.Quota.UNITS}).to_dict()},
              open(os.path.join(ds, "capture_state.json"), "w"))
    rows_ = [l for l in open(os.path.join(ds, "frames.jsonl")).read().splitlines() if '"episode": 0' not in l
             and not l.startswith('{"episode_record": {"episode": 0,')]
    open(os.path.join(ds, "frames.jsonl"), "w").write("\n".join(rows_) + "\n")
    _o = _io.StringIO()
    with _cl.redirect_stdout(_o):
        rc_ = k23_qc.main(a_)
    ok(rc_ == 2 and "record is gone" in _o.getvalue(),
       f"qc e2e r4 2: an episode an earlier report checked lost its record -> STOP rc 2 (got {rc_})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- r3 5: owned-only simulator restart
TB = {101: (100, 100, 100, "/x/thor-Linux64 -screen"),           # child of the capture
      102: (101, 100, 100, "/x/thor-Linux64 helper"),            # grandchild
      103: (1, 100, 100, "/x/thor-Linux64 orphan"),              # orphaned, still in the capture's own session / group
      104: (1, 500, 500, "/x/thor-Linux64 foreign"),             # another job's simulator
      105: (1, 100, 7, "/x/thor-Linux64 other-session"),         # same group number, other session: not ours
      106: (100, 100, 100, "python other"),                      # our child, not a simulator
      107: (999, 999, 999, "/x/thor-Linux64 person")}            # a person's simulator
ok(k23_capture.owned_simulators(TB, 100, 100, 100, "[t]hor-Linux64") == [101, 102, 103],
   "scoped restart r3 5: only descendants + members of the capture's own session / group")
ok(k23_capture.owned_simulators(TB, 100, 100, 42, "[t]hor-Linux64") == [101, 102],
   "scoped restart: group members count only when the group is a fresh session (pgid == sid)")
ok(k23_capture.owned_simulators(TB, 100, 1, 1, "[t]hor-Linux64") == [101, 102], "scoped restart: never group 1")


class _FakeBkd:
    made = []

    @staticmethod
    def make_agent(scene):
        _FakeBkd.made.append(scene)
        return "agent2"

    @staticmethod
    def _restart_agent(agent, scene):
        raise AssertionError("the frozen global-pkill restart must never run in the K23 path")


class _Agent:
    stopped = False

    def stop(self):
        _Agent.stopped = True


_kos, _sl = k23_capture.kill_owned_simulators, k23_capture.time.sleep
k23_capture.kill_owned_simulators = lambda grace=10.0: {"termed": [101], "killed": []}
k23_capture.time.sleep = lambda s: None
try:
    k23_capture.install_scoped_restart(_FakeBkd, "FloorPlan15")
    with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        out_ = _FakeBkd._restart_agent(_Agent(), "FloorPlan15")
    ok(out_ == "agent2" and _Agent.stopped and _FakeBkd.made == ["FloorPlan15"],
       "scoped restart: the runtime replacement stops the agent, kills owned only, starts a fresh agent")
finally:
    k23_capture.kill_owned_simulators, k23_capture.time.sleep = _kos, _sl
# r9: a state action known to hang AI2-THOR (OpenObject on Blinds) is recorded unreachable WITHOUT being called
import vocab as _vocab9  # noqa: E402


def _real_set_state9(agent, oid, target):
    _FakeBkd9.calls.append(oid)
    return {"target": target, "steps": [], "errors": [], "realised": True, "pieces": []}


class _FakeBkd9:
    vocab = _vocab9
    calls = []
    set_state = staticmethod(_real_set_state9)
    make_agent = staticmethod(lambda scene_: None)


class _Agent9:
    objs = {"Blinds|1": {"objectType": "Blinds", "isOpen": False, "openable": True},
            "Cabinet|1": {"objectType": "Cabinet", "isOpen": False, "openable": True}}

    def object_by_id(self, oid):
        return self.objs.get(oid)


_xd9 = __import__("tempfile").mkdtemp()
os.makedirs(os.path.join(_xd9, "FloorPlan15"))
_excl0, k23_capture.EXCL_DIR = k23_capture.EXCL_DIR, _xd9
k23_capture.install_hang_skip(_FakeBkd9, "FloorPlan15")
with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
    _r9 = _FakeBkd9.set_state(_Agent9(), "Blinds|1", {"open": True})
    _r9b = _FakeBkd9.set_state(_Agent9(), "Blinds|1", {"open": False})
    _r9c = _FakeBkd9.set_state(_Agent9(), "Cabinet|1", {"open": True})
ok(not _r9["realised"] and _FakeBkd9.calls.count("Blinds|1") == 1 and "EXCLUDED" in _r9["errors"][0][2],
   "r9: OpenObject on Blinds is not attempted and is recorded unreachable")
ok(_r9b["realised"] and _r9c["realised"] and _FakeBkd9.calls == ["Blinds|1", "Cabinet|1"],
   "r9: a no-op Blinds target and every other type still go to the real set_state")
_x9 = json.load(open(os.path.join(_xd9, "FloorPlan15", "excluded_states.json"), encoding="utf-8"))
ok(len(_x9) == 1 and _x9[0]["amendment"] == "K23-AMD-1" and _x9[0]["action"] == "OpenObject",
   "r9: the exclusion is listed in excluded_states.json (EXCLUDED, not empirically unreachable)")
k23_capture.EXCL_DIR = _excl0
_FakeBkd9.calls.clear()
_FakeBkd9.set_state = staticmethod(_real_set_state9)
k23_capture.install_hang_skip(_FakeBkd9, "FloorPlan17")
with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
    _FakeBkd9.set_state(_Agent9(), "Blinds|1", {"open": True})
ok(_FakeBkd9.calls == ["Blinds|1"], "r9: the amendment is scoped to FloorPlan15 (another scene attempts the action)")
# r10: at the agent level OpenObject / CloseObject on FloorPlan15 Blinds are never sent (paired renders); others are


class _DoAgent10:
    def __init__(self):
        self.sent = []

    def do(self, action, **kw):
        self.sent.append((action, kw.get("objectId")))
        return "real"


with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
    _a10 = k23_capture.wrap_agent_excluded(_DoAgent10(), "FloorPlan15")
    _e10 = _a10.do("OpenObject", objectId="Blinds|-01.41|+02.09|-00.30", forceAction=True, openness=1.0)
    _e10b = _a10.do("CloseObject", objectId="Blinds|-01.41|+02.09|-00.30", forceAction=True)
    _r10 = _a10.do("OpenObject", objectId="Cabinet|1", forceAction=True)
    _b10 = k23_capture.wrap_agent_excluded(_DoAgent10(), "FloorPlan17")
    _r10b = _b10.do("OpenObject", objectId="Blinds|1")
ok(_e10.metadata["lastActionSuccess"] is False and "EXCLUDED" in _e10.metadata["errorMessage"]
   and _e10b.metadata["lastActionSuccess"] is False and _a10.sent == [("OpenObject", "Cabinet|1")] and _r10 == "real",
   "r10: FloorPlan15 Blinds Open/Close are not sent (plain failure); other objects are")
ok(_r10b == "real" and _b10.sent == [("OpenObject", "Blinds|1")], "r10: the agent-level exclusion is scoped to FloorPlan15")
import build_kitchen_data as _bkd0  # noqa: E402
ok("pkill" in open(_bkd0.__file__, encoding="utf-8").read(),
   "Kitchen 1 code untouched: build_kitchen_data.py still has its own restart (replaced only at runtime in K23)")
ok(b"pkill -f '[t]hor-Linux64'" not in open(os.path.join(SIM, "..", "docs", "kitchen-datasets-k2k3-capture-run.md"), "rb").read(),
   "run plan r3 5: no global simulator kill in the emergency command")

# ---------------------------------------------------------------- r5 3-4: ownership rule (k23_procs, pure part)
import k23_procs as kp  # noqa: E402
SN = {10: {"ppid": 1, "pgid": 10, "sid": 10, "start": 100, "token": "k23:FloorPlan15:s1"},     # batch script (leader)
      11: {"ppid": 10, "pgid": 11, "sid": 11, "start": 101, "token": "k23:FloorPlan15:s1"},    # capture session leader
      12: {"ppid": 11, "pgid": 11, "sid": 11, "start": 102, "token": ""},                      # tokenless child
      13: {"ppid": 1, "pgid": 11, "sid": 11, "start": 103, "token": ""},                       # tokenless orphan, same session
      14: {"ppid": 12, "pgid": 14, "sid": 14, "start": 104, "token": ""},                      # setsid'd descendant
      20: {"ppid": 1, "pgid": 20, "sid": 20, "start": 50, "token": ""},                        # foreign session
      21: {"ppid": 20, "pgid": 20, "sid": 20, "start": 51, "token": "k23:FloorPlan15:s1"},     # tagged, foreign session
      30: {"ppid": 1, "pgid": 30, "sid": 30, "start": 60, "token": "k23:FloorPlan17:s9"},      # other scene
      40: {"ppid": 1, "pgid": 40, "sid": 40, "start": 70, "token": "k23o:FloorPlan15:i"}}      # outer, own pid
_kst, _ktok = kp._stat, kp.token_of        # the leader re-check reads /proc: answer from the synthetic table
CUR = {"t": SN}
kp._stat = lambda p: (lambda v: (v["ppid"], v["pgid"], v["sid"], v["start"]) if v else None)(CUR["t"].get(p))
kp.token_of = lambda p: (CUR["t"].get(p) or {}).get("token", "")
o_ = kp.owned(SN, "k23:FloorPlan15:s1")
ok(set(o_) == {10, 11, 12, 13, 14, 21}, f"procs r5 3: tagged + tokenless members of K23 sessions + descendants ({sorted(o_)})")
ok(20 not in o_ and 30 not in o_ and 40 not in o_,
   "procs r5 4: a session is verified only by a TAGGED LEADER (foreign session 20 untouched though 21 is tagged)")
ok(set(kp.owned(SN, "k23:FloorPlan15:s1", session=11)) == {11, 12, 13, 14}, "procs: --session restricts to that session")
ok(set(kp.owned(SN, "k23:FloorPlan15:*")) == {10, 11, 12, 13, 14, 21}, "procs: a scene prefix never reaches another scene")
st_ = {"rem": {}, "vsess": set()}
kp.owned(SN, "k23:FloorPlan15:s1", state=st_)
SN2 = {k: v for k, v in SN.items() if k in (13, 20)}                     # the tagged ones are gone, the orphan remains
CUR["t"] = SN2
ok(13 in kp.owned(SN2, "k23:FloorPlan15:s1", state=st_), "procs r5 3: a remembered tokenless member stays owned")
SN3 = {13: dict(SN[13], start=999), 20: SN[20]}                          # pid 13 reused by another process
CUR["t"] = SN3
ok(13 not in kp.owned(SN3, "k23:FloorPlan15:s1", state=st_), "procs r5 4: a reused pid (other start time) is not owned")
CUR["t"] = {**SN, 11: dict(SN[11], start=555)}          # r6 4: the leader changed during the scan
ok(12 not in kp.owned(SN, "k23:FloorPlan15:s1", session=11),
   "procs r6 4: a leader that is not the same process after the scan verifies nothing (no reuse mid-scan)")
# r8 (Astra r7 4): a ppid link counts only when re-verified after the scan (parent pid reuse mid-enumeration)
SN4 = {**SN, 50: {"ppid": 11, "pgid": 50, "sid": 50, "start": 600, "token": ""}}      # child of a REUSED pid 11
CUR["t"] = {**SN4, 11: dict(SN[11], start=555)}                                      # 11 is now another process
ok(50 not in kp.owned(SN4, "k23:FloorPlan15:*"), "procs r8: a child of a parent pid reused during the scan is not owned")
CUR["t"] = SN4
ok(50 in kp.owned(SN4, "k23:FloorPlan15:*"), "procs r8: control -- the same child of the unchanged parent is owned")
SN5 = {**SN, 51: {"ppid": 11, "pgid": 51, "sid": 51, "start": 90, "token": ""}}       # 'child' older than its parent
CUR["t"] = SN5
ok(51 not in kp.owned(SN5, "k23:FloorPlan15:*"), "procs r8: a process that started before its parent links nothing")
CUR["t"] = {**SN4, 50: dict(SN4[50], ppid=1)}                                          # re-parented after the scan
ok(50 not in kp.owned(SN4, "k23:FloorPlan15:*"), "procs r8: a link that changed after the scan is not evidence")
CUR["t"] = SN
_d = kp.descendants(10, SN)
ok(set(_d) == {11, 12, 14} and 13 not in _d, f"procs r8: verified descendants of the holder ({sorted(_d)})")
kp._stat, kp.token_of = _kst, _ktok
ok(kp.matches("k23:FloorPlan15:x", "k23:FloorPlan15:*") and not kp.matches("", "k23:FloorPlan15:*")
   and not kp.matches("k23:FloorPlan150:x", "k23:FloorPlan15:x"), "procs: token matching")
ok(kp.main(["stop", "--token", "*"]) == 2 and kp.main(["stop", "--token", "k23:*"]) == 2,
   "procs: an unscoped token is refused")
# ---------------------------------------------------------------- r5 6: Kitchen 1 priority check (fail closed)
import datetime as _dt  # noqa: E402
import k1_priority_k23 as kpr  # noqa: E402
NOW = _dt.datetime(2026, 10, 14, 16, 0)
LG = "\n".join(["## 2026-10-14 15:00 -- STAGE ACCEPTANCE-V2 -- STARTED: attempt 1",
                "## 2026-10-14 15:30 -- STAGE FREEZE-PLUMBING -- STARTED: x"])
ok(any("ACCEPTANCE-V2" in w for w in kpr.ledger_busy(LG, NOW)), "priority: an active acceptance-v2 stage = busy")
ok(kpr.ledger_busy(LG + "\n## 2026-10-14 15:50 -- STAGE ACCEPTANCE-V2 -- DONE: x", NOW) == [],
   "priority: a DONE later heading frees it; PLUMBING stages never count")
ok(kpr.ledger_busy("## 2026-10-14 15:40 -- ARASH DECISION: the acceptance starts now", NOW) != [],
   "priority: a decision of the last hour that acceptance starts = busy")
ok(kpr.ledger_busy("## 2026-10-14 13:20 -- ARASH DECISION: run acceptance in parallel", NOW) == [],
   "priority: an older decision does not block")
_run = kpr.run
try:
    kpr.run = lambda cmd, timeout=60, stdin=None: ("", "query wsl exited 1")
    ok(kpr.wsl_busy() != [] and kpr.windows_busy() != [] and kpr.gpu_busy() != [],
       "priority r5 6 (Astra repro): a failing query with empty output is BUSY, never clear")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("LOCK_FREE\n", None)
    ok(kpr.wsl_busy() != [], "priority: a WSL check without its completion marker is busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("LOCK_FREE\nCHECK_DONE\n", None)
    ok(kpr.wsl_busy() == [], "priority: a complete clean WSL check is free")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("LOCK_FREE\nWAITER 77 flock /home/arash/sim.lock python x\nCHECK_DONE\n", None)
    ok(kpr.wsl_busy() != [], "priority: a process waiting for the sim lock = busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("python.exe|python sim\\final_test\\acceptance_v2\\acc_runner.py run\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() != [], "priority: acc_runner.py running on Windows = busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("python.exe|python x.py --prompt 'about freeze.py'\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() == [], "priority: a script name inside another argument does not count")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("1, C:\\x\\ollama.exe\n2, C:\\Windows\\explorer.exe\n", None)
    ok(kpr.gpu_busy() == [], "priority K2-CAPTURE: an Ollama GPU app and a desktop app are not busy by themselves")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("1, C:\\x\\python.exe\n2, C:\\x\\llama-server.exe\n3, C:\\x\\mystery.exe\n", None)
    ok(len(kpr.gpu_busy()) == 3, "priority: python / a foreign llama-server / an unknown GPU app = busy")
    # Windows: Ollama itself is not busy; a Kitchen 1 local planner is; a remote planner stage is not
    kpr.run = lambda cmd, timeout=60, stdin=None: ("ollama.exe|ollama serve\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() == [] and kpr.OLLAMA_SEEN == ["ollama.exe"],
       "priority K2-CAPTURE: Ollama running (even all night) is not busy by itself")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("llama-server.exe|llama-server -m x.gguf\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() != [], "priority: a llama-server that is not Ollama's = busy (unknown GPU user)")
    kpr.run = lambda cmd, timeout=60, stdin=None: (
        "python.exe|python -u sim/final_test/acceptance_v2/dress/dress.py --ns d plan --stage gemma\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() != [], "priority K2-CAPTURE: a Kitchen 1 Gemma planner stage = busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: (
        "python.exe|python -u sim/final_test/acceptance_v2/dress/dress.py --ns d plan --stage opus\n"
        "python.exe|python -u sim/final_test/acceptance_v2/dress/dress.py --ns d plan --stage codex\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() == [], "priority K2-CAPTURE: remote planner stages (Batch API / Codex) are not busy")
    # K2-K1TEST (2026-10-16): rule 7 -- the Kitchen 1 locked test (test-v1)
    _k7 = tempfile.mkdtemp(prefix="k23k7_")
    _ev, _cl = os.path.join(_k7, "events.jsonl"), os.path.join(_k7, "test-chain.log")
    def _w(p, lines):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("".join(l + "\n" for l in lines))
    _S, _F0, _F1 = ('{"event": "plan_first_start"}', '{"event": "plan_first_stage", "stage": "free", "code": 0}',
                    '{"event": "plan_first_stage", "stage": "free", "code": 3}')
    _w(_ev, [_S, _F0]); ok(kpr.k1_test_free_done(_ev), "priority rule 7: free stage done after the newest start")
    _w(_ev, [_S, _F0, _S]); ok(not kpr.k1_test_free_done(_ev), "priority rule 7: a restart without free done yet = not done")
    _w(_ev, [_S, _F1]); ok(not kpr.k1_test_free_done(_ev), "priority rule 7: free stage stopped (code != 0) = not done")
    ok(not kpr.k1_test_free_done(os.path.join(_k7, "missing.jsonl")), "priority rule 7: events unreadable = not done")
    _PF = ("python.exe|C:\\Py\\python.exe sim/final_test/matrix/planfirst.py --namespace test-v1 --allow-test --batch "
           "--batch-hold-usd 25 --parallel 4 --codex-parallel 2 --ollama http://127.0.0.1:11434\nCHECK_DONE\n")
    _keep = kpr.K1_TEST_EVENTS
    try:
        kpr.run = lambda cmd, timeout=60, stdin=None: (_PF, None)
        kpr.K1_TEST_EVENTS = _ev
        _w(_ev, [_S, _F0])
        ok(kpr.windows_busy() == [] and len(kpr.K1_TEST_PLANNERS) == 1,
           "priority rule 7: a test-v1 planfirst past its free stage is not busy (remote stages only)")
        _w(_cl, ["x CHAIN START test-v1", "x STAGE plan: plan"])
        ok(kpr.k1_test_busy(_cl) == [], "priority rule 7: locked test planning remotely = free")
        _w(_ev, [_S])
        ok(kpr.windows_busy() != [] and kpr.K1_TEST_PLANNERS == [],
           "priority rule 7: a test-v1 planfirst still in its free (Gemma) stage = busy (local planner)")
        ok(kpr.k1_test_busy(_cl) != [], "priority rule 7: no remote-stage planner while the test is unfinished = busy")
        kpr.run = lambda cmd, timeout=60, stdin=None: ("CHECK_DONE\n", None)
        kpr.windows_busy()
        ok(kpr.k1_test_busy(_cl) != [], "priority rule 7: locked test stopped / restarting / before execute = busy")
        _w(_cl, ["x CHAIN START test-v1", "x STAGE plan rc=0", "x STAGE execute: execute"])
        ok(kpr.k1_test_busy(_cl) != [], "priority rule 7: execute stage running = busy")
        _w(_cl, ["x STAGE execute rc=0", "x STAGE score: score"])
        ok(kpr.k1_test_busy(_cl) == [], "priority rule 7: execute complete = rule 7 inactive")
        ok(kpr.k1_test_busy(os.path.join(_k7, "none.log")) == [], "priority rule 7: no locked test chain log = inactive")
        kpr.run = lambda cmd, timeout=60, stdin=None: (_PF.replace("test-v1", "test-v2"), None)
        _w(_ev, [_S, _F0])
        ok(kpr.windows_busy() != [], "priority rule 7: a planfirst of another namespace stays a local planner (busy)")
    finally:
        kpr.K1_TEST_EVENTS = _keep
    for _cl in ('powershell -File "D:\\Personal\\Master\'s Project\\sim\\capture_all.ps1"',
                "powershell -NoProfile -File 'D:\\a b\\amb_capture.ps1' -X", "pwsh -Command & 'C:\\x y\\train_only.ps1'",
                "powershell -File sim\\kitchen_v3\\balance_train.ps1"):
        kpr.run = lambda cmd, timeout=60, stdin=None, _c=_cl: ("powershell.exe|" + _c + "\nCHECK_DONE\n", None)
        ok(kpr.windows_busy() != [], f"priority r8 (Astra r7 6): a supervisor script with a quoted / spaced path = busy ({_cl[:40]})")
    # r8b (Astra r8 6): the prefilter is case-insensitive too
    for _cl in ('powershell -File "C:\\X Y\\CAPTURE_ALL.PS1"', "PowerShell -Command & 'D:\\a\\Amb_Train.Ps1'"):
        kpr.run = lambda cmd, timeout=60, stdin=None, _c=_cl: ("powershell.exe|" + _c + "\nCHECK_DONE\n", None)
        ok(kpr.windows_busy() != [], f"priority r8b (Astra r8 6): an upper / mixed-case supervisor path = busy ({_cl[:40]})")
    # r8b: Ollama's own runner (lib/ollama/llama-server.exe in its install folder) is Ollama; other llama-servers stay busy
    _orun = "C:\\Users\\arash\\AppData\\Local\\Programs\\Ollama\\lib\\ollama\\llama-server.exe"
    kpr.run = lambda cmd, timeout=60, stdin=None: (f'llama-server.exe|"{_orun}" --model blob --port 5\nCHECK_DONE\n', None)
    ok(kpr.windows_busy() == [] and kpr.OLLAMA_SEEN == ["llama-server.exe"], "priority r8b: Ollama's own llama-server runner is Ollama")
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"llama-server.exe|{_orun} --model blob\nCHECK_DONE\n", None)
    ok(kpr.windows_busy() == [], "priority r8b: Ollama's runner, unquoted executable path, is Ollama")
    for _cl in ("C:\\tools\\llama\\llama-server.exe -m x", "C:\\x\\Ollama\\lib\\ollama\\llama-server.exe.bak -m x",
                "C:\\x\\NotOllama\\lib\\other\\llama-server.exe -m x",
                "C:\\tools\\llama-server.exe --log-file C:\\Ollama\\lib\\ollama\\llama-server.exe",
                '"C:\\tools\\llama-server.exe" --x "C:\\Ollama\\lib\\ollama\\llama-server.exe"'):
        kpr.run = lambda cmd, timeout=60, stdin=None, _c=_cl: ("llama-server.exe|" + _c + "\nCHECK_DONE\n", None)
        ok(kpr.windows_busy() != [], f"priority r8b: a llama-server outside Ollama's lib folder = busy ({_cl[:40]})")
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"1, {_orun}\n2, C:\\x\\llama-server.exe\n", None)
    ok(len(kpr.gpu_busy()) == 1, "priority r8b: on the GPU list Ollama's runner is not busy, a foreign llama-server is")
    # Ollama /api/ps must answer when Ollama runs
    kpr.OLLAMA_SEEN[:] = ["ollama.exe"]
    ok(kpr.ollama_busy(fetch=lambda: 1 / 0) != [] and kpr.LOADED["models"] is None,
       "priority: a running Ollama whose /api/ps fails = busy (fail closed)")
    ok(kpr.ollama_busy(fetch=lambda: {"models": "x"}) != [], "priority: an unreadable /api/ps = busy")
    ok(kpr.ollama_busy(fetch=lambda: {"models": [{"name": "gemma4:e2b-it-q4_K_M"}]}) == []
       and kpr.LOADED["models"] == ["gemma4:e2b-it-q4_K_M"], "priority: a loaded Gemma is recorded, not busy")
    # utilization + VRAM rule (Gemma loaded: no reserve)
    _need, _ = kpr.capture_need()
    _ok_free = _need + kpr.MARGIN_MIB
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"10, {_ok_free}\n", None)
    ok(kpr.vram_util_busy(pause=0) == [], "priority K2-CAPTURE: Gemma loaded + idle GPU + free >= capture + margin = free")
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"10, {_ok_free - 1}\n", None)
    ok(kpr.vram_util_busy(pause=0) != [], "priority: free VRAM below capture + margin = busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"85, {_ok_free + 4000}\n", None)
    ok(kpr.vram_util_busy(pause=0) != [], "priority: the GPU working (Gemma generating) = busy")
    _u = iter(["90, 9000\n", "10, 9000\n", "12, 9000\n"])
    kpr.run = lambda cmd, timeout=60, stdin=None: (next(_u), None)
    ok(kpr.vram_util_busy(pause=0) == [], "priority: one utilization spike (median of 3) is not busy")
    kpr.LOADED["models"] = []
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"10, {_ok_free}\n", None)
    ok(kpr.vram_util_busy(pause=0) != [], "priority K2-CAPTURE: Gemma NOT loaded -> its reserve is kept free (busy here)")
    kpr.run = lambda cmd, timeout=60, stdin=None: (f"10, {_ok_free + kpr.GEMMA_RESERVE_MIB}\n", None)
    ok(kpr.vram_util_busy(pause=0) == [], "priority: Gemma not loaded but room for it + the capture = free")
    kpr.LOADED["models"] = None
    ok(kpr.vram_util_busy(pause=0) != [], "priority: Ollama's loaded models unknown = busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("garbage\n", None)
    kpr.LOADED["models"] = []
    ok(kpr.vram_util_busy(pause=0) != [], "priority: unreadable nvidia-smi numbers = busy")
    kpr.run = lambda cmd, timeout=60, stdin=None: ("", "query nvidia-smi exited 9")
    ok(kpr.vram_util_busy(pause=0) != [], "priority: a failed nvidia-smi = busy")
    _vf = kpr.VRAM_FILE
    _tdv = tempfile.mkdtemp(prefix="k23vr_")
    try:
        kpr.VRAM_FILE = os.path.join(_tdv, "v.json")
        ok(kpr.capture_need()[0] == kpr.CAPTURE_MIB_DEFAULT, "priority: no measurement -> the default need")
        json.dump({"capture_mib": 1800, "source": "probe"}, open(kpr.VRAM_FILE, "w"))
        ok(kpr.capture_need()[0] == 1800, "priority: the measured need is used")
        for _bad in ({"capture_mib": 10}, {"capture_mib": True}, {"capture_mib": "2000"}, {}):
            json.dump(_bad, open(kpr.VRAM_FILE, "w"))
            ok(kpr.capture_need()[0] == kpr.CAPTURE_MIB_DEFAULT, f"priority: a bad measurement {_bad} -> the default")
    finally:
        kpr.VRAM_FILE = _vf
finally:
    kpr.run = _run

# ---------------------------------------------------------------- inspection rule
tmp = tempfile.mkdtemp(prefix="k23in_")
try:
    sp = os.path.join(tmp, "capture_state.json")
    json.dump({"done_episodes": list(range(17))}, open(sp, "w"))
    split = {e: ("val" if e == 15 else "train") for e in range(17)}
    ok(k23_qc.inspection_due(sp, split) == [7], "qc: inspection = every 8th committed episode, TRAIN only")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- scripts parse
bash = shutil.which("bash")
for f in ("capture_one_k23.sh", "capture_inner_k23.sh", "preflight_k23.sh", "status20_k23.sh", "emergency_stop_k23.sh",
          "k23_own.sh"):
    p = os.path.join(HERE, f)
    if bash:
        r = subprocess.run([bash, "-n", p], capture_output=True, text=True)
        ok(r.returncode == 0, f"bash -n {f}: {r.stderr.strip()[:120]}")
    ok(b"\r\n" not in open(p, "rb").read(), f"{f}: LF line endings (WSL bash)")
if os.name == "nt":
    for f in ("capture_k23.ps1",):
        cmd = ("$e=$null; [System.Management.Automation.Language.Parser]::ParseFile('%s',[ref]$null,[ref]$e) | "
               "Out-Null; if($e){exit 1}" % os.path.join(HERE, f).replace("'", "''"))
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        ok(r.returncode == 0, f"PowerShell parse {f}")

print(f"_k23_selftest: {N['ok']}/{N['ok'] + N['fail']} {'OK' if not N['fail'] else 'FAILED'}")
sys.exit(1 if N["fail"] else 0)
