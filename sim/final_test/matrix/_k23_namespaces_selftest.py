"""Offline gate for the K23 namespaces + A29 roster (Astra K23-NAMESPACES-r1 items 1, 2, 5; build plan step 2)."""
import json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import a29_sample as S
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def raises(f, exc=Exception):
    try:
        f()
    except exc:
        return True
    return False


K1 = ("rehearsal-v1", "rehearsal-v2", "practice-v1", "test-v1", "practice-v2", "rehearsal-v3", "acceptance-v2", "acceptance-v2-a2", "freeze-gates-ft2")
check("K1 namespaces have no scene key (FloorPlan1 default)", all("scene" not in C.NAMESPACES[n] and C.ns_scene(n) == "FloorPlan1" for n in K1))
check("K1 not replication", not any(C.is_replication(n) for n in K1))
for k, sc in (("k2", "FloorPlan15"), ("k3", "FloorPlan17")):
    ns = {f"{x}-{k}-v1": C.NAMESPACES[f"{x}-{k}-v1"] for x in ("practice", "rehearsal", "test")}
    check(f"{k} scenes", all(v["scene"] == sc for v in ns.values()) and all(C.is_replication(n) for n in ns))
    check(f"{k} practice 0 USD Gemma pool", ns[f"practice-{k}-v1"]["items"] == "practice_pool" and ns[f"practice-{k}-v1"]["ceilings"]["total"] == 0)
    check(f"{k} test = A29 roster, replication_test", ns[f"test-{k}-v1"]["items"] == C.A29_ROSTER and ns[f"test-{k}-v1"].get("replication_test"))
    check(f"{k} ft-v2, no archived arm", all(v["treatment"] == "ft-v2" and v["archived_arms"] == () for v in ns.values()))
tot = sum(C.NAMESPACES[n]["ceilings"]["total"] for n in C.NAMESPACES if C.is_replication(n))
check("replication ceilings sum to the 70 USD approval", tot == C.REPLICATION_USD == 70.0)
man = C.load_manifest()
check("REPEAT / relocation off in replication kitchens", all(C.repeat_items(n, man) == [] and C.reloc_items(n, man) == []
                                                            for n in C.NAMESPACES if C.is_replication(n)))
check("practice-k2 items = practice_pool", C.namespace_items("practice-k2-v1", man) == list(man["practice_pool"]))

# A29 roster on a temp docs tree (the real docs/final-test/k23 stays untouched)
tmp = tempfile.mkdtemp()
orig_paths = C.paths
C.paths = lambda root=None: dict(orig_paths(None), docs=tmp) if root == "TMP" else orig_paths(root)
check("no manifest -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
rep = S.build("FloorPlan15", {"kitchen": "FloorPlan15", "items": {i: "applicable" for i in man["test"]}})
os.makedirs(os.path.join(tmp, "k23"))
mp, dp = C.a29_files("FloorPlan15", "TMP")
json.dump(rep, open(mp, "w"))
check("no size decision -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
json.dump({"kitchens": {"FloorPlan15": {"candidate": "first_60", "manifest_sha256": rep["sha256"]}}}, open(dp, "w"))
r = C.a29_roster("test-k2-v1", "TMP")
check("roster = the decided candidate, in manifest order", r == rep["order"][:60] and len(r) == 60)
check("guard: practice ns refuses a roster test item", raises(lambda: C.guard_items([r[0]], "practice-k2-v1"), C.TestItemRefused))
json.dump({"kitchens": {"FloorPlan15": {"candidate": "first_60", "manifest_sha256": "0" * 64}}}, open(dp, "w"))
check("decision for another manifest -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
json.dump({"kitchens": {"FloorPlan15": {"candidate": "all_plus", "manifest_sha256": rep["sha256"]}}}, open(dp, "w"))
check("unknown candidate -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
bad = dict(rep); bad["candidates"] = dict(rep["candidates"], first_50=list(reversed(rep["candidates"]["first_50"])))
json.dump(bad, open(mp, "w"))
check("tampered manifest (self-hash) -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
bad["sha256"] = C.self_sha(bad); json.dump(bad, open(mp, "w"))
json.dump({"kitchens": {"FloorPlan15": {"candidate": "first_50", "manifest_sha256": bad["sha256"]}}}, open(dp, "w"))
check("non-nested candidate -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
k3 = dict(rep, kitchen="FloorPlan17"); k3["sha256"] = C.self_sha(k3)
json.dump(k3, open(mp, "w"))
check("manifest of another kitchen -> refused", raises(lambda: C.a29_roster("test-k2-v1", "TMP"), RuntimeError))
C.paths = orig_paths
check("replication test items fail closed until the per-kitchen freeze (step 4)",
      raises(lambda: C.guard_items([sorted(man["test"])[0]], "test-k2-v1", allow_test=True), (C.TestItemRefused, RuntimeError)))
import treatment as T
base = T.snapshot_root()
check("K1 ft-v2 snapshot paths unchanged", T.source_dir("practice-v2", "o0k10") == os.path.join(base, "o0", "k10")
      and T.source_dir("test-v1", "S*") == os.path.join(base, "o0", "k10"))
check("K2/K3 snapshots in their own subtree", T.source_dir("practice-k2-v1", "o0k10") == os.path.join(base, "FloorPlan15", "o0", "k10")
      and T.source_dir("test-k3-v1", "S*") == os.path.join(base, "FloorPlan17", "o0", "k10"))
check("kitchens never share a snapshot dir", len({T.source_dir(n, "S*") for n in ("test-v1", "test-k2-v1", "test-k3-v1")}) == 3)
check("K2/K3 planning arms = the core four", all(C.main_arms_for(n) == ("G0", "G+M", "O0", "O+M") for n in C.NAMESPACES if C.is_replication(n)))
check("K1 main arms unchanged", C.main_arms_for("test-v1") == tuple(C.MAIN_ARMS))
check("FS / Codex / curve arms refused in K2/K3", all(raises(lambda a=a: C.arm_info(a, "test-k2-v1"), ValueError) for a in ("G+FS", "O+FS", "X0", "X+M", "G+M@o0k5")))
check("P0 / P+M allowed in K2/K3", all(not raises(lambda a=a: C.arm_info(a, "test-k2-v1")) for a in ("P0", "P+M", "G0", "O+M")))
import k23_ns as K
st = K.replication_freeze_status("test-k2-v1")
check("no K2 freeze -> refused (K1 freeze.json never satisfies it)", not st["ok"] and "missing" in st["reasons"][0])
check("non-replication ns refused by the K23 freeze check", not K.replication_freeze_status("test-v1")["ok"])
tmp2 = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp2, "k23"))
C.paths = lambda root=None: dict(orig_paths(None), docs=tmp2) if root == "T2" else orig_paths(root)
fz = {"schema": K.FREEZE_SCHEMA, "scene": "FloorPlan17", "namespaces": ["test-k3-v1"], "gates": {"g": {"pass": True}}}
fz["sha256"] = C.self_sha(fz)
json.dump(fz, open(K.freeze_path("FloorPlan15", "T2"), "w"))
r = K.replication_freeze_status("test-k2-v1", "T2")["reasons"]
check("freeze of another kitchen refused", any("not for FloorPlan15" in x for x in r))
check("no sign-off refused", any("no sign-off" in x for x in r))
fz2 = dict(fz, scene="FloorPlan15", namespaces=["test-k2-v1"]); fz2["sha256"] = "0" * 64
json.dump(fz2, open(K.freeze_path("FloorPlan15", "T2"), "w"))
check("tampered freeze refused", any("self-sha" in x for x in K.replication_freeze_status("test-k2-v1", "T2")["reasons"]))
fz3 = dict(fz2, standing_presignature={"basis": "approve-all-6 item 1", "astra_review": "nope.txt", "freeze_summary": "nope.md"})
fz3["sha256"] = C.self_sha(fz3); json.dump(fz3, open(K.freeze_path("FloorPlan15", "T2"), "w"))
check("pre-signature without the Astra review / summary files refused", any("no sign-off" in x for x in K.replication_freeze_status("test-k2-v1", "T2")["reasons"]))
C.paths = orig_paths
w = C.paths()["matrix_work"]; rel = lambda n: os.path.relpath(os.path.join(w, n, "ledger.db"), C.paths()["root"]).replace(os.sep, "/")
sp = {"ledgers": {rel("test-k2-v1"): {"liability": 30.0}, rel("test-v1"): {"liability": 50.0}}}
check("replication budget: 30 + 39 inside", K.replication_problems(os.path.join(w, "test-k3-v1", "ledger.db"), 39.0, sp) == [])
check("replication budget: 30 + 41 > 70 refused", len(K.replication_problems(os.path.join(w, "test-k3-v1", "ledger.db"), 41.0, sp)) == 1)
check("replication budget inert on Kitchen 1 ledgers", K.replication_problems(os.path.join(w, "test-v1", "ledger.db"), 999.0, sp) == [])
# Astra K23-A37-ALIGN-r1 point 9: the same 70 USD cap through the REAL reservation path (ScopedLedger.reserve -> guard_problems), past the
# A37 paid-planning hold: a VALID alignment record in a temporary 'repo' root (common.ROOT points there for the block, so no real ledger,
# study lock or record is touched; nothing is sent).
import _k23_a37_fixture as F
R = F.make_root()


def _send(ns, usd):
    led = C.open_ledger(ns)
    with led.use("core"):
        return led.reserve("anthropic", usd)


def _refused(f, needle):
    try:
        f()
    except C.CostCeilingExceeded as e:
        return needle in str(e)
    return False


with F.as_repo(R):
    for n, usd in (("test-k2-v1", 31.5), ("rehearsal-k2-v1", 3.0), ("rehearsal-k3-v1", 3.0)):
        F.seed(os.path.join(C.ns_work(n), "ledger.db"), usd)
    F.seed(os.path.join(C.paths()["docs"], "k23", "archive", "test-k2-v1", "ledger.db"), 2.0)      # an archived copy counts too: 39.5
    check("real path, no alignment record: refused by the A37 hold (not the cap)", _refused(lambda: _send("test-k3-v1", 1.0), "A37 paid-planning hold"))
    F.write_record(R)
    check("valid A37 alignment fixture: the hold is lifted", K.a37_hold_problems() == [])
    check("real path: 39.5 + 31 > 70 refused by the replication cap", _refused(lambda: _send("test-k3-v1", 31.0), "replication budget"))
    check("real path: 39.5 + 30.5 = 70 inside", isinstance(_send("test-k3-v1", 30.5), int))
    check("real path: then +0.01 refused (cap reached)", _refused(lambda: _send("test-k3-v1", 0.01), "replication budget"))
    check("real path: a Kitchen 1 ledger is not bound by the K2/K3 cap", isinstance(_send("test-v1", 1.0), int))
check("common.ROOT restored after the fixture", C.paths()["root"] == C.ROOT == os.path.dirname(C.SIM))
shutil.rmtree(R, ignore_errors=True)
shutil.rmtree(tmp2, ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
