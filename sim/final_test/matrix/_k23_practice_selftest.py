"""Offline gate for the K23 practice entry (k23_entry.py, k23_practice.py) and its launcher (sim/final_test/k23_run/practice/).
No simulator, no Ollama, no provider: practice_k.build_live is replaced by a fake wherever a live path would start."""
import contextlib, copy, io, json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import k23_ns as K
import k23_entry as E
import treatment as T
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


def quiet(f, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            r = f(*a, **kw)
        except SystemExit as e:
            r = e.code
    return r, out.getvalue(), err.getvalue()


K1_PRACTICE = ("practice-v1", "rehearsal-v1", "practice-v2", "rehearsal-v3")
ROOT = C.paths()["root"]

# ---- 1 namespaces / kinds
check("K23 practice namespaces", E.k23_namespaces("practice") == ("practice-k2-v1", "practice-k3-v1"))
check("K23 rehearsal namespaces", E.k23_namespaces("rehearsal") == ("rehearsal-k2-v1", "rehearsal-k3-v1"))
check("K23 test namespaces", E.k23_namespaces("test") == ("test-k2-v1", "test-k3-v1"))
check("kinds", [E.kind(n) for n in ("practice-k2-v1", "rehearsal-k3-v1", "test-k2-v1")] == ["practice", "rehearsal", "test"])
check("every Kitchen 1 namespace refused by kind()", all(raises(lambda n=n: E.kind(n), ValueError) for n in C.NAMESPACES if C.ns_scene(n) == "FloorPlan1"))
check("unknown namespace refused", raises(lambda: E.kind("practice-k9-v1"), ValueError))
check("scene_of", E.scene_of("practice-k3-v1") == "FloorPlan17" and E.scene_of("test-k2-v1") == "FloorPlan15")

# ---- 2 A29 sample checks (temp copies; the real files are only read)
tmp = tempfile.mkdtemp(prefix="k23prac_")
real15 = K.a29_files("FloorPlan15")[0]
orig_files = K.a29_files


def fake_files(scene, root=None):
    return os.path.join(tmp, f"a29_sample_{scene}.json"), os.path.join(tmp, "a29_size_decision.json")


K.a29_files = fake_files
try:
    check("missing sample -> problem", any("no frozen A29 sample" in p for p in E.sample_problems("FloorPlan15")))
    if os.path.exists(real15):
        shutil.copy(real15, fake_files("FloorPlan15")[0])
        check("the real FloorPlan15 sample verifies", E.sample_problems("FloorPlan15") == [])
        check("both kitchens needed: FloorPlan17 missing is reported", any("FloorPlan17" in p for p in E.samples_problems())
              and not any("FloorPlan15" in p for p in E.samples_problems()))
        m = json.load(open(real15, encoding="utf-8"))
        bad = dict(m, order=list(reversed(m["order"])))
        json.dump(bad, open(fake_files("FloorPlan15")[0], "w"))
        check("edited order -> self-hash problem", any("self-hash" in p for p in E.sample_problems("FloorPlan15")))
        shutil.copy(real15, fake_files("FloorPlan17")[0])                  # the FloorPlan15 manifest at the FloorPlan17 path
        check("a manifest of the other kitchen is refused", any("names kitchen" in p for p in E.sample_problems("FloorPlan17")))
        nonnest = copy.deepcopy(m); nonnest["candidates"]["first_50"] = list(reversed(nonnest["candidates"]["first_50"])); nonnest["sha256"] = C.self_sha(nonnest)
        json.dump(nonnest, open(fake_files("FloorPlan15")[0], "w"))
        check("non-nested candidates refused", any("nested" in p for p in E.sample_problems("FloorPlan15")))
        only_all = dict(m, candidates={"all": m["order"]}); only_all["sha256"] = C.self_sha(only_all)
        json.dump(only_all, open(fake_files("FloorPlan15")[0], "w"))
        check("a manifest holding only 'all' is refused (all three candidates, registered lengths)",
              any("first_50 / first_60 / all" in p for p in E.sample_problems("FloorPlan15")))
        short = copy.deepcopy(m); short["candidates"]["first_60"] = short["order"][:59]; short["sha256"] = C.self_sha(short)
        json.dump(short, open(fake_files("FloorPlan15")[0], "w"))
        check("a candidate of the wrong length is refused", any("registered lengths" in p for p in E.sample_problems("FloorPlan15")))
        both = dict(m, kitchen="FloorPlan17"); both["sha256"] = C.self_sha(both)
        shutil.copy(real15, fake_files("FloorPlan15")[0]); json.dump(both, open(fake_files("FloorPlan17")[0], "w"))
        check("both present and intact -> []", E.samples_problems() == [])
    else:
        print("NOTE: no FloorPlan15 sample in this tree; sample checks skipped"); check("FloorPlan15 sample present", False)
finally:
    K.a29_files = orig_files

# ---- 3 package inputs
check("FloorPlan15 package final (k23-v1, approved briefing)", E.package_problems("FloorPlan15") == [])
import ltm_package as LP
d = json.load(open(LP.uses_path("FloorPlan15"), encoding="utf-8"))
dp = os.path.join(tmp, "uses_draft.json"); json.dump(dict(d, version="k23-v1-draft"), open(dp, "w"))
orig_up = LP.uses_path
LP.uses_path = lambda kitchen="FloorPlan1": dp if kitchen == "FloorPlan15" else orig_up(kitchen)
try:
    check("a draft uses table is refused", any("uses table version" in p for p in E.package_problems("FloorPlan15")))
    json.dump(dict(d, version="k23-v1", kitchen="FloorPlan17"), open(dp, "w"))
    check("a uses table naming another kitchen is refused", any("names kitchen" in p for p in E.package_problems("FloorPlan15")))
finally:
    LP.uses_path = orig_up
check("a Kitchen 1 scene is not a replication kitchen", E.package_problems("FloorPlan1") != [])

# ---- 4 manifest injection (acceptance_manifest pattern)
orig_roster = K.a29_roster
K.a29_roster = lambda ns, root=None: ["g-fake-1", "g-fake-2"]
try:
    m = E.k23_manifest("test-k2-v1")
    check("test namespace: roster injected under a29_roster", m.get(C.A29_ROSTER) == ["g-fake-1", "g-fake-2"])
    check("load_manifest itself never carries the roster", C.A29_ROSTER not in C.load_manifest())
    check("practice / rehearsal namespaces: no roster key", C.A29_ROSTER not in E.k23_manifest("practice-k2-v1")
          and C.A29_ROSTER not in E.k23_manifest("rehearsal-k3-v1"))
finally:
    K.a29_roster = orig_roster
# hermetic: the real size decision exists once S2 has run (2026-10-21 00:15), so point the decision path at a missing file in tmp
orig_files = K.a29_files
K.a29_files = lambda scene, root=None: (orig_files(scene, root)[0], os.path.join(tmp, "no_size_decision.json"))
try:
    check("real roster without a size decision raises (never a default)", raises(lambda: E.k23_manifest("test-k2-v1"), RuntimeError))
finally:
    K.a29_files = orig_files

# ---- 5 practice_k admission + the scene-aware hint function (fake build_live, no simulator)
import practice_k as PK
check("practice_k untouched: Kitchen 1 practice namespaces only", tuple(PK.PRACTICE_NAMESPACES) == K1_PRACTICE)
import k23_practice as KP
seen = []
orig_bht = T.briefing_hint_table


def spy_bht(scene="FloorPlan1", **kw):
    seen.append(scene)
    return orig_bht(scene, **kw)


def fake_build_live(args):
    return "EX", "HOOK", "COUNTER", {}, "FP1-HINTS"


PK.build_live = fake_build_live
T.briefing_hint_table = spy_bht
try:
    check("a Kitchen 1 namespace cannot be admitted", raises(lambda: KP.admit("practice-v2"), ValueError))
    check("a K23 test namespace cannot be admitted as practice", raises(lambda: KP.admit("test-k2-v1"), ValueError))
    KP.admit("practice-k2-v1")
    check("practice-k2-v1 admitted in this process, Kitchen 1 entries kept",
          tuple(PK.PRACTICE_NAMESPACES) == K1_PRACTICE + ("practice-k2-v1",))
    KP.admit("practice-k2-v1")
    check("admission is idempotent", tuple(PK.PRACTICE_NAMESPACES).count("practice-k2-v1") == 1)
    class A: namespace = "practice-k2-v1"
    ex, hook, counter, params, hints = PK.build_live(A)
    check("live objects passed through unchanged", (ex, hook, counter) == ("EX", "HOOK", "COUNTER"))
    check("hook-B hints come from the namespace's kitchen briefing", seen and seen[-1] == "FloorPlan15" and callable(hints)
          and hints.table_sha == T.hint_table_sha(orig_bht("FloorPlan15", allow_unapproved=False)))
    check("the approved (not draft) briefing is used", T.allow_draft_briefing("practice-k2-v1") is False)
    class B: namespace = "practice-v2"
    check("another namespace through the same process keeps practice_k's own hints", PK.build_live(B)[4] == "FP1-HINTS")
    KP.admit("practice-k3-v1")                                     # Astra K23-TOOLING-G1 1: a second admission in the same process
    class C3: namespace = "practice-k3-v1"
    PK.build_live(C3)
    check("K2 admitted first, then K3: K3 gets ITS kitchen's hints", seen[-1] == "FloorPlan17")
    PK.build_live(A)
    check("... and K2 still gets FloorPlan15's", seen[-1] == "FloorPlan15")
    check("the wrapper is installed once", not getattr(getattr(PK.build_live, "k23_orig", None), "k23_scene_hints", False))
finally:
    T.briefing_hint_table = orig_bht

# ---- 6 readiness
r = KP.readiness("practice-k2-v1", [1], [5], perception=False)
check("orders other than [0] refused", any("order 0 only" in x for x in r))
check("K without 10 refused", any("K = 10" in x for x in r))
r = KP.readiness("practice-k2-v1", [0], [10], perception=False)
has17 = os.path.exists(K.a29_files("FloorPlan17")[0])
check("readiness reports the missing FloorPlan17 sample (A29 item 2) when absent", has17 or any("FloorPlan17" in x for x in r))
check("readiness never runs perception on request (Windows preflight)",
      not any(x.startswith("FloorPlan15: perception freeze record") or "frozen perception changed" in x for x in r))
# ---- 6b prereg A39.2: Kitchen 2 practice only after the COCO run + --add-coco + k23_coco_meta (fixture root)
import hashlib
cfx = tempfile.mkdtemp(prefix="k23coco_")
cd = os.path.join(cfx, "sim", "kitchen_v3", "FloorPlan15"); os.makedirs(cd)
recb = ("aa" * 32 + "  sim/weights/v3/FloorPlan15.yolo26s.pt\n").encode()
open(os.path.join(cd, "frozen_weights.sha256"), "wb").write(recb)
json.dump({"coco_pending": True, "test_evaluation_blocked": True, "n_files": 1, "record_sha256": hashlib.sha256(recb).hexdigest()},
          open(os.path.join(cd, "frozen_weights.meta.json"), "w"))
check("K2 practice refused while the record is coco_pending (A39.2)", any("A39.2" in x and "coco_pending" in x for x in E.order_problems("practice-k2-v1", cfx)))
check("K3 practice has no COCO condition", E.order_problems("practice-k3-v1", cfx) == [])
check("readiness carries the A39.2 refusal for K2", any("A39.2" in x for x in KP.readiness("practice-k2-v1", [0], [10], cfx, perception=False)))
recb2 = recb + ("bb" * 32 + "  sim/weights/v3/FloorPlan15.yolo26s_coco.pt\n").encode()
open(os.path.join(cd, "frozen_weights.sha256"), "wb").write(recb2)
json.dump({"coco_pending": False, "test_evaluation_blocked": False, "n_files": 1, "record_sha256": hashlib.sha256(recb).hexdigest()},
          open(os.path.join(cd, "frozen_weights.meta.json"), "w"))
check("--add-coco done but k23_coco_meta not run: still refused (stale successor metadata)",
      any("successor" in x for x in E.order_problems("practice-k2-v1", cfx)))
json.dump({"coco_pending": False, "test_evaluation_blocked": False, "n_files": 2, "record_sha256": hashlib.sha256(recb2).hexdigest(),
           "coco_append": {"coco": "sim/weights/v3/FloorPlan15.yolo26s_coco.pt"}}, open(os.path.join(cd, "frozen_weights.meta.json"), "w"))
check("COCO appended + successor metadata written: K2 practice allowed by the order rule", E.order_problems("practice-k2-v1", cfx) == [])
shutil.rmtree(cfx, ignore_errors=True)
check("readiness for FloorPlan17 lists its draft package while it is a draft",
      json.load(open(LP.uses_path("FloorPlan17"), encoding="utf-8")).get("version") == "k23-v1"
      or any("FloorPlan17: uses table version" in x for x in KP.readiness("practice-k3-v1", [0], [10], perception=False)))

# ---- 7 CLI refusals + dry run (fake build_live is never reached by a dry run)
check("--channels refused", quiet(KP.main, ["--namespace", "practice-k2-v1", "--channels", "places", "--dry-run"])[0] == KP.REFUSED)
check("--allow-test refused", quiet(KP.main, ["--namespace", "practice-k2-v1", "--allow-test", "--dry-run"])[0] == KP.REFUSED)
check("--kitchen refused", quiet(KP.main, ["--namespace", "practice-k2-v1", "--kitchen", "FloorPlan15", "--dry-run"])[0] == KP.REFUSED)
check("--gemma-only-counter refused", quiet(KP.main, ["--namespace", "practice-k2-v1", "--gemma-only-counter", "--dry-run"])[0] == KP.REFUSED)
for ab in (["--gemma"], ["--gemma-only"], ["--allow"], ["--kit", "FloorPlan15"], ["--chan", "places"], ["--ord", "1"], ["--gemma-only-counter=1"]):
    check(f"abbreviated / unlisted option {ab[0]} refused (Astra G1-r2 1)", quiet(KP.main, ["--namespace", "practice-k2-v1", "--dry-run"] + ab)[0] == KP.REFUSED)
check("the whitelist forwards --ollama / --limit / --steal-lock only", KP.passthrough(["--ollama", "http://h:1", "--limit", "2", "--steal-lock"])
      == ["--ollama", "http://h:1", "--limit", "2", "--steal-lock"] and raises(lambda: KP.passthrough(["--limit"]), ValueError))
check("a Kitchen 1 namespace is not a choice", quiet(KP.main, ["--namespace", "practice-v2", "--dry-run"])[0] == 2)
code, out, err = quiet(KP.main, ["--namespace", "practice-k2-v1", "--dry-run"])
last = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
check("dry run rc 0 with the seeded order printed", code == 0 and "order 0: 1:" in out)
check("dry run: 10 attempts of order 0, guard ok", '"guard": "ok"' in out and '"ks": [10]' in out and '"orders": [0]' in out)
check("dry run reports readiness + the kitchen's hint table", last.get("scene") == "FloorPlan15" and "live_blockers" in last and last.get("hint_table_sha"))
check("dry run carries the A39.2 order note for FloorPlan15", "A39.2" in str(last.get("a38_note")))
called = []
PK.main = lambda argv: called.append(argv) or 0
try:
    code, out, err = quiet(KP.main, ["--namespace", "practice-k2-v1", "--ollama", "http://x:1"])
    if KP.readiness("practice-k2-v1", [0], [10]):
        check("live run refused before practice_k when not ready", code == KP.REFUSED and not called and "REFUSED" in err)
    else:
        check("live run reaches practice_k with order 0 / K 10", called and called[0][:6] == ["--namespace", "practice-k2-v1", "--orders", "0", "--k", "10"])
finally:
    import importlib
    importlib.reload(PK)

# ---- 8 status in the kitchen's own ft-v2 tree
st, out, err = quiet(KP.status, "practice-k2-v1", [0], [10])
check("status reads the kitchen's own snapshot root", isinstance(st, dict) and st.get("snapshot_root") == "sim/final_test/snapshots_ft2/FloorPlan15")
check("snapshot root differs from Kitchen 1's", T.snapshot_root(scene="FloorPlan15") != T.snapshot_root())

# ---- 9 launcher files
LD = os.path.join(ROOT, "sim", "final_test", "k23_run", "practice")
for fn in ("run.sh", "inner.sh"):
    b = open(os.path.join(LD, fn), "rb").read()
    check(f"{fn}: LF only", b"\r\n" not in b)
run_sh, inner_sh = (open(os.path.join(LD, f), encoding="utf-8").read() for f in ("run.sh", "inner.sh"))
check("run.sh takes the ONE sim lock", "flock /home/arash/sim.lock" in run_sh)
check("run.sh accepts the two K23 practice namespaces only", "practice-k2-v1|practice-k3-v1)" in run_sh and "exit 2" in run_sh)
check("inner.sh: hang guard + k23_practice + the Windows Ollama host", "hang_guard.py sim/final_test/matrix/k23_practice.py" in inner_sh
      and '--ollama "http://$HOST:11434"' in inner_sh and "practice_k.py" not in inner_sh)
sys.path.insert(0, LD)
import launch as LCH
calls = []
LCH.subprocess.call = lambda *a, **k: calls.append(a) or 0
check("run.sh runs in the checkout launch.py chose (wsl --cd), not a hard-coded tree", "REPO=$(pwd)" in run_sh and "Master*" not in run_sh)
check("launch.py hands WSL its own checkout", "E.wsl_path(REPO)" in open(os.path.join(LD, "launch.py"), encoding="utf-8").read()
      and LCH.REPO == ROOT and E.wsl_path("D:/Personal/x y") == "/mnt/d/Personal/x y")
check("launcher refuses --gemma-only-counter", quiet(LCH.main, ["practice-k2-v1", "--gemma-only-counter"])[0] == 2 and not calls)
for ab in (["--gemma"], ["--gemma-only"], ["--allow"], ["--ollama", "http://x:1"], ["--steal-lock"]):
    check(f"launcher refuses {ab[0]} (abbreviation / set by inner.sh)", quiet(LCH.main, ["practice-k2-v1"] + ab)[0] == 2 and not calls)
check("launcher refuses a Kitchen 1 namespace", quiet(LCH.main, ["practice-v2"])[0] == 2 and not calls)
check("launcher refuses --orders / --k overrides", quiet(LCH.main, ["practice-k2-v1", "--k", "1,3,5,10"])[0] == 2 and not calls)
if KP.readiness("practice-k2-v1", [0], [10], perception=False):
    check("launcher refuses before WSL while not ready", quiet(LCH.main, ["practice-k2-v1"])[0] == 2 and not calls)

shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
