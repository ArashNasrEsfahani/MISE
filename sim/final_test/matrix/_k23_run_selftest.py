"""Offline gate for the K23 orchestrator (k23_run.py) and the rehearsal / test launchers (sim/final_test/k23_run/). No simulator, no
Ollama, no provider: nothing here reaches a live phase (capture runs only as its own --dry-run; plan objects are built, never run)."""
import contextlib, io, json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import k23_ns as K
import k23_entry as E
import k23_run as R
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def raises(f, exc=Exception, text=None):
    try:
        f()
    except exc as e:
        return text is None or text in str(e)
    return False


def quiet(f, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            r = f(*a, **kw)
        except SystemExit as e:
            r = e.code
    return r, out.getvalue(), err.getvalue()


ROOT = C.paths()["root"]
K1_DEV = ("rehearsal-v1", "rehearsal-v2", "rehearsal-v3")
K1_NSI = {"dev": "rehearsal-v1", "test": "test-v1"}

# ---- 1 CLI refusals (preflight before any phase module)
check("practice namespaces are not k23_run namespaces", quiet(R.main, ["--phase", "plan", "--namespace", "practice-k2-v1"])[0] == R.EXIT["usage"])
check("Kitchen 1 namespaces are not k23_run namespaces", quiet(R.main, ["--phase", "plan", "--namespace", "test-v1"])[0] == R.EXIT["usage"])
r, o, e = quiet(R.main, ["--phase", "capture", "--namespace", "test-k2-v1", "--dry-run"])
check("test without --allow-test refused", r == R.EXIT["refused"] and "--allow-test" in e)
r, o, e = quiet(R.main, ["--phase", "capture", "--namespace", "rehearsal-k3-v1", "--allow-test", "--dry-run"])
check("rehearsal with --allow-test refused", r == R.EXIT["refused"] and "no test items" in e)
r, o, e = quiet(R.main, ["--phase", "capture", "--namespace", "rehearsal-k3-v1", "--dry-run", "--", "--sources", "none"])
check("capture: a design option (--sources) refused", r == R.EXIT["refused"] and "--sources" in e)
for opt in ("--worlds", "--runs", "--reloc-seed", "--channels", "--reloc-all"):
    check(f"capture: {opt} refused", raises(lambda o=opt: R.capture_argv("rehearsal-k2-v1", [o, "x"], allow_test=False, steal_lock=False,
                                                                         dry_run=True, interrupt=None), R.Refused))
check("capture: an option without its value refused", raises(lambda: R.capture_argv("rehearsal-k2-v1", ["--ollama"], allow_test=False,
                                                                                     steal_lock=False, dry_run=True, interrupt=None), R.Refused))
# hermetic: the real size decision (S2 / S3) and the K2 freeze exist since 2026-10-21, so the decision path points at a missing file in tmp
_orig_files = K.a29_files
_nodec_dir = tempfile.mkdtemp(prefix="k23run_nodec_")
K.a29_files = lambda scene, root=None: (_orig_files(scene, root)[0], os.path.join(_nodec_dir, "no_size_decision.json"))
try:
    r, o, e = quiet(R.main, ["--phase", "capture", "--namespace", "test-k2-v1", "--allow-test", "--dry-run"])
finally:
    K.a29_files = _orig_files
check("test capture refused without the recorded A29 size decision (never a default roster)",
      r == R.EXIT["refused"] and ("size decision" in e or "freeze" in e))

# ---- 2 argv construction (the K23 design is forced)
a = R.capture_argv("rehearsal-k2-v1", ["--ollama", "http://h:11434"], allow_test=False, steal_lock=True, dry_run=False, interrupt=None)
check("rehearsal capture argv", a[:4] == ["--namespace", "rehearsal-k2-v1", "--items", "dev"] and all(x in a for x in R.CAPTURE_FORCED)
      and "--steal-lock" in a and a[-2:] == ["--ollama", "http://h:11434"])
check("capture forces canon, run 1, no FS, no curve", R.CAPTURE_FORCED == ["--runs", "1", "--worlds", "canon", "--no-fs", "--no-curve"])
check("live capture needs --ollama", raises(lambda: R.capture_argv("rehearsal-k2-v1", [], allow_test=False, steal_lock=False, dry_run=False,
                                                                   interrupt=None), R.Refused, "--ollama"))
a = R.capture_argv("test-k3-v1", ["--ollama", "http://h:1"], allow_test=True, steal_lock=False, dry_run=False, interrupt=None)
check("test capture argv: --items test + --allow-test", a[:4] == ["--namespace", "test-k3-v1", "--items", "test"] and "--allow-test" in a)
for opt in ("--allow-missing-anthropic", "--limit"):
    args = [opt] + (["3"] if opt == "--limit" else []) + ["--ollama", "http://h:1"]
    check(f"test capture: {opt} refused", raises(lambda a_=args: R.capture_argv("test-k2-v1", a_, allow_test=True, steal_lock=False,
                                                                                 dry_run=False, interrupt=None), R.Refused))
x = R.execute_argv("rehearsal-k2-v1", [], allow_test=False, steal_lock=False, dry_run=False, interrupt=300, partial=False)
check("rehearsal execute argv: --items dev, canon, no /p replay, 300-execution sessions", x[:4] == ["--namespace", "rehearsal-k2-v1", "--items", "dev"]
      and all(v in x for v in R.EXEC_FORCED) and x[x.index("--inject-interrupt") + 1] == "300")
x = R.execute_argv("test-k2-v1", [], allow_test=True, steal_lock=False, dry_run=False, interrupt=None, partial=False)
check("test execute argv: no --items (execute.py ties --items test to test-v1), --allow-test", "--items" not in x and "--allow-test" in x
      and "--no-replay" in x)
x = R.execute_argv("rehearsal-k2-v1", ["--g-iso"], allow_test=False, steal_lock=False, dry_run=False, interrupt=None, partial=False)
check("rehearsal G-iso (12.6) allowed with the registered n = 12", "--g-iso" in x and x[x.index("--g-iso-n") + 1] == "12")
r, o, e = quiet(R.main, ["--phase", "context", "--namespace", "test-k2-v1", "--allow-test"])
check("the context-fit phase is for the rehearsal capture only", r == R.EXIT["refused"])
for opt in ("--g-iso", "--p-arms", "--reloc-seed", "--worlds"):
    check(f"execute: {opt} refused", raises(lambda o=opt: R.execute_argv("test-k2-v1", [o, "x"], allow_test=True, steal_lock=False,
                                                                          dry_run=False, interrupt=None, partial=False), R.Refused))

# ---- 3 capture admission (fg_runner pattern) + a real capture dry run (no simulator)
CP = R.admit_capture("rehearsal-k2-v1")
check("rehearsal admitted to capture.DEV_NAMESPACES in this process", tuple(CP.DEV_NAMESPACES) == K1_DEV + ("rehearsal-k2-v1",))
R.admit_capture("rehearsal-k2-v1")
check("admission idempotent", list(CP.DEV_NAMESPACES).count("rehearsal-k2-v1") == 1)
R.admit_capture("test-k2-v1")
check("test admitted through NS_FOR_ITEMS['test'] only", CP.NS_FOR_ITEMS == {"dev": "rehearsal-v1", "test": "test-k2-v1"})
r, o, e = quiet(CP.main, R.capture_argv("rehearsal-k2-v1", [], allow_test=False, steal_lock=False, dry_run=True, interrupt=None))
try:
    d = json.loads(o)
except ValueError:
    d = {}
check("capture dry run through the admission: 20 presentations, guard ok", d.get("presentations") == 20 and d.get("guard", {}).get("n_test") == 0)
check("capture dry run: the only memory source is S* (no FS / curve in Kitchens 2 / 3)", d.get("sources") == ["S*"] and d.get("reloc_items") == 0)
check("capture dry run: rc 2 exactly when the kitchen's S* snapshot is missing", (r == 2) == bool(d.get("missing_snapshots")))
out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import capture as c; print(c.DEV_NAMESPACES, c.NS_FOR_ITEMS)" % HERE],
                     capture_output=True, text=True, timeout=300).stdout.strip()
check("a fresh process keeps capture's Kitchen 1 tuples (nothing written)", out == f"{K1_DEV} {K1_NSI}")

# ---- 4 the kitchen's type list (fixture root)
fx = tempfile.mkdtemp(prefix="k23run_")
os.makedirs(os.path.join(fx, "docs", "final-test", "matrix")); os.makedirs(os.path.join(fx, "sim", "kitchen_v3", "FloorPlan15"))
real_types = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types_FloorPlan15.json"), encoding="utf-8"))
json.dump(real_types, open(os.path.join(fx, "docs", "final-test", "matrix", "detector_types_FloorPlan15.json"), "w"))
rec = os.path.join(fx, "sim", "kitchen_v3", "FloorPlan15", "frozen_weights.sha256")
open(rec, "w").write(f"{real_types['weights_sha']}  sim/weights/v3/FloorPlan15.yolo26s.pt\n")
import plan as PL
L = R.k23_load_types("rehearsal-k2-v1")
t, src = L(None, fx)
check("rehearsal: the B5 list bound to the perception freeze loads", t == real_types["types"] and src == "detector_types_FloorPlan15.json")
check("another path than the kitchen's list is refused", raises(lambda: L(os.path.join(fx, "x.json"), fx), PL.PlanAbort))
open(rec, "w").write("0" * 64 + "  sim/weights/v3/FloorPlan15.yolo26s.pt\n")
check("a weights sha that is not the perception freeze's is refused", raises(lambda: L(None, fx), PL.PlanAbort, "perception freeze"))
open(rec, "w").write(f"{real_types['weights_sha']}  sim/weights/v3/FloorPlan15.yolo26s.pt\n")
json.dump(dict(real_types, scene="FloorPlan17"), open(os.path.join(fx, "docs", "final-test", "matrix", "detector_types_FloorPlan15.json"), "w"))
check("a list naming another kitchen is refused", raises(lambda: L(None, fx), PL.PlanAbort, "scene"))
json.dump(real_types, open(os.path.join(fx, "docs", "final-test", "matrix", "detector_types_FloorPlan15.json"), "w"))
LT = R.k23_load_types("test-k2-v1")
check("test: no agent freeze -> refused", raises(lambda: LT(None, fx), PL.PlanAbort, "agent freeze"))
os.makedirs(os.path.join(fx, "docs", "final-test", "k23"))
json.dump({"detector_types_sha256": C.sha256_json(real_types["types"])}, open(K.freeze_path("FloorPlan15", fx), "w"))
check("test: the agent freeze's type sha matches -> loads", LT(None, fx)[0] == real_types["types"])
json.dump({"detector_types_sha256": "x"}, open(K.freeze_path("FloorPlan15", fx), "w"))
check("test: a different agent-freeze type sha -> refused", raises(lambda: LT(None, fx), PL.PlanAbort, "agent freeze"))
check("Kitchen 1's plan.load_types is untouched by building the K23 one", PL.load_types.__module__ == "plan" and not hasattr(PL.load_types, "k23_ns"))

# ---- 5 plan factory: roster manifest, types, allow_test kept for the completion record (planfirst.py:135)
orig_roster = K.a29_roster
K.a29_roster = lambda ns, root=None: ["g-fake-a", "g-fake-b"]
try:
    mk = R.k23_plan_factory("test-k2-v1", True, R.types_path("FloorPlan15"))
    p = mk("test-k2-v1", passes=["gemma"], root=None, pass_id="", out=lambda *_: None, allow_test=False)   # exactly planfirst._complete's call
    check("planfirst._complete's allow_test=False is kept True for a test runner", p.allow_test is True)
    check("the kitchen's type list is the plan's types_path", p.types_path == R.types_path("FloorPlan15"))
    check("the plan manifest carries the A29 roster", p._manifest().get(C.A29_ROSTER) == ["g-fake-a", "g-fake-b"])
    check("the K23 plan is a plan.Plan (same code path)", isinstance(p, PL.Plan) and type(p).__mro__[1] is PL.Plan)
    check("the factory is bound to its namespace", raises(lambda: mk("test-k3-v1"), ValueError))
    mk2 = R.k23_plan_factory("rehearsal-k2-v1", False, R.types_path("FloorPlan15"))
    p2 = mk2("rehearsal-k2-v1", passes=["gemma"], allow_test=False)
    check("rehearsal: allow_test stays False, no roster key", p2.allow_test is False and C.A29_ROSTER not in p2._manifest())
finally:
    K.a29_roster = orig_roster
check("K23 stages = free + core0 (Gemma, then synchronous Opus r = 0)", R.K23_STAGES == ("free", "core0"))

# ---- 6 plan refusals (nothing planned)
class A:  # minimal argparse namespace
    energy = None; ack_pause = None; pause_response = None; outcome_blind = False; allow_test = True; parallel = 4
    ollama = "http://127.0.0.1:11434"; allow_replacement = False; requeue_infra = False; steal_lock = False; inject_interrupt = None
    no_gpu_sample = True
A.energy = "off"; A.energy_waiver = None
check("test plan: --energy off refused without Arash's waiver (C5 mandatory, test-v1 parity)", raises(lambda: R.run_plan("test-k2-v1", A, fx), R.Refused, "waive"))
A.energy = None; A.ack_pause = "spend evidence only, short"
check("test pause acknowledgement needs --pause-response + --outcome-blind", raises(lambda: R.run_plan("test-k2-v1", A, fx), R.Refused, "pause"))
r, o, e = quiet(R.main, ["--phase", "plan", "--namespace", "rehearsal-k2-v1", "--", "--batch"])
check("plan takes no pass-through options (no --batch: synchronous Opus)", r == R.EXIT["refused"])

# ---- 6b energy gate (test-v1 parity, Astra K23-TOOLING-G2 1), recovery dispatch, the WSL command (G2 3)
import energy as EN
orig_cov = EN.coverage
try:
    check("energy gate: not applied to a rehearsal", R.energy_gate("execute", "rehearsal-k2-v1", fx) is None)
    check("energy gate: not applied to a test plan / capture", R.energy_gate("capture", "test-k2-v1", fx) is None)
    EN.coverage = lambda ns, root=None, pj=None: {"complete": False, "gemma_plan_rows": 400, "measured": 390}
    check("energy gate: incomplete coverage refuses test execute", raises(lambda: R.energy_gate("execute", "test-k2-v1", fx), R.Refused, "energy"))
    check("energy gate: incomplete coverage refuses test score", raises(lambda: R.energy_gate("score", "test-k3-v1", fx), R.Refused, "energy"))
    g = R.energy_gate("execute", "test-k2-v1", fx, waiver="Arash: proceed, 10 rows unmeasured (meter dropout)")
    wv = os.path.join(C.ns_docs("test-k2-v1", fx), "energy_waivers.jsonl")
    check("energy gate: Arash's waiver is recorded and lets execute through", g.get("waiver") and os.path.exists(wv)
          and json.loads(open(wv, encoding="utf-8").read().splitlines()[-1])["by"] == "Arash")
    EN.coverage = lambda ns, root=None, pj=None: {"complete": True, "gemma_plan_rows": 400, "measured": 400}
    check("energy gate: complete coverage passes", R.energy_gate("execute", "test-k2-v1", fx).get("complete") is True)
finally:
    EN.coverage = orig_cov
import matrix_run as MR
o_dep, o_inv, o_ord = MR.dependency_state, MR.open_invalid, E.order_problems
E.order_problems = lambda ns, root=None: []                  # the recovery checks below are about the invalid rule only
MR.dependency_state = lambda ns, phase, root=None, partial=False: {}
MR.open_invalid = lambda ns, root=None: True
try:
    check("an unresolved invalid abort blocks a normal execute", raises(lambda: R.preflight("execute", "rehearsal-k2-v1", root=None), R.Refused, "invalid"))
    check("... but not the explicit recovery dispatch (--resolve-invalid / --declare-unrecoverable)",
          R.preflight("execute", "rehearsal-k2-v1", root=None, recovery=True)["phase"] == "execute")
    check("... and still blocks score", raises(lambda: R.preflight("score", "rehearsal-k2-v1", root=None, recovery=True), R.Refused, "invalid"))
finally:
    MR.dependency_state, MR.open_invalid, E.order_problems = o_dep, o_inv, o_ord
w = R.wsl_command(["--phase", "capture", "--namespace", "rehearsal-k2-v1", "--steal-lock", "--inject-interrupt", "5", "--", "--ollama", "http://h:1"])
check("WSL command: this checkout, every flag kept", w[w.index("--cd") + 1] == E.wsl_path(ROOT) and "--steal-lock" in w and "--inject-interrupt" in w
      and w[-3:] == ["--", "--ollama", "http://h:1"])
r, o, e = quiet(R.main, ["--phase", "capture", "--namespace", "rehearsal-k3-v1", "--dry-run", "--steal-lock", "--", "--ollama", "http://h:1"])
try:
    wl = json.loads(o)["wsl"]
except (ValueError, KeyError):
    wl = []
check("the printed WSL fallback keeps --dry-run / --steal-lock and the pass-through", "--dry-run" in wl and "--steal-lock" in wl and wl[-2:] == ["--ollama", "http://h:1"])

# ---- 7 K23 test entry checks (fixture root)
import treatment as T
os.makedirs(os.path.join(fx, "sim", "dataset"))
shutil.copy(os.path.join(ROOT, "sim", "dataset", "general.json"), os.path.join(fx, "sim", "dataset", "general.json"))
shutil.copy(C.paths()["split"], C.paths(fx)["split"])
for f in os.listdir(os.path.dirname(C.paths()["split"])):
    if f.startswith("inclusion_manifest"):
        shutil.copy(os.path.join(os.path.dirname(C.paths()["split"]), f), os.path.join(fx, "docs", "final-test", f))
ds = C.sha256_file(C.paths(fx)["dataset"])
json.dump({"treatment_id": "ID1", "dataset_sha256": ds}, open(K.freeze_path("FloorPlan15", fx), "w"))
check("test entry refused when the agent freeze does not bind the FA-examples decision (Astra K23-FA-EXAMPLES-r1)",
      any("K23-FA-EXAMPLES" in p for p in E.test_entry_problems("test-k2-v1", fx, ident_sha="ID1")))
json.dump({"treatment_id": "ID1", "dataset_sha256": ds, "code": {"docs/final-test/k23/decisions/K23-FA-EXAMPLES.json": "x"}},
          open(K.freeze_path("FloorPlan15", fx), "w"))
meta = os.path.join(fx, "sim", "kitchen_v3", "FloorPlan15", "frozen_weights.meta.json")
json.dump({"coco_pending": True, "test_evaluation_blocked": True}, open(meta, "w"))
try:
    probs = E.test_entry_problems("test-k2-v1", fx, ident_sha="ID1")
    check("identity + dataset match; COCO still pending -> refused", any("COCO" in p for p in probs) and not any("identity" in p for p in probs))
    probs = E.test_entry_problems("test-k2-v1", fx, ident_sha="ID2")
    check("a different running identity -> refused", any("identity" in p for p in probs))
    json.dump({"coco_pending": False, "test_evaluation_blocked": False}, open(meta, "w"))
    open(rec, "a").write("ab" * 32 + "  sim/weights/v3/FloorPlan15.yolo26s_coco.pt\n")
    check("COCO appended but no successor metadata (k23_coco_meta not run) -> refused",
          any("successor" in p for p in E.test_entry_problems("test-k2-v1", fx, ident_sha="ID1")))
    import hashlib
    rb = open(rec, "rb").read()
    json.dump({"coco_pending": False, "test_evaluation_blocked": False, "record_sha256": hashlib.sha256(rb).hexdigest(),
               "n_files": sum(1 for ln in rb.split(b"\n") if ln.strip()), "coco_append": {"coco": "sim/weights/v3/FloorPlan15.yolo26s_coco.pt"}},
              open(meta, "w"))
    check("COCO appended + successor metadata + identity + dataset -> []", E.test_entry_problems("test-k2-v1", fx, ident_sha="ID1") == [])
    check("A39.2: rehearsal-k2-v1 allowed by the order rule once COCO + metadata are done; K3 rehearsal never conditioned",
          E.order_problems("rehearsal-k2-v1", fx) == [] and E.order_problems("rehearsal-k3-v1", fx) == [])
    json.dump({"coco_pending": True, "test_evaluation_blocked": True}, open(meta, "w"))
    check("A39.2: the rehearsal preflight refuses rehearsal-k2-v1 while COCO is pending (before any other check)",
          raises(lambda: R.preflight("capture", "rehearsal-k2-v1", root=fx), R.Refused, "A39.2"))
    import time as _t
    jl_ev = C.journal_path("rehearsal-k2-v1", "events", fx)
    os.makedirs(os.path.dirname(jl_ev), exist_ok=True)
    open(jl_ev, "w").write(json.dumps({"event": "capture_start", "ts": "2026-10-20T20:00:00"}) + "\n")
    check("A39.2 evidence: a capture_start without a prior order event is detected (direct execute.py / bypass)",
          any("A39.2" in p for p in E.order_evidence_problems("rehearsal-k2-v1", "capture_start", fx)))
    open(jl_ev, "w").write("")
    E.record_order_ok("rehearsal-k2-v1", "capture", fx)
    _t.sleep(1.1)
    open(jl_ev, "a").write(json.dumps({"event": "capture_start", "ts": _t.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    ev_rows = [json.loads(x) for x in open(jl_ev, encoding="utf-8") if x.strip()]
    check("A39.2 evidence: the wrapper's order event (with the record / metadata shas) before the first capture_start -> []",
          ev_rows[0]["event"] == "k23_order_ok" and ev_rows[0].get("meta_sha256") and E.order_evidence_problems("rehearsal-k2-v1", "capture_start", fx) == [])
    E.record_order_ok("rehearsal-k3-v1", "capture", fx)
    check("A39.2 evidence: nothing recorded for Kitchen 3", not os.path.exists(C.journal_path("rehearsal-k3-v1", "events", fx)))
    check("A39.2: ... and the context phase refuses it too", quiet(R.main, ["--phase", "context", "--namespace", "rehearsal-k2-v1"], root=fx)[0] == R.EXIT["refused"])
    try:
        R.preflight("capture", "rehearsal-k3-v1", root=fx); _msg = ""
    except Exception as _e:                               # noqa: BLE001  (later fixture checks may refuse; only the A39.2 rule matters here)
        _msg = str(_e)
    check("A39.2: rehearsal-k3-v1 is not stopped by the FloorPlan15 COCO rule", "A39.2" not in _msg)
    json.dump({"treatment_id": "ID1", "dataset_sha256": "x", "code": {"docs/final-test/k23/decisions/K23-FA-EXAMPLES.json": "x"}}, open(K.freeze_path("FloorPlan15", fx), "w"))
    check("a dataset other than the frozen one -> refused", any("dataset" in p for p in E.test_entry_problems("test-k2-v1", fx, ident_sha="ID1")))
    check("FloorPlan17 has no COCO condition", not any("COCO" in p for p in E.test_entry_problems("test-k3-v1", fx, ident_sha="ID1")))
except Exception as ex:                                   # noqa: BLE001
    check(f"test entry checks ran ({type(ex).__name__}: {ex})", False)

# ---- 7b the perception freeze reader on either platform (the record lists ~/thor-data files by WSL paths)
check("host_path: a repo-relative entry joins the root", E.host_path("sim/x.pt", fx) == os.path.join(fx, "sim/x.pt"))
check("host_path: a WSL path is read through the WSL share on Windows, unchanged in WSL",
      E.host_path("/home/arash/thor-data/a.txt", fx) == ((E.WSL_SHARE + "/home/arash/thor-data/a.txt") if os.name == "nt" else "/home/arash/thor-data/a.txt"))
open(os.path.join(fx, "sim", "kitchen_v3", "FloorPlan15", "frozen_weights.sha256"), "w").write(
    f"{C.sha256_file(C.paths(fx)['dataset'])}  sim/dataset/general.json\n")
check("perception reader: a matching record -> []", E.perception_problems("FloorPlan15", fx) == [])
open(os.path.join(fx, "sim", "kitchen_v3", "FloorPlan15", "frozen_weights.sha256"), "a").write("0" * 64 + "  sim/dataset/general.json\n")
check("perception reader: a changed file -> refused", any("changed" in p for p in E.perception_problems("FloorPlan15", fx)))
check("perception reader: no record -> refused", any("missing" in p for p in E.perception_problems("FloorPlan17", fx)))
orig_pp = K.perception_problems
E.install_perception_reader()
check("install_perception_reader: k23_ns uses it in this process (replication_freeze_status)", K.perception_problems is E.perception_problems)
K.perception_problems = orig_pp
check("wsl_path: Windows drive paths and POSIX paths", E.wsl_path("D:/Personal/x y") == "/mnt/d/Personal/x y"
      and E.wsl_path("D:" + chr(92) + "P" + chr(92) + "a") == "/mnt/d/P/a" and E.wsl_path("/mnt/d/x") == "/mnt/d/x")

# ---- 8 launcher / chain files
D = os.path.join(ROOT, "sim", "final_test", "k23_run")
for fn in ("run_k23.sh", "chain_k23.sh"):
    check(f"{fn}: LF only", b"\r\n" not in open(os.path.join(D, fn), "rb").read())
run_sh = open(os.path.join(D, "run_k23.sh"), encoding="utf-8").read()
chain = open(os.path.join(D, "chain_k23.sh"), encoding="utf-8").read()
check("run_k23.sh: ONE sim lock + hang guard around k23_run.py", "flock /home/arash/sim.lock" in run_sh
      and "hang_guard.py sim/final_test/matrix/k23_run.py" in run_sh and "matrix_run.py" not in run_sh)
check("run_k23.sh: capture gets the Windows Ollama host after the pass-through separator", '--ollama "http://$HOST:11434"' in run_sh and "exit 12" in run_sh)
check("chain: test energy on by default, off only with Arash's ENERGY_WAIVER", "ENERGY=${ENERGY:-on}" in chain and "ENERGY_WAIVER" in chain
      and "--energy-waiver" in chain)
check("chain: the energy waiver reaches execute and score too (Astra G3 1)", '"${EW[@]}" --inject-interrupt 300' in chain
      and '"${T_FLAG[@]}" "${EW[@]}"; }' in chain)
check("chain: fresh simulator per 300 executions + score + study-wide / A37 check", "--inject-interrupt 300" in chain and "study_wide_problems" in chain
      and "a37_hold_problems" in chain)
check("chain: K23 namespaces only", "rehearsal-k2-v1|rehearsal-k3-v1)" in chain and "test-k2-v1|test-k3-v1)" in chain)
sys.path.insert(0, D)
src = open(os.path.join(D, "launch_k23.py"), encoding="utf-8").read()
check("launch_k23.py: key forwarded for capture only", "wsl_key_env(os.environ) if phase == \"capture\"" in src)
check("run_k23.sh / launch_k23.py / chain_k23.sh bind their own checkout", "REPO=$(pwd)" in run_sh and "Master*" not in run_sh
      and "E.wsl_path(REPO)" in src and "BASH_SOURCE" in chain and "Master's Project" not in chain)

shutil.rmtree(fx, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
