"""Offline gate: K23 scene propagation, cross-kitchen rejection, resource isolation and unchanged Kitchen 1 defaults
(Astra K23-AGENT-r1 items 1 + 4, K23-FREEZE-SUPERSEDE-r1 "explicit kitchen propagation, correct map selection, rejection of
missing / cross-kitchen maps, and unchanged FP1 defaults"). Part of the FP1 equivalence evidence (docs/final-test/k23/FP1-equivalence-plan.md).

    python sim/final_test/matrix/_k23_scene_isolation_selftest.py

Hermetic: Windows python, no simulator, no WSL, no provider, no network. Writes only into a temp directory (the frozen snapshot is opened
read-only for its manifest check). Lives under final_test/matrix (not sim/_*selftest.py) so the frozen K1 gate manifest is unchanged.
"""
import ast, inspect, json, os, shutil, subprocess, sys, tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402  (also puts sim/ on sys.path)
import treatment as T                                    # noqa: E402
import k23_ns as K                                       # noqa: E402

SIM = C.SIM
ok = fail = 0


def check(name, cond, detail=None):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print("FAIL", name, "" if detail is None else f"-- {str(detail)[:300]}")


def raises(f, exc=Exception, text=None):
    try:
        f()
    except exc as e:
        return text is None or text in str(e)
    return False


def src(rel):
    with open(os.path.join(SIM, *rel.split("/")), encoding="utf-8") as fh:
        return fh.read()


def calls_with_kw(rel, func_name, kw):
    """[(lineno, ast.unparse(kw value))] for every call of `func_name` (attribute or name) in the module that passes keyword `kw`."""
    out = []
    for node in ast.walk(ast.parse(src(rel))):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == func_name:
                for k in node.keywords:
                    if k.arg == kw:
                        out.append((node.lineno, ast.unparse(k.value)))
    return out


K1 = ("rehearsal-v1", "rehearsal-v2", "practice-v1", "test-v1", "practice-v2", "rehearsal-v3", "acceptance-v2", "acceptance-v2-a2",
      "freeze-gates-ft2")
REP = tuple(n for n in C.NAMESPACES if C.is_replication(n))
SCENE_OF = {"k2": "FloorPlan15", "k3": "FloorPlan17"}
SOURCES = ("S*", "o0k0", "o0k1", "o0k3", "o0k5", "o0k10")
tmp = tempfile.mkdtemp(prefix="k23iso_")

# ===================================================================================================== 1 scene propagation
print("== 1 scene propagation (namespace -> executor / skills / map / packager / params / snapshots / requests)")
check("1.1 every K1 namespace resolves to FloorPlan1 (no scene key)", all(C.ns_scene(n) == "FloorPlan1" and "scene" not in C.NAMESPACES[n] for n in K1))
check("1.2 every K2 / K3 namespace resolves to its own kitchen",
      all(C.ns_scene(n) == SCENE_OF[n.split("-")[1]] for n in REP) and len(REP) == 6, REP)
import execute as X                                      # noqa: E402
import step5_live as sl                                  # noqa: E402
_LE = sl.LiveExecutor
seen = []


class _FakeLE:
    def __init__(self, **kw):
        seen.append(kw)


try:
    sl.LiveExecutor = _FakeLE
    for ns in ("test-v1", "test-k2-v1", "test-k3-v1", "freeze-gates-ft2"):
        X.live_executor_factory(SimpleNamespace(ns=ns))
finally:
    sl.LiveExecutor = _LE
check("1.3 execute.live_executor_factory builds the executor on the namespace's scene (FP1 / FP15 / FP17 / FP1)",
      [k.get("scene") for k in seen] == ["FloorPlan1", "FloorPlan15", "FloorPlan17", "FloorPlan1"], [k.get("scene") for k in seen])
check("1.4 ... with the registered K1 settings unchanged (memory places, console policy, 900 s wall, plan after sweep, state capture)",
      all(k.get("memory") == "places" and k.get("policy") == "console" and k.get("timeout") == C.CAPS["episode_wall_s"] == 900
          and k.get("plan_after_sweep") and k.get("state_capture") for k in seen))
check("1.5 execute: the request kitchen, memory opener and snapshot sha all route through the namespace's scene",
      calls_with_kw("final_test/matrix/execute.py", "EpisodeRequest", "kitchen") == [(calls_with_kw("final_test/matrix/execute.py", "EpisodeRequest", "kitchen")[0][0], "C.ns_scene(self.ns)")]
      and any(v == "C.ns_scene(phase.ns)" for _l, v in calls_with_kw("final_test/matrix/execute.py", "open_frozen", "kitchen")),
      calls_with_kw("final_test/matrix/execute.py", "EpisodeRequest", "kitchen"))
check("1.6 capture: executor scene, capturer scene and frozen-snapshot kitchen = the namespace's scene",
      [v for _l, v in calls_with_kw("final_test/matrix/capture.py", "LiveExecutor", "scene")] == ["C.ns_scene(a.namespace)"]
      and [v for _l, v in calls_with_kw("final_test/matrix/capture.py", "Capturer", "scene")] == ["C.ns_scene(a.namespace)"]
      and [v for _l, v in calls_with_kw("final_test/matrix/capture.py", "open_frozen", "kitchen")] == ["C.ns_scene(ns)"])
check("1.7 practice: executor scene = the namespace's scene; snapshot root per scene",
      [v for _l, v in calls_with_kw("final_test/matrix/practice_k.py", "LiveExecutor", "scene")] == ["C.ns_scene(ns)"]
      and "T.snapshot_root(root, scene=C.ns_scene(namespace))" in src("final_test/matrix/practice_k.py"))
esrc = inspect.getsource(sl.LiveExecutor.execute)
check("1.8 LiveExecutor hands its scene to HonestSkills (look-alike map kitchen) and loads the navigation map of its scene ONCE, scene-checked "
      "before the simulator is touched (before _boot), and passes that object on",
      [v for _l, v in calls_with_kw("step5_live.py", "HonestSkills", "kitchen")] == ["self.scene"]
      and esrc.index("nav = (self._gm_cls or _GridMap).load(self.scene)") < esrc.index("scene_problems(self.scene, req, nav)") < esrc.index("self._boot()")
      and "HonestSkills(h, nav, self._proxy" in esrc and esrc.count(".load(self.scene)") == 1)
import honest_skills as hs                               # noqa: E402
check("1.9 HonestSkills loads the frozen look-alike map of ITS kitchen", "load_frozen_map(kitchen=kitchen)" in src("honest_skills.py"))
pk = {ns: T.Packager(ns, None) for ns in ("test-v1", "practice-v2", "practice-k2-v1", "test-k2-v1", "test-k3-v1")}
check("1.10 Packager kitchen = the namespace's scene", {n: p.scene for n, p in pk.items()} ==
      {"test-v1": "FloorPlan1", "practice-v2": "FloorPlan1", "practice-k2-v1": "FloorPlan15", "test-k2-v1": "FloorPlan15", "test-k3-v1": "FloorPlan17"})
check("1.11 Packager test side: K1 test-v1 and the replication TEST namespaces only",
      {n: p.test_side for n, p in pk.items()} == {"test-v1": True, "practice-v2": False, "practice-k2-v1": False, "test-k2-v1": True, "test-k3-v1": True})
import ltm_package as LP                                 # noqa: E402
e1, e15, e17 = (T.effective_params("pkg-1.r6b", kitchen=k) for k in ("FloorPlan1", "FloorPlan15", "FloorPlan17"))
check("1.12 effective params hash the kitchen's own uses table / inventory (three kitchens, three hashes)",
      len({e1["sha256"], e15["sha256"], e17["sha256"]}) == 3)
check("1.13 identity file list per scene: the FloorPlan15 list names FloorPlan15 weights + calibration only",
      all("FloorPlan1." not in f and "FloorPlan1/" not in f for f in T.identity_files("FloorPlan15"))
      and sum("FloorPlan15" in f for f in T.identity_files("FloorPlan15")) == 10)

# ===================================================================================================== 2 cross-kitchen rejection
print("== 2 cross-kitchen rejection (map, navigation map, request, memory, snapshot, packager, CLI)")
fp1_doc = json.load(open(os.path.join(SIM, "step5", "lookalike_map_FloorPlan1.json"), encoding="utf-8"))
p_bad = os.path.join(tmp, "lookalike_map_FloorPlan15.json")
json.dump(fp1_doc, open(p_bad, "w", encoding="utf-8"))                      # Kitchen 1's map under Kitchen 2's name
check("2.1 a look-alike map whose recorded kitchen is another kitchen is refused",
      raises(lambda: hs.load_frozen_map(p_bad, kitchen="FloorPlan15"), RuntimeError, "is for kitchen 'FloorPlan1'"))
p_nok = os.path.join(tmp, "nokitchen.json")
json.dump({k: v for k, v in fp1_doc.items() if k != "kitchen"}, open(p_nok, "w", encoding="utf-8"))
check("2.2 a look-alike map that names no kitchen is refused", raises(lambda: hs.load_frozen_map(p_nok, kitchen="FloorPlan1"), RuntimeError))
check("2.3 a kitchen with no frozen look-alike map is refused (no silent FloorPlan1 fallback, no empty map)",
      raises(lambda: hs.load_frozen_map(kitchen="FloorPlanNone"), OSError))
_tmpl = hs.LOOKALIKE_MAP_FILE
try:
    hs.LOOKALIKE_MAP_FILE = os.path.join(tmp, "lookalike_map_{kitchen}.json")
    good = dict(fp1_doc, kitchen="FloorPlan15")
    json.dump(good, open(os.path.join(tmp, "lookalike_map_FloorPlan15.json"), "w", encoding="utf-8"))
    check("2.4 the map of the asked kitchen is selected by name and accepted", hs.load_frozen_map(kitchen="FloorPlan15") == fp1_doc["map"])
    check("2.5 ... and another kitchen's file is never picked for it", raises(lambda: hs.load_frozen_map(kitchen="FloorPlan17"), OSError))
finally:
    hs.LOOKALIKE_MAP_FILE = _tmpl
import honest_map as HM                                  # noqa: E402
gm15 = HM.GridMap.load(os.path.join(SIM, "step4", "FloorPlan15.map.json"))
gm1 = HM.GridMap.load("FloorPlan1")
check("2.6 scene_problems: Kitchen 2's navigation map under a Kitchen 1 controller is a problem; Kitchen 1's own map is not",
      len(sl.scene_problems("FloorPlan1", nav_map=gm15)) == 1 and sl.scene_problems("FloorPlan1", nav_map=gm1) == [])
check("2.7 scene_problems: a request or a memory of another kitchen is a problem",
      len(sl.scene_problems("FloorPlan1", SimpleNamespace(kitchen="FloorPlan15", memory=None))) == 1
      and len(sl.scene_problems("FloorPlan15", SimpleNamespace(kitchen="FloorPlan15", memory=SimpleNamespace(kitchen="FloorPlan1")))) == 1)
le = sl.LiveExecutor(memory="places", scene="FloorPlan1")
le._boot = lambda: (_ for _ in ()).throw(AssertionError("booted"))
check("2.8 LiveExecutor.execute refuses a cross-kitchen request BEFORE booting the simulator (LiveInvariant)",
      raises(lambda: le.execute(SimpleNamespace(kitchen="FloorPlan15", memory=None)), sl.LiveInvariant, "request kitchen"))
import ltm as L                                          # noqa: E402
import ltm_store as LS                                   # noqa: E402
snap = T.source_dir("test-v1", "S*")
check("2.9 a frozen snapshot of Kitchen 1 is refused for another kitchen (manifest check, read-only)",
      raises(lambda: L.LongTermMemory.open_frozen(snap, os.path.join(tmp, "t.db"), kitchen="FloorPlan15"), LS.StoreError, "FloorPlan1"))
check("2.10 Packager: an explicit scene that is not the namespace's kitchen is refused",
      raises(lambda: T.Packager("test-v1", None, scene="FloorPlan15"), T.TreatmentError)
      and raises(lambda: T.Packager("test-k2-v1", None, scene="FloorPlan1"), T.TreatmentError))
import practice_k as PK                                  # noqa: E402
check("2.11 practice CLI: a --kitchen that conflicts with the namespace's scene is refused (before anything runs)",
      raises(lambda: PK.main(["--namespace", "practice-v2", "--kitchen", "FloorPlan15", "--status"]), SystemExit, "conflicts"))
import retention_a12 as RA                               # noqa: E402
check("2.12 retention A12: --write / --replace from a replication kitchen are refused (K1 allocation never rewritten)",
      RA.main(["--measure", "--kitchen", "FloorPlan15", "--write"]) == 2 and RA.main(["--measure", "--kitchen", "FloorPlan17", "--replace"]) == 2)
import freeze_ft2 as FT                                  # noqa: E402
check("2.13 freeze_ft2: Kitchen 1's class registration is the unchanged REGISTERED_CLASSES; K2 / K3 have none until pinned (12.12 fails closed)",
      FT.REGISTERED_CLASSES_BY_SCENE["FloorPlan1"] is FT.REGISTERED_CLASSES and FT.REGISTERED_CLASSES["n"] == 49
      and FT.REGISTERED_CLASSES_BY_SCENE["FloorPlan15"] is None and FT.REGISTERED_CLASSES_BY_SCENE["FloorPlan17"] is None)
check("2.14 a replication namespace never resolves into Kitchen 1's snapshot tree (and a missing kitchen snapshot is not replaced by K1's)",
      all(os.path.dirname(os.path.dirname(T.source_dir(n, s))).endswith(C.ns_scene(n)) for n in REP for s in SOURCES))
check("2.15 a replication namespace refuses a non-ft-v2 configuration path (source_dir)", "a replication namespace must be ft-v2" in src("final_test/matrix/treatment.py"))
# nine-target look-alike provenance supplement (Astra K23-AGENT-r1 2; FP1-EQUIV-r1 e): separately hashed, never changes the map
NINE = sorted(set(hs.AMBIG_TYPES) - set(hs.LOOKALIKE_EXCLUDE))
prov1 = hs.load_lookalike_provenance("FloorPlan1", hs.load_frozen_map(kitchen="FloorPlan1"))
check("2.16 Kitchen 1's supplement: all nine targets, bound to its frozen map, M recomputed from its rows == the frozen map",
      sorted(prov1["targets"]) == NINE and len(NINE) == 9 and all(v["status"] == "in_map" for v in prov1["targets"].values()))
p1 = os.path.join(SIM, *hs.LOOKALIKE_PROVENANCE_FILE.format(kitchen="FloorPlan1").split("/"))
base_doc = json.load(open(p1, encoding="utf-8"))


def prov_refused(mut):
    d = json.loads(json.dumps(base_doc))
    mut(d)
    if "keep_sha" not in d:
        d["sha256"] = hs._canon_sha(d)
    d.pop("keep_sha", None)
    q = os.path.join(tmp, "prov.json")
    json.dump(d, open(q, "w", encoding="utf-8"))
    return raises(lambda: hs.load_lookalike_provenance("FloorPlan1", hs.load_frozen_map(kitchen="FloorPlan1"), q), RuntimeError)


check("2.17 a supplement is refused when: a target is missing, a status is not one of the three, a count changes the recomputed map, "
      "it names another kitchen, it is bound to another map sha, or it was edited without its self sha",
      prov_refused(lambda d: d["targets"].pop("Pan")) and prov_refused(lambda d: d["targets"]["Pan"].__setitem__("status", "zero"))
      and prov_refused(lambda d: d["targets"]["Spoon"]["confused_with"].__setitem__("Fork", 0))
      and prov_refused(lambda d: d.__setitem__("kitchen", "FloorPlan15")) and prov_refused(lambda d: d.__setitem__("map_sha256", "0" * 64))
      and prov_refused(lambda d: d.update(theta=0.5, keep_sha=True)))
check("2.18 a kitchen without a supplement is refused (HonestSkills loads it with the frozen map)",
      raises(lambda: hs.load_lookalike_provenance("FloorPlanNone", {}), OSError)
      and "self.amb_provenance = load_lookalike_provenance(kitchen, self.amb_map)" in src("honest_skills.py"))
p15 = os.path.join(SIM, *hs.LOOKALIKE_PROVENANCE_FILE.format(kitchen="FloorPlan15").split("/"))
d15 = json.load(open(p15, encoding="utf-8"))
rows15 = [{"cls": t, "n": v["n"] or 0, "confused_with": v["confused_with"]} for t, v in d15["targets"].items()]
check("2.19 Kitchen 2's supplement: nine targets (5 in the map, 4 with evidence but no confusion >= theta), self-consistent, bound to the "
      "pinned five-target K2 map (07646d02...) which it reproduces row for row",
      sorted(d15["targets"]) == NINE and d15["sha256"] == hs._canon_sha(d15) and d15["map_sha256"].startswith("07646d020c1bca23")
      and sorted(t for t, v in d15["targets"].items() if v["status"] == "in_map") == ["ButterKnife", "Fork", "PepperShaker", "SaltShaker", "Spoon"]
      and sorted(t for t, v in d15["targets"].items() if v["status"] == "below_theta") == ["Pan", "Plate", "Pot", "Spatula"]
      and hs.ambiguous_objects_map(rows15, d15["theta"]) == {t: v["map_row"] for t, v in d15["targets"].items() if v["map_row"]})

# ===================================================================================================== 3 resource isolation
print("== 3 resource isolation (no Kitchen 2 / 3 output path can be a Kitchen 1 path)")


def ns_paths(n):
    out = {C.ns_docs(n), C.ns_work(n), os.path.join(C.ns_work(n), "cache"), os.path.join(C.ns_work(n), "tokcache.json")}
    out |= {C.journal_path(n, k) for k in C.JOURNALS}
    return {os.path.normcase(os.path.abspath(p)) for p in out}


k1_paths = set().union(*(ns_paths(n) for n in K1))
check("3.1 docs / work / journals / response cache / token cache of every K2 / K3 namespace are disjoint from every K1 one",
      all(not (ns_paths(n) & k1_paths) for n in REP))
check("3.2 ... and from each other", len(set().union(*(ns_paths(n) for n in REP))) == sum(len(ns_paths(n)) for n in REP))
k1_snaps = {os.path.normcase(T.source_dir(n, s)) for n in K1 for s in SOURCES}
check("3.3 ft-v2 snapshot dirs of K2 / K3 are disjoint from K1's and from each other's",
      all(os.path.normcase(T.source_dir(n, s)) not in k1_snaps for n in REP for s in SOURCES)
      and {os.path.normcase(T.source_dir(n, "S*")) for n in ("test-k2-v1", "test-k3-v1")} & k1_snaps == set())
check("3.4 per-kitchen package resources: uses table / inventory files differ from K1's frozen r6b files",
      LP.uses_path("FloorPlan1") == LP.USES_PATH and LP.inventory_path("FloorPlan1") == LP.INVENTORY_PATH
      and len({LP.uses_path(k) for k in ("FloorPlan1", "FloorPlan15", "FloorPlan17")}) == 3
      and len({LP.inventory_path(k) for k in ("FloorPlan1", "FloorPlan15", "FloorPlan17")}) == 3)
check("3.5 per-kitchen freeze record and A29 files never name docs/final-test/freeze.json",
      all(os.path.normcase(K.freeze_path(s)) != os.path.normcase(C.paths()["freeze"]) for s in K.REPLICATION_SCENES)
      and len({K.freeze_path(s) for s in K.REPLICATION_SCENES}) == 2 and len({K.a29_files(s)[0] for s in K.REPLICATION_SCENES}) == 2)
wdt = src("final_test/matrix/write_detector_types.py")
check("3.6 write_detector_types: Kitchen 1 keeps detector_types.json, another scene writes detector_types_<scene>.json",
      '"detector_types.json" if SCENE == "FloorPlan1" else f"detector_types_{SCENE}.json"' in wdt)
check("3.7 map build: Kitchen 1's map_report.json is never overwritten by another kitchen",
      'rname = "map_report.json" if scene == "FloorPlan1" else f"{scene}.map_report.json"' in src("step4/s2_build_map.py"))
check("3.8 retention A12 output: a replication kitchen writes a12_measure_<kitchen>.json, never Kitchen 1's file",
      'f"a12_measure_{a.kitchen}.json"' in src("final_test/matrix/retention_a12.py"))
check("3.9 look-alike map and navigation map file names are per kitchen", hs.LOOKALIKE_MAP_FILE.format(kitchen="FloorPlan15") != hs.LOOKALIKE_MAP_FILE.format(kitchen="FloorPlan1")
      and "{kitchen}" in hs.LOOKALIKE_MAP_FILE)
check("3.10 the K2 + K3 Opus ceilings are separate from every K1 ceiling (35 + 35 = the 70 USD approval)",
      sum(C.NAMESPACES[n]["ceilings"]["total"] for n in REP) == 70.0 and all(n not in K1 for n in REP))

# ===================================================================================================== 4 unchanged Kitchen 1 defaults
print("== 4 unchanged Kitchen 1 defaults")
sig = lambda f, p: inspect.signature(f).parameters[p].default              # noqa: E731
check("4.1 function defaults stay FloorPlan1 (LiveExecutor, HonestSkills, load_frozen_map, treatment snapshot/params/file shas, LP profile)",
      sig(sl.LiveExecutor.__init__, "scene") == "FloorPlan1" and sig(hs.HonestSkills.__init__, "kitchen") == "FloorPlan1"
      and sig(hs.load_frozen_map, "kitchen") == "FloorPlan1" and sig(T.snapshot_root, "scene") == "FloorPlan1"
      and sig(T.snapshot_dir, "scene") == "FloorPlan1" and sig(T.effective_params, "kitchen") == "FloorPlan1"
      and sig(T.file_shas, "scene") == "FloorPlan1" and sig(LP.profile_params, "kitchen") == "FloorPlan1")
import freeze as F                                       # noqa: E402
check("4.2 freeze builders default to FloorPlan1; retention A12 --kitchen defaults to FloorPlan1",
      sig(F.build, "scene") == "FloorPlan1" and sig(FT.build_ft2, "scene") == "FloorPlan1" and sig(FT.gate_1212_ft2, "scene") == "FloorPlan1"
      and "--kitchen\", default=C.DEFAULT_SCENE" in src("final_test/matrix/retention_a12.py"))
check("4.3 the default look-alike map is Kitchen 1's frozen map (9 targets), identical to the explicit FloorPlan1 load",
      hs.load_frozen_map() == hs.load_frozen_map(kitchen="FloorPlan1") == fp1_doc["map"] and len(fp1_doc["map"]) == 9)
check("4.4 FloorPlan1 identity file list = the frozen IDENTITY_FILES (same paths, same order); file_shas() = file_shas(scene=FloorPlan1)",
      T.identity_files("FloorPlan1") == T.IDENTITY_FILES and T.identity_files() == T.IDENTITY_FILES)
check("4.5 LP.profile_params() == profile_params(kitchen='FloorPlan1') for every profile; effective_params likewise",
      all(LP.profile_params(p) == LP.profile_params(p, kitchen="FloorPlan1") for p in LP.PROFILES)
      and all(T.effective_params(p) == T.effective_params(p, kitchen="FloorPlan1") for p in ("pkg-1.r4", "pkg-1.r6", "pkg-1.r6b")))
check("4.6 K1 snapshot paths are the pre-K23 ones (snapshots_ft2/oX/kY, no scene level)",
      T.source_dir("test-v1", "S*") == os.path.join(T.snapshot_root(), "o0", "k10") and T.snapshot_root() == T.snapshot_root(scene="FloorPlan1")
      and os.path.basename(T.snapshot_root()) == "snapshots_ft2")
check("4.7 every Kitchen 1 request / memory / map is scene-consistent (the new guards are inert on Kitchen 1)",
      sl.scene_problems("FloorPlan1", SimpleNamespace(kitchen="FloorPlan1", memory=SimpleNamespace(kitchen="FloorPlan1")), gm1) == [])
check("4.8 K1 arms, REPEAT / relocation rules and practice pools unchanged",
      C.main_arms_for("test-v1") == tuple(C.MAIN_ARMS) and C.repeat_items("test-v1", C.load_manifest()) != []
      and all(C.ns_scene(n) == C.ns_scene(m) for n in K1 for m in K1))
fx = os.path.join(SIM, "final_test", "_checkpoints", "K23-K1-BASELINE")
outp = os.path.join(tmp, "fixtures_now.json")
r = subprocess.run([sys.executable, os.path.join(fx, "k1_fixtures.py"), "--tree", os.path.dirname(SIM), "--out", outp],
                   capture_output=True, text=True, timeout=600)
if r.returncode == 0 and os.path.exists(outp):
    now, base = (json.load(open(p, encoding="utf-8")) for p in (outp, os.path.join(fx, "base_master.json")))
    d = sorted(k for k in set(now) | set(base) if now.get(k) != base.get(k))
    check("4.9 K1 byte-regression fixtures (namespaces, paths, journals, items, treatment / planner keys, snapshot dirs, switch / config /"
          " profile / effective params, uses / inventory, contract system texts, types, briefing hint table, rng draws, caps, test items)"
          " equal the master baseline; only the declared successor fields differ (file_shas_identity; after the authorized re-pin also config_sha + switch_sha, whose ONLY content change is the five FT2 pins recorded in docs/final-test/k23/repin.json)", d in ([], ["file_shas_identity"], ["config_sha", "file_shas_identity", "switch_sha"]), d)
else:
    check("4.9 K1 byte-regression fixtures computable", False, (r.returncode, r.stderr[-300:]))


# 4.10 (Astra re-pin review): the treatment CONFIGURATION content, not only its hash, differs from the master baseline ONLY in the five FT2
# pin values; read from the baseline commit itself, compared path by path, and the five new values are the live files' own shas
import ast as _ast
_BASE = "b8be0c67f375660c3c268e6e24fb314302632086"
_bsrc = subprocess.run(["git", "-C", os.path.dirname(SIM), "show", f"{_BASE}:sim/final_test/matrix/treatment.py"], capture_output=True,
                       text=True, encoding="utf-8").stdout
_bft2 = None
for _n in _ast.parse(_bsrc).body:
    if isinstance(_n, _ast.Assign) and any(getattr(x, "id", None) == "FT2" for x in _n.targets):
        _bft2 = eval(compile(_ast.Expression(_n.value), "<baseline FT2>", "eval"), {"__builtins__": {}}, {"FT2_VERSION": T.FT2_VERSION})


def _flat(x, pre=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(_flat(v, f"{pre}/{k}"))
        return out or {pre: {}}
    return {pre: x}


_fb, _fn = _flat(_bft2 or {}), _flat(T.config())
_changed = sorted(k for k in set(_fb) | set(_fn) if _fb.get(k) != _fn.get(k))
_FIVE = sorted(["/pins/executor/sim/honest_skills.py", "/pins/contract/sim/planners/exec_contract.json/file",
                "/pins/contract/sim/planners/exec_contract.json/self", "/pins/contract/sim/planners/ft_v2.json/file",
                "/pins/contract/sim/planners/ft_v2.json/self"])
_ec = json.load(open(os.path.join(SIM, "planners", "exec_contract.json"), encoding="utf-8"))
_ft = json.load(open(os.path.join(SIM, "planners", "ft_v2.json"), encoding="utf-8"))
import hashlib as _hl
_fsha = lambda rel: _hl.sha256(open(os.path.join(SIM, *rel.split("/")), "rb").read()).hexdigest()      # noqa: E731
check("4.10 the treatment configuration (FT2) differs from the master baseline's ONLY in the five pin values (executor honest_skills.py; "
      "exec_contract.json file + self; ft_v2.json file + self); each new value is the live file's own sha; texts sha, switches, "
      "executor honest.py / objmem.py and every other key unchanged; pin_problems() == []",
      _bft2 is not None and _changed == _FIVE
      and _fn["/pins/executor/sim/honest_skills.py"] == _fsha("honest_skills.py")
      and _fn["/pins/contract/sim/planners/exec_contract.json/file"] == _fsha("planners/exec_contract.json")
      and _fn["/pins/contract/sim/planners/exec_contract.json/self"] == _ec.get("sha256")
      and _fn["/pins/contract/sim/planners/ft_v2.json/file"] == _fsha("planners/ft_v2.json")
      and _fn["/pins/contract/sim/planners/ft_v2.json/self"] == _ft.get("sha256")
      and T.pin_problems() == [], _changed)

shutil.rmtree(tmp, ignore_errors=True)

# equivalence proof 2026-10-18: Kitchen 1 freeze gate 12.13 judges the Kitchen 1 namespaces only (the six K2/K3 namespaces are
# reported separately and isolated by their own per-kitchen freeze)
import tempfile as _tf
import freeze as _F
_g = _F.gate_namespaces(_tf.mkdtemp())
_ev = _g.get("evidence") or {}
check("12.13: Kitchen 1 namespace gate unaffected by the K2/K3 namespaces",
      _g.get("pass") is True and not any(n.startswith(("practice-k", "rehearsal-k", "test-k")) for n in _ev.get("namespaces", [])))
check("12.13: the six K2/K3 namespaces are recorded separately",
      sorted(_ev.get("replication_namespaces", [])) == ["practice-k2-v1", "practice-k3-v1", "rehearsal-k2-v1", "rehearsal-k3-v1", "test-k2-v1", "test-k3-v1"])

print(f"{ok}/{ok + fail}")
sys.exit(1 if fail else 0)
