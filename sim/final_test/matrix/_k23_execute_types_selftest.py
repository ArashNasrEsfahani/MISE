"""Offline gate for the K23 execute-phase type list (coordinator 2026-10-20 ~21:40, read-only diagnosis of the rehearsal-k3-v1 G-iso): k23_run
installed the kitchen's B5 list for plan and context but NOT for execute, so execute's G-iso rendered the none-arm request (A19 component (i))
through plan.none_request_renderer -> plan.load_types with Kitchen 1's docs/final-test/matrix/detector_types.json (49 types) instead of the
kitchen's list (FloorPlan17: 45) -> "request probe differs from Phase A" on 12/12. This test reproduces the mechanism and checks that the
execute path of k23_run now installs k23_load_types for the namespace (and that the renderer then builds its contract from the kitchen's list)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import k23_run as R
import plan as PL
import execute as X
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def types_of(name):
    d = json.load(open(os.path.join(C.paths(None)["matrix_docs"], name), encoding="utf-8"))
    return sorted(set(d["types"] if isinstance(d, dict) else d))


K1, K3, K2 = types_of("detector_types.json"), types_of("detector_types_FloorPlan17.json"), types_of("detector_types_FloorPlan15.json")
orig_load, orig_tlp, orig_pre, orig_xmain, orig_mk = PL.load_types, R.type_list_problems, R.preflight, X.main, PL.make_contract
check("the lists differ: Kitchen 1 has 49 types, FloorPlan17 45, FloorPlan15 43", (len(K1), len(K3), len(K2)) == (49, 45, 43) and K1 != K3)
check("the bug's mechanism: plan.load_types unpatched reads Kitchen 1's detector_types.json (what execute's G-iso used)",
      sorted(set(orig_load(None, None)[0])) == K1)

seen = {}
try:
    # the kitchen's list without the WSL perception binding (the binding itself is k23_run's own, tested in _k23_run_selftest)
    R.type_list_problems = lambda ns, root=None: (K3 if R.E.scene_of(ns) == "FloorPlan17" else K2, [])
    R.preflight = lambda phase, ns, **kw: {"phase": phase, "namespace": ns, "kind": R.E.kind(ns), "scene": R.E.scene_of(ns), "guard": {"n": 0},
                                           "needs_wsl": True, "platform_ok": True}

    def fake_main(argv):
        seen["load_types"] = PL.load_types
        seen["types"] = PL.load_types(None, None)[0]
        captured = {}
        PL.make_contract = lambda types, ns=None: (captured.setdefault("types", list(types)), orig_mk(types, ns))[1]
        try:
            PL.none_request_renderer(None, ns=argv[argv.index("--namespace") + 1])
        except Exception as e:                                       # noqa: BLE001  (only the contract's type list matters here)
            captured.setdefault("err", repr(e))
        seen["renderer_types"] = captured.get("types")
        return 0
    X.main = fake_main
    rc = R.main(["--phase", "execute", "--namespace", "rehearsal-k3-v1", "--dry-run"])
    check("k23_run's execute path installs k23_load_types for THE namespace before execute.main", rc == 0
          and getattr(seen.get("load_types"), "k23_ns", None) == "rehearsal-k3-v1")
    check("... so the G-iso type list is FloorPlan17's 45, not Kitchen 1's 49", seen.get("types") == K3)
    check("... and plan.none_request_renderer builds its contract from the kitchen's list", seen.get("renderer_types") is not None
          and sorted(set(seen["renderer_types"])) == K3)
    PL.load_types = orig_load
    seen.clear()
    rc = R.main(["--phase", "execute", "--namespace", "test-k2-v1", "--allow-test", "--dry-run"])
    check("the same for a test namespace (test-k2-v1 -> FloorPlan15's 43)", rc == 0 and getattr(seen.get("load_types"), "k23_ns", None) == "test-k2-v1"
          and seen.get("types") == K2)
finally:
    PL.load_types, R.type_list_problems, R.preflight, X.main, PL.make_contract = orig_load, orig_tlp, orig_pre, orig_xmain, orig_mk
src = open(os.path.join(HERE, "k23_run.py"), encoding="utf-8").read()
ex = src[src.index('if a.phase == "execute":\n        import execute as X'):]
check("k23_run.py: the execute branch patches plan.load_types before X.main (static)", ex.index("PL.load_types = k23_load_types(ns)") < ex.index("X.main(margv)"))
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
