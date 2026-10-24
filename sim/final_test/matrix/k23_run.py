"""K23 orchestrator: the rehearsal and test phases of Kitchens 2 / 3 (rehearsal-k2-v1 / test-k2-v1 = FloorPlan15, rehearsal-k3-v1 /
test-k3-v1 = FloorPlan17) through the UNCHANGED phase modules (capture.py, plan.py + planfirst.py, execute.py, score.py).

matrix_run.py / capture.py / plan.py / execute.py are Kitchen-1-shaped at their entry checks (matrix_run.py:143-146 and capture.py:1097
refuse these namespaces; plan.py:520 indexes the split manifest by the namespace's items key, which has no 'a29_roster'; plan.load_types
checks the type list against Kitchen 1's freeze.json). None of them is edited (A37 hold, treatment pins, Kitchen 1 bytes). This process:

  preflight   K23 namespace kind; --allow-test exactly for a test namespace; the namespace's items (C.namespace_items, the A29 roster for
              test) through C.guard_items (test: k23_ns.replication_freeze_status = THIS kitchen's agent freeze); upstream phases complete
              and unchanged (matrix_run.dependency_state); no unresolved invalid abort; the phase lock; for test also the K23 entry checks
              (k23_entry.test_entry_problems: the freeze's treatment identity / dataset / type list equal the running ones; FloorPlan15 COCO
              sensitivity appended to the perception freeze); for plan the A37 paid-planning hold is lifted.
  capture     (WSL) capture.main with this namespace admitted in-process (rehearsal -> capture.DEV_NAMESPACES, test -> NS_FOR_ITEMS['test'],
              the fg_runner pattern) and the K23 design forced: --runs 1 --worlds canon --no-fs --no-curve (A29 item 1: no FS, curve,
              relocation, REPEAT in Kitchens 2 / 3).
  plan        (Windows, simulator off) planfirst.run with the stages free (Gemma) + core0 (Opus, synchronous: approve-all-6 item 3), a Plan
              subclass whose manifest carries the A29 roster (acc_common.acceptance_manifest pattern), the kitchen's frozen B5 type list
              (k23_load_types: detector_types_<scene>.json bound to the kitchen's perception freeze and, for test, to the agent freeze), and the
              allow_test flag kept for the completion record (the planfirst.py:135 bug, fixed here without editing planfirst.py). Test pause
              acknowledgements need --pause-response and --outcome-blind (the test-v1 strictness of plan.Plan._ack_problem). Energy (C5,
              A24.2; Astra K23-TOOLING-G2 1): a test namespace plans with the meter ON, as test-v1 does; --energy off is refused unless
              Arash records --energy-waiver NOTE. Execute / score of a test namespace need complete strict energy coverage of the plan
              journal or a recorded waiver (the matrix_run.energy_gate rule of test-v1).
  execute     (WSL) execute.main with --worlds canon --no-replay (A29: no /p replays in Kitchens 2 / 3); --items dev for rehearsal, none for test
              (execute.py:1048 ties --items test to test-v1).
  score       score.main (any python).
  status      phase states from the integrity journals (matrix_run.status).

    python sim/final_test/matrix/k23_run.py --phase status  --namespace rehearsal-k2-v1
    python sim/final_test/matrix/k23_run.py --phase capture --namespace rehearsal-k2-v1 -- --ollama http://<win host>:11434     (WSL)
    python sim/final_test/matrix/k23_run.py --phase plan    --namespace rehearsal-k2-v1 [--parallel 4]                         (Windows)
    python sim/final_test/matrix/k23_run.py --phase execute --namespace rehearsal-k2-v1 [--inject-interrupt 300]                (WSL)
    python sim/final_test/matrix/k23_run.py --phase score   --namespace rehearsal-k2-v1
    (test-k2-v1: the same with --allow-test; the plan meters Gemma energy; --energy off only with Arash's --energy-waiver NOTE)

Exit codes: the phase module's own code; 10 preflight refusal; 11 wrong platform (WSL needed; the command is printed); 12 usage.
Python 3.10 syntax; no key is read or printed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402
import k23_entry as E                                    # noqa: E402
import k23_ns as K                                       # noqa: E402

EXIT = {"refused": 10, "platform": 11, "usage": 12}
PHASES = ("capture", "plan", "execute", "score", "status", "context")
WSL_PHASES = ("capture", "execute")
K23_STAGES = ("free", "core0")                           # Gemma G0 / G+M, then Opus O0 / O+M r = 0; every other stage has no K23 cell
CAPTURE_FORCED = ["--runs", "1", "--worlds", "canon", "--no-fs", "--no-curve"]
CAPTURE_ALLOWED = {"--ollama": 1, "--steal-lock": 0, "--limit": 1, "--inject-interrupt": 1, "--dry-run": 0, "--allow-missing-anthropic": 0}
EXEC_FORCED = ["--worlds", "canon", "--no-replay"]
EXEC_ALLOWED = {"--steal-lock": 0, "--inject-interrupt": 1, "--partial": 0, "--dry-run": 0, "--resolve-invalid": 1, "--declare-unrecoverable": 1}
REHEARSAL_EXEC_ALLOWED = dict(EXEC_ALLOWED, **{"--g-iso": 0})          # G-iso (12.6) in the kitchen's rehearsal only, n = 12 (registered)
GISO_N = 12
SCORE_ALLOWED = {"--steal-lock": 0, "--partial": 0, "--dry-run": 0}


class Refused(RuntimeError):
    pass


def _check_extra(extra: list, allowed: dict, phase: str, ns: str) -> list:
    """Only the listed pass-through options (with their arity) reach a phase module; anything else is refused (design is forced here)."""
    out, i = [], 0
    while i < len(extra):
        a = extra[i]
        if a not in allowed:
            raise Refused(f"{phase}: option {a!r} is not allowed for {ns} (the K23 design is fixed: {', '.join(sorted(allowed))})")
        n = allowed[a]
        if n and i + n >= len(extra):
            raise Refused(f"{phase}: option {a} needs a value")
        out += extra[i:i + 1 + n]
        i += 1 + n
    return out


# ================================================================================================ the kitchen's frozen type list (B5)
def types_path(scene: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["matrix_docs"], f"detector_types_{scene}.json")


def perception_entry(scene: str, rel: str, root: str | None = None):
    """The sha the kitchen's perception freeze record lists for repo-relative `rel` (None when absent or no record)."""
    p = os.path.join(C.paths(root)["root"], "sim", "kitchen_v3", scene, "frozen_weights.sha256")
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8"):
        if line.strip():
            sha, r = line.split(None, 1)
            if r.strip().lstrip("*") == rel:
                return sha
    return None


def type_list_problems(ns: str, root: str | None = None) -> tuple:
    """(types, problems) of the kitchen's B5 list: schema, scene, weights path, sorted unique names, n; weights sha = the perception freeze's
    entry for the primary; test: sha256_json(types) = the agent freeze's detector_types_sha256."""
    scene = E.scene_of(ns)
    p = types_path(scene, root)
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return [], [f"{scene}: frozen type list unreadable ({os.path.basename(p)}: {type(e).__name__})"]
    types = list(d.get("types") or [])
    wrel = f"sim/weights/v3/{scene}.yolo26s.pt"
    out = []
    if d.get("schema") != "ftm.detector-types/1" or d.get("scene") != scene or d.get("weights") != wrel:
        out.append(f"{scene}: type list schema / scene / weights path mismatch")
    if not types or types != sorted(set(types)) or d.get("n") != len(types):
        out.append(f"{scene}: type list not a sorted unique list of n names")
    pe = perception_entry(scene, wrel, root)
    if pe is None or d.get("weights_sha") != pe:
        out.append(f"{scene}: the type list's weights sha is not the perception freeze's primary weights sha")
    if E.kind(ns) == "test":
        try:
            with open(K.freeze_path(scene, root), encoding="utf-8") as f:
                fz = json.load(f)
        except (OSError, ValueError):
            fz = {}
        if fz.get("detector_types_sha256") != C.sha256_json(types):
            out.append(f"{scene}: the type list differs from the agent freeze's detector_types_sha256")
    return types, out


def k23_load_types(ns: str):
    """A drop-in for plan.load_types bound to ONE K23 namespace: the kitchen's B5 list, never Kitchen 1's detector_types.json, never
    checked against Kitchen 1's freeze.json. Any problem is a PlanAbort (fail closed)."""
    import plan as PL

    def load_types(path=None, root=None):
        want = types_path(E.scene_of(ns), root)
        if path and os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(want)):
            raise PL.PlanAbort("abort_types", f"{ns}: the type list must be {os.path.basename(want)}, not {path}")
        types, bad = type_list_problems(ns, root)
        if bad:
            raise PL.PlanAbort("abort_types", "; ".join(bad[:4]))
        return sorted(set(types)), os.path.basename(want)
    load_types.k23_ns = ns
    return load_types


# ================================================================================================ energy (C5 / A24.2), test-v1 parity
def energy_gate(phase: str, ns: str, root: str | None = None, waiver: str | None = None):
    """Test namespaces: execute / score need complete strict energy coverage of the plan journal, or a waiver recorded by Arash (written to
    <ns docs>/energy_waivers.jsonl). Returns the coverage (None when the rule does not apply). Raises Refused otherwise."""
    if E.kind(ns) != "test" or phase not in ("execute", "score"):
        return None
    import energy as EN
    cov = EN.coverage(ns, root)
    if cov.get("complete") is not False:
        return cov
    if not waiver:
        raise Refused(f"C5 energy evidence incomplete ({cov.get('measured')} of {cov.get('gemma_plan_rows')} Gemma plan rows measured): the "
                      "A24.2 Gemma cost would be UNKNOWN. Ask Arash; to proceed anyway he records --energy-waiver NOTE")
    record_waiver(ns, phase, waiver, root, coverage=cov)
    return dict(cov, waiver=waiver[:500])


def record_waiver(ns: str, phase: str, note: str, root: str | None = None, coverage=None) -> None:
    import time
    os.makedirs(C.ns_docs(ns, root), exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase, "by": "Arash", "note": str(note)[:500], "runner": "k23_run.py",
           "coverage": {k: (coverage or {}).get(k) for k in ("gemma_plan_rows", "measured", "refused_interval", "no_energy_row")}}
    with open(os.path.join(C.ns_docs(ns, root), "energy_waivers.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")


# ================================================================================================ preflight
def preflight(phase: str, ns: str, *, allow_test: bool = False, steal_lock: bool = False, root: str | None = None,
              partial: bool = False, recovery: bool = False) -> dict:
    import matrix_run as MR
    E.install_perception_reader()                                  # Windows: the perception freeze's WSL paths through the WSL share
    k = E.kind(ns)                                                 # ValueError for anything but a K23 rehearsal / test / practice namespace
    if k == "practice":
        raise Refused(f"{ns} is practice: use k23_practice.py")
    bad_order = E.order_problems(ns, root)                     # prereg A39.2: Kitchen 2 rehearsal only after COCO + --add-coco + meta
    if bad_order:
        raise Refused("; ".join(bad_order[:3]))
    if k == "test" and not allow_test:
        raise Refused(f"{ns} holds test items: pass --allow-test (and only after its agent freeze)")
    if k != "test" and allow_test:
        raise Refused(f"{ns} holds no test items: --allow-test is refused")
    try:
        man = E.k23_manifest(ns, root)
        ids = sorted(C.namespace_items(ns, man, root))
    except RuntimeError as e:
        raise Refused(str(e))
    guard = C.guard_items(ids, ns, allow_test=allow_test, root=root)          # test: THIS kitchen's agent freeze (replication_freeze_status)
    rep = {"phase": phase, "namespace": ns, "kind": k, "scene": E.scene_of(ns), "guard": guard, "deps": {}, "lock": None}
    if k == "test":
        bad = E.test_entry_problems(ns, root)
        if bad:
            raise Refused("K23 test entry: " + "; ".join(bad[:6]))
    if phase == "plan":
        hold = K.a37_hold_problems(root)
        if hold:
            raise Refused("A37 paid-planning hold: " + "; ".join(hold[:3]))
    deps = MR.dependency_state(ns, phase, root, partial=partial and phase in ("execute", "score"))
    rep["deps"] = deps
    bad = [f"{d}: {v['why']}" for d, v in deps.items() if not v["complete"] or v["unchanged"] is False]
    if bad:
        raise Refused("upstream phase not ready - " + "; ".join(bad))
    if phase in ("execute", "score") and MR.open_invalid(ns, root) and not (recovery and phase == "execute"):    # Astra G2 3: recovery
        raise Refused("an invalid execute run aborted the study and is unresolved: resolve it first (execute --resolve-invalid)")
    if phase in MR.LOCKED_PHASES:
        held = MR.lock_held(ns, phase, root)
        rep["lock"] = held
        if held is not None and not steal_lock:
            raise Refused(f"phase lock is held ({held[:120]}); if that process is dead, pass --steal-lock")
    rep["needs_wsl"] = phase in WSL_PHASES
    rep["platform_ok"] = not (phase in WSL_PHASES and not MR.in_wsl())
    return rep


# ================================================================================================ dispatch
def admit_capture(ns: str):
    import capture as CP
    k = E.kind(ns)
    if k == "rehearsal" and ns not in CP.DEV_NAMESPACES:
        CP.DEV_NAMESPACES = tuple(CP.DEV_NAMESPACES) + (ns,)
    elif k == "test" and CP.NS_FOR_ITEMS.get("test") != ns:
        CP.NS_FOR_ITEMS = dict(CP.NS_FOR_ITEMS, test=ns)
    return CP


def capture_argv(ns: str, extra: list, *, allow_test: bool, steal_lock: bool, dry_run: bool, interrupt) -> list:
    k = E.kind(ns)
    ex = _check_extra(extra, CAPTURE_ALLOWED, "capture", ns)
    if k == "test" and ("--allow-missing-anthropic" in ex or "--limit" in ex):
        raise Refused("capture of a test namespace: --allow-missing-anthropic and --limit are refused (every tokenizer, every presentation)")
    if not dry_run and "--ollama" not in ex:
        raise Refused("capture needs an explicit --ollama http://<windows host ip>:11434 (localhost is not Windows inside WSL)")
    a = ["--namespace", ns, "--items", "test" if k == "test" else "dev"] + CAPTURE_FORCED
    a += (["--allow-test"] if allow_test else []) + (["--steal-lock"] if steal_lock else []) + (["--dry-run"] if dry_run else [])
    a += (["--inject-interrupt", str(int(interrupt))] if interrupt else [])
    return a + [x for x in ex if x not in ("--steal-lock", "--dry-run")]


def execute_argv(ns: str, extra: list, *, allow_test: bool, steal_lock: bool, dry_run: bool, interrupt, partial: bool) -> list:
    ex = _check_extra(extra, REHEARSAL_EXEC_ALLOWED if E.kind(ns) == "rehearsal" else EXEC_ALLOWED, "execute", ns)
    if "--g-iso" in ex:
        ex = ex + ["--g-iso-n", str(GISO_N)]
    a = ["--namespace", ns] + (["--items", "dev"] if E.kind(ns) == "rehearsal" else []) + EXEC_FORCED
    a += (["--allow-test"] if allow_test else []) + (["--steal-lock"] if steal_lock else []) + (["--dry-run"] if dry_run else [])
    a += (["--inject-interrupt", str(int(interrupt))] if interrupt else []) + (["--partial"] if partial else [])
    return a + [x for x in ex if x not in ("--steal-lock", "--dry-run", "--partial")]


def k23_plan_factory(ns: str, allow_test: bool, types_path_: str):
    """plan_factory for planfirst.run: a Plan whose manifest carries the A29 roster, with the kitchen's type list and the runner's allow_test
    flag (also for planfirst._complete, which passes plan_kw.get('allow_test') = False: the K1 2026-10-16 completion bug)."""
    import plan as PL

    class K23Plan(PL.Plan):
        def _manifest(self):
            return E.k23_manifest(self.ns, self.root)

    def mk(n, **kw):
        if n != ns:
            raise ValueError(f"k23 plan factory is bound to {ns}, not {n}")
        kw["allow_test"] = bool(kw.get("allow_test")) or bool(allow_test)
        kw["types_path"] = types_path_
        return K23Plan(n, **kw)
    mk.k23_plan_cls = K23Plan
    return mk


def run_plan(ns: str, a, root=None, *, plan_factory=None, sim_check=None) -> int:
    import plan as PL
    import planfirst as PF
    k = E.kind(ns)
    if k == "test":
        energy = a.energy or "on"                                 # test-v1 parity: the C5 meter is mandatory (Astra K23-TOOLING-G2 1)
        if energy == "off":
            if not getattr(a, "energy_waiver", None):
                raise Refused("--energy off in a test namespace: A24.2 needs the Gemma energy of every plan cell (C5); only Arash may waive "
                              "it with --energy-waiver NOTE")
            record_waiver(ns, "plan", a.energy_waiver, root)
    else:
        energy = a.energy or "off"
    if a.ack_pause and k == "test" and (not a.pause_response or not a.outcome_blind or len(a.ack_pause.strip()) < 20):
        raise Refused("a test pause acknowledgement needs --pause-response, --outcome-blind and a note of >= 20 characters (A21 B)")
    PL.load_types = k23_load_types(ns)                          # this process only; plan._prepare reads the module global at call time
    tp = types_path(E.scene_of(ns), root)
    mk = plan_factory or k23_plan_factory(ns, a.allow_test, tp)
    ev = C.Journal(C.journal_path(ns, "events", root))
    ev.append(C.make_record("events", event="k23_plan_settings", phase="plan", namespace=ns, stages=list(K23_STAGES), energy=energy,
                            parallel=a.parallel, types=os.path.basename(tp), runner="k23_run.py", tier="synchronous (approve-all-6 item 3)"))
    kw = dict(stages=list(K23_STAGES), parallel=a.parallel, codex_parallel=1, root=root, allow_test=a.allow_test, plan_factory=mk,
              ollama=a.ollama, allow_replacement=a.allow_replacement, requeue_infra=a.requeue_infra, ack_pause=a.ack_pause,
              steal_lock=a.steal_lock, interrupt_after=a.inject_interrupt, gpu_sample=not a.no_gpu_sample)
    if a.pause_response:
        kw["pause_response"] = a.pause_response
    if a.outcome_blind:
        kw["outcome_blind"] = True
    if sim_check is not None:
        kw["sim_check"] = sim_check
    if energy == "on":
        import energy as EN
        try:
            with EN.measuring(PL.Plan, require=True):
                code = PF.run(ns, **kw)
        except EN.MeterUnavailable as e:
            print(f"REFUSED: the C5 energy meter cannot read both counters ({e})", file=sys.stderr)
            return EXIT["refused"]
        return EN.after_planning(code, ns, root, "")
    try:
        return PF.run(ns, **kw)
    except (PL.PlanAbort, C.TestItemRefused, C.PhaseLocked, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return PL.EXIT["abort"]


def wsl_command(argv: list, root: str | None = None) -> list:
    """The same k23_run invocation (every flag kept) inside WSL, in THIS checkout (Astra K23-TOOLING-G2 3)."""
    repo = E.wsl_path(C.paths(root)["root"])
    return ["wsl", "-d", "Ubuntu-22.04", "--cd", repo, "--", "env", "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA", "DISPLAY=:0",
            "TRACE_ENABLED=0", "/home/arash/thor-env/bin/python", "sim/final_test/matrix/k23_run.py"] + list(argv)


def main(argv=None, root=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="K23 orchestrator (rehearsal / test of Kitchens 2 and 3)")
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--namespace", required=True, choices=E.k23_namespaces("rehearsal") + E.k23_namespaces("test"))
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--inject-interrupt", type=int, default=None, metavar="N")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--energy", choices=("on", "off"), default=None, help="test: on (default, mandatory); rehearsal: off (default) or on")
    ap.add_argument("--energy-waiver", default=None, metavar="NOTE", help="Arash only: --energy off for a test plan, or execute / score with "
                    "incomplete C5 energy coverage (recorded in energy_waivers.jsonl)")
    ap.add_argument("--allow-replacement", action="store_true")
    ap.add_argument("--requeue-infra", action="store_true")
    ap.add_argument("--ack-pause", default=None, metavar="NOTE")
    ap.add_argument("--pause-response", default=None, choices=list(C.PAUSE_RESPONSES))
    ap.add_argument("--outcome-blind", action="store_true")
    ap.add_argument("--no-gpu-sample", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT["usage"] if e.code not in (0, None) else 0
    ns = a.namespace
    if a.phase == "status":
        import matrix_run as MR
        print(json.dumps(dict(MR.status(ns, root), kind=E.kind(ns), scene=E.scene_of(ns)), indent=1))
        return 0
    if a.phase == "context":                                       # A24 1 / 7(i), measured, on the kitchen's REHEARSAL capture
        if E.kind(ns) != "rehearsal":
            print("REFUSED: the context-fit gate is measured on the rehearsal capture", file=sys.stderr)
            return EXIT["refused"]
        if E.order_problems(ns, root):
            print("REFUSED: " + "; ".join(E.order_problems(ns, root)[:3]), file=sys.stderr)
            return EXIT["refused"]
        import context_fit as CF
        import plan as PL
        PL.load_types = k23_load_types(ns)                         # the kitchen's B5 list (context_fit reads plan.load_types)
        return int(CF.main(["--namespace", ns, "--real", "--ollama", a.ollama, "--write"] + (["--root", root] if root else [])))
    try:
        recovery = a.phase == "execute" and any(x in extra for x in ("--resolve-invalid", "--declare-unrecoverable"))
        rep = preflight(a.phase, ns, allow_test=a.allow_test, steal_lock=a.steal_lock, root=root, partial=a.partial, recovery=recovery)
        if not a.dry_run:
            rep["energy"] = energy_gate(a.phase, ns, root, a.energy_waiver)
        if a.phase == "capture":
            margv = capture_argv(ns, extra, allow_test=a.allow_test, steal_lock=a.steal_lock, dry_run=a.dry_run, interrupt=a.inject_interrupt)
        elif a.phase == "execute":
            margv = execute_argv(ns, extra, allow_test=a.allow_test, steal_lock=a.steal_lock, dry_run=a.dry_run, interrupt=a.inject_interrupt,
                                 partial=a.partial)
        elif a.phase == "score":
            ex = _check_extra(extra, SCORE_ALLOWED, "score", ns)
            margv = ["--namespace", ns, "--items", "test" if rep["kind"] == "test" else "dev"] + (["--allow-test"] if a.allow_test else [])
            margv += (["--steal-lock"] if a.steal_lock else []) + (["--partial"] if a.partial else []) + (["--dry-run"] if a.dry_run else [])
            margv += [x for x in ex if x not in ("--steal-lock", "--partial", "--dry-run")]
        else:
            if extra:
                raise Refused(f"plan takes no pass-through options ({extra}); use the k23_run flags")
            margv = None
    except (Refused, C.TestItemRefused, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT["refused"]
    except Exception as e:                                         # noqa: BLE001  (unreadable manifest, corrupt journal ...)
        print(f"REFUSED ({type(e).__name__}): {str(e)[:300]}", file=sys.stderr)
        return EXIT["refused"]
    if a.phase in WSL_PHASES and not rep["platform_ok"]:
        print(json.dumps({"preflight": {k: rep[k] for k in ("phase", "namespace", "kind", "scene", "guard", "needs_wsl", "platform_ok")},
                          "module_argv": margv, "wsl": wsl_command(list(argv) + (["--"] + extra if extra else []), root)}))
        return 0 if a.dry_run else EXIT["platform"]
    if not a.dry_run:
        E.record_order_ok(ns, a.phase, root)                       # A39.2 evidence (FloorPlan15 rehearsal only), before the live step
    if a.phase == "capture":
        import k23_fa_examples as FAX
        if FAX.install(root):                                       # A24.2: Persian-practised examples in English (Arash's decision)
            FAX.record_event(ns, "capture", root)                  # the activation boundary, checked at the agent freeze
            print("k23_fa_examples: active (decision K23-FA-EXAMPLES)", file=sys.stderr)
        import k23_capture_freeze as KCF                           # the Kitchen 1 freeze as superseded by the authorized re-pin (in memory)
        info = KCF.install(root)
        if not a.dry_run:
            KCF.record_event(ns, info, root)                       # the view used, before capture_start (checked at the agent freeze)
        print("k23_capture_freeze: re-pin view " + ("applied (exec_contract.json, ft_v2.json)" if info.get("applied") else
              "NOT applied, capture stays strict: " + "; ".join(info.get("why") or [])[:300]), file=sys.stderr)
        return int(admit_capture(ns).main(margv))
    if a.phase == "execute":
        import execute as X
        import plan as PL
        PL.load_types = k23_load_types(ns)                         # G-iso (A19 (i)) renders the none-arm request with the KITCHEN's B5 list
        return int(X.main(margv))                                  # (execute.py -> plan.none_request_renderer -> plan.load_types; was K1's 49)
    if a.phase == "score":
        import score as S
        return int(S.main(margv, root=root))
    if a.dry_run:
        print(json.dumps({"preflight": "ok", "phase": "plan", "namespace": ns, "stages": list(K23_STAGES), "n_items": rep["guard"]["n"],
                          "types": os.path.basename(types_path(E.scene_of(ns), root)), "type_list_problems": type_list_problems(ns, root)[1]}))
        return 0
    try:
        return int(run_plan(ns, a, root))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT["refused"]


if __name__ == "__main__":
    sys.exit(main())
