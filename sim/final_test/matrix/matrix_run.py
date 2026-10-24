"""Final-test matrix orchestrator (stage M-integrate; contract: SPEC.md s12 "Integrator").

One entry point for the six phases, with the same preflight in front of each:

    python sim/final_test/matrix/matrix_run.py --phase capture|plan|execute|score|monitor|analyze|status
           --namespace NS --items dev|test [--allow-test] [--steal-lock] [--dry-run] [--inject-interrupt N]
           [--wsl] [-- <extra arguments for the phase module>]

Everything after a bare `--` goes to the phase module unchanged (for example `-- --pass core --r 0` for plan, or
`-- --runs 1,2 --worlds canon,reloc` for capture).

Preflight (before anything is dispatched; any failure exits 10 and nothing is touched):
  1. `--items` fits the namespace (test items <-> test-v1; dev items <-> a non-test namespace);
  2. `C.guard_items` over EVERY item of the namespace's split set (so a test item is refused unless the full SPEC s7 guard
     holds: namespace test-v1, `--allow-test`, a valid freeze.json with Arash's check, families_approved.json); the phase
     modules call the same guard again, on every resume;
  3. dependencies: the phases upstream of this one have a `phase_complete` event whose journal sha still matches the file;
  4. the phase lock is free (a lock left by a dead process needs `--steal-lock`, which is logged by the phase module);
  5. an unresolved `abort_invalid` of the execute phase blocks execute/score/analyze;
  6. capture and execute need the simulator: they run in WSL. On Windows this tool prints the exact WSL command and exits 11
     (or runs it with `--wsl`). Plan, score, monitor and analyze run anywhere.

Exit codes: the phase module's own code when it ran; 10 preflight refusal; 11 wrong platform (WSL needed); 12 usage.
`--inject-interrupt N` (capture, plan, execute) stops the phase after N cells so the resume can be rehearsed. Score and analyze
are not interruptible from the outside (each is one pass over journals); their resume is rehearsed by truncating the journal.

Python 3.10 syntax only. No key is read, printed or written here. No test item is touched without the guard.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402  (also puts sim/ on sys.path)

PHASES = ("capture", "plan", "execute", "score", "monitor", "analyze")
WSL_PHASES = ("capture", "execute")                                      # need the simulator
#: the order the pipeline runs in, and what each phase needs complete before it starts.
ORDER = ("capture", "plan", "execute", "score", "analyze")
NEEDS = {"capture": (), "plan": ("capture",), "execute": ("capture", "plan"), "score": ("plan", "execute"), "analyze": ("score",),
         "monitor": ()}
JOURNAL_OF = {"capture": "capture", "plan": "plan", "execute": "exec"}   # phase -> journal kind carrying its phase_complete sha
LOCKED_PHASES = ("capture", "plan", "execute", "score")                  # phases that take a PhaseLock
EXIT = {"refused": 10, "platform": 11, "usage": 12}

WSL_DISTRO = "Ubuntu-22.04"
WSL_PYTHON = "/home/arash/thor-env/bin/python"
WSL_ENV = ("MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA", "DISPLAY=:0", "TRACE_ENABLED=0")


class Refused(RuntimeError):
    pass


def in_wsl() -> bool:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/version", encoding="utf-8") as f:
                return "microsoft" in f.read().lower()
        except OSError:
            return False
    return False


def ns_kind(ns: str) -> str:
    """'test' for the namespace that holds test items, else 'dev' (rehearsal and practice only touch non-test items)."""
    return "test" if C.ns_check(ns)["items"] == "test" else "dev"


def _events(ns: str, root) -> list:
    p = C.journal_path(ns, "events", root)
    return C.Journal(p).rows() if os.path.exists(p) else []


def phase_complete_event(ns: str, phase: str, root=None):
    ev = [e for e in _events(ns, root) if e.get("event") == "phase_complete" and e.get("phase") == phase]
    return ev[-1] if ev else None


def dependency_state(ns: str, phase: str, root=None, partial: bool = False) -> dict:
    """{dep: {'complete': bool, 'unchanged': bool|None, 'why': str}} for every upstream phase."""
    out = {}
    for dep in NEEDS[phase]:
        if dep == "score":
            seal = os.path.join(C.ns_docs(ns, root), "score.seal.json")
            try:
                with open(seal, encoding="utf-8") as f:
                    s = json.load(f)
                sp = C.journal_path(ns, "score", root)
                same = os.path.exists(sp) and C.sha256_file(sp) == s.get("sha256")
                out[dep] = {"complete": bool(s.get("complete")), "unchanged": bool(same),
                            "why": "" if (s.get("complete") and same) else ("score.seal.json says complete=false" if not s.get("complete")
                                                                             else "score.jsonl no longer matches its seal")}
            except (OSError, ValueError):
                out[dep] = {"complete": False, "unchanged": None, "why": "score.seal.json missing"}
            continue
        ev = phase_complete_event(ns, dep, root)
        if ev is None and partial and dep == "plan" and any(e.get("event") in ("stopping_rule", "incomplete_study") for e in _events(ns, root)):
            out[dep] = {"complete": True, "unchanged": True, "why": "", "partial": "recorded stopping rule (incomplete study)"}
            continue
        if ev is None:
            out[dep] = {"complete": False, "unchanged": None, "why": f"no phase_complete event for {dep}"}
            continue
        jp = C.journal_path(ns, JOURNAL_OF[dep], root)
        same = os.path.exists(jp) and (not ev.get("journal_sha") or ev.get("journal_sha") == C.sha256_file(jp))
        out[dep] = {"complete": True, "unchanged": bool(same), "why": "" if same else f"{JOURNAL_OF[dep]}.jsonl changed after its phase_complete"}
    return out


def open_invalid(ns: str, root=None) -> bool:
    """True if the execute phase wrote abort_invalid and no later invalid_resolved covers it."""
    evs = _events(ns, root)
    aborts = [i for i, e in enumerate(evs) if e.get("event") == "abort_invalid"]
    if not aborts:
        return False
    return not any(e.get("event") == "invalid_resolved" for e in evs[aborts[-1]:])


def lock_held(ns: str, phase: str, root=None):
    p = os.path.join(C.ns_work(ns, root), f"{phase}.lock")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return "?"


def preflight(phase: str, ns: str, items: str, *, allow_test=False, steal_lock=False, root=None, check_platform=True, partial=False) -> dict:
    """Raises Refused / C.TestItemRefused. Returns a report dict when every check holds."""
    C.ns_check(ns)
    if phase not in PHASES:
        raise Refused(f"unknown phase {phase!r}")
    if items not in ("dev", "test"):
        raise Refused("--items must be dev or test")
    if items != ns_kind(ns):
        raise Refused(f"--items {items} does not fit namespace {ns} (it holds the {C.ns_check(ns)['items']} set)")
    man = C.load_manifest(root)
    ids = sorted(man[C.ns_check(ns)["items"]])
    guard = C.guard_items(ids, ns, allow_test=allow_test, root=root)      # raises TestItemRefused
    rep = {"phase": phase, "namespace": ns, "items": items, "guard": guard, "deps": {}, "lock": None}
    if phase in ("capture", "plan", "execute", "score", "analyze") and ns == "practice-v1":
        raise Refused("practice-v1 is Gemma practice only (practice_k.py); it has no matrix phases")
    deps = dependency_state(ns, phase, root, partial=partial and phase in ('execute', 'score'))
    rep["deps"] = deps
    bad = [f"{d}: {v['why']}" for d, v in deps.items() if not v["complete"] or v["unchanged"] is False]
    if bad and phase != "monitor":
        raise Refused("upstream phase not ready - " + "; ".join(bad))
    if phase in ("execute", "score", "analyze") and open_invalid(ns, root):
        raise Refused("an invalid execute run aborted the study and is unresolved: write the amendment, then execute.py --resolve-invalid")
    if phase in LOCKED_PHASES:
        held = lock_held(ns, phase, root)
        rep["lock"] = held
        if held is not None and not steal_lock:
            raise Refused(f"phase lock is held ({held[:120]}); if that process is dead, pass --steal-lock (it is logged)")
    rep["needs_wsl"] = phase in WSL_PHASES
    if check_platform and phase in WSL_PHASES and not in_wsl():
        rep["platform_ok"] = False
    else:
        rep["platform_ok"] = True
    return rep


def module_argv(phase: str, ns: str, items: str, *, allow_test=False, steal_lock=False, dry_run=False, interrupt=None, extra=(), root=None, partial=False) -> list:
    """The argv list of the phase module's main()."""
    a = ["--namespace", ns]
    if phase in ("capture", "execute", "score", "monitor"):
        a += ["--items", items]
    if allow_test:
        a.append("--allow-test")
    if steal_lock and phase != "analyze":
        a.append("--steal-lock")
    if dry_run:
        if phase in ("capture", "plan", "execute", "score"):
            a.append("--dry-run")
    if interrupt:
        if phase == "capture":
            a += ["--inject-interrupt", str(int(interrupt))]
        elif phase in ("plan", "execute"):
            a += ["--inject-interrupt", str(int(interrupt))]
    if phase == "monitor" and not any(x in ("--once", "--every") for x in extra):
        a.append("--once")
    if partial and phase in ("execute", "score"):
        a.append("--partial")
    if root is not None and phase in ("capture", "plan", "execute", "monitor"):
        a += ["--root", root]
    return a + list(extra)


def wsl_command(phase: str, argv: list, root=None) -> list:
    """The list to hand to subprocess (no shell). Run from the repo root; the apostrophe in the path never needs quoting."""
    repo = "/mnt/d/Personal/Master's Project"
    script = f"sim/final_test/matrix/{phase}.py"
    return ["wsl", "-d", WSL_DISTRO, "--cd", repo, "--", "env", *WSL_ENV, WSL_PYTHON, script, *argv]


def wsl_env(phase: str, ns: str, margv: list):
    """Environment for the WSL child (subprocess env, never printed or logged). Capture counts tokens with the Anthropic tokenizer, so in test-v1
    the study's key is forwarded through WSLENV by common.wsl_key_env (and `--allow-missing-anthropic`, a Gemma-only coverage check, is refused). Inside WSL
    `localhost` is not Windows, so capture needs an explicit `--ollama http://<windows host ip>:11434`. Returns None when nothing is needed."""
    if phase != "capture":
        return None
    if "--ollama" not in margv:
        raise Refused("capture through WSL needs an explicit `-- --ollama http://<windows host ip>:11434` (localhost is not Windows inside WSL)")
    if ns != "test-v1":
        return None
    if "--allow-missing-anthropic" in margv:
        raise Refused("--allow-missing-anthropic is refused in test-v1: A12 coverage needs every tokenizer")
    try:
        return C.wsl_key_env(os.environ)                                    # the value never passes through this module
    except Exception as e:                                                  # noqa: BLE001
        raise Refused(f"capture in test-v1 needs the study's Anthropic key in the environment (A12 token counting): {type(e).__name__}") from None


def _load(phase: str):
    sys.path.insert(0, HERE)
    if phase == "capture":
        import capture as m
    elif phase == "plan":
        import plan as m
    elif phase == "execute":
        import execute as m
    elif phase == "score":
        import score as m
    elif phase == "monitor":
        import monitor as m
    elif phase == "analyze":
        import analyze as m
    else:
        raise Refused(f"no module for {phase}")
    return m


def energy_wanted(phase: str, argv: list, energy: str = "auto") -> bool:
    """C5 (A24.2 / A33): the plan phase measures Gemma energy -- always in test-v1 (mandatory), with --energy on elsewhere; never on a dry run."""
    if phase != "plan" or "--dry-run" in argv or energy == "off":
        return False
    ns = argv[argv.index("--namespace") + 1] if "--namespace" in argv else None
    return energy == "on" or ns == "test-v1"


def dispatch(phase: str, argv: list, root=None, energy: str = "auto", _meter=None) -> int:
    m = _load(phase)
    if phase == "score":
        return int(m.main(argv, root=root))
    if not energy_wanted(phase, argv, energy):
        return int(m.main(argv))
    import energy as EN                                                     # C5: wraps Plan._call_cell for this launch only (plan.py is pinned)
    ns = argv[argv.index("--namespace") + 1]
    pid = argv[argv.index("--pass-id") + 1] if "--pass-id" in argv else ""
    try:
        with EN.measuring(m.Plan, require=True, meter=_meter) as info:
            fl = (info or {}).get("idle_floor") or {}
            print(f"[energy] C5 meter on ({EN.METHOD_ID}): GPU {((info or {}).get('devices') or {}).get('gpu', {}).get('name')}, idle floor "
                  f"GPU {fl.get('gpu_w')} W / CPU {fl.get('cpu_w')} W; energy rows -> energy.jsonl beside the plan journal", flush=True)
            code = int(m.main(argv))
    except EN.MeterUnavailable as e:
        print(f"REFUSED: the C5 energy meter cannot read both counters ({e}); A24.2 needs measured Gemma energy, so the plan phase "
              "does not start", file=sys.stderr)
        return EXIT["refused"]
    return EN.after_planning(code, ns, root, pid)


def energy_gate(phase: str, ns: str, root=None, waiver: str | None = None) -> dict | None:
    """C5: in test-v1, execute / score / analyze need complete strict energy coverage of the plan journal, or a recorded Arash waiver."""
    if ns != "test-v1" or phase not in ("execute", "score", "analyze"):
        return None
    import energy as EN
    cov = EN.coverage(ns, root)
    if cov.get("complete") is not False:
        return cov
    if not waiver:
        raise Refused(f"C5 energy evidence incomplete ({cov.get('measured')} of {cov.get('gemma_plan_rows')} Gemma plan rows measured): the "
                      "A24.2 Gemma cost would be UNKNOWN. Ask Arash; to proceed anyway he records --energy-waiver NOTE")
    with open(os.path.join(C.ns_docs(ns, root), "energy_waivers.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase, "by": "Arash", "note": waiver[:500],
                             "coverage": {k: cov.get(k) for k in ("gemma_plan_rows", "measured", "refused_interval", "no_energy_row")}}) + "\n")
    return dict(cov, waiver=waiver[:500])


def score_path(ns: str, root=None) -> str:
    return C.journal_path(ns, "score", root)


def status(ns: str, root=None) -> dict:
    """Per-phase state, from the integrity journals and the seal only (never opens outcomes or scores)."""
    out = {"namespace": ns, "phases": {}}
    nxt = None
    for ph in ORDER:
        if ph == "score" or ph == "analyze":
            seal = os.path.join(C.ns_docs(ns, root), "score.seal.json")
            sealed = None
            with contextlib.suppress(OSError, ValueError):
                with open(seal, encoding="utf-8") as f:
                    sealed = json.load(f)
            if ph == "score":
                done = bool(sealed and sealed.get("complete"))
                out["phases"][ph] = {"complete": done, "n": (sealed or {}).get("n")}
            else:
                done = os.path.exists(os.path.join(C.ns_docs(ns, root), "analysis.json"))
                out["phases"][ph] = {"complete": done}
        else:
            ev = phase_complete_event(ns, ph, root)
            jp = C.journal_path(ns, JOURNAL_OF[ph], root)
            n = len(C.Journal(jp).keys()) if os.path.exists(jp) else 0
            out["phases"][ph] = {"complete": ev is not None, "rows": n, "unchanged": (None if ev is None else
                                 (not ev.get("journal_sha") or ev.get("journal_sha") == C.sha256_file(jp)))}
        if nxt is None and not out["phases"][ph]["complete"]:
            nxt = ph
    out["next"] = nxt
    out["invalid_open"] = open_invalid(ns, root)
    out["locks"] = {ph: lock_held(ns, ph, root) for ph in LOCKED_PHASES if lock_held(ns, ph, root) is not None}
    return out


def main(argv=None, root=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="Final-test matrix orchestrator (preflight + dispatch).")
    ap.add_argument("--phase", required=True, choices=PHASES + ("status",))
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--items", choices=("dev", "test"), required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="preflight only (and the module's own dry run where it has one)")
    ap.add_argument("--inject-interrupt", type=int, default=None, metavar="N")
    ap.add_argument("--partial", action="store_true", help="execute / score on an incomplete plan phase; needs a recorded stopping rule (prereg s13)")
    ap.add_argument("--wsl", action="store_true", help="on Windows, run capture/execute through `wsl -d Ubuntu-22.04`")
    ap.add_argument("--print-command", action="store_true", help="print the command line and exit")
    ap.add_argument("--energy", choices=("auto", "on", "off"), default="auto",
                    help="C5 Gemma energy meter for the plan phase: auto = on in test-v1 (mandatory there), off elsewhere")
    ap.add_argument("--energy-waiver", default=None, metavar="NOTE",
                    help="Arash only: proceed to execute / score / analyze in test-v1 although the C5 energy coverage is incomplete (recorded)")
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT["usage"] if e.code not in (0, None) else 0
    root = a.root if a.root is not None else root
    ns = a.namespace or ("test-v1" if a.items == "test" else "rehearsal-v1")
    try:
        if a.phase == "status":
            print(json.dumps(status(ns, root), indent=1))
            return 0
        if a.inject_interrupt and a.phase not in ("capture", "plan", "execute"):
            print(f"REFUSED: --inject-interrupt applies to capture, plan and execute, not {a.phase}", file=sys.stderr)
            return EXIT["usage"]
        rep = preflight(a.phase, ns, a.items, allow_test=a.allow_test, steal_lock=a.steal_lock, root=root, partial=a.partial)
        if not a.dry_run and not a.print_command:
            rep["energy"] = energy_gate(a.phase, ns, root, a.energy_waiver)
    except (Refused, C.TestItemRefused, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT["refused"]
    except Exception as e:                                                 # noqa: BLE001  (unreadable manifest, corrupt journal ...)
        print(f"REFUSED ({type(e).__name__}): {str(e)[:300]}", file=sys.stderr)
        return EXIT["refused"]
    margv = module_argv(a.phase, ns, a.items, allow_test=a.allow_test, steal_lock=a.steal_lock, dry_run=a.dry_run,
                        interrupt=a.inject_interrupt, extra=extra, root=root, partial=a.partial)
    if a.phase == "analyze":
        margv = ["--sealed", score_path(ns, root)] + (["--allow-test"] if a.allow_test else []) + list(extra)
    if a.print_command or (a.dry_run and a.phase in WSL_PHASES and not rep["platform_ok"]):
        print(json.dumps({"preflight": {k: rep[k] for k in ("phase", "namespace", "items", "guard", "needs_wsl", "platform_ok")},
                          "argv": margv, "wsl": wsl_command(a.phase, margv, root) if a.phase in WSL_PHASES else None}))
        return 0
    if not rep["platform_ok"]:
        cmd = wsl_command(a.phase, margv, root)
        if not a.wsl:
            print(f"PLATFORM: {a.phase} needs the simulator and must run in WSL. Run it there, or pass --wsl:\n  " + " ".join(cmd), file=sys.stderr)
            return EXIT["platform"]
        try:
            env = wsl_env(a.phase, ns, margv)
        except Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return EXIT["refused"]
        return subprocess.call(cmd, env=env)                               # list form: no shell, no quoting problems
    if a.phase == "plan" and ns == "test-v1" and a.energy == "off" and not a.dry_run:
        print("REFUSED: --energy off in test-v1: A24.2 needs measured Gemma energy for every plan cell (C5)", file=sys.stderr)
        return EXIT["refused"]
    if a.dry_run and a.phase not in ("capture", "plan", "execute", "score"):
        print(json.dumps({"preflight": "ok", "phase": a.phase, "namespace": ns, "items": a.items, "n_items": rep["guard"]["n"]}))
        return 0
    return dispatch(a.phase, margv, root, energy=a.energy)


if __name__ == "__main__":
    sys.exit(main())
