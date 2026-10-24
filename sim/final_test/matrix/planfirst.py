"""PLAN FIRST, THEN SIMULATE: the orchestrator (stage PF-safety-test; Arash approved 2026-10-12, W5 uses it only if its safety test passed).

    python sim/final_test/matrix/planfirst.py --namespace NS [--stages free,core0,...] [--parallel 4] [--codex-parallel 2]
           [--pass-id ID] [--allow-test] [--ollama URL] [--start-ollama] [-- <extra plan.py arguments>]

ALL planning (Gemma, Anthropic, Codex) is finished before any execution starts. Windows, no simulator. What this adds to `plan.py --parallel`:
  1. PREFLIGHT: refuses to start while a simulator process is running (Gemma must plan with the simulator off: the 8 GB GPU serves one of them),
     and while the namespace already has execution rows (no execution before the planning is complete);
  2. the registered stage order (free: Gemma + gpt-6-sol; core r=0; core r=1,2; FS; curve; relocation; exploratory Codex; A23: Haiku / Sonnet archived), each stage one
     `Plan(...).run()` in parallel mode; ANY non-zero exit (pause, ceiling, abort, quarantine, incomplete) stops the sequence and is returned unchanged;
  3. Gemma is unloaded from the GPU when the Gemma stage ends (`gemma_unloaded` event with the GPU memory before / after);
  4. `plan_first_start` / `plan_first_stage` / `plan_first_complete` events. The complete event is bound to the registered cell manifest (sha of the
     sorted cell keys), the plan journal sha, the spend and the scope: "full" (every cell of the design terminal) or "authorized_partial" (a stopping
     rule was recorded). `execute.py` refuses to start in a namespace that used plan-first unless a complete event matches the CURRENT plan journal.

Exit code: the first non-zero stage code, 8 = preflight refusal, 0 = all stages done.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402
import plan as PL                                                        # noqa: E402

REFUSED = 8
#: stage -> (passes, replicates). The registered order of W5 (SPEC s9): free work first, the core before everything exploratory.
STAGES = {
    "free": (("gemma", "xmain"), None),
    "core0": (("core",), (0,)),
    "core12": (("core",), (1, 2)),
    "fs": (("fs",), None),
    "curve": (("curve",), None),
    "reloc": (("reloc",), None),
    "haiku": (("haiku",), None),          # A23: ARCHIVED stage (no cell in test-v1); only a namespace that opted into H0 / H+M has cells
    "sonnet": (("sonnet",), None),        # A23: ARCHIVED stage, same rule
    "codex": (("codex",), None),
    "gemma": (("gemma",), None),          # dev rehearsals: the Gemma pass alone
    "haiku0": (("haiku",), (0,)),         # dev rehearsals: Haiku, main replicate only
}
REGISTERED_ORDER = ("free", "core0", "core12", "fs", "curve", "reloc", "codex")        # A23: the archived "haiku" / "sonnet" stages left the registered order


def simulator_running(run=subprocess.run, platform=sys.platform):
    """-> (bool, detail). Is an AI2-THOR (thor-Linux64) process alive? From Windows: only if the WSL distro is running AND pgrep finds it."""
    try:
        if platform.startswith("linux"):
            r = run(["pgrep", "-fc", "thor-Linux64"], capture_output=True, text=True, timeout=15)
            n = int((r.stdout or "0").strip() or 0)
            return n > 0, f"pgrep thor-Linux64 -> {n}"
        r = run(["wsl", "-l", "--running"], capture_output=True, timeout=30)
        raw = r.stdout if isinstance(r.stdout, bytes) else str(r.stdout or "").encode()
        try:
            txt = raw.decode("utf-16-le")
        except UnicodeDecodeError:
            txt = raw.decode("utf-8", "replace")
        if "ubuntu-22.04" not in txt.replace("\x00", "").lower():
            return False, "WSL distro Ubuntu-22.04 is not running"
        r = run(["wsl", "-d", "Ubuntu-22.04", "--", "pgrep", "-fc", "thor-Linux64"], capture_output=True, text=True, timeout=60)
        n = int((r.stdout or "0").strip() or 0)
        return n > 0, f"WSL pgrep thor-Linux64 -> {n}"
    except Exception as e:                                       # noqa: BLE001  (cannot tell: refuse, do not guess "off")
        return True, f"could not determine ({type(e).__name__}: {str(e)[:80]}): treated as running"


def exec_rows(ns: str, root=None) -> int:
    p = C.journal_path(ns, "exec", root)
    return len(C.Journal(p).rows()) if os.path.exists(p) else 0


#: BATCH-A29: stages whose passes are Anthropic-only; with --batch they plan through the Message Batches API (plan_batch.BatchPlan)
BATCH_STAGES = ("core0", "core12", "fs", "curve", "reloc", "haiku", "sonnet", "haiku0")


def batch_selecting_factory(batch_kw=None, base_factory=None, batch_cls=None):
    """-> plan_factory: a BatchPlan for the Anthropic-only passes, the ordinary Plan (or `base_factory`) for every other pass."""
    def mk(n, **kw):
        passes = set(kw.get("passes") or ())
        if passes and passes <= {"core", "fs", "curve", "reloc", "haiku", "sonnet"}:
            cls = batch_cls
            if cls is None:
                import plan_batch
                cls = plan_batch.BatchPlan
            return cls(n, **{**kw, **dict(batch_kw or {})})
        return (base_factory or (lambda n_, **k: PL.Plan(n_, **k)))(n, **kw)
    return mk


def run(ns: str, *, stages=REGISTERED_ORDER, parallel: int = 4, codex_parallel: int = 2, pass_id: str = "", root=None, allow_test: bool = False,
        sim_check=simulator_running, out=print, plan_factory=None, now=time.time, batch: bool = False, batch_kw=None, **plan_kw) -> int:
    """Runs the stages in order. `plan_factory(ns, **kw) -> Plan` is injectable (tests). `batch` (BATCH-A29): the Anthropic-only stages
    plan through the Message Batches API; the Gemma / Codex stages are unchanged."""
    if batch:
        plan_factory = batch_selecting_factory(batch_kw, plan_factory)
    C.ns_check(ns)
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; use {', '.join(STAGES)}")
    E = C.Journal(C.journal_path(ns, "events", root))
    sim, detail = sim_check()
    n_exec = exec_rows(ns, root)
    if sim or n_exec:
        why = (f"a simulator process is running ({detail})" if sim else f"{n_exec} execution row(s) already exist in {ns}")
        E.append(C.make_record("events", event="plan_first_refused", phase="plan", namespace=ns, reason=why, pass_id=pass_id or None))
        out(f"REFUSED: plan first needs the simulator off and no execution yet: {why}")
        return REFUSED
    mk = plan_factory or (lambda n, **kw: PL.Plan(n, **kw))
    t0 = now()
    gpu0 = PL.gpu_used_mib() if plan_factory is None else None
    E.append(C.make_record("events", event="plan_first_start", phase="plan", namespace=ns, stages=list(stages), parallel=parallel,
                           codex_parallel=codex_parallel, pass_id=pass_id or None, simulator_check=detail, gpu_mib=gpu0,
                           registered_order=list(REGISTERED_ORDER)))
    for st in stages:
        passes, rs = STAGES[st]
        pl = mk(ns, passes=list(passes), rs=rs, root=root, allow_test=allow_test, parallel=parallel, codex_parallel=codex_parallel,
                pass_id=pass_id, unload_gemma=("gemma" in passes), out=out, **plan_kw)
        code = pl.run()
        E.append(C.make_record("events", event="plan_first_stage", phase="plan", namespace=ns, stage=st, passes=list(passes),
                               code=code, pass_id=pass_id or None))
        if code != PL.EXIT["ok"]:
            E.append(C.make_record("events", event="plan_first_stopped", phase="plan", namespace=ns, stage=st, code=code, pass_id=pass_id or None))
            out(f"plan first stopped at stage {st!r} with exit code {code}")
            return code
    return _complete(ns, root, mk, pass_id, t0, now, out, plan_kw)


def _complete(ns, root, mk, pass_id, t0, now, out, plan_kw) -> int:
    """Binds the completion record to the registered manifest and the immutable plan journal."""
    pl = mk(ns, passes=["gemma"], root=root, pass_id=pass_id, out=out, allow_test=plan_kw.get("allow_test", False))
    pl._prepare()
    keys = sorted(c["key"] for c in pl.all_cells)
    latest = pl.latest
    terminal = [k for k in keys if PL.classify({"key": k}, latest, False) == "done"]
    evs = pl.E.rows()
    stop = [e for e in evs if e.get("event") in ("stopping_rule", "incomplete_study")]
    full = len(terminal) == len(keys)
    scope = "full" if full else ("authorized_partial" if stop else "incomplete")
    led = C.ScopedLedger(pl._ledger_path, pl.ns_cfg["ceilings"], release_stale=False)
    E = pl.E
    rec = E.append(C.make_record(
        "events", event="plan_first_complete", phase="plan", namespace=ns, scope=scope, pass_id=pass_id or None, cells_total=len(keys),
        cells_terminal=len(terminal), manifest_sha=C.sha256_json(keys), journal=os.path.basename(pl.J.path), journal_sha=C.sha256_file(pl.J.path),
        committed_usd=led.committed(), billed_usd=led.billed_scope(), open_reservations=led.open_reservations()["n"],
        wall_s=round(now() - t0, 1), stopping_rule=(stop[-1].get("reason") if stop else None)))
    out(f"plan first complete: scope={scope} cells {len(terminal)}/{len(keys)} spend {rec['billed_usd']:.4f} USD")
    return PL.EXIT["ok"] if scope != "incomplete" else PL.EXIT["incomplete"]


def attempts_report(ns: str, root=None, pass_id: str = "") -> dict:
    """Ledger attempts grouped by the cell they belong to, against the plan journal: the reconciliation Astra asked for (attempt id = ledger row id).
    -> {cells: {tag: {attempts, settled_usd, open_usd, cached}}, totals..., diffs: cells whose journal cost != ledger settled}"""
    lg = C.ScopedLedger(os.path.join(C.ns_work(ns, root), "ledger.db"), C.ns_check(ns)["ceilings"], release_stale=False)
    cells = {}
    for r in lg.rows():
        try:
            tag = (json.loads(r["entry"]) if r["entry"] else {}).get("cell")
        except ValueError:
            tag = None
        if not tag or not tag.startswith(f"{pass_id}:" if pass_id else f"{ns}|"):         # the ledger is shared: count only this pass's cells
            continue
        d = cells.setdefault(tag, {"attempts": 0, "settled_usd": 0.0, "open_usd": 0.0, "cache_replays": 0, "released": 0, "ids": []})
        d["ids"].append(r["id"])
        if r["state"] == "settled":
            d["attempts"] += 1
            d["settled_usd"] = round(d["settled_usd"] + (r["actual"] or 0.0), 6)
        elif r["state"] == "open":
            d["open_usd"] = round(d["open_usd"] + (r["reserved"] or 0.0), 6)
        elif r["state"] == "cache":
            d["cache_replays"] += 1
        elif r["state"] == "released":
            d["released"] += 1
    jp = os.path.join(C.ns_docs(ns, root), f"plan.{pass_id}.jsonl") if pass_id else C.journal_path(ns, "plan", root)
    rows = C.Journal(jp).latest() if os.path.exists(jp) else {}
    prefix = f"{pass_id}:" if pass_id else ""
    diffs = []
    for k, r in rows.items():
        if r.get("scope") == "free":
            continue
        led = cells.get(prefix + k, {"settled_usd": 0.0, "open_usd": 0.0})
        # a cell's journal cost counts only calls that were NOT cache replays; a replayed cell is booked at 0 in the ledger and 0 in the row
        if abs(round(float(r.get("cost_usd") or 0.0), 6) - led["settled_usd"]) > 1e-6 or led["open_usd"] > 0:
            diffs.append({"cell": k, "journal_usd": round(float(r.get("cost_usd") or 0.0), 6), "ledger_settled_usd": led["settled_usd"],
                          "ledger_open_usd": led["open_usd"]})
    return {"cells": cells, "journal_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows.values() if r.get("scope") != "free"), 6),
            "ledger_settled_usd": round(sum(d["settled_usd"] for d in cells.values()), 6),
            "ledger_open_usd": round(sum(d["open_usd"] for d in cells.values()), 6), "diffs": diffs,
            "open_total": lg.open_reservations()["n"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--namespace", required=True, choices=sorted(C.NAMESPACES))
    ap.add_argument("--stages", default=",".join(REGISTERED_ORDER), help="comma list of: " + ", ".join(STAGES))
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--codex-parallel", type=int, default=2)
    ap.add_argument("--pass-id", default="")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--start-ollama", action="store_true")
    ap.add_argument("--allow-replacement", action="store_true")
    ap.add_argument("--ack-pause", default=None, metavar="NOTE", help="plan.py semantics: record the decision that ends a cost-guard pause (dev namespaces; test-v1 also needs --pause-response / --outcome-blind via plan.py)")
    ap.add_argument("--requeue-infra", action="store_true", help="plan.py semantics: re-plan infra cells (at most 2 later sessions)")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--inject-interrupt", type=int, default=None, metavar="N")
    ap.add_argument("--no-gpu-sample", action="store_true")
    ap.add_argument("--attempts-report", action="store_true", help="print the ledger attempts per cell against the journal and exit")
    ap.add_argument("--batch", action="store_true", help="BATCH-A29: plan the Anthropic-only stages through the Message Batches API")
    ap.add_argument("--batch-hold-usd", type=float, default=10.0)
    ap.add_argument("--batch-max-cells", type=int, default=200)
    ap.add_argument("--batch-poll-s", type=float, default=30.0)
    ap.add_argument("--energy", choices=("auto", "on", "off"), default="auto",
                    help="C5 Gemma energy meter: auto = on in test-v1 (mandatory there; off is refused), off elsewhere")
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.attempts_report:
        print(json.dumps(attempts_report(a.namespace, a.root, a.pass_id), indent=1))
        return 0
    import contextlib
    import energy as EN                                                  # C5: the Gemma energy meter for this launch (plan.py is pinned)
    want, why = EN.require_flag(a.namespace, a.energy)
    if why:
        print(f"REFUSED: {why}")
        return REFUSED
    try:
        ctx = EN.measuring(PL.Plan, require=True) if want else contextlib.nullcontext()
        with ctx:
            code = run(a.namespace, stages=[s for s in a.stages.split(",") if s], parallel=a.parallel, codex_parallel=a.codex_parallel,
                       pass_id=a.pass_id, root=a.root, allow_test=a.allow_test, ollama=a.ollama, start_ollama=a.start_ollama,
                       allow_replacement=a.allow_replacement, requeue_infra=a.requeue_infra, ack_pause=a.ack_pause, steal_lock=a.steal_lock, interrupt_after=a.inject_interrupt,
                       gpu_sample=not a.no_gpu_sample, batch=a.batch,
                       batch_kw={"batch_hold_usd": a.batch_hold_usd, "batch_max_cells": a.batch_max_cells, "batch_poll_s": a.batch_poll_s})
    except EN.MeterUnavailable as e:
        print(f"REFUSED: the C5 energy meter cannot read both counters ({e}); A24.2 needs measured Gemma energy")
        return REFUSED
    except (PL.PlanAbort, C.TestItemRefused, C.PhaseLocked, ValueError) as e:
        print(f"ERROR: {e}")
        return PL.EXIT["abort"]
    return EN.after_planning(code, a.namespace, a.root, a.pass_id) if want else code


if __name__ == "__main__":
    sys.exit(main())
