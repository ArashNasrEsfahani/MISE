"""PF-safety-test: the live-evidence helpers for the plan-first safety test (dev items, namespace rehearsal-v2 only; Windows python; no key is read here).

    python sim/final_test/matrix/pf_safety_live.py compare-plans --a serial --b par-fresh [--exact]
    python sim/final_test/matrix/pf_safety_live.py kill-test --pass-id kill [--min-done 8] [--parallel 4]
    python sim/final_test/matrix/pf_safety_live.py stage-exec --pass serial        (archive the exec journals, put plan.serial.jsonl in place as plan.jsonl)
    python sim/final_test/matrix/pf_safety_live.py save-exec --pass serial         (copy exec / outcome / units / guard report into pf/exec-serial/)
    python sim/final_test/matrix/pf_safety_live.py exec-compare --a serial --b par-frozen
    python sim/final_test/matrix/pf_safety_live.py ledger --pass-id par-fresh      (attempts per cell against the journal)

Every command writes its JSON evidence under docs/final-test/matrix/rehearsal-v2/pf/ and prints a short verdict. Nothing here touches a test item:
the namespace check refuses anything but rehearsal-v2.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402
import planfirst as PF                                                   # noqa: E402

NS = "rehearsal-v2"
ROW_KEYS = ("arm", "pres", "r", "status", "first_ok", "repaired", "calls", "request_sha", "scope", "planner", "model")


def pf_dir(root=None) -> str:
    d = os.path.join(C.ns_docs(NS, root), "pf")
    os.makedirs(d, exist_ok=True)
    return d


def _save(name: str, obj: dict, root=None) -> str:
    p = os.path.join(pf_dir(root), name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    return p


def plan_journal(pass_id: str, root=None) -> str:
    return os.path.join(C.ns_docs(NS, root), f"plan.{pass_id}.jsonl")


def _attempts(r: dict) -> list:
    return [(a.get("k"), a.get("stage"), a.get("prompt_sha256"), bool(a.get("valid")), a.get("error_kind") or "") for a in r.get("attempts") or []]


def compare_plans(a: str, b: str, exact: bool = False, root=None, restrict_to_a: bool = False) -> dict:
    """Request bytes and plan validity of two plan passes over the same cells. Request identity: the sha of the rendered prompt of the cell and of every
    attempt (first call and repair) -- the plan phase refuses to send a prompt that differs from the Phase A capture, so equality here means the bytes handed
    to the provider are equal. Validity: status, first_ok, repaired. Plan content is reported as agreement; with `exact` (frozen responses) it must be 100 %."""
    ja, jb = C.Journal(plan_journal(a, root)).latest(), C.Journal(plan_journal(b, root)).latest()
    if restrict_to_a:                                   # b planned more cells (REPEAT replicates) than a: compare the cells a has
        jb = {k: v for k, v in jb.items() if k in ja}
    keys = sorted(set(ja) | set(jb))
    out = {"a": a, "b": b, "cells_a": len(ja), "cells_b": len(jb), "only_a": [k for k in keys if k not in jb], "only_b": [k for k in keys if k not in ja],
           "request_sha_diff": [], "attempt_prompt_diff": [], "status_diff": [], "validity_diff": [], "plan_diff": [], "cost_a": 0.0, "cost_b": 0.0}
    for k in keys:
        if k not in ja or k not in jb:
            continue
        x, y = ja[k], jb[k]
        out["cost_a"] += float(x.get("cost_usd") or 0.0)
        out["cost_b"] += float(y.get("cost_usd") or 0.0)
        if x.get("request_sha") != y.get("request_sha"):
            out["request_sha_diff"].append(k)
        if [t[2] for t in _attempts(x)][:1] != [t[2] for t in _attempts(y)][:1]:
            out["attempt_prompt_diff"].append(k)
        if x.get("status") != y.get("status"):
            out["status_diff"].append(k)
        if (x.get("first_ok"), x.get("repaired")) != (y.get("first_ok"), y.get("repaired")):
            out["validity_diff"].append(k)
        if C.canon(x.get("plan")) != C.canon(y.get("plan")):
            out["plan_diff"].append(k)
    n = len(set(ja) & set(jb))
    out["n_common"] = n
    out["plan_agreement"] = None if not n else round(1 - len(out["plan_diff"]) / n, 4)
    out["validity_agreement"] = None if not n else round(1 - len(set(out["validity_diff"]) | set(out["status_diff"])) / n, 4)
    out["ok_a"] = sum(1 for r in ja.values() if r.get("status") == "ok")
    out["ok_b"] = sum(1 for r in jb.values() if r.get("status") == "ok")
    req_ok = not out["request_sha_diff"] and not out["attempt_prompt_diff"] and not out["only_a"] and not out["only_b"]
    out["request_bytes_identical"] = req_ok
    out["exact_required"] = bool(exact)
    out["pass"] = bool(req_ok and (not exact or (not out["plan_diff"] and not out["status_diff"] and not out["validity_diff"])))
    out["cost_a"], out["cost_b"] = round(out["cost_a"], 6), round(out["cost_b"], 6)
    return out


# ---------------------------------------------------------------------------------------------------------------------- the hard kill
def _open_reserved(ledger_path: str) -> list:
    con = sqlite3.connect(ledger_path, timeout=5)
    try:
        return con.execute("SELECT id, reserved, entry FROM ledger WHERE state='open' AND provider NOT LIKE 'hold@%' AND reserved>0").fetchall()
    finally:
        con.close()


def kill_test(pass_id: str, parallel: int = 4, min_done: int = 8, timeout_s: float = 240.0, root=None) -> dict:
    """Start the plan-first Haiku stage as a separate process, wait until some cells are recorded AND at least one paid call is in flight (an open
    reservation), then kill the process (taskkill /F /T on Windows: no cleanup, no `finally`). Returns what the ledger and journal look like."""
    jp = plan_journal(pass_id, root)
    lp = os.path.join(C.ns_work(NS, root), "ledger.db")
    cmd = [sys.executable, os.path.join(HERE, "planfirst.py"), "--namespace", NS, "--stages", "haiku", "--parallel", str(parallel), "--pass-id", pass_id]
    log = open(os.path.join(pf_dir(root), f"kill-{pass_id}.run1.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    t0, snap = time.time(), None
    try:
        while time.time() - t0 < timeout_s and proc.poll() is None:
            done = len(C.Journal(jp).rows()) if os.path.exists(jp) else 0
            if done >= min_done and os.path.exists(lp):
                rows = _open_reserved(lp)
                if rows:
                    snap = {"recorded_before_kill": done, "open_paid_reservations": len(rows), "reserved_usd": round(sum(r[1] for r in rows), 6),
                            "cells": sorted({(json.loads(r[2]) if r[2] else {}).get("cell") for r in rows if r[2]}), "ids": [r[0] for r in rows]}
                    break
            time.sleep(0.02)
        if snap is not None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.kill()
            proc.wait(timeout=30)
        else:
            proc.kill() if proc.poll() is None else None
    finally:
        log.close()
    after = {"exit_code": proc.returncode, "killed_in_flight": snap is not None, "snapshot_at_kill": snap,
             "recorded_after_kill": len(C.Journal(jp).rows()) if os.path.exists(jp) else 0,
             "open_after": [(r[0], r[1]) for r in _open_reserved(lp)] if os.path.exists(lp) else []}
    _save(f"kill-{pass_id}.run1.json", after, root)
    return after


# ---------------------------------------------------------------------------------------------------------------------- execution comparison
def exec_paths(root=None) -> dict:
    d = C.ns_docs(NS, root)
    return {"plan": C.journal_path(NS, "plan", root), "exec": C.journal_path(NS, "exec", root), "outcome": C.journal_path(NS, "outcome", root),
            "units": os.path.join(d, "units.json")}


def stage_exec(pass_id: str, root=None) -> dict:
    """Archive whatever exec journals exist, then make plan.<pass>.jsonl the plan journal the execute phase reads."""
    P = exec_paths(root)
    arch = os.path.join(pf_dir(root), "exec-archive-" + time.strftime("%H%M%S"))
    moved = []
    for k in ("exec", "outcome", "units"):
        if os.path.exists(P[k]):
            os.makedirs(arch, exist_ok=True)
            shutil.move(P[k], os.path.join(arch, os.path.basename(P[k])))
            moved.append(os.path.basename(P[k]))
    src = plan_journal(pass_id, root)
    if not os.path.exists(src):
        raise SystemExit(f"no {src}")
    shutil.copyfile(src, P["plan"])
    rec = {"pass": pass_id, "plan_journal_sha": C.sha256_file(P["plan"]), "archived": moved, "archive_dir": arch if moved else None}
    _save(f"stage-exec-{pass_id}.json", rec, root)
    return rec


def save_exec(pass_id: str, root=None) -> dict:
    P = exec_paths(root)
    dst = os.path.join(pf_dir(root), f"exec-{pass_id}")
    os.makedirs(dst, exist_ok=True)
    for k in ("exec", "outcome", "units"):
        if os.path.exists(P[k]):
            shutil.copyfile(P[k], os.path.join(dst, os.path.basename(P[k])))
    gdir = os.path.join(HERE, "work")
    reps = sorted(f for f in os.listdir(gdir) if f.startswith("exec_guard.") and f.endswith(".json")) if os.path.isdir(gdir) else []
    for f in reps:
        shutil.move(os.path.join(gdir, f), os.path.join(dst, f))
    return {"saved_to": dst, "guard_reports": reps}


def _steps_key(steps) -> list:
    return [[s.get("action"), str(s.get("object")).split("|")[0] if s.get("object") is not None else None, bool(s.get("ok"))] for s in (steps or [])]


def load_exec(pass_id: str, root=None) -> dict:
    d = os.path.join(pf_dir(root), f"exec-{pass_id}")
    ex = {r["unit"]: r for r in C.Journal(os.path.join(d, "exec.jsonl")).rows() if r.get("run") == 0}
    oc = {r["unit"]: r for r in C.Journal(os.path.join(d, "outcome.sealed.jsonl")).rows() if r.get("run") == 0} if os.path.exists(os.path.join(d, "outcome.sealed.jsonl")) else {}
    out = {}
    for u, r in ex.items():
        cnt = (r.get("audit") or {}).get("counts") or {}
        o = oc.get(u) or {}
        out[u] = {"status": r.get("status"), "goal": o.get("goal_ok"), "steps": _steps_key(o.get("steps")),
                  "audit": {"violations": (r.get("audit") or {}).get("violations"), "oracle": cnt.get("oracle"), "map-build": cnt.get("map-build")},
                  "pres": r.get("pres")}
    return out


def exec_compare(a: str, b: str, noise_max: int = 2, root=None) -> dict:
    """The rule fixed before the run: PASS = status and goal identical on every common unit, violations / oracle / map-build identical (and zero),
    step lists may differ on at most `noise_max` units (simulator noise, the A20 tolerance); the unit sets must be the same and non-empty."""
    ea, eb = load_exec(a, root), load_exec(b, root)
    common = sorted(set(ea) & set(eb))
    out = {"a": a, "b": b, "units_a": len(ea), "units_b": len(eb), "common": len(common), "only_a": sorted(set(ea) - set(eb)), "only_b": sorted(set(eb) - set(ea)),
           "status_diff": [], "goal_diff": [], "audit_diff": [], "steps_diff": [], "nonzero_audit": [], "unmeasured": []}
    for u in common:
        x, y = ea[u], eb[u]
        if x["status"] != y["status"]:
            out["status_diff"].append(u)
        if x["goal"] != y["goal"]:
            out["goal_diff"].append(u)
        if C.canon(x["audit"]) != C.canon(y["audit"]):
            out["audit_diff"].append(u)
        if any(v is None for v in list(x["audit"].values()) + list(y["audit"].values())):
            out["unmeasured"].append(u)
        if any((v or 0) != 0 for v in list(x["audit"].values()) + list(y["audit"].values()) if v is not None):
            out["nonzero_audit"].append(u)
        if C.canon(x["steps"]) != C.canon(y["steps"]):
            out["steps_diff"].append(u)
    reasons = []
    if not common:
        reasons.append("no common unit")
    if out["only_a"] or out["only_b"]:
        reasons.append("the executed unit sets differ")
    for k in ("status_diff", "goal_diff", "audit_diff", "nonzero_audit", "unmeasured"):
        if out[k]:
            reasons.append(f"{k}: {len(out[k])}")
    if len(out["steps_diff"]) > noise_max:
        reasons.append(f"steps differ on {len(out['steps_diff'])} units (> {noise_max})")
    out["steps_noise_units"] = len(out["steps_diff"])
    out["reasons"] = reasons
    out["pass"] = not reasons
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare-plans")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--restrict-a", action="store_true")
    p = sub.add_parser("kill-test")
    p.add_argument("--pass-id", required=True)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--min-done", type=int, default=8)
    for n in ("stage-exec", "save-exec"):
        p = sub.add_parser(n)
        p.add_argument("--pass", dest="pass_id", required=True)
    p = sub.add_parser("exec-compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p = sub.add_parser("ledger")
    p.add_argument("--pass-id", default="")
    a = ap.parse_args(argv)
    if a.cmd == "compare-plans":
        r = compare_plans(a.a, a.b, a.exact, restrict_to_a=a.restrict_a)
        _save(f"compare-plans-{a.a}-vs-{a.b}.json", r)
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list) or len(v) < 6}, indent=1))
        return 0 if r["pass"] else 1
    if a.cmd == "kill-test":
        r = kill_test(a.pass_id, a.parallel, a.min_done)
        print(json.dumps(r, indent=1))
        return 0 if r["killed_in_flight"] else 1
    if a.cmd == "stage-exec":
        print(json.dumps(stage_exec(a.pass_id), indent=1))
        return 0
    if a.cmd == "save-exec":
        print(json.dumps(save_exec(a.pass_id), indent=1))
        return 0
    if a.cmd == "exec-compare":
        r = exec_compare(a.a, a.b)
        _save(f"exec-compare-{a.a}-vs-{a.b}.json", r)
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list) or len(v) < 6}, indent=1))
        return 0 if r["pass"] else 1
    rep = PF.attempts_report(NS, None, a.pass_id)
    _save(f"ledger-{a.pass_id or 'default'}.json", {k: v for k, v in rep.items() if k != "cells"} | {"n_cells": len(rep["cells"])})
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
