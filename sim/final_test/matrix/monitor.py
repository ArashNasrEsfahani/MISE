"""Final-test matrix M8: integrity + spend monitor (SPEC s12). Never opens outcomes.

Reads ONLY the integrity journals (capture, plan, exec, events, monitor), the read-only ledger database, the frozen files
(split, prereg, freeze, families) and, optionally, a call-plan / expected-count json. It never opens `outcome.sealed.jsonl` or
`score.jsonl` (nor lists, hashes or stats them): the allowed journal kinds are a closed list (`MONITORED`) and `_open_guard`
refuses anything else. Outcomes do not exist for it, so it cannot steer a run by them (prereg s13: resume is outcome-blind).

Counters per tick (written to `monitor.jsonl`): honesty violations, decision-path oracle reads, map-build reads (counted, not a
breach), snapshot-sha changes, state-check breaches, blocked/crash/invalid rates, plan infra/ceiling rates, spend per scope
(ledger, billed + committed) against the namespace ceilings and the projected completion, the A15 Opus cost guard, key leaks and
masking hits in the journals, journal corruption, lock steals, and progress per phase.

Exit codes of the CLI: 0 clean, 1 warnings only, 2 integrity breach or test-item refusal, 3 cost guard says pause.

    python sim/final_test/matrix/monitor.py --namespace rehearsal-v1 --items dev --once
    python sim/final_test/matrix/monitor.py --namespace test-v1 --items test --allow-test --every 1200
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

#: the only journals this module may read. 'outcome' and 'score' are deliberately absent (SPEC s12 M8).
MONITORED = ("capture", "plan", "exec", "events", "monitor")
FORBIDDEN_BASENAMES = ("outcome.sealed.jsonl", "score.jsonl", "score.seal.json")
INFRA_WARN_RATE = 0.10          # plan infra / exec crash+blocked rate above this is a warning (not a breach)
MIN_RATE_N = 10                 # rates are only judged from this many terminal rows
SPEND_MISMATCH_USD = 0.05       # |sum of plan-journal cost - ledger billed| above this is a warning (in-flight calls)


class MonitorRefused(RuntimeError):
    pass


def _open_guard(path: str):
    base = os.path.basename(path)
    if base in FORBIDDEN_BASENAMES or base.endswith(".sealed.jsonl"):
        raise MonitorRefused(f"monitor never opens outcomes: {base}")


def _rows(ns: str, kind: str, root, breaches: list) -> list:
    if kind not in MONITORED:
        raise MonitorRefused(f"monitor may not read the {kind!r} journal")
    path = C.journal_path(ns, kind, root)
    _open_guard(path)
    try:
        return C.Journal(path).rows()
    except C.JournalCorrupt as e:
        breaches.append({"code": "journal_corrupt", "journal": kind, "detail": str(e)[-120:]})
        return []


def _latest(rows: list) -> dict:
    d = {}
    for r in rows:
        d[r.get("key")] = r
    return d


def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _audit_counts(audit) -> dict:
    """honest.MetaAudit.summary() shape: {counts: {reason: n}, violations: n, invalid: bool, errors: n}. Tolerant of None."""
    a = audit if isinstance(audit, dict) else {}
    counts = a.get("counts") if isinstance(a.get("counts"), dict) else {}
    viol = a.get("violations")
    viol = len(viol) if isinstance(viol, (list, tuple)) else int(_num(viol))
    return {"violations": viol, "invalid": bool(a.get("invalid")), "oracle": int(_num(counts.get("oracle"))),
            "map_build": int(_num(counts.get("map-build")))}


# ====================================================================================================== counters
def collect(ns: str, root=None) -> dict:
    """Pure read of the integrity journals. -> {counters, breaches, warnings, items}. No writes."""
    breaches, warnings = [], []
    cap, plan, ex, ev = (_rows(ns, k, root, breaches) for k in ("capture", "plan", "exec", "events"))
    items = sorted({C.parse_pres(r["pres"])["item"] for r in cap + plan + ex
                    if isinstance(r.get("pres"), str) and r["pres"].count(":") == 2})

    # ---- capture
    cap_l = _latest(cap)
    cap_cnt = {"rows": len(cap), "keys": len(cap_l), "masking_hits": 0, "empty_summary": 0}
    for r in cap_l.values():
        blocks = r.get("blocks") if isinstance(r.get("blocks"), dict) else {}
        texts = [str(r.get("instruction") or "")] + [str(b.get("text") or "") for b in blocks.values() if isinstance(b, dict)]
        if any(C.mask_problems(t) for t in texts):
            cap_cnt["masking_hits"] += 1
        if not r.get("summary"):
            cap_cnt["empty_summary"] += 1
    if cap_cnt["masking_hits"]:
        breaches.append({"code": "masking_hit", "n": cap_cnt["masking_hits"]})

    # ---- plan
    pl_l = _latest(plan)
    st = {s: 0 for s in C.PLAN_STATUS}
    for r in pl_l.values():
        st[r.get("status")] = st.get(r.get("status"), 0) + 1
    by_scope, billed_calls, opus_calls, opus_usd = {}, 0, 0, 0.0
    for r in plan:                                          # every row is a spend event, including re-queued sessions
        usd = _num(r.get("cost_usd"))
        sc = r.get("scope") or "free"
        by_scope[sc] = round(by_scope.get(sc, 0.0) + usd, 6)
        n_billed = max(0, int(_num(r.get("calls")) - _num(r.get("cached_calls"))))
        if usd > 0:
            billed_calls += n_billed
            if "opus" in str(r.get("model") or "").lower():
                opus_calls += n_billed
                opus_usd += usd
    n_term = sum(v for k, v in st.items() if k in ("ok", "rejected", "infra"))
    plan_cnt = {"rows": len(plan), "keys": len(pl_l), "status": st,
                "infra_rate": round(st.get("infra", 0) / n_term, 4) if n_term else 0.0,
                "ceiling_pending": st.get("ceiling", 0), "journal_cost_usd": by_scope, "billed_calls": billed_calls,
                "opus_calls": opus_calls, "opus_usd": round(opus_usd, 6)}
    if n_term >= MIN_RATE_N and plan_cnt["infra_rate"] > INFRA_WARN_RATE:
        warnings.append({"code": "plan_infra_rate", "rate": plan_cnt["infra_rate"], "n": n_term})

    # ---- exec (integrity only)
    ex_l = _latest(ex)
    est = {}
    viol = oracle = mapb = invalid_audit = snap_changes = check_breaches = 0
    for r in ex:                                            # all attempts count for integrity, not only the latest
        a = _audit_counts(r.get("audit"))
        viol += a["violations"]
        oracle += a["oracle"]
        mapb += a["map_build"]
        invalid_audit += 1 if a["invalid"] else 0
        b, af = r.get("snapshot_sha_before"), r.get("snapshot_sha_after")
        if b and af and b != af:
            snap_changes += 1
        sc = r.get("state_check") if isinstance(r.get("state_check"), dict) else {}
        if sc and sc.get("ok") is False:
            check_breaches += 1
    for r in ex_l.values():
        est[r.get("status")] = est.get(r.get("status"), 0) + 1
    n_exec_term = sum(est.values())
    bad = est.get("crash", 0) + est.get("blocked_state", 0) + est.get("invalid", 0)
    ex_cnt = {"rows": len(ex), "keys": len(ex_l), "status": est, "violations": viol, "oracle_reads": oracle,
              "map_build_reads": mapb, "invalid_audits": invalid_audit, "snapshot_sha_changes": snap_changes,
              "state_check_breaches": check_breaches, "bad_rate": round(bad / n_exec_term, 4) if n_exec_term else 0.0}
    if viol:
        breaches.append({"code": "honesty_violation", "n": viol})
    if oracle:
        breaches.append({"code": "decision_path_oracle_read", "n": oracle})
    if invalid_audit or est.get("invalid"):
        breaches.append({"code": "invalid_episode", "n": max(invalid_audit, est.get("invalid", 0))})
    if snap_changes:
        breaches.append({"code": "snapshot_sha_changed", "n": snap_changes})
    if check_breaches:
        warnings.append({"code": "state_check_breach", "n": check_breaches})      # blocked + re-run is by design (prereg A14)
    if n_exec_term >= MIN_RATE_N and ex_cnt["bad_rate"] > INFRA_WARN_RATE:
        warnings.append({"code": "exec_bad_rate", "rate": ex_cnt["bad_rate"], "n": n_exec_term})

    # ---- events
    ec = {}
    for r in ev:
        ec[r.get("event")] = ec.get(r.get("event"), 0) + 1
    ev_cnt = {"rows": len(ev), "by_event": ec,
              "phase_complete": sorted({str(r.get("phase")) for r in ev if r.get("event") == "phase_complete"})}
    if ec.get("abort_invalid"):
        breaches.append({"code": "abort_invalid", "n": ec["abort_invalid"]})
    for name in ("lock_stolen", "block_refused", "pause_cost_guard"):
        if ec.get(name):
            warnings.append({"code": name, "n": ec[name]})

    # ---- a completed phase's journal changed afterwards (the phase records its own sha; the convention is theirs)
    for r in ev:
        if r.get("event") == "phase_complete" and r.get("journal_sha") and r.get("phase") in ("capture", "plan", "exec"):
            kind = r["phase"]
            p = C.journal_path(ns, kind, root)
            _open_guard(p)
            if os.path.exists(p) and C.sha256_file(p) != r["journal_sha"]:
                warnings.append({"code": "journal_sha_differs_from_phase_complete", "phase": kind})

    # ---- key leak over the integrity journals (raw text; the sealed files are never read)
    leaks = 0
    for kind in ("capture", "plan", "exec", "events"):
        p = C.journal_path(ns, kind, root)
        _open_guard(p)
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                if C.key_leak(f.read()):
                    leaks += 1
    if leaks:
        breaches.append({"code": "key_leak", "journals": leaks})

    counters = {"capture": cap_cnt, "plan": plan_cnt, "exec": ex_cnt, "events": ev_cnt, "key_leaks": leaks}
    return {"counters": counters, "breaches": breaches, "warnings": warnings, "items": items}


# ====================================================================================================== spend
STALE_OPEN_WARN_USD = 1.0       # open per-call reservations beyond this are shown as a warning (they count against the ceilings but were never billed)


def ledger_spend(ns: str, root=None) -> dict:
    """Read-only look at the namespace ledger. Never uses C.open_ledger (it would release a live writer's holds)."""
    path = os.path.join(C.ns_work(ns, root), "ledger.db")
    out = {"present": os.path.exists(path), "billed": {}, "committed": {}, "holds_open_usd": 0.0, "open_reservations_n": 0,
           "open_reservations_usd": 0.0}
    if not out["present"]:
        return out
    uri = "file:" + path.replace("\\", "/").replace("?", "%3f") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            rows = con.execute("SELECT provider, state, reserved, actual FROM ledger").fetchall()
        finally:
            con.close()
    except sqlite3.Error as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return out
    for prov, state, reserved, actual in rows:
        prov = str(prov)
        scope = prov.split("@", 1)[1] if "@" in prov else "free"
        if prov.startswith("hold@"):
            if state == "open":
                out["holds_open_usd"] = round(out["holds_open_usd"] + _num(reserved), 6)
                out["committed"][scope] = round(out["committed"].get(scope, 0.0) + _num(reserved), 6)
        elif state == "settled":
            out["billed"][scope] = round(out["billed"].get(scope, 0.0) + _num(actual), 6)
            out["committed"][scope] = round(out["committed"].get(scope, 0.0) + _num(actual), 6)
        elif state == "open":
            out["committed"][scope] = round(out["committed"].get(scope, 0.0) + _num(reserved), 6)
            out["open_reservations_n"] += 1                       # a per-call reservation that never resolved (ambiguous failure / killed run)
            out["open_reservations_usd"] = round(out["open_reservations_usd"] + _num(reserved), 6)
    return out


def spend_report(ns: str, plan_cnt: dict, prev_checked_at: int = 0, callplan=None, root=None) -> dict:
    ceil = C.NAMESPACES[ns]["ceilings"]
    led = ledger_spend(ns, root)
    breaches, warnings = [], []
    billed = led["billed"]
    spent_total = round(sum(v for s, v in billed.items() if s != "free"), 6)
    core = round(billed.get("core", 0.0), 6)
    if core > ceil["core"] + 1e-9:
        breaches.append({"code": "core_ceiling_exceeded", "core": core, "ceiling": ceil["core"]})
    if spent_total > ceil["total"] + 1e-9:
        breaches.append({"code": "total_ceiling_exceeded", "total": spent_total, "ceiling": ceil["total"]})
    unscoped = sorted(s for s, v in billed.items() if s not in ("core", "other", "free") and v > 0)
    if unscoped:
        breaches.append({"code": "paid_call_without_scope", "scopes": unscoped})
    jc = plan_cnt.get("journal_cost_usd", {})
    j_paid = round(sum(v for k, v in jc.items() if k != "free"), 6)
    if abs(j_paid - spent_total) > SPEND_MISMATCH_USD:
        warnings.append({"code": "journal_ledger_cost_differs", "journal_usd": j_paid, "ledger_usd": spent_total})
    cp = callplan if isinstance(callplan, dict) else {}
    remaining = {k: int(v) for k, v in (cp.get("remaining") or {}).items()}
    mean_cost = {k: float(v) for k, v in (cp.get("mean_cost") or {}).items()}
    guard = C.cost_guard(opus_calls=int(plan_cnt.get("opus_calls", 0)), opus_usd=float(plan_cnt.get("opus_usd", 0.0)),
                         checked_at=int(prev_checked_at), spent=spent_total, remaining=remaining, mean_cost=mean_cost,
                         ceiling=float(ceil["total"]))
    if led.get("open_reservations_usd", 0.0) > STALE_OPEN_WARN_USD:
        warnings.append({"code": "stale_open_reservations", "n": led["open_reservations_n"], "usd": led["open_reservations_usd"],
                         "action": "Arash can release them: plan.py --reconcile-open \"<note>\" (audited)"})
    return {"ceilings": ceil, "ledger_present": led["present"], "billed": billed, "committed": led["committed"],
            "holds_open_usd": led["holds_open_usd"], "open_reservations_n": led.get("open_reservations_n", 0),
            "open_reservations_usd": led.get("open_reservations_usd", 0.0), "spent_paid": spent_total, "projected": guard["projected"],
            "projection_inputs": "callplan" if remaining else "none (spent only)", "cost_guard_pause": guard["pause"],
            "cost_guard_reasons": guard["reasons"], "opus_checked_at": guard["checked_at"],
            "breaches": breaches, "warnings": warnings}


# ====================================================================================================== frozen hashes
def frozen_report(ns: str, root=None) -> dict:
    P = C.paths(root)
    breaches = []
    try:
        C.load_manifest(root)
    except Exception as e:                                  # noqa: BLE001
        breaches.append({"code": "split_manifest_changed", "detail": str(e)[:100]})
    fz = C.freeze_status(root)
    if ns == "test-v1" and not fz["ok"]:
        breaches.append({"code": "freeze_invalid_during_test_run", "reasons": fz["reasons"][:4]})
    return {"split_ok": not breaches, "freeze_present": os.path.exists(P["freeze"]), "freeze_ok": fz["ok"],
            "freeze_reasons": fz["reasons"][:6], "breaches": breaches}


# ====================================================================================================== progress
PHASE_JOURNAL = {"capture": "capture", "plan": "plan", "execute": "exec"}


def progress(counters: dict, expected=None) -> dict:
    exp = expected if isinstance(expected, dict) else {}
    done_ev = set(counters["events"]["phase_complete"])
    out = {}
    for ph, kind in PHASE_JOURNAL.items():
        done = counters[kind]["keys"]
        e = exp.get(ph) or exp.get(kind)
        out[ph] = {"done": done, "expected": e, "fraction": round(done / e, 4) if e else None,
                   "complete_event": bool(done_ev & {ph, kind})}
    return out


def _prev_tick(ns: str, root):
    rows = C.Journal(C.journal_path(ns, "monitor", root)).rows()
    return rows[-1] if rows else None


# ====================================================================================================== one tick
def tick_once(ns: str, *, items_mode: str = "dev", allow_test: bool = False, root=None, callplan=None, expected=None,
              write: bool = True, steal_lock: bool = False) -> dict:
    C.ns_check(ns)
    if items_mode not in ("dev", "test"):
        raise ValueError("items must be dev or test")
    res = {"namespace": ns, "items_mode": items_mode, "refused": None, "breaches": [], "warnings": []}
    got = collect(ns, root)
    # ---- the SPEC guard: test items refused unless it holds (for the items seen in the journals AND for --items test)
    try:
        ids = list(got["items"])
        if items_mode == "test":
            ids = sorted(set(ids) | set(C.load_manifest(root)["test"]))
        C.guard_items(ids, ns, allow_test=allow_test, root=root)
    except C.TestItemRefused as e:
        res["refused"] = str(e)[:400]
        res["breaches"].append({"code": "test_item_refused", "detail": res["refused"][:200]})
    except Exception as e:                                  # noqa: BLE001 (split manifest unreadable etc.)
        res["refused"] = f"guard error {type(e).__name__}: {str(e)[:120]}"
        res["breaches"].append({"code": "guard_error", "detail": res["refused"]})
    if res["refused"] is not None:
        res["exit_code"] = 2
        res["counters"] = None
        return res                                          # a refused run reads and writes nothing further

    counters = got["counters"]
    prev = _prev_tick(ns, root)
    prev_c = (prev or {}).get("counters") or {}
    sp = spend_report(ns, counters["plan"], int((prev_c.get("spend") or {}).get("opus_checked_at", 0)), callplan, root)
    fr = frozen_report(ns, root)
    prog = progress(counters, expected)
    res["breaches"] += got["breaches"] + sp["breaches"] + fr["breaches"]
    res["warnings"] += got["warnings"] + sp["warnings"]
    incomplete = any(not p["complete_event"] and (p["fraction"] is None or p["fraction"] < 1) for p in prog.values())
    pv = prev_c.get("progress") or {}
    stalled = bool(prev and incomplete and pv and all((pv.get(k) or {}).get("done") == prog[k]["done"] for k in prog)
                   and any(prog[k]["done"] for k in prog))
    if stalled:
        res["warnings"].append({"code": "no_progress_since_last_tick"})
    counters = {**counters, "spend": sp, "frozen": {k: v for k, v in fr.items() if k != "breaches"}, "progress": prog,
                "stalled": stalled}
    res["counters"] = counters
    res["exit_code"] = 2 if res["breaches"] else 3 if sp["cost_guard_pause"] else 1 if res["warnings"] else 0
    if write:
        with C.PhaseLock(ns, "monitor", root, steal=steal_lock):
            j = C.Journal(C.journal_path(ns, "monitor", root))
            n = len(j.rows()) + 1
            j.append(C.make_record("monitor", tick=n, counters=counters, namespace=ns, items_mode=items_mode,
                                   breaches=res["breaches"], warnings=res["warnings"], exit_code=res["exit_code"]))
        res["tick"] = n
    return res


def _fmt(res: dict) -> str:
    c = res.get("counters") or {}
    if res.get("refused"):
        return f"[monitor] REFUSED: {res['refused'][:200]}"
    sp, ex, pl = c.get("spend", {}), c.get("exec", {}), c.get("plan", {})
    prog = " ".join(f"{k}={v['done']}" + (f"/{v['expected']}" if v["expected"] else "") for k, v in c.get("progress", {}).items())
    return (f"[monitor] {res['namespace']} tick {res.get('tick', '-')} exit {res['exit_code']} | {prog} | violations "
            f"{ex.get('violations')} oracle {ex.get('oracle_reads')} snap-changes {ex.get('snapshot_sha_changes')} infra "
            f"{pl.get('infra_rate')} | spend paid {sp.get('spent_paid')} proj {sp.get('projected')} ceilings {sp.get('ceilings')} "
            f"| breaches {[b['code'] for b in res['breaches']]} warnings {[w['code'] for w in res['warnings']]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="integrity + spend monitor (never opens outcomes)")
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--items", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true")
    g.add_argument("--every", type=int, metavar="SECONDS", help="loop (1200 = every 20 min); stops on a breach or Ctrl-C")
    ap.add_argument("--callplan", help="json {remaining:{model:n}, mean_cost:{model:usd}} (final_test_callplan.py --json)")
    ap.add_argument("--expected", help="json {capture:n, plan:n, execute:n} cell counts")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", help="repo root override (tests)")
    a = ap.parse_args(argv)

    def load(p):
        if not p:
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    while True:
        res = tick_once(a.namespace, items_mode=a.items, allow_test=a.allow_test, root=a.root, callplan=load(a.callplan),
                        expected=load(a.expected), write=not a.no_write, steal_lock=a.steal_lock)
        print(json.dumps(res, ensure_ascii=False, default=str) if a.json else _fmt(res), flush=True)
        if not a.every or res["exit_code"] >= 2:
            return res["exit_code"]
        time.sleep(max(1, a.every))


if __name__ == "__main__":
    sys.exit(main())
