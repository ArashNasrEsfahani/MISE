"""Selftest for sim/final_test/matrix/monitor.py (stage M8). Offline, synthetic data in a temp root. No simulator/network/API.

    python sim/_ftm_monitor_selftest.py
"""
from __future__ import annotations

import ast
import builtins
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                   # noqa: E402
import monitor as M                                  # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


REAL = C.ROOT
TMP = tempfile.mkdtemp(prefix="ftm_monitor_")
ROOT = os.path.join(TMP, "repo")
os.makedirs(os.path.join(ROOT, "docs", "final-test"))
for f in ("split_manifest.json", "preregistration.md"):
    shutil.copy(os.path.join(REAL, "docs", "final-test", f), os.path.join(ROOT, "docs", "final-test", f))
MAN = C.load_manifest(REAL)
DEV = MAN["dev_pilot"][:3]
TEST = MAN["test"][:2]
NS = "rehearsal-v1"
SENT = "SENTINEL_OUTCOME_DO_NOT_READ"


def J(kind, ns=NS):
    return C.Journal(C.journal_path(ns, kind, ROOT))


AUD0 = {"episode": "e", "counts": {"pose": 5, "map-build": 7, "scoring": 0, "oracle": 0}, "violations": 0, "invalid": False,
        "errors": 0}


def cap_rec(item, lang="en", instruction="Open the fridge", **kw):
    p = C.pres_id(item, lang)
    return C.make_record("capture", key=C.cell_id(NS, "capture", p, "c1"), namespace=NS, pres=p, item=item, lang=lang, level="L1",
                         world="canon", capture_run=1, instruction=instruction, holding=None, summary=["Fridge seen"],
                         summary_sha="x", blocks={"none": {"text": "", "ids": [], "real_tokens": 0, "budget": 0, "empty": True,
                                                           "coverage": None}}, requests={}, state_ref="states/x.json",
                         world_hash="h", audit=AUD0, **kw)


def plan_rec(item, arm="G0", r=0, status="ok", usd=0.0, scope="free", model="gemma3:4b", calls=1, cached=0):
    p = C.pres_id(item, "en")
    return C.make_record("plan", key=C.cell_id(NS, "plan", arm, p, f"r{r}"), namespace=NS, arm=arm, block="main", pres=p, r=r,
                         session=0, request_sha="s", status=status, plan=[], first_plan=[], first_ok=True, repaired=False,
                         calls=calls, cached_calls=cached, tokens_in=1, tokens_out=1, cost_usd=usd, model=model, scope=scope)


def exec_rec(item, i, status="done", audit=None, sha=("a", "a"), ok=True):
    p = C.pres_id(item, "en")
    return C.make_record("exec", key=f"{NS}|exec|u{i}|run0", namespace=NS, unit=f"u{i}", run=0, attempt=1, pres=p, status=status,
                         state_check={"ok": ok, "breaches": [] if ok else ["x"]}, audit=audit or AUD0, prims=10, wall_s=1.0,
                         snapshot_sha_before=sha[0], snapshot_sha_after=sha[1])


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def codes(res, k="breaches"):
    return {b["code"] for b in res[k]}


# ---------------------------------------------------------------------------------------------------- static
src = open(M.__file__, encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok310 = True
except SyntaxError:
    ok310 = False
check("monitor.py parses as python 3.10", ok310)
check("MONITORED excludes outcome and score", "outcome" not in M.MONITORED and "score" not in M.MONITORED)
code_only = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith(("#", '"""')))
check("no journal_path('outcome'/'score') call in monitor", 'journal_path(ns, "outcome"' not in code_only
      and 'journal_path(ns, "score"' not in code_only and "'outcome'" not in code_only.replace("'outcome' and 'score'", ""))
check("monitor imports no simulator / planner module", not any(f"import {m}" in src for m in ("ai2thor", "honest", "webapp", "planners")))
check("_rows refuses outcome / score kinds", raises(lambda: M._rows(NS, "outcome", ROOT, []), M.MonitorRefused)
      and raises(lambda: M._rows(NS, "score", ROOT, []), M.MonitorRefused))
check("_open_guard refuses sealed basenames", raises(lambda: M._open_guard("/x/outcome.sealed.jsonl"), M.MonitorRefused)
      and raises(lambda: M._open_guard("/x/score.jsonl"), M.MonitorRefused))

# ---------------------------------------------------------------------------------------------------- empty namespace
r0 = M.tick_once(NS, root=ROOT)
check("empty namespace: clean tick, exit 0, tick 1 written", r0["exit_code"] == 0 and r0.get("tick") == 1 and not r0["breaches"],
      (r0["breaches"], r0["warnings"]))
check("empty: ledger absent reported, spend 0", r0["counters"]["spend"]["ledger_present"] is False
      and r0["counters"]["spend"]["spent_paid"] == 0)
check("monitor record carries the envelope and no outcome keys",
      J("monitor").rows()[0]["schema"] == "ftm.monitor/1" and not C.forbidden_keys(J("monitor").rows()[0], C.OUTCOME_KEYS | C.GOLD_KEYS))

# ---------------------------------------------------------------------------------------------------- poisoned outcome files
for kind in ("outcome", "score"):
    p = C.journal_path(NS, kind, ROOT)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"key": "k", "goal_ok": True, "note": SENT}) + "\n")
with open(os.path.join(os.path.dirname(C.journal_path(NS, "score", ROOT)), "score.seal.json"), "w") as f:
    f.write(SENT)
J("capture").append(cap_rec(DEV[0]))
J("plan").append(plan_rec(DEV[0]))
J("exec").append(exec_rec(DEV[0], 1))

_orig = {"open": builtins.open, "io": io.open, "os_open": os.open, "stat": os.stat, "listdir": os.listdir,
         "scandir": os.scandir}
touched = []


def _bad(p):
    b = os.path.basename(str(p)) if isinstance(p, (str, bytes, os.PathLike)) else ""
    b = b.decode() if isinstance(b, bytes) else b
    if b in M.FORBIDDEN_BASENAMES or b.endswith(".sealed.jsonl"):
        touched.append(b)
        raise AssertionError(f"POISON: monitor touched {b}")


def mk(fn):
    def w(p, *a, **k):
        _bad(p)
        return fn(p, *a, **k)
    return w


builtins.open, io.open, os.open, os.stat = mk(_orig["open"]), mk(_orig["io"]), mk(_orig["os_open"]), mk(_orig["stat"])
os.listdir, os.scandir = mk(_orig["listdir"]), mk(_orig["scandir"])
try:
    rp = M.tick_once(NS, root=ROOT)
finally:
    builtins.open, io.open, os.open, os.stat = _orig["open"], _orig["io"], _orig["os_open"], _orig["stat"]
    os.listdir, os.scandir = _orig["listdir"], _orig["scandir"]
check("poisoned outcome/score files exist and the monitor never touched them", not touched and rp["exit_code"] == 0, touched)
check("sentinel never reaches the monitor journal", SENT not in open(C.journal_path(NS, "monitor", ROOT), encoding="utf-8").read())
check("progress counts per phase", rp["counters"]["progress"]["capture"]["done"] == 1 and rp["counters"]["progress"]["execute"]["done"] == 1)

# ---------------------------------------------------------------------------------------------------- integrity breaches
base = M.collect(NS, ROOT)
check("clean exec: 0 violations / oracle; map-build reads counted, not a breach",
      base["counters"]["exec"]["violations"] == 0 and base["counters"]["exec"]["map_build_reads"] == 7 and not base["breaches"])
J("exec").append(exec_rec(DEV[0], 2, audit={**AUD0, "violations": 1, "invalid": True}))
J("exec").append(exec_rec(DEV[1], 3, audit={**AUD0, "counts": {"oracle": 2}}))
J("exec").append(exec_rec(DEV[1], 4, sha=("a", "b")))
J("exec").append(exec_rec(DEV[1], 5, ok=False, status="blocked_state"))
r1 = M.tick_once(NS, root=ROOT)
check("violation, oracle read, invalid episode, snapshot change are breaches; exit 2",
      {"honesty_violation", "decision_path_oracle_read", "invalid_episode", "snapshot_sha_changed"} <= codes(r1) and r1["exit_code"] == 2,
      codes(r1))
check("state-check breach is a warning, counted", "state_check_breach" in codes(r1, "warnings")
      and r1["counters"]["exec"]["state_check_breaches"] == 1)
J("events").append(C.make_record("events", event="abort_invalid", phase="exec"))
J("events").append(C.make_record("events", event="lock_stolen", phase="exec"))
r1b = M.collect(NS, ROOT)
check("abort_invalid is a breach, lock_stolen a warning", "abort_invalid" in codes(r1b) and "lock_stolen" in codes(r1b, "warnings"))

# masking hit in a capture record, and a key-like string
J("capture").append(cap_rec(DEV[2], instruction="Open FloorPlan1 fridge"))
r2 = M.collect(NS, ROOT)
check("masking hit in capture is a breach", "masking_hit" in codes(r2))
with open(C.journal_path(NS, "events", ROOT), "a", encoding="utf-8") as f:
    f.write(json.dumps({"event": "note", "phase": "x", "text": "sk-ant-api03-" + "A" * 40}) + "\n")
check("key-like string in a journal is a breach", "key_leak" in codes(M.collect(NS, ROOT)))
rk = M.tick_once(NS, root=ROOT, write=False)
check("breach text never echoes the key", "AAAAAAAAAA" not in json.dumps(rk, default=str))

# torn / corrupt journal
with open(C.journal_path(NS, "plan", ROOT), "ab") as f:
    f.write(b'{"torn": ')
check("torn last line tolerated", "journal_corrupt" not in codes(M.collect(NS, ROOT)))
shutil.copy(C.journal_path(NS, "exec", ROOT), C.journal_path(NS, "exec", ROOT) + ".bak")
with open(C.journal_path(NS, "exec", ROOT), "ab") as f:
    f.write(b"\nBROKEN LINE\n" + json.dumps(exec_rec(DEV[0], 9)).encode() + b"\n")
check("corrupt middle line is a breach (fail fast)", "journal_corrupt" in codes(M.collect(NS, ROOT)))
shutil.copy(C.journal_path(NS, "exec", ROOT) + ".bak", C.journal_path(NS, "exec", ROOT))

# ---------------------------------------------------------------------------------------------------- error rates
NS2 = "practice-v1"
for i in range(12):
    st = "infra" if i < 3 else "ok"
    p = C.pres_id(MAN["practice_pool"][i], "en")
    C.Journal(C.journal_path(NS2, "plan", ROOT)).append(
        C.make_record("plan", key=f"k{i}", namespace=NS2, arm="G0", block="main", pres=p, r=0, session=0, request_sha="s",
                      status=st, plan=[], first_plan=[], first_ok=True, repaired=False, calls=1, cached_calls=0, tokens_in=1,
                      tokens_out=1, cost_usd=0.0, model="gemma3:4b", scope="free"))
rr = M.collect(NS2, ROOT)
check("infra rate 3/12 = 0.25 is a warning, not a breach", rr["counters"]["plan"]["infra_rate"] == 0.25
      and "plan_infra_rate" in codes(rr, "warnings") and not rr["breaches"])

# ---------------------------------------------------------------------------------------------------- spend vs ledger
NSS = "rehearsal-v1"
os.makedirs(C.ns_work(NSS, ROOT), exist_ok=True)
led = C.open_ledger(NSS, ROOT)
with led.block("core", 0.5):
    rid = led.reserve("anthropic", 0.1)
    led.settle(rid, 0.08, {"provider": "anthropic@core", "cost_usd": 0.08})
with led.block("other", 0.3):
    rid = led.reserve("anthropic", 0.1)
    led.settle(rid, 0.05, {"provider": "anthropic@other", "cost_usd": 0.05})
sp = M.ledger_spend(NSS, ROOT)
check("read-only ledger: billed per scope", sp["billed"] == {"core": 0.08, "other": 0.05} and sp["holds_open_usd"] == 0.0, sp)
# a live writer's open hold must survive a monitor tick (C.open_ledger would release it)
live = C.open_ledger(NSS, ROOT)
cm = live.block("core", 0.2)
cm.__enter__()
before = M.ledger_spend(NSS, ROOT)["holds_open_usd"]
M.tick_once(NSS, root=ROOT, write=False)
after = M.ledger_spend(NSS, ROOT)["holds_open_usd"]
cm.__exit__(None, None, None)
check("monitor does not release a live writer's hold", before == 0.2 and after == 0.2, (before, after))
rs = M.tick_once(NSS, root=ROOT, write=False)
check("spend under the 1 USD ceiling: no ceiling breach", not ({"core_ceiling_exceeded", "total_ceiling_exceeded"} & codes(rs)))
check("spend report exposes billed vs ceilings", rs["counters"]["spend"]["spent_paid"] == 0.13
      and rs["counters"]["spend"]["ceilings"] == {"core": 0.5, "total": 1.0}, rs["counters"]["spend"])
check("journal-vs-ledger cost mismatch warned", "journal_ledger_cost_differs" in codes(rs, "warnings"))

# force an over-ceiling ledger row directly (a forged/foreign write): the monitor must flag it
import sqlite3                                       # noqa: E402
con = sqlite3.connect(os.path.join(C.ns_work(NSS, ROOT), "ledger.db"))
con.execute("INSERT INTO ledger (ts,provider,state,reserved,actual) VALUES ('t','anthropic@core','settled',0,2.0)")
con.execute("INSERT INTO ledger (ts,provider,state,reserved,actual) VALUES ('t','anthropic@weird','settled',0,0.1)")
con.commit()
con.close()
ro = M.tick_once(NSS, root=ROOT, write=False)
check("core and total ceilings and an unscoped paid row are breaches",
      {"core_ceiling_exceeded", "total_ceiling_exceeded", "paid_call_without_scope"} <= codes(ro), codes(ro))

# cost guard: Opus mean above the explicit A37 threshold 0.080 after 60 billed cells (A24.1-C was 0.040), and projection over the ceiling
rows = []
for i in range(60):
    rows.append(plan_rec(DEV[0], arm="O0", r=i, usd=0.09, scope="core", model="claude-opus-4-8"))
g = {"plan": {"opus_calls": 60, "opus_usd": 5.4, "journal_cost_usd": {"core": 5.4}}}
sg = M.spend_report("test-v1", g["plan"], 0, {"remaining": {"opus": 100}, "mean_cost": {"opus": 0.09}}, ROOT)
sg_at = M.spend_report("test-v1", {"opus_calls": 60, "opus_usd": 4.8, "journal_cost_usd": {"core": 4.8}}, 0, {"remaining": {}, "mean_cost": {}}, ROOT)
check("cost guard pauses on Opus mean 0.09 > the explicit A37 0.080; a mean of exactly 0.080 does not (K23 A37 alignment, literal values)",
      sg["cost_guard_pause"] and any("Opus mean" in x and "0.080" in x for x in sg["cost_guard_reasons"])
      and not any("Opus mean" in x for x in sg_at["cost_guard_reasons"]) and not sg_at["cost_guard_pause"])
sg2 = M.spend_report("test-v1", {"opus_calls": 10, "opus_usd": 0.05, "journal_cost_usd": {"core": 0.05}}, 0,
                     {"remaining": {"opus": 15000}, "mean_cost": {"opus": 0.01}}, ROOT)
check("projection over the test-v1 total ceiling 100 (A37) pauses", sg2["cost_guard_pause"] and sg2["projected"] > 100)
_sp0 = M.spend_report("test-v1", {"opus_calls": 0, "opus_usd": 0.0, "journal_cost_usd": {}}, 0, {"remaining": {}, "mean_cost": {}}, ROOT)["projected"]
_n_under = int((100.0 - _sp0 - 0.005) / 0.01)
sgu = M.spend_report("test-v1", {"opus_calls": 0, "opus_usd": 0.0, "journal_cost_usd": {}}, 0, {"remaining": {"opus": _n_under}, "mean_cost": {"opus": 0.01}}, ROOT)
sgo = M.spend_report("test-v1", {"opus_calls": 0, "opus_usd": 0.0, "journal_cost_usd": {}}, 0, {"remaining": {"opus": _n_under + 2}, "mean_cost": {"opus": 0.01}}, ROOT)
check("A37 boundary: the test-v1 total ceiling is 100 USD: a projection just under 100 does not pause, just over 100 pauses",
      sgu["projected"] < 100.0 and not sgu["cost_guard_pause"] and sgo["projected"] > 100.0 and sgo["cost_guard_pause"]
      and any("ceiling 100.00" in x for x in sgo["cost_guard_reasons"]), (sgu["projected"], sgo["projected"], sgo["cost_guard_reasons"]))
sge = M.spend_report("test-v1", {"opus_calls": 0, "opus_usd": 0.0, "journal_cost_usd": {}}, 0, {"remaining": {"opus": 1}, "mean_cost": {"opus": 100.0 - _sp0}}, ROOT)
check("A37 boundary: a projection of exactly 100.00 USD is accepted (pause only ABOVE the total ceiling)",
      abs(sge["projected"] - 100.0) < 1e-6 and not sge["cost_guard_pause"], (sge["projected"], sge["cost_guard_reasons"]))
sg3 = M.spend_report("test-v1", {"opus_calls": 10, "opus_usd": 0.05, "journal_cost_usd": {}}, 0,
                     {"remaining": {"opus": 100}, "mean_cost": {"opus": 0.0089}}, ROOT)
check("normal projection does not pause", not sg3["cost_guard_pause"] and sg3["projected"] < 100)   # A24.10 cap (A22 was 65)
cc = M.collect(NS, ROOT)["counters"]["plan"]
check("plan counters: status map and cost per scope", cc["status"]["ok"] >= 1 and "free" in cc["journal_cost_usd"])

# ---------------------------------------------------------------------------------------------------- test-item guard
r_ref = M.tick_once("test-v1", items_mode="test", allow_test=False, root=ROOT)
check("test namespace + --items test without --allow-test / freeze: refused, exit 2, nothing written",
      r_ref["refused"] and r_ref["exit_code"] == 2 and not os.path.exists(C.journal_path("test-v1", "monitor", ROOT)), r_ref["refused"])
r_ref2 = M.tick_once("test-v1", items_mode="test", allow_test=True, root=ROOT)
check("--allow-test alone is not enough (no freeze / families)", r_ref2["refused"] and "freeze" in r_ref2["refused"], r_ref2["refused"])
r_ref3 = M.tick_once("rehearsal-v1", items_mode="test", allow_test=True, root=ROOT)
check("test items in a non-test namespace refused", r_ref3["refused"] is not None)
# test-item rows in a dev journal are caught by the guard even in dev mode
C.Journal(C.journal_path("practice-v1", "plan", ROOT)).append(
    C.make_record("plan", key="t1", namespace="practice-v1", arm="G0", block="main", pres=C.pres_id(TEST[0], "en"), r=0, session=0,
                  request_sha="s", status="ok", plan=[], first_plan=[], first_ok=True, repaired=False, calls=1, cached_calls=0,
                  tokens_in=1, tokens_out=1, cost_usd=0.0, model="g", scope="free"))
r_leak = M.tick_once("practice-v1", root=ROOT, write=False)
check("a test item found in a practice journal is refused", r_leak["refused"] is not None and r_leak["exit_code"] == 2)

# valid freeze + families: the test namespace is monitorable, and a code change after the freeze makes it a breach
code_file = os.path.join(ROOT, "sim", "x.py")
os.makedirs(os.path.dirname(code_file), exist_ok=True)
open(code_file, "w").write("A=1\n")
P = C.paths(ROOT)
fz = {"schema": "final-test.freeze/1", "created": "t", "prereg_sha256": C.sha256_file(P["prereg"]), "split_sha256": C.SPLIT_SHA256,
      "code": {"sim/x.py": C.sha256_file(code_file)}, "gates": {"12.1": {"pass": True, "evidence": "t"}},
      "arash_check": {"by": "Arash", "date": "2026-10-12"}}
fz["sha256"] = C.self_sha(fz)
json.dump(fz, open(P["freeze"], "w"))
json.dump({"approved_by": "Arash", "date": "2026-10-11"}, open(P["families"], "w"))
r_ok = M.tick_once("test-v1", items_mode="test", allow_test=True, root=ROOT, write=False)
check("with --allow-test + valid freeze + families: monitored, exit 0", not r_ok["refused"] and r_ok["exit_code"] == 0, r_ok)
open(code_file, "w").write("A=2\n")
r_drift = M.tick_once("test-v1", items_mode="test", allow_test=True, root=ROOT, write=False)
check("code change after the freeze: refused (guard) with exit 2", r_drift["exit_code"] == 2 and r_drift["refused"])

# ---------------------------------------------------------------------------------------------------- stall, lock, CLI
NS4 = "rehearsal-v1"
a = M.tick_once(NS4, root=ROOT, expected={"capture": 5, "plan": 5, "execute": 5})
b = M.tick_once(NS4, root=ROOT, expected={"capture": 5, "plan": 5, "execute": 5})
check("no progress between two ticks while incomplete is warned", "no_progress_since_last_tick" in codes(b, "warnings"), codes(b, "warnings"))
check("tick numbers increase, monitor journal append-only", b["tick"] == a["tick"] + 1)
lock = C.PhaseLock(NS4, "monitor", ROOT)
lock.__enter__()
check("a second monitor writer is refused by the phase lock", raises(lambda: M.tick_once(NS4, root=ROOT), C.PhaseLocked))
check("read-only (--no-write) still works while locked", M.tick_once(NS4, root=ROOT, write=False)["counters"] is not None)
lock.__exit__()
rc = M.main(["--namespace", "practice-v1", "--root", ROOT, "--once", "--no-write", "--json"])
check("CLI refuses with exit 2 on the leaked test item", rc == 2, rc)
rc2 = M.main(["--namespace", "rehearsal-v1", "--root", ROOT, "--once", "--no-write"])
check("CLI returns the breach exit code (2) for the dirty rehearsal journals", rc2 == 2, rc2)

# ---------------------------------------------------------------------------------------------------- the real repo is untouched
check("selftest wrote nothing under the real docs/final-test/matrix",
      not os.path.exists(os.path.join(REAL, "docs", "final-test", "matrix", "rehearsal-v1", "monitor.jsonl")))
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
