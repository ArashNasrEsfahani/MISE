"""Offline gate for the K23 A37 paid-planning hold: k23_ns.a37_hold_problems (the strict record validator) and a37_hold_for_ledger, run
first in acc_common.guard_problems, so it binds every PAID K2 / K3 ScopedLedger.reserve, ScopedLedger.block and AccScopedLedger.block_all,
with or without a registered guard, and refuses alternate-root paid K2 / K3 ledgers (Astra K23-A37-ALIGN-r1 points 4-9).

Temporary trees only: the real-path sections point common.ROOT at a temp 'repo' (_k23_a37_fixture.as_repo), so no real ledger, study lock or
alignment record is touched; the 'provider' is a list append that runs only after an approved reservation (nothing is ever sent)."""
import json, os, shutil, sqlite3, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import acc_common as AC
import k23_ns as K
import _k23_a37_fixture as F
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def refused(f, exc, needle):
    try:
        f()
    except exc as e:
        if needle in str(e):
            return True
        print("   (refused for another reason:", str(e)[:160], ")")
        return False
    return False


def rows(p, where="1=1"):
    if not os.path.exists(p):
        return 0
    c = sqlite3.connect(p)
    try:
        return c.execute(f"SELECT COUNT(*) FROM ledger WHERE {where}").fetchone()[0]
    finally:
        c.close()


HOLD, OUTSIDE = "A37 paid-planning hold", "outside this repo's matrix work dir"
roots = []


def root():
    r = F.make_root(); roots.append(r); return r


# ------------------------------------------------------------------------------------------------ A. the strict record validator
T = root()
pr = K.a37_hold_problems(T)
check("A1 no record -> held", len(pr) == 1 and "no valid alignment record" in pr[0])
F.write_record(T)
check("A2 valid record -> lifted", K.a37_hold_problems(T) == [])


def held_by(mut=None, needle="", review_text=None, label="", mutate_art=None, acc_text=None):
    F.write_record(T, mutate=mut, review_text=review_text, mutate_art=mutate_art, acc_text=acc_text)
    pr = K.a37_hold_problems(T)
    good = bool(pr) and any(needle in x for x in pr)
    if not good:
        print("   (problems:", pr[:3], ")")
    F.write_record(T)                                             # back to valid
    check(label, good and K.a37_hold_problems(T) == [])


exp = dict(K.A37_EXPECTED)
A = F.answer
held_by(lambda r: r.update(schema="k23.a37-alignment/2"), "schema / self-sha", label="A3 old schema -> held")
held_by(lambda r: r.update(gates={g: {"rc": 0, "counts": {"passed": 0, "total": 0}} for g in exp}), "seven A37 gates", label="A4 seven 0/0 gates -> held")
held_by(lambda r: r["gates"].pop("_ftm_plan_selftest"), "seven A37 gates", label="A5 a gate missing -> held")
held_by(lambda r: r["gates"].update(_other_selftest={"rc": 0, "counts": {"passed": 1, "total": 1}}), "seven A37 gates", label="A6 an extra gate -> held")
held_by(lambda r: r["gates"]["_ftm_common_selftest"].update(rc=1), "seven A37 gates", label="A7 a gate rc 1 -> held")
held_by(lambda r: r["gates"]["_ftm_callplan_selftest"]["counts"].update(passed=86), "seven A37 gates", label="A8 86/87 -> held")
held_by(lambda r: r["gates"]["_ftm_monitor_selftest"]["counts"].update(passed=48, total=48), "seven A37 gates", label="A9 other total -> held")
held_by(lambda r: r["acc_runner_a37_checks"].pop("PASS 5.7b"), "5.7 / 5.7b / 5.19", label="A10 acc 5.7b missing -> held")
held_by(lambda r: r["acc_runner_a37_checks"].update({"PASS 5.19": False}), "5.7 / 5.7b / 5.19", label="A11 acc 5.19 false -> held")
held_by(lambda r: r.update(acc_runner_log_sha256=""), "5.7 / 5.7b / 5.19", label="A12 no acc log sha -> held")
held_by(lambda r: (r["tests"].pop("sim/_ftm_plan_selftest.py"), r["tests"].update({"sim/final_test/matrix/common.py": r["runtime"]["sim/final_test/matrix/common.py"]})),
        "exactly the eight", label="A13 seven tests + one unrelated file -> held")
held_by(lambda r: r["tests"].pop("sim/_acc_runner_selftest.py"), "exactly the eight", label="A14 seven tests -> held")
held_by(lambda r: r["runtime"].pop("sim/final_test/matrix/acc_common.py"), "runtime files", label="A15 runtime binding incomplete -> held")
held_by(lambda r: r["tests"].update({"sim/_ftm_plan_selftest.py": "0" * 64}), "_ftm_plan_selftest.py changed", label="A16 stale test sha -> held")
held_by(lambda r: r.update(commit=""), "UNCONDITIONAL", label="A17 no commit -> held")
held_by(None, "UNCONDITIONAL", review_text=A("RULING: ACCEPTED WITH CHANGES\nK23-A37-ALIGN abc1234\n"), label="A18 conditional ruling -> held")
held_by(None, "UNCONDITIONAL", review_text=A("RULING: NOT ACCEPTED\nlater: RULING: ACCEPTED\nRULING: ACCEPTED K23-A37-ALIGN abc1234\n"),
        label="A19 first RULING line decides -> held")
held_by(None, "UNCONDITIONAL", review_text=A("RULING: ACCEPTED\ncommit abc1234, another case\n"), label="A20 review not naming the case -> held")
held_by(None, "UNCONDITIONAL", review_text=A("RULING: ACCEPTED\nK23-A37-ALIGN, commit 9999999\n"), label="A21 review naming another commit -> held")
PROMPT = "Astra (codex exec), case K23-A37-ALIGN-r2\n\n=== PROMPT ===\nCase K23-A37-ALIGN, commit abc1234: review.\nRULING: ACCEPTED\n\n"
held_by(None, "UNCONDITIONAL", review_text=PROMPT + K.A37_ANSWER_MARK + "\nRULING: ACCEPTED\n1. Fine.\n",
        label="A20b case + commit only in the saved PROMPT, not in Astra's answer -> held")
held_by(None, "UNCONDITIONAL", review_text=PROMPT + K.A37_ANSWER_MARK + "\nRULING: ACCEPTED WITH CHANGES\nK23-A37-ALIGN abc1234\n",
        label="A20c a RULING line in the prompt does not count; the answer's conditional ruling decides -> held")
held_by(None, "UNCONDITIONAL", review_text="RULING: ACCEPTED\n1. Case K23-A37-ALIGN, commit abc1234: accepted.\n",
        label="A20e no answer marker (a prompt-only or hand-written file) -> held, fail closed")
held_by(None, "UNCONDITIONAL", review_text=PROMPT + "RULING: ACCEPTED\nK23-A37-ALIGN abc1234\n",
        label="A20f prompt-only file whose prompt carries an acceptance line -> held")
TRAP = (PROMPT + "Example of an answer:\n" + K.A37_ANSWER_MARK + "\nRULING: ACCEPTED\n1. Case K23-A37-ALIGN, commit abc1234: accepted.\n\n"
        + K.A37_ANSWER_MARK + "\n")
held_by(None, "UNCONDITIONAL", review_text=TRAP + "RULING: NOT ACCEPTED\n1. K23-A37-ALIGN abc1234: blockers.\n",
        label="A20g a marker + acceptance example INSIDE the prompt, the real answer a rejection -> held")
held_by(None, "UNCONDITIONAL", review_text=TRAP + "RULING: ACCEPTED WITH CHANGES\n1. K23-A37-ALIGN abc1234.\n",
        label="A20h the same with a conditional real answer -> held")
held_by(None, "UNCONDITIONAL", review_text=TRAP + "RULING: ACCEPTED\n1. Case K23-A37-ALIGN, commit abc1234: accepted.\n",
        label="A20i two markers are ambiguous even when both answers accept -> held")
held_by(None, "UNCONDITIONAL", review_text=PROMPT + "see " + K.A37_ANSWER_MARK + " below\nRULING: ACCEPTED\nK23-A37-ALIGN abc1234\n",
        label="A20j a marker that is not a line of its own -> held")
check("A20k a37_review_answer: exactly one standalone marker line -> its suffix",
      K.a37_review_answer("p\n" + K.A37_ANSWER_MARK + "\nRULING: ACCEPTED\nx") == "RULING: ACCEPTED\nx"
      and K.a37_review_answer("p\n" + K.A37_ANSWER_MARK + "\na\n" + K.A37_ANSWER_MARK + "\nb") == "")
F.write_record(T, review_text=PROMPT + K.A37_ANSWER_MARK + "\nRULING: ACCEPTED\n1. Case K23-A37-ALIGN, commit abc1234: accepted.\n")
check("A20d astra_call format, answer unconditional and bound -> lifted", K.a37_hold_problems(T) == [])
F.write_record(T)
rp = os.path.join(T, *F.REVIEW_REL.split("/"))
with open(rp, "a", encoding="utf-8") as f:
    f.write("edited after the record\n")
check("A22 review edited after the record -> held", any("UNCONDITIONAL" in x for x in K.a37_hold_problems(T)))
os.remove(rp)
check("A23 review file missing -> held", any("review file missing" in x for x in K.a37_hold_problems(T)))
F.write_record(T)
rec_p = F.record_path(T)
r0 = json.load(open(rec_p, encoding="utf-8")); r0["commit"] = "fff0000"; json.dump(r0, open(rec_p, "w", encoding="utf-8"))
check("A24 record edited without its self-sha -> held", any("schema / self-sha" in x for x in K.a37_hold_problems(T)))
F.write_record(T)
tp = os.path.join(T, "sim", "_ftm_monitor_selftest.py")
orig = open(tp, "rb").read()
open(tp, "wb").write(orig + b"# changed\n")
check("A25 an aligned test changed after the record -> held", any("_ftm_monitor_selftest.py changed" in x for x in K.a37_hold_problems(T)))
open(tp, "wb").write(orig.replace(b"\n", b"\r\n"))
check("A26 the same test with CRLF line endings (a checkout) -> still lifted", K.a37_hold_problems(T) == [])
kp = os.path.join(T, "sim", "final_test", "matrix", "k23_ns.py")
korig = open(kp, "rb").read()
open(kp, "wb").write(korig + b"x = 1\n")
check("A27 a runtime file changed after the record (same commit) -> held (re-armed)", any("k23_ns.py changed" in x for x in K.a37_hold_problems(T)))
os.remove(kp)
check("A28 a bound runtime file missing -> held", any("k23_ns.py missing" in x for x in K.a37_hold_problems(T)))
open(kp, "wb").write(korig)
rb_ = open(rp, "rb").read()
open(rp, "wb").write(rb_.replace(b"\n", b"\r\n"))
check("A29 restored bytes + a CRLF review -> lifted again", K.a37_hold_problems(T) == [])

# ------------------------------------------------------------------------------------------------ A'. the gate evidence (Astra K23-A37-HOLD-r1 1-3)
G7, ACC = "_ftm_plan_selftest", "_acc_runner_selftest"
CR = "sim/final_test/matrix/acc_common.py"


def art_set(g, **kw):
    return lambda a: a[g].update(kw)


held_by(None, "did not run on the record's commit", mutate_art=art_set(G7, head="0" * 40),
        label="A30 a gate result from another commit (a resumed artifact) -> held")
held_by(None, "did not run on the record's commit", mutate_art=art_set(ACC, head_after="1" * 40),
        label="A31 the commit moved during a gate -> held")
held_by(None, "recorded runtime / test bytes", mutate_art=lambda a: (a[G7]["runtime"].update({CR: "0" * 64}), a[G7]["runtime_after"].update({CR: "0" * 64})),
        label="A32 a gate that ran on other runtime bytes (edited, same commit) -> held")
held_by(None, "recorded runtime / test bytes", mutate_art=lambda a: a[ACC]["runtime_after"].update({CR: "0" * 64}),
        label="A33 runtime changed DURING a gate (before != after) -> held")
held_by(None, "recorded runtime / test bytes", mutate_art=lambda a: [a[g].update(runtime={}, runtime_after={}) for g in exp],
        label="A34 an artifact without runtime manifests (old run_gates) -> held")
held_by(None, "gate file that ran is not the recorded one", mutate_art=art_set(G7, src_sha_lf="0" * 64),
        label="A35 another gate file ran -> held")
held_by(None, "not rc 0 at", mutate_art=art_set(G7, last="128 passed, 0 failed (edited)"),
        label="A36 a summary line that is not in the gate's own log -> held")
held_by(None, "not rc 0 at", mutate_art=lambda a: a[G7].update(last="127 passed, 1 failed"),
        label="A37 artifact summary below the expected total -> held")
held_by(None, "5.7 / 5.7b / 5.19", acc_text="PASS 5.7 x\nFAIL 5.7b y\nPASS 5.19 z\n326/333 checks passed", label="A38 the acc log shows FAIL 5.7b -> held")
held_by(None, "5.7 / 5.7b / 5.19", acc_text="PASS 5.7 x\n  note: PASS 5.7b y\nPASS 5.19 z\n326/333 checks passed",
        label="A39 a PASS 5.7b that is not a PASS line -> held")
held_by(lambda r: r.update(acc_runner_log_sha256="not-a-log-hash"), "is not the recorded log", label="A40 acc_runner_log_sha256 'not-a-log-hash' -> held")
held_by(lambda r: r.update(gates_file={"path": "sim/nowhere.json", "sha256": "0" * 64}), "artifact is missing", label="A41 no gate artifact -> held")
held_by(lambda r: r.pop("gates_file"), "artifact is missing", label="A42 record without a gate artifact -> held")
F.write_record(T)
ap_ = F.artifact_path(T)
aj = json.load(open(ap_, encoding="utf-8")); aj[G7]["rc"] = 0; aj["_meta"]["note"] = "edited"; json.dump(aj, open(ap_, "w", encoding="utf-8"))
check("A43 the artifact edited after the record -> held", any("artifact changed after the record" in x for x in K.a37_hold_problems(T)))
F.write_record(T)
lp_ = F.log_path(T, G7)
lb = open(lp_, "rb").read()
open(lp_, "wb").write(lb + b"PASS extra\n")
check("A44 a gate log edited after the run -> held", any(f"{G7}: its log does not hash" in x for x in K.a37_hold_problems(T)))
os.remove(lp_)
check("A45 a gate log missing -> held", any(f"{G7}: its log is missing" in x for x in K.a37_hold_problems(T)))
os.remove(F.log_path(T, ACC))
check("A46 the acc log missing -> held", any(f"{ACC}: its log is missing" in x for x in K.a37_hold_problems(T)))
F.write_record(T, nl="\r\n")
check("A47 artifact + logs checked out with CRLF -> still lifted", K.a37_hold_problems(T) == [])
F.write_record(T)
check("A48 back to the valid fixture", K.a37_hold_problems(T) == [])

# ------------------------------------------------------------------------------------------------ B. the per-ledger hook
R = root()
with F.as_repo(R):
    w = C.paths()["matrix_work"]
    L = lambda n: os.path.join(w, C.ns_storage(n), "ledger.db")
    paid23 = ("rehearsal-k2-v1", "test-k2-v1", "rehearsal-k3-v1", "test-k3-v1")
    check("B1 free (0 USD) K2 entries are never held", all(K.a37_hold_for_ledger(L(n), 0.0) == [] for n in paid23))
    check("B2 every Kitchen 1 namespace: never held, any amount", all(K.a37_hold_for_ledger(L(n), 99.0) == []
                                                                      for n in C.NAMESPACES if not K.is_replication(n)))
    check("B3 no record: every paid K2 / K3 entry held", all(any(HOLD in x for x in K.a37_hold_for_ledger(L(n), 0.01)) for n in paid23))
    check("B4 practice K2 / K3 (paid amount) held too", all(K.a37_hold_for_ledger(L(n), 0.01) for n in ("practice-k2-v1", "practice-k3-v1")))
    F.write_record(R)
    check("B5 valid record in this repo: paid K2 / K3 entries pass the hook", all(K.a37_hold_for_ledger(L(n), 5.0) == [] for n in paid23))
    A = root()
    pA = os.path.join(C.paths(A)["matrix_work"], "test-k2-v1", "ledger.db")
    check("B6 alternate root: refused although this repo's record is valid", any(OUTSIDE in x for x in K.a37_hold_for_ledger(pA, 1.0)))
    check("B7 a K2 ledger file not named ledger.db: refused", any(OUTSIDE in x for x in K.a37_hold_for_ledger(os.path.join(w, "test-k2-v1", "x.db"), 1.0)))
    check("B8 a K2 ledger one level deeper: refused", any(OUTSIDE in x for x in K.a37_hold_for_ledger(os.path.join(w, "sub", "test-k2-v1", "ledger.db"), 1.0)))
    check("B9 alternate root, free entry: passes", K.a37_hold_for_ledger(pA, 0.0) == [])
    check("B10 exemption refuses a non-temporary root (this tree)", refused(lambda: K.a37_exempt_test_root(HERE), ValueError, "only a temporary"))
    check("B11 exemption refuses the temp dir itself", refused(lambda: K.a37_exempt_test_root(tempfile.gettempdir()), ValueError, "only a temporary"))
    os.remove(F.record_path(R))
    check("B12 hold back on when the record goes", any(HOLD in x for x in K.a37_hold_for_ledger(L("test-k2-v1"), 1.0)))

# ------------------------------------------------------------------------------------------------ C. real paths, hold in force
R2 = root()
sent = []


def send(led, usd, scope="core", provider="anthropic"):
    with led.use(scope):
        rid = led.reserve(provider, usd)
    sent.append(rid)
    return rid


def in_block(led, usd, scope="core"):
    with led.block(scope, usd):
        sent.append("block")


def in_block_all(led, worsts):
    with led.block_all(worsts):
        sent.append("block_all")


with F.as_repo(R2):
    w2 = C.paths()["matrix_work"]
    p2 = os.path.join(C.ns_work("test-k2-v1"), "ledger.db")
    cz = C.NAMESPACES["test-k2-v1"]["ceilings"]
    led = C.open_ledger("test-k2-v1")
    acc = AC.AccScopedLedger(p2, cz)
    lockf = os.path.join(w2, AC.STUDY_LOCK_NAME)
    check("C0 the ledger is the temp repo's (never the real one)", os.path.normcase(led.path) == os.path.normcase(p2)
          and os.path.normcase(p2).startswith(os.path.normcase(R2)))
    check("C1 reserve: refused by the hold, nothing sent, no row", refused(lambda: send(led, 0.5), C.CostCeilingExceeded, HOLD)
          and sent == [] and rows(p2) == 0)
    check("C2 block: BlockRefused by the hold, body never runs, no open hold", refused(lambda: in_block(led, 1.0), C.BlockRefused, HOLD)
          and sent == [] and rows(p2, "state='open'") == 0 and led._hold is None)
    check("C3 block_all: BlockRefused by the hold, no open hold", refused(lambda: in_block_all(acc, {"core": 1.0, "other": 0.5}), C.BlockRefused, HOLD)
          and sent == [] and rows(p2, "state='open'") == 0 and not acc._holds)
    AC.register_ledger_guard(p2, attempt_base_usd=0.0, attempt_cap_usd=100.0, study_cap_usd=180.0, lock_path=os.path.join(R2, "guard.lock"))
    check("C4 registered guard: reserve still refused by the hold", refused(lambda: send(led, 0.5), C.CostCeilingExceeded, HOLD) and sent == [])
    check("C5 registered guard: block still refused", refused(lambda: in_block(led, 1.0), C.BlockRefused, HOLD) and sent == [])
    check("C6 registered guard: block_all still refused", refused(lambda: in_block_all(acc, {"core": 1.0}), C.BlockRefused, HOLD)
          and sent == [] and rows(p2, "state='open'") == 0)
    AC.unregister_ledger_guard(p2)
    rid = send(led, 0.0, provider="ollama")
    check("C7 free (0 USD) reservation on a K2 ledger: allowed", isinstance(rid, int) and rows(p2) == 1)
    led.release(rid); sent.clear()
    k1 = C.open_ledger("test-v1")
    rid = send(k1, 0.5)
    check("C8 Kitchen 1 paid reservation in the same tree: not held", isinstance(rid, int))
    k1.release(rid); sent.clear()
    pacc = os.path.join(C.ns_work("acceptance-v2"), "ledger.db")
    check("C9 Kitchen 1 acceptance ledger without its guard: its own refusal, unchanged",
          refused(lambda: send(C.ScopedLedger(pacc, C.NAMESPACES["acceptance-v2"]["ceilings"]), 0.5), C.CostCeilingExceeded,
                  "opened without its acceptance guard") and sent == [])
    check("C10 no study lock left behind", not os.path.exists(lockf))

    # ---------------------------------------------------------------------------------------- D. lifted by a valid record
    F.write_record(R2)
    rid = send(led, 0.5)
    check("D1 valid record: reserve allowed", isinstance(rid, int))
    led.settle(rid, 0.4, {"provider": "anthropic"})
    n0 = rows(p2)
    with led.block("core", 1.0):
        r_in = led.reserve("anthropic", 0.2)
    check("D2 valid record: block allowed, its hold released at the end", rows(p2, "state='open' AND provider LIKE 'hold@%'") == 0 and rows(p2) == n0 + 2)
    led.release(r_in)
    with acc.block_all({"core": 1.0, "other": 0.5}):
        pass
    check("D3 valid record: block_all allowed and released", rows(p2, "state='open'") == 0 and not acc._holds)
    AC.register_ledger_guard(p2, attempt_base_usd=0.0, attempt_cap_usd=100.0, study_cap_usd=180.0, lock_path=os.path.join(R2, "guard.lock"))
    rid = send(led, 0.3)
    check("D4 valid record + registered guard: reserve allowed", isinstance(rid, int))
    led.release(rid)
    AC.unregister_ledger_guard(p2)
    kp2 = os.path.join(R2, "sim", "final_test", "matrix", "acc_common.py")
    kb = open(kp2, "rb").read()
    open(kp2, "wb").write(kb + b"# a later runtime change\n")
    check("D5 a later runtime change re-arms the hold on the real path", refused(lambda: send(led, 0.3), C.CostCeilingExceeded, "acc_common.py changed"))
    open(kp2, "wb").write(kb)
    check("D6 the bound bytes back: lifted again", isinstance(send(led, 0.0, provider="ollama"), int) and isinstance(send(led, 0.1), int))

    # ---------------------------------------------------------------------------------------- E. the 70 USD cap on the REGISTERED branch
    for n, usd in (("test-k2-v1", 31.0), ("rehearsal-k2-v1", 3.0), ("rehearsal-k3-v1", 3.0)):
        F.seed(os.path.join(C.ns_work(n), "ledger.db"), usd)
    F.seed(os.path.join(C.paths()["docs"], "k23", "archive", "test-k2-v1", "ledger.db"), 2.0)
    others = AC.study_spend(None, exclude=(os.path.join(C.ns_work("test-k3-v1"), "ledger.db"),))
    k23_others = sum(v["liability"] for rel_, v in others["ledgers"].items() if K.is_replication(C.namespace_of_dir(os.path.basename(os.path.dirname(rel_))) or "x"))
    p3 = os.path.join(C.ns_work("test-k3-v1"), "ledger.db")
    led3 = C.open_ledger("test-k3-v1")
    AC.register_ledger_guard(p3, attempt_base_usd=0.0, attempt_cap_usd=100.0, study_cap_usd=180.0, lock_path=os.path.join(R2, "guard3.lock"))
    head = round(70.0 - k23_others, 6)
    check("E1 setup: the other K2 / K3 ledgers carry a known liability", 39.0 < k23_others < 40.0)
    check("E2 registered guard: over the 70 USD K2+K3 cap refused", refused(lambda: send(led3, head + 0.5), C.CostCeilingExceeded, "replication budget"))
    try:
        rid = send(led3, head - 0.01)
    except C.CostCeilingExceeded as e:
        rid = print("   (E3 refused:", str(e)[:160], ")")
    check("E3 registered guard: just under the cap allowed", isinstance(rid, int))
    check("E4 registered guard: then over the cap refused", refused(lambda: send(led3, 0.02), C.CostCeilingExceeded, "replication budget"))
    AC.unregister_ledger_guard(p3)
    check("E5 the unregistered branch applies the same cap", refused(lambda: send(led3, 0.02), C.CostCeilingExceeded, "replication budget"))
    check("E6 no study lock left behind", not os.path.exists(lockf))

    # ---------------------------------------------------------------------------------------- F. alternate roots
    A2 = root()
    pA2 = os.path.join(C.paths(A2)["matrix_work"], "test-k2-v1", "ledger.db")
    la = C.ScopedLedger(pA2, cz)
    acca = AC.AccScopedLedger(pA2, cz)
    sent.clear()
    check("F1 alternate root, valid repo record: reserve refused", refused(lambda: send(la, 0.5), C.CostCeilingExceeded, OUTSIDE) and sent == [])
    check("F2 alternate root: block refused", refused(lambda: in_block(la, 1.0), C.BlockRefused, OUTSIDE) and sent == [])
    check("F3 alternate root: block_all refused", refused(lambda: in_block_all(acca, {"core": 1.0}), C.BlockRefused, OUTSIDE) and sent == [])
    AC.register_ledger_guard(pA2, attempt_base_usd=0.0, attempt_cap_usd=100.0, study_cap_usd=180.0, lock_path=os.path.join(A2, "g.lock"),
                             other_spend=lambda: AC.study_spend(A2, exclude=(pA2,)))
    check("F4 alternate root + registered guard: refused", refused(lambda: send(la, 0.5), C.CostCeilingExceeded, OUTSIDE) and sent == []
          and rows(pA2, "state='open'") == 0)
    AC.unregister_ledger_guard(pA2)
    check("F5 alternate root: a free entry passes", isinstance(send(la, 0.0, provider="ollama"), int))
    K.a37_exempt_test_root(A2)
    check("F6 a REGISTERED temporary test root keeps fake-provider testing", isinstance(send(la, 0.5), int))
    A3 = root()
    pA3 = os.path.join(C.paths(A3)["matrix_work"], "test-k3-v1", "ledger.db")
    check("F7 other temporary roots stay refused", refused(lambda: send(C.ScopedLedger(pA3, cz), 0.5), C.CostCeilingExceeded, OUTSIDE))
    os.remove(F.record_path(R2))
    check("F8 the repo hold is back: the registered test root is still exempt, the repo ledger is not",
          isinstance(send(la, 0.5), int) and refused(lambda: send(led, 0.1), C.CostCeilingExceeded, HOLD))

K._A37_TEST_ROOTS.clear()
check("G1 common.ROOT restored", C.ROOT == os.path.dirname(C.SIM))
for r in roots:
    shutil.rmtree(r, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
