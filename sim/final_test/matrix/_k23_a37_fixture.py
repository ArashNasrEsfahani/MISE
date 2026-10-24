"""Selftest helper (not a gate): a TEMPORARY 'repo' root holding a valid K23 A37 alignment record AND the run_gates-shaped evidence it
names (artifact <label>.json + logs_<label>/<gate>.log), so the hold selftests and _k23_namespaces_selftest can exercise the paid paths
past the hold (Astra K23-A37-ALIGN-r1 point 9, K23-A37-HOLD-r1 points 1-5). Every file it writes is under a tempfile.mkdtemp() root;
`as_repo` points common.ROOT at that root for one block (paths(None), the study ledgers, the study lock and the A37 record then all
resolve there), so no real ledger, lock or record is ever read for a decision or written."""
import contextlib
import hashlib
import json
import os
import tempfile

import common as C
import k23_ns as K

REVIEW_REL = "docs/final-test/k23/_astra/K23-A37-ALIGN-fixture.txt"
GATES_DIR_REL = "sim/final_test/_checkpoints/A37-FIXTURE"
LABEL = "a37fix"
COMMIT = "abc1234def56789abc1234def56789abc1234def"
PROMPT_HEAD = ("Astra (codex exec -m gpt-6-astra, read-only, stdin closed), case K23-A37-ALIGN-fixture\n\n=== PROMPT ===\n"
               "Case K23-A37-ALIGN, commit abc1234: completion review (fixture).\n\n\n")


def answer(text: str) -> str:
    """A review file in astra_call.py's format: the saved prompt, the marker, then Astra's answer."""
    return PROMPT_HEAD + K.A37_ANSWER_MARK + "\n" + text


def make_root(prefix: str = "k23a37_") -> str:
    return tempfile.mkdtemp(prefix=prefix)


def _in_temp(root: str) -> bool:
    return os.path.normcase(os.path.abspath(root)).startswith(os.path.normcase(os.path.abspath(tempfile.gettempdir())) + os.sep)


def _write(p: str, text: str, nl: str = "\n") -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline=nl) as f:
        f.write(text)


def record_path(root: str) -> str:
    return os.path.join(C.paths(root)["docs"], *K.A37_RECORD.split("/"))


def artifact_path(root: str) -> str:
    return os.path.join(root, *GATES_DIR_REL.split("/"), LABEL + ".json")


def log_path(root: str, gate: str) -> str:
    return os.path.join(root, *GATES_DIR_REL.split("/"), "logs_" + LABEL, gate + ".log")


def ensure_files(root: str) -> None:
    for rel in K.A37_TESTS + K.A37_RUNTIME:
        p = os.path.join(root, *rel.split("/"))
        if not os.path.exists(p):
            _write(p, f"# A37 fixture stand-in for {rel}\n")


def manifest(root: str) -> dict:
    return {r: K.a37_sha(os.path.join(root, *r.split("/"))) for r in K.A37_RUNTIME + K.A37_TESTS}


def write_evidence(root: str, commit: str = COMMIT, *, acc_text: str | None = None, mutate_art=None, nl: str = "\n") -> dict:
    """A run_gates-shaped artifact + logs (seven gates at their expected totals, _acc_runner with PASS 5.7 / 5.7b / 5.19 and its historical
    failures), each result bound to `commit` and to the CURRENT fixture bytes before = after. `nl` = the logs' / artifact's line ending."""
    assert _in_temp(root)
    ensure_files(root)
    man = manifest(root)
    art = {"_meta": {"tree": root, "head": commit, "python": "fixture"}}

    def one(g, rc, last, text):
        _write(log_path(root, g), text, nl)
        art[g] = {"rc": rc, "last": last, "counts": None, "src_sha_lf": man[f"sim/{g}.py"], "rel": f"sim/{g}.py",
                  "log_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(), "head": commit, "head_after": commit,
                  "runtime": dict(man), "runtime_after": dict(man)}
    for g, n in K.A37_EXPECTED.items():
        last = f"{n}/{n} passed" if n % 2 else f"{n} passed, 0 failed"
        one(g, 0, last, f"PASS 1.1 a check\n{last}\n--- stderr ---\n")
    acc = acc_text if acc_text is not None else ("PASS 5.7 the study-wide check\nPASS 5.7b the A37 boundary\nPASS 5.19 every paid namespace\n"
                                                 "FAIL 1.10 historical K1\n326/333 checks passed")
    one("_acc_runner_selftest", 1, acc.splitlines()[-1], acc + "\n--- stderr ---\nthe lock is not ours; left in place\n")
    if mutate_art is not None:
        mutate_art(art)
    _write(artifact_path(root), json.dumps(art, indent=1), nl)
    return art


def write_record(root: str, *, commit: str = COMMIT, review_text: str | None = None, mutate=None, mutate_art=None, acc_text=None,
                 nl: str = "\n") -> dict:
    """The evidence (write_evidence), an unconditional case-bound review in astra_call format, and the record (self-sha) built from that
    evidence, as make_a37_alignment_record.py builds it. `mutate(rec)` edits the record before its self-sha is taken."""
    art = write_evidence(root, commit, acc_text=acc_text, mutate_art=mutate_art, nl=nl)
    rp = os.path.join(root, *REVIEW_REL.split("/"))
    _write(rp, review_text if review_text is not None else
           answer(f"RULING: ACCEPTED\n1. Case {K.A37_CASE}, commit {commit[:7]}: fixture review (temporary test tree).\n"))
    man = art["_acc_runner_selftest"]["runtime"]
    rec = {"schema": K.A37_SCHEMA, "commit": commit,
           "tests": {t: man.get(t) for t in K.A37_TESTS}, "runtime": {t: man.get(t) for t in K.A37_RUNTIME},
           "gates": {g: {k: art[g].get(k) for k in ("rc", "last", "log_sha")} for g in K.A37_EXPECTED},
           "acc_runner_a37_checks": {x.strip(): True for x in K.A37_ACC_LINES},
           "acc_runner_log_sha256": art["_acc_runner_selftest"]["log_sha"],
           "gates_file": {"path": GATES_DIR_REL + "/" + LABEL + ".json", "sha256": K.a37_sha(artifact_path(root))},
           "astra_review": {"path": REVIEW_REL, "sha256": K.a37_sha(rp)}}
    for g, n in K.A37_EXPECTED.items():
        rec["gates"][g]["counts"] = {"passed": n, "total": n}
    if mutate is not None:
        mutate(rec)
    rec["sha256"] = C.self_sha(rec)
    _write(record_path(root), json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def seed(ledger_path: str, usd: float) -> None:
    """One SETTLED Anthropic row written straight into a (temporary) ledger file: spend that already happened, no guard involved."""
    assert _in_temp(ledger_path)
    led = C.ScopedLedger(ledger_path, {"core": 1e9, "total": 1e9}, audit=False)
    with led._conn() as c:
        c.execute("INSERT INTO ledger (ts, provider, state, reserved, actual) VALUES (?,?,?,?,?)",
                  ("2026-10-19T00:00:00", "anthropic@core", "settled", float(usd), float(usd)))


@contextlib.contextmanager
def as_repo(root: str):
    assert _in_temp(root)
    old = C.ROOT
    C.ROOT = root
    try:
        yield root
    finally:
        C.ROOT = old
