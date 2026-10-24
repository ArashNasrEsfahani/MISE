"""Acceptance-v2 plumbing shared by the matrix (stage ACC-RUNNER, 2026-10-14; kept out of common.py, which stays small).

  * the ROSTER resolver of namespace acceptance-v2: docs/final-test/acceptance_registration.json (A32 DRAFT, sealed by C11-AND-REG: file
    74b5cf56..., self cd2b2b28...), its integrity (self sha + every frozen section byte-equal to the seal; only the roster re-seal and the
    signature fields may change) and the re-sealed roster section's rules (the registered replacement rule);
  * the A29 item 6 STUDY-WIDE pre-submission check (every cost ledger of the repo + the request <= USD 100) and the per-ledger-file GUARD
    (attempt ceiling 8 USD + the study-wide check) that common.ScopedLedger applies inside every reservation and block hold of a guarded
    ledger; a namespace with "study_guard" refuses paid reservations while no guard is registered (fail closed);
  * the fake over-cap rejection test (night rule: before the first paid submission).

Pure stdlib + common. Python 3.10 compatible. Selftest: python sim/_acc_runner_selftest.py.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import sqlite3
import sys
import threading
import time
import uuid

import common as C

ACCEPTANCE_REGISTRATION = "acceptance_registration.json"
#: the registration as sealed by C11-AND-REG (file 74b5cf56..., self cd2b2b28...); a later file must keep every section except the roster
#: re-seal and the signature fields byte-equal to this one (`acceptance_registration_problems`).
ACC_REG_SEALED_FILE_SHA = "74b5cf56df3a4f449448278d54488b338f9d37a8851b20e7ca3096993f69cd67"
ACC_REG_MUTABLE = ("sha256", "signature", "signature_note", "roster")
ACC_REG_FROZEN_SECTIONS_SHA = "9f99418396cbef09555c971dd67df5a7d1d2b25135da44c6718dfe273fcec3f7"


# ================================================================================================ the acceptance roster
class AcceptanceRegistrationError(RuntimeError):
    """The acceptance registration is missing, edited outside its re-seal fields, or its roster section is malformed."""


ROSTER_SCHEMA = "ft.acceptance-roster/1"


def acceptance_registration_path(root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], ACCEPTANCE_REGISTRATION)


def acceptance_registration_problems(reg: dict) -> list:
    """Integrity of the registration: self sha, the frozen sections equal the C11-AND-REG seal (only the roster re-seal and the
    signature fields may change), the namespace, and (when present) a well-formed sealed roster section."""
    out = []
    if not isinstance(reg, dict):
        return ["registration is not a JSON object"]
    if reg.get("sha256") != C.self_sha(reg):
        out.append("registration self-sha mismatch (edited after sealing)")
    if C.sha256_json({k: v for k, v in reg.items() if k not in ACC_REG_MUTABLE}) != ACC_REG_FROZEN_SECTIONS_SHA:
        out.append("a frozen registration section differs from the C11-AND-REG seal (only the roster and the signature may change)")
    if reg.get("namespace") != C.ACCEPTANCE_NS:
        out.append(f"registration namespace {reg.get('namespace')!r} != {C.ACCEPTANCE_NS}")
    ro = reg.get("roster")
    if ro is not None:
        out += roster_section_problems(reg, ro)
    return out


def roster_section_problems(reg: dict, ro) -> list:
    """The re-sealed roster (written by acc_runner.seal_roster after the live reference checks): 24 ids = 8 L1 + 8 L2 + 8 L3; every L1 / L2
    id is the registered main item (no spare exists for them); the L3 ids are the main items with each FAILED main replaced, in slate order,
    by the next PASSING spare; the replacement log explains every difference; sealed before the first acceptance call."""
    out = []
    if not isinstance(ro, dict) or ro.get("schema") != ROSTER_SCHEMA:
        return ["roster section: wrong schema"]
    if ro.get("sealed") is not True:
        out.append("roster section: not sealed")
    ids = list(ro.get("items") or [])
    by = {i["id"]: i for i in reg.get("items") or []}
    mains = [i["id"] for i in reg.get("items") or [] if i.get("role") == "main"]
    spares = [i["id"] for i in reg.get("items") or [] if str(i.get("role", "")).startswith("spare")]
    if len(ids) != 24 or len(set(ids)) != 24:
        out.append(f"roster section: {len(ids)} ids (24 distinct required)")
    lv = {}
    for i in ids:
        if i not in by:
            out.append(f"roster section: {i} is not a registered item")
            continue
        lv[by[i]["level"]] = lv.get(by[i]["level"], 0) + 1
    if lv and lv != {"L1": 8, "L2": 8, "L3": 8}:
        out.append(f"roster section: levels {lv} (8 / 8 / 8 required)")
    if [i for i in ids if by.get(i, {}).get("level") in ("L1", "L2")] != [i for i in mains if by[i]["level"] in ("L1", "L2")]:
        out.append("roster section: the L1 / L2 items differ from the registered main items (no spare exists for them)")
    log = list(ro.get("replacements") or [])
    want = [i for i in mains if by[i]["level"] == "L3"]
    used = []
    for r in log:
        if r.get("failed") not in want or r.get("spare") not in spares or r.get("spare") in used:
            out.append(f"roster section: replacement {r} is not a registered main -> spare substitution")
            continue
        used.append(r["spare"])
        want = [r["spare"] if x == r["failed"] else x for x in want]
    if [i for i in ids if by.get(i, {}).get("level") == "L3"] != want:
        out.append("roster section: the L3 ids do not follow from the main items and the replacement log")
    failed = [r.get("failed") for r in log]
    if failed != sorted(failed, key=lambda x: mains.index(x) if x in mains else 99):
        out.append("roster section: failed main items not processed in slate order")
    skipped = list(ro.get("skipped_spares") or [])
    if used != [s for s in spares if s in used]:
        out.append("roster section: spares not used in registered order")
    if used:
        last = spares.index(used[-1])
        gap = [s for s in spares[:last] if s not in used]
        if sorted(gap) != sorted(skipped):
            out.append("roster section: a spare before the last used one was neither used nor listed as skipped (failed its own check)")
    return out


def load_acceptance_registration(root: str | None = None) -> dict:
    p = acceptance_registration_path(root)
    if not os.path.exists(p):
        p = acceptance_registration_path(None)              # a temporary test tree without its own copy uses the study's (like the inclusion)
    try:
        with open(p, encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError) as e:
        raise AcceptanceRegistrationError(f"acceptance registration unreadable: {type(e).__name__}") from None
    bad = acceptance_registration_problems(reg)
    if bad:
        raise AcceptanceRegistrationError("; ".join(bad[:4]))
    return reg


def acceptance_roster(root: str | None = None, reg: dict | None = None) -> list:
    """The item ids namespace acceptance-v2 may touch: the sealed roster section when it exists, else the 24 registered MAIN items
    (the runner's preflight refuses to start before the roster is sealed; capture / plan guards admit only these ids either way)."""
    reg = reg if reg is not None else load_acceptance_registration(root)
    ro = reg.get("roster")
    if ro:
        return list(ro["items"])
    return [i["id"] for i in reg.get("items") or [] if i.get("role") == "main"]


def acceptance_manifest(root: str | None = None, manifest: dict | None = None) -> dict:
    """The effective split manifest + the acceptance roster under C.ACCEPTANCE_ROSTER, for the acceptance runner's own phases only
    (`load_manifest` itself never carries it, so no other consumer sees a new id list)."""
    m = dict(manifest if manifest is not None else C.load_manifest(root))
    m[C.ACCEPTANCE_ROSTER] = acceptance_roster(root)
    return m


# ================================================================================================ A29 item 6: study-wide check + ledger guards
STUDY_LEDGER_GLOBS = (("sim", "**", "*.db"), ("docs", "**", "*.db"))


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _scope_of(provider: str) -> str:
    return str(provider).split("@", 1)[1] if "@" in str(provider) else ""


def reconcile_rows(rows, *, paid_prefix: str = "anthropic", pending=None) -> dict:
    """The registered reconciliation of ONE ledger file (A29 cost guard: "holds and request reservations represent the same liability
    without double counting"; Astra ACC-RUNNER r2 Q2). Pure.

    rows = (id, provider, state, reserved, actual) tuples. A paid row (provider starting with `paid_prefix`; '' = every non-hold provider)
    counts at its billed USD when settled and at its reserve when open. An OPEN block hold of scope s covers the paid rows of scope s written
    after it (id > hold id: the block's own calls, settled or open): that group counts ONCE, at max(hold, sum of its rows). A scope with more
    than one open hold is not reconciled: every hold AND every row counts (over-count, the safe direction).
    pending = ("reserve", hold_id or None, usd) | ("hold", scope, usd): the reservation / hold being checked, counted inside its hold when
    hold_id names an open hold of this file, else on top. -> {"settled", "open", "holds", "liability", "groups"}"""
    paid = [r for r in rows if not str(r[1]).startswith("hold@") and str(r[1]).startswith(paid_prefix) and r[2] in ("settled", "open")]
    holds = [r for r in rows if str(r[1]).startswith("hold@") and r[2] == "open"]

    def val(r):
        return float(r[4] or 0) if r[2] == "settled" else float(r[3] or 0)
    by_scope = {}
    for h in holds:
        by_scope.setdefault(_scope_of(h[1]), []).append(h)
    covered, groups, liab = set(), [], 0.0
    pend_kind, pend_ref, pend_usd = (pending or (None, None, 0.0))
    pend_usd = float(pend_usd or 0)
    pend_done = False
    for sc, hs in sorted(by_scope.items()):
        if len(hs) != 1:
            liab += sum(float(h[3] or 0) for h in hs)               # not reconciled: holds + rows both count (safe direction)
            groups.append({"scope": sc, "holds": len(hs), "reconciled": False})
            continue
        h = hs[0]
        mem = [r for r in paid if _scope_of(r[1]) == sc and int(r[0]) > int(h[0])]
        inside = sum(val(r) for r in mem)
        if pend_kind == "reserve" and pend_ref is not None and int(pend_ref) == int(h[0]):
            inside += pend_usd
            pend_done = True
        covered |= {r[0] for r in mem}
        g = max(float(h[3] or 0), inside)
        liab += g
        groups.append({"scope": sc, "hold": round(float(h[3] or 0), 6), "inside": round(inside, 6), "counted": round(g, 6), "reconciled": True})
    liab += sum(val(r) for r in paid if r[0] not in covered)
    if pend_kind and not pend_done:
        liab += pend_usd
    return {"settled": round(sum(float(r[4] or 0) for r in paid if r[2] == "settled"), 6),
            "open": round(sum(float(r[3] or 0) for r in paid if r[2] == "open"), 6),
            "holds": round(sum(float(h[3] or 0) for h in holds), 6), "liability": round(liab, 6), "groups": groups}


_LEDGER_ROWS_SQL = "SELECT id, provider, state, reserved, actual FROM ledger WHERE state IN ('settled','open')"


def ledger_file_liability(p: str, *, paid_prefix: str = "anthropic") -> dict:
    """Read-only reconciliation of one sqlite file. -> {"is_ledger": bool, ...reconcile_rows} ; raises on ANY read error (the caller
    refuses: an unreadable file may be a cost ledger)."""
    c = sqlite3.connect("file:%s?mode=ro" % p.replace(os.sep, "/"), uri=True)
    try:
        has = c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'").fetchone()
        if not has:
            return {"is_ledger": False}
        rows = c.execute(_LEDGER_ROWS_SQL).fetchall()
    finally:
        c.close()
    return dict(reconcile_rows(rows, paid_prefix=paid_prefix), is_ledger=True)


def study_spend(root: str | None = None, exclude=()) -> dict:
    """Reconciled study-wide Anthropic liability over EVERY cost ledger of the repo (sim/**, docs/**; read-only), each file reconciled by
    `reconcile_rows` (a hold and the reservations made inside it count once, at the larger of the two). Every FILE counts on its own: two
    files are never merged, even with identical rows, because no stable ledger identity proves that one is a copy of the other (Astra
    ACC-RUNNER r2: an archived copy is over-counted, the safe direction). ANY error while reading a file (including sqlite_master) lists it in
    `errors`, and study_wide_problems then refuses; only a readable database without a `ledger` table is not a cost ledger. `exclude` =
    ledger files the caller adds itself (its own live ledger)."""
    import glob as _g
    r = C.paths(root)["root"]
    ex = {_norm_path(p) for p in exclude}
    out = {"settled": 0.0, "open": 0.0, "holds": 0.0, "liability": 0.0, "ledgers": {}, "errors": []}
    files = sorted({p for parts in STUDY_LEDGER_GLOBS for p in _g.glob(os.path.join(r, *parts), recursive=True)})
    for p in files:
        if _norm_path(p) in ex:
            continue
        rel = os.path.relpath(p, r).replace(os.sep, "/")
        try:
            lf = ledger_file_liability(p)
        except Exception as e:                                     # noqa: BLE001 -- unreadable: it may be a cost ledger (refuse)
            out["errors"].append(f"{rel}: unreadable ({type(e).__name__})")
            continue
        if not lf.get("is_ledger"):
            continue
        if lf["settled"] or lf["open"] or lf["holds"]:
            out["ledgers"][rel] = {k: lf[k] for k in ("settled", "open", "holds", "liability")}
            for k in ("settled", "open", "holds", "liability"):
                out[k] += lf[k]
    for k in ("settled", "open", "holds", "liability"):
        out[k] = round(out[k], 6)
    out["committed"] = out["liability"]                           # the reconciled liability (settled + open, holds counted once)
    return out


def study_wide_problems(next_usd: float, *, spend: dict | None = None, cap: float = C.ANTHROPIC_CAP_USD, root: str | None = None) -> list:
    """A29 item 6 [Astra A29 7]: before ANY paid submission, reconciled study-wide spend + unresolved commitments + the proposed additional
    worst-case liability must not exceed the study cap (USD 100). Pure given `spend` (study_spend's dict). Empty list = inside."""
    sp = spend if spend is not None else study_spend(root)
    if sp.get("errors"):
        return ["study-wide: unresolved accounting, unreadable cost ledger(s): " + "; ".join(sp["errors"][:3]) + " (no submission)"]
    tot = float(sp["committed"]) + float(next_usd)
    if tot > cap + 1e-12:
        return [f"study-wide: committed {sp['committed']:.4f} (settled {sp['settled']:.4f} + open {sp['open']:.4f}) + next {float(next_usd):.4f} "
                f"= {tot:.4f} > the study cap {cap:.2f} USD (A29 item 6)"]
    return []


#: ACC-RUNNER: per-ledger-file guards (attempt ceiling + the A29 study-wide check), keyed by the ledger's absolute path, so EVERY
#: ScopedLedger instance on that file (the phase's own, a parallel worker's, the batch driver's) applies them. A namespace with
#: "study_guard": True refuses any paid reservation on its ledger while no guard is registered (fail closed: plan.py run directly on
#: acceptance-v2, bypassing the acceptance runner, cannot spend).
_LEDGER_GUARDS = {}


def register_ledger_guard(path: str, *, attempt_base_usd: float, attempt_cap_usd: float, study_cap_usd: float = C.ANTHROPIC_CAP_USD,
                          root: str | None = None, other_spend=None, lock_path: str | None = None, lifecycle=None) -> dict:
    """ACC-ATTEMPT2: `lifecycle(attempt_total_usd)` -> reasons, run inside the study-wide boundary on EVERY reservation / hold of this
    ledger (the attempt still open, its record / authorization still the bound ones, the combined acceptance total); [] = may spend."""
    g = {"attempt_base": float(attempt_base_usd), "attempt_cap": float(attempt_cap_usd), "study_cap": float(study_cap_usd),
         "other_fn": other_spend or (lambda: study_spend(root, exclude=(path,))), "other": None, "lock": lock_path, "lifecycle": lifecycle}
    _LEDGER_GUARDS[_norm_path(path)] = g
    return g


def unregister_ledger_guard(path: str) -> None:
    _LEDGER_GUARDS.pop(_norm_path(path), None)


def ledger_guard(path: str):
    return _LEDGER_GUARDS.get(_norm_path(path))


def guarded_namespace(path: str):
    """The namespace whose work dir holds this ledger file, if that namespace requires the acceptance guard (else None)."""
    d = C.namespace_of_dir(os.path.basename(os.path.dirname(os.path.abspath(path))))     # READY: storage dir -> namespace (NS_STORAGE)
    return d if (C.NAMESPACES.get(d) or {}).get("study_guard") and os.path.basename(path) == "ledger.db" else None


def paid_namespace_of_repo(path: str):
    """A29 item 6 wiring for EVERY paid namespace (Astra ACC-RUNNER Q2): the namespace of a ledger file that sits in THIS repo's matrix work
    dir and has a paid total ceiling (> 0). Temporary test trees are not the study's ledgers and keep their own behaviour."""
    ap = os.path.abspath(path)
    work = C.paths(None)["matrix_work"]
    d = C.namespace_of_dir(os.path.basename(os.path.dirname(ap)))                       # READY: storage dir -> namespace (NS_STORAGE)
    if os.path.basename(ap) != "ledger.db" or _norm_path(os.path.dirname(os.path.dirname(ap))) != _norm_path(work):
        return None
    ns = C.NAMESPACES.get(d)
    return d if ns and float((ns.get("ceilings") or {}).get("total") or 0) > 0 else None


def refresh_study(g: dict) -> dict:
    g["other"] = g["other_fn"]()
    return g["other"]


def refresh_on_block(path: str) -> None:
    g = ledger_guard(path)
    if g is not None:
        refresh_study(g)


# ------------------------------------------------------------------------------------------------ the study-wide reservation boundary
#: Astra ACC-RUNNER r2 Q2 ("no study-wide atomic reservation boundary"): every paid reservation and block hold of a GUARDED ledger or of a
#: paid namespace of this repo runs its check (which re-reads every other ledger) AND its INSERT under ONE exclusive study lock: an
#: O_CREAT | O_EXCL lock file in the matrix work dir (it works from Windows and from WSL on the same /mnt/d tree; flock / msvcrt locks do
#: not cross that boundary). Two such submissions therefore never approve against the same headroom. ACC-RUNNER-D (Astra ACC-RUNNER r3
#: NEW defect 1): there is NO age-based takeover -- age does not prove the holder is dead, and a read-then-unlink of an "old" lock can delete
#: a FRESH lock another process took in between. A lock that is still there after STUDY_LOCK_WAIT_S refuses (fail closed, nothing sent) and
#: names its holder; a lock left by a crashed process is removed only by Arash after checking that its holder is gone (`acc_runner.py
#: lock-status` shows it). Each acquisition writes a random token; release removes the file only when it still carries OUR token (a lock
#: that is not ours is left in place and reported). Residual (disclosed): a tool that writes a cost ledger WITHOUT common.ScopedLedger (the pre-matrix pilot / signal scripts)
#: does not take this lock; those tools are not run while acceptance spends (one paid job at a time), and their ledgers are still counted
#: on every check.
STUDY_LOCK_NAME = "study_spend.lock"
STUDY_LOCK_WAIT_S = 30.0
_BOUNDARY = {"depth": 0, "path": None, "token": None}
_BOUNDARY_RLOCK = threading.RLock()


class StudyBoundaryBusy(C.CostCeilingExceeded):
    """The study-wide reservation boundary could not be entered in time: nothing was checked, nothing was sent."""


def study_lock_path(ledger_path: str):
    """The lock file that serializes this ledger's paid submissions with every other study ledger's, or None (not a study ledger)."""
    g = ledger_guard(ledger_path)
    if g is not None and g.get("lock"):
        return g["lock"]
    if g is not None or guarded_namespace(ledger_path) or paid_namespace_of_repo(ledger_path):
        return os.path.join(C.paths(None)["matrix_work"], STUDY_LOCK_NAME)
    return None


def lock_holder(p: str):
    """The holder record {pid, host, ts, token} of the study lock at `p`, {} for an empty / unreadable one, None when there is no lock."""
    try:
        txt = open(p, encoding="utf-8").read()
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        d = json.loads(txt)
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def _lock_acquire(p: str, wait_s: float, now=None, sleep=None) -> str:
    """Take the study lock (O_CREAT | O_EXCL) -> our token. Never removes an existing lock, whatever its age (Astra ACC-RUNNER r3)."""
    now, sleep = now or time.time, sleep or time.sleep
    os.makedirs(os.path.dirname(p), exist_ok=True)
    token = uuid.uuid4().hex
    t0 = now()
    while True:
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if now() - t0 > wait_s:
                h = lock_holder(p)
                raise StudyBoundaryBusy(
                    f"study-wide reservation boundary busy for > {wait_s:.0f} s ({p}; holder {json.dumps(h) if h is not None else 'gone'}); "
                    "nothing sent. A lock is never taken over by age: if its holder is dead, Arash removes it after checking") from None
            sleep(0.05)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "ts": now(), "token": token}))
            f.flush()
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        return token


def _lock_release(p: str, token: str) -> bool:
    """Remove the study lock only when it carries OUR token. -> True when removed. A lock that is missing or carries another token is
    left alone and reported (it is not ours to delete). Only the owner ever removes its lock (there is no takeover), so the read and the
    unlink cannot race with a second owner of the same path."""
    h = lock_holder(p)
    if not token or not isinstance(h, dict) or h.get("token") != token:
        sys.stderr.write(f"study-wide boundary: the lock at {p} is not ours (holder {h!r}); left in place\n")
        return False
    try:
        os.remove(p)
    except OSError:
        return False
    return True


@contextlib.contextmanager
def study_boundary(ledger, wait_s: float | None = None):
    """Hold the study-wide reservation boundary around one check + insert (re-entrant inside one process). No-op for a ledger that is not
    a study ledger (a temporary test tree without a registered guard)."""
    p = study_lock_path(ledger.path)
    if p is None:
        yield None
        return
    with _BOUNDARY_RLOCK:
        if _BOUNDARY["depth"] == 0:
            _BOUNDARY["token"] = _lock_acquire(p, STUDY_LOCK_WAIT_S if wait_s is None else wait_s)
            _BOUNDARY["path"] = p
        _BOUNDARY["depth"] += 1
        try:
            yield p
        finally:
            _BOUNDARY["depth"] -= 1
            if _BOUNDARY["depth"] == 0:
                _lock_release(_BOUNDARY["path"], _BOUNDARY["token"])
                _BOUNDARY["path"], _BOUNDARY["token"] = None, None


def boundary_held(ledger) -> bool:
    p = study_lock_path(ledger.path)
    return p is None or (_BOUNDARY["depth"] > 0 and _BOUNDARY["path"] == p)


def own_liability(ledger, c, *, paid_prefix: str, pending=None) -> float:
    """The reconciled liability of the ledger's OWN file, read inside its open transaction (sees its uncommitted rows)."""
    return reconcile_rows(c.execute(_LEDGER_ROWS_SQL).fetchall(), paid_prefix=paid_prefix, pending=pending)["liability"]


def guard_problems(ledger, c, usd: float, exclude_rid=None) -> list:
    """The guard checks of one reservation / hold inside the IMMEDIATE transaction of the ledger. -> reasons (empty = inside). The other
    ledgers are re-read on EVERY check (no cached total); the own file is reconciled (a hold and its reservations count once). The caller
    must hold the study-wide boundary (common.ScopedLedger.reserve / block take it); a check outside it refuses. An acceptance ledger needs
    its registered guard (fail closed); every other paid namespace of the repo gets the A29 study-wide check automatically."""
    import k23_ns                                                 # K23 A37 hold: every PAID K2 / K3 entry, registered guard or not, any root
    hold = k23_ns.a37_hold_for_ledger(ledger.path, usd)
    if hold:
        return hold
    g = ledger_guard(ledger.path)
    if float(usd) <= 0:                                           # ACC-ATTEMPT2 (Astra r2 4): a free reservation still meets the lifecycle
        if g is not None and g.get("lifecycle") is not None:
            try:
                return list(g["lifecycle"](0.0))
            except Exception as e:                                # noqa: BLE001
                return [f"attempt lifecycle unreadable ({type(e).__name__}: {str(e)[:120]}); nothing sent"]
        return []
    pend = ("reserve", exclude_rid, float(usd))
    if g is None:
        ns = guarded_namespace(ledger.path)
        if ns:
            return [f"{ns}: the ledger was opened without its acceptance guard (attempt ceiling + A29 study-wide check); nothing sent"]
        if paid_namespace_of_repo(ledger.path):
            if not boundary_held(ledger):
                return ["study-wide: the check ran outside the study-wide reservation boundary; nothing sent"]
            own = own_liability(ledger, c, paid_prefix="anthropic", pending=pend)
            sp = study_spend(None, exclude=(ledger.path,))
            import k23_ns                                         # K23: combined K2+K3 <= 70 USD (inert for every Kitchen 1 ledger)
            return study_wide_problems(own, spend=sp) + k23_ns.replication_problems(ledger.path, own, sp)
        return []
    if not boundary_held(ledger):
        return ["study-wide: the check ran outside the study-wide reservation boundary; nothing sent"]
    tot = own_liability(ledger, c, paid_prefix="", pending=pend)
    out = []
    if tot - g["attempt_base"] > g["attempt_cap"] + 1e-12:
        out.append(f"attempt ceiling {g['attempt_cap']:.2f} USD: this attempt would commit {tot - g['attempt_base']:.4f} (incl. {float(usd):.4f})")
    if g.get("lifecycle") is not None:                            # ACC-ATTEMPT2: the attempt's lifecycle, re-checked on every submission
        try:
            out += list(g["lifecycle"](tot - g["attempt_base"]))
        except Exception as e:                                    # noqa: BLE001 -- an unreadable lifecycle input refuses (fail closed)
            out.append(f"attempt lifecycle unreadable ({type(e).__name__}: {str(e)[:120]}); nothing sent")
    own = own_liability(ledger, c, paid_prefix="anthropic", pending=pend)
    sp = refresh_study(g)                                         # K23 (Astra K23-A37-ALIGN-r1 point 4): the 70 USD K2+K3 cap also binds
    out += study_wide_problems(own, spend=sp, cap=g["study_cap"]) + k23_ns.replication_problems(ledger.path, own, sp)   # a guarded ledger
    return out


def attempt_base_of(path: str) -> float:
    """The attempt base: settled + open paid rows of the namespace ledger, holds EXCLUDED (a hold is not spend; a stale hold at the start
    would otherwise lower the attempt's later total and widen its ceiling)."""
    if not os.path.exists(path):
        return 0.0
    c = sqlite3.connect("file:%s?mode=ro" % path.replace(os.sep, "/"), uri=True)
    try:
        has = c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'").fetchone()
        rows = c.execute(_LEDGER_ROWS_SQL).fetchall() if has else []
    finally:
        c.close()
    return reconcile_rows([r for r in rows if not str(r[1]).startswith("hold@")], paid_prefix="")["liability"]


class AccScopedLedger(C.ScopedLedger):
    """ScopedLedger + ONE multi-scope block (Astra ACC-RUNNER r2 Q1): the worst cases of several scopes are held in ONE transaction under
    ONE study-wide boundary, so a single Batch submission can carry the core AND the O+FS cells in the registered order. A reservation
    inside it is checked against its own scope's hold (no double counting); the holds are released together."""

    _holds = None

    @contextlib.contextmanager
    def block_all(self, worsts: dict):
        if self._hold is not None or self._holds:
            raise RuntimeError("blocks do not nest")
        refresh_on_block(self.path)
        held = {}
        try:
            with study_boundary(self):
                with self._lock, self._conn() as c:
                    c.execute("BEGIN IMMEDIATE")
                    try:
                        for sc, w in sorted(worsts.items()):
                            if float(w) <= 0:
                                continue
                            try:
                                self._check(c, sc, float(w))
                            except C.CostCeilingExceeded as e:
                                raise C.BlockRefused(f"block not started ({sc}, worst case {float(w):.4f} USD): {e}") from None
                            held[sc] = c.execute("INSERT INTO ledger (ts, provider, state, reserved) VALUES (?,?,?,?)",
                                                 (time.strftime("%Y-%m-%dT%H:%M:%S"), f"hold@{sc}", "open", float(w))).lastrowid
                        c.execute("COMMIT")
                    except BaseException:
                        c.execute("ROLLBACK")
                        raise
        except StudyBoundaryBusy as e:
            raise C.BlockRefused(f"block not started: {e}") from None
        self._holds = {sc: (rid, float(worsts[sc]), sc) for sc, rid in held.items()}
        self._audit({"event": "block_hold_all", "ids": held, "usd": {k: round(float(v), 6) for k, v in worsts.items()}})
        try:
            yield held
        finally:
            self._holds = None
            with self._lock, self._conn() as c:
                for rid in held.values():
                    c.execute("UPDATE ledger SET state='released' WHERE id=?", (rid,))
            self._audit({"event": "block_release_all", "ids": held})

    def reserve(self, provider: str, usd: float) -> int:
        h = (self._holds or {}).get(self.scope)
        if h is None:
            return super().reserve(provider, usd)
        old, self._hold = self._hold, h
        try:
            return super().reserve(provider, usd)
        finally:
            self._hold = old


# ================================================================================================ the acceptance session barrier
#: Astra ACC-RUNNER Q7: the authorization check sits at the SHARED entry point. guard_items(acceptance-v2) refuses unless this process
#: armed an acceptance session after acc_runner.preflight passed (a direct plan.py / execute.py --namespace acceptance-v2 is refused).
#: ACC-ATTEMPT2: the session is armed for ONE namespace (acceptance-v2 = attempt 1, acceptance-v2-a2 = attempt 2); a process armed for one
#: attempt cannot touch the other attempt's namespace.
SESSION = {"armed": False, "by": None, "namespace": None}


def arm_session(preflight_problems: list, by: str, namespace: str = C.ACCEPTANCE_NS, lifecycle=None) -> None:
    """`lifecycle()` -> reasons (ACC-ATTEMPT2): re-checked on every guard_items call of the armed namespace (closed attempt, changed
    record / authorization / code refuse even inside an armed process)."""
    if preflight_problems:
        raise RuntimeError("acceptance session not armed: the preflight does not hold")
    SESSION.update(armed=True, by=str(by), namespace=namespace, lifecycle=lifecycle)


def disarm_session() -> None:
    SESSION.update(armed=False, by=None, namespace=None, lifecycle=None)


def session_problems(namespace: str = C.ACCEPTANCE_NS) -> list:
    armed_ns = SESSION.get("namespace") or C.ACCEPTANCE_NS          # a session armed without a namespace is an attempt-1 session
    if SESSION.get("armed") and armed_ns == namespace:
        fn = SESSION.get("lifecycle")
        if fn is None:
            return []
        try:
            return [f"{namespace}: {x}" for x in fn()]
        except Exception as e:                                    # noqa: BLE001
            return [f"{namespace}: session lifecycle unreadable ({type(e).__name__})"]
    return [f"{namespace}: no acceptance session armed for this namespace in this process (run through acc_runner.py, whose "
            "preflight needs the recorded authorization of Arash)"]


VERDICT_NAME = "acceptance_verdict.json"                          # = acc_attempt2.VERDICT_NAME (the selftest checks they agree)


def journal_write_problems(path: str) -> list:
    """ACC-ATTEMPT2 (Astra r2 4 / 8): consulted by common.Journal.append for every journal of an acceptance namespace. A CLOSED attempt
    (its verdict exists) refuses every write, whoever writes; a process armed for that namespace also meets its session lifecycle (the
    record / authorization / code bound at a2_start; attempt 1 closed) on EVERY row it writes. Fail closed on an unreadable input."""
    d = os.path.dirname(os.path.abspath(path))
    ns = C.namespace_of_dir(os.path.basename(d))                                          # READY: storage dir -> namespace (NS_STORAGE)
    if ns not in C.ACCEPTANCE_NAMESPACES:
        return []
    if os.path.exists(os.path.join(d, VERDICT_NAME)):
        return [f"{ns} is CLOSED (its verdict is recorded): no further journal write ({os.path.basename(path)})"]
    vl = os.path.join(C.ns_work(ns, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(d))))), "verdict.lock")
    if os.path.exists(vl):                                     # RUNNER-FINAL (Astra ACC-ATTEMPT2 r3 interim 7): no evidence write of ANY
        return [f"{ns}: its verdict is being finalized (verdict.lock held): no journal write ({os.path.basename(path)})"]   # writer meanwhile
    if SESSION.get("armed") and (SESSION.get("namespace") or C.ACCEPTANCE_NS) == ns and SESSION.get("lifecycle") is not None:
        try:
            return [f"{ns}: {x}" for x in SESSION["lifecycle"]()]
        except Exception as e:                                    # noqa: BLE001
            return [f"{ns}: session lifecycle unreadable ({type(e).__name__}); nothing written"]
    return []


def fake_overcap_test(ledger) -> dict:
    """The night rule 'fake over-cap rejection test BEFORE the first paid submission': a block hold one cent above what the attempt,
    namespace and study-wide constraints leave must be refused (C.BlockRefused); nothing is sent and no hold stays open."""
    g = ledger_guard(ledger.path)
    if g is None:
        return {"ok": False, "refused": False, "reason": "no guard registered for this ledger"}
    with ledger._lock, ledger._conn() as c:
        tot = own_liability(ledger, c, paid_prefix="")
        own = own_liability(ledger, c, paid_prefix="anthropic")
    other = refresh_study(g)
    room = min(ledger.ceilings["total"] - tot, g["attempt_cap"] - (tot - g["attempt_base"]), g["study_cap"] - other["committed"] - own)
    probe = round(max(room, 0.0) + 0.01, 6)
    refused, why = False, None
    try:
        with ledger.block("core", probe):
            pass
    except C.BlockRefused as e:
        refused, why = True, str(e)[:300]
    with ledger._lock, ledger._conn() as c:
        left = c.execute("SELECT COUNT(*) FROM ledger WHERE state='open' AND provider LIKE 'hold@%'").fetchone()[0]
    return {"refused": refused, "probe_usd": probe, "room_usd": round(room, 6), "reason": why, "open_holds_after": int(left),
            "study_committed_other_usd": other["committed"], "ok": refused and left == 0}
