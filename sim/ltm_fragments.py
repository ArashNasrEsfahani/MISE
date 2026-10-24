"""Final test / C3 - procedure FRAGMENTS (docs/step5-plan.md s13 "C2-design" C2-b..C2-c, prereg A9).

A fragment is the part of an ADMITTED practice procedure (R7: privileged terminal goal check true, every step ok, episode valid) that
carried ONE instruction role: the plan steps after the previous role's matched step up to and including this role's matched step.
Kept only when (1) the alignment is unambiguous (every prefix step names the role's object / source / destination type) and
(2) it is non-trivial (> 1 step). Key = that one role, exact on verb, req, object, states, source, destination, negation ('?' never
matches). Support = distinct episode units (ltm_lessons.unit_key) whose admitted procedure contains an identical fragment. A fragment
FAILED REUSE = a later practice episode whose executed plan contains the identical step signature and one of those steps fails; a
fragment is hidden once failed units >= support units, or when its parent procedure is demoted / retired.

Plan-time gate (test and practice identical): same kitchen; key exact; support >= 1; not contradicted; parent not demoted / retired.
Preconditions are NOT checked at plan time (nothing has been done yet); they are rendered as prerequisites. Rendering is conditional
and qualified, never an order. Whole-intent procedures keep the frozen 8.3 gate (trusted only, `Retriever.retrieve_procedures`).

Pure module: standard library + the step-5 retrieval / concept modules. Reads the store through `select`; never writes it. The
failed-reuse ledger is a sidecar JSON file (`fragments.json`, shipped inside a snapshot directory and hashed with it) because the
frozen store schema has no fragment table; it is derived only from the agent's own step outcomes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

try:
    import ltm_concepts as lc
    import ltm_retrieval as lr
except ImportError:                     # pragma: no cover
    from sim import ltm_concepts as lc
    from sim import ltm_retrieval as lr

LEDGER_NAME = "fragments.json"
KEY_FIELDS = tuple(lc.ROLE_FIELDS)

FRAG_WORDS = {
    "en": {"n": "To {k}: this worked {n} times: {steps} (needs: {pre})",
           "one": "To {k}: worked once, unconfirmed: {steps} (needs: {pre})",
           "empty": "empty hand", "holding": "holding {x}"},
    "fa": {"n": "برای «{k}»: {n} بار جواب داده: {steps} (پیش‌نیاز: {pre})",
           "one": "برای «{k}»: یک بار جواب داده، تأییدنشده: {steps} (پیش‌نیاز: {pre})",
           "empty": "دست خالی", "holding": "در دست داشتن {x}"},
}


def _h(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def role_key(role) -> tuple:
    return tuple(str(role.get(k, "-")) for k in lc.ROLE_FIELDS)


def key_known(key) -> bool:
    return "?" not in key


def step_sig(steps) -> tuple:
    return tuple((str(s.get("verb")), str(s.get("otype")), str(s.get("to"))) for s in steps)


def fragment_id(kitchen, key, sig) -> str:
    return "fr:" + _h(kitchen, key, sig)[:16]


# ------------------------------------------------------------------------------------------------------ extraction
def extract(rec, lex=None) -> tuple:
    """-> (fragments, discards) for one procedure row. A fragment: {key, sig, steps, pre, proc, role_index}.
    A discard: {proc, role_index, key, reason}. Reasons are enumerated (the C4 report counts them)."""
    lex = lex or lc.default_lexicon()
    roles = lr.intent_roles(rec, lex)
    steps = lr.body_steps(rec)
    frags, disc = [], []
    if not roles or not steps:
        return frags, [{"proc": rec.get("proc"), "role_index": None, "key": None, "reason": "no_roles_or_steps"}]
    pos = 0
    for ri, r in enumerate(roles):
        key = role_key(r)
        if r.get("neg"):
            continue
        if not key_known(key):
            disc.append({"proc": rec.get("proc"), "role_index": ri, "key": key, "reason": "role_unknown"})
            continue
        j = next((i for i in range(pos, len(steps)) if lr.step_matches_role(steps[i], r)), None)
        if j is None:
            disc.append({"proc": rec.get("proc"), "role_index": ri, "key": key, "reason": "no_matching_step"})
            break
        seg = steps[pos:j + 1]
        names = {r.get(f) for f in ("obj", "src", "dest")} - {"-", "?", None}
        prefix = seg[:-1]
        if any(not ({s.get("otype"), s.get("to")} & names) for s in prefix):
            disc.append({"proc": rec.get("proc"), "role_index": ri, "key": key, "reason": "ambiguous_prefix"})
        elif len(seg) < 2:
            disc.append({"proc": rec.get("proc"), "role_index": ri, "key": key, "reason": "one_step"})
        else:
            frags.append({"key": key, "sig": step_sig(seg), "steps": [{k: s.get(k) for k in ("verb", "otype", "to")} for s in seg],
                          "pre": dict(seg[0].get("pre") or {}), "proc": rec.get("proc"), "role_index": ri})
        pos = j + 1
    return frags, disc


# ------------------------------------------------------------------------------------------------------ failed-reuse ledger
class FragmentLedger:
    """Sidecar of fragment reuse outcomes: rows {unit, episode, fid, ok}. Failed units use the lesson rule: per unit, any failed
    reuse makes the unit failed (a counterexample wins, order independent)."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def failed_units(self, fid) -> int:
        per = {}
        for r in self.rows:
            if r["fid"] == fid:
                per[r["unit"]] = per.get(r["unit"], True) and bool(r["ok"])
        return sum(1 for ok in per.values() if not ok)

    def observe(self, unit, episode, plan_steps, step_ok, fragments) -> list:
        """Append one row per fragment whose exact step signature occurs contiguously in the executed `plan_steps`.
        `step_ok[i]` is the step's own ok flag. Returns the rows added."""
        added = []
        sig_plan = step_sig([{"verb": s.get("action"), "otype": s.get("object"), "to": s.get("to")} for s in plan_steps])
        for f in fragments:
            n = len(f["sig"])
            for a in range(0, len(sig_plan) - n + 1):          # EVERY occurrence is inspected (Astra C3 #3): one failure is a failure
                if sig_plan[a:a + n] == f["sig"]:
                    row = {"unit": unit, "episode": episode, "fid": f["fid"], "ok": all(step_ok[a:a + n]), "at": a}
                    self.rows.append(row)
                    added.append(row)
        return added

    def to_json(self) -> dict:
        return {"version": 1, "rows": self.rows}

    @classmethod
    def load(cls, directory) -> "FragmentLedger":
        p = Path(directory) / LEDGER_NAME
        if not p.exists():
            return cls()
        return cls(json.loads(p.read_text(encoding="utf-8")).get("rows") or [])

    def save(self, directory) -> str:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(self.to_json(), ensure_ascii=False, sort_keys=True, indent=1)
        (p / LEDGER_NAME).write_text(raw, encoding="utf-8")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------ index
def admission_units(store, pid) -> set:
    return {r["source"] for r in store.select("application_log", where={"record": pid})
            if r.get("retrieval") == "admit" and r.get("outcome") == "helpful" and r.get("source")}


def build_index(store, kitchen, ledger=None, lex=None) -> dict:
    """fid -> {fid, key, sig, steps, pre, procs, support (distinct units), failed, hidden, E (oldest parent E_created)}, plus the
    per-reason discard counts. Only ELIGIBLE parents (status not demoted / retired) count towards support."""
    lex = lex or lc.default_lexicon()
    ledger = ledger or FragmentLedger()
    out, disc = {}, {}
    for rec in store.select("procedure", where={"kitchen": kitchen}):
        frags, ds = extract(rec, lex)
        for d in ds:
            disc[d["reason"]] = disc.get(d["reason"], 0) + 1
        eligible = rec.get("status") not in ("demoted", "retired")
        units = admission_units(store, rec["proc"])
        for f in frags:
            fid = fragment_id(kitchen, f["key"], f["sig"])
            e = out.setdefault(fid, {"fid": fid, "key": f["key"], "sig": f["sig"], "steps": f["steps"], "pre": f["pre"],
                                     "procs": [], "units": set(), "E": None, "role_index": f["role_index"]})
            e["procs"].append(rec["proc"])
            if eligible:
                e["units"] |= units
            ec = rec.get("E_created")
            e["E"] = ec if e["E"] is None else min(e["E"], ec) if ec is not None else e["E"]
    for fid, e in out.items():
        e["support"] = len(e["units"])
        e["failed"] = ledger.failed_units(fid)
        e["hidden"] = e["support"] < 1 or e["failed"] >= e["support"]
    return {"fragments": out, "discards": disc}


def candidates_for_roles(index, kitchen, roles) -> list:
    """The plan-time gate: per fully resolved role of the CURRENT instruction (in instruction order), the one fragment with the
    exact key, support >= 1, not contradicted - highest support, then oldest E, then id. -> list of fragment dicts."""
    out = []
    for r in roles or []:
        key = role_key(r)
        if r.get("neg") or not key_known(key):
            continue
        cands = [e for e in index["fragments"].values() if e["key"] == key and not e["hidden"]]
        if not cands:
            continue
        cands.sort(key=lambda e: (-e["support"], e["E"] if e["E"] is not None else 1 << 60, e["fid"]))
        out.append(cands[0])
    return out


# ------------------------------------------------------------------------------------------------------ rendering
def _pre_text(pre, lang) -> str:
    w = FRAG_WORDS[lang]
    hold = (pre or {}).get("holding")
    return w["holding"].format(x=hold) if hold not in (None, "", "?") else w["empty"]


def steps_text(steps) -> str:
    """Tool-call syntax, identical in both languages (types are English in the planner prompt)."""
    parts = []
    for s in steps:
        a = f"{s['verb']}({s['otype']}" if s.get("otype") not in (None, "None", "-") else f"{s['verb']}("
        if s.get("to") not in (None, "None", "-"):
            a += f" -> {s['to']}"
        parts.append(a + ")")
    return " ; ".join(parts)


def render_fragment(e, lang, lex=None) -> str:
    lex = lex or lc.default_lexicon()
    role = dict(zip(lc.ROLE_FIELDS, e["key"]))
    role["neg"] = int(role.get("neg", 0))
    k = lc.render_key([role], lex, lang)
    w = FRAG_WORDS[lang]
    tpl = w["one"] if e["support"] <= 1 else w["n"]
    return tpl.format(k=k, n=e["support"], steps=steps_text(e["steps"]), pre=_pre_text(e["pre"], lang))


def fragments_of_procs(store, kitchen, lex=None) -> list:
    """Every fragment (all procedures), for reporting / audits."""
    lex = lex or lc.default_lexicon()
    out = []
    for rec in store.select("procedure", where={"kitchen": kitchen}):
        out += extract(rec, lex)[0]
    return out
