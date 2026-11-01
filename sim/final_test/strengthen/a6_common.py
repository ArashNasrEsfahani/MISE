"""Task A6 (strengthened Scene Memory ablation): shared definitions. Exploratory, post-hoc; never writes into a sealed namespace or rewrites
an existing result. Records: docs/final-test/strengthen/A/a6/. Work files: sim/final_test/strengthen/work/A6/.

New edit arms (on the recorded full-memory block, the A1 grammar; ablate_common.parse_block / render_block):
  no_req_proc     the restatement AND the procedures removed (the whole procedure section); guide, examples, templates, inventory kept
  only_request    ONLY the restatement kept (everything else removed, the NEED bundle included)
  only_procedures ONLY the procedures kept (the procedure section without the restatement)
  only_req_proc   ONLY the restatement + the procedures kept (the procedure section)

Execution sources, in the order a unit key is looked up ("plan once, execute once" across every existing execution): the sealed main
journal of the namespace (+ the GPT6-Sol addendum journal in Kitchens 2 / 3), task A's A4 pass, task L's pass, then A6's own passes. A
source's run-0 history is FINAL when its last attempt finished (done / not_executed) or its third attempt (0-2) ended. A final history
of three reset-gate stops (the A36 excluded_reset pattern) gets a no-exclusion replacement: task R's re-run for a sealed unit R
re-executed, else A6's own re-run (the R way: a separate pass, fresh attempts 0-2, same plan, same restored start state).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate_common as A          # noqa: E402  (paths, TRACE off, block grammar, journals)

ROOT = A.ROOT
DOCS = os.path.join(A.DOCS, "a6")
WORK = os.path.join(HERE, "work", "A6")
RESUME_MD = os.path.join(A.DOCS, "RESUME.md")
STRENGTHEN = os.path.join(ROOT, "docs", "final-test", "strengthen")
KITCHEN_OF = {"test-v1": "K1", "test-k2-v1": "K2", "test-k3-v1": "K3"}
ROLE = {"test-v1": "development kitchen (FloorPlan1)", "test-k2-v1": "held-out kitchen A (FloorPlan15)",
        "test-k3-v1": "held-out kitchen B (FloorPlan17)"}

NEW_ARMS = {
    "no_req_proc": {"kind": "edit", "label": "without the restatement AND the procedures (joint removal)"},
    "only_request": {"kind": "edit", "label": "only the English restatement (everything else removed)"},
    "only_procedures": {"kind": "edit", "label": "only the procedures (everything else removed)"},
    "only_req_proc": {"kind": "edit", "label": "only the restatement + the procedures (everything else removed)"},
}
for _k, _v in NEW_ARMS.items():                    # registered in this process only (ablate_common.py on disk is unchanged)
    A.ARMS.setdefault(_k, _v)

#: Opus cap of task A, cumulative over every run (one ledger, work/A/opus_ledger.db): 25 USD until 2026-10-28 21:40, then 60 USD on Arash's
#: words "You have 80$ opus budget btw." (relayed by the coordinator; recorded in A/decision.json, key opus_cap_2026_10_28). This process only.
OPUS_CAP_USD = 60.0
A.OPUS_CAP_USD = OPUS_CAP_USD
#: a task-A-only stop: while this file exists every A6 chunk exits 77 at its start and the A6 driver stops at its next boundary
A6_STOP = os.path.join(WORK, "STOP_A6")

#: the A6 cells to plan: (namespace, planner) -> arms
A6_CELLS = {
    ("test-v1", "G"): ["no_req_proc", "only_request", "only_procedures", "only_req_proc"],
    ("test-v1", "O"): ["no_request", "no_req_proc", "only_req_proc"],
    ("test-k2-v1", "G"): ["no_request", "no_procedures", "no_req_proc", "only_req_proc", "irrelevant", "rep", "kitchen_fp1"],
    ("test-k3-v1", "G"): ["no_request", "no_procedures", "no_req_proc", "only_req_proc", "irrelevant", "rep", "kitchen_fp1"],
    # 2026-10-28 21:40 extension (Opus budget 80 USD): Opus 5.5 in both held-out kitchens
    ("test-k2-v1", "O"): ["no_request", "no_procedures", "no_req_proc", "only_req_proc"],
    ("test-k3-v1", "O"): ["no_request", "no_procedures", "no_req_proc", "only_req_proc"],
}
#: arms whose blocks A6 renders (the others come from A/blocks.jsonl, rendered in A2)
A6_RENDER = {"test-v1": ["no_req_proc", "only_request", "only_procedures", "only_req_proc"],
             "test-k2-v1": ["no_req_proc", "only_req_proc"], "test-k3-v1": ["no_req_proc", "only_req_proc"]}
#: the primary Gemma arm family per kitchen for Holm (arm minus full memory); the re-plan control is not a member
PRIMARY = {
    "test-v1": ["no_request", "no_guide", "no_procedures", "no_inventory", "no_examples_templates", "kitchen_fp15", "kitchen_fp17",
                "irrelevant", "no_req_proc", "only_request", "only_procedures", "only_req_proc"],
    "test-k2-v1": ["no_request", "no_procedures", "no_req_proc", "only_req_proc", "irrelevant", "kitchen_fp1"],
    "test-k3-v1": ["no_request", "no_procedures", "no_req_proc", "only_req_proc", "irrelevant", "kitchen_fp1"],
}
#: the arms with a three-kitchen mean (the other kitchen's memory: K1 = FloorPlan15's, K2 / K3 = the development kitchen's)
THREE_KITCHEN_OPUS = ["no_request", "no_procedures", "no_req_proc", "only_req_proc"]
THREE_KITCHEN = {"no_request": "no_request", "no_procedures": "no_procedures", "no_req_proc": "no_req_proc", "only_req_proc": "only_req_proc",
                 "irrelevant": "irrelevant", "rep": "rep",
                 "other_kitchen": {"test-v1": "kitchen_fp15", "test-k2-v1": "kitchen_fp1", "test-k3-v1": "kitchen_fp1"}}


def edit(parsed: dict, arm: str) -> tuple:
    """-> (new parsed block, removed ids). The A1 arms delegate to ablate_common.edit_block."""
    if arm not in NEW_ARMS:
        return A.edit_block(parsed, arm)
    recs, removed = [], []
    is_req = lambda r: r["id"].endswith(A.REQUEST_SUFFIX)          # noqa: E731
    for r in parsed["records"]:
        if arm == "no_req_proc":
            keep = r["comp"] != "procedure"
        elif arm == "only_request":
            keep = r["comp"] == "procedure" and is_req(r)
        elif arm == "only_procedures":
            keep = r["comp"] == "procedure" and not is_req(r)
        else:                                                      # only_req_proc
            keep = r["comp"] == "procedure"
        (recs.append(r) if keep else removed.append(r["id"]))
    bundle = list(parsed["bundle"]) if arm == "no_req_proc" else []
    if parsed["bundle"] and not bundle:
        removed.append("need-bundle")
    return {"records": recs, "bundle": bundle}, removed


_orig_edit = A.edit_block


def _edit_block(parsed, arm):
    return edit(parsed, arm) if arm in NEW_ARMS else _orig_edit(parsed, arm)


A.edit_block = _edit_block                         # this process only: ablate_blocks.Builder.edit_arm then renders the new arms too


# ------------------------------------------------------------------------------------------------ paths
def ns_dir(ns: str) -> str:
    return os.path.join(DOCS, ns)


def pass_paths(ns: str, kind: str) -> dict:
    """kind: new | rerun | noise. Each pass has its own journals, so one unit key never means two different executions."""
    d = os.path.join(ns_dir(ns), kind)
    w = os.path.join(WORK, f"exec-{ns}-{kind}")
    return {"dir": d, "exec": os.path.join(d, "exec.jsonl"), "outcome": os.path.join(d, "outcome.jsonl"),
            "events": os.path.join(d, "events.jsonl"), "units": os.path.join(d, "units.json"), "work": w,
            "lock": os.path.join(w, "execute.lock")}


BLOCKS = os.path.join(DOCS, "blocks.jsonl")
PLANS = os.path.join(DOCS, "plans.jsonl")


def all_blocks(ns: str) -> dict:
    """(arm, pres) -> block row: A2's rows, then A6's (A6 never re-renders an A2 arm)."""
    out = {}
    for p in (os.path.join(A.DOCS, "blocks.jsonl"), BLOCKS):
        for r in A.read_jsonl(p):
            if r.get("ns") == ns:
                out.setdefault((r["arm"], r["pres"]), r)
    return out


def all_plans(ns: str | None = None) -> list:
    """Every ablation plan row (A3's plans.jsonl, then A6's); each carries its own key, so they never collide."""
    rows = []
    for p in (os.path.join(A.DOCS, "plans.jsonl"), PLANS):
        for r in A.read_jsonl(p):
            if ns is None or r.get("ns") == ns:
                rows.append(dict(r, a_phase="A3" if p != PLANS else "A6"))
    return rows


# ------------------------------------------------------------------------------------------------ execution sources
def _hist(path: str) -> dict:
    """exec key -> rows (attempt order) of one journal."""
    out = {}
    for r in A.read_jsonl(path):
        out.setdefault(r.get("key"), []).append(r)
    return out


def _starts(path: str) -> dict:
    out = {}
    for e in A.read_jsonl(path):
        if e.get("event") == "exec_attempt_start":
            out.setdefault(e.get("key"), []).append(e)
    return out


def final_history(rows) -> bool:
    if not rows:
        return False
    last = rows[-1]
    return last.get("status") in ("done", "not_executed") or int(last.get("attempt", 0)) >= 2


def three_blocked(rows) -> bool:
    """The A36 restoration-failure pattern: the final history is attempts 0, 1, 2, all blocked_state."""
    if not final_history(rows):
        return False
    att = {}
    for r in rows:
        att[int(r.get("attempt", 0))] = r.get("status")
    return sorted(att) == [0, 1, 2] and all(s == "blocked_state" for s in att.values())


class Source:
    def __init__(self, name: str, exec_path: str, outcome_path: str, events_path: str):
        self.name, self.exec_path = name, exec_path
        self.hist = _hist(exec_path) if os.path.exists(exec_path) else {}
        self.outc = {r.get("key"): r for r in A.read_jsonl(outcome_path)} if os.path.exists(outcome_path) else {}
        self.starts = _starts(events_path) if os.path.exists(events_path) else {}
        ev = A.read_jsonl(events_path) if os.path.exists(events_path) else []
        self.epoch = {}
        for e in ev:
            if e.get("event") == "invalid_resolved":
                for p in e.get("pres") or []:
                    self.epoch[p] = self.epoch.get(p, 0) + 1

    def get(self, ek: str):
        rows = self.hist.get(ek) or []
        return rows if final_history(rows) else None


def sources(ns: str, include_a6_new: bool = True) -> list:
    """The execution sources of a namespace, in lookup order."""
    m = A.main_dir(ns)
    out = [Source("main", os.path.join(m, "exec.jsonl"), os.path.join(m, "outcome.sealed.jsonl"), os.path.join(m, "events.jsonl"))]
    if ns != "test-v1":
        c = os.path.join(ROOT, "docs", "final-test", "k23", "codex", ns)
        out.append(Source("codex", os.path.join(c, "exec.codex.jsonl"), os.path.join(c, "outcome.codex.jsonl"), os.path.join(c, "events.jsonl")))
    a4 = os.path.join(A.DOCS, ns)
    out.append(Source("A4", os.path.join(a4, "exec.jsonl"), os.path.join(a4, "outcome.jsonl"), os.path.join(a4, "events.jsonl")))
    ld = os.path.join(STRENGTHEN, "L", KITCHEN_OF[ns])
    out.append(Source("L", os.path.join(ld, "exec.jsonl"), os.path.join(ld, "outcome.jsonl"), os.path.join(ld, "exec_events.jsonl")))
    if include_a6_new:
        p = pass_paths(ns, "new")
        out.append(Source("A6", p["exec"], p["outcome"], p["events"]))
    return out


def rerun_sources(ns: str) -> list:
    """No-exclusion replacements, in lookup order: task R's re-run pass, then A6's own re-run pass."""
    rd = os.path.join(STRENGTHEN, "R", ns)
    p = pass_paths(ns, "rerun")
    return [Source("R", os.path.join(rd, "exec.jsonl"), os.path.join(rd, "outcome.jsonl"), os.path.join(rd, "events.jsonl")),
            Source("A6-rerun", p["exec"], p["outcome"], p["events"])]


def exec_key(ns: str, unit: str, run: int = 0) -> str:
    return f"{ns}|exec|{unit}|run{run}"


def adopt(ns: str, unit: str, srcs: list):
    """-> (source, history) of the first source with a FINAL run-0 history of the unit, else (None, None)."""
    ek = exec_key(ns, unit)
    for s in srcs:
        h = s.get(ek)
        if h:
            return s, h
    return None, None


def resume(subphase: str, done: list, cmd: str, notes=()):
    """A/RESUME.md for A6 (the A1-A5 lines stay in the 'Done' list; the Opus total is always the cumulative ledger)."""
    base = ["A1-A5 complete (design.md, blocks.jsonl, plans.jsonl, test-v1/, outcome.jsonl, score.jsonl, analysis.json, summary.md)"]
    cap = [f"Opus cap {OPUS_CAP_USD:.0f} USD (cumulative) since 2026-10-28 21:40, Arash: \"You have 80$ opus budget btw.\" "
           "(A/decision.json key opus_cap_2026_10_28); A6c extended with Opus 5.5 in Kitchens 2 / 3.",
           "Stop: `python sim/final_test/strengthen/stop_all.py stop` (all tasks) or create sim/final_test/strengthen/work/A6/STOP_A6 (task A only)."]
    A.update_resume(subphase, base + list(done), None, cmd, cap + list(notes))
