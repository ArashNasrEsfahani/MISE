"""Task A7 (fixed-procedure prompt baseline, reviewer item 1): shared definitions. Exploratory follow-up; never writes into a sealed
namespace or rewrites an existing result. Records: docs/final-test/strengthen/A/a7/. Work files: sim/final_test/strengthen/work/A7/.
Design (written before any A7 outcome exists): docs/final-test/strengthen/A/a7/design.md.

Question (independent ISR review, item 1): does the Scene Memory gain also follow from a FIXED procedural prompt, i.e. the same procedure
knowledge given to every instruction without per-instruction record selection? Two Gemma 4 E2B arms, on the recorded inputs of the three
test namespaces (the same instruction, observation summary, holding state, contract and decoding as G+M; only the memory block differs):

  fixed_proc      the presentation's own English rephrasing line (procedure/*/request-types, when the recorded block has one) + the WHOLE
                  static procedure store as one fixed text: every procedure record of the store (tool documents and step recipes) and every
                  affordance line, in the store's own order. Left out, as declared: the five '-request' affordance lines, which are
                  conditional by construction ("You were asked only for something to cut with: ... stop there") and would be false for
                  almost every instruction. Identical for every presentation of a kitchen (the store is the same in all three kitchens).
                  The 300-token quota of the retrieved part does not apply (it is the point of the arm); the 2000-token package limit does.
  fixed_proc_300  the rephrasing line + a fixed, length-matched procedure text: the procedure / affordance records ranked by how many of the
                  kitchen's recorded full-memory blocks contain them (ties by id), added in that order while the fixed part stays within
                  300 tokens minus the kitchen's longest rephrasing line, so every presentation's retrieved part is within the 300-token
                  quota (the same check as the original). Ranking uses retrieval inputs only (no outcome, no plan, no gold), and it
                  favours the baseline (the most often useful records), which is conservative for the selection claim.

Execution, scoring and the no-exclusion rule are A6's: plan once, execute once across every existing execution (sealed main [+ Sol addendum],
A4, task L, A6, then A7's own pass); a three-reset-stop history is replaced by a re-run (task R, A6's re-run, else A7's own re-run pass).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a6_common as A6             # noqa: E402  (registers the A6 arms; its sources / adoption rules are reused unchanged)
import ablate_common as A          # noqa: E402

ROOT = A.ROOT
DOCS = os.path.join(A.DOCS, "a7")
WORK = os.path.join(HERE, "work", "A7")
RESUME_MD = os.path.join(DOCS, "RESUME.md")
STRENGTHEN = A6.STRENGTHEN
KITCHEN_OF = A6.KITCHEN_OF
ROLE = A6.ROLE
#: a task-A7-only stop: while this file exists every A7 chunk exits 77 at its start and the A7 driver stops at its next boundary
STOP = os.path.join(WORK, "STOP_A7")
PHASE = "A7"

NEW_ARMS = {
    "fixed_proc": {"kind": "fixed", "label": "the rephrasing + the whole static procedure store as one fixed text (no record selection)"},
    "fixed_proc_300": {"kind": "fixed", "label": "the rephrasing + a fixed, length-matched procedure text (300-token quota, no selection)"},
}
for _k, _v in NEW_ARMS.items():                    # registered in this process only (ablate_common.py on disk is unchanged)
    A.ARMS.setdefault(_k, _v)

NAMESPACES = ("test-v1", "test-k2-v1", "test-k3-v1")
#: the A7 cells (Gemma 4 E2B only): the first arm runs first in every kitchen, the second after it (a7_drive step order)
CELLS = {(ns, "G"): ["fixed_proc", "fixed_proc_300"] for ns in NAMESPACES}
QUOTA = 300                                        # the retrieved-part quota of the original package (treatment profile cap)
CONDITIONAL_SUFFIX = "-request"                    # the conditional affordance lines left out of both fixed texts

#: contrasts reported (arm minus comparator); three-kitchen ones need the comparator in all three kitchens
CONTRASTS_3K = [("G~fixed_proc", "G+M"), ("G~fixed_proc", "G~only_req_proc"), ("G~fixed_proc", "G0"),
                ("G~fixed_proc_300", "G+M"), ("G~fixed_proc_300", "G~only_req_proc"), ("G~fixed_proc_300", "G0"),
                ("G~fixed_proc_300", "G~fixed_proc")]
CONTRASTS_K1 = [("G~fixed_proc", "G~only_request"), ("G~fixed_proc_300", "G~only_request")]


# ------------------------------------------------------------------------------------------------ paths
def ns_dir(ns: str) -> str:
    return os.path.join(DOCS, ns)


def pass_paths(ns: str, kind: str) -> dict:
    """kind: new | rerun. Each pass has its own journals, so one unit key never means two different executions."""
    d = os.path.join(ns_dir(ns), kind)
    w = os.path.join(WORK, f"exec-{ns}-{kind}")
    return {"dir": d, "exec": os.path.join(d, "exec.jsonl"), "outcome": os.path.join(d, "outcome.jsonl"),
            "events": os.path.join(d, "events.jsonl"), "units": os.path.join(d, "units.json"), "work": w,
            "lock": os.path.join(w, "execute.lock")}


BLOCKS = os.path.join(DOCS, "blocks.jsonl")
PLANS = os.path.join(DOCS, "plans.jsonl")
FIXED = os.path.join(DOCS, "fixed_texts.json")


def all_blocks(ns: str) -> dict:
    """(arm, pres) -> block row: A2's rows, A6's, then A7's (no phase re-renders another phase's arm)."""
    out = A6.all_blocks(ns)
    for r in A.read_jsonl(BLOCKS):
        if r.get("ns") == ns:
            out.setdefault((r["arm"], r["pres"]), r)
    return out


def all_plans(ns: str | None = None) -> list:
    """Every ablation plan row: A3's, A6's (a_phase A3 / A6), then A7's (a_phase A7). Each carries its own key."""
    rows = A6.all_plans(ns)
    for r in A.read_jsonl(PLANS):
        if ns is None or r.get("ns") == ns:
            rows.append(dict(r, a_phase=PHASE))
    return rows


# ------------------------------------------------------------------------------------------------ execution sources
Source = A6.Source
final_history = A6.final_history
three_blocked = A6.three_blocked
exec_key = A6.exec_key
adopt = A6.adopt


def sources(ns: str, include_new: bool = True) -> list:
    """Lookup order: A6's sources (sealed main [+ Sol addendum], A4, task L, A6's new pass), then A7's own new pass."""
    out = A6.sources(ns, include_a6_new=True)
    if include_new:
        p = pass_paths(ns, "new")
        out.append(Source("A7", p["exec"], p["outcome"], p["events"]))
    return out


def rerun_sources(ns: str) -> list:
    """No-exclusion replacements, in lookup order: task R's re-run pass, A6's re-run pass, then A7's own re-run pass."""
    p = pass_paths(ns, "rerun")
    return A6.rerun_sources(ns) + [Source("A7-rerun", p["exec"], p["outcome"], p["events"])]


def stop_requested() -> bool:
    return os.path.exists(os.path.join(STRENGTHEN, "STOP")) or os.path.exists(STOP)


def resume(subphase: str, done: list, cmd: str, notes=()):
    """A/a7/RESUME.md (task A7's own resume point; A/RESUME.md of A1-A6 is not touched)."""
    A.RESUME_MD = RESUME_MD                        # this process only
    stop = ["Stop: `python sim/final_test/strengthen/stop_all.py stop` (all tasks) or create sim/final_test/strengthen/work/A7/STOP_A7 "
            "(task A7 only)."]
    A.update_resume(subphase, list(done), None, cmd, stop + list(notes))


# ------------------------------------------------------------------------------------------------ the fixed texts
def store_records(ns: str, counter) -> list:
    """The static procedure store of the kitchen's frozen S* source, as block records {id, comp, line}, in the store's own order:
    KB.records() (tool documents, step recipes), then KB.afford_records() (affordance lines) without the conditional '-request' lines.
    A record's line is its English text, which is what the package builder writes (checked against every recorded block in a7_blocks)."""
    import treatment as T
    pk = T.Packager(ns, counter)
    d = T.source_dir(ns, "S*")
    kb = pk.stores("S*", os.path.join(d, "ltm.db"), T.read_rows(d))["procedure"]
    recs = list(kb.records() if callable(kb.records) else kb.records) + list(kb.afford_records())
    out, seen = [], set()
    for r in recs:
        rid = r["id"]
        if rid in seen or rid.endswith(A.REQUEST_SUFFIX) or rid.endswith(CONDITIONAL_SUFFIX):
            continue
        if A.comp_of(rid) != "procedure":
            raise RuntimeError(f"{rid} is not a procedure record")
        line = str(((r.get("text") or {}).get("en")) or "").strip()
        if not line or "\n" in line:
            raise RuntimeError(f"{rid}: empty or multi-line text")
        seen.add(rid)
        out.append({"id": rid, "comp": "procedure", "line": line})
    return out


# ------------------------------------------------------------------------------------------------ A6-name aliases (a7_plan / a7_exec / a7_drive
# are a6_* with `import a7_common as A6`, so the same code paths run on A7's paths)
A6_CELLS = CELLS
A6_STOP = STOP
OPUS_CAP_USD = A6.OPUS_CAP_USD
_sources = sources


def sources(ns: str, include_new: bool = True, include_a6_new=None) -> list:   # noqa: F811
    return _sources(ns, include_new if include_a6_new is None else include_a6_new)
