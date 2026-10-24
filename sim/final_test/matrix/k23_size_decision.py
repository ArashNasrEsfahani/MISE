"""A29 item 6 size decision for the replication kitchens: writes docs/final-test/k23/a29_size_decision.json, the record k23_ns.a29_roster
reads (kitchens[<scene>] = {candidate, manifest_sha256, ...}). NEW module; k23_ns.py / common.py / the A29 sampler are not changed.

Registered rule (prereg A29 item 6, verbatim in substance): "After the Kitchen 1 test, before FloorPlan15 planning, and after FloorPlan15,
before FloorPlan17 planning: the remaining-cost projection is recomputed as: reconciled study-wide spend + unresolved commitments + for each
remaining replication kitchen, sum over its cells of the measured mean billed USD per cell of that arm (O0 and O+M separately; numerator =
every billed attempt of the arm's main-block core cells, repaired, rejected and infra-failed cells included; denominator = the number of
billed cells), taken from Kitchen 1 at the first checkpoint and from FloorPlan15 at the second (FloorPlan15's measurements replace Kitchen
1's), plus that kitchen's 10 % rehearsal allowance. At each checkpoint the largest permitted size no greater than the current one that keeps
the projection <= 80 USD is selected (250, 120, 100); the selection is recorded before any agent outcome of the next kitchen exists. If 100
does not fit, planning pauses and the remaining replication kitchen(s) are reported as not run."

Implementation:
  * sizes = the FROZEN manifests' nested candidates, largest first: all (the primary; 2 x pool presentations, <= 250), first_60 (120),
    first_50 (100). A later decision only ever selects an existing candidate no larger than the current one (never redraws, never grows).
  * spend = acc_common.study_spend (every ledger, reconciled; settled + open = "unresolved commitments" included); with --provider-usd
    (Arash's reading of the provider usage page, A29 item 5) the LARGER value is used and both are recorded.
  * means per arm from the measuring kitchen's ledger through planfirst.attempts_report (every billed attempt of a cell, cache replays 0):
    cells of arm O0 / O+M with a canonical presentation and r = 0 (block main); mean = sum / number of cells with a billed amount > 0.
  * each kitchen's cells = its presentations x 1 call-cell per Opus arm (Gemma is 0 USD); rehearsal allowance = 10 % of that kitchen's
    projected cost, minus what its rehearsal namespace has already billed (that spend is inside `spend`: never counted twice).
  * gate = common.COST_GATE_USD (80). Checkpoint after-k1 selects ONE size for both kitchens (the joint form of the freeze rule); after-fp15
    re-selects FloorPlan17 alone. Preconditions: after-k1 = the Kitchen 1 score is sealed complete, both A29 manifests verify, no
    test-k2-v1 plan row exists; after-fp15 = an after-k1 decision exists, test-k2-v1 has a full plan_first_complete event (all its paid
    cells billed), no test-k3-v1 plan row exists. Decisions use aggregate cost counters only (no outcome).

    python sim/final_test/matrix/k23_size_decision.py --checkpoint after-k1  [--provider-usd X] [--write]     (default: dry run)
    python sim/final_test/matrix/k23_size_decision.py --checkpoint after-fp15 [--provider-usd X] [--write]
    python sim/final_test/matrix/k23_size_decision.py --check                                               (record intact + rosters)
Python 3.10; offline (ledger + journal reads only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402
import k23_entry as E                                    # noqa: E402
import k23_ns as K                                       # noqa: E402

SCHEMA = "k23.a29-size-decision/1"
SIZES = ("all", "first_60", "first_50")                  # largest first (250 / 120 / 100 presentations at a full pool)
OPUS_ARMS = ("O0", "O+M")
REHEARSAL_FRACTION = 0.1
CHECKPOINTS = {"after-k1": {"measure_ns": "test-v1", "decide": ("FloorPlan15", "FloorPlan17")},
               "after-fp15": {"measure_ns": "test-k2-v1", "decide": ("FloorPlan17",)}}
TEST_NS = {"FloorPlan15": "test-k2-v1", "FloorPlan17": "test-k3-v1"}
REHEARSAL_NS = {"FloorPlan15": "rehearsal-k2-v1", "FloorPlan17": "rehearsal-k3-v1"}


INITIAL_PROJECTION = "docs/final-test/matrix/study_projection_a29.json"
INITIAL_MAP = (("250", "all"), ("120", "first_60"), ("100", "first_50"))
FP15_INTERPRETATION = ("'after FloorPlan15' = test-k2-v1's paid planning is complete (a full plan_first_complete event) and its ledger has no "
                       "open reservation; Kitchen 2 execution is unbilled, so every cost counter of FloorPlan15 is final (Astra K23-TOOLING-G3 2)")


class DecisionRefused(RuntimeError):
    pass


def initial_ceiling(root: str | None = None) -> dict:
    """A29 item 6 'At the freeze': the largest of 250 / 120 / 100 presentations whose item-5 projection passed the gate, from the projection
    record written for the Kitchen 1 freeze (A24.1-C restated it: 'The current S2 gate provisionally supports 250 presentations'). The
    after-k1 checkpoint may select only this size or a smaller one (Astra K23-TOOLING-G3 2)."""
    p = os.path.join(C.paths(root)["root"], *INITIAL_PROJECTION.split("/"))
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise DecisionRefused(f"the freeze-time A29 projection {INITIAL_PROJECTION} is unreadable ({type(e).__name__})")
    for n, cand in INITIAL_MAP:
        if ((d.get(n) or {}).get("gate") or {}).get("gate_pass") is True:
            return {"candidate": cand, "presentations_label": int(n), "source": INITIAL_PROJECTION, "sha256": C.sha256_file(p)}
    raise DecisionRefused("no size passed the freeze-time A29 projection gate: testing does not begin under this protocol (A29 item 6)")


def record_path(root: str | None = None) -> str:
    return K.a29_files("FloorPlan15", root)[1]


def load_record(root: str | None = None) -> dict:
    p = record_path(root)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("schema") != SCHEMA or rec.get("sha256") != C.self_sha(rec):
        raise DecisionRefused("a29_size_decision.json: schema / self-sha mismatch (edited after writing)")
    return rec


def manifest(scene: str, root: str | None = None) -> dict:
    bad = E.sample_problems(scene, root)
    if bad:
        raise DecisionRefused("; ".join(bad))
    with open(K.a29_files(scene, root)[0], encoding="utf-8") as f:
        return json.load(f)


def arm_means(ns: str, root: str | None = None, report=None) -> dict:
    """{arm: {mean, usd, billed, cells}} for O0 / O+M main-block (canonical, r = 0) cells of `ns`, from its ledger (every billed attempt)."""
    if report is None:
        import planfirst as PF
        report = PF.attempts_report(ns, root)
    out = {a: {"usd": 0.0, "billed": 0, "cells": 0} for a in OPUS_ARMS}
    for tag, d in (report.get("cells") or {}).items():
        parts = str(tag).split("|")
        if len(parts) < 5 or parts[1] != "plan" or parts[2] not in OPUS_ARMS or not parts[3].startswith("canon:") or parts[4] != "r0":
            continue
        o = out[parts[2]]
        o["cells"] += 1
        o["usd"] += float(d.get("settled_usd") or 0.0)
        o["billed"] += 1 if float(d.get("settled_usd") or 0.0) > 0 else 0
    for a, o in out.items():
        if not o["billed"]:
            raise DecisionRefused(f"{ns}: no billed {a} main-block cell to measure")
        o["usd"] = round(o["usd"], 6)
        o["mean"] = round(o["usd"] / o["billed"], 8)
    if (report.get("open_total") or 0) or (report.get("ledger_open_usd") or 0):
        raise DecisionRefused(f"{ns}: open reservations in its ledger (reconcile first; an unresolved result is never measured)")
    return out


def _rehearsal_billed(scene: str, spend: dict) -> float:
    ns = REHEARSAL_NS[scene]
    return sum(float(v.get("settled", 0.0)) for rel, v in (spend.get("ledgers") or {}).items()
               if C.namespace_of_dir(os.path.basename(os.path.dirname(rel))) == ns)


def project(spend_usd: float, means: dict, pres: dict, spend: dict) -> dict:
    """{scene: {presentations, retained, rehearsal_allowance, total}} + total = spend + every kitchen's total."""
    per_pres = sum(means[a]["mean"] for a in OPUS_ARMS)            # one call-cell per Opus arm and presentation
    out, tot = {}, float(spend_usd)
    for sc, n in pres.items():
        retained = n * per_pres
        allow = max(0.0, REHEARSAL_FRACTION * retained - _rehearsal_billed(sc, spend))
        out[sc] = {"presentations": n, "retained_usd": round(retained, 4), "rehearsal_allowance_usd": round(allow, 4),
                   "total_usd": round(retained + allow, 4)}
        tot += retained + allow
    return {"kitchens": out, "projection_usd": round(tot, 4), "within_gate": tot <= C.COST_GATE_USD + 1e-9}


def _plan_rows(ns: str, root) -> int:
    p = C.journal_path(ns, "plan", root)
    return len(C.Journal(p).rows()) if os.path.exists(p) else 0


def decide(checkpoint: str, *, root: str | None = None, provider_usd: float | None = None, spend: dict | None = None,
           means: dict | None = None, now: str | None = None) -> dict:
    """The decision document for `checkpoint` (nothing written). DecisionRefused when a precondition fails or nothing fits."""
    import acc_common as AC
    if checkpoint not in CHECKPOINTS:
        raise DecisionRefused(f"unknown checkpoint {checkpoint!r}")
    cp = CHECKPOINTS[checkpoint]
    rec = load_record(root)
    cur = dict(rec.get("kitchens") or {})
    mans = {sc: manifest(sc, root) for sc in ("FloorPlan15", "FloorPlan17")}
    if checkpoint == "after-k1":
        seal = os.path.join(C.ns_docs("test-v1", root), "score.seal.json")
        try:
            sealed = json.load(open(seal, encoding="utf-8")).get("complete") is True
        except (OSError, ValueError):
            sealed = False
        if not sealed:
            raise DecisionRefused("after-k1: the Kitchen 1 test score is not sealed complete")
        if cur:
            raise DecisionRefused("after-k1: a decision is already recorded (a later checkpoint may only shrink FloorPlan17)")
    else:
        if not cur.get("FloorPlan15") or not cur.get("FloorPlan17"):
            raise DecisionRefused("after-fp15: no after-k1 decision recorded")
        evs = C.Journal(C.journal_path("test-k2-v1", "events", root)).rows() if os.path.exists(C.journal_path("test-k2-v1", "events", root)) else []
        if not any(e.get("event") == "plan_first_complete" and e.get("scope") == "full" for e in evs):
            raise DecisionRefused("after-fp15: test-k2-v1 has no full plan_first_complete event (its paid cells are not all billed)")
    for sc in cp["decide"]:
        if _plan_rows(TEST_NS[sc], root):
            raise DecisionRefused(f"{checkpoint}: {TEST_NS[sc]} already has plan rows (the selection must precede its planning)")
    sp = spend if spend is not None else AC.study_spend(root)
    if sp.get("errors"):
        raise DecisionRefused("study-wide accounting unresolved: " + "; ".join(sp["errors"][:3]))
    used = max(float(sp["committed"]), float(provider_usd)) if provider_usd is not None else float(sp["committed"])
    mm = means if means is not None else arm_means(cp["measure_ns"], root)
    allowed, init = {}, (initial_ceiling(root) if checkpoint == "after-k1" else None)
    for sc in cp["decide"]:
        cands = mans[sc]["candidates"]
        top = cur[sc]["candidate"] if sc in cur else init["candidate"]           # never above the predecessor selection
        allowed[sc] = [s for s in SIZES[SIZES.index(top):] if s in cands]
    trials, pick = [], None
    for s in SIZES:
        if any(s not in allowed[sc] for sc in cp["decide"]):
            continue
        pres = {sc: 2 * len(mans[sc]["candidates"][s]) for sc in cp["decide"]}
        pr = project(used, mm, pres, sp)
        trials.append({"size": s, **pr})
        if pr["within_gate"] and pick is None:
            pick = s
    if pick is None:
        raise DecisionRefused(f"{checkpoint}: no permitted size keeps the projection <= {C.COST_GATE_USD} USD: planning pauses, the remaining "
                              f"replication kitchen(s) are reported as not run (A29 item 6); projections {[(t['size'], t['projection_usd']) for t in trials]}")
    ts = now or time.strftime("%Y-%m-%dT%H:%M:%S")
    entry = {"checkpoint": checkpoint, "ts": ts, "measure_ns": cp["measure_ns"], "means": mm,
             "spend": {"recorded_committed_usd": round(float(sp["committed"]), 6), "settled_usd": sp.get("settled"), "open_usd": sp.get("open"),
                       "provider_usd": provider_usd, "used_usd": round(used, 6), "provider_reconciled": provider_usd is not None},
             "gate_usd": C.COST_GATE_USD, "rehearsal_fraction": REHEARSAL_FRACTION, "trials": trials, "selected": pick,
             "decides": list(cp["decide"]), "ceiling": init if init else {"from": "the recorded after-k1 selection"},
             "interpretation": FP15_INTERPRETATION if checkpoint == "after-fp15" else None}
    kit = dict(cur)
    for sc in cp["decide"]:
        kit[sc] = {"candidate": pick, "manifest_sha256": mans[sc]["sha256"], "presentations": 2 * len(mans[sc]["candidates"][pick]),
                   "items": len(mans[sc]["candidates"][pick]), "checkpoint": checkpoint, "ts": ts,
                   "roster_sha256": C.sha256_json(list(mans[sc]["candidates"][pick]))}
    doc = {"schema": SCHEMA, "rule": "prereg A29 item 6 (cost-adaptive, aggregate cost counters only; never grows, never redraws)",
           "kitchens": kit, "checkpoints": list(rec.get("checkpoints") or []) + [entry]}
    doc["sha256"] = C.self_sha(doc)
    return doc


def check(root: str | None = None) -> dict:
    out = {"record": E.rel(record_path(root), root), "problems": [], "rosters": {}}
    try:
        rec = load_record(root)
    except DecisionRefused as e:
        out["problems"].append(str(e))
        return out
    if not rec:
        out["problems"].append("no size decision recorded")
        return out
    for sc, ns in TEST_NS.items():
        if sc in (rec.get("kitchens") or {}):
            try:
                r = K.a29_roster(ns, root)
                out["rosters"][ns] = {"n_items": len(r), "presentations": 2 * len(r), "roster_sha256": C.sha256_json(r)}
                if C.sha256_json(r) != rec["kitchens"][sc].get("roster_sha256"):
                    out["problems"].append(f"{sc}: roster differs from the recorded roster_sha256")
            except RuntimeError as e:
                out["problems"].append(str(e))
    out["ok"] = not out["problems"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A29 item 6 size decision (Kitchens 2 / 3)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", choices=sorted(CHECKPOINTS))
    g.add_argument("--check", action="store_true")
    ap.add_argument("--provider-usd", type=float, default=None, help="Arash's reading of the provider usage page (the larger value is used)")
    ap.add_argument("--write", action="store_true", help="write the record (default: dry run, nothing written)")
    a = ap.parse_args(argv)
    if a.check:
        r = check()
        print(json.dumps(r, indent=1))
        return 0 if r.get("ok") else 1
    try:
        doc = decide(a.checkpoint, provider_usd=a.provider_usd)
    except (DecisionRefused, RuntimeError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    last = doc["checkpoints"][-1]
    print(json.dumps({"checkpoint": a.checkpoint, "selected": last["selected"], "spend": last["spend"],
                      "means": {k: v["mean"] for k, v in last["means"].items()},
                      "trials": [(t["size"], t["projection_usd"]) for t in last["trials"]],
                      "kitchens": {sc: {k: v[k] for k in ("candidate", "presentations", "checkpoint")} for sc, v in doc["kitchens"].items()},
                      "written": bool(a.write)}, indent=1))
    if not last["spend"]["provider_reconciled"]:
        print("NOTE: no --provider-usd: the provider usage page is not reconciled in this record (A29 item 5 asks Arash to reconcile after "
              "each kitchen)", file=sys.stderr)
    if a.write:
        p = record_path()
        if a.checkpoint == "after-k1" and os.path.exists(p):
            print("REFUSED: the record exists", file=sys.stderr)
            return 2
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        os.replace(tmp, p)
        print("wrote", E.rel(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
