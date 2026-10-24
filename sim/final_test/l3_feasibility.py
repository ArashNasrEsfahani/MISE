"""B9 -- L3 executor-feasibility audit (stage H-B9-l3audit; PACKAGE-SPEC section 11, pkg.feas/1).

Question it answers, BEFORE the freeze and on NON-TEST items only: "can the honest executor carry out what an L3 task needs?"
It runs a reference plan (or a B4 route / B6 skeleton bound to a non-test item) as the plan of `LiveExecutor(memory="none")`, no LLM, and
classifies every failure so a general skill bug can be fixed for every arm and a thing the tool set cannot express is reported (RAG cannot
rescue it).

    python sim/final_test/l3_feasibility.py --refplans            (WSL, one simulator, under flock; see wsl_run_l3feas.sh)
    python sim/final_test/l3_feasibility.py --procedures          (B4 routes, 2 distinct non-test configurations each)
    python sim/final_test/l3_feasibility.py --templates           (B6 skeletons bound to their support items)
    python sim/final_test/l3_feasibility.py --summarize           (offline: rows -> l3_feasibility_summary.json)

Honesty: this module reads NO simulator state. The goal check is the executor's own scoring block (`LiveExecutor._score`, inside
`h.scoring()`), and arrives here as the bool `goal_ok`. Test items are refused through `C.guard_items` (namespace practice-v1 = the
non-test pool) AND by an explicit forbid list; originally eligible test tasks are never dropped because of this audit.

Classes (`class`): feasible | executor_defect | perception | navigation | api_cannot_express, plus the bookkeeping classes
`inapplicable` (goal true at reset / object absent: nothing was tested) and `unreviewed` (a failure whose cause is ambiguous until a
human reads the trace; the review overlay `l3_feasibility_review.json` turns it into one of the five). The automatic class (`class_auto`)
is deliberately conservative: it never claims `executor_defect` by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
ROOT = os.path.dirname(SIM)
for _p in (SIM, os.path.join(HERE, "matrix")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "pkg.feas/1"
KM = os.path.join(ROOT, "docs", "final-test", "kitchen_memory")
ROWS_PATH = os.path.join(KM, "l3_feasibility.jsonl")                  # phase "pre": the audit BEFORE any executor change (immutable history)
ROWS_POST_PATH = os.path.join(KM, "l3_feasibility_postfix.jsonl")     # phase "post": the same jobs re-run after fixes (never replaces the above)
SUMMARY_PATH = os.path.join(KM, "l3_feasibility_summary.json")
REVIEW_PATH = os.path.join(KM, "l3_feasibility_review.json")
PROC_RESULT_PATH = os.path.join(KM, "l3_feasibility_procedures.json")
TEMPLATES_PATH = os.path.join(SIM, "step5", "l3_templates.json")
PROCEDURES_PATH = os.path.join(SIM, "step5", "procedures_kb.json")
DATASET_PATH = os.path.join(SIM, "dataset", "general.json")
NAMESPACE = "practice-v1"                       # practice_pool == the non-test pool (116); never a test item

CLASSES = ("feasible", "executor_defect", "perception", "navigation", "api_cannot_express")
BOOKKEEPING = ("inapplicable", "unreviewed", "invariant_error")
DEFECT_STOP = 3                                 # PACKAGE-SPEC s13: executor_defect on >= 3 of the 23 L3 items = stop signal
#: honest_skills.CAUSES -> automatic class. Only the two unambiguous causes are mapped; everything else needs a read of the trace.
AUTO_BY_CAUSE = {"not_detected": "perception", "unreachable": "navigation"}


# ================================================================================================ pure helpers (offline-testable)
def sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def plan_tools(plan) -> list:
    return [str(s.get("action")) for s in (plan or [])]


def classify_auto(steps, goal_ok, inapplicable=None, audit=None) -> tuple:
    """(class_auto, why). `steps` = [{i, action, object, ok, cause}]. Conservative: see the module docstring."""
    if inapplicable:
        return "inapplicable", inapplicable
    a = audit or {}
    if int(a.get("violations") or 0) or int(a.get("oracle_reads") or 0):
        return "invariant_error", f"honesty audit: {a}"
    bad = [s for s in (steps or []) if not s.get("ok")]
    if not bad and goal_ok:
        return "feasible", "every step ok and the goal holds"
    if bad and goal_ok:
        # the world satisfies the goal although a step reported failure (e.g. a slice that succeeded without the knife being found):
        # the task IS achievable with these tools; the failed step stays visible in `why` and `steps`.
        f = bad[0]
        return "feasible", f"goal holds although step {f.get('i')} {f.get('action')} {f.get('object')} reported {f.get('cause')}"
    if not bad and not goal_ok:
        # every tool call reported success yet the world does not satisfy the goal: either the skill did something other than what its
        # contract says (defect), or the plan/goal cannot be expressed with these tools (api). Needs a read of the trace.
        return "unreviewed", "all steps ok but goal not met"
    first = bad[0]
    cause = first.get("cause")
    if cause in AUTO_BY_CAUSE:
        return AUTO_BY_CAUSE[cause], f"step {first.get('i')} {first.get('action')} {first.get('object')}: {cause}"
    return "unreviewed", f"step {first.get('i')} {first.get('action')} {first.get('object')}: {cause}"


def load_review(path=None) -> dict:
    """{key: {"class": one of CLASSES, "reason": str, "fix": optional str}} keyed by item id (refplans) or "mode:requirement"."""
    try:
        with open(path or REVIEW_PATH, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for k, v in (d.get("reviews") or {}).items():
        if isinstance(v, dict) and v.get("class") in CLASSES and v.get("reason"):
            out[k] = v
    return out


def final_class(row_auto, key, review, phase=None) -> tuple:
    """Apply the human review overlay. Astra (H-B9): the automatic `perception` / `navigation` labels are PROVISIONAL symptoms (a missed
    detection or an unreachable target can itself be an executor bug), so a review may override them; it never overrides `feasible` /
    `inapplicable` / `invariant_error`. Without a review an `unreviewed` row stays unresolved: a class is never forced."""
    if row_auto in ("feasible", "inapplicable", "invariant_error"):
        return row_auto, None
    r = review.get(f"{key}@{phase}") or review.get(key) if phase else review.get(key)
    if r:
        return r["class"], r["reason"]
    return row_auto, None


def make_row(*, item, mode, requirement, plan, res_steps, goal_ok, exec_ok, inapplicable, audit, attempt, elapsed_s, review=None,
             kitchen="FloorPlan1", extra=None, phase="pre", run=None) -> dict:
    """One pkg.feas/1 row from an executed (or inapplicable) presentation."""
    steps = [{"i": s.get("i"), "action": s.get("action"), "object": s.get("object"), "to": s.get("to"), "ok": bool(s.get("ok")),
              "cause": s.get("cause"), "retries": s.get("retries"), "prims": s.get("prims")} for s in (res_steps or [])]
    audit = {"violations": int((audit or {}).get("violations") or 0), "oracle_reads": int((audit or {}).get("oracle_reads") or 0)}
    auto, why = classify_auto(steps, goal_ok, inapplicable, audit)
    cls, rwhy = final_class(auto, item if mode == "refplans" else f"{mode}:{requirement}", review or {}, phase)
    row = {"schema": SCHEMA, "item": item, "mode": mode, "requirement": requirement, "attempt": attempt, "kitchen": kitchen,
           "plan": [{k: v for k, v in s.items() if v is not None} for s in (plan or [])], "tools": plan_tools(plan), "steps": steps,
           "goal_ok": goal_ok, "exec_ok": exec_ok, "class": cls, "class_auto": auto, "why": rwhy or why, "audit": audit,
           "inapplicable": inapplicable, "elapsed_s": round(float(elapsed_s or 0.0), 1), "phase": phase,
           "run": run or {}}
    if extra:
        row.update(extra)
    return row


def row_key(row) -> tuple:
    return (row["mode"], row["item"], row["requirement"], int(row.get("attempt") or 1))


def read_rows(path=None) -> list:
    out = []
    try:
        with open(path or ROWS_PATH, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue                         # a torn last line from a killed run: that attempt is simply repeated
                    if r.get("schema") == SCHEMA:
                        out.append(r)
    except OSError:
        pass
    return out


def append_row(row, path=None) -> None:
    p = path or ROWS_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def reclass(row, review=None) -> dict:
    """The class of a stored row recomputed from its own steps / goal_ok / audit with the CURRENT rules and overlay. Stored rows are an
    immutable history; the rules may be corrected later (an earlier version called a goal-reached run with a failed step `perception`)."""
    auto, why = classify_auto(row.get("steps"), row.get("goal_ok"), row.get("inapplicable"), row.get("audit"))
    key = row["item"] if row["mode"] == "refplans" else f"{row['mode']}:{row['requirement']}"
    cls, rwhy = final_class(auto, key, review if review is not None else load_review(), row.get("phase") or "pre")   # rows written before the field existed are pre-fix rows
    return dict(row, **{"class": cls, "class_auto": auto, "why": rwhy or why})


def per_item(rows, review=None) -> dict:
    """Collapse attempts: an (mode,item,requirement) is feasible if ANY attempt was; `flaky` when attempts disagree on feasibility."""
    groups = {}
    review = review if review is not None else load_review()
    for r in (reclass(x, review) for x in rows):
        groups.setdefault((r["mode"], r["item"], r["requirement"]), []).append(r)
    out = {}
    for k, rs in groups.items():
        rs = sorted(rs, key=lambda r: int(r.get("attempt") or 1))
        feas = [r["class"] == "feasible" for r in rs]
        best = next((r for r in rs if r["class"] == "feasible"), rs[-1])
        out[k] = {"class": best["class"], "flaky": any(feas) and not all(feas), "attempts": len(rs), "row": best,
                  "observed_achievable": any(feas), "attempt_classes": [r["class"] for r in rs]}
    return out


def summarize(rows, *, l3_ids=None, meta=None, review=None) -> dict:
    """Counts per class and mode, the list of failing L3 items by class, and the predeclared stop-signal check."""
    pi = per_item(rows, review)
    modes = {}
    for (mode, item, req), v in pi.items():
        m = modes.setdefault(mode, {"n": 0, "by_class": {}, "flaky": [], "items": {}})
        m["n"] += 1
        m["by_class"][v["class"]] = m["by_class"].get(v["class"], 0) + 1
        m["items"][f"{item}|{req}"] = {"class": v["class"], "why": v["row"]["why"], "attempts": v["attempts"],
                                       "attempt_classes": v["attempt_classes"], "observed_achievable": v["observed_achievable"]}
        if v["flaky"]:
            m["flaky"].append(item)
    l3 = set(l3_ids or [])
    # Astra H-B9: count the items with a confirmed executor defect on ANY attempt (a flaky pass does not clear it, a later fix does not either)
    defects = sorted({i for (mode, i, _), v in pi.items() if mode == "refplans" and i in l3 and "executor_defect" in v["attempt_classes"]})
    unrev = sorted({i for (mode, i, _), v in pi.items() if v["class"] == "unreviewed"})
    return {"schema": "pkg.feas.summary/1", "n_rows": len(rows), "modes": modes, "l3_items_expected": len(l3),
            "l3_executor_defect_items": defects, "unreviewed": unrev,
            "stop_signal_executor_defect_ge_3": len(defects) >= DEFECT_STOP, "meta": meta or {}}


def full_summary(l3_ids, meta=None, pre_path=None, post_path=None, review=None) -> dict:
    """The summary file: the PRE-fix audit decides the stop signal and is never edited by later fixes (Astra H-B9: a fixed defect still
    counts); the POST-fix re-run is reported beside it."""
    pre = summarize(read_rows(pre_path or ROWS_PATH), l3_ids=l3_ids, meta=meta, review=review)
    post_rows = read_rows(post_path or ROWS_POST_PATH)
    out = {"schema": "pkg.feas.summary/2", "pre": pre, "stop_signal_executor_defect_ge_3": pre["stop_signal_executor_defect_ge_3"],
           "l3_executor_defect_items_pre": pre["l3_executor_defect_items"]}
    if post_rows:
        out["post"] = summarize(post_rows, l3_ids=l3_ids, meta=meta, review=review)
    return out


def _write_summary(S) -> None:
    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(S, f, indent=1, ensure_ascii=False, sort_keys=True)


# ================================================================================================ binding B4 routes / B6 skeletons
def bind_steps(steps, item_plan) -> tuple:
    """Bind the slots (`?x`) of a route / skeleton from one item's reference plan: a slot takes the object (or target) of the FIRST plan step
    with the same action that supplies it. Returns (bound_steps, unbound_slots). Pure; no simulator, no id of any kind is typed here."""
    bound, unbound, taken = [], [], {}
    for st in steps or []:
        new = dict(st)
        for fld in ("object", "to", "arg"):
            v = new.get(fld)
            if isinstance(v, str) and v.startswith("?"):
                if v in taken:
                    new[fld] = taken[v]
                    continue
                cand = None
                for ps in item_plan or []:
                    if ps.get("action") == st.get("action") and ps.get(fld) not in (None, ""):
                        cand = ps.get(fld)
                        break
                if cand is None:
                    for ps in item_plan or []:                     # same slot in any step of the plan (e.g. the held food of a put_in)
                        if ps.get("object") not in (None, "") and fld == "object":
                            cand = ps.get("object")
                            break
                if cand is None:
                    unbound.append(v)
                else:
                    taken[v] = cand
                    new[fld] = cand
        bound.append(new)
    return bound, sorted(set(unbound))


def distinct_configs(support_items, need=2) -> list:
    """The first `need` support items with pairwise different (setup, goal-type) configurations; deterministic by id."""
    out, seen = [], set()
    for it in sorted(support_items, key=lambda x: x["id"]):
        sig = sha16({"setup": it.get("setup") or [], "tools": plan_tools(it.get("plan"))})
        if sig in seen:
            continue
        seen.add(sig)
        out.append(it)
        if len(out) >= need:
            break
    return out


# ================================================================================================ the live run (WSL)
def _load_split():
    import common as C
    return C, C.load_manifest()


def load_dataset(path=None) -> dict:
    import general_set as gs
    return {it["id"]: it for it in gs.load(path or DATASET_PATH)["items"]}


def l3_nontest_ids(items, manifest) -> list:
    nt = set(manifest["nontest"])
    return sorted(i for i, it in items.items() if it["level"] == "L3" and i in nt)


def l2_support_ids(items, manifest, templates_path=None) -> list:
    """L2 non-test items a B6 template cites as support (empty when B6 has not written its file yet)."""
    try:
        with open(templates_path or TEMPLATES_PATH, encoding="utf-8") as f:
            tpl = json.load(f)
    except (OSError, ValueError):
        return []
    entries = tpl.get("templates") if isinstance(tpl, dict) else tpl
    nt, out = set(manifest["nontest"]), set()
    for e in entries or []:
        for s in (e.get("support") or []):
            if s in nt and s in items and items[s]["level"] == "L2":
                out.add(s)
    return sorted(out)


def make_request(item, plan, cell, kitchen):
    import step5_eval as se
    iid = item["id"]
    it = dict(item)
    it["plan"] = list(plan)
    return se.EpisodeRequest(cell=cell, episode_id=re.sub(r"[^A-Za-z0-9_.-]", "-", f"l3feas-{cell}"), run_id="h-b9-l3audit", item=it,
                             lang="en", level=item["level"], mode="test", world="reset", world_kind=se.CANONICAL, world_seed=0,
                             policy_seed=0, arm="none", features=frozenset(), memory=None, plan_source="reference", kitchen=kitchen)


def _executor_sha() -> str:
    """The patch version the rows were produced with: sha256[:16] of honest_skills.py (a file read, no simulator)."""
    try:
        with open(os.path.join(SIM, "honest_skills.py"), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "unknown"


def run_one(executor, item, plan, *, mode, requirement, attempt, kitchen, review, phase="pre") -> dict:
    """Execute one plan on `executor` (a LiveExecutor, or a fake in the selftest) and build the row. Aborts (raises) on an invariant."""
    req = make_request(item, plan, f"{mode}|{item['id']}|{requirement}|a{attempt}", kitchen)
    t0 = time.time()
    buf = []
    if hasattr(executor, "log"):
        executor.log = buf.append                  # the skills' own event lines (what the agent saw and did); kept for failed runs
    try:
        res = executor.execute(req)
    finally:
        if hasattr(executor, "log"):
            executor.log = None
    ex = res.extra or {}
    core = ex.get("core") or {}
    aud = core.get("audit") or {}
    audit = {"violations": aud.get("violations", 0), "oracle_reads": (aud.get("counts") or {}).get("oracle", 0)}
    row = make_row(item=item["id"], mode=mode, requirement=requirement, plan=plan, res_steps=ex.get("steps"), goal_ok=res.goal_ok,
                    exec_ok=core.get("exec_ok"), inapplicable=res.inapplicable, audit=audit, attempt=attempt,
                    elapsed_s=time.time() - t0, review=review, kitchen=kitchen, phase=phase,
                    run={"policy": getattr(executor, "policy", None), "scene": getattr(executor, "scene", kitchen),
                         "frame": ex.get("frame"), "map_rev": ex.get("map_rev"), "detector_rev": ex.get("detector_rev"),
                         "world": req.world, "memory": "none", "plan_source": "reference", "repeat_of_failed": attempt > 1,
                         "executor_sha": _executor_sha()})
    if row["class"] not in ("feasible", "inapplicable"):
        row["trace_n"] = len(buf)
        row["trace_tail"] = [str(x)[:240] for x in buf[-int(os.environ.get("L3FEAS_TRACE_LINES", "120")):]]
    return row


def jobs_refplans(items, manifest, templates_path=None) -> list:
    ids = l3_nontest_ids(items, manifest) + [i for i in l2_support_ids(items, manifest, templates_path)]
    return [{"mode": "refplans", "item": i, "requirement": "refplan", "plan": items[i]["plan"]} for i in ids]


def jobs_routes(items, manifest, path, mode, key) -> list:
    """B4 composite routes (mode procedures) or B6 skeletons (mode templates), each bound to its support items' configurations."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return []
    entries = d.get("entries") if isinstance(d, dict) and "entries" in d else (d if isinstance(d, list) else d.get("templates", []))
    nt = set(manifest["nontest"])
    jobs = []
    for e in entries or []:
        sup = [s for s in ((e.get("provenance") or {}).get("support") or e.get("support") or []) if s in nt and s in items]
        steps = e.get(key) or []
        if not steps or not sup:
            continue
        cfgs = distinct_configs([items[s] for s in sup], need=2 if mode == "procedures" else len(sup))
        for it in cfgs:
            bound, unbound = bind_steps(steps, it["plan"])
            jobs.append({"mode": mode, "item": it["id"], "requirement": e["id"], "plan": bound, "unbound": unbound,
                         "full_plan": plan_tools(bound) == plan_tools(it["plan"])})
    return jobs


def _print_row(r) -> None:
    print(json.dumps({"item": r["item"], "req": r["requirement"], "attempt": r["attempt"], "class": r["class"], "auto": r["class_auto"],
                      "goal_ok": r["goal_ok"], "why": r["why"], "s": r["elapsed_s"]}, ensure_ascii=False), flush=True)


def done_keys(rows) -> set:
    return {row_key(r) for r in rows}


def run_jobs(jobs, executor, items, *, kitchen, review, rows_path, retry_failed=1, log=None, phase="pre") -> list:
    """Resumable: a job whose (mode,item,requirement,attempt) row exists is skipped. A failed job is repeated up to `retry_failed` times
    (the live simulator is not deterministic; a flaky item is reported as such, not hidden). Fail fast on an invariant error."""
    log = log or _print_row
    rows = read_rows(rows_path)
    have = done_keys(rows)
    new = []
    for j in jobs:
        it = items[j["item"]]
        for attempt in range(1, 2 + max(0, retry_failed)):
            k = (j["mode"], j["item"], j["requirement"], attempt)
            if k in have:
                r = next(x for x in rows + new if row_key(x) == k)
            else:
                if j.get("unbound"):
                    r = make_row(item=j["item"], mode=j["mode"], requirement=j["requirement"], plan=j["plan"], res_steps=[], goal_ok=None,
                                 exec_ok=None, inapplicable=f"slots not bound from the support item: {j['unbound']}", audit={}, attempt=attempt,
                                 elapsed_s=0, review=review, kitchen=kitchen, phase=phase)
                else:
                    r = run_one(executor, it, j["plan"], mode=j["mode"], requirement=j["requirement"], attempt=attempt, kitchen=kitchen,
                                review=review, phase=phase)
                r["full_plan"] = j.get("full_plan")
                append_row(r, rows_path)
                new.append(r)
                have.add(k)
                log(r)
                if r["class"] == "invariant_error":
                    raise RuntimeError(f"invariant error on {r['item']}: {r['why']}")
            if r["class"] in ("feasible", "inapplicable"):
                break
    return new


def write_procedure_results(rows, path=None) -> dict:
    """The result file B4's loader reads: a procedure is wsl-verified iff its route executed ok on >= 2 DISTINCT non-test configurations."""
    byp = {}
    for r in rows:
        if r["mode"] != "procedures":
            continue
        e = byp.setdefault(r["requirement"], {"ok_configs": set(), "tried": set()})
        e["tried"].add(r["item"])
        if r.get("exec_ok") and r["class"] in ("feasible", "unreviewed", "executor_defect") and not r.get("inapplicable") and \
                all(s["ok"] for s in r["steps"]):
            e["ok_configs"].add(r["item"])
    out = {"schema": "pkg.feas.proc/1", "procedures": {
        k: {"wsl": len(v["ok_configs"]) >= 2, "ok_configs": sorted(v["ok_configs"]), "tried": sorted(v["tried"])} for k, v in sorted(byp.items())}}
    p = path or PROC_RESULT_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, sort_keys=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B9 L3 executor-feasibility audit (WSL, non-test items only)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--refplans", action="store_true")
    g.add_argument("--procedures", action="store_true")
    g.add_argument("--templates", action="store_true")
    g.add_argument("--summarize", action="store_true", help="offline: rows -> summary json")
    ap.add_argument("--phase", choices=("pre", "post"), default="pre",
                    help="pre = the audit before any executor change (immutable history); post = the re-run after fixes (separate file)")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--scene", default="FloorPlan1", help="the kitchen with a walked map and detector on disk (only FloorPlan1 today)")
    ap.add_argument("--retry-failed", type=int, default=1, help="repeat a failed job this many times (simulator non-determinism)")
    ap.add_argument("--only", default=None, help="comma-separated item ids (must be non-test)")
    a = ap.parse_args(argv)

    C, manifest = _load_split()
    items = load_dataset()
    review = load_review()
    l3 = l3_nontest_ids(items, manifest)
    meta = {"dataset_sha256": manifest.get("dataset_sha256"), "split_sha256": manifest.get("sha256")}
    rows_path = a.rows or (ROWS_PATH if a.phase == "pre" else ROWS_POST_PATH)
    if a.summarize:
        S = full_summary(l3, meta)
        _write_summary(S)
        print(json.dumps({ph: {m: x["by_class"] for m, x in S[ph]["modes"].items()} for ph in ("pre", "post") if ph in S}, ensure_ascii=False))
        return 0
    mode = "refplans" if a.refplans else "procedures" if a.procedures else "templates"
    if mode == "refplans":
        jobs = jobs_refplans(items, manifest)
    elif mode == "procedures":
        jobs = jobs_routes(items, manifest, PROCEDURES_PATH, "procedures", "route")
    else:
        jobs = jobs_routes(items, manifest, TEMPLATES_PATH, "templates", "skeleton")
    if a.only:
        want = set(x.strip() for x in a.only.split(",") if x.strip())
        jobs = [j for j in jobs if j["item"] in want]
    if not jobs:
        print(json.dumps({"mode": mode, "jobs": 0, "note": "nothing to audit (the B4 / B6 file is not written yet, or no job matched)"}))
        return 0
    C.guard_items(sorted({j["item"] for j in jobs}), NAMESPACE)          # refuses any test item, BEFORE the simulator boots
    forbid = frozenset(manifest["test"]) | frozenset(manifest.get("test_uncapped_not_scored") or [])
    allow = frozenset(manifest["nontest"])
    import step5_live as sl
    ex = sl.LiveExecutor(memory="none", policy="console", allow=allow, forbid=forbid, scene=a.scene)
    try:
        run_jobs(jobs, ex, items, kitchen=a.scene, review=review, rows_path=rows_path, retry_failed=a.retry_failed, phase=a.phase)
    finally:
        ex.close()
    if mode == "procedures" and a.phase == "pre":
        write_procedure_results(read_rows(rows_path))
    S = full_summary(l3, meta)
    _write_summary(S)
    print(json.dumps({ph: {m: x["by_class"] for m, x in S[ph]["modes"].items()} for ph in ("pre", "post") if ph in S}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
