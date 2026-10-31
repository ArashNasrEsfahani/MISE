"""Why Abstract (L3) instructions fail, under the re-executed analysis. Read only on all inputs; writes only l3_breakdown.json (see run.py).

    python docs/final-test/reexecuted/run.py        (no GPU, no simulator, no language model)

Rows    compute_final's own path, by import: R's re-run rows -> apply_rescue -> A36.flagged rows left out. Gate G1 proves they are the same rows.
Plans   the recorded plan cell of each row (<ns>/plan.jsonl, latest row of r["cell"]); never regenerated. Gate G4 proves it is the scored plan.
Goals   the gold item's goal clauses, as the scorer's goal check reads them. Each clause becomes one plan_walk predicate (CLAUSE below).
Walk    plan_walk.World, the executor's action model, unchanged. Its start is the recorded truth of each clause at reset (the capture's state file,
        labels.goal_clauses, after the item's setup; extracted into inputs/l3_reset_states.json), so a clause already true and left alone is not "missing". A clause is missing when it is False after the last step.

Each L3 failure gets one class: no_plan (plan_status != ok), misses_end_state (at least one clause False on paper), reaches_on_paper (none False).
On the same plans, the two preconditions plan_walk.check_plan tests (slice without the ButterKnife in hand, toggle_on Microwave with its door open) are counted per class: such a plan can
reach its end-states on paper and still be a planning error.
Separately, how it ended (compute_final.failure_breakdown's definition): refused_plan / all_steps_no_goal / failed_skill_step.
Counts are plain presentation counts, no weighting and no bootstrap.
"""
import sys

sys.dont_write_bytecode = True

import collections   # noqa: E402
import hashlib       # noqa: E402
import json          # noqa: E402
import os            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compute_final as CF   # noqa: E402

CN, CA, AN, A36, S, NS, ROOT, S_ = CF.CN, CF.CA, CF.AN, CF.A36, CF.S, CF.NS, CF.ROOT, CF.S_
import plan_walk as MV       # noqa: E402
import general_set as GS     # noqa: E402  (sim/, on sys.path through adjusted)

OUT = os.path.join(HERE, "l3_breakdown.json")
RESET_STATES = os.path.join(CF.INPUTS, "l3_reset_states.json")
ARMS = ("G+M", "O+M")
CLASSES = ("misses_end_state", "reaches_on_paper", "no_plan")
ENDINGS = ("all_steps_no_goal", "failed_skill_step", "refused_plan")
#: the counts an earlier version of this analysis recorded on the sealed rows, for the reconstruction check
OLDER = {"G+M": {"presentations": 121, "failures": 88, "misses_end_state": 70, "reaches_on_paper": 17, "no_plan": 1},
         "O+M": {"presentations": 121, "failures": 22, "misses_end_state": 13, "reaches_on_paper": 9, "no_plan": 0}}
OLDER_KINDS = {"in": 24, "sliced": 23, "filled": 16, "off": 13, "on": 12, "closed": 7, "clean": 6, "empty": 5, "broken": 4, "cooked": 1}
gates = []


def gate(name, ok, detail=None):
    gates.append({"gate": name, "pass": bool(ok), "detail": detail})


# ---------------------------------------------------------------------------------------------- gold clause -> predicate
CLAUSE = {("isFilledWithLiquid", True): "filled", ("isFilledWithLiquid", False): "empty", ("isToggled", True): "on", ("isToggled", False): "off",
          ("isOpen", True): "open", ("isOpen", False): "closed", ("isDirty", False): "clean", ("isDirty", True): "dirty", ("isSliced", True): "sliced",
          ("isBroken", True): "broken", ("isCooked", True): "cooked", ("temperature", "Hot"): "hot"}


def predicate(a):
    """One gold clause -> a plan_walk predicate (kind, type, arg). None: not an executor state."""
    k, t = a.get("key"), a.get("type")
    if k == "parentReceptacles":
        return ("in", t, a.get("contains"))
    if k == "isPickedUp" and a.get("value") is True:
        return ("held", t, None)
    if k == "fillLiquid":
        return ("state", t, "filled")
    st = CLAUSE.get((k, a.get("value")))
    return ("state", t, st) if st else None


def kind_of(p):
    return p[0] if p[0] != "state" else p[2]


# ---------------------------------------------------------------------------------------------- the walk on paper
def start_from_labels(clauses, labels, holding):
    """The world at reset as far as the goal needs it: each clause's recorded truth (after the setup)."""
    w = MV.World(holding)
    for a, p in clauses:
        true_now = labels.get(GS.format_goal(a))
        if p is None or true_now is None:
            continue
        if p[0] == "in":
            if true_now:
                w.loc[p[1]] = MV._place(p[2])
        elif p[0] == "held":
            if true_now:
                w.hand = p[1]
        elif p[2] in MV.PROP:
            prop, val = MV.PROP[p[2]]
            w.prop[(p[1], prop)] = val if true_now else (not val)
    return w


def start_from_setup(setup, holding, summary=None):
    """Variant for the older rows only: the item's setup walked from an empty world, optionally with the summary's seen states."""
    w = MV.World(holding, MV.observed_states(summary) if summary else None)
    for i, st in enumerate(setup or [], 1):
        w.step(i, st)
    return w


def walk(plan, clauses, w):
    """-> (missing [(pred text, kind)], unchecked clause texts)."""
    fills = {}
    for i, st in enumerate(plan or [], 1):
        d = MV.as_step(st)
        if d.get("action") == "fill":
            fills[d.get("object")] = d.get("arg")
        w.step(i, d)
    missing, unchecked = [], []
    for a, p in clauses:
        if p is None or w.holds(p) is None:
            unchecked.append(GS.format_goal(a))
        elif w.holds(p) is False or (a.get("key") == "fillLiquid" and fills.get(a["type"]) not in (None, a.get("value"))):
            missing.append((MV.pred_text(p), kind_of(p)))
    return missing, unchecked


def plan_text(plan):
    out = []
    for st in plan or []:
        d = MV.as_step(st)
        out.append(" ".join(str(x) for x in (d.get("action"), d.get("object"), d.get("to"), d.get("arg")) if x not in (None, "")))
    return " ; ".join(out)


def ending(r):
    """compute_final.failure_breakdown's three-way split, unchanged."""
    if r.get("plan_status") != "ok":
        return "refused_plan"
    return "all_steps_no_goal" if r.get("steps_ok_n") == r.get("steps_n") else "failed_skill_step"


# ---------------------------------------------------------------------------------------------- inputs
def load_records():
    """Per kitchen: the latest plan row per cell (<ns>/plan.jsonl), and the reset states of the Abstract presentations."""
    plans, caps = {}, {}
    with open(RESET_STATES, encoding="utf-8") as f:
        reset = json.load(f)["presentations"]
    for ns in NS.values():
        plans[ns] = S.C.Journal(S.C.journal_path(ns, "plan")).latest()
        caps[ns] = reset[ns]
    return plans, caps


def reset_labels(ns, pres, caps):
    """pres -> (labels at reset, holding, summary lines) from its captures; all captures of a presentation must agree."""
    rows = sorted(caps[ns].get(pres) or [], key=lambda c: c.get("capture_run") or 0)
    if not rows:
        return None
    seen = [c["goal_clauses"] for c in rows]
    if any(x != seen[0] for x in seen):
        raise SystemExit(f"captures of {pres} disagree on the goal clauses at reset")
    return seen[-1], rows[-1].get("holding"), rows[-1].get("summary_lines")


def classify(r, plans, caps, items, start="labels"):
    ns, it = r["namespace"], items[r["item"]]
    clauses = [(a, None if start == "setup-no-temperature" and a.get("key") == "temperature" else predicate(a)) for a in it.get("goal") or []]
    labels, holding, summary = reset_labels(ns, r["pres"], caps) or ({}, None, None)
    pr = plans[ns].get(r["cell"])
    plan = (pr or {}).get("plan")
    if r.get("plan_status") != "ok":
        return {"cls": "no_plan", "missing": [], "unchecked": [], "plan": plan, "prec": []}
    if start == "labels":
        w = start_from_labels(clauses, labels, holding)
    else:
        w = start_from_setup(it.get("setup"), holding, summary if start == "setup+seen" else None)
    missing, unchecked = walk(plan, clauses, w)
    prec = [c for c, _f in MV.check_plan(plan, {"status": "preconditions only", "desired": []}, holding, summary)]
    return {"cls": "misses_end_state" if missing else "reaches_on_paper", "missing": missing, "unchecked": unchecked, "plan": plan, "prec": sorted(set(prec))}


def l3_rows(idx, arm):
    return [r for k, r in idx["S"].items() if k[0] == arm and r.get("level") == "L3" and not A36.flagged(r)]


def tally(rows, plans, caps, items, oc=None, start="labels", detail=False):
    t = {"presentations": len(rows), "failures": 0, **{c: 0 for c in CLASSES}, **{e: 0 for e in ENDINGS}, "by_class_and_ending": {}, "kinds": collections.Counter(),
         "missing_clauses": 0, "preconditions_by_class": {}, "first_failed_step_causes": collections.Counter(), "not_executed_failures": 0, "successes_walk_says_missing": 0, "rows": []}
    for r in rows:
        c = classify(r, plans, caps, items, start)
        if r.get("sr"):
            if c["missing"]:
                t["successes_walk_says_missing"] += 1
                if detail:
                    t.setdefault("success_rows_walk_says_missing", []).append({"pres": r["pres"], "plan": plan_text(c["plan"]), "missing": [m[0] for m in c["missing"]]})
            continue
        t["failures"] += 1
        t[c["cls"]] += 1
        e = ending(r)
        t[e] += 1
        key = f"{c['cls']} / {e}"
        t["by_class_and_ending"][key] = t["by_class_and_ending"].get(key, 0) + 1
        t["not_executed_failures"] += 0 if r.get("executed") else 1
        for code in c["prec"]:
            key2 = f"{c['cls']} / {code}"
            t["preconditions_by_class"][key2] = t["preconditions_by_class"].get(key2, 0) + 1
        for _txt, kind in c["missing"]:
            t["kinds"][kind] += 1
            t["missing_clauses"] += 1
        cause = None
        if oc is not None and e == "failed_skill_step":
            steps = (oc.get(f"{r['namespace']}|exec|{r['unit']}|run0") or {}).get("steps") or []
            first = next((s for s in steps if not s.get("ok")), {})
            cause = first.get("cause") or first.get("failure_cause") or "unknown"
            t["first_failed_step_causes"][cause] += 1
        if detail:
            lang = r.get("lang")
            t["rows"].append({"pres": r["pres"], "item": r["item"], "lang": lang, "instruction": items[r["item"]].get(lang), "class": c["cls"], "ending": e, "first_failed_cause": cause,
                              "plan": plan_text(c["plan"]), "missing": [m[0] for m in c["missing"]], "not_checked": c["unchecked"], "precondition_problems": c["prec"]})
    t["kinds"] = dict(t["kinds"].most_common())
    t["first_failed_step_causes"] = dict(t["first_failed_step_causes"].most_common())
    if not detail:
        t.pop("rows")
    return t


def pool(per):
    out = {f: sum(v[f] for v in per.values()) for f in ("presentations", "failures", *CLASSES, *ENDINGS, "missing_clauses", "not_executed_failures", "successes_walk_says_missing")}
    for f in ("kinds", "first_failed_step_causes", "by_class_and_ending", "preconditions_by_class"):
        c = collections.Counter()
        for v in per.values():
            c.update(v[f])
        out[f] = dict(c.most_common())
    return out


# ---------------------------------------------------------------------------------------------- run
def main():
    fr = json.load(open(CF.OUT, encoding="utf-8"))
    src = CN.sources()
    RAW = {K: (sum((AN.read_jsonl(p) for p in orig), []), sum((AN.read_jsonl(p) for p in new), [])) for K, (orig, new) in src.items()}
    gold, gold_sha = S._gold_load(None, None, verify=True)
    rs = CF.rescue_sources()
    RAW_final = {K: CF.apply_rescue(RAW[K][1], rs, gold) for K in NS}
    I_fin = {K: CF.idx_of(RAW_final[K]) for K in NS}
    with open(S.C.paths(None)["dataset"], "rb") as f:
        raw = f.read()
    gate("G0 gold file is the one the scorer verified", hashlib.sha256(raw).hexdigest() == gold_sha, gold_sha[:16])
    items = {i["id"]: i for i in json.loads(raw.decode("utf-8"))["items"]}
    plans, caps = load_records()
    oc = CF.outcome_lookup()

    # G1: the rows are compute_final's rows
    fb = CF.failure_breakdown(RAW_final)
    gate("G1 failure_breakdown recomputed from these rows equals final_results.json, all levels and arms", fb == fr["failure_breakdown"])
    inp = [p for K in src for p in src[K][1]]
    gate("G1b the input re-run rows have the hashes final_results.json recorded", all(fr["inputs"].get(CN.rel(p)) == CN.sha(p) for p in inp))

    res = {"schema": "reexecuted.l3_breakdown/1", "definition": " ".join(__doc__.split("\n\n")[2:]).replace("\n", " "), "kitchens": {"K1": "development", "K2": "held-out A", "K3": "held-out B"}, "arms": {}, "gates": gates}
    for arm in ARMS:
        per = {K: tally(l3_rows(I_fin[K][0], arm), plans, caps, items, oc, detail=True) for K in NS}
        res["arms"][arm] = {"pooled": pool(per), "per_kitchen": per}

    # G2: failures and presentations against the rates; the rate is an item mean, so it is rebuilt from the failure flags
    d2 = {}
    for arm in ARMS:
        for K in NS:
            rows = l3_rows(I_fin[K][0], arm)
            rate = fr["rates"][K][arm]
            by = collections.defaultdict(list)
            for r in rows:
                by[r["item"]].append(1.0 if r.get("sr") else 0.0)
            mean = sum(sum(v) / len(v) for v in by.values()) / len(by)
            n_ok = len(rows) == rate["level_presentations"]["L3"]
            sr_ok = abs(mean - rate["level"]["L3"]["sr"]) <= 1e-12 and len(by) == rate["level"]["L3"]["n_items"]
            d2[f"{K} {arm}"] = {"presentations": len(rows), "recorded_presentations": rate["level_presentations"]["L3"], "failures": sum(1 for r in rows if not r.get("sr")),
                                "item_mean_from_rows": mean, "recorded_L3_sr": rate["level"]["L3"]["sr"], "pass": n_ok and sr_ok}
    gate("G2 per arm and kitchen: L3 presentations and failure flags reproduce final_results.json rates exactly", all(v["pass"] for v in d2.values()), d2)

    # G3: all steps ran, goal missed
    d3 = {}
    for K in NS:
        for arm in ARMS:
            want = fr["diagnostics"]["steps_complete_goal_failed_L3_all_kitchens"][K][arm]
            v = res["arms"][arm]["per_kitchen"][K]
            d3[f"{K} {arm}"] = {"all_steps_no_goal": v["all_steps_no_goal"], "presentations": v["presentations"], "recorded": want, "pass": [v["all_steps_no_goal"], v["presentations"]] == want}
    gate("G3 all steps ran without the goal equals the recorded diagnostics (K1 G+M 27 of 55, O+M 4 of 55; K2, K3 as recorded)", all(v["pass"] for v in d3.values()), d3)

    # G4: the plan walked is the plan scored
    d4 = {"rows": 0, "no_plan_cell": 0, "status_differs": 0, "metrics_differ": 0, "plan_len_differs": 0, "steps_differ_from_plan": 0, "steps_compared": 0}
    for K in NS:
        for arm in ARMS:
            for r in l3_rows(I_fin[K][0], arm):
                pr = plans[r["namespace"]].get(r["cell"])
                d4["rows"] += 1
                if pr is None:
                    d4["no_plan_cell"] += 1
                    continue
                d4["status_differs"] += pr.get("status") != r.get("plan_status")
                if r.get("plan_status") != "ok":
                    continue
                m = S.plan_metrics(pr.get("plan"), gold[r["item"]]["plan"])
                d4["metrics_differ"] += abs(m["pc"] - r["pc_final"]) > 1e-12 or abs(m["tds"] - r["tds_final"]) > 1e-12
                d4["plan_len_differs"] += len(S.C.norm_plan(pr.get("plan"))) != r.get("plan_len")
                steps = (oc.get(f"{r['namespace']}|exec|{r['unit']}|run0") or {}).get("steps")
                if steps:
                    d4["steps_compared"] += 1
                    ran = [(s.get("action"), s.get("object")) for s in steps]
                    want = [(MV.as_step(x).get("action"), MV.as_step(x).get("object")) for x in pr.get("plan") or []]
                    d4["steps_differ_from_plan"] += ran != want[:len(ran)]
    gate("G4 the recorded plan is the scored plan (status, PC and TDS recomputed, length, executed steps a prefix of the plan)",
         not any(d4[k] for k in ("no_plan_cell", "status_differs", "metrics_differ", "plan_len_differs", "steps_differ_from_plan")), d4)

    # G5: the gold reference plans must reach every clause on paper, from the same start. Tests the action model and the clause mapping.
    d5 = {"presentations": 0, "reference_misses": [], "unchecked_clauses": collections.Counter()}
    for K, ns in NS.items():
        for pres, cs in caps[ns].items():
            item = cs[0].get("item")
            if items.get(item, {}).get("level") != "L3":
                continue
            clauses = [(a, predicate(a)) for a in items[item]["goal"]]
            labels, holding, _s = reset_labels(ns, pres, caps)
            missing, unchecked = walk(items[item]["plan"], clauses, start_from_labels(clauses, labels, holding))
            d5["presentations"] += 1
            d5["unchecked_clauses"].update(unchecked)
            if missing:
                d5["reference_misses"].append({"pres": f"{K} {pres}", "missing": [m[0] for m in missing]})
    d5["unchecked_clauses"] = dict(d5["unchecked_clauses"])
    gate("G5 every gold reference plan reaches its goal on paper; no clause outside the action model", not d5["reference_misses"] and not d5["unchecked_clauses"], d5)

    # the older rows: sealed, flagged rows left out
    I_old = {K: CF.idx_of(RAW[K][0]) for K in NS}
    older = {"recorded": OLDER, "recorded_kinds_G+M": OLDER_KINDS, "reconstructed": {}}
    for start in ("labels", "setup", "setup-no-temperature"):
        older["reconstructed"][start] = {}
        for arm in ARMS:
            p = pool({K: tally(l3_rows(I_old[K][0], arm), plans, caps, items, start=start) for K in NS})
            older["reconstructed"][start][arm] = {f: p[f] for f in ("presentations", "failures", *CLASSES)} | {"kinds": p["kinds"], "successes_walk_says_missing": p["successes_walk_says_missing"]}
    res["older_analysis_check"] = older
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")

    sys.stdout.reconfigure(encoding="utf-8")
    for g in gates:
        print("PASS" if g["pass"] else "FAIL", g["gate"])
        if not g["pass"]:
            print("   ", json.dumps(g["detail"], ensure_ascii=False)[:1500])
    for arm in ARMS:
        a = res["arms"][arm]
        for lab, v in [(K, a["per_kitchen"][K]) for K in NS] + [("pooled", a["pooled"])]:
            print(arm, lab, {f: v[f] for f in ("presentations", "failures", *CLASSES, *ENDINGS, "not_executed_failures", "successes_walk_says_missing")})
    print("wrote", OUT)
    if not all(g["pass"] for g in gates):
        raise SystemExit("a gate failed (see above)")


if __name__ == "__main__":
    main()
