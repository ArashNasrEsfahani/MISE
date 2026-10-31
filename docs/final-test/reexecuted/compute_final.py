"""The re-executed analysis: every rate, contrast and count of the final test under one rule, from the records. Read only on all inputs.

Re-executed: a presentation lost to a restoration failure uses its re-execution when it was restored; when it failed restoration again it is
LEFT OUT of the denominator, and in a paired contrast a presentation out in either arm leaves both (the A36 rule). Row-wise: task R's rows
(R/<ns>/score*.noexcl.jsonl; A6 / V / G: their `no-exclusion` rows), then the rescue passes, with the adjusted estimator (A36.flagged drops, adjusted.without_flagged).
Estimators (by import, unchanged): rates adjusted.tables; K1 paired paired.paired (95 %, B = 20000, seed 20261010); K1 Holm
analyze.holm_family with drop_flag (B = 9999, as the A36 block of test-v1/analysis.json); K2, K3 k23_analyze.kitchen_cr2 and held-out mean
pooled_result (nominal 96.5 % CR2); three-kitchen mean: equal weights, kitchen-stratified family bootstrap.

One path, three settings of (rows, drop); the first two are gates, the script stops if either fails:
  G1  sealed rows, drop = True   -> must reproduce the adjusted values (inputs/adjusted_reference.json, analysis.json a36, R adjusted, 13.0, 16.2, ...)
  G2  re-run rows, drop = False  -> must reproduce no-exclusion (R noexcl, A6 / V / G no-exclusion, 13.6, 16.4, ...)
  new re-run rows + rescue, drop = True -> this analysis; before the rescue its rates must equal R's `residual_adjusted` block (G3).
The A6 / V / G checks of the adjusted version need those tasks' adjusted-version rows, which are not in this repository; they are skipped.
Writes only final_results.json (see run.py).

    python docs/final-test/reexecuted/run.py        (any directory; no simulator, no language model, no network)
"""
import sys

sys.dont_write_bytecode = True

import json   # noqa: E402
import os     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noexcl as CN             # noqa: E402
import score as S               # noqa: E402  (sim/final_test/matrix, on sys.path through adjusted; read only)

CA, PLD, AN, A36, KA, P = CN.CA, CN.PLD, CN.AN, CN.A36, CN.KA, CN.P
NS, PLANNERS, FRONTIER, LEVELS, S_, ROOT, INPUTS = CN.NS, CN.PLANNERS, CN.FRONTIER, CN.LEVELS, CN.S_, CN.ROOT, CN.INPUTS
SCENE, p1, pc1 = CN.SCENE, CN.p1, CN.pc1
OUT = os.path.join(HERE, "final_results.json")
ARMS = {"K1": CA.K1_ARMS, "K2": CA.K23_ARMS, "K3": CA.K23_ARMS}
B_K1 = 9999
A6_MAIN = ("G0", "G+M", "O0", "O+M", "G+M/p", "O+M/p")
GATES_LOG = []
bad = []


def same(a, b, tol=1e-9):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def chk(name, got, want):
    ok = got == want
    GATES_LOG.append([name, got, want, ok])
    if not ok:
        bad.append(f"{name}: got {got}, expected {want}")


def k1_contrasts(i, d, drop):
    S, fam = i["S"], i["fam"]
    kw = {"drop_flag": A36.flagged} if drop else {}
    h1 = AN.h1_block(S, fam, d, B_K1, True, **kw)
    hf = AN.holm_family(S, fam, d, B_K1, True, **kw)
    ex, _ = AN.exploratory_contrasts(S, fam, d, B_K1, hf, **kw)
    dec = (h1.get("primary") or {}).get("decision") or {}
    keep = ("a", "b", "estimate", "ci95", "p", "p_holm", "reject", "n_pres", "n_items", "n_families", "dropped_flagged")
    return {"H1 G+M - O0": {"estimate": dec.get("estimate"), "interval": dec.get("interval"), "n_pres": (h1.get("primary") or {}).get("n_pres")},
            **{f"{hid} {v['a']} - {v['b']}": {k: v.get(k) for k in keep} for hid, v in hf.items()},
            **{f"{hid} {v['a']} - {v['b']}": {k: v.get(k) for k in keep if k not in ("p_holm", "reject")} for hid, v in ex.items()}}


def parts_safe(idx, d, a, b, L, drop):
    try:
        return CN.parts_of(idx, d, a, b, L, drop)
    except AN.AnalysisError:
        return None


def bundle(I, drop):
    """I: {K: (idx, invalid)}. Everything reported from the main rows under one (rows, drop) setting."""
    T = {K: CA.tables(CA.without_flagged(I[K][0]) if drop else I[K][0], I[K][1], ARMS[K]) for K in NS}
    LD = {K: {f"{a} - {b}": {L: PLD.paired(I[K][0], I[K][1], a, b, L, drop) for L in LEVELS} for a, b in CN.LEVEL_PAIRS} for K in NS}
    PARTS = {K: CN.k23_parts(I[K][0], I[K][1], drop) for K in ("K2", "K3")}
    C23 = {K: {k: CA.cr2_public(v) for k, v in PARTS[K].items()} for K in PARTS}
    fam = dict(I["K2"][0]["fam"], **I["K3"][0]["fam"])
    pick = lambda r: {k: r.get(k) for k in ("estimate", "interval", "se", "nu", "interval_label", "n_families", "n_items", "available")}  # noqa: E731
    HM = {key: pick(KA.pooled_result({SCENE[K]: PARTS[K][key] for K in PARTS}, fam)) for key in PARTS["K2"]}
    TK = {}
    for key in ("G+M - G0", "O+M - O0", "X+M - X0"):
        a, b = key.split(" - ")
        TK[key] = {L: CN.stratified_mean([parts_safe(I[K][0], I[K][1], a, b, L, drop) for K in NS]) for L in LEVELS}
    return {"T": T, "LD": LD, "C23": C23, "HM": HM, "TK": TK, "K1": k1_contrasts(I["K1"][0], I["K1"][1], drop)}


def gate_main(B_, pub_rates, pub_c1, pub_c23, pub_hm, pub_ld, label):
    for K in NS:
        for a in ARMS[K]:
            t = B_["T"][K][a]
            p = pub_rates[K][a]
            if CN.rate_pp2(t["overall"]) != {k: p["overall"][k] for k in ("sr", "ci95", "n_items")}:
                bad.append(f"{label} {K} {a} overall {CN.rate_pp2(t['overall'])} vs {p['overall']}")
            for L in ("L1", "L2", "L3"):
                if (CN.rate_pp2(t["level"].get(L)) or {}).get("sr") != (p["level"].get(L) or {}).get("sr"):
                    bad.append(f"{label} {K} {a} {L}")
            for g in ("en", "fa"):
                if (CN.rate_pp2(t["lang"].get(g)) or {}).get("sr") != (p["lang"].get(g) or {}).get("sr"):
                    bad.append(f"{label} {K} {a} {g}")
            if t["n_presentations"] != p["n_presentations"]:
                bad.append(f"{label} {K} {a} n")
    for key, v in pub_c1.items():
        m = B_["K1"].get(key)
        if m is None:
            continue
        if not same(m.get("estimate"), v.get("estimate")):
            bad.append(f"{label} K1 {key} estimate")
        if v.get("ci95") is not None and not same(m.get("ci95"), v.get("ci95")):
            bad.append(f"{label} K1 {key} ci95")
        if v.get("p_holm") is not None and not same(m.get("p_holm"), v.get("p_holm")):
            bad.append(f"{label} K1 {key} p_holm {m.get('p_holm')} vs {v.get('p_holm')}")
    for K in ("K2", "K3"):
        for key, v in pub_c23[K].items():
            m = B_["C23"][K][key]
            if not (same(m["estimate"], v["estimate"]) and same(m["interval"], v["interval"])):
                bad.append(f"{label} {K} {key}")
    for key, v in pub_hm.items():
        if not (same(B_["HM"][key]["estimate"], v["estimate"]) and same(B_["HM"][key]["interval"], v["interval"])):
            bad.append(f"{label} held-out {key}")
    for K in NS:
        for key, byL in pub_ld[K].items():
            for L, v in byL.items():
                m = B_["LD"][K][key][L]
                if not (same(m["estimate"], v["estimate"]) and same(m["ci95"], v["ci95"]) and m["n_presentations"] == v["n_presentations"]):
                    bad.append(f"{label} {K} level {key} {L}")


def v_like(rows_all, base, version, contrasts, drop, want_levels=LEVELS):
    """V / G: per-kitchen paired contrasts and the three-kitchen means, overall, per level and per language, as v_analyze.analyse."""
    out = {"kitchens": {}, "three_kitchen": {}}
    pts = {}
    for K, ns in NS.items():
        rows = [r for r in rows_all if r.get("namespace") == ns and r.get("v_version") == version]
        allr = base[K] + rows
        idx = AN.build_index(allr)
        d = AN.invalid_presentations(idx["S"])
        kb = {"contrasts": {}, "by_lang": {}, "rates": {}}
        arms = sorted({k[0] for k in idx["S"]})
        for a in arms:
            sel = {k: r for k, r in idx["S"].items() if k[0] == a}
            kept = [r for r in sel.values() if not (drop and A36.flagged(r))]
            kb["rates"][a] = {"presentations": len(sel), "retained": len(kept), "left_out": len(sel) - len(kept)}
        for x, y in contrasts:
            n = f"{x} - {y}"
            kb["contrasts"][n] = {L: PLD.paired(idx, d, x, y, L, drop) for L in want_levels}
            for L in want_levels:
                pts[(n, L)] = pts.get((n, L), []) + [parts_safe(idx, d, x, y, L, drop)]
            for lg in ("en", "fa"):
                il = CN.lang_filter(idx, lg)
                kb["by_lang"].setdefault(lg, {})[n] = PLD.paired(il, d, x, y, "all", drop)
                pts[(n, "all_" + lg)] = pts.get((n, "all_" + lg), []) + [parts_safe(il, d, x, y, "all", drop)]
        out["kitchens"][K] = kb
    for (n, L), p in pts.items():
        out["three_kitchen"].setdefault(n, {})[L] = CN.stratified_mean(p) if all(p) else {"available": False}
    return out


def gate_v_like(mine, pub, version, label):
    for n, byL in mine["three_kitchen"].items():
        for L, m in byL.items():
            v = pub["three_kitchen"][version][n].get(L)
            if not m.get("estimate_pp") == (v or {}).get("estimate_pp") or m.get("ci95_pp") != (v or {}).get("ci95_pp"):
                bad.append(f"{label} {version} three-kitchen {n} {L}: {m.get('estimate_pp')} {m.get('ci95_pp')} vs {(v or {}).get('estimate_pp')} {(v or {}).get('ci95_pp')}")
    for K, ns in NS.items():
        for n, byL in mine["kitchens"][K]["contrasts"].items():
            for L, m in byL.items():
                v = pub["kitchens"][ns]["versions"][version]["contrasts"][n][L]
                if not (same(m["estimate"], v["estimate"]) and same(m["ci95"], v["ci95"])):
                    bad.append(f"{label} {version} {K} {n} {L}")


# ---------------------------------------------------------------------------------------------- the rescue pass (LAST re-run source)
EXEC_FIELDS = ("sr", "strict_sr", "executed", "exec_failure", "failure_class", "run_class", "exec_attempt", "exec_status", "goal_ok", "exec_ok", "steps_n", "steps_ok_n",
               "unrealistic_slice", "n_slice", "n_slice_no_knife", "n_slice_unknown", "excluded_reset", "excluded_reset_evidence", "unit")
COMPARE_SKIP = ("ts", "rerun", "rerun_pass", "orig_sr", "orig_exec_status", "orig_excluded_reset", "excluded_reset_evidence")


class Source:
    """One re-execution record triple (exec, outcome, attempt starts); `recovered(unit)` -> (exec row, outcome, history, starts) when its
    last attempt ran to an outcome."""

    def __init__(self, ns, exec_p, outcome_p, events_p=None):
        self.ns = ns
        self.rows = AN.read_jsonl(exec_p) if os.path.exists(exec_p) else []
        self.hist, self.last = {}, {}
        for r in self.rows:
            self.hist.setdefault(r.get("key"), []).append(r)
            self.last[r.get("key")] = r
        self.outc = {}
        if os.path.exists(outcome_p):
            for r in AN.read_jsonl(outcome_p):
                self.outc[r.get("key")] = r
        self.starts = {}
        if events_p and os.path.exists(events_p):
            for ev in AN.read_jsonl(events_p):
                if ev.get("event") == A36.START_EVENT:
                    self.starts.setdefault(ev.get("key"), []).append(ev)

    def recovered(self, unit):
        k = f"{self.ns}|exec|{unit}|run0"
        e = self.last.get(k)
        if e is None or e.get("status") != "done" or k not in self.outc:
            return None
        return e, self.outc[k], self.hist[k], self.starts.get(k, [])


def patched(r, src_hit, gold, pass_id):
    """Row r with only the execution-dependent fields replaced, through the registered scorer. Plan-side fields stay as they are."""
    e, o, h, st = src_hit
    pp = S.C.parse_pres(r["pres"])
    new = S.score_cell(r["cell"], 0, plan_row=None, exec_row=e, outcome=o, gold_item=gold[pp["item"]], family=r.get("family"), ns=r.get("namespace"),
                       invalid_episode=r.get("invalid_episode", False), exec_history=h, exec_starts=st, block="main", exec_epoch=0)
    out = dict(r)
    for f in EXEC_FIELDS:
        out[f] = new.get(f)
    out["empty_plan_success"] = bool(out.get("executed") and r.get("plan_len") == 0 and out.get("goal_ok"))
    out.update(rerun=True, rerun_pass=pass_id, orig_sr=r.get("sr"), orig_exec_status=r.get("exec_status"), orig_excluded_reset=r.get("excluded_reset"))
    return out


RESCUE_SUMMARY = os.path.join(INPUTS, "analysis", "N_rescue_summary.json")


def rescue_complete():
    """True when the rescue summary shows every pass finished in every kitchen: no unit left untried, every pass finished all its targets,
    and the passes stopped at the maximum or after a pass that recovered nothing."""
    if not os.path.exists(RESCUE_SUMMARY):
        return False
    import rescue as NC
    kit = json.load(open(RESCUE_SUMMARY, encoding="utf-8"))["kitchens"]
    for v in kit.values():
        ps = [v["passes"][q] for q in sorted(v.get("passes", {}), key=int)]
        if v.get("not_yet_tried") or not ps or any(x["finished"] != x["targets"] for x in ps):
            return False
        if len(ps) < NC.MAX_RESCUE_PASSES and ps[-1]["recovered"]:
            return False
    return True


def rescue_sources():
    """Per kitchen, the rescue reader: per exec key, the history of the latest final pass."""
    import rescue as NC
    return {ns: NC.RescueSource(ns) for ns in NS.values()}, NC


def apply_rescue(rows, rsn, gold, tally=None):
    """A row still flagged, or crashed, whose unit a rescue pass restored gets that execution. A rescue history ending blocked or crashed, and a crash with no rescue row, stays out
    of the denominator (the planner never acted). Every other row is unchanged."""
    rs, NC = rsn
    out = []
    for r in rows:
        crash = r.get("exec_status") == "crash"
        if (A36.flagged(r) or crash) and r.get("namespace") in rs and r.get("unit"):
            src = rs[r["namespace"]]
            ek = f"{r['namespace']}|exec|{r['unit']}|run0"
            h = src.hist.get(ek)
            if h and h[-1].get("status") in NC.GOOD:
                r = patched(r, (h[-1], src.outc.get(ek), h, src.starts.get(ek, [])), gold, "rescue")
                if tally is not None:
                    tally[r["arm"]] = tally.get(r["arm"], 0) + 1
            elif crash:
                r = dict(r, excluded_reset=True, n_left_out_simulator_fault="crash")
        out.append(r)
    return out


def idx_of(rows):
    i = AN.build_index(rows)
    return i, AN.invalid_presentations(i["S"])


def gate_patch_path(RAW, gold):
    """The patch path, on R's own records: each sealed flagged row whose re-run ran must become R's row (execution and plan fields alike)."""
    n, wrong = 0, []
    for K, ns in NS.items():
        src = Source(ns, os.path.join(S_, "R", ns, "exec.jsonl"), os.path.join(S_, "R", ns, "outcome.jsonl"), os.path.join(INPUTS, "starts", f"R.{ns}.jsonl"))
        want = {r["key"]: r for r in RAW[K][1]}
        for r in RAW[K][0]:
            if not (A36.flagged(r) and int(r.get("run", 0)) == 0):
                continue
            hit = src.recovered(r.get("unit"))
            if hit is None:
                continue
            got, w = patched(r, hit, gold, "x"), want[r["key"]]
            d = [f for f in set(got) | set(w) if f not in COMPARE_SKIP and got.get(f) != w.get(f)]
            n += 1
            if d:
                wrong.append((r["key"], d[:6]))
    return {"rows_checked": n, "mismatches": len(wrong), "first": wrong[:3]}


def window_keys(ns):
    """The exec keys (ns|exec|unit|runN) of the K2 rendering-fault window record (as k23_rerun.window_keys, on the extract)."""
    import k23_rerun as KR
    with open(os.path.join(INPUTS, "render_window_K2.json"), encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("namespace") != ns:
        raise ValueError(f"window record is for {rec.get('namespace')!r}, not {ns}")
    keys = set()
    for unit, rows in (rec.get("units") or {}).items():
        for r in rows:
            if not str(r.get("key", "")).startswith(f"{ns}|exec|{unit}|run"):
                raise ValueError(f"window record row {r.get('key')!r} does not belong to unit {unit}")
            keys.add(r["key"])
    n_units = len({k.split("|")[2] for k in keys})
    if n_units != KR.WINDOW_UNITS_N[ns] or n_units != rec.get("affected_units"):
        raise ValueError(f"window record names {n_units} units; expected {KR.WINDOW_UNITS_N[ns]} (affected_units {rec.get('affected_units')})")
    return keys


def rendering_window(RAW_final, gold):
    """K2: the render-window units. What remains of them after the restoration re-runs, and what it would change."""
    ns = NS["K2"]
    win = window_keys(ns)
    d = os.path.join(ROOT, "docs", "final-test", "k23", "rerun", ns)
    src = Source(ns, os.path.join(d, "exec.rerun.jsonl"), os.path.join(d, "outcome.rerun.jsonl"), os.path.join(INPUTS, "starts", f"k23-rerun.{ns}.jsonl"))
    rows = RAW_final["K2"]

    def core_ran(r):
        return (f"{ns}|exec|{r.get('unit')}|run0" in win and r.get("arm") in ARMS["K2"] and r.get("r") == 0 and r.get("run") == 0 and r.get("world") == "canon")
    w = [r for r in rows if core_ran(r)]
    ran = [r for r in w if r.get("exec_status") == "done" and not r.get("rerun")]
    out = {"window_units": len(win), "core_rows_in_window": len(w), "rows_rerun_for_restoration": sum(1 for r in w if r.get("rerun")),
           "rows_left_out_restoration": sum(1 for r in w if A36.flagged(r)), "rows_ran_in_window_not_rerun": len(ran),
           "units_ran_in_window_not_rerun": len({r["unit"] for r in ran}), "ran_by_arm": {a: sum(1 for r in ran if r["arm"] == a) for a in ARMS["K2"]}}
    repl, gone, changes = [], [], {}
    for r in rows:
        if core_ran(r) and r.get("exec_status") == "done" and not r.get("rerun"):
            gone.append(dict(r, excluded_reset=True))
            hit = src.recovered(r.get("unit"))
            if hit is None:
                repl.append(r)
                continue
            n_ = patched(r, hit, gold, "render")
            if n_["sr"] != r["sr"]:
                c = changes.setdefault(r["arm"], [0, 0])
                c[0 if n_["sr"] > r["sr"] else 1] += 1
            repl.append(n_)
        else:
            repl.append(r)
            gone.append(r)
    out["sr_changes_if_replaced"] = {a: {"up": v[0], "down": v[1]} for a, v in changes.items()}
    for name, rs in (("as_final", rows), ("window_rows_left_out", gone), ("window_rows_replaced_by_k23_rerun", repl)):
        i, dd = idx_of(rs)
        g = PLD.paired(i, dd, "G+M", "G0", "all", True)
        t = CA.tables(CA.without_flagged(i), dd, ARMS["K2"])
        out[name] = {"rates_pp": {a: p1(t[a]["overall"]["sr"]) for a in PLANNERS + ["P0"]}, "G+M - G0": [g["estimate_pp"], g["ci95_pp"], g["n_presentations"]],
                     "n": {a: t[a]["n_presentations"] for a in PLANNERS + ["P0"]}}
    return out


def outcome_lookup():
    """Exec key -> outcome row, rescue last, as the score rows adopt them (sealed, addendum, R, rescue)."""
    out = {}
    for ns in NS.values():
        for p in (os.path.join(ROOT, "docs", "final-test", "matrix", ns, "outcome.sealed.jsonl"), os.path.join(ROOT, "docs", "final-test", "k23", "codex", ns, "outcome.codex.jsonl"),
                  os.path.join(S_, "R", ns, "outcome.jsonl"), os.path.join(S_, "N", ns, "rescue", "outcome.jsonl")):
            if os.path.exists(p):
                for r in AN.read_jsonl(p):
                    out[r.get("key")] = r
    return out


def failure_breakdown(RAW_final):
    """Of the retained failures of each with-memory planner: a failed skill step / every step ran but no goal / a refused plan, with the causes of the first failed step."""
    oc = outcome_lookup()
    out = {}
    fields = ("retained", "failures", "failed_skill_step", "all_steps_no_goal", "refused_plan")
    for arm, name in (("G+M", "Gemma 4 with Scene Memory"), ("O+M", "Opus 5.5 with Scene Memory"), ("X+M", "GPT6-Sol with Scene Memory")):
        tot = {f: 0 for f in fields}
        causes, per = {}, {}
        for K in NS:
            i, _ = idx_of(RAW_final[K])
            t = {f: 0 for f in fields}
            for k, r in i["S"].items():
                if k[0] != arm or A36.flagged(r):
                    continue
                t["retained"] += 1
                if r.get("sr"):
                    continue
                t["failures"] += 1
                if r.get("plan_status") != "ok":
                    t["refused_plan"] += 1
                elif r.get("steps_ok_n") == r.get("steps_n"):
                    t["all_steps_no_goal"] += 1
                else:
                    t["failed_skill_step"] += 1
                    steps = (oc.get(f"{r['namespace']}|exec|{r['unit']}|run0") or {}).get("steps") or []
                    first = next((s for s in steps if not s.get("ok")), {})
                    c = first.get("cause") or first.get("failure_cause") or "unknown"
                    causes[c] = causes.get(c, 0) + 1
            per[K] = t
            for f in fields:
                tot[f] += t[f]
        pc_ = lambda f: round(100 * tot[f] / tot["failures"], 1)   # noqa: E731,B023
        out[arm] = {"name": name, "pooled": tot, "pct_of_failures": {f: pc_(f) for f in fields[2:]}, "per_kitchen": per,
                    "first_failed_step_causes": dict(sorted(causes.items(), key=lambda x: -x[1]))}
    return out


def larger_planners(RAW_final, rs, gold):
    """E4B and Qwen under this analysis, from N's rows (N/<ns>/score_<tag>.noexcl.jsonl) + the rescue pass; E2B from the main rows."""
    pts, out = {}, {"kitchens": {}, "three_kitchen": {}}
    arms = ["G0", "G+M", "E4B0", "E4B+M", "Q9B0", "Q9B+M"]
    pairs = (("E4B0", "G+M"), ("Q9B0", "G+M"), ("E4B+M", "G+M"), ("Q9B+M", "G+M"), ("E4B+M", "E4B0"), ("Q9B+M", "Q9B0"), ("G+M", "G0"), ("E4B0", "G0"))
    for K, ns in NS.items():
        rows = [r for r in RAW_final[K] if r.get("arm") in ("G0", "G+M")]
        for tag in ("e4b", "q9b"):
            rows += apply_rescue(AN.read_jsonl(os.path.join(S_, "N", ns, f"score_{tag}.noexcl.jsonl")), rs, gold)
        i, d = idx_of(rows)
        T = CA.tables(CA.without_flagged(i), d, arms)
        kb = {"rates": {a: {"overall": p1(T[a]["overall"]["sr"]), "sr": T[a]["overall"]["sr"], "ci95_pp": [p1(x) for x in T[a]["overall"]["ci95"]], "n": T[a]["n_presentations"],
                            "level": {L: T[a]["level"][L]["sr"] for L in ("L1", "L2", "L3")}} for a in arms}, "paired": {}}
        kb["left_out"] = {a: sum(1 for k, r in i["S"].items() if k[0] == a and A36.flagged(r)) for a in arms}
        for a, b in pairs:
            kb["paired"][f"{a} - {b}"] = PLD.paired(i, d, a, b, "all", True)
            pts.setdefault(f"{a} - {b}", []).append(parts_safe(i, d, a, b, "all", True))
        out["kitchens"][K] = kb
    for n, p in pts.items():
        out["three_kitchen"][n] = CN.stratified_mean(p)
    out["level_means"] = {a: {L: sum(out["kitchens"][K]["rates"][a]["level"][L] for K in NS) / 3 for L in ("L1", "L2", "L3")} for a in arms}
    out["overall_means"] = {a: sum(out["kitchens"][K]["rates"][a]["sr"] for K in NS) / 3 for a in arms}
    nj = json.load(open(os.path.join(INPUTS, "analysis", "N.json"), encoding="utf-8"))
    out["N_status"] = {"written": nj.get("written"), "partial_missing": {k: v.get("partial_missing") for k, v in nj["kitchens"].items()}}
    nd = ((nj.get("three_kitchen") or {}).get("rescued") or {}).get("differences") or {}
    diffs = []
    for n, m in out["three_kitchen"].items():
        v = (nd.get(n) or {}).get("all")
        if v is None:
            continue
        if v.get("estimate_pp") != m["estimate_pp"] or v.get("ci95_pp") != m["ci95_pp"]:
            diffs.append(f"{n}: here {m['estimate_pp']} {m['ci95_pp']}, N {v.get('estimate_pp')} {v.get('ci95_pp')}")
    out["N_crosscheck"] = "equal on every contrast N reports" if nd and not diffs else ("; ".join(diffs) if diffs else "N has no rescued block")
    return out


def n1_crosscheck(R):
    """This script's larger-planner values against N's own analysis ('rescued' block); nothing is edited to agree."""
    nj = json.load(open(os.path.join(INPUTS, "analysis", "N.json"), encoding="utf-8"))
    ver = "rescued"
    out, diff = [], []
    for K, ns in NS.items():
        v = nj["kitchens"][ns]["versions"][ver]
        for a in ("G0", "G+M", "E4B0", "E4B+M", "Q9B0", "Q9B+M"):
            mine = R["larger_planners"]["kitchens"][K]["rates"][a]["overall"]
            there = (v["rates_pct"].get(a) or {}).get("all")
            out.append((f"{K} rate {a}", mine, there))
        for name, m in R["larger_planners"]["kitchens"][K]["paired"].items():
            there = ((v["differences"].get(name) or {}).get("all") or {})
            out.append((f"{K} {name}", [m["estimate_pp"], m["ci95_pp"], m["n_presentations"]], [there.get("estimate_pp"), there.get("ci95_pp"), there.get("n_presentations")]))
    t = nj["three_kitchen"][ver]["differences"]
    for name, m in R["larger_planners"]["three_kitchen"].items():
        there = (t.get(name) or {}).get("all") or {}
        out.append((f"mean {name}", [m["estimate_pp"], m["ci95_pp"]], [there.get("estimate_pp"), there.get("ci95_pp")]))
    r3 = nj["three_kitchen"][ver]["rates_pct"]
    for a, x in R["larger_planners"]["overall_means"].items():
        out.append((f"mean rate {a}", p1(x), (r3.get(a) or {}).get("all")))
        for L in ("L1", "L2", "L3"):
            out.append((f"mean rate {a} {L}", p1(R["larger_planners"]["level_means"][a][L]), (r3.get(a) or {}).get(L)))
    diff = [x for x in out if x[1] != x[2]]
    return {"n": len(out), "differences": diff, "written": nj.get("written")}


def main():
    res = {"schema": "reexecuted.final/1", "definition": __doc__.split("\n\n")[1].replace("\n", " "), "gates": {}, "inputs": {}}
    src = CN.sources()
    pub_adj = json.load(open(os.path.join(INPUTS, "adjusted_reference.json"), encoding="utf-8"))
    Rj = json.load(open(os.path.join(INPUTS, "analysis", "R.json"), encoding="utf-8"))
    A6 = json.load(open(os.path.join(INPUTS, "analysis", "A6.json"), encoding="utf-8"))
    Vj = json.load(open(os.path.join(INPUTS, "analysis", "V.json"), encoding="utf-8"))
    Gj = json.load(open(os.path.join(INPUTS, "analysis", "G.json"), encoding="utf-8"))
    k1_pub = json.load(open(CA.K1_ANALYSIS, encoding="utf-8"))
    for K, (orig, new) in src.items():
        for p in orig + new:
            h = CN.sha(p)
            res["inputs"][CN.rel(p)] = h
            if Rj["inputs"].get(CN.rel(p)) != h:
                bad.append(f"sha of {CN.rel(p)} differs from R/analysis.json")
    I_old, I_new, RAW = {}, {}, {}
    for K, (orig, new) in src.items():
        I_old[K], I_new[K] = CA.index(*orig), CA.index(*new)
        RAW[K] = (sum((AN.read_jsonl(p) for p in orig), []), sum((AN.read_jsonl(p) for p in new), []))
    # integrity of the flag: in the re-run rows a row is flagged exactly when its re-run failed restoration again
    fl_chk = {}
    for K in NS:
        S_k = {k: r for k, r in I_new[K][0]["S"].items() if k[0] in ARMS[K]}
        fl = [k for k, r in S_k.items() if A36.flagged(r)]
        wrong = [k for k in fl if not S_k[k].get("rerun")] + [k for k, r in S_k.items() if r.get("rerun") and not A36.flagged(r) and not r.get("executed")]
        fl_chk[K] = {"flagged": len(fl), "flagged_without_rerun_or_rerun_not_executed_unflagged": len(wrong)}
        if wrong:
            bad.append(f"{K}: flag integrity {wrong[:3]}")
    res["gates"]["flag_integrity_main_rows"] = fl_chk

    # ---------------- the main bundles
    print("bundle sealed / drop", flush=True)
    b_adj = bundle(I_old, True)
    print("bundle re-run / no drop", flush=True)
    b_nox = bundle(I_new, False)
    # ---- the rescue pass: the LAST re-run source, in every task
    gold, _ = S._gold_load(None, None, verify=True)
    pp_gate = gate_patch_path(RAW, gold)
    res["gates"]["G4_patch_path_on_R_records"] = pp_gate
    if pp_gate["mismatches"] or not pp_gate["rows_checked"]:
        bad.append(f"G4 patch path: {pp_gate}")
    rs = rescue_sources()
    sp = RESCUE_SUMMARY
    res["rescue"] = {"summary_present": os.path.exists(sp),
                     "summary": ({ns: {f: v.get(f) for f in ("unique_units_tried", "by_world", "recovered_units", "not_yet_tried", "still_blocked_units")} | {"passes": {q: {f: x.get(f) for f in ("targets", "finished", "recovered", "still_blocked", "statuses")} for q, x in v.get("passes", {}).items()}}
                                  for ns, v in json.load(open(sp, encoding="utf-8"))["kitchens"].items()} if os.path.exists(sp) else None),
                     "complete": rescue_complete(), "units_with_final_history": {ns: len(v.hist) for ns, v in rs[0].items()}, "recovered_cells_by_task": {}}
    RAW_final, tally = {}, {}
    for K in NS:
        tally[K] = {}
        RAW_final[K] = apply_rescue(RAW[K][1], rs, gold, tally[K])
    res["rescue"]["recovered_cells_by_task"]["main"] = tally
    I_fin = {K: idx_of(RAW_final[K]) for K in NS}
    print("bundle final: rows after rescue / drop", flush=True)
    b_new = bundle(I_fin, True)
    b_res_T = {K: CA.tables(CA.without_flagged(I_new[K][0]), I_new[K][1], ARMS[K]) for K in NS}

    # ---------------- G1 adjusted
    a36 = k1_pub["a36"]
    pub_c1 = {f"{hid} {v['a']} - {v['b']}": v for hid, v in list(a36["holm_family"].items()) + list(a36["exploratory_contrasts"].items())}
    pub_rates = {K: Rj["kitchens"][K]["rates"]["adjusted"] for K in NS}
    pub_c23 = {K: Rj["kitchens"][K]["contrasts"]["adjusted"] for K in ("K2", "K3")}
    gate_main(b_adj, pub_rates, pub_c1, pub_c23, Rj["heldout_mean"]["adjusted"], {K: Rj["kitchens"][K]["level_diffs"]["adjusted"] for K in NS}, "G1")
    for K in NS:     # exact floats vs the stored adjusted analysis
        for a in ARMS[K]:
            m, q = b_adj["T"][K][a]["overall"], pub_adj["adjusted_results"][K]["adjusted"][a]["overall"]
            if not (same(m["sr"], q["sr"]) and same(m["ci95"], q["ci95"])):
                bad.append(f"G1 {K} {a} vs adjusted_reference.json")
    tk = b_adj["TK"]["G+M - G0"]["all"]
    chk("G1 three-kitchen mean", tk["estimate_pp"], 13.0)
    chk("G1 three-kitchen interval", tk["ci95_pp"], [8.5, 17.5])
    chk("G1 per kitchen", tk["per_kitchen_pp"], [16.2, 14.0, 8.9])
    g1 = b_adj["LD"]["K1"]["G+M - G0"]["all"]
    chk("G1 K1 G+M - G0", [g1["estimate_pp"], g1["ci95_pp"], g1["n_presentations"], g1["n_items"], g1["n_families"]], [16.2, [8.5, 24.1], 223, 120, 42])
    chk("G1 held-out mean", pc1(b_adj["HM"]["G+M - G0"]["estimate"]), 11.4)
    chk("G1 Holm H2a", round(b_adj["K1"]["H2a G+M - G0"]["p_holm"], 4), 0.0006)
    old4 = pub_adj["paired_level_diffs"]["kitchens"]
    for K in NS:
        for L in ("L1", "L2", "L3", "all"):
            for x in FRONTIER:
                m, q = b_adj["LD"][K][f"G+M - {x}"][L], old4[K]["differences"][L][x]
                if not (same(m["estimate"], q["estimate"]) and same(m["ci95"], q["ci95"])):
                    bad.append(f"G1 paired level differences {K} {L} {x}")
    LM_old = CN.level_means(b_adj["T"], PLANNERS + ["P0"])
    chk("G1 Atomic / Sequential G+M", [LM_old["G+M"]["L1"]["pp"], LM_old["G+M"]["L2"]["pp"]], [88.3, 89.1])
    chk("G1 frontier Atomic", [min(LM_old[a]["L1"]["pp"] for a in FRONTIER), max(LM_old[a]["L1"]["pp"] for a in FRONTIER)], [93.6, 95.3])
    chk("G1 frontier Sequential", [min(LM_old[a]["L2"]["pp"] for a in FRONTIER), max(LM_old[a]["L2"]["pp"] for a in FRONTIER)], [83.9, 94.1])
    g_old = [PLD.paired(CN.without_items(I_old[K][0], CN.EXACT | CN.CONTAINS), I_old[K][1], "G+M", "G0", "all", True) for K in NS]
    chk("G1 overlap", [g_old[0]["estimate_pp"], round(sum(x["estimate_pp"] for x in g_old) / 3, 1)], [12.7, 9.9])
    chk("G1 25 / 56", CN.steps_complete_goal_failed(I_old["K1"][0], "G+M")[0], 25)
    chk("G1 4 / 56", CN.steps_complete_goal_failed(I_old["K1"][0], "O+M")[0], 4)
    chk("G1 controls n", [b_adj["T"]["K1"][a]["n_presentations"] for a in ("G+FS", "G+M/p", "O+M/p")], [235, 234, 233])
    h4 = PLD.paired(I_old["K1"][0], I_old["K1"][1], "G+M", "G+FS", "all", True)
    chk("G1 H4a", [h4["estimate_pp"], h4["ci95_pp"], h4["n_presentations"]], [14.3, [6.0, 22.7], 225])
    # ---------------- G2 no-exclusion
    gate_main(b_nox, {K: Rj["kitchens"][K]["rates"]["noexcl"] for K in NS}, Rj["kitchens"]["K1"]["contrasts"]["noexcl"], {K: Rj["kitchens"][K]["contrasts"]["noexcl"] for K in ("K2", "K3")},
              Rj["heldout_mean"]["noexcl"], {K: Rj["kitchens"][K]["level_diffs"]["noexcl"] for K in NS}, "G2")
    tk = b_nox["TK"]["G+M - G0"]["all"]
    chk("G2 three-kitchen mean", tk["estimate_pp"], 13.6)
    chk("G2 three-kitchen interval", tk["ci95_pp"], [9.1, 18.1])
    chk("G2 per kitchen", tk["per_kitchen_pp"], [16.4, 14.0, 10.4])
    g2 = b_nox["LD"]["K1"]["G+M - G0"]["all"]
    chk("G2 K1 G+M - G0", [g2["estimate_pp"], g2["ci95_pp"], g2["n_presentations"]], [16.4, [8.5, 24.4], 250])
    # ---------------- G3 this analysis vs R's residual-adjusted rates
    for K in NS:
        pub = Rj["kitchens"][K]["rates"]["residual_adjusted"]
        for a in ARMS[K]:
            if CN.rate_pp2(b_res_T[K][a]["overall"]) != {k: pub[a]["overall"][k] for k in ("sr", "ci95", "n_items")} or b_res_T[K][a]["n_presentations"] != pub[a]["n_presentations"]:
                bad.append(f"G3 {K} {a} vs R residual_adjusted")

    # ---------------- A6 (memory components) under the three settings
    print("A6", flush=True)
    a6rows = AN.read_jsonl(os.path.join(INPUTS, "rows", "A6.no-exclusion.jsonl"))
    a6_has_adjusted = any(r.get("a_version") == "adjusted" for r in a6rows)

    def a6_index(K, version):
        base = [r for r in (RAW[K][1] if version == "no-exclusion" else RAW[K][0]) if r.get("arm") in A6_MAIN]
        idx = AN.build_index(base + [r for r in a6rows if r["namespace"] == NS[K] and r["a_version"] == version])
        return idx, AN.invalid_presentations(idx["S"])
    AI_old = {K: a6_index(K, "adjusted") for K in NS} if a6_has_adjusted else None
    AI_new = {K: a6_index(K, "no-exclusion") for K in NS}
    a6_tally = {}
    AI_fin = {}
    for K in NS:
        base = [r for r in RAW_final[K] if r.get("arm") in A6_MAIN]
        AI_fin[K] = idx_of(base + apply_rescue([r for r in a6rows if r["namespace"] == NS[K] and r["a_version"] == "no-exclusion"], rs, gold, a6_tally))
    res["rescue"]["recovered_cells_by_task"]["A6"] = a6_tally
    arms_of = {K: list(A6["kitchens"][NS[K]]["versions"]["adjusted"]["arms"]) for K in NS}
    full = lambda lab: "G+M" if lab.startswith("G~") else "O+M"   # noqa: E731
    a6_rep = {}
    for K, ns in NS.items():
        kb = {}
        for lab in arms_of[K]:
            for (vers, ix, drop) in (("adjusted", AI_old[K] if AI_old else None, True), ("no-exclusion", AI_new[K], False)):
                if ix is None:
                    continue
                r = PLD.paired(ix[0], ix[1], lab, full(lab), "all", drop)
                v = A6["kitchens"][ns]["versions"][vers]["arms"][lab]["vs_full"]["all"]
                if not (same(r["estimate"], v["estimate"]) and same(r["ci95"], v["ci95"]) and r["n_presentations"] == v["n_presentations"]):
                    bad.append(f"A6 {vers} {K} {lab}")
            kb[lab] = {L: PLD.paired(AI_fin[K][0], AI_fin[K][1], lab, full(lab), L, True) for L in LEVELS}
            sel = [r for k, r in AI_fin[K][0]["S"].items() if k[0] == lab]
            kb[lab]["presentations"] = len(sel)
            kb[lab]["retained_marginal"] = sum(1 for r in sel if not A36.flagged(r))
        a6_rep[K] = kb
    # flag integrity of A6 rows: a row whose re-run failed again must carry the flag
    a6_bad = [r["key"] for r in a6rows if r["a_version"] == "no-exclusion" and r.get("a_replaced") and r.get("exec_status") == "blocked_state" and not A36.flagged(r)]
    a6_flag2 = [r["key"] for r in a6rows if r["a_version"] == "no-exclusion" and A36.flagged(r) and not r.get("a_replaced")]
    res["gates"]["A6_flag_integrity"] = {"blocked_rerun_without_flag": len(a6_bad), "flagged_without_replacement": len(a6_flag2)}
    if a6_bad:
        bad.append(f"A6 blocked re-run rows without the flag: {a6_bad[:3]}")
    TKMAP = {"no_request - G+M": "no_request", "no_procedures - G+M": "no_procedures", "no_req_proc - G+M": "no_req_proc", "only_req_proc - G+M": "only_req_proc",
             "irrelevant - G+M": "irrelevant", "rep - G+M": "rep", "other_kitchen - G+M": None,
             "O no_request - O+M": "O:no_request", "O no_procedures - O+M": "O:no_procedures", "O no_req_proc - O+M": "O:no_req_proc", "O only_req_proc - O+M": "O:only_req_proc"}

    def tk_arm(name, ix, drop, level="all"):
        vers_tk = A6["three_kitchen"]["adjusted"]
        pts = []
        for K, ns in NS.items():
            if name == "other_kitchen - G+M":
                lab = vers_tk[name]["arms"][ns]
            else:
                m = TKMAP[name]
                lab = ("O~" + m[2:]) if m.startswith("O:") else ("G~" + m)
            pts.append(parts_safe(ix[K][0], ix[K][1], lab, full(lab), level, drop))
        return CN.stratified_mean(pts)
    tk_new, tk_lv = {}, {}
    for name in TKMAP:
        checks = [(tk_arm(name, AI_new, False), A6["three_kitchen"]["no-exclusion"][name], "no-exclusion")]
        if AI_old:
            checks.insert(0, (tk_arm(name, AI_old, True), A6["three_kitchen"]["adjusted"][name], "adjusted"))
        for (m, v, lab) in checks:
            if m["ci95_pp"] != v["ci95_pp"] or m["estimate_pp"] != v["estimate_pp"] or m["per_kitchen_pp"] != v["per_kitchen_pp"]:
                bad.append(f"A6 three-kitchen {lab} {name}: {m} vs {v}")
        tk_new[name] = tk_arm(name, AI_fin, True)
        tk_lv[name] = {L: tk_arm(name, AI_fin, True, L) for L in ("L1", "L2", "L3")}
    tk_new["G+M - G0"] = b_new["TK"]["G+M - G0"]["all"]
    tk_new["O+M - O0"] = b_new["TK"]["O+M - O0"]["all"]
    # by language: the stored A6 adjusted values (needs the adjusted-version rows and A/by_language_k1.json)
    bl_path = os.path.join(S_, "A", "by_language_k1.json")
    if AI_old and os.path.exists(bl_path):
        bl_pub = json.load(open(bl_path, encoding="utf-8"))
        for key, v in bl_pub.items():
            lang, c_ = key.split(" ", 1)
            a, b = c_.split(" - ")
            r = PLD.paired(CN.lang_filter(AI_old["K1"][0], lang), AI_old["K1"][1], a, b, "all", True)
            if not (same(r["estimate"], v["estimate"]) and same(r["ci95"], v["ci95"])):
                bad.append(f"by_language {key}")
    lang_out = {}
    for lang in ("en", "fa"):
        for a, b in (("G+M", "G0"), ("G~no_request", "G+M"), ("G~no_procedures", "G+M"), ("G~no_req_proc", "G+M"), ("O~no_request", "O+M")):
            lang_out[f"{lang} {a} - {b}"] = PLD.paired(CN.lang_filter(AI_fin["K1"][0], lang), AI_fin["K1"][1], a, b, "all", True)
    res["ablation"] = {"three_kitchen": tk_new, "by_level_three_kitchen": tk_lv, "per_kitchen_arms": a6_rep, "by_language_k1": lang_out, "noise_floor": A6["noise_floor"]}

    # ---------------- V (plan checker) and G (goal library) under the three settings
    print("V", flush=True)
    base_old = {K: RAW[K][0] for K in NS}
    base_new = {K: RAW[K][1] for K in NS}
    pick_arms = lambda rows, arms: [r for r in rows if r.get("arm") in arms]   # noqa: E731
    vrows = AN.read_jsonl(os.path.join(INPUTS, "rows", "V.no-exclusion.jsonl"))
    V_C = (("G+M+v3", "G+M"), ("G0+v3", "G0"), ("G+M+v3", "G0"), ("G+M", "G0"))
    vb_o = {K: pick_arms(base_old[K], ("G+M", "G0")) for K in NS}
    vb_n = {K: pick_arms(base_new[K], ("G+M", "G0")) for K in NS}
    if any(r.get("v_version") == "adjusted" for r in vrows):
        gate_v_like(v_like(vrows, vb_o, "adjusted", V_C, True), Vj, "adjusted", "V")
    gate_v_like(v_like(vrows, vb_n, "no-exclusion", V_C, False), Vj, "no-exclusion", "V")
    vb_f = {K: pick_arms(RAW_final[K], ("G+M", "G0")) for K in NS}
    v_tally = {}
    res["plan_checker"] = v_like(apply_rescue(vrows, rs, gold, v_tally), vb_f, "no-exclusion", V_C, True)
    res["rescue"]["recovered_cells_by_task"]["V"] = v_tally
    v_bad = [r["key"] for r in vrows if r.get("v_version") == "no-exclusion" and r.get("v_replaced") and r.get("exec_status") == "blocked_state" and not A36.flagged(r)]
    res["gates"]["V_flag_integrity"] = {"blocked_rerun_without_flag": len(v_bad)}
    if v_bad:
        bad.append(f"V blocked re-run without flag {v_bad[:3]}")
    print("G", flush=True)
    grows = AN.read_jsonl(os.path.join(INPUTS, "rows", "G.no-exclusion.jsonl"))
    g_arms = sorted({r["arm"] for r in grows})
    G_C = tuple((a, "G+M") for a in g_arms) + (("G+M", "G0"),)
    if any(r.get("v_version") == "adjusted" for r in grows):
        gate_v_like(v_like(grows, vb_o, "adjusted", G_C, True), Gj, "adjusted", "G")
    gate_v_like(v_like(grows, vb_n, "no-exclusion", G_C, False), Gj, "no-exclusion", "G")
    g_tally = {}
    res["goal_library"] = v_like(apply_rescue(grows, rs, gold, g_tally), vb_f, "no-exclusion", G_C, True)
    res["rescue"]["recovered_cells_by_task"]["G"] = g_tally
    g_bad = [r["key"] for r in grows if r.get("v_version") == "no-exclusion" and r.get("v_replaced") and r.get("exec_status") == "blocked_state" and not A36.flagged(r)]
    res["gates"]["G_flag_integrity"] = {"blocked_rerun_without_flag": len(g_bad)}
    if g_bad:
        bad.append(f"G blocked re-run without flag {g_bad[:3]}")

    if bad:
        raise SystemExit("GATE FAILED, nothing written:\n  " + "\n  ".join(bad[:40]))
    res["gates"]["G1_adjusted_reproduced"] = True
    res["gates"]["G2_no_exclusion_reproduced"] = True
    res["gates"]["G3_R_residual_adjusted_rates_equal"] = True
    res["gates"]["literal_checks"] = GATES_LOG

    # ---------------- new values
    T, LD, C23, HM = b_new["T"], b_new["LD"], b_new["C23"], b_new["HM"]
    res["rates"] = {K: CN.tables_out(T[K]) for K in NS}
    res["three_kitchen_means"] = CN.level_means(T, PLANNERS + ["P0"])
    res["three_kitchen_gain"] = b_new["TK"]
    res["k1_contrasts_holm_path"] = b_new["K1"]
    res["level_diffs"] = LD
    res["k23_contrasts_cr2"] = C23
    res["heldout_mean_cr2"] = HM
    # counts of presentations retained and left out
    counts = {}
    for K in NS:
        counts[K] = {}
        for a in ARMS[K]:
            sel = {k: r for k, r in I_fin[K][0]["S"].items() if k[0] == a}
            sealed = {k: r for k, r in I_old[K][0]["S"].items() if k[0] == a}
            kept = [k for k, r in sel.items() if not A36.flagged(r)]
            lv = {}
            for L in ("L1", "L2", "L3"):
                keys = [k for k in sel if I_fin[K][0]["level"].get(k[1]) == L]
                lv[L] = {"presentations": len(keys), "retained": sum(1 for k in keys if not A36.flagged(sel[k]))}
            counts[K][a] = {"presentations": len(sel), "retained": len(kept), "left_out": len(sel) - len(kept), "sealed_flagged": sum(1 for r in sealed.values() if A36.flagged(r)),
                            "restored_by_rerun": sum(1 for r in sel.values() if r.get("rerun") and not A36.flagged(r)),
                            "recovered_by_rescue": sum(1 for r in sel.values() if r.get("rerun_pass") == "rescue"), "by_level": lv}
    res["counts"] = counts
    planner_arms = PLANNERS + ["P0"]
    res["count_ranges"] = {K: {"retained": [min(counts[K][a]["retained"] for a in planner_arms), max(counts[K][a]["retained"] for a in planner_arms)],
                               "left_out": [min(counts[K][a]["left_out"] for a in planner_arms), max(counts[K][a]["left_out"] for a in planner_arms)],
                               "presentations": counts[K]["G0"]["presentations"], "reference_retained": counts[K]["P0"]["retained"]} for K in NS}
    # memory effect and the G+M - O0 gap per kitchen and held-out mean: [estimate, lower, upper] in pp
    tri_out = {}
    for key, name in (("G+M - G0", "Gemma 4"), ("O+M - O0", "Opus 5.5"), ("X+M - X0", "GPT6-Sol"), ("G+M - O0", "gap G+M - O0")):
        r = LD["K1"][key]["all"]
        tri = lambda e, iv: [pc1(e), pc1(iv[0]), pc1(iv[1])]   # noqa: E731
        tri_out[key] = {"model": name, "K1_95": tri(r["estimate"], r["ci95"]), "K2_965": tri(C23["K2"][key]["estimate"], C23["K2"][key]["interval"]),
                        "K3_965": tri(C23["K3"][key]["estimate"], C23["K3"][key]["interval"]), "heldout_mean_965": tri(HM[key]["estimate"], HM[key]["interval"])}
    res["memory_effect_triples"] = tri_out
    i1, d1 = I_fin["K1"]
    res["controls"] = {"G+FS": CN.rate_out(T["K1"]["G+FS"]["overall"]), "G+M/p": CN.rate_out(T["K1"]["G+M/p"]["overall"]), "O+M/p": CN.rate_out(T["K1"]["O+M/p"]["overall"]),
                       "n": {a: T["K1"][a]["n_presentations"] for a in ("G+FS", "G+M/p", "O+M/p")},
                       "G+M - G+FS": PLD.paired(i1, d1, "G+M", "G+FS", "all", True), "G+M/p - G+M": PLD.paired(i1, d1, "G+M/p", "G+M", "all", True),
                       "O+M/p - O+M": PLD.paired(i1, d1, "O+M/p", "O+M", "all", True)}
    cells = [LD[K][f"G+M - {x}"][L] for K in NS for x in FRONTIER for L in ("L1", "L2")]
    res["paired_explicit_ranges"] = {f: [min(c[f] for c in cells), max(c[f] for c in cells)] for f in ("n_presentations", "n_items", "n_families")}
    res["language"] = {"gaps_en_minus_fa": {K: {a: float("%.1f" % (100 * (T[K][a]["lang"]["en"]["sr"] - T[K][a]["lang"]["fa"]["sr"]) + 1e-9)) for a in ("G0", "G+M")} for K in NS},
                       "paired_gain": {K: {lg: PLD.paired(CN.lang_filter(I_fin[K][0], lg), I_fin[K][1], "G+M", "G0", "all", True) for lg in ("en", "fa")} for K in NS},
                       "paired_gain_three_kitchen": {lg: CN.stratified_mean([parts_safe(CN.lang_filter(I_fin[K][0], lg), I_fin[K][1], "G+M", "G0", "all", True) for K in NS]) for lg in ("en", "fa")}}
    sens = {}
    for name, items in (("all items", set()), ("without the 10 exact matches", CN.EXACT), ("without all 18 matching items", CN.EXACT | CN.CONTAINS)):
        row = {}
        for K in NS:
            for a, b in (("G+M", "G0"), ("G+M", "O+M")):
                r = PLD.paired(CN.without_items(I_fin[K][0], items), I_fin[K][1], a, b, "all", True)
                row[f"{K} {a} - {b}"] = {"pp": r["estimate_pp"], "ci95": r["ci95_pp"], "n_items": r["n_items"], "n_families": r["n_families"], "n_presentations": r["n_presentations"],
                                         "items_removed": len({k[1] for k in I_fin[K][0]["S"]} & items)}
        row["three-kitchen mean memory gain (pp)"] = round(sum(row[f"{K} G+M - G0"]["pp"] for K in NS) / 3, 1)
        sens[name] = row
    res["contamination_sensitivity"] = sens
    kept_idx = {K: CA.without_flagged(I_fin[K][0]) for K in NS}
    res["diagnostics"] = {"steps_complete_goal_failed_L3": {a: CN.steps_complete_goal_failed(kept_idx["K1"], a) for a in ("G+M", "O+M")},
                          "steps_complete_goal_failed_L3_all_kitchens": {K: {a: CN.steps_complete_goal_failed(kept_idx[K], a) for a in PLANNERS} for K in NS},
                          "left_out_rate_pct_K1": {a: round(100 * counts["K1"][a]["left_out"] / counts["K1"][a]["presentations"], 1) for a in ARMS["K1"]},
                          "left_out_rate_pct_K2": {a: round(100 * counts["K2"][a]["left_out"] / counts["K2"][a]["presentations"], 1) for a in ARMS["K2"]},
                          "left_out_rate_pct_K3": {a: round(100 * counts["K3"][a]["left_out"] / counts["K3"][a]["presentations"], 1) for a in ARMS["K3"]},
                          "repeat_agreement_final": CN.agreement([r for r in RAW_final["K1"] if not A36.flagged(r) and not r.get("exec_failure")]),
                          "repeat_agreement_no_exclusion": CN.agreement(RAW["K1"][1]), "repeat_agreement_sealed": CN.agreement(RAW["K1"][0])}
    res["rerun_summary"] = Rj["outcome_summary"]
    # what kinds of simulator problem remain in the analysed rows
    leftover = {}
    for K in NS:
        c_ = {}
        for k, r in I_fin[K][0]["S"].items():
            if k[0] in ARMS[K]:
                key = f"{r.get('failure_class')} | flagged={A36.flagged(r)}"
                c_[key] = c_.get(key, 0) + 1
        leftover[K] = c_
    res["simulator_problems_remaining"] = leftover
    res["rendering_window_K2"] = rendering_window(RAW_final, gold)
    res["failure_breakdown"] = failure_breakdown(RAW_final)
    res["larger_planners"] = larger_planners(RAW_final, rs, gold)
    # the larger-planner cross-check against N's own analysis, on the values as written (JSON types)
    R = json.loads(json.dumps(res, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    xc = n1_crosscheck(R)
    R["larger_planners"]["N_crosscheck"] = f"{xc['n']} values compared, {len(xc['differences'])} differ"
    R["larger_planners"]["N_crosscheck_differences"] = xc["differences"]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=1, sort_keys=True)
    print("gates passed; wrote", OUT)


if __name__ == "__main__":
    main()
