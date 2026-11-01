"""A5: score and analyse the Scene Memory ablations (Windows, offline; exploratory, post-hoc).

    python sim/final_test/strengthen/ablate_analyze.py --ns test-v1

Scores every ablation cell of docs/final-test/strengthen/A/plans.jsonl with the matrix's own `score.score_cell` (SR = simulator goal;
rejected / infra plans score 0; the A36 `excluded_reset` flag from the unit's run-0 attempt history; block "main" so the A36 core rule
applies exactly as to G+M), from the execution it maps to (A/<ns>/units.json: a reused main unit -> the sealed main journals, a new
unit -> A/<ns>/exec.jsonl, outcome.jsonl, events.jsonl). Self-check first: the same call on the main journals reproduces the sealed main
score rows of G+M and G0 exactly (timestamps aside).

Analysis on the index of the sealed main score rows (G0, G+M, O0, O+M, G+M/p, O+M/p) plus the ablation rows (arm labels
`G~<arm>`, `O~<arm>`): per arm SR overall and per level (registered = every presentation; adjusted = A36, a flagged presentation leaves
its arm), and paired differences (arm minus full memory, arm minus no memory) with `analyze.contrast` -> `final_test_power.cluster_ci`
(95 % percentile family-cluster bootstrap, B = 20000, seed 20261010; adjusted = paired common subset, the call of
the paired per-level check of the paper). Reproduction gate: the published Kitchen 1 G+M - G0 (+14.4 registered,
+16.2 [+8.5, +24.1] adjusted, n = 223) must come out of the same path.

Writes ONLY docs/final-test/strengthen/A/{score.jsonl, analysis.json, summary_tables.md} (atomic). Reads the sealed journals, never writes them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate_common as A          # noqa: E402  (sets TRACE_ENABLED=0 before anything imports sim/trace.py)

import a36 as A36                  # noqa: E402
import analyze as AN               # noqa: E402
import common as C                 # noqa: E402
import score as S                  # noqa: E402

PLANS = os.path.join(A.DOCS, "plans.jsonl")
SCORE = os.path.join(A.DOCS, "score.jsonl")
ANALYSIS = os.path.join(A.DOCS, "analysis.json")
SUMMARY = os.path.join(A.DOCS, "summary_tables.md")       # the hand-written reading goes to summary.md
MAIN_ARMS = ("G0", "G+M", "O0", "O+M", "G+M/p", "O+M/p")
FULL = {"G": "G+M", "O": "O+M"}
NONE = {"G": "G0", "O": "O0"}
LEVELS = ("L1", "L2", "L3", "all")
PUBLISHED_K1 = {"registered": (14.4, 250), "adjusted": (16.2, 8.5, 24.1, 223)}


def label(planner: str, arm: str) -> str:
    return f"{planner}~{arm}"


def pc1(x):
    return None if x is None else float(f"{100 * x:.1f}")


def _same(a: dict, b: dict) -> bool:
    return {k: v for k, v in a.items() if k != "ts"} == {k: v for k, v in b.items() if k != "ts"}


class Scorer:
    def __init__(self, ns: str):
        self.ns = ns
        self.main = S.Inputs(ns, None)                               # sealed main journals (read only)
        self.gold, self.gold_sha = S._gold_load(None, None, verify=True)
        self.fam = S._family_map(None)
        d = os.path.join(A.DOCS, ns)
        self.units = json.load(open(os.path.join(d, "units.json"), encoding="utf-8"))
        self.cell_unit = {c["key"]: uk for uk, u in self.units["units"].items() for c in u["cells"]}
        self.ax, self.ahist, self.astarts = {}, {}, {}
        for r in A.read_jsonl(os.path.join(d, "exec.jsonl")):
            self.ax[r["key"]] = r
            self.ahist.setdefault(r["key"], []).append(r)
        self.ao = {r["key"]: r for r in A.read_jsonl(os.path.join(d, "outcome.jsonl"))}
        aev = A.read_jsonl(os.path.join(d, "events.jsonl"))
        for ev in aev:
            if ev.get("event") == A36.START_EVENT:
                self.astarts.setdefault(ev.get("key"), []).append(ev)
        self.aepoch = {}
        for ev in self.main.events + aev:
            if ev.get("event") == "invalid_resolved":
                for p in ev.get("pres") or []:
                    self.aepoch[p] = self.aepoch.get(p, 0) + 1
        self.inval = S._invalid_pres(self.main.events + aev)

    def self_check(self) -> dict:
        """score_cell on the MAIN journals reproduces the sealed G+M / G0 / O+M / O0 r0 rows (the calling convention used below)."""
        sealed = {r["key"]: r for r in A.read_jsonl(os.path.join(A.main_dir(self.ns), "score.jsonl"))}
        n, bad = 0, []
        for cell, prow in self.main.plan.items():
            if prow.get("arm") not in ("G+M", "G0", "O+M", "O0") or prow.get("r") != 0 or prow.get("block") != "main":
                continue
            cp = S._cell_parts(cell)
            pp = C.parse_pres(cp["pres"])
            if pp["world"] != "canon":
                continue
            unit = self.main.cell_unit.get(cell)
            ek = f"{self.ns}|exec|{unit}|run0"
            e, o = (self.main.by_unit.get((unit, 0)) or (None, None)) if prow["status"] == "ok" else (None, None)
            got = S.score_cell(cell, 0, plan_row=prow, exec_row=e, outcome=o, gold_item=self.gold[pp["item"]], family=self.fam.get(pp["item"]),
                               ns=self.ns, invalid_episode=cp["pres"] in self.inval,
                               exec_history=self.main.exec_hist.get(ek) if e else None,
                               exec_starts=(self.main.exec_starts.get(ek, []) if e else None),
                               block=S._block_of(self.main, cp, prow), exec_epoch=self.main.exec_epoch.get(cp["pres"], 0))
            want = sealed.get(C.cell_id(self.ns, "score", cp["arm"], cp["pres"], "r0", "run0"))
            n += 1
            if want is None or not _same(got, want):
                bad.append(cell)
        if bad:
            raise SystemExit(f"SELF-CHECK: score_cell does not reproduce {len(bad)} sealed main rows (first {bad[0]})")
        return {"reproduced_main_rows": n}

    def rows(self) -> tuple:
        out, missing = [], []
        for r in A.read_jsonl(PLANS):
            if r.get("ns") != self.ns:
                continue
            lab = label(r["planner"], r["arm"])
            cell = C.cell_id(self.ns, "plan", lab, r["pres"], "r0")
            pp = C.parse_pres(r["pres"])
            prow = {"status": r["status"], "plan": r["plan"], "first_plan": r["first_plan"], "first_ok": r.get("first_ok"),
                    "repaired": r.get("repaired"), "block": "main", "cost_usd": r.get("cost_usd", 0.0), "calls": r.get("calls", 0),
                    "scope": "strengthen-A", "attempts": []}
            kw = dict(plan_row=prow, gold_item=self.gold[pp["item"]], family=self.fam.get(pp["item"]), ns=self.ns,
                      invalid_episode=r["pres"] in self.inval, block="main")
            src = None
            if r["status"] != "ok":
                rec = S.score_cell(cell, 0, exec_row=None, outcome=None, **kw)
            else:
                uk = self.cell_unit.get(r["key"])
                u = self.units["units"].get(uk) if uk else None
                if u is None:
                    missing.append(r["key"])
                    continue
                ek = f"{self.ns}|exec|{uk}|run0"
                if u["reuse_main"]:
                    e, o = self.main.by_unit.get((uk, 0)) or (None, None)
                    hist, starts, ep, src = self.main.exec_hist.get(ek), self.main.exec_starts.get(ek, []), self.main.exec_epoch.get(r["pres"], 0), "main"
                else:
                    e, o = self.ax.get(ek), self.ao.get(ek)
                    hist, starts, ep, src = self.ahist.get(ek), self.astarts.get(ek, []), self.aepoch.get(r["pres"], 0), "A"
                    if e is not None and e.get("status") not in ("done", "not_executed") and int(e.get("attempt", 0)) < 2:
                        e = None                                        # an unfinished new unit (attempts left): not scored yet
                if e is None:
                    missing.append(r["key"])
                    continue
                rec = S.score_cell(cell, 0, exec_row=e, outcome=o, exec_history=hist, exec_starts=starts, exec_epoch=ep, **kw)
            rec.update(a_ns=self.ns, a_planner=r["planner"], a_arm=r["arm"], a_key=r["key"], a_source=r["source"], a_execution=src,
                       a_differs_from_full=r["differs_from_full"], a_equals_none_plan=r["equals_none_plan"],
                       a_same_request_as=r["same_request_as"])
            out.append(rec)
        return out, missing


# ------------------------------------------------------------------------------------------------ analysis
def level_index(S_, lvl_of, level):
    return S_ if level == "all" else {k: r for k, r in S_.items() if lvl_of.get(k[1]) == level}


def paired(idx, drop_invalid, a, b, level, adjusted=True):
    Sx = level_index(idx["S"], idx["level"], level)
    c = AN.contrast(Sx, idx["fam"], a, b, drop_invalid, None, drop_flag=A36.flagged if adjusted else None, amended=adjusted)
    if c is None:
        return {"a": a, "b": b, "level": level, "available": False}
    lo, hi = c["ci95"]
    return {"a": a, "b": b, "level": level, "available": True, "estimate": c["estimate"], "ci95": [lo, hi],
            "estimate_pp": pc1(c["estimate"]), "ci95_pp": [pc1(lo), pc1(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_presentations": c["n_pres"], "n_items": c["n_items"], "n_families": c["n_families"],
            "dropped_flagged_presentations": c["dropped_flagged"], "discordant_presentations": c["n_discordant"],
            "shared_execution_presentations": c["n_shared"], "families_a_better": c["families_a_better"], "families_tied": c["tie"],
            "families_b_better": c["families_b_better"]}


def rates(idx, drop_invalid, arm, adjusted):
    Sx = {k: r for k, r in idx["S"].items() if not (adjusted and A36.flagged(r))}
    out = {}
    for L in LEVELS:
        r = AN.arm_rates(Sx, idx["fam"], idx["level"], arm, drop_invalid, level=None if L == "all" else L)
        n = sum(1 for (a, i, l) in Sx if a == arm and (i, l) not in drop_invalid and (L == "all" or idx["level"].get(i) == L))
        out[L] = None if r is None else {"sr": r["sr"], "sr_pp": pc1(r["sr"]), "ci95_pp": [pc1(x) for x in r["ci95"]] if r["ci95"] else None,
                                         "n_items": r["n_items"], "n_presentations": n}
    return out


def plan_stats(plan_rows, score_rows, planner, arm) -> dict:
    rs = [r for r in plan_rows if r["planner"] == planner and r["arm"] == arm]
    sc = [r for r in score_rows if r["a_planner"] == planner and r["a_arm"] == arm]
    return {"cells": len(rs), "request_changed": sum(1 for r in rs if r["same_request_as"] is None),
            "request_equals_no_memory": sum(1 for r in rs if r["same_request_as"] == "none"),
            "planner_calls": sum(1 for r in rs if r["source"] == "call"), "shared_calls": sum(1 for r in rs if r["source"] == "shared_call"),
            "plan_differs_from_full": sum(1 for r in rs if r["differs_from_full"]),
            "plan_equals_no_memory_plan": sum(1 for r in rs if r["equals_none_plan"]),
            "plan_differs_and_equals_no_memory": sum(1 for r in rs if r["differs_from_full"] and r["equals_none_plan"]),
            "rejected": sum(1 for r in rs if r["status"] == "rejected"), "infra": sum(1 for r in rs if r["status"] == "infra"),
            "executions_new": len({r["unit"] for r in sc if r.get("a_execution") == "A"}),
            "executions_reused": len({r["unit"] for r in sc if r.get("a_execution") == "main"}),
            "usd": round(sum(float(r.get("cost_usd") or 0) for r in rs), 6)}


def analyse(ns: str, score_rows: list, check: dict) -> dict:
    main_rows = [r for r in A.read_jsonl(os.path.join(A.main_dir(ns), "score.jsonl")) if r.get("arm") in MAIN_ARMS]
    idx = AN.build_index(main_rows + score_rows)
    drop_invalid = AN.invalid_presentations(idx["S"])
    # reproduction gate: the published Kitchen 1 Scene Memory effect from the same path
    rep = {"registered": paired(idx, drop_invalid, "G+M", "G0", "all", adjusted=False), "adjusted": paired(idx, drop_invalid, "G+M", "G0", "all")}
    if ns == "test-v1":
        g, a = rep["registered"], rep["adjusted"]
        if (g["estimate_pp"], g["n_presentations"]) != PUBLISHED_K1["registered"] or \
                (a["estimate_pp"], a["ci95_pp"][0], a["ci95_pp"][1], a["n_presentations"]) != PUBLISHED_K1["adjusted"]:
            raise SystemExit(f"REPRODUCTION: G+M - G0 came out {g['estimate_pp']} / {a['estimate_pp']} {a['ci95_pp']} n={a['n_presentations']}, "
                             f"not the published {PUBLISHED_K1}")
    plan_rows = [r for r in A.read_jsonl(PLANS) if r.get("ns") == ns]
    arms = {}
    for pl in ("G", "O"):
        for arm in sorted({r["arm"] for r in plan_rows if r["planner"] == pl}):
            lab = label(pl, arm)
            arms[lab] = {"planner": pl, "arm": arm, "label": A.ARMS[arm]["label"], "plans": plan_stats(plan_rows, score_rows, pl, arm),
                         "rates": {"adjusted": rates(idx, drop_invalid, lab, True), "registered": rates(idx, drop_invalid, lab, False)},
                         "vs_full": {L: {"adjusted": paired(idx, drop_invalid, lab, FULL[pl], L),
                                         "registered": paired(idx, drop_invalid, lab, FULL[pl], L, adjusted=False)} for L in LEVELS},
                         "vs_none": {L: {"adjusted": paired(idx, drop_invalid, lab, NONE[pl], L),
                                         "registered": paired(idx, drop_invalid, lab, NONE[pl], L, adjusted=False)} for L in LEVELS}}
    context = {}
    for a, b in (("G+M", "G0"), ("G+M/p", "G0"), ("G+M", "G+M/p"), ("O+M", "O0"), ("O+M/p", "O0"), ("O+M", "O+M/p")):
        context[f"{a} - {b}"] = {L: {"adjusted": paired(idx, drop_invalid, a, b, L), "registered": paired(idx, drop_invalid, a, b, L, adjusted=False)}
                                 for L in LEVELS}
    base_rates = {arm: {"adjusted": rates(idx, drop_invalid, arm, True), "registered": rates(idx, drop_invalid, arm, False)} for arm in MAIN_ARMS}
    return {"schema": "strengthen.A.analysis/1", "ns": ns, "scene": A.KITCHENS[ns], "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "exploratory, post-hoc (decided after the results were seen); descriptive intervals, no test, no multiplicity adjustment",
            "bootstrap": {"B": 20000, "seed": 20261010, "level": 0.95, "cluster": "task family", "function": "analyze.contrast -> final_test_power.cluster_ci"},
            "adjusted_rule": "A36: a presentation whose three restoration attempts all stopped at the reset gate (excluded_reset) leaves its arm; "
                             "paired contrasts use the common subset",
            "checks": {**check, "reproduction": {k: {x: v[x] for x in ("estimate_pp", "ci95_pp", "n_presentations")} for k, v in rep.items()}},
            "baselines": base_rates, "context": context, "arms": arms}


def fmt(c):
    if not c or not c.get("available"):
        return "n/a"
    return f"{c['estimate_pp']:+.1f} [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}]"


def render_md(res: dict) -> str:
    L = [f"# Scene Memory ablations, {res['scene']} ({res['ns']}): summary", "",
         f"Exploratory, post-hoc. Generated by `sim/final_test/strengthen/ablate_analyze.py` ({res['written']}); numbers in `analysis.json`.",
         "SR in %, item-weighted; differences in percentage points, adjusted (A36 paired common subset), 95 % family-cluster bootstrap "
         "(B = 20000, seed 20261010). Every ablation arm runs with the full-memory executor policy (hook B), so arm minus full memory "
         "isolates the planner block.", "",
         "| Arm | Plans changed / cells | SR all | L1 | L2 | L3 | arm - full memory | arm - no memory |", "|---|---|---|---|---|---|---|---|"]
    for lab in ("G+M", "G0", "G+M/p"):
        r = res["baselines"][lab]["adjusted"]
        L.append(f"| {lab} (existing) | | " + " | ".join(f"{(r[x] or {}).get('sr_pp')}" for x in LEVELS[3:] + LEVELS[:3]) + " | | |")
    for lab, v in res["arms"].items():
        r = v["rates"]["adjusted"]
        p = v["plans"]
        L.append(f"| {lab} {v['arm']} | {p['plan_differs_from_full']} / {p['cells']} | " +
                 " | ".join(f"{(r[x] or {}).get('sr_pp')}" for x in LEVELS[3:] + LEVELS[:3]) +
                 f" | {fmt(v['vs_full']['all']['adjusted'])} | {fmt(v['vs_none']['all']['adjusted'])} |")
    L += ["", "Context (existing arms): " + "; ".join(f"{k} {fmt(v['all']['adjusted'])}" for k, v in res["context"].items()), ""]
    for key, title in (("vs_full", "arm minus full memory"), ("vs_none", "arm minus no memory")):
        L += [f"Per level, {title} (adjusted; registered in analysis.json):", "",
              "| Arm | L1 Atomic | L2 Sequential | L3 Abstract | Overall | n pres (overall) |", "|---|---|---|---|---|---|"]
        for lab, v in res["arms"].items():
            c = v[key]
            L.append(f"| {lab} | " + " | ".join(fmt(c[x]["adjusted"]) for x in LEVELS) +
                     f" | {(c['all']['adjusted'] or {}).get('n_presentations')} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ns", default=A.NS, choices=sorted(A.KITCHENS))
    ap.add_argument("--allow-partial", action="store_true", help="analyse although some new units are not executed yet (diagnostic only)")
    a = ap.parse_args(argv)
    sc = Scorer(a.ns)
    check = sc.self_check()
    rows, missing = sc.rows()
    if missing and not a.allow_partial:
        print(f"{len(missing)} ablation cell(s) have no finished execution yet (first {missing[0]}): run A4 first", file=sys.stderr)
        return 2
    keep = [r for r in A.read_jsonl(SCORE) if r.get("a_ns") != a.ns]
    A.atomic_write_jsonl(SCORE, keep + rows)
    res = analyse(a.ns, rows, dict(check, n_score_rows=len(rows), missing=len(missing), gold_sha256=sc.gold_sha))
    allres = {}
    if os.path.exists(ANALYSIS):
        allres = json.load(open(ANALYSIS, encoding="utf-8"))
    allres[a.ns] = res
    A.atomic_write_json(ANALYSIS, allres)
    A.atomic_write_text(SUMMARY, "".join(render_md(v) + "\n" for _k, v in sorted(allres.items())))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
