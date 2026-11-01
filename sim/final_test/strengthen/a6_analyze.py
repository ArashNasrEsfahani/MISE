"""A6 analysis (Windows, offline; exploratory, post-hoc): every ablation arm of A3 + A6 in the three kitchens, adjusted AND no-exclusion,
with Holm across the primary Gemma arm family per kitchen, three-kitchen means, and the Kitchen 1 execution-noise floor.

    python sim/final_test/strengthen/a6_analyze.py [--allow-partial]

Scoring: `score.score_cell` (the registered scorer) on the execution each cell adopts (a6_common.sources: sealed main [+ Sol addendum],
A4, task L, A6), block "main" so the A36 core rule applies as to G+M. Two versions per cell:
  adjusted      the adopted execution; a three-reset-stop history carries excluded_reset and leaves both arms of a paired contrast (A36)
  no-exclusion  a three-reset-stop history is replaced by its re-run (task R's pass for a sealed unit R re-ran, else A6's own re-run);
                nothing is dropped, a re-run that failed restoration again counts as a failure (task R's definition). The main arms'
                no-exclusion rows are task R's own (R/<ns>/score.noexcl.jsonl).
Contrasts: analyze.contrast -> final_test_power.cluster_ci (95 % family-cluster bootstrap, B = 20000, seed 20261010), per level.
Holm: two-sided wild-cluster p (final_test_power.wcr_p, H0 D = 0, B = 20000), analyze.holm over a6_common.PRIMARY[ns] (arm minus full
memory, overall), reported next to the unadjusted intervals. Three-kitchen mean: equal-weight mean of the per-kitchen estimates (the
paper's 13.0 rule), interval from a kitchen-stratified family bootstrap. Noise floor: the A6 noise pass (60 sealed G+M units executed
again) -> flip rate p; an arm's execution-noise band = 1.96 sqrt(k p) / N (k = its cells on a new execution, N = presentations).
Reproduction gates: G+M - G0 adjusted K1 +16.2 / K2 +14.0 / K3 +8.9 (mean 13.0) and R's no-exclusion K1 +16.4 from the same path.

Writes ONLY docs/final-test/strengthen/A/a6/{score.jsonl, analysis.json, summary_tables_a6.md} (atomic).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a6_common as A6             # noqa: E402
import ablate_common as A          # noqa: E402
import ablate_analyze as AA        # noqa: E402

import a36 as A36                  # noqa: E402
import analyze as AN               # noqa: E402
import common as C                 # noqa: E402
import score as S                  # noqa: E402
import final_test_power as P       # noqa: E402

SCORE = os.path.join(A6.DOCS, "score.jsonl")
ANALYSIS = os.path.join(A6.DOCS, "analysis.json")
TABLES = os.path.join(A6.DOCS, "summary_tables_a6.md")
MAIN_ARMS = ("G0", "G+M", "O0", "O+M", "G+M/p", "O+M/p")
LEVELS = AA.LEVELS
B = 20000
PUBLISHED = {"adjusted": {"test-v1": 16.2, "test-k2-v1": 14.0, "test-k3-v1": 8.9}, "mean": 13.0}


class Kitchen:
    def __init__(self, ns: str, gold, fam):
        self.ns, self.gold, self.fam = ns, gold, fam
        self.main = S.Inputs(ns, None)
        self.srcs = A6.sources(ns)
        self.rsrcs = A6.rerun_sources(ns)
        self.plans = A6.all_plans(ns)
        self.cell_unit = {}
        a4 = os.path.join(A.DOCS, ns, "units.json")
        p6 = A6.pass_paths(ns, "new")["units"]
        for p in (a4, p6):
            if os.path.exists(p):
                for uk, u in json.load(open(p, encoding="utf-8"))["units"].items():
                    for c in u["cells"]:
                        self.cell_unit[c["key"]] = uk
        self.inval = S._invalid_pres(self.main.events)

    def score(self) -> tuple:
        adj, nox, missing, info = [], [], [], {"new_execution_cells": {}}
        for r in self.plans:
            lab = AA.label(r["planner"], r["arm"])
            cell = C.cell_id(self.ns, "plan", lab, r["pres"], "r0")
            pp = C.parse_pres(r["pres"])
            prow = {"status": r["status"], "plan": r["plan"], "first_plan": r["first_plan"], "first_ok": r.get("first_ok"),
                    "repaired": r.get("repaired"), "block": "main", "cost_usd": r.get("cost_usd", 0.0), "calls": r.get("calls", 0),
                    "scope": "strengthen-A", "attempts": []}
            kw = dict(plan_row=prow, gold_item=self.gold[pp["item"]], family=self.fam.get(pp["item"]), ns=self.ns,
                      invalid_episode=r["pres"] in self.inval, block="main")
            meta = dict(a_ns=self.ns, a_planner=r["planner"], a_arm=r["arm"], a_key=r["key"], a_phase=r.get("a_phase"), a_source=r["source"],
                        a_differs_from_full=r["differs_from_full"], a_equals_none_plan=r["equals_none_plan"],
                        a_same_request_as=r["same_request_as"])
            if r["status"] != "ok":
                x = S.score_cell(cell, 0, exec_row=None, outcome=None, **kw)
                adj.append(dict(x, **meta, a_version="adjusted", a_execution=None))
                nox.append(dict(x, **meta, a_version="no-exclusion", a_execution=None))
                continue
            uk = self.cell_unit.get(r["key"])
            src, hist = A6.adopt(self.ns, uk, self.srcs) if uk else (None, None)
            if src is None:
                missing.append(r["key"])
                continue
            ek = A6.exec_key(self.ns, uk)
            x = S.score_cell(cell, 0, exec_row=hist[-1], outcome=src.outc.get(ek), exec_history=hist, exec_starts=src.starts.get(ek, []),
                             exec_epoch=src.epoch.get(r["pres"], 0), **kw)
            adj.append(dict(x, **meta, a_version="adjusted", a_execution=src.name))
            if src.name != "main" and r["differs_from_full"]:
                info["new_execution_cells"][lab] = info["new_execution_cells"].get(lab, 0) + 1
            if A6.three_blocked(hist):
                rs, rh = A6.adopt(self.ns, uk, self.rsrcs)
                if rs is None:
                    missing.append("rerun:" + r["key"])
                    continue
                y = S.score_cell(cell, 0, exec_row=rh[-1], outcome=rs.outc.get(ek), exec_history=rh, exec_starts=rs.starts.get(ek, []),
                                 exec_epoch=rs.epoch.get(r["pres"], 0), **kw)
                nox.append(dict(y, **meta, a_version="no-exclusion", a_execution=rs.name, a_replaced=src.name))
            else:
                nox.append(dict(x, **meta, a_version="no-exclusion", a_execution=src.name))
        return adj, nox, missing, info

    def index(self, rows: list, version: str):
        if version == "no-exclusion":
            base = [r for r in A.read_jsonl(os.path.join(A6.STRENGTHEN, "R", self.ns, "score.noexcl.jsonl")) if r.get("arm") in MAIN_ARMS]
        else:
            base = [r for r in A.read_jsonl(os.path.join(A.main_dir(self.ns), "score.jsonl")) if r.get("arm") in MAIN_ARMS]
        idx = AN.build_index(base + rows)
        return idx, AN.invalid_presentations(idx["S"])


def pval(idx, drop_invalid, a, b, adjusted) -> float | None:
    c = AN.contrast(idx["S"], idx["fam"], a, b, drop_invalid, None, drop_flag=A36.flagged if adjusted else None, amended=adjusted)
    if c is None:
        return None
    tot, sizes = c["_tot"], c["_sizes"]
    pg = P.wcr_p(tot, sizes, 0.0, "greater", B)
    pl = P.wcr_p(tot, sizes, 0.0, "less", B)
    return min(1.0, 2 * min(pg, pl))


def parts(idx, drop_invalid, a, b, adjusted):
    c = AN.contrast(idx["S"], idx["fam"], a, b, drop_invalid, None, drop_flag=A36.flagged if adjusted else None, amended=adjusted)
    return None if c is None else (c["_tot"], c["_sizes"])


def stratified_mean(pts: list, seed: int = P.SEED) -> dict:
    """Equal-weight mean of per-kitchen paired estimates; 95 % percentile interval, families resampled within each kitchen."""
    import numpy as np
    est = [float(sum(t)) / float(sum(s)) for t, s in pts]
    sims = []
    for k, (t, s) in enumerate(pts):
        T_, n_ = np.asarray(t, float), np.asarray(s, float)
        F = len(n_)
        W = np.random.default_rng(seed + k).multinomial(F, [1 / F] * F, size=B).astype(float)
        sims.append((W @ T_) / np.maximum(W @ n_, 1e-12))
    m = np.mean(np.vstack(sims), axis=0)
    return {"estimate": sum(est) / len(est), "estimate_pp": AA.pc1(sum(est) / len(est)), "per_kitchen_pp": [AA.pc1(e) for e in est],
            "ci95_pp": [AA.pc1(float(np.percentile(m, 2.5))), AA.pc1(float(np.percentile(m, 97.5)))]}


def noise_floor() -> dict:
    """Kitchen 1: the A6 noise pass vs the sealed run 0 of the same units; plus the sealed REPEAT re-executions (run 1 vs run 0) as context."""
    ns = "test-v1"
    pp = A6.pass_paths(ns, "noise")
    if not os.path.exists(pp["units"]):
        return {"available": False}
    units = json.load(open(pp["units"], encoding="utf-8"))["units"]
    main = A6.sources(ns)[0]
    mine = A6.Source("A6-noise", pp["exec"], pp["outcome"], pp["events"])
    n = flips = blocked = up = down = 0
    pending = 0
    for uk in units:
        ek = A6.exec_key(ns, uk)
        h0, h1 = main.get(ek), mine.get(ek)
        if not h1:
            pending += 1
            continue
        if h1[-1].get("status") != "done":
            blocked += 1
            continue
        g0 = bool((main.outc.get(ek) or {}).get("goal_ok"))
        g1 = bool((mine.outc.get(ek) or {}).get("goal_ok"))
        n += 1
        flips += g0 != g1
        up += g1 and not g0
        down += g0 and not g1
    rep = {"n": 0, "flips": 0}
    by = {}
    for r in A.read_jsonl(os.path.join(A.main_dir(ns), "score.jsonl")):
        if r.get("arm") == "G+M" and r.get("world") == "canon" and r.get("r") == 0:
            by.setdefault(r["pres"], {})[r["run"]] = r
    for v in by.values():
        if 0 in v and 1 in v and not (v[0].get("excluded_reset") or v[1].get("excluded_reset") or v[0].get("exec_failure") or v[1].get("exec_failure")):
            rep["n"] += 1
            rep["flips"] += v[0]["sr"] != v[1]["sr"]
    p = flips / n if n else None
    pooled = (flips + rep["flips"]) / (n + rep["n"]) if (n + rep["n"]) else None
    return {"available": True, "sample": len(units), "pending": pending, "blocked_three_times": blocked, "n_compared": n, "flips": flips,
            "now_succeeds": up, "now_fails": down, "flip_rate": p, "seed": json.load(open(pp["units"], encoding="utf-8")).get("noise_seed"),
            "sealed_repeat_reexecutions": rep, "flip_rate_pooled": pooled}


def analyse(allow_partial: bool) -> dict:
    gold, gold_sha = S._gold_load(None, None, verify=True)
    fam = S._family_map(None)
    res = {"schema": "strengthen.A6.analysis/1", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "status": "exploratory, post-hoc; descriptive intervals; Holm reported next to them across the primary Gemma arm family per kitchen",
           "bootstrap": {"B": B, "seed": P.SEED}, "gold_sha256": gold_sha, "kitchens": {}, "missing": {}}
    rows_out = []
    noise = noise_floor()
    res["noise_floor"] = noise
    parts_by = {}
    for ns in ("test-v1", "test-k2-v1", "test-k3-v1"):
        K = Kitchen(ns, gold, fam)
        adj, nox, missing, info = K.score()
        res["missing"][ns] = len(missing)
        if missing and not allow_partial:
            raise SystemExit(f"{ns}: {len(missing)} cell(s) without a finished execution / re-run (first {missing[0]})")
        rows_out += adj + nox
        kb = {"role": A6.ROLE[ns], "versions": {}}
        for version, rows in (("adjusted", adj), ("no-exclusion", nox)):
            idx, dinv = K.index(rows, version)
            adjusted = version == "adjusted"
            v = {"baselines": {}, "arms": {}, "context": {}, "holm": {}}
            for arm in MAIN_ARMS:
                v["baselines"][arm] = AA.rates(idx, dinv, arm, adjusted)
            for a, b in (("G+M", "G0"), ("G+M/p", "G0"), ("G+M", "G+M/p"), ("O+M", "O0")):
                if any(k[0] == a for k in idx["S"]) and any(k[0] == b for k in idx["S"]):
                    v["context"][f"{a} - {b}"] = {L: AA.paired(idx, dinv, a, b, L, adjusted) for L in LEVELS}
                    parts_by[(version, ns, f"{a} - {b}")] = parts(idx, dinv, a, b, adjusted)
            plan_rows = [r for r in K.plans]
            for pl in ("G", "O"):
                full, none = AA.FULL[pl], AA.NONE[pl]
                for arm in sorted({r["arm"] for r in plan_rows if r["planner"] == pl}):
                    lab = AA.label(pl, arm)
                    mine = [x for x in rows if x["a_planner"] == pl and x["a_arm"] == arm]
                    ps_ = AA.plan_stats(plan_rows, [], pl, arm)
                    ps_.update(executions_new=len({x["unit"] for x in mine if x.get("a_execution") not in (None, "main", "codex")}),
                               executions_reused_sealed=len({x["unit"] for x in mine if x.get("a_execution") in ("main", "codex")}),
                               execution_sources=sorted({str(x.get("a_execution")) for x in mine}))
                    v["arms"][lab] = {"planner": pl, "arm": arm, "label": A.ARMS[arm]["label"], "plans": ps_,
                                      "new_execution_cells": info["new_execution_cells"].get(lab, 0),
                                      "rates": AA.rates(idx, dinv, lab, adjusted),
                                      "vs_full": {L: AA.paired(idx, dinv, lab, full, L, adjusted) for L in LEVELS},
                                      "vs_none": {L: AA.paired(idx, dinv, lab, none, L, adjusted) for L in LEVELS}}
                    parts_by[(version, ns, lab)] = parts(idx, dinv, lab, full, adjusted)
            fam_arms = [AA.label("G", a) for a in A6.PRIMARY[ns] if AA.label("G", a) in v["arms"]]
            ps = {lab: pval(idx, dinv, lab, "G+M", adjusted) for lab in fam_arms}
            v["holm"] = {"family": fam_arms, "m": len(fam_arms), "alpha": 0.05, "test": "two-sided wild cluster bootstrap, H0 D = 0",
                         "members": AN.holm(ps)}
            kb["versions"][version] = v
        res["kitchens"][ns] = kb
    # reproduction gates (adjusted, the paper's per-kitchen Scene Memory effects and their mean)
    got = {ns: res["kitchens"][ns]["versions"]["adjusted"]["context"]["G+M - G0"]["all"]["estimate_pp"] for ns in res["kitchens"]}
    if got != PUBLISHED["adjusted"]:
        raise SystemExit(f"REPRODUCTION: G+M - G0 adjusted per kitchen {got} != published {PUBLISHED['adjusted']}")
    # three-kitchen means
    res["three_kitchen"] = {}
    for version in ("adjusted", "no-exclusion"):
        tk = {"G+M - G0": stratified_mean([parts_by[(version, ns, "G+M - G0")] for ns in res["kitchens"]])}
        for name, arm in A6.THREE_KITCHEN.items():
            labs = {ns: AA.label("G", arm[ns] if isinstance(arm, dict) else arm) for ns in res["kitchens"]}
            pts = [parts_by.get((version, ns, labs[ns])) for ns in res["kitchens"]]
            if all(pts):
                tk[name + " - G+M"] = dict(stratified_mean(pts), arms=labs)
        pts = [parts_by.get((version, ns, "O+M - O0")) for ns in res["kitchens"]]
        if all(pts):
            tk["O+M - O0"] = stratified_mean(pts)
        for arm in A6.THREE_KITCHEN_OPUS:                            # Opus 5.5 (2026-10-28 21:40 extension)
            pts = [parts_by.get((version, ns, AA.label("O", arm))) for ns in res["kitchens"]]
            if all(pts):
                tk[f"O {arm} - O+M"] = dict(stratified_mean(pts), arms={ns: AA.label("O", arm) for ns in res["kitchens"]})
        res["three_kitchen"][version] = tk
    if res["three_kitchen"]["adjusted"]["G+M - G0"]["estimate_pp"] != PUBLISHED["mean"]:
        raise SystemExit(f"REPRODUCTION: three-kitchen mean {res['three_kitchen']['adjusted']['G+M - G0']} != {PUBLISHED['mean']}")
    # execution-noise bands (Kitchen 1 flip rate, pooled with the sealed REPEAT re-executions when available)
    p = noise.get("flip_rate_pooled") if noise.get("available") else None
    for ns, kb in res["kitchens"].items():
        for version, v in kb["versions"].items():
            for lab, a in v["arms"].items():
                k = a["new_execution_cells"]
                Np = a["vs_full"]["all"].get("n_presentations") if a["vs_full"]["all"].get("available") else 0
                if p is None or not Np:
                    a["noise_band_pp"] = None
                    continue
                band = 1.96 * math.sqrt(k * p) / Np
                a["noise_band_pp"] = AA.pc1(band)
                est = a["vs_full"]["all"]["estimate"]
                a["exceeds_noise"] = abs(est) > band
    A.atomic_write_jsonl(SCORE, rows_out)
    return res


def fmt(c):
    return AA.fmt(c)


def render(res: dict) -> str:
    L = ["# A6: strengthened Scene Memory ablation (tables)", "",
         f"Generated by `sim/final_test/strengthen/a6_analyze.py` ({res['written']}); numbers in `analysis.json`. Exploratory, post-hoc.",
         "Differences in percentage points, arm minus full memory; 95 % family-cluster bootstrap intervals (B = 20000); Holm p across the",
         "primary Gemma arm family of each kitchen. Adjusted = A36 paired common subset; no-exclusion = restoration failures replaced by",
         "their re-run (a re-run that failed again counts as a failure).", ""]
    for ns, kb in res["kitchens"].items():
        L += [f"## {kb['role']}", "", "| Arm | Plans changed | Adjusted | Holm p (adj) | No-exclusion | Holm p (no-excl) | Noise band (adj) |",
              "|---|---|---|---|---|---|---|"]
        va, vn = kb["versions"]["adjusted"], kb["versions"]["no-exclusion"]
        for lab, a in va["arms"].items():
            n = vn["arms"].get(lab, {})
            ha = (va["holm"]["members"].get(lab) or {}).get("p_holm")
            hn = (vn["holm"]["members"].get(lab) or {}).get("p_holm")
            L.append(f"| {lab} | {a['plans']['plan_differs_from_full']} / {a['plans']['cells']} | {fmt(a['vs_full']['all'])} | "
                     f"{'' if ha is None else f'{ha:.3f}'} | {fmt((n.get('vs_full') or {}).get('all'))} | {'' if hn is None else f'{hn:.3f}'} | "
                     f"{'' if a.get('noise_band_pp') is None else '±%.1f' % a['noise_band_pp']} |")
        L += ["", "Per level, adjusted (L1 / L2 / L3): " + "; ".join(
            f"{lab} {' / '.join(fmt(a['vs_full'][x]) for x in ('L1', 'L2', 'L3'))}" for lab, a in va["arms"].items() if lab.startswith("G~")), "",
              "Context: " + "; ".join(f"{k} adj {fmt(v['all'])} / no-excl {fmt(vn['context'].get(k, {}).get('all'))}" for k, v in va["context"].items()), ""]
    L += ["## Three-kitchen means (equal weight)", "", "| Contrast | Adjusted | No-exclusion | per kitchen (adj) |", "|---|---|---|---|"]
    for k, v in res["three_kitchen"]["adjusted"].items():
        n = res["three_kitchen"]["no-exclusion"].get(k) or {}
        L.append(f"| {k} | {v['estimate_pp']:+.1f} [{v['ci95_pp'][0]:+.1f}, {v['ci95_pp'][1]:+.1f}] | "
                 f"{n.get('estimate_pp', float('nan')):+.1f} [{(n.get('ci95_pp') or [float('nan')] * 2)[0]:+.1f}, {(n.get('ci95_pp') or [float('nan')] * 2)[1]:+.1f}] | "
                 f"{', '.join('%+.1f' % x for x in v['per_kitchen_pp'])} |")
    nf = res["noise_floor"]
    if nf.get("available"):
        L += ["", f"Execution-noise floor (Kitchen 1): {nf['flips']} of {nf['n_compared']} re-executed G+M plans changed outcome "
                  f"({nf['now_succeeds']} now succeed, {nf['now_fails']} now fail; {nf['blocked_three_times']} blocked three times, {nf['pending']} pending); "
                  f"sealed REPEAT re-executions {nf['sealed_repeat_reexecutions']['flips']} / {nf['sealed_repeat_reexecutions']['n']}; "
                  f"pooled flip rate {nf['flip_rate_pooled']:.3f}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    res = analyse(a.allow_partial)
    A.atomic_write_json(ANALYSIS, res)
    A.atomic_write_text(TABLES, render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
