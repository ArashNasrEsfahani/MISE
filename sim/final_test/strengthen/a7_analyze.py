"""A7 analysis (Windows, offline; exploratory follow-up): the two fixed-procedure arms against full Scene Memory, the retrieved-procedure arm,
the rephrasing-only arm (development kitchen) and no memory, under the paper's final rules.

    python sim/final_test/strengthen/a7_analyze.py        (repo root)

Scoring: a6_analyze.Kitchen with a7_common in place of a6_common (A7 plan rows only; executions adopted from every source, A7's own last;
three-reset-stop histories replaced by a re-run: task R, A6's, else A7's). Final rows: the no-exclusion rows + compute_final.apply_rescue,
next to the A6 rows and the main arms exactly as docs/final-test/reexecuted/joint_bootstrap.load builds them; paired contrasts drop rows still
flagged (A36). Three-kitchen interval: the paper's joint family bootstrap (joint_bootstrap.interval, mode 'joint'); per-kitchen intervals: the
same code on one kitchen. Read-only on everything else; writes docs/final-test/strengthen/A/a7/{score.jsonl, analysis.json, summary.md}.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "docs", "final-test", "reexecuted"))
import a7_common as A7             # noqa: E402
import ablate_common as A          # noqa: E402
import a6_analyze as A6AN          # noqa: E402
import joint_bootstrap as JB       # noqa: E402

A6AN.A6 = A7                                       # this process only: Kitchen reads A7's sources, units and plans
CF, CN, AN, AA = JB.CF, JB.CN, JB.AN, A6AN.AA
LEVELS = ("all", "L1", "L2", "L3")


def a7_rows():
    gold, fam = A6AN.S._gold_load(None, None, verify=True)[0], A6AN.S._family_map(None)
    out, info = {}, {}
    for ns in A7.NAMESPACES:
        K = A6AN.Kitchen(ns, gold, fam)
        K.plans = [r for r in K.plans if r.get("a_phase") == A7.PHASE]
        adj, nox, missing, inf = K.score()
        if missing:
            raise SystemExit(f"{ns}: {len(missing)} A7 cell(s) without a finished execution / re-run (first {missing[0]})")
        out[ns] = (adj, nox)
        info[ns] = {"cells": len(nox), "new_execution_cells": inf["new_execution_cells"],
                    "replaced_by_rerun": sum(1 for x in nox if x.get("a_replaced")),
                    "execution_sources": dict((s, sum(1 for x in nox if x.get("a_execution") == s)) for s in sorted({str(x.get("a_execution")) for x in nox}))}
    return out, info


def main():
    t0 = time.time()
    rows7, info = a7_rows()
    A.atomic_write_jsonl(os.path.join(A7.DOCS, "score.jsonl"), [x for ns in rows7 for v in rows7[ns] for x in v])
    # the paper's final rows (joint_bootstrap.load), plus A7's no-exclusion rows through the same rescue rule
    src = CN.sources()
    raw_new = {K: sum((AN.read_jsonl(p) for p in new), []) for K, (_o, new) in src.items()}
    gold, _ = CF.S._gold_load(None, None, verify=True)
    rs = CF.rescue_sources()
    raw_final = {K: CF.apply_rescue(raw_new[K], rs, gold) for K in CN.NS}
    a6rows = AN.read_jsonl(os.path.join(CN.S_, "A", "a6", "score.jsonl"))
    idx = {}
    for K, ns in CN.NS.items():
        base = [r for r in raw_final[K] if r.get("arm") in CF.A6_MAIN]
        extra = [r for r in a6rows if r["namespace"] == ns and r["a_version"] == "no-exclusion"] + rows7[ns][1]
        idx[K] = CF.idx_of(base + CF.apply_rescue(extra, rs, gold))
    res = {"schema": "strengthen.A7.analysis/1", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "exploratory follow-up",
           "definition": A7.__doc__.split("\n\n")[1].replace("\n", " "), "arms": {k: v["label"] for k, v in A7.NEW_ARMS.items()},
           "execution": info, "B": JB.B, "seed": JB.SEED, "rates": {}, "contrasts": {}}
    labs = ["G~fixed_proc", "G~fixed_proc_300", "G~only_req_proc", "G~only_request", "G+M", "G0"]
    for K, ns in CN.NS.items():
        res["rates"][ns] = {lab: AA.rates(idx[K][0], idx[K][1], lab, True) for lab in labs if any(k[0] == lab for k in idx[K][0]["S"])}
    todo = [(a, b, True) for a, b in A7.CONTRASTS_3K] + [(a, b, False) for a, b in A7.CONTRASTS_K1]
    for a, b, three in todo:
        name = f"{a} - {b}"
        res["contrasts"][name] = {}
        for L in LEVELS:
            pts = [JB.parts_named(idx[K][0], idx[K][1], a, b, L) for K in (CN.NS if three else ["K1"])]
            if any(p is None for p in pts):
                continue
            est = [float(sum(t)) / float(sum(s)) for _f, t, s in pts]
            per = [JB.interval([p], "independent")["ci95_pp"] for p in pts]
            c = {"per_kitchen_pp": [JB.pc1(e) for e in est], "per_kitchen_ci95_pp": per}
            if three:
                jt = JB.interval(pts, "joint")
                c.update(estimate_pp=JB.pc1(sum(est) / 3), joint_ci95_pp=jt["ci95_pp"])
            res["contrasts"][name][L] = c
    if JB.bad:
        raise SystemExit("gate J2 failed: " + "; ".join(JB.bad[:5]))
    A.atomic_write_json(os.path.join(A7.DOCS, "analysis.json"), res)
    f = lambda c: (f"{c['estimate_pp']:+.1f} [{c['joint_ci95_pp'][0]:+.1f}, {c['joint_ci95_pp'][1]:+.1f}]" if "estimate_pp" in c   # noqa: E731
                   else f"{c['per_kitchen_pp'][0]:+.1f} [{c['per_kitchen_ci95_pp'][0][0]:+.1f}, {c['per_kitchen_ci95_pp'][0][1]:+.1f}]")
    md = ["# A7: fixed-procedure prompt baseline (exploratory follow-up)", "",
          f"Generated by `sim/final_test/strengthen/a7_analyze.py` ({res['written']}). Success-rate differences in percentage points; "
          "three-kitchen mean with the joint family bootstrap interval (B = 20000), as in the paper. K1-only rows: development kitchen.", "",
          "| Contrast | All | Atomic | Sequential | Abstract | per kitchen (all) |", "|---|---|---|---|---|---|"]
    for name, c in res["contrasts"].items():
        md.append(f"| {name} | " + " | ".join(f(c[L]) if L in c else "" for L in LEVELS) + f" | {', '.join('%+.1f' % x for x in c['all']['per_kitchen_pp'])} |")
    md += ["", "Success rate (%, all levels) per kitchen: " + "; ".join(
        f"{ns}: " + ", ".join(f"{lab} {100 * v['all']['sr']:.1f}" if isinstance(v.get("all"), dict) and "sr" in v["all"] else f"{lab} ?"
                              for lab, v in r.items()) for ns, r in res["rates"].items()), "",
           "Executions: " + json.dumps(info), ""]
    A.atomic_write_text(os.path.join(A7.DOCS, "summary.md"), "\n".join(md) + "\n")
    print("\n".join(md))
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
