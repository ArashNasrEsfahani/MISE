"""Regenerate the fixed-procedure prompt comparison (A7) from the records in this repository and check it against the shipped analysis.

    python docs/final-test/reexecuted/a7_fixed.py [--out DIR]

The A7 no-exclusion rows (inputs/rows/A7.no-exclusion.jsonl, scored from docs/final-test/strengthen/A/a7/) join the main arms and the A6 rows
exactly as joint_bootstrap.load builds them, through the same rescue rule; rows still unrestored leave their paired comparisons. Each contrast
is reported per level with per-kitchen intervals and, for three-kitchen means, the joint family bootstrap interval. Writes a7_fixed.json and
compares every value with docs/final-test/strengthen/A/a7/analysis.json (exit 1 on any difference). Python with numpy and scipy only.
"""
import sys

sys.dont_write_bytecode = True

import argparse   # noqa: E402
import json       # noqa: E402
import os         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_bootstrap as JB   # noqa: E402

CF, CN, AN = JB.CF, JB.CN, JB.AN
LEVELS = ("all", "L1", "L2", "L3")
CONTRASTS_3K = [("G~fixed_proc", "G+M"), ("G~fixed_proc", "G~only_req_proc"), ("G~fixed_proc", "G0"),
                ("G~fixed_proc_300", "G+M"), ("G~fixed_proc_300", "G~only_req_proc"), ("G~fixed_proc_300", "G0"),
                ("G~fixed_proc_300", "G~fixed_proc")]
CONTRASTS_K1 = [("G~fixed_proc", "G~only_request"), ("G~fixed_proc_300", "G~only_request")]
SHIPPED = os.path.join(CN.S_, "A", "a7", "analysis.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    src = CN.sources()
    raw_new = {K: sum((AN.read_jsonl(p) for p in new), []) for K, (_o, new) in src.items()}
    gold, _ = CF.S._gold_load(None, None, verify=True)
    rs = CF.rescue_sources()
    raw_final = {K: CF.apply_rescue(raw_new[K], rs, gold) for K in CN.NS}
    extra = AN.read_jsonl(os.path.join(CN.INPUTS, "rows", "A6.no-exclusion.jsonl")) + AN.read_jsonl(os.path.join(CN.INPUTS, "rows", "A7.no-exclusion.jsonl"))
    idx = {}
    for K, ns in CN.NS.items():
        base = [r for r in raw_final[K] if r.get("arm") in CF.A6_MAIN]
        mine = [r for r in extra if r["namespace"] == ns and r["a_version"] == "no-exclusion"]
        idx[K] = CF.idx_of(base + CF.apply_rescue(mine, rs, gold))
    out = {}
    for x, y, three in [(p, q, True) for p, q in CONTRASTS_3K] + [(p, q, False) for p, q in CONTRASTS_K1]:
        name = f"{x} - {y}"
        out[name] = {}
        for L in LEVELS:
            pts = [JB.parts_named(idx[K][0], idx[K][1], x, y, L) for K in (CN.NS if three else ["K1"])]
            if any(p is None for p in pts):
                continue
            est = [float(sum(t)) / float(sum(s)) for _f, t, s in pts]
            c = {"per_kitchen_pp": [JB.pc1(e) for e in est], "per_kitchen_ci95_pp": [JB.interval([p], "independent")["ci95_pp"] for p in pts]}
            if three:
                c.update(estimate_pp=JB.pc1(sum(est) / 3), joint_ci95_pp=JB.interval(pts, "joint")["ci95_pp"])
            out[name][L] = c
    with open(os.path.join(os.path.abspath(a.out), "a7_fixed.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    shipped = json.load(open(SHIPPED, encoding="utf-8"))["contrasts"]
    diff = [n for n in out if out[n] != shipped.get(n)]
    for n, c in out.items():
        v = c["all"]
        print(f"{n:42s} " + (f"{v['estimate_pp']:+6.1f} {v['joint_ci95_pp']}" if "estimate_pp" in v else f"{v['per_kitchen_pp'][0]:+6.1f} {v['per_kitchen_ci95_pp'][0]}"))
    print("matches the shipped analysis" if not diff else f"DIFFERS from the shipped analysis: {diff}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
