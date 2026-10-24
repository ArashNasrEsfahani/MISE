"""Selftest for the H1 parts of sim/final_test/matrix/analyze.py (stage H-H1-stats, A24.1 h1-rule-1 = C7 + exploratory E1).
Offline: no simulator, no network, no API, no test item.

    python sim/_ftm_analyze_h1_selftest.py
"""
from __future__ import annotations

import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "final_test", "matrix")
sys.path.insert(0, HERE)
sys.path.insert(0, MATRIX)

import common as C                      # noqa: E402
import analyze as A                     # noqa: E402
import final_test_h1_stats as T         # noqa: E402

N = {"pass": 0, "fail": 0}
B = 999


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def mk(arm, item, lang, sr, fam, lvl="L1", infra=False, r=0, run=0):
    return {"key": f"{arm}|{item}|{lang}|{r}|{run}", "namespace": "rehearsal-v1", "arm": arm, "pres": C.pres_id(item, lang), "r": r, "run": run,
            "unit": f"u|{arm}|{item}|{lang}", "sr": float(sr), "strict_sr": float(sr), "valid": True, "infra": infra,
            "level": lvl, "lang": lang, "family": fam}


def grid(arms_sr, n_fam=24, per_fam=3):
    rows, k = [], 0
    for f in range(n_fam):
        for j in range(per_fam):
            item = f"t-{f:02d}-{j}"
            for arm, fn in arms_sr.items():
                for lang in A.LANGS:
                    rows.append(mk(arm, item, lang, fn(k, lang), f"fam{f:02d}", lvl=("L1", "L2", "L3")[f % 3]))
            k += 1
    return rows


def h1(rows, **kw):
    idx = A.build_index(rows)
    return A.h1_block(idx["S"], idx["fam"], A.invalid_presentations(idx["S"]), B, **kw), idx


# ---- 1. the plan names the validated rule
hp = A.ANALYSIS_PLAN["h1"]
check("ANALYSIS_PLAN h1 names h1-rule-1 = C7, alpha* .0175, guard K >= 20, and says the registered rule is never used for a claim",
      "h1-rule-1 = C7" in hp["decision"] and ".0175" in hp["decision"] and "K >= 20" in hp["decision"] and "never used for a claim" in hp["decision"])
check("H1_PROCEDURE: validated, version h1-rule-1, evidence file exists, V100 sha recorded",
      A.H1_PROCEDURE["validated"] is True and A.H1_PROCEDURE["version"] == T.PROCEDURE_VERSION == "h1-rule-1"
      and os.path.exists(os.path.join(HERE, "..", A.H1_PROCEDURE["evidence"])) and A.H1_PROCEDURE["v100_stage_sha16"] == T.stage_sha("v100")[:16])
src = inspect.getsource(A)
check("static: H1 goes through H1S.h1_rule_c7 and never calls P.h1_decision", "H1S.h1_rule_c7(" in src and "P.h1_decision(" not in src)

# ---- 2. the analyze decision == the deployable rule on the same family totals (exact executable)
rows = grid({"G+M": lambda k, l: 1 if (k * 7 + (l == "fa")) % 5 else 0, "O0": lambda k, l: 1 if (k * 3) % 4 else 0}, n_fam=30, per_fam=3)
res, idx = h1(rows)
p = res["primary"]
d = T.h1_rule_c7(p["_tot"] if "_tot" in p else A.contrast(idx["S"], idx["fam"], "G+M", "O0", set(), B)["_tot"],
                 A.contrast(idx["S"], idx["fam"], "G+M", "O0", set(), B)["_sizes"], p["discordant_families"])
check("analyze primary decision == final_test_h1_stats.h1_rule_c7 on the same totals, sizes and K (estimate, interval, claims, wording)",
      abs(p["decision"]["estimate"] - d["estimate"]) < 1e-15 and p["decision"]["interval"] == d["interval"] and p["decision"]["claims"] == d["claims"]
      and p["decision"]["wording"] == d["wording"], f"{p['decision']} vs {d}")
check("K passed to the rule = analyze.discordant_families (families with >= 1 discordant used presentation)",
      p["decision"]["guard"]["K"] == A.discordant_families(idx["S"], "G+M", "O0", set(), idx["fam"]))
check("primary is labelled validated; family-weighted and infra-paired are labelled sensitivities",
      p["decision"]["role"].startswith("primary") and res["family_weighted"]["decision"]["role"].startswith("sensitivity")
      and res["infra_paired"]["decision"]["role"].startswith("sensitivity"))
check("the nominal level printed is .965", p["decision"]["interval_nominal_level"] == 0.965)

# ---- 3. wordings on constructed data
w = h1(grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0 if k % 4 else 1}))[0]["wording"]
check("large gain over 24 discordant families -> superior", w == "superior", w)
w = h1(grid({"G+M": lambda k, l: 0 if k % 4 else 1, "O0": lambda k, l: 1}))[0]["wording"]
check("large loss -> materially inferior", w.startswith("materially inferior"), w)
w = h1(grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0 if k % 4 else 1}, n_fam=12))[0]["wording"]
check("the same gain with K = 12 < 20 -> inconclusive, guard named", "guard failed (K = 12 < 20" in w, w)
w = h1(grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 1}))[0]["wording"]
check("identical arms (zero variance) -> inconclusive, interval undefined", w.startswith("non-inferiority unresolved") and "undefined" in w, w)
pat = {"G+M": lambda k, l: (1 if k % 3 else 0) if l == "en" else (1 if k % 5 else 0), "O0": lambda k, l: (1 if k % 3 else 0) if l == "fa" else (1 if k % 5 else 0)}
r0 = h1(grid(pat, n_fam=40))[0]
check("balanced differences over 40 families -> a registered wording (NI or inconclusive), never superior or inferior",
      not r0["wording"].startswith(("superior", "materially")), r0["wording"])
for nm in ("primary", "family_weighted", "infra_paired"):
    cl = r0[nm]["decision"]["claims"]
    if cl and cl["sup"] and not cl["ni"]:
        check(f"fixed sequence in {nm}", False)
check("fixed sequence: no view claims superiority without NI", True)

# ---- 4. estimation-only and incomplete paths
rows = grid({"G+M": lambda k, l: 1, "O0": lambda k, l: 0 if k % 4 else 1})
eo = h1(rows, procedure={**A.H1_PROCEDURE, "validated": False})[0]
check("procedure not validated -> estimation only: no claims, the frozen fallback wording, estimate kept",
      eo["wording"] == A.ESTIMATION_ONLY_WORDING and eo["primary"]["decision"]["claims"] is None and eo["primary"]["decision"]["estimate"] > 0.5)
inc = h1(rows, complete=False)[0]
check("incomplete study -> descriptive only, no claims", inc["wording"] == A.INCOMPLETE_WORDING and inc["primary"]["decision"]["claims"] is None)

# ---- 5. end to end + exploratory equivalence + markdown
rows = grid({"G+M": lambda k, l: 1 if k % 4 else 0, "O0": lambda k, l: 1 if k % 3 else 0, "O+M": lambda k, l: 1 if (k + 1) % 4 else 0,
             "G0": lambda k, l: 1 if k % 2 else 0}, n_fam=30)
res = A.analyse(rows, B=B, meta={"namespace": "synthetic", "synthetic": True})
eq = res["equivalence_exploratory"]
ee = T.equiv_e1(*[A.contrast(A.build_index(rows)["S"], A.build_index(rows)["fam"], "G+M", "O+M", set(), B)[k] for k in ("_tot", "_sizes")],
                eq["discordant_families"])
check("exploratory equivalence block == final_test_h1_stats.equiv_e1 (E1), labelled exploratory", eq["available"] and eq["decision"]["exploratory"]
      and eq["decision"]["equivalent"] == ee["equivalent"] and eq["decision"]["interval90"] == ee["interval90"])
md = A.render_md(res)
check("markdown: procedure line, the C7 interval column, the guard column, the equivalence section",
      "Procedure: h1-rule-1" in md and "nominal 96.5% interval (C7)" in md and "K (guard >= 20)" in md and "Exploratory equivalence" in md)
check("markdown no longer cites the wild/percentile rule or the section-9.6 caveat for H1",
      "wild test AND percentile bound" not in md and "section 9.6" not in md)
res2 = A.analyse(rows, B=B, meta={"namespace": "synthetic", "synthetic": True})
check("deterministic: the H1 and equivalence blocks are identical on a re-run", res2["H1"] == res["H1"] and res2["equivalence_exploratory"] == eq)

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
