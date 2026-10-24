"""K23 -- the Kitchens 2/3 replication analysis (prereg A29 item 3; registered text extracted in docs/final-test/k23/A29-analysis-spec.md).
Windows python, offline: no simulator, no LLM, no API.

    python sim/final_test/matrix/k23_analyze.py --allow-test [--partial] [--outdir DIR]
    python sim/final_test/matrix/k23_analyze.py --k2 DIR --k3 DIR --allow-test

Reads ONLY the two replication test stores, test-k2-v1 (FloorPlan15) and test-k3-v1 (FloorPlan17): the sealed score.jsonl and, beside
it, plan.jsonl / practice.jsonl / prep_cost.jsonl / exec.jsonl / energy*.jsonl (cost and execution links). A store directory that is not
its namespace's storage directory, or that is (or lies inside, or resolves to) a Kitchen 1 store, is refused BEFORE any file in it is
opened; both kitchens' paths are checked before either is read. Never opens outcome.sealed.jsonl, states/ or the dataset. Kitchen 1
data are never read and never pooled (A29 item 3).

Per kitchen: the four A29 contrasts G+M - O0, G+M - G0, O+M - O0, G+M - O+M, each the h1-rule-1 item-weighted estimate with the nominal
96.5 % CR2-Satterthwaite interval and the K >= 20 guard (final_test_h1_stats.h1_rule_c7 on the kitchen's family totals); EN / FA and
L1-L3 descriptive rates; the A21 infrastructure / dependence tables and sensitivities; per-kitchen cost. Combined: the equal-weight
two-kitchen summary est = (est_15 + est_17) / 2 with families as clusters across both kitchens (A29 closed form, cross-checked at run
time against a matrix Bell-McCaffrey CR2), the exploratory G+M - O0 criteria (average; each kitchen), the common-item intersection.
No superiority statement; the three other contrasts are estimation-only; no multiplicity adjustment is registered. Kitchen 1's
analysis (analyze.py) is imported, never changed.

Astra K23-ANALYZE-r1: --partial always means estimates and intervals only (no criterion), whatever the seals say; a kitchen without a store
is refused unless docs/final-test/k23/kitchen_not_run_<scene>.json records the decision (A29 item 6), and then only the completed
kitchen's estimates are reported (no pooled result, no replication criteria). This module is the A29 analysis implementation, pinned at
the agent freeze.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, SIM)

import common as C                                                 # noqa: E402
import analyze as A                                                # noqa: E402  (Kitchen 1 analysis: reused by import, never edited)
import final_test_h1_stats as H1S                                  # noqa: E402  (h1-rule-1 = C7: the A24.1-H1 item 2 algebra)
import k23_ns as K                                                 # noqa: E402

#: (scene, test namespace) in the registered order (A29 item 1: FloorPlan15, then FloorPlan17)
KITCHENS = (("FloorPlan15", "test-k2-v1"), ("FloorPlan17", "test-k3-v1"))
SCENES = tuple(s for s, _ in KITCHENS)
NS_OF = dict(KITCHENS)
assert SCENES == tuple(K.REPLICATION_SCENES) and all(C.ns_scene(n) == s and C.NAMESPACES[n].get("replication_test") for s, n in KITCHENS)
#: A29 item 3: the four paired contrasts, D = SR(a) - SR(b); the first is the H1 contrast (the only one with threshold criteria)
CONTRASTS = (("G+M", "O0"), ("G+M", "G0"), ("O+M", "O0"), ("G+M", "O+M"))
NI_CONTRAST = CONTRASTS[0]
PLANNING_ARMS = tuple(C.CORE_ARMS)
DESIGN_ARMS = PLANNING_ARMS + ("P0", "P+M")      # A29 item 1: the four core arms + the reference-plan controls at execution
LANGS = A.LANGS
MARGIN = A.MARGIN
ALPHA_STAR = H1S.ALPHA_STAR                  # .0175 per tail -> t_nu(.9825), the nominal 96.5 % interval
K_GUARD = H1S.H.K_GUARD                      # 20
KITCHEN_WEIGHTS = (0.5, 0.5)                 # equal weight, regardless of kitchen size (A29 items 3 and 6)
CROSS_CHECK_RTOL = 1e-8
OUT_JSON, OUT_MD, OUT_DIRNAME = "analysis_k23.json", "analysis_k23.md", "k23-replication"
INTERVAL_LABEL = "nominal 96.5 % CR2-Satterthwaite interval; actual coverage is not assured"
AVERAGE_SENTENCE = "met the exploratory non-inferiority criterion on average across the two replication kitchens"
EACH_SENTENCE = "met the exploratory non-inferiority criterion in each replication kitchen"
NEVER_SENTENCE = "was shown non-inferior"
assert K_GUARD == 20 and abs(ALPHA_STAR - 0.0175) < 1e-12 and abs(MARGIN - 0.05) < 1e-12


def ckey(a: str, b: str) -> str:
    return f"{a} - {b}"


#: the K2/K3 analysis plan (hashed into every output). Sources: prereg A29 items 1-3 and 6 (lines 1469-1490, 1513), A21 A.4-A.5 / D,
#: section 13, A24.1-H1 item 2; the open questions and their conservative defaults are in docs/final-test/k23/A29-analysis-spec.md.
ANALYSIS_PLAN_K23 = {
    "version": "k23-analysis-2",
    "review": "Astra K23-ANALYZE-r1 (docs/final-test/_astra/K23-ANALYZE-r1.txt): statistical core ACCEPT; changes 4 and 5 applied",
    "partial": ("--partial supplied: estimates and intervals only, no criterion and no threshold statement, even when every seal is complete "
                "(change 4); incomplete seals need --partial"),
    "kitchen_not_run": ("an unexplained missing store is refused; a kitchen without a store is reported as not run only with its recorded decision "
                        "docs/final-test/k23/kitchen_not_run_<scene>.json (schema final-test.k23-kitchen-not-run/1, self-sha; scene, namespace, reason, "
                        "date, decided_by, basis naming A29); then the completed kitchen's estimates only: no pooled result, no replication criteria, "
                        "the study reported incomplete for the replication (A29 item 6, section 13; change 5)"),
    "status": "exploratory, estimation-focused replication (A29 item 3); no confirmatory claim; Kitchen 1 never pooled",
    "kitchens": {s: n for s, n in KITCHENS},
    "contrasts": [ckey(a, b) for a, b in CONTRASTS],
    "per_kitchen": ("h1-rule-1 item-weighted estimate (A24.1-H1 item 2 algebra, final_test_h1_stats.h1_rule_c7): per item the mean of its used paired "
                    "EN / FA differences (a presentation only if both arms have it), families as clusters, CR2 se, Satterthwaite nu, interval "
                    "est -/+ t_nu(.9825) se; guard K >= 20 families with a discordant used presentation, within the kitchen"),
    "pooled": ("est = (est_15 + est_17) / 2 (kitchen fixed, no random kitchen effect); se^2 = sum_f [sum_k a_k c_fk (S_fk - n_fk est_k)]^2, a_k = 1/(2 N_k), "
               "c_fk = (1 - n_fk/N_k)^(-1/2), families as clusters across both kitchens (a shared family is one cluster; a family missing from a kitchen "
               "contributes zero there); nu = (sum_f B_ff)^2 / sum_{f,h} B_fh^2, B_fh = sum_k a_k^2 c_fk c_hk [1(f=h) n_fk - n_fk n_hk / N_k]; t_nu(.9825); "
               "guard K >= 20 counted across both kitchens (distinct families); cross-checked at run time against a matrix Bell-McCaffrey CR2 "
               f"(relative tolerance {CROSS_CHECK_RTOL}); a missing kitchen never yields a one-kitchen summary"),
    "wording": ("G+M - O0 only: the exploratory average criterion (pooled lower bound > -5 pp) and the exploratory each-kitchen criterion (both kitchen "
                "lower bounds > -5 pp; a conjunction), reported separately; the threshold statement 'upper bound < -5 pp' likewise; any threshold "
                "statement needs the guard and a finite positive se with defined df; intervals are reported whenever defined; no superiority "
                f"statement; thesis sentences '{AVERAGE_SENTENCE}' / '{EACH_SENTENCE}', never '{NEVER_SENTENCE}'; the other three contrasts are "
                "estimation-only, their intervals not simultaneous; an unmet criterion is 'not met' for that criterion only, never evidence of no difference"),
    "multiplicity": "none registered for Kitchens 2/3 (no Holm family); intervals are not simultaneous",
    "missing_infra": ("as Kitchen 1: infrastructure failures count as SR 0 in every denominator (A21 A.4); INVALID presentations dropped from every "
                      "arm (section 13); infra-paired and distinct-units-only sensitivities (A21 A.5, D) beside every contrast, the G+M - O0 criteria "
                      "also shown under the infra-paired view; infrastructure-sensitive flag > 5 %; an incomplete study evaluates no criterion"),
    "a36": "a Kitchen 1 amendment: not applied; rows carrying excluded_reset are counted and disclosed",
    "descriptive": "per kitchen and arm (P0 / P+M included): item-weighted SR overall, by language and by level, family-cluster percentile CI",
    "common_items": "when the kitchens' item sets differ: the four contrasts on the common-item intersection, descriptive only",
    "cost": ("per kitchen: A24.2 outcomes (analyze.cost_effectiveness_block, own plan journal; no unmeasured-energy estimate rule; the A24.2 "
             "statement is not made) and the A29 item 4 table (billed USD, tokens, calls per cell, standard-price sensitivity)"),
    "h1_procedure": H1S.PROCEDURE_VERSION,
}


def analysis_plan_sha() -> str:
    return C.sha256_json(ANALYSIS_PLAN_K23)


class K23AnalysisError(A.AnalysisError):
    pass


# ================================================================================================ stores (refusals come first)
def kitchen1_store_dirs(root: str | None = None) -> list:
    """Every storage (and voided) directory of a namespace that is NOT a replication kitchen (= Kitchen 1), under the matrix docs and work
    dirs of `root` AND of this repo (a test root never hides the real Kitchen 1 stores)."""
    out = set()
    for r in {root, None}:
        P = C.paths(r)
        for ns in C.NAMESPACES:
            if C.ns_scene(ns) in K.REPLICATION_SCENES:
                continue
            for name in {C.ns_storage(ns), ns, *C.VOIDED_STORAGE.get(ns, ())}:
                for base in (P["matrix_docs"], P["matrix_work"]):
                    out.add(os.path.join(base, name))
    return sorted(out)


def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def _inside(path: str, base: str) -> bool:
    p, b = _norm(path), _norm(base)
    return p == b or p.startswith(b.rstrip(os.sep) + os.sep)


def check_store_location(store_dir, scene: str, root: str | None = None) -> str:
    """-> the store's namespace, or K23AnalysisError: a replication kitchen, never inside / resolving to a Kitchen 1 store, the namespace's
    own storage directory. Nothing is opened and existence is not required here."""
    if scene not in NS_OF:
        raise K23AnalysisError(f"{scene!r} is not a replication kitchen ({', '.join(SCENES)}); Kitchen 1 is never analysed here")
    ns = NS_OF[scene]
    if not store_dir:
        raise K23AnalysisError(f"{scene}: no store given; both replication kitchens are required (a missing kitchen never becomes a one-kitchen result)")
    for k1 in kitchen1_store_dirs(root):
        if _inside(store_dir, k1):
            raise K23AnalysisError(f"{scene}: {store_dir} is a Kitchen 1 store; Kitchen 1 data are never read (A29 item 3)")
    base = os.path.basename(os.path.normpath(store_dir))
    if base != C.ns_storage(ns) or C.namespace_of_dir(base) != ns:
        raise K23AnalysisError(f"{scene}: store directory {base!r} is not the storage directory of {ns} ({C.ns_storage(ns)!r})")
    return ns


def has_store(store_dir) -> bool:
    return bool(store_dir) and os.path.isfile(os.path.join(store_dir, "score.jsonl"))


def check_store_path(store_dir, scene: str, root: str | None = None) -> str:
    """check_store_location + the store must exist. -> the store's namespace, or K23AnalysisError. Nothing in the directory is opened."""
    ns = check_store_location(store_dir, scene, root)
    if not has_store(store_dir):
        raise K23AnalysisError(f"{scene}: store missing ({store_dir} has no score.jsonl) and no recorded kitchen-not-run decision "
                               f"({os.path.basename(not_run_path(scene, root))}, A29 item 6); both replication kitchens are required "
                               "(a missing kitchen never becomes a one-kitchen result)")
    return ns


# ================================================================================================ A29 item 6: a recorded "kitchen not run" decision
#: Astra K23-ANALYZE-r1 change 5: an unexplained missing store is refused; a kitchen may be reported as NOT RUN only through this explicit,
#: self-sha'd record (A29 item 6: "the remaining replication kitchen(s) are reported as not run (the study is reported incomplete for the
#: replication; section 13)"). With it, the completed kitchen's estimates only: no pooled result, no replication criteria.
NOT_RUN_SCHEMA = "final-test.k23-kitchen-not-run/1"
NOT_RUN_KEYS = ("schema", "scene", "namespace", "reason", "date", "decided_by", "basis", "sha256")
NOT_RUN_MIN_REASON = 20


def not_run_path(scene: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", f"kitchen_not_run_{scene}.json")


def make_not_run_record(scene: str, *, reason: str, date: str, decided_by: str, basis: str = "A29 item 6") -> dict:
    """The record a decision-maker writes (by hand or with this helper) before the analysis; the analysis only reads and validates it."""
    rec = {"schema": NOT_RUN_SCHEMA, "scene": scene, "namespace": NS_OF.get(scene), "reason": reason, "date": date, "decided_by": decided_by,
           "basis": basis}
    rec["sha256"] = C.self_sha(rec)
    return rec


def not_run_problems(rec, scene: str) -> list:
    """[] iff `rec` is a well-formed, unedited kitchen-not-run record for `scene`."""
    import re
    if not isinstance(rec, dict):
        return ["not a JSON object"]
    why = []
    if set(rec) != set(NOT_RUN_KEYS):
        why.append(f"keys {sorted(rec)} != {sorted(NOT_RUN_KEYS)}")
    if rec.get("schema") != NOT_RUN_SCHEMA:
        why.append(f"schema {rec.get('schema')!r} != {NOT_RUN_SCHEMA!r}")
    if rec.get("sha256") != C.self_sha(rec):
        why.append("self-sha mismatch (edited after writing)")
    if rec.get("scene") != scene or rec.get("namespace") != NS_OF.get(scene):
        why.append(f"record is for {rec.get('scene')!r} / {rec.get('namespace')!r}, not {scene} / {NS_OF.get(scene)}")
    if not (isinstance(rec.get("reason"), str) and len(rec["reason"].strip()) >= NOT_RUN_MIN_REASON):
        why.append(f"reason missing or shorter than {NOT_RUN_MIN_REASON} characters")
    if not (isinstance(rec.get("date"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ][0-9:]+)?", rec["date"])):
        why.append("date missing or not ISO (YYYY-MM-DD)")
    if not (isinstance(rec.get("decided_by"), str) and rec["decided_by"].strip()):
        why.append("decided_by missing")
    if not (isinstance(rec.get("basis"), str) and "A29" in rec["basis"]):
        why.append("basis must name A29 (item 6)")
    return why


def load_not_run(scene: str, root: str | None = None):
    """-> the validated record, None when no record file exists; a present but invalid record is refused (never ignored)."""
    p = not_run_path(scene, root)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        raise K23AnalysisError(f"{scene}: {os.path.basename(p)} unreadable ({type(e).__name__})") from None
    why = not_run_problems(rec, scene)
    if why:
        raise K23AnalysisError(f"{scene}: {os.path.basename(p)} is not a valid kitchen-not-run record: " + "; ".join(why))
    return dict(rec, file=os.path.relpath(p, C.paths(root)["root"]).replace(os.sep, "/"))


def _check_rows(rows, scene: str, ns: str, name: str, exact: bool) -> None:
    """A store holds ONE kitchen. exact: every row's namespace must be `ns` (score rows: required; plan / exec rows: when present);
    otherwise (practice / prep rows) a namespace, when present, must be a namespace of the same kitchen. scene / kitchen fields must agree."""
    for n, r in enumerate(rows or (), 1):
        rn = r.get("namespace")
        if exact and (rn != ns if name == "score.jsonl" else (rn is not None and rn != ns)):
            raise K23AnalysisError(f"{scene}: {name} row {n} has namespace {rn!r}, not {ns}: the store mixes kitchens / namespaces")
        if not exact and rn is not None and (rn not in C.NAMESPACES or C.ns_scene(rn) != scene):
            raise K23AnalysisError(f"{scene}: {name} row {n} belongs to namespace {rn!r} ({C.ns_scene(rn) if rn in C.NAMESPACES else '?'}): "
                                   "the store mixes kitchens")
        for fld in ("scene", "kitchen"):
            if r.get(fld) not in (None, scene):
                raise K23AnalysisError(f"{scene}: {name} row {n} says {fld} = {r.get(fld)!r}: the store mixes kitchens")


#: A24.2 accounting for the replication kitchens: the Kitchen 1 numbers, WITHOUT the unmeasured-energy estimate rule (Arash's approved terms
#: bind the 10 Kitchen 1 practice-v2 rows only), so an unmeasured Gemma row stays UNKNOWN. The A24.2 statement itself is not made (spec OQ 8-9).
COST_CFG = dict(A.A24_2_COST, energy_method=dict(A.A24_2_COST["energy_method"], unmeasured_rule_basis=None, unmeasured_rule_terms=None))


def load_store(store_dir, scene: str, *, allow_partial: bool = False, root: str | None = None) -> dict:
    """One kitchen's sealed store -> rows + optional cost / execution journals. Refuses: path (Kitchen 1, wrong directory, missing), seal
    (edited, wrong n, incomplete without allow_partial, another namespace), rows of another namespace or kitchen."""
    ns = check_store_path(store_dir, scene, root)
    score_path = os.path.join(store_dir, "score.jsonl")
    seal = A.verify_seal(score_path, allow_partial)
    if seal.get("namespace") != ns:
        raise K23AnalysisError(f"{scene}: score.seal.json is for namespace {seal.get('namespace')!r}, not {ns}")
    rows = A.read_jsonl(score_path)
    _check_rows(rows, scene, ns, "score.jsonl", exact=True)

    def opt(name):
        p = os.path.join(store_dir, name)
        return A.read_jsonl(p) if os.path.exists(p) else None
    plan_rows, prac_rows, prep_rows, exec_rows = opt("plan.jsonl"), opt("practice.jsonl"), opt("prep_cost.jsonl"), opt("exec.jsonl")
    for name, rr, exact in (("plan.jsonl", plan_rows, True), ("exec.jsonl", exec_rows, True), ("practice.jsonl", prac_rows, False),
                            ("prep_cost.jsonl", prep_rows, False)):
        _check_rows(rr, scene, ns, name, exact)
    meta = {}
    if plan_rows is not None:                                   # C5 strict energy join, as Kitchen 1 (energy*.jsonl beside plan.jsonl)
        plan_rows, meta["energy_attach"] = A.attach_energy(plan_rows, A.load_energy_rows(store_dir),
                                                           A.plan_line_shas(os.path.join(store_dir, "plan.jsonl")), cfg=COST_CFG, strict=True)
    if prac_rows is not None:
        prac_rows, meta["practice_energy_fields_stripped"] = A.strip_energy_fields(prac_rows)
    if prep_rows is not None:
        prep_rows, meta["prep_energy_fields_stripped"] = A.strip_energy_fields(prep_rows)
    return {"scene": scene, "namespace": ns, "dir": store_dir, "seal": seal, "rows": rows, "plan_rows": plan_rows, "practice_rows": prac_rows,
            "prep_rows": prep_rows, "exec_rows": exec_rows, "meta": meta}


def completeness_problems_k23(rows, roster, invalid=frozenset()) -> list:
    """A complete seal covers every core arm x every roster item x EN / FA, minus INVALID presentations (the Kitchen 1 rule of
    analyze.completeness_problems, for the A29 design)."""
    have = set()
    for r in rows:
        p = C.parse_pres(r["pres"])
        if p["world"] == "canon" and r.get("r") == 0 and r.get("run") == 0:
            have.add((r["arm"], p["item"], p["lang"]))
    out = []
    for arm in PLANNING_ARMS:
        miss = [(i, lg) for i in roster for lg in LANGS if (i, lg) not in invalid and (arm, i, lg) not in have]
        if miss:
            out.append(f"{arm}: {len(miss)} of {len(roster) * len(LANGS)} presentations missing (first: {miss[0][0]}/{miss[0][1]})")
    return out


# ================================================================================================ estimators
def contrast_parts(S, fam_of, a, b, drop_invalid, **kw) -> dict:
    """The A24.1-H1 item 2 data of one contrast in one kitchen: item differences (analyze.item_diffs, unchanged), family totals S_f, kept
    items n_f, K = families with a discordant used presentation. Unavailable (with the reason) without 2 families."""
    diffs, info = A.item_diffs(S, a, b, drop_invalid, **kw)
    disc = info.pop("_disc_items")
    base = {"a": a, "b": b, **info}
    if not diffs:
        return dict(base, available=False, reason=f"no presentation of {a} paired with {b}")
    fams_used = sorted({fam_of[i] for i in diffs})
    if len(fams_used) < 2:
        return dict(base, available=False, reason=f"fewer than 2 families ({len(fams_used)})")
    fams, tot, sizes = A.by_family(diffs, fam_of)
    disc_f = sorted({fam_of[i] for i in disc if i in diffs})
    return dict(base, available=True, fams=fams, S=tot, n=sizes, K=len(disc_f), disc_families=disc_f, items=diffs,
                **A.win_tie_loss(fams, tot, sizes))


def _public(parts: dict) -> dict:
    return {k: v for k, v in parts.items() if k != "items"}


def kitchen_cr2(parts: dict) -> dict:
    """h1-rule-1 numbers of one kitchen's contrast (h1_rule_c7 on its family totals): estimate, se, nu, the nominal 96.5 % interval (reported
    whenever defined), the within-kitchen guard. The Kitchen 1 wordings of h1_rule_c7 are NOT used (see criterion())."""
    if not parts.get("available"):
        return _public(parts)
    r = H1S.h1_rule_c7(parts["S"], parts["n"], parts["K"])
    c = H1S.cr2_single(parts["S"], parts["n"], ALPHA_STAR)
    out = dict(_public(parts), procedure=r["procedure"], estimate=r["estimate"], se=r["se"], nu=r["nu"], t_crit=r["t_crit"],
               interval=list(r["interval"]), interval_defined=bool(r["defined"]), interval_nominal_level=r["interval_nominal_level"],
               interval_label=INTERVAL_LABEL, guard=r["guard"], n_families=len(parts["fams"]), n_items=int(sum(parts["n"])))
    if not r["defined"]:
        out["undefined_reason"] = c.get("reason") or "interval undefined"
    return out


def _tcrit(nu):
    try:
        q = float(H1S._tq(ALPHA_STAR, nu))
    except Exception:                                              # noqa: BLE001 -- scipy domain error -> undefined
        return None
    return q if math.isfinite(q) else None


def pooled_cr2(parts_list, weights=None) -> dict:
    """A29 item 3 closed form. parts_list: per kitchen {fams, S, n} (family totals and kept-item counts); weights w_k (default equal), a_k =
    w_k / N_k. Families are clusters across kitchens (union of family ids); a family missing from a kitchen contributes zero there. With one
    kitchen and weight 1 this is exactly A24.1-H1 (cr2_single)."""
    import numpy as np
    w = [1.0 / len(parts_list)] * len(parts_list) if weights is None else [float(x) for x in weights]
    fams = sorted(set().union(*[set(p["fams"]) for p in parts_list]))
    fi = {f: j for j, f in enumerate(fams)}
    F = len(fams)
    out = {"weights": w, "n_clusters": F}
    data = []
    for p in parts_list:
        S, n = np.zeros(F), np.zeros(F)
        for f, s_, n_ in zip(p["fams"], p["S"], p["n"]):
            S[fi[f]] += float(s_)
            n[fi[f]] += float(n_)
        if float(n.sum()) <= 0:
            return dict(out, estimate=None, se=None, nu=None, t_crit=None, interval=[None, None], defined=False, reason="a kitchen has no kept item")
        data.append((S, n, float(n.sum())))
    ests = [float(S.sum() / N) for S, n, N in data]
    est = float(sum(wk * e for wk, e in zip(w, ests)))
    out.update(estimate=est, kitchen_estimates=ests, N_k=[N for _S, _n, N in data])
    g, Bm, scale = np.zeros(F), np.zeros((F, F)), 0.0
    for (S, n, N), e_k, wk in zip(data, ests, w):
        a = wk / N
        lev = 1.0 - n / N
        if np.any(lev[n > 0] <= 1e-15):
            return dict(out, se=None, nu=None, t_crit=None, interval=[None, None], defined=False,
                        reason="a kitchen has a single family (leverage 1): CR2 undefined")
        c = np.where(n > 0, 1.0 / np.sqrt(np.where(n > 0, lev, 1.0)), 0.0)
        g += a * c * (S - n * e_k)
        Bm += (a * a) * np.outer(c, c) * (np.diag(n) - np.outer(n, n) / N)
        scale += a * a * float((S * S).sum() + e_k * e_k * (n * n).sum())
    se2 = float((g * g).sum())
    if not math.isfinite(se2) or se2 <= 1e-12 * max(scale, 1e-300):
        return dict(out, se=0.0, nu=None, t_crit=None, interval=[None, None], defined=False, reason="observed se = 0")
    se = math.sqrt(se2)
    den = float((Bm * Bm).sum())
    nu = float(np.trace(Bm) ** 2 / den) if den > 0 else math.nan
    q = _tcrit(nu) if (math.isfinite(nu) and nu > 0) else None
    if q is None:
        return dict(out, se=se, nu=nu if math.isfinite(nu) else None, t_crit=None, interval=[None, None], defined=False,
                    reason="degrees of freedom undefined")
    return dict(out, se=se, nu=nu, t_crit=q, interval=[est - q * se, est + q * se], defined=True, reason=None)


def cr2_matrix(kitchen_items, weights=None) -> dict:
    """A standard Bell-McCaffrey CR2 of the item-level model y_i = beta_k(i) + e_i (one intercept per kitchen, identity working covariance),
    families as clusters across kitchens, for the contrast w'beta: V = sum_g (u_g'y)^2 with u_g = (I - H)_{g.}' A_g X_g (X'X)^-1 w,
    A_g = (I - H_gg)^(-1/2); Satterthwaite nu = (sum_g u_g'u_g)^2 / sum_{g,h} (u_g'u_h)^2 (Pustejovsky & Tipton 2018 under the working
    model). kitchen_items: per kitchen a list of (family, y). Used as the run-time cross-check of pooled_cr2 (A29 item 3)."""
    import numpy as np
    rows = [(k, f, float(y)) for k, items in enumerate(kitchen_items) for f, y in items]
    nK, n = len(kitchen_items), len(rows)
    w = np.full(nK, 1.0 / nK) if weights is None else np.asarray(weights, float)
    X, y = np.zeros((n, nK)), np.array([r[2] for r in rows])
    for i, (k, _f, _y) in enumerate(rows):
        X[i, k] = 1.0
    if np.any(X.sum(0) == 0):
        return {"defined": False, "reason": "a kitchen has no item"}
    M = np.linalg.inv(X.T @ X)
    beta = M @ X.T @ y
    H = X @ M @ X.T
    IH = np.eye(n) - H
    fams = sorted({r[1] for r in rows})
    U = []
    for f in fams:
        ix = [i for i, r in enumerate(rows) if r[1] == f]
        vals, vecs = np.linalg.eigh(np.eye(len(ix)) - H[np.ix_(ix, ix)])
        if vals.min() <= 1e-12:
            return {"defined": False, "singular": True, "reason": "I - H_gg singular (a kitchen has a single family)", "estimate": float(w @ beta)}
        Ag = vecs @ np.diag(vals ** -0.5) @ vecs.T
        U.append(IH[ix, :].T @ (Ag @ X[ix] @ M @ w))
    U = np.array(U)
    var = float(((U @ y) ** 2).sum())
    G = U @ U.T
    den = float((G * G).sum())
    return {"defined": var > 0 and den > 0, "singular": False, "estimate": float(w @ beta), "se": math.sqrt(max(var, 0.0)),
            "nu": float(np.trace(G) ** 2 / den) if den > 0 else None, "n_clusters": len(fams)}


def _close(x, y, rtol=CROSS_CHECK_RTOL) -> bool:
    return x is not None and y is not None and abs(float(x) - float(y)) <= rtol * max(1.0, abs(float(x)), abs(float(y)))


def cross_check_agrees(r: dict, m: dict) -> bool:
    """Closed form r vs matrix CR2 m: estimates equal; defined -> se and nu equal; 'single family' -> the matrix is singular too;
    'se = 0' -> the matrix se is zero up to rounding (1e-9; item differences lie in [-1, 1]); 'df undefined' -> se equal."""
    if not _close(r.get("estimate"), m.get("estimate")):
        return False
    if r.get("defined"):
        return bool(m.get("defined")) and _close(r["se"], m["se"]) and _close(r["nu"], m["nu"])
    why = str(r.get("reason") or "")
    if "single family" in why:
        return bool(m.get("singular"))
    if "se = 0" in why:
        return not m.get("singular") and m.get("se") is not None and m["se"] <= 1e-9
    return not m.get("singular") and _close(r.get("se"), m.get("se"))


def pooled_result(parts_by_scene: dict, fam_of: dict, label: str = "") -> dict:
    """The equal-weight two-kitchen summary of one contrast, with the cross-kitchen guard and the run-time matrix-CR2 cross-check. Any
    kitchen unavailable -> the summary is unavailable (never a one-kitchen result)."""
    miss = [f"{s}: {parts_by_scene[s].get('reason')}" for s in SCENES if not (parts_by_scene.get(s) or {}).get("available")]
    if miss:
        return {"available": False, "reason": "kitchen result unavailable (" + "; ".join(miss) + "); the two-kitchen summary is never a one-kitchen result"}
    pl = [parts_by_scene[s] for s in SCENES]
    r = pooled_cr2(pl, KITCHEN_WEIGHTS)
    m = cr2_matrix([[(fam_of[i], d) for i, d in sorted(p["items"].items())] for p in pl], KITCHEN_WEIGHTS)
    if not cross_check_agrees(r, m):
        raise K23AnalysisError(f"pooled CR2 {label}: the A29 closed form and the matrix CR2 disagree "
                               f"(closed {r.get('estimate')}, {r.get('se')}, {r.get('nu')} vs matrix {m.get('estimate')}, {m.get('se')}, {m.get('nu')})")
    disc_union = sorted(set().union(*[set(p["disc_families"]) for p in pl]))
    k_union = len(disc_union)
    shared = sorted(set(pl[0]["fams"]) & set(pl[1]["fams"]))
    return dict(r, available=True, interval_defined=bool(r["defined"]), interval_nominal_level=round(1 - 2 * ALPHA_STAR, 6),
                interval_label=INTERVAL_LABEL,
                guard={"K": k_union, "threshold": K_GUARD, "passed": k_union >= K_GUARD,
                       "counted": "distinct families (one cluster each) with a discordant used presentation in either kitchen",
                       "K_per_kitchen": {s: parts_by_scene[s]["K"] for s in SCENES},
                       "K_sum_of_kitchens": sum(parts_by_scene[s]["K"] for s in SCENES)},
                shared_families=len(shared), n_items={s: int(sum(parts_by_scene[s]["n"])) for s in SCENES},
                n_families={s: len(parts_by_scene[s]["fams"]) for s in SCENES},
                cross_check={"method": "matrix Bell-McCaffrey CR2 (cr2_matrix)", "agree": True, "rtol": CROSS_CHECK_RTOL,
                             "matrix": {k: m.get(k) for k in ("estimate", "se", "nu", "defined")}})


DESCRIPTIVE_PARTIAL = "--partial supplied: estimates and intervals only, no criterion (Astra K23-ANALYZE-r1 change 4)"
DESCRIPTIVE_INCOMPLETE = "incomplete study: estimates and intervals only, no criterion (prereg s13)"


def criterion(res: dict, descriptive_reason, scope: str) -> dict:
    """The exploratory G+M - O0 non-inferiority criterion (lower bound > -5 pp) for one scope (a kitchen, or the two-kitchen summary), plus
    the threshold statement 'upper bound < -5 pp'. Any threshold statement needs the guard and a finite positive se with defined df; else
    the criterion is 'not met' with no threshold statement. Never a superiority statement. descriptive_reason (an incomplete study or
    --partial): no criterion at all ('not evaluated')."""
    out = {"scope": scope, "criterion": "lower bound > -5 pp", "exploratory": True}
    if descriptive_reason:
        return dict(out, met=None, status="not evaluated", threshold_statement_permitted=False, upper_bound_below_minus_5pp=None,
                    reason=descriptive_reason)
    if not res.get("available"):
        return dict(out, met=False, status="not met", threshold_statement_permitted=False, upper_bound_below_minus_5pp=None,
                    reason=f"no threshold statement: result unavailable ({res.get('reason')})")
    se = res.get("se")
    if not res.get("interval_defined") or not (se is not None and math.isfinite(se) and se > 0):
        return dict(out, met=False, status="not met", threshold_statement_permitted=False, upper_bound_below_minus_5pp=None,
                    reason=f"no threshold statement: interval undefined ({res.get('undefined_reason') or res.get('reason') or 'se / df'})")
    g = res["guard"]
    if not g["passed"]:
        return dict(out, met=False, status="not met", threshold_statement_permitted=False, upper_bound_below_minus_5pp=None,
                    reason=f"no threshold statement: guard failed (K = {g['K']} < {K_GUARD} families with a discordant presentation)")
    lo, hi = res["interval"]
    met = lo > -MARGIN
    return dict(out, met=bool(met), status="met" if met else "not met", threshold_statement_permitted=True,
                upper_bound_below_minus_5pp=bool(hi < -MARGIN), lower_bound=lo, upper_bound=hi,
                reason=("lower bound > -5 pp" if met else "lower bound <= -5 pp (not met for this criterion only; not evidence of no difference)"))


def ni_criteria(per_kitchen: dict, pooled: dict, descriptive_reason) -> dict:
    """A29 item 3 wording for G+M - O0: the average criterion and the each-kitchen criterion (a conjunction), reported separately."""
    pk = {s: criterion(per_kitchen[s], descriptive_reason, s) for s in SCENES}
    avg = criterion(pooled, descriptive_reason, "two-kitchen average")
    if descriptive_reason:
        each = {"met": None, "status": "not evaluated", "reason": descriptive_reason}
    else:
        met = all(pk[s]["met"] is True for s in SCENES)
        each = {"met": met, "status": "met" if met else "not met",
                "reason": "both kitchen lower bounds > -5 pp with the guard" if met else
                "; ".join(f"{s}: {pk[s]['reason']}" for s in SCENES if pk[s]["met"] is not True)}
    each["criterion"] = "both kitchen-specific lower bounds > -5 pp (a conjunction)"
    sent = []
    if avg.get("met"):
        sent.append(f"G+M {AVERAGE_SENTENCE} (relative to O0).")
    elif avg.get("status") == "not met":
        sent.append("The exploratory average non-inferiority criterion was not met (not evidence of no difference).")
    if each.get("met"):
        sent.append(f"G+M {EACH_SENTENCE} (relative to O0).")
    elif each.get("status") == "not met":
        sent.append("The exploratory each-kitchen non-inferiority criterion was not met (not evidence of no difference).")
    return {"contrast": ckey(*NI_CONTRAST), "average": avg, "each_kitchen": dict(each, per_kitchen=pk), "thesis_sentences": sent,
            "never_say": NEVER_SENTENCE, "superiority": "no superiority statement is made from Kitchens 2/3 (A29 item 3)",
            "validation": "neither criterion is a simulation-validated confirmatory conclusion; power and coverage here are unknown (A29 item 3)"}


# ================================================================================================ kitchen + combined blocks
def a29_cost_table(plan_rows, arms) -> dict:
    """A29 item 4 (Astra A29 9): per arm billed USD, tokens in / out, calls per cell and the standard-price sensitivity (each recorded
    batch charge doubled; every other charge keeps its billed rate; a charge counts as batch only when the row or the attempt records
    billing_tier 'batch')."""
    if plan_rows is None:
        return {"available": False, "reason": "no plan journal beside score.jsonl"}

    def batch(x):
        return (x or {}).get("billing_tier") == "batch"

    out = {}
    for arm in arms:
        rs = [r for r in plan_rows if r.get("arm") == arm]
        billed = std = 0.0
        nb = 0
        for r in rs:
            cost = float(r.get("cost_usd") or 0.0)
            atts = [a for a in (r.get("attempts") or ()) if isinstance(a, dict)]
            att_cost = sum(float(a.get("cost_usd") or 0.0) for a in atts)
            rb = batch(r)
            s = sum(float(a.get("cost_usd") or 0.0) * (2.0 if (rb or batch(a.get("extra"))) else 1.0) for a in atts)
            s += max(0.0, cost - att_cost) * (2.0 if rb else 1.0)
            nb += int(rb or any(batch(a.get("extra")) for a in atts))
            billed += cost
            std += s
        calls = sum(int(r.get("calls") or 0) for r in rs)
        out[arm] = {"kind": A.arm_kind(arm), "plan_rows": len(rs), "billed_usd": billed, "standard_price_usd": std, "rows_with_batch_tier": nb,
                    "tokens_in": sum(int(r.get("tokens_in") or 0) for r in rs), "tokens_out": sum(int(r.get("tokens_out") or 0) for r in rs),
                    "calls": calls, "cached_calls": sum(int(r.get("cached_calls") or 0) for r in rs),
                    "calls_per_cell": (calls / len(rs)) if rs else None}
        if A.arm_kind(arm) == "gemma":
            out[arm]["note"] = "0 USD marginal API charge (local Gemma); electricity and hardware are in the A24.2 block"
    return {"available": True, "by_arm": out,
            "rule": "standard-price sensitivity: batch charges doubled, synchronous / historical charges unchanged (A29 item 4)"}


def freeze_cost_record(scene: str, root: str | None = None) -> dict:
    """The kitchen's own freeze (docs/final-test/k23/freeze_<scene>.json) a24_2_cost record, checked against COST_CFG; never Kitchen 1's."""
    p = K.freeze_path(scene, root)
    if not os.path.exists(p):
        return {"recorded": False, "note": f"no {os.path.basename(p)}"}
    try:
        with open(p, encoding="utf-8") as f:
            fz = json.load(f)
        return A.check_cost_freeze(fz, COST_CFG)
    except (OSError, ValueError, A.AnalysisError) as e:
        return {"recorded": False, "error": f"{type(e).__name__}: {e}"}


def kitchen_cost(st, idx, arms, drop_invalid, complete, B_cost, freeze_rec) -> dict:
    S, fam = idx["S"], idx["fam"]
    prep = ([dict(r, cost_kind="gemma", cost_source="practice") for r in (st["practice_rows"] or [])]
            + [dict(r, cost_source="preparation") for r in (st["prep_rows"] or [])])
    ce = A.cost_effectiveness_block(S, fam, arms, drop_invalid, st["plan_rows"], prep, cfg=COST_CFG, h1=None, complete=complete, B=B_cost,
                                    freeze_required=True, freeze_recorded=bool((freeze_rec or {}).get("recorded")))
    evaluated = ce.get("claim")
    ce["claim"] = {"statement_permitted": False, "reading_rule_evaluated_descriptively": evaluated,
                   "reason": ("A29 item 3 reports the A24.2 outcomes per kitchen; the A24.2 statement is not registered for the replication "
                              "kitchens (spec open question 8)")}
    ce["energy_attach"] = st["meta"].get("energy_attach")
    ce["freeze_a24_2_cost"] = freeze_rec
    ce["unmeasured_rule"] = "not applied: the approved estimate binds the 10 Kitchen 1 practice-v2 rows only; unmeasured Gemma energy stays UNKNOWN"
    return {"api_spend_per_success": A.cost_block(arms, S, st["plan_rows"], st["practice_rows"]),
            "a29_item4": a29_cost_table(st["plan_rows"], arms), "a24_2": ce}


def dep_row(parts: dict) -> dict:
    """A21 D per-contrast reporting (the fields of analyze._dep_row), from the contrast's own item_diffs counts: also when the contrast has
    fewer than 2 families (analyze._dep_row would raise there)."""
    n, sh = parts.get("n_pres", 0), parts.get("n_shared", 0)
    return {"a": parts["a"], "b": parts["b"], "presentations": n, "shared_unit_presentations": sh, "discordant": parts.get("n_discordant"),
            "discordance": parts.get("discordance"), "distinct_unit_presentations": n - sh,
            "note": "a presentation in which both arms point at the same unit has difference 0 by construction (one realisation, not agreement of two runs)"}


def _views(S, fam, a, b, drop_invalid, items=None) -> dict:
    """primary + the A21 sensitivities of one contrast in one kitchen, as contrast parts."""
    kw = {} if items is None else {"items": items}
    return {"primary": contrast_parts(S, fam, a, b, drop_invalid, **kw),
            "infra_paired": contrast_parts(S, fam, a, b, drop_invalid, drop_infra=True, **kw),
            "distinct_units_only": contrast_parts(S, fam, a, b, drop_invalid, drop_shared=True, **kw)}


def kitchen_block(st, idx, complete, B_cost, freeze_rec):
    S, fam, lvl = idx["S"], idx["fam"], idx["level"]
    arms = sorted({a for (a, i, l) in S})
    drop_invalid = A.invalid_presentations(S)
    parts = {ckey(a, b): _views(S, fam, a, b, drop_invalid) for a, b in CONTRASTS}
    lp = A.execution_link_problems(st["rows"], st["exec_rows"]) if st["exec_rows"] else None
    if lp and st["seal"].get("complete"):                # integrity: a complete seal must link every row (also under --partial)
        raise K23AnalysisError(f"{st['scene']}: {len(lp)} score row(s) are not linked to their execution (first: {lp[0]})")
    a36 = {}
    for (a, i, l), r in S.items():
        if r.get("excluded_reset") is True:
            a36[a] = a36.get(a, 0) + 1
    planning = [a for a in PLANNING_ARMS if a in arms]
    blk = {
        "scene": st["scene"], "namespace": st["namespace"], "n_score_rows": len(st["rows"]), "seal_sha256": st["seal"].get("sha256"),
        "complete": bool(st["seal"].get("complete")), "arms": arms, "core_arms_missing": [a for a in PLANNING_ARMS if a not in arms],
        "items_per_arm": {a: len({i for (x, i, l) in S if x == a}) for a in arms},
        "n_items": len({i for (x, i, l) in S}), "n_families": len({fam[i] for (x, i, l) in S}),
        "levels": {L: len({i for (x, i, l) in S if lvl.get(i) == L}) for L in sorted({v for v in lvl.values() if v})},
        "invalid_presentations_dropped": len(drop_invalid), "unparsable_rows": len(idx["bad"]),
        "contrasts": {k: {v: kitchen_cr2(p) for v, p in vs.items()} for k, vs in parts.items()},
        "dependence": {"by_arm": A.dependence_block(S, arms), "contrasts": {k: dep_row(vs["primary"]) for k, vs in parts.items()}},
        "arm_tables": A.per_arm_tables(idx, arms, drop_invalid),
        "infra_by_arm": A.infra_table(S, arms),
        "execution_time": A.execution_block(st["rows"], st["exec_rows"]),
        "execution_links": {"checked": None if lp is None else len(st["rows"]), "problems": None if lp is None else len(lp), "first": (lp or [])[:10]},
        "a36_disclosure": {"excluded_reset_cells_by_arm": a36, "applied": False,
                           "note": "A36 is a Kitchen 1 amendment; not applied here (A21: infra = SR 0); spec open question 6"},
        "cost": kitchen_cost(st, idx, planning, drop_invalid, complete, B_cost, freeze_rec),
    }
    return blk, parts, drop_invalid


def analyse_k23(stores: dict, *, fam_of_manifest=None, B_cost=20000, meta=None, freeze_records=None, partial: bool = False,
                not_run=None) -> dict:
    """stores: {scene: load_store(...)}; normally BOTH replication kitchens. -> the combined result (per kitchen + two-kitchen summary).
    partial (--partial): estimates and intervals only, no criterion, whatever the seals say (Astra K23-ANALYZE-r1 change 4).
    not_run: {scene: validated kitchen-not-run record} (A29 item 6, change 5): that kitchen has no store; the completed kitchen's estimates
    only, no pooled result, no replication criteria. Without a record a missing kitchen is refused."""
    stores, not_run = dict(stores or {}), dict(not_run or {})
    for s, rec in not_run.items():
        why = not_run_problems({k: v for k, v in (rec or {}).items() if k != "file"}, s) if s in NS_OF else [f"{s!r} is not a replication kitchen"]
        if why:
            raise K23AnalysisError(f"{s}: invalid kitchen-not-run record: " + "; ".join(why))
    both = sorted(set(stores) & set(not_run))
    if both:
        raise K23AnalysisError(f"{both}: a kitchen with a store cannot also be recorded as not run")
    missing = [s for s in SCENES if s not in stores and s not in not_run]
    extra = sorted((set(stores) | set(not_run)) - set(SCENES))
    if missing or extra:
        raise K23AnalysisError(f"both replication kitchens are required and nothing else: missing {missing}, other {extra} "
                               "(a missing kitchen never becomes a one-kitchen result without a recorded kitchen-not-run decision, "
                               "A29 item 6; Kitchen 1 is never pooled)")
    run_scenes = tuple(s for s in SCENES if s in stores)
    if not run_scenes:
        raise K23AnalysisError("no replication kitchen was run (both recorded as not run): nothing to analyse")
    dirs = [_norm(stores[s]["dir"]) for s in run_scenes if stores[s].get("dir")]
    if len(set(dirs)) != len(dirs):
        raise K23AnalysisError("the two kitchens point at the same store")
    for s in run_scenes:
        if stores[s].get("scene") != s or stores[s].get("namespace") != NS_OF[s]:
            raise K23AnalysisError(f"store given for {s} is {stores[s].get('scene')} / {stores[s].get('namespace')}")
        _check_rows(stores[s]["rows"], s, NS_OF[s], "score.jsonl", exact=True)
    idx = {s: A.build_index(stores[s]["rows"], fam_of_manifest) for s in run_scenes}
    for s in run_scenes:
        if not idx[s]["S"]:
            raise K23AnalysisError(f"{s}: no scored cells (canon world, r = 0, run = 0)")
        alien = sorted({a for (a, i, l) in idx[s]["S"]} - set(DESIGN_ARMS))
        if alien:
            raise K23AnalysisError(f"{s}: arm(s) {alien} are not in the K2/K3 design ({', '.join(DESIGN_ARMS)}; A29 item 1)")
    fam_of = {}
    for s in run_scenes:
        for i, f in idx[s]["fam"].items():
            if fam_of.setdefault(i, f) != f:
                raise K23AnalysisError(f"item {i} has family {fam_of[i]} in one kitchen and {f} in the other")
    seals_complete = all(bool(stores[s]["seal"].get("complete")) for s in run_scenes) and bool((meta or {}).get("complete", True))
    reasons = ([DESCRIPTIVE_PARTIAL] if partial else []) + ([] if seals_complete else [DESCRIPTIVE_INCOMPLETE])
    descriptive = "; ".join(reasons) or None
    freeze_records = freeze_records or {}
    kit, parts, drops = {}, {}, {}
    for s in run_scenes:
        kit[s], parts[s], drops[s] = kitchen_block(stores[s], idx[s], descriptive is None, B_cost, freeze_records.get(s))
    head = {"schema": "ftm.analysis-k23/1", "spec": C.SPEC_VERSION, "meta": meta or {},
            "complete": seals_complete and not not_run, "criteria_evaluated": descriptive is None and not not_run,
            "descriptive_only": {"active": descriptive is not None or bool(not_run),
                                 "reasons": reasons + ([f"kitchen not run: {', '.join(sorted(not_run))} (A29 item 6)"] if not_run else [])},
            "partial_flag": bool(partial), "kitchens_not_run": not_run,
            "analysis_plan_k23": ANALYSIS_PLAN_K23, "analysis_plan_k23_sha": analysis_plan_sha(), "kitchens": kit}
    if not_run:
        recs = "; ".join(f"{s}: {r.get('reason')} (decided by {r.get('decided_by')}, {r.get('date')}, {r.get('file', 'record')})"
                         for s, r in sorted(not_run.items()))
        return dict(head, combined={"available": False, "pooled": None, "g_m_vs_o0_criteria": None, "common_items": None,
                                    "reason": (f"kitchen not run ({recs}): the completed kitchen's estimates only; no two-kitchen summary and no "
                                               "replication criteria (A29 item 6; Astra K23-ANALYZE-r1 change 5); the study is reported incomplete "
                                               "for the replication (section 13)"),
                                    "task_composition": {s: {"n_items": kit[s]["n_items"], "levels": kit[s]["levels"],
                                                             "n_families": kit[s]["n_families"]} for s in run_scenes}},
                    disclosures=disclosures())
    pooled = {}
    for a, b in CONTRASTS:
        k = ckey(a, b)
        pooled[k] = {v: pooled_result({s: parts[s][k][v] for s in SCENES}, fam_of, f"{k} / {v}") for v in ("primary", "infra_paired", "distinct_units_only")}
    nk = ckey(*NI_CONTRAST)
    crit = {"primary": ni_criteria({s: kit[s]["contrasts"][nk]["primary"] for s in SCENES}, pooled[nk]["primary"], descriptive),
            "infra_paired_sensitivity": ni_criteria({s: kit[s]["contrasts"][nk]["infra_paired"] for s in SCENES}, pooled[nk]["infra_paired"],
                                                    descriptive)}
    crit["infra_paired_sensitivity"]["role"] = "sensitivity (A21 A.5: shown side by side; the primary view governs)"
    crit["infrastructure_sensitive_arms"] = {s: sorted(a for a, v in kit[s]["infra_by_arm"].items() if v["infrastructure_sensitive"]) for s in SCENES}
    crit["infra_wordings_agree"] = (crit["primary"]["average"]["status"] == crit["infra_paired_sensitivity"]["average"]["status"]
                                    and crit["primary"]["each_kitchen"]["status"] == crit["infra_paired_sensitivity"]["each_kitchen"]["status"])
    return dict(head, combined={"available": True, "contrasts": pooled, "g_m_vs_o0_criteria": crit,
                                "common_items": common_items_block(idx, drops, fam_of),
                                "task_composition": {s: {"n_items": kit[s]["n_items"], "levels": kit[s]["levels"],
                                                         "n_families": kit[s]["n_families"]} for s in SCENES}},
                disclosures=disclosures())


def common_items_block(idx, drops, fam_of) -> dict:
    """A29 item 3: when applicability differs between the kitchens, the four contrasts on the common-item intersection, DESCRIPTIVE only
    (never replaces the primary samples; no criterion)."""
    sets = {s: {i for (a, i, l) in idx[s]["S"] if a in PLANNING_ARMS} for s in SCENES}
    common = sorted(sets[SCENES[0]] & sets[SCENES[1]])
    out = {"n_items": {s: len(sets[s]) for s in SCENES}, "n_common": len(common), "identical": sets[SCENES[0]] == sets[SCENES[1]],
           "role": "descriptive only; never replaces the primary samples (A29 item 3)"}
    if out["identical"]:
        out["note"] = "the kitchens have the same item set: the intersection equals the primary samples"
        return out
    out["contrasts"] = {}
    for a, b in CONTRASTS:
        ps = {s: contrast_parts(idx[s]["S"], idx[s]["fam"], a, b, drops[s], items=common) for s in SCENES}
        out["contrasts"][ckey(a, b)] = {"per_kitchen": {s: kitchen_cr2(ps[s]) for s in SCENES},
                                        "two_kitchen_average": pooled_result(ps, fam_of, f"common {ckey(a, b)}")}
    return out


def disclosures() -> list:
    return ["Kitchens 2/3 are an estimation-focused, exploratory replication (A29 item 3): power and coverage for these samples and the pooled "
            "estimator are unknown; >= 80 % power has not been demonstrated; an unmet criterion is never evidence of no difference.",
            "h1-rule-1 was validated on Kitchen 1's family sizes only (A24.1-H1 item 3); every interval is a nominal 96.5 % CR2-Satterthwaite "
            "interval whose actual coverage is not assured.",
            "Transfer of the registered task set to new kitchens, not independent task replication; claims cover applicable, reference-solvable "
            "tasks (A29 item 2).",
            "Kitchen 1 results are never pooled with Kitchens 2/3. The two kitchens are weighted equally regardless of size; their task "
            "compositions are disclosed.",
            "Estimates are unweighted item averages for the selected task sets, not sampling-weighted estimates for the eligible pool.",
            "Spend-adaptive sizing (A29 item 6) used accounting counters only; outcome-independent sampling and unconditional interval coverage "
            "are not claimed. With a fall-back size, the language and level analyses are descriptive only."]


# ================================================================================================ markdown
def _iv(r) -> str:
    iv = (r or {}).get("interval") or [None, None]
    return "undefined" if iv[0] is None else f"[{A.pct(iv[0])}, {A.pct(iv[1])}]"


def _g(r) -> str:
    g = (r or {}).get("guard") or {}
    return "" if not g else f"{g.get('K')} ({'passed' if g.get('passed') else 'failed'})"


def render_md(res) -> str:
    L = []
    m = res.get("meta") or {}
    syn = m.get("synthetic")
    ran = [s for s in SCENES if s in (res.get("kitchens") or {})]
    comb = res.get("combined") or {}
    pooled_on = bool(comb.get("available"))
    L += ["# Kitchens 2/3 replication analysis (A29 item 3)" + (" -- SYNTHETIC, not evidence" if syn else ""), "",
          "Exploratory, estimation-focused replication. Kitchen 1 is never pooled. No superiority statement; G+M - O0 has two exploratory "
          "criteria, the other three contrasts are estimation-only and their intervals are not simultaneous. Intervals: " + INTERVAL_LABEL + ".", ""]
    for s, r in sorted((res.get("kitchens_not_run") or {}).items()):
        L.append(f"**Kitchen not run: {s}** (`{NS_OF.get(s)}`). Recorded decision {r.get('file', 'record')} (sha {str(r.get('sha256'))[:16]}): "
                 f"{r.get('reason')} -- decided by {r.get('decided_by')}, {r.get('date')}, basis {r.get('basis')}. Only "
                 f"{', '.join(ran)} estimates are reported: NO two-kitchen summary and NO replication criteria (A29 item 6); the study is "
                 "reported incomplete for the replication (section 13).")
        L.append("")
    for s in ran:
        k = res["kitchens"][s]
        L.append(f"- {s} (`{k['namespace']}`): {k['n_score_rows']} score rows, {k['n_items']} items, {k['n_families']} families, levels {k['levels']}, "
                 f"INVALID dropped {k['invalid_presentations_dropped']}, seal {str(k['seal_sha256'] or 'n/a')[:16]}, complete {k['complete']}"
                 + (f", **core arms missing: {', '.join(k['core_arms_missing'])}**" if k["core_arms_missing"] else "") + ".")
    d = res.get("descriptive_only") or {}
    if d.get("active"):
        L.append("\n**Descriptive only: estimates and intervals; no criterion and no threshold statement** (" + "; ".join(d.get("reasons") or []) + ").")
    L.append(f"\nAnalysis plan {res['analysis_plan_k23']['version']} sha {res['analysis_plan_k23_sha'][:16]}.")
    L += ["", "## The four contrasts (primary view)", "",
          "| contrast | scope | estimate | nominal 96.5% interval | se | nu | K (guard >= 20) | items | families |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b in CONTRASTS:
        ck = ckey(a, b)
        for s in ran:
            r = res["kitchens"][s]["contrasts"][ck]["primary"]
            if not r.get("available"):
                L.append(f"| {ck} | {s} | not available ({r.get('reason')}) | | | | | | |")
                continue
            L.append(f"| {ck} | {s} | {A.pct(r['estimate'])} | {_iv(r)} | {r['se']:.4f} | {'n/a' if r['nu'] is None else format(r['nu'], '.2f')} | "
                     f"{_g(r)} | {r['n_items']} | {r['n_families']} |")
        if not pooled_on:
            continue
        p = comb["contrasts"][ck]["primary"]
        if not p.get("available"):
            L.append(f"| {ck} | two-kitchen average | not available ({p.get('reason')}) | | | | | | |")
        else:
            L.append(f"| {ck} | **two-kitchen average** | {A.pct(p['estimate'])} | {_iv(p)} | {(p['se'] or 0):.4f} | "
                     f"{'n/a' if p['nu'] is None else format(p['nu'], '.2f')} | {_g(p)} | {sum(p['n_items'].values())} | "
                     f"{p['n_clusters']} clusters ({p['shared_families']} shared) |")
    if not pooled_on:
        L += ["", "## Two-kitchen summary and G+M - O0 criteria", "", f"Not reported: {comb.get('reason')}."]
    else:
        cr = comb["g_m_vs_o0_criteria"]
        for view, title in (("primary", "primary"), ("infra_paired_sensitivity", "infra-paired sensitivity (A21 A.5)")):
            c = cr[view]
            L += ["", f"## G+M - O0 exploratory criteria: {title}", "",
                  f"- Average (pooled lower bound > -5 pp): **{c['average']['status']}** ({c['average']['reason']}).",
                  f"- Each kitchen (both lower bounds > -5 pp, a conjunction): **{c['each_kitchen']['status']}** ({c['each_kitchen']['reason']})."]
            for s in SCENES:
                q = c["each_kitchen"]["per_kitchen"][s]
                L.append(f"  - {s}: {q['status']} ({q['reason']})")
            ub = {sc: (c["each_kitchen"]["per_kitchen"][sc].get("upper_bound_below_minus_5pp")) for sc in SCENES}
            L.append(f"- Threshold statement 'upper bound < -5 pp' (exploratory): average {c['average'].get('upper_bound_below_minus_5pp')}; "
                     + "; ".join(f"{sc} {v}" for sc, v in ub.items()) + " (None = no threshold statement permitted).")
            if c["thesis_sentences"]:
                L.append("- Thesis wording: " + " ".join(c["thesis_sentences"]) + f" Never: \"{c['never_say']}\".")
        L.append(f"\nInfrastructure-sensitive arms (> 5 % infra cells): {cr['infrastructure_sensitive_arms']}; primary and infra-paired wordings agree: "
                 f"{cr['infra_wordings_agree']}.")
    L += ["", "## Sensitivities (estimates; same CR2 algebra; no threshold statement)", "",
          "| contrast | scope | infra-paired | items (dropped pres) | distinct units only | items |", "|---|---|---|---|---|---|"]

    def est(r):
        return f"{A.pct(r['estimate'])} {_iv(r)}" if r.get("available") else "n/a"

    def nit(r, dropped=False):
        if not r.get("available"):
            return ""
        n = r["n_items"] if isinstance(r["n_items"], int) else sum(r["n_items"].values())
        return f"{n}" + (f" ({r.get('dropped_infra')})" if dropped and "dropped_infra" in r else "")
    for a, b in CONTRASTS:
        ck = ckey(a, b)
        for s in tuple(ran) + (("two-kitchen average",) if pooled_on else ()):
            src = comb["contrasts"][ck] if s not in SCENES else res["kitchens"][s]["contrasts"][ck]
            ip, du = src["infra_paired"], src["distinct_units_only"]
            L.append(f"| {ck} | {s} | {est(ip)} | {nit(ip, True)} | {est(du)} | {nit(du)} |")
    ci = comb.get("common_items")
    if ci:
        L += ["", "## Common-item intersection (descriptive only)", "",
              f"Items: {ci['n_items']}; common {ci['n_common']}; identical sets: {ci['identical']}. {ci.get('note', '')}"]
        if ci.get("contrasts"):
            L += ["", "| contrast | FloorPlan15 | FloorPlan17 | two-kitchen average |", "|---|---|---|---|"]
            for ck, v in ci["contrasts"].items():
                cells = [est(r) for r in [v["per_kitchen"][s] for s in SCENES] + [v["two_kitchen_average"]]]
                L.append(f"| {ck} | " + " | ".join(cells) + " |")

    def cell(x):
        if not x:
            return "n/a"
        return A.rate(x["sr"]) + (f" [{A.rate(x['ci95'][0])}, {A.rate(x['ci95'][1])}]" if x.get("ci95") else "")
    for s in ran:
        k = res["kitchens"][s]
        L += ["", f"## {s}: SR per arm, language and level (descriptive, family-cluster CI)", "",
              "| arm | overall | EN | FA | L1 | L2 | L3 |", "|---|---|---|---|---|---|---|"]
        for arm, t in k["arm_tables"].items():
            L.append(f"| {arm} | {cell(t['overall'])} | {cell(t['lang'].get('en'))} | {cell(t['lang'].get('fa'))} | {cell(t['level'].get('L1'))} | "
                     f"{cell(t['level'].get('L2'))} | {cell(t['level'].get('L3'))} |")
        L += ["", "| arm | cells | planner infra | execution infra | total rate | flag | invalid | distinct units | shared share | A36 flags (not applied) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm, v in k["infra_by_arm"].items():
            dd = k["dependence"]["by_arm"].get(arm) or {}
            L.append(f"| {arm} | {v['n']} | {v['infra']} | {v['exec_infra']} | {A.rate(v['infra_total_rate'])} | "
                     f"{'infrastructure-sensitive' if v['infrastructure_sensitive'] else ''} | {v['invalid']} | {dd.get('distinct_units')} | "
                     f"{A.rate(dd.get('shared_share'))} | {k['a36_disclosure']['excluded_reset_cells_by_arm'].get(arm, 0)} |")
        t4 = k["cost"]["a29_item4"]
        if t4.get("available"):
            L += ["", f"Cost ({s}, A29 item 4; planning only):", "", "| arm | billed USD | standard-price USD | tokens in/out | calls / cell |", "|---|---|---|---|---|"]
            for arm, v in t4["by_arm"].items():
                cpc = "n/a" if v["calls_per_cell"] is None else f"{v['calls_per_cell']:.2f}"
                L.append(f"| {arm} | {v['billed_usd']:.4f} | {v['standard_price_usd']:.4f} | {v['tokens_in']}/{v['tokens_out']} | {cpc} |")
        else:
            L.append(f"\nCost ({s}): {t4.get('reason')}.")
        a24 = k["cost"]["a24_2"]
        rc = a24.get("R_C") or {}
        if rc.get("available"):
            L.append(f"\nA24.2 ({s}, descriptive; statement not made): R = SR(G+M)/SR(O0) = "
                     f"{A._ci_txt(rc.get('R'), lambda x: 'n/a' if x is None else f'{x:.3f}', 'infinite (O0 SR 0)')}; "
                     f"C (headline) = {A._ci_txt((rc.get('C') or {}).get('headline'), lambda x: 'n/a' if x is None else f'{x:.4f}', 'infinite')}.")
    L += ["", "## Disclosures", ""] + [f"- {d_}" for d_ in res["disclosures"]]
    return "\n".join(L) + "\n"


# ================================================================================================ run
def write_outputs(res, outdir, root: str | None = None):
    for k1 in kitchen1_store_dirs(root):
        if _inside(outdir, k1):
            raise K23AnalysisError(f"output directory {outdir} is inside a Kitchen 1 store")
    os.makedirs(outdir, exist_ok=True)
    jp, mp = os.path.join(outdir, OUT_JSON), os.path.join(outdir, OUT_MD)
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(res))
    return jp, mp


def run(k2_dir=None, k3_dir=None, *, allow_test=False, allow_partial=False, root=None, B_cost=20000, outdir=None):
    """allow_partial (--partial): incomplete seals are accepted AND the analysis is descriptive only (estimates and intervals, no criterion),
    even when every seal is complete (Astra K23-ANALYZE-r1 change 4). A kitchen without a store needs its recorded kitchen-not-run decision
    (not_run_path, A29 item 6; change 5); otherwise it is refused."""
    dirs = {"FloorPlan15": k2_dir or C.ns_docs(NS_OF["FloorPlan15"], root), "FloorPlan17": k3_dir or C.ns_docs(NS_OF["FloorPlan17"], root)}
    for s, d in dirs.items():                                       # BOTH kitchens' locations are checked before anything is read
        check_store_location(d, s, root)
    not_run = {}
    for s, d in dirs.items():
        rec = load_not_run(s, root)                                 # an invalid record is refused, never ignored
        if has_store(d):
            if rec is not None:
                raise K23AnalysisError(f"{s}: a store exists and a kitchen-not-run record exists ({rec['file']}): contradictory, refused")
        elif rec is None:
            check_store_path(d, s, root)                            # raises: unexplained missing store
        else:
            not_run[s] = rec
    if len(not_run) == len(SCENES):
        raise K23AnalysisError("no replication kitchen was run (both recorded as not run): nothing to analyse")
    if not not_run and _norm(dirs["FloorPlan15"]) == _norm(dirs["FloorPlan17"]):
        raise K23AnalysisError("the two kitchens point at the same store")
    stores = {s: load_store(d, s, allow_partial=allow_partial, root=root) for s, d in dirs.items() if s not in not_run}
    man = C.load_manifest(root)
    fam_of = {i: f["family"] for f in man["families"] for i in f["items"]}
    for s, st in stores.items():
        items = sorted({C.parse_pres(r["pres"])["item"] for r in st["rows"]})
        C.guard_items(items, st["namespace"], allow_test=allow_test, root=root)    # refuses unless THIS kitchen's freeze + --allow-test hold
        if st["seal"].get("complete"):
            inv = {(C.parse_pres(r["pres"])["item"], C.parse_pres(r["pres"])["lang"]) for r in st["rows"] if A.is_invalid_row(r)}
            miss = completeness_problems_k23(st["rows"], K.a29_roster(st["namespace"], root), inv)
            if miss:
                raise K23AnalysisError(f"{s}: the seal says complete but the store does not cover the A29 roster: " + "; ".join(miss[:8]))
    meta = {"namespaces": {s: NS_OF[s] for s in SCENES}, "seal_sha256": {s: stores[s]["seal"].get("sha256") for s in stores},
            "n_score_rows": {s: len(stores[s]["rows"]) for s in stores}, "complete": all(bool(stores[s]["seal"].get("complete")) for s in stores),
            "partial_flag": bool(allow_partial), "kitchens_not_run": sorted(not_run), "synthetic": False, "B_cost": B_cost}
    freeze = {s: freeze_cost_record(s, root) for s in stores}
    res = analyse_k23(stores, fam_of_manifest=fam_of, B_cost=B_cost, meta=meta, freeze_records=freeze, partial=allow_partial, not_run=not_run)
    return res, write_outputs(res, outdir or os.path.join(C.paths(root)["matrix_docs"], OUT_DIRNAME), root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k2", help="FloorPlan15 store (default: the test-k2-v1 namespace directory)")
    ap.add_argument("--k3", help="FloorPlan17 store (default: the test-k3-v1 namespace directory)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--B-cost", type=int, default=20000)
    ap.add_argument("--outdir")
    a = ap.parse_args(argv)
    try:
        res, (jp, mp) = run(a.k2, a.k3, allow_test=a.allow_test, allow_partial=a.partial, B_cost=a.B_cost, outdir=a.outdir)
    except (A.AnalysisError, C.TestItemRefused, C.JournalCorrupt, RuntimeError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}")
        return 2
    if res["combined"].get("available"):
        c = res["combined"]["g_m_vs_o0_criteria"]["primary"]
        print(f"G+M - O0: average criterion {c['average']['status']}; each-kitchen criterion {c['each_kitchen']['status']}")
    else:
        print(f"kitchen(s) not run: {', '.join(sorted(res['kitchens_not_run']))}; completed-kitchen estimates only, no pooled result, no criteria")
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
