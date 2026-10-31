"""Loading, the A36 flag filter and the per-arm tables of the sealed score rows. Read only.

Adjusted = the signed A36 rule applied in every kitchen: a presentation whose three scene-restoration attempts all stopped before any action
(score row flag `excluded_reset`) is left out of its arm; contrasts use the paired common subset (a presentation flagged in either arm
leaves both). Registered = every presentation, a fault counts as failure.

Every number comes from the project's own analysis functions (sim/final_test/matrix: analyze.per_arm_tables with its family-cluster
bootstrap, k23_analyze.kitchen_cr2 / pooled_result with the nominal 96.5% CR2 interval).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAT = os.path.join(ROOT, "sim", "final_test", "matrix")
sys.path[:0] = [MAT, os.path.join(ROOT, "sim")]
import analyze as AN          # noqa: E402
import a36 as A36             # noqa: E402
import k23_analyze as KA      # noqa: E402

D = lambda *p: os.path.join(ROOT, "docs", "final-test", *p)   # noqa: E731
K1_SCORE = D("matrix", "test-v1", "score.jsonl")
K2_SCORE = D("matrix", "test-k2-v1", "score.jsonl")
K3_SCORE = D("matrix", "test-k3-v1", "score.jsonl")
K2_CODEX = D("k23", "codex", "test-k2-v1", "score.codex.jsonl")
K3_CODEX = D("k23", "codex", "test-k3-v1", "score.codex.jsonl")
K1_ANALYSIS = D("matrix", "test-v1", "analysis.json")
K1_ARMS = ["G0", "G+M", "O0", "O+M", "P0", "P+M", "X0", "X+M", "G+FS", "O+FS", "G+M/p", "O+M/p"]
K23_ARMS = ["G0", "G+M", "O0", "O+M", "P0", "P+M", "X0", "X+M"]       # P = reference plans, X = GPT6-Sol (addendum files)


def index(path, *extra):
    rows = AN.read_jsonl(path)
    for e in extra:
        rows = rows + AN.read_jsonl(e)
    idx = AN.build_index(rows)
    return idx, AN.invalid_presentations(idx["S"])


def without_flagged(idx):
    return dict(idx, S={k: r for k, r in idx["S"].items() if not A36.flagged(r)})


def rate(r):
    return None if r is None else {"sr": r["sr"], "ci95": r["ci95"], "n_items": r["n_items"], "n_families": r["n_families"]}


def tables(idx, drop_invalid, arms):
    t = AN.per_arm_tables(idx, arms, drop_invalid)
    out = {}
    for a in arms:
        n_pres = sum(1 for (x, i, l) in idx["S"] if x == a and (i, l) not in drop_invalid)
        out[a] = {"overall": rate(t[a]["overall"]), "n_presentations": n_pres,
                  "level": {L: rate(v) for L, v in t[a]["level"].items()},
                  "lang": {g: rate(v) for g, v in t[a]["lang"].items()},
                  "level_x_lang": {k: rate(v) for k, v in t[a]["level_x_lang"].items()},
                  "level_presentations": {L: sum(1 for (x, i, l) in idx["S"] if x == a and idx["level"].get(i) == L and (i, l) not in drop_invalid)
                                          for L in t[a]["level"]}}
    return out


def contrasts(idx, drop_invalid, drop):
    kw = {"drop_flag": A36.flagged} if drop else {}
    return {f"{a} - {b}": KA.contrast_parts(idx["S"], idx["fam"], a, b, drop_invalid, **kw) for a, b in KA.CONTRASTS}


def cr2_public(c):
    r = KA.kitchen_cr2(c)
    return {k: r.get(k) for k in ("estimate", "interval", "interval_defined", "se", "nu", "K", "n_families", "n_items", "n_pres",
                                   "dropped_flagged", "available", "reason")}
