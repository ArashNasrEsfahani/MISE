"""Selftest for sim/final_test/matrix/a36.py + its hooks in score.py / analyze.py (A36 draft r2: the `excluded_reset` flag and the
common-subset analysis). Offline: no simulator, no network, no API, no test item; the acceptance demo only READS acceptance-v2-a2.

    python sim/_ftm_a36_selftest.py
"""
from __future__ import annotations

import collections
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "final_test", "matrix")
sys.path.insert(0, HERE)
sys.path.insert(0, MATRIX)

import common as C                      # noqa: E402
import a36 as A36                       # noqa: E402
import analyze as A                     # noqa: E402
import score as SC                      # noqa: E402

N = {"pass": 0, "fail": 0}
B = 199


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


# ================================================================================================ exec rows shaped like acceptance-v2-a2
def blocked(att, *, epoch=0, prims=8, sweep=8, resolve=0, ok=False, reg=False, ov=False, breaches=("seen_types",), error=None, **drop):
    r = {"key": "ns|exec|u1|run0", "unit": "u1", "run": 0, "attempt": att, "pres": "canon:x:en", "status": "blocked_state", "epoch": epoch,
         "state_check": {"ok": ok, "breaches": list(breaches), "registered_ok": reg, "override": ov, "override_why": [],
                         "obs_pipeline": {"sweep_prims": sweep}},
         "audit": {"counts": {"pose": 19, "actuator": prims, "action-resolve": resolve}}, "prims": prims, "error": error, "cells": []}
    for k in drop:
        if k == "no_obs":
            r["state_check"].pop("obs_pipeline")
        if k == "no_audit":
            r["audit"] = {}
        if k == "no_prims":
            r["prims"] = None
    return r


def crash(att, prims=40):
    r = blocked(att, prims=prims, ok=True, reg=True, breaches=())
    r.update(status="crash", error="RuntimeError: unity died")
    return r


def done(att):
    r = blocked(att, prims=150, resolve=12, ok=True, reg=True, breaches=())
    r["status"] = "done"
    return r


def start(att, epoch=0):
    return {"event": A36.START_EVENT, "key": "ns|exec|u1|run0", "attempt": att, "epoch": epoch}


ST3 = [start(0), start(1), start(2)]
ATT = (0, 1, 2)


def F(hist, rc="infra_exec", **kw):
    kw.setdefault("starts", ST3)
    return A36.excluded_reset(rc, hist, **kw)


G3 = [blocked(0), blocked(1), blocked(2)]

check("flag: 3 attempts, all at the reset gate -> flagged", F(G3)["flag"] is True, F(G3))
check("flag: per-attempt evidence is recorded (stage + breaches for each of 3 attempts)",
      [a["stage"] for a in F(G3)["attempts"]] == ["reset_gate"] * 3 and F(G3)["attempts"][0]["breaches"] == ["seen_types"])
check("mixed retries: blocked, blocked, crash -> NOT flagged", not F([blocked(0), blocked(1), crash(2)])["flag"])
check("mixed retries: crash, blocked, blocked -> NOT flagged", not F([crash(0), blocked(1), blocked(2)])["flag"])
check("mixed retries: blocked, done (class ok) -> NOT flagged", not F([blocked(0), done(1)], rc="ok")["flag"])
check("first-action crash (prims beyond the sweep, status crash) -> NOT flagged", not F([blocked(0), blocked(1), crash(2, prims=9)])["flag"])
check("blocked_state but primitives ran after the sweep -> NOT flagged", not F([blocked(0), blocked(1), blocked(2, prims=9)])["flag"])
check("blocked_state but a plan step resolved an object (action-resolve 1) -> NOT flagged",
      not F([blocked(0), blocked(1, resolve=1), blocked(2)])["flag"])
check("blocked_state with an error recorded -> NOT flagged", not F([blocked(0), blocked(1, error="x"), blocked(2)])["flag"])
check("missing evidence: only 2 attempts -> NOT flagged", not F([blocked(0), blocked(1)])["flag"])
check("missing evidence: duplicated attempt -> NOT flagged", not F([blocked(0), blocked(1), blocked(1), blocked(2)])["flag"])
for kw, nm in ((dict(no_obs=1), "no sweep count"), (dict(no_audit=1), "no audit counts"), (dict(no_prims=1), "no prims"),
               (dict(reg=None), "registered_ok unknown"), (dict(ov=None), "override unknown"), (dict(breaches=()), "no breach listed")):
    check(f"missing evidence: {nm} -> NOT flagged", not F([blocked(0), blocked(1, **kw), blocked(2)])["flag"])
check("overridden under A34-A2 (override True) -> NOT flagged", not F([blocked(0), blocked(1, ov=True), blocked(2)])["flag"])
check("hold re-check block (SummaryMismatch: ok True) -> NOT flagged; stage blocked_other",
      not F([blocked(0), blocked(1), blocked(2, ok=True, reg=True, no_prims=1)])["flag"]
      and A36.attempt_stage(blocked(2, ok=True, reg=True, no_prims=1))[0] == "blocked_other")
check("inapplicable at restore (registered_ok None) -> NOT flagged", not F([blocked(0), blocked(1), blocked(2, reg=None, no_prims=1)])["flag"])
for rc in ("ok", "agent", "infra_planner", "missing", "budget", None):
    check(f"non-infra_exec class {rc!r} never flagged (even with a full reset-gate history)", not F(G3, rc=rc)["flag"])
check("not a core cell -> NOT flagged", not F(G3, core=False)["flag"])
check("an outcome row exists -> NOT flagged", not F(G3, outcome_present=True)["flag"])
check("epochs: an older epoch's 3 blocks are ignored (latest epoch has a crash) -> NOT flagged",
      not F(G3 + [dict(crash(0), epoch=1), dict(blocked(1), epoch=1), dict(blocked(2), epoch=1)])["flag"])
check("epochs: an older epoch's crash is ignored (latest epoch: 3 blocks) -> flagged",
      F([crash(0), blocked(1)] + [dict(b, epoch=1) for b in G3], starts=ST3 + [start(a, 1) for a in ATT])["flag"])

# ---- hangs (Astra r1 item 1): a lost attempt leaves a start and no row; the resume repeats the attempt number
check("hang: attempt 1 started twice (first launch hung, no row), then 3 blocked rows -> NOT flagged",
      not F(G3, starts=[start(0), start(1), start(1), start(2)])["flag"], F(G3, starts=[start(0), start(1), start(1), start(2)])["reason"])
check("hang: a start for an attempt with no row at all -> NOT flagged", not F(G3, starts=ST3 + [start(3)])["flag"])
check("no attempt-start evidence (old executor) -> NOT flagged", not F(G3, starts=None)["flag"] and "exec_attempt_start" in F(G3, starts=None)["reason"])
check("empty start list (starts recorded elsewhere, none for this cell) -> NOT flagged", not F(G3, starts=[])["flag"])
v = F(G3, starts=ST3 + [start(0, epoch=1)])
check("epoch (Astra r2): epoch 0 complete (3 blocks + 3 starts), epoch 1 has start(0) and no row -> NOT flagged (current epoch = 1)",
      v["flag"] is False and v.get("epoch") == 1, v)
v = F(G3, epoch=1)
check("epoch: the executor's epoch state is 1 (invalid_resolved) but every row / start is epoch 0 -> NOT flagged", v["flag"] is False and v.get("epoch") == 1, v)
check("epoch: executor state 1 with a complete matching epoch-1 history -> flagged",
      F([dict(b, epoch=1) for b in G3], starts=[start(a, 1) for a in ATT], epoch=1)["flag"] is True)
check("epoch: a malformed executor epoch -> NOT flagged, no exception", F(G3, epoch="1")["flag"] is False)
check("start events of an older epoch do not count for the latest epoch -> NOT flagged",
      not F([dict(b, epoch=1) for b in G3], starts=ST3)["flag"])

# ---- malformed evidence (Astra r1 item 6): never an exception, never a flag
for bad, nm in (([dict(blocked(0), attempt=None), blocked(1), blocked(2)], "attempt id None"),
                ([{k: v for k, v in blocked(0).items() if k != "attempt"}, blocked(1), blocked(2)], "attempt id missing"),
                ([dict(blocked(0), attempt="0"), blocked(1), blocked(2)], "attempt id a string"),
                ([dict(blocked(0), attempt=True), blocked(1), blocked(2)], "attempt id a bool"),
                ([dict(blocked(0), epoch="x"), blocked(1), blocked(2)], "epoch malformed"),
                ([None, blocked(1), blocked(2)], "a row is None"),
                ([dict(blocked(0), state_check="garbage"), blocked(1), blocked(2)], "state_check not a dict"),
                ([dict(blocked(0), audit=None), blocked(1), blocked(2)], "audit None"),
                ("not-a-list", "history not a list")):
    try:
        v = F(bad)
        check(f"malformed evidence ({nm}) -> NOT flagged, no exception", v["flag"] is False, v)
    except Exception as e:                                       # noqa: BLE001
        check(f"malformed evidence ({nm}) -> NOT flagged, no exception", False, repr(e))
v = F(G3, starts=[start(0), {"event": A36.START_EVENT, "attempt": None}, start(2)])
check("malformed start event -> NOT flagged", v["flag"] is False, v)
many = [f"kind{k}" for k in range(9)]
v = F([blocked(0, breaches=many), blocked(1), blocked(2)])
check("all breach kinds are kept (no truncation)", v["flag"] and v["attempts"][0]["breaches"] == many, v["attempts"][0]["breaches"])

# ---- score_cell integration
ns = "acceptance-v2-a2"
pres = C.pres_id("g-l1-pickup-cup", "en")
gold = {"plan": [{"action": "pickup", "object": "Cup"}], "level": "L1"}


def sc(cell_r=0, run=0, hist=G3, status="ok", last=None, block="main", arm="G0", starts=ST3):
    cell = f"{ns}|plan|{arm}|{pres}|r{cell_r}"
    e = last or dict(hist[-1], unit="u1")
    return SC.score_cell(cell, run, plan_row={"status": status, "plan": gold["plan"], "block": block}, exec_row=(None if status != "ok" else e),
                         outcome=None, gold_item=gold, family="fam", ns=ns, exec_history=hist, exec_starts=starts)


r0 = sc()
check("score_cell: core infra_exec cell with 3 reset-gate attempts -> excluded_reset True, class still infra_exec, SR 0",
      r0["excluded_reset"] is True and r0["run_class"] == "infra_exec" and r0["sr"] == 0.0 and r0["excluded_reset_evidence"]["rule"] == A36.A36_VERSION, r0)
check("score_cell: the same history on an r1 (repeat block) cell -> not flagged (not core)", sc(cell_r=1)["excluded_reset"] is False)
check("score_cell: a rejected plan -> not flagged", sc(status="rejected")["excluded_reset"] is False)
check("score_cell: no history passed -> the single last row is incomplete evidence -> not flagged",
      SC.score_cell(f"{ns}|plan|G0|{pres}|r0", 0, plan_row={"status": "ok", "plan": gold["plan"], "block": "main"}, exec_row=G3[-1], outcome=None,
                    gold_item=gold, family="fam", ns=ns, exec_starts=ST3)["excluded_reset"] is False)
check("score_cell: crash history -> infra_exec but not flagged", sc(hist=[blocked(0), blocked(1), crash(2)])["excluded_reset"] is False)
check("core scope (Astra r1 item 4): a curve cell G+M@o0k3 (block curve) with 3 reset-gate attempts -> NOT flagged",
      sc(arm="G+M@o0k3", block="curve")["excluded_reset"] is False and "core" in sc(arm="G+M@o0k3", block="curve")["excluded_reset_evidence"]["reason"])
check("core scope: a plan row without a block -> NOT flagged", sc(block=None)["excluded_reset"] is False)
check("core scope: a repeat-block r0 cell -> NOT flagged", sc(block="repeat")["excluded_reset"] is False)
check("score_cell: no start evidence -> NOT flagged", sc(starts=None)["excluded_reset"] is False)

# ================================================================================================ analysis
rows0, _ = A.synthetic_rows(seed=7, infra_rate=0.0)
rows0 = [r for r in rows0 if r["r"] == 0 and r["run"] == 0]
base = A.build_index(rows0)
S0, FAM = base["S"], base["fam"]


def with_flags(flag_keys=(), share_units=False):
    rows = copy.deepcopy(rows0)
    for r in rows:
        p = C.parse_pres(r["pres"])
        r["excluded_reset"] = (r["arm"], p["item"], p["lang"]) in set(flag_keys)
        if r["excluded_reset"]:
            r["sr"], r["exec_failure"], r["run_class"] = 0.0, True, "infra_exec"
            r["excluded_reset_evidence"] = {"attempts": [{"breaches": ["seen_types"]}] * 3}
    return rows


def S_of(rows):
    return A.build_index(rows)["S"]


# registered table unchanged when nothing is flagged
res_old = A.analyse(rows0, B=B, B_cost=200, meta={"synthetic": True})
res_new = A.analyse(with_flags(), B=B, B_cost=200, meta={"synthetic": True})
strip = lambda d: json.dumps({k: v for k, v in d.items() if k not in ("a36",)}, sort_keys=True, default=str)   # noqa: E731
check("registered analysis byte-identical with vs without the (all-False) flag field", strip(res_old) == strip(res_new))
check("no a36 block when the rows predate A36 (no field)", "a36" not in res_old and "a36" in res_new)
a0 = res_new["a36"]
check("nothing flagged: amended H1 wording == registered and no differences",
      a0["H1"]["wording"] == res_new["H1"]["wording"] and not a0["H1"]["differs_from_registered"] and a0["differences"] == [])
check("nothing flagged: amended Holm p == registered p for every member",
      all(abs(a0["holm_family"][h]["p"] - res_new["holm_family"][h]["p"]) < 1e-12 for h in A.FAMILY))
check("Holm membership unchanged: H2a, H2c, H3a; m = 3", a0["holm_members"] == ["H2a", "H2c", "H3a"] == sorted(A.FAMILY), a0["holm_members"])
pa = A.ANALYSIS_PLAN.get("a36") or {}
sha_with = A.analysis_plan_sha()
A.ANALYSIS_PLAN.pop("a36", None)
sha_without = A.analysis_plan_sha()
A.ANALYSIS_PLAN["a36"] = pa
check("ANALYSIS_PLAN carries the A36 block (eligibility, retention, unavailable tests, reset-limited) and it changes analysis_plan_sha",
      all(k in pa for k in ("eligibility", "retention", "unavailable", "reset_limited", "per_arm_sr", "registered_alongside"))
      and pa.get("rule") == A36.A36_VERSION and sha_with != sha_without and A.analysis_plan_sha() == sha_with
      and A.ANALYSIS_PLAN["version"] == "ftm-analysis-3")
check("markdown section renders","A36 amended subset analysis" in A.render_md(res_new))

# one-language retention: G+M flagged on item X en only -> H1 keeps X with the fa diff alone
items = sorted({i for (a, i, l) in S0})
X = items[0]
S1 = S_of(with_flags([("G+M", X, "en")]))
d1, info1 = A.item_diffs(S1, "G+M", "O0", set(), drop_flag=A36.flagged)
fa = float(S1[("G+M", X, "fa")]["sr"]) - float(S1[("O0", X, "fa")]["sr"])
check("one-language retention: item kept with its FA difference only", X in d1 and abs(d1[X] - fa) < 1e-12 and info1["dropped_flagged"] == 1, (d1.get(X), fa))
# paired: the flagged presentation leaves BOTH arms (O0's row is not used either), but only in contrasts with G+M
_, infoO = A.item_diffs(S1, "O+M", "O0", set(), drop_flag=A36.flagged)
check("paired subset: a G+M flag does not touch a contrast without G+M (O+M - O0)", infoO["dropped_flagged"] == 0)

# empty subsets: item fully flagged -> dropped; family fully flagged -> dropped; arm fully flagged -> member unavailable, p = 1, m unchanged
S2 = S_of(with_flags([("G+M", X, "en"), ("G+M", X, "fa")]))
d2, _ = A.item_diffs(S2, "G+M", "O0", set(), drop_flag=A36.flagged)
check("empty item (both languages flagged) is dropped from the contrast", X not in d2 and len(d2) == len(items) - 1)
fam0 = FAM[X]
fam_items = [i for i in items if FAM[i] == fam0]
S3 = S_of(with_flags([("G0", i, l) for i in fam_items for l in A.LANGS]))
c3 = A.contrast(S3, FAM, "G+M", "G0", set(), 0, drop_flag=A36.flagged)
c3r = A.contrast(S3, FAM, "G+M", "G0", set(), 0)
check("empty family (all its presentations flagged) is dropped; K, N recomputed",
      c3["n_families"] == c3r["n_families"] - 1 and c3["n_items"] == c3r["n_items"] - len(fam_items), (c3["n_families"], c3r["n_families"]))
S4 = S_of(with_flags([("G0", i, l) for i in items for l in A.LANGS]))
h4 = A.holm_family(S4, FAM, set(), B, drop_flag=A36.flagged)
check("empty subset: H2a (G0 fully flagged) unavailable, p = 1, still m = 3",
      h4["H2a"]["available"] is False and h4["H2a"]["p"] is None and h4["H2a"]["p_holm"] == 1.0 and sorted(h4) == ["H2a", "H2c", "H3a"], h4["H2a"])

# K < 20 -> inconclusive (a small study: 12 families)
small, _ = A.synthetic_rows(seed=11, n_fam=12, infra_rate=0.0)
small = [dict(r, excluded_reset=False) for r in small if r["r"] == 0 and r["run"] == 0]
Ss = A.build_index(small)
hs = A.h1_block(Ss["S"], Ss["fam"], set(), B, drop_flag=A36.flagged)
check("K < 20 -> H1 inconclusive (guard failed) in the amended analysis",
      "inconclusive" in hs["wording"] and hs["primary"]["decision"]["guard"]["passed"] is False, hs["wording"])

# the 5 % reset-limited label (strictly MORE than 5 %)
nG0 = sum(1 for (a, i, l) in S0 if a == "G0")
k6 = int(0.05 * nG0) + 1
k5 = int(0.05 * nG0)
pl = [(i, l) for i in items for l in A.LANGS]
fl6 = A36.arm_flags(S_of(with_flags([("G0",) + p for p in pl[:k6]])), ["G0"], set(), FAM, base["level"])["G0"]
fl5 = A36.arm_flags(S_of(with_flags([("G0",) + p for p in pl[:k5]])), ["G0"], set())["G0"]
check(f"5 % label: {k6}/{nG0} flagged -> reset-limited", fl6["reset_limited"] is True and fl6["flagged"] == k6, fl6["share"])
check(f"5 % label: {k5}/{nG0} flagged -> not reset-limited", fl5["reset_limited"] is False, fl5["share"])
twenty = {("G0", f"i{k}", "en"): {"sr": 1.0, "excluded_reset": k == 0} for k in range(20)}
check("5 % label: exactly 1/20 = 5 % -> not reset-limited (strict)", A36.arm_flags(twenty, ["G0"], set())["G0"]["reset_limited"] is False)

check("5 % label: per-arm primary SR uses the non-flagged cells and prints both denominators (items, presentations)",
      fl6["sr_primary"]["n_presentations"] == nG0 - k6 and fl6["sr_registered"]["n_presentations"] == nG0
      and fl6["sr_registered"]["n_items"] == len(items) and fl6["sr_primary"]["n_items"] <= len(items))

# per-arm SR keeps the registered item weighting (Astra r1 item 5): A en = 1 (A fa flagged), B en = B fa = 0 -> 1/2, not 1/3
tiny = {("G0", "A", "en"): {"sr": 1.0}, ("G0", "A", "fa"): {"sr": 0.0, "excluded_reset": True},
        ("G0", "B", "en"): {"sr": 0.0}, ("G0", "B", "fa"): {"sr": 0.0}}
ft = A36.arm_flags(tiny, ["G0"], set(), {"A": "f1", "B": "f2"}, {})["G0"]
check("per-arm SR is item-weighted: 1/2 (2 items, 3 presentations), registered 1/4",
      abs(ft["sr_primary"]["sr"] - 0.5) < 1e-12 and ft["sr_primary"]["n_items"] == 2 and ft["sr_primary"]["n_presentations"] == 3
      and abs(ft["sr_registered"]["sr"] - 0.25) < 1e-12, ft)

# small / empty common subsets (Astra r1 item 3): never AnalysisError; explicit zero counts; unavailable, Holm p = 1
keepfam = FAM[items[0]]
S_one = S_of(with_flags([("G+M", i, l) for i in items if FAM[i] != keepfam for l in A.LANGS]))
try:
    b1 = A36.a36_block(S_one, FAM, set(), B, True, lvl_of=base["level"])
    check("one retained family: no exception; H1 unavailable + inconclusive wording; H2a / H3a unavailable with p Holm 1",
          b1["H1"]["available"] is False and b1["H1"]["wording"] == A36.NO_RETAINED and b1["H1"]["retained"]["amended"]["families"] == 1
          and b1["holm_family"]["H2a"]["available"] is False and b1["holm_family"]["H2a"]["p_holm"] == 1.0
          and b1["holm_family"]["H3a"]["p_holm"] == 1.0 and b1["holm_members"] == ["H2a", "H2c", "H3a"], (b1["H1"], b1["holm_family"]["H2a"]))
except Exception as e:                                           # noqa: BLE001
    check("one retained family: no exception", False, repr(e))
S_none = S_of(with_flags([("G+M", i, l) for i in items for l in A.LANGS]))
try:
    b0 = A36.a36_block(S_none, FAM, set(), B, True, lvl_of=base["level"])
    t0 = b0["H1"]["retained"]["amended"]
    check("empty common subset: explicit zero counts (ints, not None) and an inconclusive / unavailable H1, equivalence unavailable",
          t0 == {"presentations": 0, "items": 0, "families": 0, "K": 0, "dropped_flagged": 2 * len(items), "missing": 0}
          and b0["H1"]["available"] is False and "inconclusive" in b0["H1"]["wording"]
          and b0["equivalence_exploratory"]["available"] is False and b0["by_arm"]["G+M"]["sr_primary"]["n_items"] == 0, (t0, b0["H1"]["wording"]))
    check("empty common subset renders in markdown", "A36 amended subset analysis" in " ".join(A36.render_md(b0)))
except Exception as e:                                           # noqa: BLE001
    check("empty common subset: no exception", False, repr(e))

# K drops below 20 BECAUSE of exclusions (note 8): the registered H1 passes the guard, the amended one fails it -> inconclusive, difference named
_, inf_r = A.item_diffs(S0, "G+M", "O0", set())
disc_f = sorted({FAM[i] for i in inf_r["_disc_items"]})
regK = len(disc_f)
cut = disc_f[:max(0, regK - 19)]
SK = S_of(with_flags([("G+M", i, l) for i in items if FAM[i] in cut for l in A.LANGS]))
rK = A.analyse([r for r in with_flags([("G+M", i, l) for i in items if FAM[i] in cut for l in A.LANGS])], B=B, B_cost=200, meta={"synthetic": True})
aK = rK["a36"]["H1"]
check(f"K crosses below 20 through exclusions: registered K {regK} >= 20 passes the guard; amended K {aK['retained']['amended']['K']} -> inconclusive",
      regK >= 20 and rK["H1"]["primary"]["decision"]["guard"]["passed"] is True and aK["retained"]["amended"]["K"] == 19
      and aK["primary"]["decision"]["guard"]["passed"] is False and "inconclusive" in aK["wording"], (regK, aK["retained"], aK["wording"]))
check("... and the amended / registered difference is named when the wordings differ",
      aK["differs_from_registered"] == (aK["wording"] != rK["H1"]["wording"])
      and (not aK["differs_from_registered"] or any(x.startswith("H1 wording") for x in rK["a36"]["differences"])))
rows6 = with_flags([("G0",) + p for p in pl[:k6]])
r6 = A.analyse(rows6, B=B, B_cost=200, meta={"synthetic": True})
check("reset-limited arm labels every contrast that involves it (H2a) and no other (H2c)",
      r6["a36"]["holm_family"]["H2a"]["reset_limited"] == ["G0"] and r6["a36"]["holm_family"]["H2c"]["reset_limited"] == [])
check("registered analysis keeps flagged cells (SR 0, in every denominator): H2a registered n_pres > amended n_pres",
      r6["holm_family"]["H2a"]["n_pres"] - r6["a36"]["holm_family"]["H2a"]["n_pres"] == k6
      and r6["a36"]["holm_family"]["H2a"]["retained"]["amended"]["dropped_flagged"] == k6)
check("breach kinds disclosed per arm", r6["a36"]["by_arm"]["G0"]["breach_kinds"] == {"seen_types": 3 * k6})

# every contrast (Astra r1 item 2): exploratory H4 / H5 and the equivalence get amended estimates, retained counts and labels, stay exploratory
rowsOM = with_flags([("O+M",) + p for p in pl[:k6]])
rOM = A.analyse(rowsOM, B=B, B_cost=200, meta={"synthetic": True})["a36"]
exo_ok = all(("retained" in v and v.get("exploratory") is True and v.get("in_holm") is False
              and v["reset_limited"] == (["O+M"] if "O+M" in A.EXPLORATORY[h][:2] else [])) for h, v in rOM["exploratory_contrasts"].items())
check("exploratory H4a-b / H5a-c: amended estimate + retained counts + reset-limited label, still exploratory (no Holm)",
      sorted(rOM["exploratory_contrasts"]) == sorted(A.EXPLORATORY) and exo_ok, {h: v.get("reset_limited") for h, v in rOM["exploratory_contrasts"].items()})
check("H4b (O+M - O+FS) drops exactly the flagged O+M presentations",
      rOM["exploratory_contrasts"]["H4b"]["retained"]["amended"]["dropped_flagged"] == k6)
check("equivalence G+M vs O+M: amended decision, retained counts, labelled reset-limited by O+M",
      rOM["equivalence_exploratory"]["reset_limited"] == ["O+M"] and rOM["equivalence_exploratory"]["retained"]["amended"]["dropped_flagged"] == k6
      and "decision" in rOM["equivalence_exploratory"])

# shared execution units: G0 and O0 share a unit at (X, en); the flag reaches both rows -> dropped once from G0-vs-O0 and from each arm's
# other contrasts
rows_sh = with_flags([("G0", X, "en"), ("O0", X, "en")])
for r in rows_sh:
    p = C.parse_pres(r["pres"])
    if p["item"] == X and p["lang"] == "en" and r["arm"] in ("G0", "O0"):
        r["unit"] = "shared-unit"
Ssh = S_of(rows_sh)
_, i_go = A.item_diffs(Ssh, "G0", "O0", set(), drop_flag=A36.flagged)
_, i_gm = A.item_diffs(Ssh, "G+M", "G0", set(), drop_flag=A36.flagged)
_, i_om = A.item_diffs(Ssh, "O+M", "O0", set(), drop_flag=A36.flagged)
check("shared unit: the flagged shared presentation is dropped once in G0 - O0 and once in each arm's other contrast",
      i_go["dropped_flagged"] == 1 and i_gm["dropped_flagged"] == 1 and i_om["dropped_flagged"] == 1)

# the registered H1 never sees the flag
rr = A.analyse(rows6, B=B, B_cost=200, meta={"synthetic": True})
plain6 = [{k: v for k, v in r.items() if k not in ("excluded_reset", "excluded_reset_evidence")} for r in rows6]
rp = A.analyse(plain6, B=B, B_cost=200, meta={"synthetic": True})
check("registered H1 / Holm identical whether or not the flag field is present", strip(rr) == strip(rp))

# ================================================================================================ executor -> scorer (note 8): shared units end to end
NS_E = "rehearsal-v1"
MAN = C.load_manifest(C.ROOT)
DATA = json.load(open(C.paths(C.ROOT)["dataset"], encoding="utf-8"))
ITEMS = {i["id"]: i for i in DATA["items"]}
DEV = [i for i in MAN["dev_pilot"] if ITEMS[i]["plan"]]


def e2e(hang=False):
    root = tempfile.mkdtemp(prefix="ftm_a36_")
    try:
        os.makedirs(os.path.join(root, "docs", "final-test"))
        os.makedirs(os.path.join(root, "sim", "dataset"))
        os.makedirs(os.path.join(root, "sim", "step4"))
        shutil.copy(C.paths(C.ROOT)["split"], C.paths(root)["split"])
        shutil.copy(C.paths(C.ROOT)["dataset"], C.paths(root)["dataset"])
        it = DEV[0]
        pr = C.pres_id(it, "en")
        plan = [{"action": s_["action"], "object": s_.get("object"), "to": s_.get("to"), "arg": s_.get("arg")} for s_ in ITEMS[it]["plan"]]
        cell = lambda arm, r=0: C.cell_id(NS_E, "plan", arm, pr, f"r{r}")          # noqa: E731
        J = lambda k: C.Journal(C.journal_path(NS_E, k, root))                    # noqa: E731
        for arm, blk in (("G0", "main"), ("O0", "main"), ("G+M@o0k3", "curve")):
            J("plan").append(C.make_record("plan", key=cell(arm), namespace=NS_E, arm=arm, block=blk, pres=pr, r=0, session=0, request_sha="x",
                                           status="ok", plan=plan, first_plan=plan, first_ok=True, repaired=False, calls=1, cached_calls=0,
                                           tokens_in=1, tokens_out=1, cost_usd=0.0, model="m", scope="free"))
        cells = [cell("G0"), cell("O0"), cell("G+M@o0k3"), cell("P0")]          # ONE execution unit shared by four cells
        key = f"{NS_E}|exec|uS|run0"
        for a in (0, 1, 2):
            J("events").append(C.make_record("events", event=A36.START_EVENT, phase="exec", namespace=NS_E, key=key, unit="uS", run=0, attempt=a, epoch=0))
            if hang and a == 1:                                                    # the first launch of attempt 1 hung: start, no row, repeated
                J("events").append(C.make_record("events", event=A36.START_EVENT, phase="exec", namespace=NS_E, key=key, unit="uS", run=0, attempt=a, epoch=0))
            r = blocked(a)
            r.update(key=key, namespace=NS_E, unit="uS", pres=pr, cells=cells, wall_s=1.0, snapshot_sha_before=None, snapshot_sha_after=None)
            J("exec").append(C.make_record("exec", **{k: v for k, v in r.items() if k not in ("schema", "spec", "ts")}))
        for ph, k in (("plan", "plan"), ("exec", "exec")):
            J("events").append(C.make_record("events", event="phase_complete", phase=ph, n=0, journal_sha=C.sha256_file(C.journal_path(NS_E, k, root))))
        SC.run_score(NS_E, root=root, log=lambda *a, **k: None)
        return {r["arm"]: r for r in C.Journal(C.journal_path(NS_E, "score", root)).rows()}
    finally:
        shutil.rmtree(root, ignore_errors=True)


try:
    sr_ = e2e()
    check("executor -> scorer: one shared unit with 3 started + blocked attempts flags EVERY main-block cell on it (G0, O0, P0)",
          all(sr_[a]["excluded_reset"] is True and sr_[a]["run_class"] == "infra_exec" for a in ("G0", "O0", "P0")),
          {a: (v["excluded_reset"], v["excluded_reset_evidence"]["reason"]) for a, v in sr_.items()})
    check("executor -> scorer: the curve cell on the same unit is NOT flagged (not core)", sr_["G+M@o0k3"]["excluded_reset"] is False)
    Se = {(r["arm"], C.parse_pres(r["pres"])["item"], "en"): r for r in sr_.values() if r["arm"] in ("G0", "O0")}
    _, ie = A.item_diffs(Se, "G0", "O0", set(), drop_flag=A36.flagged)
    check("executor -> scorer: the shared flagged presentation leaves the G0 - O0 contrast once", ie["dropped_flagged"] == 1 and ie["n_pres"] == 0)
    sh_ = e2e(hang=True)
    check("executor -> scorer: an extra (hung) start for attempt 1 -> no cell flagged, registered infra_exec kept",
          not any(v["excluded_reset"] for v in sh_.values()) and sh_["G0"]["run_class"] == "infra_exec",
          {a: v["excluded_reset_evidence"]["reason"] for a, v in sh_.items()})
except Exception as e:                                                             # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("executor -> scorer end to end ran", False, repr(e))

# ================================================================================================ read-only acceptance demo
acc = os.path.join(os.path.dirname(HERE), "docs", "final-test", "matrix", "acceptance-v2-a2")


def _jl(name):
    with open(os.path.join(acc, name), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


if os.path.exists(os.path.join(acc, "exec.jsonl")):
    H = collections.defaultdict(list)
    for r in _jl("exec.jsonl"):
        H[r["key"]].append(r)
    starts = collections.defaultdict(list)
    for e in _jl("events.jsonl"):
        if e.get("event") == A36.START_EVENT:
            starts[e.get("key")].append(e)
    plans = {r["key"]: r for r in _jl("plan.jsonl")}
    res_ = {}
    for mode in ("strict", "rows_only"):
        units, cells = set(), []
        for k, h in H.items():
            last = A36.history_of(h)[-1]
            rc = C.run_class({"crash": "exec_crash", "blocked_state": "exec_blocked_state"}.get(last["status"], "ok"))
            if mode == "strict":
                st = starts.get(k, []) if starts else None
            else:                                                                   # start evidence assumed = one start per journaled row
                st = [{"attempt": x["attempt"], "epoch": x.get("epoch", 0)} for x in A36.history_of(h)]
            for c in last.get("cells") or ():
                p = c.split("|")
                blk = (plans.get(c) or {}).get("block") or ("main" if C.arm_info(p[2]).get("planner") == "ref" else None)
                core = blk == "main" and p[4] == "r0" and int(last["run"]) == 0 and C.parse_pres(p[3])["world"] == "canon"
                if A36.excluded_reset(rc, h, starts=st, outcome_present=last["status"] == "done", core=core)["flag"]:
                    units.add(k)
                    cells.append((p[2], p[3]))
        res_[mode] = (len(cells), len(units), dict(sorted(collections.Counter(a for a, _ in cells).items())))
        print(f"  acceptance-v2-a2 demo ({mode}, read-only): {res_[mode][0]} cells / {res_[mode][1]} units; by arm {res_[mode][2]}")
    check("acceptance demo, strict: 0 flagged (that executor journaled no exec_attempt_start, so a lost attempt cannot be ruled out)",
          res_["strict"][0] == 0 and not starts, res_["strict"])
    check("acceptance demo, rows-only (start evidence assumed): 14 cells / 7 units (attempt 2's terminal blocks), G0 4, P0 2",
          res_["rows_only"][0] == 14 and res_["rows_only"][1] == 7 and res_["rows_only"][2].get("G0") == 4 and res_["rows_only"][2].get("P0") == 2,
          res_["rows_only"])
else:
    check("acceptance-v2-a2 demo skipped (journal absent)", True)

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
