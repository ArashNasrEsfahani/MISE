"""Final test - the enumerated planner-call plan (stage P3; pure, offline, no simulator, no API).

Astra P3: "treat ~1,810 calls as provisional until an enumerated arm-by-presentation manifest reconciles sharing,
repeats, relocation, repairs and retries". This module is that enumeration. It reads the frozen split manifest
(docs/final-test/split_manifest.json) for the set sizes and lists every FIRST-PASS planner call cell of the
preregistered design (preregistration s3, s6.3, s7, s10, A6, A15) with the sharing rules made explicit:

  * main matrix, r = 0: every arm on all test presentations (A23: core four G0 / G+M / O0 / O+M, FS control G+FS / O+FS, gpt-6-sol X0 / X+M;
    Sonnet and Haiku are ARCHIVED and have no row);
  * REPEAT, r = 1, 2: the core four (G0, G+M, O0, O+M) on the REPEAT presentations (r = 0 IS the main-matrix call: shared).
    A27 (draft, PENDING ARASH SIGN-OFF AT THE FREEZE): G+FS, O+FS, X0, X+M have no REPEAT calls any more;
  * learning curve (E4), r = 0, REPEAT presentations, G+M and O+M: snapshots of order 0 only, K = 1, 3, 5 (A27 draft; orders 1, 2
    were dropped). K = 0 = the "0" arm (shared: no memory) and order 0 / K = 10 = S* (shared with the main matrix);
  * relocation probe (E6), r = 0: G0, G+M, O0, O+M on the RELOCATION presentations in the relocated world (new
    observations -> new prompts, nothing shared);
  * Codex gpt-6-sol X0 / X+M (A18): MAIN arms, in the main matrix (r = 0), executed like G / O, free (A27 draft: no REPEAT block);
  * Codex X-*: the other three models, REPEAT x {0, +M} x 3 calls per model (exploratory, subscription, outside every USD ceiling);
    PLAN-ONLY since A25 item 3 / A27: planned and scored as planning diagnostics, never executed;
  * OpenRouter rows are void (prereg A6) and absent;
  * the P (reference-plan) and /p (cached-plan replay) arms make no planner call.

Repairs (at most one per presentation) and transport retries are NOT cells: they are added as a rate on top
(``repair_rate``). Nested snapshots can render the same memory block for some presentations (then the cache would
make the call free); the enumeration does not assume that, so it is an upper bound for the curve.

Usage: ``python sim/final_test_callplan.py`` (table) / ``--json`` / ``--remaining <namespace> [--root R]``.

``--remaining`` is the monitor's projection input (``monitor.py --callplan``): ``{remaining: {O (and any archived S / H a namespace opted into): n}, mean_cost: {...: usd}}`` from the
Phase A/B journals of a namespace, cut groups excluded (``plan.Plan.remaining_report``). ``--json`` alone has no live state, so it carries
neither key.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import common as C  # noqa: E402  (SPEC names: C.ARMS, C.BLOCKS, C.scope_of, C.guard_items)

MANIFEST = C.paths()["split"]

# OpenRouter is void (A6): no R0 / R+M rows. Every arm name below is a key of C.ARMS (or a curve source via C.arm_info).
MODEL_OF = {"G": "gemma4:e2b-it-q4_K_M", "S": "claude-sonnet-5-5", "O": "claude-opus-5-5",
            "H": "claude-haiku-4-5-20251001", "X": "codex_cli (gpt-6-sol main arms X0 / X+M + 3 exploratory models, subscription)"}
NS = "test-v1"                                        # the namespace this enumeration is about (A23: no archived arm can enter it)
MAIN_ARMS = C.MAIN_ARMS                               # A23: 4 core + 2 FS + the A18 gpt-6-sol arms X0 / X+M (free, executed, REPEAT r = 1, 2)
CORE_ARMS = C.CORE_ARMS
#: A27 (draft, PENDING ARASH SIGN-OFF AT THE FREEZE): order 0, K = 1, 3, 5 only (was 11 snapshots over orders 0-2); REPEAT r = 1, 2 for the core
#: four only; exploratory Codex plan-only. The numbers live in common.py (one source for plan / capture / execute / this file).
CURVE_SNAPSHOTS = [tuple(ok) for ok in C.CURVE_SNAPSHOTS]
CURVE_SNAPSHOTS_PRE_A27 = [(0, 1), (0, 3), (0, 5), (1, 1), (1, 3), (1, 5), (1, 10), (2, 1), (2, 3), (2, 5), (2, 10)]
REPEAT_ARMS = C.REPEAT_ARMS
CODEX_MODELS = C.CODEX_EXPLORATORY                   # A18: gpt-6-sol is a main arm (X0 / X+M in MAIN_ARMS); these three stay exploratory
#: split-manifest key of the item set each block draws on (the counts here; ids only through item_ids()).
BLOCK_SET = {"main": "test", "repeat": "repeat_subset", "curve": "repeat_subset", "reloc": "relocation_probe",
             "codex": "repeat_subset"}


def sizes(manifest_path: str = MANIFEST) -> dict:
    """Presentation counts (items x EN/FA) only; no item id leaves this function."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    return {"test": 2 * len(m["test"]), "repeat": 2 * len(m["repeat_subset"]),
            "relocation": 2 * len(m["relocation_probe"]), "dev": 2 * len(m["dev_pilot"]),
            "manifest_sha256": m.get("sha256")}


def item_ids(block: str, namespace: str, *, allow_test: bool = False, root: str | None = None) -> list:
    """Item ids of a block's set. TEST items are refused (C.TestItemRefused) unless the SPEC guard holds: namespace test-v1,
    --allow-test, a valid freeze.json and the approved family audit. Non-test namespaces get the block's set only if it lies
    inside the namespace's set (the guard refuses anything else)."""
    if block not in C.BLOCKS:
        raise ValueError(f"unknown block {block!r}")
    m = C.load_manifest(root)
    ids = list(m[BLOCK_SET[block]])
    C.guard_items(ids, namespace, allow_test=allow_test, root=root)
    return ids


def _scope(arm: str, block: str, namespace: str = NS) -> str:
    return C.scope_of(arm, block, namespace)


def cells(sz: dict, namespace: str = NS) -> list:
    """-> [{block, arm, scope, world, model_key, presentations, replicates, memory, calls}] (first-pass calls only).
    ``block`` is a C.BLOCKS name; ``label`` is the human reading of the row."""
    C.archived_guard(namespace)                           # A23: test-v1 (and any namespace without the opt-in) cannot carry Sonnet / Haiku rows
    main_arms = C.main_arms_for(namespace)
    out = []

    def add(block, arm, n, reps=1, label=None, world="canon"):
        out.append({"block": block, "label": label or block, "arm": arm, "scope": _scope(arm, block, namespace), "world": world,
                    "model_key": C.arm_info(arm, namespace)["planner"][0], "presentations": n, "replicates": reps,
                    "memory": C.arm_info(arm, namespace)["prompt"] != "none", "calls": n * reps})

    for a in main_arms:
        add("main", a, sz["test"], label="main r=0")
    for a in main_arms:                                   # r = 0 is the main-matrix call (shared): only r = 1, 2 are new
        if a in REPEAT_ARMS:                              # A27 (draft): core four only
            add("repeat", a, sz["repeat"], 2, label="REPEAT r=1,2")
    for a in ("G+M", "O+M"):                              # K = 0 is the shared '0' arm; o0k10 is S* (shared with main)
        for order, k in CURVE_SNAPSHOTS:
            add("curve", f"{a}@o{order}k{k}", sz["repeat"], label=f"curve order {order} K={k}")
    for a in ("G0", "G+M", "O0", "O+M"):
        add("reloc", a, sz["relocation"], label="relocation r=0", world="reloc")
    for cm in CODEX_MODELS:                               # Codex: exploratory, subscription, never in a USD ceiling; A27 / A25.3: PLAN-ONLY
        for a in (f"X-{cm}0", f"X-{cm}+M"):
            add("codex", a, sz["repeat"], 3, label=f"Codex {cm} REPEAT r=0,1,2 (plan-only)")
    return out


def pilot_cells(sz: dict) -> list:
    return [{"block": "main", "label": "cost pilot (dev)", "arm": a, "scope": _scope(a, "main"), "world": "canon",
             "model_key": a[0], "presentations": sz["dev"], "replicates": 1, "memory": "+M" in a, "calls": sz["dev"]}
            for a in CORE_ARMS]


def totals(cs: list) -> dict:
    t = {}
    for c in cs:
        d = t.setdefault(c["model_key"], {"calls": 0, "calls_mem": 0, "core_calls": 0})
        d["calls"] += c["calls"]
        d["calls_mem"] += c["calls"] if c["memory"] else 0
        d["core_calls"] += c["calls"] if c["scope"] == "core" else 0
    return t


def by_scope(cs: list) -> dict:
    """Calls per ledger scope (A15): core / other (paid) and free. Used by M5 for ``remaining`` and the cost guard."""
    t = {"core": 0, "other": 0, "free": 0}
    for c in cs:
        t[c["scope"]] += c["calls"]
    return t


def block_reservations(cs: list, reserve_per_call_usd: dict) -> list:
    """Worst-case hold per (block, scope, model) for the ledger (A15): C.block_worst_case over the first-pass calls.
    ``reserve_per_call_usd`` maps model key (S, O, H) to the max per-call reserve; free blocks reserve nothing."""
    out = {}
    for c in cs:
        if c["scope"] == "free":
            continue
        k = (c["block"], c["scope"], c["model_key"])
        out[k] = out.get(k, 0) + c["calls"]
    return [{"block": b, "scope": sc, "model_key": m, "calls": n,
             "worst_usd": C.block_worst_case(n, reserve_per_call_usd[m])} for (b, sc, m), n in sorted(out.items())]


def usd(n_plain: int, n_mem: int, price: dict, think: int, in_plain=1300, in_mem=2600, answer=150,
        repair_rate=0.15) -> float:
    """List-price USD for first-pass calls x (1 + repair_rate). Assumptions are the P2 smoke's, labelled there."""
    per_plain = in_plain * price["input"] + (answer + think) * price["output"]
    per_mem = in_mem * price["input"] + (answer + think) * price["output"]
    return (n_plain * per_plain + n_mem * per_mem) / 1e6 * (1 + repair_rate)


#: Projection assumptions (labelled; the monitor replaces them with observed means as soon as cells are billed). USD per planner cell incl. repairs:
#: the post-A11 priors in plan.PRIOR_USD_PER_CELL (O = the 0.012 expected mean of A19). Thinking tail: the W2 re-measure put the whole plan near 50 USD
#: with the tail, against the 55 USD ceiling of A18 (A22 raised the caps to core 30 / whole 65 USD); that figure is a prior, not computed here. Seconds per executed episode: practice-v1 mean
#: (533 s over 28 executed episodes = 19 s); the test-v1 mean is unknown until W5 starts.
PRIOR_USD_PER_CELL = {"O": 0.012, "S": 0.0062, "H": 0.0026}      # S and H: ARCHIVED (A23), kept so an opted-in namespace can still be priced
EPISODE_WALL_S = 533.0 / 28
REPEAT_RUNS = C.EXEC_RUNS_REPEAT                      # A27 (draft): 2 executions per distinct unit on REPEAT presentations (was 3)
P_ARMS = ("P0", "P+M")                                # execute.P_ARMS: reference-plan controls (no call), every captured presentation
REPLAY_ARMS = ("G+M/p", "O+M/p")                      # execute.REPLAY_ARMS: cached +M plans re-run with the prior chooser (no call)


def _cell_exec(c: dict, sz: dict, runs: int, repeat_exec, plan_only) -> int:
    """Executions BEFORE dedup of one cell-row of `cells()` (each plan cell is its own unit; dedup is applied later as a ratio)."""
    if plan_only(c["arm"]):
        return 0
    if c["block"] == "repeat":
        return c["calls"] * (runs if repeat_exec(c["arm"]) else 1)
    if c["block"] == "main" and repeat_exec(c["arm"]):              # the r = 0 cells of the REPEAT presentations sit inside block main
        return c["calls"] + sz["repeat"] * (runs - 1)
    return c["calls"]


def projection(sz: dict, cs: list | None = None) -> dict:
    """Cost and time projection of the whole plan (A18): paid USD per ledger scope from the priors, and the number of simulator episodes.
    Episodes are an UPPER bound for the LISTED matrix (plan arms only; the P0 / P+M and /p controls are in `execution_enumeration`): every
    executed plan cell is one execution, except that on REPEAT presentations the core-four cells of blocks main and repeat run ``REPEAT_RUNS``
    times (A27 draft); exploratory Codex cells are plan-only (0 executions); the unit dedup is not modelled. X0 / X+M (gpt-6-sol) are free."""
    cs = cells(sz) if cs is None else cs
    usd = {"core": 0.0, "other": 0.0}
    eps = {"main": 0, "repeat": 0, "curve": 0, "reloc": 0, "codex": 0}
    for c in cs:
        if c["scope"] in usd:
            usd[c["scope"]] += c["calls"] * PRIOR_USD_PER_CELL[c["model_key"]]
        eps[c["block"]] += _cell_exec(c, sz, REPEAT_RUNS, C.repeat_exec_arm, C.plan_only)
    executions = sum(eps.values())
    return {"paid_usd_prior": {k: round(v, 2) for k, v in usd.items()}, "paid_usd_total_prior": round(sum(usd.values()), 2),
            "ceilings": C.NAMESPACES["test-v1"]["ceilings"], "episodes_upper_bound": executions,
            "episodes_by_block": eps, "episodes_upper_bound_note": f"{C.RUNTIME_PLAN}: REPEAT presentations, core four: {REPEAT_RUNS} runs per unit; "
                                                                   "exploratory Codex plan-only; P0 / P+M and /p controls not included (see execution_enumeration)",
            "episode_wall_s": round(EPISODE_WALL_S, 1),
            "hours_upper_bound": round(executions * EPISODE_WALL_S / 3600, 1),
            "x_main_arms": {"cells": sum(c["calls"] for c in cs if c["arm"] in C.X_MAIN_ARMS),
                            "episodes_upper_bound": sum(_cell_exec(c, sz, REPEAT_RUNS, C.repeat_exec_arm, C.plan_only) for c in cs if c["arm"] in C.X_MAIN_ARMS)},
            "note": "list-price priors without the thinking tail (A18: the tail brings the whole plan near 50 USD; ceilings core 30 / cap 100 since A24.10); episodes ignore unit dedup"}


# ================================================================================================ A27: the real execution units
def _pre_a27_cells(sz: dict, namespace: str = NS) -> list:
    """The call plan as registered BEFORE A27 (REPEAT r = 1, 2 for all 8 main arms, 11 curve snapshots, Codex executed once per cell): only
    for the before / after comparison in `execution_enumeration`; nothing runs from it."""
    main_arms = C.main_arms_for(namespace)
    out = [{"block": "main", "arm": a, "calls": sz["test"]} for a in main_arms]
    out += [{"block": "repeat", "arm": a, "calls": 2 * sz["repeat"]} for a in main_arms]
    out += [{"block": "curve", "arm": f"{a}@o{o}k{k}", "calls": sz["repeat"]} for a in ("G+M", "O+M") for o, k in CURVE_SNAPSHOTS_PRE_A27]
    out += [{"block": "reloc", "arm": a, "calls": sz["relocation"]} for a in ("G0", "G+M", "O0", "O+M")]
    out += [{"block": "codex", "arm": f"X-{m}{s}", "calls": 3 * sz["repeat"]} for m in CODEX_MODELS for s in ("0", "+M")]
    return out


def _controls(sz: dict, cs: list, runs: int, repeat_exec, plan_only) -> list:
    """The executor's default controls (execute.build_units): P0 / P+M on every captured presentation (canonical + relocated), one run; the /p
    replays of every executed G+M / O+M plan cell (blocks main, repeat, reloc; not curve, whose arms are G+M@... / O+M@...), under `repeat_exec`."""
    out = []
    for pa in P_ARMS:
        out.append({"group": "control_P", "block": "main", "arm": pa, "world": "canon", "cells": sz["test"], "executions": sz["test"]})
        out.append({"group": "control_P", "block": "main", "arm": pa, "world": "reloc", "cells": sz["relocation"], "executions": sz["relocation"]})
    for ra in REPLAY_ARMS:
        src = C.arm_info(ra)["planner"].split(":", 1)[1]
        for c in cs:
            if c["arm"] != src or plan_only(c["arm"]):
                continue
            rc = dict(c, arm=ra)
            out.append({"group": "control_replay", "block": c["block"], "arm": ra, "world": "reloc" if c["block"] == "reloc" else "canon",
                        "cells": c["calls"], "executions": _cell_exec(rc, sz, runs, repeat_exec, plan_only)})
    return out


def _pre_a27_repeat_exec(arm: str) -> bool:
    """execute.build_units before A27: every non-reference arm of blocks main / repeat on a REPEAT item ran 3 times (incl. FS, X0 / X+M, /p)."""
    return C.arm_info(arm)["planner"] != "ref" if not arm.startswith("X-") else False


def dev_sharing_by_block(root: str | None = None) -> dict:
    """Measured units per cell-execution, BY BLOCK, in the dev rehearsal's own units.json (rehearsal-v1: Gemma + P + /p arms only).
    A ratio is (distinct unit-runs) / (cell-runs) of that block. Not a test-set number; blocks the rehearsal did not run are absent."""
    p = os.path.join(C.ns_docs("rehearsal-v1", root), "units.json")
    try:
        with open(p, encoding="utf-8") as f:
            u = json.load(f)["units"]
    except (OSError, ValueError, KeyError):
        return {"available": False}
    cell_runs, unit_runs = {}, {}
    for x in u.values():
        bl = sorted({c["block"] for c in x.get("cells") or ()})
        b = bl[0] if len(bl) == 1 else "+".join(bl)
        n = int(x.get("runs") or 1)
        unit_runs[b] = unit_runs.get(b, 0) + n
        cell_runs[b] = cell_runs.get(b, 0) + n * len(x.get("cells") or ())
    return {"available": True, "source": os.path.relpath(p, C.paths(root)["root"]).replace("\\", "/"),
            "ratio": {b: round(unit_runs[b] / cell_runs[b], 3) for b in sorted(cell_runs)}, "cell_runs": cell_runs, "unit_runs": unit_runs,
            "note": "dev rehearsal (24 dev presentations, Gemma + reference + replay arms, no Opus / Codex / REPEAT block / relocation); "
                    "the test set will share differently"}


def execution_enumeration(sz: dict | None = None, *, wall_s: float = 19.0, uniform_ratio: float = 0.778, root: str | None = None) -> dict:
    """A27: the REAL execution units of the final test, before and after A27, including the executor's default P0 / P+M reference controls and
    the G+M/p / O+M/p replay controls (Astra, runtime-cuts.txt: "enumerate actual units including controls"). Counts are executions BEFORE dedup
    (each plan cell its own unit); three dedup assumptions follow, all labelled: none (an upper bound), Astra's uniform 0.778 (the dev
    rehearsal's overall units-per-cell), and the dev rehearsal's per-block ratios (blocks it did not run keep 1.0). Pure apart from reading
    the split-manifest sizes and the rehearsal units.json."""
    sz = sizes() if sz is None else sz
    cs = cells(sz)

    def table(cell_rows, runs, repeat_exec, plan_only, codex_exec_once=False):
        rows = []
        for c in cell_rows:
            po = plan_only(c["arm"]) and not codex_exec_once
            ex = 0 if po else _cell_exec(c, sz, runs, repeat_exec, lambda a: False)
            rows.append({"group": "plan_arm", "block": c["block"], "arm": c["arm"], "world": "reloc" if c["block"] == "reloc" else "canon",
                         "cells": c["calls"], "executions": ex, "plan_only": bool(po)})
        rows += _controls(sz, cell_rows, runs, repeat_exec, plan_only if not codex_exec_once else (lambda a: False))
        return rows

    after = table(cs, REPEAT_RUNS, C.repeat_exec_arm, C.plan_only)
    before = table(_pre_a27_cells(sz), 3, _pre_a27_repeat_exec, C.plan_only, codex_exec_once=True)
    dev = dev_sharing_by_block(root)
    dev_ratio = (dev.get("ratio") or {}) if dev.get("available") else {}

    def summary(rows):
        by_block, by_group = {}, {}
        for r in rows:
            key = "reloc" if r["world"] == "reloc" else r["block"]
            by_block[key] = by_block.get(key, 0) + r["executions"]
            by_group[r["group"]] = by_group.get(r["group"], 0) + r["executions"]
        n = sum(by_block.values())
        listed = by_group.get("plan_arm", 0)
        per_block_dev = sum(v * dev_ratio.get(b if b in ("main", "curve") else "_none", 1.0) for b, v in by_block.items())
        def h(x):
            return round(x * wall_s / 3600, 1)
        return {"executions_by_block": by_block, "executions_by_group": by_group, "plan_cells": sum(r["cells"] for r in rows if r["group"] == "plan_arm"),
                "plan_only_cells": sum(r["cells"] for r in rows if r.get("plan_only")),
                "listed_matrix": {"episodes": listed, "hours": h(listed)},
                "with_controls": {"episodes": n, "hours": h(n)},
                "dedup": {"none": {"episodes": n, "hours": h(n)},
                          "uniform_%s" % uniform_ratio: {"episodes": round(n * uniform_ratio), "hours": h(n * uniform_ratio)},
                          "dev_per_block": {"episodes": round(per_block_dev), "hours": h(per_block_dev),
                                            "ratios_used": {b: dev_ratio.get(b if b in ("main", "curve") else "_none", 1.0) for b in sorted(by_block)}}}}

    return {"schema": "ftm.runtime-a27/1", "runtime_plan": C.RUNTIME_PLAN, "status": "DRAFT, PENDING ARASH SIGN-OFF AT THE FREEZE",
            "decision": "Arash 2026-10-13: \"Accept astra's vision. it's logical.\" (docs/final-test/_astra/runtime-cuts.txt; ledger 'ARASH DECISION: accept Astra's run-time plan')",
            "sizes": {k: v for k, v in sz.items() if k != "dev"}, "episode_wall_s": wall_s,
            "policy": {"after": {"repeat_planner_arms": list(C.REPEAT_ARMS), "repeat_rs": [1, 2], "exec_runs_repeat": REPEAT_RUNS,
                                 "repeat_exec_rule": "units with a core-four or core /p cell of block main/repeat on a REPEAT item",
                                 "curve_snapshots": ["o%dk%d" % ok for ok in CURVE_SNAPSHOTS], "codex_exploratory": "plan-only (A25 item 3)",
                                 "reloc": "unchanged", "dedup": "unchanged (exact execution-unit key, A21 D)", "controls": "P0, P+M, G+M/p, O+M/p (execute defaults)"},
                       "before": {"repeat_planner_arms": list(C.main_arms_for(NS)), "repeat_rs": [1, 2], "exec_runs_repeat": 3,
                                  "repeat_exec_rule": "every non-reference unit of block main/repeat on a REPEAT item (incl. FS, X0 / X+M, /p)",
                                  "curve_snapshots": ["o%dk%d" % ok for ok in CURVE_SNAPSHOTS_PRE_A27], "codex_exploratory": "executed once per cell"}},
            "after": {"summary": summary(after), "rows": after}, "before": {"summary": summary(before), "rows": before},
            "dev_sharing_by_block": dev,
            "scope": {"counted": "the test-v1 execution matrix only: plan arms (main, core REPEAT, curve, relocation) + P0 / P+M + G+M/p, O+M/p",
                      "not_counted": ["practice (snapshots K = 1, 3, 5, 10 of order 0)", "the acceptance gate and its possible complete re-run",
                                      "the non-test dress rehearsal and the sustained thermal rehearsal", "Phase A capture (its own timing)",
                                      "the G-iso audit (Astra P-A27: a full default G-iso adds about 36 executions, ~0.19 h)",
                                      "crash / blocked_state re-runs and hang-guard relaunches"],
                      "usd_vs_episodes": "the cost-gate USD projection includes acceptance and rehearsal allowances; these episode counts do not"},
            "repeat_overhead": "the A27 1,200 = additional REPEAT execution overhead of the core four: 240 extra main-block executions (4 arms x 60 x 1) "
                               "+ 960 repeat-block executions (4 x 120 cells x 2)",
            "notes": ["counts are executions before dedup unless a dedup scenario says otherwise; each plan cell is its own unit before dedup",
                      "19 s per episode is the practice-v1 mean (533 s / 28); the test-v1 mean, tails, retries and restore failures are unmeasured",
                      "re-runs after crash / blocked_state (attempts <= 3) and hang-guard relaunches are not counted",
                      "planner time is not simulator time (plan-first finishes all planning before any execution)",
                      "a shared execution is a benchmark saving, not cheaper independent deployment, and a shared observation, never an independent replication (Astra)",
                      "'before' = the pre-A27 implementation baseline (A25 item 3 had already declared the exploratory Codex models plan-only; the code still executed them)",
                      "before freezing: measure dedup by block, profile reset / sweep / navigation / detection, sustained thermal rehearsal (Astra)"]}


# ================================================================================================ final cost projection (A21 G)
#: USD per million tokens, list price (sim/planners/prices.json, as_of 2026-10-06): input, output. Thinking tokens are billed as output.
PRICE = {"O": (4.0, 20.0), "S": (2.0, 10.0), "H": (1.0, 5.0)}      # S and H: ARCHIVED (A23), kept for a future namespace
#: The scenarios. `tail` = extra billed output tokens per call (reasoning); `repair` = share of calls that need the repair turn (a second full
#: call of its own); `failed` = share of extra billed calls that give no usable answer (transport failures after generation, timeouts, ambiguous 5xx
#: that were billed); `block` = tokens added to every +M prompt (S* / curve) and to every +FS prompt ('measured' = the real-token gate's means).
SCENARIOS = {
    "S1 pilot-like (no thinking tail)": {"tail": 0, "repair": 0.15, "failed": 0.03, "block": "measured"},
    "S2 reference (A15 / A18: +500 tail tokens per call)": {"tail": 500, "repair": 0.15, "failed": 0.03, "block": "measured"},
    "S3 stress (+1000 tail, 25 % repairs, 6 % failed billable calls, 800-token blocks)": {"tail": 1000, "repair": 0.25, "failed": 0.06, "block": "max"},
}
BLOCK_FALLBACK = {"memory": 160, "fs": 759, "max": 800}


def measured_blocks(root: str | None = None) -> dict:
    """Mean Anthropic-counted tokens of the non-empty S* / curve blocks and of the FS blocks (freeze_tokens.json, written by the real-token gate)."""
    p = os.path.join(C.paths(root)["matrix_docs"], "freeze_tokens.json")
    try:
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
    except (OSError, ValueError, KeyError):
        return dict(BLOCK_FALLBACK, source="fallback constants")
    mem = [r["counts"]["anthropic"] for r in rows if r.get("source") not in ("FS",) and r.get("counts", {}).get("anthropic")]
    fs = [r["counts"]["anthropic"] for r in rows if r.get("source") == "FS" and r.get("counts", {}).get("anthropic")]
    return {"memory": round(sum(mem) / len(mem)) if mem else BLOCK_FALLBACK["memory"], "fs": round(sum(fs) / len(fs)) if fs else BLOCK_FALLBACK["fs"],
            "max": 800, "source": "freeze_tokens.json"}


def observed_sharing(root: str | None = None) -> dict:
    """Indicative unit sharing from the dev rehearsal's own units.json (plan cells vs distinct execution units, Gemma plans only). Not a test-set number."""
    p = os.path.join(C.ns_docs("rehearsal-v1", root), "units.json")
    try:
        with open(p, encoding="utf-8") as f:
            u = json.load(f)["units"]
    except (OSError, ValueError, KeyError):
        return {"available": False}
    n_cells = sum(len(x.get("cells") or ()) for x in u.values())
    return {"available": True, "plan_cells": n_cells, "units": len(u), "units_per_cell": round(len(u) / n_cells, 3) if n_cells else None,
            "note": "dev rehearsal, Gemma plans + reference arms; the test set shares differently (more arms, 3 Anthropic models, Codex)"}


def cell_usd(c: dict, sc: dict, blocks: dict) -> float:
    """USD of ONE planned call of cell-row `c` under scenario `sc`: the post-A11 prior (it already holds the pilot's repair share) x (1 + failed),
    plus the block tokens the prior does not contain, plus the thinking tail on every call incl. the repair turns."""
    m = c["model_key"]
    if m not in PRICE:
        return 0.0
    pin, pout = PRICE[m]
    fs = c["arm"].endswith("+FS")
    blk = 0
    if fs:
        blk = blocks["max"] if sc["block"] == "max" else blocks["fs"]
    elif c["memory"]:
        blk = blocks["max"] if sc["block"] == "max" else blocks["memory"]
    base = PRIOR_USD_PER_CELL[m] * (1 + sc["failed"]) + blk * pin / 1e6
    tail = sc["tail"] * pout / 1e6 * (1 + sc["repair"]) * (1 + sc["failed"])
    return base + tail


def final_projection(sz: dict | None = None, cs: list | None = None, blocks: dict | None = None, treatment: str | None = None) -> dict:
    """The final cost projection with the final roster (A23: 8 main arms incl. REPEAT r = 1, 2 and the learning curve and relocation blocks), prompt sizes,
    reasoning tails, repairs and failed billable calls, per scenario and ledger scope, against the test-v1 ceilings (core 30 until A24.1, total = the 100 USD cap of A24.10; see `cost_gate` for the 80 USD gate), and what the registered cut
    order (codex, curve12, reloc; A23 removed haiku) would have to remove, in that order, to fit. Codex, Gemma and the reference/replay arms cost 0 USD.
    C4-open: under ft-v2 (``treatment`` or the flipped treatment.TEST_V1_TREATMENT) the 3-call model of ``final_projection_ft2`` is used."""
    if (treatment or test_treatment()) == FT2_TREATMENT:
        return final_projection_ft2(sz, cs, None if blocks is None else blocks)
    sz = sizes() if sz is None else sz
    cs = cells(sz) if cs is None else cs
    blocks = measured_blocks() if blocks is None else blocks
    ce = C.NAMESPACES["test-v1"]["ceilings"]
    out = {"ceilings": ce, "cap_usd": C.ANTHROPIC_CAP_USD, "gate_usd": C.COST_GATE_USD, "blocks_tokens": blocks, "assumptions": {"price_per_M_tokens_input_output": PRICE, "prior_usd_per_cell": PRIOR_USD_PER_CELL,
                                                                  "calls": "first-pass calls of callplan.cells(); repairs and failed calls are rates on top",
                                                                  "unit_dedup": "not modelled for USD (dedup saves simulator time, not planner calls)"},
           "scenarios": {}}
    for name, sc in SCENARIOS.items():
        usd = {"core": 0.0, "other": 0.0}
        by_group = {g: 0.0 for g in C.CUT_ORDER}
        by_model = {"O": 0.0}                                         # A23: Opus is the only paid model in test-v1 (S / H appear only if a cell carries them)
        for c in cs:
            if c["scope"] not in usd:
                continue
            u = c["calls"] * cell_usd(c, sc, blocks)
            usd[c["scope"]] += u
            by_model[c["model_key"]] = by_model.get(c["model_key"], 0.0) + u
            g = C.cut_group(c["arm"], "reloc:x:en" if c["block"] == "reloc" else None)
            if g in by_group:
                by_group[g] += u
        total = usd["core"] + usd["other"]
        need, removed, cut = total - ce["total"], 0.0, []
        for g in C.CUT_ORDER:
            if need - removed <= 1e-9:
                break
            removed += by_group[g]
            cut.append(g)
        fits_after = total - removed <= ce["total"] + 1e-9
        out["scenarios"][name] = {
            "usd_core": round(usd["core"], 2), "usd_other": round(usd["other"], 2), "usd_total": round(total, 2), "by_model": {k: round(v, 2) for k, v in by_model.items()},
            "usd_by_cuttable_group": {k: round(v, 2) for k, v in by_group.items()},
            "core_margin": round(ce["core"] - usd["core"], 2), "total_margin": round(ce["total"] - total, 2),
            "core_within_ceiling": usd["core"] <= ce["core"], "total_within_ceiling": total <= ce["total"],
            "cut_needed_in_order": cut if need > 1e-9 else [], "usd_removed_by_that_cut": round(removed, 2) if need > 1e-9 else 0.0,
            "fits_after_cut": bool(fits_after) if need > 1e-9 else True,
            "core_is_never_cut": "if usd_core exceeds its ceiling no cut helps: the study would be reported incomplete (prereg s13)"}
    ep = projection(sz, cs)
    sh = observed_sharing()
    est = round(ep["hours_upper_bound"] * sh["units_per_cell"], 1) if sh.get("units_per_cell") else None
    out["simulator"] = {"episodes_upper_bound": ep["episodes_upper_bound"], "hours_upper_bound": ep["hours_upper_bound"], "episode_wall_s": ep["episode_wall_s"],
                        "dev_observed_sharing": sh, "hours_if_test_shares_like_dev": est,
                        "note": "upper bound without unit dedup; plan-first mode removes about 1-2 h; a gpt-6-sol call took 7-8 s wall in the dev smoke (planner time, not USD)"}
    out["x_main_arms"] = ep["x_main_arms"]
    return out


def remaining(namespace: str, root: str | None = None) -> dict:
    """Not-yet-terminal paid cells per planner and the running mean cost per cell, from the namespace's own journals (reads only)."""
    import plan as PLN                                            # lazy: pulls the planner layer
    pl = PLN.Plan(namespace, passes=PLN.PASSES, root=root, dry_run=True, gpu_sample=False, out=lambda m: None)
    return pl.remaining_report()


# ================================================================================================ cost gate (A24.10)
PACKAGE_MAX_BLOCKS = {"memory": 2000, "fs": 2000, "max": 2000, "source": "A24 package hard maximum (2,000 tokens); FS has the same maximum"}
GATE_SCENARIO = "S2 reference (A15 / A18: +500 tail tokens per call)"


def cost_gate(spent_usd: float = 0.0, sz: dict | None = None, cs: list | None = None, blocks: dict | None = None, *,
              acceptance_presentations: int = 48, acceptance_rerun: bool = True, rehearsal_fraction: float = 0.1, practice_usd: float = 0.0,
              treatment: str | None = None, tokens: dict | None = None) -> dict:
    """A24.10 cost gate before the first test call: ``spent_usd`` (EVERY Anthropic dollar so far, historical Sonnet / Haiku charges included) plus the
    remaining-cost projection under the A21.G S2 scenario (a sensitivity scenario, not an upper bound) must be <= C.COST_GATE_USD (80), which leaves
    C.CONTINGENCY_USD (20) inside the C.ANTHROPIC_CAP_USD (100) cap. The projection covers: practice (``practice_usd``; practice is Gemma-only, so 0),
    the acceptance gate (``acceptance_presentations`` presentations on every PAID main arm; a possible complete re-run doubles it), the rehearsal
    (``rehearsal_fraction`` of the retained plan; a labelled assumption), every retained block of the call plan, with repair turns and billed reasoning
    inside S2. ``worst_two_repairs_usd`` is a SENSITIVITY line (every call needs 2 repair turns), reported, not gated. If the gate fails, ``cut_needed_in_order``
    lists the registered cut groups (codex, curve12, reloc) that would have to go, in order; the core four, Gemma, FS and X0 / X+M are never cut.
    Archived arms (Sonnet / Haiku) have no row in test-v1, so they add nothing.
    C4-open: when the test treatment is ft-v2 (``treatment`` or treatment.TEST_V1_TREATMENT after its flip) the cells are priced at up to 3 calls
    each (first + 2 repairs, a growing conversation; ``cost_gate_ft2``); ft-v1 keeps this function's original numbers byte for byte."""
    if (treatment or test_treatment()) == FT2_TREATMENT:
        return cost_gate_ft2(spent_usd, sz, cs, blocks, acceptance_presentations=acceptance_presentations, acceptance_rerun=acceptance_rerun,
                             rehearsal_fraction=rehearsal_fraction, practice_usd=practice_usd, tokens=tokens)
    sz = sizes() if sz is None else sz
    cs = cells(sz) if cs is None else cs
    blocks = dict(PACKAGE_MAX_BLOCKS) if blocks is None else blocks          # the gate prices every +M / +FS prompt at the package's hard maximum
    sc = SCENARIOS[GATE_SCENARIO]
    plan = {"core": 0.0, "other": 0.0}
    by_group = {g: 0.0 for g in C.CUT_ORDER}
    for c in cs:
        if c["scope"] not in plan:
            continue
        u = c["calls"] * cell_usd(c, sc, blocks)
        plan[c["scope"]] += u
        g = C.cut_group(c["arm"], "reloc:x:en" if c["block"] == "reloc" else None)
        if g in by_group:
            by_group[g] += u
    retained = plan["core"] + plan["other"]
    paid_main = [c for c in cs if c["block"] == "main" and c["scope"] in plan]
    acc_one = sum(acceptance_presentations * cell_usd(c, sc, blocks) for c in paid_main)
    acc = acc_one * (2 if acceptance_rerun else 1)
    reh = rehearsal_fraction * retained
    worst2 = sum(c["calls"] * cell_usd(c, dict(sc, repair=2.0), blocks) for c in cs if c["scope"] in plan)
    remaining = retained + acc + reh + practice_usd
    total = float(spent_usd) + remaining
    need, removed, cut = total - C.COST_GATE_USD, 0.0, []
    for g in C.CUT_ORDER:
        if need - removed <= 1e-9:
            break
        removed += by_group[g]
        cut.append(g)
    ok = total <= C.COST_GATE_USD + 1e-9
    fits_after = ok or (total - removed <= C.COST_GATE_USD + 1e-9)
    return {"scenario": GATE_SCENARIO, "cap_usd": C.ANTHROPIC_CAP_USD, "gate_usd": C.COST_GATE_USD, "contingency_usd": C.CONTINGENCY_USD,
            "spent_usd": round(float(spent_usd), 2),
            "remaining": {"retained_blocks": round(retained, 2), "acceptance": round(acc, 2), "acceptance_rerun_included": bool(acceptance_rerun),
                          "rehearsal": round(reh, 2), "practice": round(practice_usd, 2), "total": round(remaining, 2)},
            "projected_total_usd": round(total, 2), "gate_pass": bool(ok), "margin_to_gate_usd": round(C.COST_GATE_USD - total, 2),
            "within_cap_at_projection": total <= C.ANTHROPIC_CAP_USD,
            "usd_by_cuttable_group": {k: round(v, 2) for k, v in by_group.items()},
            "cut_needed_in_order": cut if not ok else [], "usd_removed_by_that_cut": round(removed, 2) if not ok else 0.0,
            "fits_after_cut": bool(fits_after),
            "if_cannot_fit": "testing does not begin under this protocol (A24.10); the core four, Gemma, FS and X0 / X+M are never cut",
            "worst_two_repairs_usd": round(float(spent_usd) + worst2, 2),
            "worst_two_repairs_note": "sensitivity only: the retained blocks if EVERY call needed 2 repair turns, plus spend; not gated",
            "blocks_tokens": blocks, "assumptions": {"price_per_M_tokens_input_output": PRICE, "rehearsal_fraction": rehearsal_fraction,
                            "acceptance": f"{acceptance_presentations} presentations on each paid main arm ({len(paid_main)} arms)",
                            "historical_spend": "pass the real ledger total as spent_usd; this function does not read it"}}


# ================================================================================================ C4-open (b): ft-v2 = up to 3 calls per cell
# A24 item 2: at most 2 repair re-prompts; A24 item 10: the cost gate covers "up to 2 repair turns per cell and billed reasoning" (S2).
# Astra C4-open Q-b: a scenario model (NOT a spending bound). One call k of a cell, k = 1..3:
#     c_k = [p_in * (I0 + B + (k-1) * (R + F + P)) + p_out * (A + tail)] / 1e6
# (the conversation grows by the replayed reply R, the repair frame F and the problem text P per repair; the tail is billed on EVERY call).
# Expected USD per cell = [c_1 + p1 * c_2 + p1 * p2 * c_3] * (1 + failed), with failed = the share of EXTRA billed workload that gives no usable
# answer (one multiplier on the whole cell, not a per-attempt retry probability). Full uncached input price (no cache-read discount).
FT2_TREATMENT = "ft-v2"
FT2_REPAIRS = 2
#: per-call token priors (Opus tokenizer), PROVISIONAL (Astra): dev ft-v2 signal rounds, dev items only; those prompts preceded the sweep, so I0 must be
#: recounted on the final post-sweep ft-v2 requests (A24.1) and passed as `tokens` before the gate is used. A = billed output incl. adaptive thinking
#: (dev max 284; not an output bound); R = the replayed visible reply (the provider keeps no thinking blocks in `raw`): R = A is a conservative proxy;
#: F = the REPAIR_V2 frame; P = the problem text at plan.REPAIR_PROBLEM_CHARS (600 chars / 4): an ESTIMATE while the runner does not cap it.
FT2_TOKENS = {"I0": 2953, "A": 284, "R": 284, "F": 40, "P": 150,
              "status": "provisional development priors (signal rounds 1-4, ft-v2, dev items; pre-sweep prompts): recount I0 on the final ft-v2 captures",
              "source": {"I0": "max Opus input of O0 cells, sim/final_test/signal/plans_round*.jsonl", "A": "max Opus output (billed, incl. thinking)",
                         "R": "= A (proxy)", "F": "REPAIR_V2 frame", "P": "plan.REPAIR_PROBLEM_CHARS / 4 (the runner does not cap the problem text)"}}
#: p2 = share of FIRST-repaired cells that need a SECOND repair; p1 = the scenario's existing `repair` share. Scenario assumptions fixed before acceptance.
FT2_P2 = {"S1 pilot-like (no thinking tail)": 0.25, "S2 reference (A15 / A18: +500 tail tokens per call)": 0.5,
          "S3 stress (+1000 tail, 25 % repairs, 6 % failed billable calls, 800-token blocks)": 1.0}
#: ft-v2 block tokens when no ft-v2 capture has been measured: the A24 maximum for both the package and FS (conservative)
FT2_BLOCK_FALLBACK = {"memory": 2000, "fs": 2000, "max": 2000, "source": "A24 2,000-token maximum (no ft-v2 capture measured yet)"}
#: the longer-repair-output sensitivity (Astra): output tokens on the REPAIR calls multiplied by this factor
FT2_REPAIR_OUTPUT_X = 3.0


def test_treatment() -> str:
    """The treatment the locked test runs under (treatment.TEST_V1_TREATMENT; ft-v1 until the A24.1 flip)."""
    import treatment as T                                         # lazy: pure python, no simulator
    return T.TEST_V1_TREATMENT


def ft2_tokens(tokens: dict | None = None) -> dict:
    t = dict(FT2_TOKENS)
    t.update(tokens or {})
    return t


def _ft2_block(c: dict, sc: dict, blocks: dict) -> int:
    if c["arm"].endswith("+FS"):
        return int(blocks["max"] if sc["block"] == "max" else blocks["fs"])
    if c["memory"]:
        return int(blocks["max"] if sc["block"] == "max" else blocks["memory"])
    return 0


def ft2_call_usd(k: int, m: str, block: int, sc: dict, tok: dict, out_x: float = 1.0) -> float:
    """USD of call k (1-based) of one cell of model key m."""
    if m not in PRICE:
        return 0.0
    pin, pout = PRICE[m]
    tin = tok["I0"] + block + (k - 1) * (tok["R"] + tok["F"] + tok["P"])
    tout = (tok["A"] * (out_x if k > 1 else 1.0)) + sc["tail"]
    return (pin * tin + pout * tout) / 1e6


def ft2_cell_usd(c: dict, sc: dict, blocks: dict, tok: dict | None = None, *, worst: bool = False, out_x: float = 1.0,
                 p2: float | None = None) -> float:
    """Expected USD of ONE planned cell (worst=True: every cell makes all 3 calls; a sensitivity, not a token-maximum bound)."""
    tok = ft2_tokens(tok)
    m = c["model_key"]
    if m not in PRICE:
        return 0.0
    b = _ft2_block(c, sc, blocks)
    calls = [ft2_call_usd(k, m, b, sc, tok, out_x) for k in range(1, FT2_REPAIRS + 2)]
    if worst:
        return sum(calls) * (1 + sc["failed"])
    p1 = float(sc["repair"])
    q2 = float(FT2_P2.get(sc.get("_name", ""), 0.5) if p2 is None else p2)
    return (calls[0] + p1 * calls[1] + p1 * q2 * calls[2]) * (1 + sc["failed"])


def ft2_expected_calls(sc_name: str) -> float:
    sc = SCENARIOS[sc_name]
    return 1 + sc["repair"] + sc["repair"] * FT2_P2[sc_name]


def ft2_measured_blocks(namespace: str, root: str | None = None) -> dict:
    """Mean Anthropic-counted block tokens of an ft-v2 namespace's run-1 captures, per kind (memory = S* / curve, fs = FS), weighted by
    presentation x source and counting an EMPTY block as 0 (Astra). Falls back to FT2_BLOCK_FALLBACK when nothing is measured."""
    try:
        rows = C.Journal(C.journal_path(namespace, "capture", root)).rows()
    except Exception:                                             # noqa: BLE001  (no journal yet)
        rows = []
    mem, fs = [], []
    for r in rows:
        if r.get("status") not in (None, "ok") or int(r.get("capture_run", 1) or 1) != 1 or r.get("treatment") != FT2_TREATMENT:
            continue
        for src, b in (r.get("blocks") or {}).items():
            if src == "none":
                continue
            n = int((b.get("counts") or {}).get("anthropic") or 0) if b.get("text") else 0
            (fs if src == "FS" else mem).append(n)
    if not mem and not fs:
        return dict(FT2_BLOCK_FALLBACK)
    return {"memory": round(sum(mem) / len(mem)) if mem else FT2_BLOCK_FALLBACK["memory"],
            "fs": round(sum(fs) / len(fs)) if fs else FT2_BLOCK_FALLBACK["fs"], "max": 2000,
            "source": "ft-v2 captures of %s (%d memory, %d FS blocks; empty = 0)" % (namespace, len(mem), len(fs))}


def _ft2_workload(cs, sc, blocks, tok, price, acceptance_presentations, acceptance_rerun, rehearsal_fraction, practice_usd):
    plan_ = {"core": 0.0, "other": 0.0}
    by_group = {g: 0.0 for g in C.CUT_ORDER}
    for c in cs:
        if c["scope"] not in plan_:
            continue
        u = c["calls"] * price(c)
        plan_[c["scope"]] += u
        g = C.cut_group(c["arm"], "reloc:x:en" if c["block"] == "reloc" else None)
        if g in by_group:
            by_group[g] += u
    retained = plan_["core"] + plan_["other"]
    paid_main = [c for c in cs if c["block"] == "main" and c["scope"] in plan_]
    acc = sum(acceptance_presentations * price(c) for c in paid_main) * (2 if acceptance_rerun else 1)
    reh = rehearsal_fraction * retained
    return {"plan": plan_, "by_group": by_group, "retained": retained, "acceptance": acc, "rehearsal": reh, "practice": practice_usd,
            "total": retained + acc + reh + practice_usd, "paid_main": len(paid_main)}


def cost_gate_ft2(spent_usd: float = 0.0, sz: dict | None = None, cs: list | None = None, blocks: dict | None = None, *,
                  acceptance_presentations: int = 48, acceptance_rerun: bool = True, rehearsal_fraction: float = 0.1, practice_usd: float = 0.0,
                  tokens: dict | None = None) -> dict:
    """A24.10 cost gate under ft-v2: same workload terms as cost_gate (retained blocks, acceptance x2 with the re-run, rehearsal fraction,
    practice), every cell at up to 3 calls (S2 expected value, the gate), plus two SENSITIVITIES on the SAME remaining workload (reported, not
    gated): every cell makes all 3 calls; repair calls with 3x the output. Blocks default to the 2,000-token maximum for every +M / FS prompt."""
    sz = sizes() if sz is None else sz
    cs = cells(sz) if cs is None else cs
    blocks = dict(FT2_BLOCK_FALLBACK) if blocks is None else blocks
    tok = ft2_tokens(tokens)
    sc = dict(SCENARIOS[GATE_SCENARIO], _name=GATE_SCENARIO)
    args = (acceptance_presentations, acceptance_rerun, rehearsal_fraction, practice_usd)
    w = _ft2_workload(cs, sc, blocks, tok, lambda c: ft2_cell_usd(c, sc, blocks, tok), *args)
    w3 = _ft2_workload(cs, sc, blocks, tok, lambda c: ft2_cell_usd(c, sc, blocks, tok, worst=True), *args)
    wx = _ft2_workload(cs, sc, blocks, tok, lambda c: ft2_cell_usd(c, sc, blocks, tok, out_x=FT2_REPAIR_OUTPUT_X), *args)
    total = float(spent_usd) + w["total"]
    need, removed, cut = total - C.COST_GATE_USD, 0.0, []
    for g in C.CUT_ORDER:
        if need - removed <= 1e-9:
            break
        removed += w["by_group"][g]
        cut.append(g)
    ok = total <= C.COST_GATE_USD + 1e-9
    fits_after = ok or (total - removed <= C.COST_GATE_USD + 1e-9)
    return {"treatment": FT2_TREATMENT, "cost_model": "ft-v2: up to 3 calls per cell (first + 2 repairs), growing conversation (C4-open)",
            "scenario": GATE_SCENARIO, "cap_usd": C.ANTHROPIC_CAP_USD, "gate_usd": C.COST_GATE_USD, "contingency_usd": C.CONTINGENCY_USD,
            "spent_usd": round(float(spent_usd), 2),
            "remaining": {"retained_blocks": round(w["retained"], 2), "acceptance": round(w["acceptance"], 2), "acceptance_rerun_included": bool(acceptance_rerun),
                          "rehearsal": round(w["rehearsal"], 2), "practice": round(practice_usd, 2), "total": round(w["total"], 2)},
            "projected_total_usd": round(total, 2), "gate_pass": bool(ok), "margin_to_gate_usd": round(C.COST_GATE_USD - total, 2),
            "within_cap_at_projection": total <= C.ANTHROPIC_CAP_USD,
            "usd_by_cuttable_group": {k: round(v, 2) for k, v in w["by_group"].items()},
            "cut_needed_in_order": cut if not ok else [], "usd_removed_by_that_cut": round(removed, 2) if not ok else 0.0,
            "fits_after_cut": bool(fits_after),
            "if_cannot_fit": "testing does not begin under this protocol (A24.10); the core four, Gemma, FS and X0 / X+M are never cut",
            "expected_calls_per_cell": round(ft2_expected_calls(GATE_SCENARIO), 4),
            "sensitivity_every_cell_3_calls_usd": round(float(spent_usd) + w3["total"], 2),
            "sensitivity_repair_output_x%g_usd" % FT2_REPAIR_OUTPUT_X: round(float(spent_usd) + wx["total"], 2),
            "sensitivity_note": "same remaining workload as the gate (retained + acceptance with re-run + rehearsal + practice); reported, not gated; "
                                "not a maximum-token cost bound",
            "worst_two_repairs_usd": round(float(spent_usd) + w3["total"], 2),
            "blocks_tokens": blocks, "tokens": tok, "p2": FT2_P2[GATE_SCENARIO],
            "assumptions": {"price_per_M_tokens_input_output": PRICE, "rehearsal_fraction": rehearsal_fraction,
                            "acceptance": f"{acceptance_presentations} presentations on each paid main arm ({w['paid_main']} arms)",
                            "failed": "share of EXTRA billed workload without a usable answer (one multiplier per cell)",
                            "cache": "full uncached input price (no cache-read discount)",
                            "historical_spend": "pass the real ledger total as spent_usd; this function does not read it"}}


def final_projection_ft2(sz: dict | None = None, cs: list | None = None, blocks: dict | None = None, tokens: dict | None = None) -> dict:
    """final_projection's per-scenario table under the ft-v2 3-call model (blocks: measured ft-v2 means when given, else the maximum)."""
    sz = sizes() if sz is None else sz
    cs = cells(sz) if cs is None else cs
    blocks = dict(FT2_BLOCK_FALLBACK) if blocks is None else blocks
    tok = ft2_tokens(tokens)
    ce = C.NAMESPACES["test-v1"]["ceilings"]
    out = {"treatment": FT2_TREATMENT, "ceilings": ce, "cap_usd": C.ANTHROPIC_CAP_USD, "gate_usd": C.COST_GATE_USD, "blocks_tokens": blocks,
           "tokens": tok, "p2": FT2_P2, "scenarios": {}}
    for name, sc0 in SCENARIOS.items():
        sc = dict(sc0, _name=name)
        usd = {"core": 0.0, "other": 0.0}
        worst = {"core": 0.0, "other": 0.0}
        for c in cs:
            if c["scope"] in usd:
                usd[c["scope"]] += c["calls"] * ft2_cell_usd(c, sc, blocks, tok)
                worst[c["scope"]] += c["calls"] * ft2_cell_usd(c, sc, blocks, tok, worst=True)
        out["scenarios"][name] = {"usd_core": round(usd["core"], 2), "usd_other": round(usd["other"], 2), "usd_total": round(usd["core"] + usd["other"], 2),
                                  "expected_calls_per_cell": round(ft2_expected_calls(name), 4),
                                  "every_cell_3_calls_usd_total": round(worst["core"] + worst["other"], 2),
                                  "core_within_ceiling": usd["core"] <= ce["core"], "total_within_ceiling": usd["core"] + usd["other"] <= ce["total"]}
    return out


# ================================================================================================ BATCH-A29: the study-wide projection
# A29 (DRAFT): Kitchen 1 = the A27 call plan (cells()); Kitchens 2/3 (FloorPlan15 / FloorPlan17) = G0, G+M, O0, O+M at K = 10, order 0, r = 0
# only (no FS, REPEAT, curve, relocation, Codex) + the free P0 / P+M controls; every paid call through the Message Batches API at
# BATCH_PRICE_FACTOR x list (docs: "All usage is charged at 50% of the standard API prices"). The ft-v2 3-call model prices every cell.
BATCH_PRICE_FACTOR = 0.5
A29_K23_PRESENTATIONS = 250
A29_K23_FALLBACK = (120, 100)
A29_K23_ARMS = ("G0", "G+M", "O0", "O+M")
A29_K23_REHEARSAL_FRACTION = 0.1


def k23_cells(n_presentations: int, kitchen: str) -> list:
    """The A29 call plan of ONE replication kitchen: the core four on n presentations, r = 0 (first-pass calls only)."""
    out = []
    for a in A29_K23_ARMS:
        info = C.arm_info(a, NS)
        out.append({"block": "main", "label": f"{kitchen} main r=0 (A29)", "arm": a, "scope": _scope(a, "main"), "world": "canon",
                    "model_key": info["planner"][0], "presentations": n_presentations, "replicates": 1, "memory": info["prompt"] != "none",
                    "calls": n_presentations, "kitchen": kitchen})
    return out


def study_projection_a29(spent_usd: float = 0.0, n_k23: int = A29_K23_PRESENTATIONS, *, price_factor: float = BATCH_PRICE_FACTOR,
                         tokens: dict | None = None, blocks: dict | None = None, acceptance_presentations: int = 48,
                         acceptance_rerun: bool = True, rehearsal_fraction: float = 0.1,
                         k23_rehearsal_fraction: float = A29_K23_REHEARSAL_FRACTION) -> dict:
    """Spend so far + Kitchen 1 (A27 plan, acceptance x2, rehearsal) + Kitchens 2/3 (A29 scope, rehearsal fraction) under the ft-v2
    3-call model, every paid call at `price_factor` x list. Gate: the S2 total <= C.COST_GATE_USD. Sensitivities (reported, not gated):
    list price, every cell 3 calls, repair output x3, S1 / S3."""
    sz = sizes()
    k1 = cells(sz)
    blocks = dict(FT2_BLOCK_FALLBACK) if blocks is None else blocks
    tok = ft2_tokens(tokens)
    out = {"amendment": "A29 (DRAFT)", "price_factor": price_factor, "price_basis": "Message Batches API, 50 % of list (Opus 5.5 batch: 2 / 10 USD per MTok)",
           "spent_usd": round(float(spent_usd), 4), "kitchen1_presentations": sz["test"], "k23_presentations_each": n_k23,
           "blocks_tokens": blocks, "tokens": tok, "gate_usd": C.COST_GATE_USD, "cap_usd": C.ANTHROPIC_CAP_USD, "scenarios": {}}

    def price(sc, f, **kw):
        return lambda c: f * ft2_cell_usd(c, sc, blocks, tok, **kw)

    for name, sc0 in SCENARIOS.items():
        sc = dict(sc0, _name=name)
        row = {}
        for label, f, kw in (("batch", price_factor, {}), ("list_price", 1.0, {}), ("batch_every_cell_3_calls", price_factor, {"worst": True}),
                             ("batch_repair_output_x%g" % FT2_REPAIR_OUTPUT_X, price_factor, {"out_x": FT2_REPAIR_OUTPUT_X})):
            w1 = _ft2_workload(k1, sc, blocks, tok, price(sc, f, **kw), acceptance_presentations, acceptance_rerun, rehearsal_fraction, 0.0)
            w23 = [_ft2_workload(k23_cells(n_k23, k), sc, blocks, tok, price(sc, f, **kw), 0, False, k23_rehearsal_fraction, 0.0)
                   for k in ("FloorPlan15", "FloorPlan17")]
            tot = float(spent_usd) + w1["total"] + sum(w["total"] for w in w23)
            row[label] = {"kitchen1": {"retained": round(w1["retained"], 2), "acceptance": round(w1["acceptance"], 2), "rehearsal": round(w1["rehearsal"], 2),
                                       "total": round(w1["total"], 2)},
                          "kitchen2": round(w23[0]["total"], 2), "kitchen3": round(w23[1]["total"], 2),
                          "study_total_usd": round(tot, 2), "within_gate": tot <= C.COST_GATE_USD + 1e-9, "within_cap": tot <= C.ANTHROPIC_CAP_USD + 1e-9}
        out["scenarios"][name] = row
    g = out["scenarios"][GATE_SCENARIO]["batch"]
    out["gate"] = {"scenario": GATE_SCENARIO, "study_total_usd": g["study_total_usd"], "gate_pass": g["within_gate"],
                   "margin_to_gate_usd": round(C.COST_GATE_USD - g["study_total_usd"], 2)}
    out["assumptions"] = {"kitchen1": "callplan.cells() (A27 scope: 8 main arms, REPEAT core four, curve order 0 K = 1, 3, 5, relocation, Codex plan-only, free)",
                          "kitchens_2_3": "A29: G0, G+M, O0, O+M, r = 0 only; P0 / P+M free; Gemma practice free; no acceptance block; rehearsal = %g of retained" % k23_rehearsal_fraction,
                          "calls": "ft-v2: first + up to 2 repairs, growing conversation (cost_gate_ft2 model)", "blocks": "every +M / FS prompt at its block size",
                          "cache": "no prompt caching assumed (the batch body equals the sync body; it carries no cache_control)",
                          "spend": "pass every Anthropic dollar so far as spent_usd (historical charges included)"}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--study-a29" in argv:                                     # BATCH-A29: --study-a29 [--spent USD] [--k23 N] [--i0 TOKENS] [--a TOKENS] [--write]
        spent = float(argv[argv.index("--spent") + 1]) if "--spent" in argv else 0.0
        tk = {}
        if "--i0" in argv:
            tk["I0"] = int(argv[argv.index("--i0") + 1])
        if "--a" in argv:
            tk["A"] = tk["R"] = int(argv[argv.index("--a") + 1])
        ns_ = [int(argv[argv.index("--k23") + 1])] if "--k23" in argv else [A29_K23_PRESENTATIONS, *A29_K23_FALLBACK]
        res = {str(n): study_projection_a29(spent, n, tokens=tk or None) for n in ns_}
        if "--write" in argv:
            p = os.path.join(C.paths()["matrix_docs"], "study_projection_a29.json")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
                f.write("\n")
            print(f"wrote {p}")
        print(json.dumps({n: {**r["gate"], "list_price_total_usd": r["scenarios"][GATE_SCENARIO]["list_price"]["study_total_usd"],
                              "k1": r["scenarios"][GATE_SCENARIO]["batch"]["kitchen1"]["total"],
                              "k2": r["scenarios"][GATE_SCENARIO]["batch"]["kitchen2"], "k3": r["scenarios"][GATE_SCENARIO]["batch"]["kitchen3"]}
                          for n, r in res.items()}, indent=1))
        return 0 if res[str(ns_[0])]["gate"]["gate_pass"] else 3
    if "--remaining" in argv:
        i = argv.index("--remaining")
        root = argv[argv.index("--root") + 1] if "--root" in argv else None
        try:
            print(json.dumps(remaining(argv[i + 1], root), indent=1))
        except Exception as e:                                    # noqa: BLE001  (capture / plan not complete, unreadable journal)
            print(f"REFUSED: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
            return 2
        return 0
    sz = sizes()
    cs = cells(sz)
    if "--json" in argv:
        print(json.dumps({"sizes": sz, "cells": cs, "totals": totals(cs), "by_scope": by_scope(cs), "ceilings": C.NAMESPACES["test-v1"]["ceilings"], "projection": projection(sz, cs), "pilot": totals(pilot_cells(sz)),
                          "final_projection": final_projection(sz, cs)}, indent=1))
        return 0
    if "--cost-gate" in argv:                                     # A24.10: --cost-gate [--spent <usd of ALL Anthropic spend so far>]
        spent = float(argv[argv.index("--spent") + 1]) if "--spent" in argv else 0.0
        tr = argv[argv.index("--treatment") + 1] if "--treatment" in argv else None     # C4-open: --treatment ft-v2 (3 calls per cell)
        g = cost_gate(spent, sz, cs, treatment=tr)
        print(json.dumps(g, indent=1))
        return 0 if g["gate_pass"] else 3
    if "--runtime-a27" in argv:                                   # A27: real execution units incl. controls; --write -> docs/final-test/matrix/runtime_a27.json
        en = execution_enumeration(sz)
        if "--write" in argv:
            p = os.path.join(C.paths()["matrix_docs"], "runtime_a27.json")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(en, f, ensure_ascii=False, indent=1)
                f.write("\n")
            print(f"wrote {p}")
        print(json.dumps({k: en[k]["summary"] for k in ("before", "after")}, indent=1))
        return 0
    if "--final-projection" in argv:
        fp = final_projection(treatment=argv[argv.index("--treatment") + 1] if "--treatment" in argv else None)
        print(json.dumps(fp, indent=1))
        return 0
    print(f"sizes (presentations): {sz}")
    print(f"{'block':34s} {'arm':18s} {'scope':5s} {'pres':>5s} {'reps':>4s} {'calls':>6s}")
    for c in cs:
        print(f"{c['label']:34s} {c['arm']:18s} {c['scope']:5s} {c['presentations']:5d} {c['replicates']:4d} {c['calls']:6d}")
    print(f"calls by scope: {by_scope(cs)}")
    for k, d in totals(cs).items():
        print(f"model {k} ({MODEL_OF[k]}): {d['calls']} first-pass calls ({d['calls_mem']} with a block; "
              f"core-four primary incl. REPEAT {d['core_calls']})")
    print(f"projection: {projection(sz, cs)}")
    print(f"pilot: {totals(pilot_cells(sz))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
