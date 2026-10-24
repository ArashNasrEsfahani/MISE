"""Final-test matrix, Phase D (stage M7): the sealed scoring journal.  Contract: SPEC.md s6, s8, s10, s12 (M7).

Windows python, offline: no simulator, no planner call, no API.

What it does
  * Needs `phase_complete` for the plan phase and the exec phase (SPEC s6), except under `--partial`, which needs a recorded
    stopping-rule event (prereg s13 "incomplete study") and marks the seal `complete: false`.
  * Maps every plan cell (and every reference / replay cell the executor lists) to its execution unit and its sealed outcome.
    The mapping is the exec record's `cells` list (plan cell keys); if a record has none, the unit key is RECOMPUTED with
    `common.execution_unit_key` from the capture (summary sha), the snapshot manifest and the gold setup. A plan that was
    accepted but has no outcome is an integrity failure (fail fast), never a silent 0.
  * One score row per (cell, run). Run 0 is the scored one (`scored: true`); REPEAT runs 1 and 2 are kept for the variance
    decomposition. Rejected / infra / unexecuted / crashed / blocked plans score SR 0 with the flags set.
  * SR is simulator-goal success (prereg s8); strict SR = goal met AND every step ok; `unrealistic_slice` (A26 item 6 / s8 as
    amended) = an EXECUTED slice step whose recorded holding is not the ButterKnife (the only cutting tool), independent of goal
    success; under the A26 executor any such event is an executor fault (the historical rule, ``rule="legacy"``: goal met and a
    slice ran with no Knife / ButterKnife held, applies only to rounds 1-5 and step 4); PC / TDS on the generated plan (first pass and final) against the item's
    reference plan with `eval.pc_score` / `eval.tds_score`, labels `action:object:to:arg`, and `search_for look_around scan_for`
    stripped from BOTH sequences (prereg s8).
  * Appends `score.jsonl` (fsync'd, resumable) and writes `score.seal.json` {sha256 of score.jsonl, n, complete, created, ...}.
  * Owns gate 12.5 (`--evaluator-check`): reference plans on dev reproduce step 4 up to a declared tolerance; the empty plan scores 0.

Gold isolation: the gold dataset file (path from `common.paths`) is opened ONLY by `_gold_load`, and its content is used only here.
Nothing planner-side imports this module (the plan selftest checks that statically).  Test items are refused unless the SPEC s7
guard holds (`common.guard_items`).  No key is ever written (`common.key_leak` on every row).

Selftest: ``python sim/_ftm_score_selftest.py`` (synthetic data, offline).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402  (also puts sim/ on sys.path)
import treatment as T                                                    # noqa: E402  (C4: ft-v2 unit keys; inert in ft-v1 namespaces)
import a36 as A36                                                        # noqa: E402  (A36, signed: the excluded_reset flag)

STRIP_ACTIONS = frozenset({"search_for", "look_around", "scan_for"})     # prereg s8: stripped from BOTH sequences
KNIVES = frozenset({"knife", "butterknife"})            # legacy rule (rounds 1-5 / step 4 history only)
#: A26 item 2 / 6 (K-B2b 2026-10-13): the only cutting tool (honest_skills.CUTTING_TOOLS, lower case)
CUTTING_TOOLS = frozenset({"butterknife"})
SLICE_RULE = "a26"                                       # "legacy" = the old s8 definition (history only)
#: failure causes of an executed step (honest_skills.CAUSES; A26 added no_cutting_tool = slice refused without a ButterKnife)
STEP_CAUSES = ("not_detected", "wrong_target", "unreachable", "action_refused", "state_unrecognised", "retry_exhausted",
               "no_cutting_tool")
#: 12.5 pass rule (declared before any dev re-run; not tuned): at least this share of compared reference presentations must agree
#: with step 4's goal result (step 4 reports run-to-run variance; its own final regression was 40/40 on the re-run items).
EVAL_MIN_AGREEMENT = 0.80
STEP4_BASELINE = os.path.join("sim", "step4", "refplans_honest.jsonl")
SEAL_NAME = "score.seal.json"
#: A27 (draft) / A25 item 3: the exploratory Codex arms are plan-only. Their cells are scored on plan validity, PC and TDS in this file and never
#: enter score.jsonl, so no SR (and no 0) is ever reported for them and analyze.py's executed-arm tables cannot pick them up.
PLAN_DIAG_NAME = "plan_diagnostics.json"


def plan_diagnostics(rows: list, namespace: str) -> dict:
    """Plan-only rows -> {schema, rows, per_arm: {arm: n, valid rate, first-pass valid rate, mean PC / TDS (final plan, scored r = 0 and all r)}}.
    Pure. Descriptive only (never pooled with executed arms, not in any hypothesis)."""
    def mean(v):
        v = [float(x) for x in v if x is not None]
        return round(sum(v) / len(v), 4) if v else None
    per = {}
    for r in rows:
        per.setdefault(r["arm"], []).append(r)
    out = {}
    for a, rs in sorted(per.items()):
        r0 = [r for r in rs if r.get("r") == 0 and r.get("world") == "canon"]
        out[a] = {"n": len(rs), "n_r0": len(r0), "plan_valid": mean([1.0 if r.get("plan_valid") else 0.0 for r in rs]),
                  "first_ok": mean([1.0 if r.get("first_ok") else 0.0 for r in rs]),
                  "pc_final_r0": mean([r.get("pc_final") for r in r0]), "tds_final_r0": mean([r.get("tds_final") for r in r0]),
                  "pc_final_all_r": mean([r.get("pc_final") for r in rs]), "tds_final_all_r": mean([r.get("tds_final") for r in rs])}
    return {"schema": "ftm.plan-diagnostics/1", "spec": C.SPEC_VERSION, "namespace": namespace, "runtime_plan": C.RUNTIME_PLAN,
            "note": "exploratory Codex models: planning diagnostics only (A25 item 3, A27 draft); not executed, no SR, never pooled with API arms",
            "per_arm": out, "rows": sorted(rows, key=lambda r: r["key"])}


class ScoreRefused(RuntimeError):
    """A precondition of Phase D does not hold (phase not complete, hash mismatch, ...). Nothing was scored."""


class ScoreIntegrity(RuntimeError):
    """The journals contradict each other (an accepted plan with no outcome, an INVALID exec row, ...). Fail fast."""


# ================================================================================================ gold (the ONLY reader)
def _gold_load(root: str | None = None, dataset_path: str | None = None, verify: bool = True) -> tuple:
    """-> ({item id: {level, plan, setup}}, sha256 of the file). The one place the gold file is opened (and hashed). Verified
    against the manifest sha when asked."""
    p = dataset_path or C.paths(root)["dataset"]
    with open(p, "rb") as f:
        raw = f.read()
    got0 = got = hashlib.sha256(raw).hexdigest()
    if verify:
        man = C.load_manifest(root)
        want = (man.get("inclusion") or {}).get("dataset_sha256") or man.get("dataset_sha256")   # A26: the successor dataset
        if want and got != want:
            raise ScoreRefused(f"gold dataset sha {got[:12]} != manifest {str(want)[:12]}: the dataset changed after the split")
    doc = json.loads(raw.decode("utf-8"))
    out = {}
    for it in doc.get("items") or []:
        out[it["id"]] = {"level": it.get("level"), "plan": list(it.get("plan") or []), "setup": list(it.get("setup") or [])}
    return out, got0


# ================================================================================================ labels and metrics
def _eval():
    import eval as ev                                                    # sim/eval.py (lazy: heavy import)
    return ev


def step_label(step) -> str:
    """`action:object:to:arg` with the evaluator's own action / type normalisation (prereg s8)."""
    ev = _eval()
    a, o, t, g = (C.norm_plan([step]) or [[None] * 4])[0]
    return f"{ev.norm_action(a)}:{ev.canon_type(o) if o else ''}:{ev.canon_type(t) if t else ''}:{'' if g is None else str(g).strip().lower()}"


def labels_of(steps) -> list:
    ev = _eval()
    out = []
    for s in C.norm_plan(steps):
        if ev.norm_action(s[0]) in STRIP_ACTIONS:
            continue
        out.append(step_label(s))
    return out


def plan_metrics(gen_steps, ref_steps) -> dict:
    """PC (sequence) and TDS (set F1 + precision / recall) of a generated plan against the reference plan. None = no plan."""
    if gen_steps is None:
        return {"pc": None, "tds": None, "tds_p": None, "tds_r": None}
    ev = _eval()
    g, r = labels_of(gen_steps), labels_of(ref_steps)
    t = ev.tds_score(g, r)
    return {"pc": round(ev.pc_score(g, r), 6), "tds": round(t["f1"], 6), "tds_p": round(t["precision"], 6), "tds_r": round(t["recall"], 6)}


def _held_type(h) -> str:
    return str(h or "").split("|")[0].strip().lower()


def slice_realism(steps, goal_ok, start_holding=None, rule=None) -> dict:
    """A26 item 6 (``rule`` "a26", the default): ``unrealistic_slice`` is True when ANY executed slice step was not done with the
    ButterKnife held, whatever the goal outcome (an executor fault under A26); legacy prereg s8 (``rule="legacy"``): a slicing goal
    met while no Knife / ButterKnife was held at the slice step. Looks only at slice steps that
    actually ran. Holding BEFORE a step = the previous step's recorded holding (the start holding for step 0); if neither is
    recorded the step's own holding is used (a slice does not change what is held). Unknown holding -> `unrealistic_slice` None."""
    ev = _eval()
    rule = rule or SLICE_RULE
    tools = CUTTING_TOOLS if rule == "a26" else KNIVES
    n = no_knife = unknown = 0
    prev = start_holding
    prev_known = start_holding is not None
    for row in steps or ():
        h_after = row.get("holding") if isinstance(row, dict) else None
        ran = isinstance(row, dict) and (row.get("ok") or int(row.get("prims") or 0) > 0)             and row.get("cause") != "no_cutting_tool"          # A26: a refused slice never ran (end-of-step tidy may cost prims)
        if isinstance(row, dict) and ev.norm_action(row.get("action")) == "slice" and ran:
            n += 1
            if "holding_before" in row:
                held, known = row.get("holding_before"), True
            elif prev_known:
                held, known = prev, True
            elif "holding" in row:
                held, known = h_after, True
            else:
                held, known = None, False
            if known and _held_type(held) == "?":                      # held, but the agent never learned the type: unknown, not "no knife"
                known = False
            if not known:
                unknown += 1
            elif _held_type(held) not in tools:
                no_knife += 1
        if isinstance(row, dict) and "holding" in row:
            prev, prev_known = h_after, True
    if n == 0 or (rule != "a26" and not goal_ok):
        flag = False
    elif no_knife:
        flag = True
    elif unknown:
        flag = None
    else:
        flag = False
    return {"unrealistic_slice": flag, "n_slice": n, "n_slice_no_knife": no_knife, "n_slice_unknown": unknown}


# ================================================================================================ journals
def _events_complete(rows, phase_names) -> dict | None:
    """Latest `phase_complete` event whose phase is one of `phase_names`."""
    best = None
    for r in rows:
        if r.get("event") == "phase_complete" and r.get("phase") in phase_names:
            best = r
    return best


def _invalid_pres(rows) -> set:
    """Presentations an amendment declared unrecoverably INVALID (prereg s13); the shared rule lives in common.invalid_pres."""
    return C.invalid_pres(rows)


#: ONLY a deliberately recorded `incomplete study` ends a study early (written by `plan.py --declare-incomplete`, Arash's command).
#: `pause_cost_guard` is a pause for a cost decision, not a stopping rule (prereg s13 / A15: resumption after a pause is outcome-blind, and
#: no aggregate over test outcomes exists until Phase C is complete or a stopping rule fires); a plain `ceiling` refusal likewise
#: records only that a block could not start.
STOPPING_EVENTS = ("stopping_rule", "incomplete_study")


def _stopping_rule(rows) -> dict | None:
    best = None
    for r in rows:
        if r.get("event") in STOPPING_EVENTS:
            best = r
    return best


def _latest_by(rows, keyf) -> dict:
    d = {}
    for r in rows:
        d[keyf(r)] = r
    return d


def _cell_parts(cell: str) -> dict:
    """`ns|plan|arm|pres|rN` -> parts. pres contains ':' but no '|'."""
    p = cell.split("|")
    if len(p) != 5 or p[1] != "plan" or not p[4].startswith("r"):
        raise ScoreIntegrity(f"not a plan cell id: {cell!r}")
    return {"namespace": p[0], "arm": p[2], "pres": p[3], "r": int(p[4][1:])}


def _restored_state(world: str, setup, reloc_seed) -> str:
    base = "canon" if world == "canon" else f"reloc:{reloc_seed}"
    return base + "+" + C.sha256_json(setup)[:16]


class Inputs:
    """The journals of one namespace, loaded once. Reading outcome.sealed.jsonl happens here and only here."""

    def __init__(self, namespace: str, root: str | None = None):
        self.ns, self.root = namespace, root
        jp = lambda k: C.journal_path(namespace, k, root)           # noqa: E731
        self.paths = {k: jp(k) for k in ("plan", "exec", "outcome", "capture", "events", "score")}
        self.plan = C.Journal(self.paths["plan"]).latest()
        self.exec_rows = C.Journal(self.paths["exec"]).latest()
        self.exec_hist = {}                                          # A36: key -> EVERY exec row (all attempts; the flag needs the retry history)
        for r in C.Journal(self.paths["exec"]).rows():
            self.exec_hist.setdefault(r.get("key"), []).append(r)
        self.outcome_rows = C.Journal(self.paths["outcome"]).latest()
        self.capture_rows = C.Journal(self.paths["capture"]).rows()
        self.events = C.Journal(self.paths["events"]).rows()
        self.exec_starts = {}                                        # A36: exec key -> its exec_attempt_start events (a hang leaves a start, no row)
        for ev in self.events:
            if ev.get("event") == A36.START_EVENT:
                self.exec_starts.setdefault(ev.get("key"), []).append(ev)
        self.exec_epoch = {}                                         # A36: the executor's epoch per presentation (execute._epochs, same rule)
        for ev in self.events:
            if ev.get("event") == "invalid_resolved":
                for p_ in ev.get("pres") or []:
                    self.exec_epoch[p_] = self.exec_epoch.get(p_, 0) + 1
        inval = C.invalid_pres(self.events)
        for r in self.exec_rows.values():
            if r.get("status") == "invalid" and r.get("pres") not in inval:
                raise ScoreIntegrity(f"exec row {r.get('key')} is INVALID (a broken honesty invariant): the phase should have aborted")
        self.by_unit = {}                                            # (unit, run) -> (exec, outcome)
        for k, e in self.exec_rows.items():
            self.by_unit[(e.get("unit"), int(e.get("run", 0)))] = (e, self.outcome_rows.get(k))
        self.cell_unit = {}                                          # plan cell key -> unit (from the exec records)
        for e in self.exec_rows.values():
            for c in e.get("cells") or ():
                self.cell_unit[c] = e.get("unit")
        self.summary_sha = {}
        for c in self.capture_rows:
            if int(c.get("capture_run", 1)) == 1:
                self.summary_sha[c["pres"]] = c.get("summary_sha")
        self.hint_sha = None                                         # C4 / ft-v2: the hook-B hint table the units were built with
        if T.is_ft2(namespace):
            up = os.path.join(C.ns_docs(namespace, root), "units.json")
            if os.path.exists(up):
                with open(up, encoding="utf-8") as f:
                    self.hint_sha = json.load(f).get("hint_table_sha")

    def runs_of(self, unit) -> list:
        return sorted(run for (u, run) in self.by_unit if u == unit)

    def exec_cells(self) -> dict:
        """Cells that exist only in the exec records (reference-plan P arms, cached-plan /p replays): key -> its unit."""
        return {c: u for c, u in self.cell_unit.items() if c not in self.plan}


def derive_unit(inp: Inputs, cell: str, plan_steps, gold: dict, *, reloc_seed=None, snap_sha=None) -> str | None:
    """Recompute the execution-unit key of a plan cell (fallback when no exec record lists the cell). None if a part is missing."""
    cp = _cell_parts(cell)
    pp = C.parse_pres(cp["pres"])
    chooser = T.effective_chooser(cp["arm"], inp.ns)              # C4: ft-v2 = prior for every arm; ft-v1 = the arm table (as before)
    try:
        policy = T.exec_policy(cp["arm"], inp.ns, getattr(inp, "hint_sha", None))
    except T.TreatmentError:
        return None                                              # a hook-B unit without the recorded hint table: not derivable
    sha = None
    if chooser != "prior":
        sha = (snap_sha or _snapshot_sha)(chooser, inp.root)
        if not sha:
            return None
    ssha = inp.summary_sha.get(cp["pres"])
    g = gold.get(pp["item"])
    if ssha is None or g is None or (pp["world"] == "reloc" and reloc_seed is None):
        return None
    return C.execution_unit_key(plan=plan_steps, pres=cp["pres"], chooser=chooser, snapshot_sha=sha,
                                restored_state=_restored_state(pp["world"], g["setup"], reloc_seed), summary_sha=ssha, exec_policy=policy)


def _snapshot_sha(source: str, root: str | None) -> str | None:
    """The snapshot sha Phase C put into the unit key: the sha of the snapshot DIRECTORY tree (execute.dir_sha), not the manifest's own sha."""
    d = C.snapshot_dir(source, root)
    if not os.path.isdir(d):
        return None
    from execute import dir_sha                                    # same function, same value (execute.py has no heavy imports)
    return dir_sha(d)


# ================================================================================================ one score row
def _failure_class(plan_status, exec_row, outcome) -> str:
    if plan_status == "rejected":
        return "plan_rejected"
    if plan_status == "infra":
        return "plan_infra"
    if plan_status == "ceiling":
        return "plan_ceiling"
    if exec_row is None:
        return "missing_exec"
    st = exec_row.get("status")
    if st == "crash":
        return "exec_crash"
    if st == "blocked_state":
        return "exec_blocked_state"
    if st == "not_executed":
        return "not_executed"
    if outcome is None:
        return "missing_outcome"
    return "ok"


def _x_flag(plan_row, key: str) -> bool:
    """A21 E: True when any attempt of a Codex (X) plan cell reported `key` (a tool event during the call, or a session-identity mismatch)."""
    for a in ((plan_row or {}).get("attempts") or ()):
        if isinstance((a or {}).get("extra"), dict) and (a["extra"].get(key) is True):
            return True
    return False


def score_cell(cell: str, run: int, *, plan_row, exec_row, outcome, gold_item, family, ns, plan_steps_for_metrics=None,
               invalid_episode: bool = False, plan_only: bool = False, exec_history=None, exec_starts=None, block=None, exec_epoch=None) -> dict:
    """Pure: one (cell, run) -> a score record. `plan_row` may be None for reference / replay cells (their plan is the executed one).

    Field meanings (M-fix; they used to be conflated in `valid`):
      executed         the plan was accepted AND run to a sealed outcome (failure_class == "ok"). False = SR 0, still in every denominator.
      valid            integrity validity: True for every presentation that reached scoring (an INVALID episode aborts Phase C).
      invalid_episode  an explicit, amended record that the presentation was an unrecoverable honesty-INVALID episode (prereg s13);
                       the ONLY thing analysis may drop a presentation for.
      infra / infra_planner   PLANNER transport failure after the retries (the registered paired sensitivity drops only these).
      exec_failure     simulator crash / blocked_state after the re-runs (counts as a failure, reported separately, prereg s13).
      excluded_reset   A36 (draft r2; analysis only, the class is unchanged): a core infra_exec cell whose 3 attempts (`exec_history`) all
                       provably stopped at the pre-execution reset gate, each started once (`exec_starts`, None = not recorded ->
                       never flagged); core = block "main" (`block`, else the plan row's), canon, r0, run0 (a36.excluded_reset)."""
    cp = _cell_parts(cell)
    pp = C.parse_pres(cp["pres"])
    status = plan_row["status"] if plan_row else "ok"
    fc = _failure_class(status, exec_row, outcome)
    ref = gold_item["plan"]
    final = (plan_row.get("plan") if plan_row else plan_steps_for_metrics)
    first = (plan_row.get("first_plan") if plan_row else plan_steps_for_metrics)
    if status != "ok":
        final = None
    done = fc == "ok"
    goal_ok = bool(outcome.get("goal_ok")) if (done and outcome) else False
    exec_ok = bool(outcome.get("exec_ok")) if (done and outcome) else False
    steps = (outcome.get("steps") if (done and outcome) else None) or []
    sr = 1.0 if goal_ok else 0.0
    sr_strict = 1.0 if (goal_ok and exec_ok) else 0.0
    fm, lm = plan_metrics(first, ref), plan_metrics(final, ref)
    real = slice_realism(steps, goal_ok, (outcome or {}).get("start_holding")) if done else \
        {"unrealistic_slice": False, "n_slice": 0, "n_slice_no_knife": 0, "n_slice_unknown": 0}
    plan_len = len(C.norm_plan(final)) if final is not None else None
    rec = {
        "schema": "ftm.score/1", "spec": C.SPEC_VERSION, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "key": C.cell_id(ns, "score", cp["arm"], cp["pres"], f"r{cp['r']}", f"run{run}"),
        "namespace": ns, "arm": cp["arm"], "pres": cp["pres"], "r": cp["r"], "run": int(run),
        "unit": (exec_row or {}).get("unit"), "cell": cell, "scored": int(run) == 0,
        "sr": sr, "strict_sr": sr_strict, "valid": True, "executed": bool(done), "invalid_episode": bool(invalid_episode),
        "infra": fc == "plan_infra", "infra_planner": fc == "plan_infra", "exec_failure": fc in ("exec_crash", "exec_blocked_state"),
        "failure_class": fc, "run_class": C.run_class(fc), "exec_attempt": (exec_row or {}).get("attempt"),
        "x_tool_events": _x_flag(plan_row, "isolation_violation"), "x_identity_problem": _x_flag(plan_row, "identity_violation"),
        "plan_status": status, "exec_status": (exec_row or {}).get("status"),
        "goal_ok": goal_ok, "exec_ok": exec_ok, "steps_n": len(steps), "steps_ok_n": sum(1 for s in steps if s.get("ok")),
        "item": pp["item"], "lang": pp["lang"], "world": pp["world"], "level": gold_item["level"], "family": family,
        "plan_len": plan_len, "first_ok": (plan_row or {}).get("first_ok"), "repaired": (plan_row or {}).get("repaired"),
        "pc_first": fm["pc"], "pc_final": lm["pc"], "tds_first": fm["tds"], "tds_final": lm["tds"],
        "tds_final_p": lm["tds_p"], "tds_final_r": lm["tds_r"],
        "empty_plan_success": bool(done and plan_len == 0 and goal_ok),
        "block": (plan_row or {}).get("block"), "cost_usd": (plan_row or {}).get("cost_usd", 0.0),
        "calls": (plan_row or {}).get("calls", 0), "scope": (plan_row or {}).get("scope"),
        **real,
    }
    if plan_only:                       # A27 / A25 item 3: a planning diagnostic, never executed: no SR (None, not 0), plan metrics only
        rec.update({"plan_only": True, "sr": None, "strict_sr": None, "executed": False, "goal_ok": None, "exec_ok": None,
                    "failure_class": "plan_only" if status == "ok" else fc, "run_class": "plan_only", "empty_plan_success": False,
                    "plan_valid": status == "ok"})
    blk = block if block is not None else (plan_row or {}).get("block")
    core = (not plan_only) and blk == "main" and int(cp["r"]) == 0 and int(run) == 0 and pp["world"] == "canon"
    ev = A36.excluded_reset(rec["run_class"], exec_history if exec_history is not None else ([exec_row] if exec_row else []),
                            starts=exec_starts, outcome_present=outcome is not None, core=core, epoch=exec_epoch)
    rec["excluded_reset"], rec["excluded_reset_evidence"] = ev["flag"], ev
    return rec


def _block_of(inp, cp, prow):
    """A36 core scope: the cell's block. A plan cell: its plan row's; a P arm (reference plan, no call): "main" (execute.build_units);
    a /p replay: the block of the plan row it replays; anything else: None (not core)."""
    if prow:
        return prow.get("block")
    planner = C.arm_info(cp["arm"]).get("planner") if cp["arm"] in C.ARMS else None
    if planner == "ref":
        return "main"
    if isinstance(planner, str) and planner.startswith("replay:"):
        src = inp.plan.get(C.cell_id(inp.ns, "plan", planner.split(":", 1)[1], cp["pres"], f"r{cp['r']}"))
        return (src or {}).get("block")
    return None


# ================================================================================================ the scorer
def run_score(namespace: str, *, root: str | None = None, items: str = "dev", allow_test: bool = False, partial: bool = False,
              steal_lock: bool = False, dry_run: bool = False, dataset_path: str | None = None, verify_gold: bool = True,
              reloc_seed=None, snap_sha=None, log=print) -> dict:
    C.ns_check(namespace)
    inp_paths = {k: C.journal_path(namespace, k, root) for k in ("plan", "exec", "events", "score")}
    seal_path = os.path.join(C.ns_docs(namespace, root), SEAL_NAME)

    # ---- already done? a complete seal that still matches score.jsonl is a no-op; a mismatch is corruption
    seal = _read_seal(seal_path)
    if seal and os.path.exists(inp_paths["score"]):
        same = C.sha256_file(inp_paths["score"]) == seal.get("sha256")
        if seal.get("complete") and not same:
            raise ScoreIntegrity("score.jsonl changed after its complete seal: refusing to continue (sealed journal)")
        pdg = seal.get("plan_diagnostics") or {}
        if seal.get("complete") and pdg.get("sha256"):                # A27 (Astra P-A27 #1): the plan-only artifact is sealed too
            dp = os.path.join(C.ns_docs(namespace, root), pdg.get("file") or PLAN_DIAG_NAME)
            if not os.path.exists(dp) or C.sha256_file(dp) != pdg["sha256"]:
                raise ScoreIntegrity(f"{PLAN_DIAG_NAME} is missing or changed after its complete seal: refusing to continue (sealed artifact)")
        if seal.get("complete") and same:
            log(f"[score] {namespace}: already complete ({seal['n']} rows, seal ok)")
            return {"status": "already-done", "n": seal["n"], "seal": seal}

    with C.PhaseLock(namespace, "score", root, steal=steal_lock) as _lk:
        inp = Inputs(namespace, root)
        # ---- guard: every item named in the plan / exec journals, on every resume (SPEC s7)
        item_ids = sorted({C.parse_pres(_cell_parts(c)["pres"])["item"] for c in list(inp.plan) + list(inp.exec_cells())} |
                          {C.parse_pres(e["pres"])["item"] for e in inp.exec_rows.values() if e.get("pres")})
        C.guard_items(item_ids, namespace, allow_test=allow_test, root=root)

        # ---- phase completion (SPEC s6)
        ev_rows = inp.events
        miss = []
        for label, names, path in (("plan", ("plan",), inp_paths["plan"]), ("exec", ("exec", "execute"), inp_paths["exec"])):
            ce = _events_complete(ev_rows, names)
            if ce is None:
                miss.append(f"no phase_complete for {label}")
            elif ce.get("journal_sha") and os.path.exists(path):
                ok_shas = {C.sha256_file(path)}
                if label == "exec" and os.path.exists(inp.paths["outcome"]):
                    ok_shas.add(C.sha256_file(inp.paths["outcome"]))
                if ce["journal_sha"] not in ok_shas:
                    miss.append(f"{label} journal changed after its phase_complete")
        sr_event = _stopping_rule(ev_rows)
        if miss:
            if not partial:
                raise ScoreRefused("; ".join(miss) + " (use --partial only with a recorded stopping rule)")
            if sr_event is None:
                raise ScoreRefused("; ".join(miss) + "; --partial needs a recorded stopping-rule event in events.jsonl (prereg s13)")
            log(f"[score] PARTIAL: {'; '.join(miss)}; stopping rule event {sr_event.get('event')!r}")

        # ---- gold (read here, nowhere else)
        gold, gold_sha = _gold_load(root, dataset_path, verify=verify_gold)
        fam = _family_map(root) if verify_gold or os.path.exists(C.paths(root)["split"]) else {}
        for i in item_ids:
            if i not in gold:
                raise ScoreIntegrity(f"item {i!r} is not in the gold dataset")

        inval = _invalid_pres(ev_rows)
        cuts = C.cut_groups(ev_rows)                                 # recorded exploratory cuts: not run, not executed, NOT scored 0
        cells = {}                                                   # cell -> plan row or None
        for c, r in inp.plan.items():
            cells[c] = r
        for c in inp.exec_cells():
            cells[c] = None
        n_cut = 0
        if cuts:
            for c in sorted(cells):
                cp0 = _cell_parts(c)
                if C.cut_group(cp0["arm"], cp0["pres"]) in cuts:
                    del cells[c]
                    n_cut += 1
        n_replay_zero = 0
        replay_on = _replay_on(namespace, root)
        for c, prow in sorted(inp.plan.items()):                     # /p replay cells of +M plans that were never executed score SR 0
            if c not in cells or (prow or {}).get("status") not in ("rejected", "infra"):
                continue
            cp0 = _cell_parts(c)
            for ra in C.ARMS:
                if C.arm_info(ra)["planner"] != f"replay:{cp0['arm']}" or not replay_on:
                    continue
                rk = C.cell_id(namespace, "plan", ra, cp0["pres"], f"r{cp0['r']}")
                if rk in cells or C.cut_group(ra, cp0["pres"]) in cuts:
                    continue
                cells[rk] = {"status": prow["status"], "block": prow.get("block"), "cost_usd": 0.0, "calls": 0, "scope": "free",
                             "replay_of": c}
                n_replay_zero += 1
        writer = C.Journal(inp_paths["score"])
        done_keys = writer.keys()
        plan_cells_missing = []
        n_new = n_skipped = 0
        rows_out = []
        diag_rows = []                       # A27 / A25.3: plan-only cells (exploratory Codex) -> plan_diagnostics.json, never score.jsonl (no SR)
        for cell in sorted(cells):
            prow = cells[cell]
            cp = _cell_parts(cell)
            pp = C.parse_pres(cp["pres"])
            g = gold[pp["item"]]
            f = fam.get(pp["item"])
            status = prow["status"] if prow else "ok"
            if status == "ceiling":
                n_skipped += 1
                if not partial:
                    raise ScoreIntegrity(f"plan cell {cell} has status 'ceiling' (not terminal) but the plan phase is complete")
                continue
            if C.plan_only(cp["arm"]):                               # A27 / A25.3: exploratory Codex -> planning diagnostics, no SR
                diag_rows.append(score_cell(cell, 0, plan_row=prow, exec_row=None, outcome=None, gold_item=g, family=f, ns=namespace,
                                            invalid_episode=cp["pres"] in inval, plan_only=True))
                continue
            if status != "ok":                                       # rejected / infra: nothing executed, SR 0
                rows_out.append((cell, 0, score_cell(cell, 0, plan_row=prow, exec_row=None, outcome=None, gold_item=g, family=f, ns=namespace,
                                                   invalid_episode=cp["pres"] in inval)))
                continue
            unit = inp.cell_unit.get(cell)
            if unit is None:
                unit = derive_unit(inp, cell, prow.get("plan") if prow else None, gold, reloc_seed=reloc_seed, snap_sha=snap_sha)
            runs = inp.runs_of(unit) if unit else []
            if (not runs or 0 not in runs) and cp["pres"] in inval:     # unrecoverable INVALID presentation: scored 0 and flagged, then dropped
                rows_out.append((cell, 0, score_cell(cell, 0, plan_row=prow, exec_row=None, outcome=None, gold_item=g, family=f, ns=namespace,
                                                   plan_steps_for_metrics=None, invalid_episode=True)))
                continue
            if not runs or 0 not in runs:
                plan_cells_missing.append(cell)
                n_skipped += 1
                continue
            for run in runs:
                e, o = inp.by_unit[(unit, run)]
                rows_out.append((cell, run, score_cell(cell, run, plan_row=prow, exec_row=e, outcome=o, gold_item=g, family=f, ns=namespace,
                                                      plan_steps_for_metrics=(None if prow else _plan_of_exec(e, o)),
                                                      invalid_episode=cp["pres"] in inval, exec_history=inp.exec_hist.get(e.get("key")),
                                                      exec_starts=(inp.exec_starts.get(e.get("key"), []) if inp.exec_starts else None),
                                                      block=_block_of(inp, cp, prow), exec_epoch=inp.exec_epoch.get(cp["pres"], 0))))
        if plan_cells_missing and not partial:
            raise ScoreIntegrity(f"{len(plan_cells_missing)} accepted plan cell(s) have no executed outcome although the exec phase is "
                                 f"complete (first: {plan_cells_missing[0]}). The exec records must list their `cells`, or the unit key "
                                 f"must be recomputable (capture summary sha, snapshot manifest, setup, reloc seed).")
        # ---- write (resume skips keys already journaled; a keyed row is never rewritten)
        leak = False
        for cell, run, rec in rows_out:
            if rec["key"] in done_keys:
                continue
            if C.key_leak(json.dumps(rec, ensure_ascii=False)):
                leak = True
                break
            if not dry_run:
                writer.append(rec)
            n_new += 1
        if leak:
            raise ScoreIntegrity("a score row looks key-like (key_leak): aborted before sealing")
        diag = plan_diagnostics(diag_rows, namespace)
        if C.key_leak(json.dumps(diag, ensure_ascii=False)):
            raise ScoreIntegrity("a plan-diagnostic row looks key-like (key_leak): aborted before sealing")
        diag_path = os.path.join(C.ns_docs(namespace, root), PLAN_DIAG_NAME)
        if not dry_run:                      # always written (an empty one when every plan-only cell is cut), so no stale file can survive
            _write_json_atomic(diag_path, diag)
        all_rows = C.Journal(inp_paths["score"]).rows() if not dry_run else []
        complete = (not miss) and not plan_cells_missing and n_skipped == 0
        result = {"status": "scored" if complete else "partial", "n": len(all_rows), "new": n_new, "skipped": n_skipped,
                  "missing_outcome": len(plan_cells_missing), "complete": complete, "plan_only_rows": len(diag_rows)}
        if not dry_run:
            seal = {"schema": "ftm.score.seal/1", "spec": C.SPEC_VERSION, "namespace": namespace,
                    "sha256": C.sha256_file(inp_paths["score"]) if os.path.exists(inp_paths["score"]) else C.sha256_text(""),
                    "n": len(all_rows), "complete": complete, "partial": bool(miss), "skipped": n_skipped,
                    "cut_groups": sorted(cuts), "n_cut_cells": n_cut, "n_replay_zero_rows": n_replay_zero,
                    "plan_diagnostics": {"file": PLAN_DIAG_NAME, "n": len(diag_rows),
                                         "sha256": C.sha256_file(diag_path) if os.path.exists(diag_path) else None,
                                         "rule": "A27-draft / A25 item 3: plan-only, no SR"},
                    "invalid_episode_pres": sorted(inval),
                    "stopping_rule": (sr_event or {}).get("event") if miss else None,
                    "inputs": {k: (C.sha256_file(p) if os.path.exists(p) else None) for k, p in
                               (("plan", inp.paths["plan"]), ("exec", inp.paths["exec"]), ("outcome", inp.paths["outcome"]))},
                    "gold_sha256": gold_sha,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            _write_json_atomic(seal_path, seal)
            C.Journal(inp_paths["events"]).append(C.make_record("events", event="phase_complete" if complete else "phase_partial",
                                                                phase="score", n=len(all_rows), journal_sha=seal["sha256"]))
            result["seal"] = seal
        log(f"[score] {namespace}: {result['status']}, {result['n']} rows ({n_new} new, {n_skipped} skipped)")
        return result


def _replay_on(namespace: str, root) -> bool:
    """Phase C ran the /p replay arms unless units.json (written by every execute run) says it did not."""
    try:
        with open(os.path.join(C.ns_docs(namespace, root), "units.json"), encoding="utf-8") as f:
            return bool(json.load(f).get("replay", True))
    except (OSError, ValueError):
        return True


def _plan_of_exec(e, o):
    """For reference / replay cells the executed plan is the exec record's `plan` (if the executor kept it), else the outcome's."""
    for src in (e, o):
        if src and src.get("plan") is not None:
            return src["plan"]
    return None


def _family_map(root) -> dict:
    try:
        m = C.load_manifest(root, verify=False)
    except (OSError, ValueError):
        return {}
    return {i: f["family"] for f in (m.get("families") or []) for i in f.get("items") or ()}


def _read_seal(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_seal(namespace: str, root: str | None = None) -> dict:
    """For downstream readers (analyze.py): the seal exists and still matches score.jsonl. Raises ScoreRefused otherwise."""
    s = _read_seal(os.path.join(C.ns_docs(namespace, root), SEAL_NAME))
    p = C.journal_path(namespace, "score", root)
    if not s or not os.path.exists(p):
        raise ScoreRefused("score journal has no seal")
    if C.sha256_file(p) != s.get("sha256"):
        raise ScoreRefused("score.jsonl does not match its seal (edited or appended after sealing)")
    return s


# ================================================================================================ gate 12.5
def evaluator_check(namespace: str = "rehearsal-v1", *, root: str | None = None, baseline_path: str | None = None,
                    min_agreement: float = EVAL_MIN_AGREEMENT, require_empty_plan: bool = True, arms=("P0",),
                    write: bool = True) -> dict:
    """Gate 12.5. (a) Reference plans on dev (arm P0, scored run 0) reproduce step 4's honest reference-plan goal results up to
    `min_agreement` of the compared items (step 4 states its own run-to-run variance); (b) the empty plan scores 0: any row that
    executed an empty plan and succeeded fails the gate, and with `require_empty_plan` at least one empty-plan row must exist."""
    rows = [r for r in C.Journal(C.journal_path(namespace, "score", root)).rows() if r.get("scored")]
    bp = baseline_path or os.path.join(C.paths(root)["root"], STEP4_BASELINE)
    base = {}
    try:
        with open(bp, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    if r.get("status") in ("ok", "FAIL"):
                        base[r["id"]] = bool(r.get("goal_ok"))
    except OSError:
        pass
    cmp_rows = [r for r in rows if r["arm"] in arms and r.get("executed") and r["item"] in base]
    agree = [r for r in cmp_rows if bool(r["goal_ok"]) == base[r["item"]]]
    dis = [{"pres": r["pres"], "now": bool(r["goal_ok"]), "step4": base[r["item"]]} for r in cmp_rows if bool(r["goal_ok"]) != base[r["item"]]]
    share = (len(agree) / len(cmp_rows)) if cmp_rows else 0.0
    empty = [r for r in rows if r.get("plan_len") == 0 and r.get("executed")]
    bad_empty = [r["pres"] for r in empty if r["goal_ok"]]
    reasons = []
    if not cmp_rows:
        reasons.append("no reference-plan (P0) rows could be compared with step 4")
    elif share < min_agreement:
        reasons.append(f"agreement with step 4 {share:.3f} < {min_agreement}")
    if bad_empty:
        reasons.append(f"empty plan succeeded on {len(bad_empty)} presentation(s): the goal was true at reset (12.4/applicability leak)")
    if require_empty_plan and not empty:
        reasons.append("no empty-plan execution rows: run the empty-plan control on dev (execute.py) before 12.5 can pass")
    res = {"schema": "final-test.evaluator-check/1", "gate": "12.5", "namespace": namespace, "baseline": os.path.relpath(bp, C.paths(root)["root"]),
           "min_agreement": min_agreement, "compared": len(cmp_rows), "agree": len(agree), "agreement": round(share, 4),
           "baseline_sr": round(sum(1 for r in cmp_rows if base[r["item"]]) / len(cmp_rows), 4) if cmp_rows else None,
           "now_sr": round(sum(1 for r in cmp_rows if r["goal_ok"]) / len(cmp_rows), 4) if cmp_rows else None,
           "disagreements": dis, "empty_plan_rows": len(empty), "empty_plan_successes": bad_empty, "reasons": reasons,
           "pass": not reasons, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if write:
        _write_json_atomic(os.path.join(C.ns_docs(namespace, root), "evaluator_check.json"), res)
    return res


# ================================================================================================ CLI
def main(argv=None, root: str | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase D: sealed scoring journal (gold isolated here).")
    ap.add_argument("--namespace", default="rehearsal-v1")
    ap.add_argument("--items", choices=("dev", "test"), default="dev")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--partial", action="store_true", help="score what exists; needs a recorded stopping-rule event; seal says complete:false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluator-check", action="store_true", help="gate 12.5 over an already scored namespace")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--min-agreement", type=float, default=EVAL_MIN_AGREEMENT)
    ap.add_argument("--reloc-seed", default=None)
    a = ap.parse_args(argv)
    try:
        if a.evaluator_check:
            r = evaluator_check(a.namespace, root=root, baseline_path=a.baseline, min_agreement=a.min_agreement)
            print(json.dumps({k: r[k] for k in ("gate", "pass", "compared", "agreement", "empty_plan_rows", "reasons")}, ensure_ascii=False))
            return 0 if r["pass"] else 2
        r = run_score(a.namespace, root=root, items=a.items, allow_test=a.allow_test, partial=a.partial, steal_lock=a.steal_lock,
                      dry_run=a.dry_run, reloc_seed=a.reloc_seed)
        return 0 if r["status"] in ("scored", "already-done") else 4
    except (C.TestItemRefused, ScoreRefused, C.PhaseLocked) as e:
        print(f"[score] REFUSED: {e}", file=sys.stderr)
        return 2
    except (ScoreIntegrity, C.JournalCorrupt) as e:
        print(f"[score] INTEGRITY FAILURE: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
