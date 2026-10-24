"""Final-test matrix, Phase C: execute the cached plans in identical restored worlds (stage M6; contract in SPEC.md s4, s5, s10, s12).

    WSL, no LLM.  Reads capture.jsonl + states/ (Phase A) and plan.jsonl (Phase B); writes exec.jsonl (integrity only), outcome.sealed.jsonl
    (the results; read only by score.py), units.json (unit -> plan cells, gold-free) and events.jsonl.

What it does
  * builds EXECUTION UNITS (prereg s7 C) from plan rows with status ok, the P arms (reference plans, no call) and the /p replays; arms with equal
    unit keys share ONE execution; on REPEAT presentations each distinct core-four / core-/p unit runs REPEAT_RUNS = 2 times (runs 0, 1; A27
    draft, was 3), each from an independent restore; exploratory Codex plans are never executed (plan-only, A25 item 3);
  * runs them in the seeded `C.exec_blocks` order (run 0 everywhere first, then the repeats), each with `LiveExecutor(plan_after_sweep=True,
    state_capture=True, state_check=<reset_gate.compare vs the Phase A state>)` and a replay hook that checks the post-sweep summary sha;
  * takes the snapshot sha before and after every LTM run (a change is `invalid`), handles crash / blocked_state re-runs (attempts 0, 1, 2, each
    from a fresh restore), and aborts the phase on `invalid` (events abort_invalid; resume needs `--resolve-invalid <amendment>`);
  * owns the G-iso audit (`--g-iso`).

Gold: this process reads the dataset (setup, reference plans, goal predicates inside `h.scoring()`), the planners never do. No gold or outcome
key reaches exec / events records (`C.make_record` firewall). A test item is refused unless the SPEC guard holds (`C.guard_items`).

Everything that needs the simulator is imported lazily; the selftest (`sim/_ftm_execute_selftest.py`) runs on Windows python with fakes.
Python 3.10 syntax only (WSL).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                              # noqa: E402  (also puts sim/ on sys.path)
import treatment as T                                           # noqa: E402  (C4: the ft-v2 switch; inert in ft-v1 namespaces)
import obs_health as OH                                         # noqa: E402  (A34-A2 2(b): observation-pipeline health evidence)

PHASE = "execute"
SCENE = "FloorPlan1"
MAX_ATTEMPT = 2                       # attempts 0, 1, 2: the first run plus two re-runs from a fresh restore (SPEC s10)
REPEAT_RUNS = C.EXEC_RUNS_REPEAT      # A27 (draft): 2 (was 3)
#: Which units run REPEAT_RUNS times (A27 draft, PENDING ARASH SIGN-OFF AT THE FREEZE; Arash 2026-10-13 "Accept astra's vision. it's logical.").
#: On a REPEAT presentation, a unit runs twice when it carries a cell of block `main` or `repeat` of a core-four arm (G0, G+M, O0, O+M) or of a
#: /p replay of a core +M arm (`C.repeat_exec_arm`); with 3 planner calls per core arm that is "3 calls x 2 executions per distinct plan".
#: Every other unit runs once: G+FS, O+FS, X0, X+M (r = 0 only since A27), the curve, the relocation probe and the P0 / P+M reference
#: controls. The exploratory Codex arms are PLAN-ONLY (A25 item 3, A27): their cells are never units. Exact unit dedup is unchanged, so a
#: unit shared by a core cell and a non-core cell runs twice (one execution, shared, as prereg s7 C).
REPEAT_RULE = ("A27-draft: 2 runs for units that carry a core-four (G0, G+M, O0, O+M) or core /p replay cell of block main or repeat on a REPEAT "
               "item; every other unit (FS, X0 / X+M, curve, reloc, P0 / P+M) runs once; exploratory Codex arms are plan-only, never executed")
P_ARMS = ("P0", "P+M")
REPLAY_ARMS = ("G+M/p", "O+M/p")      # arm -> the arm whose cached plan it replays: arm_info()["planner"] = "replay:<arm>"
EXIT = {"ok": 0, "refused": 2, "pause": 3, "abort_invalid": 4, "interrupted": 5}


class ExecAborted(RuntimeError):
    """An `invalid` run (a broken honesty invariant) aborted the phase; it needs an amendment (prereg s13)."""


class SummaryMismatch(RuntimeError):
    """The post-sweep summary (or told holding) differs from the Phase A capture: logged as blocked_state (execution variance, prereg s7 C)."""


class ExecInvalid(RuntimeError):
    """An integrity invariant of this module failed (snapshot sha changed, oracle read, key leak)."""


def is_invariant(e: BaseException) -> bool:
    return isinstance(e, ExecInvalid) or type(e).__name__ in ("LiveInvariant", "InvariantError")


# ================================================================================================ small helpers
def dir_sha(path: str) -> str:
    """The ONE shared snapshot-directory sha (common.dir_sha; capture, score and the freeze checks use it too). 'absent' = missing."""
    return C.dir_sha(path)


def to_steps(plan) -> list:
    """Normalised plan rows -> the step dicts the honest skills take (action / object / to / arg)."""
    return [{"action": a, "object": o, "to": t, "arg": g} for a, o, t, g in C.norm_plan(plan)]


def setup_sha(item: dict) -> str:
    return C.sha256_json(item.get("setup") or [])[:16]


def restored_state_of(world: str, item: dict, reloc_seed) -> str:
    return (("canon" if world == "canon" else f"reloc:{reloc_seed}") + "+" + setup_sha(item))


def world_seed_of(ns: str, pres: str) -> int:
    """The honest executor takes no simulator seed (SEED_POLICY): the value is a recorded label, and it is the SAME 0 capture uses."""
    return 0


def _atomic_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _clean_audit(a: dict) -> dict:
    """Integrity counters only; any key the firewall forbids is dropped (the audit summary should never carry one)."""
    out = {k: v for k, v in (a or {}).items() if k not in C.GOLD_KEYS | C.OUTCOME_KEYS | {"steps"}}
    if isinstance(out.get("counts"), dict):
        out["counts"] = {k: v for k, v in out["counts"].items() if k not in C.GOLD_KEYS | C.OUTCOME_KEYS}
    return out


def _audit_bad(a: dict) -> bool:
    counts = (a or {}).get("counts") or {}
    return bool(int((a or {}).get("violations") or 0) or int(counts.get("oracle") or 0) or int(counts.get("map-build") or 0)
                or (a or {}).get("invalid"))


def with_holding(steps: list, stream) -> list:
    """Step rows + `holding_before`: what the agent believed it held when the step started, from the adapter's own `step_start` events
    (its belief, a type word; "?" = held but type unknown; None = nothing held, or the step never started). The executor's rows carry no
    holding, and the scorer needs it for the unrealistic-slice flag (prereg s8)."""
    before = {}
    for ev in stream or ():
        if isinstance(ev, dict) and ev.get("kind") == "step_start" and isinstance(ev.get("i"), int):
            before[ev["i"]] = (ev.get("pre") or {}).get("holding")
    out = []
    for i, s in enumerate(steps or ()):
        d = dict(s)
        d["holding_before"] = before.get(d.get("i", i))
        out.append(d)
    return out


def default_items(root: str | None = None) -> dict:
    import general_set as gs                                                       # lazy: gold reader (this process may read the dataset)
    return {it["id"]: it for it in gs.load(C.paths(root)["dataset"])["items"]}


# ================================================================================================ replay precondition (A21 C)
def actual_holding(state) -> list:
    """What the agent ACTUALLY holds, read from the post-sweep state (the simulator's own `isPickedUp` flags, captured inside `h.scoring()`):
    [{"name", "type"}] sorted by name. This is the world's truth, not the instruction's `holding` (the 'told' word)."""
    objs = (state or {}).get("objects") or {}
    return sorted(({"name": n, "type": str(r.get("type", ""))} for n, r in objs.items() if (r.get("flags") or {}).get("isPickedUp")),
                  key=lambda d: d["name"])


def told_matches_actual(told, actual: list) -> bool:
    """The told holding (a type word from the setup, '?' = held but unknown, None = nothing) must agree with what is actually in the hand."""
    if told in (None, ""):
        return not actual
    if told == "?":
        return bool(actual)
    return any(str(told).lower() == str(h.get("type", "")).lower() for h in actual)


def presentation_of(item: dict) -> dict:
    """A34-A2 2(b): what a capture must be bound to -- the presentation's reference plan and its own goal-clause texts (the evaluator's keys)."""
    import general_set as gs
    return {"plan": list(item.get("plan") or []), "goal_clauses": sorted({gs.format_goal(a) for a in item.get("goal") or []})}


def replay_precondition(state_a, state_now, told_a, told_now, tol, *, obs_a=None, obs_b=None, presentation=None) -> dict:
    """A21 C -- the pre-execution check of the replay hook, pure. A cached plan may run only if the world just restored equals the world
    Phase A captured, on everything the task can depend on:
      * `reset_gate.compare` (A14 / A14a / A14b): object identities and types, capabilities, containment (parents / contents), open / toggled /
        sliced / dirty / filled / broken / cooked / picked-up flags, derived accessibility, relevant summary rows, seen types, labels, the reference-
        plan feasibility verdicts, per-object position and rotation within tau, the agent pose to numerical precision;
      * the ACTUAL holding (names), equal to Phase A's;
      * the told holding equal to Phase A's AND consistent with the actual holding.
    Continuous jitter inside tau, non-task summary rows and the strict world hash do not breach. -> {'ok', 'breaches', 'max_dpos', 'max_drot'}.
    A34-A2 (signed 2026-10-15): the registered comparison above runs unchanged; when its COMPLETE breach set (holding checks included) is
    observation-only (seen_types / summary / summary.coords), both captures are valid and bound to the presentation
    (presentation_of(item): its reference plan and goal-clause texts), both observation pipelines are valid and ran with the same frozen settings (obs_a = the Phase A capture row's obs_pipeline,
    obs_b = this episode's, both obs_health.record) and the world-only snapshots are identical at stored precision, the comparison is
    overridden: ok True, registered_ok False, obs_drift recorded in full (reset_gate.a34_override)."""
    from final_test import reset_gate as rg                                      # lazy: stdlib only
    if state_a is None or state_now is None:
        return {"ok": False, "breaches": [{"kind": "capture_state_missing" if state_a is None else "state_missing"}], "max_dpos": None, "max_drot": None}
    chk = rg.compare(state_a, state_now, tol)
    br = list(chk["breaches"])
    ha, hn = actual_holding(state_a), actual_holding(state_now)
    if [h["name"] for h in ha] != [h["name"] for h in hn]:
        br.append({"kind": "actual_holding", "capture": [h["name"] for h in ha][:5], "now": [h["name"] for h in hn][:5]})
    if (told_now or None) != (told_a or None):
        br.append({"kind": "told_holding", "capture": told_a, "now": told_now})
    if not told_matches_actual(told_now, hn):
        br.append({"kind": "told_vs_actual_holding", "told": told_now, "actual": [h["type"] for h in hn][:5]})
    ov = (rg.a34_override(state_a, state_now, br, obs_a=obs_a, obs_b=obs_b, presentation=presentation) if br
          else {"override": False, "why": [], "obs_drift": None})
    return {"ok": (not br) or ov["override"], "registered_ok": not br, "override": ov["override"], "override_why": ov["why"][:6],
            "obs_drift": ov["obs_drift"], "breaches": br, "max_dpos": chk.get("max_dpos"), "max_drot": chk.get("max_drot"),
            "actual_holding": [h["name"] for h in hn]}


# ================================================================================================ plan-first gate
def plan_first_problem(events: list, plan_path: str):
    """PLAN FIRST (PF-safety-test): in a namespace where `planfirst.py` was used, execution may start only when a `plan_first_complete` event is bound to the
    CURRENT plan journal (same sha) and covers the registered manifest in full, or under a recorded stopping rule. A stage-local `phase_complete` or a
    plan journal that changed after the completion record does not unlock it. -> None when the gate holds (or plan-first was never used), else the reason."""
    if not any(e.get("event") == "plan_first_start" for e in events):
        return None
    sha = C.sha256_file(plan_path) if os.path.exists(plan_path) else None
    good = [e for e in events if e.get("event") == "plan_first_complete" and e.get("journal_sha") == sha and e.get("scope") in ("full", "authorized_partial")]
    if good:
        return None
    done = [e for e in events if e.get("event") == "plan_first_complete"]
    return ("plan-first mode was used in this namespace, but no plan_first_complete record (scope full or authorized_partial) matches the current plan "
            f"journal ({'no completion record at all' if not done else 'the plan journal changed after the last completion record, or its scope was ' + str(done[-1].get('scope'))}): "
            "all planning must finish before any execution")


# ================================================================================================ unit building (SPEC s4)
def build_units(*, ns: str, captures: dict, plans: dict, items: dict, snap_sha, manifest: dict, p_arms=P_ARMS, replay=True,
                worlds=("canon",), reloc_seed=None, cuts=frozenset(), hint_sha: str | None = None) -> dict:
    """-> {"units": {unit_key: {...}}, "unexecuted": [...], "skipped_worlds": n}.

    captures: {pres: capture row (run 1)}; plans: {plan key: latest plan row}; snap_sha(source) -> sha of that frozen snapshot.
    A unit holds the plan, the chooser, the restored state and every plan cell (arm, r, block, key) that maps onto it.
    C4 / ft-v2: every arm runs the prior chooser; the +M, curve and P+M units carry exec_policy "briefing:<hint sha16>" (hook B), which
    is part of their key, so a hook-on unit never shares with a hook-off one. ft-v1: unchanged (no exec_policy anywhere)."""
    units, unexec, unexec_replay, skipped, n_cut = {}, [], [], 0, 0
    ft2 = T.is_ft2(ns)
    repeat_items = set(C.repeat_items(ns, manifest))

    def add(arm, pres, r, block, key, steps, src_row, count_skip=True):
        nonlocal skipped
        pp = C.parse_pres(pres)
        if pp["world"] not in worlds:
            skipped += count_skip                 # an accepted PLAN cell of a world that is not being executed (P arms do not count)
            return
        cap = captures.get(pres)
        if cap is None:
            raise RuntimeError(f"no Phase A capture for {pres} (needed by {key})")
        chooser = T.effective_chooser(arm, ns)
        policy = T.exec_policy(arm, ns, hint_sha)
        sha = None if chooser == "prior" else snap_sha(chooser)
        if chooser != "prior" and sha in (None, "absent"):
            raise RuntimeError(f"snapshot {chooser} is absent (needed by {key})")
        item = items[pp["item"]]
        rs = restored_state_of(pp["world"], item, reloc_seed)
        if cap.get("restored_state") not in (None, rs):          # the world restored now must be the world Phase A captured
            raise RuntimeError(f"restored state of {pres} differs from the Phase A capture ({rs} != {cap.get('restored_state')}); "
                               f"the dataset setup or the relocation seed changed (needed by {key})")
        uk = C.execution_unit_key(plan=steps, pres=pres, chooser=chooser, snapshot_sha=sha, restored_state=rs, summary_sha=cap["summary_sha"],
                                  exec_policy=policy)
        u = units.setdefault(uk, {"unit": uk, "pres": pres, "item": pp["item"], "lang": pp["lang"], "world": pp["world"], "chooser": chooser,
                                  "snapshot_sha": sha, "restored_state": rs, "summary_sha": cap["summary_sha"], "plan": C.norm_plan(steps),
                                  "cells": [], "repeat": False, **({"exec_policy": policy} if ft2 else {})})
        if ft2 and u.get("exec_policy") != policy:
            raise RuntimeError(f"unit {uk[:12]} mixes executor policies ({u.get('exec_policy')} vs {policy})")
        u["cells"].append({"arm": arm, "r": r, "block": block, "key": key})
        if pp["item"] in repeat_items and block in ("main", "repeat") and C.repeat_exec_arm(arm):
            u["repeat"] = True                                                   # A27: core four + core /p replays -> REPEAT_RUNS runs

    n_plan_only = 0
    for key in sorted(plans):
        row = plans[key]
        if C.cut_group(row.get("arm") or "", row.get("pres")) in cuts:      # a recorded exploratory cut: not executed, not scored
            n_cut += 1
            continue
        if C.plan_only(row.get("arm")):                                      # A27 / A25.3: exploratory Codex = planning diagnostics only
            n_plan_only += 1
            continue
        if row.get("status") != "ok":
            unexec.append({"key": key, "status": row.get("status"), "arm": row.get("arm"), "pres": row.get("pres")})
            if replay and row.get("status") in ("rejected", "infra"):    # the /p replay of a failed +M plan is NOT executed either: SR 0, not vanish
                for ra in REPLAY_ARMS:
                    if C.arm_info(ra)["planner"] == f"replay:{row.get('arm')}":
                        rk = C.cell_id(ns, "plan", ra, row["pres"], f"r{row['r']}")
                        if rk not in plans:
                            unexec_replay.append({"key": rk, "status": row.get("status"), "arm": ra, "pres": row["pres"], "r": row["r"],
                                                  "block": row.get("block"), "replay_of": key})
            continue
        add(row["arm"], row["pres"], row["r"], row["block"], key, row["plan"], row)
        if replay:                                                               # cached +M plans re-run with a prior chooser (no call)
            for ra in REPLAY_ARMS:
                if C.arm_info(ra)["planner"] == f"replay:{row['arm']}":
                    rk = C.cell_id(ns, "plan", ra, row["pres"], f"r{row['r']}")
                    if rk not in plans:
                        add(ra, row["pres"], row["r"], row["block"], rk, row["plan"], row)
    for pres in sorted(captures):                                                # the P arms: the reference plan of the item, no call
        pp = C.parse_pres(pres)
        if C.cut_group("P0", pres) in cuts:
            continue
        if captures[pres].get("status") not in (None, "ok"):          # a relocated presentation whose setup failed at Phase A has no world to run
            continue
        for pa in p_arms:
            add(pa, pres, 0, "main", C.cell_id(ns, "plan", pa, pres, "r0"), items[pp["item"]].get("plan") or [], None, count_skip=False)
    for u in units.values():
        u["runs"] = REPEAT_RUNS if u["repeat"] else 1
        u["cells"].sort(key=C.canon)
    return {"units": units, "unexecuted": unexec, "unexecuted_replay": unexec_replay, "skipped_worlds": skipped, "cut_plan_cells": n_cut,
            "plan_only_cells": n_plan_only}


# ================================================================================================ the phase
class ExecutePhase:
    def __init__(self, namespace: str, *, root: str | None = None, allow_test: bool = False, executor_factory=None, memory_opener=None,
                 items=None, tolerances=None, worlds=("canon",), reloc_seed=None, p_arms=P_ARMS, replay=True, require_complete=True,
                 partial=False, out=print, snapshot_sha=None, repeat_override=None, reset_patch=None, request_renderer=None, audit_probe=None,
                 briefing_table=None):
        C.ns_check(namespace)
        self.ft2 = T.is_ft2(namespace)                          # C4: prior chooser + hook B for the +M units (treatment.py)
        self._btable = briefing_table                           # tests inject a hint table; live: the namespace's approved / draft briefing
        self._hints = None
        self.ident = None
        self.ns, self.root, self.allow_test = namespace, root, allow_test
        self.factory, self.opener = executor_factory, memory_opener
        self.reset_patch = reset_patch                          # context-manager factory(seed): the relocated world (default: capture.relocated_reset)
        self._items, self._tol = items, tolerances
        self.worlds, self.reloc_seed, self.p_arms, self.replay = tuple(worlds), reloc_seed, tuple(p_arms), replay
        self.require_complete, self.out = require_complete, out
        self.partial = bool(partial)                            # W2: run on the plans that exist, ONLY under a recorded stopping rule (prereg s13)
        self._snap_sha = snapshot_sha or (lambda src: dir_sha(T.source_dir(namespace, src, root) if C.is_replication(namespace)
                                                              else C.snapshot_dir(src, root)))     # K23: the kitchen's own snapshots
        self.repeat_override = repeat_override                  # tests only (the CLI never sets it): replaces the manifest's repeat_subset
        self.docs, self.work = C.ns_docs(namespace, root), C.ns_work(namespace, root)
        self.J = {k: C.Journal(C.journal_path(namespace, k, root)) for k in ("capture", "plan", "exec", "outcome", "events")}
        self.units_path = os.path.join(self.docs, "units.json")
        self._ix, self._ix_size = None, -1                      # exec journal index: key -> rows (rebuilt only if the file changed under us)
        self.request_renderer = request_renderer   # G-iso (i): callable(capture record) -> request sha (tests inject one; default = PlannerContract)
        self.audit_probe = audit_probe              # A20.1: callable() -> sensitivity report; default = `audit_sensitivity_probe` (the real MetaAudit)
        self.exec_calls, self.last_episode_id = 0, None   # A20.1: counted at the one place that calls the executor (no dedup, no cached result)
        self.unrecoverable = set()
        self._secrets_cache = None
        self.ex = None
        self.box = {}

    # ---- inputs ------------------------------------------------------------------------------------------------------
    def event(self, event: str, **kw) -> dict:
        return self.J["events"].append(C.make_record("events", event=event, phase=PHASE, namespace=self.ns, **kw))

    def _complete(self, phase: str) -> dict | None:
        ev = [r for r in self.J["events"].rows() if r.get("event") == "phase_complete" and r.get("phase") == phase]
        return ev[-1] if ev else None

    def preflight(self) -> dict:
        """Refusals first: dependencies complete, an unresolved invalid abort, the test guard on every item named in the inputs."""
        if self.require_complete:
            pf = plan_first_problem(self.J["events"].rows(), self.J["plan"].path)
            if pf:
                raise ExecAborted(pf)
            for ph in ("capture", "plan"):
                if self._complete(ph) is None:
                    if ph == "plan" and self.partial:
                        sr = [e for e in self.J["events"].rows() if e.get("event") in ("stopping_rule", "incomplete_study")]
                        if not sr:
                            raise ExecAborted("--partial needs a recorded stopping-rule event (plan.py --declare-incomplete): Phase C does not start")
                        self.event("partial_execute", stopping_rule=sr[-1].get("reason"), by=sr[-1].get("by"))
                        continue
                    raise ExecAborted(f"phase {ph} has no phase_complete event: Phase C does not start (SPEC s6)")
        self.captures = {}
        for r in self.J["capture"].rows():
            if r.get("capture_run") == 1:
                self.captures[r["pres"]] = r                                   # latest wins
        self.plans = {k: r for k, r in self.J["plan"].latest().items() if k}
        pres_all = sorted(set(self.captures) | {r["pres"] for r in self.plans.values()})
        ids = sorted({C.parse_pres(p)["item"] for p in pres_all})
        guard = C.guard_items(ids, self.ns, allow_test=self.allow_test, root=self.root)          # raises TestItemRefused
        for p in pres_all:
            if p not in self.captures:
                raise RuntimeError(f"plan rows name {p} but there is no capture row (run 1)")
        self._epochs()
        cs = [e for e in self.J["events"].rows() if e.get("event") == "capture_start"]
        dp = C.dataset_problems(self.ns, self.root, (cs[-1].get("dataset_sha256") if cs else None))
        if dp:
            raise ExecAborted("; ".join(dp) + " (edit between Phase A and Phase C: setup and goals would differ from the captured world)")
        if self._invalid_open():
            raise ExecAborted("an earlier invalid run aborted this phase and is unresolved: write the amendment, then --resolve-invalid <A-id>")
        self._treatment_check()
        return guard

    def _treatment_check(self):
        """C4 / Astra Q10: capture and plan rows of THIS treatment identity only (no resume across a change, no mixed journal); test-v1 also
        against freeze.json."""
        rows = list(self.captures.values()) + list(self.plans.values())
        if self.ft2:
            if self.ident is None:
                self.ident = T.identity(self.ns, self.root, briefing=self._briefing_identity())
            bad = T.journal_problems(rows, self.ns, self.ident["sha"]) + T.journal_problems(self.exec_latest().values(), self.ns, self.ident["sha"])
        else:
            bad = T.journal_problems(rows, self.ns)
        fz = {}
        if self.ns == "test-v1" and os.path.exists(C.paths(self.root)["freeze"]):
            with open(C.paths(self.root)["freeze"], encoding="utf-8") as f:
                fz = json.load(f)
        bad += T.entry_problems(self.ns, fz, (self.ident or {}).get("sha"))
        if bad:
            raise ExecAborted("treatment mismatch: " + "; ".join(bad[:4]))

    def briefing_table(self) -> dict:
        if self._btable is None:
            self._btable = T.briefing_hint_table(C.ns_scene(self.ns), allow_unapproved=T.allow_draft_briefing(self.ns))
        return self._btable

    def _briefing_identity(self):
        if self._btable is None:
            return None                                          # identity() reads the briefing itself with the namespace's policy
        return {"records_sha": None, "n_records": None, "allow_unapproved": T.allow_draft_briefing(self.ns),
                "hint_table_sha": T.hint_table_sha(self._btable), "injected": True}

    def hints(self):
        if self._hints is None:
            self._hints = T.hints_fn(self.briefing_table())
        return self._hints

    def _epochs(self):
        self.epoch = {}
        rows = self.J["events"].rows()
        for r in rows:
            if r.get("event") == "invalid_resolved":
                for p in r.get("pres") or []:
                    self.epoch[p] = self.epoch.get(p, 0) + 1
        self.unrecoverable = C.invalid_pres(rows)             # prereg s13: dropped from every arm; the phase does not wait for them

    def _invalid_open(self) -> list:
        return [k for k, r in self.exec_latest().items() if r.get("status") == "invalid" and r.get("pres") not in self.unrecoverable
                and int(r.get("epoch", 0)) >= self.epoch.get(r.get("pres"), 0)]

    def items(self) -> dict:
        if self._items is None:
            self._items = default_items(self.root)
        return self._items

    def tol(self) -> dict:
        if self._tol is None:
            from final_test import reset_gate as rg
            self._tol = rg.load_tolerances(require="validated")                   # only a validated 12.4 may gate an execution
        return self._tol

    def exec_latest(self) -> dict:
        return self.J["exec"].latest()

    def build(self) -> dict:
        man = dict(C.load_manifest(self.root))
        if self.repeat_override is not None:
            man["repeat_subset"] = list(self.repeat_override)
        self.cuts = C.cut_groups(self.J["events"].rows())          # recorded exploratory cuts: not executed, not scored (never scored 0)
        hsha = T.hint_table_sha(self.briefing_table()) if self.ft2 else None
        b = build_units(ns=self.ns, captures=self.captures, plans=self.plans, items=self.items(), snap_sha=self._snap_sha, manifest=man,
                        p_arms=self.p_arms, replay=self.replay, worlds=self.worlds, reloc_seed=self.reloc_seed, cuts=self.cuts, hint_sha=hsha)
        doc = {"namespace": self.ns, "spec": C.SPEC_VERSION, "units": b["units"], "unexecuted": b["unexecuted"],
               "unexecuted_replay": b["unexecuted_replay"], "replay": bool(self.replay),
               "cut_groups": sorted(self.cuts), "cut_plan_cells": b["cut_plan_cells"],
               "plan_only_cells": b.get("plan_only_cells", 0), "runtime_plan": C.RUNTIME_PLAN,
               "worlds": list(self.worlds), "reloc_seed": self.reloc_seed, "caps": C.CAPS, "seed_policy": C.SEED_POLICY,
               "attempts_max": MAX_ATTEMPT, "repeat_rule": REPEAT_RULE}
        if self.ft2:
            doc.update(treatment=T.FT2_VERSION, treatment_id=(self.ident or {}).get("sha"), unit_key_spec=C.UNIT_KEY_SPEC_FT2["version"],
                       hint_table_sha=hsha, hook_arms="+M main, curve, P+M (A25 2 / Astra C4 Q2); prior chooser for every arm")
        _atomic_json(self.units_path, doc)
        self.b = b
        return b

    # ---- done / attempts ---------------------------------------------------------------------------------------------
    def cell_key(self, unit: str, run: int) -> str:
        return f"{self.ns}|exec|{unit}|run{run}"

    def cell_state(self, unit: dict, run: int) -> tuple:
        """-> ('done'|'todo'|'invalid', next attempt). Rows from before the presentation's current epoch are ignored (matched re-run)."""
        k = self.cell_key(unit["unit"], run)
        ep = self.epoch.get(unit["pres"], 0)
        rows = [r for r in self._exec_ix().get(k, ()) if int(r.get("epoch", 0)) >= ep]
        if unit["pres"] in self.unrecoverable:                  # declared unrecoverable (prereg s13): nothing more is run for it
            return "done", (rows[-1]["attempt"] if rows else 0)
        if not rows:
            return "todo", 0
        last = rows[-1]
        if last["status"] in ("done", "not_executed"):
            return "done", last["attempt"]
        if last["status"] == "invalid":
            return "invalid", last["attempt"]
        return ("done", last["attempt"]) if last["attempt"] >= MAX_ATTEMPT else ("todo", last["attempt"] + 1)

    def _exec_ix(self) -> dict:
        """key -> exec rows, read from the journal ONCE (and again only if the file changed under us); `_append` keeps it current. This
        replaces a full re-read per (unit, run) per call, which cost hours on a 4-5 MB journal at every resume."""
        p = self.J["exec"].path
        size = os.path.getsize(p) if os.path.exists(p) else -1
        if self._ix is None or size != self._ix_size:
            ix = {}
            for r in self.J["exec"].rows():
                ix.setdefault(r.get("key"), []).append(r)
            self._ix, self._ix_size = ix, size
        return self._ix

    def todo(self) -> dict:
        """{run: {pres: [unit, ...]}} in `C.exec_blocks` order per run (run 0 first)."""
        out = {}
        for run in range(REPEAT_RUNS):
            by = {}
            for u in self.b["units"].values():
                if run < u["runs"] and self.cell_state(u, run)[0] == "todo":
                    by.setdefault(u["pres"], []).append(u["unit"])
            out[run] = C.exec_blocks(by, self.ns, f"run{run}") if by else []
        return out

    # ---- one execution -----------------------------------------------------------------------------------------------
    def _state_of(self, cap: dict):
        ref = cap.get("state_ref")
        for p in (ref, os.path.join(self.docs, ref or ""), os.path.join(self.docs, "states", os.path.basename(ref or ""))):
            if p and os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                # capture.py writes {"schema": "ftm.capture-state/1", ..., "state": <reset-state>}; a bare reset-state is accepted too
                for k in ("post_sweep_state", "state"):
                    if isinstance(d, dict) and k in d and d.get("schema") != "final-test.reset-state/1":
                        d = d[k]
                        break
                return d
        return None

    def _request(self, u: dict, item: dict, run: int, attempt: int, mem, tag=None):
        import step5_eval as se                                                  # lazy (WSL / Windows both fine)
        key = self.cell_key(u["unit"], run) + (f"|{tag}" if tag else "") + f"|a{attempt}"
        cell = f"{self.ns}|exec|{u['unit']}|run{run}"
        return se.EpisodeRequest(cell=cell, episode_id="ep:" + C.sha256_text(key)[:24], run_id=f"{self.ns}-exec", item=item, lang=u["lang"],
                                 level=item["level"], mode="test", world="reset", world_kind=se.CANONICAL, world_seed=world_seed_of(self.ns, u["pres"]),
                                 policy_seed=0, arm="places" if mem is not None else "none", features=frozenset({"places"}) if mem is not None else frozenset(),
                                 memory=mem, plan_source="cached", kitchen=C.ns_scene(self.ns))

    def _world(self, world: str):
        """The restore context of a presentation: the relocated world re-applies the one fixed InitialRandomSpawn seed right after the clean
        reset, with the SAME context manager Phase A used (capture.relocated_reset), so both phases restore the same world."""
        if world != "reloc":
            return contextlib.nullcontext()
        if self.reloc_seed is None:
            raise ExecInvalid("a relocated presentation needs reloc_seed (freeze.json, or --reloc-seed in a rehearsal)")
        if self.reset_patch is None:
            from capture import relocated_reset                                  # lazy: capture imports the simulator layer only inside functions
            return relocated_reset(self.reloc_seed)
        return self.reset_patch(self.reloc_seed)

    def _executor(self):
        if self.ex is None:
            self.ex = self.factory(self)
        return self.ex

    def run_one(self, u: dict, run: int, attempt: int, *, tag=None) -> tuple:
        """One restore + execution -> (exec_rec, outcome_rec|None, digest). Nothing is journaled here."""
        t0 = time.time()
        cap, item = self.captures[u["pres"]], self.items()[u["item"]]
        ex = self._executor()
        pp = C.parse_pres(u["pres"])
        mem, h0, sha_b = None, None, None
        if u["chooser"] != "prior":
            sha_b = self._snap_sha(u["chooser"])
            if sha_b != u["snapshot_sha"]:
                raise ExecInvalid(f"snapshot {u['chooser']} changed since the units were built ({sha_b[:12]} != {str(u['snapshot_sha'])[:12]})")
        if pp["world"] == "reloc" and not getattr(ex, "supports_reloc", False):
            raise RuntimeError("this executor cannot build the relocation world (E1 has no relocation hook); run with --worlds canon")
        state_a = self._state_of(cap)
        box = {"chk": None, "mismatch": None, "obs": None}
        steps = to_steps(u["plan"])

        def state_check(state, post):
            obs_b = OH.record((self.ident or {}).get("sha"), (post or {}).get("sweep_prims"))      # A34-A2 2(b): this episode's pipeline
            box["obs"] = obs_b
            chk = replay_precondition(state_a, state, cap.get("holding"), (post or {}).get("holding"), self.tol(),
                                      obs_a=cap.get("obs_pipeline"), obs_b=obs_b,
                                      presentation=presentation_of(item))
            box["chk"] = chk
            return chk

        def hook(req, it, holding, ltm, post=None):
            if post is None:
                raise ExecInvalid("replay hook called without a post-sweep dict (plan_after_sweep is off)")
            # W2 rehearsal finding: a byte-hash comparison of the 40-line summary blocked ~90 % of executions, because the agent's depth-derived
            # (x, z) estimates jitter by 0.01-0.04 m between identical restores. A21 C: the pre-execution check is `replay_precondition`
            # (state comparison + ACTUAL holding + told holding, run by `state_check` BEFORE this hook, which blocks the execution on any breach).
            # The hook refuses to run a plan if that check did not run or did not pass, and re-checks the told holding it is handed.
            if box["chk"] is None or not box["chk"].get("ok"):
                raise ExecInvalid("replay hook reached without a passing replay_precondition (state_check did not run, or it breached)")
            if (holding or None) != (cap.get("holding") or None):
                box["mismatch"] = {"kind": "holding", "capture": cap["summary_sha"][:16], "now": post["summary_sha"][:16]}
                raise SummaryMismatch(json.dumps(box["mismatch"]))
            return {"steps": [dict(s) for s in steps], "info": {"replay": True, "unit": u["unit"], "n_steps": len(steps)}}

        ex.state_check, ex.plan_hook = state_check, hook
        OH.install()                                                             # A34-A2: behaviour-preserving pipeline health counters
        OH.reset()
        status, res, err = "done", None, None
        if self.ft2:                                                             # C4: hook B for THIS episode only (or explicitly none)
            T.install_policy(ex, u.get("exec_policy"), self.hints() if u.get("exec_policy") else None)
        try:
            if u["chooser"] != "prior":
                mem = self.opener(self, u["chooser"], f"{u['unit'][:12]}.r{run}.a{attempt}", ex)
                h0 = mem.state_hash()
            req = self._request(u, item, run, attempt, mem, tag)
            with self._world(pp["world"]):
                self.exec_calls += 1                                          # A20.1: every call really reaches the executor
                self.last_episode_id = req.episode_id
                res = ex.execute(req)
        except SummaryMismatch:
            status = "blocked_state"
        except BaseException as e:                                                # noqa: BLE001 - classified below
            if isinstance(e, KeyboardInterrupt):
                raise
            err = f"{type(e).__name__}: {str(e)[:300]}"
            status = "invalid" if is_invariant(e) else "crash"
            if status == "crash":
                self._drop_executor()
        finally:
            sha_a, h1 = (self._snap_sha(u["chooser"]) if u["chooser"] != "prior" else None), None
            if mem is not None:
                with contextlib.suppress(Exception):
                    h1 = mem.state_hash()
                with contextlib.suppress(Exception):
                    mem.close()
        if u["chooser"] != "prior" and (sha_a != sha_b or (h0 is not None and h1 is not None and h0 != h1)):
            status, err = "invalid", f"snapshot {u['chooser']} changed during the run (sha {str(sha_b)[:12]} -> {str(sha_a)[:12]}; state {h0} -> {h1})"
        extra = (getattr(res, "extra", None) or {}) if res is not None else {}
        chk = box["chk"] or ({"ok": False, "breaches": [box["mismatch"]]} if box["mismatch"] else {"ok": None, "breaches": []})
        if status == "done" and res is not None:
            if getattr(res, "inapplicable", None):                                 # setup / gate differs from Phase A: the restore is not the captured world
                status, chk = "blocked_state", {"ok": False, "breaches": [{"kind": "inapplicable_at_restore", "why": str(res.inapplicable)[:160]}]}
            elif extra.get("blocked_state") is not None:
                status = "blocked_state"
            elif extra.get("plan_not_executed"):
                status = "not_executed"
            else:
                aud = (extra.get("core") or {}).get("audit") or {}
                if _audit_bad(aud):
                    status, err = "invalid", f"honesty audit failed: {json.dumps(_clean_audit(aud), default=str)[:200]}"
        core = extra.get("core") or {}
        audit = _clean_audit(core.get("audit") or {})
        wh = getattr(res, "initial_state_hash", None) if res is not None else None
        rec = C.make_record("exec", key=self.cell_key(u["unit"], run), namespace=self.ns, unit=u["unit"], run=run, attempt=attempt, pres=u["pres"],
                            status=status, state_check={"ok": chk.get("ok"), "breaches": (chk.get("breaches") or [])[:20], "max_dpos": chk.get("max_dpos"),
                                                        "max_drot": chk.get("max_drot"), "registered_ok": chk.get("registered_ok"),
                                                        "override": chk.get("override"), "override_why": chk.get("override_why"),
                                                        "obs_drift": chk.get("obs_drift"), "obs_pipeline": box.get("obs")},
                            audit=audit, prims=core.get("prims"), wall_s=round(time.time() - t0, 2), snapshot_sha_before=sha_b, snapshot_sha_after=sha_a,
                            epoch=self.epoch.get(u["pres"], 0), error=err, chooser=u["chooser"], cells=[c["key"] for c in u["cells"]], world_hash_match=(None if wh is None else wh == cap.get("world_hash")),
                            world_seed=world_seed_of(self.ns, u["pres"]))
        last = getattr(ex, "last", None) or {}
        blog = None
        if self.ft2:                                                             # C4: where the briefing entered (hook B audit)
            sk = last.get("skills") if isinstance(last, dict) else None
            blog = list(getattr(sk, "briefing_log", []) or []) if sk is not None else []
            rec.update(treatment=T.FT2_VERSION, treatment_id=(self.ident or {}).get("sha"), exec_policy=u.get("exec_policy"), briefing_log=blog)
        ad = last.get("adapter") if isinstance(last, dict) else None
        outc = None
        if status == "done" and res is not None:
            outc = C.make_record("outcome", key=rec["key"], namespace=self.ns, unit=u["unit"], run=run, goal_ok=bool(res.goal_ok), exec_ok=bool(core.get("exec_ok")),
                                 steps=with_holding(extra.get("steps") or [], getattr(ad, "stream", None)), pres=u["pres"], attempt=attempt, sr=res.sr, prims=core.get("prims"), found=getattr(res, "found", None),
                                 retries=getattr(res, "retries", None), walking_m=getattr(res, "walking_m", None), opens=getattr(res, "opens", None),
                                 stale_visits=getattr(res, "stale_visits", None), gate_pass=getattr(res, "gate_pass", None), applied=getattr(res, "applied", None))
        dig = {"status": status, "prims": core.get("prims"), "audit_violations": audit.get("violations"), "summary_sha": extra.get("summary_sha"), "sweep_prims": extra.get("sweep_prims"),
               "steps": [[s.get("action"), s.get("object"), s.get("to"), bool(s.get("ok")), s.get("cause"), s.get("retries"), s.get("prims")]
                         for s in (extra.get("steps") or [])],
               "audit_counts": audit.get("counts"), "n_events": core.get("n_events"),
               "events_sha": C.sha256_json(list(getattr(ad, "stream", []) or []))[:16] if ad is not None else None,
               "goal_ok": None if res is None else res.goal_ok, "exec_ok": core.get("exec_ok")}
        if self.ft2:
            dig["briefing_log"] = blog
            dig["exec_policy"] = u.get("exec_policy")
        return rec, outc, dig

    def _drop_executor(self):
        if self.ex is not None:
            with contextlib.suppress(Exception):
                self.ex.close()
            self.ex = None

    def _secrets(self) -> tuple:
        if self._secrets_cache is None:
            try:
                self._secrets_cache = (C.thesis_key(),)
            except Exception:                                    # no key in this process (WSL): the patterns alone apply
                self._secrets_cache = ()
        return self._secrets_cache

    def _append(self, kind: str, rec: dict) -> None:
        """Key patterns on structural fields, the exact secret everywhere (free text such as a step cause or an error may hold a harmless
        hyphenated word: `a task-specific plan` must not crash the phase)."""
        hits = C.leak_paths(rec, self._secrets())
        if hits:
            raise ExecInvalid(f"key-like text in a {kind} record ({hits}): nothing written")
        self.J[kind].append(rec)
        if kind == "exec" and self._ix is not None:
            self._ix.setdefault(rec.get("key"), []).append(rec)
            self._ix_size = os.path.getsize(self.J["exec"].path)

    def do_cell(self, u: dict, run: int) -> dict:
        _, attempt = self.cell_state(u, run)
        while attempt <= MAX_ATTEMPT:
            # A36: journal the attempt's START before it runs, so an attempt lost to a hang (exit 75, no exec row; the resume repeats the
            # attempt number) stays visible; scoring never counts a history with an unmatched start as 3 reset-gate stops.
            self.event("exec_attempt_start", key=self.cell_key(u["unit"], run), unit=u["unit"], run=run, attempt=attempt,
                       epoch=self.epoch.get(u["pres"], 0))
            rec, outc, _ = self.run_one(u, run, attempt)
            if outc is not None:
                self._append("outcome", outc)                                        # outcome first; the exec record is the commit marker
            self._append("exec", rec)
            if rec["status"] == "invalid":
                self.event("abort_invalid", pres=u["pres"], unit=u["unit"], run=run, why=rec.get("error"))
                raise ExecAborted(f"invalid run in {u['pres']} unit {u['unit'][:12]} run {run}: {rec.get('error')}")
            if rec["status"] in ("done", "not_executed"):
                return rec
            attempt += 1
            if rec["status"] == "crash":
                self._drop_executor()
        return rec

    # ---- the phase ---------------------------------------------------------------------------------------------------
    def run(self, stop_after: int | None = None, dry_run: bool = False) -> dict:
        guard = self.preflight()
        b = self.build()
        n_cells = sum(u["runs"] for u in b["units"].values())
        info = {"guard": guard, "units": len(b["units"]), "cells": n_cells, "plan_cells": sum(len(u["cells"]) for u in b["units"].values()),
                "unexecuted_plans": len(b["unexecuted"]), "skipped_worlds": b["skipped_worlds"], "executed": 0, "interrupted": False, "complete": False}
        if dry_run:
            info["todo"] = {r: sum(len(us) for _, us in blocks) for r, blocks in self.todo().items()}
            return info
        with C.PhaseLock(self.ns, PHASE, self.root, steal=getattr(self, "steal", False)):
            try:
                for run in range(REPEAT_RUNS):
                    for pres, unit_keys in self.todo()[run]:
                        for uk in unit_keys:
                            if stop_after is not None and info["executed"] >= stop_after:
                                info["interrupted"] = True
                                self.event("interrupt", n=info["executed"])
                                return info
                            rec = self.do_cell(self.b["units"][uk], run)
                            info["executed"] += 1
                            self.out(json.dumps({"t": time.strftime("%H:%M:%S"), "pres": pres, "unit": uk[:10], "run": run, "status": rec["status"],
                                                 "attempt": rec["attempt"], "s": rec["wall_s"]}, ensure_ascii=False))
            finally:
                self._drop_executor()
            remaining = sum(len(us) for blocks in self.todo().values() for _, us in blocks)
            if remaining == 0:
                self._finish(info)
        return info

    def _finish(self, info: dict) -> None:
        for kind in ("exec", "outcome", "events"):
            p = self.J[kind].path
            if os.path.exists(p) and C.scan_jsonl_leaks(p, self._secrets()):
                raise ExecInvalid(f"key-like text found in {p}: phase_complete withheld")
        sha = C.sha256_file(self.J["exec"].path) if os.path.exists(self.J["exec"].path) else ""
        prev = self._complete(PHASE)
        if prev is None or prev.get("journal_sha") != sha:
            self.event("phase_complete", n=info["cells"], journal_sha=sha, units=info["units"])
        info["complete"] = True

    def declare_unrecoverable(self, amendment: str, pres=None) -> dict:
        """Prereg s13: an INVALID presentation that cannot be re-run after its amendment is dropped from EVERY arm (the pairing is kept).
        Records `invalid_unrecoverable` (pres, amendment): execute stops waiting for it, score flags its rows `invalid_episode`, analysis
        drops it. Default: the presentations that are invalid right now."""
        if not str(amendment or "").strip():
            raise ValueError("an unrecoverable presentation needs its amendment id")
        self._epochs()
        cur = sorted({r["pres"] for r in self.exec_latest().values() if r.get("status") == "invalid"
                      and int(r.get("epoch", 0)) >= self.epoch.get(r.get("pres"), 0)})
        pres = sorted(set(pres) if pres else cur)
        if not pres:
            return {"declared": []}
        self.event("invalid_unrecoverable", pres=pres, amendment=str(amendment)[:80], by="Arash")
        return {"declared": pres}

    def resolve_invalid(self, amendment: str) -> dict:
        """Matched re-run after an amendment (prereg s13): every unit of the invalid presentation(s) is run again from scratch."""
        self._epochs()
        pres = sorted({r["pres"] for r in self.exec_latest().values() if r.get("status") == "invalid" and int(r.get("epoch", 0)) >= self.epoch.get(r.get("pres"), 0)})
        if not pres:
            return {"resolved": []}
        self.event("invalid_resolved", pres=pres, amendment=amendment)
        return {"resolved": pres}

    # ---- G-iso (prereg s4 "Isolation audit") -------------------------------------------------------------------------------
    @contextlib.contextmanager
    def iso_condition(self, cond: str):
        """absent: the S* snapshot and sim/step4/honest_memory.db are moved aside for the condition and restored (also after a crash: recovered on entry);
        present: both must exist."""
        paths = [T.source_dir(self.ns, "S*", self.root), os.path.join(C.SIM if self.root is None else os.path.join(self.root, "sim"), "step4", "honest_memory.db")]
        for p in paths:                                                             # crash recovery
            a = p + ".giso-aside"
            if os.path.exists(a) and not os.path.exists(p):
                os.replace(a, p)
        if cond == "present":
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise ExecInvalid(f"G-iso 'present' needs S* and a populated honest_memory.db; missing {missing}")
            yield {"present": True, "shas": [dir_sha(paths[0]), C.sha256_file(paths[1])]}
            return
        moved = []
        try:
            for p in paths:
                if os.path.exists(p):
                    os.replace(p, p + ".giso-aside")
                    moved.append(p)
            yield {"present": False, "shas": [None, None]}
        finally:
            for p in moved:
                os.replace(p + ".giso-aside", p)

    #: A19: the five components G-iso compares. Continuous poses (coordinates in the summary, the sweep event stream, the post-sweep summary hash)
    #: and timings are NOT compared: they jitter between identical restores (W2: the absent-vs-absent control differed in 12 of 12 runs).
    ISO_COMPONENTS = ("request", "status", "goal_reached", "steps", "audit")
    ISO_CONTROL_MAX = 2                                           # A20: the control may differ on at most 2 presentations, only on (iv) steps

    def _none_request_sha(self, cap: dict) -> str:
        """Component (i): sha256 of the `none`-arm request rendered from the Phase A capture record (its frozen summary, instruction and told holding)
        by the one renderer (`PlannerContract`), with NO memory and NO few-shot block. It is rendered inside the condition (S* and honest_memory.db
        moved aside or present), so a renderer that reads either store would change it. The LIVE post-sweep summary is not used: its coordinates jitter."""
        if self.request_renderer is not None:                                        # tests inject a renderer
            return self.request_renderer(cap)
        import plan as PLN                                                           # lazy: the planner layer stays out of this module (Phase C makes no LLM call)
        # A20.1: rebuilt through the FULL construction path on EVERY call (type list read from disk, PlannerContract built, messages rendered, hashed);
        # nothing is cached between conditions, so a store-dependent step anywhere on that path shows up in the condition that has the store.
        if self.ft2:                                                                 # C4: the ft-v2 contract (48 types, ft-v2 text)
            return PLN.none_request_renderer(self.root, ns=self.ns)(cap)
        return PLN.none_request_renderer(self.root)(cap)

    @staticmethod
    def iso_components(dig: dict, request_sha: str) -> dict:
        """The A19 comparison key of one run: (i) request bytes, (ii) run status, (iii) goal reached, (iv) executed step list as (action, object type,
        ok), (v) audit counts. A20: (v) is violations + oracle reads + map-build reads ONLY; motion bookkeeping (pose, actuator, action-resolve, scoring)
        follows the trajectory length and is excluded."""
        cnt = dig.get("audit_counts") if isinstance(dig.get("audit_counts"), dict) else {}
        steps = [[st[0], str(st[1]).split("|")[0] if st[1] is not None else None, bool(st[3])] for st in (dig.get("steps") or [])]
        return {"request": request_sha, "status": dig.get("status"), "goal_reached": dig.get("goal_ok"), "steps": steps,
                "audit": {"violations": dig.get("audit_violations"), "oracle": cnt.get("oracle"), "map-build": cnt.get("map-build")}}   # A20.1: absent -> None, NEVER 0

    @staticmethod
    def iso_instrumentation_missing(dig: dict) -> list:
        """A20.1: the component-(v) keys a run did not report at all (violations, oracle, map-build). A run without them is not 'clean', it is unmeasured.
        A blocked_state run executes nothing and is handled as blocked instead."""
        if dig.get("status") == "blocked_state":
            return []
        cnt = dig.get("audit_counts")
        miss = [] if dig.get("audit_violations") is not None else ["violations"]
        if not isinstance(cnt, dict):
            return miss + ["oracle", "map-build"]
        return miss + [k for k in ("oracle", "map-build") if k not in cnt or cnt[k] is None]

    @classmethod
    def iso_diff(cls, x: dict, y: dict, blocked: bool = False) -> list:
        """The components on which two runs of one presentation differ, over what BOTH runs measured (Astra A20.1 review: a blocked run or a missing audit
        key may prevent a PASS but must not hide an established difference). Audit: only the keys both runs reported. A blocked_state presentation executed
        nothing, so only (i) the request is compared."""
        out = []
        for k in cls.ISO_COMPONENTS:
            if blocked and k != "request":
                continue
            if k == "audit":
                if any(x[k][kk] is not None and y[k][kk] is not None and C.canon(x[k][kk]) != C.canon(y[k][kk]) for kk in x[k]):
                    out.append(k)
            elif C.canon(x[k]) != C.canon(y[k]):
                out.append(k)
        return out

    def g_iso(self, n: int = 12, conds=("absent", "present", "absent")) -> dict:
        """A20 + A20.1. The `0` arms (prior chooser, no memory) run with no S* / honest_memory.db, then with both present, then once more with none (the
        control). For every selected presentation the components (i)-(v) (`ISO_COMPONENTS`) are compared; (v) = violations + oracle + map-build only.
        Run rules (A20.1): the PRESENT run is compared with the FIRST absent run; all 3 x n executions really run (counted at the executor call, unique
        episode ids, no dedup, no cached result); the no-memory request is rebuilt through the full construction path inside every condition; the audit
        is shown once to be sensitive (a forbidden read injected into the real MetaAudit raises the counted violation / oracle numbers); a run that did
        not report violations / oracle / map-build is unmeasured, not zero.
        fail         any absent-vs-present difference on a component both runs measured (takes precedence over a noisy control and over every inconclusive
                     reason; a blocked or partly unmeasured presentation is compared on what is available, it just cannot support a pass);
        pass         no such difference, n as registered, nothing blocked, instrumentation present and shown sensitive, 3 x n real executions, AND the
                     control differs on at most `ISO_CONTROL_MAX` (2) presentations, only on (iv) steps (recorded as simulator noise);
        inconclusive otherwise (control differs on more than 2 or on (i), (ii), (iii), (v); a blocked run; fewer than n; missing or insensitive
                     instrumentation; the executions were not all real). An inconclusive gate is NOT passed: W5 needs Arash's explicit written waiver.
        The old strict digest (all fields incl. event stream sha and post-sweep summary sha) is kept only as an informational field."""
        self.preflight()
        b = self.build()
        zero = [u for u in b["units"].values() if u["chooser"] == "prior" and any(C.arm_info(c["arm"], self.ns)["prompt"] == "none" for c in u["cells"])]   # a `0` arm (or P0)
        if self.ft2:                                       # C4: every ft-v2 unit is prior; P+M (prompt none) runs hook B, so it is not a `0` unit
            zero = [u for u in zero if u.get("exec_policy") is None]
        by_pres = {}
        for u in zero:
            by_pres.setdefault(u["pres"], u)
        sel = C.seeded_order(by_pres, self.ns, "g-iso")[:n]
        runs, exec_log = {}, []
        probe = (self.audit_probe or audit_sensitivity_probe)()                       # once, before the conditions; touches no simulator and no store
        with C.PhaseLock(self.ns, "g-iso", self.root, steal=getattr(self, "steal", False)):
            try:
                for i, cond in enumerate(conds):
                    with self.iso_condition(cond) as env:
                        dg = {}
                        for p in sel:
                            c0 = self.exec_calls
                            dg[p] = self.run_one(by_pres[p], 0, 0, tag=f"giso{i}")[2]
                            exec_log.append({"run": i, "pres": p, "episode": self.last_episode_id, "executor_calls": self.exec_calls - c0})
                        runs[i] = {"cond": cond, "env": env, "digests": dg,
                                   "comps": {p: self.iso_components(dg[p], self._none_request_sha(self.captures[p])) for p in sel}}
            finally:
                self._drop_executor()
        i2 = 2 if len(runs) == 3 else 0
        a, bb, ctl = runs[0]["comps"], runs[1]["comps"], runs[i2]["comps"]
        da, db, dc = runs[0]["digests"], runs[1]["digests"], runs[i2]["digests"]
        blocked = sorted(p for p in sel if "blocked_state" in (da[p]["status"], db[p]["status"], dc[p]["status"]))
        missing = {p: sorted({k for d in (da, db, dc) for k in self.iso_instrumentation_missing(d[p])}) for p in sel}
        missing = {p: v for p, v in missing.items() if v}
        diff_kinds = {p: self.iso_diff(a[p], bb[p], blocked=p in blocked) for p in sel}            # present vs the FIRST absent run, on what both measured
        diff = sorted(p for p in sel if diff_kinds[p])
        noise = sorted(p for p in sel if C.canon(a[p]) != C.canon(ctl[p]))
        noise_kinds = {p: [k for k in self.ISO_COMPONENTS if C.canon(a[p][k]) != C.canon(ctl[p][k])] for p in noise}
        noise_steps_only = sorted(p for p in noise if noise_kinds[p] == ["steps"])
        noise_bad = sorted(p for p in noise if noise_kinds[p] != ["steps"])
        noise_too_many = len(noise) > self.ISO_CONTROL_MAX
        short = len(sel) < n
        # (i) is a probe of the renderer: also tie it to the bytes Phase A recorded for the none arm, in each condition (None = the capture holds no hash)
        rec_none = {p: ((self.captures[p].get("requests") or {}).get("none")) for p in sel}
        req_vs_phase_a = {runs[i]["cond"] + f"#{i}": sorted(p for p in sel if rec_none[p] and runs[i]["comps"][p]["request"] != rec_none[p]) for i in sorted(runs)}
        req_drift = sorted({p for v in req_vs_phase_a.values() for p in v})
        eps = [e["episode"] for e in exec_log]
        executions = {"expected": len(conds) * len(sel), "executor_calls": sum(e["executor_calls"] for e in exec_log), "unique_episode_ids": len(set(eps)),
                      "all_real": len(exec_log) == len(conds) * len(sel) and all(e["executor_calls"] == 1 for e in exec_log) and len(set(eps)) == len(eps),
                      "dedup": False, "result_cache": False}
        sens_ok = bool(probe.get("ok"))
        reasons = [r for r, bad in (("no presentations", not sel), ("blocked run", bool(blocked)), ("fewer presentations than registered", short),
                                    ("instrumentation missing", bool(missing)), ("audit insensitive", not sens_ok), ("request probe differs from the Phase A none-arm bytes", bool(req_drift)),
                                    ("executions not all real", not executions["all_real"]), ("control differs outside (iv)", bool(noise_bad)),
                                    ("control differs on more than 2", noise_too_many)) if bad]
        hook_leak = []
        if self.ft2:                                       # C4 (A20.1 for ft-v2): a 0-arm unit has no hook and an empty briefing log
            hook_leak = sorted(p for p in sel if by_pres[p].get("exec_policy") is not None
                               or any((runs[i]["digests"][p].get("briefing_log") or []) for i in runs))
        if diff or hook_leak:
            status = "fail"
        elif reasons:
            status = "inconclusive"
        else:
            status = "pass"
        strict_diff = sorted(p for p in sel if C.canon(da[p]) != C.canon(db[p]))
        strict_noise = sorted(p for p in sel if C.canon(da[p]) != C.canon(dc[p]))
        comps = {}
        if self.ns != "test-v1":                                   # components (ii)-(v) hold outcome fields: dev reports only
            for k in self.ISO_COMPONENTS:
                comps[k] = {"absent_vs_present": sum(1 for q in sel if C.canon(a[q][k]) != C.canon(bb[q][k])),
                            "absent_vs_control_absent": sum(1 for q in sel if C.canon(a[q][k]) != C.canon(ctl[q][k]))}
        rep = {"schema": "ftm.g_iso/2", "rule": "A20.1", "namespace": self.ns, "n": len(sel), "n_registered": n, "pres": sel, "status": status, "diff": diff, "diff_components": {p: v for p, v in diff_kinds.items() if v},
               "request_probe_vs_phase_a": req_vs_phase_a, "inconclusive_reasons": reasons, "anchor": "present compared with the FIRST absent run (condition 0); the second absent run is the control",
               "control_noise": noise, "control_noise_components": noise_kinds, "control_noise_steps_only": noise_steps_only, "control_noise_bad": noise_bad,
               "control_noise_max": self.ISO_CONTROL_MAX, "blocked": blocked, "short": short, "instrumentation_missing": missing,
               "executions": executions, "audit_sensitivity": probe,
               "conditions": [{"cond": r["cond"], "present": r["env"]["present"], "shas": r["env"]["shas"]} for r in runs.values()],
               "component_sha": {p: C.sha256_json([runs[i]["comps"][p] for i in sorted(runs)])[:16] for p in sel}, "components": comps,
               "compared": list(self.ISO_COMPONENTS),
               "excluded": "motion bookkeeping audit counts (pose, actuator, action-resolve, scoring), continuous poses, summary coordinates, sweep event stream, post-sweep summary sha, timings",
               "strict_digest_informational": {"diff": strict_diff, "control_noise": strict_noise,
                                               "digest_sha": {p: C.sha256_json([runs[i]["digests"][p] for i in sorted(runs)])[:16] for p in sel}},
               "note": "A20.1 (A20 components of A19): (v) = violations + oracle + map-build reads only, an unreported key is None (never 0) and makes the presentation unmeasured; "
                       "control may differ on <= 2 presentations, only on steps (unexplained execution variability consistent with simulator noise, NOT a calibrated tolerance: 0/12 "
                       "mismatches still allows a one-sided 95 % upper bound of about 22 %); an absent-vs-present difference takes precedence over any inconclusive reason; "
                       "(i) is a renderer probe rebuilt through the full construction path in every condition from the frozen Phase A capture record (execution builds no prompt, "
                       "the live summary jitters); a blocked_state run makes the result inconclusive (eligibility rule A19 does not state: needs Arash's OK); an INCONCLUSIVE gate "
                       "is not passed (A20.1: W5 only with Arash's explicit written waiver, claims about memory isolation narrowed); the strict digest is informational only"}
        if self.ns != "test-v1":                                   # diagnosis only (dev): the component values themselves, so a mismatch can be read
            _atomic_json(os.path.join(self.docs, "g_iso_components.json"), {"pres": sel, "conds": [runs[i]["cond"] for i in sorted(runs)],
                         "comps": {p: [runs[i]["comps"][p] for i in sorted(runs)] for p in sel}, "exec_log": exec_log})
        if self.ft2:
            rep["hook_b_on_zero_arm"] = hook_leak
        rep["sha256"] = C.self_sha(rep)
        _atomic_json(os.path.join(self.docs, "g_iso.json"), rep)
        self.event("g_iso", status=status, n=len(sel), diff=diff, rule="A20.1", reasons=reasons)
        return rep


def audit_sensitivity_probe() -> dict:
    """A20.1 audit sensitivity: a forbidden read injected into the REAL `honest.MetaAudit` (through `HonestAgent.meta`, the one door the executor's
    audit uses) must raise the counted violation and oracle / map-build numbers that G-iso component (v) compares, and `_audit_bad` must call the
    episode invalid. Also: a strict audit raises. A silent audit (counts that stay 0 under a forbidden read) would make 'identical audit counts'
    meaningless. Touches no simulator: the agent wraps a stub with empty metadata."""
    import honest as H
    stub = SimpleNamespace(event=SimpleNamespace(metadata={"objects": [], "actionReturn": None}))

    def agent(strict):
        return H.HonestAgent(stub, detector=object(), mode="honest", audit=H.MetaAudit(strict=strict))

    def key(h):
        s = h.audit.summary()
        aud = _clean_audit({**s, "counts": dict(h.audit.counts)})
        dig = {"status": "done", "goal_ok": True, "steps": [], "audit_counts": aud.get("counts"), "audit_violations": aud.get("violations")}
        return ExecutePhase.iso_components(dig, "r")["audit"], _audit_bad(aud)

    clean = agent(False)
    base, base_bad = key(clean)
    dirty = agent(False)
    dirty.meta("oracle")                                           # an oracle read in honest mode
    dirty.meta("map-build", ["actionReturn"])                      # a map-build return read with no GetReachablePositions before it
    inj, inj_bad = key(dirty)
    raised = False
    try:
        agent(True).meta("oracle")
    except H.HonestyViolation:
        raised = True
    ok = (base == {"violations": 0, "oracle": 0, "map-build": 0} and not base_bad and (inj["violations"] or 0) >= 2 and (inj["oracle"] or 0) >= 1
          and (inj["map-build"] or 0) >= 1 and inj != base and inj_bad and raised)
    return {"ok": bool(ok), "clean_key": base, "injected_key": inj, "injected_flagged_invalid": bool(inj_bad), "strict_audit_raises": raised,
            "injected": ["oracle read in honest mode", "map-build return read without GetReachablePositions"]}


# ================================================================================================ live factories (WSL only)
def live_executor_factory(phase: ExecutePhase):
    """One LiveExecutor for the whole phase (memory="places" serves both choosers: a request with memory=None is the no-memory arm). `run_one`
    replaces `ex.state_check` and `ex.plan_hook` per episode; the constructor placeholder only has to be non-None."""
    import step5_live as sl
    from final_test import reset_gate as rg                                       # noqa: F401  (fail early if E3 is absent)
    ex = sl.LiveExecutor(memory="places", policy="console", hints=False, hint_goals=False, timeout=C.CAPS["episode_wall_s"],
                         plan_after_sweep=True, state_capture=True, scene=C.ns_scene(phase.ns),
                         state_check=lambda state, post: {"ok": False, "breaches": [{"kind": "state_check_not_set"}]})
    ex.supports_reloc = True                                                       # relocation = capture.relocated_reset around ex.execute (see ExecutePhase._world)
    return ex


def live_memory_opener(phase: ExecutePhase, source: str, tag: str, ex):
    import ltm as L
    import step5_live as sl
    d = os.path.join(phase.work, "testlogs")
    os.makedirs(d, exist_ok=True)
    snap = T.source_dir(phase.ns, source, phase.root) if C.is_replication(phase.ns) else C.snapshot_dir(source, phase.root)   # K23
    m = L.LongTermMemory.open_frozen(snap, os.path.join(d, f"{tag}.db"), kitchen=C.ns_scene(phase.ns),
                                     params=L.load_params())
    sl.assert_b_memory(m)
    m.begin_test_epoch(frame=ex.frame, map_rev=ex.map_rev)
    return m


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase C: execute cached plans in identical restored worlds (WSL only; no LLM).")
    ap.add_argument("--namespace", required=True, choices=sorted(C.NAMESPACES))
    ap.add_argument("--items", choices=("dev", "test"), default=None, help="consistency check against the namespace")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build units and print counts; no simulator")
    ap.add_argument("--worlds", default="canon", help="canon[,reloc]")
    ap.add_argument("--reloc-seed", default=None)
    ap.add_argument("--p-arms", default=",".join(P_ARMS))
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--inject-interrupt", type=int, default=None, metavar="N", help="stop after N executions (rehearsal resume test)")
    ap.add_argument("--partial", action="store_true", help="execute the plans that exist; needs a recorded stopping-rule event (prereg s13)")
    ap.add_argument("--g-iso", action="store_true")
    ap.add_argument("--g-iso-n", type=int, default=12)
    ap.add_argument("--resolve-invalid", default=None, metavar="AMENDMENT")
    ap.add_argument("--declare-unrecoverable", default=None, metavar="AMENDMENT",
                    help="Arash only: the invalid presentation(s) cannot be re-run; drop them from every arm (prereg s13)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    if (a.items == "test") != (a.namespace == "test-v1") and a.items is not None:
        print(f"--items {a.items} does not fit namespace {a.namespace}", file=sys.stderr)
        return EXIT["refused"]
    worlds = tuple(w for w in a.worlds.split(",") if w)
    seed = a.reloc_seed
    if seed is None and "reloc" in worlds:
        with contextlib.suppress(Exception):
            seed = json.load(open(C.paths(a.root)["freeze"], encoding="utf-8")).get("reloc_seed")
    ph = ExecutePhase(a.namespace, root=a.root, allow_test=a.allow_test, worlds=worlds, reloc_seed=seed, p_arms=tuple(x for x in a.p_arms.split(",") if x and x != "none"),
                      replay=not a.no_replay, partial=a.partial, executor_factory=None if a.dry_run else live_executor_factory,
                      memory_opener=None if a.dry_run else live_memory_opener)
    ph.steal = a.steal_lock
    try:
        if a.declare_unrecoverable:
            print(json.dumps(ph.declare_unrecoverable(a.declare_unrecoverable)))
            return EXIT["ok"]
        if a.resolve_invalid:
            ph._epochs()
            print(json.dumps(ph.resolve_invalid(a.resolve_invalid)))
            return EXIT["ok"]
        if a.g_iso:
            rep = ph.g_iso(n=a.g_iso_n)
            print(json.dumps({k: rep[k] for k in ("status", "n", "diff", "control_noise", "blocked")}))
            return EXIT["ok"] if rep["status"] == "pass" else EXIT["abort_invalid"]
        info = ph.run(stop_after=a.inject_interrupt, dry_run=a.dry_run)
        print(json.dumps({k: v for k, v in info.items() if k != "guard"}))
        return EXIT["ok"]
    except C.TestItemRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT["refused"]
    except C.PhaseLocked as e:
        print(f"LOCKED: {e}", file=sys.stderr)
        return EXIT["refused"]
    except ExecAborted as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return EXIT["abort_invalid"]
    except ExecInvalid as e:                                  # e.g. a real key in a record: recorded, no traceback, and the resume refuses cleanly
        with contextlib.suppress(Exception):
            ph.event("abort_key_leak" if "key-like" in str(e) else "abort_invalid", msg=str(e)[:300])
        print(f"ABORTED (integrity): {e}", file=sys.stderr)
        return EXIT["abort_invalid"]


if __name__ == "__main__":
    sys.exit(main())
