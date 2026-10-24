"""Step 5 / B1 - the live executor for workflow B (docs/step5-plan.md s13 "B0-plan").

`make_executor(memory="places"|"none", ...)` returns an `Executor` for `step5_eval.EvalRunner`; `main()` is the B2 smoke CLI.
It MIRRORS the loop of `step4_refplans` (same reset, same applicability gate, same oracle SETUP, same honest episode, same goal
check) without editing it. With `memory="none"` and no hints the agent is exactly step 4's console-policy agent; the memory arm
differs only by (1) a chooser that puts learned T0 / T2 place types first and (2) navigation hints. Both are inputs to the
agent's search; neither is a prompt (reference plans: no planner), and memory reaches a planner ONLY through
`render_planner_messages` below, i.e. `planners.contract.PlannerContract(memory_block=...)` (gate 12.11).

Honesty: the decision path (everything outside `_score*` functions and the world-prep helpers) never names the controller,
`metadata`, `last_event` or `state_readers`; `h.scoring()` appears only inside `_score*` functions (statically gated in
`_step5_adapter_selftest.py`). The goal verdict leaves the episode as a bool (`EpisodeResult.admission`) and reaches the working
memory only as `admission_verdict` after `terminal`, in practice (the working memory refuses it otherwise).

Simulator runs happen ONLY in WSL (see the B2 stage); every heavy import below is inside a function so the offline selftests can
import this module on Windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ltm_adapter as la                                   # noqa: E402
import step5_eval as se                                    # noqa: E402

SCENE = "FloorPlan1"
EP_PRIM_CAP = 3000                                         # as step4_refplans
LEVELS = ("L1", "L2", "L3")


class LiveInvariant(RuntimeError):
    """A per-episode invariant failed (honesty audit, rejected event, hook error): the run aborts, resumable."""


# ================================================================================================ the honest part
def start_core(*, h, sk, adapter, holding, text, level, lang, chooser=None) -> dict:
    """The start of an honest episode: task context, then the fixed start sweep (`HonestSkills.start_episode`). Returns the sweep result
    (`tracks`, `summary`). The ONLY place that sweeps; the old order (`plan_after_sweep=False`) reaches it through `run_episode_core`."""
    adapter.on_task(text, level, lang)
    seed = sk.start_episode(f"s5:{adapter.episode}", holding=holding)
    _check(h, sk, adapter, chooser)
    return seed


def close_unexecuted(*, h, sk, adapter, practice, chooser=None) -> dict:
    """A11: an episode whose plan was not executed (rejected / infra / blocked state) after the sweep. No step runs and there is NO scoring
    call (the goal is false at reset: the applicability gate refused otherwise). The episode closes normally with exec_ok False; in
    practice the admission verdict is False (a started attempt, its honest sweep sightings are consolidated as a failed attempt)."""
    _check(h, sk, adapter, chooser)
    counts = dict(h.audit.counts)
    invalid = bool(h.audit.invalid or int(h.audit.summary().get("violations") or 0) > 0
                   or int(counts.get("oracle") or 0) > 0 or int(counts.get("map-build") or 0) > 0)
    if invalid:
        raise LiveInvariant(f"honesty audit failed in {adapter.episode}: {h.audit.summary()}")
    adapter.on_terminal(False, h.step, adapter.n_opens, adapter.walk_m, invalid=False)
    if practice:
        adapter.on_admission(False)
    adapter.end()
    _check(h, sk, adapter, chooser)
    return {"prims": h.step, "audit": {**h.audit.summary(), "counts": counts}, "metrics": adapter.metrics(), "n_events": len(adapter.stream)}


def post_sweep_dict(*, seed, adapter, holding, sweep_prims) -> dict:
    """What the plan hook receives under A11: the observation summary right after the fixed sweep, the told holding, the sweep's validated
    event stream and the live working memory. Nothing from the scoring side (goal truth, oracle state, the oracle-derived world hash) is
    in it (E-fix): exactly summary, summary_sha, holding, events, wm, episode_id, sweep_prims, seed_tracks."""
    summary = list(seed.get("summary") or [])
    return {"summary": summary, "summary_sha": hashlib.sha256(chr(10).join(summary).encode("utf-8")).hexdigest(), "holding": holding,
            "events": list(adapter.stream), "wm": adapter.wm, "episode_id": adapter.episode, "sweep_prims": int(sweep_prims),
            "seed_tracks": seed.get("tracks")}


def render_blocks(post, sources, level, lang, channels, holding=None) -> dict:
    """The memory block of SEVERAL frozen long-term memories from the SAME post-sweep evidence: {source: block}. Each source gets a fresh
    test-mode working memory fed the sweep's event stream; the live working memory is never touched. Deterministic; works when the
    executor ran with memory="none". For the live LTM it equals `ltm.memory_block(post["wm"], ...)`."""
    out = {}
    for name, src in sources.items():
        if getattr(src, "frozen", False) and getattr(src, "_overlay", None) is None:      # E-fix: else practice claims would read as current (T0)
            raise LiveInvariant(f"render_blocks: frozen source {name!r} has no test-epoch overlay (call begin_test_epoch(frame, map_rev) "
                                "on its fresh read-only copy first); it would render the wrong epoch")
        wm = src.begin_episode(post["episode_id"], "test")
        for ev in post["events"]:
            wm.ingest(dict(ev))
        out[name] = src.memory_block(wm, level, lang, channels=channels, holding=holding)
    return out


def run_episode_core(*, h, sk, adapter, steps, holding, text, level, lang, score, practice, ep_prim_cap=EP_PRIM_CAP,
                     policy="console", chooser=None, started=None, wall_cap_s=None, t_start=None) -> dict:
    """The honest episode over an already-built agent. Pure of the simulator API: `h` (HonestAgent-like), `sk`
    (HonestSkills wired to `adapter` through `on_event` and an ObjectMemoryProxy), `score()` -> bool (the executor's scoring block).
    Returns a row; raises LiveInvariant on any broken invariant BEFORE anything can be consolidated."""
    _t_core = time.time() if t_start is None else t_start
    seed = start_core(h=h, sk=sk, adapter=adapter, holding=holding, text=text, level=level, lang=lang, chooser=chooser)         if started is None else started                     # A11: the caller already swept (plan_after_sweep); never sweep twice
    rows, retries, budget_hit = [], 0, False
    wall_hit = False
    for i, st in enumerate(steps):
        # K23 (Astra K23-A29-APPLIC-r1 item 4): the registered 900 s episode wall cap (C.CAPS episode_wall_s), checked between steps
        # like the primitive cap; None = not checked (the pre-K23 behaviour; no Kitchen 1 test execution exceeded 78 s)
        if not budget_hit and wall_cap_s is not None and time.time() - _t_core > wall_cap_s:
            budget_hit = wall_hit = True
        if budget_hit or h.step > ep_prim_cap:
            budget_hit = True
            rows.append({"i": i, "action": st["action"], "object": st.get("object"), "to": st.get("to"), "ok": False,
                         "cause": "episode_budget", "retries": 0, "prims": 0, **({"under": "episode_wall"} if wall_hit else {})})
            continue
        adapter.on_step_start(i, st, {"holding": h.holding})
        r = sk.exec_step(st, upcoming=list(steps[i + 1:])) if policy == "console" else sk.exec_step(st)
        adapter.on_step_end(i, r)
        d = r["detail"]
        rows.append({"i": i, "action": st["action"], "object": st.get("object"), "to": st.get("to"), "ok": r["ok"],
                     "cause": r["cause"], "under": d.get("under"), "retries": r["retries"], "prims": d["prims"]})
        retries += int(r["retries"])
        _check(h, sk, adapter, chooser)
        if h.audit.invalid:
            break
    _check(h, sk, adapter, chooser)
    goal_ok = bool(score())                                 # the executor's scoring block; a bool, never fed back
    exec_ok = all(r["ok"] for r in rows)
    counts = dict(h.audit.counts)
    invalid = bool(h.audit.invalid or int(h.audit.summary().get("violations") or 0) > 0
                   or int(counts.get("oracle") or 0) > 0 or int(counts.get("map-build") or 0) > 0)
    if invalid:
        raise LiveInvariant(f"honesty audit failed in {adapter.episode}: {h.audit.summary()}")
    adapter.on_terminal(exec_ok, h.step, adapter.n_opens, adapter.walk_m, invalid=False)
    if practice:
        adapter.on_admission(goal_ok)
    adapter.end()
    _check(h, sk, adapter, chooser)
    return {"steps": rows, "goal_ok": goal_ok, "exec_ok": exec_ok, "prims": h.step, "retries": retries,
            "seed_tracks": seed.get("tracks"), "audit": {**h.audit.summary(), "counts": counts},
            "metrics": adapter.metrics(), "n_events": len(adapter.stream),
            "chooser": (chooser.decisions if chooser is not None and hasattr(chooser, "decisions") else None)}


def _check(h, sk, adapter, chooser=None):
    """Fail fast (per step): rejected events, a failing hook, a failing adapter callback or a failing LTM chooser are broken
    invariants, not noise (the search policies swallow chooser exceptions and would silently fall back otherwise)."""
    if adapter.rejected:
        raise LiveInvariant(f"{len(adapter.rejected)} event(s) refused by the contract: {adapter.rejected[:2]}")
    if adapter.errors:
        raise LiveInvariant(f"adapter callback failed: {adapter.errors[:2]}")
    if sk.event_errors:
        raise LiveInvariant(f"on_event / place_hints failed: {sk.event_errors[:2]}")
    if chooser is not None and getattr(chooser, "errors", None):
        raise LiveInvariant(f"LTM chooser failed (the search would have silently used the prior): {chooser.errors[:2]}")


def aggregate_result(core: dict, *, include_search: bool = True) -> dict:
    """Core row -> the cost fields of an EpisodeResult (B-h). `found` is None when the episode made no search."""
    m = core["metrics"]
    found = None if not m["n_searches"] else (m["censored"] == 0)
    return {"sr": 1.0 if core["goal_ok"] else 0.0, "goal_ok": core["goal_ok"], "found": found,
            "steps_to_find": float(m["steps_to_find"]) if m["n_searches"] else None, "walking_m": m["walk_m"],
            "opens": float(m["opens"]), "stale_visits": float(m["stale_visits"]), "retries": float(core["retries"]),
            "steps": float(core["prims"])}


# ------------------------------------------------------------------------------------------------ one renderer (12.11)
def render_planner_messages(contract, command, ltm, wm, *, level, lang, channels, memory_summary=(), holding=None) -> tuple:
    """Memory reaches a planner ONLY here: the facade's composed block becomes `PlannerContract(memory_block=...)`.
    Returns (messages, block). An empty block (nothing learned) gives the byte-identical no-memory prompt."""
    blk = ltm.memory_block(wm, level, lang, channels=channels, holding=holding)
    return contract.messages(command, memory_summary, holding, memory_block=blk["text"] or None), blk


def bilingual_check(ltm, wm, level, channels) -> dict:
    """Gate 12.2 style (offline, per test): EN and FA render the same record identities, order and qualifiers."""
    en, fa = ltm.memory_block(wm, level, "en", channels=channels), ltm.memory_block(wm, level, "fa", channels=channels)
    ids = lambda b: [r["id"] for r in b["records"]]                    # noqa: E731
    return {"same_ids": ids(en) == ids(fa), "same_dropped": [d["id"] for d in en["dropped"]] == [d["id"] for d in fa["dropped"]],
            "tokens_en": en["tokens"], "tokens_fa": fa["tokens"], "n_records": len(en["records"]), "empty": en["empty"]}


def _bilingual_report(ltm, wm, level) -> dict:
    """The per-episode bilingual diagnostic. Under A12 retention the block needs the real max-token counter, which an execution process does not
    carry (Phase A rendered every prompt); then the diagnostic is skipped and says so, instead of crashing the execution (W2 rehearsal finding)."""
    try:
        return bilingual_check(ltm, wm, level, ("places",))
    except ValueError as e:
        if "real max-token counter" in str(e) or "BOTH tokenizers" in str(e):
            return {"skipped": "no real token counter in this process (A12 retention)"}
        raise


def twin_check(ltm, wm_en, wm_fa, level, channels) -> dict:
    """Gate 12.2 as registered, TWIN form (E-fix): the block of the EN presentation of an item (its working memory, its own parse of the EN
    instruction) against the block of the FA presentation (its own working memory and parse). `bilingual_check` compares the EN and FA
    renderings of ONE working memory, which agree by construction and so cannot fail. Offline per pair; reports per-language emptiness."""
    en = ltm.memory_block(wm_en, level, "en", channels=channels)
    fa = ltm.memory_block(wm_fa, level, "fa", channels=channels)
    ids = lambda b: [r["id"] for r in b["records"]]                    # noqa: E731
    return {"same_ids": ids(en) == ids(fa), "ids_en": ids(en), "ids_fa": ids(fa), "empty_en": en["empty"], "empty_fa": fa["empty"],
            "same_dropped": [d["id"] for d in en["dropped"]] == [d["id"] for d in fa["dropped"]]}


# ------------------------------------------------------------------------------------------------ route / chain reports
def route_diff(rows_a, rows_b) -> list:
    """Per search (by order): did the ORDER of inspected containers change? rows = adapter.searches (ordered `inspected`)."""
    out = []
    for i, (a, b) in enumerate(zip(rows_a, rows_b)):
        ia = [(t, tuple(round(v, 1) for v in x)) for t, x in a.get("inspected", [])]
        ib = [(t, tuple(round(v, 1) for v in x)) for t, x in b.get("inspected", [])]
        out.append({"i": i, "otype": a.get("otype"), "inspected_a": ia, "inspected_b": ib, "prims_a": a.get("steps_to_find"),
                    "prims_b": b.get("steps_to_find"), "changed": ia != ib})
    return out


def chain_report(practice_ltm, test_adapter, chooser, otype=None, search_index=None) -> dict:
    """Astra A7: observation -> claim -> where_is -> LtmChooser order / hint -> the search's first visit -> sighting, as plain
    data from the databases and the working memory. Linked precisely: the decision is the chooser's call that belongs to the
    chosen search (the k-th search of that type), its claim is the one at the decision's first learned place, and the first
    visit / sighting are the first events AFTER that search started."""
    decs = [d for d in chooser.decisions if d["learned"] and (otype is None or d["target"] == otype)]
    if not decs:
        return {"found_chain": False, "reason": "no learned candidate was offered for the target"}
    dec = decs[0] if search_index is None else decs[search_index]
    pid, tier, rtype = dec["learned"][0]
    store = practice_ltm.store
    claims = [r for r in store.select("location_claim", where={"kitchen": practice_ltm.kitchen, "otype": dec["target"]})
              if r["place"] == pid]
    if not claims:
        return {"found_chain": False, "reason": f"no location claim for {dec['target']} at {pid} in the practice database"}
    ev_rows = []
    for c in claims:
        for e in store.select("evidence", where={"claim_table": "location_claim", "claim": f"location_claim/{c['claim']}"}):
            wl = [w["wl"] for w in store.select("write_log", where={"event_id": e["event_id"]})]
            ev_rows.append({"claim": c["claim"], "place": c["place"], "rule": e["rule"], "mutation": e["mutation"],
                            "episode": e["episode"], "src_ev": e["src_ev"], "event_id": e["event_id"], "write_log": wl})
    place = store.select("place_instance", where={"place": pid})
    searches = [x for x in test_adapter.searches if x["otype"] == dec["target"]]
    srch = searches[0] if search_index is None else searches[search_index]
    t0 = srch["t0"]
    first_sighting = next((e for e in test_adapter.stream if e["kind"] == "sighting" and e["otype"] == dec["target"] and e["t"] >= t0), None)
    return {"found_chain": True, "target": dec["target"], "claim": claims[0]["claim"], "claim_place": pid, "tier": tier, "rtype": rtype,
            "place_xyz": [place[0]["cx"], place[0]["cy"], place[0]["cz"]] if place else None, "evidence": ev_rows,
            "chooser_order": dec["order"], "prior_order": dec["prior_order"], "changed": dec["changed"], "n_hints": dec["n_hints"],
            "first_visit": srch["visit_rows"][0] if srch["visit_rows"] else None, "search": srch,
            "first_sighting_ev": first_sighting["ev"] if first_sighting else None}


# ================================================================================================ the live executor
B_EPISODE_RE = re.compile(r"^(ep:[0-9a-f]{24}|smoke-.+)$")      # runner cells and smoke runs; replay seeds are `rp:...`


def assert_b_memory(ltm) -> None:
    """B0: a B memory may hold ONLY B episodes. A database carrying any other consolidation receipt (a step5_replay seed
    `rp:...`, an A-stage fixture) is refused: reference-plan executions are gold supervision and never become evaluation memory."""
    rows = ltm.store.select("commit_receipt")
    bad = sorted({r["episode"] for r in rows if r.get("advance_clock") and r.get("episode") and not B_EPISODE_RE.match(str(r["episode"]))})
    if bad:
        raise LiveInvariant(f"memory holds non-B episode receipts (replay / A-stage seed?): {bad[:3]}")


def sha_file(path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "absent"


def scene_problems(scene: str, req=None, nav_map=None) -> list:
    """K23 (Astra K23-AGENT-r1 1): the controller scene, the request's kitchen, the memory's kitchen and the navigation map's scene
    must be ONE kitchen. -> list of problems ([] = consistent). Pure (no simulator); every Kitchen 1 path gives []."""
    out = []
    k = getattr(req, "kitchen", None)
    if k is not None and k != scene:
        out.append(f"request kitchen {k!r} != controller scene {scene!r}")
    mk = getattr(getattr(req, "memory", None), "kitchen", None)
    if mk is not None and mk != scene:
        out.append(f"memory kitchen {mk!r} != controller scene {scene!r}")
    if nav_map is not None and getattr(nav_map, "scene", None) != scene:
        out.append(f"navigation map scene {getattr(nav_map, 'scene', None)!r} != controller scene {scene!r}")
    return out


def not_executed_reason(plan_info) -> str:
    """Why a plan hook returned `steps=None`: the hook says so explicitly (`info["not_executed"]` in {"reuse", "rejected", "infra"});
    otherwise `infra_failure` marks infra and everything else is a rejection. `blocked_state` is never a hook reason: only the state
    check raises it."""
    info = plan_info or {}
    r = info.get("not_executed")
    if r in ("reuse", "rejected", "infra"):
        return r
    return "infra" if info.get("infra_failure") else "rejected"


class LiveExecutor(se.Executor):
    """Runs one episode of an `EpisodeRequest` on the live simulator (WSL). `memory` is the arm's capability: "none" never
    reads long-term memory (same working memory, map, detector, priors, budgets), "places" uses the T0 / T2 channel."""
    kind = "live"

    def __init__(self, memory="none", policy="console", hints=False, scene=SCENE, out_dir=None, timeout=900,
                 allow=None, forbid=frozenset(), hint_boost=None, hint_goals=False,
                 plan_after_sweep=False, state_capture=False, state_check=None, ltm_chooser=True):
        # A11 (prereg s16): True = the planner runs AFTER the fixed start sweep and sees the post-sweep summary. False (default) is the
        # pre-A11 order, byte-for-byte, so the C3 / C4 records stay reproducible. Final-test runners set it ONLY from the SPEC.
        self.plan_after_sweep = bool(plan_after_sweep)
        self.state_capture = bool(state_capture)               # A14: record the post-sweep state (scoring side; never given to the hook)
        self.state_check = state_check                         # callable(state, post) -> {"ok", "breaches"}; not ok blocks planning and execution
        if (state_capture or state_check is not None) and not plan_after_sweep:
            raise ValueError("state_capture / state_check describe the post-sweep state: they need plan_after_sweep=True")
        self.hint_goals = bool(hint_goals)                    # B2: remembered places become look goals (default OFF)
        # C4 / ft-v2 (default OFF, the legacy path is unchanged): ltm_chooser=False keeps a request's memory for LEARNING (the adapter's
        # working memory, practice consolidation) but plans the search with the prior chooser (practice-v2 runs prior + hook B).
        self.ltm_chooser = bool(ltm_chooser)
        # C4 / ft-v2: a per-episode executor policy (hook B: {"briefing_hints": fn}), set by the runner right before execute() and
        # CONSUMED by that one episode (it can never leak into the next one). None = HonestSkills gets no briefing keyword at all.
        self._episode_policy = None
        self.log = None                                       # B2 debugging: a callable(str) handed to HonestSkills(log=...)
        self.plan_hook = None                                 # C3: callable(req, it, holding, ltm[, post=post_dict under A11]) -> {"steps": [...]|None, "info": {...}}; None = reference plan
        self.hint_boost = hint_boost                          # B2 sensitivity analysis; None = honest_skills.HINT_BOOST
        if memory not in ("none", "places"):
            raise ValueError("memory must be 'none' or 'places' (habits, lessons and plans stay dormant in B1)")
        self.memory, self.policy, self.hints, self.scene, self.timeout = memory, policy, bool(hints), scene, timeout
        self.out_dir = out_dir
        self.allow, self.forbid = allow, forbid
        self._ag = self._det = self._gm_cls = None
        self.frame = "f0"
        self.map_rev = "m0"
        self.detector_rev = "unknown"
        self.contract = None
        self.last = None                      # the last episode's adapter / chooser / skills (the smoke trace reads them)

    def set_episode_policy(self, policy):
        """C4 / ft-v2: the executor policy of the NEXT episode only: None, or {"briefing_hints": callable(otype) -> [container types]}
        (briefing-guided search, +M arms only; HonestSkills is then built with allow_briefing_hints=True)."""
        if policy is not None and (set(policy) - {"briefing_hints"} or not callable(policy.get("briefing_hints"))):
            raise ValueError(f"unknown episode policy {sorted(policy)}")
        self._episode_policy = policy

    # -- lazy simulator bring-up (WSL only) -------------------------------------------------------------------------
    def _boot(self):
        if self._ag is not None:
            return
        # B2: the live run must not write sim/trace.db (the thesis trace store; a B2 smoke wrote to it and hit a malformed image).
        # Set BEFORE thor_actions / webapp import trace; an explicit TRACE_ENABLED=1 in the environment is respected.
        os.environ.setdefault("TRACE_ENABLED", "0")
        import perception                                               # noqa: F401
        import thor_actions as ta
        import webapp
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._cvd = cvd
        self._ag = ta.ThorAgent(scene=self.scene, width=600, height=600, render_depth=True)   # Controller BEFORE ultralytics
        if "fast" in webapp.MOTION_PROFILES:
            webapp._MOTION["speed"] = "fast"
        import honest
        import objmem
        from honest_map import GridMap
        v3 = perception.v3_paths(self.scene)
        self._det = honest.load_detector(self.scene)
        self._gm_cls = GridMap
        self._honest, self._objmem = honest, objmem
        self.detector_rev = "v3-" + hashlib.sha256(json.dumps({k: sha_file(v) if isinstance(v, str) else str(v)
                                                                for k, v in sorted(v3.items())}).encode()).hexdigest()[:8]
        self.map_rev = "m" + sha_file(os.path.join(HERE, "step4", f"{self.scene}.map.json"))
        self._scene_types = self._ag.type_list(visible_only=False)
        self._real_om = objmem.ObjectMemory(self.scene, store=None, episode=None, fov=self._ag.field_of_view or 90.0)
        self._proxy = la.ObjectMemoryProxy(self._real_om)

    def close(self):
        if self._ag is not None:
            try:
                self._ag.stop()
            except Exception:
                pass
            self._ag = None

    # -- scoring side (the ONLY place that reads goal truth) ----------------------------------------------------------
    def _score(self, h, ev, goal) -> bool:
        import step4_refplans as rp
        with h.scoring():
            h.meta("scoring")
            return bool(rp._goal_ok(ev, self._ag, goal))

    def _score_reset_labels(self, ev, gs, goal) -> dict:
        """A14 c-iii: the EVALUATOR's own readings at reset, outcome-free: the truth value of each goal clause on the world right after
        reset + setup, keyed by clause text. Setup side (before the agent exists); the result is carried to the scoring-side capture and
        never to the plan hook. Identical across restores of the same presentation iff the evaluator sees the same world."""
        import step4_refplans as rp
        unmet = set(rp._unmet(ev, gs, self._ag, goal))
        return {"goal_clauses": {gs.format_goal(a): (gs.format_goal(a) not in unmet) for a in goal},
                "goal_true_at_reset": bool(goal) and not unmet}

    def _score_post_sweep_state(self, h, req, it, sk, post, labels, world_hash=None) -> dict:
        """A14: capture the post-sweep world state for the reset gate (12.4). Scoring side: reads the simulator inside `h.scoring()`;
        the result goes to the result record only and is NEVER passed to the plan hook.

        `labels` = the evaluator's readings at reset (`_score_reset_labels`): nothing of the presentation or the arm (item, language, arm
        differ between Phase A and Phase C by design and would breach 12.4 spuriously). The strict world hash is only a DIAGNOSTIC
        (A14: it matched in 14/48 pilot pairs) and is stored under `diagnostics`, never compared."""
        from final_test import reset_gate as rg                      # E3; imported lazily (offline selftests do not need it)
        seen = sorted({str(t.type) for t in sk.om.tracks.values()})
        with h.scoring():
            h.meta("scoring")
            objs, pose = self._ag.objects(), self._ag.agent_pose()
        st0 = rg.capture_state(objs, pose, summary=post["summary"], seen_types=seen, labels=labels, feasibility={}, world_hash=world_hash)
        return rg.capture_state(objs, pose, summary=post["summary"], seen_types=seen, labels=labels, world_hash=world_hash,
                                feasibility=rg.feasibility_of(st0, it.get("plan") or []))

    def _world_hash(self) -> str:
        """Pairing audit: the hash of the world AFTER setup (types + rounded positions + open flags), setup side only."""
        rows = sorted((str(o.get("objectType")), round(float(o["position"]["x"]), 2), round(float(o["position"]["y"]), 2),
                       round(float(o["position"]["z"]), 2), bool(o.get("isOpen"))) for o in self._ag.objects())
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]

    # -- one episode -------------------------------------------------------------------------------------------------------
    def execute(self, req):
        from honest_map import GridMap as _GridMap                          # stdlib only: safe before the Controller
        nav = (self._gm_cls or _GridMap).load(self.scene)                   # K23: the episode's map, loaded ONCE and checked with the request
        bad = scene_problems(self.scene, req, nav)                          # and memory BEFORE the simulator is touched (Astra K23-AGENT-r1 1;
        if bad:                                                             # review 2026-10-18); Kitchen 1 always gives []
            raise LiveInvariant("; ".join(bad))
        import eval as ev
        import general_set as gs
        import levels
        import step4_refplans as rp
        self._boot()
        t_episode = time.time()                                             # K23: the 900 s episode wall cap counts from here
        ep_policy, self._episode_policy = self._episode_policy, None        # C4: consumed by THIS episode, whatever happens below
        it, practice = req.item, req.mode == "practice"
        if self.allow is not None:
            se.assert_items_allowed([it["id"]], self.allow, self.forbid, f"episode {req.cell}")
        mem_on = self.memory == "places" and "places" in req.features and req.memory is not None
        if mem_on:
            assert_b_memory(req.memory)
        ag = self._ag
        res = se.EpisodeResult()
        rec = levels.CommandRecord(id=it["id"], level=it["level"], lang="en", command=it["en"], gold=gs.gold_of(it),
                                   setup=it.get("setup") or None)
        goal = rec.gold["goal"]
        ev._reset_world(ag, self.scene)
        absent = ev.missing_types(ev.gold_types(rec), self._scene_types)
        if absent:
            res.inapplicable = f"object does not exist: {', '.join(absent)}"
            return res
        setup = rec.setup or []
        sres = ev.run_steps(ag, setup)                                    # world preparation: ORACLE dispatch, before the episode
        bad = next((f"{st['action']}: {r['error']}" for st, r in zip(setup, sres) if not r["ok"]), None)
        if bad:
            res.inapplicable = f"setup failed -- {bad}"[:200]
            return res
        if rp._goal_ok(ev, ag, goal):
            res.inapplicable = "goal already true before acting"
            return res
        res.initial_state_hash = self._world_hash()
        reset_labels = self._score_reset_labels(ev, gs, goal) if (self.plan_after_sweep and (self.state_capture or self.state_check is not None)) else None
        holding = rp.setup_holding(setup, sres)
        ltm = req.memory if mem_on else None
        steps_to_run = it["plan"]
        plan_info = None
        if self.plan_hook is not None and not self.plan_after_sweep:      # pre-A11 order (C3 / C4): the planner runs BEFORE the sweep
            ph = self.plan_hook(req, it, holding, ltm)
            plan_info = ph.get("info")
            if ph.get("steps") is None:                                   # rejected / infra: nothing is executed, the item counts as a failure
                res.sr, res.goal_ok, res.steps = 0.0, False, 0.0
                res.extra = {"plan": plan_info, "plan_not_executed": True, "frame": self.frame, "map_rev": self.map_rev,
                             "detector_rev": self.detector_rev}
                return res
            steps_to_run = ph["steps"]
        h = self._honest.HonestAgent(ag, self._det, mode="honest")
        adapter = la.HonestEventAdapter(ltm, episode=req.episode_id, mode=req.mode, world=req.world, frame=self.frame,
                                        map_rev=self.map_rev, kitchen=req.kitchen, detector_rev=self.detector_rev,
                                        fov=self._real_om.fov, clock=lambda: h.step, geometry=self._objmem)
        import honest_skills as hs_mod
        chooser = None
        if mem_on and self.ltm_chooser:
            chooser = la.LtmChooser(ltm, adapter.wm, channels=("places",), fallback=hs_mod.prior_chooser)
            adapter.hint_source = chooser.hints_for
        self._proxy.bind(adapter)
        pol = dict(search="exhaustive", tidy=True) if self.policy == "console" else {}
        if ep_policy is not None:                                         # C4 hook B: only when the runner set it for this episode
            pol.update(briefing_hints=ep_policy["briefing_hints"], allow_briefing_hints=True)
        sk = hs_mod.HonestSkills(h, nav, self._proxy, kitchen=self.scene, chooser=chooser or hs_mod.prior_chooser,
                                 on_event=adapter.on_event, place_hints=(chooser.hints_for if (chooser and self.hints) else None),
                                 hint_boost=self.hint_boost, hint_goals=self.hint_goals, log=self.log, **pol)
        self.last = {"adapter": adapter, "chooser": chooser, "skills": sk}
        started, extra_a11 = None, {}
        if self.plan_after_sweep:                                         # A11: sweep first, THEN the planner (prereg s4 / s16)
            started = start_core(h=h, sk=sk, adapter=adapter, holding=holding, text=it[req.lang], level=it["level"], lang=req.lang,
                                 chooser=chooser)
            post = post_sweep_dict(seed=started, adapter=adapter, holding=holding, sweep_prims=h.step)
            extra_a11 = {"plan_order": "after_sweep", "summary_sha": post["summary_sha"], "summary_n": len(post["summary"]),
                         "sweep_prims": post["sweep_prims"]}
            if self.state_capture or self.state_check is not None:
                state = self._score_post_sweep_state(h, req, it, sk, post, reset_labels, world_hash=res.initial_state_hash)
                extra_a11["post_sweep_state"] = state
                if self.state_check is not None:
                    chk = self.state_check(state, post)
                    if not chk.get("ok"):
                        extra_a11["blocked_state"] = list(chk.get("breaches") or [])
                        return self._unexecuted(res, h, sk, adapter, chooser, practice, mem_on, extra_a11, None, reason="blocked_state")
            if self.plan_hook is not None:
                ph = self.plan_hook(req, it, holding, ltm, post=post)
                plan_info = ph.get("info")
                if ph.get("steps") is None:                               # rejected / infra: no step runs, no scoring call
                    return self._unexecuted(res, h, sk, adapter, chooser, practice, mem_on, extra_a11, plan_info,
                                            reason=not_executed_reason(plan_info))
                steps_to_run = ph["steps"]
        core = run_episode_core(h=h, sk=sk, adapter=adapter, steps=steps_to_run, holding=holding, text=it[req.lang],
                                level=it["level"], lang=req.lang, practice=practice, policy=self.policy, chooser=chooser,
                                score=lambda: self._score(h, ev, goal), started=started,
                                wall_cap_s=self.timeout, t_start=t_episode)
        for k, v in aggregate_result(core).items():
            setattr(res, k, v)
        if practice and mem_on:
            res.wm, res.admission = adapter.wm, core["goal_ok"]
        res.gate_pass = bool(chooser and any(d["learned"] for d in chooser.decisions)) if mem_on else None
        res.applied = bool(chooser and any(d["changed"] for d in chooser.decisions)) if mem_on else None
        res.extra = {"core": {k: core[k] for k in ("goal_ok", "exec_ok", "prims", "retries", "seed_tracks", "n_events", "audit")},
                     "steps": core["steps"], "searches": core["metrics"]["searches"], "chooser": core["chooser"],
                     "frame": self.frame, "map_rev": self.map_rev, "detector_rev": self.detector_rev, "plan": plan_info,
                     "block": (_bilingual_report(ltm, adapter.wm, it["level"]) if (mem_on and not practice) else None)}
        res.extra.update(extra_a11)
        return res

    def _unexecuted(self, res, h, sk, adapter, chooser, practice, mem_on, extra_a11, plan_info, reason="rejected"):
        """A11: close an episode that was swept but not executed. `reason` is EXPLICIT (never guessed from `plan_info`):
        "rejected" / "infra" / "reuse" (the plan hook declined to execute; sr 0 / goal_ok False; in practice the started attempt is
        consolidated with admission False) or "blocked_state" (the A14 d state check failed: INFRASTRUCTURE, not an outcome).
        A blocked execution is MISSING data (sr = goal_ok = None, status "infra_blocked"), is never consolidated and is re-run under the
        crash rule (prereg s13)."""
        if reason not in ("rejected", "infra", "reuse", "blocked_state"):
            raise ValueError(f"unknown reason {reason!r}")
        blocked = reason == "blocked_state"
        closed = close_unexecuted(h=h, sk=sk, adapter=adapter, practice=practice and not blocked, chooser=chooser)
        if blocked:
            res.sr, res.goal_ok, res.steps = None, None, float(closed["prims"])
            extra_a11 = dict(extra_a11, status="infra_blocked")
        else:
            res.sr, res.goal_ok, res.steps = 0.0, False, float(closed["prims"])
            if practice and mem_on:
                res.wm, res.admission = adapter.wm, False
        res.extra = {"plan": plan_info, "plan_not_executed": True, "plan_not_executed_reason": reason, "frame": self.frame,
                     "map_rev": self.map_rev, "detector_rev": self.detector_rev, "core": {"goal_ok": False, "exec_ok": False, "prims": closed["prims"],
                                                                  "audit": closed["audit"], "n_events": closed["n_events"]}, **extra_a11}
        return res


def make_executor(memory="none", policy="console", hints=False, **kw) -> LiveExecutor:
    """Everything is OFF by default (gate: new behaviour behind explicit flags). B3 activates the arm explicitly:
    `step5_eval run --executor step5_live:make_executor --executor-args '{"memory":"places","hints":true}'`; even then the runner
    gives the `none` arm memory=None, which the executor treats as no memory whatever this flag says."""
    return LiveExecutor(memory=memory, policy=policy, hints=hints, **kw)


# ================================================================================================ B2 smoke CLI
def _request(mode, n, iid, item, lang, mem, kitchen):
    return se.EpisodeRequest(cell=f"smoke|{mode}|{n}|{iid}", episode_id=f"smoke-{mode}-{n}-{iid}".replace(":", "-"),
                             run_id="b2-smoke", item=item, lang=lang, level=item["level"], mode=mode, world="reset",
                             world_kind=se.CANONICAL, world_seed=0, policy_seed=0, arm="places" if mem is not None else "none",
                             features=frozenset({"places"}) if mem is not None else frozenset(), memory=mem,
                             plan_source="reference", kitchen=kitchen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B2 live smoke: a few FloorPlan1 items with a disposable ltm.db (WSL only). "
                                             "Everything defaults OFF: memory off, hints off.")
    ap.add_argument("--memory", choices=("off", "places"), default="off")
    ap.add_argument("--hints", choices=("on", "off"), default="off")
    ap.add_argument("--mode", choices=("practice", "test"), default="practice",
                    help="practice: each episode is consolidated into --ltm-db; test: each episode runs on its own frozen copy of --snapshot")
    ap.add_argument("--ltm-db", required=True, help="disposable database path (never sim/memory.db or sim/trace.db)")
    ap.add_argument("--snapshot", default=None, help="practice: write a snapshot here at the end; test: read it")
    ap.add_argument("--items", required=True, help="comma-separated item ids")
    ap.add_argument("--lang", choices=("en", "fa"), default="en")
    ap.add_argument("--policy", choices=("bounded", "console"), default="console")
    ap.add_argument("--allow-final-test-manifest", default=os.path.join(HERE, "..", "docs", "final-test", "split_manifest.json"))
    ap.add_argument("--allow-dev-pilot", action="store_true",
                    help="DEBUGGING ONLY: also allow the 12 dev-pilot items (recorded in every row; never used by B3)")
    ap.add_argument("--route-diff", action="store_true", help="test mode, memory places: run each item with hints on AND off and print the diff")
    ap.add_argument("--hint-boost", type=float, default=None,
                    help="B2 sensitivity analysis: weight multiplier of a plan item at a hinted place (default: honest_skills.HINT_BOOST)")
    ap.add_argument("--hint-goals", choices=("on", "off"), default="off",
                    help="B2: a hinted place no known container covers becomes a look goal (needs --hints on); default off")
    ap.add_argument("--repeat", type=int, default=1,
                    help="test mode: run every item this many times (the live simulator is NOT deterministic: the same item and "
                         "config can succeed or fail run to run, so a paired comparison needs repeats). Default 1 = unchanged")
    ap.add_argument("--out", default=None, help="write the per-episode rows (jsonl) here")
    a = ap.parse_args(argv)
    low = os.path.abspath(a.ltm_db).replace("\\", "/")
    if low.endswith("/sim/memory.db") or low.endswith("/sim/trace.db"):
        print("refusing to write the thesis databases", file=sys.stderr)
        return 3
    allowed, forbidden, allow_sha = se.final_test_allowlist(a.allow_final_test_manifest)
    if a.allow_dev_pilot:
        allowed = allowed | frozenset(json.loads(open(a.allow_final_test_manifest, encoding="utf-8").read())["dev_pilot"])
    ids = [x.strip() for x in a.items.split(",") if x.strip()]
    se.assert_items_allowed(ids, allowed, forbidden, "smoke items")
    if a.mode == "test" and a.memory == "places" and not a.snapshot:
        print("test mode with memory needs --snapshot", file=sys.stderr)
        return 3
    import general_set as gs
    import ltm as L
    items = {it["id"]: it for it in gs.load(os.path.join(HERE, "dataset", "general.json"))["items"]}
    ex = LiveExecutor(memory="none" if a.memory == "off" else "places", policy=a.policy, hints=a.hints == "on", allow=allowed, forbid=forbidden, hint_boost=a.hint_boost, hint_goals=a.hint_goals == "on")
    P = L.load_params()
    out_rows, mem = [], None
    try:
        ex._boot()                                                  # the simulator first: frame / map revision are known after it
        if a.memory == "places" and a.mode == "practice":
            mem = L.LongTermMemory(a.ltm_db, kitchen=SCENE, params=P)
            assert_b_memory(mem)
        reps = max(1, a.repeat) if a.mode == "test" else 1
        plan_ep = [(n, iid, rep) for rep in range(reps) for n, iid in enumerate(ids, 1)]
        for n, iid, rep in plan_ep:
            runs = [(ex.hints, "hints on" if ex.hints else "hints off")]
            if a.route_diff and a.mode == "test" and a.memory == "places":
                runs = [(True, "hints on"), (False, "hints off")]
            collected = []
            for hints_on, label in runs:
                ex.hints = hints_on
                frozen = None
                if a.memory == "places" and a.mode == "test":      # every test: its OWN fresh read-only copy + overlay + hash check
                    frozen = L.LongTermMemory.open_frozen(a.snapshot, f"{a.ltm_db}.testlog.{n}.{rep}.{int(hints_on)}", kitchen=SCENE, params=P)
                    assert_b_memory(frozen)
                    frozen.begin_test_epoch(frame=ex.frame, map_rev=ex.map_rev)
                memory = frozen if frozen is not None else mem
                h0 = memory.state_hash() if memory is not None else None
                req = _request(a.mode, f"{n}r{rep}" if reps > 1 else n, iid, items[iid], a.lang, memory, SCENE)
                if memory is not None and a.mode == "practice":
                    memory.reset_world(req.episode_id, "task_reset", frame=ex.frame, map_rev=ex.map_rev)
                    h0 = memory.state_hash()
                t0 = time.time()
                res = ex.execute(req)
                if memory is not None and memory.state_hash() != h0:
                    raise LiveInvariant(f"the executor wrote to memory during {req.cell}")
                row = {"item": iid, "rep": rep, "mode": a.mode, "variant": label, "inapplicable": res.inapplicable, "goal_ok": res.goal_ok,
                       "elapsed_s": round(time.time() - t0, 1), "allowlist": "nontest+dev_pilot" if a.allow_dev_pilot else "b3",
                       "allowlist_sha": allow_sha, "public": res.public()}
                if mem is not None and a.mode == "practice" and res.wm is not None:
                    row["consolidation"] = mem.end_episode(res.wm, admission=res.admission)
                if frozen is not None and ex.last and ex.last.get("chooser") is not None:
                    row["chain"] = chain_report(frozen, ex.last["adapter"], ex.last["chooser"])
                collected.append(res)
                out_rows.append(row)
                print(json.dumps({k: row[k] for k in ("item", "rep", "mode", "variant", "inapplicable", "goal_ok", "elapsed_s")}, default=str), flush=True)
                if row.get("chain"):
                    print("CHAIN", json.dumps(row["chain"], default=str, ensure_ascii=False), flush=True)
                if frozen is not None:
                    frozen.close()
            if len(collected) == 2 and all(c.extra for c in collected):
                rd = route_diff(collected[0].extra["searches"], collected[1].extra["searches"])
                out_rows.append({"item": iid, "rep": rep, "route_diff": rd})
                print("ROUTE_DIFF", json.dumps(rd, default=str), flush=True)
        if mem is not None and a.mode == "practice" and a.snapshot:
            print("snapshot:", json.dumps(mem.snapshot(a.snapshot, overwrite=True).get("state_hash")))
    finally:
        if mem is not None:
            mem.close()
        ex.close()
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
