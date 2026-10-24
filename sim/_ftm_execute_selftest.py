"""Selftest for sim/final_test/matrix/execute.py (stage M6). Offline: no simulator, no WSL, no Ollama, no API; a fake executor stands in for
`step5_live.LiveExecutor`. Synthetic ids only for execution: dev-pilot item ids with invented content; test items appear only in guard checks.

    python sim/_ftm_execute_selftest.py
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                   # noqa: E402
import execute as X                                  # noqa: E402
from final_test import reset_gate as rg              # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


class LiveInvariant(RuntimeError):                  # same class NAME as step5_live.LiveInvariant (classified by name)
    pass


MAN = C.load_manifest()
DEV = ["g-l1-off-stove-knob", "g-l2-find-pickup-lettuce", "g-l3-tool-flip"]   # A26 (K-B2b): g-l3-tool-cut archived
check("dev ids are in the dev-pilot set", all(d in MAN["dev_pilot"] for d in DEV))
TEST_ID = MAN["test"][0]
OPEN = {"action": "open", "object": "Fridge", "to": None, "arg": None}
CLOSE = {"action": "close", "object": "Fridge", "to": None, "arg": None}
TOL = {"tau_pos_m": 0.05, "tau_rot_deg": 5.0}
ITEMS = {d: {"id": d, "level": f"L{i + 1}", "en": f"do {d}", "fa": f"انجام {d}", "setup": [], "plan": [dict(OPEN)]} for i, d in enumerate(DEV)}
AGENT = {"position": [1, 0.9, 1], "rotation": [0, 90, 0], "horizon": 0, "standing": True}


def state_for(pres, shift=0.0, held=False):
    objs = [{"name": "Knife_1", "objectId": "Knife|+0|1|0", "objectType": "Knife", "position": {"x": 0.5, "y": 1, "z": 0.5}, "rotation": {"x": 0, "y": 0, "z": 0},
             "pickupable": True, "isPickedUp": True}] if held else []
    objs += [{"name": "Fridge_1", "objectId": "Fridge|+1|0|1", "objectType": "Fridge", "position": {"x": 2.0 + shift, "y": 0, "z": 1.0},
             "rotation": {"x": 0, "y": 0, "z": 0}, "openable": True, "receptacle": True, "isOpen": False}]
    summ = [f"I see a fridge ({pres})"]
    s0 = rg.capture_state(objs, AGENT, summary=summ, seen_types=["Fridge"], labels={"goal_clauses": {"x": False}}, feasibility={})
    return rg.capture_state(objs, AGENT, summary=summ, seen_types=["Fridge"], labels={"goal_clauses": {"x": False}}, feasibility=rg.feasibility_of(s0, []))


def summary_of(pres):
    return [f"I see a fridge ({pres})"]


# ---------------------------------------------------------------------------------------------------------- fakes
class FakeMem:
    def __init__(self, d):
        self.d, self.closed = d, False

    def state_hash(self):
        return "H" + C.sha256_text(open(os.path.join(self.d, "db"), "rb").read().decode("latin1"))[:8]

    def close(self):
        self.closed = True


class FakeEx:
    """Behaves like LiveExecutor(plan_after_sweep=True, state_capture=True, state_check=...): reset -> sweep -> state check -> hook -> steps."""
    supports_reloc = False
    frame, map_rev = "f0", "m0"

    def __init__(self, script=None, snaproot=None):
        self.script, self.snaproot = script or {}, snaproot
        self.state_check = self.plan_hook = None
        self.calls, self.closed, self.last = [], 0, None
        self.drift = {}

    def close(self):
        self.closed += 1

    def execute(self, req):
        it = req.item
        pres = f"canon:{it['id']}:{req.lang}"
        beh = (self.script.get(pres) or [None])
        b = beh.pop(0) if len(beh) > 0 else None
        self.script[pres] = beh
        self.calls.append((pres, req.cell, req.memory is not None, b))
        summ = summary_of(pres) if b != "summary_drift" else ["something else"]
        post = {"summary": summ, "summary_sha": C.sha256_text("\n".join(summ)), "holding": ("Knife" if b == "told_wrong" else None), "events": [], "wm": None, "episode_id": req.episode_id,
                "sweep_prims": 5, "seed_tracks": 1}
        if b == "crash":
            raise RuntimeError("simulator died")
        if b == "invariant":
            raise LiveInvariant("honesty audit failed")
        if b == "mutate_snapshot" and req.memory is not None:
            with open(os.path.join(req.memory.d, "db"), "ab") as f:
                f.write(b"x")
        if b == "inapplicable":
            return SimpleNamespace(inapplicable="goal already true before acting", extra={}, sr=None, goal_ok=None, steps=None, initial_state_hash=None)
        state = state_for(pres, shift=0.5 if b == "drift" else 0.0, held=(b == "held_wrong"))
        chk = self.state_check(state, post)
        self.last = {"adapter": SimpleNamespace(stream=[{"kind": "sighting", "t": 1}]), "chooser": None, "skills": None}
        if not chk.get("ok"):
            return SimpleNamespace(inapplicable=None, sr=None, goal_ok=None, steps=5.0, initial_state_hash="wh", found=None, retries=None, walking_m=None,
                                   opens=None, stale_visits=None, gate_pass=None, applied=None,
                                   extra={"blocked_state": chk["breaches"], "plan_not_executed": True, "core": {"goal_ok": False, "exec_ok": False, "prims": 5, "audit": {"violations": 0, "counts": {}}, "n_events": 1}})
        ph = self.plan_hook(req, it, None, req.memory, post=post)
        steps = ph["steps"]
        flip = b in ("noisy", "leak_no_audit") or (b == "leak_if_present" and self.snaproot and os.path.exists(self.snaproot))     # A19 (iv): a changed executed step list
        rows = [{"i": i, "action": s["action"], "object": s.get("object"), "to": s.get("to"), "ok": not (flip and i == 0), "cause": None, "retries": 0, "prims": 3}
                for i, s in enumerate(steps)]
        jitter = b == "jitter" or b == "noisy"            # continuous noise: the post-sweep summary hash / event stream / primitive count move, nothing A19 compares does
        if jitter:
            self.jit = getattr(self, "jit", 0) + 1
            post["summary_sha"] = C.sha256_text("jitter%d" % self.jit)
            self.last = {"adapter": SimpleNamespace(stream=[{"kind": "sighting", "t": 1, "x": 0.01 * self.jit}]), "chooser": None, "skills": None}
        goal = any(s["action"] == "open" for s in steps)
        prims = 5 + 3 * len(steps) + (1 if b == "noisy" else 0) + (7 if (b == "leak_if_present" and self.snaproot and os.path.exists(self.snaproot)) else 0)
        oracle = 1 if b == "oracle" else 0
        return SimpleNamespace(inapplicable=None, sr=1.0 if goal else 0.0, goal_ok=goal, steps=float(prims), initial_state_hash="wh", found=None, retries=0, walking_m=1.0,
                               opens=0, stale_visits=0, gate_pass=None, applied=None,
                               extra={"core": {"goal_ok": goal, "exec_ok": True, "prims": prims, "audit": ({} if b in ("no_audit", "leak_no_audit") else {"violations": 0, "counts": {"oracle": oracle, "map-build": 0}}),
                                               "n_events": 1}, "steps": rows, "summary_sha": post["summary_sha"], "sweep_prims": 5})


def mkroot(tmp):
    for sub in ("docs/final-test", "sim/final_test/snapshots/o0/k10", "sim/final_test/snapshots/o1/k3", "sim/step4"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    for f in ("split_manifest.json", "preregistration.md", "families_approved.json"):
        shutil.copy(os.path.join(C.ROOT, "docs", "final-test", f), os.path.join(tmp, "docs", "final-test", f))
    for d in ("o0/k10", "o1/k3"):
        with open(os.path.join(tmp, "sim", "final_test", "snapshots", d, "db"), "wb") as f:
            f.write(b"snapshot " + d.encode())
    with open(os.path.join(tmp, "sim", "step4", "honest_memory.db"), "wb") as f:
        f.write(b"populated")
    return tmp


def seed_journals(root, ns="rehearsal-v1", complete=True, plans=None):
    """Capture rows for DEV x (en, fa) + plan rows; returns (pres list)."""
    cj, pj, ej = (C.Journal(C.journal_path(ns, k, root)) for k in ("capture", "plan", "events"))
    sd = os.path.join(C.ns_docs(ns, root), "states")
    os.makedirs(sd, exist_ok=True)
    pres_all = []
    for d in DEV:
        for lang in ("en", "fa"):
            pres = C.pres_id(d, lang)
            pres_all.append(pres)
            ref = f"states/{pres.replace(':', '_')}.c1.json"
            json.dump(state_for(pres), open(os.path.join(C.ns_docs(ns, root), ref), "w", encoding="utf-8"))
            cj.append(C.make_record("capture", key=C.cell_id(ns, "capture", pres, "c1"), namespace=ns, pres=pres, item=d, lang=lang, level=ITEMS[d]["level"], world="canon",
                                    capture_run=1, instruction=ITEMS[d][lang], holding=None, summary=summary_of(pres), summary_sha=C.sha256_text("\n".join(summary_of(pres))),
                                    blocks={}, requests={}, state_ref=ref, world_hash="wh", audit={"violations": 0}))
    if plans is None:
        plans = []
        for pres in pres_all:
            for arm in ("G0", "O0", "G+M"):
                plans.append((arm, pres, 0, "main", "ok", [OPEN]))
        d0 = DEV[0]
        for lang in ("en", "fa"):                                              # REPEAT-style rows
            for r in (1, 2):
                plans.append(("G0", C.pres_id(d0, lang), r, "repeat", "ok", [OPEN, CLOSE] if r == 2 else [OPEN]))
        plans.append(("O+FS", pres_all[0], 0, "main", "rejected", []))
        plans.append(("O+M", pres_all[0], 0, "main", "ok", [OPEN, CLOSE]))
    for arm, pres, r, blk, st, plan in plans:
        pj.append(C.make_record("plan", key=C.cell_id(ns, "plan", arm, pres, f"r{r}"), namespace=ns, arm=arm, block=blk, pres=pres, r=r, session=0,
                                request_sha="x", status=st, plan=plan, first_plan=plan, first_ok=st == "ok", repaired=False, calls=1, cached_calls=0, tokens_in=1,
                                tokens_out=1, cost_usd=0.0, model="m", scope=C.scope_of(arm, blk)))
    if complete:
        for ph in ("capture", "plan"):
            ej.append(C.make_record("events", event="phase_complete", phase=ph, n=1, journal_sha="s"))
    return pres_all


def mkphase(root, ex_holder, script=None, ns="rehearsal-v1", out=None, leak_req=False, **kw):
    snaproot = os.path.join(root, "sim", "final_test", "snapshots", "o0", "k10")

    def factory(phase):
        e = FakeEx(script if script is not None else {}, snaproot)
        ex_holder.append(e)
        return e
    # the fake renderer stands for PlannerContract; leak_req=True makes it (wrongly) depend on S* being present
    kw.setdefault("request_renderer", lambda cap: C.sha256_text(cap["instruction"] + "|".join(cap["summary"]) + ("S" if leak_req and os.path.exists(snaproot) else "")))
    return X.ExecutePhase(ns, root=root, executor_factory=factory, memory_opener=lambda ph, src, tag, ex: FakeMem(C.snapshot_dir(src, root)), items=ITEMS,
                          tolerances=TOL, repeat_override=[DEV[0]], out=out or (lambda s: None), **kw)


def rows(root, kind, ns="rehearsal-v1"):
    return C.Journal(C.journal_path(ns, kind, root)).rows()


# ---------------------------------------------------------------------------------------------------------- static
src = open(X.__file__, encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok310 = True
except SyntaxError:
    ok310 = False
check("execute.py parses as python 3.10 (WSL thor-env)", ok310)
check("execute.py has LF line endings (WSL)", "\r" not in src)
tops = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
top_mods = {a.name.split(".")[0] for n in tops for a in n.names} | {n.module.split(".")[0] for n in tops if isinstance(n, ast.ImportFrom) and n.module}
check("no simulator or step5 module imported at top level", not (top_mods & {"ai2thor", "thor_actions", "honest", "webapp", "step5_live", "step5_eval", "ltm", "perception", "general_set"}), top_mods)
check("execute.py never opens score.jsonl or reads its own outcome journal for resume", "score.jsonl" not in src and 'J["outcome"].rows' not in src and 'J["outcome"].latest' not in src)
check("execute.py never prints or reads a key", "THESIS_ANTHROPIC_API_KEY" not in src and "get_anthropic_key" not in src)
check("execute.py does not import planners", "import planners" not in src and "from planners" not in src)

# ---------------------------------------------------------------------------------------------------------- unit building
tmp = tempfile.mkdtemp(prefix="ftm-exec-")
try:
    root = mkroot(os.path.join(tmp, "r1"))
    pres_all = seed_journals(root)
    holder = []
    ph = mkphase(root, holder)
    guard = ph.preflight()
    b = ph.build()
    U = b["units"]
    p0 = pres_all[0]
    cells = {c["arm"]: u for u in U.values() if u["pres"] == p0 for c in u["cells"] if c["r"] == 0}
    check("G0 and O0 with equal plans share ONE unit", cells["G0"]["unit"] == cells["O0"]["unit"])
    check("G+M (S* chooser) is a different unit from G0 (same plan)", cells["G+M"]["unit"] != cells["G0"]["unit"] and cells["G+M"]["chooser"] == "o0k10")
    check("G+M/p replays the cached plan with a PRIOR chooser: it joins the G0 unit", cells["G+M/p"]["unit"] == cells["G0"]["unit"])
    check("P0 (reference plan, prior) equals the G0 plan here and joins the same unit; P+M joins G+M", cells["P0"]["unit"] == cells["G0"]["unit"] and cells["P+M"]["unit"] == cells["G+M"]["unit"])
    check("a different plan (O+M [open, close]) is its own unit", cells["O+M"]["unit"] not in (cells["G0"]["unit"], cells["G+M"]["unit"]))
    check("rejected plans are not units (listed as unexecuted)", [x["status"] for x in b["unexecuted"]] == ["rejected"] and cells.get("O+FS") is None)
    check("an LTM unit carries the snapshot sha, a prior unit none", cells["G+M"]["snapshot_sha"] == X.dir_sha(C.snapshot_dir("S*", root)) and cells["G0"]["snapshot_sha"] is None)
    check("unit key == C.execution_unit_key of its parts", cells["G+M"]["unit"] == C.execution_unit_key(plan=cells["G+M"]["plan"], pres=p0, chooser="o0k10",
          snapshot_sha=cells["G+M"]["snapshot_sha"], restored_state=cells["G+M"]["restored_state"], summary_sha=cells["G+M"]["summary_sha"]))
    check("restored_state = world + setup sha", cells["G0"]["restored_state"] == "canon+" + X.setup_sha(ITEMS[DEV[0]]))
    rep = [u for u in U.values() if u["item"] == DEV[0]]
    nrep = [u for u in U.values() if u["item"] != DEV[0]]
    _core_u = lambda u: any(C.repeat_exec_arm(c["arm"]) and c["block"] in ("main", "repeat") for c in u["cells"])
    check("A27 REPEAT presentations: every distinct unit with a core-four / core-/p cell runs 2 times, every other unit (P-only, FS, X) once; others once",
          X.REPEAT_RUNS == 2 and all(u["runs"] == (2 if _core_u(u) else 1) for u in rep) and any(u["runs"] == 2 for u in rep)
          and all(u["runs"] == 1 for u in nrep))
    check("a REPEAT item's r=1/r=2 plans that differ add units (the repeat sampling)", len({u["unit"] for u in rep if u["pres"] == p0}) >= 3)
    check("units.json is written, gold-free and byte-stable", os.path.isfile(ph.units_path) and not C.forbidden_keys(json.load(open(ph.units_path, encoding="utf-8")), {"goal", "gold", "setup", "goal_predicates"}))
    sha_u = C.sha256_file(ph.units_path)
    ph.build()
    check("rebuilding the units is deterministic", C.sha256_file(ph.units_path) == sha_u)
    keys = [c["key"] for u in U.values() for c in u["cells"]]
    check("no plan cell key appears in two units", len(keys) == len(set(keys)))
    check("equal units share across arms: fewer units than plan cells", len(U) < len(keys))
    check("worlds=canon skips relocation presentations", b["skipped_worlds"] == 0)
    # A27 / A25.3: an exploratory Codex plan cell is plan-only: never a unit, counted in plan_only_cells, written to units.json
    _man = dict(C.load_manifest(root))
    if ph.repeat_override is not None:
        _man["repeat_subset"] = list(ph.repeat_override)
    _g0 = next(r for r in ph.plans.values() if r.get("arm") == "G0" and r.get("status") == "ok")
    _xk = C.cell_id(ph.ns, "plan", "X-gpt-6-luna0", _g0["pres"], "r0")
    _plans2 = dict(ph.plans, **{_xk: dict(_g0, arm="X-gpt-6-luna0", key=_xk, block="codex")})
    _b2 = X.build_units(ns=ph.ns, captures=ph.captures, plans=_plans2, items=ph.items(), snap_sha=ph._snap_sha, manifest=_man,
                        p_arms=ph.p_arms, replay=ph.replay, worlds=ph.worlds, reloc_seed=ph.reloc_seed, cuts=frozenset())
    check("A27: an exploratory Codex plan cell is plan-only (not in any unit, counted in plan_only_cells; the rest of the units unchanged)",
          _b2["plan_only_cells"] == 1 and b["plan_only_cells"] == 0 and not any(c["arm"].startswith("X-") for u in _b2["units"].values() for c in u["cells"])
          and set(_b2["units"]) == set(U) and json.load(open(ph.units_path, encoding="utf-8")).get("plan_only_cells") == 0)

    # ---- order -------------------------------------------------------------------------------------------------------
    t1 = ph.todo()
    ph2 = mkphase(root, [])
    ph2.preflight()
    ph2.build()
    check("the execution order is the seeded C.exec_blocks order and is identical across processes", ph2.todo() == t1)
    flat0 = [p for p, _ in t1[0]]
    check("run 0 covers every presentation block once", sorted(flat0) == sorted(pres_all) and len(flat0) == len(set(flat0)))
    check("repeat runs only list REPEAT units", {p for p, _ in t1[1]} == {C.pres_id(DEV[0], "en"), C.pres_id(DEV[0], "fa")})
    check("block order differs from the sorted order (it is a permutation)", flat0 != sorted(flat0) or len(flat0) < 3)

    # ---- a full run ----------------------------------------------------------------------------------------------------
    lines = []
    ph = mkphase(root, holder, out=lines.append)
    info = ph.run()
    ex_rows, oc_rows, ev_rows = rows(root, "exec"), rows(root, "outcome"), rows(root, "events")
    n_cells = sum(u["runs"] for u in U.values())
    check("full run: one exec record per (unit, run)", len(ex_rows) == n_cells and info["executed"] == n_cells and info["complete"], (len(ex_rows), n_cells))
    check("all done, attempt 0, state check ok", all(r["status"] == "done" and r["attempt"] == 0 and r["state_check"]["ok"] for r in ex_rows))
    check("one outcome record per exec record with goal_ok / exec_ok / steps", len(oc_rows) == n_cells and all({"goal_ok", "exec_ok", "steps"} <= set(r) for r in oc_rows))
    check("exec records carry no gold / outcome / steps key", all(not C.forbidden_keys(r, C.GOLD_KEYS | C.OUTCOME_KEYS | {"steps"}) for r in ex_rows))
    check("exec and outcome records share keys", {r["key"] for r in ex_rows} == {r["key"] for r in oc_rows})
    check("phase_complete(execute) appended with the exec journal sha", any(e["event"] == "phase_complete" and e["phase"] == "execute" and e["journal_sha"] == C.sha256_file(C.journal_path("rehearsal-v1", "exec", root)) for e in ev_rows))
    check("snapshot sha recorded before and after, equal, for LTM runs; None for prior runs", all((r["snapshot_sha_before"] == r["snapshot_sha_after"] is not None) if r["chooser"] != "prior" else (r["snapshot_sha_before"] is None) for r in ex_rows))
    check("a fresh restore per run: the executor was asked once per (unit, run) with a distinct cell+attempt", sum(len(h.calls) for h in holder) == n_cells)
    check("prior-chooser runs get memory=None; LTM runs get a frozen memory", all((c[2] == (u_ch != "prior")) for h in holder for c in h.calls for u_ch in [next(u["chooser"] for u in U.values() if f"{u['unit']}|run" in c[1])]))
    check("the replay hook executed the cached plan (open+close unit executed 2 steps)", any(len(o["steps"]) == 2 for o in oc_rows) and any(len(o["steps"]) == 1 for o in oc_rows))
    check("no key in any written file", not any(C.key_leak(open(C.journal_path("rehearsal-v1", k, root), encoding="utf-8").read()) for k in ("exec", "outcome", "events")))
    check("run printed one progress line per cell", len(lines) == n_cells)
    again = mkphase(root, [])
    info2 = again.run()
    check("resume after completion executes nothing and does not duplicate phase_complete", info2["executed"] == 0 and len(rows(root, "exec")) == n_cells
          and sum(1 for e in rows(root, "events") if e["event"] == "phase_complete" and e["phase"] == "execute") == 1)
    check("the phase lock is released after a run", not os.path.exists(os.path.join(C.ns_work("rehearsal-v1", root), "execute.lock")))

    # ---- resume after an interrupt ---------------------------------------------------------------------------------
    root2 = mkroot(os.path.join(tmp, "r2"))
    seed_journals(root2)
    h2 = []
    a = mkphase(root2, h2).run(stop_after=3)
    first = [r["key"] for r in rows(root2, "exec")]
    check("interrupt after N executions: N exec records, event written, no phase_complete", a["interrupted"] and len(first) == 3 and not a["complete"]
          and any(e["event"] == "interrupt" for e in rows(root2, "events")) and not any(e["event"] == "phase_complete" and e["phase"] == "execute" for e in rows(root2, "events")))
    h3 = []
    bb = mkphase(root2, h3).run()
    allk = [r["key"] for r in rows(root2, "exec")]
    check("resume skips the done cells and finishes the rest exactly once", len(allk) == len(set(allk)) == n_cells and set(first) <= set(allk) and bb["executed"] == n_cells - 3 and bb["complete"])
    seq1 = [r["key"] for r in rows(root, "exec")]
    check("the interrupted + resumed run executed in the same seeded order as the uninterrupted run", allk == seq1)

    # ---- torn tail (crash mid-write) ------------------------------------------------------------------------------------
    root3 = mkroot(os.path.join(tmp, "r3"))
    seed_journals(root3)
    mkphase(root3, []).run(stop_after=2)
    pth = C.journal_path("rehearsal-v1", "exec", root3)
    with open(pth, "ab") as f:
        f.write(b'{"schema": "ftm.exec/1", "key": "torn')
    r3 = mkphase(root3, []).run()
    check("a torn last line is ignored, moved aside, and its cell is redone", r3["complete"] and os.path.exists(pth + ".torn") and len(rows(root3, "exec")) == n_cells)

    # ---- crash / blocked / invalid -----------------------------------------------------------------------------------
    def one_cell_root(name, script, **kw):
        rt = mkroot(os.path.join(tmp, name))
        pa = seed_journals(rt, plans=[("G0", C.pres_id(DEV[1], "en"), 0, "main", "ok", [OPEN])])
        hh = []
        phx = mkphase(rt, hh, script=script, p_arms=(), replay=False, **kw)
        return rt, hh, phx

    pe = C.pres_id(DEV[1], "en")
    rt, hh, phx = one_cell_root("crash1", {pe: ["crash"]})
    phx.run()
    er = rows(rt, "exec")
    check("crash once -> re-run from a fresh restore (attempt 1) -> done; the executor was rebuilt", [r["status"] for r in er] == ["crash", "done"] and [r["attempt"] for r in er] == [0, 1]
          and len(hh) == 2 and hh[0].closed == 1)
    check("a crash leaves no outcome record, the retry does", len(rows(rt, "outcome")) == 1)
    rt, hh, phx = one_cell_root("crash3", {pe: ["crash", "crash", "crash"]})
    info = phx.run()
    er = rows(rt, "exec")
    check("crash 3 times -> terminal after attempt 2, reported, phase completes", [r["status"] for r in er] == ["crash"] * 3 and [r["attempt"] for r in er] == [0, 1, 2] and info["complete"]
          and len(rows(rt, "outcome")) == 0)
    again = mkphase(rt, [], p_arms=(), replay=False).run()
    check("a terminal crash is not retried on resume", again["executed"] == 0 and len(rows(rt, "exec")) == 3)
    rt, hh, phx = one_cell_root("blocked", {pe: ["drift", "drift", "drift"]})
    phx.run()
    er = rows(rt, "exec")
    check("state drift beyond tau -> blocked_state, re-run, terminal after attempt 2 (missing data, no outcome)", [r["status"] for r in er] == ["blocked_state"] * 3
          and not er[0]["state_check"]["ok"] and any(b["kind"].startswith("continuous") for b in er[0]["state_check"]["breaches"]) and len(rows(rt, "outcome")) == 0)
    rt, hh, phx = one_cell_root("blocked1", {pe: ["drift"]})
    phx.run()
    check("a transient drift: blocked_state then a clean re-run is done", [r["status"] for r in rows(rt, "exec")] == ["blocked_state", "done"])
    rt, hh, phx = one_cell_root("summ", {pe: ["summary_drift"]})
    phx.run()
    er = rows(rt, "exec")
    check("a post-sweep summary HASH that differs from Phase A does not block when the A14b state check passes (the registered criterion is the state comparison; W2 finding: hashes of the 40-line summary jitter)", er[0]["status"] == "done" and len(er) == 1)
    rt, hh, phx = one_cell_root("nostate", {pe: []})
    os.remove(os.path.join(C.ns_docs("rehearsal-v1", rt), "states", f"{pe.replace(':', '_')}.c1.json"))
    phx.run()
    check("a missing Phase A state file blocks the execution (missing data), never silently passes", rows(rt, "exec")[0]["status"] == "blocked_state"
          and rows(rt, "exec")[0]["state_check"]["breaches"][0]["kind"] == "capture_state_missing" and len(rows(rt, "outcome")) == 0)
    rt, hh, phx = one_cell_root("inapp", {pe: ["inapplicable"]})
    phx.run()
    check("inapplicable at restore -> blocked_state (infra), then re-run", rows(rt, "exec")[0]["status"] == "blocked_state" and rows(rt, "exec")[0]["state_check"]["breaches"][0]["kind"] == "inapplicable_at_restore")

    # ---- A21 C: the replay hook uses the ACTUAL holding; seeded fault injection --------------------------------------------------
    rt, hh, phx = one_cell_root("heldwrong", {pe: ["held_wrong", "held_wrong", "held_wrong"]})
    phx.run()
    er = rows(rt, "exec")
    check("A21 C: a restore in which the agent actually holds an object Phase A did not -> blocked_state, no step ran, no outcome",
          [r["status"] for r in er] == ["blocked_state"] * 3
          and any(b["kind"] == "actual_holding" for b in er[0]["state_check"]["breaches"]) and len(rows(rt, "outcome")) == 0)
    rt, hh, phx = one_cell_root("toldwrong", {pe: ["told_wrong", "told_wrong", "told_wrong"]})
    phx.run()
    er = rows(rt, "exec")
    check("A21 C: a told holding that differs from Phase A (and from what is actually held) blocks the execution",
          [r["status"] for r in er] == ["blocked_state"] * 3 and {b["kind"] for b in er[0]["state_check"]["breaches"]} >= {"told_holding", "told_vs_actual_holding"})

    import random as _rnd

    def kitchen(held="Knife_1", fridge_open=False, apple_in="Fridge_1", extra=None, drop=(), shift=None, sliced=False, summary=None, seen=None,
                agent=None, labels=None, n_apple=1, world_hash="w0"):
        shift = shift or {}
        objs = {
            "Fridge_1": {"objectId": "Fridge|+1|0|0", "objectType": "Fridge", "openable": True, "receptacle": True, "isOpen": fridge_open, "pos": (2.0, 0.0, 1.0)},
            "Cabinet_1": {"objectId": "Cabinet|+0|1|2", "objectType": "Cabinet", "openable": True, "receptacle": True, "isOpen": False, "pos": (0.0, 1.0, 2.0)},
            "Counter_1": {"objectId": "CounterTop|+1|1|1", "objectType": "CounterTop", "receptacle": True, "pos": (1.0, 0.9, 1.0)},
            "Apple_1": {"objectId": "Apple|+2|0|1", "objectType": "Apple", "pickupable": True, "sliceable": True, "isSliced": sliced, "pos": (2.0, 0.5, 1.0)},
            "Mug_1": {"objectId": "Mug|+1|1|1", "objectType": "Mug", "pickupable": True, "pos": (1.0, 1.0, 1.0)},
            "Knife_1": {"objectId": "Knife|+0|1|0", "objectType": "Knife", "pickupable": True, "pos": (0.5, 1.0, 0.5)},
            "Spoon_1": {"objectId": "Spoon|+0|1|3", "objectType": "Spoon", "pickupable": True, "pos": (0.3, 1.0, 0.5)},
        }
        if n_apple == 2:
            objs["Apple_2"] = {"objectId": "Apple|+9|0|9", "objectType": "Apple", "pickupable": True, "sliceable": True, "pos": (9.0, 0.5, 9.0)}
        for nm in drop:
            objs.pop(nm, None)
        for nm, o in (extra or {}).items():
            objs[nm] = o
        parent = {"Apple_1": apple_in, "Mug_1": "Counter_1"}
        lst = []
        for nm, o in sorted(objs.items()):
            x, y, z = o["pos"]
            dx, dy, dz = shift.get(nm, (0.0, 0.0, 0.0))
            d = {"name": nm, "objectId": o["objectId"], "objectType": o["objectType"], "position": {"x": x + dx, "y": y + dy, "z": z + dz},
                 "rotation": {"x": 0, "y": shift.get(nm + ".rot", 0.0), "z": 0}}
            for k in ("openable", "receptacle", "pickupable", "sliceable", "isOpen", "isSliced"):
                if k in o:
                    d[k] = o[k]
            d["isPickedUp"] = nm == held
            par = parent.get(nm)
            if par and par in objs and nm != held:
                d["parentReceptacles"] = [objs[par]["objectId"]]
            lst.append(d)
        by_id = {d["objectId"]: d for d in lst}
        for d in lst:
            for pid in d.get("parentReceptacles") or []:
                by_id[pid].setdefault("receptacleObjectIds", []).append(d["objectId"])
        summ = summary if summary is not None else ["Apple seen at (2.00, 1.00) in Fridge", "Mug seen at (1.00, 1.00) on CounterTop", "Spoon seen at (0.30, 0.50) on CounterTop"]
        ag = agent or {"position": [1.0, 0.9, 1.0], "rotation": [0, 90, 0], "horizon": 0, "standing": True}
        lab = labels if labels is not None else {"goal_clauses": {"Apple in Fridge": False}, "goal_true_at_reset": False}
        st_types = seen or ["Apple", "Fridge", "Mug", "Spoon"]
        s0 = rg.capture_state(lst, ag, summary=summ, seen_types=st_types, labels=lab, feasibility={}, world_hash=world_hash)
        feas = rg.feasibility_of(s0, [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}])
        return rg.capture_state(lst, ag, summary=summ, seen_types=st_types, labels=lab, feasibility=feas, world_hash=world_hash)

    base_state = kitchen()
    check("A21 C: the synthetic kitchen holds exactly the Knife (actual holding read from isPickedUp)", [h["name"] for h in X.actual_holding(base_state)] == ["Knife_1"])
    ok0 = X.replay_precondition(base_state, kitchen(), "Knife", "Knife", TOL)
    check("A21 C: an identical restore passes", ok0["ok"] and ok0["actual_holding"] == ["Knife_1"], str(ok0["breaches"]))
    check("A21 C: told_matches_actual covers None / '?' / a type word / a wrong word",
          X.told_matches_actual(None, []) and not X.told_matches_actual(None, [{"type": "Knife"}]) and X.told_matches_actual("?", [{"type": "Knife"}])
          and not X.told_matches_actual("?", []) and X.told_matches_actual("knife", [{"type": "Knife"}]) and not X.told_matches_actual("Fork", [{"type": "Knife"}]))

    rg_ = _rnd.Random(20261012)

    def unit_vec(norm):
        v = [rg_.random() * 2 - 1 for _ in range(3)]
        n = (sum(q * q for q in v) ** 0.5) or 1.0
        return tuple(c / n * norm for c in v)

    def jitter_case(i):
        sh = {n: unit_vec(rg_.random() * 0.04) for n in ("Apple_1", "Mug_1", "Fridge_1", "Spoon_1")}
        sh["Apple_1.rot"] = (rg_.random() * 2 - 1) * 3.0
        j = (rg_.random() * 2 - 1) * 0.02
        sm = ["Apple seen at (%.2f, %.2f) in Fridge" % (2.00 + j, 1.00 - j), "Mug seen at (1.00, 1.00) on CounterTop", "Spoon seen at (0.30, 0.50) on CounterTop"]
        if i % 3 == 0:                                                                 # a non-task row differs: diagnostic only
            sm.append("Lettuce seen at (%.2f, 3.00) in Cabinet" % rg_.random())
        ag = {"position": [1.0 + (rg_.random() * 2 - 1) * 0.0005, 0.9, 1.0], "rotation": [0, 90 + (rg_.random() * 2 - 1) * 0.005, 0], "horizon": 0, "standing": True}
        return kitchen(shift=sh, summary=sm, agent=ag, world_hash="w%d" % i)

    bad_j = [i for i in range(300) if not X.replay_precondition(base_state, jitter_case(i), "Knife", "Knife", TOL)["ok"]]
    check("A21 C: 300 seeded cases of harmless jitter (positions <= 0.04 m, rotations <= 3 deg, summary coordinates <= 0.03 m, agent pose to precision, "
          "non-task rows, world hash) all pass", not bad_j, str(bad_j[:5]))

    K_ = "Knife"
    faults = {
        "wrong_object_held": lambda: (kitchen(held="Spoon_1"), K_, K_),
        "nothing_held_told_knife": lambda: (kitchen(held=None), K_, K_),
        "nothing_held_told_none_but_capture_knife": lambda: (kitchen(held=None), K_, None),
        "told_differs_actual_same": lambda: (kitchen(), K_, "Spoon"),
        "holding_something_told_nothing": lambda: (kitchen(), K_, None),
        "apple_moved_to_counter": lambda: (kitchen(apple_in="Counter_1"), K_, K_),
        "apple_in_closed_cabinet": lambda: (kitchen(apple_in="Cabinet_1"), K_, K_),
        "fridge_open": lambda: (kitchen(fridge_open=True), K_, K_),
        "object_missing": lambda: (kitchen(drop=("Mug_1",)), K_, K_),
        "extra_object": lambda: (kitchen(extra={"Egg_1": {"objectId": "Egg|+1|1|2", "objectType": "Egg", "pickupable": True, "pos": (1.0, 1.0, 2.0)}}), K_, K_),
        "apple_displaced_0.3m": lambda: (kitchen(shift={"Apple_1": (0.3, 0.0, 0.0)}), K_, K_),
        "apple_sliced": lambda: (kitchen(sliced=True), K_, K_),
        "relevant_summary_row_differs": lambda: (kitchen(summary=["Apple seen at (2.00, 1.00) in Cabinet", "Mug seen at (1.00, 1.00) on CounterTop", "Spoon seen at (0.30, 0.50) on CounterTop"]), K_, K_),
        "relevant_summary_row_missing": lambda: (kitchen(summary=["Mug seen at (1.00, 1.00) on CounterTop", "Spoon seen at (0.30, 0.50) on CounterTop"]), K_, K_),
        "summary_coordinate_shifted_0.3m": lambda: (kitchen(summary=["Apple seen at (2.30, 1.00) in Fridge", "Mug seen at (1.00, 1.00) on CounterTop", "Spoon seen at (0.30, 0.50) on CounterTop"]), K_, K_),
        "type_not_seen": lambda: (kitchen(seen=["Fridge", "Mug", "Spoon"]), K_, K_),
        "agent_moved_0.2m": lambda: (kitchen(agent={"position": [1.2, 0.9, 1.0], "rotation": [0, 90, 0], "horizon": 0, "standing": True}), K_, K_),
        "agent_horizon_differs": lambda: (kitchen(agent={"position": [1.0, 0.9, 1.0], "rotation": [0, 90, 0], "horizon": 30, "standing": True}), K_, K_),
        "agent_crouching": lambda: (kitchen(agent={"position": [1.0, 0.9, 1.0], "rotation": [0, 90, 0], "horizon": 0, "standing": False}), K_, K_),
        "goal_already_true_label": lambda: (kitchen(labels={"goal_clauses": {"Apple in Fridge": True}}), K_, K_),
        "second_apple_changes_feasibility": lambda: (kitchen(n_apple=2), K_, K_),
    }
    leaked = {}
    for name, mk_ in faults.items():
        for rep in range(5):                                                              # 5 repeats per fault class; repeats 1-4 add small position jitter on top
            st, told_a, told_n = mk_()
            if rep:
                st = dict(st)
                st["objects"] = {n: dict(o, pos=[c + (rg_.random() * 2 - 1) * 0.01 for c in o["pos"]]) for n, o in st["objects"].items()}
            if X.replay_precondition(base_state, st, told_a, told_n, TOL)["ok"]:
                leaked[name] = leaked.get(name, 0) + 1
    check("A21 C: all %d seeded fault classes (wrong or missing holding, told/actual mismatch, moved or hidden object, open/closed, sliced, missing/extra "
          "object, relevant summary row, unseen type, agent pose, labels, feasibility) are rejected, each also with added jitter" % len(faults), not leaked, str(leaked))
    only_hold = X.replay_precondition(base_state, kitchen(held="Spoon_1"), "Knife", "Knife", TOL)["breaches"]
    check("A21 C: the actual-holding breach is reported by its own kind, not only through the generic flag compare", any(b["kind"] == "actual_holding" for b in only_hold))
    check("A21 C: the told holding alone is not enough -- the hand is empty but the instruction says Knife -> rejected",
          not X.replay_precondition(kitchen(held=None), kitchen(held=None), "Knife", "Knife", TOL)["ok"])

    # ---- A34-A2 (signed 2026-10-15): the Phase C override, with the same frozen settings on both sides
    SET_ = {"schema": "obs-pipeline/1", "treatment_id": "7a2ee211b85cc534d7677291fce80b04e92cd24a860f1ddd7665185b3a03b67d", "sweep_prims": 12,
            "health": {"detect_calls": 8, "state_calls": 8, "state_errors": 0, "configs": [["auto", "p.pt", "s.pt", True, 0.25]]}}
    PR_ = {"plan": [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}], "goal_clauses": ["Apple in Fridge"]}
    OBS_ONLY = {"relevant_summary_row_differs", "relevant_summary_row_missing", "summary_coordinate_shifted_0.3m", "type_not_seen"}
    import re as _re2

    def _g(st):                                       # the real summary grammar ("T [s] on/in P at (x, z), step N"), shas recomputed
        st = dict(st)
        st["summary"] = [_re2.sub(r"^(\S+) seen at (\(.*?\)) (?:in|on) (\S+)$", r"\1 [x] on/in \3 at \2, step 3", x) for x in st["summary"]]
        st["summary_sha"] = rg.sha256_text("\n".join(st["summary"]))
        st["symbolic_sha"] = rg.sha256_text(rg.canon(rg.symbolic_part(st)))
        return st
    base_g = _g(base_state)
    check("A34-A2 fixture: the converted states are valid A34 captures", rg.state_problems(base_g, presentation=PR_) == [], rg.state_problems(base_g, presentation=PR_))
    leaked2, missed2 = {}, []
    for name, mk_ in faults.items():
        for rep in range(5):
            st, told_a, told_n = mk_()
            if rep:
                st = dict(st)
                st["objects"] = {n: dict(o, pos=[c + (rg_.random() * 2 - 1) * 0.01 for c in o["pos"]]) for n, o in st["objects"].items()}
            r_ = X.replay_precondition(base_g, _g(st), told_a, told_n, TOL, obs_a=SET_, obs_b=SET_, presentation=PR_)
            if name in OBS_ONLY and rep == 0:
                if not (r_["ok"] and r_["override"] and not r_["registered_ok"] and r_["obs_drift"]["amendment"] == "A34-A2"):
                    missed2.append(name)
            elif r_["ok"]:
                leaked2[name] = leaked2.get(name, 0) + 1
    check("A34-A2: with equal settings, every non-observation fault class and every observation fault on a jittered (non-identical) world stays "
          "BLOCKED", not leaked2, str(leaked2))
    check("A34-A2: an observation-only fault (summary row / coordinate / unseen type) on an identical world is overridden with obs_drift", not missed2, str(missed2))
    r_ = X.replay_precondition(base_g, _g(kitchen(seen=["Fridge", "Mug", "Spoon"])), "Knife", "Knife", TOL, obs_a=SET_, obs_b=dict(SET_, sweep_prims=13), presentation=PR_)
    check("A34-A2: different settings on the two sides -> BLOCKED", not r_["ok"] and not r_["override"] and r_["override_why"])
    r_ = X.replay_precondition(base_g, _g(kitchen(seen=["Fridge", "Mug", "Spoon"])), "Knife", "Spoon", TOL, obs_a=SET_, obs_b=SET_, presentation=PR_)
    check("A34-A2: an observation breach + a told-holding breach -> BLOCKED (the holding checks are part of the complete breach set)",
          not r_["ok"] and r_["registered_ok"] is False and not r_["override"])
    r_ = X.replay_precondition(base_state, kitchen(), "Knife", "Knife", TOL, obs_a=SET_, obs_b=SET_, presentation=PR_)
    check("A34-A2: an identical restore is an ordinary pass (registered_ok, no obs_drift)", r_["ok"] and r_["registered_ok"] and r_["obs_drift"] is None)
    r_ = X.replay_precondition(None, kitchen(), "Knife", "Knife", TOL, obs_a=SET_, obs_b=SET_, presentation=PR_)
    check("A34-A2: a missing Phase A state stays BLOCKED", not r_["ok"])

    # ---- A21 C: the same fault injection on the REAL Phase A states of the 24 dev presentations (captured from the live simulator) -------------
    import copy as _copy
    import glob as _glob
    real_dir = os.path.join(C.ns_docs("rehearsal-v1"), "states")
    real_files = sorted(_glob.glob(os.path.join(real_dir, "*.json")))
    cap_hold = {}
    cap_p = C.journal_path("rehearsal-v1", "capture")
    if os.path.exists(cap_p):
        for ln in open(cap_p, encoding="utf-8"):
            if ln.strip():
                rr = json.loads(ln)
                if rr.get("capture_run") == 1 and rr.get("status") == "ok":
                    cap_hold[rr["pres"]] = rr.get("holding")
    if real_files and cap_hold:
        rg2 = _rnd.Random(20261012)
        real = []
        for fpath in real_files:
            dd = json.load(open(fpath, encoding="utf-8"))
            real.append((dd["pres"], dd["state"]))

        def jitter_real(st):
            s2 = _copy.deepcopy(st)
            for o in s2["objects"].values():
                v = [rg2.random() * 2 - 1 for _ in range(3)]
                n = (sum(q * q for q in v) ** 0.5) or 1.0
                m = rg2.random() * 0.04
                o["pos"] = [c + q / n * m for c, q in zip(o["pos"], v)]
                o["rot"] = [c + (rg2.random() * 2 - 1) * 1.0 for c in o["rot"]]
            s2["agent"]["position"] = [c + (rg2.random() * 2 - 1) * 0.0004 for c in s2["agent"]["position"]]
            s2["diagnostics"] = {"world_hash": "other-%d" % rg2.randrange(10 ** 6)}
            s2["summary"] = [re_.sub(r"\((-?\d+\.\d+), (-?\d+\.\d+)\)", lambda mm: "(%.2f, %.2f)" % (float(mm.group(1)) + (rg2.random() * 2 - 1) * 0.01, float(mm.group(2)) + (rg2.random() * 2 - 1) * 0.01), ln_)
                             for ln_ in s2["summary"]]
            return s2

        import re as re_

        def fault_real(st, kind):
            s2 = _copy.deepcopy(st)
            names = sorted(s2["objects"])
            held = [n for n in names if s2["objects"][n]["flags"].get("isPickedUp")]
            if kind == "drop_held":
                if not held:
                    return None
                s2["objects"][held[0]]["flags"]["isPickedUp"] = False
            elif kind == "pick_up_other":
                cands = [n for n in names if "pickupable" in s2["objects"][n]["caps"] and n not in held]
                s2["objects"][rg2.choice(cands)]["flags"]["isPickedUp"] = True
            elif kind == "toggle_open":
                cands = [n for n in names if "openable" in s2["objects"][n]["caps"]]
                n = rg2.choice(cands)
                s2["objects"][n]["flags"]["isOpen"] = not s2["objects"][n]["flags"].get("isOpen")
            elif kind == "displace":
                n = rg2.choice(names)
                s2["objects"][n]["pos"] = [s2["objects"][n]["pos"][0] + 0.25, s2["objects"][n]["pos"][1], s2["objects"][n]["pos"][2]]
            elif kind == "remove":
                s2["objects"].pop(rg2.choice(names))
            elif kind == "rewire_parent":
                cands = [n for n in names if s2["objects"][n]["parents"]]
                n = rg2.choice(cands)
                s2["objects"][n]["parents"] = []
            elif kind == "sliced":
                cands = [n for n in names if "sliceable" in s2["objects"][n]["caps"]]
                n = rg2.choice(cands)
                s2["objects"][n]["flags"]["isSliced"] = not s2["objects"][n]["flags"].get("isSliced")
            elif kind == "agent":
                s2["agent"]["position"] = [s2["agent"]["position"][0] + 0.2] + list(s2["agent"]["position"][1:])
            return s2

        bad_pass, bad_fault, n_fault = [], [], 0
        for pres, st in real:
            told = cap_hold.get(pres)
            if not X.replay_precondition(st, _copy.deepcopy(st), told, told, TOL)["ok"]:
                bad_pass.append(("identical", pres))
            for _ in range(4):
                if not X.replay_precondition(st, jitter_real(st), told, told, TOL)["ok"]:
                    bad_pass.append(("jitter", pres))
            for kind in ("drop_held", "pick_up_other", "toggle_open", "displace", "remove", "rewire_parent", "sliced", "agent"):
                try:
                    f2 = fault_real(st, kind)
                except IndexError:
                    continue
                if f2 is None:
                    continue
                n_fault += 1
                if X.replay_precondition(st, f2, told, told, TOL)["ok"]:
                    bad_fault.append((kind, pres))
            fake_told = "Knife" if told != "Knife" else "Fork"
            n_fault += 1
            if X.replay_precondition(st, _copy.deepcopy(st), told, fake_told, TOL)["ok"]:
                bad_fault.append(("told_differs", pres))
        check("A21 C (real states): %d real Phase A states of the live simulator -- identical restores and 4 seeded jitters each (positions <= 0.04 m, rotations <= 1 deg, summary "
              "coordinates <= 0.014 m, agent pose to precision, new world hash) all pass" % len(real), not bad_pass, str(bad_pass[:4]))
        check("A21 C (real states): %d seeded faults on the real states (dropped or swapped held object, opened/closed, displaced 0.25 m, removed object, cleared parent, "
              "sliced flag, agent moved 0.2 m, wrong told holding) are all rejected" % n_fault, not bad_fault and n_fault >= 150, str(bad_fault[:4]) + f" n={n_fault}")
        check("A21 C (real states): the live captures include states in which the agent really holds an object (isPickedUp), so the actual-holding path is exercised",
              sum(1 for _, st in real if X.actual_holding(st)) >= 3)

    rt, hh, phx = one_cell_root("inv", {pe: ["invariant"]})
    aborted = raises(phx.run, X.ExecAborted)
    check("an honesty invariant -> exec invalid + events abort_invalid + the phase aborts", aborted and rows(rt, "exec")[0]["status"] == "invalid"
          and any(e["event"] == "abort_invalid" for e in rows(rt, "events")) and not os.path.exists(os.path.join(C.ns_work("rehearsal-v1", rt), "execute.lock")))
    check("resume after an invalid run is REFUSED until an amendment resolves it", raises(lambda: mkphase(rt, [], p_arms=(), replay=False).run(), X.ExecAborted))
    res = mkphase(rt, [], p_arms=(), replay=False)
    res._epochs()
    check("--resolve-invalid names the presentation", res.resolve_invalid("A99")["resolved"] == [pe])
    r_after = mkphase(rt, [], p_arms=(), replay=False).run()
    er = rows(rt, "exec")
    check("after the amendment the cell is re-run matched (epoch 1) and completes", r_after["complete"] and er[-1]["status"] == "done" and er[-1]["epoch"] == 1)
    rt, hh, phx = one_cell_root("orc", {pe: ["oracle"]})
    check("a decision-path oracle read in the audit -> invalid -> abort", raises(phx.run, X.ExecAborted) and rows(rt, "exec")[0]["status"] == "invalid")
    check("the invalid run wrote no outcome", len(rows(rt, "outcome")) == 0)

    # ---- snapshot integrity ------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "snap"))
    seed_journals(rt, plans=[("G+M", pe, 0, "main", "ok", [OPEN])])
    hh = []
    phx = mkphase(rt, hh, script={pe: ["mutate_snapshot"]}, p_arms=(), replay=False)
    check("a snapshot that changes during an LTM run -> invalid -> abort", raises(phx.run, X.ExecAborted) and rows(rt, "exec")[0]["status"] == "invalid"
          and "snapshot" in rows(rt, "exec")[0]["error"])
    rt = mkroot(os.path.join(tmp, "snap2"))
    seed_journals(rt, plans=[("G+M", pe, 0, "main", "ok", [OPEN])])
    phx = mkphase(rt, [], p_arms=(), replay=False)
    phx.preflight()
    phx.build()
    with open(os.path.join(rt, "sim", "final_test", "snapshots", "o0", "k10", "db"), "ab") as f:
        f.write(b"edited between build and run")
    u0 = next(iter(phx.b["units"].values()))
    check("a snapshot edited after the units were built is refused before the run", raises(lambda: phx.run_one(u0, 0, 0), X.ExecInvalid))

    # ---- prerequisites + guard ----------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "pre"))
    seed_journals(rt, complete=False)
    check("no phase_complete for capture / plan -> Phase C refuses to start", raises(lambda: mkphase(rt, []).run(), X.ExecAborted))
    rt = mkroot(os.path.join(tmp, "g1"))
    seed_journals(rt, plans=[("G0", C.pres_id(DEV[0], "en"), 0, "main", "ok", [OPEN])])
    cj = C.Journal(C.journal_path("rehearsal-v1", "capture", rt))
    bad = dict(rows(rt, "capture")[0])
    bad.update(pres=C.pres_id(TEST_ID, "en"), item=TEST_ID, key="k-test")
    cj.append(bad)
    check("a test item in the rehearsal namespace is refused (outside its set)", raises(lambda: mkphase(rt, []).run(), C.TestItemRefused))
    rt = mkroot(os.path.join(tmp, "g2"))
    seed_journals(rt, ns="test-v1", plans=[("G0", C.pres_id(TEST_ID, "en"), 0, "main", "ok", [OPEN])], complete=True)
    # capture rows for the test pres are needed for the preflight to get as far as the guard; the guard runs before the pres check
    ITEMS[TEST_ID] = {"id": TEST_ID, "level": "L1", "en": "x", "fa": "y", "setup": [], "plan": []}
    refused = []
    for label, kw in (("no --allow-test", {}), ("--allow-test but no freeze.json", {"allow_test": True})):
        try:
            mkphase(rt, [], ns="test-v1", **kw).preflight()
            refused.append((label, False))
        except C.TestItemRefused as e:
            refused.append((label, True))
        except Exception as e:                                                      # noqa: BLE001
            refused.append((label, f"{type(e).__name__}: {e}"))
    check("test-v1 refuses test items without --allow-test and without a valid freeze (SPEC s7)", all(v is True for _, v in refused), refused)
    check("dev namespaces refuse to be asked for test items even with --allow-test", raises(lambda: C.guard_items([TEST_ID], "rehearsal-v1", allow_test=True, root=rt), C.TestItemRefused))
    check("the CLI refuses --items test on a dev namespace", X.main(["--namespace", "rehearsal-v1", "--items", "test", "--root", rt]) == X.EXIT["refused"])
    ITEMS.pop(TEST_ID, None)

    # ---- no test id ever executed in this selftest -----------------------------------------------------------------------
    allrows = []
    for d, _, fs in os.walk(tmp):
        if os.path.basename(os.path.dirname(d.rstrip("/\\"))) in ("g1", "g2") or any(x in d.replace("\\", "/") for x in ("/g1/", "/g2/")):
            continue                                                                 # the two guard fixtures name a test item on purpose
        for f in fs:
            if f.endswith(".jsonl"):
                allrows.append(open(os.path.join(d, f), encoding="utf-8").read())
    check("no executed record in any other fixture names a test item", allrows and not any(TEST_ID in t for t in allrows))

    # ---- dry run + lock ---------------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "dry"))
    seed_journals(rt)
    dph = mkphase(rt, [])
    di = dph.run(dry_run=True)
    check("--dry-run builds units, counts the todo cells, executes nothing and starts no executor",
          di["units"] > 0 and sum(di["todo"].values()) == sum(u["runs"] for u in dph.b["units"].values()) and dph.ex is None)
    check("dry-run wrote no exec record", rows(rt, "exec") == [])
    lk = C.PhaseLock("rehearsal-v1", "execute", rt)
    lk.__enter__()
    check("a second writer for the same (namespace, phase) is refused", raises(lambda: mkphase(rt, []).run(), C.PhaseLocked))
    lk.__exit__(None, None, None)

    # ---- reloc ------------------------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "reloc"))
    seed_journals(rt)
    ph_r = mkphase(rt, [], worlds=("canon", "reloc"), reloc_seed=7)
    ph_r.preflight()
    check("relocation worlds are skipped by default and need executor support", ph_r.build()["skipped_worlds"] == 0)
    ex_ns = SimpleNamespace(supports_reloc=False, state_check=None, plan_hook=None, close=lambda: None)
    pp = mkphase(rt, [])
    pp.preflight()
    pp.build()
    pp.ex = ex_ns
    pp.captures["reloc:x:en"] = pp.captures[pres_all[0]]
    ur = dict(next(iter(pp.b["units"].values())))
    ur["pres"] = "reloc:" + DEV[0] + ":en"
    ur["world"] = "reloc"
    pp.captures[ur["pres"]] = pp.captures[pres_all[0]]
    check("a relocation unit on an executor without a relocation hook is refused, not run in the canonical world", raises(lambda: pp.run_one(ur, 0, 0), RuntimeError))
    check("restored_state differs between the canonical and the relocated world", X.restored_state_of("canon", ITEMS[DEV[0]], 7) != X.restored_state_of("reloc", ITEMS[DEV[0]], 7)
          and "reloc:7" in X.restored_state_of("reloc", ITEMS[DEV[0]], 7))

    # ---- G-iso ------------------------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "giso"))
    seed_journals(rt)
    gi = mkphase(rt, []).g_iso(n=4)
    check("G-iso: identical traces absent vs present -> pass; report written with a self sha", gi["status"] == "pass" and gi["n"] == 4 and not gi["diff"]
          and gi["sha256"] == C.self_sha(gi) and os.path.isfile(os.path.join(C.ns_docs("rehearsal-v1", rt), "g_iso.json")))
    check("G-iso restored the moved-aside files", os.path.isdir(C.snapshot_dir("S*", rt)) and os.path.isfile(os.path.join(rt, "sim", "step4", "honest_memory.db"))
          and not os.path.exists(C.snapshot_dir("S*", rt) + ".giso-aside"))
    check("G-iso conditions: absent, present, absent (control)", [c["cond"] for c in gi["conditions"]] == ["absent", "present", "absent"] and [c["present"] for c in gi["conditions"]] == [False, True, False])
    check("G-iso wrote no exec / outcome record", rows(rt, "exec") == [] and rows(rt, "outcome") == [])
    rt = mkroot(os.path.join(tmp, "giso2"))
    pa = seed_journals(rt)
    snap = os.path.join(rt, "sim", "final_test", "snapshots", "o0", "k10")
    sc = {p: ["leak_if_present", "leak_if_present", "leak_if_present"] for p in pa}
    # the fake leaks +7 primitives only while the S* directory exists: the 'present' condition differs from 'absent'
    gi2 = mkphase(rt, [], script=sc).g_iso(n=3)
    check("G-iso: a trace that depends on S* being present -> fail (and the control run is clean)", gi2["status"] == "fail" and gi2["diff"] and not gi2["control_noise"], gi2["status"])
    rt = mkroot(os.path.join(tmp, "giso3"))
    pa = seed_journals(rt)
    sc = {p: ["noisy", None, None] for p in pa}                      # the first absent run differs from BOTH later runs: simulator noise
    gi3 = mkphase(rt, [], script=sc).g_iso(n=3)
    check("G-iso (A20.1): present is compared with the FIRST absent run, so a first-run-only difference is a difference (fail), even though the control also differs from it",
          gi3["status"] == "fail" and gi3["diff"] and gi3["control_noise"] and "FIRST absent" in gi3["anchor"], gi3["status"])
    rt = mkroot(os.path.join(tmp, "giso3b"))
    pa = seed_journals(rt)
    gi3b = mkphase(rt, [], script={p: [None, None, "noisy"] for p in pa}).g_iso(n=3)
    check("G-iso (A19): the absent-vs-absent control differing on (i)-(v) makes the result 'inconclusive' even if absent == present for the first pair",
          gi3b["status"] == "inconclusive" and gi3b["control_noise"] and not gi3b["diff"], gi3b["status"])
    rt = mkroot(os.path.join(tmp, "giso5"))
    pa = seed_journals(rt)
    gi5 = mkphase(rt, [], script={p: ["jitter", "jitter", "jitter"] for p in pa}).g_iso(n=4)
    check("G-iso (A19): jitter in the summary hash, the event stream and nothing else does not matter: pass; the strict digest is informational and DOES differ",
          gi5["status"] == "pass" and not gi5["diff"] and not gi5["control_noise"] and gi5["strict_digest_informational"]["diff"]
          and gi5["strict_digest_informational"]["control_noise"], (gi5["status"], gi5["strict_digest_informational"]["diff"]))
    check("G-iso (A19): the report names the five compared components, the excluded poses/timings, and the rule", gi5["compared"] == ["request", "status", "goal_reached", "steps", "audit"]
          and "poses" in gi5["excluded"] and gi5["rule"] == "A20.1" and gi5["schema"] == "ftm.g_iso/2" and set(gi5["components"]) == set(gi5["compared"]))
    rt = mkroot(os.path.join(tmp, "giso6"))
    pa = seed_journals(rt)
    gi6 = mkphase(rt, [], leak_req=True).g_iso(n=3)
    check("G-iso (A19) component (i): a request that depends on S* being present -> fail, the request component carries it",
          gi6["status"] == "fail" and gi6["components"]["request"]["absent_vs_present"] == 3 and gi6["components"]["request"]["absent_vs_control_absent"] == 0, gi6["components"])
    rt = mkroot(os.path.join(tmp, "giso7"))
    pa = seed_journals(rt)
    sc = {p: ["oracle", None, None] for p in pa}                      # an oracle read in the first run only: component (v) differs from both later runs
    gi7 = mkphase(rt, [], script=sc).g_iso(n=2)
    check("G-iso (A19) component (v): audit counts are compared (violations and oracle reads); an oracle read in the first run shows as absent-vs-present", gi7["status"] == "fail" and gi7["components"]["audit"]["absent_vs_present"] >= 1)
    rt = mkroot(os.path.join(tmp, "giso8"))
    pa = seed_journals(rt)
    gi8 = mkphase(rt, []).g_iso(n=50)
    check("G-iso (A19): fewer presentations than the registered n -> inconclusive, never a silent pass", gi8["status"] == "inconclusive" and gi8["short"] and gi8["n"] < 50)
    comp = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [["open", "Fridge|+1|0|1", None, True, None, 0, 3]], "audit_counts": {"oracle": 0, "map-build": 0}, "audit_violations": 0}, "r")
    check("iso_components: step objects reduced to the object type, poses/timings absent",
          comp == {"request": "r", "status": "done", "goal_reached": True, "steps": [["open", "Fridge", True]], "audit": {"violations": 0, "oracle": 0, "map-build": 0}})
    cm = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [], "audit_counts": {"oracle": 0}, "audit_violations": None}, "r")
    cn = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [], "audit_counts": None, "audit_violations": 0}, "r")
    check("A20.1: an unreported audit key is None, never 0 (missing instrumentation is not 'clean')", cm["audit"] == {"violations": None, "oracle": 0, "map-build": None}
          and cn["audit"] == {"violations": 0, "oracle": None, "map-build": None})
    miss = X.ExecutePhase.iso_instrumentation_missing
    check("A20.1: iso_instrumentation_missing names exactly the unreported keys; a blocked run is exempt",
          miss({"status": "done", "audit_counts": {"oracle": 0, "map-build": 0}, "audit_violations": 0}) == []
          and miss({"status": "done", "audit_counts": {"oracle": 0}, "audit_violations": None}) == ["violations", "map-build"]
          and miss({"status": "done", "audit_counts": None, "audit_violations": 0}) == ["oracle", "map-build"]
          and miss({"status": "blocked_state", "audit_counts": {}, "audit_violations": None}) == [])
    comp2 = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [], "audit_counts": {"oracle": 0, "map-build": 0, "pose": 3288, "actuator": 9, "action-resolve": 4}, "audit_violations": 0}, "r")
    comp3 = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [], "audit_counts": {"oracle": 0, "map-build": 0, "pose": 3168, "actuator": 12, "action-resolve": 1, "scoring": 2}, "audit_violations": 0}, "r")
    check("A20 component (v): pose / actuator / action-resolve / scoring counts are excluded from the audit key", comp2 == comp3 and set(comp2["audit"]) == {"violations", "oracle", "map-build"})
    comp4 = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [], "audit_counts": {"oracle": 1, "map-build": 0, "pose": 3168}, "audit_violations": 0}, "r")
    comp5 = X.ExecutePhase.iso_components({"status": "done", "goal_ok": True, "steps": [], "audit_counts": {"oracle": 0, "map-build": 2, "pose": 3168}, "audit_violations": 1}, "r")
    check("A20 component (v): an oracle read, a map-build read or a violation does change the key", comp4 != comp3 and comp5 != comp3 and comp4 != comp5)

    # ---- A20 pass / inconclusive / fail ------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "giso_a20a"))
    pa = seed_journals(rt)
    sel_a = None
    ga = mkphase(rt, [], script={p: [None, None, "noisy"] for p in pa[:2]}).g_iso(n=4)
    check("G-iso (A20): the control differing on <= 2 presentations, only on steps, with zero absent-vs-present differences -> pass, recorded as noise",
          ga["status"] == "pass" and not ga["diff"] and len(ga["control_noise"]) == len(ga["control_noise_steps_only"]) and 0 < len(ga["control_noise"]) <= 2
          and not ga["control_noise_bad"], (ga["status"], ga["control_noise"], ga["pres"]))
    rt = mkroot(os.path.join(tmp, "giso_a20b"))
    pa = seed_journals(rt)
    gb = mkphase(rt, [], script={p: [None, None, "noisy"] for p in pa}).g_iso(n=4)
    check("G-iso (A20): control noise on more than 2 presentations -> inconclusive", gb["status"] == "inconclusive" and len(gb["control_noise"]) > 2, gb["status"])
    rt = mkroot(os.path.join(tmp, "giso_a20c"))
    pa = seed_journals(rt)
    gc = mkphase(rt, [], script={p: [None, None, "oracle"] for p in pa[:1]}).g_iso(n=4)
    check("G-iso (A20): control noise on (v) (one oracle read) even on one presentation -> inconclusive", gc["status"] == "inconclusive" and gc["control_noise_bad"], gc["status"])
    rt = mkroot(os.path.join(tmp, "giso_a20d"))
    pa = seed_journals(rt)
    sd_ = {p: ["leak_if_present"] * 3 for p in pa}
    gd = mkphase(rt, [], script=sd_).g_iso(n=4)
    check("G-iso (A20): an absent-vs-present difference with a clean control -> fail", gd["status"] == "fail" and gd["diff"] and not gd["control_noise"], gd["status"])
    # ---- A20.1 run rules --------------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "giso_a201a"))
    pa = seed_journals(rt)
    hold = []
    g1 = mkphase(rt, hold, script={p: [None, None, None] for p in pa}).g_iso(n=4)
    ex_ = g1["executions"]
    check("G-iso (A20.1): all 3 x n executions really run (executor reached 3n times, unique episode ids), no dedup, no result cache",
          ex_["all_real"] and ex_["expected"] == 12 and ex_["executor_calls"] == 12 and ex_["unique_episode_ids"] == 12 and not ex_["dedup"] and not ex_["result_cache"]
          and sum(len(e.calls) for e in hold) == 12, ex_)
    check("G-iso (A20.1): the report carries the audit-sensitivity evidence and the anchor", g1["audit_sensitivity"]["ok"] and g1["audit_sensitivity"]["injected_key"]["oracle"] >= 1
          and g1["audit_sensitivity"]["injected_key"]["violations"] >= 1 and g1["audit_sensitivity"]["clean_key"] == {"violations": 0, "oracle": 0, "map-build": 0}
          and "FIRST absent" in g1["anchor"] and g1["rule"] == "A20.1" and not g1["inconclusive_reasons"])
    rt = mkroot(os.path.join(tmp, "giso_a201b"))
    pa = seed_journals(rt)
    gp = mkphase(rt, [], script={p: [None, None, None] for p in pa}, audit_probe=lambda: {"ok": False, "why": "silent audit"}).g_iso(n=4)
    check("G-iso (A20.1): an audit that is not shown sensitive -> inconclusive, never a pass", gp["status"] == "inconclusive" and "audit insensitive" in gp["inconclusive_reasons"], gp["status"])
    rt = mkroot(os.path.join(tmp, "giso_a201c"))
    pa = seed_journals(rt)
    gm = mkphase(rt, [], script={p: ["no_audit", "no_audit", "no_audit"] for p in pa}).g_iso(n=4)
    check("G-iso (A20.1): runs that report no audit are unmeasured, not zero: identical empty audits do NOT pass",
          gm["status"] == "inconclusive" and len(gm["instrumentation_missing"]) == 4 and "instrumentation missing" in gm["inconclusive_reasons"], gm["status"])
    rt = mkroot(os.path.join(tmp, "giso_a201d"))
    pa = seed_journals(rt)
    gq = mkphase(rt, [], script={p: [None, "no_audit", None] for p in pa[:1]}).g_iso(n=4)
    check("G-iso (A20.1): a presentation with missing instrumentation is excluded from the difference count and makes the result inconclusive (not a fail from None vs 0)",
          gq["status"] == "inconclusive" and not gq["diff"] and len(gq["instrumentation_missing"]) == 1, (gq["status"], gq["diff"]))
    rt = mkroot(os.path.join(tmp, "giso_a201e"))
    pa = seed_journals(rt)
    ge = mkphase(rt, [], script={p: [None, "leak_if_present", "noisy"] for p in pa}).g_iso(n=4)
    check("G-iso (A20.1): an absent-vs-present difference takes precedence over a control that would be inconclusive (control differs on all 4)",
          ge["status"] == "fail" and ge["diff"] and len(ge["control_noise"]) > 2, (ge["status"], ge["control_noise"]))
    rt = mkroot(os.path.join(tmp, "giso_a201h"))
    pa = seed_journals(rt)
    gh = mkphase(rt, [], script={p: [None, "leak_no_audit", None] for p in pa[:1]}).g_iso(n=4)
    check("G-iso (A20.1, Astra): a presentation with missing audit keys but a different step list absent-vs-present is a FAIL (available components still compared), not hidden as inconclusive",
          gh["status"] == "fail" and len(gh["diff"]) == 1 and gh["diff_components"][gh["diff"][0]] == ["steps"] and len(gh["instrumentation_missing"]) == 1, (gh["status"], gh["diff_components"]))
    xa = {"request": "r", "status": "done", "goal_reached": True, "steps": [], "audit": {"violations": 0, "oracle": 0, "map-build": 0}}
    xb = dict(xa, status="blocked_state", goal_reached=None, steps=[["x", "Y", True]], audit={"violations": None, "oracle": None, "map-build": None})
    xc = dict(xa, request="r2")
    xd = dict(xa, audit={"violations": 0, "oracle": 1, "map-build": None})
    check("iso_diff: a blocked presentation compares only (i); unreported audit keys are skipped, reported ones compared",
          X.ExecutePhase.iso_diff(xa, xb, blocked=True) == [] and X.ExecutePhase.iso_diff(xa, xc, blocked=True) == ["request"]
          and X.ExecutePhase.iso_diff(xa, xb) == ["status", "goal_reached", "steps"] and X.ExecutePhase.iso_diff(xa, xd) == ["audit"]
          and X.ExecutePhase.iso_diff(xa, dict(xa, audit={"violations": None, "oracle": 0, "map-build": 0})) == [])
    rt = mkroot(os.path.join(tmp, "giso_a201i"))
    pa = seed_journals(rt)
    pi = mkphase(rt, [], script={p: [None, None, None] for p in pa})
    orig_pf = pi.preflight

    def pf2():
        r_ = orig_pf()
        for c_ in pi.captures.values():
            c_["requests"] = {"none": "deadbeef"}
        return r_
    pi.preflight = pf2
    gi_ = pi.g_iso(n=4)
    check("G-iso (A20.1): a request probe that differs from the Phase A none-arm bytes (in every condition) -> inconclusive, not pass",
          gi_["status"] == "inconclusive" and "request probe differs from the Phase A none-arm bytes" in gi_["inconclusive_reasons"] and len(gi_["request_probe_vs_phase_a"]) == 3, gi_["inconclusive_reasons"])
    rt = mkroot(os.path.join(tmp, "giso_a201f"))
    pa = seed_journals(rt)
    pf = mkphase(rt, [], script={p: [None, None, None] for p in pa})
    real = pf.run_one
    cache = {}

    def deduped(u, run, attempt, *, tag=None):                       # a (wrong) implementation that reuses the first result for the same presentation
        if u["pres"] not in cache:
            cache[u["pres"]] = real(u, run, attempt, tag=tag)
        return cache[u["pres"]]
    pf.run_one = deduped
    gf = pf.g_iso(n=4)
    check("G-iso (A20.1): a deduplicated / cached execution is detected (executor reached fewer than 3n times) -> inconclusive, never a pass",
          gf["status"] == "inconclusive" and not gf["executions"]["all_real"] and "executions not all real" in gf["inconclusive_reasons"], gf["executions"])
    # the no-memory request is rebuilt through the full construction path on every call (nothing cached between conditions)
    built = []
    fake_plan = SimpleNamespace(none_request_renderer=lambda root=None: (built.append(1), (lambda cap: "sha"))[1])
    saved = sys.modules.get("plan")
    sys.modules["plan"] = fake_plan
    try:
        rt = mkroot(os.path.join(tmp, "giso_a201g"))
        seed_journals(rt)
        pg = mkphase(rt, [], request_renderer=None)
        for _ in range(3):
            pg._none_request_sha({"instruction": "i", "summary": []})
    finally:
        if saved is None:
            sys.modules.pop("plan", None)
        else:
            sys.modules["plan"] = saved
    check("G-iso (A20.1): the default no-memory request path builds the renderer afresh on every call (3 calls -> 3 constructions)", len(built) == 3, built)
    sp = X.audit_sensitivity_probe()
    check("G-iso (A20.1): the audit-sensitivity probe raises the counted violation / oracle / map-build numbers on the real MetaAudit, flags the episode invalid, and a strict audit raises",
          sp["ok"] and sp["injected_key"] == {"violations": 2, "oracle": 1, "map-build": 1} and sp["injected_flagged_invalid"] and sp["strict_audit_raises"], sp)

    rt = mkroot(os.path.join(tmp, "giso4"))
    os.remove(os.path.join(rt, "sim", "step4", "honest_memory.db"))
    seed_journals(rt)
    check("G-iso needs a populated honest_memory.db in the present condition", raises(lambda: mkphase(rt, []).g_iso(n=2), X.ExecInvalid))
    check("G-iso left S* in place after the failed condition", os.path.isdir(C.snapshot_dir("S*", rt)) and not os.path.exists(C.snapshot_dir("S*", rt) + ".giso-aside"))

    # ---- CLI dry-run ------------------------------------------------------------------------------------------------------
    rt = mkroot(os.path.join(tmp, "cli"))
    seed_journals(rt)
    # the CLI loads the real dataset + tolerances; here only argument handling and refusal paths are exercised
    check("CLI: an unknown namespace is rejected by argparse", raises(lambda: X.main(["--namespace", "nope"]), SystemExit))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- A34-A2 2(b): obs_health is behaviour-preserving instrumentation (counts, records the configuration, re-raises; the detector's own
#      handling of a state failure is unchanged)
import obs_health as OH                                  # noqa: E402
import perception as PCP                                 # noqa: E402
_it_ = {"plan": [{"action": "pickup", "object": "Apple", "to": None}], "goal": [{"type": "Apple", "key": "isPickedUp", "value": True}]}
check("A34-A2 presentation_of: the item's reference plan and its own goal-clause texts", X.presentation_of(_it_) == {"plan": _it_["plan"], "goal_clauses": ["holding Apple"]},
      X.presentation_of(_it_))
check("A34-A2 obs_health: no evidence before install", OH.snapshot() is None or OH._INSTALLED["ok"])
check("A34-A2 obs_health: install is idempotent", OH.install() and OH.install() and getattr(PCP.TwoStageDetector.detect, "_obs_health", False))


class _Prim:
    weights_path = "/w/FloorPlan1.yolo26s.pt"

    def detect(self, frame, conf=0.25):
        return [{"type": "Apple", "conf": 0.9, "bbox_xyxy": [0, 0, 1, 1]}]


_bad_states = PCP.StateEstimator.__new__(PCP.StateEstimator)        # no attributes: annotate raises inside
_tsd = PCP.TwoStageDetector(_Prim(), None, mode="auto", states=_bad_states)
OH.reset()
_out = _tsd.detect("frame", conf=0.3)
_sn = OH.snapshot()
check("A34-A2 obs_health: the detections are returned unchanged when the state stage fails (the detector still swallows it)",
      _out == [dict(_Prim().detect(None)[0], source="primary")], _out)
check("A34-A2 obs_health: the swallowed state failure is COUNTED, with the call and the effective configuration",
      _sn["detect_calls"] == 1 and _sn["state_calls"] == 1 and _sn["state_errors"] == 1
      and _sn["configs"] == [["auto", "FloorPlan1.yolo26s.pt", None, True, 0.3]], _sn)
check("A34-A2 obs_health: such a record is refused as override evidence", rg.pipeline_problems(OH.record("7" * 64, 12)) != [])
_tsd2 = PCP.TwoStageDetector(_Prim(), None, mode="auto", states=None)
OH.reset()
_tsd2.detect("frame")
check("A34-A2 obs_health: a healthy sweep record is accepted", rg.pipeline_problems(OH.record("7" * 64, 12)) == [], OH.record("7" * 64, 12))
check("A34-A2 obs_health: reset zeroes the counters", (OH.reset() or OH.snapshot()) == {"detect_calls": 0, "state_calls": 0, "state_errors": 0, "configs": []})

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
