"""Step 4 (s5-refplans) -- run every FloorPlan1 reference plan with the HONEST agent, and the oracle as the baseline.

A sibling of ``validate_general.py``: same reset, same applicability gate in the same order (absent types ->
setup -> goal already true), same goal check (``eval.sr_score``), same continue-after-a-failed-step policy. No LLM.

    --mode oracle   the existing console dispatch (``eval.run_steps`` -> ``webapp._exec_one``), 300x300, forced
                    actions -- re-run first on this tree as the paired baseline (expected 263 ok / 10 skip / 4 FAIL,
                    the four being cook-potato; an expectation, not a target).
    --mode honest   ``honest_skills.HonestSkills.exec_step`` per plan step: RGB + depth through the step-3 detector,
                    screen-point acting, forceAction False, own map + own per-episode memory, 3 retries, and the
                    labelled no-LLM ``prior_chooser`` stand-in for search.

What the comparison may claim (Astra): the two arms differ in perception, targeting, navigation, recovery and
forceAction AT ONCE, so the gap is a reference-plan *execution* gap; attributing it to perception alone needs a
matched-executor control (``--force`` is kept for that). No planning, language or RAG is evaluated here.

Setup contract: an item's ``setup`` is world preparation, run through the ORACLE dispatch on the raw ThorAgent
before the honest episode starts (counted in the row as ``pre.oracle_setup``). The honest agent is told only what a
person would be told: ``holding`` = the fold of the setup's successful pickups/releases. The gate and the goal
check are scoring reads (``pre.scoring``, and ``with h.scoring()`` after the plan) -- never fed back.

Outputs (appended per item, resumable): ``sim/step4/refplans_{oracle,honest}.jsonl``; ``--report`` writes
``sim/step4/refplans_summary.json`` and ``sim/step4/refplans.md``. ``--resume`` skips ids whose COMPLETED row
carries the same config hash; a file with another hash is moved aside (never deleted) and a new one is started.

Run inside WSL from the repo root (one controller; ultralytics is imported only after the Controller exists):
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python sim/step4_refplans.py --mode oracle --resume
    ... sim/step4_refplans.py --mode honest --smoke          # 5 diverse items -- deep-check them by hand first
    ... sim/step4_refplans.py --mode honest --resume         # then everything
    python sim/step4_refplans.py --report                    # pure; Windows is fine
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

OUT = os.path.join(HERE, "step4")
DATASET = os.path.join(HERE, "dataset", "general.json")
SCENE = "FloorPlan1"
LEVELS = ("L1", "L2", "L3")
#: the 7 honest step causes (honest_skills.CAUSES; A26 / K-B2b 2026-10-13 added no_cutting_tool) + outcome categories of an item
STEP_CAUSES = ("not_detected", "wrong_target", "unreachable", "action_refused", "state_unrecognised",
               "retry_exhausted", "no_cutting_tool")
ITEM_CAUSES = STEP_CAUSES + ("goal_unmet", "setup_failed", "episode_budget", "exception", "hung",
                             "oracle_step_error", "honesty_violation")
COMPLETED = ("ok", "skipped", "FAIL")          # a HUNG row is an attempt: kept in the file, re-run on resume
BASELINE = {"ok": 263, "skipped": 10, "FAIL": 4}
SMOKE_IDS = ("g-l1-close-fridge", "g-l1-pickup-apple", "g-l2-fridge-apple", "g-l2-wash-mug", "g-l3-apple-slices")
EP_PRIM_CAP = 3000                             # primitives one honest episode may send (sweep + every step)
ABORT_AFTER = 5                                # consecutive exception/hung items -> abort
SOURCES = {"honest": ("honest.py", "honest_skills.py", "honest_map.py", "objmem.py", "memory.py", "thor_actions.py",
                      "eval.py", "general_set.py", "perception.py", "search.py", "step4_refplans.py",
                      "type_scope.py"),
           "oracle": ("webapp.py", "thor_actions.py", "eval.py", "general_set.py", "step4_refplans.py")}


# ================================================================================================ pure helpers
def jsonl_path(mode: str, out_dir: str = OUT) -> str:
    return os.path.join(out_dir, f"refplans_{mode}.jsonl")


def sha_file(path) -> str:
    if not path or not os.path.exists(path):
        return "absent"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def config_hash(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path) -> list:
    """Rows of a jsonl file; a truncated / corrupt line (a run killed mid-write) is skipped, not fatal."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def done_ids(rows, h: str) -> set:
    return {r["id"] for r in rows if r.get("hash") == h and r.get("status") in COMPLETED}


def latest(rows) -> dict:
    """id -> the last completed row (else the last attempt) -- the row the summary counts."""
    out = {}
    for r in rows:
        prev = out.get(r["id"])
        if prev is None or r.get("status") in COMPLETED or prev.get("status") not in COMPLETED:
            out[r["id"]] = r
    return out


def setup_holding(setup, results) -> str | None:
    """What the agent holds after setup, folded over the SUCCESSFUL setup steps (a person would be told this)."""
    held = None
    for st, r in zip(setup or [], results or []):
        if not r.get("ok"):
            continue
        a = str(st.get("action") or "").lower()
        if a == "pickup":
            held = st.get("object")
        elif a in ("put_in", "pick_and_place", "drop", "put"):
            held = None
    return held


def underlying(cause, why: str) -> str | None:
    """retry_exhausted wraps the cause of the attempt that used the budget up ("<cause>: why"); unwrap it."""
    if cause != "retry_exhausted":
        return cause
    head = str(why or "").split(":", 1)[0].strip()
    return head if head in STEP_CAUSES else "retry_exhausted"


def classify(row: dict) -> str | None:
    """The item's primary failure cause (None for ok / skipped)."""
    if row.get("status") == "HUNG":
        return "hung"
    if row.get("status") != "FAIL":
        return None
    if row.get("cause") in ITEM_CAUSES:
        return row["cause"]
    ff = row.get("first_fail")
    if ff:
        return ff.get("cause") or "action_refused"
    return "goal_unmet"


def invalid(row: dict) -> bool:
    """An honest-arm row that must not be scored (s7 audit): a recorded violation, or any oracle / map-build read
    inside the episode. Such a row is reported as INVALID and left out of every rate and of the pairing."""
    if row.get("mode") != "honest":
        return False
    if row.get("status") == "INVALID":
        return True
    a = row.get("audit") or {}
    c = a.get("counts") or {}
    return bool(a.get("invalid")) or int(a.get("violations") or 0) > 0 or int(c.get("oracle") or 0) > 0         or int(c.get("map-build") or 0) > 0


def pick_smoke(items, ids=SMOKE_IDS) -> list:
    want = list(ids)
    by = {it["id"]: it for it in items}
    return [by[i] for i in want if i in by]


def _bump(d, k, n=1):
    d[k] = d.get(k, 0) + n


def summarize_mode(rows) -> dict:
    lat = latest(rows)
    per = {lv: {"ok": 0, "skipped": 0, "FAIL": 0, "HUNG": 0} for lv in LEVELS}
    tally, causes, under, step_causes, audit = {}, {}, {}, {}, {}
    retries = prims = n_run = violations = 0
    bad = sorted(i for i, r in lat.items() if invalid(r))
    for r in lat.values():
        if invalid(r):                                  # never scored; listed under "invalid"
            violations += int((r.get("audit") or {}).get("violations") or 0)
            continue
        st = r.get("status", "FAIL")
        per.setdefault(r.get("level", "?"), {"ok": 0, "skipped": 0, "FAIL": 0, "HUNG": 0})
        _bump(per[r.get("level", "?")], st)
        _bump(tally, st)
        c = classify(r)
        if c:
            _bump(causes, c)
            ff = r.get("first_fail") or {}
            if ff.get("under"):
                _bump(under, ff["under"])
        for s in r.get("steps") or []:
            if not s.get("ok") and s.get("cause"):
                _bump(step_causes, s["cause"])
        if st != "skipped":
            n_run += 1
            retries += int(r.get("retries") or 0)
            prims += int(r.get("prims") or 0)
        a = r.get("audit") or {}
        for k, v in (a.get("counts") or {}).items():
            _bump(audit, k, int(v))
        for k, v in (r.get("pre") or {}).items():
            _bump(audit, f"pre:{k}", int(v))
        violations += int(a.get("violations") or 0)
    for lv, d in per.items():
        run = d["ok"] + d["FAIL"] + d["HUNG"]
        d["success"] = round(d["ok"] / run, 3) if run else None
    return {"n": len(lat), "tally": tally, "per_level": per, "causes": causes, "underlying": under,
            "step_causes": step_causes, "mean_retries": round(retries / n_run, 2) if n_run else None,
            "mean_prims": round(prims / n_run, 1) if n_run else None, "audit": audit, "violations": violations,
            "invalid": bad,
            "hashes": sorted({r.get("hash") for r in lat.values()})}


def paired(oracle_rows, honest_rows) -> dict:
    """Outcomes on the ids that are eligible (not skipped) in BOTH arms."""
    o, h = latest(oracle_rows), latest(honest_rows)
    ids = sorted(i for i in set(o) & set(h)
                 if o[i].get("status") != "skipped" and h[i].get("status") != "skipped" and not invalid(h[i]))
    out = {"n": len(ids), "both_ok": 0, "oracle_only": 0, "honest_only": 0, "both_fail": 0, "per_level": {}}
    for i in ids:
        ok_o, ok_h = o[i].get("status") == "ok", h[i].get("status") == "ok"
        k = "both_ok" if ok_o and ok_h else "oracle_only" if ok_o else "honest_only" if ok_h else "both_fail"
        out[k] += 1
        lv = out["per_level"].setdefault(o[i].get("level", "?"), {"both_ok": 0, "oracle_only": 0, "honest_only": 0,
                                                                   "both_fail": 0})
        lv[k] += 1
    out["skip_disagree"] = sorted(i for i in set(o) & set(h) if not invalid(h[i])
                                  and (o[i].get("status") == "skipped") != (h[i].get("status") == "skipped"))
    out["invalid_excluded"] = sorted(i for i in set(o) & set(h) if invalid(h[i]))
    return out


def summarize(oracle_rows, honest_rows) -> dict:
    so, sh = summarize_mode(oracle_rows), summarize_mode(honest_rows)
    base = {k: so["tally"].get(k, 0) for k in BASELINE}
    return {"scene": SCENE, "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "oracle": so, "honest": sh,
            "paired": paired(oracle_rows, honest_rows), "baseline_expected": BASELINE,
            "baseline_reproduced": base == BASELINE if so["n"] else None,
            "claim": "reference-plan execution comparison (perception + targeting + navigation + recovery + "
                     "forceAction differ together); no planning / language / RAG evaluated"}


def render_md(s: dict) -> str:
    o, h, p = s["oracle"], s["honest"], s["paired"]

    def lv_row(m, lv):
        d = m["per_level"].get(lv) or {}
        run = d.get("ok", 0) + d.get("FAIL", 0) + d.get("HUNG", 0)
        return f"{d.get('ok', 0)}/{run} ok, {d.get('skipped', 0)} skip" if (run or d.get("skipped")) else "-"

    L = [f"# Step 4 s5 -- reference plans, honest vs oracle ({s['scene']})", "",
         f"Generated {s['generated_ts']}. {s['claim']}.", "",
         f"Oracle: {o['n']} items, tally {o['tally']} (expected {s['baseline_expected']}; reproduced: "
         f"{s['baseline_reproduced']}). Honest: {h['n']} items, tally {h['tally']}.", "",
         "| level | oracle | honest |", "|---|---|---|"]
    L += [f"| {lv} | {lv_row(o, lv)} | {lv_row(h, lv)} |" for lv in LEVELS]
    L += ["", f"Paired on {p['n']} ids eligible in both: both ok {p['both_ok']}, oracle only {p['oracle_only']}, "
              f"honest only {p['honest_only']}, both fail {p['both_fail']}."
              + (f" Gate disagreements: {', '.join(p['skip_disagree'])}." if p["skip_disagree"] else ""), "",
          "Honest failure causes (first failing step; `retry_exhausted` unwrapped below):", ""]
    L += [f"- {k}: {v}" for k, v in sorted(h["causes"].items(), key=lambda kv: -kv[1])] or ["- none"]
    if h["underlying"]:
        L += ["", "Underlying cause of the first failed step: "
              + ", ".join(f"{k} {v}" for k, v in sorted(h["underlying"].items(), key=lambda kv: -kv[1]))]
    L += ["", f"Mean retries per run item: honest {h['mean_retries']}; mean primitives: honest {h['mean_prims']} "
              f"(includes the 8-view start sweep).",
          f"Audit reads per reason (honest, summed): {h['audit']}; violations: {h['violations']}; "
          f"invalid episodes (not scored): {len(h.get('invalid') or [])}.", ""]
    return "\n".join(L)


# ================================================================================================ live run
def _goal_ok(ev, ag, goal) -> bool:
    get_state, get_all = ev.state_readers(ag)
    return bool(ev.sr_score(goal, get_state, get_all))


def _unmet(ev, gs, ag, goal) -> list:
    get_state, get_all = ev.state_readers(ag)
    return [gs.format_goal(a) for a in goal if not ev.sr_score([a], get_state, get_all)]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("oracle", "honest"), default="honest")
    p.add_argument("--resume", action="store_true", help="skip ids already completed under the same config hash")
    p.add_argument("--smoke", action="store_true", help=f"only the 5 diverse items {SMOKE_IDS}")
    p.add_argument("--only", default="", help="comma-separated item ids")
    p.add_argument("--level", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true", help="honest ablation: forceAction True (matched-executor control)")
    p.add_argument("--timeout", type=int, default=0, help="watchdog seconds per item (default oracle 60, honest 900)")
    p.add_argument("--report", action="store_true", help="write refplans_summary.json + refplans.md and exit (pure)")
    p.add_argument("--out-dir", default=OUT)
    p.add_argument("--policy", choices=("bounded", "console"), default="bounded",
                   help="honest search policy: bounded (s5, as published) or console (exhaustive search + tidy + "
                        "pose-first, the console's 2026-10-08 behaviour)")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.report:
        s = summarize(read_jsonl(jsonl_path("oracle", args.out_dir)), read_jsonl(jsonl_path("honest", args.out_dir)))
        with open(os.path.join(args.out_dir, "refplans_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=1)
        with open(os.path.join(args.out_dir, "refplans.md"), "w", encoding="utf-8") as fh:
            fh.write(render_md(s))
        print(render_md(s))
        return 0

    import general_set as gs
    items = gs.load(DATASET)["items"]
    if args.smoke:
        items = pick_smoke(items)
    if args.level:
        items = [it for it in items if it["level"] == args.level]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        items = [it for it in items if it["id"] in want]
    if args.limit:
        items = items[:args.limit]
    honest_mode = args.mode == "honest"
    timeout = args.timeout or (900 if honest_mode else 60)
    W = H = (600 if honest_mode else 300)

    # ---- config hash (no simulator needed)
    import perception
    v3 = perception.v3_paths(SCENE) if honest_mode else {}
    import honest_skills as hs_mod
    parts = {"mode": args.mode, "scene": SCENE, "dataset": sha_file(DATASET), "size": [W, H], "grid": 0.25,
             "src": {f: sha_file(os.path.join(HERE, f)) for f in SOURCES[args.mode]}}
    if honest_mode:
        parts.update(weights={k: sha_file(v) if isinstance(v, str) else str(v) for k, v in sorted(v3.items())},
                     map=sha_file(os.path.join(OUT, f"{SCENE}.map.json")), force=bool(args.force),
                     conf=0.25, render_depth=True, chooser=hs_mod.prior_chooser.label, policy=args.policy,
                     budgets={"retries": hs_mod.MAX_RETRIES, "step_prims": hs_mod.STEP_PRIM_CAP,
                              "episode_prims": EP_PRIM_CAP})
    h_cfg = config_hash(parts)
    jl = jsonl_path(args.mode, args.out_dir)
    old = read_jsonl(jl)
    if old and (not args.resume or any(r.get("hash") != h_cfg for r in old)):
        stale = jl.replace(".jsonl", f".stale-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        os.replace(jl, stale)
        print(f"{'config changed' if args.resume else 'no --resume'}: moved {jl} -> {stale}", flush=True)
        old = []
    skip = done_ids(old, h_cfg) if args.resume else set()
    todo = [it for it in items if it["id"] not in skip]
    print(f"{args.mode}: {len(items)} items, {len(items) - len(todo)} already done (hash {h_cfg}), "
          f"{len(todo)} to run", flush=True)

    import signal

    class _Done(Exception):             # the item is settled before the plan runs (skipped / setup failed)
        pass

    class Hung(BaseException):          # BaseException: no `except Exception` on the way may eat it
        pass

    def _alarm(*_):
        raise Hung()

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm)

    import eval as ev
    import levels
    import thor_actions as ta
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")       # ultralytics may change it; restore before a new Controller

    def boot():
        if cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        if honest_mode:
            return ta.ThorAgent(scene=SCENE, width=W, height=H, render_depth=True)
        return ta.ThorAgent(scene=SCENE, width=W, height=H)

    ag = boot()                                        # Controller BEFORE ultralytics
    import webapp
    if "fast" in webapp.MOTION_PROFILES:               # oracle dispatch (and honest setup) pushes this profile
        webapp._MOTION["speed"] = "fast"
    scene_types = ag.type_list(visible_only=False)     # scoring: the applicability gate (pre.scoring)
    det = store = om = None
    if honest_mode:
        import honest
        import objmem
        from honest_map import GridMap
        from memory import Memory
        det = honest.load_detector(SCENE)
        store = Memory(path=os.path.join(args.out_dir, "honest_memory.db"),
                       audit_path=os.path.join(args.out_dir, "honest_memory_audit.log"))
        om = objmem.ObjectMemory(SCENE, store=store, episode=None, fov=ag.field_of_view or 90.0)

    bad_run = 0
    aborted = ""
    try:
        for n, it in enumerate(todo, 1):
            t0 = time.time()
            rec = levels.CommandRecord(id=it["id"], level=it["level"], lang="en", command=it["en"],
                                       gold=gs.gold_of(it), setup=it.get("setup") or None)
            goal = rec.gold["goal"]
            row = {"id": it["id"], "level": it["level"], "mode": args.mode, "hash": h_cfg, "status": "ok",
                   "detail": "", "cause": None, "goal_ok": None, "exec_ok": None, "steps": [], "first_fail": None,
                   "retries": 0, "prims": 0, "pre": {"scoring": 1, "oracle_setup": 0}, "audit": None}
            h = None
            try:
                if has_alarm:
                    signal.alarm(timeout)
                ev._reset_world(ag, SCENE)
                absent = ev.missing_types(ev.gold_types(rec), scene_types)
                if absent:
                    row.update(status="skipped", detail=f"object does not exist: {', '.join(absent)}")
                    raise _Done
                setup = rec.setup or []
                sres = ev.run_steps(ag, setup)           # world preparation: ORACLE dispatch, before the episode
                row["pre"]["oracle_setup"] = len(setup)
                bad = next((f"{st['action']} {st.get('object') or ''}: {r['error']}"
                            for st, r in zip(setup, sres) if not r["ok"]), None)
                if bad:
                    row.update(status="FAIL", cause="setup_failed", detail=f"setup failed -- {bad}")
                    raise _Done
                row["pre"]["scoring"] += 1
                if _goal_ok(ev, ag, goal):
                    row.update(status="skipped", detail="goal already true before acting")
                    raise _Done
                holding = setup_holding(setup, sres)
                row["told_holding"] = holding
                # ---------------------------------------------------------------- the plan
                if honest_mode:
                    h = honest.HonestAgent(ag, det, mode="honest", force=True if args.force else None)
                    pol = dict(search="exhaustive", tidy=True) if args.policy == "console" else {}
                    sk = hs_mod.HonestSkills(h, GridMap.load(SCENE), om, chooser=hs_mod.prior_chooser, **pol)
                    seed = sk.start_episode(f"s5:{it['id']}", holding=holding)   # oracle setup has ended here
                    row["seed_tracks"] = seed["tracks"]
                    row["sweep_prims"] = h.step
                    budget_hit = False
                    for i, st in enumerate(it["plan"]):
                        if budget_hit or h.step > EP_PRIM_CAP:
                            budget_hit = True
                            srow = {"i": i, "action": st["action"], "object": st.get("object"), "to": st.get("to"),
                                    "ok": False, "cause": "episode_budget", "under": None, "retries": 0,
                                    "prims": 0, "why": f"episode primitive cap {EP_PRIM_CAP} reached; not run"}
                        else:
                            r = (sk.exec_step(st, upcoming=list(it["plan"][i + 1:])) if args.policy == "console"
                                 else sk.exec_step(st))
                            d = r["detail"]
                            srow = {"i": i, "action": st["action"], "object": st.get("object"), "to": st.get("to"),
                                    "ok": r["ok"], "cause": r["cause"], "under": underlying(r["cause"], d["why"]),
                                    "retries": r["retries"], "prims": d["prims"], "why": d["why"],
                                    "verify": d["verify"], "holding": d["holding"],
                                    "search": d.get("search"),
                                    # every event of a failed step (E, 2026-10-09: the cause was often cut off), the
                                    # last 12 of an ok one
                                    "events": d["events"] if (not r["ok"] or os.environ.get("STEP4_ALL_EVENTS")) else d["events"][-12:]}
                        if st.get("object") and srow.get("prims"):       # B3: where the step's object rests (report only)
                            try:
                                with h.scoring():
                                    _gs, _ga = ev.state_readers(ag)
                                    srow["where_after"] = _ga(st["object"], "parentReceptacles")
                            except Exception:
                                pass
                        row["steps"].append(srow)
                        row["retries"] += srow["retries"]
                        if h.audit.invalid:
                            break
                    row["prims"] = h.step
                    with h.scoring():                    # goal truth: report only, after the plan
                        h.meta("scoring")
                        row["goal_ok"] = _goal_ok(ev, ag, goal)
                        unmet = [] if row["goal_ok"] else _unmet(ev, gs, ag, goal)
                        if not row["goal_ok"]:           # B2: where the goal objects really rest (report only)
                            try:
                                _gs, _ga = ev.state_readers(ag)       # the named door; scoring side only
                                row["final_where"] = {t_: _ga(t_, "parentReceptacles")
                                                      for t_ in sorted({a.get("type") for a in goal if a.get("type")})}
                            except Exception:
                                pass
                    row["audit"] = {**h.audit.summary(), "counts": dict(h.audit.counts),
                                    "errors_logged": [(s_, a_, e_[:100]) for s_, a_, e_ in h.audit.errors][-20:]}
                else:
                    sres = ev.run_steps(ag, it["plan"])
                    for i, (st, r) in enumerate(zip(it["plan"], sres)):
                        row["steps"].append({"i": i, "action": st["action"], "object": st.get("object"),
                                             "to": st.get("to"), "ok": r["ok"],
                                             "cause": None if r["ok"] else "oracle_step_error",
                                             "why": (r["error"] or "")[:200]})
                    row["goal_ok"] = _goal_ok(ev, ag, goal)
                    unmet = [] if row["goal_ok"] else _unmet(ev, gs, ag, goal)
                failed = [s for s in row["steps"] if not s["ok"]]
                row["exec_ok"] = not failed
                if failed:
                    f0 = failed[0]
                    row["first_fail"] = {"i": f0["i"], "action": f0["action"], "object": f0.get("object"),
                                         "cause": f0["cause"], "under": f0.get("under")}
                errs = " / ".join(f"{s['action']} {s.get('object') or ''}: {s['cause']} {s.get('why', '')[:60]}"
                                  for s in failed)
                if not row["goal_ok"]:
                    row.update(status="FAIL", cause=(failed[0]["cause"] if failed else "goal_unmet"),
                               detail=f"goal unmet: {'; '.join(unmet)}" + (f" | step errors: {errs}" if errs else ""))
                elif failed:
                    row["detail"] = f"goal met despite step errors: {errs}"
                bad_run = 0
            except _Done:
                bad_run = 0
            except Hung:
                row.update(status="HUNG", cause="hung", detail=f"no response for {timeout}s -- simulator restarted")
                bad_run += 1
                try:
                    ag.stop()
                except Exception:
                    pass
                ag = boot()
            except Exception as exc:
                row.update(status="FAIL", cause="exception", detail=f"exception: {type(exc).__name__}: {exc}"[:300])
                bad_run += 1
            finally:
                if has_alarm:
                    signal.alarm(0)
            if h is not None and row["audit"] is None:
                row["audit"] = {**h.audit.summary(), "counts": dict(h.audit.counts)}
            if invalid(row):                           # never scored, never resumed past (s7 audit)
                row.update(status="INVALID", cause="honesty_violation",
                           detail=f"honesty violation / privileged read in the episode -- {row['detail']}"[:300])
            row["elapsed_s"] = round(time.time() - t0, 1)
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(jl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(f"[{n:3d}/{len(todo)}] {row['status']:7s} {row['id']:40s} retries={row['retries']} "
                  f"prims={row['prims']} {row['detail'][:100]}", flush=True)
            if row["status"] == "INVALID":
                first = h.audit.violations[:1] if h is not None else row["detail"]
                aborted = f"honesty violation in {it['id']}: {first}"
                break
            if bad_run >= ABORT_AFTER:
                aborted = f"{bad_run} consecutive exception/hung items"
                break
    finally:
        try:
            ag.stop()
        except Exception:
            pass
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
    if aborted:
        print(f"ABORT: {aborted}", flush=True)
        return 2
    s = summarize_mode(read_jsonl(jl))
    print(json.dumps({"tally": s["tally"], "per_level": s["per_level"], "causes": s["causes"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
