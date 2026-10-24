"""A29 item 2 applicability for the replication kitchens (procedure: docs/final-test/k23/A29.1-applicability-draft.md; Astra
docs/final-test/_astra/K23-A29-APPLIC-r1.txt, K23-A29-RUNNER-r1.txt).

Each registered TEST item (the effective split manifest's `test`) is run ONCE with its registered reference plan through the
frozen final-test executor (step5_live.LiveExecutor, honest mode, console policy, A11 order: sweep first, then the reference
plan because no plan hook is set; registered reset / setup / scoring; the post-sweep reset state is captured as evidence;
C.CAPS caps), condition P0 (no learned memory, prior chooser, no briefing hook), the kitchen's FROZEN perception stack (every
file of its frozen_weights.sha256 re-hashed before the run). No model is called. EN is the canonical validation presentation
(the reference plan is language-independent; `--equivalence` runs FA on a registered 10-item sample and reports agreement).

Classes (A29.1 item 5):
  applicable    goal false after setup AND true after the reference plan, within the caps
  inapplicable  object_absent | goal_true_at_setup (the executor's pre-episode checks) | step_cap (reference > 12 steps) |
                reference_execution_failure (a valid execution that does not reach the goal, incl. a 900 s wall-cap overrun)
  unresolved    infrastructure (crash / hang) after one run + up to 2 reruns in a retry ROUND; or setup_failed_review (a setup
                that failed: never auto-excluded; resolved by a recorded decision `--resolve id=inapplicable:<why>|retry:<why>`)
  INVALID       an honesty-invariant break (LiveInvariant): the run STOPS at once, evidence kept, eligibility stays unresolved
                until a registered correction; never retried automatically
Retry rounds: round 1 = the primary pass; a later round for unresolved items needs `--round N` (N = last round + 1), the
"retried again under the same rule before sampling" of A29; an interrupted invocation resumes its own round only.

Binding: the first row of runs.jsonl records the code / asset identity (executor, scorer, reference plans, setup, dataset, split,
inclusion, perception freeze file + every frozen file); every later invocation must match it exactly or is refused.
Output (quarantined: read ONLY by a29_sample.py): docs/final-test/k23/applicability/<kitchen>/{runs.jsonl, applicability.json,
resolutions.json, equivalence_fa.json}.

    WSL, simulator, under the lock + hang guard (exit 75 = hang: re-launch, the attempt is repeated without counting):
    flock /home/arash/sim.lock ~/thor-env/bin/python sim/final_test/matrix/hang_guard.py \
        sim/final_test/matrix/a29_applicability.py --kitchen FloorPlan15 [--round N] [--only id,id]
    python sim/final_test/matrix/a29_applicability.py --kitchen FloorPlan15 --summarize | --resolve id=inapplicable:why
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

KITCHENS = ("FloorPlan15", "FloorPlan17")
MAX_ATTEMPTS = 3                       # one run + up to 2 infrastructure reruns per round (C.RUN_CLASSES retry rule)
VERSION = "a29-applicability-2"
EQUIV_N = 10
CODE_FILES = ("sim/step5_live.py", "sim/honest_skills.py", "sim/honest.py", "sim/honest_map.py", "sim/objmem.py", "sim/ltm_adapter.py",
              "sim/step4_refplans.py", "sim/eval.py", "sim/general_set.py", "sim/levels.py", "sim/step5_eval.py",
              "sim/final_test/matrix/a29_applicability.py", "sim/final_test/matrix/common.py")


class Invalid(RuntimeError):
    """An honesty-invariant break: the applicability run stops."""


def out_dir(kitchen: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "applicability", kitchen)


def freeze_record(kitchen: str, root: str | None = None) -> str:
    return os.path.join(C.paths(root)["root"], "sim", "kitchen_v3", kitchen, "frozen_weights.sha256")


def verify_freeze(kitchen: str, root: str | None = None) -> dict:
    """Every file listed in the kitchen's freeze record ('<sha256> <relpath>' lines) must exist with that hash."""
    p = freeze_record(kitchen, root)
    if not os.path.exists(p):
        raise RuntimeError(f"{kitchen} perception is not frozen ({p} missing; A29.1 item 3)")
    base, files, bad = C.paths(root)["root"], {}, []
    for line in open(p, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        sha, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        fp = rel if os.path.isabs(rel) else os.path.join(base, rel)
        got = C.sha256_file(fp) if os.path.exists(fp) else None
        files[rel] = sha
        if got != sha:
            bad.append(rel)
    if not files or bad:
        raise RuntimeError(f"{kitchen} frozen perception does not match its record: {bad[:5] or 'empty record'}")
    return {"record": os.path.relpath(p, base).replace(os.sep, "/"), "record_sha256": C.sha256_file(p), "files": files}


def binding(kitchen: str, root: str | None = None) -> dict:
    base = C.paths(root)["root"]
    man = C.load_manifest(root)
    return {"kind": "binding", "version": VERSION, "kitchen": kitchen,
            "code": {f: (C.sha256_file(os.path.join(base, f)) if os.path.exists(os.path.join(base, f)) else None) for f in CODE_FILES},
            "dataset_sha256": C.sha256_file(C.paths(root)["dataset"]), "split_sha256": man.get("sha256"),
            "inclusion_sha256": (man.get("inclusion") or {}).get("sha256"), "caps": C.CAPS, "perception": verify_freeze(kitchen, root),
            "condition": "P0: memory none, prior chooser, no briefing hook; reference plan as the plan; EN canonical"}


def classify(res, err: str | None, n_steps: int, invariant: bool = False) -> tuple:
    """(class, cause, detail) of ONE attempt."""
    if invariant:
        return "INVALID", "honesty_invariant", (err or "")[:300]
    if err is not None:
        return "unresolved", "infrastructure", err[:300]
    if n_steps > C.CAPS["max_steps"]:
        return "inapplicable", "step_cap", f"reference plan has {n_steps} steps > {C.CAPS['max_steps']}"
    inap = getattr(res, "inapplicable", None)
    if inap:
        s = str(inap)
        if s.startswith("object does not exist"):
            return "inapplicable", "object_absent", s[:200]
        if s.startswith("goal already true"):
            return "inapplicable", "goal_true_at_setup", s[:200]
        if s.startswith("setup failed"):
            return "unresolved", "setup_failed_review", s[:200]
        return "unresolved", "inapplicable_other_review", s[:200]
    extra = getattr(res, "extra", None) or {}
    steps = extra.get("steps") or []
    wall = extra.get("_wall_s")
    if wall is not None and wall > C.CAPS["episode_wall_s"]:
        return "inapplicable", "reference_execution_failure", f"episode_wall {wall:.0f} s > {C.CAPS['episode_wall_s']} s"
    if bool(getattr(res, "goal_ok", False)):
        return "applicable", "goal_reached", None
    causes = [f"{s.get('cause')}/{s.get('under')}" if s.get("under") else str(s.get("cause")) for s in steps if not s.get("ok")]
    return "inapplicable", "reference_execution_failure", ",".join(causes)[:200] or "goal not reached"


def final_class(attempts: list, resolution: dict | None = None) -> dict:
    """The item's class: a recorded resolution of a review case wins; else the first decisive attempt; else unresolved."""
    if resolution and resolution.get("decision") == "inapplicable":
        return {"class": "inapplicable", "cause": "resolved:" + resolution.get("why", ""), "attempts": len(attempts)}
    for a in attempts:
        if a["class"] in ("applicable", "inapplicable"):
            return {"class": a["class"], "cause": a["cause"], "detail": a.get("detail"), "attempts": len(attempts)}
    last = attempts[-1] if attempts else {}
    return {"class": "unresolved", "cause": last.get("cause", "missing"), "detail": last.get("detail"), "attempts": len(attempts)}


def load_runs(path: str) -> tuple:
    bind, by = None, {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "binding":
                bind = bind or r
            else:
                by.setdefault(r["item"], []).append(r)
    return bind, by


def load_resolutions(d: str) -> dict:
    p = os.path.join(d, "resolutions.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def request(kitchen: str, item: dict, attempt: int, rnd: int, lang: str = "en"):
    import step5_eval as se
    key = f"a29|{kitchen}|{item['id']}|{lang}|r{rnd}|a{attempt}"
    return se.EpisodeRequest(cell=f"a29-applicability|{kitchen}|{item['id']}|{lang}", episode_id="ep:" + C.sha256_text(key)[:24],
                             run_id=f"a29-{kitchen}", item=item, lang=lang, level=item["level"], mode="test", world="reset",
                             world_kind=se.CANONICAL, world_seed=0, policy_seed=0, arm="none", features=frozenset(), memory=None,
                             plan_source="reference", kitchen=kitchen)


def live_executor(kitchen: str):
    import step5_live as sl
    return sl.LiveExecutor(memory="none", policy="console", hints=False, hint_goals=False, timeout=C.CAPS["episode_wall_s"],
                           plan_after_sweep=True, state_capture=True, scene=kitchen)


def _is_invariant(e: BaseException) -> bool:
    return type(e).__name__ in ("LiveInvariant", "ExecInvalid")


def execute_once(ex, kitchen, item, n, rnd, lang="en"):
    t0, res, err, inv = time.time(), None, None, False
    try:
        res = ex.execute(request(kitchen, item, n, rnd, lang))
    except BaseException as e:                                   # noqa: BLE001 - classified
        if isinstance(e, KeyboardInterrupt):
            raise
        err, inv = f"{type(e).__name__}: {e}", _is_invariant(e)
    wall = round(time.time() - t0, 1)
    if res is not None:
        res.extra = dict(getattr(res, "extra", None) or {}, _wall_s=wall)
    return res, err, inv, wall


def row_of(kitchen, item, n, rnd, lang, res, err, inv, wall):
    cls, cause, detail = classify(res, err, len(item.get("plan") or []), inv)
    extra = (getattr(res, "extra", None) or {}) if res is not None else {}
    core = extra.get("core") or {}
    inap = (getattr(res, "inapplicable", None) or "") if res is not None else ""
    return {"kind": "attempt", "version": VERSION, "kitchen": kitchen, "item": item["id"], "lang": lang, "round": rnd, "attempt": n,
            "class": cls, "cause": cause, "detail": detail, "level": item["level"], "wall_s": wall,
            "goal_false_after_setup": (None if res is None else (False if inap.startswith("goal already true") else
                                                                 (None if inap else True))),
            "goal_true_after_plan": (None if res is None or inap else bool(getattr(res, "goal_ok", False))),
            "steps": [{k: s.get(k) for k in ("i", "action", "object", "to", "ok", "cause", "under", "retries", "prims")}
                      for s in (extra.get("steps") or [])],
            "prims": core.get("prims"), "audit": core.get("audit"), "sweep_prims": extra.get("sweep_prims"),
            "reset_state_sha256": C.sha256_json(extra["post_sweep_state"]) if extra.get("post_sweep_state") is not None else None,
            "initial_state_hash": getattr(res, "initial_state_hash", None) if res is not None else None,
            "world_seed": 0, "world_seed_note": "a label: the simulator has no seed", "when": time.strftime("%Y-%m-%d %H:%M:%S")}


def run(kitchen: str, ids: list, items: dict, runs_path: str, rnd: int, bind_now: dict, resolutions: dict,
        executor_factory=live_executor) -> dict:
    bind, by = load_runs(runs_path)
    if bind is None:
        with open(runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(bind_now, default=str) + chr(10))
    elif bind != json.loads(json.dumps(bind_now, default=str)):
        raise RuntimeError("the code / asset identity differs from this applicability run's binding: refused (new run = new amendment)")
    last_round = max([a.get("round", 1) for att in by.values() for a in att] or [1])
    if rnd not in (last_round, last_round + 1):
        raise RuntimeError(f"--round {rnd}: the last round is {last_round} (resume it, or start round {last_round + 1})")
    ex = None
    for iid in ids:
        att = by.get(iid, [])
        fc = final_class(att, resolutions.get(iid))
        if fc["class"] != "unresolved":
            continue                                             # decided: a valid run is never repeated
        if any(a["class"] == "INVALID" for a in att):
            continue                                             # waits for a registered correction
        if att and fc["cause"] in ("setup_failed_review", "inapplicable_other_review") and \
                (resolutions.get(iid) or {}).get("decision") != "retry":
            continue                                             # waits for a recorded review decision
        in_round = [a for a in att if a.get("round", 1) == rnd]
        while len(in_round) < MAX_ATTEMPTS:
            n = len(att) + 1
            if ex is None:
                ex = executor_factory(kitchen)
            res, err, inv, wall = execute_once(ex, kitchen, items[iid], n, rnd)
            row = row_of(kitchen, items[iid], n, rnd, "en", res, err, inv, wall)
            with open(runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + chr(10))
            att.append(row)
            in_round.append(row)
            print(json.dumps({k: row[k] for k in ("item", "round", "attempt", "class", "cause", "wall_s")}), flush=True)
            if row["class"] == "INVALID":
                raise Invalid(f"honesty invariant broken on {iid} (attempt {n}): run stopped, evidence in {runs_path}")
            if err is not None:
                ex = None                                        # a crashed executor is rebuilt (fresh controller)
            if row["class"] != "unresolved" or row["cause"] != "infrastructure":
                break
        by[iid] = att
    return by


def equivalence_sample(kitchen: str, applicable: list) -> list:
    """The registered FA/EN equivalence sample: EQUIV_N applicable items drawn with C.rng('A29-equiv', kitchen)."""
    r = C.rng("A29-equiv", kitchen)
    pool = sorted(applicable)
    return sorted(r.sample(pool, min(EQUIV_N, len(pool))))


def summarize(kitchen: str, by: dict, test: list, family_of: dict, level_of: dict, bind: dict, resolutions: dict) -> dict:
    fin = {i: final_class(by[i], resolutions.get(i)) for i in test if i in by}
    doc = {"version": VERSION, "kitchen": kitchen, "items": {i: f["class"] for i, f in fin.items()},
           "details": fin, "missing": sorted(set(test) - set(fin)), "binding": bind, "resolutions": resolutions,
           "procedure": "docs/final-test/k23/A29.1-applicability-draft.md"}
    for key, of in (("by_level", level_of), ("by_family", family_of)):
        agg = {}
        for i, f in fin.items():
            g = agg.setdefault(of.get(i, "?"), {"applicable": 0, "inapplicable": 0, "unresolved": 0})
            g[f["class"]] += 1
        doc[key] = agg
    doc["counts"] = {c: sum(1 for f in fin.values() if f["class"] == c) for c in ("applicable", "inapplicable", "unresolved")}
    return doc


def main(argv=None, root: str | None = None, executor_factory=live_executor) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitchen", required=True, choices=KITCHENS)
    ap.add_argument("--only", default="")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--resolve", default="", help="id=inapplicable:<why> | id=retry:<why> (review cases only)")
    ap.add_argument("--equivalence", action="store_true", help="FA run of the registered 10-item sample; reports EN/FA agreement")
    a = ap.parse_args(argv)
    d = out_dir(a.kitchen, root)
    os.makedirs(d, exist_ok=True)
    runs = os.path.join(d, "runs.jsonl")
    man = C.load_manifest(root)
    test = sorted(man["test"])
    gen = {it["id"]: it for it in json.load(open(C.paths(root)["dataset"], encoding="utf-8"))["items"]}
    family_of = {i: f["family"] for f in man["families"] for i in f["items"]}
    level_of = {i: gen[i]["level"] for i in test}
    res_ = load_resolutions(d)
    if a.resolve:
        iid, _, dec = a.resolve.partition("=")
        decision, _, why = dec.partition(":")
        _b, by = load_runs(runs)
        fc = final_class(by.get(iid, []), None)
        if decision not in ("inapplicable", "retry") or not why or fc["cause"] not in ("setup_failed_review", "inapplicable_other_review"):
            print(f"REFUSED: --resolve is for review cases only, with a reason (item {iid}: {fc['cause']})")
            return 2
        res_[iid] = {"decision": decision, "why": why, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(res_, open(os.path.join(d, "resolutions.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps({iid: res_[iid]}))
        return 0
    if a.summarize:
        bind, by = load_runs(runs)
    else:
        try:
            bind = binding(a.kitchen, root)
        except RuntimeError as e:
            print(f"REFUSED: {e}")
            return 2
        ids = [x for x in a.only.split(",") if x] or test
        if set(ids) - set(test):
            print(f"REFUSED: not registered TEST items: {sorted(set(ids) - set(test))[:5]}")
            return 2
        if a.equivalence:
            _b, by0 = load_runs(runs)
            app = [i for i in test if final_class(by0.get(i, []), res_.get(i))["class"] == "applicable"]
            sample, eq = equivalence_sample(a.kitchen, app), []
            ex = executor_factory(a.kitchen)
            for iid in sample:
                res, err, inv, wall = execute_once(ex, a.kitchen, gen[iid], 1, 0, "fa")
                row = row_of(a.kitchen, gen[iid], 1, 0, "fa", res, err, inv, wall)
                eq.append({"item": iid, "en": "applicable", "fa": row["class"], "fa_cause": row["cause"], "agree": row["class"] == "applicable"})
            out = {"kitchen": a.kitchen, "sample": sample, "rows": eq, "agreement": sum(r["agree"] for r in eq), "n": len(eq)}
            json.dump(out, open(os.path.join(d, "equivalence_fa.json"), "w", encoding="utf-8"), indent=1)
            print(json.dumps({k: out[k] for k in ("agreement", "n")}))
            return 0
        try:
            by = run(a.kitchen, ids, gen, runs, a.round, bind, res_, executor_factory)
        except Invalid as e:
            print(f"STOPPED: {e}")
            return 3
        bind, _ = load_runs(runs)
    doc = summarize(a.kitchen, by, test, family_of, level_of, bind, res_)
    with open(os.path.join(d, "applicability.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=str)
    print(json.dumps({"kitchen": a.kitchen, "counts": doc["counts"], "missing": len(doc["missing"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
