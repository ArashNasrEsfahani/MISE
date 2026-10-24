"""W2 / gate 12.5 (b): the empty plan scores 0 on the 12 dev items (WSL, simulator, NO LLM, no memory).

For each dev item (EN text, canonical world) it restores the world, applies the item's setup, runs the fixed start sweep (A11 order), executes an
EMPTY plan (no step at all) and lets the evaluator read the goal (inside the executor's scoring block, as in every other run). The goal must be false
at reset for a dev item, so the empty plan must score 0 (goal_ok False, sr 0). Writes ``docs/final-test/matrix/rehearsal-v1/empty_plan_control.jsonl``
(append-only, resumable). Dev items only. Python 3.10 syntax.

    python sim/final_test/matrix/w2_empty_plan.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402

OUT = os.path.join(C.ns_docs("rehearsal-v1"), "empty_plan_control.jsonl")


class EmptyHook:
    def __call__(self, req, it, holding, ltm, post=None):
        if post is None:
            raise RuntimeError("needs the A11 order")
        if ltm is not None:
            raise RuntimeError("memory must be off")
        return {"steps": [], "info": {"empty_plan_control": True}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    man = C.load_manifest()
    ids = sorted(man["dev_pilot"])
    C.guard_items(ids, "rehearsal-v1", allow_test=False)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["item"] for l in open(OUT, encoding="utf-8") if l.strip()}
    todo = [i for i in ids if i not in done]
    if a.limit:
        todo = todo[: a.limit]
    print("[empty-plan] %d to run (%d done)" % (len(todo), len(done)), flush=True)
    if not todo:
        return 0
    os.environ.setdefault("TRACE_ENABLED", "0")
    import step5_eval as se
    import step5_live as sl
    items = {i["id"]: i for i in json.load(open(C.paths()["dataset"], encoding="utf-8"))["items"]}
    ex = sl.LiveExecutor(memory="none", policy="console", hints=False, plan_after_sweep=True)
    ex.plan_hook = EmptyHook()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, "a", encoding="utf-8")
    try:
        ex._boot()
        for iid in todo:
            it = items[iid]
            key = "w2|empty|%s" % iid
            req = se.EpisodeRequest(cell=key, episode_id="emp:" + C.sha256_text(key)[:24], run_id="ftm-empty", item=it, lang="en",
                                    level=it["level"], mode="test", world="reset", world_kind=se.CANONICAL, world_seed=0, policy_seed=0,
                                    arm="none", features=frozenset(), memory=None, plan_source="capture", kitchen=sl.SCENE)
            t0 = time.time()
            res = ex.execute(req)
            exx = res.extra or {}
            aud = ((exx.get("core") or {}).get("audit")) or {}
            counts = aud.get("counts") or {}
            row = {"item": iid, "level": it["level"], "inapplicable": getattr(res, "inapplicable", None), "goal_ok": getattr(res, "goal_ok", None),
                   "sr": getattr(res, "sr", None), "plan_len": 0, "n_steps_run": len(((exx.get("core") or {}).get("steps")) or []),
                   "violations": int(aud.get("violations") or 0), "oracle_reads": int(counts.get("oracle") or 0),
                   "wall_s": round(time.time() - t0, 2), "ts": time.time()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print("[empty-plan] %s goal_ok=%s sr=%s steps=%s" % (iid, row["goal_ok"], row["sr"], row["n_steps_run"]), flush=True)
            if row["violations"] or row["oracle_reads"]:
                print("ABORT: honesty audit", flush=True)
                return 4
    finally:
        f.close()
        try:
            ex.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
