"""Run every general instruction's REFERENCE PLAN in the simulator -> dataset/general_validation_<scene>.json.

No LLM is involved. For each item the harness does exactly what `eval._score_record` does around the model
call -- reset the world, apply the applicability gate, run the setup -- then executes the item's own reference
plan and checks the goal. An item lands in one of three bins:

  ok       the reference plan reaches the goal: the instruction is scoreable here, and SR = 1 is attainable
  skipped  the gate would skip it in this scene (object absent / goal already true / setup failed)
  FAIL     the reference plan does NOT reach the goal -- a defect in the item (or the executor), not in any model

FAIL is the bin this script exists for. A record whose gold is unreachable scores SR = 0 for every model forever,
and nothing in a results table distinguishes that from a hard instruction. The `ok` rate is also the EXECUTOR
CEILING of the dataset: the best SR a perfect planner could obtain through this function library.

Run inside WSL (one controller; stop nothing -- it is low-resolution and uses no LLM):

    cd /mnt/d/Personal/Master*/sim
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python validate_general.py \\
        --scene FloorPlan1 [--level L1] [--only g-l1-open-fridge,...] [--failed-from dataset/general_validation_FloorPlan1.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="FloorPlan1")
    p.add_argument("--level", default="", help="L1 | L2 | L3 (default: all)")
    p.add_argument("--only", default="", help="comma-separated item ids")
    p.add_argument("--failed-from", default="", help="re-run only the FAIL items of a previous validation file")
    p.add_argument("--out", default="")
    p.add_argument("--timeout", type=int, default=60, help="seconds before a record is declared HUNG")
    args = p.parse_args(argv)

    import eval as ev
    import general_set as gs
    import levels
    import thor_actions as ta
    import webapp

    items = gs.load()["items"]
    if args.level:
        items = [it for it in items if it["level"] == args.level]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        items = [it for it in items if it["id"] in want]
    if args.failed_from:
        with open(args.failed_from, encoding="utf-8") as fh:
            want = {r["id"] for r in json.load(fh)["results"] if r["status"] == "FAIL"}
        items = [it for it in items if it["id"] in want]

    out_path = args.out or os.path.join(HERE, "dataset", f"general_validation_{args.scene}.json")
    print(f"validating {len(items)} reference plans in {args.scene}", flush=True)

    # WATCHDOG. A step can block forever inside the controller's pipe read (seen in FloorPlan5: python and Unity
    # both idle at 0% CPU for 12 minutes). SIGALRM interrupts the read; the controller is then out of sync with
    # Unity, so the record is marked HUNG and the simulator is restarted rather than trusted again.
    import signal

    class Hung(BaseException):          # BaseException: run_steps' per-step `except Exception` must not eat it
        pass

    def _alarm(*_):
        raise Hung()

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm)

    ag = ta.ThorAgent(scene=args.scene, width=300, height=300)
    if "fast" in webapp.MOTION_PROFILES:            # _exec_one pushes this profile onto the agent each step;
        webapp._MOTION["speed"] = "fast"            # nobody is watching, so take the coarsest one
    scene_types = ag.type_list(visible_only=False)
    results = []
    try:
        for i, it in enumerate(items, 1):
            t0 = time.time()
            rec = levels.CommandRecord(id=it["id"], level=it["level"], lang="en", command=it["en"],
                                       gold=gs.gold_of(it), setup=it.get("setup") or None)
            row = {"id": it["id"], "level": it["level"], "en": it["en"], "status": "ok", "detail": ""}
            try:
                if has_alarm:
                    signal.alarm(args.timeout)
                ev._reset_world(ag, args.scene)
                get_state, get_all = ev.state_readers(ag)
                goal = rec.gold["goal"]
                absent = ev.missing_types(ev.gold_types(rec), scene_types)
                if absent:
                    row.update(status="skipped", detail=f"object does not exist: {', '.join(absent)}")
                else:
                    bad = next((f"{st['action']} {st.get('object') or ''}: {r['error']}"
                                for st, r in zip(rec.setup or [], ev.run_steps(ag, rec.setup or []))
                                if not r["ok"]), None)
                    if bad:
                        row.update(status="FAIL", detail=f"setup failed -- {bad}")
                    elif ev.sr_score(goal, get_state, get_all):
                        row.update(status="skipped", detail="goal already true before acting")
                    else:
                        steps = ev.run_steps(ag, it["plan"])
                        failed = [f"{st['action']} {st.get('object') or ''}: {r['error']}"
                                  for st, r in zip(it["plan"], steps) if not r["ok"]]
                        if not ev.sr_score(goal, get_state, get_all):
                            unmet = [gs.format_goal(a) for a in goal
                                     if not ev.sr_score([a], get_state, get_all)]
                            row.update(status="FAIL", detail=f"goal unmet: {'; '.join(unmet)}"
                                       + (f" | step errors: {' / '.join(failed)}" if failed else ""))
                        elif failed:
                            row["detail"] = f"goal met despite step errors: {' / '.join(failed)}"
            except Hung:
                row.update(status="HUNG", detail=f"no response from the simulator for {args.timeout}s -- restarted")
                try:
                    ag.stop()
                except Exception:
                    pass
                ag = ta.ThorAgent(scene=args.scene, width=300, height=300)
            except Exception as exc:
                row.update(status="FAIL", detail=f"exception: {exc}")
            finally:
                if has_alarm:
                    signal.alarm(0)
            row["elapsed_s"] = round(time.time() - t0, 1)
            results.append(row)
            print(f"[{i:3d}/{len(items)}] {row['status']:7s} {row['id']:44s} {row['detail'][:110]}", flush=True)
    finally:
        try:
            ag.stop()
        except Exception:
            pass

    tally = {}
    for r in results:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    doc = {"scene": args.scene, "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(results),
           "tally": tally, "results": results}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"\n{tally}  ->  {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
