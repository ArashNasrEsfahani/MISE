"""s4-act live smoke (WSL, FloorPlan1): four honest actions by screen point, with retries.

    open Fridge -> pickup Apple -> put Apple in Fridge -> toggle Faucet on

The agent decides from RGB + depth + the step-3 detector, its own pose, its own map (sim/step4/FloorPlan1.map.json)
and its own memory (its own store sim/step4/s4_smoke_memory.db). Simulator truth is read only AFTER each step,
inside ``h.scoring()``, to report whether the goal holds -- never fed back. Writes sim/step4/s4_smoke.jsonl (one row
per step, appended as it goes) and sim/step4/s4_smoke.json. Fail fast: any honesty violation aborts.

Run inside WSL from the repo root:
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python sim/step4_smoke.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thor_actions import ThorAgent                                   # noqa: E402
import honest                                                         # noqa: E402
import honest_skills as hs                                            # noqa: E402
import objmem                                                         # noqa: E402
from honest_map import GridMap                                        # noqa: E402
from memory import Memory                                             # noqa: E402

OUT = os.path.join(HERE, "step4")
JL, JS, DB = (os.path.join(OUT, n) for n in ("s4_smoke.jsonl", "s4_smoke.json", "s4_smoke_memory.db"))
for p in (JL, DB, DB + "-wal", DB + "-shm"):
    if os.path.exists(p):
        os.remove(p)

STEPS = [({"action": "open", "object": "Fridge"}, ("Fridge", "isOpen", True)),
         ({"action": "pickup", "object": "Apple"}, ("Apple", "isPickedUp", True)),
         ({"action": "put_in", "object": "Fridge"}, ("Apple", "parentReceptacles", "Fridge")),
         ({"action": "toggle_on", "object": "Faucet"}, ("Faucet", "isToggled", True))]


def goal_holds(h, goal):
    """Scoring read (report only, after the step): does any object of the type satisfy the goal?"""
    typ, key, val = goal
    with h.scoring():
        md = h.meta("scoring")
    objs = [o for o in md["objects"] if o["objectType"] == typ]
    if key == "parentReceptacles":
        return any(any(str(p).startswith(val + "|") for p in (o.get(key) or [])) for o in objs)
    return any(o.get(key) == val for o in objs)


sim_agent = ThorAgent("FloorPlan1", render_depth=True)             # Controller BEFORE ultralytics
t0 = time.time()
try:
    det = honest.load_detector("FloorPlan1")
    h = honest.HonestAgent(sim_agent, det, mode="honest")
    gm = GridMap.load("FloorPlan1")
    store = Memory(path=DB)
    om = objmem.ObjectMemory("FloorPlan1", store=store, episode="s4-smoke", fov=sim_agent.field_of_view or 90.0)
    sk = hs.HonestSkills(h, gm, om, chooser=hs.prior_chooser, log=lambda m: print("   ", m, flush=True))
    seed = sk.start_episode("s4-smoke")
    print(f"seeded memory: {seed['tracks']} tracks; types {len(sk.type_list)}", flush=True)
    out = {"scene": "FloorPlan1", "mode": "honest", "force": h.force, "chooser": hs.prior_chooser.label,
           "seed_tracks": seed["tracks"], "seed_summary": seed["summary"], "steps": []}
    for step, goal in STEPS:
        ts = time.time()
        r = sk.exec_step(step)
        row = {"step": step, "ok": r["ok"], "cause": r["cause"], "retries": r["retries"],
               "prims": r["detail"]["prims"], "verify": r["detail"]["verify"], "why": r["detail"]["why"],
               "sub": r["detail"]["sub"], "events": r["detail"]["events"], "search": r["detail"].get("search"),
               "holding": r["detail"]["holding"], "goal": list(goal), "goal_holds": goal_holds(h, goal),
               "secs": round(time.time() - ts, 1), "audit": h.audit.summary()}
        out["steps"].append(row)
        with open(JL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(f"{step['action']} {step['object']}: ok={r['ok']} cause={r['cause']} retries={r['retries']} "
              f"prims={row['prims']} goal_holds={row['goal_holds']} ({row['secs']} s)", flush=True)
        if h.audit.invalid:                                          # fail fast on any honesty violation
            print("ABORT: honesty violation", h.audit.violations, flush=True)
            break
    out["audit"] = h.audit.summary()
    out["audit_counts"] = dict(h.audit.counts)
    out["errors_logged"] = [(s, a, e[:80]) for s, a, e in h.audit.errors]   # failure breakdown only
    out["honest_ok"] = sum(s["ok"] for s in out["steps"])
    out["goal_true"] = sum(bool(s["goal_holds"]) for s in out["steps"])
    out["memory_tracks"] = len(om.tracks)
    out["places_seen_rows"] = len(store.places_seen("FloorPlan1"))
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(JS, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("honest_ok", "goal_true", "audit", "secs")}, default=str), flush=True)
    store.close()
finally:
    sim_agent.stop()                                                 # lifecycle only
