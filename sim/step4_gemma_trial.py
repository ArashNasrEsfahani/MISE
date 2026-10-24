"""s6-gemma live TRIAL (an integration pilot, not the eval): Gemma 4 plans, the honest agent executes, FloorPlan1.

Six instructions from sim/dataset/general.json, 2 per level, mixed EN / FA. Selection rule (fixed before the run):
per level, the first two items in dataset order that the s5 oracle run did not skip in FloorPlan1; the first is
run in English, the second in Persian.

Per item (fresh world reset + fresh per-episode memory, as in s5):
  1. reset; applicability gate (absent types -> oracle setup -> goal already true), scoring reads, as in s5;
  2. the item's ``setup`` runs through the ORACLE dispatch before the episode (world preparation); the honest agent
     is told only ``holding`` (s5 setup contract);
  3. the honest episode starts: a 360-degree sweep seeds object memory;
  4. Gemma gets the tool list + detector types + the memory summary + holding + the command, and returns a plan;
     it is validated (<= 1 repair); a rejected plan is NOT executed;
  5. each step goes through ``HonestSkills.exec_step`` (search chooser = Gemma via LLMChooser);
  6. AFTER the plan, inside ``h.scoring()``: the goal check (report only).
Saved per item (appended as it goes): prompt (messages), raw output(s), parsed plan, validation, plan-vs-reference
normalised Levenshtein / PC / TDS (first and final plan, ``eval.labels``: search_for is scored), execution result per
step, goal success, latency (planner, repair, chooser, execution), VRAM (nvidia-smi) and the audit.
Outputs: sim/step4/s6_trial.jsonl + sim/step4/s6_trial_summary.json.

Run inside WSL from the repo root (Ollama on Windows, bound to 0.0.0.0; endpoint = the default-route gateway):
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python sim/step4_gemma_trial.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "step4")
JL = os.path.join(OUT, "s6_trial.jsonl")
JS = os.path.join(OUT, "s6_trial_summary.json")
DB = os.path.join(OUT, "s6_trial_memory.db")
SCENE = "FloorPlan1"
LANGS = ("en", "fa")
EP_PRIM_CAP = 3000


class _Settled(Exception):
    """The item is settled before the plan runs (skipped / setup failed)."""


def select_items(items, oracle_rows) -> list:
    """The fixed rule: per level, the first two items (dataset order) that the s5 oracle run did not skip."""
    skipped = {r["id"] for r in oracle_rows if r.get("status") == "skipped"}
    seen = {r["id"] for r in oracle_rows}
    out = []
    for lv in ("L1", "L2", "L3"):
        pick = [it for it in items if it["level"] == lv and it["id"] in seen and it["id"] not in skipped][:2]
        out += [(it, LANGS[k]) for k, it in enumerate(pick)]
    return out


def vram() -> dict:
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi")
                                         else None)
    if not exe:
        return {"error": "nvidia-smi not found"}
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
        return {"used_mib": used, "total_mib": total}
    except Exception as e:
        return {"error": str(e)[:120]}


def _get(endpoint, path):
    with urllib.request.urlopen(endpoint + path, timeout=10) as r:
        return json.loads(r.read().decode())


def plan_scores(ev, gen, ref) -> dict:
    a, b = ev.labels(gen), ev.labels(ref)
    lev = ev.levenshtein(a, b)
    m = max(len(a), len(b))
    return {"gen_labels": a, "ref_labels": b, "lev": lev, "nlev": round(lev / m, 3) if m else 0.0,
            "pc": round(ev.pc_score(a, b), 3), "tds": ev.tds_score(set(a), set(b))}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--endpoint", default="")
    p.add_argument("--model", default="")
    p.add_argument("--only", default="", help="comma-separated item ids (default: the 6 by the rule)")
    p.add_argument("--keep", action="store_true", help="append to the jsonl instead of starting fresh")
    p.add_argument("--ids", default="", help="supplementary run: 'id:lang,id:lang' instead of the rule's 6 items; "
                                            "written to s6_trial_<tag>.jsonl so the 6-item trial stays untouched")
    p.add_argument("--tag", default="extra")
    args = p.parse_args(argv)
    global JL, JS, DB
    if args.ids:
        JL, JS, DB = (os.path.join(OUT, f"s6_trial_{args.tag}{ext}") for ext in (".jsonl", "_summary.json",
                                                                                   "_memory.db"))

    import agent_llm
    import general_set as gs
    import step4_gemma as sg
    import step4_refplans as rp
    model = args.model or sg.MODEL
    endpoint = (args.endpoint or agent_llm.gateway_endpoint() or "http://127.0.0.1:11434").rstrip("/")
    items = gs.load(rp.DATASET)["items"]
    chosen = select_items(items, rp.read_jsonl(rp.jsonl_path("oracle")))
    if args.ids:
        by = {it["id"]: it for it in items}
        chosen = [(by[i], lang) for i, lang in (x.strip().split(":") for x in args.ids.split(",") if x.strip())]
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        chosen = [c for c in chosen if c[0]["id"] in want]
    print(f"trial items: {[(it['id'], it['level'], lang) for it, lang in chosen]}", flush=True)

    # ---- the LLM side: reachable? model present? (no pull here -- the operator checks `ollama list`)
    try:
        ver = _get(endpoint, "/api/version").get("version")
        tags = {m["name"]: m for m in _get(endpoint, "/api/tags").get("models", [])}
    except Exception as e:
        print(f"ABORT: Ollama not reachable at {endpoint}: {e}", flush=True)
        return 3
    if model not in tags:
        print(f"ABORT: model {model} not in `ollama list` at {endpoint}: {sorted(tags)}", flush=True)
        return 3
    llm_info = {"endpoint": endpoint, "model": model, "ollama_version": ver, "digest": tags[model].get("digest"),
                "details": tags[model].get("details"), "options": sg.OPTIONS, "think": sg.THINK,
                "prompt_version": sg.PROMPT_VERSION}
    print(f"LLM: {llm_info}", flush=True)
    provider = agent_llm.make_provider("ollama", endpoint=endpoint)

    if not args.keep:
        for pth in (JL, DB, DB + "-wal", DB + "-shm"):
            if os.path.exists(pth):
                os.remove(pth)
    import eval as ev
    import levels
    import thor_actions as ta
    vram0 = vram()
    ag = ta.ThorAgent(scene=SCENE, width=600, height=600, render_depth=True)   # Controller BEFORE ultralytics
    import webapp
    if "fast" in webapp.MOTION_PROFILES:
        webapp._MOTION["speed"] = "fast"
    import honest
    import honest_skills as hs
    import objmem
    from honest_map import GridMap
    from memory import Memory
    det = honest.load_detector(SCENE)
    types = hs.detector_types(det)
    store = Memory(path=DB)
    om = objmem.ObjectMemory(SCENE, store=store, episode=None, fov=ag.field_of_view or 90.0)
    scene_types = ag.type_list(visible_only=False)       # scoring: the applicability gate (pre.scoring)
    vram_sim = vram()
    print(f"detector types: {len(types)}; VRAM before sim {vram0}, with sim + detector {vram_sim}", flush=True)
    rows = []
    aborted = ""
    try:
        for n, (it, lang) in enumerate(chosen, 1):
            t0 = time.time()
            command = it[lang]
            rec = levels.CommandRecord(id=it["id"], level=it["level"], lang=lang, command=command,
                                       gold=gs.gold_of(it), setup=it.get("setup") or None)
            goal = rec.gold["goal"]
            row = {"id": it["id"], "level": it["level"], "lang": lang, "command": command, "status": "ok",
                   "detail": "", "ref_plan": it["plan"], "pre": {"scoring": 1, "oracle_setup": 0}, "llm": llm_info,
                   "planner": None, "plan_scores": None, "first_plan_scores": None, "steps": [], "exec_ok": None,
                   "goal_ok": None, "vram": {}, "latency": {}, "audit": None}
            h = None
            try:
                ev._reset_world(ag, SCENE)
                absent = ev.missing_types(ev.gold_types(rec), scene_types)
                if absent:
                    row.update(status="skipped", detail=f"object does not exist: {', '.join(absent)}")
                    raise _Settled
                setup = rec.setup or []
                sres = ev.run_steps(ag, setup)
                row["pre"]["oracle_setup"] = len(setup)
                bad = next((f"{st['action']} {st.get('object') or ''}: {r['error']}"
                            for st, r in zip(setup, sres) if not r["ok"]), None)
                if bad:
                    row.update(status="FAIL", detail=f"setup failed -- {bad}")
                    raise _Settled
                row["pre"]["scoring"] += 1
                if rp._goal_ok(ev, ag, goal):
                    row.update(status="skipped", detail="goal already true before acting")
                    raise _Settled
                holding = rp.setup_holding(setup, sres)
                row["told_holding"] = holding
                # ------------------------------------------------------------ honest episode: plan + execute
                h = honest.HonestAgent(ag, det, mode="honest")
                planner = sg.GemmaPlanner(provider=provider, model=model, types=types)
                sk = hs.HonestSkills(h, GridMap.load(SCENE), om, chooser=planner.chooser())
                seed = sk.start_episode(f"s6:{it['id']}:{lang}", holding=holding)
                row["seed_tracks"], row["sweep_prims"] = seed["tracks"], h.step
                row["memory_summary"] = seed["summary"]
                row["vram"]["before_plan"] = vram()
                res = planner.plan(command, seed["summary"], holding)
                row["vram"]["after_plan"] = vram()
                row["planner"] = res
                row["plan_scores"] = plan_scores(ev, res["plan"], it["plan"]) if res["ok"] else None
                row["first_plan_scores"] = plan_scores(ev, res["first_plan"], it["plan"])
                print(f"  plan ({res['secs']} s, first_ok={res['first_ok']}, repaired={res['repaired']}): "
                      f"{[(s['action'], s['object'], s['to']) for s in res['plan']]}", flush=True)
                te = time.time()
                if res["ok"]:
                    for i, st in enumerate(res["plan"]):
                        if h.step > EP_PRIM_CAP:
                            row["steps"].append({"i": i, **st, "ok": False, "cause": "episode_budget",
                                                 "why": f"episode primitive cap {EP_PRIM_CAP}; not run"})
                            continue
                        ts = time.time()
                        sk.chooser.last = None
                        r = sk.exec_step(st)
                        d = r["detail"]
                        row["steps"].append({"i": i, **st, "ok": r["ok"], "cause": r["cause"],
                                             "under": rp.underlying(r["cause"], d["why"]), "retries": r["retries"],
                                             "prims": d["prims"], "why": d["why"], "verify": d["verify"],
                                             "holding": d["holding"], "search": d.get("search"),
                                             "chooser": (sk.chooser.last if d.get("search") else None),
                                             "events": d["events"][-12:], "secs": round(time.time() - ts, 1)})
                        print(f"  step {i} {st['action']} {st['object']} {st['to'] or ''}: ok={r['ok']} "
                              f"cause={r['cause']} retries={r['retries']} prims={d['prims']}", flush=True)
                        if h.audit.invalid:
                            break
                row["latency"] = {"plan_s": res["secs"], "exec_s": round(time.time() - te, 1),
                                  "calls": list(planner.calls),
                                  "chooser_s": round(sum(c["secs"] for c in planner.calls if c["kind"] == "chooser"), 2)}
                row["prims"] = h.step
                row["vram"]["after_exec"] = vram()
                with h.scoring():                        # goal truth: report only, after the plan
                    h.meta("scoring")
                    row["goal_ok"] = rp._goal_ok(ev, ag, goal)
                    unmet = [] if row["goal_ok"] else rp._unmet(ev, gs, ag, goal)
                row["audit"] = {**h.audit.summary(), "counts": dict(h.audit.counts),
                                "errors_logged": [(s_, a_, e_[:100]) for s_, a_, e_ in h.audit.errors][-20:]}
                failed = [s for s in row["steps"] if not s["ok"]]
                row["exec_ok"] = bool(res["ok"]) and not failed
                if not res["ok"]:
                    row.update(status="FAIL", detail="plan rejected after one repair -- not executed: "
                               + "; ".join(e["reason"] for e in res["attempts"][-1]["validation"]["errors"])[:300])
                elif not row["goal_ok"]:
                    row.update(status="FAIL", detail=f"goal unmet: {'; '.join(unmet)}" + (
                        " | step errors: " + " / ".join(f"{s['action']} {s.get('object') or ''}: {s['cause']}"
                                                        for s in failed) if failed else ""))
                elif failed:
                    row["detail"] = "goal met despite step errors"
            except _Settled:
                pass
            except Exception as exc:
                row.update(status="FAIL", detail=f"exception: {type(exc).__name__}: {exc}"[:300])
            if h is not None and row["audit"] is None:
                row["audit"] = {**h.audit.summary(), "counts": dict(h.audit.counts)}
            if rp.invalid({**row, "mode": "honest"}):
                row.update(status="INVALID", detail=f"honesty violation / privileged read -- {row['detail']}"[:300])
            row["elapsed_s"] = round(time.time() - t0, 1)
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            rows.append(row)
            with open(JL, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            ps = row["plan_scores"] or {}
            print(f"[{n}/{len(chosen)}] {row['status']:7s} {row['id']} ({lang}) goal_ok={row['goal_ok']} "
                  f"pc={ps.get('pc')} {row['elapsed_s']} s {row['detail'][:120]}", flush=True)
            if row["status"] == "INVALID":
                aborted = f"honesty violation in {it['id']}"
                break
    finally:
        try:
            ag.stop()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass
    summ = summarize(rows, llm_info, vram0, vram_sim)
    summ["aborted"] = aborted
    with open(JS, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: summ[k] for k in ("tally", "per_level", "plans", "goal_ok", "mean_pc")}, ensure_ascii=False),
          flush=True)
    return 2 if aborted else 0


def summarize(rows, llm_info, vram0, vram_sim) -> dict:
    run = [r for r in rows if r["status"] != "skipped"]
    tally, per = {}, {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
        d = per.setdefault(r["level"], {"n": 0, "goal_ok": 0, "plan_valid": 0})
        d["n"] += 1
        d["goal_ok"] += int(bool(r.get("goal_ok")))
        d["plan_valid"] += int(bool((r.get("planner") or {}).get("ok")))
    pl = [r["planner"] for r in run if r.get("planner")]
    pcs = [r["plan_scores"]["pc"] for r in run if r.get("plan_scores")]
    lat = [c["secs"] for r in run for c in (r.get("latency") or {}).get("calls", []) if c["kind"] != "chooser"]
    used = [v.get("used_mib") for r in run for v in (r.get("vram") or {}).values() if v.get("used_mib")]
    return {"scene": SCENE, "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "integration pilot (6 items)",
            "llm": llm_info, "n": len(rows), "tally": tally, "per_level": per,
            "plans": {"valid_first_pass": sum(p["first_ok"] for p in pl), "repaired": sum(p["repaired"] for p in pl),
                      "rejected": sum(p["rejected"] for p in pl)},
            "goal_ok": sum(bool(r.get("goal_ok")) for r in run), "exec_ok": sum(bool(r.get("exec_ok")) for r in run),
            "mean_pc": round(sum(pcs) / len(pcs), 3) if pcs else None,
            "planner_call_s": {"n": len(lat), "mean": round(sum(lat) / len(lat), 2) if lat else None,
                               "max": max(lat) if lat else None},
            "chooser_calls": sum(1 for r in run for c in (r.get("latency") or {}).get("calls", [])
                                 if c["kind"] == "chooser"),
            "vram_mib": {"before_sim": vram0, "with_sim_detector": vram_sim, "max_during": max(used) if used else None},
            "violations": sum(int((r.get("audit") or {}).get("violations") or 0) for r in rows),
            "oracle_reads_in_episode": sum(int(((r.get("audit") or {}).get("counts") or {}).get("oracle") or 0)
                                           for r in rows),
            "rows": [{"id": r["id"], "level": r["level"], "lang": r["lang"], "status": r["status"],
                      "goal_ok": r.get("goal_ok"), "plan": (r.get("planner") or {}).get("plan"),
                      "first_ok": (r.get("planner") or {}).get("first_ok"),
                      "pc": (r.get("plan_scores") or {}).get("pc"), "nlev": (r.get("plan_scores") or {}).get("nlev"),
                      "plan_s": (r.get("latency") or {}).get("plan_s"), "elapsed_s": r.get("elapsed_s"),
                      "detail": r.get("detail")} for r in rows]}


if __name__ == "__main__":
    sys.exit(main())
