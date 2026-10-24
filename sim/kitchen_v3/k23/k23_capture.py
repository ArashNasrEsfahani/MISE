"""K23 capture wrapper: the Kitchen 1 capture code (file unchanged; FloorPlan15 Blinds Open/Close excluded by amendment K23-AMD-1), run in BATCHES, for FloorPlan15 / FloorPlan17 only.

It imports ``build_kitchen_data`` and calls its own stages with the Kitchen 1 configuration objects (``kd.CAPTURE``,
``kd.CAPTURE_AMB``, the frozen plans) -- no protocol value is copied or changed here (A28 item 4: capture reused from
Kitchen 1 unchanged). The only addition is a batch limit: the run stops cleanly after ``--max-new`` newly finished
episodes (committed or recorded failed), through ``stage_capture``'s existing ``stop_when`` hook, so a quality check
can run between batches. A stop is the same as the resume Kitchen 1 used many times (state is saved per episode; the
next run continues at the next planned episode).

    # WSL, inside the sim lock (capture_one_k23.sh does this):
    python sim/kitchen_v3/k23/k23_capture.py --scene FloorPlan15 --pipeline main --max-new 8
    python sim/kitchen_v3/k23/k23_capture.py --scene FloorPlan15 --pipeline probe       # probe only (skipped if present)

Pipelines in THIS release (Astra K23-PREP 10): probe | main (probe if absent, then the 168-episode main capture) |
thermaltest (opens the simulator and renders views for at most 5 minutes, writes NO data: the target of the thermal
pause test). topup and amb are refused until they are verified separately (their batching and QC paths differ).
The audit runs at the end of every capture call, as in Kitchen 1 (``stage_capture`` returns ``stage_audit``); its
summary is logged (a partial capture is expected to be short of its targets).

Refuses: any scene other than FloorPlan15 / FloorPlan17 (Kitchen 1 data is never written by this tool); a per-kitchen
output root (~/thor-data/kitchen_v3/<scene>) that is, or resolves through a symlink to, a Kitchen 1 store.

Exit codes: 0 batch done or pipeline complete (a COMPLETE line says which), 3 fail-fast abort (as Kitchen 1),
4 refused, 5 the probe class list does not match the expected list (stop and consult).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
if SIM not in sys.path:
    sys.path.insert(0, SIM)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

K23_SCENES = ("FloorPlan15", "FloorPlan17")
EXIT_OK, EXIT_ABORT, EXIT_REFUSED, EXIT_CLASSES = 0, 3, 4, 5


def roots():
    home = os.path.expanduser("~")
    return (os.path.join(home, "thor-data", "kitchen_v3"), os.path.join(home, "thor-data", "kitchen_v3_amb"))


def log(scene, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {scene}: K23 {msg}", flush=True)


def finished(state_path: str) -> set:
    """Episodes this store has finished: committed (done) or recorded failed."""
    if not os.path.exists(state_path):
        return set()
    with open(state_path, encoding="utf-8") as fh:
        st = json.load(fh)
    return set(st.get("done_episodes") or []) | {int(k) for k in (st.get("failed_episodes") or {})}


def batch_stop(state_path: str, max_new: int, scene: str, plan_ids: set):
    """``stop_when`` hook: True once ``max_new`` episodes of THIS plan were finished since the call started.

    ``stage_capture`` calls it before each pending episode; it reads the saved state, so it counts exactly the
    episodes the capture itself recorded (a redo inside one episode is not counted twice)."""
    start = finished(state_path) & plan_ids
    fired = {"done": False}

    def stop(_quota) -> bool:
        if max_new is None or max_new <= 0:
            return False
        n = len((finished(state_path) & plan_ids) - start)
        if n >= max_new:
            if not fired["done"]:
                log(scene, f"BATCH LIMIT: {n} new episode(s) finished in this run (limit {max_new}); stopping "
                           f"cleanly (the next line's generic wording comes from stage_capture)")
                fired["done"] = True
            return True
        return False
    return stop


def plan_status(state_path: str, plan: list) -> dict:
    ids = {e["episode"] for e in plan}
    fin = finished(state_path) & ids
    return {"planned": len(ids), "finished": len(fin), "remaining": len(ids) - len(fin)}


def episode_files(ds: str) -> list:
    out = []
    for sub in ("", "pairs"):
        for kind in ("images", "depth", "seg"):
            for split in ("train", "val", "test"):
                d = os.path.join(ds, sub, kind, split)
                if os.path.isdir(d):
                    out += [os.path.join(d, f) for f in os.listdir(d)]
    return out


def clean_uncommitted(bkd, ds: str) -> int:
    """Only when NO capture_state.json exists (nothing committed): remove every frame file through the Kitchen 1
    resume cleaner. Returns how many files were there; refuses (raises) if a state file exists."""
    if os.path.exists(os.path.join(ds, "capture_state.json")):
        raise RuntimeError("clean_uncommitted called with a committed state")
    n = len(episode_files(ds))
    if n:
        bkd._truncate(os.path.join(ds, "frames.jsonl"), ds, set())
    left = episode_files(ds)
    if left:
        raise RuntimeError(f"{len(left)} uncommitted files survived the cleanup, e.g. {left[:2]}")
    return n


SIM_PATTERN = "[t]hor-Linux64"
if os.environ.get("K23_TEST_MODE") == "1" and os.environ.get("K23_TEST_SIM_PATTERN"):
    SIM_PATTERN = os.environ["K23_TEST_SIM_PATTERN"]          # integration test only: a stub stands for the simulator


def proc_table(proc: str = "/proc") -> dict:
    """{pid: (ppid, pgid, sid, cmdline)} from /proc (Linux); {} where /proc does not exist."""
    out = {}
    if not os.path.isdir(proc):
        return out
    for d in os.listdir(proc):
        if not d.isdigit():
            continue
        try:
            stat = open(os.path.join(proc, d, "stat"), encoding="utf-8", errors="replace").read()
            rest = stat[stat.rindex(")") + 2:].split()             # after "pid (comm) "
            cmd = open(os.path.join(proc, d, "cmdline"), "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
            out[int(d)] = (int(rest[1]), int(rest[2]), int(rest[3]), cmd.strip())
        except (OSError, ValueError, IndexError):
            continue
    return out


def owned_simulators(table: dict, me: int, me_pgid: int, me_sid: int, pattern: str = None) -> list:
    """Simulator processes THIS capture owns (Astra K23-PREP r3 5), never a name-wide match: the descendants of this
    process, plus members of its own process group when that group is a fresh session (pgid == sid, as the batch
    script's ``setsid`` makes it; an orphaned simulator keeps its group). Everything else -- another job's simulator,
    a simulator of a person's session -- is never touched."""
    import re
    pat = re.compile(pattern or SIM_PATTERN)
    kids = {}
    for pid, (ppid, _g, _s, _c) in table.items():
        kids.setdefault(ppid, []).append(pid)
    desc, todo = set(), [me]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in desc:
                desc.add(c)
                todo.append(c)
    own_group = me_pgid > 1 and me_pgid == me_sid
    out = []
    for pid, (_p, g, s, cmd) in table.items():
        if pid == me or not pat.search(cmd):
            continue
        if pid in desc or (own_group and g == me_pgid and s == me_sid):
            out.append(pid)
    return sorted(out)


def kill_owned_simulators(grace: float = 10.0) -> dict:
    """TERM the owned simulators, wait, KILL what is left. Every signal is LIFETIME-SAFE (Astra K23-PREP r5 4): the
    pid's start time is taken from a /proc snapshot that agrees with the ownership table, and k23_procs.safe_signal
    signals through a pidfd re-verified against it, so a pid reused meanwhile is never hit. Returns what was done."""
    import signal
    import k23_procs
    me = os.getpid()

    def owned_now():
        tab = proc_table()
        snap = k23_procs.snapshot()
        out = {}
        for p in owned_simulators(tab, me, os.getpgid(0), os.getsid(0)):
            v = snap.get(p)
            if v and (v["ppid"], v["pgid"], v["sid"]) == tab[p][:3]:
                out[p] = v["start"]
        return out
    first = owned_now()
    for p, st in first.items():
        k23_procs.safe_signal(p, st, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < grace and owned_now():
        time.sleep(0.5)
    left = owned_now()
    for p, st in left.items():
        k23_procs.safe_signal(p, st, signal.SIGKILL)
    return {"termed": sorted(first), "killed": sorted(left)}


def install_scoped_restart(bkd, scene: str) -> None:
    """Replace, AT RUNTIME and for this process only, the frozen Kitchen 1 hang / render-glitch recovery
    ``build_kitchen_data._restart_agent`` (which runs a global ``pkill -f thor-Linux64``) by the same steps with an
    OWNED-ONLY kill: stop the agent, stop the simulators this capture owns, wait 3 s, start a fresh agent. The Kitchen 1
    file itself is not edited (k1_unchanged checks it); the capture protocol is unchanged (only the kill scope)."""
    def _restart_agent_scoped(agent, scene_):
        try:
            agent.stop()
        except BaseException:
            pass
        done = kill_owned_simulators()
        log(scene_, f"scoped simulator restart (owned only): TERM {done['termed']}, KILL {done['killed']}")
        time.sleep(3)
        return bkd.make_agent(scene_)
    bkd._restart_agent = _restart_agent_scoped


# r9 / r10 (K2-CAPTURE 2026-10-15), scoped amendment K23-AMD-1 (k23/amendment_K23-AMD-1.json + ledger; Astra r9, r10):
# AI2-THOR 5.0 timed out on OpenObject for FloorPlan15 Blinds (probe 1 and main batch 1). On FloorPlan15 only:
#  - probe (set_state): a Blinds target that needs OpenObject is EXCLUDED (infrastructure) -- not attempted, error text
#    "EXCLUDED (infrastructure, K23-AMD-1)", listed in <scene>/excluded_states.json; it is NOT empirically unreachable
#    (the main capture uses only probe["reachable"], so it never targets that state);
#  - agent level (every action, incl. paired renders): OpenObject (observed timeout) and CloseObject (PRECAUTIONARY:
#    not observed to hang; its only use there is the back-step of the excluded open) on Blinds are never sent.
# This departs from the registered "unchanged capture" (A28 sections 4-5) for these Blinds actions only; Blinds stays a
# type-only detection class; every other scene / type / action is as registered.
HANG_ACTIONS = {("FloorPlan15", "Blinds"): frozenset({"OpenObject"})}
EXCL_DIR = os.path.dirname(os.path.abspath(__file__))          # <EXCL_DIR>/<scene>/excluded_states.json (tests override)


class _ExcludedEvent:
    """What an excluded action returns instead of a simulator step: a plain failure, the scene unchanged."""
    def __init__(self, action: str, oid: str):
        self.metadata = {"lastActionSuccess": False, "lastAction": action,
                         "errorMessage": f"EXCLUDED (infrastructure, K23-AMD-1): {action} on {oid} not sent"}


# r10 (K23-AMD-1, agent level): build_kitchen_data.paired_render flips a visible object's state (OpenObject, then
# CloseObject back); main batch 1 died there. An excluded action returns a plain failure without a simulator step
# (the scene event is not replaced); the pair is recorded failed (ok / back_ok False, valid_evidence False), never evidence.
AGENT_EXCLUDED = {("FloorPlan15", "Blinds"): frozenset({"OpenObject", "CloseObject"})}


def wrap_agent_excluded(agent, scene: str):
    real_do = agent.do

    def do_k23(action, **kw):
        oid = str(kw.get("objectId") or "")
        if action in AGENT_EXCLUDED.get((scene, oid.split("|")[0]), frozenset()):
            log(scene, f"action NOT sent (K23-AMD-1): {action} {oid[:48]}")
            return _ExcludedEvent(action, oid)
        return real_do(action, **kw)
    agent.do = do_k23
    return agent


def topup_terminated(ds: str, sp: str, top_plan: list, kd) -> bool:
    """The TRAIN top-up really terminated: a topup_complete.json marker exists AND either the top-up plan is exhausted in
    the store, or the STORED quota meets every TRAIN target (kd.train_complete) -- the marker text alone proves nothing."""
    if not os.path.exists(os.path.join(ds, "topup_complete.json")):
        return False
    try:                                         # a malformed / unreadable state is never a terminated top-up
        if not plan_status(sp, top_plan)["remaining"]:
            return True
        st = json.load(open(sp, encoding="utf-8"))
        return bool(kd.train_complete(kd.Quota.from_dict(st["quota"])))
    except Exception:
        return False


def parity_prereq(scene: str, ds: str, sp: str, main_plan: list, kd) -> list:
    """What is not yet complete / verified before parity capture (empty = may run)."""
    import k23_decide
    try:
        qc_bad = k23_decide.qc_unresolved(scene)
    except Exception as exc:
        qc_bad = [f"QC evidence unreadable: {exc!r}"]
    return [n for n, ok in (("main", not plan_status(sp, main_plan)["remaining"]),
                            ("topup", topup_terminated(ds, sp, kd.topup_train_plan(scene, kd.TOPUP_TRAIN_MAX), kd)),
                            ("close", not plan_status(sp, kd.close_plan(scene, kd.CLOSE_EPISODES))["remaining"]),
                            (f"QC resolved {qc_bad[:2]}", not qc_bad)) if not ok]


def parity_target_new_train(ds: str, k1_train: int = 16299, retention: float = 0.949) -> tuple:
    """(kept TRAIN frames now, provisional NEW raw TRAIN frames needed): the cleaned TRAIN set must reach Kitchen 1's
    16,299; the measured cleaning retention 0.949 is a PROVISIONAL checkpoint (Astra K2-PARITY-r1 2): after a parity run
    the store is cleaned, verified and recounted, and parity resumes within the same 150-episode plan if still short."""
    import math
    kept = 0
    with open(os.path.join(ds, "frames.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if '"img"' not in line:
                continue
            r = json.loads(line)
            if "img" in r and r["split"] == "train" and not r.get("quarantined"):
                kept += 1
    return kept, max(0, math.ceil((k1_train - kept) / retention))


def run_parity(a, bkd, kd, ds: str, sp: str, main_plan: list, main_root: str) -> int:
    """PARITY capture (Arash 2026-10-17 '(b-150) ... more close ups'; Astra K2-PARITY-r1): extra TRAIN episodes with the
    FROZEN close-up view policy (build_kitchen_data.choose_view's CLOSE_PROTOCOL branch), every close-up TRAIN frame kept
    (kd.accept_train -> True in THIS process only; render / label / validity checks unchanged), held views off
    (p_held 0 in a parity-only config copy), stop when the provisional raw-frame target is met, the batch limit hits, or the
    150-episode plan is exhausted."""
    import k23_plans
    why_not = parity_prereq(a.scene, ds, sp, main_plan, kd)
    if why_not:
        log(a.scene, f"REFUSED: parity before {why_not} complete")
        return EXIT_REFUSED
    kept, need = parity_target_new_train(ds)
    plan = k23_plans.parity_plan(a.scene, kd)
    if need <= 0:
        log(a.scene, f"parity after batch: {plan_status(sp, plan)} -- COMPLETE parity (target achieved: kept TRAIN {kept})")
        return EXIT_OK
    cfg = dict(kd.CAPTURE, p_held=0.0)                      # r1 1: no in-hand views in parity episodes
    orig_cv, orig_acc, orig_sd = bkd.choose_view, kd.accept_train, kd.Quota.split_done

    def choose_view_parity(agent, positions, probe, quota, split, rng, cfg_=kd.CAPTURE, protocol=None):
        return orig_cv(agent, positions, probe, quota, split, rng, cfg_,
                       protocol=kd.CLOSE_PROTOCOL if protocol == k23_plans.PARITY_PROTOCOL else protocol)
    bkd.choose_view = choose_view_parity
    kd.accept_train = lambda *_a, **_k: True               # every close-up TRAIN frame kept (this process only)
    # the frozen quota skip ('every train target already met' -> train episode skipped) must not skip parity episodes
    kd.Quota.split_done = lambda self, split: False if split == "train" else orig_sd(self, split)
    bstop = batch_stop(sp, a.max_new, a.scene, {e["episode"] for e in plan})
    start = {"train": None}
    met = {"v": False}

    def stop(quota) -> bool:
        tr = int((getattr(quota, "frames", {}) or {}).get("train", 0))
        if start["train"] is None:
            start["train"] = tr
        if tr - start["train"] >= need:
            met["v"] = True
            return True
        return bstop(quota)
    log(a.scene, f"parity batch: kept TRAIN {kept}, provisional new raw TRAIN needed {need}; max {a.max_new} new "
                 f"episodes; before: {plan_status(sp, plan)}")
    try:
        aud = bkd.stage_capture(a.scene, main_root, cfg, plan_override=plan, stop_when=stop)
    finally:
        bkd.choose_view, kd.accept_train, kd.Quota.split_done = orig_cv, orig_acc, orig_sd
    stt = plan_status(sp, plan)
    try:
        log(a.scene, "AUDIT: " + json.dumps({k: aud.get(k) for k in ("ok", "n_short", "missing_types", "failed_episodes")
                                             if isinstance(aud, dict) and k in aud}, default=str)[:400])
    except Exception:
        pass
    done = met["v"] or not stt["remaining"]
    why = ("provisional raw target met; clean + recount next" if met["v"] else "plan exhausted (cap 150)") if done else ""
    log(a.scene, f"parity after batch: {stt}" + (f" -- COMPLETE parity ({why})" if done else ""))
    return EXIT_OK


def install_dup_ruling(bkd, scene: str, amb: bool = False) -> None:
    """Arash 2026-10-16 16:40 ("nothing like this even if it reaached 65 should stop it. in the end they should just be
    deleted"): near-duplicates never stop a FloorPlan15 / FloorPlan17 capture. The K23 QC already applies it; the frozen
    capture's own every-N checkpoint (bkd._checkpoint) still aborted on a near_duplicates FAIL (close protocol, 2026-10-17
    16:13). This wrapper runs the SAME verifier and writes the same checkpoint file; it continues only when EVERY FAIL is
    near_duplicates in a ruled scene (logged as a NOTE), any other FAIL aborts exactly as before."""
    import k23_qc
    allowed = set(["near_duplicates"] if scene in k23_qc.DUP_RULING_SCENES else [])
    if amb:                                  # K2-AMB r2 2: fixed-plan amb coverage is reported, never an abort
        allowed |= set(k23_qc.CC_NOTE_CHECKS)
    if not allowed:
        return

    def checkpoint_k23(scene_, ds, cfg, n_done):
        every = int(cfg.get("check_every") or 0)
        if not (every and n_done % every == 0):
            return
        import check_capture
        rep = check_capture.check(ds, cfg)
        fails = [x for x in rep["results"] if x["status"] == "FAIL"]
        only_dups = bool(fails) and all(x["check"] in allowed for x in fails)
        bkd._log(scene_, f"CHECKPOINT after {n_done} episodes — "
                         f"{'NOTE (ruled checks only: ' + ','.join(sorted(allowed)) + ')' if only_dups else ('FAIL' if rep['fail'] else 'ok')}\n"
                 + check_capture.render(rep))
        bkd._dump(os.path.join(ds, "checkpoints", f"after_{n_done:03d}.json"), rep)
        if only_dups:
            log(scene_, f"{sorted({x['check'] for x in fails})} FAIL recorded as NOTE ({k23_qc.DUP_RULING}"
                        f"{'; ' + k23_qc.AMB_RULING if amb else ''}): "
                        + "; ".join(x["detail"][:160] for x in fails))
            return
        if rep["fail"]:
            raise bkd.CaptureAborted(f"checkpoint after {n_done} episodes failed: "
                                     + "; ".join(f"{x['check']}: {x['detail']}" for x in fails))
    bkd._checkpoint = checkpoint_k23


def install_hang_skip(bkd, scene: str) -> None:
    orig = bkd.set_state
    orig_make = bkd.make_agent
    bkd.make_agent = lambda scene_: wrap_agent_excluded(orig_make(scene_), scene_)

    def set_state_k23(agent, oid, target):
        obj = agent.object_by_id(oid) or {}
        skip = HANG_ACTIONS.get((scene, obj.get("objectType")), frozenset())
        if skip:
            steps = bkd.vocab.actions_for(target, bkd.vocab.state_record(obj))
            bad = [s[0] for s in steps if s[0] in skip]
            if bad:
                log(scene, f"state target NOT attempted (known AI2-THOR hang: {bad[0]} on {obj.get('objectType')}): "
                           f"{oid[:48]} {target}")
                why = "EXCLUDED (infrastructure, K23-AMD-1): not attempted, AI2-THOR 5.0 step timeout observed 2026-10-15"
                try:
                    xp = os.path.join(EXCL_DIR, scene, "excluded_states.json")
                    xs = json.load(open(xp, encoding="utf-8")) if os.path.exists(xp) else []
                    xs.append({"object": oid, "type": obj.get("objectType"), "target": target, "action": bad[0],
                               "why": why, "amendment": "K23-AMD-1", "t": time.strftime("%Y-%m-%d %H:%M:%S")})
                    with open(xp, "w", encoding="utf-8") as fh:
                        json.dump(xs, fh, indent=1)
                except OSError as exc:
                    log(scene, f"excluded_states.json not written: {exc!r}")
                return {"target": target, "steps": [s[0] for s in steps], "realised": False, "pieces": [],
                        "errors": [[bad[0], oid, why]]}
        return orig(agent, oid, target)
    bkd.set_state = set_state_k23


def check_classes(scene: str, probe_path: str) -> bool:
    import k23_classes
    res = k23_classes.check_probe(probe_path, json.load(open(k23_classes.EXPECTED, encoding="utf-8")))
    log(scene, f"CLASS LIST {'MATCH' if res['ok'] else 'MISMATCH'}: probe {res['n_probe']} classes "
               f"({res['n_shown']} shown), registry sha256 {res['probe_registry_sha256']}"
               + ("" if res["ok"] else f"; missing {res.get('missing_vs_expected')}, extra "
                                       f"{res.get('extra_vs_expected')}"))
    return bool(res["ok"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pipeline", required=True, choices=("probe", "main", "thermaltest", "topup", "close", "amb", "parity"))
    ap.add_argument("--max-new", type=int, default=8, help="episodes per batch (0 = no limit)")
    a = ap.parse_args(argv)
    if a.scene not in K23_SCENES:
        log(a.scene, f"REFUSED: this tool captures only {K23_SCENES}; Kitchen 1 data is never written by it")
        return EXIT_REFUSED
    import build_kitchen_data as bkd
    import kitchen_data as kd

    install_scoped_restart(bkd, a.scene)          # no global simulator kill in the K23 path (Astra K23-PREP r3 5)
    install_hang_skip(bkd, a.scene)               # r9: never call a state action known to hang the simulator
    install_dup_ruling(bkd, a.scene, amb=a.pipeline == "amb")   # near_duplicates never abort (ruling); amb coverage NOTE
    main_root, amb_root = roots()
    if os.path.abspath(bkd.DEFAULT_ROOT) != os.path.abspath(main_root):
        log(a.scene, f"REFUSED: build_kitchen_data.DEFAULT_ROOT is {bkd.DEFAULT_ROOT}, expected {main_root}")
        return EXIT_REFUSED
    ds = os.path.join(main_root, a.scene)
    k1 = {os.path.realpath(os.path.join(main_root, "FloorPlan1")), os.path.realpath(os.path.join(amb_root, "FloorPlan1"))}
    if os.path.lexists(ds) and (os.path.islink(ds) or os.path.realpath(ds) in k1):
        log(a.scene, f"REFUSED: {ds} is a symlink / resolves to {os.path.realpath(ds)} (Kitchen 1 stores are never written)")
        return EXIT_REFUSED
    probe_path = os.path.join(ds, "probe.json")
    try:
        if a.pipeline == "thermaltest":
            # the pause-test target: a real simulator load, nothing written (no store, no frames, no state)
            agent = bkd.make_agent(a.scene)
            try:
                agent.reset(a.scene)
                pos = agent.reachable_positions()
                import random
                rng = random.Random(0)
                t0 = time.time()
                n = 0
                log(a.scene, "THERMALTEST load running (no data written)")
                while time.time() - t0 < 300:
                    q = rng.choice(pos)
                    agent.teleport_full(q["x"], q["y"], q["z"], round(rng.uniform(0, 360), 1), horizon=0.0,
                                        standing=True)
                    n += 1
                log(a.scene, f"THERMALTEST ran {n} renders without being stopped (the guard did not fire)")
            finally:
                agent.stop()
            return EXIT_OK
        if not os.path.exists(probe_path):
            log(a.scene, "probe (not present yet)")
            bkd.stage_probe(a.scene, main_root)
        else:
            log(a.scene, "probe present, reused")
        if not check_classes(a.scene, probe_path):
            return EXIT_CLASSES
        if a.pipeline == "probe":
            log(a.scene, "COMPLETE probe")
            return EXIT_OK
        cfg = kd.CAPTURE
        main_plan = kd.episode_plan(a.scene, cfg["episodes"])
        sp = os.path.join(ds, "capture_state.json")
        if a.pipeline == "amb":
            # K2-AMB (Arash 2026-10-17 "Approve 1–5." item 2; Astra K2-AMB-r1): the frozen Kitchen 1 `amb` stage, batched --
            # kd.CAPTURE_AMB, all 70 kd.amb_plan episodes (sealed holdout 3300+ stored as split test, protocol AMB_HOLDOUT),
            # its own store <amb_root>/<scene>, the main store's probe byte-identical. Only after main + top-up + close.
            close_plan = kd.close_plan(a.scene, kd.CLOSE_EPISODES)
            top_plan = kd.topup_train_plan(a.scene, kd.TOPUP_TRAIN_MAX)
            top_ok = topup_terminated(ds, sp, top_plan, kd)    # K2-AMB r2 4 / r3 1: verified, never the marker's word
            try:                                  # ... and no QC report of the scene is unresolved (both stores)
                import k23_decide
                qc_bad = k23_decide.qc_unresolved(a.scene)
            except Exception as exc:
                qc_bad = [f"QC evidence unreadable: {exc!r}"]
            why_not = [n for n, ok in (("main", not plan_status(sp, main_plan)["remaining"]),
                                       ("topup", top_ok),
                                       ("close", not plan_status(sp, close_plan)["remaining"]),
                                       (f"QC resolved {qc_bad[:2]}", not qc_bad)) if not ok]
            if why_not:
                log(a.scene, f"REFUSED: amb before {why_not} complete")
                return EXIT_REFUSED
            ads = os.path.join(amb_root, a.scene)
            if os.path.lexists(ads) and (os.path.islink(ads) or os.path.realpath(ads) in k1):
                log(a.scene, f"REFUSED: {ads} is a symlink / resolves to a Kitchen 1 store")
                return EXIT_REFUSED
            os.makedirs(ads, exist_ok=True)
            app = os.path.join(ads, "probe.json")
            src = open(probe_path, "rb").read()
            if os.path.exists(app):
                if open(app, "rb").read() != src:
                    log(a.scene, f"REFUSED: {app} differs from the main store's probe (must be byte-identical)")
                    return EXIT_REFUSED
            else:
                with open(app + ".tmp", "wb") as fh:
                    fh.write(src)
                os.replace(app + ".tmp", app)
                log(a.scene, "amb store: probe copied byte-identical from the main store")
            asp = os.path.join(ads, "capture_state.json")
            if not os.path.exists(asp):
                log(a.scene, f"fresh amb store: {clean_uncommitted(bkd, ads)} file(s) of an uncommitted episode removed")
            plan = kd.amb_plan(a.scene)
            stop = batch_stop(asp, a.max_new, a.scene, {e["episode"] for e in plan})
            log(a.scene, f"amb batch: max {a.max_new} new episodes; before: {plan_status(asp, plan)}")
            aud = bkd.stage_capture(a.scene, amb_root, kd.CAPTURE_AMB, plan_override=plan, stop_when=stop)
            stt = plan_status(asp, plan)
            try:
                log(a.scene, "AUDIT: " + json.dumps({k: aud.get(k) for k in ("ok", "n_short", "missing_types",
                                                                               "failed_episodes")
                                                     if isinstance(aud, dict) and k in aud}, default=str)[:400])
            except Exception:
                pass
            log(a.scene, f"amb after batch: {stt}" + (" -- COMPLETE amb (plan exhausted; quotas NOT necessarily met)"
                                                      if not stt["remaining"] else ""))
            return EXIT_OK
        if a.pipeline == "parity":
            return run_parity(a, bkd, kd, ds, sp, main_plan, main_root)
        if a.pipeline in ("topup", "close"):
            # K2-TOPUP (Arash 2026-10-17 "Go for the top ups."): the frozen Kitchen 1 `topup` stage, split into its two
            # registered halves and batched -- capped TRAIN top-up (kd.topup_train_plan, stop at kd.train_complete),
            # then the close protocol (kd.close_plan, VAL / TEST). Only after the main plan is complete.
            if plan_status(sp, main_plan)["remaining"]:
                log(a.scene, f"REFUSED: {a.pipeline} before the main capture is complete ({plan_status(sp, main_plan)})")
                return EXIT_REFUSED
            marker = os.path.join(ds, "topup_complete.json")         # written once the TRAIN top-up terminated
            if a.pipeline == "close" and not os.path.exists(marker):
                log(a.scene, "REFUSED: close before the TRAIN top-up terminated (no topup_complete.json; Astra "
                             "K2-TOPUP-r1 1: Kitchen 1 order = top-up, then close)")
                return EXIT_REFUSED
            plan = (kd.topup_train_plan(a.scene, kd.TOPUP_TRAIN_MAX) if a.pipeline == "topup"
                    else kd.close_plan(a.scene, kd.CLOSE_EPISODES))
            bstop = batch_stop(sp, a.max_new, a.scene, {e["episode"] for e in plan})
            met = {"v": False}

            def stop(quota) -> bool:
                if a.pipeline == "topup" and kd.train_complete(quota):
                    met["v"] = True
                    return True
                return bstop(quota)
            log(a.scene, f"{a.pipeline} batch: max {a.max_new} new episodes; before: {plan_status(sp, plan)}")
            aud = bkd.stage_capture(a.scene, main_root, cfg, plan_override=plan, stop_when=stop)
            stt = plan_status(sp, plan)
            try:
                log(a.scene, "AUDIT: " + json.dumps({k: aud.get(k) for k in ("ok", "n_short", "missing_types",
                                                                               "failed_episodes")
                                                     if isinstance(aud, dict) and k in aud}, default=str)[:400])
            except Exception:
                pass
            done = met["v"] or not stt["remaining"]
            # "plan exhausted" does NOT mean the quotas passed (Astra K2-TOPUP-r1 1): the audit says what is still short
            why = ("every TRAIN target met" if met["v"] else "plan exhausted; quotas NOT necessarily met") if done else ""
            if done and a.pipeline == "topup":
                with open(marker + ".tmp", "w", encoding="utf-8") as fh:
                    json.dump({"scene": a.scene, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "why": why, "status": stt,
                               "audit_ok": aud.get("ok") if isinstance(aud, dict) else None}, fh, indent=1)
                os.replace(marker + ".tmp", marker)
            log(a.scene, f"{a.pipeline} after batch: {stt}" + (f" -- COMPLETE {a.pipeline} ({why})" if done else ""))
            return EXIT_OK
        plan = main_plan
        if not os.path.exists(sp):
            # no episode committed yet: stage_capture's fresh-start branch removes only frames.jsonl, so files of an
            # episode interrupted before the first commit would survive (Astra K23-PREP r2 9). The frozen Kitchen 1
            # resume cleaner (_truncate) is CALLED, not changed, with nothing done: every such file is removed.
            n_left = clean_uncommitted(bkd, ds)
            log(a.scene, f"fresh store: {n_left} file(s) of an uncommitted episode removed before the first episode")
        stop = batch_stop(sp, a.max_new, a.scene, {e["episode"] for e in plan})
        log(a.scene, f"main capture batch: max {a.max_new} new episodes; before: {plan_status(sp, plan)}")
        aud = bkd.stage_capture(a.scene, main_root, cfg, stop_when=stop)
        stt = plan_status(sp, plan)
        try:
            log(a.scene, "AUDIT (expected short while partial): " + json.dumps(
                {k: aud.get(k) for k in ("ok", "n_short", "missing_types", "failed_episodes")
                 if isinstance(aud, dict) and k in aud}, default=str)[:400])
        except Exception:
            pass
        log(a.scene, f"main capture after batch: {stt}" + (" -- COMPLETE main" if not stt["remaining"] else ""))
    except bkd.CaptureAborted as exc:
        print(f"[{time.strftime('%H:%M:%S')}] ABORTED (fail-fast): {exc}", flush=True)
        return EXIT_ABORT
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
