"""E1 / A11 - offline selftest: the planner runs AFTER the fixed start sweep (prereg s16 A11; docs/final-test/progress.md E1-plantiming).

    python sim/_step5_plantiming_selftest.py        (Windows python, repo root; no simulator, no WSL, no Ollama, no API)

  1  flags: plan_after_sweep / state_capture default OFF; the old order is untouched (one start_core, same stream as before)
  2  the post-sweep dict: the summary right after the sweep, nothing from scoring
  3  render_blocks: equals ltm.memory_block(post["wm"]) for the live LTM, deterministic, the live WM untouched, several sources
  4  an unexecuted plan closes the started attempt: terminal False, practice admission False, no scoring call, test writes nothing
  5  source order and honesty: hook after start_core, `.scoring()` only in _score* functions, state never reaches the hook
  6  c3_run PlanHook: post=None is the pre-A11 prompt, post=dict puts the post-sweep summary in the user message and the block from post["wm"]
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import re
import sys
import tempfile
import types

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                           # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "c3"))

import importlib                                             # noqa: E402

import ltm as L                                             # noqa: E402
import ltm_adapter as A                                     # noqa: E402
import step5_live as live                                   # noqa: E402

# W2 (A12 written into sim/step5/params.json): these checks are about the C2-e path and the event plumbing, so they read the registry WITHOUT the A12 block.
# The A12 block itself is tested in _step5_retention_a12_selftest.py.
_load_params_real = L.load_params


def _load_params_c2e(path=None):
    import copy as _copy
    P = _copy.deepcopy(_load_params_real(path))
    P["retrieval"].pop("retention", None)
    P["retrieval"].pop("a12", None)
    return P


L.load_params = _load_params_c2e

# The step-4 modules are pure geometry / agent code over the FakeThor kitchen (no controller, no socket). They are loaded BY NAME so that
# the static honesty gate (`_step5_honesty_selftest`: selftests import no step-4 module at top level) keeps seeing none, exactly as
# `_step5_retention_a12_selftest.py` does. (Alternative: whitelist this file in SELFTEST_STEP4; that file belongs to its own owner.)
honest = importlib.import_module("honest")
hs = importlib.import_module("honest_skills")
objmem = importlib.import_module("objmem")
GridMap = importlib.import_module("honest_map").GridMap

_Raw = A.HonestEventAdapter


class _AdapterG(_Raw):
    def __init__(self, *a, **k):
        k.setdefault("geometry", objmem)
        super().__init__(*a, **k)


A.HonestEventAdapter = _AdapterG
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def fake_ns():
    src = open(os.path.join(HERE, "_step4_act_selftest.py"), encoding="utf-8").read().split("\n")
    marker = next(i for i, ln in enumerate(src) if re.match(r"^# =+ 1$", ln))
    ns = {"__name__": "actfake", "__file__": os.path.join(HERE, "_step4_act_selftest.py")}
    exec(compile("\n".join(src[:marker]), "actfake_head", "exec"), ns)
    return ns


NS = fake_ns()
WB_CAB = "Cabinet|-0.55|+0.60|+2.60"


def world():
    objs = NS["kitchen"]()
    objs.append(dict(id=WB_CAB, type="Cabinet", pos=(-0.55, 0.6, 2.6), size=0.4, openable=True, receptacle=True, isOpen=False))
    objs.append(dict(id="WineBottle|x", type="WineBottle", pos=(-0.45, 0.6, 2.6), size=0.12, pickupable=True, inside=WB_CAB))
    return objs


def build(adapter):
    t = NS["FakeThor"](world(), (4, 4), 0.0)
    det = NS["FakeDetector"](t)
    h = honest.HonestAgent(t, det, mode="honest")
    gm = GridMap("Fake", NS["CELL"], (0.0, 0.0), NS["FREE"])
    om = objmem.ObjectMemory("Fake", store=None, episode="t")
    sk = hs.HonestSkills(h, gm, A.ObjectMemoryProxy(om, adapter), chooser=lambda *a: [], on_event=adapter.on_event,
                         search="exhaustive", tidy=True)
    adapter.clock = lambda: h.step
    return t, h, sk


tmp = tempfile.mkdtemp(prefix="a11-")
STEPS = [{"action": "search_for", "object": "WineBottle"}]
TEXT = "Find the wine bottle"


def score_fn(h):
    def _score():
        with h.scoring():
            h.meta("scoring")
            return True
    return _score


def old_run(ltm, practice, ep):
    a = A.HonestEventAdapter(ltm, episode=ep, mode="practice" if practice else "test")
    t, h, sk = build(a)
    out = live.run_episode_core(h=h, sk=sk, adapter=a, steps=STEPS, holding=None, text=TEXT, level="L1", lang="en",
                                score=score_fn(h), practice=practice, policy="console")
    return out, a, t, h, sk


# ================================================================================================ 1
print("1. flags and the old order")
ps = inspect.signature(live.LiveExecutor.__init__).parameters
check("plan_after_sweep, state_capture default OFF; state_check None", ps["plan_after_sweep"].default is False
      and ps["state_capture"].default is False and ps["state_check"].default is None)
try:
    live.LiveExecutor(state_capture=True)
    raised = False
except ValueError:
    raised = True
check("state capture / check without plan_after_sweep is refused", raised)
check("run_episode_core takes started= (default None); start_core is the only sweep call",
      inspect.signature(live.run_episode_core).parameters["started"].default is None
      and inspect.getsource(live).count("sk.start_episode(") == 1)
mA = L.LongTermMemory(os.path.join(tmp, "old.db"), kitchen="Fake")
mA.reset_world("z1")
o1, a1, t1, h1, sk1 = old_run(mA, True, "old-1")
mB = L.LongTermMemory(os.path.join(tmp, "split.db"), kitchen="Fake")
mB.reset_world("z1")
a2 = A.HonestEventAdapter(mB, episode="old-1", mode="practice")
t2, h2, sk2 = build(a2)
seed2 = live.start_core(h=h2, sk=sk2, adapter=a2, holding=None, text=TEXT, level="L1", lang="en")
o2 = live.run_episode_core(h=h2, sk=sk2, adapter=a2, steps=STEPS, holding=None, text=TEXT, level="L1", lang="en",
                           score=score_fn(h2), practice=True, policy="console", started=seed2)
check("split start (start_core, then started=) gives the same episode as the old single call: same events, steps, simulator calls",
      [e["kind"] for e in a1.stream] == [e["kind"] for e in a2.stream] and o1["steps"] == o2["steps"] and t1.calls == t2.calls
      and o1["prims"] == o2["prims"])
check("... and the sweep ran once (one episode_start, one task_context)", sum(1 for e in a2.stream if e["kind"] == "episode_start") == 1
      and sum(1 for e in a2.stream if e["kind"] == "task_context") == 1)

# ================================================================================================ 2
print("\n2. the post-sweep dict")
mC = L.LongTermMemory(os.path.join(tmp, "p.db"), kitchen="Fake")
mC.reset_world("z1")
a3 = A.HonestEventAdapter(mC, episode="new-1", mode="test")
t3, h3, sk3 = build(a3)
seed3 = live.start_core(h=h3, sk=sk3, adapter=a3, holding=None, text=TEXT, level="L1", lang="en")
post = live.post_sweep_dict(seed=seed3, adapter=a3, holding="Apple", sweep_prims=h3.step)
check("post: summary is the objmem summary right after the sweep; sha is sha256 of the lines joined by newlines",
      post["summary"] == sk3.om.summary() and len(post["summary"]) > 0
      and post["summary_sha"] == hashlib.sha256("\n".join(post["summary"]).encode()).hexdigest())
check("post: told holding, episode id, sweep primitives, events and the live WM",
      post["holding"] == "Apple" and post["episode_id"] == "new-1" and post["sweep_prims"] == h3.step > 0
      and post["events"] and post["wm"] is a3.wm)
check("post: no goal / score / oracle key reaches the hook", not any(re.search(r"goal|score|oracle|admission|truth", k) for k in post))
check("E-fix: post holds EXACTLY summary, summary_sha, holding, events, wm, episode_id, sweep_prims, seed_tracks (no oracle-derived world hash)",
      set(post) == {"summary", "summary_sha", "holding", "events", "wm", "episode_id", "sweep_prims", "seed_tracks"}
      and "world_hash" not in inspect.signature(live.post_sweep_dict).parameters)
check("the sweep stream holds no terminal / admission_verdict / step events (the planner cannot see a verdict)",
      not any(e["kind"] in ("terminal", "admission_verdict", "step_start", "step_outcome") for e in post["events"]))

# ================================================================================================ 3
print("\n3. render_blocks")
oP, aP, tP, hP, skP = old_run(mC, True, "prac-1")                # a practice episode teaches the LTM where the bottle was
check("practice consolidates", mC.end_episode(aP.wm, admission=oP["goal_ok"])["status"] == "committed")
aT = A.HonestEventAdapter(mC, episode="test-1", mode="test")
tT, hT, skT = build(aT)
seedT = live.start_core(h=hT, sk=skT, adapter=aT, holding=None, text=TEXT, level="L1", lang="en")
postT = live.post_sweep_dict(seed=seedT, adapter=aT, holding=None, sweep_prims=hT.step)
n_before = len(aT.wm.events)
live_blk = mC.memory_block(aT.wm, "L1", "en", channels=("places",), holding=None)
rb = live.render_blocks(postT, {"live": mC, "again": mC}, "L1", "en", ("places",), holding=None)
check("the LTM learned something (the block is not empty, so the equality below is not vacuous)", bool(live_blk["text"]), repr(live_blk)[:200])
check("render_blocks(live LTM) == ltm.memory_block(post['wm'], ...) (text and record ids)",
      rb["live"]["text"] == live_blk["text"] and [r["id"] for r in rb["live"]["records"]] == [r["id"] for r in live_blk["records"]])
check("... deterministic across sources and calls; the live WM was not touched",
      rb["live"]["text"] == rb["again"]["text"] and len(aT.wm.events) == n_before and aT.wm.episode == "test-1")
ad_none = A.HonestEventAdapter(None, episode="test-1", mode="test", kitchen="Fake")
post_none = dict(postT, wm=ad_none.wm)
check("... works when the executor ran with memory='none' (no LTM): the event stream alone feeds the block",
      live.render_blocks(post_none, {"s": mC}, "L1", "en", ("places",))["s"]["text"] == live_blk["text"])

# ================================================================================================ 4
print("\n4. an unexecuted plan closes the started attempt")


def closed(practice, ltm, ep):
    a = A.HonestEventAdapter(ltm, episode=ep, mode="practice" if practice else "test")
    t, h, sk = build(a)
    live.start_core(h=h, sk=sk, adapter=a, holding=None, text=TEXT, level="L1", lang="en")
    calls0 = len(t.calls)
    out = live.close_unexecuted(h=h, sk=sk, adapter=a, practice=practice)
    return out, a, t, calls0, h


mD = L.LongTermMemory(os.path.join(tmp, "rej.db"), kitchen="Fake")
mD.reset_world("z1")
out, a4, t4, c0, h4 = closed(True, mD, "rej-p")
kinds = [e["kind"] for e in a4.stream]
check("practice: terminal(exec_ok False), admission_verdict False, episode_end, in that order",
      kinds[-3:] == ["terminal", "admission_verdict", "episode_end"]
      and a4.wm.terminal["exec_ok"] is False and a4.wm.admission["goal_ok"] is False)
check("no step events and no simulator call after the sweep (no scoring call)",
      not any(k in ("step_start", "step_outcome") for k in kinds) and len(t4.calls) == c0)
check("the audit is clean and the attempt counts its sweep primitives", out["audit"]["counts"].get("oracle", 0) == 0 and out["prims"] == h4.step > 0)
res_c = mD.end_episode(a4.wm, admission=False)
check("practice consolidation of the unexecuted attempt runs (a started, failed attempt)", isinstance(res_c, dict) and "status" in res_c, repr(res_c)[:120])
out, a5, t5, c5, h5 = closed(False, mD, "rej-t")
k5 = [e["kind"] for e in a5.stream]
check("test: terminal then episode_end, NO admission verdict anywhere", k5[-2:] == ["terminal", "episode_end"] and "admission_verdict" not in k5)
a6 = A.HonestEventAdapter(mD, episode="rej-bad", mode="test")
t6, h6, sk6 = build(a6)
live.start_core(h=h6, sk=sk6, adapter=a6, holding=None, text=TEXT, level="L1", lang="en")
h6.audit.invalid = True
try:
    live.close_unexecuted(h=h6, sk=sk6, adapter=a6, practice=False)
    failed_fast = False
except live.LiveInvariant:
    failed_fast = True
check("an invalid honesty audit still fails fast on the unexecuted path", failed_fast)

# ================================================================================================ 5
print("\n5. source order and honesty")
src = inspect.getsource(live.LiveExecutor.execute)
i_start, i_hook = src.index("start_core("), src.index("plan_hook(req, it, holding, ltm, post=post)")
i_old = src.index("plan_hook(req, it, holding, ltm)")
i_state = src.index("self._score_post_sweep_state(")
check("new order: start_core -> state capture / check -> plan hook (post=) -> run_episode_core(started=)",
      i_start < i_state < i_hook < src.index("run_episode_core(") and "started=started" in src)
check("old order: the un-flagged hook call stays before the agent is built and only runs when the flag is off",
      i_old < src.index("HonestAgent(") and "not self.plan_after_sweep" in src[:i_old])
check("state is never an argument of the hook", "post_sweep_state" not in src[i_hook - 40:i_hook + 80])
tree = ast.parse(inspect.getsource(live))
bad = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and not f.name.startswith("_score")
       and any(isinstance(n, ast.Attribute) and n.attr == "scoring" for n in ast.walk(f))]
check("`.scoring()` appears only inside _score* functions", not bad, str(bad))
forbid = re.compile(r"\.metadata\b|last_event|state_readers|\.controller\b")
check("the new helpers name no controller / metadata / last_event", not any(
    forbid.search(inspect.getsource(f)) for f in (live.start_core, live.close_unexecuted, live.post_sweep_dict, live.render_blocks)))

# ================================================================================================ 6
print("\n6. c3_run PlanHook under both orders")
try:
    import c3_run                                              # noqa: E402
    ok_import, err = True, ""
except Exception as e:                                         # pragma: no cover
    ok_import, err = False, repr(e)
check("c3_run imports offline", ok_import, err)
if ok_import:
    class FakeContract:
        def messages(self, command, summary, holding, memory_block=None, fewshot_block=None):
            return [{"role": "user", "content": command + "|" + "\n".join(summary) + "|" + str(memory_block)}]

    class FakeRunner:
        def __init__(self):
            self.calls = []

        def plan(self, command, summary=(), holding=None, memory_block=None, fewshot_block=None, replicate=0):
            self.calls.append((list(summary), memory_block))
            return {"ok": True, "first_ok": True, "repaired": False, "rejected": False, "infra_failure": False, "calls": 1,
                    "tokens_in": 1, "tokens_out": 1, "plan": [{"action": "pickup", "object": "WineBottle"}],
                    "first_plan": [], "say": "", "normalised": [], "attempts": []}

    class FakeCounter:
        header_cost = 0

        def counts(self, t):
            return {"g": len(t)}

    runner = FakeRunner()
    hook = c3_run.PlanHook(runner, FakeContract(), FakeCounter())
    it = {"id": "x", "level": "L1", "en": TEXT, "fa": TEXT}
    req = types.SimpleNamespace(arm="P", lang="en", episode_id="test-1", mode="test", world="reset", kitchen="Fake", cell="c")
    pre = hook(req, it, None, mC)
    post_ = hook(req, it, None, mC, post=postT)
    check("post=None: empty summary (the pre-A11 prompt), labelled", runner.calls[0][0] == [] and pre["info"]["plan_order"] == "pre_A11"
          and pre["info"]["summary_n"] == 0)
    check("post=dict: the post-sweep summary is in the prompt; label after_sweep; summary sha recorded",
          runner.calls[1][0] == postT["summary"] and post_["info"]["plan_order"] == "after_sweep"
          and post_["info"]["summary_sha"] == postT["summary_sha"])
    check("post=dict: the block is the facade's block of the post-sweep WM (one renderer)", runner.calls[1][1] == (live_blk["text"] or None))
    msrc = inspect.getsource(c3_run.main)
    check("c3_run: --plan-after-sweep defaults ON; --no-plan-after-sweep reproduces pre-A11",
          "--plan-after-sweep" in msrc and "--no-plan-after-sweep" in msrc and "default=True" in msrc)

# ================================================================================================ 7
print("\n7. E-fix: the state capture (labels = evaluator readings, world hash = diagnostic), explicit unexecuted reasons")
from final_test import reset_gate as RG                    # noqa: E402  (pure stdlib)


class FakeAg:
    """The oracle side of the executor: only objects() and agent_pose() are read by the capture."""
    def __init__(self, shift=0.0, suffix="a"):
        self.shift, self.suffix = shift, suffix

    def objects(self):
        return [{"name": "Box_1", "objectType": "Box", "objectId": f"Box|1|{self.suffix}", "position": {"x": 1.0, "y": 0.0, "z": 1.0},
                 "rotation": {"x": 0, "y": 0, "z": 0}, "openable": True, "isOpen": False},
                {"name": "Cup_1", "objectType": "Cup", "objectId": f"Cup|1|{self.suffix}", "position": {"x": 2.0 + self.shift, "y": 1.0, "z": 1.0},
                 "rotation": {"x": 0, "y": 10, "z": 0}, "pickupable": True}]

    def agent_pose(self):
        return {"position": {"x": 0.5, "y": 0.9, "z": 0.25}, "rotation": {"x": 0, "y": 90, "z": 0}, "horizon": 0, "standing": True}


aS = A.HonestEventAdapter(None, episode="cap-1", mode="test", kitchen="Fake")
tS, hS, skS = build(aS)
seedS = live.start_core(h=hS, sk=skS, adapter=aS, holding=None, text=TEXT, level="L1", lang="en")
postS = live.post_sweep_dict(seed=seedS, adapter=aS, holding=None, sweep_prims=hS.step)
exS = live.LiveExecutor(plan_after_sweep=True, state_capture=True)
LAB = {"goal_clauses": {"Cup in Box": False, "any Box.isOpen = true": False}, "goal_true_at_reset": False}
itS = {"id": "z1", "level": "L1", "plan": [{"action": "pickup", "object": "Cup"}, {"action": "put_in", "object": "Cup", "to": "Box"}]}
rq = lambda arm, lang: types.SimpleNamespace(arm=arm, lang=lang, cell="c")                            # noqa: E731
exS._ag = FakeAg()
stA = exS._score_post_sweep_state(hS, rq("G0", "en"), itS, skS, postS, LAB, world_hash="h1")
exS._ag = FakeAg(suffix="b")
stB = exS._score_post_sweep_state(hS, rq("PF", "fa"), dict(itS, id="z1"), skS, postS, LAB, world_hash="h2")
TOL = {"tau_pos_m": 0.05, "tau_rot_deg": 5.0}
check("the capture's labels are the evaluator readings only (no item / level / lang / arm / world hash in them)",
      stA["labels"] == LAB and not ({"item", "level", "lang", "arm", "world_hash"} & set(stA["labels"])))
check("the strict world hash is stored as a diagnostic, never compared", stA["diagnostics"]["world_hash"] == "h1" and stB["diagnostics"]["world_hash"] == "h2")
check("two captures of the same world differing ONLY in arm, language and world hash compare ok (Phase A vs Phase C)", RG.compare(stA, stB, TOL)["ok"],
      str(RG.compare(stA, stB, TOL)["breaches"])[:200])
exS._ag = FakeAg(suffix="b")
flip = {"goal_clauses": {"Cup in Box": True, "any Box.isOpen = true": False}, "goal_true_at_reset": False}
stC = exS._score_post_sweep_state(hS, rq("G0", "en"), itS, skS, postS, flip, world_hash="h1")
check("a flipped goal-clause label breaches (A14 c iii is really checked)", [b["kind"] for b in RG.compare(stA, stC, TOL)["breaches"]] == ["labels"])
exS._ag = FakeAg(shift=0.3, suffix="b")
stD = exS._score_post_sweep_state(hS, rq("G0", "en"), itS, skS, postS, LAB, world_hash="h1")
check("a displaced object still breaches the continuous part", [b["kind"] for b in RG.compare(stA, stD, TOL)["breaches"]] == ["continuous.pos"])
check("the labels are carried by execute() from the setup side (before the agent exists), the capture gets them as an argument",
      "reset_labels = self._score_reset_labels(" in inspect.getsource(live.LiveExecutor.execute)
      and "post, reset_labels" in inspect.getsource(live.LiveExecutor.execute))


class FakeEv:
    """step4_refplans._unmet reads ev.state_readers / ev.sr_score; here each atom is a plain truth value."""
    @staticmethod
    def state_readers(ag):
        return None, None

    @staticmethod
    def sr_score(goal, gs_, ga_):
        return all(a["truth"] for a in goal)


class FakeGs:
    @staticmethod
    def format_goal(a):
        return a["text"]


try:
    lab = live.LiveExecutor()._score_reset_labels(FakeEv, FakeGs, [{"text": "Cup in Box", "truth": False}, {"text": "Box open", "truth": True}])
    check("_score_reset_labels: per-clause truth keyed by clause text, plus the reset verdict",
          lab == {"goal_clauses": {"Cup in Box": False, "Box open": True}, "goal_true_at_reset": False}, str(lab))
except Exception as e:                                          # pragma: no cover
    check("_score_reset_labels: per-clause truth keyed by clause text", False, repr(e))

# ---- explicit reasons; a blocked execution is missing data, not a failed task
se = importlib.import_module("step5_eval")


def unexec(reason, practice, info=None):
    mem = L.LongTermMemory(os.path.join(tmp, f"ue-{reason}-{practice}.db"), kitchen="Fake")
    mem.reset_world("z1")
    a = A.HonestEventAdapter(mem, episode=f"ue-{reason}-{int(practice)}", mode="practice" if practice else "test")
    t, h, sk = build(a)
    live.start_core(h=h, sk=sk, adapter=a, holding=None, text=TEXT, level="L1", lang="en")
    res = se.EpisodeResult()
    out = live.LiveExecutor(plan_after_sweep=True)._unexecuted(res, h, sk, a, None, practice, True, {"plan_order": "after_sweep"}, info, reason=reason)
    return out, a


rj, arj = unexec("rejected", True, {"ok": False})
check("rejected: sr 0, goal_ok False, the started attempt goes to practice consolidation with admission False",
      rj.sr == 0.0 and rj.goal_ok is False and rj.wm is arj.wm and rj.admission is False and rj.extra["plan_not_executed_reason"] == "rejected")
bl, abl = unexec("blocked_state", True)
check("blocked_state: sr and goal_ok are None (missing, not a failed task), status infra_blocked, reason blocked_state",
      bl.sr is None and bl.goal_ok is None and bl.extra["status"] == "infra_blocked" and bl.extra["plan_not_executed_reason"] == "blocked_state")
check("blocked_state: nothing is consolidated (no wm, no admission) and no admission verdict was emitted",
      bl.wm is None and bl.admission is None and "admission_verdict" not in [e["kind"] for e in abl.stream])
blt, ablt = unexec("blocked_state", False)
check("blocked_state in test mode: also missing data", blt.sr is None and blt.extra["status"] == "infra_blocked")
try:
    unexec("whatever", True)
    unknown_refused = False
except ValueError:
    unknown_refused = True
check("... for real: _unexecuted refuses an unknown reason", unknown_refused)
nr = live.not_executed_reason
check("hook reasons are explicit: not_executed wins; infra_failure -> infra; None / {} / plain rejection -> rejected (never blocked_state)",
      nr({"not_executed": "reuse"}) == "reuse" and nr({"infra_failure": True}) == "infra" and nr(None) == "rejected" and nr({}) == "rejected"
      and nr({"not_executed": "blocked_state"}) == "rejected" and nr({"ok": False}) == "rejected")
esrc = inspect.getsource(live.LiveExecutor.execute)
check("execute passes reason='blocked_state' only from the state-check branch and not_executed_reason(plan_info) from the hook branch",
      esrc.count('reason="blocked_state"') == 1 and "reason=not_executed_reason(plan_info)" in esrc and "plan_info is None" not in inspect.getsource(live.LiveExecutor._unexecuted))

# ---- render_blocks on a FROZEN source needs its test-epoch overlay
snap = os.path.join(tmp, "snap")
mC.snapshot(snap, overwrite=True)
fz = L.LongTermMemory.open_frozen(snap, os.path.join(tmp, "fz.testlog.db"), kitchen="Fake")
try:
    live.render_blocks(postT, {"fz": fz}, "L1", "en", ("places",))
    refused = False
except live.LiveInvariant:
    refused = True
check("render_blocks refuses a frozen source without its test-epoch overlay (the wrong epoch would read practice claims as current)", refused)
fz.begin_test_epoch(frame="f0", map_rev="m0")
rbz = live.render_blocks(postT, {"fz": fz}, "L1", "en", ("places",))
check("... with the overlay it equals the facade's block of the post-sweep WM on the same frozen copy (the executor path)",
      rbz["fz"]["text"] == fz.memory_block(postT["wm"], "L1", "en", channels=("places",))["text"] and bool(rbz["fz"]["text"]))
fz.close()

# ---- gate 12.2 as a TWIN comparison (EN presentation vs FA presentation)
a_en = A.HonestEventAdapter(mC, episode="tw-en", mode="test")
a_en.on_task(TEXT, "L1", "en")
a_fa = A.HonestEventAdapter(mC, episode="tw-fa", mode="test")
a_fa.on_task("\u06a9\u0644\u0645\u0647 \u0646\u0627\u0645\u0639\u0644\u0648\u0645 \u0631\u0627 \u067e\u06cc\u062f\u0627 \u06a9\u0646", "L1", "fa")        # an instruction the lexicon cannot parse
tw_same = live.twin_check(mC, a_en.wm, a_en.wm, "L1", ("places",))
tw_diff = live.twin_check(mC, a_en.wm, a_fa.wm, "L1", ("places",))
check("twin_check: identical presentations agree; the two languages' OWN parses are what is compared",
      tw_same["same_ids"] and not tw_same["empty_en"] and set(tw_same) >= {"same_ids", "ids_en", "ids_fa", "empty_en", "empty_fa", "same_dropped"})
check("twin_check: a presentation the FA lexicon cannot parse gives a different (empty) block and is reported, which bilingual_check can never show",
      (not tw_diff["same_ids"]) and tw_diff["empty_fa"] and not tw_diff["empty_en"], str(tw_diff))

# ================================================================================================ 8
print("\n8. E-fix: c3 practice consolidates a rejected plan after the sweep (A11), not a blocked one")
import argparse                                              # noqa: E402
if ok_import:
    class FakeCounterP:
        tokenizers = ["gemma", "anthropic"]
        header_cost = 0

        def save(self):
            pass

    def run_fake_practice(plan_after_sweep, reason, status=None, tag="x"):
        work = os.path.join(tmp, f"pw-{tag}")
        out = os.path.join(tmp, f"po-{tag}")
        os.makedirs(work, exist_ok=True)
        os.makedirs(out, exist_ok=True)

        class FakeEx:
            frame, map_rev = "f0", "m0"

            def __init__(self):
                self.plan_after_sweep = plan_after_sweep

            def execute(self, req):
                a = A.HonestEventAdapter(req.memory, episode=req.episode_id, mode="practice")
                t, h, sk = build(a)
                live.start_core(h=h, sk=sk, adapter=a, holding=None, text=TEXT, level="L1", lang="en")
                res = se.EpisodeResult()
                ex_ = live.LiveExecutor(plan_after_sweep=True)
                res = ex_._unexecuted(res, h, sk, a, None, True, True, {"plan_order": "after_sweep"}, {"ok": False, "not_executed": reason}, reason=reason)
                if not plan_after_sweep:                    # pre-A11 shape: nothing was swept, nothing to consolidate
                    res.wm = res.admission = None
                return res

            def close(self):
                pass

        items_ = {"x": {"id": "x", "level": "L1", "en": TEXT, "fa": TEXT}}
        split_ = {"practice_order0": [{"item": "x", "lang": "en", "family": "f"}]}
        orig = c3_run.build
        c3_run.build = lambda args: (FakeEx(), None, FakeCounterP(), items_, split_, {"phase": "practice"})
        try:
            c3_run.run_practice(argparse.Namespace(work=work, out=out, limit=0, snapshot=None, plan_after_sweep=plan_after_sweep))
        finally:
            c3_run.build = orig
        rows = [__import__("json").loads(x) for x in open(os.path.join(out, "practice.jsonl"), encoding="utf-8")]
        mem = L.LongTermMemory(os.path.join(work, "ltm.practice.db"), kitchen="Fake" if False else c3_run.SCENE)
        n_rec = len([r for r in mem.store.select("commit_receipt") if r.get("advance_clock") and str(r.get("episode", "")).startswith("ep:")])
        mem.close()
        return rows[0], n_rec

    row_rej, n_rej = run_fake_practice(True, "rejected", tag="rej")
    check("c3 practice: a rejected plan after the sweep is a started attempt: consolidated with admission False, recorded in the row",
          row_rej.get("consolidated_unexecuted") is True and row_rej["plan_not_executed_reason"] == "rejected" and n_rej == 1
          and (row_rej.get("consolidation") or {}).get("status") == "committed", str(row_rej.get("consolidation"))[:200] + f" n={n_rej}")
    row_inf, n_inf = run_fake_practice(True, "infra", tag="inf")
    check("c3 practice: an infra-failed plan after the sweep is consolidated the same way", row_inf.get("consolidated_unexecuted") is True and n_inf == 1)
    row_reu, n_reu = run_fake_practice(True, "reuse", tag="reu")
    check("c3 practice: a 'reuse' sentinel is not a failed attempt: nothing consolidated", not row_reu.get("consolidated_unexecuted") and n_reu == 0)
    row_pre, n_pre = run_fake_practice(False, "rejected", tag="pre")
    check("c3 practice: pre-A11 (flag off) keeps the old rule: nothing consolidated", not row_pre.get("consolidated_unexecuted") and n_pre == 0)
    try:
        run_fake_practice(True, "blocked_state", tag="blk")
        blocked_raises = False
    except live.LiveInvariant:
        blocked_raises = True
    check("c3 practice: a blocked execution (infra_blocked) is never journaled or consolidated: it aborts, resumable", blocked_raises)
    check("c3 hook states why it did not execute (explicit not_executed: reuse / rejected)",
          '"not_executed"] = "reuse"' in inspect.getsource(c3_run.PlanHook.__call__) and '"not_executed"] = "rejected"' in inspect.getsource(c3_run.PlanHook.__call__))

print(f"\nstep5 plan-timing (A11) selftest: {PASS}/{PASS + FAIL}")
sys.exit(0 if FAIL == 0 else 1)
