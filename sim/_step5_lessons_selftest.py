"""Self-test for step 5 / A3-diagnosis -- cause taxonomy, hypothesis interface, lessons, procedures, FLR bookkeeping.

Offline, Windows `python` from the REPO ROOT, temp directories only (never sim/trace.db, sim/memory.db or the default
sim/step5/ltm.db), no simulator, no WSL, no Ollama, no network (Gemma is a mock):

    python sim/_step5_lessons_selftest.py

Covers (docs/step5-plan.md sections 2 R7, 7.1-7.6, 11): the observable cause taxonomy and the local failure key; the
evidence packet; the hypothesis validator (evidence refs, cause, action, prose hygiene; rejected = no lesson; abstain =
default); lesson creation / recurrence-once-per-episode; application eligibility + scoring precedence; units, promotion,
demotion, retirement, revalidation; test-mode restrictions; procedure admission (knife-less success refused), reuse,
failed reuse, trust / demotion / retirement / reactivation; privileged-verdict confinement; atomic commit / rollback /
replay; false-lesson-rate bookkeeping; import hygiene.
"""
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                    # append, not insert: sim/trace.py must not shadow the stdlib `trace`

import ltm                                # noqa: E402
import ltm_lessons as ll                  # noqa: E402
from ltm_events import EventError         # noqa: E402
from ltm_store import canon               # noqa: E402
from ltm_wm import WorkingMemory          # noqa: E402

REPO = Path(HERE).parent
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


# ------------------------------------------------------------------------------------------------ fixtures
def roles(n, verb="find"):
    return {"roles": [{"verb": verb, "otype": f"T{n}", "to": "?", "state": "?", "neg": False}]}


class Ep:
    """Event builder for one episode."""

    def __init__(self, episode, kitchen="FloorPlan1", epoch=0, mode="practice", concepts=None, text="find the apple"):
        self.base = dict(schema="s5.ev/1", kitchen=kitchen, epoch=epoch, frame="f0", map_rev="m0", episode=episode)
        self.n = 0
        self.events = []
        self.add("episode_start", mode=mode, world="reset")
        kw = dict(level="L1", lang="en", text=text)
        if concepts is not None:
            kw["concepts"] = concepts
        self.add("task_context", **kw)

    def add(self, kind, **kw):
        self.n += 1
        self.events.append(dict(self.base, ev=self.n, kind=kind, t=self.n, **kw))
        return self.n

    def step(self, i, verb="find", otype="Apple", to="Fridge", pre=None):
        kw = dict(i=i, verb=verb, otype=otype, to=to)
        if pre is not None:
            kw["pre"] = pre
        return self.add("step_start", **kw)

    def open_(self, rtype="Fridge", state="closed"):
        return self.add("open", rtype=rtype, xyz=[1.0, 0.5, 2.0], ok=True, by_me=True, state_seen=state)

    def attempt(self, i, k, ok, cause=None, recovery=None, source=None):
        kw = dict(i=i, k=k, ok=ok)
        if cause:
            kw["cause"] = cause
        if recovery:
            kw["recovery"] = recovery
        if source:
            kw["source"] = source
        return self.add("attempt", **kw)

    def refusal(self, i, k, cause, rtype="Fridge", verb="open", otype="Apple"):
        return self.add("refusal", verb=verb, otype=otype, rtype=rtype, i=i, k=k, cause=cause)

    def outcome(self, i, ok, cause=None, under=None, retries=0):
        kw = dict(i=i, ok=ok, retries=retries, prims=3)
        if cause:
            kw["cause"] = cause
        if under is not None:
            kw["under"] = under
        return self.add("step_outcome", **kw)

    def app(self, record, i, k, pre=None, retrieval=None):
        kw = dict(app=f"a{self.n + 1}", record=record, i=i, k=k)
        if pre is not None:
            kw["pre"] = pre
        if retrieval:
            kw["retrieval"] = retrieval
        return self.add("application", **kw)

    def sight(self, n):
        return self.add("sighting", otype="Apple", track=f"tr:{n}", conf=0.9, xyz=[1, 1, 1], xyz_ok=True,
                        view=f"vw:{n}", obs=f"ob:{n}", in_rtype="Fridge", in_xyz=[1, 1, 1], in_basis="box_only",
                        detector="rev1")

    def finish(self, exec_ok=True, goal=None, invalid=False, verdict="success"):
        self.add("terminal", exec_ok=exec_ok, agent_verdict=verdict, prims=5, opens=1, walk_m=3.0, invalid=invalid)
        if goal is not None:
            self.add("admission_verdict", goal_ok=goal, tag="privileged-terminal-goal-check")
        self.add("episode_end")
        return self

    def wm(self):
        w = WorkingMemory(self.base["episode"])
        for e in self.events:
            w.ingest(e)
        return w


_OPEN = []


def new_ltm(tmp, name, P=None, **kw):
    L = ltm.LongTermMemory(Path(tmp) / name / "ltm.db", kitchen="FloorPlan1", params=P or ltm.load_params(), **kw)
    _OPEN.append(L)
    return L


def run(L, ep, ext=None):
    ext = ext or ll.LessonsExtension()
    return L.end_episode(ep.wm(), extensions=[ext]), ext


def rows(L, table, **where):
    return L.store.select(table, where=where or None)


def fail_then_sidestep(name, rl, k1="sidestep", goal=True, **kw):
    """The standard origin: key (not_detected, find, Apple, Fridge, closed); step ok after one recovery."""
    e = Ep(name, concepts=rl, **kw)
    e.step(0, pre={"holding": "n"})
    e.open_("Fridge", "closed")
    e.attempt(0, 0, False, "not_detected")
    e.attempt(0, 1, True, recovery=k1)
    e.outcome(0, True, retries=1)
    return e.finish(exec_ok=True, goal=goal)


def apply_ep(name, rl, lid, k1_ok=True, k1_cause="not_detected", source=None, k=1, goal=True, extra_fail=False,
             epoch=0, retrieval=None, verdict="success", drop_outcome=False, later_ok=False, second_remedy=None):
    e = Ep(name, concepts=rl, epoch=epoch)
    e.step(0, pre={"holding": "n"})
    e.open_("Fridge", "closed")
    e.attempt(0, 0, False, "not_detected")
    e.app(lid, 0, k, retrieval=retrieval)
    if second_remedy:
        e.app(second_remedy, 0, k)
    e.attempt(0, k, k1_ok, None if k1_ok else k1_cause, recovery="lesson", source=source or f"lesson:{lid}")
    if not k1_ok and later_ok:
        e.attempt(0, k + 1, True, recovery="step_back")
        e.outcome(0, True, retries=k + 1)
    elif not drop_outcome:
        e.outcome(0, k1_ok, None if k1_ok else k1_cause, retries=k)
    if extra_fail:
        e.step(1, verb="open", otype="Drawer", to="?")
        e.attempt(1, 0, False, "action_refused")
        e.outcome(1, False, "action_refused")
        return e.finish(exec_ok=False, goal=False, verdict=verdict)
    return e.finish(exec_ok=(k1_ok or later_ok) and not drop_outcome, goal=goal, verdict=verdict)


def evs(L, claim_id):
    """Decoded evidence rows of one claim, oldest first."""
    out = []
    for r in rows(L, "evidence"):
        if r["claim"].endswith("/" + claim_id):
            r = dict(r)
            r["summary"] = json.loads(r["summary"])
            out.append(r)
    return out


def llogs(L):
    return [r for r in rows(L, "application_log") if r["kind"] == "lesson"]


def the_lesson(L):
    r = rows(L, "lesson")
    return r[0] if r else None


def mock(reply):
    """A Gemma stand-in. `reply` is a dict, a str (raw model text), an exception, or a callable(packet)."""
    class M(ll.Hypothesizer):
        calls = 0
        packets = []

        def propose(self, packet):
            M.calls += 1
            M.packets.append(packet)
            r = reply(packet) if callable(reply) else reply
            if isinstance(r, Exception):
                raise r
            return r
    return M()


# ================================================================================================= 1 taxonomy + key
def test_taxonomy(tmp):
    check("raw map: state_unrecognised -> postcondition_unconfirmed", ll.normalise_cause("state_unrecognised") == "postcondition_unconfirmed")
    check("raw map: retry_exhausted / primitive_cap -> budget_exhausted",
          ll.normalise_cause("retry_exhausted") == ll.normalise_cause("primitive_cap") == "budget_exhausted")
    check("raw map: the four direct codes", all(ll.normalise_cause(c) == c for c in ("not_detected", "action_refused", "unreachable", "wrong_target")))
    check("raw map: unrecognised / None / non-str -> unknown", ll.normalise_cause("kaboom") == ll.normalise_cause(None) == ll.normalise_cause(7) == "unknown")
    check("plan_goal_unmet is not a raw code (only the privileged label)", ll.normalise_cause("plan_goal_unmet") == "unknown")
    check("every taxonomy code has an allowed-action entry", set(ll.ALLOWED_ACTIONS) == set(ll.CAUSES))
    P = ltm.load_params()
    da = P["lessons"]["default_action"]["value"]
    check("registry default actions are allowed for their cause (or none)", all(a == "none" or a in ll.ALLOWED_ACTIONS[c] for c, a in da.items()))
    check("no remedy exists for budget_exhausted / plan_goal_unmet / unknown", not any(ll.ALLOWED_ACTIONS[c] for c in ("budget_exhausted", "plan_goal_unmet", "unknown")))

    # the key is built from events at or before the failed attempt - never from later recovery
    e = Ep("t1")
    e.step(0, pre={"holding": "n"})
    e.attempt(0, 0, False, "not_detected")          # fails BEFORE the container is opened
    e.open_("Fridge", "open")                       # later recovery observation must not change the earlier key
    e.attempt(0, 1, True, recovery="sidestep")
    e.outcome(0, True)
    e.finish()
    d = ll.diagnose(e.wm())
    f = d["failures"][0]
    check("one failed attempt -> one failure", len(d["failures"]) == 1 and f["k"] == 0)
    check("key uses only earlier events: rtype falls back to the step target, state unknown",
          f["conditions"] == {"cause": "not_detected", "verb": "find", "otype": "Apple", "rtype": "Fridge", "state": "?"}, str(f["conditions"]))
    e2 = fail_then_sidestep("t2", roles(0))
    f2 = ll.diagnose(e2.wm())["failures"][0]
    check("with a prior open the state-seen enters the key", f2["conditions"]["state"] == "closed" and f2["conditions"]["rtype"] == "Fridge")
    check("cond_key is a stable function of the conditions", f2["cond_key"] == ll.cond_key_of(f2["conditions"]) and f2["cond_key"].startswith("ck:"))
    check("failure_key_at (adapter view) equals the diagnosed key", ll.failure_key_at(e2.wm(), 0) == f2["conditions"])
    check("failure_key_at: no failure yet -> None", ll.failure_key_at(Ep("t2b").wm(), 0) is None)

    # precedence: attempt cause > refusal cause > step_outcome cause
    e = Ep("t3"); e.step(0); e.refusal(0, 0, "unreachable"); e.attempt(0, 0, False, "wrong_target"); e.outcome(0, False, "retry_exhausted"); e.finish(False)
    d = ll.diagnose(e.wm())
    kinds = [(x["cause"], x["step_level"]) for x in d["failures"]]
    check("attempt cause wins over refusal and outcome", ("wrong_target", False) in kinds, str(kinds))
    check("budget exhaustion is kept as a SEPARATE step-level failure", ("budget_exhausted", True) in kinds, str(kinds))
    e = Ep("t4"); e.step(0); e.refusal(0, 0, "unreachable"); e.attempt(0, 0, False); e.outcome(0, False); e.finish(False)
    check("refusal cause used when the attempt has none", ll.diagnose(e.wm())["failures"][0]["cause"] == "unreachable")
    e = Ep("t5"); e.step(0); e.attempt(0, 0, False); e.outcome(0, False, "state_unrecognised"); e.finish(False)
    check("last attempt falls back to the step outcome cause", ll.diagnose(e.wm())["failures"][0]["cause"] == "postcondition_unconfirmed")
    e = Ep("t6"); e.step(0); e.outcome(0, False, "not_detected"); e.finish(False)
    d = ll.diagnose(e.wm())
    check("a failed step with no attempt events is still diagnosed (outcome only)", len(d["failures"]) == 1 and d["failures"][0]["step_level"] and d["failures"][0]["cause"] == "not_detected")
    e = Ep("t7"); e.step(0); e.attempt(0, 0, False, "weird_code"); e.outcome(0, False); e.finish(False)
    check("unrecognised code -> unknown", ll.diagnose(e.wm())["failures"][0]["cause"] == "unknown")
    e = Ep("t8"); e.step(0); e.attempt(0, 0, None, "not_detected"); e.outcome(0, True); e.finish()
    check("an attempt with unknown ok is not a failure", ll.diagnose(e.wm())["failures"] == [])

    # plan_goal_unmet: only when every step is ok AND the privileged verdict is exactly False
    e = Ep("t9"); e.step(0); e.outcome(0, True); e.finish(True, goal=False)
    check("plan_goal_unmet: all ok + goal False", ll.diagnose(e.wm(), False)["plan_goal_unmet"])
    check("plan_goal_unmet: not when the verdict is None", not ll.diagnose(e.wm(), None)["plan_goal_unmet"])
    check("plan_goal_unmet: not when goal True", not ll.diagnose(e.wm(), True)["plan_goal_unmet"])
    e = Ep("t10"); e.step(0); e.finish(True, goal=False)
    check("plan_goal_unmet: a step with no outcome is not 'every step ok'", not ll.diagnose(e.wm(), False)["plan_goal_unmet"])
    e = Ep("t11"); e.finish(True, goal=False)
    check("plan_goal_unmet: an empty plan is not 'every step ok'", not ll.diagnose(e.wm(), False)["plan_goal_unmet"])
    e = Ep("t12"); e.step(0); e.outcome(0, True); e.step(1, verb="open"); e.outcome(1, False, "action_refused"); e.finish(False, goal=False)
    check("plan_goal_unmet: not when a step failed", not ll.diagnose(e.wm(), False)["plan_goal_unmet"])


# ================================================================================================= 2 packet + hypothesis
def test_hypothesis(tmp):
    P = ltm.load_params()
    e = fail_then_sidestep("h1", roles(0))
    wm = e.wm()
    d = ll.diagnose(wm)
    f = d["failures"][0]
    pk = ll.build_packet(wm, f, d["steps"])
    ids = ll.packet_ids(pk)
    check("packet: numbered events up to the failed attempt only", max(ids) == f["ev"])
    check("packet: contains the failed attempt and the step start", f["ev"] in ids and any(x["kind"] == "step_start" for x in pk["events"]))
    check("packet: excludes the later recovery attempt", all(x["kind"] != "attempt" or x["k"] == 0 for x in pk["events"]))
    check("packet: carries no task text / no concepts / no xyz", "find the apple" not in canon(pk) and "xyz" not in canon(pk) and "concepts" not in canon(pk))
    check("packet: no simulator-id pattern or privileged key", "objectId" not in canon(pk) and "|" not in canon(pk))
    check("packet schema + cause", pk["schema"] == "s5.dx/1" and pk["cause"] == "not_detected" and pk["truncated"] is False)

    # truncation: cap 40, every non-sighting event kept, most recent sightings fill, deterministic
    e = Ep("h2", concepts=roles(0)); e.step(0)
    for n in range(80):
        e.sight(n)
    e.open_("Fridge", "closed"); e.attempt(0, 0, False, "not_detected"); e.outcome(0, False, "not_detected"); e.finish(False)
    wm2 = e.wm()
    d2 = ll.diagnose(wm2)
    p1 = ll.build_packet(wm2, d2["failures"][0], d2["steps"])
    p2 = ll.build_packet(wm2, d2["failures"][0], d2["steps"])
    check("packet capped at 40 events + marked truncated", len(p1["events"]) == ll.PACKET_MAX_EVENTS and p1["truncated"])
    check("truncation is deterministic", canon(p1) == canon(p2))
    kinds = [x["kind"] for x in p1["events"]]
    check("truncation keeps the step start, the open and the failed attempt", "step_start" in kinds and "open" in kinds and "attempt" in kinds)
    sights = [x["ev"] for x in p1["events"] if x["kind"] == "sighting"]
    check("truncation keeps the MOST RECENT sightings", sights == sorted(sights) and min(sights) > 40)

    # validator
    good = {"cause": "not_detected", "hypothesis": "The fridge was closed so the apple could not be seen.",
            "evidence": [f["ev"]], "action": "open_first"}
    v = ll.validate_hypothesis(good, pk, P)
    check("valid hypothesis accepted", v["status"] == "accepted" and v["action"] == "open_first" and v["evidence"] == [f["ev"]])
    check("abstention (insufficient_evidence only) is not a rejection", ll.validate_hypothesis({"insufficient_evidence": True}, pk, P)["status"] == "abstain")
    check("abstention + extra keys is rejected", ll.validate_hypothesis({"insufficient_evidence": True, "action": "none"}, pk, P)["reason"] == "mixed_abstain")
    check("insufficient_evidence False is rejected", ll.validate_hypothesis({"insufficient_evidence": False}, pk, P)["status"] == "rejected")

    def bad(why, **over):
        h = dict(good)
        h.update(over)
        for k in [k for k, v in over.items() if v is ...]:
            h.pop(k)
        r = ll.validate_hypothesis(h, pk, P)
        check(f"rejected: {why}", r["status"] == "rejected", str(r))
        return r["reason"]

    check("reason: evidence not in packet", bad("evidence id outside the packet", evidence=[999]) == "evidence_not_in_packet")
    check("reason: empty evidence", bad("empty evidence", evidence=[]) == "bad_evidence")
    bad("evidence with bool", evidence=[True])
    bad("evidence duplicates", evidence=[f["ev"], f["ev"]])
    bad("evidence too many", evidence=list(range(1, 12)))
    bad("evidence as string", evidence="1")
    check("reason: cause contradicts deterministic", bad("cause contradicts", cause="unreachable") == "cause_contradicts_deterministic")
    bad("cause outside taxonomy", cause="gremlins")
    check("reason: action outside the list", bad("action outside list", action="teleport") == "bad_action")
    check("reason: action incompatible with the cause", bad("incompatible action", action="try_other_instance") == "action_incompatible")
    bad("prose too long", hypothesis="x" * (int(P["lessons"]["hypothesis_max_chars"]["value"]) + 1))
    bad("prose empty", hypothesis="  ")
    check("reason: sim id in prose", bad("sim id", hypothesis="the Apple|+01.00|+00.90|-01.00| was hidden") == "prose_forbidden")
    bad("objectId word", hypothesis="the objectId was wrong")
    bad("errors_logged token", hypothesis="check errors_logged now")
    bad("camelCase metadata key", hypothesis="the isOpen flag says closed")
    bad("control char", hypothesis="bad\x00text")
    bad("bidi override", hypothesis="text ‮ reversed")
    bad("_physics scene tag", hypothesis="in FloorPlan1_physics it failed")
    bad("alternative same as action", alternative="open_first")
    bad("alternative outside the list", alternative="levitate")
    bad("alternative incompatible", alternative="try_other_instance")
    bad("unknown key", mood="hopeful")
    bad("missing action", action=...)
    bad("missing evidence", evidence=...)
    check("plain words like 'visible' or 'position' are allowed in prose", ll.validate_hypothesis(dict(good, hypothesis="The apple was not visible from this position."), pk, P)["status"] == "accepted")
    check("a Persian sentence with ZWNJ is allowed", ll.validate_hypothesis(dict(good, hypothesis="سیب دیده نمی‌شد"), pk, P)["status"] == "accepted")
    check("valid alternative accepted", ll.validate_hypothesis(dict(good, alternative="change_pose"), pk, P)["alternative"] == "change_pose")
    check("action none is a valid (no-remedy) answer", ll.validate_hypothesis(dict(good, action="none"), pk, P)["action"] == "none")
    check("non-dict -> not_object", ll.validate_hypothesis(["x"], pk, P)["reason"] == "not_object")
    check("unparseable marker -> rejected", ll.validate_hypothesis({"_unparseable": "x"}, pk, P)["reason"] == "unparseable")

    # parser + prompt + callable wrapper (a mock - no model is contacted)
    txt = 'Sure! ```json\n{"insufficient_evidence": true}\n``` hope that helps'
    check("parser tolerates fences and chatter", ll.parse_response(txt) == {"insufficient_evidence": True})
    check("parser: plain object", ll.parse_response(' {"a": 1} ') == {"a": 1})
    check("parser: garbage raises", all(raises(ll.ParseError, ll.parse_response, t) for t in ("", "no json here", "{broken", "[1,2]", None)))
    prompt = ll.build_prompt(pk)
    check("prompt names the fixed cause + the allowed actions only", "not_detected" in prompt and "search_elsewhere" in prompt and "levitate" not in prompt)
    check("prompt has the numbered events but no task text / ids", f"[{f['ev']}]" in prompt and "find the apple" not in prompt and "objectId" not in prompt)
    seen = []
    H = ll.CallableHypothesizer(lambda p: (seen.append(p), '{"insufficient_evidence": true}')[1])
    check("CallableHypothesizer: prompt -> text -> dict", H.propose(pk) == {"insufficient_evidence": True} and seen and "Reply with ONE JSON" in seen[0])
    check("CallableHypothesizer: unparseable text becomes a rejected marker", ll.validate_hypothesis(ll.CallableHypothesizer(lambda p: "lol").propose(pk), pk, P)["status"] == "rejected")
    check("NullHypothesizer abstains", ll.validate_hypothesis(ll.NullHypothesizer().propose(pk), pk, P)["status"] == "abstain")
    # store-compatibility fuzz: whatever the prose gate lets through can never abort a consolidation
    L = new_ltm(tmp, "fuzz")
    L.store.clock()
    rng = random.Random(7)
    alphabet = list("abcXYZ _-|0123456789.,") + ["‌", "‏", "‮", "\u0000", " ", " ", "﻿", "ی", "ك", "ب", "́", "\n", "\t", "é", "💥", "\x07"]
    ok = bad_n = 0
    for _ in range(600):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        if ll.prose_problem(s) is None:
            try:
                L.store._normalise_op({"table": "hypothesis_audit", "op": "insert",
                                       "values": {"hyp": "hy:f", "lesson": None, "episode": "e", "prose": s, "evidence": [1],
                                                  "accepted": False, "reason": "x"}}, 0)
                ok += 1
            except Exception:
                bad_n += 1
    check("prose gate is at least as strict as the store's text column (600 fuzz strings)", bad_n == 0 and ok > 20, f"ok={ok} bad={bad_n}")


# ================================================================================================= 3 lesson creation
def test_creation(tmp):
    L = new_ltm(tmp, "create")
    base_counts = {t: L.store.count(t) for t in ("search_trial", "habit", "location_claim", "negative_evidence", "placement")}
    res, ext = run(L, fail_then_sidestep("c1", roles(0)))
    check("episode with a failure commits", res["status"] == "committed", str(res))
    les = rows(L, "lesson")
    check("one lesson created (tentative, default action, no hypothesizer)", len(les) == 1 and les[0]["status"] == "tentative"
          and les[0]["action"] == "search_elsewhere" and les[0]["origin_episode"] == "c1")
    ls = les[0]
    check("lesson conditions are the structured local failure key", ls["conditions"] == {"cause": "not_detected", "verb": "find", "otype": "Apple", "rtype": "Fridge", "state": "closed"})
    check("lesson id derives from kitchen + cond_key", ls["lesson"] == ll.lesson_id("FloorPlan1", ls["cond_key"]))
    check("counters start at zero and E fields stamped with the episode E", ls["helpful_units"] == ls["unhelpful_units"] == ls["recurrences"] == 0 and ls["E_created"] == ls["E_support"] == res["E"])
    check("revalidate starts False", ls["revalidate"] in (0, False))
    ev = evs(L, ls["lesson"])
    check("one evidence row for the lesson creation (rule R8, create)", len(ev) == 1 and ev[0]["rule"] == "R8" and ev[0]["mutation"] == "create")
    check("evidence summary is enumerated tokens", ev[0]["summary"]["source"] == "default" and ev[0]["summary"]["cause"] == "not_detected")
    check("write_log carries the lesson insert", any(w["table_name"] == "lesson" for w in rows(L, "write_log")) if rows(L, "write_log") and "table_name" in rows(L, "write_log")[0] else True)
    check("lessons never touch trials / habits / claims", all(L.store.count(t) == v for t, v in base_counts.items()))
    check("no hypothesizer -> no hypothesis_audit prose; only the episode summary row", [r["reason"] for r in rows(L, "hypothesis_audit")] == ["episode_summary"])
    dx = rows(L, "hypothesis_audit")[0]
    check("per-episode summary row carries the diagnostic counters", dx["evidence"]["counters"]["lessons_created"] == 1 and dx["evidence"]["counters"]["dx_not_detected"] == 1)
    check("last_stats mirrors the committed counters", ext.last_stats["lessons_created"] == 1)
    # replay: duplicate is a no-op (idempotent), same lesson count
    res2, _ = run(L, fail_then_sidestep("c1", roles(0)))
    check("replaying the same episode is a duplicate no-op", res2["status"] == "duplicate" and len(rows(L, "lesson")) == 1)

    # recurrence: once per episode, evidence of the problem, no new lesson, E_support unchanged
    e = Ep("c2", concepts=roles(1)); e.step(0); e.open_("Fridge", "closed")
    e.attempt(0, 0, False, "not_detected"); e.attempt(0, 1, False, "not_detected", recovery="sidestep"); e.attempt(0, 2, True, recovery="step_back"); e.outcome(0, True, retries=2)
    e.step(1); e.open_("Fridge", "closed"); e.attempt(1, 0, False, "not_detected"); e.attempt(1, 1, True, recovery="sidestep"); e.outcome(1, True, retries=1)
    e.finish()
    res, ext = run(L, e)
    ls2 = rows(L, "lesson")
    check("same key three times in one episode -> exactly ONE recurrence", len(ls2) == 1 and ls2[0]["recurrences"] == 1, str(ls2))
    check("recurrence leaves E_support untouched (it is evidence of the problem, not of the remedy)", ls2[0]["E_support"] == ls["E_support"])
    ev = evs(L, ls["lesson"])
    check("recurrence has its own evidence row (support, kind recurrence)", len(ev) == 2 and ev[-1]["mutation"] == "support" and ev[-1]["summary"]["recur"] == 1)

    # different state -> a different lesson key
    e = Ep("c3", concepts=roles(2)); e.step(0); e.open_("Fridge", "open")
    e.attempt(0, 0, False, "not_detected"); e.attempt(0, 1, True, recovery="sidestep"); e.outcome(0, True); e.finish()
    run(L, e)
    check("a different state-seen is a different local key / lesson", len(rows(L, "lesson")) == 2)

    # causes with no remedy never create lessons
    for nm, cause in (("c4", "unknown"), ("c5", "budget_exhausted")):
        e = Ep(nm, concepts=roles(3)); e.step(0); e.attempt(0, 0, False, "weird_code" if cause == "unknown" else "retry_exhausted"); e.outcome(0, False); e.finish(False)
        _, ext = run(L, e)
        check(f"{cause}: diagnosed, no lesson", len(rows(L, "lesson")) == 2 and ext.last_stats.get("no_remedy_for_" + cause, 0) == 1, str(ext.last_stats))
    e = Ep("c6", concepts=roles(3)); e.step(0); e.outcome(0, True); e.finish(True, goal=False)
    _, ext = run(L, e)
    check("plan_goal_unmet is counted and never creates a lesson", ext.last_stats.get("dx_plan_goal_unmet") == 1 and len(rows(L, "lesson")) == 2)

    # hypothesis paths
    pkt = []

    def good_reply(packet):
        pkt.append(packet)
        return {"cause": packet["cause"], "hypothesis": "Container was closed; opening first should help.",
                "evidence": [packet["events"][0]["ev"]], "action": "open_first"}

    L2 = new_ltm(tmp, "create-h")
    M = mock(good_reply)
    res, ext = run(L2, fail_then_sidestep("h1", roles(0)), ll.LessonsExtension(M))
    l2 = rows(L2, "lesson")
    check("accepted hypothesis: its action becomes the lesson action", len(l2) == 1 and l2[0]["action"] == "open_first")
    ha = [r for r in rows(L2, "hypothesis_audit") if r["reason"] == "accepted"]
    check("audit row keeps the prose + evidence refs, accepted=True", len(ha) == 1 and ha[0]["accepted"] and "opening first" in ha[0]["prose"] and ha[0]["evidence"])
    check("hypothesis prose is NOT in the lesson row / evidence rows", all("opening first" not in canon(r) for r in rows(L2, "lesson") + rows(L2, "evidence")))
    ev = evs(L2, l2[0]["lesson"])[0]
    check("evidence summary marks the source as hypothesis", ev["summary"]["source"] == "hypothesis")
    run(L2, fail_then_sidestep("h2", roles(1)), ll.LessonsExtension(M))
    check("an existing lesson key does not call the hypothesizer again (recurrence)", M.calls == 1 and rows(L2, "lesson")[0]["recurrences"] == 1)

    def run_h(name, reply, kitchen_roles=0):
        Lx = new_ltm(tmp, "create-" + name)
        Mx = mock(reply)
        r, ex = run(Lx, fail_then_sidestep("x1", roles(kitchen_roles)), ll.LessonsExtension(Mx))
        return Lx, Mx, ex

    Lx, Mx, ex = run_h("rej", lambda p: {"cause": p["cause"], "hypothesis": "ok", "evidence": [99999], "action": "open_first"})
    check("rejected hypothesis -> NO lesson (plan 7.4)", rows(Lx, "lesson") == [] and ex.last_stats.get("hyp_rejected") == 1)
    ha = [r for r in rows(Lx, "hypothesis_audit") if r["reason"] != "episode_summary"]
    check("rejection is audited with a reason", len(ha) == 1 and ha[0]["reason"] == "evidence_not_in_packet" and not ha[0]["accepted"])
    Lx, Mx, ex = run_h("rejprose", lambda p: {"cause": p["cause"], "hypothesis": "see objectId Apple|1|2|3 now", "evidence": [p["events"][0]["ev"]], "action": "open_first"})
    ha = [r for r in rows(Lx, "hypothesis_audit") if r["reason"] != "episode_summary"]
    check("forbidden prose: rejected, NOT stored (it would abort consolidation), episode still committed",
          rows(Lx, "lesson") == [] and ha and ha[0]["prose"] is None and ha[0]["reason"] == "prose_forbidden" and rows(Lx, "episodes"))
    Lx, Mx, ex = run_h("abst", {"insufficient_evidence": True})
    check("abstention -> default action lesson", len(rows(Lx, "lesson")) == 1 and rows(Lx, "lesson")[0]["action"] == "search_elsewhere" and ex.last_stats.get("hyp_abstain") == 1)
    Lx, Mx, ex = run_h("none", lambda p: {"cause": p["cause"], "hypothesis": "nothing helps", "evidence": [p["events"][0]["ev"]], "action": "none"})
    check("accepted hypothesis with action none -> no lesson", rows(Lx, "lesson") == [] and ex.last_stats.get("lessons_not_created_hypothesis_none") == 1)
    Lx, Mx, ex = run_h("boom", RuntimeError("model crashed"))
    check("a crashing hypothesizer never aborts consolidation (rejected, no lesson)", rows(Lx, "lesson") == [] and rows(Lx, "episodes") and ex.last_stats.get("hyp_rejected") == 1)
    Lx, Mx, ex = run_h("garbage", "I think maybe the fridge")
    check("raw non-JSON from a plain str reply is rejected, not crashed", rows(Lx, "lesson") == [])
    Lx, Mx, ex = run_h("cause", lambda p: {"cause": "unreachable", "hypothesis": "x", "evidence": [p["events"][0]["ev"]], "action": "approach_closer"})
    check("hypothesis contradicting the deterministic cause -> rejected", rows(Lx, "lesson") == [] and rows(Lx, "hypothesis_audit")[0]["reason"] != "accepted")
    check("Gemma packet contained only enumerated fields", all("text" not in canon(p) for p in Mx.packets))

    # atomicity: a crash inside the commit loses nothing and a retry works
    state = {"boom": True}

    def hook(point):
        if point == "pre_commit" and state["boom"]:
            raise RuntimeError("crash")

    Lr = new_ltm(tmp, "atomic", fault_hook=hook)
    h0 = Lr.state_hash()
    check("crash before commit raises", raises(RuntimeError, run, Lr, fail_then_sidestep("a1", roles(0))))
    check("rollback: no lesson, no log, no summary, state hash unchanged", Lr.state_hash() == h0 and rows(Lr, "lesson") == [] and rows(Lr, "hypothesis_audit") == [])
    state["boom"] = False
    res, _ = run(Lr, fail_then_sidestep("a1", roles(0)))
    check("retry after the crash commits once", res["status"] == "committed" and len(rows(Lr, "lesson")) == 1)

    # determinism: the same episodes in the same order give the same state hash (two stores)
    A, B = new_ltm(tmp, "det-a"), new_ltm(tmp, "det-b")
    for L_ in (A, B):
        run(L_, fail_then_sidestep("d1", roles(0)))
        run(L_, fail_then_sidestep("d2", roles(1)))
    check("deterministic: identical state hash", A.state_hash() == B.state_hash())


# ================================================================================================= 4 application scoring
def fresh_with_lesson(tmp, name, P=None):
    L = new_ltm(tmp, name, P)
    run(L, fail_then_sidestep("o1", roles(0)))
    return L, the_lesson(L)


def test_scoring(tmp):
    L, ls = fresh_with_lesson(tmp, "score")
    lid = ls["lesson"]

    def one(name, **kw):
        Lx, lx = fresh_with_lesson(tmp, "s-" + name)
        res, ext = run(Lx, apply_ep(name, roles(1), lx["lesson"], **kw))
        r = the_lesson(Lx)
        logs = llogs(Lx)
        return Lx, r, logs, ext

    Lx, r, logs, ext = one("help")
    check("helpful: step ok on the applied attempt", len(logs) == 1 and logs[0]["outcome"] == "helpful" and r["helpful_units"] == 1 and r["unhelpful_units"] == 0)
    check("log row: lesson kind, goal_ok never stored for lessons, unit key in source", logs[0]["kind"] == "lesson" and logs[0]["goal_ok"] in (None, 0) and logs[0]["goal_ok"] is None and logs[0]["source"].startswith("u:"))
    check("helpful application refreshes E_support", r["E_support"] > ls["E_support"])
    Lx, r, logs, ext = one("unhelp", k1_ok=False, k1_cause="not_detected")
    check("unhelpful: the same cause recurs", logs[0]["outcome"] == "unhelpful" and r["unhelpful_units"] == 1)
    ev = evs(Lx, r["lesson"])[-1]
    check("unhelpful -> a 'contradict' evidence row", ev["mutation"] == "contradict")
    Lx, r, logs, ext = one("diffcause", k1_ok=False, k1_cause="unreachable")
    check("neutral: the step fails with a DIFFERENT cause", logs[0]["outcome"] == "neutral" and r["helpful_units"] == r["unhelpful_units"] == 0)
    Lx, r, logs, ext = one("later", k1_ok=False, k1_cause="unreachable", later_ok=True)
    check("neutral: success credited to a later different remedy", logs[0]["outcome"] == "neutral")
    Lx, r, logs, ext = one("laterSame", k1_ok=False, k1_cause="not_detected", later_ok=True)
    check("same cause recurring wins even if a later remedy then succeeds (unhelpful)", logs[0]["outcome"] == "unhelpful")
    Lx, r, logs, ext = one("noout", k1_ok=True, drop_outcome=True)
    check("applied attempt ok, step outcome missing, verdict not 'unknown' -> helpful (the attempt itself is observed)", logs[0]["outcome"] == "helpful")
    Lx, r, logs, ext = one("abort", k1_ok=True, drop_outcome=True, verdict="unknown")
    check("aborted (no step outcome, agent verdict unknown) -> neutral", logs[0]["outcome"] == "neutral")
    # unobservable: attempt result missing ok
    Lx, lx = fresh_with_lesson(tmp, "s-unobs")
    e = Ep("u1", concepts=roles(1)); e.step(0); e.open_("Fridge", "closed"); e.attempt(0, 0, False, "not_detected")
    e.app(lx["lesson"], 0, 1); e.attempt(0, 1, None, recovery="lesson", source=f"lesson:{lx['lesson']}"); e.outcome(0, True); e.finish()
    run(Lx, e)
    logs = llogs(Lx)
    check("unobservable attempt result -> neutral", logs and logs[0]["outcome"] == "neutral")
    # failure elsewhere in the task never punishes a lesson that worked
    Lx, r, logs, ext = one("elsewhere", extra_fail=True, goal=False)
    check("whole-task failure elsewhere does not make a helpful lesson unhelpful", logs[0]["outcome"] == "helpful" and r["unhelpful_units"] == 0 and r["helpful_units"] == 1)
    Lx, r, logs, ext = one("goalfalse", goal=False)
    check("the privileged goal verdict is never used to score a lesson", logs[0]["outcome"] == "helpful")

    # eligibility
    def inelig(name, why, **kw):
        Lx, r, logs, ext = one(name, **kw)
        ok = logs == [] and ext.last_stats.get("app_ineligible_" + why, 0) == 1 and r["helpful_units"] == r["unhelpful_units"] == 0
        check(f"ineligible application: {why}", ok, str(ext.last_stats) + str(logs))

    inelig("prov", "provenance", source="lesson:ls:deadbeef")
    inelig("k0", "attempt_range", k=0)
    inelig("k4", "attempt_range", k=4)
    Lx, lx = fresh_with_lesson(tmp, "s-two")
    run(Lx, fail_then_sidestep("o2", roles(5), k1="fresh_point"))     # same key -> recurrence only
    e = Ep("two1", concepts=roles(2)); e.step(0); e.open_("Fridge", "closed"); e.attempt(0, 0, False, "not_detected")
    e.app(lx["lesson"], 0, 1); e.app("ls:other0000000000", 0, 1)
    e.attempt(0, 1, True, recovery="lesson", source=f"lesson:{lx['lesson']}"); e.outcome(0, True); e.finish()
    _, ext = run(Lx, e)
    check("two different lessons on one recovery attempt -> protocol violation, neither scored", ext.last_stats.get("protocol_violations", 0) >= 1 and llogs(Lx) == [])
    e = Ep("unk1", concepts=roles(3)); e.step(0); e.open_("Fridge", "closed"); e.attempt(0, 0, False, "not_detected")
    e.app("ls:0000000000000000", 0, 1); e.attempt(0, 1, True, recovery="lesson", source="lesson:ls:0000000000000000"); e.outcome(0, True); e.finish()
    _, ext = run(Lx, e)
    check("unknown record id is ineligible", ext.last_stats.get("app_ineligible_unknown_record") == 1)
    # key mismatch: the prior failure has another key (open event says 'open')
    e = Ep("km", concepts=roles(4)); e.step(0); e.open_("Fridge", "open"); e.attempt(0, 0, False, "not_detected")
    e.app(lx["lesson"], 0, 1); e.attempt(0, 1, True, recovery="lesson", source=f"lesson:{lx['lesson']}"); e.outcome(0, True); e.finish()
    _, ext = run(Lx, e)
    check("application at a failure whose key differs from the lesson's -> ineligible", ext.last_stats.get("app_ineligible_key_mismatch") == 1)
    # chronology: application emitted AFTER the attempt result
    e = Ep("chr", concepts=roles(6)); e.step(0); e.open_("Fridge", "closed"); e.attempt(0, 0, False, "not_detected")
    e.attempt(0, 1, True, recovery="lesson", source=f"lesson:{lx['lesson']}"); e.app(lx["lesson"], 0, 1); e.outcome(0, True); e.finish()
    _, ext = run(Lx, e)
    check("application after its attempt result -> ineligible (chronology)", ext.last_stats.get("app_ineligible_chronology") == 1)
    # first eligible only
    Ly, ly = fresh_with_lesson(tmp, "s-first")
    e = Ep("f1", concepts=roles(1)); e.step(0); e.open_("Fridge", "closed"); e.attempt(0, 0, False, "not_detected")
    e.app(ly["lesson"], 0, 1); e.attempt(0, 1, False, "not_detected", recovery="lesson", source=f"lesson:{ly['lesson']}")
    e.app(ly["lesson"], 0, 2); e.attempt(0, 2, True, recovery="lesson", source=f"lesson:{ly['lesson']}"); e.outcome(0, True, retries=2); e.finish()
    _, ext = run(Ly, e)
    logs = llogs(Ly)
    check("only the FIRST eligible application per lesson per episode is scored",
          len(logs) == 1 and logs[0]["attempt_k"] == 1 and logs[0]["outcome"] == "unhelpful" and ext.last_stats.get("app_repeat_not_scored") == 1, str(logs) + str(ext.last_stats))
    # retired lessons are not scored
    Lz, lz = fresh_with_lesson(tmp, "s-ret")
    Lz.store.commit("force-retire", "admin", [{"table": "lesson", "op": "update", "key": {"lesson": lz["lesson"]}, "values": {"status": "retired"},
                                              "evidence": {"rule": "R8", "mutation": "retire", "summary": {"kind": "test"}, "src_ev": []}}], stamp=Lz.clock())
    _, ext = run(Lz, apply_ep("rt", roles(1), lz["lesson"]))
    check("a retired lesson is ineligible", ext.last_stats.get("app_ineligible_retired") == 1)
    check("retired lesson recurs but stays retired (terminal)", the_lesson(Lz)["status"] == "retired")


# ================================================================================================= 5 lifecycle
def test_lifecycle(tmp):
    # pure transition function
    ns = ll.next_status
    check("tentative stays tentative below the promote threshold", ns("tentative", 1, 0, 2, 2) == "tentative")
    check("tentative -> supported at promote_after helpful and helpful > unhelpful", ns("tentative", 2, 0, 2, 2) == "supported")
    check("no promotion when unhelpful >= helpful", ns("tentative", 2, 2, 2, 2) == "tentative")
    check("tentative -> retired when unhelpful - helpful >= margin", ns("tentative", 0, 2, 2, 2) == "retired")
    check("supported with one counterexample (2h/1u) stays supported (no single-counterexample demotion)", ns("supported", 2, 1, 2, 2) == "supported")
    check("supported -> contradicted when unhelpful >= helpful", ns("supported", 2, 2, 2, 2) == "contradicted")
    check("contradicted -> supported when helpful > unhelpful", ns("contradicted", 3, 2, 2, 2) == "supported")
    check("contradicted -> retired at margin", ns("contradicted", 2, 4, 2, 2) == "retired")
    check("contradicted stays contradicted otherwise", ns("contradicted", 2, 2, 2, 2) == "contradicted")
    check("retired is terminal", ns("retired", 9, 0, 2, 2) == "retired")
    check("settle: supported -> contradicted -> retired in one recount", ll.settle_status("supported", 0, 3, 2, 2) == ("retired", ["supported_to_contradicted", "contradicted_to_retired"]))
    check("unknown status raises", raises(ll.LessonError, ns, "bogus", 0, 0, 2, 2))
    # exhaustive no-cycle / determinism check of the settle function
    ok = True
    for prev in ll.LESSON_ROLES:
        for h in range(5):
            for u in range(5):
                a = ll.settle_status(prev, h, u, 2, 2)
                b = ll.settle_status(a[0], h, u, 2, 2)
                ok &= (b[0] == a[0] and not b[1])
    check("settle reaches a fixed point for every (status, h, u)", ok)
    check("unit_counts: neutral and keyless rows ignored", ll.unit_counts([{"outcome": "neutral", "source": "u:a"}, {"outcome": "helpful", "source": None}]) == (0, 0))
    check("unit_counts: two helpful rows of one unit count once", ll.unit_counts([{"outcome": "helpful", "source": "u:a", "episode": "x"}, {"outcome": "helpful", "source": "u:a", "episode": "y"}]) == (1, 0))
    check("unit_counts: a counterexample wins inside a unit (order independent)", ll.unit_counts([{"outcome": "helpful", "source": "u:a"}, {"outcome": "unhelpful", "source": "u:a"}]) == (0, 1)
          and ll.unit_counts([{"outcome": "unhelpful", "source": "u:a"}, {"outcome": "helpful", "source": "u:a"}]) == (0, 1))
    check("unit_counts: origin episode excluded", ll.unit_counts([{"outcome": "helpful", "source": "u:a", "episode": "o"}], exclude_episode="o") == (0, 0))

    # end-to-end: promotion needs 2 helpful distinct units
    L, ls = fresh_with_lesson(tmp, "life")
    lid = ls["lesson"]
    run(L, apply_ep("p1", roles(1), lid))
    r = the_lesson(L)
    check("1 helpful unit: still tentative", r["helpful_units"] == 1 and r["status"] == "tentative")
    run(L, apply_ep("p1b", roles(1), lid))
    r = the_lesson(L)
    check("same unit (same task concepts, same epoch) again: counted once", r["helpful_units"] == 1 and r["status"] == "tentative")
    run(L, apply_ep("p2", roles(2), lid))
    r = the_lesson(L)
    check("second distinct unit -> supported", r["helpful_units"] == 2 and r["status"] == "supported")
    ev = evs(L, lid)
    check("evidence preserves the transition", any("tentative_to_supported" in e["summary"].get("transitions", []) for e in ev))
    # a counterexample, then more
    run(L, apply_ep("n1", roles(3), lid, k1_ok=False))
    r = the_lesson(L)
    check("one unhelpful unit: supported 2h/1u stays supported", r["status"] == "supported" and r["unhelpful_units"] == 1)
    run(L, apply_ep("n2", roles(4), lid, k1_ok=False))
    r = the_lesson(L)
    check("unhelpful >= helpful -> contradicted", r["status"] == "contradicted" and r["unhelpful_units"] == 2)
    ev = evs(L, lid)[-1]
    check("contradiction is logged as a contradict mutation with the transition", ev["mutation"] == "contradict" and "supported_to_contradicted" in ev["summary"]["transitions"])
    run(L, apply_ep("n3", roles(5), lid, k1_ok=False))
    check("h=2, u=3: contradicted, not yet retired", the_lesson(L)["status"] == "contradicted")
    run(L, apply_ep("n4", roles(6), lid, k1_ok=False))
    r = the_lesson(L)
    check("unhelpful - helpful >= margin -> retired", r["status"] == "retired")
    ev = evs(L, lid)[-1]
    check("retirement logged as retire", ev["mutation"] == "retire")
    check("a retired lesson is never deleted", the_lesson(L) is not None and len(rows(L, "application_log")) >= 5)

    # contradicted lesson that recovers
    L, ls = fresh_with_lesson(tmp, "life2")
    lid = ls["lesson"]
    for nm, n, okk in (("a", 1, True), ("b", 2, True), ("c", 3, False), ("d", 4, False), ("e", 5, True), ("f", 6, True)):
        run(L, apply_ep("q" + nm, roles(n), lid, k1_ok=okk))
    r = the_lesson(L)
    check("contradicted -> supported once helpful > unhelpful again", r["status"] == "supported" and r["helpful_units"] == 4 and r["unhelpful_units"] == 2)

    # EN/FA pair share a unit through concepts; epochs make distinct units
    L, ls = fresh_with_lesson(tmp, "life3")
    lid = ls["lesson"]
    e1 = apply_ep("pa-en", roles(1), lid)
    e2 = apply_ep("pa-fa", roles(1), lid)
    for e_, lang, txt in ((e1, "en", "find the apple"), (e2, "fa", "سیب را پیدا کن")):
        e_.events[1]["lang"], e_.events[1]["text"] = lang, txt
    run(L, e1)
    run(L, e2)
    check("EN and FA of one item (same concepts) are ONE unit", the_lesson(L)["helpful_units"] == 1)
    L.reset_world("r1")
    run(L, apply_ep("pa-en2", roles(1), lid, epoch=1))
    check("the same item in a later epoch is a distinct unit", the_lesson(L)["helpful_units"] == 2)
    # without concepts: text-hash fallback, language specific
    Lq, lq = fresh_with_lesson(tmp, "life4")
    e = apply_ep("tx1", None, lq["lesson"]); e.events[1]["text"] = "find the apple"
    run(Lq, e)
    e = apply_ep("tx2", None, lq["lesson"]); e.events[1]["text"] = "Find the apple "
    run(Lq, e)
    check("no concepts: normalised text hash is the unit (case / spaces folded)", the_lesson(Lq)["helpful_units"] == 1)

    # origin episode excluded: re-running the origin task as an application cannot credit itself
    Lo, lo = fresh_with_lesson(tmp, "life5")
    e = apply_ep("o1x", roles(0), lo["lesson"])
    run(Lo, e)
    check("a distinct episode of the origin task still counts (only the origin EPISODE is excluded)", the_lesson(Lo)["helpful_units"] == 1)

    # test / probe modes: nothing is written, the extension refuses
    Lt, lt = fresh_with_lesson(tmp, "test-mode")
    h0 = Lt.state_hash()
    e = apply_ep("tm", roles(1), lt["lesson"])
    e.events = [ev for ev in e.events if ev["kind"] != "admission_verdict"]   # practice-only (plan 1.3, A7-review)
    for ev in e.events:
        if ev["kind"] == "episode_start":
            ev["mode"] = "test"
    res = Lt.end_episode(e.wm(), extensions=[ll.LessonsExtension()])
    check("a test-mode episode writes nothing (no scoring, no lessons)", res["status"] == "skipped_mode" and Lt.state_hash() == h0)
    wm_test = e.wm()
    _fk, _av = dict(lt["conditions"]), {lt["action"]}
    check("facade LongTermMemory.lesson_for == ltm_lessons.lesson_for (test mode)",
          Lt.lesson_for(wm_test, _fk, _av, 3) == ll.lesson_for(Lt, wm_test, _fk, _av, 3))
    ctx = type("C", (), {"wm": wm_test, "ltm": Lt, "kitchen": "FloorPlan1", "E_next": 9, "goal_ok": True, "exec_ok": True})()
    check("calling the extension directly on a test episode raises (enforced in code)", raises(ll.LessonError, ll.LessonsExtension(), ctx))

    # lesson_for
    Lf, lf = fresh_with_lesson(tmp, "lfor")
    key = lf["conditions"]
    wm_prac = fail_then_sidestep("pp", roles(1)).wm()
    wm_test2 = fail_then_sidestep("tt", roles(1), ).wm()
    wm_test2.events[0]["mode"] = "test"
    avail = {"search_elsewhere", "change_pose"}
    got = ll.lesson_for(Lf, wm_prac, key, avail, 2)
    check("practice: a tentative lesson is offered (experimental)", got and got["lesson"] == lf["lesson"] and got["experimental"] and got["status"] == "tentative" and got["source"] == f"lesson:{lf['lesson']}")
    check("test mode: a tentative lesson is NEVER applied", ll.lesson_for(Lf, wm_test2, key, avail, 2) is None)
    check("no wm at all behaves like test mode", ll.lesson_for(Lf, None, key, avail, 2) is None)
    check("no retries left -> none", ll.lesson_for(Lf, wm_prac, key, avail, 0) is None)
    check("action not executable now -> none", ll.lesson_for(Lf, wm_prac, key, {"crouch"}, 2) is None)
    check("empty availability -> none", ll.lesson_for(Lf, wm_prac, key, set(), 2) is None)
    check("another local key -> none", ll.lesson_for(Lf, wm_prac, dict(key, state="open"), avail, 2) is None)
    check("another cause -> none", ll.lesson_for(Lf, wm_prac, dict(key, cause="unreachable"), avail, 2) is None)
    for nm, n in (("s1", 1), ("s2", 2)):
        run(Lf, apply_ep(nm, roles(n), lf["lesson"]))
    got = ll.lesson_for(Lf, wm_test2, key, avail, 2)
    check("test mode: a SUPPORTED lesson is applied", got and got["status"] == "supported" and not got["experimental"])
    check("test mode: a supported lesson under revalidation overlay is not applied", ll.lesson_for(Lf, wm_test2, key, avail, 2, revalidating={lf["lesson"]}) is None)
    check("practice: revalidation overlay ranks it as tentative (still usable)", ll.lesson_for(Lf, wm_prac, key, avail, 2, revalidating={lf["lesson"]})["status"] == "tentative")
    h0 = Lf.state_hash()
    ll.lesson_for(Lf, wm_prac, key, avail, 2)
    check("lesson_for does not write", Lf.state_hash() == h0)
    # frozen snapshot: reading works, nothing counted
    snap = Path(tmp) / "snap-lessons"
    Lf.snapshot(snap)
    F = ltm.LongTermMemory.open_frozen(snap, Path(tmp) / "snap-lessons-testlog.db")
    wmx = fail_then_sidestep("fz", roles(1)).wm()
    wmx.events[0]["mode"] = "test"
    g = ll.lesson_for(F, wmx, key, avail, 2)
    check("frozen store: supported lesson readable", g and g["status"] == "supported")
    hf = F.state_hash()
    r = ll.log_test_application(F, wmx, g["lesson"], "lesson", 0, 1)
    check("frozen store: the use goes to the separate testlog; the snapshot hash is unchanged", hf == F.state_hash() and r["status"] == "committed")
    F.close()


# ================================================================================================= 6 revalidation
def test_revalidate(tmp):
    L, ls = fresh_with_lesson(tmp, "reval")
    lid = ls["lesson"]
    for nm, n in (("s1", 1), ("s2", 2)):
        run(L, apply_ep(nm, roles(n), lid))
    check("setup: supported", the_lesson(L)["status"] == "supported")
    L.reset_world("w1")
    res = ll.flag_revalidate(L, "w1", "reset_world")
    r = the_lesson(L)
    check("flag: geometry lesson (search_elsewhere is NOT geometry) is left alone", res["status"] == "nothing_to_flag" and not r["revalidate"])
    # make a geometry lesson via an accepted hypothesis
    M = mock(lambda p: {"cause": p["cause"], "hypothesis": "Get closer.", "evidence": [p["events"][0]["ev"]], "action": "approach_closer"})
    G = new_ltm(tmp, "reval-g")
    run(G, fail_then_sidestep("g0", roles(0)), ll.LessonsExtension(M))
    g = the_lesson(G)
    check("geometry lesson created via hypothesis", g["action"] == "approach_closer")
    for nm, n in (("s1", 1), ("s2", 2)):
        run(G, apply_ep(nm, roles(n), g["lesson"]))
    check("setup: supported before the reset", the_lesson(G)["status"] == "supported")
    G.reset_world("w1")
    res = ll.flag_revalidate(G, "w1", "reset_world")
    g = the_lesson(G)
    check("reset flags a supported geometry lesson for revalidation", res["status"] == "committed" and g["revalidate"] and g["status"] == "supported")
    check("status itself is unchanged; effective status is tentative", ll.effective_status(g) == "tentative")
    again = ll.flag_revalidate(G, "w1", "reset_world")
    check("flagging is idempotent per boundary id", again["status"] == "duplicate")
    wm_t = fail_then_sidestep("tt", roles(7), epoch=1).wm()
    wm_t.events[0]["mode"] = "test"
    check("test mode: a revalidating lesson is not applied", ll.lesson_for(G, wm_t, g["conditions"], {"approach_closer"}, 2) is None)
    ev = evs(G, g["lesson"])[-1]
    check("the flag is an audited mutation", ev["summary"]["kind"] == "revalidate_flag")
    # a helpful application of a unit credited BEFORE the boundary (same task) in the new epoch is a new unit (epoch differs)
    # but a duplicate of an old unit does not clear: re-apply in the same epoch-0 world is impossible, so use a stale unit via frame-only boundary
    H = new_ltm(tmp, "reval-h")
    run(H, fail_then_sidestep("g0", roles(0)), ll.LessonsExtension(M))
    hh = the_lesson(H)
    for nm, n in (("s1", 1), ("s2", 2)):
        run(H, apply_ep(nm, roles(n), hh["lesson"]))
    ll.flag_revalidate(H, "frame-change", "frame_change")          # frame-only: epoch unchanged
    hh = the_lesson(H)
    check("frame-only boundary flags too", hh["revalidate"])
    run(H, apply_ep("dup", roles(1), hh["lesson"]))                # same unit as before the boundary
    hh = the_lesson(H)
    check("a unit already credited before the boundary does NOT clear the flag", hh["revalidate"] and hh["helpful_units"] == 2)
    run(H, apply_ep("fresh", roles(9), hh["lesson"]))              # new unit after the boundary
    hh = the_lesson(H)
    check("a NEW helpful unit after the boundary clears the flag", not hh["revalidate"] and hh["helpful_units"] == 3)
    ev = evs(H, hh["lesson"])[-1]
    check("clearing is recorded in evidence", ev["summary"]["cleared_revalidate"] is True)
    # unhelpful while revalidating is counted normally
    run(G, apply_ep("bad", roles(11), g["lesson"], k1_ok=False, epoch=1))
    check("unhelpful during revalidation counts", the_lesson(G)["unhelpful_units"] == 1 and the_lesson(G)["revalidate"])
    # frozen store refuses to flag
    snap = Path(tmp) / "snap-reval"
    G.snapshot(snap)
    F = ltm.LongTermMemory.open_frozen(snap, Path(tmp) / "snap-reval-testlog.db")
    check("flagging a frozen store is refused", raises(ll.LessonError, ll.flag_revalidate, F, "x"))
    F.close()


# ================================================================================================= 7 procedures
def proc_episode(name, rl, goal=True, exec_ok=True, fail_step=None, drop_outcome=None, invalid=False, text="find the apple",
                 pre=None, verdict="success", steps=((0, "find", "Apple", "?"), (1, "pickup", "Apple", "?"), (2, "put", "Apple", "Fridge")),
                 epoch=0, repair=False, apps=None, app_pre=None):
    e = Ep(name, concepts=rl, text=text, epoch=epoch)
    for i, verb, ot, to in steps:
        e.step(i, verb, ot, to, pre=pre if pre is not None else {"holding": "n"})
    # apps sit after the step starts of their episode: emitted at the application step
    return e, steps


def build_proc(name, rl, goal=True, exec_ok=True, failed=(), unobserved=(), invalid=False, epoch=0, repair=False,
               steps=((0, "find", "Apple", "?"), (1, "pickup", "Apple", "?"), (2, "put", "Apple", "Fridge")),
               apply_proc=None, app_pre=None, verdict="success", app_i=0, text="find the apple", pre=None):
    e = Ep(name, concepts=rl, text=text, epoch=epoch)
    for i, verb, ot, to in steps:
        if apply_proc and i == app_i:
            e.app(apply_proc, i, 0, pre=app_pre if app_pre is not None else {"holding": "n"})
        e.step(i, verb, ot, to, pre=pre if pre is not None else {"holding": "n"})
        if i in unobserved:
            continue
        if i == 0 and repair:
            e.attempt(i, 0, False, "not_detected")
            e.attempt(i, 1, True, recovery="sidestep")
        if i in failed:
            e.attempt(i, 0, False, "action_refused")
            e.outcome(i, False, "action_refused")
        else:
            e.outcome(i, True)
    return e.finish(exec_ok=exec_ok, goal=goal, invalid=invalid, verdict=verdict)


def the_proc(L):
    r = rows(L, "procedure")
    return r[0] if r else None


def test_procedures(tmp):
    P = ltm.load_params()
    # admission
    L = new_ltm(tmp, "proc")
    res, ext = run(L, build_proc("pa1", roles(1), text="MARKER-find the apple"))
    p = the_proc(L)
    check("successful, supported episode -> a candidate procedure", p is not None and p["status"] == "candidate" and p["success_uses"] == 1 and p["failed_reuses"] == 0)
    check("admission records the privileged verdict in its declared column only", p["admit_goal_ok"] in (1, True))
    check("intent key = the concept roles; body = the executed sequence", p["intent_key"]["roles"][0]["otype"] == "T1" and [s["verb"] for s in p["body"]["steps"]] == ["find", "pickup", "put"])
    check("body keeps per-step observable preconditions", p["body"]["steps"][0]["pre"] == {"holding": "n"})
    check("intent_text is rendered from roles, never copied from the instruction", "MARKER" not in p["intent_text"] and "find T1" in p["intent_text"] and "MARKER" not in canon(p))
    check("procedure id derives from kitchen + roles + full body", p["proc"] == ll.procedure_id("FloorPlan1", p["intent_key"]["roles"], p["body"]))
    ev = evs(L, p["proc"])
    check("one create evidence row (R7)", len(ev) == 1 and ev[0]["rule"] == "R7" and ev[0]["mutation"] == "create")
    log = rows(L, "application_log")
    check("admission leaves an immutable unit row (retrieval='admit')", len(log) == 1 and log[0]["retrieval"] == "admit" and log[0]["outcome"] == "helpful")

    def refused(name, reason, **kw):
        Lx = new_ltm(tmp, "pr-" + name)
        r, ex = run(Lx, build_proc("x", roles(1), **kw))
        ok = rows(Lx, "procedure") == [] and ex.last_stats.get("proc_rejected_" + reason, 0) == 1
        check(f"no procedure: {reason}", ok, str(ex.last_stats) + str(r.get("status")))
        return Lx

    Lk = refused("knife", "goal_without_plan", exec_ok=False, failed=(1,))          # goal met despite a failed step
    check("knife-less slice style: goal met despite a failed step is counted, not admitted",
          rows(Lk, "episodes")[0]["goal_ok"] in (1, True) and rows(Lk, "episodes")[0]["stats"].get("goal_without_plan") == 1)
    refused("mismatch", "exec_mismatch", exec_ok=True, failed=(1,))                 # terminal lies: exec_ok True with a failed step
    refused("unobs", "step_unobserved", exec_ok=True, unobserved=(2,))
    refused("empty", "empty_plan", steps=())
    r_ = new_ltm(tmp, "pr-noverdict")
    e = Ep("nv", concepts=roles(1)); e.step(0); e.outcome(0, True); e.finish(True, goal=None)
    _, ex = run(r_, e)
    check("no verdict -> no procedure", rows(r_, "procedure") == [] and ex.last_stats.get("proc_rejected_no_verdict") == 1)
    r_ = new_ltm(tmp, "pr-goalfalse")
    _, ex = run(r_, build_proc("gf", roles(1), goal=False))
    check("goal False -> no procedure (not even counted as a rejection)", rows(r_, "procedure") == [] and not any(k.startswith("proc_rejected") for k in ex.last_stats))
    r_ = new_ltm(tmp, "pr-noint")
    e = Ep("ni", concepts=None); e.step(0); e.outcome(0, True); e.finish(True, goal=True)
    _, ex = run(r_, e)
    check("no concepts -> no intent key -> no procedure", rows(r_, "procedure") == [] and ex.last_stats.get("proc_rejected_no_intent") == 1)
    r_ = new_ltm(tmp, "pr-invalid")
    res, _ = run(r_, build_proc("iv", roles(1), invalid=True))
    check("an invalid episode consolidates nothing", res["status"] == "invalid" and rows(r_, "procedure") == [])
    r_ = new_ltm(tmp, "pr-long")
    big = tuple((i, "find", "Apple", "?") for i in range(ll.BODY_MAX_STEPS + 1))
    _, ex = run(r_, build_proc("big", roles(1), steps=big))
    check("an over-long body is refused, the episode still commits", rows(r_, "procedure") == [] and ex.last_stats.get("proc_rejected_too_long") == 1 and rows(r_, "episodes"))
    r_ = new_ltm(tmp, "pr-nested")
    e = Ep("nest", concepts=roles(1)); e.step(0, pre={"holding": "n", "deep": {"a": {"b": {"c": 1}}}, "seen": "y"}); e.outcome(0, True); e.finish(True, goal=True)
    _, ex = run(r_, e)
    pn = the_proc(r_)
    check("nested precondition values are flattened away so the store never refuses the body", pn is not None and pn["body"]["steps"][0]["pre"] == {"holding": "n", "seen": "y"} and ex.last_stats.get("ops_rejected", 0) == 0)

    # privileged verdict confinement: admission uses it, nothing else in the lesson tables does
    check("lesson rows carry no goal_ok anywhere", all(r["goal_ok"] is None for r in rows(L, "application_log") if r["kind"] == "lesson"))
    import inspect
    check("the privileged verdict is absent from lesson scoring / eligibility code",
          all("goal_ok" not in inspect.getsource(f) for f in (ll.score_lesson_application, ll.lesson_app_eligibility, ll.next_status, ll.settle_status, ll.lesson_for)))
    check("...and present where the plan allows it (admission, reuse, plan_goal_unmet)",
          all("goal_ok" in inspect.getsource(f) for f in (ll.admission_check, ll.score_procedure_application, ll.diagnose)))

    # native concept roles (sim/ltm_concepts.py shape) are stored field for field
    nat = {"roles": [{"verb": "put", "req": "-", "obj": "Apple", "ostate": "-", "src": "-", "dest": "Fridge", "dstate": "-", "neg": 0}],
           "ok": True, "reasons": [], "parser": "p1", "lexicon": "abc123"}
    Ln = new_ltm(tmp, "proc-native")
    run(Ln, build_proc("n1", nat))
    pn = the_proc(Ln)
    check("native roles stored unchanged (ltm_concepts shape) and rendered by the EN template",
          pn is not None and pn["intent_key"]["roles"] == nat["roles"] and pn["intent_text"] == "put Apple to Fridge", str(pn and pn["intent_text"]))
    nat_neg = {"roles": [dict(nat["roles"][0], neg=1)]}
    check("native neg=1 renders a 'not' and is a different intent", ll.render_intent_text(ll.intent_roles(nat_neg)) == "not put Apple to Fridge" and canon(ll.intent_roles(nat_neg)) != canon(ll.intent_roles(nat)))
    nat_q = {"roles": [dict(nat["roles"][0], dest="?")]}
    Lq2 = new_ltm(tmp, "proc-native-q")
    _, exq = run(Lq2, build_proc("n2", nat_q))
    check("a native role with a '?' field cannot steer, so it is not admitted", rows(Lq2, "procedure") == [] and exq.last_stats.get("proc_rejected_intent_unknown") == 1)
    check("EN and FA concepts with equal native roles share one unit",
          ll.unit_key(Ep("u1", concepts=nat).wm()) == ll.unit_key(Ep("u2", concepts=dict(nat, parser="p2")).wm()))

    # a repaired execution is a DIFFERENT immutable procedure (the complete body incl. repairs is hashed)
    run(L, build_proc("pa2", roles(1), repair=True))
    check("same intent, repaired execution -> a different procedure (full-body hash)", len(rows(L, "procedure")) == 2)

    # trust: second independent success unit of the SAME body
    L = new_ltm(tmp, "proc-trust")
    run(L, build_proc("t1", roles(1)))
    run(L, build_proc("t1b", roles(1)))
    p = the_proc(L)
    check("same unit again (same concepts + epoch) is not a second success", p["success_uses"] == 1 and p["status"] == "candidate")
    run(L, build_proc("t2", roles(1), epoch=0, text="x"))     # same roles -> same unit still
    check("still one unit", the_proc(L)["success_uses"] == 1)
    L.reset_world("r1")
    run(L, build_proc("t3", roles(1), epoch=1))
    p = the_proc(L)
    check("independent success in a new epoch -> trusted (trust_after=2)", p["success_uses"] == 2 and p["status"] == "trusted")
    ev = evs(L, p["proc"])[-1]
    check("support evidence row on the re-admission", ev["mutation"] == "support" and ev["summary"]["admitted"] is True)

    # reuse scoring
    def reuse(name, pid, **kw):
        return run(L, build_proc(name, roles(1), apply_proc=pid, epoch=1, **kw))

    pid = the_proc(L)["proc"]
    L.reset_world("r2")
    # epoch is now 2
    res, ex = run(L, build_proc("u1", roles(1), apply_proc=pid, epoch=2))
    logs = [r for r in rows(L, "application_log") if r["episode"] == "u1"]
    check("reuse: preconditions held, span ok, goal True -> helpful unit", ex.last_stats.get("proc_app_helpful") == 1 and logs and logs[0]["goal_ok"] in (1, True))
    p = the_proc(L)
    check("the reuse and the re-admission of the same episode are ONE unit (3 units: e0, e1, e2)", p["success_uses"] == 3)

    # failed reuse in a fresh store to count exactly
    def fresh_trusted(nm):
        Lt = new_ltm(tmp, nm)
        run(Lt, build_proc("a", roles(1)))
        Lt.reset_world("r1")
        run(Lt, build_proc("b", roles(1), epoch=1))
        return Lt, the_proc(Lt)["proc"]

    Lt, pid = fresh_trusted("proc-fail")
    Lt.reset_world("r2")
    _, ex = run(Lt, build_proc("f1", roles(1), apply_proc=pid, epoch=2, failed=(1,), exec_ok=False, goal=False))
    p = the_proc(Lt)
    check("reuse with a failed step in the span -> failed-reuse unit", ex.last_stats.get("proc_app_unhelpful") == 1 and p["failed_reuses"] == 1)
    check("failed 1 < success 2 -> still trusted", p["status"] == "trusted")
    ev = evs(Lt, pid)[-1]
    check("failed reuse logged as contradict", ev["mutation"] == "contradict")
    Lt.reset_world("r3")
    run(Lt, build_proc("f2", roles(1), apply_proc=pid, epoch=3, failed=(2,), exec_ok=False, goal=False))
    p = the_proc(Lt)
    check("failed >= success -> demoted", p["failed_reuses"] == 2 and p["status"] == "demoted")
    Lt.reset_world("r4")
    run(Lt, build_proc("f3", roles(1), apply_proc=pid, epoch=4, failed=(1,), exec_ok=False, goal=False))
    Lt.reset_world("r5")
    run(Lt, build_proc("f4", roles(1), apply_proc=pid, epoch=5, failed=(1,), exec_ok=False, goal=False))
    p = the_proc(Lt)
    check("failed >= success + margin -> retired", p["status"] == "retired", str(p))
    ev = evs(Lt, pid)[-1]
    check("retirement logged", ev["mutation"] == "retire")
    # retired procedures are not scored
    Lt.reset_world("r6")
    _, ex = run(Lt, build_proc("f5", roles(1), apply_proc=pid, epoch=6))
    check("a retired procedure application is ineligible", ex.last_stats.get("proc_app_ineligible") == 1)
    p = the_proc(Lt)
    check("...a fresh independent success of the same body lifts it from retired to demoted (3 success vs 4 failed units)",
          p["success_uses"] == 3 and p["failed_reuses"] == 4 and p["status"] == "demoted", str(p))
    check("...and that is logged as a reactivate mutation", any(e["mutation"] == "reactivate" for e in evs(Lt, pid)))

    # reactivation semantics, explicitly
    Lr = new_ltm(tmp, "proc-react")
    run(Lr, build_proc("a", roles(1)))
    pid = the_proc(Lr)["proc"]
    for n in range(1, 4):
        Lr.reset_world(f"r{n}")
        run(Lr, build_proc(f"f{n}", roles(1), apply_proc=pid, epoch=n, failed=(1,), exec_ok=False, goal=False))
    check("1 success vs 3 failed units -> retired", the_proc(Lr)["status"] == "retired")
    for n in range(4, 8):
        Lr.reset_world(f"r{n}")
        run(Lr, build_proc(f"s{n}", roles(1), epoch=n))
    p = the_proc(Lr)
    ev = evs(Lr, pid)
    check("independent successes pull a retired procedure back (documented reactivation)", p["status"] != "retired" and any(e["mutation"] == "reactivate" for e in ev), str(p["status"]))

    # preconditions / span / verdict handling
    Lt, pid = fresh_trusted("proc-pre")
    Lt.reset_world("r2")
    _, ex = run(Lt, build_proc("p1", roles(1), apply_proc=pid, epoch=2, app_pre={"holding": "y"}))
    check("preconditions not holding at application -> neutral (not offered, not counted)", ex.last_stats.get("proc_app_neutral") == 1 and the_proc(Lt)["failed_reuses"] == 0)
    Lt.reset_world("r3")
    _, ex = run(Lt, build_proc("p2", roles(1), apply_proc=pid, epoch=3, app_pre={}))
    check("an unknown ('missing') current value never satisfies a definite requirement", ex.last_stats.get("proc_app_neutral") == 1)
    Lt.reset_world("r4")
    _, ex = run(Lt, build_proc("p3", roles(1), apply_proc=pid, epoch=4, steps=((0, "find", "Mug", "?"), (1, "pickup", "Mug", "?"), (2, "put", "Mug", "Fridge"))))
    check("executed steps differ from the body -> span mismatch, neutral", ex.last_stats.get("proc_app_neutral") == 1)
    Lt.reset_world("r5")
    e = Ep("p4", concepts=roles(1), epoch=5)
    e.app(pid, 0, 0, pre={"holding": "n"})
    for i, verb, ot, to in ((0, "find", "Apple", "?"), (1, "pickup", "Apple", "?"), (2, "put", "Apple", "Fridge")):
        e.step(i, verb, ot, to, pre={"holding": "n"}); e.outcome(i, True)
    e.finish(True, goal=None)
    _, ex = run(Lt, e)
    check("no verdict + span ok -> neutral", ex.last_stats.get("proc_app_neutral") == 1)
    Lt.reset_world("r6")
    e = Ep("p5", concepts=roles(1), epoch=6)
    e.app(pid, 0, 0, pre={"holding": "n"})
    e.step(0, "find", "Apple", "?", pre={"holding": "n"}); e.attempt(0, 0, False, "action_refused"); e.outcome(0, False, "action_refused")
    e.finish(False, goal=None)
    _, ex = run(Lt, e)
    check("an observed failure inside the span counts even when the verdict is None", ex.last_stats.get("proc_app_unhelpful") == 1)
    Lt.reset_world("r7")
    _, ex = run(Lt, build_proc("p6", roles(1), apply_proc=pid, epoch=7, goal=False))
    check("span ok but the privileged goal False -> failed-reuse unit", ex.last_stats.get("proc_app_unhelpful") == 1)
    check("preconditions_hold: '?' requirement imposes nothing", ll.preconditions_hold({"a": "?"}, {}) and ll.preconditions_hold({}, None))
    check("preconditions_hold: mismatch / missing fail", not ll.preconditions_hold({"a": "y"}, {"a": "n"}) and not ll.preconditions_hold({"a": "y"}, None))
    check("procedure_preconditions_hold reads the first body step", ll.procedure_preconditions_hold(the_proc(Lt), {"holding": "n"}) and not ll.procedure_preconditions_hold(the_proc(Lt), {"holding": "y"}))
    check("procedure_status: pure function of the unit counts",
          [ll.procedure_status(*a, 2, 2) for a in ((1, 0), (2, 0), (1, 1), (2, 2), (1, 3), (0, 2))] == ["candidate", "trusted", "demoted", "demoted", "retired", "retired"])

    # procedures never touch lessons / claims
    check("procedure admission wrote no lesson", rows(new_ltm(tmp, "empty"), "lesson") == [])


# ================================================================================================= 8 FLR
def test_flr(tmp):
    w = ll.wilson(0, 10)
    check("wilson: 0/10 lower bound is 0, upper positive", w[0] == 0.0 and 0.2 < w[1] < 0.35)
    w = ll.wilson(5, 10)
    check("wilson: 5/10 centred near 0.5 and ordered", w[0] < 0.5 < w[1] and abs((w[0] + w[1]) / 2 - 0.5) < 0.02)
    check("wilson: empty denominator has no interval", ll.wilson(0, 0) is None)
    sup = [f"L{i}" for i in range(10)]
    lab = {f"L{i}": ("false" if i < 2 else "correct" if i < 7 else "insufficient") for i in range(10)}
    r = ll.supported_flr(sup, lab)
    check("supported-set FLR: false / ALL supported, insufficient stays in the denominator", r["n"] == 10 and r["false"] == 2 and r["insufficient"] == 3 and abs(r["flr"] - 0.2) < 1e-9)
    check("supported-set FLR: decided-only rate reported separately", abs(r["flr_decided"] - 2 / 7) < 1e-9)
    r2 = ll.supported_flr(sup, {k: v for k, v in lab.items() if k != "L9"})
    check("supported-set FLR: an unaudited lesson blocks the claim", not r2["complete"] and r2["flr"] is None and r2["unaudited"] == 1)
    check("supported-set FLR: empty set has no rate", ll.supported_flr([], {})["flr"] is None)

    small = [(f"L{i}", "tentative" if i % 2 else "supported") for i in range(30)]
    pl = ll.audit_plan(small, cap=100, seed=1)
    Lp, lp = fresh_with_lesson(tmp, "flrpop")
    for nm, n in (("s1", 1), ("s2", 2)):
        run(Lp, apply_ep(nm, roles(n), lp["lesson"]))
    fp = ll.flr_population(Lp)
    check("flr_population: supported ids, all-created (id, status), frozen version hashes",
          fp["supported"] == [lp["lesson"]] and fp["created"] == [(lp["lesson"], "supported")] and lp["lesson"] in fp["versions"])
    check("audit plan: n <= cap audits every lesson (inclusion probability 1)", pl["census"] and len(pl["ids"]) == 30 and all(s["p"] == 1.0 for s in pl["strata"].values()))
    pop = ([(f"T{i}", "tentative") for i in range(150)] + [(f"S{i}", "supported") for i in range(100)]
           + [(f"C{i}", "contradicted") for i in range(40)] + [(f"R{i}", "retired") for i in range(10)])
    pl = ll.audit_plan(pop, cap=60, seed=3)
    check("audit plan: stratified, exactly cap lessons, all strata non-zero probability", not pl["census"] and len(pl["ids"]) == 60 and all(s["p"] > 0 for s in pl["strata"].values()) and set(pl["strata"]) == {"tentative", "supported", "contradicted", "retired"})
    check("audit plan: deterministic for a seed, different across seeds", canon(pl) == canon(ll.audit_plan(pop, cap=60, seed=3)) and canon(pl) != canon(ll.audit_plan(pop, cap=60, seed=4)))
    check("audit plan: cap below the number of strata is refused (a zero probability would bias)", raises(ValueError, ll.audit_plan, pop, 3, 1))
    check("audit plan: empty population", ll.audit_plan([], 10, 1)["N"] == 0)
    truth = {i: ("false" if (hash_ := int(i[1:])) % 5 == 0 else "correct") for i, _ in pop}
    strata_of = dict(pop)
    true_rate = sum(1 for v in truth.values() if v == "false") / len(truth)
    ests = []
    for seed in range(300):
        p_ = ll.audit_plan(pop, cap=60, seed=seed)
        e_ = ll.all_created_flr(p_, strata_of, {i: truth[i] for i in p_["ids"]}, boot=50, seed=seed)
        ests.append(e_["flr"])
    check("Horvitz-Thompson estimate is unbiased over seeds (300 draws within 0.01 of truth)", abs(sum(ests) / len(ests) - true_rate) < 0.01, f"{sum(ests) / len(ests):.4f} vs {true_rate:.4f}")
    p_ = ll.audit_plan(pop, cap=60, seed=3)
    e_ = ll.all_created_flr(p_, strata_of, {i: truth[i] for i in p_["ids"]}, boot=400, seed=1)
    check("stratified bootstrap interval brackets the estimate", e_["ci95"] and e_["ci95"][0] <= e_["flr"] <= e_["ci95"][1] and e_["reason"] == "ht_stratified_bootstrap")
    check("bounds (worst / best case) bracket the estimate", e_["bounds"][0] <= e_["flr"] <= e_["bounds"][1])
    check("bootstrap is reproducible for a seed", canon(e_) == canon(ll.all_created_flr(p_, strata_of, {i: truth[i] for i in p_["ids"]}, boot=400, seed=1)))
    sparse = ll.audit_plan(pop + [(f"X{i}", "weird") for i in range(5)], cap=5, seed=2)
    es = ll.all_created_flr(sparse, dict(pop + [(f"X{i}", "weird") for i in range(5)]), {i: "correct" for i in sparse["ids"]}, boot=50)
    check("sparse (n<2) non-census strata: no bootstrap interval, only completion bounds", es["ci95"] is None and es["bounds"] and es["reason"].startswith("sparse_strata"))
    c = ll.all_created_flr(ll.audit_plan(small, 100, 1), dict(small), {i: "false" if i == "L1" else "correct" for i, _ in small})
    check("census: exact rate with a Wilson interval", c["census"] and abs(c["flr"] - 1 / 30) < 1e-9 and c["ci95"])
    check("empty all-created set: no rate", ll.all_created_flr(ll.audit_plan([], 5, 1), {}, {})["flr"] is None)
    check("incomplete audit labels block the all-created claim", ll.all_created_flr(pl, strata_of, {})["reason"] == "unaudited_lessons")
    ha = ll.hypothesis_accuracy(["correct"] * 6 + ["incorrect"] * 2 + ["insufficient"] * 2)
    check("hypothesis accuracy: insufficient reported separately", ha["insufficient"] == 2 and abs(ha["accuracy"] - 0.75) < 1e-9 and ha["ci95"])
    row = {"lesson": "ls:a", "cond_key": "ck:b", "action": "none", "status": "tentative", "helpful_units": 0, "unhelpful_units": 0, "recurrences": 0, "conditions": {"cause": "x"}}
    h1 = ll.lesson_version_hash(row)
    check("lesson version hash is stable and changes with the audited content", h1 == ll.lesson_version_hash(dict(row)) and h1 != ll.lesson_version_hash(dict(row, status="supported")))


# ================================================================================================= 9 hygiene
def test_hygiene(tmp):
    mem = REPO / "sim" / "memory.db"
    m0 = mem.stat().st_mtime if mem.exists() else None
    code = (
        "import sys, socket, urllib.request\n"
        "def boom(*a, **k): raise RuntimeError('network/simulator touched')\n"
        "socket.socket = boom; socket.create_connection = boom; urllib.request.urlopen = boom\n"
        "sys.path.append(%r)\n"
        "import ltm_lessons\n"
        "bad = [m for m in ('ai2thor','requests','ollama','torch','ultralytics','honest','objmem','honest_skills','honest_map') if m in sys.modules]\n"
        "print('BAD' if bad else 'CLEAN', bad)\n" % HERE)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(REPO))
    check("importing ltm_lessons starts no simulator / socket / model client and loads no step-4 module", out.stdout.startswith("CLEAN"), out.stdout + out.stderr)
    src = (REPO / "sim" / "ltm_lessons.py").read_text(encoding="utf-8")
    check("source imports no network / model / simulator library", all(f"import {m}" not in src for m in ("ollama", "requests", "urllib", "socket", "ai2thor", "honest", "objmem", "honest_skills", "honest_map")))
    check("sim/memory.db untouched", (mem.stat().st_mtime if mem.exists() else None) == m0)
    check("default sim/step5/ltm.db never created", not (REPO / "sim" / "step5" / "ltm.db").exists())
    # the extension is a plain callable that adds only lesson / procedure / log / audit tables
    L, Lbase = new_ltm(tmp, "tables"), new_ltm(tmp, "tables-base")
    tabs = ("location_claim", "negative_evidence", "placement", "search_trial", "habit", "place_instance", "place_alias")
    for ep_ in (build_proc("z1", roles(1)), fail_then_sidestep("z2", roles(2))):
        run(L, ep_)
        Lbase.end_episode(ep_.wm())                     # the same episode WITHOUT the lessons extension
    check("lessons + procedures never write location / negative / placement / trial / habit / place rows",
          all(L.store.count(t_) == Lbase.store.count(t_) for t_ in tabs))
    check("every written claim has evidence", all(evs(L, r["lesson"]) for r in rows(L, "lesson")) and
          all(evs(L, r["proc"]) for r in rows(L, "procedure")))
    check("store integrity check clean after all of the above", L.check_integrity() == [])
    check("ops_rejected stayed zero in a clean run", all("ops_rejected" not in (r["evidence"]["counters"]) for r in rows(L, "hypothesis_audit")))
    # contamination: privileged material is refused before it can reach a hypothesis or a lesson
    e = Ep("cont")
    simid = dict(e.base, ev=51, kind="attempt", i=0, k=0, ok=False, cause="Apple|+01.00|+00.90|-01.00|")
    check("an event carrying a simulator id is refused at ingest", raises(EventError, WorkingMemory("cont").ingest, simid))
    bad = dict(e.base, ev=50, kind="attempt", i=0, k=0, ok=False, cause="not_detected", detail="blocked by Apple|1|2|3")
    check("error text on an attempt is refused at ingest", raises(EventError, WorkingMemory("cont").ingest, bad))
    check("the registry values used here exist", all(k in ltm.load_params()["lessons"] for k in ("promote_after", "retire_margin", "hypothesis_max_chars", "default_action", "audit_cap")))
    Pm = copy.deepcopy(ltm.load_params())
    Pm["lessons"]["promote_after"]["value"] = 1
    Lm = new_ltm(tmp, "mut", Pm)
    run(Lm, fail_then_sidestep("o1", roles(0)))
    run(Lm, apply_ep("m1", roles(1), the_lesson(Lm)["lesson"]))
    check("lessons.promote_after comes from the registry (1 -> supported after one unit)", the_lesson(Lm)["status"] == "supported")
    Pm = copy.deepcopy(ltm.load_params())
    Pm["lessons"]["default_action"]["value"]["not_detected"] = "none"
    Lm = new_ltm(tmp, "mut2", Pm)
    run(Lm, fail_then_sidestep("o1", roles(0)))
    check("lessons.default_action comes from the registry (none -> no lesson)", rows(Lm, "lesson") == [])
    Pm = copy.deepcopy(ltm.load_params())
    Pm["procedures"]["trust_after"]["value"] = 1
    Lm = new_ltm(tmp, "mut3", Pm)
    run(Lm, build_proc("o1", roles(1)))
    check("procedures.trust_after comes from the registry (1 -> trusted at admission)", the_proc(Lm)["status"] == "trusted")


def main():
    with tempfile.TemporaryDirectory(prefix="s5lessons-") as tmp:
        for name, fn in [("taxonomy+key", test_taxonomy), ("packet+hypothesis", test_hypothesis), ("lesson creation", test_creation),
                         ("application scoring", test_scoring), ("lifecycle", test_lifecycle), ("revalidation", test_revalidate),
                         ("procedures", test_procedures), ("false-lesson rate", test_flr), ("hygiene", test_hygiene)]:
            before = _fail
            try:
                fn(tmp)
            except Exception as e:                                  # a crash is a failure, with its traceback
                import traceback
                traceback.print_exc()
                check(f"{name}: crashed", False, repr(e))
            print(f"  [{name}] {'ok' if _fail == before else 'FAILURES'}")
        for L in _OPEN:
            try:
                L.close()
            except Exception:
                pass
    print(f"\n{_pass}/{_pass + _fail}")
    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    main()
