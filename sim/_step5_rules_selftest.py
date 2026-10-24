"""Self-test for step 5 / A2-rules -- event contract, working memory, identity, consolidation, decay, habits, where_is.

Offline, Windows `python` from the REPO ROOT, temp directories only (never sim/trace.db, sim/memory.db or the default
sim/step5/ltm.db), no simulator, no WSL, no Ollama, no network:

    python sim/_step5_rules_selftest.py

Covers (docs/step5-plan.md sections 2, 3, 5, 6, 8.1, 11): useful-type eligibility; qualified vs unqualified negatives;
placements vs natural discoveries; contradictions (moved objects, tier demotion); repeated-discovery inflation blocked;
self-placement never reinforcing habits; identity (adjacent identical drawers, abstention, merge only with positive
evidence, unmerge, no double counting); decay monotone + half-life; reset semantics; kitchen vs general habits;
where_is ranking (evidence outranks habits); frozen clock; determinism / replay / idempotence; import hygiene.
"""
import copy
import itertools
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                    # append, not insert: sim/trace.py must not shadow the stdlib `trace`

import ltm                                # noqa: E402
import ltm_consolidate as lc              # noqa: E402
import ltm_decay as dec                   # noqa: E402
import ltm_events as lev                  # noqa: E402
import ltm_habits as hab                  # noqa: E402
import ltm_identity as lid                # noqa: E402
import ltm_retrieve as lrt                # noqa: E402
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
FRIDGE = (1.0, 0.5, 2.0)
COUNTER = (4.0, 0.9, 0.5)
DRAW_A = (3.0, 0.8, 1.0)
DRAW_B = (3.35, 0.8, 1.0)
CAB = (5.0, 1.6, 2.0)


def params(**over):
    P = copy.deepcopy(ltm.load_params())
    P["detector_capability"]["values"] = {"rev1": {"Apple": {"recall": 0.9}, "Mug": {"recall": 0.9},
                                                    "Egg": {"recall": 0.9}, "Knife": {"recall": 0.4}}}
    for path, val in over.items():
        sec, key = path.split("__")
        P[sec][key]["value"] = val
    return P


class Ep:
    """Event builder for one episode."""

    def __init__(self, episode, kitchen="FloorPlan1", epoch=0, frame="f0", mode="practice", world="reset"):
        self.base = dict(schema="s5.ev/1", kitchen=kitchen, epoch=epoch, frame=frame, map_rev="m0", episode=episode)
        self.n = 0
        self.events = []
        self.add("episode_start", mode=mode, world=world)

    def add(self, kind, **kw):
        self.n += 1
        self.events.append(dict(self.base, ev=self.n, kind=kind, t=self.n, **kw))
        return self.n

    def recv(self, rtype, xyz, view="v1", obs=None, covers=None):
        kw = dict(rtype=rtype, xyz=list(xyz), view=view, obs=obs or f"ob:{self.n + 1}")
        if covers:
            kw["covers"] = [list(c) for c in covers]
        return self.add("receptacle_seen", **kw)

    def sight(self, otype, track, in_rtype, in_xyz, basis="inside_inspection", conf=0.9, view="v1", xyz=None):
        return self.add("sighting", otype=otype, track=track, conf=conf, xyz=list(xyz or in_xyz), xyz_ok=True, view=view,
                        obs=f"ob:{self.n + 1}", in_rtype=in_rtype, in_xyz=list(in_xyz), in_basis=basis, detector="rev1")

    def open_(self, rtype, xyz, ok=True):
        return self.add("open", rtype=rtype, xyz=list(xyz), ok=ok, by_me=True, state_seen="open")

    def close(self, rtype, xyz):
        return self.add("close", rtype=rtype, xyz=list(xyz), ok=True, by_me=True, state_seen="closed")

    def insp(self, rtype, xyz, open_ev, targets=("Apple",), detected=(), verdict="visible", frac=0.8, thirds=3,
             support=10, range_m=1.0, rev="rev1"):
        return self.add("inspection", rtype=rtype, xyz=list(xyz), open_ev=open_ev, verdict=verdict, frac=frac,
                        thirds=thirds, support=support, range_m=range_m, views=3, crouched=False,
                        detected=list(detected), targets=list(targets), detector_rev=rev)

    def pickup(self, track, otype, ok=True):
        return self.add("pickup", track=track, otype=otype, ok=ok)

    def put(self, track, otype, rtype, xyz, ok=True, resighted=False):
        return self.add("put", track=track, otype=otype, rtype=rtype, xyz=list(xyz), ok=ok, resighted=resighted)

    def drop(self, track, otype, xyz, ok=True, resighted=False):
        return self.add("drop", track=track, otype=otype, xyz=list(xyz), ok=ok, resighted=resighted)

    def contradiction(self, track, otype, views=2):
        return self.add("contradiction", track=track, otype=otype, views=views, src=[])

    def step(self, otype):
        return self.add("step_start", i=0, verb="find", otype=otype)

    def finish(self, exec_ok=True, goal=None, invalid=False):
        self.add("terminal", exec_ok=exec_ok, agent_verdict="success", prims=5, opens=1, walk_m=3.0, invalid=invalid)
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


def new_ltm(tmp, name, P=None, kitchen="FloorPlan1"):
    L = ltm.LongTermMemory(Path(tmp) / name / "ltm.db", kitchen=kitchen, params=P or params())
    _OPEN.append(L)
    return L


def run(L, ep):
    return L.end_episode(ep.wm())


def rows(L, table, **where):
    return L.store.select(table, where=where or None)


def pos_trials(L, otype=None):
    return [r for r in rows(L, "search_trial") if r["outcome"] == "positive" and (otype is None or r["otype"] == otype)]


def neg_trials(L):
    return [r for r in rows(L, "search_trial") if r["outcome"] == "negative"]


def cand_ids(ans):
    return [c["id"] for c in ans["candidates"]]


def natural_apple_in_fridge(name, **kw):
    e = Ep(name, **kw)
    e.recv("Fridge", FRIDGE)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    return e.finish(goal=True)


# ================================================================================================= 1 events + WM
def test_events(tmp):
    e = Ep("e1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.finish()
    w = e.wm()
    check("wm complete + practice", w.complete and w.mode == "practice" and not w.invalid)
    # contamination is refused at ingest
    bad = [
        ("objectId key", dict(kind="sighting", otype="Apple", track="tr:1", objectId="Apple|1|2|3")),
        ("sim id in token", dict(kind="sighting", otype="Apple|+01.00|+00.90|-01.00", track="tr:1")),
        ("metadata key", dict(kind="open", rtype="Fridge", xyz=[1, 1, 1], metadata={"a": 1})),
        ("group key", dict(kind="task_context", group="g1")),
        ("error text", dict(kind="refusal", verb="open", detail="blocked by something")),
        ("goal predicate", dict(kind="step_outcome", i=0, ok=True, goal="Apple in Fridge")),
        ("oracle state", dict(kind="sighting", otype="Apple", track="tr:2", isOpen=True)),
        ("errors_logged", dict(kind="terminal", exec_ok=True, errors_logged=3)),
        ("goal_ok outside admission", dict(kind="terminal", exec_ok=True, goal_ok=True)),
        ("unknown kind", dict(kind="telepathy")),
        ("unknown field", dict(kind="pickup", track="tr:1", flavour="x")),
        ("wrong type", dict(kind="pickup", track="tr:1", ok="yes")),
        ("bad xyz", dict(kind="receptacle_seen", rtype="Fridge", xyz=[1, 2])),
        ("non-finite", dict(kind="receptacle_seen", rtype="Fridge", xyz=[1, 2, float("nan")])),
        ("free text outside task_context", dict(kind="step_start", i=0, verb="خودت پیدا کن")),
        ("sim id in free text", dict(kind="task_context", text="go to Apple|+01|+02|+03|")),
        ("bad enum", dict(kind="admission_verdict", goal_ok=True, tag="trust-me")),
    ]
    for name, ev in bad:
        w2 = WorkingMemory("e2")
        full = dict(schema="s5.ev/1", ev=1, t=0, kitchen="FloorPlan1", epoch=0, frame="f0", map_rev="m0", episode="e2")
        full.update(ev)
        check(f"refused: {name}", raises(lev.EventError, w2.ingest, full))
    ok_ev = dict(schema="s5.ev/1", ev=1, t=0, kitchen="FloorPlan1", epoch=0, frame="f0", map_rev="m0", episode="e2",
                 kind="task_context", level="L2", lang="fa", text="سیب را در یخچال بگذار", concepts={"verbs": ["put"]})
    check("free text with Persian admitted in task_context", WorkingMemory("e2").ingest(ok_ev)["lang"] == "fa")
    # ordering / duplicates / reset / frame
    w = WorkingMemory("e3")
    base = dict(schema="s5.ev/1", t=0, kitchen="FloorPlan1", epoch=0, frame="f0", map_rev="m0", episode="e3")
    w.ingest(dict(base, ev=1, kind="episode_start", mode="practice", world="reset"))
    w.ingest(dict(base, ev=3, kind="open", rtype="Fridge", xyz=[1, 1, 1], ok=True))
    check("out-of-order ev refused", raises(lev.EventError, w.ingest, dict(base, ev=2, kind="close", rtype="Fridge", xyz=[1, 1, 1])))
    n0 = len(w.events)
    w.ingest(dict(base, ev=3, kind="open", rtype="Fridge", xyz=[1, 1, 1], ok=True))
    check("identical duplicate ignored", len(w.events) == n0)
    check("conflicting duplicate fails", raises(lev.EventError, w.ingest, dict(base, ev=3, kind="open", rtype="Fridge", xyz=[1, 1, 1], ok=False)))
    check("reset mid-episode refused", raises(lev.EventError, w.ingest, dict(base, ev=4, epoch=1, kind="close", rtype="Fridge", xyz=[1, 1, 1])))
    check("frame change mid-episode refused", raises(lev.EventError, w.ingest, dict(base, ev=4, frame="f9", kind="close", rtype="Fridge", xyz=[1, 1, 1])))
    check("other kitchen refused", raises(lev.EventError, w.ingest, dict(base, ev=4, kitchen="FloorPlan6", kind="close", rtype="Fridge", xyz=[1, 1, 1])))
    check("other episode refused", raises(lev.EventError, w.ingest, dict(base, ev=4, episode="zz", kind="close", rtype="Fridge", xyz=[1, 1, 1])))
    # replay: shuffled complete batch -> same working memory; conflicting dup fails
    e = natural_apple_in_fridge("e4")
    sh = list(e.events)
    random.Random(1).shuffle(sh)
    check("replay sorts shuffled batch", [x["ev"] for x in WorkingMemory.replay(sh + sh[:2]).events] == [x["ev"] for x in e.events])
    cf = copy.deepcopy(e.events[1])
    cf["conf"] = 0.1
    check("replay with conflicting duplicate fails", raises(lev.EventError, WorkingMemory.replay, e.events + [dict(cf, ev=e.events[1]["ev"], kind=e.events[1]["kind"])]))
    t = Ep("e6")
    t.step("Mug")
    t.insp("Fridge", FRIDGE, 1, targets=("Egg", "Apple"))
    check("targets()", t.wm().targets() == {"Mug", "Egg", "Apple"})


# ================================================================================================= 2 consolidation basics
def test_basic(tmp):
    L = new_ltm(tmp, "basic")
    r = run(L, natural_apple_in_fridge("ep1"))
    check("committed, E=1", r["status"] == "committed" and r["E"] == 1 and L.clock() == 1, r)
    pl = rows(L, "place_instance")
    check("one place minted pl:1", len(pl) == 1 and pl[0]["place"] == "pl:1" and pl[0]["rtype"] == "Fridge")
    lcs = rows(L, "location_claim")
    check("one natural verified location claim", len(lcs) == 1 and lcs[0]["provenance"] == "natural" and lcs[0]["verified"] == "verified"
          and lcs[0]["status"] == "current" and lcs[0]["contradictions"] == 0 and lcs[0]["E_support"] == 1)
    check("one positive trial + kitchen & general habit rows",
          len(pos_trials(L)) == 1 and {h["level"]: h["active"] for h in rows(L, "habit")} == {"kitchen": 1, "general": 0})
    ep = rows(L, "episodes")[0]
    check("episode row (practice, goal_ok recorded, E)", ep["mode"] == "practice" and ep["goal_ok"] == 1 and ep["E"] == 1 and ep["exec_ok"] == 1)
    check("integrity clean (every claim has evidence)", L.check_integrity() == [], L.check_integrity())
    ev = rows(L, "evidence")
    check("one evidence row per claim mutation + rule ids", len(ev) >= 4 and all(x["rule"] in ("R-ID", "R2", "R3") for x in ev)
          and {x["claim_table"] for x in ev} >= {"place_instance", "location_claim", "search_trial", "habit"})
    ans = L.where_is("Apple")
    c0 = ans["candidates"][0]
    check("where_is: T0 place first with source/age/support/contradictions/status", c0["id"] == "pl:1" and c0["tier"] == "T0"
          and c0["source"] == "location_claim" and c0["age"] == 0 and c0["support"] == 1 and c0["contradictions"] == 0
          and c0["status"] == "current" and "weight" in c0, c0)
    check("where_is: <= where_k candidates and EXPLORE present", len(ans["candidates"]) <= 3 and ans["explore"]["id"] == "EXPLORE"
          and ans["explore"]["unknown"] is True)
    # idempotent retry
    h0 = L.state_hash()
    r2 = run(L, natural_apple_in_fridge("ep1"))
    check("same episode again -> duplicate, nothing written", r2["status"] == "duplicate" and L.state_hash() == h0 and L.clock() == 1)
    # not-eligible episodes write nothing and do not advance E
    for name, ep_, st in [("test", Ep("t1", mode="test"), "skipped_mode"), ("probe", Ep("t2", mode="probe"), "skipped_mode")]:
        ep_.sight("Apple", "tr:1", "Fridge", FRIDGE)
        ep_.finish()
        check(f"{name} mode writes nothing", run(L, ep_)["status"] == st and L.state_hash() == h0)
    inc = Ep("t3")
    inc.sight("Apple", "tr:1", "Fridge", FRIDGE)
    check("incomplete episode writes nothing", run(L, inc)["status"] == "incomplete" and L.state_hash() == h0)
    inv = Ep("t4")
    inv.sight("Apple", "tr:1", "Fridge", FRIDGE)
    inv.finish(invalid=True)
    check("invalid episode consolidates nothing", run(L, inv)["status"] == "invalid" and L.state_hash() == h0)
    wrong = Ep("t5", epoch=3)
    wrong.finish()
    check("epoch not declared via reset_world -> no commit", run(L, wrong)["status"] == "epoch_mismatch" and L.state_hash() == h0)
    other = Ep("t6", kitchen="FloorPlan6")
    other.finish()
    check("other kitchen -> no commit", run(L, other)["status"] == "kitchen_mismatch")
    # an episode with no claim still advances E
    e7 = Ep("ep7")
    e7.finish(goal=False, exec_ok=False)
    r7 = run(L, e7)
    check("claim-less episode still advances E; goal_without_plan counted", r7["E"] == 2 and L.clock() == 2)
    e8 = Ep("ep8")
    e8.finish(goal=True, exec_ok=False)
    r8 = run(L, e8)
    check("goal met with failed step -> stats.goal_without_plan", r8["stats"]["goal_without_plan"] == 1)
    check("privileged goal_ok lives only in the episodes row", all("goal" not in str(r["summary"]) for r in rows(L, "evidence")))
    # R1: non-useful types dropped unless targeted
    L2 = new_ltm(tmp, "r1")
    e = Ep("r1a")
    e.recv("Fridge", FRIDGE)
    e.sight("Toaster", "tr:9", "CounterTop", COUNTER)
    e.sight("Cabinet", "tr:8", "Fridge", FRIDGE)
    e.finish()
    r = run(L2, e)
    check("R1: non-useful types leave no claim", rows(L2, "location_claim") == [] and r["stats"]["ineligible_types"] == 2)
    e = Ep("r1b")
    e.step("Toaster")
    e.sight("Toaster", "tr:9", "CounterTop", COUNTER)
    e.finish()
    run(L2, e)
    check("R1: this task's target is kept even if not on the public list", [c["otype"] for c in rows(L2, "location_claim")] == ["Toaster"])
    # low confidence ignored; threshold comes from the registry
    L3 = new_ltm(tmp, "conf")
    e = Ep("c1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, conf=0.45)
    e.finish()
    run(L3, e)
    check("conf below registry min_conf -> no claim", rows(L3, "location_claim") == [])
    L4 = new_ltm(tmp, "conf2", params(consolidate__min_conf=0.4))
    e = Ep("c1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, conf=0.45)
    e.finish()
    run(L4, e)
    check("lowering consolidate.min_conf in the registry changes the outcome (no hard-coded threshold)", len(rows(L4, "location_claim")) == 1)
    # box_only: trusted claim, no natural trial
    L5 = new_ltm(tmp, "box")
    e = Ep("b1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, basis="box_only")
    e.finish()
    run(L5, e)
    cl = rows(L5, "location_claim")
    check("box_only single view: trusted claim, NO natural trial", len(cl) == 1 and cl[0]["verified"] == "trusted" and pos_trials(L5) == [])
    e = Ep("b2")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, basis="box_only", view="v1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, basis="box_only", view="v2")
    e.finish()
    run(L5, e)
    check("two distinct views -> verified", rows(L5, "location_claim")[0]["verified"] == "verified")


# ================================================================================================= 3 qualified negatives
def neg_episode(name, *, close_before=False, open_first=True, **kw):
    e = Ep(name)
    e.recv("Fridge", FRIDGE)
    o = e.open_("Fridge", FRIDGE) if open_first else None
    if close_before:
        e.close("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o if o is not None else 99, **kw)
    return e.finish()


def test_negatives(tmp):
    cases = {
        "partly": dict(verdict="partly"), "not-visible": dict(verdict="not-visible"), "unsure": dict(verdict="unsure"),
        "found": dict(verdict="found"), "low frac": dict(frac=0.3), "one third": dict(thirds=1),
        "low support": dict(support=2), "far": dict(range_m=2.4), "incapable type": dict(targets=("Knife",)),
        "unknown type": dict(targets=("Mug2",)), "unknown detector rev": dict(rev="revX"),
        "no open event": dict(open_first=False), "closed after open": dict(close_before=True),
    }
    for name, kw in cases.items():
        L = new_ltm(tmp, "neg-" + name.replace(" ", "_"))
        run(L, neg_episode("n1", **kw))
        check(f"unqualified negative ignored: {name}", rows(L, "negative_evidence") == [] and neg_trials(L) == [])
    # registry says no detector is capable -> nothing is ever stored cross-task
    L = new_ltm(tmp, "neg-emptycap", ltm.load_params())
    run(L, neg_episode("n1"))
    check("empty detector_capability table -> no negatives (conservative default)", rows(L, "negative_evidence") == [])
    # qualified
    L = new_ltm(tmp, "neg-ok")
    r = run(L, neg_episode("n1"))
    ng = rows(L, "negative_evidence")
    check("qualified negative stored once, with detector revision + support", len(ng) == 1 and ng[0]["detector_rev"] == "rev1"
          and ng[0]["otype"] == "Apple" and ng[0]["support"] == 10 and ng[0]["superseded"] == 0)
    check("qualified negative = one negative search trial", len(neg_trials(L)) == 1 and pos_trials(L) == [])
    evs = [x for x in rows(L, "evidence") if x["claim_table"] == "negative_evidence"]
    check("negative evidence worded as 'not detected after qualified inspection', never 'empty'",
          len(evs) == 1 and "not_detected_after_qualified_inspection" in evs[0]["summary"] and "empty" not in evs[0]["summary"])
    ans = L.where_is("Apple")
    check("where_is: negative place not offered, listed LAST in EXPLORE as suppressed",
          "pl:1" not in cand_ids(ans) and [s["place"] for s in ans["explore"]["suppressed"]] == ["pl:1"], ans)
    # repeated failed inspections = 1 trial
    for i in range(4):
        run(L, neg_episode(f"nn{i}"))
    check("repeated failed inspections of the same place = 1 negative trial, 1 negative row", len(neg_trials(L)) == 1 and len(rows(L, "negative_evidence")) == 1)
    # a later positive supersedes the negative and wins
    r = run(L, natural_apple_in_fridge("pos1"))
    ng = rows(L, "negative_evidence")[0]
    check("later positive supersedes the negative (flag + surprise counted)", ng["superseded"] == 1 and ng["superseded_by"].startswith("lc:")
          and r["stats"]["negatives_superseded"] == 1 and r["stats"]["surprises"] >= 1)
    check("positive trial added; negative trial kept (history), collapses to positive at read time",
          len(pos_trials(L)) == 1 and len(neg_trials(L)) == 1)
    col = hab.collapse_trials(rows(L, "search_trial"), {})
    check("read-time collapse: positive wins inside one unit", len(col) == 1 and col[0]["outcome"] == "positive")
    check("where_is lists the place again (T0)", cand_ids(L.where_is("Apple"))[:1] == ["pl:1"] and not L.where_is("Apple")["explore"]["suppressed"])
    # positive in the same episode defeats a negative about the same place
    L = new_ltm(tmp, "neg-same")
    e = Ep("s1")
    e.recv("Fridge", FRIDGE)
    o = e.open_("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o, targets=("Apple",), detected=())
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.finish()
    run(L, e)
    check("positive + negative at one place in one episode: positive wins, no negative stored", rows(L, "negative_evidence") == [] and len(pos_trials(L)) == 1)
    # detected type is not a negative
    L = new_ltm(tmp, "neg-det")
    run(L, neg_episode("d1", detected=("Apple",)))
    check("target that WAS detected is not a negative", rows(L, "negative_evidence") == [])
    # negative about a type that is also searched next to a found one
    L = new_ltm(tmp, "neg-two")
    run(L, neg_episode("d1", targets=("Apple", "Egg"), detected=("Apple",)))
    check("only the not-detected target gets a negative", [r["otype"] for r in rows(L, "negative_evidence")] == ["Egg"])
    # agent-disturbed place: the agent removed the apple, then inspected -> no negative
    L = new_ltm(tmp, "neg-disturbed")
    e = Ep("x1")
    e.recv("Fridge", FRIDGE)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.pickup("tr:1", "Apple")
    o = e.open_("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o)
    e.finish()
    run(L, e)
    check("agent-disturbed place: no qualified negative, positive discovery kept", rows(L, "negative_evidence") == [] and len(pos_trials(L)) == 1)
    # old-epoch negative does not suppress a new epoch
    L = new_ltm(tmp, "neg-epoch")
    run(L, neg_episode("n1"))
    L.reset_world("r1")
    ans = L.where_is("Apple")
    check("a negative of an earlier epoch is kept but never suppresses the new epoch", len(rows(L, "negative_evidence")) == 1
          and ans["explore"]["suppressed"] == [])


# ================================================================================================= 4 inflation + self placement
def test_inflation(tmp):
    L = new_ltm(tmp, "infl")
    e = Ep("i1")
    e.recv("Fridge", FRIDGE)
    for i in range(100):
        e.sight("Apple", "tr:1", "Fridge", FRIDGE, view=f"v{i}")
    e.finish()
    r = run(L, e)
    check("100 frames of one object = 1 trial, 1 claim", len(pos_trials(L)) == 1 and len(rows(L, "location_claim")) == 1 and r["stats"]["trials_pos"] == 1)
    for i in range(5):
        run(L, natural_apple_in_fridge(f"rev{i}"))
    check("5 revisits in the same epoch = still 1 trial (unit U)", len(pos_trials(L)) == 1)
    check("revisits refresh the claim (E_support advances), do not duplicate it",
          len(rows(L, "location_claim")) == 1 and rows(L, "location_claim")[0]["E_support"] == L.clock())
    pf = L.habits("Apple")["kitchen"]
    fr = [x for x in pf if x["rtype"] == "Fridge"][0]
    check("habit counts distinct discoveries only (n_pos == 1)", fr["n_pos"] == 1)
    two_tracks = Ep("i2")
    two_tracks.sight("Apple", "tr:7", "Fridge", FRIDGE)
    two_tracks.sight("Apple", "tr:8", "Fridge", FRIDGE)
    two_tracks.finish()
    run(L, two_tracks)
    check("two objects of one type at one place in one epoch = still 1 unit", len(pos_trials(L)) == 1)
    L.reset_world("w2")
    run(L, natural_apple_in_fridge("ep-new-epoch", epoch=1))
    check("a new world epoch is a new unit (2nd independent discovery)", len(pos_trials(L)) == 2)
    # resets of an unchanged scene cannot create unlimited strength: still one trial per epoch and place
    for i in range(3):
        run(L, natural_apple_in_fridge(f"again{i}", epoch=1))
    check("repeats inside the new epoch add nothing", len(pos_trials(L)) == 2)

    # self placement never reinforces natural habits
    L = new_ltm(tmp, "selfp")
    e = Ep("s1")
    e.recv("Fridge", FRIDGE)
    e.recv("CounterTop", COUNTER)
    e.sight("Apple", "tr:1", "CounterTop", COUNTER, view="v1")
    e.pickup("tr:1", "Apple")
    e.put("tr:1", "Apple", "Fridge", FRIDGE, resighted=True)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, view="v2")
    e.finish()
    r = run(L, e)
    pt = pos_trials(L)
    check("natural discovery on the counter counted once (even though picked up later); NOTHING for the fridge",
          [t["rtype"] for t in pt] == ["CounterTop"], pt)
    plc = rows(L, "placement")
    check("verified placement stored in its own table", len(plc) == 1 and plc[0]["provenance"] == "verified" and plc[0]["otype"] == "Apple")
    lcs = {c["place"]: c for c in rows(L, "location_claim")}
    fridge_pl = [p["place"] for p in rows(L, "place_instance") if p["rtype"] == "Fridge"][0]
    check("re-sighted placement becomes a location claim with agent_placed provenance", lcs[fridge_pl]["provenance"] == "agent_placed")
    prefs = {x["rtype"]: x for x in L.habits("Apple")["kitchen"]}
    check("kitchen habit has no Fridge entry from self-placement", "Fridge" not in prefs and prefs["CounterTop"]["n_pos"] == 1)
    # ambiguous reacquisition: NEW track of the same type near an own placement -> agent_placed, no natural trial
    L = new_ltm(tmp, "reacq")
    e = Ep("q1")
    e.recv("Fridge", FRIDGE)
    e.put("tr:1", "Apple", "Fridge", FRIDGE, resighted=False)
    e.sight("Apple", "tr:99", "Fridge", FRIDGE, xyz=(1.2, 0.5, 2.1))
    e.finish()
    run(L, e)
    check("unseen put -> actuator placement, no location claim from the put itself", [p["provenance"] for p in rows(L, "placement")] == ["actuator"])
    check("ambiguous reacquisition: agent_placed claim, NO natural trial", pos_trials(L) == []
          and [c["provenance"] for c in rows(L, "location_claim")] == ["agent_placed"])
    # far from the own placement the same sighting is natural
    L = new_ltm(tmp, "reacq2")
    e = Ep("q2")
    e.recv("Fridge", FRIDGE)
    e.put("tr:1", "Apple", "CounterTop", COUNTER)
    e.sight("Apple", "tr:99", "Fridge", FRIDGE, xyz=(1.0, 0.5, 2.0))
    e.finish()
    run(L, e)
    check("a sighting far from any own placement is a natural discovery", len(pos_trials(L)) == 1)
    # drop without re-detection: uncertain placement; with later resight: attaches to the place
    L = new_ltm(tmp, "drop")
    e = Ep("d1")
    e.drop("tr:1", "Apple", (2.0, 0.0, 2.0))
    e.finish()
    run(L, e)
    p0 = rows(L, "placement")
    check("drop without re-detection = uncertain (actuator) placement without a place", len(p0) == 1 and p0[0]["provenance"] == "actuator" and p0[0]["place"] is None)
    a = L.where_is("Apple")["candidates"][0]
    check("where_is offers it as T1 by xyz", a["tier"] == "T1" and a["kind"] == "xyz" and a["source"] == "placement")
    # placements outrank habits but not current evidence
    L = new_ltm(tmp, "tiers")
    run(L, natural_apple_in_fridge("t1"))
    e = Ep("t2")
    e.recv("CounterTop", COUNTER)
    e.put("tr:5", "Apple", "CounterTop", COUNTER, resighted=False)
    e.finish()
    run(L, e)
    ids = L.where_is("Apple")["candidates"]
    check("T0 current evidence > T1 placement > T3 habit/T4 prior", [c["tier"] for c in ids][:2] == ["T0", "T1"] and ids[0]["source"] == "location_claim")
    check("habit/prior entries fill the remaining slots after evidence", all(c["tier"] in ("T3", "T3b", "T4") for c in ids[2:]))


# ================================================================================================= 5 moved objects + contradictions
def test_moved(tmp):
    L = new_ltm(tmp, "moved")
    run(L, natural_apple_in_fridge("m1"))
    e = Ep("m2")
    e.recv("Fridge", FRIDGE)
    e.recv("CounterTop", COUNTER)
    o = e.open_("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o, targets=("Apple",), detected=())
    e.sight("Apple", "tr:1", "CounterTop", COUNTER, basis="depth_consistent")
    r = run(L, e.finish())
    old = [c for c in rows(L, "location_claim") if c["place"] == "pl:1"][0]
    new = [c for c in rows(L, "location_claim") if c["place"] != "pl:1"][0]
    check("moved object: old claim contradicted once (c=1), new claim current c=0", old["contradictions"] == 1 and new["contradictions"] == 0
          and new["status"] == "current" and r["stats"]["claims_contradicted"] == 1)
    ev = [x for x in rows(L, "evidence") if x["mutation"] == "contradict"]
    check("contradiction has its own evidence row (R6)", len(ev) == 1 and ev[0]["rule"] == "R6")
    ans = L.where_is("Apple")
    check("new location outranks the contradicted one; contradicted place suppressed to EXPLORE", cand_ids(ans)[0] == new["place"]
          and "pl:1" not in cand_ids(ans) and ans["explore"]["suppressed"][0]["place"] == "pl:1")
    # a natural positive elsewhere is NOT displacement: no contradiction without absence
    L = new_ltm(tmp, "multi")
    run(L, natural_apple_in_fridge("m1"))
    e = Ep("m2")
    e.recv("CounterTop", COUNTER)
    e.sight("Apple", "tr:2", "CounterTop", COUNTER)
    run(L, e.finish())
    check("a second apple elsewhere is not displacement (c stays 0)", all(c["contradictions"] == 0 for c in rows(L, "location_claim")) and len(rows(L, "location_claim")) == 2)
    # tier demotion by contradiction count via track-level contradictions
    L = new_ltm(tmp, "tiers2")
    run(L, natural_apple_in_fridge("c0"))
    e = Ep("c1")
    e.recv("CounterTop", COUNTER)
    e.sight("Apple", "tr:5", "CounterTop", COUNTER)
    run(L, e.finish())
    cands = {c["id"]: c for c in L.where_is("Apple")["candidates"]}
    check("before: both claims T0", all(c["tier"] == "T0" for c in cands.values() if c["kind"] == "place"))
    for i in (1, 2):
        e = Ep(f"tc{i}")
        e.sight("Apple", "tr:1", "Fridge", FRIDGE)
        e.contradiction("tr:1", "Apple")
        run(L, e.finish())
        c = [x for x in rows(L, "location_claim") if x["place"] == "pl:1"][0]
        check(f"track contradiction {i}: c == {i}", c["contradictions"] == i)
        a = {x["id"]: x for x in L.where_is("Apple")["candidates"]}
        tier = a["pl:1"]["tier"] if "pl:1" in a else None
        check(f"c={i} -> tier {'T2' if i == 1 else 'T3b'} (demotion by tier, not only weight)", tier == ("T2" if i == 1 else "T3b"), a)
    order = [c["tier"] for c in L.where_is("Apple")["candidates"]]
    check("doubted (T3b) ranks after kitchen habits and before the fixed prior", order == sorted(order, key=lambda t: dec.TIER_RANK[t]) and "T3b" in order)
    # support after contradiction resets c
    e = Ep("sup")
    e.recv("Fridge", FRIDGE)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    run(L, e.finish())
    c = [x for x in rows(L, "location_claim") if x["place"] == "pl:1"][0]
    check("fresh grounded support resets the contradiction count", c["contradictions"] == 0)
    # dedupe: qualified negative AND track contradiction in one episode = ONE contradiction
    L = new_ltm(tmp, "dedupe")
    run(L, natural_apple_in_fridge("d0"))
    e = Ep("d1")
    e.recv("Fridge", FRIDGE)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE, basis="box_only", conf=0.9)
    e.contradiction("tr:1", "Apple")
    o = e.open_("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o, targets=("Apple",))
    run(L, e.finish())
    check("negative + track contradiction in one episode = exactly one increment", rows(L, "location_claim")[0]["contradictions"] == 1)
    # another object of the type still there: a track vanishing proves nothing about the type
    L = new_ltm(tmp, "other")
    run(L, natural_apple_in_fridge("o0"))
    e = Ep("o1")
    e.recv("Fridge", FRIDGE)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.sight("Apple", "tr:2", "Fridge", FRIDGE)
    e.contradiction("tr:1", "Apple")
    run(L, e.finish())
    check("track vanished but a same-type object is still there: no contradiction", rows(L, "location_claim")[0]["contradictions"] == 0)
    # fewer than 2 views: ignored
    L = new_ltm(tmp, "oneview")
    run(L, natural_apple_in_fridge("v0"))
    e = Ep("v1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.contradiction("tr:1", "Apple", views=1)
    run(L, e.finish())
    check("contradiction with a single view is not enough (objmem miss rule)", rows(L, "location_claim")[0]["contradictions"] == 0)
    # pickup supersedes the location
    L = new_ltm(tmp, "pick")
    run(L, natural_apple_in_fridge("p0"))
    e = Ep("p1")
    e.recv("Fridge", FRIDGE)
    e.recv("CounterTop", COUNTER)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.pickup("tr:1", "Apple")
    e.put("tr:1", "Apple", "CounterTop", COUNTER, resighted=False)
    r = run(L, e.finish())
    cl = rows(L, "location_claim")
    check("pickup supersedes the old location (retired, not deleted)", len(cl) == 1 and cl[0]["status"] == "retired" and r["stats"]["claims_retired"] == 1)
    check("re-sighting at the old unit in the same epoch added no 2nd trial", len(pos_trials(L)) == 1)
    a = L.where_is("Apple")["candidates"]
    check("where_is: retired claim not offered; unseen put offered as T1 actuator", a[0]["tier"] == "T1" and a[0]["provenance"] == "agent_placed/actuator" and "pl:1" not in cand_ids(L.where_is("Apple")))
    ev = [x for x in rows(L, "evidence") if x["mutation"] == "supersede"]
    check("supersede evidence recorded", len(ev) == 1 and ev[0]["claim_table"] == "location_claim")
    # a later sighting reactivates it
    run(L, natural_apple_in_fridge("p2"))
    check("a new support reactivates the retired claim", rows(L, "location_claim")[0]["status"] == "current"
          and any(x["mutation"] == "reactivate" for x in rows(L, "evidence")))
    # held tracks produce nothing
    L = new_ltm(tmp, "held")
    e = Ep("h1")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.pickup("tr:1", "Apple")
    e.sight("Apple", "tr:1", "CounterTop", COUNTER)
    run(L, e.finish())
    check("held tracks produce no claim", rows(L, "location_claim") == [])
    # failed pickup changes nothing
    L = new_ltm(tmp, "pickfail")
    e = Ep("h2")
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.pickup("tr:1", "Apple", ok=False)
    run(L, e.finish())
    check("failed pickup: location kept", len(rows(L, "location_claim")) == 1)
    # a qualified negative at a place holding the agent's own placement -> historical
    L = new_ltm(tmp, "plneg")
    e = Ep("pn1")
    e.recv("Fridge", FRIDGE)
    e.put("tr:1", "Apple", "Fridge", FRIDGE)
    run(L, e.finish())
    run(L, neg_episode("pn2"))
    check("qualified absence turns the agent's old placement historical", rows(L, "placement")[0]["status"] == "historical")


# ================================================================================================= 6 identity
def test_identity(tmp):
    # adjacent identical drawers, alone, in different episodes
    L = new_ltm(tmp, "id1")
    for i, c in enumerate((DRAW_A, DRAW_B)):
        e = Ep(f"a{i}")
        e.recv("Drawer", c)
        run(L, e.finish())
    check("adjacent drawers (0.35 m) seen alone in different episodes -> two places", len(rows(L, "place_instance")) == 2)
    # same view, two boxes closer than the gate: two places + separation witness; a spanning box does NOT merge them
    L = new_ltm(tmp, "id2")
    a, b = (3.0, 0.8, 1.0), (3.12, 0.8, 1.0)
    e = Ep("s1")
    e.recv("Drawer", a, view="v1", obs="ob:a")
    e.recv("Drawer", b, view="v1", obs="ob:b")
    run(L, e.finish())
    pls = rows(L, "place_instance")
    check("two boxes in one view (0.12 m apart, inside the 0.2 m gate) stay two places", len(pls) == 2)
    check("separation witness persisted in both extents", all(set(p["extent"].get("sep", [])) == {q["place"] for q in pls} - {p["place"]} for p in pls))
    e = Ep("s2")
    e.recv("Drawer", (3.06, 0.8, 1.0), view="v9", obs="ob:big", covers=[a, b])
    r = run(L, e.finish())
    check("oversized box spanning both: NO merge (witness blocks it); refusal counted", rows(L, "place_alias") == [] and r["stats"]["merges_refused"] == 1 and r["stats"]["merges"] == 0)
    # determinism over detection order inside a view
    hs = []
    for order in itertools.permutations([0, 1, 2]):
        Lx = new_ltm(tmp, "perm-" + "".join(map(str, order)))
        boxes = [("Drawer", (3.0, 0.8, 1.0)), ("Drawer", (3.3, 0.8, 1.0)), ("Drawer", (3.62, 0.8, 1.0))]
        e = Ep("p1")
        for i in order:
            e.recv(boxes[i][0], boxes[i][1], view="v1", obs=f"ob:{i}")
        run(Lx, e.finish())
        hs.append(sorted((p["place"], round(p["cx"], 2)) for p in rows(Lx, "place_instance")))
    check("association is independent of detection order within a view", all(h == hs[0] for h in hs) and len(hs[0]) == 3, hs[:2])
    # abstain: a lone sighting midway between two drawers
    L = new_ltm(tmp, "id3")
    for i, c in enumerate((DRAW_A, DRAW_B)):
        e = Ep(f"a{i}")
        e.recv("Drawer", c)
        run(L, e.finish())
    n0 = len(rows(L, "place_instance"))
    e = Ep("mid")
    e.recv("Drawer", (3.175, 0.8, 1.0))
    e.sight("Apple", "tr:1", "Drawer", (3.175, 0.8, 1.0))
    r = run(L, e.finish())
    check("equidistant between two drawers: ABSTAIN (no new place, no claim, no trial)", len(rows(L, "place_instance")) == n0
          and rows(L, "location_claim") == [] and pos_trials(L) == [] and r["stats"]["abstained"] >= 1, r["stats"])
    # near one clearly: associate
    e = Ep("near")
    e.recv("Drawer", (3.03, 0.8, 1.0))
    run(L, e.finish())
    check("clearly nearer one drawer: associate, no new place", len(rows(L, "place_instance")) == n0)
    # merge needs positive evidence and obeys the registry gates
    P = params()
    P["identity"]["place_gate_m"]["value"]["Drawer"] = 0.1
    L = new_ltm(tmp, "id4", P)
    c1, c2 = (3.0, 0.8, 1.0), (3.12, 0.8, 1.0)
    for i, c in enumerate((c1, c2)):
        e = Ep(f"m{i}")
        e.recv("Drawer", c)
        e.sight("Egg", f"tr:{i}", "Drawer", c)
        run(L, e.finish())
    check("two places 0.12 m apart created in different episodes (gate 0.1)", len(rows(L, "place_instance")) == 2 and len(pos_trials(L, "Egg")) == 2)
    check("before merge both units count in the habit", L.habits("Egg")["kitchen"][0]["n_pos"] == 2)
    e = Ep("nospan")
    e.recv("Drawer", c1)
    run(L, e.finish())
    check("no spanning box -> no merge", rows(L, "place_alias") == [])
    e = Ep("far")
    e.recv("Drawer", (3.0, 0.8, 1.0), view="vx", obs="ob:z", covers=[(3.0, 0.8, 1.0), (3.6, 0.8, 1.0)])
    run(L, e.finish())
    check("covered centre too far / not a place -> no merge", rows(L, "place_alias") == [])
    e = Ep("span")
    e.recv("Drawer", (3.06, 0.8, 1.0), view="vs", obs="ob:s", covers=[c1, c2])
    r = run(L, e.finish())
    al = rows(L, "place_alias")
    check("one box spanning both centres, no separation witness -> merged; root = lowest id", len(al) == 1 and al[0]["place"] == "pl:2"
          and al[0]["canonical"] == "pl:1" and al[0]["active"] == 1 and r["stats"]["merges"] == 1)
    check("merge has evidence", any(x["mutation"] == "merge" for x in rows(L, "evidence")))
    check("claims keep their original place ids (history untouched)", {c["place"] for c in rows(L, "location_claim")} == {"pl:1", "pl:2"})
    check("habit: two original trials collapse to ONE unit after the merge (no double count)", L.habits("Egg")["kitchen"][0]["n_pos"] == 1)
    ans = L.where_is("Egg")
    check("where_is: merged places appear once", cand_ids(ans).count("pl:1") == 1 and "pl:2" not in cand_ids(ans))
    e = Ep("mnew")
    e.recv("Drawer", c2)
    e.sight("Egg", "tr:7", "Drawer", c2)
    run(L, e.finish())
    check("a new sighting at the loser resolves to the canonical place (no third place)", len(rows(L, "place_instance")) == 2)
    # unmerge: reversible, nothing lost, counts recompute, history kept, witness prevents re-merging
    n_ev, n_wl = len(rows(L, "evidence")), len(rows(L, "write_log"))
    r = L.unmerge("pl:2", "reviewed_by_hand")
    check("unmerge commits", r["status"] == "committed")
    check("alias inactive; evidence + write_log only grow", rows(L, "place_alias")[0]["active"] == 0 and len(rows(L, "evidence")) > n_ev and len(rows(L, "write_log")) > n_wl
          and any(x["mutation"] == "unmerge" for x in rows(L, "evidence")))
    check("after unmerge both claims survive, ids unchanged", {c["place"] for c in rows(L, "location_claim")} == {"pl:1", "pl:2"})
    check("habit counts follow canonical places again (2 units)", L.habits("Egg")["kitchen"][0]["n_pos"] == 2)
    e = Ep("remerge")
    e.recv("Drawer", (3.06, 0.8, 1.0), view="vs", obs="ob:s", covers=[c1, c2])
    r = run(L, e.finish())
    check("witness added by unmerge blocks re-merging", r["stats"]["merges"] == 0 and rows(L, "place_alias")[0]["active"] == 0)
    check("unmerge of a non-merged place is a no-op", L.unmerge("pl:1")["status"] == "no_active_alias")
    check("integrity after merge/unmerge", L.check_integrity() == [], L.check_integrity())
    # incompatible types / different registry gate
    P2 = params()
    P2["identity"]["place_gate_m"]["value"]["Drawer"] = 0.1
    P2["identity"]["place_gate_m"]["value"]["Cabinet"] = 0.1
    book = lid.PlaceBook(P2, [dict(place="pl:1", rtype="Drawer", cx=0, cy=0, cz=0, extent={}), dict(place="pl:2", rtype="Cabinet", cx=0.05, cy=0, cz=0, extent={})])
    ok, why = book.try_merge("pl:1", "pl:2", True)
    check("different receptacle types never merge", not ok and why == "incompatible_type")
    ok, why = book.try_merge("pl:1", "pl:1", True)
    check("a place cannot merge with its own component", not ok)
    # merge gate comes from the registry
    P3 = params(identity__merge_gate_m=0.05)
    P3["identity"]["place_gate_m"]["value"]["Drawer"] = 0.1
    L = new_ltm(tmp, "id5", P3)
    for i, c in enumerate((c1, c2)):
        e = Ep(f"m{i}")
        e.recv("Drawer", c)
        run(L, e.finish())
    e = Ep("span")
    e.recv("Drawer", (3.06, 0.8, 1.0), view="vs", obs="ob:s", covers=[c1, c2])
    run(L, e.finish())
    check("tightening identity.merge_gate_m in the registry blocks the same merge", rows(L, "place_alias") == [])
    # alias components: separation against ANY member
    book = lid.PlaceBook(P, [dict(place=f"pl:{i}", rtype="Drawer", cx=0.02 * i, cy=0, cz=0, extent={}) for i in (1, 2, 3)])
    book.alias["pl:2"] = "pl:1"
    book.places["pl:3"].sep.add("pl:2")
    book.places["pl:2"].sep.add("pl:3")
    ok, why = book.try_merge("pl:1", "pl:3", True)
    check("separation is checked across EVERY member of both alias components", not ok and why == "separation_witness")
    # regions: points on one counter at one height form one region; another height is another surface
    L = new_ltm(tmp, "region")
    e = Ep("r1")
    for i, x in enumerate((4.0, 4.4, 4.9, 5.4)):
        e.sight("Apple", f"tr:{i}", "CounterTop", (x, 0.9, 0.5), view=f"v{i}")
    e.sight("Mug", "tr:50", "CounterTop", (4.5, 1.6, 0.5), view="v9")
    run(L, e.finish())
    pl = [p for p in rows(L, "place_instance") if p["rtype"] == "CounterTop"]
    check("chained points (<0.6 m links) = one region; a different height = another surface", len(pl) == 2)
    reg = [p for p in pl if abs(p["cy"] - 0.9) < 0.01][0]
    check("region extent grows with linked points", reg["extent"]["x1"] - reg["extent"]["x0"] >= 1.39, reg["extent"])
    check("region claims: one claim per type at the region", len([c for c in rows(L, "location_claim") if c["otype"] == "Apple"]) == 1)
    # distinct fronts opened separately are a separation witness
    L = new_ltm(tmp, "opens")
    e = Ep("o1")
    e.recv("Drawer", (3.0, 0.8, 1.0), view="v1", obs="o:a")
    e.open_("Drawer", (3.0, 0.8, 1.0))
    e.recv("Drawer", (3.4, 0.8, 1.0), view="v2", obs="o:b")
    e.open_("Drawer", (3.4, 0.8, 1.0))
    run(L, e.finish())
    pls = rows(L, "place_instance")
    check("two fronts opened in one episode = separation witness", len(pls) == 2 and all(p["extent"].get("sep") for p in pls))


# ================================================================================================= 7 decay
def test_decay(tmp):
    P = params()
    for kind in ("location", "placement", "negative", "habit", "lesson", "procedure"):
        H = dec.half_life(P, kind)
        v = "verified" if kind in ("location", "placement") else ("supported" if kind == "lesson" else ("trusted" if kind == "procedure" else None))
        w0 = dec.w0(P, kind, v)
        check(f"half-life {kind}: w(H) == w0/2", abs(dec.weight(P, kind, v, 10 + H, 10) - w0 / 2) < 1e-12)
        check(f"half-life {kind}: w(2H) == w0/4", abs(dec.weight(P, kind, v, 10 + 2 * H, 10) - w0 / 4) < 1e-12)
        ws = [dec.weight(P, kind, v, e, 0) for e in range(0, 200)]
        check(f"decay {kind}: strictly decreasing", all(a > b for a, b in zip(ws, ws[1:])))
        check(f"decay {kind}: age 0 == w0", abs(ws[0] - w0) < 1e-12)
    check("declared half-lives are the spec's (3, 2, 1, 30, 20, 50)", [dec.half_life(P, k) for k in ("location", "placement", "negative", "habit", "lesson", "procedure")] == [3, 2, 1, 30, 20, 50])
    check("each contradiction multiplies by gamma", abs(dec.weight(P, "location", "verified", 5, 5, c=2) - 0.25) < 1e-12)
    check("each reset multiplies by rho (locations, placements only)", abs(dec.weight(P, "location", "verified", 5, 5, n_resets=2) - 0.25) < 1e-12
          and abs(dec.weight(P, "habit", None, 5, 5, n_resets=2) - 1.0) < 1e-12 and abs(dec.weight(P, "negative", None, 5, 5, n_resets=2) - 1.0) < 1e-12)
    check("weights ordered by verification: verified > trusted; placement verified > actuator",
          dec.weight(P, "location", "verified", 0, 0) > dec.weight(P, "location", "trusted", 0, 0)
          and dec.weight(P, "placement", "verified", 0, 0) > dec.weight(P, "placement", "actuator", 0, 0))
    check("clamped: explicit clock older than the support never exceeds w0", dec.weight(P, "location", "verified", 0, 9) == 1.0)
    # half-lives come from the registry
    P2 = params()
    P2["decay"]["H_location"]["value"] = 6
    check("registry H_location is used (no literal)", abs(dec.weight(P2, "location", "verified", 6, 0) - 0.5) < 1e-12)
    # retirement is computed on read, nothing deleted
    L = new_ltm(tmp, "decay")
    run(L, natural_apple_in_fridge("d1"))
    for i in range(30):
        e = Ep(f"fill{i}")
        e.finish()
        run(L, e)
    row0 = dict(rows(L, "location_claim")[0])
    n_claims = len(rows(L, "location_claim"))
    ages = {}
    for e_now in (1, 4, 7, 10, 13, 16, 40):
        a = L.where_is("Apple", e_now=e_now)
        loc = [c for c in a["candidates"] if c["source"] == "location_claim"]
        ages[e_now] = loc[0]["weight"] if loc else None
    exp = lambda en: dec.weight(P, "location", "verified", en, 1)
    check("where_is weights follow the formula at several clocks", all(abs(ages[e] - exp(e)) < 1e-5 for e in (1, 4, 7, 10, 13)), ages)
    check("retired from retrieval once w < retire_below (computed on read)", ages[13] is not None and ages[16] is None and ages[40] is None
          and exp(13) >= 0.05 > exp(14))
    check("retired claim still in the database, unchanged", len(rows(L, "location_claim")) == n_claims and dict(rows(L, "location_claim")[0]) == row0)
    a = L.where_is("Apple", e_now=40)
    check("after retirement habits / prior still answer", len(a["candidates"]) >= 1 and all(c["source"] != "location_claim" for c in a["candidates"]))
    # new support reactivates
    run(L, natural_apple_in_fridge("again"))
    check("new support brings it back", L.where_is("Apple")["candidates"][0]["source"] == "location_claim")
    # decay is monotone across the whole where_is for fixed state
    ws = []
    for e_now in range(L.clock(), L.clock() + 12):
        c = [x for x in L.where_is("Apple", e_now=e_now)["candidates"] if x["source"] == "location_claim"]
        ws.append(c[0]["weight"] if c else 0.0)
    check("where_is weight is monotone non-increasing in the clock", all(a >= b for a, b in zip(ws, ws[1:])))
    # habits decay slower than locations
    ph = L.habits("Apple", e_now=L.clock() + 3)["kitchen"][0]
    check("habit P decays with H_habit (3 episodes later)", abs(ph["P"] - 2 ** (-(L.clock() + 3 - rows(L, "search_trial")[0]["E"]) / 30)) < 1e-9)
    # no wall clock
    real = time.time
    time.time = lambda: (_ for _ in ()).throw(RuntimeError("wall clock read"))
    try:
        L.where_is("Apple")
        okc = True
    except RuntimeError:
        okc = False
    finally:
        time.time = real
    check("where_is never reads the wall clock", okc)


# ================================================================================================= 8 reset semantics
def test_reset(tmp):
    L = new_ltm(tmp, "reset")
    run(L, natural_apple_in_fridge("r1"))
    e = Ep("r2")
    e.recv("CounterTop", COUNTER)
    e.put("tr:9", "Egg", "CounterTop", COUNTER)
    run(L, e.finish())
    h_before = L.habits("Apple")["kitchen"]
    clock = L.clock()
    check("before reset: current location T0, placement T1", L.where_is("Apple")["candidates"][0]["tier"] == "T0" and L.where_is("Egg")["candidates"][0]["tier"] == "T1")
    r = L.reset_world("w1")
    check("reset_world: epoch +1, clock unchanged", r["epoch"] == 1 and L.clock() == clock)
    check("reset is idempotent per reset_id", L.reset_world("w1")["status"] == "duplicate" and L.current_epoch() == 1)
    c = L.where_is("Apple")["candidates"][0]
    check("locations become 'maybe' (T2) from the next epoch, weight lowered by rho",
          c["tier"] == "T2" and c["status"] == "maybe" and abs(c["weight"] - dec.weight(L.params, "location", "verified", clock, 1, 0, 1)) < 1e-5, c)
    check("earlier-epoch placements are historical: not offered", all(x["source"] != "placement" for x in L.where_is("Egg")["candidates"]))
    check("habits unchanged by the reset", L.habits("Apple")["kitchen"] == h_before)
    check("claims not deleted", len(rows(L, "location_claim")) == 1 and len(rows(L, "placement")) == 1)
    e = natural_apple_in_fridge("r3", epoch=1)
    run(L, e)
    c = L.where_is("Apple")["candidates"][0]
    check("re-observed in the new epoch: current again (T0), reset penalty cleared by fresh support", c["tier"] == "T0" and abs(c["weight"] - 1.0) < 1e-9)
    # persistent world: no reset, claims stay current and decay with E
    L = new_ltm(tmp, "persist")
    run(L, natural_apple_in_fridge("p1", world="persistent"))
    for i in range(2):
        e = Ep(f"pf{i}", world="persistent")
        e.finish()
        run(L, e)
    c = L.where_is("Apple")["candidates"][0]
    check("persistent world: stays T0 and decays with E only", c["tier"] == "T0" and abs(c["weight"] - 2 ** (-2 / 3)) < 1e-5 and c["age"] == 2)
    # frame change -> maybe
    L = new_ltm(tmp, "frame")
    run(L, natural_apple_in_fridge("f1"))
    L.reset_world("fr", frame="f7")
    run(L, Ep("f2", epoch=1, frame="f7").finish())
    c = L.where_is("Apple")["candidates"][0]
    check("a new coordinate frame also makes old geometry 'maybe'", c["tier"] == "T2" and c["status"] == "maybe")
    # clear_memory
    L = new_ltm(tmp, "clr")
    run(L, natural_apple_in_fridge("c1"))
    p = Path(tmp) / "clr" / "ltm.db"
    L.close()
    ltm.LongTermMemory.clear_memory(p)
    L2 = ltm.LongTermMemory(p, kitchen="FloorPlan1", params=params())
    check("clear_memory (arm none): empty store, clock 0", L2.clock() == 0 and rows(L2, "location_claim") == [])
    check("where_is on an empty memory: no evidence, prior/explore only", all(c["tier"] in ("T3", "T4") for c in L2.where_is("Apple")["candidates"])
          and L2.where_is("Apple")["explore"]["unknown"])
    check("the fixed prior is what a cold agent gets", L2.where_is("Apple")["candidates"][0]["id"] == "Fridge" and L2.where_is("Apple")["candidates"][0]["source"] == "prior")


# ================================================================================================= 9 habits
def synth_trials(L, kitchen, otype, spec, E0=1, epoch0=0):
    """Commit synthetic search_trial rows: spec = {rtype: n_positive_units}. One place per rtype per kitchen."""
    n = len(L.store.select("place_instance"))
    ops, tid = [], 0
    for rtype, count in spec.items():
        n += 1
        pid = f"pl:{n}"
        ops.append({"table": "place_instance", "op": "insert", "values": {"place": pid, "kitchen": kitchen, "frame": "f0", "epoch": 0, "rtype": rtype,
                                                                          "cx": float(n), "cy": 0.0, "cz": 0.0, "extent": {}, "created_E": 0},
                    "evidence": {"rule": "R-ID", "mutation": "create", "summary": {}, "src_ev": []}})
        for i in range(count):
            tid += 1
            ops.append({"table": "search_trial", "op": "insert", "values": {"u": f"tp:{kitchen}-{otype}-{rtype}-{i}", "kitchen": kitchen, "epoch": epoch0 + i,
                                                                            "otype": otype, "place": pid, "rtype": rtype, "outcome": "positive", "E": E0},
                        "evidence": {"rule": "R3", "mutation": "create", "summary": {}, "src_ev": []}})
    L.store.commit(f"synth:{kitchen}:{otype}:{len(L.store.select('commit_receipt'))}", "import", ops, stamp=E0)


def test_habits(tmp):
    P = params()
    # one kitchen: built, INACTIVE
    L = new_ltm(tmp, "hab1", P)
    synth_trials(L, "FloorPlan1", "Apple", {"Fridge": 5, "CounterTop": 1})
    g = L.habits("Apple", e_now=1)["general"]
    check("one kitchen: general habit inactive (reasons say why)", g["active"] is False and g["validated"] is False and g["reasons"])
    k = L.habits("Apple", e_now=1)["kitchen"]
    check("kitchen habit is active and ordered by preference", k[0]["rtype"] == "Fridge" and k[0]["pref"] > k[1]["pref"])
    a, p0 = 2.0, hab.p0_for(hab.default_prior("Apple"), "Fridge", P)
    check("pref = (P + a*p0)/(P + N + a)", abs(k[0]["pref"] - (k[0]["P"] + a * p0) / (k[0]["P"] + k[0]["N"] + a)) < 1e-12)
    check("p0 normalised over the listed prior, unlisted -> p0_unlisted", abs(sum(hab.p0_for({"A": 1, "B": 3}, r, P) for r in ("A", "B")) - 1.0) < 1e-12
          and hab.p0_for({"A": 1}, "Z", P) == P["habits"]["p0_unlisted"]["value"])
    # where_is: kitchen habit appears (T3) below evidence; no general habit
    a_ = L.where_is("Apple")
    check("kitchen habit offered as T3 (source habit)", any(c["tier"] == "T3" and c["source"] == "habit" for c in a_["candidates"]) and all(c["source"] != "general_habit" for c in a_["candidates"]))
    # two kitchens, no third: not validated
    Pon = params(habits__general_active=True)
    L = new_ltm(tmp, "hab2", Pon)
    synth_trials(L, "FloorPlan1", "Apple", {"Fridge": 4})
    synth_trials(L, "FloorPlan6", "Apple", {"Fridge": 4})
    g = L.habits("Apple", e_now=1)["general"]
    check("2 kitchens, no validation kitchen: still inactive even with the switch on", g["active"] is False and g["validated"] is False and len(g["fit_kitchens"]) == 2)
    # 3 kitchens agree -> active (switch on)
    synth_trials(L, "FloorPlan8", "Apple", {"Fridge": 4, "Cabinet": 1})
    g = L.habits("Apple", e_now=1, general_validation="FloorPlan8")["general"]
    check("fit on 2 kitchens, validated on a different 3rd: ACTIVE", g["active"] is True and g["validated"] and g["validation_kitchen"] == "FloorPlan8"
          and g["fit_kitchens"] == ["FloorPlan1", "FloorPlan6"] and g["prefs"][0]["rtype"] == "Fridge")
    check("the validation kitchen is excluded from the fit", "FloorPlan8" not in g["fit_kitchens"])
    # same data, registry switch off -> inactive
    Loff = new_ltm(tmp, "hab2off", params())
    synth_trials(Loff, "FloorPlan1", "Apple", {"Fridge": 4})
    synth_trials(Loff, "FloorPlan6", "Apple", {"Fridge": 4})
    synth_trials(Loff, "FloorPlan8", "Apple", {"Fridge": 4})
    g = Loff.habits("Apple", e_now=1, general_validation="FloorPlan8")["general"]
    check("validated but registry switch off -> inactive", g["validated"] and g["active"] is False)
    check("inactive general habit is never offered by where_is", all(c["source"] != "general_habit" for c in Loff.where_is("Apple")["candidates"]))
    # validation disagrees
    Ld = new_ltm(tmp, "hab3", Pon)
    synth_trials(Ld, "FloorPlan1", "Apple", {"Fridge": 4})
    synth_trials(Ld, "FloorPlan6", "Apple", {"Fridge": 4})
    synth_trials(Ld, "FloorPlan8", "Apple", {"CounterTop": 6})
    g = Ld.habits("Apple", e_now=1, general_validation="FloorPlan8")["general"]
    check("validation kitchen disagrees -> not promoted", g["active"] is False and not g["validated"] and any("disagrees" in r for r in g["reasons"]))
    # too few discoveries per kitchen
    Lf = new_ltm(tmp, "hab4", Pon)
    synth_trials(Lf, "FloorPlan1", "Apple", {"Fridge": 4})
    synth_trials(Lf, "FloorPlan6", "Apple", {"Fridge": 2})
    synth_trials(Lf, "FloorPlan8", "Apple", {"Fridge": 4})
    g = Lf.habits("Apple", e_now=1, general_validation="FloorPlan8")["general"]
    check("a kitchen under min_discoveries_per_kitchen does not count", "FloorPlan6" not in g["eligible_kitchens"] and g["active"] is False)
    # equal weight per kitchen: volume does not dominate
    Le = new_ltm(tmp, "hab5", Pon)
    synth_trials(Le, "FloorPlan1", "Apple", {"Fridge": 4})
    synth_trials(Le, "FloorPlan6", "Apple", {"CounterTop": 60})
    synth_trials(Le, "FloorPlan8", "Apple", {"Fridge": 4})
    g = Le.habits("Apple", e_now=1, general_validation="FloorPlan8")["general"]
    pf = {x["rtype"]: x["pref"] for x in g["prefs"]}
    check("equal weight per kitchen: 60 counter discoveries in one kitchen do not swamp the 4 fridge ones elsewhere", abs(pf["Fridge"] - pf["CounterTop"]) < 0.5
          and g["prefs"][0]["rtype"] in ("CounterTop", "Fridge"))
    # when promoted, general habit fills T3 after the kitchen habit for rtypes the kitchen never tried
    Lg = new_ltm(tmp, "hab6", Pon)
    synth_trials(Lg, "FloorPlan6", "Apple", {"Fridge": 4})
    synth_trials(Lg, "FloorPlan8", "Apple", {"Fridge": 4})
    synth_trials(Lg, "FloorPlan15", "Apple", {"Fridge": 4})
    Lg.kitchen = "FloorPlan1"
    synth_trials(Lg, "FloorPlan1", "Apple", {"CounterTop": 1})
    g = Lg.habits("Apple", e_now=1, general_validation="FloorPlan15")["general"]
    check("synthetic multi-kitchen set promoted", g["active"])
    # place-level: ties are deterministic
    col = hab.collapse_trials([{"kitchen": "k", "epoch": 0, "otype": "A", "place": "pl:1", "rtype": "R2", "outcome": "positive", "E": 1},
                               {"kitchen": "k", "epoch": 0, "otype": "A", "place": "pl:2", "rtype": "R1", "outcome": "positive", "E": 1}], {})
    pr = hab.kitchen_prefs(col, P, 1, "A", "k", prior={})
    check("ties broken by receptacle name (deterministic)", [x["rtype"] for x in pr] == ["R1", "R2"])


# ================================================================================================= 10 where_is behaviours
def test_where(tmp):
    # evidence outranks habits even when the habit is much stronger
    L = new_ltm(tmp, "w1")
    synth_trials(L, "FloorPlan1", "Apple", {"Cabinet": 20, "Drawer": 3}, E0=0)
    synth = Ep("cnt")
    synth.recv("CounterTop", COUNTER)
    synth.sight("Apple", "tr:2", "CounterTop", COUNTER, basis="box_only")
    run(L, synth.finish())
    a = L.where_is("Apple")
    top = a["candidates"][0]
    check("a fresh trusted counter sighting (T0) outranks a very strong cabinet habit (T3)", top["tier"] == "T0" and top["rtype"] == "CounterTop"
          and [c["id"] for c in a["candidates"] if c["source"] == "habit"][:1] == ["Cabinet"], a["candidates"])
    ranks = [dec.TIER_RANK[c["tier"]] for c in a["candidates"]]
    check("candidates are tier-ordered", ranks == sorted(ranks))
    check("candidate fields complete", all(k in top for k in ("id", "kind", "rtype", "tier", "source", "age", "support", "contradictions", "status", "weight", "provenance")))
    # six independent discoveries in the fridge: the habit is strong, but the specific fridge place is evidence (T0), and the habit does not repeat it
    L = new_ltm(tmp, "w1b")
    for i in range(6):
        L.reset_world(f"hw{i}")
        run(L, natural_apple_in_fridge(f"h{i}", epoch=i + 1))
    a = L.where_is("Apple")
    check("the fridge habit does not repeat a fridge place already offered as evidence", [c["id"] for c in a["candidates"]].count("Fridge") == 0
          and a["candidates"][0]["id"] == "pl:1" and L.habits("Apple")["kitchen"][0]["n_pos"] == 6)
    # verified before trusted inside T0
    L = new_ltm(tmp, "w2")
    e = Ep("v1")
    e.recv("Fridge", FRIDGE)
    e.recv("CounterTop", COUNTER)
    e.sight("Apple", "tr:1", "CounterTop", COUNTER, basis="box_only")
    run(L, e.finish())
    run(L, natural_apple_in_fridge("v2"))
    for i in range(2):
        run(L, Ep(f"gap{i}").finish())
    e = Ep("v3")
    e.sight("Apple", "tr:1", "CounterTop", COUNTER, basis="box_only")
    run(L, e.finish())
    t0 = [c for c in L.where_is("Apple")["candidates"] if c["tier"] == "T0"]
    check("T0: verified before trusted (sub-tier), even if the trusted one is fresher",
          [c["provenance"] for c in t0][:2] == ["natural/verified", "natural/trusted"], t0)
    # where_k comes from the registry
    L = new_ltm(tmp, "w3", params(retrieval__where_k=2))
    check("retrieval.where_k from the registry", len(L.where_is("Apple")["candidates"]) == 2 and L.where_is("Apple")["explore"]["n_beyond_cut"] > 0)
    # WM masks: an inspection this task suppresses (qualified) or defers (partial) the place
    L = new_ltm(tmp, "w4")
    run(L, natural_apple_in_fridge("a1"))
    e = Ep("mask")
    e.recv("Fridge", FRIDGE)
    o = e.open_("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o, targets=("Apple",))
    ans = L.where_is("Apple", wm=e.wm())
    check("WM mask: a place qualified-inspected this task goes to EXPLORE", "pl:1" not in cand_ids(ans) and ans["explore"]["suppressed"][0]["reason"] == "inspected_this_task")
    ans0 = L.where_is("Apple")
    check("without the mask the place is the top candidate", cand_ids(ans0)[0] == "pl:1")
    e = Ep("mask2")
    e.recv("Fridge", FRIDGE)
    o = e.open_("Fridge", FRIDGE)
    e.insp("Fridge", FRIDGE, o, targets=("Apple",), verdict="partly")
    ans = L.where_is("Apple", wm=e.wm())
    check("WM mask: a partial inspection defers the place (not ruled out, not suppressed)", "pl:1" in cand_ids(ans) and ans["explore"]["deferred"] == ["pl:1"]
          and ans["explore"]["suppressed"] == [] and ans["candidates"][0]["status"].endswith("+deferred"))
    # Gemma may reorder only T2-T4 and only known ids
    ans = {"candidates": [{"id": "pl:1", "tier": "T0"}, {"id": "pl:2", "tier": "T2"}, {"id": "Fridge", "tier": "T3"}]}
    check("reorder_within: cannot lift a habit above current evidence", lrt.reorder_within(ans, ["Fridge", "pl:2", "pl:1"]) == ["pl:1", "Fridge", "pl:2"])
    check("reorder_within: unknown ids ignored, missing ids keep order", lrt.reorder_within(ans, ["Nope"]) == ["pl:1", "pl:2", "Fridge"])
    # determinism
    A = new_ltm(tmp, "w5a")
    B = new_ltm(tmp, "w5b")
    for L_ in (A, B):
        run(L_, natural_apple_in_fridge("x1"))
    check("same inputs -> identical answers and state hash", A.where_is("Apple") == B.where_is("Apple") and A.state_hash() == B.state_hash())
    # replaying a shuffled complete batch gives the same database
    e = Ep("sh")
    e.recv("Fridge", FRIDGE)
    e.recv("CounterTop", COUNTER)
    e.sight("Apple", "tr:1", "Fridge", FRIDGE)
    e.sight("Mug", "tr:2", "CounterTop", COUNTER)
    e.finish(goal=True)
    sh = list(e.events)
    random.Random(7).shuffle(sh)
    X = new_ltm(tmp, "w6a")
    Y = new_ltm(tmp, "w6b")
    X.end_episode(WorkingMemory.replay(e.events))
    Y.end_episode(WorkingMemory.replay(sh))
    check("shuffled replay -> identical state", X.state_hash() == Y.state_hash())


# ================================================================================================= 11 frozen clock
def test_frozen(tmp):
    L = new_ltm(tmp, "fz")
    run(L, natural_apple_in_fridge("z1"))
    run(L, neg_episode("z2", targets=("Egg",)))
    snap = Path(tmp) / "fz-snap"
    L.snapshot(snap)
    F = ltm.LongTermMemory.open_frozen(snap, Path(tmp) / "fz-testlog.db", params=params())
    h0 = F.state_hash()
    a1 = F.where_is("Apple")
    for _ in range(5):
        F.where_is("Egg")
        F.where_is("Mug")
    a2 = F.where_is("Apple")
    check("frozen store: answers independent of how many / which queries ran before", a1 == a2 and F.state_hash() == h0 and F.clock() == L.clock())
    r = F.end_episode(Ep("zz").finish().wm())
    check("frozen store: end_episode writes nothing and the clock does not move", r["status"] == "frozen" and F.state_hash() == h0 and F.clock() == L.clock())
    # test/probe on the live store do not advance the clock either
    c0 = L.clock()
    for m in ("test", "probe"):
        e = Ep("zz-" + m, mode=m)
        e.sight("Apple", "tr:3", "Fridge", FRIDGE)
        run(L, e.finish())
    check("test/probe episodes never advance E", L.clock() == c0)
    # explicit frozen clock: answer identical to where_is at that clock even after the store moved on
    w_frozen = F.where_is("Apple", e_now=c0)["candidates"][0]["weight"]
    run(L, Ep("later").finish())
    check("explicit e_now pins the ranking whatever the store clock does", L.where_is("Apple", e_now=c0)["candidates"][0]["weight"] == w_frozen)
    check("frozen where_is returns the snapshot's E as e_now", F.where_is("Apple")["e_now"] == c0)
    F.close()


# ================================================================================================= 12 hygiene
def test_hygiene(tmp):
    mem = REPO / "sim" / "memory.db"
    trc = REPO / "sim" / "trace.db"
    m0 = mem.stat().st_mtime if mem.exists() else None
    t0 = trc.stat().st_mtime if trc.exists() else None
    code = (
        "import sys, socket, urllib.request\n"
        "def boom(*a, **k): raise RuntimeError('network/simulator touched')\n"
        "socket.socket = boom; socket.create_connection = boom; urllib.request.urlopen = boom\n"
        "sys.path.append(%r)\n"
        "import ltm, ltm_events, ltm_wm, ltm_identity, ltm_decay, ltm_habits, ltm_consolidate, ltm_retrieve\n"
        "bad = [m for m in ('ai2thor','requests','torch','ultralytics','honest','objmem','honest_skills') if m in sys.modules]\n"
        "print('BAD' if bad else 'CLEAN', bad)\n" % HERE)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(REPO))
    check("importing every ltm module starts no simulator / socket and loads no step-4 module", out.stdout.startswith("CLEAN"), out.stdout + out.stderr)
    check("sim/memory.db untouched", (mem.stat().st_mtime if mem.exists() else None) == m0)
    # sim/trace.db is written by the concurrent step-4 workflow, so its mtime proves nothing here (same note as A1)
    check("default sim/step5/ltm.db never created", not (REPO / "sim" / "step5" / "ltm.db").exists())
    # no step-4 module is edited / imported
    src = "".join((REPO / "sim" / f).read_text(encoding="utf-8") for f in ("ltm_consolidate.py", "ltm_identity.py", "ltm_retrieve.py", "ltm_habits.py", "ltm_decay.py"))
    check("no import of step-4 modules in the rules layer", all(f"import {m}" not in src for m in ("honest", "objmem", "honest_skills", "honest_map", "ai2thor")))
    # registry: every number the rules use is in params.json (spot-check by mutation)
    L = new_ltm(tmp, "mut", params(inspect__min_support=20))
    run(L, neg_episode("n1"))
    check("inspect.min_support comes from the registry", rows(L, "negative_evidence") == [])
    L = new_ltm(tmp, "mut2", params(inspect__max_range_m=0.5))
    run(L, neg_episode("n1"))
    check("inspect.max_range_m comes from the registry", rows(L, "negative_evidence") == [])
    L = new_ltm(tmp, "mut3", params(inspect__vis_ok=0.95))
    run(L, neg_episode("n1"))
    check("inspect.vis_ok comes from the registry", rows(L, "negative_evidence") == [])
    Pm = params()
    Pm["decay"]["retire_below"]["value"] = 0.9
    L = new_ltm(tmp, "mut4", Pm)
    run(L, natural_apple_in_fridge("a"))
    run(L, Ep("b").finish())
    check("decay.retire_below comes from the registry", all(c["source"] != "location_claim" for c in L.where_is("Apple")["candidates"]))


def main():
    with tempfile.TemporaryDirectory(prefix="s5rules-") as tmp:
        for name, fn in [("events+wm", test_events), ("basic", test_basic), ("negatives", test_negatives),
                         ("inflation+self-placement", test_inflation), ("moved+contradictions", test_moved),
                         ("identity", test_identity), ("decay", test_decay), ("reset", test_reset),
                         ("habits", test_habits), ("where_is", test_where), ("frozen", test_frozen),
                         ("hygiene", test_hygiene)]:
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
