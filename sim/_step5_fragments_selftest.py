"""Self-test for final-test stage C3 - procedure fragments, the procedures channel of the memory block, retention, A10 verb aliases.

Offline, Windows `python` from the REPO ROOT, temp directories only, no simulator, no WSL, no Ollama, no network:

    python sim/_step5_fragments_selftest.py
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import ltm                                # noqa: E402
import ltm_adapter as A                   # noqa: E402
import ltm_concepts as lc                 # noqa: E402
import ltm_fragments as lf                # noqa: E402
import ltm_lessons as ll                  # noqa: E402
import ltm_retrieval as lr                # noqa: E402
from _step5_lessons_selftest import Ep    # noqa: E402

# W2 (A12 written into sim/step5/params.json): these checks are about the C2-e path, so they read the registry WITHOUT the A12 block
# (the A12 block is tested in _step5_retention_a12_selftest.py).
_load_params_real = ltm.load_params


def _load_params_c2e(path=None):
    import copy as _copy
    P = _copy.deepcopy(_load_params_real(path))
    P["retrieval"].pop("retention", None)
    P["retrieval"].pop("a12", None)
    return P


ltm.load_params = _load_params_c2e

_pass = _fail = 0
PLAN3 = [("go_to", "Fridge", "None"), ("open", "Fridge", "None"), ("pick_and_place", "Apple", "Fridge")]


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def concepts_of(text):
    p = lc.parse(text)
    return {"roles": p["roles"], "ok": bool(p["ok"]), "reasons": p["reasons"], "parser": p["parser"], "lexicon": p["lexicon"]}


def episode(L, name, text, plan, ok=None, goal=True, exec_ok=True, mode="practice"):
    """Run one practice episode through the real consolidation (+ LessonsExtension)."""
    L.reset_world(name)
    ep = Ep(name, epoch=L.current_epoch(), concepts=concepts_of(text), text=text, mode=mode)
    ok = ok if ok is not None else [True] * len(plan)
    for i, (v, o, t) in enumerate(plan):
        ep.step(i, verb=v, otype=o, to=t, pre=None)
        ep.outcome(i, ok[i])
    ep.finish(exec_ok=exec_ok, goal=goal)
    return L.end_episode(ep.wm(), extensions=[ll.LessonsExtension()])


def plan_steps(plan):
    return [{"action": s[0], "object": s[1], "to": s[2] if s[2] != "None" else None} for s in plan]


def new(tmp, name, **kw):
    return ltm.LongTermMemory(Path(tmp) / name / "ltm.db", kitchen="FloorPlan1", params=ltm.load_params(), **kw)


def wm_for(L, text, lang="en", mode="practice"):
    w = L.begin_episode("plan-" + "".join(c for c in text if c.isalnum())[:8] + lang, mode)
    A.HonestEventAdapter(L, wm=w, episode="plan-" + "".join(c for c in text if c.isalnum())[:8] + lang, mode=mode).on_task(text, "L2", lang)
    return w


def test_alias(tmp):
    rolePick = {"verb": "pick", "req": "-", "obj": "Apple", "ostate": "-", "src": "-", "dest": "-", "dstate": "-", "neg": 0}
    rolePut = dict(rolePick, verb="put", dest="Fridge")
    S = lambda v, o, t="None": {"verb": v, "otype": o, "to": t}      # noqa: E731
    check("A10 pickup matches pick", lr.step_matches_role(S("pickup", "Apple"), rolePick))
    check("A10 pick_and_place matches pick (object)", lr.step_matches_role(S("pick_and_place", "Apple", "Sink"), rolePick))
    check("A10 pick_and_place matches put only with the destination",
          lr.step_matches_role(S("pick_and_place", "Apple", "Fridge"), rolePut) and not lr.step_matches_role(S("pick_and_place", "Apple", "Sink"), rolePut))
    check("A10 put_in names the receptacle", lr.step_matches_role(S("put_in", "Fridge"), rolePut) and not lr.step_matches_role(S("put_in", "Apple"), rolePut))
    check("A10 unrelated verb never matches", not lr.step_matches_role(S("open", "Apple"), rolePick))
    check("A10 old identity still matches", lr.step_matches_role(S("slice", "Apple"), dict(rolePick, verb="slice")))
    check("A10 wrong object never matches pickup", not lr.step_matches_role(S("pickup", "Pear"), rolePick))
    check("A10 put with unknown destination never matches put_in", not lr.step_matches_role(S("put_in", "Fridge"), dict(rolePut, dest="?")))
    # composite: one pick_and_place cannot satisfy BOTH a pick role and the following put role (body_consistent advances past it)
    rec = {"body": {"steps": [{"verb": "pick_and_place", "otype": "Apple", "to": "Fridge"}]}}
    check("A10 composite step cannot serve two roles", not lr.body_consistent(rec, [rolePick, rolePut]))
    check("A10 pick_and_place serves the put role alone", lr.body_consistent(rec, [rolePut]))
    neg = dict(rolePick, neg=1)
    check("A10 negated role: the verb must NOT appear (pickup now counts as appearing)",
          not lr.body_consistent({"body": {"steps": [{"verb": "pickup", "otype": "Apple", "to": "None"}]}}, [neg]))


def test_extract(tmp):
    L = new(tmp, "x")
    episode(L, "e1", "Put the apple in the fridge.", PLAN3)
    procs = L.store.select("procedure")
    check("admitted procedure exists", len(procs) == 1, str(len(procs)))
    fr, ds = lf.extract(procs[0])
    check("one fragment, 3 steps", len(fr) == 1 and len(fr[0]["sig"]) == 3, str((fr, ds)))
    check("fragment key is the exact role", bool(fr) and fr[0]["key"] == ("put", "-", "Apple", "-", "-", "Fridge", "-", "0"), str(fr and fr[0]["key"]))
    episode(L, "e2", "Pick up the apple.", [("pickup", "Apple", "None")])
    episode(L, "e3", "Put the apple in the fridge.", [("go_to", "Sink", "None"), ("pick_and_place", "Apple", "Fridge")])
    reasons = {}
    for r in L.store.select("procedure"):
        for d in lf.extract(r)[1]:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    check("one_step discard counted", reasons.get("one_step", 0) >= 1, str(reasons))
    check("ambiguous_prefix discard counted", reasons.get("ambiguous_prefix", 0) >= 1, str(reasons))
    r4 = episode(L, "e4", "Put the apple in the fridge.", PLAN3, goal=False)
    check("a goal-unmet episode is not admitted (no new procedure)", len(L.store.select("procedure")) == 3, str(r4)[:200])
    L.close()


def test_support_and_gate(tmp):
    L = new(tmp, "g")
    episode(L, "g1", "Put the apple in the fridge.", PLAN3)
    idx = lf.build_index(L.store, "FloorPlan1")
    f = next(iter(idx["fragments"].values()))
    check("support 1 after one admission", f["support"] == 1 and not f["hidden"], str(f["support"]))
    roles = concepts_of("Put the apple in the fridge.")["roles"]
    check("gate offers the exact-key fragment", len(lf.candidates_for_roles(idx, "FloorPlan1", roles)) == 1)
    other = concepts_of("Put the apple in the cabinet.")["roles"]
    check("gate: a different destination never matches", lf.candidates_for_roles(idx, "FloorPlan1", other) == [])
    check("gate: wrong kitchen returns nothing", lf.build_index(L.store, "FloorPlan2")["fragments"] == {})
    check("gate: '?' never matches", lf.candidates_for_roles(idx, "FloorPlan1", [dict(roles[0], obj="?")]) == [])
    en1, fa1 = lf.render_fragment(f, "en"), lf.render_fragment(f, "fa")
    check("support-1 rendering is qualified", "worked once, unconfirmed" in en1 and "تأییدنشده" in fa1, en1)
    check("steps in tool syntax, language-neutral", lf.steps_text(f["steps"]) in en1 and lf.steps_text(f["steps"]) in fa1)
    episode(L, "g2", "Put the apple in the fridge.", PLAN3)
    f = next(iter(lf.build_index(L.store, "FloorPlan1")["fragments"].values()))
    check("support 2 after a second distinct unit", f["support"] == 2, str(f["support"]))
    check("support-2 rendering names the count", "worked 2 times" in lf.render_fragment(f, "en") and "2 بار" in lf.render_fragment(f, "fa"))
    led = lf.FragmentLedger()
    led.observe("u:a", "ep1", plan_steps(PLAN3), [True, True, False], [dict(f)])
    check("one failed unit < support 2: still shown", not lf.build_index(L.store, "FloorPlan1", led)["fragments"][f["fid"]]["hidden"])
    led.observe("u:b", "ep2", plan_steps(PLAN3), [True, False, True], [dict(f)])
    check("failed units >= support: hidden", lf.build_index(L.store, "FloorPlan1", led)["fragments"][f["fid"]]["hidden"])
    led2 = lf.FragmentLedger()
    led2.observe("u:a", "ep1", plan_steps(PLAN3), [True, True, False], [dict(f)])
    led2.observe("u:a", "ep1b", plan_steps(PLAN3), [True, True, True], [dict(f)])
    check("per-unit counterexample wins, order independent", led2.failed_units(f["fid"]) == 1)
    led4 = lf.FragmentLedger()
    twice = PLAN3 + PLAN3
    led4.observe("u:d", "ep4", plan_steps(twice), [True, True, True, True, False, True], [dict(f)])
    check("every occurrence is inspected: a later failure is not hidden by an earlier success", led4.failed_units(f["fid"]) == 1 and len(led4.rows) == 2)
    led3 = lf.FragmentLedger()
    check("a plan without the exact signature adds no row", led3.observe("u:c", "ep3", plan_steps(PLAN3[1:]), [True, True], [dict(f)]) == [])

    class FS:
        def select(self, table, where=None, **k):
            rows = L.store.select(table, where=where, **k)
            return [dict(r, status="retired") for r in rows] if table == "procedure" else rows
    check("retired parent: fragment hidden", all(e["hidden"] for e in lf.build_index(FS(), "FloorPlan1")["fragments"].values()))
    L.close()


def test_block(tmp):
    L = new(tmp, "b")
    for n in range(3):
        episode(L, f"b{n}", "Put the apple in the fridge.", PLAN3)
    episode(L, "b9", "Pick up the apple.", [("pickup", "Apple", "None")])
    episode(L, "b10", "Pick up the apple.", [("pickup", "Apple", "None")])
    L.fragment_ledger = lf.FragmentLedger()
    L.reset_world("te1")
    w = wm_for(L, "Put the apple in the fridge.")
    P0 = L.memory_block(w, "L2", "en", channels=("places",))
    P1 = L.memory_block(w, "L2", "en", channels=("places", "procedures"))
    check("fragment line rendered", "Worked before:" in P1["text"] and "this worked 3 times" in P1["text"], P1["text"])
    check("procedure ids recorded", any(r["id"].startswith("proc/fr:") for r in P1["records"]))
    fa = L.memory_block(wm_for(L, "Put the apple in the fridge.", "en"), "L2", "fa", channels=("places", "procedures"))
    check("EN and FA retain identical record ids", [r["id"] for r in P1["records"]] == [r["id"] for r in fa["records"]])
    check("places-only block unchanged by the new code path", P0["text"] == L.memory_block(w, "L2", "en", channels=("places",))["text"])
    check("no lesson text is ever rendered", "Failure hints" not in P1["text"] and "نکته‌های شکست" not in fa["text"])
    w2 = wm_for(L, "Open the microwave.")
    check("unmatched command: procedures add nothing",
          L.memory_block(w2, "L2", "en", channels=("places",))["text"] == L.memory_block(w2, "L2", "en", channels=("places", "procedures"))["text"])
    w3 = wm_for(L, "Pick up the apple.")
    rep = L.memory_block(w3, "L2", "en", channels=("procedures",))["proc_report"]
    check("whole-intent trusted procedure steers (A10: pick matches pickup)", len(rep["whole"]) == 1, str(rep))
    tiny = L.memory_block(w, "L2", "en", channels=("places", "procedures"), budget_tokens=25)
    check("tiny budget: nothing partial, within budget", (not tiny["text"]) or tiny["tokens"] <= 25, str(tiny["tokens"]))
    L.close()
    Lc = ltm.LongTermMemory(Path(tmp) / "b" / "ltm.db", kitchen="FloorPlan1", params=ltm.load_params(), count_tokens=lambda t: len(t))
    Lc.fragment_ledger = lf.FragmentLedger()
    w4 = wm_for(Lc, "Put the apple in the fridge.")
    for bud in (60, 120, 400):
        a = Lc.memory_block(w4, "L2", "en", channels=("places", "procedures"), budget_tokens=bud)
        b = Lc.memory_block(w4, "L2", "fa", channels=("places", "procedures"), budget_tokens=bud)
        check(f"char-counter budget {bud}: EN/FA same ids, both within budget",
              [r["id"] for r in a["records"]] == [r["id"] for r in b["records"]] and len(a["text"]) <= bud and len(b["text"]) <= bud,
              f"{len(a['text'])} {len(b['text'])}")
    fid = next(iter(lf.build_index(Lc.store, "FloorPlan1")["fragments"]))
    Lc.fragment_ledger.rows.append({"unit": "u:x", "episode": "x", "fid": fid, "ok": False})
    snap = Path(tmp) / "b" / "snap"
    res = Lc.snapshot(snap, overwrite=True)
    check("snapshot records the ledger sha", bool(res.get("fragments_sha256")))
    Lc.close()
    fz = ltm.LongTermMemory.open_frozen(snap, Path(tmp) / "b" / "t.db", kitchen="FloorPlan1")
    check("frozen store loads the ledger", fz.fragment_ledger is not None and len(fz.fragment_ledger.rows) == 1)
    h0 = fz.state_hash()
    fz.begin_test_epoch()
    w5 = fz.begin_episode("plan-f", "test")
    A.HonestEventAdapter(fz, wm=w5, episode="plan-f", mode="test").on_task("Put the apple in the fridge.", "L2", "en")
    fz.memory_block(w5, "L2", "en", channels=("places", "procedures"))
    check("rendering a block never writes the frozen store", fz.state_hash() == h0)
    fz.close()


def test_hygiene(tmp):
    src = Path(HERE, "ltm_fragments.py").read_text(encoding="utf-8")
    for bad in ("import ai2thor", "import requests", "urllib", "socket", "ollama", ".commit(", ".apply("):
        check(f"fragments module free of {bad!r}", bad not in src)


def main():
    with tempfile.TemporaryDirectory(prefix="s5frag-") as tmp:
        for name, fn in [("A10 aliases", test_alias), ("extraction", test_extract), ("support + gate", test_support_and_gate),
                         ("block + retention", test_block), ("hygiene", test_hygiene)]:
            before = _fail
            try:
                fn(tmp)
            except Exception as e:
                import traceback
                traceback.print_exc()
                check(f"{name}: crashed", False, repr(e))
            print(f"  [{name}] {'ok' if _fail == before else 'FAILURES'}")
    print(f"\n{_pass}/{_pass + _fail} checks passed")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
