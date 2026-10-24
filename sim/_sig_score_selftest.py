"""Gate for sim/final_test/signal/score_equiv.py (stage I-metric). Pure, offline, Windows `python`, no simulator.
Run from the repo root:  python sim/_sig_score_selftest.py
"""
import ast
import inspect
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "sim", "final_test", "signal"))
import score_equiv as S  # noqa: E402

N = [0, 0]


def ok(cond, name):
    N[1] += 1
    if cond:
        N[0] += 1
    else:
        print("FAIL:", name)


def st(a, o=None, to=None):
    return {"action": a, "object": o, "to": to, "arg": None}


def eq(g, r, holding=None):
    return S.canonical(g, holding) == S.canonical(r, holding)


# --- C1 pick_and_place
ok(eq([st("pick_and_place", "Apple", "Fridge")], [st("search_for", "Apple"), st("pickup", "Apple"), st("open", "Fridge"),
                                                   st("put_in", "Fridge"), st("close", "Fridge")]), "C1 full expansion with open/close")
ok(eq([st("pick_and_place", "Apple", "Fridge")], [st("pickup", "Apple"), st("put_in", "Fridge")]), "C1 plain")
ok(eq([st("pick_and_place", "Lettuce", "SinkBasin")], [st("put_in", "SinkBasin")], holding="Lettuce"), "C1 holding: pickup is a no-op")
ok(not eq([st("pick_and_place", "Lettuce", "SinkBasin")], [st("put_in", "SinkBasin")], holding=None), "C1 not holding: pickup needed")
ok(not eq([st("pick_and_place", "Bowl", "Cabinet")], [st("pick_and_place", "Bowl", "CounterTop")]), "wrong receptacle stays different")
ok(not eq([st("pickup", "Apple")], [st("pickup", "Apple"), st("put_in", "Fridge")]), "missing put stays different")
# --- C2 heat / cook
ok(eq([st("heat", "Egg")], [st("cook", "Egg")]), "C2 heat == cook")
ok(eq([st("heat", "Egg")], [st("pickup", "Egg"), st("put_in", "Microwave"), st("close", "Microwave"), st("toggle_on", "Microwave"),
                            st("toggle_off", "Microwave"), st("open", "Microwave"), st("pickup", "Egg")]), "C2 expansion")
ok(not eq([st("heat", "Egg")], [st("pickup", "Egg"), st("put_in", "Microwave"), st("toggle_on", "Microwave"), st("toggle_off", "Microwave"),
                            st("open", "Microwave"), st("pickup", "Egg")]), "K-round2: a microwave route WITHOUT the close is NOT the heat route (close-folding bug fixed)")
ok(eq([st("put_in", "Fridge"), st("close", "Fridge")], [st("put_in", "Fridge")]), "C6 close after put_in still folds for a plain container")
ok(not eq([st("heat", "Egg")], [st("pickup", "Egg"), st("put_in", "Microwave"), st("toggle_on", "Microwave")]), "C2 vs cut-short route differs")
ok(eq([st("heat", "Egg")], [st("heat", "Egg")], holding="Egg"), "C2 holding same")
# --- C3 wash
ok(eq([st("wash", "Pan")], [st("pickup", "Pan"), st("put_in", "SinkBasin"), st("toggle_on", "Faucet"), st("clean", "Pan"),
                            st("toggle_off", "Faucet")]), "C3 expansion")
ok(not eq([st("wash", "Pan")], [st("pickup", "Pan"), st("put_in", "SinkBasin")]), "C3 vs only putting in the sink differs")
# --- C4 / C5
ok(eq([st("pickup", "Mug")], [st("pickup", "Mug")], holding="Mug"), "C4 trivial")
ok(S.canonical([st("pickup", "Mug")], "Mug") == [], "C4 pickup while holding is empty")
ok(eq([st("search_for", "Pan"), st("pickup", "Pan")], [st("pickup", "Pan")]), "C5 search_for before own pickup")
ok(eq([st("go_to", "Pan"), st("search_for", "Pan"), st("pickup", "Pan")], [st("pickup", "Pan")]), "C5 two helpers")
ok(not eq([st("search_for", "Pan")], [st("pickup", "Pan")]), "C5 a lone search_for is NOT a pickup")
ok(not eq([st("search_for", "Pan"), st("pickup", "Spatula")], [st("pickup", "Spatula")]), "C5 search for another object stays")
ok(eq([st("go_to", "Fridge"), st("open", "Fridge")], [st("open", "Fridge")]), "C5 go_to before open of same object")
# --- C6
ok(eq([st("open", "Fridge"), st("put_in", "Fridge")], [st("put_in", "Fridge")]), "C6 open before put_in")
ok(not eq([st("open", "Fridge")], []), "C6 a lone open stays")
ok(not eq([st("close", "Fridge")], []), "C6 a lone close stays")
ok(not eq([st("open", "CounterTop"), st("put_in", "CounterTop")], [st("put_in", "CounterTop")]), "C6 only for openable receptacles")
# --- C7 aliases
ok(eq([st("put", "Sink")], [st("put_in", "Sink")]), "C7 put alias")
ok(eq([st("find", "Egg"), st("pickup", "Egg")], [st("pickup", "Egg")]), "C7 find alias")
# --- scores
r = S.equiv_scores([st("pick_and_place", "Apple", "Fridge")], [st("pickup", "Apple"), st("put_in", "Fridge")])
ok(r["pc_equiv"] == 1.0 and r["tds_equiv"] == 1.0 and r["exact_equiv"], "equiv scores perfect")
r = S.equiv_scores([], [st("pickup", "Apple")], ok=False)
ok(r["pc_equiv"] == 0.0 and r["tds_equiv"] == 0.0 and not r["exact_equiv"], "invalid plan scores zero")
r = S.equiv_scores([], [], ok=True)
ok(r["pc_equiv"] == 1.0, "both empty = 1.0")
r = S.equiv_scores([st("pickup", "Fork"), st("put_in", "SinkBasin")], [st("pickup", "Fork"), st("put_in", "SinkBasin"), st("pickup", "Spoon"),
                                                                    st("put_in", "SinkBasin")])
ok(abs(r["pc_equiv"] - 0.5) < 1e-9, "half the steps = 0.5")
ok(S.pc([], []) == 1.0 and S.tds_f1([], ["a"]) == 0.0 and S.levenshtein("kitten", "sitting") == 3, "metric primitives")
ok(S.goal_reached(None) is None and S.goal_reached({"goal_ok": True}) is True and S.goal_reached({"goal_ok": False}) is False
   and S.goal_reached({}) is False, "goal_reached")
# equivalence never uses a reference: signature has no outcome input
ok(list(inspect.signature(S.canonical).parameters) == ["plan", "holding"], "canonical takes the plan and the holding state only")

# --- contract check against honest_skills.py by source text (derived, not copied from outcomes)
src = open(os.path.join(ROOT, "sim", "honest_skills.py"), encoding="utf-8").read()
m = re.search(r"^OPENABLE = frozenset\((\{.*?\})\)", src, re.S | re.M)
ok(m is not None and set(ast.literal_eval(m.group(1))) == set(S.OPENABLE), "OPENABLE mirrors honest_skills")
m = re.search(r"^ALIASES = (\{.*?\})", src, re.S | re.M)
ok(m is not None and ast.literal_eval(m.group(1)) == S.ALIASES, "ALIASES mirror honest_skills")
i = src.index('if action in ("heat", "cook"):')
blk = src[i:i + 700]
want = ['self._put("Microwave")', '_interact("close", "Microwave")', '_interact("toggle_on", "Microwave")',
        '_interact("toggle_off", "Microwave")', '_interact("open", "Microwave")', '_interact("pickup", ot']
pos = [blk.find(w) for w in want]
ok(all(p >= 0 for p in pos) and pos == sorted(pos), "heat/cook contract order matches honest_skills")
i = src.index('if action == "wash":')
blk = src[i:i + 900]
want = ['_ensure_holding(ot)', 'self._put("SinkBasin")', '_interact("toggle_on", "Faucet")', '_interact("clean", ot', '_interact("toggle_off", "Faucet")']
pos = [blk.find(w) for w in want]
ok(all(p >= 0 for p in pos) and pos == sorted(pos), "wash contract order matches honest_skills")
ok('if action == "pick_and_place":\n            self._ensure_holding(ot)\n            return self._put(to)' in src, "pick_and_place contract")
ok('if self.h.holding == ot:\n                return {"verify": "already"}' in src, "pickup-while-holding contract")

# --- rescoring invariants on the real data (skipped if absent)
d = os.path.join(ROOT, "sim", "final_test", "signal", "plans.jsonl")
if os.path.exists(d):
    res = S.rescore()
    for tag, sets in res["sets"].items():
        for c, v in sets.items():
            ok(v["equiv_pc"] >= v["raw_pc_step"] - 1e-9 or c.startswith("G"), f"{tag}/{c}: equiv PC not below raw (non-generated arms)")
            ok(v["equiv_exact"] >= v["raw_exact"] - 2, f"{tag}/{c}: exact count sane")
    ok(res["sets"]["ft-v1 ceiling run"]["G0"]["goal_reached"] == 4 and res["sets"]["ft-v1 ceiling run"]["GH1"]["goal_reached"] == 6,
       "execution counts G0 4/8, GH1 6/8 reproduced")
    ok(res["sets"]["ft-v2 package run"]["GP"]["goal_reached"] == 3 and res["sets"]["ft-v2 package run"]["OP"]["goal_reached"] == 6,
       "execution counts GP 3/8, OP 6/8 reproduced")

print(f"{N[0]}/{N[1]}", "OK" if N[0] == N[1] else "FAILED")
sys.exit(0 if N[0] == N[1] else 1)
