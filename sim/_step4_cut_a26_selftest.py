"""A26 (K-B2b 2026-10-13) gate: the cutting-tool rule and the Knife exposure filter of the honest executor, offline.

Uses the fake kitchen of _step4_act_selftest (no simulator, no LLM). Checks (K-B2b-TODO section G2):
  * slice with the ButterKnife held is dispatched and slices;
  * empty hand / another object (Spoon) / the excluded Knife / an unknown "?" held type -> cause no_cutting_tool, 0 primitives,
    no SliceObject sent, not retried within the step, counted in ``cut_refusals``;
  * a drop between the pickup and the slice -> refused;
  * a refused slice is bounded: five refused slices cost 0 primitives and five counted refusals;
  * no other path sends SliceObject (the guard sits in ``_dispatch`` before any primitive AND at the top of ``_interact``);
  * Knife boxes are dropped at the exposure boundary (memory / search / type list) while the in-hand identity check still sees the
    raw frame (A7);
  * the validator accepts a slice plan without a pickup (A26 3(c): no new repair code) and the generated contract carries the rule;
  * no agent-facing text names the excluded type (system prompt, contract, procedures, templates, FloorPlan1 briefing records).
Run from the repo root: python sim/_step4_cut_a26_selftest.py
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FAIL = 0
N = 0


def check(name, ok, detail=""):
    global FAIL, N
    N += 1
    if ok:
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


with contextlib.redirect_stdout(io.StringIO()):                      # the act gate's fake kitchen (its own checks stay quiet)
    import _step4_act_selftest as A
import honest                                                         # noqa: E402
import honest_skills as hs                                            # noqa: E402
import type_scope                                                     # noqa: E402

check("the act gate's fake kitchen imported clean", A.FAIL == 0, f"{A.FAIL} failing")


def kit():
    return A.kitchen() + [
        dict(id="ButterKnife|+1.50|+0.95|+2.95", type="ButterKnife", pos=(1.5, 0.95, 2.95), size=0.1, pickupable=True),
        dict(id="Spoon|+0.90|+0.95|+2.95", type="Spoon", pos=(0.9, 0.95, 2.95), size=0.08, pickupable=True),
        dict(id="Knife|+1.80|+0.95|+2.95", type="Knife", pos=(1.8, 0.95, 2.95), size=0.1, pickupable=True)]


def slices(t):
    return [c for c in t.calls if c[0] == "SliceObject"]


print("1. module constants")
check("CUTTING_TOOLS == {ButterKnife}; rule and filter switches on", hs.CUTTING_TOOLS == frozenset({"ButterKnife"})
      and hs.CUT_RULE_ON and hs.TYPE_SCOPE_ON)
check("no_cutting_tool is an honest step cause", "no_cutting_tool" in hs.CAUSES)
check("Knife out of CUTLERY and of the cut / slice purpose families", "Knife" not in hs.CUTLERY
      and hs.PURPOSE_TYPES["cut"] == ("ButterKnife",) and hs.PURPOSE_TYPES["slice"] == ("ButterKnife",))
check("Knife kept as an internal identity-check class (A7)", "Knife" in hs.AMBIG_TYPES)

print("\n2. the rule, hand state by hand state")
t, det, h, sk = A.setup(objs=kit())
check("Knife is not in the agent's type list; ButterKnife is", "Knife" not in sk.type_list and "ButterKnife" in sk.type_list,
      sk.type_list)
check("start sweep: no Knife in memory (filtered at exposure), whatever was in view", not sk.om.lookup("Knife"))
n0 = len(t.calls)
r = sk.exec_step({"action": "slice", "object": "Apple"})
check("empty hand: slice refused, cause no_cutting_tool", not r["ok"] and r["cause"] == "no_cutting_tool", json.dumps(r)[:300])
check("empty hand: 0 primitives, no SliceObject, not retried", r["detail"]["prims"] == 0 and len(t.calls) == n0
      and r["retries"] == 0 and not slices(t), str(t.calls[n0:]))
check("empty hand: the apple is not sliced; one refusal counted", not A.obj(t, "Apple")["isSliced"] and sk.cut_refusals == 1)

r = sk.exec_step({"action": "pickup", "object": "Spoon"})
check("pickup Spoon ok", r["ok"] and h.holding == "Spoon", json.dumps(r)[:200])
n0 = len(t.calls)
r = sk.exec_step({"action": "slice", "object": "Apple"})
check("holding a Spoon: refused no_cutting_tool, 0 primitives, Spoon still held",
      not r["ok"] and r["cause"] == "no_cutting_tool" and len(t.calls) == n0 and h.holding == "Spoon" and not slices(t))
r = sk.exec_step({"action": "drop"})
check("drop the Spoon", r["ok"] and h.holding is None)

r = sk.exec_step({"action": "pickup", "object": "ButterKnife"})
check("pickup ButterKnife ok (identity check keeps it)", r["ok"] and h.holding == "ButterKnife", json.dumps(r)[:300])
r = sk.exec_step({"action": "slice", "object": "Apple"})
check("ButterKnife held: slice dispatched and the apple is sliced", r["ok"] and A.obj(t, "Apple")["isSliced"] and len(slices(t)) == 1,
      json.dumps(r)[:300])
check("ButterKnife held: no refusal counted", sk.cut_refusals == 2)

r = sk.exec_step({"action": "drop"})
r = sk.exec_step({"action": "slice", "object": "Potato"})
check("put / drop between pickup and slice: the slice is refused (the hand trace, not the plan, decides)",
      not r["ok"] and r["cause"] == "no_cutting_tool" and len(slices(t)) == 1)

print("\n3. start holdings from the setup record")
t, det, h, sk = A.setup(objs=kit())
k = A.obj(t, "Knife")
t.held = k["id"]                                                       # oracle world preparation put the Knife in the hand
sk.start_episode("t-knife", holding="Knife")
n0 = len(t.calls)
r = sk.exec_step({"action": "slice", "object": "Apple"})
check("setup put the (excluded) Knife in hand: refused no_cutting_tool, 0 primitives (Knife is no cutting tool under A26)",
      not r["ok"] and r["cause"] == "no_cutting_tool" and len(t.calls) == n0 and not slices(t), json.dumps(r)[:300])
h.holding = "?"
r = sk.exec_step({"action": "slice", "object": "Apple"})
check("held type unknown ('?'): refused (uncertain identity never authorizes a slice)",
      not r["ok"] and r["cause"] == "no_cutting_tool" and not slices(t))
t2, det2, h2, sk2 = A.setup(objs=kit())
b = A.obj(t2, "ButterKnife")
t2.held = b["id"]
sk2.start_episode("t-bk", holding="ButterKnife")
r = sk2.exec_step({"action": "slice", "object": "Apple"})
check("setup put the ButterKnife in hand: slice dispatched", r["ok"] and len(slices(t2)) == 1, json.dumps(r)[:300])

print("\n4. refusals are bounded")
t, det, h, sk = A.setup(objs=kit())
n0 = len(t.calls)
rs = [sk.exec_step({"action": "slice", "object": "Apple"}) for _ in range(5)]
check("five refused slices: 0 primitives, five failed steps, five counted refusals",
      all(not x["ok"] and x["cause"] == "no_cutting_tool" for x in rs) and len(t.calls) == n0 and sk.cut_refusals == 5)
sk.start_episode("t-next")
check("the refusal counter is episode-scoped", sk.cut_refusals == 0)

print("\n5. no other path sends SliceObject")
src = open(os.path.join(HERE, "honest_skills.py"), encoding="utf-8").read()
tree = ast.parse(src)
fn = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def calls_in(node, attr):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


check("_interact checks the cutting tool before anything else when the verb is slice",
      isinstance(fn["_interact"].body[0], ast.If) and calls_in(fn["_interact"].body[0], "_cutting_tool_check"))
check("_dispatch checks it in a slice branch before the generic VERBS dispatch", bool(calls_in(fn["_dispatch"], "_cutting_tool_check")))
check("'SliceObject' appears once in the executor (the VERBS table)", src.count('"SliceObject"') == 1)
check("every observation of the skills goes through _observe (no direct h.observe() left)",
      "self.h.observe()" not in src.replace("obs = self.h.observe()\n        if TYPE_SCOPE_ON", ""))

print("\n6. exposure filter vs the in-hand identity check (A7)")
t, det, h, sk = A.setup(objs=kit())
knife = {"type": "Knife", "conf": 0.9, "bbox_xyxy": (10, 10, 40, 40), "states": {}, "state": "?"}
fork = {"type": "Fork", "conf": 0.9, "bbox_xyxy": (50, 50, 70, 70), "states": {}, "state": "?"}
real_observe = h.observe
h.observe = lambda: honest.Obs(step=h.step, rgb=None, depth=None, pose=h.pose(), dets=[dict(knife), dict(fork)],
                               holding=h.holding, low=[dict(knife, conf=0.2)])
o = sk._observe()
check("_observe drops Knife from dets and low, keeps the other boxes", [d["type"] for d in o.dets] == ["Fork"] and o.low == [])
check("... and keeps the unfiltered frame as raw_dets", [d["type"] for d in o.raw_dets] == ["Knife", "Fork"])
seen = {}
real_iv = hs.identity_verdict
hs.identity_verdict = lambda dets, asked, depth=None, **kw: (seen.setdefault("types", [d["type"] for d in dets]), ("ok", asked, 0.9))[1]
h.holding = "ButterKnife"
sk.identity_check = True
_ident_on, hs.IDENT_ON = hs.IDENT_ON, True
sk._identity_check("ButterKnife", {})
hs.IDENT_ON = _ident_on
hs.identity_verdict = real_iv
h.observe = real_observe
check("the identity check of a held ButterKnife sees the specialist's Knife box (raw frame)", "Knife" in seen.get("types", []),
      str(seen))

print("\n7. planner side: contract, validator, prompt")
from planners import validator_v2 as V                                # noqa: E402
from planners import contract as PC                                   # noqa: E402
con = V.exec_contract()["tools"]
check("exec_contract: slice.requires_tool == ['ButterKnife'], requires_held stays false (validator does not enforce it)",
      con["slice"]["requires_tool"] == ["ButterKnife"] and con["slice"]["requires_held"] is False)
check("exec_contract on disk == build_exec_contract() (generated, never by hand)", V.build_exec_contract() == V.exec_contract())
types = sorted(set(hs.detector_types(det)) | {"ButterKnife", "Apple", "Tomato"})
v = V.validate({"say": "ok", "plan": [{"action": "slice", "object": "Apple", "to": None, "arg": None}]}, types, None)
check("validator accepts a slice plan without any pickup (A26 3(c): no new repair code)", v["ok"] and not v["errors"], str(v["errors"]))
check("no repair code for a cutting tool (no E_NEEDS_TOOL / knife / cut code)",
      not any(w in c for c in V.CODES for w in ("KNIFE", "CUT", "NEEDS_TOOL")), V.CODES)
check("ft-v2 system prompt example no longer names the Knife", not type_scope.mentions_excluded(PC.SYSTEM_V2), type_scope.mentions_excluded(PC.SYSTEM_V2))

print("\n8. no agent-facing text names the excluded type")
import ltm_procedural as LP                                           # noqa: E402
kb = LP.load_kb(verified=False)
bad = [r["id"] for r in kb.records() if type_scope.mentions_excluded(r["text"]["en"]) or type_scope.mentions_excluded(r["text"].get("fa", ""))]
check("procedure records: no Knife in any EN / FA text", not bad, str(bad[:5]))
tp = json.load(open(os.path.join(HERE, "step5", "l3_templates.json"), encoding="utf-8"))
bad = [x["id"] for x in tp["templates"] if type_scope.mentions_excluded(json.dumps(x, ensure_ascii=False))]
check("shipped L3 templates: no Knife", not bad, str(bad))
bad = [n for n, c in con.items() if type_scope.mentions_excluded(json.dumps(c))]
check("tool contract: no Knife", not bad, str(bad))
recs = [json.loads(ln) for ln in open(os.path.join(HERE, "..", "docs", "final-test", "kitchen_memory", "briefings",
                                                    "FloorPlan1.records.jsonl"), encoding="utf-8") if ln.strip()]
bad = [r["id"] for r in recs if type_scope.mentions_excluded(r["text"]["en"]) or type_scope.mentions_excluded(r["text"].get("fa", ""))]
check("FloorPlan1 briefing records: no Knife", not bad, str(bad))

print(f"\n{N - FAIL}/{N} checks passed")
sys.exit(1 if FAIL else 0)
