"""Gate for sim/final_test/signal/twopass.py (stage I-F4-twopass). Offline, Windows `python`, no Ollama, no simulator.
Run from the repo root:  python sim/_sig_twopass_selftest.py
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "sim", "final_test", "signal"))
import twopass as T  # noqa: E402
import planners as P  # noqa: E402

N = [0, 0]


def ok(c, name):
    N[1] += 1
    N[0] += bool(c)
    if not c:
        print("FAIL:", name)


TYPES = ["Apple", "Egg", "Microwave", "Fridge", "Knife", "Bread", "SinkBasin", "Cabinet", "CounterTop", "Bowl"]


def form(ents, ops, dest=None, imp=()):
    return {"entities": [{"name": n, "role": r} for n, r in ents], "operations": list(ops), "destination": dest,
            "implicit": [{"need": n, "contract": c} for n, c in imp]}


# --- schema / form validation
ok(not T.check_form(form([("Egg", "item")], ["heat"]), TYPES), "valid form")
ok(T.check_form(form([("Egg", "banana")], ["heat"]), TYPES), "bad role rejected")
ok(T.check_form(form([("Egg", "item")], ["fly"]), TYPES), "bad operation rejected")
ok(T.check_form(form([("Egg", "item")], []), TYPES), "empty operations rejected")
ok(T.check_form([1], TYPES), "non-object rejected")
ok(T.check_form({"entities": []}, TYPES), "missing keys rejected")
ok(T.check_form(form([("Egg", "item")], ["heat"], imp=[("x", "made_up")]), TYPES), "unknown contract key rejected by enum")

# --- triage: eligible vs uncertain
f = form([("Egg", "item")], ["heat"], imp=[("close the door first", "microwave_heat"), ("wash it", "sink_wash"),
                                           ("x", "none"), ("y", "fridge_chill")])
t = T.triage(f, TYPES)
ok([r["contract"] for r in t["supported"]] == ["microwave_heat"], "only the contract whose trigger is present is eligible")
ok({r["contract"] for r in t["uncertain"]} == {"sink_wash", "none", "fridge_chill"}, "the rest are uncertain")
ok(all("why" in r for r in t["uncertain"]), "uncertain items say why")
fu = form([("Egg", "item"), ("Unicorn", "item")], ["heat"])
ok(T.triage(fu, TYPES)["unknown_entities"] == ["Unicorn"], "unknown entity flagged")
t2 = T.triage(form([("Apple", "item")], ["put"], dest="Fridge", imp=[("keep cold", "fridge_chill")]), TYPES)
ok(len(t2["supported"]) == 1, "appliance in destination makes the contract eligible")

# --- retrieval: operation-keyed, cap, order, not duplicated
f3 = form([("Apple", "item")], ["chill", "put", "find", "wash", "cut"])
k = T.contract_keys_for(f3, T.triage(f3, TYPES))
ok(len(k) == T.MAX_RECORDS == 3, "cap 3 records")
ok(len(set(k)) == len(k), "no duplicate records")
fo = form([("Egg", "item")], ["other"])
ok(T.contract_keys_for(fo, T.triage(fo, TYPES)) == [], "no operation -> no evidence")
ok(T.contract_keys_for(f, t) == ["microwave_heat"], "heat -> microwave recipe, eligible duplicate not repeated")

# --- applicability: closing-after-put only for an openable receptacle
fs = form([("Lettuce", "item")], ["put"], dest="SinkBasin", imp=[("close it", "store_in_openable")])
ok(T.contract_keys_for(fs, T.triage(fs, TYPES)) == [] and not T.triage(fs, TYPES)["supported"], "put into a sink does not pull the close-it recipe")
ff = form([("Apple", "item")], ["put"], dest="Fridge")
ok(T.contract_keys_for(ff, T.triage(ff, TYPES)) == ["store_in_openable"], "put into a Fridge does")

# --- rendering: holding-aware, ordered steps
r = T.render_evidence(["microwave_heat"], "Egg", None)
ok(r.index("put_in Microwave") < r.index("close Microwave") < r.index("toggle_on Microwave"), "microwave recipe order put, close, on")
ok("pick_and_place Apple to Fridge" in T.render_evidence(["fridge_chill"], "Apple", None), "fridge recipe uses pick_and_place when not held")
rh = T.render_evidence(["fridge_chill"], "Apple", "Apple")
ok("put_in Fridge" in rh and "pick_and_place" not in rh, "holding-aware: held target uses put_in")
ok("already in your hand" in T.render_evidence(["find_then_take"], "Apple", "Apple"), "holding-aware find")
ok(T.main_item(form([("Microwave", "appliance"), ("Egg", "item")], ["heat"]), TYPES) == "Egg", "main item = known item entity")

# --- notes shared by TP and TP0: known types only, eligible needs only, no contract text
nt = T.render_notes(f, t, TYPES)
ok("Egg (item)" in nt and "Operations: heat" in nt and "close the door first" in nt, "notes carry entities, ops, eligible needs")
ok("wash it" not in nt and "Recipes" not in nt and "put_in" not in nt, "uncertain needs and contract text are NOT in the notes")
ok("Unicorn" not in T.render_notes(fu, T.triage(fu, TYPES), TYPES), "unknown entity not rendered")

# --- contracts are generic
con = json.load(open(os.path.join(ROOT, "sim", "planners", "exec_contract.json"), encoding="utf-8"))["tools"]
for key, c in T.CONTRACTS.items():
    for fld in ("steps", "held"):
        for tool in re.findall(r"\b([a-z]+_[a-z_]+)\b", c[fld]):
            if tool in ("search_for", "pick_and_place", "put_in", "toggle_on", "toggle_off", "go_to"):
                ok(tool in con, f"{key}.{fld} names a real tool {tool}")
ok(all(set(c["ops"]) <= set(T.OPS) for c in T.CONTRACTS.values()), "contract ops are in the closed vocabulary")
ok(all(not re.search(r"FloorPlan|\bfp\d|g-l\d", json.dumps(c)) for c in T.CONTRACTS.values()), "no scene/item ids in contracts")
src = open(os.path.join(ROOT, "sim", "final_test", "signal", "twopass.py"), encoding="utf-8").read()
import ast  # noqa: E402
tree = ast.parse(src)
BAD = {"it", "by", "ref", "man", "S"}
fns = {"triage", "contract_keys_for", "main_item", "render_evidence", "render_notes", "run_pass1", "two_pass"}
used = {n.id for f_ in tree.body if isinstance(f_, ast.FunctionDef) and f_.name in fns for n in ast.walk(f_) if isinstance(n, ast.Name)}
ok(not (used & BAD), "triage/retrieval/render/two_pass never touch a reference plan or an item row")

# --- end to end with a fake provider (no network)


class Fake:
    def __init__(self, replies):
        self.replies, self.n, self.seen = replies, 0, []

    def plan(self, messages, schema, budget=None, replicate=0, use_cache=True, stage="plan"):
        raw = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        self.seen.append(messages)
        parsed, note = P.parse_reply(raw)
        return {"raw": raw, "parsed": parsed, "parse_note": note, "tokens_in": 100, "tokens_out": 20, "cost_usd": 0.0,
                "latency_s": 0.5, "cached": False, "error": "", "error_kind": "", "stop_reason": "stop"}


PLAN = json.dumps({"say": "ok", "plan": [{"action": "pickup", "object": "Egg", "to": None, "arg": None}]})
import honest_skills as hs  # noqa: E402
contract = P.PlannerContract(TYPES, tools=hs.TOOLS, version="ft-v2")
P1 = json.dumps(form([("Egg", "item"), ("Microwave", "appliance")], ["heat"],
                     imp=[("close the door", "microwave_heat"), ("wash", "sink_wash")]))


def run(arm, replies, cmd="heat the egg"):
    fake = Fake(replies)
    return T.two_pass(arm, cmd, None, TYPES, fake, P.PlannerRunner(fake, contract)), fake


for arm in ("TP", "TP0"):
    r, fake = run(arm, [P1, PLAN])
    ok(r["pass1_ok"] and r["ok"], f"{arm} end to end ok")
    ok(("Recipes:" in (r["block"] or "")) == (arm == "TP"), f"{arm}: recipes only in TP")
    ok("Egg (item)" in r["block"], f"{arm}: shared notes present")
    ok(r["tokens_in"] == 200 and r["tokens_out"] == 40 and r["gen_secs"] == 1.0, f"{arm}: tokens and seconds summed over both passes")
    ok("heat the egg" in fake.seen[0][1]["content"] and "Recipes" not in fake.seen[0][1]["content"], f"{arm}: pass 1 sees only the instruction")
    ok(r["triage"]["uncertain"] and r["triage"]["uncertain"][0]["contract"] == "sink_wash", f"{arm}: uncertain item logged")
r, _ = run("TP", ["not json", "still not", PLAN])
ok(not r["pass1_ok"] and r["block"] is None and r["ok"] and r["pass1"]["calls"] == 2, "invalid pass 1 degrades to an unassisted pass 2 (1 repair)")
r, _ = run("TP", ["{}", P1, PLAN])
ok(r["pass1_ok"] and r["pass1"]["calls"] == 2 and "Recipes:" in r["block"], "pass 1 repaired once")
a, _ = run("TP", [P1, PLAN])
b, _ = run("TP0", [P1, PLAN])
ok(a["block"].split("\nRecipes:")[0] == b["block"], "TP0 block == TP block minus the recipes (matched control)")
ok(not any("twopass" in open(os.path.join(ROOT, "sim", "final_test", "signal", fn), encoding="utf-8").read()
           for fn in ("signal_pkg_plan.py", "signal_pkg_exec.py", "package_removals.py")), "not wired into the default package files")

print(f"{N[0]}/{N[1]} " + ("OK" if N[0] == N[1] else "FAILED"))
sys.exit(0 if N[0] == N[1] else 1)
