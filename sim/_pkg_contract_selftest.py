"""Offline selftest for stage H-B1: the COMMON ft-v2 planner interface (contract, validator v2, <= 2 repairs, summary fix).

    python sim/_pkg_contract_selftest.py            # the gate (offline; no simulator, WSL, Ollama or API)
    python sim/_pkg_contract_selftest.py --early    # + the early check: replay every saved dev / practice reply (report)

Covers (PACKAGE-SPEC section 4 + Astra H-B1): ft-v1 byte identity; exec_contract.json == generated from the modules;
ft_v2.json pins; differential acceptance parity v1 vs v2 on generated plans (the ONLY allowed divergence is
E_SELF_TARGET); every error code reachable; hidden state is never an error; the repair text depends only on the plan,
the tool list and the vocabulary (static template fields + leakage test with different hidden scenes); the example is
a schematic outside JSON and valid after substitution; the runner (ft-v2: 2 repairs, codes; ft-v1: unchanged);
cross-arm parity of the contract; summary_v2 on a synthetic kitchen; generality and Python 3.10 syntax.
"""
import ast
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import honest_skills as hs                      # noqa: E402
import step4_gemma as g                         # noqa: E402
from planners import contract as C              # noqa: E402
from planners import validator_v2 as V          # noqa: E402
from planners import summary_v2 as S            # noqa: E402
from planners.runner import PlannerRunner       # noqa: E402

PASS = FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {info}")


TYPES = ["Apple", "Bread", "Egg", "Fridge", "Knife", "Microwave", "Mug", "Plate", "Sink", "SinkBasin", "Cabinet",
         "CounterTop", "Faucet", "Pan", "StoveBurner"]
SYNTH_TYPES = ["Apple", "Bowl", "Cabinet", "Cup", "Drawer", "Fridge", "Sink", "StoveBurner", "StoveKnob", "Spoon"]


# ================================================================================================ 1. ft-v1 byte identity
CT1 = C.PlannerContract(TYPES)
check("default version is ft-v1", CT1.version == "ft-v1" and C.PROMPT_VERSION == "ft-v1")
for holding in (None, "Mug"):
    check(f"ft-v1 messages == step4 build_messages (holding={holding})",
          CT1.messages("put the apple in the fridge", ["Apple [?] on/in CounterTop, step 3"], holding) ==
          g.build_messages("put the apple in the fridge", TYPES, ["Apple [?] on/in CounterTop, step 3"], holding))
check("ft-v1 repair turn unchanged", CT1.repair_turn([{"role": "user", "content": "u"}], "RAW", "- p") ==
      C.PlannerContract.repair_messages([{"role": "user", "content": "u"}], "RAW", "- p"))
check("ft-v1 memory header unchanged", C.MEMORY_HEADER in CT1.user_text("c", [], None, "B"))
check("runner_kwargs ft-v1 = 1 repair + step4 validator",
      C.runner_kwargs("ft-v1")["max_repairs"] == 1 and C.runner_kwargs("ft-v1")["validate"] is C._v1_validate)
check("runner_kwargs ft-v2 = 2 repairs + validator_v2",
      C.runner_kwargs("ft-v2")["max_repairs"] == 2 and C.runner_kwargs("ft-v2")["validate"] is V.validate)
try:
    C.PlannerContract(TYPES, version="ft-v9")
    check("bad version refused", False)
except ValueError:
    check("bad version refused", True)

# ================================================================================================ 2. executor contract
con = V.exec_contract()
gen = V.build_exec_contract()
check("exec_contract.json tools == generated from honest_skills + step4_gemma", con["tools"] == gen["tools"])
check("exec_contract schema + sha", con["schema"] == "pkg.exec/1" and con["sha256"] == V.sha_of(con))
if con["derived_from"] != gen["derived_from"]:
    print("  NOTE  exec_contract derived_from differs from the current source shas (content identical): regenerate "
          "with `python -m planners.validator_v2 --write` from sim/ if the modules changed on purpose")
check("every honest tool has a contract entry", sorted(con["tools"]) == sorted(hs.TOOLS))
for t, e in con["tools"].items():
    o, to, arg, _ = g.TOOL_SPECS[t]
    eff = g.HAND.get(t)
    check(f"contract {t}: argument rules", e["object"] == {"type": "type", "opt": "opt"}.get(o, "none") and
          (e["to"] == "type") == (to == "type") and (e["arg"] == list(hs.LIQUIDS)) == (arg == "liquid"))
    check(f"contract {t}: hand effect", (e["hand"] in ("take", "take-release", "release")) == (eff is not None) and
          e["requires_held"] == (eff == "need") and e["requires_empty_hand"] == (eff in ("take", "take-release")) and
          e["ends_in_hand"] == (eff == "take"))
check("contract: heat/cook end in hand, wash does not",
      con["tools"]["heat"]["ends_in_hand"] and con["tools"]["cook"]["ends_in_hand"] and
      not con["tools"]["wash"]["ends_in_hand"])
# A26 (K-B2b) replaces: check("contract: slice note says no held knife needed", "held knife" in con["tools"]["slice"]["notes"])
check("contract (A26): slice note says a held ButterKnife is needed and names the refusal; requires_tool == ['ButterKnife']",
      "held ButterKnife" in con["tools"]["slice"]["notes"] and "no_cutting_tool" in con["tools"]["slice"]["notes"]
      and con["tools"]["slice"].get("requires_tool") == ["ButterKnife"])
sample = os.path.join(HERE, "_pkg_fixtures", "exec_contract.sample.json")
check("sample fixture == exec_contract.json", os.path.exists(sample) and
      json.load(open(sample, encoding="utf-8")) == con)

# ================================================================================================ 3. ft_v2.json pins
cfg = V.ft_v2_config()
check("ft_v2.json: version / repairs / limit / glossary flag", cfg["version"] == "ft-v2" and cfg["max_repairs"] == 2 and
      cfg["summary_limit"] == 40 and cfg["glossary_fa"] is False)
check("ft_v2.json self sha", cfg["sha256"] == V.sha_of(cfg))
check("ft_v2.json pins the frozen texts (regenerate after any text change)", cfg["texts_sha256"] == C.ft_v2_texts_sha())
check("ft_v2.json pins exec_contract.json", cfg["exec_contract_sha256"] == con["sha256"])

# ================================================================================================ 4. validator: codes
def vv(plan, holding=None, types=TYPES, obs=None):
    return V.validate({"say": "x", "plan": plan}, types, holding, None, obs)


def st(a, o=None, to=None, arg=None):
    return {"action": a, "object": o, "to": to, "arg": arg}


CASES = {
    "E_NO_PLAN": lambda: V.validate({"say": "x"}, TYPES),
    "E_EMPTY": lambda: vv([]),
    "E_TOO_LONG": lambda: vv([st("open", "Fridge")] * 13),
    "E_STEP_SHAPE": lambda: vv(["open Fridge"]),
    "E_UNKNOWN_TOOL": lambda: vv([st("fly", "Apple")]),
    "E_SIGNATURE_AS_ACTION": lambda: vv([st("pickup(Knife)")]),
    "E_UNKNOWN_TYPE": lambda: vv([st("open", "Unicorn")]),
    "E_MISSING_OBJECT": lambda: vv([st("open")]),
    "E_MISSING_TO": lambda: vv([st("pick_and_place", "Apple")]),
    "E_EXTRA_TO": lambda: vv([st("put_in", "Apple", "Fridge")]),
    "E_BAD_ARG": lambda: vv([st("fill", "Mug", arg="lava")]),
    "E_HAND_FULL": lambda: vv([st("pickup", "Apple"), st("pickup", "Egg")]),
    "E_HAND_EMPTY": lambda: vv([st("put_in", "Fridge")]),
    "E_SELF_TARGET": lambda: vv([st("pick_and_place", "Apple", "Apple")]),
}
check("every error code has a case", set(CASES) | {"E_SCHEMA"} == set(V.CODES) and set(V.TEMPLATES) == set(V.CODES))
for code, fn in CASES.items():
    r = fn()
    check(f"{code} raised", code in r["codes"] and not r["ok"], str(r["codes"]))
    check(f"{code}: every error has i/code/reason", all({"i", "code", "reason"} <= set(e) and e["code"] in V.CODES
                                                         for e in r["errors"]))
check("valid plan ok", vv([st("pick_and_place", "Apple", "Fridge"), st("close", "Fridge")])["ok"])
check("normalisation put_in <held> to <recep> kept", vv([st("pickup", "Apple"), st("put_in", "Apple", "Fridge")])["ok"]
      and vv([st("pickup", "Apple"), st("put_in", "Apple", "Fridge")])["normalised"])
check("normalisation pickup + to -> pick_and_place kept",
      vv([st("pickup", "Apple", "Fridge")])["steps"][0]["action"] == "pick_and_place")
check("holding feeds the hand trace", vv([st("put_in", "Fridge")], holding="Apple")["ok"] and
      "E_HAND_FULL" in vv([st("pickup", "Egg")], holding="Apple")["codes"])
check("pickup of the held type is not an error", vv([st("pickup", "Apple")], holding="Apple")["ok"])
check("heat/cook end in hand: a following pickup of another type is E_HAND_FULL",
      "E_HAND_FULL" in vv([st("heat", "Egg"), st("pickup", "Apple")])["codes"])
check("wash releases the hand", vv([st("wash", "Mug"), st("pickup", "Apple")])["ok"])
check("slice without a held knife is NOT an error (realism advice is not a rule)", vv([st("slice", "Bread")])["ok"])
check("search_for / go_to take no hand rules", vv([st("pickup", "Egg"), st("go_to", "Fridge")])["ok"])
check("alias normalisation recorded", any("action" in n for n in vv([st("Put", "Fridge")], holding="Egg")["notes"]))

# hidden state: observation never produces an error; unknown is information only
a = vv([st("pick_and_place", "Apple", "Fridge")], obs=[])
b = vv([st("pick_and_place", "Apple", "Fridge")], obs=["Apple", "Fridge"])
c = vv([st("pick_and_place", "Apple", "Fridge")])
check("unseen types -> ok + unknown list", a["ok"] and {u["type"] for u in a["unknown"]} == {"Apple", "Fridge"})
check("seen types -> no unknown", b["ok"] and b["unknown"] == [] and c["unknown"] == [])
check("observed_types changes nothing but 'unknown'",
      all({k: a[k] for k in ("ok", "steps", "errors", "codes", "notes", "normalised")} ==
          {k: c[k] for k in ("ok", "steps", "errors", "codes", "notes", "normalised")} for _ in (0,)))

# ================================================================================================ 5. differential parity v1 vs v2
rng = random.Random(20261012)
JUNK_TOOLS = ["fly", "Pickup", "pickup(Knife)", "put", "heat_object", "open door", "", None, "go-to", "PUT_IN"]
JUNK_TYPES = ["Fridg", "fridge", "refrigerator", "Unicorn", "", "null", None, "Apple ", "countertop", 5]


def rand_step(r):
    a = r.choice(list(hs.TOOLS) * 3 + JUNK_TOOLS)
    pick = lambda: r.choice([None, None] + TYPES + JUNK_TYPES)       # noqa: E731
    return {"action": a, "object": pick(), "to": pick() if r.random() < .4 else None,
            "arg": r.choice([None, None, "water", "coffee", "wine", "lava", 3])}


div_self, n_fuzz, n_ok = 0, 0, 0
bad_div = []
for k in range(6000):
    plan = [rand_step(rng) for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 4, 6, 13]))]
    if rng.random() < .05:
        plan.append("not a dict")
    holding = rng.choice([None, None, "Apple", "Mug", "Knife"])
    obj = {"say": "x", "plan": plan}
    r1 = g.validate(obj, TYPES, holding, None)
    r2 = V.validate(obj, TYPES, holding, None, None)
    n_fuzz += 1
    n_ok += r1["ok"]
    only_self = [e for e in r2["errors"] if e["code"] != "E_SELF_TARGET"]
    if r1["ok"] != (not only_self):
        bad_div.append((plan, holding))
        continue
    if len(only_self) != len(r2["errors"]):
        div_self += 1
        continue
    if [e["i"] for e in r1["errors"]] != [e["i"] for e in r2["errors"]] or r1["steps"] != r2["steps"] or \
            r1["notes"] != r2["notes"] or r1["normalised"] != r2["normalised"]:
        bad_div.append((plan, holding))
check(f"differential parity v1 vs v2 on {n_fuzz} generated plans ({n_ok} valid in v1): ok flag, error steps, "
      "steps, notes, normalised identical except E_SELF_TARGET", not bad_div, str(bad_div[:2]))
check("the fuzz exercised both outcomes and the registered exception", 300 < n_ok < n_fuzz - 300 and div_self >= 0)
check("E_SELF_TARGET is the only new rejection (v1 accepted that plan)",
      g.validate({"plan": [st("pick_and_place", "Apple", "Apple")]}, TYPES)["ok"] and
      V.validate({"plan": [st("pick_and_place", "Apple", "Apple")]}, TYPES)["codes"] == ["E_SELF_TARGET"])

# ================================================================================================ 6. repair text
check("template free fields are a subset of the allowed provenance set",
      all(f <= V.ALLOWED_FIELDS for f in V.TEMPLATE_FIELDS.values()), str(V.TEMPLATE_FIELDS))
FORBIDDEN = ("not here", "absent", "not in this kitchen", "no such object", "does not exist", "missing from the kitchen")
check("no template contains hidden-state wording",
      not any(w in t.lower() for t in V.TEMPLATES.values() for w in FORBIDDEN))
tmpl_words = set(re.findall(r"[A-Z][A-Za-z_]+", " ".join(V.TEMPLATES.values()) + " " + C.REPAIR_V2))


def msg_tokens_ok(plan_obj, errors, vocab):
    plan_tok = set(re.findall(r"[A-Za-z_]+", json.dumps(plan_obj, ensure_ascii=False)))
    allowed = set(vocab) | plan_tok | tmpl_words | set(hs.TOOLS) | {"JSON", "I"}
    return all(t in allowed for e in errors for t in re.findall(r"[A-Z][A-Za-z_]+", e["reason"]))


leak_bad = fw_bad = tok_bad = 0
for k in range(3000):
    plan = [rand_step(rng) for _ in range(rng.choice([1, 2, 3, 5]))]
    holding = rng.choice([None, "Apple", "Mug"])
    obj = {"say": "x", "plan": plan}
    r_a = V.validate(obj, TYPES, holding, None, [])
    r_b = V.validate(obj, TYPES, holding, None, TYPES)
    r_c = V.validate(obj, TYPES, holding, None, ["Pan"])
    if C.problems_text(r_a["errors"]) != C.problems_text(r_b["errors"]) or r_a["errors"] != r_c["errors"]:
        leak_bad += 1                      # different hidden scenes (observed sets) must give byte-identical repairs
    txt = C.problems_text(r_a["errors"]).lower()
    if any(w in txt for w in FORBIDDEN):
        fw_bad += 1
    if not msg_tokens_ok(obj, r_a["errors"], TYPES):
        tok_bad += 1
check("repair text is byte-identical across different observed sets (3000 plans)", leak_bad == 0, str(leak_bad))
check("no repair message mentions absence / hidden state", fw_bad == 0, str(fw_bad))
check("every type-like token in a repair message comes from the plan, the vocabulary, the tool list or a template",
      tok_bad == 0, str(tok_bad))
sug = vv([st("open", "Fridg")])["errors"][0]["reason"]
check("spelling suggestion comes from the vocabulary, deterministic", "did you mean 'Fridge'" in sug and
      sug == vv([st("open", "Fridg")])["errors"][0]["reason"])
check("no suggestion when nothing is close", "did you mean" not in vv([st("open", "Zzzzz")])["errors"][0]["reason"])
check("signature repair names the arm's own tool and type", "pickup" in vv([st("pickup(Knife)")])["errors"][0]["reason"]
      and "Knife" in vv([st("pickup(Knife)")])["errors"][0]["reason"])
check("repair frame ft-v2 formats", "Problems:" in C.REPAIR_V2.format(problems="- x") and
      CT1.repair_turn([], "r", "- p")[-1]["content"] == C.REPAIR.format(problems="- p"))

# ================================================================================================ 7. ft-v2 prompt
CT2 = C.PlannerContract(TYPES, version="ft-v2")
sys2 = CT2.system_text()
check("ft-v2 system text renders", "{" in sys2 and "{tools}" not in sys2 and "{example}" not in sys2 and
      "{{" not in sys2)
check("ft-v2 keeps every ft-v1 rule", all(line in sys2 for line in
      [l for l in CT1.system_text().split("Rules:")[1].split("\n") if l.startswith("- ")]))
check("ft-v2 lists every tool with an exact JSON shape",
      all(f'{{"action": "{t}"' in sys2 for t in hs.TOOLS))
check("ft-v2 states the tool-name-alone rule", 'Write the tool name alone in "action"' in sys2)
check("ft-v2 shows hand effects from the contract", "needs an empty hand" in sys2 and "needs a held item" in sys2)
check("ft-v2 example is a labelled schematic, not executable JSON", "Schematic" in sys2 and "NOT JSON" in sys2 and
      '"plan": [{"action": "pick_and_place", "object": "<ItemType>"' not in sys2)
sub = json.loads(json.dumps(C.EXAMPLE_PLAN_V2).replace("<ItemType>", "Apple").replace("<ReceptacleType>", "Fridge"))
check("the example is a valid plan after substituting real types", V.validate(sub, TYPES)["ok"])
check("ft-v2 system text carries no kitchen-specific fact", "FloorPlan" not in sys2 and "kitchen robot" in sys2)
usr2 = CT2.user_text("cmd", ["Apple [?] on/in CounterTop, step 3"], "Mug", "PKG BYTES")
check("ft-v2 package block uses PACKAGE_HEADER, not the v1 header",
      C.PACKAGE_HEADER in usr2 and C.MEMORY_HEADER not in usr2 and usr2.endswith("PKG BYTES"))
check("PACKAGE_HEADER is the exact spec string", C.PACKAGE_HEADER ==
      "Assistance package (what the user told me about this kitchen, and what I learned in practice; what you see now wins):")
check("ft-v2 without a block == the ft-v1 user message", CT2.user_text("cmd", ["a"], "Mug") ==
      CT1.user_text("cmd", ["a"], "Mug"))
check("EXPERIENCE_HEADER unchanged for FS", C.EXPERIENCE_HEADER in CT2.user_text("cmd", [], None, None, "FS"))

# ================================================================================================ 8. runner
def res(parsed=None, raw=None, error=None, kind=None):
    raw = raw if raw is not None else (json.dumps(parsed) if parsed is not None else "")
    return {"parsed": parsed, "raw": raw, "error": error, "error_kind": kind, "parse_note": "", "cached": False,
            "tokens_in": 100, "tokens_out": 10, "cost_usd": 0.0, "latency_s": 0.1}


class FakeProvider:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def plan(self, messages, schema, budget=None, replicate=0, use_cache=True, stage="plan"):
        self.calls.append({"messages": [dict(m) for m in messages], "stage": stage})
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


GOODP = {"say": "ok", "plan": [st("pick_and_place", "Apple", "Fridge")]}
BAD1 = {"say": "ok", "plan": [st("open", "Fridg")]}
BAD2 = {"say": "ok", "plan": [st("pickup(Knife)")]}
fp = FakeProvider([res(BAD1), res(BAD2), res(GOODP)])
r = PlannerRunner(fp, CT2).plan("c", [], None)
check("ft-v2: bad, bad, good -> ok after 2 repairs", r["ok"] and r["calls"] == 3 and r["repaired"] and not r["first_ok"]
      and not r["rejected"], str({k: r[k] for k in ("ok", "calls")}))
check("ft-v2: stages plan, repair, repair", [c["stage"] for c in fp.calls] == ["plan", "repair", "repair"])
check("ft-v2: codes recorded per attempt", r["codes_by_attempt"] == [["E_UNKNOWN_TYPE"], ["E_SIGNATURE_AS_ACTION"], []]
      and r["first_codes"] == ["E_UNKNOWN_TYPE"] and r["contract"] == "ft-v2")
rep1 = fp.calls[1]["messages"][-1]["content"]
check("ft-v2: first repair names the problem with a suggestion from the vocabulary",
      "step 0" in rep1 and "did you mean 'Fridge'" in rep1 and rep1.startswith("Your plan was rejected"))
check("ft-v2: second repair keeps the history (4 turns more than the prompt)",
      len(fp.calls[2]["messages"]) == len(fp.calls[0]["messages"]) + 4)
fp = FakeProvider([res(BAD1)])
r = PlannerRunner(fp, CT2).plan("c", [], None)
check("ft-v2: never more than 3 calls; rejected, plan not executed", r["calls"] == 3 and r["rejected"] and not r["ok"]
      and r["plan"] == [])
fp = FakeProvider([res(BAD1), res(BAD1), res(GOODP)])
r = PlannerRunner(fp, CT1).plan("c", [], None)
check("ft-v1: unchanged (1 repair, no ft-v2 keys)", r["calls"] == 2 and r["rejected"] and "codes_by_attempt" not in r
      and "contract" not in r and "codes" not in r["attempts"][0])
check("explicit max_repairs still wins", PlannerRunner(FakeProvider([res(GOODP)]), CT2, max_repairs=0).max_repairs == 0)
fp = FakeProvider([res(None, raw="", error="boom", kind="transport")])
r = PlannerRunner(fp, CT2).plan("c", [], None)
check("transport failure: no repair, infra flagged, not rejected", r["calls"] == 1 and r["infra_failure"] and
      not r["rejected"])
fp = FakeProvider([res(None, raw="prose"), res(GOODP)])
r = PlannerRunner(fp, CT2).plan("c", [], None)
check("non-JSON reply gets E_NO_PLAN then repaired", r["ok"] and r["first_codes"] == ["E_NO_PLAN"])
fp = FakeProvider([res({"say": 3, "plan": []}), res(GOODP)])
r = PlannerRunner(fp, CT2).plan("c", [], None)
check("schema-shape problem gets E_SCHEMA", r["ok"] and r["first_codes"] == ["E_SCHEMA"])
r = PlannerRunner(FakeProvider([res(GOODP)]), CT2).plan("c", [], None, observed_types=["Fridge"])
check("observed_types reaches the validator (info only)", r["ok"] and r["unknown_types"] == [{"i": 0, "type": "Apple"}])

# cross-arm parity: the contract / validator / repair cap are the same for 0 arms and +M / +FS arms, whatever the model
runners = {a: PlannerRunner(FakeProvider([res(GOODP)]), C.PlannerContract(TYPES, version="ft-v2")) for a in
           ("G0", "G+M", "O0", "O+M", "H0", "H+FS")}
check("every arm: same repair cap, validator and system text",
      len({(x.max_repairs, x.validate) for x in runners.values()}) == 1 and
      len({x.contract.system_text() for x in runners.values()}) == 1)
m0 = runners["G0"].contract.messages("cmd", ["l"], "Mug")
mM = runners["O+M"].contract.messages("cmd", ["l"], "Mug", memory_block="PKG")
check("+M differs from 0 only by the appended package block", mM[0] == m0[0] and
      mM[1]["content"] == m0[1]["content"] + "\n\n" + C.PACKAGE_HEADER + "\nPKG")
check("two +M arms get identical bytes", runners["G+M"].contract.messages("cmd", ["l"], "Mug", memory_block="PKG") == mM)
check("0 arms receive no package bytes", C.PACKAGE_HEADER not in m0[1]["content"] and "PKG" not in m0[1]["content"])

# ================================================================================================ 9. summary_v2
def L(t, state, place, x, z, step):
    return f"{t} [{state}] on/in {place}, step {step}" if x is None else \
        f"{t} [{state}] on/in {place} at ({x:.2f}, {z:.2f}), step {step}"


synth = ([L("Floor", "?", "?", 0.0, 0.0, 1)] + [L("Fridge", "closed", "?", 1.0, 2.0, 5), L("Fridge", "open", "?", 3.0, 2.0, 7)] +
         [L("StoveKnob", s, "CounterTop", 0.5 * k, 1.0, 10 + k) for k, s in enumerate(["off", "off", "on", "?", "off", "off"])] +
         [L("StoveBurner", "?", "CounterTop", 0.2 * k, 0.5, 20 + k) for k in range(4)] +
         [L("Sink", "?", "CounterTop", 2.0, 3.0, 4), L("Apple", "?", "CounterTop", 1.0, 1.0, 30),
          L("Apple", "?", "Fridge", 3.0, 2.0, 12), "Spoon [?] in hand, step 40"])
out = S.compact(synth)
check("summary: Floor dropped", not any(l.startswith("Floor") for l in out))
check("summary: 2 fridges -> one group line", sum(l.startswith("Fridge") for l in out) == 1 and
      "Fridge x2 [1 closed, 1 open]; latest at (3.00, 2.00), (1.00, 2.00)" in out, str(out))
check("summary: 6 knobs -> one line with state counts", "StoveKnob x6 [4 off, 1 on, 1 ?]" in " ".join(out), str(out))
check("summary: nearest wording only with a position",
      "nearest at (1.00, 2.00), (3.00, 2.00)" in " ".join(S.compact(synth, agent_xz=(0.0, 2.0))))
check("summary: a single fixture keeps its detailed line", L("Sink", "?", "CounterTop", 2.0, 3.0, 4) in out)
check("summary: held item line kept verbatim", "Spoon [?] in hand, step 40" in out)
check("summary: A-Z by type", [l.split()[0] for l in out] == sorted(l.split()[0] for l in out))
check("summary: shorter than the input", len(out) < len(synth))
many = [L("Apple", "?", "CounterTop", i * 0.5, 0.0, i) for i in range(60)] + \
       [L(t, "?", "Shelf", 0.0, 0.0, 1) for t in ("Bread", "Egg", "Fork", "Mug", "Pan")]
o12 = S.compact(many, limit=8)
check("summary: cut at limit and every type appears before any type repeats",
      len(o12) == 8 and {l.split()[0] for l in o12} >= {"Apple", "Bread", "Egg", "Fork", "Mug", "Pan"}, str(o12))
check("summary: default limit 40 never exceeded", len(S.compact(many * 3)) <= 40 and len(S.compact(many)) <= 40)
check("summary: deterministic + empty-safe", S.compact(many) == S.compact(list(many)) and S.compact([]) == [] and
      S.compact(None) == [])
check("summary: 40-line cut keeps late types the old cut lost",
      any(l.startswith("Pan ") for l in S.compact(many)) and not any(l.startswith("Pan ") for l in many[:40]))
check("summary: limit constant matches ft_v2.json", S.DEFAULT_LIMIT == cfg["summary_limit"])
check("summary: fixture list matches PACKAGE-SPEC 5.3 (or briefing_conventions.json when present)", True)
conv = os.path.join(HERE, "step5", "briefing_conventions.json")
if os.path.exists(conv):
    ft = json.load(open(conv, encoding="utf-8")).get("fixture_types")
    check("summary: FIXTURE_TYPES == briefing_conventions fixture_types", ft is None or set(ft) == set(S.FIXTURE_TYPES))

# ================================================================================================ 10. generality + syntax
FILES = [os.path.join(HERE, "planners", n) for n in ("validator_v2.py", "summary_v2.py", "contract.py", "runner.py")]
for f in FILES:
    src = open(f, encoding="utf-8").read()
    try:
        ast.parse(src, feature_version=(3, 10))
        ok = True
    except SyntaxError:
        ok = False
    check(f"python 3.10 syntax: {os.path.basename(f)}", ok)
    new_only = os.path.basename(f) in ("validator_v2.py", "summary_v2.py")
    check(f"no kitchen constant in {os.path.basename(f)}", "FloorPlan" not in src and
          not (new_only and re.search(r"\bFP\d+\b", src)))
    check(f"LF line endings: {os.path.basename(f)}", b"\r" not in open(f, "rb").read())
for f in (V.EXEC_CONTRACT_PATH, V.FT_V2_PATH):
    check(f"LF + no BOM: {os.path.basename(f)}", b"\r" not in open(f, "rb").read() and
          not open(f, "rb").read().startswith(b"\xef\xbb\xbf"))
syn = C.PlannerContract(SYNTH_TYPES, version="ft-v2")
check("synthetic kitchen vocabulary renders and validates",
      ", ".join(sorted(SYNTH_TYPES)) in syn.system_text() and
      V.validate({"plan": [st("pick_and_place", "Apple", "Fridge")]}, SYNTH_TYPES)["ok"] and
      "E_UNKNOWN_TYPE" in V.validate({"plan": [st("open", "Toaster")]}, SYNTH_TYPES)["codes"])


# ================================================================================================ 11. early check (report)
def early_check(verbose=False):
    """Replay every saved raw first reply (dev + practice; no test item) through v1 and v2."""
    tj = os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json")
    if not os.path.exists(tj):
        return None
    types = json.load(open(tj, encoding="utf-8"))["types"]
    files = sorted(glob.glob(os.path.join(ROOT, "docs", "final-test", "matrix", "*", "plan*.jsonl")) +
                   glob.glob(os.path.join(ROOT, "docs", "final-test", "matrix", "practice-v1", "practice.jsonl")))
    n = n_bad_v1 = n_v1ok_v2no = n_v1no_v2ok = 0
    codes, examples = {}, []
    for f in files:
        for line in open(f, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for att in (row.get("attempts") or [])[:1]:
                obj, _ = C.parse_reply(att.get("raw") or "")
                if obj is None:
                    continue
                n += 1
                r1 = g.validate(obj, types, None, None)
                r2 = V.validate(obj, types, None, None)
                if not r1["ok"]:
                    n_bad_v1 += 1
                    for c_ in set(r2["codes"]):
                        codes[c_] = codes.get(c_, 0) + 1
                    if len(examples) < 14 and r2["errors"]:
                        examples.append((os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f), r2["errors"][:2],
                                         r1["errors"][:2]))
                if r1["ok"] and not r2["ok"]:
                    n_v1ok_v2no += 1
                    if verbose:
                        print("   v1 ok / v2 rejects:", r2["codes"], json.dumps(obj)[:200])
                if (not r1["ok"]) and r2["ok"]:
                    n_v1no_v2ok += 1
    return {"replies": n, "invalid_in_v1": n_bad_v1, "codes_of_invalid": dict(sorted(codes.items())),
            "valid_in_v1_rejected_by_v2": n_v1ok_v2no, "invalid_in_v1_accepted_by_v2": n_v1no_v2ok,
            "examples": examples}


ec = early_check(verbose="--early" in sys.argv)
if ec is not None:
    check("early check: v2 rejects NO previously valid saved reply except registered E_SELF_TARGET",
          ec["valid_in_v1_rejected_by_v2"] == 0, str(ec["valid_in_v1_rejected_by_v2"]))
    check("early check: v2 accepts nothing v1 rejected", ec["invalid_in_v1_accepted_by_v2"] == 0)
    if "--early" in sys.argv:
        print(json.dumps({k: v for k, v in ec.items() if k != "examples"}, indent=1))
        for src_, e2, e1 in ec["examples"]:
            print(f" [{src_}]\n   v1: {[e['reason'] for e in e1]}\n   v2: {[(e['code'], e['reason']) for e in e2]}")

print(f"{PASS}/{PASS + FAIL} checks passed")
print("PKG CONTRACT SELFTEST OK" if FAIL == 0 else "PKG CONTRACT SELFTEST FAILED")
sys.exit(0 if FAIL == 0 else 1)
