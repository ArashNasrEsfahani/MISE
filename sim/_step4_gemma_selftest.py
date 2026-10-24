"""Offline selftest for sim/step4_gemma.py (s6-gemma) with a MOCKED LLM -- no Ollama, no simulator.

    python sim/_step4_gemma_selftest.py

Covers: tool list / spec agreement, prompt contents (tools, detector types, memory, holding, Persian command
verbatim, nothing from the reference plan or goal), a good plan, a bad verb (rejected with a reason, one repair),
a bad verb twice (rejected, nothing to execute), a Persian command with alias / case normalisation, unknown types,
hand-trace checks, fill args, a non-JSON reply, an empty plan, a provider outage, the search chooser (constrained
to the candidate list, outage -> no ranking), and the reproducibility settings handed to the transport.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import honest_skills as hs          # noqa: E402
import step4_gemma as g             # noqa: E402

PASS = FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {info}")


TYPES = ["Apple", "Bowl", "Bread", "Cabinet", "CoffeeMachine", "CounterTop", "Drawer", "Faucet", "Fridge", "Knife",
         "Microwave", "Mug", "Potato", "SinkBasin", "StoveBurner", "StoveKnob", "Tomato"]
MEM = ["Apple [whole] on/in CounterTop at (-0.50, 0.90), step 3", "Fridge [closed] on/in ? at (-2.10, 1.20), step 5"]


class Mock:
    """Scripted transport: replies in order; records (messages, fmt) of each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, fmt):
        self.seen.append(([dict(m) for m in messages], fmt))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


def P(*steps, say="ok"):
    return {"say": say, "plan": [dict(zip(("action", "object", "to", "arg"), s)) for s in steps]}


def planner(mock, **kw):
    return g.GemmaPlanner(provider=None, types=TYPES, chat=mock, **kw)


# ============================================================================================ 1
print("1. tool list and prompt")
check("TOOL_SPECS covers exactly honest_skills.TOOLS", set(g.TOOL_SPECS) == set(hs.TOOLS),
      str(set(g.TOOL_SPECS) ^ set(hs.TOOLS)))
check("every tool is something exec_step dispatches (VERBS or composite)",
      all(t in hs.VERBS or t in ("pick_and_place", "put_in", "heat", "cook", "wash", "go_to", "search_for", "drop",
                                 "close_all") or t in hs.NAV for t in g.TOOL_SPECS))
msgs = g.build_messages("سیب را داخل یخچال بگذار.", TYPES, MEM, "Bread")
sys_txt, usr = msgs[0]["content"], msgs[1]["content"]
check("system prompt lists every tool", all(f"- {t}(" in sys_txt for t in hs.TOOLS))
check("system prompt lists the detector types", "Apple, Bowl, Bread" in sys_txt and "StoveKnob" in sys_txt)
check("prompt says the vocabulary is not an inventory", "not every type is necessarily in this kitchen" in sys_txt)
check("user turn carries memory, holding and the Persian command verbatim",
      MEM[0] in usr and "holding: Bread" in usr and "سیب را داخل یخچال بگذار." in usr)
check("empty memory / hand render as such",
      "(nothing yet)" in g.build_messages("x", TYPES, [], None)[1]["content"]
      and "holding: nothing" in g.build_messages("x", TYPES, [], None)[1]["content"])
leak = [w for w in ("isPickedUp", "isOpen", "parentReceptacles", "goal", "reference", "objectId", "gold")
        if w in sys_txt + usr]
check("prompt carries no goal / reference / simulator-state words", not leak, str(leak))

# ============================================================================================ 2
print("\n2. good plan (EN)")
m = Mock(P(("pick_and_place", "Apple", "Fridge", None), ("close", "Fridge", None, None)))
r = planner(m).plan("Put the apple in the fridge and close the door.", MEM, None)
check("valid on the first pass, no repair", r["ok"] and r["first_ok"] and not r["repaired"] and len(r["attempts"]) == 1)
check("plan is exec_step's format", r["plan"] == [{"action": "pick_and_place", "object": "Apple", "to": "Fridge",
                                                    "arg": None},
                                                   {"action": "close", "object": "Fridge", "to": None, "arg": None}],
      str(r["plan"]))
check("transport got the JSON-Schema format", m.seen[0][1] is g.PLAN_SCHEMA)
check("raw + parsed + validation saved", r["attempts"][0]["raw"] and r["attempts"][0]["parsed"]["plan"]
      and r["attempts"][0]["validation"]["ok"])

# ============================================================================================ 3
print("\n3. bad verb -> one repair")
m = Mock(P(("teleport", "Fridge", None, None)), P(("open", "Fridge", None, None)))
pl = planner(m)
r = pl.plan("Open the fridge.", MEM, None)
v0 = r["attempts"][0]["validation"]
check("bad verb rejected with a reason", not v0["ok"] and "unknown action 'teleport'" in v0["errors"][0]["reason"],
      str(v0["errors"]))
rep = m.seen[1][0]
check("repair turn = previous answer + the reasons", rep[-2]["role"] == "assistant" and "teleport" in rep[-2]["content"]
      and rep[-1]["role"] == "user" and "unknown action 'teleport'" in rep[-1]["content"])
check("repair text carries only tool/type-list reasons (no scene facts)",
      rep[-1]["content"].startswith("Your plan was rejected") and "Fridge [closed]" not in rep[-1]["content"])
check("repaired plan accepted", r["ok"] and r["repaired"] and not r["first_ok"]
      and r["plan"][0]["action"] == "open")
check("call log: plan then repair", [c["kind"] for c in pl.calls] == ["plan", "repair"])

print("\n4. bad verb twice -> rejected, nothing to execute")
m = Mock(P(("teleport", "Fridge", None, None)), P(("fly", "Fridge", None, None)))
r = planner(m).plan("Open the fridge.", MEM, None)
check("at most one repair (2 calls)", len(m.seen) == 2 and len(r["attempts"]) == 2)
check("rejected, empty plan", r["rejected"] and not r["ok"] and r["plan"] == [])
check("first plan still recorded", r["first_plan"][0]["action"] == "teleport")
m = Mock(P(("search_for(Tomato)", None, None, None)), P(("search_for", "Tomato", None, None)))
r = planner(m).plan("گوجه‌فرنگی را پیدا کن.", MEM, None)
e0 = r["attempts"][0]["validation"]["errors"][0]["reason"]
check("'tool(Type)' action: rejected (not guessed), the reason shows the right form, then repaired",
      "unknown action 'search_for(Tomato)'" in e0 and '"action": "search_for", "object": "Tomato"' in e0
      and r["repaired"] and r["plan"][0] == {"action": "search_for", "object": "Tomato", "to": None, "arg": None}, e0)
check("max_repairs=0 -> no repair call", len(planner(Mock(P(("fly", "Fridge", None, None))),
                                                 max_repairs=0).plan("x", MEM)["attempts"]) == 1)

# ============================================================================================ 5
print("\n5. Persian command, alias + case normalisation")
m = Mock(P(("put", "refrigerator", None, None), ("Close", "fridge", None, None), say="سیب را در یخچال گذاشتم."))
r = planner(m).plan("سیب را داخل یخچال بگذار و درش را ببند.", MEM, "Apple")
check("Persian command planned (told holding Apple)", r["ok"], str(r["attempts"][0]["validation"]))
check("put->put_in, refrigerator->Fridge, Close->close, fridge->Fridge",
      r["plan"] == [{"action": "put_in", "object": "Fridge", "to": None, "arg": None},
                    {"action": "close", "object": "Fridge", "to": None, "arg": None}], str(r["plan"]))
check("normalisation is recorded", len(r["attempts"][0]["validation"]["notes"]) >= 3)
check("Persian 'say' kept", r["say"] == "سیب را در یخچال گذاشتم.")
m = Mock(P(("pickup", "سیب", None, None)), P(("pickup", "Apple", None, None)))
r = planner(m).plan("سیب را بردار.", MEM, None)
check("a Persian type name is rejected, then repaired", r["repaired"] and r["plan"][0]["object"] == "Apple"
      and "unknown object type 'سیب'" in r["attempts"][0]["validation"]["errors"][0]["reason"])

# ============================================================================================ 6
print("\n6. semantic checks")
V = lambda plan, holding=None: g.validate(P(*plan), TYPES, holding)   # noqa: E731
v = V([("pickup", "Banana", None, None)])
check("unknown type rejected", not v["ok"] and "not in the detector's type list" in v["errors"][0]["reason"])
check("missing object rejected", "needs 'object'" in V([("open", None, None, None)])["errors"][0]["reason"])
check("pick_and_place needs 'to'", "needs 'to'" in V([("pick_and_place", "Apple", None, None)])["errors"][0]["reason"])
check("put_in with an empty hand rejected", "hand is empty" in V([("put_in", "Fridge", None, None)])["errors"][0]["reason"])
check("pickup then put_in is fine", V([("pickup", "Apple", None, None), ("put_in", "Fridge", None, None)])["ok"])
check("pickup while holding something else rejected",
      "already holding Bread" in V([("pickup", "Apple", None, None)], "Bread")["errors"][0]["reason"])
check("heat the held item is fine; put_in after heat is fine",
      V([("heat", "Potato", None, None), ("put_in", "Bowl", None, None)], "Potato")["ok"])
check("drop after pick_and_place rejected (hand empty again)",
      not V([("pick_and_place", "Apple", "Fridge", None), ("drop", None, None, None)])["ok"])
v = V([("fill", "Mug", None, None)])
check("fill: null arg -> water", v["ok"] and v["steps"][0]["arg"] == "water")
check("fill: bad liquid rejected", not V([("fill", "Mug", None, "milk")])["ok"])
check("drop ignores a stray object (noted)", V([("drop", "Apple", None, None)], "Apple")["notes"])
check("empty plan rejected", "empty" in g.validate({"say": "", "plan": []}, TYPES)["errors"][0]["reason"])
# console verify 2026-10-06: put_in {object: held, to: receptacle} -- normalised under the plan's hand trace, else rejected
v = V([("pickup", "Apple", None, None), ("go_to", "Fridge", None, None), ("put_in", "Apple", "Fridge", None)])
check("put_in Apple->Fridge after pickup Apple: normalised to put_in Fridge (live console case)",
      v["ok"] and v["steps"][2] == {"action": "put_in", "object": "Fridge", "to": None, "arg": None}
      and v["normalised"] and v["normalised"][0]["from"]["object"] == "Apple"
      and any("predicts you hold Apple" in n for n in v["notes"]))
v = V([("put_in", "Apple", "Fridge", None)], "Apple")
check("put_in Apple->Fridge with the agent told it holds Apple: normalised", v["ok"] and v["steps"][0]["object"] == "Fridge")
v = V([("put_in", "apple", "refrigerator", None)], "Apple")
check("put_in normalisation canonicalises aliases / case", v["ok"] and v["steps"][0]["object"] == "Fridge", str(v))
v = V([("put_in", "Bread", "Fridge", None)], "Apple")
check("put_in X->Y while the plan predicts holding Z: rejected, says what it predicts",
      not v["ok"] and "predicts you hold Apple" in v["errors"][0]["reason"] and v["steps"][0]["to"] == "Fridge")
v = V([("put_in", "Apple", "Fridge", None)])
check("put_in X->Y with an empty hand: rejected (never guessed)", not v["ok"]
      and "predicts an empty hand" in v["errors"][0]["reason"])
v = V([("put_in", "Apple", "Spaceship", None)], "Apple")
check("put_in with an unknown 'to' type: rejected, never silently dropped", not v["ok"]
      and "'to'" in v["errors"][0]["reason"])
v = V([("put_in", "Fridge", None, None)], "Apple")
check("canonical put_in Fridge unchanged, nothing normalised", v["ok"] and not v["normalised"])
v = V([("pickup", "Apple", None, None), ("put_in", "Apple", "Fridge", None),
       ("pickup", "Bread", None, None), ("put_in", "Bread", "Cabinet", None)])
check("two consecutive placements both normalised", v["ok"] and len(v["normalised"]) == 2
      and [s["object"] for s in v["steps"]] == ["Apple", "Fridge", "Bread", "Cabinet"])
check("other tools keep note-and-ignore for a stray 'to'",
      V([("open", "Fridge", "Apple", None)])["ok"] and V([("open", "Fridge", "Apple", None)])["notes"])
m = Mock(P(("pickup", "Apple", None, None), ("put_in", "Apple", "Fridge", None)))
r = planner(m).plan("put the apple in the fridge", MEM)
check("planner result reports the normalisation separately from first_ok",
      r["ok"] and r["first_ok"] and r["normalised"] and r["plan"][1]["object"] == "Fridge")
check("prompt says put_in's object is the receptacle", "object = the RECEPTACLE" in g.tool_lines())
check("over-long plan rejected", not V([("toggle_on", "Faucet", None, None)] * (g.MAX_STEPS + 1))["ok"])
check("close_all with or without a filter", V([("close_all", None, None, None)])["ok"]
      and V([("close_all", "Cabinet", None, None)])["ok"])
check("whole plan rejected on one bad step (no step is dropped)",
      not V([("open", "Fridge", None, None), ("zap", "Fridge", None, None)])["ok"])

# ============================================================================================ 7
print("\n7. broken replies and outages")
m = Mock("I would open the fridge.", P(("open", "Fridge", None, None)))
r = planner(m).plan("Open the fridge.", MEM)
check("non-JSON reply -> rejected with reason, then repaired",
      r["repaired"] and "not a JSON plan" in r["attempts"][0]["validation"]["errors"][0]["reason"])
m = Mock(RuntimeError("connection refused"), RuntimeError("connection refused"))
pl = planner(m)
r = pl.plan("Open the fridge.", MEM)
check("provider outage -> rejected, recorded, no crash", r["rejected"] and "connection refused" in pl.calls[0]["error"])
m = Mock('Sure: {"say": "ok", "plan": [{"action": "open", "object": "Fridge", "to": null, "arg": null}]} done')
check("JSON inside prose is still parsed", planner(m).plan("Open the fridge.", MEM)["ok"])

# ============================================================================================ 8
print("\n8. search chooser")
m = Mock({"places": ["Fridge", "Banana", "CounterTop", "Apple", "fridge"]})
pl = planner(m)
ch = pl.chooser()
places = ch("Apple", TYPES, MEM)
check("chooser keeps only candidate types, no target, no duplicates", places == ["Fridge", "CounterTop"], str(places))
check("chooser call used its own schema and is logged", m.seen[0][1] is g.CHOOSER_SCHEMA
      and pl.calls[-1]["kind"] == "chooser" and ch.last["places"] == places)
check("chooser label names the model", "llm" in ch.label and g.MODEL in ch.label)
m = Mock(RuntimeError("timeout"))
ch = planner(m).chooser()
check("chooser outage -> no ranking (falls to viewpoints), never oracle",
      ch("Apple", TYPES, MEM) == [] and "timeout" in ch.last["error"])

# ============================================================================================ 9
print("\n9. reproducibility settings")


class Prov:
    def __init__(self):
        self.kw = None

    def chat(self, model, messages, **kw):
        self.kw = dict(kw, model=model)
        return json.dumps(P(("open", "Fridge", None, None)))


pv = Prov()
r = g.GemmaPlanner(provider=pv, types=TYPES).plan("Open the fridge.", MEM)
check("provider path: model, schema format, options passed", r["ok"] and pv.kw["model"] == g.MODEL
      and pv.kw["fmt"] is g.PLAN_SCHEMA and pv.kw["options"]["temperature"] == 0.0 and "seed" in pv.kw["options"])
check("prompt version recorded", g.PROMPT_VERSION.startswith("s6-"))
check("thinking is off by default and sent to the provider", g.THINK is False and pv.kw["think"] is False)

print("\n10. output budget exhausted (s6 live: Gemma 4 thinking ate num_predict)")


class LenProv:
    """Ollama-like: returns '' with done_reason 'length' for the first n calls, then a good answer."""

    def __init__(self, n, good):
        self.n, self.good, self.last_meta = n, good, {}

    def chat(self, model, messages, **kw):
        if self.n > 0:
            self.n -= 1
            self.last_meta = {"done_reason": "length", "eval_count": 768}
            return ""
        self.last_meta = {"done_reason": "stop", "eval_count": 20}
        return self.good


pl = g.GemmaPlanner(provider=LenProv(5, "{}"), types=TYPES)
ch = pl.chooser()
check("chooser: length-truncated empty reply -> explicit error, no ranking",
      ch("Apple", TYPES, MEM) == [] and "done_reason=length" in ch.last["error"]
      and pl.calls[-1]["done_reason"] == "length", str(ch.last))
pl = g.GemmaPlanner(provider=LenProv(1, json.dumps(P(("open", "Fridge", None, None)))), types=TYPES)
r = pl.plan("Open the fridge.", MEM)
check("planner: truncated reply rejected with the reason, then repaired",
      r["repaired"] and "done_reason=length" in r["attempts"][0]["validation"]["errors"][0]["reason"]
      and [c["done_reason"] for c in pl.calls] == ["length", "stop"])

class TruncOK(LenProv):
    def chat(self, model, messages, **kw):
        self.last_meta = {"done_reason": "length"}
        return json.dumps(P(("open", "Fridge", None, None)))


pl = g.GemmaPlanner(provider=TruncOK(0, ""), types=TYPES, max_repairs=0)
check("length-truncated reply is an error even if it parses", pl.plan("Open the fridge.", MEM)["rejected"])
m = Mock("   ")
ch = planner(m).chooser()
check("chooser: blank reply -> explicit 'empty reply' error", ch("Apple", TYPES, MEM) == []
      and "empty reply" in ch.last["error"])

# 2026-10-08: "put the plate in the garbage can" (Persian) -> Gemma wrote pickup {Plate, to: GarbageCan}; the plan must
# not silently become "pick up the plate"
v = g.validate({"plan": [{"action": "pickup", "object": "Plate", "to": "GarbageCan"}]}, ["Plate", "GarbageCan"])
check("pickup with a known destination -> pick_and_place (normalisation recorded), not a silent pickup only",
      v["ok"] and v["steps"] == [{"action": "pick_and_place", "object": "Plate", "to": "GarbageCan", "arg": None}]
      and v["normalised"], str(v))
v = g.validate({"plan": [{"action": "pickup", "object": "Plate", "to": "Moon"}]}, ["Plate", "GarbageCan"])
check("pickup with an unknown destination -> rejected (repair), never dropped", not v["ok"], str(v))

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("STEP4 GEMMA SELFTEST OK")
