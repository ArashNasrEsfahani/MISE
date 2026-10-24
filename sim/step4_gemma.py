"""Step 4 (s6-gemma) -- Gemma 4 on Ollama as the PLANNER (and search chooser) of the honest agent. Pure module.

The planner sees only what the honest agent knows (Arash decisions 1-5; docs/step4-plan.md s6):
  * the honest tool list -- the verbs ``honest_skills.HonestSkills.exec_step`` runs (``honest_skills.TOOLS``), each
    with its argument rule;
  * the FloorPlan1 detector's class names (``honest_skills.detector_types`` -- trained vocabulary, NOT a room
    inventory: the prompt says so, and nothing says which of them are present);
  * the object-memory summary (``objmem.ObjectMemory.summary`` -- the agent's own sightings this episode);
  * what the agent was told it holds (action history).
No simulator truth, no reference plan, no goal, no oracle setup facts enter a prompt or a repair message.

The model answers ``{"say": ..., "plan": [{"action", "object", "to", "arg"}]}`` -- exec_step's step format. The plan
is parsed (``agent_llm.parse_plan``) and VALIDATED as a whole (Astra: never drop steps -- a dropped prerequisite
changes the task):
  * an action outside the tool list, an object/receptacle type outside the detector list, a missing required
    field, a bad ``arg``, a step needing an empty / a full hand that the plan's own hand-trace contradicts, an
    empty or over-long plan -> rejected with a reason per step;
  * documented aliases (``put`` -> ``put_in``, ``refrigerator`` -> ``Fridge``, case) are normalised first and
    the normalisation is recorded;
  * at most ONE repair re-prompt (the reasons are sent back; they name only the tool list / type list -- no
    hidden scene fact); still invalid -> the plan is NOT executed.
The same planner object backs ``honest_skills.LLMChooser`` (``ask_text``): it ranks place types from the fixed
candidate list; the labelled no-LLM ``prior_chooser`` stays the other option.

Reproducibility (Astra): temperature 0, fixed seed, fixed context / output limits, stream off, a JSON-Schema
``format`` that fixes the reply SHAPE only (no enums -- semantic validation is ours, so a wrong verb or type is
visible and the repair path is real). The trial records model digest, Ollama version and the options.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional

import honest_skills as hs
from agent_llm import TYPE_ALIASES, parse_plan

PROMPT_VERSION = "s6-v4"            # v2: think=False, length/blank = error; v3: repair hint for "tool(Type)" actions;
#                                    v4: put_in(object=held, to=receptacle) normalised / rejected, never silently dropped
MODEL = "gemma4:e2b-it-q4_K_M"     # the spec said gemma4:e2b-it; that tag does not exist in the Ollama registry (404, 2026-10-06)
MAX_STEPS = 12
MAX_REPAIRS = 1
#: Gemma 4 "thinks" unless told not to; its thinking can eat num_predict and leave an empty reply (s6 live, chooser).
#: Off for planner, repair and chooser (Astra): a reproducible non-thinking baseline. Sent top-level, not in OPTIONS.
THINK = False
OPTIONS = {"temperature": 0.0, "seed": 7, "num_ctx": 8192, "num_predict": 768, "top_k": 1}

#: tool -> (object rule, to rule, arg rule, one-line meaning). rule: "type" (a detector type) | None (must be empty)
#: | "opt" (ignored). arg rule: None | "liquid".
TOOL_SPECS = {
    "pickup": ("type", None, None, "walk to the object and pick it up (hand must be empty)"),
    "put_in": ("type", None, None, "put what you are holding into / onto this receptacle (opens it if closed); "
                                 "object = the RECEPTACLE, not the held item"),
    "pick_and_place": ("type", "type", None, "pick up `object` and put it into / onto receptacle `to` "
                                             "(walks, searches and opens containers itself)"),
    "open": ("type", None, None, "open it (fridge, cabinet, drawer, microwave, ...)"),
    "close": ("type", None, None, "close it"),
    "toggle_on": ("type", None, None, "switch it on (faucet, stove knob, coffee machine, toaster, light ...)"),
    "toggle_off": ("type", None, None, "switch it off"),
    "slice": ("type", None, None, "slice it"),
    "break": ("type", None, None, "break / crack it"),
    "dirty": ("type", None, None, "make it dirty"),
    "clean": ("type", None, None, "clean it where it is"),
    "fill": ("type", None, "liquid", "fill the container with a liquid; arg = water | coffee | wine"),
    "empty": ("type", None, None, "pour the liquid out of the container"),
    "heat": ("type", None, None, "heat the food in the microwave; it ends back in your hand"),
    "cook": ("type", None, None, "cook the food (in the microwave); it ends back in your hand"),
    "wash": ("type", None, None, "rinse the item in the sink (it is left in the sink)"),
    "go_to": ("type", None, None, "walk to the object and look at it"),
    "search_for": ("type", None, None, "look for the object in likely places (opens closed containers)"),
    "drop": (None, None, None, "drop what you are holding"),
    "close_all": ("opt", None, None, "close every container you have seen open"),
    "crouch": (None, None, None, "crouch (see low places)"),
    "stand": (None, None, None, "stand up again"),
}
#: plan-level effect on the hand, for the hand-trace check: "take" (ends holding object), "release",
#: "take-release" (holds object in between, ends empty), "need" (needs a held object, ends empty)
HAND = {"pickup": "take", "heat": "take", "cook": "take", "pick_and_place": "take-release",
        "wash": "take-release", "put_in": "need", "drop": "need"}

SYSTEM = """You are the planner of a kitchen robot. The robot sees only through its camera and an object detector;
it knows only what it has seen itself. Turn the user's command (English or Persian) into an ordered plan of the
robot's tools.

Reply with ONLY a JSON object of exactly this shape:
{{"say": "<one short sentence to the user, in the user's language>",
 "plan": [{{"action": "<tool>", "object": "<ObjectType or null>", "to": "<ObjectType or null>", "arg": "<liquid or null>"}}]}}

TOOLS (the only allowed actions):
{tools}

OBJECT TYPES the detector can recognise (its vocabulary -- not every type is necessarily in this kitchen; write
types EXACTLY as spelled here, in English, even when the command is in Persian):
{types}

Rules:
- Use only the tools and object types above. Never invent a tool or a type.
- High-level tools (pick_and_place, put_in, heat, cook, wash, pickup, go_to) walk, search and open containers by
  themselves: you do not need go_to / open / search_for before them.
- The hand holds one thing. Do not pick something up while holding something else; put_in and drop need a held
  object.
- Keep the plan minimal and in order (at most {max_steps} steps). Do the whole command, nothing more.
"""

USER = """What the robot has seen so far (type [state] place, position):
{memory}
The robot is holding: {holding}

Command: {command}"""

REPAIR = """Your plan was rejected, so nothing was executed. Problems:
{problems}
Reply again with ONLY the corrected JSON object, using only the listed tools and object types."""

PLAN_SCHEMA = {"type": "object", "required": ["say", "plan"],
               "properties": {"say": {"type": "string"},
                              "plan": {"type": "array", "items": {
                                  "type": "object", "required": ["action", "object", "to", "arg"],
                                  "properties": {"action": {"type": "string"},
                                                 "object": {"type": ["string", "null"]},
                                                 "to": {"type": ["string", "null"]},
                                                 "arg": {"type": ["string", "number", "null"]}}}}}}
CHOOSER_SCHEMA = {"type": "object", "required": ["places"],
                  "properties": {"places": {"type": "array", "items": {"type": "string"}}}}


# ============================================================================================== prompt
def tool_lines(tools=None) -> str:
    out = []
    for t in sorted(tools or hs.TOOLS):
        o, to, arg, what = TOOL_SPECS[t]
        sig = ", ".join(x for x in (("object" if o == "type" else "object?" if o == "opt" else ""),
                                    ("to" if to == "type" else ""), ("arg" if arg else "")) if x)
        out.append(f"- {t}({sig}): {what}")
    return "\n".join(out)


def build_messages(command: str, types, memory_summary, holding: Optional[str], tools=None) -> list:
    sys_txt = SYSTEM.format(tools=tool_lines(tools), types=", ".join(sorted(types)), max_steps=MAX_STEPS)
    usr = USER.format(memory="\n".join(memory_summary or ["(nothing yet)"]), holding=holding or "nothing",
                      command=command)
    return [{"role": "system", "content": sys_txt}, {"role": "user", "content": usr}]


# ============================================================================================== validation
def _canon(name, types) -> tuple:
    """(canonical type or None, note). Exact, then case-insensitive, then a documented alias."""
    if name is None or (isinstance(name, str) and not name.strip()) or str(name).strip().lower() in ("null", "none"):
        return None, ""
    s = str(name).strip()
    if s in types:
        return s, ""
    low = {t.lower(): t for t in types}
    if s.lower() in low:
        return low[s.lower()], f"{s!r}->{low[s.lower()]}"
    a = TYPE_ALIASES.get(s.lower()) or TYPE_ALIASES.get(s.lower().replace(" ", ""))
    if a and a in types:
        return a, f"{s!r}->{a}"
    return s, "unknown"


def validate(obj: dict, types, holding: Optional[str] = None, tools=None) -> dict:
    """Whole-plan validation. Returns {ok, steps (normalised), errors [{i, reason}], notes}."""
    tools = set(tools or hs.TOOLS)
    types = set(types)
    errors, notes, steps, normalised = [], [], [], []
    plan = obj.get("plan") if isinstance(obj, dict) else None
    if not isinstance(plan, list):
        return {"ok": False, "steps": [], "errors": [{"i": None, "reason": "no 'plan' list in the reply"}],
                "notes": [], "normalised": []}
    if not plan:
        errors.append({"i": None, "reason": "the plan is empty"})
    if len(plan) > MAX_STEPS:
        errors.append({"i": None, "reason": f"the plan has {len(plan)} steps; at most {MAX_STEPS}"})
    hand = holding
    for i, st in enumerate(plan):
        if not isinstance(st, dict):
            errors.append({"i": i, "reason": "a step must be a JSON object"})
            continue
        raw_a = str(st.get("action") or "").strip()
        a = raw_a.lower().replace(" ", "_").replace("-", "_")
        a = hs.ALIASES.get(a, a)
        if a != raw_a:
            notes.append(f"step {i}: action {raw_a!r}->{a}")
        if a not in tools:
            reason = f"unknown action {raw_a!r} (not in the tool list)"
            m = re.match(r"^\s*([A-Za-z_]+)\s*\((.*)\)\s*$", raw_a)
            if m and hs.ALIASES.get(m.group(1).lower(), m.group(1).lower()) in tools:
                # s6 live: Gemma wrote the tool list's signature notation as the action ("search_for(Tomato)").
                # Not auto-accepted (that would guess the arguments); the reason says how to write it.
                reason += (f"; write the tool name alone in \"action\" and the type in \"object\", e.g. "
                           f"{{\"action\": \"{m.group(1)}\", \"object\": \"{m.group(2).strip()}\"}}")
            errors.append({"i": i, "reason": reason})
            steps.append({"action": raw_a, "object": st.get("object"), "to": st.get("to"), "arg": st.get("arg")})
            continue
        if a == "pickup" and st.get("to") not in (None, ""):
            # 2026-10-08 (console, "put the plate in the garbage can" in Persian): Gemma wrote pickup {object: Plate,
            # to: GarbageCan}. The generic "takes no 'to'" branch below dropped GarbageCan -- the plan silently became
            # "pick up the plate" (a task change). A pickup WITH a known destination is a pick_and_place; rewrite it
            # (recorded as a normalisation, like put_in above); an unknown destination is rejected (-> repair).
            o_val, _ = _canon(st.get("object"), types)
            t_val, t_note = _canon(st.get("to"), types)
            if t_val is None or t_note == "unknown":
                errors.append({"i": i, "reason": f"pickup takes no 'to' ({st.get('to')!r} is not a known type); to "
                                                 "move it somewhere use pick_and_place with 'object' and 'to'"})
                steps.append({"action": a, "object": st.get("object"), "to": st.get("to"), "arg": st.get("arg")})
                continue
            normalised.append({"i": i, "from": {"action": a, "object": o_val, "to": t_val},
                               "to": {"action": "pick_and_place", "object": o_val, "to": t_val}})
            notes.append(f"step {i}: pickup {o_val} to {t_val} -> pick_and_place {o_val} {t_val} (a pickup with a "
                         "destination)")
            a = "pick_and_place"
        o_rule, to_rule, arg_rule, _ = TOOL_SPECS[a]
        norm = {"action": a, "object": None, "to": None, "arg": None}
        if a == "put_in" and st.get("to") not in (None, ""):
            # console verify 2026-10-06 (Astra-reviewed): Gemma wrote put_in {object: Apple, to: Fridge}. The generic
            # "takes no 'to'" branch dropped Fridge and kept Apple as the RECEPTACLE -- a silent task change. Rewrite
            # only when the plan's own hand trace PREDICTS the object is what is held and `to` is a known type;
            # anything else rejects (-> the single repair). Never a simulator read: `hand` is the plan's prediction.
            o_val, _ = _canon(st.get("object"), types)
            t_val, t_note = _canon(st.get("to"), types)
            if t_val is None or t_note == "unknown":
                errors.append({"i": i, "reason": f"unknown object type {st.get('to')!r} in 'to' (not in the "
                                                 "detector's type list); put_in takes the RECEPTACLE as its object"})
                steps.append({"action": a, "object": st.get("object"), "to": st.get("to"), "arg": st.get("arg")})
                continue
            if hand and o_val == hand:
                normalised.append({"i": i, "from": {"action": a, "object": o_val, "to": t_val},
                                   "to": {"action": a, "object": t_val, "to": None}})
                notes.append(f"step {i}: put_in {o_val} to {t_val} -> put_in {t_val} "
                             f"(the plan predicts you hold {o_val}; put_in takes the receptacle)")
                st = dict(st, object=t_val, to=None)
            else:
                held = f"the plan predicts you hold {hand} at this step" if hand else                     "the plan predicts an empty hand at this step"
                errors.append({"i": i, "reason": f"put_in takes the RECEPTACLE as 'object' and no 'to'; {held}: "
                                                 f"write {{\"action\": \"put_in\", \"object\": \"<receptacle>\"}}"})
                steps.append({"action": a, "object": st.get("object"), "to": st.get("to"), "arg": st.get("arg")})
                continue
        for field, rule in (("object", o_rule), ("to", to_rule)):
            val, note = _canon(st.get(field), types)
            if rule == "type":
                if val is None:
                    errors.append({"i": i, "reason": f"{a} needs '{field}' (an object type)"})
                elif note == "unknown":
                    errors.append({"i": i, "reason": f"unknown object type {val!r} in '{field}' (not in the "
                                                     "detector's type list)"})
                elif note:
                    notes.append(f"step {i}: {field} {note}")
                norm[field] = val
            elif rule == "opt":
                if val is not None and note != "unknown":
                    norm[field] = val
            elif val is not None:
                notes.append(f"step {i}: {a} takes no '{field}'; {val!r} ignored")
        if arg_rule == "liquid":
            liq = str(st.get("arg") or "water").strip().lower()
            if liq not in hs.LIQUIDS:
                errors.append({"i": i, "reason": f"fill arg must be one of {', '.join(hs.LIQUIDS)}, not {liq!r}"})
            norm["arg"] = liq
        # hand trace (the plan's own consequences, from what the agent was told it holds)
        eff = HAND.get(a)
        if eff in ("take", "take-release") and hand and hand != norm["object"]:
            errors.append({"i": i, "reason": f"{a} {norm['object']} while already holding {hand} "
                                             "(put it somewhere or drop it first)"})
        if eff == "need" and not hand:
            errors.append({"i": i, "reason": f"{a} needs a held object, but the hand is empty at this step"})
        if eff == "take":
            hand = norm["object"]
        elif eff in ("take-release", "need"):
            hand = None
        steps.append(norm)
    return {"ok": not errors, "steps": steps, "errors": errors, "notes": notes, "normalised": normalised}


def problems_text(errors) -> str:
    return "\n".join(f"- {'step ' + str(e['i']) + ': ' if e.get('i') is not None else ''}{e['reason']}"
                     for e in errors)


# ============================================================================================== planner
class GemmaPlanner:
    """Plan + validate + (at most one) repair. ``chat(messages, fmt) -> text`` is the transport; by default it is
    an ``llm.LLMProvider`` (``agent_llm.make_provider``) -- tests pass a mock."""

    def __init__(self, provider=None, model: str = MODEL, types=(), tools=None, options: Optional[dict] = None,
                 chat: Optional[Callable] = None, timeout: float = 240.0, max_repairs: int = MAX_REPAIRS,
                 think: Optional[bool] = THINK):
        self.provider, self.model = provider, model
        self.types = sorted(types)
        self.tools = sorted(tools or hs.TOOLS)
        self.options = dict(OPTIONS if options is None else options)
        self.timeout, self.max_repairs = timeout, int(max_repairs)
        self.think = think
        self._chat = chat or self._provider_chat
        self.calls = []                    # every LLM call: {kind, secs, chars, error}

    def _provider_chat(self, messages, fmt):
        return self.provider.chat(self.model, messages, timeout=self.timeout, fmt=fmt, options=self.options,
                                  think=self.think)

    def _call(self, kind, messages, fmt):
        t0 = time.time()
        err = ""
        meta = {}
        try:
            if self.provider is not None:
                self.provider.last_meta = {}
            text = self._chat(messages, fmt) or ""
            meta = dict(getattr(self.provider, "last_meta", None) or {})
            # s6 live: Gemma 4 thinks by default; the thinking can use the whole output budget, leaving an empty
            # reply (done_reason "length"). That is an explicit error, never a silent "no answer".
            if meta.get("done_reason") == "length":
                err = f"output budget exhausted (done_reason=length, num_predict={self.options.get('num_predict')})"
            elif not text.strip():
                err = "empty reply"
        except Exception as e:                       # an outage is recorded, never replaced by the oracle
            text, err = "", f"{type(e).__name__}: {e}"[:300]
        secs = round(time.time() - t0, 2)
        self.calls.append({"kind": kind, "secs": secs, "chars": len(text), "error": err, "think": self.think, **meta})
        return text, secs, err

    def plan(self, command: str, memory_summary=(), holding: Optional[str] = None) -> dict:
        messages = build_messages(command, self.types, memory_summary, holding, self.tools)
        attempts = []
        for k in range(1 + self.max_repairs):
            raw, secs, err = self._call("plan" if k == 0 else "repair", messages, PLAN_SCHEMA)
            try:
                obj = parse_plan(raw) if raw and not err else None
                perr = "" if obj is not None else (err or "empty reply")
            except ValueError as e:
                obj, perr = None, str(e)[:300]
            v = (validate(obj, self.types, holding, self.tools) if obj is not None else
                 {"ok": False, "steps": [], "errors": [{"i": None, "reason": f"reply is not a JSON plan ({perr})"}],
                  "notes": [], "normalised": []})
            attempts.append({"k": k, "messages": [dict(m) for m in messages], "raw": raw, "secs": secs,
                             "error": err, "parsed": obj, "validation": v})
            if v["ok"]:
                break
            messages = messages + [{"role": "assistant", "content": raw or "(no reply)"},
                                   {"role": "user", "content": REPAIR.format(problems=problems_text(v["errors"]))}]
        last = attempts[-1]
        ok = last["validation"]["ok"]
        return {"ok": ok, "plan": last["validation"]["steps"] if ok else [],
                "say": (last["parsed"] or {}).get("say", "") if isinstance(last["parsed"], dict) else "",
                "first_ok": attempts[0]["validation"]["ok"], "repaired": ok and len(attempts) > 1,
                # Astra: "valid on the first pass" must not hide an argument rewrite -- report it separately, so
                # raw schema correctness and accepted-after-normalisation stay distinguishable in the thesis
                "normalised": last["validation"].get("normalised", []) if ok else [],
                "notes": last["validation"].get("notes", []),
                "rejected": not ok, "attempts": attempts,
                "first_plan": attempts[0]["validation"]["steps"],
                "secs": round(sum(a["secs"] for a in attempts), 2)}

    # -------------------------------------------------------------- search chooser transport
    def ask_text(self, prompt: str) -> str:
        """For ``honest_skills.LLMChooser``: one user turn, a {"places": [...]} schema; the chooser's
        ``parse_places`` keeps only names from the fixed candidate list."""
        text, _, err = self._call("chooser", [{"role": "user", "content": prompt}], CHOOSER_SCHEMA)
        if err:
            raise RuntimeError(err)
        return text

    def chooser(self, k: int = 6) -> hs.LLMChooser:
        c = hs.LLMChooser(self.ask_text, k=k)
        c.label = f"llm ({self.model}, constrained to the detector type list)"
        return c
