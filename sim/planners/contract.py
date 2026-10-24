"""The planner contract: ONE renderer for the prompt every provider sees (final test, stage P1).

The prompt builder is a COPY of ``step4_gemma`` (SYSTEM, USER, TOOL_SPECS, PLAN_SCHEMA, ``build_messages``) so that
step-4 files stay untouched; ``_planners_selftest.py`` asserts byte equality with ``step4_gemma.build_messages`` for
the same inputs (and equality of every copied constant), so any drift fails loudly.

Without a memory or experience block the messages equal step 4's exactly. The optional blocks (preregistration s4:
"one block appended to the user message under a fixed header") are appended after the command by default.

Providers only TRANSPORT these messages: Ollama / OpenRouter send them as chat roles, Anthropic sends the system
text as the top-level ``system``, Codex gets ``flat_prompt`` (fixed role headers, one stdin text). The role mapping
and the serialized request hash are recorded per call, so what is equal is the APPLICATION CONTENT, not the wire.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

PROMPT_VERSION = "ft-v1"                    # prereg s4: step4_gemma's text frozen as ft-v1 (hashed in freeze.json)
MAX_STEPS = 12

MEMORY_HEADER = "Long-term memory (retrieved for this command; learned from earlier practice):"
EXPERIENCE_HEADER = "Earlier practice records (raw, from the same practice as the memory):"

#: tool -> (object rule, to rule, arg rule, one-line meaning). COPY of step4_gemma.TOOL_SPECS.
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


# ================================================================================================= ft-v2 (stage H-B1)
# Version-switched, additive: PlannerContract(version="ft-v1") (the default) is byte-identical to step 4's prompt.
# ft-v2 is COMMON to every arm (Gemma, Opus, Haiku, with or without a package): the same system text, the same
# validator (validator_v2), at most 2 repairs. Only the memory/package block and the FS block differ between arms.
VERSIONS = ("ft-v1", "ft-v2")
MAX_REPAIRS_V2 = 2

PACKAGE_HEADER = ("Assistance package (what the user told me about this kitchen, and what I learned in practice; "
                  "what you see now wins):")

SYSTEM_V2 = SYSTEM.replace(
    "TOOLS (the only allowed actions):\n{tools}",
    "TOOLS (the only allowed actions). Each line gives the exact step shape, what the tool does and what it does to the\n"
    "hand; {{...}} is one step, \"<...>\" is a type from the list below, null stays null:\n{tools}\n\n"
    "Write the tool name alone in \"action\" (for example \"pickup\", never \"pickup(Apple)\") and the type in \"object\".\n"
    "Schematic of a two-step plan (NOT JSON and not a command; each <...> stands for a type from the list below):\n{example}"
) + "- If the plan is rejected you get the problems and may correct it; fix exactly those problems.\n"

#: the one example plan: placeholders, no kitchen type (the selftest validates it after substituting real types)
EXAMPLE_PLAN_V2 = {"say": "<one short sentence>", "plan": [
    {"action": "pick_and_place", "object": "<ItemType>", "to": "<ReceptacleType>", "arg": None},
    {"action": "close", "object": "<ReceptacleType>", "to": None, "arg": None}]}

REPAIR_V2 = """Your plan was rejected, so nothing was executed. Problems:
{problems}
Reply again with ONLY the corrected JSON object, using only the listed tools and object types."""


def example_schematic() -> str:
    """The example as a labelled schematic outside executable JSON (Astra H-B1: a literal placeholder plan can be copied)."""
    lines = []
    for n, st in enumerate(EXAMPLE_PLAN_V2["plan"], 1):
        parts = [f"{k}={st[k]}" for k in ("object", "to") if st.get(k)]
        lines.append(f"{n}. {st['action']}: " + ", ".join(parts))
    return "\n".join(lines)


def _exec_tools():
    from . import validator_v2
    return validator_v2.exec_contract()["tools"]


_HAND_TEXT = {"take": "needs an empty hand; you hold the object afterwards",
              "take-release": "needs an empty hand; the hand is empty again afterwards",
              "release": "needs a held item; the hand is empty afterwards",
              "keep": "the hand is unchanged", "none": "the hand is unchanged"}
_RECEPTACLE_OBJECT = frozenset({"put_in"})


def tool_lines_v2(tools=None) -> str:
    """One line per tool: exact step shape (JSON), meaning, hand effect -- generated from ``exec_contract.json``."""
    con = _exec_tools()
    out = []
    for t in sorted(tools or _default_tools()):
        c = con[t]
        if c["object"] == "type":
            obj = "<ReceptacleType>" if t in _RECEPTACLE_OBJECT else "<ItemType>"
        else:
            obj = "<ItemType or null>" if c["object"] == "opt" else None
        to = "<ReceptacleType>" if c["to"] == "type" else None
        arg = "|".join(c["arg"]) if isinstance(c["arg"], list) else None
        shape = json.dumps({"action": t, "object": obj, "to": to, "arg": arg}, ensure_ascii=False)
        extra = " Walks and opens containers itself." if c["opens_itself"] else ""
        out.append(f"- {shape}: {c['meaning']}. Hand: {_HAND_TEXT[c['hand']]}.{extra}")
    return "\n".join(out)


def runner_kwargs(version: str = "ft-v1") -> dict:
    """-> {"max_repairs", "validate"} for ``PlannerRunner``: ft-v1 = 1 repair + step4's validator; ft-v2 = 2 + validator_v2."""
    if version not in VERSIONS:
        raise ValueError(f"unknown contract version {version!r}")
    if version == "ft-v1":
        return {"max_repairs": 1, "validate": _v1_validate}
    from . import validator_v2
    return {"max_repairs": MAX_REPAIRS_V2, "validate": validator_v2.validate}


def _v1_validate(obj, types, holding, tools):
    import step4_gemma
    return step4_gemma.validate(obj, types, holding, tools)


def _default_tools():
    import honest_skills as hs                       # sim/ must be on sys.path, like every sim module
    return hs.TOOLS


def tool_lines(tools=None) -> str:
    out = []
    for t in sorted(tools or _default_tools()):
        o, to, arg, what = TOOL_SPECS[t]
        sig = ", ".join(x for x in (("object" if o == "type" else "object?" if o == "opt" else ""),
                                    ("to" if to == "type" else ""), ("arg" if arg else "")) if x)
        out.append(f"- {t}({sig}): {what}")
    return "\n".join(out)


def problems_text(errors) -> str:
    return "\n".join(f"- {'step ' + str(e['i']) + ': ' if e.get('i') is not None else ''}{e['reason']}"
                     for e in errors)


class PlannerContract:
    """Renders the identical messages for every provider. ``types`` = the detector vocabulary."""

    def __init__(self, types, tools=None, max_steps: int = MAX_STEPS, block_position: str = "after_command",
                 version: str = "ft-v1"):
        if block_position not in ("after_command", "before_command"):
            raise ValueError("block_position must be after_command or before_command")
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}")
        self.types, self.tools, self.max_steps, self.block_position = sorted(types), tools, max_steps, block_position
        self.version = version
        self.schema = PLAN_SCHEMA

    def runner_kwargs(self) -> dict:
        return runner_kwargs(self.version)

    def example_plan_text(self) -> str:
        return example_schematic()

    def system_text(self) -> str:
        if self.version == "ft-v2":
            return SYSTEM_V2.format(tools=tool_lines_v2(self.tools), types=", ".join(sorted(self.types)),
                                    max_steps=self.max_steps, example=self.example_plan_text())
        return SYSTEM.format(tools=tool_lines(self.tools), types=", ".join(sorted(self.types)),
                             max_steps=self.max_steps)

    def user_text(self, command: str, memory_summary=(), holding: Optional[str] = None,
                  memory_block: Optional[str] = None, fewshot_block: Optional[str] = None) -> str:
        usr = USER.format(memory="\n".join(memory_summary or ["(nothing yet)"]), holding=holding or "nothing",
                          command=command)
        blocks = []
        if memory_block:
            hdr = PACKAGE_HEADER if self.version == "ft-v2" else MEMORY_HEADER
            blocks.append(f"{hdr}\n{memory_block}")
        if fewshot_block:
            blocks.append(f"{EXPERIENCE_HEADER}\n{fewshot_block}")
        if not blocks:
            return usr
        extra = "\n\n".join(blocks)
        if self.block_position == "after_command":
            return usr + "\n\n" + extra
        tail = "\n\nCommand: " + command
        return usr[: len(usr) - len(tail)] + "\n\n" + extra + tail

    def messages(self, command: str, memory_summary=(), holding: Optional[str] = None,
                 memory_block: Optional[str] = None, fewshot_block: Optional[str] = None) -> list:
        return [{"role": "system", "content": self.system_text()},
                {"role": "user", "content": self.user_text(command, memory_summary, holding, memory_block,
                                                           fewshot_block)}]

    @staticmethod
    def repair_messages(messages, raw: str, problems: str) -> list:
        return list(messages) + [{"role": "assistant", "content": raw or "(no reply)"},
                                 {"role": "user", "content": REPAIR.format(problems=problems)}]

    def repair_turn(self, messages, raw: str, problems: str) -> list:
        """Version-aware repair turn (ft-v2 uses REPAIR_V2; the wording of the frame is the same, the PROBLEMS differ)."""
        tmpl = REPAIR_V2 if self.version == "ft-v2" else REPAIR
        return list(messages) + [{"role": "assistant", "content": raw or "(no reply)"},
                                 {"role": "user", "content": tmpl.format(problems=problems)}]


def split_system(messages):
    """-> (system text, the remaining chat turns). Several system messages are joined with a blank line."""
    sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return sys_txt, [m for m in messages if m["role"] != "system"]


def flat_prompt(messages) -> str:
    """One text with fixed role headers, for a CLI that takes a single prompt (Codex)."""
    return "\n\n".join(f"### {m['role'].upper()}\n{m['content']}" for m in messages) + "\n"


def prompt_sha256(messages) -> str:
    return hashlib.sha256(json.dumps([[m["role"], m["content"]] for m in messages], ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ reply parsing
def parse_reply(raw: str):
    """-> (dict or None, note). Tolerates a code fence or prose around ONE JSON object; the note says what was done
    ('' = the reply was a clean JSON object)."""
    s = (raw or "").strip()
    if not s:
        return None, "empty reply"
    try:
        obj = json.loads(s)
        return (obj, "") if isinstance(obj, dict) else (None, "reply is JSON but not an object")
    except ValueError:
        pass
    m = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", s, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            return (obj, "code fence removed") if isinstance(obj, dict) else (None, "fenced JSON is not an object")
        except ValueError:
            pass
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        try:
            obj = json.loads(s[a:b + 1])
            return (obj, "JSON object extracted from surrounding text") if isinstance(obj, dict) else \
                (None, "extracted JSON is not an object")
        except ValueError as e:
            return None, f"no valid JSON object ({str(e)[:80]})"
    return None, "no JSON object in the reply"


_JSON_TYPES = {"object": dict, "array": list, "string": str, "number": (int, float), "integer": int,
               "boolean": bool, "null": type(None)}


def schema_problems(obj, schema, path="$") -> list:
    """Minimal JSON-Schema shape check (type incl. type lists, required, properties, items). Returns reasons."""
    out = []
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        ok = any(isinstance(obj, _JSON_TYPES[x]) and not (x in ("number", "integer") and isinstance(obj, bool))
                 for x in ts)
        if not ok:
            return [f"{path}: expected {'|'.join(ts)}, got {type(obj).__name__}"]
    if isinstance(obj, dict):
        for r in schema.get("required", []):
            if r not in obj:
                out.append(f"{path}: missing required field {r!r}")
        for k, sub in (schema.get("properties") or {}).items():
            if k in obj:
                out += schema_problems(obj[k], sub, f"{path}.{k}")
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        for i, it in enumerate(obj):
            out += schema_problems(it, schema["items"], f"{path}[{i}]")
    return out


def ft_v2_texts_sha(types=("T",)) -> str:
    """sha256 of EVERY frozen ft-v2 text: system, repair frame, package header, error templates, example, tool lines."""
    from . import validator_v2
    blob = {"system_v2": SYSTEM_V2, "repair_v2": REPAIR_V2, "package_header": PACKAGE_HEADER,
            "templates": validator_v2.TEMPLATES, "example": EXAMPLE_PLAN_V2, "schematic": example_schematic(),
            "tool_lines": tool_lines_v2(), "user": USER, "max_repairs": MAX_REPAIRS_V2}
    return hashlib.sha256(json.dumps(blob, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                          ).hexdigest()
