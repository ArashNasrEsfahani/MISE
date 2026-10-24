"""Natural-language → AI2-THOR action planner (+ the planner prompt and plan parsing).

This is the front-end of the thesis's LLM "brain": a chat command is turned into a JSON plan of low-level skills
(the L1/L2 atomic/sequential actions of `thor_actions.ThorAgent`), which the GUI then executes step by step.

The LLM transport now lives in the pluggable `llm` module: `Ollama` is re-exported below as a backward-compatible
alias for `llm.OllamaProvider`, alongside `make_provider`/`LLMProvider` for choosing a backend. The GUI runs inside
WSL, so when Ollama lives on Windows the endpoint must be the Windows host (see `gateway_endpoint`), and Ollama must
be started with `OLLAMA_HOST=0.0.0.0` so it accepts connections from the WSL NAT network.
"""
from __future__ import annotations

import json
import re

# Backward-compatible re-exports: keep `agent_llm.Ollama(...)` working while decoupling the brain from one vendor.
from llm import LLMProvider, OllamaProvider as Ollama, make_provider  # noqa: F401

# ----------------------------------------------------------------------------------------------------------------------
# Planner prompt: the model must answer with ONE strict-JSON object describing a plan of named actions.
# ----------------------------------------------------------------------------------------------------------------------
PLANNER_SYSTEM = """You are the planning brain of an embodied robot in an AI2-THOR kitchen.
Turn the user's command into a JSON plan of low-level actions the robot can execute.

Reply with ONLY a JSON object of EXACTLY this shape (no prose, no markdown):
{"say": "<one short sentence to the user>",
 "plan": [ {"action": "<name>", "object": "<ObjectType or null>", "to": "<ReceptacleType or null>", "arg": <number, liquid-name, or null>} ]}

HIGH-LEVEL actions (PREFER these — each handles its own walking, opening, retries):
- pick_and_place: object = item to move, to = destination receptacle type (e.g. move the Apple onto the CounterTop)
- put_in: object = receptacle type (puts the item you're currently holding into/onto it)
- heat:  object = food type (microwaves it and hands it back to you, hot)
- cool:  object = food type (chills it in the fridge and hands it back)
- cook:  object = food type (cooks it — cookable food comes out cooked and hot)
- wash:  object = item type (rinses it clean in the sink)

PERCEPTION actions (use to find things or survey the room):
- look_around (object=null): spin 360° to see everything in the room
- scan_for (object=ObjectType): rotate until that object is in view and center it (use when it isn't currently visible)
- search_for (object=ObjectType): GO AND LOOK for that object — walks to likely places, opens closed
  containers, and reports whether it was actually seen. Use this for "find / where is / look for" commands, and
  whenever the object is not already in the observation. Prefer it over look_around and scan_for, which only
  spin on the spot and cannot find anything in another part of the room or inside a cupboard.

BATCH actions (apply to MANY objects at once):
- open_all / close_all (object = ObjectType to filter, or null for everything openable)
- turn_on_all / turn_off_all (object = ObjectType to filter, or null for every toggleable APPLIANCE;
  the room lights are NOT included unless you name LightSwitch)
- collect_all: object = item type to gather, to = receptacle type to put them all in

LOW-LEVEL actions (use when no high-level action fits):
- Navigation (object=null): move_ahead, move_back, move_left, move_right,
  rotate_left, rotate_right, look_up, look_down, crouch, stand
- Approach an object (object=ObjectType): go_to (walks over and centers it), face_object
- Manipulate (object=ObjectType): pickup, put (object = the receptacle type),
  open, close, toggle_on, toggle_off, slice, cook, break, dirty, clean,
  fill (arg = "water"|"coffee"|"wine"), empty, push, pull
- Held object (object=null): drop, throw
- Other: capture (take a photo, object=null), done

Rules:
- Use ONLY object TYPES that appear in the OBSERVATION list. Never invent objects.
- The OBSERVATION shows held=<type> — what the robot is already holding. If you are ALREADY holding the object the
  user wants placed, do NOT go_to or pickup it again; just go to the destination and put it (object=receptacle), or
  use put_in. Only pick up an object you are not already holding; if your hand is full with something else, drop first.
- PREFER one high-level action over many low-level ones when it fits (e.g. "put the apple in the fridge"
  -> a single pick_and_place with object=Apple, to=Fridge; "heat the potato" -> a single heat with object=Potato).
- For low-level manipulation, emit a go_to for that object FIRST (high-level actions do this for you).
- For "pick up/take/grab X", the plan MUST include pickup X after reaching it; go_to X alone is incomplete.
- To pick something out of a closed receptacle: go_to it, open it, pickup the item, close it.
- Keep the plan minimal and ordered. If the command can't be done with the present objects, return "plan": [] and say why.
"""

# A few forgiving aliases for object types the model may phrase differently.
TYPE_ALIASES = {
    "refrigerator": "Fridge", "fridge": "Fridge", "tap": "Faucet", "sink": "SinkBasin",
    "trash": "GarbageCan", "trashcan": "GarbageCan", "garbage": "GarbageCan", "bin": "GarbageCan",
    "counter": "CounterTop", "countertop": "CounterTop", "stove": "StoveBurner", "burner": "StoveBurner",
}


# `Ollama` (the original stdlib HTTP client) now lives in `llm.py` as `OllamaProvider` and is re-exported above
# under its old name, so `agent_llm.Ollama(endpoint).chat(...)` / `.list_models()` keep working unchanged.


def gateway_endpoint() -> str | None:
    """Best-effort Ollama endpoint for a Windows host as seen from WSL (the default-route gateway IP)."""
    try:
        import subprocess
        out = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=4).stdout
        for line in out.splitlines():
            if line.startswith("default") and " via " in line:
                ip = line.split(" via ")[1].split()[0]
                return f"http://{ip}:11434"
    except Exception:
        pass
    return None


def parse_plan(content: str) -> dict:
    """Parse the model's reply into {'say':..., 'plan':[...]}; tolerant of stray text around the JSON."""
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError(f"no JSON object in model reply: {content[:200]}")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model reply was not a JSON object")
    obj.setdefault("say", "")
    obj.setdefault("plan", [])
    if not isinstance(obj["plan"], list):
        obj["plan"] = []
    return obj


def canonical_type(name: str | None) -> str | None:
    if not name:
        return None
    return TYPE_ALIASES.get(str(name).strip().lower(), str(name).strip())
