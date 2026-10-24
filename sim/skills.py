"""Machine-readable skill manifest for the L1/L2 hierarchy — the agent's tool catalogue.

`thor_actions.ThorAgent` is the big working control surface; this module does NOT re-implement it. Instead it
declares, as plain data, *which* of those methods are the agent's planning skills and what level each sits at:

| Level | Meaning                                            | Metric                          |
|-------|----------------------------------------------------|---------------------------------|
| L1    | one command -> one atomic AI2-THOR action          | Success Rate                    |
| L2    | a composite/sequential generator over L1 actions   | Procedural Correctness          |

(L3 — abstract goals needing fresh decomposition — is not a fixed skill; it is produced by the planner, so it has
no SkillSpec here. See `levels.py` for the record schema spanning all three levels.)

The manifest is the single source the planner is told about, the eval harness scores against, and the knowledge
base is *seeded* from (`skill_docs()` -> one human-readable line per skill, ready for `KnowledgeBase.add_fact`).

Pure stdlib, no `ai2thor` import — safe to load anywhere (Windows authoring host included).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkillSpec:
    """One declarative skill entry. `level` in {'L1','L2'}.

    params/preconditions/effects are short tuples distilled from the matching `ThorAgent` method's signature and
    docstring — enough for a planner to choose the skill and for the eval to know what 'success' means.
    """
    name: str                     # the ThorAgent method name (machine identifier)
    level: str                    # 'L1' (atomic) | 'L2' (composite/sequential generator)
    params: tuple                 # call parameters (besides self)
    preconditions: tuple          # what must hold before it can succeed
    effects: tuple                # the state change it brings about
    doc: str                      # one-line human/LLM description


# ----------------------------------------------------------------------------------------------------------------------
# L1 — atomic skills: each wraps exactly ONE AI2-THOR action (the action name is given in the doc, in parentheses).
# ----------------------------------------------------------------------------------------------------------------------
_NAVIGATION = [
    SkillSpec("move_ahead", "L1", ("magnitude",), ("path ahead clear",),
              ("agent translated forward by moveMagnitude (m)",),
              "Step the agent forward one grid step (MoveAhead)."),
    SkillSpec("move_back", "L1", ("magnitude",), ("path behind clear",),
              ("agent translated backward by moveMagnitude (m)",),
              "Step the agent backward one grid step (MoveBack)."),
    SkillSpec("move_left", "L1", ("magnitude",), ("path to the left clear",),
              ("agent strafed left by moveMagnitude (m)",),
              "Strafe the agent left without turning (MoveLeft)."),
    SkillSpec("move_right", "L1", ("magnitude",), ("path to the right clear",),
              ("agent strafed right by moveMagnitude (m)",),
              "Strafe the agent right without turning (MoveRight)."),
    SkillSpec("rotate_left", "L1", ("degrees",), (),
              ("agent yaw decreased by `degrees` (turns left in place)",),
              "Rotate the agent left in place (RotateLeft)."),
    SkillSpec("rotate_right", "L1", ("degrees",), (),
              ("agent yaw increased by `degrees` (turns right in place)",),
              "Rotate the agent right in place (RotateRight)."),
    SkillSpec("look_up", "L1", ("degrees",), ("resulting camera pitch within agent limits (>= -30)",),
              ("camera horizon tilted up by `degrees`",),
              "Tilt the camera up (LookUp)."),
    SkillSpec("look_down", "L1", ("degrees",), ("resulting camera pitch within agent limits (<= 60)",),
              ("camera horizon tilted down by `degrees`",),
              "Tilt the camera down (LookDown)."),
    SkillSpec("crouch", "L1", (), ("agent currently standing",),
              ("agent lowers; standing=False (reaches low objects)",),
              "Lower the agent to a crouch (Crouch)."),
    SkillSpec("stand", "L1", (), ("agent currently crouching",),
              ("agent rises; standing=True (reaches high objects)",),
              "Raise the agent to standing (Stand)."),
]

_MANIPULATION = [
    SkillSpec("pickup", "L1", ("object_id", "force"),
              ("object is pickupable", "object visible / within reach", "a free hand"),
              ("object isPickedUp=True; held in the agent's hand",),
              "Pick up an object into the agent's hand (PickupObject)."),
    SkillSpec("put", "L1", ("receptacle_object_id", "force"),
              ("agent is holding an object", "receptacle visible / within reach"),
              ("held object placed into/onto receptacle; parentReceptacles updated; hand freed",),
              "Put the currently-held object into/onto a receptacle (PutObject)."),
    SkillSpec("drop", "L1", ("force",), ("agent is holding an object",),
              ("held object released under physics; hand freed",),
              "Drop the held object straight down (DropHandObject)."),
    SkillSpec("open", "L1", ("object_id", "openness", "force"),
              ("object is openable", "object within reach", "swing path not blocked"),
              ("isOpen=True, opened to `openness` (0..1)",),
              "Open an openable object/receptacle (OpenObject)."),
    SkillSpec("close", "L1", ("object_id", "force"),
              ("object is openable and currently open", "object within reach"),
              ("isOpen=False",),
              "Close an open object/receptacle (CloseObject)."),
    SkillSpec("toggle_on", "L1", ("object_id", "force"),
              ("object is toggleable and currently off", "object within reach"),
              ("isToggled=True (device powered/active)",),
              "Switch a toggleable device on (ToggleObjectOn)."),
    SkillSpec("toggle_off", "L1", ("object_id", "force"),
              ("object is toggleable and currently on", "object within reach"),
              ("isToggled=False",),
              "Switch a toggleable device off (ToggleObjectOff)."),
    SkillSpec("slice", "L1", ("object_id", "force"),
              ("object is sliceable", "object within reach", "agent holding a Knife"),
              ("isSliced=True; object replaced by its slices",),
              "Slice a sliceable object (SliceObject) — needs a held knife."),
    SkillSpec("cook", "L1", ("object_id", "force"),
              ("object is cookable", "object within reach"),
              ("isCooked=True; temperature -> Hot",),
              "Cook a cookable object directly (CookObject)."),
    SkillSpec("break_object", "L1", ("object_id", "force"),
              ("object is breakable", "object within reach"),
              ("isBroken=True",),
              "Break a breakable object (BreakObject)."),
    SkillSpec("dirty", "L1", ("object_id", "force"),
              ("object is dirtyable and currently clean", "object within reach"),
              ("isDirty=True",),
              "Make an object dirty (DirtyObject)."),
    SkillSpec("clean", "L1", ("object_id", "force"),
              ("object is dirtyable and currently dirty", "object within reach"),
              ("isDirty=False",),
              "Clean a dirty object (CleanObject)."),
    SkillSpec("fill_with_liquid", "L1", ("object_id", "liquid", "force"),
              ("object canFillWithLiquid", "object within reach"),
              ("isFilledWithLiquid=True; fillLiquid=`liquid` (water|coffee|wine)",),
              "Fill a container with a liquid (FillObjectWithLiquid)."),
    SkillSpec("empty_liquid", "L1", ("object_id", "force"),
              ("object currently filled with liquid", "object within reach"),
              ("isFilledWithLiquid=False; fillLiquid cleared",),
              "Empty the liquid from a container (EmptyLiquidFromObject)."),
    SkillSpec("push", "L1", ("object_id", "magnitude", "force"),
              ("object is moveable or pickupable", "object within reach"),
              ("object shoved away from the agent (pose changes)",),
              "Push a moveable object away (PushObject)."),
    SkillSpec("pull", "L1", ("object_id", "magnitude", "force"),
              ("object is moveable or pickupable", "object within reach"),
              ("object pulled toward the agent (pose changes)",),
              "Pull a moveable object toward the agent (PullObject)."),
]

# ----------------------------------------------------------------------------------------------------------------------
# L2 — composite/sequential skills: generators that chain L1 actions (navigate, open, retry, ...) into a task-level
# action. Each navigates to its target(s) itself and uses forceAction on the final interaction.
# ----------------------------------------------------------------------------------------------------------------------
_COMPOSITE = [
    SkillSpec("pick_and_place", "L2", ("obj_name", "receptacle", "force"),
              ("`obj_name` exists in the scene", "`receptacle` exists in the scene"),
              ("`obj_name` ends up in/on `receptacle` (walk -> pick up -> carry -> put)",),
              "Pick up an object and place it into/onto a receptacle — the canonical rearrangement skill."),
    SkillSpec("put_in", "L2", ("receptacle", "force"),
              ("agent is holding an object", "`receptacle` exists in the scene"),
              ("held object placed into/onto `receptacle` (navigates there first)",),
              "Carry the currently-held object to a receptacle and put it down."),
    SkillSpec("heat_object", "L2", ("food", "appliance", "force"),
              ("`food` exists", "`appliance` (default Microwave) exists"),
              ("`food` heated (temperature -> Hot) and returned to the agent's hand",),
              "Heat food in the microwave: open -> put in -> close -> run -> open -> take back out."),
    SkillSpec("cool_object", "L2", ("food", "appliance", "force"),
              ("`food` exists", "`appliance` (default Fridge) exists"),
              ("`food` chilled (temperature -> Cold) and returned to the agent's hand",),
              "Chill food in the fridge: open -> put in -> close -> wait -> open -> take back out."),
    SkillSpec("wash_object", "L2", ("item", "force"),
              ("`item` exists", "a Sink/SinkBasin exists"),
              ("`item` rinsed in the sink and cleaned (isDirty=False)",),
              "Wash an item in the sink: place in basin, run the faucet, scrub clean, faucet off."),
    SkillSpec("cook_food", "L2", ("food", "force"),
              ("`food` exists", "a Microwave exists"),
              ("`food` cooked/heated (isCooked and/or temperature -> Hot)",),
              "Cook food — cookable items come out cooked and hot (delegates to heat_object)."),
    SkillSpec("collect_all", "L2", ("object_type", "receptacle", "force", "limit"),
              ("instances of `object_type` exist", "`receptacle` exists"),
              ("every pickupable `object_type` carried into `receptacle`, one trip each",),
              "Gather every object of a type and place each into a receptacle."),
    SkillSpec("open_all", "L2", ("object_type", "force", "limit"),
              ("openable objects exist (optionally filtered by `object_type`)",),
              ("every matching openable object isOpen=True",),
              "Open every openable object (optionally only a given type)."),
    SkillSpec("close_all", "L2", ("object_type", "force", "limit"),
              ("open objects exist (optionally filtered by `object_type`)",),
              ("every matching open object isOpen=False",),
              "Close every open object (optionally only a given type)."),
    SkillSpec("switch_all", "L2", ("on", "object_type", "force", "limit"),
              ("toggleable objects exist (optionally filtered by `object_type`)",),
              ("every matching toggleable object isToggled=`on`",),
              "Turn every toggleable object on (on=True) or off."),
]

# ----------------------------------------------------------------------------------------------------------------------
# L2 — active perception: sweep/aim generators that change the agent's view to gather information (no world change).
# ----------------------------------------------------------------------------------------------------------------------
_PERCEPTION = [
    SkillSpec("look_around", "L2", ("turn",), (),
              ("agent spins a full 360°; self.last_seen = sorted object types seen (no world state change)",),
              "Spin in place 360°, accumulating every object type seen — active-perception survey."),
    SkillSpec("scan_for", "L2", ("name", "turn"),
              ("an object whose type matches `name` exists in the scene",),
              ("agent rotates to bring that object to the centre of view (no world state change)",),
              "Rotate in place to find and centre an object of a given type."),
    SkillSpec("search_for", "L2", ("name", "detector", "memory", "kb", "speed", "max_stops"),
              (),
              ("agent walks a ranked route of likely locations, opening closed containers, until it SEES the "
               "object or exhausts the plan; sets self.last_search (no world change beyond doors it opened)",),
              "Search the scene systematically for an object: recall where it was seen, else consult the "
              "knowledge base and a containment prior, else cover the discovered map — walking, opening and "
              "looking at each stop. Unlike look_around/scan_for this LEAVES where it is standing, and it "
              "reports honestly whether the object was actually seen."),
]

# The full ordered manifest. L1 first (atomic vocabulary), then L2 (the composites a planner reaches for).
SKILLS: list = _NAVIGATION + _MANIPULATION + _COMPOSITE + _PERCEPTION


def skills_manifest() -> list:
    """SKILLS as plain JSON-able dicts (tuples flattened to lists) — for the planner prompt / eval / disk."""
    return [
        {"name": s.name, "level": s.level, "params": list(s.params),
         "preconditions": list(s.preconditions), "effects": list(s.effects), "doc": s.doc}
        for s in SKILLS
    ]


def skill_docs() -> list:
    """One human-readable line per skill for seeding the knowledge base.

    Format: 'NAME (Lx): doc | params=... | effects=...'. Pass each straight to `KnowledgeBase.add_fact`.
    """
    out = []
    for s in SKILLS:
        params = ", ".join(s.params) or "none"
        effects = "; ".join(s.effects) or "none"
        out.append(f"{s.name} ({s.level}): {s.doc} | params={params} | effects={effects}")
    return out


def skills_by_level(level: str) -> list:
    """All SkillSpecs at a given level ('L1' or 'L2')."""
    return [s for s in SKILLS if s.level == level]


def skill_by_name(name: str):
    """The SkillSpec with this method name, or None."""
    return next((s for s in SKILLS if s.name == name), None)


if __name__ == "__main__":
    n1 = len(skills_by_level("L1"))
    n2 = len(skills_by_level("L2"))
    print(f"skills manifest: {len(SKILLS)} total | L1={n1} atomic | L2={n2} composite/perception")
    assert {s.level for s in SKILLS} <= {"L1", "L2"}, "SkillSpec.level must be L1 or L2"
    assert len({s.name for s in SKILLS}) == len(SKILLS), "skill names must be unique"
    for line in skill_docs()[:3]:
        print("  ", line)
    print("skills self-test OK")
