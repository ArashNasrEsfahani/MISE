"""Canonical record schema for the three-level command hierarchy — shared by planner, dataset and eval.

The thesis is organized around three levels of command, each with its own metric (formulas in docs/THESIS.md §6):

| Level | Meaning                                                        | Metric                              |
|-------|---------------------------------------------------------------|-------------------------------------|
| L1    | Atomic — one command maps to one function/action              | Success Rate (SR)                   |
| L2    | Sequential — an ordered chain of atomic functions             | Procedural Correctness (PC)         |
| L3    | Abstract goal — needs reasoning + decomposition into a plan   | Task Decomposition Score (TDS, F1)  |

`CommandRecord` is the one record type every part of the system agrees on: a dataset row, the gold target an
evaluator scores against, and (with its `gold` left empty) the shape of a planner's task input. It is **bilingual**:
`lang` is 'en' or 'fa' (Persian).

The `gold` field is the level-appropriate ground truth:
  * **final-state goal** — used for SR at *all* levels (final environment state the command should produce);
  * **action sequence** — additionally for L2 (ordered list, scored by PC / normalized Levenshtein);
  * **action set**      — for L3 (unordered required-action set, scored by TDS / F1).
Its concrete value is JSON-able (a dict / list); this module fixes the schema, not the gold encoding.

Pure stdlib — safe to import anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

# The three hierarchy levels, in order. `validate()` checks membership here.
LEVELS = ("L1", "L2", "L3")

# One-line description per level (kept in sync with CLAUDE.md / docs/THESIS.md §6).
LEVEL_DESCRIPTIONS = {
    "L1": "Atomic: one command maps to exactly one function / AI2-THOR action.",
    "L2": "Sequential: an ordered chain of atomic functions executed in turn.",
    "L3": "Abstract goal: needs reasoning and decomposition into a fresh plan.",
}

# The metric each level is scored with (see docs/THESIS.md §6 for formulas).
LEVEL_METRICS = {
    "L1": "SR",   # Success Rate — final state matches goal
    "L2": "PC",   # Procedural Correctness — normalized Levenshtein over the action sequence
    "L3": "TDS",  # Task Decomposition Score — F1 over required vs generated action set
}

# Accepted language codes for the bilingual dataset.
LANGS = ("en", "fa")


@dataclass
class CommandRecord:
    """One bilingual command + its level-appropriate gold target.

    Fields:
      id      — stable unique identifier (string).
      level   — one of LEVELS ('L1' | 'L2' | 'L3').
      lang    — one of LANGS ('en' | 'fa').
      command — the natural-language instruction.
      gold    — the ground truth (see module docstring): final-state goal (SR, all levels) /
                action sequence (L2 PC) / action set (L3 TDS). JSON-able value.
    """
    id: str
    level: str
    lang: str
    command: str
    gold: object = None
    scene: str | None = None       # AI2-THOR scene the gold is grounded in (e.g. 'FloorPlan2'); None = scene-agnostic
    setup: list | None = None      # plan steps run BEFORE the agent plans, to put the world in the state the
                                   # command presupposes ("close the fridge" needs an open fridge); general set only

    def to_dict(self) -> dict:
        """Plain JSON-able dict (round-trips through `from_dict`)."""
        d = {"id": self.id, "level": self.level, "lang": self.lang,
             "command": self.command, "gold": self.gold}
        if self.scene:
            d["scene"] = self.scene
        if self.setup:
            d["setup"] = self.setup
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CommandRecord":
        """Build a record from a dict, ignoring any extra keys (tolerant dataset loader)."""
        return cls(id=d.get("id"), level=d.get("level"), lang=d.get("lang"),
                   command=d.get("command"), gold=d.get("gold"), scene=d.get("scene"),
                   setup=(d.get("setup") or None))

    def validate(self) -> bool:
        """Raise ValueError on the first schema problem; return True if the record is well-formed.

        Checks: level in LEVELS, lang in LANGS, gold present (not None). `id`/`command` must be non-empty too —
        a record with no instruction or identifier can't be scored or referenced.
        """
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")
        if self.lang not in LANGS:
            raise ValueError(f"lang must be one of {LANGS}, got {self.lang!r}")
        if self.gold is None:
            raise ValueError(f"gold target is required (record {self.id!r})")
        if not self.id:
            raise ValueError("id is required")
        if not self.command:
            raise ValueError(f"command text is required (record {self.id!r})")
        return True


def metric_for(level: str) -> str:
    """Abbreviated metric name for a level ('SR' | 'PC' | 'TDS'); raises on an unknown level."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return LEVEL_METRICS[level]


if __name__ == "__main__":
    import sys
    try:                                    # the dataset is bilingual; make sure a non-UTF8 console can't crash us
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rows = [
        CommandRecord("l1-en-001", "L1", "en", "open the fridge",
                      gold={"goal": [{"type": "Fridge", "key": "isOpen", "value": True}]}),
        CommandRecord("l2-fa-001", "L2", "fa", "سیب را در یخچال بگذار",
                      gold={"goal": [{"type": "Apple", "key": "parentReceptacles", "contains": "Fridge"}],
                            "sequence": ["pickup Apple", "open Fridge", "put Fridge", "close Fridge"]}),
        CommandRecord("l3-en-001", "L3", "en", "make me a cold snack",
                      gold={"goal": [{"type": "Apple", "key": "temperature", "value": "Cold"}],
                            "actions": {"pickup Apple", "cool_object Apple"}}),
    ]
    for r in rows:
        assert r.validate()
        assert CommandRecord.from_dict(r.to_dict()) == r, "round-trip mismatch"
        print(f"{r.id}: {r.level}/{metric_for(r.level)} [{r.lang}] -> {r.command}")
    bad = CommandRecord("x", "L4", "en", "hi", gold={})
    try:
        bad.validate()
    except ValueError as e:
        print("rejected bad record:", e)
    print("levels self-test OK")
