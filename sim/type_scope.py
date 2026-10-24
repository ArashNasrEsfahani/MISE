"""Round-6 amendment D2 (2026-10-13): the ONE global type filter for agent-facing resources.

Arash's decision D2: the Knife is archived everywhere and treated as if the type does not exist. The filter is the same in every
arm and every kitchen and is chosen without looking at held-out kitchens (amendment sim/final_test/signal/r6/amendment_r6.json).
The scientific record keeps the facts (the Knife is present in FloorPlan1 but renders with 0 pixels; its earlier failures stay in
the history); only what the agent / planner is SHOWN is filtered.

No simulator, no LLM, no randomness.
"""
from __future__ import annotations

import re

EXCLUDED_TYPES = ("Knife",)
REASON = "excluded_type:D2"
# a whole type name: "Knife" but not "ButterKnife" (no letter on either side)
_RX = re.compile(r"(?<![A-Za-z])(%s)(?![A-Za-z])" % "|".join(map(re.escape, EXCLUDED_TYPES)))


def in_scope(t) -> bool:
    return t not in EXCLUDED_TYPES


def filter_types(types) -> list:
    """The given type names without the excluded ones, order kept."""
    return [t for t in types if in_scope(t)]


def mentions_excluded(text) -> list:
    """Excluded type names that appear as whole words in the text (sorted, unique)."""
    return sorted(set(_RX.findall(str(text or ""))))
