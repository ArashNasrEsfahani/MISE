"""Seed the agent's knowledge base — the one-time bootstrap of long-term memory.

Before the self-improving loop can *retrieve* useful grounding, the knowledge base needs a starting corpus. This
script builds it, **idempotently** and **auditably**, on top of `rag.HybridRAG` / `knowledge.KnowledgeBase`:

  1. **Function docs** — one ``fact`` per skill in :mod:`skills` (``skill_docs()``), tagged ``source='prior'`` /
     ``scope='global'``. This is the planner's tool catalogue, made retrievable by RAG so the brain can be reminded
     of which skills exist and what they do.
  2. **Kitchen priors** — a handful of common-sense placement facts ("Mugs are usually in Cabinets or on
     CounterTops"), also ``source='prior'`` — seed grounding for object search before anything has been observed.
  3. **Migration** — procedures/lessons already in ``knowledge_base.json`` (hand-taught through the GUI) predate the
     provenance schema; they are upgraded **in place** to carry ``source``/``ts``/``confidence``/``scope`` and tagged
     ``source='human'`` (a human demonstrated them), then embedded alongside everything else.

Idempotent: re-running adds nothing already present — facts are matched by their text, legacy entries by the absence
of a ``source`` field. Every write goes through the auditable KnowledgeBase API (skill/kitchen facts) or an explicit
audit line (migration), so the seeded corpus is fully traceable.

Pure stdlib at import: ``rag`` / ``skills`` / ``knowledge`` are all stdlib-first (numpy and Ollama are only ever
reached lazily inside ``rag``), so this module imports and runs on the Windows authoring host too. Run once to
bootstrap the store::

    python seed_memory.py
"""
from __future__ import annotations

import os
import time

import rag
import skills

# Common-sense kitchen placement priors as (text, tags). All seeded source='prior', scope='global'. The tags echo
# the salient object/receptacle types so the keyword half of retrieval fires even without the semantic (Ollama) half.
KITCHEN_FACTS = [
    ("Mugs and Cups are usually found in Cabinets or on CounterTops.",
     ["mug", "cup", "cabinet", "countertop"]),
    ("Apples and Tomatoes are usually in the Fridge or on a CounterTop.",
     ["apple", "tomato", "fridge", "countertop"]),
    ("Plates and Bowls are usually stored in Cabinets.",
     ["plate", "bowl", "cabinet"]),
    ("Knives and cutlery are usually in Drawers or on the CounterTop; pick up a Knife before slicing.",
     ["knife", "cutlery", "drawer", "slice"]),
    ("Bread, Lettuce, and Potatoes are sliceable foods, usually on the CounterTop or in the Fridge.",
     ["bread", "lettuce", "potato", "fridge", "countertop"]),
    ("Eggs and most perishable food are kept in the Fridge.",
     ["egg", "fridge", "food"]),
    ("To heat or cook food use the Microwave or a StoveBurner; to chill it use the Fridge.",
     ["microwave", "stoveburner", "fridge", "cook", "heat", "cool"]),
    ("Wash dirty items in the Sink: place them in the SinkBasin and run the Faucet.",
     ["sink", "sinkbasin", "faucet", "wash", "clean"]),
]


def _ts_from_time_str(s) -> float:
    """Best-effort epoch from a stored ``'%Y-%m-%d %H:%M:%S'`` time string; falls back to ``time.time()``."""
    try:
        return time.mktime(time.strptime(str(s), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return time.time()


def seed(kb_path: str | None = None, reindex: bool = True) -> dict:
    """Build/extend the knowledge base at ``kb_path`` (default ``knowledge_base.json`` next to this module).

    Returns a summary dict ``{skill_docs_added, kitchen_facts_added, migrated, embedded, embeddings_ok, stats}``.
    Idempotent (safe to re-run) and auditable. ``reindex`` embeds new/legacy entries via Ollama when reachable,
    degrading silently to keyword-only when it is not.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = kb_path or os.path.join(here, "knowledge_base.json")
    hr = rag.HybridRAG(path=path)
    kb = hr.kb

    existing_facts = {e.get("text") for e in kb.data.get("facts", [])}

    # 1) function docs -> facts (source='prior', scope='global'); one line per L1/L2 skill, skipped if already present.
    docs_added = 0
    for spec, line in zip(skills.SKILLS, skills.skill_docs()):
        if line in existing_facts:
            continue
        hr.add("fact", line, source="prior", scope="global",
               tags=["skill", spec.level, spec.name], why="seed: skill documentation")
        existing_facts.add(line)
        docs_added += 1

    # 2) kitchen common-sense priors (source='prior', scope='global'), skipped if already present.
    priors_added = 0
    for text, tags in KITCHEN_FACTS:
        if text in existing_facts:
            continue
        hr.add("fact", text, source="prior", scope="global", tags=tags,
               why="seed: kitchen common-sense prior")
        existing_facts.add(text)
        priors_added += 1

    # 3) migrate legacy procedures/lessons IN PLACE: stamp provenance and tag source='human' (a human taught them via
    #    the GUI). Idempotent — only entries lacking a `source` field are touched. Each migration is audited.
    migrated = 0
    for kind in ("procedures", "lessons"):
        for e in kb.data.get(kind, []):
            if "source" in e:
                continue
            e["source"] = "human"
            e.setdefault("ts", _ts_from_time_str(e.get("time", "")))
            e.setdefault("confidence", 1.0)
            e.setdefault("scope", e.get("scene") or "global")
            kb._audit("MIGRATE " + kind[:-1], e.get("command") or str(e.get("id")),
                      "seed_memory: stamp legacy entry with provenance", source="human")
            migrated += 1
    if migrated:
        kb._save()

    # 4) embed everything missing a cached vector (best-effort; keyword-only if Ollama is down).
    embedded = 0
    if reindex:
        try:
            embedded = hr.reindex()
        except Exception:
            embedded = 0

    return {"skill_docs_added": docs_added, "kitchen_facts_added": priors_added,
            "migrated": migrated, "embedded": embedded, "embeddings_ok": hr._embed_ok,
            "stats": kb.stats()}


def main():
    s = seed()
    print("seed_memory: knowledge base bootstrapped")
    print(f"  skill-doc facts added  : {s['skill_docs_added']}")
    print(f"  kitchen prior facts add: {s['kitchen_facts_added']}")
    print(f"  legacy entries migrated: {s['migrated']}")
    print(f"  entries embedded       : {s['embedded']} "
          f"({'Ollama up' if s['embeddings_ok'] else 'keyword-only (no Ollama)'})")
    print(f"  store now              : {s['stats']}")


if __name__ == "__main__":
    main()
