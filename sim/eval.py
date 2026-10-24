"""Phase-8 evaluation harness for the three-level command hierarchy (SR / PC / TDS).

This is the thesis's scoring bench: it drives a pluggable LLM planner (Ollama by default) over a bilingual
(English + Persian) dataset of L1/L2/L3 commands, executes each generated plan on the live AI2-THOR simulator,
and scores every record with the level-appropriate metric from docs/THESIS.md section 6:

  * L1 Atomic     -> Success Rate (SR)            : final env state matches the gold goal.
  * L2 Sequential -> Procedural Correctness (PC)  : normalized Levenshtein of planned vs gold action sequence.
  * L3 Abstract   -> Task Decomposition Score (TDS): F1 over planned vs required action sets.

SR is measured at ALL levels (from EXECUTION + oracle goal verification); PC additionally on L2 and TDS
additionally on L3 (both from the PLANNED, post-repair label sequence -- so a record can score PC/TDS even when
execution/SR fails).

Design (mirrors sim/bench_phase7.py):
  * Module import is pure stdlib. The two WSL/WSLg render env vars are set at the very top (setdefault, so an
    explicit outer env still wins) BEFORE any ai2thor import; thor_actions / webapp / memory / rag / improve are
    imported LAZILY inside run(), so `python eval.py --selftest` runs on a bare Windows box (no ai2thor, no Ollama).
  * The pure metric functions (levenshtein, action_label, labels, sr_score, pc_score, tds_score) and the plan
    repair are defined at module top with NO heavy sim import, so they are unit-testable in isolation.
  * MEMORY IS FROZEN during scoring: with `--loop off` (the default) NO knowledge-base write-back and NO passive
    feedback writes happen, so command N+1 never depends on command N. `--loop on` re-enables improve.close_loop
    (the RQ3 self-improving axis) so the KB is enriched between commands.
  * EXCEPTION-SAFE per record: any crash (model hang, parse error, sim error) scores that record 0/failure with an
    'error' field and the run CONTINUES. Every result is stamped with the full run config (model/rag/loop/
    perception/scene/lang) so the results JSON is self-describing and re-loadable by a report generator.

Ablation axes (CLI flags): --models (the LLM axis), --rag {off,keyword,hybrid,probe,agentic,full} (retrieval
grounding), --loop {on,off} (self-improving loop), --perception {oracle,perceived} (what the agent senses; SR
stays oracle).

THE --rag LADDER (Phase 10). The first three arms are unchanged, byte-for-byte, so every already-published
`eval_*.json` number stays comparable; the last three are the chunkless agentic retriever in sim/agentic_rag.py:

  off      no memory block at all (the floor).
  keyword  HybridRAG at alpha=1.0 -- pure keyword, no embedding round-trip.
  hybrid   HybridRAG at alpha=0.5 -- keyword + cosine (needs a reachable nomic-embed-text).
  probe    chunkless, index-free, ZERO extra model calls: bilingual normalisation -> deterministic lexical pivot
           -> regex probe over whole records -> provenance ranking -> typed quota -> budgeted expansion.
  agentic  probe PLUS a bounded multi-hop loop: when a deterministic sufficiency gate says the block is
           inadequate, up to --agentic-hops strict-JSON model calls decompose the command into language-neutral
           AI2-THOR object types (the Persian->English bridge) and the probe re-runs. Hops are hard-capped and
           the gate is LLM-free; any failure degrades to keyword-only and is stamped in the telemetry.
  full     dump EVERY in-scope record under a hard character budget -- no retrieval at all. The CAG-style
           upper-bound control that makes the whole retrieval axis falsifiable ("would stuffing everything work
           just as well?").

The agentic arms need NO embedding server but they DO need a POPULATED knowledge base -- an empty KB makes
probe/agentic/full byte-identical to `off`, which is exactly how the Phase-8 --rag ablation came out void. run()
probes for that and warns as loudly as it warns about unreachable embeddings for `hybrid`.

Mechanism switches (all default to the pre-Phase-10 behaviour where one exists, so old invocations are
unaffected): --rank {plain,provenance,provenance+utility}, --agentic-hops, --agentic-pivot {off,lex,llm,auto},
--agentic-gate, --agentic-cache (turn OFF for timing runs), --agentic-compact, --agentic-quota {mixed,typed},
--agentic-budget, --agentic-level {off,est,oracle} (level-adaptive retrieval; 'oracle' reads the dataset's own
L1/L2/L3 label and is an explicit UPPER BOUND, not the deployed behaviour), and --agentic-verify {on,off} (a
deterministic plan verifier + ONE repair re-plan; a SEPARATE axis from --rag so its SR contribution is never
confounded with the retrieval mode). Every record carries its retrieval telemetry in res["retrieval"] (hops,
llm_calls, pivot source, gate decision, injected tokens, latency) so no memory gain is ever reported without
its token and latency cost.

ACTION-LABEL CONVENTION (the one function both S_gen and the dataset gold agree on):
    action_label(step) = "<normalized_action> <canonical_object_or_to>"
  e.g. {"action":"pickup","object":"Apple"} -> "pickup Apple"; {"action":"heat","object":"Potato"} ->
  "heat_object Potato". action names are normalized to the skills_manifest vocabulary (heat->heat_object,
  cool->cool_object, cook->cook_food, wash->wash_object, break->break_object, turn_on_all/turn_off_all->switch_all,
  pick_up/pickup_object->pickup) and object types are canonicalized (refrigerator->Fridge, sink->SinkBasin, ...),
  because the planner PROMPT emits surface verbs while the gold uses the manifest names. PC/TDS additionally drop
  navigation/perception/no-op helper steps (go_to, scan_for, ...) from the generated plan, since the plan repair
  injects a go_to before every manipulation and gold sequences are authored nav-free. So gold must be written with
  canonical manifest labels and no navigation steps.

Run it inside WSL (ai2thor needs the GPU + WSLg X display), e.g. all five Ollama models, hybrid RAG, loop off:

    wsl -d Ubuntu-22.04 -- sh -c "cd /mnt/d/Personal/Master*/sim && \\
      MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 /home/arash/thor-env/bin/python eval.py \\
      --dataset dataset/commands.jsonl \\
      --models 'gemma3:4b,qwen2.5:1.5b,gemma4:e2b-it-q4_K_M,gemma4:e4b,yinw1590/gemma4-e2b-text:latest' \\
      --rag hybrid --loop off --perception oracle --scenes FloorPlan1 --lang all \\
      --timeout 300 --out eval_results.json"

and the Phase-10 headline arm (chunkless agentic retrieval, provenance ranking, all three kitchens):

    wsl -d Ubuntu-22.04 -- sh -c "cd /mnt/d/Personal/Master*/sim && \\
      MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 /home/arash/thor-env/bin/python eval.py \\
      --dataset dataset/commands.jsonl --models gemma3:4b \\
      --rag agentic --rank provenance --agentic-hops 2 --agentic-pivot auto --agentic-level est \\
      --loop off --perception oracle --lang all --timeout 300 --out eval_p10_agentic.json"

Pure self-test (Windows, no simulator, no Ollama):  python eval.py --selftest
"""
import os

# WSL/WSLg rendering env -- force the NVIDIA adapter under Mesa-D3D12 and the WSLg X display BEFORE ai2thor's
# Controller is ever constructed (setdefault so an explicit outer env still wins). Mirrors bench_phase7.py.
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import argparse
import json
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Pure-stdlib siblings only at import time (agent_llm -> llm are stdlib-only; levels is stdlib). thor_actions,
# webapp, memory, rag, improve, knowledge, perception are imported LAZILY inside run()/live helpers.
import agent_llm as al
import levels

DEFAULT_DATASET = os.path.join(HERE, "dataset", "commands.jsonl")
DEFAULT_OUT = os.path.join(HERE, "eval_results.json")
DEFAULT_MODELS = "gemma3:4b"
DEFAULT_ENDPOINT = "http://localhost:11434"

# --------------------------------------------------------------------------------------------------------------
# ACTION-LABEL NORMALIZATION (pure; no heavy sim import) -- see module docstring.
# --------------------------------------------------------------------------------------------------------------
# Planner surface verb -> skills_manifest / gold action name. Without this, S_gen ("heat Potato") never matches
# gold ("heat_object Potato") and PC/TDS collapse.
_ACTION_ALIASES = {
    "pick_up": "pickup", "pickup_object": "pickup",
    "heat": "heat_object", "cool": "cool_object", "cook": "cook_food",
    "wash": "wash_object", "break": "break_object",
    "turn_on_all": "switch_all", "turn_off_all": "switch_all",
    # PLANNER_SYSTEM offers `fill` / `empty`; the manifest calls them fill_with_liquid / empty_liquid.
    "fill": "fill_with_liquid", "empty": "empty_liquid",
}

# INVERSE of the composite aliases above: canonical MANIFEST action name -> the SURFACE verb that webapp._exec_one
# dispatches on. The planner PROMPT emits surface verbs (heat, cool, ...), so a plan usually carries those; but a
# step can arrive with the manifest name instead (e.g. echoed back from a retrieved procedure whose stored steps
# use manifest names, or a model that mirrored a label it saw in memory). webapp._exec_one keys its comp/objmap
# tables ONLY on the surface verbs, so a manifest-named step matches no branch and is silently skipped -- SR would
# then fail while labels()/PC/TDS still credit the step (a spurious SR-vs-label divergence). Remapping before
# dispatch keeps the executed action set aligned with the scored label set. `switch_all` is deliberately OMITTED:
# its on/off direction is ambiguous (turn_on_all vs turn_off_all), so we never guess it.
_EXEC_ACTION_ALIASES = {
    "heat_object": "heat", "cool_object": "cool", "cook_food": "cook",
    "wash_object": "wash", "break_object": "break",
    "pickup_object": "pickup", "pick_up": "pickup",
}


def _exec_action_alias(step):
    """Shallow copy of `step` with a canonical manifest action name remapped to the surface verb webapp._exec_one
    understands (heat_object->heat, ...). Non-dict / already-surface steps pass through unchanged. Used ONLY for
    execution dispatch; the stored plan/labels keep their normalized (manifest) names for scoring."""
    if not isinstance(step, dict):
        return step
    a = str(step.get("action") or "").strip().lower()
    if a in _EXEC_ACTION_ALIASES:
        s = dict(step)
        s["action"] = _EXEC_ACTION_ALIASES[a]
        return s
    return step

# Object-type canonicalization -- kept in sync with agent_llm.TYPE_ALIASES so labels canonicalize WITHOUT importing
# agent_llm into the pure metric layer.
_TYPE_ALIASES = {
    "refrigerator": "Fridge", "fridge": "Fridge", "tap": "Faucet", "sink": "SinkBasin",
    "trash": "GarbageCan", "trashcan": "GarbageCan", "garbage": "GarbageCan", "bin": "GarbageCan",
    "counter": "CounterTop", "countertop": "CounterTop", "stove": "StoveBurner", "burner": "StoveBurner",
}

# Canonical FloorPlan1 kitchen objectTypes (PascalCase). Used to canonicalize an object name case-insensitively:
# a model that emits "mug"/"MUG" must score against gold's "Mug" (a case mismatch otherwise zeroes an otherwise
# correct PC/TDS label). Compound types (CounterTop, SinkBasin) cannot be recovered by .title(), so we match the
# lowercased name against this known set instead. Unknown/hallucinated types pass through unchanged (fairly unmatched).
_KNOWN_TYPES = frozenset({
    "Apple", "Bread", "Tomato", "Potato", "Lettuce", "Egg", "Mug", "Cup", "Bowl", "Plate", "Pan", "Pot",
    "Knife", "ButterKnife", "Fork", "Spoon", "Spatula", "Ladle", "Fridge", "Microwave", "Toaster",
    "CoffeeMachine", "StoveBurner", "StoveKnob", "Cabinet", "Drawer", "CounterTop", "Sink", "SinkBasin",
    "Faucet", "LightSwitch", "GarbageCan", "Bottle", "SoapBottle", "DishSponge", "PepperShaker",
    "SaltShaker", "HousePlant", "Window", "Stool", "Chair", "WineBottle", "Kettle", "Book",
    "DiningTable", "SideTable", "Shelf", "PaperTowelRoll", "Floor",
})
_TYPE_CANON = {t.lower(): t for t in _KNOWN_TYPES}

# Navigation / perception / no-op helpers dropped from the GENERATED plan before PC/TDS scoring (the repair injects
# a go_to before every manipulation; gold sequences are authored nav-free -- see module docstring).
_NAV_HELPERS = frozenset({
    "go_to", "face_object", "move_ahead", "move_back", "move_left", "move_right",
    "rotate_left", "rotate_right", "look_up", "look_down", "crouch", "stand",
    "look_around", "scan_for", "capture", "done", "pass", "wait", "noop", "throw",
})

# `search_for` is deliberately ABSENT from _NAV_HELPERS: unlike look_around/scan_for it leaves where the agent is
# standing, opens containers, and determines whether the object is found at all, so it is a real step in the plan
# and is SCORED in PC and TDS.
#
# That is a change to the evaluation action space. Every PC/TDS number published before 2026-08-22 was computed
# without it, so `--score-search off` restores the old space and makes a historical run reproducible like for
# like. The effective set is recorded in the results JSON (`config.scored_actions`) so no future reader has to
# guess which space produced a number.
_SCORE_SEARCH = {"on": True}


def _dropped_actions() -> frozenset:
    """Actions stripped from a GENERATED plan before PC/TDS scoring, honouring `--score-search`."""
    return _NAV_HELPERS if _SCORE_SEARCH["on"] else (_NAV_HELPERS | {"search_for"})


def norm_action(action) -> str:
    """Normalize an action name to the skills_manifest / gold vocabulary (lowercased, aliases applied)."""
    a = str(action or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ACTION_ALIASES.get(a, a)


def canon_type(name):
    """Canonical object type for a loose name; case-insensitive against aliases + known types; '' passes through.

    Order: explicit alias (sink->SinkBasin) wins, then a case-insensitive match against _KNOWN_TYPES
    (mug->Mug, countertop->CounterTop), else the stripped original (unknown types stay as-authored)."""
    if not name:
        return ""
    s = str(name).strip()
    low = s.lower()
    if low in _TYPE_ALIASES:
        return _TYPE_ALIASES[low]
    return _TYPE_CANON.get(low, s)


def action_label(step) -> str:
    """The ONE canonical label for a plan step: '<normalized_action> <canonical_object_or_to>' (stripped).

    Follows the mandated convention `f"{action} {object or to or ''}"` but normalizes the action name to the
    manifest vocabulary and canonicalizes the object/to type, so a generated step lines up with gold labels that
    were authored with canonical manifest names. `object` wins over `to` (so pick_and_place Apple->Fridge labels
    as "pick_and_place Apple").
    """
    if not isinstance(step, dict):
        return str(step or "").strip()
    obj = canon_type(step.get("object") or step.get("to"))
    return f"{norm_action(step.get('action'))} {obj}".strip()


def labels(steps) -> list:
    """Semantic action-label sequence of a plan: action_label per step, dropping navigation/perception/no-op
    helpers (see `_dropped_actions`). This is S_gen for PC and (as a set) A_gen for TDS."""
    dropped = _dropped_actions()
    out = []
    for s in (steps or []):
        act = norm_action(s.get("action") if isinstance(s, dict) else s)
        if act in dropped:
            continue
        lab = action_label(s)
        if lab:
            out.append(lab)
    return out


# --------------------------------------------------------------------------------------------------------------
# METRICS (pure; NO sim import) -- exact formulas from docs/THESIS.md section 6.
# --------------------------------------------------------------------------------------------------------------
def levenshtein(a, b) -> int:
    """Edit distance between two SEQUENCES (lists) whose elements are whole labels; textbook two-row DP."""
    a = list(a or [])
    b = list(b or [])
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def pc_score(seq_gen, seq_gt) -> float:
    """Procedural Correctness (L2): 1 - Levenshtein(S_gen, S_gt) / max(|S_gen|, |S_gt|).

    Both sequences empty -> 1.0 (nothing to do, done correctly). Result clamped to [0, 1].
    """
    a = list(seq_gen or [])
    b = list(seq_gt or [])
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    val = 1.0 - levenshtein(a, b) / m
    return max(0.0, min(1.0, val))


def tds_score(set_gen, set_req) -> dict:
    """Task Decomposition Score (L3): F1 over action-label SETS. Returns {precision, recall, f1}.

    P = |A_gen & A_req| / |A_gen|, R = |A_gen & A_req| / |A_req|, F1 = 2PR/(P+R). Edge cases (per THESIS section 6):
    empty A_gen or empty A_req -> all zero; F1 = 0 when P + R = 0. No ZeroDivisionError.
    """
    a = set(set_gen or [])
    b = set(set_req or [])
    if not a or not b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    inter = len(a & b)
    if inter == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = inter / len(a)
    r = inter / len(b)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


def _assertion_holds(a, val) -> bool:
    """Does ONE state value satisfy ONE assertion (`value` equality, or `contains` membership)?"""
    if "contains" in a:
        want = str(a.get("contains"))
        if isinstance(val, (list, tuple)):
            vals = list(val)
        elif val is None:
            vals = []
        else:
            vals = [val]
        return any(want in str(x) for x in vals)
    if "max" in a or "min" in a:
        # numeric bound, for the one goal equality cannot express: "go to the fridge" = Fridge.distance <= 1.5
        try:
            x = float(val)
        except (TypeError, ValueError):
            return False
        return (("max" not in a or x <= float(a["max"])) and ("min" not in a or x >= float(a["min"])))
    return val == a.get("value")


def sr_score(goal, get_state, get_all=None) -> bool:
    """Success Rate verdict for ONE record (all levels): True iff EVERY gold-goal assertion holds.

    `goal` is a list of assertions, each `{"type","key","value"}` (equality) or `{"type","key","contains"}`
    (membership -- e.g. a receptacle type inside parentReceptacles). `get_state(type, key) -> value` reads the
    resolved object's live metadata (in run() a closure over ThorAgent.object_state, with an isPickedUp special
    case). An empty/None goal returns False (cannot confirm success without a target).

    QUANTIFIERS. `get_state` resolves ONE object -- the nearest of its type -- which is exact for a Fridge and
    arbitrary for a Cabinet (2..28 per kitchen): "open a cabinet" would be judged on whichever door the agent
    happens to be standing next to. An assertion may therefore carry `"q": "any"` (some instance satisfies it)
    or `"q": "all"` (every instance does -- "close everything"), evaluated over `get_all(type, key) -> [values]`.
    Without `get_all`, or without a `q`, the behaviour is exactly the single-object one above, so every record
    written before quantifiers existed scores as it always did. `all` over zero instances is False: an absent
    type must never count as a success.
    """
    if not goal:
        return False
    for a in goal:
        if not isinstance(a, dict):
            return False
        q = a.get("q")
        if q in ("any", "all") and get_all is not None:
            vals = list(get_all(a.get("type"), a.get("key")) or [])
            hits = [_assertion_holds(a, v) for v in vals]
            if not hits or not (any(hits) if q == "any" else all(hits)):
                return False
        elif not _assertion_holds(a, get_state(a.get("type"), a.get("key"))):
            return False
    return True


# --------------------------------------------------------------------------------------------------------------
# PLAN REPAIR (pure; ported standalone from webapp._repair_plan_for_command so it reads passed-in scene types /
# held type instead of webapp module globals -- deterministic and unit-testable on Windows).
# --------------------------------------------------------------------------------------------------------------
def _type_eq(a, b) -> bool:
    return str(a or "").replace(" ", "").lower() == str(b or "").replace(" ", "").lower()


def _type_forms(object_type) -> set:
    """Spellings of an object type that might appear in a command (lowercase, space-split CamelCase, + aliases)."""
    ot = str(object_type or "")
    forms = {ot.lower(), re.sub(r"(?<=[a-z])(?=[A-Z])", " ", ot).lower()}
    forms |= {k for k, v in getattr(al, "TYPE_ALIASES", {}).items() if _type_eq(v, ot)}
    return {f for f in forms if f}


def _command_mentions_type(command, object_type) -> bool:
    cmd = str(command or "").lower()
    return any(re.search(rf"\b{re.escape(f)}s?\b", cmd) for f in _type_forms(object_type))


def pickup_target(command, scene_types) -> "str | None":
    """Object type targeted by a pickup/take/grab/hold command, or None (ported from webapp._pickup_target)."""
    cmd = str(command or "")
    if not re.search(r"\b(pick\s*up|pickup|grab|take|hold)\b", cmd, flags=re.I):
        return None
    types = sorted({t for t in (scene_types or []) if t})
    phrase = list(re.finditer(
        r"\b(?:pick\s*up|pickup|grab|take|hold)\s+(?:the\s+|a\s+|an\s+)?(.+?)"
        r"(?=\s+(?:and|then|after|before|from|off|on|in|inside|to)\b|[,.!?;]|$)", cmd, flags=re.I))
    if phrase:
        frag = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", phrase[-1].group(1))).strip().lower()
        for ot in sorted(types, key=len, reverse=True):
            if any(re.search(rf"\b{re.escape(f)}s?\b", frag) for f in _type_forms(ot)):
                return ot
        if frag:
            can = al.canonical_type(frag)
            for ot in types:
                if _type_eq(can, ot) or _type_eq(frag, ot):
                    return ot
            return str(can or frag).strip().title().replace(" ", "")
    for ot in sorted(types, key=len, reverse=True):
        if _command_mentions_type(cmd, ot):
            return ot
    m = re.search(r"\b(?:pick\s*up|pickup|grab|take|hold)\s+(?:the\s+|a\s+|an\s+)?(.+?)"
                  r"(?:\s+(?:from|off|on|in|inside|to)\b|$)", cmd, flags=re.I)
    if not m:
        return None
    frag = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", m.group(1))).strip()
    if not frag:
        return None
    can = al.canonical_type(frag)
    for ot in types:
        if _type_eq(can, ot) or _type_eq(frag, ot):
            return ot
    return str(can or frag).strip().title().replace(" ", "")


def repair_plan(command, steps, held_type=None, scene_types=None):
    """Patch incomplete simple pickup plans (webapp._repair_plan_for_command, standalone).

    Normalizes pick_up/pickup_object -> pickup; for a pickup/take/grab command whose target is NOT already held and
    has no pickup step, appends go_to <target> (if no approach step) + pickup <target>. Returns (repaired, changes).
    """
    repaired, changes = [], []
    for step in (steps or []):
        s = dict(step or {})
        if norm_action(s.get("action")) in ("pickup",) and str(s.get("action") or "").strip().lower() in (
                "pick_up", "pickup_object"):
            s["action"] = "pickup"
            changes.append("normalized pickup action")
        repaired.append(s)

    target = pickup_target(command, scene_types or [])
    if not target or _type_eq(held_type, target):
        return repaired, changes

    def is_target_step(step, actions):
        obj = step.get("object") or step.get("to") or step.get("arg")
        return norm_action(step.get("action")) in actions and _type_eq(obj, target)

    if any(is_target_step(s, {"pickup"}) for s in repaired):
        return repaired, changes
    if not any(is_target_step(s, {"go_to", "face_object", "scan_for", "search_for"}) for s in repaired):
        repaired.append({"action": "go_to", "object": target, "to": None, "arg": None})
        changes.append(f"added go_to {target}")
    repaired.append({"action": "pickup", "object": target, "to": None, "arg": None})
    changes.append(f"added pickup {target}")
    return repaired, changes


# --------------------------------------------------------------------------------------------------------------
# DATASET LOADING (pure stdlib)
# --------------------------------------------------------------------------------------------------------------
def load_dataset(path, lang="all", limit=None) -> list:
    """Load a JSONL (one record per line) or JSON-array dataset of CommandRecord dicts; validate each.

    Filters by `lang` ('all' | 'en' | 'fa') and caps at `limit` records (after filtering). Raises a clear error
    if the file is missing or a record fails levels.CommandRecord.validate().
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset not found at {path!r}. Author it as a JSONL/JSON list of CommandRecord dicts "
            f"(see sim/levels.py for the schema and example rows).")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    rows = []
    stripped = text.strip()
    import general_set
    whole = None
    if stripped.startswith("{"):
        try:                                   # a JSONL file also starts with "{", but is not ONE document
            whole = json.loads(stripped)
        except ValueError:
            whole = None
    if general_set.is_general_doc(whole):
        # The GENERAL instruction set (sim/general_set.py): one item per instruction holding BOTH languages and
        # a reference plan. It expands to scene-less records, which run()'s scene routing sends into every
        # requested scene; gold labels are derived from the plan, never hand-typed.
        rows = general_set.to_records(whole)
    elif stripped.startswith("["):
        rows = json.loads(stripped)
    else:
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{os.path.basename(path)} line {i}: invalid JSON ({exc})") from exc

    records = []
    for d in rows:
        rec = levels.CommandRecord.from_dict(d)
        rec.validate()
        if lang != "all" and rec.lang != lang:
            continue
        records.append(rec)
        if limit and len(records) >= int(limit):
            break
    return records


# --------------------------------------------------------------------------------------------------------------
# MEMORY BLOCK per --rag axis (live; imports rag/memory lazily)
# --------------------------------------------------------------------------------------------------------------
def render_hit_body(kind, entry) -> str:
    """The single source of truth for ONE rendered memory line (was inlined in _render_hits).

    Extracted verbatim so agentic_rag.kb_read can be handed the exact same templates (render=...) and the memory
    block stays byte-comparable with every existing eval_*.json run. Do not "improve" these strings: they are the
    prompt contract every published number was measured under.
    """
    e = entry or {}
    try:
        conf = float(e.get("confidence", 1.0))
    except Exception:
        conf = 1.0
    tag = f"_(src={e.get('source', '?')}, conf={conf:.2f})_"
    if kind == "procedures":
        steps = " ; ".join(f"{s.get('action')} {s.get('object') or ''}".strip()
                           for s in e.get("plan", []) or [])
        return f"- **worked before** -- \"{e.get('command')}\" -> {steps} {tag}"
    if kind == "lessons":
        return f"- **lesson** -- for \"{e.get('command')}\": {e.get('lesson')} {tag}"
    return f"- **fact** -- {e.get('text')} {tag}"


def _render_hits(hits) -> str:
    """Markdown for RAG hits (mirrors rag.HybridRAG.recall_text formatting)."""
    if not hits:
        return ""
    lines = ["**Relevant memory** (retrieved -- most relevant first):"]
    for h in hits:
        lines.append(render_hit_body(h.get("kind"), h.get("entry", {})))
        # The agentic arms may attach a one-line justification (e.g. agentic_rag.compact's "(xN)" merge marker).
        # Rendered ONLY when present, so the block stays byte-identical for off/keyword/hybrid and the existing
        # eval_*.json runs remain comparable.
        if h.get("why_line"):
            lines[-1] += f" {h['why_line']}"
    return "\n".join(lines)


def _est_tokens(text) -> int:
    """Stdlib token PROXY (a small local copy of agentic_rag.est_tokens, kept here so the pure module top has no
    sibling import): ceil(n_ascii/4 + n_nonascii/1.5). Persian costs far more tokens per character under a BPE
    trained on English, which is why the two halves are counted separately. A proxy, never a tokenizer."""
    s = str(text or "")
    n_ascii = sum(1 for c in s if ord(c) < 128)
    n_other = len(s) - n_ascii
    return -(-(3 * n_ascii + 8 * n_other) // 12)      # exact ceil(n_ascii/4 + n_other/1.5), no float rounding


def build_memory_block(mem, hybrid, scene, command, rag_mode, k=5, *,
                       agentic=None, kb=None, provider=None, model=None,
                       level=None, observation="", scene_types=None, cfg=None):
    """Build the planner's RELEVANT-MEMORY block, the retrieved-id/kind list, and the retrieval telemetry.

    Returns a THREE-tuple (block, retrieved, telemetry) -- the telemetry is new in Phase 10 and lands in
    res["retrieval"] so a memory gain can never be reported without its token / latency / model-call cost.

    Arms (see the module docstring for the full ladder):
      off                -> ("", [], telemetry). No memory block.
      keyword / hybrid   -> HybridRAG at alpha 1.0 / 0.5. UNCHANGED from Phase 8 -- these two arms must stay
                            byte-comparable with the published eval_*.json runs, so the alpha line is scoped to
                            them rather than acting as a catch-all default.
      probe / agentic    -> AgenticRAG.search (chunkless, index-free). Any failure falls back to
                            agentic_rag.keyword_fallback so this arm can never score worse than `keyword` for
                            INFRASTRUCTURAL reasons -- only for retrieval-quality ones.
      full               -> AgenticRAG.dump: every in-scope record, no retrieval, under a hard budget.
      anything else      -> ValueError. Load-bearing: before Phase 10 an unrecognised mode silently became
                            hybrid-at-0.5, i.e. a "new arm" could secretly be an old one.

    Structured spatial beliefs (read-only) are prepended for every non-off arm, matching memory.recall_text.
    """
    cfg = cfg or {}
    t0 = time.perf_counter()
    telemetry = {"mode": rag_mode, "hops": 0, "llm_calls": 0, "pivot_source": None, "terms": [],
                 "probe_ms": 0.0, "retrieval_ms": 0.0, "gate_stopped": None, "gate_reason": None,
                 "block_chars": 0, "est_tokens": 0, "cache": None, "degraded": None,
                 "level": level, "level_source": cfg.get("agentic_level", "est"),
                 "rank": cfg.get("rank", "plain"), "retrieved_ids": []}
    if rag_mode == "off":
        return "", [], telemetry

    hits = []
    if rag_mode in ("keyword", "hybrid"):
        if hybrid is None:
            return "", [], telemetry
        alpha = 1.0 if rag_mode == "keyword" else 0.5      # scoped to these two arms ONLY (see the docstring)
        try:
            hits = hybrid.search(command, scene=scene, k=k, alpha=alpha)
        except Exception as exc:
            telemetry["degraded"] = f"hybrid: {exc}"
            hits = []
    elif rag_mode in ("probe", "agentic"):
        if agentic is None:
            return "", [], telemetry
        try:
            hits = agentic.search(command, scene=scene, k=k, level=level, observation=observation,
                                  model=model, scene_types=scene_types)
            telemetry.update(agentic.last_telemetry())
        except Exception as exc:                            # never worse than keyword for infra reasons
            telemetry["degraded"] = f"agentic: {exc}"
            try:
                import agentic_rag as _ar                   # lazy: the module top stays pure stdlib
                hits = _ar.keyword_fallback(kb if kb is not None else getattr(agentic, "kb", None),
                                            command, scene=scene, k=k)
            except Exception:
                hits = []
    elif rag_mode == "full":
        if agentic is None:
            return "", [], telemetry
        try:
            hits = agentic.dump(scene=scene)
            telemetry.update(agentic.last_telemetry())
        except Exception as exc:
            telemetry["degraded"] = f"full: {exc}"
            hits = []
    else:                                                   # an unknown mode must FAIL LOUD, never be an old arm
        raise ValueError(f"unknown rag mode {rag_mode!r}")

    lines = []
    try:
        spatial = mem.query(scene, command, with_text=False).get("spatial", []) if mem else []
        if spatial:
            lines.append("**Known object locations** (structured memory):")
            lines.extend(mem._belief_line(b) for b in spatial)
    except Exception:
        pass
    block_hits = _render_hits(hits)
    if block_hits:
        if lines:
            lines.append("")
        lines.append(block_hits)
    block = "\n".join(lines)
    retrieved = [{"id": h.get("entry", {}).get("id"), "kind": h.get("kind")} for h in hits]
    telemetry["mode"] = rag_mode                            # last_telemetry() reports the RETRIEVER's own mode
    telemetry["rank"] = cfg.get("rank", "plain")
    telemetry["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    telemetry["block_chars"] = len(block)
    telemetry["est_tokens"] = _est_tokens(block)
    telemetry["retrieved_ids"] = [r["id"] for r in retrieved]
    return block, retrieved, telemetry


# --------------------------------------------------------------------------------------------------------------
# OBSERVATION per --perception axis (live)
# --------------------------------------------------------------------------------------------------------------
def observation_text(ag, scene, perception_mode, weights_dir=None):
    """OBSERVATION string for the planner prompt. Returns (text, degraded_bool).

    Thin delegate to ``perception.observation_text`` so the eval arm and the live console share ONE definition
    of what "perceived" means (they used to hold two copies, and this one looked for ``weights/<scene>.pt``
    while AI2-THOR names the live scene ``<scene>_physics`` — so the perceived arm degraded to oracle on a
    machine that had trained weights sitting right there). SR is ALWAYS scored from oracle metadata regardless
    of this axis; only the planner's view of the room changes.
    """
    try:
        import perception
    except Exception as exc:                                   # numpy-less environment: fail visibly, not silently
        try:
            return ag.describe(max_items=30) + f"{chr(10)}[perceived-mode degraded: {exc}]", True
        except Exception:
            return "", True
    return perception.observation_text(ag, scene=scene, mode=perception_mode,
                                       weights_dir=(weights_dir or os.path.join(HERE, "weights")))

# --------------------------------------------------------------------------------------------------------------
# AGGREGATION (pure)
# --------------------------------------------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def aggregate(results) -> dict:
    """Group per-record results and compute mean SR (all), PC (L2 only), TDS-f1 (L3 only) per group.

    Each group also carries the RETRIEVAL COST of that group -- mean retrieval latency, mean injected memory
    tokens and mean extra model calls -- so no memory gain is ever reported without its price. All three are
    None-safe and stay None for pre-Phase-10 result documents that have no res["retrieval"].
    """
    def block(subset):
        # A record the applicability gate skipped (object absent from the scene / goal already true) has no
        # verdict, so it enters no mean. `n` stays "records SCORED" -- what every mean below is over -- and the
        # skips are counted beside it rather than silently shrinking the denominator.
        n_skipped = sum(1 for r in subset if r.get("skipped"))
        subset = [r for r in subset if not r.get("skipped")]
        return {
            "n": len(subset),
            "n_skipped": n_skipped,
            "SR": _mean([1.0 if r.get("SR") else 0.0 for r in subset]),
            "PC": _mean([r.get("PC") for r in subset if r.get("level") == "L2"]),
            "TDS": _mean([(r.get("TDS") or {}).get("f1") for r in subset if r.get("level") == "L3"]),
            "retrieval_ms": _mean([(r.get("retrieval") or {}).get("retrieval_ms") for r in subset]),
            "mem_tokens": _mean([(r.get("retrieval") or {}).get("est_tokens") for r in subset]),
            "llm_calls": _mean([(r.get("retrieval") or {}).get("llm_calls") for r in subset]),
        }

    def group_by(key):
        buckets = {}
        for r in results:
            buckets.setdefault(r.get(key), []).append(r)
        return {str(k): block(v) for k, v in buckets.items()}

    by_config = {}
    for r in results:
        ck = f"{r.get('model')}|rag={r.get('rag')}|loop={r.get('loop')}|perc={r.get('perception')}|{r.get('scene')}"
        by_config.setdefault(ck, []).append(r)

    # by_retrieval: the Phase-10 sibling of by_config, crossing the retrieval MECHANISM (arm x ranking x hops x
    # pivot source x level) with the metrics. Added alongside by_config -- whose key format is pinned verbatim by
    # sim/_eval_selftest.py and must not be touched.
    by_retrieval = {}
    for r in results:
        t = r.get("retrieval") or {}
        rk = (f"rag={r.get('rag')}|rank={r.get('rank')}|hops={t.get('hops')}"
              f"|pivot={t.get('pivot_source')}|level={t.get('level')}")
        by_retrieval.setdefault(rk, []).append(r)

    return {
        "overall": block(results),
        "by_level": group_by("level"),
        "by_lang": group_by("lang"),
        "by_model": group_by("model"),
        "by_scene": group_by("scene"),
        "by_config": {k: block(v) for k, v in by_config.items()},
        "by_retrieval": {k: block(v) for k, v in by_retrieval.items()},
    }


def _fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def print_table(results, agg):
    """Human-readable summary table to stdout."""
    print("\n" + "=" * 78)
    print(f"EVAL SUMMARY  ({len(results)} records scored)")
    print("=" * 78)
    print(f"{'group':<40} {'n':>4} {'SR':>7} {'PC':>7} {'TDS':>7}")
    print("-" * 78)

    def row(name, b):
        print(f"{name[:40]:<40} {b['n']:>4} {_fmt(b['SR']):>7} {_fmt(b['PC']):>7} {_fmt(b['TDS']):>7}")

    row("OVERALL", agg["overall"])
    for section in ("by_level", "by_lang", "by_model", "by_scene"):
        if not agg.get(section):
            continue
        print(f"[{section}]")
        for name in sorted(agg[section]):
            row("  " + name, agg[section][name])
    print("[by_config]")
    for name in sorted(agg["by_config"]):
        row("  " + name, agg["by_config"][name])
    # Phase-10 cost table: quality is never reported without the tokens and latency it was bought with.
    if agg.get("by_retrieval") and any(b.get("mem_tokens") is not None or b.get("retrieval_ms") is not None
                                       for b in agg["by_retrieval"].values()):
        print("-" * 78)
        print(f"{'retrieval config':<40} {'n':>4} {'SR':>7} {'ms':>7} {'mtok':>7} {'calls':>6}")
        for name in sorted(agg["by_retrieval"]):
            b = agg["by_retrieval"][name]
            print(f"{('  ' + name)[:40]:<40} {b['n']:>4} {_fmt(b['SR']):>7} "
                  f"{_fmt(b.get('retrieval_ms')):>7} {_fmt(b.get('mem_tokens')):>7} "
                  f"{_fmt(b.get('llm_calls')):>6}")
    print("=" * 78)


# --------------------------------------------------------------------------------------------------------------
# RUN (live -- boots the simulator; thor_actions/webapp/memory/rag/improve imported lazily here)
# --------------------------------------------------------------------------------------------------------------
def run(cfg) -> dict:
    """Execute the full evaluation and return the results document {config, results, aggregates}."""
    import thor_actions as ta          # lazy: ai2thor exists only in the WSL sim environment
    import webapp                       # lazy: reuse _exec_one (the exact step dispatch); importing it is side-effect-free
    import knowledge as knowledge_mod
    import memory as memory_mod
    import rag as rag_mod

    # The scored-action space is a property of the RUN, so it is set once here and echoed into the results doc
    # below -- not read from cfg at every scoring call, where a later mutation could silently split one run
    # across two action spaces.
    _SCORE_SEARCH["on"] = str(cfg.get("score_search", "on")) != "off"
    cfg["scored_actions"] = sorted(set(_NAV_HELPERS) ^ set(_dropped_actions())) or ["(baseline)"]
    endpoint = cfg["endpoint"]
    provider = al.make_provider("ollama", endpoint=endpoint)
    records = load_dataset(cfg["dataset"], lang=cfg["lang"], limit=cfg["limit"])
    print(f"loaded {len(records)} records from {cfg['dataset']} "
          f"(lang={cfg['lang']}, limit={cfg['limit']})", flush=True)
    if not records:
        raise SystemExit("no records to evaluate (check --dataset / --lang / --limit)")

    # Retrieval / loop stores. With --loop off (default) these are used READ-ONLY (no close_loop, no passive
    # feedback writes) so a scoring pass is frozen; --loop on lets improve.close_loop enrich them between commands.
    # --kb / --memory / --loop-reports point a run at SNAPSHOT copies, so a `--loop on` ablation (which writes by
    # design) never mutates the canonical research stores. Defaults are the canonical files, as before.
    kb_path = cfg.get("kb_path") or os.path.join(HERE, "knowledge_base.json")
    mem_path = cfg.get("memory_path") or os.path.join(HERE, "memory.db")
    kb = knowledge_mod.KnowledgeBase(path=kb_path)
    mem = memory_mod.Memory(path=mem_path, kb_path=kb_path)
    # webapp._exec_one's `search_for` reads memory/KB through webapp's own lazy singletons; bind them to THIS run's
    # stores, or a search in a snapshot run would silently recall from the canonical files.
    webapp._kb, webapp._memory = kb, mem
    hybrid = None
    agentic = None
    if cfg["rag"] in ("keyword", "hybrid"):
        hybrid = rag_mod.HybridRAG(kb=kb, endpoint=endpoint)
    elif cfg["rag"] in ("probe", "agentic", "full"):
        import agentic_rag as ar_mod                 # lazy, inside run() -- the module top stays stdlib-pure
        agentic = ar_mod.AgenticRAG(
            kb=kb, provider=provider, model=(cfg["models"][0] if cfg["models"] else None),
            hops=(0 if cfg["rag"] == "probe" else int(cfg.get("agentic_hops", 2))),
            k=5,
            pivot_mode=("lex" if cfg["rag"] == "probe" else cfg.get("agentic_pivot", "auto")),
            rank_mode=cfg.get("rank", "plain"), quota_mode=cfg.get("agentic_quota", "typed"),
            gate=(cfg.get("agentic_gate", "on") == "on"), cache=(cfg.get("agentic_cache", "on") == "on"),
            compact=(cfg.get("agentic_compact", "on") == "on"),
            level_policy=cfg.get("agentic_level", "est"),
            budget_chars=int(cfg.get("agentic_budget", 6000)), timeout=cfg["timeout"])
        print(f"agentic retriever: {len(ar_mod.kb_entries(kb))} entries, "
              f"fingerprint={ar_mod.kb_fingerprint(kb)}, hops={agentic.hops}, "
              f"pivot={agentic.pivot_mode}, rank={agentic.rank_mode}, quota={agentic.quota_mode}, "
              f"gate={'on' if agentic.gate else 'off'}, level={agentic.level_policy}", flush=True)

    # RAG-axis integrity: `--rag hybrid` needs a reachable embedding model (nomic-embed-text). If it is down, the
    # cosine half is 0 for every entry and hybrid retrieval COLLAPSES to the exact keyword ranking -- so `--rag
    # hybrid` and `--rag keyword` would produce identical prompts and statistically indistinguishable scores,
    # silently voiding the thesis's keyword-vs-hybrid ablation. Probe ONCE; if degraded, warn loudly and stamp
    # every result (res['rag_degraded']=True) so the report can never conflate the two axes as if both ran.
    rag_degraded = False
    if cfg["rag"] == "hybrid":
        try:
            rag_mod.embed(["ping"], endpoint=endpoint)
        except Exception as exc:
            rag_degraded = True
            bar = "!" * 78
            print("\n" + bar
                  + f"\n[WARN] --rag hybrid requested but embeddings are UNAVAILABLE ({exc})."
                  + "\n       Hybrid retrieval falls back to keyword-only: --rag hybrid and --rag keyword give"
                  + "\n       IDENTICAL retrieval this run. Every result is stamped rag_degraded=true."
                  + "\n" + bar + "\n", flush=True)

    # RAG-axis integrity, agentic arms: probe/agentic/full need NO embedding server (nothing is ever embedded), so
    # the probe above is meaningless for them -- but they DO need a POPULATED knowledge base. An empty KB makes all
    # three retrieve nothing and score identically to `--rag off`, which is precisely how the Phase-8
    # off/keyword/hybrid ablation came out void (docs/phase8-eval.md). Same failure class, same loud banner, same
    # rag_degraded stamp -- fail LOUD rather than produce another null result.
    if cfg["rag"] in ("probe", "agentic", "full"):
        import agentic_rag as ar_mod
        n_entries = len(ar_mod.kb_entries(kb))
        if n_entries == 0:
            rag_degraded = True
            bar = "!" * 78
            print("\n" + bar
                  + f"\n[WARN] --rag {cfg['rag']} requested but the knowledge base is EMPTY (0 entries)."
                  + "\n       Every arm will retrieve nothing and score identically to --rag off."
                  + "\n       Populate the KB (teach pass / --loop on) before claiming a retrieval result."
                  + "\n       Every result is stamped rag_degraded=true."
                  + "\n" + bar + "\n", flush=True)
        # The agentic pivot is a MODEL call, so `--agentic-pivot llm|auto` with more than zero hops silently
        # becomes a pure lexical probe when the provider is unreachable. Warn (do not stamp: the arm still runs
        # exactly as `probe` would, which is a degradation of MECHANISM, not of the never-worse-than-keyword floor).
        if cfg["rag"] == "agentic" and cfg["models"] and int(cfg.get("agentic_hops", 2)) > 0 \
                and cfg.get("agentic_pivot", "auto") in ("llm", "auto"):
            try:
                provider.chat(cfg["models"][0], [{"role": "user", "content": "ping"}],
                              timeout=min(30.0, float(cfg["timeout"])), options={"temperature": 0.0, "seed": 0})
            except Exception as exc:
                print(f"[WARN] --rag agentic: the LLM pivot's provider is unreachable ({exc}); every hop will "
                      f"fail over to the lexical probe, so this run is effectively --rag probe.", flush=True)

    loop_on = (cfg["loop"] == "on")
    results = []
    interrupted = False
    # Stream every scored record to a JSONL sidecar as it completes so a late crash / Ctrl-C / GPU hiccup in a
    # multi-hour multi-model run never discards the hours already scored (the final JSON is only assembled at the
    # very end). Best-effort: if the sidecar can't be opened we still run, just without the incremental safety net.
    partial_path = os.path.splitext(cfg["out"])[0] + ".partial.jsonl"
    try:
        partial_fh = open(partial_path, "w", encoding="utf-8")
    except Exception:
        partial_fh = None
    # Resolve which scenes to boot and which records run in each. A record's own 'scene' field is authoritative
    # (its gold is grounded there); scene-less records fall back to the requested scenes (default FloorPlan1).
    tagged = [r for r in records if getattr(r, "scene", None)]
    sceneless = [r for r in records if not getattr(r, "scene", None)]
    if cfg["scenes"]:
        run_scenes = list(cfg["scenes"])                       # explicit --scenes filter
        tagged = [r for r in tagged if r.scene in run_scenes]
    elif tagged:
        run_scenes = sorted({r.scene for r in tagged})         # multi-scene dataset -> run all its scenes
    else:
        run_scenes = ["FloorPlan1"]
    cfg["scenes"] = run_scenes                                 # record the resolved set in the run config
    by_scene = {s: [r for r in tagged if r.scene == s] + sceneless for s in run_scenes}
    print("scene routing: " + ", ".join(f"{s}={len(by_scene[s])}" for s in run_scenes), flush=True)
    try:
        for scene in run_scenes:
            print(f"\n=== scene {scene}: booting ThorAgent (needs WSL sim + GPU) ===", flush=True)
            ag = None
            try:
                t0 = time.time()
                ag = ta.ThorAgent(scene=scene, width=600, height=600)
                ag.checkpoint()
                scene_types = ag.type_list(visible_only=False)
                print(f"booted in {time.time() - t0:.1f}s; {len(ag.objects())} objects", flush=True)
            except Exception as exc:
                print(f"scene {scene}: boot FAILED ({exc}); skipping", flush=True)
                if ag is not None:
                    try:
                        ag.stop()
                    except Exception:
                        pass
                continue
            try:
                for model in cfg["models"]:
                    # Per-model COLD start for the probe cache. The ProbeCache key is (kb fingerprint, scene,
                    # kinds, whole_word, terms, limit) and contains NO model, so in a --models sweep every model
                    # after the first replayed the same commands out of a warm cache; the per-record
                    # retrieval_ms that aggregate()/print_table report as the retrieval cost then mixed one cold
                    # sample with N-1 warm ones (measured 1.555 ms cold vs 0.767 ms warm on the bench fixture),
                    # making the 2nd and 3rd models look ~2x cheaper to retrieve for. Clearing between models
                    # makes every model's timing column comparable; --agentic-cache off still measures the
                    # no-cache path throughout. Review finding.
                    if agentic is not None:
                        try:
                            agentic.clear_cache()
                        except Exception:
                            pass
                    for rec in by_scene[scene]:
                        r = _score_record(
                            ag, provider, mem, hybrid, scene, scene_types, model, rec, cfg, loop_on, kb,
                            rag_degraded=rag_degraded, agentic=agentic)
                        results.append(r)
                        if partial_fh is not None:
                            try:
                                partial_fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                                partial_fh.flush()
                            except Exception:
                                pass
            finally:
                try:
                    ag.stop()
                except Exception:
                    pass
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] Ctrl-C received; assembling the results scored so far ...", flush=True)
    finally:
        try:
            mem.close()
        except Exception:
            pass
        if partial_fh is not None:
            try:
                partial_fh.close()
            except Exception:
                pass

    agg = aggregate(results)
    skips = {}
    for r in results:
        if r.get("skipped"):
            why = str(r["skipped"]).split(":")[0]
            skips[why] = skips.get(why, 0) + 1
    doc = {"config": cfg, "results": results, "aggregates": agg,
           "generated_ts": time.time(), "n_records": len(results), "n_skipped": sum(skips.values()),
           "skip_reasons": skips, "interrupted": interrupted}
    with open(cfg["out"], "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote {len(results)} results -> {cfg['out']}"
          + (f" (incremental stream: {partial_path})" if partial_fh is not None else ""), flush=True)
    print_table(results, agg)
    return doc


def _reset_world(ag, scene):
    """Reset the world + agent to the scene's clean initial state BEFORE scoring each record.

    ag.restore() only replays teleport + place_object_at_point + open/close + toggle from a pose/open/toggle
    checkpoint, so it CANNOT undo the irreversible mutations the L2/L3 vocabulary produces -- isCooked (cook_food),
    isSliced (slice; which even respawns *Sliced objectIds and removes the whole 'Tomato'), isBroken, isDirty,
    isFilledWithLiquid -- and it can leave an item stuck in the agent's hand. Left unreset, an early 'slice the
    tomato' / 'heat the potato' would corrupt every following record (and every later model, since the scene is
    booted once), making SR order-dependent and non-reproducible. A full controller.reset(scene) reloads the
    FloorPlan to its default state, undoing ALL of that AND emptying the hand. Falls back to restore() (plus an
    explicit hand-drop) only if a full reset is somehow unavailable."""
    try:
        ag.reset(scene)
        return
    except Exception:
        pass
    try:
        ag.restore()
    except Exception:
        pass
    try:
        if ag.held_object() is not None:
            ag.drop()
    except Exception:
        pass


# --------------------------------------------------------------------------------------------------------------
# APPLICABILITY GATE (pure helpers) -- a record that CANNOT be scored in this scene is reported, not failed.
# --------------------------------------------------------------------------------------------------------------
AGENT_TYPE = "Agent"            # pseudo-type for assertions about the robot itself (`Agent.isStanding`)


def gold_types(rec) -> list:
    """Every scene object type a record depends on: its goal types and destinations, the objects its gold
    labels name, and whatever its setup touches."""
    gold = rec.gold if isinstance(rec.gold, dict) else {}
    types = []
    for a in (gold.get("goal") or []):
        if isinstance(a, dict):
            types.append(a.get("type"))
            if a.get("key") == "parentReceptacles":
                types.append(a.get("contains"))
    for lab in list(gold.get("sequence") or []) + list(gold.get("actions") or []):
        parts = str(lab).split(" ", 1)
        if len(parts) == 2:
            types.append(parts[1].strip())
    for st in (getattr(rec, "setup", None) or []):
        types += [st.get("object"), st.get("to")]
    return sorted({t for t in types if t and t != AGENT_TYPE})


def missing_types(types, scene_types) -> list:
    """Which of `types` the scene does not have -- by EXACT type (after the usual alias canonicalisation).

    Deliberately stricter than ThorAgent.resolve, which falls back to a substring: in a kitchen with a Potato
    and no Pot, resolve("Pot") happily returns the potato. For execution that is a forgiving default; for the
    question "does this object exist here?" it is simply the wrong answer."""
    have = {str(t).lower() for t in (scene_types or [])}
    return [t for t in types if canon_type(t).lower() not in have]


def state_readers(ag):
    """(get_state, get_all) over a live ThorAgent -- the two readers `sr_score` judges a goal with.

    `get_state` resolves ONE object of the type (nearest), with two special cases: `Agent.*` reads the robot's
    own metadata, and `isPickedUp` asks "is the held object of this type?" (metadata says which instance is
    held, not which type the goal meant). `get_all` returns the value on EVERY instance, for any/all goals."""
    def get_state(otype, key):
        if otype == AGENT_TYPE:
            return (ag.event.metadata.get("agent") or {}).get(key)
        if key == "isPickedUp":
            h = ag.held_object()
            return bool(h) and _type_eq((h or {}).get("objectType"), otype)
        return ag.object_state(otype, key)

    def get_all(otype, key):
        return [o.get(key) for o in ag.objects() if _type_eq(o.get("objectType"), otype)]

    return get_state, get_all


def run_steps(ag, steps) -> list:
    """Execute plan steps through the EXACT console dispatch, headless. -> [{"ok", "error"}] per step."""
    import webapp
    out = []
    for step in (steps or []):
        try:
            gen = webapp._exec_one(ag, _exec_action_alias(step))
            if gen is not None and hasattr(gen, "__next__"):
                ag.drain(gen)
            ok = bool(ag.success)
            err = None if ok else (ag.error or "action failed")
        except Exception as exc:
            ok, err = False, str(exc)
        out.append({"ok": ok, "error": err})
    return out


def _skip(res, reason, t0):
    """Mark a record NOT APPLICABLE. It carries no SR/PC/TDS and `aggregate` leaves it out of every mean (and
    counts it in `n_skipped`): scoring it 0 would charge the model for the dataset's choice of scene, and
    scoring it 1 would credit the model for doing nothing."""
    res.update({"skipped": reason, "SR": None, "PC": None, "TDS": None,
                "elapsed_s": round(time.time() - t0, 3)})
    print(f"  [skip] {res['id']} @ {res['scene']}: {reason}", flush=True)
    return res


def _score_record(ag, provider, mem, hybrid, scene, scene_types, model, rec, cfg, loop_on, kb,
                  rag_degraded=False, agentic=None) -> dict:
    """Score ONE (model, record) pair. Exception-safe: any crash -> a 0/failure result with an 'error' field."""
    import webapp
    import improve
    t0 = time.time()
    level = rec.level
    gold = rec.gold or {}
    res = {
        "id": rec.id, "level": level, "lang": rec.lang, "command": rec.command,
        "scene": scene, "model": model, "rag": cfg["rag"], "loop": cfg["loop"],
        "perception": cfg["perception"], "metric": levels.metric_for(level),
        "SR": False, "PC": None, "TDS": None,
        "labels": [], "gold": gold, "say": "", "plan": [], "repairs": [],
        "n_steps": 0, "step_ok": [], "error": None, "perception_degraded": False,
        "rag_degraded": bool(rag_degraded), "elapsed_s": 0.0,
        "rank": cfg.get("rank", "plain"), "retrieval": {}, "violations": [], "repaired": False,
    }
    try:
        _reset_world(ag, scene)                         # FULL scene reload each record: undoes irreversible
                                                        # mutations (slice/cook/break/dirty/fill) AND empties the
                                                        # hand -- ag.restore() alone cannot (see _reset_world).

        get_state, get_all = state_readers(ag)          # shared by the gate below and the SR verdict at the end

        # ---- applicability gate, part 1: does the scene HAVE what the record is about?
        gate = cfg.get("gate", "on") == "on"
        if gate:
            absent = missing_types(gold_types(rec), scene_types)
            if absent:
                return _skip(res, f"object does not exist in this scene: {', '.join(absent)}", t0)

        # ---- setup: put the world in the state the command presupposes. Part of the RECORD, so it runs whether
        # or not the gate is on; it is executed by the same dispatch as a plan and is never shown to the model
        # except through the OBSERVATION it changes.
        for st in (getattr(rec, "setup", None) or []):
            s = run_steps(ag, [st])[0]
            if not s["ok"]:
                return _skip(res, f"setup step failed ({st.get('action')} {st.get('object') or ''}): "
                                  f"{s['error']}", t0)

        # ---- applicability gate, part 2: is the goal ALREADY true? Food spawns inside the fridge in some
        # kitchens (Apple is Cold at reset in FloorPlan2/5/9/14/27), so "make me a cold snack" is satisfied before
        # the agent moves, and an agent that returns an empty plan scores SR = 1.
        if gate and sr_score(gold.get("goal"), get_state, get_all):
            return _skip(res, "goal already true before the agent acted", t0)

        held = ag.held_object()
        held_type = (held or {}).get("objectType")

        obs, degraded = observation_text(ag, scene, cfg["perception"], weights_dir=cfg.get("weights_dir") or None)
        res["perception_degraded"] = degraded
        # LEVEL: only `--agentic-level oracle` may see the dataset's own L1/L2/L3 label (an explicit UPPER-BOUND
        # arm). Otherwise the retriever estimates the level from the command itself, because a deployed agent
        # never receives that label -- reading it by default would be an oracle the thesis does not have.
        lvl = level if cfg.get("agentic_level", "est") == "oracle" else None
        block, retrieved, tele = build_memory_block(
            mem, hybrid, scene, rec.command, cfg["rag"], agentic=agentic, kb=kb,
            provider=provider, model=model, level=lvl, observation=obs,
            scene_types=scene_types, cfg=cfg)
        res["retrieval"] = tele
        user = f"OBSERVATION:\n{obs}\n\nCOMMAND: {rec.command}"
        if block:
            user = f"RELEVANT MEMORY:\n{block}\n\n" + user

        # ---- plan (model call is the slow/fragile part: generous timeout + swallow any failure -> empty plan)
        content = ""
        try:
            content = provider.chat(
                model,
                [{"role": "system", "content": al.PLANNER_SYSTEM}, {"role": "user", "content": user}],
                timeout=cfg["timeout"], options={"temperature": 0.0, "seed": 0})
        except Exception as exc:
            res["error"] = f"model call failed: {exc}"
        try:
            parsed = al.parse_plan(content)
        except Exception as exc:
            parsed = {"say": "", "plan": []}
            if not res["error"]:
                res["error"] = f"plan parse failed: {exc}"
        res["say"] = parsed.get("say") or ""
        raw_steps = (parsed.get("plan") or [])[:25]

        steps, repairs = repair_plan(rec.command, raw_steps, held_type=held_type, scene_types=scene_types)
        steps = steps[:25]
        res["plan"] = steps
        res["repairs"] = repairs
        res["labels"] = labels(steps)

        # ---- M5: deterministic plan verification (+ ONE repair re-plan), BEFORE execution.
        # A SEPARATE axis from --rag (--agentic-verify, default off) so its SR contribution is never confounded
        # with the retrieval mode. The verifier itself costs zero model calls: it just enforces the two rules
        # PLANNER_SYSTEM already states in prose (known action, object present in the OBSERVATION).
        if cfg.get("agentic_verify") == "on":
            try:
                import agentic_rag as ar_mod
                import skills as skills_mod
                names = {s["name"] for s in skills_mod.skills_manifest()} | set(_NAV_HELPERS)
                viol = ar_mod.verify_plan(steps, set(scene_types or []), names)
                res["violations"] = viol
                if viol:
                    # ONE repair and no second try: re-probe memory on the offending symbols, re-plan once,
                    # re-verify, and keep the new plan ONLY if it violates strictly less (a repair may never
                    # make a record worse -- that would turn a verifier into a regression).
                    #
                    # The guard used to read `if viol and agentic is not None`, which made the repair branch
                    # UNREACHABLE on --rag off/keyword/hybrid: the verifier recorded violations there and never
                    # acted on them, so --agentic-verify had a structurally different effect per retrieval arm --
                    # exactly the confound the docstring above denies. build_memory_block already dispatches on
                    # `cfg["rag"]`, so the re-probe uses whichever retriever the arm under test defines (and no
                    # memory block at all under --rag off, where the re-plan still gets the rejection hint).
                    # Review finding; the axis is orthogonal now, and only now.
                    bad = sorted({str(v.get("value") or "") for v in viol if v.get("value")})
                    hint = ", ".join(bad)
                    rblock, rretr, rtele = build_memory_block(
                        mem, hybrid, scene, f"{rec.command} ({hint})", cfg["rag"], agentic=agentic, kb=kb,
                        provider=provider, model=model, level=lvl, observation=obs,
                        scene_types=scene_types, cfg=cfg)
                    ruser = (f"OBSERVATION:\n{obs}\n\nCOMMAND: {rec.command}\n\n"
                             f"YOUR PREVIOUS PLAN WAS REJECTED: unusable {hint}. Use ONLY actions from the skill "
                             f"library and ONLY object TYPES listed in the OBSERVATION.")
                    if rblock:
                        ruser = f"RELEVANT MEMORY:\n{rblock}\n\n" + ruser
                    rcontent = provider.chat(
                        model,
                        [{"role": "system", "content": al.PLANNER_SYSTEM}, {"role": "user", "content": ruser}],
                        timeout=cfg["timeout"], options={"temperature": 0.0, "seed": 0})
                    rparsed = al.parse_plan(rcontent)
                    rsteps, rrepairs = repair_plan(rec.command, (rparsed.get("plan") or [])[:25],
                                                   held_type=held_type, scene_types=scene_types)
                    rsteps = rsteps[:25]
                    rviol = ar_mod.verify_plan(rsteps, set(scene_types or []), names)
                    res["retrieval"] = dict(res["retrieval"] or {})
                    res["retrieval"]["repair"] = {"violations_before": len(viol), "violations_after": len(rviol),
                                                  "retrieval": rtele}
                    if len(rviol) < len(viol):
                        steps, repairs = rsteps, (repairs + ["verifier re-plan"] + rrepairs)
                        res["plan"], res["repairs"], res["labels"] = steps, repairs, labels(steps)
                        res["violations"] = rviol
                        res["repaired"] = True
                        seen_ids = {r.get("id") for r in retrieved}
                        retrieved = retrieved + [r for r in rretr if r.get("id") not in seen_ids]
            except Exception as exc:
                res["error"] = (res["error"] + " | " if res["error"] else "") + f"verify: {exc}"

        # ---- execute each step via the EXACT webapp dispatch (composites -> generators drained headless)
        step_results = run_steps(ag, steps)
        res["n_steps"] = len(steps)
        res["step_ok"] = [r["ok"] for r in step_results]

        # ---- SR (all levels): execution verified against the gold goal via oracle metadata (readers: see top)
        res["SR"] = bool(sr_score(gold.get("goal"), get_state, get_all))
        # ---- PC (L2) / TDS (L3): over the PLANNED semantic labels (S_gen), NOT execution
        if level == "L2":
            res["PC"] = pc_score(res["labels"], gold.get("sequence") or [])
        elif level == "L3":
            res["TDS"] = tds_score(set(res["labels"]), set(gold.get("actions") or []))

        # ---- self-improving loop (RQ3 axis): frozen unless --loop on
        # rag_enabled covers every non-off arm (keyword/hybrid/probe/agentic/full are all "retrieval enabled");
        # it only gates improve.close_loop's re-embed, which the agentic arms neither need nor use.
        if loop_on:
            try:
                improve.close_loop(
                    kb, mem, scene=scene, command=rec.command, level=level, arm="reuse_memory",
                    executed_steps=steps, step_results=step_results,
                    goal={"applicable": True, "ok": res["SR"]}, retrieved_ids=retrieved,
                    rag_enabled=(cfg["rag"] != "off"), loop_enabled=True,
                    embed_endpoint=cfg["endpoint"], report_path=(cfg.get("loop_reports_path") or None))
                # ^ was the bare name `endpoint`. `_score_record` has no such local and there is no module-level
                # one, so on the committed tree every --loop on record raised NameError here -- caught two lines
                # down and recorded as `loop closure: name 'endpoint' is not defined`, i.e. the loop silently did
                # nothing for the whole run.
                #
                # HONESTY NOTE, because the obvious inference from that is wrong: this does NOT retract the
                # published Phase-8 loop numbers. sim/eval_g3_exp1.json (loop=on, n=24) records zero errors, and
                # the loop reports it journaled matched its records. Those files (loop_reports.jsonl,
                # knowledge_base_audit.log) were deleted by a console reset on 2026-08-22; three report lines
                # survive verbatim in docs/research/observability-map-server.md. docs/eval-provenance.md (item 6)
                # establishes that exp1 ran on the UNCOMMITTED tree before e2349cb, whose call had no
                # `embed_endpoint` argument at all -- the NameError was introduced afterwards by the "wiring" fix
                # and committed untested. What the bug invalidates is REPRODUCIBILITY from any commit up to HEAD.
                # The like-for-like re-run on this tree is sim/ablation_suite.py --plans loop24.
            except Exception as exc:
                res["error"] = (res["error"] + " | " if res["error"] else "") + f"loop closure: {exc}"
            # Retrieval credit (feeds rank_key's utility term, i.e. the ONLY thing that separates
            # `--rank provenance+utility` from `--rank provenance`). Gated on --rank provenance+utility AND
            # --loop on, because it is a WRITE derived from this record's outcome and a frozen scoring pass must
            # stay frozen. `rag.credit` is implemented as of this pass; a failure here is REPORTED, not swallowed
            # -- the previous bare `except Exception: pass` is what let the arm silently collapse into a
            # duplicate of `--rank provenance` and still look healthy.
            if cfg.get("rank") == "provenance+utility":
                try:
                    import rag as rag_mod_local
                    rag_mod_local.credit(kb.path, retrieved, bool(res["SR"]),
                                         why=f"eval {rec.id} [{model}] {cfg['rag']}")
                    if agentic is not None:
                        # Re-read the sidecar the write just produced. AgenticRAG loads it once in __init__, so
                        # without this the credit accumulated during a run only takes effect on the NEXT run and
                        # the arm would still be indistinguishable from `--rank provenance` within one pass.
                        agentic.util = agentic._load_utility()
                except Exception as exc:
                    res["error"] = ((res["error"] + " | " if res["error"] else "")
                                    + f"retrieval credit: {exc}")
    except Exception as exc:
        res["error"] = (res["error"] + " | " if res["error"] else "") + f"record crashed: {exc}"
    res["elapsed_s"] = round(time.time() - t0, 2)
    m = res["metric"]
    mval = (res["SR"] if m == "SR" else res["PC"] if m == "PC"
            else (res["TDS"] or {}).get("f1") if m == "TDS" else None)
    print(f"  [{model}] {rec.id} {level}/{m} lang={rec.lang} "
          f"SR={int(res['SR'])} {m}={_fmt(mval if isinstance(mval, float) else (1.0 if mval else 0.0))} "
          f"steps={res['n_steps']}" + (f"  ERR={res['error'][:60]}" if res["error"] else ""), flush=True)
    return res


# --------------------------------------------------------------------------------------------------------------
# SELF-TEST (pure stdlib -- no simulator, no Ollama; `python eval.py --selftest`)
# --------------------------------------------------------------------------------------------------------------
def _selftest() -> int:
    fails = []

    def ok(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # -- action_label + normalization
    ok("label pickup", action_label({"action": "pickup", "object": "Apple"}) == "pickup Apple")
    ok("label heat->heat_object", action_label({"action": "heat", "object": "Potato"}) == "heat_object Potato")
    ok("label object wins over to",
       action_label({"action": "pick_and_place", "object": "Apple", "to": "Fridge"}) == "pick_and_place Apple")
    ok("label put receptacle", action_label({"action": "put", "object": "Fridge"}) == "put Fridge")
    ok("label to-only fallback", action_label({"action": "put_in", "to": "Fridge"}) == "put_in Fridge")
    ok("label canon type", action_label({"action": "open", "object": "refrigerator"}) == "open Fridge")
    ok("label turn_on_all->switch_all",
       action_label({"action": "turn_on_all", "object": "LightSwitch"}) == "switch_all LightSwitch")

    # -- labels() drops nav helpers
    plan = [{"action": "go_to", "object": "Knife"}, {"action": "pickup", "object": "Knife"},
            {"action": "go_to", "object": "Tomato"}, {"action": "slice", "object": "Tomato"}]
    ok("labels strip nav", labels(plan) == ["pickup Knife", "slice Tomato"])

    # -- levenshtein
    ok("lev equal", levenshtein(["a", "b"], ["a", "b"]) == 0)
    ok("lev one sub", levenshtein(["a", "b"], ["a", "c"]) == 1)
    ok("lev empty", levenshtein([], ["a", "b"]) == 2)

    # -- PC edge cases
    ok("pc both empty", pc_score([], []) == 1.0)
    ok("pc identical", pc_score(["pickup Apple"], ["pickup Apple"]) == 1.0)
    ok("pc one gen empty vs one gt", pc_score([], ["pickup Apple"]) == 0.0)
    ok("pc half", abs(pc_score(["a", "b"], ["a", "c"]) - 0.5) < 1e-9)

    # -- TDS edge cases
    t_full = tds_score({"pickup Apple", "cool_object Apple"}, {"pickup Apple", "cool_object Apple"})
    ok("tds full f1=1", abs(t_full["f1"] - 1.0) < 1e-9)
    ok("tds empty gen", tds_score(set(), {"a"}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    ok("tds empty req", tds_score({"a"}, set()) == {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    ok("tds disjoint f1=0", tds_score({"a"}, {"b"})["f1"] == 0.0)
    t_half = tds_score({"cool_object Apple"}, {"pickup Apple", "cool_object Apple"})
    ok("tds P=1 R=0.5 F1~0.667",
       abs(t_half["precision"] - 1.0) < 1e-9 and abs(t_half["recall"] - 0.5) < 1e-9
       and abs(t_half["f1"] - 2 / 3) < 1e-6)

    # -- sr_score with a dict-backed fake state reader
    state = {("Fridge", "isOpen"): True, ("Apple", "parentReceptacles"): ["Fridge|+00.1|+00.9|-01.3"],
             ("LightSwitch", "isToggled"): False}
    getf = lambda t, k: state.get((t, k))
    ok("sr equality all hold", sr_score([{"type": "Fridge", "key": "isOpen", "value": True}], getf) is True)
    ok("sr equality fails", sr_score([{"type": "Fridge", "key": "isOpen", "value": False}], getf) is False)
    ok("sr contains matches",
       sr_score([{"type": "Apple", "key": "parentReceptacles", "contains": "Fridge"}], getf) is True)
    ok("sr contains misses",
       sr_score([{"type": "Apple", "key": "parentReceptacles", "contains": "Microwave"}], getf) is False)
    ok("sr multi-assert AND",
       sr_score([{"type": "Fridge", "key": "isOpen", "value": True},
                 {"type": "LightSwitch", "key": "isToggled", "value": False}], getf) is True)
    ok("sr empty goal is False", sr_score([], getf) is False)

    # -- repair: adds go_to + pickup for a bare pickup command; leaves a complete plan alone
    r1, c1 = repair_plan("pick up the apple", [], held_type=None, scene_types=["Apple", "Fridge"])
    ok("repair adds go_to+pickup",
       labels(r1) == ["pickup Apple"] and any("added pickup" in c for c in c1))
    r2, c2 = repair_plan("pick up the apple",
                         [{"action": "go_to", "object": "Apple"}, {"action": "pickup", "object": "Apple"}],
                         held_type=None, scene_types=["Apple"])
    ok("repair leaves complete plan", labels(r2) == ["pickup Apple"] and c2 == [])
    r3, _ = repair_plan("heat the potato", [{"action": "heat", "object": "Potato"}],
                        held_type=None, scene_types=["Potato"])
    ok("repair no pickup for non-pickup command", labels(r3) == ["heat_object Potato"])

    # -- aggregate shape
    fake = [
        {"level": "L1", "lang": "en", "model": "m", "rag": "off", "loop": "off", "perception": "oracle",
         "scene": "FloorPlan1", "SR": True, "PC": None, "TDS": None},
        {"level": "L2", "lang": "fa", "model": "m", "rag": "off", "loop": "off", "perception": "oracle",
         "scene": "FloorPlan1", "SR": False, "PC": 0.5, "TDS": None},
        {"level": "L3", "lang": "en", "model": "m", "rag": "off", "loop": "off", "perception": "oracle",
         "scene": "FloorPlan1", "SR": True, "PC": None, "TDS": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
    ]
    agg = aggregate(fake)
    ok("agg overall SR=2/3", abs(agg["overall"]["SR"] - 2 / 3) < 1e-9)
    ok("agg L2 PC=0.5", abs(agg["by_level"]["L2"]["PC"] - 0.5) < 1e-9)
    ok("agg L3 TDS=1.0", abs(agg["by_level"]["L3"]["TDS"] - 1.0) < 1e-9)
    ok("agg by_lang keys", set(agg["by_lang"].keys()) == {"en", "fa"})

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(
        description="Phase-8 evaluation harness (SR/PC/TDS over the bilingual L1/L2/L3 dataset).")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="JSONL/JSON list of CommandRecord dicts")
    p.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated Ollama model tags")
    p.add_argument("--rag", choices=("off", "keyword", "hybrid", "probe", "agentic", "full"), default="hybrid",
                   help="retrieval grounding for the planner prompt: off | keyword (HybridRAG alpha=1.0) | "
                        "hybrid (alpha=0.5, needs embeddings) | probe (chunkless lexical, 0 extra model calls) | "
                        "agentic (probe + a bounded multi-hop object-type pivot) | full (dump the whole in-scope "
                        "KB -- the no-retrieval control)")
    # ---- Phase-10 mechanism switches. Every default reproduces the pre-Phase-10 behaviour where one exists
    # (--rank plain = similarity-only ordering; --agentic-verify off), so an existing invocation is unaffected.
    p.add_argument("--rank", choices=("plain", "provenance", "provenance+utility"), default="plain",
                   help="ranking function for probe/agentic (plain = similarity only, the control arm). "
                        "'provenance+utility' additionally writes retrieval credit via rag.credit and reads it "
                        "back through rank_key's utility term -- it needs --loop on to differ from 'provenance', "
                        "because the credit write is an outcome-derived write and a frozen pass stays frozen")
    p.add_argument("--agentic-hops", type=int, default=2, choices=(0, 1, 2, 3),
                   help="hop budget for extra LLM retrieval hops (0 = probe-only); ignored by --rag probe. "
                        "The LEVEL POLICY scales it (L1 x0, L2 x0.5, L3 x1), so at the default "
                        "--agentic-level est this is the L3 budget and L1 always gets 0")
    p.add_argument("--agentic-pivot", choices=("off", "lex", "llm", "auto"), default="auto",
                   help="query decomposition into AI2-THOR object types (the FA->EN bridge); "
                        "auto spends a model call only when a Persian command resolved no types lexically")
    p.add_argument("--agentic-gate", choices=("on", "off"), default="on",
                   help="deterministic sufficiency gate (off = always spend the full hop budget)")
    p.add_argument("--agentic-cache", choices=("on", "off"), default="on",
                   help="exact-match probe cache keyed on the KB content hash; turn OFF for timing runs")
    p.add_argument("--agentic-compact", choices=("on", "off"), default="on",
                   help="merge near-duplicate lessons and truncate the memory block to the budget")
    p.add_argument("--agentic-verify", choices=("on", "off"), default="off",
                   help="deterministic plan verifier + ONE repair re-plan. A genuinely separate axis from --rag: "
                        "the repair fires on EVERY retrieval arm, --rag off included")
    p.add_argument("--agentic-quota", choices=("mixed", "typed"), default="typed",
                   help="per-kind retrieval budget (typed = lessons are never crowded out by procedures)")
    p.add_argument("--agentic-budget", type=int, default=6000,
                   help="character budget for the retrieved memory block, at L3. The LEVEL POLICY scales it "
                        "(L1 x0.4, L2 x0.8, L3 x1.0), so the default 6000 reproduces the documented "
                        "2400/4800/6000 table and raising it genuinely grows the block at every level")
    p.add_argument("--agentic-level", choices=("off", "est", "oracle"), default="est",
                   help="level-adaptive retrieval policy; 'oracle' reads the dataset's L1/L2/L3 label and is an "
                        "explicit UPPER BOUND, not deployable behaviour; 'off' applies one uniform policy")
    p.add_argument("--loop", choices=("on", "off"), default="off",
                   help="self-improving loop: off = teach-then-freeze (memory NOT written during scoring)")
    p.add_argument("--perception", choices=("oracle", "perceived"), default="oracle",
                   help="observation source (SR is always scored from oracle metadata)")
    p.add_argument("--score-search", choices=("on", "off"), default="on", dest="score_search",
                   help="count search_for as a scored plan step in PC/TDS (default on). 'off' drops it like the "
                        "other perception helpers, reproducing the pre-2026-08-22 action space so results "
                        "published before search_for existed can be re-run like for like")
    p.add_argument("--gate", choices=("on", "off"), default="on",
                   help="applicability gate (default on): a record whose object is absent from the scene, or whose "
                        "goal is already true at reset, is reported as SKIPPED and left out of every mean. 'off' "
                        "scores every record regardless, reproducing runs made before the gate existed")
    p.add_argument("--scenes", default="", help="comma-separated iTHOR scenes to include; "
                   "empty = auto (each record runs in its own 'scene' field, or FloorPlan1 if untagged)")
    p.add_argument("--lang", choices=("all", "en", "fa"), default="all", help="dataset language filter")
    p.add_argument("--limit", type=int, default=None, help="cap the number of records (quick runs)")
    p.add_argument("--endpoint", default=None,
                   help="Ollama endpoint (default: gateway_endpoint() from WSL, else localhost)")
    p.add_argument("--timeout", type=float, default=180.0, help="per-plan-call timeout (seconds)")
    p.add_argument("--out", default=DEFAULT_OUT, help="results JSON path")
    p.add_argument("--kb", default="", dest="kb_path",
                   help="knowledge-base JSON to use (default: sim/knowledge_base.json). Point at a copy for ablations")
    p.add_argument("--memory", default="", dest="memory_path",
                   help="memory.db to use (default: sim/memory.db). Point at a copy for ablations")
    p.add_argument("--weights-dir", default="", dest="weights_dir",
                   help="directory holding <scene>.pt detectors for --perception perceived (default: sim/weights). "
                        "Lets a candidate detector be scored without replacing the deployed one")
    p.add_argument("--loop-reports", default="", dest="loop_reports_path",
                   help="loop-report JSONL for --loop on (default: sim/loop_reports.jsonl)")
    p.add_argument("--selftest", action="store_true", help="run the pure metric/repair self-tests and exit")
    return p


def _config_from_args(args) -> dict:
    endpoint = args.endpoint or al.gateway_endpoint() or DEFAULT_ENDPOINT
    return {
        "dataset": args.dataset,
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "rag": args.rag, "loop": args.loop, "perception": args.perception,
        "score_search": args.score_search, "gate": args.gate,
        "scenes": [s.strip() for s in args.scenes.split(",") if s.strip()],
        "lang": args.lang, "limit": args.limit, "endpoint": endpoint,
        "timeout": args.timeout, "out": args.out,
        # Phase-10 retrieval axes -- echoed at startup and stored in the results doc, so every run is
        # self-describing and an ablation can be reconstructed from its own JSON.
        "rank": args.rank,
        "agentic_hops": args.agentic_hops, "agentic_pivot": args.agentic_pivot,
        "agentic_gate": args.agentic_gate, "agentic_cache": args.agentic_cache,
        "agentic_compact": args.agentic_compact, "agentic_verify": args.agentic_verify,
        "agentic_quota": args.agentic_quota, "agentic_budget": args.agentic_budget,
        "agentic_level": args.agentic_level,
        # Which stores the run read (and, with --loop on, wrote). Empty = the canonical sim/ files.
        "kb_path": args.kb_path, "memory_path": args.memory_path, "loop_reports_path": args.loop_reports_path,
        "weights_dir": args.weights_dir,
    }


def main(argv=None) -> int:
    try:                                    # bilingual output: never let a non-UTF8 console crash the run
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    cfg = _config_from_args(args)
    print(f"eval config: {json.dumps(cfg, ensure_ascii=False)}", flush=True)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
