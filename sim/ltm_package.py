"""The assistance package: query, knowledge-graph expansion, hybrid retrieval, quotas with borrowing, a hard token budget
(final test, stage B7; PACKAGE-SPEC section 9, with Arash's English-only override).

  instruction (EN or FA) + holding + observation summary
      -> make_query   (A17 concept parser: types, ops, goal states; a not-ok parse still contributes its typed mentions)
      -> expand       (B4 knowledge graph: needed tools / appliances / receptacles; B3 contains / tool_at; B2 adjacency)
      -> candidates   (every store returns pkg.cand/1 with its own structural match on exact keys)
      -> score        (0.6 * match + 0.25 * BM25 over language-neutral keys + 0.15 * dense; dense is 0 without an embedder)
      -> quotas + borrowing, whole records, one fixed order, deterministic ties
      -> package text, ALWAYS English, <= 2,000 real tokens under EVERY tokenizer of the counter, header included

Rules that make this a retrieval system and not a prompt dump:
  * selective: nothing qualifies -> nothing is rendered (no padding, no header, `empty` is reported, never hidden);
  * the level label, the item id, the reference plan and the goal predicates are NOT inputs (`build` has no level
    parameter and refuses a query that carries one);
  * no simulator read anywhere in this module; the only privileged channel is the briefing records the caller hands in;
  * every kept record is listed with its provenance, so no rendered fact is unattributable;
  * EN and FA presentations of one command retrieve the same ids: lexical scoring uses language-neutral tokens only (the
    query's type and operation keys against the record's keys and English lex), never the instruction's surface words.

Pure Python 3.10 syntax. Imports nothing heavy at module load; bge-m3 is injected by the caller (`embedder`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARAMS_PATH = os.path.join(HERE, "step5", "package_params.json")
CONV_PATH = os.path.join(HERE, "step5", "briefing_conventions.json")

try:
    import pkg_schema as S
except ImportError:                                           # pragma: no cover
    from sim import pkg_schema as S
try:
    import ltm_concepts as LC
except ImportError:                                           # pragma: no cover
    from sim import ltm_concepts as LC

VERSION = S.VERSION
#: output order of the sections; the profile section also carries the `places` records (params.places_in)
EXAMPLE_LABEL = "practice-kitchen"
SECTIONS = ("briefing", "profile", "procedure", "example", "template", "uncertainty")
_FORBIDDEN_QUERY_KEYS = ("level", "item", "item_id", "reference_plan", "plan", "goal", "gold", "outcome")
_TYPE = re.compile(r"^[A-Z][A-Za-z0-9]+$")
_SKIP_ROLE = ("-", "?", "PRON", None, "")

DEFAULT_PARAMS = {
    "schema": S.SCHEMA_PARAMS, "version": VERSION, "enabled": False, "budget": 2000,
    "quotas": {"briefing": 300, "profile": 350, "procedure": 350, "example": 550, "template": 300, "uncertainty": 150},
    "borrow_order": ["procedure", "example", "template", "profile", "briefing", "uncertainty"],
    "places_in": "profile",
    "weights": {"exact": 0.6, "lexical": 0.25, "dense": 0.15},
    "irrelevant": {"min_match": 0.0, "min_dense_without_key": 0.5},
    "need_discount": 0.75,
    "k_max": {"example": 2, "template": 1},
    "fallback_core": {"component": "briefing", "max_lines": 5, "match": 0.3},
    "places_max_per_type": 2,
    "log_irrelevant_ids": False,
    "headers": {"en": {"briefing": "What the homeowner told me about this kitchen:",
                       "profile": "What I learned here in practice:",
                       "procedure": "How the tools work:",
                       "example": "",
                       "template": "A plan outline for this kind of task:",
                       "uncertainty": "Not known:"}},
    "uncertainty_text": {
        "knob_map": "Not learned: which StoveKnob drives which StoveBurner.",
        "never_seen": "Never seen in my practice: {types}.",
        "parser": "The command was only partly understood, so some relevant notes may be missing.",
    },
}


# ----------------------------------------------------------------------------------------------------------------- params
def params_sha(params: dict) -> str:
    return S.self_sha(params)


def load_params(path: str = None, verify: bool = True) -> dict:
    with open(path or PARAMS_PATH, encoding="utf-8") as f:
        p = json.load(f)
    if verify and p.get("sha256") != params_sha(p):
        raise ValueError("package_params.json: sha256 does not match its content (edit the params with --write-params, not by hand)")
    if p.get("version") != VERSION or p.get("schema") != S.SCHEMA_PARAMS:
        raise ValueError("package_params.json has the wrong schema or version")
    return p


def default_params_doc() -> dict:
    d = json.loads(json.dumps(DEFAULT_PARAMS))
    d["sha256"] = params_sha(d)
    return d


def check_params(p: dict) -> list:
    """Problems with a params dict (empty = valid). The budget arithmetic is checked here so a bad edit cannot reach a build."""
    out = []
    q, bo = p.get("quotas") or {}, p.get("borrow_order") or []
    if set(q) != set(SECTIONS):
        out.append("quotas must name exactly %s" % list(SECTIONS))
    if sorted(bo) != sorted(SECTIONS):
        out.append("borrow_order must list every section once")
    if not isinstance(p.get("budget"), int) or p["budget"] <= 0:
        out.append("budget must be a positive integer")
    w = p.get("weights") or {}
    if set(w) != {"exact", "lexical", "dense"} or abs(sum(w.values()) - 1.0) > 1e-9:
        out.append("weights must be exact/lexical/dense summing to 1")
    if p.get("places_in") != "profile":
        out.append("places_in must be 'profile' (the only supported placement)")
    if not (0.0 < float(p.get("need_discount", 0)) <= 1.0):
        out.append("need_discount must be in (0,1]")
    for c in ("example", "template"):
        if int((p.get("k_max") or {}).get(c, 0)) < 1:
            out.append("k_max.%s must be >= 1" % c)
    return out


# ------------------------------------------------------------------------------------------------------------ categories
def load_categories(path: str = CONV_PATH) -> dict:
    """{'fixture_types', 'tool_types', 'categories': {name: [types]}} from the written-once conventions file (kitchen independent)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {"fixture_types": list(d["fixture_types"]), "tool_types": list(d["tool_types"]),
            "categories": {k: list(v["types"]) for k, v in d["categories"].items()}}


# ----------------------------------------------------------------------------------------------------------------- query
def _is_type(x) -> bool:
    return isinstance(x, str) and bool(_TYPE.match(x)) and x not in ("PRON",)


_LINE = re.compile(r"^(?P<type>[A-Za-z]+)(?: x\d+)? \[(?P<state>[^\]]*)\]")


def observed_from_summary(summary_lines) -> list:
    """Types and states only, from the observation summary lines every arm already receives (never positions)."""
    seen, out = set(), []
    for ln in summary_lines or []:
        m = _LINE.match(str(ln).strip())
        if not m:
            continue
        t = m.group("type")
        states = [re.sub(r"^\d+\s+", "", s.strip()) for s in m.group("state").split(",") if s.strip()] or ["?"]
        for s in states:
            if (t, s) not in seen:
                seen.add((t, s))
                out.append({"type": t, "state": s})
    return sorted(out, key=lambda o: (o["type"], o["state"]))


#: stage I-F1-goalschema: the deterministic goal schema (sim/goal_schema.py) feeds the query. The instruction's operation / object /
#: destination / state words become the types, ops, goal_states and the (not rendered) `intent` key that retrieval is keyed on.
USE_GOAL_SCHEMA = True
#: schema op -> the legacy op tokens the retrieval and the procedural KB were written against
_SCHEMA_OP_LEGACY = {"toggle:on": ("toggle", "toggle_on"), "toggle:off": ("toggle", "toggle_off")}


def _apply_goal_schema(q: dict, instruction, lang) -> dict:
    """Adds the schema reading to a query (additive: legacy fields keep their values; types / ops / goal_states are unioned).

    `parser_fallback` / `parse` keep their legacy meaning (the package validator pins parser_fallback == not parse.ok). The schema's own
    verdict is `intent.status` and `intent.fallback`: True only when the schema could not read the request (`partial` / `unsupported`).
    An `indirect` request (facts / wishes, no operation word) is not a fallback: it carries the stated states in `intent.key`."""
    try:
        import goal_schema as GS
    except ImportError:                                       # pragma: no cover
        from sim import goal_schema as GS
    g = GS.parse_goal(instruction, lang if lang in ("en", "fa") else None, q.get("holding"))
    c = GS.compact(g)
    q = json.loads(json.dumps(q))
    types = set(q.get("types") or []) | {t for t in c["types"] if _is_type(t)}
    ops = list(q.get("ops") or [])
    for o in c["ops"]:
        for legacy in _SCHEMA_OP_LEGACY.get(o, (o,)):
            if legacy not in ops:
                ops.append(legacy)
    goals = list(q.get("goal_states") or [])
    for f in g["final_predicates"]:
        if f.get("state") and f.get("from") in ("stated_state", "wish", "declarative") and _is_type(f.get("type")):
            item = {"type": f["type"], "state": f["state"]}
            if item not in goals:
                goals.append(item)
    q["types"], q["ops"], q["goal_states"] = sorted(types), sorted(ops), goals
    q["intent"] = c
    return q


def make_query(instruction, lang, holding, summary_lines, lexicon=None) -> dict:
    """pkg.query/1. Reads only the instruction text, its language, the holding state and the summary lines (types / states)."""
    q = _make_query_legacy(instruction, lang, holding, summary_lines, lexicon)
    return _apply_goal_schema(q, instruction, lang) if USE_GOAL_SCHEMA else q


def make_query_profile(profile, instruction, lang, holding, summary_lines, lexicon=None) -> dict:
    """make_query under a named profile (see PROFILES) without touching the USE_GOAL_SCHEMA global. Same inputs, same honesty: the
    signature of make_query itself is pinned by the package gate (no level / item / plan parameter)."""
    if profile not in PROFILES:
        raise ValueError("unknown package profile %r" % (profile,))
    q = _make_query_legacy(instruction, lang, holding, summary_lines, lexicon)
    return _apply_goal_schema(q, instruction, lang) if profile in _SCHEMA_PROFILES else q


def _make_query_legacy(instruction, lang, holding, summary_lines, lexicon=None) -> dict:
    lex = lexicon or LC.default_lexicon()
    p = LC.parse(instruction, lex)
    roles = p["roles"]
    types, ops, goals = set(), [], []
    for r in roles:
        for f in ("obj", "src", "dest", "req"):
            if _is_type(r.get(f)):
                types.add(r[f])
        v = r.get("verb")
        if v not in _SKIP_ROLE and v not in ops:
            ops.append(v)
        for tf, sf in (("obj", "ostate"), ("dest", "dstate")):
            if _is_type(r.get(tf)) and r.get(sf) not in _SKIP_ROLE:
                g = {"type": r[tf], "state": r[sf]}
                if g not in goals:
                    goals.append(g)
    # typed mentions the parse could not place (a not-ok parse still contributes them; it never establishes eligibility)
    toks, i = LC.prep(instruction), 0
    while i < len(toks):
        n, t = LC._match_type(toks, i, lex)
        if n:
            if _is_type(t):
                types.add(t)
            i += n
        else:
            i += 1
    return {"schema": S.SCHEMA_QUERY, "instruction": instruction, "lang": lang, "holding": holding if _is_type(holding) else None,
            "observed": observed_from_summary(summary_lines),
            "parse": {"ok": bool(p["ok"]), "roles": roles, "reasons": list(p["reasons"])},
            "types": sorted(types), "ops": sorted(ops), "goal_states": goals,
            "need": {"types": [], "tools": [], "appliances": [], "receptacles": [], "ops": []},
            "parser_fallback": not p["ok"]}


def _query_check(query: dict) -> None:
    bad = [k for k in query if k in _FORBIDDEN_QUERY_KEYS or str(k).endswith("_ok")]     # also any verdict-like flag
    if bad:
        raise ValueError("package query carries forbidden keys %s: retrieval may use the instruction, language, holding state and "
                         "the observed types/states only" % bad)


def _members(categories: dict) -> dict:
    return {k: list(v) for k, v in ((categories or {}).get("categories") or {}).items()}


def expand(query: dict, kb, categories: dict, briefing_recs=(), profile=None, bound: dict = None) -> dict:
    """Fills query['need'] and returns a NEW query (the input is not mutated).

    B4 `kb.expand` gives the tools and appliances the operations require; B3 contains / tool_at give the receptacles that hold the
    needed types; B2 adjacency is used only for receptacles that are already in need (it never adds a type).

    bound (precision package, I-F3): {"tools": k, "appliances": k, "receptacles": k, "profile_receptacles": bool}. Each set is cut to its k
    first members (types the instruction names first, then alphabetical), and the profile-derived receptacles (where the needed types
    were seen in practice) are added only when profile_receptacles is true. None = the original unbounded expansion."""
    q = json.loads(json.dumps(query))
    types, ops = list(q.get("types") or []), list(q.get("ops") or [])
    tools, apps, recs, acts = set(), set(), set(), set()
    if kb is not None:
        e = kb.expand(ops, types, _members(categories))
        tools |= set(e.get("tools") or [])
        apps |= set(e.get("appliances") or [])
        recs |= set(e.get("receptacles") or [])
        acts |= set(e.get("actions") or [])
    toolset = set((categories or {}).get("tool_types") or [])
    # B3: where the needed types were found in practice -> those receptacles are needed too
    if profile is not None and (bound is None or bound.get("profile_receptacles", True)):
        for t in types + sorted(tools):
            fx = getattr(profile, "fixtures", {}) or {}
            for ft, d in fx.items():
                if t in (d.get("contents") or {}):
                    recs.add(ft)
            tl = (getattr(profile, "tools", {}) or {}).get(t)
            if tl and tl.get("last_parent_type"):
                recs.add(tl["last_parent_type"])
    direct = set(types)
    if bound:
        def cut(xs, k):
            return set(sorted(xs, key=lambda x: (x not in direct, x))[: int(k)]) if k is not None else set(xs)
        tools, apps, recs = cut(tools, bound.get("tools")), cut(apps, bound.get("appliances")), cut(recs, bound.get("receptacles"))
    q["need"] = {
        "types": sorted(direct | apps | recs | tools),
        "tools": sorted(tools | (direct & toolset)),
        "appliances": sorted(apps),
        "receptacles": sorted(recs),
        "ops": sorted(acts | set(ops)),
    }
    return q


# ------------------------------------------------------------------------------------------------------ scoring helpers
def _words(s: str) -> list:
    out, cur = [], []
    for ch in str(s).lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def bm25_norm(docs: list, qwords: list, k1: float = 1.5, b: float = 0.75) -> list:
    """Plain BM25 over the candidate set, normalised to [0,1] by the best score (0 everywhere when nothing matches). Same formula as
    fs_block._bm25_norm, so the example component ranks identically in the package and in the FS control."""
    toks = [_words(d) for d in docs]
    n = len(toks)
    if not n or not qwords:
        return [0.0] * n
    avg = (sum(len(t) for t in toks) / n) or 1.0
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    raw = []
    for t in toks:
        sc = 0.0
        for w in set(qwords):
            f = t.count(w)
            if not f:
                continue
            idf = max(0.0, math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)))
            sc += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avg))
        raw.append(sc)
    m = max(raw)
    return [r / m if m > 0 else 0.0 for r in raw]


def _why_types(why) -> list:
    out = []
    for w in why or []:
        parts = str(w).split(":")
        if parts[0] == "type" and len(parts) == 2:
            out.append(parts[1])
        elif parts[0] == "rel" and len(parts) == 3:
            out.append(parts[2])
    return out


def _rec_text(rec: dict) -> str:
    return str((rec.get("text") or {}).get("en") or "").strip()


def _anchor(rec: dict):
    t = (rec.get("keys") or {}).get("types") or []
    return t[0] if t else None


def _facts(rec: dict) -> set:
    """The (anchor, relation) facts a record states; used only to drop a duplicate (the same fact from two stores)."""
    a = _anchor(rec)
    out = set()
    for r in (rec.get("keys") or {}).get("rels") or []:
        rel, t = S.split_rel(r)
        if rel in ("contains", "tool_at", "opens", "toggles", "near", "stores"):
            out.add((a, r))
    return out


# --------------------------------------------------------------------------------------------------------- the sources
def briefing_candidates(recs, query: dict, params: dict) -> list:
    """pkg.cand/1 over the briefing records (the only privileged channel). Exact-key match only; B7 scores the rest.

    Direct = a type the instruction mentions; need = a type reached through the knowledge-graph expansion. A neighbour listed inside an
    adjacency line never matches by itself (adjacency is used only for an anchor that is already needed)."""
    direct = set(query.get("types") or [])
    need = set(query.get("need", {}).get("types") or []) | set(query.get("need", {}).get("tools") or []) \
        | set(query.get("need", {}).get("appliances") or []) | set(query.get("need", {}).get("receptacles") or [])
    need_only = need - direct
    out, seen = [], set()
    for r in recs or []:
        k = r.get("keys") or {}
        kind = r.get("kind") or str(r.get("id", "")).split("/")[-1].split(":", 1)[0]
        types = list(k.get("types") or [])
        anchor = types[0] if types else None
        m, why = 0.0, []
        if kind == "adj":
            if anchor in direct:
                m, why = 1.0, ["type:%s" % anchor]
            elif anchor in need_only:
                m, why = 0.6, ["type:%s" % anchor]
        elif kind == "conv":
            recept = {S.split_rel(x)[1] for x in k.get("rels") or [] if S.split_rel(x)[0] == "stores"}
            members = [t for t in types if t not in recept]
            hit_d, hit_n = sorted(direct & set(members)), sorted(need_only & set(members))
            if hit_d:
                m, why = 1.0, ["type:%s" % hit_d[0]]
            elif hit_n:
                m, why = 0.6, ["type:%s" % hit_n[0]]
            elif direct & recept:
                m, why = 0.5, ["type:%s" % sorted(direct & recept)[0]]
        else:
            if anchor in direct:
                m, why = 1.0, ["type:%s" % anchor]
            elif anchor in need_only:
                m, why = 0.7, ["type:%s" % anchor]
            elif kind not in ("core", "exist", "place", "open") and (set(types) & direct):
                m, why = 0.6, ["type:%s" % sorted(set(types) & direct)[0]]
        if m > 0:
            out.append({"schema": S.SCHEMA_CAND, "rec": r, "match": m, "why": why})
            seen.add(r["id"])
    if query.get("parser_fallback"):
        fc = params.get("fallback_core") or {}
        core = sorted((r for r in recs or [] if (r.get("kind") or "") == "core" or "/core:" in str(r.get("id", ""))),
                      key=lambda r: (int(r.get("prio", 0)), r["id"]))
        for r in core[: int(fc.get("max_lines", 5))]:
            if r["id"] not in seen:
                out.append({"schema": S.SCHEMA_CAND, "rec": r, "match": float(fc.get("match", 0.3)), "why": ["fallback:core"]})
    out.sort(key=lambda c: (-c["match"], c["rec"].get("prio", 0), c["rec"]["id"]))
    return out


class PlacesSource:
    """Adapter over the existing long-term memory (no edit to ltm.py): the learned `places` channel as pkg.rec/1.

    One record per asked type, grouped by receptacle type: 'Mug was last seen at: CounterTop (9 sightings)'. Only the places channel
    is read (practice location claims); it reads the store through `ltm.where_is`, so its tiers, epochs and retirement rules are the
    memory's own. Never a position, never a simulator id."""

    def __init__(self, ltm, wm=None, channels=("places",), tool_types=(), max_per_type: int = 2):
        self.ltm, self.wm, self.channels = ltm, wm, tuple(channels)
        self.tool_types, self.max_per_type = frozenset(tool_types), int(max_per_type)

    def _evidence(self, otype: str, rtype: str) -> list:
        try:
            rows = self.ltm.store.select("location_claim", where={"kitchen": self.ltm.kitchen, "otype": otype})
            places = {r["place"]: r["rtype"] for r in self.ltm.store.select("place_instance", where={"kitchen": self.ltm.kitchen})}
            return sorted(r["claim"] for r in rows if places.get(r["place"]) == rtype)[:6]
        except Exception:                                       # pragma: no cover - evidence is best effort, never decisive
            return []

    def candidates(self, query: dict) -> list:
        asked = list(dict.fromkeys(list(query.get("types") or []) + list((query.get("need") or {}).get("tools") or [])))
        kitchen = self.ltm.kitchen
        out = []
        for t in asked:
            ans = self.ltm.where_is(t, self.wm, channels=self.channels)
            groups = {}
            for c in ans.get("candidates") or []:
                rt = c.get("rtype")
                if not rt or c.get("kind") != "place":
                    continue
                g = groups.setdefault(rt, {"support": 0, "weight": 0.0, "tier": c.get("tier")})
                g["support"] += int(c.get("support") or 1)
                g["weight"] = max(g["weight"], float(c.get("weight") or 0.0))
            ranked = sorted(groups.items(), key=lambda kv: (-kv[1]["weight"], -kv[1]["support"], kv[0]))[: self.max_per_type]
            if not ranked:
                continue
            names = ", ".join(rt for rt, _g in ranked)
            n = sum(g["support"] for _rt, g in ranked)
            rel = ("tool_at" if t in self.tool_types else "contains")
            ev = []
            for rt, _g in ranked:
                ev += self._evidence(t, rt)
            rec = {"schema": S.SCHEMA_REC, "id": "places/%s/at:%s" % (kitchen, t), "component": "places", "kitchen": kitchen,
                   "keys": {"types": [ranked[0][0], t] if rel == "contains" else [ranked[0][0], t], "ops": [],
                            "rels": ["%s:%s" % (rel, t)]},
                   "lex": {"en": t.lower(), "fa": ""},
                   "text": {"en": "%s was last seen at: %s (%d sighting%s)." % (t, names, n, "" if n == 1 else "s"), "fa": ""},
                   "provenance": {"source": "practice:ltm/%s" % kitchen, "evidence": ev or ["where_is:%s" % t],
                                  "authored_by": "script:ltm_package.PlacesSource", "support": []},
                   "created": 0, "recency": 0, "confidence": round(min(1.0, n / 3.0), 4), "privileged": False, "prio": 1}
            out.append({"schema": S.SCHEMA_CAND, "rec": rec, "match": 1.0, "why": ["type:%s" % t]})
        out.sort(key=lambda c: (-c["match"], c["rec"]["prio"], c["rec"]["id"]))
        return out


# ------------------------------------------------------------------------------------------------ precision package (I-F3)
#: Operation words that mean "the object has to be found first". Only then do storage habits and learned places help the plan.
SEARCH_OPS = frozenset({"find", "search", "search_for", "look", "locate", "where"})
PUT_FAMILY = frozenset({"put", "put_in", "place", "pick_and_place"})
PRECISION_DEFAULTS = {
    "enabled": True,
    "recipe_max": 2,            # operation recipes ("To put food in the Microwave: ...")
    "procedure_max": 3,         # recipes + holding rule + tool lines together
    "example_max": 1, "template_max": 1,
    "max_fixtures": 4,          # fixtures named in the one kitchen line
    "show_near": False,         # relative positions ("next to"); compass lines are never rendered
    "storage": "search_only",   # generic storage conventions only when the goal needs a search
    "places": "search_only",    # learned places / profile only when the goal needs a search
    "keep_uncertainty": [],     # no "Not known" note by default: none of them can change a plan
    "holding_rule": True,       # one contract-derived line when the hand is full
    "hold_rewrite": True,       # pick_and_place -> put_in when the target is already held
    "expand_cap": {"tools": 2, "appliances": 2, "receptacles": 3},
    "headers": {"briefing": "This kitchen:", "profile": "Seen in practice:", "procedure": "Steps that work here:",
                "template": "Plan outline:", "uncertainty": "Not known:"},
}
PRECISION_QUOTAS = {"briefing": 90, "profile": 70, "procedure": 200, "example": 80, "template": 70, "uncertainty": 40}
_TOOL_VERB = re.compile(r"^\s*([a-z_]+)")
_VERBS = re.compile(r"\b(pickup|put_in|pick_and_place|toggle_on|toggle_off|close_all|close|open|search_for|go_to|slice|heat|cook|wash|drop)\b")


def precision_params(base: dict = None, budget: int = 300, quotas: dict = None, **over) -> dict:
    """The package params with the precision block on: small budget, small quotas, re-hashed. `over` overrides PRECISION_DEFAULTS keys
    (e.g. procedure_max=2). The base file is not touched."""
    p = json.loads(json.dumps(base if base is not None else load_params(verify=False)))
    pr = json.loads(json.dumps(PRECISION_DEFAULTS))
    pr.update(over)
    p["precision"] = pr
    p["budget"] = int(budget)
    p["quotas"] = dict(quotas or PRECISION_QUOTAS)
    p["enabled"] = True
    hd = json.loads(json.dumps(p.get("headers") or {"en": {}}))
    hd.setdefault("en", {}).update(pr["headers"])
    p["headers"] = hd
    p["sha256"] = params_sha(p)
    return p


#: stage I-round1-integrate: the named package profiles (the "flag" of PACKAGE-SPEC 14: default off, all arms together).
#:   pkg-1     the shipped, frozen behaviour: package_params.json as it is, the legacy query (no goal schema), 2,000-token budget.
#:   pkg-1.r1  round 1: the goal-schema query (I-F1) + the precision pass with the 300-token cap (I-F2 recipes arrive through the
#:             procedure KB, I-F3 through the precision block). Nothing is edited on disk: the params are a re-hashed copy.
PROFILE_LEGACY, PROFILE_R1 = "pkg-1", "pkg-1.r1"
#: L-round3: pkg-1.r3 = pkg-1.r1 + the parser-derived English request line (F1) + stated-fact affordances (F2). Declared expansion of
#: the intervention ("GP with parser-derived English hints", Astra round3-repair); r1 keeps its round-2 behaviour except the KB text (F3).
PROFILE_R3 = "pkg-1.r3"
#: N-round4: pkg-1.r4 = pkg-1.r3 + G1 (request_ops: the request rendered in English from the parser's own reading, shown only when the
#: parser typed every operation; declared expansion "GP with parser-derived English instruction rendering", Astra round4-repair) + G2
#: (precondition_closure: a requested toggle_on Microwave keeps the close X line, because the executor refuses to switch a Microwave on
#: with its door open). Both are mechanisms; their development provenance (which dev cells motivated them) is kept OUTSIDE this file by
#: the signal harness, which switches a mechanism off for the items it was tuned on in the strict condition (profile_params(off=...)).
PROFILE_R4 = "pkg-1.r4"
#: K-round6 (amendment R6-D1D2, sim/final_test/signal/r6/amendment_r6.json; declared CHANGE OF SCOPE, Astra round6-need-rule):
#: pkg-1.r6 = pkg-1.r4 + need_rule (a NEED presentation, goal_schema.need_trigger, gets the homeowner inventory with the frozen uses
#: table and one instruction line, and loses every word -> type AFFORDANCE hint) + type_scope (the D2 global type filter: no record
#: that names an excluded type reaches the planner). need_rule is a mechanism (STRICT switches it off on the items that motivated it,
#: sim/final_test/signal/r6/mechanisms.json); type_scope is the same in every arm and cannot be switched off.
PROFILE_R6 = "pkg-1.r6"
#: K-round6b (amended development protocol R6b, sim/final_test/signal/r6b/amendment_r6b.json; Astra round6b-fixes, build spec
#: docs/final-test/kitchen_memory/ROUND6B-FIXES.md sections 2-4): pkg-1.r6b = pkg-1.r6 + need_hand. A placeholder NEED whose placeholder
#: is a THING word (goal_schema.need_trigger_r6b -> placeholder "thing", frozen token list sim/step5/lexicon/need_cues_r6b.json) gets
#: NEED_THING_INSTRUCTION INSTEAD OF NEED_INSTRUCTION (replacement, not append). Place placeholders, desired-state NEEDs and non-NEEDs are
#: exactly pkg-1.r6. Everything else in the bundle, strip_need_hints and the route audit are unchanged (the rejected F2(a) hint
#: restoration is NOT here). "In hand" is a declared task convention (protocol choice R6b-C1), not a contract. need_hand is a mechanism:
#: STRICT switches it off on the items that motivated it (sim/final_test/signal/r6b/mechanisms.json), giving them the r6 line.
PROFILE_R6B = "pkg-1.r6b"
PROFILES = (PROFILE_LEGACY, PROFILE_R1, PROFILE_R3, PROFILE_R4, PROFILE_R6, PROFILE_R6B)
R4_MECHANISMS = ("request_ops", "precondition_closure")
R6_MECHANISMS = R4_MECHANISMS + ("need_rule",)
R6B_MECHANISMS = R6_MECHANISMS + ("need_hand",)
_SCHEMA_PROFILES = (PROFILE_R1, PROFILE_R3, PROFILE_R4, PROFILE_R6, PROFILE_R6B)
R1_BUDGET = 300
NEED_INSTRUCTION = "Select suitable in-scope resources and plan to satisfy the stated need or desired state, using supported actions."
NEED_THING_INSTRUCTION = ("Select a suitable in-scope item and get it into your hand. pickup finds and takes the item and requires "
                          "an empty hand; search_for does not pick it up. Do not perform the item's intended use unless the request "
                          "explicitly asks you to.")
USES_PATH = os.path.join(HERE, "step5", "uses_table.json")
NEED_CUES_R6B_PATH = os.path.join(HERE, "step5", "lexicon", "need_cues_r6b.json")
INVENTORY_PATH = os.path.join(HERE, "step5", "homeowner_inventory.json")
DEFAULT_KITCHEN = "FloorPlan1"


def uses_path(kitchen: str = DEFAULT_KITCHEN) -> str:
    """K23 (Astra K23-AGENT-r1): the uses table of a kitchen. FloorPlan1 keeps the frozen r6b file; a replication kitchen has its own
    successor file step5/uses_table_<kitchen>.json (never the FloorPlan1 file, never silently missing)."""
    return USES_PATH if kitchen == DEFAULT_KITCHEN else os.path.join(HERE, "step5", "uses_table_%s.json" % kitchen)


def inventory_path(kitchen: str = DEFAULT_KITCHEN) -> str:
    """K23: the homeowner inventory file of a kitchen (FloorPlan1: the frozen r6 file; else step5/homeowner_inventory_<kitchen>.json)."""
    return INVENTORY_PATH if kitchen == DEFAULT_KITCHEN else os.path.join(HERE, "step5", "homeowner_inventory_%s.json" % kitchen)


def _rel_step5(path: str) -> str:
    return "step5/" + os.path.basename(path)


def profile_params(profile: str = PROFILE_R1, base: dict = None, off=(), kitchen: str = DEFAULT_KITCHEN) -> dict:
    """The params of a profile, always with enabled=True (a caller that asks for a profile wants the package). The shipped file is not
    edited. Returns a dict that also carries `profile` and `goal_schema` so a caller can pass both to make_query / PackageBuilder.
    `off` (pkg-1.r4 / pkg-1.r6 / pkg-1.r6b only): mechanism names of R4_MECHANISMS / R6_MECHANISMS / R6B_MECHANISMS switched off (the
    strict leave-out condition of the signal harness)."""
    if profile not in PROFILES:
        raise ValueError("unknown package profile %r (known: %s)" % (profile, ", ".join(PROFILES)))
    off = tuple(off or ())
    allowed = {PROFILE_R4: R4_MECHANISMS, PROFILE_R6: R6_MECHANISMS, PROFILE_R6B: R6B_MECHANISMS}.get(profile, ())
    if off and set(off) - set(allowed):
        raise ValueError("off=%r is only for %s mechanisms %s / %s mechanisms %s / %s mechanisms %s"
                         % (off, PROFILE_R4, R4_MECHANISMS, PROFILE_R6, R6_MECHANISMS, PROFILE_R6B, R6B_MECHANISMS))
    if profile == PROFILE_LEGACY:
        p = json.loads(json.dumps(base if base is not None else load_params(verify=False)))
        p["enabled"] = True
    elif profile == PROFILE_R3:
        p = precision_params(base, budget=R1_BUDGET, request_line=True, state_afford=True)
    elif profile == PROFILE_R4:
        p = precision_params(base, budget=R1_BUDGET, request_line=True, state_afford=True,
                             **{m: m not in off for m in R4_MECHANISMS})
    elif profile == PROFILE_R6:
        p = precision_params(base, budget=R1_BUDGET, request_line=True, state_afford=True, type_scope=True,
                             uses_sha=_file_sha(uses_path(kitchen)), inventory_sha=_file_sha(inventory_path(kitchen)),
                             **{m: m not in off for m in R6_MECHANISMS})
    elif profile == PROFILE_R6B:
        p = precision_params(base, budget=R1_BUDGET, request_line=True, state_afford=True, type_scope=True,
                             uses_sha=_file_sha(uses_path(kitchen)), inventory_sha=_file_sha(inventory_path(kitchen)),
                             need_cues_r6b_sha=_file_sha(NEED_CUES_R6B_PATH),
                             **{m: m not in off for m in R6B_MECHANISMS})
    else:
        p = precision_params(base, budget=R1_BUDGET)
    p["profile"], p["goal_schema"] = profile, profile in _SCHEMA_PROFILES
    p["sha256"] = params_sha(p)
    return p


def _file_sha(path: str) -> str:
    import hashlib
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return ""


def _type_scope():
    try:
        import type_scope as TS_
    except ImportError:                                           # pragma: no cover
        from sim import type_scope as TS_
    return TS_


_USES_CACHE, _INV_CACHE = {}, {}


def load_uses_table(path: str = USES_PATH) -> list:
    """K-round6: the frozen uses table -> [(type, use)] in its fixed order, D2-excluded types removed (they are not in the file either)."""
    if path not in _USES_CACHE:
        doc = json.load(open(path, encoding="utf-8"))
        TS_ = _type_scope()
        _USES_CACHE[path] = [(r["type"], r["use"]) for r in doc["rows"] if TS_.in_scope(r["type"])]
    return list(_USES_CACHE[path])


def load_inventory(kitchen: str, path: str = INVENTORY_PATH) -> list:
    """K-round6 option C: the homeowner-reported types of a kitchen (privileged, authorized briefing), D2 filter applied; [] if unknown."""
    if path not in _INV_CACHE:
        _INV_CACHE[path] = json.load(open(path, encoding="utf-8"))
    TS_ = _type_scope()
    return sorted(TS_.filter_types((_INV_CACHE[path].get("kitchens") or {}).get(kitchen) or []))


def need_bundle_text(inventory: list, uses: list, seen: list, instruction: str = NEED_INSTRUCTION) -> str:
    """The NEED bundle: every permitted uses row of the inventory types in the table's fixed order (no matching to the need word, no
    ranking, no truncation), the agent's own sightings as a separate list, and the one instruction line (pkg-1.r6b need_hand: a
    placeholder-THING NEED gets NEED_THING_INSTRUCTION in its place; the default is the r6 line, byte for byte)."""
    inv = set(inventory)
    rows = ["%s: %s." % (t, u) for t, u in uses if t in inv]
    rows += ["%s." % t for t in sorted(inv - {t for t, _u in uses})]       # an inventory type without a row is still listed
    seen_txt = ", ".join(sorted(set(seen))) if seen else "none yet"
    return ("Kitchen inventory: the homeowner reports these types are in this kitchen (reported, not seen by you; pick up only what "
            "you actually find), each with its everyday use (uses add no actions beyond the tools):\n" + "\n".join(rows)
            + "\nSeen by you so far: %s.\n" % seen_txt + instruction)


def strip_need_hints(q: dict) -> dict:
    """K-round6: a NEED presentation keeps the request's own words but loses every word -> type hint: the parser's affordance types,
    and the operations a state word only implies (via_state: 'cold' -> chill), which would pull the implied fixture in through the
    expansion. Returns a new query."""
    q = json.loads(json.dumps(q))
    it = q.get("intent")
    if not isinstance(it, dict):
        return q
    implied = {k.get("op") for k in it.get("key") or [] if k.get("via_state")}
    stated = {k.get("op") for k in it.get("key") or [] if not k.get("via_state")}
    drop = {o for o in implied - stated if o}
    it["key"] = [k for k in it.get("key") or [] if not k.get("via_state")]
    it["affordance_types"] = []
    it["ops"] = [o for o in it.get("ops") or [] if o not in drop]
    legacy = {x for o in drop for x in _SCHEMA_OP_LEGACY.get(o, (o,))}
    q["ops"] = sorted(o for o in q.get("ops") or [] if o not in legacy)
    return q


def _norm_text(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", str(t).lower()))


def _is_tool_rec(r: dict) -> bool:
    return str(r.get("id", "")).split("/")[-1].startswith("tool-")


def _rec_verb(r: dict) -> str:
    m = _TOOL_VERB.match(_rec_text(r))
    return m.group(1) if m else ""


def _polarity(instruction: str):
    """'on' / 'off' / None from the instruction words (English and Persian), used only to pick between toggle_on and toggle_off."""
    low = " " + re.sub(r"[^\w؀-ۿ]+", " ", str(instruction).lower()) + " "
    on = any(w in low for w in (" on ", "turn on", "switch on", "start ", "روشن"))
    off = any(w in low for w in (" off ", "turn off", "switch off", "stop ", "خاموش"))
    return "on" if on and not off else ("off" if off and not on else None)


#: "...something to cut with." / "get me a butter knife to cut the tomato with.": a purpose clause that ends the request asks for the TOOL
#: (J-P2, Astra). English only; the Persian side relies on the parser's need=tool.
_PURPOSE_WITH = re.compile(r"\bto\s+[a-z]+(?:\s+[a-z]+){0,6}\s+with\s*[.!?]*\s*$", re.I)
_MUTATING = re.compile(r"\b(slice|heat|cook|wash|break|fill|empty)\b")


def is_tool_request(q: dict) -> bool:
    """True when the request asks for a tool and nothing else: the parser said need=tool with no object, or a purpose clause
    ('... to cut the tomato with.') ends an instruction that names the tool. Reads nothing but the instruction text and its parse."""
    for k in (q.get("intent") or {}).get("key") or []:
        if k.get("need") == "tool" and not k.get("obj") and not k.get("neg"):
            return True
    return bool(_PURPOSE_WITH.search(str(q.get("instruction", "")))) and bool((q.get("intent") or {}).get("affordance_types"))


def holding_rewrite(text: str, held: str) -> str:
    """'pick_and_place it/X to R' -> 'put_in R' when the target is already in the hand. Text-level, contract-only."""
    if not held:
        return text
    return re.sub(r"pick_and_place (?:it|each one|[A-Za-z ]{1,24}?) to (\w+)", r"put_in \1", text)


def contradiction(rec: dict, q: dict, categories: dict = None):
    """A reason when a record would mislead given the CURRENT state, else None. Reasons: contradicts_holding (tells the agent to pick up
    what it holds, or to pick up with a full hand) and conflicts_destination (a storage habit that points an object somewhere other
    than the destination the instruction names; never in a search, where the habit tells the source, not the destination)."""
    held = q.get("holding")
    txt = _rec_text(rec)
    ops = set((rec.get("keys") or {}).get("ops") or [])
    types = set((rec.get("keys") or {}).get("types") or [])
    if held:
        if "pickup" in ops and (not types or held in types) and _is_tool_rec(rec):
            return "contradicts_holding"                      # 'the hand must be empty' with a full hand
        if re.search(r"\bpick ?up (?:the )?%s\b" % re.escape(held), txt, re.I):
            return "contradicts_holding"
        if _is_tool_rec(rec) and "pick_and_place" in ops:
            return "contradicts_holding"
    dest = stated_destinations(q, categories) if not needs_search(q) else set()      # (Astra) a search needs the SOURCE habit whatever the destination
    if dest:
        rels = [S.split_rel(x) for x in (rec.get("keys") or {}).get("rels") or []]
        stores = {t for r, t in rels if r == "stores"}
        members = {t for t in types if t not in stores}
        if stores and (members & set(q.get("types") or [])) and not (stores & dest):
            return "conflicts_destination"
    return None


def stated_destinations(q: dict, categories: dict = None) -> set:
    """Receptacle / fixture types the instruction itself names as a destination (parse role 'dest', else a named fixture type)."""
    out = {r.get("dest") for r in (q.get("parse") or {}).get("roles") or [] if _is_type(r.get("dest"))}
    if not out and (PUT_FAMILY & set(q.get("ops") or []) or q.get("holding")):
        fx = set((categories or {}).get("fixture_types") or [])
        out = {t for t in (q.get("types") or []) if t in fx}
    return {x for x in out if x}


def needs_search(q: dict) -> bool:
    return bool(SEARCH_OPS & set(q.get("ops") or []))


def _G_AFFORD() -> dict:
    """The goal schema's English affordance table (word -> [(type, role)]), imported lazily."""
    try:
        import goal_schema as G_
    except ImportError:                                           # pragma: no cover
        from sim import goal_schema as G_
    return G_.AFFORDANCE


#: N-round4 G1: plain English verbs for the parser's operation keys. An operation missing here (fill: its liquid is not kept by the
#: parser; make, keep, serve, tidy: their object is often a dish word the parser cannot type) makes the whole rendering fall back.
_RENDER_VERB = {"put": "put", "pick": "pick up", "close": "close", "open": "open", "toggle:on": "switch on", "toggle:off": "switch off",
                "slice": "slice", "wash": "wash", "clean": "clean", "find": "find", "heat": "heat", "cook": "cook", "drop": "drop",
                "empty": "empty", "break": "break", "eat": "eat", "discard": "throw away", "chill": "chill", "flip": "flip",
                "scrub": "scrub"}
#: fixtures are rendered as plain nouns, not type names: the kitchen line and the tool lines name the part a tool acts on (a burner is
#: switched through its StoveKnob, a put goes into the SinkBasin), and a parser alias must not pull the plan off it (r3 rule)
_RENDER_FIXTURE = {"Microwave": "the microwave", "Fridge": "the fridge", "Cabinet": "a cabinet", "Drawer": "a drawer", "StoveBurner": "the stove",
                   "StoveKnob": "the stove knob", "Faucet": "the faucet", "LightSwitch": "the light", "CoffeeMachine": "the coffee machine",
                   "Toaster": "the toaster", "Sink": "the sink", "SinkBasin": "the sink", "CounterTop": "the counter", "Shelf": "a shelf",
                   "ShelvingUnit": "the shelves", "GarbageCan": "the garbage can", "Floor": "the floor", "Window": "the window", "Stool": "a stool"}
_RENDER_FN = {"washed": "where things get washed", "cool": "where things are kept cool", "stay_fresh": "where things stay fresh"}


#: N-round4 G2: (requested operation, object type) -> (the tool whose line it needs, the note appended to that line). Contract facts of
#: the executor only: toggle_on on a Microwave whose door is open is refused (honest_skills; the toggle_on line already says so).
_PRECONDITIONS = {("toggle:on", "Microwave"): ("close", "Needed here: a Microwave must be closed when toggle_on runs.")}


def render_request(schema: dict):
    """N-round4 G1: the request as the deterministic parser read it, in plain English, or None when a faithful rendering is not
    possible. A translation of the user's own words, nothing more: operations in execution order with their objects, quantities,
    stated states, destinations (type or function), sources and tools; a negated operation as 'do not ...'; a wish as 'wants to ...';
    a purpose need as 'wants a tool to <op> with' (never the tool type: that stays behind the affordance gate). NOT rendered: implied
    operations (a state word such as 'cold' never becomes 'chill'), affordances, facts. Falls back (None) when the parser did not read
    the request as 'ok', left anything unresolved or ambiguous, or met an operation / object it could not type."""
    if not isinstance(schema, dict) or schema.get("status") != "ok" or schema.get("unresolved"):
        return None
    # the parser's declared need-role policy ("something for <verb>ing" read as a tool) is rendered with the user's own vague word
    # ("something to flip with"), so it stays as open as the request; any other ambiguity means no rendering
    if any(a.get("kind") != "need_role" for a in schema.get("ambiguities") or []):
        return None
    if any(e.get("note") for e in schema.get("entities") or []):          # an ambiguous alias ('table' -> CounterTop): no rendering
        return None
    seen_fx = set()

    def fx(t):
        if t not in _RENDER_FIXTURE:
            return t
        n = _RENDER_FIXTURE[t]
        if t in seen_fx and n.startswith("a "):
            n = "the " + n[2:]                                             # 'put Bowl to a cabinet; close the cabinet'
        seen_fx.add(t)
        return n
    ops = schema.get("operations") or []
    parts = []
    for o in [ops[k] for k in (schema.get("order") or range(len(ops)))]:
        verb = _RENDER_VERB.get(o.get("op"))
        objs = o.get("obj") or []
        if verb is None or not objs:
            return None
        names = []
        for x in objs:
            t = x.get("type")
            if not t:
                return None
            n = "everything" if t == "*" else fx(t)
            if x.get("quantity") and t != "*":
                n = "%s %s" % (x["quantity"], n)
            if x.get("state"):
                n += " (stated: %s)" % x["state"]
            names.append(n)
        txt = "%s %s" % (verb, " and ".join(names))
        src, dest, tool = o.get("src") or {}, o.get("dest") or {}, o.get("tool") or {}
        if src.get("type"):
            txt += " from %s" % fx(src["type"])
        elif src:
            return None
        if dest.get("type"):
            txt += (" in " if o.get("op") in ("heat", "cook", "chill") else " to ") + fx(dest["type"])
        elif dest.get("fn"):
            if dest["fn"] not in _RENDER_FN:
                return None
            txt += " " + _RENDER_FN[dest["fn"]]
        elif dest:
            return None
        if tool.get("type"):
            txt += " with %s" % fx(tool["type"])
        elif tool:
            return None
        if o.get("neg"):
            txt = "do not " + txt
        elif o.get("mood") == "desire":
            txt = "wants to " + txt
        elif o.get("mood") not in (None, "imperative"):
            return None
        parts.append(txt)
    for n in schema.get("needs") or []:
        if n.get("kind", "tool") == "tool" and n.get("op") in _RENDER_VERB and not n.get("type"):
            fo = n.get("for") or []
            typed = [fx(x["type"]) for x in fo if x.get("type")]
            txt = "wants something to %s%s with" % (_RENDER_VERB[n["op"]], (" " + " and ".join(typed)) if typed else "")
            if len(typed) < len(fo):
                txt += " (an object word was not translated)"
            parts.append(txt)
        elif n.get("kind") == "item" and n.get("type"):
            parts.append("needs %s" % n["type"] + (" to %s" % _RENDER_VERB[n["op"]] if n.get("op") in _RENDER_VERB else ""))
        else:
            return None
    return "; ".join(parts) if parts else None


def _parse_request(q: dict):
    try:
        import goal_schema as G_
    except ImportError:                                           # pragma: no cover
        from sim import goal_schema as G_
    lang = q.get("lang")
    return G_.parse_goal(q.get("instruction") or "", lang if lang in ("en", "fa") else None, q.get("holding"))


def _state_afford_words(kb, k: dict) -> tuple:
    """L-round3 F2: a STATED fact with no operation ("the egg is rotten": state bad) whose parser affordance is a destination or an
    appliance reaches the KB affordance rows that use that type ("bad" -> GarbageCan -> the discard row). Mechanism only: the
    words returned are the KB's own row keys; nothing new is written."""
    st = k.get("state")
    if not st or k.get("op") or k.get("neg") or not hasattr(kb, "afford_map"):
        return ()
    types = {t for t, role in _G_AFFORD().get(st, []) if role in ("dest", "appliance")}
    return tuple(sorted(key for key, row in kb.afford_map.items() if types & set(row.get("appliances") or ())))


# --------------------------------------------------------------------------------------------------------- the builder
class PackageBuilder:
    """Builds the package of ONE presentation. Stateless across presentations except for caches of token counts and embeddings.

    sources: {"briefing": list[pkg.rec/1], "profile": Profile, "places": PlacesSource, "procedure": KB, "example": ExampleIndex,
              "template": Templates}; any may be missing (= empty).
    counter: anything with counts(text) -> {tokenizer: n} (RealMaxCounter in production, FakeCounter in selftests).
    embedder: None, an object with .embed(texts) -> unit vectors (ltm_retrieval.Embedder), or a callable (query, cands) -> [0..1].
    strict: a source that raises stops the build (the default). strict=False records the error in the package and goes on (dev
            screening only; the package then says `source_errors`)."""

    def __init__(self, kitchen, sources: dict, counter, params: dict, embedder=None, cache=None, categories=None, strict: bool = True):
        probs = check_params(params)
        if probs:
            raise ValueError("bad package params: %s" % probs)
        self.kitchen, self.sources, self.counter, self.params = kitchen, dict(sources or {}), counter, params
        self.embedder, self.cache = embedder, cache if cache is not None else {}
        self.categories = categories or {}
        self.strict = strict
        self._cnt = {}
        self._need_active = False
        try:
            from planners.contract import PACKAGE_HEADER
        except ImportError:                                       # pragma: no cover
            from sim.planners.contract import PACKAGE_HEADER
        self.header = PACKAGE_HEADER
        self.hdrs = (params.get("headers") or {}).get("en") or {}

    # -- counting ----------------------------------------------------------------------------------------------
    def _counts(self, text: str) -> dict:
        if not text:
            return {}
        c = self._cnt.get(text)
        if c is None:
            c = {k: int(v) for k, v in self.counter.counts(text).items()}
            self._cnt[text] = c
        return c

    def _max(self, text: str) -> int:
        c = self._counts(text)
        return max(c.values()) if c else 0

    def _section_text(self, comp: str, recs: list) -> str:
        if not recs:
            return ""
        h = self.hdrs.get(comp, "")
        body = "\n".join(_rec_text(r) for r in recs)
        return (h + "\n" + body) if h else body

    def _complete(self, sections: dict) -> str:
        body = "\n".join(t for t in (sections[c] for c in SECTIONS) if t)
        return (self.header + "\n" + body) if body else ""

    # -- candidates --------------------------------------------------------------------------------------------
    def _source_cands(self, comp: str, query: dict, errors: list) -> list:
        src = self.sources.get(comp)
        if src is None:
            return []
        try:
            if comp == "briefing":
                return briefing_candidates(src if isinstance(src, (list, tuple)) else src.records(), query, self.params)
            if comp == "procedure":
                if self._prec() and (query.get("intent") or {}).get("key") is not None and hasattr(src, "afford_key"):
                    return self._procedure_with_afford(src, query)
                return list(src.candidates(query, include_zero=True))
            if comp == "example":
                return list(src.candidates(query, include_failures=False,
                                           failure_fallback=bool(self.params.get("example_failure_fallback", False))))
            return list(src.candidates(query))
        except Exception as e:
            if self.strict:
                raise
            errors.append({"component": comp, "error": "%s: %s" % (type(e).__name__, e)})
            return []

    def _request_rec(self, q: dict):
        """L-round3 F1 (declared expansion: "GP with parser-derived English hints"; Astra round3-repair). ONE English line with the
        types the deterministic parser resolved for the request, so a Persian word reaches the planner as the right type name
        ("This request names: Spoon."; movable objects only, fixtures are left to the kitchen line). A stated state is shown as the parser read it ("Egg (stated: bad)"). A purpose need with
        no object ("something to scrub with") adds the tool type(s) of the parser's affordance table, unless the KB affordance row
        of that operation is left out for this presentation (tuned on it; the same leave-out as every record). Type names only,
        English only, identical for EN and FA; nothing from a reference plan, a level or an item id."""
        intent = q.get("intent") or {}
        if not (self._prec() or {}).get("request_line"):
            return None
        names, seen, fit = [], set(), []
        fixtures = set(self.categories.get("fixture_types") or [])
        for k in intent.get("key") or []:
            for role in ("obj", "dest", "src", "tool"):
                t = k.get(role)
                # movable objects only: fixtures are named by the kitchen line, and a parser alias for a fixture (stove ->
                # StoveBurner, sink -> Sink) could pull the plan away from the part the tools act on (StoveKnob, SinkBasin)
                if _is_type(t) and t not in seen and t not in fixtures:
                    seen.add(t)
                    st = k.get("state") if role == "obj" and not k.get("op") and k.get("state") and not k.get("neg") else None
                    names.append("%s (stated: %s)" % (t, st) if st else t)
            if k.get("need") == "tool" and not k.get("obj") and not k.get("neg") and k.get("op") and not self._need_active:
                # (K-round6: a NEED presentation never gets the word -> type tool hint; the uses table replaces it)
                kb = self.sources.get("procedure")
                ak = kb.afford_key(k["op"]) if kb is not None and hasattr(kb, "afford_key") else None
                if ak is not None:
                    alive = {r["id"] for r in kb.afford_records()}
                    if "procedure/*/afford-" + ak not in alive:
                        continue                                  # the row is left out for this item: so is its information
                for t, role in _G_AFFORD().get(k["op"], []):
                    if role == "tool" and t not in fit:
                        fit.append(t)
        rendered = render_request(_parse_request(q)) if (self._prec() or {}).get("request_ops") else None
        key_types = [n.split(" ")[0] for n in names] + fit
        if rendered:
            # G1: the rendering already names every typed object of the operations; the names line keeps only the rest (a stated
            # fact's object such as "Egg (stated: bad)")
            names = [n for n in names if not re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(n.split(" ")[0]), rendered)]
        if not names and not fit and not rendered:
            return None
        parts = []
        if rendered:
            parts.append("The request in English: %s." % rendered)
        if names:
            parts.append("This request names: %s." % ", ".join(names))
        if fit:
            parts.append("A tool that fits the request: %s." % " or ".join(fit))
        return {"schema": S.SCHEMA_REC, "id": "procedure/*/request-types", "component": "procedure", "kitchen": "*",
                "keys": {"types": key_types, "ops": [], "rels": []}, "lex": {"en": "", "fa": ""},
                "text": {"en": " ".join(parts), "fa": ""},
                "provenance": {"source": "parser:goal_schema", "evidence": ["intent.key"] + (["render:request_ops"] if rendered else []),
                               "authored_by": "script:ltm_package.request_line",
                               "support": []},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": -1}

    def _procedure_with_afford(self, kb, query: dict) -> list:
        """Round 1 (I-round1-integrate): the procedure candidates for a query that carries a goal-schema `intent`.
        1. The types the request's words make likely (intent.affordance_types, e.g. Sink for "where they get washed", Fridge for "cold")
           count as FIT types for the matching only. They are not added to the query and never rendered as facts.
        2. Each operation word of the request (and a state word that implies one, via_state) is looked up in the English affordance map.
           A word with a recipe makes that recipe a direct match; a word without one (flip, spread, scoop) brings its affordance line.
        A negated item is skipped. Nothing here reads a reference plan, a level or an item id."""
        intent = query.get("intent") or {}
        q2 = dict(query)
        q2["types"] = sorted(set(query.get("types") or []) | {t for t in intent.get("affordance_types") or [] if _is_type(t)})
        cands = list(kb.candidates(q2, include_zero=True))
        by_id = {c["rec"]["id"]: i for i, c in enumerate(cands)}
        recs = None
        seen = set()
        for k in ([] if self._need_active else intent.get("key") or []):      # K-round6: no affordance route for a NEED presentation
            if k.get("neg"):
                continue
            for w in (k.get("op"), k.get("via_state")) + (_state_afford_words(kb, k) if (self._prec() or {}).get("state_afford") else ()):
                key = kb.afford_key(w) if w else None
                if not key or key in seen:
                    continue
                seen.add(key)
                a = kb.afford(w)
                # a purpose-only request ("something to cut with": need=tool, no object) wants the TOOL line, not the
                # recipe that applies the operation to food (review I-close, finding 2)
                tool_only = (k.get("need") == "tool" and not k.get("obj")) or is_tool_request(query)
                if a and a.get("request") and tool_only:
                    if recs is None:
                        recs = {r["id"]: r for r in kb.afford_records()}
                    r = recs.get("procedure/*/afford-" + key + "-request")
                    if r is not None and r["id"] not in by_id:
                        cands.append({"schema": S.SCHEMA_CAND, "rec": r, "match": 1.0, "why": ["afford:" + key, "tool-request"]})
                        by_id[r["id"]] = len(cands) - 1
                elif a and a.get("recipe") and not tool_only and (("procedure/*/" + a["recipe"].split(":", 1)[1]) in by_id
                                                                   or not (self._prec() or {}).get("state_afford")):
                    rid = "procedure/*/" + a["recipe"].split(":", 1)[1]
                    i = by_id.get(rid)
                    if i is not None and float(cands[i]["match"]) < 1.0:
                        cands[i] = dict(cands[i], match=1.0, why=list(cands[i].get("why") or []) + ["afford:" + key])
                elif a:
                    # L-round3 F2: also when the row's recipe is not among the candidates (left out for this item): the row's own,
                    # separately authored line is eligible unless the same leave-out drops it too (afford_records is filtered)
                    if recs is None:
                        recs = {r["id"]: r for r in kb.afford_records()}
                    r = recs.get("procedure/*/afford-" + key)
                    if r is not None and r["id"] not in by_id:
                        cands.append({"schema": S.SCHEMA_CAND, "rec": r, "match": 1.0, "why": ["afford:" + key]})
                        by_id[r["id"]] = len(cands) - 1
        cands.sort(key=lambda c: (-c["match"], c["rec"].get("prio", 0), c["rec"]["id"]))
        return cands

    def _dense(self, query: dict, cands: list) -> list:
        if self.embedder is None or not cands:
            return [0.0] * len(cands)
        if callable(self.embedder) and not hasattr(self.embedder, "embed"):
            return [max(0.0, min(1.0, float(x))) for x in self.embedder(query, cands)]
        texts = [query.get("instruction", "")] + [(_rec_text(c["rec"]) + " " + (c["rec"].get("lex") or {}).get("en", "")).strip() for c in cands]
        vecs, need, idx = [None] * len(texts), [], []
        for i, t in enumerate(texts):
            k = S.sha256_text(t)
            if k in self.cache:
                vecs[i] = self.cache[k]
            else:
                need.append(t)
                idx.append(i)
        if need:
            for i, v in zip(idx, self.embedder.embed(need)):
                vecs[i] = list(v)
                self.cache[S.sha256_text(texts[i])] = list(v)
        q = vecs[0]
        out = []
        for v in vecs[1:]:
            dot = sum(a * b for a, b in zip(q, v))
            na, nb = math.sqrt(sum(a * a for a in q)) or 1.0, math.sqrt(sum(b * b for b in v)) or 1.0
            out.append(max(0.0, min(1.0, dot / (na * nb))))
        return out

    def _rank(self, section: str, cands: list, query: dict) -> tuple:
        """-> (ranked [(cand, score)], dropped [{id, component, reason}], log rows). Order: score desc, prio asc, id asc."""
        w, disc = self.params["weights"], float(self.params["need_discount"])
        mn = float(self.params["irrelevant"]["min_dense_without_key"])
        qtypes = set(query.get("types") or [])
        docs = [" ".join(list((c["rec"].get("keys") or {}).get("types") or []) + list((c["rec"].get("keys") or {}).get("ops") or [])
                         + [(c["rec"].get("lex") or {}).get("en", "")]) for c in cands]
        qw = _words(" ".join(list(query.get("types") or []) + list(query.get("ops") or [])))
        lex = bm25_norm(docs, qw)
        dense = self._dense(query, cands)
        ranked, dropped, rows = [], [], []
        for c, lx, dn in zip(cands, lex, dense):
            r = c["rec"]
            m = float(c.get("match", 0.0))
            if not _rec_text(r):                           # nothing to render (e.g. a crashed practice attempt): never a candidate
                dropped.append({"id": r["id"], "component": r["component"], "reason": "irrelevant"})
                continue
            wt = _why_types(c.get("why"))
            if m > 0 and wt and not (set(wt) & qtypes):
                m *= disc                                  # reached only through the expansion: ranked below a direct mention
            if float(c.get("match", 0.0)) <= float(self.params["irrelevant"]["min_match"]) and dn < mn:
                dropped.append({"id": r["id"], "component": r["component"], "reason": "irrelevant"})
                continue
            sc = w["exact"] * m + w["lexical"] * lx + w["dense"] * dn
            ranked.append((c, sc))
        ranked.sort(key=lambda cs: (-round(cs[1], 12), cs[0]["rec"].get("prio", 0), cs[0]["rec"]["id"]))
        rows = [{"id": c["rec"]["id"], "component": c["rec"]["component"], "match": round(float(c.get("match", 0.0)), 6), "score": round(sc, 6)}
                for c, sc in ranked]                                     # in rank order, so arrival order never reaches the log
        return ranked, dropped, rows

    # -- precision (I-F3) --------------------------------------------------------------------------------------
    def _prec(self):
        pr = self.params.get("precision")
        return pr if isinstance(pr, dict) and pr.get("enabled") else None

    def _kitchen_line(self, recs, q, pr):
        """ONE natural sentence about the fixtures the request needs (named in the instruction, then the appliances / receptacles the
        expansion reached). Compass lines, counts of unrelated fixtures, adjacency and storage habits are not rendered here."""
        try:
            import kitchen_briefing as KB_
        except ImportError:                                           # pragma: no cover
            from sim import kitchen_briefing as KB_
        fx = set(self.categories.get("fixture_types") or [])
        need = q.get("need") or {}
        order = []
        for grp in (q.get("types") or [], need.get("appliances") or [], need.get("receptacles") or []):
            for t in grp:
                if t in fx and t not in order:
                    order.append(t)
        facts = KB_.fixture_facts(recs)
        order = [t for t in order if (facts.get(t) or {}).get("ops")][: int(pr.get("max_fixtures", 4))]    # a bare count never changes a plan
        text = KB_.natural_fixture_line(facts, order, with_near=bool(pr.get("show_near")), prefix="")
        if not text:
            return None
        src_ids = sorted({i for t in order for i in (facts.get(t) or {}).get("ids", [])})
        base = next((r for r in recs if r["id"] in src_ids), recs[0])
        return {"schema": S.SCHEMA_REC, "id": "briefing/%s/fixtures" % self.kitchen, "component": "briefing", "kitchen": self.kitchen,
                "keys": {"types": order, "ops": [], "rels": []}, "lex": {"en": " ".join(order).lower(), "fa": ""},
                "text": {"en": text, "fa": ""},
                "provenance": {"source": (base.get("provenance") or {}).get("source", "briefing"), "evidence": src_ids[:6],
                               "authored_by": "script:kitchen_briefing.natural_fixture_line", "support": []},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": True, "prio": 1, "kind": "fixtures"}

    def _hold_rec(self, held):
        return {"schema": S.SCHEMA_REC, "id": "procedure/*/holding-rule", "component": "procedure", "kitchen": "*",
                "keys": {"types": [held], "ops": ["put", "put_in"], "rels": []}, "lex": {"en": "holding put_in", "fa": ""},
                "text": {"en": "You are holding the %s, so your hand is full: do not pick anything up. To place it, use put_in "
                               "with the destination as the object." % held, "fa": ""},
                "provenance": {"source": "contract:honest_skills.put_in", "evidence": ["holding:" + held],
                               "authored_by": "script:ltm_package.holding_rule", "support": []},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": 0}

    def _drop_subsumed(self, recipes: list, drop) -> list:
        """A recipe whose route is the leading part of a longer matched recipe's route says nothing new (put-in-and-close inside
        run-the-microwave). The longer one stays, the shorter is dropped as a duplicate. Needs the procedure KB's routes; without
        them (a synthetic source) nothing is dropped."""
        kb = self.sources.get("procedure")
        by_id = getattr(kb, "by_id", None)
        if not by_id or len(recipes) < 2:
            return recipes

        def route(c):
            for ev in (c["rec"].get("provenance") or {}).get("evidence") or []:
                e = by_id.get(ev)
                if e and e.get("route"):
                    return [(st.get("action"), st.get("object"), st.get("to")) for st in e["route"]]
            return None
        routes = [(c, route(c)) for c in recipes]
        keep = []
        for c, rt in routes:
            sub = rt is not None and any(o is not c and ro is not None and len(ro) > len(rt) and ro[: len(rt)] == rt
                                         and float(o["match"]) >= float(c["match"]) for o, ro in routes)
            if sub:
                drop(c, "procedure", "subsumed_by_longer_recipe", "duplicate")
            else:
                keep.append(c)
        return keep

    def _precision(self, raw: dict, q: dict, plog: list) -> tuple:
        """Precision pass over the raw candidates. -> (raw', dropped[{id,component,reason}]). Every drop and rewrite is logged in plog
        ({id, why}). Keeps only what can change the plan: the one kitchen line, 0-2 operation recipes, the tool lines for the operation
        words that no recipe already states, the holding rule, and (only for a search) storage habits and learned places."""
        pr = self._prec()
        held, direct, ops = q.get("holding"), set(q.get("types") or []), set(q.get("ops") or [])
        search = needs_search(q)
        dropped = []
        fixtures = set(self.categories.get("fixture_types") or [])

        def drop(c, comp, why, reason="irrelevant"):
            dropped.append({"id": c["rec"]["id"], "component": comp, "reason": reason})
            plog.append({"id": c["rec"]["id"], "why": why})
        out = {k: list(v) for k, v in raw.items()}
        # -- learned places / profile: only for a search; a places answer replaces both the profile line and the generic habit
        answered = {t for c in raw["places"] for t in (c["rec"].get("keys") or {}).get("types", [])[1:]} if search else set()
        for comp in ("places", "profile"):
            keep = []
            for c in raw[comp]:
                ctypes = set((c["rec"].get("keys") or {}).get("types") or [])
                if pr.get("places") == "search_only" and not search:
                    drop(c, comp, "learned_place_without_search")
                elif comp == "profile" and not (ctypes & direct) and not re.search(
                        r"objects seen there: [^;]*\b(%s)\b" % "|".join(map(re.escape, sorted(direct)) or ["$^"]), _rec_text(c["rec"])):
                    drop(c, comp, "profile_line_about_other_types")
                elif comp == "profile" and answered & direct:
                    drop(c, comp, "answered_by_places")
                else:
                    keep.append(c)
            out[comp] = keep
        # -- briefing: one natural kitchen line; a storage habit only for a search, only if it does not fight the destination and
        #    only if practice has not already answered where the object is
        recs = self.sources.get("briefing")
        recs = recs if isinstance(recs, (list, tuple)) else (recs.records() if recs is not None else [])
        br = []
        for c in raw["briefing"]:
            kind = c["rec"].get("kind") or str(c["rec"]["id"]).split("/")[-1].split(":", 1)[0]
            if kind == "conv":
                why = contradiction(c["rec"], q, self.categories)
                members = {t for t in (c["rec"].get("keys") or {}).get("types", []) if t not in fixtures}
                if why:
                    drop(c, "briefing", why)
                elif pr.get("storage") == "search_only" and not search:
                    drop(c, "briefing", "storage_habit_without_search")
                elif members & answered:
                    drop(c, "briefing", "superseded_by_practice")
                else:
                    br.append(c)
            else:
                drop(c, "briefing", "folded_into_kitchen_line" if kind in ("core", "exist", "open", "adj", "place") else "irrelevant")
        line = self._kitchen_line(recs, q, pr) if recs else None
        if line:
            br.insert(0, {"schema": S.SCHEMA_CAND, "rec": line, "match": 1.0, "why": ["type:" + (line["keys"]["types"] or [""])[0]]})
        out["briefing"] = br[:2]
        for c in br[2:]:
            drop(c, "briefing", "briefing_cap", "quota")
        # -- procedure: recipes, then the tool lines no recipe already states; holding-aware
        recipes, tools = [], []
        has_obj = bool(held) or bool(q.get("goal_states")) or any(t not in fixtures for t in direct)
        treq = is_tool_request(q)
        rcat = set((self.categories.get("categories", self.categories).get("receptacle")) or ())
        dests_named = {t for t in direct if t in rcat and t != "Microwave"}
        pol0 = _polarity(q.get("instruction", ""))
        kbp = self.sources.get("procedure")
        by_id = getattr(kbp, "by_id", None) or {}
        closure = {}                                                    # G2: tool verb -> note, for a requested op that needs it
        if pr.get("precondition_closure"):
            for k in (q.get("intent") or {}).get("key") or []:
                pc = _PRECONDITIONS.get((k.get("op"), k.get("obj")))
                if pc and not k.get("neg") and k.get("mood") != "desire":
                    closure.setdefault(pc[0], pc[1])
        for c in raw["procedure"]:
            r = c["rec"]
            why = contradiction(r, q, self.categories)
            if why:
                drop(c, "procedure", why)
                continue
            if float(c.get("match", 0)) < 0.99 and not (_is_tool_rec(r) and _rec_verb(r) in closure):
                drop(c, "procedure", "not_a_direct_match")
                continue
            if treq and not str(r.get("id", "")).endswith("-request") and not _is_tool_rec(r) and _MUTATING.search(_rec_text(r)):
                drop(c, "procedure", "tool_request_asks_for_the_tool_only")      # a recipe that performs the operation is not wanted
                continue
            ent = next((by_id[ev] for ev in (r.get("provenance") or {}).get("evidence") or [] if ev in by_id), None)
            if ent is not None and ent.get("polarity") and pol0 and pol0 != ent["polarity"]:
                drop(c, "procedure", "wrong_polarity")                           # 'turn the microwave off' is not the run-it route
                continue
            if ent is not None and ent.get("needs_dest") and not (stated_destinations(q, self.categories) or dests_named):
                drop(c, "procedure", "recipe_needs_a_destination_the_request_does_not_name")      # 'heat the egg' is heat, not heat-then-place
                continue
            if ent is not None and ent.get("composite") and (ent.get("applies_to") or {}).get("categories"):
                cats_ = {m for cn in ent["applies_to"]["categories"] for m in (list((getattr(kbp, "categories", None) or {}).get(cn) or ()) + list(self.categories.get("categories", self.categories).get(cn) or ()))}
                objs = {t for t in direct if t not in fixtures}
                if objs and cats_ and not ((objs | ({held} if held else set())) & cats_):
                    drop(c, "procedure", "recipe_for_another_kind_of_object")      # a pot-on-the-stove line for a potato
                    continue
            if _is_tool_rec(r):
                tools.append(c)
            else:
                txt = _rec_text(r)
                if ("pick_and_place" in txt or "put_in" in txt) and not has_obj:
                    drop(c, "procedure", "recipe_moves_an_object_the_request_does_not_name")
                    continue
                recipes.append(c)
        recipes.sort(key=lambda c: (-float(c["match"]), c["rec"].get("prio", 0), c["rec"]["id"]))
        rmax = int(pr.get("recipe_max", 2))
        recipes = self._drop_subsumed(recipes, drop)
        for c in recipes[rmax:]:
            drop(c, "procedure", "recipe_cap", "quota")
        recipes = recipes[:rmax]
        covered = set()
        for c in recipes:
            covered |= set(_VERBS.findall(_rec_text(c["rec"])))
        if "close" in covered:                                          # close_all is the less specific sibling of a stated close
            covered.add("close_all")
        pol = _polarity(q.get("instruction", ""))
        # siblings: of the tool lines that share an operation word keep the most specific one (close over close_all)
        kept_tools = []
        for c in sorted(tools, key=lambda c: (len((c["rec"].get("keys") or {}).get("ops") or []), c["rec"]["id"])):
            r = c["rec"]
            verb = _rec_verb(r)
            rops = set((r.get("keys") or {}).get("ops") or [])
            if treq and verb not in ("pickup", "go_to", "search_for"):
                drop(c, "procedure", "tool_request_needs_only_the_pickup")
            elif verb == "put_in" and not held:
                drop(c, "procedure", "needs_a_held_object")
            elif verb in covered:
                drop(c, "procedure", "already_in_recipe", "duplicate")
            elif not (rops & ops) and verb not in closure:
                drop(c, "procedure", "operation_not_asked")
            elif (verb.endswith("_on") and pol == "off") or (verb.endswith("_off") and pol == "on"):
                drop(c, "procedure", "wrong_polarity")
            elif any(rops > set((k["rec"].get("keys") or {}).get("ops") or []) and (rops & ops) <= set((k["rec"].get("keys") or {}).get("ops") or [])
                     for k in kept_tools):
                drop(c, "procedure", "less_specific_sibling", "duplicate")
            else:
                if not (rops & ops) and verb in closure:
                    r2 = dict(r, text={"en": _rec_text(r) + " " + closure[verb], "fa": ""})
                    r2["provenance"] = dict(r.get("provenance") or {}, evidence=list((r.get("provenance") or {}).get("evidence") or [])
                                            + ["precondition_closure:" + verb])
                    plog.append({"id": r["id"], "why": "precondition_of_a_requested_operation"})
                    c = dict(c, rec=r2, match=1.0, why=list(c.get("why") or []) + ["precondition_closure"])
                kept_tools.append(c)
        if held and pr.get("hold_rewrite"):
            rew = []
            for c in recipes + kept_tools:
                t2 = holding_rewrite(_rec_text(c["rec"]), held)
                if t2 != _rec_text(c["rec"]):
                    r2 = dict(c["rec"], text={"en": t2, "fa": ""})
                    r2["provenance"] = dict(c["rec"]["provenance"], evidence=list(c["rec"]["provenance"].get("evidence") or []) + ["rewrite:holding"])
                    plog.append({"id": c["rec"]["id"], "why": "rewritten_for_holding"})
                    c = dict(c, rec=r2)
                rew.append(c)
            recipes, kept_tools = rew[: len(recipes)], rew[len(recipes):]
        proc = []
        if held and pr.get("holding_rule") and (PUT_FAMILY & ops or stated_destinations(q, self.categories) or not ops):
            proc.append({"schema": S.SCHEMA_CAND, "rec": self._hold_rec(held), "match": 1.0, "why": ["type:" + held]})
        proc += recipes + kept_tools
        seen, uniq = set(), []
        for c in proc:                                                  # dedup on the rendered words
            k = _norm_text(_rec_text(c["rec"]))
            if k in seen:
                drop(c, "procedure", "same_text", "duplicate")
            else:
                seen.add(k)
                uniq.append(c)
        pmax = int(pr.get("procedure_max", 3))
        for c in uniq[pmax:]:
            drop(c, "procedure", "procedure_cap", "quota")
        out["procedure"] = uniq[:pmax]
        rq = self._request_rec(q)                                       # F1: outside the procedure cap; always first
        if rq is not None:
            out["procedure"] = [{"schema": S.SCHEMA_CAND, "rec": rq, "match": 1.0, "why": ["parser:intent"]}] + out["procedure"]
        # -- examples / templates: only a record that shares an object AND an operation with the request
        keep = []
        for c in raw["example"]:
            k = c["rec"].get("keys") or {}
            if (set(k.get("types") or []) & direct) and (set(k.get("ops") or []) & ops):
                keep.append(c)
            else:
                drop(c, "example", "example_does_not_share_object_and_operation")
        out["example"] = keep
        keep = []
        for c in raw["template"]:
            if float(c.get("match", 0)) >= 0.99:
                keep.append(c)
            else:
                drop(c, "template", "template_not_a_direct_match")
        out["template"] = keep
        return out, dropped

    # -- the uncertainty section ----------------------------------------------------------------------------------
    def _uncertainty(self, query: dict, profile_known: set, have_profile: bool) -> list:
        tx = self.params["uncertainty_text"]
        need = set(query.get("need", {}).get("types") or []) | set(query.get("types") or [])
        out = []

        def rec(local, text, ev):
            return {"schema": S.SCHEMA_REC, "id": "uncertainty/%s/%s" % (self.kitchen, local), "component": "uncertainty",
                    "kitchen": self.kitchen, "keys": {"types": [], "ops": [], "rels": []}, "lex": {"en": "", "fa": ""},
                    "text": {"en": text, "fa": ""},
                    "provenance": {"source": "spec:package_params", "evidence": ev, "authored_by": "script:ltm_package", "support": []},
                    "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": len(out)}
        if need & {"StoveKnob", "StoveBurner"}:
            prof = self.sources.get("profile")
            km = getattr(prof, "knob_map", "not_learned") if prof is not None else "not_learned"
            if not km or km == "not_learned":
                out.append(rec("knob_map", tx["knob_map"], ["knob_map:not_learned"]))
        if have_profile:
            fixture = set(self.categories.get("fixture_types") or [])
            unseen = sorted(t for t in (query.get("types") or []) if t not in fixture and t not in profile_known)
            if unseen:
                out.append(rec("never_seen", tx["never_seen"].format(types=", ".join(unseen)), ["never_seen:" + ",".join(unseen)]))
        if query.get("parser_fallback"):
            out.append(rec("parser", tx["parser"], ["parser:not_ok"]))
        return [{"schema": S.SCHEMA_CAND, "rec": r, "match": 1.0, "why": []} for r in out]

    # -- build ---------------------------------------------------------------------------------------------------
    def build_text(self, instruction, lang, holding, summary_lines, lexicon=None) -> dict:
        return self.build(make_query(instruction, lang, holding, summary_lines, lexicon))

    def build(self, query: dict) -> dict:
        """pkg.pkg/1. No level parameter, by design. The query's `need` is recomputed here, so the result never depends on a caller's
        stale expansion."""
        _query_check(query)
        probs = S.validate_query(query)
        if probs:
            raise ValueError("bad package query: %s" % probs)
        P = self.params
        errors = []
        prec = self._prec()
        bound = None
        # K-round6 need rule: the trigger reads the instruction text and its own parse only; a NEED presentation loses the word -> type
        # hints before the expansion (strip_need_hints), and gets the inventory + uses bundle after the budget pass
        need = None
        self._need_active = False
        if prec and prec.get("need_rule"):
            try:
                import goal_schema as GS_
            except ImportError:                                       # pragma: no cover
                from sim import goal_schema as GS_
            if "need_hand" in prec:                           # pkg-1.r6b: same verdict + the placeholder kind (thing / place)
                need = GS_.need_trigger_r6b(query.get("instruction") or "", query.get("lang"))
            else:
                need = GS_.need_trigger(query.get("instruction") or "", query.get("lang"))
            if need["fires"]:
                self._need_active = True
                query = strip_need_hints(query)
        if prec:
            bound = dict(prec.get("expand_cap") or {}, profile_receptacles=needs_search(query))
        q = expand(query, self.sources.get("procedure"), self.categories,
                   self.sources.get("briefing") if isinstance(self.sources.get("briefing"), (list, tuple)) else (), self.sources.get("profile"),
                   bound=bound)
        raw = {c: self._source_cands(c, q, errors) for c in ("briefing", "profile", "places", "procedure", "example", "template")}
        plog, pdrop = [], []
        if prec:
            raw, pdrop = self._precision(raw, q, plog)
        if prec and prec.get("type_scope"):                  # K-round6 D2: no record that names an excluded type reaches the planner
            TS_ = _type_scope()
            for comp in list(raw):
                keep = []
                for c in raw[comp]:
                    if TS_.mentions_excluded(_rec_text(c["rec"])):
                        pdrop.append({"id": c["rec"]["id"], "component": c["rec"].get("component", comp), "reason": TS_.REASON})
                        plog.append({"id": c["rec"]["id"], "why": TS_.REASON})
                    else:
                        keep.append(c)
                raw[comp] = keep
        if self._need_active:                                # K-round6 route audit: on a NEED presentation no retrieved record may
            allowed = set(q.get("types") or []) | ({q["holding"]} if q.get("holding") else set())     # recommend a type the request
            vocab = {t for t, _u in load_uses_table(uses_path(self.kitchen))}                          # does not name itself
            for comp in list(raw):
                keep = []
                for c in raw[comp]:
                    named = {m for m in re.findall(r"(?<![A-Za-z])([A-Z][A-Za-z]+)(?![A-Za-z])", _rec_text(c["rec"])) if m in vocab}
                    if named - allowed:
                        pdrop.append({"id": c["rec"]["id"], "component": c["rec"].get("component", comp), "reason": "need_rule:target_type_hint"})
                        plog.append({"id": c["rec"]["id"], "why": "need_rule:target_type_hint"})
                    else:
                        keep.append(c)
                raw[comp] = keep
        # the profile section carries the places records; a fact a profile record already states is kept from the profile
        pfacts = set()
        for c in raw["profile"]:
            pfacts |= _facts(c["rec"])
        dup = []
        if pfacts:
            for comp in ("places", "briefing"):
                keep = []
                for c in raw[comp]:
                    f = _facts(c["rec"])
                    if f and f <= pfacts:
                        dup.append({"id": c["rec"]["id"], "component": comp, "reason": "duplicate"})
                    else:
                        keep.append(c)
                raw[comp] = keep
        groups = {"briefing": raw["briefing"], "profile": raw["profile"] + raw["places"], "procedure": raw["procedure"],
                  "example": raw["example"], "template": raw["template"]}
        dropped, log_rows, ranked_by = list(dup) + list(pdrop), [], {}
        irrelevant = {}
        for sec, cs in groups.items():
            rk, dr, rows = self._rank(sec, cs, q)
            ranked_by[sec] = [c for c, _s in rk]
            if prec and sec == "procedure":                 # holding rule first, then the operation recipes, then the tool lines
                ranked_by[sec].sort(key=lambda c: (-1 if c["rec"]["id"].endswith("/request-types") else 0 if c["rec"]["id"].endswith("/holding-rule") else (2 if _is_tool_rec(c["rec"]) else 1)))
            log_rows += rows
            irrelevant[sec] = len(dr)
            if P.get("log_irrelevant_ids"):
                dropped += dr
        # caps (k_max): beyond the cap = quota
        kmax = dict(P.get("k_max") or {})
        if prec:
            kmax.update(example=int(prec.get("example_max", 1)), template=int(prec.get("template_max", 1)))
        for sec, k in kmax.items():
            lst = ranked_by.get(sec, [])
            for c in lst[int(k):]:
                dropped.append({"id": c["rec"]["id"], "component": sec, "reason": "quota"})
            ranked_by[sec] = lst[: int(k)]
        profile_known = set()
        pr = self.sources.get("profile")
        if pr is not None:
            for r in pr.records():
                profile_known |= set((r.get("keys") or {}).get("types") or [])
        for c in raw["places"]:
            profile_known |= set((c["rec"].get("keys") or {}).get("types") or [])
        unc = self._uncertainty(q, profile_known, self.sources.get("profile") is not None)
        if prec:                                              # only a note that can change the plan: no 'never seen', no parser note
            keepu = set(prec.get("keep_uncertainty") or [])
            for c in unc:
                if c["rec"]["id"].rsplit("/", 1)[-1] not in keepu:
                    dropped.append({"id": c["rec"]["id"], "component": "uncertainty", "reason": "irrelevant"})
                    plog.append({"id": c["rec"]["id"], "why": "note_cannot_change_the_plan"})
            unc = [c for c in unc if c["rec"]["id"].rsplit("/", 1)[-1] in keepu]
        # the "command only partly understood" note annotates retrieved content; on its own it would be a package with nothing in it
        if not any(ranked_by.get(sec) for sec in SECTIONS if sec != "uncertainty"):
            unc = [c for c in unc if not c["rec"]["id"].endswith("/parser")]
        ranked_by["uncertainty"] = unc
        recs = {sec: [c["rec"] for c in ranked_by.get(sec, [])] for sec in SECTIONS}
        keep_n = {sec: 0 for sec in SECTIONS}
        quotas, budget = P["quotas"], int(P["budget"])
        # pass 1: per-section quota, the longest prefix of whole records (monotone, so binary search)
        for sec in SECTIONS:
            n = len(recs[sec])
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._max(self._section_text(sec, recs[sec][:mid])) <= int(quotas[sec]):
                    lo = mid
                else:
                    hi = mid - 1
            keep_n[sec] = lo

        def total_text(kn):
            return self._complete({s: self._section_text(s, recs[s][: kn[s]]) for s in SECTIONS})
        # pass 2: borrowing, once, in the fixed order: a section that dropped records may use what the total budget still has free
        for sec in P["borrow_order"]:
            n = len(recs[sec])
            lo, hi = keep_n[sec], n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                trial = dict(keep_n)
                trial[sec] = mid
                if self._max(total_text(trial)) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            keep_n[sec] = lo
        # pass 3: the complete text is the hard rule; drop from the end in reverse borrow order until it fits (token counts need not be additive)
        guard = 0
        while self._max(total_text(keep_n)) > budget:
            guard += 1
            for sec in reversed(P["borrow_order"]):
                if keep_n[sec] > 0:
                    keep_n[sec] -= 1
                    break
            else:                                              # pragma: no cover
                break
            if guard > 500:                                    # pragma: no cover
                raise RuntimeError("package fit did not converge")
        sections_out, kept, prov = [], [], []
        for sec in SECTIONS:
            for r in recs[sec][keep_n[sec]:]:
                if r["id"] not in {d["id"] for d in dropped}:
                    dropped.append({"id": r["id"], "component": r["component"], "reason": "budget"})
            ks = recs[sec][: keep_n[sec]]
            for r in ks:
                kept.append(r["id"])
                pv = r["provenance"]
                prov.append({"id": r["id"], "component": r["component"], "source": pv.get("source"), "authored_by": pv.get("authored_by"),
                             "evidence": list(pv.get("evidence") or [])[:6], "confidence": r.get("confidence"), "recency": r.get("recency"),
                             "privileged": bool(r.get("privileged"))})
            st = self._section_text(sec, ks)
            if ks:
                sections_out.append({"component": sec, "ids": [r["id"] for r in ks], "quota": int(quotas[sec]),
                                     "used": {"en": self._counts(st)}})
        body = "\n".join(t for t in (self._section_text(s, recs[s][: keep_n[s]]) for s in SECTIONS) if t)
        retrieved_max = self._max((self.header + "\n" + body) if body else "")
        bundle, thing = None, False
        if self._need_active:
            # outside the retrieval budget by design: all permitted rows, fixed order, no truncation (Astra round6-need-rule)
            TS_ = _type_scope()
            inv = load_inventory(self.kitchen, inventory_path(self.kitchen))
            seen = sorted({o["type"] for o in q.get("observed") or [] if TS_.in_scope(o.get("type"))})
            thing = bool(prec.get("need_hand")) and need.get("kind") == "placeholder" and need.get("placeholder") == "thing"
            bundle = need_bundle_text(inv, load_uses_table(uses_path(self.kitchen)), seen, NEED_THING_INSTRUCTION if thing else NEED_INSTRUCTION) if inv else None
            if bundle:
                body = (body + "\n" + bundle) if body else bundle
                for rid, comp_, src_, auth_, priv in (
                        ("briefing/%s/homeowner-inventory" % self.kitchen, "briefing", "briefing:homeowner_inventory",
                         "file:" + _rel_step5(inventory_path(self.kitchen)), True),
                        ("procedure/*/uses-table", "procedure", "table:" + _rel_step5(uses_path(self.kitchen)),
                         "file:" + _rel_step5(uses_path(self.kitchen)), False),
                        (("procedure/*/need-instruction", "procedure", "amendment:R6b-need_hand", "script:ltm_package.need_hand", False) if thing
                         else ("procedure/*/need-instruction", "procedure", "amendment:R6-D1", "script:ltm_package.need_rule", False))):
                    kept.append(rid)
                    prov.append({"id": rid, "component": comp_, "source": src_, "authored_by": auth_,
                                 "evidence": ["need_trigger:" + str(need.get("kind")), "inventory:%d" % len(inv)][:6], "confidence": 1.0,
                                 "recency": 0, "privileged": priv})
                sections_out.append({"component": "need_bundle", "ids": ["briefing/%s/homeowner-inventory" % self.kitchen,
                                                                         "procedure/*/uses-table", "procedure/*/need-instruction"],
                                     "quota": 0, "used": {"en": self._counts(bundle)}})
            else:
                plog.append({"id": "briefing/%s/homeowner-inventory" % self.kitchen, "why": "no_homeowner_inventory_for_kitchen"})
        full = (self.header + "\n" + body) if body else ""
        cnt = self._counts(full) if full else {}
        n_max = max(cnt.values()) if cnt else 0
        parser_ok = not q.get("parser_fallback")
        fb_used = bool(q.get("parser_fallback")) and any("fallback:core" in (c.get("why") or []) for c in raw["briefing"])
        reason = None
        intent = q.get("intent")
        # the goal schema's own verdict (F1 handoff): an `indirect` or `ok` reading is not a parser failure; `partial` / `unsupported` is
        schema_fb = bool(intent.get("fallback")) if isinstance(intent, dict) and prec else (not parser_ok)
        if not body:
            reason = "parser_failed_no_match" if schema_fb else "parsed_zero_matches"
        dropped.sort(key=lambda d: (d["component"], d["id"], d["reason"]))
        pk = {"schema": S.SCHEMA_PKG, "version": VERSION, "kitchen": self.kitchen, "lang": q["lang"], "rendered_lang": "en",
              "text": body, "text_en": body, "sections": sections_out, "kept": kept, "dropped": dropped,
              "irrelevant": irrelevant,
              "tokens": {"en": cnt, "max": n_max, "budget": budget},
              "empty": not body, "empty_reason": reason, "parser_ok": parser_ok, "fallback_used": fb_used,
              "query_sha": S.sha256_text(S.canon(q)), "params_sha": params_sha(P),
              "store_shas": self.store_shas(), "provenance": prov, "candidates": log_rows,
              "need": q["need"]}
        if errors:
            pk["source_errors"] = errors
        if prec and prec.get("need_rule") is not None:
            pk["need_rule"] = {"trigger": need, "active": self._need_active, "bundle": bool(bundle)}
            if "need_hand" in prec:                           # pkg-1.r6b only (r6 rows keep their bytes)
                pk["need_rule"]["instruction"] = ("thing" if thing else "r6") if bundle else None
            pk["tokens"]["retrieved_max"] = retrieved_max
            pk["tokens"]["need_bundle"] = self._max(bundle) if bundle else 0
            pk["need_audit"] = need_hint_audit(pk, q) if self._need_active else []
        if prec:
            pk["precision"] = {"log": sorted(plog, key=lambda d: (d["id"], d["why"])), "needs_search": needs_search(q),
                               "unsupported_request": bool(pk["empty"] and (schema_fb if isinstance(intent, dict) else q.get("parser_fallback"))
                                                      and not q.get("types") and not q.get("ops"))}
        pk["sha256"] = S.self_sha(pk)
        return pk

    def store_shas(self) -> dict:
        out = {}
        for k, s in sorted(self.sources.items()):
            if s is None:
                continue
            if isinstance(s, (list, tuple)):
                out[k] = S.sha256_text(S.canon(sorted(s, key=lambda r: r.get("id", ""))))[:16]
                continue
            for attr in ("sha", "sha256"):
                v = getattr(s, attr, None)
                v = v() if callable(v) else v
                if isinstance(v, str) and v:
                    out[k] = v[:16]
                    break
            else:
                try:
                    out[k] = S.sha256_text(S.canon(sorted(s.records(), key=lambda r: r.get("id", ""))))[:16]
                except Exception:
                    out[k] = "n/a"
        return out


def need_hint_audit(pkg: dict, q: dict) -> list:
    """K-round6 route audit for a NEED presentation: every type name rendered OUTSIDE the need bundle must be one the request itself
    names (or the held object). A type outside that set would be a leftover target-type recommendation. -> list of offending types."""
    text = pkg.get("text") or ""
    i = text.find("Kitchen inventory: the homeowner reports")
    outside = text[:i] if i >= 0 else text
    vocab = {t for t, _u in load_uses_table(uses_path(pkg.get("kitchen") or DEFAULT_KITCHEN))} | set(_type_scope().EXCLUDED_TYPES)
    allowed = set(q.get("types") or []) | ({q["holding"]} if q.get("holding") else set())
    found = {m for m in re.findall(r"(?<![A-Za-z])([A-Z][A-Za-z]+)(?![A-Za-z])", outside) if m in vocab}
    return sorted(found - allowed)


# ------------------------------------------------------------------------------------------------------- logging, compat
def log_row(pkg: dict, pres: str, query: dict) -> dict:
    """One pkg.log/1 row per presentation (appended by the caller to docs/final-test/matrix/<ns>/package.jsonl). No level, no item
    outcome; the candidate list is the ranked candidates with their match and score."""
    return {"schema": S.SCHEMA_LOG, "pres": pres, "query": {"instruction_sha": S.sha256_text(query.get("instruction", ""))[:16],
                                                          "lang": query.get("lang"), "types": query.get("types"), "ops": query.get("ops"),
                                                          "parser_ok": not query.get("parser_fallback")},
            "need": pkg.get("need"), "candidates": pkg.get("candidates"), "kept": pkg.get("kept"), "dropped": pkg.get("dropped"),
            "irrelevant": pkg.get("irrelevant"), "tokens": pkg.get("tokens"), "empty": pkg.get("empty"),
            "empty_reason": pkg.get("empty_reason"), "package_sha": pkg.get("sha256")}


def audit_provenance(pkg: dict) -> list:
    """Problems: every kept record needs a listed provenance with a source and an author; stores that observe (profile, places,
    example) also need evidence. Empty list = every rendered fact is attributable."""
    out = []
    byid = {p["id"]: p for p in pkg.get("provenance") or []}
    for rid in pkg.get("kept") or []:
        p = byid.get(rid)
        if not p or not p.get("source") or not p.get("authored_by"):
            out.append("no provenance for %s" % rid)
        elif p["component"] in ("profile", "places", "example") and not p.get("evidence"):
            out.append("no evidence for %s" % rid)
    return out


def memory_block_compat(pkg: dict, level=None) -> dict:
    """The package shaped like `LongTermMemory.memory_block`'s result, so the planner path (`memory_block=` of the contract) is unchanged."""
    recs = []
    kept = set(pkg.get("kept") or [])
    comp = {p["id"]: p["component"] for p in pkg.get("provenance") or []}
    text_of = {}
    for s in pkg.get("sections") or []:
        for rid in s["ids"]:
            text_of[rid] = s["component"]
    for rid in pkg.get("kept") or []:
        recs.append({"id": rid, "section": comp.get(rid) or text_of.get(rid), "kind": "package", "otype": None})
    return {"text": pkg.get("text", ""), "records": recs,
            "dropped": [dict(d, tokens=0) for d in pkg.get("dropped") or [] if d["id"] not in kept],
            "tokens": int((pkg.get("tokens") or {}).get("max", 0)), "budget": int((pkg.get("tokens") or {}).get("budget", 0)),
            "types": [], "level": level, "lang": pkg.get("lang"), "channels": ["package"], "empty": bool(pkg.get("empty")),
            "joint": False, "token_claim": "real max over tokenizers, header included", "tokenizer": "max",
            "package_sha": pkg.get("sha256"), "package_version": pkg.get("version"), "rendered_lang": "en",
            "empty_reason": pkg.get("empty_reason"), "parser_ok": pkg.get("parser_ok"), "fallback_used": pkg.get("fallback_used")}


# ------------------------------------------------------------------------------------------------------ default stores
def _common():
    sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
    import common as C                                           # noqa: E402
    return C


def default_sources(kitchen: str, snapshot_db: str, practice_rows: list = None, briefing_dir: str = None, allow_unapproved: bool = True,
                    categories: dict = None, embedder=None) -> dict:
    """The stores built from EXISTING non-test practice data: briefing records (drafts allowed for dev), the profile (from the
    practice store), the places adapter, the procedure KB, the example index (from the practice rows) and the L3 templates.
    The practice store is copied to a temp file before the long-term memory opens it, so the frozen snapshot is never touched."""
    import shutil
    import tempfile
    cats = categories or load_categories()
    out = {}
    try:
        import kitchen_briefing as KB_
    except ImportError:                                           # pragma: no cover
        from sim import kitchen_briefing as KB_
    out["briefing"] = KB_.load_records(kitchen, allow_unapproved=allow_unapproved, out_dir=briefing_dir)
    try:
        import ltm_profile as LP
        import ltm_procedural as LPR
        import ltm_templates as LT
        import ltm_examples as LE
        import ltm as L
    except ImportError:                                           # pragma: no cover
        from sim import ltm_profile as LP, ltm_procedural as LPR, ltm_templates as LT, ltm_examples as LE, ltm as L
    out["profile"] = LP.Profile.from_store(snapshot_db, kitchen, cats)
    tmp = os.path.join(tempfile.mkdtemp(prefix="pkgdb_"), "ltm.db")
    shutil.copy(snapshot_db, tmp)
    out["_ltm"] = L.LongTermMemory(tmp, kitchen=kitchen)
    out["places"] = PlacesSource(out["_ltm"], None, ("places",), tool_types=cats.get("tool_types") or ())
    out["procedure"] = LPR.load_kb()
    out["template"] = LT.load()
    if practice_rows is not None:
        # B5's corpus audit refuses the substring "FloorPlan" anywhere in an example (ids and kitchen label included), so, like B5's own
        # early check, examples are built under the neutral practice label. The audit is NOT bypassed or edited; the example records
        # simply carry kitchen "practice-kitchen" (their namespace is already one kitchen). Reported in the ledger for the owner.
        out["example"] = LE.ExampleIndex(LE.build_corpus(practice_rows, EXAMPLE_LABEL), embedder=embedder, categories=cats["categories"])
    return out


def load_practice_rows(namespace: str = "practice-v1", order0_only: bool = True) -> list:
    p = os.path.join(ROOT, "docs", "final-test", "matrix", namespace, "practice.jsonl")
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return [r for r in rows if r.get("order") == 0] if order0_only else rows


def nontest_items() -> list:
    """The 116 non-test items ([{id, level, en, fa, plan, goal}]); the split guard is applied before anything is returned."""
    C = _common()
    m = C.load_manifest()
    ids = list(m["nontest"])
    C.guard_items(ids, "practice-v1")
    with open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8") as f:
        items = {i["id"]: i for i in json.load(f)["items"]}
    return [items[i] for i in ids if i in items]


# ---------------------------------------------------------------------------------------------------------- diagnostics
def leakage_ids(paths) -> dict:
    """Counts only: every record id, item and `support` entry in the given corpus files checked against the split guard."""
    C = _common()
    m = C.load_manifest()
    test_like = set(m["test"]) | set(m["test_uncapped_not_scored"])
    hits, n = 0, 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        rows = []
        try:
            d = json.loads(text)
            rows = d if isinstance(d, list) else [d]
        except ValueError:
            rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        for r in rows:
            n += 1
            # whole-token match: a test id may be a prefix of a non-test id ('g-l2-slice-bread' vs 'g-l2-slice-bread-tomato')
            if set(re.findall(r"[A-Za-z0-9_\-]+", json.dumps(r, ensure_ascii=False))) & test_like:
                hits += 1
    return {"files": len(paths), "rows": n, "test_id_hits": hits}


def needed_facts(item: dict) -> dict:
    """Diagnostic proxy only (never an input): the types and tool names of a non-test item's reference plan."""
    types, ops = set(), set()
    for st in item.get("plan") or []:
        for f in ("object", "to"):
            if _is_type(st.get(f)):
                types.add(st[f])
        if st.get("action"):
            ops.add(st["action"])
    return {"types": types, "ops": ops}


def covered(pkg: dict, facts: dict, recs_by_id: dict) -> tuple:
    """-> (covered types, covered ops) among the facts, judged by the kept records' keys and the package text."""
    ktypes, kops = set(), set()
    for rid in pkg.get("kept") or []:
        r = recs_by_id.get(rid)
        if r:
            ktypes |= set((r.get("keys") or {}).get("types") or [])
            kops |= set((r.get("keys") or {}).get("ops") or [])
    text = pkg.get("text", "")
    words = set(re.findall(r"[A-Za-z_]+", text))
    return (facts["types"] & (ktypes | words)), (facts["ops"] & (kops | words))


def context_fit(system: str, user_with_package: str, counter_gemma, repair_turns: int = 2, repair_text: str = "", out_tokens: int = 768,
                limit: int = 8192) -> dict:
    """A24 item 1: system + user (with the package) + `repair_turns` repair turns (each an assistant reply of up to `out_tokens` plus a
    repair prompt) + one more output <= limit, under the Gemma count. `counter_gemma(text) -> tokens`."""
    base = counter_gemma(system) + counter_gemma(user_with_package)
    per_turn = out_tokens + counter_gemma(repair_text)
    total = base + repair_turns * per_turn + out_tokens
    return {"system": counter_gemma(system), "user": counter_gemma(user_with_package), "repair_turns": repair_turns,
            "per_repair_turn": per_turn, "output": out_tokens, "total": total, "limit": limit, "fits": total <= limit}


def early_check(builder: PackageBuilder, items: list, recs_by_id: dict, summary_lines=()) -> dict:
    """PACKAGE-SPEC 9.6 on the non-test items, EN and FA. Diagnostic, outcome-blind, never an input to retrieval.

    Needed facts = the types and tool names of the item's reference plan (a proxy; the plan never reaches the builder). Two recalls are
    reported: over ALL needed types, and over the HIDDEN ones (needed types the instruction does not name, e.g. the Microwave for 'heat the
    apple'), which is what the package is for; the first is flattered by the instruction itself."""
    out = {"version": VERSION, "n_items": len(items), "dense": builder.embedder is not None, "langs": {}}
    pk_by, q_by = {}, {}
    for lang in ("en", "fa"):
        rows = []
        for it in items:
            text = it.get(lang)
            if not text:
                continue
            q = make_query(text, lang, None, summary_lines)
            pk = builder.build(q)
            pk_by[(it["id"], lang)], q_by[(it["id"], lang)] = pk, q
            f = needed_facts(it)
            ct, co = covered(pk, f, recs_by_id)
            hidden = f["types"] - set(q["types"])
            nk = len(pk["kept"])
            useful = 0
            for rid in pk["kept"]:
                r = recs_by_id.get(rid) or {}
                k = r.get("keys") or {}
                if (set(k.get("types") or []) & (f["types"])) or (set(k.get("ops") or []) & f["ops"]):
                    useful += 1
            rows.append({"id": it["id"], "level": it.get("level"), "kept": nk, "empty": pk["empty"], "parser_ok": pk["parser_ok"],
                         "tokens": pk["tokens"]["max"], "type_recall": (len(ct) / len(f["types"])) if f["types"] else None,
                         "hidden_n": len(hidden), "hidden_recall": (len(hidden & ct) / len(hidden)) if hidden else None,
                         "op_recall": (len(co) / len(f["ops"])) if f["ops"] else None, "precision": (useful / nk) if nk else None,
                         "irrelevant": sum(pk["irrelevant"].values()),
                         "comp_tokens": {s["component"]: max(s["used"]["en"].values()) for s in pk["sections"] if s["used"]["en"]},
                         "comp_tok": {s["component"]: dict(s["used"]["en"]) for s in pk["sections"] if s["used"]["en"]}})
        n = len(rows) or 1

        def mean(key, rows=rows):
            v = [r[key] for r in rows if r[key] is not None]
            return round(sum(v) / len(v), 4) if v else None
        out["langs"][lang] = {"n": len(rows), "empty_rate": round(sum(1 for r in rows if r["empty"]) / n, 4),
                              "empty_while_parsed": round(sum(1 for r in rows if r["empty"] and r["parser_ok"]) / n, 4),
                              "parser_failed_rate": round(sum(1 for r in rows if not r["parser_ok"]) / n, 4),
                              "type_recall": mean("type_recall"), "hidden_recall": mean("hidden_recall"),
                              "items_with_hidden_types": sum(1 for r in rows if r["hidden_n"]),
                              "op_recall": mean("op_recall"), "precision": mean("precision"),
                              "mean_irrelevant_dropped": round(sum(r["irrelevant"] for r in rows) / n, 2),
                              "mean_tokens": round(sum(r["tokens"] for r in rows) / n, 1), "max_tokens": max((r["tokens"] for r in rows), default=0),
                              "mean_component_tokens": _mean_comp(rows), "mean_component_tokens_by_tokenizer": _mean_comp_tok(rows), "rows": rows}
    ids_diff, by_cause = [], {"query_keys_differ": 0, "same_keys_different_ids": 0}
    for it in items:
        a, b = pk_by.get((it["id"], "en")), pk_by.get((it["id"], "fa"))
        if a and b and a["kept"] != b["kept"]:
            ids_diff.append(it["id"])
            qa, qb = q_by[(it["id"], "en")], q_by[(it["id"], "fa")]
            same = (qa["types"], qa["ops"], qa["parser_fallback"]) == (qb["types"], qb["ops"], qb["parser_fallback"])
            by_cause["same_keys_different_ids" if same else "query_keys_differ"] += 1
    both = sum(1 for it in items if (it["id"], "en") in pk_by and (it["id"], "fa") in pk_by)
    out["bilingual"] = {"pairs": both, "different_kept_ids": len(ids_diff), "agreement": round(1 - len(ids_diff) / both, 4) if both else None,
                        "cause": by_cause, "different_item_ids": ids_diff}
    out["over_budget"] = sum(1 for pk in pk_by.values() if pk["tokens"]["max"] > pk["tokens"]["budget"])
    out["provenance_problems"] = sum(len(audit_provenance(pk)) for pk in pk_by.values())
    out["packages"] = {"%s|%s" % k: v["sha256"] for k, v in sorted(pk_by.items())}
    return out


def real_token_gate(packages: list, counter, header: str = None) -> dict:
    """The real-token gate: every package's complete text (header + body) re-counted with `counter` (the real two-tokenizer counter in
    production). A package passes only if EVERY tokenizer's count is <= its budget. Reports the worst case per tokenizer."""
    hdr = header
    if hdr is None:
        try:
            from planners.contract import PACKAGE_HEADER as hdr
        except ImportError:                                   # pragma: no cover
            from sim.planners.contract import PACKAGE_HEADER as hdr
    worst, over, n = {}, [], 0
    for pk in packages:
        if pk.get("empty"):
            continue
        n += 1
        c = counter.counts(hdr + chr(10) + pk["text"])
        for k, v in c.items():
            worst[k] = max(worst.get(k, 0), int(v))
        if any(int(v) > int(pk["tokens"]["budget"]) for v in c.values()):
            over.append({"sha": pk["sha256"][:12], "counts": c})
    return {"checked": n, "worst": worst, "over_budget": over, "pass": not over,
            "tokenizers": list(getattr(counter, "tokenizers", []) or [])}


def _mean_comp_tok(rows) -> dict:
    """Mean tokens per component and tokenizer over ALL presentations (a component absent from a package counts 0)."""
    tot = {}
    for r in rows:
        for comp, d in r["comp_tok"].items():
            for tk, v in d.items():
                tot.setdefault(comp, {}).setdefault(tk, 0)
                tot[comp][tk] += v
    return {c: {tk: round(v / len(rows), 1) for tk, v in sorted(d.items())} for c, d in sorted(tot.items())} if rows else {}


def _mean_comp(rows) -> dict:
    tot, n = {}, {}
    for r in rows:
        for k, v in r["comp_tokens"].items():
            tot[k] = tot.get(k, 0) + v
            n[k] = n.get(k, 0) + 1
    return {k: round(tot[k] / len(rows), 1) for k in sorted(tot)} if rows else {}


def context_fit_items(builder: PackageBuilder, items: list, count_gemma, n_summary: int = 40, repair_problems: str = None) -> dict:
    """A24 item 1 on the non-test instructions (both languages): system + user (with the package and a full 40-line summary) + 2 repair
    turns + 768 output tokens <= 8192 under the Gemma count. Uses the ft-v2 contract text; the package is the one the builder renders."""
    try:
        from planners.contract import PlannerContract, REPAIR_V2
    except ImportError:                                           # pragma: no cover
        from sim.planners.contract import PlannerContract, REPAIR_V2
    k = getattr(builder, "kitchen", DEFAULT_KITCHEN)               # K23: the kitchen's own B5 list (write_detector_types.py --scene)
    types = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix",
                                        "detector_types.json" if k == DEFAULT_KITCHEN else "detector_types_%s.json" % k), encoding="utf-8"))
    types = types if isinstance(types, list) else types.get("types", types)
    ctr = PlannerContract(types, version="ft-v2")
    system = ctr.system_text()
    summary = ["%s [closed] at (%d.%02d, -%d.%02d), step %d" % (types[i % len(types)], i % 5, i, i % 4, 10 + i, i) for i in range(n_summary)]
    problems = repair_problems or ("step 2: object 'Apple' is not a receptacle for put_in; the hand is empty so put_in has nothing to put; "
                                   "step 3: 'go_to(CounterTop)' is not a tool name; use the tool name alone in action. " * 2)
    repair = REPAIR_V2.format(problems=problems)
    worst, n, bad = None, 0, []
    for it in items:
        for lang in ("en", "fa"):
            if not it.get(lang):
                continue
            pk = builder.build(make_query(it[lang], lang, None, summary))
            user = ctr.user_text(it[lang], summary, None, pk["text"] or None)
            r = context_fit(system, user, count_gemma, 2, repair, 768, 8192)
            n += 1
            if worst is None or r["total"] > worst["total"]:
                worst = dict(r, item=it["id"], lang=lang)
            if not r["fits"]:
                bad.append({"item": it["id"], "lang": lang, "total": r["total"]})
    return {"presentations": n, "worst": worst, "not_fitting": bad, "fits": not bad, "limit": 8192}


def _open_world(a):
    """-> (builder, recs_by_id, items, counter). Fake counter unless --real."""
    cats = load_categories()
    src = default_sources(a.kitchen, a.snapshot, load_practice_rows(a.namespace), categories=cats)
    sys.path.insert(0, os.path.join(HERE, "_pkg_fixtures"))
    if a.real:
        sys.path.insert(0, os.path.join(HERE, "final_test", "c3"))
        from real_counter import RealMaxCounter                   # noqa: E402
        try:
            from planners.contract import PACKAGE_HEADER
        except ImportError:                                       # pragma: no cover
            from sim.planners.contract import PACKAGE_HEADER
        counter = RealMaxCounter(os.path.join(ROOT, "sim", "final_test", "matrix", "work", "tokcache.pkg.json"), PACKAGE_HEADER)
    else:
        from fake_counter import FakeCounter                      # noqa: E402
        counter = FakeCounter()
    params = dict(load_params(), enabled=True)
    b = PackageBuilder(a.kitchen, {k: v for k, v in src.items() if not k.startswith("_")}, counter, params, categories=cats)
    by = {}
    for r in src["briefing"]:
        by[r["id"]] = r
    for store in ("profile", "procedure"):
        for r in src[store].records():
            by[r["id"]] = r
    for c in src["example"].corpus:
        r = src["example"].to_rec(c)
        by[r["id"]] = r
    return b, by, nontest_items(), counter


# ------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-params", action="store_true", help="(re)write sim/step5/package_params.json from DEFAULT_PARAMS")
    ap.add_argument("--check-params", action="store_true")
    ap.add_argument("--leakage-ids", nargs="+", metavar="FILE")
    ap.add_argument("--early-check", action="store_true", help="PACKAGE-SPEC 9.6 on the 116 non-test items (offline)")
    ap.add_argument("--context-fit", action="store_true", help="A24 item 1: the 8192-token context check on the non-test instructions")
    ap.add_argument("--real-token-gate", action="store_true", help="re-count every non-test package with the REAL two-tokenizer counter (needs the local Gemma server and the API key)")
    ap.add_argument("--real", action="store_true", help="use the real counter (local Gemma server + Anthropic count_tokens) instead of the fake one")
    ap.add_argument("--dump-dev", metavar="JSONL", help="write the packages of the dev_pilot items (EN and FA) to this file, for reading")
    ap.add_argument("--out", help="write the JSON report here (UTF-8) instead of stdout")
    ap.add_argument("--kitchen", default="FloorPlan1")
    ap.add_argument("--snapshot", default=os.path.join(HERE, "final_test", "snapshots", "o0", "k10", "ltm.db"))
    ap.add_argument("--namespace", default="practice-v1")
    a = ap.parse_args(argv)

    def emit(obj):
        txt = json.dumps(obj, ensure_ascii=False, indent=1)
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(txt + "\n")
        else:
            sys.stdout.buffer.write((txt + "\n").encode("utf-8"))
    if a.write_params:
        d = default_params_doc()
        with open(PARAMS_PATH, "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
        print(json.dumps({"written": PARAMS_PATH, "sha256": d["sha256"]}))
        return 0
    if a.check_params:
        p = load_params()
        probs = check_params(p)
        print(json.dumps({"sha256": p["sha256"], "problems": probs}))
        return 1 if probs else 0
    if a.leakage_ids:
        r = leakage_ids(a.leakage_ids)
        print(json.dumps(r))
        return 1 if r["test_id_hits"] else 0
    if a.dump_dev:
        b, by, items, counter = _open_world(a)
        C = _common()
        dev = set(C.load_manifest()["dev_pilot"])
        with open(a.dump_dev, "w", encoding="utf-8", newline=chr(10)) as f:
            for it in items:
                if it["id"] not in dev:
                    continue
                for lg in ("en", "fa"):
                    pk = b.build(make_query(it[lg], lg, None, []))
                    f.write(json.dumps({"item": it["id"], "lang": lg, "instruction": it[lg], "package_sha": pk["sha256"], "kept": pk["kept"],
                                        "tokens": pk["tokens"], "parser_ok": pk["parser_ok"], "provenance": pk["provenance"],
                                        "dropped": pk["dropped"], "text": pk["text"]}, ensure_ascii=False) + chr(10))
        print(json.dumps({"written": a.dump_dev, "items": len(dev)}))
        return 0
    if a.early_check or a.context_fit or a.real_token_gate:
        b, by, items, counter = _open_world(a)
        if a.early_check:
            emit(early_check(b, items, by))
        elif a.context_fit:
            if a.real:
                g = lambda t: int(counter.counts(t)["gemma"])      # noqa: E731
            else:
                g = lambda t: counter.counts(t)["gemma"]           # noqa: E731
            emit(context_fit_items(b, items, g))
        else:
            if not a.real:
                ap.error("--real-token-gate needs --real")
            pks = [b.build(make_query(it[lg], lg, None, [])) for it in items for lg in ("en", "fa") if it.get(lg)]
            emit(real_token_gate(pks, counter))
            counter.save()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    sys.exit(main() or 0)
