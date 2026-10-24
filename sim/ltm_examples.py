"""Stage B5 (PACKAGE-SPEC pkg-1, section 8): worked-example retrieval from PRACTICE ONLY.

One example = one practice attempt (corpus schema `pkg.ex/1`). `ExampleIndex.candidates(query)` is the STRUCTURAL source of the package:
it matches by the operations a plan needs and by the shape of the command (verb, destination CATEGORY, state slots), never by object
names, and it never does lexical or dense scoring (B7 does that once for every component; the `embedder` argument is only kept so that
B7 can reach it).

    match = 0.6 * jaccard(op classes of the example's plan, op classes of the query) + 0.4 * role_pattern(query roles, example roles)

Honesty / generality (see the spec, sections 1 and 8):
  * corpus items must be in the split's `practice_pool` (`build_corpus` calls `C.guard_items`); a test item is refused;
  * retrieval reads only the query (instruction text, language, holding, observation types, parse) and the FROZEN corpus. It never
    reads a level, an item id of the query, a reference plan, a goal or the outcome of an evaluation presentation. The practice row's
    own `level` is dropped at build time so it cannot even leak into the corpus;
  * no kitchen constant: the kitchen is DATA (`kitchen` argument); category slots come from a domain table or a caller's `categories`.

CLI:
    python sim/ltm_examples.py --check --kitchen <label>           early check B5 (offline, practice-v1 + non-test items; diagnostic only)
    python sim/ltm_examples.py --build --rows <practice.jsonl> --kitchen <label> --out <examples.jsonl> [--upto K] [--namespace practice-v1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_MATRIX = os.path.join(HERE, "final_test", "matrix")

try:
    import ltm_concepts as lc
except ImportError:                                        # pragma: no cover
    from sim import ltm_concepts as lc

EX_SCHEMA = "pkg.ex/1"
REC_SCHEMA = "pkg.rec/1"
CAND_SCHEMA = "pkg.cand/1"
VERSION = "ex-1"
W_OPS, W_ROLE = 0.6, 0.4                                    # frozen a priori (spec 8.2)

# tool name -> op class (a class is what a plan DOES; parser verbs are already classes)
TOOL_CLASS = {"pickup": "pick", "pick_and_place": "put", "put_in": "put", "drop": "put", "open": "open", "close": "close",
              "toggle_on": "toggle", "toggle_off": "toggle", "slice": "slice", "heat": "heat", "cook": "cook", "wash": "wash",
              "fill": "fill", "empty": "empty", "break": "break", "go_to": "find", "search_for": "find", "look_around": "find",
              "scan_for": "find"}
VERB_CLASS = {"pick": "pick", "put": "put", "drop": "put", "open": "open", "close": "close", "toggle": "toggle", "slice": "slice",
              "heat": "heat", "cook": "cook", "wash": "wash", "fill": "fill", "empty": "empty", "break": "break", "find": "find"}

# Domain table of type categories used ONLY to turn concrete types into slots (skeleton) and to compare destination CATEGORIES.
# It is a domain convention (detector vocabulary), not data from any kitchen. A caller's `categories` (pkg.conv/1 style
# {name: {"types": [...]}}) is consulted first.
DEFAULT_SLOTS = {
    "food": ["Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato"],
    "drinkware": ["Mug", "Cup", "WineBottle", "Bottle"],
    "cutlery": ["Fork", "Spoon", "ButterKnife", "Spatula", "Ladle"],      # A26 (K-B2b): Knife out of scope
    "vessel": ["Pot", "Pan", "Kettle", "Bowl", "Plate"],
    "container": ["Fridge", "Cabinet", "Drawer", "Microwave", "GarbageCan", "Toaster"],
}
DEFAULT_SLOT_ORDER = ["drinkware", "cutlery", "vessel", "container", "food"]


def _sha16(text) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()[:16]


def _module_sha() -> str:
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return _sha16(f.read())
    except OSError:                                         # pragma: no cover
        return "0" * 16


def type_categories(categories=None) -> dict:
    """type -> category name. Caller categories first, then the domain table. Pure."""
    out = {}
    for name, e in (categories or {}).items():
        types = e.get("types", []) if isinstance(e, dict) else (e or [])
        for t in types:
            out.setdefault(t, name)
    for name in DEFAULT_SLOT_ORDER:
        for t in DEFAULT_SLOTS[name]:
            out.setdefault(t, name)
    return out


# ---------------------------------------------------------------------------------------------------------------- plan helpers
def _norm_plan(p) -> list:
    if isinstance(p, dict):
        p = p.get("plan")
    out = []
    for s in p or []:
        if isinstance(s, dict):
            out.append({"action": str(s.get("action") or ""), "object": s.get("object"), "to": s.get("to"), "arg": s.get("arg")})
        else:
            s = list(s) + [None] * 4
            out.append({"action": str(s[0] or ""), "object": s[1], "to": s[2], "arg": s[3]})
    return out


def _tok(s) -> str:
    return str(s).replace(" ", "_")


def plan_tokens(plan, short: bool = False) -> str:
    """`action object>to; ...` (the FS part-1 spelling: no parenthesised id after a tool word)."""
    parts = []
    for s in plan:
        t = s["action"] + (" " + _tok(s["object"]) if s.get("object") else "")
        if s.get("to") and not short:
            t += ">" + _tok(s["to"])
        parts.append(t)
    return "; ".join(parts) or "(no plan)"


def op_classes(tools) -> set:
    return {TOOL_CLASS[t] for t in tools if t in TOOL_CLASS}


def query_classes(query) -> set:
    names = list(query.get("ops") or [])
    names += list((query.get("need") or {}).get("ops") or [])
    out = set()
    for n in names:
        if n in TOOL_CLASS:
            out.add(TOOL_CLASS[n])
        elif n in VERB_CLASS:
            out.add(VERB_CLASS[n])
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


# ---------------------------------------------------------------------------------------------------------------- role pattern
def _known(v) -> bool:
    return v not in ("?", None)


def _role_pat(r: dict) -> dict:
    return {k: r.get(k, "-") for k in ("verb", "req", "obj", "ostate", "dest", "dstate", "neg")}


def _cat(t, tcat) -> str:
    if not _known(t) or t == "-":
        return "-" if t == "-" else "?"
    return tcat.get(t, t)


def role_sim(q: dict, e: dict, tcat: dict) -> float:
    """Similarity of two role patterns: verb (+ polarity) 0.5, destination CATEGORY 0.3, state slots 0.2. '?' never matches."""
    if not _known(q.get("verb")) or q.get("verb") != e.get("verb") or int(q.get("neg", 0)) != int(e.get("neg", 0)):
        return 0.0
    s = 0.5
    qd, ed = _cat(q.get("dest"), tcat), _cat(e.get("dest"), tcat)
    if qd == ed and qd != "?":
        s += 0.3
    slots = [(q.get("ostate"), e.get("ostate")), (q.get("dstate"), e.get("dstate")), (q.get("req"), e.get("req"))]
    s += 0.2 * sum(1 for a, b in slots if a == b and _known(a)) / len(slots)
    return s


def role_pattern(qroles, eroles, tcat=None) -> float:
    """Order-preserving best alignment (LCS style) of the query's roles against the example's roles, divided by the longer length so a
    one-role example does not fully explain a three-role command (and the converse). Roles whose verb is '?' are ignored."""
    tcat = tcat or type_categories()
    q = [_role_pat(r) for r in (qroles or []) if _known(r.get("verb"))]
    e = [_role_pat(r) for r in (eroles or []) if _known(r.get("verb"))]
    if not q or not e:
        return 0.0
    n, m = len(q), len(e)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + role_sim(q[i - 1], e[j - 1], tcat))
    return dp[n][m] / max(n, m)


def roles_sig_strings(roles) -> list:
    out = []
    for r in roles or []:
        if not _known(r.get("verb")):
            continue
        s = f"{r['verb']}/{r.get('obj', '-')}/->{r.get('dest', '-')}"
        for k, tag in (("ostate", "os"), ("dstate", "ds"), ("req", "rq")):
            if r.get(k) not in ("-", None):
                s += f"/{tag}={r[k]}"
        if int(r.get("neg", 0)):
            s += "/neg"
        out.append(s)
    return out


# ---------------------------------------------------------------------------------------------------------------- corpus
def verdict_of(r: dict) -> str:
    """success / failure / crash / inapplicable: an explicit verdict, then crash / inapplicable status, a rejected plan = failure, then
    the practice `admission` (bool, or {"admitted"|"ok": bool}) = the declared door of the goal verdict into memory (A11). Unlike
    fs_block._verdict there is NO fallback to the raw goal verdict and NO `applied` alias (in step5_live / c3 rows `applied` means "a
    memory hint changed a decision", not an admission): a row with none of these is refused (fail closed). On every real practice
    row the result equals fs_block._verdict (practice_k always writes `admission`)."""
    v = r.get("verdict")
    if isinstance(v, str) and v.lower() in ("success", "failure"):
        return v.lower()
    st = r.get("status")
    if st in ("crash", "inapplicable"):
        return st
    if st == "rejected":
        return "failure"
    a = r.get("admission")
    if isinstance(a, dict):
        a = a.get("admitted", a.get("ok"))
    if isinstance(a, bool):
        return "success" if a else "failure"
    raise ValueError("practice row has no verdict, no crash / inapplicable / rejected status and no admission")


def _steps_of(r: dict, n_plan: int) -> list:
    steps = r.get("exec_steps", r.get("steps"))
    if not steps or r.get("plan_not_executed_reason") or r.get("plan_not_executed"):
        return []
    return [{"i": int(s.get("i", k)), "ok": bool(s.get("ok")), "cause": (None if s.get("ok") else (s.get("cause") or "failed"))}
            for k, s in enumerate(steps)]


def _initial_lines(r: dict, plan: list, tcat: dict) -> list:
    """Lines of the attempt's OWN observations (its sightings, a practice stream) for the types the plan touches. A row carries no
    pre-plan summary text, so this is labelled by the renderer as seen-in-that-episode, never as the planner's input."""
    want = set()
    for s in plan:
        for k in ("object", "to"):
            if s.get(k):
                want.add(s[k])
    out, seen = [], set()
    for sg in r.get("sightings") or []:
        if not isinstance(sg, dict):
            continue
        t, st = sg.get("otype", sg.get("type")), sg.get("state")
        if t in want and st and st != "?":
            line = f"{t} [{st}]"
            if line not in seen:
                seen.add(line)
                out.append(line)
    out.append("holding nothing")
    return out


def skeleton_of(plan: list, tcat: dict) -> list:
    """Types -> category slots: `Apple` -> `?food`; a second distinct type of the same category gets `?food2`. A type with no category
    (a fixture such as Sink or StoveBurner) stays literal."""
    slots, counts = {}, {}

    def slot(t):
        if not t:
            return t
        if t in slots:
            return slots[t]
        c = tcat.get(t)
        if c is None:
            slots[t] = t
        else:
            counts[c] = counts.get(c, 0) + 1
            slots[t] = "?" + c + ("" if counts[c] == 1 else str(counts[c]))
        return slots[t]

    return [{"action": s["action"], "object": slot(s.get("object")), "to": slot(s.get("to"))} for s in plan]


def build_corpus(practice_rows, kitchen, kb=None, upto_attempt=None, categories=None, lex=None, guard=True, root=None) -> list:
    """pkg.ex/1 list from practice journal rows. Refuses any item outside `practice_pool` (C.guard_items). `kb` (optional, duck typed)
    may offer `why_order(ops_sig, plan) -> {"id","en","fa"} | None` (B4); without it `why_order` is None. Recorded outcomes (`steps`,
    `verdict`) are copied as recorded and are never predicted."""
    rows = [r for r in practice_rows if upto_attempt is None or int(r.get("attempt", 0)) <= int(upto_attempt)]
    if not kitchen or not isinstance(kitchen, str):
        raise ValueError("build_corpus needs a kitchen label (data, not a branch)")
    if guard:
        C = _common()
        for ns in sorted({r.get("namespace") for r in rows}):
            ids = [r.get("item") for r in rows if r.get("namespace") == ns]
            C.guard_items(ids, ns, root=root)
    tcat = type_categories(categories)
    lex = lex or lc.default_lexicon()
    out = []
    for r in rows:
        plan = _norm_plan(r.get("plan"))
        verdict = verdict_of(r)
        parse = lc.parse(r.get("instruction") or "", lex)
        tools = sorted(s["action"] for s in plan)
        roles_pat = [_role_pat(x) for x in parse["roles"] if _known(x.get("verb"))]
        why = None
        if kb is not None and hasattr(kb, "why_order") and plan:
            try:
                why = kb.why_order(tools, plan)
            except Exception:                               # pragma: no cover - a B4 adapter must never break the corpus
                why = None
        ns = r.get("namespace") or "practice"
        out.append({
            "schema": EX_SCHEMA, "id": f"example/{kitchen}/{ns}:o{int(r.get('order', 0))}:a{int(r.get('attempt', 0))}",
            "kitchen": kitchen, "ns": ns, "order": int(r.get("order", 0)), "attempt": int(r.get("attempt", 0)), "item": r.get("item"),
            "lang": r.get("lang") or "en", "instruction": r.get("instruction") or "",
            "ops_sig": tools, "roles_sig": roles_sig_strings(parse["roles"]), "roles_pat": roles_pat, "parse_ok": bool(parse["ok"]),
            "initial": _initial_lines(r, plan, tcat), "plan": plan, "skeleton": skeleton_of(plan, tcat),
            "steps": _steps_of(r, len(plan)), "verdict": verdict,
            "why_order": (why or {}).get("id") if isinstance(why, dict) else None,
            "why_text": ({"en": why.get("en"), "fa": why.get("fa")} if isinstance(why, dict) else None),
            "teacher": "gemma",
            "provenance": {"source": f"practice:{ns}/o{int(r.get('order', 0))}a{int(r.get('attempt', 0))}",
                           "evidence": [str(r.get("key") or r.get("request_sha") or "")],
                           "authored_by": f"script:ltm_examples@{_module_sha()}", "support": [r.get("item")]},
        })
    problems = audit_corpus(out)
    if problems:
        raise ValueError("example corpus audit failed: " + "; ".join(problems[:5]))
    return out


_FORBIDDEN_KEYS = ("level", "goal", "goal_ok", "reference_plan", "ref_plan", "scene")


def audit_corpus(corpus) -> list:
    """Problems that would be a leak (Astra H-B5): a level / goal / scene key, a simulator object id or a scene name in any text field."""
    bad = []
    for e in corpus:
        for k in _FORBIDDEN_KEYS:
            if k in e:
                bad.append(f"{e.get('id')}: forbidden key {k!r}")
        blob = json.dumps({k: v for k, v in e.items() if k not in ("provenance",)}, ensure_ascii=False)
        if "FloorPlan" in blob or "|" in "".join(str(s.get("object") or "") + str(s.get("to") or "") for s in e.get("plan") or []):
            bad.append(f"{e.get('id')}: scene name or simulator object id in a text/plan field")
    return bad


def write_corpus(corpus, path) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in corpus:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    return _sha16(open(path, "rb").read())


def read_corpus(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _common():
    if _MATRIX not in sys.path:
        sys.path.insert(0, _MATRIX)
    import common as C
    return C


# ---------------------------------------------------------------------------------------------------------------- index
class ExampleIndex:
    def __init__(self, corpus, embedder=None, cache=None, categories=None):
        self.corpus = [dict(e) for e in (corpus or [])]
        self.embedder, self.cache = embedder, cache          # kept for B7; this source does no dense scoring (spec 3.3)
        self.tcat = type_categories(categories)
        self.version = VERSION
        for e in self.corpus:
            if e.get("schema") not in (EX_SCHEMA, None):
                raise ValueError(f"unknown example schema {e.get('schema')!r}")

    @classmethod
    def load(cls, path, embedder=None, cache=None, categories=None):
        return cls(read_corpus(path), embedder=embedder, cache=cache, categories=categories)

    def sha(self) -> str:
        return _sha16(json.dumps(self.corpus, sort_keys=True, ensure_ascii=False))

    # -- eligibility -------------------------------------------------------------------------------------------------
    @staticmethod
    def eligible(ex, include_failures=False) -> bool:
        if not ex.get("plan"):
            return False
        v = ex.get("verdict")
        return v == "success" or (include_failures and v == "failure")

    # -- retrieval ---------------------------------------------------------------------------------------------------
    @staticmethod
    def example_classes(ex) -> set:
        """What the example REQUIRES. A success: the classes of the plan that worked. A failure (Astra H-B5): the plan may omit or add
        operations, so the classes come from the instruction's own parsed roles (instruction-derived), falling back to the plan."""
        plan_cls = op_classes(ex.get("ops_sig") or [])
        if ex.get("verdict") == "success":
            return plan_cls
        role_cls = {VERB_CLASS[r["verb"]] for r in ex.get("roles_pat") or [] if r.get("verb") in VERB_CLASS}
        return role_cls or plan_cls

    def _match(self, query, ex, qcls):
        ecls = self.example_classes(ex)
        if "find" not in qcls:
            ecls = ecls - {"find"}
        j = jaccard(ecls, qcls)
        roles = (query.get("parse") or {}).get("roles") or []
        r = role_pattern(roles, ex.get("roles_pat") or [], self.tcat)
        return round(W_OPS * j + W_ROLE * r, 6), j, r, ecls

    def candidates(self, query, include_failures=False, exclude_items=None, failure_fallback=False, fallback_min=0.5) -> list:
        """pkg.cand/1 list, best first. Reads only: query instruction-derived fields (ops, need.ops, parse.roles, types, lang) and the
        frozen corpus. `exclude_items` exists for leave-one-out diagnostics on practice items; production passes None.
        `failure_fallback` (Astra H-B5, default OFF, a frozen-rule candidate for A24.1): when include_failures is False and NO success
        reaches `fallback_min`, the best failure is appended once, to be rendered with its failure label (never as a solution)."""
        if failure_fallback and not include_failures:
            succ = self.candidates(query, False, exclude_items)
            if succ and succ[0]["match"] >= fallback_min:
                return succ
            fails = [c for c in self.candidates(query, True, exclude_items) if c["rec"]["provenance"].get("verdict") == "failure"]
            return succ + fails[:1]
        qcls = query_classes(query)
        qtypes = set(query.get("types") or [])
        excl = set(exclude_items or ())
        best = {}
        scored = []
        for ex in self.corpus:
            if ex.get("item") in excl or not self.eligible(ex, include_failures):
                continue
            m, j, r, ecls = self._match(query, ex, qcls)
            if m <= 0:
                continue
            etypes = {s[k] for s in ex["plan"] for k in ("object", "to") if s.get(k)}
            shared = sorted(qtypes & etypes)
            why = [f"op:{c}" for c in sorted(ecls & qcls)] + [f"type:{t}" for t in shared]
            if r > 0:
                why.append("pattern:roles")
            scored.append((m, len(shared), ex, why))
        # one example per equivalent group: the most recent practice attempt (deterministic and LANGUAGE-INDEPENDENT, so an item's EN and
        # FA presentations retain the same ids; the example is shown in its own practice language)
        for m, ns_, ex, why in scored:
            fail_cause = next((st.get("cause") for st in ex.get("steps") or [] if not st.get("ok")), None)
            key = (json.dumps(ex["plan"], sort_keys=True), ex.get("verdict"), tuple(ex.get("roles_sig") or []), fail_cause)
            cur = best.get(key)
            rank = (m, ex.get("attempt", 0), ex.get("order", 0), ex["id"])
            if cur is None or rank > cur[0]:
                best[key] = (rank, m, ns_, ex, why)
        rows = sorted(best.values(), key=lambda t: (-t[1], -t[2], -t[3].get("attempt", 0), t[3]["id"]))
        return [{"schema": CAND_SCHEMA, "rec": self.to_rec(ex), "match": m, "why": why} for _r, m, _s, ex, why in rows]

    # -- rendering ---------------------------------------------------------------------------------------------------
    def render(self, ex, lang, raw=False, query=None) -> str:
        """raw=False: the package line (a success; a failure only in its labelled did-NOT-work form; '' for crash / inapplicable). raw=True: the FS part-1 line
        `n. lang | instruction | plan tokens | per-step outcome | verdict`."""
        if lang not in ("en", "fa"):
            raise ValueError("lang must be en or fa")
        plan = ex.get("plan") or []
        if raw:
            steps = " ".join("ok" if s.get("ok") else str(s.get("cause") or "failed") for s in ex.get("steps") or []) or \
                ("not_run" if plan else "-")
            return f"{ex.get('attempt', 0)}. {ex.get('lang', 'en')} | {ex['instruction']} | {plan_tokens(plan)} | {steps} | {ex.get('verdict')}"
        if not plan or ex.get("verdict") not in ("success", "failure"):
            return ""
        txt = plan_tokens(plan)
        if ex["verdict"] == "failure":                      # labelled, never a solution; reached only through the failure fallback / FS
            cause = next((st.get("cause") for st in ex.get("steps") or [] if not st.get("ok")), None) or "goal not reached"
            if lang == "en":
                return f'Example from my practice (did NOT work, do not copy): "{ex["instruction"]}" -> {txt}; what went wrong: {cause}'
            return f"نمونه‌ای از تمرین من (جواب نداد، تکرار نکن): «{ex['instruction']}» -> {txt}؛ مشکل: {cause}"
        if lang == "en":
            s = f'Example from my practice (succeeded): "{ex["instruction"]}" -> {txt}'
        else:
            s = f"نمونه‌ای از تمرین من (موفق): «{ex['instruction']}» -> {txt}"
        sk = ex.get("skeleton") or []
        if query is not None and sk and set(query.get("types") or []) - {s_.get(k) for s_ in plan for k in ("object", "to")} \
                and plan_tokens(sk) != txt:
            s += ("; pattern: " if lang == "en" else "؛ الگو: ") + plan_tokens(sk)
        wt = ex.get("why_text") or {}
        if wt.get(lang):
            s += ("; why this order: " if lang == "en" else "؛ دلیل این ترتیب: ") + wt[lang]
        return s

    def to_rec(self, ex) -> dict:
        types = sorted({s[k] for s in ex["plan"] for k in ("object", "to") if s.get(k)})
        ops = sorted(set(ex.get("ops_sig") or []))
        ok = ex.get("verdict") == "success"
        words_en = " ".join(types + ops)
        # the instruction is a fact in its own language; the other side gets only language-free tokens (types, tool names)
        lex = {"en": (ex["instruction"] + " " if ex.get("lang") == "en" else "") + words_en,
               "fa": (ex["instruction"] + " " if ex.get("lang") == "fa" else "") + words_en}
        en, fa = self.render(ex, "en"), self.render(ex, "fa")
        a = int(ex.get("attempt", 0))
        return {"schema": REC_SCHEMA, "id": ex["id"], "component": "example", "kitchen": ex.get("kitchen", "*"),
                "keys": {"types": types, "ops": ops, "rels": []}, "lex": lex, "text": {"en": en, "fa": fa},
                "provenance": dict(ex.get("provenance") or {}, verdict=ex.get("verdict")), "created": a, "recency": a,
                "confidence": 1.0 if ok else 0.5, "privileged": False, "prio": a}


# ---------------------------------------------------------------------------------------------------------------- diagnostics
def mini_query(instruction, lang, lex=None, holding=None, observed=()) -> dict:
    """Minimal pkg.query/1 for diagnostics and selftests (B7's make_query is the production builder)."""
    p = lc.parse(instruction, lex or lc.default_lexicon())
    roles = p["roles"]
    ops = sorted({r["verb"] for r in roles if _known(r.get("verb"))})
    types = sorted({r[k] for r in roles for k in ("obj", "src", "dest") if r.get(k) not in (None, "-", "?", "PRON")})
    return {"schema": "pkg.query/1", "instruction": instruction, "lang": lang, "holding": holding, "observed": list(observed),
            "parse": {"ok": p["ok"], "roles": roles, "reasons": p["reasons"]}, "types": types, "ops": ops, "goal_states": [],
            "need": {"types": [], "tools": [], "appliances": [], "receptacles": [], "ops": []}, "parser_fallback": not p["ok"]}


def early_check(corpus, items, tcat=None) -> dict:
    """Early check B5 (diagnostic only). `items` = [{id, en, fa, plan}] non-test items with their reference plans (used ONLY to score the
    diagnostic, never as retrieval input). Leave-one-out by ITEM: an item's own practice attempts are excluded from its candidates."""
    idx = ExampleIndex(corpus)
    lex = lc.default_lexicon()
    res = {}
    for include_failures in (False, True):
        for lang in ("en", "fa"):
            n = empty = same_sig = same_cls = 0
            for it in items:
                if not it.get(lang):
                    continue
                q = mini_query(it[lang], lang, lex)
                cands = idx.candidates(q, include_failures=include_failures, exclude_items={it["id"]})
                n += 1
                if not cands:
                    empty += 1
                    continue
                top = next(e for e in corpus if e["id"] == cands[0]["rec"]["id"])
                ref = sorted(_norm_plan_actions(it["plan"]))
                same_sig += int(sorted(top["ops_sig"]) == ref)
                same_cls += int(op_classes(top["ops_sig"]) - {"find"} == op_classes(ref) - {"find"})
            res[f"{lang}{'_with_failures' if include_failures else ''}"] = {
                "queries": n, "empty": empty, "empty_rate": round(empty / n, 3) if n else None,
                "top1_same_ops_sig": same_sig, "top1_same_op_classes": same_cls,
                "top1_same_ops_sig_rate_of_answered": round(same_sig / (n - empty), 3) if n - empty else None,
                "top1_same_op_classes_rate_of_answered": round(same_cls / (n - empty), 3) if n - empty else None}
    return res


def _norm_plan_actions(plan):
    return [s["action"] for s in _norm_plan(plan)]


def self_loo(corpus) -> dict:
    """Leave-one-out over the corpus itself (spec 8.3): each example's instruction as the query, its own item excluded."""
    idx = ExampleIndex(corpus)
    lex = lc.default_lexicon()
    out = {}
    for lang in ("en", "fa"):
        rows = [e for e in corpus if e.get("lang") == lang]
        n = empty = same = 0
        for e in rows:
            cands = idx.candidates(mini_query(e["instruction"], lang, lex), include_failures=True, exclude_items={e["item"]})
            n += 1
            if not cands:
                empty += 1
                continue
            top = next(x for x in corpus if x["id"] == cands[0]["rec"]["id"])
            same += int(sorted(top["ops_sig"]) == sorted(e["ops_sig"]))
        out[lang] = {"queries": n, "empty": empty, "empty_rate": round(empty / n, 3) if n else None, "top1_same_ops_sig": same}
    return out


def _cli_check(args):
    C = _common()
    rows = []
    with open(os.path.join(HERE, "..", "docs", "final-test", "matrix", args.namespace, "practice.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows0 = [r for r in rows if r.get("order") == 0]
    m = C.load_manifest()
    nontest = set(m["nontest"])
    with open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8") as f:
        gen = json.load(f)
    items = [{"id": it["id"], "en": it.get("en"), "fa": it.get("fa"), "plan": it.get("plan")} for it in gen["items"] if it["id"] in nontest]
    out = {"data": "practice-v1 rows (practice_pool) + non-test items (reference plans used only to score the diagnostic)",
           "kitchen": args.kitchen, "version": VERSION}
    for name, rs in (("o0", rows0), ("all_orders", rows)):
        corpus = build_corpus(rs, args.kitchen, root=None)
        out[name] = {"rows": len(rs), "verdicts": _count(c["verdict"] for c in corpus), "self_leave_one_item_out": self_loo(corpus),
                     "non_test_queries": early_check(corpus, items), "corpus_sha16": _sha16(json.dumps(corpus, sort_keys=True))}
    print(json.dumps(out, ensure_ascii=False, indent=1))


def _count(it):
    d = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rows")
    ap.add_argument("--out")
    ap.add_argument("--kitchen")
    ap.add_argument("--upto", type=int)
    ap.add_argument("--namespace", default="practice-v1")
    a = ap.parse_args(argv)
    if not a.kitchen:
        ap.error("--kitchen is required (a kitchen label is data)")
    if a.check:
        return _cli_check(a)
    if a.build:
        if not (a.rows and a.out):
            ap.error("--build needs --rows and --out")
        with open(a.rows, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        corpus = build_corpus(rows, a.kitchen, upto_attempt=a.upto)
        print(json.dumps({"examples": len(corpus), "sha16": write_corpus(corpus, a.out), "out": a.out}))
        return 0
    ap.error("choose --check or --build")


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main() or 0)
