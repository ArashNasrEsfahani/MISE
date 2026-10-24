"""Shape checkers for the assistance-package schemas (final test, stage B7; PACKAGE-SPEC section 3).

pkg.rec/1   one knowledge record (every store emits these)
pkg.query/1 one presentation's query (ltm_package.make_query)
pkg.cand/1  what a source returns for a query
pkg.pkg/1   the rendered package of one presentation

Every validator returns a LIST of problem strings ([] = valid); none raises. The English-only override (Arash, 2026-10-12)
is enforced here: a stored record has a non-empty ``text.en`` and its ``text.fa`` is optional and never rendered.
Pure functions, no I/O, no simulator, no network. Python 3.10 syntax.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA_REC = "pkg.rec/1"
SCHEMA_QUERY = "pkg.query/1"
SCHEMA_CAND = "pkg.cand/1"
SCHEMA_PKG = "pkg.pkg/1"
SCHEMA_LOG = "pkg.log/1"
SCHEMA_PARAMS = "pkg.params/1"
VERSION = "pkg-1"

COMPONENTS = ("briefing", "profile", "places", "procedure", "example", "template")
#: the package's own synthetic component (built by ltm_package from the query and the stores; never stored)
SYNTHETIC_COMPONENTS = ("uncertainty",)
#: closed relation vocabulary (PACKAGE-SPEC 3.4)
RELS = ("contains", "tool_at", "requires", "appliance", "controls", "near", "opens", "toggles", "stores", "effect", "confuse")
DROP_REASONS = ("budget", "quota", "irrelevant", "duplicate")

_ID = re.compile(r"^[A-Za-z0-9_.:/@*\-]+$")
#: the dev data showed that "tool(Type)" invites go_to(CounterTop)-style plan errors: no parenthesised id after a tool-like word
_TOOL_CALL = re.compile(r"\b(go_to|pickup|pick_and_place|put_in|open|close|slice|heat|cook|toggle_on|toggle_off|drop|wash|clean|fill|empty|break)\s*\(\s*[A-Z][A-Za-z0-9]*\s*[,)]")
_TYPE_TOKEN = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_STATE_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def self_sha(obj: dict) -> str:
    """sha256 of canonical JSON without the object's own 'sha256' field (the repo convention, common.self_sha)."""
    return sha256_text(canon({k: v for k, v in obj.items() if k != "sha256"}))


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _strs(x) -> bool:
    return isinstance(x, list) and all(isinstance(s, str) for s in x)


def split_rel(rel: str):
    """'contains:Egg' -> ('contains', 'Egg'); anything else -> (rel, None)."""
    if isinstance(rel, str) and ":" in rel:
        a, b = rel.split(":", 1)
        return a, b
    return rel, None


def validate_record(rec, *, allow_synthetic: bool = False) -> list:
    p = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema") != SCHEMA_REC:
        p.append("schema must be %r" % SCHEMA_REC)
    rid = rec.get("id")
    if not isinstance(rid, str) or not rid or not _ID.match(rid):
        p.append("id must be a non-empty ASCII string without spaces: %r" % (rid,))
    comps = COMPONENTS + (SYNTHETIC_COMPONENTS if allow_synthetic else ())
    comp = rec.get("component")
    if comp not in comps:
        p.append("component %r not in %s" % (comp, list(comps)))
    elif isinstance(rid, str) and not rid.startswith(comp + "/"):
        p.append("id %r must start with its component %r + '/'" % (rid, comp))
    if not isinstance(rec.get("kitchen"), str) or not rec.get("kitchen"):
        p.append("kitchen must be a non-empty string ('*' = scene independent)")
    k = rec.get("keys")
    if not isinstance(k, dict) or not all(_strs(k.get(f)) for f in ("types", "ops", "rels")):
        p.append("keys must hold string lists types/ops/rels")
    else:
        for r in k["rels"]:
            rel, t = split_rel(r)
            if rel not in RELS:
                p.append("rel %r not in the closed vocabulary" % r)
            elif rel in ("effect", "confuse"):                 # op -> state token; a frequent confusion names an op: lower-case tokens
                if t is None or not _STATE_TOKEN.match(t):
                    p.append("rel %r needs a '<rel>:<token>' form" % r)
            elif t is None or not _TYPE_TOKEN.match(t):
                p.append("rel %r needs a '<rel>:<Type>' form" % r)
        for t in k["types"]:
            if not _TYPE_TOKEN.match(t):
                p.append("type token %r is not a CamelCase type name" % t)
    lex = rec.get("lex")
    if not isinstance(lex, dict) or not all(isinstance(lex.get(f, ""), str) for f in ("en", "fa")):
        p.append("lex must hold strings en/fa")
    tx = rec.get("text")
    if not isinstance(tx, dict) or not isinstance(tx.get("en"), str) or not tx.get("en", "").strip():
        p.append("text.en must be a non-empty string (knowledge is English only)")
    else:
        if not isinstance(tx.get("fa", ""), str):
            p.append("text.fa must be a string when present")
        if _TOOL_CALL.search(tx["en"]):
            p.append("text.en has a parenthesised id after a tool word (invites go_to(Type) errors)")
        # a worked example quotes the practice instruction in the language the user typed it (user input, not knowledge)
        body = re.sub(r'"[^"]*"', '""', tx["en"]) if comp == "example" else tx["en"]
        if re.search(r"[؀-ۿ]", body):
            p.append("text.en contains Persian characters (knowledge is stored in English only)")
    pv = rec.get("provenance")
    if not isinstance(pv, dict) or not isinstance(pv.get("source"), str) or not pv.get("source") \
            or not isinstance(pv.get("authored_by"), str) or not pv.get("authored_by") \
            or not _strs(pv.get("evidence")) or not _strs(pv.get("support")):
        p.append("provenance needs source, authored_by (strings) and evidence, support (string lists)")
    for f in ("created", "recency", "prio"):
        if not isinstance(rec.get(f), int) or isinstance(rec.get(f), bool):
            p.append("%s must be an integer" % f)
    c = rec.get("confidence")
    if not _num(c) or not 0.0 <= float(c) <= 1.0:
        p.append("confidence must be in [0,1]")
    if not isinstance(rec.get("privileged"), bool):
        p.append("privileged must be a boolean")
    elif rec["privileged"] != (comp == "briefing"):
        p.append("privileged must be true only for component 'briefing'")
    return p


def validate_query(q) -> list:
    p = []
    if not isinstance(q, dict):
        return ["query is not an object"]
    if q.get("schema") != SCHEMA_QUERY:
        p.append("schema must be %r" % SCHEMA_QUERY)
    if not isinstance(q.get("instruction"), str):
        p.append("instruction must be a string")
    if q.get("lang") not in ("en", "fa"):
        p.append("lang must be en or fa")
    h = q.get("holding")
    if h is not None and not (isinstance(h, str) and _TYPE_TOKEN.match(h)):
        p.append("holding must be a type or null")
    for f in ("types", "ops"):
        if not _strs(q.get(f)):
            p.append("%s must be a string list" % f)
    ob = q.get("observed")
    if not isinstance(ob, list) or not all(isinstance(o, dict) and isinstance(o.get("type"), str) and isinstance(o.get("state"), str) for o in ob):
        p.append("observed must be a list of {type, state}")
    ps = q.get("parse")
    if not isinstance(ps, dict) or not isinstance(ps.get("ok"), bool) or not isinstance(ps.get("roles"), list):
        p.append("parse must hold ok (bool) and roles (list)")
    nd = q.get("need")
    if not isinstance(nd, dict) or not all(_strs(nd.get(f)) for f in ("types", "tools", "appliances", "receptacles", "ops")):
        p.append("need must hold string lists types/tools/appliances/receptacles/ops")
    if not isinstance(q.get("parser_fallback"), bool):
        p.append("parser_fallback must be a boolean")
    elif isinstance(ps, dict) and isinstance(ps.get("ok"), bool) and q["parser_fallback"] == ps["ok"]:
        p.append("parser_fallback must equal (not parse.ok)")
    for bad in ("level", "item", "item_id", "reference_plan", "plan", "goal", "gold"):
        if bad in q:
            p.append("query carries the forbidden key %r (retrieval may not see level, item id, plan or goal)" % bad)
    return p


def validate_cand(c) -> list:
    p = []
    if not isinstance(c, dict):
        return ["candidate is not an object"]
    if "rec" not in c:
        return ["candidate has no rec"]
    p += ["rec: " + x for x in validate_record(c["rec"])]
    m = c.get("match")
    if not _num(m) or not 0.0 <= float(m) <= 1.0:
        p.append("match must be in [0,1]")
    if not _strs(c.get("why", [])):
        p.append("why must be a string list")
    return p


def validate_pkg(pk) -> list:
    p = []
    if not isinstance(pk, dict):
        return ["package is not an object"]
    if pk.get("schema") != SCHEMA_PKG:
        p.append("schema must be %r" % SCHEMA_PKG)
    if pk.get("version") != VERSION:
        p.append("version must be %r" % VERSION)
    if pk.get("lang") not in ("en", "fa"):
        p.append("lang must be en or fa")
    if not isinstance(pk.get("text"), str):
        p.append("text must be a string")
    if not _strs(pk.get("kept", [])):
        p.append("kept must be a string list")
    for d in pk.get("dropped") or []:
        if not isinstance(d, dict) or d.get("reason") not in DROP_REASONS or "id" not in d:
            p.append("dropped entry malformed: %r" % (d,))
            break
    tk = pk.get("tokens")
    if not isinstance(tk, dict) or "max" not in tk or "budget" not in tk:
        p.append("tokens must hold max and budget")
    elif tk["max"] > tk["budget"]:
        p.append("package is over budget: %s > %s" % (tk["max"], tk["budget"]))
    if pk.get("empty") != (pk.get("text") == ""):
        p.append("empty must equal (text == '')")
    if pk.get("empty") and pk.get("kept"):
        p.append("an empty package cannot keep records")
    if not pk.get("empty") and not pk.get("kept"):
        p.append("a non-empty package must list the records it kept")
    if pk.get("sha256") and pk["sha256"] != self_sha(pk):
        p.append("sha256 does not match the package content")
    for pv in pk.get("provenance") or []:
        if not pv.get("source") or not pv.get("authored_by"):
            p.append("kept record %r has no provenance" % pv.get("id"))
    return p
