"""Step 5 / A4 - concept extraction for retrieval (docs/step5-plan.md 8.2). Rule based, bilingual, NO LLM, pure.

`parse(text, lex)` turns an instruction into ordered ACTION ROLES. It FAILS CLOSED (Astra A4 review): anything the grammar below does not
understand becomes '?', never a silent '-'. A role is

    {"verb", "req", "obj", "ostate", "src", "dest", "dstate", "neg"}

    verb   canonical verb (pick/put/open/close/slice/toggle/find/heat/cook/wash/fill/empty/drop/break) or '?'
    req    REQUESTED state implied by the verb ('on'/'off' for toggle), '-' otherwise, '?' when a toggle has no direction
    obj    canonical object type (a detector class) or '?'
    ostate / dstate   CURRENT-state adjective attached to the object / destination ("the open fridge"), '-' if none
    src / dest        source / destination type after a from/in/on preposition: '-' = none stated, '?' = stated but unresolved
    neg    1 when the clause is negated, else 0 (negation is scoped to ITS clause)

Clauses are split at hard separators (then/but/./;/سپس/ولی/اما) always, and at soft ones (and/و/comma) or at "without" ONLY when both
sides contain a verb (so "put the apple and the bread in the fridge" stays ONE clause with an ambiguous object -> '?'). There is no
verb inheritance. A pronoun resolves to the previous clause's object only when that clause mentions exactly one type. before / after
(not "after that") / until / if / unless / when / or ... mark the instruction `ok = False` (order or condition cannot be encoded).

Language handling: both languages are looked up in one pass (a mixed sentence works). Persian: the object is the type before را (or the
single unmarked type), the destination follows در/توی/داخل/روی/به, the source follows از, the verb is last; compound verbs are
"noun + light verb" (باز کن / باز نکن), negation is the attached ن- form, نه, بدون, هرگز or نباید. Known limits (documented, tested as
fail-closed): plural/number agreement, relative clauses, ellipsis across sentences, nouns outside the 49 detector classes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    from agentic_rag import normalize_text
except ImportError:                     # pragma: no cover
    from sim.agentic_rag import normalize_text

PARSER_VERSION = "2"      # W4a / prereg A17: FA partitive drop + FA elided-object rule (see parse())
LEXICON_PATH = Path(__file__).resolve().parent / "step5" / "lexicon" / "concepts.json"
ROLE_FIELDS = ("verb", "req", "obj", "ostate", "src", "dest", "dstate", "neg")
H, S = "§H", "§S"                                   # hard / soft separator sentinels (never produced by the lexicon)
_PERSIAN_RE = re.compile("[؀-ۿ]")
FA_SUFFIXES = ("های", "ها")                         # plural suffixes stripped ONLY when the stripped form is a known alias


def prep(text) -> list:
    """normalize_text (NFKC + Persian folding, ZWNJ -> space) -> lowercase -> apostrophes dropped -> punctuation to sentinels."""
    s = normalize_text(text).lower()
    s = s.replace("'", "").replace("’", "").replace("`", "")
    out = []
    for ch in s:
        if ch in ".!?;؛؟\n":
            out.append(f" {H} ")
        elif ch in ",،":
            out.append(f" {S} ")
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out).split()


def _phrase(s) -> tuple:
    return tuple(prep(s))


class Lexicon:
    """Loaded, normalised lookup tables + provenance (sha256 of the file) + load-time collision detection."""

    def __init__(self, data: dict, sha: str):
        self.data, self.sha = data, sha
        self.version = data.get("version")
        self.collisions: list = []
        self.types: dict = {}                                  # phrase tuple -> canonical type
        self.max_type = 1
        for t, e in data["types"].items():
            for lang_key in ("en", "en_plural", "fa"):
                for a in e.get(lang_key, []):
                    ph = _phrase(a)
                    if not ph:
                        continue
                    if ph in self.types and self.types[ph] != t:
                        self.collisions.append((" ".join(ph), self.types[ph], t))
                        continue
                    self.types[ph] = t
                    self.max_type = max(self.max_type, len(ph))
        self.verbs: dict = {}                                  # phrase -> (verb, req, neg)
        self.max_verb = 1
        for key, e in data["verbs"].items():
            verb, _, req = key.partition(":")
            for fld, neg in (("en", 0), ("en_ing", 0), ("fa", 0), ("fa_neg", 1)):
                for a in e.get(fld, []):
                    ph = _phrase(a)
                    if not ph:
                        continue
                    val = (verb, req or "-", neg)
                    if ph in self.verbs and self.verbs[ph] != val:
                        self.collisions.append((" ".join(ph), self.verbs[ph], val))
                        continue
                    self.verbs[ph] = val
                    self.max_verb = max(self.max_verb, len(ph))
        self.adj: dict = {}
        self.max_adj = 1
        for st, e in data["adjectives"].items():
            for a in e.get("en", []) + e.get("fa", []):
                ph = _phrase(a)
                self.adj[ph] = st
                self.max_adj = max(self.max_adj, len(ph))
        self.neg = self._set(data["negation"]["en"] + data["negation"]["fa"])
        self.dest = self._set(data["prep_dest"]["en"] + data["prep_dest"]["fa"])
        self.src = self._set(data["prep_src"]["en"] + data["prep_src"]["fa"])
        self.acc = self._set(data["accusative_fa"])
        self.pron = self._set(data["pronouns"]["en"] + data["pronouns"]["fa"])
        self.quant = self._set(data["quantifiers"]["en"] + data["quantifiers"]["fa"])
        self.complex = self._set(data["complex_markers"]["en"] + data["complex_markers"]["fa"])
        self.complex_bi = self._set(data["complex_bigrams_fa"])
        self.hard = self._set(data["hard_sep"]["en"] + data["hard_sep"]["fa"])
        self.soft = self._set(data["soft_sep"]["en"] + data["soft_sep"]["fa"])
        self.det = self._set(data["determiners"]["en"] + data["determiners"]["fa"])
        self.partitive = self._set(data.get("partitive_fa", []))
        el = data.get("elided_object_fa", {})
        self.elide_src, self.elide_dst = frozenset(el.get("source_verbs", [])), frozenset(el.get("target_verbs", []))
        dof = data.get("door_of_fa", {})
        self.door_words, self.door_verbs = self._set(dof.get("words", [])), frozenset(dof.get("verbs", []))
        self.light = {t for ph in self._set(data["fa_light_verb_tokens"]) for t in ph}
        self.dual = {ph for ph in self.verbs if len(ph) == 1 and ph in self.adj}
        # a verb phrase whose first token is also a negation word ("no" ...) is not expected; a type alias equal to a verb is a collision
        for ph in self.types:
            if ph in self.verbs:
                self.collisions.append((" ".join(ph), "type", "verb"))
        self.max_prep = max([len(p) for p in self.dest | self.src] + [1])

    @staticmethod
    def _set(items) -> frozenset:
        return frozenset(p for p in (_phrase(x) for x in items) if p)

    @classmethod
    def load(cls, path=None) -> "Lexicon":
        p = Path(path) if path else LEXICON_PATH
        raw = p.read_bytes()
        return cls(json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())

    # -- rendering names ---------------------------------------------------------------------------------------
    def type_name(self, t, lang):
        e = self.data["types"].get(t)
        if not e:
            return str(t)
        return e["name_fa"] if lang == "fa" else e["name_en"]

    def verb_name(self, verb, req, lang):
        i = 1 if lang == "fa" else 0
        if verb == "toggle" and req in self.data["req_names"]:
            return self.data["req_names"][req][i]
        return self.data["names"].get(verb, [verb, verb])[i]

    def aliases_for(self, otype) -> list:
        """All normalised alias phrases (EN canonical + extras + plurals, FA seeds) that resolve to `otype`, sorted."""
        return sorted(" ".join(ph) for ph, t in self.types.items() if t == otype)


_DEFAULT = None


def default_lexicon() -> Lexicon:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Lexicon.load()
    return _DEFAULT


# ---------------------------------------------------------------------------------------------------------------
# matching helpers
# ---------------------------------------------------------------------------------------------------------------
def _match_at(toks, i, table, max_len):
    """Longest phrase of `table` starting at i -> (length, key) or (0, None)."""
    for n in range(min(max_len, len(toks) - i), 0, -1):
        k = tuple(toks[i:i + n])
        if k in table:
            return n, k
    return 0, None


def _match_type(toks, i, lex):
    """Longest type alias at i; the LAST token may lose a plural suffix only if the result is a known alias."""
    n, k = _match_at(toks, i, lex.types, lex.max_type)
    if n:
        return n, lex.types[k]
    for n in range(min(lex.max_type, len(toks) - i), 0, -1):
        last = toks[i + n - 1]
        for suf in FA_SUFFIXES:
            if last.endswith(suf) and len(last) > len(suf) + 1:
                k = tuple(toks[i:i + n - 1]) + (last[:-len(suf)],)
                if k in lex.types:
                    return n, lex.types[k]
    return 0, None


def type_spans(toks, lex) -> set:
    """Token positions covered by a type mention (longest match). A type phrase wins over a verb word inside it ("light switch")."""
    cov, i = set(), 0
    while i < len(toks):
        n, _ = _match_type(toks, i, lex)
        if n:
            cov.update(range(i, i + n))
            i += n
        else:
            i += 1
    return cov


def _verb_matches(toks, lex):
    """Non-overlapping left-to-right longest verb phrases -> [(start, end, verb, req, neg, dual)]."""
    out, i = [], 0
    prot = type_spans(toks, lex)
    while i < len(toks):
        n, k = _match_at(toks, i, lex.verbs, lex.max_verb)
        if n and any(j in prot for j in range(i, i + n)):
            n = 0
        if n:
            v, rq, ng = lex.verbs[k]
            out.append((i, i + n, v, rq, ng, k in lex.dual))
            i += n
        else:
            i += 1
    return out


def _has_verb(toks, lex) -> bool:
    for (s, e, v, rq, ng, dual) in _verb_matches(toks, lex):
        if not dual:
            return True
        if all(t in {x[0] for x in lex.det if len(x) == 1} or t in (H, S) for t in toks[:s]):
            return True                                          # a dual word (open/clean...) counts only clause-initially
    return False


def split_clauses(toks, lex):
    """-> (clauses, reasons). See the module docstring for the splitting rules."""
    reasons: list = []
    # 1. multi-word hard separators -> sentinel
    t, i = [], 0
    hard_multi = sorted((p for p in lex.hard if len(p) > 1), key=len, reverse=True)
    while i < len(toks):
        hit = next((p for p in hard_multi if tuple(toks[i:i + len(p)]) == p), None)
        if hit:
            t.append(H)
            i += len(hit)
            continue
        t.append(toks[i])
        i += 1
    # 2. single-token hard separators, complex markers
    u = []
    for i, w in enumerate(t):
        if (w,) in lex.hard and w not in (H, S):
            if w == "بعد" and i + 1 < len(t) and t[i + 1] == "از":
                reasons.append("complex:بعد از")
                u.append(w)
                continue
            u.append(H)
        else:
            u.append(w)
    for i, w in enumerate(u):
        if (w,) in lex.complex:
            reasons.append("complex:" + w)
        if i + 1 < len(u) and (w, u[i + 1]) in lex.complex_bi:
            reasons.append("complex:" + w + "_" + u[i + 1])
    # 3. hard segments
    segs, cur = [], []
    for w in u:
        if w == H:
            segs.append(cur)
            cur = []
        else:
            cur.append(w)
    segs.append(cur)
    # 4. soft splits only when both sides carry a verb
    out = []
    for seg in segs:
        cut, start = [], 0
        for i, w in enumerate(seg):
            is_soft = w == S or (w,) in lex.soft
            is_without = w in ("without", "بدون")
            if not (is_soft or is_without):
                continue
            left, right = seg[start:i], seg[i + (0 if is_without else 1):]
            if _has_verb(left, lex) and _has_verb(right, lex):
                cut.append((start, i))
                start = i + (0 if is_without else 1)
        cut.append((start, len(seg)))
        for a, b in cut:
            part = [w for w in seg[a:b] if w != S]
            if part:
                out.append(part)
    return out, reasons


# ---------------------------------------------------------------------------------------------------------------
# clause parser
# ---------------------------------------------------------------------------------------------------------------
def _parse_clause(toks, lex):
    reasons: list = []
    n = len(toks)
    used = [False] * n
    # -- verbs ---------------------------------------------------------------------------------------------------
    vm = _verb_matches(toks, lex)
    strong = [m for m in vm if not m[5]]
    chosen = strong if strong else vm[:1]
    verbs = {(m[2]) for m in chosen}
    verb, req, vneg = "?", "-", 0
    if len(verbs) > 1:
        reasons.append("multi_verb")
    elif chosen:
        verb = chosen[0][2]
        reqs = {m[3] for m in chosen if m[3] != "-"}
        req = next(iter(reqs)) if len(reqs) == 1 else "-"
        if len(reqs) > 1:
            reasons.append("conflicting_req")
            req = "?"
        vneg = 1 if any(m[4] for m in chosen) else 0
        for m in chosen:
            for j in range(m[0], m[1]):
                used[j] = True
    else:
        reasons.append("no_verb")
    if verb == "toggle" and req == "-":                          # "turn the stove on/off"
        particles = [(j, t) for j, t in enumerate(toks) if not used[j] and t in ("on", "off")]
        pick = None
        for j, t in particles:
            if t == "off":
                pick = (j, "off")
            elif t == "on" and _match_type(toks, j + 1, lex)[0] == 0 and not any(_match_type(toks, k, lex)[0] for k in range(j + 1, min(n, j + 3))):
                pick = (j, "on")
        if pick:
            req = pick[1]
            used[pick[0]] = True
        else:
            req = "?"
            reasons.append("toggle_without_direction")
    if verb not in ("toggle", "?"):
        req = "-"
    # -- negation ------------------------------------------------------------------------------------------------
    nmarks, ascii_marks, nabayad = 0, 0, 0
    for j, t in enumerate(toks):
        if not used[j] and (t,) in lex.neg:
            nmarks += 1
            ascii_marks += 1 if t.isascii() else 0
            nabayad += 1 if t == "نباید" else 0
            used[j] = True
        elif not used[j] and j + 1 < n and (t, toks[j + 1]) in lex.neg and not used[j + 1]:
            nmarks += 1
            used[j] = used[j + 1] = True
    # Persian has negative CONCORD (هرگز ... نکن is one negation); English does not. Two English markers, or "نباید" + a negated verb,
    # are true double negation and fail closed.
    if ascii_marks >= 2 or (vneg and nabayad):
        reasons.append("double_negation")
    neg = 1 if (vneg or nmarks) else 0
    # -- prepositions, adjectives, type mentions --------------------------------------------------------------------
    preps, j = [], 0
    while j < n:
        if used[j]:
            j += 1
            continue
        hit = None
        for ln in range(min(lex.max_prep, n - j), 0, -1):
            ph = tuple(toks[j:j + ln])
            if any(used[j:j + ln]):
                continue
            if ph in lex.src:
                hit = (j, j + ln, "src")
                break
            if ph in lex.dest:
                hit = (j, j + ln, "dest")
                break
        if hit:
            preps.append(hit)
            j = hit[1]
        else:
            j += 1
    pused = {k for (a, b, _) in preps for k in range(a, b)}
    mentions, j = [], 0
    while j < n:
        if used[j] or j in pused:
            j += 1
            continue
        ln, t = _match_type(toks, j, lex)
        if ln and not any(used[j:j + ln]):
            mentions.append({"s": j, "e": j + ln, "type": t, "slot": None, "state": [], "qty": False})
            j += ln
        else:
            j += 1
    for m in mentions:
        for k in range(m["s"], m["e"]):
            used[k] = True
    adjs, j = [], 0
    while j < n:
        if used[j] or j in pused:
            j += 1
            continue
        ln, k = _match_at(toks, j, lex.adj, lex.max_adj)
        if ln:
            adjs.append((j, j + ln, lex.adj[k]))
            j += ln
        else:
            j += 1
    # -- slot assignment ---------------------------------------------------------------------------------------------
    skip_ok = lambda tok: (tok,) in lex.det
    adj_by_end = {e: (s, st) for (s, e, st) in adjs}
    qtys = {j for j, t in enumerate(toks) if (t,) in lex.quant and not used[j]}
    bound_adj = set()
    bound_prep = set()
    for m in mentions:
        j = m["s"]
        while True:
            if j in adj_by_end:                                  # adjective directly before
                s0, st = adj_by_end[j]
                m["state"].append(st)
                bound_adj.add(s0)
                j = s0
                continue
            if j - 1 >= 0 and (j - 1) in qtys:
                m["qty"] = True
                j -= 1
                continue
            if j - 1 >= 0 and skip_ok(toks[j - 1]) and not used[j - 1]:
                j -= 1
                continue
            break
        for pi, (a, b, slot) in enumerate(preps):
            if b == j:
                m["slot"] = slot
                bound_prep.add(pi)
        # accusative / post-nominal adjective (Persian order)
        k = m["e"]
        if k < n and (toks[k],) in lex.acc:
            if m["slot"] is None:
                m["slot"] = "obj"
            used[k] = True
            k += 1
        a_ln, a_k = _match_at(toks, k, lex.adj, lex.max_adj) if k < n else (0, None)
        if a_ln and k not in bound_adj and not any(used[k:k + a_ln]) and (k + a_ln >= n or toks[k + a_ln] not in lex.light):
            m["state"].append(lex.adj[a_k])
            bound_adj.add(k)
            for q in range(k, k + a_ln):
                used[q] = True
    # leftover adjectives that bound to nothing -> unresolved state
    dangling = [a for a in adjs if a[0] not in bound_adj]
    if dangling:
        reasons.append("dangling_adjective")
    # -- pronouns ----------------------------------------------------------------------------------------------------
    pron_obj = False
    mention_starts = {m["s"] for m in mentions}
    for j, t in enumerate(toks):
        if used[j] or (t,) not in lex.pron:
            continue
        nxt = j + 1
        if t in ("this", "that", "these", "those", "این", "آن") and _match_type(toks, nxt, lex)[0] and nxt not in used:
            continue                                             # a determiner, not a pronoun
        after_prep = any(b == j for (_, b, _) in preps)
        if after_prep:
            reasons.append("pronoun_in_slot")
            pron_obj = pron_obj
            for pi, (a, b, slot) in enumerate(preps):
                if b == j:
                    bound_prep.add(pi)
                    mentions.append({"s": j, "e": j + 1, "type": "?", "slot": slot, "state": [], "qty": False})
        else:
            pron_obj = True
    # -- prepositions with nothing resolved after them -----------------------------------------------------------------
    for pi, (a, b, slot) in enumerate(preps):
        if pi in bound_prep:
            continue
        gap = [t for t in toks[b:b + 3] if not skip_ok(t) and (t,) not in lex.quant]
        has_following_noise = bool(gap) and not (b >= n)
        if has_following_noise and slot in ("dest", "src"):
            if verb == "toggle":
                continue
            reasons.append(f"unknown_noun_after_{slot}")
            mentions.append({"s": b, "e": b, "type": "?", "slot": slot, "state": [], "qty": False})
    # -- assemble -------------------------------------------------------------------------------------------------------
    role = {"verb": verb, "req": req, "obj": "-", "ostate": "-", "src": "-", "dest": "-", "dstate": "-", "neg": neg}
    unmarked = [m for m in mentions if m["slot"] in (None, "obj")]
    marked_obj = [m for m in mentions if m["slot"] == "obj"]
    objm = None
    if len(marked_obj) == 1 and len([m for m in unmarked if m["slot"] is None]) == 0:
        objm = marked_obj[0]
    elif len(unmarked) == 1:
        objm = unmarked[0]
    elif len(unmarked) > 1:
        reasons.append("multiple_objects")
        role["obj"] = "?"
    if objm is not None:
        role["obj"] = "?" if (objm["qty"] or objm["type"] == "?") else objm["type"]
        if objm["qty"]:
            reasons.append("quantifier")
        st = sorted(set(objm["state"]))
        role["ostate"] = "+".join(st) if st else "-"
    for slot, key, skey in (("dest", "dest", "dstate"), ("src", "src", None)):
        ms = [m for m in mentions if m["slot"] == slot]
        if len(ms) > 1:
            role[key] = "?"
            reasons.append("multiple_" + slot)
        elif ms:
            m = ms[0]
            role[key] = "?" if (m["qty"] or m["type"] == "?") else m["type"]
            if skey:
                st = sorted(set(m["state"]))
                role[skey] = "+".join(st) if st else "-"
    if role["obj"] == "-":
        if pron_obj:
            role["obj"] = "PRON"
        elif verb != "?":
            role["obj"] = "?"
            reasons.append("no_object")
    if dangling and role["obj"] not in ("?",):
        role["ostate"] = "?"
    return role, mentions, reasons


def _drop_partitive(cl, lex):
    """Persian "یکی از X" ("one of the X") is a determiner, not a source preposition: drop the phrase so 'از' is not read as 'from'."""
    if not lex.partitive:
        return cl
    out, i = [], 0
    while i < len(cl):
        hit = next((p for p in lex.partitive if tuple(cl[i:i + len(p)]) == p), None)
        if hit:
            i += len(hit)
            continue
        out.append(cl[i])
        i += 1
    return out


def _drop_door_of(cl, lex):
    """Persian "در X را باز کن": 'در' + type + 'را' + open/close verb = "open the door of X". Drop the 'در' so X is the (marked) object."""
    if not lex.door_words or not lex.door_verbs:
        return cl
    if not any(v in lex.door_verbs for (_, _, v, _, _, _) in _verb_matches(cl, lex)):
        return cl
    out, i = [], 0
    while i < len(cl):
        if (cl[i],) in lex.door_words:
            n, _ = _match_type(cl, i + 1, lex)
            if n and i + 1 + n < len(cl) and (cl[i + 1 + n],) in lex.acc:
                i += 1
                continue
        out.append(cl[i])
        i += 1
    return out


def parse(text, lex: Lexicon | None = None) -> dict:
    """-> {"roles": [role...], "ok": bool, "reasons": [...], "parser": version, "lexicon": sha}. ok means: >= 1 role, no '?' field, no
    unsupported construct. A not-ok parse may still be used for display/lexical recall but NEVER to establish eligibility."""
    lex = lex or default_lexicon()
    toks = prep(text)
    clauses, reasons = split_clauses(toks, lex)
    roles, prev_types = [], None
    for cl in clauses:
        cl = _drop_door_of(_drop_partitive(cl, lex), lex)
        role, mentions, rs = _parse_clause(cl, lex)
        # W4a / A17, Persian only: "X را بردار و در سینک بگذار" - a clause whose object is elided (reason no_object, no pronoun, no
        # unresolved mention) inherits the PREVIOUS role's object, but only along a carry chain (lexicon elided_object_fa) and never
        # across a negation. English keeps failing closed ("put it" needs an antecedent), so the two languages are not forced equal.
        if (role["obj"] == "?" and "no_object" in rs and _PERSIAN_RE.search("".join(cl)) and roles and not role["neg"]
                and role["verb"] in lex.elide_dst):
            pv = roles[-1]
            if pv["verb"] in lex.elide_src and not pv["neg"] and pv["obj"] not in ("?", "-", "PRON"):
                role["obj"] = pv["obj"]
                rs = [r for r in rs if r != "no_object"]
                reasons.append("fa_elided_object_inherited")
        reasons += rs
        if role["obj"] == "PRON":
            if prev_types is not None and len(prev_types) == 1 and prev_types[0] != "?":
                role["obj"] = prev_types[0]
            else:
                role["obj"] = "?"
                reasons.append("pronoun_without_unique_antecedent")
        roles.append(role)
        prev_types = [m["type"] for m in mentions] if not role["neg"] else None
    ok = bool(roles) and not any(v == "?" for r in roles for k, v in r.items() if k != "neg") \
        and not any(r.startswith(("complex", "multi", "double", "dangling", "quantifier", "pronoun", "unknown", "no_", "toggle", "conflicting"))
                    for r in reasons)
    return {"roles": roles, "ok": ok, "reasons": sorted(set(reasons)), "parser": PARSER_VERSION, "lexicon": lex.sha, "n_tokens": len(toks)}


# ---------------------------------------------------------------------------------------------------------------
# role utilities: signature, tokens, exact equality, key text
# ---------------------------------------------------------------------------------------------------------------
def role_sig(role) -> str:
    return "/".join(str(role[k]) for k in ROLE_FIELDS)


def roles_known(roles) -> bool:
    return bool(roles) and all(r.get(k) != "?" for r in roles for k in ROLE_FIELDS)


def roles_equal(a, b) -> bool:
    """Exact equality on every field, '?' never matches (even '?' == '?'), roles in order, equal count."""
    if not a or not b or len(a) != len(b) or not roles_known(a) or not roles_known(b):
        return False
    return all(all(x[k] == y[k] for k in ROLE_FIELDS) for x, y in zip(a, b))


def concept_tokens(roles) -> list:
    """Clause-indexed, role-bound tokens. Polarity is ALWAYS present (r0.n=0|1); a whole-role signature token binds the fields together."""
    out = []
    for i, r in enumerate(roles):
        out.append(f"r{i}.sig={role_sig(r)}")
        for k in ROLE_FIELDS:
            if r.get(k) not in ("-", None):
                out.append(f"r{i}.{k}={r[k]}")
        out.append(f"r{i}.neg={int(r.get('neg', 0))}")
    return out


def render_key(roles, lex: Lexicon, lang: str) -> str:
    parts = []
    for r in roles:
        v = lex.verb_name(r["verb"], r["req"], lang) if r["verb"] != "?" else "?"
        nm = lambda t: lex.type_name(t, lang) if t not in ("?", "-") else t
        sf = lambda s: "" if s in ("-", None) else s + " "
        if lang == "fa":
            s = ("نباید " if r["neg"] else "")
            s += f"{sf(r['ostate'])}{nm(r['obj'])} را"
            if r["src"] != "-":
                s += f" از {nm(r['src'])}"
            if r["dest"] != "-":
                s += f" در {sf(r['dstate'])}{nm(r['dest'])}"
            s += f" {v}"
        else:
            s = ("not " if r["neg"] else "") + f"{v} {sf(r['ostate'])}{nm(r['obj'])}"
            if r["src"] != "-":
                s += f" from {nm(r['src'])}"
            if r["dest"] != "-":
                s += f" into {sf(r['dstate'])}{nm(r['dest'])}"
        parts.append(s)
    return (" ؛ " if lang == "fa" else " then ").join(parts)


def key_text(roles, lex: Lexicon | None = None) -> str:
    """Bilingual key text rendered from structured roles by fixed templates (never an LLM, never copied from free text)."""
    lex = lex or default_lexicon()
    return render_key(roles, lex, "en") + " / " + render_key(roles, lex, "fa")


def surface_tokens(text) -> list:
    return [t for t in prep(text) if t not in (H, S) and len(t) > 1]


def idf(df: int, n: int) -> float:
    return math.log((n + 1.0) / (df + 1.0)) + 1.0


if __name__ == "__main__":                                       # pragma: no cover
    for q in sys.argv[1:]:
        print(json.dumps(parse(q), ensure_ascii=False, indent=1))
