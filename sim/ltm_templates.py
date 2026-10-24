"""B6 (package spec pkg-1, section 9.1): L3 decomposition templates, selected from the INSTRUCTION, never from a level label.

A template is a short, general "plan shape" for a kind of abstract command ("keep it cold", "serve it hot", "it is too dark",
"I need something to cut with"): an ordered list of tool steps with typed slots, the completion checks, and the non-test items
whose reference plan has that shape.  The planner is shown at most a few instantiated templates (B7 decides how many) as advice;
it is never executed from here.

Contract with the rest of the package (docs/final-test/kitchen_memory/PACKAGE-SPEC.md):
  * ``load(path=None) -> Templates``;  ``Templates.candidates(query) -> [pkg.cand/1]``;
    ``Templates.check(exec_contract, nontest_items) -> [problem]``.
  * A candidate is ``{"rec": pkg.rec/1, "match": 0..1, "why": [...]}``.  ``match`` is this source's own structural match on
    exact keys (intent phrase AND typed slots).  No lexical or dense scoring here (B7 does that once).
  * Knowledge is ENGLISH only (Arash 2026-10-12).  The stored templates, the rendered text and ``rec.text.fa`` ("" always) are
    English.  Persian appears only in ``FA_LEXICON`` below: a PARSING lexicon that recognises the intent of a Persian
    instruction (it is never rendered, never stored in the template file).
  * Retrieval reads only the instruction text, the language, the holding state, the observed types/states and the frozen file.
    It never reads the item id, the reference plan or the goal of the presentation being planned.  (A static selftest checks it.)
  * Authorship: ``domain spec`` (see the file's ``authored_from``).  Non-test support is a separate, computed field.

CLI (offline, no simulator):
  python sim/ltm_templates.py --build [--write]   recompute static texts + non-test support, print a summary
  python sim/ltm_templates.py --show "Serve the potato hot on a plate."
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import ltm_concepts as LC
except ImportError:                      # pragma: no cover
    from sim import ltm_concepts as LC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_PATH = HERE / "step5" / "l3_templates.json"
SPLIT_PATH = ROOT / "docs" / "final-test" / "split_manifest.json"
DATASET_PATH = HERE / "dataset" / "general.json"
SCHEMA_TPLS = "pkg.tpls/1"
SCHEMA_REC = "pkg.rec/1"
MAX_EACH = 4                             # an `each` step is expanded for at most this many bound items

# ====================================================================================================================
# Persian parsing lexicon (instruction understanding ONLY).  Keys are the group / purpose / liquid names used in the JSON.
# Phrases are matched on the normalised token stream (ZWNJ folded to a space, Arabic yeh/kaf folded); a trailing * means
# "any word starting with this".  This is the same kind of resource as step5/lexicon/concepts.json, kept in code because the
# template file must stay English only.
# ====================================================================================================================
FA_LEXICON = {
    "groups": {
        "cold": ["سرد", "سرد*", "خنک*", "تازه*", "یخچال دار", "نگه دار برای بعد سرد"],
        "spoiled": ["خراب*", "گندیده*", "فاسد*", "بیات*", "پلاسیده*", "کپک*", "پوسیده*", "دور بینداز", "دور بریز", "دورش بینداز",
                    "دورش بریز", "دور انداخته شود", "کسی نمی خورد", "کسی نخواهد خورد"],
        "hot": ["داغ*", "گرم*", "سرو*", "حرارت*", "دوباره گرم"],
        "cook": ["بپز*", "پخته*", "پخت*", "سرخ کن*", "بجوشان*", "بیز*"],
        "dark": ["تاریک*", "روشنایی بیشتر", "نور کم", "چیزی نمی بینم", "نور لازم", "روشن کن تاریک"],
        "bright": ["خیلی روشن", "خیلی نور", "چشمم را می زند", "چشمم رو می زند", "چشم زن", "نور زیاد", "نور بیش از حد"],
        "left_open": ["باز مانده", "باز مانده اند", "باز گذاشته", "همه چیز باز", "همه باز", "همه را ببند", "همه رو ببند",
                      "همه چیز را ببند", "مرتب کن", "درها باز"],
        "used": ["استفاده شده", "استفاده کرده ام", "کثیف*", "شسته شود", "شسته شوند", "برای شستن", "باید شسته", "نشسته"],
        "prepare": ["سالاد*", "خرد کن*", "ریز کن*", "ریزش کن", "خردش کن", "مواد را آماده"],
        "meal": ["صبحانه*", "ناهار*", "شام", "میان وعده*", "عصرانه*", "وعده"],
        "drink": ["یک قهوه", "فنجان قهوه", "یک فنجان قهوه", "قهوه می خواهم", "قهوه می خوام", "چیزی برای نوشیدن", "تشنه*",
                  "لیوان آب", "یک لیوان آب", "یک آب", "کمی آب"],
        "shutdown": ["همه چیز را خاموش", "همه چیز رو خاموش", "همه را خاموش", "همه رو خاموش", "دارم می روم بیرون",
                     "دارم می رم بیرون", "از خانه می روم", "از خونه می رم", "قبل از خواب", "قبل از رفتن", "قبل از اینکه بروم"],
    },
    "words": {"salad": ["سالاد*"]},
    "liquids": {
        "coffee": ["قهوه*", "اسپرسو", "لاته"],
        "water": ["آب", "تشنه*"],
        "wine": ["شراب*", "وین"],
    },
    "purposes": {
        "cut": ["چیزی برای بریدن", "برای بریدن", "برای برش", "برای خرد کردن", "چیزی تیز", "با آن ببرم", "چیزی که با آن ببرم"],
        "spread": ["چیزی برای مالیدن", "برای مالیدن", "کره بمالم", "کره را بمالم", "برای کره مالیدن"],
        "soup": ["سوپم", "سوپ خوردن", "برای سوپ", "هم زدن", "همزدن", "چیزی برای هم زدن", "حریره", "ماست خوردن"],
        "eat": ["چیزی برای خوردن", "ماکارونی خوردن", "ماکارونی ام", "ماکارونیم", "اسپاگتی", "نودل", "سالادم را بخورم"],
        "flip": ["برگرداندن", "برگردانم", "پنکیک*", "وارونه کردن", "چیزی برای برگرداندن"],
        "salt": ["نمک لازم", "نمک می خواهد", "نمک می خواد", "نمک بیشتر", "کم نمک", "نمک کم", "بی نمک", "بی مزه", "نمک بزنم"],
        "pepper": ["فلفل لازم", "فلفل می خواهد", "فلفل می خواد", "فلفل بیشتر", "فلفل بزنم"],
        "scrub": ["چیزی برای ساییدن", "برای ساییدن", "سابیدن*", "ساییدن*", "برای شستن ظرف", "چیزی برای شستن ظرف", "اسکاچ"],
        "hands": ["دستم را بشویم", "دستهایم را بشویم", "دستامو بشورم", "دست هایم را بشویم", "دست شویی", "دستم کثیف", "دست هایم کثیف",
                  "دستام کثیف", "دستم چسبناک", "صابون دست"],
        "wipe": ["چیزی برای پاک کردن", "برای پاک کردن", "ریخته", "دستمال کاغذی بخواهم", "خشک کردن دست"],
        "drinkfrom": ["چیزی برای نوشیدن از", "چیزی برای آب خوردن", "برای آب خوردن", "برای نوشیدن"],
    },
}


# ====================================================================================================================
# small helpers
# ====================================================================================================================
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_json(obj) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _self_sha(obj: dict) -> str:
    return sha_json({k: v for k, v in obj.items() if k != "sha256"})


def _module_sha16() -> str:
    try:
        return sha_bytes(Path(__file__).read_bytes())[:16]
    except OSError:                                   # pragma: no cover
        return "0" * 16


_PHRASE_CACHE: dict = {}


def compile_phrase(phrase: str) -> tuple:
    """'wilt*' / 'get rid of' -> tuple of (token, is_prefix) pairs on the same normaliser as the instruction."""
    key = phrase
    if key in _PHRASE_CACHE:
        return _PHRASE_CACHE[key]
    out = []
    words = str(phrase).split()
    for wi, w in enumerate(words):
        star = w.endswith("*")
        w = w.rstrip("*")
        toks = LC.prep(w)
        toks = [t for t in toks if t not in (LC.H, LC.S)] if hasattr(LC, "H") else toks
        for ti, t in enumerate(toks):
            out.append((t, bool(star and ti == len(toks) - 1)))
    res = tuple(out)
    _PHRASE_CACHE[key] = res
    return res


def phrase_in(toks, cp: tuple) -> bool:
    n = len(cp)
    if n == 0 or n > len(toks):
        return False
    for i in range(len(toks) - n + 1):
        ok = True
        for j, (t, pre) in enumerate(cp):
            x = toks[i + j]
            if not (x.startswith(t) if pre else x == t):
                ok = False
                break
        if ok:
            return True
    return False


def first_hit(toks, phrases):
    """The first phrase of `phrases` found in `toks` (as written), or None."""
    for p in phrases:
        if phrase_in(toks, compile_phrase(p)):
            return p
    return None


def mentions(instruction: str, toks=None) -> list:
    """Detector types named in the instruction, in order of mention (A17 lexicon, EN + FA)."""
    toks = toks if toks is not None else LC.prep(instruction)
    lex = LC.default_lexicon()
    out, i = [], 0
    while i < len(toks):
        n, t = LC._match_type(toks, i, lex)
        if n:
            if t not in out:
                out.append(t)
            i += n
        else:
            i += 1
    return out


def _is_type(x) -> bool:
    return isinstance(x, str) and x not in ("", "?", "-")


def _fmt_step(st: dict) -> str:
    a = st["action"]
    s = a
    if st.get("object"):
        s += " " + str(st["object"])
    if st.get("to"):
        s += " to " + str(st["to"])
    if st.get("arg"):
        s += " with " + str(st["arg"]) if a == "fill" else " " + str(st["arg"])
    return s


# ====================================================================================================================
# plan-shape matching (used only for the support computation and `check`; never in candidates)
# ====================================================================================================================
def normalise_plan(plan, hand_effect=None) -> list:
    """Reference/skeleton plans written as pick_and_place(held, R) right after a step that leaves `held` in hand are the same
    plan as put_in(R); fold them so the two spellings compare equal.  `hand_effect` maps tool -> take/release/..."""
    eff = hand_effect or {"pickup": "take", "heat": "take", "cook": "take", "pick_and_place": "take-release",
                          "wash": "take-release", "put_in": "need", "drop": "need"}
    out, hand = [], None
    for st in plan or []:
        a = st.get("action")
        o, t, arg = st.get("object"), st.get("to"), st.get("arg")
        if a == "pick_and_place" and hand and o == hand:
            out.append({"action": "put_in", "object": t, "to": None, "arg": None})
            hand = None
            continue
        out.append({"action": a, "object": o, "to": t, "arg": arg})
        e = eff.get(a)
        if e == "take":
            hand = o
        elif e in ("take-release", "need"):
            hand = None
    return out


class _Shape:
    """Slot-aware matcher of a skeleton against a concrete plan."""

    def __init__(self, cats: dict, slots: dict):
        self.cats, self.slots = cats, slots

    def _ok_value(self, slot: str, val) -> bool:
        spec = self.slots.get(slot, {})
        if spec.get("liquid"):
            return val in ("water", "coffee", "wine")
        if "choices" in spec:
            return val in spec["choices"]
        cat = spec.get("cat")
        return cat is None or val in self.cats.get(cat, [])

    def _term(self, term, val, bind, used) -> bool:
        if term is None:
            return val in (None, "")
        if isinstance(term, str) and term.startswith("?"):
            slot = term[1:]
            if val in (None, ""):
                return False
            if not self._ok_value(slot, val):
                return False
            if slot in bind:
                return bind[slot] == val
            bind[slot] = val
            return True
        return term == val

    def matches(self, skeleton: list, plan: list) -> bool:
        plan = normalise_plan(plan)
        return self._m(skeleton, 0, plan, 0, {})

    def _m(self, sk, i, plan, j, bind) -> bool:
        if i == len(sk):
            return j == len(plan)
        st = sk[i]
        each = st.get("each")
        if st.get("if") and self._m(sk, i + 1, plan, j, bind):       # an `if` step may be absent
            return True
        k, cur, seen = j, dict(bind), []
        while k < len(plan):
            p = plan[k]
            if p["action"] != st["action"]:
                return False
            trial = dict(cur)
            if each:
                trial.pop(each, None)                                  # a fresh binding per repetition
            if not (self._term(st.get("object"), p.get("object"), trial, None)
                    and self._term(st.get("to"), p.get("to"), trial, None)
                    and self._arg_ok(st.get("arg"), p.get("arg"), trial)):
                return False
            nxt = dict(trial)
            if each:
                v = trial.get(each)
                if v in seen:
                    return False
                seen.append(v)
                nxt.pop(each, None)
            k += 1
            if self._m(sk, i + 1, plan, k, nxt):
                return True
            if not each:
                return False
            cur = nxt
        return False

    def _arg_ok(self, term, val, bind) -> bool:
        if term is None:
            return val in (None, "")
        if isinstance(term, str) and term.startswith("?"):
            if val in (None, ""):
                return False
            return self._term(term, val, bind, None)
        return term == val


# ====================================================================================================================
class Templates:
    def __init__(self, data: dict, path=None):
        self.data = data
        self.path = str(path) if path else None
        self.sha = data.get("sha256") or _self_sha(data)
        self.cats = {k: list(v) for k, v in data["categories"].items()}
        self.groups_en = {k: list(v) for k, v in data["groups"].items()}
        self.liquids_en = {k: list(v) for k, v in data["liquids"].items()}
        self.purposes = [dict(p) for p in data["purposes"]]
        self.templates = [dict(t) for t in data["templates"]]
        self.by_id = {t["id"]: t for t in self.templates}
        self.rejected = list(data.get("rejected", []))

    # ----------------------------------------------------------------------------------------------- intent
    def _group_hit(self, toks, name):
        p = first_hit(toks, self.groups_en.get(name, []))
        if p is None:
            p = first_hit(toks, FA_LEXICON["groups"].get(name, []))
        return p

    def _purpose_hits(self, toks) -> list:
        out = []
        for pu in self.purposes:
            p = first_hit(toks, pu.get("en", []))
            if p is None:
                p = first_hit(toks, FA_LEXICON["purposes"].get(pu["id"], []))
            if p is not None:
                out.append((pu, p))
        return out

    def _liquid(self, toks):
        """The liquid the instruction asks for (water | coffee | wine) and the phrase that showed it."""
        best = None
        for name in ("coffee", "wine", "water"):
            p = first_hit(toks, self.liquids_en.get(name, []))
            if p is None:
                p = first_hit(toks, FA_LEXICON["liquids"].get(name, []))
            if p is not None:
                return name, p
        return best

    # ----------------------------------------------------------------------------------------------- binding
    def _bind(self, tpl: dict, ment: list, toks, observed, group_hits: dict, purpose_hits: list):
        """Return (binding, defaulted, ok).  binding: slot -> list[str] (liquid -> [name])."""
        obs_types = {o.get("type") for o in (observed or []) if isinstance(o, dict)}
        binding, defaulted, used, weak = {}, set(), set(), set()
        for sname, spec in tpl.get("slots", {}).items():
            vals = []
            if spec.get("liquid"):
                lq = self._liquid(toks)
                if lq:
                    vals = [lq[0]]
            elif "choices" in spec:
                present = [c for c in spec["choices"] if c in obs_types]
                vals = [present[0] if present else spec["choices"][0]]
                defaulted.add(sname)
                weak.add(sname)
            elif spec.get("from") == "purpose":
                if purpose_hits:
                    vals = [purpose_hits[0][0]["tool"]]
            else:
                catv = self.cats.get(spec.get("cat"), [])
                cand = [t for t in ment if t in catv and t not in used]
                if cand:
                    vals = cand if spec.get("multi") else cand[:1]
                if not vals:
                    for g, dflt in (spec.get("defaults_by_group") or {}).items():
                        if first_hit(toks, [g] + FA_LEXICON.get("words", {}).get(g, [])) is not None:
                            vals = list(dflt)
                            defaulted.add(sname)
                            break
                if not vals and spec.get("from_observed_state"):
                    want = spec["from_observed_state"]
                    vals = [o["type"] for o in (observed or [])
                            if isinstance(o, dict) and o.get("type") in catv and str(o.get("state", "")).lower() == want]
                    vals = list(dict.fromkeys(vals))
                    if vals:
                        defaulted.add(sname)
                        weak.add(sname)
                if not vals and spec.get("default"):
                    vals = [spec["default"]]
                    defaulted.add(sname)
                    weak.add(sname)
            if vals and not spec.get("liquid") and "choices" not in spec:
                used.update(vals)
            binding[sname] = vals
        ok = all(binding.get(s) for s, sp in tpl.get("slots", {}).items() if sp.get("required"))
        req_any = tpl.get("requires_any")
        if ok and req_any:
            ok = any(binding.get(s) and s not in weak for s in req_any)
        return binding, defaulted, ok

    # ----------------------------------------------------------------------------------------------- rendering
    def expand(self, tpl: dict, binding=None) -> list:
        """The skeleton as concrete steps.  binding None = the static, unbound form (placeholders like <food>)."""
        out = []

        def val(term, bound_list, idx=0):
            if term is None:
                return None
            if isinstance(term, str) and term.startswith("?"):
                slot = term[1:]
                if binding is None:
                    return "<%s>" % slot
                lst = binding.get(slot) or []
                if not lst:
                    return "<%s>" % slot
                return lst[idx] if idx < len(lst) else lst[0]
            return term

        for st in tpl["skeleton"]:
            cond = st.get("if")
            if cond and binding is not None and not binding.get(cond):
                continue
            each = st.get("each")
            if each and binding is not None and binding.get(each):
                for idx, _ in enumerate(binding[each][:MAX_EACH]):
                    out.append({"action": st["action"], "object": val(st.get("object"), None, idx),
                                "to": val(st.get("to"), None, idx), "arg": val(st.get("arg"), None, idx)})
            else:
                out.append({"action": st["action"], "object": val(st.get("object"), None),
                            "to": val(st.get("to"), None), "arg": val(st.get("arg"), None)})
        return out

    def render(self, tpl: dict, binding=None, holding=None, extra_each=0) -> str:
        steps = self.expand(tpl, binding)
        body = ", then ".join(_fmt_step(s) for s in steps)
        txt = 'Template "%s": %s. %s' % (tpl["title"], body, tpl["note"])
        if extra_each:
            txt += " Do the same for the other %d item(s) named." % extra_each
        if holding and steps and steps[0]["action"] in ("pickup", "heat", "cook", "pick_and_place") \
                and steps[0].get("object") != holding:
            txt += " You are holding %s now: free the hand first (put_in a receptacle or drop)." % holding
        return txt

    # ----------------------------------------------------------------------------------------------- candidates
    def candidates(self, query: dict, with_steps: bool = False) -> list:
        """pkg.cand/1 for every template whose intent AND typed slots fit the instruction.  Reads only the instruction,
        language, holding state, observed types/states and the frozen template file."""
        instr = str(query.get("instruction") or "")
        toks = LC.prep(instr)
        toks = [t for t in toks if t not in (getattr(LC, "H", None), getattr(LC, "S", None))]
        ment = mentions(instr, LC.prep(instr))
        for t in list(query.get("types") or []):
            if _is_type(t) and t not in ment:
                ment.append(t)
        for r in ((query.get("parse") or {}).get("roles") or []):
            for f in ("obj", "src", "dest", "req"):
                t = r.get(f) if isinstance(r, dict) else None
                if _is_type(t) and t not in ment:
                    ment.append(t)
        observed = query.get("observed") or []
        holding = query.get("holding")
        holding = holding if _is_type(holding) else None
        purpose_hits = self._purpose_hits(toks)
        group_cache = {}
        out = []
        for prio, tpl in enumerate(self.templates):
            intent = tpl.get("intent", {})
            blocked = False
            for g in intent.get("not", []):
                if g not in group_cache:
                    group_cache[g] = self._group_hit(toks, g)
                blocked = blocked or group_cache[g] is not None
            if blocked:
                continue
            hit, why_intent = False, []
            for g in intent.get("any", []):
                if g not in group_cache:
                    group_cache[g] = self._group_hit(toks, g)
                if group_cache[g] is not None:
                    hit = True
                    why_intent.append("intent:%s" % g)
            if intent.get("purpose") and purpose_hits:
                hit = True
                why_intent.append("intent:purpose-%s" % purpose_hits[0][0]["id"])
            if not hit:
                continue
            gh = {g: v for g, v in group_cache.items() if v is not None}
            binding, defaulted, ok = self._bind(tpl, ment, toks, observed, gh, purpose_hits)
            if not ok:
                continue
            if not self.expand(tpl, binding):
                continue
            slots = tpl.get("slots", {})
            counted = [s for s, sp in slots.items() if "choices" not in sp]
            if counted:
                score = sum((0.5 if s in defaulted else 1.0) for s in counted if binding.get(s)) / float(len(counted))
            else:
                score = 1.0
            match = round(0.5 * 1.0 + 0.5 * score, 6)
            extra = 0
            for st in tpl["skeleton"]:
                if st.get("each") and binding.get(st["each"]) and len(binding[st["each"]]) > MAX_EACH:
                    extra = len(binding[st["each"]]) - MAX_EACH
            text = self.render(tpl, binding, holding, extra)
            steps = self.expand(tpl, binding)
            bound_types = []
            for s, sp in slots.items():
                if sp.get("liquid"):
                    continue
                for v in binding.get(s, []):
                    if v not in bound_types:
                        bound_types.append(v)
            lit_types = []
            for st in steps:
                for f in ("object", "to"):
                    v = st.get(f)
                    if _is_type(v) and not v.startswith("<") and v not in lit_types:
                        lit_types.append(v)
            ops = list(dict.fromkeys(st["action"] for st in steps))
            rec = {
                "schema": SCHEMA_REC,
                "id": "template/*/" + tpl["id"].split(":", 1)[-1],
                "component": "template", "kitchen": "*",
                "keys": {"types": sorted(set(bound_types) | set(lit_types)), "ops": ops,
                         "rels": ["requires:%s" % t for t in lit_types[:6]]},
                "lex": {"en": " ".join([tpl["title"]] + [p for p in (group_cache.get(g) for g in intent.get("any", [])) if p]),
                        "fa": ""},
                "text": {"en": text, "fa": ""},
                "provenance": {"source": "spec:%s" % self.sha[:16], "evidence": [],
                               "authored_by": "script:ltm_templates@%s" % _module_sha16(),
                               "support": list(tpl.get("support", []))},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": prio,
            }
            why = list(why_intent) + ["type:%s" % t for t in bound_types] + ["op:%s" % o for o in ops]
            cand = {"rec": rec, "match": match, "why": why}
            if with_steps:                                  # diagnostics only (early check); never part of pkg.cand/1
                cand["steps"] = steps
            out.append(cand)
        fired = {c["rec"]["id"] for c in out}
        gone = set()
        for c in out:
            tpl = self.templates[c["rec"]["prio"]]
            for sup in tpl.get("supersedes", []):
                gone.add("template/*/" + sup.split(":", 1)[-1])
        out = [c for c in out if c["rec"]["id"] not in gone]
        out.sort(key=lambda c: (-c["match"], c["rec"]["prio"], c["rec"]["id"]))
        return out

    def fingerprint(self) -> dict:
        """Hashes that a freeze should pin: the template file, this module, and the Persian parsing lexicon."""
        return {"templates_file_sha": self.sha, "module_sha16": _module_sha16(),
                "fa_lexicon_sha": sha_json(FA_LEXICON), "version": self.data.get("version")}

    def static_records(self) -> list:
        """The un-instantiated record of every template (for id / leakage / schema audits)."""
        recs = []
        for prio, tpl in enumerate(self.templates):
            steps = self.expand(tpl, None)
            lit = []
            for st in steps:
                for f in ("object", "to"):
                    v = st.get(f)
                    if _is_type(v) and not v.startswith("<") and v not in lit:
                        lit.append(v)
            recs.append({
                "schema": SCHEMA_REC, "id": "template/*/" + tpl["id"].split(":", 1)[-1], "component": "template",
                "kitchen": "*",
                "keys": {"types": sorted(lit), "ops": list(dict.fromkeys(s["action"] for s in steps)),
                         "rels": ["requires:%s" % t for t in lit[:6]]},
                "lex": {"en": tpl["title"], "fa": ""}, "text": {"en": self.render(tpl, None), "fa": ""},
                "provenance": {"source": "spec:%s" % self.sha[:16], "evidence": [],
                               "authored_by": "script:ltm_templates@%s" % _module_sha16(),
                               "support": list(tpl.get("support", []))},
                "created": 0, "recency": 0, "confidence": 1.0, "privileged": False, "prio": prio})
        return recs

    # ----------------------------------------------------------------------------------------------- shape + support
    def shape_matches(self, tpl_id: str, plan: list) -> bool:
        tpl = self.by_id[tpl_id]
        sh = _Shape(self.cats, tpl.get("slots", {}))
        if sh.matches(tpl["skeleton"], plan):
            return True
        return any(sh.matches(alt, plan) for alt in tpl.get("alt_skeletons", []))

    def compute_support(self, items: list) -> dict:
        """tpl id -> {"support": [...], "plan_only": [...]}.  An item supports a template when its reference plan has the
        template's shape AND the template fires on its English instruction; ``plan_only`` lists plan matches that do not fire.
        `items`: dicts with id, en, plan (NON-TEST items only; the caller guarantees it)."""
        res = {t["id"]: {"support": [], "plan_only": []} for t in self.templates}
        for it in items:
            fired = {c["rec"]["id"] for c in self.candidates({"instruction": it.get("en", ""), "lang": "en",
                                                              "holding": None, "observed": [], "types": []})}
            for t in self.templates:
                if not self.shape_matches(t["id"], it.get("plan") or []):
                    continue
                rid = "template/*/" + t["id"].split(":", 1)[-1]
                (res[t["id"]]["support"] if rid in fired else res[t["id"]]["plan_only"]).append(it["id"])
        return res

    # ----------------------------------------------------------------------------------------------- check
    def check(self, exec_contract, nontest_items, validator=None, nontest_ids=None) -> list:
        """Problems (empty list = clean).  `exec_contract`: a pkg.exec/1 dict or a path to it.  `nontest_items`: list of item
        dicts (or id -> dict) used to re-verify the support; `nontest_ids`: the manifest's non-test id set (loaded from the
        split manifest when omitted).  `validator`: an optional callable (steps, types) -> {"ok", "errors"} such as
        validator_v2.validate wrapped; the contract-level check always runs."""
        probs = []
        if isinstance(exec_contract, (str, Path)):
            exec_contract = json.loads(Path(exec_contract).read_text(encoding="utf-8"))
        tools = (exec_contract or {}).get("tools") or {}
        items = nontest_items.values() if isinstance(nontest_items, dict) else (nontest_items or [])
        by_id = {it["id"]: it for it in items}
        if nontest_ids is None:
            nontest_ids = set(load_manifest()["nontest"])
        test_like = set()
        try:
            m = load_manifest()
            test_like = set(m["test"]) | set(m["test_uncapped_not_scored"])
        except Exception:                                  # pragma: no cover
            pass
        vocab = set(LC.default_lexicon().types.values())

        def prob(tid, code, detail):
            probs.append({"tpl": tid, "code": code, "detail": detail})

        seen = set()
        for t in self.templates:
            tid = t["id"]
            if tid in seen:
                prob(tid, "DUPLICATE_ID", "id repeated")
            seen.add(tid)
            for need in ("title", "note", "intent", "slots", "skeleton", "checks"):
                if need not in t:
                    prob(tid, "SCHEMA", "missing field %s" % need)
            if re.search(r"[؀-ۿ]", json.dumps(t, ensure_ascii=False)):
                prob(tid, "NOT_ENGLISH", "Persian text inside a stored template")
            # representative instantiation: first value of every slot, two for `each`
            rep = {}
            for s, sp in t.get("slots", {}).items():
                if sp.get("liquid"):
                    rep[s] = ["coffee"]
                elif "choices" in sp:
                    rep[s] = [sp["choices"][0]]
                else:
                    vals = self.cats.get(sp.get("cat"), [])
                    if sp.get("from") == "purpose":
                        vals = [p["tool"] for p in self.purposes][:2]
                    rep[s] = vals[:2] if sp.get("multi") else vals[:1]
            steps = self.expand(t, rep)
            problems = _validate_steps(steps, tools, vocab)
            for pr in problems:
                prob(tid, "SKELETON_INVALID", pr)
            if validator is not None:
                try:
                    r = validator(steps, sorted(vocab))
                    if not r.get("ok", False):
                        prob(tid, "SKELETON_INVALID_V2", json.dumps(r.get("errors", []))[:200])
                except Exception as e:                    # pragma: no cover
                    prob(tid, "VALIDATOR_ERROR", repr(e)[:160])
            # alternatives must also be valid under the contract
            for k, alt in enumerate(t.get("alt_skeletons", [])):
                tt = dict(t, skeleton=alt)
                for pr in _validate_steps(self.expand(tt, rep), tools, vocab):
                    prob(tid, "ALT_SKELETON_INVALID", "alt %d: %s" % (k, pr))
            # a static text that disagrees with the skeleton is stale
            if t.get("text", {}).get("en") != self.render(t, None):
                prob(tid, "TEXT_STALE", "text.en differs from the rendered skeleton (re-run --build)")
            sup = t.get("support", [])
            if not sup:
                prob(tid, "SUPPORT_EMPTY", "no non-test support")
            for i in sup:
                if i in test_like:
                    prob(tid, "SUPPORT_TEST_ITEM", "a test-set id is cited as support")
                elif i not in nontest_ids:
                    prob(tid, "SUPPORT_NOT_NONTEST", "id %s is not a non-test item" % i)
                elif i not in by_id:
                    prob(tid, "SUPPORT_UNVERIFIED", "item %s not supplied to check()" % i)
                elif not self.shape_matches(tid, by_id[i].get("plan") or []):
                    prob(tid, "SUPPORT_NO_MATCH", "reference plan of %s no longer has this shape" % i)
        return probs


def _validate_steps(steps: list, tools: dict, vocab: set) -> list:
    """Contract-level validation of a concrete step list (signatures, types, hand trace). Returns a list of strings."""
    errs, hand = [], None
    if not steps:
        return ["empty skeleton"]
    if len(steps) > 12:
        errs.append("more than 12 steps")
    for i, st in enumerate(steps):
        a = st["action"]
        spec = tools.get(a)
        if spec is None:
            errs.append("step %d: unknown tool %s" % (i, a))
            continue
        for f, rule in (("object", spec.get("object", "none")), ("to", spec.get("to", "none"))):
            v = st.get(f)
            if rule == "type":
                if not v:
                    errs.append("step %d: %s needs %s" % (i, a, f))
                elif v not in vocab:
                    errs.append("step %d: %s %r is not a detector type" % (i, f, v))
            elif rule == "opt":
                if v and v not in vocab:
                    errs.append("step %d: %s %r is not a detector type" % (i, f, v))
            elif v:
                errs.append("step %d: %s takes no %s" % (i, a, f))
        arg_rule = spec.get("arg", "none")
        if arg_rule not in (None, "none", False):
            if st.get("arg") not in ("water", "coffee", "wine"):
                errs.append("step %d: %s arg %r is not a liquid" % (i, a, st.get("arg")))
        elif st.get("arg"):
            errs.append("step %d: %s takes no arg" % (i, a))
        obj = st.get("object")
        if spec.get("requires_empty_hand") and hand and hand != obj:
            errs.append("step %d: %s %s while holding %s" % (i, a, obj, hand))
        if spec.get("requires_held") and not hand:
            errs.append("step %d: %s needs a held object but the hand is empty" % (i, a))
        eff = spec.get("hand", "none")
        if eff == "take":
            hand = obj
        elif eff in ("take-release", "release"):
            hand = None
    return errs


# ====================================================================================================================
def load_manifest() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def load(path=None) -> Templates:
    p = Path(path) if path else DEFAULT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA_TPLS:
        raise ValueError("not a %s file: %s" % (SCHEMA_TPLS, p))
    return Templates(data, p)


def load_nontest_items() -> list:
    """The non-test items of the general dataset (id, level, en, fa, plan).  Test items are never retained."""
    m = load_manifest()
    keep = set(m["nontest"])
    g = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    out = []
    for it in g["items"]:
        if it["id"] in keep:
            out.append({"id": it["id"], "level": it.get("level"), "en": it.get("en"), "fa": it.get("fa"),
                        "plan": it.get("plan") or []})
    return out


def build(write: bool = False, path=None) -> dict:
    """Recompute the static `text` and the non-test `support` of every template, then partition the pool: a template ships
    (`templates`) only when at least one NON-TEST item backs it; otherwise it is archived in `rejected` with the reason.
    support = items whose reference plan has the template's shape AND on whose English instruction the template fires
    (strength "strong"); failing that, plan-shape-only items (strength "weak"); failing that, the template is rejected.
    Idempotent: archived templates are re-evaluated on every run."""
    p = Path(path) if path else DEFAULT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    pool = list(data["templates"]) + [r["template"] for r in data.get("rejected", [])]
    for i, t in enumerate(pool):
        t.setdefault("order", i)
    pool.sort(key=lambda t: t["order"])
    work = dict(data, templates=pool, rejected=[])
    tp = Templates(work, p)
    items = load_nontest_items()
    sup = tp.compute_support(items)
    active, rejected = [], []
    for t in pool:
        t["text"] = {"en": tp.render(tp.by_id[t["id"]], None)}
        t.pop("support", None)
        t.pop("support_plan_only", None)
        t.pop("support_strength", None)
        strong, weak = sup[t["id"]]["support"], sup[t["id"]]["plan_only"]
        if strong:
            t["support"], t["support_strength"] = strong, "strong"
            t["support_plan_only"] = weak
            active.append(t)
        else:
            why = ("no non-test item has this plan shape" if not weak else
                   "plan-shape-only support is not enough: no non-test item both has this shape and fires the template (Astra H-B6)")
            rejected.append({"id": t["id"], "reason": why, "template": t})
    data["templates"], data["rejected"] = active, rejected
    data["knowledge_class"] = ("functional affordance, triggered by the instruction (what a kind of request needs); NOT a "
                               "usual-location or storage-convention claim, which belongs to the briefing alone")
    data["support_rule"] = ("a template ships only if >=1 NON-TEST item has its plan shape AND the template fires on that item's "
                            "English instruction (strong support). Weak (plan-shape-only) support is not accepted. Support is an "
                            "offline audit field: it never reaches retrieval ranking.")
    data["sha256"] = _self_sha(data)
    summary = {t["id"]: {"strength": t["support_strength"], "support": len(t["support"])} for t in active}
    summary.update({r["id"]: {"strength": "REJECTED", "support": 0} for r in rejected})
    if write:
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8", newline=chr(10))
    return {"summary": summary, "sha256": data["sha256"], "n_items": len(items), "active": len(active),
            "rejected": [r["id"] for r in rejected]}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--build" in argv:
        r = build(write="--write" in argv)
        print(json.dumps(r, indent=1))
        return 0
    if "--show" in argv:
        q = argv[argv.index("--show") + 1]
        tp = load()
        for c in tp.candidates({"instruction": q, "lang": "en", "holding": None, "observed": [], "types": []}):
            print(c["match"], c["rec"]["id"], c["why"])
            print("   ", c["rec"]["text"]["en"])
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
