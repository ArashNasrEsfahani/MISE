"""Step 5 / A3 - failure diagnosis, lessons and procedures (docs/step5-plan.md sections 2 R7, 7.1-7.6).

What lives here
  * an observable CAUSE TAXONOMY and a rule-based classifier over working-memory events (no LLM);
  * the LOCAL FAILURE KEY (cause, verb, otype, rtype, state seen) built only from events at or before the failed attempt;
  * the Gemma HYPOTHESIS interface: evidence-packet builder, prompt builder, response parser and a strict validator.
    Nothing in this module contacts a model: the caller injects `Hypothesizer.propose(packet) -> dict`;
  * LESSON records and their lifecycle (tentative -> supported -> contradicted -> retired), scored from immutable
    application_log rows, counted in distinct UNITS, never a prohibition, never a habit count;
  * PROCEDURES: admission from the executed sequence only under the declared privileged terminal verdict AND an
    executed plan that supports it, success / failed-reuse units, status derived from the unit counts;
  * the false-lesson-rate bookkeeping (Wilson, supported-set FLR, stratified Horvitz-Thompson for all-created);
  * `LessonsExtension` - the callable handed to `ltm.end_episode(wm, extensions=[...])`; all writes join the SAME
    transaction as the episode, so a rollback loses nothing and a replay is a no-op.

Honesty (plan, intro): the privileged goal verdict (`ctx.goal_ok`) is read in exactly three places here - procedure
admission, failed-reuse counting, and the `plan_goal_unmet` label. It is never used to score a lesson.

Pure standard library; imports no simulator, opens no socket, contacts no model.
"""
from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter

try:
    from ltm_events import ev_hash
    from ltm_store import SIM_ID_RE, TOKEN_RE, WhitelistError, canon, is_privileged_name, sha256_hex
except ImportError:                     # pragma: no cover
    from sim.ltm_events import ev_hash
    from sim.ltm_store import SIM_ID_RE, TOKEN_RE, WhitelistError, canon, is_privileged_name, sha256_hex

# ------------------------------------------------------------------------------------------------ declared tables
CAUSES = ("not_detected", "action_refused", "unreachable", "wrong_target", "postcondition_unconfirmed",
          "budget_exhausted", "plan_goal_unmet", "unknown")
ACTIONS = ("try_other_instance", "change_pose", "approach_closer", "crouch", "open_first", "search_elsewhere", "none")
# step-4 raw code -> taxonomy code (plan 7.3). Anything else is `unknown`.
RAW_CAUSE_MAP = {
    "not_detected": "not_detected", "action_refused": "action_refused", "unreachable": "unreachable",
    "wrong_target": "wrong_target", "state_unrecognised": "postcondition_unconfirmed",
    "postcondition_unconfirmed": "postcondition_unconfirmed", "retry_exhausted": "budget_exhausted",
    "primitive_cap": "budget_exhausted", "budget_exhausted": "budget_exhausted",
}
# DECLARED engineering table (not a tunable threshold): which remedy actions are sensible for which observable cause.
# Causes with an empty set never produce a lesson (budget_exhausted, plan_goal_unmet, unknown).
ALLOWED_ACTIONS = {
    "not_detected": frozenset({"search_elsewhere", "change_pose", "approach_closer", "crouch", "open_first"}),
    "action_refused": frozenset({"change_pose", "approach_closer", "try_other_instance", "crouch", "open_first"}),
    "unreachable": frozenset({"approach_closer", "change_pose", "try_other_instance", "search_elsewhere"}),
    "wrong_target": frozenset({"try_other_instance", "change_pose", "approach_closer"}),
    "postcondition_unconfirmed": frozenset({"change_pose", "approach_closer", "crouch"}),
    "budget_exhausted": frozenset(), "plan_goal_unmet": frozenset(), "unknown": frozenset(),
}
# remedies whose effect depends on where things are / how the agent stands: flagged for revalidation after a world reset
GEOMETRY_ACTIONS = frozenset({"change_pose", "approach_closer", "crouch", "try_other_instance"})
# STRUCTURAL caps (not tuned thresholds; named here instead of the registry because they bound data size, not behaviour)
MAX_RECOVERY_K = 3          # step 4's existing retry budget: recovery attempts are k = 1..3
PACKET_MAX_EVENTS = 40      # plan 7.2 evidence packet cap; truncation policy in `build_packet`
HYP_MAX_EVIDENCE = 8
BODY_MAX_STEPS = 64
LESSON_ROLES = ("tentative", "supported", "contradicted", "retired")
STATE_WORDS = ("open", "closed", "?")
PACKET_FIELDS = {
    "step_start": ("i", "verb", "otype", "to", "pre"), "attempt": ("i", "k", "recovery", "source", "ok", "cause"),
    "refusal": ("verb", "otype", "rtype", "i", "k", "cause"), "step_outcome": ("i", "ok", "cause", "under", "retries", "prims"),
    "open": ("rtype", "ok", "by_me", "state_seen"), "close": ("rtype", "ok", "by_me", "state_seen"),
    "inspection": ("rtype", "verdict", "support", "range_m", "crouched", "detector_rev"),
    "pickup": ("otype", "ok"), "put": ("otype", "rtype", "ok", "resighted"), "drop": ("otype", "ok", "resighted"),
    "sighting": ("otype", "state", "conf", "in_rtype", "in_basis"), "receptacle_seen": ("rtype",),
    "contradiction": ("otype", "views"), "application": ("record", "i", "k"),
}
PACKET_KEEP_KINDS = {k for k in PACKET_FIELDS if k not in ("sighting", "receptacle_seen")}


class LessonError(RuntimeError):
    """A rule of the lesson lifecycle was violated (for example a lesson application in a non-practice episode)."""


# ------------------------------------------------------------------------------------------------ small helpers
def _h(*parts) -> str:
    return sha256_hex("|".join(str(p) for p in parts))[:16]


def _tok(x, default="?") -> str:
    if isinstance(x, str) and x and x.isascii() and TOKEN_RE.match(x) and "|" not in x and not SIM_ID_RE.search(x):
        return x
    return default


def _clean_raw(x):
    return x if isinstance(x, str) and x.strip() not in ("", "?", "none") else None


def normalise_cause(raw) -> str:
    """Raw step-4 code (or None) -> taxonomy code. Unrecognised -> `unknown` (never guessed)."""
    return RAW_CAUSE_MAP.get(raw, "unknown") if isinstance(raw, str) else "unknown"


def _flat(d) -> dict:
    """Scalar-valued top level only (the store's json depth/size limits must never abort a consolidation)."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, bool) or v is None:
                out[k] = v
            elif isinstance(v, int):
                out[k] = v
            elif isinstance(v, float) and math.isfinite(v):
                out[k] = round(v, 3)
            elif isinstance(v, str) and _tok(v, None) is not None:
                out[k] = v
    return out


def make_conditions(cause, verb, otype, rtype, state) -> dict:
    cause = cause if cause in CAUSES else "unknown"
    st = state if state in STATE_WORDS else "?"
    return {"cause": cause, "verb": _tok(verb), "otype": _tok(otype), "rtype": _tok(rtype), "state": st}


def cond_key_of(conditions: dict) -> str:
    return "ck:" + _h(canon({k: conditions[k] for k in ("cause", "verb", "otype", "rtype", "state")}))


def lesson_id(kitchen: str, cond_key: str) -> str:
    return "ls:" + _h(kitchen, cond_key)


# ------------------------------------------------------------------------------------------------ step segmentation
def step_index(wm) -> dict:
    """i -> {start, attempts{k: ev}, refusals{k: ev}, outcome, apps[], pos[]}. Positional events (open, sighting, ...)
    belong to the step that was running when they happened; typed events carry their own `i`."""
    steps: dict = {}
    cur = None

    def slot(i):
        return steps.setdefault(i, {"start": None, "attempts": {}, "refusals": {}, "outcome": None, "apps": [], "pos": []})

    for e in wm.events:
        kd = e["kind"]
        if kd == "step_start":
            i = e.get("i")
            if i is None:
                i = (max(steps) + 1) if steps else 0
            slot(i)["start"] = e
            cur = i
        elif kd in ("attempt", "refusal", "step_outcome", "application"):
            i = e.get("i", cur)
            if i is None:
                continue
            s = slot(i)
            if kd == "attempt":
                s["attempts"][e.get("k", 0)] = e
            elif kd == "refusal":
                s["refusals"][e.get("k", 0)] = e
            elif kd == "step_outcome":
                s["outcome"] = e
            else:
                s["apps"].append(e)
            cur = i if kd != "application" else cur
        elif kd in PACKET_FIELDS and cur is not None:
            steps[cur]["pos"].append(e)
    return steps


def all_steps_ok(steps: dict) -> bool:
    """True only if there is at least one step and EVERY step has a final outcome that is exactly ok (plan, Astra A3-3)."""
    if not steps:
        return False
    return all(s["start"] is not None and s["outcome"] is not None and s["outcome"].get("ok") is True for s in steps.values())


# ------------------------------------------------------------------------------------------------ diagnosis
def _attempt_cause(s: dict, i: int, k: int, a: dict):
    """Precedence (Astra A3-1): the attempt's own cause, else the refusal of the same (i, k), else - for the last attempt
    of a step that finally failed - the step outcome's cause. Returns (raw, taxonomy)."""
    ref = s["refusals"].get(k)
    raw = _clean_raw(a.get("cause")) or (_clean_raw(ref.get("cause")) if ref else None)
    last = k == max(s["attempts"]) if s["attempts"] else True
    if raw is None and last and s["outcome"] is not None and s["outcome"].get("ok") is False:
        raw = _clean_raw(s["outcome"].get("cause"))
    return raw, normalise_cause(raw)


def _failure_context(s: dict, upto_ev: int, rtype_hint):
    """verb / otype / rtype / state-seen from events at or before `upto_ev` ONLY (never from later recovery)."""
    st = s["start"]
    verb = (st or {}).get("verb") or "?"
    otype = (st or {}).get("otype") or "?"
    rtype = _clean_raw(rtype_hint)
    state = None
    for e in s["pos"]:
        if e["ev"] > upto_ev:
            break
        if e["kind"] in ("open", "close", "inspection") and e.get("rtype"):
            if rtype is None or e["rtype"] == rtype:
                rtype = rtype or e["rtype"]
                if e["kind"] in ("open", "close") and e.get("state_seen") in STATE_WORDS:
                    state = e["state_seen"]
    if rtype is None:
        rtype = _clean_raw((st or {}).get("to"))
    if state is None:
        pre = (st or {}).get("pre") or {}
        for key in ("container", "state_seen"):
            if pre.get(key) in STATE_WORDS:
                state = pre[key]
                break
    return verb, otype, rtype or "?", state or "?"


def diagnose(wm, goal_ok=None) -> dict:
    """Deterministic diagnosis of one episode. Returns
      failures: [{i, k, ev, raw, cause, under, conditions, cond_key, step_level}]  (one per failed attempt, plus a
                 separate `budget_exhausted` entry for a step that ended by exhausting its budget)
      plan_goal_unmet: bool   (every step ok AND the privileged verdict is False - privileged use 3)
    """
    steps = step_index(wm)
    fails = []
    for i in sorted(steps):
        s = steps[i]
        so = s["outcome"]
        under = _flat(so.get("under")) if so else {}
        found = False
        for k in sorted(s["attempts"]):
            a = s["attempts"][k]
            if a.get("ok") is not False:
                continue
            raw, cause = _attempt_cause(s, i, k, a)
            ref = s["refusals"].get(k)
            verb, otype, rtype, state = _failure_context(s, a["ev"], ref.get("rtype") if ref else None)
            if ref and ref.get("verb"):
                verb = verb if verb != "?" else ref["verb"]
            cond = make_conditions(cause, verb, otype, rtype, state)
            fails.append({"i": i, "k": k, "ev": a["ev"], "raw": raw, "cause": cause, "under": under, "conditions": cond,
                          "cond_key": cond_key_of(cond), "step_level": False})
            found = True
        if so is not None and so.get("ok") is False:
            raw = _clean_raw(so.get("cause"))
            cause = normalise_cause(raw)
            if not found or cause == "budget_exhausted":
                ref = s["refusals"].get(max(s["refusals"])) if s["refusals"] else None
                verb, otype, rtype, state = _failure_context(s, so["ev"], ref.get("rtype") if ref else None)
                cond = make_conditions(cause, verb, otype, rtype, state)
                fails.append({"i": i, "k": max(s["attempts"]) if s["attempts"] else 0, "ev": so["ev"], "raw": raw,
                              "cause": cause, "under": under, "conditions": cond, "cond_key": cond_key_of(cond),
                              "step_level": True})
    return {"steps": steps, "failures": fails,
            "plan_goal_unmet": bool(goal_ok is False and all_steps_ok(steps))}


def failure_key_at(wm, i: int):
    """For the adapter: the local failure key of the most recent failed attempt of step `i` using the events seen so far
    (the key a recovery attempt k+1 is looked up with). None when the step has no failed attempt yet."""
    steps = step_index(wm)
    s = steps.get(i)
    if not s:
        return None
    best = None
    for k in sorted(s["attempts"]):
        a = s["attempts"][k]
        if a.get("ok") is False:
            raw, cause = _attempt_cause(s, i, k, a)
            ref = s["refusals"].get(k)
            verb, otype, rtype, state = _failure_context(s, a["ev"], ref.get("rtype") if ref else None)
            best = make_conditions(cause, verb, otype, rtype, state)
    return best


# ------------------------------------------------------------------------------------------------ evidence packet
def build_packet(wm, failure: dict, steps: dict | None = None) -> dict:
    """Numbered WM events of the failing step up to the failed attempt (enumerated fields only; no task text).
    Cap PACKET_MAX_EVENTS; truncation keeps every non-sighting event (or its first + last N-1 if even those overflow),
    then fills the remainder with the MOST RECENT sightings / receptacle sightings. Deterministic."""
    steps = steps or step_index(wm)
    s = steps[failure["i"]]
    evs = []
    if s["start"] is not None:
        evs.append(s["start"])
    evs += [e for e in s["pos"]]
    evs += [s["attempts"][k] for k in sorted(s["attempts"])]
    evs += [s["refusals"][k] for k in sorted(s["refusals"])]
    if s["outcome"] is not None:
        evs.append(s["outcome"])
    evs = sorted({e["ev"]: e for e in evs if e["ev"] <= failure["ev"]}.values(), key=lambda e: e["ev"])
    truncated = False
    if len(evs) > PACKET_MAX_EVENTS:
        truncated = True
        keep = [e for e in evs if e["kind"] in PACKET_KEEP_KINDS]
        soft = [e for e in evs if e["kind"] not in PACKET_KEEP_KINDS]
        if len(keep) > PACKET_MAX_EVENTS:
            keep = keep[:1] + keep[-(PACKET_MAX_EVENTS - 1):]
            soft = []
        room = PACKET_MAX_EVENTS - len(keep)
        keep += soft[-room:] if room > 0 else []
        evs = sorted(keep, key=lambda e: e["ev"])
    out = []
    for e in evs:
        rec = {"ev": e["ev"], "kind": e["kind"]}
        for f in PACKET_FIELDS.get(e["kind"], ()):
            if f in e and e[f] is not None:
                v = e[f]
                rec[f] = _flat(v) if isinstance(v, dict) else (round(v, 2) if isinstance(v, float) else v)
        out.append(rec)
    return {"schema": "s5.dx/1", "step": failure["i"], "k": failure["k"], "cause": failure["cause"],
            "conditions": dict(failure["conditions"]), "under": dict(failure["under"]), "events": out,
            "truncated": truncated}


def packet_ids(packet: dict) -> set:
    return {e["ev"] for e in packet["events"]}


# ------------------------------------------------------------------------------------------------ Gemma hypothesis
class Hypothesizer:
    """Interface. Implementations must return a dict (see `validate_hypothesis`). The default does nothing."""

    def propose(self, packet: dict) -> dict:           # pragma: no cover - interface
        return {"insufficient_evidence": True}


class NullHypothesizer(Hypothesizer):
    pass


def render_packet(packet: dict) -> str:
    lines = []
    for e in packet["events"]:
        extra = " ".join(f"{k}={json.dumps(v, sort_keys=True) if isinstance(v, dict) else v}"
                         for k, v in e.items() if k not in ("ev", "kind"))
        lines.append(f"[{e['ev']}] {e['kind']} {extra}".rstrip())
    return "\n".join(lines)


def build_prompt(packet: dict) -> str:
    """The prompt for the optional hypothesis. English, no task text, no simulator fields (the packet has none)."""
    allowed = sorted(ALLOWED_ACTIONS.get(packet["cause"], ()))
    c = packet["conditions"]
    return (
        "A household robot step failed. The observable cause code is fixed: "
        f"{packet['cause']}.\n"
        f"Step {packet['step']} (verb={c['verb']}, object={c['otype']}, receptacle={c['rtype']}, container state={c['state']}).\n"
        "Numbered events of this step, oldest first:\n" + render_packet(packet) + "\n\n"
        "Reply with ONE JSON object and nothing else. Either {\"insufficient_evidence\": true} or\n"
        "{\"cause\": \"" + packet["cause"] + "\", \"hypothesis\": \"<one short sentence>\", \"evidence\": [<event numbers "
        "from the list above>], \"action\": \"<one of " + "|".join(allowed + ["none"]) + ">\", "
        "\"alternative\": \"<optional, one of the same>\"}.\n"
        "Cite at least one event number. Do not invent events. Do not mention object ids or hidden state."
    )


class ParseError(ValueError):
    pass


def parse_response(text) -> dict:
    """Extract the first JSON object from a model reply (code fences and chatter tolerated). Raises ParseError."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty reply")
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    i = s.find("{")
    if i < 0:
        raise ParseError("no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[i:])
    except json.JSONDecodeError as e:
        raise ParseError(f"bad JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ParseError("not an object")
    return obj


class CallableHypothesizer(Hypothesizer):
    """Wraps `call(prompt:str) -> str` (a real Gemma client in workflow B, a mock in tests). Never imports a client."""

    def __init__(self, call):
        self.call = call

    def propose(self, packet: dict) -> dict:
        try:
            return parse_response(self.call(build_prompt(packet)))
        except ParseError as e:
            return {"_unparseable": str(e)[:60]}


_CAMEL = re.compile(r"[a-z][A-Z]")
_WORDISH = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_BAD_CATS = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}


def prose_problem(s) -> str | None:
    """Why this hypothesis prose may not be stored / shown, or None. At least as strict as the store's text columns
    (so a stored audit row can never abort a consolidation), plus: simulator ids and metadata-key-like tokens
    (underscored or camelCase tokens that normalise to a privileged key). Plain words such as 'visible' are fine."""
    if not isinstance(s, str) or not s.strip():
        return "empty"
    if unicodedata.normalize("NFC", s) != s:
        return "not_nfc"
    if len(s) > 2000:
        return "too_long"
    for ch in s:
        if ch in ("‌", "\n", "\t"):
            continue
        if unicodedata.category(ch) in _BAD_CATS:
            return "control_char"
    if SIM_ID_RE.search(s) or "|" in s:
        return "sim_id"
    for w in _WORDISH.findall(s):
        if ("_" in w or _CAMEL.search(w)) and is_privileged_name(w):
            return "metadata_key"
    return None


def validate_hypothesis(h, packet: dict, params: dict) -> dict:
    """status: accepted | abstain | rejected (+reason). Rejected hypotheses produce NO lesson; an abstention
    (`{"insufficient_evidence": true}` only) is not a rejection: the deterministic default may still apply."""
    res = {"status": "rejected", "reason": "", "hypothesis": None, "evidence": [], "action": None, "alternative": None}
    if not isinstance(h, dict):
        res["reason"] = "not_object"
        return res
    if "_unparseable" in h:
        res["reason"] = "unparseable"
        return res
    if "insufficient_evidence" in h:
        if h == {"insufficient_evidence": True}:
            res.update(status="abstain", reason="insufficient_evidence")
        else:
            res["reason"] = "mixed_abstain"
        return res
    allowed_keys = {"cause", "hypothesis", "evidence", "action", "alternative"}
    if set(h) - allowed_keys or not {"cause", "hypothesis", "evidence", "action"} <= set(h):
        res["reason"] = "bad_schema"
        return res
    det = packet["cause"]
    if h["cause"] not in CAUSES:
        res["reason"] = "bad_cause"
        return res
    if h["cause"] != det:
        res["reason"] = "cause_contradicts_deterministic"
        return res
    cap = int(params["lessons"]["hypothesis_max_chars"]["value"])
    prose = h["hypothesis"]
    if not isinstance(prose, str) or not prose.strip() or len(prose) > cap:
        res["reason"] = "prose_length"
        return res
    pp = prose_problem(prose)
    if pp:
        res["reason"] = "prose_forbidden"
        return res
    ev = h["evidence"]
    if (not isinstance(ev, list) or not ev or len(ev) > HYP_MAX_EVIDENCE or len(set(map(repr, ev))) != len(ev)
            or any(isinstance(x, bool) or not isinstance(x, int) for x in ev)):
        res["reason"] = "bad_evidence"
        return res
    if not set(ev) <= packet_ids(packet):
        res["reason"] = "evidence_not_in_packet"
        return res
    ok_actions = ALLOWED_ACTIONS.get(det, frozenset())
    act = h["action"]
    if act not in ACTIONS:
        res["reason"] = "bad_action"
        return res
    if act != "none" and act not in ok_actions:
        res["reason"] = "action_incompatible"
        return res
    alt = h.get("alternative")
    if alt is not None:
        if alt not in ACTIONS or (alt != "none" and alt not in ok_actions):
            res["reason"] = "bad_alternative"
            return res
        if alt == act:
            res["reason"] = "alternative_same"
            return res
    res.update(status="accepted", reason="accepted", hypothesis=prose, evidence=sorted(ev), action=act, alternative=alt)
    return res


# ------------------------------------------------------------------------------------------------ units + statuses
def unit_key(wm) -> str:
    """The credit unit of one episode: (task concept key, epoch). EN and FA of one item share concepts, hence a unit.
    Without concepts the fallback is the language-specific normalised text hash (no cross-language pairing then), and
    without text the episode itself."""
    tc = wm.first("task_context") or {}
    roles = intent_roles(tc.get("concepts"))
    if roles:
        base = canon(roles)
    elif isinstance(tc.get("text"), str) and tc["text"].strip():
        base = "t:" + unicodedata.normalize("NFC", tc["text"]).strip().lower()
    else:
        base = "e:" + str(wm.episode)
    return "u:" + _h(base, wm.epoch)


def unit_counts(rows, exclude_episode=None):
    """(helpful_units, unhelpful_units) from application_log rows of ONE record. Per unit key: any unhelpful outcome makes
    the unit unhelpful (a counterexample wins - order independent), otherwise helpful. Neutral rows and rows without a
    unit key are ignored; `exclude_episode` drops the origin episode."""
    per: dict = {}
    for r in rows:
        if r.get("outcome") == "neutral" or not r.get("source"):
            continue
        if exclude_episode is not None and r.get("episode") == exclude_episode:
            continue
        per.setdefault(r["source"], set()).add(r["outcome"])
    helpful = sum(1 for v in per.values() if "unhelpful" not in v)
    unhelpful = sum(1 for v in per.values() if "unhelpful" in v)
    return helpful, unhelpful


def next_status(prev: str, h: int, u: int, promote_after: int, retire_margin: int) -> str:
    """One transition step of the lesson lifecycle (plan 7.5). Retired is terminal; `supported -> contradicted` needs
    unhelpful >= helpful (one counterexample alone does not demote - Astra A3-5)."""
    if prev == "retired":
        return "retired"
    if prev == "tentative":
        if u - h >= retire_margin:
            return "retired"
        if h >= promote_after and h > u:
            return "supported"
        return "tentative"
    if prev == "supported":
        return "contradicted" if u >= h else "supported"
    if prev == "contradicted":
        if u - h >= retire_margin:
            return "retired"
        if h > u:
            return "supported"
        return "contradicted"
    raise LessonError(f"unknown lesson status {prev!r}")


def settle_status(prev: str, h: int, u: int, promote_after: int, retire_margin: int):
    """Iterate `next_status` to a fixed point (the rules are acyclic). Returns (status, [transitions as 'a_to_b'])."""
    path, cur = [], prev
    for _ in range(4):
        nxt = next_status(cur, h, u, promote_after, retire_margin)
        if nxt == cur:
            break
        path.append(f"{cur}_to_{nxt}")
        cur = nxt
    return cur, path


def procedure_status(success_units: int, failed_units: int, trust_after: int, retire_margin: int) -> str:
    """Derived purely from the unit counts (plan 7.1); counts can move a retired procedure back (documented 'reactivate')."""
    if failed_units >= success_units + retire_margin:
        return "retired"
    if failed_units >= success_units:
        return "demoted"
    if success_units >= trust_after:
        return "trusted"
    return "candidate"


def effective_status(row: dict, revalidating=()) -> str:
    """The status used for ranking / gating: a revalidating lesson counts as tentative (never revives a retired one)."""
    if row["status"] == "retired":
        return "retired"
    if row.get("revalidate") or row["lesson"] in revalidating:
        return "tentative"
    return row["status"]


# ------------------------------------------------------------------------------------------------ intent + body
NATIVE_FIELDS = ("verb", "req", "obj", "ostate", "src", "dest", "dstate")      # sim/ltm_concepts.py ROLE_FIELDS (+ neg 0/1)


def _is_native(r) -> bool:
    return isinstance(r, dict) and bool(set(r) & {"obj", "dest", "req", "ostate", "src", "dstate"})


def intent_roles(concepts):
    """Normalise `task_context.concepts` into ordered action roles or None. Two shapes are read:
      * native (`ltm_concepts.concepts()`): {"roles": [{verb, req, obj, ostate, src, dest, dstate, neg: 0|1}, ...]}
        - kept field for field, so `ltm_concepts.roles_equal` sees exactly what was admitted;
      * legacy / test: {"roles": [{verb, otype, to, state, neg: bool}, ...]} or role lists [verb, otype, to, state, neg].
    """
    if not isinstance(concepts, dict):
        return None
    raw = concepts.get("roles")
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for r in raw[:32]:
        if _is_native(r):
            role = {f: _tok(r.get(f)) for f in NATIVE_FIELDS}
            role["neg"] = 1 if r.get("neg") in (1, True) else 0
            if role["verb"] == "?" and role["obj"] == "?":
                continue
            out.append(role)
            continue
        if isinstance(r, dict):
            verb, otype = r.get("verb"), r.get("otype", r.get("type"))
            to = r.get("to", r.get("dest", r.get("destination")))
            state, neg = r.get("state"), r.get("neg", r.get("negated", False))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            pad = list(r) + [None] * (5 - len(r))
            verb, otype, to, state, neg = pad[:5]
        else:
            return None
        role = {"verb": _tok(verb), "otype": _tok(otype), "to": _tok(to), "state": _tok(state), "neg": neg is True}
        if role["verb"] == "?" and role["otype"] == "?":
            continue
        out.append(role)
    return out or None


def roles_fully_known(roles) -> bool:
    """A native role with any '?' field can never steer (plan 8.3: '?' never matches), so it is not admitted."""
    return all(not _is_native(r) or all(r[f] != "?" for f in NATIVE_FIELDS) for r in roles)


def render_intent_text(roles: list) -> str:
    """Key text rendered from structured roles by a fixed EN template (plan 8.2) - never copied from the instruction."""
    parts = []
    for r in roles:
        neg = "not " if r["neg"] else ""
        if _is_native(r):
            t = neg + r["verb"]
            for f, pre in (("obj", " "), ("ostate", " ("), ("src", " from "), ("dest", " to "), ("dstate", " (")):
                if r[f] not in ("-", "?"):
                    t += pre + r[f] + (")" if pre.endswith("(") else "")
        else:
            t = f"{neg}{r['verb']} {r['otype']}"
            if r["to"] != "?":
                t += f" to {r['to']}"
            if r["state"] != "?":
                t += f" ({r['state']})"
        parts.append(t)
    return " ; ".join(parts)


def body_signature(body: dict) -> list:
    return [[s["verb"], s["otype"], s["to"]] for s in body["steps"]]


def build_body(steps: dict):
    """The executed sequence with per-step observable preconditions and the repairs used. (body, reason)."""
    if not steps:
        return None, "empty_plan"
    if len(steps) > BODY_MAX_STEPS:
        return None, "too_long"
    out = []
    for i in sorted(steps):
        s = steps[i]
        if s["start"] is None or s["outcome"] is None:
            return None, "step_unobserved"
        if s["outcome"].get("ok") is not True:
            return None, "step_failed"
        st = s["start"]
        rec = [_tok(s["attempts"][k].get("recovery")) for k in sorted(s["attempts"]) if k >= 1
               and s["attempts"][k].get("recovery")]
        out.append({"i": i, "verb": _tok(st.get("verb")), "otype": _tok(st.get("otype")), "to": _tok(st.get("to")),
                    "pre": _flat(st.get("pre")), "rec": rec[:MAX_RECOVERY_K]})
    return {"steps": out}, None


def preconditions_hold(required: dict, current) -> bool:
    """Body-step preconditions vs a snapshot. A required value of '?'/None imposes nothing; an unknown/missing current
    value never satisfies a definite requirement (conservative: the procedure is simply not offered)."""
    cur = current if isinstance(current, dict) else {}
    for key, want in (required or {}).items():
        if want in ("?", None):
            continue
        if cur.get(key) != want:
            return False
    return True


def procedure_preconditions_hold(proc_row: dict, current_pre, start_i=None) -> bool:
    steps = (proc_row.get("body") or {}).get("steps") or []
    if not steps:
        return False
    first = next((s for s in steps if s.get("i") == start_i), steps[0])
    return preconditions_hold(first.get("pre"), current_pre)


def admission_check(wm, goal_ok, exec_ok) -> tuple:
    """(ok, reason, body, roles). R7 / privileged use 1: admitted only if the privileged verdict is True AND the executed
    plan supports it (terminal.exec_ok AND the WM-derived check agree: every step observed with a final ok outcome).
    Goal met with a failed step (e.g. a refused step the goal did not need) is NOT a procedure."""
    if wm.invalid:
        return False, "invalid", None, None
    if goal_ok is None:
        return False, "no_verdict", None, None
    if goal_ok is not True:
        return False, "goal_not_met", None, None
    steps = step_index(wm)
    ok_all = all_steps_ok(steps)
    if not exec_ok:
        return False, "goal_without_plan", None, None
    if not steps:
        return False, "empty_plan", None, None
    if not ok_all:
        any_failed = any(s["outcome"] is not None and s["outcome"].get("ok") is False for s in steps.values())
        return False, ("exec_mismatch" if any_failed else "step_unobserved"), None, None
    body, why = build_body(steps)
    if body is None:
        return False, why, None, None
    tc = wm.first("task_context") or {}
    roles = intent_roles(tc.get("concepts"))
    if not roles:
        return False, "no_intent", None, None
    if not roles_fully_known(roles):
        return False, "intent_unknown", None, None
    return True, "admit", body, roles


def procedure_id(kitchen, roles, body) -> str:
    return "pr:" + _h(kitchen, canon(roles), canon(body))


# ------------------------------------------------------------------------------------------------ application scoring
def score_lesson_application(steps: dict, fails: list, row: dict, app_ev: dict, agent_verdict) -> tuple:
    """(outcome, reason). Precedence (plan 7.5 as amended by Astra A3-4): unobservable / aborted -> neutral; the lesson's
    cause recurring in this step at or after the applied attempt -> unhelpful; the applied attempt ok -> helpful;
    otherwise (a different cause, or success credited to a later remedy) -> neutral. Failure elsewhere in the task is
    irrelevant by construction (only step `i` is read)."""
    i, k = app_ev.get("i"), app_ev.get("k")
    s = steps.get(i)
    if s is None or k is None:
        return "neutral", "unobservable"
    a = s["attempts"].get(k)
    if a is None or a.get("ok") is None:
        return "neutral", "unobservable"
    if agent_verdict == "unknown" and s["outcome"] is None:
        return "neutral", "aborted"
    lcause = row["conditions"]["cause"]
    kk = k
    while kk in s["attempts"]:
        att = s["attempts"][kk]
        if att.get("ok") is True:
            break
        if att.get("ok") is False and _attempt_cause(s, i, kk, att)[1] == lcause:
            return "unhelpful", "same_cause_recurs"
        kk += 1
    if a.get("ok") is True:
        if s["outcome"] is not None and s["outcome"].get("ok") is False:
            return "neutral", "inconsistent"
        return "helpful", "step_ok"
    return "neutral", "diff_cause_or_later"


def lesson_app_eligibility(steps, fails, row, app_ev, apps_by_slot) -> tuple:
    if row is None:
        return False, "unknown_record"
    if row["status"] == "retired":
        return False, "retired"
    i, k = app_ev.get("i"), app_ev.get("k")
    if i is None or k is None or not 1 <= k <= MAX_RECOVERY_K:
        return False, "attempt_range"
    if len({a["record"] for a in apps_by_slot[(i, k)]}) > 1:
        return False, "two_remedies"
    s = steps.get(i)
    a = s["attempts"].get(k) if s else None
    if a is None or a.get("recovery") != "lesson" or a.get("source") != f"lesson:{row['lesson']}":
        return False, "provenance"
    if not app_ev["ev"] < a["ev"]:
        return False, "chronology"
    prior = [f for f in fails if f["i"] == i and f["k"] == k - 1 and not f["step_level"]]
    if not prior or prior[0]["cond_key"] != row["cond_key"]:
        return False, "key_mismatch"
    return True, "ok"


def score_procedure_application(steps: dict, row: dict, app_ev: dict, goal_ok) -> tuple:
    """(outcome, reason) for a reused procedure (privileged use 2). Needs: preconditions held at application, an
    identified body span (the executed steps from application.i must match the body's (verb, otype, to) in order).
    A failure observed inside the span counts even without a verdict; success needs the full span ok AND goal True."""
    body = row["body"]["steps"]
    if not preconditions_hold(body[0].get("pre"), app_ev.get("pre")):
        return "neutral", "preconditions_not_holding"
    start = app_ev.get("i", 0)
    span = [steps[j] for j in sorted(steps) if j >= start][:len(body)]
    for s, b in zip(span, body):
        st = s["start"]
        if st is None or [_tok(st.get("verb")), _tok(st.get("otype")), _tok(st.get("to"))] != [b["verb"], b["otype"], b["to"]]:
            return "neutral", "span_mismatch"
    failed = any(s["outcome"] is not None and s["outcome"].get("ok") is False for s in span)
    if failed:
        return "unhelpful", "failed_step_in_span"
    if len(span) < len(body) or any(s["outcome"] is None for s in span):
        return "neutral", "span_incomplete"
    if goal_ok is True:
        return "helpful", "span_ok_goal_met"
    if goal_ok is False:
        return "unhelpful", "goal_unmet"
    return "neutral", "no_verdict"


# ------------------------------------------------------------------------------------------------ the extension
class LessonsExtension:
    """`ltm.end_episode(wm, extensions=[LessonsExtension(hypothesizer)])`. Practice episodes only (raises otherwise).
    After the call `last_stats` holds the counters that were also written (as one `dx:<episode>` summary row) in the
    same commit; if the consolidation context offers `ctx.stats` they are mirrored there under `x_` keys too."""

    def __init__(self, hypothesizer: Hypothesizer | None = None):
        self.hypothesizer = hypothesizer
        self.last_stats: dict = {}
        self.last_report: dict = {}

    def __call__(self, ctx) -> list:
        wm = ctx.wm
        if wm.mode != "practice":
            raise LessonError("lessons / procedures are written from practice episodes only")
        return self._build(ctx)

    # ------------------------------------------------------------------------------------------------------
    def _build(self, ctx) -> list:
        ltm, wm, kitchen = ctx.ltm, ctx.wm, ctx.kitchen
        store, P = ltm.store, ltm.params
        E, ep = ctx.E_next, wm.episode
        goal_ok = ctx.goal_ok
        promote_after = int(P["lessons"]["promote_after"]["value"])
        l_margin = int(P["lessons"]["retire_margin"]["value"])
        trust_after = int(P["procedures"]["trust_after"]["value"])
        p_margin = int(P["procedures"]["retire_margin"]["value"])
        default_action = P["lessons"]["default_action"]["value"]
        term = wm.terminal or {}
        cnt: Counter = Counter()
        diag = diagnose(wm, goal_ok)
        steps, fails = diag["steps"], diag["failures"]
        ukey = unit_key(wm)
        ops, log_rows, hyp_rows = [], [], []
        report = {"episode": ep, "lessons": {}, "procedures": {}, "failures": [], "hypotheses": []}

        for f in fails:
            cnt["dx_" + f["cause"]] += 1
        cnt["failures_diagnosed"] = len(fails)
        if diag["plan_goal_unmet"]:
            cnt["dx_plan_goal_unmet"] += 1                      # counted + labelled; never a lesson (no recovery attempt)

        # ---------------- lessons: existing state
        ex = {r["lesson"]: r for r in store.select("lesson", where={"kitchen": kitchen})}
        by_ck = {r["cond_key"]: r for r in ex.values()}
        state = {lid: dict(r) for lid, r in ex.items()}      # working copy
        touched: dict = {}                                   # lid -> {"scored": outcome|None, "recur": 0/1, "evs": set()}

        def touch(lid):
            return touched.setdefault(lid, {"scored": None, "recur": 0, "evs": set(), "created": False})

        # ---------------- A. score lesson applications
        apps = [e for e in wm.events if e["kind"] == "application" and str(e.get("record", "")).startswith("ls:")]
        slot: dict = {}
        for e in apps:
            slot.setdefault((e.get("i"), e.get("k")), []).append(e)
        scored_once: set = set()
        for e in apps:
            row = ex.get(e["record"])
            ok, why = lesson_app_eligibility(steps, fails, row, e, slot)
            if not ok:
                cnt["app_ineligible_" + why] += 1
                if why == "two_remedies":
                    cnt["protocol_violations"] += 1
                continue
            if e["record"] in scored_once:
                cnt["app_repeat_not_scored"] += 1
                continue
            scored_once.add(e["record"])
            outcome, reason = score_lesson_application(steps, fails, row, e, term.get("agent_verdict"))
            cnt["app_" + outcome] += 1
            cnt["app_reason_" + reason] += 1
            t = touch(e["record"])
            t["scored"] = outcome
            a = steps[e["i"]]["attempts"][e["k"]]
            t["evs"] |= {e["ev"], a["ev"]}
            log_rows.append({"app": "ap:" + _h(ep, e["record"]), "episode": ep, "record": e["record"], "kind": "lesson",
                             "step_i": e["i"], "attempt_k": e["k"], "outcome": outcome, "goal_ok": None,
                             "retrieval": e.get("retrieval"), "E": "$E", "source": ukey})

        # ---------------- B. failures -> recurrence / new lessons
        seen_ck: set = set()
        created_rows: dict = {}
        for f in fails:
            ck = f["cond_key"]
            if ck in seen_ck:
                if ck in by_ck:
                    touch(by_ck[ck]["lesson"])["evs"].add(f["ev"])
                continue
            seen_ck.add(ck)
            allowed = ALLOWED_ACTIONS.get(f["cause"], frozenset())
            report["failures"].append({"i": f["i"], "k": f["k"], "cause": f["cause"], "cond_key": ck})
            if not allowed:
                cnt["no_remedy_for_" + f["cause"]] += 1
                continue
            if ck in by_ck:
                t = touch(by_ck[ck]["lesson"])
                t["recur"] = 1
                t["evs"].add(f["ev"])
                cnt["lesson_recurrences"] += 1
                continue
            packet = build_packet(wm, f, steps)
            hid = "hy:" + _h(ep, ck)
            status, reason, action, hyp = "abstain", "no_hypothesizer", None, None
            if self.hypothesizer is not None:
                cnt["hyp_requested"] += 1
                try:
                    raw = self.hypothesizer.propose(packet)
                    hyp = validate_hypothesis(raw, packet, P)
                    status, reason = hyp["status"], hyp["reason"]
                except Exception as exc:                        # a failing model must never abort consolidation
                    hyp = {"hypothesis": None, "evidence": [], "action": None, "alternative": None}
                    status, reason = "rejected", "hypothesizer_error"
                cnt["hyp_" + status] += 1
            lid = lesson_id(kitchen, ck)
            source = "default"
            if status == "accepted":
                action, source = hyp["action"], "hypothesis"
                if action == "none":
                    cnt["lessons_not_created_hypothesis_none"] += 1
                    action = None
            elif status == "abstain":
                action = default_action.get(f["cause"], "none")
                source = "default"
                if action in (None, "none") or action not in ALLOWED_ACTIONS[f["cause"]]:
                    action = None
            else:
                cnt["lessons_not_created_hypothesis_rejected"] += 1
            if self.hypothesizer is not None:
                prose = hyp.get("hypothesis") if hyp else None
                hyp_rows.append({"hyp": hid, "lesson": lid if action else None, "episode": ep,
                                 "prose": prose if prose and prose_problem(prose) is None else None,
                                 "evidence": (hyp or {}).get("evidence") or [], "accepted": status == "accepted",
                                 "reason": reason})
                report["hypotheses"].append({"hyp": hid, "status": status, "reason": reason})
            if action is None:
                continue
            evs = {f["ev"]} | set((hyp or {}).get("evidence") or []) | {packet["events"][0]["ev"]} \
                if packet["events"] else {f["ev"]}
            row = {"lesson": lid, "kitchen": kitchen, "conditions": f["conditions"], "cond_key": ck, "action": action,
                   "status": "tentative", "helpful_units": 0, "unhelpful_units": 0, "recurrences": 0,
                   "origin_episode": ep, "E_created": "$E", "E_support": "$E", "revalidate": False}
            state[lid] = dict(row)
            created_rows[lid] = {"row": row, "source": source, "alt": (hyp or {}).get("alternative")}
            t = touch(lid)
            t["created"] = True
            t["evs"] |= evs
            cnt["lessons_created"] += 1

        # ---------------- C. recompute + emit lessons
        new_log_by_record: dict = {}
        for r in log_rows:
            new_log_by_record.setdefault(r["record"], []).append(dict(r, E=E))
        for lid in sorted(touched):
            t, st = touched[lid], state[lid]
            old = ex.get(lid)
            if t["created"]:
                ops.append({"table": "lesson", "op": "insert", "values": dict(created_rows[lid]["row"]),
                            "evidence": {"rule": "R8", "mutation": "create",
                                         "summary": {"kind": "lesson", "cause": st["conditions"]["cause"],
                                                     "action": st["action"], "source": created_rows[lid]["source"],
                                                     "alt": created_rows[lid]["alt"] or "none"},
                                         "src_ev": self._src(ep, t["evs"])}})
                report["lessons"][lid] = {"created": True}
                continue
            prior_rows = [r for r in store.select("application_log", where={"record": lid})]
            all_rows = prior_rows + new_log_by_record.get(lid, [])
            h, u = unit_counts(all_rows, exclude_episode=old["origin_episode"])
            new_status, path = settle_status(old["status"], h, u, promote_after, l_margin)
            revalidate = bool(old["revalidate"])
            cleared = False
            if revalidate and t["scored"] == "helpful":
                bounds = [r["E"] for r in all_rows if r.get("retrieval") == "reval"]
                bE = max(bounds) if bounds else -1
                pre_units = {r["source"] for r in all_rows if r["outcome"] != "neutral" and r.get("source")
                             and r["E"] <= bE}
                post = [r for r in all_rows if r["outcome"] == "helpful" and r["E"] > bE and r.get("source")
                        and r["source"] not in pre_units and r["episode"] != old["origin_episode"]]
                if post:
                    revalidate, cleared = False, True
            changed = (h, u, new_status, revalidate, t["recur"]) != (old["helpful_units"], old["unhelpful_units"],
                                                                     old["status"], old["revalidate"], 0)
            vals = {"lesson": lid, "kitchen": kitchen, "conditions": old["conditions"], "cond_key": old["cond_key"],
                    "action": old["action"], "status": new_status, "helpful_units": h, "unhelpful_units": u,
                    "recurrences": old["recurrences"] + t["recur"], "origin_episode": old["origin_episode"],
                    "E_created": old["E_created"],
                    "E_support": E if t["scored"] in ("helpful", "unhelpful") else old["E_support"],
                    "revalidate": revalidate}
            if t["scored"] in ("helpful", "unhelpful"):
                vals["E_support"] = "$E"
            if not changed and t["scored"] in (None, "neutral"):
                continue
            if new_status == "retired" and old["status"] != "retired":
                mut = "retire"
            elif t["scored"] == "unhelpful" or any(p.endswith("_to_contradicted") for p in path):
                mut = "contradict"
            else:
                mut = "support"
            ops.append({"table": "lesson", "op": "upsert", "values": vals,
                        "evidence": {"rule": "R8", "mutation": mut,
                                     "summary": {"kind": "lesson", "scored": t["scored"] or "none", "recur": t["recur"],
                                                 "transitions": path, "h": h, "u": u, "cleared_revalidate": cleared},
                                     "src_ev": self._src(ep, t["evs"])}})
            report["lessons"][lid] = {"scored": t["scored"], "transitions": path, "h": h, "u": u, "status": new_status}
            if path:
                cnt["lesson_transitions"] += len(path)

        # ---------------- D. procedures: reuse scoring + admission
        pr_rows = {r["proc"]: r for r in store.select("procedure", where={"kitchen": kitchen})}
        p_new_logs: dict = {}
        p_evs: dict = {}
        p_scored: dict = {}
        scored_pr: set = set()
        for e in [e for e in wm.events if e["kind"] == "application" and str(e.get("record", "")).startswith("pr:")]:
            row = pr_rows.get(e["record"])
            if row is None or row["status"] == "retired":
                cnt["proc_app_ineligible"] += 1
                continue
            if e["record"] in scored_pr:
                cnt["proc_app_repeat_not_scored"] += 1
                continue
            scored_pr.add(e["record"])
            outcome, reason = score_procedure_application(steps, row, e, goal_ok)
            cnt["proc_app_" + outcome] += 1
            p_scored[e["record"]] = outcome
            p_evs.setdefault(e["record"], set()).add(e["ev"])
            p_new_logs.setdefault(e["record"], []).append({
                "app": "ap:" + _h(ep, e["record"]), "episode": ep, "record": e["record"], "kind": "procedure",
                "step_i": e.get("i", 0), "attempt_k": e.get("k", 0), "outcome": outcome,
                "goal_ok": goal_ok if outcome != "neutral" else None, "retrieval": e.get("retrieval"), "E": "$E",
                "source": ukey})
        ok, why, body, roles = admission_check(wm, goal_ok, ctx.exec_ok)
        admitted_pid, created_proc = None, None
        if not ok:
            if why not in ("goal_not_met",):
                cnt["proc_rejected_" + why] += 1
        else:
            pid = procedure_id(kitchen, roles, body)
            admitted_pid = pid
            p_new_logs.setdefault(pid, []).append({
                "app": "ad:" + _h(pid, ep), "episode": ep, "record": pid, "kind": "procedure", "step_i": 0,
                "attempt_k": 0, "outcome": "helpful", "goal_ok": True, "retrieval": "admit", "E": "$E", "source": ukey})
            p_evs.setdefault(pid, set()).update({s["outcome"]["ev"] for s in steps.values()})
            if pid not in pr_rows:
                created_proc = {"proc": pid, "kitchen": kitchen, "intent_key": {"roles": roles},
                                "intent_text": render_intent_text(roles), "body": body, "status": "candidate",
                                "success_uses": 0, "failed_reuses": 0, "E_created": "$E", "E_support": "$E",
                                "admit_goal_ok": True}
                cnt["proc_created"] += 1
            else:
                cnt["proc_admission_support"] += 1
        for pid in sorted(p_new_logs):
            base = pr_rows.get(pid) or created_proc
            prior = [r for r in store.select("application_log", where={"record": pid})] if pid in pr_rows else []
            allr = prior + [dict(r, E=E) for r in p_new_logs[pid]]
            sh, fu = unit_counts(allr)
            new_status = procedure_status(sh, fu, trust_after, p_margin)
            created = pid not in pr_rows
            vals = {"proc": pid, "kitchen": kitchen, "intent_key": base["intent_key"], "intent_text": base["intent_text"],
                    "body": base["body"], "status": new_status, "success_uses": sh, "failed_reuses": fu,
                    "E_created": base["E_created"], "E_support": base["E_support"], "admit_goal_ok": base["admit_goal_ok"]}
            if not created:
                old = pr_rows[pid]
                observed = any(r["outcome"] != "neutral" for r in p_new_logs[pid])
                if (sh, fu, new_status) == (old["success_uses"], old["failed_reuses"], old["status"]):
                    continue
                if observed:
                    vals["E_support"] = "$E"
                if new_status == "retired" and old["status"] != "retired":
                    mut = "retire"
                elif old["status"] == "retired" and new_status != "retired":
                    mut = "reactivate"
                elif fu > old["failed_reuses"]:
                    mut = "contradict"
                else:
                    mut = "support"
            else:
                mut = "create"
            ops.append({"table": "procedure", "op": "insert" if created else "upsert", "values": vals,
                        "evidence": {"rule": "R7", "mutation": mut,
                                     "summary": {"kind": "procedure", "success_units": sh, "failed_units": fu,
                                                 "status": new_status, "admitted": pid == admitted_pid,
                                                 "scored": p_scored.get(pid, "none")},
                                     "src_ev": self._src(ep, p_evs.get(pid, set()))}})
            report["procedures"][pid] = {"status": new_status, "s": sh, "f": fu, "created": created}
            for r in p_new_logs[pid]:
                log_rows.append(r)
        # log rows for the lesson side were appended earlier; make sure each is emitted exactly once
        # ---------------- E. rows that are not claims: application_log, hypothesis_audit, per-episode summary
        emitted = set()
        for r in log_rows:
            if r["app"] in emitted:
                continue
            emitted.add(r["app"])
            ops.append({"table": "application_log", "op": "insert", "values": r})
        for r in hyp_rows:
            ops.append({"table": "hypothesis_audit", "op": "insert", "values": r})
        # defence in depth: an op the store would refuse must never abort the episode's core claims (counted instead)
        ops = self._guard(store, ops, cnt)
        stats = {k[:41]: int(v) for k, v in sorted(cnt.items())}      # json key regex: <= 41 chars
        ops += self._guard(store, [{"table": "hypothesis_audit", "op": "insert",
                                    "values": {"hyp": "dx:" + ep[:80], "lesson": None, "episode": ep, "prose": None,
                                               "evidence": {"counters": stats}, "accepted": True,
                                               "reason": "episode_summary"}}], cnt)
        self.last_stats = dict(stats)
        self.last_report = report
        target = getattr(ctx, "stats", None)
        if isinstance(target, dict):
            for k, v in self.last_stats.items():
                target["x_" + k] = v
        return ops

    @staticmethod
    def _src(ep, evs) -> list:
        return sorted({ev_hash(ep, e) for e in evs})[:64]

    @staticmethod
    def _guard(store, ops, cnt) -> list:
        norm = getattr(store, "_normalise_op", None)
        if norm is None:
            return ops
        good = []
        for idx, op in enumerate(ops):
            try:
                norm(op, idx)
                good.append(op)
            except WhitelistError:
                cnt["ops_rejected"] += 1
        return good


# ------------------------------------------------------------------------------------------------ read side
def lesson_for(ltm, wm, failure_key: dict, available, retries_left: int, revalidating=()) -> dict | None:
    """Zero or one lesson remedy for a recovery attempt (plan 7.5; Astra A3-7). Scoped to this kitchen and the EXACT
    local failure key; the remedy must be executable now (`available`: set of action names) and budget must remain.
    Practice: any non-retired lesson (supported first, then tentative / revalidating, then contradicted, then the older
    E_created, then id). Test / probe / no wm: ONLY supported lessons that are not revalidating. Read-only: nothing is
    counted here; the application is scored later from the episode's events."""
    if retries_left < 1 or not available:
        return None
    cond = make_conditions(failure_key.get("cause"), failure_key.get("verb"), failure_key.get("otype"),
                           failure_key.get("rtype"), failure_key.get("state"))
    ck = cond_key_of(cond)
    practice = wm is not None and wm.mode == "practice"
    rank = {"supported": 0, "tentative": 1, "contradicted": 2}
    best = None
    for r in ltm.store.select("lesson", where={"kitchen": ltm.kitchen, "cond_key": ck}):
        eff = effective_status(r, revalidating)
        if eff == "retired" or r["action"] == "none" or r["action"] not in available:
            continue
        if not practice and eff != "supported":
            continue
        key = (rank[eff], r["E_created"], r["lesson"])
        if best is None or key < best[0]:
            best = (key, r, eff)
    if best is None:
        return None
    _, r, eff = best
    return {"lesson": r["lesson"], "action": r["action"], "status": eff, "source": f"lesson:{r['lesson']}",
            "cond_key": ck, "experimental": eff != "supported"}


def flag_revalidate(ltm, boundary_id: str, reason: str = "reset_world") -> dict:
    """After a world reset / frame change: geometry-dependent, non-retired lessons become 'revalidating'. Persisted as a
    flag plus an immutable boundary row in application_log (retrieval='reval'); cleared only by a later helpful unit
    that is new (not a unit credited before the boundary). Idempotent per `boundary_id`."""
    if getattr(ltm, "frozen", False):
        raise LessonError("frozen store")
    ops, E = [], ltm.store.clock()
    for r in sorted(ltm.store.select("lesson", where={"kitchen": ltm.kitchen}), key=lambda r: r["lesson"]):
        if r["status"] == "retired" or r["action"] not in GEOMETRY_ACTIONS:
            continue
        ops.append({"table": "lesson", "op": "update", "key": {"lesson": r["lesson"]}, "values": {"revalidate": True},
                    "evidence": {"rule": "R8", "mutation": "contradict",
                                 "summary": {"kind": "revalidate_flag", "reason": _tok(reason)}, "src_ev": []}})
        ops.append({"table": "application_log", "op": "insert",
                    "values": {"app": "rv:" + _h(r["lesson"], boundary_id), "episode": "rv-" + _tok(boundary_id, "x")[:60],
                               "record": r["lesson"], "kind": "lesson", "step_i": 0, "attempt_k": 0, "outcome": "neutral",
                               "goal_ok": None, "retrieval": "reval", "E": "$E", "source": "b:" + _h(boundary_id)}})
    if not ops:
        return {"status": "nothing_to_flag"}
    return ltm.store.commit("revalidate:" + _h(ltm.kitchen, boundary_id), "admin", ops, stamp=E)


def log_test_application(ltm, wm, lesson_or_proc_id: str, kind: str, i: int, k: int, retrieval=None) -> dict:
    """Test mode: record that a record was used into the frozen store's separate testlog (never counted back)."""
    return ltm.log_test("application_log", {
        "app": "tp:" + _h(wm.episode, lesson_or_proc_id, i, k), "episode": wm.episode, "record": lesson_or_proc_id,
        "kind": kind, "step_i": i, "attempt_k": k, "outcome": "neutral", "goal_ok": None, "retrieval": retrieval,
        "E": ltm.store.clock(), "source": None})


def lesson_version_hash(row: dict) -> str:
    """Frozen identity of an audited lesson version (FLR adjudication never feeds learning)."""
    keys = ("lesson", "cond_key", "action", "status", "helpful_units", "unhelpful_units", "recurrences", "conditions")
    return sha256_hex(canon({k: row[k] for k in keys}))


# ------------------------------------------------------------------------------------------------ false-lesson rate
def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson score interval; None for an empty denominator (an empty set has no rate)."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def flr_population(ltm) -> dict:
    """Populations for the audit, read from the store: supported-at-checkpoint ids (effective status, so revalidating
    lessons are not counted as supported) and every lesson ever created as (id, status) plus frozen version hashes."""
    rows = ltm.store.select("lesson", where={"kitchen": ltm.kitchen})
    return {"supported": sorted(r["lesson"] for r in rows if effective_status(r) == "supported"),
            "created": sorted((r["lesson"], r["status"]) for r in rows),
            "versions": {r["lesson"]: lesson_version_hash(r) for r in rows}}


def supported_flr(supported_ids, labels: dict) -> dict:
    """Supported-set FLR: false / ALL supported lessons at the checkpoint (every one must be audited; insufficient cases
    stay in the denominator and are reported separately)."""
    ids = sorted(set(supported_ids))
    missing = [i for i in ids if i not in labels]
    cnt = Counter(labels[i] for i in ids if i in labels)
    n = len(ids)
    return {"n": n, "false": cnt["false"], "correct": cnt["correct"], "insufficient": cnt["insufficient"],
            "unaudited": len(missing), "complete": not missing, "flr": (cnt["false"] / n) if n and not missing else None,
            "ci95": wilson(cnt["false"], n) if not missing else None,
            "flr_decided": (cnt["false"] / (cnt["false"] + cnt["correct"])) if (cnt["false"] + cnt["correct"]) else None}


def audit_plan(lessons, cap: int, seed: int) -> dict:
    """All-created audit plan. `lessons`: iterable of (id, status). n <= cap: audit all (inclusion probability 1).
    Otherwise stratify by status with a deterministic proportional allocation (largest remainder, >= 1 per non-empty
    stratum) and a seeded draw; every stratum gets a non-zero known inclusion probability. Raises if cap < #strata."""
    by: dict = {}
    for lid, st in lessons:
        by.setdefault(st, []).append(lid)
    for v in by.values():
        v.sort()
    N = sum(len(v) for v in by.values())
    if N == 0:
        return {"seed": seed, "N": 0, "ids": [], "strata": {}, "census": True}
    if N <= cap:
        return {"seed": seed, "N": N, "ids": sorted(i for v in by.values() for i in v), "census": True,
                "strata": {s: {"N": len(v), "n": len(v), "p": 1.0} for s, v in sorted(by.items())}}
    if cap < len(by):
        raise ValueError("audit cap smaller than the number of non-empty strata: inclusion probabilities would be zero")
    alloc = {s: 1 for s in by}
    rest = cap - len(by)
    quota = {s: rest * len(v) / N for s, v in by.items()}
    for s in by:
        alloc[s] += int(quota[s])
    left = cap - sum(alloc.values())
    for s in sorted(by, key=lambda s: (-(quota[s] - int(quota[s])), s))[:left]:
        alloc[s] += 1
    rng = random.Random(seed)
    ids, strata = [], {}
    for s in sorted(by):
        n = min(alloc[s], len(by[s]))
        pick = sorted(rng.sample(by[s], n))
        ids += pick
        strata[s] = {"N": len(by[s]), "n": n, "p": n / len(by[s])}
    return {"seed": seed, "N": N, "ids": sorted(ids), "strata": strata, "census": False}


def all_created_flr(plan: dict, strata_of: dict, labels: dict, boot: int = 2000, seed: int = 0) -> dict:
    """Horvitz-Thompson estimate of false / all-created from the audited sample. `strata_of`: id -> status. Insufficient
    cases are estimated separately. The interval is a seeded stratified bootstrap; a non-census stratum with fewer than
    two audited lessons makes the bootstrap meaningless, so then only the worst/best-case completion bounds are given."""
    N = plan["N"]
    if N == 0:
        return {"N": 0, "flr": None, "ci95": None, "bounds": None, "reason": "empty_set"}
    aud = [i for i in plan["ids"] if i in labels]
    if len(aud) != len(plan["ids"]):
        return {"N": N, "flr": None, "ci95": None, "bounds": None, "reason": "unaudited_lessons"}
    by: dict = {}
    for i in aud:
        by.setdefault(strata_of[i], []).append(labels[i])

    def est(label, groups):
        tot = 0.0
        for s, labs in groups.items():
            info = plan["strata"][s]
            tot += sum(1 for x in labs if x == label) / info["p"]
        return tot / N

    flr, insuff = est("false", by), est("insufficient", by)
    sparse = [s for s, i in plan["strata"].items() if i["n"] < i["N"] and i["n"] < 2]
    lo_b = hi_b = None
    known = sum(sum(1 for x in labs if x == "false") / 1 for labs in by.values())
    if plan["census"]:
        return {"N": N, "flr": flr, "insufficient": insuff, "ci95": wilson(round(flr * N), N), "bounds": None,
                "census": True, "reason": "census"}
    unaud = sum(i["N"] - i["n"] for i in plan["strata"].values())
    lo_b, hi_b = known / N, (known + unaud) / N
    if sparse:
        return {"N": N, "flr": flr, "insufficient": insuff, "ci95": None, "bounds": (lo_b, hi_b), "census": False,
                "reason": "sparse_strata_bootstrap_undefined:" + ",".join(sorted(sparse))}
    rng = random.Random(seed)
    draws = []
    for _ in range(boot):
        g = {s: [rng.choice(labs) for _ in labs] for s, labs in by.items()}
        draws.append(est("false", g))
    draws.sort()
    ci = (draws[int(0.025 * (boot - 1))], draws[int(math.ceil(0.975 * (boot - 1)))])
    return {"N": N, "flr": flr, "insufficient": insuff, "ci95": ci, "bounds": (lo_b, hi_b), "census": False,
            "reason": "ht_stratified_bootstrap"}


def hypothesis_accuracy(labels) -> dict:
    """Separate small labelled failure audit of Gemma's hypotheses: correct / incorrect / insufficient."""
    c = Counter(labels)
    n = c["correct"] + c["incorrect"]
    return {"correct": c["correct"], "incorrect": c["incorrect"], "insufficient": c["insufficient"],
            "accuracy": (c["correct"] / n) if n else None, "ci95": wilson(c["correct"], n)}
