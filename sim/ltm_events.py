"""Step 5 / A2 - the event contract (docs/step5-plan.md section 1).

A closed schema: unknown kinds / keys are rejected, types are checked, and privileged material (simulator ids, metadata
keys, error text, oracle state, goal predicates, `group`) is refused at ingest. Free text is admitted in exactly one
place, `task_context.text`. This is an input-validation boundary; that the adapter does not PRODUCE privileged values
is gated separately (A7/A8).

Pure standard library; imports nothing of the simulator.
"""
from __future__ import annotations

import math
import re

try:
    from ltm_store import SIM_ID_RE, TOKEN_RE, is_privileged_name, sha256_hex
except ImportError:                     # pragma: no cover
    from sim.ltm_store import SIM_ID_RE, TOKEN_RE, is_privileged_name, sha256_hex

SCHEMA = "s5.ev/1"
ENVELOPE = ("schema", "ev", "kind", "kitchen", "epoch", "frame", "map_rev", "episode", "t")
ID_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:.\-]{0,95}$")

ENUMS = {
    "mode": {"practice", "test", "probe"}, "world": {"reset", "persistent"}, "level": {"L1", "L2", "L3"},
    "lang": {"en", "fa"},
    "in_basis": {"inside_inspection", "depth_consistent", "box_only", "none"},
    "verdict": {"found", "visible", "partly", "not-visible", "unsure"},
    "recovery": {"fresh_point", "sidestep", "step_back", "lesson", "procedure"},
    "agent_verdict": {"success", "failure", "unknown"},
    "state_seen": {"open", "closed", "?"},
    "tag": {"privileged-terminal-goal-check"},
}
# field -> type.  tok = ASCII token, id = minted id, int/num/bool, xyz = [x, y, z], toks = list of tokens,
# xyzs = list of xyz, text = free text (task_context only), json = recursive allowlisted dict, enum:<name>
KIND_FIELDS = {
    "episode_start": {"mode": "enum:mode", "world": "enum:world"},
    "task_context": {"level": "enum:level", "lang": "enum:lang", "text": "text", "concepts": "json"},
    "step_start": {"i": "int", "verb": "tok", "otype": "tok", "to": "tok", "pre": "json"},
    "attempt": {"i": "int", "k": "int", "recovery": "enum:recovery", "source": "tok", "ok": "bool", "cause": "tok"},
    "application": {"app": "id", "record": "id", "i": "int", "k": "int", "pre": "json", "retrieval": "id"},
    "sighting": {"otype": "tok", "state": "tok", "conf": "num", "detector": "tok", "track": "id", "xyz": "xyz",
                 "xyz_ok": "bool", "view": "id", "obs": "id", "in_rtype": "tok", "in_basis": "enum:in_basis",
                 "in_xyz": "xyz"},
    "receptacle_seen": {"rtype": "tok", "xyz": "xyz", "extent": "json", "view": "id", "obs": "id",
                        "n_boxes_same_type": "int", "covers": "xyzs"},
    "inspection": {"rtype": "tok", "xyz": "xyz", "open_ev": "int", "verdict": "enum:verdict", "frac": "num",
                   "thirds": "int", "support": "int", "range_m": "num", "views": "int", "crouched": "bool",
                   "detected": "toks", "targets": "toks", "detector_rev": "tok"},
    "open": {"rtype": "tok", "xyz": "xyz", "ok": "bool", "by_me": "bool", "state_seen": "enum:state_seen"},
    "close": {"rtype": "tok", "xyz": "xyz", "ok": "bool", "by_me": "bool", "state_seen": "enum:state_seen"},
    "pickup": {"track": "id", "otype": "tok", "ok": "bool"},
    "put": {"track": "id", "otype": "tok", "rtype": "tok", "xyz": "xyz", "ok": "bool", "resighted": "bool"},
    "drop": {"track": "id", "otype": "tok", "xyz": "xyz", "ok": "bool", "resighted": "bool"},
    "refusal": {"verb": "tok", "otype": "tok", "rtype": "tok", "i": "int", "k": "int", "cause": "tok"},
    "contradiction": {"track": "id", "otype": "tok", "views": "int", "src": "toks"},
    "step_outcome": {"i": "int", "ok": "bool", "cause": "tok", "under": "json", "retries": "int", "prims": "int"},
    "terminal": {"exec_ok": "bool", "agent_verdict": "enum:agent_verdict", "prims": "int", "opens": "int",
                 "walk_m": "num", "invalid": "bool"},
    "admission_verdict": {"goal_ok": "bool", "tag": "enum:tag"},
    "reset_world": {"reason": "tok"},
    "episode_end": {},
}
KINDS = tuple(KIND_FIELDS)
REQUIRED = {"episode_start": ("mode", "world"), "sighting": ("otype", "track"), "receptacle_seen": ("rtype", "xyz"),
            "inspection": ("rtype", "xyz"), "open": ("rtype", "xyz"), "close": ("rtype", "xyz"),
            "pickup": ("track",), "put": ("track", "otype", "xyz"), "drop": ("track", "otype", "xyz"),
            "contradiction": ("track",), "terminal": ("exec_ok",), "admission_verdict": ("goal_ok", "tag")}


class EventError(ValueError):
    """The event violates the contract (unknown kind/field, wrong type, or privileged material)."""


def _tok(v, where, ident=False):
    if not isinstance(v, str):
        raise EventError(f"{where}: expected string")
    if SIM_ID_RE.search(v) or "|" in v or not v.isascii():
        raise EventError(f"{where}: simulator-id-like or non-ASCII string refused")
    if not (ID_TOKEN if ident else TOKEN_RE).match(v):
        raise EventError(f"{where}: {v[:24]!r} is not an admitted token")
    return v


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise EventError(f"{where}: expected a finite number")
    return float(v)


def _xyz(v, where):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise EventError(f"{where}: expected [x, y, z]")
    return [_num(x, f"{where}[{i}]") for i, x in enumerate(v)]


def _json(v, where, depth=0):
    if depth > 4:
        raise EventError(f"{where}: nested too deep")
    if v is None or isinstance(v, bool) or isinstance(v, int):
        return v
    if isinstance(v, float):
        return _num(v, where)
    if isinstance(v, str):
        return _tok(v, where)
    if isinstance(v, (list, tuple)):
        if len(v) > 128:
            raise EventError(f"{where}: list too long")
        return [_json(x, where, depth + 1) for x in v]
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str) or not re.match(r"^[a-z][a-z0-9_]{0,40}$", k) or is_privileged_name(k):
                raise EventError(f"{where}: key {k!r} not admitted")
            out[k] = _json(x, f"{where}.{k}", depth + 1)
        return out
    raise EventError(f"{where}: unsupported value type {type(v).__name__}")


def validate(ev: dict) -> dict:
    """Return a normalised copy of `ev` or raise EventError."""
    if not isinstance(ev, dict):
        raise EventError("event must be a dict")
    kind = ev.get("kind")
    if kind not in KIND_FIELDS:
        raise EventError(f"unknown event kind {kind!r}")
    for k in ev:
        if not isinstance(k, str):
            raise EventError("non-string key")
        if is_privileged_name(k) and not (kind == "admission_verdict" and k == "goal_ok"):
            raise EventError(f"privileged field {k!r} refused")
    allowed = set(ENVELOPE) | set(KIND_FIELDS[kind])
    extra = set(ev) - allowed
    if extra:
        raise EventError(f"{kind}: unknown fields {sorted(extra)}")
    out = {"schema": ev.get("schema", SCHEMA)}
    if out["schema"] != SCHEMA:
        raise EventError(f"unsupported schema {out['schema']!r}")
    for name in ("ev", "epoch", "t"):
        v = ev.get(name, 0 if name == "t" else None)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise EventError(f"envelope {name}: expected non-negative int")
        out[name] = v
    out["kind"] = kind
    for name in ("kitchen", "frame", "map_rev", "episode"):
        out[name] = _tok(ev.get(name), f"envelope {name}", ident=True)
    for req in REQUIRED.get(kind, ()):
        if req not in ev or ev[req] is None:
            raise EventError(f"{kind}: missing required field {req!r}")
    for name, typ in KIND_FIELDS[kind].items():
        if name not in ev:
            continue
        v, where = ev[name], f"{kind}.{name}"
        if v is None:
            out[name] = None
        elif typ == "tok":
            out[name] = _tok(v, where)
        elif typ == "id":
            out[name] = _tok(v, where, ident=True)
        elif typ == "int":
            if isinstance(v, bool) or not isinstance(v, int):
                raise EventError(f"{where}: expected int")
            out[name] = v
        elif typ == "num":
            out[name] = _num(v, where)
        elif typ == "bool":
            if not isinstance(v, bool):
                raise EventError(f"{where}: expected bool")
            out[name] = v
        elif typ == "xyz":
            out[name] = _xyz(v, where)
        elif typ == "xyzs":
            if not isinstance(v, (list, tuple)) or len(v) > 16:
                raise EventError(f"{where}: expected a short list of xyz")
            out[name] = [_xyz(x, f"{where}[{i}]") for i, x in enumerate(v)]
        elif typ == "toks":
            if not isinstance(v, (list, tuple)) or len(v) > 64:
                raise EventError(f"{where}: expected a list of tokens")
            out[name] = [_tok(x, f"{where}[{i}]") for i, x in enumerate(v)]
        elif typ == "json":
            out[name] = _json(v, where)
        elif typ == "text":
            if not isinstance(v, str) or len(v) > 2000 or SIM_ID_RE.search(v):
                raise EventError(f"{where}: free text refused")
            out[name] = v
        elif typ.startswith("enum:"):
            if v not in ENUMS[typ[5:]]:
                raise EventError(f"{where}: {v!r} not one of {sorted(ENUMS[typ[5:]])}")
            out[name] = v
        else:                                                    # pragma: no cover
            raise EventError(f"{where}: unknown type {typ}")
    return out


def ev_hash(episode: str, ev: int) -> str:
    """The 16-hex evidence reference for one event (stored in evidence.src_ev)."""
    return sha256_hex(f"{episode}#{ev}")[:16]
