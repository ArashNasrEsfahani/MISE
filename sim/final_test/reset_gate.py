"""Gate 12.4 (prereg A14): are two independent restores of the same presentation the same world for the task?

Pure functions over plain dicts, stdlib only. The executor reads the world inside ``h.scoring()`` and hands the
raw object list to :func:`capture_state`; nothing here touches a simulator, so the decision path is not involved.
No object, scene or kitchen constant appears in this file (A14: "no object is named"); everything is derived
from the object list itself.

What is compared (A14 c)
    symbolic (must be identical):   object identities (the ``name``, not the renamed ``objectId``), types,
        capabilities, receptacle parents and contents (containment / support), the open / toggled / sliced /
        dirty / filled / broken / cooked / picked-up flags, derived accessibility, the set of types seen in the
        start sweep, the post-sweep summary (byte-identical), the evaluator labels at reset, and the
        reference-plan feasibility verdicts.
    continuous:   per-object position (Euclidean, metres) and rotation (geodesic angle,
        degrees) within tau_pos_m / tau_rot_deg.
    agent pose:   exact (rounded to 1e-4 so that float printing cannot matter).
    NOTE (A14a, 2026-10-11, before calibration): the post-sweep summary embeds the agent's own 2-decimal (x, z) depth estimates of SIGHTED objects.
        A14a relaxes the byte identity of A14 c-ii as follows: with each (x, z) pair replaced by a placeholder the lines must be byte-identical
        (types, states, places, step numbers, order, count; a malformed field fails), and each corresponding pair must be within tau_pos_m in EUCLIDEAN
        distance. That distance also enters the calibration statistic (max_dpos), so tau covers it.
    rotation:     the geodesic angle between the two orientations (quaternions, Unity ZXY), not a per-axis Euler difference.
    diagnostic:   the strict world hash is stored under ``diagnostics`` and never compared.

Tolerance protocol (A14 a-d), what fills docs/final-test/reset_tolerances.json and in which order
    0. ``init-ceilings``  writes the A14a ceilings (position, rotation, justification, date). The numbers come from
       the dated prereg amendment A14a, which is written BEFORE any calibration restore; status becomes "ceilings".
    1. ``calibrate``      reads 12 dev items x 5 independent restores (a json ``{item: [state, ...]}``), compares all
       10 pairs per item, takes the observed maxima, rounds them UP to 0.05 m / 5 deg (never below one step, so
       that float noise cannot breach a zero tolerance) and writes them as tau. If a symbolic breach occurs, or
       a tau exceeds a ceiling, the status is "failed" and stays failed: the remedy is a reset fix in every arm
       and a new amendment, never a wider tolerance (the writer refuses to replace a calibrated / failed file).
    2. ``validate``       reads 24 pairs (each presentation captured twice from fresh restores), applies tau and
       records ``validation = {pairs, pass, ok}``; ok iff 24 of 24 pass.
    3. Phase C (A14 d)    ``load_tolerances()`` + :func:`compare` between the Phase A capture and the state right
       after the fixed sweep of each execution; any breach blocks that execution -- except under the SIGNED amendment A34-A2
       (2026-10-15; `compare_a34` / `a34_override`): an observation-only breach set (seen_types / summary / summary.coords) of two valid
       captures with identical pipeline settings and identical world-only snapshots at stored precision is overridden and logged as obs_drift.
       ``validate-a34`` runs the A34-A2 rule-4 dev validation read-only (it never writes the tolerance file).

    python sim/final_test/reset_gate.py show
    python sim/final_test/reset_gate.py init-ceilings --pos-m P --rot-deg R --justification TEXT --dated YYYY-MM-DD
    python sim/final_test/reset_gate.py calibrate --restores restores.json
    python sim/final_test/reset_gate.py validate --pairs pairs.json        # [[stateA, stateB], ...]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import re
import sys
import tempfile

STATE_SCHEMA = "final-test.reset-state/1"
TOL_SCHEMA = "final-test.reset-tolerances/1"
STEP_POS_M = 0.05                   # rounding step of tau (A14 b)
STEP_ROT_DEG = 5.0
EXPECTED_RESTORES = 5               # per item (A14 b)
EXPECTED_PAIRS = 24                 # A14 c
AGENT_POS_TOL_M = 0.001             # A14b: numerical-precision tolerance on the agent pose (fixed, not calibrated)
AGENT_ROT_TOL_DEG = 0.01
STATUSES = ("unset", "ceilings", "calibrated", "failed")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TOL_PATH = os.path.join(ROOT, "docs", "final-test", "reset_tolerances.json")

#: boolean state flags of an AI2-THOR object that the gate treats as symbolic
FLAGS = ("isOpen", "isToggled", "isSliced", "isDirty", "isFilledWithLiquid", "isBroken", "isCooked", "isUsedUp",
         "isPickedUp")
#: capability flags (what an object can do); they decide action feasibility
CAPS = ("openable", "toggleable", "sliceable", "pickupable", "receptacle", "dirtyable", "breakable",
        "canFillWithLiquid", "cookable", "moveable")

#: reference-plan action -> (capability the target needs, capability the destination needs)
_ACTION_CAPS = {
    "pickup": ("pickupable", None), "pick_and_place": ("pickupable", "receptacle"), "put_in": ("pickupable", "receptacle"),
    "open": ("openable", None), "close": ("openable", None), "toggle_on": ("toggleable", None),
    "toggle_off": ("toggleable", None), "slice": ("sliceable", None), "fill": ("canFillWithLiquid", None),
    "empty": ("canFillWithLiquid", None), "dirty": ("dirtyable", None), "clean": ("dirtyable", None),
    "wash": ("pickupable", None), "heat": ("pickupable", None), "cook": ("cookable", None),
    "break": ("breakable", None), "drop": (None, None), "go_to": (None, None), "search_for": (None, None),
    "crouch": (None, None), "stand": (None, None), "close_all": (None, None),
}


# ------------------------------------------------------------------------------------------------- small helpers
def canon(x) -> str:
    """Canonical json, identical to final_test/matrix/common.canon."""
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def self_sha(obj: dict) -> str:
    """sha256 of the object without its own 'sha256' field (same convention as common.self_sha)."""
    return sha256_text(canon({k: v for k, v in obj.items() if k != "sha256"}))


def _num(v, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: not a number: {v!r}")
    if not math.isfinite(f):
        raise ValueError(f"{what}: not finite: {v!r}")
    return f


def _xyz(d, what: str, default=None) -> list:
    if d is None:
        if default is None:
            raise ValueError(f"{what}: missing")
        return list(default)
    if isinstance(d, dict):
        return [round(_num(d.get(k, 0.0), f"{what}.{k}"), 5) for k in ("x", "y", "z")]
    seq = list(d)
    if len(seq) == 1:
        seq = [0.0, seq[0], 0.0]                   # a bare yaw
    if len(seq) != 3:
        raise ValueError(f"{what}: need x,y,z")
    return [round(_num(v, what), 5) for v in seq]


def _ceil_step(x: float, step: float) -> float:
    k = math.ceil(round(x / step, 9))
    return round(max(1, k) * step, 6)              # never below one step


def _quat(e: list) -> tuple:
    """Unity Euler (x, y, z degrees) -> unit quaternion (w, x, y, z). Unity applies Z, then X, then Y (extrinsic): q = qy * qx * qz."""
    hx, hy, hz = (math.radians(float(v)) / 2.0 for v in e)
    qx, qy, qz = (math.cos(hx), math.sin(hx), 0.0, 0.0), (math.cos(hy), 0.0, math.sin(hy), 0.0), (math.cos(hz), 0.0, 0.0, math.sin(hz))

    def mul(p, q):
        pw, px, py, pz = p
        qw, qx_, qy_, qz_ = q
        return (pw * qw - px * qx_ - py * qy_ - pz * qz_, pw * qx_ + px * qw + py * qz_ - pz * qy_,
                pw * qy_ - px * qz_ + py * qw + pz * qx_, pw * qz_ + px * qy_ - py * qx_ + pz * qw)
    return mul(mul(qy, qx), qz)


def _rot_delta(a: list, b: list) -> float:
    """E-fix: the geodesic angle (degrees) between two orientations, NOT a per-axis Euler difference (an Euler triple is not unique:
    near pitch +-90 deg, (x, y, z) and (180 - x, y + 180, z + 180) are the same orientation). Stable atan2 form of 2*acos(|q1.q2|)."""
    p, q = _quat(a), _quat(b)
    w = p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]          # relative rotation conj(p) * q
    x = p[0] * q[1] - p[1] * q[0] - p[2] * q[3] + p[3] * q[2]
    y = p[0] * q[2] + p[1] * q[3] - p[2] * q[0] - p[3] * q[1]
    z = p[0] * q[3] - p[1] * q[2] + p[2] * q[1] - p[3] * q[0]
    return math.degrees(2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w)))


def _pos_delta(a: list, b: list) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


# ------------------------------------------------------------------------------------------------- capture
def _agent_pose(agent: dict) -> dict:
    if not isinstance(agent, dict):
        raise ValueError("agent: dict expected")
    h = agent.get("horizon", agent.get("cameraHorizon"))
    return {"position": [round(v, 4) for v in _xyz(agent.get("position"), "agent.position")],
            "rotation": [round(v, 4) for v in _xyz(agent.get("rotation"), "agent.rotation", default=(0, 0, 0))],
            "horizon": round(_num(0.0 if h is None else h, "agent.horizon"), 4),
            "standing": bool(agent.get("standing", agent.get("isStanding", True)))}


def _name_of(oid: str, id_to_name: dict) -> str:
    if oid in id_to_name:
        return id_to_name[oid]
    return "?" + str(oid).split("|")[0]            # unknown id: keep only the type prefix (ids get renamed)


def capture_state(objects: list, agent: dict, *, summary: list, seen_types: list, labels: dict, feasibility: dict,
                  world_hash: str | None = None) -> dict:
    """Plain-dict snapshot of the world right after the fixed start sweep. Identity is the object ``name``."""
    id_to_name, objs = {}, {}
    for o in objects:
        name = o.get("name")
        if not name:
            raise ValueError("object without a name")
        if name in objs:
            raise ValueError(f"duplicate object name {name!r}")
        id_to_name[o.get("objectId", name)] = name
        objs[name] = o
    out = {}
    for name, o in sorted(objs.items()):
        flags = {f: bool(o.get(f)) for f in FLAGS}
        out[name] = {
            "type": str(o.get("objectType", "")),
            "caps": sorted(c for c in CAPS if o.get(c)),
            "flags": flags,
            "parents": sorted(_name_of(p, id_to_name) for p in (o.get("parentReceptacles") or [])),
            "contains": sorted(_name_of(p, id_to_name) for p in (o.get("receptacleObjectIds") or [])),
            "pos": _xyz(o.get("position"), f"{name}.position"),
            "rot": _xyz(o.get("rotation"), f"{name}.rotation", default=(0, 0, 0)),
        }
    for name, rec in out.items():                  # derived accessibility: no closed openable ancestor
        rec["accessible"] = _accessible(name, out)
    state = {"schema": STATE_SCHEMA, "objects": out, "agent": _agent_pose(agent),
             "summary": [str(s) for s in summary], "summary_sha": sha256_text("\n".join(str(s) for s in summary)),
             "seen_types": sorted({str(t) for t in seen_types}), "labels": json.loads(canon(labels)),
             "feasibility": json.loads(canon(feasibility)), "diagnostics": {"world_hash": world_hash}}
    state["symbolic_sha"] = sha256_text(canon(symbolic_part(state)))
    return state


def _accessible(name: str, objs: dict) -> bool:
    seen, stack = set(), list(objs[name]["parents"])
    if objs[name]["flags"].get("isPickedUp"):
        return True
    while stack:
        p = stack.pop()
        if p in seen or p not in objs:
            continue
        seen.add(p)
        r = objs[p]
        if "openable" in r["caps"] and not r["flags"].get("isOpen"):
            return False
        stack.extend(r["parents"])
    return True


def _lead(line) -> str:
    """The object type a summary line starts with (AI2-THOR type names hold no space)."""
    return str(line).split(" ", 1)[0]


def relevant_types(state: dict) -> list:
    """A14b frozen dependency rule: the task-relevant types of a state, read from the state itself (no gold is needed at compare time).
    T0 = the object / destination types of the reference-plan steps (feasibility record) + the object types named by the goal clauses (labels).
    Relevant = T0 + the types of every transitive container ancestor (parent receptacle chain) of every instance of a T0 type."""
    objs = state["objects"]
    present = {r["type"] for r in objs.values()}
    t0 = set()
    for step in (state.get("feasibility") or {}).values():
        for k in ("object", "to"):
            info = step.get(k) if isinstance(step, dict) else None
            if isinstance(info, dict) and info.get("type"):
                t0.add(str(info["type"]))
    for clause in ((state.get("labels") or {}).get("goal_clauses") or {}):
        for tok in re.findall(r"[A-Za-z]+", str(clause)):
            if tok in present:
                t0.add(tok)
    out = set(t0)
    for r in objs.values():
        if r["type"] in t0:
            stack, seen = list(r["parents"]), set()
            while stack:
                q = stack.pop()
                if q in seen or q not in objs:
                    continue
                seen.add(q)
                out.add(objs[q]["type"])
                stack.extend(objs[q]["parents"])
    return sorted(out)


def symbolic_part(state: dict) -> dict:
    """Everything that must be identical (no continuous values, no diagnostics). A14b: the agent pose enters by horizon / standing only
    (position and rotation are compared to numerical precision), and the summary by its task-relevant rows with the (x, z) pairs removed."""
    rel = set(relevant_types(state))
    return {"objects": {n: {k: v for k, v in r.items() if k not in ("pos", "rot")} for n, r in state["objects"].items()},
            "agent": {"horizon": state["agent"]["horizon"], "standing": state["agent"]["standing"]},
            "summary_task_rows": _strip_coords([x for x in state["summary"] if _lead(x) in rel]),
            "seen_types": state["seen_types"], "labels": state["labels"], "feasibility": state["feasibility"], "task_types": sorted(rel)}


def feasibility_of(state: dict, ref_plan: list) -> dict:
    """Per reference-plan step, precondition verdicts from the SYMBOLIC state only (no simulator action).

    A verdict is made of counts and booleans, so it is identical across restores exactly when the task-relevant
    world is: how many instances of the target type exist, how many are reachable without opening anything, and
    whether the type has the capability the action needs (same for the destination). It is a necessary check,
    not a proof that the plan succeeds."""
    objs = state["objects"]

    def info(typ, cap):
        if not typ:
            return None
        inst = [r for r in objs.values() if r["type"] == typ]
        return {"type": typ, "n": len(inst), "n_accessible": sum(1 for r in inst if r["accessible"]),
                "capable": (None if cap is None else any(cap in r["caps"] for r in inst))}

    out = {}
    for i, step in enumerate(ref_plan or []):
        s = step if isinstance(step, dict) else {"action": str(step)}
        act = str(s.get("action"))
        tcap, dcap = _ACTION_CAPS.get(act, (None, None))
        out[f"{i:02d}"] = {"action": act, "object": info(s.get("object"), tcap), "to": info(s.get("to"), dcap)}
    return out


# ------------------------------------------------------------------------------------------------- compare
def _tau(tol: dict):
    if not isinstance(tol, dict):
        raise ValueError("tol: dict expected")
    tp, tr = tol.get("tau_pos_m"), tol.get("tau_rot_deg")
    if tp is None or tr is None:
        raise ValueError("tolerances are not set (tau_pos_m / tau_rot_deg missing): load_tolerances(require='calibrated')")
    return _num(tp, "tau_pos_m"), _num(tr, "tau_rot_deg")


def _short(v, n=160):
    s = canon(v)
    return s if len(s) <= n else s[:n] + "..."


_COORD_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def _strip_coords(lines) -> list:
    """The summary lines with the two-decimal (x, z) estimates replaced by a placeholder (A14a summary rule)."""
    return [_COORD_RE.sub("(#)", str(x)) for x in lines]


def _summary_coords(lines) -> list:
    """All (x, z) pairs of the summary, in order (one list per line)."""
    return [[(float(m.group(1)), float(m.group(2))) for m in _COORD_RE.finditer(str(x))] for x in lines]


def summary_distance(a_lines, b_lines):
    """A14a summary rule. Returns (structure_ok, max_euclid_m, why). structure_ok: the lines are byte-identical once only the
    designated (x, z) pairs are replaced by a placeholder AND each line holds the same number of pairs. max_euclid_m: the largest
    Euclidean distance between corresponding (x, z) pairs (0.0 when there is none). A malformed field cannot match the pattern, so it
    stays in the stripped text and fails the byte-identity of the structure."""
    a_lines, b_lines = [str(x) for x in a_lines], [str(x) for x in b_lines]
    if _strip_coords(a_lines) != _strip_coords(b_lines):
        return False, None, "lines differ outside the coordinate pairs"
    ca, cb = _summary_coords(a_lines), _summary_coords(b_lines)
    worst = 0.0
    for la_, lb_ in zip(ca, cb):
        if len(la_) != len(lb_):
            return False, None, "different number of coordinate pairs in a line"
        for (x1, z1), (x2, z2) in zip(la_, lb_):
            worst = max(worst, math.hypot(x1 - x2, z1 - z2))
    return True, worst, ""


def _agent_close(a: dict, b: dict):
    """A14b: agent pose to numerical precision: position <= AGENT_POS_TOL_M per coordinate, rotation (geodesic, wrapped) <= AGENT_ROT_TOL_DEG,
    horizon and standing exact. Returns (ok, max position delta, rotation delta)."""
    dpos = max(abs(p - q) for p, q in zip(a["position"], b["position"]))
    drot = _rot_delta(a["rotation"], b["rotation"])
    ok = (dpos <= AGENT_POS_TOL_M + 1e-9 and drot <= AGENT_ROT_TOL_DEG + 1e-9 and a["horizon"] == b["horizon"]
          and a["standing"] == b["standing"])
    return ok, dpos, drot


def _compare(a: dict, b: dict, tau_pos, tau_rot) -> dict:
    for k, s in (("a", a), ("b", b)):
        if not isinstance(s, dict) or s.get("schema") != STATE_SCHEMA:
            raise ValueError(f"state {k}: not a {STATE_SCHEMA}")
    br = []
    oa, ob = a["objects"], b["objects"]
    only_a, only_b = sorted(set(oa) - set(ob)), sorted(set(ob) - set(oa))
    if only_a or only_b:
        br.append({"kind": "objects.set", "only_a": only_a[:20], "only_b": only_b[:20]})
    maxp = maxr = 0.0
    for n in sorted(set(oa) & set(ob)):
        ra, rb = oa[n], ob[n]
        for key in ("type", "caps", "parents", "contains", "accessible"):
            if ra[key] != rb[key]:
                br.append({"kind": f"symbolic.{key}", "object": n, "a": _short(ra[key]), "b": _short(rb[key])})
        for f in sorted(set(ra["flags"]) | set(rb["flags"])):
            if ra["flags"].get(f) != rb["flags"].get(f):
                br.append({"kind": f"symbolic.{f}", "object": n, "a": ra["flags"].get(f), "b": rb["flags"].get(f)})
        dp, dr = _pos_delta(ra["pos"], rb["pos"]), _rot_delta(ra["rot"], rb["rot"])
        maxp, maxr = max(maxp, dp), max(maxr, dr)
        if tau_pos is not None and dp > tau_pos + 1e-9:
            br.append({"kind": "continuous.pos", "object": n, "delta": round(dp, 4), "tau": tau_pos})
        if tau_rot is not None and dr > tau_rot + 1e-9:
            br.append({"kind": "continuous.rot", "object": n, "delta": round(dr, 3), "tau": tau_rot})
    aok, apos, arot = _agent_close(a["agent"], b["agent"])
    if not aok:
        br.append({"kind": "agent_pose", "a": a["agent"], "b": b["agent"], "dpos": round(apos, 6), "drot": round(arot, 6)})
    if a["seen_types"] != b["seen_types"]:
        br.append({"kind": "seen_types", "only_a": sorted(set(a["seen_types"]) - set(b["seen_types"])),
                   "only_b": sorted(set(b["seen_types"]) - set(a["seen_types"]))})
    ra, rb = relevant_types(a), relevant_types(b)
    if ra != rb:
        br.append({"kind": "task_types", "only_a": sorted(set(ra) - set(rb)), "only_b": sorted(set(rb) - set(ra))})
    rel = set(ra) & set(rb)
    ta_, tb_ = [x for x in a["summary"] if _lead(x) in rel], [x for x in b["summary"] if _lead(x) in rel]
    ok_s, dist_s, why_s = summary_distance(ta_, tb_)
    if not ok_s:
        idx = next((i for i, (x, y) in enumerate(itertools.zip_longest(_strip_coords(ta_), _strip_coords(tb_))) if x != y), None)
        br.append({"kind": "summary", "first_diff_line": idx, "len_a": len(ta_), "len_b": len(tb_), "why": why_s})
    else:
        maxp = max(maxp, dist_s)                       # A14a: calibration takes the max over object deltas AND summary-coordinate distance
        if tau_pos is not None and dist_s > tau_pos + 1e-9:
            br.append({"kind": "summary.coords", "delta": round(dist_s, 4), "tau": tau_pos})
    # A14b diagnostic (never a criterion): the rows of types outside the relevant set that differ between the two summaries
    oa = collections.Counter(_strip_coords([x for x in a["summary"] if _lead(x) not in rel]))
    ob = collections.Counter(_strip_coords([x for x in b["summary"] if _lead(x) not in rel]))
    dif = (oa - ob) + (ob - oa)
    diag = {"other_rows_differing": sum(dif.values()), "other_types": sorted({_lead(k) for k in dif})}
    if canon(a["labels"]) != canon(b["labels"]):
        br.append({"kind": "labels", "a": _short(a["labels"]), "b": _short(b["labels"])})
    if canon(a["feasibility"]) != canon(b["feasibility"]):
        steps = sorted(k for k in set(a["feasibility"]) | set(b["feasibility"])
                       if canon(a["feasibility"].get(k)) != canon(b["feasibility"].get(k)))
        br.append({"kind": "feasibility", "steps": steps})
    return {"ok": not br, "breaches": br, "max_dpos": round(maxp, 5), "max_drot": round(maxr, 4), "diag": diag}


def compare(a: dict, b: dict, tol: dict) -> dict:
    """The gate. ``tol`` carries tau_pos_m / tau_rot_deg (a loaded tolerances file works). Any breach -> ok False."""
    tp, tr = _tau(tol)
    return _compare(a, b, tp, tr)


# ------------------------------------------------------------------------------------------------- A34-A2 (signed 2026-10-15)
#: A34-A2 (Arash, 2026-10-15, signatures.json "A34-A2+A35"; text docs/final-test/amendments/A34-A2_A35.md). The registered comparison
#: (`compare`) is UNCHANGED and always runs first. A failing comparison is OVERRIDDEN (the execution proceeds) iff ALL hold:
#:   (a) the complete breach set is non-empty and confined to the observation-derived kinds below;
#:   (b) both captures are valid (no missing, malformed or non-finite data; self-consistent shas; summary syntax; labels / feasibility
#:       complete for the presentation) and both observation pipelines are valid (health evidence present, detector ran, no state-stage
#:       failure) and ran with identical frozen settings (treatment identity + effective detector configuration + sweep length);
#:   (c) the canonical world-only snapshots are identical at the stored precision (objects: identity, type, pose, parents / contents,
#:       capabilities, every state flag, accessibility; evaluator labels; feasibility verdicts; agent position, rotation, horizon, standing).
#: Anything else, or any missing data, stays blocking. Each override is recorded in full as `obs_drift`. No tolerance changes (rule 4).
A34_AMENDMENT = "A34-A2"
#: the ONE canonical append-only log of every rule-4 validation (a failure stays on record: Common terms 4; Astra ACC2-IMPL-r2 5)
A34_VALIDATION_LOG = os.path.join(ROOT, "docs", "final-test", "amendments", "A34-A2_A35.validation.jsonl")
OBS_KINDS = ("seen_types", "summary", "summary.coords")
OBS_PIPELINE_SCHEMA = "obs-pipeline/1"
_OBJ_KEYS = ("type", "caps", "flags", "parents", "contains", "pos", "rot", "accessible")
#: the ONE line shape the post-sweep summary has (4160 / 4160 lines of the attempt-1 + rehearsal-v3 captures): "Type [state] on/in Place at
#: (x.xx, z.zz), step N" -- a line that does not match it exactly is malformed (Astra ACC2-IMPL-r2 1)
_LINE_RE = re.compile(r"^[A-Z][A-Za-z]* \[[^\[\]\n]*\] on/in (?:\?|[A-Za-z][A-Za-z0-9_]*) at \(-?\d+\.\d{2}, -?\d+\.\d{2}\), step \d+$")
_NONFINITE_RE = re.compile(r"(?i)\b(nan|inf|infinity)\b")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _finite_list(v, n=3) -> bool:
    return isinstance(v, list) and len(v) == n and all(_finite(x) for x in v)


def _all_finite(x) -> bool:
    """Every number anywhere inside a JSON value is finite (labels / feasibility / health)."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(isinstance(k, str) and _all_finite(v) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return all(_all_finite(v) for v in x)
    return False


def _summary_line_ok(line: str) -> bool:
    """The full summary-line grammar (_LINE_RE); no non-finite token."""
    return bool(_LINE_RE.match(line)) and not _NONFINITE_RE.search(line)


def _feas_info_ok(v) -> bool:
    if v is None:
        return True
    return (isinstance(v, dict) and sorted(v) == ["capable", "n", "n_accessible", "type"] and isinstance(v.get("type"), str) and v["type"]
            and isinstance(v.get("n"), int) and not isinstance(v["n"], bool)
            and isinstance(v.get("n_accessible"), int) and not isinstance(v["n_accessible"], bool) and 0 <= v["n_accessible"] <= v["n"]
            and (v.get("capable") is None or isinstance(v.get("capable"), bool)))


def state_problems(s, *, presentation: dict | None = None) -> list:
    """A34-A2 2(b): validity of one captured state. [] = valid. Every field the override relies on must be present, of the right kind and
    finite; the stored summary / symbolic shas must match the content; summary lines must have the A14a syntax; the evaluator labels must
    hold a non-empty goal-clause map (str -> bool) and the reset flag; the feasibility record must hold one well-formed verdict per
    reference-plan step. `presentation` = {"plan": [steps], "goal_clauses": [clause texts]} binds them to the ACTUAL presentation: exactly one complete
    verdict per reference-plan step with that step's action / object type / destination type, and the labels keyed by exactly the
    presentation's own goal clauses (Astra ACC2-IMPL-r2 1 / r3 1). An override always passes it; without it the state is never override-valid (see a34_override)."""
    out = []
    if not isinstance(s, dict) or s.get("schema") != STATE_SCHEMA:
        return [f"not a {STATE_SCHEMA}"]
    objs = s.get("objects")
    if not isinstance(objs, dict) or not objs:
        out.append("objects missing or empty")
        objs = {}
    for n, r in objs.items():
        tag = f"object {str(n)[:40]}"
        if not isinstance(r, dict) or any(k not in r for k in _OBJ_KEYS):
            out.append(f"{tag}: missing field")
            continue
        if not isinstance(r["type"], str) or not r["type"]:
            out.append(f"{tag}: type")
        if not _finite_list(r["pos"]) or not _finite_list(r["rot"]):
            out.append(f"{tag}: pose missing / not finite")
        if not isinstance(r["flags"], dict) or sorted(r["flags"]) != sorted(FLAGS) or not all(isinstance(v, bool) for v in r["flags"].values()):
            out.append(f"{tag}: flags")
        if not all(isinstance(r[k], list) and all(isinstance(x, str) for x in r[k]) for k in ("caps", "parents", "contains")):
            out.append(f"{tag}: caps / parents / contains")
        if not isinstance(r["accessible"], bool):
            out.append(f"{tag}: accessible")
    ag = s.get("agent")
    if (not isinstance(ag, dict) or not _finite_list(ag.get("position")) or not _finite_list(ag.get("rotation"))
            or not _finite(ag.get("horizon")) or not isinstance(ag.get("standing"), bool)):
        out.append("agent pose missing / malformed / not finite")
    sm = s.get("summary")
    if not isinstance(sm, list) or not sm or not all(isinstance(x, str) and x for x in sm):
        out.append("summary missing / empty / malformed")
    else:
        if s.get("summary_sha") != sha256_text("\n".join(sm)):
            out.append("summary_sha does not match the summary")
        bad = [i for i, x in enumerate(sm) if not _summary_line_ok(x)]
        if bad:
            out.append(f"summary line(s) {bad[:4]} malformed (coordinate pair / non-finite token)")
    st = s.get("seen_types")
    if not isinstance(st, list) or not all(isinstance(x, str) and x for x in st) or st != sorted(set(st)):
        out.append("seen_types missing / malformed")
    lab = s.get("labels")
    gc = lab.get("goal_clauses") if isinstance(lab, dict) else None
    if (not isinstance(gc, dict) or not gc or not all(isinstance(k, str) and k and isinstance(v, bool) for k, v in gc.items())
            or not isinstance(lab.get("goal_true_at_reset"), bool) or not _all_finite(lab) or sorted(lab) != ["goal_clauses", "goal_true_at_reset"]):
        out.append("labels missing / incomplete (exactly goal_clauses str -> bool and goal_true_at_reset bool)")
    elif presentation is not None:
        want = presentation.get("goal_clauses") if isinstance(presentation, dict) else None
        if (not isinstance(want, list) or not want or sorted(gc) != sorted(set(want))
                or lab["goal_true_at_reset"] != all(gc.values())):
            out.append("labels are not keyed by the presentation's own goal clauses (or the reset flag is inconsistent)")
    fe = s.get("feasibility")
    if not isinstance(fe, dict) or not fe or not _all_finite(fe):
        out.append("feasibility missing / empty / not finite")
    else:
        keys_ok = sorted(fe) == [f"{i:02d}" for i in range(len(fe))]
        if not keys_ok:
            out.append("feasibility steps are not 00..NN")
        if not all(isinstance(v, dict) and isinstance(v.get("action"), str) and v["action"] and "object" in v and "to" in v
                   and _feas_info_ok(v.get("object")) and _feas_info_ok(v.get("to")) for v in fe.values()):
            out.append("feasibility verdict malformed")
        elif presentation is not None and keys_ok:
            plan = presentation.get("plan") if isinstance(presentation, dict) else None
            if not isinstance(plan, list) or len(fe) != len(plan):
                out.append("feasibility does not have one verdict per step of the presentation's reference plan")
            else:
                for i, st_ in enumerate(plan):
                    st_ = st_ if isinstance(st_, dict) else {"action": str(st_)}
                    v = fe[f"{i:02d}"]
                    if (v["action"] != str(st_.get("action")) or (v["object"] or {}).get("type") != st_.get("object")
                            or (v["to"] or {}).get("type") != st_.get("to")):
                        out.append(f"feasibility step {i:02d} is not the presentation's reference-plan step")
                        break
    if not out:
        try:
            if s.get("symbolic_sha") != sha256_text(canon(symbolic_part(s))):
                out.append("symbolic_sha does not match the state")
        except Exception as e:                                     # noqa: BLE001 -- unreadable = invalid
            out.append(f"symbolic part unreadable ({type(e).__name__})")
    return out


def pipeline_problems(obs) -> list:
    """A34-A2 2(b): one episode's observation-pipeline record (obs_health.record) is present and healthy. [] = valid."""
    if not isinstance(obs, dict) or obs.get("schema") != OBS_PIPELINE_SCHEMA:
        return ["observation-pipeline record missing"]
    out = []
    if not isinstance(obs.get("treatment_id"), str) or not _HEX64.match(obs["treatment_id"]):
        out.append("treatment id missing")
    sp = obs.get("sweep_prims")
    if not isinstance(sp, int) or isinstance(sp, bool) or sp <= 0:
        out.append("sweep length missing")
    h = obs.get("health")
    if not isinstance(h, dict) or not _all_finite(h):
        return out + ["pipeline health evidence missing"]
    dc, sc, se = h.get("detect_calls"), h.get("state_calls"), h.get("state_errors")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (dc, sc, se)):
        out.append("pipeline health counters malformed")
    else:
        if dc == 0:
            out.append("the detector never ran during the sweep")
        if se != 0:
            out.append(f"{se} state-classifier failure(s) during the sweep (swallowed by the detector)")
    cf = h.get("configs")
    ok_cf = isinstance(cf, list) and cf and all(
        isinstance(c, list) and len(c) == 5 and c[0] in ("auto", "off", "always") and isinstance(c[1], str) and c[1]
        and (c[2] is None or (isinstance(c[2], str) and c[2])) and isinstance(c[3], bool) and _finite(c[4]) and 0 < c[4] < 1 for c in cf)
    if not ok_cf:
        out.append("effective detector configuration missing / malformed")
    elif not out:
        n_state = sum(1 for c in cf if c[3])
        if n_state == len(cf) and sc != dc:
            out.append(f"the state stage is enabled but ran {sc} time(s) for {dc} detector call(s)")
        elif n_state == 0 and sc != 0:
            out.append("state-stage calls without a state-enabled configuration")
        elif 0 < n_state < len(cf) and not 0 < sc <= dc:
            out.append("state-stage call count inconsistent with the mixed configuration")
    return out


def pipeline_settings(obs: dict) -> str:
    """The frozen settings an observation pipeline ran with: treatment identity (code + weights), the effective detector configuration(s)
    and the sweep length. Equal on both sides is condition 2(b)'s 'identical frozen settings'."""
    return sha256_text(canon({"treatment_id": obs.get("treatment_id"), "configs": (obs.get("health") or {}).get("configs"),
                              "sweep_prims": obs.get("sweep_prims")}))


def world_snapshot(s: dict) -> dict:
    """A34-A2 2(c): the canonical WORLD-ONLY snapshot at the stored precision (objects 1e-5, agent 1e-4, as captured). Nothing derived
    from the observation pipeline (summary, seen types, task types) enters it."""
    return {"objects": {n: {k: r[k] for k in _OBJ_KEYS} for n, r in sorted(s["objects"].items())},
            "agent": {k: s["agent"][k] for k in ("position", "rotation", "horizon", "standing")},
            "labels": s["labels"], "feasibility": s["feasibility"]}


def obs_drift_record(a: dict, b: dict, breaches: list, obs_a=None, obs_b=None) -> dict:
    """A34-A2 rule 3, COMPLETE (nothing truncated): types only in A / only in B, every differing summary row, every coordinate delta,
    every breach, both pipelines' settings ids."""
    ca, cb = collections.Counter(a["summary"]), collections.Counter(b["summary"])
    rows_a, rows_b = sorted((ca - cb).elements()), sorted((cb - ca).elements())
    deltas = []
    sa, sb = _strip_coords(a["summary"]), _strip_coords(b["summary"])
    ib = collections.defaultdict(list)
    for j, x in enumerate(sb):
        ib[x].append(j)
    for i, x in enumerate(sa):                                     # rows equal up to the (x, z) pairs: their coordinate delta
        if ib.get(x):
            j = ib[x].pop(0)
            pa, pb = _summary_coords([a["summary"][i]])[0], _summary_coords([b["summary"][j]])[0]
            if len(pa) == len(pb):
                d = max((math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(pa, pb)), default=0.0)
                if d > 0:
                    deltas.append({"row": _lead(a["summary"][i]), "delta_m": round(d, 4)})
    return {"amendment": A34_AMENDMENT, "kinds": sorted({str(x.get("kind")) for x in breaches}),
            "types_only_a": sorted(set(a["seen_types"]) - set(b["seen_types"])),
            "types_only_b": sorted(set(b["seen_types"]) - set(a["seen_types"])),
            "summary_rows_only_a": rows_a, "summary_rows_only_b": rows_b, "n_rows_differing": len(rows_a) + len(rows_b),
            "positional_diffs": [{"i": i, "a": x, "b": y} for i, (x, y) in enumerate(itertools.zip_longest(a["summary"], b["summary"])) if x != y],
            "coord_deltas": deltas, "max_coord_delta_m": max((d["delta_m"] for d in deltas), default=0.0),
            "breaches": [dict(x) for x in breaches], "n_breaches": len(breaches),
            "summary_sha_a": a["summary_sha"], "summary_sha_b": b["summary_sha"],
            "settings": pipeline_settings(obs_a) if isinstance(obs_a, dict) else None}


def obs_drift_problems(d, breaches=None) -> list:
    """Rule 3 record check (used by acc_runner criterion (d)): required fields, kinds, and internal consistency. [] = complete."""
    if not isinstance(d, dict) or d.get("amendment") != A34_AMENDMENT:
        return ["no A34-A2 obs_drift record"]
    out = []
    for k, t in (("kinds", list), ("types_only_a", list), ("types_only_b", list), ("summary_rows_only_a", list), ("summary_rows_only_b", list),
                 ("coord_deltas", list), ("breaches", list), ("n_rows_differing", int), ("n_breaches", int), ("max_coord_delta_m", (int, float)),
                 ("summary_sha_a", str), ("summary_sha_b", str), ("settings", str), ("positional_diffs", list)):
        if not isinstance(d.get(k), t) or isinstance(d.get(k), bool):
            out.append(f"obs_drift.{k} missing / malformed")
    if out:
        return out
    br = d["breaches"]
    if not br or len(br) != d["n_breaches"] or any(not isinstance(x, dict) or x.get("kind") not in OBS_KINDS for x in br):
        out.append("obs_drift breaches missing / not observation-only")
    if d["kinds"] != sorted({x.get("kind") for x in br if isinstance(x, dict)}):
        out.append("obs_drift kinds do not match its breaches")
    if breaches is not None and [dict(x) for x in breaches][:len(br)] != br[:len(breaches)]:
        out.append("obs_drift breaches differ from the state check's breaches")
    if d["n_rows_differing"] != len(d["summary_rows_only_a"]) + len(d["summary_rows_only_b"]):
        out.append("obs_drift row count inconsistent")
    if "seen_types" in d["kinds"] and not (d["types_only_a"] or d["types_only_b"]):
        out.append("obs_drift: a seen_types breach without the differing types")
    pdl = d["positional_diffs"]
    if not all(isinstance(x, dict) and sorted(x) == ["a", "b", "i"] and isinstance(x["i"], int) and not isinstance(x["i"], bool) and x["i"] >= 0
               and all(v is None or (isinstance(v, str) and _summary_line_ok(v)) for v in (x["a"], x["b"])) and x["a"] != x["b"] for x in pdl):
        out.append("obs_drift positional differences malformed")
    elif [x["i"] for x in pdl] != sorted({x["i"] for x in pdl}):
        out.append("obs_drift positional differences not in order")
    if "summary" in d["kinds"] and not pdl:
        out.append("obs_drift: a summary breach without the differing rows")
    if "summary.coords" in d["kinds"] and not d["coord_deltas"]:
        out.append("obs_drift: a summary.coords breach without coordinate deltas")
    if not _all_finite(d):
        out.append("obs_drift holds a non-finite number")
    for k in ("types_only_a", "types_only_b"):
        if not all(isinstance(x, str) and x for x in d[k]) or d[k] != sorted(set(x for x in d[k] if isinstance(x, str))):
            out.append(f"obs_drift.{k} entries malformed")
    if {x for x in d["types_only_a"] if isinstance(x, str)} & {x for x in d["types_only_b"] if isinstance(x, str)}:
        out.append("obs_drift: a type both only-in-A and only-in-B")
    for k in ("summary_rows_only_a", "summary_rows_only_b"):
        if not all(isinstance(x, str) and _summary_line_ok(x) for x in d[k]):
            out.append(f"obs_drift.{k} entries malformed")
    if not (_HEX64.match(d["summary_sha_a"]) and _HEX64.match(d["summary_sha_b"]) and _HEX64.match(d["settings"])):
        out.append("obs_drift shas malformed")
    if (d["summary_sha_a"] == d["summary_sha_b"]) != (not d.get("positional_diffs")):
        out.append("obs_drift summary shas inconsistent with the recorded row differences")
    dl = d["coord_deltas"]
    if not all(isinstance(x, dict) and sorted(x) == ["delta_m", "row"] and isinstance(x["row"], str) and x["row"] and _finite(x["delta_m"])
               and x["delta_m"] > 0 for x in dl):
        out.append("obs_drift coordinate deltas malformed (each {row, delta_m > 0})")
    elif not _finite(d["max_coord_delta_m"]) or d["max_coord_delta_m"] != max((x["delta_m"] for x in dl), default=0.0):
        out.append("obs_drift maximum coordinate delta inconsistent")
    for x in br:
        if not isinstance(x, dict):
            continue
        if x.get("kind") == "seen_types" and (x.get("only_a") != d["types_only_a"] or x.get("only_b") != d["types_only_b"]):
            out.append("obs_drift types do not reconcile with the seen_types breach")
        if x.get("kind") == "summary.coords" and not (_finite(x.get("delta")) and x["delta"] <= d["max_coord_delta_m"] + 1e-9):
            out.append("obs_drift deltas do not reconcile with the summary.coords breach")
    if "seen_types" not in d["kinds"] and (d["types_only_a"] or d["types_only_b"]):
        out.append("obs_drift: differing types without a seen_types breach")
    return out


def a34_override(a, b, breaches, *, obs_a, obs_b, presentation: dict | None = None) -> dict:
    """A34-A2 rule 2. `breaches` = the COMPLETE breach set of the registered comparison plus every further pre-execution check (e.g. the
    holding checks of execute.replay_precondition). obs_a / obs_b = the two episodes' observation-pipeline records (obs_health.record).
    -> {"override": bool, "why": [reasons it does not apply], "obs_drift": record|None}. An empty breach set is no override."""
    if not isinstance(breaches, list) or not breaches:
        return {"override": False, "why": ["no breach: an ordinary pass, nothing to override"], "obs_drift": None}
    why = []
    kinds = [x.get("kind") if isinstance(x, dict) else None for x in breaches]
    bad = sorted({str(k) for k in kinds if k not in OBS_KINDS})
    if bad:
        why.append(f"(a) non-observation breach kind(s): {bad[:6]}")
    invalid = False
    if not isinstance(presentation, dict):
        why.append("(b) the presentation (reference plan + goal count) the captures must be bound to is missing")
    for tag, s in (("a", a), ("b", b)):
        p = state_problems(s, presentation=presentation if isinstance(presentation, dict) else None)
        if p:
            invalid = True
            why.append(f"(b) capture {tag} invalid: {p[:3]}")
    pa, pb = pipeline_problems(obs_a), pipeline_problems(obs_b)
    if pa:
        why.append(f"(b) observation pipeline a invalid: {pa[:3]}")
    if pb:
        why.append(f"(b) observation pipeline b invalid: {pb[:3]}")
    if not pa and not pb and pipeline_settings(obs_a) != pipeline_settings(obs_b):
        why.append("(b) the observation pipelines ran with different settings (treatment identity / detector configuration / sweep length)")
    if not invalid:
        try:
            wa, wb = world_snapshot(a), world_snapshot(b)
            if canon(wa) != canon(wb):
                diff = sorted(n for n in set(wa["objects"]) | set(wb["objects"]) if wa["objects"].get(n) != wb["objects"].get(n))
                parts = ([f"objects {diff[:4]}"] if diff else []) + [k for k in ("agent", "labels", "feasibility") if canon(wa[k]) != canon(wb[k])]
                why.append(f"(c) world snapshots differ: {parts[:5]}")
        except Exception as e:                                     # noqa: BLE001 -- missing data blocks
            why.append(f"(c) world snapshot unreadable ({type(e).__name__})")
    if why:
        return {"override": False, "why": why, "obs_drift": None}
    return {"override": True, "why": [], "obs_drift": obs_drift_record(a, b, breaches, obs_a, obs_b)}


def compare_a34(a: dict, b: dict, tol: dict, *, obs_a, obs_b, presentation: dict | None = None, extra_breaches=()) -> dict:
    """The Phase C gate under A34-A2: the registered `compare` first (unchanged), then the override. -> the registered result plus
    `registered_ok` (the registered verdict incl. extra_breaches), `override`, `override_why`, `obs_drift`; `ok` = registered_ok or override."""
    r = compare(a, b, tol)
    br = list(r["breaches"]) + [dict(x) for x in extra_breaches]
    out = dict(r, breaches=br, registered_ok=not br, override=False, override_why=[], obs_drift=None)
    if br:
        ov = a34_override(a, b, br, obs_a=obs_a, obs_b=obs_b, presentation=presentation)
        out.update(override=ov["override"], override_why=ov["why"], obs_drift=ov["obs_drift"])
    out["ok"] = out["registered_ok"] or out["override"]
    return out


def validate_a34(pairs: list, tol: dict, *, obs: list, presentations: list | None = None, expected: int = EXPECTED_PAIRS) -> dict:
    """A34-A2 rule 4, the prospective dev validation: PASS iff there are exactly `expected` pairs and in every pair the registered
    comparison passes or is overridden under rule 2. `obs` = [(obs_a, obs_b)] per pair. Writes nothing (the registered tolerance file and
    its one-shot validation stay untouched)."""
    if len(obs) != len(pairs) or (presentations is not None and len(presentations) != len(pairs)):
        raise ValueError("one observation-pipeline pair (and presentation) per state pair")
    passed = overridden = 0
    bad, drift = [], []
    kinds = collections.Counter()
    pipe_bad = []
    for k, ((a, b), (oa, ob)) in enumerate(zip(pairs, obs)):
        pl = presentations[k] if presentations is not None else None
        for tag, o in (("a", oa), ("b", ob)):
            if pipeline_problems(o):
                pipe_bad.append({"pair": k, "side": tag, "why": pipeline_problems(o)[:3]})
        try:
            r = compare_a34(a, b, tol, obs_a=oa, obs_b=ob, presentation=pl)
        except Exception as e:                                     # noqa: BLE001 -- malformed data: the pair is blocked, never an escape
            bad.append({"pair": k, "breaches": [], "why": [f"malformed data ({type(e).__name__}: {str(e)[:80]})"]})
            continue
        if r["registered_ok"]:
            passed += 1
        elif r["override"]:
            overridden += 1
            drift.append(dict(r["obs_drift"], pair=k))
            for x in r["obs_drift"]["kinds"]:
                kinds[x] += 1
        else:
            bad.append({"pair": k, "breaches": r["breaches"][:5], "why": r["override_why"]})
    n = len(pairs)
    return {"amendment": A34_AMENDMENT, "pairs": n, "expected": expected, "registered_pass": passed, "overridden": overridden,
            "blocked": len(bad), "ok": n == expected and passed + overridden == n, "obs_drift_by_kind": dict(sorted(kinds.items())),
            "obs_drift_types": dict(sorted(collections.Counter(t for d in drift for t in d["types_only_a"] + d["types_only_b"]).items())),
            "obs_drift": drift, "failures": bad, "pipeline_problems": pipe_bad}


# ------------------------------------------------------------------------------------------------- calibration
def _acc_other(acc: dict, r: dict) -> None:
    d = r.get("diag") or {}
    n = int(d.get("other_rows_differing") or 0)
    if n:
        acc["pairs_affected"] += 1
        acc["rows_differing"] += n
        for t in d.get("other_types") or []:
            acc["by_type"][t] += 1


def _fin_other(acc: dict) -> dict:
    return {"pairs_affected": acc["pairs_affected"], "rows_differing": acc["rows_differing"], "pairs_by_type": dict(sorted(acc["by_type"].items()))}


def calibrate(restores: dict, ceilings: dict, *, restores_per_item: int = EXPECTED_RESTORES, items: int | None = None) -> dict:
    """tau = observed maxima over all pairs of independent restores, rounded UP to 0.05 m / 5 deg.

    Fails (status "failed") if any pair breaks the symbolic part, or if a tau exceeds its ceiling. The ceilings
    are numbers from the dated amendment A14a, fixed before the restores were taken."""
    cp, cr = _num(ceilings.get("pos_m"), "ceilings.pos_m"), _num(ceilings.get("rot_deg"), "ceilings.rot_deg")
    if cp <= 0 or cr <= 0:
        raise ValueError("ceilings must be positive")
    if items is not None and len(restores) != items:
        raise ValueError(f"expected {items} items, got {len(restores)}")
    maxp = maxr = 0.0
    pairs, sym, per_item, worst = 0, [], {}, {"pos": None, "rot": None}
    other = {"pairs_affected": 0, "rows_differing": 0, "by_type": collections.Counter()}      # A14b diagnostic, never a criterion
    for item in sorted(restores):
        states = restores[item]
        if len(states) != restores_per_item:
            raise ValueError(f"{item}: expected {restores_per_item} restores, got {len(states)}")
        ip = ir = 0.0
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            r = _compare(a, b, None, None)
            pairs += 1
            ip, ir = max(ip, r["max_dpos"]), max(ir, r["max_drot"])
            _acc_other(other, r)
            if not r["ok"]:
                sym.append({"item": item, "pair": [i, j], "breaches": r["breaches"][:5]})
        per_item[item] = {"max_pos_m": ip, "max_rot_deg": ir}
        if ip > maxp:
            maxp, worst["pos"] = ip, item
        if ir > maxr:
            maxr, worst["rot"] = ir, item
    tp, tr = _ceil_step(maxp, STEP_POS_M), _ceil_step(maxr, STEP_ROT_DEG)
    reasons = []
    if sym:
        reasons.append(f"{len(sym)} pair(s) break the symbolic part")
    if tp > cp + 1e-9:
        reasons.append(f"tau_pos_m {tp} exceeds the ceiling {cp}")
    if tr > cr + 1e-9:
        reasons.append(f"tau_rot_deg {tr} exceeds the ceiling {cr}")
    return {"status": "failed" if reasons else "calibrated", "reasons": reasons, "tau_pos_m": tp, "tau_rot_deg": tr,
            "calibration": {"items": len(restores), "restores_per_item": restores_per_item, "pairs": pairs,
                            "observed_max_pos_m": round(maxp, 5), "observed_max_rot_deg": round(maxr, 4),
                            "worst_item": worst, "per_item": per_item, "symbolic_breaches": sym[:10],
                            "summary_other_rows": _fin_other(other)}}


def validate(pairs: list, tol: dict, *, expected: int = EXPECTED_PAIRS) -> dict:
    """A14 c: PASS iff every pair passes and there are exactly ``expected`` pairs."""
    tp, tr = _tau(tol)
    passed, bad = 0, []
    other = {"pairs_affected": 0, "rows_differing": 0, "by_type": collections.Counter()}
    for k, (a, b) in enumerate(pairs):
        r = _compare(a, b, tp, tr)
        _acc_other(other, r)
        if r["ok"]:
            passed += 1
        else:
            bad.append({"pair": k, "breaches": r["breaches"][:5]})
    return {"pairs": len(pairs), "pass": passed, "ok": len(pairs) == expected and passed == len(pairs),
            "failures": bad[:10], "summary_other_rows": _fin_other(other)}


# ------------------------------------------------------------------------------------------------- tolerance file
def template() -> dict:
    d = {"schema": TOL_SCHEMA, "amendment": "A14a", "status": "unset",
         "ceilings": None, "calibration": None, "tau_pos_m": None, "tau_rot_deg": None, "agent_pose": "exact",
         "validation": None, "validation_log": [],
         "protocol": ["0 init-ceilings: A14a numbers (dated BEFORE any calibration restore) -> status ceilings",
                      "1 calibrate: 12 dev items x 5 restores, 10 pairs per item; tau = observed max rounded up to 0.05 m / 5 deg; "
                      "status calibrated, or failed if a symbolic breach or tau > ceiling (failed is final: new amendment + reset fix)",
                      "2 validate: 24 presentations captured twice from fresh restores; ok iff 24/24 pairs pass; ONE-SHOT (a recorded validation is never "
                      "replaced; a failed one makes the file failed)",
                      "3 Phase C: compare(phase_A_state, post_sweep_state, load_tolerances()) before every execution"]}
    d["sha256"] = self_sha(d)
    return d


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tolerances(path: str | None = None, require: str = "calibrated") -> dict:
    """Load and verify the tolerance file. require: 'validated' (calibrated AND validation ok with 24/24 pairs: the ONLY level an
    execution may run under), 'calibrated' (taus set; for the validate step itself), 'ceilings' (ceilings set), 'any'. A failed or
    unset file never satisfies 'calibrated' or 'validated'."""
    path = path or DEFAULT_TOL_PATH
    d = _read(path)
    if d.get("schema") != TOL_SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r}")
    if d.get("sha256") != self_sha(d):
        raise ValueError(f"{path}: self sha mismatch (file edited by hand?)")
    st = d.get("status")
    if st not in STATUSES:
        raise ValueError(f"{path}: unknown status {st!r}")
    if require == "calibrated":
        if st != "calibrated":
            raise ValueError(f"{path}: status is {st!r}, not 'calibrated'")
        _tau(d)
    elif require == "validated":                        # E-fix: what Phase C / the freeze / matrix_run preflight must use
        if st != "calibrated":
            raise ValueError(f"{path}: status is {st!r}, not 'calibrated'")
        _tau(d)
        v = d.get("validation")
        if not isinstance(v, dict) or not v.get("ok") or v.get("pairs") != EXPECTED_PAIRS or v.get("pass") != EXPECTED_PAIRS:
            raise ValueError(f"{path}: gate 12.4 is not validated ({EXPECTED_PAIRS}/{EXPECTED_PAIRS} pairs required): {v!r}")
    elif require == "ceilings":
        if st not in ("ceilings", "calibrated"):
            raise ValueError(f"{path}: status is {st!r}, ceilings not set")
    elif require != "any":
        raise ValueError(f"require={require!r}")
    return d


def write_tolerances(d: dict, path: str | None = None, *, allow_replace: bool = False) -> str:
    """Atomic write with the self sha. A calibrated or failed file is final: replacing it needs ``allow_replace``
    (a new amendment, and the caller keeps the old copy under _checkpoints)."""
    path = path or DEFAULT_TOL_PATH
    if d.get("status") not in STATUSES:
        raise ValueError("unknown status")
    if os.path.exists(path):
        old = _read(path)
        if old.get("status") in ("calibrated", "failed") and not allow_replace:
            if not (d.get("tau_pos_m") == old.get("tau_pos_m") and d.get("tau_rot_deg") == old.get("tau_rot_deg")):
                raise PermissionError(f"{path} is {old['status']!r}: final; a change needs a new amendment")
            if old.get("validation") is not None and d.get("validation") != old["validation"]:
                raise PermissionError(f"{path}: validation is one-shot (A14 c): a recorded validation cannot be replaced; "
                                      "a failure is fixed by a new amendment and restarts at (b)")
            if d.get("status") != old["status"] and not (old["status"] == "calibrated" and d.get("status") == "failed"
                                                         and old.get("validation") is None and (d.get("validation") or {}).get("ok") is False):
                raise PermissionError(f"{path} is {old['status']!r}: final; only a FAILED validation may turn calibrated into failed")
            ol = old.get("validation_log") or []
            if (d.get("validation_log") or [])[:len(ol)] != ol:
                raise PermissionError(f"{path}: validation_log is append-only")
    d = dict(d)
    d["sha256"] = self_sha(d)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------------- A14a check
DEFAULT_PREREG = os.path.join(ROOT, "docs", "final-test", "preregistration.md")


def check_prereg_a14a(prereg_path: str, pos_m: float, rot_deg: float, dated: str) -> dict:
    """E-fix: init-ceilings only accepts ceilings that the DATED prereg amendment A14a states. The prereg must hold a heading line
    starting with 'A14a' (any markdown level) whose section (up to the next heading) names the date, a position number in metres and a
    rotation number in degrees equal to the arguments. Returns {"prereg_sha256", "section_sha256"} for the tolerance file; raises
    ValueError otherwise (no amendment, no ceilings)."""
    with open(prereg_path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(r"^#{1,6}\s*A14a\b", ln)), None)
    if start is None:
        raise ValueError(f"{prereg_path}: no '### A14a' amendment heading yet: write the dated amendment BEFORE any calibration restore")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^#{1,6}\s", lines[i])), len(lines))
    sec = "\n".join(lines[start:end])
    if dated not in sec:
        raise ValueError(f"A14a does not carry the date {dated!r}")
    pos = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*m(?![a-zA-Z])", sec)}
    rot = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:deg\b|degrees?\b|\u00b0)", sec)}
    if float(pos_m) not in pos:
        raise ValueError(f"A14a does not state a position ceiling of {pos_m} m (it states {sorted(pos)})")
    if float(rot_deg) not in rot:
        raise ValueError(f"A14a does not state a rotation ceiling of {rot_deg} deg (it states {sorted(rot)})")
    return {"prereg_sha256": sha256_text(text), "section_sha256": sha256_text(sec)}


# ------------------------------------------------------------------------------------------------- CLI
def _relroot(p: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(p), ROOT).replace(os.sep, "/")
    except ValueError:                                     # another drive (a temporary test directory)
        return os.path.abspath(p).replace(os.sep, "/")


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tolerances", default=DEFAULT_TOL_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    p = sub.add_parser("init-ceilings")
    p.add_argument("--pos-m", type=float, required=True)
    p.add_argument("--rot-deg", type=float, required=True)
    p.add_argument("--justification", required=True)
    p.add_argument("--dated", required=True)
    p.add_argument("--prereg", default=DEFAULT_PREREG, help="the preregistration holding the dated A14a amendment (checked, not optional)")
    p = sub.add_parser("calibrate")
    p.add_argument("--restores", required=True)
    p.add_argument("--items", type=int, default=12)
    p = sub.add_parser("validate")
    p.add_argument("--pairs", required=True)
    p = sub.add_parser("validate-a34", help="A34-A2 rule 4 dev validation (read-only: writes only --out, never the tolerance file)")
    p.add_argument("--pairs", required=True, help='{"pairs": [[stateA, stateB], ...], "obs": [[obsA, obsB], ...], "plan_lens": [...]}')
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate-a34":
        d = load_tolerances(a.tolerances, "validated")
        with open(a.pairs, "rb") as fh:
            pairs_sha = hashlib.sha256(fh.read()).hexdigest()
        try:
            doc = _read(a.pairs)
            v = validate_a34([tuple(p) for p in doc["pairs"]], d, obs=[tuple(s) for s in doc["obs"]], presentations=doc.get("presentation_specs"))
        except Exception as e:                                     # noqa: BLE001 -- recorded as a FAILED validation (terminal), then re-raised
            os.makedirs(os.path.dirname(A34_VALIDATION_LOG), exist_ok=True)
            with open(A34_VALIDATION_LOG, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"amendment": A34_AMENDMENT, "pairs_file_sha256": pairs_sha, "ok": False,
                                    "error": f"{type(e).__name__}: {str(e)[:160]}"}, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            raise
        v.update(presentations=doc.get("presentations"), pairs_file_sha256=pairs_sha, tolerances_sha256=d.get("sha256"),
                 tau_pos_m=d.get("tau_pos_m"), tau_rot_deg=d.get("tau_rot_deg"))
        body = json.dumps(v, indent=1, ensure_ascii=False) + "\n"
        # the canonical log FIRST (mandatory, fixed path): a failure is recorded before any result file exists
        os.makedirs(os.path.dirname(A34_VALIDATION_LOG), exist_ok=True)
        with open(A34_VALIDATION_LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"amendment": A34_AMENDMENT, "pairs_file_sha256": pairs_sha, "result_file": _relroot(a.out),
                                "result_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "ok": v["ok"], "pairs": v["pairs"],
                                "registered_pass": v["registered_pass"], "overridden": v["overridden"], "blocked": v["blocked"]}, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        print(json.dumps({k: v[k] for k in ("pairs", "registered_pass", "overridden", "blocked", "ok", "obs_drift_by_kind", "obs_drift_types")}))
        return 0 if v["ok"] else 2
    if a.cmd == "show":
        print(json.dumps(load_tolerances(a.tolerances, "any"), indent=2, ensure_ascii=False))
        return 0
    if a.cmd == "init-ceilings":
        d = load_tolerances(a.tolerances, "any")
        if d["status"] != "unset":
            raise SystemExit(f"status is {d['status']!r}; ceilings are written once, before any calibration restore")
        if a.pos_m <= 0 or a.rot_deg <= 0 or not a.justification.strip():
            raise SystemExit("ceilings need positive numbers and a justification")
        try:
            pre = check_prereg_a14a(a.prereg, a.pos_m, a.rot_deg, a.dated)
        except ValueError as e:
            raise SystemExit(f"refused: {e}")
        d.update(status="ceilings", ceilings={"pos_m": a.pos_m, "rot_deg": a.rot_deg,
                                              "justification": a.justification, "dated": a.dated, **pre})
        write_tolerances(d, a.tolerances)
        print("ceilings written")
        return 0
    if a.cmd == "calibrate":
        d = load_tolerances(a.tolerances, "ceilings")
        if d["status"] != "ceilings":
            raise SystemExit(f"status is {d['status']!r}: calibration is done once")
        r = calibrate(_read(a.restores), d["ceilings"], items=a.items)
        d.update(status=r["status"], tau_pos_m=r["tau_pos_m"], tau_rot_deg=r["tau_rot_deg"], calibration=r["calibration"],
                 failure_reasons=r["reasons"])
        write_tolerances(d, a.tolerances)
        print(r["status"], r["tau_pos_m"], r["tau_rot_deg"], r["reasons"])
        return 0 if r["status"] == "calibrated" else 2
    d = load_tolerances(a.tolerances, "calibrated")
    if d.get("validation") is not None:
        raise SystemExit("refused: validation is one-shot (A14 c); a recorded validation is never re-run or replaced. "
                         "A failure needs a new amendment and restarts at (b) with new restores")
    with open(a.pairs, "rb") as fh:
        pairs_sha = hashlib.sha256(fh.read()).hexdigest()
    v = validate([tuple(p) for p in _read(a.pairs)], d)
    d["validation"] = dict(v, pairs_file_sha256=pairs_sha)
    d["validation_log"] = list(d.get("validation_log") or []) + [dict(d["validation"])]
    if not v["ok"]:
        d["status"] = "failed"                                 # A14 c: any breach fails; final
        d["failure_reasons"] = [f"validation: {v['pass']}/{v['pairs']} pairs pass (24/24 required)"]
    write_tolerances(d, a.tolerances)
    print(json.dumps(v, ensure_ascii=False))
    return 0 if v["ok"] else 2


if __name__ == "__main__":
    sys.exit(_cli())
