"""Selftest for sim/final_test/reset_gate.py (stage E3-resetgate). Offline: no simulator, no network, no API.

    python sim/final_test/_reset_gate_selftest.py
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "matrix"))

import reset_gate as R                                # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ------------------------------------------------------------------------------------------ synthetic world
def obj(name, typ, oid, pos, parents=(), contains=(), rot=(0, 0, 0), **kw):
    d = {"name": name, "objectType": typ, "objectId": oid, "position": dict(zip("xyz", pos)),
         "rotation": dict(zip("xyz", rot)), "parentReceptacles": list(parents), "receptacleObjectIds": list(contains)}
    d.update(kw)
    return d


def world(suffix="a", shift=0.0, rot_shift=0.0):
    """Cupboard (openable, closed) holding a cup; a loose widget on the counter; ids differ per restore."""
    return [
        obj("Box_1", "Box", f"Box|+01.00|+00.00|+01.00{suffix}", (1.0, 0.0, 1.0), contains=[f"Cup|+01.00|+00.50|+01.00{suffix}"],
            openable=True, receptacle=True, isOpen=False),
        obj("Cup_1", "Cup", f"Cup|+01.00|+00.50|+01.00{suffix}", (1.0, 0.5, 1.0), parents=[f"Box|+01.00|+00.00|+01.00{suffix}"],
            pickupable=True, canFillWithLiquid=True),
        obj("Counter_1", "Counter", f"Counter|+02.00|+00.90|+01.00{suffix}", (2.0, 0.9, 1.0), receptacle=True,
            contains=[f"Widget|+02.00|+01.00|+01.00{suffix}"]),
        obj("Widget_1", "Widget", f"Widget|+02.00|+01.00|+01.00{suffix}", (2.0 + shift, 1.0, 1.0), rot=(0, 10 + rot_shift, 0),
            parents=[f"Counter|+02.00|+00.90|+01.00{suffix}"], pickupable=True, moveable=True),
    ]


AGENT = {"position": {"x": 0.5, "y": 0.9, "z": 0.25}, "rotation": {"x": 0, "y": 90, "z": 0}, "horizon": 0, "standing": True}
PLAN = [{"action": "pickup", "object": "Cup", "to": None}, {"action": "put_in", "object": "Cup", "to": "Box"},
        {"action": "slice", "object": "Cup", "to": None}]


SUMM = ("Cup [clean] on/in Box at (1.00, 1.00), step 3", "Widget [clean] on/in Counter at (2.00, 1.00), step 4")


def cap(objs=None, agent=None, summary=SUMM, seen=("Cup", "Box"), labels=None, plan=PLAN, suffix="a"):
    objs = objs if objs is not None else world(suffix)
    labels = labels if labels is not None else {"goal_at_reset": False, "applicable": True}
    # feasibility is computed from the state itself, as the executor will do
    s0 = R.capture_state(objs, agent or AGENT, summary=list(summary), seen_types=list(seen), labels=labels, feasibility={})
    return R.capture_state(objs, agent or AGENT, summary=list(summary), seen_types=list(seen), labels=labels,
                           feasibility=R.feasibility_of(s0, plan))


TOL = {"tau_pos_m": 0.05, "tau_rot_deg": 5.0}
kinds = lambda r: sorted({b["kind"] for b in r["breaches"]})

# ------------------------------------------------------------------------------------------ static guards
src = open(os.path.join(HERE, "reset_gate.py"), encoding="utf-8").read()
tree = ast.parse(src)
imports = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
          {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
check("static: stdlib only (no ai2thor / honest / executor import)", not imports & {"ai2thor", "honest", "step5_live", "eval", "numpy", "torch"}, str(imports))
strings = " ".join(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
code_only = re.sub(r'""".*?"""', "", src, flags=re.S)
check("static: no scene / kitchen constant in the code", not re.search(r"FloorPlan|_physics|iTHOR", code_only), "")
check("static: no concrete object type is named", not re.search(
    r"\b(Apple|Fridge|Mug|Cup|Sponge|DishSponge|Pot|Pan|Knife|Spoon|Plate|Bowl|Egg|Bread|Potato|Tomato|Lettuce)\b", code_only), "")
check("static: does not read a key or the network", not re.search(r"urllib|requests|socket|API_KEY|environ", code_only), "")

# ------------------------------------------------------------------------------------------ capture / identity
s1, s2 = cap(suffix="a"), cap(suffix="b")                    # same world, different objectIds (InitialRandomSpawn renames)
check("capture: schema", s1["schema"] == R.STATE_SCHEMA)
check("capture: identity is the name, parents are names", s1["objects"]["Cup_1"]["parents"] == ["Box_1"], str(s1["objects"]["Cup_1"]["parents"]))
check("capture: renamed objectIds do not matter", R.compare(s1, s2, TOL)["ok"] and s1["symbolic_sha"] == s2["symbolic_sha"])
check("capture: cup inside a closed openable is not accessible", not s1["objects"]["Cup_1"]["accessible"] and s1["objects"]["Widget_1"]["accessible"])
check("capture: unknown parent id keeps only the type", R.capture_state([obj("W", "Widget", "W|1", (0, 0, 0), parents=["Zed|9|9|9"])], AGENT,
      summary=[], seen_types=[], labels={}, feasibility={})["objects"]["W"]["parents"] == ["?Zed"])
check("capture: duplicate names are refused", raises(lambda: R.capture_state([obj("W", "W", "a", (0, 0, 0)), obj("W", "W", "b", (0, 0, 0))],
      AGENT, summary=[], seen_types=[], labels={}, feasibility={}), ValueError))
check("capture: nan position is refused", raises(lambda: R.capture_state([obj("W", "W", "a", (float("nan"), 0, 0))], AGENT,
      summary=[], seen_types=[], labels={}, feasibility={}), ValueError))
check("capture: object order does not matter", R.compare(s1, cap(objs=list(reversed(world("a")))), TOL)["ok"])
check("capture: world hash is a diagnostic only", R.compare(R.capture_state(world(), AGENT, summary=["x"], seen_types=[], labels={}, feasibility={}, world_hash="h1"),
      R.capture_state(world(), AGENT, summary=["x"], seen_types=[], labels={}, feasibility={}, world_hash="h2"), TOL)["ok"])

# ------------------------------------------------------------------------------------------ continuous part
below = cap(objs=world("b", shift=0.04))
above = cap(objs=world("b", shift=0.30))
at = cap(objs=world("b", shift=0.05))
r = R.compare(s1, below, TOL)
check("continuous: a settled loose object BELOW tau passes", r["ok"] and abs(r["max_dpos"] - 0.04) < 1e-6, str(r))
check("continuous: AT tau passes (inclusive)", R.compare(s1, at, TOL)["ok"])
r = R.compare(s1, above, TOL)
check("continuous: ABOVE tau breaches, names the object, and nothing else", not r["ok"] and kinds(r) == ["continuous.pos"] and r["breaches"][0]["object"] == "Widget_1", str(r))
check("continuous: a 0.44 m settle (the pilot case) breaches", not R.compare(s1, cap(objs=world("b", shift=0.44)), TOL)["ok"])
check("continuous: rotation below tau passes", R.compare(s1, cap(objs=world("b", rot_shift=4.0)), TOL)["ok"])
check("continuous: rotation above tau breaches", kinds(R.compare(s1, cap(objs=world("b", rot_shift=20.0)), TOL)) == ["continuous.rot"])
w = world("b"); w[3]["rotation"]["y"] = 355.0; wa = world("a"); wa[3]["rotation"]["y"] = 358.0     # wrap-around: 3 deg apart
check("continuous: rotation wraps at 360", R.compare(cap(objs=wa), cap(objs=w, suffix="b"), TOL)["ok"])
check("continuous: position is Euclidean", abs(R.compare(s1, cap(objs=[*world("b")[:3], obj("Widget_1", "Widget", "Widget|+02.00|+01.00|+01.00b", (2.03, 1.04, 1.0), rot=(0, 10, 0),
      parents=["Counter|+02.00|+00.90|+01.00b"], pickupable=True, moveable=True)]), TOL)["max_dpos"] - 0.05) < 1e-6)
check("continuous: compare is symmetric", R.compare(s1, above, TOL)["ok"] == R.compare(above, s1, TOL)["ok"])
check("continuous: tolerances not set -> refused", raises(lambda: R.compare(s1, s2, {"tau_pos_m": None, "tau_rot_deg": None}), ValueError))

# ------------------------------------------------------------------------------------------ symbolic part
def mut(fn, **kw):
    w = world("b"); fn(w); return cap(objs=w, **kw)

def ch_parent(w): w[3]["parentReceptacles"] = [w[0]["objectId"]]; w[3]["position"]["x"] = 2.0
check("symbolic: a changed parent/support breaches", "symbolic.parents" in kinds(R.compare(s1, mut(ch_parent), TOL)))
def ch_open(w): w[0]["isOpen"] = True
r = R.compare(s1, mut(ch_open), TOL)
check("symbolic: a changed open flag breaches (and accessibility follows)", {"symbolic.isOpen", "symbolic.accessible"} <= set(kinds(r)), str(kinds(r)))
for flag in ("isToggled", "isSliced", "isDirty", "isFilledWithLiquid", "isBroken", "isCooked"):
    def f(w, flag=flag): w[3][flag] = True
    check(f"symbolic: flag {flag} breaches", f"symbolic.{flag}" in kinds(R.compare(s1, mut(f), TOL)))
def ch_type(w): w[3]["objectType"] = "Gadget"
check("symbolic: a changed type breaches", "symbolic.type" in kinds(R.compare(s1, mut(ch_type), TOL)))
def ch_cap(w): w[3]["pickupable"] = False
check("symbolic: a changed capability breaches", "symbolic.caps" in kinds(R.compare(s1, mut(ch_cap), TOL)))
check("symbolic: a missing object breaches", "objects.set" in kinds(R.compare(s1, cap(objs=world("b")[:3]), TOL)))
extra = world("b") + [obj("Extra_1", "Extra", "Extra|1", (0, 0, 0))]
check("symbolic: an extra object breaches", "objects.set" in kinds(R.compare(s1, cap(objs=extra), TOL)))
def ch_cont(w): w[2]["receptacleObjectIds"] = []
check("symbolic: changed contents breach", "symbolic.contains" in kinds(R.compare(s1, mut(ch_cont), TOL)))
check("symbolic: seen types breach", "seen_types" in kinds(R.compare(s1, cap(suffix="b", seen=("Cup",)), TOL)))
check("symbolic: a task row must be byte-identical (outside the coordinates)", "summary" in kinds(R.compare(s1, cap(suffix="b", summary=(SUMM[0] + " ", SUMM[1])), TOL)))
check("symbolic: a missing task row breaches", "summary" in kinds(R.compare(s1, cap(suffix="b", summary=(SUMM[1],)), TOL)))
check("symbolic: labels breach", "labels" in kinds(R.compare(s1, cap(suffix="b", labels={"goal_at_reset": True, "applicable": True}), TOL)))

# ------------------------------------------------------------------------------------------ agent pose
for key, val in (("position", {"x": 0.75, "y": 0.9, "z": 0.25}), ("rotation", {"x": 0, "y": 180, "z": 0}), ("horizon", 30), ("standing", False)):
    ag = dict(AGENT); ag[key] = val
    check(f"agent: {key} drift breaches", kinds(R.compare(s1, cap(agent=ag, suffix="b"), TOL)) == ["agent_pose"])
ag = dict(AGENT); ag["position"] = {"x": 0.50001, "y": 0.9, "z": 0.25}
check("agent: sub-1e-4 print noise is not drift", R.compare(s1, cap(agent=ag, suffix="b"), TOL)["ok"])
check("agent: honest-pose aliases (cameraHorizon / isStanding)", R.compare(s1, cap(agent={"position": AGENT["position"], "rotation": AGENT["rotation"],
      "cameraHorizon": 0, "isStanding": True}, suffix="b"), TOL)["ok"])

# ------------------------------------------------------------------------------------------ feasibility
fz = s1["feasibility"]
check("feasibility: counts and capability from the symbolic state", fz["00"]["object"] == {"type": "Cup", "n": 1, "n_accessible": 0, "capable": True}, str(fz["00"]))
check("feasibility: destination capability", fz["01"]["to"]["capable"] is True and fz["01"]["to"]["type"] == "Box")
check("feasibility: slice needs sliceable (cup has none)", fz["02"]["object"]["capable"] is False)
check("feasibility: unknown action is tolerated", R.feasibility_of(s1, [{"action": "x"}])["00"] == {"action": "x", "object": None, "to": None})
def open_it(w): w[0]["isOpen"] = True
r = R.compare(s1, mut(open_it), TOL)
check("feasibility: opening the box changes the reachable count and breaches", "feasibility" in kinds(r), str(kinds(r)))
other = cap(objs=world("b")[:1] + world("b")[2:], plan=PLAN)                  # cup gone: type counts differ
check("feasibility: a verdict difference is its own breach", "feasibility" in kinds(R.compare(s1, other, TOL)))
check("feasibility: a plain-string plan step does not crash", isinstance(R.feasibility_of(s1, ["go_to"]), dict))

# ------------------------------------------------------------------------------------------ calibration
CEIL = {"pos_m": 0.10, "rot_deg": 15.0}
def restores(items=12, shift=0.03, k=5):
    out = {}
    for i in range(items):
        out[f"it{i:02d}"] = [cap(objs=world(chr(97 + j), shift=shift * j / (k - 1) if i == 3 else 0.0)) for j in range(k)]
    return out

c = R.calibrate(restores(), CEIL, items=12)
check("calibrate: ok within ceilings, tau rounded up to 0.05 / 5", c["status"] == "calibrated" and c["tau_pos_m"] == 0.05 and c["tau_rot_deg"] == 5.0, str(c)[:300])
check("calibrate: 12 items x C(5,2) = 120 pairs", c["calibration"]["pairs"] == 120 and c["calibration"]["items"] == 12 and c["calibration"]["restores_per_item"] == 5)
check("calibrate: observed max is recorded and attributed", abs(c["calibration"]["observed_max_pos_m"] - 0.03) < 1e-6 and c["calibration"]["worst_item"]["pos"] == "it03")
zero = R.calibrate(restores(shift=0.0), CEIL, items=12)
check("calibrate: zero observed settling gives one step, not zero", zero["tau_pos_m"] == 0.05 and zero["tau_rot_deg"] == 5.0)
edge = R.calibrate(restores(shift=0.05 + 1e-12), CEIL, items=12)
check("calibrate: float noise at a step edge does not round up a step", edge["tau_pos_m"] == 0.05, str(edge["tau_pos_m"]))
big = R.calibrate(restores(shift=0.44), CEIL, items=12)
check("calibrate: 0.44 m settle exceeds the ceiling -> FAILED (not a wider tau)", big["status"] == "failed" and big["tau_pos_m"] == 0.45 and any("ceiling" in x for x in big["reasons"]), str(big["reasons"]))
at_ceiling = R.calibrate(restores(shift=0.10), CEIL, items=12)
check("calibrate: tau equal to the ceiling is allowed", at_ceiling["status"] == "calibrated" and at_ceiling["tau_pos_m"] == 0.10)
rs = restores(); rs["it05"][2] = mut(ch_parent)
rs2 = R.calibrate(rs, CEIL, items=12)
check("calibrate: a symbolic breach in any pair fails", rs2["status"] == "failed" and rs2["calibration"]["symbolic_breaches"], str(rs2["reasons"]))
check("calibrate: wrong restore count refused", raises(lambda: R.calibrate(restores(k=4), CEIL), ValueError))
check("calibrate: wrong item count refused", raises(lambda: R.calibrate(restores(items=11), CEIL, items=12), ValueError))
check("calibrate: bad ceilings refused", raises(lambda: R.calibrate(restores(), {"pos_m": 0, "rot_deg": 1}), ValueError))
rrot = restores(); rrot["it00"][1] = cap(objs=world("b", rot_shift=11.0))
check("calibrate: rotation tau rounds up to 15 and passes at the ceiling", R.calibrate(rrot, CEIL, items=12)["tau_rot_deg"] == 15.0)

# ------------------------------------------------------------------------------------------ validation
pairs = [(cap(suffix="a"), cap(suffix="b", objs=world("b", shift=0.02))) for _ in range(24)]
v = R.validate(pairs, c)
check("validate: 24/24 pass", v["ok"] and v["pass"] == 24 and v["pairs"] == 24, str(v))
pairs[7] = (pairs[7][0], cap(objs=world("b", shift=0.3)))
v = R.validate(pairs, c)
check("validate: one breach fails the gate and is located", (not v["ok"]) and v["pass"] == 23 and v["failures"][0]["pair"] == 7, str(v))
check("validate: wrong pair count is not ok", not R.validate(pairs[:23], c)["ok"])
check("validate: uncalibrated tolerances refused", raises(lambda: R.validate(pairs, {"tau_pos_m": None, "tau_rot_deg": None}), ValueError))

# ------------------------------------------------------------------------------------------ tolerance file
tmp = tempfile.mkdtemp()
PRE = os.path.join(tmp, "prereg.md")
open(PRE, "w", encoding="utf-8").write("# prereg\n### A13 (2026-10-11) x\nnothing 99 m\n### A14a (2026-10-11, before any calibration restore) -- ceilings\n"
                                       "Position ceiling 0.10 m; rotation ceiling 15 deg.\n### A15 (2026-10-11) y\n")
try:
    p = os.path.join(tmp, "tol.json")
    R.write_tolerances(R.template(), p)
    check("file: template loads as 'any'/unset", R.load_tolerances(p, "any")["status"] == "unset")
    check("file: unset never satisfies calibrated", raises(lambda: R.load_tolerances(p, "calibrated"), ValueError))
    check("file: unset never satisfies ceilings", raises(lambda: R.load_tolerances(p, "ceilings"), ValueError))
    PRE_NONE = os.path.join(tmp, "none.md")
    open(PRE_NONE, "w", encoding="utf-8").write("# prereg\n### A14 (2026-10-11) x\n")
    check("A14a: init-ceilings refused when the prereg has no A14a heading", raises(lambda: R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE_NONE,
          "--pos-m", "0.1", "--rot-deg", "15", "--justification", "j", "--dated", "2026-10-11"]), SystemExit))
    check("A14a: refused when the numbers differ from the amendment", raises(lambda: R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE,
          "--pos-m", "0.2", "--rot-deg", "15", "--justification", "j", "--dated", "2026-10-11"]), SystemExit)
          and raises(lambda: R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "0.1", "--rot-deg", "30",
                                      "--justification", "j", "--dated", "2026-10-11"]), SystemExit))
    check("A14a: refused when the date is not the amendment's (the 99 m of another section does not count either)",
          raises(lambda: R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "99", "--rot-deg", "15", "--justification", "j",
                                  "--dated", "2026-10-11"]), SystemExit)
          and raises(lambda: R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "0.1", "--rot-deg", "15",
                                      "--justification", "j", "--dated", "2026-01-01"]), SystemExit))
    check("A14a: the file stays unset after every refusal", R.load_tolerances(p, "any")["status"] == "unset")
    check("cli: init-ceilings", R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "0.1", "--rot-deg", "15", "--justification", "grid step 0.25 m", "--dated", "2026-10-11"]) == 0)
    d = R.load_tolerances(p, "ceilings")
    check("file: ceilings recorded with their date", d["ceilings"]["pos_m"] == 0.1 and d["ceilings"]["dated"] == "2026-10-11" and d["status"] == "ceilings")
    check("file: ceilings carry the prereg sha and the A14a section sha", len(d["ceilings"].get("prereg_sha256", "")) == 64 and len(d["ceilings"].get("section_sha256", "")) == 64)
    check("cli: ceilings cannot be rewritten after being set", raises(lambda: R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "0.4", "--rot-deg", "30",
          "--justification", "x", "--dated", "2026-10-12"]), SystemExit))
    check("file: ceilings status still not calibrated", raises(lambda: R.load_tolerances(p, "calibrated"), ValueError))
    rp = os.path.join(tmp, "restores.json")
    json.dump(restores(), open(rp, "w"));
    check("cli: calibrate fills tau", R._cli(["--tolerances", p, "calibrate", "--restores", rp]) == 0)
    d = R.load_tolerances(p)
    check("file: calibrated tolerances load", d["status"] == "calibrated" and d["tau_pos_m"] == 0.05 and d["calibration"]["pairs"] == 120)
    check("file: compare accepts a loaded file", R.compare(s1, below, d)["ok"])
    check("cli: calibration is done once", raises(lambda: R._cli(["--tolerances", p, "calibrate", "--restores", rp]), SystemExit))
    pp = os.path.join(tmp, "pairs.json")
    json.dump([[a, b] for a, b in [(cap(suffix="a"), cap(suffix="b")) for _ in range(24)]], open(pp, "w"))
    check("file: a calibrated but not yet validated file never loads as 'validated'", raises(lambda: R.load_tolerances(p, "validated"), ValueError))
    check("cli: validate records 24/24", R._cli(["--tolerances", p, "validate", "--pairs", pp]) == 0 and R.load_tolerances(p)["validation"]["ok"])
    check("file: after a passed validation the file loads as 'validated'", R.load_tolerances(p, "validated")["validation"]["pairs"] == 24
          and len(R.load_tolerances(p)["validation_log"]) == 1)
    check("cli: validation is one-shot: a second run is refused", raises(lambda: R._cli(["--tolerances", p, "validate", "--pairs", pp]), SystemExit))
    d2 = R.load_tolerances(p); d2["validation"] = dict(d2["validation"], failures=["x"])
    check("file: a recorded validation cannot be rewritten", raises(lambda: R.write_tolerances(d2, p), PermissionError))
    # tamper: edit tau by hand -> sha mismatch
    t = json.load(open(p)); t["tau_pos_m"] = 0.5; json.dump(t, open(p, "w"))
    check("file: a hand-edited tau is detected", raises(lambda: R.load_tolerances(p), ValueError))
    # final: a calibrated file cannot be widened
    R.write_tolerances(R.template(), p, allow_replace=True)
    R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "0.1", "--rot-deg", "15", "--justification", "j", "--dated", "2026-10-11"])
    json.dump(restores(shift=0.44), open(rp, "w"))
    check("cli: out-of-ceiling calibration is FAILED and exits non-zero", R._cli(["--tolerances", p, "calibrate", "--restores", rp]) == 2)
    check("file: failed never loads as calibrated", raises(lambda: R.load_tolerances(p), ValueError) and R.load_tolerances(p, "any")["status"] == "failed")
    w = R.load_tolerances(p, "any"); w["tau_pos_m"] = 0.5; w["status"] = "calibrated"
    check("file: a failed file cannot be widened into a calibrated one", raises(lambda: R.write_tolerances(w, p), PermissionError))
    # a FAILED validation is final: status failed, kept reasons, no re-run, never 'validated'
    R.write_tolerances(R.template(), p, allow_replace=True)
    R._cli(["--tolerances", p, "init-ceilings", "--prereg", PRE, "--pos-m", "0.1", "--rot-deg", "15", "--justification", "j", "--dated", "2026-10-11"])
    json.dump(restores(), open(rp, "w"))
    R._cli(["--tolerances", p, "calibrate", "--restores", rp])
    badpairs = [[cap(suffix="a"), cap(suffix="b")] for _ in range(24)]
    badpairs[3][1] = cap(objs=world("b", shift=0.3), suffix="b")
    json.dump(badpairs, open(pp, "w"))
    check("cli: a failed validation exits non-zero and makes the file failed", R._cli(["--tolerances", p, "validate", "--pairs", pp]) == 2
          and R.load_tolerances(p, "any")["status"] == "failed" and R.load_tolerances(p, "any")["failure_reasons"])
    json.dump([[a, b] for a, b in [(cap(suffix="a"), cap(suffix="b")) for _ in range(24)]], open(pp, "w"))
    check("cli: a later passing re-validation on new captures cannot overwrite the failure",
          raises(lambda: R._cli(["--tolerances", p, "validate", "--pairs", pp]), Exception) and R.load_tolerances(p, "any")["status"] == "failed")
    fd = R.load_tolerances(p, "any"); fd["status"] = "calibrated"; fd["validation"] = {"pairs": 24, "pass": 24, "ok": True, "failures": []}
    check("file: a failed validation cannot be rewritten into a pass", raises(lambda: R.write_tolerances(fd, p), PermissionError))
    check("file: a failed file never loads as 'validated'", raises(lambda: R.load_tolerances(p, "validated"), ValueError))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------------------------------ rotation = geodesic angle (E-fix)
check("rot: wrapped equal angles", abs(R._rot_delta([0, 358, 0], [0, 3, 0]) - 5.0) < 1e-6)
check("rot: a plain 10 degree yaw", abs(R._rot_delta([0, 10, 0], [0, 20, 0]) - 10.0) < 1e-6)
check("rot: identical orientations in DIFFERENT Euler form (pitch 90 gimbal pair) are 0 apart, not ~180",
      R._rot_delta([100, 20, 30], [80, 200, 210]) < 1e-3 and max(abs((p - q + 180) % 360 - 180) for p, q in zip([100, 20, 30], [80, 200, 210])) > 100)
check("rot: identical triples are exactly 0", R._rot_delta([12.5, 33, 270], [12.5, 33, 270]) == 0.0)
check("rot: a 90 degree roll about z", abs(R._rot_delta([0, 0, 0], [0, 0, 90]) - 90.0) < 1e-6)
sa, sb = cap(objs=world("a")), cap(objs=world("b"))
sa["objects"]["Widget_1"]["rot"] = [100.0, 20.0, 30.0]
sb["objects"]["Widget_1"]["rot"] = [80.0, 200.0, 210.0]
check("rot: compare() does not breach on a physically identical pair written in another Euler form", R.compare(sa, sb, TOL)["ok"])
sm = {"schema": R.STATE_SCHEMA}
def csum(row, *rest):
    return cap(summary=[row] + list(rest))


sx, sy = csum("Cup [clean] on/in Box at (1.20, 2.30), step 3"), csum("Cup [clean] on/in Box at (1.21, 2.30), step 3")
check("summary (A14a): a coordinate-only shift within tau passes", R.compare(sx, sy, TOL)["ok"])
sy2 = csum("Cup [clean] on/in Box at (1.20, 2.40), step 3")
br = R.compare(sx, sy2, TOL)["breaches"]
check("summary (A14a): a coordinate shift above tau is a summary.coords breach", [b["kind"] for b in br] == ["summary.coords"], br)
sd = csum("Cup [clean] on/in Box at (1.28, 2.37), step 3")    # 0.08 / 0.07 per axis, 0.106 Euclidean: above a 0.05 tau
check("summary (A14a): the distance is Euclidean, not per axis", R.compare(sx, sd, {"tau_pos_m": 0.10, "tau_rot_deg": 5.0})["breaches"][0]["kind"] == "summary.coords")
sz = csum("Box [open] on/in Box at (1.20, 2.30), step 3")
check("summary (A14a): a different task row is a structural summary breach", R.compare(sx, sz, TOL)["breaches"][0]["kind"] == "summary")
check("summary (A14a): a different step number is a structural breach",
      R.compare(csum("Cup [clean] on/in Box at (1.20, 2.30), step 3"), csum("Cup [clean] on/in Box at (1.20, 2.30), step 4"), TOL)["breaches"][0]["kind"] == "summary")
check("summary (A14a): a malformed coordinate field fails", R.compare(sx, csum("Cup [clean] on/in Box at (1.20; 2.30), step 3"), TOL)["breaches"][0]["kind"] == "summary")
check("summary (A14a): a different task-row count fails", R.compare(sx, csum("Cup [clean] on/in Box at (1.20, 2.30), step 3", "Cup [dirty] on/in Box at (3.0, 3.0), step 5"), TOL)["breaches"][0]["kind"] == "summary")
check("summary (A14a): order matters", R.compare(csum("Cup [a] on/in Box at (1.0, 1.0), step 1", "Cup [b] on/in Box at (2.0, 2.0), step 2"),
      csum("Cup [b] on/in Box at (2.0, 2.0), step 2", "Cup [a] on/in Box at (1.0, 1.0), step 1"), TOL)["breaches"][0]["kind"] == "summary")
cm = R.compare(sx, sy2, {"tau_pos_m": 0.15, "tau_rot_deg": 5.0})
check("summary (A14a): the summary distance enters max_dpos (the calibration statistic)", cm["ok"] and abs(cm["max_dpos"] - 0.10) < 1e-6, cm["max_dpos"])

# ------------------------------------------------------------------------------------------ A14b: scope + agent precision
check("A14b: relevant types = plan types + container ancestors (Cup, Box; not Widget / Counter)", R.relevant_types(s1) == ["Box", "Cup"], str(R.relevant_types(s1)))
w3 = world("a")
w3[3]["parentReceptacles"] = [w3[2]["objectId"]]
check("A14b: relevant types read goal-clause types from the labels", "Widget" in R.relevant_types(cap(labels={"goal_clauses": {"Widget in Counter": False}})))
nt1 = cap(summary=[SUMM[0], "Widget [clean] on/in Counter at (2.00, 1.00), step 4"])
nt2 = cap(summary=[SUMM[0], "Widget [dirty] on/in Counter at (2.50, 1.00), step 9", "Spoon [?] on/in ? at (0.0, 0.0), step 2"])
r14 = R.compare(nt1, nt2, TOL)
check("A14b: differing NON-task rows do not breach", r14["ok"], r14["breaches"])
check("A14b: ...but are counted as a diagnostic by type", r14["diag"]["other_rows_differing"] == 3 and r14["diag"]["other_types"] == ["Spoon", "Widget"], str(r14["diag"]))
check("A14b: a task row difference still breaches", "summary" in kinds(R.compare(nt1, cap(summary=[SUMM[0].replace("clean", "dirty"), SUMM[1]]), TOL)))
ag = dict(AGENT); ag["rotation"] = {"x": 0, "y": 90.0006, "z": 0}
check("A14b: agent rotation within 0.01 deg passes", R.compare(s1, cap(agent=ag, suffix="b"), TOL)["ok"])
ag = dict(AGENT); ag["rotation"] = {"x": 0, "y": 90.02, "z": 0}
check("A14b: agent rotation of 0.02 deg breaches", kinds(R.compare(s1, cap(agent=ag, suffix="b"), TOL)) == ["agent_pose"])
ag = dict(AGENT); ag["position"] = {"x": 0.5008, "y": 0.9, "z": 0.25}
check("A14b: agent position within 0.001 m passes", R.compare(s1, cap(agent=ag, suffix="b"), TOL)["ok"])
ag = dict(AGENT); ag["position"] = {"x": 0.502, "y": 0.9, "z": 0.25}
check("A14b: agent position 0.002 m off breaches", kinds(R.compare(s1, cap(agent=ag, suffix="b"), TOL)) == ["agent_pose"])
ag = dict(AGENT); ag["rotation"] = {"x": 0, "y": 359.995, "z": 0}
ag0 = dict(AGENT); ag0["rotation"] = {"x": 0, "y": 0.0, "z": 0}
check("A14b: agent rotation wraps around 360", R.compare(cap(agent=ag0), cap(agent=ag, suffix="b"), TOL)["ok"])

# ------------------------------------------------------------------------------------------ A34-A2 override (signed 2026-10-15)
print("== A34-A2: observation-only override of a world-identical restore")
TID = "7a2ee211b85cc534d7677291fce80b04e92cd24a860f1ddd7665185b3a03b67d"
OBS = {"schema": "obs-pipeline/1", "treatment_id": TID, "sweep_prims": 12,
       "health": {"detect_calls": 8, "state_calls": 8, "state_errors": 0, "configs": [["auto", "p.pt", "s.pt", True, 0.25]]}}
LAB = {"goal_clauses": {"Cup in Box": False}, "goal_true_at_reset": False}
PRES = {"plan": PLAN, "goal_clauses": ["Cup in Box"]}
_cap0 = cap


def cap(*a, **k):
    k.setdefault("labels", LAB)
    return _cap0(*a, **k)


A0 = cap()
okk = lambda r: (r["ok"], r["registered_ok"], r["override"])
r = R.compare_a34(A0, cap(suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 1: an ordinary pass stays an ordinary pass (no override, no obs_drift)", okk(r) == (True, True, False) and r["obs_drift"] is None, okk(r))
check("A34 2: the registered compare is unchanged (no override fields, same verdict)", "override" not in R.compare(A0, cap(seen=("Cup",)), TOL)
      and not R.compare(A0, cap(seen=("Cup",)), TOL)["ok"])
B_seen = cap(seen=("Cup", "Box", "Stool"), suffix="b")
r = R.compare_a34(A0, B_seen, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 3: a seen_types-only breach of an identical world is overridden", okk(r) == (True, False, True), (okk(r), r["override_why"]))
check("A34 3b: ...and recorded as obs_drift (types only in B = Stool)", r["obs_drift"]["types_only_b"] == ["Stool"] and r["obs_drift"]["kinds"] == ["seen_types"]
      and r["obs_drift"]["amendment"] == "A34-A2", r["obs_drift"])
B_sum = cap(summary=(SUMM[0].replace("clean", "dirty"), SUMM[1]), suffix="b")
r = R.compare_a34(A0, B_sum, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 4: a task-row summary breach of an identical world is overridden, rows recorded", okk(r) == (True, False, True)
      and r["obs_drift"]["n_rows_differing"] == 2 and r["obs_drift"]["kinds"] == ["summary"], r.get("obs_drift"))
B_xy = cap(summary=(SUMM[0].replace("(1.00, 1.00)", "(1.10, 1.00)"), SUMM[1]), suffix="b")
r = R.compare_a34(A0, B_xy, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 5: a summary.coords breach (0.10 > tau) of an identical world is overridden with its delta", okk(r) == (True, False, True)
      and r["obs_drift"]["max_coord_delta_m"] == 0.1 and r["obs_drift"]["kinds"] == ["summary.coords"], r.get("obs_drift"))
r = R.compare_a34(A0, cap(seen=("Cup", "Box", "Stool"), summary=B_xy["summary"], suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 6: several observation kinds together are still overridden", okk(r) == (True, False, True) and r["obs_drift"]["kinds"] == ["seen_types", "summary.coords"])
# ---- negatives: any world difference stays blocking (even INSIDE tau, because (c) is exact at the stored precision)
w = world("b"); w[3]["position"]["x"] = 2.00001
r = R.compare_a34(A0, cap(objs=w, seen=("Cup", "Box", "Stool"), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 N1: a 1e-5 m object move (within tau) + an observation breach stays BLOCKING", okk(r) == (False, False, False)
      and any(x.startswith("(c)") for x in r["override_why"]), r["override_why"])
w = world("b"); w[3]["rotation"]["y"] = 10.001
r = R.compare_a34(A0, cap(objs=w, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 N2: a rotation difference (within tau) + an observation breach stays BLOCKING", not r["ok"] and any(x.startswith("(c)") for x in r["override_why"]))
ag = dict(AGENT); ag["position"] = {"x": 0.5001, "y": 0.9, "z": 0.25}
r = R.compare_a34(A0, cap(agent=ag, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 N3: an agent position difference (within the A14b 0.001 m) + an observation breach stays BLOCKING", not r["ok"]
      and any("agent" in x for x in r["override_why"]), r["override_why"])
ag = dict(AGENT); ag["horizon"] = 30
r = R.compare_a34(A0, cap(agent=ag, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 N4: an agent horizon difference stays BLOCKING (non-observation kind agent_pose)", not r["ok"] and any(x.startswith("(a)") for x in r["override_why"]))
ag = dict(AGENT); ag["standing"] = False
check("A34 N5: an agent standing difference stays BLOCKING", not R.compare_a34(A0, cap(agent=ag, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)["ok"])
w = world("b"); w[0]["isOpen"] = True
check("A34 N6: a state flag difference stays BLOCKING", not R.compare_a34(A0, cap(objs=w, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)["ok"])
r = R.compare_a34(A0, cap(labels={"goal_clauses": {"Cup in Box": True}, "goal_true_at_reset": True}, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 N7: an evaluator-label difference stays BLOCKING", not r["ok"] and r["override"] is False)
r = R.compare_a34(A0, cap(plan=PLAN[:2], seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 N8: a feasibility difference stays BLOCKING", not r["ok"] and r["override"] is False)
w = world("b")[:3]
check("A34 N9: a missing object stays BLOCKING", not R.compare_a34(A0, cap(objs=w, seen=("Cup",), suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)["ok"])
# ---- (b): settings and validity
import copy as _cp
_o_tid = dict(OBS, treatment_id="0" * 64)
_o_cfg = _cp.deepcopy(OBS); _o_cfg["health"]["configs"] = [["always", "p.pt", "s.pt", True, 0.25]]
_o_sw = dict(OBS, sweep_prims=13)
_o_err = _cp.deepcopy(OBS); _o_err["health"]["state_errors"] = 1
_o_nodet = _cp.deepcopy(OBS); _o_nodet["health"]["detect_calls"] = 0
_o_nohealth = dict(OBS, health=None)
_o_nocfg = _cp.deepcopy(OBS); _o_nocfg["health"]["configs"] = []
for oa, ob, nm in ((OBS, _o_tid, "a different treatment identity"), (OBS, _o_cfg, "a different detector configuration"),
                   (OBS, _o_sw, "a different sweep length"), (OBS, _o_err, "a swallowed state-classifier failure"),
                   (_o_nodet, OBS, "a detector that never ran"), (OBS, _o_nohealth, "no health evidence"), (OBS, _o_nocfg, "no configuration"),
                   (None, OBS, "no pipeline record A"), (OBS, None, "no pipeline record B"), (None, None, "no pipeline records"),
                   (OBS, dict(OBS, schema="x"), "a foreign record schema")):
    r = R.compare_a34(A0, B_seen, TOL, obs_a=oa, obs_b=ob, presentation=PRES)
    check(f"A34 N10: observation pipelines with {nm} -> BLOCKING", not r["ok"] and any(x.startswith("(b)") for x in r["override_why"]), r["override_why"])
for nm, mut in (("a (nan, 1.00) coordinate pair", lambda s: s["summary"].__setitem__(0, s["summary"][0].replace("(1.00, 1.00)", "(nan, 1.00)"))),
                ("an 'inf' token", lambda s: s["summary"].__setitem__(0, s["summary"][0] + " inf")),
                ("a malformed pair (1.0.0, 2)", lambda s: s["summary"].__setitem__(0, s["summary"][0].replace("(1.00, 1.00)", "(1.0.0, 2)")))):
    for side in ("a", "b"):
        x, y = _cp.deepcopy(A0), _cp.deepcopy(B_seen)
        mut(x if side == "a" else y)
        for s_ in (x, y):
            s_["summary_sha"] = R.sha256_text("\n".join(s_["summary"]))
            s_["symbolic_sha"] = R.sha256_text(R.canon(R.symbolic_part(s_)))
        check(f"A34 N10b: {nm} in capture {side} -> invalid, BLOCKING", R.state_problems(x if side == "a" else y) != []
              and not R.a34_override(x, y, [{"kind": "seen_types"}], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
for nm, lab in (("empty labels", {}), ("no goal clauses", {"goal_true_at_reset": False}), ("an empty clause map", {"goal_clauses": {}, "goal_true_at_reset": False}),
                ("a non-bool clause", {"goal_clauses": {"Cup in Box": 0.5}, "goal_true_at_reset": False}),
                ("a NaN inside labels", {"goal_clauses": {"Cup in Box": False}, "goal_true_at_reset": False, "x": float("nan")})):
    x, y = cap(labels=lab), cap(labels=lab, seen=("Cup", "Box", "Stool"), suffix="b")
    check(f"A34 N10c: identical but incomplete labels ({nm}) on both sides -> BLOCKING", not R.compare_a34(x, y, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)["ok"])
x, y = _cp.deepcopy(A0), _cp.deepcopy(B_seen)
for s_ in (x, y):
    s_["feasibility"] = {}
    s_["symbolic_sha"] = R.sha256_text(R.canon(R.symbolic_part(s_)))
check("A34 N10d: identical but EMPTY feasibility on both sides -> BLOCKING", not R.a34_override(x, y, [{"kind": "seen_types"}], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
_bs = [{"kind": "seen_types", "only_a": [], "only_b": ["Stool"]}]
check("A34 N10e: captures not bound to the presentation (another plan / goal count / none) -> BLOCKING",
      not R.a34_override(A0, B_seen, _bs, obs_a=OBS, obs_b=OBS, presentation=dict(PRES, plan=PLAN + PLAN[:1]))["override"]
      and not R.a34_override(A0, B_seen, _bs, obs_a=OBS, obs_b=OBS, presentation=dict(PRES, plan=[dict(PLAN[0], object="Box")] + PLAN[1:]))["override"]
      and not R.a34_override(A0, B_seen, _bs, obs_a=OBS, obs_b=OBS, presentation={"plan": PLAN, "goal_clauses": ["Cup in Box", "Cup.isDirty = true"]})["override"]
      and not R.a34_override(A0, B_seen, _bs, obs_a=OBS, obs_b=OBS)["override"]
      and R.a34_override(A0, B_seen, _bs, obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
# ---- Astra ACC2-IMPL-r2 1-3: full summary grammar, exact labels, complete feasibility, malformed pipeline evidence, nested drift
for nm, line in (("an unclosed coordinate", "Cup [clean] on/in Box at (1.00, 1.00, step 3"), ("arbitrary text", "anything at all"),
                 ("three decimals", "Cup [clean] on/in Box at (1.000, 1.00), step 3"), ("no step", "Cup [clean] on/in Box at (1.00, 1.00)")):
    x = _cp.deepcopy(A0); x["summary"][0] = line
    x["summary_sha"] = R.sha256_text("\n".join(x["summary"])); x["symbolic_sha"] = R.sha256_text(R.canon(R.symbolic_part(x)))
    check(f"A34 G1: a summary line with {nm} is malformed", R.state_problems(x) != [])
check("A34 G2: a place '?' is a well-formed summary line", R._summary_line_ok("Cabinet [closed] on/in ? at (-1.46, -2.43), step 8"))
x, y = cap(labels={"goal_clauses": {"Other thing": False}, "goal_true_at_reset": False}), cap(labels={"goal_clauses": {"Other thing": False}, "goal_true_at_reset": False}, seen=("Cup", "Box", "Stool"), suffix="b")
check("A34 G3: identical labels with an unrelated clause key still count 1 = n_goal (bound by count), but an extra key is refused",
      R.state_problems(cap(labels={"goal_clauses": {"a": False, "b": False}, "goal_true_at_reset": False}), presentation=PRES) != [])
check("A34 G4: labels with an extra field are refused", R.state_problems(cap(labels=dict(LAB, extra=1))) != [])
check("A34 G5: an inconsistent reset flag is refused", R.state_problems(cap(labels={"goal_clauses": {"Cup in Box": True}, "goal_true_at_reset": False}), presentation=PRES) != [])
x = _cp.deepcopy(A0)
x["feasibility"] = {k: {"action": v["action"]} for k, v in x["feasibility"].items()}
x["symbolic_sha"] = R.sha256_text(R.canon(R.symbolic_part(x)))
check("A34 G6: feasibility entries with only an action are refused", R.state_problems(x) != [])
for nm, cfg in (("[[null]]", [[None]]), ("a 4-field entry", [["auto", "p.pt", None, True]]), ("an unknown mode", [["fast", "p.pt", None, True, 0.25]]),
                ("conf 0", [["auto", "p.pt", None, True, 0.0]]), ("an empty primary", [["auto", "", None, True, 0.25]])):
    o_ = _cp.deepcopy(OBS); o_["health"]["configs"] = cfg
    check(f"A34 G7: pipeline configuration {nm} is refused", R.pipeline_problems(o_) != [])
o_ = _cp.deepcopy(OBS); o_["health"]["state_calls"] = 0
check("A34 G8: a state-enabled configuration with 0 state calls is refused", R.pipeline_problems(o_) != [])
o_ = _cp.deepcopy(OBS); o_["health"]["configs"] = [["auto", "p.pt", None, False, 0.25]]
check("A34 G9: state calls without a state-enabled configuration are refused", R.pipeline_problems(o_) != [])
check("A34 G10: the healthy record passes", R.pipeline_problems(OBS) == [])
r = R.compare_a34(A0, cap(seen=("Cup", "Box", "Stool"), summary=B_xy["summary"], suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
d_ = r["obs_drift"]
for nm, mut in (("a null type entry", dict(d_, types_only_b=[None])), ("an object type entry", dict(d_, types_only_b=[{"x": 1}])),
                ("an invalid summary sha", dict(d_, summary_sha_a="xyz")), ("a negative delta", dict(d_, coord_deltas=[{"row": "Cup", "delta_m": -0.1}], max_coord_delta_m=-0.1)),
                ("an inconsistent maximum", dict(d_, max_coord_delta_m=9.0)), ("types not matching the breach", dict(d_, types_only_b=["Chair"])),
                ("a malformed row", dict(d_, summary_rows_only_a=["junk"], n_rows_differing=d_["n_rows_differing"] + 1))):
    check(f"A34 G11: an obs_drift with {nm} is refused", R.obs_drift_problems(mut, r["breaches"]) != [], nm)
check("A34 G12: the complete drift still passes", R.obs_drift_problems(d_, r["breaches"]) == [], R.obs_drift_problems(d_, r["breaches"]))
# ---- Astra ACC2-IMPL-r3 1, 2, 4
x_, y_ = cap(labels={"goal_clauses": {"Other thing": False}, "goal_true_at_reset": False}), cap(labels={"goal_clauses": {"Other thing": False}, "goal_true_at_reset": False}, seen=("Cup", "Box", "Stool"), suffix="b")
check("A34 H1: identical labels keyed by an UNRELATED clause name are refused (bound to the presentation's own clauses)",
      not R.a34_override(x_, y_, _bs, obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
x_ = _cp.deepcopy(A0)
for v_ in x_["feasibility"].values():
    if v_.get("object"):
        v_["object"].pop("capable")
x_["symbolic_sha"] = R.sha256_text(R.canon(R.symbolic_part(x_)))
check("A34 H2: a feasibility verdict without its capability field is refused", R.state_problems(x_) != [])
sw = list(SUMM) + ["Widget [clean] on/in Counter at (2.00, 2.00), step 4"]
x_, y_ = cap(summary=sw), cap(summary=[sw[0], sw[2], sw[1]], seen=("Cup", "Box", "Stool"), suffix="b")
r_ = R.compare_a34(x_, y_, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 H3: a pure row-ORDER difference (same multiset) of an identical world is overridden and its record is complete",
      r_["override"] and r_["obs_drift"]["positional_diffs"] and R.obs_drift_problems(r_["obs_drift"], r_["breaches"]) == [],
      (r_["override_why"], R.obs_drift_problems(r_["obs_drift"], r_["breaches"]) if r_["obs_drift"] else None))
check("A34 H4: a drift record without its positional differences is refused", R.obs_drift_problems(dict(d_, positional_diffs=[])) != [])
x_ = _cp.deepcopy(A0)
x_["feasibility"] = {"zz": list(x_["feasibility"].values())[0]}
x_["symbolic_sha"] = R.sha256_text(R.canon(R.symbolic_part(x_)))
check("A34 H5: malformed feasibility keys are refused WITHOUT an exception", R.state_problems(x_, presentation=PRES) != [])
check("A34 H6: validate_a34 turns a malformed pair into a blocked pair (no exception escapes)",
      R.validate_a34([(A0, {"schema": R.STATE_SCHEMA})], TOL, obs=[(OBS, OBS)], presentations=[PRES], expected=1)["ok"] is False)
bad = copy.deepcopy(B_seen); bad["summary"] = bad["summary"] + ["Stool [x] on/in Floor at (0.00, 0.00), step 9"]
check("A34 N11: a summary edited after capture (summary_sha mismatch) is invalid -> BLOCKING",
      not R.compare_a34(A0, bad, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
bad = copy.deepcopy(B_seen); bad["objects"]["Cup_1"]["pos"] = [float("nan"), 0.5, 1.0]
check("A34 N12: a non-finite pose is invalid -> BLOCKING", R.state_problems(bad) != [] and not R.a34_override(A0, bad, [{"kind": "seen_types"}],
      obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
bad = copy.deepcopy(B_seen); del bad["objects"]["Cup_1"]["flags"]
check("A34 N13: a missing field is invalid -> BLOCKING", not R.a34_override(A0, bad, [{"kind": "seen_types"}], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
bad = copy.deepcopy(B_seen); bad.pop("labels")
check("A34 N14: missing labels -> BLOCKING", not R.a34_override(A0, bad, [{"kind": "seen_types"}], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
bad = copy.deepcopy(B_seen); bad["symbolic_sha"] = "0" * 64
check("A34 N15: a symbolic sha that does not match the state -> BLOCKING", not R.a34_override(A0, bad, [{"kind": "seen_types"}], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
check("A34 N16: a state that is not a reset-state -> BLOCKING", not R.a34_override(A0, None, [{"kind": "seen_types"}], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"])
# ---- (a): extra (holding) breaches and non-observation kinds
r = R.compare_a34(A0, B_seen, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES, extra_breaches=[{"kind": "told_holding", "capture": None, "now": "Cup"}])
check("A34 N17: an observation breach + a holding breach stays BLOCKING", okk(r) == (False, False, False) and any("told_holding" in x for x in r["override_why"]))
check("A34 N18: an empty breach list is no override", R.a34_override(A0, B_seen, [], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"] is False)
check("A34 N19: a malformed breach entry is no override", R.a34_override(A0, B_seen, ["seen_types"], obs_a=OBS, obs_b=OBS, presentation=PRES)["override"] is False)
check("A34 N20: OBS_KINDS is exactly the three observation kinds", R.OBS_KINDS == ("seen_types", "summary", "summary.coords"))
# ---- rule 4 dev validation (read-only)
pairs = [(A0, cap(suffix="b"))] * 22 + [(A0, B_seen), (A0, B_xy)]
v = R.validate_a34(pairs, TOL, obs=[(OBS, OBS)] * 24, presentations=[PRES] * 24)
check("A34 V1: 22 passes + 2 overrides = PASS 24/24 with obs_drift counts", v["ok"] and v["registered_pass"] == 22 and v["overridden"] == 2
      and v["obs_drift_by_kind"] == {"seen_types": 1, "summary.coords": 1} and v["obs_drift_types"] == {"Stool": 1}, {k: v[k] for k in ("ok", "registered_pass", "overridden")})
w = world("b"); w[3]["position"]["x"] = 2.00001
v = R.validate_a34(pairs[:23] + [(A0, cap(objs=w, seen=("Cup",), suffix="b"))], TOL, obs=[(OBS, OBS)] * 24, presentations=[PRES] * 24)
check("A34 V2: one world-different observation breach fails the validation (23/24)", not v["ok"] and v["blocked"] == 1)
v = R.validate_a34(pairs[:23], TOL, obs=[(OBS, OBS)] * 23)
check("A34 V3: 23 pairs is not 24 -> not ok", not v["ok"])
check("A34 V4: the observation-pipeline list must match the pairs", raises(lambda: R.validate_a34(pairs, TOL, obs=[(OBS, OBS)])))
_tmpd = tempfile.mkdtemp()
_tp = os.path.join(_tmpd, "tol.json")
shutil.copy(os.path.join(R.ROOT, "docs", "final-test", "reset_tolerances.json"), _tp)
_before = open(_tp, "rb").read()
_pf = os.path.join(_tmpd, "pairs.json")
json.dump({"pairs": [list(p) for p in pairs], "obs": [[OBS, OBS]] * 24, "presentation_specs": [PRES] * 24}, open(_pf, "w", encoding="utf-8"))
_lg = os.path.join(_tmpd, "log.jsonl")
_real_log, R.A34_VALIDATION_LOG = R.A34_VALIDATION_LOG, _lg              # the CLI always logs to the canonical path (redirected in this test)
_rc = R._cli(["--tolerances", _tp, "validate-a34", "--pairs", _pf, "--out", os.path.join(_tmpd, "out.json")])
R.A34_VALIDATION_LOG = _real_log
check("A34 V5: the validate-a34 CLI passes on the shipped (validated) tolerances and NEVER writes the tolerance file",
      _rc == 0 and open(_tp, "rb").read() == _before and json.load(open(os.path.join(_tmpd, "out.json"), encoding="utf-8"))["ok"])
_lr = [json.loads(x) for x in open(_lg, encoding="utf-8")]
_bad_pf = os.path.join(_tmpd, "bad.json")
open(_bad_pf, "w", encoding="utf-8").write('{"pairs": [[1, 2]]}')
_real_log, R.A34_VALIDATION_LOG = R.A34_VALIDATION_LOG, _lg
_raised = raises(lambda: R._cli(["--tolerances", _tp, "validate-a34", "--pairs", _bad_pf, "--out", os.path.join(_tmpd, "bad_out.json")]))
R.A34_VALIDATION_LOG = _real_log
_lr2 = [json.loads(x) for x in open(_lg, encoding="utf-8")]
check("A34 V7: a validation that raises is first LOGGED as failed (terminal), then re-raised", _raised and len(_lr2) == 2 and _lr2[1]["ok"] is False
      and "error" in _lr2[1])
import hashlib as _hl
check("A34 V6: the result is appended to the validation log (ok, counts, result sha)", len(_lr) == 1 and _lr[0]["ok"] is True and _lr[0]["pairs"] == 24
      and _lr[0]["result_sha256"] == _hl.sha256(open(os.path.join(_tmpd, "out.json"), "rb").read()).hexdigest())
shutil.rmtree(_tmpd, ignore_errors=True)
# ---- rule 3: the obs_drift record is complete and checked
r = R.compare_a34(A0, cap(seen=("Cup", "Box", "Stool"), summary=B_xy["summary"], suffix="b"), TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
d_ = r["obs_drift"]
check("A34 R1: a complete obs_drift passes its own check", R.obs_drift_problems(d_, r["breaches"]) == [], R.obs_drift_problems(d_, r["breaches"]))
check("A34 R2: obs_drift keeps every breach, both summary shas and the pipeline settings", d_["n_breaches"] == len(r["breaches"]) and d_["summary_sha_a"]
      and d_["settings"] == R.pipeline_settings(OBS))
for k_ in ("types_only_b", "coord_deltas", "breaches", "settings", "kinds"):
    y_ = dict(d_); y_.pop(k_)
    check(f"A34 R3: an obs_drift without {k_} is refused", R.obs_drift_problems(y_) != [])
check("A34 R4: a bare {'amendment': 'A34-A2'} is refused", R.obs_drift_problems({"amendment": "A34-A2"}) != [])
check("A34 R5: a seen_types drift without the differing types is refused", R.obs_drift_problems(dict(d_, types_only_a=[], types_only_b=[])) != [])
check("A34 R6: a summary.coords drift without deltas is refused", R.obs_drift_problems(dict(d_, coord_deltas=[])) != [])
check("A34 R7: kinds that do not match the breaches are refused", R.obs_drift_problems(dict(d_, kinds=["seen_types"])) != [])
check("A34 R8: obs_drift breaches that differ from the state check's are refused", R.obs_drift_problems(d_, [{"kind": "summary"}]) != [])
big = cap(summary=list(SUMM) + [f"Widget [clean] on/in Counter at ({i}.00, 1.00), step {i}" for i in range(40)], seen=("Cup", "Box", "Stool"), suffix="b")
r = R.compare_a34(A0, big, TOL, obs_a=OBS, obs_b=OBS, presentation=PRES)
check("A34 R9: nothing is truncated (40 extra non-task rows all recorded)", r["obs_drift"] is not None and len(r["obs_drift"]["summary_rows_only_b"]) == 40)

# ------------------------------------------------------------------------------------------ the shipped file + common parity
import common as C                                    # noqa: E402
x = {"a": 1, "b": [1, 2], "sha256": "z"}
check("parity: canon and self_sha equal common's", R.canon(x) == C.canon(x) and R.self_sha(x) == C.self_sha(x))
shipped = os.path.join(R.ROOT, "docs", "final-test", "reset_tolerances.json")
if os.path.exists(shipped):
    sd = R.load_tolerances(shipped, "any")
    check("shipped file: valid schema + self sha, status in the protocol", sd["schema"] == R.TOL_SCHEMA and sd["status"] in R.STATUSES)
    check("shipped file: no tolerance before ceilings", sd["status"] != "unset" or (sd["tau_pos_m"] is None and sd["ceilings"] is None))
else:
    check("shipped file exists", False, shipped)

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
