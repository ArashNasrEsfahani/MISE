"""Self-test for step 5 -- the parameter registry gate promised by docs/step5-plan.md 5.3 (added by the A7 review).

Offline, Windows `python` from the REPO ROOT, standard library only:

    python sim/_step5_params_selftest.py

Checks: (1) every registry leaf carries value, range, unit, a real reason and a boolean fp1_tuned; (2) the declared
half-lives and switches match the frozen spec (5.3, 6); (3) every literal registry access in step-5 code
(`params["sect"]["key"]`, `_v(params, "sect", "key")`, `p["sect"]["key"]`) names a key that exists, so code cannot
read a number the registry does not declare; (4) the FP1 dependency list and the FP1-tuned flags are disclosed.
"""
import json
import re
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
REG_PATH = SIM / "step5" / "params.json"
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def main():
    reg = json.loads(REG_PATH.read_text(encoding="utf-8"))
    sections = {k: v for k, v in reg.items() if isinstance(v, dict) and k not in ("fp1_dependencies", "detector_capability")}
    leaves = {(s, k): v for s, sec in sections.items() for k, v in sec.items()
              if not k.startswith("_") and not (s == "retrieval" and k in ("retention", "a12"))}      # A12 (W2) amendment keys: checked below
    check("registry has leaves", len(leaves) >= 40, len(leaves))
    missing = [f"{s}.{k}" for (s, k), v in leaves.items()
               if not isinstance(v, dict) or "value" not in v or not ({"range", "grid"} & set(v)) or "unit" not in v]
    check("every leaf has value + range/grid + unit", not missing, missing)
    weak = [f"{s}.{k}" for (s, k), v in leaves.items() if not isinstance(v.get("reason"), str) or len(v["reason"].strip()) < 12]
    check("every leaf has a real reason (>= 12 chars)", not weak, weak)
    nob = [f"{s}.{k}" for (s, k), v in leaves.items() if not isinstance(v.get("fp1_tuned"), bool)]
    check("every leaf declares fp1_tuned as a bool", not nob, nob)
    rr = reg["retrieval"]
    check("A12 switch is consistent: retention 'a12' <=> an a12 block with A, B, N for every level",
          (rr.get("retention") in (None, "c2e")) if "a12" not in rr else
          (rr.get("retention") == "a12" and all(set(rr["a12"][k]) == {"L1", "L2", "L3"} for k in ("A", "B", "N"))))
    # (2) frozen spec values
    hl = {k: reg["decay"][k]["value"] for k in ("H_location", "H_placement", "H_negative", "H_habit", "H_lesson", "H_procedure")}
    check("half-lives match spec 5.3 (3/2/1/30/20/50)",
          hl == {"H_location": 3, "H_placement": 2, "H_negative": 1, "H_habit": 30, "H_lesson": 20, "H_procedure": 50}, hl)
    check("locations decay faster than habits", hl["H_location"] < hl["H_habit"])
    check("general habits INACTIVE (one kitchen)", reg["habits"]["general_active"]["value"] is False)
    check("general-habit promotion needs >= 2 kitchens", reg["habits"]["min_kitchens"]["value"] >= 2)
    check("lexical-only retrieval until G-dense", reg["retrieval"]["channels"]["value"] == ["lexical"], reg["retrieval"]["channels"]["value"])
    # (4) disclosure
    dep = reg.get("fp1_dependencies", {})
    check("fp1_dependencies listed with a reason", isinstance(dep.get("value"), list) and len(dep["value"]) >= 3 and dep.get("reason"))
    flagged = sorted(f"{s}.{k}" for (s, k), v in leaves.items() if v.get("fp1_tuned"))
    for name in ("consolidate.min_conf", "inspect.vis_ok", "inspect.min_support"):
        check(f"{name} is flagged fp1_tuned (it comes from step 3/4 on FloorPlan1)", name in flagged, flagged)
        check(f"{name} is named in fp1_dependencies", name in dep.get("value", []))
    cap = reg.get("detector_capability", {})
    check("detector capability table is versioned per detector revision (dict or empty)", isinstance(cap.get("values"), (dict, list)))
    # (3) code reads only declared keys
    pats = [re.compile(r"""params\[\s*["'](\w+)["']\s*\]\s*\[\s*["'](\w+)["']\s*\]"""),
            re.compile(r"""_v\(\s*\w+\s*,\s*["'](\w+)["']\s*,\s*["'](\w+)["']"""),
            re.compile(r"""\bp\[\s*["'](\w+)["']\s*\]\s*\[\s*["'](\w+)["']\s*\]""")]
    files = sorted(SIM.glob("ltm*.py")) + [SIM / "step5_eval.py", SIM / "step5_replay.py"]
    seen, unknown = set(), []
    for f in files:
        src = f.read_text(encoding="utf-8")
        for pat in pats:
            for s, k in pat.findall(src):
                if s not in reg or not isinstance(reg[s], dict):
                    continue                    # not a registry access (e.g. a different dict called p)
                seen.add((s, k))
                if k not in reg[s]:
                    unknown.append(f"{f.name}: {s}.{k}")
    check("literal registry accesses found in code", len(seen) >= 10, len(seen))
    check("every literal registry access names a declared key", not unknown, unknown)
    check("positive control: the access scanner sees a bad key",
          pats[0].findall('x = params["decay"]["H_bogus"]["value"]') == [("decay", "H_bogus")])
    print(f"\n{_pass}/{_pass + _fail}")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
