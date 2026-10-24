"""Build-time verification of the procedural KB (sim/step5/procedures_kb.json) against the planner's ft-v2 validator.

    python sim/planners/procedures_check.py            check_static of the KB against the generated executor contract
    python sim/planners/procedures_check.py --early    + the B4 early-check report (diagnostic, non-test items only)

Moved out of sim/ltm_procedural.py (K-B2b, Astra K-B2b-honesty 1): planners.validator_v2.validate imports honest_skills / step4_gemma
lazily, so the step-5 library takes the validator as an argument and only this planner-side caller selects it. Offline, no simulator.
"""
import json
import os
import sys

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIM not in sys.path:
    sys.path.insert(0, SIM)


def validator():
    """(name, fn): the ft-v2 validator every planner plan passes."""
    from planners import validator_v2 as v2
    return "validator_v2", v2.validate


def check(kb=None, contract=None, types=None) -> list:
    import ltm_procedural as P
    kb = kb or P.load_kb(verified=False)
    contract = contract or P.load_exec_contract()
    return kb.check_static(contract, P.contract_tools(contract), types or P.detector_types(), validator())


def main(argv=None) -> int:
    import ltm_procedural as P
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--early" in argv:
        print(json.dumps(P.early_check(validator=validator()), ensure_ascii=False, indent=1))
        return 0
    pr = check()
    print("check_static:", pr or "OK", "| validator: validator_v2")
    return 1 if pr else 0


if __name__ == "__main__":
    sys.exit(main())
