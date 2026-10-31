"""Regenerate final_results.json, l3_breakdown.json and joint_bootstrap.json of the re-executed analysis from the records in this repository.

    python docs/final-test/reexecuted/run.py [--out DIR]

Python with numpy and scipy; no simulator, no language model, no network. By default the three files are rewritten next to this script,
so `git status` shows at once whether the regenerated values differ from the shipped ones. With --out they are written to DIR instead.
"""
import sys

sys.dont_write_bytecode = True

import argparse   # noqa: E402
import os         # noqa: E402
import time       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=HERE, help="directory for the three output files (default: next to this script)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    import compute_final as CF
    import l3_breakdown as L3
    import joint_bootstrap as JB
    CF.OUT = os.path.join(os.path.abspath(a.out), "final_results.json")
    L3.OUT = os.path.join(os.path.abspath(a.out), "l3_breakdown.json")
    JB.OUT = os.path.join(os.path.abspath(a.out), "joint_bootstrap.json")
    CF.main()
    L3.main()
    JB.main()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
