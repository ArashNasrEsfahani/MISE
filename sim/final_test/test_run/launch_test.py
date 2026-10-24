"""KITCHEN 1 LOCKED TEST (test-v1) Windows-side launcher (repo root), modelled on acceptance_v2/run/launch.py: runs
test_run/run_test.sh <job> <timeout_s> <matrix_run args> in Ubuntu-22.04 under flock /home/arash/sim.lock + hang_guard.
For the capture phase only, the study key is forwarded through WSLENV (common.wsl_key_env; never printed): test-v1 capture counts
tokens with the free Anthropic count_tokens endpoint (A12 needs every tokenizer). Execute gets no key. Infrastructure only.
    python sim/final_test/test_run/launch_test.py t-capture 21600 --phase capture --items test --namespace test-v1 --allow-test -- --runs 1 --worlds canon,reloc"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "matrix"))
import common as C  # noqa: E402

args = sys.argv[1:]
env = C.wsl_key_env(os.environ) if "capture" in args[2:4] else dict(os.environ)
rc = subprocess.call(["wsl", "-d", "Ubuntu-22.04", "--cd", "/mnt/d/Personal/Master's Project", "--", "bash",
                      "sim/final_test/test_run/run_test.sh"] + args, env=env)
print("launcher exit", rc)
sys.exit(rc)
