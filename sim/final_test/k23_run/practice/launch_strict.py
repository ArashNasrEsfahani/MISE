"""K23 PRACTICE strict launcher (Windows, repo root) for the overnight runner (k23_overnight.py; Astra K23-OVERNIGHT-r1 item 1, r3 A3).

    python sim/final_test/k23_run/practice/launch_strict.py practice-k2-v1 <lock_wait_s> <token>
        <token> = the repo-relative one-time launch authorization file the runner created under sim/final_test/k23_run/overnight/auth/

The preflight of practice/launch.py, through ITS functions (the module is loaded, never edited): a K23 practice namespace; k23_practice
readiness = [] (both A29 samples frozen, package final, treatment pins hold; the perception freeze is verified inside WSL by k23_practice);
Ollama at 127.0.0.1:11434 serving the frozen Gemma; the token exists. Then practice/run_strict.sh in Ubuntu-22.04: ONE try of k23_practice.py
under the sim lock (waited for at most lock_wait s, never stolen), started only after it CLAIMS the token (atomic rename) while holding the
lock -- a token the runner revoked means nothing runs (exit 74) -- with NO restart loop. The study key reaches WSL through WSLENV only (never
printed), as launch.py does. Exit: the try's rc; 73 = lock not acquired; 74 = authorization gone; 2 = refused (nothing started).
Infrastructure only: no prompt, item, threshold or rule lives here.
"""
import importlib.util
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("k23_practice_launch", os.path.join(HERE, "launch.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)                                           # practice/launch.py: GEMMA, REPO, ollama_ok, C, E, KP
TOKEN_RE = re.compile(r"^sim/final_test/k23_run/overnight/auth/[A-Za-z0-9._-]+\.token$")


def main(argv):
    nss = L.E.k23_namespaces("practice")
    if len(argv) != 3 or argv[0] not in nss or not argv[1].isdigit() or not TOKEN_RE.match(argv[2]):
        print(f"usage: launch_strict.py <{'|'.join(nss)}> <lock_wait_s> <sim/final_test/k23_run/overnight/auth/<name>.token>")
        return 2
    ns, lw, tok = argv
    if not os.path.isfile(os.path.join(L.REPO, *tok.split("/"))):
        print(f"REFUSED: the launch authorization {tok} is gone (revoked or used): nothing started")
        return 74
    why = L.KP.readiness(ns, [0], [10], perception=False)
    if why:
        print("REFUSED (nothing started): " + "; ".join(why[:8]))
        return 2
    ok, info = L.ollama_ok()
    if not ok:
        print(f"REFUSED: Ollama at 127.0.0.1:11434 does not serve {L.GEMMA} ({info}); start it with OLLAMA_HOST=0.0.0.0")
        return 2
    env = L.C.wsl_key_env(os.environ)
    rc = subprocess.call(["wsl", "-d", "Ubuntu-22.04", "--cd", L.E.wsl_path(L.REPO), "--", "bash",
                          "sim/final_test/k23_run/practice/run_strict.sh", ns, lw, tok], env=env)
    print("launcher exit", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
