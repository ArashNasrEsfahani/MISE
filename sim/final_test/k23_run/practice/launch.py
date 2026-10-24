"""K23 PRACTICE Windows-side launcher (repo root, from PowerShell), modelled on sim/final_test/diag/practice_v2/launch.py (Kitchen 1).

    python sim/final_test/k23_run/practice/launch.py practice-k2-v1 [extra k23_practice args, e.g. --limit 2]
    python sim/final_test/k23_run/practice/launch.py practice-k3-v1

Preflight on Windows (nothing is started if any fails): the namespace is a K23 practice namespace; k23_practice readiness = [] (both A29
samples frozen, package final, treatment pins hold; the perception freeze is verified inside WSL); Ollama answers at 127.0.0.1:11434 with the frozen
Gemma (gemma4:e2b-it-q4_K_M); no K2/K3 practice journal of ANOTHER treatment identity. Then the study key is forwarded to WSL through
WSLENV (never printed; the free count_tokens counter needs it) and run.sh runs k23_practice.py under the ONE sim lock + hang guard.
Infrastructure only: no prompt, item, threshold or rule lives here.
"""
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "matrix"))
import common as C  # noqa: E402
import k23_entry as E  # noqa: E402
import k23_practice as KP  # noqa: E402

GEMMA = "gemma4:e2b-it-q4_K_M"
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))       # this checkout: the WSL side runs here, never another tree


def ollama_ok(url="http://127.0.0.1:11434"):
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=10) as r:
            names = [m.get("name") for m in json.load(r).get("models") or []]
        return GEMMA in names, names
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main(argv):
    if not argv or argv[0] not in E.k23_namespaces("practice"):
        print(f"usage: launch.py <{'|'.join(E.k23_namespaces('practice'))}> [k23_practice args]")
        return 2
    ns, extra = argv[0], argv[1:]
    try:                                                            # Astra K23-TOOLING-G1-r2 1: whitelist (abbreviations refused)
        KP.passthrough(extra)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    if any(x.split("=", 1)[0] in ("--ollama", "--steal-lock") for x in extra):
        print("REFUSED: inner.sh sets --ollama (the Windows host) and --steal-lock (inside the sim lock); pass only --limit N")
        return 2
    if "--orders" in extra or "--k" in extra:
        print("REFUSED: the registered K23 practice is --orders 0 --k 10 (k23_practice defaults); do not override them here")
        return 2
    why = KP.readiness(ns, [0], [10], perception=False)       # the perception freeze is verified inside WSL by k23_practice
    if why:
        print("REFUSED (nothing started): " + "; ".join(why[:8]))
        return 2
    ok, info = ollama_ok()
    if not ok:
        print(f"REFUSED: Ollama at 127.0.0.1:11434 does not serve {GEMMA} ({info}); start it with OLLAMA_HOST=0.0.0.0")
        return 2
    env = C.wsl_key_env(os.environ)
    rc = subprocess.call(["wsl", "-d", "Ubuntu-22.04", "--cd", E.wsl_path(REPO), "--", "bash",
                          "sim/final_test/k23_run/practice/run.sh", ns] + extra, env=env)
    print("launcher exit", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
