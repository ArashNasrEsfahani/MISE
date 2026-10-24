# Environment addendum -- acceptance attempt 2 (append-only)

Plumbing fix to the run environment, not to repo code, records or rules. Astra ACC2-LAUNCH-r1: ACCEPT WITH NOTES (docs/final-test/_astra/ACC2-LAUNCH-r1.raw.txt).

- **Problem:** acc_attempt2.py's AST fingerprints use ast.dump. Attempt-1 values were recorded on Windows Python 3.14.0, and 3.13+ omits None-valued optional fields and empty lists. The WSL simulator venv runs Python 3.10.12, so byte-identical code was refused at 13:46:17 and 14:13:35. Both refusals came before any capture: nothing was written and nothing was spent.
- **Fix:** /home/arash/thor-env/lib/python3.10/site-packages/acc_ast_shim.py (sha256 1adf0dea55aaca698b450d1c3a1b130c9ab82ed118d4ff7052b78783e7485051; repo copy sim/final_test/_checkpoints/acc_ast_shim.py sha256 1adf0dea55aaca698b450d1c3a1b130c9ab82ed118d4ff7052b78783e7485051) plus acc_ast_shim.pth (sha256 2540e52c6fa52312d58a27a3efe678506d738514b84aee21e3fc5ae9b701112f, content "import acc_ast_shim").
  - It is active only for processes whose command line names acc_runner.py, on Python < 3.13.
  - It emulates the 3.13+ default ast.dump output.
- **Why not sitecustomize:** a first attempt as venv sitecustomize.py was shadowed by Ubuntu's /usr/lib/python3.10/sitecustomize.py and had no effect. It was removed.
- **Evidence:** run by Arash on WSL 3.10.12, the fingerprints planner ef73e8b27664, rendering c4390007fcab and rule d43568cd4785 equal the Windows 3.14 values, and preflight2(capture) = [].
- **Chronology:**
  - Arash approved: "i approve them both".
  - Arash installed the fix himself through the session prompt, because the permission classifier blocked the agent.
  - Arash launched chain_a2.sh at 14:23:51.
  - Capture passed the guard.
- **Reproducibility:** the shim and .pth are external to code_state and are archived here. No fingerprint was re-recorded.
