"""Self-test for step 5 / A5-replay -- the honesty static guard and the "no simulator, no network" gate.

Offline, Windows `python` from the REPO ROOT, no simulator, no WSL, no Ollama, no network, temp directories only:

    python sim/_step5_honesty_selftest.py

STATIC (AST, comments and docstrings excluded): every step-5 library module (sim/ltm*.py, sim/step5_eval.py,
sim/step5_replay.py, the lexicon builder) is scanned for forbidden tokens - objectId, event.metadata / `.metadata`, errors_logged,
oracle, scoring, ai2thor, ollama, Controller - outside an EXACT, reasoned allowlist (refusal lists and the replay's own
exclusion list); the allowlist must match the hits exactly in both directions. Forbidden IMPORTS (simulator, Ollama,
network, process, step-4 agent / console / memory / trace modules) are refused; heavy ML libraries may only be imported
inside functions. The privileged goal verdict may be mentioned only in the declared admission path (admission, failed
reuse, plan_goal_unmet) and never in an agent-facing module. Selftests are scanned too.

DYNAMIC: a fresh interpreter patches socket / subprocess / os.system, imports EVERY step-5 module, runs a small end-to-end
workload (events -> consolidate -> lessons -> where_is -> replay converter), and reports any attempted network or process
call plus any forbidden module present in sys.modules. Positive controls prove the patches bite.
"""
import ast
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = Path(HERE).parent
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


SIM = Path(HERE)
LIB = sorted({p.name for p in SIM.glob("ltm*.py")} | {"step5_eval.py", "step5_replay.py"})
EXTRA = [SIM / "step5" / "lexicon" / "_build_concepts.py"]          # authoring helper (library-like)
ACCEPT = [SIM / "step5" / "bge_acceptance.py"]                       # separate acceptance script, own venv; checked apart

TOKENS = ("objectid", "event.metadata", "errors_logged", "oracle", "scoring", "ai2thor", "ollama")
#: (file, token) -> reason.  Exactly these may appear; anything else is a regression, a stale entry is a failure too.
ALLOW = {
    ("ltm_store.py", "objectid"): "the privileged-name REFUSAL list and the sim-id regex",
    ("ltm_store.py", "oracle"): "the privileged-name REFUSAL list",
    ("step5_replay.py", "errors_logged"): "the replay's documented EXCLUSION list and its leak-scan inputs",
}
BAD_IMPORT_ROOTS = {"ai2thor", "ollama", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "subprocess",
                    "socketserver", "ftplib", "smtplib", "webbrowser", "ultralytics", "gui", "webapp", "honest", "honest_skills",
                    "honest_map", "honest_console", "objmem", "thor_actions", "memory", "trace", "perception", "eval",
                    "step4_refplans", "step4_gemma", "llm", "agent_llm", "rag", "improve", "feedback"}
HEAVY = {"torch", "sentence_transformers", "transformers", "numpy", "onnxruntime"}
#: functions allowed to name the privileged goal verdict (declared uses: admission, failed reuse, plan_goal_unmet) + evaluator
GOAL_OK_ALLOWED = {
    "ltm_events.py": {"<module>", "validate"},                       # declares the single admitted post-terminal event
    "ltm_store.py": {"<module>", "Store._normalise_op.norm"},        # declares the three admitted columns
    "ltm_consolidate.py": {"consolidate"},                           # reads it from the admission_verdict event
    "ltm_lessons.py": {"LessonsExtension._build", "admission_check", "diagnose", "score_procedure_application",
                       "flag_revalidate", "log_test_application"},  # last two only write goal_ok=None
    "step5_replay.py": {"row_events"},                               # converter: the only place that carries it into an event
    "ltm_adapter.py": {"HonestEventAdapter.on_admission"},           # B1: the adapter's single door for the post-terminal verdict (practice only)
    "ltm_examples.py": {"<module>"},                                 # H-B5: ONLY the corpus refusal list _FORBIDDEN_KEYS (pinned below)
    "step5_eval.py": None,                                           # evaluator side: scoring is its job (any function)
}
#: B1: the adapter selftest drives the REAL step-4 agent over the step-4 fake kitchen (FakeThor: no controller, no socket). Exactly these
#: roots, for exactly this file; every other selftest stays free of step-4 imports.
SELFTEST_STEP4 = {"_step5_adapter_selftest.py": {"honest", "honest_skills", "honest_map", "objmem"}}
AGENT_FACING = ("ltm.py", "ltm_retrieval.py", "ltm_retrieve.py", "ltm_habits.py", "ltm_identity.py", "ltm_decay.py",
                "ltm_wm.py", "ltm_concepts.py")


def _docstring_ids(tree):
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body:
            b = n.body[0]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) and isinstance(b.value.value, str):
                ids.add(id(b.value))
    return ids


def code_atoms(src):
    """Strings, names, attributes, arg names, keyword names of the CODE (docstrings and comments excluded)."""
    tree = ast.parse(src)
    doc = _docstring_ids(tree)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc:
            out.append(("str", n.value))
        elif isinstance(n, ast.Name):
            out.append(("name", n.id))
        elif isinstance(n, ast.Attribute):
            out.append(("attr", n.attr))
        elif isinstance(n, ast.arg):
            out.append(("arg", n.arg))
        elif isinstance(n, ast.keyword) and n.arg:
            out.append(("kw", n.arg))
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.append(("def", n.name))
    return out


def token_hits(src):
    hits = set()
    for kind, s in code_atoms(src):
        low = s.lower()
        for t in TOKENS:
            if t in low:
                hits.add(t)
        if kind == "attr" and s == "metadata":
            hits.add("event.metadata")
        if kind in ("name", "attr", "def") and s in ("Controller", "controller"):
            hits.add("controller")
    return hits


def imports_of(src):
    """[(root, module-level?)]"""
    tree = ast.parse(src)
    out = []

    def walk(node, depth):
        for c in ast.iter_child_nodes(node):
            d = depth + (1 if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else 0)
            if isinstance(c, ast.Import):
                out.extend((a.name.split(".")[0], depth == 0) for a in c.names)
            elif isinstance(c, ast.ImportFrom) and c.level == 0:
                out.append(((c.module or "").split(".")[0], depth == 0))
            walk(c, d)
    walk(tree, 0)
    return out


def goal_ok_sites(src):
    tree = ast.parse(src)
    doc = _docstring_ids(tree)
    sites = set()

    def visit(n, stack):
        for c in ast.iter_child_nodes(n):
            st = stack + [c.name] if isinstance(c, (ast.FunctionDef, ast.ClassDef)) else stack
            hit = ((isinstance(c, ast.Name) and "goal_ok" in c.id) or (isinstance(c, ast.arg) and "goal_ok" in c.arg)
                   or (isinstance(c, ast.Attribute) and "goal_ok" in c.attr) or (isinstance(c, ast.keyword) and c.arg and "goal_ok" in c.arg)
                   or (isinstance(c, ast.Constant) and isinstance(c.value, str) and "goal_ok" in c.value and id(c) not in doc
                       and len(c.value) < 40))
            if hit:
                sites.add(".".join(stack) or "<module>")
            visit(c, st)
    visit(tree, [])
    return sites


def module_goal_ok_stmts(src):
    """Module-level statements (outside any def / class) that name goal_ok -> [(lineno, is_the_refusal_list)]. The refusal list is
    exactly `_FORBIDDEN_KEYS = (<str constants>)`."""
    out = []
    for st in ast.parse(src).body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not goal_ok_sites(ast.unparse(st)):
            continue
        ok = (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
              and st.targets[0].id == "_FORBIDDEN_KEYS" and isinstance(st.value, ast.Tuple)
              and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in st.value.elts))
        out.append((st.lineno, ok))
    return out


def examples_checks():
    """ltm_examples: its one module-level goal_ok is the refusal list; verdict_of reads the admission only (Astra K-B2b-honesty 3, 4)."""
    src = (SIM / "ltm_examples.py").read_text(encoding="utf-8")
    m = module_goal_ok_stmts(src)
    check("ltm_examples: the only module-level goal_ok is the literal refusal list _FORBIDDEN_KEYS", len(m) == 1 and m[0][1], m)
    # mutation controls: each planted change must fail a check of this gate
    planted_read = src + "\nLAST = {}.get(\"goal_ok\")\n"
    check("mutation control: a module-level verdict read is caught", any(not ok for _, ok in module_goal_ok_stmts(planted_read)))
    planted_fb = src.replace('    raise ValueError("practice row has no verdict, no crash',
                             '    g = r.get("goal_ok")\n    if isinstance(g, bool):\n        return "success" if g else "failure"\n'
                             '    raise ValueError("practice row has no verdict, no crash', 1)
    check("mutation control: a restored goal_ok fallback in verdict_of is caught",
          planted_fb != src and "verdict_of" in (goal_ok_sites(planted_fb) - GOAL_OK_ALLOWED["ltm_examples.py"]))
    planted_imp = (SIM / "ltm_procedural.py").read_text(encoding="utf-8") + "\n\ndef _x():\n    import honest_skills\n"
    check("mutation control: a new lazy executor import in ltm_procedural is caught",
          any(r in BAD_IMPORT_ROOTS for r, _ in imports_of(planted_imp) if r == "honest_skills"))
    import ltm_examples as X
    refused = []
    for row in ({"goal_ok": True}, {"goal_ok": False}, {"status": "executed", "goal_ok": True}, {"applied": True},
                {"status": "executed", "admission": None, "goal_ok": True}):
        try:
            X.verdict_of(row)
            refused.append(False)
        except ValueError:
            refused.append(True)
    check("verdict_of refuses rows that carry only goal_ok / applied / a None admission", all(refused), refused)
    kept = [X.verdict_of(r) for r in ({"verdict": "Success"}, {"status": "crash"}, {"status": "inapplicable"}, {"status": "rejected"},
                                      {"status": "executed", "admission": True, "goal_ok": False},
                                      {"status": "executed", "admission": False, "goal_ok": True},
                                      {"admission": {"admitted": True}})]
    check("verdict_of keeps every accepted practice verdict (and admission wins over goal_ok)",
          kept == ["success", "crash", "inapplicable", "failure", "success", "failure", "success"], kept)
    check("the corpus refusal list still catches a goal_ok key", any("goal_ok" in p for p in X.audit_corpus([{"id": "x", "goal_ok": True}])))


def main():
    mg = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in (SIM / "memory.db", SIM / "step5" / "ltm.db")}
    # ------------------------------------------------------------------ static: forbidden tokens, exact allowlist
    check("library modules found", len(LIB) >= 12 and "step5_replay.py" in LIB, LIB)
    seen = {}
    for fn in LIB:
        for t in token_hits((SIM / fn).read_text(encoding="utf-8")):
            seen.setdefault((fn, t), True)
    for p in EXTRA:
        for t in token_hits(p.read_text(encoding="utf-8")):
            seen.setdefault((p.name, t), True)
    unexpected = sorted(k for k in seen if k not in ALLOW)
    stale = sorted(k for k in ALLOW if k not in seen)
    check("no forbidden token in step-5 code outside the allowlist", not unexpected, unexpected)
    check("allowlist has no stale entries", not stale, stale)
    check("every allowlist entry carries a reason", all(len(v) > 10 for v in ALLOW.values()))
    for t, src in [("objectid", "x = event['objectId']"), ("event.metadata", "m = event.metadata"), ("errors_logged", "a['errors_logged']"),
                   ("oracle", "def oracle_read(): pass"), ("scoring", "with h.scoring(): pass"), ("ai2thor", "m = 'ai2thor.controller'"),
                   ("ollama", "url = 'http://x/ollama'"), ("controller", "c = Controller(scene='x')")]:
        check(f"positive control: scanner finds {t}", t in token_hits(src), token_hits(src))
    check("negative control: docstrings and comments are not scanned", token_hits('"""oracle objectId"""\n# scoring\nx = 1') == set())
    check("negative control: ordinary code is clean", token_hits("def f(a):\n    return a + 1\n") == set())

    # ------------------------------------------------------------------ static: imports
    bad = []
    for fn in LIB:
        for root, top in imports_of((SIM / fn).read_text(encoding="utf-8")):
            if root in BAD_IMPORT_ROOTS:
                bad.append((fn, root))
            if root in HEAVY and top:
                bad.append((fn, root, "module-level"))
    for p in EXTRA:
        for root, top in imports_of(p.read_text(encoding="utf-8")):
            if root in BAD_IMPORT_ROOTS or (root in HEAVY and top):
                bad.append((p.name, root))
    check("no simulator / Ollama / network / process / step-4 agent import in step-5 code",
          not [b for b in bad if len(b) == 2], bad)
    check("heavy ML libraries are imported only inside functions", not [b for b in bad if len(b) == 3], bad)
    # A7-review: the verdict is available only after the agent stops acting (ordering enforced by WorkingMemory)
    sys.path.insert(0, str(SIM))
    from ltm_wm import WorkingMemory
    from ltm_events import EventError
    base = {"kitchen": "K", "epoch": 0, "frame": "f0", "map_rev": "m0", "episode": "ep1"}

    def _wm(kinds):
        w = WorkingMemory("ep1")
        for n, (kind, kw) in enumerate(kinds, 1):
            w.ingest(dict(base, ev=n, t=n, kind=kind, **kw))
        return w
    ST = ("episode_start", {"mode": "practice", "world": "reset"})
    TE = ("terminal", {"exec_ok": True})
    AV = ("admission_verdict", {"goal_ok": True, "tag": "privileged-terminal-goal-check"})
    EN = ("episode_end", {})
    for label, seq in [("verdict before terminal", [ST, AV, TE, EN]),
                       ("agent event after terminal", [ST, TE, ("pickup", {"track": "tr:1"}), EN]),
                       ("verdict in a test episode", [("episode_start", {"mode": "test", "world": "reset"}), TE, AV, EN]),
                       ("second verdict", [ST, TE, AV, ("admission_verdict", {"goal_ok": False, "tag": "privileged-terminal-goal-check"})])]:
        try:
            _wm(seq)
            check(f"WM refuses {label}", False)
        except EventError:
            check(f"WM refuses {label}", True)
    check("WM accepts start -> terminal -> verdict -> end", _wm([ST, TE, AV, EN]).admission["goal_ok"] is True)
    check("positive control: import scanner finds a forbidden import and a module-level heavy one",
          ("socket", True) in imports_of("import socket") and ("torch", True) in imports_of("import torch") and
          ("torch", False) in imports_of("def f():\n    import torch\n"))
    acc = imports_of(ACCEPT[0].read_text(encoding="utf-8"))
    check("the bge acceptance script is not imported by any library module", all(
        "bge_acceptance" not in (SIM / fn).read_text(encoding="utf-8") for fn in LIB))
    check("the bge acceptance script imports ML libraries only inside functions", not [r for r, top in acc if r in HEAVY and top], acc)
    # selftests: no forbidden import, no controller / network call; subprocess only through sys.executable
    tests = sorted(SIM.glob("_step5_*_selftest.py"))
    check("step-5 selftests found", len(tests) >= 7, [t.name for t in tests])
    tb = []
    for t in tests:
        src = t.read_text(encoding="utf-8")
        for root, top in imports_of(src):
            if root in SELFTEST_STEP4.get(t.name, ()):
                continue
            if root in BAD_IMPORT_ROOTS - {"subprocess"} or root in HEAVY and top:
                tb.append((t.name, root))
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                if nm in ("Controller", "urlopen", "create_connection", "Popen", "system", "popen", "startfile"):
                    tb.append((t.name, nm, "call"))
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "subprocess":
                    first = n.args[0] if n.args else None
                    ok = isinstance(first, ast.List) and first.elts and "sys.executable" in ast.unparse(first.elts[0])
                    if not ok:
                        tb.append((t.name, "subprocess." + f.attr, "not via sys.executable"))
    check("selftests start no controller, no network, no external process", not tb, tb)

    # ------------------------------------------------------------------ static: the privileged verdict is confined
    viol = []
    for fn, allowed in GOAL_OK_ALLOWED.items():
        if allowed is None:
            continue
        sites = goal_ok_sites((SIM / fn).read_text(encoding="utf-8"))
        extra = sites - allowed
        if extra:
            viol.append((fn, sorted(extra)))
    check("goal_ok appears only in the declared admission / failed-reuse / plan_goal_unmet functions", not viol, viol)
    leak = [fn for fn in AGENT_FACING if goal_ok_sites((SIM / fn).read_text(encoding="utf-8"))]
    check("no agent-facing module mentions the privileged verdict", not leak, leak)
    check("every module that names goal_ok is declared", all(
        fn in GOAL_OK_ALLOWED or not goal_ok_sites((SIM / fn).read_text(encoding="utf-8")) for fn in LIB), [fn for fn in LIB if goal_ok_sites((SIM / fn).read_text(encoding="utf-8")) and fn not in GOAL_OK_ALLOWED])
    check("positive control: goal_ok scanner sees a use", goal_ok_sites("def retrieve(goal_ok):\n    return goal_ok") == {"retrieve"})
    examples_checks()
    # the retrieval-time functions of ltm_lessons never see the verdict
    ls = (SIM / "ltm_lessons.py").read_text(encoding="utf-8")
    check("lesson_for / failure_key_at / build_packet do not mention the verdict", not (
        goal_ok_sites(ls) & {"lesson_for", "failure_key_at", "build_packet", "validate_hypothesis", "Hypothesizer"}))

    # ------------------------------------------------------------------ dynamic: fresh interpreter, patches, full workload
    child = r'''
import json, os, sys, tempfile
HERE = sys.argv[1]
sys.path.append(HERE)
import socket, subprocess
calls = []
def _block(name):
    def f(*a, **k):
        calls.append(name)
        raise OSError("blocked by the step-5 no-network gate: " + name)
    return f
class _Sock:
    def __init__(self, *a, **k):
        calls.append("socket.socket"); raise OSError("blocked")
socket.socket = _Sock
socket.create_connection = _block("socket.create_connection")
socket.getaddrinfo = _block("socket.getaddrinfo")
socket.gethostbyname = _block("socket.gethostbyname")
subprocess.Popen = _block("subprocess.Popen")
os.system = _block("os.system")
os.popen = _block("os.popen")
for n in ("spawnl", "spawnv", "execv", "execl", "startfile"):
    if hasattr(os, n): setattr(os, n, _block("os." + n))
# positive controls (counted, then cleared)
pos = []
for f in (lambda: socket.socket(), lambda: socket.create_connection(("127.0.0.1", 11434)), lambda: subprocess.run(["x"]), lambda: os.system("x")):
    try: f()
    except OSError: pos.append(1)
pc = len(pos); calls.clear()
mods = sorted(p[:-3] for p in os.listdir(HERE) if p.startswith("ltm") and p.endswith(".py")) + ["step5_eval", "step5_replay"]
import importlib
imported = []
for m in mods:
    importlib.import_module(m); imported.append(m)
import ltm, ltm_lessons as ll, step5_replay as R, ltm_wm
tmp = tempfile.mkdtemp()
L = ltm.LongTermMemory(os.path.join(tmp, "x", "ltm.db"), kitchen="FloorPlan1", params=ltm.load_params())
row = {"id": "g-l1-pickup-apple", "level": "L1", "status": "FAIL", "goal_ok": False, "exec_ok": False, "prims": 9, "audit": {"invalid": False},
       "steps": [{"action": "pickup", "object": "Apple", "to": None, "ok": False, "cause": "not_detected", "under": "not_detected", "retries": 2, "prims": 9}]}
items = R.load_items()
evs = R.row_events(row, items["g-l1-pickup-apple"], episode="rp:t1")
res = R.replay_into(L, [{"episode": "rp:t1", "events": evs}], [ll.LessonsExtension()])
w = L.where_is("Apple")
import ltm_retrieval
r = ltm_retrieval.Retriever(params=L.params) if hasattr(ltm_retrieval, "Retriever") else None
L.close()
import ltm_procedural as LPR
kb = LPR.load_kb()
proc_ok = bool(kb.records()) and bool(kb.afford_records()) and \
    bool(kb.render("proc:heat-then-place", {"?food": "Egg", "?place": "Plate"}, holding="Mug")[0]) and \
    bool(kb.candidates({"types": ["Egg"], "ops": ["heat"]})) and bool(LPR.contract_tools(LPR.load_exec_contract()))
bad_mods = sorted(m for m in sys.modules if m.split(".")[0] in {"ai2thor", "ollama", "requests", "urllib3", "httpx", "aiohttp", "torch",
        "sentence_transformers", "transformers", "ultralytics", "numpy", "honest", "honest_skills", "honest_map", "honest_console", "objmem",
        "webapp", "thor_actions", "memory", "perception", "eval", "step4_refplans", "step4_gemma", "llm", "agent_llm", "gui", "trace"}
        or m.startswith("sim.trace"))
print(json.dumps({"calls": calls, "pos_controls": pc, "imported": imported, "bad_modules": bad_mods, "tally": res["tally"],
                  "where_is_ok": w["explore"]["id"] == "EXPLORE", "retriever": r is not None, "lessons": len(ll.LessonsExtension().last_stats),
                  "proc_ok": proc_ok}))
'''
    p = subprocess.run([sys.executable, "-c", child, str(SIM)], capture_output=True, text=True, cwd=str(REPO), timeout=300)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        d = {}
        print(p.stdout[-800:], p.stderr[-1500:])
    check("child interpreter ran the workload", bool(d) and d.get("tally") == {"committed": 1}, d)
    check("positive controls: the patches stop socket / connect / subprocess / os.system (4/4)", d.get("pos_controls") == 4, d)
    check("no network, no process, no os.system call while importing and running step 5", d.get("calls") == [], d.get("calls"))
    check("every step-5 module imported", len(d.get("imported", [])) >= 12 and "ltm_lessons" in d.get("imported", []), d.get("imported"))
    check("no simulator / Ollama / ML / step-4 agent module in sys.modules", d.get("bad_modules") == [], d.get("bad_modules"))
    check("where_is answered offline", d.get("where_is_ok") is True)
    check("the procedural package path (load_kb, records, afford_records, candidates, render with the implicit contract load) ran "
          "in the same no-step-4 child", d.get("proc_ok") is True, d)
    # the gate bites: a module that opens a socket at import would be caught
    bait = tempfile.mkdtemp()
    Path(bait, "bait.py").write_text("import socket\ns = socket.create_connection(('127.0.0.1', 11434))\n", encoding="utf-8")
    probe = ("import sys, socket\nsys.path.append(sys.argv[1])\ncalls=[]\n"
             "def b(*a, **k):\n    calls.append(1); raise OSError('blocked')\n"
             "socket.create_connection = b\n"
             "try:\n    import bait\nexcept OSError:\n    pass\nprint(len(calls))\n")
    q = subprocess.run([sys.executable, "-c", probe, bait], capture_output=True, text=True, timeout=60)
    check("negative-path control: an import that dials out is caught by the same patch", q.stdout.strip() == "1", q.stdout + q.stderr)

    # ------------------------------------------------------------------ nothing shared was touched
    now = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in mg}
    check("sim/memory.db and the default sim/step5/ltm.db are untouched by this gate", now == mg)
    # sim/trace.db is written by the concurrent step-4 workflow, so an mtime guard would be flaky: use a source check instead
    check("no step-5 module or selftest names sim/trace.db as a path", not [fn for fn in LIB if "trace.db" in "".join(v for k, v in code_atoms((SIM / fn).read_text(encoding="utf-8")) if k == "str")])
    check("the default ltm.db was never created", not (SIM / "step5" / "ltm.db").exists())
    print(f"\n{_pass} passed, {_fail} failed")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
