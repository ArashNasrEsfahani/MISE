"""M3 freeze (final test, workflow stage M3-freeze; contract in SPEC.md s12 "M3 freeze.py" and prereg s12 gate 1).

Builds, verifies and seals ``docs/final-test/freeze.json`` (schema ``final-test.freeze/1``): ONE file that pins everything the test
may not change afterwards, and records every pre-test gate (12.1-12.13, G-iso, A12 coverage, A14 validation, snapshot hash,
leakage, real-token budget) with its evidence.

    python sim/final_test/matrix/freeze.py --write  [--rehearsal-ns rehearsal-v1] [--blocks-from rehearsal-v1]
    python sim/final_test/matrix/freeze.py --check  [--blocks-from test-v1 --allow-test]
    python sim/final_test/matrix/freeze.py --record-arash-check --by Arash --date YYYY-MM-DD --arash-said "<his words>"
    python sim/final_test/matrix/freeze.py --record-treatment     (TREAT-FREEZE: after the test-v1 flip; treatment record only)

What it hashes (everything is relative to the repo root, forward slashes):
  prereg, split manifest, dataset, families approval, params.json (+ the A12 values), the prompt (version, system template, rendered
  system text with the frozen 49-type list, user template, PLAN_SCHEMA), detector weights, the walked map, every snapshot (S* = o0/k10
  and the nested curve snapshots), the FS blocks, the reset tolerances, the evidence file of every gate, and every ``sim/`` module on
  the planning and execution path (the static import closure of capture, plan, execute, score, real_counter, reset_gate).

How a gate gets its verdict (never by a hand-typed flag):
  * computed here from files: 12.1 (completeness), 12.4 + A14 (reset_gate tolerances validated 24/24), 12.6 snapshot hash and nesting,
    12.8 (families approved by Arash), 12.9 (Ollama digest = the A8 digest), 12.11 (one renderer), 12.12 (type list = detector classes),
    12.13 (cache namespaces), A12 coverage and the REAL-TOKEN gate (every S* / curve / FS block, complete ``header + block``, under
    BOTH tokenizers, EN and FA, <= its level budget, record ids identical across languages);
  * read from ``docs/final-test/matrix/gates/<id>.json`` (written by the stage that ran the check; pass must be a real ``true``,
    synthetic evidence is refused): 12.2, 12.3, 12.5, 12.7, 12.10, G-iso, leakage.
A gate that cannot be evaluated is FAILED with a reason, never skipped.

``--check`` rebuilds the whole file from the disk and compares every deterministic field with the stored one, so any later edit to a
frozen file, a code module, a gate's evidence or the freeze itself is reported. ``--record-arash-check`` fills ``arash_check`` only on
Arash's explicit instruction and only if every gate passes; the file is then re-sealed (self sha). Test items: the module itself
never reads an instruction or a plan; blocks of a test namespace can be read only through ``common.guard_items`` (needs
``--allow-test``, namespace ``test-v1``, a valid freeze and Arash's family approval), and ``--write`` accepts non-test blocks only.

Selftest: ``python sim/_ftm_freeze_selftest.py`` (offline, synthetic data). Python 3.10 compatible.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                              # noqa: E402  (also puts sim/ on sys.path)

FREEZE_SCHEMA = "final-test.freeze/1"
GATE_SCHEMA = "ftm.gate/1"
EXPECTED_TYPES = 49                    # prereg 12.12: the class list of the frozen detector weights
OLLAMA_MODEL = "gemma4:e2b-it-q4_K_M"
OLLAMA_DIGEST_PREFIX = "b37049369adf"  # prereg A8 (Arash, 2026-10-10)
LEVELS = ("L1", "L2", "L3")
LANGS = ("en", "fa")

#: required code entry points (relative to the repo root); a missing one fails gate 12.1.
ENTRY_REQUIRED = ("sim/final_test/matrix/capture.py", "sim/final_test/matrix/plan.py", "sim/final_test/matrix/execute.py",
                  "sim/final_test/matrix/score.py", "sim/final_test/c3/real_counter.py", "sim/final_test/reset_gate.py",
                  "sim/final_test/matrix/common.py", "sim/planners/contract.py")
#: further entry points hashed when they exist (they run before / around the freeze).
ENTRY_OPTIONAL = ("sim/final_test/matrix/practice_k.py", "sim/final_test/matrix/fs_block.py", "sim/final_test/matrix/matrix_run.py",
                  "sim/final_test/matrix/monitor.py", "sim/final_test/matrix/analyze.py", "sim/final_test/matrix/retention_a12.py",
                  "sim/final_test/matrix/x_isolation_probe.py", "sim/final_test_power.py", "sim/final_test_power_secondary.py", "sim/final_test_callplan.py",
                  "sim/planners/codex_provider.py")
#: gates whose verdict is an evidence file written by the stage that ran the check.
EVIDENCE_GATES = ("12.2", "12.3", "12.5", "12.7", "12.10", "G-iso", "leakage", "X-iso")
GATE_ORDER = ("12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "12.10", "12.11", "12.12", "12.13",
              "G-iso", "A12", "A14", "snapshot", "leakage", "tokens", "X-iso")
SNAP_ORDERS, SNAP_KS = (0, 1, 2), (0, 1, 3, 5, 10)
SNAP_FILES = ("manifest.json", "ltm.db", "fragments.json")
#: fields of freeze.json that --check ignores when comparing a rebuild with the stored file.
VOLATILE = ("created", "arash_check", "sha256")


# ================================================================================================ small helpers
def _rel(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def _abs(root, rel):
    return os.path.join(root, *rel.split("/"))


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha_file(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def gate(ok, **evidence):
    return {"pass": bool(ok), "evidence": evidence}


def matrix_gates_dir(root):
    return os.path.join(C.paths(root)["matrix_docs"], "gates")


def tokens_rows_path(root):
    return os.path.join(C.paths(root)["matrix_docs"], "freeze_tokens.json")


def detector_types_path(root):
    return os.path.join(C.paths(root)["matrix_docs"], "detector_types.json")


# ================================================================================================ code closure (gate 12.1)
def _search_dirs(root):
    sim = os.path.join(root, "sim")
    return [sim, os.path.join(sim, "final_test"), os.path.join(sim, "final_test", "matrix"), os.path.join(sim, "final_test", "c3")]


def _resolve(root, parts, here):
    """Module path parts -> the local .py file (or package __init__) it names, else None (stdlib / third party)."""
    for base in [here] + _search_dirs(root):
        p = os.path.join(base, *parts)
        if os.path.isfile(p + ".py"):
            return p + ".py"
        if os.path.isfile(os.path.join(p, "__init__.py")):
            return os.path.join(p, "__init__.py")
    return None


def _resolve_local_only(root, parts, here):
    """A stdlib-named module only counts when it sits NEXT TO the importing file (a sibling shadows the stdlib there)."""
    p = os.path.join(here, *parts)
    return os.path.isfile(p + ".py") or os.path.isfile(os.path.join(p, "__init__.py"))


def _imports_of(path):
    """(absolute dotted names, [(level, module, [names])] for from-imports) of one file; lazy imports inside functions count."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    plain, froms = [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            plain += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            froms.append((n.level, n.module or "", [a.name for a in n.names]))
    return plain, froms


def import_closure(root, entries):
    """Static import closure of `entries` over files under `root/sim` (nothing is executed). Returns (files, missing_entries, errors):
    files = sorted repo-relative paths, including the entries themselves."""
    seen, missing, errors, stack = {}, [], [], []
    for e in entries:
        p = _abs(root, e)
        if os.path.isfile(p):
            stack.append(p)
        else:
            missing.append(e)
    simdir = os.path.join(root, "sim")
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen[p] = True
        here = os.path.dirname(p)
        try:
            plain, froms = _imports_of(p)
        except (OSError, SyntaxError, ValueError) as ex:
            errors.append(f"{_rel(root, p)}: {type(ex).__name__}")
            continue
        cands = [(n.split("."), here) for n in plain]
        for level, mod, names in froms:
            base = here
            if level:
                for _ in range(level - 1):
                    base = os.path.dirname(base)
                parts = mod.split(".") if mod else []
                cands.append((parts, base)) if parts else None
                for nm in names:
                    cands.append((parts + [nm], base))
            else:
                parts = mod.split(".")
                cands.append((parts, here))
                for nm in names:
                    cands.append((parts + [nm], here))
        for parts, h in cands:
            if not parts or not parts[0]:
                continue
            if parts[0] in sys.stdlib_module_names and h == here and not _resolve_local_only(root, parts, here):
                continue
            for k in range(1, len(parts) + 1):               # a.b.c also pulls a/__init__ and a/b/__init__ (packages)
                f = _resolve(root, parts[:k], h)
                if f and os.path.abspath(f).startswith(os.path.abspath(simdir)):
                    stack.append(f)
    files = sorted(_rel(root, p) for p in seen)
    return files, missing, errors


# ================================================================================================ probes (live facts; injectable)
def probe_ollama(endpoint="http://127.0.0.1:11434", model=OLLAMA_MODEL, timeout=10):
    """Digest, version and parent model of the served Gemma tag. Never raises; an unreachable server is `error`."""
    out = {"model": model, "digest": None, "version": None, "parent_model": None, "error": None}
    try:
        r = urllib.request.Request(endpoint.rstrip("/") + "/api/show", json.dumps({"model": model}).encode("utf-8"),
                                   {"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
        out["parent_model"] = (d.get("details") or {}).get("parent_model")
        r2 = urllib.request.Request(endpoint.rstrip("/") + "/api/tags")
        for m in json.loads(urllib.request.urlopen(r2, timeout=timeout).read()).get("models", []):
            if m.get("name") == model or m.get("model") == model:
                out["digest"] = m.get("digest")
        out["version"] = json.loads(urllib.request.urlopen(endpoint.rstrip("/") + "/api/version", timeout=timeout).read()).get("version")
    except Exception as e:                                       # noqa: BLE001  (transport / parse: recorded, the gate then fails)
        out["error"] = type(e).__name__
    return out


def probe_codex(timeout=20):
    try:
        r = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=timeout)
        return {"version": r.stdout.strip() or None, "error": None if r.returncode == 0 else f"exit {r.returncode}"}
    except Exception as e:                                       # noqa: BLE001
        return {"version": None, "error": type(e).__name__}


# ================================================================================================ real-token counter (check mode)
class CachedCounter:
    """Offline view of `RealMaxCounter`'s on-disk cache (same format): `counts(text)` per tokenizer, minus the one-token baseline.
    A text that is not in the cache raises KeyError (never a silent estimate). No network, no key."""

    def __init__(self, cache_path, tokenizers=("gemma", "anthropic")):
        self.cache = _read_json(cache_path) if os.path.exists(cache_path) else {}
        self.tokenizers = list(tokenizers)
        self.header = ""

    def _raw(self, txt, field):
        import hashlib
        k = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]
        return self.cache[k][field]

    def counts(self, text):
        if not text.strip():
            return {k: 1 for k in self.tokenizers}
        out = {"gemma": self._raw(text, "gemma_raw") - self._raw(".", "gemma_raw")}
        if "anthropic" in self.tokenizers:
            out["anthropic"] = self._raw(text, "anth_raw") - self._raw(".", "anth_raw")
        return out


def live_counter(root, ollama_endpoint="http://127.0.0.1:11434"):
    """The real counter for --write: Ollama (Gemma) + the free /v1/messages/count_tokens, cached under the work dir. Needs
    THESIS_ANTHROPIC_API_KEY (through planners.keys); never prints it."""
    sys.path.insert(0, os.path.join(root, "sim", "final_test", "c3"))
    from real_counter import RealMaxCounter                      # noqa: E402
    from planners.contract import MEMORY_HEADER                  # noqa: E402
    return RealMaxCounter(os.path.join(C.paths(root)["matrix_work"], "tokcache.json"), MEMORY_HEADER, ollama_endpoint)


# ================================================================================================ blocks and the token gate
def load_capture_rows(root, ns, allow_test=False):
    """Capture rows of a namespace for the gates. Goes through `common.guard_items` first (test items need the full guard)."""
    if ns == "test-v1":                                             # the whole guard holds before the journal is even opened
        why = C.freeze_status(root)["reasons"] + C.families_status(root)["reasons"]
        if not allow_test:
            why.append("--allow-test not given")
        if why:
            raise C.TestItemRefused("test namespace refused: " + "; ".join(why))
    p = C.journal_path(ns, "capture", root)
    rows = C.Journal(p).rows() if os.path.exists(p) else []
    C.guard_items(sorted({r["item"] for r in rows}), ns, allow_test=allow_test, root=root)
    return rows


def block_rows(capture_rows, fs_blocks=None):
    """Flatten capture rows (+ the FS file) into the rows the token gate counts: only text, ids, level, lang, item, source."""
    out = []
    for r in capture_rows:
        for src, b in sorted((r.get("blocks") or {}).items()):
            if src == "none" or not isinstance(b, dict):
                continue
            out.append({"id": f"{r['pres']}#{src}", "item": r["item"], "level": r["level"], "lang": r["lang"], "source": src,
                        "text": b.get("text") or "", "ids": list(b.get("ids") or []), "coverage": b.get("coverage"), "world": r.get("world"),
                        "kind": "fs" if src == "FS" else "memory"})
    if fs_blocks:
        for lv in LEVELS:
            b = fs_blocks.get(lv)
            if isinstance(b, dict):
                out.append({"id": f"fs_blocks.json#{lv}", "item": None, "level": lv, "lang": None, "source": "FS",
                            "text": b.get("text") or "", "ids": [], "coverage": None, "kind": "fs"})
    return out


def budgets_of(params):
    r = (params.get("retrieval") or {})
    a = r.get("a12") or {}
    B = a.get("B")
    if r.get("retention") != "a12" or not isinstance(B, dict) or not all(isinstance(B.get(lv), int) and B[lv] > 0 for lv in LEVELS):
        return None
    return {lv: int(B[lv]) for lv in LEVELS}


def token_gate(rows, counter, budgets):
    """Real-token gate: for every non-empty block, the COMPLETE `header + block` counted by each tokenizer in `counter` must be
    <= the level budget; both tokenizers must be present; the EN and FA block of one item and source must hold identical record ids."""
    from planners.contract import EXPERIENCE_HEADER, MEMORY_HEADER
    reasons = []
    toks = sorted(getattr(counter, "tokenizers", []) or [])
    if toks != ["anthropic", "gemma"]:
        reasons.append(f"both tokenizers are required, counter has {toks}")
    if budgets is None:
        reasons.append("no A12 level budgets in params.json (retrieval.retention != 'a12' or retrieval.a12.B missing)")
    if not any(r["text"] for r in rows):
        reasons.append("no non-empty block to check (a vacuous pass is not a pass)")
    out, by_lang, n_empty, worst = [], {}, 0, {lv: 0 for lv in LEVELS}
    for r in rows:
        if not r["text"]:
            n_empty += 1
            continue
        hdr = EXPERIENCE_HEADER if r["kind"] == "fs" else MEMORY_HEADER
        try:
            c = counter.counts(hdr + "\n" + r["text"])
        except KeyError:
            reasons.append(f"{r['id']}: text not in the token cache")
            continue
        B = (budgets or {}).get(r["level"])
        ok = B is not None and all(v <= B for v in c.values())
        if not ok:
            reasons.append(f"{r['id']}: counts {c} exceed budget {B}")
        worst[r["level"]] = max(worst.get(r["level"], 0), max(c.values()))
        out.append({"id": r["id"], "level": r["level"], "lang": r["lang"], "source": r["source"], "counts": c, "budget": B, "ok": ok,
                    "text_sha": C.sha256_text(r["text"])})
        if r["item"] and r["lang"] in LANGS and r["kind"] == "memory":
            by_lang.setdefault((r["item"], r.get("world"), r["source"]), {})[r["lang"]] = r["ids"]
    n_id = 0
    for key, d in sorted(by_lang.items()):
        if len(d) == 2:
            n_id += 1
            if d["en"] != d["fa"]:
                reasons.append(f"{key[0]}/{key[1]}/{key[2]}: record ids differ between EN and FA")
    return {"ok": not reasons, "reasons": reasons[:40], "n_blocks": len(out), "n_empty": n_empty, "n_id_pairs": n_id, "max_by_level": worst,
            "budgets": budgets, "tokenizers": toks, "rows": out}


# ================================================================================================ the gates
def read_evidence(root, gid):
    """Evidence file of an evidence-only gate. The verdict is the file's own `pass`, which must be a real True; synthetic or
    key-bearing evidence, a wrong gate id or a missing file fails the gate."""
    p = os.path.join(matrix_gates_dir(root), f"{gid}.json")
    if not os.path.exists(p):
        return gate(False, reason=f"no evidence file docs/final-test/matrix/gates/{gid}.json", sha=None)
    try:
        d = _read_json(p)
    except ValueError:
        return gate(False, reason="evidence file unreadable", sha=_sha_file(p))
    raw = open(p, encoding="utf-8").read()
    why = []
    if d.get("schema") != GATE_SCHEMA:
        why.append("schema")
    if d.get("gate") != gid:
        why.append("gate id differs")
    if d.get("pass") is not True:
        why.append("pass is not true")
    if d.get("synthetic"):
        why.append("synthetic evidence")
    if not d.get("evidence"):
        why.append("empty evidence")
    if C.key_leak(raw):
        why.append("key-like text in evidence")
    return gate(not why, reason=why or None, sha=C.sha256_file(p), source=d.get("source"), created=d.get("created"))


def gate_families(root):
    st = C.families_status(root)
    p = C.paths(root)["families"]
    return gate(st["ok"], reasons=st["reasons"], sha=_sha_file(p))


def gate_tolerances(root):
    """12.4 (replaced by A14) + A14 validation: reset_gate's tolerance file must be calibrated AND validated 24/24."""
    p = C.paths(root)["tolerances"]
    try:
        sys.path.insert(0, os.path.join(root, "sim", "final_test"))
        import reset_gate as RG
        d = RG.load_tolerances(p, require="validated")
        return gate(True, status=d.get("status"), tau_pos_m=d.get("tau_pos_m"), tau_rot_deg=d.get("tau_rot_deg"),
                    ceilings=d.get("ceilings"), validation=d.get("validation"), sha=_sha_file(p))
    except Exception as e:                                       # noqa: BLE001  (missing / unset / failed / edited file)
        return gate(False, reason=f"{type(e).__name__}: {str(e).replace(p, 'reset_tolerances.json')}"[:300], sha=_sha_file(p))


def gate_ollama(info):
    d = (info or {}).get("digest") or ""
    ok = bool(d) and d.replace("sha256:", "").startswith(OLLAMA_DIGEST_PREFIX) and not info.get("error")
    return gate(ok, digest=d or None, expected_prefix=OLLAMA_DIGEST_PREFIX, version=(info or {}).get("version"),
                parent_model=(info or {}).get("parent_model"), error=(info or {}).get("error"), model=(info or {}).get("model"))


def gate_one_renderer(root, code_files):
    """12.11: the renderer module declares the frozen prompt version, and no other frozen module re-declares the memory header."""
    from planners import contract as PC
    bad = []
    for rel in code_files:
        if rel.endswith("planners/contract.py") or "selftest" in rel:
            continue
        try:
            txt = open(_abs(root, rel), encoding="utf-8").read()
        except OSError:
            continue
        if PC.MEMORY_HEADER[:24] in txt or PC.EXPERIENCE_HEADER[:24] in txt:
            bad.append(rel)
    return gate(PC.PROMPT_VERSION == "ft-v1" and not bad, prompt_version=PC.PROMPT_VERSION, redeclared_in=bad)


def detector_types(root, weights_path):
    """The class list of the detector weights. Reads the .pt when ultralytics is importable; otherwise the recorded
    `detector_types.json` (written in WSL) is accepted ONLY if it names this exact weights sha."""
    wsha = _sha_file(weights_path)
    try:
        from ultralytics import YOLO                              # noqa: WPS433
        names = list(dict(YOLO(weights_path).names).values())
        return {"types": sorted(set(names)), "from": "weights", "weights_sha": wsha}
    except Exception:                                            # noqa: BLE001  (not installed / unreadable)
        pass
    p = detector_types_path(root)
    if os.path.exists(p):
        d = _read_json(p)
        if d.get("weights_sha") == wsha and isinstance(d.get("types"), list):
            return {"types": sorted(set(d["types"])), "from": "detector_types.json", "weights_sha": wsha}
    return {"types": None, "from": None, "weights_sha": wsha}


def gate_types(root, weights_path, contract=None):
    from planners import contract as PC
    t = detector_types(root, weights_path)
    types = t["types"]
    if not types:
        return gate(False, reason="class list unreadable: ultralytics missing and no detector_types.json for these weights", **{"weights_sha": t["weights_sha"]}), None
    sysx = PC.PlannerContract(types).system_text()
    ok = len(types) == EXPECTED_TYPES and ", ".join(sorted(types)) in sysx
    return gate(ok, n=len(types), expected=EXPECTED_TYPES, source=t["from"], types_sha=C.sha256_json(sorted(types)),
                system_text_sha=C.sha256_text(sysx)), types


def gate_namespaces(root):
    """12.13: three isolated namespaces; the test namespace starts with an empty cache (no reply reused from pilot / practice)."""
    cache = os.path.join(C.ns_work("test-v1", root), "cache")
    n_cached = sum(len(fs) for _, _, fs in os.walk(cache)) if os.path.isdir(cache) else 0
    # K23 (equivalence proof 2026-10-18): the replication kitchens' namespaces (scene != FloorPlan1) are isolated by their own
    # per-kitchen freeze (k23_ns.replication_freeze_status); this Kitchen 1 gate judges the Kitchen 1 namespaces only and records the others
    repl = sorted(n for n in C.NAMESPACES if C.ns_scene(n) != "FloorPlan1")
    names = sorted(n for n in C.NAMESPACES if n not in repl)
    # rehearsal-v2 (PF-safety-test, the plan-first dev test) is a second DEV rehearsal namespace: it may exist, but only on dev items and never as the test namespace
    base = ["practice-v1", "rehearsal-v1", "test-v1"]
    # C4: practice-v2 (fresh ft-v2 practice) and rehearsal-v3 (the ft-v2 runner rehearsal) may exist too: non-test items only, never the test namespace
    optional = {"rehearsal-v2", "practice-v2", "rehearsal-v3", "acceptance-v2", "acceptance-v2-a2", "freeze-gates-ft2"}     # + FREEZE-GATES-FT2 (dev, Gemma only)     # ACC-RUNNER: the registered acceptance roster only, never test (-a2 = ACC-ATTEMPT2, the one correction cycle)
    extra_ok = set(base) <= set(names) and set(names) - set(base) <= optional and all(C.NAMESPACES[n]["items"] != "test" for n in names if n != "test-v1")
    started_empty = None                                           # A37 (mid-test re-freeze): the cache STARTED empty under a signed predecessor
    if n_cached:                                                   # freeze (archived in freeze_history/, 12.13 PASS + Arash's check); later files
        hist = os.path.join(root or C.paths(None)["root"], "docs", "final-test", "freeze_history")   # are this test's own replies
        for fn in sorted(os.listdir(hist)) if os.path.isdir(hist) else ():
            try:
                h = _read_json(os.path.join(hist, fn))
            except (OSError, ValueError):
                continue
            g = (h.get("gates") or {}).get("12.13") or {}
            if h.get("sha256") == C.self_sha(h) and (h.get("arash_check") or {}).get("by") == "Arash" and g.get("pass") is True \
                    and (g.get("evidence") or {}).get("test_cache_files") == 0:
                started_empty = fn
    ok = extra_ok and (n_cached == 0 or started_empty is not None)
    return gate(ok, namespaces=names, replication_namespaces=repl, test_cache_files=n_cached, cache_started_empty_under=started_empty,
                ceilings=C.NAMESPACES["test-v1"]["ceilings"])


def snapshots_info(root):
    """Per snapshot: file hashes + manifest facts; nesting check (E_frozen non-decreasing in K within an order)."""
    out, reasons = {}, []
    base = C.paths(root)["snapshots"]
    for o in SNAP_ORDERS:
        prev = -1
        for k in SNAP_KS:
            d = os.path.join(base, f"o{o}", f"k{k}")
            key = f"o{o}k{k}"
            files = {f: _sha_file(os.path.join(d, f)) for f in SNAP_FILES}
            if not all(files.values()):
                reasons.append(f"{key}: missing {[f for f, v in files.items() if not v]}")
                out[key] = {"files": files}
                continue
            m = _read_json(os.path.join(d, "manifest.json"))
            ef = m.get("E_frozen")
            pm = _read_json(os.path.join(d, "practice_manifest.json")) if os.path.exists(os.path.join(d, "practice_manifest.json")) else {}
            out[key] = {"files": files, "state_hash": m.get("state_hash"), "E_frozen": ef, "file_sha256": m.get("file_sha256"),
                        "integrity": m.get("integrity"), "channels": pm.get("channels")}
            if m.get("integrity"):
                reasons.append(f"{key}: manifest lists integrity problems")
            if isinstance(ef, int):
                if ef < prev:
                    reasons.append(f"{key}: E_frozen {ef} < the smaller snapshot's {prev} (not nested)")
                prev = ef
            else:
                reasons.append(f"{key}: no E_frozen")
    return out, reasons


# ================================================================================================ build
def build(root=None, *, counter=None, probes=None, blocks_ns=None, allow_test=False, stored=None, reloc_seed=None, now=None,
          scene: str = "FloorPlan1"):
    """The freeze dict (without arash_check, which is carried over from `stored` when given). Deterministic given the disk and
    `probes`. counter=None + stored given: the token section is re-verified from the stored rows file instead of recounted."""
    P = C.paths(root)
    root = P["root"]
    from planners import ARMS as PLANNER_ARMS
    from planners import contract as PC
    probes = probes or {}
    fz = {"schema": FREEZE_SCHEMA, "spec": C.SPEC_VERSION, "created": now or time.strftime("%Y-%m-%dT%H:%M:%S")}
    gates = {}

    # ---- the frozen files
    fz["prereg_sha256"] = _sha_file(P["prereg"])
    fz["split_sha256"] = C.load_manifest(root)["sha256"]                       # verified against the registered constant
    fz["dataset_sha256"] = _sha_file(P["dataset"])
    fz["families_approved_sha256"] = _sha_file(P["families"])
    params_path = os.path.join(root, "sim", "step5", "params.json")
    params = _read_json(params_path)
    fz["params_sha256"] = _sha_file(params_path)
    fz["a12"] = {"retention": (params.get("retrieval") or {}).get("retention"), "a12": (params.get("retrieval") or {}).get("a12")}
    fz["reset_tolerances_sha256"] = _sha_file(P["tolerances"])
    lex_path = os.path.join(root, "sim", "step5", "lexicon", "concepts.json")      # W4a: the Persian/English concept lexicon the memory parser reads
    try:
        import ltm_concepts as _lc
        _pv = getattr(_lc, "PARSER_VERSION", None)
    except Exception:                                                              # noqa: BLE001
        _pv = None
    fz["lexicon"] = {"path": _rel(root, lex_path), "sha256": _sha_file(lex_path), "parser_version": _pv}
    fz["fs_blocks_sha256"] = _sha_file(P["fs_blocks"])
    # TREAT-FREEZE / A24.1-T item 4 (iii): after the flip, the test-v1 treatment identity (deterministic; --check compares it).
    import treatment as _T
    if _T.TEST_V1_TREATMENT == _T.FT2_VERSION:
        fz["treatment"] = _T.identity("test-v1", root)
    fz["caps"] = dict(C.CAPS)
    fz["seed_policy"] = C.SEED_POLICY
    fz["master_seed"] = C.MASTER_SEED
    fz["reloc_seed"] = int(reloc_seed) if reloc_seed is not None else (stored or {}).get("reloc_seed") or C.rng("reloc_seed").randrange(1, 2 ** 31 - 1)
    fz["namespaces"] = {k: v["ceilings"] for k, v in sorted(C.NAMESPACES.items())}
    fz["arms"] = {a: dict(C.arm_info(a)) for a in C.MAIN_ARMS}                 # A21: the main roster incl. the gpt-6-sol arms X0 / X+M
    fz.update(freeze_entries())                                                                                       # A21 B + A21 A
    ck = C.unit_key_selfcheck()                                                                                       # A21 D
    fz["dedup"] = {"spec": C.UNIT_KEY_SPEC, "sha256": C.sha256_json(C.UNIT_KEY_SPEC), "golden_key": ck["golden"], "golden_ok": ck["ok"]}
    try:
        import analyze as _an
        fz["analysis_plan"] = {"sha256": _an.analysis_plan_sha(), "plan": _an.ANALYSIS_PLAN}                          # A21 F
    except Exception as ex:                                                                                           # noqa: BLE001
        fz["analysis_plan"] = {"sha256": None, "error": type(ex).__name__}
    pw = os.path.join(root, "docs", "final-test", "matrix", "power_secondary.json")
    fz["power_secondary_sha256"] = _sha_file(pw) if os.path.isfile(pw) else None
    fp_ = os.path.join(root, "docs", "final-test", "matrix", "final_projection.json")
    fz["final_projection_sha256"] = _sha_file(fp_) if os.path.isfile(fp_) else None
    xp = os.path.join(root, "docs", "final-test", "matrix", "x_isolation.json")
    try:
        import planners.codex_provider as _cp
        _x = _cp.CodexCliPlanner(C.CODEX_MAIN, which=lambda c: c)
        fz["x_arms"] = {"model": C.CODEX_MAIN, "arms": list(C.X_MAIN_ARMS), "reasoning_effort": _x.reasoning_effort, "sandbox": _x.sandbox, "hardened_features": list(_cp.HARDEN_FEATURES),
                        "cli_flags_sha256": C.sha256_json(_x.decoding()["cli_flags"]), "session_log_filter": _cp.SESSION_LOG,
                        "isolation_report_sha256": _sha_file(xp) if os.path.isfile(xp) else None,
                        "server_side_model": "not observable through the CLI (disclosed, A21 E)"}
    except Exception as ex:                                                                                           # noqa: BLE001
        fz["x_arms"] = {"error": type(ex).__name__}

    # ---- prompt
    fz["prompt_version"] = PC.PROMPT_VERSION
    fz["system_template_sha"] = C.sha256_text(PC.SYSTEM)
    fz["user_template_sha"] = C.sha256_text(PC.USER)
    fz["plan_schema_sha"] = C.sha256_json(PC.PLAN_SCHEMA)
    fz["headers_sha"] = C.sha256_json([PC.MEMORY_HEADER, PC.EXPERIENCE_HEADER])

    # ---- weights, types, map
    wdir = os.path.join(root, "sim", "weights", "v3")
    primary, specialist = os.path.join(wdir, f"{scene}.yolo26s.pt"), os.path.join(wdir, f"{scene}.amb26m.pt")   # K23: per scene
    fz["detector_weights"] = {"primary": {"path": _rel(root, primary), "sha256": _sha_file(primary)},
                              "specialist": {"path": _rel(root, specialist), "sha256": _sha_file(specialist)}}
    gates["12.12"], types = gate_types(root, primary)
    fz["detector_types"] = {"n": len(types) if types else None, "sha256": C.sha256_json(sorted(types)) if types else None}
    fz["system_text_sha"] = C.sha256_text(PC.PlannerContract(types).system_text()) if types else None
    map_path = os.path.join(root, "sim", "step4", f"{scene}.map.json")
    fz["map"] = {"path": _rel(root, map_path), "sha256": _sha_file(map_path)}
    fz["map_sha"] = fz["map"]["sha256"]

    # ---- code
    entries = list(ENTRY_REQUIRED) + [e for e in ENTRY_OPTIONAL if os.path.isfile(_abs(root, e))]
    files, missing, errors = import_closure(root, entries)
    fz["code_entries"] = entries
    fz["code"] = {rel: _sha_file(_abs(root, rel)) for rel in files}
    man = C.load_manifest(root)
    inc = man.get("inclusion") or {}
    fz["inclusion_manifest"] = {"path": inc.get("path"), "sha256": inc.get("sha256"), "archived": inc.get("archived")} if inc else None
    # the dataset on disk is the one the split was cut from, or (A26, K-B2b) the successor its effective inclusion manifest names
    ds_ok = fz["dataset_sha256"] == (inc.get("dataset_sha256") or man.get("dataset_sha256"))
    gates["12.1"] = gate(ds_ok and not missing and not errors and bool(fz["code"]) and None not in fz["code"].values() and
                         all(fz[k] for k in ("prereg_sha256", "dataset_sha256", "params_sha256", "map_sha", "system_text_sha", "plan_schema_sha")) and
                         all(v["sha256"] for v in fz["detector_weights"].values()),
                         dataset_matches_split=ds_ok, n_code_files=len(fz["code"]), missing_entries=missing, parse_errors=errors,
                         unhashed=[k for k, v in fz["code"].items() if v is None])
    gates["12.11"] = gate_one_renderer(root, files)

    # ---- models, serving
    fz["models"] = {"planners": {k: list(v) for k, v in sorted(PLANNER_ARMS.items())}, "codex": list(C.CODEX_MODELS), "gemma_tag": OLLAMA_MODEL}
    oll = probes.get("ollama") or (stored or {}).get("ollama") or {}
    fz["ollama"] = {k: oll.get(k) for k in ("model", "digest", "version", "parent_model")}
    cdx = probes.get("codex") or {"version": (stored or {}).get("codex_cli_version")}
    fz["codex_cli_version"] = cdx.get("version")
    gates["12.9"] = gate_ollama(oll)
    if not fz["codex_cli_version"]:
        gates["12.9"] = gate(False, reason="codex CLI version not recorded", **gates["12.9"]["evidence"])

    # ---- snapshots
    snaps, why = snapshots_info(root)
    fz["snapshots"] = dict(snaps, **{"S*": dict(snaps.get("o0k10") or {}, source="o0k10")})
    s_state = (snaps.get("o0k10") or {}).get("state_hash")
    fz["channels"] = (snaps.get("o0k10") or {}).get("channels")      # the memory channels S* was built with (capture refuses others, A13)
    gates["snapshot"] = gate(not why and bool(s_state), reasons=why[:20], S_star_state_hash=s_state, n=len(snaps))
    # ---- files with their own verdict
    gates["12.8"] = gate_families(root)
    gates["12.4"] = gate_tolerances(root)
    gates["A14"] = dict(gates["12.4"])
    gates["12.13"] = gate_namespaces(root)
    for gid in EVIDENCE_GATES:
        gates[gid] = read_evidence(root, gid)
    fz["evidence_sha"] = {gid: gates[gid]["evidence"].get("sha") for gid in EVIDENCE_GATES}

    # ---- blocks: A12 coverage + real tokens (pre-freeze: dev presentations, never test)
    fs_blocks = _read_json(P["fs_blocks"]) if os.path.exists(P["fs_blocks"]) else None
    if fs_blocks is not None and fs_blocks.get("sha256") not in (None, C.self_sha(fs_blocks)):
        fs_blocks = None
    cap_rows = []
    if blocks_ns:
        if blocks_ns == "test-v1" and not allow_test:
            raise C.TestItemRefused("test-v1 blocks need --allow-test")
        cap_rows = load_capture_rows(root, blocks_ns, allow_test=allow_test)
    rows = block_rows(cap_rows, fs_blocks)
    fz["blocks_ns"] = blocks_ns
    fz["blocks_sha"] = C.sha256_json([{k: r[k] for k in ("id", "level", "lang", "source", "text", "ids")} for r in rows])
    budgets = budgets_of(params)
    if counter is None and stored is not None:
        tr = _read_json(tokens_rows_path(root)) if os.path.exists(tokens_rows_path(root)) else None
        st = (stored.get("tokens") or {})
        same = tr is not None and C.sha256_json(tr) == st.get("rows_sha256") and st.get("blocks_sha") == fz["blocks_sha"]
        tg = {"ok": bool(same and st.get("ok")), "reasons": [] if same else ["stored token rows changed or blocks changed"],
              "n_blocks": st.get("n_blocks"), "n_empty": st.get("n_empty"), "n_id_pairs": st.get("n_id_pairs"),
              "max_by_level": st.get("max_by_level"), "budgets": budgets, "tokenizers": st.get("tokenizers"), "rows": (tr or {}).get("rows")}
        counted = st.get("counted")
    elif counter is None:
        tg = {"ok": False, "reasons": ["no token counter given (needs Ollama + the free count_tokens)"], "rows": [], "n_blocks": 0,
              "n_empty": 0, "n_id_pairs": 0, "max_by_level": None, "budgets": budgets, "tokenizers": []}
        counted = "none"
    else:
        tg = token_gate(rows, counter, budgets)
        counted = "counter"
    rows_doc = {"schema": "ftm.freeze-tokens/1", "rows": tg.get("rows")}
    fz["tokens"] = {"ok": tg["ok"], "counted": counted, "n_blocks": tg["n_blocks"], "n_empty": tg["n_empty"], "n_id_pairs": tg["n_id_pairs"],
                    "max_by_level": tg["max_by_level"], "budgets": tg["budgets"], "tokenizers": tg["tokenizers"],
                    "rows_sha256": C.sha256_json(rows_doc), "blocks_sha": fz["blocks_sha"]}
    gates["tokens"] = gate(tg["ok"], reasons=tg["reasons"][:10], **{k: fz["tokens"][k] for k in ("n_blocks", "n_empty", "max_by_level", "budgets", "tokenizers")})
    cov = [r["coverage"] for r in rows if r["source"] == "S*" and r["kind"] == "memory"]
    cov_bad = [r["id"] for r in rows if r["source"] == "S*" and r["kind"] == "memory" and not (isinstance(r["coverage"], dict) and r["coverage"].get("ok") is True)]
    gates["A12"] = gate(budgets is not None and bool(cov) and not cov_bad, budgets=budgets, n_S_star_blocks=len(cov), failing=cov_bad[:20],
                        reason=None if budgets is not None else "params.json has no retention a12 / a12.B")

    gates["12.6"] = gate(all(gates[g]["pass"] for g in ("snapshot", "G-iso", "leakage")), parts={g: gates[g]["pass"] for g in ("snapshot", "G-iso", "leakage")})
    fz["gates"] = {g: gates[g] for g in GATE_ORDER if g in gates}
    # the gate evidence blobs hold only deterministic facts, so a rebuild equals the stored file
    fz["all_gates_pass"] = all(v["pass"] for v in fz["gates"].values()) and set(fz["gates"]) == set(GATE_ORDER)
    fz["arash_check"] = (stored or {}).get("arash_check")
    fz["_rows_doc"] = rows_doc
    return fz


def seal(fz):
    fz = {k: v for k, v in fz.items() if k != "_rows_doc"}
    fz["sha256"] = C.self_sha(fz)
    return fz


def _atomic_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def write(root=None, *, counter=None, probes=None, blocks_ns=None, allow_test=False, supersede=False, reloc_seed=None, now=None):
    """--write. Refuses to overwrite a freeze that carries Arash's check unless supersede=True (the old file is then archived)."""
    P = C.paths(root)
    root = P["root"]
    if blocks_ns and blocks_ns == "test-v1":
        raise C.TestItemRefused("--write uses non-test blocks only (the pre-freeze set); test blocks are checked after the freeze with --check")
    old = None
    if os.path.exists(P["freeze"]):
        try:
            old = _read_json(P["freeze"])
        except ValueError:
            old = None
        if old and old.get("arash_check") and not supersede:
            raise PermissionError("freeze.json already carries Arash's check; a re-freeze needs --supersede (the old file is archived)")
        if old:
            arc = os.path.join(root, "sim", "final_test", "_checkpoints", "M3-freeze")
            os.makedirs(arc, exist_ok=True)
            shutil.copyfile(P["freeze"], os.path.join(arc, f"freeze.{time.strftime('%Y%m%dT%H%M%S')}.json"))
    fz = build(root, counter=counter, probes=probes, blocks_ns=blocks_ns, allow_test=allow_test,
               stored=None if supersede else old, reloc_seed=reloc_seed, now=now)
    fz["arash_check"] = None                                      # a new freeze always needs a new check
    rows_doc = fz["_rows_doc"]
    if counter is not None:
        _atomic_write(tokens_rows_path(root), rows_doc)
        try:
            counter.save()
        except Exception:                                         # noqa: BLE001
            pass
        fz["tokens"]["rows_sha256"] = C.sha256_json(rows_doc)
    out = seal(fz)
    if C.key_leak(json.dumps(out)):
        raise RuntimeError("key-like text in freeze.json; refusing to write")
    _atomic_write(P["freeze"], out)
    return out


def check(root=None, *, probes=None, blocks_ns=None, allow_test=False, counter=None):
    """--check. Rebuilds from the disk and compares with the stored file. Returns {ok, reasons, diff}. `ok` also needs the
    common.freeze_status test (Arash's check) only when require_arash is used by the caller."""
    P = C.paths(root)
    root = P["root"]
    try:
        stored = _read_json(P["freeze"])
    except (OSError, ValueError) as e:
        return {"ok": False, "reasons": [f"freeze.json missing or unreadable: {type(e).__name__}"], "diff": []}
    reasons = []
    if stored.get("schema") != FREEZE_SCHEMA:
        reasons.append("schema")
    if stored.get("sha256") != C.self_sha(stored):
        reasons.append("freeze.json self-sha mismatch (edited after writing)")
    ns = blocks_ns if blocks_ns is not None else stored.get("blocks_ns")
    try:
        fz = build(root, counter=counter, probes=probes, blocks_ns=ns, allow_test=allow_test, stored=stored, now=stored.get("created"))
    except C.TestItemRefused as e:
        return {"ok": False, "reasons": reasons + [f"refused: {e}"], "diff": []}
    diff = sorted(k for k in set(stored) | set(fz) if k not in VOLATILE and k != "_rows_doc" and stored.get(k) != fz.get(k))
    bad = sorted(g for g, v in fz["gates"].items() if not v["pass"])
    if diff:
        reasons.append(f"differs from the disk: {diff}")
    if bad:
        reasons.append(f"gates not passed: {bad}")
    if not (stored.get("arash_check") or {}).get("by"):
        reasons.append("Arash's freeze check not recorded")
    return {"ok": not reasons, "reasons": reasons, "diff": diff, "failed_gates": bad}


def record_arash_check(root=None, *, by, date, said, probes=None):
    """--record-arash-check: ONLY on Arash's explicit instruction (`said` = his words, required). Refuses unless every gate passes
    on a fresh rebuild and the file is intact; fills arash_check and re-seals."""
    P = C.paths(root)
    root = P["root"]
    if by != "Arash" or not date or not (said or "").strip():
        raise PermissionError("arash_check needs by='Arash', a date and his words (--arash-said)")
    stored = _read_json(P["freeze"])
    if stored.get("sha256") != C.self_sha(stored):
        raise RuntimeError("freeze.json self-sha mismatch; refusing")
    if stored.get("arash_check"):
        raise PermissionError("already recorded")
    r = check(root, probes=probes)
    pending = [x for x in r["reasons"] if "Arash's freeze check" not in x]
    if pending:
        raise RuntimeError("gates / files are not clean, no check recorded: " + "; ".join(pending)[:500])
    if not stored.get("all_gates_pass"):
        raise RuntimeError("not every gate passed")
    stored["arash_check"] = {"by": "Arash", "date": date, "said": said.strip()}
    out = seal({k: v for k, v in stored.items() if k != "sha256"})
    _atomic_write(P["freeze"], out)
    return out


def record_treatment(root=None):
    """--record-treatment (TREAT-FREEZE; A24.1-T item 4 (ii)-(iii)): in THIS (fresh) process, after the one-constant flip, write ONLY the
    test-v1 treatment identity into the existing freeze.json under "treatment" and mark the file freeze_status = treatment.TREATMENT_ONLY
    (Astra TREAT-FREEZE 4: never a validated freeze; common.freeze_status and treatment.entry_problems refuse that status, and the full
    --write later drops it and must reproduce the same identity). The old file is archived; refuses a file that carries Arash's check."""
    import treatment as _T
    P = C.paths(root)
    root = P["root"]
    if _T.TEST_V1_TREATMENT != _T.FT2_VERSION:
        raise RuntimeError("TEST_V1_TREATMENT is not ft-v2: flip the one constant first (A24.1-T item 4 (i))")
    stored = _read_json(P["freeze"])
    if stored.get("arash_check"):
        raise PermissionError("freeze.json carries Arash's check; a treatment change there is a re-freeze (--write --supersede)")
    if stored.get("sha256") != C.self_sha(stored):
        raise RuntimeError("freeze.json self-sha mismatch; refusing")
    ident = _T.identity("test-v1", root)                         # fails closed on any pin difference
    arc = os.path.join(root, "sim", "final_test", "_checkpoints", "M3-freeze")
    os.makedirs(arc, exist_ok=True)
    shutil.copyfile(P["freeze"], os.path.join(arc, f"freeze.{time.strftime('%Y%m%dT%H%M%S')}.pre-treatment.json"))
    out = {k: v for k, v in stored.items() if k != "sha256"}
    out["treatment"] = ident
    out["freeze_status"] = _T.TREATMENT_ONLY
    out["freeze_status_note"] = ("treatment record only (TREAT-FREEZE " + time.strftime("%Y-%m-%d") + "); every other field and gate is from "
                                 + str(stored.get("created")) + " and STALE; not a validated freeze; regenerate with --write")
    out = seal(out)
    if C.key_leak(json.dumps(out)):
        raise RuntimeError("key-like text in freeze.json; refusing to write")
    _atomic_write(P["freeze"], out)
    return out


def freeze_entries() -> dict:
    """The two A21 rule entries of every freeze (cost_guard = A21 B, blocked_run_rule = A21 A), with their sha256 scopes. ONE builder, used by
    build() (so by the ft-v2 freeze too) and by --record-entries (A24.1-C owed item 'freeze-entries')."""
    return {"cost_guard": {"spec": C.COST_GUARD_SPEC, "sha256": C.sha256_json(C.COST_GUARD_SPEC)},
            "blocked_run_rule": {"spec": C.BLOCKED_RULE, "classes": C.RUN_CLASSES, "infra_flag_rate": C.INFRA_FLAG_RATE,
                                 "sha256": C.sha256_json([C.BLOCKED_RULE, C.RUN_CLASSES, C.INFRA_FLAG_RATE])}}


def record_entries(root=None):
    """--record-entries (RUNNER-FINAL; A24.1-C step 7 / owed item 'freeze-entries'): BEFORE the first acceptance call, write ONLY the
    cost_guard and blocked_run_rule entries (freeze_entries()) into the existing freeze.json. The file stays freeze_status =
    treatment.TREATMENT_ONLY (never a validated freeze; the full --write after acceptance rebuilds the same two entries and --check compares
    them); its recorded treatment identity must be the one computed now; the old file is archived; refuses a file that carries Arash's
    check or fails its self-sha."""
    import treatment as _T
    P = C.paths(root)
    root = P["root"]
    stored = _read_json(P["freeze"])
    if stored.get("arash_check"):
        raise PermissionError("freeze.json carries Arash's check; new entries there are a re-freeze (--write --supersede)")
    if stored.get("sha256") != C.self_sha(stored):
        raise RuntimeError("freeze.json self-sha mismatch; refusing")
    if stored.get("freeze_status") != _T.TREATMENT_ONLY:
        raise RuntimeError(f"freeze.json status {stored.get('freeze_status')!r}: --record-entries only adds to the treatment-only record")
    if (stored.get("treatment") or {}).get("sha") != _T.identity("test-v1", root)["sha"]:
        raise RuntimeError("freeze.json's treatment identity is not the one computed now; refusing")
    arc = os.path.join(root, "sim", "final_test", "_checkpoints", "M3-freeze")
    os.makedirs(arc, exist_ok=True)
    shutil.copyfile(P["freeze"], os.path.join(arc, f"freeze.{time.strftime('%Y%m%dT%H%M%S')}.pre-entries.json"))
    out = {k: v for k, v in stored.items() if k != "sha256"}
    out.update(freeze_entries())
    out["entries_recorded"] = {"keys": ["cost_guard", "blocked_run_rule"], "date": time.strftime("%Y-%m-%d"), "stage": "RUNNER-FINAL",
                               "rule": "A24.1-C step 7: frozen before the first acceptance call; the treatment-only status is unchanged"}
    out["freeze_status_note"] = (str(stored.get("freeze_status_note") or "") + " | + cost_guard / blocked_run_rule recorded "
                                 + time.strftime("%Y-%m-%d") + " (A24.1-C, before the first acceptance call; still not a validated freeze)")
    out = seal(out)
    if C.key_leak(json.dumps(out)):
        raise RuntimeError("key-like text in freeze.json; refusing to write")
    _atomic_write(P["freeze"], out)
    return out


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--record-arash-check", action="store_true")
    g.add_argument("--record-treatment", action="store_true", help="TREAT-FREEZE: write only the test-v1 treatment identity (after the flip)")
    g.add_argument("--record-entries", action="store_true", help="A24.1-C: write only the cost_guard + blocked_run_rule entries (treatment-only file)")
    ap.add_argument("--blocks-from", default=None, help="namespace whose capture.jsonl supplies the S*/FS blocks (rehearsal-v1; test-v1 only with --check --allow-test)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--reloc-seed", type=int, default=None)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--live", action="store_true", help="--check: re-probe Ollama / codex instead of trusting the stored versions")
    ap.add_argument("--no-count", action="store_true", help="--write: skip the live token counter (the tokens gate then fails)")
    ap.add_argument("--by"), ap.add_argument("--date"), ap.add_argument("--arash-said")
    ap.add_argument("--legacy-ft1", action="store_true", help="use the ft-v1 builder (history only; refused while test-v1 runs ft-v2)")
    a = ap.parse_args(argv)
    root = C.ROOT
    import treatment as _T
    ft2 = _T.TEST_V1_TREATMENT == _T.FT2_VERSION                 # ACC-RUNNER: test-v1 runs ft-v2 -> the ft-v2 builder (freeze_ft2.py, F1-F9)
    if a.legacy_ft1 and ft2 and (a.write or a.record_arash_check):
        raise SystemExit("--legacy-ft1 --write / --record-arash-check is refused: test-v1 runs ft-v2 (a freeze on the ft-v1 builder would rest on stale evidence)")
    if a.record_entries:
        out = record_entries(root)
        print("entries recorded: cost_guard", out["cost_guard"]["sha256"][:16], "blocked_run_rule", out["blocked_run_rule"]["sha256"][:16],
              "status", out["freeze_status"], "freeze self-sha", out["sha256"][:16])
        return 0
    if ft2 and not a.legacy_ft1 and not a.record_treatment:
        import freeze_ft2 as FT2
        if a.record_arash_check:
            out = FT2.record_arash_check_ft2(root, by=a.by, date=a.date, said=a.arash_said)
            print("recorded:", out["arash_check"]["date"], out["sha256"][:16])
            return 0
        probes = {"ollama": probe_ollama(a.ollama), "codex": probe_codex()} if (a.write or a.live) else None
        if a.write:
            out = FT2.write_ft2(root, probes=probes, supersede=a.supersede)
            print(f"freeze.json (ft-v2) written sha256={out['sha256'][:16]} code={len(out['code'])} files; status {out['status_summary']}")
            return 0 if out["all_gates_pass"] else 2
        r = FT2.check_ft2(root, probes=probes)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 2
    if a.record_arash_check:
        out = record_arash_check(root, by=a.by, date=a.date, said=a.arash_said)
        print("recorded:", out["arash_check"]["date"], out["sha256"][:16])
        return 0
    if a.record_treatment:
        out = record_treatment(root)
        print("treatment recorded:", out["treatment"]["sha"], "profile", out["treatment"]["profile"], "status", out["freeze_status"],
              "freeze self-sha", out["sha256"][:16])
        return 0
    if a.write:
        probes = {"ollama": probe_ollama(a.ollama), "codex": probe_codex()}
        counter = None if a.no_count else live_counter(root, a.ollama)
        out = write(root, counter=counter, probes=probes, blocks_ns=a.blocks_from, allow_test=a.allow_test, supersede=a.supersede,
                    reloc_seed=a.reloc_seed)
        bad = [g for g, v in out["gates"].items() if not v["pass"]]
        print(f"freeze.json written sha256={out['sha256'][:16]} code={len(out['code'])} files; gates failing: {bad or 'none'}")
        return 0 if not bad else 2
    probes = {"ollama": probe_ollama(a.ollama), "codex": probe_codex()} if a.live else None
    r = check(root, probes=probes, blocks_ns=a.blocks_from, allow_test=a.allow_test)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
