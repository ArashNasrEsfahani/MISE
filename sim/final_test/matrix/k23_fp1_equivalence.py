"""K23 FP1 equivalence harness (Astra K23-AGENT-r1 item 4; K23-FREEZE-SUPERSEDE-r1): the PREDECLARED comparison of the master
baseline and the k23-agent-side successor on Kitchen 1 development data. Protocol, decision rule and commands:
docs/final-test/k23/FP1-equivalence-plan.md. This module only prepares, collects and compares; the live phases are the frozen
matrix scripts themselves (capture.py through fg_runner.py, plan.py, fg_runner.py execute, score.py), run from each worktree.

Part A, deterministic (Windows python, offline; no simulator, no provider, no network; nothing written into either tree):
    python sim/final_test/matrix/k23_fp1_equivalence.py offline --tree D:/Personal/MP-base       --out <E>/offline_base.json
    python sim/final_test/matrix/k23_fp1_equivalence.py offline --tree D:/Personal/MP-agent-side --out <E>/offline_branch.json
    python sim/final_test/matrix/k23_fp1_equivalence.py compare-offline <E>/offline_base.json <E>/offline_branch.json --out <E>/verdict_offline.json
  For every non-test presentation of the frozen K1 captures (freeze-gates-ft2, rehearsal-v3, acceptance-v2, acceptance-v2-a2) it re-renders,
  with THIS tree's code, every memory package and FS block and every planner request from the capture's own inputs (instruction,
  post-sweep prompt summary, told holding, frozen snapshot), using a CACHE-ONLY token counter over the namespace's own token cache (a count
  that is not cached is reported, never fetched). Plus: contract system texts, canonical exports + state hashes of every frozen snapshot,
  the step5_replay corpus replayed into a fresh memory (Step-5 writes), and the seeded sample/order outputs. Byte identity is required.

Part B, live (WSL, simulator, ONE run at a time under flock /home/arash/sim.lock + hang_guard; see the plan for the exact commands):
    stage   --tree T --run B1|S1|B2|S2|BC|SC --evidence E [--expect-commit SHA]   prepare namespace freeze-gates-ft2 inside T for one run
    launch  --run R --job J -- <script.py> [args]       one WSL job of that run (k23_fp1_equiv_run.sh: sim lock, timeout, hang guard)
    collect --tree T --run R --evidence E               copy the run's journals into E/R and restore T byte for byte
    compare-live --evidence E --offline-verdict V ... --out E/verdict_live.json      fail-closed verdict EQUIVALENT / UNRESOLVED / NOT_EQUIVALENT
  Decisive runs B1, S1, B2, S2 EXECUTE the frozen 2026-10-15 capture + plans (B0) on both trees (S1 / S2: the one declared rewrite is the
  top-level treatment_id of the copied capture / plan rows, after a render proof by the successor's own code); BC / SC are independent
  captures (diagnostics). Volatile fields excluded from every comparison are declared in VOLATILE (and nothing else is excluded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

NS = "freeze-gates-ft2"
#: frozen K1 captures re-rendered in part A (dev / acceptance rosters only; a test item is never rendered)
OFFLINE_NAMESPACES = ("freeze-gates-ft2", "rehearsal-v3", "acceptance-v2", "acceptance-v2-a2")
#: the runs of part B, in their predeclared order (B0 = the frozen 2026-10-15 freeze-gates-ft2 run, read-only)
RUNS = ("B1", "S1", "B2", "S2")
RUN_TREE = {"B1": "base", "B2": "base", "BC": "base", "S1": "branch", "S2": "branch", "SC": "branch"}
EXEC_ARMS = ("G0", "G+M", "P0", "P+M")             # memory-off: G0, P0; memory-on: G+M (S* package + hook B), P+M (hook B)
#: the INDEPENDENTLY fixed roster of part B (the applicable dev_pilot items of the frozen 2026-10-15 run) and the frozen B0 totals
EXPECTED_ITEMS = ("g-l1-off-stove-knob", "g-l1-on-stove-knob", "g-l1-place-bowl-counter-top", "g-l1-place-lettuce-sink-basin",
                  "g-l2-find-pickup-lettuce", "g-l2-find-pickup-pan", "g-l2-microwave-run-egg", "g-l2-microwave-run-potato",
                  "g-l3-chilled-apple", "g-l3-tool-flip")
EXPECTED_PRES = tuple(sorted(f"canon:{i}:{lg}" for i in EXPECTED_ITEMS for lg in ("en", "fa")))
B0_TOTALS = {"presentations": 20, "requests": 84, "blocks": 64, "plans": 84}
#: fixed totals of the four frozen K1 captures re-rendered in part A (presentations, requests, blocks): Astra FP1-EQUIV-r3 1
OFFLINE_TOTALS = {"freeze-gates-ft2": (20, 84, 64), "rehearsal-v3": (56, 200, 144), "acceptance-v2": (48, 144, 96),
                  "acceptance-v2-a2": (48, 144, 96)}


def summary_complete(s, totals=None) -> bool:
    """A render summary is complete when every expected request and block rendered, all equal to the capture, with zero misses /
    errors / mismatches; with `totals` = (presentations, requests, blocks) the counts must also be exactly those."""
    if not isinstance(s, dict):
        return False
    ok = (s.get("requests") == s.get("requests_expected") and s.get("blocks") == s.get("blocks_expected") and s.get("requests", 0) > 0
          and s.get("match_capture") == s.get("requests") and s.get("differs_from_capture") == 0 and s.get("miss") == 0 and s.get("errors") == 0)
    if totals is not None:
        ok = ok and (s.get("presentations"), s.get("requests"), s.get("blocks")) == tuple(totals)
    return ok
#: declared volatile fields (wall-clock, process, path and the expected successor identity); NOTHING else is excluded
VOLATILE = {
    "*": ("ts", "wall_s", "created", "written", "written_utc", "session", "treatment_id", "s", "elapsed_s", "pid"),
    "capture": ("capture_run",),
    "exec": ("epoch",),
    "plan": ("cost_usd",),
}
#: capture fields that depend on the live post-sweep observation (world noise): compared, and every other field is compared CONDITIONALLY
#: on them being equal (equal inputs -> byte-identical blocks and requests)
OBS_FIELDS = ("summary_sha", "prompt_summary_sha", "holding", "observed_types", "full_summary_n", "world_hash", "restored_state")
DOCS_REL = ("docs", "final-test", "matrix", NS)
WORK_REL = ("sim", "final_test", "matrix", "work", NS)


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _sha_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _dir_sha(d):
    if not os.path.isdir(d):
        return "absent"
    acc = []
    for base, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(base, f)
            acc.append((os.path.relpath(p, d).replace(os.sep, "/"), _sha_file(p)))
    return _sha_json(acc)


def _rows(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _strip(rec, kind):
    drop = set(VOLATILE["*"]) | set(VOLATILE.get(kind, ()))
    if isinstance(rec, dict):
        return {k: _strip(v, kind) for k, v in rec.items() if k not in drop}
    if isinstance(rec, list):
        return [_strip(v, kind) for v in rec]
    return rec


# ===================================================================================================== part A: deterministic artefacts
def _import_tree(tree):
    for p in (os.path.join(tree, "sim", "final_test", "matrix"), os.path.join(tree, "sim")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(tree)
    import common as C
    if os.path.normcase(os.path.abspath(C.ROOT)) != os.path.normcase(os.path.abspath(tree)):
        raise SystemExit(f"imported common.py from {C.ROOT}, not from {tree} (run one tree per process)")
    return C


def offline(tree: str, live_capture: str | None = None, only=None) -> dict:
    C = _import_tree(tree)
    import treatment as T
    import honest_skills as hs
    import ltm as L
    import ltm_store as LS
    import step5_replay as R
    from final_test.c3.real_counter import RealMaxCounter
    from planners import PlannerContract, prompt_sha256
    from planners.contract import PACKAGE_HEADER

    class Miss(Exception):
        pass

    class CacheOnly(RealMaxCounter):
        """The namespace's real-token cache, read-only: a text that was never counted raises Miss (no Ollama / Anthropic call)."""
        def __init__(self, path, header):
            super().__init__(path, header, allow_missing_anthropic=True)
            self._key = "cache-only"                 # never sent: _anth_raw is overridden; only makes counts() read the cached field
            self.tokenizers = ["gemma", "anthropic"]

        def _gemma_raw(self, txt):
            raise Miss("gemma")

        def _anth_raw(self, txt):
            raise Miss("anthropic")

        def save(self):
            pass

    man = C.load_manifest()
    test_like = set(man["test"]) | set(man.get("test_uncapped_not_scored") or [])
    types = json.load(open(os.path.join(C.paths()["matrix_docs"], "detector_types.json"), encoding="utf-8"))
    types = types if isinstance(types, list) else types.get("types", types)
    contract = PlannerContract(T.planner_types(types), tools=hs.TOOLS, version=T.FT2_VERSION)
    head = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"tree": "<tree>", "head": head, "inputs": {}, "contract": {"system_sha": hashlib.sha256(contract.system_text().encode("utf-8")).hexdigest(),
                                          "tools_sha": _sha_json(getattr(contract, "tools", None))},
           "renders": {}, "render_summary": {}}
    jobs = [(ns, C.journal_path(ns, "capture"), os.path.join(C.ns_work(ns), "tokcache.json")) for ns in OFFLINE_NAMESPACES
            if not only or ns in only]
    if live_capture:                     # part B: a paired run's own live capture (e.g. <E>/B1), re-rendered by THIS tree's code
        jobs.append((NS, os.path.join(live_capture, "docs", "capture.jsonl"), os.path.join(live_capture, "work", "tokcache.json")))
    for ns, cap_path, tok_path in jobs:
        rows = [r for r in _rows(cap_path) if r.get("item") and r.get("blocks") is not None]
        ns_label = ns if cap_path == C.journal_path(ns, "capture") else f"{ns}@live"
        out["inputs"][ns_label] = {"capture_sha": _sha_file(cap_path) if os.path.exists(cap_path) else None,
                                   "tokcache_sha": _sha_file(tok_path) if os.path.exists(tok_path) else None}
        if not rows:
            out["render_summary"][ns_label] = "no capture"
            continue
        ctr = CacheOnly(tok_path, PACKAGE_HEADER)
        pk = T.Packager(ns, ctr)
        fs = T.FSRenderer(pk, test_ids=test_like)
        n = {"presentations": 0, "blocks": 0, "requests": 0, "miss": 0, "errors": 0, "skipped_test": 0, "match_capture": 0,
             "differs_from_capture": 0, "blocks_expected": 0, "requests_expected": 0}
        for r in rows:
            if r["item"] in test_like:
                n["skipped_test"] += 1
                continue
            n["presentations"] += 1
            psum, instr, hold = list(r.get("prompt_summary") or []), r["instruction"], r.get("holding")
            n["requests_expected"] += len(r.get("blocks") or {})
            n["blocks_expected"] += sum(1 for x in (r.get("blocks") or {}) if x != "none")
            got = {}
            for s, b in sorted((r.get("blocks") or {}).items()):
                key = f"{ns_label}|{r['pres']}|{r.get('capture_run')}|{s}"
                try:
                    if s == "none":
                        text, msgs = "", contract.messages(instr, psum, hold)
                    elif s == "FS":
                        d = T.source_dir(ns, "S*")
                        fb = fs.render("S*", os.path.join(d, "ltm.db"), T.read_rows(d), item=r["item"], instruction=instr, lang=r["lang"],
                                       holding=hold, summary_lines=psum)
                        text = fb["text"] or ""
                        msgs = contract.messages(instr, psum, hold, fewshot_block=text or None)
                    else:
                        d = T.source_dir(ns, s)
                        p = pk.render(s, os.path.join(d, "ltm.db"), T.read_rows(d), item=r["item"], instruction=instr, lang=r["lang"],
                                      holding=hold, summary_lines=psum)
                        text = p["text"] or ""
                        msgs = contract.messages(instr, psum, hold, memory_block=text or None)
                except Miss as e:
                    n["miss"] += 1
                    got[key] = {"miss": str(e)}
                    continue
                except Exception as e:                       # noqa: BLE001 - recorded, counted and compared (never silently dropped)
                    n["errors"] += 1
                    got[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                    continue
                req = prompt_sha256(msgs)
                tsha = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
                same = (tsha == (b.get("sha256") if s != "none" else None)) and (req == (r.get("requests") or {}).get(s))
                n["blocks"] += s != "none"
                n["requests"] += 1
                n["match_capture" if same else "differs_from_capture"] += 1
                got[key] = {"text_sha": tsha, "request_sha": req, "matches_capture": same}
            out["renders"].update(got)
        out["render_summary"][ns_label] = n
    if only:
        return out
    # canonical exports of every frozen snapshot (read path), with the manifest's own recorded state hash
    snaps = {}
    for base in (os.path.join(C.SIM, "final_test", "snapshots_ft2"), os.path.join(C.SIM, "final_test", "snapshots")):
        for d, _dirs, files in sorted(os.walk(base)):
            if "ltm.db" in files and "manifest.json" in files:
                rel = os.path.relpath(d, C.ROOT).replace(os.sep, "/")
                tmp = tempfile.mkdtemp(prefix="k23eq_")
                try:
                    m = L.LongTermMemory.open_frozen(d, os.path.join(tmp, "t.db"), kitchen="FloorPlan1", params=L.load_params())
                    dump = m.store.canonical_dump() if hasattr(m, "store") else None
                    snaps[rel] = {"state_hash": m.store.state_hash(), "dump_sha": _sha_json(dump),
                                  "manifest_state_hash": json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8")).get("state_hash"),
                                  "dir_sha": C.dir_sha(d)}
                except Exception as e:                       # noqa: BLE001
                    snaps[rel] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
    out["snapshots"] = snaps
    # Step-5 writes: the registered replay corpus (step5_replay sources) consolidated into a fresh Kitchen 1 memory
    tmp = tempfile.mkdtemp(prefix="k23eq_replay_")
    try:
        srcs = R.snapshot_sources(os.path.join(tmp, "src"))
        eps, rman = R.convert(srcs)
        mem = L.LongTermMemory(os.path.join(tmp, "replay.db"), kitchen="FloorPlan1", params=L.load_params())
        tally = R.replay_into(mem, eps)
        out["replay"] = {"episodes": len(eps), "episode_ids_sha": _sha_json([e["episode"] for e in eps]),
                         "events_sha": _sha_json([e["events"] for e in eps]), "tally": tally["tally"],
                         "results_sha": _sha_json(tally["results"]), "state_hash": mem.store.state_hash(),
                         "sources": [{"name": os.path.basename(str(s.get("copy") or s.get("path") or s)), "sha": s.get("sha256_before") if isinstance(s, dict) else None}
                                     for s in (srcs if isinstance(srcs, list) else [])]}
    except Exception as e:                                   # noqa: BLE001
        out["replay"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # seeded sample / order outputs (pure functions of the master seed and labels)
    caps = {ns: [r["pres"] for r in _rows(C.journal_path(ns, "capture")) if r.get("item") not in test_like] for ns in OFFLINE_NAMESPACES}
    out["orders"] = {
        "rng": {lab: C.rng(*lab.split("|")).random() for lab in ("A29-sample|FloorPlan1", "order|test-v1", "plan|freeze-gates-ft2", "exec|freeze-gates-ft2")},
        "plan_blocks": {ns: C.plan_blocks(sorted(set(p)), lambda q: [("G0", 0), ("G+M", 0)], ns) for ns, p in caps.items()},
        "exec_blocks": {ns: C.exec_blocks({q: [f"u:{q}:a", f"u:{q}:b"] for q in set(p)}, ns) for ns, p in caps.items()},
        "dev_pilot": list(man["dev_pilot"]), "repeat_dev": {ns: C.repeat_items(ns, man) for ns in ("rehearsal-v1", "freeze-gates-ft2")},
        "split_manifest_sha": _sha_file(C.paths()["split"]),
    }
    return out


def compare_offline(a: dict, b: dict, files=None) -> dict:
    keys = sorted(set(a) | set(b))
    diffs = {}
    for k in keys:
        if k in ("tree", "head", "inputs"):
            continue
        if k == "renders":
            ra, rb = a.get(k, {}), b.get(k, {})
            d = sorted(x for x in set(ra) | set(rb) if ra.get(x) != rb.get(x))
            if d:
                diffs[k] = {"n": len(d), "first": [(x, ra.get(x), rb.get(x)) for x in d[:20]]}
        elif a.get(k) != b.get(k):
            diffs[k] = {"base": a.get(k), "branch": b.get(k)}
    miss = {t: sum(1 for v in (x.get("renders") or {}).values() if "miss" in v or "error" in v) for t, x in (("base", a), ("branch", b))}
    errs = [t for t, x in (("base", a), ("branch", b)) if (x.get("replay") or {}).get("error") or not x.get("renders")
            or any("error" in v for v in (x.get("snapshots") or {}).values())]
    incomplete = []
    for t, x in (("base", a), ("branch", b)):
        rs = x.get("render_summary") or {}
        for ns, tot in OFFLINE_TOTALS.items():
            if not summary_complete(rs.get(ns), tot):
                incomplete.append(f"{t}:{ns}")
        for lab, sm in rs.items():
            if lab.endswith("@live") and not (summary_complete(sm) and sm.get("presentations") == len(EXPECTED_PRES)):
                incomplete.append(f"{t}:{lab}")
    ok = not diffs and not any(miss.values()) and not errs and not incomplete
    inputs = {t: {"file": (files or {}).get(t), "sha256": _sha_file(files[t]) if files and files.get(t) and os.path.exists(files[t]) else None,
                  "head": x.get("head"), "inputs": x.get("inputs")} for t, x in (("base", a), ("branch", b))}
    return {"schema": "k23.fp1-equiv.offline-verdict/2", "identical": ok, "differences": diffs, "errors_in": errs, "incomplete": incomplete,
            "inputs": inputs,
            "render_summary": {"base": a.get("render_summary"), "branch": b.get("render_summary")}, "not_renderable": miss,
            "rule": "byte identity of every deterministic artefact (re-rendered packages, FS blocks, request shas, contract system text, "
                    "snapshot canonical exports, Step-5 replay state, seeded orders); no tolerance"}


# ===================================================================================================== part B: stage / collect / compare
#: decisive runs execute the FROZEN 2026-10-15 capture + plans (B0) on both trees (Astra 2026-10-18 review: "execution of the same frozen
#: capture/plans on both trees for all G0/G+M cells"); the independent captures BC / SC are diagnostics (conditional rendering + plans)
DIAG = ("BC", "SC")
B0_IDENTITY = "7a2ee211b85cc534d7677291fce80b04e92cd24a860f1ddd7665185b3a03b67d"   # the K1 final treatment identity (freeze.json, B0 rows)
BASELINE_COMMIT = "b8be0c67f375660c3c268e6e24fb314302632086"
FROZEN_SOURCES = ("S*", "o0k1", "o0k3", "o0k5")       # the freeze-gates-ft2 memory sources (snapshots_ft2/o0/k10, k1, k3, k5)
B0_KEEP = ("capture.jsonl", "plan.jsonl", "gpu_gemma.jsonl", "states")


def _tree_paths(tree):
    return os.path.join(tree, *DOCS_REL), os.path.join(tree, *WORK_REL)


def _without_id(row: dict) -> dict:
    """A journal row with ONLY its top-level treatment_id removed (the one declared rewrite; nothing else is excluded)."""
    return {k: v for k, v in row.items() if k != "treatment_id"}


def _git(tree, *a):
    return subprocess.run(["git", "-C", tree, *a], capture_output=True, text=True)


def _git_clean(tree, rel):
    r = _git(tree, "diff", "--quiet", "--ignore-cr-at-eol", "HEAD", "--", rel)
    u = _git(tree, "ls-files", "--others", "--exclude-standard", "--", rel)
    return r.returncode == 0 and not u.stdout.strip()


def _copytree(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=False)


def _snap_dirs(tree):
    base = os.path.join(tree, "sim", "final_test", "snapshots_ft2", "o0")
    return {s: os.path.join(base, "k10" if s == "S*" else "k" + s.split("k")[1]) for s in FROZEN_SOURCES}


def _snap_shas(tree):
    return {s: _dir_sha(d) for s, d in _snap_dirs(tree).items()}


def successor_identity(tree):
    """THIS tree's treatment identity of the namespace, computed by its own code in a clean process (fails closed before the re-pin)."""
    code = ("import sys; sys.path.insert(0, 'sim/final_test/matrix'); import treatment as T; print(T.identity('%s')['sha'])" % NS)
    r = subprocess.run([sys.executable, "-c", code], cwd=tree, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{tree}: treatment.identity({NS}) failed (the reviewed re-pin is a precondition): {r.stderr.strip()[-300:]}")
    return r.stdout.strip().splitlines()[-1]


def proof_complete(s) -> bool:
    """A render summary of the frozen B0 capture is complete only with all 20 presentations, 84 requests and 64 blocks rendered, every one
    byte-identical to B0, and zero misses / errors (Astra FP1-EQUIV-r2)."""
    return summary_complete(s, (B0_TOTALS["presentations"], B0_TOTALS["requests"], B0_TOTALS["blocks"]))


def render_proof(tree, ev, run):
    """THIS tree re-renders every request + block of the frozen B0 capture byte for byte (offline, cache-only counter, clean process)."""
    outp = os.path.join(ev, run, "render_proof.json")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "offline", "--tree", tree, "--out", outp, "--only", NS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"render proof failed to run: {r.stderr.strip()[-300:]}")
    s = json.load(open(outp, encoding="utf-8"))["render_summary"].get(NS) or {}
    ok = proof_complete(s)
    if not ok:
        raise SystemExit(f"render proof FAILED for {tree}: {s}")
    return {"file": os.path.relpath(outp, ev).replace(os.sep, "/"), "sha256": _sha_file(outp), "summary": s}


def stage(tree: str, run: str, ev: str, expect_commit: str | None = None, identity_fn=successor_identity, proof_fn=render_proof) -> dict:
    """Prepare ONE run inside `tree` (no simulator). Refuses: a repeated stage, a tree with uncommitted code, a commit other than the
    expected one. The first stage of a tree archives its committed namespace docs dir and its untracked work dir into <ev>/orig_<label>/."""
    label = RUN_TREE[run]
    rundir = os.path.join(ev, run)
    if os.path.exists(os.path.join(rundir, "staged.json")):
        raise SystemExit(f"{run} was already staged ({rundir}): a run is staged once (never re-run to obtain a match)")
    head = _git(tree, "rev-parse", "HEAD").stdout.strip()
    want = expect_commit or (BASELINE_COMMIT if label == "base" else None)
    if want is None:
        raise SystemExit(f"{run}: --expect-commit <the reviewed, re-pinned successor commit> is required")
    if head != want:
        raise SystemExit(f"{tree} is at {head}, not the expected {want}")
    if not _git_clean(tree, "sim") or not _git_clean(tree, "docs/final-test"):
        st_ = _git(tree, "status", "--short", "--", "sim", "docs/final-test").stdout.strip().splitlines()
        raise SystemExit(f"{tree}: uncommitted changes under sim/ or docs/final-test/ ({st_[:5]}): a run binds to committed, reviewed code only")
    docs, work = _tree_paths(tree)
    orig = os.path.join(ev, f"orig_{label}")
    if not os.path.isdir(orig):
        os.makedirs(orig)
        _copytree(docs, os.path.join(orig, "docs"))
        _copytree(work, os.path.join(orig, "work"))
        json.dump({"tree": tree, "docs_sha": _dir_sha(docs), "work_sha": _dir_sha(work), "head": head, "when": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(os.path.join(orig, "orig.json"), "w", encoding="utf-8"), indent=1)
        if not os.path.isdir(os.path.join(ev, "B0")):                       # B0 = the frozen run itself (read-only reference)
            _copytree(os.path.join(orig, "docs"), os.path.join(ev, "B0", "docs"))
    b0 = os.path.join(ev, "B0", "docs")
    os.makedirs(rundir, exist_ok=True)
    rec = {"run": run, "tree": tree, "head": head, "staged": time.strftime("%Y-%m-%dT%H:%M:%S"), "snapshots_before": _snap_shas(tree)}
    proof = None
    if run in ("S1", "S2"):                                                  # checks BEFORE the tree is touched
        rec["identity_old"], rec["identity_new"] = B0_IDENTITY, identity_fn(tree)
        proof = proof_fn(tree, ev, run)
        rec["render_proof"] = proof
    elif run in ("B1", "B2"):
        rec["identity_old"] = rec["identity_new"] = B0_IDENTITY
    elif run == "SC" and not os.path.isdir(os.path.join(ev, "BC", "work")):
        raise SystemExit("SC needs BC collected first (its caches)")
    for d in (docs, work):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(docs)
    os.makedirs(os.path.join(work, "cache"))
    if run in RUNS:                                                          # the frozen B0 capture + plans, executed (no capture, no plan job)
        rebound = {}
        for f in B0_KEEP:
            p = os.path.join(b0, f)
            if os.path.isdir(p):
                shutil.copytree(p, os.path.join(docs, f))
            elif f in ("capture.jsonl", "plan.jsonl"):
                rows = _rows(p)
                if any(r.get("treatment_id") != B0_IDENTITY for r in rows):
                    raise SystemExit(f"B0 {f}: a row is not of the frozen identity {B0_IDENTITY[:12]}")
                if rec["identity_new"] == B0_IDENTITY:                       # baseline: the frozen file itself, byte for byte
                    shutil.copy2(p, os.path.join(docs, f))
                else:                                                        # successor: the ONE rewrite, the top-level identity field
                    with open(os.path.join(docs, f), "w", encoding="utf-8", newline="\n") as fh:
                        fh.write("".join(json.dumps(dict(r, treatment_id=rec["identity_new"]), ensure_ascii=False) + "\n" for r in rows))
                rebound[f] = {"rows": len(rows), "sha_b0": _sha_file(p), "sha_staged": _sha_file(os.path.join(docs, f))}
            elif os.path.exists(p):
                shutil.copy2(p, os.path.join(docs, f))
        evs = [e for e in _rows(os.path.join(b0, "events.jsonl")) if e.get("phase") in ("capture", "plan")]
        with open(os.path.join(docs, "events.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evs))
        rec["journals"] = rebound
    else:                                                                    # BC / SC diagnostics: fresh capture + cached plan
        src = os.path.join(orig, "work") if run == "BC" else os.path.join(ev, "BC", "work")
        if run == "SC" and not os.path.isdir(src):
            raise SystemExit("SC needs BC collected first (its caches)")
        if os.path.isdir(os.path.join(src, "cache")):
            shutil.rmtree(os.path.join(work, "cache"))
            shutil.copytree(os.path.join(src, "cache"), os.path.join(work, "cache"))
        if os.path.exists(os.path.join(src, "tokcache.json")):
            shutil.copy2(os.path.join(src, "tokcache.json"), os.path.join(work, "tokcache.json"))
    rec["docs_sha"], rec["work_sha"] = _dir_sha(docs), _dir_sha(work)
    json.dump(rec, open(os.path.join(rundir, "staged.json"), "w", encoding="utf-8"), indent=1)
    return rec


def collect(tree: str, run: str, ev: str) -> dict:
    label = RUN_TREE[run]
    docs, work = _tree_paths(tree)
    orig = os.path.join(ev, f"orig_{label}")
    dst = os.path.join(ev, run)
    st = json.load(open(os.path.join(dst, "staged.json"), encoding="utf-8"))
    for sub, src in (("docs", docs), ("work", work)):
        if os.path.isdir(os.path.join(dst, sub)):
            raise SystemExit(f"{dst}/{sub} exists: a run is collected once (never overwritten)")
        _copytree(src, os.path.join(dst, sub))
    rec = {"run": run, "collected": time.strftime("%Y-%m-%dT%H:%M:%S"), "head": _git(tree, "rev-parse", "HEAD").stdout.strip(),
           "docs_sha": _dir_sha(os.path.join(dst, "docs")), "work_sha": _dir_sha(os.path.join(dst, "work")), "snapshots_after": _snap_shas(tree)}
    rec["head_unchanged"] = rec["head"] == st["head"]
    for d, sub in ((docs, "docs"), (work, "work")):                          # restore the tree byte for byte and verify
        if os.path.isdir(d):
            shutil.rmtree(d)
        if os.path.isdir(os.path.join(orig, sub)):
            shutil.copytree(os.path.join(orig, sub), d)
    o = json.load(open(os.path.join(orig, "orig.json"), encoding="utf-8"))
    rec["restored"] = {"docs": _dir_sha(docs) == o["docs_sha"], "work": _dir_sha(work) == o["work_sha"], "git_clean": _git_clean(tree, "/".join(DOCS_REL))}
    json.dump(rec, open(os.path.join(dst, "collected.json"), "w", encoding="utf-8"), indent=1)
    if not all(rec["restored"].values()):
        raise SystemExit(f"restore of {tree} FAILED: {rec['restored']} (the archive is in {orig})")
    return rec


def _by(rows, keyf):
    out = {}
    for r in rows:
        out[keyf(r)] = r                                     # latest row wins (journals are append-only; resume rows supersede)
    return out


def _testlog_dumps(work):
    d = os.path.join(work, "testlogs")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".db"):
            continue
        c = sqlite3.connect(f"file:{os.path.join(d, f)}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        dump = {}
        for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            rows = [dict(r) for r in c.execute(f"SELECT * FROM {t}")]
            dump[t] = sorted(json.dumps(_strip(r, "testlog"), sort_keys=True, default=str) for r in rows)
        c.close()
        out[f] = {"tables": {t: len(v) for t, v in dump.items()}, "sha": _sha_json(dump)}
    return out


def load_run(ev, run):
    d = os.path.join(ev, run, "docs")
    j = {k: _rows(os.path.join(d, f"{k}.jsonl")) for k in ("capture", "plan", "exec", "outcome.sealed", "score", "events")}
    meta = {}
    for f in ("staged.json", "collected.json"):
        p = os.path.join(ev, run, f)
        meta[f] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    return {"capture": _by(j["capture"], lambda r: r["pres"]), "plan": _by(j["plan"], lambda r: r["key"]),
            "exec": _by(j["exec"], lambda r: (r["unit"], r.get("run", 0))), "outcome": _by(j["outcome.sealed"], lambda r: (r.get("unit"), r.get("run", 0))),
            "score": _by([r for r in j["score"] if r.get("arm") in EXEC_ARMS], lambda r: (r["cell"], r.get("run", 0))),
            "events": j["events"], "meta": meta,
            "testlogs": _testlog_dumps(os.path.join(ev, run, "work")) if os.path.isdir(os.path.join(ev, run, "work")) else {}}


def _cmp(a, b, kind):
    sa, sb = _strip(a, kind), _strip(b, kind)
    if sa == sb:
        return []
    keys = sorted(set(sa or {}) | set(sb or {})) if isinstance(sa, dict) and isinstance(sb, dict) else ["<value>"]
    return [k for k in keys if (sa or {}).get(k) != (sb or {}).get(k)] if keys != ["<value>"] else keys


#: decision fields of a plan row (a reply served from the same cache must reproduce all of them)
PLAN_BEHAVIOR = ("status", "plan", "steps", "first_plan", "first_ok", "repaired", "normalised", "unknown_types", "types_sha", "model",
                 "decoding", "prompt_version", "say", "request_sha")
#: decision fields of an executed cell (score row, exec row, sealed outcome); everything else is a measurement or volatile
SCORE_BEHAVIOR = ("sr", "strict_sr", "goal_ok", "exec_ok", "executed", "exec_status", "exec_failure", "exec_attempt", "failure_class", "valid",
                  "invalid_episode", "infra", "infra_planner", "plan_status", "plan_len", "steps_n", "steps_ok_n", "pc_first", "pc_final",
                  "tds_first", "tds_final", "tds_final_p", "tds_final_r", "unrealistic_slice", "n_slice", "n_slice_no_knife",
                  "n_slice_unknown", "excluded_reset", "empty_plan_success", "run_class", "first_ok", "repaired", "scored", "level", "family")
STEP_BEHAVIOR = ("action", "object", "to", "ok", "cause", "under", "retries")


def behaviour_view(s: dict, e: dict, o: dict) -> dict:
    """One flat dict of decision fields; the step trace is ELEMENT-WISE (steps[i].field), never one compound field."""
    out = {f"score.{k}": s.get(k) for k in SCORE_BEHAVIOR}
    au = e.get("audit") or {}
    sc = e.get("state_check") or {}
    out.update({"exec.status": e.get("status"), "exec.attempt": e.get("attempt"), "exec.error": e.get("error"), "exec.chooser": e.get("chooser"),
                "exec.exec_policy": json.dumps(e.get("exec_policy"), sort_keys=True), "exec.briefing_log": json.dumps(e.get("briefing_log"), sort_keys=True),
                "exec.snapshot_sha_before": e.get("snapshot_sha_before"), "exec.snapshot_sha_after": e.get("snapshot_sha_after"),
                "exec.audit.violations": au.get("violations"), "exec.audit.invalid": au.get("invalid"), "exec.audit.errors": au.get("errors"),
                "exec.audit.oracle": (au.get("counts") or {}).get("oracle"),
                "exec.state_check.ok": sc.get("ok"), "exec.state_check.registered_ok": sc.get("registered_ok"),
                "exec.state_check.override": sc.get("override"),
                "exec.state_check.breach_kinds": ",".join(sorted(str(b.get("kind")) for b in sc.get("breaches") or [])),
                "outcome.present": bool(o), "outcome.goal_ok": o.get("goal_ok"), "outcome.exec_ok": o.get("exec_ok"), "outcome.sr": o.get("sr"),
                "outcome.found": o.get("found"), "outcome.gate_pass": o.get("gate_pass"), "outcome.applied": o.get("applied"),
                "outcome.steps.n": len(o.get("steps") or [])})
    for i, st in enumerate(o.get("steps") or []):
        for k in STEP_BEHAVIOR:
            out[f"outcome.steps[{i}].{k}"] = st.get(k)
    return out


def measurement_view(e: dict, o: dict) -> dict:
    sc = e.get("state_check") or {}
    m = {"exec.prims": e.get("prims"), "exec.world_hash_match": e.get("world_hash_match"), "exec.max_dpos": sc.get("max_dpos"),
         "exec.max_drot": sc.get("max_drot"), "exec.obs_drift": json.dumps(sc.get("obs_drift"), sort_keys=True), "outcome.prims": o.get("prims"),
         "outcome.walking_m": o.get("walking_m"), "outcome.opens": o.get("opens"), "outcome.retries": o.get("retries"),
         "outcome.stale_visits": o.get("stale_visits")}
    for i, st in enumerate(o.get("steps") or []):
        m[f"outcome.steps[{i}].prims"] = st.get("prims")
    return m


def _wall_rows(R, run):
    return [k for k, o in R[run]["outcome"].items() for st in o.get("steps") or [] if st.get("under") == "episode_wall"]


def compare_live(ev: str, offline_verdicts=()) -> dict:
    """Fail-closed comparison. verdict = EQUIVALENT only when every check passes and no behaviour field differs between baseline and
    successor; UNRESOLVED when every difference is on a field that also differs between the two baseline runs (pending investigation,
    never automatically equivalent); NOT_EQUIVALENT otherwise (including any missing evidence)."""
    out = {"schema": "k23.fp1-equiv.live-verdict/2", "blocking": [], "unresolved": [], "cells": {}, "capture_diag": {}, "plan_diag": {},
           "measure_spread": {}}
    block = lambda what, **kw: out["blocking"].append(dict(what=what, **kw))          # noqa: E731
    # ---- evidence present: offline verdicts, every run staged + collected + restored, heads bound
    vs = [json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None for p in offline_verdicts]
    if len(offline_verdicts) != 2 or len({os.path.normcase(os.path.abspath(p)) for p in offline_verdicts}) != 2:
        block("exactly two distinct offline verdicts are required (part A, and the BC live-capture re-render)", given=list(offline_verdicts))
    for p, v in zip(offline_verdicts, vs):
        if not v or not v.get("identical") or any((v.get("not_renderable") or {}).values()) or v.get("errors_in"):
            block("an offline verdict is missing, not identical, or has unrenderable / errored items", file=p)
    if len(vs) == 2 and all(vs):
        a_sum, l_sum = (vs[0].get("render_summary") or {}), (vs[1].get("render_summary") or {})
        if not all(summary_complete((a_sum.get(t) or {}).get(ns), tot) for ns, tot in OFFLINE_TOTALS.items() for t in ("base", "branch")):
            block("the part A verdict is not complete (fixed totals of all four frozen K1 captures, 0 misses / errors / mismatches) in both trees")
        lv = [(l_sum.get(t) or {}).get(NS + "@live") for t in ("base", "branch")]
        if not all(summary_complete((l_sum.get(t) or {}).get(ns), tot) for ns, tot in OFFLINE_TOTALS.items() for t in ("base", "branch")) \
                or not all(summary_complete(x) and x.get("presentations") == len(EXPECTED_PRES) for x in lv):
            block("the live verdict is not complete (all four frozen captures + all 20 BC presentations, 0 misses / errors / mismatches)")
    R = {}
    for r in ("B0",) + RUNS + DIAG:
        if not os.path.isdir(os.path.join(ev, r, "docs")):
            block("run missing", run=r)
            continue
        R[r] = load_run(ev, r)
        if r != "B0":
            m = R[r]["meta"]
            if not m["staged.json"] or not m["collected.json"]:
                block("run not staged / collected by this harness", run=r)
            elif not (m["collected.json"].get("head_unchanged") and all(m["collected.json"].get("restored", {}).values())):
                block("tree moved during the run, or its restore failed", run=r)
    if out["blocking"]:
        out["verdict"] = "NOT_EQUIVALENT"
        return out
    heads = {r: R[r]["meta"]["staged.json"]["head"] for r in RUNS + DIAG}
    if len({heads[r] for r in ("B1", "B2", "BC")}) != 1 or len({heads[r] for r in ("S1", "S2", "SC")}) != 1:
        block("runs of one tree were made at different commits", heads=heads)
    out["heads"] = heads
    bc_cap, bc_tok = (os.path.join(ev, "BC", "docs", "capture.jsonl"), os.path.join(ev, "BC", "work", "tokcache.json"))
    for i, v in enumerate(vs if len(vs) == 2 and all(vs) else []):
        ins = v.get("inputs") or {}
        for t, run in (("base", "B1"), ("branch", "S1")):
            e = ins.get(t) or {}
            if e.get("head") != heads.get(run):
                block("an offline verdict was not made on the commit of the paired runs", verdict=offline_verdicts[i], tree=t)
            # the verdict's artifact (the offline output it compared) must be named, hashed, present, unchanged, and its CONTENT must be
            # what the verdict says (head, rendered inputs, render summary): a missing, renamed or substituted artifact is refused
            af, ah = e.get("file"), e.get("sha256")
            if not af or not ah:
                block("an offline verdict does not name and hash its artifact", verdict=offline_verdicts[i], tree=t)
            elif not os.path.exists(af) or _sha_file(af) != ah:
                block("an offline verdict's artifact is missing or changed since the verdict", verdict=offline_verdicts[i], tree=t)
            else:
                art = json.load(open(af, encoding="utf-8"))
                if (art.get("head") != e.get("head") or art.get("inputs") != e.get("inputs")
                        or art.get("render_summary") != (v.get("render_summary") or {}).get(t)):
                    block("an offline verdict's artifact content does not match the verdict (substituted artifact)",
                          verdict=offline_verdicts[i], tree=t)
            if i == 1:
                li = (e.get("inputs") or {}).get(NS + "@live") or {}
                if li.get("capture_sha") != (_sha_file(bc_cap) if os.path.exists(bc_cap) else "absent") or \
                        li.get("tokcache_sha") != (_sha_file(bc_tok) if os.path.exists(bc_tok) else None):
                    block("the live verdict did not render the collected BC capture / token cache", tree=t)
    out["identities"] = {r: (R[r]["meta"]["staged.json"].get("identity_old"), R[r]["meta"]["staged.json"].get("identity_new")) for r in RUNS}
    # ---- decisive runs: exactly the frozen capture + plans (identity field aside), the expected cells, all records
    b0cap, b0plan = R["B0"]["capture"], R["B0"]["plan"]
    expected = sorted(f"{NS}|plan|{a}|{p}|r0" for a in EXEC_ARMS for p in EXPECTED_PRES)     # fixed independently of B0's content
    out["expected_cells"] = len(expected)
    if tuple(sorted(b0cap)) != EXPECTED_PRES:
        block("B0 capture is not the fixed 20-presentation roster", have=sorted(b0cap)[:5])
    if len(b0plan) != B0_TOTALS["plans"] or any(((b0plan.get(f"{NS}|plan|{a}|{p}|r0") or {}).get("status") != "ok")
                                                for a in ("G0", "G+M") for p in EXPECTED_PRES):
        block("B0 plans are not complete (84 rows; an ok G0 and G+M plan for every presentation)")
    b0files = {f: _sha_file(os.path.join(ev, "B0", "docs", f)) for f in ("capture.jsonl", "plan.jsonl")}
    ids_new = set()
    for r in RUNS:
        if {k: _without_id(v) for k, v in R[r]["capture"].items()} != {k: _without_id(v) for k, v in b0cap.items()}:
            block("decisive run capture differs from the frozen B0 capture in more than the top-level identity", run=r)
        if {k: _without_id(v) for k, v in R[r]["plan"].items()} != {k: _without_id(v) for k, v in b0plan.items()}:
            block("decisive run plans differ from the frozen B0 plans in more than the top-level identity", run=r)
        st = R[r]["meta"]["staged.json"]
        jr = st.get("journals") or {}
        for f in ("capture.jsonl", "plan.jsonl"):
            now = _sha_file(os.path.join(ev, r, "docs", f)) if os.path.exists(os.path.join(ev, r, "docs", f)) else None
            if (jr.get(f) or {}).get("sha_b0") != b0files[f] or (jr.get(f) or {}).get("sha_staged") != now:
                block("recorded journal provenance does not match the evidence", run=r, file=f)
        if r in ("B1", "B2"):
            if st.get("identity_old") != B0_IDENTITY or st.get("identity_new") != B0_IDENTITY or any(
                    _sha_file(os.path.join(ev, r, "docs", f)) != b0files[f] for f in b0files):
                block("baseline run is not B0 byte for byte under the frozen identity", run=r)
        else:
            pr = st.get("render_proof") or {}
            pf = os.path.join(ev, *str(pr.get("file") or "missing").split("/"))
            if st.get("identity_old") != B0_IDENTITY or not st.get("identity_new") or st.get("identity_new") == B0_IDENTITY:
                block("successor run without a recorded new identity", run=r)
            pdoc = json.load(open(pf, encoding="utf-8")) if os.path.exists(pf) else {}
            if not os.path.exists(pf) or _sha_file(pf) != pr.get("sha256") or not proof_complete((pdoc.get("render_summary") or {}).get(NS)):
                block("successor render proof missing, altered or incomplete (summary re-read from the hashed proof file)", run=r)
            elif pdoc.get("head") != st.get("head") or ((pdoc.get("inputs") or {}).get(NS) or {}).get("capture_sha") != b0files["capture.jsonl"]:
                block("successor render proof was not made on this run's commit over the frozen B0 capture (stale or substituted)", run=r)
            if any(x.get("treatment_id") != st.get("identity_new") for x in list(R[r]["capture"].values()) + list(R[r]["plan"].values())):
                block("successor rows do not all carry the recorded new identity", run=r)
            ids_new.add(st.get("identity_new"))
        if any(e.get("event") in ("abort_invalid", "invalid") for e in R[r]["events"]):
            block("invalid / abort event", run=r)
        if _wall_rows(R, r):
            block("episode_wall row (the 900 s cap fired)", run=r, units=_wall_rows(R, r)[:10])
        sb, sa = R[r]["meta"]["staged.json"]["snapshots_before"], R[r]["meta"]["collected.json"]["snapshots_after"]
        if sb != sa or sb != R["B1"]["meta"]["staged.json"]["snapshots_before"]:
            block("a frozen snapshot changed during the run, or differs between trees", run=r)
        for (u, _rr), e in R[r]["exec"].items():
            if e.get("snapshot_sha_before") != e.get("snapshot_sha_after"):
                block("snapshot sha changed inside an execution", run=r, unit=u)
    if len(ids_new) != 1:
        block("S1 and S2 ran under different successor identities", ids=sorted(map(str, ids_new)))

    def view(run, cell):
        s = R[run]["score"].get((cell, 0))
        if not s:
            return None
        e = R[run]["exec"].get((s.get("unit"), 0))
        if e is None:
            return None
        o = R[run]["outcome"].get((s.get("unit"), 0)) or {}
        if e.get("status") == "done" and not o:
            return None
        return behaviour_view(s, e, o), measurement_view(e, o)
    spread = {}
    for c in expected:
        v = {r: view(r, c) for r in RUNS}
        miss = [r for r in RUNS if v[r] is None]
        if miss:
            block("expected cell has no complete record (score + exec + outcome)", cell=c, runs=miss)
            continue
        d = lambda x, y, i: sorted(k for k in set(v[x][i]) | set(v[y][i]) if v[x][i].get(k) != v[y][i].get(k))   # noqa: E731
        bs, bs2, bb, ss = d("B1", "S1", 0), d("B2", "S2", 0), d("B1", "B2", 0), d("S1", "S2", 0)
        out["cells"][c] = {"B1~S1": bs, "B2~S2": bs2, "B1~B2": bb, "S1~S2": ss}
        for pair in ("B1~S1", "B2~S2", "B1~B2", "S1~S2"):
            for k in d(*pair.split("~"), 1):
                spread.setdefault(k, {}).setdefault(pair, 0)
                spread[k][pair] += 1
        diff = set(bs) | set(bs2)
        if diff and diff <= set(bb):
            out["unresolved"].append({"cell": c, "fields": sorted(diff), "note": "also differs within the baseline: investigate, report; never re-run"})
        elif diff:
            block("behaviour field differs between baseline and successor and is stable within the baseline", cell=c, fields=sorted(diff - set(bb)))
    out["measure_spread"] = spread
    # ---- memory: test-epoch logs (if any unit opened one) must be the same multiset per run pair
    tl = {r: sorted(x["sha"] for x in R[r]["testlogs"].values()) for r in RUNS}
    out["testlogs"] = {r: len(v) for r, v in tl.items()}
    if tl["B1"] != tl["S1"] or tl["B2"] != tl["S2"]:
        (out["unresolved"] if tl["B1"] != tl["B2"] else out["blocking"]).append({"what": "test-epoch memory logs differ", "counts": out["testlogs"]})
    # ---- diagnostics: independent captures (BC, SC) and their cached plans (complete: 20 presentations, 84 plans each)
    for r in DIAG:
        if tuple(sorted(R[r]["capture"])) != EXPECTED_PRES or len(R[r]["plan"]) != B0_TOTALS["plans"]:
            block("diagnostic run incomplete", run=r, presentations=len(R[r]["capture"]), plans=len(R[r]["plan"]))
    for p in sorted(set(R["BC"]["capture"]) | set(R["SC"]["capture"])):
        a, b = R["BC"]["capture"].get(p), R["SC"]["capture"].get(p)
        if not a or not b:
            block("diagnostic capture missing a presentation", pres=p)
            continue
        obs_eq = all(a.get(f) == b.get(f) for f in OBS_FIELDS if f not in ("world_hash", "restored_state"))
        dd = _cmp(a, b, "capture")
        out["capture_diag"][p] = {"obs_equal": obs_eq, "differs": dd, "B0_obs_equal_BC": all((b0cap.get(p) or {}).get(f) == a.get(f) for f in OBS_FIELDS[:3])}
        if obs_eq and set(dd) - set(OBS_FIELDS):
            block("diagnostic capture: equal observation, different rendered fields", pres=p, fields=sorted(set(dd) - set(OBS_FIELDS)))
    for k in sorted(set(R["BC"]["plan"]) | set(R["SC"]["plan"])):
        a, b = R["BC"]["plan"].get(k), R["SC"]["plan"].get(k)
        if not a or not b:
            block("diagnostic plan missing a cell", cell=k)
            continue
        same_req = a.get("request_sha") == b.get("request_sha")
        dd = [f for f in PLAN_BEHAVIOR if a.get(f) != b.get(f)]
        out["plan_diag"][k] = {"same_request": same_req, "differs": dd, "sc_calls": b.get("calls"), "sc_cached": b.get("cached_calls")}
        if same_req and (dd or (b.get("calls") or 0) > (b.get("cached_calls") or 0)):
            block("diagnostic plan: same request, different plan or a planner call", cell=k, fields=dd)
    out["comparable_cells"] = len(out["cells"])
    out["verdict"] = "NOT_EQUIVALENT" if out["blocking"] else ("UNRESOLVED" if out["unresolved"] else "EQUIVALENT")
    out["rule"] = ("EQUIVALENT only if: part A and the BC live re-render are byte-identical; the four decisive runs (B1, S1, B2, S2) executed "
                   "exactly the frozen B0 capture and plans (only the declared identity field rebound for S1 / S2, with a render proof); all "
                   f"{len(expected)} expected cells have complete records; no behaviour field differs between baseline and successor; no "
                   "snapshot change, invalid / abort event or episode_wall row; diagnostics: equal observation -> byte-identical capture rows, "
                   "same request -> same cached plan with no planner call. A difference that also appears between B1 and B2 makes the verdict "
                   "UNRESOLVED (investigate; never automatically equivalent); any other failure is NOT_EQUIVALENT. Nothing is re-run.")
    return out


RUN_SH = "/mnt/d/Personal/MP-agent-side/sim/final_test/matrix/k23_fp1_equiv_run.sh"
WSL_TREE = {"base": "/mnt/d/Personal/MP-base", "branch": "/mnt/d/Personal/MP-agent-side"}


def launch(run: str, job: str, evidence_wsl: str, script_args: list) -> int:
    """Windows side of one live job: `wsl -d Ubuntu-22.04 --cd <tree> -- bash <RUN_SH> <tree> <evidence> <run> <job> <script> [args]` as a
    list (no shell string). The capture job counts tokens with both tokenizers, so the study key is forwarded through WSLENV exactly as the
    FREEZE-GATES launcher did (common.wsl_key_env; never printed). No other job receives the key."""
    tree = WSL_TREE[RUN_TREE[run]]
    env = dict(os.environ)
    if job == "capture":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import common as C
        env = C.wsl_key_env(env)
    cmd = ["wsl", "-d", "Ubuntu-22.04", "--cd", tree, "--", "bash", RUN_SH, tree, evidence_wsl, run, job] + list(script_args)
    print("launch:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch"); la.add_argument("--run", required=True, choices=RUNS + DIAG); la.add_argument("--job", required=True)
    la.add_argument("--evidence-wsl", default="/mnt/d/Personal/K23-FP1-EQUIV-RUN"); la.add_argument("script_args", nargs=argparse.REMAINDER)
    o = sub.add_parser("offline"); o.add_argument("--tree", required=True); o.add_argument("--out", required=True)
    o.add_argument("--live-capture", default=None, help="a collected run dir (<E>/BC): also re-render ITS live capture")
    o.add_argument("--only", default=None, help="comma list of namespaces: render checks only (the S1 / S2 render proof)")
    co = sub.add_parser("compare-offline"); co.add_argument("a"); co.add_argument("b"); co.add_argument("--out", required=True)
    st = sub.add_parser("stage"); st.add_argument("--tree", required=True); st.add_argument("--run", required=True, choices=RUNS + DIAG)
    st.add_argument("--evidence", required=True); st.add_argument("--expect-commit", default=None)
    cl = sub.add_parser("collect"); cl.add_argument("--tree", required=True); cl.add_argument("--run", required=True, choices=RUNS + DIAG)
    cl.add_argument("--evidence", required=True)
    cv = sub.add_parser("compare-live"); cv.add_argument("--evidence", required=True); cv.add_argument("--out", required=True)
    cv.add_argument("--offline-verdict", action="append", default=[], help="repeat: part A verdict and the BC live re-render verdict")
    a = ap.parse_args(argv)
    if a.cmd == "launch":
        args = a.script_args[1:] if a.script_args[:1] == ["--"] else a.script_args
        return launch(a.run, a.job, a.evidence_wsl, args)
    if a.cmd == "offline":
        outp = os.path.abspath(a.out)
        res = offline(os.path.abspath(a.tree), a.live_capture, tuple(a.only.split(",")) if a.only else None)
        json.dump(res, open(outp, "w", encoding="utf-8"), indent=1, sort_keys=True, default=str)
        print(json.dumps(res["render_summary"], indent=1), "replay:", {k: (res.get("replay") or {}).get(k) for k in ("episodes", "tally", "state_hash", "error")})
        return 0
    if a.cmd == "compare-offline":
        res = compare_offline(*(json.load(open(p, encoding="utf-8")) for p in (a.a, a.b)), files={"base": a.a, "branch": a.b})
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True, default=str)
        print("identical" if res["identical"] else f"DIFFERENT: {sorted(res['differences'])}", res["not_renderable"])
        return 0 if res["identical"] else 1
    if a.cmd == "stage":
        print(json.dumps(stage(os.path.abspath(a.tree), a.run, os.path.abspath(a.evidence), a.expect_commit), indent=1))
        return 0
    if a.cmd == "collect":
        print(json.dumps(collect(os.path.abspath(a.tree), a.run, os.path.abspath(a.evidence)), indent=1))
        return 0
    res = compare_live(os.path.abspath(a.evidence), tuple(a.offline_verdict))
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True, default=str)
    print(res["verdict"], f"blocking {len(res['blocking'])}, unresolved {len(res['unresolved'])}, comparable cells {res.get('comparable_cells')}")
    return 0 if res["verdict"] == "EQUIVALENT" else 1


if __name__ == "__main__":
    sys.exit(main())
