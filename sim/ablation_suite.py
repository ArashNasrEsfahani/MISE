"""Reproducible ablation suite: every evaluation the thesis still owes, run against SNAPSHOT stores.

    # inside WSL, web console STOPPED, from the repo root
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python sim/ablation_suite.py \
        --suite sim/ablation_runs/2026-09-15 --plans loop24,loop102,matrix,xling,perception --reps 3

Why this exists. Three of the thesis's open items are evaluations that were either never run or run on a tree
nobody can identify (CLAUDE.md, "Known integrity item"):

  loop24      the published loop ablation (tab.ablation), replicated like-for-like on the FIXED tree: the exact
              24-record Phase-8 dataset recovered from commit e2349cb, gemma3:4b, FloorPlan1, hybrid RAG, oracle
              perception, the pre-search action space (--score-search off). Three arms per repetition:
                off_cold  loop off, empty KB                      (the g3_hybrid control)
                on        loop on,  empty KB at the start         (exp1)
                off_warm  loop off, the KB `on` left behind       (exp2)
  loop102     the same three arms on the full 102-record, 3-kitchen dataset.
  matrix      the full retrieval ablation (RQ2): off/keyword/hybrid/probe/agentic/full, loop off, over the KB a
              loop-on teach pass accumulated (loop102 `on` rep 1). Test records = teach records, i.e. "the agent
              revisits commands it has executed" - stated, not hidden.
  xling       the held-out variant: teach on the ENGLISH half (loop on), score the six arms on the PERSIAN half
              only. Nothing Persian is ever written to the KB, so any gain is cross-lingual transfer.
  perception  the never-run perceived arm: FloorPlan1 records, rag off, loop off - oracle vs perceived with the
              thesis detector vs perceived with a candidate detector (--candidate-weights).

Each run gets its own directory holding its KB, memory.db, loop reports, trace store, eval output and log, so no
run can touch `sim/knowledge_base.json`, `sim/memory.db`, `sim/loop_reports.jsonl` or `sim/trace.db`, and every
number can be traced to the exact store it was scored against. The suite manifest records the code provenance
(git HEAD + a hash of every file on the scoring path + whether each is modified relative to HEAD) - the thing
whose absence created the `--loop on` provenance question in the first place.

Resumable: a run whose output JSON already exists (and is not marked interrupted) is skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PHASE8_COMMIT = "e2349cb"
SCORING_PATH = ("eval.py", "improve.py", "agent_llm.py", "rag.py", "agentic_rag.py", "knowledge.py", "memory.py",
                "webapp.py", "thor_actions.py", "perception.py", "search.py", "skills.py", "levels.py", "feedback.py")
RAG_ARMS = ("off", "keyword", "hybrid", "probe", "agentic", "full")
EMPTY_KB = {"procedures": [], "lessons": [], "facts": []}
ALL_PLANS = ("loop24", "loop102", "matrix", "xling", "perception")


# ---------------------------------------------------------------------------------------------------------------
# Pure planning (tested headless)
# ---------------------------------------------------------------------------------------------------------------
def plan_runs(plans, reps: int = 3, model: str = "gemma3:4b", candidate: str = "") -> list:
    """The ordered list of run specs. Pure. Order matters: a run that SEEDS another's KB comes first."""
    common = {"models": model, "perception": "oracle", "score_search": "off", "loop": "off", "rag": "hybrid"}
    out = []

    def add(name, plan, **kw):
        spec = dict(common, name=name, plan=plan, dataset="full", scenes="", lang="all", kb_from=None,
                    weights_dir="")
        spec.update(kw)
        out.append(spec)

    for rep in range(1, reps + 1):
        if "loop24" in plans:
            add(f"loop24_off_cold_r{rep}", "loop24", dataset="phase8", scenes="FloorPlan1")
            add(f"loop24_on_r{rep}", "loop24", dataset="phase8", scenes="FloorPlan1", loop="on")
            add(f"loop24_off_warm_r{rep}", "loop24", dataset="phase8", scenes="FloorPlan1",
                kb_from=f"loop24_on_r{rep}")
    for rep in range(1, reps + 1):
        if "loop102" in plans:
            add(f"loop102_off_cold_r{rep}", "loop102")
            add(f"loop102_on_r{rep}", "loop102", loop="on")
            add(f"loop102_off_warm_r{rep}", "loop102", kb_from=f"loop102_on_r{rep}")
    if "matrix" in plans:
        teach = "loop102_on_r1"
        if not any(s["name"] == teach for s in out):
            add(teach, "loop102", loop="on")
        for arm in RAG_ARMS:
            add(f"matrix_{arm}", "matrix", rag=arm, kb_from=teach)
    if "xling" in plans:
        add("xling_teach_en", "xling", lang="en", loop="on")
        for arm in RAG_ARMS:
            add(f"xling_{arm}_fa", "xling", lang="fa", rag=arm, kb_from="xling_teach_en")
    if "perception" in plans:
        add("perception_oracle", "perception", scenes="FloorPlan1", rag="off")
        add("perception_thesis_detector", "perception", scenes="FloorPlan1", rag="off", perception="perceived")
        if candidate:
            add("perception_candidate_detector", "perception", scenes="FloorPlan1", rag="off",
                perception="perceived", weights_dir="@candidate")
    return out


def eval_argv(spec: dict, run_dir: str, dataset_path: str, weights_dir: str = "") -> list:
    """The `sim/eval.py` argument vector for one run. Pure."""
    argv = ["--dataset", dataset_path, "--models", spec["models"], "--rag", spec["rag"], "--loop", spec["loop"],
            "--perception", spec["perception"], "--score-search", spec["score_search"], "--lang", spec["lang"],
            "--kb", os.path.join(run_dir, "knowledge_base.json"), "--memory", os.path.join(run_dir, "memory.db"),
            "--loop-reports", os.path.join(run_dir, "loop_reports.jsonl"),
            "--out", os.path.join(run_dir, "eval.json")]
    if spec.get("scenes"):
        argv += ["--scenes", spec["scenes"]]
    if weights_dir:
        argv += ["--weights-dir", weights_dir]
    return argv


def summarize(docs: dict) -> dict:
    """{run name: eval results doc} -> per-plan tables with the mean and spread over repetitions. Pure."""
    def cell(doc):
        a = ((doc or {}).get("aggregates") or {}).get("overall") or {}
        lv = ((doc or {}).get("aggregates") or {}).get("by_level") or {}
        errs = sum(1 for r in (doc or {}).get("results", []) if r.get("error"))
        loop_errs = sum(1 for r in (doc or {}).get("results", []) if "loop closure" in str(r.get("error") or ""))
        pdeg = sum(1 for r in (doc or {}).get("results", []) if r.get("perception_degraded"))
        rdeg = sum(1 for r in (doc or {}).get("results", []) if r.get("rag_degraded"))
        return {"n": a.get("n"), "SR": a.get("SR"), "PC": a.get("PC"), "TDS": a.get("TDS"),
                "SR_L1": (lv.get("L1") or {}).get("SR"), "SR_L2": (lv.get("L2") or {}).get("SR"),
                "SR_L3": (lv.get("L3") or {}).get("SR"),
                "errors": errs, "loop_errors": loop_errs, "perception_degraded": pdeg, "rag_degraded": rdeg,
                "mem_tokens": a.get("mem_tokens"), "llm_calls": a.get("llm_calls")}

    def stats(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return {"mean": None, "min": None, "max": None, "k": 0}
        return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "k": len(xs)}

    cells = {name: cell(doc) for name, doc in docs.items()}
    groups = {}
    for name, c in cells.items():
        arm = name.rsplit("_r", 1)[0] if name.rsplit("_r", 1)[-1].isdigit() else name
        groups.setdefault(arm, []).append(c)
    table = {arm: {m: stats([c[m] for c in cs]) for m in ("SR", "PC", "TDS", "SR_L1", "SR_L2", "SR_L3")}
             | {"n": cs[0]["n"], "errors": sum(c["errors"] for c in cs),
                "loop_errors": sum(c["loop_errors"] for c in cs),
                "perception_degraded": sum(c["perception_degraded"] for c in cs),
                "rag_degraded": sum(c["rag_degraded"] for c in cs)}
             for arm, cs in groups.items()}
    return {"runs": cells, "arms": table}


def render_summary(summary: dict) -> str:
    """Markdown table of the arms. ASCII-safe."""
    f = lambda s: "-" if s["mean"] is None else (f"{s['mean']:.3f}" if s["k"] <= 1
                                                 else f"{s['mean']:.3f} [{s['min']:.2f}-{s['max']:.2f}]")
    L = ["| arm | n | reps | SR | PC | TDS | SR L1 | SR L2 | SR L3 | errors | loop err | perc. degraded | rag degraded |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, t in summary["arms"].items():
        L.append(f"| {arm} | {t['n']} | {t['SR']['k']} | {f(t['SR'])} | {f(t['PC'])} | {f(t['TDS'])} | "
                 f"{f(t['SR_L1'])} | {f(t['SR_L2'])} | {f(t['SR_L3'])} | {t['errors']} | {t['loop_errors']} | "
                 f"{t['perception_degraded']} | {t['rag_degraded']} |")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------------------------
def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as exc:
            return f"(git unavailable: {exc})"
    dirty = set(line[3:].strip() for line in git("status", "--porcelain", "--", "sim").splitlines())
    files = {}
    for f in SCORING_PATH:
        p = os.path.join(HERE, f)
        if os.path.isfile(p):
            files[f] = {"md5": _md5(p), "modified_vs_HEAD": (f"sim/{f}" in dirty)}
    return {"git_head": git("rev-parse", "HEAD"), "files": files, "python": sys.version.split()[0],
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------------------------
def _done(run_dir):
    p = os.path.join(run_dir, "eval.json")
    if not os.path.isfile(p):
        return None
    try:
        doc = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    # A run that scored NOTHING (the simulator failed to boot, every scene skipped) is not done, however cleanly
    # eval.py exited: counting it would let a resume skip it and a summary report an empty arm as measured.
    if doc.get("interrupted") or not doc.get("results"):
        return None
    return doc


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", required=True, help="suite directory (created); every run lives under it")
    ap.add_argument("--plans", default=",".join(ALL_PLANS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--candidate-weights", default="", help="candidate detector .pt for the perception plan")
    ap.add_argument("--only", default="", help="comma-separated run names to restrict to")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summary from finished runs")
    args = ap.parse_args(argv)

    suite = os.path.abspath(args.suite)
    plans = [p.strip() for p in args.plans.split(",") if p.strip()]
    bad = [p for p in plans if p not in ALL_PLANS]
    if bad:
        print(f"unknown plan(s): {bad}; choose from {ALL_PLANS}")
        return 2
    specs = plan_runs(plans, reps=args.reps, model=args.model, candidate=args.candidate_weights)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s["name"] in keep]
    os.makedirs(os.path.join(suite, "runs"), exist_ok=True)

    # datasets: the live 102-record file, and the 24-record Phase-8 file recovered verbatim from git
    datasets = {"full": os.path.join(HERE, "dataset", "commands.jsonl"),
                "phase8": os.path.join(suite, f"commands.phase8.{PHASE8_COMMIT}.jsonl")}
    if any(s["dataset"] == "phase8" for s in specs) and not os.path.isfile(datasets["phase8"]):
        txt = subprocess.run(["git", "show", f"{PHASE8_COMMIT}:sim/dataset/commands.jsonl"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8").stdout
        if txt.count("\n") < 20:
            print(f"could not recover the Phase-8 dataset from {PHASE8_COMMIT}")
            return 3
        with open(datasets["phase8"], "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    cand_dir = ""
    if args.candidate_weights:
        cand_dir = os.path.join(suite, "detectors", "candidate")
        os.makedirs(cand_dir, exist_ok=True)
        dst = os.path.join(cand_dir, "FloorPlan1_physics.pt")
        if not os.path.isfile(dst):
            shutil.copy2(args.candidate_weights, dst)

    manifest_path = os.path.join(suite, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.isfile(manifest_path) else {"runs": {}}
    manifest.setdefault("provenance_first", provenance())
    manifest["provenance_last"] = provenance()
    manifest["datasets"] = {k: {"path": v, "md5": _md5(v) if os.path.isfile(v) else None} for k, v in datasets.items()}
    manifest["plans"], manifest["reps"], manifest["model"] = plans, args.reps, args.model
    if args.candidate_weights:
        manifest["candidate_weights"] = {"source": args.candidate_weights,
                                         "md5": _md5(os.path.join(cand_dir, "FloorPlan1_physics.pt"))}

    print(f"suite {suite}: {len(specs)} run(s)", flush=True)
    for spec in specs:
        run_dir = os.path.join(suite, "runs", spec["name"])
        if args.summarize:
            break
        if _done(run_dir):
            print(f"  [skip] {spec['name']} (done)", flush=True)
            continue
        weights_dir = cand_dir if spec.get("weights_dir") == "@candidate" else ""
        argv_e = eval_argv(spec, run_dir, datasets[spec["dataset"]], weights_dir)
        print(f"  [run ] {spec['name']}: eval.py {' '.join(argv_e)}", flush=True)
        if args.dry_run:
            continue
        os.makedirs(run_dir, exist_ok=True)
        # fresh stores: an empty KB, or a COPY of the seeding run's final KB (never the file itself)
        kb_dst = os.path.join(run_dir, "knowledge_base.json")
        if spec["kb_from"]:
            src = os.path.join(suite, "runs", spec["kb_from"], "knowledge_base.json")
            if not _done(os.path.join(suite, "runs", spec["kb_from"])) or not os.path.isfile(src):
                print(f"  [FAIL] {spec['name']}: seeding run {spec['kb_from']} has not finished", flush=True)
                continue
            shutil.copy2(src, kb_dst)
        else:
            with open(kb_dst, "w", encoding="utf-8") as fh:
                json.dump(EMPTY_KB, fh)
        for stale in ("memory.db", "loop_reports.jsonl", "trace.db"):
            p = os.path.join(run_dir, stale)
            if os.path.isfile(p):
                os.remove(p)
        env = dict(os.environ, TRACE_DB=os.path.join(run_dir, "trace.db"), TRACE_KB_PATH=kb_dst,
                   TRACE_KB_AUDIT_PATH=os.path.join(run_dir, "knowledge_base_audit.log"),
                   TRACE_MEM_DB_PATH=os.path.join(run_dir, "memory.db"),
                   TRACE_LOOP_REPORTS_PATH=os.path.join(run_dir, "loop_reports.jsonl"))
        kb_before = json.load(open(kb_dst, encoding="utf-8"))
        t0 = time.time()
        with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, "-u", os.path.join(HERE, "eval.py"), *argv_e], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        kb_after = json.load(open(kb_dst, encoding="utf-8"))
        manifest["runs"][spec["name"]] = {
            "spec": spec, "argv": argv_e, "returncode": rc, "seconds": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kb_entries_before": {k: len(v) for k, v in kb_before.items() if isinstance(v, list)},
            "kb_entries_after": {k: len(v) for k, v in kb_after.items() if isinstance(v, list)}}
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        doc = _done(run_dir)
        sr = ((doc or {}).get("aggregates") or {}).get("overall", {}).get("SR") if doc else None
        print(f"  [{'ok  ' if rc == 0 and doc else 'FAIL'}] {spec['name']} rc={rc} "
              f"{manifest['runs'][spec['name']]['seconds']} s SR={sr}", flush=True)
        if not doc:
            # Stop rather than burn through the queue: a run that scored nothing almost always means the
            # environment is broken (no display, no GPU, Ollama down), and later runs may be seeded from this one.
            print(f"  ABORT: {spec['name']} scored no records - see {os.path.join(run_dir, 'log.txt')}", flush=True)
            return 5

    docs = {s["name"]: _done(os.path.join(suite, "runs", s["name"])) for s in specs}
    docs = {k: v for k, v in docs.items() if v}
    summary = summarize(docs)
    with open(os.path.join(suite, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    md = render_summary(summary)
    with open(os.path.join(suite, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print("\n" + md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
