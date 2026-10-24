"""Self-test for `sim/ablation_suite.py` - the run plan, the eval argv and the summary. Headless: no ai2thor,
no Ollama, no GPU, and it never launches an evaluation.

    python sim/_ablation_suite_selftest.py        # from the REPO ROOT
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


print("1. import hygiene")
heavy = ("torch", "ultralytics", "ai2thor", "cv2", "thor_actions", "webapp")
before = {m for m in heavy if m in sys.modules}
import ablation_suite as AS                                          # noqa: E402
check("importing the suite pulls in nothing heavy", {m for m in heavy if m in sys.modules} == before)

print("\n2. the run plan")
specs = AS.plan_runs(list(AS.ALL_PLANS), reps=3, candidate="/x/cand.pt")
names = [s["name"] for s in specs]
check("names are unique", len(names) == len(set(names)))
check("loop24 has 3 arms x 3 reps", sum(1 for n in names if n.startswith("loop24_")) == 9)
check("every seeded run comes AFTER the run that seeds it",
      all(names.index(s["kb_from"]) < names.index(s["name"]) for s in specs if s["kb_from"]))
l24 = [s for s in specs if s["plan"] == "loop24"]
check("loop24 is like-for-like with the published run: phase8 data, FloorPlan1, hybrid, oracle, gemma3:4b, "
      "pre-search action space",
      all(s["dataset"] == "phase8" and s["scenes"] == "FloorPlan1" and s["rag"] == "hybrid"
          and s["perception"] == "oracle" and s["models"] == "gemma3:4b" and s["score_search"] == "off" for s in l24))
on = {s["name"]: s for s in specs}
check("off_cold starts from an EMPTY KB and does not loop",
      on["loop24_off_cold_r1"]["kb_from"] is None and on["loop24_off_cold_r1"]["loop"] == "off")
check("on starts EMPTY and loops", on["loop24_on_r2"]["kb_from"] is None and on["loop24_on_r2"]["loop"] == "on")
check("off_warm reads the KB its OWN repetition's loop-on run left behind",
      on["loop24_off_warm_r2"]["kb_from"] == "loop24_on_r2" and on["loop24_off_warm_r2"]["loop"] == "off")
m = [s for s in specs if s["plan"] == "matrix"]
check("matrix covers all six retrieval arms", sorted(s["rag"] for s in m) == sorted(AS.RAG_ARMS))
check("matrix arms are frozen and read the teach pass", all(s["loop"] == "off" and s["kb_from"] == "loop102_on_r1"
                                                            for s in m))
x = [s for s in specs if s["plan"] == "xling"]
check("xling teaches on English only (loop on)", on["xling_teach_en"]["lang"] == "en" and on["xling_teach_en"]["loop"] == "on")
check("xling scores all six arms on Persian only, frozen",
      sorted(s["rag"] for s in x if s["name"] != "xling_teach_en") == sorted(AS.RAG_ARMS)
      and all(s["lang"] == "fa" and s["loop"] == "off" for s in x if s["name"] != "xling_teach_en"))
p = [s for s in specs if s["plan"] == "perception"]
check("perception isolates perception: rag off, loop off, FloorPlan1",
      all(s["rag"] == "off" and s["loop"] == "off" and s["scenes"] == "FloorPlan1" for s in p))
check("the candidate detector arm exists only when a candidate is given",
      "perception_candidate_detector" in names
      and "perception_candidate_detector" not in [s["name"] for s in AS.plan_runs(["perception"])])
solo = [s["name"] for s in AS.plan_runs(["matrix"], reps=1)]
check("matrix alone still schedules its teach pass first", solo[0] == "loop102_on_r1" and len(solo) == 7, str(solo))

print("\n3. the eval argv")
av = AS.eval_argv(on["loop24_on_r1"], "/s/runs/loop24_on_r1", "/s/phase8.jsonl")
check("every store is redirected into the run dir",
      av[av.index("--kb") + 1].startswith("/s/runs/loop24_on_r1") and av[av.index("--memory") + 1].startswith("/s/runs/")
      and av[av.index("--loop-reports") + 1].startswith("/s/runs/") and av[av.index("--out") + 1].startswith("/s/runs/"))
check("the argv parses with eval.py's own parser", True if __import__("eval")._build_parser().parse_args(av) else False)
av2 = AS.eval_argv(on["perception_candidate_detector"], "/s/r", "/d.jsonl", "/s/detectors/candidate")
check("the candidate arm passes --weights-dir", av2[av2.index("--weights-dir") + 1] == "/s/detectors/candidate")
check("no --weights-dir unless asked", "--weights-dir" not in av)

print("\n4. the summary")
def doc(sr, n=4, errs=0, loop_err=0):
    res = [{"error": ("loop closure: x" if i < loop_err else ("boom" if i < errs else None))} for i in range(n)]
    return {"aggregates": {"overall": {"n": n, "SR": sr, "PC": 0.5, "TDS": None},
                           "by_level": {"L1": {"SR": 1.0}}}, "results": res}
s = AS.summarize({"loop24_on_r1": doc(0.5), "loop24_on_r2": doc(0.25, loop_err=1), "matrix_off": doc(0.75)})
check("repetitions collapse into one arm", set(s["arms"]) == {"loop24_on", "matrix_off"}, str(set(s["arms"])))
check("mean and spread over repetitions", abs(s["arms"]["loop24_on"]["SR"]["mean"] - 0.375) < 1e-12
      and s["arms"]["loop24_on"]["SR"]["min"] == 0.25 and s["arms"]["loop24_on"]["SR"]["k"] == 2)
check("loop-closure errors are counted, not hidden", s["arms"]["loop24_on"]["loop_errors"] == 1)
check("a metric nobody measured stays None", s["arms"]["matrix_off"]["TDS"]["mean"] is None)
md = AS.render_summary(s)
check("the markdown renders and is ASCII", "loop24_on" in md and all(ord(c) < 128 for c in md))

print("\n5. what counts as finished")
import json, tempfile                                               # noqa: E401,E402
with tempfile.TemporaryDirectory() as td:
    check("no output -> not done", AS._done(td) is None)
    json.dump({"results": [], "interrupted": False}, open(os.path.join(td, "eval.json"), "w"))
    check("an output that scored 0 records (simulator never booted) is NOT done", AS._done(td) is None)
    json.dump({"results": [{"SR": True}], "interrupted": True}, open(os.path.join(td, "eval.json"), "w"))
    check("an interrupted run is not done", AS._done(td) is None)
    json.dump({"results": [{"SR": True}], "interrupted": False}, open(os.path.join(td, "eval.json"), "w"))
    check("a run with scored records is done", AS._done(td) is not None)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
if FAIL:
    sys.exit(1)
print("ABLATION-SUITE SELFTEST OK")
