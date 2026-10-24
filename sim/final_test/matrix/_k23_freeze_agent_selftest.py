"""Offline gate for the K23 agent-freeze writer (k23_freeze_agent.py). Readiness-independent (Astra K23-TOOLING-G3 3): nothing here
depends on whether the real kitchen has practised or rehearsed yet, so this gate stays green before, during and after the real freezes.
On a fixture root: the record the writer produces is accepted by THE READER (k23_ns.replication_freeze_status + k23_entry
.test_entry_problems), and tampering, a failing gate, an unbound Astra review or a changed summary are refused. The individual gates
are exercised on fixture journals. No simulator, no provider; the Ollama probe is injected."""
import json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import k23_ns as K
import k23_entry as E
import k23_freeze_agent as FA
import treatment as T
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def raises(f, exc=Exception, text=None):
    try:
        f()
    except exc as e:
        return text is None or text in str(e)
    return False


def jl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


ROOT = C.paths()["root"]

# ---- 1 structure (readiness-independent)
check("gate list covers every Kitchen 1 freeze gate through a disposition", FA.gate_dispositions({})["pass"] is True)
check("every K1 gate has a disposition", set(FA.K1_GATES) == set(FA.DISPOSITIONS))
check("every 'current' disposition names existing K23 gates", all(set(v[1]) <= set(FA.GATES) for v in FA.DISPOSITIONS.values()))
check("non-applicable: A12, X-iso, acceptance, registration only", sorted(k for k, v in FA.DISPOSITIONS.items() if v[0] == "not_applicable")
      == ["A12", "X-iso", "acceptance", "registration"])
check("the gate order lists every registered gate once", sorted(FA.GATE_ORDER) == sorted(FA.GATES) and len(set(FA.GATE_ORDER)) == len(FA.GATE_ORDER))
check("the size decision record is NOT bound in the code map (a later after-fp15 update must not break the K2 freeze)",
      not any("a29_size_decision" in f for f in FA.bound_files("FloorPlan15")))
check("audit count helper: int, list, None", FA._n(0) == 0 and FA._n([1, 2]) == 2 and FA._n(None) == -1 and FA._n(False) == 0)
real = FA.build("FloorPlan15", no_selftests=True, probes={"ollama": None})
check("a build of the real tree has every field the reader needs", all(k in real for k in (
    "schema", "scene", "namespaces", "split_sha256", "prereg_sha256", "code", "a29_roster_sha256", "snapshots", "treatment_id",
    "dataset_sha256", "detector_types_sha256", "gates", "sha256", "k1_gate_dispositions", "shared_evidence")))
check("the real build: the selftest gate is not run inside it and the probe was not made", real["gates"]["selftests"]["pass"] is False
      and real["gates"]["gemma"]["pass"] is False)
check("the code map covers the K23 entry points, phase modules, launchers and bound data", all(f in real["code"] for f in (
    "sim/final_test/matrix/k23_run.py", "sim/final_test/matrix/plan.py", "sim/final_test/k23_run/chain_k23.sh",
    "sim/step4/FloorPlan15.map.json", "docs/final-test/matrix/detector_types_FloorPlan15.json")))
check("write refuses a build with a failing gate", raises(lambda: FA.write("FloorPlan15", fz=real), RuntimeError, "gates failed"))
check("default root (None, the CLI path): the shared reset / families gates run and pass (Astra G3-r2)",
      FA.gate_reset({"root": None})["pass"] is True and FA.gate_families({"root": None})["pass"] is True
      and not any("gate error" in " ".join(real["gates"][k].get("problems") or []) for k in real["gates"]))

# ---- 2 individual gates on fixture journals
fx = tempfile.mkdtemp(prefix="k23fz_")
dft = os.path.join(fx, "docs", "final-test")
os.makedirs(os.path.join(dft, "k23")); os.makedirs(os.path.join(dft, "matrix"))
shutil.copy(C.paths()["split"], C.paths(fx)["split"])
for f in os.listdir(os.path.dirname(C.paths()["split"])):
    if f.startswith("inclusion_manifest"):
        shutil.copy(os.path.join(os.path.dirname(C.paths()["split"]), f), os.path.join(dft, f))
os.makedirs(os.path.join(fx, "sim", "dataset")); shutil.copy(C.paths()["dataset"], C.paths(fx)["dataset"])
ctx = {"scene": "FloorPlan15", "root": fx, "ns": FA.NS["FloorPlan15"]}
R = "rehearsal-k2-v1"
cap = [{"key": f"c{i}", "pres": f"canon:{it}:{lg}", "item": it, "lang": lg, "world": "canon", "capture_run": 1,
        "audit": {"honesty": {"violations": 0, "oracle": 0, "map_build": 0}},
        "blocks": {"none": {"text": ""}, "S*": {"text": "pkg", "ids": ["r1", "r2"], "real_tokens": 400}}}
       for i, (it, lg) in enumerate((it, lg) for it in C.load_manifest()["dev_pilot"] for lg in ("en", "fa"))]
jl(C.journal_path(R, "capture", fx), cap)
jl(C.journal_path(R, "events", fx), [{"event": "capture_start", "worlds": ["canon"]},
                                       {"event": "capture_gates", "failed": [], "partial": False,
                                        "gates": {"masking": {"pass": True, "n": len(cap)}, "leakage": {"pass": True, "n": len(cap)},
                                                  "g_iso": {"pass": True, "n": len(cap)}, "key_leak": {"pass": True}}}])
check("bilingual: identical EN / FA package ids over the registered dev set -> pass", FA.gate_bilingual(ctx)["pass"] is True)
cap2 = [dict(r, blocks=dict(r["blocks"], **{"S*": dict(r["blocks"]["S*"], ids=["r9"])})) if r["lang"] == "fa" and r is cap[1] else r for r in cap]
jl(C.journal_path(R, "capture", fx), cap2)
check("bilingual: one FA pair differs -> fail", FA.gate_bilingual(ctx)["pass"] is False)
jl(C.journal_path(R, "capture", fx), cap)
check("masking / leakage: every presentation checked -> pass", FA.gate_masking_leakage(ctx)["pass"] is True)
check("tokens: package blocks within the cap -> pass", FA.gate_tokens(ctx)["pass"] is True)
jl(C.journal_path(R, "capture", fx), cap[:-1] + [dict(cap[-1], blocks={"S*": {"text": "x", "real_tokens": 2001}})])
check("tokens: a block over 2,000 tokens -> fail", FA.gate_tokens(ctx)["pass"] is False)
jl(C.journal_path(R, "capture", fx), cap)
jl(C.journal_path("practice-k2-v1", "practice", fx), [{"audit": {"violations": 0, "counts": {"oracle": 0, "map-build": 0}, "invalid": False}}] * 10)
jl(C.journal_path(R, "exec", fx), [{"key": "e1", "audit": {"violations": 0, "counts": {"oracle": 0, "map-build": 0}, "invalid": False}}])
gi = {"namespace": R, "status": "pass", "n": 12, "n_registered": 12, "components": {"audit": {"absent_vs_present": 0}}}
gi["sha256"] = C.self_sha(gi)
json.dump(gi, open(os.path.join(C.ns_docs(R, fx), "g_iso.json"), "w"))
check("honesty: zero everywhere, G-iso audit clean -> pass", FA.gate_honesty(ctx)["pass"] is True)
check("G-iso: sealed pass report, n 12 -> pass", FA.gate_g_iso(ctx)["pass"] is True)
jl(C.journal_path(R, "exec", fx), [{"key": "e1", "audit": {"violations": 1, "counts": {"oracle": 0, "map-build": 0}}}])
check("honesty: one violation in an execution -> fail", FA.gate_honesty(ctx)["pass"] is False)
jl(C.journal_path(R, "exec", fx), [{"key": "e1", "audit": {"violations": 0, "counts": {"map-build": 0}}}])
check("honesty: a missing oracle count is never a 0 -> fail", FA.gate_honesty(ctx)["pass"] is False)
gi2 = dict(gi, status="inconclusive"); gi2["sha256"] = C.self_sha(gi2)
json.dump(gi2, open(os.path.join(C.ns_docs(R, fx), "g_iso.json"), "w"))
check("G-iso: inconclusive -> fail", FA.gate_g_iso(ctx)["pass"] is False)
cf = {"namespace": R, "freeze_grade": True, "counting": "MEASURED (Gemma server, whole conversation)", "verdict": "PASS_MEASURED",
      "capture_sha": C.sha256_file(C.journal_path(R, "capture", fx))}
json.dump(cf, open(os.path.join(C.ns_docs(R, fx), "context_fit.json"), "w"))
check("context: measured, freeze-grade, bound to the capture -> pass", FA.gate_context(ctx)["pass"] is True)
json.dump(dict(cf, counting="FORECAST (caches + conservative estimate)"), open(os.path.join(C.ns_docs(R, fx), "context_fit.json"), "w"))
check("context: a forecast is not freeze-grade evidence -> fail", FA.gate_context(ctx)["pass"] is False)
check("rehearsal gate: a FloorPlan15 capture_start without the wrapper's A39.2 order event is reported (direct-execution detection)",
      any("A39.2" in p for p in FA.gate_rehearsal(dict(ctx, ident="ID-fixture"))["problems"]))
check("namespace: an untouched test namespace -> pass", FA.gate_namespace(ctx)["pass"] is True)
jl(C.journal_path("test-k2-v1", "capture", fx), [{"key": "x"}])
check("namespace: a test capture row before the freeze -> fail", FA.gate_namespace(ctx)["pass"] is False)
shutil.copy(C.paths()["prereg"], C.paths(fx)["prereg"])
has = FA.gate_amendments(ctx)["pass"]
open(C.paths(fx)["prereg"], "a", encoding="utf-8").write("\n### A29.1 (fixture) -- applicability procedure\n")
check("amendments: refused without an A29.1 heading, passes with one", FA.gate_amendments(ctx)["pass"] is True
      and (has is False or "A29.1" in open(C.paths()["prereg"], encoding="utf-8").read()))
# ---- FA equivalence (A29.1 item 4), summary fields only
m15x = json.load(open(K.a29_files("FloorPlan15")[0], encoding="utf-8"))
json.dump(m15x, open(K.a29_files("FloorPlan15", fx)[0], "w"))
check("fa_equivalence: no result -> fail (required before the freeze)", FA.gate_fa_equivalence(ctx)["pass"] is False)
ep = FA.equivalence_path("FloorPlan15", fx); os.makedirs(os.path.dirname(ep), exist_ok=True)
json.dump({"kitchen": "FloorPlan15", "sample": ["x"] * 10, "rows": [{"item": "x", "agree": True}] * 10, "agreement": 9, "n": 10}, open(ep, "w"))
r_eq = FA.gate_fa_equivalence(ctx)
check("fa_equivalence: kitchen + registered n + agreement count -> pass, agreement reported, rows not copied",
      r_eq["pass"] is True and r_eq["agreement"] == 9 and r_eq["n"] == 10 and "rows" not in r_eq and "sample" not in r_eq)
json.dump({"kitchen": "FloorPlan17", "agreement": 9, "n": 10}, open(ep, "w"))
check("fa_equivalence: another kitchen's result -> fail", FA.gate_fa_equivalence(ctx)["pass"] is False)
json.dump({"kitchen": "FloorPlan15", "agreement": 5, "n": 6}, open(ep, "w"))
check("fa_equivalence: not the registered sample size -> fail", FA.gate_fa_equivalence(ctx)["pass"] is False)
json.dump({"kitchen": "FloorPlan15", "agreement": 11, "n": 10}, open(ep, "w"))
check("fa_equivalence: agreement outside 0..n -> fail", FA.gate_fa_equivalence(ctx)["pass"] is False)
json.dump({"kitchen": "FloorPlan15", "agreement": 0, "n": 10}, open(ep, "w"))
check("fa_equivalence: the agreement is reported, not thresholded (0 of 10 passes the gate)", FA.gate_fa_equivalence(ctx)["pass"] is True)
check("the equivalence result is bound in the record's code map (re-verified at every entry)",
      "docs/final-test/k23/applicability/FloorPlan15/equivalence_fa.json" in FA.bound_files("FloorPlan15"))
os.remove(ep); os.remove(K.a29_files("FloorPlan15", fx)[0])
# ---- the FA-examples decision gate (Astra K23-FA-EXAMPLES-r1)
import k23_fa_examples as FAX0
check("fa_examples: no decision record -> fail", FA.gate_fa_examples(ctx)["pass"] is False)
os.makedirs(os.path.dirname(FAX0.decision_path(fx)), exist_ok=True)
_d0 = {"schema": FAX0.SCHEMA, "id": FAX0.DECISION_ID, "choice": FAX0.CHOICE, "rule": FAX0.RULE, "by": "Arash", "date": "2026-10-20", "said": "x"}
_d0["sha256"] = C.self_sha(_d0)
json.dump(_d0, open(FAX0.decision_path(fx), "w"))
check("fa_examples: a valid decision, no activation yet -> pass", FA.gate_fa_examples(ctx)["pass"] is True)
FAX0.record_event("practice-k2-v1", "practice", fx)
check("fa_examples: activations under the bound decision -> pass", FA.gate_fa_examples(ctx)["pass"] is True
      and FA.gate_fa_examples(ctx)["activations"] == 1)
with open(C.journal_path("rehearsal-k2-v1", "events", fx), "a") as _f:
    _f.write(json.dumps({"event": FAX0.EVENT, "decision_sha256": "0" * 64}) + chr(10))
check("fa_examples: an activation under another decision record -> fail", FA.gate_fa_examples(ctx)["pass"] is False)
os.remove(FAX0.decision_path(fx))
check("renderer: no K23 entry point renders a prompt itself", FA.gate_renderer({"root": None, "scene": "FloorPlan15", "ns": FA.NS["FloorPlan15"]})["pass"] is True)
check("gemma: an unreachable / unknown model fails; the registered digest passes",
      FA.gate_gemma({"probes": {"ollama": None}})["pass"] is False
      and FA.gate_gemma({"probes": {"ollama": {"digest": "sha256:" + __import__("freeze").OLLAMA_DIGEST_PREFIX + "00", "model": "g"}}})["pass"] is True)

# ---- 3 writer + reader on a fixture root (every gate stubbed to pass)
m15 = json.load(open(K.a29_files("FloorPlan15")[0], encoding="utf-8"))
json.dump(m15, open(K.a29_files("FloorPlan15", fx)[0], "w"))
dec = {"schema": "k23.a29-size-decision/1", "kitchens": {"FloorPlan15": {"candidate": "all", "manifest_sha256": m15["sha256"],
                                                                          "roster_sha256": C.sha256_json(m15["candidates"]["all"]), "checkpoint": "after-k1"}},
       "checkpoints": []}
dec["sha256"] = C.self_sha(dec)
json.dump(dec, open(K.a29_files("FloorPlan15", fx)[1], "w"))
pk = os.path.join(fx, "sim", "kitchen_v3", "FloorPlan15"); os.makedirs(pk)
wv = os.path.join(fx, "sim", "weights", "v3"); os.makedirs(wv)
open(os.path.join(wv, "FloorPlan15.yolo26s.pt"), "wb").write(b"weights")
open(os.path.join(wv, "FloorPlan15.yolo26s_coco.pt"), "wb").write(b"coco")
open(os.path.join(pk, "frozen_weights.sha256"), "w").write(
    f"{C.sha256_file(os.path.join(wv, 'FloorPlan15.yolo26s.pt'))}  sim/weights/v3/FloorPlan15.yolo26s.pt\n"
    f"{C.sha256_file(os.path.join(wv, 'FloorPlan15.yolo26s_coco.pt'))}  sim/weights/v3/FloorPlan15.yolo26s_coco.pt\n")
import hashlib
_rb = open(os.path.join(pk, "frozen_weights.sha256"), "rb").read()
json.dump({"coco_pending": False, "test_evaluation_blocked": False, "record_sha256": hashlib.sha256(_rb).hexdigest(), "n_files": 2,
           "coco_append": {"coco": "sim/weights/v3/FloorPlan15.yolo26s_coco.pt"}}, open(os.path.join(pk, "frozen_weights.meta.json"), "w"))
snap = T.source_dir("test-k2-v1", "S*", fx); os.makedirs(snap)
open(os.path.join(snap, "practice_manifest.json"), "w").write("{}")
code_dir = os.path.join(fx, "sim", "final_test", "matrix"); os.makedirs(code_dir)
open(os.path.join(code_dir, "k23_run.py"), "w").write("# fixture\n")
json.dump({"types": ["Apple", "Bowl"]}, open(os.path.join(dft, "matrix", "detector_types_FloorPlan15.json"), "w"))
os.remove(C.journal_path("test-k2-v1", "capture", fx))


def make_stub(name):
    def f(c):
        if name == "roster":
            c["roster"] = K.a29_roster("test-k2-v1", c["root"])
        if name == "practice":
            c["snapshots"] = {"S*": C.dir_sha(T.source_dir("test-k2-v1", "S*", c["root"]))}
        if name == "dataset":
            c["dataset_sha256"] = C.sha256_file(C.paths(c["root"])["dataset"])
        return FA.g(True, stub=True)
    return f


orig_gates = dict(FA.GATES)
import freeze as F
orig_closure = F.import_closure
F.import_closure = lambda base, entries: (["sim/final_test/matrix/k23_run.py"], [], [])
orig_bound = FA.bound_files
FA.bound_files = lambda scene: ("docs/final-test/matrix/detector_types_FloorPlan15.json", "docs/final-test/k23/decisions/K23-FA-EXAMPLES.json")
import k23_fa_examples as FAX
os.makedirs(os.path.dirname(FAX.decision_path(fx)), exist_ok=True)
_dec = {"schema": FAX.SCHEMA, "id": FAX.DECISION_ID, "choice": FAX.CHOICE, "rule": FAX.RULE, "by": "Arash", "date": "2026-10-20", "said": "fixture"}
_dec["sha256"] = C.self_sha(_dec)
json.dump(_dec, open(FAX.decision_path(fx), "w"))
FA.GATES.update({k: make_stub(k) for k in FA.GATE_ORDER})
try:
    fz = FA.build("FloorPlan15", fx, now="T0")
    check("fixture: every (stubbed) gate passes", fz["all_gates_pass"] is True)
    FA.write("FloorPlan15", fx, fz=fz)
    p = K.freeze_path("FloorPlan15", fx)
    check("written: the freeze and its generated summary", os.path.exists(p) and open(FA.summary_path("FloorPlan15", fx), encoding="utf-8").read() == FA.summary_text(fz))
    st = K.replication_freeze_status("test-k2-v1", fx)
    check("reader before sign-off: refused only for the missing sign-off", not st["ok"] and st["reasons"] and all("sign-off" in r for r in st["reasons"]))
    rvd = os.path.join(dft, "k23", "_astra"); os.makedirs(rvd)
    sha16 = fz["sha256"][:16]

    def review(name, answer):
        path = os.path.join(rvd, name)
        open(path, "w").write("=== PROMPT ===\nfreeze FloorPlan15 " + sha16 + "\n\n=== ANSWER (verbatim) ===\n" + answer + "\n")
        return path
    check("presign refuses CHANGES REQUIRED", raises(lambda: FA.presign("FloorPlan15", review("a.txt", f"FloorPlan15 {sha16}\nRULING: CHANGES REQUIRED"), fx), RuntimeError))
    check("presign refuses an ACCEPTED answer that does not name THIS freeze (an older acceptance)",
          raises(lambda: FA.presign("FloorPlan15", review("b.txt", "FloorPlan15 looks fine.\nRULING: ACCEPTED"), fx), RuntimeError, "self-sha prefix"))
    open(FA.summary_path("FloorPlan15", fx), "a").write("edited\n")
    check("presign refuses an edited summary", raises(lambda: FA.presign("FloorPlan15", review("c.txt", f"FloorPlan15 freeze {sha16} ok.\nRULING: ACCEPTED"), fx),
                                                      RuntimeError, "summary"))
    open(FA.summary_path("FloorPlan15", fx), "w").write(FA.summary_text(fz))
    FA.presign("FloorPlan15", review("d.txt", f"FloorPlan15 freeze {sha16}: all gates pass.\nRULING: ACCEPTED"), fx)
    check("reader after the bound standing pre-signature: ok", K.replication_freeze_status("test-k2-v1", fx)["ok"] is True)
    check("K23 test entry checks pass on the same record (identity, dataset, COCO)", E.test_entry_problems("test-k2-v1", fx) == [])
    fz2 = json.load(open(p, encoding="utf-8")); fz2["a29_roster_sha256"] = "x"; json.dump(fz2, open(p, "w"))
    check("tampering is refused by the reader (self-sha)", any("self-sha" in r for r in K.replication_freeze_status("test-k2-v1", fx)["reasons"]))
    FA.write("FloorPlan15", fx, fz=fz, supersede=True)
    check("--supersede archives the old record", os.listdir(os.path.join(dft, "k23", "freeze_history")))
    FA.presign("FloorPlan15", os.path.join(rvd, "d.txt"), fx)
    open(os.path.join(snap, "practice_manifest.json"), "w").write('{"changed": 1}')
    check("a changed practice snapshot is refused by the reader", any("snapshot" in r for r in K.replication_freeze_status("test-k2-v1", fx)["reasons"]))
    open(os.path.join(snap, "practice_manifest.json"), "w").write("{}")
    open(os.path.join(code_dir, "k23_run.py"), "w").write("# edited\n")
    check("a changed frozen code file is refused by the reader", any("code changed" in r for r in K.replication_freeze_status("test-k2-v1", fx)["reasons"]))
    open(os.path.join(code_dir, "k23_run.py"), "w").write("# fixture\n")
    json.dump({"types": ["Apple"]}, open(os.path.join(dft, "matrix", "detector_types_FloorPlan15.json"), "w"))
    check("a changed bound data file (type list) is refused by the reader", any("code changed" in r for r in K.replication_freeze_status("test-k2-v1", fx)["reasons"]))
    json.dump({"types": ["Apple", "Bowl"]}, open(os.path.join(dft, "matrix", "detector_types_FloorPlan15.json"), "w"))
    check("restored: ok again", K.replication_freeze_status("test-k2-v1", fx)["ok"] is True)
    check("record_arash_check needs Arash + date + words", raises(lambda: FA.record_arash_check("FloorPlan15", "someone", "2026-10-20", "x", fx), RuntimeError))
    FA.record_arash_check("FloorPlan15", "Arash", "2026-10-20", "I sign the Kitchen 2 freeze", fx)
    check("Arash's check keeps the record readable", K.replication_freeze_status("test-k2-v1", fx)["ok"] is True)
    FA.GATES["budget"] = lambda c: FA.g(False, problems=["over"])
    check("one failing gate: not written", raises(lambda: FA.write("FloorPlan15", fx, fz=FA.build("FloorPlan15", fx), supersede=True), RuntimeError, "budget"))
finally:
    FA.GATES.clear(); FA.GATES.update(orig_gates)
    F.import_closure = orig_closure
    FA.bound_files = orig_bound

# ---- 4 the real roster gate: FloorPlan17 needs the after-fp15 selection
m17 = dict(m15, kitchen="FloorPlan17"); m17["sha256"] = C.self_sha(m17)
json.dump(m17, open(K.a29_files("FloorPlan17", fx)[0], "w"))
dec17 = dict(dec, kitchens=dict(dec["kitchens"], FloorPlan17={"candidate": "all", "manifest_sha256": m17["sha256"],
                                                             "roster_sha256": C.sha256_json(m17["candidates"]["all"]), "checkpoint": "after-k1"}))
dec17["sha256"] = C.self_sha(dec17)
json.dump(dec17, open(K.a29_files("FloorPlan17", fx)[1], "w"))
r = FA.gate_roster({"scene": "FloorPlan17", "root": fx, "ns": FA.NS["FloorPlan17"]})
check("FloorPlan17 freeze refused while its size is only the after-k1 selection (A29 item 6)", not r["pass"] and any("after-fp15" in x for x in r["problems"]))
r = FA.gate_roster({"scene": "FloorPlan15", "root": fx, "ns": FA.NS["FloorPlan15"]})
check("FloorPlan15 roster gate passes on the after-k1 decision", r["pass"] and r["presentations"] == 2 * len(m15["candidates"]["all"]))
shutil.rmtree(fx, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
