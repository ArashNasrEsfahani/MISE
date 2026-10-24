"""Offline gate for the A29 item 6 size decision writer (k23_size_decision.py), on a fixture root (spend and means injected; no ledger,
no provider)."""
import copy, json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import k23_ns as K
import k23_size_decision as SD
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


ROOT = C.paths()["root"]
fx = tempfile.mkdtemp(prefix="k23size_")
d_ft = os.path.join(fx, "docs", "final-test")
os.makedirs(os.path.join(d_ft, "k23")); os.makedirs(os.path.join(d_ft, "matrix", "test-v1"))
shutil.copy(C.paths()["split"], C.paths(fx)["split"])
for f in os.listdir(os.path.dirname(C.paths()["split"])):
    if f.startswith("inclusion_manifest"):
        shutil.copy(os.path.join(os.path.dirname(C.paths()["split"]), f), os.path.join(d_ft, f))
os.makedirs(os.path.join(fx, "sim", "dataset")); shutil.copy(C.paths()["dataset"], C.paths(fx)["dataset"])
real15 = K.a29_files("FloorPlan15")[0]
m15 = json.load(open(real15, encoding="utf-8"))
m17 = copy.deepcopy(m15); m17["kitchen"] = "FloorPlan17"; m17["sha256"] = C.self_sha(m17)      # stand-in manifest (same sizes)
json.dump(m15, open(K.a29_files("FloorPlan15", fx)[0], "w")); json.dump(m17, open(K.a29_files("FloorPlan17", fx)[0], "w"))
MEANS = {"O0": {"mean": 0.01874254, "usd": 4.68, "billed": 250, "cells": 250}, "O+M": {"mean": 0.01892682, "usd": 4.73, "billed": 250, "cells": 250}}
SPEND = {"committed": 29.964269, "settled": 29.88, "open": 0.08, "ledgers": {}, "errors": []}
seal = os.path.join(C.ns_docs("test-v1", fx), "score.seal.json")

# ---- preconditions
check("after-k1 refused before the Kitchen 1 score is sealed", raises(lambda: SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "sealed"))
json.dump({"complete": True}, open(seal, "w"))
check("after-fp15 refused without an after-k1 decision", raises(lambda: SD.decide("after-fp15", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "after-k1"))
check("unresolved accounting refused", raises(lambda: SD.decide("after-k1", root=fx, spend=dict(SPEND, errors=["x.db"]), means=MEANS), SD.DecisionRefused))
os.rename(K.a29_files("FloorPlan17", fx)[0], K.a29_files("FloorPlan17", fx)[0] + ".x")
check("both frozen manifests are required", raises(lambda: SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "FloorPlan17"))
os.rename(K.a29_files("FloorPlan17", fx)[0] + ".x", K.a29_files("FloorPlan17", fx)[0])
check("the freeze-time projection record is required (the predecessor ceiling)",
      raises(lambda: SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "freeze-time"))
proj_real = json.load(open(os.path.join(ROOT, *SD.INITIAL_PROJECTION.split("/")), encoding="utf-8"))
proj_fx = os.path.join(fx, *SD.INITIAL_PROJECTION.split("/"))
only120 = {k: dict(v, gate=dict(v["gate"], gate_pass=(k != "250"))) for k, v in proj_real.items()}
json.dump(only120, open(proj_fx, "w"))
d120 = SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS)
check("after-k1 never exceeds the freeze-time selection (only 120 passed then -> first_60 at most)",
      d120["checkpoints"][-1]["selected"] == "first_60" and [t["size"] for t in d120["checkpoints"][-1]["trials"]] == ["first_60", "first_50"]
      and d120["checkpoints"][-1]["ceiling"]["candidate"] == "first_60")
json.dump({k: dict(v, gate=dict(v["gate"], gate_pass=False)) for k, v in proj_real.items()}, open(proj_fx, "w"))
check("no size passed at the freeze: refused (testing does not begin)", raises(lambda: SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS),
                                                                               SD.DecisionRefused, "does not begin"))
json.dump(proj_real, open(proj_fx, "w"))
check("the real freeze-time record allows 'all' (250 passed)", SD.initial_ceiling(ROOT)["candidate"] == "all")

# ---- after-k1
doc = SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS, now="T1")
last = doc["checkpoints"][-1]
check("after-k1 records its ceiling source", last["ceiling"]["source"] == SD.INITIAL_PROJECTION and last["interpretation"] is None)
check("after-k1 at 30 USD spent: 'all' for both kitchens", last["selected"] == "all" and all(doc["kitchens"][s]["candidate"] == "all" for s in ("FloorPlan15", "FloorPlan17")))
check("projection = spend + 2 kitchens x 200 pres x (O0 + O+M means) x 1.1", abs(last["trials"][0]["projection_usd"] - round(29.964269 + 2 * 200 * (0.01874254 + 0.01892682) * 1.1, 4)) < 1e-3)
check("record: schema, self-sha, roster shas of the selected candidate", doc["schema"] == SD.SCHEMA and doc["sha256"] == C.self_sha(doc)
      and doc["kitchens"]["FloorPlan15"]["roster_sha256"] == C.sha256_json(m15["candidates"]["all"])
      and doc["kitchens"]["FloorPlan15"]["manifest_sha256"] == m15["sha256"])
check("trials are the permitted sizes, largest first", [t["size"] for t in last["trials"]] == ["all", "first_60", "first_50"])
doc66 = SD.decide("after-k1", root=fx, spend=dict(SPEND, committed=66.0), means=MEANS)
check("at 66 USD spent 'all' does not fit: first_60 is selected for both", doc66["checkpoints"][-1]["selected"] == "first_60")
check("at 75 USD spent nothing fits: planning pauses (refused, reported not run)",
      raises(lambda: SD.decide("after-k1", root=fx, spend=dict(SPEND, committed=75.0), means=MEANS), SD.DecisionRefused, "not run"))
check("the provider figure is used when larger", SD.decide("after-k1", root=fx, spend=dict(SPEND, committed=30.0), means=MEANS,
                                                         provider_usd=66.0)["checkpoints"][-1]["selected"] == "first_60")
sp_r = dict(SPEND, ledgers={"sim/final_test/matrix/work/rehearsal-k2-v1/ledger.db": {"settled": 0.5, "open": 0, "holds": 0, "liability": 0.5}})
t = SD.decide("after-k1", root=fx, spend=sp_r, means=MEANS)["checkpoints"][-1]["trials"][0]["kitchens"]
check("rehearsal spend already billed is not counted twice (allowance reduced)",
      t["FloorPlan15"]["rehearsal_allowance_usd"] < t["FloorPlan17"]["rehearsal_allowance_usd"])
os.makedirs(C.ns_docs("test-k2-v1", fx), exist_ok=True)
open(C.journal_path("test-k2-v1", "plan", fx), "w").write('{"key": "x"}\n')
check("after-k1 refused once test-k2-v1 has plan rows", raises(lambda: SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "plan rows"))
os.remove(C.journal_path("test-k2-v1", "plan", fx))

# ---- write the after-k1 record (first_60 case) and read rosters through k23_ns
json.dump(doc66, open(SD.record_path(fx), "w"))
check("k23_ns.a29_roster reads the decision", K.a29_roster("test-k2-v1", fx) == m15["candidates"]["first_60"]
      and K.a29_roster("test-k3-v1", fx) == m17["candidates"]["first_60"])
check("check(): record intact, rosters match", SD.check(fx).get("ok") is True)
check("after-k1 refused when a record exists", raises(lambda: SD.decide("after-k1", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "already"))

# ---- after-fp15
check("after-fp15 refused before test-k2-v1's full plan_first_complete", raises(lambda: SD.decide("after-fp15", root=fx, spend=SPEND, means=MEANS),
                                                                                  SD.DecisionRefused, "plan_first_complete"))
open(C.journal_path("test-k2-v1", "events", fx), "w").write(json.dumps({"event": "plan_first_complete", "scope": "full"}) + "\n")
d2 = SD.decide("after-fp15", root=fx, spend=SPEND, means=MEANS, now="T2")
check("after-fp15: FloorPlan17 never grows (first_60 stays, although 'all' would fit)", d2["kitchens"]["FloorPlan17"]["candidate"] == "first_60"
      and [t["size"] for t in d2["checkpoints"][-1]["trials"]] == ["first_60", "first_50"])
check("after-fp15: FloorPlan15's entry unchanged", d2["kitchens"]["FloorPlan15"] == doc66["kitchens"]["FloorPlan15"])
check("after-fp15 measures test-k2-v1", d2["checkpoints"][-1]["measure_ns"] == "test-k2-v1" and len(d2["checkpoints"]) == 2)
check("after-fp15 records the 'after FloorPlan15' interpretation", "paid planning" in str(d2["checkpoints"][-1]["interpretation"]))
d3 = SD.decide("after-fp15", root=fx, spend=dict(SPEND, committed=75.5), means=MEANS)
check("after-fp15 may shrink FloorPlan17 to first_50", d3["kitchens"]["FloorPlan17"]["candidate"] == "first_50")
os.makedirs(C.ns_docs("test-k3-v1", fx), exist_ok=True)
open(C.journal_path("test-k3-v1", "plan", fx), "w").write('{"key": "x"}\n')
check("after-fp15 refused once test-k3-v1 has plan rows", raises(lambda: SD.decide("after-fp15", root=fx, spend=SPEND, means=MEANS), SD.DecisionRefused, "plan rows"))

# ---- tampering, means
bad = dict(doc66); bad["kitchens"] = dict(doc66["kitchens"], FloorPlan15=dict(doc66["kitchens"]["FloorPlan15"], candidate="all"))
json.dump(bad, open(SD.record_path(fx), "w"))
check("an edited record is refused", raises(lambda: SD.load_record(fx), SD.DecisionRefused, "self-sha"))
rep = {"cells": {"test-v1|plan|O0|canon:a:en|r0": {"settled_usd": 0.02}, "test-v1|plan|O0|canon:b:fa|r0": {"settled_usd": 0.0},
                 "test-v1|plan|O+M|canon:a:en|r0": {"settled_usd": 0.03}, "test-v1|plan|O0|reloc:a:en|r0": {"settled_usd": 9.0},
                 "test-v1|plan|O0|canon:a:en|r1": {"settled_usd": 9.0}, "test-v1|plan|G0|canon:a:en|r0": {"settled_usd": 9.0}},
       "open_total": 0, "ledger_open_usd": 0.0}
mm = SD.arm_means("test-v1", report=rep)
check("means: canonical r0 cells of O0 / O+M only; denominator = billed cells", mm["O0"]["mean"] == 0.02 and mm["O0"]["cells"] == 2
      and mm["O0"]["billed"] == 1 and mm["O+M"]["mean"] == 0.03)
check("open reservations refuse the measurement", raises(lambda: SD.arm_means("test-v1", report=dict(rep, open_total=1)), SD.DecisionRefused, "open"))
check("no billed cell refuses the measurement", raises(lambda: SD.arm_means("x", report={"cells": {}, "open_total": 0}), SD.DecisionRefused))
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    check("CLI refuses without a mode", raises(lambda: SD.main([]), SystemExit))

shutil.rmtree(fx, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
