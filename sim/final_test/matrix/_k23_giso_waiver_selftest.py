"""Offline gate for k23_giso_waiver.py (Arash's written waiver of the inconclusive Kitchen 3 G-iso, narrowed claim, A20.1) and the agent
freeze's gate_g_iso path: accepted ONLY for FloorPlan17, ONLY status 'inconclusive' with diff == [] / no absent-vs-present difference, ONLY
for the archived result the decision names, ONLY with a valid record of Arash's; a 'fail' never; the freeze summary states the narrowed
claim; the overnight runner applies the same conditions before it resumes rehearsal-k3-v1 at its context stage. Temporary roots only."""
import copy, hashlib, json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "k23_run"))
import common as C
import k23_giso_waiver as GW
import k23_freeze_agent as FA
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


REP = {"schema": "ftm.g_iso/2", "rule": "A20.1", "namespace": GW.NS, "status": "inconclusive", "n": 12, "n_registered": 12, "diff": [],
       "diff_components": {}, "blocked": ["canon:g-l1-place-bowl-counter-top:en", "canon:g-l1-place-bowl-counter-top:fa",
                                          "canon:g-l2-microwave-run-egg:en"],
       "control_noise": ["canon:g-l1-place-bowl-counter-top:fa"],
       "components": {"request": {"absent_vs_present": 0, "absent_vs_control_absent": 0}, "audit": {"absent_vs_present": 0},
                      "status": {"absent_vs_present": 1}, "goal_reached": {"absent_vs_present": 1}},
       "inconclusive_reasons": ["blocked run", "request probe differs from the Phase A none-arm bytes", "control differs outside (iv)"]}


def seal(rep):
    r = copy.deepcopy(rep)
    r.pop("sha256", None)
    r["sha256"] = C.self_sha(r)
    return r


def fixture(rep=REP, archive_same=True, decision=True, **dec_over):
    t = tempfile.mkdtemp(prefix="k23gw_")
    rp = GW.report_path(t)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    body = json.dumps(seal(rep), indent=1)
    open(rp, "w", encoding="utf-8", newline="\n").write(body)
    ap = os.path.join(t, *GW.ARCHIVE.split("/"))
    os.makedirs(os.path.dirname(ap), exist_ok=True)
    open(ap, "w", encoding="utf-8", newline="\n").write(body if archive_same else body + "\n")
    if decision:
        d = {"schema": GW.SCHEMA, "id": GW.DECISION_ID, "choice": GW.CHOICE, "rule": GW.RULE, "claim": GW.CLAIM, "scene": GW.SCENE,
             "namespace": GW.NS, "archive": GW.ARCHIVE, "g_iso_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "by": "Arash",
             "date": "2026-10-20", "said": "Written waiver, narrowed claim (Recommended)"}
        d.update(dec_over)
        d.pop("sha256", None)
        d["sha256"] = C.self_sha(d)
        os.makedirs(os.path.dirname(GW.decision_path(t)), exist_ok=True)
        json.dump(d, open(GW.decision_path(t), "w", encoding="utf-8"))
    return t


t = fixture()
check("a valid record of Arash's naming the archived inconclusive result (diff []) -> the waiver holds", GW.waiver_problems(t) == [])
ctx = {"root": t, "scene": "FloorPlan17", "ns": FA.NS["FloorPlan17"]}
gg = FA.gate_g_iso(ctx)
check("freeze gate_g_iso: FloorPlan17 inconclusive + valid waiver -> PASS, marked waived, with the narrowed claim and the decision sha",
      gg["pass"] and gg.get("waived") is True and gg.get("narrowed_claim") == GW.CLAIM and gg.get("waiver_decision_sha256") == GW.decision_sha(t))
fz = {"scene": "FloorPlan17", "namespaces": ["test-k3-v1"], "created": "x", "sha256": "f" * 64, "gate_order": ["g_iso"], "gates": {"g_iso": gg},
      "prereg_sha256": "p" * 64, "split_sha256": "s" * 64, "code": {}}
check("the freeze summary states the narrowed claim (A20.1)", GW.CLAIM in FA.summary_text(fz) and "Narrowed claim" in FA.summary_text(fz))
check("bound_files binds the waiver record and the archived result for FloorPlan17 only",
      f"docs/final-test/k23/decisions/{GW.DECISION_ID}.json" in FA.bound_files("FloorPlan17") and GW.ARCHIVE in FA.bound_files("FloorPlan17")
      and not any("GISO" in f or "K3-GISO" in f for f in FA.bound_files("FloorPlan15")))
import k23_overnight as O
st = O.State(t, type("S", (), {"read": staticmethod(lambda p: None), "win_cmdlines": staticmethod(lambda: [])})())
gi = st.giso(GW.NS)
check("the overnight runner sees the same waiver (resume rehearsal-k3-v1 at its context stage)", gi["status"] == "inconclusive" and gi["waived"])
shutil.rmtree(t, ignore_errors=True)

for name, kw, want in (("no decision record", dict(decision=False), "no decision record"),
                       ("a record not by Arash", dict(by="someone"), "Arash"),
                       ("an altered claim", dict(claim="Kitchen 3 memory isolation shown."), "claim"),
                       ("a record naming another result", dict(g_iso_sha256="0" * 64), "archived"),
                       ("an archive that differs from the report", dict(archive_same=False), "archived")):
    t = fixture(**kw)
    wp = GW.waiver_problems(t)
    check(f"waiver refused: {name}", wp != [] and any(want in w for w in wp) and not FA.gate_g_iso({"root": t, "scene": "FloorPlan17",
                                                                                                    "ns": FA.NS["FloorPlan17"]})["pass"])
    shutil.rmtree(t, ignore_errors=True)

for name, over, want in (("status fail", dict(status="fail"), "never"), ("a non-empty diff", dict(diff=["canon:x:en"]), "absent-vs-present"),
                         ("an absent-vs-present request difference", dict(components=dict(REP["components"], request={"absent_vs_present": 1})),
                          "absent-vs-present"),
                         ("n 11", dict(n=11), "n 11")):
    t = fixture(rep=dict(REP, **over))
    wp = GW.waiver_problems(t)
    check(f"never waivable: {name}", any(want in w for w in wp) and not FA.gate_g_iso({"root": t, "scene": "FloorPlan17",
                                                                                         "ns": FA.NS["FloorPlan17"]})["pass"])
    shutil.rmtree(t, ignore_errors=True)

t = fixture(rep=dict(REP, namespace="rehearsal-k2-v1"))
rp2 = os.path.join(C.ns_docs("rehearsal-k2-v1", t), "g_iso.json")
os.makedirs(os.path.dirname(rp2), exist_ok=True)
shutil.copyfile(GW.report_path(t), rp2)
check("Kitchen 2: an inconclusive G-iso is never waived by this record (gate fails)", not FA.gate_g_iso({"root": t, "scene": "FloorPlan15",
                                                                                                       "ns": FA.NS["FloorPlan15"]})["pass"])
shutil.rmtree(t, ignore_errors=True)
t = fixture(rep=dict(REP, status="pass", blocked=[], control_noise=[]), decision=False)
gg = FA.gate_g_iso({"root": t, "scene": "FloorPlan17", "ns": FA.NS["FloorPlan17"]})
check("a passed G-iso still passes without any waiver (unchanged rule), not marked waived", gg["pass"] and not gg.get("waived"))
shutil.rmtree(t, ignore_errors=True)
check("the module never edits the preregistration and refuses --write without Arash's words", "preregistration.md" not in
      open(os.path.join(HERE, "k23_giso_waiver.py"), encoding="utf-8").read().split('"""', 2)[2] and GW.main(["--write", "--by", "x"]) == 2)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
