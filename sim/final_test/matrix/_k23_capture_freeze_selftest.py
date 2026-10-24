"""Offline gate for k23_capture_freeze.py: it reproduces the rehearsal-k3-v1 capture abort of 2026-10-20 20:28 (capture.py's own artefact step
on the historical Kitchen 1 freeze) and shows the capture passes that step on the re-pin view, and that every other drift still aborts."""
import copy, json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import capture as CP
import k23_capture_freeze as KCF
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


LIVE_ABORT = "frozen artefact(s) changed: artefact exec_contract.json changed after the freeze; artefact ft_v2.json changed after the freeze"
TWO = ["artefact exec_contract.json changed after the freeze", "artefact ft_v2.json changed after the freeze"]
ROOT = C.paths(None)["root"]
orig_load = getattr(CP.load_freeze, "k23_orig", CP.load_freeze)


def step(fz, root=None, need=("S*",)):
    """capture.py's artefact step verbatim (Capturer.run: freeze_status None -> snapshot + artefact problems -> CaptureAbort message)."""
    if not fz or fz.get("freeze_status") is not None:
        return None
    bad = C.snapshot_freeze_problems(list(need), root, fz) + C.freeze_artifact_problems(fz, root)
    return ("frozen artefact(s) changed: " + "; ".join(bad[:6])) if bad else ""


hist = orig_load(None)
if C.sha256_file(C.paths(None)["freeze"]) == KCF.HIST_FREEZE_SHA:
    msg = step(hist)
    check("the historical freeze reproduces the live abort (both re-pinned artefacts named)", msg is not None and all(t in msg for t in TWO))
    hp = C.snapshot_freeze_problems(["S*"], None, hist) + C.freeze_artifact_problems(hist, None)
    if set(hp) == set(TWO):
        check("... byte for byte the message rehearsal-k3-v1 logged", msg == LIVE_ABORT)
    v, info = KCF.view(hist, None)
    check("the re-pin view applies on the real tree (historical freeze 6ab8ee5f, Arash's re-pin record)", info["applied"] and not info["why"])
    vp = C.snapshot_freeze_problems(["S*"], None, v) + C.freeze_artifact_problems(v, None)
    check("the view removes exactly the two re-pinned artefact problems and nothing else", vp == [p for p in hp if p not in TWO])
    check("the view differs from the freeze only in those two artefact shas", {k: x for k, x in v.items() if k != "artefacts"} ==
          {k: x for k, x in hist.items() if k != "artefacts"} and {k for k in v["artefacts"] if v["artefacts"][k] != hist["artefacts"][k]}
          == set(KCF.ARTEFACTS) and all(v["artefacts"][k]["sha256"] == KCF.ARTEFACTS[k][2] for k in KCF.ARTEFACTS))
    check("the historical freeze object itself is untouched (deep copy)", all(hist["artefacts"][k]["sha256"] == KCF.ARTEFACTS[k][1] for k in KCF.ARTEFACTS))
    check("the view carries no new key and keeps freeze_status / arash_check as they are", set(v) == set(hist)
          and v.get("freeze_status") is None and v.get("arash_check") == hist.get("arash_check"))
    # a third drift still aborts under the view
    other = next(k for k in sorted(hist["artefacts"]) if k not in KCF.ARTEFACTS)
    h3 = copy.deepcopy(hist); h3["artefacts"][other]["sha256"] = "0" * 64
    v3, i3 = KCF.view(h3, None)
    check("a third changed artefact still aborts under the view", i3["applied"] and f"artefact {other} changed after the freeze" in step(v3))
    # an entry that is not exactly the historical one -> no substitution at all
    h4 = copy.deepcopy(hist); h4["artefacts"]["ft_v2.json"]["path"] = "sim/planners/ft_v2_other.json"
    v4, i4 = KCF.view(h4, None)
    check("an artefact entry with another path -> no substitution (all or nothing)", not i4["applied"] and v4 is h4)
    h5 = copy.deepcopy(hist); h5["artefacts"]["exec_contract.json"]["sha256"] = KCF.ARTEFACTS["exec_contract.json"][2]
    check("an entry that is not the 'before' sha -> no substitution", not KCF.view(h5, None)[1]["applied"])
    # install: capture.load_freeze returns the view in this process; the original stays reachable
    saved = CP.load_freeze
    try:
        inf = KCF.install(None)
        check("install() patches capture.load_freeze (view applied) and keeps the original", inf["applied"] and getattr(CP.load_freeze, "k23_capture_freeze", False)
              and CP.load_freeze.k23_orig is orig_load and CP.load_freeze(None)["artefacts"]["ft_v2.json"]["sha256"] == KCF.ARTEFACTS["ft_v2.json"][2])
        check("the capture's artefact step on what capture.main now loads passes the two re-pinned artefacts", not any(t in (step(CP.load_freeze(None)) or "") for t in TWO))
        check("install() is idempotent (one wrapper)", KCF.install(None)["applied"] and CP.load_freeze.k23_orig is orig_load)
    finally:
        CP.load_freeze = saved
        KCF._STATE["installed"] = False
else:
    print("note: this tree's freeze.json is not the historical 6ab8ee5f; real-tree reproduction skipped")

# ---- fail-closed cases on a temporary root (copies of the real records)
t = tempfile.mkdtemp(prefix="k23capfz_")
for rel in ("docs/final-test/freeze.json", KCF.REPIN, "sim/planners/exec_contract.json", "sim/planners/ft_v2.json"):
    os.makedirs(os.path.dirname(os.path.join(t, *rel.split("/"))), exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, *rel.split("/")), os.path.join(t, *rel.split("/")))
fzt = json.load(open(os.path.join(t, "docs", "final-test", "freeze.json"), encoding="utf-8"))
check("temp root: the view applies on byte-identical copies", KCF.view(fzt, t)[1]["applied"])
vt = KCF.view(fzt, t)[0]
check("temp root: under the view the two re-pinned files are not reported", not any(x in C.freeze_artifact_problems(vt, t) for x in TWO))
open(os.path.join(t, "sim", "planners", "exec_contract.json"), "ab").write(b" ")
check("a re-pinned file whose bytes differ from the approved 'after' sha still aborts under the view",
      "artefact exec_contract.json changed after the freeze" in C.freeze_artifact_problems(vt, t))
rp = json.load(open(os.path.join(t, *KCF.REPIN.split("/")), encoding="utf-8"))
bad = copy.deepcopy(rp); bad["identity_file_shas_after"]["sim/planners/ft_v2.json"] = "1" * 64
json.dump(bad, open(os.path.join(t, *KCF.REPIN.split("/")), "w", encoding="utf-8"))
check("a re-pin record with another 'after' sha -> no substitution", not KCF.view(fzt, t)[1]["applied"] and KCF.repin_problems(t))
bad = copy.deepcopy(rp); bad["authorization"]["by"] = "someone"
json.dump(bad, open(os.path.join(t, *KCF.REPIN.split("/")), "w", encoding="utf-8"))
check("a re-pin record not authorized by Arash -> no substitution", not KCF.view(fzt, t)[1]["applied"])
json.dump(rp, open(os.path.join(t, *KCF.REPIN.split("/")), "w", encoding="utf-8"))
check("the restored re-pin record content applies again (content, not bytes)", KCF.view(fzt, t)[1]["applied"])
open(os.path.join(t, "docs", "final-test", "freeze.json"), "ab").write(b"\n")
v6, i6 = KCF.view(fzt, t)
check("a freeze.json that is not byte-identical to 6ab8ee5f -> no substitution, capture stays strict", not i6["applied"] and v6 is fzt)
saved = CP.load_freeze
try:
    KCF._STATE["installed"] = False
    inf = KCF.install(t)
    check("install() on another freeze: applied False, capture.load_freeze returns the freeze unchanged",
          not inf["applied"] and CP.load_freeze(t)["artefacts"]["ft_v2.json"]["sha256"] == KCF.ARTEFACTS["ft_v2.json"][1])
finally:
    CP.load_freeze = saved
    KCF._STATE["installed"] = False

# ---- the audit trail + the agent-freeze check
ns = "rehearsal-k3-v1"
good = {"applied": True, "historical_freeze_sha256": KCF.HIST_FREEZE_SHA, "repin_sha256": "x",
        "substitutions": {lb: {"path": pa, "historical": b, "repinned": a} for lb, (pa, b, a) in KCF.ARTEFACTS.items()}}
KCF.record_event(ns, good, t)
jp = C.journal_path(ns, "events", t)
J = C.Journal(jp)
check("record_event writes the view event into the namespace's events journal", [e["event"] for e in J.rows()] == [KCF.EVENT])
J.append(C.make_record("events", event="capture_start", phase="capture"))
check("capture_start after an applied view -> no problem", KCF.journal_problems(ns, t) == [])
J.append(C.make_record("events", event="capture_start", phase="capture"))
check("a second capture_start without a new view event -> problem", len(KCF.journal_problems(ns, t)) == 1)
KCF.record_event(ns, dict(good, applied=False, substitutions=None), t)
J.append(C.make_record("events", event="capture_start", phase="capture"))
check("a capture_start after an UNapplied view -> problem", len(KCF.journal_problems(ns, t)) == 2)
check("no events journal -> no problem (nothing captured yet)", KCF.journal_problems("rehearsal-k2-v1", t) == [])
shutil.rmtree(t, ignore_errors=True)

# ---- wiring
src = open(os.path.join(HERE, "k23_run.py"), encoding="utf-8").read()
check("k23_run's capture path installs the view and records it before capture.main", "KCF.install(root)" in src and "KCF.record_event(ns, info, root)" in src
      and src.index("KCF.install(root)") < src.index("return int(admit_capture(ns).main(margv))"))
fa = open(os.path.join(HERE, "k23_freeze_agent.py"), encoding="utf-8").read()
check("the agent freeze binds the module, runs this selftest and checks the rehearsal journal", '"k23_capture_freeze.py"' in fa
      and "_k23_capture_freeze_selftest.py" in fa and "KCF.journal_problems(ns, root)" in fa)
check("the module never writes freeze.json (no open-for-write of the freeze path)", "paths(root)[\"freeze\"], \"w\"" not in
      open(os.path.join(HERE, "k23_capture_freeze.py"), encoding="utf-8").read())
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
