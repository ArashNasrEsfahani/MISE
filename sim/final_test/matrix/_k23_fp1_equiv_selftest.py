"""Offline gate for the K23 FP1 equivalence harness (k23_fp1_equivalence.py, protocol v2 after the Astra review of 2026-10-18).

stage / collect: a run is staged once, only at the expected commit and from committed code; the decisive runs carry exactly the frozen B0
capture + plans (S1 / S2: only the top-level treatment_id rebound, after an identity + render proof, and nothing is touched when the proof
fails); collect restores the tree byte for byte and is never repeated. compare-live is FAIL-CLOSED: missing evidence, a missing run, a
missing offline verdict, a missing cell, an episode_wall row, a snapshot change, a behaviour difference stable within the baseline, or a
diagnostic rendering / plan difference -> NOT_EQUIVALENT; a difference that also appears between B1 and B2 -> UNRESOLVED (never
automatically equivalent); measurement-only differences -> reported, still EQUIVALENT. Hermetic: throw-away git repositories in a temp dir
hold a copy of the frozen freeze-gates-ft2 journals (read-only source: this tree's docs); no simulator, no worktree is touched.

    python sim/final_test/matrix/_k23_fp1_equiv_selftest.py
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k23_fp1_equivalence as E                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, *E.DOCS_REL)
ok = fail = 0


def check(n, c, d=None):
    global ok, fail
    if c:
        ok += 1
    else:
        fail += 1
        print("FAIL", n, "" if d is None else f"-- {str(d)[:300]}")


def refused(f):
    try:
        f()
    except SystemExit:
        return True
    return False


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout.strip()


tmp = tempfile.mkdtemp(prefix="k23eqst_")
trees, heads = {}, {}
for lab in ("base", "branch"):
    t = os.path.join(tmp, lab)
    shutil.copytree(SRC, os.path.join(t, *E.DOCS_REL))
    w = os.path.join(t, *E.WORK_REL)
    os.makedirs(os.path.join(w, "cache", "ab"))
    open(os.path.join(w, "cache", "ab", "abc.json"), "w").write('{"key": "abc", "response": {"x": 1}}')
    open(os.path.join(w, "tokcache.json"), "w").write("{}")
    for s, d in E._snap_dirs(t).items():                    # stand-in frozen snapshot dirs (their shas are bound at stage / collect)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "manifest.json"), "w").write(json.dumps({"s": s}))
    os.makedirs(os.path.join(t, "sim"), exist_ok=True)
    open(os.path.join(t, "sim", "code.py"), "w").write("x = 1\n")
    git(t, "init", "-q"); git(t, "config", "user.email", "t@t"); git(t, "config", "user.name", "t"); git(t, "config", "core.autocrlf", "false")
    open(os.path.join(t, ".gitignore"), "w").write("sim/final_test/matrix/work/\n")
    git(t, "add", "-A"); git(t, "commit", "-qm", "fixture")
    trees[lab], heads[lab] = t, git(t, "rev-parse", "HEAD")
ev = os.path.join(tmp, "evidence")
B, S = trees["base"], trees["branch"]
NEW = "f" * 64
FULL = {"presentations": 20, "requests": 84, "requests_expected": 84, "blocks": 64, "blocks_expected": 64, "match_capture": 84,
        "differs_from_capture": 0, "miss": 0, "errors": 0}


def proof_ok(tree, ev_, run, summary=None, head=None):
    os.makedirs(os.path.join(ev_, run), exist_ok=True)
    pf = os.path.join(ev_, run, "render_proof.json")
    json.dump({"head": head or git(tree, "rev-parse", "HEAD"), "inputs": {E.NS: {"capture_sha": E._sha_file(os.path.join(SRC, "capture.jsonl"))}},
               "render_summary": {E.NS: summary or FULL}}, open(pf, "w"))
    return {"file": f"{run}/render_proof.json", "sha256": E._sha_file(pf), "summary": FULL}
ident_ok = lambda tree: NEW                                                                                 # noqa: E731
docs_b, work_b = E._tree_paths(B)
sha0 = E._dir_sha(docs_b), E._dir_sha(work_b)
b0_cap = E._rows(os.path.join(SRC, "capture.jsonl"))
PRES = sorted({r["pres"] for r in b0_cap})
CELLS = [f"{E.NS}|plan|{a}|{p}|r0" for a in E.EXEC_ARMS for p in PRES]


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def fake_exec(tree, run, tweak=None):
    """What an execute + score job leaves behind: one exec / outcome / score row per expected cell (synthetic, deterministic)."""
    d, w = E._tree_paths(tree)
    ex, oc, sc = [], [], []
    for c in CELLS:
        u = hashlib.sha256(c.encode()).hexdigest()[:32]
        e = {"unit": u, "run": 0, "key": f"x|{u}", "status": "done", "attempt": 0, "error": None, "chooser": "prior", "exec_policy": None,
             "briefing_log": [], "snapshot_sha_before": None, "snapshot_sha_after": None, "prims": 30, "ts": run,
             "audit": {"violations": 0, "invalid": False, "errors": 0, "counts": {"oracle": 0}},
             "state_check": {"ok": True, "breaches": [], "registered_ok": True, "override": False, "max_dpos": 1e-05, "max_drot": 0.01}}
        o = {"unit": u, "run": 0, "goal_ok": True, "exec_ok": True, "sr": 1.0, "prims": 30, "walking_m": 3.0, "ts": run,
             "steps": [{"i": 0, "action": "search_for", "object": "Pan", "ok": True, "cause": None, "under": None, "retries": 0, "prims": 19},
                       {"i": 1, "action": "pickup", "object": "Pan", "ok": True, "cause": None, "under": None, "retries": 0, "prims": 3}]}
        s = {"cell": c, "run": 0, "arm": c.split("|")[2], "unit": u, "sr": 1.0, "goal_ok": True, "exec_ok": True, "scored": True, "ts": run}
        if tweak:
            tweak(c, e, o, s)
        ex.append(e); oc.append(o); sc.append(s)
    write_rows(os.path.join(d, "exec.jsonl"), ex); write_rows(os.path.join(d, "outcome.sealed.jsonl"), oc); write_rows(os.path.join(d, "score.jsonl"), sc)


def fake_diag(tree, run):
    d, _w = E._tree_paths(tree)
    shutil.copy2(os.path.join(SRC, "capture.jsonl"), os.path.join(d, "capture.jsonl"))
    pl = E._rows(os.path.join(SRC, "plan.jsonl"))
    write_rows(os.path.join(d, "plan.jsonl"), [dict(r, cached_calls=r.get("calls")) for r in pl] if run == "SC" else pl)


# ---------------------------------------------------------------------------------------------------- stage / collect
check("stage refused at a commit other than the expected one", refused(lambda: E.stage(B, "B1", ev, "0" * 40)) and not os.path.isdir(os.path.join(ev, "B1", "docs")))
open(os.path.join(B, "sim", "code.py"), "w").write("x = 2\n")
check("stage refused with uncommitted code under sim/", refused(lambda: E.stage(B, "B1", ev, heads["base"])))
git(B, "checkout", "--", "sim/code.py")
r = E.stage(B, "B1", ev, heads["base"])
check("stage B1 archives the tree's namespace (docs + work) and B0, and binds the head and snapshot shas",
      os.path.isdir(os.path.join(ev, "orig_base", "docs")) and os.path.isdir(os.path.join(ev, "B0", "docs")) and r["head"] == heads["base"]
      and set(r["snapshots_before"]) == set(E.FROZEN_SOURCES))
check("B1 carries exactly the frozen B0 capture + plans (byte-identical, identity unchanged) and only capture / plan events",
      E._sha_file(os.path.join(docs_b, "capture.jsonl")) == E._sha_file(os.path.join(SRC, "capture.jsonl"))
      and E._sha_file(os.path.join(docs_b, "plan.jsonl")) == E._sha_file(os.path.join(SRC, "plan.jsonl"))
      and all(e.get("phase") in ("capture", "plan") for e in E._rows(os.path.join(docs_b, "events.jsonl"))))
check("a run is staged once (a second stage is refused)", refused(lambda: E.stage(B, "B1", ev, heads["base"])))
fake_exec(B, "B1")
c = E.collect(B, "B1", ev)
check("collect B1 restores the tree byte for byte (docs, work, git clean) with the head unchanged",
      all(c["restored"].values()) and c["head_unchanged"] and (E._dir_sha(docs_b), E._dir_sha(work_b)) == sha0, c)
check("a run is collected once", refused(lambda: E.collect(B, "B1", ev)))
docs_s, work_s = E._tree_paths(S)
s_before = E._dir_sha(docs_s)
check("S1 without --expect-commit is refused", refused(lambda: E.stage(S, "S1", ev)))
check("S1 refused when the successor cannot compute its identity (no re-pin); the tree is untouched",
      refused(lambda: E.stage(S, "S1", ev, heads["branch"], identity_fn=lambda t: (_ for _ in ()).throw(SystemExit("no re-pin")), proof_fn=proof_ok))
      and E._dir_sha(docs_s) == s_before and not os.path.exists(os.path.join(ev, "S1", "staged.json")))
check("S1 refused when the render proof fails; the tree is untouched",
      refused(lambda: E.stage(S, "S1", ev, heads["branch"], identity_fn=ident_ok, proof_fn=lambda *a: (_ for _ in ()).throw(SystemExit("x"))))
      and E._dir_sha(docs_s) == s_before)
check("proof_complete: an errored render, a missing presentation or a mismatch is not complete",
      E.proof_complete(FULL) and not E.proof_complete(dict(FULL, errors=1)) and not E.proof_complete(dict(FULL, presentations=19))
      and not E.proof_complete(dict(FULL, requests=83, requests_expected=83)) and not E.proof_complete(dict(FULL, differs_from_capture=1)))
r = E.stage(S, "S1", ev, heads["branch"], identity_fn=ident_ok, proof_fn=proof_ok)
cap_s = E._rows(os.path.join(docs_s, "capture.jsonl"))
check("S1: the ONE rewrite is the top-level treatment_id of the copied capture / plan rows (everything else identical), recorded",
      all(x["treatment_id"] == NEW for x in cap_s) and [dict(x, treatment_id=None) for x in cap_s] == [dict(x, treatment_id=None) for x in b0_cap]
      and r["identity_old"] == E.B0_IDENTITY and r["identity_new"] == NEW and r["journals"]["capture.jsonl"]["rows"] == len(b0_cap))
fake_exec(S, "S1"); E.collect(S, "S1", ev)
for run, t in (("B2", B), ("S2", S)):
    E.stage(t, run, ev, heads[RUN := ("base" if run == "B2" else "branch")], identity_fn=ident_ok, proof_fn=proof_ok)
    fake_exec(t, run); E.collect(t, run, ev)
check("SC before BC is refused", refused(lambda: E.stage(S, "SC", ev, heads["branch"])))
E.stage(B, "BC", ev, heads["base"]); fake_diag(B, "BC"); E.collect(B, "BC", ev)
E.stage(S, "SC", ev, heads["branch"]); fake_diag(S, "SC"); E.collect(S, "SC", ev)
def summ(p, q, b):
    return {"presentations": p, "requests": q, "requests_expected": q, "blocks": b, "blocks_expected": b, "match_capture": q,
            "differs_from_capture": 0, "miss": 0, "errors": 0}


ns_ok = {ns: summ(*tot) for ns, tot in E.OFFLINE_TOTALS.items()}
bc_in = {"capture_sha": E._sha_file(os.path.join(ev, "BC", "docs", "capture.jsonl")), "tokcache_sha": E._sha_file(os.path.join(ev, "BC", "work", "tokcache.json"))}


def verdict(path, live=False, base_head=None, bc=None, summary=None, artifact=True):
    rs = dict(ns_ok, **({E.NS + "@live": summ(20, 84, 64)} if live else {}))
    rs = summary or rs
    ins = {}
    for t in ("base", "branch"):
        e = {"head": (base_head or heads["base"]) if t == "base" else heads["branch"], "inputs": ({E.NS + "@live": bc or bc_in} if live else {})}
        if artifact:                                       # the offline output the verdict compared: named, hashed, content = the verdict's
            af = path.replace(".json", f".{t}.offline.json")
            json.dump({"head": e["head"], "inputs": e["inputs"], "render_summary": rs}, open(af, "w"))
            e.update(file=af, sha256=E._sha_file(af))
        ins[t] = e
    json.dump({"identical": True, "not_renderable": {"base": 0, "branch": 0}, "errors_in": [], "render_summary": {"base": rs, "branch": rs},
               "inputs": ins}, open(path, "w"))
    return path


vf = verdict(os.path.join(tmp, "verdict_offline.json"))
vl = verdict(os.path.join(tmp, "verdict_offline_BC.json"), live=True)
VS = (vf, vl)

# ---------------------------------------------------------------------------------------------------- compare-live (fail closed)
v = E.compare_live(ev, VS)
check("complete identical evidence -> EQUIVALENT, every expected cell compared", v["verdict"] == "EQUIVALENT" and v["comparable_cells"] == len(CELLS) == 80,
      (v["verdict"], v["blocking"][:2], v.get("comparable_cells")))
check("no offline verdict -> NOT_EQUIVALENT", E.compare_live(ev, ())["verdict"] == "NOT_EQUIVALENT")
check("only one offline verdict, or the same file twice -> NOT_EQUIVALENT",
      E.compare_live(ev, (vf,))["verdict"] == E.compare_live(ev, (vf, vf))["verdict"] == "NOT_EQUIVALENT")
bad_live = json.load(open(vl)); bad_live["render_summary"]["branch"][E.NS] = dict(FULL, errors=1, requests=83)
vl2 = os.path.join(tmp, "verdict_bad_live.json"); json.dump(bad_live, open(vl2, "w"))
check("a live verdict whose B0 render is incomplete / errored in one tree -> NOT_EQUIVALENT (identical but incomplete never passes)",
      E.compare_live(ev, (vf, vl2))["verdict"] == "NOT_EQUIVALENT")
check("empty evidence dir -> NOT_EQUIVALENT", E.compare_live(os.path.join(tmp, "nothing"), (vf,))["verdict"] == "NOT_EQUIVALENT")


def edit_run(run, fn):
    d = os.path.join(ev, run, "docs")
    rows = {k: E._rows(os.path.join(d, f"{k}.jsonl")) for k in ("exec", "outcome.sealed", "score")}
    fn(rows)
    for k, rr in rows.items():
        write_rows(os.path.join(d, f"{k}.jsonl"), rr)


snap = {r: {k: E._rows(os.path.join(ev, r, "docs", f"{k}.jsonl")) for k in ("exec", "outcome.sealed", "score")} for r in E.RUNS}


def restore():
    for r in E.RUNS:
        for k, rr in snap[r].items():
            write_rows(os.path.join(ev, r, "docs", f"{k}.jsonl"), rr)


edit_run("S2", lambda R: R.__setitem__("score", R["score"][1:]))
check("an expected cell missing in one run -> NOT_EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
restore()
edit_run("S1", lambda R: R["outcome.sealed"][0]["steps"][1].__setitem__("under", "episode_wall"))
v = E.compare_live(ev, VS)
check("an episode_wall row -> NOT_EQUIVALENT", v["verdict"] == "NOT_EQUIVALENT" and any("episode_wall" in b["what"] for b in v["blocking"]))
restore()
flip = lambda R: R["outcome.sealed"][3]["steps"][1].__setitem__("ok", False)          # noqa: E731
edit_run("S1", flip)
v = E.compare_live(ev, VS)
check("a step difference only within the successor (S1 != S2, B1 == B2) -> NOT_EQUIVALENT, element-wise field named",
      v["verdict"] == "NOT_EQUIVALENT" and any("outcome.steps[1].ok" in (b.get("fields") or []) for b in v["blocking"]), v["blocking"][:2])
edit_run("S2", flip)
v = E.compare_live(ev, VS)
check("the same difference in S1 and S2, stable within the baseline -> NOT_EQUIVALENT", v["verdict"] == "NOT_EQUIVALENT")
edit_run("B2", flip)
v = E.compare_live(ev, VS)
check("... also differing between B1 and B2 -> UNRESOLVED (investigate; never automatically EQUIVALENT)",
      v["verdict"] == "UNRESOLVED" and v["unresolved"] and not v["blocking"], (v["verdict"], v["blocking"][:2]))
restore()
edit_run("S1", lambda R: R["exec"][5].__setitem__("prims", 31))
v = E.compare_live(ev, VS)
check("a measurement-only difference is reported (spread) and does not decide", v["verdict"] == "EQUIVALENT"
      and v["measure_spread"].get("exec.prims", {}).get("B1~S1") == 1, v["measure_spread"])
restore()
cj = os.path.join(ev, "S2", "collected.json")
cd = json.load(open(cj)); cd["snapshots_after"]["S*"] = "changed"; json.dump(cd, open(cj, "w"))
check("a frozen snapshot changed during a run -> NOT_EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
cd["snapshots_after"]["S*"] = json.load(open(os.path.join(ev, "S2", "staged.json")))["snapshots_before"]["S*"]; json.dump(cd, open(cj, "w"))
cp = os.path.join(ev, "SC", "docs", "capture.jsonl")
cap = E._rows(cp); k = next(x for x in cap[0]["blocks"] if x != "none"); cap[0]["blocks"][k]["sha256"] = "0" * 64; write_rows(cp, cap)
v = E.compare_live(ev, VS)
check("diagnostic capture: equal observation but a different rendered block -> NOT_EQUIVALENT",
      any(b["what"].startswith("diagnostic capture: equal observation") for b in v["blocking"]))
shutil.copy2(os.path.join(SRC, "capture.jsonl"), cp)
pp = os.path.join(ev, "SC", "docs", "plan.jsonl")
pl = E._rows(pp); pl[0]["cached_calls"] = 0; pl[0]["calls"] = 1; write_rows(pp, pl)
check("diagnostic plan: a planner call for a request BC made -> NOT_EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
pl[0]["cached_calls"] = pl[0]["calls"]; write_rows(pp, pl)
check("evidence back to complete and identical -> EQUIVALENT again", E.compare_live(ev, VS)["verdict"] == "EQUIVALENT")
# Astra FP1-EQUIV-r3: completeness of identical offline evidence, binding to the commits and the collected BC evidence, proof re-read
inc = dict(ns_ok); inc["rehearsal-v3"] = summ(55, 196, 140)
x = E.compare_offline({"renders": {"k": {"text_sha": "a"}}, "render_summary": inc, "replay": {}, "snapshots": {}},
                      {"renders": {"k": {"text_sha": "a"}}, "render_summary": inc, "replay": {}, "snapshots": {}})
check("compare_offline: identical but incomplete renders (a capture below its fixed totals) are NOT identity", not x["identical"] and x["incomplete"])
check("a part A verdict below the fixed totals -> NOT_EQUIVALENT",
      E.compare_live(ev, (verdict(os.path.join(tmp, "v_inc.json"), summary=inc), vl))["verdict"] == "NOT_EQUIVALENT")
check("an offline verdict made on another commit -> NOT_EQUIVALENT",
      E.compare_live(ev, (verdict(os.path.join(tmp, "v_old.json"), base_head="1" * 40), vl))["verdict"] == "NOT_EQUIVALENT")
check("a live verdict that rendered another BC capture -> NOT_EQUIVALENT",
      E.compare_live(ev, (vf, verdict(os.path.join(tmp, "v_bc.json"), live=True, bc=dict(bc_in, capture_sha="0" * 64))))["verdict"] == "NOT_EQUIVALENT")
st1 = os.path.join(ev, "S1", "staged.json"); keep_st = open(st1, "rb").read(); keep_pf = open(os.path.join(ev, "S1", "render_proof.json"), "rb").read()
d = json.load(open(st1)); pr = proof_ok(S, ev, "S1", summary=dict(FULL, errors=1)); d["render_proof"] = pr; json.dump(d, open(st1, "w"))
check("a successor proof whose FILE summary is incomplete (the staged copy says complete) -> NOT_EQUIVALENT (summary re-read from the file)",
      E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
d = json.load(open(st1)); pr = proof_ok(S, ev, "S1", head="2" * 40); d["render_proof"] = pr; json.dump(d, open(st1, "w"))
check("a stale successor proof (made on another commit) -> NOT_EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
open(st1, "wb").write(keep_st); open(os.path.join(ev, "S1", "render_proof.json"), "wb").write(keep_pf)
check("restored again -> EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "EQUIVALENT")
# Astra re-pin review: the verdict's artifact must be named + hashed, present, unchanged, and its content must match the verdict
check("a verdict that does not name / hash its artifact -> NOT_EQUIVALENT",
      E.compare_live(ev, (verdict(os.path.join(tmp, "v_noart.json"), artifact=False), vl))["verdict"] == "NOT_EQUIVALENT")
va = verdict(os.path.join(tmp, "v_miss.json")); os.remove(va.replace(".json", ".base.offline.json"))
check("a missing artifact -> NOT_EQUIVALENT", E.compare_live(ev, (va, vl))["verdict"] == "NOT_EQUIVALENT")
vs_ = verdict(os.path.join(tmp, "v_sub.json")); vdoc = json.load(open(vs_))
af = vdoc["inputs"]["branch"]["file"]; json.dump({"head": heads["branch"], "inputs": {}, "render_summary": inc}, open(af, "w"))
check("a substituted artifact (changed after the verdict) -> NOT_EQUIVALENT", E.compare_live(ev, (vs_, vl))["verdict"] == "NOT_EQUIVALENT")
vdoc["inputs"]["branch"]["sha256"] = E._sha_file(af); json.dump(vdoc, open(vs_, "w"))
v = E.compare_live(ev, (vs_, vl))
check("a substituted artifact whose sha the verdict records but whose content differs from the verdict -> NOT_EQUIVALENT",
      v["verdict"] == "NOT_EQUIVALENT" and any("substituted artifact" in b["what"] for b in v["blocking"]))
check("the positive fixture carries the artifact fields -> EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "EQUIVALENT"
      and all(json.load(open(p))["inputs"][t].get("file") and json.load(open(p))["inputs"][t].get("sha256") for p in VS for t in ("base", "branch")))
s1c = os.path.join(ev, "S1", "docs", "capture.jsonl"); keep = open(s1c, "rb").read()
cap = E._rows(s1c); cap[0]["holding"] = "Tampered"; write_rows(s1c, cap)
v = E.compare_live(ev, VS)
check("a rebound capture that differs in more than the identity (and no longer matches its recorded sha) -> NOT_EQUIVALENT",
      v["verdict"] == "NOT_EQUIVALENT" and any("more than the top-level identity" in b["what"] for b in v["blocking"])
      and any("provenance" in b["what"] for b in v["blocking"]))
open(s1c, "wb").write(keep)
pf = os.path.join(ev, "S2", "render_proof.json"); keep = open(pf, "rb").read(); open(pf, "a").write(" ")
check("an altered render proof file -> NOT_EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
open(pf, "wb").write(keep)
bcp = os.path.join(ev, "BC", "docs", "capture.jsonl"); keep = open(bcp, "rb").read()
write_rows(bcp, E._rows(bcp)[1:])
check("an incomplete diagnostic capture (19 presentations) -> NOT_EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "NOT_EQUIVALENT")
open(bcp, "wb").write(keep)
check("restored -> EQUIVALENT", E.compare_live(ev, VS)["verdict"] == "EQUIVALENT")
shutil.rmtree(tmp, ignore_errors=True)
print(f"{ok}/{ok + fail}")
sys.exit(1 if fail else 0)
