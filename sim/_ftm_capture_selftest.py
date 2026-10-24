"""Selftest for sim/final_test/matrix/capture.py (stage M4). Offline: no simulator, no WSL, no Ollama, no API, no test item.

    python sim/_ftm_capture_selftest.py

Phase A is driven end to end with a FAKE executor (it calls the capture hook exactly as step5_live.LiveExecutor does under A11) and
FAKE frozen memories, on the real split manifest and the real dev items. Everything is written under a temporary root.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                   # noqa: E402
import capture as K                                  # noqa: E402
import step5_eval as se                              # noqa: E402
from planners.contract import MEMORY_HEADER, PlannerContract, TOOL_SPECS, prompt_sha256   # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


MAN = C.load_manifest()
ITEMS = K.load_items()
DEV = sorted(MAN["dev_pilot"])
#: A26 (K-B2b 2026-10-13): canonical dev presentations = 2 x the EFFECTIVE dev-pilot items (12 -> 10 items after the archive: 24 -> 20)
NP = 2 * len(DEV)
TEST = sorted(MAN["test"])
TYPES = ["Apple", "Fridge", "Mug", "Cabinet", "Knife"]
TOOLS = sorted(TOOL_SPECS)[:6]
QUIET = lambda *a, **k: None                          # noqa: E731


# ---------------------------------------------------------------------------------------------------- fakes
class FakeCounter:
    header = MEMORY_HEADER
    tokenizers = ["gemma", "anthropic"]

    def counts(self, text):
        return {"gemma": len(text) // 4 + 1, "anthropic": len(text) // 3 + 1}

    def __call__(self, text):
        return max(self.counts(text).values()) if text else 0

    @property
    def header_cost(self):
        return max(self.counts(self.header + "\n.").values())


class FakeWM:
    def __init__(self, eid):
        self.eid, self.events = eid, []

    def ingest(self, ev):
        self.events.append(ev)

    @property
    def types(self):
        return {e.get("otype") for e in self.events if e.get("otype")}


class FakeRetriever:
    def __init__(self, budget):
        self.budgets = {"L1": budget, "L2": budget, "L3": budget}


class FakeLtm:
    """Frozen, read-only: begin_test_epoch sets the RAM overlay (render_blocks refuses a frozen source without one)."""
    frozen = True

    def __init__(self, name, learned, budget=400, text_extra=""):
        self.name, self.learned, self.budget, self.text_extra = name, learned, budget, text_extra
        self._overlay = None
        self.count_tokens = FakeCounter()
        self.n_blocks = 0

    def begin_test_epoch(self, frame="f0", map_rev="m0"):
        self._overlay = {"epoch": 1, "frame": frame, "map_rev": map_rev}
        return self._overlay

    def begin_episode(self, eid, mode="practice", world="reset"):
        return FakeWM(eid)

    def retriever(self):
        return FakeRetriever(self.budget)

    def place_candidate_count(self, wm, channels=("places",)):
        return sum(1 for t in wm.types if t in self.learned)

    def memory_block(self, wm, level="L2", lang="en", channels=("places",), holding=None, **kw):
        self.n_blocks += 1
        recs = [{"id": "%s/pl:1" % t, "text": ("%s -> %s" if lang == "en" else "%s <- %s") % (t, self.learned[t])}
                for t in sorted(wm.types) if t in self.learned]
        text = "\n".join(r["text"] for r in recs)
        if text and self.text_extra:
            text += "\n" + self.text_extra
        return {"text": text, "records": recs, "tokens": len(text) // 4, "empty": not text, "dropped": []}


def _obj(name, typ, oid, pos, parents=(), contains=(), **kw):
    d = {"name": name, "objectType": typ, "objectId": oid, "position": dict(zip("xyz", pos)), "rotation": {"x": 0, "y": 0, "z": 0},
         "parentReceptacles": list(parents), "receptacleObjectIds": list(contains)}
    d.update(kw)
    return d


def make_state(pres, run, drift=False):
    """A real reset_gate state (so the preview verdict is the real comparison). drift: run 2 holds the cup in a different parent."""
    from final_test import reset_gate as rg
    sfx = "r%d" % run                                          # ids differ per restore (InitialRandomSpawn renames them)
    box, ctr, cup = "Box|+01.00|+00.00|+01.00" + sfx, "Counter|+02.00|+00.90|+01.00" + sfx, "Cup|+01.00|+00.50|+01.00" + sfx
    par = ctr if (drift and run == 2) else box
    objs = [_obj("Box_1", "Box", box, (1.0, 0.0, 1.0), contains=[] if par == ctr else [cup], openable=True, receptacle=True, isOpen=False),
            _obj("Counter_1", "Counter", ctr, (2.0, 0.9, 1.0), contains=[cup] if par == ctr else [], receptacle=True),
            _obj("Cup_1", "Cup", cup, (1.0, 0.5, 1.0), parents=[par], pickupable=True)]
    agent = {"position": {"x": 0.5, "y": 0.9, "z": 0.25}, "rotation": {"x": 0, "y": 90, "z": 0}, "horizon": 0, "standing": True}
    s0 = rg.capture_state(objs, agent, summary=["Cup [in] Box"], seen_types=["Cup", "Box"], labels={"goal_clauses": {"Cup in Box": False}},
                          feasibility={}, world_hash="wh-%s-%d" % (pres, run))
    return rg.capture_state(objs, agent, summary=["Cup [in] Box"], seen_types=["Cup", "Box"], labels={"goal_clauses": {"Cup in Box": False}},
                            feasibility=rg.feasibility_of(s0, [{"action": "pickup", "object": "Cup", "to": None}]),
                            world_hash="wh-%s-%d" % (pres, run))


class FakeExecutor:
    """Calls the plan hook the way LiveExecutor does with plan_after_sweep=True: post dict, then close the episode unexecuted."""
    frame, map_rev = "f0", "m0"

    def __init__(self, inapplicable=(), violation_on=None, drift=False, leak_state=False, mutate=None, wrong_order=False):
        self.plan_hook = None
        self.last = {"adapter": object(), "chooser": None, "skills": object()}
        self.inapplicable, self.violation_on, self.drift, self.leak_state = set(inapplicable), violation_on, drift, leak_state
        self.mutate, self.wrong_order = mutate, wrong_order
        self.calls = []

    def execute(self, req):
        n = len(self.calls)
        self.calls.append(req.cell)
        if self.mutate and n == self.mutate[0]:
            self.mutate[1]()
        res = se.EpisodeResult()
        it = req.item
        if it["id"] in self.inapplicable:
            res.inapplicable = "goal already true before acting"
            return res
        pres = req.cell.split("|capture|")[1].rsplit("|", 1)[0]
        run = int(req.cell.rsplit("|c", 1)[1])
        summary = ["Apple [on] CounterTop (1.0, 2.0)", "Mug [in] Cabinet (0.5, 1.5)", "item=%s" % it["id"][:12]]
        events = [{"kind": "task", "otype": "Apple"}, {"kind": "sighting", "otype": "Mug"}]
        post = {"summary": summary, "summary_sha": C.sha256_text("\n".join(summary)), "holding": None, "events": events, "wm": None,
                "episode_id": req.episode_id, "sweep_prims": 41, "seed_tracks": 2}
        out = self.plan_hook(req, it, None, req.memory, post=None if self.wrong_order else post)
        assert out["steps"] is None
        res.initial_state_hash = "wh-%s" % pres
        st = make_state(pres, run, self.drift)
        if self.leak_state:
            st["note"] = "sk-ant-api03-" + "A" * 40
        aud = {"violations": 1 if self.violation_on == n else 0, "counts": {"oracle": 0, "map-build": 0}}
        res.sr, res.goal_ok, res.steps = 0.0, False, 41.0
        res.extra = {"plan_not_executed": True, "plan_not_executed_reason": "reuse", "summary_sha": post["summary_sha"],
                     "post_sweep_state": st, "core": {"goal_ok": False, "exec_ok": False, "prims": 41, "audit": aud, "n_events": 2}}
        return res


class Env:
    """A temporary root with the real split manifest, all snapshot directories and (optionally) fs_blocks.json."""

    def __init__(self, ns="rehearsal-v1"):
        self.root = tempfile.mkdtemp(prefix="ftm-capture-")
        self.ns = ns
        d = os.path.join(self.root, "docs", "final-test")
        os.makedirs(d)
        shutil.copy(C.paths()["split"], os.path.join(d, "split_manifest.json"))
        for s in ["S*"] + K.CURVE_SOURCES:
            sd = C.snapshot_dir(s, self.root)
            os.makedirs(sd, exist_ok=True)
            with open(os.path.join(sd, "ltm.db"), "wb") as f:
                f.write(b"db-" + s.encode())
            with open(os.path.join(sd, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"state_hash": "h-" + s, "extra": {"leakage": {"ok": True, "hits": []}}}, f)
        self.write_fs()

    def write_fs(self, extra=""):
        doc = {"schema": "final-test.fs_blocks/1", "spec": C.SPEC_VERSION}
        for lv in K.LEVELS:
            text = "Attempts (oldest first): put apple in fridge | ok" + extra
            doc[lv] = {"level": lv, "text": text, "real_tokens": 30, "counts": {"gemma": 30, "anthropic": 28}, "budget": 400,
                       "sha256": C.sha256_text(text)}
        doc["sha256"] = C.self_sha(doc)
        os.makedirs(C.paths(self.root)["matrix_docs"], exist_ok=True)
        with open(C.paths(self.root)["fs_blocks"], "w", encoding="utf-8") as f:
            json.dump(doc, f)
        self.fs = doc

    def opener(self, learned_s=None, budget=400, extra=""):
        def _open(names):
            out = {}
            for s in names:
                out[s] = FakeLtm(s, learned_s or {"Apple": "Fridge", "Mug": "Cabinet"} if s == "S*" else {"Apple": "Fridge"}, budget,
                                 extra if s == "S*" else "")
            return out
        return _open

    def cap(self, executor=None, **kw):
        ex = executor or FakeExecutor()
        params = kw.pop("params", {"retrieval": {"a12": {"N": {"L1": 1, "L2": 2, "L3": 3}, "B": {"L1": 400, "L2": 400, "L3": 400},
                                                           "incapable": []}}})
        opener = kw.pop("opener", None) or self.opener()
        seeds = []

        class _Reset:
            def __init__(self, seed):
                seeds.append(seed)

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

        c = K.Capturer(namespace=self.ns, executor=ex, contract=PlannerContract(TYPES, tools=TOOLS), counter=FakeCounter(),
                       open_sources=opener, items=ITEMS, manifest=C.load_manifest(self.root), fs_doc=kw.pop("fs_doc", self.fs),
                       root=self.root, params=params, log=QUIET, reset_patch=_Reset, tol_loader=kw.pop("tol_loader", None), **kw)
        ex.plan_hook = c.hook
        c.seeds = seeds
        return c

    def rows(self):
        return C.Journal(C.journal_path(self.ns, "capture", self.root)).rows()

    def events(self):
        return C.Journal(C.journal_path(self.ns, "events", self.root)).rows()

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------- static
def t_static():
    print("== static")
    src = open(os.path.join(HERE, "final_test", "matrix", "capture.py"), encoding="utf-8").read()
    try:
        ast.parse(src, feature_version=(3, 10))
        ok = True
    except SyntaxError as e:
        ok = False
        print(e)
    check("capture.py parses as python 3.10", ok)
    check("no key printing or env key reads", "THESIS_ANTHROPIC_API_KEY" not in src and "reveal(" not in src)
    check("simulator imports are lazy (eval, ltm, step5_live inside functions)",
          not any(l.startswith(("import eval", "import ltm", "import step5_live", "import honest", "import ai2thor")) for l in src.splitlines()))
    check("no gold reader in the capture module (no gold_of / general_set)", "gold_of" not in src and "general_set" not in src)
    check("test items never named in the module", "g-l" not in src)


# ---------------------------------------------------------------------------------------------------- pure helpers
def t_pure():
    print("== pure helpers")
    fn = K.state_filename("canon:g-l1-pickup-apple:fa", 2)
    check("state file name has no ':' and keeps the run", ":" not in fn and fn.endswith(".c2.json") and fn.startswith("canon__"), fn)
    a = K.restored_state("canon", None, [{"action": "x"}])
    b = K.restored_state("reloc", 7, [{"action": "x"}])
    check("restored_state: canon / reloc:<seed> + setup sha", a.startswith("canon+") and b.startswith("reloc:7+") and a.split("+")[1] == b.split("+")[1])
    check("restored_state: a different setup is a different restore", K.restored_state("canon", None, []) != a)
    tmp = tempfile.mkdtemp()
    try:
        open(os.path.join(tmp, "ltm.db"), "wb").write(b"x")
        open(os.path.join(tmp, "manifest.json"), "w").write("{}")
        s1 = K.dir_sha(tmp)
        check("dir_sha stable", K.dir_sha(tmp) == s1)
        open(os.path.join(tmp, "ltm.db"), "wb").write(b"y")
        check("dir_sha changes with content", K.dir_sha(tmp) != s1)
        check("dir_sha refuses a missing directory", raises(lambda: K.dir_sha(os.path.join(tmp, "nope")), K.CaptureAbort))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("norm_text folds Persian letter variants and punctuation",
          K.norm_text("كتاب،  را‌بردار.") == K.norm_text("کتاب را بردار") == "کتاب را بردار")
    fb = K.forbidden_texts(MAN, ITEMS)
    t0 = ITEMS[TEST[0]]
    check("forbidden_texts covers EN and FA test instructions and ids", K.norm_text(t0["en"]) in fb and K.norm_text(t0["fa"]) in fb and ("id:" + TEST[0]) in fb)
    check("leakage: a test instruction inside a block is found", K.leakage_problems("Earlier: " + t0["en"] + " | ok", fb) == [TEST[0]] or TEST[0] in K.leakage_problems("Earlier: " + t0["en"] + " | ok", fb))
    check("leakage: a test item id inside a block is found", TEST[1] in K.leakage_problems("see " + TEST[1], fb))
    check("leakage: clean text passes", K.leakage_problems("Apple -> Fridge (seen 2x)", fb) == [] and K.leakage_problems("", fb) == [])
    check("leakage: a dev item's instruction is not a test leak", K.leakage_problems(ITEMS[DEV[0]]["en"], fb) == [])
    ct = PlannerContract(TYPES, tools=TOOLS)
    summ, hold = ["Apple [on] CounterTop"], None
    base = ct.messages("Put the apple away.", summ, hold)
    blk = {"slot": "memory", "text": "Apple -> Fridge"}
    ok = {"none": base, "S*": ct.messages("Put the apple away.", summ, hold, memory_block="Apple -> Fridge")}
    check("G-iso: none + block is clean", K.g_iso_problems(ok, {"S*": blk}) == [])
    fs = {"slot": "fewshot", "text": "Attempts..."}
    ok2 = dict(ok, FS=ct.messages("Put the apple away.", summ, hold, fewshot_block="Attempts..."))
    check("G-iso: the FS slot uses the experience header", K.g_iso_problems(ok2, {"S*": blk, "FS": fs}) == [])
    bad = dict(ok, **{"S*": ct.messages("Put the apple away.", ["Apple [on] CounterTop", "extra line"], hold, memory_block="Apple -> Fridge")})
    check("G-iso: a changed observation part is caught", len(K.g_iso_problems(bad, {"S*": blk})) == 1)
    leaky = {"none": ct.messages("Put the apple away.", summ, hold, memory_block="Apple -> Fridge"), "S*": ok["S*"]}
    check("G-iso: a none request carrying a block is caught", any("header" in p for p in K.g_iso_problems(leaky, {"S*": blk})))
    empty = {"none": base, "S*": base}
    check("G-iso: an empty block equals none", K.g_iso_problems(empty, {"S*": {"slot": "memory", "text": ""}}) == [])
    check("G-iso: an empty block that changes the request is caught",
          K.g_iso_problems({"none": base, "S*": ok["S*"]}, {"S*": {"slot": "memory", "text": ""}}) != [])
    check("sources_for canon repeat", K.sources_for("canon", True) == ["none", "S*", "FS"] + K.CURVE_SOURCES)
    check("sources_for canon non-repeat has no curve", K.sources_for("canon", False) == ["none", "S*", "FS"])
    check("sources_for reloc: none and S* only", K.sources_for("reloc", True) == ["none", "S*"])
    check("sources_for honours --sources and keeps none", K.sources_for("canon", True, only=["S*"]) == ["none", "S*"])
    check("curve list: A27 3 snapshots (order 0, K = 1, 3, 5), none of them S*", K.CURVE_SOURCES == ["o0k1", "o0k3", "o0k5"] and "o0k10" not in K.CURVE_SOURCES)
    pl = K.build_presentations(["a", "b"], ("canon", "reloc"), {"b"})
    check("build_presentations: reloc only for relocation items", [p[0] for p in pl] == ["canon:a:en", "canon:a:fa", "canon:b:en", "canon:b:fa", "reloc:b:en", "reloc:b:fa"])
    rows = [{"level": "L1", "pres": "p1", "blocks": {"S*": {"empty": False, "coverage": {"ok": True, "exempt": None}}}},
            {"level": "L3", "pres": "p2", "blocks": {"S*": {"empty": True, "coverage": {"ok": True, "exempt": "fewer than N place candidates"}}}},
            {"level": "L2", "pres": "p3", "blocks": {"S*": {"empty": False, "coverage": {"ok": False, "held": {"en": 1}, "N": 2}}}}]
    ag = K.aggregate_coverage(rows)
    check("aggregate_coverage: counts, exempt, failure list", ag["n"] == 3 and not ag["ok"] and len(ag["failures"]) == 1 and ag["by_level"]["L3"]["exempt"] == 1)
    check("aggregate_coverage: empty input is not a pass", K.aggregate_coverage([])["ok"] is False)


def t_reloc_patch():
    print("== relocation patch")
    saved = sys.modules.get("eval")
    calls = []
    fake = types.ModuleType("eval")
    fake._reset_world = lambda ag, scene: calls.append(("reset", scene))

    class Ag:
        def random_spawn(self, seed):
            calls.append(("spawn", seed))
            return types.SimpleNamespace(metadata={"lastActionSuccess": True})

    sys.modules["eval"] = fake
    try:
        orig = fake._reset_world
        with K.relocated_reset(11):
            fake._reset_world(Ag(), "FloorPlan1")
        check("relocated_reset: reset, THEN the seeded spawn", calls == [("reset", "FloorPlan1"), ("spawn", 11)], calls)
        check("relocated_reset restores the module attribute", fake._reset_world is orig)

        class Bad:
            def random_spawn(self, seed):
                return types.SimpleNamespace(metadata={"lastActionSuccess": False, "errorMessage": "nope"})
        with K.relocated_reset(3):
            check("relocated_reset: a failed spawn aborts", raises(lambda: fake._reset_world(Bad(), "F"), K.CaptureAbort))
    finally:
        if saved is not None:
            sys.modules["eval"] = saved
        else:
            sys.modules.pop("eval", None)


# ---------------------------------------------------------------------------------------------------- guard
def t_guard():
    print("== test-item guard")
    env = Env("rehearsal-v1")
    try:
        for ns in ("rehearsal-v1", "practice-v1"):
            e2 = Env(ns)
            try:
                c = e2.cap()
                c.manifest = dict(c.manifest)
                c.manifest[C.NAMESPACES[ns]["items"]] = [TEST[0]]
                check("%s: a test item is refused by the capture run" % ns, raises(c.run, C.TestItemRefused))
                check("%s: nothing was written when refused" % ns, e2.rows() == [] and e2.events() == [])
            finally:
                e2.close()
        et = Env("test-v1")
        try:
            c = et.cap()
            check("test-v1 without --allow-test is refused", raises(c.run, C.TestItemRefused))
            c2 = et.cap(allow_test=True)
            check("test-v1 with --allow-test but no valid freeze.json / families file is refused", raises(c2.run, C.TestItemRefused))
            check("test-v1: refused before anything was written", et.rows() == [] and et.events() == [])
        finally:
            et.close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r1 = raises(lambda: K.main(["--namespace", "test-v1", "--items", "test", "--dry-run"]), C.TestItemRefused)
        check("CLI: --items test --dry-run without the guard conditions is refused", r1)
        check("CLI: --items dev with the test namespace is refused", raises(lambda: K.main(["--namespace", "test-v1", "--items", "dev", "--dry-run"]), SystemExit))
        check("CLI: --runs 1,2 is refused in test-v1", raises(lambda: K.main(["--namespace", "test-v1", "--items", "test", "--runs", "1,2", "--dry-run"]), SystemExit))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = K.main(["--namespace", "rehearsal-v1", "--items", "dev", "--dry-run", "--runs", "1,2", "--worlds", "canon,reloc", "--reloc-all"])
        d = json.loads(out.getvalue())
        check(f"CLI dry-run (dev): {NP} presentations x 2 worlds x 2 runs = {4 * NP} cells, guard ok",
              d["presentations"] == 2 * NP and d["cells"] == 4 * NP and d["guard"]["n_test"] == 0, d)
        check("CLI dry-run lists missing snapshots (none exist yet) and exits 2", rc == 2 or d["missing_snapshots"] == [], (rc, d["missing_snapshots"][:2]))
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- end to end
def t_e2e():
    print("== end to end (fakes)")
    env = Env()
    try:
        c = env.cap(worlds=("canon", "reloc"), runs=(1, 2), reloc_seed=777, reloc_all=True)
        rc = c.run()
        rows = env.rows()
        evs = env.events()
        check("full dev capture: exit 0 and phase_complete", rc == 0 and any(e["event"] == "phase_complete" for e in evs), (rc, [e["event"] for e in evs]))
        check(f"{4 * NP} records ({NP} canon + {NP} reloc presentations x runs 1, 2)", len(rows) == 4 * NP and len({r["key"] for r in rows}) == 4 * NP, len(rows))
        pc = [e for e in evs if e["event"] == "phase_complete"][0]
        check("phase_complete carries n and the journal sha", pc["n"] == 4 * NP and pc["journal_sha"] == C.sha256_file(C.journal_path(env.ns, "capture", env.root)))
        r0 = [r for r in rows if r["world"] == "canon" and r["capture_run"] == 1][0]
        check("record schema: envelope + required fields", r0["schema"] == "ftm.capture/1" and all(k in r0 for k in C.SCHEMAS["capture"]["req"]))
        check("no gold / outcome key anywhere in any record", all(C.forbidden_keys(r, C.GOLD_KEYS | C.OUTCOME_KEYS) == [] for r in rows))
        check("instruction is the verbatim item text", r0["instruction"] == ITEMS[r0["item"]][r0["lang"]])
        check("summary + sha recorded; holding recorded", r0["summary"] and r0["summary_sha"] == C.sha256_text("\n".join(r0["summary"])) and r0["holding"] is None)
        check("state_ref is under states/ and the file exists", r0["state_ref"].startswith("states/") and ":" not in r0["state_ref"]
              and os.path.exists(os.path.join(C.ns_docs(env.ns, env.root), r0["state_ref"])))
        st = json.load(open(os.path.join(C.ns_docs(env.ns, env.root), r0["state_ref"]), encoding="utf-8"))
        check("state file holds the evaluator labels (scoring side), the capture record does not",
              "labels" in st["state"] and "labels" not in json.dumps(r0) and "goal_clauses" not in json.dumps(r0))
        check("world hash recorded as a diagnostic", r0["world_hash"].startswith("wh-"))
        check("audit: honesty, G-iso, masking, leakage, order", r0["audit"]["honesty"] == {"violations": 0, "oracle": 0, "map_build": 0}
              and r0["audit"]["g_iso"]["ok"] and r0["audit"]["masking"]["ok"] and r0["audit"]["plan_order"] == "after_sweep")
        # sources
        rep = [r for r in rows if r["world"] == "canon" and r["item"] in C.repeat_items(env.ns, MAN)]
        nonrep = [r for r in rows if r["world"] == "canon" and r["item"] not in C.repeat_items(env.ns, MAN)]
        check("canon non-repeat presentations: sources none, S*, FS", all(sorted(r["requests"]) == ["FS", "S*", "none"] for r in nonrep) and nonrep)
        check("REPEAT presentations (if any dev item is one) also carry the curve sources",
              all(set(K.CURVE_SOURCES) <= set(r["requests"]) for r in rep))
        rl = [r for r in rows if r["world"] == "reloc"]
        check("relocated presentations: none and S* only", rl and all(sorted(r["requests"]) == ["S*", "none"] for r in rl))
        check("relocated records name the seed and the restore", all(r["reloc_seed"] == 777 and r["restored_state"].startswith("reloc:777+") for r in rl)
              and all(r["restored_state"].startswith("canon+") for r in rows if r["world"] == "canon"))
        check("the relocation context was entered for reloc cells only, with the seed", len(c.seeds) == 2 * NP and set(c.seeds) == {777}, (len(c.seeds), set(c.seeds)))
        # request shas re-render
        ct = PlannerContract(TYPES, tools=TOOLS)
        r = r0
        none_sha = prompt_sha256(ct.messages(r["instruction"], r["summary"], r["holding"]))
        check("none request sha equals an independent re-render", r["requests"]["none"] == none_sha)
        sb = r["blocks"]["S*"]
        s_sha = prompt_sha256(ct.messages(r["instruction"], r["summary"], r["holding"], memory_block=sb["text"] or None))
        check("S* request sha equals an independent re-render from the recorded block", r["requests"]["S*"] == s_sha)
        fb = r["blocks"]["FS"]
        f_sha = prompt_sha256(ct.messages(r["instruction"], r["summary"], r["holding"], fewshot_block=fb["text"]))
        check("FS request sha equals an independent re-render (fewshot slot)", r["requests"]["FS"] == f_sha and fb["slot"] == "fewshot")
        check("S* and FS requests differ from none; none differs by block only", r["requests"]["S*"] != r["requests"]["none"] != r["requests"]["FS"])
        check("block entries carry text, ids, real_tokens, budget, empty, coverage",
              all(k in sb for k in ("text", "ids", "real_tokens", "budget", "empty", "coverage")) and sb["ids"] == ["Apple/pl:1", "Mug/pl:1"] and sb["budget"] == 400)
        check("real tokens counted on header + block", sb["real_tokens"] == max(FakeCounter().counts(MEMORY_HEADER + "\n" + sb["text"]).values()))
        check("same evidence: run 1 and run 2 give the same request shas", all(
            a["requests"] == b["requests"] for a in rows for b in rows if a["pres"] == b["pres"] and a["capture_run"] == 1 and b["capture_run"] == 2))
        check("EN and FA of one item render the same record ids",
              all(a["blocks"]["S*"]["ids"] == b["blocks"]["S*"]["ids"] for a in rows for b in rows
                  if a["item"] == b["item"] and a["world"] == b["world"] and a["capture_run"] == b["capture_run"] and a["lang"] == "en" and b["lang"] == "fa"))
        cov_levels = {r["level"]: r["blocks"]["S*"]["coverage"] for r in rows if r["capture_run"] == 1 and r["world"] == "canon"}
        check("A12 coverage recorded per presentation; L3 exempt (2 < N=3), L1 ok", cov_levels.get("L1", {"ok": True})["ok"]
              and (cov_levels.get("L3") is None or cov_levels["L3"]["exempt"]))
        gates = [e for e in evs if e["event"] == "capture_gates"][0]["gates"]
        check("gates event: snapshot hash, coverage, masking, G-iso, leakage, key leak, reset pairs all pass",
              all(v["pass"] for v in gates.values()) and {"snapshot_hash", "a12_coverage", "masking", "g_iso", "leakage", "key_leak", "reset_pairs"} <= set(gates), gates)
        check(f"reset_pairs.json: {2 * NP} pairs ({NP} canon + {NP} reloc)", gates["reset_pairs"]["n_pairs"] == 2 * NP and
              len(json.load(open(os.path.join(C.ns_docs(env.ns, env.root), "reset_pairs.json"), encoding="utf-8"))) == 2 * NP)
        check("reset gate preview is recorded (skipped while the shipped tolerances are uncalibrated, a verdict once W2 calibrated them)", bool(gates["reset_pairs"]["preview"]))
        check("hook was never given a memory; executor chooser stayed None", c.ex.last["chooser"] is None)
        check("source LTMs rendered but never executed: block counts > 0", all(getattr(l, "n_blocks", 1) > 0 for l in c.ltms.values()))
    finally:
        env.close()


def t_resume():
    print("== resume after an interrupt")
    env = Env()
    try:
        c = env.cap(worlds=("canon",), runs=(1,), interrupt=10)
        rc = c.run()
        check("interrupted at 10 new cells: exit 5, 10 records, no phase_complete", rc == 5 and len(env.rows()) == 10
              and not any(e["event"] == "phase_complete" for e in env.events()))
        c2 = env.cap(worlds=("canon",), runs=(1,))
        rc2 = c2.run()
        keys = [r["key"] for r in env.rows()]
        check(f"resume completes the rest: exit 0, {NP} distinct records, no duplicates", rc2 == 0 and len(keys) == NP == len(set(keys)), (rc2, len(keys)))
        check(f"resume executed only the remaining {NP - 10} cells", len(c2.ex.calls) == NP - 10, len(c2.ex.calls))
        check("phase_complete after resume", sum(e["event"] == "phase_complete" for e in env.events()) == 1)
        # a torn tail is repaired, not trusted
        p = C.journal_path(env.ns, "capture", env.root)
        with open(p, "ab") as f:
            f.write(b'{"key": "torn')
        c3 = env.cap(worlds=("canon",), runs=(1,))
        check("a torn last line is ignored; the run is still complete", c3.run() == 0 and len(env.rows()) == NP)
        before = sum(e["event"] == "phase_complete" for e in env.events())
        c4 = env.cap(worlds=("canon",), runs=(1,), limit=3)
        check("--limit: a partial smoke run never writes phase_complete", c4.run() == 0 and sum(e["event"] == "phase_complete" for e in env.events()) == before)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- failures
def run_fail(label, want_rc, want_in_reason=None, expect_rows=None, expect_complete=False, **kw):
    env = Env()
    try:
        ex = kw.pop("executor", None)
        envkw = kw.pop("env_setup", None)
        if envkw:
            envkw(env)
        cap_kw = {k: v for k, v in kw.items() if k not in ("opener",)}
        opener = kw.get("opener")
        c = env.cap(executor=ex, worlds=cap_kw.pop("worlds", ("canon",)), runs=cap_kw.pop("runs", (1,)), opener=opener(env) if opener else None, **cap_kw)
        rc = c.run()
        evs = env.events()
        done = any(e["event"] == "phase_complete" for e in evs)
        reason = " ".join(str(e.get("reason", "")) + json.dumps(e.get("failed", "")) for e in evs)
        ok = rc == want_rc and done == expect_complete and (want_in_reason is None or want_in_reason in reason)
        if expect_rows is not None:
            ok = ok and len(env.rows()) == expect_rows
        check(label, ok, (rc, done, reason[:200], len(env.rows())))
        return env, c
    finally:
        env.close()


def t_failures():
    print("== fail-fast paths")
    # masking: an FS block that leaks the scene name aborts and writes nothing for the presentation
    run_fail("masking: a FloorPlan name in a block aborts (exit 4), nothing written", 4, "masking",
             env_setup=lambda e: e.write_fs(extra=" FloorPlan1"), expect_rows=0)
    run_fail("masking: an objectId-like token with a signed coordinate aborts", 4, "masking",
             opener=lambda e: e.opener(extra="Apple|+01.20|+00.90|-01.10"), expect_rows=0)
    # leakage: a block holding a test instruction aborts
    t0 = ITEMS[TEST[0]]["en"]
    run_fail("leakage: a test instruction inside the S* block aborts", 4, "leakage", opener=lambda e: e.opener(extra="Earlier: " + t0 + " ok"), expect_rows=0)
    run_fail("leakage: a test item id inside the FS block aborts", 4, "leakage", env_setup=lambda e: e.write_fs(extra=" " + TEST[3]), expect_rows=0)
    # honesty violation in the sweep
    run_fail("honesty: an audit violation during the sweep aborts", 4, "honesty", executor=FakeExecutor(violation_on=2), expect_rows=2)
    # wrong order
    run_fail("A11: a hook call without the post-sweep dict aborts", 4, "A11", executor=FakeExecutor(wrong_order=True), expect_rows=0)
    # key-like text in a state file
    run_fail("key leak: key-like text in a state file aborts before it is written", 4, "key-like", executor=FakeExecutor(leak_state=True), expect_rows=0)
    # snapshot changes while the phase runs
    def mut(env_holder):
        def _m():
            p = os.path.join(C.snapshot_dir("S*", env_holder[0].root), "ltm.db")
            open(p, "ab").write(b"tamper")
        return _m
    holder = []
    env = Env()
    holder.append(env)
    try:
        c = env.cap(executor=FakeExecutor(mutate=(3, mut(holder))), worlds=("canon",), runs=(1,))
        rc = c.run()
        evs = env.events()
        fl = [e for e in evs if e["event"] == "capture_gates"]
        check("snapshot hash: a snapshot changed during the run fails the gate (exit 2, no phase_complete)",
              rc == 2 and fl and "snapshot_hash" in fl[0]["failed"] and not any(e["event"] == "phase_complete" for e in evs), (rc, fl and fl[0]["failed"]))
    finally:
        env.close()
    # a snapshot whose manifest records a failed leakage gate is refused up front
    env = Env()
    try:
        p = os.path.join(C.snapshot_dir("S*", env.root), "manifest.json")
        json.dump({"extra": {"leakage": {"ok": False, "hits": ["x"]}}}, open(p, "w"))
        rc = env.cap(worlds=("canon",), runs=(1,)).run()
        check("snapshot leakage verdict failed in its manifest: exit 4, nothing captured", rc == 4 and env.rows() == [])
    finally:
        env.close()
    env = Env()
    try:
        shutil.rmtree(C.snapshot_dir("S*", env.root))
        c = env.cap(worlds=("canon",), runs=(1,))
        rc = c.run()
        check("a missing snapshot: exit 4 before any simulator work", rc == 4 and env.rows() == [] and c.ex.calls == [])
    finally:
        env.close()
    # A12 coverage failing is a gate failure (no phase_complete), a record still exists
    run_fail("A12 coverage: a block that does not fit B_L fails the gate (exit 2)", 2, "a12_coverage",
             params={"retrieval": {"a12": {"N": {"L1": 1, "L2": 2, "L3": 3}, "B": {"L1": 8, "L2": 8, "L3": 8}, "incapable": []}}})
    # reset-gate pairs: a symbolic drift between run 1 and run 2 fails the preview
    tol = lambda: {"tau_pos_m": 0.05, "tau_rot_deg": 5.0}         # noqa: E731
    run_fail("reset pairs: a changed parent between run 1 and run 2 fails the gate (exit 2)", 2, "reset_pairs",
             executor=FakeExecutor(drift=True), runs=(1, 2), tol_loader=tol)
    run_fail("reset pairs: identical states pass the preview", 0, None, runs=(1, 2), tol_loader=tol, expect_complete=True)


def t_inapplicable():
    print("== inapplicable items")
    env = Env()
    try:
        bad = DEV[0]
        c = env.cap(executor=FakeExecutor(inapplicable={bad}), worlds=("canon",), runs=(1,))
        rc = c.run()
        rows = env.rows()
        ina = [r for r in rows if r["item"] == bad]
        check("inapplicable presentation: terminal record with status, reason, no state, no requests, 3 tries (M-fix2)",
              len(ina) == 2 and all(r["status"] == "inapplicable" and r["state_ref"] is None and r["requests"] == {} and r["inapplicable"]
                                    and r["tries"] == 3 for r in ina))
        check("M-fix2: a canonical inapplicable presentation fails the `applicable` gate (exit 2) and never completes the phase",
              rc == 2 and not any(e["event"] == "phase_complete" for e in env.events())
              and any(e["event"] == "capture_gates" and "applicable" in e["failed"] for e in env.events())
              and sum(1 for e in env.events() if e["event"] == "inapplicable_rerun") == 2 * 3)
        check("inapplicable records still pass the firewall", all(C.forbidden_keys(r, C.GOLD_KEYS | C.OUTCOME_KEYS) == [] for r in rows))
    finally:
        env.close()


def t_hook():
    print("== hook contract")
    env = Env()
    try:
        c = env.cap()
        h = c.hook
        check("hook refuses a memory handed to the executor", raises(lambda: h(None, ITEMS[DEV[0]], None, object(), post={}), K.CaptureAbort))
        check("hook refuses the pre-A11 order", raises(lambda: h(None, ITEMS[DEV[0]], None, None, post=None), K.CaptureAbort))
        check("no snapshot is opened for a run limited to none", env.cap(only_sources=["none"], worlds=("canon",)).needed_sources() == [])
    finally:
        env.close()


def t_a31():
    """A31 (DRAFT, approved by Arash 2026-10-13): whole ids, whole delimiter-defined segments, exempt spans as barriers, fail closed."""
    import practice_k as PK
    import treatment as T
    print("== A31 leakage scan")
    res = PK.reserved_acceptance(None, ITEMS)
    fb = K.forbidden_texts(MAN, ITEMS, res)
    n_ids = len(set(MAN["test"]) | set(MAN.get("test_uncapped_not_scored") or []) | set(res["ids"]))
    check("A31.1 the forbidden set = test + test-uncapped + the 29 reserved acceptance ids (whole ids) and their EN/FA segment keys",
          len([k for k in fb if isinstance(k, str) and k.startswith("id:")]) == n_ids and len(res["ids"]) == 29
          and all(("id:" + i) in fb for i in res["ids"]) and any(isinstance(k, tuple) for k in fb))
    sample = "A b, c.\r\nD; e: f | g \"h\" «i» j ؛ k ؟ l -> m “n” o، p"
    check("A31.2 segments: CRLF / . ; : | quotes guillemets Arabic ; ? and '->' split; Latin and Arabic commas do not",
          K.segments(sample) == ["a b c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o p"], K.segments(sample))
    rows = [json.loads(l) for l in open(os.path.join(C.ROOT, "docs", "final-test", "matrix", "practice-v2", "practice.jsonl"), encoding="utf-8")]
    a8 = [r for r in rows if r["attempt"] == 8][0]
    line8 = "8. en | %s | pickup ButterKnife; slice Bread; slice Tomato | ok ok ok | success" % a8["instruction"]
    eid8 = "example/practice-kitchen/practice-v2:o0:a8"
    refused = {v for k, v in fb.items() if isinstance(k, str) and k.startswith("id:")}
    pools = {"practice-v2": set(MAN["practice_pool"]) - set(res["ids"])}
    check("A31.3 the a8 practice line holds test sentences as FRAGMENTS (the pre-A31 scan flags it; ft-v1 keeps that scan)",
          "g-l1-slice-bread" in K.leakage_problems_legacy(line8, fb))
    sp8, au8 = K.exempt_spans(line8, g1=None, examples=[eid8], rows=rows, items=ITEMS, refused=refused, pools=pools, form="fs")
    check("A31.4 the a8 FS line is ALLOWED (whole-segment rule alone and with its validated example span); the audit names a8 and the span",
          K.leakage_problems(line8, fb) == [] and K.leakage_problems(line8, fb, sp8) == []
          and au8[0]["kind"] == "example" and au8[0]["item"] == "g-l2-slice-bread-tomato"
          and line8[au8[0]["start"]:au8[0]["end"]] == a8["instruction"], au8)
    pk8 = 'Example from my practice (succeeded): "%s" -> pickup ButterKnife; slice Bread; slice Tomato' % a8["instruction"]
    sp8p, _a = K.exempt_spans(pk8, g1=None, examples=[eid8], rows=rows, items=ITEMS, refused=refused, pools=pools, form="package")
    check("A31.4b the a8 package example line is ALLOWED (its quoted instruction field is the validated span)",
          K.leakage_problems(pk8, fb, sp8p) == [] and pk8[sp8p[0][0]:sp8p[0][1]] == a8["instruction"])
    egg = ITEMS["g-l2-microwave-run-egg"]
    g1 = "The request in English: put Egg to the microwave; close the microwave; switch on the microwave."
    blk = "Kitchen: a Microwave.\n" + g1 + " This request names: Egg."
    sp5 = K.exempt_spans(blk, g1=g1, form="package")[0]
    check("A31.5 the egg-FA G1 line: caught WITHOUT the exemption ('close the microwave' is a whole segment), ALLOWED with it",
          K.leakage_problems(blk, fb) == ["g-l1-close-microwave"] and K.leakage_problems(blk, fb, sp5) == [])
    import ltm_package as LP
    r = LP.render_request(LP._parse_request(LP.make_query_profile("pkg-1.r6b", egg["fa"], "fa", None, [])))
    check("A31.6 the G1 recompute from the presentation's own pinned FA text is exactly that line", (K.G1_PREFIX + str(r) + ".") == g1, r)
    planted = [("A31.7a a test sentence outside the exemption", blk + "\nTip: Close the microwave.", "g-l1-close-microwave"),
               ("A31.7b a whole test id, even next to an exempt span", blk + " (g-l1-close-microwave)", "g-l1-close-microwave"),
               ("A31.7c a test id in a row key", blk + " practice-v2|practice|o0|3|g-l2-slice-bread|en", "g-l2-slice-bread"),
               ("A31.7d a Persian test sentence", blk + "\n" + ITEMS["g-l1-close-microwave"]["fa"], "g-l1-close-microwave"),
               ("A31.7e a reserved acceptance id", blk + " " + res["ids"][0], res["ids"][0])]
    for name, txt, want in planted:
        check(name + " is CAUGHT (planted real leak)", want in K.leakage_problems(txt, fb, sp5), K.leakage_problems(txt, fb, sp5))
    check("A31.8 the 4 slice collisions are not id hits; their whole short ids are",
          K.leakage_problems("g-l2-slice-apple-potato g-l2-slice-bread-tomato g-l2-slice-potato-tomato g-l2-slice-tomato-lettuce", fb) == []
          and K.leakage_problems("g-l2-slice-bread", fb) == ["g-l2-slice-bread"])
    multi = {("seg", ("open the fridge", "take the apple")): "X"}
    t9 = "Open the fridge. ZZZ. Take the apple."
    check("A31.9 a multi-segment text must match whole CONSECUTIVE segments; an exempt span is a barrier that never joins neighbours",
          K.leakage_problems("Hint: Open the fridge. Take the apple.", multi) == ["X"]
          and K.leakage_problems("Open the fridge, take the apple.", multi) == []
          and K.leakage_problems(t9, multi, [(t9.index("ZZZ"), t9.index("ZZZ") + 3)]) == []
          and K.leakage_problems("Close the QQ microwave.", fb, [(10, 12)]) == [])
    bad = [("A31.10a two G1 lines", blk + "\n" + g1, dict(g1=g1)),
           ("A31.10b a G1 line that differs from the recompute", blk.replace("switch on", "turn on"), dict(g1=g1)),
           ("A31.10c a G1 line where the recompute gives none", blk, dict(g1=None)),
           ("A31.10d an example whose row instruction differs from the pinned text", line8.replace("Take", "take"),
            dict(examples=[eid8], rows=[dict(a8, instruction=a8["instruction"].replace("Take", "take"))], form="fs")),
           ("A31.10e an example of a refused (test) item", line8, dict(examples=[eid8], rows=[dict(a8, item="g-l2-slice-bread")], form="fs")),
           ("A31.10f an example id that is not a row of the source", line8, dict(examples=[eid8], rows=[], form="fs")),
           ("A31.10g the rendered field occurs twice", line8 + "\n" + line8, dict(examples=[eid8], rows=rows, form="fs")),
           ("A31.10h the render log lists fewer examples than the block shows", line8 + "\n" + line8.replace("8. en", "7. en"),
            dict(examples=[eid8], rows=rows, form="fs")),
           ("A31.10i an example outside its namespace's pinned pool", line8, dict(examples=[eid8], rows=rows, form="fs", pools={}))]
    for name, txt, kw in bad:
        kw2 = dict(dict(g1=None, examples=(), rows=(), items=ITEMS, refused=refused, pools=pools, form="package"), **kw)
        check(name + ": integrity abort (fail closed)", raises(lambda: K.exempt_spans(txt, **kw2), K.CaptureAbort))
    check("A31.11 ids are matched with treatment.id_in (one shared matcher)", K.T.id_in is T.id_in)


def main() -> int:
    t_static()
    t_pure()
    t_reloc_patch()
    t_guard()
    t_e2e()
    t_resume()
    t_failures()
    t_inapplicable()
    t_hook()
    t_a31()
    print("\n%d passed, %d failed" % (N["pass"], N["fail"]))
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
