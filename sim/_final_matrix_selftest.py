"""Integration gate for the final-test matrix (stage M-integrate).  Offline: no simulator, no WSL, no Ollama, no API, no test item.

    python sim/_final_matrix_selftest.py            # every _ftm_* selftest (in parallel) + the end-to-end below
    python sim/_final_matrix_selftest.py --e2e      # only the end-to-end
    python sim/_final_matrix_selftest.py --modules  # only the per-module selftests

The end-to-end drives the REAL modules against each other on synthetic fakes, in a temporary root, on a synthetic namespace that
maps onto the 12 dev items (the REPEAT set and the relocation set are taken from dev items, so every block of the design exists):

    capture (fake executor + fake frozen memories)  ->  plan (fake provider transports, real cache / ledger / runner)
    ->  execute (fake executor, real unit building / state check / journals)  ->  score (real gold of the dev items)
    ->  monitor  ->  analyze

Two full runs are compared: one uninterrupted, one interrupted and resumed in every phase that can be interrupted. The journals must
agree (up to timestamps). The CLI (`matrix_run.py`) is exercised for its preflight and dispatch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)[:600]), flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# =============================================================================================================== module gates
MODULE_GATES = ["_ftm_common_selftest.py", "_ftm_fs_block_selftest.py", "_ftm_freeze_selftest.py", "_ftm_practice_k_selftest.py",
                "_ftm_capture_selftest.py", "_ftm_plan_selftest.py", "_ftm_execute_selftest.py", "_ftm_score_selftest.py",
                "_ftm_monitor_selftest.py", "_ftm_analyze_selftest.py", "_ftm_callplan_selftest.py",
                "_ftm_a36_selftest.py"]   # A36 draft r2: excluded_reset + common-subset analysis
OTHER_GATES = ["_planners_selftest.py", "_step4_rules_selftest.py", "_step4_gemma_selftest.py"]


def _run_gate(name):
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, name)], capture_output=True, text=True, cwd=os.path.dirname(HERE))
    out = (p.stdout or "") + (p.stderr or "")
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
    return name, p.returncode, tail[0][:120], round(time.time() - t0, 1)


def t_modules(include_other=True):
    print("== per-module selftests (parallel)")
    names = [n for n in MODULE_GATES + (OTHER_GATES if include_other else []) if os.path.exists(os.path.join(HERE, n))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(_run_gate, names))
    for name, rc, tail, secs in res:
        check(f"{name} exits 0 ({tail}) [{secs}s]", rc == 0, tail)
    check("every module gate exists", all(os.path.exists(os.path.join(HERE, n)) for n in MODULE_GATES),
          [n for n in MODULE_GATES if not os.path.exists(os.path.join(HERE, n))])


# =============================================================================================================== e2e world
def e2e():
    import common as C
    import capture as K
    import plan as PL
    import execute as X
    import score as SC
    import monitor as MO
    import analyze as AN
    import matrix_run as MR
    import honest_skills as hs
    import step5_eval as se
    from final_test import reset_gate as rg
    from planners import PlannerContract, make_planner
    from planners.contract import MEMORY_HEADER
    from planners.keys import Secret

    NS = "matrix-e2e"
    C.NAMESPACES[NS] = {"items": "dev_pilot", "ceilings": {"core": 50.0, "total": 100.0}, "archived_arms": C.ARCHIVED_ARM_NAMES}   # A23: explicit opt-in keeps the archived S / H paths exercised
    FAKEKEY = "sk-ant-FAKEKEY0123456789abcdefSELFTEST"
    MAN0 = C.load_manifest()
    DEV = sorted(MAN0["dev_pilot"])
    REPEAT = DEV[:2]
    MAN = copy.deepcopy(MAN0)
    MAN["repeat_subset"] = list(REPEAT)                 # synthetic: dev items stand in for the REPEAT set
    MAN["relocation_probe"] = list(DEV[:3])
    ITEMS = K.load_items()
    TYPES = ["Apple", "Bread", "Fridge", "Knife", "Mug", "Plate", "Sink"]
    contract = PlannerContract(TYPES, tools=hs.TOOLS)
    QUIET = lambda *a, **k: None                          # noqa: E731

    # ---------------------------------------------------------------------------------------------- fakes: capture side
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
        frozen = True

        def __init__(self, name, learned, budget=400):
            self.name, self.learned, self.budget = name, learned, budget
            self.count_tokens = FakeCounter()
            self._overlay = None

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
            recs = [{"id": "%s/pl:1" % t, "text": "%s -> %s" % (t, self.learned[t])} for t in sorted(wm.types) if t in self.learned]
            text = "\n".join(r["text"] for r in recs)
            return {"text": text, "records": recs, "tokens": len(text) // 4, "empty": not text, "dropped": []}

    def _obj(name, typ, oid, pos, parents=(), contains=(), **kw):
        d = {"name": name, "objectType": typ, "objectId": oid, "position": dict(zip("xyz", pos)), "rotation": {"x": 0, "y": 0, "z": 0},
             "parentReceptacles": list(parents), "receptacleObjectIds": list(contains)}
        d.update(kw)
        return d

    def make_state(pres, run):
        sfx = "r%d" % run                                          # ids differ per restore (InitialRandomSpawn renames them)
        box, ctr, cup = "Box|+01.00|+00.00|+01.00" + sfx, "Counter|+02.00|+00.90|+01.00" + sfx, "Cup|+01.00|+00.50|+01.00" + sfx
        objs = [_obj("Box_1", "Box", box, (1.0, 0.0, 1.0), contains=[cup], openable=True, receptacle=True, isOpen=False),
                _obj("Counter_1", "Counter", ctr, (2.0, 0.9, 1.0), receptacle=True),
                _obj("Cup_1", "Cup", cup, (1.0, 0.5, 1.0), parents=[box], pickupable=True)]
        agent = {"position": {"x": 0.5, "y": 0.9, "z": 0.25}, "rotation": {"x": 0, "y": 90, "z": 0}, "horizon": 0, "standing": True}
        s0 = rg.capture_state(objs, agent, summary=["Cup [in] Box"], seen_types=["Cup", "Box"], labels={"goal_clauses": {"Cup in Box": False}},
                              feasibility={}, world_hash="wh-%s-%d" % (pres, run))
        return rg.capture_state(objs, agent, summary=["Cup [in] Box"], seen_types=["Cup", "Box"], labels={"goal_clauses": {"Cup in Box": False}},
                                feasibility=rg.feasibility_of(s0, [{"action": "pickup", "object": "Cup", "to": None}]),
                                world_hash="wh-%s-%d" % (pres, run))

    def summary_of(item_id):
        return ["Apple [on] CounterTop (1.0, 2.0)", "Mug [in] Cabinet (0.5, 1.5)", "item=%s" % item_id[:12]]

    class CapExecutor:
        frame, map_rev = "f0", "m0"

        def __init__(self, inapp=()):
            self.plan_hook = None
            self.last = {"adapter": object(), "chooser": None, "skills": object()}
            self.calls = []
            self.world = "canon"
            self.inapp = set(inapp)                    # presentations whose setup "fails at restore" (M-fix2 tests)

        def execute(self, req):
            self.calls.append(req.cell)
            res = se.EpisodeResult()
            it = req.item
            pres = req.cell.split("|capture|")[1].rsplit("|", 1)[0]
            run = int(req.cell.rsplit("|c", 1)[1])
            if pres in self.inapp:
                res.inapplicable = "setup failed at restore (test)"
                res.initial_state_hash = "wh-inapp"
                return res
            summary = summary_of(it["id"])
            events = [{"kind": "task", "otype": "Apple"}, {"kind": "sighting", "otype": "Mug"}]
            post = {"summary": summary, "summary_sha": C.sha256_text("\n".join(summary)), "holding": None, "events": events, "wm": None,
                    "episode_id": req.episode_id, "sweep_prims": 41, "seed_tracks": 2}
            out = self.plan_hook(req, it, None, req.memory, post=post)
            assert out["steps"] is None
            res.initial_state_hash = "wh-%s" % pres
            st = make_state(pres, run)
            res.sr, res.goal_ok, res.steps = 0.0, False, 41.0
            res.extra = {"plan_not_executed": True, "plan_not_executed_reason": "reuse", "summary_sha": post["summary_sha"],
                         "post_sweep_state": st, "core": {"goal_ok": False, "exec_ok": False, "prims": 41,
                                                           "audit": {"violations": 0, "counts": {"oracle": 0, "map-build": 0}}, "n_events": 2}}
            return res

    class _Reset:
        seeds = []

        def __init__(self, seed):
            _Reset.seeds.append(seed)

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

    # ---------------------------------------------------------------------------------------------- the tree
    def build_root():
        root = tempfile.mkdtemp(prefix="ftm-e2e-")
        d = os.path.join(root, "docs", "final-test")
        os.makedirs(d)
        shutil.copy(C.paths()["split"], d)
        shutil.copy(C.paths()["families"], d)
        os.makedirs(os.path.join(root, "sim", "dataset"))
        shutil.copy(C.paths()["dataset"], os.path.join(root, "sim", "dataset", "general.json"))
        os.makedirs(C.paths(root)["matrix_docs"], exist_ok=True)
        for s in ["S*"] + K.CURVE_SOURCES:
            sd = C.snapshot_dir(s, root)
            os.makedirs(sd, exist_ok=True)
            with open(os.path.join(sd, "ltm.db"), "wb") as f:
                f.write(b"db-" + s.encode())
            with open(os.path.join(sd, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"state_hash": "h-" + s, "extra": {"leakage": {"ok": True, "hits": []}}}, f)
        doc = {"schema": "final-test.fs_blocks/1", "spec": C.SPEC_VERSION}
        for lv in K.LEVELS:
            text = "Attempts (oldest first): put apple in fridge | ok"
            doc[lv] = {"level": lv, "text": text, "real_tokens": 30, "counts": {"gemma": 30, "anthropic": 28}, "budget": 400,
                       "sha256": C.sha256_text(text)}
        doc["sha256"] = C.self_sha(doc)
        with open(C.paths(root)["fs_blocks"], "w", encoding="utf-8") as f:
            json.dump(doc, f)
        json.dump(TYPES, open(os.path.join(root, "types.json"), "w"))
        return root, doc

    def open_sources(names):
        return {s: FakeLtm(s, {"Apple": "Fridge", "Mug": "Cabinet"} if s == "S*" else {"Apple": "Fridge"}) for s in names}

    def capture(root, fs_doc, interrupt=0, executor=None):
        ex = executor or CapExecutor()
        params = {"retrieval": {"a12": {"N": {"L1": 1, "L2": 2, "L3": 3}, "B": {"L1": 400, "L2": 400, "L3": 400}, "incapable": []}}}
        c = K.Capturer(namespace=NS, executor=ex, contract=contract, counter=FakeCounter(), open_sources=open_sources, items=ITEMS,
                       manifest=MAN, fs_doc=fs_doc, root=root, params=params, log=QUIET, worlds=("canon", "reloc"), runs=(1,),
                       reloc_seed=777, reloc_all=True, reset_patch=_Reset, interrupt=interrupt)
        ex.plan_hook = c.hook
        return c, c.run()

    # ---------------------------------------------------------------------------------------------- fakes: plan side
    class Crash(BaseException):
        pass

    class LiveInvariant(RuntimeError):                  # execute.is_invariant matches on the class NAME
        pass

    class Fake:
        """Transports for the real providers. The reply depends on the instruction text, so different items give different plans."""
        def __init__(self, garble_gemma_memory=False):
            self.calls = []
            self.garble = garble_gemma_memory          # M-fix2: Gemma answers garbage whenever the prompt carries a memory block (G+M rejected)

        @staticmethod
        def reply(text):
            k = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 3
            plan = [[{"action": "open", "object": "Fridge", "to": None, "arg": None}],
                    [{"action": "open", "object": "Fridge", "to": None, "arg": None},
                     {"action": "close", "object": "Fridge", "to": None, "arg": None}],
                    [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]][k]
            return json.dumps({"say": "ok", "plan": plan})

        def ollama(self, url, headers, body, timeout):
            self.calls.append("gemma")
            user = body["messages"][-1]["content"]
            if self.garble and any(MEMORY_HEADER in m["content"] for m in body["messages"]):      # the repair turn too
                return 200, {}, {"message": {"content": "this is not a plan"}, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 5}
            return 200, {}, {"message": {"content": self.reply(user.split("Command:")[-1])}, "done_reason": "stop",
                             "prompt_eval_count": 900, "eval_count": 30}

        def anth(self, url, headers, body, timeout):
            assert headers["x-api-key"] == FAKEKEY
            self.calls.append(body["model"])
            user = body["messages"][0]["content"]
            return 200, {}, {"content": [{"type": "text", "text": self.reply(user.split("Command:")[-1] + body["model"])}],
                             "usage": {"input_tokens": 1200, "output_tokens": 40}, "stop_reason": "end_turn", "model": body["model"],
                             "id": "msg_%d" % len(self.calls)}

        def codex(self, cmd, prompt, cwd, env, timeout):
            if "--version" in cmd:
                return types.SimpleNamespace(stdout="codex-cli 9.9.9\n", stderr="", returncode=0)
            self.calls.append("codex")
            ev = [{"type": "item.completed", "item": {"type": "agent_message", "text": self.reply(prompt.split("Command:")[-1])}},
                  {"type": "turn.completed", "usage": {"input_tokens": 14000, "output_tokens": 60}}]
            model = cmd[cmd.index("-m") + 1]
            sess = ('INFO codex_exec: Codex initialized with event: SessionConfiguredEvent { model: "%s", model_provider_id: "openai", approval_policy: Never, '
                    'permission_profile: Managed { file_system: Restricted { entries: [FileSystemSandboxEntry { path: Special { value: Root }, access: Read, '
                    'missing_path_behavior: None }], glob_scan_max_depth: None }, network: Restricted }, active_permission_profile: None, '
                    'reasoning_effort: Some(High), network_proxy: None }\n' % model)
            return types.SimpleNamespace(stdout="\n".join(json.dumps(e) for e in ev), stderr=sess, returncode=0)

    class FakeCtl:
        def ensure(self):
            pass

        def close(self):
            pass

    def factory_for(fake):
        def factory(spec, **kw):
            kw.setdefault("sleep", lambda s: None)
            if spec == "gemma":
                return make_planner("gemma", transport=fake.ollama, **kw)
            if spec.startswith("codex:"):
                return make_planner(spec, runner=fake.codex, which=lambda x: "codex", **kw)
            return make_planner(spec, transport=fake.anth, env={"THESIS_ANTHROPIC_API_KEY": FAKEKEY}, **kw)
        return factory

    def plan(root, interrupt=None, passes=None, fake=None, **kw):
        fake = fake or Fake()
        p = PL.Plan(NS, passes=passes, root=root, provider_factory=factory_for(fake), ollama_ctl=FakeCtl(), gpu_sample=False,
                    key_getter=lambda: Secret(FAKEKEY), out=QUIET, manifest=MAN, types_path=os.path.join(root, "types.json"),
                    interrupt_after=interrupt, **kw)
        return p, p.run(), fake

    # ---------------------------------------------------------------------------------------------- fakes: execute side
    class ExecEx:
        """Behaves like LiveExecutor(plan_after_sweep=True, state_capture=True, state_check=...): sweep -> state check -> hook -> steps."""
        supports_reloc = True
        frame, map_rev = "f0", "m0"

        def __init__(self, log, boom=()):
            self.state_check = self.plan_hook = None
            self.log = log
            self.last = None
            self.closed = 0
            self.boom = set(boom)                       # presentations that break an honesty invariant (M-fix2 tests)

        def close(self):
            self.closed += 1

        def execute(self, req):
            it = req.item
            world = "reloc" if _Reset.active else "canon"
            pres = C.pres_id(it["id"], req.lang, world)
            summary = summary_of(it["id"])
            post = {"summary": summary, "summary_sha": C.sha256_text("\n".join(summary)), "holding": None, "events": [], "wm": None,
                    "episode_id": req.episode_id, "sweep_prims": 5, "seed_tracks": 1}
            if pres in self.boom:
                raise LiveInvariant("honesty invariant broken (test)")
            state = make_state(pres, 2)                               # run 2: ids renamed against Phase A's run 1
            chk = self.state_check(state, post)
            self.last = {"adapter": types.SimpleNamespace(stream=[{"kind": "sighting", "t": 1}]), "chooser": None, "skills": None}
            self.log.append((pres, req.memory is not None, _Reset.active))
            if not chk.get("ok"):
                return types.SimpleNamespace(inapplicable=None, sr=None, goal_ok=None, steps=5.0, initial_state_hash="wh", found=None, retries=None,
                                             walking_m=None, opens=None, stale_visits=None, gate_pass=None, applied=None,
                                             extra={"blocked_state": chk["breaches"], "plan_not_executed": True,
                                                    "core": {"goal_ok": False, "exec_ok": False, "prims": 5, "audit": {"violations": 0, "counts": {}},
                                                             "n_events": 1}})
            ph = self.plan_hook(req, it, None, req.memory, post=post)
            steps = ph["steps"]
            rows = [{"i": i, "action": s["action"], "object": s.get("object"), "to": s.get("to"), "ok": True, "cause": None, "retries": 0, "prims": 3}
                    for i, s in enumerate(steps)]
            h = int(C.sha256_text(C.canon([[s["action"], s.get("object"), s.get("to")] for s in steps]) + pres)[:4], 16)
            goal = bool(steps) and h % 4 != 0                          # deterministic pseudo-outcome
            prims = 5 + 3 * len(steps)
            return types.SimpleNamespace(inapplicable=None, sr=1.0 if goal else 0.0, goal_ok=goal, steps=float(prims), initial_state_hash="wh",
                                         found=None, retries=0, walking_m=1.0, opens=0, stale_visits=0, gate_pass=None, applied=None,
                                         extra={"core": {"goal_ok": goal, "exec_ok": True, "prims": prims,
                                                         "audit": {"violations": 0, "counts": {"oracle": 0, "map-build": 0}}, "n_events": 1},
                                                "steps": rows, "summary_sha": post["summary_sha"], "sweep_prims": 5})

    class FakeMem:
        def __init__(self, d):
            self.d = d

        def state_hash(self):
            return "H" + C.sha256_text(open(os.path.join(self.d, "ltm.db"), "rb").read().decode("latin1"))[:8]

        def close(self):
            pass

    class _ResetExec(_Reset):
        active = False

        def __enter__(self):
            _Reset.active = True
            return self

        def __exit__(self, *e):
            _Reset.active = False
            return False

    _Reset.active = False

    def execute(root, log, interrupt=None, boom=()):
        ph = X.ExecutePhase(NS, root=root, executor_factory=lambda phase: ExecEx(log, boom),
                            memory_opener=lambda phase, src, tag, ex: FakeMem(C.snapshot_dir(src, root)),
                            tolerances={"tau_pos_m": 0.05, "tau_rot_deg": 5.0}, worlds=("canon", "reloc"), reloc_seed=777,
                            repeat_override=list(REPEAT), out=QUIET, reset_patch=_ResetExec)
        info = ph.run(stop_after=interrupt)
        return ph, info

    def rows(root, kind):
        return C.Journal(C.journal_path(NS, kind, root)).rows()

    def events(root, name=None):
        return [e for e in rows(root, "events") if name is None or e["event"] == name]

    def norm(rs, drop=("ts", "wall_s", "session", "latency", "attempts", "cached_calls", "world_hash_match", "pid")):
        out = []
        for r in rs:
            out.append({k: v for k, v in r.items() if k not in drop})
        return sorted(out, key=lambda r: json.dumps(r, sort_keys=True, default=str))

    # ============================================================================================ run A: uninterrupted
    print("== end to end: uninterrupted run")
    rootA, fs_doc = build_root()
    t0 = time.time()
    try:
        c, rc = capture(rootA, fs_doc)
        cap_rows = rows(rootA, "capture")
        # A26 (K-B2b): 2 x 2 x the EFFECTIVE dev items (48 with the original 12; 40 after the archive)
        check(f"capture: exit 0, phase_complete, {4 * len(DEV)} canon+reloc presentations", rc == 0 and events(rootA, "phase_complete") and len(cap_rows) == 4 * len(DEV), (rc, len(cap_rows)))
        pres_rep = [r for r in cap_rows if r["world"] == "canon" and r["item"] in REPEAT]
        check("capture: the REPEAT presentations carry the curve sources (A27: o0k1, o0k3, o0k5), the others do not",
              pres_rep and all(set(K.CURVE_SOURCES) <= set(r["requests"]) for r in pres_rep)
              and all(not (set(K.CURVE_SOURCES) & set(r["requests"])) for r in cap_rows if r["world"] == "canon" and r["item"] not in REPEAT))
        check("capture: relocation presentations exist and carry the seed", any(r["world"] == "reloc" and r["reloc_seed"] == 777 for r in cap_rows))
        check("capture -> monitor: a mid-pipeline tick is clean (no outcome file exists yet)",
              MO.tick_once(NS, root=rootA, write=False, items_mode="dev")["exit_code"] in (0, 1))

        import freeze as FZ
        brs = FZ.block_rows(cap_rows, json.load(open(C.paths(rootA)["fs_blocks"], encoding="utf-8")))
        sblocks = [r for r in brs if r["source"] == "S*" and r["kind"] == "memory"]
        check("capture -> freeze: block_rows flattens the capture rows (S*, curve, FS) the token gate counts", sblocks and any(r["kind"] == "fs" for r in brs), len(brs))
        bad_cov = [r["id"] for r in sblocks if not (isinstance(r["coverage"], dict) and r["coverage"].get("ok") is True)]
        check("capture -> freeze: every S* block carries coverage.ok = True (what the A12 gate in freeze.py demands)", not bad_cov, bad_cov[:3])
        tg = FZ.token_gate(brs, FakeCounter(), {"L1": 400, "L2": 400, "L3": 400})
        check("capture -> freeze: the real-token gate runs on the captured blocks (EN and FA record ids identical)", tg["ok"] and tg["n_id_pairs"] > 0, tg["reasons"][:3])

        p, prc, fake = plan(rootA)
        pl = rows(rootA, "plan")
        check("plan: exit 0 (every pass: gemma core fs curve reloc haiku codex)", prc == 0, prc)
        check("plan: phase_complete for plan, capture request shas verified (no abort)", any(e.get("phase") == "plan" for e in events(rootA, "phase_complete")))
        check("plan: every record is ok (no rejected / infra on the fake provider)", {r["status"] for r in pl} == {"ok"}, {r["status"] for r in pl})
        check("plan: cells of every block exist", {r["block"] for r in pl} >= {"main", "repeat", "curve", "reloc", "codex"}, {r["block"] for r in pl})
        check("plan: the paid ledger spent > 0 and stayed within the namespace ceilings", 0 < sum(r["cost_usd"] for r in pl) < 100)

        log = []
        xph, info = execute(rootA, log)
        ex_rows = rows(rootA, "exec")
        check("execute: complete, no unexecuted-by-error", info["complete"] and all(r["status"] in ("done", "not_executed") for r in ex_rows),
              ({r["status"] for r in ex_rows}, [r.get("error") for r in ex_rows if r["status"] == "crash"][:2]))
        check("execute: exec records list their plan cells (score's mapping)", all(r.get("cells") for r in ex_rows if r["status"] == "done"),
              [r["key"] for r in ex_rows if not r.get("cells")][:2])
        check("execute: the relocated world was entered through the reset patch", any(w for (_, _, w) in log) and any(not w for (_, _, w) in log))
        check("execute: the exec journal holds no outcome field and no step list (integrity only)",
              all(C.forbidden_keys(r, C.OUTCOME_KEYS | C.GOLD_KEYS) == [] and "steps" not in r for r in ex_rows))
        oc = rows(rootA, "outcome")
        check("execute: outcome steps carry holding_before when the sweep stream has it (key present or None)",
              oc and all(all("holding_before" in s for s in o["steps"]) for o in oc))

        # monitor between execute and score
        m = MO.tick_once(NS, root=rootA, write=True, items_mode="dev")
        check("monitor after execute: no breach", m["exit_code"] in (0, 1) and not m["breaches"], m["breaches"])

        sr = SC.run_score(NS, root=rootA, items="dev", log=QUIET)
        sc_rows = rows(rootA, "score")
        check("score: scored, seal written and complete", sr["status"] == "scored" and json.load(open(os.path.join(C.ns_docs(NS, rootA), "score.seal.json")))["complete"], sr)
        check("score: rows exist for every arm of the matrix", {r["arm"] for r in sc_rows} >= {"G0", "G+M", "S0", "S+M", "O0", "O+M", "G+FS", "O+FS", "P0", "P+M"},
              sorted({r["arm"] for r in sc_rows}))
        check("score: no accepted plan lost its outcome (missing_outcome / missing_exec = 0)", not [r for r in sc_rows if r.get("failure_class") in ("missing_outcome", "missing_exec")],
              [r.get("failure_class") for r in sc_rows if r.get("failure_class") not in (None, "ok")][:3])
        check("score: A27 REPEAT runs 0-1 are kept for the REPEAT items (2 executions per distinct core unit, was 3)",
              {r["run"] for r in sc_rows if C.parse_pres(r["pres"])["item"] in REPEAT and r["arm"] in ("G0", "O0")} == {0, 1})
        _diag = json.load(open(os.path.join(C.ns_docs(NS, rootA), SC.PLAN_DIAG_NAME), encoding="utf-8"))
        check("score: A27 / A25.3 the exploratory Codex cells are plan-only: no score row, one plan-diagnostic row each (no SR, plan metrics)",
              not [r for r in sc_rows if r["arm"].startswith("X-")] and _diag["rows"] and all(r["plan_only"] and r["sr"] is None and r["arm"].startswith("X-") for r in _diag["rows"])
              and set(_diag["per_arm"]) == {r["arm"] for r in rows(rootA, "plan") if r["arm"].startswith("X-")})
        _dpath = os.path.join(C.ns_docs(NS, rootA), SC.PLAN_DIAG_NAME)
        _dsave = open(_dpath, "rb").read()
        with open(_dpath, "ab") as _f:
            _f.write(b" ")
        _tamper = raises(lambda: SC.run_score(NS, root=rootA, items="dev", log=QUIET), SC.ScoreIntegrity)
        with open(_dpath, "wb") as _f:
            _f.write(_dsave)
        check("score: A27 (Astra P-A27 #1) plan_diagnostics.json is sealed: a changed file after the complete seal is refused; restored, the seal is a no-op again",
              _tamper and SC.run_score(NS, root=rootA, items="dev", log=QUIET)["status"] == "already-done")

        ar = AN.run_sealed(C.journal_path(NS, "score", rootA), root=rootA, outdir=os.path.join(rootA, "out"), B=199)
        check("analyze --sealed: analysis.json + analysis.md written next to the seal's data", os.path.exists(os.path.join(rootA, "out", "analysis.json")))
        syn = AN.run_synthetic(B=199, outdir=os.path.join(rootA, "syn"))
        check("analyze --synthetic: runs and writes to its own folder", os.path.exists(os.path.join(rootA, "syn", "analysis.json")))
        final_A = {k: norm(rows(rootA, k)) for k in ("capture", "plan", "exec", "outcome", "score")}
        timeA = time.time() - t0
    finally:
        pass

    # ============================================================================================ run B: interrupted + resumed
    print("== end to end: interrupted and resumed in every phase")
    rootB, fs_doc = build_root()
    try:
        c, rc = capture(rootB, fs_doc, interrupt=10)
        check("B capture: injected interrupt -> exit 5, 10 rows, no phase_complete", rc == 5 and len(rows(rootB, "capture")) == 10 and not events(rootB, "phase_complete"))
        p, prc, _ = plan(rootB)
        check("B plan before capture is complete: refused", prc == 1 and not rows(rootB, "plan"), prc)
        c, rc = capture(rootB, fs_doc)
        check(f"B capture resumed: exit 0, {4 * len(DEV)} distinct rows", rc == 0 and len({r['key'] for r in rows(rootB, 'capture')}) == 4 * len(DEV) == len(rows(rootB, "capture")))
        p, prc, f1 = plan(rootB, interrupt=25)
        check("B plan: injected interrupt -> exit 5, 25 cells, no plan phase_complete", prc == 5 and len(rows(rootB, "plan")) == 25
              and not any(e.get("phase") == "plan" for e in events(rootB, "phase_complete")), (prc, len(rows(rootB, "plan"))))
        n_before = len(f1.calls)
        p, prc, f2 = plan(rootB)
        check("B plan resumed: exit 0, no cell planned twice", prc == 0 and len(rows(rootB, "plan")) == len({r["key"] for r in rows(rootB, "plan")}))
        xlog = []
        xph0 = X.ExecutePhase(NS, root=rootB, executor_factory=lambda phase: ExecEx(xlog), memory_opener=lambda phase, src, tag, ex: FakeMem(C.snapshot_dir(src, rootB)),
                              tolerances={"tau_pos_m": 0.05, "tau_rot_deg": 5.0}, worlds=("canon", "reloc"), reloc_seed=777, repeat_override=list(REPEAT), out=QUIET,
                              reset_patch=_ResetExec)
        info = xph0.run(stop_after=15)
        check("B execute: injected interrupt -> 15 executed, not complete", info["interrupted"] and not info["complete"] and info["executed"] == 15)
        sr_try = None
        try:
            SC.run_score(NS, root=rootB, items="dev", log=QUIET)
        except Exception as e:
            sr_try = type(e).__name__
        check("B score before execute is complete: refused", sr_try is not None and not rows(rootB, "score"), sr_try)
        xph1, info = execute(rootB, xlog)
        check("B execute resumed: complete; only the missing cells ran", info["complete"], info)
        sr = SC.run_score(NS, root=rootB, items="dev", log=QUIET)
        full = rows(rootB, "score")
        # score interrupt: keep the first 40 rows, drop the seal, resume
        sp = C.journal_path(NS, "score", rootB)
        sealp = os.path.join(C.ns_docs(NS, rootB), "score.seal.json")
        keep = open(sp, encoding="utf-8").read().splitlines(True)[:40]
        open(sp, "w", encoding="utf-8").write("".join(keep))
        os.remove(sealp)
        sr = SC.run_score(NS, root=rootB, items="dev", log=QUIET)
        check("B score: resumed from a truncated journal, sealed again, same rows", sr["status"] == "scored" and norm(rows(rootB, "score")) == norm(full), sr)
        final_B = {k: norm(rows(rootB, k)) for k in ("capture", "plan", "exec", "outcome", "score")}
        for k in ("capture", "plan", "exec", "outcome", "score"):
            same = final_A[k] == final_B[k]
            check(f"A == B ({k}): an interrupted and resumed run gives the same journal as an uninterrupted one", same,
                  "" if same else "len %d vs %d" % (len(final_A[k]), len(final_B[k])))
    finally:
        shutil.rmtree(rootB, ignore_errors=True)

    # ============================================================================================ cross-module contracts
    print("== cross-module contracts")
    cap_rows = rows(rootA, "capture")
    pl = rows(rootA, "plan")
    ex_rows = rows(rootA, "exec")
    units = json.load(open(os.path.join(C.ns_docs(NS, rootA), "units.json"), encoding="utf-8"))
    ukeys = set(units["units"]) if "units" in units else set(units)
    check("every exec unit key is a unit in units.json", {r["unit"] for r in ex_rows} <= ukeys)
    sc_rows = rows(rootA, "score")
    plan_keys = {r["key"] for r in pl if r["status"] == "ok" and not C.plan_only(r["arm"])}      # A27 / A25.3: plan-only cells are never executed
    scored_cells = {r["key"] for r in sc_rows} | {"|".join(r["key"].split("|")[:5]) for r in sc_rows}
    covered = {c for r in ex_rows for c in (r.get("cells") or [])}
    check("every ok plan cell (except the plan-only exploratory Codex cells) is covered by the cells of some exec record", plan_keys <= covered, sorted(plan_keys - covered)[:2])
    check("A27: no plan-only (X-*) cell is in any exec record", not [c for c in covered if C.plan_only(c.split("|")[2])])
    check("one plan record per (arm, pres, r): no duplicate keys in the plan journal", len(pl) == len({r["key"] for r in pl}))
    check("capture request shas are what plan verified: re-render of a sample equals the capture sha",
          all(PL.prompt_sha256(PL.render_messages(contract, r, "none")) == r["requests"]["none"] for r in cap_rows[:6]))
    for kind in ("capture", "plan", "exec", "events", "monitor"):
        p = C.journal_path(NS, kind, rootA)
        if os.path.exists(p):
            body = open(p, encoding="utf-8").read()
            check(f"{kind}.jsonl carries no key-like text and no FloorPlan name", not C.key_leak(body) and "FloorPlan" not in body)
    check("exec/plan/capture/events/monitor journals hold no gold or outcome key",
          all(C.forbidden_keys(r, C.GOLD_KEYS | C.OUTCOME_KEYS) == [] for k in ("capture", "plan", "exec", "events") for r in rows(rootA, k)))
    sr_body = open(C.journal_path(NS, "score", rootA), encoding="utf-8").read()
    check("score.jsonl carries no key-like text", not C.key_leak(sr_body))
    # holding_before: the adapter's step_start events -> the slice-realism flag in score
    stream = [{"kind": "step_start", "i": 0, "pre": {"holding": "ButterKnife"}}, {"kind": "step_start", "i": 1, "pre": {}},   # A26: was Knife
              {"kind": "step_start", "i": 2, "pre": {"holding": "?"}}]
    wh = X.with_holding([{"i": 0, "action": "slice", "object": "Bread", "ok": True, "prims": 3}, {"i": 1, "action": "slice", "object": "Bread", "ok": True, "prims": 3},
                         {"i": 2, "action": "slice", "object": "Bread", "ok": True, "prims": 3}, {"i": 3, "action": "slice", "ok": False, "prims": 0}], stream)
    check("execute.with_holding: belief before each step from step_start (None = nothing held / never started)",
          [s["holding_before"] for s in wh] == ["ButterKnife", None, "?", None], wh)
    check("score.slice_realism: ButterKnife held -> realistic (A26; was 'Knife held -> realistic')", SC.slice_realism(wh[:1], True)["unrealistic_slice"] is False)
    check("score.slice_realism (A26): the Knife held -> unrealistic (no cutting tool)",
          SC.slice_realism([dict(wh[0], holding_before="Knife")], True)["unrealistic_slice"] is True)
    check("score.slice_realism: nothing held at a slice that ran -> unrealistic", SC.slice_realism(wh[1:2], True)["unrealistic_slice"] is True)
    check("score.slice_realism: held but type unknown ('?') -> unknown (None), never a false 'no knife'", SC.slice_realism(wh[2:3], True)["unrealistic_slice"] is None)
    check("score.slice_realism: a slice step that never ran is ignored", SC.slice_realism(wh[3:], True)["unrealistic_slice"] is False)
    # score's unit-key fallback agrees with execute's snapshot sha scheme
    check("score._snapshot_sha == execute.dir_sha of the snapshot directory (unit keys agree without the cells list)",
          SC._snapshot_sha("S*", rootA) == X.dir_sha(C.snapshot_dir("S*", rootA)) and SC._snapshot_sha("S*", rootA) not in (None, "absent"))
    st = MR.status(NS, rootA)
    check("matrix_run status: capture, plan, execute, score complete; analyze not yet (nothing written next to the seal)",
          all(st["phases"][p]["complete"] for p in ("capture", "plan", "execute", "score")) and st["next"] == "analyze", st)

    # ============================================================================================ M-fix2
    mfix2_block(locals())

    # ============================================================================================ CLI
    print("== matrix_run CLI")
    cli_tests(rootA, NS, MR, C, K, PL, X, SC, MO, AN, build_root, fs_doc, capture)
    print(f"(end-to-end wall time {timeA:.1f}s for the uninterrupted pipeline)")
    shutil.rmtree(rootA, ignore_errors=True)


def mfix2_block(L):
    """The M-fix2 contracts, each on the real modules. L = the locals of e2e() (fakes, helpers, rootA = a finished uninterrupted run)."""
    import common as C
    import capture as K
    import plan as PL
    import execute as X
    import score as SC
    import analyze as AN
    import matrix_run as MR
    import fs_block as FB
    import final_test_callplan as CP
    from planners.keys import Secret
    NS, rootA, MAN, DEV, REPEAT = L["NS"], L["rootA"], L["MAN"], L["DEV"], L["REPEAT"]
    build_root, capture, plan, execute = L["build_root"], L["capture"], L["plan"], L["execute"]
    rows, events, Fake, CapExecutor, ExecEx, FakeCtl, FAKEKEY = L["rows"], L["events"], L["Fake"], L["CapExecutor"], L["ExecEx"], L["FakeCtl"], L["FAKEKEY"]
    contract, factory_for, FakeCounter, QUIET = L["contract"], L["factory_for"], L["FakeCounter"], L["QUIET"]
    ExecEx_, _ResetExec, FakeMem = L["ExecEx"], L["_ResetExec"], L["FakeMem"]

    print("== M-fix2: fs_block reads the journal practice_k really writes")
    pool = sorted(MAN["practice_pool"])[:10]
    prac = []
    for n, iid in enumerate(pool, 1):                                       # practice_k.practice_order numbers attempts 1..10
        r = {"order": 0, "attempt": n, "item": iid, "lang": "en" if n % 2 else "fa", "instruction": "do thing %d" % n, "status": "executed",
             "goal_ok": n % 3 != 0, "admission": n % 3 != 0, "plan": [{"action": "open", "object": "Fridge", "to": None}],
             "exec_steps": [{"ok": True}], "sightings": [{"otype": "Apple", "in_rtype": "Fridge", "state": "closed"}]}
        prac.append(r)
    prac[3].update(status="crash", goal_ok=None, admission=None, crash="RuntimeError: x", exec_steps=None, sightings=[], plan=None)
    prac[5].update(status="rejected", goal_ok=None, admission=False, exec_steps=None)
    prac[7].update(status="inapplicable", goal_ok=None, admission=None, exec_steps=None, sightings=[])
    ctr = lambda t: len(t) // 4 + 1                                         # noqa: E731
    try:
        nm = FB.normalise_rows(prac)
        ok_norm = [a["attempt"] for a in nm] == list(range(10))
    except Exception as e:                                                  # noqa: BLE001
        nm, ok_norm = [], False
    check("fs_block: attempts 1..10 (practice_k numbering) are accepted, the 10th is kept, all ten are normalised", ok_norm)
    check("fs_block: otype / in_rtype sightings are read (none silently dropped)", nm and sum(len(a["sights"]) for a in nm) == 8 and nm[0]["sights"] == [("Apple", "Fridge")])
    check("fs_block: a crash / rejected / inapplicable started attempt gets an explicit verdict, not a ValueError",
          nm and [nm[3]["verdict"], nm[5]["verdict"], nm[7]["verdict"]] == ["crash", "failure", "inapplicable"], [a["verdict"] for a in nm])
    blk = FB.render(prac, "L2", ctr, 4000, namespace="practice-v1") if nm else {}
    check("fs_block: the rendered block holds all ten attempts and its sightings part is not empty",
          blk and blk["retained"]["attempts"] == 10 and blk["retained"]["sightings"] >= 1 and "Apple -> Fridge" in blk["text"], blk.get("retained"))
    bad = [dict(r, sightings=[{"foo": 1}]) if r["attempt"] == 2 else r for r in prac]
    check("fs_block: a sighting with no recognised keys is refused, not skipped", raises(lambda: FB.normalise_rows(bad), ValueError))
    check("fs_block: the older 0..9 numbering is still accepted when complete", len(FB.normalise_rows([dict(r, attempt=r["attempt"] - 1) for r in prac])) == 10)
    check("fs_block: one attempt too few is refused", raises(lambda: FB.normalise_rows(prac[:9]), ValueError))

    print("== M-fix2: every arm x r is an independent planner call (cell-keyed cache)")
    cc = PL.CellCache("x", "ns")
    k0 = cc.key("anthropic", "m", "sha", {"t": 0}, 0, {"s": 1}, "plan")
    cc.cell = "ns|plan|O0|canon:a:en|r0"
    k1 = cc.key("anthropic", "m", "sha", {"t": 0}, 0, {"s": 1}, "plan")
    cc.cell = "ns|plan|O+M|canon:a:en|r0"
    k2 = cc.key("anthropic", "m", "sha", {"t": 0}, 0, {"s": 1}, "plan")
    cc.cell = "ns|plan|O0|canon:a:en|r0"
    check("CellCache: the same request in two cells has two keys; the same cell keeps its key (resume replays its own answer)",
          len({k0, k1, k2}) == 3 and cc.key("anthropic", "m", "sha", {"t": 0}, 0, {"s": 1}, "plan") == k1)
    pl = rows(rootA, "plan")
    check("plan (e2e run): no cell was answered from the cache in a fresh run (cached_calls == 0 everywhere)", pl and all(r["cached_calls"] == 0 for r in pl),
          sum(1 for r in pl if r["cached_calls"]))

    print("== M-fix2: cut groups, replay rows, unrecoverable presentations, reloc casualties, end to end")
    check("A23 cut order: codex, curve12, reloc; haiku is NOT a cuttable group any more; core / gemma / fs cannot be named; no skipping",
          C.cut_problem(set(), ["codex"]) is None and C.cut_problem({"codex"}, ["curve12"]) is None and C.cut_problem(set(), ["curve12"]) is not None
          and C.cut_problem({"codex"}, ["haiku"]) is not None and C.cut_problem(set(), ["haiku"]) is not None
          and C.cut_problem(set(), ["core"]) is not None and C.cut_problem(set(), ["reloc"]) is not None
          and C.cut_group("O+M@o0k1") is None and C.cut_group("O+M@o1k3") == "curve12" and C.cut_group("X-gpt-6-sol0") == "codex"
          and C.cut_group("H+M") is None and C.cut_group("G0", "reloc:a:en") == "reloc" and C.cut_group("G0", "canon:a:en") is None)
    rootD, fsD = build_root()
    try:
        casualty = C.pres_id(DEV[0], "en", "reloc")
        c, rc = capture(rootD, fsD, executor=CapExecutor(inapp={casualty}))
        crow = [r for r in rows(rootD, "capture") if r["pres"] == casualty]
        check("capture: a relocated setup failure is re-run 3 times, journaled inapplicable, does not fail the phase (canon would)",
              rc == 0 and crow and crow[0]["status"] == "inapplicable" and crow[0]["tries"] == 3
              and events(rootD, "phase_complete")[-1]["reloc_inapplicable"] == [casualty])
        mk = lambda fake: PL.Plan(NS, passes=None, root=rootD, provider_factory=factory_for(fake), ollama_ctl=FakeCtl(), gpu_sample=False,   # noqa: E731
                                  key_getter=lambda: Secret(FAKEKEY), out=QUIET, manifest=MAN, types_path=os.path.join(rootD, "types.json"))
        try:
            mk(Fake()).record_cut(["curve12"], "x")
            skip_ok = False
        except ValueError:
            skip_ok = True
        check("plan --cut: curve12 before codex is refused (registered order); haiku is not cuttable; a cut needs a note",
              skip_ok and raises(lambda: mk(Fake()).record_cut(["haiku"], "x"), ValueError) and raises(lambda: mk(Fake()).record_cut(["codex"], ""), ValueError))
        cp_before = CP.remaining(NS, rootD)
        mk(Fake()).record_cut(["codex"], "test: budget decision outcome-blind")
        cp_after = CP.remaining(NS, rootD)
        check("callplan --remaining: the paid cells still to run per planner (S, O, H) with a mean cost; cut groups leave it",
              cp_before["remaining"].get("O", 0) > 0 and set(cp_before["mean_cost"]) == {"O", "S", "H"} and "H" in cp_before["remaining"]
              and cp_after["cut_groups"] == ["codex"] and cp_after["cut_cells"] > 0, (cp_before, cp_after))
        fk = Fake(garble_gemma_memory=True)
        p = PL.Plan(NS, passes=None, root=rootD, provider_factory=factory_for(fk), ollama_ctl=FakeCtl(), gpu_sample=False,
                    key_getter=lambda: Secret(FAKEKEY), out=QUIET, manifest=MAN, types_path=os.path.join(rootD, "types.json"))
        prc = p.run()
        pld = rows(rootD, "plan")
        pc = [e for e in events(rootD, "phase_complete") if e.get("phase") == "plan"]
        check("plan after a cut: exit 0 and phase_complete WITHOUT the cut cells (the core study is complete)",
              prc == 0 and pc and pc[-1]["cut_groups"] == ["codex"] and pc[-1]["cut_cells"] > 0, (prc, pc[-1:] if pc else None))
        check("plan after a cut: no codex row; the casualty presentation has no cell",
              not [r for r in pld if r["arm"].startswith("X-")] and not [r for r in pld if r["pres"] == casualty])
        # A27: the learning curve has no order 1 / 2 cells any more, so curve12 is an empty group; the "cut AFTER plan rows exist" path is exercised
        # with the next group in the registered order, reloc, whose PLANNING rows already exist here (score must drop them, never score 0).
        # This does not test a cut after execution or after a score seal (Astra P-A27 #5).
        mk(Fake()).record_cut(["curve12", "reloc"], "test: second step, after the relocation planning rows already existed")      # cut AFTER plan rows exist
        gm = [r for r in pld if r["arm"] == "G+M" and r["block"] == "main"]
        check("plan: the garbled Gemma replies make every G+M plan rejected (setup of the replay test)", gm and all(r["status"] == "rejected" for r in gm),
              ({r["status"] for r in gm}, [a["raw"] for a in gm[0]["attempts"]][:2] if gm else None))
        boom = {C.pres_id(REPEAT[0], "en")}
        xlog = []
        try:
            execute(rootD, xlog, boom=boom)
            aborted = False
        except X.ExecAborted:
            aborted = True
        check("execute: an INVALID run (broken honesty invariant) aborts the phase", aborted and not [e for e in events(rootD, "phase_complete") if e.get("phase") == "execute"])
        xph = X.ExecutePhase(NS, root=rootD, executor_factory=lambda phase: ExecEx(xlog, boom), memory_opener=lambda phase, src, tag, ex: FakeMem(C.snapshot_dir(src, rootD)),
                             tolerances={"tau_pos_m": 0.05, "tau_rot_deg": 5.0}, worlds=("canon", "reloc"), reloc_seed=777, repeat_override=list(REPEAT),
                             out=QUIET, reset_patch=_ResetExec)
        check("execute: without a declaration the invalid presentation keeps blocking (resume refuses)", raises(lambda: xph.run(), X.ExecAborted))
        d = xph.declare_unrecoverable("A99")
        info = xph.run()
        check("execute: declared unrecoverable (prereg s13) -> the phase completes without it", d["declared"] == sorted(boom) and info["complete"], (d, info))
        ujs = json.load(open(os.path.join(C.ns_docs(NS, rootD), "units.json"), encoding="utf-8"))
        gmp = [r for r in pld if r["arm"] == "G+M" and r["status"] == "rejected" and C.cut_group(r["arm"], r["pres"]) is None]   # A27: the reloc rows are cut here
        check("execute: the /p replay cells of the rejected G+M plans are persisted in units.json (they must score SR 0, not vanish)",
              len(ujs["unexecuted_replay"]) == len(gmp) and {u["arm"] for u in ujs["unexecuted_replay"]} == {"G+M/p"}
              and ujs["cut_groups"] == ["codex", "curve12", "reloc"] and "repeat_rule" in ujs and not [x for x in ujs["units"].values() if x["cells"] and
                                                                                       any(("@o1" in c["arm"] or "@o2" in c["arm"]) for c in x["cells"])], (len(ujs["unexecuted_replay"]), len(gmp)))
        sr = SC.run_score(NS, root=rootD, items="dev", log=QUIET)
        scd = rows(rootD, "score")
        seal = json.load(open(os.path.join(C.ns_docs(NS, rootD), "score.seal.json"), encoding="utf-8"))
        check("score: complete seal with the cut recorded (cut cells are neither scored 0 nor missing)",
              sr["status"] == "scored" and seal["complete"] and seal["cut_groups"] == ["codex", "curve12", "reloc"] and seal["n_cut_cells"] > 0, (sr, seal.get("cut_groups")))
        rep0 = [r for r in scd if r["arm"] == "G+M/p" and r["run"] == 0]
        check("score: every G+M/p cell of a rejected +M plan has an SR-0 row with plan_status rejected (replay_zero)",
              len(rep0) >= len(gmp) and all(r["sr"] == 0.0 and r["plan_status"] == "rejected" and r["failure_class"] == "plan_rejected" for r in rep0)
              and seal["n_replay_zero_rows"] == len(gmp), (len(rep0), len(gmp), seal.get("n_replay_zero_rows")))
        check("score: no codex / curve12 / reloc score row after the cut (the relocation plan rows existed, so they were dropped, not scored 0); A27: no order 1 / 2 curve plan row exists at all; the archived Haiku cells are scored normally",
              not [r for r in scd if r["arm"].startswith("X-") or "@o1" in r["arm"] or "@o2" in r["arm"] or C.parse_pres(r["pres"])["world"] == "reloc"]
              and any(C.parse_pres(r["pres"])["world"] == "reloc" for r in pld) and not any("@o1" in r["arm"] or "@o2" in r["arm"] for r in pld)
              and any(r["arm"].startswith("H") for r in scd))
        inv = [r for r in scd if r["pres"] in boom]
        check("score: the unrecoverable presentation is scored with invalid_episode = True in every row, and nothing else is flagged",
              inv and all(r["invalid_episode"] for r in inv) and not [r for r in scd if r["pres"] not in boom and r["invalid_episode"]]
              and seal["invalid_episode_pres"] == sorted(boom))
        ar = AN.run_sealed(C.journal_path(NS, "score", rootD), root=rootD, outdir=os.path.join(rootD, "out"), B=199)
        res = ar[0]
        check("analyze: the unrecoverable presentation leaves EVERY arm (keeps the pairing)", res["invalid_presentations_dropped"] == 1, res["invalid_presentations_dropped"])
        sc2 = {(r["arm"], C.parse_pres(r["pres"])["item"], C.parse_pres(r["pres"])["lang"]): r for r in scd
               if r["r"] == 0 and r["run"] == 0 and C.parse_pres(r["pres"])["world"] == "canon"}
        drop = {(DEV_ITEM, "en") for DEV_ITEM in [REPEAT[0]]}
        per = {}
        for (a, i, lg), r in sc2.items():
            if a == "G+M" and ("O0", i, lg) in sc2 and (i, lg) not in drop:
                per.setdefault(i, []).append(float(r["sr"]) - float(sc2[("O0", i, lg)]["sr"]))
        hand = sum(sum(v) / len(v) for v in per.values()) / len(per)
        got = res["H1"]["primary"]["estimate"]
        check("analyze: the G+M - O0 estimate equals the hand-computed value over the retained presentations", abs(hand - got) < 1e-9, (hand, got))
        check("analyze: a complete study prints a registered H1 wording (not the incomplete one)", res["H1"]["wording"] != AN.INCOMPLETE_WORDING and res["complete"])
        # execute internals
        check("execute: the exec index agrees with the journal after the run, and nothing is left to do",
              sum(len(v) for v in xph._exec_ix().values()) == len(rows(rootD, "exec")) and all(not b for b in xph.todo().values()))
        try:
            xph._append("exec", {"key": "k", "status": "done", "error": "a task-specific plan failed"})
            fp_ok = True
        except X.ExecInvalid:
            fp_ok = False
        xph._secrets_cache = (Secret(FAKEKEY),)
        check("execute: a hyphenated word in free text is no key leak; the exact secret is", fp_ok and raises(
            lambda: xph._append("exec", {"key": "k2", "status": "done", "error": "boom " + FAKEKEY}), X.ExecInvalid))
        cs = C.Journal(C.journal_path(NS, "events", rootD)).rows()
        pinned = [e for e in cs if e.get("event") == "capture_start"][-1].get("dataset_sha256")
        check("capture records the dataset sha; the dataset check passes unchanged and fails on another sha",
              pinned and C.dataset_problems(NS, rootD, pinned) == [] and C.dataset_problems(NS, rootD, "0" * 64) != [])
    finally:
        shutil.rmtree(rootD, ignore_errors=True)

    print("== M-fix2: analysis rules")
    sc = rows(rootA, "score")
    fam_of = {i: f["family"] for f in C.load_manifest()["families"] for i in f["items"]}
    inc = AN.analyse(sc, fam_of_manifest=fam_of, B=199, meta={"namespace": NS, "complete": False})
    md = AN.render_md(inc)

    def no_confirmatory(h1):
        """C9: since A24.1 the H1 decision is h1-rule-1 (claims / wording, no p_ni). An incomplete study carries NO confirmatory decision:
        the primary decision must exist, be descriptive_only with claims None (the key itself, so a missing field cannot pass), and hold
        none of the confirmatory fields of either rule (old p_ni / p_sup, h1-rule-1 fixed_sequence / estimation_only); the same for each
        sensitivity that is present."""
        conf = ("p_ni", "p_sup", "fixed_sequence", "estimation_only")
        decs = [(h1.get("primary") or {}).get("decision")] + [h1[k]["decision"] for k in ("family_weighted", "infra_paired") if h1.get(k)]
        return all(isinstance(d, dict) and d.get("descriptive_only") is True and "claims" in d and d["claims"] is None
                   and d.get("wording") == AN.INCOMPLETE_WORDING and not any(k in d for k in conf) for d in decs)

    check("analyze: an INCOMPLETE study never prints a confirmatory H1 wording and rejects nothing (prereg s13)",
          inc["H1"]["wording"] == AN.INCOMPLETE_WORDING and all(v.get("reject") is None for v in inc["holm_family"].values() if v.get("available"))
          and not any(w in md for w in ("Wording: superior", "non-inferior (within", "materially inferior"))
          and no_confirmatory(inc["H1"]),
          inc["H1"]["wording"])
    fs_dec = AN.fixed_sequence({"wording": "superior", "p_ni": 0.4, "ci95": [0.01, 0.08]})
    ok_dec = AN.fixed_sequence({"wording": "superior", "p_ni": 0.001, "ci95": [0.01, 0.08]})
    check("analyze: 'superior' without non-inferiority is downgraded (fixed sequence); with it, kept",
          fs_dec["wording"] == "non-inferiority unresolved (inconclusive)" and fs_dec["fixed_sequence"]["downgraded"] and ok_dec["wording"] == "superior")
    mk_row = lambda arm, item, lg, sr, **kw: dict({"arm": arm, "pres": C.pres_id(item, lg), "r": 0, "run": 0, "sr": sr, "strict_sr": sr, "valid": True,   # noqa: E731
                                                  "level": "L1", "family": "f" + item[-1], "lang": lg, "unrealistic_slice": False, "n_slice": 0}, **kw)
    S_ = {}
    for i in ("a1", "a2", "b1", "b2"):
        for lg in ("en", "fa"):
            S_[("O0", i, lg)] = mk_row("O0", i, lg, 1.0, unrealistic_slice=(True if (i, lg) == ("a1", "en") else None if (i, lg) == ("a2", "fa") else False))
    rs = AN.realistic_sr(S_, {"a1": "fa", "a2": "fa", "b1": "fb", "b2": "fb"}, {}, "O0", set())
    fc = AN.slice_flag_counts(S_, "O0", set())
    check("analyze: prerequisite-realistic SR zeroes the known-unrealistic slice success only; unknown (None) rows are counted, not zeroed",
          rs and abs(rs["sr"] - (1 - 0.5 / 4)) < 1e-9 and fc["unrealistic_true"] == 1 and fc["unknown_none"] == 1, (rs, fc))
    miss_rows = [r for r in sc if not (r["arm"] == "O0" and r["pres"] == C.pres_id(DEV[1], "fa") and r["r"] == 0 and r["run"] == 0)]
    mp = AN.completeness_problems(miss_rows, C.load_manifest(), "rehearsal-v1")
    check("analyze: a complete seal with a missing core presentation is a completeness problem (the estimand would shift)", mp and "O0" in mp[0], mp)
    check("analyze: a flagged INVALID presentation is exempt from that check", not AN.completeness_problems(miss_rows, dict(C.load_manifest()), "rehearsal-v1",
                                                                                                          {(DEV[1], "fa")}))

    print("== M-fix2: budget, key, capture, matrix_run")
    led_path = os.path.join(tempfile.mkdtemp(prefix="ftm-led-"), "l.db")
    done_flag = {"v": False}
    led = C.ScopedLedger(led_path, {"core": 25.0, "total": 45.0}, core_done=lambda: done_flag["v"])
    with led.use("core"):
        led.reserve("anthropic", 6.0)
    with led.use("other"):
        a_ok = led.reserve("anthropic", 20.0)                      # 6 + 20 + (25 - 6) = 45 -> exactly fits
        refused = raises(lambda: led.reserve("anthropic", 1.0), CostCeilingExceeded_())
    done_flag["v"] = True
    with led.use("other"):
        later = led.reserve("anthropic", 5.0)
    check("ledger: while the core is unfinished, non-core spend may use only what the core ceiling leaves unreserved inside the total; "
          "once every core cell is terminal the reserve ends", a_ok and refused and later)
    with led.use("other"):
        rid = led.reserve("anthropic", 0.17)
    n0 = led.open_reservations()
    res = led.reconcile_open("test: ambiguous 5xx, billing console shows nothing")
    check("ledger: stale open per-call reservations are listed, and an audited reconcile releases them (a note is mandatory)",
          n0["n"] >= 1 and led.open_reservations()["n"] == 0 and rid in res["released"] and raises(lambda: led.reconcile_open(""), ValueError))
    old = os.environ.get("THESIS_ANTHROPIC_API_KEY")
    old_plain = os.environ.get("ANTHROPIC_API_KEY")
    try:
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-NOTTHESTUDYKEY0123456789abcdef"
        os.environ.pop("THESIS_ANTHROPIC_API_KEY", None)
        got = None
        try:
            got = C.thesis_key().reveal() if os.name != "nt" else None
        except Exception:                                           # noqa: BLE001
            got = "skipped"
        check("thesis_key: the generic ANTHROPIC_API_KEY (Claude Code's own) is never used for the study", got in (None, "skipped"))
        os.environ["THESIS_ANTHROPIC_API_KEY"] = FAKEKEY
        env = C.wsl_key_env({"PATH": "x", "WSLENV": "FOO/p"})
        check("wsl_key_env: the key travels through WSLENV (named, with /u), not on the command line", env["WSLENV"] == "FOO/p:THESIS_ANTHROPIC_API_KEY/u"
              and env["THESIS_ANTHROPIC_API_KEY"] == FAKEKEY)
        env2 = MR.wsl_env("capture", "test-v1", ["--ollama", "http://1.2.3.4:11434"])
        check("matrix_run.wsl_env: capture in test-v1 forwards the key; without --ollama or with --allow-missing-anthropic it refuses",
              env2 and "THESIS_ANTHROPIC_API_KEY/u" in env2["WSLENV"]
              and raises(lambda: MR.wsl_env("capture", "test-v1", []), MR.Refused)
              and raises(lambda: MR.wsl_env("capture", "test-v1", ["--ollama", "u", "--allow-missing-anthropic"]), MR.Refused)
              and MR.wsl_env("plan", "test-v1", []) is None)
    finally:
        for k, v in (("THESIS_ANTHROPIC_API_KEY", old), ("ANTHROPIC_API_KEY", old_plain)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    check("capture.a12_gate: required (test-v1) fails when presentations are unchecked; dev passes only if nothing could be checked",
          not K.a12_gate({"ok": True, "unchecked": 3, "n": 3}, True) and K.a12_gate({"ok": True, "unchecked": 0, "n": 3}, True)
          and K.a12_gate({"ok": True, "unchecked": 3, "n": 3}, False) and not K.a12_gate({"ok": False, "unchecked": 0, "n": 3}, False))
    capT = K.Capturer(namespace="test-v1", executor=None, contract=None, counter=FakeCounter(), open_sources=None, items={}, manifest=MAN,
                      channels=("places",), freeze={}, params={}, root=tempfile.mkdtemp(prefix="ftm-capT-"))
    p_a12 = {"retrieval": {"a12": {"N": {}, "B": {}}}}
    probs = []
    # TREAT-FREEZE 2026-10-13: test-v1 now runs ft-v2 (A24.1-T item 4 (B)), so a test-v1 Capturer takes the ft-v2 branch; the legacy
    # ft-v1 branch below is exercised explicitly (ft2 False), and the ft-v2 branch is checked right after it.
    capT_ft2 = capT.ft2
    capT.ft2 = False
    for fr, ch, pm in (({}, ("places",), p_a12), ({"channels": ["places"]}, ("places", "procedures"), p_a12), ({"channels": ["places"]}, ("places",), {}),
                       ({"channels": ["places"]}, ("places",), p_a12)):
        capT.freeze, capT.channels, capT.params = fr, ch, pm
        try:
            capT.test_v1_checks()
            probs.append(None)
        except K.CaptureAbort as e:
            probs.append(str(e)[:40])
    check("capture (test-v1): no frozen channels / a CLI channel set that differs from the frozen one / no A12 allocation each abort; the right set passes",
          probs[0] and probs[1] and probs[2] and probs[3] is None, probs)
    probs2 = []
    capT.ft2, capT.ident, capT.params = True, {"sha": "I" * 64}, {}
    for fr in ({}, {"treatment": {"version": "ft-v2", "sha": "J" * 64}}, {"treatment": {"version": "ft-v2", "sha": "I" * 64}, "freeze_status": K.T.TREATMENT_ONLY},
               {"treatment": {"version": "ft-v2", "sha": "I" * 64}}):
        capT.freeze = fr
        try:
            capT.test_v1_checks()
            probs2.append(None)
        except K.CaptureAbort as e:
            probs2.append(str(e)[:40])
    check("capture (test-v1, ft-v2 after the TREAT-FREEZE flip): the Capturer is ft-v2; no treatment record / a different identity / a "
          "treatment-only freeze each abort; the matching identity passes", capT_ft2 is True and probs2[0] and probs2[1] and probs2[2]
          and probs2[3] is None, probs2)
    check("repeat / relocation sets: test-v1 = the split's sets; a rehearsal without declared sets gets the declared dev subset (4 dev items each)",
          C.repeat_items("test-v1", C.load_manifest()) == sorted(C.load_manifest()["repeat_subset"])
          and len(C.repeat_items("rehearsal-v1", C.load_manifest())) == 4 and set(C.repeat_items("rehearsal-v1", C.load_manifest())) <= set(C.load_manifest()["dev_pilot"])
          and len(C.reloc_items("rehearsal-v1", C.load_manifest())) == 4 and C.repeat_items("practice-v1", C.load_manifest()) == [])


def CostCeilingExceeded_():
    from planners.base import CostCeilingExceeded
    return CostCeilingExceeded


def cli_tests(rootA, NS, MR, C, K, PL, X, SC, MO, AN, build_root, fs_doc, capture):
    def run(argv, root=rootA):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = MR.main(argv, root=root)
        return rc, out.getvalue(), err.getvalue()

    rc, out, err = run(["--phase", "status", "--items", "dev", "--namespace", NS])
    check("CLI status: prints a json state", rc == 0 and json.loads(out)["namespace"] == NS)
    rc, out, err = run(["--phase", "score", "--items", "dev", "--namespace", NS])
    check("CLI score on an already sealed namespace: no-op, exit 0", rc == 0, (rc, err[-200:]))
    rc, out, err = run(["--phase", "analyze", "--items", "dev", "--namespace", NS])
    check("CLI analyze: reads the sealed score journal and writes analysis.json beside it", rc == 0 and os.path.exists(os.path.join(C.ns_docs(NS, rootA), "analysis.json")), (rc, err[-300:], out[-200:]))
    rc, out, err = run(["--phase", "monitor", "--items", "dev", "--namespace", NS, "--", "--no-write"])
    check("CLI monitor: --once by default, exit 0/1", rc in (0, 1), (rc, out[-200:], err[-200:]))
    # refusals
    rc, out, err = run(["--phase", "plan", "--items", "test", "--namespace", NS])
    check("CLI refuses --items test on a non-test namespace (10)", rc == 10 and "does not fit" in err, (rc, err))
    rc, out, err = run(["--phase", "plan", "--items", "dev", "--namespace", "test-v1"])
    check("CLI refuses --items dev on test-v1 (10)", rc == 10, (rc, err))
    rc, out, err = run(["--phase", "plan", "--items", "test", "--namespace", "test-v1"])
    check("CLI refuses test items without --allow-test (10, TestItemRefused text)", rc == 10 and "refused" in err.lower(), (rc, err))
    rc, out, err = run(["--phase", "plan", "--items", "test", "--namespace", "test-v1", "--allow-test"])
    check("CLI refuses test items with --allow-test but no freeze.json in this root (10)", rc == 10 and "freeze" in err.lower(), (rc, err))
    rc, out, err = run(["--phase", "execute", "--items", "dev", "--namespace", "practice-v1"])
    check("CLI refuses matrix phases in practice-v1 (10)", rc == 10, (rc, err))
    rc, out, err = run(["--phase", "plan", "--items", "dev", "--namespace", "no-such-ns"])
    check("CLI refuses an unknown namespace (10)", rc == 10, (rc, err))
    rc, out, err = run(["--phase", "score", "--items", "dev", "--inject-interrupt", "3", "--namespace", NS])
    check("CLI: --inject-interrupt on score is a usage error (12)", rc == 12, (rc, err))
    # dependency refusals in an empty root
    root2, fs2 = build_root()
    try:
        rc, out, err = run(["--phase", "plan", "--items", "dev", "--namespace", NS], root=root2)
        check("CLI plan with no capture: refused (10, upstream not ready)", rc == 10 and "upstream" in err, (rc, err))
        rc, out, err = run(["--phase", "execute", "--items", "dev", "--namespace", NS, "--print-command"], root=root2)
        check("CLI execute with no capture/plan: refused before the platform check (10)", rc == 10, (rc, err))
        rc, out, err = run(["--phase", "score", "--items", "dev", "--namespace", NS], root=root2)
        check("CLI score with nothing upstream: refused (10)", rc == 10, (rc, err))
        rc, out, err = run(["--phase", "analyze", "--items", "dev", "--namespace", NS], root=root2)
        check("CLI analyze with no seal: refused (10)", rc == 10, (rc, err))
        # a held lock
        lk = os.path.join(C.ns_work(NS, root2), "capture.lock")
        os.makedirs(os.path.dirname(lk), exist_ok=True)
        open(lk, "w").write('{"pid": 1}')
        rc, out, err = run(["--phase", "capture", "--items", "dev", "--namespace", NS, "--dry-run"], root=root2)
        check("CLI: a held phase lock is refused unless --steal-lock (10)", rc == 10 and "lock" in err.lower(), (rc, err))
        os.remove(lk)
        # capture dry run / platform
        rc, out, err = run(["--phase", "capture", "--items", "dev", "--namespace", NS, "--dry-run"], root=root2)
        check("CLI capture on this platform: either dry-run output (WSL) or the printed WSL command (Windows)", rc in (0, 2, 11) or out, (rc, out[-200:], err[-200:]))
        rc, out, err = run(["--phase", "capture", "--items", "dev", "--namespace", NS, "--print-command"], root=root2)
        d = json.loads(out) if out.strip().startswith("{") else {}
        check("CLI --print-command: a WSL command as a list (no shell), env vars before python, repo path with the apostrophe unquoted",
              rc == 0 and d.get("wsl", [None])[0] == "wsl" and "--cd" in d["wsl"] and "/home/arash/thor-env/bin/python" in d["wsl"]
              and any("MESA_D3D12" in x for x in d["wsl"]) and "Master's Project" in " ".join(d["wsl"]), (rc, out[-300:]))
        check("CLI --print-command: the argv carries the namespace, the items and the root", d.get("argv", [])[:4] == ["--namespace", NS, "--items", "dev"])
        # invalid abort blocks execute/score
        C.Journal(C.journal_path(NS, "events", root2)).append(C.make_record("events", event="abort_invalid", phase="execute"))
        check("open_invalid: an abort_invalid without invalid_resolved is open", MR.open_invalid(NS, root2))
        C.Journal(C.journal_path(NS, "events", root2)).append(C.make_record("events", event="invalid_resolved", phase="execute", pres=[], amendment="A99"))
        check("open_invalid: closed by invalid_resolved", not MR.open_invalid(NS, root2))
    finally:
        shutil.rmtree(root2, ignore_errors=True)
    # module argv builder
    a = MR.module_argv("plan", NS, "dev", allow_test=True, steal_lock=True, dry_run=True, interrupt=7, extra=["--pass", "core"])
    check("module_argv(plan): namespace, flags, interrupt, extras, no --items", a == ["--namespace", NS, "--allow-test", "--steal-lock", "--dry-run", "--inject-interrupt", "7", "--pass", "core"], a)
    a = MR.module_argv("capture", NS, "dev", interrupt=3)
    check("module_argv(capture): --items given, interrupt passed", a == ["--namespace", NS, "--items", "dev", "--inject-interrupt", "3"], a)
    a = MR.module_argv("monitor", NS, "dev")
    check("module_argv(monitor): --once is the default", a[-1] == "--once")
    # tamper: a changed upstream journal is refused
    shutil.copy(C.journal_path(NS, "capture", rootA), C.journal_path(NS, "capture", rootA) + ".bak")
    try:
        with open(C.journal_path(NS, "capture", rootA), "ab") as f:
            f.write(b"\n")
        d = MR.dependency_state(NS, "plan", rootA)
        check("dependency_state: a capture journal changed after its phase_complete is flagged", d["capture"]["unchanged"] is False, d)
    finally:
        shutil.move(C.journal_path(NS, "capture", rootA) + ".bak", C.journal_path(NS, "capture", rootA))


# =============================================================================================================== static
def static():
    print("== static")
    src = open(os.path.join(HERE, "final_test", "matrix", "matrix_run.py"), encoding="utf-8").read()
    import ast
    try:
        ast.parse(src, feature_version=(3, 10))
        ok = True
    except SyntaxError:
        ok = False
    check("matrix_run.py parses as python 3.10", ok)
    check("matrix_run.py has LF line endings", "\r" not in src)
    check("matrix_run.py never reads or prints a key", "THESIS_ANTHROPIC_API_KEY" not in src and "get_anthropic_key" not in src)
    check("matrix_run.py never opens outcome.sealed.jsonl or score.jsonl itself", "outcome.sealed" not in src and 'open(score' not in src)
    check("matrix_run.py never shells out through a string (no shell=True, no bash -lc)", "shell=True" not in src and "bash -lc" not in src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--modules", action="store_true")
    ap.add_argument("--no-other", action="store_true", help="skip _planners / _step4 gates in --modules")
    a = ap.parse_args()
    both = not (a.e2e or a.modules)
    static()
    if a.e2e or both:
        e2e()
    if a.modules or both:
        t_modules(include_other=not a.no_other)
    print("\n%d passed, %d failed" % (N["pass"], N["fail"]))
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
