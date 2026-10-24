"""Self-test for step 5 / A6-eval -- the evaluation runner (sim/step5_eval.py). Offline, Windows `python` from the repo root.

    python sim/_step5_eval_selftest.py

Uses a FAKE executor (and, for end-to-end plumbing, the real LongTermMemory on temp databases). Never touches
sim/trace.db, sim/memory.db or the default sim/step5/ltm.db; no simulator, no WSL, no Ollama, no network.
Covers: task groups + frozen split manifest, disjointness, giant-component infeasibility, paired plans (seeds, languages,
layout kinds), bootstrap over blocks, learning-curve checkpoints with frozen copies, none vs places, invariants
(practice executor writing memory), pairing audit, K definition, inapplicable handling, persistent sequences,
crash/resume equality, journal mismatch refusal, report writer, import hygiene.
"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import step5_eval as ev                  # noqa: E402
from ltm_wm import WorkingMemory         # noqa: E402

REPO = Path(HERE).parent
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except BaseException as e:
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


ITEMS, DSHA = ev.load_items(REPO / "sim" / "dataset" / "general.json")
FRIDGE = (1.0, 0.5, 2.0)


def apple_episode(episode, world="reset"):
    base = dict(schema="s5.ev/1", kitchen="FloorPlan1", epoch=0, frame="f0", map_rev="m0", episode=episode)
    evs, n = [], [0]

    def add(kind, **kw):
        n[0] += 1
        evs.append(dict(base, ev=n[0], kind=kind, t=n[0], **kw))

    add("episode_start", mode="practice", world=world)
    add("receptacle_seen", rtype="Fridge", xyz=list(FRIDGE), view="v1", obs="ob:2")
    add("sighting", otype="Apple", track="tr:1", conf=0.9, xyz=list(FRIDGE), xyz_ok=True, view="v1", obs="ob:3", in_rtype="Fridge",
        in_xyz=list(FRIDGE), in_basis="inside_inspection", detector="rev1")
    add("terminal", exec_ok=True, agent_verdict="success", prims=5, opens=1, walk_m=3.0, invalid=False)
    add("admission_verdict", goal_ok=True, tag="privileged-terminal-goal-check")
    add("episode_end")
    w = WorkingMemory(episode)
    for e in evs:
        w.ingest(e)
    return w


class FakeExec(ev.Executor):
    """Deterministic. Outcome noise depends on (world seed, item, lang) only, NOT on the arm (so pairing is exact); the
    memory effect is that a test with a memory that knows where an Apple is finds it in 2 steps instead of 6."""
    kind = "fake"

    def __init__(self, inapplicable_pos=(), bad_initial=False, write_in_practice=False, write_in_test=False):
        self.calls = []
        self.inapplicable_pos = set(inapplicable_pos)
        self.bad_initial = bad_initial
        self.write_in_practice = write_in_practice
        self.write_in_test = write_in_test
        self.tokens = []

    def execute(self, req):
        self.calls.append(req.cell)
        h = ev.derive(req.world_seed, req.item["id"], req.lang)
        init = ev.sha256_hex(f"{req.world_seed}|{req.item['id']}" + (f"|{req.arm}" if self.bad_initial else ""))
        r = ev.EpisodeResult(initial_state_hash=init)
        if req.mode == "practice" and req.position in self.inapplicable_pos and req.world == "reset":
            r.inapplicable = "goal true at reset"
            return r
        steps = 6.0
        r.gate_pass = False
        if req.mode == "test" and req.memory is not None and "places" in req.features:
            if self.write_in_test:
                req.memory.commit("evil", "x", [])
            ans = req.memory.where_is("Apple")
            if any(c["tier"] == "T0" for c in ans["candidates"]):      # learned evidence, not the fixed prior (T4)
                steps = 2.0
                r.gate_pass = True
        if req.mode == "practice" and req.memory is not None and self.write_in_practice:
            req.memory.reset_world(f"evil{len(self.calls)}")
        r.sr = 1.0 if h % 10 < 8 else 0.0
        r.goal_ok = bool(r.sr)
        r.pc = (0.5 + (h % 5) / 10.0) if req.level == "L2" else None
        r.tds = (0.4 + (h % 6) / 10.0) if req.level == "L3" else None
        r.found, r.steps_to_find, r.walking_m, r.opens, r.stale_visits, r.retries, r.steps = True, steps, steps * 1.5, 1.0, 0.0, 0.0, steps + 3
        if req.mode == "practice":
            r.wm = apple_episode(req.episode_id, req.world)
            r.admission = True
        if req.world == "persistent":
            self.tokens.append(req.world_token)
            r.world_token_out = {"n": (req.world_token or {"n": 0})["n"] + 1}
        return r


def small_cfg(**over):
    kw = dict(run_id="t", arms=("none", "places"), Ks=(0, 1), blocks=3, protocols=("P-familiar",), world_kinds=("reset_fresh",),
              tests_per_block=2, persistent=False, bootstrap_resamples=300)
    kw.update(over)
    return ev.EvalConfig(**kw)


MAN = ev.build_split(ITEMS, DSHA, 11)


def mk_runner(tmp, cfg, ex=None, manifest=None, name="w"):
    return ev.EvalRunner(cfg, ITEMS, manifest or MAN, ex or FakeExec(), Path(tmp) / name, params_sha="p" * 8)


# =================================================================================================== 1 groups + manifest
def test_groups(tmp):
    acc = json.loads((Path(HERE).parent / "docs" / "final-test" / "inclusion_manifest_a26.json").read_text(encoding="utf-8")).get("acceptance_reserved")
    n_acc = len((acc or {}).get("items") or [])                     # ACC-ADD (2026-10-14): + the 10 approved new L3 acceptance items
    check("dataset loaded (%d items: 277 minus the 6 A26-archived, plus %d ACC-ADD acceptance L3)" % (271 + n_acc, n_acc),
          len(ITEMS) == 271 + n_acc and n_acc in (0, 10) and len(DSHA) == 64)
    g = ev.task_groups(ITEMS)
    check("groups partition items", sorted(i for x in g for i in x) == sorted(it["id"] for it in ITEMS))
    m2 = ev.build_split(ITEMS, DSHA, 11)
    check("manifest deterministic", m2["sha256"] == MAN["sha256"])
    check("manifest differs for another seed", ev.build_split(ITEMS, DSHA, 12)["sha256"] != MAN["sha256"])
    pr = {p["item"] for p in MAN["practice"]}
    fam, comp = set(MAN["test_familiar"]), set(MAN["test_composition"])
    check("practice / familiar / composition item sets disjoint", not (pr & fam) and not (pr & comp) and not (fam & comp))
    check("all items accounted for", pr | fam | comp == {it["id"] for it in ITEMS})
    gid = {i: x["id"] for x in MAN["groups"] for i in x["items"]}
    check("every familiar test item shares a group with a practised item", all(any(gid[p] == gid[f] for p in pr) for f in fam))
    cg = {x["id"] for x in MAN["groups"] if x["heldout_composition"]}
    check("composition groups never practised", not any(gid[p] in cg for p in pr) and all(gid[c] in cg for c in comp))
    check("composition feasible on the real dataset, largest component small", MAN["composition"]["feasible"] and MAN["stats"]["largest_fraction"] < 0.1, MAN["stats"])
    check("translation set = other language of practice items", all(t["lang"] != p["lang"] and t["item"] == p["item"] for t, p in zip(MAN["test_translation"], MAN["practice"])))
    a = {"id": "a", "level": "L1", "en": "", "fa": "", "setup": [], "plan": [{"action": "put_in", "object": "Apple", "to": "Fridge", "arg": None}],
         "goal": [{"type": "Apple", "key": "parentReceptacles", "contains": "Fridge"}]}
    b = copy.deepcopy(a)
    b["id"] = "b"
    b["plan"][0]["to"] = "CounterTop"
    b["goal"][0]["contains"] = "CounterTop"
    check("plan key keeps `to`", ev.plan_concept_key(a) != ev.plan_concept_key(b))
    c = copy.deepcopy(a)
    c["id"] = "c"
    c["setup"] = [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]
    check("setup ignored in goal signature; identical goal+plan -> same group", ev.goal_signature(a) == ev.goal_signature(c)
          and len(ev.task_groups([a, b, c])) == 2)
    d = copy.deepcopy(a)
    d["id"] = "d"
    d["goal"] = [{"type": "Apple", "key": "isSliced", "value": True, "any": True}]
    d2 = copy.deepcopy(d)
    d2["id"] = "d2"
    d2["goal"][0].pop("any")
    check("operators / typed values kept in goal signature", ev.goal_signature(d) != ev.goal_signature(d2))
    giant = []
    for k in range(30):
        x = copy.deepcopy(a)
        x["id"] = f"g{k}"
        x["goal"] = [{"type": "Apple", "key": "isSliced", "value": bool(k % 2), "n": k}]
        giant.append(x)
    gm = ev.build_split(giant, "d" * 64, 1)
    check("giant component -> composition infeasible with a reason, no silent repair",
          gm["composition"]["feasible"] is False and "giant" in gm["composition"]["reason"] and gm["test_composition"] == [])
    p = Path(tmp) / "man" / "split.json"
    ev.write_manifest(MAN, p)
    ev.write_manifest(MAN, p)
    check("manifest written once, idempotent", ev.load_manifest(p)["sha256"] == MAN["sha256"])
    check("a different manifest cannot overwrite the frozen one", raises(ev.EvalError, ev.write_manifest, ev.build_split(ITEMS, DSHA, 99), p))
    t = json.loads(p.read_text(encoding="utf-8"))
    t["practice"][0]["lang"] = "xx"
    p2 = Path(tmp) / "man" / "tampered.json"
    p2.write_text(json.dumps(t), encoding="utf-8")
    check("tampered manifest rejected", raises(ev.EvalError, ev.load_manifest, p2))
    leak = copy.deepcopy(MAN)
    leak["test_familiar"].append(MAN["practice"][0]["item"])
    leak["sha256"] = ev.sha256_hex(ev.canon({k: v for k, v in leak.items() if k != "sha256"}))
    check("practice/test overlap rejected even with a valid hash", raises(ev.EvalError, ev.validate_manifest, leak))


# =================================================================================================== 2 paired plans
def test_plans(tmp):
    cfg = small_cfg(Ks=(0, 1, 3, 5, 10), protocols=("P-familiar", "P-composition", "P-translation"), tests_per_block=4)
    pf = ev.make_block_plan(MAN, ITEMS, cfg, 0, "reset_fresh")
    check("plan is a pure function (identical on rebuild)", ev.canon(pf) == ev.canon(ev.make_block_plan(MAN, ITEMS, cfg, 0, "reset_fresh")))
    check("another block differs", ev.canon(pf["practice"]) != ev.canon(ev.make_block_plan(MAN, ITEMS, cfg, 1, "reset_fresh")["practice"]))
    check("practice has max(K)=10 attempts", len(pf["practice"]) == 10)
    ps = {p["world_seed"] for p in pf["practice"]}
    ts = [t for v in pf["tests"].values() for t in v]
    check("fresh worlds: test seeds disjoint from practice seeds", not (ps & {t["world_seed"] for t in ts}))
    fam = pf["tests"]["P-familiar"]
    byitem = {}
    for t in fam:
        byitem.setdefault(t["item"], []).append(t)
    check("familiar tests exist; EN and FA of a test item both tested, in the same world",
          byitem and all({x["lang"] for x in v} == {"en", "fa"} and len({x["world_seed"] for x in v}) == 1 for v in byitem.values()))
    pr_items = {p["item"] for p in pf["practice"]}
    check("P-familiar items are held out of practice (disjoint ids)", not (set(byitem) & pr_items))
    check("P-composition items come from composition groups only", pf["tests"]["P-composition"] and all(t["item"] in set(MAN["test_composition"]) for t in pf["tests"]["P-composition"]))
    check("P-translation items were practised in the other language", pf["tests"]["P-translation"] and all(t["item"] in pr_items for t in pf["tests"]["P-translation"]))
    pa = fam[0]["practised_at_K"]
    check("practised_at_K is monotone in K", all((not pa[str(a)]) or pa[str(b)] for a, b in zip((0, 1, 3, 5), (1, 3, 5, 10))))
    check("K=0 practises nothing", all(not t["practised_at_K"]["0"] for t in fam))
    check("some familiar test concept is practised by K=10", any(t["practised_at_K"]["10"] for t in fam))
    sh = ev.make_block_plan(MAN, ITEMS, cfg, 0, "reset_shared")
    check("shared layouts: seeds come from a small per-block pool, test layouts recur from practice",
          len({p["world_seed"] for p in sh["practice"]}) <= cfg.layout_pool and {t["world_seed"] for v in sh["tests"].values() for t in v} <= set(sh["layouts"]))
    check("fresh and shared plans share the practice order but not the worlds", [p["item"] for p in sh["practice"]] != [p["item"] for p in pf["practice"]]
          or [p["world_seed"] for p in sh["practice"]] != [p["world_seed"] for p in pf["practice"]])
    check("policy seeds are a separate namespace", all(p["policy_seed"] not in ps for p in pf["practice"]))
    check("derive is stable and namespaced", ev.derive(1, "a", 2) == ev.derive(1, "a", 2) and ev.derive(1, "a", 2) != ev.derive(1, "b", 2))
    seqs = ev.make_sequences(MAN, ITEMS, small_cfg(n_sequences=3, seq_len=5))
    check("sequences: distinct initial seeds, unique items per sequence, stable", len({s["initial_seed"] for s in seqs}) == 3
          and all(len({t["item"] for t in s["tasks"]}) == 5 for s in seqs) and seqs == ev.make_sequences(MAN, ITEMS, small_cfg(n_sequences=3, seq_len=5)))
    cc = ev.core_config({"eval": {"blocks": {"value": 5}, "K": {"value": [0, 1, 3, 5, 10]}, "protocols": {"value": ["P-familiar", "P-composition"]}}})
    check("core preset: none vs places, K 0/1, reference plans, 5 blocks, no persistent", cc.arms == ("none", "places") and cc.Ks == (0, 1)
          and cc.blocks == 5 and cc.plan_source == "reference" and not cc.persistent)
    pp = ev.load_params() if hasattr(ev, "load_params") else None
    check("config from the real registry reads blocks and K", ev.config_from_params(json.loads((REPO / "sim" / "step5" / "params.json").read_text(encoding="utf-8"))).Ks == (0, 1, 3, 5, 10))


# =================================================================================================== 3 statistics
def test_stats(tmp):
    check("bootstrap needs >= 2 units", ev.bootstrap_ci([1.0]) is None)
    c1 = ev.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], 500, 3)
    check("bootstrap deterministic + brackets the mean", c1 == ev.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], 500, 3) and c1[0] <= 3.0 <= c1[1])
    check("constant differences -> degenerate CI", ev.bootstrap_ci([0.5] * 5, 200, 1) == [0.5, 0.5])
    recs_a = [{"block": b, "level": "L1", "lang": l, "sr": 1.0, "steps_to_find": 2.0, "goal_ok": True} for b in range(3) for l in ("en", "fa") for _ in range(4)]
    recs_r = [{"block": b, "level": "L1", "lang": l, "sr": 0.0 if i % 2 else 1.0, "steps_to_find": 6.0, "goal_ok": bool(i % 2 == 0)}
              for b in range(3) for l in ("en", "fa") for i in range(4)]
    c = ev.paired_comparison(recs_a, recs_r, "steps_to_find", "all", "all", 200, 1, 8)
    check("unit of analysis is the block (n_units=3 though 24 episodes)", c["n_units"] == 3 and c["mean_diff"] == -4.0 and c["low_n"] and c["n_negative"] == 3)
    c2 = ev.paired_comparison(recs_a, recs_r, "sr", "all", "all", 200, 1, 8)
    check("mean sr diff 0.5, per-block diffs listed", abs(c2["mean_diff"] - 0.5) < 1e-9 and len(c2["diffs"]) == 3)
    c3 = ev.paired_comparison(recs_a, recs_r, "steps_to_find", "L2", "all", 200, 1, 8)
    check("empty selector -> no units", c3["n_units"] == 0 and c3["ci95"] is None)
    skip = [dict(r, inapplicable="x") for r in recs_a]
    check("inapplicable records excluded from means", ev.block_values(skip, "sr") == {})
    check("successful-only filter", ev.block_values(recs_r, "sr", successful_only=True) == {0: 1.0, 1: 1.0, 2: 1.0})
    many = [{"block": b, "level": "L1", "lang": "en", "sr": 1.0, "goal_ok": True} for b in range(10)]
    check("low_n flag clears with >= 8 blocks", ev.paired_comparison(many, many, "sr", "all", "all", 100, 1, 8)["low_n"] is False)


# =================================================================================================== 4 end-to-end learning curve
def test_learning_curve(tmp):
    cfg = small_cfg(Ks=(0, 1, 3), blocks=3, protocols=("P-familiar", "P-composition"), world_kinds=("reset_fresh", "reset_shared"),
                    tests_per_block=2, none_at_every_K=False)
    ex = FakeExec()
    r = mk_runner(tmp, cfg, ex, name="lc")
    r.run_all()
    tests, prac, ck = r.journal.records("test"), r.journal.records("practice"), r.journal.records("ckpt")
    nt = lambda arm, wk, k: len([t for t in tests if t["arm"] == arm and t["world_kind"] == wk and t["K"] == k])
    check("none runs tests once (K=0 baseline) and no practice", all(t["K"] == 0 for t in tests if t["arm"] == "none") and not [p for p in prac if p["arm"] == "none"])
    check("places runs tests at every K", {t["K"] for t in tests if t["arm"] == "places"} == {0, 1, 3})
    check("same test episodes at every K and for every arm", nt("places", "reset_fresh", 0) == nt("places", "reset_fresh", 3) == nt("none", "reset_fresh", 0) > 0)
    check("practice attempts: 3 per (world kind, block) for places", len([p for p in prac if p["arm"] == "places" and p["world_kind"] == "reset_fresh"]) == 3 * 3)
    check("one checkpoint per (world, block, K)", len(ck) == 2 * 3 * 3)
    check("every frozen test: hash unchanged, memory read-only", all(t["frozen_hash_before"] == t["frozen_hash_after"] and t["memory_read_only"] for t in tests if t["arm"] == "places")
          and all(not t["memory_read_only"] for t in tests if t["arm"] == "none"))
    ok = True
    for c in ck:
        d = Path(tmp) / "lc" / "ckpt" / c["world_kind"] / f"b{c['block']}" / c["arm"] / f"K{c['K']}"
        ok &= d.exists() and not r.factory.verify(d)
    check("every checkpoint snapshot verifies", ok)
    r0 = [c["memory_rows"] for c in ck if c["K"] == 0 and c["block"] == 0 and c["world_kind"] == "reset_fresh"][0]
    r1 = [c["memory_rows"] for c in ck if c["K"] == 1 and c["block"] == 0 and c["world_kind"] == "reset_fresh"][0]
    check("K=0 snapshot is the empty store; memory grows with K", r0 < r1)
    check("consolidation committed for practice attempts, latency recorded", all(p["consolidation_status"] == "committed" and p["consolidation_latency_s"] >= 0 for p in prac),
          [p["consolidation_status"] for p in prac][:5])
    rep = ev.write_report(Path(tmp) / "lc" / "journal.jsonl", Path(tmp) / "lc" / "report")
    check("report files written", (Path(tmp) / "lc" / "report" / "report.md").exists() and (Path(tmp) / "lc" / "report" / "report.json").exists())
    check("pairing audit OK on deterministic worlds", rep["pairing"]["ok"] and rep["pairing"]["checked"] > 0)
    cm = {(c["world_kind"], c["protocol"], c["arm"], c["K"], c["metric"]): c for c in rep["comparisons"] if c["level"] == "all" and c["lang"] == "all" and not c["successful_only"]}
    k0, k1 = cm[("reset_fresh", "P-familiar", "places", 0, "steps_to_find")], cm[("reset_fresh", "P-familiar", "places", 1, "steps_to_find")]
    check("places == none at K=0 (no memory yet)", k0["mean_diff"] == 0.0, k0)
    check("places beats none at K=1 steps-to-find (-4) over 3 blocks, all blocks agree", k1["mean_diff"] == -4.0 and k1["n_units"] == 3 and k1["n_negative"] == 3 and k1["low_n"], k1)
    check("SR difference 0 (outcome noise is arm-independent => exact pairing)", cm[("reset_fresh", "P-familiar", "places", 1, "sr")]["mean_diff"] == 0.0)
    gains = [g for g in rep["k_gains"] if g["metric"] == "sr"]
    check("K-gain table exists and uses block units", gains and all(g["n_units"] <= 3 for g in gains))
    md = (Path(tmp) / "lc" / "report" / "report.md").read_text(encoding="utf-8")
    check("caveat names the fake executor", any("EXECUTOR KIND = 'fake'" in c for c in rep["caveats"]) and "NOT evidence" in md)
    check("shared and fresh worlds reported separately", {m["world_kind"] for m in rep["means"]} == {"reset_fresh", "reset_shared"})
    comp_tests = [t for t in tests if t["protocol"] == "P-composition"]
    check("composition tests ran and none of those items was ever practised", comp_tests and not ({t["item"] for t in comp_tests} & {p["item"] for p in prac}))
    check("practised coverage reported for P-familiar", any(m.get("practised_coverage") is not None for m in rep["means"] if m["protocol"] == "P-familiar"))
    ckrep = [c for c in rep["checkpoints"] if c["arm"] == "places"]
    check("memory size + latency + FLR placeholder in checkpoint table", ckrep and all(c["memory_rows"] is not None and c["flr"] == "not measured" for c in ckrep))
    check("markdown has caveats, paired differences and the learning-gain table", "## Caveats" in md and "Paired differences vs `none`" in md and "Learning gains" in md)
    cfg2 = small_cfg(Ks=(0, 1), blocks=2, none_at_every_K=True)
    r2 = mk_runner(tmp, cfg2, FakeExec(), name="lc2")
    r2.run_all()
    check("none_at_every_K runs the baseline at each K", {t["K"] for t in r2.journal.records("test") if t["arm"] == "none"} == {0, 1})
    pa_ = [a for a in rep["comparisons"] if a["metric"] == "steps_to_find" and a["successful_only"]]
    check("successful-only cost rows are marked descriptive", pa_ and all(a.get("descriptive") for a in pa_))
    check("missing-value denominators reported (n_<metric>)", all("n_steps_to_find" in m for m in rep["means"]))


# =================================================================================================== 5 invariants + K + pairing
def test_invariants(tmp):
    cfg = small_cfg(blocks=1)
    check("practice executor that writes memory -> InvariantError", raises(ev.InvariantError, mk_runner(tmp, cfg, FakeExec(write_in_practice=True), name="inv1").run_all))
    check("test executor cannot write the frozen copy", raises(Exception, mk_runner(tmp, cfg, FakeExec(write_in_test=True), name="inv2").run_all))
    r = mk_runner(tmp, small_cfg(blocks=2), FakeExec(bad_initial=True), name="inv3")
    r.run_all()
    rep = ev.build_report(Path(tmp) / "inv3" / "journal.jsonl")
    check("pairing audit catches differing initial states across arms", not rep["pairing"]["ok"] and rep["pairing"]["violations"])
    ex = FakeExec(inapplicable_pos={1})
    r = mk_runner(tmp, small_cfg(blocks=2, Ks=(0, 1, 3)), ex, name="inv4")
    r.run_all()
    pr = [p for p in r.journal.records("practice") if p["pos"] == 1]
    cks = r.journal.records("ckpt")
    ck1 = [c for c in cks if c["K"] == 1 and c["arm"] == "places"]
    check("inapplicable attempt still counts toward K and is flagged", pr and all(p["inapplicable"] and p["consolidation_status"] == "inapplicable" for p in pr)
          and len(ck1) == 2 and all(len(c["exposure"]["items"]) == 1 for c in ck1))
    check("inapplicable practice leaves memory unchanged at K=1",
          all(c["memory_rows"] == [k0["memory_rows"] for k0 in cks if k0["K"] == 0 and k0["block"] == c["block"] and k0["arm"] == "places"][0] for c in ck1))
    rep = ev.build_report(Path(tmp) / "inv4" / "journal.jsonl")
    check("report counts inapplicable separately", rep["inapplicable"]["practice"] >= 1)
    check("changed config refuses an existing journal", raises(ev.JournalMismatch, mk_runner, tmp, small_cfg(blocks=2, Ks=(0, 1, 5)), None, None, "inv4"))
    check("changed manifest refuses an existing journal", raises(ev.JournalMismatch, mk_runner, tmp, small_cfg(blocks=2, Ks=(0, 1, 3)), None, ev.build_split(ITEMS, DSHA, 12), "inv4"))
    jp = Path(tmp) / "inv4" / "journal.jsonl"
    with open(jp, "ab") as f:
        f.write(b'{"cell":"torn","kind"')
    j = ev.Journal(jp)
    check("torn journal tail ignored and next append starts a fresh line", j.torn and j.append("ok-after-torn", "x", {"a": 1}) and ev.Journal(jp).has("ok-after-torn"))


# =================================================================================================== 6 persistent worlds
def test_persistent(tmp):
    cfg = small_cfg(persistent=True, n_sequences=3, seq_len=4, arms=("none", "places"), Ks=(0, 1), blocks=1)
    ex = FakeExec()
    r = mk_runner(tmp, cfg, ex, name="pers")
    r.run_persistent()
    seqs = r.journal.records("seq")
    check("3 sequences x 2 arms x 4 positions", len(seqs) == 24)
    good = True
    for a in ("none", "places"):
        for q in range(3):
            got = sorted((s["pos"], s["world_token_out"]["n"]) for s in seqs if s["arm"] == a and s["seq"] == q)
            good &= got == [(p, p) for p in range(1, 5)]
    check("world token threaded from each episode to the next", good)
    check("first episode of each sequence gets no token", ex.tokens.count(None) == 6)
    sqs = ev.make_sequences(MAN, ITEMS, cfg)
    ord_none = [(s["item"], s["lang"]) for s in seqs if s["arm"] == "none" and s["seq"] == 0]
    ord_pl = [(s["item"], s["lang"]) for s in seqs if s["arm"] == "places" and s["seq"] == 0]
    check("same task order and initial seed in both arms", ord_none == ord_pl == [(t["item"], t["lang"]) for t in sqs[0]["tasks"]]
          and len({s["world_seed"] for s in seqs if s["seq"] == 0}) == 1)
    check("persistent practice never used a frozen copy (online adaptation)", all(s["arm"] == "none" or s["consolidation_status"] in ("committed", "duplicate") for s in seqs))
    rep = ev.write_report(Path(tmp) / "pers" / "journal.jsonl", Path(tmp) / "pers" / "report")
    P = rep["persistent"]
    cmp_ = [c for c in P["comparisons"] if c["arm"] == "places" and c["metric"] == "steps_to_find"]
    check("persistent comparison exists", bool(cmp_))
    check("unit = sequence; steps-to-find unchanged because the fake only reads memory in test mode", cmp_ and cmp_[0]["n_units"] == 3 and cmp_[0]["low_n"])
    bp = {(b["arm"], b["pos"]): b["n"] for b in P["by_position"]}
    check("per-position table", bp[("places", 1)] == 3 and bp[("none", 4)] == 3)
    for stop in (3, 7, 13):
        w = f"pers-r{stop}"
        r1 = mk_runner(tmp, cfg, FakeExec(), name=w)
        r1.stop_after = stop
        check(f"stop after {stop} calls", raises(KeyboardInterrupt, r1.run_persistent))
        r2 = mk_runner(tmp, cfg, FakeExec(), name=w)
        r2.run_persistent()
        a = ev.build_report(Path(tmp) / w / "journal.jsonl")["persistent"]
        check(f"resume@{stop}: persistent result equals the uninterrupted run", ev.canon(a["comparisons"]) == ev.canon(P["comparisons"]) and ev.canon(a["by_position"]) == ev.canon(P["by_position"]))
        check(f"resume@{stop}: no duplicated journal cells", len(r2.journal.records("seq")) == 24)

    class InapplicableSeq(FakeExec):
        def execute(self, req):
            if req.position == 2:
                return ev.EpisodeResult(inapplicable="goal true", initial_state_hash="h", world_token_out=req.world_token)
            return super().execute(req)

    r3 = mk_runner(tmp, cfg, InapplicableSeq(), name="pers3")
    r3.run_persistent()
    rep3 = ev.build_report(Path(tmp) / "pers3" / "journal.jsonl")
    check("persistent inapplicable counted, excluded", rep3["inapplicable"]["sequence"] == 6 and rep3["persistent"]["by_position"][1]["n"] == 0)
    # a memory-reading persistent executor: later positions can use what earlier ones consolidated
    class Reader(FakeExec):
        def execute(self, req):
            res = super().execute(req)
            if req.memory is not None and "places" in req.features and req.memory.where_is("Apple")["candidates"][0]["tier"] == "T0":
                res.steps_to_find = 2.0
            return res

    r4 = mk_runner(tmp, cfg, Reader(), name="pers4")
    r4.run_persistent()
    c4 = [c for c in ev.build_report(Path(tmp) / "pers4" / "journal.jsonl")["persistent"]["comparisons"] if c["arm"] == "places" and c["metric"] == "steps_to_find"][0]
    check("persistent memory carries over between episodes (places faster than none)", c4["mean_diff"] < 0 and c4["n_units"] == 3, c4)


# =================================================================================================== 7 crash / resume (learning curve)
def test_resume(tmp):
    cfg = small_cfg(Ks=(0, 1, 3), blocks=2, protocols=("P-familiar",), world_kinds=("reset_fresh", "reset_shared"))
    full = mk_runner(tmp, cfg, FakeExec(), name="full")
    full.run_all()
    ref = ev.build_report(Path(tmp) / "full" / "journal.jsonl")
    n_calls = full.calls

    def stable(rep):
        return ev.canon([rep["means"], rep["comparisons"], rep["k_gains"], rep["training"],
                         [{k: v for k, v in c.items() if "latency" not in k} for c in rep["checkpoints"]]])

    for stop in (2, 5, 9, 14, 19, 26):
        w = f"res{stop}"
        r1 = mk_runner(tmp, cfg, FakeExec(), name=w)
        r1.stop_after = stop
        interrupted = raises(KeyboardInterrupt, r1.run_all)
        r2 = mk_runner(tmp, cfg, FakeExec(), name=w)
        r2.run_all()
        rep = ev.build_report(Path(tmp) / w / "journal.jsonl")
        check(f"resume after {stop} calls reproduces the uninterrupted report", interrupted and stable(rep) == stable(ref))
        check(f"resume@{stop}: journal cells = uninterrupted run", len(ev.Journal(Path(tmp) / w / "journal.jsonl").order) == len(ev.Journal(Path(tmp) / "full" / "journal.jsonl").order))
    ex = FakeExec()
    mk_runner(tmp, cfg, FakeExec(), name="done").run_all()
    mk_runner(tmp, cfg, ex, name="done").run_all()
    check("re-running a finished run executes nothing", ex.calls == [] and n_calls > 0)
    w = "res-corrupt"
    r1 = mk_runner(tmp, cfg, FakeExec(), name=w)
    r1.stop_after = 12
    raises(KeyboardInterrupt, r1.run_all)
    for d in sorted((Path(tmp) / w / "ckpt").rglob("ltm.db")):
        d.write_bytes(d.read_bytes() + b"x")
        break
    r2 = mk_runner(tmp, cfg, FakeExec(), name=w)
    r2.run_all()
    check("a tampered checkpoint is detected and rebuilt, report unchanged", stable(ev.build_report(Path(tmp) / w / "journal.jsonl")) == stable(ref))


# =================================================================================================== 8 hygiene
def test_hygiene(tmp):
    src = (Path(HERE) / "step5_eval.py").read_text(encoding="utf-8")
    check("no simulator / network imports", not any(x in src for x in ("import ai2thor", "import socket", "import urllib", "import requests", "subprocess")))
    check("never names the shared databases", "trace.db" not in src and "memory.db" not in src)
    check("importing the runner loaded no simulator package", "ai2thor" not in sys.modules)
    check("default ltm.db never created", not (REPO / "sim" / "step5" / "ltm.db").exists())


def main():
    global _fail
    with tempfile.TemporaryDirectory(prefix="s5eval_") as tmp:
        for fn in (test_groups, test_plans, test_stats, test_learning_curve, test_invariants, test_persistent, test_resume, test_hygiene):
            try:
                fn(tmp)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                _fail += 1
                print(f"  FAIL  {fn.__name__} crashed: {e}")
    print(f"step5 eval selftest: {_pass}/{_pass + _fail}")
    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    main()
