"""C4 gate: the ft-v2 treatment switch of the locked runner (sim/final_test/matrix/treatment.py and its wiring in common, capture, plan,
execute, score, practice_k, step5_live). Offline: Windows python, repo root; no simulator, no LLM, no network, no Anthropic call.

    python sim/_ftm_treatment_selftest.py

  0  static: python 3.10 syntax, no simulator import at module level, the pinned switch, TEST_V1_TREATMENT = ft-v1
  1  LEGACY BYTE IDENTITY: the pre-change modules (sim/final_test/_checkpoints/C4-integration/pre/) and the new ones give identical
     namespaces, unit keys, request bytes, worst cases, G-iso verdicts, execution units and derived score keys for rehearsal-v1 /
     rehearsal-v2 / practice-v1 / test-v1; the practice-v1 hook is the same source
  2  the switch: treatment per namespace, D2 types, the A24 summary rule, completeness of the full list, effective params (r4 + D2
     forced, re-hashed), package-only STRICT masks (by name, refused outside rehearsal-v3), hook-B arm table, unit-key policy field,
     leave-out (logged; a test presentation FAILS), journal / entry identity checks
  3  the REAL package (ltm_package on the practice-v1 S* store and its order-0 rows, draft briefing in rehearsal-v3): identical bytes
     on re-render, within 2,000 tokens with the header and the profile cap, no Knife, no Persian, profile switch r4 / r6 / r6b
  4  capture ft-v2 (fake executor): prompt summary, observed types, 48 types, package block, G-iso with PACKAGE_HEADER; a profile switch
     changes only the package bytes (none / FS requests identical); incomplete summary refused; ft-v1 capture untouched
  5  plan ft-v2: request identity re-renders the captured shas, observed types reach the runner, worst case = 3 calls
  6  execute / score ft-v2: prior chooser, hook units {G+M, O+M, X+M, P+M, curve} never share with {G0, O0, FS, X0, P0, /p};
     score.derive_unit = execute's key; hook B installed per episode; LiveExecutor's pass-through is default-off and consumed once
  7  practice ft-v2 (fake executor, real practice LTM): package from the pre-attempt store + earlier rows, hook B each attempt, rows
     carry the identity, every snapshot holds its own row prefix (content-bound), resume refused across an identity change, order /
     channel / hook refusals; practice-v2 refuses the unapproved briefing
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import types as _types

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                     # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATRIX = os.path.join(HERE, "final_test", "matrix")
PRE = os.path.join(HERE, "final_test", "_checkpoints", "C4-integration", "pre")
sys.path.insert(0, HERE)
sys.path.insert(0, MATRIX)

import common as C                                                    # noqa: E402
import treatment as T                                                 # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)[:400]))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def load_pre(name: str, alias: str, common_mod):
    """Load a pre-change module from the checkpoint with `common` bound to the given (pre-change) common module."""
    saved, path0 = sys.modules.get("common"), list(sys.path)
    sys.modules["common"] = common_mod
    try:
        spec = importlib.util.spec_from_file_location(alias, os.path.join(PRE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.modules["common"] = saved
        sys.path[:] = path0                     # a pre-change module puts its own (checkpoint) folder first: undo it


MAN = C.load_manifest()
GEN = {i["id"]: i for i in json.load(open(os.path.join(ROOT, "sim", "dataset", "general.json"), encoding="utf-8"))["items"]}
DEV = sorted(MAN["dev_pilot"])
TYPES49 = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))["types"]
SIGNAL = json.load(open(os.path.join(ROOT, "docs", "final-test", "kitchen_memory", "signal_set.json"), encoding="utf-8"))["items"]
QUIET = lambda *a, **k: None                                          # noqa: E731


class FakeCounter:
    tokenizers = ["gemma", "anthropic"]

    def counts(self, text):
        return {"gemma": len(text) // 4 + 1, "anthropic": len(text) // 3 + 1}

    def __call__(self, text):
        return max(self.counts(text).values()) if text else 0

    def save(self):
        pass


class FakeOM:
    def __init__(self, lines):
        self.lines = list(lines)

    def summary(self, limit=40):
        return self.lines[:limit]


def full_lines(n_extra=0):
    base = ["Apple [?] on/in CounterTop at (1.00, 2.00), step 3", "Cabinet [closed] on/in Floor at (0.10, 0.20), step 1",
            "Cabinet [open] on/in Floor at (0.30, 0.20), step 2", "Floor [?] on/in Floor at (0.00, 0.00), step 1",
            "Fridge [closed] on/in Floor at (2.00, 1.00), step 4", "Knife [?] on/in CounterTop at (1.10, 2.00), step 5",
            "Microwave [closed] on/in CounterTop at (1.50, 2.50), step 6", "Mug [?] on/in CounterTop at (1.20, 2.10), step 7",
            "SinkBasin [?] on/in Sink at (0.50, 2.50), step 8", "StoveKnob [off] on/in StoveBurner at (1.80, 2.80), step 9"]
    extra = ["Type%s [?] on/in CounterTop at (0.%02d, 0.10), step %d" % (chr(65 + i // 26) + chr(65 + i % 26), i % 100, 10 + i)
             for i in range(n_extra)]
    return sorted(base + extra, key=lambda l: l.split(" ", 1)[0])


# ================================================================================================ 0 static
print("== 0 static")
EDITED = [os.path.join(MATRIX, f) for f in ("treatment.py", "common.py", "capture.py", "plan.py", "execute.py", "score.py",
                                           "practice_k.py", "freeze.py")] + [os.path.join(HERE, "step5_live.py")]
ok = True
for p in EDITED:
    try:
        ast.parse(open(p, encoding="utf-8").read(), feature_version=(3, 10))
    except SyntaxError as e:
        ok = False
        print("   syntax", p, e)
check("0.1 every edited module parses as python 3.10", ok)
tsrc = open(os.path.join(MATRIX, "treatment.py"), encoding="utf-8").read()
top = [n for n in ast.parse(tsrc).body if isinstance(n, (ast.Import, ast.ImportFrom))]
mods = {a.name for n in top for a in getattr(n, "names", [])} | {getattr(n, "module", None) for n in top}
check("0.2 treatment.py imports no simulator / LLM module at module level", not (mods & {"ai2thor", "step5_live", "honest_skills", "honest", "objmem",
                                                                                        "planners.anthropic_provider", "planners.ollama_provider"}), mods)
# TREAT-FREEZE 2026-10-13: test-v1 flipped (A24.1-T item 4 (B)). Replaces "0.3 test-v1 stays on ft-v1 until the A24.1 flip (Astra C4 Q1 (b))";
# the pre-flip behaviour stays covered under a temporary ft-v1 toggle in 1.3 / 2.14 / 2.20.
check("0.3 test-v1 is flipped to ft-v2 by the ONE constant (TREAT-FREEZE; A24.1-T item 4 (B)); common.NAMESPACES['test-v1'] has no treatment key",
      T.TEST_V1_TREATMENT == "ft-v2" and T.is_ft2("test-v1") and "treatment" not in C.NAMESPACES["test-v1"])
import ltm_package as LP                                              # noqa: E402
check("0.4 the pinned profile is a known package profile; universal; 2 repairs; D2; 2,000 tokens",
      T.FT2["package_profile"] in LP.PROFILES and T.FT2["mask"] == "universal" and T.FT2["repairs"] == 2 and T.FT2["type_scope"] == "D2"
      and T.FT2["package_max_tokens"] == 2000 and T.FT2["practice_orders"] == [0])
print("   switch sha", T.switch_sha()[:16], "profile", T.FT2["package_profile"])
check("0.5 the switch dict is never handed out mutable (config() is a copy)", T.config() == T.FT2 and T.config() is not T.FT2)

# ================================================================================================ 1 legacy byte identity
print("== 1 legacy byte identity (pre-change modules from the checkpoint)")
spec = importlib.util.spec_from_file_location("common_pre", os.path.join(PRE, "common.py"))
C0 = importlib.util.module_from_spec(spec)
sys.modules["common_pre"] = C0
spec.loader.exec_module(C0)
LEGACY_NS = ("rehearsal-v1", "rehearsal-v2", "practice-v1", "test-v1")
check("1.1 every pre-change namespace entry is unchanged (test-v1: only its ceilings, re-derived by A24.1-C to 30 / 50 and by A37 to 60 / 100); only practice-v2 "
      "and rehearsal-v3 are new (+ acceptance-v2, ACC-RUNNER 2026-10-14)",
      all(C.NAMESPACES[n] == C0.NAMESPACES[n] for n in C0.NAMESPACES if n != "test-v1")
      and dict(C.NAMESPACES["test-v1"], ceilings=None) == dict(C0.NAMESPACES["test-v1"], ceilings=None)
      and C.NAMESPACES["test-v1"]["ceilings"] == {"core": 60.0, "total": 100.0}
      and set(C.NAMESPACES) - set(C0.NAMESPACES) - {"acceptance-v2", "acceptance-v2-a2", "freeze-gates-ft2"}
      - {"practice-k2-v1", "rehearsal-k2-v1", "test-k2-v1", "practice-k3-v1", "rehearsal-k3-v1", "test-k3-v1"} == {"practice-v2", "rehearsal-v3"}
      and set(C0.NAMESPACES) == set(LEGACY_NS))
check("1.2 the new namespaces are ft-v2 and never the test items", all(T.is_ft2(n) and C.NAMESPACES[n]["items"] != "test" for n in ("practice-v2", "rehearsal-v3")))
_sv13 = T.TEST_V1_TREATMENT
T.TEST_V1_TREATMENT = "ft-v1"
try:
    _pre13 = all(T.treatment_of(n) == "ft-v1" for n in LEGACY_NS)
finally:
    T.TEST_V1_TREATMENT = _sv13
check("1.3 every legacy namespace is ft-v1 (pre-flip: all four incl. test-v1; after the TREAT-FREEZE flip: test-v1 follows TEST_V1_TREATMENT = ft-v2)",
      _pre13 and all(T.treatment_of(n) == "ft-v1" for n in LEGACY_NS if n != "test-v1") and T.treatment_of("test-v1") == "ft-v2")
samples = [dict(C.UNIT_KEY_GOLDEN_INPUT),
           dict(plan=[{"action": "pickup", "object": "Mug", "to": None, "arg": None}, {"action": "put_in", "object": "Mug", "to": "Sink", "arg": None}],
                pres="canon:g-l2-x:fa", chooser="prior", snapshot_sha=None, restored_state="canon+" + "a" * 16, summary_sha="b" * 64),
           dict(plan=[], pres="reloc:g-l1-y:en", chooser="o0k3", snapshot_sha="c" * 64, restored_state="reloc:7+" + "d" * 16, summary_sha="e" * 64)]
check("1.4 unit keys: old == new == new(exec_policy=None) on 3 samples; golden unchanged",
      all(C0.execution_unit_key(**s) == C.execution_unit_key(**s) == C.execution_unit_key(**s, exec_policy=None) for s in samples)
      and C.unit_key_selfcheck()["ok"] and C0.UNIT_KEY_SPEC == C.UNIT_KEY_SPEC)
check("1.5 a set policy changes the key (hook-on never equals hook-off)",
      C.execution_unit_key(**samples[1], exec_policy="briefing:" + "f" * 16) != C.execution_unit_key(**samples[1]))

PLN0 = load_pre("plan", "plan_pre", C0)
import plan as PLN                                                    # noqa: E402
assert os.path.dirname(os.path.abspath(PLN.__file__)) == MATRIX, PLN.__file__
import honest_skills as hs                                            # noqa: E402
from planners.contract import MEMORY_HEADER, PACKAGE_HEADER, PlannerContract, prompt_sha256   # noqa: E402
rec_leg = {"instruction": "Put the mug in the sink", "summary": ["Mug [?] on/in CounterTop at (1.00, 2.00), step 3"], "holding": None,
           "blocks": {"none": {"text": ""}, "S*": {"text": "Mug -> Cabinet (last seen in practice)"}, "FS": {"text": "example: pickup Mug"},
                      "o0k3": {"text": ""}}}
cleg_old, cleg_new = PlannerContract(TYPES49, tools=hs.TOOLS), PLN.make_contract(TYPES49, "rehearsal-v1")
check("1.6 ft-v1 contract: make_contract(rehearsal-v1) renders the same system text as the frozen PlannerContract",
      cleg_old.system_text() == cleg_new.system_text() and cleg_new.version == "ft-v1")
check("1.7 request bytes: old render_messages == new for none / S* / FS / an empty curve block",
      all(prompt_sha256(PLN0.render_messages(cleg_old, rec_leg, s)) == prompt_sha256(PLN.render_messages(cleg_new, rec_leg, s))
          for s in ("none", "S*", "FS", "o0k3")))
check("1.8 G-iso none-request renderer: old == new (ft-v1)", PLN0.none_request_renderer(ROOT)(rec_leg) == PLN.none_request_renderer(ROOT)(rec_leg))


class FakeProv:
    def reserve_usd(self, msgs, _):
        return round(sum(len(m["content"]) for m in msgs) / 2 * 1e-6, 9)

    def max_output_tokens(self, _):
        return 100


msgs_leg = PLN.render_messages(cleg_new, rec_leg, "S*")
check("1.9 worst case of an ft-v1 cell (first + ONE repair): old == new", PLN0.cell_worst(FakeProv(), cleg_old, msgs_leg) == PLN.cell_worst(FakeProv(), cleg_new, msgs_leg))

K0 = load_pre("capture", "capture_pre", C0)
import capture as K                                                   # noqa: E402
msgs_by = {s: PLN.render_messages(cleg_new, rec_leg, s) for s in ("none", "S*", "FS")}
blocks = {s: {"slot": ("fewshot" if s == "FS" else (None if s == "none" else "memory")), "text": rec_leg["blocks"][s]["text"]} for s in msgs_by}
bad_msgs = dict(msgs_by, **{"S*": [msgs_by["S*"][0], {"role": "user", "content": msgs_by["S*"][1]["content"] + " extra"}]})
check("1.10 capture G-iso: old == new on a clean and on a broken request set (default header)",
      K0.g_iso_problems(msgs_by, blocks) == K.g_iso_problems(msgs_by, blocks) == [] and K0.g_iso_problems(bad_msgs, blocks) == K.g_iso_problems(bad_msgs, blocks) != [])

X0 = load_pre("execute", "execute_pre", C0)
import execute as X                                                   # noqa: E402
PLAN_STEPS = [{"action": "pickup", "object": "Mug", "to": None, "arg": None}]


def synth(ns, items, arms=("G0", "G+M", "O0", "O+M", "G+FS", "O+FS", "X0", "X+M"), plan=PLAN_STEPS, extra=()):
    caps, plans, its = {}, {}, {}
    for iid in items:
        it = dict(GEN[iid], plan=list(plan), setup=[])
        its[iid] = it
        for lg in ("en", "fa"):
            pres = C.pres_id(iid, lg)
            caps[pres] = {"pres": pres, "summary_sha": C.sha256_text(pres), "restored_state": X.restored_state_of("canon", it, None), "status": "ok"}
            for a in list(arms) + list(extra):
                k = C.cell_id(ns, "plan", a, pres, "r0")
                plans[k] = {"key": k, "arm": a, "pres": pres, "r": 0, "block": "curve" if "@" in a else "main", "status": "ok", "plan": C.norm_plan(plan)}
    return caps, plans, its


snap = lambda s: "5" * 64                                            # noqa: E731
caps, plans, its = synth("rehearsal-v1", DEV[:3])
b_old = X0.build_units(ns="rehearsal-v1", captures=caps, plans=plans, items=its, snap_sha=snap, manifest=MAN)
b_new = X.build_units(ns="rehearsal-v1", captures=caps, plans=plans, items=its, snap_sha=snap, manifest=MAN)
check("1.11 execution units of an ft-v1 namespace: old == new, byte for byte (keys, cells, choosers, no exec_policy field)",
      C.canon(b_old) == C.canon(b_new) and not any("exec_policy" in u for u in b_new["units"].values()), (len(b_old["units"]), len(b_new["units"])))

S0 = load_pre("score", "score_pre", C0)
import score as SC                                                    # noqa: E402
inp = _types.SimpleNamespace(ns="rehearsal-v1", root=None, summary_sha={p: c["summary_sha"] for p, c in caps.items()}, hint_sha=None)
gold = {i: {"setup": []} for i in its}
cells = sorted(plans)[:12]
check("1.12 score.derive_unit of ft-v1 cells: old == new == the execute unit key",
      all(S0.derive_unit(inp, c, PLAN_STEPS, gold, snap_sha=lambda s, r: "5" * 64) == SC.derive_unit(inp, c, PLAN_STEPS, gold, snap_sha=lambda s, r: "5" * 64)
          and SC.derive_unit(inp, c, PLAN_STEPS, gold, snap_sha=lambda s, r: "5" * 64) in b_new["units"] for c in cells))

PK0 = load_pre("practice_k", "practice_k_pre", C0)
import practice_k as PK                                               # noqa: E402
check("1.13 the practice-v1 hook is the same source; the seeded orders are equal",
      inspect.getsource(PK0.PracticeHook) == inspect.getsource(PK.PracticeHook)
      and PK0.practice_order(sorted(MAN["practice_pool"]), PK.level_of_items(GEN), 0) == PK.practice_order(sorted(MAN["practice_pool"]), PK.level_of_items(GEN), 0))
check("1.14 practice-v1 snapshot tree is untouched by ft-v2 (different roots)",
      os.path.normpath(T.snapshot_dir("S*")) != os.path.normpath(C.snapshot_dir("S*")) and T.source_dir("practice-v1", "S*") == C.snapshot_dir("S*"))

# ================================================================================================ 2 the switch
print("== 2 the switch (pure)")
pt = T.planner_types(TYPES49)
check("2.1 planner types: 49 detector classes -> 48 shown, no Knife, ButterKnife kept, order A-Z",
      len(TYPES49) == 49 and len(pt) == 48 and "Knife" not in pt and "ButterKnife" in pt and pt == sorted(pt))
check("2.2 type_scope 'none' (A26 not approved) keeps all 49", len(T.planner_types(TYPES49, dict(T.FT2, type_scope="none"))) == 49)
fl = full_lines()
ps = T.prompt_summary(fl)
check("2.3 prompt summary: Floor and Knife lines dropped, the Cabinet pair grouped, every other type kept",
      not any(l.startswith(("Floor", "Knife")) for l in ps) and any(l.startswith("Cabinet x2") for l in ps)
      and {l.split(" ")[0] for l in ps} == {"Apple", "Cabinet", "Fridge", "Microwave", "Mug", "SinkBasin", "StoveKnob"}, ps)
fl50 = full_lines(45)
ps50 = T.prompt_summary(fl50)
ntypes = len({l.split(" ")[0] for l in fl50} - {"Floor", "Knife"})
check("2.4 A24 overflow: > 40 types -> exactly 40 lines = the first 39 types A-Z + one omitted-count line",
      len(ps50) == 40 and ps50[-1] == T.OMITTED_LINE.format(n=ntypes - 39)
      and [l.split(" ")[0] for l in ps50[:39]] == sorted({l.split(" ")[0] for l in fl50} - {"Floor", "Knife"})[:39], (len(ps50), ps50[-1:]))
check("2.5 the D2 filter runs BEFORE compaction (a Knife-only summary renders nothing about the Knife)",
      T.prompt_summary(["Knife [?] on/in Drawer at (1.00, 1.00), step 2"]) == [])
ex_ok = _types.SimpleNamespace(last={"skills": _types.SimpleNamespace(om=FakeOM(fl50))})
post50 = {"summary": fl50[:40]}
check("2.6 full summary: read from ex.last['skills'].om, proven to start with the executor's 40 lines", T.full_summary(ex_ok, post50) == fl50)
check("2.7 full summary refused: another belief state / a short cut that hides lines / no skills object",
      raises(lambda: T.full_summary(ex_ok, {"summary": fl50[1:41]}), T.TreatmentError)
      and raises(lambda: T.full_summary(ex_ok, {"summary": fl50[:10]}), T.TreatmentError)
      and raises(lambda: T.full_summary(_types.SimpleNamespace(last={}), post50), T.TreatmentError))
p4, p4raw = T.effective_params("pkg-1.r4"), LP.profile_params("pkg-1.r4")
p6 = T.effective_params("pkg-1.r6")
check("2.8 r4 + D2: type_scope forced on and re-hashed (as the r6b gate builds it); r6 keeps its own params",
      p4["precision"]["type_scope"] is True and not p4raw["precision"].get("type_scope") and p4["sha256"] != p4raw["sha256"]
      and p4["sha256"] == LP.params_sha(p4) and p6["sha256"] == LP.profile_params("pkg-1.r6")["sha256"])
p6n = T.effective_params("pkg-1.r6", cfg=dict(T.FT2, type_scope="none"))
check("2.9 type_scope 'none' switches the filter off in r6 too (A26 item 9: one freeze never mixes)", p6n["precision"]["type_scope"] is False and p6n["sha256"] != p6["sha256"])
check("2.10 package-only STRICT from the profile's own list (r4 has no 'kind' field: both r4 masks survive)",
      T.strict_off("pkg-1.r4", "g-l2-microwave-run-potato") == ("request_ops",) and T.strict_off("pkg-1.r4", "g-l2-microwave-run-egg") == ("precondition_closure",)
      and T.strict_off("pkg-1.r6", "g-l3-chilled-apple") == ("need_rule",) and "need_hand" in T.strict_off("pkg-1.r6b", "g-l3-need-read")
      and "tidy_switch_closed" not in T.strict_off("pkg-1.r6", "g-l2-microwave-run-egg"))
check("2.11 off_for: universal = mechanisms_off_all only; strict adds the item masks; an unknown global mechanism is refused",
      T.off_for("pkg-1.r4", "g-l2-microwave-run-potato", "universal") == ()
      and T.off_for("pkg-1.r4", "g-l3-tool-flip", "universal", dict(T.FT2, mechanisms_off_all=["request_ops"])) == ("request_ops",)
      and T.off_for("pkg-1.r4", "g-l3-tool-flip", "strict") == ("request_ops",)
      and raises(lambda: T.off_for("pkg-1.r4", "x", "universal", dict(T.FT2, mechanisms_off_all=["need_rule"])), T.TreatmentError))
check("2.12 STRICT refused in practice-v2, test-v1 and every legacy namespace; allowed in rehearsal-v3 only",
      all(raises(lambda n=n: T.mask_check(n, "strict"), T.TreatmentError) for n in ("practice-v2", "test-v1", "rehearsal-v1", "practice-v1"))
      and T.mask_check("rehearsal-v3", "strict") is None and raises(lambda: T.mask_check("rehearsal-v3", "bogus"), T.TreatmentError))
HOOK = {"G+M", "O+M", "X+M", "P+M", "G+M@o0k1", "O+M@o0k5"}
NOHOOK = {"G0", "O0", "X0", "G+FS", "O+FS", "P0", "G+M/p", "O+M/p"}
check("2.13 hook B: +M, curve and P+M only; never 0, FS, P0 or /p (Astra C4 Q2)",
      all(T.gets_hook(a) for a in HOOK) and not any(T.gets_hook(a) for a in NOHOOK))
_sv214 = T.TEST_V1_TREATMENT
T.TEST_V1_TREATMENT = "ft-v1"
try:
    _pre214 = all(T.exec_policy(a, n, "a" * 64) is None for a in HOOK | NOHOOK for n in LEGACY_NS)
finally:
    T.TEST_V1_TREATMENT = _sv214
check("2.14 exec_policy: None in every ft-v1 namespace and for hook-off arms; briefing:<sha16> for hook arms; a hook arm without a table is refused "
      "(after the flip test-v1 behaves as an ft-v2 namespace)",
      _pre214 and all(T.exec_policy(a, n, "a" * 64) is None for a in HOOK | NOHOOK for n in LEGACY_NS if n != "test-v1")
      and all(T.exec_policy(a, "test-v1", "a" * 64) == "briefing:" + "a" * 16 for a in HOOK)
      and all(T.exec_policy(a, "test-v1", "a" * 64) is None for a in NOHOOK)
      and all(T.exec_policy(a, "rehearsal-v3", "a" * 64) == "briefing:" + "a" * 16 for a in HOOK)
      and all(T.exec_policy(a, "rehearsal-v3", "a" * 64) is None for a in NOHOOK)
      and raises(lambda: T.exec_policy("G+M", "rehearsal-v3", None), T.TreatmentError))
check("2.15 chooser: prior for every ft-v2 arm; the arm table in ft-v1",
      all(T.effective_chooser(a, "rehearsal-v3") == "prior" for a in HOOK | NOHOOK)
      and T.effective_chooser("G+M", "rehearsal-v1") == "o0k10" and T.effective_chooser("G+M@o0k3", "rehearsal-v1") == "o0k3" and T.effective_chooser("G0", "rehearsal-v1") == "prior")
tab = {"Mug": ["Cabinet", "Shelf"], "Apple": ["Fridge"]}
check("2.16 hint-table sha: order-preserving per type, key-order independent",
      T.hint_table_sha(tab) == T.hint_table_sha(dict(reversed(list(tab.items())))) != T.hint_table_sha({"Mug": ["Shelf", "Cabinet"], "Apple": ["Fridge"]}))


class FakeStore:
    def __init__(self, recs):
        self.recs = recs

    def candidates(self, q):
        return [{"rec": r} for r in self.recs]

    def afford_records(self):
        return list(self.recs)


recs = [{"id": "a", "provenance": {"support": ["g-x"]}}, {"id": "b", "provenance": {"tuned_on": ["g-x"]}},
        {"id": "c", "provenance": {"evidence": ["ep:g-x:1"]}}, {"id": "d", "provenance": {"support": ["g-y"]}}]
log = []
lo = T.ItemLeaveOut(FakeStore(recs), "g-x", log, "procedure")
check("2.17 leave-out: support, tuned_on and evidence citations dropped and LOGGED; others kept",
      [c["rec"]["id"] for c in lo.candidates({})] == ["d"] and [r["id"] for r in lo.afford_records()] == ["d"]
      and sorted({(d["id"], d["why"]) for d in log}) == [("a", "support"), ("b", "tuned_on"), ("c", "evidence")])
check("2.18 leave-out on a TEST presentation: an attributable record is an integrity failure, never a silent drop (Astra Q7)",
      raises(lambda: T.ItemLeaveOut(FakeStore(recs), "g-x", [], "procedure", fail=True).candidates({}), T.TreatmentError)
      and [c["rec"]["id"] for c in T.ItemLeaveOut(FakeStore(recs[3:]), "g-x", [], "procedure", fail=True).candidates({})] == ["d"])
rows_ok = [{"key": "k1", "treatment": "ft-v2", "treatment_id": "I"}]
check("2.19 journal identity: an ft-v1 row in an ft-v2 namespace, a different identity, an ft-v2 row in a legacy namespace are refused",
      T.journal_problems(rows_ok, "rehearsal-v3", "I") == [] and T.journal_problems([{"key": "k"}], "rehearsal-v3", "I")
      and T.journal_problems(rows_ok, "rehearsal-v3", "J") and T.journal_problems(rows_ok, "rehearsal-v1") and T.journal_problems([{"key": "k"}], "rehearsal-v1") == [])
_sv220 = T.TEST_V1_TREATMENT
T.TEST_V1_TREATMENT = "ft-v1"
try:
    _pre220 = (T.entry_problems("test-v1", {}, None) == [] and T.entry_problems("test-v1", {"treatment": {"version": "ft-v2", "sha": "x"}}, None))
finally:
    T.TEST_V1_TREATMENT = _sv220
check("2.20 test-v1 entry: ft-v1 (pre-flip toggle) refuses a freeze that records ft-v2; after the flip an empty freeze is refused; no check "
      "outside test-v1", _pre220 and T.entry_problems("test-v1", {}, None) != []
      and T.entry_problems("rehearsal-v3", {"treatment": {"version": "ft-v2"}}, "y") == [])
_saved = T.TEST_V1_TREATMENT
T.TEST_V1_TREATMENT = "ft-v2"
try:
    flip = (T.is_ft2("test-v1") and T.entry_problems("test-v1", {}, "I") and T.entry_problems("test-v1", {"treatment": {"sha": "J"}}, "I")
            and T.entry_problems("test-v1", {"treatment": {"sha": "I"}}, "I") == [])
finally:
    T.TEST_V1_TREATMENT = _saved
check("2.21 after the flip: test-v1 is ft-v2 and every entry needs freeze.json's identity", bool(flip))
br = {"records_sha": "r", "n_records": 1, "allow_unapproved": True, "hint_table_sha": "h"}
i1 = T.identity("rehearsal-v3", briefing=br)
i2 = T.identity("rehearsal-v3", briefing=br, cfg=dict(T.FT2, package_profile="pkg-1.r6"))
i3 = T.identity("rehearsal-v3", briefing=dict(br, records_sha="r2"))
check("2.22 identity: deterministic; a profile switch or a briefing edit changes it; it pins every source file",
      i1["sha"] == T.identity("rehearsal-v3", briefing=br)["sha"] and len({i1["sha"], i2["sha"], i3["sha"]}) == 3
      and set(i1["files"]) == set(T.IDENTITY_FILES) and all(v != "absent" for v in i1["files"].values()), [k for k, v in i1["files"].items() if v == "absent"])

# ================================================================================================ 3 the real package
print("== 3 the real package (practice-v1 S* store + its order-0 rows; rehearsal-v3 draft briefing)")
SNAP_DB = os.path.join(ROOT, "sim", "final_test", "snapshots", "o0", "k10", "ltm.db")
ROWS10 = LP.load_practice_rows("practice-v1")
PK_RES = {}


def mk_packager(cfg=None, test_side=None, ns="rehearsal-v3"):
    return T.Packager(ns, FakeCounter(), cfg=cfg or T.config(), test_side=test_side)


pa = mk_packager()
pres3 = [(i, l) for i in SIGNAL if i not in ("g-l3-cutlery-sink", "g-l3-tool-cut") for l in ("en", "fa")][:10]
outs, problems = {}, []
for i, l in pres3:
    it = GEN[i]
    try:
        outs[(i, l)] = pa.render("S*", SNAP_DB, ROWS10, item=i, instruction=it[l], lang=l, holding=None, summary_lines=ps)
    except Exception as e:                                            # noqa: BLE001
        problems.append(f"{i}/{l}: {type(e).__name__}: {str(e)[:200]}")
check("3.1 the real package renders on 10 signal presentations (no abort)", not problems and len(outs) == 10, problems[:3])
re_ = {k: pa.render("S*", SNAP_DB, ROWS10, item=k[0], instruction=GEN[k[0]][k[1]], lang=k[1], holding=None, summary_lines=ps)["text"] for k in list(outs)[:4]}
check("3.2 identical bytes on re-render (G+M / O+M / X+M read one block)", all(re_[k] == outs[k]["text"] for k in re_))
check("3.3 every package within 2,000 tokens with PACKAGE_HEADER and within the profile cap; no Knife; no Persian",
      all(o["tokens"]["real_max"] <= 2000 and (not o["tokens"]["budget"] or o["tokens"]["retrieved_max"] <= o["tokens"]["budget"])
          and not T._ts().mentions_excluded(o["text"]) and not any("؀" <= ch <= "ۿ" for ch in o["text"]) for o in outs.values()),
      {k: o["tokens"] for k, o in list(outs.items())[:2]})
check("3.4 at least one package is non-empty and carries the PINNED profile's universal params sha (TREAT-FREEZE: pkg-1.r6b; was the r4 + D2 placeholder)",
      any(o["text"] for o in outs.values()) and T.FT2["package_profile"] == "pkg-1.r6b"
      and all(o["params_sha"] == T.effective_params(T.FT2["package_profile"])["sha256"] for o in outs.values()))
cited = [(k, o["leave_out"]) for k, o in outs.items() if o["leave_out"]]
print("   leave-out drops (dev presentations):", [(k, [d["id"] for d in v][:3]) for k, v in cited][:3])
if cited:
    (ci, cl), _ = cited[0]
    pt_ = mk_packager(test_side=True)
    check("3.5 the same presentation as a TEST presentation: the cited record is an integrity failure",
          raises(lambda: pt_.render("S*", SNAP_DB, ROWS10, item=ci, instruction=GEN[ci][cl], lang=cl, holding=None, summary_lines=ps), T.TreatmentError))
else:
    check("3.5 (no dev presentation cites a record; the fail mode is covered by 2.18)", True)
alt = {}
for prof in ("pkg-1.r6", "pkg-1.r6b"):
    pp = mk_packager(cfg=dict(T.config(), package_profile=prof))
    alt[prof] = {k: pp.render("S*", SNAP_DB, ROWS10, item=k[0], instruction=GEN[k[0]][k[1]], lang=k[1], holding=None, summary_lines=ps)
                 for k in list(outs)[:6]}
check("3.6 profile switch r4 / r6 / r6b: each renders under its own params sha, within budget, no Knife",
      all(o["profile"] == prof and o["params_sha"] == T.effective_params(prof)["sha256"] and o["tokens"]["real_max"] <= 2000
          and not T._ts().mentions_excluded(o["text"]) for prof, d in alt.items() for o in d.values()))
pst = mk_packager(cfg=dict(T.config()), test_side=False)
check("3.7 a STRICT packager cannot even be built outside rehearsal-v3", raises(lambda: T.Packager("practice-v2", FakeCounter(), mask="strict"), T.TreatmentError))
_old_rows = list(ROWS10)
check("3.8 the source rows are not mutated by a render", ROWS10 == _old_rows)

# ================================================================================================ 4 capture ft-v2
print("== 4 capture ft-v2 (fake executor; real package)")
tmp = tempfile.mkdtemp(prefix="ftm_c4_")
try:
    def seed_root(root):
        d = T.snapshot_dir("S*", root)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(SNAP_DB, os.path.join(d, "ltm.db"))
        T.write_rows(d, ROWS10)
        return d

    sdir = seed_root(tmp)
    check("4.0 the ft-v2 S* snapshot carries its own row prefix (content-bound)", T.read_rows(sdir) == ROWS10 and T.rows_sha(T.read_rows(sdir)) == T.rows_sha(ROWS10))

    class FakeEx:
        frame, map_rev = "f0", "m0"

        def __init__(self, lines):
            self.last = {"skills": _types.SimpleNamespace(om=FakeOM(lines))}

    FS_DOC = {lv: {"text": "Earlier: pickup Mug, put_in Mug Sink", "real_tokens": 10, "counts": {}, "budget": 2000} for lv in ("L1", "L2", "L3")}
    c2 = PlannerContract(T.planner_types(TYPES49), tools=hs.TOOLS, version="ft-v2")

    def mk_cap(ns="rehearsal-v3", cfg=None, ex=None, contract=c2):
        cap = K.Capturer(namespace=ns, executor=ex or FakeEx(fl), contract=contract, counter=FakeCounter(), open_sources=None, items=GEN,
                         manifest=MAN, fs_doc=FS_DOC, root=tmp, only_sources=["none", "S*", "FS"], log=QUIET,
                         packager=T.Packager(ns, FakeCounter(), cfg=cfg or T.config(), root=tmp) if T.is_ft2(ns) else None)
        cap.ident = T.identity(ns, briefing=br) if T.is_ft2(ns) else None
        return cap

    iid = DEV[0]
    post = {"summary": fl[:40], "summary_sha": C.sha256_text("\n".join(fl[:40])), "episode_id": "ep:1", "sweep_prims": 3}
    cap = mk_cap()
    cap.cur = {"pres": C.pres_id(iid, "en"), "level": GEN[iid]["level"], "lang": "en", "world": "canon"}
    out = cap.render_presentation(None, GEN[iid], None, post)
    none_u = c2.messages(GEN[iid]["en"], out["prompt_summary"], None)[1]["content"]
    check("4.1 sources none / S* / FS rendered; the none request has no block header of any kind",
          sorted(out["requests"]) == ["FS", "S*", "none"] and PACKAGE_HEADER not in none_u and MEMORY_HEADER not in none_u)
    check("4.2 the planner sees the prompt summary (A24 rule over the full list), never the Knife or the Floor; observed types follow it",
          out["prompt_summary"] == T.prompt_summary(fl) and out["observed_types"] == T.observed_types(out["prompt_summary"])
          and "Knife" not in none_u.split("Command:")[0] and out["summary_sha"] == post["summary_sha"] and out["full_summary_n"] == len(fl))
    sys_txt = c2.system_text()
    check("4.3 the ft-v2 system text lists the 48 types (no Knife in the type list)", ", ".join(pt) in sys_txt and ", ".join(sorted(TYPES49)) not in sys_txt)
    print("   INFO (K-B2b dependency, Astra C4 Q9): the ft-v2 system text still names the Knife in its example:",
          "pickup(Knife)" in sys_txt)
    sb = out["blocks"]["S*"]
    check("4.4 the S* block IS the package (identity of params, source rows sha, tokens)",
          sb["package"]["profile"] == T.FT2["package_profile"] and sb["source"]["rows_sha"] == T.rows_sha(ROWS10) and sb["real_tokens"] <= 2000)
    m_s = c2.messages(GEN[iid]["en"], out["prompt_summary"], None, memory_block=sb["text"] or None)
    check("4.5 the S* request = none + exactly one PACKAGE_HEADER block (G-iso with the package header)",
          prompt_sha256(m_s) == out["requests"]["S*"] and (not sb["text"] or m_s[1]["content"].count(PACKAGE_HEADER) == 1))
    cap6 = mk_cap(cfg=dict(T.config(), package_profile="pkg-1.r6"))
    cap6.cur = dict(cap.cur)
    out6 = cap6.render_presentation(None, GEN[iid], None, post)
    check("4.6 profile switch r4 -> r6: the none and FS request bytes are identical; only the package block may change",
          out6["requests"]["none"] == out["requests"]["none"] and out6["requests"]["FS"] == out["requests"]["FS"]
          and out6["blocks"]["S*"]["package"]["profile"] == "pkg-1.r6")
    capx = mk_cap(ex=FakeEx(fl[1:]))
    capx.cur = dict(cap.cur)
    check("4.7 capture refuses an executor whose full list is another belief state", raises(lambda: capx.render_presentation(None, GEN[iid], None, post), K.CaptureAbort))
    capn = mk_cap()
    capn.cur = dict(cap.cur)
    capn.packager = None
    check("4.8 an ft-v2 capture without a packager refuses the +M sources", raises(lambda: capn.render_presentation(None, GEN[iid], None, post), K.CaptureAbort))
    leg = K.Capturer(namespace="rehearsal-v1", executor=FakeEx(fl), contract=cleg_new, counter=FakeCounter(), open_sources=None, items=GEN,
                     manifest=MAN, fs_doc=FS_DOC, root=tmp, only_sources=["none", "FS"], log=QUIET)
    leg.cur = dict(cap.cur)
    outl = leg.render_presentation(None, GEN[iid], None, post)
    check("4.9 ft-v1 capture is the legacy path (no prompt summary, the executor's summary in the prompt, MEMORY slot)",
          not leg.ft2 and "prompt_summary" not in outl and outl["requests"]["none"] == prompt_sha256(cleg_new.messages(GEN[iid]["en"], post["summary"], None)))

    # ============================================================================================ 5 plan ft-v2
    print("== 5 plan ft-v2")
    caprec = {"instruction": GEN[iid]["en"], "summary": post["summary"], "holding": None, "blocks": out["blocks"], "treatment": "ft-v2",
              "prompt_summary": out["prompt_summary"], "observed_types": out["observed_types"], "requests": out["requests"]}
    cv2 = PLN.make_contract(TYPES49, "rehearsal-v3")
    check("5.1 make_contract(ft-v2 ns): version ft-v2, 48 types", cv2.version == "ft-v2" and cv2.types == pt)
    check("5.2 request identity: plan re-renders every captured sha (none / S* / FS) from the capture row",
          all(prompt_sha256(PLN.render_messages(cv2, caprec, s)) == out["requests"][s] for s in ("none", "S*", "FS")))
    check("5.3 summary_of: ft-v2 rows -> prompt summary; ft-v1 rows -> summary (unchanged)",
          PLN.summary_of(caprec) == out["prompt_summary"] and PLN.summary_of(rec_leg) == rec_leg["summary"])
    m2 = PLN.render_messages(cv2, caprec, "S*")
    w1, w2 = PLN.cell_worst(FakeProv(), cleg_new, m2), PLN.cell_worst(FakeProv(), cv2, m2)
    check("5.4 the ft-v2 worst case holds first + 2 repairs (3 calls; larger than the ft-v1 first + 1)", w2 > w1 > 0, (w1, w2))
    psrc = inspect.getsource(PLN.Plan._call_cell)
    check("5.5 the runner gets observed_types in ft-v2 only; the treatment check runs before the request gate",
          "observed_types" in psrc and "if self.ft2" in psrc
          and inspect.getsource(PLN.Plan._prepare).index("_treatment_check") < inspect.getsource(PLN.Plan._prepare).index("_verify_requests"))
    check("5.6 practice-v2 has no Phase B", raises(lambda: PLN.Plan("practice-v2", root=tmp), ValueError))

    # ============================================================================================ 6 execute / score ft-v2
    print("== 6 execute / score ft-v2")
    ns3 = "rehearsal-v3"
    caps3, plans3, its3 = synth(ns3, DEV[:2], extra=("G+M@o0k1",))
    hsha = T.hint_table_sha(tab)
    b3 = X.build_units(ns=ns3, captures=caps3, plans=plans3, items=its3, snap_sha=snap, manifest=MAN, hint_sha=hsha)
    by_p = {}
    for u in b3["units"].values():
        by_p.setdefault(u["pres"], []).append(u)
    groups_ok = True
    for pres, us in by_p.items():
        arms = [sorted(c["arm"] for c in u["cells"]) for u in us]
        hook_u = [u for u in us if u["exec_policy"]]
        off_u = [u for u in us if not u["exec_policy"]]
        groups_ok &= (len(us) == 2 and len(hook_u) == 1 and len(off_u) == 1
                      and {c["arm"] for c in hook_u[0]["cells"]} == {"G+M", "O+M", "X+M", "P+M", "G+M@o0k1"}
                      and {c["arm"] for c in off_u[0]["cells"]} == {"G0", "O0", "X0", "G+FS", "O+FS", "P0", "G+M/p", "O+M/p"})
    check("6.1 equal plans: ONE hook unit {G+M, O+M, X+M, P+M, curve} and ONE hook-off unit {G0, O0, X0, FS, P0, /p} per presentation",
          groups_ok and by_p, {p: [sorted(c["arm"] for c in u["cells"]) for u in us] for p, us in list(by_p.items())[:1]})
    check("6.2 every ft-v2 unit runs the prior chooser with no snapshot; the hook units carry briefing:<sha16>",
          all(u["chooser"] == "prior" and u["snapshot_sha"] is None for u in b3["units"].values())
          and all(u["exec_policy"] in (None, "briefing:" + hsha[:16]) for u in b3["units"].values()))
    inp3 = _types.SimpleNamespace(ns=ns3, root=None, summary_sha={p: c["summary_sha"] for p, c in caps3.items()}, hint_sha=hsha)
    gold3 = {i: {"setup": []} for i in its3}
    allcells = [c for u in b3["units"].values() for c in u["cells"]]
    der = all(SC.derive_unit(inp3, c["key"], PLAN_STEPS, gold3) == u["unit"] for u in b3["units"].values() for c in u["cells"])
    check("6.3 score.derive_unit reproduces execute's key for every ft-v2 cell (incl. P+M, /p and the curve)", der and len(allcells) >= 26)
    check("6.4 derive_unit without the recorded hint table cannot derive a hook unit (None, never a hook-off key)",
          SC.derive_unit(_types.SimpleNamespace(ns=ns3, root=None, summary_sha=inp3.summary_sha, hint_sha=None),
                         C.cell_id(ns3, "plan", "G+M", sorted(caps3)[0], "r0"), PLAN_STEPS, gold3) is None)

    class PolEx:
        def __init__(self):
            self.got = []

        def set_episode_policy(self, p):
            self.got.append(p)

    pe = PolEx()
    fn = T.hints_fn(tab)
    T.install_policy(pe, "briefing:x", fn)
    T.install_policy(pe, None)
    check("6.5 install_policy: hook B for a hook unit, an explicit None otherwise; an executor without the pass-through is refused",
          pe.got[0]["briefing_hints"]("Mug") == ["Cabinet", "Shelf"] and pe.got[1] is None
          and raises(lambda: T.install_policy(_types.SimpleNamespace(), "briefing:x", fn), T.TreatmentError)
          and raises(lambda: T.install_policy(pe, "briefing:x", None), T.TreatmentError))
    xsrc = inspect.getsource(X.ExecutePhase.run_one)
    check("6.6 run_one installs the per-episode policy only in ft-v2 and records briefing_log + exec_policy there",
          "if self.ft2:" in xsrc and "install_policy" in xsrc and "briefing_log" in xsrc)
    gsrc = inspect.getsource(X.ExecutePhase.g_iso)
    check("6.7 ft-v2 G-iso: P+M (prompt none, hook on) is not a 0 unit; hook B in a 0 unit fails the gate",
          "exec_policy\") is None" in gsrc and "hook_leak" in gsrc and "status = \"fail\"" in gsrc)
    import step5_live as SL                                           # noqa: E402
    lx = SL.LiveExecutor(memory="places")
    lx.set_episode_policy({"briefing_hints": fn})
    check("6.8 LiveExecutor: pass-through default off (ltm_chooser True, no policy); a bad policy is refused",
          SL.LiveExecutor().ltm_chooser is True and SL.LiveExecutor()._episode_policy is None and lx._episode_policy["briefing_hints"] is fn
          and raises(lambda: lx.set_episode_policy({"x": 1}), ValueError) and SL.LiveExecutor(ltm_chooser=False).ltm_chooser is False)
    esrc = inspect.getsource(SL.LiveExecutor.execute)
    check("6.9 LiveExecutor.execute consumes the policy first; HonestSkills gets the briefing keywords only when set; the LTM chooser obeys ltm_chooser",
          esrc.index("ep_policy, self._episode_policy = self._episode_policy, None") < esrc.index("ev._reset_world")
          and "if ep_policy is not None:" in esrc and "allow_briefing_hints=True" in esrc and "if mem_on and self.ltm_chooser:" in esrc)
    old_live = open(os.path.join(PRE, "step5_live.py"), encoding="utf-8").read()
    check("6.10 the legacy HonestSkills call is unchanged (the new keywords only via `pol`)",
          "sk = hs_mod.HonestSkills(h, self._gm_cls.load(self.scene), self._proxy, chooser=chooser or hs_mod.prior_chooser," in old_live
          and "sk = hs_mod.HonestSkills(h, nav, self._proxy, kitchen=self.scene, chooser=chooser or hs_mod.prior_chooser," in esrc
          and "nav = (self._gm_cls or _GridMap).load(self.scene)" in esrc)
          # K23 agent-side B4 (Astra K23-AGENT-r1): the ONE added keyword is the look-alike kitchen (FloorPlan1 = the old default); the map is
          # the same file, loaded once per episode before the simulator is touched and scene-checked (Astra FP1-equivalence review 2026-10-18)

    # ============================================================================================ 7 practice ft-v2
    print("== 7 practice ft-v2 (fake executor, real practice LTM, real package)")
    import ltm as _L                                                  # noqa: E402
    _lp_real = _L.load_params

    def _lp(path=None):
        import copy
        P = copy.deepcopy(_lp_real(path))
        P["retrieval"].pop("retention", None)
        P["retrieval"].pop("a12", None)
        return P

    _L.load_params = _lp

    class FakeRunner2:
        def __init__(self):
            self.calls = []

        def plan(self, command, memory_summary=(), holding=None, memory_block=None, fewshot_block=None, replicate=0, observed_types=None):
            self.calls.append({"summary": list(memory_summary), "block": memory_block, "observed": observed_types})
            plan = [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]
            return {"ok": True, "first_ok": True, "repaired": False, "rejected": False, "infra_failure": False, "calls": 1, "tokens_in": 10,
                    "tokens_out": 5, "plan": plan, "first_plan": plan, "say": "ok", "normalised": False, "attempts": [], "codes_by_attempt": [[]]}

    class FakeExec2:
        frame, map_rev = "f0", "m0"

        def __init__(self):
            self.plan_hook = None
            self.last = None
            self.policies = []
            self._pol = None

        def set_episode_policy(self, p):
            self._pol = p

        def execute(self, req):
            import ltm_adapter as A
            import step5_eval as se
            pol, self._pol = self._pol, None
            self.policies.append(pol)
            it = req.item
            res = se.EpisodeResult()
            ad = A.HonestEventAdapter(req.memory, episode=req.episode_id, mode="practice", world=req.world, frame=self.frame, map_rev=self.map_rev,
                                      kitchen=req.kitchen)
            ad.on_task(it[req.lang], it["level"], req.lang)
            for j, ot in enumerate(("Apple", "Mug", "Bread")):
                ad._emit("sighting", otype=ot, state="?", conf=0.9, detector="det", track=f"tr:{j + 1}", xyz=(1.0, 0.9, 1.0 + 0.3 * j), xyz_ok=True,
                         view="vw:1", obs="ob:1", in_rtype="Fridge", in_basis="box_only", in_xyz=(1.0, 0.9, 1.1))
            blog = [{"otype": "Apple", "hint": (pol or {}).get("briefing_hints", lambda t: [])("Apple"), "asked": 1}] if pol else []
            self.last = {"adapter": ad, "skills": _types.SimpleNamespace(om=FakeOM(fl), briefing_log=blog)}
            post = {"wm": ad.wm, "summary": fl[:40], "summary_sha": C.sha256_text("\n".join(fl[:40])), "holding": None, "events": ad.stream, "sweep_prims": 3}
            ph = self.plan_hook(req, it, None, req.memory, post=post)
            for i, s in enumerate(ph["steps"] or []):
                ad.on_step_start(i, {"action": s["action"], "object": s["object"]})
                ad.on_step_end(i, {"ok": True, "retries": 0, "detail": {"prims": 5}})
            ad.on_terminal(True, 8, 0)
            ad.on_admission(True)
            ad.end()
            res.goal_ok, res.wm, res.admission = True, ad.wm, True
            res.extra = {"plan": ph["info"], "plan_not_executed": False, "steps": [{"ok": True, "action": "pickup"}],
                         "core": {"prims": 8, "audit": {"violations": 0, "counts": {"oracle": 0}}}}
            return res

        def close(self):
            pass

    ptmp = os.path.join(tmp, "practice")
    os.makedirs(ptmp, exist_ok=True)
    ident3 = T.identity("rehearsal-v3", briefing=br)
    hints3 = T.hints_fn(T.briefing_hint_table(allow_unapproved=True))

    def mk_practice(ex, hook, ident=ident3, orders=(0,), ks=(0, 1, 3), sub="a", **kw):
        return PK.Practice(namespace="rehearsal-v3", orders=list(orders), ks=list(ks), executor=ex, hook=hook, counter=FakeCounter(),
                           snap_root=os.path.join(ptmp, sub, "snap"), journal=os.path.join(ptmp, sub, "docs", "practice.jsonl"),
                           events=os.path.join(ptmp, sub, "docs", "events.jsonl"), work=os.path.join(ptmp, sub, "work"), log=QUIET,
                           ident=ident, hints=hints3, **kw)

    ex2, run2 = FakeExec2(), FakeRunner2()
    hook2 = PK.PackagePracticeHook(run2, c2, T.Packager("rehearsal-v3", FakeCounter()))
    ex2.plan_hook = hook2
    pr = mk_practice(ex2, hook2)
    res = pr.run()
    rows = C.Journal(os.path.join(ptmp, "a", "docs", "practice.jsonl")).rows()
    check("7.1 a 3-attempt ft-v2 practice run completes; every row carries the treatment identity, exec_policy briefing and the briefing log",
          res and res[0]["complete"] and len(rows) == 3 and all(r["treatment"] == "ft-v2" and r["treatment_id"] == ident3["sha"]
                                                                and r["exec_policy"] == "briefing" for r in rows) and rows[0]["briefing_log"], res)
    check("7.2 hook B reached the executor on EVERY attempt (prior chooser + B)", len(ex2.policies) == 3 and all(p and callable(p["briefing_hints"]) for p in ex2.policies))
    check("7.3 the planner got the prompt summary, observed types and the package text (ft-v2 contract)",
          all(c["summary"] == T.prompt_summary(fl) and c["observed"] == set(T.observed_types(T.prompt_summary(fl))) for c in run2.calls)
          and all((r.get("block") or {}).get("package", {}).get("profile") == T.FT2["package_profile"] for r in rows))
    check("7.4 the package of attempt n is built from the earlier rows only (0, 1, 2 rows)",
          [(r.get("block") or {}).get("source", {}).get("rows_n") for r in rows] == [0, 1, 2])
    sn = {k: os.path.join(ptmp, "a", "snap", "o0", "k%d" % k) for k in (0, 1, 3)}
    snap_ok = all(T.read_rows(sn[k]) == rows[:k] for k in (0, 1, 3))
    mans = {k: PK.read_snapshot_manifest(sn[k]) for k in (0, 1, 3)}
    check("7.5 every snapshot holds exactly its own row prefix, equal to the journal (content-bound by rows_sha in the manifest)",
          snap_ok and all(mans[k]["rows_sha"] == T.rows_sha(rows[:k]) and mans[k]["rows_n"] == k and mans[k]["treatment_id"] == ident3["sha"] for k in (0, 1, 3)))
    check("7.6 the snapshots verify (store + manifest + leakage) with the extra rows file", all(PK.verify_snapshot_dir(sn[k], 0, k) == [] for k in (0, 1, 3)))
    ident_other = T.identity("rehearsal-v3", briefing=dict(br, records_sha="edited"))
    pr2 = mk_practice(FakeExec2(), hook2, ident=ident_other, ks=(0, 1, 3, 5))
    check("7.7 resume across a treatment change (briefing edited) is refused", raises(lambda: pr2.run(), PK.PracticeError))
    check("7.8 ft-v2 practice refuses orders 1 / 2 (A27), memory channels, and the places hook",
          raises(lambda: mk_practice(FakeExec2(), hook2, orders=(1,), sub="b"), PK.PracticeError)
          and raises(lambda: mk_practice(FakeExec2(), hook2, sub="c", channels=("places", "procedures")), PK.PracticeError)
          and raises(lambda: mk_practice(FakeExec2(), PK.PracticeHook(run2, c2, FakeCounter(), ("places",)), sub="d"), PK.PracticeError))
    import kitchen_briefing as KB                                     # noqa: E402
    # K-B2b 2026-10-13: Arash approved the briefings (decision 7; overrides v1 approved_by "Arash"), so the registered path now LOADS.
    # Replaces: "7.9 ... today it is refused (approved_by is empty)" (raises BriefingNotApproved on both calls). The refusal of an
    # unapproved briefing on that path stays covered by _pkg_briefing_selftest (load_records without allow_unapproved).
    def _no_raise(f):
        try:
            f()
            return True
        except Exception:
            return False
    check("7.9 registered practice (practice-v2) runs on the APPROVED briefing: identity and the hint table load without allow_unapproved",
          _no_raise(lambda: T.identity("practice-v2")) and _no_raise(lambda: T.briefing_hint_table(allow_unapproved=False))
          and KB._load(KB._paths("FloorPlan1")["overrides"]).get("approved_by") == "Arash")
    _L.load_params = _lp_real
    blsrc = inspect.getsource(PK.build_live)
    check("7.10 the live practice build: prior chooser (ltm_chooser False), the package hook, the namespace's briefing policy",
          '{"ltm_chooser": False} if ft2' in blsrc and "PackagePracticeHook" in blsrc and "allow_unapproved=T.allow_draft_briefing(ns)" in blsrc)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 8 TREAT-FREEZE pins (Astra TREAT-FREEZE 1-7)
print("8 TREAT-FREEZE: the pinned executor, switches and contract pair; fail closed; treatment-only freeze refused")
import copy                                                            # noqa: E402
PINS = T.FT2.get("pins") or {}
check("8.1 the treatment is pinned: pkg-1.r6b, universal, D2, 2 repairs; executor 3 files, switches (F-dup, E3, PICK_MULTI, cut rule, D2), "
      "contract pair; the live tree matches every pin", T.FT2["package_profile"] == "pkg-1.r6b" and T.FT2["mask"] == "universal"
      and T.FT2["type_scope"] == "D2" and T.FT2["repairs"] == 2 and len(PINS.get("executor") or {}) == 3
      and (PINS.get("switches") or {}).get("objmem") == {"DEDUP_FRAME": True}
      and all((PINS["switches"]["honest_skills"] or {}).get(k) is True for k in ("LOOKALIKE_ON", "PICK_MULTI_ON", "CUT_RULE_ON", "TYPE_SCOPE_ON"))
      and PINS["switches"]["honest_skills"].get("CUTTING_TOOLS") == ["ButterKnife"] and PINS["switches"]["type_scope"] == {"EXCLUDED_TYPES": ["Knife"]}
      and set(PINS.get("contract") or {}) == {"sim/planners/exec_contract.json", "sim/planners/ft_v2.json"} and T.pin_problems() == [],
      T.pin_problems()[:3])
_id8 = T.identity("practice-v2")
check("8.2 the identity covers the executor, contract, perception weights and providers (present, executor shas = the pins)",
      all(f in T.IDENTITY_FILES for f in list(PINS["executor"]) + list(PINS["contract"]) + ["sim/weights/v3/FloorPlan1.yolo26s.pt",
                                                                                           "sim/weights/v3/FloorPlan1.amb26m.pt", "sim/step4_gemma.py"])
      and all(v != "absent" for v in _id8["files"].values()) and all(_id8["files"][k] == v for k, v in PINS["executor"].items()),
      [k for k, v in _id8["files"].items() if v == "absent"])


def _cfg(path, val):
    c = copy.deepcopy(T.FT2)
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val
    return c


_neg = {"switch": _cfg(("pins", "switches", "honest_skills", "LOOKALIKE_ON"), False),
        "objmem switch": _cfg(("pins", "switches", "objmem", "DEDUP_FRAME"), False),
        "executor sha": _cfg(("pins", "executor", "sim/honest.py"), "0" * 64),
        "contract texts": _cfg(("pins", "contract", "sim/planners/ft_v2.json", "texts"), "1" * 64),
        "contract self": _cfg(("pins", "contract", "sim/planners/exec_contract.json", "self"), "2" * 64),
        "no pins": _cfg(("pins",), {})}
check("8.3 fail closed: identity() raises TreatmentError on a switch, executor sha, contract or missing-pin difference (never a new silent sha)",
      all(raises(lambda c=c: T.identity("practice-v2", cfg=c), T.TreatmentError) for c in _neg.values()),
      {k: T.pin_problems(c)[:1] for k, c in _neg.items()})
import honest_skills as _HS8                                           # noqa: E402
_HS8.PICK_MULTI_ON = False
try:
    _rt = T.pin_problems()
finally:
    _HS8.PICK_MULTI_ON = True
check("8.4 a run-time override of a loaded switch is caught (effective value, Astra 3); restored -> clean",
      any("at run time" in p and "PICK_MULTI_ON" in p for p in _rt) and T.pin_problems() == [], _rt[:2])
_tmp8 = tempfile.mkdtemp(prefix="ftm_tf8_")
try:
    _p = os.path.join(_tmp8, "m.py")
    with open(_p, "w", encoding="utf-8") as f:
        f.write("A_ON = True\nB = frozenset({'x'})\nA_ON = False\n")
    _v, _pp = T._source_values(_p, {"A_ON", "B"})
    check("8.5 a second top-level assignment of a pinned switch is a problem; frozenset literals read as sorted lists",
          _v.get("B") == ["x"] and any("more than once" in x for x in _pp), (_v, _pp))
    os.makedirs(os.path.join(_tmp8, "docs", "final-test"))
    _fz = {"schema": "final-test.freeze/1", "freeze_status": T.TREATMENT_ONLY, "treatment": {"sha": _id8["sha"], "version": "ft-v2"},
           "gates": {"12.1": {"pass": True}}, "arash_check": {"by": "Arash", "date": "2026-10-13"}}
    _fz["sha256"] = C.self_sha(_fz)
    with open(os.path.join(_tmp8, "docs", "final-test", "freeze.json"), "w", encoding="utf-8") as f:
        json.dump(_fz, f)
    _fs = C.freeze_status(_tmp8, check_code=False)
    check("8.6 a treatment-only freeze.json never authorizes: entry_problems refuses it even with a matching sha, and common.freeze_status "
          "reports it", T.entry_problems("test-v1", _fz, _id8["sha"]) != [] and not _fs["ok"]
          and any("not a validated freeze" in r for r in _fs["reasons"]), (T.entry_problems("test-v1", _fz, _id8["sha"]), _fs["reasons"][:3]))
finally:
    shutil.rmtree(_tmp8, ignore_errors=True)
check("8.7 cross-stage equivalence: practice-v2 and test-v1 resolve to ONE identity (the namespace is not part of it; same params, briefing)",
      T.identity("test-v1")["sha"] == _id8["sha"] and _id8["profile"] == "pkg-1.r6b" and _id8["briefing"]["allow_unapproved"] is False)
check("8.8 exec_contract derived_from = the live executor and step4_gemma (LF-normalised sha16), ft_v2 points at that contract",
      not [p for p in T.pin_problems() if "derived_from" in p or "one build" in p])

# ---------------------------------------------------------------------------------------------- 9 A31 whole-id matching (B1)
print("9 A31: whole-id matching in _cites (evidence) and corpus_test_citations")
_COLL = [("g-l2-slice-apple", "g-l2-slice-apple-potato"), ("g-l2-slice-bread", "g-l2-slice-bread-tomato"),
         ("g-l2-slice-potato", "g-l2-slice-potato-tomato"), ("g-l2-slice-tomato", "g-l2-slice-tomato-lettuce")]
_M9 = C.load_manifest()
_T9 = set(_M9["test"]) | set(_M9.get("test_uncapped_not_scored") or [])
check("9.1 the 4 slice collisions are real: each short id is a TEST id, each long id a practice-pool id and a strict superstring",
      all(t in _T9 and o in set(_M9["practice_pool"]) and o not in _T9 and t in o for t, o in _COLL))


def _key9(iid, n=8, lang="en"):
    return "practice-v2|practice|o0|%d|%s|%s" % (n, iid, lang)


check("9.2 id_in: the 4 collisions do NOT match (a test id inside a longer practice id, in the real row-key form and bare)",
      not any(T.id_in(t, _key9(o)) or T.id_in(t, o) or T.id_in(t, "example/practice-kitchen/x:" + o) for t, o in _COLL))
check("9.3 id_in: real citations still match: '|' ':' '/' space quote start end, and case variants (stricter than before)",
      all(T.id_in(t, _key9(t)) for t, _o in _COLL) and T.id_in("g-l2-slice-bread", "g-l2-slice-bread")
      and T.id_in("g-l2-slice-bread", "see:g-l2-slice-bread/x") and T.id_in("g-l2-slice-bread", '"g-l2-slice-bread"')
      and T.id_in("g-l2-slice-bread", "x G-L2-SLICE-BREAD y") and not T.id_in("g-l2-slice-bread", "xg-l2-slice-bread")
      and not T.id_in("g-l2-slice-bread", "g-l2-slice-bread_2") and not T.id_in("", "anything"))
_pv9 = {"support": ["g-l2-slice-bread-tomato"], "evidence": [_key9("g-l2-slice-bread-tomato")]}
check("9.4 _cites: a8's provenance does not cite test g-l2-slice-bread (was 'evidence' under the substring rule); it does cite its own item",
      T._cites(_pv9, "g-l2-slice-bread") is None and T._cites(_pv9, "g-l2-slice-bread-tomato") == "support"
      and T._cites({"evidence": [_key9("g-l2-slice-bread-tomato")]}, "g-l2-slice-bread-tomato") == "evidence")
check("9.5 _cites: a real citation is still caught in each clause (support, tuned_on, evidence row key, evidence prose)",
      T._cites({"support": ["g-l2-slice-bread"]}, "g-l2-slice-bread") == "support"
      and T._cites({"tuned_on": ["g-l2-slice-bread"]}, "g-l2-slice-bread") == "tuned_on"
      and T._cites({"evidence": [_key9("g-l2-slice-bread")]}, "g-l2-slice-bread") == "evidence"
      and T._cites({"evidence": ["from g-l2-slice-bread, run 2"]}, "g-l2-slice-bread") == "evidence")


class _S9:
    def __init__(self, recs):
        self.recs = recs

    def candidates(self, *a, **k):
        return [{"rec": r} for r in self.recs]


try:
    T.ItemLeaveOut(_S9([{"id": "ex8", "provenance": _pv9}]), "g-l2-slice-bread", [], "example", fail=True).candidates({})
    _ok96 = True
except T.TreatmentError:
    _ok96 = False
try:
    T.ItemLeaveOut(_S9([{"id": "exT", "provenance": {"evidence": [_key9("g-l2-slice-bread")]}}]), "g-l2-slice-bread", [], "example",
                   fail=True).candidates({})
    _ok96b = False
except T.TreatmentError:
    _ok96b = True
check("9.6 ItemLeaveOut(fail=True) on test item g-l2-slice-bread: a8 passes (no integrity failure); a record really citing it still raises",
      _ok96 and _ok96b)


class _X9:
    def __init__(self, corpus):
        self.corpus = corpus


_c9 = [{"id": "example/practice-kitchen/practice-v2:o0:a8", "item": "g-l2-slice-bread-tomato", "provenance": _pv9},
       {"id": "example/practice-kitchen/practice-v2:o0:a9", "item": "g-l1-pickup-apple",
        "provenance": {"support": ["g-l1-pickup-apple"], "evidence": [_key9("g-l1-pickup-apple", 9)]}}]
_c9b = _c9 + [{"id": "example/p/x:o0:a3", "item": "g-l1-pickup-apple", "provenance": {"support": [], "evidence": [_key9("g-l2-slice-bread", 3)]}},
              {"id": "example/p/x:o0:a4", "item": "g-l2-slice-tomato", "provenance": {"support": ["g-l2-slice-tomato"], "evidence": []}}]
check("9.7 corpus_test_citations: the a8 pattern is clean; a planted evidence citation and a planted test item are both caught",
      T.corpus_test_citations(_X9(_c9), _T9) == []
      and T.corpus_test_citations(_X9(_c9b), _T9) == [{"id": "example/p/x:o0:a3", "items": ["g-l2-slice-bread"]},
                                                      {"id": "example/p/x:o0:a4", "items": ["g-l2-slice-tomato"]}],
      T.corpus_test_citations(_X9(_c9b), _T9))
try:
    import ltm_package as _LP9
    _sd9 = os.path.join(C.ROOT, "sim", "final_test", "snapshots_ft2", "o0", "k10")
    _rows9 = T.read_rows(_sd9)
    _cx9 = _LP9.default_sources(T.SCENE, os.path.join(_sd9, "ltm.db"), _rows9, allow_unapproved=False)["example"]
    _r97 = T.corpus_test_citations(_cx9, _T9)
except Exception as e:                                     # pragma: no cover
    _r97 = "error %s" % e
check("9.8 the REAL K=10 S* example corpus has no test citation under A31 (it had one false one, a8 -> g-l2-slice-bread, before)",
      _r97 == [], _r97)

print(f"\n_ftm_treatment_selftest: {N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
