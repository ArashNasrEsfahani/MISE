"""Selftest for sim/final_test/matrix/freeze.py (stage M3-freeze). Offline: synthetic repo root in a temp dir, fake token counter,
no simulator, no network, no Ollama, no API, no test item read.

    python sim/_ftm_freeze_selftest.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
sys.path.insert(0, os.path.join(HERE, "final_test"))

import common as C                                       # noqa: E402
import freeze as F                                       # noqa: E402
import reset_gate as RG                                  # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


REAL = C.ROOT
TMPS = []
MAN = C.load_manifest(REAL)                                   # A26 (K-B2b): the EFFECTIVE split (archived ids removed)
DEV, TESTI = MAN["dev_pilot"], MAN["test"]
BUDGET = {"L1": 400, "L2": 500, "L3": 600}


# ---------------------------------------------------------------------------------------------------- synthetic world
class FakeCounter:
    tokenizers = ["anthropic", "gemma"]

    def __init__(self, k_gemma=4.0, k_anth=3.0, tokenizers=None):
        self.k = (k_gemma, k_anth)
        if tokenizers is not None:
            self.tokenizers = tokenizers

    def counts(self, text):
        out = {"gemma": int(len(text) / self.k[0]) + 1}
        if "anthropic" in self.tokenizers:
            out["anthropic"] = int(len(text) / self.k[1]) + 1
        return out

    def save(self):
        pass


def w(root, rel, text, binary=False):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as f:
        f.write(text)
    return p


def jw(root, rel, obj):
    return w(root, rel, json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1))


def make_root(tag="a"):
    root = tempfile.mkdtemp(prefix=f"ftm_freeze_{tag}_")
    TMPS.append(root)
    shutil.copyfile(os.path.join(REAL, "docs", "final-test", "split_manifest.json"), _mk(root, "docs/final-test/split_manifest.json"))
    shutil.copyfile(os.path.join(REAL, "sim", "dataset", "general.json"), _mk(root, "sim/dataset/general.json"))
    inc = os.path.join(REAL, "docs", "final-test", "inclusion_manifest_a26.json")      # A26 (K-B2b): the successor's inclusion manifest
    if os.path.exists(inc):
        shutil.copyfile(inc, _mk(root, "docs/final-test/inclusion_manifest_a26.json"))
    w(root, "docs/final-test/preregistration.md", "# prereg (synthetic)\n")
    jw(root, "docs/final-test/families_approved.json", {"approved_by": "Arash", "date": "2026-10-11", "note": "synthetic"})
    jw(root, "sim/step5/params.json", {"version": 1, "retrieval": {"retention": "a12", "a12": {"B": BUDGET, "N": {"L1": 1, "L2": 2, "L3": 3}}}})
    types = [f"Type{i:02d}" for i in range(F.EXPECTED_TYPES)]
    w(root, "sim/weights/v3/FloorPlan1.yolo26s.pt", b"primary-weights", True)
    w(root, "sim/weights/v3/FloorPlan1.amb26m.pt", b"specialist-weights", True)
    jw(root, "docs/final-test/matrix/detector_types.json", {"weights_sha": hashlib.sha256(b"primary-weights").hexdigest(), "types": types})
    jw(root, "sim/step4/FloorPlan1.map.json", {"grid": 0.25})
    # code: tiny modules; capture imports common (sibling, `import common as C`) lazily + a relative import + a stdlib-named non-import
    w(root, "sim/final_test/matrix/common.py", "X = 1\n")
    w(root, "sim/final_test/matrix/capture.py", "import common as C\ndef f():\n    from planners import contract\n    import real_counter\n")
    w(root, "sim/final_test/matrix/plan.py", "from . import helper\nimport json\n")
    w(root, "sim/final_test/matrix/helper.py", "Y = 2\n")
    w(root, "sim/final_test/matrix/execute.py", "Z = 3\n")
    w(root, "sim/final_test/matrix/score.py", "W = 4\n")
    w(root, "sim/final_test/c3/real_counter.py", "R = 5\n")
    w(root, "sim/final_test/reset_gate.py", "G = 6\n")
    w(root, "sim/planners/contract.py", "PROMPT_VERSION = 'ft-v1'\n")
    w(root, "sim/planners/__init__.py", "")
    w(root, "sim/select.py", "SHADOW = 1\n")                  # a sim module named like a stdlib one: only counts when imported as a sibling
    # snapshots (nested: E_frozen grows with K)
    for o in F.SNAP_ORDERS:
        for k in F.SNAP_KS:
            d = f"sim/final_test/snapshots/o{o}/k{k}"
            jw(root, d + "/manifest.json", {"E_frozen": k, "state_hash": hashlib.sha256(f"{o}{k}".encode()).hexdigest(), "integrity": [],
                                            "file_sha256": "f" * 64})
            w(root, d + "/ltm.db", f"db{o}{k}", False)
            jw(root, d + "/fragments.json", {"o": o, "k": k})
    # tolerances: validated
    tol = {"schema": RG.TOL_SCHEMA, "amendment": "A14a", "status": "calibrated", "ceilings": {"pos_m": 0.1, "rot_deg": 15},
           "calibration": {"ok": True}, "tau_pos_m": 0.05, "tau_rot_deg": 5.0, "agent_pose": "exact",
           "validation": {"ok": True, "pairs": 24, "pass": 24}}
    tol["sha256"] = RG.self_sha(tol)
    jw(root, "docs/final-test/reset_tolerances.json", tol)
    # evidence gates
    for g in F.EVIDENCE_GATES:
        ev(root, g)
    # fs blocks
    fs = {lv: {"text": f"- earlier attempt record for {lv}", "real_tokens": 9} for lv in F.LEVELS}
    fs["source_sha"] = "a" * 64
    fs["sha256"] = C.self_sha(fs)
    jw(root, "docs/final-test/matrix/fs_blocks.json", fs)
    # rehearsal capture rows: every dev item in EN and FA with identical ids
    j = C.Journal(C.journal_path("rehearsal-v1", "capture", root))
    for n, it in enumerate(DEV):
        lv = ("L1", "L2", "L3")[n % 3]
        for lang in F.LANGS:
            j.append(cap_row(it, lang, lv, ids=[f"pl:{it}"]))
    return root


def _mk(root, rel):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def ev(root, g, **kw):
    d = {"schema": F.GATE_SCHEMA, "gate": g, "pass": True, "evidence": {"n": 12}, "source": "synthetic selftest stage", "created": "2026-10-11"}
    d.update(kw)
    jw(root, f"docs/final-test/matrix/gates/{g}.json", d)


def cap_row(it, lang, lv, ids, text=None, cov_ok=True):
    pres = C.pres_id(it, lang)
    text = text if text is not None else f"Where to look for {it}: Fridge (seen 2 times)"
    return {"key": f"k|{pres}", "namespace": "rehearsal-v1", "pres": pres, "item": it, "lang": lang, "level": lv, "world": "canon",
            "blocks": {"none": {"text": ""}, "S*": {"text": text, "ids": ids, "coverage": {"ok": cov_ok}}}}


PROBES = {"ollama": {"model": F.OLLAMA_MODEL, "digest": "sha256:b37049369adf" + "0" * 44, "version": "0.9.9", "parent_model": "x", "error": None},
          "codex": {"version": "codex-cli 0.162.1", "error": None}}


def write_ok(root, **kw):
    return F.write(root, counter=FakeCounter(), probes=PROBES, blocks_ns="rehearsal-v1", now="2026-10-11T12:00:00", **kw)


# ---------------------------------------------------------------------------------------------------- static
src = open(os.path.join(HERE, "final_test", "matrix", "freeze.py"), encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok310 = True
except SyntaxError as e:
    ok310 = False
check("freeze.py parses as python 3.10", ok310)
check("no key is read or printed by freeze.py (get_anthropic_key only through real_counter)", "get_anthropic_key" not in src and "reveal(" not in src and "environ" not in src)
check("freeze.py never reads the dataset content (hash only)", src.count("P[\"dataset\"]") == 1 and "json.load(open(P[\"dataset\"]" not in src)
check("spec: gate list covers 12.1-12.13 + G-iso, A12, A14, snapshot, leakage, tokens",
      set(F.GATE_ORDER) == {f"12.{i}" for i in range(1, 14)} | {"G-iso", "A12", "A14", "snapshot", "leakage", "tokens", "X-iso"})
check("spec: evidence-only gates are exactly those with no local computation", set(F.EVIDENCE_GATES) == {"12.2", "12.3", "12.5", "12.7", "12.10", "G-iso", "leakage", "X-iso"})

# ---------------------------------------------------------------------------------------------------- the green path
R = make_root("green")
fz = write_ok(R)
check("write: all gates pass on the synthetic world", fz["all_gates_pass"], [g for g, v in fz["gates"].items() if not v["pass"]])
check("write: schema + self sha", fz["schema"] == "final-test.freeze/1" and fz["sha256"] == C.self_sha(fz))
check("write: every hashed field present", all(fz.get(k) for k in ("prereg_sha256", "split_sha256", "dataset_sha256", "families_approved_sha256",
      "params_sha256", "system_text_sha", "system_template_sha", "user_template_sha", "plan_schema_sha", "map_sha", "reset_tolerances_sha256",
      "fs_blocks_sha256", "detector_weights", "snapshots", "code", "models", "ollama", "codex_cli_version", "caps", "reloc_seed", "blocks_sha")))
check("write: S* = o0k10 with its state hash", fz["snapshots"]["S*"]["state_hash"] == fz["snapshots"]["o0k10"]["state_hash"] and fz["snapshots"]["S*"]["source"] == "o0k10")
check("write: all 15 snapshots hashed", len([k for k in fz["snapshots"] if k != "S*"]) == 15)
check("write: split sha is the registered constant", fz["split_sha256"] == C.SPLIT_SHA256)
check("write: a12 values recorded", fz["a12"]["retention"] == "a12" and fz["a12"]["a12"]["B"] == BUDGET)
check("write: real tokens counted for both tokenizers and EN/FA", fz["tokens"]["ok"] and fz["tokens"]["tokenizers"] == ["anthropic", "gemma"] and fz["tokens"]["n_id_pairs"] == len(DEV))
check("write: FS blocks counted too", any(r["id"].startswith("fs_blocks.json") for r in json.load(open(F.tokens_rows_path(R)))["rows"]))
check("write: arash_check is null", fz["arash_check"] is None)
check("write: code closure follows relative, sibling and lazy imports", all(k in fz["code"] for k in
      ("sim/final_test/matrix/helper.py", "sim/final_test/matrix/common.py", "sim/planners/contract.py", "sim/final_test/c3/real_counter.py")))
check("write: a stdlib-named sim module is not followed from a plain import", "sim/select.py" not in fz["code"])
check("write: reloc seed is derived from the master seed and stable", fz["reloc_seed"] == C.rng("reloc_seed").randrange(1, 2 ** 31 - 1))
check("write: freeze.json holds no key-like text", not C.key_leak(open(C.paths(R)["freeze"], encoding="utf-8").read()))
check("write: namespaces / ceilings recorded", fz["namespaces"]["test-v1"] == {"core": 60.0, "total": 100.0})   # A37: core 60 / total 100 (A24.1-C 30 / 50; A24.10 30 / 100; A22 30 / 65)
check("A21: the freeze pins the cost-guard spec, the blocked-run rule, the dedup key spec with its golden key, the analysis plan, the power note and the X-arm configuration",
      fz["cost_guard"]["sha256"] == C.sha256_json(C.COST_GUARD_SPEC) and fz["blocked_run_rule"]["classes"] == C.RUN_CLASSES and fz["dedup"]["golden_ok"] is True
      and fz["dedup"]["golden_key"] == C.UNIT_KEY_GOLDEN and fz["analysis_plan"]["sha256"] and fz["analysis_plan"]["plan"]["holm"]["m"] == 3
      and fz["x_arms"]["model"] == "gpt-6-sol" and fz["x_arms"]["reasoning_effort"] == "high" and len(fz["x_arms"]["hardened_features"]) == 16
      and fz["x_arms"]["cli_flags_sha256"] and "X0" in fz["arms"] and "X+M" in fz["arms"] and "power_secondary_sha256" in fz)
check("A21: the X-iso gate is part of the freeze and passes on its (synthetic, wired) evidence file", fz["gates"]["X-iso"]["pass"])
import gate_evidence as GE                                  # noqa: E402
_xr = {"schema": "ftm.x-isolation/1", "model": "gpt-6-sol", "pass": True, "problems": [], "flags_sha": "abc", "n_calls": 9, "trials": 2, "canary_leaks": 0,
       "cli_version": "codex-cli 0.162.1", "canary_places": ["sibling_temp", "repo_docs", "repo_work"]}
check("A21 E: X-iso passes only on a complete, passing, current probe report", GE.x_iso_verdict(_xr, "abc") == (True, []))
check("A21 E: X-iso fails when the flag set changed since the probe", GE.x_iso_verdict(_xr, "different")[0] is False)
check("A21 E: X-iso fails on a leaked canary, a failing probe, too few calls, a wrong model, a missing report or a missing CLI version",
      not GE.x_iso_verdict(dict(_xr, canary_leaks=1), "abc")[0] and not GE.x_iso_verdict({**_xr, "pass": False, "problems": ["x"]}, "abc")[0]
      and not GE.x_iso_verdict(dict(_xr, n_calls=2), "abc")[0] and not GE.x_iso_verdict(dict(_xr, model="gpt-6-luna"), "abc")[0]
      and not GE.x_iso_verdict(None, "abc")[0] and not GE.x_iso_verdict(dict(_xr, cli_version=None), "abc")[0]
      and not GE.x_iso_verdict(dict(_xr, canary_places=["sibling_temp"]), "abc")[0])
r = F.check(R)                                          # stored-mode check: no counter, no live probe
check("check (before Arash): only 'Arash's freeze check not recorded' remains", r["reasons"] == ["Arash's freeze check not recorded"], r["reasons"])
check("freeze_status refuses before Arash's check", not C.freeze_status(R)["ok"])

# test items refused until the whole guard holds
t0 = TESTI[0]
check("guard: test item refused with no valid freeze", raises(lambda: C.guard_items([t0], "test-v1", allow_test=True, root=R), C.TestItemRefused))
check("record_arash_check refuses without his words", raises(lambda: F.record_arash_check(R, by="Arash", date="2026-10-12", said=" ", probes=PROBES), PermissionError))
check("record_arash_check refuses a different name", raises(lambda: F.record_arash_check(R, by="Claude", date="2026-10-12", said="go"), PermissionError))
out = F.record_arash_check(R, by="Arash", date="2026-10-12", said="freeze check is done, go", probes=PROBES)
check("record_arash_check fills and re-seals", out["arash_check"]["by"] == "Arash" and out["sha256"] == C.self_sha(out))
check("freeze_status ok after Arash's check", C.freeze_status(R)["ok"], C.freeze_status(R)["reasons"])
check("guard: test item still refused without --allow-test", raises(lambda: C.guard_items([t0], "test-v1", allow_test=False, root=R), C.TestItemRefused))
check("guard: test item refused outside test-v1", raises(lambda: C.guard_items([t0], "rehearsal-v1", allow_test=True, root=R), C.TestItemRefused))
check("guard: test item passes with every condition", C.guard_items([t0], "test-v1", allow_test=True, root=R)["n_test"] == 1)
check("check (after Arash): ok", F.check(R)["ok"], F.check(R))
check("record_arash_check twice refused", raises(lambda: F.record_arash_check(R, by="Arash", date="2026-10-13", said="again", probes=PROBES), PermissionError))
check("write refuses to overwrite a checked freeze", raises(lambda: write_ok(R), PermissionError))
fz2 = write_ok(R, supersede=True)
check("supersede: re-freeze allowed, old file archived, check reset", fz2["arash_check"] is None and os.path.isdir(os.path.join(R, "sim", "final_test", "_checkpoints", "M3-freeze")))
check("supersede: same inputs give the same reloc seed", fz2["reloc_seed"] == fz["reloc_seed"])
check("write: test blocks refused by --write", raises(lambda: F.write(R, counter=FakeCounter(), probes=PROBES, blocks_ns="test-v1"), C.TestItemRefused))
check("check: test blocks refused without --allow-test", F.check(R, blocks_ns="test-v1")["ok"] is False)
check("check: test blocks refused even with --allow-test while Arash's check is missing",
      any("refused" in x for x in F.check(R, blocks_ns="test-v1", allow_test=True)["reasons"]))
F.record_arash_check(R, by="Arash", date="2026-10-12", said="ok", probes=PROBES)

# ---------------------------------------------------------------------------------------------------- tamper after the freeze
def fresh_checked(tag):
    root = make_root(tag)
    write_ok(root)
    F.record_arash_check(root, by="Arash", date="2026-10-12", said="ok", probes=PROBES)
    return root


T = fresh_checked("t1")
w(T, "sim/final_test/matrix/execute.py", "Z = 33\n")
rr = F.check(T)
check("tamper: a code edit is found", not rr["ok"] and "code" in rr["diff"])
check("tamper: freeze_status names the changed file", any("execute.py" in x for x in C.freeze_status(T)["reasons"]))
check("tamper: guard refuses test items after a code edit", raises(lambda: C.guard_items([t0], "test-v1", allow_test=True, root=T), C.TestItemRefused))
T = fresh_checked("t2")
jw(T, "sim/step5/params.json", {"version": 1, "retrieval": {"retention": "a12", "a12": {"B": {"L1": 999, "L2": 500, "L3": 600}}}})
check("tamper: params edit is found", "params_sha256" in F.check(T)["diff"])
T = fresh_checked("t3")
ev(T, "12.3", evidence={"n": 13})
check("tamper: gate evidence edit is found", "evidence_sha" in F.check(T)["diff"] or "gates" in F.check(T)["diff"])
T = fresh_checked("t4")
w(T, "docs/final-test/preregistration.md", "# changed\n")
check("tamper: prereg edit refused by freeze_status", any("preregistration" in x for x in C.freeze_status(T)["reasons"]))
T = fresh_checked("t5")
w(T, "sim/weights/v3/FloorPlan1.amb26m.pt", b"other", True)
check("tamper: weights edit is found", "detector_weights" in F.check(T)["diff"])
T = fresh_checked("t6")
w(T, "sim/final_test/snapshots/o0/k10/ltm.db", "changed")
check("tamper: S* edit is found", "snapshots" in F.check(T)["diff"])
T = fresh_checked("t7")
p = C.paths(T)["freeze"]
d = json.load(open(p, encoding="utf-8"))
d["caps"]["max_steps"] = 99
json.dump(d, open(p, "w", encoding="utf-8"))
check("tamper: hand-edited freeze.json fails the self sha", any("self-sha" in x for x in F.check(T)["reasons"]) and not C.freeze_status(T)["ok"])
T = fresh_checked("t8")
d = json.load(open(C.paths(T)["freeze"], encoding="utf-8"))
d["gates"]["12.3"]["pass"] = False
d["sha256"] = C.self_sha(d)
json.dump(d, open(C.paths(T)["freeze"], "w", encoding="utf-8"))
check("tamper: a gate flipped to false and re-hashed is refused by freeze_status", not C.freeze_status(T)["ok"])
T = fresh_checked("t9")
d = json.load(open(C.paths(T)["freeze"], encoding="utf-8"))
d["gates"]["12.3"]["pass"] = "yes"
d["sha256"] = C.self_sha(d)
json.dump(d, open(C.paths(T)["freeze"], "w", encoding="utf-8"))
check("freeze_status: a non-boolean gate pass is not a pass", not C.freeze_status(T)["ok"])
for d_ in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"):
    pass

# ---------------------------------------------------------------------------------------------------- individual gates fail loudly
def gates_of(root, **kw):
    kw.setdefault("counter", FakeCounter())
    kw.setdefault("probes", PROBES)
    kw.setdefault("blocks_ns", "rehearsal-v1")
    return F.build(root, **kw)["gates"]


G = make_root("g1")
os.remove(os.path.join(F.matrix_gates_dir(G), "12.2.json"))
check("gate 12.2 missing evidence fails", not gates_of(G)["12.2"]["pass"])
ev(G, "12.2", synthetic=True)
check("gate evidence flagged synthetic is refused", not gates_of(G)["12.2"]["pass"])
ev(G, "12.2", **{"pass": "true"})
check("gate evidence with pass 'true' (string) is refused", not gates_of(G)["12.2"]["pass"])
ev(G, "12.2", gate="12.3")
check("gate evidence for another gate id is refused", not gates_of(G)["12.2"]["pass"])
ev(G, "12.2", evidence={})
check("gate evidence with no evidence is refused", not gates_of(G)["12.2"]["pass"])
ev(G, "12.2")
w(G, "docs/final-test/matrix/gates/12.2.json", json.dumps(
    {"schema": F.GATE_SCHEMA, "gate": "12.2", "pass": True, "evidence": {"k": "sk-ant-api03-" + "A" * 40}}))
check("gate evidence containing a key-like string is refused", not gates_of(G)["12.2"]["pass"])
ev(G, "12.2")
check("12.6 needs snapshot + G-iso + leakage", gates_of(G)["12.6"]["pass"])
os.remove(os.path.join(F.matrix_gates_dir(G), "leakage.json"))
gg = gates_of(G)
check("12.6 fails when leakage evidence is missing", not gg["12.6"]["pass"] and not gg["leakage"]["pass"])
ev(G, "leakage")

# 12.4 / A14
tol = json.load(open(C.paths(G)["tolerances"], encoding="utf-8"))
for name, mod in (("unset status", {"status": "unset"}), ("failed status", {"status": "failed"}), ("23/24 pairs", {"validation": {"ok": True, "pairs": 24, "pass": 23}}),
                  ("no validation", {"validation": None})):
    t = dict(tol, **mod)
    t["sha256"] = RG.self_sha(t)
    jw(G, "docs/final-test/reset_tolerances.json", t)
    gg = gates_of(G)
    check(f"12.4 and A14 fail on {name}", not gg["12.4"]["pass"] and not gg["A14"]["pass"])
t = dict(tol, tau_pos_m=0.5)                                  # edited by hand, sha not refreshed
jw(G, "docs/final-test/reset_tolerances.json", t)
check("12.4 fails on a hand-edited tolerance file", not gates_of(G)["12.4"]["pass"])
jw(G, "docs/final-test/reset_tolerances.json", tol)
check("12.4 passes again on the intact file", gates_of(G)["12.4"]["pass"])

# 12.8
jw(G, "docs/final-test/families_approved.json", {"approved_by": "Claude", "date": "2026-10-11"})
check("12.8 fails unless Arash approved", not gates_of(G)["12.8"]["pass"])
jw(G, "docs/final-test/families_approved.json", {"approved_by": "Arash", "date": "2026-10-11"})

# 12.9
bad = dict(PROBES, ollama=dict(PROBES["ollama"], digest="sha256:" + "1" * 64))
check("12.9 fails on a different Ollama digest", not gates_of(G, probes=bad)["12.9"]["pass"])
bad = dict(PROBES, ollama=dict(PROBES["ollama"], digest=None, error="URLError"))
check("12.9 fails when Ollama is unreachable", not gates_of(G, probes=bad)["12.9"]["pass"])
bad = dict(PROBES, codex={"version": None, "error": "FileNotFoundError"})
check("12.9 fails without a codex CLI version", not gates_of(G, probes=bad)["12.9"]["pass"])

# 12.12
jw(G, "docs/final-test/matrix/detector_types.json", {"weights_sha": hashlib.sha256(b"primary-weights").hexdigest(), "types": [f"T{i}" for i in range(48)]})
check("12.12 fails with 48 types", not gates_of(G)["12.12"]["pass"])
jw(G, "docs/final-test/matrix/detector_types.json", {"weights_sha": "0" * 64, "types": [f"T{i}" for i in range(49)]})
check("12.12 fails when the type file names other weights", not gates_of(G)["12.12"]["pass"])
jw(G, "docs/final-test/matrix/detector_types.json", {"weights_sha": hashlib.sha256(b"primary-weights").hexdigest(), "types": [f"Type{i:02d}" for i in range(49)]})

# 12.13
w(G, "sim/final_test/matrix/work/test-v1/cache/ab/x.json", "{}")
check("12.13 fails when the test cache is not empty", not gates_of(G)["12.13"]["pass"])
shutil.rmtree(os.path.join(G, "sim", "final_test", "matrix", "work"))

# 12.1
os.remove(os.path.join(G, "sim", "final_test", "matrix", "score.py"))
gg = gates_of(G)
check("12.1 fails when an entry module is missing", not gg["12.1"]["pass"] and "sim/final_test/matrix/score.py" in gg["12.1"]["evidence"]["missing_entries"])
w(G, "sim/final_test/matrix/score.py", "W = 4\n")
w(G, "sim/final_test/matrix/score.py", "def (:\n")
gg = gates_of(G)
check("12.1 fails on a module that does not parse", not gg["12.1"]["pass"] and gg["12.1"]["evidence"]["parse_errors"])
w(G, "sim/final_test/matrix/score.py", "W = 4\n")
check("12.1 passes again", gates_of(G)["12.1"]["pass"])
d = json.load(open(os.path.join(G, "docs", "final-test", "split_manifest.json"), encoding="utf-8"))
shutil.copyfile(os.path.join(REAL, "sim", "dataset", "general.json"), os.path.join(G, "sim", "dataset", "general.json"))
w(G, "sim/dataset/general.json", open(os.path.join(G, "sim", "dataset", "general.json"), encoding="utf-8").read() + " ")
check("12.1 fails when the dataset differs from the one the split was cut from", not gates_of(G)["12.1"]["pass"])
shutil.copyfile(os.path.join(REAL, "sim", "dataset", "general.json"), os.path.join(G, "sim", "dataset", "general.json"))

# 12.11
w(G, "sim/final_test/matrix/execute.py", 'H = "Long-term memory (retrieved for this command; learned"\n')
check("12.11 fails when a frozen module re-declares the memory header", not gates_of(G)["12.11"]["pass"])
w(G, "sim/final_test/matrix/execute.py", "Z = 3\n")

# snapshot
w(G, "sim/final_test/snapshots/o1/k3/ltm.db", "x")
os.remove(os.path.join(G, "sim", "final_test", "snapshots", "o1", "k3", "ltm.db"))
check("snapshot gate fails on a missing snapshot file", not gates_of(G)["snapshot"]["pass"])
w(G, "sim/final_test/snapshots/o1/k3/ltm.db", "db13")
m = json.load(open(os.path.join(G, "sim/final_test/snapshots/o1/k5/manifest.json")))
jw(G, "sim/final_test/snapshots/o1/k5/manifest.json", dict(m, E_frozen=1))
check("snapshot gate fails when snapshots are not nested", not gates_of(G)["snapshot"]["pass"])
jw(G, "sim/final_test/snapshots/o1/k5/manifest.json", m)
jw(G, "sim/final_test/snapshots/o0/k10/manifest.json", dict(json.load(open(os.path.join(G, "sim/final_test/snapshots/o0/k10/manifest.json"))), integrity=["x"]))
check("snapshot gate fails when a manifest lists integrity problems", not gates_of(G)["snapshot"]["pass"])
jw(G, "sim/final_test/snapshots/o0/k10/manifest.json", dict(m, E_frozen=10, state_hash="a" * 64, integrity=[]))

# ---------------------------------------------------------------------------------------------------- the real-token gate
rows = F.block_rows([cap_row(DEV[0], "en", "L1", ["a"]), cap_row(DEV[0], "fa", "L1", ["a"])])
ok_tg = F.token_gate(rows, FakeCounter(), BUDGET)
check("tokens: ok when each complete block fits both tokenizers", ok_tg["ok"] and ok_tg["n_blocks"] == 2 and ok_tg["n_id_pairs"] == 1, ok_tg["reasons"])
big = F.block_rows([cap_row(DEV[0], "en", "L1", ["a"], text="x" * 3000), cap_row(DEV[0], "fa", "L1", ["a"])])
tg = F.token_gate(big, FakeCounter(), BUDGET)
check("tokens: a block over the budget in one tokenizer fails", not tg["ok"] and any("exceed" in x for x in tg["reasons"]))
# a text that fits Gemma but not Anthropic
txt = "y" * 1250                                                    # gemma 313 tokens, anthropic 417 with the header: > 400 only for anthropic
tg = F.token_gate(F.block_rows([cap_row(DEV[0], "en", "L1", ["a"], text=txt)]), FakeCounter(k_gemma=4.0, k_anth=2.0), BUDGET)
check("tokens: the MAX over tokenizers decides", not tg["ok"] and tg["rows"][0]["counts"]["gemma"] <= 400 < tg["rows"][0]["counts"]["anthropic"], tg["rows"])
tg = F.token_gate(rows, FakeCounter(tokenizers=["gemma"]), BUDGET)
check("tokens: a Gemma-only counter is refused", not tg["ok"] and any("both tokenizers" in x for x in tg["reasons"]))
tg = F.token_gate(F.block_rows([cap_row(DEV[0], "en", "L1", ["a"]), cap_row(DEV[0], "fa", "L1", ["b"])]), FakeCounter(), BUDGET)
check("tokens: EN/FA record ids that differ fail", not tg["ok"] and any("differ" in x for x in tg["reasons"]))
check("tokens: no blocks at all is a failure, not a pass", not F.token_gate([], FakeCounter(), BUDGET)["ok"])
check("tokens: no A12 budgets is a failure", not F.token_gate(rows, FakeCounter(), None)["ok"])
tg = F.token_gate(F.block_rows([cap_row(DEV[0], "en", "L1", ["a"], text=""), cap_row(DEV[0], "fa", "L1", ["a"], text="")]), FakeCounter(), BUDGET)
check("tokens: only empty blocks is still a failure (nothing was measured)", not tg["ok"] and tg["n_empty"] == 2)
t_big = "z" * 1800                                            # ~600 gemma / 450 anth tokens + header: over L1 (400) and L2 (500)... fits L3 (600) only in anthropic terms
check("tokens: the budget is per level", not F.token_gate(F.block_rows([cap_row(DEV[0], "en", "L1", ["a"], text=t_big)]), FakeCounter(), BUDGET)["ok"] and
      F.token_gate(F.block_rows([cap_row(DEV[0], "en", "L3", ["a"], text="z" * 1500)]), FakeCounter(), BUDGET)["ok"])
g = gates_of(G, counter=None)
check("tokens: no counter given fails the gate (never skipped)", not g["tokens"]["pass"])
w(G, "docs/final-test/matrix/rehearsal-v1/capture.jsonl", "")
check("A12: no S* blocks fails", not gates_of(G)["A12"]["pass"])

# A12 coverage from the capture rows
G2 = make_root("g2")
j = C.Journal(C.journal_path("rehearsal-v1", "capture", G2))
os.remove(j.path)
for n, it in enumerate(DEV):
    for lang in F.LANGS:
        j.append(cap_row(it, lang, "L1", ["a"], cov_ok=(n != 3)))
gg = gates_of(G2)
check("A12 fails when one S* block fails the coverage check", not gg["A12"]["pass"] and len(gg["A12"]["evidence"]["failing"]) == 2)
jw(G2, "sim/step5/params.json", {"version": 1, "retrieval": {"retention": "c2e"}})
check("A12 fails when the registry is not on a12", not gates_of(G2)["A12"]["pass"])

# CachedCounter: the same on-disk format as RealMaxCounter; a miss is an error, never an estimate
cache = {}
def put(t, g_, a_):
    cache[hashlib.sha256(t.encode()).hexdigest()[:24]] = {"gemma_raw": g_, "anth_raw": a_}
put(".", 1, 1)
cp = os.path.join(G2, "tok.json")
json.dump(cache, open(cp, "w"))
cc = F.CachedCounter(cp)
check("CachedCounter: a miss raises KeyError", raises(lambda: cc.counts("never counted"), KeyError))
put("hello", 11, 16)
json.dump(cache, open(cp, "w"))
check("CachedCounter: hit = raw - baseline for each tokenizer", F.CachedCounter(cp).counts("hello") == {"gemma": 10, "anthropic": 15})
check("CachedCounter: whitespace-only text is one token (as RealMaxCounter)", F.CachedCounter(cp).counts("  ") == {"gemma": 1, "anthropic": 1})
tg = F.token_gate(rows, F.CachedCounter(cp), BUDGET)
check("tokens: an uncached block fails the gate with a reason", not tg["ok"] and any("not in the token cache" in x for x in tg["reasons"]))

# ---------------------------------------------------------------------------------------------------- import closure edge cases
E = tempfile.mkdtemp(prefix="ftm_closure_")
w(E, "sim/a.py", "import b\nimport os, json\nfrom pkg import mod\nfrom pkg.sub import thing\n")
w(E, "sim/b.py", "def f():\n    import c\n")
w(E, "sim/c.py", "from . import d\n")
w(E, "sim/d.py", "")
w(E, "sim/pkg/__init__.py", "")
w(E, "sim/pkg/mod.py", "")
w(E, "sim/pkg/sub/__init__.py", "")
w(E, "sim/pkg/sub/thing.py", "")
w(E, "sim/unused.py", "")
w(E, "sim/outside.py", "")
files, miss, errs = F.import_closure(E, ["sim/a.py", "sim/nope.py"])
check("closure: transitive, lazy, relative, package and from-package imports", set(files) == {"sim/a.py", "sim/b.py", "sim/c.py", "sim/d.py", "sim/pkg/__init__.py", "sim/pkg/mod.py", "sim/pkg/sub/__init__.py", "sim/pkg/sub/thing.py"}, files)
check("closure: missing entry reported, unused module not included", miss == ["sim/nope.py"] and "sim/unused.py" not in files)
w(E, "sim/os.py", "")
w(E, "sim/pkg/user.py", "import os\n")
files, _, _ = F.import_closure(E, ["sim/pkg/user.py"])
check("closure: stdlib name is not shadowed by a non-sibling sim module", "sim/os.py" not in files)

# ---------------------------------------------------------------------------------------------------- C9: key scan of freeze.json
# The key pattern used to match inside English words ("task-sampling" in the A24.2 cost text of the analysis plan), so the green path could
# not write. The fix (planners.keys._sk_sub) narrows only that prose shape; the checks below hold the refusal path and every key format.
from planners.keys import redact, Secret                 # noqa: E402

_plan_txt = json.dumps(fz["analysis_plan"]["plan"], ensure_ascii=False)
check("C9: the green freeze embeds the real analysis plan with its registered hyphenated prose (task-sampling) and still writes",
      "task-sampling" in _plan_txt and os.path.isfile(C.paths(R)["freeze"]))
check("C9: hyphenated English words ending in 'sk-' are not key-like",
      not any(C.key_leak(t) for t in ("CIs describe task-sampling uncertainty", "a task-specific plan failed", "risk-adjusted", "desk-lamp-holder")))
_K = ("sk-ant-api03-AbCd_ef0123456789xyz", "sk-or-v1-0123456789abcdef0123", "sk-proj-Ab12Cd34Ef56Gh78", "sk-ant-FAKEKEY0123456789abcdefSELFTEST")
_ctx = ("{k}", "api_key={k}", '"{k}"', "x {k}", "x_{k}", "abc{k}", "9{k}", "-{k}")
check("C9: every key format is still key-like at the start, after '=', '\"', a space, '_', a letter, a digit and '-'",
      all(C.key_leak(c.format(k=k)) for k in _K for c in _ctx))
check("C9: a key after a JSON-escaped newline / tab / \\uXXXX (json.dumps) is still key-like, also an all-lowercase one",
      all(C.key_leak(json.dumps(p + k)) for k in _K + ("sk-abcdefghijkl",) for p in ("line\n", "a\t", "b\r", "ê", "ب")))
check("C9: an all-lowercase sk- token is key-like unless glued to a preceding letter (the one disclosed narrowing)",
      C.key_leak("my sk-ant-ccccccccccc here") and C.key_leak("sk-abcdefghij") and not C.key_leak("xsk-abcdefghij"))
check("C9: the exact loaded secret is redacted even glued to letters / digits and without any key shape",
      redact("ab9plainsecret99zz", extra=[Secret("plainsecret99")]) == "ab9***zz" and C.key_leak("abplainsecret99", (Secret("plainsecret99"),)))

_real_build = F.build


def _inject(where):
    def b(*a, **kw):
        out = _real_build(*a, **kw)
        out["caps"] = dict(out.get("caps") or {}, note=where + "sk-ant-api03-Injected0123456789")
        return out
    return b


KR = make_root("keyinject")
F.build = _inject("x ")
try:
    _e1 = None
    try:
        write_ok(KR)
    except RuntimeError as ex:
        _e1 = str(ex)
    check("C9: a key-shaped value in a fresh freeze is refused with the key-like reason and no freeze.json is created",
          _e1 == "key-like text in freeze.json; refusing to write" and not os.path.exists(C.paths(KR)["freeze"]), _e1)
finally:
    F.build = _real_build
write_ok(KR)
_before = open(C.paths(KR)["freeze"], "rb").read()
F.build = _inject("prose task-sampling then\n")
try:
    _e2 = None
    try:
        write_ok(KR)
    except RuntimeError as ex:
        _e2 = str(ex)
    check("C9: a key after an escaped newline is refused on a re-write and the existing freeze.json is not replaced",
          _e2 == "key-like text in freeze.json; refusing to write" and open(C.paths(KR)["freeze"], "rb").read() == _before, _e2)
finally:
    F.build = _real_build

# ---------------------------------------------------------------------------------------------------- CLI + probes
check("probe_ollama never raises on a closed port", F.probe_ollama("http://127.0.0.1:9", timeout=1)["error"] is not None)
check("probe_codex never raises", isinstance(F.probe_codex(timeout=2), dict))
check("CLI parser exposes --write --check --record-arash-check --allow-test", all(x in src for x in ("--write", "--check", "--record-arash-check", "--allow-test")))
M = make_root("cli")
check("check on a missing freeze.json fails cleanly", F.check(M)["ok"] is False and "missing" in F.check(M)["reasons"][0])

# ---------------------------------------------------------------------------------------------------- TREAT-FREEZE (2026-10-13)
# A24.1-T item 4: after the one-constant flip, the freeze records the test-v1 treatment identity; --record-treatment writes ONLY that record
# into an existing (stale) freeze and marks it treatment-only (Astra TREAT-FREEZE 4: never a validated freeze).
import treatment as TT                                   # noqa: E402
TR = fresh_checked("treat0")                              # a freeze WITH Arash's check
TR1 = make_root("treat1")
write_ok(TR1)
TMPS += [TR, TR1]
fz_t = json.load(open(C.paths(TR1)["freeze"], encoding="utf-8"))
check("TREAT-FREEZE: after the flip --write records treatment = treatment.identity('test-v1', root) and no freeze_status",
      TT.TEST_V1_TREATMENT == "ft-v2" and fz_t.get("treatment") == TT.identity("test-v1", TR1) and "freeze_status" not in fz_t,
      sorted(fz_t.get("treatment") or {}))
check("TREAT-FREEZE: --record-treatment refuses a freeze that carries Arash's check", raises(lambda: F.record_treatment(TR), PermissionError))
out_t = F.record_treatment(TR1)
_arc = os.path.join(TR1, "sim", "final_test", "_checkpoints", "M3-freeze")
check("TREAT-FREEZE: --record-treatment writes the identity + freeze_status treatment-only, re-seals, archives the old file",
      out_t["treatment"]["sha"] == TT.identity("test-v1", TR1)["sha"] and out_t["freeze_status"] == TT.TREATMENT_ONLY
      and out_t["sha256"] == C.self_sha(out_t) and any(f.endswith(".pre-treatment.json") for f in os.listdir(_arc)))
rt_ = F.check(TR1)
check("TREAT-FREEZE: a treatment-only freeze is never clean: --check reports freeze_status, common.freeze_status and entry_problems refuse it",
      not rt_["ok"] and "freeze_status" in rt_["diff"] and not C.freeze_status(TR1, check_code=False)["ok"]
      and TT.entry_problems("test-v1", out_t, out_t["treatment"]["sha"]) != [], rt_["reasons"][:2])
_sv_t = TT.TEST_V1_TREATMENT
TT.TEST_V1_TREATMENT = "ft-v1"
try:
    _ref_t = raises(lambda: F.record_treatment(TR1), RuntimeError)
    _b_t = F.build(TR1, counter=FakeCounter(), probes=PROBES, blocks_ns="rehearsal-v1", now="2026-10-11T12:00:00")
finally:
    TT.TEST_V1_TREATMENT = _sv_t
check("TREAT-FREEZE: before the flip --record-treatment refuses and build records no treatment", _ref_t and "treatment" not in _b_t)
check("CLI parser exposes --record-treatment", "--record-treatment" in src)

for d_ in TMPS + [E]:
    shutil.rmtree(d_, ignore_errors=True)
print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
