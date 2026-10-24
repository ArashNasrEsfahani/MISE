"""Selftest for sim/final_test/matrix/plan.py (stage M5-plan). Offline: no simulator, no network, no real API call, no Ollama.

    python sim/_ftm_plan_selftest.py

Everything runs against a temporary root with SYNTHETIC captures (dev-item ids and made-up instruction text; no dataset text, no
test-item text) and FAKE transports plugged into the real provider classes, so the real response cache, spend ledger, runner,
validator and repair logic are exercised. A test-item id appears only as a bare string in the guard-refusal checks.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from planners import prompt_sha256

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                         # noqa: E402
import plan as PL                                          # noqa: E402
import final_test_callplan as CP                           # noqa: E402
import honest_skills as hs                                 # noqa: E402
from planners import PlannerContract, make_planner         # noqa: E402
from planners.base import CostCeilingExceeded, PlannerSkip  # noqa: E402
from planners.keys import Secret                           # noqa: E402

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


FAKEKEY = "sk-ant-FAKEKEY0123456789abcdefSELFTEST"
SYN = "synthetic-v1"
REAL_NS = dict(C.NAMESPACES)
MANIFEST = C.load_manifest()
DEV = list(MANIFEST["dev_pilot"])
#: A26 (K-B2b 2026-10-13): canonical dev presentations = 2 x the EFFECTIVE dev-pilot items (24 -> 20 after the archive)
NP = 2 * len(DEV)
TYPES = ["Apple", "Bread", "Fridge", "Knife", "Mug", "Plate", "Sink"]
REPEAT_ITEMS, RELOC_ITEMS = DEV[:2], DEV[:2]


def set_ceilings(core, total):
    # A23: the synthetic namespace EXPLICITLY opts into the archived Sonnet / Haiku arms so the archived code paths (haiku / sonnet passes, 12-arm
    # matrix, S/H cost scopes) stay exercised; test-v1 never does (see the A23 block below and _ftm_common_selftest).
    C.NAMESPACES[SYN] = {"items": "dev_pilot", "ceilings": {"core": core, "total": total}, "archived_arms": C.ARCHIVED_ARM_NAMES}


set_ceilings(50.0, 100.0)


# ================================================================================================== synthetic world
def contract():
    return PlannerContract(TYPES, tools=hs.TOOLS)


def idx_of(item):
    return DEV.index(item) if item in DEV else 20 + sum(map(ord, item)) % 50


def instruction(item, lang):
    i = idx_of(item)
    return f"Pick up the apple number {i}." if lang == "en" else f"سیب شماره {i} را بردار."


def block_text(src, item, lang, empty=False):
    if empty or src == "none":
        return ""
    k = {"S*": 10, "FS": 10}.get(src) or int(src.split("k")[1])
    lines = [f"Where to look for Apple: Fridge ({lang}, {item[-6:]})"] + [f"record {j}" for j in range(min(k, 3))]
    if src == "FS":
        lines = [f"attempt log {j} ({lang})" for j in range(3)]
    return "\n".join(lines)


def build_capture(root, ns=SYN, items=None, breaker=None, repeat_items=REPEAT_ITEMS, reloc_items=RELOC_ITEMS, complete=True,
                  empty_items=()):
    """Writes capture.jsonl (+ the phase_complete event) under `root`. Requests are computed with the contract directly."""
    items = DEV if items is None else items
    ct = contract()
    J = C.Journal(C.journal_path(ns, "capture", root))
    if os.path.exists(J.path):
        os.remove(J.path)
    curve = [f"o{o}k{k}" for o, k in PL.CURVE_SNAPSHOTS]

    def one(item, lang, world):
        srcs = ["none", "S*", "FS"] + (curve if (world == "canon" and item in repeat_items) else [])
        if world == "reloc":
            srcs = ["none", "S*"]
        rec = {"instruction": instruction(item, lang), "summary": [f"Fridge [closed] at ({world})", "Apple on CounterTop"],
               "holding": None}
        blocks, reqs = {}, {}
        for s in srcs:
            text = block_text(s, item, lang, empty=(item in empty_items and s != "none"))
            blocks[s] = {"text": text, "ids": [], "real_tokens": len(text), "budget": 150, "empty": not text, "coverage": {}}
            if s == "none":
                m = ct.messages(rec["instruction"], rec["summary"], None)
            elif s == "FS":
                m = ct.messages(rec["instruction"], rec["summary"], None, fewshot_block=text or None)
            else:
                m = ct.messages(rec["instruction"], rec["summary"], None, memory_block=text or None)
            reqs[s] = hashlib.sha256(json.dumps([[x["role"], x["content"]] for x in m], ensure_ascii=False,
                                                separators=(",", ":")).encode("utf-8")).hexdigest()
        pres = C.pres_id(item, lang, world)
        row = C.make_record("capture", key=C.cell_id(ns, "capture", pres, "c1"), namespace=ns, pres=pres, item=item, lang=lang,
                            level=f"L{1 + idx_of(item) % 3}", world=world, capture_run=1, instruction=rec["instruction"],
                            holding=None, summary=rec["summary"], summary_sha=C.sha256_text("\n".join(rec["summary"])),
                            blocks=blocks, requests=reqs, state_ref=f"states/{pres}.c1.json", world_hash="h", audit={})
        if breaker:
            row = breaker(row)
        J.append(row)

    for it in items:
        for lg in ("en", "fa"):
            one(it, lg, "canon")
    for it in reloc_items:
        for lg in ("en", "fa"):
            one(it, lg, "reloc")
    if complete:
        C.Journal(C.journal_path(ns, "events", root)).append(
            C.make_record("events", event="phase_complete", phase="capture", n=len(J.rows()), journal_sha=C.sha256_file(J.path)))
    return J


# ================================================================================================== fake transports
class Crash(BaseException):
    """Not an Exception: the transport layer does not mistake it for a connection error, so it really unwinds."""


class Fake:
    def __init__(self):
        self.calls = []
        self.fail_idx = set()           # item indices whose Anthropic call fails (HTTP 400 -> infra)
        self.fail_models = ()
        self.bad_first = set()          # first reply invalid, repair valid
        self.reject = set()             # always invalid
        self.opus_out = 40              # output tokens billed per Opus call
        self.crash_on_call = None
        self.err500 = set()
        self.tool_idx = set()           # Codex tool events
        self.n = 0

    @staticmethod
    def _idx(text):
        for tok in text.replace(".", " ").split():
            if tok.isdigit():
                return int(tok)
        for tok in text.split():
            t = tok.strip("،.")
            if t.isdigit():
                return int(t)
        return -1

    def reply(self, idx, repair):
        if idx in self.reject or (idx in self.bad_first and not repair):
            return "this is not json"
        return json.dumps({"say": "ok", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]})

    def _tick(self, who, idx):
        self.n += 1
        self.calls.append((who, idx))
        if self.crash_on_call is not None and self.n == self.crash_on_call:
            self.crash_on_call = None
            raise Crash("synthetic crash mid-cell")

    def ollama(self, url, headers, body, timeout):
        user = body["messages"][-1]["content"]
        idx = self._idx(user.split("Command:")[-1])
        repair = len(body["messages"]) > 2
        self._tick("gemma", idx)
        return 200, {}, {"message": {"content": self.reply(idx, repair)}, "done_reason": "stop", "prompt_eval_count": 900,
                         "eval_count": 30}

    def anth(self, url, headers, body, timeout):
        assert headers["x-api-key"] == FAKEKEY
        msgs = body["messages"]
        idx = self._idx(msgs[0]["content"].split("Command:")[-1])
        repair = len(msgs) > 1
        model = body["model"]
        self._tick(model, idx)
        if self.n in self.err500:
            return 500, {}, {"error": "synthetic server error"}
        if idx in self.fail_idx and (not self.fail_models or any(m in model for m in self.fail_models)):
            return 400, {}, {"error": "synthetic bad request"}
        out = self.opus_out if "opus" in model else 30
        return 200, {}, {"content": [{"type": "text", "text": self.reply(idx, repair)}], "usage": {"input_tokens": 1200,
                         "output_tokens": out}, "stop_reason": "end_turn", "model": model, "id": f"msg_{self.n}"}

    def codex(self, cmd, prompt, cwd, env, timeout):
        if "--version" in cmd:
            return types.SimpleNamespace(stdout="codex-cli 9.9.9\n", stderr="", returncode=0)
        idx = self._idx(prompt.split("Command:")[-1])
        self._tick("codex", idx)
        ev = [{"type": "item.completed", "item": {"type": "agent_message", "text": self.reply(idx, False)}},
              {"type": "turn.completed", "usage": {"input_tokens": 14000, "output_tokens": 60}}]
        if idx in self.tool_idx:
            ev.insert(0, {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}})
        model = cmd[cmd.index("-m") + 1]
        model = getattr(self, "codex_model_override", None) or model
        sess = ('2026-10-12T00:00:00Z  INFO codex.exec{otel.kind="internal"}: codex_exec: Codex initialized with event: SessionConfiguredEvent { session_id: SessionId { uuid: u }, '
                'thread_id: ThreadId { uuid: u }, model: "%s", model_provider_id: "openai", service_tier: None, approval_policy: Never, approvals_reviewer: User, '
                'permission_profile: Managed { file_system: Restricted { entries: [FileSystemSandboxEntry { path: Special { value: Root }, access: Read, '
                'missing_path_behavior: None }], glob_scan_max_depth: None }, network: Restricted }, active_permission_profile: None, cwd: AbsolutePathBuf("C:\\x"), '
                'reasoning_effort: Some(High), network_proxy: None, rollout_path: None }\n' % model)
        return types.SimpleNamespace(stdout="\n".join(json.dumps(e) for e in ev), stderr="Reading prompt from stdin...\n" + sess, returncode=0)


class FakeCtl:
    def __init__(self, fail=False):
        self.fail, self.ensured, self.closed = fail, 0, 0

    def ensure(self):
        self.ensured += 1
        if self.fail:
            raise PlannerSkip("synthetic: ollama down")

    def close(self):
        self.closed += 1


def make_factory(fake, key=FAKEKEY):
    def factory(spec, **kw):
        kw.setdefault("sleep", lambda s: None)
        if spec == "gemma":
            return make_planner("gemma", transport=fake.ollama, **kw)
        if spec.startswith("codex:"):
            return make_planner(spec, runner=fake.codex, which=lambda x: "codex", **kw)
        env = {"THESIS_ANTHROPIC_API_KEY": key} if key else {}
        return make_planner(spec, transport=fake.anth, env=env, **kw)
    return factory


def mk(root, fake, passes, rs=None, ns=SYN, ctl=None, **kw):
    kw.setdefault("types_path", os.path.join(root, "types.json"))
    return PL.Plan(ns, passes=passes, rs=rs, root=root, provider_factory=make_factory(fake), ollama_ctl=ctl or FakeCtl(),
                   gpu_sample=False, key_getter=lambda: Secret(FAKEKEY), out=lambda m: None, **kw)


def new_root(**cap):
    root = tempfile.mkdtemp(prefix="ftm_plan_")
    d = os.path.join(root, "docs", "final-test")
    os.makedirs(d)
    shutil.copy(C.paths()["split"], d)
    shutil.copy(C.paths()["families"], d)
    json.dump(TYPES, open(os.path.join(root, "types.json"), "w"))
    build_capture(root, **cap)
    return root


def tree_digest(root):
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            h.update(p[len(root):].encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()


def plan_rows(root, ns=SYN):
    return C.Journal(C.journal_path(ns, "plan", root)).rows()


def event_rows(root, ns=SYN, name=None):
    r = C.Journal(C.journal_path(ns, "events", root)).rows()
    return [e for e in r if name is None or e["event"] == name]


def ledger_rows(root, ns=SYN):
    lg = C.ScopedLedger(os.path.join(C.ns_work(ns, root), "ledger.db"), C.NAMESPACES[ns]["ceilings"])
    return lg, lg.rows()


def leftovers(root, ns=SYN):
    return [r for r in ledger_rows(root, ns)[1] if r["state"] == "open"]


def cleanup(*roots):
    for r in roots:
        shutil.rmtree(r, ignore_errors=True)


# ================================================================================================== 1. static
src = open(PL.__file__, encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok310 = True
except SyntaxError:
    ok310 = False
check("plan.py parses as python 3.10", ok310)
FORBID = ("general.json", "states/", "outcome.sealed", "score.jsonl", "gold_of", "general_set", "import eval", "from eval",
          "thor_actions", "ai2thor", "h.scoring", "scoring()")
check("static gold isolation: plan.py names no gold / state / outcome / score path and imports no evaluator",
      not [f for f in FORBID if f in src], [f for f in FORBID if f in src])
check("plan.py never seeds the global RNG (seeded orders come from common)", "random." not in src and "import random" not in src)
import re as _re
check("plan.py contains no hard-coded key and no os.environ read of a key",
      not _re.search(r"sk-[A-Za-z0-9_-]{8,}", src) and "environ[" not in src and "environ.get" not in src)
sub = subprocess.run([sys.executable, "-I", "-c", (
    "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import plan; "
    "bad=[m for m in sys.modules if any(w in m.lower() for w in ('general','gold','ai2thor','thor_actions','score','outcome','webapp'))]; "
    "print('BAD', bad)"), os.path.join(HERE, "final_test", "matrix"), HERE], capture_output=True, text=True)
check("importing plan.py in a clean interpreter loads no dataset / simulator / scorer module", sub.stdout.strip() == "BAD []",
      (sub.stdout + sub.stderr)[-300:])
check("exit codes are distinct", len(set(PL.EXIT.values())) == len(PL.EXIT))

# ================================================================================================== 2. enumeration
syn_canon = [C.pres_id(f"T{i:03d}", lg) for i in range(125) for lg in ("en", "fa")]
syn_rep = [C.pres_id(f"T{i:03d}", lg) for i in range(30) for lg in ("en", "fa")]
syn_rel = [C.pres_id(f"T{i:03d}", lg, "reloc") for i in range(10) for lg in ("en", "fa")]
cells = PL.enumerate_cells("test-v1", syn_canon, syn_rep, syn_rel)
check("enumeration: keys are unique", len({c["key"] for c in cells}) == len(cells))
mine = {}
for c in cells:
    mine[(c["block"], c["arm"])] = mine.get((c["block"], c["arm"]), 0) + 1
plan_cp = CP.cells(CP.sizes())
theirs = {(c["block"], c["arm"]): c["calls"] for c in plan_cp}
check("enumeration reconciles with the call plan, per (block, arm), at the real sizes", mine == theirs,
      sorted(set(mine.items()) ^ set(theirs.items()))[:4])
bs = {"core": 0, "other": 0, "free": 0}
for c in cells:
    bs[c["scope"]] += 1
check("enumeration reconciles with the call plan per ledger scope (core / other / free)", bs == CP.by_scope(plan_cp), (bs, CP.by_scope(plan_cp)))
check("every cell belongs to exactly one known pass", all(c["pass"] in PL.PASSES for c in cells))
check("core pass = the four core arms (A23) in main + repeat, paid O only; Sonnet is its own archived pass",
      {c["arm"] for c in cells if c["pass"] == "core"} == {"O0", "O+M"}
      and {c["block"] for c in cells if c["pass"] == "core"} == {"main", "repeat"})
check("the core scope holds exactly the core pass", {c["pass"] for c in cells if c["scope"] == "core"} == {"core"})
check("gemma pass = every Gemma cell in every block (free)", {c["scope"] for c in cells if c["pass"] == "gemma"} == {"free"}
      and {c["block"] for c in cells if c["pass"] == "gemma"} == {"main", "repeat", "curve", "reloc"})
check("fs pass = O+FS only; curve pass = Opus curve arms; reloc pass = Opus reloc arms",
      {c["arm"] for c in cells if c["pass"] == "fs"} == {"O+FS"}
      and {c["arm"][0] for c in cells if c["pass"] == "curve"} == {"O"} and {c["arm"] for c in cells if c["pass"] == "reloc"} == {"O0", "O+M"})
check("A23: codex cells exist and are free; test-v1 has NO haiku / sonnet cell and no archived arm (they are archived passes)",
      {c["scope"] for c in cells if c["pass"] == "codex"} == {"free"} and not [c for c in cells if c["pass"] in ("haiku", "sonnet")]
      and not ({c["arm"] for c in cells} & set(C.ARCHIVED_ARMS)))
cells_syn = PL.enumerate_cells(SYN, syn_canon, syn_rep, syn_rel)
check("A23: an opted-in namespace (explicit archived_arms) DOES get haiku / sonnet cells, scope 'other', in their own passes; core stays Opus only",
      {c["scope"] for c in cells_syn if c["pass"] == "haiku"} == {"other"} and {c["arm"] for c in cells_syn if c["pass"] == "haiku"} == {"H0", "H+M"}
      and {c["arm"] for c in cells_syn if c["pass"] == "sonnet"} == {"S0", "S+M"} and {c["arm"] for c in cells_syn if c["pass"] == "core"} == {"O0", "O+M"})
_sv = dict(C.NAMESPACES["test-v1"])
C.NAMESPACES["test-v1"]["archived_arms"] = ("H0",)
try:
    _refused = False
    try:
        PL.enumerate_cells("test-v1", syn_canon, syn_rep, syn_rel)
    except ValueError:
        _refused = True
finally:
    C.NAMESPACES["test-v1"].clear()
    C.NAMESPACES["test-v1"].update(_sv)
check("A23: ... and it raises ValueError when test-v1 is tampered to opt into an archived arm", _refused)
curve_srcs = {c["source"] for c in cells if c["block"] == "curve"}
check("curve cells skip K = 0 and (order 0, K = 10) because those are shared with the '0' arm and S*",
      "o0k10" not in curve_srcs and not any(s.endswith("k0") for s in curve_srcs) and curve_srcs == {"o0k1", "o0k3", "o0k5"})   # A27: order 0 only
xm = [c for c in cells if c["pass"] == "xmain"]
check("A18: X0 / X+M (gpt-6-sol) are main-matrix cells of pass xmain, free, never in a cuttable group (A27: no REPEAT block)",
      {c["arm"] for c in xm} == {"X0", "X+M"} and {c["block"] for c in xm} == {"main"} and {c["scope"] for c in xm} == {"free"}
      and not any(C.cut_group(c["arm"], c["pres"]) for c in xm) and {c["planner"] for c in xm} == {"X:gpt-6-sol"}
      and {c["source"] for c in xm if c["arm"] == "X0"} == {"none"} and {c["source"] for c in xm if c["arm"] == "X+M"} == {"S*"})
check("A18: the exploratory Codex block no longer holds gpt-6-sol",
      not any("gpt-6-sol" in c["arm"] for c in cells if c["block"] == "codex") and len({c["planner"] for c in cells if c["block"] == "codex"}) == 3)
check("A27: repeat cells belong to the core four only (G0, G+M, O0, O+M)", {c["arm"] for c in cells if c["block"] == "repeat"} == set(C.REPEAT_ARMS))
check("repeat cells are r = 1, 2 only (r = 0 IS the main-matrix call); codex cells are r = 0, 1, 2",
      {c["r"] for c in cells if c["block"] == "repeat"} == {1, 2} and {c["r"] for c in cells if c["block"] == "codex"} == {0, 1, 2})
check("no reference-plan or replay arm makes a planner call", not [c for c in cells if c["arm"].startswith("P") or c["arm"].endswith("/p")])
check("classify / session_for: ok and rejected are terminal, infra is terminal after the third session, ceiling is redone",
      PL.classify({"key": "a"}, {}, False) == "pending"
      and PL.classify({"key": "a"}, {"a": {"status": "ok"}}, False) == "done"
      and PL.classify({"key": "a"}, {"a": {"status": "rejected"}}, False) == "done"
      and PL.classify({"key": "a"}, {"a": {"status": "infra", "session": 0}}, False) == "awaiting_requeue"
      and PL.classify({"key": "a"}, {"a": {"status": "infra", "session": 1}}, True) == "pending"
      and PL.classify({"key": "a"}, {"a": {"status": "infra", "session": 2}}, True) == "done"
      and PL.classify({"key": "a"}, {"a": {"status": "ceiling", "session": 0}}, False) == "pending"
      and PL.session_for({"status": "infra", "session": 0}) == 1 and PL.session_for({"status": "ceiling", "session": 1}) == 1
      and PL.session_for(None) == 0)
sec = Secret(FAKEKEY)
check("leak check: the exact key anywhere is caught, a key pattern in a structural field is caught, harmless reply text is not",
      PL.leak_paths({"a": 1, "attempts": [{"raw": "x " + FAKEKEY}]}, (sec,)) != []
      and PL.leak_paths({"model": "sk-ant-abcdefghijkl123"}, ()) != []
      and PL.leak_paths({"attempts": [{"raw": "a task-planning-style note"}], "say": "ok"}, (sec,)) == [])

# ================================================================================================== 3. the test-item guard
test_ids = list(MANIFEST["test"])[:2]
capped = list(MANIFEST["test_uncapped_not_scored"])[:1]
d0 = tempfile.mkdtemp(prefix="ftm_guard_")
os.makedirs(os.path.join(d0, "docs", "final-test"))
shutil.copy(C.paths()["split"], os.path.join(d0, "docs", "final-test"))
check("guard: a test item is refused in a non-test namespace", raises(lambda: C.guard_items(test_ids, "rehearsal-v1", root=d0), C.TestItemRefused))
check("guard: a test item in test-v1 is refused without --allow-test", raises(lambda: C.guard_items(test_ids, "test-v1", root=d0), C.TestItemRefused))
check("guard: --allow-test alone (no freeze.json, no families approval) is still refused",
      raises(lambda: C.guard_items(test_ids, "test-v1", allow_test=True, root=d0), C.TestItemRefused))
check("guard: a capped-out item is refused in every namespace",
      all(raises(lambda ns=ns: C.guard_items(capped, ns, allow_test=True, root=d0), C.TestItemRefused) for ns in REAL_NS))
cleanup(d0)
rootg = new_root()                    # a capture that holds test-item ids (as bare strings; instruction text is synthetic)


def test_breaker(row):
    return row


rg = tempfile.mkdtemp(prefix="ftm_guard2_")
os.makedirs(os.path.join(rg, "docs", "final-test"))
shutil.copy(C.paths()["split"], os.path.join(rg, "docs", "final-test"))
shutil.copy(C.paths()["families"], os.path.join(rg, "docs", "final-test"))
fake_g = Fake()
build_capture(rg, ns="test-v1", items=test_ids, repeat_items=(), reloc_items=())
code = PL.Plan("test-v1", passes=["core"], rs=[0], root=rg, provider_factory=make_factory(fake_g), ollama_ctl=FakeCtl(), gpu_sample=False,
               key_getter=lambda: Secret(FAKEKEY), out=lambda m: None, types_path=os.path.join(rootg, "types.json")).run()
check("Plan on test-v1 with test items: refused (exit 1), zero calls, zero plan records",
      code == PL.EXIT["abort"] and not fake_g.calls and not plan_rows(rg, "test-v1"), (code, fake_g.calls[:2]))
code = PL.Plan("test-v1", passes=["core"], rs=[0], root=rg, allow_test=True, provider_factory=make_factory(fake_g), ollama_ctl=FakeCtl(),
               gpu_sample=False, key_getter=lambda: Secret(FAKEKEY), out=lambda m: None,
               types_path=os.path.join(rootg, "types.json")).run()
check("Plan on test-v1 with --allow-test but no freeze.json: still refused, zero calls", code == PL.EXIT["abort"] and not fake_g.calls)
_old_fs = C.freeze_status
C.freeze_status = lambda root=None, check_code=True: {"ok": True, "reasons": []}
try:
    code = PL.Plan("test-v1", passes=["core"], rs=[0], root=rg, allow_test=True, provider_factory=make_factory(fake_g), ollama_ctl=FakeCtl(),
                   gpu_sample=False, key_getter=lambda: Secret(FAKEKEY), out=lambda m: None,
                   types_path=os.path.join(rootg, "types.json")).run()
finally:
    C.freeze_status = _old_fs
check("with every guard condition met the next gate (design = the whole test set) still stops an incomplete capture; zero calls",
      code == PL.EXIT["abort"] and not fake_g.calls and any(e["event"] == "abort_design" for e in event_rows(rg, "test-v1")))
check("practice-v1 has no Phase B", raises(lambda: PL.Plan("practice-v1"), ValueError))
cleanup(rg, rootg)

# the frozen type list: explicit path, matrix detector_types.json ({types: [...]}), pilot copy; freeze.json cross-check
rt = new_root()
os.makedirs(C.paths(rt)["matrix_docs"], exist_ok=True)
check("type list: with no list anywhere the run aborts", raises(lambda: PL.load_types(None, rt), PL.PlanAbort))
os.makedirs(os.path.join(C.paths(rt)["docs"], "pilot"), exist_ok=True)
json.dump(TYPES, open(os.path.join(C.paths(rt)["docs"], "pilot", "types.json"), "w"))
check("type list: the pilot copy (a plain list) is the last fallback", PL.load_types(None, rt) == (sorted(TYPES), "types.json"))
dt_path = os.path.join(C.paths(rt)["matrix_docs"], "detector_types.json")
json.dump({"types": TYPES, "weights_sha": "abc"}, open(dt_path, "w"))
check("type list: matrix/detector_types.json ({types: [...]}) is preferred and sorted", PL.load_types(None, rt) == (sorted(TYPES), "detector_types.json"))
json.dump({"detector_types": {"n": 7, "sha256": C.sha256_json(sorted(TYPES))}}, open(C.paths(rt)["freeze"], "w"))
_rend = PL.none_request_renderer(rt)
_rec = {"instruction": "open the fridge", "summary": ["Fridge [closed] on/in ? at (1.00, 2.00), step 3"], "holding": None}
_want = prompt_sha256(PL.render_messages(PL.PlannerContract(sorted(TYPES), tools=PL.hs.TOOLS), _rec, "none"))
check("A19: none_request_renderer(root) gives the sha of the none-arm request rendered from a capture record (execute G-iso component (i))", _rend(_rec) == _want and len(_want) == 64)

check("type list: a list equal to freeze.json's recorded sha is accepted", PL.load_types(None, rt)[0] == sorted(TYPES))
json.dump({"detector_types": {"n": 7, "sha256": "0" * 64}}, open(C.paths(rt)["freeze"], "w"))
check("type list: a list that differs from freeze.json's recorded sha aborts", raises(lambda: PL.load_types(None, rt), PL.PlanAbort))
check("type list: an explicit path that does not exist aborts", raises(lambda: PL.load_types(os.path.join(rt, "nope.json"), rt), PL.PlanAbort))
cleanup(rt)

# ================================================================================================== 4. gates before any call
r1 = new_root(complete=False)
f1 = Fake()
check("capture without phase_complete: Phase B refuses (abort_phase_order), zero calls",
      mk(r1, f1, ["core"], [0]).run() == PL.EXIT["abort"] and not f1.calls and event_rows(r1, name="abort_phase_order"))
cleanup(r1)

r2 = new_root()
open(C.journal_path(SYN, "capture", r2), "a", encoding="utf-8").write("")
J = C.Journal(C.journal_path(SYN, "capture", r2))
rows = J.rows()
rows[0]["instruction"] = rows[0]["instruction"] + " (edited)"
open(J.path, "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
f2 = Fake()
check("capture edited after its phase_complete: abort_capture_changed, zero calls",
      mk(r2, f2, ["core"], [0]).run() == PL.EXIT["abort"] and event_rows(r2, name="abort_capture_changed") and not f2.calls)
cleanup(r2)


def break_request(row):
    if row["pres"] == C.pres_id(DEV[3], "fa"):
        row["requests"]["S*"] = "0" * 64
    return row


r3 = new_root(breaker=break_request)
f3 = Fake()
check("REQUEST-IDENTITY GATE: one wrong captured request sha aborts the pass before the first call",
      mk(r3, f3, ["core"], [0]).run() == PL.EXIT["abort"] and not f3.calls and not plan_rows(r3) and event_rows(r3, name="abort_request_mismatch"))
cleanup(r3)

r3b = new_root()
json.dump(TYPES + ["Spatula"], open(os.path.join(r3b, "types.json"), "w"))
f3b = Fake()
check("REQUEST-IDENTITY GATE: a different frozen type list changes the system text and is caught, zero calls",
      mk(r3b, f3b, ["gemma"], [0]).run() == PL.EXIT["abort"] and not f3b.calls and event_rows(r3b, name="abort_request_mismatch"))
cleanup(r3b)


def break_block(row):
    if row["pres"] == C.pres_id(DEV[2], "en"):
        row["blocks"]["S*"]["text"] += " extra"                       # block text no longer matches its request sha
    return row


r3c = new_root(breaker=break_block)
f3c = Fake()
check("REQUEST-IDENTITY GATE: an edited memory block is caught, zero calls",
      mk(r3c, f3c, ["core"], [0]).run() == PL.EXIT["abort"] and not f3c.calls and event_rows(r3c, name="abort_request_mismatch"))
cleanup(r3c)


def mask_fp(row):
    row["blocks"]["S*"]["text"] = "see FloorPlan1 layout"
    ct = contract()
    m = ct.messages(row["instruction"], row["summary"], None, memory_block=row["blocks"]["S*"]["text"])
    row["requests"]["S*"] = hashlib.sha256(json.dumps([[x["role"], x["content"]] for x in m], ensure_ascii=False,
                                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    return row


def mask_oid(row):
    row["blocks"]["S*"]["text"] = "Apple|+01.20|+00.90|-01.50 is near"       # the '+' the registered pattern misses
    return _rehash(row)


def _rehash(row):
    ct = contract()
    m = ct.messages(row["instruction"], row["summary"], None, memory_block=row["blocks"]["S*"]["text"])
    row["requests"]["S*"] = hashlib.sha256(json.dumps([[x["role"], x["content"]] for x in m], ensure_ascii=False,
                                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    return row


for nm, brk in (("a scene name", mask_fp), ("a signed objectId", mask_oid)):
    rm = new_root(breaker=brk)
    fm = Fake()
    check(f"MASKING GATE: {nm} in a prompt aborts before any call",
          mk(rm, fm, ["core"], [0]).run() == PL.EXIT["abort"] and not fm.calls and event_rows(rm, name="abort_masking"))
    cleanup(rm)

r4 = new_root(items=DEV[:-1])                 # A26: one dev item left out (was DEV[:11] of 12)
f4 = Fake()
check("design check: a missing canonical presentation aborts (abort_design), zero calls",
      mk(r4, f4, ["core"], [0]).run() == PL.EXIT["abort"] and not f4.calls and event_rows(r4, name="abort_design"))
cleanup(r4)

# ================================================================================================== 5. a full clean run, pass by pass
root = new_root()
fake = Fake()
fake.bad_first = {5, 6}
fake.reject = {7}
fake.tool_idx = {1}
opened = []
_real_open = builtins.open


def spy_open(file, *a, **k):
    opened.append(str(file))
    return _real_open(file, *a, **k)


poison = [os.path.join(root, "sim", "dataset", "general.json"), os.path.join(C.ns_docs(SYN, root), "states", "p.c1.json"),
          C.journal_path(SYN, "outcome", root), C.journal_path(SYN, "score", root)]
for pth in poison:
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    _real_open(pth, "w").write("POISON")
codes = {}
builtins.open = spy_open
try:
    ctl = FakeCtl()
    for label, passes, rs in (("gemma", ["gemma"], None), ("core0", ["core"], [0]), ("core12", ["core"], [1, 2]), ("fs", ["fs"], None),
                              ("curve", ["curve"], None), ("reloc", ["reloc"], None), ("haiku", ["haiku"], None), ("sonnet", ["sonnet"], None), ("xmain", ["xmain"], None), ("codex", ["codex"], None)):
        codes[label] = mk(root, fake, passes, rs, ctl=ctl).run()
finally:
    builtins.open = _real_open
check("every pass of the clean run exits 0", set(codes.values()) == {0}, codes)
rows = plan_rows(root)
# A26: main = 12 arms x NP canonical presentations (288 with the original 24); the other passes do not depend on the dev-pilot size
N_A27 = 12 * NP + 32 + 24 + 16 + 72
check(f"A27: {N_A27} plan records (main {12 * NP} + repeat 32 (core four only) + curve 24 (order 0, K = 1, 3, 5) + reloc 16 + exploratory codex 72), one per cell, unique keys",
      len(rows) == N_A27 and len({r["key"] for r in rows}) == N_A27, len(rows))
check("every record passes the firewall (no gold / outcome key anywhere)",
      all(not C.forbidden_keys(r, C.GOLD_KEYS | C.OUTCOME_KEYS) for r in rows))
by_status = {}
for r in rows:
    by_status[r["status"]] = by_status.get(r["status"], 0) + 1
check("statuses: ok and rejected only in a healthy run; item 7 is rejected after its one repair",
      set(by_status) == {"ok", "rejected"} and all(r["calls"] == 2 for r in rows if r["status"] == "rejected" and r["planner"] != "X:x"),
      by_status)
rej = [r for r in rows if r["status"] == "rejected"]
check("a rejected cell stores no plan and is terminal", all(r["plan"] == [] and r["steps"] == [] for r in rej) and rej)
rep = [r for r in rows if r["repaired"]]
check("an invalid first reply followed by a valid repair is recorded: repaired, first_ok False, calls 2, plan present",
      rep and all(not r["first_ok"] and r["calls"] >= 2 and r["plan"] == [["pickup", "Apple", None, None]] for r in rep))
check("plans are normalised [[action, object, to, arg]] lists",
      all(r["plan"] == [["pickup", "Apple", None, None]] for r in rows if r["status"] == "ok"))
check("request sha in every record equals the captured one",
      all(r["request_sha"] == next(c for c in C.Journal(C.journal_path(SYN, "capture", root)).rows() if c["pres"] == r["pres"])["requests"][r["source"]]
          for r in rows))
check("scope field matches common.scope_of for every record", all(r["scope"] == C.scope_of(r["arm"], r["block"], SYN) for r in rows))
lg, lrows = ledger_rows(root)
check("the ledger holds no open reservation or block hold after a clean run", not [x for x in lrows if x["state"] == "open"])
core_usd = round(sum(r["cost_usd"] for r in rows if r["scope"] == "core"), 6)
other_usd = round(sum(r["cost_usd"] for r in rows if r["scope"] == "other"), 6)
check("ledger spend per scope equals the sum of the records' cost (core / other)",
      abs(lg.billed_scope("core") - core_usd) < 1e-6 and abs(lg.billed_scope("other") - other_usd) < 1e-6, (lg.billed_scope("core"), core_usd))
check("Gemma, Codex and replays cost nothing in the ledger", abs(lg.billed_scope("free")) < 1e-9)
check("spend stays under the ceilings", lg.committed() <= C.NAMESPACES[SYN]["ceilings"]["total"])
opus_row = next(r for r in rows if r["planner"] == "O" and r["calls"] == 1 and r["cached_calls"] == 0)
check("Opus cost = billed tokens x list price (1200 in, 40 out at 4 / 20 USD per M)", abs(opus_row["cost_usd"] - (1200 * 4 + 40 * 20) / 1e6) < 1e-9, opus_row["cost_usd"])
pairs = {}
for r in rows:
    if r["planner"] == "S" and r["r"] == 0 and r["arm"] in ("S0", "S+M"):
        pairs.setdefault(r["pres"], {})[r["arm"]] = r
empty_same = [p for p, d in pairs.items() if len(d) == 2 and d["S0"]["request_sha"] == d["S+M"]["request_sha"]]
check("identical requests (an empty +M block) are served from the response cache once: exactly one of the pair is a replay at 0 USD",
      not empty_same or all(sum(1 for r in pairs[p].values() if r["cached_calls"] == r["calls"]) == 1 for p in empty_same))
check("pass_complete events exist for every pass and a single phase_complete for plan",
      {e["pass"] for e in event_rows(root, name="pass_complete")} == set(PL.PASSES) and len(event_rows(root, name="phase_complete")) == 2,
      [e["event"] for e in event_rows(root)][-4:])
pc = [e for e in event_rows(root, name="phase_complete") if e["phase"] == "plan"]
check("phase_complete plan carries n and the journal sha",
      len(pc) == 1 and pc[0]["n"] == N_A27 and pc[0]["journal_sha"] == C.sha256_file(C.journal_path(SYN, "plan", root)))
check("Gemma went through the Ollama check (cache misses only); close() ran", ctl.ensured > 0 and ctl.closed == 10)
check("the Codex tool-event item is flagged as an isolation violation", event_rows(root, name="codex_isolation_violation"))
check("plan_start events record the prompt version, types sha and decoding shas",
      all(e.get("prompt_version") == "ft-v1" and e.get("types_sha") and e.get("decoding_sha") for e in event_rows(root, name="plan_start")))
# interleaving and contiguity inside the core r0 pass
core0 = [r for r in rows if r["planner"] in ("S", "O") and r["block"] == "main" and r["arm"] in C.CORE_ARMS]
order = []
for r in core0:
    if not order or order[-1] != r["pres"]:
        order.append(r["pres"])
check("all cells of one presentation are one contiguous block in the journal", len(order) == len(set(order)) == NP)          # A26: was 24
want = [p for p, _ in C.plan_blocks(sorted({r["pres"] for r in core0}), lambda p: [1], SYN, "core", "0")]
check("block order = the seeded C.plan_blocks order", order == want)
first_arms = [tuple(r["arm"] for r in core0 if r["pres"] == p) for p in order]
check("arms are interleaved: the arm order differs between blocks", len(set(first_arms)) > 1)
# gold isolation at runtime
bad_open = [o for o in opened if any(x in o.replace("\\", "/") for x in ("general.json", "/states/", "outcome.sealed", "score.jsonl"))]
check("RUNTIME gold isolation: the run never opened the poisoned dataset / state / outcome / score files", not bad_open, bad_open[:3])
txt = ""
for dp, _, fns in os.walk(root):
    for fn in fns:
        if fn != "general.json":
            try:
                txt += _real_open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
            except OSError:
                pass
check("the (fake) key value appears in no file the run wrote", FAKEKEY not in txt)
check("every journal line is free of key patterns in its structural fields",
      all(not PL.leak_paths(r, (Secret(FAKEKEY),)) for r in rows + event_rows(root)))
digest_done = tree_digest(root)
codes2 = mk(root, fake, PL.PASSES).run()
rows_after = plan_rows(root)
check("re-running a finished study is a no-op: exit 0, no new plan record, no new call",
      codes2 == 0 and len(rows_after) == N_A27 and tree_digest(root) != "" and len(fake.calls) == fake.n)
n_calls_before = fake.n
mk(root, fake, PL.PASSES).run()
check("...and still makes zero calls", fake.n == n_calls_before)
st = mk(root, fake, PL.PASSES).status()
check("status(): per pass counts add up", sum(d["cells"] for d in st.values()) == N_A27 and sum(d["ok"] + d["rejected"] for d in st.values()) == N_A27 and st["core"]["pending"] == 0)
cleanup(root)

# ================================================================================================== 6. determinism, interrupt, resume, crash
ra, rb = new_root(), new_root()
fa_, fb_ = Fake(), Fake()
mk(ra, fa_, ["core"], [0]).run()
code = mk(rb, fb_, ["core"], [0], interrupt_after=7).run()
check("injected interrupt: exit 5 after exactly 7 records, hold released, no open reservation",
      code == PL.EXIT["interrupt"] and len(plan_rows(rb)) == 7 and not leftovers(rb), (code, len(plan_rows(rb))))
with open(C.journal_path(SYN, "plan", rb), "a", encoding="utf-8") as fh:        # a crash mid-write: torn last line
    fh.write('{"key": "torn')
calls_b1 = fb_.n
code = mk(rb, fb_, ["core"], [0]).run()
check("resume after an interrupt and a torn tail: exit 0, the torn line is moved aside", code == 0 and os.path.exists(C.journal_path(SYN, "plan", rb) + ".torn"))
rows_a, rows_b = plan_rows(ra), plan_rows(rb)
check("resumed journal = clean journal: same keys, same plans, same cells order",
      [r["key"] for r in rows_a] == [r["key"] for r in rows_b] and [r["plan"] for r in rows_a] == [r["plan"] for r in rows_b]
      and len(rows_b) == 2 * NP)                    # A26: was 48
check("resume re-did nothing: the second run made only the missing calls", fb_.n - calls_b1 <= len(fa_.calls) - 7 + 2, (fb_.n, calls_b1, len(fa_.calls)))
la, lb = ledger_rows(ra)[0], ledger_rows(rb)[0]
check("resume never double-bills: total billed equals the clean run's", abs(la.billed_scope() - lb.billed_scope()) < 1e-9)
check("a different, independent root gives the identical seeded order (machine independent)",
      [r["pres"] for r in rows_a] == [r["pres"] for r in rows_b])
fc = Fake()
rc = new_root()
fc.crash_on_call = 9
try:
    mk(rc, fc, ["core"], [0]).run()
    crashed = False
except Crash:
    crashed = True
check("a crash mid-cell propagates, leaves no half record, releases the hold and the phase lock",
      crashed and len(plan_rows(rc)) == 8 and not leftovers(rc) and not os.path.exists(os.path.join(C.ns_work(SYN, rc), "plan.lock")))
check("...and the redo completes the study", mk(rc, fc, ["core"], [0]).run() == 0 and len(plan_rows(rc)) == 2 * NP)
check("a held lock refuses a second writer; --steal-lock proceeds and logs lock_stolen",
      (lambda: [C.PhaseLock(SYN, "plan", rc).__enter__(), raises(lambda: mk(rc, Fake(), ["core"], [0]).run(), C.PhaseLocked),
                mk(rc, Fake(), ["core"], [0], steal_lock=True).run(), event_rows(rc, name="lock_stolen")][-1])())
cleanup(ra, rb, rc)

# ================================================================================================== 7. infrastructure failures and re-queue
rh = new_root()
fh_ = Fake()
fh_.fail_idx = {1, 4}
c1 = mk(rh, fh_, ["haiku"], [0]).run()
rows_h = plan_rows(rh)
inf = [r for r in rows_h if r["status"] == "infra"]
check("transport failures are recorded as infra, session 0, no plan, no repair attempt, and the pass is incomplete (exit 6)",
      c1 == PL.EXIT["incomplete"] and len(inf) == 2 * 2 * 2 and all(r["session"] == 0 and r["plan"] == [] and r["calls"] == 1 for r in inf), (c1, len(inf)))
n_before = len(plan_rows(rh))
c2 = mk(rh, fh_, ["haiku"], [0]).run()
check("without --requeue-infra nothing is retried", c2 == PL.EXIT["incomplete"] and len(plan_rows(rh)) == n_before)
fh_.fail_idx = set()
c3 = mk(rh, fh_, ["haiku"], [0], requeue_infra=True).run()
rl = C.Journal(C.journal_path(SYN, "plan", rh)).latest()
check("--requeue-infra re-runs infra cells in a later session (session 1) and they succeed", c3 == 0 and not [r for r in rl.values() if r["status"] == "infra"]
      and all(r["session"] == 1 for k, r in rl.items() if any(r["item"] == DEV[i] for i in (1, 4))))
check("pass_complete is written only once every haiku cell is terminal", len(event_rows(rh, name="pass_complete")) == 1)
rp = new_root()
fp_ = Fake()
fp_.fail_idx = {2}
codes_p = [mk(rp, fp_, ["haiku"], [0], requeue_infra=(i > 0)).run() for i in range(5)]
rl2 = C.Journal(C.journal_path(SYN, "plan", rp)).rows()
k_inf = [r for r in rl2 if r["status"] == "infra"]
sess = {}
for r in k_inf:
    sess.setdefault(r["key"], []).append(r["session"])
check("a persistent failure is re-queued in sessions 1 and 2, then is terminal (3 records per cell, scored 0 and reported)",
      codes_p == [6, 6, 0, 0, 0] and all(v == [0, 1, 2] for v in sess.values()) and len(sess) == 4, (codes_p, sess))
cleanup(rh, rp)

rq = new_root()
fq = Fake()
fq.err500 = {3}
mk(rq, fq, ["haiku"], [0]).run()
rows_q = plan_rows(rq)
lq, lrq = ledger_rows(rq)
check("an ambiguous transport failure (HTTP 500, maybe billed) is retried; the cell still succeeds and records the unresolved reservation",
      len(rows_q) == 2 * NP and all(r["status"] == "ok" for r in rows_q)
      and sum(a["extra"]["unresolved_reservations"] for r in rows_q for a in r["attempts"]) == 1)
open_rows = [x for x in lrq if x["state"] == "open"]
check("...and that reservation stays OPEN in the ledger and keeps counting against the ceiling",
      len(open_rows) == 1 and lq.committed("other") > lq.billed_scope("other"))
cleanup(rq)

# ================================================================================================== 8. ledger: worst-case holds and ceilings
# M-fix2: worst-case holds are PER CELL (an Opus cell ~0.397 USD, a Sonnet cell ~0.198 here). The cost guard is switched off for these two
# checks (it would pause on the projection first); the ledger alone must refuse.
_old_boundary = PL.Plan._boundary
PL.Plan._boundary = lambda self, pres: None
set_ceilings(0.3, 2.0)
rc1 = new_root()
fc1 = Fake()
c = mk(rc1, fc1, ["core"], [0]).run()
rows1 = plan_rows(rc1)
check("a cell whose worst case crosses the ceiling is refused before anything is sent (exit 4): the Opus cell never runs",
      c == PL.EXIT["ceiling"] and event_rows(rc1, name="block_refused") and not any(r["planner"] == "O" for r in rows1)
      and len(fc1.calls) == len(rows1) and not leftovers(rc1), (c, len(fc1.calls), len(rows1)))
cleanup(rc1)
set_ceilings(round(0.62 * NP / 24, 4), 2.0)    # A26: the 0.62 ceiling was sized for 24 presentations; scaled to NP
rc2 = new_root()
fc2 = Fake()
c = mk(rc2, fc2, ["core"], [0]).run()
PL.Plan._boundary = _old_boundary
r_ = plan_rows(rc2)
per = {}
for r in r_:
    per[r["pres"]] = per.get(r["pres"], 0) + 1
lgc, lrc = ledger_rows(rc2)
check("when the ceiling stops the study at most ONE presentation is partial (per-cell holds, M-fix2); the event says so",
      c == PL.EXIT["ceiling"] and r_ and sorted(per.values())[1:] == [2] * (len(per) - 1) and not leftovers(rc2)
      and (event_rows(rc2, name="block_refused") or [{}])[0].get("partial") == (sorted(per.values())[0] != 2), (c, per))
check("committed spend never passed the ceiling", lgc.committed("core") <= 1.2 + 1e-9)
holds = [json.loads(x) for x in open(os.path.join(C.ns_work(SYN, rc2), "ledger.db.jsonl"), encoding="utf-8")]
holds = [h for h in holds if h.get("event") == "block_hold"]
prov_o = make_factory(Fake())("opus", cache=None, ledger=PL._DryLedger())
msgs = contract().messages("x " * 200, ["Fridge [closed]"], None)
first_o = prov_o.reserve_usd(msgs, None)
check("the block hold covers every first transmission plus one repair at the provider's own reserve",
      holds and max(h["usd"] for h in holds) >= 2 * first_o * 0.9 and PL.cell_worst(prov_o, contract(), msgs) > 2 * first_o)
check("the hold is a worst case, far above what the block really billed",
      holds[0]["usd"] > r_[0]["cost_usd"] * 20)
ev_ = event_rows(rc2, name="block_refused")
check("block_refused names the block and the committed spend", ev_ and "committed" in ev_[0] and "pres" in ev_[0])
cleanup(rc2)
set_ceilings(50.0, 100.0)

# ================================================================================================== 9. cost guard: pause only at block boundaries
_old_every = C.OPUS_GUARD_EVERY
C.OPUS_GUARD_EVERY = 10
rg1 = new_root()
fg1 = Fake()
fg1.opus_out = 6000                                     # ~0.12 USD per Opus call: above the A37 threshold 0.080 (3000 = 0.06 paused only under A24.1-C's 0.040)
c = mk(rg1, fg1, ["core"], None).run()
rows_g = plan_rows(rg1)
core_sel = {}
for cl in PL.enumerate_cells(SYN, [C.pres_id(i, lg) for i in DEV for lg in ("en", "fa")],
                             [C.pres_id(i, lg) for i in REPEAT_ITEMS for lg in ("en", "fa")], []):
    if cl["pass"] == "core":
        core_sel[cl["pres"]] = core_sel.get(cl["pres"], 0) + 1
per_g = {}
for r in rows_g:
    per_g[r["pres"]] = per_g.get(r["pres"], 0) + 1
pe = event_rows(rg1, name="pause_cost_guard")
check("Opus mean cost above the A37 threshold 0.080 pauses the run (exit 3) with a pause_cost_guard event naming 0.080",
      c == PL.EXIT["pause"] and pe and any("Opus mean" in x and "0.080" in x for x in pe[0]["reasons"]), (c, pe[:1]))
check("the pause fell at a BLOCK BOUNDARY: every presentation seen so far has all its core cells, none half done",
      all(per_g[p] == core_sel[p] for p in per_g) and pe[0]["after_block"] in per_g, per_g)
check("no hold or reservation is left open by a pause", not leftovers(rg1))
n_rows = len(rows_g)
calls_g = fg1.n
c = mk(rg1, fg1, ["core"], None).run()
check("a paused run refuses to continue without a recorded decision (exit 3, zero new calls)",
      c == PL.EXIT["pause"] and fg1.n == calls_g and len(plan_rows(rg1)) == n_rows)
def _raises_ve(fn):
    try:
        fn()
    except ValueError:
        return True
    except BaseException:
        return False
    return False
check("A21 B: an acknowledgement with response 'cut' but no cut recorded after the pause is refused",
      _raises_ve(lambda: mk(rg1, fg1, ["core"], None, ack_pause="Arash: cut", pause_response="cut").run()))
check("A21 B: an unknown pause response is refused", _raises_ve(lambda: mk(rg1, fg1, ["core"], None, ack_pause="x", pause_response="change-prompt").run()))
check("A21 B: a test-v1 pause decision needs a real note and the outcome-blind attestation; the three responses are the only ones",
      PL.Plan._ack_problem("test-v1", "short", "continue", True) and PL.Plan._ack_problem("test-v1", "spend 12.3 USD, mean 0.031 per Opus cell, no errors", "continue", False)
      and PL.Plan._ack_problem("test-v1", "spend 12.3 USD, mean 0.031 per Opus cell, no errors", "continue", True) is None
      and PL.Plan._ack_problem("test-v1", "spend 12.3 USD, mean 0.031 per Opus cell, no errors", "new-prompt", True)
      and PL.Plan._ack_problem("rehearsal-v1", "x", "cut", False) is None)
check("A21 B: the pause event names the guard spec version", (event_rows(rg1, name="pause_cost_guard") or [{}])[-1].get("guard_spec") == C.COST_GUARD_SPEC["version"])
c = mk(rg1, fg1, ["core"], None, ack_pause="Arash: continue, thinking regime accepted", pause_response="continue").run()
ack = event_rows(rg1, name="pause_acknowledged")
check("--ack-pause records the decision and the run goes on (outcome-blind: only spend counters were read)",
      ack and len(plan_rows(rg1)) > n_rows and ack[0]["note"].startswith("Arash") and ack[0].get("response") == "continue")
check("after the acknowledgement the Opus check is re-armed (the next pause needs another 10 billed calls)",
      len(plan_rows(rg1)) > n_rows)
pl_o = mk(rg1, fg1, ["core"], None)
pl_o._prepare()
want_obs = {}
for r in plan_rows(rg1):
    if r["planner"] in ("S", "O", "H") and r["calls"] > r["cached_calls"] and r["cost_usd"] > 0:
        d = want_obs.setdefault(r["planner"], {"n": 0, "usd": 0.0})
        d["n"] += 1
        d["usd"] += r["cost_usd"]
got = pl_o._observed()
check("the billed-cell counters kept incrementally equal a recount of the journal",
      {k: v["n"] for k, v in got.items()} == {k: v["n"] for k, v in want_obs.items()}
      and all(abs(got[k]["usd"] - want_obs[k]["usd"]) < 1e-9 for k in got))
_o = {}
for _c in (0.010, 0.012, 0.0, 0.020):
    pl_o._count(_o, {"planner": "O", "calls": 2, "cached_calls": 0, "cost_usd": _c})
pl_o._count(_o, {"planner": "O", "calls": 1, "cached_calls": 1, "cost_usd": 0.0})
check("A21 B: the guard denominator counts BILLED Opus cells only (an unbilled transport failure and a cache replay are not cells); costs kept in order",
      _o["O"]["n"] == 3 and abs(_o["O"]["usd"] - 0.042) < 1e-12 and _o["O"]["costs"] == [0.010, 0.012, 0.020])
cleanup(rg1)
C.OPUS_GUARD_EVERY = _old_every

rgf = new_root()
fgf = Fake()
fgf.opus_out = 6000                                     # a cost-guard pause must be pending (above the A37 0.080)
C.OPUS_GUARD_EVERY = 10
mk(rgf, fgf, ["core"], None).run()
C.OPUS_GUARD_EVERY = _old_every
fgf2 = Fake()
check("a pending cost-guard pause does not hold up a free (Gemma) pass", mk(rgf, fgf2, ["gemma"], [0]).run() == 0 and fgf2.calls)
cleanup(rgf)
set_ceilings(2.0, 100.0)
_old_prior = dict(PL.PRIOR_USD_PER_CELL)
PL.PRIOR_USD_PER_CELL = {k: v * 5 for k, v in _old_prior.items()}      # prior projection of the remaining core cells ~4.3 USD > 2.0
rg2 = new_root()
fg2 = Fake()
c = mk(rg2, fg2, ["core"], [0]).run()
pe2 = event_rows(rg2, name="pause_cost_guard")
check("projected completion above the ceiling pauses BEFORE the first call (zero calls, zero records)",
      c == PL.EXIT["pause"] and not fg2.calls and not plan_rows(rg2) and pe2 and any("projected" in x for x in pe2[0]["reasons"]), (c, pe2[:1]))
c = mk(rg2, fg2, ["core"], [0], ack_pause="accepted").run()
check("after the acknowledgement the pass completes if the projection does not grow by more than 5 %", c == 0 and len(plan_rows(rg2)) == 2 * NP, (c, len(plan_rows(rg2))))
check("projection is evaluated on core and total separately (both in the pause event)", pe2 and set(pe2[0]["projected"]) == {"core", "total"})
cleanup(rg2)
PL.PRIOR_USD_PER_CELL = _old_prior
set_ceilings(50.0, 100.0)

# ================================================================================================== 10. skips, Ollama, key
rs1 = new_root()
fs1 = Fake()
c = mk(rs1, fs1, ["gemma"], [0], ctl=FakeCtl(fail=True)).run()
check("Ollama down: the Gemma pass is skipped cleanly (exit 2), nothing recorded, event written",
      c == PL.EXIT["skip"] and not plan_rows(rs1) and event_rows(rs1, name="planner_skipped") and not fs1.calls)
cleanup(rs1)
rs2 = new_root()
fs2 = Fake()
pl2 = PL.Plan(SYN, passes=["core"], rs=[0], root=rs2, provider_factory=make_factory(fs2, key=""), ollama_ctl=FakeCtl(), gpu_sample=False,
              key_getter=lambda: (_ for _ in ()).throw(PlannerSkip("no key")), out=lambda m: None, types_path=os.path.join(rs2, "types.json"))
c = pl2.run()
check("no API key: the paid pass is skipped cleanly (exit 2), nothing recorded, no hold left", c == PL.EXIT["skip"] and not plan_rows(rs2) and not leftovers(rs2))
cleanup(rs2)
started = []
procs = []


class FakeProc:
    pid = 4242


def fake_popen(cmd, env=None, **kw):
    started.append((cmd, env.get("OLLAMA_HOST")))
    return FakeProc()


alive_state = {"up": False}
killed = []
oc = PL.OllamaCtl("http://x", False, popen=fake_popen, get=lambda u, t=0: alive_state["up"] or (_ for _ in ()).throw(OSError()),
                  sleep=lambda s: None, killer=lambda p: killed.append(p.pid))
check("OllamaCtl: down and no --start-ollama -> clean skip", raises(oc.ensure, PlannerSkip) and not started)
oc2 = PL.OllamaCtl("http://x", True, popen=lambda *a, **k: (alive_state.update(up=True), fake_popen(*a, **k))[1],
                   get=lambda u, t=0: alive_state["up"] or (_ for _ in ()).throw(OSError()), sleep=lambda s: None,
                   killer=lambda p: killed.append(p.pid))
oc2.ensure()
oc2.close()
check("OllamaCtl: --start-ollama starts `ollama serve` with OLLAMA_HOST=0.0.0.0 and stops exactly that process",
      started and started[-1][0] == ["ollama", "serve"] and started[-1][1] == "0.0.0.0" and killed == [4242])
alive_state["up"] = True
oc3 = PL.OllamaCtl("http://x", True, popen=fake_popen, get=lambda u, t=0: True, sleep=lambda s: None, killer=lambda p: killed.append(("other", p.pid)))
n_started = len(started)
oc3.ensure()
oc3.close()
check("OllamaCtl: a server somebody else started is never started again and never stopped", len(started) == n_started and killed == [4242])

# ================================================================================================== 11. dry run, status, incomplete study
rd = new_root()
before = tree_digest(rd)
fd = Fake()
c = mk(rd, fd, ["core", "fs"], None, dry_run=True).run()
check("--dry-run verifies and prices the prompts but writes nothing, makes no call, creates no ledger",
      c == 0 and tree_digest(rd) == before and not fd.calls and not os.path.exists(C.ns_work(SYN, rd)))
rd2 = new_root(breaker=break_request)
check("--dry-run runs the request-identity gate too (a bad request raises PlanAbort)",
      raises(lambda: mk(rd2, Fake(), ["core"], None, dry_run=True).run(), PL.PlanAbort))
pl = mk(rd, fd, ["core"], [0])
ev0 = pl.declare_incomplete("budget stopped the study")
check("declare_incomplete records the stopping rule with counts per pass",
      ev0["event"] == "stopping_rule" and ev0["cells_missing"] == N_A27 and ev0["by_pass"]["core"] > 0)
cleanup(rd, rd2)

# ================================================================================================== 12. the CLI
rcli = new_root()
rc_ = PL.main(["--namespace", SYN, "--root", rcli, "--pass", "core", "--r", "0", "--dry-run", "--types", os.path.join(rcli, "types.json")]) \
    if SYN in C.NAMESPACES else 1
check("CLI: argparse namespace choices come from common, dry-run through main() works", rc_ == 0, rc_)
check("CLI: an unknown pass is an error, not a crash", PL.main(["--namespace", SYN, "--root", rcli, "--pass", "nonsense"]) == PL.EXIT["abort"])
cleanup(rcli)

# ---- A21 E: the X arms abort the planning phase when the Codex session is not the frozen one
rx_ = new_root()
fx_ = Fake()
fx_.codex_model_override = "gpt-5.6-terra"                       # the CLI starts a different model than the one requested for X0 / X+M (gpt-6-sol)
cx_ = mk(rx_, fx_, ["xmain"], [0]).run()
ev_x = event_rows(rx_, name="abort_codex_identity")
check("A21 E: a Codex session that is not the requested model aborts the plan phase (fail fast) with the difference recorded",
      cx_ == PL.EXIT["abort"] and ev_x and event_rows(rx_, name="codex_identity_violation") and "gpt-5.6-terra" in json.dumps(event_rows(rx_, name="codex_identity_violation")[0]))
rows_x = plan_rows(rx_)
check("A21 E: the recorded plan row carries the session facts (requested / served model, identity flags, pinned CLI), never a path or account field",
      rows_x and all(a["extra"].get("requested_model") == "gpt-6-sol" for a in rows_x[-1]["attempts"] if a.get("extra")) and "arash" not in json.dumps(rows_x).lower()
      and any(a["extra"].get("identity_violation") for a in rows_x[-1]["attempts"]))
cleanup(rx_)
ry_ = new_root()
check("A21 E: in test-v1 the Codex pin comes from freeze.json; elsewhere it is off unless given",
      PL.Plan._frozen_codex_pin("rehearsal-v1", ry_) is None and PL.Plan._frozen_codex_pin("test-v1", ry_) is None
      and mk(ry_, Fake(), ["xmain"], [0], codex_pin="codex-cli 0.162.1").codex_pin == "codex-cli 0.162.1")
cleanup(ry_)

C.NAMESPACES.clear()
C.NAMESPACES.update(REAL_NS)
print(f"\n{N['pass']} passed, {N['fail']} failed")
sys.exit(1 if N["fail"] else 0)
