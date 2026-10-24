"""Selftest for PLAN FIRST (stage PF-safety-test): plan.py --parallel, throttle.py, planfirst.py, execute_guard.py, the execute gate.
Offline: no simulator, no network, no real API call, no Ollama. Checks (a), (b), (c) and the offline part of (d) of the safety test.

    python sim/_ftm_planfirst_selftest.py

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
TYPES = ["Apple", "Bread", "Fridge", "Knife", "Mug", "Plate", "Sink"]
REPEAT_ITEMS, RELOC_ITEMS = DEV[:2], DEV[:2]


def set_ceilings(core, total):
    # A23: the synthetic namespace EXPLICITLY opts into the archived Sonnet / Haiku arms (the plan-first stages "haiku" / "sonnet" stay exercised); test-v1 never does
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




# ================================================================================================== PLAN FIRST: fakes with concurrency and rate limits
import random                                               # noqa: E402
import re as _re                                            # noqa: E402  (the test only; plan.py itself never seeds a global RNG)
import threading                                            # noqa: E402
import time                                                 # noqa: E402
import throttle as TH                                       # noqa: E402
import planfirst as PF                                      # noqa: E402
import execute as EX                                        # noqa: E402
from planners.cache import ResponseCache                    # noqa: E402


class FakeTime:
    """A shared fake clock: sleeping advances it (no real waiting); every reader sees the same time."""

    def __init__(self):
        self.t, self.lock = 1000.0, threading.Lock()

    def now(self):
        with self.lock:
            return self.t

    def sleep(self, dt):
        with self.lock:
            self.t += max(dt, 0.001)
        time.sleep(0.0003)


class PFake(Fake):
    """Thread-safe Fake: counts requests in flight per family (the PEAK is what the caps are checked against), naps a random time per call so
    answers come back out of order, records the sha of every OUTBOUND body, can answer the first calls with HTTP 429, and echoes the item number in
    `say` so a cell that received another cell's answer is detected."""

    def __init__(self, delay=0.0, seed=1, clock=None):
        super().__init__()
        self.lock = threading.Lock()
        self.inflight, self.peak = {}, {}
        self.bodies = set()
        self.delay, self.rng = delay, random.Random(seed)
        self.rate_script = []                                # [(status, retry_after)] consumed by the first Anthropic calls
        self.req_times, self.resp429 = [], []                # (fake time at request start, status)
        self.clock = clock

    def reply(self, idx, repair):
        if idx in self.reject or (idx in self.bad_first and not repair):
            return "this is not json"
        return json.dumps({"say": f"ok {idx}", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]})

    def _tick(self, who, idx):
        with self.lock:
            self.n += 1
            self.calls.append((who, idx))
            n = self.n
        if self.crash_on_call is not None and n == self.crash_on_call:
            self.crash_on_call = None
            raise Crash("synthetic crash mid-cell")

    def _enter(self, fam):
        with self.lock:
            self.inflight[fam] = self.inflight.get(fam, 0) + 1
            self.peak[fam] = max(self.peak.get(fam, 0), self.inflight[fam])
        if self.delay:
            with self.lock:
                d = self.delay * (0.2 + 1.6 * self.rng.random())
            time.sleep(d)

    def _leave(self, fam):
        with self.lock:
            self.inflight[fam] -= 1

    def ollama(self, url, headers, body, timeout):
        user = body["messages"][-1]["content"]
        idx = self._idx(user.split("Command:")[-1])
        repair = len(body["messages"]) > 2
        self._enter("gemma")
        try:
            with self.lock:
                self.bodies.add(("gemma", idx, repair, C.sha256_json(body)))
            self._tick("gemma", idx)
            return 200, {}, {"message": {"content": self.reply(idx, repair)}, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 30}
        finally:
            self._leave("gemma")

    def anth(self, url, headers, body, timeout):
        assert headers["x-api-key"] == FAKEKEY
        msgs = body["messages"]
        idx = self._idx(msgs[0]["content"].split("Command:")[-1])
        repair = len(msgs) > 1
        model = body["model"]
        t = self.clock() if self.clock else 0.0
        with self.lock:
            self.req_times.append(t)
            scripted = self.rate_script.pop(0) if self.rate_script else None
        if scripted:
            with self.lock:
                self.resp429.append(t)
            return scripted[0], {"retry-after": str(scripted[1])}, {"error": "synthetic rate limit"}
        self._enter("anthropic")
        try:
            with self.lock:
                self.bodies.add((model, idx, repair, C.sha256_json(body)))
            self._tick(model, idx)
            if self.n in self.err500:
                return 500, {}, {"error": "synthetic server error"}
            if idx in self.fail_idx and (not self.fail_models or any(m in model for m in self.fail_models)):
                return 400, {}, {"error": "synthetic bad request"}
            out = self.opus_out if "opus" in model else 30
            return 200, {}, {"content": [{"type": "text", "text": self.reply(idx, repair)}], "usage": {"input_tokens": 1200, "output_tokens": out},
                             "stop_reason": "end_turn", "model": model, "id": f"msg_{idx}_{repair}"}
        finally:
            self._leave("anthropic")

    def codex(self, cmd, prompt, cwd, env, timeout):
        if "--version" in cmd:
            return types.SimpleNamespace(stdout="codex-cli 9.9.9\n", stderr="", returncode=0)
        self._enter("codex")
        try:
            with self.lock:
                self.bodies.add(("codex", self._idx(prompt.split("Command:")[-1]), False, C.sha256_text(prompt)))     # (the command line holds per-call temp paths)
            return Fake.codex(self, cmd, prompt, cwd, env, timeout)
        finally:
            self._leave("codex")


class PFakeCtl(FakeCtl):
    def __init__(self):
        super().__init__()
        self.unloaded = []

    def unload(self, model):
        self.unloaded.append(model)
        return []


def norm_row(r):
    keep = ("key", "arm", "pres", "r", "status", "plan", "first_plan", "first_ok", "repaired", "calls", "cached_calls", "tokens_in", "tokens_out",
            "cost_usd", "request_sha", "say", "scope", "planner", "model", "normalised", "notes", "item", "lang", "level", "source")
    d = {k: r.get(k) for k in keep}
    d["attempts"] = [(a["k"], a["stage"], a["prompt_sha256"], a["valid"], a["tokens_in"], a["tokens_out"], a["cost_usd"], a["error_kind"])
                     for a in r.get("attempts") or []]
    return d


def rows_by_key(root, ns=SYN, pass_id=""):
    p = os.path.join(C.ns_docs(ns, root), f"plan.{pass_id}.jsonl") if pass_id else C.journal_path(ns, "plan", root)
    return C.Journal(p).latest(), C.Journal(p).rows()


def mkp(root, fake, passes, rs=None, ns=SYN, ctl=None, **kw):
    """A Plan in PARALLEL mode with a throttle that is not slowed by real sleeping."""
    kw.setdefault("throttle_kw", {"poll_s": 0.0005, "base_cooldown_s": 0.05})
    return mk(root, fake, passes, rs=rs, ns=ns, ctl=ctl, **kw)


def misc(fake, **kw):
    for k, v in kw.items():
        setattr(fake, k, v)
    return fake


ALL_PASSES = ["gemma", "xmain", "core", "haiku", "sonnet"]       # A23: the archived passes stay exercised in the opted-in synthetic namespace
BAD_FIRST, REJECT, FAIL = {idx_of(DEV[2]), idx_of(DEV[3])}, {idx_of(DEV[4])}, {idx_of(DEV[5])}
PRINT_FAIL_NAMES = []


def fresh_fake(**kw):
    f = PFake(**kw)
    f.bad_first, f.reject, f.fail_idx, f.fail_models = set(BAD_FIRST), set(REJECT), set(FAIL), ("haiku",)
    return f


# ================================================================================================== P1. serial == parallel
print("\n== P1. the parallel plan phase gives the same requests, validity and plans as the serial path ==")
roots = []
root_s, root_p = new_root(), new_root()
roots += [root_s, root_p]
fk_s = fresh_fake()
fk_p = fresh_fake(delay=0.02)
DONE = (0, PL.EXIT["incomplete"])               # 6 = every cell terminal except infra cells awaiting a re-queue (the fakes inject infra failures on purpose)
cs = mk(root_s, fk_s, ALL_PASSES, rs=[0]).run()
pl_p = mkp(root_p, fk_p, ALL_PASSES, rs=[0], parallel=4, codex_parallel=2)
cp = pl_p.run()
check("serial run finishes (exit 0, or 6 for the injected infra cells)", cs in DONE, cs)
check("parallel run (4 anthropic workers, 2 codex) ends with the same exit code as the serial one", cp == cs, (cs, cp))
ls, rs_ = rows_by_key(root_s)
lp, rp = rows_by_key(root_p)
check("the same set of cells was planned", set(ls) == set(lp) and len(ls) > 40, (len(ls), len(lp)))
diff = [k for k in ls if norm_row(ls[k]) != norm_row(lp.get(k, {}))]
check("every cell row is identical (request sha, status, first_ok, repaired, plan, tokens, cost, attempts) serial vs parallel", not diff, diff[:3])
check("outbound request bytes are identical: same set of (model, item, repair, body sha) on the wire", fk_s.bodies == fk_p.bodies,
      (len(fk_s.bodies), len(fk_p.bodies), list(fk_s.bodies ^ fk_p.bodies)[:2]))
check("the number of provider calls is identical (no cell planned twice, none lost)", len(fk_s.calls) == len(fk_p.calls), (len(fk_s.calls), len(fk_p.calls)))
check("routing: every cell received ITS OWN answer (the echoed item number equals the item of the cell), out of order completion included",
      all(r["status"] != "ok" or r["repaired"] or r["say"] == f"ok {idx_of(r['item'])}" for r in lp.values()),
      [k for k, r in lp.items() if r["status"] == "ok" and not r["repaired"] and r["say"] != f"ok {idx_of(r['item'])}"][:3])
check("statuses cover ok, rejected and infra (the repair / failure paths ran under concurrency)",
      {r["status"] for r in lp.values()} >= {"ok", "rejected", "infra"}, {r["status"] for r in lp.values()})
check("caps: peak Anthropic requests in flight <= 4, and more than one at a time (it really was parallel)",
      1 < fk_p.peak.get("anthropic", 0) <= 4, fk_p.peak)
check("caps: Gemma strictly one request at a time", fk_p.peak.get("gemma") == 1, fk_p.peak)
check("caps: Codex <= 2 at a time", 1 <= fk_p.peak.get("codex", 0) <= 2, fk_p.peak)
check("the serial run never had two requests in flight", max(fk_s.peak.values()) == 1, fk_s.peak)
thr = {t["family"]: t for ev in event_rows(root_p, name="parallel_report") for t in ev["throttles"]}
check("parallel_report records both quantities: active cells and active transmissions (peaks) per family",
      set(thr) >= {"anthropic", "gemma", "codex"} and all("peak_cells" in t and "peak_transmissions" in t for t in thr.values()), thr)
check("transmissions never exceeded the family caps in the throttle's own count", thr["anthropic"]["peak_transmissions"] <= 4 and thr["gemma"]["peak_transmissions"] == 1, thr)

# ---- ledger invariants after a clean parallel run
lg, lrows = ledger_rows(root_p)
check("ledger: no open reservation and no live block hold after a clean parallel run", not [r for r in lrows if r["state"] == "open"], [r for r in lrows if r["state"] == "open"][:2])
rep = PF.attempts_report(SYN, root_p)
check("ledger: settled spend equals the journal's cost, cell by cell (no difference list)", rep["diffs"] == [] and abs(rep["ledger_settled_usd"] - rep["journal_usd"]) < 1e-6, rep["diffs"][:2])
check("ledger: every paid reservation carries its cell (attempt id = row id, cell = plan cell)",
      all(json.loads(r["entry"] or "{}").get("cell") for r in lrows if r["provider"].startswith("anthropic@")), [r for r in lrows if r["provider"].startswith("anthropic@")][:1])
check("ledger: committed spend is under the ceilings", lg.committed() <= C.NAMESPACES[SYN]["ceilings"]["total"])
raw_ok = True
with open(C.journal_path(SYN, "plan", root_p), "rb") as fh:
    for ln in fh.read().split(b"\n"):
        if ln.strip():
            try:
                json.loads(ln.decode("utf-8"))
            except ValueError:
                raw_ok = False
check("journal: every line of the parallel plan journal is a complete JSON record (no torn / interleaved line)", raw_ok)
check("journal: exactly one terminal row per cell", len(rp) == len({r["key"] for r in rp}), (len(rp), len(lp)))
check("journal: no torn-line side file", not os.path.exists(C.journal_path(SYN, "plan", root_p) + ".torn"))

# ================================================================================================== P2. ceilings under concurrency
print("\n== P2. ceilings: parallel workers cannot spend past them, and the resume finishes ==")
root_c = new_root()
roots.append(root_c)
set_ceilings(0.62, 2.0)
_old_boundary = PL.Plan._boundary
PL.Plan._boundary = lambda self, pres: None                  # the cost guard would pause on the projection first; the ledger alone must refuse (as in the serial selftest)
fk_c = fresh_fake(delay=0.005)
pc = mkp(root_c, fk_c, ["core"], rs=[0], parallel=4)
cc = pc.run()
PL.Plan._boundary = _old_boundary
lgc, rc_ = ledger_rows(root_c)
check("a tight ceiling stops the parallel run with exit 4 (ceiling), not a crash", cc == PL.EXIT["ceiling"], cc)
check("committed spend never passed the core ceiling (the holds of all workers are counted)", lgc.committed("core") <= 0.62 + 1e-9, lgc.committed("core"))
check("the refusal is recorded (block_refused)", bool(event_rows(root_c, name="block_refused")))
check("no open reservation survives a ceiling stop (in-flight cells finished, holds released)", not [r for r in rc_ if r["state"] == "open"], [r for r in rc_ if r["state"] == "open"][:2])
done_before = len(rows_by_key(root_c)[0])
set_ceilings(50.0, 100.0)
cc2 = mkp(root_c, fk_c, ["core"], rs=[0], parallel=4).run()
lc2, rc2 = rows_by_key(root_c)
check("after the ceiling is raised the resume completes without repeating a finished cell", cc2 in DONE and len(lc2) > done_before, (cc2, done_before, len(lc2)))
ok_by_key = {}
for r in rc2:
    if r["status"] == "ok":
        ok_by_key[r["key"]] = ok_by_key.get(r["key"], 0) + 1
check("no cell has two ok rows after stop + resume", all(v == 1 for v in ok_by_key.values()), [k for k, v in ok_by_key.items() if v > 1][:2])

# ================================================================================================== P3. interrupt + resume
print("\n== P3. injected interrupt, then resume ==")
root_i = new_root()
roots.append(root_i)
fk_i = fresh_fake(delay=0.01)
ci = mkp(root_i, fk_i, ALL_PASSES, rs=[0], parallel=4, interrupt_after=9, codex_parallel=2).run()
check("the injected interrupt ends the run with exit 5 and leaves recorded work", ci == PL.EXIT["interrupt"] and 9 <= len(rows_by_key(root_i)[0]) < len(lp), (ci, len(rows_by_key(root_i)[0])))
lg_i, rows_i = ledger_rows(root_i)
check("the drain leaves no open reservation (in-flight cells finished and were recorded)", not [r for r in rows_i if r["state"] == "open"], [r for r in rows_i if r["state"] == "open"][:2])
ci2 = mkp(root_i, fk_i, ALL_PASSES, rs=[0], parallel=4, codex_parallel=2).run()
li, ri = rows_by_key(root_i)
check("the resume completes", ci2 == cs, (ci2, cs))
check("interrupted + resumed rows equal the uninterrupted parallel run, cell by cell", all(norm_row(li[k]) == norm_row(lp[k]) for k in lp) and set(li) == set(lp),
      [k for k in lp if k not in li or norm_row(li[k]) != norm_row(lp[k])][:3])
check("idempotent: the interrupted + resumed run made the same number of provider calls as the uninterrupted one (nothing billed twice)",
      len(fk_i.calls) == len(fk_p.calls), (len(fk_i.calls), len(fk_p.calls)))
rep_i = PF.attempts_report(SYN, root_i)
check("ledger after interrupt + resume: settled == journal, no open reservation", rep_i["diffs"] == [] and rep_i["open_total"] == 0, (rep_i["diffs"][:2], rep_i["open_total"]))

# ================================================================================================== P4. a killed run: quarantine
print("\n== P4. a hard kill leaves open reservations; the parallel start quarantines those cells ==")
root_k = new_root()
roots.append(root_k)
fk_k = fresh_fake(delay=0.005)
fk_k.crash_on_call = 12
crashed = False
try:
    mkp(root_k, fk_k, ["core", "haiku"], rs=[0], parallel=4).run()
except Crash:
    crashed = True
check("a crash inside a worker is re-raised in the main thread after the pool drained (fail fast, nothing swallowed)", crashed)
lk, rk = rows_by_key(root_k)
pending_keys = [c["key"] for c in PL.enumerate_cells(SYN, [C.pres_id(i, lg) for i in DEV for lg in ("en", "fa")], [], [], [])
                if c["pass"] in ("core", "haiku") and c["r"] == 0 and c["key"] not in lk]
check("some cells are still pending after the crash", len(pending_keys) > 0, len(pending_keys))
victim = pending_keys[0]
lgk = C.ScopedLedger(os.path.join(C.ns_work(SYN, root_k), "ledger.db"), C.NAMESPACES[SYN]["ceilings"], release_stale=False)
with lgk.use("other"):
    lgk.cell_tag = victim
    stale_rid = lgk.reserve("anthropic", 0.05)                       # what a killed process leaves: a reservation that never resolved
    lgk.cell_tag = None
cq = mkp(root_k, fk_k, ["core", "haiku"], rs=[0], parallel=4).run()
lq, _ = rows_by_key(root_k)
check("restart without a decision: exit 7 (quarantine), the victim cell is NOT re-run, every other cell is", cq == PL.EXIT["quarantine"] and victim not in lq and len(lq) == len(pending_keys) + len(lk) - 1, (cq, len(lq)))
check("the quarantine is recorded as an event", bool(event_rows(root_k, name="quarantine")))
cq2 = mkp(root_k, fk_k, ["core", "haiku"], rs=[0], parallel=4, allow_replacement=True).run()
lq2, _ = rows_by_key(root_k)
check("with --allow-replacement the victim cell is planned and the decision is recorded", cq2 in DONE and victim in lq2 and bool(event_rows(root_k, name="replacement_attempts_allowed")), cq2)
rep_k = PF.attempts_report(SYN, root_k)
check("the stale reservation stays reserved (conservative) and is reported against its cell until it is reconciled", rep_k["open_total"] == 1 and rep_k["cells"][victim]["open_usd"] == 0.05, rep_k["open_total"])
plr = mk(root_k, fk_k, ["core"], rs=[0])
before, res = plr.reconcile_open("selftest: audited release of the killed run's reservation")
check("audited reconcile releases it; afterwards no open reservation and settled == journal", PF.attempts_report(SYN, root_k)["open_total"] == 0 and len(res["released"]) == 1)
# an UNTAGGED open paid reservation (older tooling) must stop a parallel start rather than be guessed at
root_u = new_root()
roots.append(root_u)
fk_u = fresh_fake()
mkp(root_u, fk_u, ["haiku"], rs=[0], parallel=2, interrupt_after=1).run()
lgu = C.ScopedLedger(os.path.join(C.ns_work(SYN, root_u), "ledger.db"), C.NAMESPACES[SYN]["ceilings"], release_stale=False)
with lgu.use("other"):
    lgu.reserve("anthropic", 0.02)
cu = mkp(root_u, fk_u, ["haiku"], rs=[0], parallel=2).run()
check("an untagged open paid reservation aborts the parallel start (exit 1) until it is reconciled or replacement is allowed", cu == PL.EXIT["abort"], cu)
check("the serial path is unchanged by stale reservations (it keeps running, as before)", mk(root_u, fk_u, ["haiku"], rs=[0]).run() in DONE)

# ================================================================================================== P5. rate limits
print("\n== P5. a rate-limit storm: shared cooldown, halved concurrency, nothing lost, nothing leaked ==")
root_r = new_root()
roots.append(root_r)
ft = FakeTime()
fk_r = fresh_fake(delay=0.002, clock=ft.now)
fk_r.fail_idx = set()
fk_r.rate_script = [(429, 3), (429, 3), (529, 3)]          # (at most 3 in a row for any cell, so no cell exhausts the provider's 3 retries)
plr_ = mk(root_r, fk_r, ["core", "haiku"], rs=[0], parallel=4,
          throttle_kw={"clock": ft.now, "sleep": ft.sleep, "base_cooldown_s": 1.0, "jitter": 0.0, "poll_s": 0.05, "restore_after": 20})
cr = plr_.run()
lr, rr = rows_by_key(root_r)
t_first = min(fk_r.resp429)
late = [t for t in fk_r.req_times if t > t_first and t < t_first + 3.0 - 1e-9]
check("every cell finished despite three rate-limit answers (no infra row)", cr in DONE and all(r["status"] != "infra" for r in lr.values()), (cr, {r["status"] for r in lr.values()}))
check("COOLDOWN: no request started inside the 3 s after the first 429 (retries and repairs obey the shared cooldown too)", not late, late[:3])
th = plr_._throttles["anthropic"]
check("the throttle saw three rate-limit answers, halved the limit at least once and never let more than the cap through", th.n_rate_limited == 3 and th.min_limit < 4 and th.peak <= 4, th.report())
check("request count is bounded: calls = billed calls + 3 rate-limited retries (no storm)", len(fk_r.req_times) == len(fk_r.calls) + 3, (len(fk_r.req_times), len(fk_r.calls)))
lgr, lrr = ledger_rows(root_r)
check("429 / 529 reservations were released (unbilled): no open row, settled == journal", not [x for x in lrr if x["state"] == "open"] and PF.attempts_report(SYN, root_r)["diffs"] == [])
check("released rows exist for the rejected transmissions", len([x for x in lrr if x["state"] == "released"]) >= 3)

# --- throttle unit behaviour on a fake clock
ft2 = FakeTime()
t1 = TH.Throttle("x", 4, clock=ft2.now, sleep=ft2.sleep, base_cooldown_s=2.0, jitter=0.0, restore_after=3)
t1.rate_limited(None)
check("throttle: first rate limit -> cooldown base (2 s) and limit 4 -> 2", t1.limit == 2 and abs(t1.cooldown_until - (ft2.now() + 2.0)) < 1e-6)
t1.rate_limited(None)
check("throttle: second in a row doubles the cooldown (4 s) and limit 2 -> 1; never below 1", t1.limit == 1 and abs(t1.cooldown_until - (ft2.now() + 4.0)) < 1e-6)
t1.rate_limited(10)
check("throttle: Retry-After wins over the exponential rule", abs(t1.cooldown_until - (ft2.now() + 10.0)) < 1e-6 and t1.limit == 1)
t0_ = ft2.now()
t1.acquire()
check("throttle: acquire waits out the whole cooldown", ft2.now() - t0_ >= 10.0 - 1e-6)
t1.release()
for _ in range(3):
    t1.ok()
check("throttle: restores one step of concurrency after `restore_after` clean calls", t1.limit == 2)
t1.stop_check = lambda: True
check("throttle: an abort refuses a NEW transmission", raises(lambda: t1.acquire(), TH.Aborted))
t3 = TH.Throttle("y", 2, clock=ft2.now, sleep=ft2.sleep, jitter=0.0)
run_ = t3.wrap_runner(lambda cmd, p, cwd, env, to: types.SimpleNamespace(stdout="", stderr="stream error: 429 Too Many Requests, usage limit", returncode=1))
run_(["codex", "exec"], "p", None, {}, 5)
check("throttle: a Codex CLI failure that says 'rate limit' / 429 triggers the same cooldown", t3.n_rate_limited == 1 and t3.limit == 1)
run_(["codex", "--version"], None, None, {}, 5)
check("throttle: the `--version` probe is not a transmission", t3.n_sent == 1)

# ================================================================================================== P6. the cache under threads
print("\n== P6. the response cache under threads ==")
rc_dir = tempfile.mkdtemp(prefix="ftm_cache_")
roots.append(rc_dir)
cache = ResponseCache(rc_dir, "ns")
key = "ab" + "0" * 62
threads = [threading.Thread(target=lambda i=i: cache.put(key, {"raw": f"r{i}"}, {"m": i})) for i in range(16)]
[t.start() for t in threads]
[t.join() for t in threads]
check("cache: 16 threads writing the SAME key leave one valid entry and no .tmp file", cache.get(key) is not None
      and not [f for dp, _, fns in os.walk(rc_dir) for f in fns if f.endswith(".tmp")])
real_replace = os.replace
boom = {"n": 0}


def bad_replace(a, b):
    boom["n"] += 1
    raise OSError("synthetic: disk full while publishing the cache entry")


key2 = "cd" + "1" * 62
os.replace = bad_replace
try:
    try:
        cache.put(key2, {"raw": "x"}, {})
    except OSError:
        pass
finally:
    os.replace = real_replace
check("cache: an interrupted write is NOT a hit (the half-written entry is only a .tmp file)", cache.get(key2) is None and boom["n"] == 1)
cache.put(key2, {"raw": "x"}, {})
check("cache: the same key can be written afterwards (a leftover .tmp does not block it)", cache.get(key2) is not None)

# ================================================================================================== P7. the real Opus / Sonnet / Codex adapters under the pool
print("\n== P7. the real provider adapters (Opus, Sonnet, Haiku, Codex, Ollama) under concurrency, only the wire replaced ==")
models_seen = {w for (w, _) in fk_p.calls}
check("the pool exercised the real Anthropic adapters for Opus, Sonnet and Haiku, the Ollama adapter and the Codex adapter",
      any("opus" in m for m in models_seen) and any("sonnet" in m for m in models_seen) and any("haiku" in m for m in models_seen)
      and "gemma" in models_seen and "codex" in models_seen, sorted(models_seen))
check("repairs happened: an invalid first reply was repaired inside the pool (calls == 2 on a repaired row)", any(r["repaired"] and r["calls"] == 2 for r in lp.values()))
codex_rows = [r for r in lp.values() if r["planner"].startswith("X:")]
check("Codex cells carry the session identity facts and no isolation violation under concurrency",
      codex_rows and all(not (a.get("extra") or {}).get("identity_violation") for r in codex_rows for a in r["attempts"]), len(codex_rows))
check("no secret leaked into the plan journal, events or ledger audit under concurrency",
      FAKEKEY not in open(C.journal_path(SYN, "plan", root_p), encoding="utf-8").read()
      and FAKEKEY not in open(C.journal_path(SYN, "events", root_p), encoding="utf-8").read()
      and FAKEKEY not in open(os.path.join(C.ns_work(SYN, root_p), "ledger.db.jsonl"), encoding="utf-8").read())

# ================================================================================================== P8. pass ids: separate journal and cache
print("\n== P8. --pass-id: a pass never shares rows or cached answers with another ==")
root_q = new_root()
roots.append(root_q)
fk_a, fk_b = fresh_fake(), fresh_fake()
ca = mkp(root_q, fk_a, ["haiku"], rs=[0], parallel=3, pass_id="serialish").run()
cb = mkp(root_q, fk_b, ["haiku"], rs=[0], parallel=3, pass_id="fresh").run()
la, _ = rows_by_key(root_q, pass_id="serialish")
lb, _ = rows_by_key(root_q, pass_id="fresh")
check("two passes in one namespace have their own journals (plan.<id>.jsonl) and the default journal stays untouched", ca in DONE and cb in DONE and la and lb and not os.path.exists(C.journal_path(SYN, "plan", root_q)), (ca, cb, len(la), len(lb)))
check("the second pass really called the provider again (its own cache), not a replay", len(fk_b.calls) == len(fk_a.calls) and all(r["cached_calls"] == 0 for r in lb.values()))
check("a pass-id never writes a phase_complete (it cannot unlock execute); it writes plan_pass_complete only",
      not [e for e in event_rows(root_q) if e["event"] == "phase_complete" and e.get("phase") == "plan"])
lgq = C.ScopedLedger(os.path.join(C.ns_work(SYN, root_q), "ledger.db"), C.NAMESPACES[SYN]["ceilings"], release_stale=False)
check("the ledger is SHARED across passes (spend accumulates under one ceiling)", abs(lgq.billed_scope() - PF.attempts_report(SYN, root_q, "serialish")["ledger_settled_usd"] - PF.attempts_report(SYN, root_q, "fresh")["ledger_settled_usd"]) < 1e-6)
check("a bad pass id is refused", raises(lambda: mkp(root_q, fk_a, ["haiku"], pass_id="../x"), ValueError))
frozen_root = new_root()
roots.append(frozen_root)
shutil.copytree(os.path.join(C.ns_work(SYN, root_q), "passes", "serialish"), os.path.join(C.ns_work(SYN, frozen_root), "passes", "frozen"))
fk_f = fresh_fake()
cf = mkp(frozen_root, fk_f, ["haiku"], rs=[0], parallel=3, pass_id="frozen").run()
lf, _ = rows_by_key(frozen_root, pass_id="frozen")
check("FROZEN RESPONSES: feeding a serial pass's cache to a parallel pass replays every answer (only the never-cached transport failures are called again) and gives the same plans",
      cf in DONE and len(fk_f.calls) == sum(1 for r in la.values() if r["status"] == "infra")
      and all(norm_row(lf[k])["plan"] == norm_row(la[k])["plan"] and norm_row(lf[k])["status"] == norm_row(la[k])["status"] for k in la), len(fk_f.calls))

# ================================================================================================== P9. planfirst orchestrator + execute gate
print("\n== P9. planfirst.py: preflight, stage order, completion record, the execute gate ==")


def pf_factory(root, fake, ctl):
    def f(n, **kw):
        passes, rs = kw.pop("passes"), kw.pop("rs", None)
        kw.pop("out", None)
        kw.pop("root", None)
        kw.pop("allow_test", None)
        kw.setdefault("throttle_kw", {"poll_s": 0.0005})
        return PL.Plan(n, passes=passes, rs=rs, root=root, provider_factory=make_factory(fake), ollama_ctl=ctl, gpu_sample=False,
                       key_getter=lambda: Secret(FAKEKEY), out=lambda m: None, types_path=os.path.join(root, "types.json"), **kw)
    return f


root_f = new_root()
roots.append(root_f)
ctl_f, fk_pf = PFakeCtl(), fresh_fake()
fk_pf.fail_idx = set()                                    # no injected infra failure here: a stage with infra cells ends 'incomplete' and stops the sequence
msgs = []
rc = PF.run(SYN, stages=["gemma", "haiku"], parallel=3, root=root_f, sim_check=lambda: (True, "pgrep thor-Linux64 -> 1"), out=msgs.append,
            plan_factory=pf_factory(root_f, fk_pf, ctl_f))
check("preflight: a running simulator refuses plan-first (exit 8) before any call", rc == PF.REFUSED and not fk_pf.calls and bool(event_rows(root_f, name="plan_first_refused")), rc)
C.Journal(C.journal_path(SYN, "exec", root_f)).append({"schema": "x", "key": "k", "status": "done"})
rc = PF.run(SYN, stages=["gemma", "haiku"], parallel=3, root=root_f, sim_check=lambda: (False, "none"), out=msgs.append, plan_factory=pf_factory(root_f, fk_pf, ctl_f))
check("preflight: an existing execution row refuses plan-first (no execution before planning is complete)", rc == PF.REFUSED and not fk_pf.calls, rc)
os.remove(C.journal_path(SYN, "exec", root_f))
check("stage names are validated", raises(lambda: PF.run(SYN, stages=["nonsense"], root=root_f, sim_check=lambda: (False, "")), ValueError))
check("A23 the registered stage order is free, core r0, core r1+2, FS, curve, reloc, Codex (the archived haiku / sonnet stages are valid names but not registered)",
      PF.REGISTERED_ORDER == ("free", "core0", "core12", "fs", "curve", "reloc", "codex") and {"haiku", "sonnet"} <= set(PF.STAGES) and not ({"haiku", "sonnet"} & set(PF.REGISTERED_ORDER)))
rc = PF.run(SYN, stages=["gemma", "haiku"], parallel=3, root=root_f, sim_check=lambda: (False, "none"), out=msgs.append, plan_factory=pf_factory(root_f, fk_pf, ctl_f))
check("a partial design without a stopping rule ends 'incomplete' (exit 6) and the execute gate stays shut", rc == PL.EXIT["incomplete"], (rc, msgs[-3:]))
ev_f = C.Journal(C.journal_path(SYN, "events", root_f)).rows()
check("events: plan_first_start, one plan_first_stage per stage in order, plan_first_complete(scope incomplete)",
      [e["event"] for e in ev_f if e["event"] in ("plan_first_start", "plan_first_stage", "plan_first_complete")] == ["plan_first_start", "plan_first_stage", "plan_first_stage", "plan_first_complete"]
      and [e["stage"] for e in ev_f if e["event"] == "plan_first_stage"] == ["gemma", "haiku"]
      and [e for e in ev_f if e["event"] == "plan_first_complete"][-1]["scope"] == "incomplete", [(e["event"], e.get("stage"), e.get("code")) for e in ev_f if e["event"].startswith("plan_first")])
check("Gemma was unloaded from the GPU when its stage ended (and only then)", ctl_f.unloaded and set(ctl_f.unloaded) == {__import__("planners").ARMS["gemma"][1]}, ctl_f.unloaded)
check("the unload is recorded as an event", bool(event_rows(root_f, name="gemma_unloaded")))
p_path = C.journal_path(SYN, "plan", root_f)
check("EXECUTE GATE: plan-first used + completion scope incomplete -> execution refused", EX.plan_first_problem(ev_f, p_path) is not None)
pl_decl = pf_factory(root_f, fk_pf, ctl_f)(SYN, passes=["gemma"], rs=None)
pl_decl.declare_incomplete("selftest: dev rehearsal, only the Gemma and Haiku passes", by="selftest")
msgs.clear()
n_calls_before = len(fk_pf.calls)
rc = PF.run(SYN, stages=["gemma", "haiku"], parallel=3, root=root_f, sim_check=lambda: (False, "none"), out=msgs.append, plan_factory=pf_factory(root_f, fk_pf, ctl_f))
ev_f = C.Journal(C.journal_path(SYN, "events", root_f)).rows()
comp = [e for e in ev_f if e["event"] == "plan_first_complete"][-1]
check("with a recorded stopping rule the completion scope is authorized_partial (exit 0), bound to the manifest and the journal sha",
      rc == 0 and comp["scope"] == "authorized_partial" and comp["journal_sha"] == C.sha256_file(p_path) and len(comp["manifest_sha"]) == 64
      and comp["cells_total"] > comp["cells_terminal"] > 0 and comp["open_reservations"] == 0, comp)
check("the second run planned nothing new (all cells were already terminal): resume is idempotent", len(fk_pf.calls) == n_calls_before)
check("EXECUTE GATE: a completion bound to the CURRENT plan journal opens it", EX.plan_first_problem(ev_f, p_path) is None)
C.Journal(p_path).append({"schema": "x", "key": "late", "status": "ok"})
check("EXECUTE GATE: a plan journal that changed after the completion record shuts it again", EX.plan_first_problem(ev_f, p_path) is not None)
check("EXECUTE GATE: a namespace that never used plan-first is not affected", EX.plan_first_problem([{"event": "phase_complete"}], p_path) is None)
check("EXECUTE GATE: a stage-local phase_complete does not unlock a plan-first namespace",
      EX.plan_first_problem([{"event": "plan_first_start"}, {"event": "phase_complete", "phase": "plan", "journal_sha": C.sha256_file(p_path)}], p_path) is not None)
# simulator probe parsing (no real WSL call)
fake_run = lambda out: (lambda *a, **k: types.SimpleNamespace(stdout=out, stderr="", returncode=0))
check("simulator probe: WSL not running -> off", PF.simulator_running(run=fake_run("Windows Subsystem for Linux Distributions:\r\n".encode("utf-16-le")), platform="win32")[0] is False)
seq = iter([types.SimpleNamespace(stdout="Ubuntu-22.04\r\n".encode("utf-16-le"), stderr="", returncode=0), types.SimpleNamespace(stdout="1\n", stderr="", returncode=0)])
check("simulator probe: distro running and thor-Linux64 found -> on", PF.simulator_running(run=lambda *a, **k: next(seq), platform="win32")[0] is True)
check("simulator probe: an error is treated as running (never guess 'off')", PF.simulator_running(run=lambda *a, **k: (_ for _ in ()).throw(OSError("x")), platform="win32")[0] is True)

# ================================================================================================== P10. execute reads cached plans only
print("\n== P10. the execute phase cannot plan: guard, missing and corrupt plan rows ==")
gpy = os.path.join(HERE, "final_test", "matrix", "execute_guard.py")
snippet = r"""
import json, os, socket, subprocess, sys, threading, urllib.request
os.environ["THESIS_ANTHROPIC_API_KEY"] = "x"; os.environ["OPENAI_API_KEY"] = "y"
sys.path.insert(0, sys.argv[1])
import execute_guard as G
G.install()
out = {"keys_left": [k for k in G.KEY_VARS if k in os.environ]}
def attempt(name, fn):
    try:
        fn(); out[name] = "NOT BLOCKED"
    except G.GuardViolation:
        out[name] = "blocked"
    except Exception as e:
        out[name] = "other:" + type(e).__name__
from planners import base as pb
attempt("provider_plan", lambda: pb.PlannerProvider.plan(object(), [], {}))
attempt("http_post", lambda: pb.http_post_json("https://api.anthropic.com/v1/messages", {}, {}, 1))
attempt("urlopen_external", lambda: urllib.request.urlopen("https://api.anthropic.com/", timeout=1))
attempt("socket_external", lambda: socket.socket().connect(("8.8.8.8", 443)))
attempt("codex", lambda: subprocess.Popen(["codex", "--version"]))
attempt("ollama", lambda: subprocess.Popen(["ollama", "serve"]))
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
try:
    c = socket.socket(); c.connect(("127.0.0.1", srv.getsockname()[1])); out["loopback"] = "allowed"; c.close()
except Exception as e:
    out["loopback"] = "other:" + type(e).__name__
rep = G.report(); out["blocked_count"] = rep["blocked"]; out["loopback_connects"] = rep["loopback_connects"]
print(json.dumps(out))
"""
sub = subprocess.run([sys.executable, "-I", "-c", snippet, os.path.join(HERE, "final_test", "matrix")], capture_output=True, text=True, cwd=HERE, env={**os.environ, "PYTHONPATH": HERE})
try:
    g = json.loads(sub.stdout.strip().splitlines()[-1])
except Exception:
    g = {"error": (sub.stdout + sub.stderr)[-400:]}
check("execute guard: provider.plan, HTTP helpers, urlopen to an external host, an external socket, codex and ollama are all blocked",
      all(g.get(k) == "blocked" for k in ("provider_plan", "http_post", "urlopen_external", "socket_external", "codex", "ollama")), g)
check("execute guard: loopback (the simulator) stays allowed and the planner credentials are removed from the environment",
      g.get("loopback") == "allowed" and g.get("keys_left") == [] and g.get("blocked_count") == 6, g)
r_ = EX.build_units(ns=SYN, captures={}, plans={}, items={}, snap_sha=lambda s: "x", manifest=C.load_manifest(), p_arms=())
check("execute: with no plan rows there is nothing to execute and nothing is planned (no unit appears out of thin air)", r_["units"] == {} and r_["unexecuted"] == [])
bad = tempfile.mkdtemp(prefix="ftm_corrupt_")
roots.append(bad)
jp = os.path.join(bad, "plan.jsonl")
open(jp, "w", encoding="utf-8").write('{"key":"a","status":"ok"}\nTHIS IS NOT JSON\n{"key":"b","status":"ok"}\n')
check("execute: a corrupt plan journal is corruption (fail fast), never silently skipped", raises(lambda: C.Journal(jp).rows(), C.JournalCorrupt))
src_new = {f: open(os.path.join(HERE, "final_test", "matrix", f), encoding="utf-8").read() for f in ("planfirst.py", "throttle.py", "execute_guard.py")}
check("static: the plan-first modules name no dataset / gold / sealed outcome / score path",
      not [(f, w) for f, s in src_new.items() for w in ("general.json", "outcome.sealed", "score.jsonl", "gold_of", "h.scoring") if w in s])
check("static: the plan-first modules contain no key literal and read no key from the environment (execute_guard only REMOVES them)",
      not [f for f, s in src_new.items() if _re.search(r"sk-[A-Za-z0-9_-]{8,}", s) or "environ[" in s
           or [m for m in _re.findall(r"environ\.get\(([^)]*)\)", s) if m.strip("\"'") != "EXEC_GUARD_REPORT"]])

# ================================================================================================== done
for r_ in roots:
    shutil.rmtree(r_, ignore_errors=True)
C.NAMESPACES.clear()
C.NAMESPACES.update(REAL_NS)
print(f"\n_ftm_planfirst_selftest: {N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(1 if N["fail"] else 0)
