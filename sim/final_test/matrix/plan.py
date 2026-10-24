"""M5 - Phase B planning for the locked final-test matrix (stage M5-plan; contract: SPEC.md s12 M5, s9, s10).

Windows, no simulator. Reads ONLY the Phase A capture journal (it carries the instruction, the post-sweep summary, the
told holding and every memory / FS block) plus the frozen type list. It never opens the dataset, a state file, the
sealed outcomes or a score (SPEC s8; ``_ftm_plan_selftest.py`` checks this statically, at runtime and in a subprocess).

    python sim/final_test/matrix/plan.py --namespace test-v1 --pass core --r 0 [--allow-test] [--dry-run]

What it does, in order
  1. refuses to start unless the capture phase is complete (and unchanged since), the test-item guard holds
     (``C.guard_items``, again on every resume) and the pending-pause rule (A15) is satisfied;
  2. enumerates every planner-call CELL (arm x presentation x replicate r) of the registered design - main (r = 0), REPEAT
     (r = 1, 2; A27 draft: core four only), learning curve (A27 draft: order 0, K = 1, 3, 5), relocation probe, Codex (A27 / A25.3:
     the exploratory models are plan-only) - from the capture, and checks it against the split manifest;
  3. REQUEST-IDENTITY GATE: for every selected cell re-renders the prompt with ``PlannerContract`` and asserts that its
     sha256 equals the captured request sha, then applies the masking gate; any mismatch aborts BEFORE a single call;
  4. visits presentations in a seeded order (``C.plan_blocks``): all cells of one presentation form a BLOCK, arms are
     interleaved inside it, so a stop between blocks leaves every arm balanced;
  5. each paid block runs inside ``ledger.block(scope, worst_case)``: the block's worst case (every first transmission plus
     one repair, at the provider's own pre-call reserve) is held BEFORE it starts (A15; ``BlockRefused`` -> nothing sent);
  6. at BLOCK BOUNDARIES ONLY it runs ``C.cost_guard`` (every 60 billed Opus calls the mean vs 2 x 0.012 = 0.024 USD (A19), and the
     projected completion of the core scope against 30 USD and of the whole call plan against 65 USD, A22). ``pause`` appends
     ``events pause_cost_guard`` and exits 3; the next start refuses until ``--ack-pause "<note>"`` records Arash's decision.
     Resumption is outcome-blind: only spend, token counts and integrity counters are read;
  7. appends one fsync'd ``plan`` record per cell (``ok`` / ``rejected`` are terminal; ``infra`` is re-queued in up to 2 later
     sessions with ``--requeue-infra``; ``ceiling`` is not terminal) and resumes from the journal and the response cache.

Typical order in test-v1 (SPEC s9; exploratory work is cut first): ``--pass core --r 0``, ``--pass core --r 1,2``, ``--pass fs``,
``--pass curve``, ``--pass reloc``, ``--pass codex``; ``--pass gemma`` (free, needs Ollama) can run at any time.  (A23: ``haiku`` and ``sonnet`` are ARCHIVED passes with no test-v1 cell)
Passes are selected by planner family: ``gemma`` = every Gemma cell in every block; ``core`` = S / O core arms in main + repeat;
``fs`` = O+FS; ``curve`` / ``reloc`` = the Opus cells of those blocks; ``codex`` = the four Codex systems.

PLAN FIRST (PF-safety-test, 2026-10-12): `--parallel N` runs the cells of ONE presentation block in a bounded thread pool (anthropic <= N,
codex <= `--codex-parallel`, gemma = 1) behind a shared rate-limit throttle (`throttle.py`; it limits TRANSMISSIONS, so retries and repairs obey the
same cooldown). The cost-guard boundary stays between presentation blocks, exactly as in the serial path (the pool is drained first). Each worker
thread owns its provider objects, its CellCache and its own ScopedLedger object on the same SQLite file; every reservation carries the cell it
belongs to (attempt id = ledger row id). A run killed while calls were in flight leaves open reservations: the next parallel start QUARANTINES
those cells (exit 7) until Arash reconciles (`--reconcile-open`) or explicitly allows replacement attempts (`--allow-replacement`; recorded).
`--pass-id ID` gives a run its own plan journal (plan.ID.jsonl) and response cache, so two passes of a test never share rows or cached answers
(the ledger stays shared, so spend is cumulative under one ceiling). Without `--parallel` nothing of this is active: the serial path is unchanged.

Exit codes: 0 done, 1 integrity abort, 2 planner skipped (no key / no Ollama / no Codex CLI: clean, nothing recorded),
3 paused for a decision, 4 ceiling (block refused or reservation refused), 5 injected interrupt (test hook), 6 incomplete
(infra cells still awaiting a re-queue), 7 quarantine (open reservations of a killed run block their cells).

Python 3.10 syntax only (same as the rest of the matrix). Console output carries ids and counts, never instruction text.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402  (also puts sim/ on sys.path)
import throttle as TH                                                    # noqa: E402
import treatment as T                                                    # noqa: E402  (C4: the ft-v2 switch; inert in ft-v1 namespaces)

import honest_skills as hs                                               # noqa: E402
from planners import (ARMS as PLANNER_ARMS, PROMPT_VERSION, PlannerContract, PlannerRunner,    # noqa: E402
                      ResponseCache, make_planner, prompt_sha256)
from planners.base import CostCeilingExceeded, PlannerSkip, http_get_json, sha256_json  # noqa: E402

PASSES = ("gemma", "xmain", "core", "fs", "curve", "reloc", "haiku", "codex", "sonnet")   # A23: "haiku" and "sonnet" are ARCHIVED passes (only a namespace that opted into the archived arms has cells for them)
MAIN_ARMS = C.MAIN_ARMS                   # incl. the A18 gpt-6-sol arms X0 / X+M (pass "xmain": free, never cut)
RELOC_ARMS = ("G0", "G+M", "O0", "O+M")
CURVE_PLANNERS = ("G", "O")
#: (order, K); K = 0 is the shared '0' arm and (0, 10) is S* (shared with the main matrix), so neither is a cell.
#: A27 (draft): order 0 only, K = 1, 3, 5 (was 11 snapshots over orders 0-2). One list: common.CURVE_SNAPSHOTS.
CURVE_SNAPSHOTS = C.CURVE_SNAPSHOTS
REPEAT_RS = (1, 2)
#: A27 (draft): the REPEAT block (r = 1, 2) is planned for the core four only (common.REPEAT_ARMS); the other main arms have r = 0 only.
REPEAT_ARMS = C.REPEAT_ARMS
CODEX_RS = (0, 1, 2)
MAX_SESSION = 2                           # prereg s13: infra cells are re-queued in up to 2 later sessions
CURVE_RE = re.compile(r"o[0-2]k(?:1|3|5|10)")

#: C4 pilot, USD per presentation including its repair calls (docs/final-test/pilot.md): priors until 10 cells are billed.
PRIOR_USD_PER_CELL = {"O": 0.012, "S": 0.0062, "H": 0.0026}     # A19: C4 pilot x 1.35 for the post-A11 prompts (O = the 0.012 expected mean)
PRIOR_MIN_CELLS = 10
PAUSE_HYSTERESIS = 1.05                   # after --ack-pause a projection pauses again only if it grows > 5 % beyond the accepted one
REPAIR_PROBLEM_CHARS = 600                # the repair turn lists at most a few validator reasons

EXIT = {"ok": 0, "abort": 1, "skip": 2, "pause": 3, "ceiling": 4, "interrupt": 5, "incomplete": 6, "quarantine": 7}
PASS_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")


class CellCache(ResponseCache):
    """Response cache whose key also hashes the CELL (namespace | arm | presentation | r) being planned. The base key has no arm, and an
    empty S* / curve block renders no block at all, so a +M request can be byte-identical to its 0 request; with the base key the arm
    that runs second would silently get the first arm's cached answer (cached = True, 0 USD) and the two arms would share ONE
    stochastic draw. Prereg s7 declares sharing for EXECUTIONS only; every arm x r here is an independent planner call. Resume still
    replays a cell's own answer (same cell, same key). Set ``cell`` before each call; '' = the plain key (dry runs, tools)."""

    def __init__(self, root=None, namespace="default"):
        super().__init__(root, namespace)
        self.cell = ""

    def key(self, *a, **k) -> str:
        base = super().key(*a, **k)
        return sha256_json({"base": base, "cell": self.cell}) if self.cell else base


class PlanAbort(RuntimeError):
    """An integrity violation: fail fast. ``event`` names the events record."""

    def __init__(self, event: str, msg: str):
        super().__init__(msg)
        self.event = event


class _Stop(Exception):
    def __init__(self, code: int, event: str | None = None, **info):
        super().__init__(event or "stop")
        self.code, self.event, self.info = code, event, info


# ================================================================================================ pure: cells
def planner_key(arm: str, ns: str | None = None) -> str:
    """'G' | 'S' | 'O' | 'H' | 'X:<codex model>' for an arm that makes a planner call."""
    p = C.arm_info(arm, ns)["planner"]
    if p in C.PLANNER_SPEC or p.startswith("X:"):
        return p
    raise ValueError(f"arm {arm!r} makes no planner call")


def pass_of(arm: str, block: str, ns: str | None = None) -> str:
    pk = planner_key(arm, ns)
    if pk.startswith("X:"):
        return "xmain" if arm in C.X_MAIN_ARMS else "codex"
    if pk == "G":
        return "gemma"
    if pk == "H":
        return "haiku"
    if pk == "S":
        return "sonnet"                                                      # A23: archived (no test-v1 cell has it)
    if block == "curve":
        return "curve"
    if block == "reloc":
        return "reloc"
    if arm.endswith("+FS"):
        return "fs"
    if arm in C.CORE_ARMS:
        return "core"
    raise ValueError(f"no pass for arm {arm!r} in block {block!r}")


def make_cell(ns: str, arm: str, pres: str, r: int, block: str) -> dict:
    info = C.arm_info(arm, ns)
    return {"key": C.cell_id(ns, "plan", arm, pres, f"r{r}"), "arm": arm, "pres": pres, "r": r, "block": block,
            "scope": C.scope_of(arm, block, ns), "planner": planner_key(arm, ns), "pass": pass_of(arm, block, ns),
            "source": info["prompt"]}


def enumerate_cells(ns: str, canon_pres, repeat_pres, reloc_pres, curve_pres=None) -> list:
    """Every first-pass planner-call cell of the registered design (callplan reconciles the counts). Sharing is explicit:
    r = 0 of a REPEAT presentation IS the main-matrix call; K = 0 and (order 0, K = 10) are shared arms. `curve_pres` (default: every
    REPEAT presentation) are the REPEAT presentations that carry the learning-curve requests; a dev rehearsal without curve snapshots
    passes a smaller set, test-v1 insists on all of them (`Plan._design`)."""
    curve_pres = set(repeat_pres if curve_pres is None else curve_pres)
    out = []
    for p in sorted(canon_pres):
        out += [make_cell(ns, a, p, 0, "main") for a in C.main_arms_for(ns)]
    for p in sorted(repeat_pres):
        for r in REPEAT_RS:                                                  # A27: core four only (r = 0 of every main arm is the main cell)
            out += [make_cell(ns, a, p, r, "repeat") for a in C.main_arms_for(ns) if a in REPEAT_ARMS]
        for pl in CURVE_PLANNERS if p in curve_pres else ():
            out += [make_cell(ns, f"{pl}+M@o{o}k{k}", p, 0, "curve") for o, k in CURVE_SNAPSHOTS]
        for m in C.CODEX_EXPLORATORY:
            for r in CODEX_RS:
                out += [make_cell(ns, a, p, r, "codex") for a in (f"X-{m}0", f"X-{m}+M")]
    for p in sorted(reloc_pres):
        out += [make_cell(ns, a, p, 0, "reloc") for a in RELOC_ARMS]
    keys = [c["key"] for c in out]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate plan cell keys: the design enumeration is not injective")
    return out


def classify(cell: dict, latest: dict, requeue: bool) -> str:
    """-> done | pending | awaiting_requeue. ok / rejected are terminal, infra is terminal after the third session,
    ceiling (and a crash, which leaves no record) is redone."""
    rec = latest.get(cell["key"])
    if rec is None or rec["status"] == "ceiling":
        return "pending"
    if rec["status"] in ("ok", "rejected"):
        return "done"
    if rec["status"] == "infra":
        if int(rec.get("session", 0)) >= MAX_SESSION:
            return "done"
        return "pending" if requeue else "awaiting_requeue"
    return "pending"


def session_for(prev: dict | None) -> int:
    if prev is None:
        return 0
    return int(prev.get("session", 0)) + 1 if prev["status"] == "infra" else int(prev.get("session", 0))


def render_parts(rec: dict, source: str):
    """-> (memory_block, fewshot_block) for a memory source; an empty block renders no block at all."""
    blk = (rec.get("blocks") or {}).get(source) or {}
    text = blk.get("text") or None
    if source == "FS":
        return None, text
    if source == "none":
        return None, None
    return text, None


def summary_of(rec: dict) -> list:
    """The observation lines the planner sees: an ft-v2 capture row carries its own prompt summary (A24 rule over the full list);
    every ft-v1 row is rendered from `summary` exactly as before."""
    if rec.get("treatment") == T.FT2_VERSION:
        return list(rec.get("prompt_summary") or [])
    return rec.get("summary") or []


def render_messages(contract: PlannerContract, rec: dict, source: str) -> list:
    mem, fs = render_parts(rec, source)
    return contract.messages(rec["instruction"], summary_of(rec), rec.get("holding"), memory_block=mem,
                             fewshot_block=fs)


def make_contract(types, ns: str | None = None) -> PlannerContract:
    """ft-v1: the frozen PlannerContract(types) (unchanged). ft-v2 (C4): the D2-filtered list, version ft-v2 (<= 2 repairs)."""
    if ns is not None and T.is_ft2(ns):
        return PlannerContract(T.planner_types(types), tools=hs.TOOLS, version=T.FT2_VERSION)
    return PlannerContract(types, tools=hs.TOOLS)


def none_request_renderer(root: str | None = None, types_path: str | None = None, ns: str | None = None):
    """-> callable(capture record) -> sha256 of the `none`-arm request (no memory, no few-shot block) rendered from that record by the one
    renderer. Used by execute.py's G-iso (A19 component (i)), which must not import the planner layer itself. `ns` (C4): an ft-v2
    namespace renders with its own contract; None / ft-v1 = the frozen one."""
    types, _ = load_types(types_path, root)
    contract = make_contract(types, ns)
    return lambda rec: prompt_sha256(render_messages(contract, rec, "none"))


def cell_worst(prov, contract: PlannerContract, msgs: list) -> float:
    """Worst case of one cell: its first transmission plus ONE repair, each at the provider's own pre-call reserve. The repair
    turn is priced with a failed reply as long as the output budget allows (4 chars per token, reserve counts chars / 2).
    ft-v2 (C4, H-B1 note 2): first transmission + TWO repairs through contract.repair_turn (3 calls; the second repair carries both
    failed replies)."""
    first = prov.reserve_usd(msgs, None)
    big = "x" * (4 * prov.max_output_tokens(None))
    if getattr(contract, "version", "ft-v1") == T.FT2_VERSION:
        # fb-1 (FREEZE-GATES step 2): the problem text is reserved at its API token bound (planners.runner.problem_token_bound("max"),
        # 1,200 tokens) -> a filler of 2 x that many chars, because the pre-call reserve counts chars / 2
        from planners import runner as _R
        pfill = "-" * (2 * _R.problem_token_bound("max"))
        tot, cur = C.block_worst_case(1, first, repairs=False), list(msgs)
        for _ in range(int(T.FT2["repairs"])):
            cur = contract.repair_turn(cur, big, pfill)
            tot += C.block_worst_case(1, prov.reserve_usd(cur, None), repairs=False)
        return tot
    rep = prov.reserve_usd(contract.repair_messages(msgs, big, "-" * REPAIR_PROBLEM_CHARS), None)
    return C.block_worst_case(1, first, repairs=False) + C.block_worst_case(1, rep, repairs=False)


def leak_paths(rec: dict, secrets=()) -> list:
    """Key-leak check of one record (the shared rule, common.leak_paths): structural fields get the registered key patterns; model-generated
    free text only the exact loaded secret, so a harmless hyphenated word in a reply cannot stop a run."""
    return C.leak_paths(rec, secrets)


def load_types(path: str | None, root: str | None = None):
    """-> (types, source). The detector's frozen class list: --types, else the matrix's ``detector_types.json`` (freeze.py, keyed to the
    detector weights sha), else the pilot's copy. A freeze.json that records a different list is an integrity abort."""
    P = C.paths(root)
    cands = [path] if path else [os.path.join(P["matrix_docs"], "detector_types.json"), os.path.join(P["docs"], "pilot", "types.json")]
    for p in cands:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            types = sorted(set(d["types"] if isinstance(d, dict) else d))
            break
    else:
        raise PlanAbort("abort_types", f"no frozen type list found (tried {', '.join(c for c in cands if c)})")
    if os.path.exists(P["freeze"]):
        with open(P["freeze"], encoding="utf-8") as f:
            rec = (json.load(f).get("detector_types") or {}).get("sha256")
        if rec and rec != C.sha256_json(types):
            raise PlanAbort("abort_types", "the type list differs from the one recorded in freeze.json")
    return types, os.path.basename(p)


# ================================================================================================ Ollama control
def _kill_tree(proc):
    """Stops the server process this run started (and its runner children on Windows); nothing else."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()


class OllamaCtl:
    """Checks that Ollama is up (Gemma calls only). Starts it only when allowed, and stops ONLY a server it started itself."""

    def __init__(self, endpoint: str, allow_start: bool = False, popen=subprocess.Popen, get=http_get_json, sleep=time.sleep,
                 wait_s: float = 45.0, killer=None):
        self.endpoint, self.allow_start, self._popen, self._get, self._sleep = endpoint.rstrip("/"), allow_start, popen, get, sleep
        self._kill = killer or _kill_tree
        self.wait_s, self.proc, self.started, self._ok = wait_s, None, False, False
        self._lock = threading.Lock()                                # parallel planning: several worker threads call ensure()

    def alive(self) -> bool:
        try:
            return bool(self._get(self.endpoint + "/api/version", 3.0))
        except Exception:
            return False

    def ensure(self):
        with self._lock:
            self._ensure()

    def _ensure(self):
        if self._ok:
            return
        if not self.alive():
            if not self.allow_start:
                raise PlannerSkip(f"Ollama is not running at {self.endpoint}: start it, or pass --start-ollama")
            env = dict(os.environ, OLLAMA_HOST="0.0.0.0")
            self.proc = self._popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.started = True
            for _ in range(int(self.wait_s)):
                if self.alive():
                    break
                self._sleep(1.0)
            else:
                self.close()
                raise PlannerSkip("Ollama did not come up after being started")
        self._ok = True

    def loaded_models(self) -> list:
        """Names of the models Ollama holds in memory right now (/api/ps); [] on any error."""
        try:
            return [m.get("name") for m in (self._get(self.endpoint + "/api/ps", 5.0) or {}).get("models", [])]
        except Exception:
            return []

    def unload(self, model: str) -> list:
        """Frees the GPU for the simulator: keep_alive 0 for `model`; returns the models still loaded afterwards (best effort, never raises)."""
        try:
            from planners.base import http_post_json
            http_post_json(self.endpoint + "/api/generate", {"Content-Type": "application/json"}, {"model": model, "keep_alive": 0}, 30.0)
        except Exception:
            pass
        for _ in range(10):
            if model not in self.loaded_models():
                break
            self._sleep(0.5)
        return self.loaded_models()

    def close(self):
        if self.started and self.proc is not None:                  # never touch a server somebody else started
            try:
                self._kill(self.proc)
            except Exception:
                pass
        self.proc, self.started = None, False


class _DryLedger:
    """Dry runs touch no database: every method is a no-op, so pricing a block needs no file."""

    def reserve(self, provider, usd):
        return 0

    def settle(self, *a, **k):
        pass

    def release(self, *a, **k):
        pass

    def book(self, *a, **k):
        pass


class _Ctx:
    """What one thread plans with: its own response cache object, its own ledger object (same SQLite file) and its own providers/runners."""

    def __init__(self, cache, ledger, provs):
        self.cache, self.ledger, self.provs = cache, ledger, provs


def gpu_used_mib():
    """GPU memory in use (MiB) from nvidia-smi, or None."""
    import shutil
    if not shutil.which("nvidia-smi"):
        return None
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10).stdout.strip().splitlines()
        return int(o[0]) if o else None
    except Exception:
        return None


# ================================================================================================ the planner
class Plan:
    def __init__(self, ns: str, *, passes=None, rs=None, root: str | None = None, allow_test: bool = False,
                 steal_lock: bool = False, requeue_infra: bool = False, dry_run: bool = False, interrupt_after: int | None = None,
                 ollama: str = "http://127.0.0.1:11434", start_ollama: bool = False, types_path: str | None = None,
                 ack_pause: str | None = None, provider_factory=None, ollama_ctl=None, gpu_sample: bool = True, out=print,
                 manifest: dict | None = None, key_getter=None, pause_response: str | None = None, codex_pin: str | None = None,
                 outcome_blind: bool = False, parallel: int = 1, codex_parallel: int = 2, pass_id: str = "", allow_replacement: bool = False,
                 unload_gemma: bool = False, throttle_kw: dict | None = None):
        self.ns_cfg = C.ns_check(ns)
        self.parallel, self.codex_parallel = max(1, int(parallel)), max(1, int(codex_parallel))
        self.pass_id = pass_id or ""
        if self.pass_id and not PASS_ID_RE.fullmatch(self.pass_id):
            raise ValueError(f"--pass-id must match {PASS_ID_RE.pattern}")
        self.allow_replacement, self.unload_gemma, self.throttle_kw = bool(allow_replacement), bool(unload_gemma), dict(throttle_kw or {})
        self._tl = threading.local()
        self._lock = threading.RLock()                                # journals, latest, counters (parallel planning)
        self._main_thread = threading.current_thread()
        self._stop = None                                             # (kind, exception): drain | abort | skip, first of the highest priority wins
        self._abort_flag = False
        self._interrupted = False
        self._throttles: dict = {}
        self._quarantined: set = set()
        self._quarantine_skipped = 0
        self.pause_response = pause_response
        self.outcome_blind = bool(outcome_blind)
        self.codex_pin = codex_pin if codex_pin is not None else self._frozen_codex_pin(ns, root)
        if ns in ("practice-v1", "practice-v2"):
            raise ValueError(f"{ns} is Gemma practice only: it has no Phase B planning")
        self.ft2 = T.is_ft2(ns)
        self.ident = None
        self.ns, self.root = ns, root
        self.passes = tuple(passes or PASSES)
        bad = [p for p in self.passes if p not in PASSES]
        if bad:
            raise ValueError(f"unknown pass {bad}; use {', '.join(PASSES)}")
        self.rs = tuple(sorted(set(rs))) if rs is not None else None
        self.allow_test, self.steal_lock, self.requeue = allow_test, steal_lock, requeue_infra
        self.dry_run, self.interrupt_after, self.ack_pause, self.out = dry_run, interrupt_after, ack_pause, out
        self.types_path = types_path
        self.ollama = ollama
        self.ctl = ollama_ctl or OllamaCtl(ollama, start_ollama)
        self.factory = provider_factory or make_planner
        self.gpu_sample = gpu_sample
        self.manifest = manifest
        self.key_getter = key_getter or C.thesis_key                  # THESIS_ANTHROPIC_API_KEY only, never ANTHROPIC_API_KEY (A15)
        self.cuts: set = set()
        self.J = C.Journal(os.path.join(C.ns_docs(ns, root), f"plan.{self.pass_id}.jsonl") if self.pass_id else C.journal_path(ns, "plan", root))
        self.E = C.Journal(C.journal_path(ns, "events", root))
        self._ledger_path = os.path.join(C.ns_work(ns, root), "ledger.db")
        self._cache_dir = (os.path.join(C.ns_work(ns, root), "passes", self.pass_id, "cache") if self.pass_id
                           else os.path.join(C.ns_work(ns, root), "cache"))
        self.cap_path = C.journal_path(ns, "capture", root)
        self.recorded = 0
        self.latest: dict = {}
        self.secrets: tuple = ()
        self.ledger = None
        self._provs: dict = {}
        self._gpu = None
        self._obs = None

    # ---------------------------------------------------------------------------------------- events
    def ev(self, event: str, **kw):
        if self.pass_id:
            kw.setdefault("pass_id", self.pass_id)
        with self._lock:
            rec = C.make_record("events", event=event, phase="plan", **kw)
            self.E.append(rec)
        return rec

    def say(self, msg: str):
        self.out(msg)

    # ---------------------------------------------------------------------------------------- prepare
    def _capture_rows(self) -> dict:
        evs = self.E.rows()
        done = [e for e in evs if e.get("event") == "phase_complete" and e.get("phase") == "capture"]
        if not done:
            raise PlanAbort("abort_phase_order", "the capture phase has no phase_complete event: Phase B refuses to start")
        want = done[-1].get("journal_sha")
        if want and want != C.sha256_file(self.cap_path):
            raise PlanAbort("abort_capture_changed", "capture.jsonl changed after its phase_complete event")
        rows = {}
        for r in C.Journal(self.cap_path).rows():
            if r.get("namespace") != self.ns:
                raise PlanAbort("abort_design", f"capture record of namespace {r.get('namespace')!r} inside {self.ns}")
            if int(r.get("capture_run", 1)) == 1:
                rows[r["pres"]] = r                  # run 1 only; the latest record of a presentation wins
        if not rows:
            raise PlanAbort("abort_design", "capture run 1 holds no presentation")
        return rows

    def _manifest(self):
        return self.manifest or C.load_manifest(self.root)

    def _design(self, cap: dict):
        canon_pres = sorted(p for p in cap if C.parse_pres(p)["world"] == "canon")
        reloc_all = sorted(p for p in cap if C.parse_pres(p)["world"] == "reloc")
        # a relocated presentation whose setup failed after the capture re-runs is an exploratory casualty, never a design error:
        # it has no request, so it is excluded from every arm (and from execution / scoring) rather than aborting the study
        self.reloc_inapplicable = [p for p in reloc_all if cap[p].get("status") == "inapplicable"]
        reloc_pres = [p for p in reloc_all if p not in set(self.reloc_inapplicable)]
        m = self._manifest()
        rep_items = set(C.repeat_items(self.ns, m))                  # the namespace's REPEAT items (declared dev subset in rehearsal)
        if self.ns == "test-v1":
            repeat_pres = [p for p in canon_pres if C.parse_pres(p)["item"] in rep_items]
        else:                                                        # a rehearsal follows what Phase A captured as REPEAT
            repeat_pres = [p for p in canon_pres if cap[p].get("repeat") or any(CURVE_RE.fullmatch(s) for s in (cap[p].get("requests") or {}))]
        want_curve = ["o%dk%d" % ok for ok in CURVE_SNAPSHOTS]
        self.curve_pres = [p for p in repeat_pres if all(s in (cap[p].get("requests") or {}) for s in want_curve)]
        ns_items = list(m[self.ns_cfg["items"]])
        want_main = sorted(C.pres_id(i, lg) for i in ns_items for lg in ("en", "fa"))
        if canon_pres != want_main:
            raise PlanAbort("abort_design", f"capture holds {len(canon_pres)} canonical presentations, the {self.ns} set needs "
                                            f"{len(want_main)} (missing {len(set(want_main) - set(canon_pres))}, "
                                            f"extra {len(set(canon_pres) - set(want_main))})")
        if self.ns == "test-v1":
            want_rep = sorted(C.pres_id(i, lg) for i in m["repeat_subset"] for lg in ("en", "fa"))
            want_rel = sorted(C.pres_id(i, lg, "reloc") for i in m["relocation_probe"] for lg in ("en", "fa"))
            if repeat_pres != want_rep or sorted(reloc_all) != want_rel:
                raise PlanAbort("abort_design", "REPEAT / relocation presentations differ from the split manifest")
            if self.curve_pres != repeat_pres:
                raise PlanAbort("abort_design", "test-v1: %d REPEAT presentation(s) lack the learning-curve requests (Phase A must capture "
                                                "all %d snapshots of CURVE_SNAPSHOTS)" % (len(repeat_pres) - len(self.curve_pres), len(CURVE_SNAPSHOTS)))
        elif not set(repeat_pres) <= set(canon_pres) or not {C.parse_pres(p)["item"] for p in reloc_pres} <= \
                {C.parse_pres(p)["item"] for p in canon_pres}:
            raise PlanAbort("abort_design", "REPEAT / relocation presentations lie outside the main set")
        return canon_pres, repeat_pres, reloc_pres

    def _prepare(self):
        self.cap = self._capture_rows()
        items = sorted({C.parse_pres(p)["item"] for p in self.cap})
        C.guard_items(items, self.ns, allow_test=self.allow_test, root=self.root)      # again on every resume (SPEC s7)
        canon_pres, repeat_pres, reloc_pres = self._design(self.cap)
        self.cuts = C.cut_groups(self.E.rows())                       # recorded exploratory cuts (prereg 3.1 / A15), Arash's only
        self.all_cells_full = enumerate_cells(self.ns, canon_pres, repeat_pres, reloc_pres, self.curve_pres)
        self.all_cells = [c for c in self.all_cells_full if C.cut_group(c["arm"], c["pres"]) not in self.cuts]
        self.n_cut = len(self.all_cells_full) - len(self.all_cells)
        self.sel = [c for c in self.all_cells if c["pass"] in self.passes and (self.rs is None or c["r"] in self.rs)]
        self.types, self.types_source = load_types(self.types_path, self.root)   # the freeze check runs on the UNFILTERED list
        self.contract = make_contract(self.types, self.ns)
        self.latest = self.J.latest()
        self._treatment_check()
        self._verify_requests()

    def _treatment_check(self):
        """C4 / Astra Q10: no resume across a treatment change, no mixed journal, and (test-v1) the switch equals freeze.json."""
        if self.ft2:
            if self.ident is None:
                self.ident = T.identity(self.ns, self.root)
            bad = T.journal_problems(self.cap.values(), self.ns, self.ident["sha"]) + \
                T.journal_problems(self.latest.values(), self.ns, self.ident["sha"])
        else:
            bad = T.journal_problems(self.cap.values(), self.ns) + T.journal_problems(self.latest.values(), self.ns)
        fz = {}
        if self.ns == "test-v1" and os.path.exists(C.paths(self.root)["freeze"]):
            with open(C.paths(self.root)["freeze"], encoding="utf-8") as f:
                fz = json.load(f)
        bad += T.entry_problems(self.ns, fz, (self.ident or {}).get("sha"))
        if bad:
            raise PlanAbort("abort_treatment", "treatment mismatch: " + "; ".join(bad[:4]))

    def _verify_requests(self):
        """The request-identity and masking gates: nothing is sent unless every selected prompt equals its captured request."""
        seen = set()
        for c in self.sel:
            k = (c["pres"], c["source"])
            if k in seen:
                continue
            seen.add(k)
            rec = self.cap[c["pres"]]
            want = (rec.get("requests") or {}).get(c["source"])
            if not want:
                raise PlanAbort("abort_design", f"capture of {c['pres']} has no request for source {c['source']!r}")
            msgs = render_messages(self.contract, rec, c["source"])
            if prompt_sha256(msgs) != want:
                raise PlanAbort("abort_request_mismatch", f"re-rendered prompt differs from the Phase A request: "
                                                          f"{c['pres']} / {c['source']} (types file, prompt version or block changed)")
            hit = C.mask_problems("\n".join(m["content"] for m in msgs))
            if hit:
                raise PlanAbort("abort_masking", f"masking gate: {hit} in the prompt of {c['pres']} / {c['source']}")

    # ---------------------------------------------------------------------------------------- providers
    @staticmethod
    def _frozen_codex_pin(ns: str, root) -> str | None:
        """A21 E: in the test namespace the Codex CLI version is pinned to the one freeze.json records; a different CLI skips the planner before any call."""
        if ns != "test-v1":
            return None
        try:
            with open(C.paths(root)["freeze"], encoding="utf-8") as f:
                return json.load(f).get("codex_cli_version")
        except (OSError, ValueError):
            return None

    @staticmethod
    def _family(planner: str) -> str:
        return "gemma" if planner == "G" else ("codex" if planner.startswith("X:") else "anthropic")

    def _throttle(self, family: str):
        with self._lock:
            if family not in self._throttles:
                cap = {"anthropic": self.parallel, "codex": self.codex_parallel, "gemma": 1}[family]
                self._throttles[family] = TH.Throttle(family, cap, stop_check=lambda: self._abort_flag, **self.throttle_kw)
            return self._throttles[family]

    def _ctx(self) -> _Ctx:
        """The calling thread's planning context. The main thread (and the whole serial path) uses the plan's own cache / ledger / providers."""
        if self.parallel <= 1 or threading.current_thread() is self._main_thread:
            return _Ctx(self.cache, self.ledger, self._provs)
        c = getattr(self._tl, "ctx", None)
        if c is None:
            c = _Ctx(CellCache(self._cache_dir, namespace=self.ns),
                     C.ScopedLedger(self._ledger_path, self.ns_cfg["ceilings"], core_done=self._core_done, release_stale=False), {})
            self._tl.ctx = c
        return c

    def _runner(self, planner: str):
        ctx = self._ctx()
        if planner not in ctx.provs:
            spec = C.PLANNER_SPEC[planner] if planner in C.PLANNER_SPEC else "codex:" + planner[2:]
            kw = {"cache": ctx.cache, "ledger": ctx.ledger}
            if planner.startswith("X:") and self.codex_pin:
                kw["pin_cli_version"] = self.codex_pin
            if spec == "gemma":
                kw["endpoint"] = self.ollama
            prov = self.factory(spec, **kw)
            if spec == "gemma" and not self.dry_run:
                prov.check_available = self.ctl.ensure             # called by the provider only on a cache miss
            if self.parallel > 1 and not self.dry_run:             # EVERY transmission (first call, retry, repair) goes through the family throttle
                th = self._throttle(self._family(planner))
                if self._family(planner) == "codex":
                    prov._runner = th.wrap_runner(prov._runner)
                else:
                    prov._transport = th.wrap_transport(prov._transport)
            ctx.provs[planner] = (prov, PlannerRunner(prov, self.contract))
        return ctx.provs[planner]

    def _worst(self, cells: list) -> float:
        tot = 0.0
        for c in cells:
            prov, _ = self._runner(c["planner"])
            tot += cell_worst(prov, self.contract, render_messages(self.contract, self.cap[c["pres"]], c["source"]))
        return round(tot, 6)

    # ---------------------------------------------------------------------------------------- one cell
    def _call_cell(self, c: dict, rec: dict) -> dict:
        msgs = render_messages(self.contract, rec, c["source"])
        sha = prompt_sha256(msgs)
        if sha != rec["requests"][c["source"]]:
            raise PlanAbort("abort_request_mismatch", f"prompt changed since the gate: {c['key']}")
        prov, runner = self._runner(c["planner"])
        ctx = self._ctx()
        ctx.cache.cell = c["key"]                                   # every arm x r is an independent call (CellCache)
        ctx.ledger.cell_tag = (f"{self.pass_id}:" if self.pass_id else "") + c["key"]       # every reservation names its cell (attempt id = ledger row id)
        mem, fs = render_parts(rec, c["source"])
        prev = self.latest.get(c["key"])
        session = session_for(prev)
        t0 = time.time()
        kw = {"observed_types": set(rec.get("observed_types") or [])} if self.ft2 else {}
        try:
            res = runner.plan(rec["instruction"], summary_of(rec), rec.get("holding"), memory_block=mem,
                              fewshot_block=fs, replicate=c["r"], **kw)
        except CostCeilingExceeded as e:
            ctx.ledger.cell_tag = None
            row = self._record(c, rec, prov, sha, session, "ceiling", None, time.time() - t0, note=str(e)[:300])
            self.ev("ceiling", cell=c["key"], scope=c["scope"], detail=str(e)[:300])
            raise _Stop(EXIT["ceiling"], info={"cell": c["key"]}) from None
        if res["attempts"][0]["result"].get("prompt_sha256") != sha:
            raise PlanAbort("abort_request_mismatch", f"the provider sent a different prompt than the captured one: {c['key']}")
        ctx.ledger.cell_tag = None
        status = "ok" if res["ok"] else ("infra" if res["infra_failure"] else "rejected")
        row = self._record(c, rec, prov, sha, session, status, res, time.time() - t0)
        if any((a["result"].get("extra") or {}).get("isolation_violation") for a in res["attempts"]):
            self.ev("codex_isolation_violation", cell=c["key"])
        bad = [p_ for a in res["attempts"] for p_ in ((a["result"].get("extra") or {}).get("identity_problems") or [])]
        if bad:                                                   # A21 E: the session is not the frozen one (model, effort, approval, sandbox, CLI): fail fast
            self.ev("codex_identity_violation", cell=c["key"], problems=bad[:5])
            raise PlanAbort("abort_codex_identity", f"the Codex session of {c['key']} is not the frozen configuration: {'; '.join(bad[:3])}")
        return row

    def _record(self, c, rec, prov, sha, session, status, res, wall, note=""):
        atts = []
        for a in (res["attempts"] if res else []):
            r_ = a["result"]
            ex = r_.get("extra") or {}
            atts.append({"k": a["k"], "stage": r_.get("stage"), "tokens_in": r_["tokens_in"], "tokens_out": r_["tokens_out"],
                         "cost_usd": r_["cost_usd"], "latency_s": r_["latency_s"], "stop_reason": r_.get("stop_reason"),
                         "error": r_.get("error"), "error_kind": r_.get("error_kind"), "cached": r_["cached"],
                         "original_cost_usd": r_.get("original_cost_usd"), "original_latency_s": r_.get("original_latency_s"),
                         "raw": r_.get("raw"), "parse_note": r_.get("parse_note"), "valid": a["validation"]["ok"],
                         "errors": a["validation"].get("errors"), "prompt_sha256": r_.get("prompt_sha256"),
                         "extra": {k: ex.get(k) for k in ("served_model", "request_id", "cache_read_tokens", "thinking_chars",
                                                          "http_status", "unresolved_reservations", "tool_events",
                                                          "isolation_violation", "overhead_tokens_in", "usage_reported",
                                                          "cli_version", "wall_s", "total_duration_ns", "requested_model", "cli_configured_model", "session",
                                                          "identity_violation", "identity_problems", "pin_cli_version")}})
        ok = status == "ok"
        row = C.make_record(
            "plan", key=c["key"], namespace=self.ns, arm=c["arm"], block=c["block"], pres=c["pres"], r=c["r"], session=session,
            request_sha=sha, status=status, plan=C.norm_plan(res["plan"]) if ok else [],
            first_plan=C.norm_plan(res["first_plan"]) if res else [], first_ok=bool(res and res["first_ok"]),
            repaired=bool(res and res["repaired"]), calls=res["calls"] if res else 0,
            cached_calls=res["cached_calls"] if res else 0, tokens_in=res["tokens_in"] if res else 0,
            tokens_out=res["tokens_out"] if res else 0, cost_usd=res["cost_usd"] if res else 0.0, model=prov.model,
            scope=c["scope"], steps=res["plan"] if ok else [], say=res["say"] if res else "",
            normalised=res["normalised"] if res else [], notes=res["notes"] if res else [], decoding=prov.decoding(),
            attempts=atts, item=rec["item"], lang=rec["lang"], level=rec["level"], planner=c["planner"], source=c["source"],
            wall_s=round(wall, 3), prompt_version=self.contract_version, types_sha=self.types_sha, note=note)
        if self.ft2:                                              # C4: ft-v2 rows only (ft-v1 plan rows keep their fields)
            row.update(treatment=T.FT2_VERSION, treatment_id=self.ident["sha"],
                       codes_by_attempt=(res or {}).get("codes_by_attempt"),
                       unknown_types=[(a.get("validation") or {}).get("unknown") for a in (res["attempts"] if res else [])])
        hits = leak_paths(row, self.secrets)
        if hits:
            self.ev("abort_key_leak", cell=c["key"], where=hits)
            raise PlanAbort("abort_key_leak", f"key-like content in the record of {c['key']} ({hits}); not written")
        with self._lock:                                          # one writer at a time: journal line, latest map, counters
            obs = self._observed()                                # (read from the journal before this row exists)
            self.J.append(row)
            self.latest[c["key"]] = row
            self._count(obs, row)
            self.recorded += 1
            self.say(f"  {c['arm']:<14s} {c['pres']:<32s} r{c['r']} {status:<8s} calls={row['calls']}"
                     f"({row['cached_calls']} cached) usd={row['cost_usd']:.4f}")
        return row

    # ---------------------------------------------------------------------------------------- blocks, ledger, guard
    def _run_block(self, pres: str, todo: list):
        """One presentation: its cells run scope by scope. The worst-case hold (A15) is taken PER CELL (first call + one repair at the
        provider's own reserve), not for the whole presentation: a presentation hold (four core cells ~ 1.2 USD) would refuse blocks
        that cost ~0.03 USD and waste the last dollars of a ceiling. Cost-guard pauses stay at presentation boundaries (`_boundary`),
        so arms stay balanced; a ceiling refusal mid-presentation records `partial` and the resume finishes that presentation first."""
        rec = self.cap[pres]
        for scope in ("core", "other", "free"):
            cs = [c for c in todo if c["scope"] == scope]
            if not cs:
                continue
            if scope == "free":
                self._run_cells(cs, rec)
                continue
            for c in cs:
                worst = self._worst([c])
                with contextlib.ExitStack() as st:
                    try:
                        st.enter_context(self.ledger.block(scope, worst))
                    except C.BlockRefused as e:
                        self.ev("block_refused", pres=pres, scope=scope, cell=c["key"], worst_usd=worst, detail=str(e)[:300],
                                partial=any(x["key"] in self.latest for x in self.all_cells if x["pres"] == pres),
                                committed=self.ledger.committed(scope), total=self.ledger.committed())
                        raise _Stop(EXIT["ceiling"], info={"pres": pres}) from None
                    self._run_cells([c], rec)

    def _run_cells(self, cs: list, rec: dict):
        for c in cs:
            self._call_cell(c, rec)
            if self.interrupt_after is not None and self.recorded >= self.interrupt_after:
                self.ev("injected_interrupt", after=self.recorded)
                raise _Stop(EXIT["interrupt"])

    # ---------------------------------------------------------------------------------------- parallel block (plan first)
    def _set_stop(self, kind: str, exc: BaseException):
        """First stop of the highest priority wins (abort > skip > drain). An abort also refuses every NEW transmission; a drain lets cells that already
        started finish and be recorded (never abandoned half-way)."""
        rank = {"drain": 0, "skip": 1, "abort": 2}
        with self._lock:
            if self._stop is None or rank[kind] > rank[self._stop[0]]:
                self._stop = (kind, exc)
            if kind == "abort":
                self._abort_flag = True

    def _raise_stop(self):
        if self._stop is not None:
            kind, exc = self._stop
            self._stop = None
            raise exc

    def _one_cell_parallel(self, c: dict, rec: dict, pres: str):
        ctx = self._ctx()
        if c["scope"] == "free":
            self._call_cell(c, rec)
        else:
            worst = self._worst([c])
            with contextlib.ExitStack() as st:
                try:
                    st.enter_context(ctx.ledger.block(c["scope"], worst))
                except C.BlockRefused as e:
                    self.ev("block_refused", pres=pres, scope=c["scope"], cell=c["key"], worst_usd=worst, detail=str(e)[:300],
                            partial=any(x["key"] in self.latest for x in self.all_cells if x["pres"] == pres),
                            committed=self.ledger.committed(c["scope"]), total=self.ledger.committed())
                    raise _Stop(EXIT["ceiling"], info={"pres": pres}) from None
                self._call_cell(c, rec)
        with self._lock:
            if self.interrupt_after is not None and self.recorded >= self.interrupt_after and not self._interrupted:
                self._interrupted = True
                self.ev("injected_interrupt", after=self.recorded)
                self._set_stop("drain", _Stop(EXIT["interrupt"]))

    def _run_block_parallel(self, pres: str, todo: list):
        """One presentation block in a bounded pool. Cells of one family never exceed its cap (cell-level semaphore); the throttle bounds the
        transmissions again (retries and repairs included). The pool is drained before the cost-guard boundary that follows."""
        rec = self.cap[pres]
        gates = {"anthropic": threading.Semaphore(self.parallel), "codex": threading.Semaphore(self.codex_parallel), "gemma": threading.Semaphore(1)}

        def task(c):
            fam = self._family(c["planner"])
            with gates[fam]:
                if self._stop is not None:                         # a stop was raised: no NEW cell starts (it stays pending)
                    return
                th = self._throttle(fam)
                th.cell_enter()
                try:
                    self._one_cell_parallel(c, rec, pres)
                except TH.Aborted:
                    pass                                          # an abort refused this cell's next transmission; nothing was sent
                except _Stop as s:
                    self._set_stop("drain", s)
                except PlanAbort as e:
                    self._set_stop("abort", e)
                except PlannerSkip as e:
                    self._set_stop("skip", e)
                except BaseException as e:                        # noqa: BLE001  (a crash: fail fast, re-raised in the main thread)
                    self._set_stop("abort", e)
                finally:
                    th.cell_leave()
        pool = self.parallel + self.codex_parallel + 1
        with ThreadPoolExecutor(max_workers=pool, thread_name_prefix="plan") as ex:
            for f in [ex.submit(task, c) for c in todo]:
                f.result()
        self._raise_stop()

    def _quarantine_check(self):
        """Parallel planning only. Open per-call reservations are attempts that may have been billed without a recorded answer (a killed run, an
        ambiguous transport failure). Their cells are QUARANTINED (not re-run) until the reservations are reconciled (audited), or replacement
        attempts are explicitly allowed (recorded, spend stays conservatively reserved)."""
        if self.parallel <= 1 or not any(c["scope"] != "free" for c in self.sel):
            return
        rows = [r for r in self.ledger.open_reservations()["rows"] if r["reserved"] > 0]
        if not rows:
            return
        cells = sorted({r.get("cell") or "?" for r in rows})
        usd = round(sum(r["reserved"] for r in rows), 6)
        if self.allow_replacement:
            self.ev("replacement_attempts_allowed", n=len(rows), reserved_usd=usd, cells=cells[:50], by="--allow-replacement")
            return
        if "?" in cells:
            raise PlanAbort("abort_open_reservations", f"{len(rows)} open paid reservation(s) ({usd:.4f} USD) carry no cell: reconcile them "
                                                       "(--reconcile-open) or allow replacement attempts (--allow-replacement) before planning in parallel")
        self._quarantined = set(cells)
        self.ev("quarantine", n=len(rows), reserved_usd=usd, cells=cells[:50])
        self.say(f"QUARANTINE: {len(rows)} open reservation(s) of {len(cells)} cell(s); those cells wait for --reconcile-open or --allow-replacement")

    def _count(self, obs: dict, r: dict):
        """A21 B: a cell is BILLED when any of its attempts was not a cache replay AND it cost something (cost_usd > 0). The numerator is the
        cell's full cost_usd (first call, repair, billed retries; rejected and infra-failed billed cells included); `costs` keeps the per-cell
        costs in recording order for the recent-window check; `_usd` is the total of the cost column over EVERY paid row (ledger reconciliation)."""
        if r.get("planner") in ("S", "O", "H"):
            self._journal_usd = getattr(self, "_journal_usd", 0.0) + float(r.get("cost_usd") or 0.0)
            if r.get("calls", 0) > r.get("cached_calls", 0) and float(r.get("cost_usd") or 0.0) > 0:
                d = obs.setdefault(r["planner"], {"n": 0, "usd": 0.0, "costs": []})
                d["n"] += 1
                d["usd"] += float(r["cost_usd"])
                d.setdefault("costs", []).append(float(r["cost_usd"]))

    def _observed(self) -> dict:
        """Per planner letter: billed cells and USD, kept incrementally (read from the journal once, at the start)."""
        if self._obs is None:
            self._obs = {}
            self._journal_usd = 0.0
            for r in self.J.rows():
                self._count(self._obs, r)
        return self._obs

    def _other_journals_usd(self) -> float:
        """The ledger is shared by every pass of a namespace, so the ledger-vs-journal reconciliation must count the paid cost of the OTHER plan journals
        too (plan.jsonl and plan.<id>.jsonl except this run's own). Read once: this run holds the namespace's phase lock."""
        if getattr(self, "_other_usd", None) is None:
            tot = 0.0
            d = C.ns_docs(self.ns, self.root)
            try:
                names = [n for n in os.listdir(d) if n == "plan.jsonl" or (n.startswith("plan.") and n.endswith(".jsonl"))]
            except OSError:
                names = []
            seen = {C.sha256_file(self.J.path)} if os.path.exists(self.J.path) else set()
            for n in sorted(names):
                p = os.path.join(d, n)
                if os.path.abspath(p) == os.path.abspath(self.J.path) or C.sha256_file(p) in seen:
                    continue                                   # (a byte-identical copy, e.g. a pass journal staged as plan.jsonl for execution, is one journal)
                seen.add(C.sha256_file(p))
                for r in C.Journal(p).rows():
                    if r.get("planner") in ("S", "O", "H"):
                        tot += float(r.get("cost_usd") or 0.0)
            self._other_usd = tot
        return self._other_usd

    def _mean_costs(self) -> dict:
        obs = self._observed()
        return {k: (obs[k]["usd"] / obs[k]["n"] if obs.get(k, {}).get("n", 0) >= PRIOR_MIN_CELLS else PRIOR_USD_PER_CELL[k])
                for k in PRIOR_USD_PER_CELL}

    def remaining_report(self) -> dict:
        """What the monitor's projected-completion check needs (final_test_callplan.py --remaining): the not-yet-terminal PAID cells per planner
        letter (cut groups excluded, `classify` semantics) and the running mean USD per cell (billed mean once 10 cells are billed, else the
        C4 pilot prior). Reads the capture and plan journals only; sends nothing."""
        self._capture_rows_ready()
        rem = {}
        for c in self.all_cells:
            if c["scope"] == "free" or classify(c, self.latest, self.requeue) == "done":
                continue
            rem[c["planner"]] = rem.get(c["planner"], 0) + 1
        return {"namespace": self.ns, "remaining": rem, "mean_cost": self._mean_costs(), "cut_groups": sorted(self.cuts),
                "cut_cells": self.n_cut}

    def _capture_rows_ready(self):
        """Cells + cuts + latest statuses without the request-identity gate (a pure read of the journals)."""
        self.cap = self._capture_rows()
        canon_pres, repeat_pres, reloc_pres = self._design(self.cap)
        self.cuts = C.cut_groups(self.E.rows())
        full = enumerate_cells(self.ns, canon_pres, repeat_pres, reloc_pres, self.curve_pres)
        self.all_cells = [c for c in full if C.cut_group(c["arm"], c["pres"]) not in self.cuts]
        self.n_cut = len(full) - len(self.all_cells)
        self.latest = self.J.latest()

    def _guard_eval(self) -> dict:
        obs = self._observed()
        mean = self._mean_costs()
        rem_core, rem_all = {}, {}
        for c in self.all_cells:
            if c["scope"] == "free" or classify(c, self.latest, self.requeue) == "done":
                continue
            rem_all[c["planner"]] = rem_all.get(c["planner"], 0) + 1
            if c["scope"] == "core":
                rem_core[c["planner"]] = rem_core.get(c["planner"], 0) + 1
        checked_at, _, accepted = self._guard_state()
        o = obs.get("O", {"n": 0, "usd": 0.0, "costs": []})
        ce = self.ns_cfg["ceilings"]
        out = {}
        try:
            ledger_usd = self.ledger.billed_scope()
        except Exception:                                             # noqa: BLE001  (dry runs hold a stub ledger)
            ledger_usd = None
        for name, spent, rem in (("core", self.ledger.committed("core"), rem_core), ("total", self.ledger.committed(), rem_all)):
            g = C.cost_guard(opus_calls=o["n"], opus_usd=o["usd"], checked_at=checked_at, spent=spent, remaining=rem,
                             mean_cost=mean, ceiling=ce[name], opus_recent=o.get("costs"),
                             ledger_usd=(ledger_usd if name == "total" else None),
                             journal_usd=((getattr(self, "_journal_usd", 0.0) + self._other_journals_usd()) if name == "total" else None))
            if accepted[name] and g["projected"] <= accepted[name] * PAUSE_HYSTERESIS:
                g["reasons"] = [x for x in g["reasons"] if not x.startswith("projected")]
                g["pause"] = bool(g["reasons"])
            rc = g.get("reconcile")
            if rc and not rc["ok"] and rc["gap_usd"] <= float(accepted.get("gap", 0.0)) + rc["tolerance_usd"] + 1e-12:
                g["reasons"] = [x for x in g["reasons"] if not x.startswith("ledger ")]      # Arash accepted this gap at an earlier pause
                g["pause"] = bool(g["reasons"])
            out[name] = g
        return {"scopes": out, "mean": mean, "opus": o, "checked_at": checked_at,
                "new_checked": max(g["checked_at"] for g in out.values()),
                "pause": any(g["pause"] for g in out.values()),
                "reasons": sorted({x for g in out.values() for x in g["reasons"]}),
                "projected": {k: g["projected"] for k, g in out.items()}}

    def _guard_state(self):
        checked_at, pending, accepted = 0, None, {"core": 0.0, "total": 0.0, "gap": 0.0}
        for e in self.E.rows():
            if e["event"] in ("cost_guard_check", "pause_cost_guard"):
                checked_at = max(checked_at, int(e.get("checked_at", 0)))
            if e["event"] == "pause_cost_guard":
                pending = e
            elif e["event"] == "pause_acknowledged" and pending is not None:
                for k in ("core", "total"):
                    accepted[k] = max(accepted[k], float((pending.get("projected") or {}).get(k, 0.0)))
                accepted["gap"] = max(accepted["gap"], float(pending.get("reconcile_gap") or 0.0))
                pending = None
        return checked_at, pending, accepted

    def _boundary(self, pres: str | None):
        """The only place a pause can happen: between blocks, never inside one (arms stay balanced)."""
        if not any(c["scope"] != "free" for c in self.sel):
            return
        g = self._guard_eval()
        if g["pause"]:
            rc = (g["scopes"].get("total") or {}).get("reconcile")
            self.ev("pause_cost_guard", after_block=pres, reasons=g["reasons"], checked_at=g["new_checked"],
                    projected=g["projected"], opus_billed=g["opus"]["n"], opus_usd=round(g["opus"]["usd"], 4),
                    spent={"core": self.ledger.committed("core"), "total": self.ledger.committed()},
                    reconcile_gap=(rc or {}).get("gap_usd"), guard_spec=C.COST_GUARD_SPEC["version"])
            self.say(f"PAUSE for a decision (cost guard): {'; '.join(g['reasons'])}")
            raise _Stop(EXIT["pause"])
        if g["new_checked"] > g["checked_at"]:
            self.ev("cost_guard_check", after_block=pres, checked_at=g["new_checked"], projected=g["projected"],
                    opus_billed=g["opus"]["n"], opus_usd=round(g["opus"]["usd"], 4))

    @staticmethod
    def _ack_problem(ns: str, note, response, outcome_blind: bool) -> str | None:
        """A21 B: what a pause acknowledgement must carry. The response is one of three; in the test namespace the note must say on what spend / integrity
        evidence the decision was made and Arash attests (--outcome-blind) that no outcome, score or plan content was looked at."""
        if response not in C.PAUSE_RESPONSES:
            return f"--ack-pause needs --pause-response {'|'.join(C.PAUSE_RESPONSES)} (A21 B: a pause has three allowed responses)"
        if ns == "test-v1":
            if len(str(note or "").strip()) < 20:
                return "a test-v1 pause decision needs a note of at least 20 characters naming the spend / token / integrity evidence it rests on"
            if not outcome_blind:
                return "a test-v1 pause decision needs --outcome-blind: the decision may use only spend, token, error and integrity counters (A21 B)"
        return None

    def _pause_gate(self):
        if not any(c["scope"] != "free" for c in self.sel):
            return                                                  # a pause is about spend: free passes are not held up
        _, pending, _ = self._guard_state()
        if pending is None:
            return
        if not self.ack_pause:
            self.say("A cost-guard pause is waiting for a decision. Re-run with --ack-pause \"<note>\" once Arash has decided.")
            raise _Stop(EXIT["pause"])
        resp = self.pause_response
        if resp is None and self.ns != "test-v1":
            resp = "continue"
        bad = self._ack_problem(self.ns, self.ack_pause, resp, self.outcome_blind)
        if bad:
            raise ValueError(bad)
        if resp == "cut":
            later = [e for e in self.E.rows() if e.get("event") == "exploratory_cut" and str(e.get("ts", "")) > str(pending.get("ts", ""))]
            if not later:
                raise ValueError("--pause-response cut needs an exploratory cut recorded AFTER the pause (--cut GROUPS --note ...), in the registered order")
        if resp == "stop":
            self.ev("pause_acknowledged", note=self.ack_pause[:300], pause_ts=pending.get("ts"), response="stop")
            self.say("Response 'stop': declare the study incomplete (--declare-incomplete); section 13 applies.")
            raise _Stop(EXIT["pause"])
        self.ev("pause_acknowledged", note=self.ack_pause[:300], pause_ts=pending.get("ts"), response=resp)

    # ---------------------------------------------------------------------------------------- GPU footprint (Gemma path)
    def _start_gpu(self):
        import shutil
        if "gemma" not in self.passes or not self.gpu_sample or not shutil.which("nvidia-smi"):
            return
        stop = threading.Event()
        path = os.path.join(C.ns_docs(self.ns, self.root), "gpu_gemma.jsonl")

        def loop():
            while not stop.is_set():
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"t": round(time.time(), 2), "v": o}) + "\n")
                except Exception:
                    pass
                stop.wait(1.0)
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        self._gpu = (stop, th)

    def _stop_gpu(self):
        if self._gpu:
            self._gpu[0].set()
            self._gpu[1].join(timeout=3)
            self._gpu = None

    # ---------------------------------------------------------------------------------------- run
    def run(self) -> int:
        if self.dry_run:
            self._prepare()
            return self._dry_report()
        lock = os.path.join(C.ns_work(self.ns, self.root), "plan.lock")
        if self.steal_lock and os.path.exists(lock):
            self.ev("lock_stolen", lock="plan")
        code = EXIT["ok"]
        with C.PhaseLock(self.ns, "plan", self.root, steal=self.steal_lock):
            try:
                self._prepare()
                self._open()
                self._pause_gate()
                self._quarantine_check()
                code = self._loop()
            except _Stop as s:
                code = s.code
            except PlanAbort as e:
                self.ev(e.event, msg=str(e)[:400])
                self.say(f"ABORT {e.event}: {e}")
                code = EXIT["abort"]
            except C.TestItemRefused as e:
                self.ev("test_items_refused", msg=str(e)[:300])
                self.say(f"REFUSED: {e}")
                code = EXIT["abort"]
            except PlannerSkip as e:
                self.ev("planner_skipped", reason=str(e)[:300])
                self.say(f"SKIPPED (nothing recorded): {e}")
                code = EXIT["skip"]
            finally:
                self._stop_gpu()
                self._finish_parallel_report()
                self._maybe_unload_gemma()
                self.ctl.close()
        return code

    def _finish_parallel_report(self):
        if self.parallel <= 1 or not self._throttles:
            return
        try:
            self.ev("parallel_report", workers={"anthropic": self.parallel, "codex": self.codex_parallel, "gemma": 1},
                    throttles=[t.report() for t in self._throttles.values()], recorded=self.recorded)
        except Exception:                                          # noqa: BLE001  (a report must never hide the real outcome)
            pass

    def _maybe_unload_gemma(self):
        """Plan first: Gemma planned with the simulator off; free the GPU for the simulator before this process exits (best effort, recorded)."""
        if not self.unload_gemma or self.dry_run or "gemma" not in self.passes or not hasattr(self.ctl, "unload"):
            return
        try:
            before = gpu_used_mib()
            left = self.ctl.unload(PLANNER_ARMS["gemma"][1])
            self.ev("gemma_unloaded", gpu_mib_before=before, gpu_mib_after=gpu_used_mib(), still_loaded=left)
        except Exception:                                          # noqa: BLE001
            pass

    def _open(self):
        self.cache = CellCache(self._cache_dir, namespace=self.ns)
        self.ledger = C.ScopedLedger(self._ledger_path, self.ns_cfg["ceilings"], core_done=self._core_done)
        self.contract_version = self.contract.version if self.ft2 else PROMPT_VERSION
        self.types_sha = C.sha256_json(sorted(self.contract.types if self.ft2 else self.types))
        if any(c["scope"] != "free" for c in self.sel):
            # THESIS_ANTHROPIC_API_KEY only. The provider falls back to ANTHROPIC_API_KEY (Claude Code's own key) when the study's key is
            # absent, and the leak check would not know that secret, so a missing study key is a clean skip BEFORE any call (exit 2).
            self.secrets = (self.key_getter(),)
        decs = {}
        for pk in sorted({c["planner"] for c in self.sel}):
            decs[pk] = C.sha256_json(self._runner(pk)[0].decoding())
        self.ev("plan_start", parallel=self.parallel, codex_parallel=self.codex_parallel, passes=list(self.passes), rs=list(self.rs) if self.rs else None, n_selected=len(self.sel),
                n_pending=sum(1 for c in self.sel if classify(c, self.latest, self.requeue) == "pending"),
                requeue_infra=self.requeue, allow_test=self.allow_test, prompt_version=self.contract_version,
                types_sha=self.types_sha, types_source=self.types_source, system_text_sha=C.sha256_text(self.contract.system_text()),
                ceilings=self.ns_cfg["ceilings"], decoding_sha=decs, cache_namespace=self.ns)
        if "G" in decs and hasattr(self.ctl, "alive") and self.ctl.alive():
            self.ev("gemma_environment", **self._runner("G")[0].environment())     # Ollama version + model digest, for the freeze check
        self._start_gpu()

    def _core_done(self) -> bool:
        """True once every core cell is terminal: from then on the A15 reserve for the core ceiling is released for other scopes.
        (Monotone; evaluated lazily by the ledger, from the in-memory latest-status map.)"""
        if getattr(self, "_core_done_flag", False):
            return True
        core = [c for c in self.all_cells if c["scope"] == "core"]
        if core and all(classify(c, self.latest, self.requeue) == "done" for c in core):
            self._core_done_flag = True
        return bool(getattr(self, "_core_done_flag", False))

    def _order(self) -> list:
        by = {}
        for c in self.sel:
            by.setdefault(c["pres"], []).append(c)
        return C.plan_blocks(sorted(by), lambda p: by[p], self.ns, ",".join(sorted(self.passes)),
                             ",".join(map(str, self.rs)) if self.rs else "all")

    def _loop(self) -> int:
        self._boundary(None)                                       # projected-completion check from the first call
        if self.parallel > 1:
            self._observed()                                       # read the journal once, before any worker thread exists
        for pres, cells in self._order():
            todo = [c for c in cells if classify(c, self.latest, self.requeue) == "pending"]
            if self._quarantined:
                held = [c for c in todo if ((f"{self.pass_id}:" if self.pass_id else "") + c["key"]) in self._quarantined]
                self._quarantine_skipped += len(held)
                todo = [c for c in todo if c not in held]
            if not todo:
                continue
            if self.parallel > 1:
                self._run_block_parallel(pres, todo)
            else:
                self._run_block(pres, todo)
            self._boundary(pres)
        return self._finish()

    def _finish(self) -> int:
        awaiting = [c for c in self.sel if (self.latest.get(c["key"]) or {}).get("status") == "infra"
                    and classify(c, self.latest, True) != "done"]
        for p in self.passes:
            cs = [c for c in self.sel if c["pass"] == p]
            if cs and all(classify(c, self.latest, self.requeue) == "done" for c in cs):
                self.ev("pass_complete", **{"pass": p}, rs=list(self.rs) if self.rs else None, n=len(cs),
                        journal_sha=C.sha256_file(self.J.path))
        if all(classify(c, self.latest, self.requeue) == "done" for c in self.all_cells):
            self._leak_scan()
            sha = C.sha256_file(self.J.path)
            prior = [e for e in self.E.rows() if e["event"] == "phase_complete" and e.get("phase") == "plan"]
            cutl = sorted(self.cuts)
            if self.pass_id:                                   # a test pass never unlocks the execute phase by itself
                self.ev("plan_pass_complete", n=len(self.J.keys()), journal_sha=sha, cells=len(self.all_cells))
            elif not prior or prior[-1].get("journal_sha") != sha or prior[-1].get("cut_groups") != cutl:
                self.ev("phase_complete", n=len(self.J.keys()), journal_sha=sha, cells=len(self.all_cells), cut_groups=cutl,
                        cut_cells=self.n_cut, reloc_inapplicable=list(self.reloc_inapplicable))
        if self._quarantine_skipped:
            self.say(f"{self._quarantine_skipped} cell(s) are quarantined (open reservations of a killed run)")
            return EXIT["quarantine"]
        if awaiting:
            self.say(f"{len(awaiting)} infra cell(s) await a re-queue (--requeue-infra)")
            return EXIT["incomplete"]
        return EXIT["ok"]

    def _leak_scan(self):
        for j in (self.J, self.E):
            for i, r in enumerate(j.rows()):
                if leak_paths(r, self.secrets):
                    raise PlanAbort("abort_key_leak", f"key-like content in {os.path.basename(j.path)} row {i}")
        exact = [s.reveal() for s in self.secrets if len(s.reveal()) >= 8]
        if exact:
            roots = [C.ns_work(self.ns, self.root), C.ns_docs(self.ns, self.root)]
            for base in roots:
                for dp, _, fns in os.walk(base):
                    for fn in fns:
                        try:
                            txt = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
                        except OSError:
                            continue
                        if any(x in txt for x in exact):
                            raise PlanAbort("abort_key_leak", f"the key value is present in {fn}")

    def record_cut(self, groups, note: str):
        """Arash's registered exploratory cut (prereg 3.1 / A15): the groups are cut in the registered order (codex, haiku, curve12,
        reloc); cut cells are not planned, not executed and not scored (never scored 0), and the study stays COMPLETE. Core, Gemma and
        the FS arms cannot be named. The decision is outcome-blind (spend only); the note is recorded verbatim."""
        if not str(note or "").strip():
            raise ValueError("a cut needs a note (who decided, on what spend evidence)")
        with C.PhaseLock(self.ns, "plan", self.root, steal=self.steal_lock):
            return self._record_cut(groups, note)

    def _record_cut(self, groups, note):
        self._prepare()
        bad = C.cut_problem(self.cuts, groups)
        if bad:
            raise ValueError(bad)
        new = sorted(set(groups) | self.cuts, key=C.CUT_ORDER.index)
        n = sum(1 for c in self.all_cells_full if C.cut_group(c["arm"], c["pres"]) in new and
                C.cut_group(c["arm"], c["pres"]) not in self.cuts)
        return self.ev("exploratory_cut", groups=new, newly=sorted(set(groups) - self.cuts), n_cells=n, by="Arash", note=str(note)[:400])

    def reconcile_open(self, note: str):
        """Audited release of stale open per-call reservations (ambiguous transport failures, killed processes). Arash-run."""
        with C.PhaseLock(self.ns, "plan", self.root, steal=self.steal_lock):
            self._open_ledger_only()
            before = self.ledger.open_reservations()
            res = self.ledger.reconcile_open(note)
            self.ev("reservations_reconciled", note=str(note)[:300], n=before["n"], reserved_usd=before["usd"],
                    released=len(res["released"]), settled=len(res["settled"]))
            return before, res

    def _open_ledger_only(self):
        self.ledger = C.ScopedLedger(os.path.join(C.ns_work(self.ns, self.root), "ledger.db"), self.ns_cfg["ceilings"])

    def declare_incomplete(self, reason: str, by: str = "Arash"):
        """Records the stopping rule (prereg s13, 'incomplete study'); score.py --partial needs this event."""
        self._prepare()
        missing = [c["key"] for c in self.all_cells if classify(c, self.latest, self.requeue) != "done"]
        return self.ev("stopping_rule", rule="incomplete_study", reason=reason[:300], by=(by if self.ns != "test-v1" else "Arash"), cells_missing=len(missing),
                       by_pass={p: sum(1 for c in self.all_cells if c["pass"] == p and c["key"] in set(missing)) for p in PASSES})

    # ---------------------------------------------------------------------------------------- reports
    def status(self) -> dict:
        self._prepare()
        out = {}
        for p in PASSES:
            cs = [c for c in self.all_cells if c["pass"] == p]
            d = {"cells": len(cs), "ok": 0, "rejected": 0, "infra": 0, "ceiling": 0, "pending": 0}
            for c in cs:
                rec = self.latest.get(c["key"])
                if rec is None:
                    d["pending"] += 1
                else:
                    d[rec["status"]] += 1
            out[p] = d
        return out

    def _dry_report(self) -> int:
        self.ledger, self.cache = _DryLedger(), CellCache(os.path.join(C.ns_work(self.ns, self.root), "cache"), namespace=self.ns)
        self.say(f"DRY RUN {self.ns}: nothing is written, no call is made")
        total = 0.0
        for p in self.passes:
            cs = [c for c in self.sel if c["pass"] == p]
            pend = [c for c in cs if classify(c, self.latest, self.requeue) == "pending"]
            est = sum(PRIOR_USD_PER_CELL.get(c["planner"], 0.0) for c in pend)
            total += est
            hold = 0.0
            if cs and cs[0]["scope"] != "free":
                by = {}
                for c in pend:
                    by.setdefault(c["pres"], []).append(c)
                hold = max([self._worst(v) for v in by.values()] or [0.0])
            self.say(f"  {p:<6s} cells={len(cs):5d} pending={len(pend):5d} prior-estimate={est:8.2f} USD  largest block hold={hold:.4f} USD")
        self.say(f"  prompts verified against Phase A and masked: {len({(c['pres'], c['source']) for c in self.sel})}; "
                 f"prior-estimate total {total:.2f} USD")
        return EXIT["ok"]


# ================================================================================================ CLI
def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--namespace", required=True, choices=sorted(C.NAMESPACES))
    ap.add_argument("--pass", dest="passes", default="gemma,core",
                    help="comma list of: " + ", ".join(PASSES) + ". The default never mixes scopes (free + core); the exploratory "
                         "passes (fs, curve, reloc, codex) are named explicitly, after the core is done (A15)")
    ap.add_argument("--r", dest="rs", default=None, help="replicates to run, e.g. 0 or 1,2 (default: all)")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--requeue-infra", action="store_true")
    ap.add_argument("--inject-interrupt", type=int, default=None, metavar="N", help="test hook: stop after N recorded cells")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--start-ollama", action="store_true", help="start `ollama serve` if it is down (and stop it afterwards)")
    ap.add_argument("--types", default=None, help="frozen detector type list (json list or {types: [...]}); default: the matrix detector_types.json, else the pilot copy")
    ap.add_argument("--ack-pause", default=None, metavar="NOTE", help="record the decision that ends a cost-guard pause")
    ap.add_argument("--pause-response", default=None, choices=list(C.PAUSE_RESPONSES),
                    help="A21 B: with --ack-pause, what Arash decided: continue (nothing changes), cut (the next group of the registered cut order was "
                         "recorded first), stop (declare the study incomplete). Required in test-v1")
    ap.add_argument("--outcome-blind", action="store_true", help="test-v1 pause acknowledgements: attest that the decision used only spend, token, error and integrity counters")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--declare-incomplete", default=None, metavar="REASON")
    ap.add_argument("--declare-by", default="Arash", help="who decided; anything but Arash is refused in test-v1 (rehearsal runs by a build stage say so)")
    ap.add_argument("--cut", default=None, metavar="GROUPS", help="Arash only: record an exploratory cut, comma list in the order "
                    + ", ".join(C.CUT_ORDER) + " (needs --note)")
    ap.add_argument("--note", default=None, help="the note recorded with --cut")
    ap.add_argument("--reconcile-open", default=None, metavar="NOTE", help="Arash only: release stale open per-call reservations (audited)")
    ap.add_argument("--no-gpu-sample", action="store_true")
    ap.add_argument("--parallel", type=int, default=1, metavar="N", help="plan first: up to N Anthropic cells of one presentation block at once (1 = the serial path)")
    ap.add_argument("--codex-parallel", type=int, default=2, metavar="M", help="plan first: Codex CLI cells at once (account rate limits; default 2)")
    ap.add_argument("--pass-id", default="", help="a separate plan journal (plan.ID.jsonl) and response cache for a test pass; spend stays on the shared ledger")
    ap.add_argument("--allow-replacement", action="store_true", help="parallel mode: re-run cells whose earlier attempt left an open reservation (recorded)")
    ap.add_argument("--unload-gemma", action="store_true", help="when the Gemma pass ends, free the GPU for the simulator (Ollama keep_alive 0; recorded)")
    ap.add_argument("--batch", action="store_true", help="BATCH-A29: plan the Anthropic cells through the Message Batches API (plan_batch.py; same bodies, batch price)")
    ap.add_argument("--batch-hold-usd", type=float, default=10.0, help="batch mode: worst-case hold per chunk of presentations (USD, batch price)")
    ap.add_argument("--batch-max-cells", type=int, default=200, help="batch mode: at most this many cells per chunk")
    ap.add_argument("--batch-poll-s", type=float, default=30.0, help="batch mode: seconds between status polls")
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    rs = [int(x) for x in a.rs.split(",")] if a.rs else None
    try:
        cls, bkw = Plan, {}
        if a.batch:                                                  # BATCH-A29: the Message Batches path (new module; the default path is unchanged)
            import plan_batch
            cls, bkw = plan_batch.BatchPlan, {"batch_hold_usd": a.batch_hold_usd, "batch_max_cells": a.batch_max_cells, "batch_poll_s": a.batch_poll_s}
        pl = cls(a.namespace, passes=[p for p in a.passes.split(",") if p], rs=rs, root=a.root, allow_test=a.allow_test,
                  steal_lock=a.steal_lock, requeue_infra=a.requeue_infra, dry_run=a.dry_run, interrupt_after=a.inject_interrupt,
                  ollama=a.ollama, start_ollama=a.start_ollama, types_path=a.types, ack_pause=a.ack_pause,
                  gpu_sample=not a.no_gpu_sample, pause_response=a.pause_response, outcome_blind=a.outcome_blind, parallel=a.parallel,
                  codex_parallel=a.codex_parallel, pass_id=a.pass_id, allow_replacement=a.allow_replacement, unload_gemma=a.unload_gemma, **bkw)
        if a.cut:
            pl.record_cut([g for g in a.cut.split(",") if g], a.note or "")
            print(json.dumps({"cut_groups": sorted(pl.cuts | set(a.cut.split(","))), "status": pl.status()}, indent=1))
            return EXIT["ok"]
        if a.reconcile_open:
            before, res = pl.reconcile_open(a.reconcile_open)
            print(json.dumps({"open_before": {k: before[k] for k in ("n", "usd")}, "released": len(res["released"])}))
            return EXIT["ok"]
        if a.status or a.declare_incomplete:
            if a.declare_incomplete:
                pl.declare_incomplete(a.declare_incomplete, by=a.declare_by)
            print(json.dumps(pl.status(), indent=1))
            return EXIT["ok"]
        return pl.run()
    except (PlanAbort, C.TestItemRefused, C.PhaseLocked, ValueError) as e:
        print(f"ERROR: {e}")
        return EXIT["abort"]


if __name__ == "__main__":
    sys.exit(main())
