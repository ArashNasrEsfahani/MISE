"""Console v2 -- the honest agent inside the web console: session, execution lease, job controller, simulator port.

HARNESS code, not decision code. Every decision is still made by the step-4 modules (``honest_skills`` over
``honest.HonestAgent``, planned by ``step4_gemma``); this module only schedules them on the console's single
simulator thread, lets the operator stop them, and copies what they did into plain dicts for the browser.
Spec: docs/console-v2-spec.md section 2 (backend contract) + the backend-stage Astra review in
docs/console-v2-progress.md.

How it is wired (webapp.py injects every outside dependency, so the offline selftest can inject fakes):

* ``ThorPort`` -- what ``HonestAgent`` is given instead of the raw simulator agent: exactly four public names
  (``do``, ``frame``, ``depth_frame``, ``event``) and no ``__getattr__``. ``event`` exists only because
  HonestAgent's audited adapter region reads it; the static guard (sim/_step4_rules_selftest.py) keeps every
  other module away from it. ``do`` checks for a stop / an abort BEFORE a primitive is sent -- never between the
  primitive and HonestAgent's bookkeeping, so a successful pickup is always recorded -- then sends it, then calls
  the console's frame hook (whose failures are swallowed: a display problem never erases an action result).
* ``Lease`` -- one execution lease for every job that drives the simulator (owners: oracle-cmd, honest, eval,
  scan, explore, mode-switch, reset). A simulator-bound lease whose generation is no longer current has expired.
* ``Console`` -- the honest session. Three lifetimes are kept apart: the WORLD (``world_gen``; any reload, reset,
  shuffle or simulator restart starts a new one -- ``_new_world`` is the single owner of that counter), the MEMORY
  EPISODE (object memory + action history; carried between instructions, restarted by Forget, a new world or a
  tainted run) and the INSTRUCTION (its own audit episode, plan and step results).

Worker contract: every job built here runs on the console's simulator thread and never lets an exception escape
except a failure raised by the simulator call itself (so the supervisor still restarts a dead simulator exactly as
before). A stale job -- one whose simulator generation or world is no longer current -- is dropped, not applied.
Cancellation (``HonestCancelled``) and aborts (``HonestAborted``: a render glitch, a map mismatch) derive from
BaseException so no broad ``except Exception`` inside the skills or the chooser can swallow them.

Scoring is report-only: the goal check of a picked dataset item runs inside ``h.scoring()`` and its result is
stored on the instruction for display; it never reaches memory, the planner prompt or the chooser.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import io
import json
import os
import threading
import time
import uuid
from typing import Callable, Optional

import honest
import honest_map as hm
import honest_skills as hs
import objmem
import step4_gemma as sg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "step4")
SCENE = "FloorPlan1"
RUNS_FILE = "console_runs.jsonl"
MEMORY_FILE = "console_memory.db"

MAX_HISTORY = 20            # instructions kept in the payload history
MAX_EVENTS = 12             # events shown per step
MAX_TRACKS = 200
MAX_PLACES = 60
EP_PRIM_CAP = 6000          # logical primitives one instruction may send (was 3000; the exhaustive search has
#                             its own 2500 per step -- 2026-10-07 search fix)
PLAN_TIMEOUT = 60.0         # Ollama HTTP timeout per planner call (plan, repair)
PLAN_HARD_TIMEOUT = 2 * PLAN_TIMEOUT + 15.0
CHOOSER_HTTP_TIMEOUT = 30.0
CHOOSER_HARD_TIMEOUT = 35.0
WORK_TIMEOUT = 1800.0       # a worker job that never finishes (and no restart was seen): give up waiting
#                             (was 600 s; an exhaustive search shown smoothly can take minutes)
SEARCH_MODE = "exhaustive"  # console policy (2026-10-07, Arash): search everything; the s5 runner keeps "bounded"
TIDY = True                 # console policy: close what the agent opened

OWNERS = ("oracle-cmd", "honest", "eval", "scan", "explore", "mode-switch", "reset")
ACTIVE = ("prepare", "sweep", "planning", "executing", "checking", "stopping")
FINAL = ("done", "failed", "rejected", "cancelled")
STEP_STATUS = ("pending", "running", "ok", "failed", "skipped", "cancelled")
STEP4_MODULES = ("honest.py", "honest_map.py", "objmem.py", "honest_skills.py", "step4_gemma.py", "honest_console.py")
ASSUMPTIONS = ["pose from the simulator (proprioception)", "walked map file sim/step4/FloorPlan1.map.json"]
#: exceptions the simulator call may raise that are argument problems, not a dead simulator
_NOT_SIM_DEATH = (ValueError, TypeError, KeyError)


class HonestCancelled(BaseException):
    """The operator pressed Stop. Raised before the next primitive (or while waiting for an LLM reply)."""


class HonestAborted(BaseException):
    """The instruction cannot go on through no choice of the operator (render glitch, map mismatch)."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


def base_scene(name) -> str:
    return str(name or "").replace("_physics", "")


def code_hash(names=STEP4_MODULES) -> str:
    h = hashlib.sha256()
    for n in names:
        try:
            with open(os.path.join(HERE, n), "rb") as f:
                h.update(n.encode() + b"\0" + f.read())
        except OSError:
            h.update(n.encode() + b"\0missing")
    return h.hexdigest()[:16]


# =============================================================================================== the lease
class Lease:
    """One holder at a time. ``claim`` and ``release`` are atomic; ``admit`` runs a callable (e.g. an enqueue)
    under the same lock, so a check and its enqueue cannot be split by a claim from another thread."""

    def __init__(self, generation: Callable[[], int] = lambda: 0):
        self._lk = threading.RLock()
        self._gen = generation
        self._cur = None

    def _live(self):
        c = self._cur
        if c is not None and c["sim_bound"] and c["generation"] != self._gen():
            self._cur = None                       # the simulator it was bound to is gone
        return self._cur

    def claim(self, owner: str, job_id: str, sim_bound: bool = True, on_claim: Optional[Callable] = None):
        """``(True, None)`` on success, else ``(False, holder)``. ``on_claim`` runs inside the lock."""
        if owner not in OWNERS:
            raise ValueError(f"unknown lease owner {owner!r}")
        with self._lk:
            cur = self._live()
            if cur is not None:
                return False, dict(cur)
            self._cur = {"owner": owner, "job_id": job_id, "generation": self._gen(), "sim_bound": bool(sim_bound),
                         "since": time.time()}
            if on_claim is not None:
                on_claim()
            return True, None

    def release(self, job_id: str) -> bool:
        with self._lk:
            if self._cur is not None and self._cur["job_id"] == job_id:
                self._cur = None
                return True
            return False

    def holder(self) -> Optional[dict]:
        with self._lk:
            cur = self._live()
            return dict(cur) if cur else None

    def owner(self) -> Optional[str]:
        h = self.holder()
        return h["owner"] if h else None

    def admit(self, refuse_owners, fn: Callable):
        """Run ``fn()`` unless the lease is held by one of ``refuse_owners``; returns ``(ok, holder_or_result)``."""
        with self._lk:
            cur = self._live()
            if cur is not None and cur["owner"] in refuse_owners:
                return False, dict(cur)
            return True, fn()


# =============================================================================================== the port
class ThorPort:
    """The simulator as the honest agent may use it: four names, nothing else (the guard checks the set)."""

    __slots__ = ("_ag", "_pre", "_post", "_fault")

    def __init__(self, ag, pre, post, fault):
        self._ag, self._pre, self._post, self._fault = ag, pre, post, fault

    def do(self, action, **kw):
        self._pre(action)                          # stop / abort: only ever BEFORE a primitive is sent
        try:
            event = self._ag.do(action, **kw)
        except Exception as e:                     # the simulator call itself failed
            self._fault(e)
            raise
        try:
            self._post(self._ag, action)
        except Exception:
            pass                                   # presentation must never erase an action result
        return event

    @property
    def frame(self):
        return self._ag.frame

    @property
    def depth_frame(self):
        return self._ag.depth_frame

    @property
    def event(self):
        return self._ag.event


class _LockedDetector:
    """The honest detector with every ``detect`` call serialised on the console's GPU lock."""

    def __init__(self, det, lock=None):
        self._det, self._lock = det, lock or threading.Lock()

    def detect(self, *a, **kw):
        with self._lock:
            return self._det.detect(*a, **kw)

    def __getattr__(self, name):                   # names / primary / specialist / last_inspecting
        return getattr(self._det, name)


class _StoreTap:
    """Forwards object-memory writes to the real store and keeps the newest places-seen rows for the payload."""

    def __init__(self, store, places: collections.deque, on_place=None):
        self._store, self._places, self._on_place = store, places, on_place

    def add_sighting(self, row):
        return self._store.add_sighting(row) if self._store is not None else 0

    def set_sighting_status(self, *a, **kw):
        if self._store is not None:
            self._store.set_sighting_status(*a, **kw)

    def add_place_seen(self, row):
        rid = self._store.add_place_seen(row) if self._store is not None else 0
        xyz = row.get("xyz") or (None, None, None)
        self._places.append({"type": row.get("object_type"), "place": row.get("place_type"),
                             "x": _r(xyz[0]), "z": _r(xyz[2]) if len(xyz) > 2 else None,
                             "step": row.get("step"), "episode": row.get("episode")})
        if self._on_place:
            self._on_place()
        return rid

    def load_sightings(self, *a, **kw):
        return self._store.load_sightings(*a, **kw) if self._store is not None else []

    def close(self):
        if self._store is not None:
            self._store.close()


def _r(v, n=2):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _underlying(cause, why) -> Optional[str]:
    if cause != "retry_exhausted":
        return cause
    head = str(why or "").split(":", 1)[0].strip()
    return head if head in hs.CAUSES else "retry_exhausted"


# =============================================================================================== the console
class Console:
    """The honest session behind /honest*. Thread-safe; see the module docstring for the contract."""

    def __init__(self, submit: Callable, sim_generation: Callable[[], int], *, scene_now=lambda: "",
                 on_primitive=None, goal_check=None, load_detector=None, make_provider=None, apply_sensors=None,
                 clear_drive=None, act_rev=lambda: 0, items=None, model_info=None, log=None, availability=None,
                 load_map=None, make_store=None, default_endpoint=lambda: "http://127.0.0.1:11434",
                 llm_on=lambda: True, detect_lock=None, out_dir: str = OUT, wait_tick: float = 0.1,
                 work_timeout: float = WORK_TIMEOUT, default_model: str = sg.MODEL, smooth=None):
        self.submit, self.sim_generation, self.scene_now = submit, sim_generation, scene_now
        self.on_primitive, self.goal_check, self.load_detector = on_primitive, goal_check, load_detector
        self.make_provider, self.apply_sensors, self.clear_drive = make_provider, apply_sensors, clear_drive
        self.act_rev, self.items, self.model_info = act_rev, items, model_info
        self._logfn, self.availability_fn = log, availability
        self.load_map = load_map or (lambda: hm.GridMap.load(SCENE))
        self.make_store = make_store
        self.default_endpoint, self.llm_on, self.detect_lock = default_endpoint, llm_on, detect_lock
        self.out_dir, self.wait_tick, self.work_timeout = out_dir, float(wait_tick), float(work_timeout)
        self.default_model = default_model
        #: display-only smooth profile for HonestAgent (callable -> dict | None; the console's motion speed)
        self.smooth = smooth
        self.lease = Lease(sim_generation)
        self.lock = threading.RLock()
        # ---- shared state (under self.lock)
        self.on = False
        self.why = ""                      # why honest mode is off / unavailable (last refusal)
        self.rev = 0
        self.world_gen = 0
        self.session = None
        self.job = None
        self.steps = []
        self.planner_info = None
        self.outcome = None
        self.history = []
        self._live_events = []
        self._mem = {"tracks": [], "lines": []}
        self._holding = None
        self._audit = None
        self._map = None
        self._map_rev = 0
        self._look = None
        self._look_seq = 0
        self._look_jpeg = (None, b"")
        self._waiting = None               # (label, t0) while the worker waits for an LLM reply
        self._llm_busy = {}
        self._places = collections.deque(maxlen=MAX_PLACES)
        self._n = 0
        # ---- worker-owned (touched only by jobs on the simulator thread)
        self._ag = None
        self._gen = None
        self._dirty = True
        self.h = self.sk = self.om = self.gm = None
        self.det = None
        self.types = []
        self.store = None
        self._base = None
        self._base_edges = set()
        self._lp_seen = None
        self._cancel = threading.Event()
        self._abort = None
        self._sim_fault = None

    # ------------------------------------------------------------------ small helpers
    def log(self, msg):
        if self._logfn:
            try:
                self._logfn(f"honest: {msg}")
            except Exception:
                pass

    def _bump(self):
        with self.lock:
            self.rev += 1

    def _next_id(self, prefix):
        with self.lock:
            self._n += 1
            return f"{prefix}{self._n}"

    def job_running(self) -> bool:
        return self.lease.owner() in ("honest", "mode-switch")

    def waiting_text(self) -> str:
        w = self._waiting
        return f"{w[0]} ({time.time() - w[1]:.1f} s)" if w else ""

    def status_brief(self) -> dict:
        with self.lock:
            j = self.job or {}
            return {"on": self.on, "phase": j.get("phase", "idle"), "rev": self.rev, "job_id": j.get("id"),
                    "waiting": self.waiting_text()}

    def availability(self) -> tuple:
        if self.availability_fn is None:
            return True, ""
        try:
            return self.availability_fn()
        except Exception as e:
            return False, f"availability check failed: {e}"

    # ------------------------------------------------------------------ worker jobs
    def _pre(self, action):
        if self._cancel.is_set():
            raise HonestCancelled()
        if self._abort:
            raise HonestAborted(self._abort)

    def _post(self, ag, action):
        if self.on_primitive is not None:
            self.on_primitive(ag, action)

    def _fault(self, exc):
        self._sim_fault = exc

    def render_trip(self, why: str):
        """The display hook saw the renderer degrade during a blocking step: stop before the next primitive."""
        if self.lease.owner() == "honest":
            self._abort = f"render glitch ({why}): the simulator needs a restart"

    def _work(self, fn, *, check_world=True, ensure=True):
        """Submit ``fn(ag)`` as a protected worker job. Returns ``(event, box)``."""
        gen0 = self.sim_generation()
        with self.lock:
            wg0 = self.world_gen
        ev, box = threading.Event(), {"ok": False}

        def job(ag):
            escape = None
            try:
                if self.sim_generation() != gen0:
                    box["stale"] = "simulator restarted"
                    return None
                self._sim_fault = None
                if ensure:
                    self._ensure(ag)
                with self.lock:
                    wg = self.world_gen
                if check_world and wg != wg0:
                    box["stale"] = "the world changed"
                    return None
                box["result"] = fn(ag)
                box["ok"] = True
            except HonestCancelled:
                box["cancelled"] = True
            except HonestAborted as e:
                box["aborted"] = e.why
            except honest.HonestyViolation as e:
                box["violation"] = str(e)[:300]
            except Exception as e:
                if self._sim_fault is not None and not isinstance(self._sim_fault, _NOT_SIM_DEATH):
                    box["sim_death"] = f"{type(e).__name__}: {e}"[:300]
                    escape = e                     # a dead simulator: let the supervisor restart it
                else:
                    box["error"] = f"{type(e).__name__}: {e}"[:300]
            except BaseException as e:             # interpreter shutdown etc.: never swallow
                box["error"] = f"{type(e).__name__}"
                escape = e
            finally:
                try:
                    if self.h is not None and ag is self._ag:
                        self._snap_worker()
                except Exception:
                    pass
                ev.set()
            if escape is not None:
                raise escape
            return None

        self.submit(job)
        return ev, box

    def _wait(self, ev, box, gen0=None):
        """Wait for a worker job; ends early (box['stale']) when the simulator generation changes."""
        gen0 = self.sim_generation() if gen0 is None else gen0
        t0 = time.time()
        while not ev.wait(self.wait_tick):
            if self.sim_generation() != gen0:
                box["stale"] = "simulator restarted"
                return box
            if time.time() - t0 > self.work_timeout:
                box["stale"] = f"the simulator did not run the job within {self.work_timeout:.0f} s"
                return box
        return box

    def _call(self, fn, gen0=None):
        ev, box = self._work(fn)
        return self._wait(ev, box, gen0)

    # ------------------------------------------------------------------ building the honest objects (worker)
    def _ensure(self, ag):
        gen = self.sim_generation()
        if ag is self._ag and gen == self._gen and self.h is not None and not self._dirty:
            return
        replaced = self._ag is not None and (ag is not self._ag or gen != self._gen)
        self._ag, self._gen = ag, gen
        if replaced:
            self._new_world("simulator restarted", hand="empty")
        self._build(ag)

    def _build(self, ag):
        port = ThorPort(ag, self._pre, self._post, self._fault)
        self.h = honest.HonestAgent(port, self.det, mode="honest", smooth=self.smooth)
        fov = float(ag.field_of_view or objmem.FOV_DEFAULT)
        self.om = objmem.ObjectMemory(SCENE, store=self.store, episode=None, fov=fov)
        self.gm = copy.deepcopy(self._base) if self._base is not None else self.load_map()
        self.sk = hs.HonestSkills(self.h, self.gm, self.om, chooser=hs.prior_chooser, type_list=self.types,
                                  log=self._on_event, search=SEARCH_MODE, tidy=TIDY)
        orig = self.sk._look

        def look():
            r0 = self.act_rev()
            obs = orig()
            self._capture(obs, r0)
            return obs

        self.sk._look = look
        self._lp_seen = None
        self._dirty = False
        with self.lock:
            self._map_rev += 1

    def _new_world(self, reason: str, hand: str = "empty", scene: str = ""):
        """The ONLY place world_gen changes. hand: empty (reload) | unknown | keep."""
        with self.lock:
            if self.session is None:
                return
            carry = None
            if hand == "unknown":
                carry = "?"
            elif hand == "keep":
                carry = self.h.holding if self.h is not None else None
            self.world_gen += 1
            s = self.session
            s["world_gen"] = self.world_gen
            s["need_new"] = True
            s["carry"] = carry
            s["pose_stale"] = True
            s["worlds"].append({"world_gen": self.world_gen, "why": reason, "ts": time.strftime("%H:%M:%S")})
            s["worlds"] = s["worlds"][-10:]
            self._look = None
            self._dirty = True
            self._map_rev += 1
            self.rev += 1
            if scene and base_scene(scene) != SCENE:
                self.on = False
                self.why = f"scene is {base_scene(scene)}; honest mode is FloorPlan1 only"
        self.log(f"new world #{self.world_gen}: {reason}")

    def world_changed(self, reason: str, hand: str = "empty", scene: str = ""):
        """Called by the console after a world change it allowed (scene / reset / shuffle / restore)."""
        if self.on:
            self._new_world(reason, hand, scene)

    def intervention(self, kind: str, n: int = 1):
        """Manual input between instructions: counted; the pose shown on the map is marked stale."""
        with self.lock:
            if not self.on or self.session is None:
                return
            iv = self.session["interventions"]
            iv[kind] = iv.get(kind, 0) + int(n)
            self.session["pose_stale"] = True
            self.rev += 1

    # ------------------------------------------------------------------ snapshots (worker side)
    def _on_event(self, msg):
        with self.lock:
            self._live_events.append(f"[{self.h.step if self.h else 0}] {msg}")
            self._live_events = self._live_events[-MAX_EVENTS:]
            self.rev += 1

    def _capture(self, obs, r0):
        r1 = self.act_rev()
        dets = []
        for d in obs.dets:
            b = d.get("bbox_xyxy")
            p = d.get("point")
            dets.append({"type": d.get("type"), "state": d.get("state") or "?", "conf": _r(d.get("conf"), 3),
                         "bbox": [int(v) for v in b] if b else None, "point": [int(v) for v in p] if p else None,
                         "point_ok": bool(d.get("point_ok")), "source": d.get("source", "primary"),
                         "track": d.get("track")})
        with self.lock:
            self._look_seq += 1
            self._look = {"seq": self._look_seq, "step": obs.step, "rgb": obs.rgb, "dets": dets,
                          "tag": (self._gen, self.world_gen, r1) if r0 == r1 else None,
                          "inspecting": bool(getattr(self.det, "last_inspecting", False)),
                          "holding": obs.holding}
        self._snap_worker()

    def _snap_worker(self):
        om, h, gm = self.om, self.h, self.gm
        tracks = []
        for t in sorted(om.tracks.values(), key=lambda t: -t.step):
            if t.status == "superseded":
                continue
            xyz = t.anchor if t.status == "placed" else t.xyz
            tracks.append({"tid": t.tid.rsplit(":", 1)[-1], "type": t.type, "status": t.status, "state": t.state,
                           "conf": _r(t.conf, 3), "place": t.place, "x": _r(xyz[0]) if xyz else None,
                           "y": _r(xyz[1]) if xyz else None, "z": _r(xyz[2]) if xyz else None, "step": t.step,
                           "seen": t.seen})
            if len(tracks) >= MAX_TRACKS:
                break
        lines = om.summary()
        audit = h.audit.summary()
        audit.update(counts=dict(h.audit.counts), violation_list=[str(v[4])[:160] for v in h.audit.violations][-10:],
                     errors_logged=[(s, a, str(e)[:100]) for s, a, e in h.audit.errors][-20:])
        lp = h.last_pose
        agent = None
        with self.lock:
            if lp is not None and lp is not self._lp_seen:
                self._lp_seen = lp
                if self.session is not None:
                    self.session["pose_stale"] = False
            if lp is not None:
                x, z = float(lp["position"]["x"]), float(lp["position"]["z"])
                gx, gz = gm.cell_of(x, z)
                agent = {"gx": gx, "gz": gz, "x": _r(x), "z": _r(z), "yaw": _r(lp["rotation"]["y"], 1),
                         "step": lp.get("step"), "stale": bool((self.session or {}).get("pose_stale"))}
            learned = sorted([[list(a), list(b)] for a, b in gm.blocked_edges - self._base_edges])
            if self._map is None or len(learned) != len(self._map.get("learned_blocked") or []):
                self._map_rev += 1
            self._map = {"rev": self._map_rev, "cell": gm.cell, "origin": list(gm.origin), "n_free": len(gm.free),
                         "learned_blocked": learned, "agent": agent}
            self._mem = {"tracks": tracks, "lines": lines}
            self._holding = h.holding
            self._audit = audit
            self.rev += 1

    # ------------------------------------------------------------------ LLM calls with a bound
    def _bounded(self, kind, fn, timeout, waiting=None, tick=0.05):
        """Run ``fn`` on a helper thread; at most one outstanding call per kind; the caller polls for Stop.
        A late reply lands in a request-local box and is discarded."""
        with self.lock:
            if self._llm_busy.get(kind):
                raise RuntimeError(f"a previous {kind} call is still running")
            self._llm_busy[kind] = True
        box, done = {}, threading.Event()

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:
                box["e"] = e
            finally:
                with self.lock:
                    self._llm_busy[kind] = False
                done.set()

        threading.Thread(target=run, daemon=True, name=f"honest-{kind}").start()
        t0 = time.time()
        try:
            if waiting:
                self._waiting = (waiting, t0)
            while not done.wait(tick):
                if self._cancel.is_set():
                    raise HonestCancelled()
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"{kind} call timed out after {timeout:.0f} s")
        finally:
            if waiting:
                self._waiting = None
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def _chooser(self, model, endpoint):
        provider = self.make_provider(endpoint) if self.make_provider else None
        planner = sg.GemmaPlanner(provider=provider, model=model, types=self.types, timeout=CHOOSER_HTTP_TIMEOUT)

        def ask(prompt):
            return self._bounded("chooser", lambda: planner.ask_text(prompt), CHOOSER_HARD_TIMEOUT,
                                 waiting="Gemma: choosing where to look")

        c = hs.LLMChooser(ask, k=6)
        c.label = f"llm ({model}, constrained to the detector type list)"
        c.planner = planner
        return c

    # ------------------------------------------------------------------ mode switch
    def switch(self, on: bool, timeout: float = 180.0) -> dict:
        on = bool(on)
        with self.lock:
            if on == self.on:
                return {"ok": True, "on": self.on}
        if not on and self.job_running():
            return {"ok": False, "code": 409, "error": "the agent is acting - Stop first",
                    "owner": self.lease.owner()}
        if on:
            ok, why = self.availability()
            if not ok:
                with self.lock:
                    self.why = why
                    self.rev += 1
                return {"ok": False, "code": 400, "error": why}
        jid = self._next_id("mode")
        ok, holder = self.lease.claim("mode-switch", jid)
        if not ok:
            return {"ok": False, "code": 409, "error": f"busy: {holder['owner']} is running", "owner": holder["owner"]}
        try:
            if on:
                return self._enter(timeout)
            return self._leave(timeout)
        finally:
            self.lease.release(jid)
            self._bump()

    def _enter(self, timeout):
        try:
            if self.det is None:
                if self.load_detector is None:
                    raise RuntimeError("no detector loader")
                self.det = _LockedDetector(self.load_detector(), self.detect_lock)
                self.types = hs.detector_types(self.det)
            if self.store is None and self.make_store is not None:
                self.store = _StoreTap(self.make_store(), self._places, on_place=self._bump)
            self._base = self.load_map()
            self._base_edges = set(self._base.blocked_edges)
        except Exception as e:
            with self.lock:
                self.why = f"could not start honest mode: {type(e).__name__}: {e}"[:300]
            self.log(self.why)
            return {"ok": False, "code": 500, "error": self.why}
        gen0 = self.sim_generation()

        def enter(ag):
            if self.apply_sensors is not None:
                self.apply_sensors(ag, True)          # one announced scene reload: depth on, hand empty
            self._ag, self._gen = ag, self.sim_generation()
            self._build(ag)
            return True

        ev, box = self._work(enter, check_world=False, ensure=False)
        self._wait(ev, box, gen0)
        if not box.get("ok"):
            why = box.get("stale") or box.get("error") or box.get("sim_death") or "the switch did not complete"
            with self.lock:
                self.why = f"could not start honest mode: {why}"
            self._ag = None
            return {"ok": False, "code": 500, "error": self.why}
        with self.lock:                                 # commit point
            self.on, self.why = True, ""
            self.world_gen = 1
            self.session = {"id": uuid.uuid4().hex[:6], "world_gen": 1, "episode": None, "mem_n": 0,
                            "instructions": 0, "interventions": {}, "tainted": False, "need_new": True,
                            "carry": None, "pose_stale": False, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "worlds": [{"world_gen": 1, "why": "honest mode on (scene reloaded, depth on)",
                                        "ts": time.strftime("%H:%M:%S")}]}
            self.job, self.steps, self.planner_info, self.outcome, self.history = None, [], None, None, []
            self._look, self._live_events = None, []
            self._places.clear()
            self.rev += 1
        self.log("ON - the agent now decides from its camera + depth, its own walked map and memory")
        return {"ok": True, "on": True}

    def _leave(self, timeout):
        gen0 = self.sim_generation()
        with self.lock:
            self.on = False                              # first, so the sensor change drops depth
            self.rev += 1

        def leave(ag):
            if self.apply_sensors is not None:
                self.apply_sensors(ag, False)
            return True

        ev, box = self._work(leave, check_world=False, ensure=False)
        self._wait(ev, box, gen0)
        with self.lock:
            self.h = self.sk = self.om = None
            self._ag, self._dirty, self.session = None, True, None
            self._look = None
            self.rev += 1
        self.log("OFF - back to the oracle agent")
        return {"ok": True, "on": False}

    # ------------------------------------------------------------------ instructions
    def _item(self, item_id):
        if not item_id:
            return None
        for it in (self.items() if self.items else []):
            if it.get("id") == item_id:
                return it
        return None

    def run(self, text: str = "", model: str = "", endpoint: str = "", item: str = "", lang: str = "") -> dict:
        with self.lock:
            if not self.on:
                return {"ok": False, "code": 409, "error": "honest mode is off"}
        if not self.llm_on():
            return {"ok": False, "code": 409, "error": "the LLM is off - turn it on to plan"}
        it = self._item(item)
        if item and it is None:
            return {"ok": False, "code": 400, "error": f"unknown item {item!r}"}
        if it is not None and it.get("setup"):
            return {"ok": False, "code": 400, "error": f"{item} needs a setup - run it via sim/step4_gemma_trial.py"}
        lang = lang if lang in ("en", "fa") else ("fa" if any("؀" <= ch <= "ۿ" for ch in text or "") else "en")
        if it is not None and not text:
            text = it.get(lang) or it.get("en") or ""
        text = (text or "").strip()
        if not text:
            return {"ok": False, "code": 400, "error": "empty instruction"}
        model = model or self.default_model
        endpoint = (endpoint or self.default_endpoint() or "").rstrip("/")
        jid = self._next_id("i")

        def claimed():
            if self.clear_drive:
                self.clear_drive()
            self._cancel = threading.Event()
            self._abort = None
            with self.lock:
                self.job = {"id": jid, "kind": "instruction", "phase": "prepare", "command": text, "lang": lang,
                            "item": item or None, "level": it.get("level") if it else None,
                            "level_source": "dataset" if it else "none", "model": model, "endpoint": endpoint,
                            "t_start": time.time(), "elapsed_s": 0.0, "error": None}
                self.steps, self.planner_info, self.outcome, self._live_events = [], None, None, []
                self.rev += 1

        ok, holder = self.lease.claim("honest", jid, on_claim=claimed)
        if not ok:
            return {"ok": False, "code": 409, "error": f"busy: {holder['owner']} is running", "owner": holder["owner"]}
        threading.Thread(target=self._control, args=(jid, text, lang, it, model, endpoint), daemon=True,
                         name=f"honest-{jid}").start()
        return {"ok": True, "job_id": jid}

    def _phase(self, phase, error=None):
        with self.lock:
            if self.job is not None:
                if self.job["phase"] == "stopping" and phase in ACTIVE:
                    return
                self.job["phase"] = phase
                if error is not None:
                    self.job["error"] = error
                self.job["elapsed_s"] = round(time.time() - self.job["t_start"], 1)
            self.rev += 1

    def _prepare(self, iid, item, chooser, sweep=True, look=False):
        """Worker: start / continue the memory episode, then (optionally) a report-only pre-check of the goal."""
        h, sk = self.h, self.sk
        with self.lock:
            s = self.session
            new = bool(s["need_new"] or s["tainted"] or self.om.episode is None)
            if new:
                s["mem_n"] += 1
                ep = f"console:{s['id']}:w{self.world_gen}:m{s['mem_n']}"
                holding = "?" if s["tainted"] else (s["carry"] if s["carry"] != "keep" else h.holding)
                s["need_new"] = s["tainted"] = False
                s["carry"] = "keep"
                s["episode"] = ep
        if chooser is not None:
            sk.chooser = chooser
        if new:
            if not look:
                self._phase("sweep" if sweep else "prepare")
            sk.start_episode(ep, holding=holding, sweep=sweep)
            h.audit.episode = iid
        else:
            h.audit.new_episode(iid)
            if look:
                sk.look_around()
        cur, _ = hm.current_cell(h, self.gm)
        if cur not in self.gm.free:
            raise HonestAborted(f"map mismatch: the agent stands in cell {cur}, which the walked map does not "
                                "have - honest mode needs the FloorPlan1 start world")
        pre = None
        if item is not None and self.goal_check is not None:
            h = self.h
            with h.scoring():
                h.meta("scoring")
                pre = self.goal_check(self._ag, item)
        return {"new": new, "summary": self.om.summary(), "holding": h.holding, "pre": pre, "start": h.step}

    def _box_end(self, box, what):
        """Translate a finished worker box into (phase, error) or None when it succeeded."""
        if box.get("ok"):
            return None
        if box.get("cancelled"):
            return "cancelled", "stopped by the operator"
        if box.get("stale"):
            return "failed", box["stale"]
        if box.get("aborted"):
            return "failed", box["aborted"]
        if box.get("violation"):
            return "failed", f"honesty violation: {box['violation']}"
        if box.get("sim_death"):
            return "failed", f"simulator died: {box['sim_death']}"
        return "failed", f"{what}: {box.get('error') or 'did not complete'}"

    def _control(self, jid, text, lang, item, model, endpoint):
        end = None
        gen0 = self.sim_generation()
        steps_done = 0
        try:
            chooser = self._chooser(model, endpoint) if self.make_provider else None
            box = self._wait(*self._work(lambda ag: self._prepare(jid, item, chooser), check_world=False), gen0)
            end = self._box_end(box, "prepare")
            if end:
                return
            prep = box["result"]
            with self.lock:
                if prep["pre"] is not None:
                    self.outcome = {"pre": prep["pre"]}
                self.session["instructions"] += 1
            # ---------------------------------------------------- planning (this thread, never the worker)
            self._phase("planning")
            provider = self.make_provider(endpoint) if self.make_provider else None
            planner = sg.GemmaPlanner(provider=provider, model=model, types=self.types, timeout=PLAN_TIMEOUT)
            try:
                res = self._bounded("plan", lambda: planner.plan(text, prep["summary"], prep["holding"]),
                                    PLAN_HARD_TIMEOUT, tick=self.wait_tick)
            except HonestCancelled:
                end = ("cancelled", "stopped while Gemma was planning (its late reply is discarded)")
                return
            except Exception as e:
                end = ("failed", f"planner: {type(e).__name__}: {e}"[:300])
                return
            last = res["attempts"][-1] if res.get("attempts") else {}
            info = {"ok": res["ok"], "first_ok": res["first_ok"], "repaired": res["repaired"],
                    "rejected": res["rejected"], "secs": res["secs"], "say": res.get("say", ""),
                    "plan": res["plan"] if res["ok"] else last.get("validation", {}).get("steps", []),
                    "errors": [e["reason"] for e in (last.get("validation") or {}).get("errors", [])][:12],
                    "notes": (last.get("validation") or {}).get("notes", [])[:12], "model": model,
                    "normalised": res.get("normalised", []),
                    "calls": [{k: c.get(k) for k in ("kind", "secs", "error", "done_reason")} for c in planner.calls]}
            with self.lock:
                self.planner_info = info
                self.steps = [{"i": i, "action": st.get("action"), "object": st.get("object"), "to": st.get("to"),
                               "arg": st.get("arg"), "status": "pending", "retries": 0, "cause": None, "under": None,
                               "why": "", "prims": 0, "secs": None, "events": [], "verify": [], "search": None,
                               "chooser": None} for i, st in enumerate(info["plan"])]
                self.rev += 1
            if self._cancel.is_set():
                end = ("cancelled", "stopped after planning")
                return
            if not res["ok"]:
                end = ("rejected", "plan rejected after one repair - not executed")
                return
            # ---------------------------------------------------- executing (one worker job per step)
            self._phase("executing")
            for i, st in enumerate(res["plan"]):
                if self._cancel.is_set():
                    end = ("cancelled", "stopped by the operator")
                    break
                with self.lock:
                    used = (self.h.step - prep["start"]) if self.h else 0
                    self._live_events = []
                if used > EP_PRIM_CAP:
                    self._step_update(i, status="skipped", cause="episode_budget",
                                      why=f"instruction primitive cap {EP_PRIM_CAP}; not run")
                    continue
                self._step_update(i, status="running")
                t0 = time.time()
                rest = [dict(x) for x in res["plan"][i + 1:]]
                box = self._wait(*self._work(lambda ag, st=st, rest=rest: self._exec(st, rest)), gen0)
                info = box.get("result") or box.get("partial") or {}
                rec = self._step_record(box, info, round(time.time() - t0, 1))
                self._step_update(i, **rec)
                steps_done += 1
                end = self._box_end(box, f"step {i}")
                if end:
                    break
                if self.h is not None and self.h.audit.invalid:
                    end = ("failed", "honesty violation recorded - stopped")
                    break
            if end is None and self._cancel.is_set():
                end = ("cancelled", "stopped by the operator")
            # ---------------------------------------------------- tidy: close what this agent opened (not on a stop)
            if steps_done and TIDY and not self._cancel.is_set() and (end is None or not any(
                    w in end[1] for w in ("restarted", "world changed", "died", "violation", "render glitch",
                                          "map mismatch"))):
                box = self._wait(*self._work(lambda ag: self._tidy()), gen0)
                if box.get("ok"):
                    with self.lock:
                        during = [x["type"] for st in self.steps for x in (st.get("tidy") or [])
                                  if x.get("result") in ("closed", "already")]
                        self.outcome = {**(self.outcome or {}), "tidy": {**box["result"], "closed_during": during}}
                        self.rev += 1
            # ---------------------------------------------------- report-only goal check
            if item is not None and self.goal_check is not None and (end is None or end[0] != "cancelled") \
                    and not (end and ("restarted" in end[1] or "world changed" in end[1] or "died" in end[1])):
                self._phase("checking")
                box = self._wait(*self._work(lambda ag: self._goal(item)), gen0)
                if box.get("ok"):
                    with self.lock:
                        self.outcome = {**(self.outcome or {}), "goal": box["result"]}
            if end is None:
                with self.lock:
                    bad = [s for s in self.steps if s["status"] != "ok"]
                end = ("done", None) if not bad else ("failed", f"{len(bad)} step(s) did not succeed")
        except Exception as e:                                # the controller itself must never die silently
            end = ("failed", f"console error: {type(e).__name__}: {e}"[:300])
        finally:
            self._finish(jid, end or ("failed", "ended without a result"), text, lang, item, model, endpoint)

    def _tidy(self):
        """Worker: end of a run -- close every container this agent opened and left open (explicit opens kept)."""
        with self.lock:
            self._live_events = []
        r = self.sk.tidy_up(keep_explicit=True)
        return {"closed": r["closed"], "failed": r["failed"], "kept_open": r["kept_open"],
                "left_open": [e["type"] for e in self.sk.opened_by_me], "prims": r["prims"],
                "events": [e for e in r["events"] if "close" in e or "tidy" in e][-MAX_EVENTS:]}

    def _exec(self, st, upcoming=()):
        sk, h = self.sk, self.h
        start = h.step
        s0 = getattr(h, "sends", 0)
        partial = {}
        if getattr(sk.chooser, "last", None) is not None:
            sk.chooser.last = None
        try:
            r = sk.exec_step(dict(st), upcoming=list(upcoming))
            return {**r, "start": start}
        finally:
            partial.update(prims=h.step - start, sends=getattr(h, "sends", 0) - s0,
                           events=list(sk.events)[-MAX_EVENTS:], holding=h.holding,
                           search=copy.deepcopy(sk.search_info), verify=list(sk.verify), retries=sk.retries)
            ch = getattr(sk.chooser, "last", None)
            if ch:
                partial["chooser"] = {"places": list(ch.get("places") or []), "error": ch.get("error") or "",
                                      "answer": str(ch.get("answer") or "")[:300]}
            self._partial = partial

    def _step_record(self, box, info, secs) -> dict:
        part = getattr(self, "_partial", {}) or {}
        self._partial = {}
        d = (info.get("detail") or {}) if info else {}
        if box.get("ok"):
            status = "ok" if info.get("ok") else "failed"
            cause = info.get("cause")
            why = d.get("why", "")
        elif box.get("cancelled"):
            status, cause, why = "cancelled", None, "stopped before the next primitive"
        else:
            status, cause = "failed", None
            why = box.get("aborted") or box.get("stale") or box.get("violation") or box.get("sim_death") \
                or box.get("error") or "did not complete"
        ch = part.get("chooser")
        if ch is not None:
            calls = getattr(getattr(self.sk, "chooser", None), "planner", None)
            secs_c = [c["secs"] for c in (calls.calls if calls else []) if c.get("kind") == "chooser"]
            ch = {**ch, "secs": secs_c[-1] if secs_c else None}
        return {"status": status, "retries": info.get("retries", part.get("retries", 0)) if info else part.get("retries", 0),
                "cause": cause, "under": _underlying(cause, why), "why": str(why)[:300],
                "prims": d.get("prims", part.get("prims", 0)), "secs": secs,
                "events": (d.get("events") or part.get("events") or [])[-MAX_EVENTS:],
                "verify": d.get("verify") or part.get("verify") or [],
                "search": copy.deepcopy(d.get("search") or part.get("search")), "chooser": ch,
                "sends": part.get("sends"), "tidy": copy.deepcopy(d.get("tidy")), "left_open": d.get("left_open"),
                "holding": d.get("holding", part.get("holding"))}

    def _step_update(self, i, **kw):
        with self.lock:
            if 0 <= i < len(self.steps):
                self.steps[i].update(kw)
            self.rev += 1

    def _goal(self, item):
        h = self.h
        with h.scoring():
            h.meta("scoring")
            r = self.goal_check(self._ag, item)
        return dict(r or {}, checked=True)

    def _finish(self, jid, end, text, lang, item, model, endpoint):
        phase, err = end
        try:
            with self.lock:
                if self.job is not None and self.job["id"] == jid:
                    if phase == "cancelled":
                        for s in self.steps:
                            if s["status"] in ("pending", "running"):
                                s["status"] = "cancelled" if s["status"] == "running" else "skipped"
                    else:
                        for s in self.steps:
                            if s["status"] == "pending":
                                s["status"] = "skipped"
                    self.job.update(phase=phase, error=err, elapsed_s=round(time.time() - self.job["t_start"], 1))
                audit = copy.deepcopy(self._audit) or {}
                tainted = bool(audit.get("invalid"))
                if tainted and self.session is not None:
                    self.session["tainted"] = True
                ok_steps = sum(1 for s in self.steps if s["status"] == "ok")
                self.outcome = {**(self.outcome or {}),
                                "exec": {"ok_steps": ok_steps, "n_steps": len(self.steps)},
                                "audit": {"valid": not tainted, "violations": int(audit.get("violations") or 0)}}
                self.outcome.setdefault("goal", None)
                entry = {"id": jid, "command": text, "lang": lang, "level": (item or {}).get("level"),
                         "item": (item or {}).get("id"), "phase": phase, "error": err,
                         "exec": self.outcome["exec"], "goal": self.outcome.get("goal"), "valid": not tainted,
                         "ticket": self._ticket()}
                self.history.append(entry)
                self.history = self.history[-MAX_HISTORY:]
                self.rev += 1
            self._archive("instruction", jid, text, lang, item, model, endpoint, phase, err, audit)
            self.log(f"{jid} {phase}{(': ' + err) if err else ''}")
        except Exception as e:
            self.log(f"finish error: {e}")
        finally:
            self._waiting = None
            self.lease.release(jid)
            self._bump()

    def _ticket(self) -> dict:
        return copy.deepcopy({"job": self.job, "planner": self.planner_info, "steps": self.steps,
                              "outcome": self.outcome})

    def _archive(self, kind, jid, text, lang, item, model, endpoint, phase, err, audit):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with self.lock:
                s = copy.deepcopy(self.session or {})
                row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "id": jid, "command": text,
                       "lang": lang, "item": (item or {}).get("id"), "level": (item or {}).get("level"),
                       "phase": phase, "error": err, "session": s.get("id"), "world_gen": self.world_gen,
                       "memory_episode": s.get("episode"), "interventions": s.get("interventions"),
                       "planner": copy.deepcopy(self.planner_info) if kind == "instruction" else None,
                       "steps": [{k: st.get(k) for k in ("i", "action", "object", "to", "status", "retries", "cause",
                                                          "under", "prims", "sends", "secs", "why")}
                                 for st in self.steps]
                       if kind == "instruction" else [],
                       "outcome": copy.deepcopy(self.outcome) if kind == "instruction" else None,
                       "audit": audit, "code_hash": code_hash(), "model": model, "endpoint": endpoint}
            if self.model_info and model:
                try:
                    row["model_digest"] = self.model_info(endpoint, model)
                except Exception:
                    row["model_digest"] = None
            with open(os.path.join(self.out_dir, RUNS_FILE), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            self.log(f"archive failed: {e}")

    # ------------------------------------------------------------------ buttons
    def stop(self) -> dict:
        owner = self.lease.owner()                   # lock order: lease before console, never the reverse
        with self.lock:
            j = self.job
            running = j is not None and j["phase"] in ACTIVE and owner == "honest"
            if running:
                self._cancel.set()
                if j["phase"] != "planning":
                    j["phase"] = "stopping"
                self.rev += 1
        return {"ok": True, "stopping": bool(running)}

    def look(self) -> dict:
        with self.lock:
            if not self.on:
                return {"ok": False, "code": 409, "error": "honest mode is off"}
        jid = self._next_id("look")

        def claimed():
            if self.clear_drive:
                self.clear_drive()
            self._cancel = threading.Event()
            self._abort = None
            with self.lock:
                self.job = {"id": jid, "kind": "look", "phase": "sweep", "command": "look around", "lang": "en",
                            "item": None, "level": None, "level_source": "none", "model": None, "endpoint": None,
                            "t_start": time.time(), "elapsed_s": 0.0, "error": None}
                self.steps, self.planner_info, self.outcome, self._live_events = [], None, None, []
                self.rev += 1

        ok, holder = self.lease.claim("honest", jid, on_claim=claimed)
        if not ok:
            return {"ok": False, "code": 409, "error": f"busy: {holder['owner']} is running", "owner": holder["owner"]}

        def go():
            end = ("failed", "ended without a result")
            try:
                box = self._wait(*self._work(lambda ag: self._prepare(jid, None, None, sweep=True, look=True),
                                             check_world=False))
                end = self._box_end(box, "look around") or ("done", None)
            except Exception as e:
                end = ("failed", f"console error: {e}")
            finally:
                try:
                    with self.lock:
                        if self.job is not None and self.job["id"] == jid:
                            self.job.update(phase=end[0], error=end[1],
                                            elapsed_s=round(time.time() - self.job["t_start"], 1))
                        audit = copy.deepcopy(self._audit) or {}
                        if audit.get("invalid") and self.session is not None:
                            self.session["tainted"] = True
                    self._archive("look", jid, "look around", "en", None, None, None, end[0], end[1], audit)
                finally:
                    self.lease.release(jid)
                    self._bump()

        threading.Thread(target=go, daemon=True, name=f"honest-{jid}").start()
        return {"ok": True, "job_id": jid}

    def forget(self) -> dict:
        with self.lock:
            if not self.on or self.session is None:
                return {"ok": False, "code": 409, "error": "honest mode is off"}
        if self.job_running():
            return {"ok": False, "code": 409, "error": "the agent is acting - Stop first"}
        with self.lock:
            self.session["need_new"] = True
            self.session["carry"] = "keep"           # Forget never invents an empty hand
            self.rev += 1
        self.log("forget: the next instruction starts a new memory episode (the hand is kept)")
        return {"ok": True}

    # ------------------------------------------------------------------ display
    def overlay(self, sim_gen, act_rev):
        """The agent's last-look detections, only while the camera has not moved or acted since it looked."""
        with self.lock:
            lk = self._look
            if not self.on or lk is None or lk["tag"] != (sim_gen, self.world_gen, act_rev):
                return None
            return [dict(d) for d in lk["dets"]]

    def last_look_jpeg(self) -> bytes:
        with self.lock:
            lk = self._look
            if lk is None or lk.get("rgb") is None:
                return b""
            if self._look_jpeg[0] == lk["seq"]:
                return self._look_jpeg[1]
            rgb, dets, step = lk["rgb"], [dict(d) for d in lk["dets"]], lk["step"]
            seq = lk["seq"]
        data = render_look(rgb, dets, step)
        with self.lock:
            self._look_jpeg = (seq, data)
        return data

    def snapshot(self, full: bool = False) -> dict:
        ok, why = (True, "") if self.on else self.availability()
        holder = self.lease.holder()                 # lock order: lease before console
        with self.lock:
            j = copy.deepcopy(self.job)
            if j is not None and j["phase"] in ACTIVE:
                j["elapsed_s"] = round(time.time() - j["t_start"], 1)
            steps = copy.deepcopy(self.steps)
            if j is not None and j["phase"] in ACTIVE:
                for s in steps:
                    if s["status"] == "running":
                        s["events"] = list(self._live_events)
            s = copy.deepcopy(self.session)
            if s is not None:
                s["interventions"] = [{"kind": k, "n": n} for k, n in sorted(s["interventions"].items())]
                for k in ("need_new", "carry"):
                    s.pop(k, None)
            lk = self._look
            m = copy.deepcopy(self._map) if self._map else None
            if m is not None and full and self._base is not None:
                m["free"] = sorted([list(c) for c in self._base.free])
            out = {"ok": True, "on": self.on, "available": {"ok": bool(ok), "why": self.why or why},
                   "model": (j or {}).get("model") or self.default_model,
                   "endpoint": (j or {}).get("endpoint") or self.default_endpoint(),
                   "assumptions": list(ASSUMPTIONS), "session": s, "job": j,
                   "outcome": copy.deepcopy(self.outcome), "planner": copy.deepcopy(self.planner_info),
                   "steps": steps, "holding": self._holding if self.on else None,
                   "memory": copy.deepcopy(self._mem), "places_seen": list(self._places), "map": m,
                   "last_look": ({"seq": lk["seq"], "step": lk["step"], "n_dets": len(lk["dets"]),
                                  "inspecting": lk["inspecting"],
                                  "dets": [{k: d[k] for k in ("type", "state", "conf", "source")}
                                           for d in lk["dets"]][:30]} if lk else None),
                   "audit": copy.deepcopy(self._audit),
                   "history": [{k: v for k, v in hh.items() if k != "ticket"} for hh in self.history],
                   "lease": holder, "waiting": self.waiting_text(), "rev": self.rev}
        return out

    def history_ticket(self, jid) -> Optional[dict]:
        with self.lock:
            for hh in self.history:
                if hh["id"] == jid:
                    return copy.deepcopy(hh.get("ticket"))
        return None


# =============================================================================================== drawing
def render_look(rgb, dets, step) -> bytes:
    """JPEG of an observation with its boxes, grounding points and source (primary / specialist)."""
    from PIL import Image, ImageDraw
    import numpy as np
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    img = Image.fromarray(a.astype("uint8")).convert("RGB")
    dr = ImageDraw.Draw(img)
    draw_dets(dr, dets, img.size)
    dr.rectangle([4, 4, 96, 22], fill=(12, 16, 19))
    dr.text((8, 7), f"step {step}", fill=(228, 234, 236))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def draw_dets(dr, dets, size):
    w, h = size
    for d in dets:
        b = d.get("bbox")
        if not b:
            continue
        col = (242, 178, 78) if d.get("source") == "specialist" else (91, 211, 139)
        dr.rectangle(b, outline=col, width=2)
        label = f"{d.get('type')} {d.get('conf') or 0:.2f}"
        st = d.get("state")
        if st and st != "?":
            label += f" [{st}]"
        ty = b[1] - 14 if b[1] >= 28 else min(h - 14, b[3] + 2)     # flip below near the top strip
        dr.rectangle([b[0], ty, min(w - 1, b[0] + 7 * len(label)), ty + 13], fill=(12, 16, 19))
        dr.text((b[0] + 2, ty + 1), label, fill=col)
        p = d.get("point")
        if p:
            x, y = p
            c2 = (91, 211, 139) if d.get("point_ok") else (255, 130, 119)
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=c2, width=2)
