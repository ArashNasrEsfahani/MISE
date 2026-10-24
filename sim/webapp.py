"""Full web control panel for AI2-THOR — feature-parity with the Tkinter GUI, served as a chic browser app.

Pure stdlib http.server (no pip installs). The AI2-THOR controller lives in ONE worker thread (kept single-threaded);
HTTP handlers only enqueue jobs and read shared buffers. The egocentric view is pushed as an MJPEG stream. A job may
RETURN a generator, which the worker advances one step per rendered frame, so walking/skills animate live.

Run inside WSL (same env as the GUI):
    cd /mnt/d/Personal/Master*/sim
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python webapp.py
Then open  http://localhost:8000  (or the printed WSL IP from Windows if localhost doesn't forward).
"""
import collections
import contextlib
import io
import importlib.util
import json
import os
import queue
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import agent_llm as al

try:
    import thor_actions as ta
    _TA_ERROR = None
except Exception as _exc:              # ai2thor is WSL-only (CLAUDE.md): on the Windows dev box the module
    ta = None                          # cannot import, but the observability API and every pure helper below
    _TA_ERROR = _exc                   # must still be importable/testable. main() re-raises this immediately.

try:                                   # observability sidecar — optional by design (docs/observability-design.md §6.0)
    import trace as _trace             # sim/trace.py; it shadows the stdlib coverage module on sys.path
    if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
        _trace = None                  # we resolved the STDLIB `trace` module instead: run without tracing
except Exception:                      # a broken/absent sidecar must never stop the console from starting
    _trace = None


# ---------------------------------------------------------------- write-side tracing helpers (never raise)
# The read-only /trace_* endpoints live at the bottom of this file; THESE are the hooks that put anything in
# the store to read. Every one of them swallows its own errors: a tracer exception inside the worker thread
# would kill the run loop permanently, and inside a plan step it would abort the command.
def _th(name, *a, **kw):
    """Call ``trace.<name>(*a, **kw)``; return None on absence or failure. Resolved by string (see knowledge._t)."""
    if _trace is None:
        return None
    try:
        fn = getattr(_trace, name, None)
        return None if fn is None else fn(*a, **kw)
    except Exception:
        return None


def _trun(**kw):
    """``trace.run`` context manager, or a null context that still yields a usable run id."""
    if _trace is None:
        return contextlib.nullcontext(None)
    try:
        return _trace.run(**kw)
    except Exception:
        return contextlib.nullcontext(None)


def _tspan(name, **kw):
    """``trace.span`` context manager, or a null context yielding None."""
    if _trace is None:
        return contextlib.nullcontext(None)
    try:
        return _trace.span(name, **kw)
    except Exception:
        return contextlib.nullcontext(None)


def _tbound(ctx):
    """Re-attach a run/span captured on the PLAN thread to whatever thread is running now.

    The AI2-THOR controller lives in one worker thread that outlives every command, so it can never own a run
    of its own; without this the episode a step writes is an orphan (or, worse, inherits whichever run happened
    to touch the process default last)."""
    if _trace is None or not ctx:
        return contextlib.nullcontext(None)
    try:
        return _trace.bound(ctx.get("run_id"), ctx.get("span_id"), scene=ctx.get("scene"),
                            level=ctx.get("level"))
    except Exception:
        return contextlib.nullcontext(None)


def _tsnap():
    """Serializable copy of the ambient trace context, to hand into a worker-thread closure."""
    if _trace is None:
        return None
    try:
        return _trace.snapshot() or None
    except Exception:
        return None


def _install_trace_hooks():
    """Wrap ``ThorAgent.do/reset/set_sensors`` so every executed AI2-THOR primitive and every scene reset is
    recorded. Without this call ``record_action`` is unreachable code and the run tree can never show what the
    robot physically did under a plan step. Idempotent; safe when ai2thor is absent."""
    if _trace is None or ta is None:
        return None
    return _th("install_thor_hooks", ta)


_install_trace_hooks()

PORT = 8000
# Bumped whenever web/ starts depending on a server route or /status field that an older webapp.py lacks.
# sim/web/* is re-read from disk on every request but THIS file is loaded once at process start, so editing the
# server and merely reloading the browser pairs a new console with an old backend -- new controls 404, new
# /status fields are absent, and nothing anywhere says why. app.js compares its own UI_API against /status.api
# and puts a banner up on a mismatch. Keep the two numbers in step.
WEBAPP_API = 7      # 5: /llm switch + /status.llm; 6: two-stage detectors (/inspect, /status.detectors);
#                     7: honest mode (/honest, /honest_run|stop|look|forget, /honest_last_look.jpg, /status.honest|lease)
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

# The LLM switch. `webapp.py --no-llm` (START-TESTING.bat option [2]) boots with the planner OFF: nothing
# contacts Ollama -- no model list, no /cmd planning, no eval runs -- so the simulator, manual driving, the
# dataset editor and the trace viewer all work with Ollama stopped. The topbar LLM chip flips it at runtime
# (/llm?on=1|0); turning it on starts the Windows Ollama app over WSL interop when the endpoint is silent.
_LLM = {"on": True, "note": ""}
OLLAMA_APP = os.environ.get("OLLAMA_APP", r"C:\Users\arash\AppData\Local\Programs\Ollama\ollama app.exe")
_LLM_OFF_MSG = "LLM is off - turn it on with the LLM chip in the topbar"

_jobs: "queue.Queue" = queue.Queue()
_state = {"jpeg": b"", "fps": 0.0, "scene": "", "held": "none", "busy": False, "log": [],
          "objects": [], "view": "rgb", "recording": False, "plan": None, "pose": {}, "gpu": "",
          "resolution": "600x600", "train": None, "ncam": 0, "demo": {"active": False, "steps": 0},
          "loop_last": None, "seq": 0, "dropped": 0, "motion_label": ""}
_view = {"mode": "rgb", "map_cam": False}
# `oracle` is AI2-THOR's ground-truth object list (what describe() reports); `text` is what the planner ACTUALLY
# saw last. They are the same string in oracle mode and deliberately different in perceived mode, where `text`
# comes only from the detector -- keeping both is what lets the console say "degraded" honestly instead of
# quietly handing the planner the oracle answer under a perceived-mode badge.
_OBS = {"text": "", "oracle": ""}
# oracle | perceived -- the grounding source for the agent's senses. `degraded` holds the REASON perceived mode
# could not run (missing weights, no ultralytics, inference error) on the last attempt; None when it ran clean.
_PERCEPTION = {"mode": "oracle", "degraded": None}
_detector = {"scene": None, "obj": None, "mtime": 0.0}  # lazily-built per-scene YOLO Detector for the 'detect' view
# step-3 inspection specialist: "auto" = only while holding an object up (the deployment rule), "off", "always"
_INSPECT = {"mode": "auto"}
_drive = {"action": None, "t": 0.0}                    # held-key continuous drive (action + last-keepalive timestamp)
_lock = threading.Lock()

# ---- frame fan-out -------------------------------------------------------------------------------------------
# The worker publishes ONE jpeg per rendered frame and bumps `seq`; each /stream client blocks on this condition
# and writes every seq it has not written yet. The previous loop slept 1/30 s and re-read whatever happened to be
# current, so at any worker rate that was not exactly 30 fps it silently dropped (or duplicated) frames -- which
# is why an executing plan looked like a slideshow even when the robot had moved smoothly. STREAM_BACKLOG bounds
# how far a slow client may lag before it is fast-forwarded to the newest frame; the skip is COUNTED and reported
# in /status, never hidden.
STREAM_BACKLOG = 4
_frames = collections.deque(maxlen=STREAM_BACKLOG)     # [(seq, jpeg_bytes)], newest last
_frame_cv = threading.Condition(_lock)                 # shares _lock, so (seq, jpeg) are published atomically

# ---- motion detail -------------------------------------------------------------------------------------------
# The worker renders exactly one frame per generator `yield`, so the yield granularity of the locomotion
# generators IS the frame rate of the robot's movement. These profiles set that granularity (via ThorAgent.motion,
# which every composite skill inherits) plus two things the simulator gives no animation for at all:
#   `hold`           extra frames rendered either side of an INSTANTANEOUS primitive (pickup / slice / toggle),
#                    so the state change is something you watch happen instead of a jump between two frames;
#   `openness_steps` how many increments an open/close is driven through, so a fridge door swings.
# `delay` paces the worker while (and only while) a plan is animating -- idle and manual drive stay full speed.
MOTION_PROFILES = {
    "fast":      {"turn": 30.0, "step": 0.25, "delay": 0.00, "hold": 0, "openness_steps": 1,
                  "note": "30 deg turns, 0.25 m steps - fewest frames"},
    "normal":    {"turn": 15.0, "step": 0.15, "delay": 0.00, "hold": 3, "openness_steps": 6,
                  "note": "15 deg turns, 0.15 m steps, doors swing"},
    "cinematic": {"turn": 8.0,  "step": 0.08, "delay": 0.05, "hold": 8, "openness_steps": 12,
                  "note": "8 deg turns, 0.08 m steps, paced - every move visible"},
}
MOTION_DEFAULT = "fast"     # 2026-10-05: "normal" made open/close ~8x and go-to ~3x slower than the old console
_MOTION = {"speed": MOTION_DEFAULT}


#: Honest-mode smooth display (2026-10-07, Arash: "it happened way too fast, I need it smoother"). HonestAgent sends
#: ONE logical primitive as these smaller physical sends, each streamed as a frame with a short pause; the agent's
#: decisions and its logical primitive count are the same at every speed. "fast" = one send per primitive (as before).
#: Honest mode starts at "normal" even though the oracle default stays "fast"; the Display menu sets both.
HONEST_SMOOTH = {
    "fast": None,
    "normal": {"step": 0.0625, "turn": 15.0, "look": 5.0, "doors": 6, "pause": 0.03},
    "cinematic": {"step": 0.05, "turn": 8.0, "look": 3.0, "doors": 12, "pause": 0.05},
}
_HONEST_MOTION = {"speed": "normal"}


def _honest_smooth():
    """The live smooth profile HonestAgent reads before every logical primitive (None = plain sends)."""
    return HONEST_SMOOTH.get(_HONEST_MOTION.get("speed"), HONEST_SMOOTH["normal"])


def _mp():
    """The active motion profile (always a valid dict, even if _MOTION was set to something unknown)."""
    return MOTION_PROFILES.get(_MOTION.get("speed"), MOTION_PROFILES[MOTION_DEFAULT])


def _apply_motion(ag):
    """Push the active profile onto the agent so every locomotion generator -- including the composite skills
    that never take a `step`/`turn` argument -- animates at the chosen granularity."""
    mp = _mp()
    try:
        ag.motion["step"], ag.motion["turn"] = mp["step"], mp["turn"]
    except Exception:
        pass


def _publish_frame(jpg):
    """Append one rendered frame to the fan-out ring and bump the sequence. Caller must hold ``_frame_cv``."""
    _state["jpeg"] = jpg
    _state["seq"] += 1
    _frames.append((_state["seq"], jpg))


def _stream_pending(last):
    """The frames a /stream client sitting at ``last`` should write next, as ``[(seq, jpeg)]``.

    ``last is None`` means a fresh connection: start at the NEWEST frame rather than replaying the whole
    backlog into a browser that just opened the page. A client that fell further behind than STREAM_BACKLOG is
    fast-forwarded and the frames it will never see are counted into ``_state["dropped"]`` -- reported in
    /status, because a stream that silently skips motion is exactly the failure this replaced.

    Caller must hold ``_frame_cv``.
    """
    cur = _state["seq"]
    if last is None or cur == last:
        return [(cur, _state["jpeg"])]                 # first frame, or a keepalive resend
    pending = [(sq, j) for sq, j in _frames if sq > last]
    if not pending:
        return [(cur, _state["jpeg"])]
    if pending[0][0] > last + 1:
        _state["dropped"] += pending[0][0] - last - 1
    return pending


def _hold(n, label):
    """Yield `n` extra frames without acting.

    AI2-THOR applies PickupObject / SliceObject / ToggleObjectOn in a single controller step: the object is
    whole in frame N and sliced in frame N+1, which at 20 fps is not something a human can see happen. Holding
    the camera either side of the call is the only way to make an instantaneous primitive observable.
    """
    for _ in range(max(0, int(n))):
        yield label

# Per-frame increments for smooth continuous driving — one small step each rendered frame while a key is held.
_DRIVE_STEP = {"move": 0.04, "rotate": 2.0, "look": 2.0}
# Auto-stop driving this long after the last /drive keepalive — survives a lost keyup, lost focus, or a closed tab so
# a held key can NEVER hang the worker. The browser re-sends /drive?...&on=1 every ~150ms while a key is down.
DRIVE_TIMEOUT = 0.4


def _log(msg):
    with _lock:
        _state["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        _state["log"] = _state["log"][-80:]
    # The transcript the operator reads is a ring of 80 lines in RAM: the reason a run failed scrolls away
    # within seconds and is gone at restart. Tee it, at DEBUG, into the durable log where it sits on the same
    # timeline as the writes it explains.
    _th("emit", "console.log", summary=str(msg)[:200], level="DEBUG", actor="system:webapp",
        payload={"message": str(msg)})


# ---------------------------------------------------------------- action helpers
def _animate_openness(ag, oid, target):
    """GENERATOR: swing a door from where it is now to `target` openness in profile-sized increments.

    AI2-THOR's OpenObject is instantaneous -- the fridge is shut in frame N and wide open in frame N+1 -- so
    without this the single most legible action in the whole kitchen is the one you cannot watch. If the
    simulator refuses an intermediate openness we stop stepping and land the object with the canonical
    primitive, so the END STATE is identical to the un-animated path either way.
    """
    steps = max(1, int(_mp()["openness_steps"]))
    obj = ag.object_by_id(oid) or {}
    cur = obj.get("openness")
    start = float(cur if cur is not None else (1.0 if obj.get("isOpen") else 0.0))
    if steps > 1 and abs(target - start) > 1e-3:
        for k in range(1, steps):                       # intermediate frames only -- the last throw is below
            v = start + (target - start) * (k / float(steps))
            ag.open(oid, openness=max(0.02, min(1.0, v)), force=True)
            if not ag.success:
                break                                   # incremental openness refused: fall through to the throw
            yield ("opening" if target > start else "closing")
    # Land on the exact state with the canonical primitive: AI2-THOR closes with CloseObject, and an openness
    # of exactly 0 is not a value OpenObject reliably accepts.
    if target <= 1e-6:
        ag.close(oid, force=True)
    else:
        ag.open(oid, openness=min(1.0, target), force=True)


def _open_parent(ag, obj, opened):
    """Generator: open the closed container `obj` is sitting inside, if there is one, recording it in `opened`.

    Execution-side only. This reads AI2-THOR's containment (`parentReceptacles`) for an object the agent has
    already been told to act on, which is the same latitude every other skill takes when it resolves an
    objectId -- it is the PLANNER's view of the room that perceived mode restricts, not how a decided action
    reaches its target. See docs/search.md.

    `opened` is the caller's list of doors to shut afterwards. Only what THIS step opened goes in it: a door the
    operator opened by hand, or one that was already open, is not ours to close.
    """
    parents = obj.get("parentReceptacles") or []
    for pid in parents[:1]:                       # innermost first; opening that one is enough to reach in
        p = ag.object_by_id(pid)
        if not p or not p.get("openable") or p.get("isOpen"):
            continue
        _log(f"opening {p.get('objectType')} to reach the {obj.get('objectType')}")
        yield from ag.walk_to(pid)
        yield from _animate_openness(ag, pid, 1.0)
        opened.append(pid)


def _shut_parents(ag, opened):
    """Generator: close the containers `_open_parent` opened for this step.

    Without this the tidy-up is only half done: `search_for` closes what IT opened, but a later `pick up the
    egg` opens the fridge to reach in and leaves it standing open -- and the next search finds it already open,
    so it is not that search's to close either. Every door the agent opens has to be owned by whoever opened it.
    """
    while opened:
        pid = opened.pop()
        p = ag.object_by_id(pid) or {}
        if not p.get("isOpen"):
            continue
        _log(f"closing the {p.get('objectType')} again")
        yield from _animate_openness(ag, pid, 0.0)


REACH_SKIP_M = 1.5                             # in view and within AI2-THOR interaction range (1.5 m) -> act now


def _approach_then(ag, oid, fn, label, close_in=True):
    """Walk to the object, get as close to it as the geometry allows, then act -- with the camera HELD either
    side of the action.

    It used to skip the walk whenever the object was already flagged `visible`, which in AI2-THOR means only
    "within visibilityDistance and in frame" -- so a pickup could happen from 1.4 m away, across a counter,
    with the agent never having moved. Now it always approaches (unless the thing is already in hand) and then
    edges in; `close_in` stops the moment a step fails or stops closing the gap, so this costs nothing when the
    agent is already standing at the object.

    `fn` may return a generator (an animated primitive such as open/close), in which case it is advanced one
    frame at a time in place of the action; otherwise it is a one-shot call and the surrounding holds are what
    make it visible. `ag.success`/`ag.error` always describe the LAST controller call, so the log line below
    reports the real outcome in both cases.
    """
    opened_parents = []                # doors THIS step opened, shut again once the action is done
    o = ag.object_by_id(oid)
    if o and not o.get("isPickedUp"):
        # If the target sits inside a CLOSED container, open that first. Otherwise the action fails for a
        # reason that has nothing to do with the action -- and it is why a search no longer has to leave the
        # fridge hanging open on the chance that a "pick it up" follows.
        yield from _open_parent(ag, o, opened_parents)
        o = ag.object_by_id(oid) or o
        # already in view and within arm's reach: act now (2026-10-05: re-planning a walk to an object the agent
        # was standing at made every open/close ~4x slower than the old console)
        if not (o.get("visible") and float(o.get("distance", 9.0)) <= REACH_SKIP_M):
            yield from ag.walk_to(oid)
            if close_in:
                yield from ag.close_in(oid)
    hold = _mp()["hold"]
    yield from _hold(hold, label + "…")
    r = fn()
    if hasattr(r, "__next__"):
        yield from r
    _log(f"{label}: {'ok' if ag.success else ag.error[:48]}")
    yield from _hold(hold, label)
    yield from _shut_parents(ag, opened_parents)
    yield "done"


def _step_label(step):
    a = (step.get("action") or "?").replace("_", " ")
    o = step.get("object") or ""
    to = step.get("to") or ""
    return (a + (f" → {o}" if o else "") + (f" ▸ {to}" if to else "")).strip()


# action -> (state key, the value an instance must currently have for the action to DO anything)
_ACTIONABLE = {"open": ("isOpen", False), "close": ("isOpen", True),
               "toggle_on": ("isToggled", False), "toggle_off": ("isToggled", True),
               "dirty": ("isDirty", False), "clean": ("isDirty", True),
               "fill": ("isFilledWithLiquid", False), "fill_with_liquid": ("isFilledWithLiquid", False),
               "empty": ("isFilledWithLiquid", True), "empty_liquid": ("isFilledWithLiquid", True)}


def _resolve_for(ag, action, otype):
    """`ag.resolve`, but among several instances of a type prefer one the action can actually change.

    A kitchen has up to 28 cabinets and 6 stove knobs, and `resolve` returns the nearest. So "close the drawer"
    went to whichever drawer was closest -- usually a closed one -- while the open drawer stayed open, and the
    step failed as "already in that state". Choosing the nearest OPEN drawer is what the instruction means.
    With one instance, or none in the useful state, this is exactly `resolve`."""
    obj = ag.resolve(otype)
    want = _ACTIONABLE.get(action)
    if not obj or not want or obj.get(want[0]) == want[1]:
        return obj
    try:
        same = [o for o in ag.objects() if o["objectType"] == obj["objectType"] and o.get(want[0]) == want[1]]
    except Exception:
        return obj
    return min(same, key=lambda o: o.get("distance", 9e9)) if same else obj


def _exec_one(ag, step):
    """Return a GENERATOR (animated by the worker) or None for one-shot actions."""
    _apply_motion(ag)                  # every generator below (composites included) inherits the profile
    action = str(step.get("action") or "").lower().strip()
    ot = al.canonical_type(step.get("object"))
    to = al.canonical_type(step.get("to") or step.get("arg"))
    comp = {
        "pick_and_place": lambda: ag.pick_and_place(ot, to), "put_in": lambda: ag.put_in(ot),
        "heat": lambda: ag.heat_object(ot), "cool": lambda: ag.cool_object(ot),
        "cook": lambda: ag.cook_food(ot), "wash": lambda: ag.wash_object(ot),
        "look_around": lambda: ag.look_around(), "scan_for": lambda: ag.scan_for(ot),
        "search_for": lambda: _search_gen(ag, ot),
        "open_all": lambda: ag.open_all(ot), "close_all": lambda: ag.close_all(ot),
        "turn_on_all": lambda: ag.switch_all(True, ot), "turn_off_all": lambda: ag.switch_all(False, ot),
        "collect_all": lambda: ag.collect_all(ot, to),
    }
    if action in comp:
        return comp[action]()
    nav = {"move_ahead": ag.move_ahead, "move_back": ag.move_back, "move_left": ag.move_left,
           "move_right": ag.move_right, "rotate_left": ag.rotate_left, "rotate_right": ag.rotate_right,
           "look_up": ag.look_up, "look_down": ag.look_down, "crouch": ag.crouch, "stand": ag.stand}
    if action in nav:
        nav[action](); return None
    if action == "drop":
        # force: AI2-THOR refuses a plain drop whenever the held object touches anything -- the agent's own
        # collider included ("must be clear of all other collision first") -- which is most of the time right
        # after a pickup. Every other manipulation below already forces.
        ag.drop(force=True); return None
    if action in ("done", "pass", "wait", "noop"):
        return None
    obj = _resolve_for(ag, action, ot)
    if not obj:
        _log(f"skip {action}: no '{ot}'"); return None
    oid = obj["objectId"]
    if action == "go_to":
        return ag.walk_to(oid)
    if action == "face_object":
        return ag.turn_to_object(oid)
    if action in ("open", "close"):     # the one primitive AI2-THOR CAN animate -- swing it
        return _approach_then(ag, oid, lambda: _animate_openness(ag, oid, 1.0 if action == "open" else 0.0),
                              f"{action} {obj['objectType']}")
    # PLANNER_SYSTEM has always offered `fill` (arg = water|coffee|wine) and `empty`, but neither was dispatched
    # here: a plan that used them fell through to "skip unknown" and did nothing. `to` doubles as the liquid
    # because canonical_type(arg) is what the line above parsed it into.
    if action in ("fill", "fill_with_liquid"):
        liquid = str(step.get("arg") or step.get("to") or "water").strip().lower()
        liquid = liquid if liquid in ("water", "coffee", "wine") else "water"
        return _approach_then(ag, oid, lambda: ag.fill_with_liquid(oid, liquid, force=True),
                              f"fill {obj['objectType']} with {liquid}")
    if action in ("empty", "empty_liquid"):
        return _approach_then(ag, oid, lambda: ag.empty_liquid(oid, force=True), f"empty {obj['objectType']}")
    objmap = {"pickup": ag.pickup, "open": ag.open, "close": ag.close, "toggle_on": ag.toggle_on,
              "toggle_off": ag.toggle_off, "slice": ag.slice, "cook": ag.cook, "break": ag.break_object,
              "dirty": ag.dirty, "clean": ag.clean, "put": ag.put}
    fn = objmap.get(action)
    if not fn:
        _log(f"skip unknown '{action}'"); return None
    return _approach_then(ag, oid, lambda: fn(oid, force=True), f"{action} {obj['objectType']}")


# Actions whose generator already writes its own "<action> <ObjectType>: ok" line (through _approach_then).
# Everything NOT in here -- the composites (scan_for, pick_and_place, heat, wash, ...) and the nav primitives --
# used to finish in total silence, so a one-step plan such as "find the apple" produced a transcript with the
# plan on one line, the reward on the next, and no evidence in between that anything had happened at all.
_SELF_LOGGING_ACTIONS = frozenset((
    "pickup", "open", "close", "toggle_on", "toggle_off", "slice", "cook", "break",
    "dirty", "clean", "put", "fill", "fill_with_liquid", "empty", "empty_liquid",
))


_OBJ_ACT = {
    "pick up": "pickup", "put here (held →)": "put", "open": "open", "close": "close",
    "toggle on": "toggle_on", "toggle off": "toggle_off", "slice": "slice", "cook": "cook",
    "break": "break_object", "dirty": "dirty", "clean": "clean", "fill": "fill_with_liquid",
    "empty": "empty_liquid", "use up": "use_up", "push": "push", "pull": "pull",
}


def _search_event(kind, text, data):
    """Narration sink for `ThorAgent.search_for`: the transcript line plus a durable trace event.

    The search's decisions -- what it recalled, the route it chose and why, what it opened, whether it saw
    anything -- are the whole point of it being a strategy rather than a spin. They exist nowhere else once the
    generator is exhausted, so they go into the event log on the same timeline as the writes they explain.
    """
    _log(text)
    _th("emit", f"search.{kind}", summary=str(text)[:200], scene=_scene_now(), actor="system:search",
        level=("WARNING" if kind in ("notfound", "skip", "error") else "DEBUG"),
        payload=dict(data or {}, kind=kind))


def _search_gen(ag, object_type):
    """Build the search generator with whatever knowledge sources happen to be available.

    Every source is optional by design: memory, the knowledge base and the detector each degrade to "skip that
    input" rather than failing the step, so a search still runs on a cold agent in an untrained scene -- it just
    falls back further down the authority order in sim/search.py.
    """
    det = _detector_for(ag.scene) if _PERCEPTION["mode"] == "perceived" else None
    try:
        mem = _get_memory()
    except Exception as exc:
        _log(f"search: memory unavailable ({exc}) - searching without recall")
        mem = None
    try:
        kb = _get_kb()
    except Exception:
        kb = None
    return ag.search_for(object_type, detector=det, memory=mem, kb=kb, scene=ag.scene,
                         speed=_MOTION["speed"], turn=_mp()["turn"], on_event=_search_event)


def _act_obj(ag, oid, action, liquid="water", openness=1.0):
    _apply_motion(ag)
    a = action.lower().strip()
    if a in ("go to", "goto"):
        return ag.walk_to(oid)
    if a in ("look at", "lookat"):
        return ag.turn_to_object(oid)
    name = _OBJ_ACT.get(a)
    if not name:
        _log(f"unknown action {action}"); return None
    if name == "open":
        act = lambda: _animate_openness(ag, oid, float(openness))
    elif name == "close":
        act = lambda: _animate_openness(ag, oid, 0.0)
    elif name == "fill_with_liquid":
        act = lambda: ag.fill_with_liquid(oid, liquid, force=True)
    else:
        fn = getattr(ag, name)
        act = lambda: fn(oid, force=True)
    return _approach_then(ag, oid, act, f"{a} {oid.split('|')[0]}")


def _drive_active():
    """True iff a movement key is held AND its keepalive is fresh (within DRIVE_TIMEOUT) — else clear and return False."""
    if _drive["action"] and (time.time() - _drive["t"]) < DRIVE_TIMEOUT and not _honest_acting():
        return True
    _drive["action"] = None
    return False


def _do_drive_step(ag):
    """One small movement step for smooth continuous driving (called once per worker frame while a key is held)."""
    a = _drive["action"]
    try:
        if a in ("move_ahead", "move_back", "move_left", "move_right"):
            getattr(ag, a)(_DRIVE_STEP["move"])
        elif a in ("rotate_left", "rotate_right"):
            getattr(ag, a)(_DRIVE_STEP["rotate"])
        elif a in ("look_up", "look_down"):
            getattr(ag, a)(_DRIVE_STEP["look"])
    except Exception:
        pass


def _set_view(ag, mode, depth=None):
    """``depth`` None = honest-aware (the honest agent's depth sensor stays on whatever the view needs)."""
    keep = _honest_on() if depth is None else bool(depth)
    if mode in ("map", "mapnav"):                          # both use the top-down camera; mapnav overlays the nav map
        if not _view["map_cam"]:
            ag.add_top_down_camera()
            _view["map_cam"] = True
    else:
        need = {"depth": dict(depth=True), "instance": dict(instance=True),
                "semantic": dict(semantic=True), "normals": dict(normals=True)}
        if mode in need:
            ag.set_sensors(**{"depth": keep, **need[mode]})
            _view["map_cam"] = False
        elif mode in ("rgb", "detect"):    # 'detect' renders RGB, then draws YOLO boxes in the worker pipeline
            ag.set_sensors(depth=keep)
            _view["map_cam"] = False
    _view["mode"] = mode
    with _lock:
        _state["view"] = mode
    _log(f"view: {mode}")


def _do_capture(ag, allstreams=False):
    if allstreams:
        kinds = ["rgb", "depth", "instance", "semantic", "normals", "map"]
        ag.set_sensors(depth=True, instance=True, semantic=True, normals=True)
        if not _view["map_cam"]:
            ag.add_top_down_camera(); _view["map_cam"] = True
    else:
        kinds = [_view["mode"] if _view["mode"] in ("rgb", "depth", "instance", "semantic", "normals", "map") else "rgb"]
    saved = ag.capture(directory="captures", kinds=kinds)
    _log(f"captured: {', '.join(os.path.basename(str(s)) for s in saved)}")


def _tracked_step(ag, i, step, ctx=None, episodes=None):
    """Generator wrapper that flags a plan step running -> done/failed while it animates.

    Also emits a PASSIVE environment-feedback record for the step: a before/after effect snapshot plus the
    oracle spatial observation (ag.spatial_snapshot()), funnelled through EnvFeedback.log_step. The whole
    feedback path is wrapped so a memory-backend failure only logs — it can NEVER break the worker or the plan
    (mirrors `_retrieve_memory`'s safety). The command/level come from `_state` (set by `_plan_and_run`).

    ``ctx`` is the trace context captured on the PLAN thread (this runs on the worker thread, which owns no
    run); ``episodes`` is the list this step appends its ``memory.episodes`` row id to, so loop closure can
    draw the ``procedure wasDerivedFrom episode`` edges that answer "which executed actions produced this
    knowledge-base entry?". Both are optional: without them the step still executes exactly as before.
    """
    with _lock:
        if _state["plan"]:
            _state["plan"]["active"] = i
            _state["plan"]["steps"][i]["status"] = "running"
        cur_cmd = _state.get("cur_cmd")
        cur_level = _state.get("cur_level")
        scene = _state.get("scene") or None
    with _tbound(ctx):
        # One TOOL span per plan step. Every AI2-THOR primitive executed inside it (through the do() hook) and
        # every memory write it causes lands underneath, so the run tree reads command -> step -> action -> write.
        with _tspan(f"execute_tool {_step_label(step)}"[:120], kind="TOOL", op="execute_tool", step_ix=i,
                    scene=scene, actor="system:webapp",
                    input={"step": step, "step_ix": i, "command": cur_cmd, "level": cur_level}) as sid:
            _th("emit", "step.started", summary=f"[{i}] {_step_label(step)}", scene=scene, span_id=sid,
                actor="system:webapp", payload={"step_ix": i, "step": step, "command": cur_cmd})
            before = None
            try:
                before = ag.snapshot()                      # effect-key snapshot BEFORE the action (opaque; audit)
            except Exception:
                before = None
            gen = _exec_one(ag, step)
            if gen is not None and hasattr(gen, "__next__"):
                yield from gen
            ok = ag.success
            err = None if ok else (getattr(ag, "error", None) or "action failed")
            if str(step.get("action") or "").lower().strip() not in _SELF_LOGGING_ACTIONS:
                _log(f"{_step_label(step)}: {'ok' if ok else str(err)[:48]}")
            with _lock:
                if _state["plan"]:
                    _state["plan"]["steps"][i]["status"] = "done" if ok else "failed"
                    if err:                                 # kept for the Phase-7 loop-closure lesson (D5)
                        _state["plan"]["steps"][i]["error"] = str(err)[:200]
            try:                                            # PASSIVE logging — never allowed to break the worker
                after = ag.snapshot()
                obs = ag.spatial_snapshot()
                res = _get_feedback().log_step(
                    scene, command=cur_cmd, level=cur_level, action=step.get("action"), args=step, success=ok,
                    error=(None if ok else (getattr(ag, "error", None) or "failed")),
                    before=before, after=after, observations=obs,
                    run_id=(ctx or {}).get("run_id"), step_ix=i, span_id=sid)
                if episodes is not None and isinstance(res, dict) and res.get("episode_id"):
                    episodes.append(res["episode_id"])
            except Exception as exc:
                _log(f"env-feedback log skipped: {exc}")
            _th("emit", "step.finished", summary=f"[{i}] {_step_label(step)} -> {'ok' if ok else 'FAILED'}",
                level=("ERROR" if not ok else "DEFAULT"), scene=scene, span_id=sid, actor="system:webapp",
                payload={"step_ix": i, "success": bool(ok), "error": err, "step": step})


# ---------------------------------------------------------------- RAG grounding (lazy singleton; ablate via flag)
RAG_ENABLED = True            # set False to ablate retrieval — the planner then runs with NO memory grounding
ENV_FEEDBACK_ENABLED = True   # set False to ablate Phase-5 environment feedback (passive logging + active exploration)
# Phase-6: the LinUCB contextual bandit that picks ONE planning strategy per command. env-overridable so an ablation
# run can disable it (BANDIT_ENABLED=0) — when disabled the bandit still logs decisions but always plays the safe
# default arm (reuse_memory), so behaviour is byte-for-byte the Phase-5 path.
BANDIT_ENABLED = os.environ.get("BANDIT_ENABLED", "1") not in ("0", "false", "False", "no")
# Phase-7: the self-improving LOOP-CLOSURE toggle -- its own ablation axis, independent of RAG_ENABLED (which
# governs retrieval). env-overridable like BANDIT_ENABLED and flippable at runtime via GET /loop?on=1|0. When
# disabled: NO KB write-back, NO reindex, and the bandit's LinUCB updates are FROZEN (decide still runs but
# nothing learns) -- behaviour reproduces the Phase-6 path byte-for-byte.
LOOP_ENABLED = os.environ.get("LOOP_ENABLED", "1") not in ("0", "false", "False", "no")
# Phase-7 D11: retry is MANUAL by design (the user re-issues the command; the second run benefits from what
# the loop just wrote back). AUTO_RETRY=1 allows at most ONE automatic re-plan after a failed run's loop
# closure -- default OFF, no UI beyond this flag.
AUTO_RETRY = os.environ.get("AUTO_RETRY", "0") in ("1", "true", "True", "yes")
_memory = None                # structured + RAG memory singleton, built on first command
_memory_lock = threading.Lock()
_feedback = None              # EnvFeedback policy layer over _memory (passive log_step + active exploration), lazy
_feedback_lock = threading.Lock()
_bandit = None                # select.Bandit (Phase-6 planning-strategy selector) over the shared memory.db, lazy
_bandit_mod = None
_bandit_lock = threading.Lock()
_kb = None                    # knowledge.KnowledgeBase JSON store, lazy
_kb_lock = threading.Lock()
_demo = {"active": False, "command": "", "steps": [], "scene": None, "started": 0.0}


# ---------------------------------------------------------------- reset
# Two scopes, kept apart on purpose. "session" is the one you press between experiments and should never make
# you hesitate; "all" throws away evidence and is the one that needs a deliberate second click in the UI.
RESET_SCOPES = ("session", "all")


def _reset_session(ag):
    """Worker job: reload the scene to its initial state and clear everything the console shows about the run.

    Deliberately NOT a settings reset. Motion detail, perception mode, camera view, model and endpoint are the
    operator's CHOICES, not run state; silently reverting them is how a reset button becomes something people
    are afraid to press.
    """
    ag.reset()                                   # reload this FloorPlan: objects, doors, agent pose, hands
    ag.last_search = {"found": False, "type": None, "receptacle": None, "stops_planned": 0,
                      "stops_visited": 0, "visited": [], "elapsed": 0.0, "reason": "no search yet"}
    _view["map_cam"] = False                     # the top-down camera does not survive a scene reload
    try:
        _set_view(ag, _view["mode"])             # re-apply the sensors the chosen view needs
    except Exception:
        pass
    with _lock:
        _state["log"] = []
        _state["plan"] = None
        _state["cur_cmd"] = None
        _state["cur_level"] = None
        _state["loop_last"] = None
        _state["demo"] = {"active": False, "steps": 0}
        _state["dropped"] = 0
    _demo.update(active=False, command="", steps=[], scene=None, started=0.0)
    _OBS["text"] = _OBS["oracle"] = ""
    _PERCEPTION["degraded"] = None
    _log("reset: scene reloaded, transcript cleared")


def _reset_learned():
    """Wipe everything the agent has LEARNED: knowledge base + its embedding sidecar, spatial memory, the
    bandit's weights, the loop reports, the audit logs and the trace store.

    Runs on the request thread -- none of it touches the AI2-THOR controller -- and drops every lazy singleton
    so the next command rebuilds them from nothing. Each store is closed before its file is removed, and each
    step is independently guarded: a reset that half-fails must still say what it managed, not raise.
    """
    done, failed = [], []

    def step(name, fn):
        try:
            fn()
            done.append(name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")

    global _memory, _kb, _feedback, _bandit

    def wipe_kb():
        global _kb
        path = getattr(_get_kb(), "path", os.path.join(HERE, "knowledge_base.json"))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"procedures": [], "lessons": [], "facts": []}, fh, ensure_ascii=False, indent=2)
        for extra in (os.path.splitext(path)[0] + "_embeddings.json",
                      os.path.splitext(path)[0] + "_audit.log"):
            if os.path.exists(extra):
                os.remove(extra)
        _kb = None

    def wipe_memory():
        global _memory, _feedback, _bandit
        path = None
        try:
            m = _get_memory()
            path = getattr(m, "path", None)
            m.close()
        except Exception:
            pass
        _memory = _feedback = _bandit = None      # bandit weights live in memory.db; drop it with the rest
        for f in ([path] if path else []) + [os.path.join(HERE, "memory_audit.log"),
                                             os.path.join(HERE, "memory_bandit_audit.log")]:
            if f and os.path.exists(f):
                os.remove(f)

    def wipe_reports():
        for f in (improve_reports_path(), os.path.join(HERE, "loop_reports.jsonl.lock")):
            if f and os.path.exists(f):
                os.remove(f)

    def wipe_trace():
        if _trace is None:
            return
        path = (_trace.status() or {}).get("db_path")
        _trace.close()                            # flush, stop the writer, close the connection
        for suffix in ("", "-wal", "-shm", "-journal"):
            f = (path or "") + suffix
            if path and os.path.exists(f):
                os.remove(f)
        _install_trace_hooks()                    # the next emit reopens a fresh store (enabled() re-latches)

    step("knowledge base", wipe_kb)
    step("memory + bandit", wipe_memory)
    step("loop reports", wipe_reports)
    step("trace store", wipe_trace)
    return {"cleared": done, "failed": failed}


def improve_reports_path():
    """Where the loop writes its reports (asked of `improve` so the two can never disagree)."""
    try:
        import improve as _improve
        return _improve.LOOP_REPORTS_PATH
    except Exception:
        return os.path.join(HERE, "loop_reports.jsonl")


def _get_memory():
    """Lazily build the structured+RAG memory singleton (imported lazily so webapp's top stays import-light)."""
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                import memory as _mem
                _memory = _mem.Memory()
    return _memory


def _get_kb():
    """Lazily build the JSON knowledge base used by the hybrid RAG layer."""
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                import knowledge as _knowledge
                _kb = _knowledge.KnowledgeBase(path=os.path.join(HERE, "knowledge_base.json"))
    return _kb


def _scene_now():
    with _lock:
        return _state.get("scene") or None


def _clean_teach_text(text):
    s = str(text or "").strip()
    s = re.sub(r"^(teach|correction|correct|remember)\s*:\s*", "", s, flags=re.I).strip()
    return s


def _obj_type(text):
    s = re.sub(r"\b(the|a|an)\b", " ", str(text or ""), flags=re.I).strip()
    s = re.sub(r"\s+", " ", s)
    return al.canonical_type(s) or s


def _parse_spatial(text):
    s = re.sub(r"^(spatial|where)\s*:\s*", "", str(text or "").strip(), flags=re.I).strip()
    patterns = [
        r"^(.+?)\s+(?:is\s+)?(?:in|inside|on|at)\s+(.+)$",
        r"^(.+?)\s*->\s*(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, s, flags=re.I)
        if m:
            return _obj_type(m.group(1)), _obj_type(m.group(2))
    return None, None


def _store_human_teaching(text):
    """Inline NL correction channel: fact/lesson/spatial writes with source=human."""
    raw = _clean_teach_text(text)
    if not raw:
        return {"ok": False, "error": "empty teaching"}
    scene = _scene_now()
    low = raw.lower()
    kb = _get_kb()

    # A teaching run of its own (set_ambient=False, see _rate_plan): the KB entry this writes must name the
    # human who typed it and the moment it happened, not whatever command happened to be running.
    with _trun(kind="feedback", command=raw, scene=scene, lang=_lang_of(raw),
               config={"channel": "teach"}, set_ambient=False):
        _th("emit", "feedback.received", summary=f"human teaching: {raw}"[:200], actor="human:arash",
            scene=scene, payload={"channel": "teach", "text": raw})
        return _store_human_teaching_body(raw, low, scene, kb)


def _store_human_teaching_body(raw, low, scene, kb):
    """The original body of :func:`_store_human_teaching`, unchanged except for its enclosing trace run."""
    try:
        if low.startswith("lesson:"):
            body = raw.split(":", 1)[1].strip()
            left, sep, right = body.partition("=>")
            if not sep:
                left, sep, right = body.partition("->")
            lesson = (right or body).strip()
            problem = left.strip() if sep else "human correction"
            command = None
            with _lock:
                command = _state.get("cur_cmd") or ((_state.get("plan") or {}).get("command"))
            entry = kb.add_lesson(command or problem or "human correction", problem or "human correction",
                                  lesson, why="inline human correction", scene=scene, source="human")
            _log(f"human lesson stored #{entry.get('id')}")
            return {"ok": True, "kind": "lesson", "entry": entry}

        if low.startswith("fact:"):
            body = raw.split(":", 1)[1].strip()
            entry = kb.add_fact(body, why="inline human correction", tags=[], source="human", scope=scene)
            _log(f"human fact stored #{entry.get('id')}")
            return {"ok": True, "kind": "fact", "entry": entry}

        if low.startswith("spatial:") or low.startswith("where:"):
            obj, rec = _parse_spatial(raw)
            if obj and rec:
                _get_memory().remember_location(scene, obj, receptacle=rec, track="perceived", source="human",
                                                confidence=1.0, why="inline human spatial correction")
                fact = kb.add_fact(f"{obj} is at {rec}" + (f" in {scene}" if scene else ""),
                                   why="inline human spatial correction", tags=[obj, rec],
                                   source="human", scope=scene)
                _log(f"human spatial correction: {obj} -> {rec}")
                return {"ok": True, "kind": "spatial", "object_type": obj, "receptacle": rec, "entry": fact}

        entry = kb.add_fact(raw, why="inline human correction", tags=[], source="human", scope=scene)
        _log(f"human fact stored #{entry.get('id')}")
        return {"ok": True, "kind": "fact", "entry": entry}
    except Exception as exc:
        _log(f"human teaching skipped: {exc}")
        return {"ok": False, "error": str(exc)}


def _demo_snapshot():
    with _lock:
        return {"active": bool(_demo["active"]), "command": _demo.get("command") or "",
                "steps": len(_demo.get("steps") or [])}


def _set_demo_state():
    with _lock:
        _state["demo"] = {"active": bool(_demo["active"]), "command": _demo.get("command") or "",
                          "steps": len(_demo.get("steps") or [])}


def _start_demo(command):
    with _lock:
        scene = _state.get("scene") or None
    _demo.update(active=True, command=(command or "").strip(), steps=[], scene=scene, started=time.time())
    _set_demo_state()
    _log(f"demo recording started{': ' + _demo['command'] if _demo['command'] else ''}")
    return {"ok": True, "demo": _demo_snapshot()}


def _record_demo_step(step):
    if not _demo.get("active"):
        return
    try:
        s = dict(step)
        s["ts"] = time.time()
        _demo.setdefault("steps", []).append(s)
        _set_demo_state()
    except Exception:
        pass


def _stop_demo(command=None, save=True):
    steps = list(_demo.get("steps") or [])
    cmd = (command or _demo.get("command") or "").strip()
    scene = _demo.get("scene") or _scene_now()
    _demo.update(active=False, command="", steps=[], scene=None, started=0.0)
    _set_demo_state()
    if not save:
        _log("demo recording discarded")
        return {"ok": True, "saved": False, "steps": len(steps)}
    if not cmd:
        cmd = f"human demonstration in {scene or 'scene'}"
    plan = [{k: v for k, v in s.items() if k in ("action", "object", "to", "arg")} for s in steps]
    with _trun(kind="feedback", command=cmd, scene=scene, lang=_lang_of(cmd),
               config={"channel": "demonstration", "steps": len(plan)}, set_ambient=False):
        _th("emit", "feedback.received", summary=f"teach-by-demonstration: {cmd} ({len(plan)} steps)"[:200],
            actor="human:arash", scene=scene,
            payload={"channel": "demonstration", "command": cmd, "plan": plan})
        try:
            entry = _get_kb().add_procedure(cmd, plan, why=f"teach-by-demonstration ({len(plan)} manual steps)",
                                            scene=scene, tags=["demo"], source="demo", confidence=1.0,
                                            scope=scene)
            _log(f"demo stored as procedure #{entry.get('id')} ({len(plan)} steps)")
            return {"ok": True, "saved": True, "steps": len(plan), "entry": entry}
        except Exception as exc:
            _log(f"demo save skipped: {exc}")
            return {"ok": False, "error": str(exc), "steps": len(plan)}


def _planner_action(label):
    a = str(label or "").lower().strip()
    if a in ("go to", "goto"):
        return "go_to"
    if a in ("look at", "lookat"):
        return "face_object"
    return _OBJ_ACT.get(a, a.replace(" ", "_"))


def _norm_action(action):
    return str(action or "").lower().strip().replace(" ", "_").replace("-", "_")


def _type_eq(a, b):
    return (str(a or "").replace(" ", "").lower() == str(b or "").replace(" ", "").lower())


def _scene_object_types():
    with _lock:
        objs = list(_state.get("objects") or [])
    return sorted({o.get("type") for o in objs if o.get("type")})


def _command_mentions_type(command, object_type):
    cmd = str(command or "").lower()
    ot = str(object_type or "")
    forms = {ot.lower(), re.sub(r"(?<=[a-z])(?=[A-Z])", " ", ot).lower()}
    alias_forms = {k for k, v in getattr(al, "TYPE_ALIASES", {}).items() if _type_eq(v, ot)}
    forms |= alias_forms
    return any(f and re.search(rf"\b{re.escape(f)}s?\b", cmd) for f in forms)


def _pickup_target(command):
    """Object type targeted by a pickup/take/grab command, or None."""
    cmd = str(command or "")
    if not re.search(r"\b(pick\s*up|pickup|grab|take|hold)\b", cmd, flags=re.I):
        return None
    types = _scene_object_types()
    phrase_matches = list(re.finditer(
        r"\b(?:pick\s*up|pickup|grab|take|hold)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?=\s+(?:and|then|after|before|from|off|on|in|inside|to)\b|[,.!?;]|$)",
        cmd, flags=re.I))
    if phrase_matches:
        frag = phrase_matches[-1].group(1)
        frag = re.sub(r"[^\w\s]", " ", frag).strip()
        frag = re.sub(r"\s+", " ", frag)
        frag_low = frag.lower()
        for ot in sorted(types, key=len, reverse=True):
            forms = {ot.lower(), re.sub(r"(?<=[a-z])(?=[A-Z])", " ", ot).lower()}
            alias_forms = {k for k, v in getattr(al, "TYPE_ALIASES", {}).items() if _type_eq(v, ot)}
            forms |= alias_forms
            if any(f and re.search(rf"\b{re.escape(f)}s?\b", frag_low) for f in forms):
                return ot
        if frag:
            can = al.canonical_type(frag)
            for ot in types:
                if _type_eq(can, ot) or _type_eq(frag, ot):
                    return ot
            return str(can or frag).strip().title().replace(" ", "")
    for ot in sorted(types, key=len, reverse=True):
        if _command_mentions_type(cmd, ot):
            return ot
    m = re.search(r"\b(?:pick\s*up|pickup|grab|take|hold)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:from|off|on|in|inside|to)\b|$)",
                  cmd, flags=re.I)
    if not m:
        return None
    frag = re.sub(r"[^\w\s]", " ", m.group(1)).strip()
    frag = re.sub(r"\s+", " ", frag)
    if not frag:
        return None
    can = al.canonical_type(frag)
    for ot in types:
        if _type_eq(can, ot) or _type_eq(frag, ot):
            return ot
    return str(can or frag).strip().title().replace(" ", "")


def _search_types(command):
    """Every object type a find/search/locate command names, in order. Empty when it is not a search command.

    Bilingual by construction: `agentic_rag.lexical_pivot` resolves both surface forms onto the same
    language-neutral AI2-THOR type, so "find the apple" and "سیب را پیدا کن" both come back as `Apple`.

    `scene_types` is deliberately NOT passed. It would sharpen the match, but "which object types exist in this
    room" is exactly the oracle that perceived mode withholds -- letting it in through the command parser would
    quietly hand the search half its answer.
    """
    try:
        import agentic_rag as _ar
        piv = _ar.lexical_pivot(str(command or ""))
    except Exception:
        return []
    if "search_for" not in (piv.get("verbs") or []):
        return []
    return list(piv.get("types") or [])


def _search_target(command):
    """The single object type a find/search/locate command is looking for, or None.

    Bilingual by construction: `agentic_rag.lexical_pivot` resolves both surface forms onto the same
    language-neutral AI2-THOR type, so "find the apple" and "سیب را پیدا کن" both come back as `Apple`.

    `scene_types` is deliberately NOT passed. It would sharpen the match, but "which object types exist in this
    room" is exactly the oracle that perceived mode withholds -- letting it in through the command parser would
    quietly hand the search half its answer.
    """
    types = _search_types(command)
    return types[0] if types else None


def _repair_plan_for_command(command, steps):
    """Deterministically patch a plan before execution: upgrade spin-in-place "searches" into real ones, and
    complete an incomplete simple pickup.

    Order matters. The search upgrade runs FIRST and unconditionally, because a search command has no pickup
    target and would otherwise take the early return below before ever being looked at.
    """
    repaired = []
    changed = []
    for step in (steps or []):
        s = dict(step or {})
        act = _norm_action(s.get("action"))
        if act in ("pick_up", "pickup_object"):
            s["action"] = "pickup"
            changed.append("normalized pickup action")
        repaired.append(s)

    # A search command whose plan only spins on the spot gets UPGRADED to a real search. This is also what
    # neutralises the two degenerate `find the apple -> [look_around]` procedures already in the knowledge base:
    # the bandit's replay_procedure arm will keep replaying them, and this turns each replay into a real search
    # instead of requiring the operator to hand-edit their own KB.
    st = _search_target(command)
    # A plan may name a BROADER type than the command did -- "find the wine bottle" produced a `search_for
    # Bottle` step. The search then succeeds on its own terms while the goal check, which scores against the
    # command, reports the object was never seen. One command must have one target, so a search step naming a
    # type the command never mentioned is retargeted (and said out loud). A step whose type IS one of the
    # command's -- "find the apple and the knife" -- is left alone.
    want = _search_types(command)
    if want:
        for x in repaired:
            if _norm_action(x.get("action")) != "search_for":
                continue
            obj = x.get("object")
            if obj and not any(_type_eq(obj, w) for w in want):
                x["object"] = want[0]
                changed.append(f"retargeted search_for {obj} -> {want[0]} (the command asked for {want[0]})")

    if st and not any(_norm_action(x.get("action")) == "search_for" for x in repaired):
        spin = [x for x in repaired if _norm_action(x.get("action")) in ("look_around", "scan_for")]
        if spin:
            for x in spin:
                x["action"], x["object"] = "search_for", st
            changed.append(f"upgraded {len(spin)} spin-in-place step(s) to search_for {st}")
        else:
            repaired.insert(0, {"action": "search_for", "object": st, "to": None, "arg": None})
            changed.append(f"added search_for {st}")

    target = _pickup_target(command)
    if not target:
        return repaired, changed
    with _lock:
        held = _state.get("held")
    if _type_eq(held, target):
        return repaired, changed

    def is_target_step(step, actions):
        act = _norm_action(step.get("action"))
        obj = step.get("object") or step.get("to") or step.get("arg")
        return act in actions and _type_eq(obj, target)

    has_pickup = any(is_target_step(s, {"pickup"}) for s in repaired)
    if has_pickup:
        return repaired, changed
    has_approach = any(is_target_step(s, {"go_to", "face_object", "scan_for", "search_for"}) for s in repaired)
    if not has_approach:
        repaired.append({"action": "go_to", "object": target, "to": None, "arg": None})
        changed.append(f"added go_to {target}")
    repaired.append({"action": "pickup", "object": target, "to": None, "arg": None})
    changed.append(f"added pickup {target}")
    return repaired, changed


def _live_goal_check(command, timeout=8):
    """Verify command-level success on the worker thread. Handles pickup/take/grab goals and SEARCH goals.

    The search branch exists because without it a search was scored from its step outcomes: `look around: ok`
    meant `environment=1.00`, so the loop stored "find the apple" as a proven procedure for a run in which no
    apple was ever seen. A search succeeds if and only if the agent actually saw the thing.
    """
    target = _pickup_target(command)
    if not target:
        st = _search_target(command)
        if not st:
            return {"applicable": False}
        result = {"applicable": True, "ok": False, "kind": "search", "target": st, "error": "timeout"}
        ev = threading.Event()

        def sjob(ag):
            try:
                ls = dict(getattr(ag, "last_search", None) or {})
                ok = bool(ls.get("found")) and _type_eq(ls.get("type"), st)
                result.update(ok=ok, error=None, search=ls)
            except Exception as exc:
                result.update(error=str(exc))
            finally:
                ev.set()

        _jobs.put(sjob)
        ev.wait(timeout=timeout)
        return result
    result = {"applicable": True, "ok": False, "kind": "pickup", "target": target, "held": None,
              "error": "timeout"}
    ev = threading.Event()

    def job(ag):
        try:
            held = ag.held_object()
            held_type = held.get("objectType") if held else "none"
            result.update(ok=_type_eq(held_type, target), held=held_type, error=None)
        except Exception as exc:
            result.update(error=str(exc))
        finally:
            ev.set()

    _jobs.put(job)
    ev.wait(timeout=timeout)
    return result


def _record_search_outcome(scene, command, goal):
    """Write back what a search established: a spatial belief on success, a "not here" lesson on failure.

    Uses the SINGLE-object memory writer. `EnvFeedback.observe_scene(ag.spatial_snapshot())` -- the established
    pattern elsewhere in this codebase -- reads ground-truth `parentReceptacles` for every object in the room,
    so one call would record the true location of the whole kitchen whether or not the agent looked at any of
    it. Under perceived mode that does not just overstate what was learned, it destroys the ablation.

    A failed search is evidence too: it rules places out, and the lesson naming them is what lets the next
    search start somewhere else.
    """
    ls = (goal or {}).get("search") or {}
    target = goal.get("target")
    try:
        if goal.get("ok"):
            rec = ls.get("receptacle") or None
            _get_memory().observe_location(
                scene, target, receptacle=rec, pos=None, state=None,
                track=("perceived" if _PERCEPTION["mode"] == "perceived" else "oracle"),
                source="environment", confidence=0.8,
                why=f"seen during a systematic search for {target!r}")
            _log(f"memory: {target} @ {rec or 'in view'}"
                 f" (track={'perceived' if _PERCEPTION['mode'] == 'perceived' else 'oracle'})")
        else:
            import search as _search
            les = _search.not_found_lesson(target, scene, ls.get("visited"))
            entry = _get_kb().add_lesson(command=str(command), problem=les["problem"], lesson=les["lesson"],
                                         why="a search that found nothing rules those places out",
                                         scene=scene, source="environment", confidence=0.6)
            # add_lesson returns the whole ENTRY dict, not an id -- printing it raw put a full record in the
            # transcript where a "#7" belonged.
            _log(f"loop: insert lesson #{(entry or {}).get('id', '?')} - {les['lesson'][:90]}")
    except Exception as exc:
        _log(f"search write-back skipped: {exc}")


def _record_goal_check(scene, command, level, goal):
    if not goal.get("applicable"):
        return
    try:
        ok = bool(goal.get("ok"))
        target = goal.get("target")
        held = goal.get("held")
        err = None if ok else f"goal failed: expected held={target}, got {held}"
        _get_memory().log_episode(scene=scene, command=command, level=level, action="goal_check",
                                  args={"target": target, "held": held}, success=ok, error=err,
                                  source="environment", why="command-level goal verification")
        if not ok:
            _get_memory().record_failure("goal_check", err, object_type=target, scene=scene, command=command)
    except Exception as exc:
        _log(f"goal-check memory skipped: {exc}")


def _promote_or_lesson_from_rating(raw):
    """Plan approve/rate channel: up stores procedure; down stores lesson."""
    try:
        with _lock:
            plan = dict(_state.get("plan") or {})
            scene = _state.get("scene") or None
            cur_cmd = _state.get("cur_cmd")
        command = plan.get("command") or cur_cmd or "rated plan"
        steps = plan.get("raw_steps") or []
        # Upsert (not plain add): with the loop on, a successful run already auto-stored this command+plan as an
        # environment procedure; a thumbs-up must REINFORCE that entry (bumping its provenance-weighted signal),
        # not append a near-duplicate human row that then splits retrieval and defeats the Phase-7 dedup.
        if raw > 0 and steps:
            res = _get_kb().upsert_procedure(command, steps, why="human thumbs-up on executed plan",
                                             scene=scene, tags=["rated-plan"], source="human", scope=scene)
            _log(f"rated plan {res.get('action')} procedure #{res.get('id')}")
        elif raw < 0:
            res = _get_kb().upsert_lesson(command, "human thumbs-down",
                                          "Do not repeat this plan without revising the failed steps.",
                                          why="human thumbs-down on executed plan", scene=scene, source="human",
                                          scope=scene, error="human thumbs-down")
            _log(f"rated plan {res.get('action')} lesson #{res.get('id')}")
    except Exception as exc:
        _log(f"rating memory promotion skipped: {exc}")


def _parse_bbox(raw):
    if not raw:
        return None
    try:
        vals = [float(x.strip()) for x in str(raw).split(",")]
        return vals if len(vals) == 4 else None
    except Exception:
        return None


def _store_perception_feedback(obj, receptacle="", bbox="", note=""):
    """Confirm/correct perception: persist a human label and optional spatial correction."""
    obj = _obj_type(obj)
    rec = _obj_type(receptacle) if receptacle else ""
    if not obj:
        return {"ok": False, "error": "object is required"}
    scene = _scene_now()
    box = {"object_type": obj, "source": "human"}
    bb = _parse_bbox(bbox)
    if bb:
        box["bbox_xyxy"] = bb
    if rec:
        box["receptacle"] = rec
    if note:
        box["note"] = str(note)
    try:
        # Its own feedback run (set_ambient=False, see _rate_plan): the label and the spatial correction this
        # writes are HUMAN evidence, and must not be filed under whatever command happens to be running.
        with _trun(kind="feedback", command=f"perception: {obj}" + (f" -> {rec}" if rec else ""),
                   scene=scene, config={"channel": "perception"}, set_ambient=False):
            _th("emit", "feedback.received",
                summary=f"human perception label: {obj}" + (f" -> {rec}" if rec else ""),
                actor="human:arash", scene=scene,
                payload={"channel": "perception", "object_type": obj, "receptacle": rec, "bbox": bb,
                         "note": note})
            os.makedirs(os.path.join(HERE, "captures"), exist_ok=True)
            image_path = os.path.join("captures", f"perception_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            with _lock:
                jpg = _state.get("jpeg") or b""
            if jpg:
                with open(os.path.join(HERE, image_path), "wb") as fh:
                    fh.write(jpg)
            else:
                image_path = f"live://{scene or 'unknown'}/{time.time():.3f}"
            label_id = _get_memory().add_perception_label(image_path, scene=scene, boxes=[box],
                                                          source="human",
                                                          why="human perception confirm/correct")
            spatial = None
            if rec:
                spatial = _get_memory().observe_location(scene, obj, receptacle=rec, track="perceived",
                                                         source="human", confidence=1.0,
                                                         why="human perception spatial correction")
            _log(f"perception feedback stored: {obj}" + (f" -> {rec}" if rec else ""))
            return {"ok": True, "label_id": label_id, "box": box, "spatial": spatial}
    except Exception as exc:
        _log(f"perception feedback skipped: {exc}")
        return {"ok": False, "error": str(exc)}


def _get_feedback():
    """Lazily build the EnvFeedback singleton over the shared Memory (imported lazily; ablated via the flag).

    EnvFeedback is the Phase-5 policy layer: it funnels raw episodes to the append-only tier, distils changed
    spatial beliefs (change-gated), counts failures by signature, and picks needed types to actively explore.
    """
    global _feedback
    if _feedback is None:
        with _feedback_lock:
            if _feedback is None:
                import feedback as _fb
                _feedback = _fb.EnvFeedback(_get_memory(), enabled=ENV_FEEDBACK_ENABLED)
    return _feedback


def _load_bandit_module():
    """Load local sim/select.py by path; plain `import select` resolves to Python's stdlib extension module."""
    global _bandit_mod
    if _bandit_mod is None:
        path = os.path.join(HERE, "select.py")
        spec = importlib.util.spec_from_file_location("phase6_select", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load local select.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _bandit_mod = mod
    return _bandit_mod


def _get_bandit():
    """Lazily build the Phase-6 LinUCB bandit over the same sqlite DB used by Memory."""
    global _bandit
    if _bandit is None:
        with _bandit_lock:
            if _bandit is None:
                mod = _load_bandit_module()
                # updates_enabled follows the Phase-7 loop toggle (D2): loop off => the bandit still decides
                # (and logs) but reward() is an audited no-op, so nothing learns during ablation runs.
                _bandit = mod.Bandit(_get_memory().path, enabled=BANDIT_ENABLED, updates_enabled=LOOP_ENABLED)
    return _bandit


def _retrieve_memory(text):
    """Best-effort grounding for `text`: fused known-locations + retrieved procedures/lessons/facts, or ''.

    Wrapped so a memory/Ollama failure NEVER breaks planning — on any error it logs and returns ''. When it returns
    '' (disabled, nothing relevant, or backend down) the planner prompt is byte-for-byte identical to the no-RAG path.
    """
    if not RAG_ENABLED:
        return ""
    try:
        with _lock:
            scene = _state.get("scene") or None
        m = _get_memory()
        return (m.recall_text(scene, text) or "") if m is not None else ""
    except Exception as exc:
        _log(f"memory retrieval skipped: {exc}")
        return ""


def _retrieve_memory_hits(text):
    """The KB entries (``[{id, kind}]``) the hybrid retriever surfaces for `text` -- the provenance record of
    what actually grounded the plan, for the loop report's ``retrieved`` field (D13).

    Mirrors the retriever ``memory.recall_text`` injects with: a ``rag.HybridRAG`` over the SAME KB, scene-
    filtered, k=5 -- so the report reflects the keyword+vector entries actually put in the prompt, not a
    separate keyword-only re-query (which ranks differently and can credit entries the prompt never saw).
    Spatial beliefs fused by ``recall_text`` are not KB entries and correctly carry no id. Returns [] when RAG
    is off, nothing matched, or the embedding backend is unavailable (then injection degraded the same way).
    """
    if not RAG_ENABLED:
        return []
    try:
        with _lock:
            scene = _state.get("scene") or None
        import rag
        hits = rag.HybridRAG(kb=_get_kb()).search(text, scene=scene, k=5)
        return [{"id": h["entry"].get("id"), "kind": h.get("kind")} for h in hits]
    except Exception as exc:
        _log(f"retrieval provenance skipped: {exc}")
        return []


def _failure_seen(scene):
    """True when the Phase-5 failure tier has any relevant recent failure for the scene."""
    try:
        return bool(_get_memory().top_failures(scene=scene, limit=1))
    except Exception as exc:
        _log(f"failure context skipped: {exc}")
        return False


def _match_procedure(text):
    """Best stored KB procedure for `text`: (score 0..1, entry|None) -- the replay_procedure eligibility signal.

    Score is the Jaccard similarity between the CONTENT-word token sets (knowledge._tokens, which drops
    stopwords like 'the'/'up' and short tokens), forced to 1.0 on an exact normalized-command match. Keying on
    content words -- and additionally requiring the stored plan's object type(s) to appear in the command --
    stops a one-object swap from clearing the threshold: 'pick up the tomato' must NOT replay a stored 'pick up
    the apple' procedure (bare normalized-token Jaccard scores that at exactly 0.60, the eligibility cutoff,
    because the three shared stopwords dominate). The bandit gates the replay arm on this score
    (select.REPLAY_MATCH_THRESHOLD); webapp keeps the matched ENTRY so the winning arm can execute its stored
    plan directly.

    Replay executes stored KB knowledge, so it is part of the retrieval channel: when RAG_ENABLED is False
    (the no-grounding ablation) replay is disabled too, returning (0.0, None), so the arm cannot leak KB
    knowledge into the retrieval-off condition. Exception-safe: (0.0, None) on any error.
    """
    if not RAG_ENABLED:
        return 0.0, None
    try:
        import knowledge as _kn
        kb = _get_kb()
        norm = _kn.normalize_command
        a = _kn._tokens(text)                                 # content words only (stopword-stripped)
        cmd_words = set(norm(text).split())                  # full normalized words, for the object-mention check
        if not a:
            return 0.0, None
        best, best_score = None, 0.0
        for h in kb.search(text, kinds=("procedures",), limit=3):
            e = h["entry"]
            plan = e.get("plan") or []
            if not plan:
                continue
            # The stored plan's object types must be named in the command, so a swapped object disqualifies
            # replay even when the surrounding verbs match. Empty-object steps (e.g. bare 'look') don't gate.
            plan_objs = {str(s.get("object")).lower() for s in plan
                         if isinstance(s, dict) and s.get("object")}
            if plan_objs and not (plan_objs <= cmd_words):
                continue
            b = _kn._tokens(e.get("command"))
            if not b:
                continue
            score = 1.0 if norm(text) == norm(e.get("command")) else len(a & b) / float(len(a | b))
            if score > best_score:
                best, best_score = e, score
        return best_score, best
    except Exception as exc:
        _log(f"procedure match skipped: {exc}")
        return 0.0, None


def _decide_plan_strategy(text, scene, memory_block, procedure_match=None, procedure_entry=None):
    """Return a bandit decision dict; on any error, fall back to Phase-5 behavior without breaking planning."""
    try:
        bandit = _get_bandit()
        x = bandit.featurize(command=text, scene=scene,
                             memory_hit=bool(memory_block), failure_seen=_failure_seen(scene))
        dec = bandit.decide(x, scene=scene, command=text,
                            procedure_match=procedure_match, procedure_entry=procedure_entry) or {}
        arm = dec.get("arm") or "reuse_memory"
        legal = tuple(getattr(_load_bandit_module(), "ARMS", ()) or ())  # never hardcode -- a new arm must fire
        if arm not in (legal or ("reuse_memory", "plan_fresh", "explore_then_plan")):
            arm = "reuse_memory"
        dec["arm"] = arm
        return dec
    except Exception as exc:
        _log(f"bandit decision skipped: {exc}")
        return {"arm": "reuse_memory", "decision_id": -1, "scores": {}}


def _reward_plan(decision_id, scene, command, override=None, reason="environment"):
    """Credit the selected arm from the latest environment-scored episode batch, if one exists."""
    try:
        reward = float(override) if override is not None else _get_memory().plan_reward(scene, command)
        if reward is None:
            _log("bandit reward skipped: no scored episode batch")
            return None
        _get_bandit().reward(decision_id, reward, source="environment")
        _log(f"bandit reward: {reason}={float(reward):.2f}")
        return float(reward)
    except Exception as exc:
        _log(f"bandit reward skipped: {exc}")
        return None


def _rate_plan(decision_id, rating):
    """Apply a human thumb to Memory and the bandit. Returns a JSON-serializable status dict."""
    try:
        did = int(decision_id)
    except Exception:
        return {"ok": False, "error": "bad decision_id"}
    try:
        raw = float(rating)
    except Exception:
        return {"ok": False, "error": "bad rating"}
    if raw not in (-1.0, 1.0):
        return {"ok": False, "error": "rating must be 1 or -1"}

    with _lock:
        plan = dict(_state.get("plan") or {})
        scene = _state.get("scene") or None
        command = plan.get("command") or _state.get("cur_cmd")
    if plan.get("decision_id") not in (None, did) and did >= 0:
        _log(f"rating: decision_id {did} does not match current plan {plan.get('decision_id')}")

    # Its OWN run, with set_ambient=False: a thumbs-up arrives on an HTTP handler thread while the command run
    # may still be open on the plan thread, and stealing the process default would file that command's episodes
    # under this rating.
    with _trun(kind="feedback", command=command, scene=scene, level=_state.get("cur_level"),
               lang=_lang_of(command), decision_id=(did if did >= 0 else None),
               config={"channel": "rating", "rating": raw}, set_ambient=False):
        _th("emit", "feedback.received", summary=f"human thumbs-{'up' if raw > 0 else 'down'} on {command!r}"[:200],
            actor="human:arash", scene=scene,
            payload={"channel": "rating", "rating": raw, "decision_id": did, "command": command})
        rows = 0
        try:
            rows = _get_memory().set_rating(scene, command, raw)
        except Exception as exc:
            _log(f"rating memory skipped: {exc}")

        reward = 1.0 if raw > 0 else 0.0
        try:
            _get_bandit().reward(did, reward, source="human")
        except Exception as exc:
            _log(f"rating bandit skipped: {exc}")
        _promote_or_lesson_from_rating(raw)

    _log(f"rating: {'up' if raw > 0 else 'down'} decision={did} rows={rows}")
    return {"ok": True, "decision_id": did, "rating": raw, "reward": reward, "rows": rows}


def _bandit_status():
    """Debug snapshot for /bandit; never raises to the HTTP layer."""
    out = {"enabled": BANDIT_ENABLED, "loop_enabled": LOOP_ENABLED, "stats": {}, "plan": {}}
    try:
        out["stats"] = _get_bandit().stats()                 # includes 'updates_enabled' (the Phase-7 freeze)
    except Exception as exc:
        out["error"] = str(exc)
    with _lock:
        p = _state.get("plan") or {}
        out["plan"] = {k: p.get(k) for k in ("decision_id", "arm", "scores", "command", "environment_reward")}
    return out


# ---------------------------------------------------------------- Phase-7 loop closure (toggle + status + hook)
def _set_loop_enabled(flag):
    """Runtime flip of the Phase-7 loop toggle; also freezes/unfreezes the bandit's LinUCB updates (D2)."""
    global LOOP_ENABLED
    LOOP_ENABLED = bool(flag)
    _log(f"self-improving loop: {'on' if LOOP_ENABLED else 'off'}")
    try:                                                     # an already-built bandit follows the toggle at once
        _get_bandit().set_updates_enabled(LOOP_ENABLED)
    except Exception as exc:
        _log(f"bandit freeze toggle skipped: {exc}")
    return LOOP_ENABLED


def _loop_summary(report):
    """Compact one-glance summary of a close_loop report for /status (the UI's single 400ms poll)."""
    outcome = report.get("outcome") or {}
    return {"ts": report.get("ts"), "command": report.get("command"),
            "success": bool(outcome.get("success")), "source": outcome.get("source"),
            "writes": [{"kind": w.get("kind"), "id": w.get("id"), "action": w.get("action")}
                       for w in (report.get("writes") or [])],
            "reindexed": report.get("reindexed"), "loop_enabled": bool(report.get("loop_enabled"))}


def _loop_status(limit="10"):
    """Toggle state + newest-first recent loop reports (from improve's jsonl trail) for GET /loop.

    Exception-safe data endpoint (pattern: /episodes): any failure lands in an {'error':...} field, never a 500.
    """
    out = {"enabled": LOOP_ENABLED, "auto_retry": AUTO_RETRY, "reports": []}
    try:
        n = max(1, int(limit))
        import improve
        if os.path.isfile(improve.LOOP_REPORTS_PATH):
            with open(improve.LOOP_REPORTS_PATH, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            for ln in lines[-n:]:
                try:
                    out["reports"].append(json.loads(ln))
                except Exception:
                    pass
            out["reports"].reverse()                         # newest first
    except Exception as exc:
        out["error"] = str(exc)
    with _lock:
        out["last"] = _state.get("loop_last")
    return out


def _close_loop_after_run(scene, text, level, arm, steps, goal, replay_entry, memory_hits, did, completed,
                          run_id=None, span_id=None, episode_ids=None):
    """Post-run Phase-7 hook: hand the executed outcome to improve.close_loop and surface the write-back.

    Runs on the plan thread AFTER the goal check + bandit reward, holding the KB write lock via the singleton.
    Reads per-step outcomes from _state['plan']['steps'] (truncated to len(steps), which drops the appended
    goal pseudo-step) under the lock, releases it, then does all IO. Returns the loop report or None. NEVER
    raises -- the caller additionally wraps it, mirroring the Phase-5 hook safety.

    Guarded against writing bogus knowledge from an unfinished run: if the step queue did not drain before the
    done-timeout (`completed` is False) or the live plan is no longer this run (a /reset or a new command
    changed _state['plan']['decision_id']), we skip write-back entirely rather than record still-'running'
    steps as failures or an empty plan as 'nothing executed'.
    """
    with _lock:
        plan = _state.get("plan") or {}
        matches = (plan.get("decision_id") == did)
        psteps = [dict(s) for s in (plan.get("steps") or [])]
    if not completed or not matches:
        _log("loop closure skipped: run did not finish cleanly (timeout or plan changed) - no write-back")
        return None
    psteps = psteps[:len(steps)]                             # the 'goal: holding X' pseudo-step is not a step
    step_results = []
    for st in psteps:
        s_ok = st.get("status") == "done"                    # tolerate 'running'/'pending' after a done-timeout
        step_results.append({"ok": s_ok, "label": st.get("label"),
                             "error": None if s_ok else (st.get("error") or f"step {st.get('status')}")})
    if replay_entry:                                         # replay arm: the stored procedure IS the grounding
        retrieved = [{"id": replay_entry.get("id"), "kind": "procedures"}]
    else:                                                    # the actual hybrid-retrieved entries injected (D13)
        retrieved = list(memory_hits or [])
    import improve
    # The observability context is what turns the write-back into an auditable exhibit: `run_id` binds the
    # entry to the command that produced it, `decision_id` to the bandit arm that chose the strategy, and
    # `episode_ids` to the executed steps it was distilled from (the prov:wasDerivedFrom edges).
    report = improve.close_loop(_get_kb(), _get_memory(), scene=scene, command=text, level=level, arm=arm,
                                executed_steps=steps, step_results=step_results, goal=goal,
                                retrieved_ids=retrieved, rag_enabled=RAG_ENABLED, loop_enabled=LOOP_ENABLED,
                                run_id=run_id, decision_id=(did if isinstance(did, int) and did >= 0 else None),
                                span_id=span_id, episode_ids=list(episode_ids or []))
    summary = _loop_summary(report)
    with _lock:
        _state["loop_last"] = summary
    for w in (report.get("writes") or []):
        _log(f"loop: {w.get('action')} {w.get('kind')} #{w.get('id')} - {w.get('why')}")
    if not report.get("writes"):
        _log("loop: no write-back" + ("" if LOOP_ENABLED else " (loop off)"))
    if report.get("reindexed"):
        _log(f"loop: embedded {report['reindexed']} new KB entr{'y' if report['reindexed'] == 1 else 'ies'}")
    return report


def _explore_target(ag, object_type):
    """Generator job: actively locate `object_type` in the scene, then fold the resulting oracle spatial snapshot
    into memory so the belief exists BEFORE retrieval/planning.

    `scan_for` is a bounded generator that turns to face the type (the thesis's active-perception gesture); once
    it settles we push `ag.spatial_snapshot()` through EnvFeedback.observe_scene (change-gated, so it costs at most
    one row per object). Fully exception-safe — any failure just logs and the plan proceeds ungrounded for this type.
    """
    try:
        yield from ag.scan_for(object_type)
    except Exception as exc:
        _log(f"explore {object_type}: {exc}")
    try:
        with _lock:
            scene = _state.get("scene") or None
        _get_feedback().observe_scene(scene, ag.spatial_snapshot(), track="oracle")  # obs are metadata-derived
    except Exception as exc:
        _log(f"explore observe {object_type}: {exc}")


def _active_exploration(text, force=False):
    """Before planning, actively locate any needed object types we hold NO belief for (Phase-5 active perception).

    Reads the live scene's object types on the worker (non-blocking, timeout-guarded), asks EnvFeedback which needed
    types are misses, and enqueues bounded `_explore_target` jobs (capped at 2) so spatial memory is enriched BEFORE
    retrieval+planning. Loop-guarded by EnvFeedback._explored (already applied inside needs_exploration, then marked
    here) and fully exception-safe — any failure just logs and skips exploration so planning always proceeds.
    """
    if not ENV_FEEDBACK_ENABLED:
        return
    try:
        with _lock:
            scene = _state.get("scene") or None
        holder = {"types": None}                            # read live object types on the worker thread that owns ag
        got = threading.Event()

        def _read(ag):
            try:
                holder["types"] = sorted({o["objectType"] for o in ag.objects()})
            except Exception:
                holder["types"] = []
            got.set()
        _jobs.put(_read)
        if not got.wait(timeout=10):                        # never block the command forever if the worker is busy
            return
        known = holder["types"] or []
        if not known:
            return
        fb = _get_feedback()
        needed = fb.parse_needed_types(text, known)
        if not needed:
            return
        misses = (needed if force else (fb.needs_exploration(scene, needed) or []))[:2]
        if not misses:
            return
        _log(f"active exploration{' (forced)' if force else ''}: locating {misses}")
        done = threading.Event()
        for t in misses:
            _jobs.put(lambda ag, tt=t: _explore_target(ag, tt))
        _jobs.put(lambda ag: done.set())
        done.wait(timeout=120)                              # bounded — scan_for is finite; enrichment then completes
        for t in misses:
            fb.mark_explored(scene, t)
    except Exception as exc:
        _log(f"active exploration skipped: {exc}")


def _lang_of(text) -> str:
    """'fa' when the command contains Arabic-script (Persian) letters, else 'en'. The dataset is bilingual by
    contract, so every run must declare which half it belongs to."""
    return "fa" if any("؀" <= ch <= "ۿ" for ch in str(text or "")) else "en"


def _plan_and_run(text, model, endpoint, attempt=0):
    with _lock:
        _state["busy"] = True
        scene0 = _state.get("scene") or None
    # ONE run per command: everything below (retrieval, the LLM call, each plan step, every AI2-THOR primitive,
    # every knowledge/memory write, loop closure) hangs off this id, which is what makes "which command
    # produced this knowledge-base entry?" a single indexed query instead of a guess.
    rid = _th("start_run", kind="command", command=text, scene=scene0, lang=_lang_of(text),
              agent_id=model, config={"model": model, "endpoint": endpoint, "attempt": attempt,
                                      "rag_enabled": RAG_ENABLED, "loop_enabled": LOOP_ENABLED,
                                      "bandit_enabled": BANDIT_ENABLED,
                                      "env_feedback_enabled": ENV_FEEDBACK_ENABLED,
                                      "perception": _PERCEPTION.get("mode")})
    run_status, run_error = "OK", None
    try:
        _th("emit", "command.received", summary=text[:200], scene=scene0, actor="human:arash",
            payload={"command": text, "model": model, "endpoint": endpoint, "attempt": attempt,
                     "lang": _lang_of(text)})
        _log(f"you: {text}")
        with _lock:
            scene = _state.get("scene") or None

        with _tspan("select_strategy", kind="RETRIEVER", op="retrieval", scene=scene, actor="system:rag",
                    input={"command": text, "rag_enabled": RAG_ENABLED}):
            memory_probe = _retrieve_memory(text)            # context feature only; final injection depends on the arm
            proc_match, proc_entry = _match_procedure(text)  # replay_procedure eligibility (D10); entry kept local
            decision = _decide_plan_strategy(text, scene, memory_probe, procedure_match=proc_match,
                                             procedure_entry=(proc_entry or {}).get("id"))
        arm = decision.get("arm") or "reuse_memory"
        did = decision.get("decision_id", -1)
        _th("annotate_run", rid, arm=arm, decision_id=(did if isinstance(did, int) and did >= 0 else None),
            scene=scene)
        _log(f"bandit: arm={arm} decision={did}")

        memory_block = ""
        memory_hits = []                                     # KB entries actually injected (loop-report provenance)
        replay = bool(arm == "replay_procedure" and proc_entry and (proc_entry.get("plan") or []))
        if replay:
            # D10: a stored procedure closely matches the command -- execute it DIRECTLY, skip the LLM call.
            raw_steps = [dict(s) for s in (proc_entry.get("plan") or []) if isinstance(s, dict)][:25]
            say = f"replaying proven procedure #{proc_entry.get('id')}: '{proc_entry.get('command')}'"
            plan_level = None
            _log(f"bandit: replaying procedure #{proc_entry.get('id')} (match {proc_match:.2f}) - LLM skipped")
        else:
            if arm == "replay_procedure":                    # matched entry vanished between match and dispatch
                _log("replay arm: stored procedure unavailable - planning normally")
            perceived = (_PERCEPTION["mode"] == "perceived")
            if arm == "plan_fresh":
                _log("bandit: planning fresh (memory/exploration skipped)")
            else:
                if perceived:
                    # The YOLO sweep further down IS this mode's search. _active_exploration turns toward
                    # objects it looked up in AI2-THOR metadata -- exactly the oracle this mode withholds.
                    _log("active search: turning to look (detector only — no scene metadata)")
                else:
                    _active_exploration(text, force=(arm == "explore_then_plan"))
                with _tspan("retrieval knowledge_base", kind="RETRIEVER", op="retrieval", scene=scene,
                            actor="system:rag", input={"query": text}) as _rsid:
                    memory_block = _retrieve_memory(text)
                    memory_hits = _retrieve_memory_hits(text)  # provenance for the loop report (D13)
                    # prov:used — ONLY the entries that actually reached the prompt. Recording every KB read
                    # would make "used" meaningless as a grounding signal.
                    for hit in (memory_hits or []):
                        try:
                            _th("record_use", store="kb",
                                table_or_kind=str(hit.get("kind") or "procedures").rstrip("s"),
                                pointer=hit.get("id"), why=f"injected into the planner prompt for {text!r}",
                                store_path=getattr(_get_kb(), "path", None))
                        except Exception:
                            pass
                    _th("emit", "retrieval.performed",
                        summary=f"{len(memory_hits or [])} hit(s), {len(memory_block or '')} chars injected",
                        scene=scene, actor="system:rag", span_id=_rsid,
                        payload={"query": text, "hits": list(memory_hits or []),
                                 "memory_block": memory_block, "rag_enabled": RAG_ENABLED})
            if perceived:
                pobs = _perceived_observation()
                _PERCEPTION["degraded"] = pobs.get("error") if pobs.get("degraded") else None
                if pobs.get("degraded"):
                    obs_text = ((_OBS.get("oracle") or "")
                                + "\n[perceived-mode degraded: " + str(pobs.get("error")) + "]")
                    _log(f"perception DEGRADED to oracle: {pobs.get('error')}")
                else:
                    obs_text = pobs["text"]
                    _OBS["text"] = obs_text
                    _log(f"perceived: {pobs['n']} object type(s) seen by YOLO — scene metadata withheld")
            else:
                obs_text = _OBS.get("text") or ""
            _th("emit", "perception.observation",
                summary=f"{_PERCEPTION['mode']} observation, {len(obs_text)} chars"
                        + (" (DEGRADED)" if _PERCEPTION.get("degraded") else ""),
                level=("WARNING" if _PERCEPTION.get("degraded") else "DEBUG"), scene=scene,
                actor="system:perception",
                payload={"mode": _PERCEPTION["mode"], "degraded": _PERCEPTION.get("degraded"),
                         "observation": obs_text})
            user = f"OBSERVATION:\n{obs_text}\n\nCOMMAND: {text}"
            if memory_block:
                user = f"RELEVANT MEMORY:\n{memory_block}\n\n" + user
                _log(f"+memory grounding ({len(memory_block)} chars)")
            _t0 = time.perf_counter()
            _llm_error = None
            try:
                content = al.make_provider(endpoint=endpoint).chat(model, [
                    {"role": "system", "content": al.PLANNER_SYSTEM},
                    {"role": "user", "content": user},
                ])
            except Exception as exc:
                _llm_error = repr(exc)
                raise
            finally:
                # The prompt (memory block included) and the raw completion are the evidence for every claim
                # about whether retrieval changed the plan — they exist nowhere else once the reply is parsed.
                _th("record_llm", model=model, endpoint=endpoint, prompt=user,
                    completion=(None if _llm_error else content), system=al.PLANNER_SYSTEM,
                    memory_block=(memory_block or None), error=_llm_error,
                    duration_ms=(time.perf_counter() - _t0) * 1000.0)
            plan = al.parse_plan(content)
            raw_steps = (plan.get("plan") or [])[:25]
            say = plan.get("say") or "(no description)"
            plan_level = plan.get("level") or None
        steps, repairs = _repair_plan_for_command(text, raw_steps)
        steps = steps[:25]
        for r in repairs:
            _log(f"plan repair: {r}")
        _th("annotate_run", rid, level=plan_level)
        _th("emit", "plan.produced", summary=f"{len(steps)} step(s): {say}"[:200], scene=scene,
            actor=("system:webapp" if replay else f"llm:{model}"),
            payload={"say": say, "level": plan_level, "arm": arm, "replay": bool(replay),
                     "raw_steps": raw_steps, "steps": steps})
        if repairs:
            # A repaired plan is NOT the plan the model produced; without this the procedure the loop stores
            # looks like the LLM's own output.
            _th("emit", "plan.repaired", summary="; ".join(str(r) for r in repairs)[:200], level="WARNING",
                scene=scene, actor="system:webapp",
                payload={"repairs": list(repairs), "before": raw_steps, "after": steps})
        _log(f"agent: {say}  [{len(steps)} step(s)]")
        with _lock:
            _state["cur_cmd"] = text                         # threaded into _tracked_step's passive feedback record
            _state["cur_level"] = plan_level
            _state["plan"] = {"say": say, "active": -1, "decision_id": did, "arm": arm,
                              "scores": decision.get("scores") or {}, "command": text, "level": plan_level,
                              "raw_steps": steps, "repaired": bool(repairs),
                              "steps": [{"label": _step_label(s), "status": "pending"} for s in steps]}
        done = threading.Event()
        ctx = _tsnap()                                       # this run, handed to the worker thread explicitly
        episode_ids = []                                     # filled in by the steps; feeds the lineage edges
        for i, step in enumerate(steps):
            _jobs.put(lambda ag, i=i, s=step: _tracked_step(ag, i, s, ctx=ctx, episodes=episode_ids))
        _jobs.put(lambda ag: done.set())
        completed = done.wait(timeout=180)                   # False => steps still on the worker (loop skips)
        with _lock:
            level = _state.get("cur_level")
            if _state["plan"]:
                _state["plan"]["active"] = -1
        goal = _live_goal_check(text)
        reward_override = None
        if goal.get("applicable"):
            reward_override = 1.0 if goal.get("ok") else 0.0
            _record_goal_check(scene, text, level, goal)
            with _lock:
                if _state["plan"] and _state["plan"].get("decision_id") == did:
                    status = "done" if goal.get("ok") else "failed"
                    _state["plan"]["goal"] = goal
                    lbl = (f"goal: found {goal.get('target')}" if goal.get("kind") == "search"
                           else f"goal: holding {goal.get('target')}")
                    _state["plan"]["steps"].append({"label": lbl, "status": status})
            if goal.get("kind") == "search":
                ls = goal.get("search") or {}
                n, planned = ls.get("stops_visited", 0), ls.get("stops_planned", 0)
                if goal.get("ok"):
                    where = ls.get("receptacle")
                    _log(f"goal check: found {goal.get('target')}"
                         + (f" at {where}" if where else "")
                         + f" after {n} of {planned} stops ({ls.get('elapsed', 0):.0f}s)")
                else:
                    _log(f"goal check FAILED: {goal.get('target')} was never seen "
                         f"({n} of {planned} stops searched)")
                _record_search_outcome(scene, text, goal)
            elif goal.get("ok"):
                _log(f"goal check: holding {goal.get('target')}")
            else:
                _log(f"goal check failed: expected {goal.get('target')}, held {goal.get('held')}")
        env_reward = _reward_plan(did, scene, text, override=reward_override,
                                  reason=("goal" if reward_override is not None else "environment"))
        with _lock:
            if _state["plan"] and _state["plan"].get("decision_id") == did:
                _state["plan"]["environment_reward"] = env_reward
        # ---- Phase-7 loop closure: write what this run taught us back into the KB. Wrapped in its OWN
        # try/except (like the Phase-5 hooks) so a loop failure can NEVER mark the plan failed or leak out.
        loop_report = None
        try:
            loop_report = _close_loop_after_run(scene, text, level, arm, steps, goal,
                                                (proc_entry if replay else None), memory_hits,
                                                did, completed, run_id=rid, episode_ids=episode_ids)
        except Exception as exc:
            _log(f"loop closure skipped: {exc}")
        # D11: at most ONE automatic retry after a failed run's write-back (flag default OFF; retry is manual).
        if (AUTO_RETRY and LOOP_ENABLED and attempt == 0 and loop_report
                and not (loop_report.get("outcome") or {}).get("success")):
            _log("auto-retry: re-planning once with the newly learned memory")
            _th("end_run", rid, status="OK", summary={"retry": True, "attempt": attempt})
            rid = None
            _plan_and_run(text, model, endpoint, attempt=1)   # the retry opens its OWN run, child of none
            return
        run_status = "OK" if (goal.get("applicable") is not True or goal.get("ok")) else "ERROR"
        _log("plan complete.")
    except Exception as exc:
        run_status, run_error = "ERROR", repr(exc)
        _log(f"error: {exc}")
    finally:
        if rid:
            _th("end_run", rid, status=run_status, error=run_error,
                summary={"attempt": attempt, "episodes": len(locals().get("episode_ids") or [])})
        with _lock:
            _state["busy"] = False
            jid = _state.pop("cmd_job", None) if attempt == 0 else None
        if jid:
            _LEASE.release(jid)


# ---------------------------------------------------------------- Phase-4 perception (scan / train / detect)
def _scan_job(ag, params):
    """Generator job: stream scan_scene's progress strings into the transcript (worker steps it like walk_to)."""
    _log(f"scan: {ag.scene} starting (target={params['per_class_target']}, positions={params['max_positions']}, "
         f"states={params['do_states']}, destructive={params['do_destructive']}, "
         f"containers={params.get('do_containers', True)}/{params.get('max_containers', 10)})")
    # A dataset scan is the single biggest writer of perception labels and nav cells in the system; without a
    # run of its own every one of those rows is an orphan in the ledger.
    rid = _th("start_run", kind="perception", command=f"scan_scene {ag.scene}", scene=ag.scene,
              config={"channel": "scan", **{k: params.get(k) for k in sorted(params)}}, set_ambient=False)
    _th("emit", "perception.run", summary=f"scan_scene {ag.scene} starting", scene=ag.scene, run_id=rid,
        actor="system:perception", payload={"scene": ag.scene, "params": params})
    try:
        with _tbound({"run_id": rid, "scene": ag.scene}):
            # The container-pass knobs are read with .get so an older client (or a queued job from before this
            # existed) still scans, with the defaults scan_scene ships.
            for msg in ag.scan_scene(per_class_target=params["per_class_target"],
                                     max_positions=params["max_positions"],
                                     do_states=params["do_states"],
                                     do_destructive=params["do_destructive"],
                                     do_containers=params.get("do_containers", True),
                                     max_containers=params.get("max_containers", 10),
                                     container_views=params.get("container_views", 3),
                                     memory=_get_memory()):
                _log(f"scan: {msg}")
                yield msg
        global _navmap_version
        _navmap_version += 1                                # invalidate the cached mapnav overlay layer
        _log("scan: complete — train the detector to build weights/<scene>.pt")
        _th("end_run", rid, status="OK")
    except Exception as exc:
        _log(f"scan error: {exc}")
        _th("end_run", rid, status="ERROR", error=str(exc))


def _explore_job(ag):
    """Generator job: drive the whole reachable area to build a COMPLETE discovered nav map; stream progress."""
    _log(f"explore: mapping {ag.scene} navigation…")
    try:
        for msg in ag.explore_navigation(memory=_get_memory()):
            _log(f"explore: {msg}")
            yield msg
        global _navmap_version
        _navmap_version += 1                                # invalidate the cached mapnav overlay layer
        _log("explore: navigation map complete")
    except Exception as exc:
        _log(f"explore error: {exc}")


# The discovered cells are static between scans — caching the rendered cell layer keeps the mapnav stream at full fps
# (re-reading sqlite + re-projecting ~260 cells every frame drops it to ~8 fps). _navmap_version is bumped after a
# scan/explore writes new cells, forcing a one-frame rebuild; otherwise each frame only pastes the cached layer.
_navmap_version = 0
_mapnav_cache = {"key": None, "layer": None, "cam": None, "cam_scene": None}


def _draw_map_overlay(agent):
    """Overlay the discovered nav map (+ live agent marker) onto AI2-THOR's top-down orthographic frame."""
    if not agent.third_party_frames:
        return agent.frame
    frame = agent.third_party_frames[0]
    if getattr(frame, "ndim", 2) == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    try:
        import math
        import numpy as np

        import navmap
        from PIL import ImageDraw
        # map_view_camera_properties() is a controller STEP — fetching it per frame halts the stream. The map camera is
        # fixed per scene, so fetch once and cache it (invalidating the layer when the scene changes).
        if _mapnav_cache["cam_scene"] != agent.scene:
            _mapnav_cache["cam"] = agent.map_view_camera_properties()
            _mapnav_cache["cam_scene"] = agent.scene
            _mapnav_cache["key"] = None
        cam = _mapnav_cache["cam"]
        if not cam:
            return frame
        h, w = frame.shape[0], frame.shape[1]
        key = (agent.scene, _navmap_version, w, h)
        if _mapnav_cache["key"] != key:                   # snapshot the top-down + bake in the static cells ONCE
            nm = navmap.NavMap(_get_memory(), agent.scene)
            bg = Image.fromarray(frame).convert("RGBA")
            d = ImageDraw.Draw(bg)
            k = h / (2.0 * (float(cam.get("orthographicSize") or 1.0) or 1.0))
            half = max(2.0, nm.cell_size * k / 2.0)
            col = {"free": (86, 217, 138, 95), "blocked": (255, 111, 111, 110), "unknown": (90, 102, 117, 70)}
            for c in nm.cells():
                px, py = navmap.map_project(c["x"], c["z"], cam, w, h)
                d.rectangle([px - half, py - half, px + half, py + half], fill=col.get(c["state"], col["unknown"]))
            _mapnav_cache.update(key=key, layer=bg.convert("RGB"))
        img = _mapnav_cache["layer"].copy()              # cheap: reuse the baked background, just stamp the live marker
        d = ImageDraw.Draw(img)
        p = agent.agent_pose()
        apx, apy = navmap.map_project(p["position"]["x"], p["position"]["z"], cam, w, h)
        yaw = p["rotation"]["y"] * math.pi / 180.0        # +z up, clockwise -> heading line on the map
        d.line([apx, apy, apx + 16 * math.sin(yaw), apy - 16 * math.cos(yaw)], fill=(255, 180, 84), width=3)
        d.ellipse([apx - 5, apy - 5, apx + 5, apy + 5], fill=(255, 180, 84))
        return np.asarray(img)
    except Exception:
        return frame


def _archive_weights(path):
    """Copy an existing detector aside before a retrain replaces it. Returns the archive path, or None.

    Train writes weights/<scene>.pt, and for FloorPlan1 that file is the detector whose per-class AP the thesis
    cites. A retrain with a different base or recipe must ADD a model, never erase the one a published number
    came from -- otherwise "is the new detector better?" stops being a question anyone can answer.
    """
    if not os.path.isfile(path):
        return None
    import shutil
    d = os.path.join(os.path.dirname(path), "archive")
    os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(d, f"{stem}.{time.strftime('%Y%m%d_%H%M%S')}.pt")
    shutil.copy2(path, dst)
    return dst


def _train_scene_bg(scene, epochs, device):
    """Train the current scene's YOLO detector in a daemon thread (training BLOCKS — never run it in the worker loop)."""
    import gen_dataset
    import yolo_recipe
    # epochs<=0 means "whatever the recipe says" so the schedule is defined in exactly one place (yolo_recipe), not
    # pinned in this handler; likewise the base checkpoint, which is reported back so the log says what was trained.
    epochs = int(epochs) if epochs and int(epochs) > 0 else yolo_recipe.DEFAULT_EPOCHS
    dataset_dir = os.path.join(HERE, "datasets", scene)
    weights_out = os.path.join(HERE, "weights", f"{scene}.pt")
    archived = _archive_weights(weights_out)
    if archived:
        _log(f"train: kept the previous detector as {os.path.relpath(archived, HERE)}")
    ts = time.strftime("%H:%M:%S")
    with _lock:
        _state["train"] = {"status": "running", "scene": scene, "weights_path": None, "metrics": {}, "error": None,
                           "ts": ts}
    _log(f"train: {scene} started ({epochs} epochs, base {yolo_recipe.resolve_base()})")
    try:
        res = gen_dataset.train_scene(scene, dataset_dir, weights_out, epochs=epochs, device=device)
        metrics = res.get("metrics") or {}
        # train_scene returns metrics={'per_class_ap': {...}, 'summary': {...}, ['error']}; the scalars live in the
        # nested 'summary' map, so reach in (a top-level isinstance(float) filter would always yield {}).
        summary = {k: round(float(v), 4) for k, v in (metrics.get("summary") or {}).items()}
        rnd = lambda d: {k: (None if v is None else round(float(v), 4)) for k, v in (d or {}).items()}
        per_class = rnd(metrics.get("per_class_ap"))
        per_recall = rnd(metrics.get("per_class_recall"))
        per_ap50 = rnd(metrics.get("per_class_ap50"))
        # The three lowest-recall classes, for the one-line status. A single mAP says the detector is weak;
        # this says what to go and photograph next.
        worst = sorted(((k, v) for k, v in per_recall.items() if v is not None), key=lambda kv: kv[1])[:3]
        with _lock:
            _state["train"] = {"status": "done", "scene": scene, "weights_path": res.get("weights_path"),
                               "metrics": summary, "per_class_ap": per_class, "per_class_recall": per_recall,
                               "per_class_ap50": per_ap50, "worst_recall": worst, "base": res.get("base"),
                               "n_images": res.get("n_images"), "archived": archived,
                               "error": metrics.get("error"), "ts": time.strftime("%H:%M:%S")}
        _log(f"train: {scene} done — weights {res.get('weights_path')} (base {res.get('base')}, "
             f"{res.get('n_images')} images)")
    except Exception as exc:
        with _lock:
            _state["train"] = {"status": "error", "scene": scene, "weights_path": None, "metrics": {},
                               "error": str(exc), "ts": time.strftime("%H:%M:%S")}
        _log(f"train error: {exc}")


# ---------------------------------------------------------------- Phase 8/9 evaluation panel
# Browse the bilingual dataset, launch an eval run (SR/PC/TDS), and view results in-console. The run spawns
# sim/eval.py as a subprocess (it boots its OWN ThorAgent), so it SHARES the GPU with the live console -- fine
# for small models + short runs; stop the console first for big-model sweeps.
_EVAL = {"running": False, "config": None, "started": None, "done": 0, "total": 0,
         "out": None, "results": None, "error": None}
DATASET_PATH = os.path.join(HERE, "dataset", "commands.jsonl")


def _load_dataset_rows():
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _dataset_summary(scene="", level="", lang="", limit="200"):
    """Dataset browser: overall counts + a filtered slice of records (id/level/lang/scene/command/gold)."""
    try:
        from collections import Counter
        rows = _load_dataset_rows()
        counts = {"total": len(rows),
                  "by_level": dict(Counter(r.get("level") for r in rows)),
                  "by_scene": dict(Counter(r.get("scene") for r in rows)),
                  "by_lang": dict(Counter(r.get("lang") for r in rows))}
        try:
            lim = max(1, int(limit))
        except Exception:
            lim = 200
        sel = [r for r in rows
               if (not scene or r.get("scene") == scene)
               and (not level or r.get("level") == level)
               and (not lang or r.get("lang") == lang)][:lim]
        return {"counts": counts, "shown": len(sel), "items": sel}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------- the GENERAL instruction set (editable)
# sim/general_set.py owns the file, the validation and the audit log; these are thin HTTP shapes over it.
GENERAL_PATH = os.path.join(HERE, "dataset", "general.json")


def _general_list(level="", query="", show=""):
    """Every general instruction, both languages side by side, with the editor's text form and -- from the
    30-kitchen inventory scan -- in how many kitchens it can be scored.

    `show` narrows the list to what a reviewer is hunting for: `unchecked` / `checked` (the per-item review
    mark), `partial` (does not apply in every kitchen), `setup` (presupposes a state the kitchen is first put in).
    Counts are always over the WHOLE set, so the level chips and the progress bar do not move with the filter."""
    try:
        from collections import Counter
        import general_set as gs
        doc = gs.load()
        inv = gs.load_inventory()
        items = doc["items"]
        counts = {"total": len(items), "by_level": dict(Counter(it.get("level") for it in items)),
                  "checked": sum(1 for it in items if it.get("reviewed")),
                  "checked_by_level": dict(Counter(it.get("level") for it in items if it.get("reviewed"))),
                  "kitchens": len((inv or {}).get("scenes") or [])}
        ql = (query or "").strip().lower()
        views = [gs.view(it, inv) for it in items
                 if (not level or it.get("level") == level)
                 and (not ql or ql in it.get("en", "").lower() or ql in it.get("fa", "") or ql in it.get("id", ""))]
        keep = {"unchecked": lambda v: not v.get("reviewed"), "checked": lambda v: v.get("reviewed"),
                "partial": lambda v: (v.get("applies") or {}).get("n") != (v.get("applies") or {}).get("total"),
                "setup": lambda v: bool(v.get("setup"))}.get(show)
        if keep:
            views = [v for v in views if keep(v)]
        return {"counts": counts, "shown": len(views), "updated_ts": doc.get("updated_ts"), "items": views}
    except Exception as exc:
        return {"error": str(exc)}


def _general_review(payload):
    import general_set as gs
    return gs.set_reviewed(str((payload or {}).get("id") or ""), bool((payload or {}).get("reviewed")),
                           actor="user:console")


def _general_save(payload):
    """Add (no `original_id`) or replace an instruction. The editor posts goals and plans as one-line text."""
    import general_set as gs
    try:
        item = gs.from_editor(payload or {})
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": []}
    res = gs.upsert(item, original_id=(payload or {}).get("original_id") or None, actor="user:console")
    if res.get("ok"):
        _log(f"general set: {res['op']} {item['id']} [{item['level']}]")
        # The dataset is the instrument every score is read against, so an edit to it is provenance, not
        # housekeeping: it goes on the same timeline as the KB and memory writes.
        _th("record_write", store="file", table_or_kind="general.json", pointer=item["id"],
            verdict=("insert" if res["op"] == "add" else "update"),
            why="general instruction edited in the console", source="human", actor="user:console",
            summary=f"{item['level']} {item['en']}"[:200], content=res.get("item"))
        res["item"] = gs.view(res["item"], gs.load_inventory())
    return res


def _general_delete(payload):
    import general_set as gs
    iid = str((payload or {}).get("id") or "")
    res = gs.delete(iid, actor="user:console")
    if res.get("ok"):
        _log(f"general set: deleted {iid}")
        _th("record_write", store="file", table_or_kind="general.json", pointer=iid, verdict="delete",
            why="general instruction deleted in the console", source="human", actor="user:console",
            summary=iid)
    return res


def _eval_files():
    """Recent eval result JSONs (newest first) for the results picker."""
    out = []
    try:
        for n in os.listdir(HERE):
            if n.startswith("eval") and n.endswith(".json") and ".partial" not in n:
                p = os.path.join(HERE, n)
                out.append({"file": n, "mtime": os.path.getmtime(p), "size": os.path.getsize(p)})
        out.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        pass
    return out


def _read_eval_results(fname=""):
    """Config + aggregates for a chosen (or the latest) eval result JSON."""
    files = _eval_files()
    try:
        if not fname and files:
            fname = files[0]["file"]
        res = {"files": files, "selected": fname}
        if fname:
            safe = os.path.basename(fname)
            with open(os.path.join(HERE, safe), encoding="utf-8") as fh:
                doc = json.load(fh)
            res["config"] = doc.get("config")
            res["aggregates"] = doc.get("aggregates")
            res["n_records"] = doc.get("n_records")
            res["generated_ts"] = doc.get("generated_ts")
        return res
    except Exception as exc:
        return {"error": str(exc), "files": files}


def _eval_snapshot():
    with _lock:
        e = {k: _EVAL[k] for k in ("running", "config", "started", "done", "total", "out", "results", "error")}
    return e


def _run_eval_bg(cfg, lease_job=None):
    """Daemon thread: run sim/eval.py as a subprocess and stream progress from its .partial.jsonl into _EVAL."""
    try:
        _run_eval_body(cfg)
    finally:
        if lease_job:
            _LEASE.release(lease_job)


def _run_eval_body(cfg):
    import subprocess
    import sys
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"eval_webapp_{ts}.json")
    partial = os.path.splitext(out)[0] + ".partial.jsonl"
    total = 0
    general = cfg.get("dataset") == "general"
    try:
        scs = [s.strip() for s in (cfg.get("scenes") or "").split(",") if s.strip()]
        if general:
            # Scene-less records run once in EVERY requested scene (eval.py defaults to FloorPlan1 when none is).
            import general_set as gs
            rows = gs.to_records(gs.load(), lang=cfg["lang"])
            if cfg.get("limit"):
                rows = rows[:int(cfg["limit"])]
            per_model = len(rows) * max(1, len(scs))
        else:
            rows = _load_dataset_rows()
            if cfg["lang"] != "all":
                rows = [r for r in rows if r.get("lang") == cfg["lang"]]
            if scs:
                rows = [r for r in rows if r.get("scene") in set(scs)]
            if cfg.get("limit"):
                rows = rows[:int(cfg["limit"])]
            per_model = len(rows)
        n_models = len([m for m in cfg["models"].split(",") if m.strip()])
        total = per_model * max(1, n_models)
    except Exception:
        total = 0
    with _lock:
        _EVAL.update({"running": True, "config": cfg, "started": ts, "done": 0, "total": total,
                      "out": os.path.basename(out), "results": None, "error": None})
    _log(f"eval: started ({cfg['models']} rag={cfg['rag']} loop={cfg['loop']} lang={cfg['lang']} "
         f"dataset={'general' if general else 'scene-grounded'}) -- shares the GPU with the live view")
    cmd = [sys.executable, os.path.join(HERE, "eval.py"), "--models", cfg["models"], "--rag", cfg["rag"],
           "--loop", cfg["loop"], "--lang", cfg["lang"], "--out", out]
    if general:
        cmd += ["--dataset", GENERAL_PATH]
    if cfg.get("scenes"):
        cmd += ["--scenes", cfg["scenes"]]
    if cfg.get("limit"):
        cmd += ["--limit", str(cfg["limit"])]
    # The eval harness is a SEPARATE process, and sim/eval.py is owned by another workstream — so the parent
    # hands the run down through the environment instead of an import: TRACE_RUN_ID makes every record the
    # subprocess writes a child of this run, TRACE_DB points both processes at the same store (whose seq
    # allocator is cross-process safe by design, see trace._reserve_seq_block).
    rid = _th("start_run", kind="eval", command=f"eval {cfg['models']}", config=dict(cfg), set_ambient=False)
    env = os.environ.copy()
    if _trace is not None and rid:
        env["TRACE_RUN_ID"] = str(rid)
        env["TRACE_ORIGIN"] = "eval"
        try:
            env["TRACE_DB"] = (getattr(_trace, "_CFG", {}) or {}).get("db_path") or env.get("TRACE_DB", "")
        except Exception:
            pass
        env = {k: v for k, v in env.items() if v is not None and v != ""}
    _th("emit", "run.started", summary=f"eval subprocess: {cfg['models']}"[:200], run_id=rid,
        actor="system:eval", payload={"config": cfg, "out": os.path.basename(out)})
    try:
        proc = subprocess.Popen(cmd, cwd=HERE, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while proc.poll() is None:
            try:
                n = sum(1 for _ in open(partial, encoding="utf-8")) if os.path.exists(partial) else 0
                with _lock:
                    _EVAL["done"] = n
            except Exception:
                pass
            time.sleep(1.5)
        results = None
        if os.path.exists(out):
            try:
                doc = json.load(open(out, encoding="utf-8"))
                results = {"aggregates": doc.get("aggregates"), "n_records": doc.get("n_records"),
                           "config": doc.get("config")}
            except Exception:
                pass
        with _lock:
            _EVAL.update({"running": False, "results": results,
                          "error": None if proc.returncode == 0 else f"eval.py exited {proc.returncode}"})
        _log(f"eval: done ({(results or {}).get('n_records')} records) -> {os.path.basename(out)}")
        _th("record_write", store="file", table_or_kind=os.path.basename(out), pointer=os.path.basename(out),
            verdict="insert", why="evaluation run results", source="environment", actor="system:eval",
            summary=f"{(results or {}).get('n_records')} records", content=results, run_id=rid)
        _th("end_run", rid, status=("OK" if proc.returncode == 0 else "ERROR"), summary=results)
    except Exception as exc:
        with _lock:
            _EVAL.update({"running": False, "error": str(exc)})
        _log(f"eval error: {exc}")
        _th("end_run", rid, status="ERROR", error=str(exc))


def _detector_for(scene):
    """Lazily build (and cache) this scene's detector; rebuild when a weights file changes.

    Step-3 models first (weights/v3/<scene>.yolo26s.pt primary + <scene>.amb26m.pt inspection specialist ->
    perception.TwoStageDetector, gated on "the robot is holding something"); otherwise the legacy single
    weights/<scene>.pt. The path comes from ``perception`` helpers rather than an f-string, because AI2-THOR names the
    live scene ``FloorPlan1_physics`` while the scan/train side writes ``FloorPlan1.pt`` (or the reverse)."""
    try:
        import perception as _perc
        v3 = _perc.v3_paths(scene, os.path.join(HERE, "weights"))
        legacy = _perc.weights_path(scene, os.path.join(HERE, "weights"))
    except Exception:
        v3, legacy = {"primary": "", "specialist": ""}, os.path.join(HERE, "weights", f"{scene}.pt")
    files = [v3["primary"], v3["specialist"]] if v3["primary"] else [legacy]
    files = [f for f in files if f and os.path.isfile(f)]
    if not files:
        return None
    mt = tuple(os.path.getmtime(f) for f in files)
    if _detector["scene"] == scene and _detector["mtime"] == mt:
        det = _detector["obj"]
        if det is not None and hasattr(det, "mode"):
            det.mode = _INSPECT["mode"]
        return det                                     # cached (Detector, or None if a prior load failed)
    _detector.update(scene=scene, mtime=mt, obj=None)
    try:
        import perception
        if v3["primary"]:
            spec = perception.Detector(v3["specialist"]) if v3["specialist"] else None
            states = None
            try:
                if perception.StateEstimator.available(scene, os.path.join(HERE, "weights")):
                    states = perception.StateEstimator(scene, os.path.join(HERE, "weights"))
            except Exception as exc:
                _log(f"state estimator not loaded: {exc}")
            _detector["obj"] = perception.TwoStageDetector(
                perception.Detector(v3["primary"]), spec, mode=_INSPECT["mode"], states=states,
                gate=lambda: (_state.get("held") or "none") != "none")      # proprioception: holding something
            if states is not None:
                _log(f"state estimator loaded: {', '.join(sorted(states.models))}")
            _log(f"detectors loaded for {scene}: primary {os.path.basename(v3['primary'])}"
                 + (f" + inspection {os.path.basename(v3['specialist'])}" if spec else " (no inspection model yet)"))
        else:
            _detector["obj"] = perception.Detector(legacy)
            _log(f"detector loaded for {scene}")
    except Exception as exc:
        _log(f"detector load failed: {exc}")
    return _detector["obj"]


def _detectors_info(scene):
    """What the console shows about the perception stack (cheap: file stats only, no model load)."""
    try:
        import perception as _perc
        v3 = _perc.v3_paths(scene, os.path.join(HERE, "weights"))
    except Exception:
        v3 = {"primary": "", "specialist": ""}
    det = _detector.get("obj") if _detector.get("scene") == scene else None
    return {"primary": os.path.basename(v3["primary"]) if v3["primary"] else None,
            "specialist": os.path.basename(v3["specialist"]) if v3["specialist"] else None,
            "inspect_mode": _INSPECT["mode"],
            "states": sorted(getattr(getattr(det, "states", None), "models", {}) or []),
            "inspecting": bool(getattr(det, "last_inspecting", False)),
            "holding": (_state.get("held") or "none") != "none"}


#: the 'detect' view's models run OFF the render loop (2026-10-05: running the detector, the specialist and up to a
#: dozen state classifiers inside the loop dragged the whole live view down to ~10 fps). A background thread takes
#: the newest frame, runs the stack, and publishes boxes; the loop keeps rendering at full speed and draws the latest
#: boxes on every frame (they are at most one inference old).
_DETBG = {"frame": None, "held": False, "scene": None, "dets": [], "seq": 0, "busy": False, "err": None}
_DETBG_LOCK = threading.Lock()
_DETECT_LOCK = threading.Lock()                       # one model call at a time (the plan sweep uses the same stack)


def _detect_bg():
    while True:
        with _DETBG_LOCK:
            job = (_DETBG["frame"], _DETBG["held"], _DETBG["scene"])
            _DETBG["frame"] = None
        frame, held_now, scene = job
        if frame is None:
            time.sleep(0.02)
            continue
        det = _detector_for(scene)
        if det is None:
            continue
        try:
            with _DETECT_LOCK:
                dets = det.detect(frame, inspecting=held_now) if hasattr(det, "specialist") else det.detect(frame)
            with _DETBG_LOCK:
                _DETBG.update(dets=dets, seq=_DETBG["seq"] + 1, err=None)
            if _DETBG["seq"] % 20 == 1:                # trace a sample, not every inference (the store is on /mnt/d)
                _th("emit", "perception.run", summary=f"YOLO {scene}: {len(dets)} detection(s)", level="DEBUG",
                    scene=scene, actor="system:perception",
                    payload={"scene": scene, "mode": _PERCEPTION.get("mode"),
                             "detections": [{"cls": o.get("cls"), "conf": o.get("conf")} for o in dets][:40]})
        except Exception as exc:
            with _DETBG_LOCK:
                first = _DETBG["err"] is None
                _DETBG["err"] = str(exc)
            if first:
                _log(f"detect failed: {exc}")


threading.Thread(target=_detect_bg, daemon=True).start()


def _draw_detections(agent):
    """Return agent.frame with the LATEST detections drawn (PIL). Hands the newest frame to the background detector."""
    frame = agent.frame
    if getattr(frame, "ndim", 2) == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]                          # strip alpha once, up front — feeds both detect() and drawing
    held_now = False
    try:
        held_now = bool(agent.held_object())
    except Exception:
        pass
    with _DETBG_LOCK:
        if _DETBG["scene"] != agent.scene:
            _DETBG["dets"] = []
        _DETBG.update(frame=frame.copy(), held=held_now, scene=agent.scene)
        dets = list(_DETBG["dets"])
    if not dets:
        return frame
    from PIL import ImageDraw
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    for o in dets:
        x1, y1, x2, y2 = o["bbox_xyxy"]
        # inspection-specialist boxes in amber, primary in teal — so the operator SEES the second model take over
        col = (255, 176, 46) if o.get("source") == "specialist" else (51, 224, 196)
        d.rectangle([x1, y1, x2, y2], outline=col, width=3 if o.get("source") == "specialist" else 2)
        label = (("* " if o.get("source") == "specialist" else "") + f"{o['cls']} {o['conf']:.2f}"
                 + (f" · {o['state']}" if o.get("state") else ""))
        tw = 6 * len(label) + 6
        d.rectangle([x1, max(0, y1 - 13), x1 + tw, y1], fill=col)
        d.text((x1 + 3, max(0, y1 - 13)), label, fill=(4, 19, 15))
    import numpy as np
    return np.asarray(img)


_percready = {"scene": None, "ok": False, "t": 0.0}


def _perception_ready(scene):
    """True when this scene has trained YOLO weights -- i.e. perceived mode can actually run.

    Cached for 5 s because /status polls every 400 ms and this is a filesystem stat across the WSL mount.
    """
    now = time.time()
    if _percready["scene"] == scene and (now - _percready["t"]) < 5.0:
        return _percready["ok"]
    ok = False
    try:
        import perception as _perc
        ok = bool(_perc.weights_path(scene, os.path.join(HERE, "weights"))
                  or _perc.v3_paths(scene, os.path.join(HERE, "weights"))["primary"])
    except Exception:
        ok = False
    _percready.update(scene=scene, ok=ok, t=now)
    return ok


def _perceived_observation(timeout=90):
    """Build the OBSERVATION from the DETECTOR ALONE: spin in place, run YOLO at each stop, fold what was seen.

    Nothing here reads AI2-THOR metadata except the held object (proprioception, not vision). The sweep runs as
    a worker job -- only the worker thread may touch the controller -- and it is a GENERATOR, so the search is
    animated in the live view instead of freezing it for several seconds. Returns
    ``{"text", "degraded", "n", "error"}``; ``degraded`` True means the caller MUST fall back to oracle and say
    so out loud, because a silent fallback would make the perception switch a lie.
    """
    result = {"text": "", "degraded": True, "n": 0, "error": None}
    done = threading.Event()

    def job(ag):
        det = _detector_for(ag.scene)
        if det is None:
            result["error"] = (f"no trained detector for {ag.scene} — run 'scan & learn' then "
                               f"'train detector' in Scene & setup")
            done.set()
            return None

        def gen():
            seen = {}
            try:
                import perception as _perc
                # rotate at the operator's chosen motion granularity, but keep YOLO to one call per 45 deg
                yield from _perc.sweep_detections(ag, det, seen, sweep=True, turn=_mp()["turn"],
                                                  detect_every=45.0)
                held = ag.held_object()
                result["text"] = _perc.observation_from(seen, (held or {}).get("objectType") or "none")
                result["n"] = len(seen)
                result["degraded"] = False
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done.set()

        return gen()

    _jobs.put(job)
    if not done.wait(timeout=timeout):
        result["error"] = result["error"] or f"perceived sweep did not finish within {timeout}s"
    return result


def _frame_glitch(frame):
    """The render-corruption signatures of this machine (Unity over WSLg / Mesa D3D12; docs/kitchen-datasets.md):
    a whole-image red or blue colour cast, or a pure-white frame. Same rule as build_kitchen_data.frame_anomaly."""
    import numpy as np
    a = np.asarray(frame, dtype=np.float32)[..., :3]
    if float(((a > 245).all(-1)).mean()) >= 0.99:
        return "pure_white"
    m = a.reshape(-1, 3).mean(0)
    if min(m[0] - m[1], m[0] - m[2]) > 60:
        return "red_cast"
    if min(m[2] - m[0], m[2] - m[1]) > 60:
        return "blue_cast"
    return None


#: render watchdog — the simulator PROCESS degrades at a random moment and stays degraded until restarted
#: (glitch experiment 2026-09-29: restart-on-first-glitch cut glitched views from 6-44 % to 0.02 %)
_GLITCH = {"streak": 0, "restarts": 0, "last_restart": 0.0, "content_pose": None, "last": None, "hits": []}
GLITCH_STREAK, GLITCH_COOLDOWN_S = 3, 45.0
GLITCH_WINDOW_S, GLITCH_HITS = 3.0, 3            # >= 3 corrupted frames within 3 s -> the process has degraded


def _new_agent(scene="FloorPlan1"):
    # fail fast on a hung / dead simulator (ai2thor's default is 100 s); the timeout is fixed when the controller
    # starts, so it must be passed here, not set afterwards
    return ta.ThorAgent(scene=scene, width=600, height=600, render_instance_seg=False,
                        render_depth=_honest_on(), server_timeout=SIM_STEP_TIMEOUT_S)


def _restart_simulator(agent, why):
    """Replace a degraded simulator with a fresh process in the same scene, put the agent back where it stood."""
    scene = (agent.event.metadata.get("sceneName") or "FloorPlan1").replace("_physics", "")
    try:
        p = agent.agent_pose()
        pose = (p["position"]["x"], p["position"]["y"], p["position"]["z"], p["rotation"]["y"], p["horizon"],
                p.get("standing", True))
    except Exception:
        pose = None
    try:
        agent.stop()
    except Exception:
        pass
    import subprocess
    subprocess.run(["pkill", "-f", "thor-Linux64"], check=False)
    time.sleep(2)
    fresh = _new_agent(scene)
    if pose:
        try:
            fresh.teleport_full(*pose[:4], horizon=pose[4], standing=pose[5])
        except Exception:
            pass
    _apply_motion(fresh)
    if _view["mode"] not in ("rgb", "detect"):
        _view["map_cam"] = False
        try:
            _set_view(fresh, _view["mode"])
        except Exception:
            pass
    _GLITCH.update(streak=0, restarts=_GLITCH["restarts"] + 1, last_restart=time.time())
    _SIM["generation"] += 1                     # a new simulator: queued honest work tagged with the old one is stale
    _log(f"render glitch ({why}) — simulator restarted on a fresh process; scene and position restored "
         f"(restart #{_GLITCH['restarts']}). A held object and any running plan were dropped.")
    return fresh


def _glitch_check(agent):
    """Called every frame (subsampled check). Returns a fresh agent when >= GLITCH_HITS frames within GLITCH_WINDOW_S
    show a corruption signature — a degraded process glitches intermittently, so consecutive hits are not required
    (2026-10-05: a red-cast session never tripped the old 3-in-a-row rule); otherwise None. A cast that SURVIVES a restart at the same pose is
    content (a close-up of a red toaster), not corruption: that pose is then exempt until the agent moves."""
    forced = _GLITCH.pop("force", None)        # the honest frame hook saw the renderer degrade mid-step
    if forced and time.time() - _GLITCH["last_restart"] >= GLITCH_COOLDOWN_S:
        _GLITCH["hits"] = []
        return _restart_simulator(agent, forced)
    try:
        pose = agent.agent_pose()
        key = (round(pose["position"]["x"], 1), round(pose["position"]["z"], 1), round(pose["rotation"]["y"]),
               round(pose["horizon"]))
        sig = _frame_glitch(agent.frame[::6, ::6])
    except Exception:
        return None
    now = time.time()
    _GLITCH["hits"] = [t for t in _GLITCH["hits"] if now - t < GLITCH_WINDOW_S]
    if sig is not None:
        _GLITCH["last"] = sig
    if sig is None or key == _GLITCH["content_pose"]:
        if sig is None and key != _GLITCH["content_pose"]:
            _GLITCH["content_pose"] = None if not _GLITCH["hits"] else _GLITCH["content_pose"]
        return None
    _GLITCH["hits"].append(now)
    if len(_GLITCH["hits"]) < GLITCH_HITS:
        return None
    _GLITCH["hits"] = []                        # intermittent corruption: no "re-render cleared it" escape hatch
    if time.time() - _GLITCH["last_restart"] < GLITCH_COOLDOWN_S:
        _GLITCH["content_pose"] = key           # it came straight back after a restart: treat as content here
        _GLITCH["streak"] = 0
        _log(f"colour cast ({sig}) persists right after a restart at this pose — treating it as scene content")
        return None
    return _restart_simulator(agent, sig)


#: simulator supervision (2026-10-05). A backend timeout used to kill the worker thread silently: the HTTP server kept
#: serving the last frame and a stale fps, so the console looked "painfully slow" while it was actually dead.
_SIM = {"state": "starting", "generation": 0, "error": None, "beat": 0.0, "recoveries": 0, "frame_ts": 0.0}
SIM_STEP_TIMEOUT_S = 25.0                       # a controller step longer than this = a hung simulator


def _worker():
    """Supervisor: run the simulator loop; when the simulator dies or hangs (backend timeout, broken pipe, crash),
    report it, kill the Unity process, start a fresh one in the same scene and carry on. The job that was running is
    FAILED, not replayed (a timed-out action may already have happened), and queued jobs are dropped."""
    backoff = 2.0
    while True:
        scene = ((_state.get("scene") or "FloorPlan1").replace("_physics", "")) or "FloorPlan1"
        try:
            _SIM.update(state="starting", error=None)
            agent = _new_agent(scene)
            _SIM.update(state="ok", generation=_SIM["generation"] + 1, beat=time.time())
            backoff = 2.0
            _worker_loop(agent)
        except Exception as exc:
            _SIM.update(state="recovering", error=f"{type(exc).__name__}: {str(exc)[:160]}",
                        recoveries=_SIM["recoveries"] + 1)
            dropped = 0
            while True:
                try:
                    _jobs.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            _log(f"SIMULATOR DIED ({type(exc).__name__}) — restarting it in {scene}; the running action was "
                 f"stopped{f' and {dropped} queued action(s) dropped' if dropped else ''}. "
                 f"Recovery #{_SIM['recoveries']}.")
            import subprocess
            subprocess.run(["pkill", "-f", "thor-Linux64"], check=False)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


#: idle cadence (2026-10-07). The idle loop used to Pass + re-render the unchanged kitchen at full speed (40-70 fps,
#: thor-Linux64 ~107 % CPU, + depth in honest mode); on WSLg/D3D12 that sustained load degraded rendering to 2-4 fps
#: until a restart. After IDLE_AFTER_S with no job / drive / animation the loop renders at 1/IDLE_PERIOD_S instead,
#: still Passing (physics keeps settling), and wakes within ~20 ms when a job is queued or a drive key is held.
IDLE_AFTER_S = 1.5
IDLE_PERIOD_S = 0.2
_IDLE = {"on": False, "last": time.time()}


def _idle_wait(t_loop):
    """Hold the rest of one idle period; return early when work arrives."""
    end = t_loop + IDLE_PERIOD_S
    while time.monotonic() < end and _jobs.empty() and not _drive_active():
        time.sleep(0.02)


def _worker_loop(agent):
    _log(f"AI2-THOR ready — {agent.event.metadata.get('sceneName')}, {len(agent.objects())} objects")
    try:
        info = agent.gpu_status(probe_frames=8)
        with _lock:
            _state["gpu"] = (info.get("gpu_name") or "GPU") if info.get("gpu") else "CPU (no GPU!)"
    except Exception:
        pass
    _apply_motion(agent)
    n, t0, active = 0, time.time(), None
    motion_label = ""                      # what the running generator says it is doing RIGHT NOW, per frame
    while True:
        _SIM["beat"] = time.time()
        t_loop = time.monotonic()
        if active is not None or _drive_active() or not _jobs.empty():
            _IDLE["last"] = time.time()
        if active is not None:
            try:
                motion_label = str(next(active) or "")
            except StopIteration:
                active, motion_label = None, ""
            except Exception as e:
                _log(f"action error: {e}"); active, motion_label = None, ""
        elif _drive_active():
            _do_drive_step(agent)          # smooth manual drive — one small step/frame; auto-stops via watchdog
            motion_label = _drive["action"] or ""
        else:
            motion_label = ""
            try:
                r = _jobs.get_nowait()(agent)
                if hasattr(r, "__next__"):
                    active = r
            except queue.Empty:
                pass
        agent.pass_()
        if _view["mode"] in ("map", "mapnav") and agent.third_party_frames:
            frame = _draw_map_overlay(agent) if _view["mode"] == "mapnav" else agent.third_party_frames[0]
        else:
            frame = agent.frame
            m = _view["mode"]
            try:
                if m == "depth" and agent.depth_frame is not None:
                    frame = ta.depth_to_gray(agent.depth_frame)
                elif m in ("instance", "semantic", "normals"):
                    f = agent.frame_of_kind(m)
                    if f is not None:
                        frame = f
            except Exception:
                frame = agent.frame
            if m == "detect":
                try:
                    frame = _honest_overlay_frame(agent) if _honest_on() else _draw_detections(agent)
                except Exception:
                    frame = agent.frame
        if getattr(frame, "ndim", 2) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=80)
        held = agent.held_object()
        p = agent.agent_pose()
        h, w = agent.frame.shape[:2]
        jpg = buf.getvalue()
        with _frame_cv:
            # Sequence + backlog: /stream sends every seq it has not sent yet, so no rendered frame is silently
            # skipped. This is the whole difference between watching the robot move and watching it teleport.
            _publish_frame(jpg)
            _state["motion_label"] = motion_label
            _state["scene"] = agent.event.metadata.get("sceneName", "")
            _state["held"] = held["objectType"] if held else "none"
            _state["pose"] = {"x": round(p["position"]["x"], 2), "y": round(p["position"]["y"], 2),
                              "z": round(p["position"]["z"], 2), "rot": round(p["rotation"]["y"]),
                              "horizon": round(p["horizon"]), "standing": p["standing"]}
            _state["resolution"] = f"{w}x{h}"
            _state["recording"] = bool(getattr(agent, "_recording", False))
            _SIM["frame_ts"] = time.time()
            _frame_cv.notify_all()
        # The oracle description is refreshed every frame either way (it is what a DEGRADED perceived run has to
        # fall back to), but in perceived mode it is NOT allowed to become the observation the planner reads --
        # that text is written only by the detector sweep in _plan_and_run.
        _OBS["oracle"] = agent.describe(max_items=30)
        if _PERCEPTION["mode"] != "perceived":
            _OBS["text"] = _OBS["oracle"]
        if active is not None and _mp()["delay"]:
            time.sleep(_mp()["delay"])     # pace ONLY while a plan animates; idle and manual drive stay full speed
        _IDLE["on"] = active is None and time.time() - _IDLE["last"] > IDLE_AFTER_S
        if _IDLE["on"]:
            _idle_wait(t_loop)
        n += 1
        dt = time.time() - t0
        if _view["mode"] in ("rgb", "detect"):
            fresh = _glitch_check(agent)
            if fresh is not None:
                agent, active, motion_label = fresh, None, ""
        if dt >= 0.5:
            objs = [{"id": o["objectId"], "type": o["objectType"], "dist": round(o.get("distance", 0), 1),
                     "visible": bool(o.get("visible")), "flags": ta.object_state_flags(o),
                     "actions": [s.label for s in ta.applicable_actions(o)]}
                    for o in sorted(agent.objects(), key=lambda o: o.get("distance", 9e9))]
            with _lock:
                _state["fps"] = round(n / dt, 1)
                _state["objects"] = objs
                _state["ncam"] = len(getattr(agent, "third_party_frames", []) or [])
            n, t0 = 0, time.time()


# ------------------------------------------------------- observability API (read-only, over sim/trace.py)
# Thin JSON wrappers, nothing more. ALL of the logic lives in sim/trace.py (stdlib-only, importable without
# ai2thor/PIL, self-tested by sim/_trace_selftest.py); these functions only parse the query string and shape
# the envelope, so the API is testable on Windows where `import webapp` is impossible (no ai2thor).
#
# Contract for every function below (the /episodes contract, docs/observability-design.md §9):
#   * returns a plain dict carrying "error" (None on success) — NEVER raises, so do_GET can never 500;
#   * the common first-run case "sim/trace.db does not exist yet" is a clean {"error": "no trace store"};
#   * strictly READ-ONLY: memory.db is opened `file:...?mode=ro` inside trace.py, the KB JSON / JSONL stores
#     are read and filtered in Python, and no table name from the client ever reaches an SQL string (it is
#     looked up in trace.py's own catalog first);
#   * every list endpoint clamps `limit` server-side (trace.py clamps again) — a whole-log response at 10^5
#     events would be ~30 MB on one thread and would stall the console's 400 ms /status poll.
_TRACE_OFF = ("tracing unavailable — sim/trace.py did not import (or the stdlib `trace` module shadowed it); "
              "the console is unaffected")


def _tr_call(fn_name, **kw):
    """Call one sim/trace.py read function. Returns (value, error) and never raises."""
    if _trace is None:
        return None, _TRACE_OFF
    fn = getattr(_trace, fn_name, None)
    if not callable(fn):
        return None, f"trace.{fn_name}() unavailable (api v{getattr(_trace, 'TRACE_API_VERSION', 0)})"
    try:
        return fn(**kw), None
    except Exception as exc:                     # trace.py is already _safe()-wrapped; this is belt and braces
        return None, f"{type(exc).__name__}: {exc}"


def _tr_int(val, default=None, name="value"):
    """Parse an int query param. Raises ValueError on garbage so the caller turns it into an envelope."""
    s = str(val or "").strip()
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        raise ValueError(f"{name} must be an integer, got {s!r}")


def _tr_str(val):
    s = str(val or "").strip()
    return s or None


def _tr_bool(val, default=False):
    s = str(val or "").strip().lower()
    return default if not s else s not in ("0", "false", "no", "off")


def _trace_events(one):
    """GET /trace_events — the filtered, cursor-paged event feed. Rows omit payload_json (see /trace_event)."""
    out = {"events": [], "cursor": {"first_seq": None, "last_seq": None}, "has_more": False,
           "total": None, "error": None}
    try:
        kw = {"after": _tr_int(one("after"), None, "after"),
              "before": _tr_int(one("before"), None, "before"),
              "limit": _tr_int(one("limit"), 200, "limit"),
              "type": _tr_str(one("type")), "run_id": _tr_str(one("run_id")),
              "span_id": _tr_str(one("span_id")), "actor": _tr_str(one("actor")),
              "level": _tr_str(one("level")), "min_level": _tr_str(one("min_level")),
              "scene": _tr_str(one("scene")), "q": _tr_str(one("q")),
              "order": (one("order", "desc") or "desc")}
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("query_events", **kw)
    if err:
        out["error"] = err
        return out
    res.setdefault("error", None)
    return res


def _trace_event(one):
    """GET /trace_event?seq= — one event with its full payload, the writes it produced, and its run/span."""
    out = {"event": None, "writes": [], "run": None, "span": None, "error": None}
    try:
        seq = _tr_int(one("seq"), None, "seq")
    except Exception as exc:
        out["error"] = str(exc)
        return out
    if seq is None:
        out["error"] = "seq is required"
        return out
    ev, err = _tr_call("get_event", seq=seq)
    if err:
        out["error"] = err
        return out
    if not ev:
        out["error"] = f"no event with seq={seq}"
        return out
    out["writes"] = ev.pop("writes", []) or []
    out["event"] = ev
    if ev.get("run_id"):                          # run + span context, from the same indexed run tree
        tree, terr = _tr_call("run_tree", run_id=ev["run_id"])
        if isinstance(tree, dict) and not terr:
            r = tree.get("run") or None
            if r:
                out["run"] = {k: r.get(k) for k in ("run_id", "kind", "command", "scene", "level", "lang",
                                                    "arm", "status", "started_ts", "decision_id", "agent_id")}
            for s in tree.get("spans") or []:
                if s.get("span_id") == ev.get("span_id"):
                    out["span"] = {k: s.get(k) for k in ("span_id", "name", "kind", "op", "step_ix",
                                                         "status", "duration_ms")}
                    break
    return out


def _trace_runs(one):
    """GET /trace_runs — newest-first run list with the denormalised span/event/write counters."""
    out = {"runs": [], "has_more": False, "error": None}
    try:
        kw = {"limit": _tr_int(one("limit"), 50, "limit"), "kind": _tr_str(one("kind")),
              "scene": _tr_str(one("scene")), "level": _tr_str(one("level")),
              "lang": _tr_str(one("lang")), "status": _tr_str(one("status")), "q": _tr_str(one("q"))}
        bt = _tr_str(one("before_ts"))
        kw["before_ts"] = float(bt) if bt else None
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("query_runs", **kw)
    if err:
        out["error"] = err
        return out
    for r in res.get("runs") or []:
        if r.get("started_ts") and r.get("ended_ts"):
            r["duration_ms"] = round((r["ended_ts"] - r["started_ts"]) * 1000.0, 1)
    res.setdefault("error", None)
    return res


def _trace_run(one):
    """GET /trace_run?run_id= — the run tree: spans FLAT with children ids + depth, each with its writes."""
    out = {"run": None, "spans": [], "roots": [], "events_by_span": {},
           "timeline": {"t0": None, "t1": None}, "truncated": False, "error": None}
    run_id = _tr_str(one("run_id"))
    if not run_id:
        out["error"] = "run_id is required"
        return out
    try:
        kw = {"run_id": run_id, "max_spans": _tr_int(one("max_spans"), 2000, "max_spans"),
              "min_level": (one("min_level", "DEBUG") or "DEBUG")}
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("run_tree", **kw)
    if err:
        out["error"] = err
        return out
    run = res.get("run")
    if run and run.get("config_json"):            # decoded once here so the client never parses JSON-in-JSON
        try:
            run["config"] = json.loads(run["config_json"])
        except Exception:
            run["config"] = None
    res.setdefault("error", None)
    return res


def _trace_entity(one):
    """GET /trace_entity?node=art:kb:procedure:12 (or ?store=&kind=&ref=) — the entity exhibit: the artifact,
    its LIVE current content, every cause ("what produced this KB entry?"), every later run that used it, and
    the lineage graph."""
    out = {"artifact": None, "live": None, "causes": [], "uses": [], "lineage": None, "error": None}
    node = _tr_str(one("node"))
    if node:
        aid = node[4:] if node.startswith("art:") else node
    else:
        store, kind, ref = _tr_str(one("store")), _tr_str(one("kind")), _tr_str(one("ref"))
        aid = None
        if store and kind and ref:
            safe = ref
            mk = getattr(_trace, "_safe_ref", None) if _trace is not None else None
            if callable(mk):
                try:
                    safe = mk(ref)
                except Exception:
                    safe = ref
            aid = f"{store}:{kind}:{safe}"
    if not aid:
        out["error"] = "node (or store + kind + ref) is required"
        return out
    art, err = _tr_call("artifact", artifact_id=aid)
    if err:
        out["error"] = err
        return out
    if not art:
        out["error"] = f"unknown artifact {aid!r}"
        return out
    out["causes"] = art.pop("causes", []) or []
    out["uses"] = art.pop("uses", []) or []
    out["live"] = art.pop("live", None)
    out["artifact"] = art
    try:
        depth, limit = _tr_int(one("depth"), 6, "depth"), _tr_int(one("limit"), 300, "limit")
    except Exception as exc:
        out["error"] = str(exc)
        return out
    lin, lerr = _tr_call("lineage", node="art:" + aid,
                         direction=(one("dir") or one("direction") or "both"), depth=depth, limit=limit)
    out["lineage"] = lin if isinstance(lin, dict) else {"root": "art:" + aid, "nodes": [], "edges": [],
                                                        "truncated": False, "error": lerr}
    return out


def _trace_lineage(one):
    """GET /trace_lineage?node=&dir=up|down|both — nodes+edges for the SVG graph. `node` is always bound as a
    SQL parameter inside trace.py; nodes[].depth is the MIN(depth) layer assignment."""
    out = {"root": _tr_str(one("node")), "direction": (one("dir") or "both"),
           "nodes": [], "edges": [], "truncated": False, "error": None}
    node = _tr_str(one("node"))
    if not node:
        out["error"] = "node is required"
        return out
    try:
        kw = {"node": node, "direction": (one("dir") or one("direction") or "both"),
              "depth": _tr_int(one("depth"), 6, "depth"), "limit": _tr_int(one("limit"), 300, "limit")}
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("lineage", **kw)
    if err:
        out["error"] = err
        return out
    res.setdefault("error", None)
    return res


def _trace_span_effects(one):
    """GET /trace_span_effects?span_id=&descendants=1 — "this action: what did it write?", one indexed query.
    (Also available per span inside /trace_run; this is the direct, cheap form for a drill-down.)"""
    out = {"span_id": _tr_str(one("span_id")), "writes": [], "error": None}
    if not out["span_id"]:
        out["error"] = "span_id is required"
        return out
    res, err = _tr_call("span_effects", span_id=out["span_id"],
                        include_descendants=_tr_bool(one("descendants"), False))
    if err:
        out["error"] = err
        return out
    out["writes"] = res or []
    return out


def _trace_db(one):
    """GET /trace_db — the browsable-table catalog across all three stores (kb JSON, memory.db, trace.db)."""
    out = {"tables": [], "stores": {}, "error": None}
    tables, err = _tr_call("db_tables")
    if err:
        out["error"] = err
        return out
    out["tables"] = tables or []

    def _rel(p):
        try:
            return os.path.relpath(p, os.path.dirname(HERE)).replace("\\", "/")
        except Exception:
            return str(p or "")
    cfg = getattr(_trace, "_CFG", {}) or {}
    out["stores"] = {"kb": _rel(getattr(_trace, "KB_PATH", "")),
                     "mem": _rel(getattr(_trace, "MEM_DB_PATH", "")) + " (read-only)",
                     "trace": _rel(cfg.get("db_path", "")),
                     "file": _rel(getattr(_trace, "LOOP_REPORTS_PATH", ""))}
    return out


def _trace_db_rows(one):
    """GET /trace_db_rows?table=&limit=&offset=&order_by=&desc=&q=&where_col=&where_val= — page one table.
    `table` MUST be a catalog id (trace.py rejects anything else before any SQL is built); order_by is
    validated against that table's own column list; memory.db is read through a mode=ro connection."""
    out = {"table": _tr_str(one("table")), "columns": [], "rows": [], "total": 0,
           "offset": 0, "limit": 100, "error": None}
    if not out["table"]:
        out["error"] = "table is required"
        return out
    try:
        kw = {"table": out["table"], "limit": _tr_int(one("limit"), 100, "limit"),
              "offset": _tr_int(one("offset"), 0, "offset"), "order_by": _tr_str(one("order_by")),
              "desc": _tr_bool(one("desc"), True), "q": _tr_str(one("q")),
              "where_col": _tr_str(one("where_col")), "where_val": _tr_str(one("where_val"))}
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("db_rows", **kw)
    if err:
        out["error"] = err
        return out
    res.setdefault("error", None)
    return res


def _trace_db_row(one):
    """GET /trace_db_row?table=&pk= — one row with its provenance fields, its artifact id, the writes that
    caused it, what it went on to influence, and the legacy audit lines that mention it."""
    out = {"table": _tr_str(one("table")), "pk": _tr_str(one("pk")), "row": None, "provenance": {},
           "artifact_id": None, "causes": [], "effects": [], "audit_lines": [], "error": None}
    if not out["table"] or out["pk"] is None:
        out["error"] = "table and pk are required"
        return out
    res, err = _tr_call("db_row", table=out["table"], pk=out["pk"])
    if err:
        out["error"] = err
        return out
    res.setdefault("error", None)
    return res


def _trace_stats(one):
    """GET /trace_stats — SQL COUNT(*) GROUP BY aggregates + tracer health (never ships rows to the client)."""
    out = {"events_by_type": {}, "events_by_level": {}, "spans_by_kind": {}, "writes_by_store_verdict": [],
           "artifacts_by_source": {}, "artifacts_by_store_kind": [],
           "runs": {"total": 0, "ok": 0, "error": 0, "open": 0}, "orphans": {"writes": 0, "events": 0},
           "top_artifacts": [], "recent_rate": [], "db": {}, "tracer": {}, "error": None}
    try:
        w = _tr_str(one("window_s"))
        window = float(w) if w else None
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("stats", window_s=window)
    if err:
        out["error"] = err
        return out
    res.setdefault("error", None)
    if not res.get("tracer"):
        st, _ = _tr_call("status")
        res["tracer"] = st or {"enabled": False, "degraded": True, "reason": _TRACE_OFF}
    return res


def _trace_tail(one):
    """GET /trace_tail?after=<seq>&limit=&min_level= — the LIVE TAIL, ascending, cursor-paged.

    Transport decision (map-server.md §5.1, design §9.3): cursor polling, NOT a held stream. This server is
    HTTP/1.0 ThreadingHTTPServer — one thread per connection — and /stream (MJPEG) plus the 400 ms /status
    poll already eat the browser's ~6 connections per origin. So this handler is bounded by construction: it
    runs ONE indexed query and returns immediately, holding no thread and no server state, which also makes
    it restart-safe (the client just feeds `last_seq` back as `after`)."""
    out = {"events": [], "last_seq": 0, "has_more": False,
           "counts": {"events": 0, "runs": 0, "writes": 0},
           "tracer": {"queue_depth": 0, "dropped": 0, "degraded": True, "enabled": False}, "error": None}
    try:
        after = _tr_int(one("after"), 0, "after")
        kw = {"after": after, "limit": _tr_int(one("limit"), 200, "limit"), "order": "asc",
              "min_level": _tr_str(one("min_level")), "run_id": _tr_str(one("run_id")),
              "type": _tr_str(one("type"))}
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("query_events", **kw)
    if err:
        out["error"] = err
        return out
    out["events"] = res.get("events") or []
    out["has_more"] = bool(res.get("has_more"))
    out["error"] = res.get("error")
    out["last_seq"] = (res.get("cursor") or {}).get("last_seq") or after or 0
    st, _serr = _tr_call("status")
    if isinstance(st, dict):
        out["counts"] = {"events": st.get("events", 0), "runs": st.get("runs", 0),
                         "writes": st.get("writes", 0)}
        out["tracer"] = {"queue_depth": st.get("queue_depth", 0), "dropped": st.get("dropped", 0),
                         "degraded": bool(st.get("degraded")), "enabled": bool(st.get("enabled")),
                         "reason": st.get("reason")}
    return out


def _trace_search(one):
    """GET /trace_search?q= — cross-store LIKE search (run commands, event summaries, artifacts, write whys)."""
    out = {"q": _tr_str(one("q")) or "", "runs": [], "events": [], "artifacts": [], "writes": [], "error": None}
    if not out["q"]:
        out["error"] = "q is required"
        return out
    try:
        limit = _tr_int(one("limit"), 40, "limit")
    except Exception as exc:
        out["error"] = str(exc)
        return out
    res, err = _tr_call("search", q=out["q"], limit=limit)
    if err:
        out["error"] = err
        return out
    res.setdefault("error", None)
    res["q"] = out["q"]
    return res


def _trace_prov(one):
    """GET /trace_prov?run_id=|node= — PROV-JSON export of one run or one node's lineage (thesis appendix)."""
    out = {"prov": None, "error": None}
    run_id, node = _tr_str(one("run_id")), _tr_str(one("node"))
    if not run_id and not node:
        out["error"] = "run_id or node is required"
        return out
    res, err = _tr_call("export_prov_json", run_id=run_id, node=node)
    if err:
        out["error"] = err
        return out
    out["prov"] = res
    return out


# Exact-path dispatch table (the house style is exact matches in do_GET; this keeps the chain readable and
# every handler individually callable from a test with a plain `one(key, default)` accessor).
_TRACE_ROUTES = {
    "/trace_events": _trace_events,
    "/trace_event": _trace_event,
    "/trace_runs": _trace_runs,
    "/trace_run": _trace_run,
    "/trace_entity": _trace_entity,
    "/trace_lineage": _trace_lineage,
    "/trace_span_effects": _trace_span_effects,
    "/trace_db": _trace_db,
    "/trace_db_rows": _trace_db_rows,
    "/trace_db_row": _trace_db_row,
    "/trace_stats": _trace_stats,
    "/trace_tail": _trace_tail,
    "/trace_search": _trace_search,
    "/trace_prov": _trace_prov,
}


# ----------------------------------------------------------------------- server
# ---------------------------------------------------------------- honest mode (console v2, docs/console-v2-spec.md)
# Everything the honest agent decides from goes through sim/honest_console.py -> honest.HonestAgent (guarded by
# sim/_step4_rules_selftest.py). What lives HERE is console plumbing only: the execution lease every simulator job
# claims, the per-primitive frame hook (so a blocking honest step still streams), the sensor switch, the
# report-only goal check, and the admission rules for manual input while the honest agent acts.
try:
    import honest_console as _hc
    _HC_ERROR = None
except Exception as _exc:              # honest mode must never stop the oracle console from starting
    _hc = None
    _HC_ERROR = _exc
_HC_OFF = f"honest mode unavailable: honest_console did not import ({_HC_ERROR!r})"


class _PlainLease:
    """Fallback lease when honest_console cannot import (same claim/release contract, no generations)."""

    def __init__(self):
        self._lk, self._cur = threading.Lock(), None

    def claim(self, owner, job_id, sim_bound=True, on_claim=None):
        with self._lk:
            if self._cur is not None:
                return False, dict(self._cur)
            self._cur = {"owner": owner, "job_id": job_id, "generation": 0, "sim_bound": sim_bound,
                         "since": time.time()}
            if on_claim:
                on_claim()
            return True, None

    def release(self, job_id):
        with self._lk:
            if self._cur is not None and self._cur["job_id"] == job_id:
                self._cur = None
                return True
            return False

    def holder(self):
        with self._lk:
            return dict(self._cur) if self._cur else None

    def owner(self):
        h = self.holder()
        return h["owner"] if h else None


#: action revision: bumped for every simulator action that can change what the camera sees, WHOEVER sends it
#: (honest agent, manual drive, oracle plan). A drawn detection is valid only while it is unchanged.
_ACT_REV = {"n": 0}
_REV_FREE = frozenset({"Pass", "Done", "GetObjectInFrame", "GetReachablePositions"})


def _install_act_rev():
    """Wrap ThorAgent.do / reset / set_sensors (after the trace hooks) to count camera-changing actions."""
    if ta is None or getattr(ta.ThorAgent, "_act_rev_installed", False):
        return
    import functools
    o_do, o_reset, o_sens = ta.ThorAgent.do, ta.ThorAgent.reset, ta.ThorAgent.set_sensors

    @functools.wraps(o_do)
    def do(self, action, **kw):
        if action not in _REV_FREE:
            _ACT_REV["n"] += 1
        return o_do(self, action, **kw)

    @functools.wraps(o_reset)
    def reset(self, *a, **kw):
        _ACT_REV["n"] += 1
        return o_reset(self, *a, **kw)

    @functools.wraps(o_sens)
    def set_sensors(self, *a, **kw):
        _ACT_REV["n"] += 1
        return o_sens(self, *a, **kw)

    ta.ThorAgent.do, ta.ThorAgent.reset, ta.ThorAgent.set_sensors = do, reset, set_sensors
    ta.ThorAgent._act_rev_installed = True


_install_act_rev()

#: routes that act with simulator truth: refused in honest mode even between instructions (spec 2.4)
_HONEST_TRUTH_ROUTES = frozenset({"/batch", "/act_obj", "/held", "/scan", "/explore", "/cmd", "/eval_run"})
#: routes allowed between honest instructions but refused while the honest agent acts (spec 2.2)
_HONEST_IDLE_ROUTES = frozenset({"/action", "/drive", "/setparam", "/scene", "/randomize", "/checkpoint",
                                 "/restore", "/resolution", "/reset"})


def _honest_on():
    return _HONEST is not None and _HONEST.on


def _honest_acting():
    return _HONEST is not None and _HONEST.job_running()


def _honest_world(reason, hand="empty", scene=""):
    if _HONEST is not None:
        _HONEST.world_changed(reason, hand, scene)


def _honest_after(r, reason, hand):
    """Record a world change after ``r`` -- at once, or when ``r`` (a generator job) has finished."""
    if hasattr(r, "__next__"):
        def gen():
            try:
                yield from r
            finally:
                _honest_world(reason, hand)
        return gen()
    _honest_world(reason, hand)
    return r


def _manual(fn, kind=None):
    """A manual-input job, re-checked ON THE WORKER: a job admitted just before the honest agent claimed the
    lease is dropped instead of running between two of its steps (Astra, backend review 1)."""
    def job(ag):
        if _honest_on() and _LEASE.owner() in ("honest", "mode-switch"):
            _log(f"manual input dropped: the honest agent is acting{f' ({kind})' if kind else ''}")
            return None
        if kind and _honest_on():
            _HONEST.intervention(kind)
        return fn(ag)
    return job


def _leased(job_fn, jid):
    """A generator job that holds the lease ``jid`` until the generator finishes (or is dropped)."""
    def job(ag):
        try:
            g = job_fn(ag)
        except BaseException:
            _LEASE.release(jid)
            raise
        if not hasattr(g, "__next__"):
            _LEASE.release(jid)
            return g

        def run():
            try:
                yield from g
            finally:
                _LEASE.release(jid)
        return run()
    return job


_HGLITCH = {"hits": []}


def _honest_prim_hook(ag, action):
    """After every honest primitive (the worker is blocked inside a step): stream the frame, refresh the DISPLAY
    telemetry (simulator display, never shown to the agent), watch for render corruption, apply the motion pace."""
    if action in _REV_FREE:
        return
    now = time.time()
    _SIM["beat"] = now
    _IDLE["last"] = now                  # an honest run is activity: the idle cadence never throttles it
    try:
        m = _view["mode"]
        if m in ("map", "mapnav") and ag.third_party_frames:
            frame = ag.third_party_frames[0]
        elif m == "depth" and ag.depth_frame is not None:
            frame = ta.depth_to_gray(ag.depth_frame)
        else:
            frame = ag.frame
        if getattr(frame, "ndim", 2) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=80)
        p = ag.agent_pose()
        held = ag.held_object()
        with _frame_cv:
            _publish_frame(buf.getvalue())
            _state["motion_label"] = f"honest: {action}"
            _state["held"] = held["objectType"] if held else "none"
            _state["pose"] = {"x": round(p["position"]["x"], 2), "y": round(p["position"]["y"], 2),
                              "z": round(p["position"]["z"], 2), "rot": round(p["rotation"]["y"]),
                              "horizon": round(p["horizon"]), "standing": p["standing"]}
            _SIM["frame_ts"] = now
            _frame_cv.notify_all()
        if m in ("rgb", "detect"):
            sig = _frame_glitch(ag.frame[::6, ::6])
            if sig:
                _HGLITCH["hits"] = [t for t in _HGLITCH["hits"] if now - t < GLITCH_WINDOW_S] + [now]
                if len(_HGLITCH["hits"]) >= GLITCH_HITS:
                    _HGLITCH["hits"] = []
                    _GLITCH["force"] = sig
                    _HONEST.render_trip(sig)
    except Exception:
        pass
    # pacing is HonestAgent's smooth profile (a pause per physical send, see HONEST_SMOOTH), not the oracle delay


def _honest_overlay_frame(agent):
    """Detect view in honest mode: the AGENT's last-look boxes while nothing has moved since it looked; otherwise
    the live detector on the current camera frame (display only -- the same camera + detector the agent uses,
    never fed back to it), so the Detect view always shows what the detector sees."""
    dets = _HONEST.overlay(_SIM["generation"], _ACT_REV["n"])
    if not dets:
        return _draw_detections(agent)
    import numpy as np
    from PIL import ImageDraw
    a = np.asarray(agent.frame)
    img = Image.fromarray(a[:, :, :3] if a.ndim == 3 and a.shape[2] == 4 else a)
    _hc.draw_dets(ImageDraw.Draw(img), dets, img.size)
    return np.asarray(img)


def _honest_apply_sensors(ag, on):
    """Worker: entering = ONE announced scene reload with depth on (fresh world, empty hand); leaving = the
    sensors the current view needs (may reload, announced)."""
    mode = _view["mode"]
    if on:
        if mode in ("instance", "semantic", "normals"):
            mode = "rgb"
        ag.set_sensors(depth=True, force=True)
        _view["map_cam"] = False
        _set_view(ag, mode, depth=True)
        _log("honest mode ON: scene reloaded once with the depth sensor on (fresh world, empty hand)")
    else:
        _set_view(ag, mode, depth=False)
        _log("honest mode OFF: sensors back to what the view needs (this may reload the scene)")


def _honest_goal_check(ag, item):
    """Report-only goal check of a picked dataset item. Called by honest_console ONLY inside h.scoring()."""
    import eval as ev
    import general_set as gs
    import step4_refplans as rp
    goal = gs.gold_of(item)["goal"]
    ok = rp._goal_ok(ev, ag, goal)
    return {"ok": bool(ok), "unmet": [] if ok else rp._unmet(ev, gs, ag, goal)}


def _honest_availability():
    if ta is None:
        return False, "the simulator is not available (WSL only)"
    scene = (_state.get("scene") or "").replace("_physics", "")
    if scene and scene != "FloorPlan1":
        return False, f"honest mode is FloorPlan1 only (the scene is {scene})"
    if not os.path.isfile(os.path.join(HERE, "step4", "FloorPlan1.map.json")):
        return False, "the walked map sim/step4/FloorPlan1.map.json is missing"
    try:
        import perception as _perc
        if not _perc.v3_paths("FloorPlan1", os.path.join(HERE, "weights"))["primary"]:
            return False, "no step-3 detector weights for FloorPlan1 (weights/v3)"
    except Exception as exc:
        return False, f"perception unavailable: {exc}"
    return True, ""


_MODEL_DIGEST = {}


def _honest_model_info(endpoint, model):
    key = (endpoint, model)
    if key not in _MODEL_DIGEST:
        import urllib.request
        with urllib.request.urlopen(endpoint.rstrip("/") + "/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode())
        d = next((m.get("digest") for m in tags.get("models", []) if m.get("name") == model), None)
        _MODEL_DIGEST[key] = (d or "")[:12] or None
    return _MODEL_DIGEST[key]


def _honest_items():
    import general_set as gs
    return gs.load(GENERAL_PATH)["items"]


def _honest_store():
    from memory import Memory
    return Memory(path=os.path.join(HERE, "step4", "console_memory.db"))


def _honest_load_detector():
    import honest
    return honest.load_detector("FloorPlan1", os.path.join(HERE, "weights"))


def _honest_clear_drive():
    _drive["action"] = None


_HONEST = None
if _hc is not None:
    _HONEST = _hc.Console(
        submit=_jobs.put, sim_generation=lambda: _SIM["generation"], scene_now=lambda: _state.get("scene") or "",
        on_primitive=_honest_prim_hook, goal_check=_honest_goal_check, load_detector=_honest_load_detector,
        make_provider=lambda ep: al.make_provider("ollama", endpoint=ep), apply_sensors=_honest_apply_sensors,
        clear_drive=_honest_clear_drive, act_rev=lambda: _ACT_REV["n"], items=_honest_items,
        model_info=_honest_model_info, log=_log, availability=_honest_availability, make_store=_honest_store,
        default_endpoint=lambda: _llm_endpoint(), llm_on=lambda: bool(_LLM["on"]), detect_lock=_DETECT_LOCK,
        smooth=_honest_smooth)
_LEASE = _HONEST.lease if _HONEST is not None else _PlainLease()


_CT = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
       ".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg", ".woff2": "font/woff2"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        """JSON with ensure_ascii=False. Persian commands are first-class in this data and the legacy _send()
        escapes them to \\uXXXX, inflating a Persian payload ~6x. default=str keeps a stray sqlite BLOB or
        bytes value from turning a read-only viewer request into a broken response."""
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", code)

    def _file(self, path):
        if not os.path.isfile(path):
            self._send(b"not found", "text/plain", 404); return
        ctype = _CT.get(os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as fh:
            self._send(fh.read(), ctype)

    def do_POST(self):
        """The console's only writes that carry a BODY: edits to the general instruction set. Everything else is
        a GET with query parameters, which is no place for a paragraph of Persian plus three text boxes."""
        path = urllib.parse.urlparse(self.path).path
        routes = {"/general_save": _general_save, "/general_delete": _general_delete,
                  "/general_review": _general_review}
        fn = routes.get(path)
        if fn is None:
            self._send(b"not found", "text/plain", 404); return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 256 * 1024:
                self._json({"ok": False, "errors": ["request too large"]}, 413); return
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except Exception as exc:
            self._json({"ok": False, "errors": [f"bad request: {exc}"]}, 400); return
        try:
            self._json(fn(payload))
        except Exception as exc:
            self._json({"ok": False, "errors": [str(exc)]}, 500)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        one = lambda k, d="": (q.get(k) or [d])[0]
        path = u.path

        if self._honest_gate(path, one):
            return
        if path == "/":
            self._file(os.path.join(WEB, "index.html"))
        elif path.startswith("/static/"):
            self._file(os.path.join(WEB, os.path.basename(path)))
        elif path == "/stream":
            try:
                import socket as _socket
                self.connection.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 192 * 1024)
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            last = None                  # None => first frame: start at the newest, never replay a backlog
            try:
                while True:
                    with _frame_cv:
                        _frame_cv.wait_for(lambda: last is None or _state["seq"] != last, timeout=5.0)
                        pending = _stream_pending(last)
                    for sq, jpg in pending:
                        if not jpg:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        last = sq
            except (BrokenPipeError, ConnectionResetError):
                return
        elif path == "/status":
            with _lock:
                st = {k: _state[k] for k in ("fps", "scene", "held", "busy", "view", "recording",
                                             "plan", "pose", "gpu", "resolution", "log", "train", "ncam", "demo")}
                loop_last = _state.get("loop_last")
                motion_label, seq, dropped = (_state.get("motion_label") or "",
                                              _state.get("seq", 0), _state.get("dropped", 0))
            st["perception"] = _PERCEPTION["mode"]
            # `degraded` is the REASON perceived mode fell back, and `ready` says whether it could run at all.
            # Both are surfaced so the console can never show a "perceived" badge over an oracle observation.
            st["perception_degraded"] = _PERCEPTION.get("degraded")
            st["perception_ready"] = _perception_ready(st.get("scene") or "")
            st["detectors"] = _detectors_info(st.get("scene") or "")
            st["render"] = {"glitch": _GLITCH["last"], "restarts": _GLITCH["restarts"]}
            st["idle"] = bool(_IDLE["on"])                 # idle cadence: the fps is low on purpose
            age = time.time() - (_SIM.get("frame_ts") or _SIM["beat"] or 0)
            waiting = _HONEST.waiting_text() if _HONEST is not None else ""
            sim_state = _SIM["state"]
            if sim_state == "ok" and age >= 3.0:
                # an honest step computing between primitives (detector warm-up, A*) is busy, not stalled, until the
                # simulator step timeout -- after that a hung simulator is reported as stalled as before
                sim_state = ("waiting-llm" if waiting else
                             "busy" if _honest_acting() and age < SIM_STEP_TIMEOUT_S else "stalled")
            st["sim"] = {"state": sim_state, "frame_age_s": round(age, 1), "generation": _SIM["generation"],
                         "recoveries": _SIM["recoveries"], "error": _SIM["error"], "waiting": waiting,
                         "beat_age_s": round(time.time() - (_SIM["beat"] or 0), 1)}
            st["honest"] = (_HONEST.status_brief() if _HONEST is not None
                            else {"on": False, "phase": "idle", "rev": 0, "job_id": None, "waiting": ""})
            st["lease"] = _LEASE.holder()
            st["busy"] = bool(st["busy"] or _honest_acting())
            if age >= 3.0:
                st["fps"] = 0.0                         # never show a stale fps over a frozen picture
            hon = _honest_on()
            st["motion"] = {"speed": _HONEST_MOTION["speed"] if hon else _MOTION["speed"], "label": motion_label,
                            "seq": seq, "dropped": dropped, "honest_speed": _HONEST_MOTION["speed"],
                            "note": (MOTION_PROFILES.get(_HONEST_MOTION["speed"], {}).get("note", "")
                                     + " (honest: display only)") if hon else _mp().get("note", "")}
            st["api"] = WEBAPP_API
            st["llm"] = dict(_LLM)
            st["loop"] = {"enabled": LOOP_ENABLED, "last": loop_last}
            st["eval"] = _eval_snapshot()
            self._send(st)
        elif path == "/objects":
            with _lock:
                self._send(_state["objects"])
        elif path == "/honest":
            if _HONEST is None:
                self._json({"ok": False, "on": False, "error": _HC_OFF}, 503); return
            if one("ticket"):
                self._json({"ok": True, "ticket": _HONEST.history_ticket(one("ticket"))}); return
            on = one("on", "")
            if on in ("0", "1"):
                r = _HONEST.switch(on == "1")
                if not r.get("ok"):
                    self._json({**r, "on": _HONEST.on}, r.get("code", 400)); return
            self._json(_HONEST.snapshot(full=one("full") == "1"))
        elif path in ("/honest_run", "/honest_stop", "/honest_look", "/honest_forget"):
            if _HONEST is None:
                self._json({"ok": False, "error": _HC_OFF}, 503); return
            if path == "/honest_run":
                r = _HONEST.run(one("text"), one("model"), one("endpoint"), one("item"), one("lang"))
            elif path == "/honest_stop":
                r = _HONEST.stop()
            elif path == "/honest_look":
                r = _HONEST.look()
            else:
                r = _HONEST.forget()
            self._json(r, 200 if r.get("ok") else r.get("code", 400))
        elif path == "/honest_last_look.jpg":
            data = _HONEST.last_look_jpeg() if _HONEST is not None else b""
            if data:
                self._send(data, "image/jpeg")
            else:
                self._send(b"no observation yet", "text/plain", 404)
        elif path == "/episodes":
            # Phase-5 distillation funnel, surfaced as JSON for the benchmark/GUI: the RAW append-only episode log
            # alongside the DISTILLED tiers (row counts + failure counters + dedup ratio). Fully exception-safe —
            # a memory-backend failure returns an {'error':...} envelope, never a 500, and never touches the worker.
            out = {"episodes": [], "stats": {}, "failures": [], "dedup": {}}
            try:
                limit = int(one("limit", "50") or "50")     # parse inside the guard so ?limit=abc -> envelope, not 500
                m = _get_memory()
                out["episodes"] = m.recent_episodes(limit)
                out["stats"] = m.stats()
                out["failures"] = m.top_failures()
                out["dedup"] = m.dedup_stats()
            except Exception as exc:
                out["error"] = str(exc)
            self._send(out)
        elif path == "/bandit":
            self._send(_bandit_status())
        elif path == "/loop":
            # Phase-7 loop endpoint: ?on=1|0 flips the toggle (runtime, also freezes bandit updates); the
            # response always carries the toggle state + last write-back summary + recent loop reports.
            on = one("on", "")
            if on in ("0", "1"):
                _set_loop_enabled(on == "1")
            self._send(_loop_status(one("limit", "10")))
        elif path == "/rating":
            self._send(_rate_plan(one("decision_id"), one("rating")))
        elif path == "/teach":
            self._send(_store_human_teaching(one("text")))
        elif path == "/demo":
            on = one("on", "")
            if on == "1":
                self._send(_start_demo(one("command")))
            elif on == "0":
                self._send(_stop_demo(one("command"), save=(one("save", "1") != "0")))
            else:
                self._send({"ok": True, "demo": _demo_snapshot()})
        elif path == "/perception_feedback":
            self._send(_store_perception_feedback(one("object"), one("receptacle"), one("bbox"), one("note")))
        elif path == "/defaults":
            cfg = {}
            try:
                cfg = json.load(open(os.path.join(HERE, "gui_config.json")))
            except Exception:
                pass
            self._send({"endpoint": cfg.get("endpoint") or (al.gateway_endpoint() or "http://localhost:11434"),
                        "model": cfg.get("model", ""), "scenes": ta.KITCHEN_SCENES,
                        "liquids": ta.LIQUIDS})
        elif path == "/llm":
            on = one("on", "")
            if on in ("0", "1"):
                self._send(_set_llm(on == "1", one("endpoint")))
            else:
                self._send({"ok": True, **_LLM})
        elif path == "/models":
            ep = one("endpoint") or (al.gateway_endpoint() or "http://localhost:11434")
            if not _LLM["on"]:
                self._send({"ok": False, "off": True, "models": [], "endpoint": ep, "error": _LLM_OFF_MSG})
                return
            try:
                models = al.make_provider(endpoint=ep).list_models(timeout=5)
                self._send({"ok": True, "models": models, "endpoint": ep})
            except Exception as exc:
                self._send({"ok": False, "models": [], "endpoint": ep, "error": str(exc)})
        elif path == "/setmodel":
            model, ep = one("model"), one("endpoint")
            try:
                cfgp = os.path.join(HERE, "gui_config.json")
                cfg = json.load(open(cfgp)) if os.path.isfile(cfgp) else {}
                if model:
                    cfg["model"] = model
                if ep:
                    cfg["endpoint"] = ep
                json.dump(cfg, open(cfgp, "w"), indent=2)
                _log(f"model → {model}")
                self._send({"ok": True})
            except Exception as exc:
                self._send({"ok": False, "error": str(exc)})
        elif path == "/action":
            name = one("name")
            ok = {"move_ahead", "move_back", "move_left", "move_right", "rotate_left", "rotate_right",
                  "look_up", "look_down", "crouch", "stand", "drop"}
            if name in ok:
                _record_demo_step({"action": name})
                _jobs.put(_manual(lambda ag, n=name: getattr(ag, n)(), "moved by hand"))
            self._send({"ok": True})
        elif path == "/drive":                             # held-key continuous smooth drive (keepalive start/stop)
            act, on = one("action"), one("on", "1")
            allowed = {"move_ahead", "move_back", "move_left", "move_right",
                       "rotate_left", "rotate_right", "look_up", "look_down"}
            if on == "0":
                _drive["action"] = None                    # explicit stop (keyup)
            elif act in allowed:
                if _drive["action"] != act and _honest_on() and _HONEST is not None:
                    _HONEST.intervention("driven by hand")
                _drive["action"] = act
                _drive["t"] = time.time()                  # refresh keepalive; the worker auto-stops if it goes stale
            self._send({"ok": True})
        elif path == "/held":
            name = one("name")
            m = {"drop": lambda ag: ag.drop(), "throw": lambda ag: ag.throw(150),
                 "nudge": lambda ag: ag.move_held(ahead=0.1), "spin": lambda ag: ag.rotate_held(yaw=30)}
            if name in m:
                _record_demo_step({"action": name})
                _jobs.put(_manual(m[name]))
            self._send({"ok": True})
        elif path == "/cmd":
            text, model, ep = one("text"), one("model"), one("endpoint", "http://localhost:11434")
            if re.match(r"^\s*(teach|correction|correct|remember|fact|lesson|spatial)\s*:", text or "", flags=re.I):
                self._send(_store_human_teaching(text))
                return
            if not _LLM["on"]:
                _log(f"not planned: {_LLM_OFF_MSG}")
                self._send({"ok": False, "error": _LLM_OFF_MSG})
                return
            holder = None
            with _lock:                                       # serialize commands: an overlapping plan would race on
                busy = _state.get("busy")                      # cur_cmd/cur_level and cross-attribute passive episodes
                start = bool(not busy and text and model)
                if start:
                    jid = f"cmd{time.time():.3f}"
                    got, holder = _LEASE.claim("oracle-cmd", jid)
                    start = got
                    if got:
                        _state["busy"] = True                  # claim synchronously so two quick /cmd posts can't both pass
                        _state["cmd_job"] = jid
            if start:
                threading.Thread(target=_plan_and_run, args=(text, model, ep), daemon=True).start()
                self._send({"ok": True})
            else:
                self._send({"ok": False, "error": ("a command is already running" if busy else
                                                   f"busy: {holder['owner']} is running" if holder else
                                                   "missing text/model"),
                            "owner": (holder or {}).get("owner")})
        elif path == "/act_obj":
            oid, action, liq = one("id"), one("action"), one("liquid", "water")
            opn = float(one("openness", "1"))
            if oid and action:
                step = {"action": _planner_action(action), "object": oid.split("|")[0]}
                if step["action"] == "put":
                    with _lock:
                        held = _state.get("held")
                    step["object"] = held if held and held != "none" else "held_object"
                    step["to"] = oid.split("|")[0]
                elif step["action"] == "fill_with_liquid":
                    step["arg"] = liq
                _record_demo_step(step)
                _jobs.put(_manual(lambda ag, i=oid, a=action, l=liq, o=opn: _act_obj(ag, i, a, l, o)))
            self._send({"ok": True})
        elif path == "/click":
            x, y, shift = float(one("x", "0")), float(one("y", "0")), one("shift") == "1"
            self._send(self._do_click(x, y, shift))
        elif path == "/scene":
            name = one("name", "FloorPlan1")
            _jobs.put(_manual(lambda ag, n=name: (ag.reset(n), _set_view(ag, "rgb"), _log(f"loaded {n}"),
                                                  _honest_world(f"scene {n} loaded", "empty", n))[-2]))
            self._send({"ok": True})
        elif path == "/view":
            _jobs.put(lambda ag, m=one("mode", "rgb"): _set_view(ag, m))
            self._send({"ok": True})
        elif path == "/scan":
            params = {"per_class_target": int(one("per_class_target", "40")),
                      "max_positions": int(one("max_positions", "24")),
                      "do_states": one("do_states", "1") != "0",
                      "do_destructive": one("do_destructive", "0") == "1",
                      # Container pass ON by default: without it every class that lives inside a shut cupboard
                      # trains on zero frames. `do_containers=0` reproduces the pre-2026-08-22 scan for comparison.
                      "do_containers": one("do_containers", "1") != "0",
                      "max_containers": int(one("max_containers", "10")),
                      "container_views": int(one("container_views", "3"))}
            jid = f"scan{time.time():.3f}"
            got, holder = _LEASE.claim("scan", jid)
            if not got:
                self._json({"ok": False, "error": f"busy: {holder['owner']} is running", "owner": holder["owner"]},
                           409); return
            _jobs.put(_leased(lambda ag, p=params: _scan_job(ag, p), jid))   # generator -> worker streams progress
            self._send({"ok": True})
        elif path == "/explore":
            jid = f"explore{time.time():.3f}"
            got, holder = _LEASE.claim("explore", jid)
            if not got:
                self._json({"ok": False, "error": f"busy: {holder['owner']} is running", "owner": holder["owner"]},
                           409); return
            _jobs.put(_leased(lambda ag: _explore_job(ag), jid))   # full-map navigation exploration (streams progress)
            self._send({"ok": True})
        elif path == "/train":
            with _lock:
                scene = _state.get("scene") or "FloorPlan1"
                busy = (_state.get("train") or {}).get("status") == "running"
            if busy:
                self._send({"ok": False, "error": "training already running"})
            else:
                epochs, device = int(one("epochs", "0")), one("device", "0")   # 0 -> yolo_recipe.DEFAULT_EPOCHS
                dev = int(device) if device.lstrip("-").isdigit() else device
                threading.Thread(target=_train_scene_bg, args=(scene, epochs, dev), daemon=True).start()
                self._send({"ok": True, "scene": scene})
        elif path == "/dataset":
            self._send(_dataset_summary(one("scene"), one("level"), one("lang"), one("limit", "200")))
        elif path == "/general":
            self._json(_general_list(one("level"), one("q"), one("show")))
        elif path == "/general_export":
            import general_set as gs
            # BOM: without it Excel opens a UTF-8 CSV as ANSI and the Persian column turns to mojibake
            body = ("﻿" + gs.export_csv(gs.load())).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="general_instructions.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/eval_results":
            self._send(_read_eval_results(one("file")))
        elif path == "/eval_status":
            self._send(_eval_snapshot())
        elif path == "/eval_run":
            with _lock:
                running = _EVAL["running"]
            if running:
                self._send({"ok": False, "error": "an evaluation is already running"})
            elif not _LLM["on"]:
                self._send({"ok": False, "error": _LLM_OFF_MSG})
            elif not _LEASE.claim("eval", "eval-pending", sim_bound=False)[0]:
                self._json({"ok": False, "error": f"busy: {_LEASE.owner()} is running", "owner": _LEASE.owner()},
                           409)
            else:
                cfg = {"models": one("models", "gemma3:4b") or "gemma3:4b",
                       "rag": one("rag", "off"), "loop": one("loop", "off"),
                       "lang": one("lang", "all"), "scenes": one("scenes", ""), "limit": one("limit", ""),
                       "dataset": ("general" if one("dataset", "scene") == "general" else "scene")}
                if cfg["rag"] not in ("off", "keyword", "hybrid"):
                    cfg["rag"] = "off"
                if cfg["loop"] not in ("on", "off"):
                    cfg["loop"] = "off"
                if cfg["lang"] not in ("all", "en", "fa"):
                    cfg["lang"] = "all"
                threading.Thread(target=_run_eval_bg, args=(cfg, "eval-pending"), daemon=True).start()
                self._send({"ok": True, "config": cfg})
        elif path == "/navmap":
            self._send(self._navmap_json())
        elif path == "/inspect":
            mode = one("mode", "auto")
            if mode not in ("auto", "off", "always"):
                self._send({"ok": False, "error": "mode must be auto | off | always"}, code=400)
                return
            _INSPECT["mode"] = mode
            if _detector.get("obj") is not None and hasattr(_detector["obj"], "mode"):
                _detector["obj"].mode = mode
            _log({"auto": "inspection model: AUTO — the ambiguity specialist takes over its 10 look-alike classes "
                          "only while the agent holds an object up",
                  "off": "inspection model: OFF — primary detector only",
                  "always": "inspection model: ALWAYS — specialist on every frame (ablation, not the deployment "
                            "rule)"}[mode])
            with _lock:
                scene_now = _state.get("scene") or ""
            self._send({"ok": True, "detectors": _detectors_info(scene_now)})
        elif path == "/perception":
            mode = one("mode", "oracle")
            if mode in ("oracle", "perceived"):
                _PERCEPTION["mode"] = mode
                _PERCEPTION["degraded"] = None          # a fresh choice starts with a clean slate
                if mode == "perceived":
                    _log("perception: PERCEIVED — the planner now sees ONLY what the YOLO detector finds while "
                         "the agent turns and searches; the scene object list is withheld")
                    if one("view", "1") not in ("0", "false", "no"):
                        _jobs.put(lambda ag: _set_view(ag, "detect"))   # show what the model itself sees
                else:
                    _log("perception: ORACLE — the planner is handed AI2-THOR's ground-truth object list")
            with _lock:
                scene_now = _state.get("scene") or ""
            self._send({"ok": True, "mode": _PERCEPTION["mode"], "degraded": _PERCEPTION.get("degraded"),
                        "ready": _perception_ready(scene_now)})
        elif path == "/reset":
            scope = one("scope", "session")
            if scope not in RESET_SCOPES:
                self._send({"ok": False, "error": f"scope must be one of {RESET_SCOPES}"}, code=400)
                return
            with _lock:
                busy = _state.get("busy")
            if busy:
                # Wiping the stores under a running plan would race every write it is still making.
                self._send({"ok": False, "error": "a command is still running - wait for it to finish"},
                           code=409)
                return
            jid = f"reset{time.time():.3f}"
            got, holder = _LEASE.claim("reset", jid)
            if not got:
                self._json({"ok": False, "error": f"busy: {holder['owner']} is running", "owner": holder["owner"]},
                           409); return
            try:
                wiped = _reset_learned() if scope == "all" else None
                done = threading.Event()

                def _rs(ag):
                    try:
                        _reset_session(ag)
                        _honest_world("scene reset", "empty")
                    finally:
                        done.set()
                _jobs.put(_rs)
                done.wait(timeout=30)
            finally:
                _LEASE.release(jid)
            if wiped:
                _log("reset: cleared " + ", ".join(wiped["cleared"]))
                for f in wiped["failed"]:
                    _log(f"reset FAILED for {f}")
            self._send({"ok": True, "scope": scope, "wiped": wiped})
        elif path == "/motion":
            sp = one("speed", "")
            if sp in MOTION_PROFILES:
                _MOTION["speed"] = sp
                _HONEST_MOTION["speed"] = sp           # the same menu paces the honest agent (display only)
                _jobs.put(_apply_motion)               # re-time the live agent, not just the next plan
                _log(f"motion detail: {sp} — {MOTION_PROFILES[sp]['note']}")
            self._send({"ok": True, "speed": _MOTION["speed"],
                        "profiles": {k: v["note"] for k, v in MOTION_PROFILES.items()}})
        elif path == "/resolution":
            w, h = int(one("w", "600")), int(one("h", "600"))
            _jobs.put(_manual(lambda ag: (ag.change_resolution(w, h), _log(f"resolution {w}x{h}"))[-1]))
            self._send({"ok": True})
        elif path == "/randomize":
            what = one("what")
            m = {"materials": lambda ag: ag.randomize_materials(), "lighting": lambda ag: ag.randomize_lighting(),
                 "spawn": lambda ag: ag.random_spawn()}
            if what in m:
                _jobs.put(_manual(lambda ag, f=m[what], w=what: (
                    f(ag), _log(f"randomize {w}"),
                    _honest_world(f"shuffled ({w})", "unknown" if w == "spawn" else "keep"))[1]))
            self._send({"ok": True})
        elif path == "/setparam":
            mv, rot = one("move"), one("rotate")
            def _sp(ag):
                if mv:
                    ag.move_magnitude = float(mv)
                if rot:
                    ag.rotate_degrees = float(rot)
            _jobs.put(_manual(_sp))
            self._send({"ok": True})
        elif path == "/batch":
            gens = {"open_all": lambda ag: ag.open_all(), "close_all": lambda ag: ag.close_all(),
                    "lights_on": lambda ag: ag.switch_all(True, "LightSwitch"),
                    "lights_off": lambda ag: ag.switch_all(False, "LightSwitch")}
            fn = gens.get(one("name"))
            if fn:
                _jobs.put(_manual(fn))
            self._send({"ok": True})
        elif path == "/checkpoint":
            _jobs.put(_manual(lambda ag: (ag.checkpoint(), _log("checkpoint saved"))[-1]))
            self._send({"ok": True})
        elif path == "/restore":
            _jobs.put(_manual(lambda ag: _honest_after(ag.restore_steps(), "restored a checkpoint", "unknown")))
            self._send({"ok": True})
        elif path == "/record":
            on = one("on") == "1"
            _jobs.put(lambda ag: (ag.start_recording() if on else ag.stop_recording(),
                                  _log("recording started" if on else "recording stopped"))[-1])
            self._send({"ok": True})
        elif path == "/save_gif":
            def _sg(ag):
                try:
                    os.makedirs("captures", exist_ok=True)
                    p = ag.save_gif(os.path.join("captures", f"episode_{time.strftime('%H%M%S')}.gif"))
                    _log(f"saved GIF: {os.path.basename(p)}")
                except Exception as e:
                    _log(f"save gif: {e}")
            _jobs.put(_sg)
            self._send({"ok": True})
        elif path == "/capture":
            _jobs.put(lambda ag: _do_capture(ag, allstreams=one("all") == "1"))
            self._send({"ok": True})
        elif path == "/export":
            os.makedirs("captures", exist_ok=True)
            fp = os.path.join("captures", f"session_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            with _lock, open(fp, "w", encoding="utf-8") as fh:
                fh.write("\n".join(_state["log"]))
            _log(f"exported {os.path.basename(fp)}")
            self._send({"ok": True, "file": os.path.basename(fp)})
        elif path == "/gallery":
            import glob
            files = []
            for ext in ("*.png", "*.gif", "*.jpg"):
                files += glob.glob(os.path.join(HERE, "captures", ext))
            self._send([os.path.basename(f) for f in sorted(files, key=os.path.getmtime, reverse=True)[:60]])
        elif path == "/cap_img":
            name = os.path.basename(one("name"))
            fp = os.path.join(HERE, "captures", name)
            if name.lower().endswith((".png", ".gif", ".jpg")) and os.path.isfile(fp):
                self._file(fp)
            else:
                self._send(b"404", "text/plain", 404)
        elif path == "/trace":
            # The trace viewer page. Its assets (/static/trace.css, /static/trace.js) are already served by
            # the /static/ branch above, so this alias is the ONLY static-serving change the viewer needs.
            self._file(os.path.join(WEB, "trace.html"))
        elif path in _TRACE_ROUTES:
            # Observability API: read-only, exception-safe, never touches the AI2-THOR worker. Each handler
            # returns its own {"error": ...} envelope (including "trace store missing" on a fresh checkout),
            # so a trace failure can never 500 the console or stall the 400 ms /status poll.
            self._json(_TRACE_ROUTES[path](one))
        else:
            self._send(b"404", "text/plain", 404)

    def _honest_gate(self, path, one) -> bool:
        """Honest-mode admission (spec 2.2 / 2.4). True = refused (the reply is sent). Oracle mode: never."""
        if not _honest_on():
            return False
        truth = (path in _HONEST_TRUTH_ROUTES or (path == "/click" and one("shift") == "1")
                 or (path == "/capture" and one("all") == "1") or (path == "/action" and one("name") == "drop")
                 or (path == "/view" and one("mode", "rgb") in ("instance", "semantic", "normals")))
        if truth:
            self._json({"ok": False, "honest": True,
                        "error": "this acts with simulator truth - switch the agent to Oracle to use it"}, 409)
            return True
        if path in _HONEST_IDLE_ROUTES and _honest_acting() and not (path == "/drive" and one("on", "1") == "0"):
            self._json({"ok": False, "honest": True, "owner": _LEASE.owner(),
                        "error": "the agent is acting - Stop first"}, 409)
            return True
        return False

    def _navmap_json(self):
        """Build navmap.NavMap(current scene).to_json(agent) on the worker (it owns the agent) and return it."""
        result = {"data": None}
        ev = threading.Event()

        def job(ag):
            try:
                import navmap
                nm = navmap.NavMap(_get_memory(), ag.scene)
                result["data"] = nm.to_json(ag)
            except Exception as exc:
                result["data"] = {"error": str(exc)}
            ev.set()
        _jobs.put(job)
        ev.wait(timeout=8)
        data = result["data"] or {"error": "timeout"}
        with _lock:
            data["pose"] = _state.get("pose", {})     # live agent marker for the canvas
            data["mode"] = _PERCEPTION["mode"]
        return data

    def _do_click(self, x, y, shift):
        """Resolve the object under a normalized (x,y) click; shift = pick it up."""
        result = {"hit": None}
        ev = threading.Event()

        def job(ag):
            oid = ag.object_in_frame_best(x, y)            # snaps to a small item on a counter, not the counter itself
            if oid:
                result["hit"] = {"id": oid, "type": oid.split("|")[0]}
                if shift:
                    o = ag.object_by_id(oid)
                    if o and o.get("pickupable"):
                        ag.pickup(oid, force=True)
                        if ag.success:
                            _record_demo_step({"action": "pickup", "object": oid.split("|")[0]})
                        _log(f"picked up {oid.split('|')[0]}: {'ok' if ag.success else ag.error[:30]}")
                else:
                    _log(f"clicked {oid.split('|')[0]}")
            ev.set()
        # GetObjectInFrame is a server-side raycast — it does NOT need instance-seg rendering (the Tkinter GUI
        # hit-tests the same way). We must NOT call set_sensors here: it does a controller.reset() that wipes
        # object states + agent pose, which on the first click would reset the scene and resolve the wrong object.
        _jobs.put(job)
        ev.wait(timeout=8)
        return result


def _llm_endpoint():
    try:
        cfg = json.load(open(os.path.join(HERE, "gui_config.json")))
    except Exception:
        cfg = {}
    return cfg.get("endpoint") or (al.gateway_endpoint() or "http://localhost:11434")


def _llm_models(ep):
    """The endpoint's model list, or None when nothing answers."""
    try:
        return al.make_provider(endpoint=ep).list_models(timeout=3)
    except Exception:
        return None


def _start_ollama():
    """Launch the Windows Ollama app from WSL (interop). False when there is no interop or no app."""
    import subprocess
    try:
        subprocess.Popen(["cmd.exe", "/c", "start", "", OLLAMA_APP], cwd="/mnt/c",
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        _log(f"could not start Ollama: {exc}")
        return False


def _set_llm(on, endpoint=""):
    """Flip the LLM switch. Turning it ON is only reported once the endpoint actually lists models."""
    if not on:
        _LLM.update(on=False, note="")
        _log("LLM off - planning disabled, manual driving still works")
        return {"ok": True, "on": False}
    ep = endpoint or _llm_endpoint()
    models = _llm_models(ep)
    started = False
    if models is None:
        _log(f"LLM: no answer at {ep}, starting Ollama on Windows...")
        started = _start_ollama()
        deadline = time.time() + 25
        while started and models is None and time.time() < deadline:
            time.sleep(1.5)
            models = _llm_models(ep)
    if models is None:
        note = f"Ollama did not answer at {ep}"
        _LLM.update(on=False, note=note)
        _log(f"LLM stays off: {note}")
        return {"ok": False, "on": False, "error": note, "note": note}
    _LLM.update(on=True, note="")
    _log(f"LLM on - {len(models)} models at {ep}" + (" (started Ollama)" if started else ""))
    return {"ok": True, "on": True, "models": models, "endpoint": ep, "started": started}


def main():
    import socket
    import sys
    if "--no-llm" in sys.argv[1:] or os.environ.get("WEBAPP_NO_LLM", "") in ("1", "true", "yes"):
        _LLM["on"] = False
        print("LLM OFF (--no-llm): nothing contacts Ollama until the LLM chip is switched on")
    if ta is None:                     # ai2thor absent: the console cannot drive anything, so say why, loudly
        raise RuntimeError("thor_actions could not be imported (AI2-THOR is WSL-only, see CLAUDE.md): "
                           f"{_TA_ERROR!r}")
    _install_trace_hooks()             # idempotent; also covers a reload where the module-level call was a no-op
    threading.Thread(target=_worker, daemon=True).start()
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "127.0.0.1"
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"web UI on  http://localhost:{PORT}   (or  http://{ip}:{PORT}  from Windows)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
