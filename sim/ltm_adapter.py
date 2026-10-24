"""Step 5 / B1 - the honest-agent adapter (docs/step5-plan.md sections 9, 13 "B0-plan", Astra A7).

Everything that connects the honest agent (step 4) to the long-term memory (step 5) lives here, and nothing here may
read what the agent may not read. Four pieces:

  HonestEventAdapter   turns the agent's OWN observations and actions into the closed event contract (`ltm_events`)
                       and feeds one WorkingMemory. Counters `tr: / vw: / ob:` are minted here; simulator ids never
                       enter. Positions are the agent's own: its depth + its own pose (objmem math). The place a
                       sighting sits in is the smallest structural receptacle box that contains the detection's box
                       centre (`in_basis = box_only`); a stronger basis needs observation evidence the core does not
                       claim. `step_outcome.under` is a structured code the skills expose, never parsed from text.
  ObjectMemoryProxy    wraps the real `objmem.ObjectMemory`: forwards everything, and after `update` / `on_pickup` /
                       `on_put` / `on_drop` tells the adapter what the (honest) report says.
  LtmChooser           `chooser(target, type_list, memory_summary) -> ordered place types` for `HonestSkills`: learned
                       T0 / T2 location claims first (channel filter applied BEFORE ranking), then exactly the prior
                       chooser's order. With empty memory its output IS the prior chooser's output. It also publishes
                       instance HINTS (claim xyz + the type's place-gate radius) for `HonestSkills(place_hints=...)`.
                       A hint is a navigation goal only; acting still needs a current sighting.
  FakeHonestAgent      a scripted agent over the REAL objmem (no simulator, no detector) for the offline selftests.

Imports: the standard library and the step-5 contract modules ONLY (the honesty gate scans this file like every ltm*.py).
Step-4 pieces are INJECTED by the executor: `geometry` (the `objmem` module: back-projection, depth lookup, receptacle set) and
`fallback` (the prior chooser). `honest`, `honest_skills`, `thor_actions` and `ai2thor` are never imported here. The static
guard in `_step5_adapter_selftest.py` also checks that no name in this file refers to the controller, `metadata`, `last_event`
or `state_readers`.
"""
from __future__ import annotations

import math
import re
from collections import Counter

try:
    import ltm_events as lev
except ImportError:                     # pragma: no cover
    from sim import ltm_events as lev

REFRESH_CAP = 3                         # re-sightings of a track emitted after its first one (a verified claim needs >= 2 views)
CELL = 0.25                             # m: receptacle dedupe cell (same-view boxes are always all emitted)
HINT_GATE_FACTOR = 2.0                  # a hint's radius = this x the type's place gate: a place centre is the ground point of a
                                        # detection box on a door (not a surveyed centre), and the plan item it should match is a
                                        # measured front; both carry that noise, so the match must be wider than the identity gate
CAUSES = ("not_detected", "wrong_target", "unreachable", "action_refused", "state_unrecognised", "retry_exhausted",
          "step_prim_cap", "episode_budget", "unknown", "no_cutting_tool")   # no_cutting_tool: A26 / K-B2b 2026-10-13
_TOK_BAD = re.compile(r"[^A-Za-z0-9_:.\-?/ ]")


class AdapterError(Exception):
    """An event the adapter built was refused by the contract (recorded in `adapter.rejected`)."""


def tokify(s, default="?") -> str:
    """ASCII token for a detector word such as 'closed, off' -> 'closed/off' (the contract refuses commas and non-ASCII)."""
    if not isinstance(s, str) or not s or "|" in s or not s.isascii():
        return default                    # a simulator-id-like or non-ASCII string is never repaired into a token
    return _TOK_BAD.sub("/", s.replace(", ", "/")).strip()[:96] or default


REASON_HEX = "u8x"                       # P2-FIX 2026-10-14: marks a hex-encoded (UTF-8) non-ASCII parser reason
REASON_SHA = "u8h"                       # ... or, past the 96-character token limit, a sha256 prefix of it


def reason_token(r: str) -> str:
    """One ltm_concepts reason -> an event-safe ASCII token (P2-FIX, Astra ACCEPT-V2-A1-CAPTURE Q3).

    An ASCII reason is returned UNCHANGED (byte-identical). A non-ASCII one (the FA complex-clause markers, e.g.
    'complex:' + a Persian connective) keeps its ASCII head up to the first ':' and replaces the rest by 'u8x' + the
    lower-case hex of its UTF-8 bytes: deterministic, injective (distinct reasons stay distinct) and made of [0-9a-f]
    only, so it passes ltm_events._tok (no backslash, no '|', no simulator-id shape). If that would exceed the
    96-character token limit, the rest becomes 'u8h' + the first 32 hex of sha256 over the WHOLE reason's UTF-8 (and, if
    the ASCII head alone is too long, the whole token is 'u8h' + that sha prefix)."""
    if not isinstance(r, str):
        raise AdapterError("task_context.concepts.reasons: a parser reason is not a string")
    if r.isascii():
        return r
    head, sep, rest = r.partition(":")
    if not (sep and head and head.isascii() and re.fullmatch(r"[A-Za-z0-9_]+", head)):
        head, sep, rest = "", "", r
    out = head + sep + REASON_HEX + rest.encode("utf-8").hex()
    if len(out) > 96:
        import hashlib
        out = head + sep + REASON_SHA + hashlib.sha256(r.encode("utf-8")).hexdigest()[:32]
        if len(out) > 96:                                # an over-long ASCII head: the whole reason by its sha prefix
            out = REASON_SHA + hashlib.sha256(r.encode("utf-8")).hexdigest()[:32]
    return out


def structural_place_types(geometry) -> frozenset:
    """Receptacle types that are PLACES (fixed furniture): the geometry module's receptacle set minus the movable ontology and Floor."""
    try:
        from ltm_consolidate import load_useful_types
    except ImportError:                  # pragma: no cover
        from sim.ltm_consolidate import load_useful_types
    return frozenset(geometry.RECEPTACLE_TYPES) - load_useful_types() - {"Floor"}


class HonestEventAdapter:
    def __init__(self, ltm, wm=None, *, episode, mode="practice", world="reset", epoch=None, frame=None, map_rev="m0",
                 kitchen=None, detector_rev="unknown", fov=90.0, clock=None, hint_source=None, place_types=None, geometry=None):
        self.ltm, self.kitchen = ltm, kitchen or (ltm.kitchen if ltm is not None else None)
        if not self.kitchen:
            raise AdapterError("an adapter without a long-term memory needs an explicit kitchen label")
        self.episode, self.mode, self.world = episode, mode, world
        # ltm=None is the `none` arm: the SAME validated event stream and metrics, into a throw-away working memory
        self.epoch = (ltm.current_epoch() if ltm is not None else 0) if epoch is None else int(epoch)
        self.frame = frame or (ltm.frame_of_epoch(self.epoch) if ltm is not None else None) or "f0"
        self.map_rev, self.detector_rev, self.fov = map_rev, detector_rev, float(fov)
        self.clock = clock or (lambda: 0)
        self.hint_source = hint_source                  # callable(otype) -> [hint] (the LtmChooser's) for the metrics
        self.geo = geometry                              # the injected objmem module (observations need it)
        self.place_types = (frozenset(place_types) if place_types is not None
                            else structural_place_types(geometry) if geometry is not None else frozenset())
        if wm is None:
            if ltm is not None:
                wm = ltm.begin_episode(episode, mode, world)
            else:
                try:
                    from ltm_wm import WorkingMemory
                except ImportError:      # pragma: no cover
                    from sim.ltm_wm import WorkingMemory
                wm = WorkingMemory(episode)
        self.wm = wm
        self.walk_m = 0.0
        self._last_pos = None
        self._n = 0
        self.stream: list = []                          # the validated events, in order
        self.rejected: list = []
        self.errors: list = []                          # callback failures other than contract refusals (fail fast)
        self._tr: dict = {}                             # objmem track id -> minted tr:n
        self._otype_of: dict = {}
        self._nv = 0
        self._emitted: Counter = Counter()
        self._rcell: dict = {}
        self._open_evs: list = []                       # (ev, rtype, xyz) of own opens
        self._cur_search = None
        self.searches: list = []                        # B-h per-search metrics
        self.inspections: list = []
        self.n_opens = 0
        self.started = False
        self.ended = False

    # -- the one door into the working memory ---------------------------------------------------------------------
    def _emit(self, kind, **f):
        self._n += 1
        ev = {"schema": lev.SCHEMA, "ev": self._n, "kind": kind, "kitchen": self.kitchen, "epoch": self.epoch,
              "frame": self.frame, "map_rev": self.map_rev, "episode": self.episode, "t": max(0, int(self.clock() or 0))}
        ev.update({k: v for k, v in f.items() if v is not None})
        try:
            out = self.wm.ingest(ev)
        except lev.EventError as e:
            self.rejected.append({"ev": self._n, "kind": kind, "error": str(e)[:160]})
            raise AdapterError(f"{kind}: {e}") from e
        self.stream.append(out)
        return out

    def _track(self, real_tid, otype=None) -> str:
        if real_tid not in self._tr:
            self._tr[real_tid] = f"tr:{len(self._tr) + 1}"
        if otype:
            self._otype_of[self._tr[real_tid]] = otype
        return self._tr[real_tid]

    # -- task / step / episode hooks --------------------------------------------------------------------------------
    def on_task(self, text, level, lang):
        """episode_start + task_context. Concepts are parsed from the command text, never from a reference plan."""
        try:
            import ltm_concepts as lc
        except ImportError:              # pragma: no cover
            from sim import ltm_concepts as lc
        parsed = lc.parse(text)
        # P2-FIX: FA complex-clause reasons are non-ASCII; the event contract admits ASCII tokens only -> reason_token
        concepts = {"roles": parsed["roles"], "ok": bool(parsed["ok"]), "reasons": [reason_token(r) for r in parsed["reasons"]],
                    "parser": parsed["parser"], "lexicon": parsed["lexicon"]}
        self._emit("episode_start", mode=self.mode, world=self.world)
        self._emit("task_context", level=level, lang=lang, text=text, concepts=concepts)
        self.started = True

    def on_step_start(self, i, step, pre=None):
        self._emit("step_start", i=int(i), verb=tokify(step.get("action"), "?"), otype=step.get("object") and tokify(step.get("object")),
                   to=step.get("to") and tokify(step.get("to")), pre={k: tokify(v) if isinstance(v, str) else v
                                                                       for k, v in (pre or {}).items()
                                                                       if isinstance(v, (str, bool, int, float))})

    def on_step_end(self, i, result):
        """From the skills' own result: ok / cause / retries / prims / the structured `under` code. `why`, `events`, `verify`
        and `search` (free text) are never read."""
        d = result.get("detail") or {}
        cause = result.get("cause")
        under = d.get("under")
        self._emit("step_outcome", i=int(i), ok=bool(result.get("ok")),
                   cause=cause if cause in CAUSES else (None if cause is None else "unknown"),
                   under={"code": under if under in CAUSES else "unknown"} if cause else None,
                   retries=int(result.get("retries") or 0), prims=int(d.get("prims") or 0))

    def on_action(self, verb, ok, otype=None, rtype=None, i=None, k=None, cause=None):
        """A refused action as a bool plus an ENUMERATED cause; error text is not an argument and is never kept."""
        if not ok:
            self._emit("refusal", verb=tokify(verb), otype=otype and tokify(otype), rtype=rtype and tokify(rtype),
                       i=i, k=k, cause=cause if cause in CAUSES else "unknown")

    def on_terminal(self, exec_ok, prims, opens, walk_m=0.0, invalid=False):
        self._emit("terminal", exec_ok=bool(exec_ok), agent_verdict="success" if exec_ok else "failure", prims=int(prims),
                   opens=int(opens), walk_m=round(float(walk_m), 3), invalid=bool(invalid))

    def on_admission(self, goal_ok):
        """Practice only, after `terminal` (the working memory refuses it otherwise)."""
        self._emit("admission_verdict", goal_ok=bool(goal_ok), tag="privileged-terminal-goal-check")

    def end(self):
        if not self.ended:
            self._emit("episode_end")
            self.ended = True

    # -- observations -------------------------------------------------------------------------------------------
    def _ground(self, d, depth, pose, wh):
        om = self.geo
        if not (d.get("point") and d.get("point_ok")) or wh is None:
            return None
        dd = om._depth_at(depth, d["point"][0], d["point"][1])
        return om.backproject(d["point"][0], d["point"][1], dd, pose, wh, self.fov) if dd else None

    def container_of(self, d, rec_items):
        """The smallest OTHER structural receptacle box that contains the centre of d's box -> (rtype, xyz) or None.
        `rec_items` = [(det, xyz)] of this view's grounded structural receptacles."""
        b = d.get("bbox_xyxy")
        if not b:
            return None
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        best, area = None, None
        for o, xyz in rec_items:
            ob = o.get("bbox_xyxy")
            if o is d or not ob or o.get("type") == d.get("type"):
                continue
            if ob[0] <= cx <= ob[2] and ob[1] <= cy <= ob[3]:
                a = (ob[2] - ob[0]) * (ob[3] - ob[1])
                if area is None or a < area:
                    best, area = (o["type"], xyz), a
        return best

    def on_observation(self, obs, rep, tracks):
        """One honest view. `obs` is read only for step / depth / pose / dets (type, state, conf, bbox, point, track);
        `rep` is objmem's update report; `tracks` its track table (own measurements)."""
        if rep.get("duplicate"):
            return
        self._nv += 1
        view, ob = f"vw:{self._nv}", f"ob:{self._nv}"
        depth, pose = obs.depth, obs.pose
        pp = (pose.get("position") or {}) if isinstance(pose, dict) else {}
        if pp.get("x") is not None and pp.get("z") is not None:       # own pose: straight-line path between views
            cur = (float(pp["x"]), float(pp["z"]))
            if self._last_pos is not None:
                self.walk_m += math.dist(self._last_pos, cur)
            self._last_pos = cur
        if self.geo is None:
            raise AdapterError("observations need the injected geometry module (objmem)")
        om = self.geo
        wh = om._wh(depth)
        dets = [d for d in obs.dets if d.get("type")]
        rec_items = []
        for d in dets:
            if d["type"] in self.place_types:
                xyz = self._ground(d, depth, pose, wh)
                if xyz is not None:
                    rec_items.append((d, xyz))
        same = Counter(d["type"] for d, _ in rec_items)
        for d, xyz in rec_items:
            cell = (d["type"], round(xyz[0] / CELL), round(xyz[2] / CELL))
            views = self._rcell.setdefault(cell, set())
            if len(views) >= 2 and view not in views:
                continue                                 # a box seen in two other views already: not again
            views.add(view)
            self._emit("receptacle_seen", rtype=d["type"], xyz=xyz, view=view, obs=ob, n_boxes_same_type=int(same[d["type"]]))
        new, sup, ref = set(rep.get("new") or []), set(rep.get("superseded") or []), set(rep.get("refreshed") or [])
        for d in dets:
            tid = d.get("track")
            t = tracks.get(tid) if tid else None
            if t is None or t.xyz is None or d["type"] in self.place_types:
                continue
            if not (tid in new or tid in sup or (tid in ref and self._emitted[tid] < 1 + REFRESH_CAP)):
                continue
            self._emitted[tid] += 1
            cont = self.container_of(d, rec_items)
            self._emit("sighting", otype=d["type"], state=tokify(d.get("state")), conf=round(float(d.get("conf") or 0.0), 4),
                       detector=tokify(self.detector_rev), track=self._track(tid, d["type"]), xyz=tuple(t.xyz), xyz_ok=True,
                       view=view, obs=ob, in_rtype=cont[0] if cont else None, in_basis="box_only" if cont else "none",
                       in_xyz=tuple(cont[1]) if cont else None)
        for tid in rep.get("contradicted") or []:
            t = tracks.get(tid)
            if t is not None and t.type not in self.place_types:
                self._emit("contradiction", track=self._track(tid, t.type), otype=t.type, views=len(t.misses), src=[ob])

    # -- own actions (through the object-memory proxy) ------------------------------------------------------------------
    def on_pickup(self, ok, real_tid, otype):
        if ok and real_tid:
            self._emit("pickup", track=self._track(real_tid, otype), otype=otype and tokify(otype), ok=True)

    def on_put(self, ok, real_tid, otype, rtype, xyz):
        if ok and real_tid and xyz is not None:
            self._emit("put", track=self._track(real_tid, otype), otype=tokify(otype or "?"), rtype=tokify(rtype or "?"),
                       xyz=tuple(xyz), ok=True, resighted=False)

    def on_drop(self, ok, real_tid, otype, xyz):
        if ok and real_tid and xyz is not None:
            self._emit("drop", track=self._track(real_tid, otype), otype=tokify(otype or "?"), xyz=tuple(xyz), ok=True,
                       resighted=False)

    # -- the skills' structured events (HonestSkills.on_event) ----------------------------------------------------------
    def on_event(self, kind, f):
        if kind in ("open", "close"):
            ev = self._emit(kind, rtype=tokify(f["rtype"]), xyz=tuple(f["xyz"]), ok=bool(f.get("ok")), by_me=bool(f.get("by_me", True)),
                            state_seen=f.get("state_seen") if f.get("state_seen") in ("open", "closed", "?") else "?")
            if kind == "open":
                self.n_opens += 1
                self._open_evs.append((ev["ev"], ev["rtype"], tuple(ev["xyz"])))
        elif kind == "inspection":
            self._on_inspection(f)
        elif kind == "search_start":
            self._cur_search = {"otype": f["otype"], "t0": int(f.get("prims") or 0),
                                "hints": [dict(h) for h in (self.hint_source(f["otype"]) if self.hint_source else [])],
                                "visits": []}
        elif kind == "search_end":
            self._on_search_end(f)

    def _open_for(self, rtype, xyz):
        best = None
        for ev, rt, x in reversed(self._open_evs):
            if rt == rtype and math.dist(x, xyz) <= 0.6:
                best = ev
                break
        return best

    def _on_inspection(self, f):
        xyz, rtype = tuple(f["xyz"]), tokify(f["rtype"])
        verdict = f.get("verdict") if f.get("verdict") in ("found", "visible", "partly", "not-visible", "unsure") else "unsure"
        fields = dict(rtype=rtype, xyz=xyz, open_ev=self._open_for(rtype, xyz), verdict=verdict, frac=f.get("frac"),
                      thirds=f.get("thirds"), support=f.get("support"), range_m=f.get("range_m"), views=f.get("views"),
                      crouched=f.get("crouched"), detected=[tokify(t) for t in f.get("detected") or []],
                      targets=[tokify(t) for t in f.get("targets") or []], detector_rev=tokify(self.detector_rev))
        self._emit("inspection", **fields)
        cs = self._cur_search
        hints = [dict(h) for h in (self.hint_source(cs["otype"]) if (self.hint_source and cs) else [])]
        if cs is not None:
            cs["hints"] = hints                               # the chooser ran after search_start: take the hints as of now
        ranked = any(h.get("rtype") == rtype and math.dist(tuple(h["xyz"]), xyz) <= float(h.get("radius") or 0.75) for h in hints)
        row = {"rtype": rtype, "xyz": [round(float(x), 3) for x in xyz], "verdict": verdict, "memory_ranked": ranked,
               "stale": bool(ranked and verdict != "found")}
        self.inspections.append(row)
        if self._cur_search is not None:
            self._cur_search["visits"].append(row)

    def _on_search_end(self, f):
        s = self._cur_search or {"otype": f.get("otype"), "t0": 0, "hints": [], "visits": []}
        found = bool(f.get("found"))
        end = int(f.get("prims") or 0)
        first = f.get("first_seen")
        cap_abs = f.get("cap")
        if found and first is not None:                  # primitives from the search start to the FIRST detection of the type
            steps = int(first) - int(s["t0"])
        elif f.get("budget") and cap_abs is not None:    # censored at the cap the search actually stopped at
            steps = int(cap_abs) - int(s["t0"])
        else:                                            # censored where the search ended (everything reachable was checked)
            steps = end - int(s["t0"])
        v = s["visits"]
        self.searches.append({"otype": s["otype"], "found": found, "censored": not found, "steps_to_find": steps,
                              "search_prims": end - int(s["t0"]), "cap": (int(cap_abs) - int(s["t0"])) if cap_abs is not None else None,
                              "first_seen": first, "t0": int(s["t0"]),
                              "inspected": [(x["rtype"], x["xyz"]) for x in v], "visit_rows": [dict(x) for x in v], "visited_types": list(f.get("visited") or []),
                              "budget": bool(f.get("budget")), "n_hints": len(s["hints"]),
                              "visits": len(v), "memory_ranked_visits": sum(1 for x in v if x["memory_ranked"]),
                              "other_visits": sum(1 for x in v if not x["memory_ranked"]),
                              "stale_visits": sum(1 for x in v if x["stale"]),
                              "unreachable": list(f.get("unreachable") or []), "opened": len(f.get("opened") or []),
                              "viewpoints": int(f.get("viewpoints") or 0), "hinted": bool(f.get("hinted"))})
        self._cur_search = None

    def metrics(self) -> dict:
        s = self.searches
        return {"walk_m": round(self.walk_m, 3), "n_searches": len(s), "found": sum(1 for x in s if x["found"]), "censored": sum(1 for x in s if x["censored"]),
                "steps_to_find": sum(x["steps_to_find"] for x in s), "stale_visits": sum(x["stale_visits"] for x in s),
                "memory_ranked_visits": sum(x["memory_ranked_visits"] for x in s),
                "other_visits": sum(x["other_visits"] for x in s), "opens": self.n_opens, "searches": s,
                "events": len(self.stream), "rejected": len(self.rejected)}


class ObjectMemoryProxy:
    """Wraps an `objmem.ObjectMemory`. Everything the skills use is forwarded; the four mutating entry points also
    tell the bound adapter. An adapter failure never reaches the agent: it is recorded (`adapter.rejected`) and the
    executor aborts the episode as invalid (fail fast)."""

    def __init__(self, real, adapter=None):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "adapter", adapter)

    def bind(self, adapter):
        object.__setattr__(self, "adapter", adapter)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _tell(self, fn, *a):
        if self.adapter is None:
            return
        try:
            fn(*a)
        except AdapterError:
            pass                                          # recorded in adapter.rejected
        except Exception as e:                            # noqa: BLE001 - never into the agent; the executor aborts the episode
            self.adapter.errors.append({"hook": getattr(fn, "__name__", "?"), "error": f"{type(e).__name__}: {str(e)[:120]}"})

    def update(self, obs, holding=None):
        rep = self._real.update(obs, holding)
        if self.adapter is not None:
            self._tell(self.adapter.on_observation, obs, rep, self._real.tracks)
        return rep

    def on_pickup(self, ok, track=None, otype=None, step=0):
        tid = self._real.on_pickup(ok, track, otype, step)
        if tid is not None and self.adapter is not None:
            self._tell(self.adapter.on_pickup, ok, tid, otype or self._real.tracks[tid].type)
        return tid

    def on_put(self, ok, place_type=None, place_xyz=None, otype=None, step=0, why=""):
        held = self._real.held
        ot = self._real.tracks[held].type if held and held in self._real.tracks else otype
        tid = self._real.on_put(ok, place_type, place_xyz, otype, step, why)
        if tid is not None and self.adapter is not None:
            self._tell(self.adapter.on_put, ok, tid, ot or self._real.tracks[tid].type, place_type, place_xyz)
        return tid

    def on_drop(self, ok, pose=None, step=0):
        held = self._real.held
        ot = self._real.tracks[held].type if held and held in self._real.tracks else None
        tid = self._real.on_drop(ok, pose, step)
        if tid is not None and self.adapter is not None:
            p = (pose or {}).get("position")
            xyz = (float(p["x"]), float(p["y"]), float(p["z"])) if p else None
            self._tell(self.adapter.on_drop, ok, tid, ot or self._real.tracks[tid].type, xyz)
        return tid


class LtmChooser:
    """Search chooser over the long-term memory. `channels` is the arm's feature set (`places`); `fallback` is the prior
    chooser (a callable). Learned place types come first, in tier order, de-duplicated and restricted to the detector's
    type list; the rest is the fallback's order unchanged. Empty memory -> the fallback's list, exactly."""

    def __init__(self, ltm, wm, channels=("places",), fallback=None):
        if not callable(fallback):
            raise AdapterError("LtmChooser needs the prior chooser as `fallback` (injected by the executor)")
        self.ltm, self.wm, self.channels = ltm, wm, frozenset(channels)
        self.fallback = fallback
        self.label = "ltm(" + "+".join(sorted(self.channels)) + ") then prior"
        self.decisions: list = []
        self.errors: list = []                       # a failure here must INVALIDATE the episode (the search would silently fall back)
        self._hints: dict = {}

    def _fallback(self):
        return self.fallback

    def _radius(self, rtype) -> float:
        g = self.ltm.params["identity"]["place_gate_m"]["value"]
        return HINT_GATE_FACTOR * float(g.get(rtype, g["default"]))

    def hints_for(self, otype) -> list:
        return [dict(h) for h in self._hints.get(otype, [])]

    def __call__(self, target, type_list, memory_summary):
        try:
            return self._choose(target, type_list, memory_summary)
        except Exception as e:                       # noqa: BLE001 - recorded, then re-raised; the executor aborts the episode
            self.errors.append({"target": target, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            raise

    def _choose(self, target, type_list, memory_summary):
        base = list(self._fallback()(target, type_list, memory_summary))
        self._hints[target] = []
        if not self.channels:
            return base
        ans = self.ltm.where_is(target, self.wm, channels=self.channels)
        frame_now = self.ltm.frame_of_epoch(ans["epoch"])
        learned, hints, seen = [], [], set()
        for c in ans["candidates"]:
            if c["kind"] != "place" or c["tier"] not in ("T0", "T2") or not c.get("rtype"):
                continue
            row = self.ltm.store.select("place_instance", where={"place": c["id"]})
            if row and (frame_now is None or row[0]["frame"] == frame_now):    # coordinates are only comparable inside one frame
                hints.append({"place": c["id"], "rtype": c["rtype"], "tier": c["tier"], "weight": c["weight"],
                              "xyz": [row[0]["cx"], row[0]["cy"], row[0]["cz"]], "radius": self._radius(c["rtype"])})
            if c["rtype"] in type_list and c["rtype"] != target and c["rtype"] not in seen:
                seen.add(c["rtype"])
                learned.append(c["rtype"])
        order = learned + [t for t in base if t not in seen]
        self._hints[target] = hints
        self.decisions.append({"target": target, "epoch": ans["epoch"], "learned": [(c["id"], c["tier"], c["rtype"]) for c in ans["candidates"]
                                                                             if c["kind"] == "place" and c["tier"] in ("T0", "T2")],
                               "order": order[:8], "prior_order": base[:8], "n_hints": len(hints), "changed": order != base})
        return order


# ====================================================================================================== test fake
class FakeObs:
    """The honest Obs shape (`step, rgb, depth, pose, dets, holding`) plus optional JUNK attributes a real simulator
    handle could carry (metadata, event, objectId ...). The adapter must ignore every one of them."""

    def __init__(self, step, depth, pose, dets, holding=None, **junk):
        self.step, self.rgb, self.depth, self.pose, self.dets, self.holding = step, None, depth, pose, dets, holding
        for k, v in junk.items():
            setattr(self, k, v)


class FakeHonestAgent:
    """A scripted agent over the REAL `objmem.ObjectMemory` (the module is injected as `geometry`): no simulator, no detector. `look()` builds a depth image with
    one plane per detection, runs `om.update` through the proxy (so the adapter sees exactly what a live run shows it)
    and returns the report. Used by the offline selftests and by nothing else."""

    def __init__(self, adapter, geometry, size=(96, 96), fov=90.0, scene="Fake", episode="fake"):
        self.size, self.fov = size, float(fov)
        self.om = ObjectMemoryProxy(geometry.ObjectMemory(scene, store=None, episode=episode, fov=fov), adapter)
        self.adapter, self.step = adapter, 0
        adapter.clock = lambda: self.step
        self.pose = {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"x": 0, "y": 0.0, "z": 0},
                     "horizon": 0.0, "standing": True, "camera": {"x": 0.0, "y": 1.5, "z": 0.0}}

    def look(self, dets, step=None, **junk):
        """dets: [{type, box:[x0,y0,x1,y1], depth, conf?, state?, ...junk}] -> the update report."""
        w, h = self.size
        depth = [[6.0] * w for _ in range(h)]
        out = []
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            for y in range(max(0, y0), min(h, y1 + 1)):
                for x in range(max(0, x0), min(w, x1 + 1)):
                    depth[y][x] = min(depth[y][x], float(d["depth"]))
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            det = {k: v for k, v in d.items() if k not in ("box", "depth")}
            det.update(bbox_xyxy=(x0, y0, x1, y1), point=((x0 + x1) // 2, (y0 + y1) // 2), point_ok=True,
                       conf=d.get("conf", 0.9), state=d.get("state", "?"))
            out.append(det)
        self.step += 1 if step is None else 0
        if step is not None:
            self.step = step
        obs = FakeObs(self.step, depth, self.pose, out, **junk)
        return self.om.update(obs)
