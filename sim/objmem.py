"""Step 4 (s3-memory) -- the honest agent's simple object memory (Arash decision 3). Pure: no simulator access.

Every detection the agent makes becomes a *sighting*: object type, state (may be ``?``), step + time, confidence,
the screen box and point, the agent's pose when it looked, and -- when depth is valid at the point -- a world xyz
back-projected through a pinhole camera from the agent's own pose. Sightings of the same type within 0.4 m are one
*track*; a track's current belief is its newest sighting. Inputs are only ``honest.Obs`` observations and the
outcomes of the agent's own actions, so nothing here can leak simulator truth.

Trust rules (docs/step4-plan.md, s3-memory; Astra's review folded in):

* A sighting is **trusted** until contradicted, or until the agent itself moves the object.
* Re-seeing a track with a different state (or > 0.1 m away) **supersedes** the old row. An unchanged re-sighting
  only refreshes the in-RAM belief (counted, not stored) so the database is evidence, not a frame log.
* **Contradiction** needs 2 later views, from distinct observations AND distinct viewpoints, in which the track's
  xyz projects inside the frame, lies within 2.5 m, is not occluded (depth there >= expected - 0.3 m), and no
  detection of the type is nearby. It contradicts the *location* belief, not the object's existence. A match resets
  the miss count; an ineligible view counts nothing.
* Pickup (successful) -> the bound track becomes **held** (no navigation xyz). If the pickup was not bound to a
  track, a new held track of the aimed type is made (identity unknown) rather than guessing one.
* Put (successful) -> the held track becomes **placed**: an anchor at the receptacle's sighted xyz, state ``?``,
  until seen again. Placed tracks are never contradicted by misses (they may be inside a closed receptacle).
* Lookup order: trusted, then placed, then held. Contradicted tracks are not offered.
* ``places_seen`` is an append-only log (type, place type = the smallest detected receptacle box containing the
  detection's centre, else ``?``), written for later RAG and not read in step 4.

Persistence: ``store`` is a ``memory.Memory`` (tables ``sightings`` / ``places_seen``; every write audited). With
``store=None`` the memory lives in RAM only (tests). ``new_episode`` clears the RAM beliefs (the world was reset);
``ObjectMemory.resume`` rebuilds one episode's beliefs from the store.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

ASSOC_GATE = 0.40          # m: same type within this distance = same track
PLACED_GATE = 1.00         # m: a placed track's anchor is the receptacle point, so the object may be farther from it
AMBIG_EPS = 0.10           # m: best and second-best track nearer than this apart = ambiguous
#: F-dup (E3 diagnosis, 2026-10-13): two boxes of the SAME type in ONE view whose grounded points are within AMBIG_EPS and whose
#: boxes overlap (IoU >= DUP_IOU) are one object seen twice by the detector (a 0.48 + 0.27 + 0.27 Spoon triple in the FloorPlan1
#: drawer made two co-located tracks; every later 0.90 sighting was then "ambiguous" and never associated). The second box joins
#: the first one's track instead of opening a track of its own. False = the old behaviour.
#: Priority (Astra E3 review): an EXISTING association wins (a box already matched to a track keeps it, whatever its
#: confidence); among unmatched boxes the highest confidence leads. A box whose duplicates are matched to DIFFERENT tracks is
#: left ambiguous (kept as evidence, not associated). Limits: overlap + proximity is a duplicate HYPOTHESIS, not proof of one
#: object -- two real forks touching (IoU >= 0.30, points < DUP_DIST apart) are merged into one track; DUP_DIST / DUP_IOU are
#: development settings, not validated optima. Existing co-located tracks are never merged.
DEDUP_FRAME = True
DUP_IOU = 0.30
DUP_DIST = 0.10                 # m: duplicate-detection tolerance (frozen separately from the association's AMBIG_EPS)
MOVED_EPS = 0.10           # m: a re-sighting farther than this from the belief writes a new row
MISS_RANGE = 2.5           # m: a miss counts only if the belief is within this distance of the camera
OCCLUSION_TOL = 0.30       # m: depth at the pixel must be >= expected - this, else something is in front
MISSES_TO_CONTRADICT = 2
EDGE_MARGIN = 8            # px: a projection this close to the border does not count
NEAR_DET_PX = 10           # px: a same-type box this close to the projected pixel blocks a miss
VIEW_MOVE, VIEW_YAW, VIEW_PITCH = 0.20, 20.0, 15.0   # distinct viewpoint: moved, turned or tilted at least this much
HELD_NEAR = 0.80           # m: a detection of the held type closer than this is the object in the agent's hand
FOV_DEFAULT = 90.0         # deg, vertical FOV of the robot's camera (its calibration, not a simulator read)

STATUSES = ("trusted", "contradicted", "superseded", "held", "placed")
RECEPTACLE_TYPES = frozenset({
    "CounterTop", "DiningTable", "SideTable", "Sink", "SinkBasin", "Fridge", "Cabinet", "Drawer", "Microwave",
    "StoveBurner", "Shelf", "ShelvingUnit", "Stool", "Chair", "GarbageCan", "CoffeeMachine", "Toaster", "Pan", "Pot",
    "Plate", "Bowl", "Mug", "Cup", "Floor"})


# ------------------------------------------------------------------------------------------------ geometry
def _basis(pose: dict):
    yaw = math.radians(float(pose["rotation"]["y"]))
    pitch = math.radians(float(pose.get("horizon") or 0.0))
    sy, cy, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    fwd = (sy * cp, -sp, cy * cp)        # horizon > 0 looks down
    up = (sy * sp, cp, cy * sp)
    right = (cy, 0.0, -sy)
    cam = pose.get("camera")
    if cam:
        eye = (float(cam["x"]), float(cam["y"]), float(cam["z"]))
    else:
        p = pose["position"]
        eye = (float(p["x"]), float(p["y"]) + (0.675 if pose.get("standing", True) else 0.40), float(p["z"]))
    return eye, right, up, fwd


def _focal(h: int, fov: float) -> float:
    return (h / 2.0) / math.tan(math.radians(fov) / 2.0)


def backproject(px, py, d, pose: dict, wh, fov: float = FOV_DEFAULT):
    """World ``(x, y, z)`` of pixel ``(px, py)`` at axial depth ``d`` (m), from the agent's own pose. None if no depth."""
    if d is None or not math.isfinite(d) or d <= 0:
        return None
    w, h = wh
    f = _focal(h, fov)
    xc, yc = ((px + 0.5) - w / 2.0) / f * d, -((py + 0.5) - h / 2.0) / f * d
    eye, r, u, fw = _basis(pose)
    return tuple(eye[i] + xc * r[i] + yc * u[i] + d * fw[i] for i in range(3))


def project(xyz, pose: dict, wh, fov: float = FOV_DEFAULT):
    """Pixel ``(px, py)`` and axial depth of world ``xyz`` in the agent's camera, or None if behind the camera."""
    eye, r, u, fw = _basis(pose)
    dv = [xyz[i] - eye[i] for i in range(3)]
    zc = sum(dv[i] * fw[i] for i in range(3))
    if zc <= 1e-6:
        return None
    w, h = wh
    f = _focal(h, fov)
    xc, yc = sum(dv[i] * r[i] for i in range(3)), sum(dv[i] * u[i] for i in range(3))
    return (xc / zc * f + w / 2.0 - 0.5, -yc / zc * f + h / 2.0 - 0.5, zc)


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _depth_at(depth, px, py):
    if depth is None:
        return None
    try:
        v = float(depth[int(round(py))][int(round(px))])
    except (IndexError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


# SPEC-INHAND (2026-10-13; Astra SPEC-INHAND Q2/Q4): the SHARED "is this box the object in the hand?" depth predicate. The old test
# read the depth at a box's ground point (a pixel agreeing with the box MEDIAN depth); a held utensil / shaker fills 5-60 % of its
# own box, so that point is the background 1-2 m behind it and the held object was never recognised as held. Declared development
# constants (chosen on the specialist's amb TRAIN + VAL held frames only; proposals, not validated optima):
HAND_DEPTH_Q = 2               # the band depth of a box = this percentile of its own valid depth
HAND_SUPPORT_MIN_PX = 16       # a box is at the hand only if a CONNECTED patch of valid depth < HELD_NEAR inside it has at least
HAND_SUPPORT_SHARE = 0.02      # max(HAND_SUPPORT_MIN_PX, ceil(HAND_SUPPORT_SHARE x clipped box area)) pixels ...
HAND_FIT_AXIS = 0.0            # optional: the patch must span this share of the box's width AND height (0 = off). Measured on the
                               # FUSED path (offline_replay_v2.json, 400 amb train + val held frames): fit 0 / .3 / .5 -> ok 387 / 379 /
                               # 349 (fused top-label 386), false 'wrong' 6 / 5 / 6 (ground point 8), but .5 drops the glass SaltShaker
                               # to 10 / 40 (its depth sees through the body; only the cap is near). Off by default (Astra's rule).


def _largest_component(mask):
    """(pixel count, (x1, y1, x2, y2)) of the largest 4-connected True region of a 2-D bool array; (0, None) if none."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0, None
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        if n == 0:
            return 0, None
        cnt = np.bincount(lab.ravel())
        cnt[0] = 0
        k = int(cnt.argmax())
        ys, xs = np.nonzero(lab == k)
        return int(cnt[k]), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    except ImportError:                               # pure fallback (small boxes only in practice)
        h, w = m.shape
        seen = np.zeros_like(m)
        best = (0, None)
        for y0, x0 in zip(*np.nonzero(m)):
            if seen[y0, x0]:
                continue
            stack, pts = [(y0, x0)], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            if len(pts) > best[0]:
                ys_ = [p[0] for p in pts]
                xs_ = [p[1] for p in pts]
                best = (len(pts), (min(xs_), min(ys_), max(xs_), max(ys_)))
        return best


def hand_depth(depth, bbox, near: float = None, q: float = None, fit: float = None):
    """The NEAR depth of a box when -- and only when -- it is SUPPORTED as the object in the hand (own depth only; pure):
    a valid, in-frame box; a connected patch of valid depth < ``near`` (HELD_NEAR) inside it of at least
    max(HAND_SUPPORT_MIN_PX, ceil(HAND_SUPPORT_SHARE x clipped area)) pixels that spans at least ``fit`` (HAND_FIT_AXIS)
    of the clipped box's width and of its height. Returns the ``q``-th percentile (HAND_DEPTH_Q) of the box's valid depth,
    else None."""
    import numpy as np
    near = HELD_NEAR if near is None else float(near)
    q = HAND_DEPTH_Q if q is None else float(q)
    fit = HAND_FIT_AXIS if fit is None else float(fit)
    if depth is None or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None                                   # malformed box: rejected BEFORE clipping
    d = np.asarray(depth, dtype="float32")
    if d.ndim != 2:
        return None
    h, w = d.shape
    if x2 < 0 or y2 < 0 or x1 > w - 1 or y1 > h - 1:
        return None                                   # entirely outside the frame
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
    if ix2 < ix1 or iy2 < iy1:
        return None
    win = d[iy1:iy2 + 1, ix1:ix2 + 1]
    valid = np.isfinite(win) & (win > 0)
    if not valid.any():
        return None
    area = int(win.size)
    need = max(HAND_SUPPORT_MIN_PX, int(math.ceil(HAND_SUPPORT_SHARE * area)))
    n, cb = _largest_component(valid & (win < near))
    if n < need or cb is None:
        return None
    bw, bh = ix2 - ix1 + 1, iy2 - iy1 + 1
    cw, ch = cb[2] - cb[0] + 1, cb[3] - cb[1] + 1
    if cw < fit * bw or ch < fit * bh:
        return None
    return float(np.percentile(win[valid], q))


def at_hand(depth, det, near: float = None) -> bool:
    """The shared predicate: ``det``'s box is supported as the object in the hand (``hand_depth`` is not None)."""
    return hand_depth(depth, (det or {}).get("bbox_xyxy"), near) is not None


def _wh(depth):
    if depth is None:
        return None
    try:
        return (int(depth.shape[1]), int(depth.shape[0]))
    except AttributeError:
        return (len(depth[0]), len(depth))


def _pose_brief(pose: dict) -> dict:
    p, cam = pose.get("position") or {}, pose.get("camera") or {}
    r3 = lambda d: {k: round(float(d[k]), 3) for k in ("x", "y", "z") if k in d}   # noqa: E731
    return {"position": r3(p), "yaw": round(float(pose["rotation"]["y"]), 1),
            "horizon": round(float(pose.get("horizon") or 0.0), 1), "camera": r3(cam)}


def place_of(det: dict, dets: list) -> str:
    """Weak association: the smallest *other* receptacle box containing ``det``'s box centre, else ``'?'``."""
    b = det.get("bbox_xyxy")
    if not b:
        return "?"
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    best, area = "?", None
    for o in dets:
        ob = o.get("bbox_xyxy")
        if o is det or not ob or o.get("type") not in RECEPTACLE_TYPES or o.get("type") == det.get("type"):
            continue
        if ob[0] <= cx <= ob[2] and ob[1] <= cy <= ob[3]:
            a = (ob[2] - ob[0]) * (ob[3] - ob[1])
            if area is None or a < area:
                best, area = o["type"], a
    return best


# ------------------------------------------------------------------------------------------------ tracks
@dataclass
class Track:
    tid: str
    type: str
    status: str = "trusted"
    xyz: Optional[tuple] = None        # measured position of the object (None while held / placed-unseen)
    anchor: Optional[tuple] = None     # placed only: the receptacle point it was put on/in
    state: str = "?"
    conf: float = 0.0
    step: int = 0
    obs_id: Optional[str] = None
    place: str = "?"
    row: Optional[int] = None          # current sighting row id in the store
    seen: int = 1                      # sightings that supported this belief (incl. unchanged repeats)
    misses: list = field(default_factory=list)   # [(obs_id, eye, yaw, horizon)]


class ObjectMemory:
    """Per-scene object memory for the honest agent. See the module docstring for the rules."""

    def __init__(self, scene: str, store=None, episode: Optional[str] = None, fov: float = FOV_DEFAULT,
                 clock=time.time):
        self.scene, self.store, self.fov, self.clock = scene, store, float(fov), clock
        self.episode = episode
        self.tracks: dict = {}
        self.held: Optional[str] = None       # tid of the track in hand
        self._n = 0
        self._seen_obs: set = set()
        self._places_logged: set = set()
        self._mem_rows = 0                    # rows written in RAM-only mode (fake ids)

    # ---------------------------------------------------------------- episodes
    def new_episode(self, episode: str):
        """The world was reset: drop RAM beliefs (stored rows stay as evidence / for later RAG)."""
        self.episode = episode
        self.tracks, self.held, self._seen_obs, self._places_logged = {}, None, set(), set()

    @classmethod
    def resume(cls, store, scene: str, episode: str, **kw) -> "ObjectMemory":
        """Rebuild one episode's beliefs from the store (the newest non-superseded row per track)."""
        om = cls(scene, store=store, episode=episode, **kw)
        for r in store.load_sightings(scene, episode):
            tid = r.get("track")
            if not tid:
                continue
            try:
                om._n = max(om._n, int(str(tid).rsplit(":", 1)[-1]))
            except ValueError:
                pass
            if r["status"] == "superseded":
                continue
            xyz = (r["x"], r["y"], r["z"]) if r["x"] is not None else None
            t = Track(tid=tid, type=r["object_type"], status=r["status"], state=r["state"] or "?",
                      conf=r["conf"] or 0.0, step=r["step"] or 0, obs_id=r["obs_id"], place=r["place"] or "?",
                      row=r["id"])
            if r["status"] == "placed":
                t.anchor = xyz
            elif r["status"] != "held":
                t.xyz = xyz
            om.tracks[tid] = t
            if r["status"] == "held":
                om.held = tid
        return om

    # ---------------------------------------------------------------- store plumbing
    def _write(self, t: Track, step, obs_id, why, box=None, point=None, pose=None, xyz=None, xyz_ok=None) -> int:
        row = {"scene": self.scene, "episode": self.episode, "step": step, "ts": self.clock(), "obs_id": obs_id,
               "object_type": t.type, "state": t.state, "conf": round(float(t.conf), 4), "box": box,
               "point": point, "pose": _pose_brief(pose) if pose else None,
               "xyz": xyz, "xyz_ok": bool(xyz_ok if xyz_ok is not None else xyz is not None),
               "track": t.tid, "status": t.status, "place": t.place, "why": why}
        if self.store is not None:
            return self.store.add_sighting(row)
        self._mem_rows += 1
        return self._mem_rows

    def _restatus(self, t: Track, status: str, why: str):
        if t.row is not None and self.store is not None:
            self.store.set_sighting_status(t.row, status, why)

    def _new_tid(self) -> str:
        self._n += 1
        return f"{self.episode}:{self._n}"

    # ---------------------------------------------------------------- observation update
    def update(self, obs, holding: Optional[str] = None) -> dict:
        """Fold one ``honest.Obs`` into memory. Sets ``det['track']`` on each detection it associated.

        Returns a report ``{obs_id, new, refreshed, superseded, missed, contradicted, skipped, unplaced}``.
        """
        holding = obs.holding if holding is None else holding
        oid = f"{self.episode}:{obs.step}"
        rep = {"obs_id": oid, "new": [], "refreshed": [], "superseded": [], "missed": [], "contradicted": [],
               "skipped": [], "unplaced": 0, "duplicate": False}
        if oid in self._seen_obs:                     # the same captured view is never counted twice
            rep["duplicate"] = True
            return rep
        self._seen_obs.add(oid)
        depth, pose = obs.depth, obs.pose
        wh = _wh(depth)
        dets = [d for d in obs.dets if d.get("type")]
        # 1. ground each detection in the world (own pose + depth at its grounded point)
        cand = []
        for i, d in enumerate(dets):
            d["track"] = None
            xyz, dd = None, None
            if wh and d.get("point") and d.get("point_ok"):
                dd = _depth_at(depth, *d["point"])
                xyz = backproject(d["point"][0], d["point"][1], dd, pose, wh, self.fov)
            if holding and d["type"] == holding and ((dd is not None and dd < HELD_NEAR) or at_hand(depth, d)):  # SPEC-INHAND Q4
                rep["skipped"].append((d["type"], "the object in hand"))
                continue
            cand.append((i, d, xyz))
        # 2. one-to-one association, nearest first, independent of state
        pairs = []
        for i, d, xyz in cand:
            if xyz is None:
                continue
            for t in self.tracks.values():
                if t.type != d["type"] or t.status in ("held", "superseded"):
                    continue
                ref, gate = (t.anchor, PLACED_GATE) if t.status == "placed" else (t.xyz, ASSOC_GATE)
                if ref is None:
                    continue
                dist = _dist(xyz, ref)
                if dist <= gate:
                    pairs.append((dist, i, t.tid))
        pairs.sort()
        ambiguous = set()                              # two tracks almost equally near: do not guess which
        for i in {p[1] for p in pairs}:
            ds = [p[0] for p in pairs if p[1] == i]
            if len(ds) >= 2 and ds[1] - ds[0] < AMBIG_EPS:
                ambiguous.add(i)
        used_d, used_t, match = set(ambiguous), set(), {}
        for dist, i, tid in pairs:
            if i in used_d or tid in used_t:
                continue
            used_d.add(i)
            used_t.add(tid)
            match[i] = tid
        dup = self._frame_duplicates(dets, cand, match, ambiguous) if DEDUP_FRAME else {}
        for i in [i for i, (kind, _) in dup.items() if kind == "ambig"]:
            del dup[i]                                 # F-dup: duplicates matched to different tracks -> do not choose
            ambiguous.add(i)
        # 3. write sightings
        for i, d, xyz in cand:
            if i in dup:                               # F-dup: a second box of an object already in this view
                continue
            box = list(d["bbox_xyxy"]) if d.get("bbox_xyxy") else None
            point = list(d["point"]) if d.get("point") else None
            if xyz is None or i in ambiguous:          # kept as evidence, never associated or contradicted
                rep["unplaced"] += 1
                t0 = Track(tid=None, type=d["type"], state=d.get("state") or "?", conf=d.get("conf", 0.0))
                why = ("ambiguous: two tracks of the type equally near, not associated" if xyz is not None else
                       "no valid depth at the point: not located, not associated")
                self._write(t0, obs.step, oid, why, box=box, point=point, pose=pose, xyz=xyz, xyz_ok=False)
                continue
            place = place_of(d, dets)
            state = d.get("state") or "?"
            tid = match.get(i)
            if tid is None:
                t = Track(tid=self._new_tid(), type=d["type"], xyz=xyz, state=state, conf=d.get("conf", 0.0),
                          step=obs.step, obs_id=oid, place=place)
                t.row = self._write(t, obs.step, oid, "first sighting", box, point, pose, xyz)
                self.tracks[t.tid] = t
                rep["new"].append(t.tid)
            else:
                t = self.tracks[tid]
                old_ref = t.anchor if t.status == "placed" else t.xyz
                changed = []
                if t.status != "trusted":
                    changed.append(f"{t.status} -> seen again")
                if state != t.state:
                    changed.append(f"state {t.state} -> {state}")
                if old_ref is None or _dist(xyz, old_ref) > MOVED_EPS:
                    changed.append(f"moved {(_dist(xyz, old_ref) if old_ref else 0):.2f} m")
                t.misses = []
                if changed:
                    why = "re-seen: " + "; ".join(changed)
                    self._restatus(t, "superseded", why)
                    t.status, t.xyz, t.anchor, t.state, t.place = "trusted", xyz, None, state, place
                    t.conf, t.step, t.obs_id, t.seen = d.get("conf", 0.0), obs.step, oid, t.seen + 1
                    t.row = self._write(t, obs.step, oid, why, box, point, pose, xyz)
                    rep["superseded"].append(tid)
                else:                                  # unchanged: refresh the belief, no new evidence row
                    t.conf, t.step, t.obs_id, t.seen = max(t.conf, d.get("conf", 0.0)), obs.step, oid, t.seen + 1
                    rep["refreshed"].append(tid)
            d["track"] = t.tid
            self._log_place(t, place, xyz, pose, obs.step)
        for i, (kind, ref) in dup.items():
            dets[i]["track"] = ref if kind == "track" else dets[ref].get("track")
            rep["skipped"].append((dets[i]["type"], "a second box of an object already seen in this view"))
        # 4. misses: trusted tracks this view should have seen but did not
        if wh:
            for t in list(self.tracks.values()):
                if t.status != "trusted" or t.xyz is None or t.tid in used_t:
                    continue
                if self._eligible_miss(t, dets, cand, depth, pose, wh):
                    eye = _basis(pose)[0]
                    yaw, hor = float(pose["rotation"]["y"]), float(pose.get("horizon") or 0.0)
                    if t.misses and not _distinct_view(t.misses[-1], eye, yaw, hor):
                        continue                       # the same viewpoint again counts once
                    t.misses.append((oid, eye, yaw, hor))
                    rep["missed"].append(t.tid)
                    if len(t.misses) >= MISSES_TO_CONTRADICT:
                        why = ("not seen in %d clear views (%s)" %
                               (len(t.misses), ", ".join(m[0] for m in t.misses)))
                        self._restatus(t, "contradicted", why)
                        t.status = "contradicted"
                        rep["contradicted"].append(t.tid)
        return rep

    @staticmethod
    def _frame_duplicates(dets, cand, match, ambiguous) -> dict:
        """F-dup: {det index: ("track", tid) | ("det", lead index) | ("ambig", None)} for each unmatched box that is a second
        box of an object already in this view (same type, grounded points within DUP_DIST, boxes overlapping >= DUP_IOU).
        A box overlapping a MATCHED box joins that box's track (existing association first); one overlapping matched boxes
        of two different tracks is "ambig" (the caller keeps it unassociated). Among unmatched boxes the highest confidence
        leads (ties: lower index). Pure."""
        pos = {i: xyz for i, d, xyz in cand if xyz is not None and i not in ambiguous}

        def same(i, j):
            a, b = dets[i].get("bbox_xyxy"), dets[j].get("bbox_xyxy")
            if not a or not b or dets[i]["type"] != dets[j]["type"] or _dist(pos[i], pos[j]) > DUP_DIST:
                return False
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
            return u > 0 and ix * iy / u >= DUP_IOU

        out, leads = {}, []
        for i in sorted(pos, key=lambda k: (-float(dets[k].get("conf", 0.0)), k)):
            if i in match:
                continue
            near = [j for j in match if j in pos and same(i, j)]
            if near:
                tids = {match[j] for j in near}
                out[i] = ("track", tids.pop()) if len(tids) == 1 else ("ambig", None)
                continue
            lead = next((j for j in leads if same(i, j)), None)
            if lead is not None:
                out[i] = ("det", lead)
            else:
                leads.append(i)
        return out

    def _eligible_miss(self, t: Track, dets, cand, depth, pose, wh) -> bool:
        pr = project(t.xyz, pose, wh, self.fov)
        if pr is None:
            return False
        px, py, zc = pr
        w, h = wh
        if not (EDGE_MARGIN <= px < w - EDGE_MARGIN and EDGE_MARGIN <= py < h - EDGE_MARGIN):
            return False
        if _dist(t.xyz, _basis(pose)[0]) > MISS_RANGE:
            return False
        dd = _depth_at(depth, px, py)
        if dd is None or dd < zc - OCCLUSION_TOL:      # no depth, or something in front: says nothing
            return False
        for d in dets:                                 # any same-type detection near the spot blocks the miss
            if d["type"] != t.type:
                continue
            b = d.get("bbox_xyxy")
            if b and b[0] - NEAR_DET_PX <= px <= b[2] + NEAR_DET_PX and b[1] - NEAR_DET_PX <= py <= b[3] + NEAR_DET_PX:
                return False
        for _, d, xyz in cand:
            if d["type"] == t.type and xyz is not None and _dist(xyz, t.xyz) <= ASSOC_GATE:
                return False
        return True

    def _log_place(self, t: Track, place: str, xyz, pose, step):
        key = (t.tid, place)
        if key in self._places_logged:
            return
        self._places_logged.add(key)
        if self.store is not None:
            self.store.add_place_seen({"scene": self.scene, "episode": self.episode, "step": step,
                                       "ts": self.clock(), "object_type": t.type, "place_type": place,
                                       "xyz": xyz, "pose": _pose_brief(pose), "track": t.tid,
                                       "why": "sighting (weak place association: box centre in receptacle box)"})

    # ---------------------------------------------------------------- the agent's own actions
    def on_pickup(self, ok: bool, track: Optional[str] = None, otype: Optional[str] = None, step: int = 0) -> Optional[str]:
        """After a pickup aimed at a detection (``track`` = its ``det['track']``). Only a success changes memory."""
        if not ok:
            return None
        t = self.tracks.get(track) if track else None
        if t is not None and otype and t.type != otype:
            t = None                                   # binding disagrees with the aimed type: do not guess
        if t is None:
            t = Track(tid=self._new_tid(), type=otype or "?", status="held", step=step)
            self.tracks[t.tid] = t
            why = "picked up (identity unbound: no track for the aimed detection)"
        else:
            self._restatus(t, "superseded", f"picked up by the agent at step {step}")
            why = f"picked up by the agent at step {step}"
        t.status, t.xyz, t.anchor, t.place, t.step, t.misses = "held", None, None, "hand", step, []
        t.row = self._write(t, step, None, why)
        self.held = t.tid
        return t.tid

    def relabel_held(self, otype: str, conf: float = 0.0, step: int = 0, why: str = "") -> Optional[str]:
        """E3: the held object's identity corrected from the agent's OWN evidence (the specialist's look at the hand). The old
        row is superseded and a new row with the corrected type is written (auditable; nothing is rewritten in place)."""
        t = self.tracks.get(self.held) if self.held else None
        if t is None or not otype:
            return None
        if t.type == otype:
            return t.tid
        old = t.type
        self._restatus(t, "superseded", f"identity corrected at step {step}: {old} -> {otype}")
        t.type, t.conf, t.step = otype, float(conf), step
        t.row = self._write(t, step, None, why or f"identity corrected in the hand by the specialist: {old} -> {otype} "
                                                  f"({float(conf):.2f})")
        return t.tid

    def on_put(self, ok: bool, place_type: Optional[str] = None, place_xyz=None, otype: Optional[str] = None,
               step: int = 0, why: str = "") -> Optional[str]:
        """After putting the held object on/in a receptacle sighted at ``place_xyz``. Only a success changes memory."""
        if not ok:
            return None
        t = self.tracks.get(self.held) if self.held else None
        if t is None:
            t = Track(tid=self._new_tid(), type=otype or "?", step=step)
            self.tracks[t.tid] = t
        else:
            self._restatus(t, "superseded", f"put down by the agent at step {step}")
        t.status, t.xyz, t.anchor, t.state = "placed", None, tuple(place_xyz) if place_xyz else None, "?"
        t.place, t.step, t.misses = place_type or "?", step, []
        t.row = self._write(t, step, None, why or f"put in/on {place_type or '?'} by the agent (state unknown "
                                                  "until seen again)", xyz=t.anchor, xyz_ok=False)
        self.held = None
        return t.tid

    def on_drop(self, ok: bool, pose: Optional[dict] = None, step: int = 0) -> Optional[str]:
        """After dropping the held object: placed on the floor at the agent's feet (anchor), until seen again."""
        if not ok:
            return None
        p = (pose or {}).get("position")
        xyz = (float(p["x"]), float(p["y"]), float(p["z"])) if p else None
        return self.on_put(True, "Floor", xyz, step=step, why="dropped by the agent (on the floor near it)")

    # ---------------------------------------------------------------- reads
    def lookup(self, otype: str) -> list:
        """Beliefs about ``otype`` in order: trusted (newest first), placed, held. Contradicted are left out."""
        rank = {"trusted": 0, "placed": 1, "held": 2}
        ts = [t for t in self.tracks.values() if t.type == otype and t.status in rank]
        ts.sort(key=lambda t: (rank[t.status], -t.step, -t.conf))
        return [{"track": t.tid, "type": t.type, "status": t.status,
                 "xyz": t.anchor if t.status == "placed" else t.xyz, "state": t.state, "place": t.place,
                 "step": t.step, "conf": round(t.conf, 3)} for t in ts]

    def summary(self, limit: int = 40) -> list:
        """One line per usable belief (for the LLM prompt in s4/s6)."""
        out = []
        for t in sorted(self.tracks.values(), key=lambda t: (t.type, -t.step)):
            if t.status in ("trusted", "placed", "held"):
                where = ("in hand" if t.status == "held" else
                         f"put on/in {t.place} (not re-seen)" if t.status == "placed" else f"on/in {t.place}")
                xyz = t.anchor if t.status == "placed" else t.xyz
                pos = f" at ({xyz[0]:.2f}, {xyz[2]:.2f})" if xyz else ""
                out.append(f"{t.type} [{t.state}] {where}{pos}, step {t.step}")
        return out[:limit]


def _distinct_view(prev, eye, yaw, hor) -> bool:
    _, peye, pyaw, phor = prev
    dyaw = abs((yaw - pyaw + 180.0) % 360.0 - 180.0)
    return _dist(peye, eye) >= VIEW_MOVE or dyaw >= VIEW_YAW or abs(hor - phor) >= VIEW_PITCH
