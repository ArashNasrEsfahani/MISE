"""Step 5 / A2 - place identity (docs/step5-plan.md section 3, Astra A2 review).

`PlaceBook` is an in-memory index of the places already in LTM (same kitchen + frame) plus the provisional places an
episode creates; the final `pl:<n>` ids are minted only at the consolidation commit.

Rules (all numbers from the registry, `params["identity"]`):
  * association: candidates of the same type within `place_gate_m[rtype]`; none -> create; one, or the best nearer than
    the second by >= `ambig_margin_m` -> associate; otherwise ABSTAIN (no place, no trial).
  * same-view exclusion: two boxes in one view are two places (this is what keeps adjacent identical drawers apart).
    Within a view the assignment is greedy by distance, so it does not depend on detection order.
  * regions (`region_types`): a large surface is a box that grows with linked points (`region_link_m`, same height
    within `region_height_m`).
  * merge needs POSITIVE duplicate evidence: one box covering both centres, same type, same frame, centres within
    `merge_gate_m`, and no separation witness between ANY member of the two alias components. `covers` never waives
    separation. Roots are the lowest id; aliases only ever point at roots (no chains, no cycles).
  * separation witnesses ("these two were seen as distinct boxes / opened as distinct fronts") are persisted in the
    place extent json (`sep`), so a later spanning box cannot merge them.
  * unmerge flips the alias inactive and adds a witness; all trials and claims keep their original place ids and are
    collapsed to the canonical place at READ time, so nothing is double counted and nothing is lost.
"""
from __future__ import annotations

import math
import re

PROV_RE = re.compile(r"^pl:t(\d+)$")
FINAL_RE = re.compile(r"^pl:(\d+)$")


def dist3(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


class Place:
    __slots__ = ("id", "rtype", "c", "extent", "sep", "dirty", "new")

    def __init__(self, pid, rtype, c, extent=None, sep=(), new=False):
        self.id = pid
        self.rtype = rtype
        self.c = tuple(float(x) for x in c)
        self.extent = dict(extent or {})
        self.sep = set(sep)
        self.dirty = new          # geometry / witnesses changed this episode -> must be written
        self.new = new


def id_key(pid: str):
    """Deterministic order: final ids by number, then provisional ones by creation number."""
    m = FINAL_RE.match(pid)
    if m:
        return (0, int(m.group(1)))
    m = PROV_RE.match(pid)
    return (1, int(m.group(1))) if m else (2, 0)


class PlaceBook:
    def __init__(self, params: dict, places=(), aliases=None):
        self.p = params["identity"]
        self.places: dict[str, Place] = {}
        for r in places:
            ext = r.get("extent") or {}
            self.places[r["place"]] = Place(r["place"], r["rtype"], (r["cx"], r["cy"], r["cz"]),
                                            {k: v for k, v in ext.items() if k != "sep"}, ext.get("sep", ()))
        self.alias: dict[str, str] = {k: v for k, v in (aliases or {}).items()}
        self._prov = 0
        self.merges: list[tuple] = []          # (loser, winner, evidence-info) made this episode
        self.separations: list[tuple] = []

    # -- parameters ---------------------------------------------------------------------------------------
    def gate(self, rtype: str) -> float:
        g = self.p["place_gate_m"]["value"]
        return float(g.get(rtype, g["default"]))

    def is_region(self, rtype: str) -> bool:
        return rtype in self.p["region_types"]["value"]

    def _v(self, key):
        return self.p[key]["value"]

    # -- alias components -----------------------------------------------------------------------------------
    def canonical(self, pid):
        if pid is None:
            return None
        seen = set()
        while pid in self.alias and pid not in seen:
            seen.add(pid)
            pid = self.alias[pid]
        return pid

    def component(self, root: str) -> set:
        return {p for p in self.places if self.canonical(p) == root} | {root}

    def roots(self, rtype: str):
        return sorted((p for p in self.places.values() if p.rtype == rtype and self.canonical(p.id) == p.id),
                      key=lambda p: id_key(p.id))

    # -- geometry -----------------------------------------------------------------------------------------
    def distance(self, place: Place, xyz) -> float:
        """Centre distance, or for a region the xz distance to its box (None when at a different height)."""
        if self.is_region(place.rtype) and "x0" in place.extent:
            e = place.extent
            if abs(float(xyz[1]) - place.c[1]) > self._v("region_height_m"):
                return math.inf
            dx = max(e["x0"] - xyz[0], 0.0, xyz[0] - e["x1"])
            dz = max(e["z0"] - xyz[2], 0.0, xyz[2] - e["z1"])
            return math.hypot(dx, dz)
        return dist3(place.c, xyz)

    def candidates(self, rtype: str, xyz):
        gate = self._v("region_link_m") if self.is_region(rtype) else self.gate(rtype)
        out = []
        for pl in self.roots(rtype):
            # a merged place is the union of its members: nearest member decides
            d = min(self.distance(self.places[m], xyz) for m in self.component(pl.id) if m in self.places)
            if d <= gate:
                out.append((d, pl.id))
        out.sort(key=lambda t: (round(t[0], 9), id_key(t[1])))
        return out

    def _classify(self, rtype, xyz):
        c = self.candidates(rtype, xyz)
        if not c:
            return "new", None, c
        if len(c) == 1 or c[1][0] - c[0][0] >= self._v("ambig_margin_m"):
            return "assoc", c[0][1], c
        return "abstain", None, c

    # -- creation -------------------------------------------------------------------------------------------
    def _create(self, rtype, xyz, extent=None) -> str:
        self._prov += 1
        pid = f"pl:t{self._prov}"
        ext = dict(extent or {})
        if self.is_region(rtype):
            ext.update({"x0": float(xyz[0]), "x1": float(xyz[0]), "z0": float(xyz[2]), "z1": float(xyz[2])})
        self.places[pid] = Place(pid, rtype, xyz, ext, new=True)
        return pid

    def _grow(self, pl: Place, xyz):
        if self.is_region(pl.rtype) and "x0" in pl.extent:
            e = pl.extent
            n = {"x0": min(e["x0"], float(xyz[0])), "x1": max(e["x1"], float(xyz[0])),
                 "z0": min(e["z0"], float(xyz[2])), "z1": max(e["z1"], float(xyz[2]))}
            if any(n[k] != e[k] for k in n):
                e.update(n)
                pl.dirty = True

    # -- resolution -----------------------------------------------------------------------------------------
    def resolve(self, rtype: str, xyz, create: bool = True):
        """Single resolution (objects, opens, inspections, puts): returns (kind, place_id, candidates)."""
        kind, pid, cands = self._classify(rtype, xyz)
        if kind == "new":
            if not create:
                return "none", None, cands
            pid = self._create(rtype, xyz)
            return "new", pid, cands
        if kind == "assoc":
            self._grow(self.places[pid], xyz)
        return kind, pid, cands

    def resolve_view(self, sightings):
        """sightings: list of (key, rtype, xyz). One view; boxes in a view are distinct places.

        Returns {key: (kind, place_id|None)}. Deterministic: independent of the order the boxes were detected in.
        """
        res, wants = {}, []
        for key, rtype, xyz in sorted(sightings, key=lambda s: (s[1], tuple(s[2]), s[0])):
            if self.is_region(rtype):               # a surface is seen as many points: no same-view exclusion
                kind, pid, _ = self.resolve(rtype, xyz)
                res[key] = (kind, pid)
                continue
            kind, pid, cands = self._classify(rtype, xyz)
            if kind == "assoc":
                wants.append((cands[0][0], key, rtype, xyz, pid))
            elif kind == "abstain":
                res[key] = ("abstain", None)
            else:
                wants.append((0.0, key, rtype, xyz, None))
        taken = {}
        for d, key, rtype, xyz, pid in sorted((w for w in wants if w[4] is not None), key=lambda w: (w[0], w[1])):
            if pid in taken:
                continue
            taken[pid] = key
            self._grow(self.places[pid], xyz)
            res[key] = ("assoc", pid)
        for d, key, rtype, xyz, pid in sorted(wants, key=lambda w: (w[2], tuple(w[3]), w[1])):
            if key in res:
                continue
            res[key] = ("new", self._create(rtype, xyz))
        # separation witnesses: distinct boxes of one view that resolved to distinct places of one type
        bytype = {s[0]: s[1] for s in sightings if not self.is_region(s[1])}
        placed = sorted(((k, v[1]) for k, v in res.items() if v[1] and k in bytype), key=lambda t: id_key(t[1]))
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                a, b = placed[i][1], placed[j][1]
                if a != b and bytype[placed[i][0]] == bytype[placed[j][0]]:
                    self.add_separation(a, b, "same_view")
        return res

    # -- separation / merge -----------------------------------------------------------------------------------
    def add_separation(self, a: str, b: str, why: str):
        a, b = self.canonical(a), self.canonical(b)
        if a == b or a is None or b is None:
            return
        pa, pb = self.places[a], self.places[b]
        if b not in pa.sep or a not in pb.sep:
            pa.sep.add(b)
            pb.sep.add(a)
            pa.dirty = pb.dirty = True
            self.separations.append((a, b, why))

    def separated(self, ra: str, rb: str) -> bool:
        ca, cb = self.component(ra), self.component(rb)
        return any(self.places[x].sep & cb for x in ca if x in self.places)

    def near_limit(self, rtype: str) -> float:
        """Pairs further apart than this could never be confused (derived from the registry, no new constant)."""
        return 2.0 * max(self.gate(rtype), self._v("merge_gate_m"))

    def try_merge(self, a: str, b: str, covered_by_one_box: bool, same_frame: bool = True):
        """Merge the components of a and b iff there is positive duplicate evidence. Returns (merged, reason)."""
        ra, rb = self.canonical(a), self.canonical(b)
        if ra is None or rb is None or ra == rb:
            return False, "same_component"
        pa, pb = self.places[ra], self.places[rb]
        if pa.rtype != pb.rtype:
            return False, "incompatible_type"
        if not same_frame:
            return False, "frame"
        if not covered_by_one_box:
            return False, "no_spanning_box"
        if dist3(pa.c, pb.c) > self._v("merge_gate_m"):
            return False, "too_far"
        if self.separated(ra, rb):
            return False, "separation_witness"
        winner, loser = sorted((ra, rb), key=id_key)
        if self.canonical(winner) != winner:                       # pragma: no cover (cannot happen: roots only)
            return False, "cycle"
        self.alias[loser] = winner
        self.merges.append((loser, winner))
        return True, "merged"

    # -- minting ----------------------------------------------------------------------------------------------
    def provisional(self):
        return sorted((p for p in self.places if PROV_RE.match(p)), key=id_key)

    @staticmethod
    def max_final(ids) -> int:
        n = 0
        for i in ids:
            m = FINAL_RE.match(i)
            if m:
                n = max(n, int(m.group(1)))
        return n
