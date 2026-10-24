"""B3 - learned kitchen profile from honest practice events (docs/final-test/kitchen_memory/PACKAGE-SPEC.md section 6).

What the agent has learned about THIS kitchen by acting in it: which fixtures it met, what state it last saw them in, what its own
opens / closes / toggles did, what it detected inside them, where it last saw a tool, and (only if its own interventions identify
it) which stove knob drives which burner.  Every line is a historical, hedged fact with an evidence reference, never advice, and
none of it is read from the simulator: the only inputs are `s5.ev/1` events (the validated `HonestEventAdapter.stream`) or, for
the early check, the practice store's own claim tables.

  * No FloorPlan1 constant: a kitchen name is data; fixture and tool types come from the `categories` argument.
  * Evidence semantics are frozen in spec 6.4: contents only from an inspection with verdict found / visible; "nothing detected"
    only from a `visible` inspection with an empty `detected`; a failed open, a partial / not-visible / unsure inspection and a
    missing detection give no content evidence; the newest observation wins a state and a contradiction keeps both counts.
  * Resets do not erase practice facts.  Facts are historical hypotheses ("last seen ... in practice attempt N of K").
  * Pure standard library; imports nothing of the simulator.  Python 3.10 compatible.

    python sim/ltm_profile.py --early-check          (offline, < 1 minute: the spec 6.6 early check on the practice-v1 store)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys

SCHEMA_PROFILE = "pkg.profile/1"
SCHEMA_REC = "pkg.rec/1"
SCHEMA_CAND = "pkg.cand/1"
SCHEMA_EVS = "pkg.evs/1"
VERSION = "profile-1"
CELL = 0.5                      # quantisation of positions, metres
MERGE_M = 0.75                  # two sightings of one fixture type closer than this are the same place
KNOB_TYPE = "StoveKnob"
BURNER_TYPE = "StoveBurner"
ON_STATES = frozenset({"on", "lit", "burning", "toggled", "true"})
OFF_STATES = frozenset({"off", "unlit", "untoggled", "false"})
CONTENT_VERDICTS = ("found", "visible")
MAX_REFUSALS = 5
MAX_CONTENT_TYPES = 8
MAX_EVIDENCE = 12
HERE = os.path.dirname(os.path.abspath(__file__))



class ProfileError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------------------ small helpers
def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(x) -> str:
    return hashlib.sha256(_canon(x).encode("utf-8")).hexdigest()


def self_sha(obj: dict) -> str:
    return _sha({k: v for k, v in obj.items() if k != "sha256"})


def ev_hash(episode: str, ev: int) -> str:
    """Same formula as ltm_events.ev_hash (the selftest asserts equality): the 16-hex reference of one event."""
    return hashlib.sha256(f"{episode}#{ev}".encode("utf-8")).hexdigest()[:16]


def _file_sha16() -> str:
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:                                                      # pragma: no cover
        return "unknown"


def _cell(x, z):
    return [round(float(x) / CELL) * CELL, round(float(z) / CELL) * CELL]


def _types(categories: dict, key: str) -> frozenset:
    cats = categories or {}
    v = cats.get(key)
    if v is None and isinstance(cats.get("categories"), dict):
        v = cats["categories"].get(key)
    return frozenset(v or ())


def _pl(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


# ------------------------------------------------------------------------------------------------------------ accumulators
def _new_fixture():
    return {"n_seen": 0, "cells": [], "parent_type": None, "last_state": None, "states_seen": {},
            "opened_by_me": {"ok": 0, "failed": 0}, "closed_by_me": {"ok": 0, "failed": 0},
            "toggled": {"on_ok": 0, "off_ok": 0, "refused": 0}, "refusals": [], "contents": {},
            "contents_basis": "inspection", "inspected_nothing": {"n": 0, "last_ord": 0},
            "support": 0, "first_ord": 0, "last_ord": 0}


def _new_tool():
    return {"n_seen": 0, "last_parent_type": None, "last_ord": 0, "support": 0, "first_ord": 0}


class _Acc:
    """Collects entities plus the per-facet evidence, episode support and recency while the streams are folded in."""

    def __init__(self):
        self.fixtures: dict = {}
        self.tools: dict = {}
        self.ev: dict = {}              # (kind, type, facet) -> {ev hash: ord}
        self.eps: dict = {}             # (kind, type) -> set of episode ordinals that back any facet
        self.clusters: dict = {}        # fixture type -> [[x, z, cell]]
        self.tool_tracks: dict = {}     # tool type -> set of track ids
        self.knob_pairs: dict = {}      # (knob_cell tuple, burner_cell tuple) -> {"n": int, "ords": set}
        self.knob_unlocatable = 0
        self.knob_ambiguous = 0

    def fixture(self, t):
        return self.fixtures.setdefault(t, _new_fixture())

    def tool(self, t):
        return self.tools.setdefault(t, _new_tool())

    def note(self, kind, t, facet, ord_, h):
        self.ev.setdefault((kind, t, facet), {})[h] = ord_
        self.eps.setdefault((kind, t), set()).add(ord_)


def _fold_episode(acc: _Acc, ord_: int, stream: list, fix: frozenset, tools: frozenset) -> None:
    """One practice episode's events, in `ev` order."""
    steps: dict = {}                    # i -> {"verb", "otype", "pre", "burners": {track: (state, cell)}}
    cur = None
    burner_prev: dict = {}              # track -> state, from earlier in this episode
    for e in sorted(stream, key=lambda x: int(x.get("ev", 0))):
        kind = e.get("kind")
        h = ev_hash(str(e.get("episode")), int(e.get("ev", 0)))
        if kind == "receptacle_seen":
            t, xyz = e.get("rtype"), e.get("xyz")
            if t in fix and xyz:
                fx = acc.fixture(t)
                cl = acc.clusters.setdefault(t, [])
                best = None
                for c in cl:
                    d = math.hypot(c[0] - float(xyz[0]), c[1] - float(xyz[2]))
                    if d <= MERGE_M and (best is None or d < best[0]):
                        best = (d, c)
                if best is None:
                    cl.append([float(xyz[0]), float(xyz[2]), _cell(xyz[0], xyz[2])])
                acc.note("fixture", t, "seen", ord_, h)
                fx["last_ord"] = max(fx["last_ord"], ord_)
        elif kind in ("open", "close"):
            t = e.get("rtype")
            if t in fix:
                fx = acc.fixture(t)
                if e.get("by_me", True):
                    slot = fx["opened_by_me" if kind == "open" else "closed_by_me"]
                    slot["ok" if e.get("ok") else "failed"] += 1
                    acc.note("fixture", t, "opened" if kind == "open" else "closed", ord_, h)
                seen = e.get("state_seen")
                val = seen if seen in ("open", "closed") else (("open" if kind == "open" else "closed") if e.get("ok") else None)
                if val:
                    _set_state(acc, t, fx, val, ord_, int(e["ev"]), h)
        elif kind == "inspection":
            t, verdict = e.get("rtype"), e.get("verdict")
            if t in fix:
                fx = acc.fixture(t)
                det = [d for d in (e.get("detected") or []) if d != t and d not in fix]
                if verdict in CONTENT_VERDICTS:
                    for d in det:
                        c = fx["contents"].setdefault(d, {"n": 0, "last_ord": 0})
                        c["n"] += 1
                        c["last_ord"] = max(c["last_ord"], ord_)
                        acc.note("fixture", t, "contents:" + d, ord_, h)
                    if verdict == "visible" and not e.get("detected"):
                        fx["inspected_nothing"]["n"] += 1
                        fx["inspected_nothing"]["last_ord"] = max(fx["inspected_nothing"]["last_ord"], ord_)
                        acc.note("fixture", t, "nothing", ord_, h)
        elif kind == "sighting":
            t = e.get("otype")
            if t in tools:
                tl = acc.tool(t)
                acc.tool_tracks.setdefault(t, set()).add((ord_, e.get("track")))
                acc.note("tool", t, "seen", ord_, h)
                if e.get("in_rtype") and e.get("in_basis") in ("inside_inspection", "depth_consistent", "box_only"):
                    if ord_ >= tl["last_ord"]:
                        tl["last_parent_type"], tl["last_ord"] = e["in_rtype"], ord_
                    acc.note("tool", t, "at", ord_, h)
                else:
                    tl["last_ord"] = max(tl["last_ord"], ord_)
            elif t == BURNER_TYPE and cur is not None:
                xyz = e.get("xyz")
                cur["burners"][e.get("track")] = (e.get("state"), _cell(xyz[0], xyz[2]) if xyz else None,
                                                  burner_prev.get(e.get("track")))
            if t == BURNER_TYPE:
                burner_prev[e.get("track")] = e.get("state")
        elif kind == "put":
            t = e.get("otype")
            if t in tools and e.get("ok") and e.get("rtype") not in (None, "?"):
                tl = acc.tool(t)
                if ord_ >= tl["last_ord"]:
                    tl["last_parent_type"], tl["last_ord"] = e["rtype"], ord_
                acc.note("tool", t, "at", ord_, h)
        elif kind == "step_start":
            cur = {"verb": e.get("verb"), "otype": e.get("otype"), "pre": e.get("pre") or {}, "burners": {}, "ev": h}
            steps[int(e.get("i", 0))] = cur
        elif kind == "step_outcome":
            st = steps.get(int(e.get("i", 0)))
            if st is not None and st["verb"] in ("toggle_on", "toggle_off") and st["otype"] in fix:
                t = st["otype"]
                fx = acc.fixture(t)
                if e.get("ok"):
                    fx["toggled"]["on_ok" if st["verb"] == "toggle_on" else "off_ok"] += 1
                    acc.note("fixture", t, "toggled", ord_, h)
                    _set_state(acc, t, fx, "on" if st["verb"] == "toggle_on" else "off", ord_, int(e["ev"]), h)
                    if t == KNOB_TYPE:
                        _learn_knob(acc, st, ord_)
            cur = None
        elif kind == "refusal":
            t = e.get("rtype") if e.get("rtype") in fix else (e.get("otype") if e.get("otype") in fix else None)
            if t is not None:
                fx = acc.fixture(t)
                bs = fx["last_state"]["value"] if fx["last_state"] else None
                fx["refusals"].append({"verb": e.get("verb"), "cause": e.get("cause") or "unknown", "believed_state": bs, "ord": ord_})
                if str(e.get("verb", "")).startswith("toggle"):
                    fx["toggled"]["refused"] += 1
                acc.note("fixture", t, "refusals", ord_, h)


def _set_state(acc, t, fx, val, ord_, ev, h):
    fx["states_seen"][val] = fx["states_seen"].get(val, 0) + 1
    ls = fx["last_state"]
    if ls is None or (ord_, ev) >= (ls["ord"], ls.get("_ev", 0)):
        fx["last_state"] = {"value": val, "ord": ord_, "_ev": ev}
    acc.note("fixture", t, "state", ord_, h)


def _learn_knob(acc: _Acc, st: dict, ord_: int) -> None:
    """Spec 6.4: a pair is learned only when, inside ONE step, a successful knob toggle was followed by exactly ONE burner track
    whose sighting state changed accordingly.  The knob's own cell must be known (from the step's `pre`); otherwise not learned."""
    want = ON_STATES if st["verb"] == "toggle_on" else OFF_STATES
    changed = [c for (s, c, prev) in st["burners"].values() if prev is not None and s != prev and s in want]
    if len(changed) != 1 or changed[0] is None:
        return
    pre = st.get("pre") or {}
    x, z = pre.get("x"), pre.get("z")
    if not isinstance(x, (int, float)) or not isinstance(z, (int, float)) or isinstance(x, bool) or isinstance(z, bool):
        acc.knob_unlocatable += 1
        return
    key = (tuple(_cell(x, z)), tuple(changed[0]))
    p = acc.knob_pairs.setdefault(key, {"n": 0, "ords": set()})
    p["n"] += 1
    p["ords"].add(ord_)


# ------------------------------------------------------------------------------------------------------------ the profile
class Profile:
    def __init__(self, kitchen, fixtures, tools, knob_map, *, partial=False, k_total=0, upto_attempt=None, source=None,
                 evidence=None, support=None, categories=None):
        self.kitchen = kitchen
        self.fixtures, self.tools, self.knob_map = fixtures, tools, knob_map
        self.partial, self.k_total, self.upto_attempt = bool(partial), int(k_total), upto_attempt
        self.source = source or {}
        self.evidence = evidence or {}          # "kind/type/facet" -> {hash: ord}
        self.support = support or {}            # "kind/type" -> [ords]
        self.categories = {"fixture_types": sorted(_types(categories, "fixture_types")),
                           "tool_types": sorted(_types(categories, "tool_types"))}

    # -- construction ----------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, kitchen: str, episodes: list, categories: dict, upto_attempt=None) -> "Profile":
        fix, tools = _types(categories, "fixture_types"), _types(categories, "tool_types")
        rows, seen_keys = [], set()
        for i, ep in enumerate(episodes, 1):
            key = ep.get("key") or ep.get("episode") or f"ep{i}"
            if key in seen_keys:
                continue                                                   # an append-only resume may repeat a row
            seen_keys.add(key)
            at = ep.get("attempt")
            rows.append((int(at) if at is not None else i, str(key), ep))
        rows.sort(key=lambda r: (r[0], r[1]))
        if upto_attempt is not None:
            rows = [r for r in rows if r[0] <= int(upto_attempt)]
        acc = _Acc()
        for ord_, _key, ep in rows:
            _fold_episode(acc, ord_, list(ep.get("stream") or []), fix, tools)
        for t, cl in acc.clusters.items():
            fx = acc.fixture(t)
            cells = sorted({tuple(c[2]) for c in cl})
            fx["n_seen"], fx["cells"] = len(cl), [list(c) for c in cells]
        for t, tr in acc.tool_tracks.items():
            acc.tool(t)["n_seen"] = len({x for x in tr})
        for (kind, t), ords in acc.eps.items():
            ent = acc.fixture(t) if kind == "fixture" else acc.tool(t)
            ent["support"] = len(ords)
            ent["first_ord"], ent["last_ord"] = min(ords), max(ords)
        for fx in acc.fixtures.values():
            fx["refusals"] = fx["refusals"][-MAX_REFUSALS:]
            if fx["last_state"]:
                fx["last_state"] = {"value": fx["last_state"]["value"], "ord": fx["last_state"]["ord"]}
        knob = _knob_map(acc)
        k_total = len(rows)
        return cls(kitchen, acc.fixtures, acc.tools, knob, partial=False, k_total=k_total, upto_attempt=upto_attempt,
                   source={"kind": "events", "n_episodes": len(rows), "episode_keys": [r[1] for r in rows],
                           "knob_unlocatable": acc.knob_unlocatable},
                   evidence={f"{k}/{t}/{f}": v for (k, t, f), v in acc.ev.items()},
                   support={f"{k}/{t}": sorted(v) for (k, t), v in acc.eps.items()}, categories=categories)

    @classmethod
    def from_store(cls, ltm_db_path: str, kitchen: str, categories: dict) -> "Profile":
        """A PARTIAL profile from a practice store (read-only): places seen, location claims (contents of a fixture; the last
        place of a tool) and negative evidence.  No states, no own-action counts, no knob map: the store does not hold them."""
        fix, tools = _types(categories, "fixture_types"), _types(categories, "tool_types")
        con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(ltm_db_path).replace("\\", "/"), uri=True)
        con.row_factory = sqlite3.Row
        try:
            ep_ord = {r["episode"]: int(r["E"]) for r in con.execute("select episode, E from episodes where kitchen=?", (kitchen,))}
            ev_by_claim: dict = {}
            for r in con.execute("select evidence, claim, E from evidence"):
                ev_by_claim.setdefault(r["claim"], []).append((int(r["E"]), r["evidence"]))
            aliased = {r["place"] for r in con.execute("select place from place_alias where active=1")} \
                if _has(con, "place_alias") else set()
            places = {}
            for r in con.execute("select place, rtype, cx, cz from place_instance where kitchen=?", (kitchen,)):
                if r["rtype"] in fix and r["place"] not in aliased:
                    places[r["place"]] = (r["rtype"], r["cx"], r["cz"])
            claims = list(con.execute("select claim, otype, place, verified, status, E_support from location_claim "
                                      "where kitchen=? and status in ('current','maybe')", (kitchen,)))
            negs = list(con.execute("select neg, place, otype, support, E, superseded from negative_evidence where kitchen=?", (kitchen,))) \
                if _has(con, "negative_evidence") else []
        finally:
            con.close()
        acc = _Acc()
        for pl, (rt, cx, cz) in sorted(places.items()):
            fx = acc.fixture(rt)
            cl = acc.clusters.setdefault(rt, [])
            if not any(math.hypot(c[0] - cx, c[1] - cz) <= MERGE_M for c in cl):
                cl.append([cx, cz, _cell(cx, cz)])
            for e, h in ev_by_claim.get("place_instance/" + pl, []):
                acc.note("fixture", rt, "seen", e, h)
        for c in sorted(claims, key=lambda r: r["claim"]):
            ot, pl = c["otype"], c["place"]
            ev = ev_by_claim.get("location_claim/" + c["claim"], [])
            if ot in tools:
                tl = acc.tool(ot)
                tl["n_seen"] += 1
                parent = _place_type(places, pl)
                e = max([x[0] for x in ev] or [int(c["E_support"] or 0)])
                if parent and e >= tl["last_ord"]:
                    tl["last_parent_type"], tl["last_ord"] = parent, e
                for e2, h in ev:
                    acc.note("tool", ot, "seen", e2, h)
                    if parent:
                        acc.note("tool", ot, "at", e2, h)
            elif pl in places and ot not in fix and c["verified"] == "verified":
                rt = places[pl][0]
                fx = acc.fixture(rt)
                fx["contents_basis"] = "claims"
                d = fx["contents"].setdefault(ot, {"n": 0, "last_ord": 0})
                d["n"] += 1
                d["last_ord"] = max([d["last_ord"]] + [x[0] for x in ev] + [int(c["E_support"] or 0)])
                for e2, h in ev:
                    acc.note("fixture", rt, "contents:" + ot, e2, h)
        for n in sorted(negs, key=lambda r: r["neg"]):
            if n["superseded"] or n["place"] not in places:
                continue
            rt = places[n["place"]][0]
            fx = acc.fixture(rt)
            fx["inspected_nothing"]["n"] += 1
            fx["inspected_nothing"]["last_ord"] = max(fx["inspected_nothing"]["last_ord"], int(n["E"] or 0))
            for e2, h in ev_by_claim.get("negative_evidence/" + n["neg"], []) or [(int(n["E"] or 0), "neg:" + n["neg"])]:
                acc.note("fixture", rt, "nothing", e2, h)
        for t, cl in acc.clusters.items():
            fx = acc.fixture(t)
            fx["n_seen"], fx["cells"] = len(cl), [list(c) for c in sorted({tuple(x[2]) for x in cl})]
        for (kind, t), ords in acc.eps.items():
            ent = acc.fixture(t) if kind == "fixture" else acc.tool(t)
            ent["support"] = len(ords)
            ent["first_ord"], ent["last_ord"] = min(ords), max(ords)
        k_total = max(ep_ord.values()) if ep_ord else 0
        return cls(kitchen, acc.fixtures, acc.tools, "not_learned", partial=True, k_total=k_total, upto_attempt=None,
                   source={"kind": "store", "n_episodes": len(ep_ord), "db": os.path.basename(os.path.dirname(ltm_db_path))},
                   evidence={f"{k}/{t}/{f}": v for (k, t, f), v in acc.ev.items()},
                   support={f"{k}/{t}": sorted(v) for (k, t), v in acc.eps.items()}, categories=categories)

    # -- serialisation ----------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        d = {"schema": SCHEMA_PROFILE, "version": VERSION, "kitchen": self.kitchen, "partial": self.partial,
             "k_total": self.k_total, "upto_attempt": self.upto_attempt, "source": self.source,
             "categories": self.categories, "fixtures": self.fixtures, "tools": self.tools, "knob_map": self.knob_map,
             "evidence": {k: {h: o for h, o in sorted(v.items())} for k, v in sorted(self.evidence.items())},
             "support": self.support}
        d["sha256"] = self_sha(d)
        return d

    @classmethod
    def load(cls, path: str) -> "Profile":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") != SCHEMA_PROFILE:
            raise ProfileError(f"{path}: not a {SCHEMA_PROFILE} file")
        if d.get("sha256") != self_sha(d):
            raise ProfileError(f"{path}: sha256 mismatch (the profile was edited)")
        p = cls(d["kitchen"], d["fixtures"], d["tools"], d["knob_map"], partial=d["partial"], k_total=d["k_total"],
                upto_attempt=d.get("upto_attempt"), source=d.get("source"), evidence=d.get("evidence"), support=d.get("support"))
        p.categories = d.get("categories") or p.categories
        return p

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = self.to_json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1, sort_keys=True)
        return d["sha256"]

    # -- evidence ----------------------------------------------------------------------------------------------------
    def _evidence(self, kind, t, facets):
        out = {}
        for f in facets:
            out.update(self.evidence.get(f"{kind}/{t}/{f}", {}))
        # newest first, then hash, capped; deterministic
        return [h for h, _o in sorted(out.items(), key=lambda x: (-x[1], x[0]))][:MAX_EVIDENCE]

    # -- rendering ---------------------------------------------------------------------------------------------------
    def _fixture_facets(self, t, fx):
        """[(facet, en, fa)] - one entry per rendered assertion; an assertion is rendered only if it has evidence."""
        K, L = self.k_total, fx["last_ord"]
        out = []
        n = fx["n_seen"]
        if n > 0 and self.evidence.get(f"fixture/{t}/seen"):
            out.append(("seen", f"{t}: seen at {_pl(n, 'place')}", f"{t}: در {n} جا دیده شد"))
        st = fx.get("last_state")
        if st and self.evidence.get(f"fixture/{t}/state"):
            ss = fx.get("states_seen") or {}
            many = n > 1                                     # states are kept per type: with several places say so (Astra B3)
            ov = {"open": "باز", "closed": "بسته", "on": "روشن", "off": "خاموش"}
            if len(ss) > 1 and "open" in ss and "closed" in ss:
                out.append(("state", f"seen open and closed in practice; last: {st['value']}" + (f", across {n} places" if many else ""),
                            f"هم باز و هم بسته دیده شد؛ آخرین بار: {ov.get(st['value'], st['value'])}" + (f"، در {n} جا" if many else "")))
            elif len(ss) > 1 and "on" in ss and "off" in ss:
                out.append(("state", f"seen on and off in practice; last: {st['value']}",
                            f"هم روشن و هم خاموش دیده شد؛ آخرین بار: {ov.get(st['value'], st['value'])}"))
            else:
                out.append(("state", f"last seen {st['value']} in practice" + (f", at one of {n} places" if many else ""),
                            f"آخرین بار در تمرین {ov.get(st['value'], st['value'])} دیده شد" + (f"، در یکی از {n} جا" if many else "")))
        for facet, key, en_v, fa_v in (("opened", "opened_by_me", "opened", "باز کردم"), ("closed", "closed_by_me", "closed", "بستم")):
            c = fx[key]
            if (c["ok"] or c["failed"]) and self.evidence.get(f"fixture/{t}/{facet}"):
                en = f"{en_v} by me {c['ok']}x" + (f", failed {c['failed']}x" if c["failed"] else "")
                fa = f"{c['ok']} بار {fa_v}" + (f"، {c['failed']} بار ناموفق بود" if c["failed"] else "")
                out.append((facet, en, fa))
        tg = fx["toggled"]
        if (tg["on_ok"] or tg["off_ok"]) and self.evidence.get(f"fixture/{t}/toggled"):
            out.append(("toggled", f"turned on by me {tg['on_ok']}x, turned off by me {tg['off_ok']}x",
                        f"{tg['on_ok']} بار روشن و {tg['off_ok']} بار خاموش کردم"))
        if fx["refusals"] and self.evidence.get(f"fixture/{t}/refusals"):
            last = fx["refusals"][-1]
            nref = len(fx["refusals"])
            cause = last["cause"]
            out.append(("refusals", f"my {last['verb']} was refused {nref}x, last cause: {cause}",
                        f"{nref} بار {last['verb']} رد شد؛ آخرین علت: {cause}"))
        names = _content_names(fx)
        if names and all(self.evidence.get(f"fixture/{t}/contents:{x}") for x in names):
            lst = ", ".join(names)
            if fx.get("contents_basis") == "claims":
                out.append(("contents", f"objects seen there: {lst}", f"اشیای دیده‌شده در آنجا: {lst}"))
            else:
                out.append(("contents", f"inside I detected: {lst}", f"داخل آن تشخیص دادم: {lst}"))
        nn = fx["inspected_nothing"]
        if nn["n"] and self.evidence.get(f"fixture/{t}/nothing"):
            out.append(("nothing", f"looked inside {nn['n']}x and detected nothing", f"{nn['n']} بار داخل آن را دیدم و چیزی تشخیص ندادم"))
        if out:
            out.append(("recency", f"last seen in practice attempt {L} of {K}", f"آخرین بار در تلاش {L} از {K} دیده شد"))
        return out

    def _tool_facets(self, t, tl):
        K, L = self.k_total, tl["last_ord"]
        out = []
        if tl["n_seen"] and self.evidence.get(f"tool/{t}/seen"):
            out.append(("seen", f"{t}: seen {tl['n_seen']}x", f"{t}: {tl['n_seen']} بار دیده شد"))
        if tl.get("last_parent_type") and self.evidence.get(f"tool/{t}/at"):
            p = tl["last_parent_type"]
            out.append(("at", f"last seen at {p}", f"آخرین بار کنار {p} یا درون آن دیده شد"))
        if out:
            out.append(("recency", f"last seen in practice attempt {L} of {K}", f"آخرین بار در تلاش {L} از {K} دیده شد"))
        return out

    def _knob_facets(self):
        if self.knob_map in ("not_learned", None) or not self.knob_map:
            return []
        parts_en, parts_fa = [], []
        for m in self.knob_map:
            parts_en.append(f"the knob near ({m['knob_cell'][0]:g}, {m['knob_cell'][1]:g}) lit the burner near "
                            f"({m['burner_cell'][0]:g}, {m['burner_cell'][1]:g}) {m['n']}x")
            parts_fa.append(f"دستگیره نزدیک ({m['knob_cell'][0]:g}, {m['knob_cell'][1]:g}) مشعل نزدیک "
                            f"({m['burner_cell'][0]:g}, {m['burner_cell'][1]:g}) را {m['n']} بار روشن کرد")
        return [("map", f"{KNOB_TYPE} to {BURNER_TYPE}: " + "; ".join(parts_en), f"{KNOB_TYPE} به {BURNER_TYPE}: " + "؛ ".join(parts_fa))]

    def assertions(self) -> list:
        """[{rec, facet, evidence}] - every rendered assertion with its evidence hashes (the early check counts empty lists)."""
        out = []
        for t, fx in sorted(self.fixtures.items()):
            for facet, _en, _fa in self._fixture_facets(t, fx):
                ev = (self._evidence("fixture", t, [facet] if facet != "contents" else
                                     ["contents:" + x for x in _content_names(fx)]) if facet != "recency"
                      else self._evidence("fixture", t, _all_facets(self.evidence, "fixture", t)))
                out.append({"rec": self._rid("fixture", t), "facet": facet, "evidence": ev})
        for t, tl in sorted(self.tools.items()):
            for facet, _en, _fa in self._tool_facets(t, tl):
                ev = self._evidence("tool", t, [facet] if facet != "recency" else _all_facets(self.evidence, "tool", t))
                out.append({"rec": self._rid("tool", t), "facet": facet, "evidence": ev})
        if self._knob_facets():
            out.append({"rec": self._rid("knob", "map"), "facet": "map", "evidence": self._knob_evidence()})
        return out

    def _knob_evidence(self):
        if self.knob_map in ("not_learned", None) or not self.knob_map:
            return []
        return sorted({h for m in self.knob_map for h in m.get("evidence", [])})[:MAX_EVIDENCE]

    def _rid(self, kind, t):
        return f"profile/{self.kitchen}/{kind}:{t}" if kind != "knob" else f"profile/{self.kitchen}/knob_map"

    def records(self) -> list:
        """pkg.rec/1, component 'profile'.  Nothing that lacks evidence is rendered."""
        recs = []
        authored = f"script:ltm_profile@{_file_sha16()}"
        ns = self.source.get("db") or "practice"
        for kind, ents in (("fixture", self.fixtures), ("tool", self.tools)):
            for t, ent in sorted(ents.items()):
                facets = self._fixture_facets(t, ent) if kind == "fixture" else self._tool_facets(t, ent)
                if not facets:
                    continue
                en = self._join([f[1] for f in facets], "; ")
                fa = self._join([f[2] for f in facets], "؛ ")
                ev = self._evidence(kind, t, _all_facets(self.evidence, kind, t))
                types = [t] + (_content_names(ent) if kind == "fixture" else [])
                rels, ops = [], []
                if kind == "fixture":
                    rels += ["contains:" + x for x in _content_names(ent)]
                    if ent["opened_by_me"]["ok"] or ent["opened_by_me"]["failed"] or ent["closed_by_me"]["ok"]:
                        ops += ["open", "close"]
                        rels.append("opens:" + t)
                    if ent["toggled"]["on_ok"] or ent["toggled"]["off_ok"] or ent["toggled"]["refused"]:
                        ops += ["toggle_on", "toggle_off"]
                        rels.append("toggles:" + t)
                else:
                    if ent.get("last_parent_type"):
                        rels.append("tool_at:" + ent["last_parent_type"])
                        types.append(ent["last_parent_type"])
                recs.append(self._record(self._rid(kind, t), types, ops, rels, en, fa, ev, ent, authored, ns,
                                         lex=(t + " " + " ".join(types[1:])).strip()))
        kf = self._knob_facets()
        if kf:
            ev = self._knob_evidence() or []
            if ev:
                ords = sorted({o for m in self.knob_map for o in m.get("ords", [])})
                ent = {"support": len(ords), "first_ord": min(ords), "last_ord": max(ords)}
                recs.append(self._record(self._rid("knob", "map"), [KNOB_TYPE, BURNER_TYPE], ["toggle_on", "toggle_off"],
                                         ["controls:" + BURNER_TYPE], kf[0][1] + ".", kf[0][2] + ".", ev, ent, authored, ns,
                                         lex=f"{KNOB_TYPE} {BURNER_TYPE} knob burner"))
        recs.sort(key=lambda r: (-r["recency"], r["id"]))
        for i, r in enumerate(recs):
            r["prio"] = i
        return recs

    @staticmethod
    def _join(parts, sep):
        return sep.join(parts) + "."

    def _record(self, rid, types, ops, rels, en, fa, ev, ent, authored, ns, lex=""):
        support = int(ent.get("support", 0))
        return {"schema": SCHEMA_REC, "id": rid, "component": "profile", "kitchen": self.kitchen,
                "keys": {"types": list(dict.fromkeys(types)), "ops": list(dict.fromkeys(ops)), "rels": list(dict.fromkeys(rels))},
                "lex": {"en": lex, "fa": ""}, "text": {"en": en, "fa": fa},
                "provenance": {"source": "practice:%s/%s" % (ns, self.upto_attempt if self.upto_attempt is not None else self.k_total),
                               "evidence": list(ev), "authored_by": authored, "support": []},
                "created": int(ent.get("first_ord", 0)), "recency": int(ent.get("last_ord", 0)),
                "confidence": round(min(1.0, support / 3.0), 4), "privileged": False, "prio": 0}

    # -- selective retrieval ---------------------------------------------------------------------------------------
    def candidates(self, query: dict) -> list:
        """pkg.cand/1.  1.0: the record's own type is asked for; 0.6: it holds (contains:) an asked-for type; 0.4: it is where a
        needed tool was last seen.  Otherwise not returned: nothing qualifies -> nothing is rendered (D5)."""
        need = query.get("need") or {}
        asked = set(query.get("types") or []) | set(need.get("types") or [])
        tools_asked = set(need.get("tools") or []) | (asked & set(self.tools))
        recs = {r["id"]: r for r in self.records()}
        out = []
        for rid, r in recs.items():
            t = rid.rsplit(":", 1)[-1] if ":" in rid.rsplit("/", 1)[-1] else None
            kind = rid.rsplit("/", 1)[-1].split(":", 1)[0]
            why, match = [], 0.0
            if kind in ("fixture", "tool") and (t in asked or (kind == "tool" and t in tools_asked)):
                match, why = 1.0, ["type:" + t]
            elif kind == "knob_map" and asked & {KNOB_TYPE, BURNER_TYPE}:
                match, why = 1.0, ["type:" + sorted(asked & {KNOB_TYPE, BURNER_TYPE})[0]]
            if match < 1.0 and kind == "fixture":
                hit = sorted(asked & set(self.fixtures[t]["contents"]))
                if hit:
                    match, why = 0.6, ["rel:contains:" + x for x in hit]
            if match < 0.6 and kind == "fixture":
                at = sorted(x for x in tools_asked if x in self.tools and self.tools[x].get("last_parent_type") == t)
                if at:
                    match, why = 0.4, ["rel:tool_at:" + x for x in at]
            if match > 0:
                out.append({"schema": SCHEMA_CAND, "rec": r, "match": match, "why": why})
        out.sort(key=lambda c: (-c["match"], c["rec"]["prio"], c["rec"]["id"]))
        return out


def _content_names(fx) -> list:
    c = fx["contents"]
    return [k for k, _v in sorted(c.items(), key=lambda kv: (-kv[1]["last_ord"], -kv[1]["n"], kv[0]))][:MAX_CONTENT_TYPES]


def _all_facets(evidence: dict, kind: str, t: str) -> list:
    pre = f"{kind}/{t}/"
    return [k[len(pre):] for k in evidence if k.startswith(pre)]


def _place_type(places, pl):
    return places[pl][0] if pl in places else None


def _has(con, table) -> bool:
    return bool(con.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone())


def _knob_map(acc: _Acc):
    by_knob: dict = {}
    for (kc, bc), p in acc.knob_pairs.items():
        by_knob.setdefault(kc, []).append((bc, p))
    out = []
    for kc, lst in sorted(by_knob.items()):
        if len(lst) != 1:
            acc.knob_ambiguous += 1                                  # one knob cell tied to two burners: not identified
            continue
        bc, p = lst[0]
        out.append({"knob_cell": list(kc), "burner_cell": list(bc), "n": p["n"], "ords": sorted(p["ords"]),
                    "evidence": sorted(h for (k, t, f), v in acc.ev.items() if (k, t, f) == ("fixture", KNOB_TYPE, "toggled") for h in v)[:MAX_EVIDENCE]})
    return out or "not_learned"


# ------------------------------------------------------------------------------------------------------------ producer hook
def dump_stream(adapter, journal, **meta) -> None:
    """Append one `pkg.evs/1` row for a finished practice episode (I-pkg calls it after each episode).  `meta` may carry key,
    order, attempt, item, lang; the episode id and the validated stream come from the adapter."""
    row = {"schema": SCHEMA_EVS, "key": meta.get("key") or getattr(adapter, "episode", None), "order": meta.get("order"),
           "attempt": meta.get("attempt"), "item": meta.get("item"), "lang": meta.get("lang"),
           "episode": getattr(adapter, "episode", None), "stream": list(getattr(adapter, "stream", None) or [])}
    journal.append(row)


def load_streams(path: str) -> list:
    """Read an ev_stream.jsonl back (a torn last line is ignored)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    break
    return rows


# ------------------------------------------------------------------------------------------------------------ early check 6.6
def early_check(db=None, kitchen=None, categories=None) -> dict:
    sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
    import common as C                                               # noqa: E402
    root = os.path.dirname(HERE)
    db = db or os.path.join(HERE, "final_test", "snapshots", "o0", "k10", "ltm.db")
    if categories is None:
        cp = os.path.join(HERE, "_pkg_fixtures", "categories.sample.json")
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                categories = json.load(f)
        else:
            categories = {"fixture_types": ["Fridge", "Microwave", "Sink", "SinkBasin", "StoveBurner", "StoveKnob", "CounterTop", "Cabinet",
                                            "Drawer", "GarbageCan", "CoffeeMachine", "Toaster", "Faucet", "LightSwitch", "Shelf",
                                            "ShelvingUnit", "Window"],
                          "tool_types": ["ButterKnife", "Spatula", "Fork", "Spoon", "Ladle"]}   # A26 (K-B2b): Knife out
    m = C.load_manifest()
    ids = list(m["nontest"])
    C.guard_items(ids, "practice-v1")
    with open(os.path.join(root, "sim", "dataset", "general.json"), encoding="utf-8") as f:
        items = {i["id"]: i for i in json.load(f)["items"]}
    fixt, tl = _types(categories, "fixture_types"), _types(categories, "tool_types")
    implied = {"slice": "ButterKnife", "heat": "Microwave", "cook": "Microwave", "wash": "Sink"}   # A26 item 2: was Knife
    needed: dict = {}
    n_items = 0
    for i in ids:
        it = items.get(i)
        if not it:
            continue
        n_items += 1
        for st in it.get("plan") or []:
            for t in (st.get("object"), st.get("to"), implied.get(st.get("action"))):
                if t in fixt or t in tl:
                    needed.setdefault(t, set()).add(i)
    if kitchen is None:                                              # the kitchen is DATA: read it from the store itself
        con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(db).replace("\\", "/"), uri=True)
        try:
            ks = [r[0] for r in con.execute("select distinct kitchen from episodes order by 1")]
        finally:
            con.close()
        if len(ks) != 1:
            raise ProfileError(f"the store holds {len(ks)} kitchens; pass --kitchen")
        kitchen = ks[0]
    p = Profile.from_store(db, kitchen, categories)
    covered = set(p.fixtures) | set(p.tools)
    recs = p.records()
    rec_types = {r["id"].split(":", 1)[1] for r in recs if ":" in r["id"]}
    a = p.assertions()
    unev = [x for x in a if not x["evidence"]]
    items_needing = set().union(*needed.values()) if needed else set()
    items_fully = [i for i in items_needing if all(i not in v or t in rec_types for t, v in needed.items())]
    per_type = {t: {"items": len(v), "supported": t in rec_types} for t, v in sorted(needed.items())}
    return {"db": db, "kitchen": kitchen, "partial": p.partial, "items_in_nontest": n_items,
            "needed_types": per_type, "types_needed": len(needed), "types_supported": sum(1 for v in per_type.values() if v["supported"]),
            "items_needing_any": len(items_needing),
            "items_with_all_needed_supported": len(items_fully),
            "records": len(recs), "rendered_assertions": len(a), "assertions_without_evidence": len(unev),
            "unsupported_types": sorted(t for t, v in per_type.items() if not v["supported"]),
            "profile_sha256": p.to_json()["sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early-check", action="store_true")
    ap.add_argument("--db")
    ap.add_argument("--kitchen", default=None)
    a = ap.parse_args(argv)
    if a.early_check:
        r = early_check(a.db, a.kitchen)
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0 if r["assertions_without_evidence"] == 0 else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
